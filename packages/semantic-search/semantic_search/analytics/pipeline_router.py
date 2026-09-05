"""Analytics path orchestrator.

Wires the analytics components in front of the existing 6-stage NL-to-SQL
pipeline::

  question
     ↓
  exact NL-to-SQL cache lookup              ── verified hit ──► fast path
     ↓ (miss OR unverified hit)                                      │
  existing NLToSQLPipeline.run()  ─── verified result ─── cache.upsert
     ↓                                                              │
  AnalyticsResult                                                    ▼
                                                  AstSecurityValidator → MVRouter
                                                  → ClickHouseExecutor → Verifier
                                                  (the verifier may be skipped
                                                   when the cache template is
                                                   warm + schema unchanged.)

Fast path NEVER skips:
- AST security validation (cached SQL re-validated)
- MV router (so a stale cache referencing an invalidated MV degrades to raw)

The verifier MAY be conditionally skipped when ALL hold:
- ``SqlVerifierConfig.skip_enabled = true``
- ``cached_entry.verified_count >= SqlVerifierConfig.skip_warmth_threshold``
- ``cached_entry.schema_version == current_schema_version`` (default ``''``)

Otherwise the verifier runs normally. Skips and invocations are counted
independently so the ``verifier_skip_rate`` proxy signal is computed from
real fast-path traffic, not inferred.

Failure modes degrade gracefully:
- Cache lookup error  → log + treat as miss → existing pipeline
- ClickHouse error    → log + treat fast-path as failure → pipeline retry
- MV router exception → log + raw-table SQL → pipeline retry
- Existing pipeline exception bubbles up as `AnalyticsResult(success=False)`

The router NEVER raises — it always returns a typed `AnalyticsResult` so
callers can treat both paths uniformly.
"""
import asyncio
import threading
import time
from typing import Dict, List, Optional

from semantic_search.analytics.clickhouse_client import ClickHouseQueryError
from semantic_search.analytics.clickhouse_executor import ClickHouseExecutor
from semantic_search.analytics.multi_period_handler import MultiPeriodHandler
from semantic_search.analytics.mv_router import MVRouter
from semantic_search.analytics.nl_sql_exact_cache import NLSqlExactCache, NLSqlCacheLookup
from semantic_search.config.analytics_models import AnalyticsConfig
from semantic_search.core.logging_utils import get_logger
from semantic_search.nl_to_sql.contracts import AnalyticsResult, SqlExecutionResult, SqlValidationResult, VerifierVerdict
from semantic_search.nl_to_sql.pipeline import GenAndValidate, NLToSQLPipeline
from semantic_search.nl_to_sql.security import AstSecurityValidator
from semantic_search.nl_to_sql.verifier import Verifier

logger = get_logger(__name__)


class AnalyticsRouter:
    """Coordinates the analytics exact-cache fast-path and the ClickHouse
    cache-miss path. ClickHouse is the only wired execution substrate; there
    is no Athena/DuckDB fallback, so a ClickHouse-path failure is terminal.

    :param config: AnalyticsConfig - Analytics-path config
    :param pipeline: NLToSQLPipeline - Existing 6-stage pipeline (LLM SQL gen)
    :param cache: NLSqlExactCache - Exact NL-to-SQL cache
    :param mv_router: MVRouter - MV-aware rewrite layer
    :param security_validator: AstSecurityValidator - Re-validates cache SQL
    :param ch_executor: ClickHouseExecutor - Real-time executor (CH HTTP)
    :param verifier: Verifier - Post-execution verifier gate
    :param snapshot_port: Optional - HistoricalSnapshotAnalyticsPort for completed-sales queries; None disables the snapshot substrate
    """

    def __init__(self, config: AnalyticsConfig, pipeline: NLToSQLPipeline, cache: NLSqlExactCache, mv_router: MVRouter, security_validator: AstSecurityValidator, ch_executor: ClickHouseExecutor, verifier: Verifier, verifier_config=None, current_schema_version: str = '', engine_dispatcher=None, snapshot_port=None):
        for name, dep in (
            ('config', config), ('pipeline', pipeline), ('cache', cache),
            ('mv_router', mv_router), ('security_validator', security_validator),
            ('ch_executor', ch_executor), ('verifier', verifier),
        ):
            if dep is None:
                raise ValueError(f"AnalyticsRouter requires {name}")
        if not isinstance(config, AnalyticsConfig):
            raise ValueError("AnalyticsRouter requires a typed AnalyticsConfig")
        # Pipeline duck-typed on its `run` coroutine — keeps tests substitutable
        # without subclassing while production callers always pass NLToSQLPipeline.
        if not callable(getattr(pipeline, 'run', None)):
            raise ValueError("AnalyticsRouter requires a pipeline exposing async `run(question, sql_hint, request_id)`")
        if not isinstance(current_schema_version, str):
            raise ValueError("AnalyticsRouter.current_schema_version must be a string (defaults to '')")
        self._config = config
        self._pipeline = pipeline
        self._cache = cache
        self._mv_router = mv_router
        self._security = security_validator
        self._executor = ch_executor
        self._verifier = verifier
        # Verifier-skip eligibility config. We
        # accept both the typed `SqlVerifierConfig` and `None` (legacy: skip
        # disabled). Duck-typed on the two fields the router reads to keep
        # tests substitutable without importing the typed config class here.
        self._verifier_config = verifier_config
        self._current_schema_version = current_schema_version
        self._engine_dispatcher = engine_dispatcher  # Optional[AnalyticsEngineDispatcher]; duck-typed
        # Counters for `verifier_skip_rate` proxy signal. `_skips`
        # counts auto-passes (no provider call); `_invocations` counts
        # actual `_verifier.verify(...)` calls on the fast path. The pair
        # gives the evaluator a real numerator/denominator pair and
        # avoids inferring the rate from logs.
        self._verifier_lock = threading.Lock()
        self._verifier_skips_total = 0
        self._verifier_invocations_total = 0
        self._snapshot_port = snapshot_port
        _caps = getattr(config, 'capabilities', None)
        _snap_cfg = getattr(_caps, 'historical_snapshots', None) if _caps is not None else None
        self._sale_signal_terms: List[str] = list(getattr(_snap_cfg, 'sale_signal_terms', []) or [])
        # Multi-period handler: enabled when config block is present and toggled on.
        _mp_cfg = getattr(config, 'multi_period', None)
        _mp_enabled = _mp_cfg is not None and getattr(_mp_cfg, 'enabled', False)
        self._multi_period: Optional[MultiPeriodHandler] = (
            MultiPeriodHandler(ch_executor, config.mv_router, _mp_cfg)
            if _mp_enabled else None
        )

    @property
    def enabled(self) -> bool:
        """True iff analytics path is enabled in config AND CH executor is up."""
        return bool(self._config.enabled) and self._executor.credentials_available

    @property
    def clickhouse_executor(self) -> ClickHouseExecutor:
        """Shared ClickHouse executor (guidance snapshot + explore rails reuse the same handle)."""
        return self._executor

    @property
    def cache_size(self) -> int:
        """Current entry count in the exact cache (audit/observability)."""
        return self._cache.size

    @property
    def current_schema_version(self) -> str:
        """The schema version currently considered authoritative.

        Set at construction time; bumped externally on schema migrations
        via :meth:`set_current_schema_version` so any cached SQL templates
        that reference an older version are forced through the verifier
        until they earn re-confirmation.
        """
        return self._current_schema_version

    def clear_nl_sql_cache(self) -> int:
        """Drop all NL-to-SQL exact cache entries (in-memory + Redis mirror).
        :return: int - In-memory entries dropped (0 when cache disabled or empty)
        """
        return self._cache.invalidate_all()

    def set_current_schema_version(self, version: str) -> None:
        """Bump the authoritative schema version (called on schema migration).

        Invalidates verifier-skip eligibility for every cached SQL whose
        ``schema_version`` does not match. The cache entries are NOT purged
        (they may still be served correctly); they simply lose their
        skip-fast-path privilege until the verifier re-confirms.
        """
        if not isinstance(version, str):
            raise ValueError("AnalyticsRouter.current_schema_version must be a string")
        self._current_schema_version = version

    def verifier_skip_stats(self) -> Dict[str, int]:
        """Return per-fast-path verifier skip / invocation counters.

        Backs the verifier-skip-rate KPI (target ≥ 0.85 once warm). ``skips``
        increments only when the fast path returns an auto-pass verdict
        (no provider call). ``invocations`` increments only when the fast
        path actually calls ``Verifier.verify(...)``. The legacy pipeline
        path also runs the verifier but is NOT counted here — this counter
        is scoped to the *fast-path* verifier-skip rate.

        :return: Dict[str, int] - {'skips': int, 'invocations': int, 'total': int}
        """
        with self._verifier_lock:
            return {
                'skips': int(self._verifier_skips_total),
                'invocations': int(self._verifier_invocations_total),
                'total': int(self._verifier_skips_total + self._verifier_invocations_total),
            }

    def _verifier_skip_enabled(self) -> bool:
        """True iff conditional verifier-skip is configured ON."""
        cfg = self._verifier_config
        if cfg is None:
            return False
        return bool(getattr(cfg, 'skip_enabled', False))

    def _verifier_skip_threshold(self) -> int:
        """Configured warmth threshold (verified_count gate); 0 = always skip if enabled."""
        cfg = self._verifier_config
        if cfg is None:
            return 0
        return int(getattr(cfg, 'skip_warmth_threshold', 0) or 0)

    def _verifier_mandatory_classes(self) -> frozenset:
        """Cost classes that ALWAYS run the verifier regardless of cache warmth."""
        cfg = self._verifier_config
        if cfg is None:
            return frozenset()
        raw = getattr(cfg, 'mandatory_for_classes', None) or []
        try:
            return frozenset(str(x) for x in raw)
        except TypeError:
            return frozenset()

    def _should_skip_verifier(self, entry, cost_class: Optional[str] = None) -> bool:
        """Eligibility predicate for conditional verifier-skip.

        All of:
        - ``verified_count >= skip_warmth_threshold`` (template has earned trust).
        - ``entry.schema_version == current_schema_version`` (no schema drift
          since the last verifier pass; equality is intentionally strict — an
          empty stored version equals the empty default but mismatches any
          non-empty current version, forcing the verifier to re-confirm).
        - ``cost_class`` is NOT in
          ``SqlVerifierConfig.mandatory_for_classes`` — expensive plans bypass
          the warmth-based skip even when the template is hot, because the
          blast radius of a bad answer scales with cost.
        """
        if not self._verifier_skip_enabled():
            return False
        if cost_class is not None and cost_class in self._verifier_mandatory_classes():
            return False
        if int(getattr(entry, 'verified_count', 0)) < self._verifier_skip_threshold():
            return False
        return str(getattr(entry, 'schema_version', '') or '') == str(self._current_schema_version)

    @staticmethod
    def _is_historical_sales_query(question: str, terms: List[str]) -> bool:
        """Return True when question contains a historical-sales signal term (from config)."""
        if not terms:
            return False
        q = question.lower()
        return any(t in q for t in terms)

    async def run(self, question: str, sql_hint: str = "", request_id: Optional[str] = None) -> AnalyticsResult:
        """Run the analytics request — fast path on cache hit, pipeline otherwise.

        Returns a typed `AnalyticsResult` regardless of which path served the
        query (so the caller treats both uniformly).
        """
        rid = request_id or AnalyticsResult.new_request_id()
        question = question or ""
        sql_hint = sql_hint or ""
        t0 = time.monotonic()

        if not question.strip():
            return AnalyticsResult(
                request_id=rid, question="_", sql_hint=sql_hint, success=False,
                failure_mode='generation', failure_reason='empty question',
                pruned_schema=None, generation=None, validation=None,
                execution=None, verifier=None,
                total_latency_ms=(time.monotonic() - t0) * 1000.0,
                analytics_substrate='',
            )

        if not self.enabled:
            logger.info(f"analytics_router_disabled request_id={rid} reason=config_or_executor_down")
            return await self._pipeline.run(question=question, sql_hint=sql_hint, request_id=rid)

        # Historical-sales detection: past-tense queries (sold/completed/ended) target
        # analytics.domain_snapshots, not active-auction data. Route to snapshot substrate
        # when available; return beyond_data_window boundary when not.
        if self._is_historical_sales_query(question, self._sale_signal_terms):
            logger.info(f"analytics_historical_sales_query_detected request_id={rid}")
            if self._snapshot_port is not None:
                try:
                    snap_res = await self._snapshot_port.try_answer(question=question, sql_hint=sql_hint, request_id=rid)
                except Exception as e:
                    logger.warning(f"analytics_snapshot_port_error request_id={rid} error_type={type(e).__name__} error={e}")
                    snap_res = None
                if snap_res is not None:
                    return snap_res
                # Snapshot port configured but returned no result — fall through to NL-to-SQL pipeline.
                # The pipeline may serve the query if the schema catalog includes the relevant table.
                logger.info(f"analytics_snapshot_port_miss_fallthrough request_id={rid}")
            else:
                total_ms = (time.monotonic() - t0) * 1000.0
                logger.warning(f"analytics_historical_sales_boundary request_id={rid}")
                return AnalyticsResult(
                    request_id=rid, question=question, sql_hint=sql_hint,
                    success=False, failure_mode='beyond_data_window',
                    failure_reason='query asks for completed-sales history; only active-auction data is available',
                    pruned_schema=None, generation=None, validation=None,
                    execution=None, verifier=None,
                    total_latency_ms=total_ms, analytics_substrate='',
                )

        # Start engine dispatch as a background task so the cache lookup and (on
        # cache miss) the LLM pipeline can run concurrently. Returns None
        # immediately when there is no keyword match, so non-dispatched queries
        # incur only task-creation overhead.
        _engine_task: Optional[asyncio.Task] = None
        if self._engine_dispatcher is not None:
            _engine_task = asyncio.create_task(
                self._engine_dispatcher.try_dispatch(
                    question=question, sql_hint=sql_hint, request_id=rid, t0=t0,
                )
            )

        # Fast path — cache lookup
        try:
            lookup = await asyncio.to_thread(self._cache.lookup, question)
        except Exception as e:
            logger.warning(f"analytics_cache_lookup_error request_id={rid} error_type={type(e).__name__} error={e}")
            lookup = NLSqlCacheLookup(hit=False, verified=False, entry=None)

        if lookup.hit and lookup.verified and lookup.entry is not None:
            if _engine_task is not None and not _engine_task.done():
                _engine_task.cancel()
            fast_result = await self._try_fast_path(rid, question, sql_hint, lookup, t0)
            if fast_result is not None:
                return fast_result
            # Fast path failed — fall through to the miss path.
            logger.info(f"analytics_fast_path_failed_falling_back request_id={rid}")

        # Multi-period shortcut: fires parallel ClickHouse MV queries for dimensional
        # aggregate questions detected from sql_hint. Runs before the NL-to-SQL pipeline
        # — no LLM call, no schema discovery. Skipped when sql_hint carries no recognized
        # dimension signal (price-range hints or unknown grain columns).
        if self._multi_period is not None and self._ch_on_miss_eligible():
            if not self._multi_period.can_handle(sql_hint):
                logger.debug(f"multi_period_skipped reason=can_handle_false request_id={rid}")
            else:
                mp_result = await self._multi_period.handle(question=question, sql_hint=sql_hint, request_id=rid)
                if mp_result is not None:
                    if _engine_task is not None and not _engine_task.done():
                        _engine_task.cancel()
                    return mp_result
                # All periods returned empty — fall through to NL-to-SQL pipeline.

        # Cache miss — ClickHouse is the only wired execution substrate; there
        # is no Athena/DuckDB fallback in this router, so a ClickHouse failure
        # here is terminal for the request.
        if not self._ch_on_miss_eligible():
            if _engine_task is not None and not _engine_task.done():
                _engine_task.cancel()
            total_ms = (time.monotonic() - t0) * 1000.0
            logger.warning(f"analytics_miss_no_clickhouse request_id={rid}")
            return AnalyticsResult(
                request_id=rid, question=question, sql_hint=sql_hint,
                success=False, failure_mode='no_substrate_available',
                failure_reason='ClickHouse not reachable',
                pruned_schema=None, generation=None, validation=None,
                execution=None, verifier=None,
                total_latency_ms=total_ms, analytics_substrate='',
            )

        # Miss path: race engine_task (if still running) against the LLM pipeline.
        # When engine completes first with a non-None result the LLM task is
        # cancelled. When LLM completes first the engine task is cancelled.
        # If engine already completed (fast CH or immediate no-match), use its
        # result or skip directly to the LLM path without spawning a second task.
        if _engine_task is not None and _engine_task.done():
            try:
                engine_result = _engine_task.result()
            except Exception:
                engine_result = None
            if engine_result is not None:
                return engine_result
            # Engine returned None — fall through to LLM miss path.
            ch_result = await self._try_miss_path_clickhouse(rid, question, sql_hint, t0)
            if ch_result is not None:
                return ch_result
        elif _engine_task is not None:
            # Engine still running — race it against the LLM miss path.
            _lm_task = asyncio.create_task(self._try_miss_path_clickhouse(rid, question, sql_hint, t0))
            try:
                done, _ = await asyncio.wait({_engine_task, _lm_task}, return_when=asyncio.FIRST_COMPLETED)
            except Exception as e:
                logger.warning(f"analytics_concurrent_race_error request_id={rid} error_type={type(e).__name__} error={e}")
                for t in (_engine_task, _lm_task):
                    if not t.done():
                        t.cancel()
                done = set()

            if _engine_task in done:
                try:
                    engine_result = _engine_task.result()
                except Exception:
                    engine_result = None
                if engine_result is not None:
                    _lm_task.cancel()
                    return engine_result
                # Engine returned None (no match or timed out) — LLM was already
                # running; await it instead of starting a fresh call.
                try:
                    lm_result = await _lm_task
                except Exception:
                    lm_result = None
                if lm_result is not None:
                    return lm_result
            else:
                # LLM finished first — cancel engine, return LLM result.
                _engine_task.cancel()
                try:
                    lm_result = _lm_task.result()
                except Exception:
                    lm_result = None
                if lm_result is not None:
                    return lm_result
        else:
            # No engine dispatcher — run LLM miss path directly.
            ch_result = await self._try_miss_path_clickhouse(rid, question, sql_hint, t0)
            if ch_result is not None:
                return ch_result

        total_ms = (time.monotonic() - t0) * 1000.0
        logger.warning(f"analytics_clickhouse_substrate_failed request_id={rid}")
        return AnalyticsResult(
            request_id=rid, question=question, sql_hint=sql_hint,
            success=False, failure_mode='no_substrate_available',
            failure_reason='ClickHouse execution failed and no fallback substrate is configured',
            pruned_schema=None, generation=None, validation=None,
            execution=None, verifier=None,
            total_latency_ms=total_ms, analytics_substrate='',
        )

    async def _try_fast_path(self, rid: str, question: str, sql_hint: str, lookup: NLSqlCacheLookup, t0: float) -> Optional[AnalyticsResult]:
        """Run validate → MV-route → execute → verify on a cached SQL.

        Returns None when any stage fails so the caller can fall back to the
        existing pipeline. Returns a typed `AnalyticsResult` on full success
        (analogous to the verifier-pass branch of `NLToSQLPipeline.run`).
        """
        assert lookup.entry is not None
        cached_sql = lookup.entry.sql_template

        # 1) Re-validate cached SQL (defence in depth).
        try:
            sec_result = self._security.validate(cached_sql)
        except Exception as e:
            logger.warning(f"analytics_fast_path_security_exception request_id={rid} error_type={type(e).__name__} error={e}")
            return None
        if not sec_result.is_valid:
            logger.info(f"analytics_fast_path_security_rejected request_id={rid} mode={sec_result.failure_mode}")
            return None

        # 2) MV route (best-effort — pass-through on no match)
        sql_to_run = sec_result.sql
        mv_used = ''
        if self._mv_router.enabled:
            try:
                decision = self._mv_router.route(sec_result.sql)
                if decision.matched:
                    sql_to_run = decision.rewritten_sql
                    mv_used = decision.mv_name
                    # Re-validate the rewritten SQL — security gate must hold.
                    rewritten_sec = self._security.validate(sql_to_run)
                    if not rewritten_sec.is_valid:
                        logger.info(f"analytics_fast_path_rewrite_security_rejected request_id={rid} mv={mv_used} mode={rewritten_sec.failure_mode}")
                        return None
                    sql_to_run = rewritten_sec.sql
            except Exception as e:
                logger.warning(f"analytics_fast_path_mv_router_error request_id={rid} error_type={type(e).__name__} error={e}")
                # Pass-through with the original validated SQL.
                sql_to_run = sec_result.sql
                mv_used = ''

        # 3) Execute on ClickHouse
        try:
            execution: SqlExecutionResult = await self._executor.execute(sql_to_run)
        except Exception as e:
            logger.warning(f"analytics_fast_path_execution_error request_id={rid} mv_used={mv_used or 'raw'} error_type={type(e).__name__} error={e}")
            if mv_used and isinstance(e, ClickHouseQueryError):
                self._mv_router.mark_invalid(mv_used)
            return None

        # 4) Verifier gate — conditional skip when the cached entry is warm
        # AND its schema_version matches the current authoritative version.
        # Cold cache (verified_count below threshold) and schema-drift cases
        # ALWAYS run the verifier; silent skipping on uncertainty would let
        # a poisoned or stale cache surface bad analytics.
        # Fast path has no cardinality estimate yet (cache stores neither
        # estimated_rows nor cost_class), so cost_class is None and the
        # mandatory-class gate is a no-op for cached templates. The
        # warmth + schema_version gates remain in force.
        cached_cost_class = getattr(lookup.entry, 'cost_class', None)
        if self._should_skip_verifier(lookup.entry, cost_class=cached_cost_class):
            with self._verifier_lock:
                self._verifier_skips_total += 1
            verdict = VerifierVerdict(
                sufficient=True,
                failure_mode='ok',
                confidence=1.0,
                model='skipped_warm_cache',
                latency_ms=0.0,
                notes=(
                    f"verifier auto-skipped: verified_count="
                    f"{lookup.entry.verified_count} >= warmth_threshold="
                    f"{self._verifier_skip_threshold()} AND schema_version="
                    f"{lookup.entry.schema_version!r} matches current"
                ),
            )
            logger.info(
                f"analytics_fast_path_verifier_skipped request_id={rid} "
                f"verified_count={lookup.entry.verified_count} "
                f"warmth_threshold={self._verifier_skip_threshold()} "
                f"schema_version={lookup.entry.schema_version!r}"
            )
        else:
            try:
                verdict = await self._verifier.verify(question, execution, request_id=rid)
            except Exception as e:
                logger.warning(f"analytics_fast_path_verifier_error request_id={rid} error_type={type(e).__name__} error={e}")
                return None
            with self._verifier_lock:
                self._verifier_invocations_total += 1

        total_ms = (time.monotonic() - t0) * 1000.0
        validation = SqlValidationResult(
            is_valid=True, sql=sql_to_run, failure_mode=None,
            failure_reasons=[], mutations=list(sec_result.mutations),
            estimated_rows=None, latency_ms=sec_result.latency_ms,
        )

        if not verdict.sufficient:
            logger.info(f"analytics_fast_path_verifier_rejected request_id={rid} mode={verdict.failure_mode} total_latency_ms={total_ms:.1f}")
            # Decrement trust by NOT recording a verifier-pass; the cached
            # template will need a real pipeline run to re-earn its trust.
            return AnalyticsResult(
                request_id=rid, question=question, sql_hint=sql_hint, success=False,
                failure_mode='verifier',
                failure_reason=f"fast-path verifier mode={verdict.failure_mode}: {verdict.notes}",
                pruned_schema=None, generation=None, validation=validation,
                execution=execution, verifier=verdict, total_latency_ms=total_ms,
                analytics_substrate='exact_cache',
                as_of_hot=max(0.0, time.time() - self._freshness_lag_for(mv_used)),
                as_of_analytics=float(lookup.entry.last_used),
            )

        # Verified hit confirmed by verifier — bump verified_count
        try:
            await asyncio.to_thread(self._cache.upsert, question=question, sql_template=cached_sql, mv_used=mv_used, verifier_passed=True, schema_version='')
        except Exception as e:
            logger.warning(f"analytics_fast_path_cache_bump_error request_id={rid} error_type={type(e).__name__} error={e}")

        logger.info(f"analytics_fast_path_success request_id={rid} mv_used={mv_used or 'raw'} rows={execution.row_count} total_latency_ms={total_ms:.1f}")
        as_of = max(0.0, time.time() - self._freshness_lag_for(mv_used))
        return AnalyticsResult(
            request_id=rid, question=question, sql_hint=sql_hint, success=True,
            failure_mode=None, failure_reason="",
            pruned_schema=None, generation=None, validation=validation,
            execution=execution, verifier=verdict, total_latency_ms=total_ms,
            as_of=as_of,
            analytics_substrate='exact_cache',
            as_of_hot=as_of,
            as_of_analytics=float(lookup.entry.last_used),
        )

    def _ch_on_miss_eligible(self) -> bool:
        """True iff the cache-miss path may execute on ClickHouse.

        Both the ``analytics.clickhouse_on_miss_enabled`` toggle (default
        true) and the executor's credential availability must hold. When
        either is false the request fails fast with
        ``failure_mode='no_substrate_available'`` — ClickHouse is the only
        wired execution substrate, so there is no Athena/DuckDB fallback to
        delegate to.
        """
        if not bool(getattr(self._config, 'clickhouse_on_miss_enabled', True)):
            return False
        return bool(self._executor.credentials_available)

    async def _try_miss_path_clickhouse(self, rid: str, question: str, sql_hint: str, t0: float) -> Optional[AnalyticsResult]:
        """Generate + validate SQL via the pipeline, then execute on ClickHouse.

        Mirrors the staging contract of the cache-hit fast path: AST
        security re-validation → MV-aware rewrite → ClickHouse execution →
        verifier gate. Returns a typed ``AnalyticsResult`` for deterministic
        failures (validation reject, oversized cost-class, verifier reject)
        so the caller does not re-run generation. Returns ``None`` only for
        transient/infra errors (generation exception, ClickHouse execution
        exception, verifier exception); the caller then surfaces
        ``no_substrate_available`` since no fallback substrate is wired. On
        success, upserts the verified SQL into the exact cache so the next
        identical question hits the fast-path.

        :param rid: str - Correlation id
        :param question: str - Natural-language analytics question
        :param sql_hint: str - Optional structured hint
        :param t0: float - Wall-clock start (``time.monotonic()``)
        :return: Optional[AnalyticsResult] - Typed result on the CH path;
            ``None`` on a transient infra error (no fallback substrate exists)
        """
        try:
            # Qualify generated SQL with the ClickHouse database (e.g. analytics)
            # rather than the Athena database from nl_to_sql.database — a raw
            # (non-MV-rewritten) query must target a schema that exists in CH.
            gv: GenAndValidate = await self._pipeline.generate_and_validate(
                question=question, sql_hint=sql_hint, request_id=rid,
                database_override=self._executor.database,
            )
        except Exception as e:
            logger.warning(f"analytics_miss_generate_validate_error request_id={rid} error_type={type(e).__name__} error={e}")
            return None

        if not gv.success or gv.validation is None or not gv.validation.is_valid:
            # Validation failures are deterministic for this question — retrying
            # would hit the same failure. Return a typed AnalyticsResult
            # instead of returning None to avoid re-running gen.
            total_ms = (time.monotonic() - t0) * 1000.0
            return AnalyticsResult(
                request_id=rid,
                question=question,
                sql_hint=sql_hint,
                success=False,
                failure_mode=gv.failure_mode or 'unknown',
                failure_reason=gv.failure_reason or 'generate_and_validate failed',
                pruned_schema=gv.pruned,
                generation=gv.generation,
                validation=gv.validation,
                execution=None,
                verifier=None,
                total_latency_ms=total_ms,
                analytics_substrate='',
            )

        # Gap 7: schema-drift detection. If the catalog version snapshotted
        # at prompt-build time no longer matches the executor's authoritative
        # version, we proceed but mark drift on the AnalyticsResult and emit
        # a structured log so the operator can see how often this happens.
        # The verifier is already mandatory on the miss path so drift does
        # not need to flip an additional flag here.
        schema_drift_detected = False
        gen_version = str(getattr(gv, 'schema_version_at_gen', '') or '')
        if gen_version and self._current_schema_version and gen_version != self._current_schema_version:
            schema_drift_detected = True
            logger.warning(f"analytics_schema_drift_detected request_id={rid} gen_version={gen_version[:12]} current_version={self._current_schema_version[:12]}")

        # Gap 10: cost-class routing. ``oversized`` plans are hard-rejected
        # before any executor sees them so we cannot exhaust the warehouse on
        # a runaway estimate; ``expensive`` plans are still run but ALWAYS
        # gated by the verifier (gap 6).
        cost_class = getattr(gv.validation, 'cost_class', None)
        if cost_class == 'oversized':
            total_ms = (time.monotonic() - t0) * 1000.0
            logger.warning(f"analytics_miss_oversized_rejected request_id={rid} estimated_rows={gv.validation.estimated_rows}")
            return AnalyticsResult(
                request_id=rid, question=question, sql_hint=sql_hint, success=False,
                failure_mode='security',
                failure_reason='cost_class=oversized: estimated cost exceeds expensive ceiling',
                pruned_schema=gv.pruned, generation=gv.generation, validation=gv.validation,
                execution=None, verifier=None, total_latency_ms=total_ms,
                analytics_substrate='',
                schema_version_at_gen=gen_version,
                schema_drift_detected=schema_drift_detected,
            )

        validated_sql = gv.validation.sql
        sql_to_run = validated_sql
        mv_used = ''
        if self._mv_router.enabled:
            try:
                decision = self._mv_router.route(validated_sql)
                if decision.matched:
                    rewritten_sec = self._security.validate(decision.rewritten_sql)
                    if rewritten_sec.is_valid:
                        sql_to_run = rewritten_sec.sql
                        mv_used = decision.mv_name
                    else:
                        logger.info(f"analytics_miss_rewrite_security_rejected request_id={rid} mv={decision.mv_name} mode={rewritten_sec.failure_mode}")
            except Exception as e:
                logger.warning(f"analytics_miss_mv_router_error request_id={rid} error_type={type(e).__name__} error={e}")
                sql_to_run = validated_sql
                mv_used = ''

        try:
            execution: SqlExecutionResult = await self._executor.execute(sql_to_run)
        except Exception as e:
            total_ms = (time.monotonic() - t0) * 1000.0
            err_type = type(e).__name__
            logger.warning(f"analytics_miss_clickhouse_execution_error request_id={rid} mv_used={mv_used or 'raw'} error_type={err_type} error={e}")
            if mv_used and isinstance(e, ClickHouseQueryError):
                self._mv_router.mark_invalid(mv_used)
            return AnalyticsResult(request_id=rid, question=question, sql_hint=sql_hint, success=False, failure_mode='no_substrate_available', failure_reason=f"{err_type}: {e}", pruned_schema=gv.pruned, generation=gv.generation, validation=gv.validation, execution=None, verifier=None, total_latency_ms=total_ms, analytics_substrate='clickhouse_hot', schema_version_at_gen=gen_version, schema_drift_detected=schema_drift_detected)

        try:
            verdict = await self._verifier.verify(question, execution, request_id=rid)
        except Exception as e:
            logger.warning(f"analytics_miss_verifier_error request_id={rid} error_type={type(e).__name__} error={e}")
            return None
        with self._verifier_lock:
            self._verifier_invocations_total += 1

        total_ms = (time.monotonic() - t0) * 1000.0
        if not verdict.sufficient:
            logger.info(f"analytics_miss_verifier_rejected request_id={rid} mv_used={mv_used or 'raw'} mode={verdict.failure_mode} total_latency_ms={total_ms:.1f}")
            return AnalyticsResult(
                request_id=rid, question=question, sql_hint=sql_hint, success=False,
                failure_mode='verifier',
                failure_reason=f"miss-path verifier mode={verdict.failure_mode}: {verdict.notes}",
                pruned_schema=gv.pruned, generation=gv.generation, validation=gv.validation,
                execution=execution, verifier=verdict, total_latency_ms=total_ms,
                analytics_substrate='clickhouse_hot',
                as_of_hot=max(0.0, time.time() - self._freshness_lag_for(mv_used)),
                as_of_analytics=None,
                schema_version_at_gen=gen_version,
                schema_drift_detected=schema_drift_detected,
            )

        # Cache the verified SQL so the next equivalent question hits the
        # fast path instead of paying the gen+validate cost again.
        try:
            await asyncio.to_thread(self._cache.upsert, question=question, sql_template=validated_sql, mv_used=mv_used, verifier_passed=True, schema_version=self._current_schema_version)
        except Exception as e:
            logger.warning(f"analytics_miss_cache_upsert_error request_id={rid} error_type={type(e).__name__} error={e}")

        logger.info(f"analytics_miss_clickhouse_success request_id={rid} mv_used={mv_used or 'raw'} rows={execution.row_count} total_latency_ms={total_ms:.1f}")
        as_of = max(0.0, time.time() - self._freshness_lag_for(mv_used))
        return AnalyticsResult(
            request_id=rid, question=question, sql_hint=sql_hint, success=True,
            failure_mode=None, failure_reason="",
            pruned_schema=gv.pruned, generation=gv.generation, validation=gv.validation,
            execution=execution, verifier=verdict, total_latency_ms=total_ms,
            as_of=as_of,
            analytics_substrate='clickhouse_hot',
            as_of_hot=as_of,
            as_of_analytics=None,
            schema_version_at_gen=gen_version,
            schema_drift_detected=schema_drift_detected,
        )

    def _freshness_lag_for(self, mv_used: str) -> float:
        """Return the per-path freshness lag (seconds) for ``as_of`` derivation.

        :param mv_used: str - MV name (empty string when SQL ran on raw table)
        :return: float - lag in seconds; falls back to ``freshness_target_seconds``
            when ``mv_used`` is unknown so the anchor stays defined even if the
            catalog and runtime drift apart.
        """
        if mv_used:
            for mv in self._config.mv_router.materialized_views:
                if mv.name == mv_used:
                    return float(mv.freshness_lag_seconds)
        return float(self._config.freshness_target_seconds)


__all__ = ['AnalyticsRouter']
