"""End-to-end NL-to-SQL Analytics pipeline.

Wiring contract (one direction, no skip-back):

  schema discovery
       ↓
  LLM SQL generation  ─────────────► retry-loop with validator failure context
       ↓                              (capped by generation.max_retries)
  AST security validator (sqlglot, Trino)
       ↓
  parallel logic validator (asyncio.gather)
       ↓
  Athena execution (timeout + row cap)
       ↓
  post-execution verifier gate

Every stage emits structured logs with the request_id so downstream
observability (cost / quality dashboards) can stitch a single analytics
request end-to-end. The pipeline NEVER raises on stage failure — it returns a
typed `AnalyticsResult` with `success=False` and a `failure_mode` so the
orchestrator can route through the Zero-Result Guard.
"""
import asyncio
import time
from dataclasses import dataclass
from typing import Optional

import sqlglot as _sqlglot
import sqlglot.expressions as _sqlglot_exp

from semantic_search.config.nl_to_sql_models import NLToSQLConfig
from semantic_search.core.exceptions import AgentSearchError, LLMRefusalError
from semantic_search.core.logging_utils import get_logger
from semantic_search.nl_to_sql.contracts import AnalyticsResult, PrunedSchema, SqlExecutionResult, SqlGenerationResult, SqlValidationResult, VerifierVerdict
from semantic_search.nl_to_sql.executor import SqlExecutor
from semantic_search.nl_to_sql.generator import SqlGenerator
from semantic_search.nl_to_sql.logic_validator import LogicValidator
from semantic_search.nl_to_sql.schema import SchemaDiscoverer
from semantic_search.nl_to_sql.security import AstSecurityValidator
from semantic_search.nl_to_sql.verifier import Verifier

# Forward-only import contract: ``MVRouter`` lives in the analytics
# subsystem, which depends on this pipeline (not the other way around).
# The pipeline accepts MVRouter via duck typing through a string-annotated
# Optional so the layer direction declared in workspace-graph.mdc is
# preserved — at runtime the ``mv_router`` parameter is any object
# exposing ``enabled: bool`` and ``route(sql) -> MVRewriteDecision``.

logger = get_logger(__name__)


def _extract_column_refs(sql: str) -> frozenset:
    """Return lowercase column names referenced in sql (excluding aliases and function names)."""
    try:
        tree = _sqlglot.parse_one(sql)
        return frozenset(
            node.name.lower()
            for node in tree.find_all(_sqlglot_exp.Column)
            if node.name
        )
    except Exception:
        return frozenset()


@dataclass
class GenAndValidate:
    """Output of :meth:`NLToSQLPipeline.generate_and_validate`.

    Carries the discovered schema, the most recent generation result, and the
    consolidated validation result. ``success=True`` only when every retry
    loop terminated with a security-valid AND logic-valid SQL. ``failure_*``
    fields mirror the ``AnalyticsResult`` failure surface so callers can lift
    a failed generate-and-validate pass straight into a typed
    ``AnalyticsResult`` via :meth:`NLToSQLPipeline._fail`.

    :param success: bool - True iff a valid SQL was produced
    :param pruned: Optional[PrunedSchema] - Schema discovery output (None on
        early failure)
    :param generation: Optional[SqlGenerationResult] - Last generation result
    :param validation: Optional[SqlValidationResult] - Consolidated validation
        result (security + logic merged)
    :param failure_mode: Optional[str] - Failure mode (None on success)
    :param failure_reason: str - Human-readable failure reason
    """
    success: bool
    pruned: Optional[PrunedSchema]
    generation: Optional[SqlGenerationResult]
    validation: Optional[SqlValidationResult]
    failure_mode: Optional[str] = None
    failure_reason: str = ""
    # SchemaCatalog.version captured at prompt-build time. Surfaced so the
    # analytics router can detect schema drift between gen and execute on
    # warm-cache fast paths.
    schema_version_at_gen: str = ""


class NLToSQLPipeline:
    """Orchestrates the six-stage NL-to-SQL pipeline.

    Construction is dependency-injected — the pipeline does NOT build its own
    components so tests can swap in fakes for the schema catalog, LLM router,
    or Athena client without touching the production wiring.

    :param config: NLToSQLConfig - Top-level pipeline config
    :param schema_discoverer: SchemaDiscoverer - Stage 1
    :param sql_generator: SqlGenerator - Stage 2
    :param security_validator: AstSecurityValidator - Stage 3
    :param logic_validator: LogicValidator - Stage 4
    :param sql_executor: SqlExecutor - Stage 6 (execution; canonical
        6-stage pipeline numbers MV-aware rewrite as Stage 5)
    :param verifier: Verifier - Post-execution verifier gate
    :param mv_router: Optional[MVRouter] - Stage 5: MV-aware rewrite. When
        ``None`` (Athena-only deployments) or ``enabled=False``, the stage
        is a no-op. When supplied AND enabled, the pipeline runs the
        rewrite, re-validates the rewritten SQL through the same
        ``security_validator``, and only swaps in the rewritten SQL when
        re-validation passes; a security-rejected rewrite falls back to
        the original validated SQL with a structured log.
    """

    def __init__(
        self,
        config: NLToSQLConfig,
        schema_discoverer: SchemaDiscoverer,
        sql_generator: SqlGenerator,
        security_validator: AstSecurityValidator,
        logic_validator: LogicValidator,
        sql_executor: SqlExecutor,
        verifier: Verifier,
        mv_router: Optional['MVRouter'] = None,
    ):
        if not isinstance(config, NLToSQLConfig):
            raise AgentSearchError("NLToSQLPipeline requires a NLToSQLConfig")
        for name, dep in (
            ('schema_discoverer', schema_discoverer),
            ('sql_generator', sql_generator),
            ('security_validator', security_validator),
            ('logic_validator', logic_validator),
            ('sql_executor', sql_executor),
            ('verifier', verifier),
        ):
            if dep is None:
                raise AgentSearchError(f"NLToSQLPipeline requires {name}")
        self._config = config
        self._discover = schema_discoverer
        self._generate = sql_generator
        self._security = security_validator
        self._logic = logic_validator
        self._execute = sql_executor
        self._verifier = verifier
        self._mv_router = mv_router

    @property
    def enabled(self) -> bool:
        """True iff the pipeline is enabled in config and the executor has credentials."""
        return self._config.enabled and self._execute.credentials_available

    @property
    def schema_catalog(self):
        """Read-only handle on the schema catalog (used by analytics router to
        wire the catalog into its own AstSecurityValidator instance)."""
        return self._discover.catalog

    async def generate_and_validate(self, question: str, sql_hint: str, request_id: str, database_override: Optional[str] = None) -> GenAndValidate:
        """Run schema discovery + LLM SQL generation + AST + logic validation.

        Stops short of execution and verification so callers can route the
        validated SQL to an alternate executor (e.g. ClickHouse on the
        analytics fast/miss path) before deciding whether to continue with
        Athena execution. Honours the same retry budget
        (``generation.max_retries``) as :meth:`run`.

        :param question: str - Natural-language analytics question
        :param sql_hint: str - Optional structured hint
        :param request_id: str - Caller-supplied correlation id (required)
        :param database_override: Optional[str] - When set, qualifies the
            generated SQL with this database instead of ``nl_to_sql.database``.
            Used by the ClickHouse miss path to emit ``analytics.<table>`` rather
            than the Athena database (which does not exist in ClickHouse).
        :return: GenAndValidate - Stages produced; ``success=False`` when any
            mandatory stage failed
        """
        # Snapshot the catalog version at prompt-build time so the analytics
        # router can spot schema drift between gen and execute on warm cache
        # paths. Best-effort: a catalog without a consistent hash returns ''.
        schema_version_at_gen = ''
        try:
            schema_version_at_gen = str(self._discover.catalog.version or '')
        except Exception:
            schema_version_at_gen = ''

        try:
            pruned = self._discover.discover(
                table=self._config.table,
                database=database_override or self._config.database,
                question=question,
                sql_hint=sql_hint,
            )
        except Exception as e:
            logger.warning(
                f"pipeline_schema_discovery_failed request_id={request_id} "
                f"error={type(e).__name__}: {e}"
            )
            return GenAndValidate(
                success=False, pruned=None, generation=None, validation=None,
                failure_mode='schema_unavailable', failure_reason=str(e),
                schema_version_at_gen=schema_version_at_gen,
            )

        if not pruned.columns:
            return GenAndValidate(
                success=False, pruned=pruned, generation=None, validation=None,
                failure_mode='schema_unavailable',
                failure_reason=(
                    f"schema discovery selected zero columns for table={self._config.table}"
                ),
                schema_version_at_gen=schema_version_at_gen,
            )

        generation: Optional[SqlGenerationResult] = None
        validation: Optional[SqlValidationResult] = None
        previous_sql = ""
        previous_error = ""
        max_attempts = max(1, int(self._config.generation.max_retries) + 1)

        _gen_timeout = float(self._config.generation.generation_timeout_seconds)
        for attempt in range(1, max_attempts + 1):
            try:
                _coro = self._generate.generate(
                    question=question,
                    sql_hint=sql_hint,
                    pruned=pruned,
                    previous_sql=previous_sql,
                    previous_error=previous_error,
                    attempt=attempt,
                    request_id=request_id,
                )
                generation = (
                    await asyncio.wait_for(_coro, timeout=_gen_timeout)
                    if _gen_timeout > 0
                    else await _coro
                )
            except LLMRefusalError as e:
                # Hard refusal — model committed to kind='refuse'.
                # No retry; the same model on the same schema will refuse
                # again. Surface as a distinct failure mode so dashboards
                # can attribute it separately from soft generator failures.
                logger.warning(
                    f"pipeline_generation_refused request_id={request_id} attempt={attempt} "
                    f"reason={str(e.reason)[:200]}"
                )
                return GenAndValidate(
                    success=False, pruned=pruned, generation=None, validation=None,
                    failure_mode='llm_refused',
                    failure_reason=f"LLMRefusalError: {e.reason}",
                    schema_version_at_gen=schema_version_at_gen,
                )
            except Exception as e:
                logger.warning(
                    f"pipeline_generation_failed request_id={request_id} attempt={attempt} "
                    f"error_type={type(e).__name__} error={e}"
                )
                return GenAndValidate(
                    success=False, pruned=pruned, generation=None, validation=None,
                    failure_mode='generation', failure_reason=f"{type(e).__name__}: {e}",
                    schema_version_at_gen=schema_version_at_gen,
                )

            # Check generated SQL references only columns that exist in the full catalog.
            # Catches LLM hallucinations before the network EXPLAIN round-trip.
            # Best-effort: silently skips when catalog is unavailable.
            try:
                _catalog_cols = frozenset(
                    c.name.lower()
                    for c in self.schema_catalog.columns_for(pruned.table)
                )
            except Exception:
                _catalog_cols = frozenset()
            if _catalog_cols:
                _unknown_cols = _extract_column_refs(generation.sql) - _catalog_cols
                if _unknown_cols:
                    previous_sql = generation.sql
                    previous_error = (
                        f"columns_not_in_schema: {', '.join(sorted(_unknown_cols))}. "
                        f"Only use columns from the provided schema."
                    )
                    if attempt < max_attempts:
                        logger.info(
                            f"pipeline_unknown_columns_retry request_id={request_id} "
                            f"attempt={attempt} unknown={sorted(_unknown_cols)}"
                        )
                        continue
                    return GenAndValidate(
                        success=False, pruned=pruned, generation=generation, validation=None,
                        failure_mode='invalid_columns', failure_reason=previous_error,
                        schema_version_at_gen=schema_version_at_gen,
                    )

            sec_result, logic_result = await asyncio.gather(
                asyncio.to_thread(self._security.validate, generation.sql),
                self._logic.validate(generation.sql),
            )
            if not sec_result.is_valid:
                previous_sql = generation.sql
                previous_error = "; ".join(sec_result.failure_reasons) or "security failure"
                validation = sec_result
                if attempt < max_attempts:
                    logger.info(f"pipeline_security_retry request_id={request_id} attempt={attempt} reason={previous_error!r}")
                    continue
                return GenAndValidate(
                    success=False, pruned=pruned, generation=generation, validation=sec_result,
                    failure_mode=sec_result.failure_mode or 'security', failure_reason=previous_error,
                    schema_version_at_gen=schema_version_at_gen,
                )
            if not logic_result.is_valid:
                previous_sql = sec_result.sql
                previous_error = "; ".join(logic_result.failure_reasons) or "logic failure"
                merged = self._merge_validation(sec_result, logic_result)
                validation = merged
                if attempt < max_attempts:
                    logger.info(f"pipeline_logic_retry request_id={request_id} attempt={attempt} reason={previous_error!r}")
                    continue
                return GenAndValidate(
                    success=False, pruned=pruned, generation=generation, validation=merged,
                    failure_mode=logic_result.failure_mode or 'syntax', failure_reason=previous_error,
                    schema_version_at_gen=schema_version_at_gen,
                )
            validation = self._merge_validation(sec_result, logic_result)
            return GenAndValidate(
                success=True, pruned=pruned, generation=generation, validation=validation,
                schema_version_at_gen=schema_version_at_gen,
            )

        # Should be unreachable — every retry path either continues, returns
        # success, or returns a failure. Defensive return to satisfy mypy.
        return GenAndValidate(
            success=False, pruned=pruned, generation=generation, validation=validation,
            failure_mode='unknown', failure_reason='exhausted retry loop without resolution',
            schema_version_at_gen=schema_version_at_gen,
        )

    async def run(self, question: str, sql_hint: str = "", request_id: Optional[str] = None) -> AnalyticsResult:
        """Run the full pipeline and return a typed `AnalyticsResult`.

        :param question: str - Natural-language analytics question
        :param sql_hint: str - Optional structured hint (from QI engine)
        :param request_id: Optional[str] - Caller-supplied correlation id (auto-gen if None)
        :return: AnalyticsResult - Always typed; `success=False` on any stage failure
        """
        rid = request_id or AnalyticsResult.new_request_id()
        question = question or ""
        sql_hint = sql_hint or ""
        t0 = time.monotonic()
        if not question.strip():
            return self._fail(rid, question, sql_hint, 'generation', "empty question", t0)
        if not self._config.enabled:
            return self._fail(rid, question, sql_hint, 'generation', "nl_to_sql disabled by config", t0)

        gv = await self.generate_and_validate(question=question, sql_hint=sql_hint, request_id=rid)
        if not gv.success:
            return self._fail(
                rid, question, sql_hint,
                gv.failure_mode or 'unknown',
                gv.failure_reason,
                t0,
                pruned_schema=gv.pruned,
                generation=gv.generation,
                validation=gv.validation,
                schema_version_at_gen=gv.schema_version_at_gen,
            )

        pruned = gv.pruned
        generation = gv.generation
        validation = gv.validation
        if generation is None or validation is None or not validation.is_valid:
            return self._fail(
                rid, question, sql_hint, 'unknown',
                "pipeline reached executor without a valid SQL — invariant violated", t0,
                pruned_schema=pruned, generation=generation, validation=validation,
                schema_version_at_gen=gv.schema_version_at_gen,
            )

        # Stage 5: MV-aware rewrite. Wrapped in a helper so the same
        # rewrite + security re-validation pattern is reused everywhere
        # the pipeline applies MV routing (currently `run` only; future
        # callers like a streaming variant or batch executor can call
        # `_apply_mv_rewrite` directly without copy-paste).
        sql_to_run, mv_used = self._apply_mv_rewrite(validation.sql, rid)

        try:
            execution: SqlExecutionResult = await self._execute.execute(sql_to_run)
        except Exception as e:
            logger.error(
                f"pipeline_execution_failed request_id={rid} error_type={type(e).__name__} error={e}"
            )
            return self._fail(
                rid, question, sql_hint, 'execution', f"{type(e).__name__}: {e}", t0,
                pruned_schema=pruned, generation=generation, validation=validation,
                schema_version_at_gen=gv.schema_version_at_gen,
            )

        try:
            verdict: VerifierVerdict = await self._verifier.verify(question, execution, request_id=rid)
        except Exception as e:
            logger.error(
                f"pipeline_verifier_failed request_id={rid} error_type={type(e).__name__} error={e}"
            )
            return self._fail(
                rid, question, sql_hint, 'verifier', f"{type(e).__name__}: {e}", t0,
                pruned_schema=pruned, generation=generation, validation=validation, execution=execution,
                schema_version_at_gen=gv.schema_version_at_gen,
            )

        total_ms = (time.monotonic() - t0) * 1000.0
        if not verdict.sufficient:
            logger.info(
                f"pipeline_verifier_rejected request_id={rid} mode={verdict.failure_mode} "
                f"total_latency_ms={total_ms:.1f}"
            )
            return AnalyticsResult(
                request_id=rid,
                question=question,
                sql_hint=sql_hint,
                success=False,
                failure_mode='verifier',
                failure_reason=f"verifier mode={verdict.failure_mode}: {verdict.notes}",
                pruned_schema=pruned,
                generation=generation,
                validation=validation,
                execution=execution,
                verifier=verdict,
                total_latency_ms=total_ms,
                schema_version_at_gen=gv.schema_version_at_gen,
            )

        logger.info(
            f"pipeline_success request_id={rid} rows={execution.row_count} "
            f"total_latency_ms={total_ms:.1f}"
        )
        # Athena snap tables refresh daily; the as_of anchor surfaces a
        # conservative "now − snap_lag" so the UI never misrepresents
        # freshness. ClickHouse path computes its own anchor in pipeline_router.
        as_of = max(0.0, time.time() - float(self._config.athena.freshness_lag_seconds))
        return AnalyticsResult(
            request_id=rid,
            question=question,
            sql_hint=sql_hint,
            success=True,
            failure_mode=None,
            failure_reason="",
            pruned_schema=pruned,
            generation=generation,
            validation=validation,
            execution=execution,
            verifier=verdict,
            total_latency_ms=total_ms,
            as_of=as_of,
            mv_used=mv_used,
            analytics_substrate='clickhouse_hot',
            as_of_hot=None,
            as_of_analytics=as_of,
            schema_version_at_gen=gv.schema_version_at_gen,
        )

    @staticmethod
    def _merge_validation(security: SqlValidationResult, logic: SqlValidationResult) -> SqlValidationResult:
        """Combine security + logic results into one consolidated validation record.

        On full success, carries forward security's mutated SQL (with the LIMIT
        clamp applied) and logic's cardinality estimate.
        """
        if not logic.is_valid or not security.is_valid:
            failure_mode = logic.failure_mode or security.failure_mode or 'unknown'
            reasons = list(security.failure_reasons) + list(logic.failure_reasons)
            return SqlValidationResult(
                is_valid=False,
                sql=security.sql,
                failure_mode=failure_mode,
                failure_reasons=reasons,
                mutations=list(security.mutations),
                estimated_rows=logic.estimated_rows,
                latency_ms=security.latency_ms + logic.latency_ms,
                cost_class=logic.cost_class,
            )
        return SqlValidationResult(
            is_valid=True,
            sql=security.sql,
            failure_mode=None,
            failure_reasons=[],
            mutations=list(security.mutations),
            estimated_rows=logic.estimated_rows,
            latency_ms=security.latency_ms + logic.latency_ms,
            cost_class=logic.cost_class,
        )

    def _apply_mv_rewrite(self, validated_sql: str, request_id: str) -> tuple:
        """Stage 5: MV-aware rewrite + post-rewrite security re-validation.

        :param validated_sql: str - The SQL produced by stages 1-4 (already
            security- and logic-valid)
        :param request_id: str - Correlation id for structured logs
        :return: tuple[str, str] - ``(sql_to_run, mv_used)``. ``sql_to_run``
            is the rewritten SQL when an MV matched AND the rewrite passed
            security re-validation; otherwise the original ``validated_sql``.
            ``mv_used`` is the matched MV name (empty when no rewrite
            occurred — router missing, router disabled, no match, parse
            failure, or rewrite-security-rejected).

        The router is invoked best-effort: any exception (sqlglot parse
        crash, malformed MV catalog, etc.) falls back to the original SQL
        with a WARNING log. This mirrors the long-standing pattern in
        :class:`AnalyticsRouter` so a single brittle MV catalog entry
        cannot brick the whole canonical pipeline.
        """
        if self._mv_router is None or not self._mv_router.enabled:
            return validated_sql, ''
        try:
            decision = self._mv_router.route(validated_sql)
        except Exception as e:
            logger.warning(
                f"pipeline_mv_router_error request_id={request_id} "
                f"error_type={type(e).__name__} error={e}"
            )
            return validated_sql, ''
        if not decision.matched:
            return validated_sql, ''
        try:
            rewritten_sec = self._security.validate(decision.rewritten_sql)
        except Exception as e:
            logger.warning(
                f"pipeline_mv_rewrite_security_validate_error request_id={request_id} "
                f"mv={decision.mv_name} error_type={type(e).__name__} error={e}"
            )
            return validated_sql, ''
        if not rewritten_sec.is_valid:
            logger.info(
                f"pipeline_mv_rewrite_security_rejected request_id={request_id} "
                f"mv={decision.mv_name} mode={rewritten_sec.failure_mode}"
            )
            return validated_sql, ''
        logger.info(
            f"pipeline_mv_rewrite_applied request_id={request_id} mv={decision.mv_name}"
        )
        return rewritten_sec.sql, decision.mv_name

    @staticmethod
    def _fail(
        rid: str,
        question: str,
        sql_hint: str,
        failure_mode: str,
        failure_reason: str,
        t0: float,
        pruned_schema: Optional[PrunedSchema] = None,
        generation: Optional[SqlGenerationResult] = None,
        validation: Optional[SqlValidationResult] = None,
        execution: Optional[SqlExecutionResult] = None,
        verifier: Optional[VerifierVerdict] = None,
        schema_version_at_gen: str = '',
    ) -> AnalyticsResult:
        """Build a failed `AnalyticsResult` carrying whatever stage outputs exist."""
        return AnalyticsResult(
            request_id=rid,
            question=question or "_",
            sql_hint=sql_hint,
            success=False,
            failure_mode=failure_mode,
            failure_reason=failure_reason,
            pruned_schema=pruned_schema,
            generation=generation,
            validation=validation,
            execution=execution,
            verifier=verifier,
            total_latency_ms=(time.monotonic() - t0) * 1000.0,
            schema_version_at_gen=schema_version_at_gen,
        )


__all__ = ['NLToSQLPipeline', 'GenAndValidate']
