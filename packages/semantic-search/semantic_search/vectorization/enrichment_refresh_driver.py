"""Periodic seed-time enrichment refresh driver — Qdrant-only.

Polls majestic / semrush / estibot / search_rollup Athena sources, each on
its own independent poll loop with its own cursor and failure counter
(mirrors ``EventIngestDriver``'s bid/watch loop split, generalized to N
sources), and patches Qdrant payloads for the corresponding fields.

Unlike ``DeltaRefreshDriver``, this driver has NO ClickHouse write path:
the 5 SEO columns on ``auction_audit_cln``/``domain_snapshots`` are declared
but never written or read anywhere in the codebase, and the remaining
fields (estibot_*, unique_search_count, is_gem, is_boosted_aftermarket)
have zero ClickHouse representation at all — user-confirmed Qdrant-only
scope.

Unlike ``DeltaRefreshDriver._patch_qdrant`` (auction_id-keyed via
``cfg.payload_id_field``), this driver's Qdrant filter key is the literal
string ``"domain_name"`` — a single domain backs multiple concurrent Qdrant
points (concurrent live auctions for that domain), and a filter-based
``set_payload`` patches all matching points in one call.

Layer rules: imports stdlib + core + vectorization.enrichment_source +
nl_to_sql.athena_client + retrieval.qdrant_adapter + config.models.
Never imports orchestrator or registry.
"""
import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from semantic_search.config.models import EnrichmentRefreshConfig, SeedDatabaseConfig
from semantic_search.core.exceptions import ValidationError
from semantic_search.core.logging_utils import get_logger
from semantic_search.nl_to_sql.athena_client import AthenaClient
from semantic_search.retrieval.qdrant_adapter import QdrantClientFactory
from semantic_search.vectorization.enrichment_source import (
    fetch_estibot_delta,
    fetch_majestic_delta,
    fetch_rollup_full,
    fetch_semrush_delta,
)

try:
    from qdrant_client import models as _qm
except ImportError:
    _qm = None

logger = get_logger(__name__)

_MAX_HISTORY: int = 100
_QDRANT_FILTER_KEY: str = "domain_name"

_SOURCE_MAJESTIC: str = "majestic"
_SOURCE_SEMRUSH: str = "semrush"
_SOURCE_ESTIBOT: str = "estibot"
_SOURCE_ROLLUP: str = "search_rollup"
# search_rollup has no incremental cursor — every cycle recomputes the full
# rolling lookback_days window (see enrichment_source.fetch_rollup_full).
_DELTA_SOURCES: Tuple[str, ...] = (_SOURCE_MAJESTIC, _SOURCE_SEMRUSH, _SOURCE_ESTIBOT)
_ALL_SOURCES: Tuple[str, ...] = _DELTA_SOURCES + (_SOURCE_ROLLUP,)


@dataclass
class EnrichmentCycleSummary:
    """Single per-source cycle outcome."""

    source: str
    started_at: float
    duration_seconds: float
    rows_fetched: int
    rows_patched: int
    rows_skipped: int
    success: bool
    error: Optional[str] = None


@dataclass
class EnrichmentSourceSummary:
    """Per-source state snapshot."""

    active: bool
    paused: bool
    consecutive_failures: int
    last_polled_at: float


@dataclass
class EnrichmentDriverSummary:
    """Aggregate snapshot of driver state for ops endpoints."""

    enabled: bool
    running: bool
    interval_seconds: float
    sources: Dict[str, EnrichmentSourceSummary]
    cycles_total: int
    cycles_success: int
    cycles_failed: int
    history: List[EnrichmentCycleSummary] = field(default_factory=list)


class _SourceState:
    """Mutable per-source loop state (cursor, failure counter, task, lock)."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.last_polled_at: float = 0.0
        self.consecutive_failures: int = 0
        self.paused: bool = False
        self.task: Optional[asyncio.Task] = None
        self.lock = asyncio.Lock()

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()


class EnrichmentRefreshDriver:
    """Polls seed-time enrichment sources and patches Qdrant payloads by domain_name.

    :param config: EnrichmentRefreshConfig - Validated cadence/behavior params.
    :param seed_database: SeedDatabaseConfig - Source-table pointers reused from the
        already-loaded seed config (majestic required; semrush/estibot/search_rollup
        optional — a None sub-config means that source is skipped, not an error).
    :param athena_client: AthenaClient - Pre-built Athena client for source queries.
    :param qdrant_factory: QdrantClientFactory - Holds the AsyncQdrantClient.
    :raises ValidationError: When required dependencies are missing or wrong-typed.
    """

    def __init__(
        self,
        config: EnrichmentRefreshConfig,
        seed_database: SeedDatabaseConfig,
        athena_client: AthenaClient,
        qdrant_factory: QdrantClientFactory,
    ) -> None:
        if config is None or not isinstance(config, EnrichmentRefreshConfig):
            raise ValidationError("EnrichmentRefreshDriver requires a typed EnrichmentRefreshConfig")
        if seed_database is None or not isinstance(seed_database, SeedDatabaseConfig):
            raise ValidationError("EnrichmentRefreshDriver requires a typed SeedDatabaseConfig")
        if athena_client is None or not isinstance(athena_client, AthenaClient):
            raise ValidationError("EnrichmentRefreshDriver requires an AthenaClient")
        if qdrant_factory is None or not isinstance(qdrant_factory, QdrantClientFactory):
            raise ValidationError("EnrichmentRefreshDriver requires a QdrantClientFactory")
        self._config = config
        self._athena = athena_client
        self._qdrant = qdrant_factory
        self._cycles_total: int = 0
        self._cycles_success: int = 0
        self._cycles_failed: int = 0
        self._history: List[EnrichmentCycleSummary] = []

        self._states: Dict[str, _SourceState] = {}
        _cfg_by_source = {
            _SOURCE_MAJESTIC: seed_database.majestic,
            _SOURCE_SEMRUSH: seed_database.semrush,
            _SOURCE_ESTIBOT: seed_database.estibot,
            _SOURCE_ROLLUP: seed_database.search_rollup,
        }
        for name in _ALL_SOURCES:
            cfg = _cfg_by_source[name]
            if cfg is None:
                logger.info(f"enrichment_refresh_source_skipped source={name} reason=seed_config_absent")
                continue
            self._states[name] = _SourceState(cfg)

    @property
    def enabled(self) -> bool:
        return bool(self._config.enabled)

    @property
    def running(self) -> bool:
        return any(state.running for state in self._states.values())

    async def start(self) -> None:
        """Spawn one background polling task per configured source. Idempotent; no-op when disabled."""
        if not self.enabled:
            logger.info("enrichment_refresh_driver_disabled")
            return
        now_ts = time.time()
        lookback_sec = float(self._config.lookback_minutes) * 60.0
        for name, state in self._states.items():
            if state.running:
                continue
            state.paused = False
            state.consecutive_failures = 0
            state.last_polled_at = now_ts - lookback_sec
            state.task = asyncio.create_task(self._poll_loop(name))
            logger.info(f"enrichment_refresh_driver_started source={name} interval_seconds={self._config.interval_seconds}")

    async def stop(self) -> None:
        """Cancel and await all background tasks. Idempotent."""
        for state in self._states.values():
            task = state.task
            if task is None:
                continue
            state.task = None
            if not task.done():
                task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        logger.info("enrichment_refresh_driver_stopped")

    async def _poll_loop(self, source: str) -> None:
        try:
            while True:
                state = self._states[source]
                if state.paused:
                    return
                await self._tick(source)
                await asyncio.sleep(float(self._config.interval_seconds))
        except asyncio.CancelledError:
            raise

    async def _tick(self, source: str) -> None:
        """One poll iteration for a given source — at most one cycle per source at a time."""
        state = self._states[source]
        if state.lock.locked():
            logger.warning(f"enrichment_refresh_skip_overlap source={source}")
            return
        async with state.lock:
            await self._run_cycle(source, state)

    async def _run_cycle(self, source: str, state: _SourceState) -> None:
        """Fetch + patch Qdrant for one source (lock held).

        majestic/semrush/estibot walk ``state.last_polled_at`` -> now in
        chunk_minutes slices, advancing the cursor after each chunk so a
        mid-cycle failure recovers from the last committed chunk on the next
        tick. search_rollup has no cursor — one full-window fetch per cycle.
        """
        if not self._athena.credentials_available:
            logger.warning(f"enrichment_refresh_skipped source={source} reason=credentials_unavailable last_polled_at preserved catchup_on_recovery=true")
            return
        now_ts = time.time()
        started_at = time.monotonic()
        wall_started = now_ts
        success = False
        error_str: Optional[str] = None
        rows_fetched = 0
        rows_patched = 0
        rows_skipped = 0
        try:
            if source == _SOURCE_ROLLUP:
                pairs = await fetch_rollup_full(self._athena, state.cfg, self._config.batch_size, self._config.timeout_seconds)
                rows_fetched += len(pairs)
                patched, skipped = await self._apply_pairs(pairs, source)
                rows_patched += patched
                rows_skipped += skipped
                state.last_polled_at = now_ts
            else:
                fetch_fn = {
                    _SOURCE_MAJESTIC: fetch_majestic_delta,
                    _SOURCE_SEMRUSH: fetch_semrush_delta,
                    _SOURCE_ESTIBOT: fetch_estibot_delta,
                }[source]
                since_ts = state.last_polled_at if state.last_polled_at > 0.0 else now_ts - float(self._config.lookback_minutes) * 60.0
                chunk_sec = float(self._config.chunk_minutes) * 60.0
                cursor = since_ts
                while cursor < now_ts:
                    chunk_end = min(cursor + chunk_sec, now_ts)
                    pairs = await fetch_fn(self._athena, state.cfg, cursor, chunk_end, self._config.batch_size, self._config.timeout_seconds)
                    rows_fetched += len(pairs)
                    patched, skipped = await self._apply_pairs(pairs, source)
                    rows_patched += patched
                    rows_skipped += skipped
                    state.last_polled_at = chunk_end
                    cursor = chunk_end
            success = True
        except RuntimeError as e:
            error_str = f"{type(e).__name__}: {e}"
            logger.error(f"enrichment_refresh_cycle_failed source={source} error_type={type(e).__name__} error={e}")
        except (TimeoutError, OSError, ValueError, TypeError, KeyError, AttributeError) as e:
            error_str = f"{type(e).__name__}: {e}"
            logger.error(f"enrichment_refresh_cycle_unexpected source={source} error_type={type(e).__name__} error={e}")
        duration = time.monotonic() - started_at
        self._cycles_total += 1
        if success:
            self._cycles_success += 1
            state.consecutive_failures = 0
        else:
            self._cycles_failed += 1
            state.consecutive_failures += 1
            if state.consecutive_failures >= int(self._config.max_consecutive_failures):
                state.paused = True
                logger.error(f"enrichment_refresh_driver_source_paused source={source} consecutive_failures={state.consecutive_failures} max_consecutive_failures={self._config.max_consecutive_failures}")
        cycle = EnrichmentCycleSummary(source=source, started_at=wall_started, duration_seconds=duration, rows_fetched=rows_fetched, rows_patched=rows_patched, rows_skipped=rows_skipped, success=success, error=error_str)
        self._history.append(cycle)
        if len(self._history) > _MAX_HISTORY:
            del self._history[: len(self._history) - _MAX_HISTORY]
        logger.info(
            f"enrichment_refresh_cycle_complete source={source} rows_fetched={rows_fetched} "
            f"rows_patched={rows_patched} rows_skipped={rows_skipped} "
            f"duration_seconds={duration:.2f} success={success}"
        )

    async def _apply_pairs(self, pairs: List[Tuple[str, Dict[str, Any]]], source: str) -> Tuple[int, int]:
        """Patch Qdrant for one fetched batch. Returns (patched_count, skipped_count)."""
        if not pairs:
            return 0, 0
        if not self._qdrant.available or self._qdrant.client is None:
            logger.warning(f"enrichment_refresh_qdrant_unavailable source={source} rows_skipped={len(pairs)}")
            return 0, len(pairs)
        return await self._patch_qdrant(pairs)

    async def _patch_qdrant(self, pairs: List[Tuple[str, Dict[str, Any]]]) -> Tuple[int, int]:
        """Issue set_payload calls in parallel, filtering by domain_name (not payload_id_field).

        One domain can back multiple concurrent Qdrant points (multiple live
        auctions for that domain) — a domain_name filter patches all of them
        in a single set_payload call.

        :return: Tuple (patched_count, skipped_count)
        """
        if _qm is None:
            logger.warning("enrichment_refresh_qdrant_client_not_installed")
            return 0, len(pairs)
        cfg = self._qdrant.config
        client = self._qdrant.client

        def _mk_filter(domain: str) -> Any:
            return _qm.Filter(must=[_qm.FieldCondition(key=_QDRANT_FILTER_KEY, match=_qm.MatchValue(value=domain))])

        coros = [client.set_payload(collection_name=cfg.collection_name, payload=upd, points=_mk_filter(domain), wait=True) for domain, upd in pairs]
        results = await asyncio.gather(*coros, return_exceptions=True)
        patched = 0
        skipped = 0
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                skipped += 1
                logger.warning(f"enrichment_refresh_patch_failed domain_name={pairs[idx][0]} error_type={type(result).__name__} error={result}")
            else:
                patched += 1
        return patched, skipped

    def get_summary(self) -> EnrichmentDriverSummary:
        """Snapshot of driver state for ops endpoints."""
        sources = {
            name: EnrichmentSourceSummary(
                active=True,
                paused=state.paused,
                consecutive_failures=state.consecutive_failures,
                last_polled_at=state.last_polled_at,
            )
            for name, state in self._states.items()
        }
        return EnrichmentDriverSummary(
            enabled=self.enabled,
            running=self.running,
            interval_seconds=float(self._config.interval_seconds),
            sources=sources,
            cycles_total=self._cycles_total,
            cycles_success=self._cycles_success,
            cycles_failed=self._cycles_failed,
            history=list(self._history),
        )
