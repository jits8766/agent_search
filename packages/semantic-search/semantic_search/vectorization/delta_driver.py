"""Time-based mutable-field delta refresh driver.

Polls ``auction_audit_cln`` at a configurable cadence, deduplicates events
per domain, and for each chunk:
  1. Qdrant ``set_payload`` for mutable fields (price/auction_price ask, current_bid_price, bid_count, auction_type,
     ends_at, …) — no re-encoding; vectors unchanged.
  2. ClickHouse ``write_delta_to_clickhouse`` (when a CH executor is wired) —
     ``INSERT … SELECT FINAL`` patch of the same mutables on
     ``signals_platform_cln.auction_audit_cln``. Soft-fails so a CH outage
     never pauses the Qdrant leg.

Polling is time-based (wall-clock) rather than version-bump-driven because
``auction_audit_cln`` is an external stream independent of the process-local
``SnapshotVersionRegistry``.

Lifecycle mirrors ``VectorRefreshDriver`` (start/stop, get_summary) so the
FastAPI lifespan can wire both drivers with the same pattern.

Layer rules: imports stdlib + core + vectorization.delta_source +
nl_to_sql.athena_client + retrieval.qdrant_adapter + explore.ch_delta_analytics.
Never imports orchestrator or registry.
"""
import asyncio, time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from semantic_search.config.models import DeltaRefreshConfig
from semantic_search.core.exceptions import ValidationError
from semantic_search.core.logging_utils import get_logger
from semantic_search.explore.ch_delta_analytics import write_delta_to_clickhouse
from semantic_search.nl_to_sql.athena_client import AthenaClient
from semantic_search.retrieval.qdrant_adapter import QdrantClientFactory
from semantic_search.vectorization.delta_source import fetch_delta

try:
    from qdrant_client import models as _qm
except ImportError:
    _qm = None

logger = get_logger(__name__)

_MAX_HISTORY: int = 100


@dataclass
class DeltaCycleSummary:
    """Single delta refresh cycle outcome."""

    started_at: float
    duration_seconds: float
    rows_fetched: int
    rows_patched: int
    rows_skipped: int
    success: bool
    error: Optional[str] = None


@dataclass
class DeltaDriverSummary:
    """Aggregate snapshot of driver state for ops endpoints."""

    enabled: bool
    running: bool
    paused_after_failures: bool
    interval_seconds: float
    consecutive_failures: int
    cycles_total: int
    cycles_success: int
    cycles_failed: int
    last_polled_at: float
    history: List[DeltaCycleSummary] = field(default_factory=list)


class DeltaRefreshDriver:
    """Polls auction_audit_cln and patches Qdrant payloads for mutable fields.

    :param config: DeltaRefreshConfig - Validated cadence + source + field params.
    :param athena_client: AthenaClient - Pre-built Athena client for the source query.
    :param qdrant_factory: QdrantClientFactory - Holds the AsyncQdrantClient.
    :raises ValidationError: When required dependencies are missing or wrong-typed.
    """

    def __init__(self, config: DeltaRefreshConfig, athena_client: AthenaClient, qdrant_factory: QdrantClientFactory):
        if config is None or not isinstance(config, DeltaRefreshConfig):
            raise ValidationError("DeltaRefreshDriver requires a typed DeltaRefreshConfig")
        if athena_client is None or not isinstance(athena_client, AthenaClient):
            raise ValidationError("DeltaRefreshDriver requires an AthenaClient")
        if qdrant_factory is None or not isinstance(qdrant_factory, QdrantClientFactory):
            raise ValidationError("DeltaRefreshDriver requires a QdrantClientFactory")
        self._config = config
        self._athena = athena_client
        self._qdrant = qdrant_factory
        self._task: Optional[asyncio.Task[None]] = None
        self._last_polled_at: float = 0.0
        self._consecutive_failures: int = 0
        self._paused_after_failures: bool = False
        self._cycles_total: int = 0
        self._cycles_success: int = 0
        self._cycles_failed: int = 0
        self._history: List[DeltaCycleSummary] = []
        self._lock = asyncio.Lock()
        self._ch_executor: Optional[Any] = None

    def set_ch_executor(self, ch_executor: Optional[Any]) -> None:
        """Wire a ClickHouseExecutor so each delta cycle also writes to signals_platform_cln.auction_audit_cln.

        Call after construction when the analytics ClickHouse executor is available.
        Passing None disables CH writes for this driver instance.
        """
        self._ch_executor = ch_executor

    @property
    def enabled(self) -> bool:
        return bool(self._config.enabled)

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        """Spawn the background polling task. Idempotent; no-op when disabled."""
        if not self.enabled:
            logger.info("delta_refresh_driver_disabled")
            return
        if self.running:
            return
        self._paused_after_failures = False
        self._consecutive_failures = 0
        self._last_polled_at = time.time() - float(self._config.lookback_minutes) * 60.0
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(f"delta_refresh_driver_started interval_seconds={self._config.interval_seconds} max_consecutive_failures={self._config.max_consecutive_failures}")

    async def stop(self) -> None:
        """Cancel and await the background task. Idempotent."""
        if self._task is None:
            return
        task = self._task
        self._task = None
        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        logger.info("delta_refresh_driver_stopped")

    async def _poll_loop(self) -> None:
        try:
            while True:
                if self._paused_after_failures:
                    return
                await self._tick()
                await asyncio.sleep(float(self._config.interval_seconds))
        except asyncio.CancelledError:
            raise

    async def _tick(self) -> None:
        """One poll iteration — at most one delta cycle runs at a time."""
        if self._lock.locked():
            logger.warning("delta_refresh_skip_overlap")
            return
        async with self._lock:
            await self._run_cycle()

    async def _write_clickhouse(self, pairs: List[Tuple[str, Dict[str, Any]]]) -> int:
        """Patch mutable fields in ClickHouse. Soft-fail; never raises.

        :return: rows_written from ChDeltaWriteSummary (0 when CH unwired / empty / error).
        """
        if not pairs or self._ch_executor is None:
            return 0
        try:
            summary = await write_delta_to_clickhouse(pairs, self._ch_executor)
            return int(summary.rows_written)
        except (TimeoutError, OSError, RuntimeError, ValueError, TypeError) as e:
            logger.warning(
                f"delta_refresh_ch_write_failed error_type={type(e).__name__} error={e} "
                f"pairs={len(pairs)}"
            )
            return 0

    async def _run_cycle(self) -> None:
        """Fetch delta rows in chunk_minutes slices; patch Qdrant + ClickHouse (lock held).

        Walks from ``_last_polled_at`` to ``now_ts`` in ``chunk_minutes``-sized
        time windows.  ``_last_polled_at`` advances after each chunk so a
        mid-cycle failure recovers from the last committed chunk on the next
        tick rather than replaying the full missed window.

        ClickHouse writes are best-effort: failures are logged and do not mark
        the cycle failed (Athena + Qdrant progress still commits).
        """
        if not self._athena.credentials_available:
            logger.warning("delta_refresh_skipped reason=credentials_unavailable last_polled_at preserved catchup_on_recovery=true")
            return
        now_ts = time.time()
        since_ts = self._last_polled_at if self._last_polled_at > 0.0 else now_ts - float(self._config.lookback_minutes) * 60.0
        started_at = time.monotonic()
        wall_started = now_ts
        success = False
        error_str: Optional[str] = None
        rows_fetched = 0
        rows_patched = 0
        rows_skipped = 0
        rows_ch_written = 0
        chunk_sec = float(self._config.chunk_minutes) * 60.0
        cursor = since_ts
        try:
            while cursor < now_ts:
                chunk_end = min(cursor + chunk_sec, now_ts)
                pairs: List[Tuple[str, Dict[str, Any]]] = await fetch_delta(self._athena, self._config, cursor, chunk_end)
                rows_fetched += len(pairs)
                if pairs and self._qdrant.available and self._qdrant.client is not None:
                    patched, skipped = await self._patch_qdrant(pairs)
                    rows_patched += patched
                    rows_skipped += skipped
                elif pairs and not self._qdrant.available:
                    rows_skipped += len(pairs)
                    logger.warning(f"delta_refresh_qdrant_unavailable chunk_end={chunk_end} rows_skipped={len(pairs)}")
                if pairs:
                    rows_ch_written += await self._write_clickhouse(pairs)
                self._last_polled_at = chunk_end
                cursor = chunk_end
            success = True
        except RuntimeError as e:
            error_str = f"{type(e).__name__}: {e}"
            logger.error(f"delta_refresh_cycle_failed error_type={type(e).__name__} error={e}")
        except (TimeoutError, OSError, ValueError, TypeError, KeyError, AttributeError) as e:
            error_str = f"{type(e).__name__}: {e}"
            logger.error(f"delta_refresh_cycle_unexpected error_type={type(e).__name__} error={e}")
        duration = time.monotonic() - started_at
        self._cycles_total += 1
        if success:
            self._cycles_success += 1
            self._consecutive_failures = 0
        else:
            self._cycles_failed += 1
            self._consecutive_failures += 1
            if self._consecutive_failures >= int(self._config.max_consecutive_failures):
                self._paused_after_failures = True
                logger.error(f"delta_refresh_driver_paused consecutive_failures={self._consecutive_failures} max_consecutive_failures={self._config.max_consecutive_failures}")
        cycle = DeltaCycleSummary(started_at=wall_started, duration_seconds=duration, rows_fetched=rows_fetched, rows_patched=rows_patched, rows_skipped=rows_skipped, success=success, error=error_str)
        self._history.append(cycle)
        if len(self._history) > _MAX_HISTORY:
            del self._history[: len(self._history) - _MAX_HISTORY]
        logger.info(
            f"delta_refresh_cycle_complete rows_fetched={rows_fetched} rows_patched={rows_patched} "
            f"rows_skipped={rows_skipped} rows_ch_written={rows_ch_written} "
            f"duration_seconds={duration:.2f} success={success}"
        )

    async def _patch_qdrant(self, pairs: List[Tuple[str, Dict[str, Any]]]) -> Tuple[int, int]:
        """Issue set_payload calls in parallel for each (item_id, updates) pair.

        :return: Tuple (patched_count, skipped_count)
        """
        if _qm is None:
            logger.warning("delta_refresh_qdrant_client_not_installed")
            return 0, len(pairs)
        cfg = self._qdrant.config
        client = self._qdrant.client
        def _mk_filter(iid: str) -> Any: return _qm.Filter(must=[_qm.FieldCondition(key=cfg.payload_id_field, match=_qm.MatchValue(value=iid))])
        coros = [client.set_payload(collection_name=cfg.collection_name, payload=upd, points=_mk_filter(iid), wait=True) for iid, upd in pairs]
        results = await asyncio.gather(*coros, return_exceptions=True)
        patched = 0
        skipped = 0
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                skipped += 1
                logger.warning(f"delta_refresh_patch_failed item_id={pairs[idx][0]} error_type={type(result).__name__} error={result}")
            else:
                patched += 1
        return patched, skipped

    def get_summary(self) -> DeltaDriverSummary:
        """Snapshot of driver state for ops endpoints."""
        return DeltaDriverSummary(
            enabled=self.enabled,
            running=self.running,
            paused_after_failures=self._paused_after_failures,
            interval_seconds=float(self._config.interval_seconds),
            consecutive_failures=self._consecutive_failures,
            cycles_total=self._cycles_total,
            cycles_success=self._cycles_success,
            cycles_failed=self._cycles_failed,
            last_polled_at=self._last_polled_at,
            history=list(self._history),
        )
