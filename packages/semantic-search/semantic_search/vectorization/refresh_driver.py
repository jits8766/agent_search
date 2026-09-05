"""Snapshot-version-aware refresh driver for the offline vector index.

Periodically polls the shared ``SnapshotVersionRegistry`` and triggers an
``OfflineIndexer.run`` whenever the inventory snapshot version advances.
The driver itself owns no document source — the caller injects an async
``document_source_factory(snapshot_version)`` callable that returns the
documents the indexer should re-encode for that snapshot. This keeps the
driver decoupled from the warehouse / streaming source we read from.

Lifecycle mirrors ``LLMProvider.start_background_refresh / stop_background_refresh``
so the registry can wire it into the FastAPI lifespan with the same shape.

Design choices
--------------
- **Edge-triggered.** A run fires only when the registry's current
  version differs from the last version we indexed. Two consecutive
  ticks at the same version do nothing.
- **Drop-overlapping policy.** If a previous run is still in flight when
  the next tick comes around, the driver logs and skips that tick rather
  than queueing a parallel run. The next version bump (or the next tick
  while idle) resumes processing — this prevents pile-ups when the
  indexer is slower than the tick interval.
- **Bounded back-off.** ``max_consecutive_failures`` consecutive failed
  runs pause the loop until ``stop()`` + ``start()`` is called. The pause
  is a defensive guard against runaway error loops; the registry can
  detect the pause via ``get_summary()`` and surface it.
- **Soft-fail on construction.** Like the indexer, the driver is happy
  to be constructed when Qdrant is unavailable — every tick will be a
  no-op until a future tick observes the registry is healthy.

Layer rules: imports stdlib + ``core`` + sibling vectorization +
``ingest.in_memory_consumer.SnapshotVersionRegistry``. Never imports
orchestration code.
"""
import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, List, Optional

from semantic_search.config.models import VectorRefreshConfig
from semantic_search.core.exceptions import ValidationError
from semantic_search.core.logging_utils import get_logger
from semantic_search.ingest.in_memory_consumer import SnapshotVersionRegistry
from semantic_search.vectorization.indexer import IndexerRunSummary, OfflineIndexer

logger = get_logger(__name__)


DocumentSourceFactory = Callable[[int], Awaitable[Any]]


@dataclass
class RefreshCycleSummary:
    """Single refresh cycle outcome (kept in driver history)."""

    snapshot_version: int
    started_at: float
    duration_seconds: float
    points_seen: int
    points_upserted: int
    success: bool
    error: Optional[str] = None


@dataclass
class RefreshDriverSummary:
    """Aggregate snapshot of driver state for ops endpoints."""

    enabled: bool
    running: bool
    paused_after_failures: bool
    interval_seconds: float
    last_seen_version: int
    consecutive_failures: int
    cycles_total: int
    cycles_success: int
    cycles_failed: int
    history: List[RefreshCycleSummary] = field(default_factory=list)


class VectorRefreshDriver:
    """Polls a snapshot registry and drives ``OfflineIndexer.run`` on bumps.

    :param config: VectorRefreshConfig - Validated cadence + back-off
    :param indexer: OfflineIndexer - The indexer the driver invokes
    :param snapshot_registry: SnapshotVersionRegistry - Shared registry
    :param document_source_factory: DocumentSourceFactory - Async callable
        ``(snapshot_version) -> document iterable``. Awaited per cycle so
        the caller can hit the warehouse fresh on every refresh.
    :raises ValidationError: When required dependencies are missing/wrong-typed.
    """

    _MAX_HISTORY = 100

    def __init__(self, config: VectorRefreshConfig, indexer: OfflineIndexer, snapshot_registry: SnapshotVersionRegistry, document_source_factory: DocumentSourceFactory):
        if config is None or not isinstance(config, VectorRefreshConfig):
            raise ValidationError("VectorRefreshDriver requires a typed VectorRefreshConfig")
        if indexer is None or not isinstance(indexer, OfflineIndexer):
            raise ValidationError("VectorRefreshDriver requires an OfflineIndexer")
        if snapshot_registry is None or not isinstance(snapshot_registry, SnapshotVersionRegistry):
            raise ValidationError("VectorRefreshDriver requires a SnapshotVersionRegistry")
        if document_source_factory is None or not callable(document_source_factory):
            raise ValidationError("VectorRefreshDriver requires a callable document_source_factory")

        self._config = config
        self._indexer = indexer
        self._registry = snapshot_registry
        self._source_factory = document_source_factory

        self._task: Optional[asyncio.Task[None]] = None
        self._last_seen_version: int = -1
        self._consecutive_failures: int = 0
        self._paused_after_failures: bool = False
        self._priority_pending: bool = False
        self._cycles_total: int = 0
        self._cycles_success: int = 0
        self._cycles_failed: int = 0
        self._history: List[RefreshCycleSummary] = []
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self._config.enabled)

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        """Spawn the background polling task. Idempotent + safe when disabled."""
        if not self.enabled:
            logger.info("vector_refresh_driver_disabled")
            return
        if self.running:
            return
        self._paused_after_failures = False
        self._consecutive_failures = 0
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(
            f"vector_refresh_driver_started interval_seconds={self._config.interval_seconds} "
            f"max_consecutive_failures={self._config.max_consecutive_failures}"
        )

    async def stop(self) -> None:
        """Cancel + await the background task. Idempotent."""
        if self._task is None:
            return
        task = self._task
        self._task = None
        if not task.done():
            task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        logger.info("vector_refresh_driver_stopped")

    def notify_priority_refresh(self) -> None:
        """Signal that an auction auto-extension was detected.

        Called by the SnapshotVersionRegistry priority hook (wired in registry.py).
        Sets a flag consumed by _poll_loop to use the shorter
        ``priority_interval_seconds`` cadence on the next sleep, ensuring the
        extended auction's vector is refreshed faster than the normal interval.
        Thread-safe: flag is a plain bool write, read only from the async loop.
        """
        self._priority_pending = True
        logger.info("vector_refresh_priority_scheduled")

    async def _poll_loop(self) -> None:
        try:
            while True:
                if self._paused_after_failures:
                    return
                await self._tick()
                if self._priority_pending:
                    self._priority_pending = False
                    interval = float(self._config.priority_interval_seconds)
                else:
                    interval = float(self._config.interval_seconds)
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise

    async def _tick(self) -> None:
        """One poll iteration — runs at most one indexer cycle."""
        current_version = self._registry.version
        if current_version == self._last_seen_version:
            return
        if self._lock.locked():
            logger.warning(
                f"vector_refresh_skip_overlap snapshot_version={current_version} "
                f"last_seen_version={self._last_seen_version}"
            )
            return

        async with self._lock:
            await self._run_cycle(current_version)

    async def _run_cycle(self, snapshot_version: int) -> None:
        """Invoke the indexer for one snapshot version (lock already held)."""
        started_at = time.monotonic()
        wall_started = time.time()
        success = False
        error_str: Optional[str] = None
        seen = 0
        upserted = 0
        try:
            source = await self._source_factory(snapshot_version)
            summary: IndexerRunSummary = await self._indexer.run(source)
            seen = summary.points_seen
            upserted = summary.points_upserted
            success = summary.failures == 0 and summary.qdrant_available
        except (ValidationError, RuntimeError) as e:
            error_str = f"{type(e).__name__}"
            logger.error(
                f"vector_refresh_cycle_failed snapshot_version={snapshot_version} "
                f"error_type={type(e).__name__}"
            )
        except Exception as e:
            error_str = f"{type(e).__name__}"
            logger.error(
                f"vector_refresh_cycle_unexpected snapshot_version={snapshot_version} "
                f"error_type={type(e).__name__}"
            )
        duration = time.monotonic() - started_at

        self._cycles_total += 1
        if success:
            self._cycles_success += 1
            self._consecutive_failures = 0
            self._last_seen_version = snapshot_version
        else:
            self._cycles_failed += 1
            self._consecutive_failures += 1
            if self._consecutive_failures >= int(self._config.max_consecutive_failures):
                self._paused_after_failures = True
                logger.error(
                    f"vector_refresh_driver_paused consecutive_failures={self._consecutive_failures} "
                    f"max_consecutive_failures={self._config.max_consecutive_failures}"
                )

        cycle = RefreshCycleSummary(
            snapshot_version=snapshot_version,
            started_at=wall_started,
            duration_seconds=duration,
            points_seen=seen,
            points_upserted=upserted,
            success=success,
            error=error_str,
        )
        self._history.append(cycle)
        if len(self._history) > self._MAX_HISTORY:
            del self._history[: len(self._history) - self._MAX_HISTORY]

    def get_summary(self) -> RefreshDriverSummary:
        """Snapshot of driver state for ops + ``/vectorization/*`` endpoints."""
        return RefreshDriverSummary(
            enabled=self.enabled,
            running=self.running,
            paused_after_failures=self._paused_after_failures,
            interval_seconds=float(self._config.interval_seconds),
            last_seen_version=self._last_seen_version,
            consecutive_failures=self._consecutive_failures,
            cycles_total=self._cycles_total,
            cycles_success=self._cycles_success,
            cycles_failed=self._cycles_failed,
            history=list(self._history),
        )
