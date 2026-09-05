"""Real-time event ingest driver: Athena → ClickHouse → Qdrant enrichment.

Polls two Athena sources on separate configurable cadences:
  1. ``the_resale_place.item_bids_cln``        → analytics.bid_events
  2. ``the_resale_place.member_items_watch_cln`` → analytics.watch_events

After each ClickHouse write, optionally enriches Qdrant payloads with
real-time signals read from the freshly-updated ClickHouse tables:

  Bid cycle (post bid_events write):
    - ``mv_bid_velocity_by_auction_hour``       → Qdrant payload ``bid_velocity_1h``
    - ``mv_unique_bidder_count_by_item``        → Qdrant payload ``unique_bidder_count_4h``

  Watch cycle (post watch_events write):
    - ``mv_watch_density_by_item_day``          → Qdrant payload ``watch_density_1d``
    - ``mv_bidder_watch_density_by_item_day``   → Qdrant payload ``bidder_watch_density_1d``
    - derived: bidder_density / watch_density   → Qdrant payload ``engagement_ratio``
    - ``watch_events`` direct scan (today)      → Qdrant payload ``net_watch_delta_1d``

Lifecycle mirrors DeltaRefreshDriver (start/stop/get_summary) so the
FastAPI lifespan can wire both drivers with the same pattern.

Layer rules: imports stdlib + core + vectorization.*_source +
nl_to_sql.athena_client + analytics.clickhouse_executor +
retrieval.qdrant_adapter. Never imports orchestrator or registry.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from semantic_search.analytics.ch_schema import EVENT_TABLE_DDL_STATEMENTS
from semantic_search.config.analytics_models import _coerce_bool
from semantic_search.core.exceptions import ValidationError
from semantic_search.core.logging_utils import get_logger
from semantic_search.nl_to_sql.athena_client import AthenaClient
from semantic_search.vectorization.bid_event_source import fetch_bid_events
from semantic_search.vectorization.watch_event_source import fetch_watch_events

try:
    from qdrant_client import models as _qm
except ImportError:
    _qm = None

logger = get_logger(__name__)

_MAX_HISTORY: int = 100
_CH_BID_TABLE: str = "analytics.bid_events"
_CH_WATCH_TABLE: str = "analytics.watch_events"

# ClickHouse MV queries for Qdrant signal enrichment.
# Reads the last 1-hour bid velocity per auction_id.
_BID_VELOCITY_SQL = (
    "SELECT auction_id,"
    " countMerge(bid_count_state) AS bid_velocity_1h"
    " FROM analytics.mv_bid_velocity_by_auction_hour"
    " WHERE event_hour >= now() - INTERVAL 1 HOUR"
    "   AND auction_id > 0"
    " GROUP BY auction_id"
    " HAVING bid_velocity_1h > 0"
    " LIMIT {limit}"
)

# Reads today's watch density per member_item_id, joined to auction_id via bid_events.
# Fetches both passive (type 1+9) and bidder-intent (type 9 only) counts in one round-trip.
_WATCH_DENSITY_SQL = (
    "SELECT b.auction_id,"
    " countMerge(w.active_watch_state) AS watch_density_1d,"
    " ifNull(countMerge(bw.bidder_watch_state), 0) AS bidder_watch_density_1d"
    " FROM analytics.mv_watch_density_by_item_day w"
    " LEFT JOIN analytics.mv_bidder_watch_density_by_item_day bw"
    " ON w.member_item_id = bw.member_item_id AND w.event_day = bw.event_day"
    " INNER JOIN ("
    "   SELECT member_item_id, auction_id"
    "   FROM analytics.bid_events"
    "   WHERE auction_id > 0"
    "   GROUP BY member_item_id, auction_id"
    " ) b ON w.member_item_id = b.member_item_id"
    " WHERE w.event_day = toDate(now())"
    " GROUP BY b.auction_id"
    " HAVING watch_density_1d > 0"
    " LIMIT {limit}"
)

# Reads unique distinct bidder count per auction over the last 4 hours.
# Uses mv_unique_bidder_count_by_item (keyed on member_item_id) joined to
# auction_id via bid_events bridge.
_UNIQUE_BIDDER_SQL = (
    "SELECT b.auction_id,"
    " toUInt32(uniqMerge(u.unique_bidder_state)) AS unique_bidder_count_4h"
    " FROM analytics.mv_unique_bidder_count_by_item u"
    " INNER JOIN ("
    "   SELECT member_item_id, auction_id"
    "   FROM analytics.bid_events"
    "   WHERE auction_id > 0"
    "   GROUP BY member_item_id, auction_id"
    " ) b ON u.member_item_id = b.member_item_id"
    " WHERE u.event_hour >= now() - INTERVAL 4 HOUR"
    " GROUP BY b.auction_id"
    " HAVING unique_bidder_count_4h > 0"
    " LIMIT {limit}"
)

# Net watch delta today: adds minus removes per auction.
# Direct scan on watch_events for today's partition — fast because
# event_utc_ts is the TTL/partition column and the window is a single day.
_NET_WATCH_DELTA_SQL = (
    "SELECT b.auction_id,"
    " toInt32(countIf(we.is_deleted = 0)) - toInt32(countIf(we.is_deleted = 1))"
    "   AS net_watch_delta_1d"
    " FROM analytics.watch_events we"
    " INNER JOIN ("
    "   SELECT member_item_id, auction_id"
    "   FROM analytics.bid_events"
    "   WHERE auction_id > 0"
    "   GROUP BY member_item_id, auction_id"
    " ) b ON we.member_item_id = b.member_item_id"
    " WHERE toDate(we.event_utc_ts) = today()"
    " GROUP BY b.auction_id"
    " LIMIT {limit}"
)


@dataclass
class EventIngestCycleSummary:
    """Outcome of one ingest cycle (bid or watch)."""

    source: str
    started_at: float
    duration_seconds: float
    rows_fetched: int
    rows_written: int
    rows_skipped: int
    qdrant_enriched: int
    success: bool
    error: Optional[str] = None


@dataclass
class EventIngestDriverSummary:
    """Aggregate snapshot of driver state for ops endpoints."""

    bid_enabled: bool
    watch_enabled: bool
    running: bool
    paused_bid: bool
    paused_watch: bool
    bid_interval_seconds: float
    watch_interval_seconds: float
    bid_consecutive_failures: int
    watch_consecutive_failures: int
    cycles_total: int
    cycles_success: int
    cycles_failed: int
    last_bid_polled_at: float
    last_watch_polled_at: float
    history: List[EventIngestCycleSummary] = field(default_factory=list)


class EventIngestDriver:
    """Polls bid + watch Athena sources, writes to ClickHouse, enriches Qdrant.

    :param bid_config: Optional dict with bid ingest params (enabled, interval_seconds,
        max_consecutive_failures, source_database, source_table, winning_bids_table,
        lookback_minutes, chunk_minutes, batch_size, timeout_seconds).
    :param watch_config: Optional dict with watch ingest params (same shape, plus
        watch_types_table instead of winning_bids_table).
    :param qdrant_enrich_config: Optional dict with Qdrant enrichment params
        (enabled, enrich_limit).
    :param athena_client: AthenaClient - Pre-built Athena client.
    :param ch_executor: Any - ClickHouseExecutor instance (set via set_ch_executor).
    :param qdrant_factory: Optional - QdrantClientFactory (None disables enrichment).
    :raises ValidationError: When required dependencies are missing.
    """

    def __init__(
        self,
        bid_config: Optional[Dict[str, Any]],
        watch_config: Optional[Dict[str, Any]],
        qdrant_enrich_config: Optional[Dict[str, Any]],
        athena_client: AthenaClient,
        qdrant_factory: Optional[Any] = None,
    ) -> None:
        if athena_client is None or not isinstance(athena_client, AthenaClient):
            raise ValidationError("EventIngestDriver requires an AthenaClient")
        self._athena = athena_client
        self._ch_executor: Optional[Any] = None
        self._qdrant = qdrant_factory

        # Bid ingest config
        self._bid_cfg = bid_config or {}
        # _coerce_bool: Katana/env inject strings ("false" must not become True via bool()).
        self._bid_enabled = _coerce_bool(self._bid_cfg.get("enabled", False))
        self._bid_interval = float(self._bid_cfg.get("interval_seconds", 60.0))
        self._bid_max_failures = int(self._bid_cfg.get("max_consecutive_failures", 3))
        self._bid_db = str(self._bid_cfg.get("source_database", "the_resale_place"))
        self._bid_table = str(self._bid_cfg.get("source_table", "item_bids_cln"))
        self._bid_winning_table = str(self._bid_cfg.get("winning_bids_table", "item_winning_bids_cln"))
        self._bid_lookback = int(self._bid_cfg.get("lookback_minutes", 60))
        self._bid_chunk = int(self._bid_cfg.get("chunk_minutes", 60))
        self._bid_batch = int(self._bid_cfg.get("batch_size", 10000))
        self._bid_timeout = float(self._bid_cfg.get("timeout_seconds", 30.0))

        # Watch ingest config
        self._watch_cfg = watch_config or {}
        self._watch_enabled = _coerce_bool(self._watch_cfg.get("enabled", False))
        self._watch_interval = float(self._watch_cfg.get("interval_seconds", 120.0))
        self._watch_max_failures = int(self._watch_cfg.get("max_consecutive_failures", 3))
        self._watch_db = str(self._watch_cfg.get("source_database", "the_resale_place"))
        self._watch_table = str(self._watch_cfg.get("source_table", "member_items_watch_cln"))
        self._watch_types_table = str(self._watch_cfg.get("watch_types_table", "member_items_watch_types_cln"))
        self._watch_lookback = int(self._watch_cfg.get("lookback_minutes", 60))
        self._watch_chunk = int(self._watch_cfg.get("chunk_minutes", 60))
        self._watch_batch = int(self._watch_cfg.get("batch_size", 10000))
        self._watch_timeout = float(self._watch_cfg.get("timeout_seconds", 30.0))

        # Qdrant enrichment config — default off (matches EVENT_INGEST_QDRANT_ENRICH_ENABLED:-false)
        _qe = qdrant_enrich_config or {}
        self._enrich_enabled = _coerce_bool(_qe.get("enabled", False)) and qdrant_factory is not None
        self._enrich_limit = int(_qe.get("enrich_limit", 5000))
        if not (1 <= self._enrich_limit <= 100_000):
            raise ValidationError(f"enrich_limit out of range [1, 100000]: {self._enrich_limit}")
        # Pre-compute enrichment SQL with validated int limit so runtime execute calls
        # receive a fully-resolved string (no formatting at query time).
        self._bid_velocity_sql: str = _BID_VELOCITY_SQL.format(limit=self._enrich_limit)
        self._watch_density_sql: str = _WATCH_DENSITY_SQL.format(limit=self._enrich_limit)
        self._unique_bidder_sql: str = _UNIQUE_BIDDER_SQL.format(limit=self._enrich_limit)
        self._net_watch_delta_sql: str = _NET_WATCH_DELTA_SQL.format(limit=self._enrich_limit)

        # Driver state
        self._bid_task: Optional[asyncio.Task] = None
        self._watch_task: Optional[asyncio.Task] = None
        self._bid_last_polled: float = 0.0
        self._watch_last_polled: float = 0.0
        self._bid_consecutive_failures: int = 0
        self._watch_consecutive_failures: int = 0
        self._bid_paused: bool = False
        self._watch_paused: bool = False
        self._cycles_total: int = 0
        self._cycles_success: int = 0
        self._cycles_failed: int = 0
        self._history: List[EventIngestCycleSummary] = []
        self._bid_lock = asyncio.Lock()
        self._watch_lock = asyncio.Lock()
        self._schema_ready: bool = False

    def set_ch_executor(self, ch_executor: Optional[Any]) -> None:
        """Wire a ClickHouseExecutor. Call after construction before start()."""
        self._ch_executor = ch_executor

    @property
    def running(self) -> bool:
        bid_running = self._bid_task is not None and not self._bid_task.done()
        watch_running = self._watch_task is not None and not self._watch_task.done()
        return bid_running or watch_running

    async def _ensure_event_schema(self) -> bool:
        """Create analytics.bid_events / watch_events tables if they don't exist.

        Returns True when all DDL statements succeed, False on any failure.
        """
        if self._ch_executor is None:
            logger.warning("event_ingest_schema_skipped reason=ch_executor_not_wired")
            return False
        failures = 0
        for stmt in EVENT_TABLE_DDL_STATEMENTS:
            try:
                await self._ch_executor.execute_insert(stmt.strip(), timeout_seconds=30.0)
            except Exception as exc:
                failures += 1
                logger.warning(f"event_ingest_schema_ddl_failed stmt_prefix={stmt[:60]!r} error_type={type(exc).__name__} error={str(exc)[:200]}")
        if failures == 0:
            self._schema_ready = True
            logger.info("event_ingest_schema_ensured tables=bid_events,watch_events")
        return failures == 0

    async def start(self) -> None:
        """Spawn bid + watch background polling tasks. Idempotent."""
        await self._ensure_event_schema()
        if self._bid_enabled and (self._bid_task is None or self._bid_task.done()):
            self._bid_paused = False
            self._bid_consecutive_failures = 0
            self._bid_last_polled = time.time() - float(self._bid_lookback) * 60.0
            self._bid_task = asyncio.create_task(self._bid_poll_loop())
            logger.info(
                f"event_ingest_driver_bid_started"
                f" source={self._bid_db}.{self._bid_table}"
                f" interval_seconds={self._bid_interval}"
            )
        if self._watch_enabled and (self._watch_task is None or self._watch_task.done()):
            self._watch_paused = False
            self._watch_consecutive_failures = 0
            self._watch_last_polled = time.time() - float(self._watch_lookback) * 60.0
            self._watch_task = asyncio.create_task(self._watch_poll_loop())
            logger.info(
                f"event_ingest_driver_watch_started"
                f" source={self._watch_db}.{self._watch_table}"
                f" interval_seconds={self._watch_interval}"
            )

    async def stop(self) -> None:
        """Cancel and await both polling tasks. Idempotent."""
        for task_attr in ("_bid_task", "_watch_task"):
            task = getattr(self, task_attr, None)
            if task is not None:
                setattr(self, task_attr, None)
                if not task.done():
                    task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        logger.info("event_ingest_driver_stopped")

    # ── Bid polling loop ──────────────────────────────────────────────────────

    async def _bid_poll_loop(self) -> None:
        try:
            while True:
                if self._bid_paused:
                    return
                await self._bid_tick()
                await asyncio.sleep(self._bid_interval)
        except asyncio.CancelledError:
            raise

    async def _bid_tick(self) -> None:
        if self._bid_lock.locked():
            logger.warning("event_ingest_bid_skip_overlap")
            return
        async with self._bid_lock:
            await self._run_bid_cycle()

    async def _run_bid_cycle(self) -> None:
        """Fetch bid events in chunk slices, INSERT to ClickHouse, enrich Qdrant."""
        if not self._athena.credentials_available:
            logger.warning("event_ingest_bid_skipped reason=credentials_unavailable")
            return
        if self._ch_executor is None:
            logger.warning("event_ingest_bid_skipped reason=ch_executor_not_wired")
            return
        now_ts = time.time()
        since_ts = self._bid_last_polled
        chunk_sec = float(self._bid_chunk) * 60.0
        cursor = since_ts
        started_at = time.monotonic()
        wall_started = now_ts
        success = False
        error_str: Optional[str] = None
        rows_fetched = 0
        rows_written = 0
        rows_skipped = 0
        qdrant_enriched = 0
        try:
            while cursor < now_ts:
                chunk_end = min(cursor + chunk_sec, now_ts)
                rows = await fetch_bid_events(
                    self._athena,
                    self._bid_db,
                    self._bid_table,
                    self._bid_winning_table,
                    cursor,
                    chunk_end,
                    self._bid_batch,
                    self._bid_timeout,
                )
                rows_fetched += len(rows)
                if rows:
                    written, skipped = await self._write_to_clickhouse(rows, _CH_BID_TABLE)
                    rows_written += written
                    rows_skipped += skipped
                self._bid_last_polled = chunk_end
                cursor = chunk_end
            # Enrich Qdrant with bid signals after writing
            if self._enrich_enabled and rows_written > 0:
                vel_enriched = await self._enrich_qdrant_bid_velocity()
                uniq_enriched = await self._enrich_qdrant_unique_bidders()
                qdrant_enriched = vel_enriched + uniq_enriched
            success = True
        except RuntimeError as e:
            error_str = f"{type(e).__name__}: {e}"
            logger.error(f"event_ingest_bid_cycle_failed error={e}")
        except Exception as e:
            error_str = f"{type(e).__name__}: {e}"
            logger.error(f"event_ingest_bid_cycle_unexpected error_type={type(e).__name__} error={e}")
        self._record_cycle("bid_events", wall_started, time.monotonic() - started_at, rows_fetched, rows_written, rows_skipped, qdrant_enriched, success, error_str)
        self._update_failure_counter("bid", success)

    # ── Watch polling loop ────────────────────────────────────────────────────

    async def _watch_poll_loop(self) -> None:
        try:
            while True:
                if self._watch_paused:
                    return
                await self._watch_tick()
                await asyncio.sleep(self._watch_interval)
        except asyncio.CancelledError:
            raise

    async def _watch_tick(self) -> None:
        if self._watch_lock.locked():
            logger.warning("event_ingest_watch_skip_overlap")
            return
        async with self._watch_lock:
            await self._run_watch_cycle()

    async def _run_watch_cycle(self) -> None:
        """Fetch watch events in chunk slices, INSERT to ClickHouse, enrich Qdrant."""
        if not self._athena.credentials_available:
            logger.warning("event_ingest_watch_skipped reason=credentials_unavailable")
            return
        if self._ch_executor is None:
            logger.warning("event_ingest_watch_skipped reason=ch_executor_not_wired")
            return
        now_ts = time.time()
        since_ts = self._watch_last_polled
        chunk_sec = float(self._watch_chunk) * 60.0
        cursor = since_ts
        started_at = time.monotonic()
        wall_started = now_ts
        success = False
        error_str: Optional[str] = None
        rows_fetched = 0
        rows_written = 0
        rows_skipped = 0
        qdrant_enriched = 0
        try:
            while cursor < now_ts:
                chunk_end = min(cursor + chunk_sec, now_ts)
                rows = await fetch_watch_events(
                    self._athena,
                    self._watch_db,
                    self._watch_table,
                    self._watch_types_table,
                    cursor,
                    chunk_end,
                    self._watch_batch,
                    self._watch_timeout,
                )
                rows_fetched += len(rows)
                if rows:
                    written, skipped = await self._write_to_clickhouse(rows, _CH_WATCH_TABLE)
                    rows_written += written
                    rows_skipped += skipped
                self._watch_last_polled = chunk_end
                cursor = chunk_end
            # Enrich Qdrant with watch signals after writing
            if self._enrich_enabled and rows_written > 0:
                watch_enriched = await self._enrich_qdrant_watch_density()
                delta_enriched = await self._enrich_qdrant_net_watch_delta()
                qdrant_enriched = watch_enriched + delta_enriched
            success = True
        except RuntimeError as e:
            error_str = f"{type(e).__name__}: {e}"
            logger.error(f"event_ingest_watch_cycle_failed error={e}")
        except Exception as e:
            error_str = f"{type(e).__name__}: {e}"
            logger.error(f"event_ingest_watch_cycle_unexpected error_type={type(e).__name__} error={e}")
        self._record_cycle("watch_events", wall_started, time.monotonic() - started_at, rows_fetched, rows_written, rows_skipped, qdrant_enriched, success, error_str)
        self._update_failure_counter("watch", success)

    # ── ClickHouse INSERT ─────────────────────────────────────────────────────

    async def _write_to_clickhouse(
        self,
        rows: List[Dict[str, Any]],
        table: str,
    ) -> Tuple[int, int]:
        """INSERT rows into ClickHouse table via executor."""
        if not rows or self._ch_executor is None:
            return 0, len(rows)
        if not self._schema_ready:
            await self._ensure_event_schema()
        _BATCH = 500
        written = 0
        failed = 0
        try:
            for i in range(0, len(rows), _BATCH):
                batch = rows[i : i + _BATCH]
                json_lines = "\n".join(json.dumps(row) for row in batch)
                sql = f"INSERT INTO {table} FORMAT JSONEachRow\n{json_lines}"
                await self._ch_executor.execute_insert(sql, timeout_seconds=60.0)
                written += len(batch)
            logger.info(f"event_ingest_ch_write table={table} rows={written}")
        except Exception as e:
            failed = len(rows) - written
            logger.warning(f"event_ingest_ch_write_failed table={table} rows={len(rows)} written={written} error_type={type(e).__name__} error={e}")
        return written, failed

    # ── Qdrant signal enrichment ──────────────────────────────────────────────

    async def _enrich_qdrant_bid_velocity(self) -> int:
        """Read mv_bid_velocity_by_auction_hour from CH, set_payload on Qdrant."""
        if not self._enrich_enabled or self._ch_executor is None:
            return 0
        if _qm is None:
            return 0
        qdrant_factory = self._qdrant
        if qdrant_factory is None or not qdrant_factory.available or qdrant_factory.client is None:
            return 0
        try:
            sql = self._bid_velocity_sql
            _res = await self._ch_executor.execute(sql)
            rows, latency_ms = _res.rows, _res.latency_ms
            if not rows:
                return 0
            cfg = qdrant_factory.config
            coros = []
            for row in rows:
                auction_id = str(row.get("auction_id", "")).strip()
                velocity = int(float(row.get("bid_velocity_1h") or 0))
                if not auction_id or velocity <= 0:
                    continue
                f = _qm.Filter(must=[_qm.FieldCondition(
                    key=cfg.payload_id_field,
                    match=_qm.MatchValue(value=auction_id),
                )])
                coros.append(qdrant_factory.client.set_payload(
                    collection_name=cfg.collection_name,
                    payload={"bid_velocity_1h": velocity},
                    points=f,
                    wait=False,
                ))
            if not coros:
                return 0
            results = await asyncio.gather(*coros, return_exceptions=True)
            enriched = sum(1 for r in results if not isinstance(r, Exception))
            errors = [r for r in results if isinstance(r, Exception)]
            if errors:
                logger.warning(f"event_ingest_qdrant_bid_enrich_partial enriched={enriched} errors={len(errors)}")
            logger.info(f"event_ingest_qdrant_bid_velocity enriched={enriched} latency_ms={latency_ms:.1f}")
            return enriched
        except Exception as e:
            logger.warning(f"event_ingest_qdrant_bid_enrich_failed error_type={type(e).__name__} error={e}")
            return 0

    async def _enrich_qdrant_watch_density(self) -> int:
        """Read watch density MVs (joined to auction_id) from CH, set_payload on Qdrant.

        Pushes two signals per auction:
          - watch_density_1d: net active watches today (type 1 + type 9)
          - bidder_watch_density_1d: type-9 only (bidder-intent watches)
        """
        if not self._enrich_enabled or self._ch_executor is None:
            return 0
        if _qm is None:
            return 0
        qdrant_factory = self._qdrant
        if qdrant_factory is None or not qdrant_factory.available or qdrant_factory.client is None:
            return 0
        try:
            sql = self._watch_density_sql
            _res = await self._ch_executor.execute(sql)
            rows, latency_ms = _res.rows, _res.latency_ms
            if not rows:
                return 0
            cfg = qdrant_factory.config
            coros = []
            for row in rows:
                auction_id = str(row.get("auction_id", "")).strip()
                density = int(float(row.get("watch_density_1d") or 0))
                bidder_density = int(float(row.get("bidder_watch_density_1d") or 0))
                if not auction_id or density <= 0:
                    continue
                f = _qm.Filter(must=[_qm.FieldCondition(
                    key=cfg.payload_id_field,
                    match=_qm.MatchValue(value=auction_id),
                )])
                engagement_ratio = round(bidder_density / max(density, 1), 3)
                coros.append(qdrant_factory.client.set_payload(
                    collection_name=cfg.collection_name,
                    payload={
                        "watch_density_1d": density,
                        "bidder_watch_density_1d": bidder_density,
                        "engagement_ratio": engagement_ratio,
                    },
                    points=f,
                    wait=False,
                ))
            if not coros:
                return 0
            results = await asyncio.gather(*coros, return_exceptions=True)
            enriched = sum(1 for r in results if not isinstance(r, Exception))
            errors = [r for r in results if isinstance(r, Exception)]
            if errors:
                logger.warning(f"event_ingest_qdrant_watch_enrich_partial enriched={enriched} errors={len(errors)}")
            logger.info(f"event_ingest_qdrant_watch_density enriched={enriched} latency_ms={latency_ms:.1f}")
            return enriched
        except Exception as e:
            logger.warning(f"event_ingest_qdrant_watch_enrich_failed error_type={type(e).__name__} error={e}")
            return 0

    async def _enrich_qdrant_unique_bidders(self) -> int:
        """Read mv_unique_bidder_count_by_item (4h window) from CH, set_payload on Qdrant."""
        if not self._enrich_enabled or self._ch_executor is None:
            return 0
        if _qm is None:
            return 0
        qdrant_factory = self._qdrant
        if qdrant_factory is None or not qdrant_factory.available or qdrant_factory.client is None:
            return 0
        try:
            sql = self._unique_bidder_sql
            _res = await self._ch_executor.execute(sql)
            rows, latency_ms = _res.rows, _res.latency_ms
            if not rows:
                return 0
            cfg = qdrant_factory.config
            coros = []
            for row in rows:
                auction_id = str(row.get("auction_id", "")).strip()
                count = int(float(row.get("unique_bidder_count_4h") or 0))
                if not auction_id or count <= 0:
                    continue
                f = _qm.Filter(must=[_qm.FieldCondition(
                    key=cfg.payload_id_field,
                    match=_qm.MatchValue(value=auction_id),
                )])
                coros.append(qdrant_factory.client.set_payload(
                    collection_name=cfg.collection_name,
                    payload={"unique_bidder_count_4h": count},
                    points=f,
                    wait=False,
                ))
            if not coros:
                return 0
            results = await asyncio.gather(*coros, return_exceptions=True)
            enriched = sum(1 for r in results if not isinstance(r, Exception))
            errors = [r for r in results if isinstance(r, Exception)]
            if errors:
                logger.warning(f"event_ingest_qdrant_unique_bidder_partial enriched={enriched} errors={len(errors)}")
            logger.info(f"event_ingest_qdrant_unique_bidders enriched={enriched} latency_ms={latency_ms:.1f}")
            return enriched
        except Exception as e:
            logger.warning(f"event_ingest_qdrant_unique_bidder_failed error_type={type(e).__name__} error={e}")
            return 0

    async def _enrich_qdrant_net_watch_delta(self) -> int:
        """Read net watch delta (adds-removes today) from watch_events, set_payload on Qdrant."""
        if not self._enrich_enabled or self._ch_executor is None:
            return 0
        if _qm is None:
            return 0
        qdrant_factory = self._qdrant
        if qdrant_factory is None or not qdrant_factory.available or qdrant_factory.client is None:
            return 0
        try:
            sql = self._net_watch_delta_sql
            _res = await self._ch_executor.execute(sql)
            rows, latency_ms = _res.rows, _res.latency_ms
            if not rows:
                return 0
            cfg = qdrant_factory.config
            coros = []
            for row in rows:
                auction_id = str(row.get("auction_id", "")).strip()
                delta = int(float(row.get("net_watch_delta_1d") or 0))
                if not auction_id:
                    continue
                f = _qm.Filter(must=[_qm.FieldCondition(
                    key=cfg.payload_id_field,
                    match=_qm.MatchValue(value=auction_id),
                )])
                coros.append(qdrant_factory.client.set_payload(
                    collection_name=cfg.collection_name,
                    payload={"net_watch_delta_1d": delta},
                    points=f,
                    wait=False,
                ))
            if not coros:
                return 0
            results = await asyncio.gather(*coros, return_exceptions=True)
            enriched = sum(1 for r in results if not isinstance(r, Exception))
            errors = [r for r in results if isinstance(r, Exception)]
            if errors:
                logger.warning(f"event_ingest_qdrant_net_watch_delta_partial enriched={enriched} errors={len(errors)}")
            logger.info(f"event_ingest_qdrant_net_watch_delta enriched={enriched} latency_ms={latency_ms:.1f}")
            return enriched
        except Exception as e:
            logger.warning(f"event_ingest_qdrant_net_watch_delta_failed error_type={type(e).__name__} error={e}")
            return 0

    # ── State helpers ─────────────────────────────────────────────────────────

    def _record_cycle(
        self,
        source: str,
        started_at: float,
        duration: float,
        rows_fetched: int,
        rows_written: int,
        rows_skipped: int,
        qdrant_enriched: int,
        success: bool,
        error_str: Optional[str],
    ) -> None:
        self._cycles_total += 1
        if success:
            self._cycles_success += 1
        else:
            self._cycles_failed += 1
        summary = EventIngestCycleSummary(
            source=source,
            started_at=started_at,
            duration_seconds=duration,
            rows_fetched=rows_fetched,
            rows_written=rows_written,
            rows_skipped=rows_skipped,
            qdrant_enriched=qdrant_enriched,
            success=success,
            error=error_str,
        )
        self._history.append(summary)
        if len(self._history) > _MAX_HISTORY:
            del self._history[: len(self._history) - _MAX_HISTORY]
        logger.info(
            f"event_ingest_cycle_complete source={source}"
            f" rows_fetched={rows_fetched} rows_written={rows_written}"
            f" qdrant_enriched={qdrant_enriched}"
            f" duration_seconds={duration:.2f} success={success}"
        )

    def _update_failure_counter(self, which: str, success: bool) -> None:
        if which == "bid":
            if success:
                self._bid_consecutive_failures = 0
            else:
                self._bid_consecutive_failures += 1
                if self._bid_consecutive_failures >= self._bid_max_failures:
                    self._bid_paused = True
                    logger.error(
                        f"event_ingest_bid_paused"
                        f" consecutive_failures={self._bid_consecutive_failures}"
                    )
        else:
            if success:
                self._watch_consecutive_failures = 0
            else:
                self._watch_consecutive_failures += 1
                if self._watch_consecutive_failures >= self._watch_max_failures:
                    self._watch_paused = True
                    logger.error(
                        f"event_ingest_watch_paused"
                        f" consecutive_failures={self._watch_consecutive_failures}"
                    )

    def get_summary(self) -> EventIngestDriverSummary:
        """Snapshot of driver state for ops endpoints."""
        return EventIngestDriverSummary(
            bid_enabled=self._bid_enabled,
            watch_enabled=self._watch_enabled,
            running=self.running,
            paused_bid=self._bid_paused,
            paused_watch=self._watch_paused,
            bid_interval_seconds=self._bid_interval,
            watch_interval_seconds=self._watch_interval,
            bid_consecutive_failures=self._bid_consecutive_failures,
            watch_consecutive_failures=self._watch_consecutive_failures,
            cycles_total=self._cycles_total,
            cycles_success=self._cycles_success,
            cycles_failed=self._cycles_failed,
            last_bid_polled_at=self._bid_last_polled,
            last_watch_polled_at=self._watch_last_polled,
            history=list(self._history),
        )
