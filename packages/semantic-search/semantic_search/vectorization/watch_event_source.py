"""Athena-backed watch event source for ClickHouse ingest.

Queries ``the_resale_place.member_items_watch_cln`` (watchlist activity stream)
with a LEFT JOIN to ``the_resale_place.member_items_watch_types_cln`` to resolve
``watch_type_label`` at ingest time.

Column mapping from Athena → analytics.watch_events (ClickHouse):
  - ``member_items_watch_id_num`` → ``watch_event_id``        (String)
  - ``member_item_id_num``        → ``member_item_id``         (Int64)
  - ``member_id_num``             → ``member_id``              (Int64)
  - ``watch_type_num``            → ``watch_type``             (Int32)
  - ``wt.watch_type_description`` → ``watch_type_label``   (String)
  - ``is_deleted_flag``           → ``is_deleted``             (UInt8)
  - ``create_date_utc_ts``        → ``created_at``             (DateTime64)
  - ``modified_date_utc_ts``      → ``modified_at``            (Nullable DateTime64)
  - ``event_receive_utc_ts``      → ``event_utc_ts``           (DateTime64)

Partition pruning: uses src_receive_utc_year_num / month_num / day_num / hour_num.
Time window filter: event_receive_utc_ts >= since_ts AND < now_ts.
Watch types 1 (saved search) and 9 (explicit watch) are the primary engagement signals;
all watch_type values are ingested — the ClickHouse MV filter (watch_type IN (1, 9))
handles the scope reduction for watch_density aggregation.

Layer rules: imports stdlib + core + nl_to_sql.athena_client. Never imports
registry, orchestrator, or retrieval code.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from semantic_search.core.logging_utils import get_logger
from semantic_search.nl_to_sql.athena_client import AthenaClient

logger = get_logger(__name__)

# Dedup: one row per (member_item_id_num, member_id_num), latest modification wins.
_SQL_TEMPLATE = (
    "WITH ranked AS ("
    " SELECT"
    "  w.member_items_watch_id_num,"
    "  w.member_item_id_num,"
    "  w.member_id_num,"
    "  w.watch_type_num,"
    "  COALESCE(wt.watch_type_description, '')  AS watch_type_label,"
    "  w.is_deleted_flag,"
    "  w.create_date_utc_ts,"
    "  w.modified_date_utc_ts,"
    "  w.event_receive_utc_ts,"
    "  ROW_NUMBER() OVER ("
    "   PARTITION BY w.member_item_id_num, w.member_id_num, w.watch_type_num"
    "   ORDER BY w.src_receive_utc_year_num DESC,"
    "            w.src_receive_utc_month_num DESC,"
    "            w.src_receive_utc_day_num DESC,"
    "            w.src_receive_utc_hour_num DESC,"
    "            w.event_receive_utc_ts DESC"
    "  ) AS rn"
    " FROM {source_database}.{source_table} w"
    " LEFT JOIN {source_database}.{watch_types_table} wt"
    "   ON w.watch_type_num = wt.member_items_watch_types_id_num"
    " WHERE ({partition_clauses})"
    " AND w.event_receive_utc_ts >= TIMESTAMP '{since_ts}'"
    " AND w.event_receive_utc_ts < TIMESTAMP '{now_ts}'"
    ")"
    " SELECT"
    "  member_items_watch_id_num, member_item_id_num, member_id_num, watch_type_num,"
    "  watch_type_label, is_deleted_flag, create_date_utc_ts, modified_date_utc_ts,"
    "  event_receive_utc_ts"
    " FROM ranked WHERE rn = 1"
    " LIMIT {batch_size}"
)


def _build_partition_clauses(since_dt: datetime, now_dt: datetime) -> str:
    """Build OR-joined hour-level partition predicates for Athena partition pruning."""
    clauses: List[str] = []
    current = since_dt.replace(minute=0, second=0, microsecond=0)
    end = now_dt.replace(minute=0, second=0, microsecond=0)
    while current <= end:
        c = current
        clauses.append(
            f"(w.src_receive_utc_year_num = {c.year}"
            f" AND w.src_receive_utc_month_num = {c.month}"
            f" AND w.src_receive_utc_day_num = {c.day}"
            f" AND w.src_receive_utc_hour_num = {c.hour})"
        )
        current += timedelta(hours=1)
    return " OR ".join(clauses) if clauses else "1=1"


def _ts_to_str(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _row_to_ch_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    """Map one Athena result row to a ClickHouse analytics.watch_events insert dict."""
    def _int(v: Any, default: int = 0) -> int:
        try:
            return int(float(v)) if v is not None else default
        except (ValueError, TypeError):
            return default

    def _str(v: Any, default: str = "") -> str:
        return str(v).strip() if v is not None else default

    return {
        "watch_event_id":   _str(row.get("member_items_watch_id_num")),
        "member_item_id":   _int(row.get("member_item_id_num")),
        "member_id":        _int(row.get("member_id_num")),
        "watch_type":       _int(row.get("watch_type_num")),
        "watch_type_label": _str(row.get("watch_type_label")),
        "is_deleted":       _int(row.get("is_deleted_flag")),
        "created_at":       _str(row.get("create_date_utc_ts")),
        "modified_at":      _str(row.get("modified_date_utc_ts")) or None,
        "event_utc_ts":     _str(row.get("event_receive_utc_ts")),
    }


async def fetch_watch_events(
    athena_client: AthenaClient,
    source_database: str,
    source_table: str,
    watch_types_table: str,
    since_ts: float,
    now_ts: float,
    batch_size: int,
    timeout_seconds: float,
) -> List[Dict[str, Any]]:
    """Query member_items_watch_cln for new watch events in the (since_ts, now_ts) window.

    :param athena_client: AthenaClient - Pre-built Athena client.
    :param source_database: str - Athena database name (``the_resale_place``).
    :param source_table: str - Watch events table (``member_items_watch_cln``).
    :param watch_types_table: str - Dimension table for labels (``member_items_watch_types_cln``).
    :param since_ts: float - Unix timestamp lower bound (inclusive).
    :param now_ts: float - Unix timestamp upper bound (exclusive).
    :param batch_size: int - LIMIT applied to the dedup CTE.
    :param timeout_seconds: float - Athena query wall-clock cap.
    :return: List of ClickHouse-ready row dicts for analytics.watch_events.
    :raises RuntimeError: When Athena credentials unavailable or query fails.
    """
    if not athena_client.credentials_available:
        raise RuntimeError("watch_event_source athena_client credentials unavailable")
    since_dt = datetime.fromtimestamp(since_ts, tz=timezone.utc)
    now_dt = datetime.fromtimestamp(now_ts, tz=timezone.utc)
    partition_clauses = _build_partition_clauses(since_dt, now_dt)
    sql = _SQL_TEMPLATE.format(
        source_database=source_database,
        source_table=source_table,
        watch_types_table=watch_types_table,
        partition_clauses=partition_clauses,
        since_ts=_ts_to_str(since_ts),
        now_ts=_ts_to_str(now_ts),
        batch_size=batch_size,
    )
    t0 = time.monotonic()
    rows, _cols, latency_ms = await athena_client.fetch_sql_async(sql, timeout_seconds)
    logger.info(
        f"watch_event_source_fetch rows={len(rows)}"
        f" since={_ts_to_str(since_ts)} now={_ts_to_str(now_ts)}"
        f" latency_ms={latency_ms:.1f}"
    )
    result: List[Dict[str, Any]] = []
    for row in rows:
        d = _row_to_ch_dict(row)
        if not d["watch_event_id"] or not d["member_item_id"]:
            continue
        result.append(d)
    return result
