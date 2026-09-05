"""Athena-backed bid event source for ClickHouse ingest.

Queries ``the_resale_place.item_bids_cln`` (human bids stream) with a LEFT JOIN
to ``the_resale_place.item_winning_bids_cln`` to populate ``auction_id``.

Column mapping from Athena → analytics.bid_events (ClickHouse):
  - ``item_bid_id_num``        → ``bid_event_id``          (String)
  - ``member_item_id_num``     → ``member_item_id``         (Int64)
  - ``iwb.auction_id``         → ``auction_id``             (Int64; 0 when join misses)
  - ``bidder_member_id_num``   → ``bidder_id``              (Int64)
  - ``seller_member_id_num``   → ``seller_id``              (Int64)
  - ``bid_usd_amount``         → ``bid_usd_amount``         (Float64)
  - ``bid_source_txt``         → ``bid_source``             (String)
  - ``buy_it_now_flag``        → ``buy_it_now_flag``        (UInt8)
  - ``bid_accepted_flag``      → ``bid_accepted``           (UInt8)
  - ``counter_offer_flag``     → ``counter_offer``          (UInt8)
  - ``bid_start_date_utc_ts``  → ``bid_start_at``           (DateTime64)
  - ``bid_end_date_utc_ts``    → ``bid_end_at``             (Nullable DateTime64)
  - ``event_receive_utc_ts``   → ``event_utc_ts``           (DateTime64)

Partition pruning: uses src_receive_utc_year_num / month_num / day_num / hour_num.
Time window filter: event_receive_utc_ts >= since_ts AND < now_ts.
Human-bid filter: buy_it_now_flag = 0 (enforced in ClickHouse MV, not source query).

Layer rules: imports stdlib + core + nl_to_sql.athena_client. Never imports
registry, orchestrator, or retrieval code.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

from semantic_search.core.logging_utils import get_logger
from semantic_search.nl_to_sql.athena_client import AthenaClient

logger = get_logger(__name__)

# Athena source column → ClickHouse target column.
_COL_MAP: Dict[str, str] = {
    "item_bid_id_num": "bid_event_id",
    "member_item_id_num": "member_item_id",
    "auction_id": "auction_id",
    "bidder_member_id_num": "bidder_id",
    "seller_member_id_num": "seller_id",
    "bid_usd_amount": "bid_usd_amount",
    "bid_source_txt": "bid_source",
    "buy_it_now_flag": "buy_it_now_flag",
    "bid_accepted_flag": "bid_accepted",
    "counter_offer_flag": "counter_offer",
    "bid_start_date_utc_ts": "bid_start_at",
    "bid_end_date_utc_ts": "bid_end_at",
    "event_receive_utc_ts": "event_utc_ts",
}

# Dedup: one row per (member_item_id_num, bid_start_date_utc_ts), latest event wins.
_SQL_TEMPLATE = (
    "WITH ranked AS ("
    " SELECT"
    "  b.item_bid_id_num,"
    "  b.member_item_id_num,"
    "  COALESCE(iwb.auction_id_num, 0)    AS auction_id,"
    "  b.bidder_member_id_num,"
    "  b.seller_member_id_num,"
    "  b.bid_usd_amount,"
    "  b.bid_source_txt,"
    "  b.buy_it_now_flag,"
    "  b.bid_accepted_flag,"
    "  b.counter_offer_flag,"
    "  b.bid_start_date_utc_ts,"
    "  b.bid_end_date_utc_ts,"
    "  b.event_receive_utc_ts,"
    "  ROW_NUMBER() OVER ("
    "   PARTITION BY b.member_item_id_num, b.bid_start_date_utc_ts"
    "   ORDER BY b.src_receive_utc_year_num DESC,"
    "            b.src_receive_utc_month_num DESC,"
    "            b.src_receive_utc_day_num DESC,"
    "            b.src_receive_utc_hour_num DESC,"
    "            b.event_receive_utc_ts DESC"
    "  ) AS rn"
    " FROM {source_database}.{source_table} b"
    " LEFT JOIN {source_database}.{winning_bids_table} iwb"
    "   ON b.item_bid_id_num = iwb.item_bid_id_num"
    " WHERE ({partition_clauses})"
    " AND b.event_receive_utc_ts >= TIMESTAMP '{since_ts}'"
    " AND b.event_receive_utc_ts < TIMESTAMP '{now_ts}'"
    ")"
    " SELECT"
    "  item_bid_id_num, member_item_id_num, auction_id, bidder_member_id_num, seller_member_id_num,"
    "  bid_usd_amount, bid_source_txt, buy_it_now_flag, bid_accepted_flag, counter_offer_flag,"
    "  bid_start_date_utc_ts, bid_end_date_utc_ts, event_receive_utc_ts"
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
            f"(b.src_receive_utc_year_num = {c.year}"
            f" AND b.src_receive_utc_month_num = {c.month}"
            f" AND b.src_receive_utc_day_num = {c.day}"
            f" AND b.src_receive_utc_hour_num = {c.hour})"
        )
        current += timedelta(hours=1)
    return " OR ".join(clauses) if clauses else "1=1"


def _ts_to_str(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _row_to_ch_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    """Map one Athena result row to a ClickHouse analytics.bid_events insert dict."""
    def _int(v: Any, default: int = 0) -> int:
        try:
            return int(float(v)) if v is not None else default
        except (ValueError, TypeError):
            return default

    def _float(v: Any, default: float = 0.0) -> float:
        try:
            return float(v) if v is not None else default
        except (ValueError, TypeError):
            return default

    def _str(v: Any, default: str = "") -> str:
        return str(v).strip() if v is not None else default

    return {
        "bid_event_id":    _str(row.get("item_bid_id_num")),
        "member_item_id":  _int(row.get("member_item_id_num")),
        "auction_id":      _int(row.get("auction_id")),
        "bidder_id":       _int(row.get("bidder_member_id_num")),
        "seller_id":       _int(row.get("seller_member_id_num")),
        "bid_usd_amount":  _float(row.get("bid_usd_amount")),
        "bid_source":      _str(row.get("bid_source_txt")),
        "buy_it_now_flag": _int(row.get("buy_it_now_flag")),
        "bid_accepted":    _int(row.get("bid_accepted_flag")),
        "counter_offer":   _int(row.get("counter_offer_flag")),
        "bid_start_at":    _str(row.get("bid_start_date_utc_ts")),
        "bid_end_at":      _str(row.get("bid_end_date_utc_ts")) or None,
        "event_utc_ts":    _str(row.get("event_receive_utc_ts")),
    }


async def fetch_bid_events(
    athena_client: AthenaClient,
    source_database: str,
    source_table: str,
    winning_bids_table: str,
    since_ts: float,
    now_ts: float,
    batch_size: int,
    timeout_seconds: float,
) -> List[Dict[str, Any]]:
    """Query item_bids_cln for new bid events in the (since_ts, now_ts) window.

    :param athena_client: AthenaClient - Pre-built Athena client.
    :param source_database: str - Athena database name (``the_resale_place``).
    :param source_table: str - Bid events table (``item_bids_cln``).
    :param winning_bids_table: str - Join table for auction_id (``item_winning_bids_cln``).
    :param since_ts: float - Unix timestamp lower bound (inclusive).
    :param now_ts: float - Unix timestamp upper bound (exclusive).
    :param batch_size: int - LIMIT applied to the dedup CTE.
    :param timeout_seconds: float - Athena query wall-clock cap.
    :return: List of ClickHouse-ready row dicts for analytics.bid_events.
    :raises RuntimeError: When Athena credentials unavailable or query fails.
    """
    if not athena_client.credentials_available:
        raise RuntimeError("bid_event_source athena_client credentials unavailable")
    since_dt = datetime.fromtimestamp(since_ts, tz=timezone.utc)
    now_dt = datetime.fromtimestamp(now_ts, tz=timezone.utc)
    partition_clauses = _build_partition_clauses(since_dt, now_dt)
    sql = _SQL_TEMPLATE.format(
        source_database=source_database,
        source_table=source_table,
        winning_bids_table=winning_bids_table,
        partition_clauses=partition_clauses,
        since_ts=_ts_to_str(since_ts),
        now_ts=_ts_to_str(now_ts),
        batch_size=batch_size,
    )
    t0 = time.monotonic()
    rows, _cols, latency_ms = await athena_client.fetch_sql_async(sql, timeout_seconds)
    logger.info(
        f"bid_event_source_fetch rows={len(rows)}"
        f" since={_ts_to_str(since_ts)} now={_ts_to_str(now_ts)}"
        f" latency_ms={latency_ms:.1f}"
    )
    result: List[Dict[str, Any]] = []
    for row in rows:
        d = _row_to_ch_dict(row)
        if not d["bid_event_id"] or not d["member_item_id"]:
            continue
        result.append(d)
    return result
