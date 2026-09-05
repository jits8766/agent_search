"""Athena-backed delta source for real-time mutable-field refreshes.

Queries ``signals_platform_cln.auction_audit_cln`` (real-time event stream)
for rows newer than a given timestamp, deduplicates to one row per domain
using ROW_NUMBER(), and returns a list of payload-update dicts keyed by
auction_id (the ``item_id`` in Qdrant payloads).

Column mapping from auction_audit_cln to Qdrant payload field names:
  - ``auction_id``               -> used as ``item_id`` filter key
  - ``price_usd_amt``            -> ``price`` (ask; FIND aliases from config)
  - ``current_price_usd_amt``    -> ``current_bid_price``
  - ``bid_cnt``                  -> ``bid_count``
  - ``auction_type_id``          -> ``auction_type``
  - ``auction_end_utc_ts``       -> ``ends_at``
  - ``buy_it_now_flag``          -> ``buy_it_now``
  - ``buy_it_now_usd_amt``       -> ``buy_it_now_price``
  - ``feature_listing_flag``     -> ``is_featured``

Partition columns (Athena partition pruning - required for cost control):
  src_receive_utc_year_num / month_num / day_num / hour_num

event_receive_utc_ts is used as the time-window filter and sub-hour
dedup tiebreaker within each partition hour bucket.

Layer rules: imports stdlib + core + nl_to_sql.athena_client.
Never imports registry, orchestrator, or retrieval code.
"""

import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

from semantic_search.config.models import DeltaRefreshConfig
from semantic_search.core.logging_utils import get_logger
from semantic_search.nl_to_sql.athena_client import AthenaClient

logger = get_logger(__name__)

# Source column -> primary Qdrant payload field name (internal filter keys).
_COL_TO_FIELD: Dict[str, str] = {
    "price_usd_amt": "price",
    "current_price_usd_amt": "current_bid_price",
    "bid_cnt": "bid_count",
    "auction_type_id": "auction_type",
    "auction_end_utc_ts": "ends_at",
    "buy_it_now_flag": "buy_it_now",
    "buy_it_now_usd_amt": "buy_it_now_price",
    "feature_listing_flag": "is_featured",
}

# SQL template - partition clauses injected at call time; all variable parts
# (database, table, batch_size) come from DeltaRefreshConfig.
_SQL_TEMPLATE = (
    "WITH ranked AS ("
    " SELECT auction_id, domain_name, price_usd_amt, current_price_usd_amt, bid_cnt,"
    " auction_type_id, auction_end_utc_ts,"
    " buy_it_now_flag, buy_it_now_usd_amt, feature_listing_flag, event_receive_utc_ts,"
    " ROW_NUMBER() OVER (PARTITION BY domain_name ORDER BY src_receive_utc_year_num DESC,"
    " src_receive_utc_month_num DESC, src_receive_utc_day_num DESC, src_receive_utc_hour_num DESC,"
    " event_receive_utc_ts DESC) AS rn"
    " FROM {source_database}.{source_table}"
    " WHERE ({partition_clauses})"
    " AND event_receive_utc_ts >= TIMESTAMP '{since_ts}'"
    " AND event_receive_utc_ts < TIMESTAMP '{now_ts}'"
    " AND auction_type_id IN (16, 20, 38, 39)"
    ")"
    " SELECT auction_id, domain_name, price_usd_amt, current_price_usd_amt, bid_cnt,"
    " auction_type_id, auction_end_utc_ts,"
    " buy_it_now_flag, buy_it_now_usd_amt, feature_listing_flag"
    " FROM ranked WHERE rn = 1"
    " LIMIT {batch_size}"
)


def _build_partition_clauses(since_dt: datetime, now_dt: datetime) -> str:
    """Build OR-joined hour-level partition predicates for Athena partition pruning."""
    clauses: List[str] = []
    current = since_dt.replace(minute=0, second=0, microsecond=0)
    end = now_dt.replace(minute=0, second=0, microsecond=0)
    while current <= end:
        _c = current
        _clause = (
            f"(src_receive_utc_year_num = {_c.year} AND src_receive_utc_month_num = {_c.month} "
            f"AND src_receive_utc_day_num = {_c.day} AND src_receive_utc_hour_num = {_c.hour})"
        )
        clauses.append(_clause)
        current += timedelta(hours=1)
    return " OR ".join(clauses) if clauses else "1=1"


def _ts_to_str(ts: float) -> str:
    """Format a Unix timestamp as the Athena TIMESTAMP literal string."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _coerce_ends_at(raw: Any) -> Any:
    """Parse Athena TIMESTAMP string to Unix epoch float; return None on failure."""
    if raw is None or str(raw).strip() == "":
        return None
    try:
        s = str(raw).strip().replace(" ", "T")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return float(dt.timestamp())
    except (ValueError, OverflowError):
        return None


def _dual_write_find_aliases(
    updates: Dict[str, Any],
    payload_aliases: Dict[str, List[str]],
    bool_aliases: Dict[str, str],
) -> None:
    """Mirror internal mutable keys onto FIND listing names from config."""
    for src, targets in payload_aliases.items():
        if src not in updates:
            continue
        for target in targets:
            updates[target] = updates[src]
    for src, target in bool_aliases.items():
        if src not in updates:
            continue
        raw = updates[src]
        try:
            updates[target] = bool(int(raw))
        except (TypeError, ValueError):
            updates[target] = bool(raw)


def _row_to_update(
    row: Dict[str, Any], config: DeltaRefreshConfig
) -> Tuple[str, Dict[str, Any]]:
    """Convert one deduped auction_audit_cln row to (item_id, payload_updates).

    Only fields listed in mutable_fields are included in the update dict.
    FIND aliases are dual-written from config whenever the corresponding internal key updates.
    """
    item_id: str = str(row.get("auction_id", "")).strip()
    updates: Dict[str, Any] = {}
    mutable_fields = config.mutable_fields
    for src_col, field_name in _COL_TO_FIELD.items():
        if field_name not in mutable_fields:
            continue
        raw = row.get(src_col)
        if raw is None or str(raw).strip() == "":
            continue
        if field_name in ("price", "current_bid_price", "buy_it_now_price"):
            try:
                updates[field_name] = float(raw)
            except (ValueError, TypeError):
                pass
        elif field_name == "bid_count":
            try:
                updates[field_name] = int(float(raw))
            except (ValueError, TypeError):
                pass
        elif field_name == "auction_type":
            updates[field_name] = str(raw).strip()
        elif field_name == "ends_at":
            coerced = _coerce_ends_at(raw)
            if coerced is not None:
                updates[field_name] = coerced
        elif field_name in ("is_featured", "buy_it_now"):
            try:
                updates[field_name] = int(float(raw))
            except (ValueError, TypeError):
                pass
    _dual_write_find_aliases(
        updates, config.find_payload_aliases, config.find_bool_aliases
    )
    return item_id, updates


async def fetch_delta(
    athena_client: AthenaClient,
    config: DeltaRefreshConfig,
    since_ts: float,
    now_ts: float,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Query auction_audit_cln for mutable-field changes in (since_ts, now_ts).

    :param athena_client: AthenaClient - Pre-built Athena client.
    :param config: DeltaRefreshConfig - Source table / batch / timeout params.
    :param since_ts: float - Unix timestamp lower bound (exclusive).
    :param now_ts: float - Unix timestamp upper bound (exclusive).
    :return: List of (item_id, payload_updates) pairs - one per deduplicated domain.
    :raises RuntimeError: When Athena credentials unavailable or query fails.
    """
    if not athena_client.credentials_available:
        raise RuntimeError("delta_source athena_client credentials unavailable")
    since_dt = datetime.fromtimestamp(since_ts, tz=timezone.utc)
    now_dt = datetime.fromtimestamp(now_ts, tz=timezone.utc)
    partition_clauses = _build_partition_clauses(since_dt, now_dt)
    _db, _tbl, _pc, _sts, _nts, _bs = (
        config.source_database,
        config.source_table,
        partition_clauses,
        _ts_to_str(since_ts),
        _ts_to_str(now_ts),
        config.batch_size,
    )
    sql = _SQL_TEMPLATE.format(
        source_database=_db,
        source_table=_tbl,
        partition_clauses=_pc,
        since_ts=_sts,
        now_ts=_nts,
        batch_size=_bs,
    )
    t0 = time.monotonic()
    rows, _cols, latency_ms = await athena_client.fetch_sql_async(
        sql, config.timeout_seconds
    )
    out: List[Tuple[str, Dict[str, Any]]] = []
    for row in rows or []:
        item_id, updates = _row_to_update(row, config)
        if item_id and updates:
            out.append((item_id, updates))
    elapsed = (time.monotonic() - t0) * 1000.0
    logger.info(
        f"delta_source_fetch rows={len(rows or [])} updates={len(out)} "
        f"athena_ms={latency_ms:.1f} total_ms={elapsed:.1f}"
    )
    return out


__all__ = ["fetch_delta", "_row_to_update", "_COL_TO_FIELD"]
