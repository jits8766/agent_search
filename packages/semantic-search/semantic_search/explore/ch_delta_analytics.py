"""ClickHouse delta writer — refreshes mutable auction fields for records arriving in the delta stream.

Called from DeltaRefreshDriver after each Qdrant set_payload cycle. For each
(auction_id, updates) pair from delta_source, issues a single batch INSERT that
reads the existing row from signals_platform_cln.auction_audit_cln (preserving non-mutable
fields: tld, govalue_score, monthly_traffic) and overrides the four mutable
columns: current_price, bid_count, auction_type_id, ends_at.

Uses a single INSERT … SELECT FINAL with multiIf per auction_id so the entire
delta batch is one round-trip to ClickHouse. ReplacingMergeTree(updated_at)
deduplicates on background merge, keeping the row with the highest updated_at.

Auction_ids not yet present in signals_platform_cln.auction_audit_cln are silently skipped
(the SELECT FINAL returns zero rows for unknown ids; the analytics back-fill
endpoint handles initial population with the full historical window).
"""
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)

_DEFAULT_TARGET_TABLE = "signals_platform_cln.auction_audit_cln"
_DEFAULT_TIMEOUT_SECONDS: float = 30.0


@dataclass(frozen=True)
class ChDeltaWriteSummary:
    """Outcome of one write_delta_to_clickhouse call.

    :param rows_written: int - Auction_ids submitted to the INSERT (not confirmed written)
    :param errors: int - Number of INSERT calls that raised
    :param elapsed_ms: float - Wall-clock time for this call
    :param first_error: Optional[str] - First error message, if any
    """

    rows_written: int
    errors: int
    elapsed_ms: float
    first_error: Optional[str] = None


def _coerce_int(v: Any) -> Optional[int]:
    try:
        return int(str(v).strip())
    except (ValueError, TypeError):
        return None


def _coerce_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v) if v is not None else default
    except (ValueError, TypeError):
        return default


def _ends_at_to_ms(v: Any) -> int:
    """Convert Unix epoch float (seconds) to integer milliseconds for DateTime64(3)."""
    try:
        return int(float(v) * 1000) if v is not None else 0
    except (ValueError, TypeError):
        return 0


def _build_delta_sql(
    coerced: List[Tuple[int, float, int, int, int, float, int]],
    target_table: str,
) -> str:
    """Build a single batch INSERT SELECT that overrides mutable fields for all auctions in the delta batch.

    :param coerced: List[(auction_id, price, bid_count, auction_type_id, ends_at_ms, buy_it_now_price, is_featured)]
    :return: SQL string — one INSERT … SELECT FINAL with multiIf routing per auction_id

    multiIf(cond1, val1, cond2, val2, …, default) — ClickHouse variadic conditional.
    Non-mutable columns (tld, govalue_score, monthly_traffic) retain prior snap values.
    """

    def _multi_if(col_idx: int, default_col: str) -> str:
        parts = [f"snap.auction_id = {c[0]}, {c[col_idx]}" for c in coerced]
        return f"multiIf({', '.join(parts)}, snap.{default_col})"

    def _multi_if_dt64(col_idx: int, default_col: str) -> str:
        # Each literal arm is cast to DateTime64(3) to match the column type.
        # Raw UInt64 millisecond integers cause Code 386 (no common supertype
        # between UInt64 and DateTime64(3)) inside multiIf.
        parts = [f"snap.auction_id = {c[0]}, toDateTime64({c[col_idx] // 1000}, 3)" for c in coerced]
        return f"multiIf({', '.join(parts)}, snap.{default_col})"

    in_clause = ", ".join(str(c[0]) for c in coerced)
    at_expr = _multi_if(3, "auction_type_id")
    price_expr = _multi_if(1, "current_price")
    ends_expr = _multi_if_dt64(4, "ends_at")
    bc_expr = _multi_if(2, "bid_count")
    bin_price_expr = _multi_if(5, "buy_it_now_price")
    featured_expr = _multi_if(6, "is_featured")

    return f"""INSERT INTO {target_table}
(auction_id, domain_name, tld, auction_type_id, current_price,
 govalue_score, ends_at, bid_count, monthly_traffic,
 buy_it_now_price, is_featured, updated_at)
SELECT
snap.auction_id, snap.domain_name, snap.tld,
toUInt32({at_expr}) AS auction_type_id,
toFloat64({price_expr}) AS current_price,
snap.govalue_score,
{ends_expr} AS ends_at,
toUInt32({bc_expr}) AS bid_count,
snap.monthly_traffic,
toFloat64({bin_price_expr}) AS buy_it_now_price,
toUInt8({featured_expr}) AS is_featured,
now64(3) AS updated_at
FROM {target_table} AS snap FINAL
WHERE snap.auction_id IN ({in_clause})"""  # noqa: S608


async def write_delta_to_clickhouse(
    pairs: List[Tuple[str, Dict[str, Any]]],
    ch_executor: Any,
    target_table: str = _DEFAULT_TARGET_TABLE,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> ChDeltaWriteSummary:
    """Write mutable-field changes for a delta batch to the configured ClickHouse table.

    Reads existing rows (FINAL dedup), overrides the four mutable columns, and
    inserts the result rows so ReplacingMergeTree(updated_at) keeps the latest
    state per auction_id on next background merge.

    :param pairs: List[(auction_id_str, updates)] — same format as delta_source output.
        updates keys: price (float, optional), bid_count (int, optional),
        auction_type (str|int, optional), ends_at (float Unix seconds, optional).
    :param ch_executor: ClickHouseExecutor — shared analytics executor.
    :param target_table: str — fully-qualified ClickHouse table name (from config).
    :param timeout_seconds: float — wall-clock cap for the INSERT call.
    :return: ChDeltaWriteSummary
    """
    t0 = time.monotonic()
    if not pairs:
        return ChDeltaWriteSummary(rows_written=0, errors=0, elapsed_ms=0.0)

    coerced: List[Tuple[int, float, int, int, int, float, int]] = []
    for item_id, updates in pairs:
        aid = _coerce_int(item_id)
        if aid is None or aid <= 0:
            continue
        # CH column current_price stores listing ask (payload price / price_usd_amt).
        price = _coerce_float(updates.get("price"), 0.0)
        bc = max(0, int(_coerce_float(updates.get("bid_count"), 0.0)))
        at_id = max(0, int(_coerce_float(updates.get("auction_type"), 0.0)))
        ea_ms = _ends_at_to_ms(updates.get("ends_at"))
        bin_price = _coerce_float(updates.get("buy_it_now_price"), 0.0)
        featured = max(0, int(_coerce_float(updates.get("is_featured"), 0.0)))
        coerced.append((aid, price, bc, at_id, ea_ms, bin_price, featured))

    if not coerced:
        return ChDeltaWriteSummary(rows_written=0, errors=0, elapsed_ms=0.0)

    _BATCH_SIZE = 250
    first_error: Optional[str] = None
    rows_written = 0
    errors = 0
    for i in range(0, len(coerced), _BATCH_SIZE):
        batch = coerced[i : i + _BATCH_SIZE]
        sql = _build_delta_sql(batch, target_table)
        try:
            await ch_executor.execute_insert(sql, timeout_seconds=timeout_seconds)
            rows_written += len(batch)
        except (TimeoutError, OSError, RuntimeError, ValueError, TypeError) as exc:
            if first_error is None:
                first_error = f"{type(exc).__name__}: {str(exc)[:300]}"
            errors += 1
            logger.warning(
                f"ch_delta_write_failed auction_ids_count={len(batch)} "
                f"error_type={type(exc).__name__} error={str(exc)[:200]}"
            )

    elapsed_ms = (time.monotonic() - t0) * 1000.0
    logger.info(
        f"ch_delta_write_complete rows_written={rows_written} "
        f"errors={errors} elapsed_ms={elapsed_ms:.1f}"
    )
    return ChDeltaWriteSummary(
        rows_written=rows_written,
        errors=errors,
        elapsed_ms=elapsed_ms,
        first_error=first_error,
    )
