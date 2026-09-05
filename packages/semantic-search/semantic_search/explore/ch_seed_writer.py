"""Seed-time ClickHouse writer — inserts Athena seed docs into signals_platform_cln.auction_audit_cln.

Called from the /data-build/seed endpoint after docs are fetched from Athena
so the explore rails (trending, ending-soon), guidance market snapshot, and
analytics NL-to-SQL pipeline all have data without a separate ETL process.

Schema is created automatically on first run (CREATE DATABASE/TABLE IF NOT EXISTS)
so no separate init step is required.

DateTime64(3) note: ClickHouse scale=3 means milliseconds. All timestamp
fields (ends_at, updated_at) are stored as integer milliseconds since epoch.

The table uses ReplacingMergeTree(updated_at), so re-running the build
upserts rather than duplicating rows. Deduplication is eventually consistent
(ClickHouse background merge), which is acceptable for explore rails.
"""
import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone as _tz
from typing import Any, Dict, List, Optional

from semantic_search.analytics.ch_schema import get_ddl_statements as _get_ddl_statements
from semantic_search.core.exceptions import AgentSearchError, DataIngestInterruptedError
from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)

_DEFAULT_TARGET_TABLE = "signals_platform_cln.auction_audit_cln"
_DEFAULT_SNAPSHOT_TABLE = "signals_platform_cln.domain_snapshots"

# Proxy-backfill: project the freshly-seeded active-auction rows into the
# historical-snapshot table, keyed by auction end date. This is a PROXY for
# completed-sales history (snapshot_port.py treats it as "listings active during
# the period", not confirmed sales) — it lets past-tense queries ("sold last
# week", "median sale price this quarter") return data instead of an empty set.
# ReplacingMergeTree(snapped_at) on (snapshot_date, domain_name) dedupes on
# re-seed, so this is idempotent.
def _snapshot_backfill_sql(now_ms: int, target_table: str, snapshot_table: str) -> str:  # noqa: S608
    """Build the snapshot backfill INSERT scoped to rows written in this seed run.

    Filtering by updated_at = now_ms avoids a full-table scan of the source table,
    which can OOM ClickHouse when the table holds millions of historical rows.
    """
    return f"""\
INSERT INTO {snapshot_table}
    (snapshot_date, domain_name, tld, current_price, govalue_score, bid_count, auction_type_id, snapped_at)
SELECT
    toDate(ends_at)  AS snapshot_date,
    domain_name,
    tld,
    current_price,
    govalue_score,
    bid_count,
    auction_type_id,
    now64(3)         AS snapped_at
FROM {target_table}
WHERE domain_name != '' AND updated_at = {now_ms}"""

# Schema DDL is the canonical set from analytics/ch_schema.py — shared verbatim
# with scripts/init_clickhouse.py so the auto-seed path and the manual
# initializer can never provision divergent schemas. Callers pass
# ensure_schema=True/False (YAML once-per-job vs every page).
# CREATE ... IF NOT EXISTS makes repeated ensures idempotent.

_INSERT_COLS = (
    "auction_id",
    "domain_name",
    "tld",
    "auction_type_id",
    "current_price",
    "govalue_score",
    "ends_at",
    "bid_count",
    "monthly_traffic",
    "listed_at",
    "user_id",
    "buy_it_now_price",
    "is_featured",
    "gd_transfer",
    "on_sale_rate",
    "starting_bid",
    "category_id",
    "updated_at",
)


@dataclass(frozen=True)
class ChSeedWriteSummary:
    """Outcome of one ``insert_seed_to_clickhouse`` call.

    :param rows_attempted: int - Docs offered to the writer
    :param rows_written: int - Docs included in successful INSERT batches
    :param batches: int - Total INSERT round-trips issued
    :param errors: int - Batches that raised (rows in those batches not written)
    :param elapsed_ms: float - Wall-clock write time
    :param first_error: Optional[str] - First error message for surfacing in API response
    """
    rows_attempted: int
    rows_written: int
    batches: int
    errors: int
    elapsed_ms: float
    first_error: Optional[str] = None


def _to_ms(ts: float) -> int:
    """Convert Unix timestamp in seconds to integer milliseconds for DateTime64(3)."""
    return int(ts * 1000)


def _doc_to_ch_row(doc: Dict[str, Any], now_ms: int) -> Optional[Dict[str, Any]]:
    """Convert one seed doc to a ClickHouse row dict.

    Returns None when the doc has no domain_name (unindexable).
    Timestamp fields (ends_at, updated_at) are integer milliseconds since epoch
    to match the DateTime64(3) column scale.
    """
    domain = str(doc.get("domain_name") or "").strip()
    if not domain:
        return None

    def _int(v: Any, default: int = 0) -> int:
        try:
            return int(v) if v is not None else default
        except (ValueError, TypeError):
            return default

    def _float(v: Any, default: float = 0.0) -> float:
        try:
            return float(v) if v is not None else default
        except (ValueError, TypeError):
            return default

    auction_id = _int(doc.get("auction_id"), 0)
    tld = str(doc.get("tld") or "").strip().lower()
    auction_type_id = _int(doc.get("auction_type"), 0)
    # CH column current_price stores listing ask (payload price / price_usd_amt).
    current_price = _float(doc.get("price") if doc.get("price") is not None else doc.get("current_price"), 0.0)
    govalue_score = _float(doc.get("govalue_score"), 0.0)
    # ends_at in docs is float Unix seconds -> convert to integer milliseconds
    ends_at_s = _float(doc.get("ends_at"), 0.0)
    ends_at_ms = _to_ms(ends_at_s) if ends_at_s > 0.0 else 0
    bid_count = _int(doc.get("bid_count"), 0)
    monthly_traffic = _int(doc.get("monthly_traffic"), 0)
    # listed_at: auction_list_utc_ts stored as string "listed_at" in seed doc
    listed_at_s = _float(doc.get("listed_at_epoch") or 0.0, 0.0)
    if listed_at_s <= 0.0:
        # Parse ISO string form stored by db_seed_source as "listed_at"
        _listed_str = str(doc.get("listed_at") or "").strip()
        if _listed_str:
            for _fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
                try:
                    _dt = datetime.strptime(_listed_str, _fmt).replace(tzinfo=_tz.utc)
                    listed_at_s = _dt.timestamp()
                    break
                except ValueError:
                    continue
    listed_at_ms = _to_ms(listed_at_s) if listed_at_s > 0.0 else 0
    user_id = _int(doc.get("member_id"), 0)
    buy_it_now_price = _float(doc.get("buy_it_now_price") or 0.0, 0.0)
    is_featured = _int(doc.get("is_featured"), 0)
    gd_transfer = _int(doc.get("gd_transfer"), 0)
    on_sale_rate = _float(doc.get("on_sale_rate") or 0.0, 0.0)
    starting_bid = _float(doc.get("starting_bid") or 0.0, 0.0)
    category_id = _int(doc.get("category_id"), 0)

    return {
        "auction_id": auction_id,
        "domain_name": domain,
        "tld": tld,
        "auction_type_id": auction_type_id,
        "current_price": current_price,
        "govalue_score": govalue_score,
        "ends_at": ends_at_ms,
        "bid_count": bid_count,
        "monthly_traffic": monthly_traffic,
        "listed_at": listed_at_ms,
        "user_id": user_id,
        "buy_it_now_price": buy_it_now_price,
        "is_featured": is_featured,
        "gd_transfer": gd_transfer,
        "on_sale_rate": on_sale_rate,
        "starting_bid": starting_bid,
        "category_id": category_id,
        "updated_at": now_ms,
    }


def _build_insert_payload(rows: List[Dict[str, Any]], target_table: str) -> str:
    """Build the full INSERT … FORMAT JSONEachRow payload for one batch."""
    header = f"INSERT INTO {target_table} ({', '.join(_INSERT_COLS)}) FORMAT JSONEachRow"
    lines = [header]
    for row in rows:
        lines.append(json.dumps(row, separators=(',', ':')))
    return "\n".join(lines)


async def _ensure_schema(
    ch_executor: Any, source_table: str, *, timeout_seconds: float
) -> Optional[str]:
    """Run CREATE DATABASE / TABLE / MV IF NOT EXISTS DDL statements.

    Returns None on success, or an error string on the first failure.
    DDL failures are non-fatal for the INSERT phase — the caller still attempts
    inserts so that an already-initialised schema is not blocked by MV failures.

    :param timeout_seconds: float - Per-DDL HTTP timeout from caller config
    """
    if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool) or float(timeout_seconds) <= 0.0:
        raise AgentSearchError("ch_seed_writer._ensure_schema timeout_seconds must be a number > 0")
    first_error: Optional[str] = None
    for stmt in _get_ddl_statements(source_table):
        try:
            await ch_executor.execute_insert(stmt.strip(), timeout_seconds=float(timeout_seconds))
        except (RuntimeError, OSError, ValueError, TypeError, AgentSearchError) as exc:
            err = f"{type(exc).__name__}: {str(exc)[:300]}"
            logger.warning(f"ch_seed_ddl_failed stmt_prefix={stmt[:60]!r} error={err}")
            if first_error is None:
                first_error = err
    if first_error is None:
        logger.info("ch_seed_schema_ensured")
    return first_error


async def insert_seed_to_clickhouse(
    docs: List[Dict[str, Any]],
    ch_executor: Any,
    *,
    ensure_schema: bool,
    batch_size: int,
    insert_timeout_seconds: float,
    schema_timeout_seconds: float,
    target_table: str,
    snapshot_table: str,
) -> ChSeedWriteSummary:
    """Optionally ensure schema, then insert seed docs into ClickHouse.

    When ``ensure_schema`` is True, runs CREATE IF NOT EXISTS DDL first.
    Callers that set ``clickhouse_ensure_schema_once`` / ``ensure_schema_once``
    pass True only on the first page of a job.

    Skips docs without a domain_name. Inserts in batches of batch_size.
    A batch failure or snapshot-backfill failure raises DataIngestInterruptedError,
    aborting remaining batches — callers must stop the data build until fixed.

    :param docs: List[Dict] - Seed payload dicts from db_seed_source
    :param ch_executor: ClickHouseExecutor - Shared analytics executor
    :param ensure_schema: bool - Whether to run DDL ensure for this call
    :param batch_size: int - Rows per INSERT round-trip
    :param insert_timeout_seconds: float - Per-batch HTTP timeout
    :param schema_timeout_seconds: float - Per-DDL statement timeout
    :param target_table: str - Fully-qualified ClickHouse target table
    :param snapshot_table: str - Fully-qualified ClickHouse snapshot table
    :return: ChSeedWriteSummary
    """
    if not isinstance(ensure_schema, bool):
        raise AgentSearchError("insert_seed_to_clickhouse ensure_schema must be a bool")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise AgentSearchError("insert_seed_to_clickhouse batch_size must be int >= 1")
    if not isinstance(insert_timeout_seconds, (int, float)) or isinstance(insert_timeout_seconds, bool) or float(insert_timeout_seconds) <= 0.0:
        raise AgentSearchError("insert_seed_to_clickhouse insert_timeout_seconds must be a number > 0")
    if not isinstance(schema_timeout_seconds, (int, float)) or isinstance(schema_timeout_seconds, bool) or float(schema_timeout_seconds) <= 0.0:
        raise AgentSearchError("insert_seed_to_clickhouse schema_timeout_seconds must be a number > 0")
    if not isinstance(target_table, str) or not target_table.strip():
        raise AgentSearchError("insert_seed_to_clickhouse target_table must be a non-empty string")
    if not isinstance(snapshot_table, str) or not snapshot_table.strip():
        raise AgentSearchError("insert_seed_to_clickhouse snapshot_table must be a non-empty string")

    t0 = time.monotonic()
    now_ms = _to_ms(time.time())
    schema_error: Optional[str] = None

    if ensure_schema:
        try:
            schema_error = await _ensure_schema(
                ch_executor,
                source_table=target_table,
                timeout_seconds=float(schema_timeout_seconds),
            )
        except asyncio.CancelledError as cancel_exc:
            raise DataIngestInterruptedError(
                'ClickHouse seed interrupted during schema ensure (0 rows written). '
                'Often ALB idle timeout or client disconnect.',
                stage='clickhouse_schema',
                records_completed=0,
                records_attempted=len(docs),
                reason='cancelled',
                detail=type(cancel_exc).__name__,
            ) from cancel_exc

    ch_rows: List[Dict[str, Any]] = []
    for doc in docs:
        row = _doc_to_ch_row(doc, now_ms)
        if row is not None:
            ch_rows.append(row)

    rows_attempted = len(docs)
    rows_written = 0
    batches_ok = 0
    batches_err = 0
    first_error: Optional[str] = schema_error  # surface schema error if DDL failed
    insert_timeout = float(insert_timeout_seconds)

    for start in range(0, len(ch_rows), batch_size):
        batch = ch_rows[start:start + batch_size]
        if not batch:
            continue
        sql = _build_insert_payload(batch, target_table)
        try:
            await ch_executor.execute_insert(sql, timeout_seconds=insert_timeout)
            rows_written += len(batch)
            batches_ok += 1
        except asyncio.CancelledError as cancel_exc:
            raise DataIngestInterruptedError(
                (
                    f"ClickHouse seed interrupted after {rows_written} of {len(ch_rows)} rows "
                    f"(batch_start={start}). Often ALB idle timeout or client disconnect."
                ),
                stage='clickhouse_seed',
                records_completed=rows_written,
                records_attempted=len(ch_rows),
                reason='cancelled',
                detail=f"{type(cancel_exc).__name__}: batch_start={start}",
            ) from cancel_exc
        except (RuntimeError, OSError, ValueError, TypeError) as exc:
            err_msg = f"{type(exc).__name__}: {str(exc)[:300]}"
            logger.error(f"ch_seed_insert_batch_failed batch_start={start} batch_size={len(batch)} error={err_msg}")
            raise DataIngestInterruptedError(
                (
                    f"ClickHouse seed insert failed after {rows_written} of {len(ch_rows)} rows "
                    f"(batch_start={start}): {err_msg}"
                ),
                stage='clickhouse_seed',
                records_completed=rows_written,
                records_attempted=len(ch_rows),
                reason='error',
                detail=err_msg,
            ) from exc

    # Proxy-backfill the historical-snapshot table from the rows just written.
    if rows_written > 0:
        try:
            await ch_executor.execute_insert(
                _snapshot_backfill_sql(now_ms, target_table, snapshot_table),
                timeout_seconds=insert_timeout,
            )
            logger.info(f"ch_seed_snapshot_backfill_ok source_rows={rows_written}")
        except (RuntimeError, OSError, ValueError, TypeError, AgentSearchError) as exc:
            err_msg = f"{type(exc).__name__}: {str(exc)[:300]}"
            logger.error(f"ch_seed_snapshot_backfill_failed error={err_msg}")
            raise DataIngestInterruptedError(
                f"ClickHouse snapshot backfill failed after {rows_written} rows written: {err_msg}",
                stage='clickhouse_snapshot_backfill',
                records_completed=rows_written,
                records_attempted=len(ch_rows),
                reason='error',
                detail=err_msg,
            ) from exc

    elapsed_ms = (time.monotonic() - t0) * 1000.0
    logger.info(f"ch_seed_write_complete rows_attempted={rows_attempted} rows_written={rows_written} batches_ok={batches_ok} batches_err={batches_err} elapsed_ms={elapsed_ms:.1f} ensure_schema={ensure_schema}")
    return ChSeedWriteSummary(rows_attempted=rows_attempted, rows_written=rows_written, batches=batches_ok + batches_err, errors=batches_err, elapsed_ms=elapsed_ms, first_error=first_error)


__all__ = ['insert_seed_to_clickhouse', 'ChSeedWriteSummary']
