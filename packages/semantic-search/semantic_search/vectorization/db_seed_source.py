"""Database-backed seed source for the data-build seed endpoint.

Fetches real-time auction domain records from the database configured at
``vectorization.seed.database.name`` (table ``auction_audit_cln``).  Two fetch
strategies are supported:

``datewise``
    Pull records where ``auction_start_utc_ts`` falls within the last
    ``lookback_days`` days (``auction_start_utc_ts >= NOW() - N DAY``), capped at
    ``max_records``.  When ``active_only=True`` the query additionally requires
    ``auction_end_utc_ts > CURRENT_TIMESTAMP`` so only live, not-yet-closed auctions
    enter the Qdrant index.

``count``
    Pull the most recent ``max_records`` rows ordered by
    ``auction_start_utc_ts DESC``.  ``active_only=True`` adds the same
    ``auction_end_utc_ts > CURRENT_TIMESTAMP`` guard.

``auction_audit_cln`` is an event log - each auction_id may appear multiple times.
Queries use a ROW_NUMBER() CTE to deduplicate to the latest event per auction_id,
ordered by ``src_receive_utc_year_num / month_num / day_num / hour_num`` DESC.

Column names from ``signals_platform_cln.auction_audit_cln``:
- ``domain_name``              - string; full domain (e.g. "example.com")
- ``tld``                      - string; normalised with LOWER()
- ``auction_id``               - int; primary key
- ``auction_type_id``          - int; 16 = GoDaddy AutoExtend, 20 = GoDaddy BuyNow, 38 = Partner AutoExtend, 39 = Partner Closeout
- ``price_usd_amt``            - decimal; listing/ask price (FIND ``auction_price``).
  Prefer this for ``minPrice`` / payload ``price``. When ``bid_cnt=0`` this is
  start/BIN; when bids exist it tracks the high bid (validated Jul 2026).
- ``current_price_usd_amt``    - decimal; current bid amount (FIND ``current_bid_price``).
  ``0`` when no bids - do NOT use as ask/filter price.
- ``valuation_usd_amt``        - decimal; GoDaddy valuation (GoValue); maps to ``govalue_score`` / FIND
  ``valuation_price`` / ``appraised_value`` (dual-write aliases, see ``find_payload_aliases``)
- ``auction_start_utc_ts``     - timestamp; when bidding opens (datewise strategy filter)
- ``auction_end_utc_ts``       - timestamp; when bidding closes (``end_time`` / ``time_remaining_max``)
- ``bid_cnt``                  - int; bid count (FIND ``bids``)
- ``monthly_traffic_cnt``      - int; monthly visitor/pageview estimate
- ``domain_create_utc_dt``     - string; domain registration date; used to compute ``domain_age_years``
- ``last_14day_traffic_cnt``   - int; 14-day rolling traffic estimate (bonus signal)
- ``valuation_rank``           - int; relative valuation rank among active auctions (bonus signal)
- ``parking_revenue_usd_amt``  - decimal; parking revenue estimate (bonus signal)

Bid-offer time, Majestic metrics, unique-search-count, estibot, and
aftermarket-boost enrichment are all joined Athena-side by
``vectorization.seed_merge`` (one merged+paged final table per configured
source table) rather than fetched here via separate round-trips; the
COLUMN_NOT_FOUND fallback retry for those enrichment columns also lives in
``seed_merge.py``. This module builds each table's base seed query
(``_build_query``) for ``seed_merge`` to join against, then reads back and
maps the merged final table's rows to indexer payload docs.

Character fields (computed from domain_name, no SQL column required):
- ``sld``        - second-level domain (domain without TLD suffix)
- ``has_hyphen`` - 1 if SLD contains a hyphen, 0 otherwise
- ``has_number`` - 1 if SLD contains a digit, 0 otherwise
- ``is_idn``     - 1 if SLD is IDN: punycode (``xn--`` prefix) OR contains non-ASCII characters

Time fields (computed from auction_end_utc_ts):
- ``ends_at``             - Unix epoch float (for filter compatibility with time_remaining_max)
- ``time_to_end_seconds`` - seconds until auction ends (0 when already ended; None when unparseable)
- ``time_to_end_minutes`` - minutes until auction ends (derived; None when unparseable)
- ``time_to_end_hours``   - hours until auction ends (derived; None when unparseable)
- ``time_to_end_days``    - days until auction ends (derived; None when unparseable)

Only ``domain_name`` is vectorized; all other columns are structured payload
for filter retrieval (plan_agentic_search.md §3.3).

Layer rules: imports stdlib + ``core`` only (plus ``vectorization.seed_merge``
for the Athena-side enrichment merge). Never imports registry, orchestrator,
or retrieval code.
"""
import contextlib
import json
import math as _math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from semantic_search.core.exceptions import DataIngestInterruptedError
from semantic_search.core.logging_utils import get_logger
from semantic_search.vectorization import seed_merge

logger = get_logger(__name__)

_STRATEGY_DATEWISE = "datewise"
_STRATEGY_COUNT = "count"

# Athena timestamp formats returned by auction_end_utc_ts. Tried in order; first
# match wins. Plain float strings (epoch) are also accepted as a fallback.
_END_TIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
)

# Dedup CTE base for both strategies. auction_audit_cln is an event log;
# ROW_NUMBER() picks the latest event per auction_id via partition columns.
# Enrichment columns use bare CAST (no COALESCE default) so NULL source values
# are preserved as None in Python - filters treat None as "data unavailable"
# rather than zero-value matches. When a column is absent, seed_merge's
# COLUMN_NOT_FOUND fallback retries the stg_base CTAS with the CAST expression
# patched to NULL (see vectorization.seed_merge._patch_query).
_SELECT = """\
WITH deduped AS (
    SELECT
        CAST(auction_id AS VARCHAR)                                          AS auction_id,
        LOWER(domain_name)                                                   AS domain_name,
        LOWER(COALESCE(
            NULLIF(TRIM(tld), ''),
            SUBSTR(domain_name, POSITION('.' IN domain_name) + 1)
        ))                                                                   AS tld,
        CAST(auction_type_id AS VARCHAR)                                     AS auction_type,
        CAST(price_usd_amt AS DOUBLE)                                        AS price,
        CAST(current_price_usd_amt AS DOUBLE)                                AS current_bid_price,
        CAST(valuation_usd_amt AS DOUBLE)                                    AS govalue_score,
        CAST(auction_end_utc_ts AS VARCHAR)                                  AS end_time,
        CAST(bid_cnt AS INTEGER)                                             AS bid_count,
        CAST(domain_create_utc_dt AS VARCHAR)                                AS domain_create_date,
        CAST(monthly_traffic_cnt AS INTEGER)                                 AS monthly_traffic,
        CAST(last_14day_traffic_cnt AS INTEGER)                              AS last_14day_traffic,
        CAST(valuation_rank AS INTEGER)                                      AS valuation_rank,
        CAST(parking_revenue_usd_amt AS DOUBLE)                              AS parking_revenue,
        CAST(include_in_search_result_flag AS INTEGER)                        AS search_eligible,
        CAST(hide_flag AS INTEGER)                                            AS hidden,
        CAST(buy_it_now_flag AS INTEGER)                                      AS buy_it_now,
        CAST(buy_it_now_usd_amt AS DOUBLE)                                    AS buy_it_now_price,
        CAST(reserve_price_flag AS INTEGER)                                   AS has_reserve_price,
        CAST(reserve_price_usd_amt AS DOUBLE)                                 AS reserved_price_amount,
        CAST(feature_listing_flag AS INTEGER)                                 AS is_featured,
        CAST(gd_transfer_flag AS INTEGER)                                     AS gd_transfer,
        CAST(adult_listing_flag AS INTEGER)                                   AS adult_listing,
        CAST(status_code_id AS INTEGER)                                       AS status_code_id,
        CAST(category_id AS BIGINT)                                           AS category_id,
        CAST(auction_list_utc_ts AS VARCHAR)                                  AS listed_at,
        CAST(member_id AS INTEGER)                                            AS member_id,
        CAST(on_sale_rate AS DOUBLE)                                          AS on_sale_rate,
        CAST(starting_bid_usd_amt AS DOUBLE)                                  AS starting_bid,
        CAST(traffic_cnt AS INTEGER)                                          AS traffic_cnt,
        CAST(website_include_flag AS INTEGER)                                 AS website_include,
        CAST(item_description AS VARCHAR)                                     AS item_description,
        CAST(vendor_id AS INTEGER)                                            AS vendor_id,
        CAST(domain_extension_id AS INTEGER)                                  AS domain_extension_id,
        CAST(highest_bidder_id AS INTEGER)                                    AS highest_bidder_id,
        CAST(bid_accept_flag AS INTEGER)                                      AS bid_accepted_flag,
        CAST(display_in_category_listing_flag AS INTEGER)                     AS display_in_category,
        CAST(subcategory_feature_listing_flag AS INTEGER)                     AS sub_category_featured,
        CAST(additional_category_listing_flag AS INTEGER)                     AS add_i_category,
        CAST(update_utc_ts AS VARCHAR)                                        AS data_update_time,
        auction_start_utc_ts,
        auction_end_utc_ts,
        ROW_NUMBER() OVER (
            PARTITION BY auction_id
            ORDER BY src_receive_utc_year_num DESC, src_receive_utc_month_num DESC,
                     src_receive_utc_day_num DESC, src_receive_utc_hour_num DESC
        )                                                                    AS _rn
    FROM {database}.{table}
    WHERE auction_type_id IN (16, 20, 38, 39)
      AND domain_name IS NOT NULL
      AND LENGTH(TRIM(domain_name)) > 0
      AND ({partition_clauses})
)
SELECT auction_id, domain_name, tld, auction_type, price, current_bid_price,
       govalue_score, end_time,
       bid_count, domain_create_date, monthly_traffic, last_14day_traffic,
       valuation_rank, parking_revenue,
       search_eligible, hidden,
       buy_it_now, buy_it_now_price, has_reserve_price, reserved_price_amount,
       is_featured, gd_transfer,
       adult_listing, status_code_id, category_id, listed_at, member_id,
       on_sale_rate, starting_bid, traffic_cnt,
       website_include, item_description, vendor_id, domain_extension_id,
       highest_bidder_id, bid_accepted_flag,
       display_in_category, sub_category_featured, add_i_category, data_update_time
FROM deduped WHERE _rn = 1"""

_SQL_DATEWISE = (
    _SELECT.replace(
        "AND ({partition_clauses})",
        "AND ({partition_clauses})\n      AND auction_start_utc_ts >= CURRENT_TIMESTAMP - INTERVAL '{lookback_days}' DAY",
    )
    + "\nORDER BY auction_start_utc_ts DESC\nLIMIT {max_records}\n"
)

_SQL_COUNT = (
    _SELECT
    + "\nORDER BY auction_start_utc_ts DESC\nLIMIT {max_records}\n"
)

_SQL_DATEWISE_ACTIVE = (
    _SELECT.replace(
        "AND ({partition_clauses})",
        "AND ({partition_clauses})\n      AND auction_start_utc_ts >= CURRENT_TIMESTAMP - INTERVAL '{lookback_days}' DAY\n      AND auction_end_utc_ts > CURRENT_TIMESTAMP",
    )
    + "\nORDER BY auction_start_utc_ts DESC\nLIMIT {max_records}\n"
)

_SQL_COUNT_ACTIVE = (
    _SELECT.replace(
        "AND ({partition_clauses})",
        "AND ({partition_clauses})\n      AND auction_end_utc_ts > CURRENT_TIMESTAMP",
    )
    + "\nORDER BY auction_start_utc_ts DESC\nLIMIT {max_records}\n"
)


@dataclass(frozen=True)
class DbSeedSummary:
    """Outcome of one ``fetch_seed_from_db`` invocation.

    :param source: str - ``"daily_snapshot"`` or ``"realtime"``
    :param database: str - Database name queried
    :param documents: int - Total domain documents returned across all tables
    :param tables_queried: int - Tables successfully queried
    :param tables_skipped: int - Tables skipped (credentials unavailable / error)
    :param elapsed_ms: float - Wall-clock fetch time
    :param missing_columns: List[str] - Enrichment columns absent from Athena;
        payload fields for these columns are set to None (not zero) in every
        indexed document so downstream filters skip them correctly.
        Empty list when all columns were present.
    """
    source: str
    database: str
    documents: int
    tables_queried: int
    tables_skipped: int
    elapsed_ms: float
    missing_columns: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'missing_columns', list(self.missing_columns or []))


# Common date formats for domain_create_utc_dt (Athena string column).
_CREATE_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%m/%d/%Y",
    "%d-%b-%Y",
)


def _compute_domain_age_years(create_date_str: str) -> Optional[int]:
    """Parse domain_create_utc_dt string and return domain age in whole years.

    Tries common Athena string formats in order; returns None when the string
    is empty, None, or matches no known format. Age is floored (a domain
    registered 2.9 years ago returns 2).

    :param create_date_str: str - Raw domain_create_utc_dt value from Athena
    :return: Optional[int] - Age in years, or None
    """
    if not create_date_str or not create_date_str.strip():
        return None
    raw = create_date_str.strip()
    for fmt in _CREATE_DATE_FORMATS:
        try:
            dt = datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            return max(0, int((now - dt).days // 365))
        except ValueError:
            continue
    return None


def _build_seed_partition_clauses(lookback_days: int) -> str:
    """Build OR-joined day-level partition predicates for Athena partition pruning.

    Covers every calendar day from ``lookback_days + 1`` days ago through today
    (UTC) so the seed query scans only the relevant partitions.

    :param lookback_days: int - Number of days of history to include
    :return: str - OR-joined partition clause string, or "1=1" as safe fallback
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days + 1)
    clauses = []
    day = cutoff.date()
    today = datetime.now(timezone.utc).date()
    while day <= today:
        clauses.append(
            f"(src_receive_utc_year_num = {day.year} AND "
            f"src_receive_utc_month_num = {day.month} AND "
            f"src_receive_utc_day_num = {day.day})"
        )
        day += timedelta(days=1)
    return " OR ".join(clauses) if clauses else "1=1"


def _build_query(database: str, table_cfg: Any) -> str:
    """Build the SQL query for one table config entry.

    Both strategies apply ``LIMIT {max_records}`` so the result set is always
    bounded regardless of which strategy is chosen.  When ``table_cfg.active_only``
    is True, the selected template adds ``AND auction_end_utc_ts > CURRENT_TIMESTAMP``
    so only live, not-yet-closed auctions are returned.

    :param database: str - Resolved database name
    :param table_cfg: SeedTableConfig - Per-table configuration
    :return: str - Rendered SQL string ready for AthenaClient.fetch_sql_async
    """
    partition_clauses = _build_seed_partition_clauses(table_cfg.lookback_days)
    if table_cfg.strategy == _STRATEGY_COUNT:
        tmpl = _SQL_COUNT_ACTIVE if table_cfg.active_only else _SQL_COUNT
        return tmpl.format(database=database, table=table_cfg.table_name,
                           max_records=table_cfg.max_records,
                           partition_clauses=partition_clauses)
    tmpl = _SQL_DATEWISE_ACTIVE if table_cfg.active_only else _SQL_DATEWISE
    return tmpl.format(database=database, table=table_cfg.table_name,
                       lookback_days=table_cfg.lookback_days,
                       max_records=table_cfg.max_records,
                       partition_clauses=partition_clauses)


def _parse_end_time(end_time_str: Optional[str]) -> Optional[float]:
    """Parse an Athena timestamp string to a Unix epoch float.

    Tries the common Athena timestamp formats in order; falls back to treating
    the string as a plain float (epoch seconds).  Returns None when the string
    is empty or matches no known format so that time_to_end_* fields remain
    None for items whose end time cannot be determined.

    :param end_time_str: Optional[str] - Raw value of the end_time column
    :return: Optional[float] - Unix epoch seconds, or None
    """
    if not end_time_str or not end_time_str.strip():
        return None
    raw = end_time_str.strip()
    for fmt in _END_TIME_FORMATS:
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None


def _safe_optional_int(value: Any) -> Optional[int]:
    """Coerce to int; return None when value is None or unconvertible.

    :param value: Any - Value to coerce
    :return: Optional[int]
    """
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _safe_optional_float(value: Any) -> Optional[float]:
    """Coerce to float; return None when value is None, unconvertible, or NaN.

    :param value: Any - Value to coerce
    :return: Optional[float]
    """
    if value is None:
        return None
    try:
        f = float(value)
        return None if f != f else f  # NaN -> None
    except (ValueError, TypeError):
        return None


def _safe_optional_json_list(value: Any) -> Optional[List[Any]]:
    """Parse an Athena ``CAST(... AS JSON)`` cell into a Python list.

    :param value: Any - Raw CSV cell (JSON text, None, or empty string)
    :return: Optional[List[Any]] - Parsed list, or None when value is
        None/empty/not valid JSON (never raises)
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, list) else None


def _parse_update_time_iso(raw: Any) -> Optional[str]:
    """Normalize Athena update_utc_ts to ISO-8601 UTC string (None if absent)."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    unix = _parse_end_time(s)
    if unix is None:
        return s
    return datetime.fromtimestamp(unix, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _apply_find_aliases(
    doc: Dict[str, Any],
    payload_aliases: Dict[str, List[str]],
    bool_aliases: Dict[str, str],
) -> None:
    """Dual-write FIND listing keys from config alias maps (in-place)."""
    for src, targets in payload_aliases.items():
        if src not in doc:
            continue
        for target in targets:
            doc[target] = doc[src]
    for src, target in bool_aliases.items():
        if src not in doc:
            continue
        raw = doc[src]
        if raw is None:
            doc[target] = None
        else:
            try:
                doc[target] = bool(int(raw))
            except (TypeError, ValueError):
                doc[target] = bool(raw)


def _compute_traffic_features(
    monthly_traffic: Optional[int],
    last_14day_traffic: Optional[int],
    traffic_cnt: Optional[int],
    parking_revenue: Optional[float],
    semrush_search_volume: Optional[int],
    semrush_authority_score: Optional[float],
    semrush_indexed_pages: Optional[int],
    majestic_tf: Optional[int],
) -> Dict[str, Any]:
    """Derive traffic proxy features from all available signals.

    Returns traffic_proxy_score (float), has_web_traffic_signal (int 0/1),
    estimated_traffic_tier (int 0-4).
    """
    mt = monthly_traffic or 0
    l14 = last_14day_traffic or 0
    tc = traffic_cnt or 0
    pr = parking_revenue or 0.0
    sv = semrush_search_volume or 0
    sa = semrush_authority_score or 0.0
    ip = semrush_indexed_pages or 0
    mj = majestic_tf or 0

    # Best available direct traffic estimate (prefer monthly, then 14-day annualised, then generic)
    best_traffic = mt or int(l14 * 2.14) or tc

    # Composite proxy: weighted log-sum of orthogonal signals
    proxy = (
        _math.log1p(best_traffic) * 0.40
        + _math.log1p(sv) * 0.25
        + _math.log1p(sa * 10.0) * 0.15
        + _math.log1p(ip) * 0.10
        + _math.log1p(mj) * 0.05
        + _math.log1p(pr * 100.0) * 0.05
    )

    has_signal = int(best_traffic > 0 or sv > 0 or ip > 100 or mj > 0)

    if best_traffic >= 10000:
        tier = 4
    elif best_traffic >= 1000:
        tier = 3
    elif best_traffic >= 100:
        tier = 2
    elif best_traffic >= 1 or sv > 0 or mj > 0:
        tier = 1
    else:
        tier = 0

    return {
        "traffic_proxy_score": round(proxy, 4),
        "has_web_traffic_signal": has_signal,
        "estimated_traffic_tier": tier,
    }


def _row_to_doc(
    row: Dict[str, Any],
    payload_aliases: Dict[str, List[str]],
    bool_aliases: Dict[str, str],
    source_label: str = "",
) -> Optional[Dict[str, Any]]:
    """Map an Athena result row to the indexer payload format.

    Returns None when the listing is hidden or excluded from search results
    (Python-side quality filter for include_in_search_result_flag / hide_flag).
    When those columns are absent (NULL after CAST fallback), the filter is
    skipped so all rows pass through (safe degradation).

    Only ``domain_name`` is embedded; everything else is structured payload
    stored verbatim for filter retrieval. FIND listing aliases come from
    ``payload_aliases`` / ``bool_aliases`` config maps.

    :param row: Dict[str, Any] - Raw row dict from AthenaClient.fetch_sql_async
    :param payload_aliases: Dict[str, List[str]] - Internal key to FIND aliases
    :param bool_aliases: Dict[str, str] - Internal 0/1 key to FIND bool key
    :param source_label: str - Seed run label (``"daily_snapshot"`` / ``"realtime"``),
        stamped verbatim into the ``data_source`` payload field.
    :return: Optional[Dict[str, Any]] - Payload or None when listing is hidden/excluded
    """
    _search_eligible = row.get("search_eligible")
    if _search_eligible is not None:
        try:
            if int(_search_eligible) == 0:
                return None
        except (ValueError, TypeError):
            pass
    _hidden = row.get("hidden")
    if _hidden is not None:
        try:
            if int(_hidden) == 1:
                return None
        except (ValueError, TypeError):
            pass
    sld = str(row.get("domain_name") or "").strip().lower()
    govalue = _safe_optional_float(row.get("govalue_score"))
    tld = str(row.get("tld") or "").strip().lower()
    if tld and "." not in sld:
        domain = f"{sld}.{tld}"
    elif not tld and "." in sld:
        tld = sld.rsplit(".", 1)[-1]
        domain = sld
    else:
        domain = sld
    _raw_price = row.get("price")
    _price: Optional[float] = float(_raw_price) if _raw_price is not None else None
    _current_bid = _safe_optional_float(row.get("current_bid_price"))
    _quality = min(max(govalue / 100.0, 0.0), 1.0) if govalue is not None else None
    sld = domain.split(".")[0] if "." in domain else domain
    _bid_count = _safe_optional_int(row.get("bid_count"))
    _buy_it_now = _safe_optional_int(row.get("buy_it_now"))
    _buy_it_now_price = _safe_optional_float(row.get("buy_it_now_price"))
    _has_reserve = _safe_optional_int(row.get("has_reserve_price"))
    _reserved_amt = _safe_optional_float(row.get("reserved_price_amount"))
    _starting_bid = _safe_optional_float(row.get("starting_bid"))
    _is_idn = int(sld.startswith("xn--") or not sld.isascii())
    _adult = _safe_optional_int(row.get("adult_listing"))
    _featured = _safe_optional_int(row.get("is_featured"))
    _website = _safe_optional_int(row.get("website_include"))
    _search_eligible = _safe_optional_int(row.get("search_eligible"))
    _on_sale = _safe_optional_float(row.get("on_sale_rate"))
    _member_id = _safe_optional_int(row.get("member_id"))
    _bid_accepted = _safe_optional_int(row.get("bid_accepted_flag"))
    _display_in_category = _safe_optional_int(row.get("display_in_category"))
    _sub_category_featured = _safe_optional_int(row.get("sub_category_featured"))
    _add_i_category = _safe_optional_int(row.get("add_i_category"))

    _ends_at_unix = _parse_end_time(str(row.get("end_time") or "").strip())
    _now_ts = time.time()
    if _ends_at_unix is not None:
        _remaining_s: Optional[float] = max(0.0, _ends_at_unix - _now_ts)
        _time_to_end_minutes: Optional[float] = round(_remaining_s / 60.0, 2)
        _time_to_end_hours: Optional[float] = round(_remaining_s / 3600.0, 4)
        _time_to_end_days: Optional[float] = round(_remaining_s / 86400.0, 4)
    else:
        _remaining_s = None
        _time_to_end_minutes = None
        _time_to_end_hours = None
        _time_to_end_days = None

    _majestic_tf = _safe_optional_int(row.get("majestic_tf"))
    _majestic_cf = _safe_optional_int(row.get("majestic_cf"))
    _majestic_backlinks = _safe_optional_int(row.get("majestic_backlinks"))
    _majestic_ref_domains = _safe_optional_int(row.get("majestic_ref_domains"))
    # unique_search_count is joined in Athena-side by seed_merge (domain_search_rollup
    # join on the final merged table) - real value when search_rollup is configured,
    # else NULL/None.
    _unique_search_count = _safe_optional_int(row.get("unique_search_count"))
    _is_gem = bool(
        _unique_search_count is not None
        and _unique_search_count > 0
        and govalue is not None
        and govalue < 1000
    )
    doc: Dict[str, Any] = {
        "domain_name": domain,
        "item_id": str(row.get("auction_id") or domain),
        "auction_id": str(row.get("auction_id") or ""),
        "tld": tld,
        "auction_type": str(row.get("auction_type") or ""),
        "price": _price,
        "current_bid_price": _current_bid,
        "name_length": len(sld),
        "quality": _quality,
        "score": _quality if _quality is not None else 0.0,
        "govalue_score": govalue,
        "ends_at": _ends_at_unix,
        "time_to_end_seconds": _remaining_s,
        "time_to_end_minutes": _time_to_end_minutes,
        "time_to_end_hours": _time_to_end_hours,
        "time_to_end_days": _time_to_end_days,
        "sld": sld,
        "has_hyphen": int("-" in sld),
        "has_number": int(bool(re.search(r"\d", sld))),
        "is_idn": _is_idn,
        "bid_count": _bid_count,
        "domain_create_date": str(row.get("domain_create_date") or "") or None,
        "domain_age_years": _safe_optional_int(row.get("domain_age_years")) or _compute_domain_age_years(str(row.get("domain_create_date") or "")),
        "monthly_traffic": _safe_optional_int(row.get("monthly_traffic")),
        "last_14day_traffic": _safe_optional_int(row.get("last_14day_traffic")),
        "valuation_rank": _safe_optional_int(row.get("valuation_rank")),
        "parking_revenue": _safe_optional_float(row.get("parking_revenue")),
        "majestic_tf": _majestic_tf,
        "majestic_cf": _majestic_cf,
        "majestic_backlinks": _majestic_backlinks,
        "majestic_ref_domains": _majestic_ref_domains,
        "tlf_exact_match": _safe_optional_int(row.get("tlf_exact_match")),
        "tlf_keyword_regs": _safe_optional_int(row.get("tlf_keyword_regs")),
        "tlf_developed": _safe_optional_int(row.get("tlf_developed")),
        "semrush_backlinks": _safe_optional_int(row.get("semrush_backlinks")),
        "semrush_indexed_pages": _safe_optional_int(row.get("semrush_indexed_pages")),
        "semrush_ref_domains": _safe_optional_int(row.get("semrush_ref_domains")),
        "semrush_authority_score": _safe_optional_float(row.get("semrush_authority_score")),
        **_compute_traffic_features(
            monthly_traffic=_safe_optional_int(row.get("monthly_traffic")),
            last_14day_traffic=_safe_optional_int(row.get("last_14day_traffic")),
            traffic_cnt=_safe_optional_int(row.get("traffic_cnt")),
            parking_revenue=_safe_optional_float(row.get("parking_revenue")),
            semrush_search_volume=_safe_optional_int(row.get("semrush_search_volume")),
            semrush_authority_score=_safe_optional_float(row.get("semrush_authority_score")),
            semrush_indexed_pages=_safe_optional_int(row.get("semrush_indexed_pages")),
            majestic_tf=_majestic_tf,
        ),
        "buy_it_now": _buy_it_now,
        "buy_it_now_price": _buy_it_now_price,
        "has_reserve_price": _has_reserve,
        "reserved_price_amount": _reserved_amt,
        "is_featured": _featured,
        "gd_transfer": _safe_optional_int(row.get("gd_transfer")),
        "adult_listing": _adult,
        "status_code_id": _safe_optional_int(row.get("status_code_id")),
        "category_id": _safe_optional_int(row.get("category_id")),
        "listed_at": str(row.get("listed_at") or ""),
        "member_id": _member_id,
        "on_sale_rate": _on_sale,
        "on_sale_percent": int(_on_sale) if _on_sale is not None else None,
        "starting_bid": _starting_bid,
        "traffic_cnt": _safe_optional_int(row.get("traffic_cnt")),
        "website_include": _website,
        "search_eligible": _search_eligible,
        "item_description": str(row.get("item_description") or "") or None,
        "vendor_id": _safe_optional_int(row.get("vendor_id")),
        "domain_extension_id": _safe_optional_int(row.get("domain_extension_id")),
        "highest_bidder_id": _safe_optional_int(row.get("highest_bidder_id")),
        "bid_accepted_flag": _bid_accepted,
        "display_in_category": _display_in_category,
        "sub_category_featured": _sub_category_featured,
        "add_i_category": _add_i_category,
        "data_update_time": _parse_update_time_iso(row.get("data_update_time")),
        "active": True,
        "data_source": source_label or None,
        # Joined Athena-side onto the final merged table by seed_merge
        # (domain_search_rollup join, when search_rollup is configured).
        "unique_search_count": _unique_search_count,
        "is_gem": _is_gem,
        # Joined Athena-side onto the final merged table by seed_merge
        # (bid-offer table join, per-table).
        "last_bid_offer_dtm": str(row.get("last_bid_offer_dtm") or "") or None,
        # Joined Athena-side onto the final merged table by seed_merge
        # (domain_majestic_metric_snap join, shared across tables).
        "majestic_ext_back_links": _safe_optional_int(row.get("majestic_ext_back_links")),
        "majestic_ref_domains_fm": _safe_optional_int(row.get("majestic_ref_domains_fm")),
        "majestic_citation_flow_score": _safe_optional_int(row.get("majestic_citation_flow_score")),
        "majestic_trust_flow_score": _safe_optional_int(row.get("majestic_trust_flow_score")),
        "majestic_metric_exists": _safe_optional_int(row.get("majestic_metric_exists")) or 0,
        # Joined Athena-side onto the final merged table by seed_merge
        # (semrush_domain_enrichments join, when semrush is configured).
        "semrush_ascore": _safe_optional_float(row.get("semrush_ascore")),
        "semrush_total": _safe_optional_int(row.get("semrush_total")),
        "semrush_domains_num": _safe_optional_int(row.get("semrush_domains_num")),
        "semrush_urls_num": _safe_optional_int(row.get("semrush_urls_num")),
        "semrush_keyword": str(row.get("semrush_keyword") or "") or None,
        "semrush_search_volume": _safe_optional_int(row.get("semrush_search_volume")),
        "semrush_cpc": _safe_optional_float(row.get("semrush_cpc")),
        "semrush_refdomains": _safe_optional_json_list(row.get("semrush_refdomains")),
        # Joined Athena-side onto the final merged table by seed_merge
        # (estibot_domain_enrichments join, when estibot is configured).
        "estibot_domain_count": _safe_optional_int(row.get("estibot_domain_count")),
        "estibot_domain_count_dev": _safe_optional_float(row.get("estibot_domain_count_dev")),
        "estibot_ext_count": _safe_optional_int(row.get("estibot_ext_count")),
        "estibot_ext_count_dev": _safe_optional_float(row.get("estibot_ext_count_dev")),
        # Joined Athena-side onto the final merged table by seed_merge
        # (aftermarket_boost join, when aftermarket_boost is configured).
        "is_boosted_aftermarket": bool(_safe_optional_int(row.get("is_boosted_aftermarket")) or 0),
    }
    _apply_find_aliases(doc, payload_aliases, bool_aliases)
    return doc


def _build_page_count_query(staging_database: str, scratch_table: str) -> str:
    """Build the query returning the highest ``_page_num`` in a scratch table.

    :return: str - ``SELECT MAX(_page_num) AS max_page FROM `{db}`.`{table}```
    """
    return f"SELECT MAX(_page_num) AS max_page FROM {seed_merge._quoted(staging_database, scratch_table)}"


def _build_page_select_query(staging_database: str, scratch_table: str, page_num: int) -> str:
    """Build the query to read one page back from a scratch table.

    :return: str - ``SELECT * FROM `{db}`.`{table}` WHERE _page_num = {page_num}``
    """
    return f"SELECT * FROM {seed_merge._quoted(staging_database, scratch_table)} WHERE _page_num = {page_num}"


def _rows_to_docs(rows: Optional[List[Any]], db_cfg: Any, source_label: str) -> List[Dict[str, Any]]:
    """Map Athena result rows to indexer payload docs, dropping rows with no domain_name.

    :param rows: Optional[List] - Raw rows from AthenaClient (dicts with string values)
    :param db_cfg: SeedDatabaseConfig
    :param source_label: str
    :return: List[Dict] - Mapped docs, ``None`` results from ``_row_to_doc`` dropped
    """
    return [
        d for d in (
            _row_to_doc(row, db_cfg.find_payload_aliases, db_cfg.find_bool_aliases, source_label)
            for row in (rows or [])
            if row.get("domain_name") and str(row.get("domain_name")).strip()
        )
        if d is not None
    ]


async def fetch_seed_pages_from_db(
    athena_client: Any,
    db_cfg: Any,
    source_label: str,
    run_token: str,
    timing: Any,
) -> AsyncIterator[Tuple[List[Dict[str, Any]], List[str], str]]:
    """Fetch seed documents from the configured database tables, one page at a time.

    Every configured ``SeedTableConfig``'s seed query is merged Athena-side
    (``seed_merge.build_seed_merge_plan`` / ``run_seed_merge``) with bid-offer,
    Majestic, search-rollup, estibot, and aftermarket-boost enrichment into
    one paged final table per table, then read back one bounded page at a
    time - bounding peak RAM to O(page_size) instead of O(max_records).
    Every staging and final table this run creates is always dropped on the
    way out, including on exception or cancellation inside the page loop.

    Callers MUST drain this generator via ``contextlib.aclosing`` (or ensure
    it is always fully iterated) so a mid-loop exception in the caller's
    ``async for`` body still triggers this generator's ``finally``/cleanup
    deterministically rather than eventually via garbage collection.

    :param athena_client: AthenaClient
    :param db_cfg: SeedDatabaseConfig
    :param source_label: str - ``"daily_snapshot"`` or ``"realtime"`` for logging
    :param run_token: str - Caller-supplied opaque identifier folded into every
        merge table name so concurrent/successive runs never collide
    :param timing: StageTimingSession - Config-driven stage timer (required)
    :yield: Tuple[List[Dict], List[str], str] - (page_docs, missing_cols, table_name)
    """
    if not athena_client.credentials_available:
        for table_cfg in db_cfg.tables:
            logger.warning(
                f"db_seed_skipped reason=credentials_unavailable "
                f"source={source_label} database={db_cfg.name} table={table_cfg.table_name}"
            )
        return

    base_queries = {tc.table_name: _build_query(db_cfg.name, tc) for tc in db_cfg.tables}
    plan = seed_merge.build_seed_merge_plan(
        db_cfg, run_token, db_cfg.merge_database_location, base_queries
    )

    try:
        missing_by_table = await seed_merge.run_seed_merge(
            athena_client, plan, db_cfg.timeout_seconds, timing,
        )
    except (RuntimeError, TimeoutError, OSError, ValueError, TypeError, KeyError) as exc:
        logger.error(
            f"db_seed_merge_failed source={source_label} database={db_cfg.name} "
            f"error_type={type(exc).__name__} error={str(exc)[:300]}"
        )
        await seed_merge.cleanup_seed_merge(athena_client, plan, db_cfg.timeout_seconds)
        raise DataIngestInterruptedError(
            f"Athena seed merge failed for database={db_cfg.name} source={source_label}: "
            f"{type(exc).__name__}: {str(exc)[:300]}",
            stage='athena_seed_merge',
            records_completed=0,
            records_attempted=0,
            reason='error',
            detail=type(exc).__name__,
        ) from exc

    page_gate = bool(timing.config.log_page_stages)
    try:
        for table_cfg in db_cfg.tables:
            final_table = plan.final_tables[table_cfg.table_name]
            missing_cols = missing_by_table.get(table_cfg.table_name, [])
            if missing_cols:
                logger.warning(
                    f"db_seed_enrichment_columns_absent source={source_label} "
                    f"database={db_cfg.name} table={table_cfg.table_name} "
                    f"missing_columns={missing_cols} payload_value=None "
                    f"hint='run data-build after columns appear in Athena to restore enrichment filtering'"
                )

            with timing.stage(
                "athena_page_count",
                gate=page_gate,
                source=source_label,
                table=table_cfg.table_name,
            ):
                count_rows, _cols, _lat = await athena_client.fetch_sql_async(
                    query=_build_page_count_query(plan.database, final_table),
                    timeout_seconds=db_cfg.timeout_seconds,
                )
            max_page = _safe_optional_int(count_rows[0].get("max_page")) if count_rows else None
            total_pages = (max_page + 1) if max_page is not None else 0

            if total_pages == 0:
                # Query succeeded but returned zero rows - still counts as
                # queried (not skipped), matching the legacy always-yield
                # behavior on fetch success.
                yield [], missing_cols, table_cfg.table_name
                continue

            for page_num in range(total_pages):
                with timing.stage(
                    "athena_page_fetch",
                    gate=page_gate,
                    source=source_label,
                    table=table_cfg.table_name,
                    page=page_num + 1,
                    total_pages=total_pages,
                ):
                    page_rows, _cols, _lat = await athena_client.fetch_sql_async(
                        query=_build_page_select_query(plan.database, final_table, page_num),
                        timeout_seconds=db_cfg.timeout_seconds,
                    )
                    page_docs = _rows_to_docs(page_rows, db_cfg, source_label)
                logger.info(
                    f"db_seed_page_fetched source={source_label} database={db_cfg.name} "
                    f"table={table_cfg.table_name} page={page_num + 1}/{total_pages} "
                    f"rows={len(page_rows or [])} docs={len(page_docs)}"
                )
                yield page_docs, missing_cols, table_cfg.table_name
    finally:
        with timing.stage(
            "athena_merge_cleanup",
            gate=bool(timing.config.log_merge_phases),
            source=source_label,
            database=db_cfg.name,
        ):
            await seed_merge.cleanup_seed_merge(athena_client, plan, db_cfg.timeout_seconds)


async def fetch_seed_from_db(
    athena_client: Any,
    db_cfg: Any,
    source_label: str,
    timing: Any,
) -> Tuple[List[Dict[str, Any]], DbSeedSummary]:
    """Fetch seed documents from the configured database tables.

    Thin wrapper draining ``fetch_seed_pages_from_db`` into one flat list -
    kept for callers that need the whole batch at once (e.g. the legacy
    Qdrant/BM25 index-loading path). Callers with a large expected result set
    should use ``fetch_seed_pages_from_db`` directly to bound peak RAM.

    Each configured table's underlying generator page size is bounded by
    ``seed_merge``'s ``_PAGE_SIZE`` (paging the Athena-side merged final
    table), so this wrapper may drain more than one page per table.

    :param athena_client: AthenaClient - ``nl_to_sql.AthenaClient`` instance.
    :param db_cfg: SeedDatabaseConfig - Database and table configuration.
    :param source_label: str - ``"daily_snapshot"`` or ``"realtime"`` for logging.
    :param timing: StageTimingSession - Config-driven stage timer (required).
    :return: Tuple[List[Dict], DbSeedSummary] - Domain payload list and summary.
    """
    start = time.monotonic()
    all_docs: List[Dict[str, Any]] = []
    all_missing_cols: List[str] = []
    tables_seen: set = set()

    async with contextlib.aclosing(
        fetch_seed_pages_from_db(
            athena_client, db_cfg, source_label, run_token=source_label, timing=timing,
        )
    ) as pages:
        async for docs, missing_cols, table_name in pages:
            all_docs.extend(docs)
            tables_seen.add(table_name)
            all_missing_cols.extend(c for c in missing_cols if c not in all_missing_cols)

    queried = len(tables_seen)
    skipped = len(db_cfg.tables) - queried
    elapsed = (time.monotonic() - start) * 1000
    summary = DbSeedSummary(
        source=source_label,
        database=db_cfg.name,
        documents=len(all_docs),
        tables_queried=queried,
        tables_skipped=skipped,
        elapsed_ms=round(elapsed, 1),
        missing_columns=all_missing_cols,
    )
    logger.info(
        f"db_seed_complete source={source_label} documents={summary.documents} "
        f"tables_queried={queried} tables_skipped={skipped} "
        f"missing_columns={all_missing_cols} elapsed_ms={summary.elapsed_ms}"
    )
    return all_docs, summary
