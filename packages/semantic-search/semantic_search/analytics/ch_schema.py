"""Canonical ClickHouse analytics schema — single source of truth.

Both the manual initializer (``scripts/init_clickhouse.py``) and the auto-seed
writer (``explore/ch_seed_writer.py``) import ``DDL_STATEMENTS`` from here, so
the provisioned schema can never drift between the two paths.

The materialized-view set and the per-view aggregate-state columns are kept in
lock-step with ``nl_to_sql.analytics.mv_router.materialized_views`` in
``config/base.yaml``: every MV declared there is created here, and every view
builds the full set of aggregate states the multi-period query builder may emit
(count / avg / min / max / median / sum). A mismatch causes runtime
``ClickHouseQueryError`` because the query references a state column the MV does
not contain — which is exactly the failure this module exists to prevent.

All views read from ``signals_platform_cln.auction_audit_cln`` and use AggregatingMergeTree
so the *State / *Merge aggregate-function pattern works.

Coverage map for the 82-query analytics roadmap:
  Marketplace / TLD sold analytics    -> mv_sold_by_tld_day, mv_sold_by_type_day
  Category keyword analytics          -> mv_auctions_by_category_name_day, mv_sold_by_category_name_day
  Sell-through rate (Q6, Q12, Q29)    -> mv_sell_through_by_tld_day
  Sell-through by auction type        -> mv_sell_through_by_type_day (new)
  Registrar drop/expiry (Q32, Q36)    -> mv_auctions_by_registrar_day
  Auction type name analytics         -> mv_auctions_by_type_name_day (new)
  Expiry lifecycle (Q33-36)           -> mv_expiry_lifecycle_day (new) + expiry_status column
  Hold time / listing age             -> hold_days MATERIALIZED + mv_hold_time_by_type_day (new)
  Weekly TLD sold trends              -> mv_sold_by_tld_week (new)
  Bid distribution (Q17, Q25-27)      -> sum_bids_state / avg_bids_state in all MVs
  SEO correlation (Q19-24)            -> seo_bucket_sql query template (no MV needed)
  Investor/flip analytics (Q37-42)    -> signals_platform_cln.domain_transactions table
  Buyer segmentation (Q43-54)         -> buyer_segment column
  Market summary (Q82)                -> market_summary_sql query template

Auction type ID->label mapping (source: base.yaml nl_to_sql.qi.known_auction_type_ids comments):
  16 = GoDaddy AutoExtend (expiry auction)   38 = Partner AutoExtend (expiry auction)
  20 = GoDaddy BuyNow (fixed price)          39 = Partner Closeout
"""
from __future__ import annotations

import re as _re

from typing import Dict, List, Sequence, Tuple

# Auction type ID->label mapping, synced with base.yaml known_auction_type_ids comments.
# Used in transform() DEFAULT expressions for auction_type_name and in mutation backfill.
_AUCTION_TYPE_IDS = [16, 20, 38, 39]
_AUCTION_TYPE_LABELS = ["expiry auction", "buynow", "expiry auction", "closeout"]

def _auction_type_transform_expr() -> str:
    """Render a ClickHouse transform() expression mapping auction_type_id -> label string."""
    ids = ", ".join(str(i) for i in _AUCTION_TYPE_IDS)
    labels = ", ".join(f"'{lbl}'" for lbl in _AUCTION_TYPE_LABELS)
    return f"transform(auction_type_id, [{ids}], [{labels}], 'other')"

def _expiry_status_expr() -> str:
    """Render a ClickHouse multiIf() expression deriving expiry_status from ends_at + sold_flag."""
    return (
        "multiIf("
        "sold_flag = 1, 'sold', "
        "ends_at < now() AND sold_flag = 0, 'expired', "
        "dateDiff('day', now64(3), ends_at) <= 7 AND sold_flag = 0, 'expiring_soon', "
        "'active'"
        ")"
    )


def _category_name_expr() -> str:
    """Render a ClickHouse multiIf() expression deriving category_name from domain_name patterns.

    Checks the second-level domain (SLD) against keyword regexes in priority order.
    Returns the first matching category label or '' when no pattern matches.
    """
    sld = "lower(splitByChar('.', domain_name)[1])"
    return (
        "multiIf("
        f"match({sld}, '(^|[-_])(ai|ml|llm|gpt|nlp|neural|deeplearn|genai|intelli)'), 'ai', "
        f"match({sld}, '(^|[-_])(health|med|clinic|hospital|pharma|doctor|care|dental|therapy|wellness)'), 'healthcare', "
        f"match({sld}, '(^|[-_])(fin|bank|pay|crypto|trade|invest|fund|wealth|loan|lending|defi|forex)'), 'fintech', "
        f"match({sld}, '(^|[-_])(law|legal|attorney|court|lawyer|notary|counsel|litigation)'), 'legal', "
        f"match({sld}, '(^|[-_])(tech|software|app|code|dev|cloud|saas|api|digital|platform|cyber)'), 'tech', "
        f"match({sld}, '(^|[-_])(realt|estate|home|house|propert|mortgage|reit|realtor)'), 'real-estate', "
        f"match({sld}, '(^|[-_])(shop|store|ecommerce|retail|market|deal|merch|brand|goods)'), 'ecommerce', "
        "'')"
    )

# Source table every MV aggregates from.
_SOURCE_TABLE = "signals_platform_cln.auction_audit_cln"

# ---------------------------------------------------------------------------
# Aggregate-state projection shared by every standard MV.
# Column names MUST match the ``aggregate_columns`` keys in the mv_router
# config — the multi-period query builder derives the *Merge function from
# each column-name prefix (count_ -> countMerge, avg_ -> avgMerge, ...).
# New states added (2024): sum_bids_state, avg_bids_state, avg_govalue_state,
# avg_age_state — cover bid-volume, underpriced detection, and SEO signals_platform_cln.
# ---------------------------------------------------------------------------
_AGG_STATES: str = (
    "    countState(*)                   AS count_state,\n"
    "    avgState(current_price)         AS avg_price_state,\n"
    "    minState(current_price)         AS min_price_state,\n"
    "    maxState(current_price)         AS max_price_state,\n"
    "    medianState(current_price)      AS median_price_state,\n"
    "    sumState(current_price)         AS sum_price_state,\n"
    "    avgState(monthly_traffic)       AS avg_traffic_state,\n"
    "    medianState(monthly_traffic)    AS median_traffic_state,\n"
    "    avgState(domain_authority)      AS avg_authority_state,\n"
    "    avgState(backlink_count)        AS avg_backlinks_state,\n"
    "    medianState(backlink_count)     AS median_backlinks_state,\n"
    "    sumState(bid_count)             AS sum_bids_state,\n"
    "    avgState(bid_count)             AS avg_bids_state,\n"
    "    avgState(govalue_score)         AS avg_govalue_state,\n"
    "    avgState(domain_age_days)       AS avg_age_state"
)

# Aggregate states for sold-specific MVs (uses sold_at time column + sold_flag=1 filter).
# Omits backlinks/traffic median to keep sold MVs focused on sale-outcome metrics.
_SOLD_AGG_STATES: str = (
    "    countState(*)                   AS count_state,\n"
    "    avgState(current_price)         AS avg_price_state,\n"
    "    minState(current_price)         AS min_price_state,\n"
    "    maxState(current_price)         AS max_price_state,\n"
    "    medianState(current_price)      AS median_price_state,\n"
    "    sumState(current_price)         AS sum_price_state,\n"
    "    avgState(domain_authority)      AS avg_authority_state,\n"
    "    avgState(monthly_traffic)       AS avg_traffic_state,\n"
    "    sumState(bid_count)             AS sum_bids_state,\n"
    "    avgState(bid_count)             AS avg_bids_state,\n"
    "    avgState(govalue_score)         AS avg_govalue_state,\n"
    "    avgState(domain_age_days)       AS avg_age_state"
)


def _mv_ddl(name: str, time_alias: str, time_expr: str, grain: Sequence[Tuple[str, str]]) -> str:
    """Render a CREATE MATERIALIZED VIEW statement.

    :param name: str - Fully-qualified MV name (e.g. ``signals_platform_cln.mv_auctions_by_tld_day``)
    :param time_alias: str - Time-bucket column alias (``event_day`` / ``event_hour`` / ``event_week``)
    :param time_expr: str - Expression producing the bucket (e.g. ``toDate(ends_at)``)
    :param grain: Sequence[Tuple[str, str]] - (alias, expr) pairs for grain columns;
        ``alias == expr`` emits the bare column (no redundant ``AS``)
    :return: str - DDL statement
    """
    grain_aliases = [alias for alias, _ in grain]
    key_cols = ", ".join([time_alias] + grain_aliases)

    select_lines: List[str] = [f"    {time_expr} AS {time_alias},"]
    for alias, expr in grain:
        select_lines.append(f"    {expr} AS {alias}," if alias != expr else f"    {alias},")
    select_block = "\n".join(select_lines)

    query = (  # nosec B608 - config-driven identifiers, no user input
        f"CREATE MATERIALIZED VIEW IF NOT EXISTS {name}\n"
        f"ENGINE = AggregatingMergeTree()\n"
        f"ORDER BY ({key_cols})\n"
        f"AS SELECT\n"
        f"{select_block}\n"
        f"{_AGG_STATES}\n"
        f"FROM {_SOURCE_TABLE}\n"
        f"GROUP BY {key_cols}"
    )
    return query


def _sold_mv_ddl(name: str, grain: Sequence[Tuple[str, str]]) -> str:
    """Render a sold-only MATERIALIZED VIEW (WHERE sold_flag = 1, time = sold_at).

    Sold MVs pre-filter on sold_flag=1 so queries for sold-domain analytics
    (marketplace totals, TLD sold volume, category revenue) hit a compact MV
    rather than scanning the full auction table.

    :param name: str - Fully-qualified MV name
    :param grain: Sequence[Tuple[str, str]] - (alias, expr) grain column pairs
    :return: str - DDL statement
    """
    grain_aliases = [alias for alias, _ in grain]
    key_cols = ", ".join(["event_day"] + grain_aliases)

    select_lines: List[str] = ["    toDate(assumeNotNull(sold_at)) AS event_day,"]
    for alias, expr in grain:
        select_lines.append(f"    {expr} AS {alias}," if alias != expr else f"    {alias},")
    select_block = "\n".join(select_lines)

    query = (  # nosec B608 - config-driven identifiers, no user input
        f"CREATE MATERIALIZED VIEW IF NOT EXISTS {name}\n"
        f"ENGINE = AggregatingMergeTree()\n"
        f"ORDER BY ({key_cols})\n"
        f"AS SELECT\n"
        f"{select_block}\n"
        f"{_SOLD_AGG_STATES}\n"
        f"FROM {_SOURCE_TABLE}\n"
        f"WHERE sold_flag = 1 AND sold_at IS NOT NULL\n"
        f"GROUP BY {key_cols}"
    )
    return query



# MV specs — one entry per config-declared materialized view. Keep in sync with
# config/base.yaml nl_to_sql.analytics.mv_router.materialized_views.
#   (name, time_alias, time_expr, grain[(alias, expr), ...])
_MV_SPECS: List[Tuple[str, str, str, List[Tuple[str, str]]]] = [
    ("signals_platform_cln.mv_auctions_by_tld_day",       "event_day",  "toDate(ends_at)",        [("tld", "tld")]),
    ("signals_platform_cln.mv_auctions_by_type_day",      "event_day",  "toDate(ends_at)",        [("auction_type_id", "auction_type_id")]),
    ("signals_platform_cln.mv_auctions_by_tld_hour",      "event_hour", "toStartOfHour(ends_at)", [("tld", "tld")]),
    ("signals_platform_cln.mv_auctions_by_category_day",  "event_day",  "toDate(ends_at)",        [("auction_type_id", "auction_type_id"), ("tld", "tld")]),
    ("signals_platform_cln.mv_auctions_growth_week",      "event_week", "toStartOfWeek(ends_at)", [("tld", "tld"), ("auction_type_id", "auction_type_id")]),
    ("signals_platform_cln.mv_auctions_by_tld_week",      "event_week", "toStartOfWeek(ends_at)", [("tld", "tld")]),
    ("signals_platform_cln.mv_auctions_by_type_week",     "event_week", "toStartOfWeek(ends_at)", [("auction_type_id", "auction_type_id")]),
    ("signals_platform_cln.mv_user_engagement_day",       "event_day",  "toDate(ends_at)",        [("user_id", "user_id")]),
    ("signals_platform_cln.mv_auctions_by_type_hour",     "event_hour", "toStartOfHour(ends_at)", [("auction_type_id", "auction_type_id")]),
    ("signals_platform_cln.mv_auctions_by_type_tld_hour", "event_hour", "toStartOfHour(ends_at)", [("tld", "tld"), ("auction_type_id", "auction_type_id")]),
    ("signals_platform_cln.mv_auctions_by_type_tld_day",       "event_day",  "toDate(ends_at)",        [("tld", "tld"), ("auction_type_id", "auction_type_id")]),
    ("signals_platform_cln.mv_auctions_by_category_name_week", "event_week", "toStartOfWeek(ends_at)", [("category_name", "category_name")]),
    # New listings by TLD — time bucket uses listed_at so "recently added" queries
    # get accurate listing counts rather than auction-close counts.
    ("signals_platform_cln.mv_new_listings_by_tld_day",           "listing_day", "toDate(listed_at)",        [("tld", "tld")]),
    # New listings by category name — same listed_at axis for category-level new-listing queries.
    ("signals_platform_cln.mv_new_listings_by_category_name_day", "listing_day", "toDate(listed_at)",        [("category_name", "category_name")]),
    # TLD × category_name cross-dim — covers "tld distribution for healthcare/legal/professional".
    ("signals_platform_cln.mv_auctions_by_tld_category_name_day", "event_day",   "toDate(ends_at)",          [("tld", "tld"), ("category_name", "category_name")]),
    # Expiry status × registrar — covers "pending delete distribution by registrar" queries.
    ("signals_platform_cln.mv_auctions_by_expiry_registrar_day",  "event_day",   "toDate(ends_at)",          [("expiry_status", "expiry_status"), ("registrar_name", "registrar_name")]),
]

# Sold-specific MV specs — time column is sold_at, WHERE sold_flag=1.
# Powers Q1-9 (marketplace totals), Q30 (avg winning bid), Q61-64 (top sold lists).
_MV_SOLD_SPECS: List[Tuple[str, List[Tuple[str, str]]]] = [
    ("signals_platform_cln.mv_sold_by_tld_day",            [("tld", "tld")]),
    ("signals_platform_cln.mv_sold_by_type_day",           [("auction_type_id", "auction_type_id")]),
    ("signals_platform_cln.mv_sold_by_category_name_day",  [("category_name", "category_name")]),
    ("signals_platform_cln.mv_sold_by_category_name_week", [("category_name", "category_name")]),
]

# Sell-through rate MV: total count + sold count by (tld, day).
# sumState(sold_flag) gives count-of-sold since sold_flag ∈ {0,1}.
# Powers Q6 (overall sell-through), Q12 (TLD comparison), Q29 (auction success rate).
_MV_SELL_THROUGH_BY_TLD_DAY = """\
CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_sell_through_by_tld_day
ENGINE = AggregatingMergeTree()
ORDER BY (event_day, tld)
AS SELECT
    toDate(ends_at) AS event_day,
    tld,
    countState(*)               AS count_state,
    sumState(sold_flag)         AS sold_count_state,
    avgState(current_price)     AS avg_price_state,
    sumState(current_price)     AS sum_price_state,
    avgState(bid_count)         AS avg_bids_state
FROM signals_platform_cln.auction_audit_cln
GROUP BY event_day, tld"""

# Registrar-level aggregation MV.
# Powers Q32 (registrar drop volume), Q36 (expiry sales trend by registrar).
_MV_AUCTIONS_BY_REGISTRAR_DAY = """\
CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_auctions_by_registrar_day
ENGINE = AggregatingMergeTree()
ORDER BY (event_day, registrar_name)
AS SELECT
    toDate(ends_at) AS event_day,
    registrar_name,
    countState(*)               AS count_state,
    sumState(sold_flag)         AS sold_count_state,
    avgState(current_price)     AS avg_price_state,
    sumState(current_price)     AS sum_price_state,
    avgState(domain_authority)  AS avg_authority_state,
    avgState(bid_count)         AS avg_bids_state
FROM signals_platform_cln.auction_audit_cln
GROUP BY event_day, registrar_name"""

# Category keyword (human-readable category_name) MV — all auctions.
# Distinct from mv_auctions_by_category_day which uses auction_type_id integer.
# Powers Q13-18 (ai/fintech/healthcare category analytics), Q62 (top categories by revenue).
_MV_AUCTIONS_BY_CATEGORY_NAME_DAY = """\
CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_auctions_by_category_name_day
ENGINE = AggregatingMergeTree()
ORDER BY (event_day, category_name)
AS SELECT
    toDate(ends_at) AS event_day,
    category_name,
    countState(*)               AS count_state,
    sumState(sold_flag)         AS sold_count_state,
    avgState(current_price)     AS avg_price_state,
    minState(current_price)     AS min_price_state,
    maxState(current_price)     AS max_price_state,
    medianState(current_price)  AS median_price_state,
    sumState(current_price)     AS sum_price_state,
    sumState(bid_count)         AS sum_bids_state,
    avgState(bid_count)         AS avg_bids_state,
    avgState(domain_authority)  AS avg_authority_state,
    avgState(monthly_traffic)   AS avg_traffic_state
FROM signals_platform_cln.auction_audit_cln
GROUP BY event_day, category_name"""

# Base table — point-in-time active-auction snapshot the MVs aggregate from.
# PARTITION BY toYYYYMM(listed_at): bulk range scans skip irrelevant month-partitions
#   (critical for 90-day avg queries on 8.6M in-scope records).
# TTL 120 days: 30-day buffer above the 90-day analytics window so a "last 90 days"
#   query never clips a partition that dropped at midnight. Auction lifecycle max is
#   72 days (analysis_auction.md §2), so 120-day TTL retains all historical auctions
#   needed for analytics before cleanup.
# Scope: auction_type_id IN (16, 20, 38, 39) only — enforced at ingest, not DDL.
_DAM_AUCTION_SNAP = """\
CREATE TABLE IF NOT EXISTS signals_platform_cln.auction_audit_cln (
    auction_id                 Int64,
    domain_name                String,
    tld                        LowCardinality(String),
    auction_type_id            Int32,
    user_id                    Int64                   DEFAULT 0,
    current_price              Float64                 DEFAULT 0.0,
    govalue_score              Float64                 DEFAULT 0.0,
    ends_at                    DateTime64(3),
    bid_count                  Int32                   DEFAULT 0,
    created_at                 DateTime64(3)           DEFAULT now64(3),
    updated_at                 DateTime64(3)           DEFAULT now64(3),
    sold_flag                  UInt8                   DEFAULT 0,
    listed_at                  DateTime64(3)           DEFAULT now64(3),
    sold_at                    Nullable(DateTime64(3)) DEFAULT NULL,
    domain_authority           Float32                 DEFAULT 0.0,
    semrush_authority_score    Float32                 DEFAULT 0.0,
    monthly_traffic            UInt32                  DEFAULT 0,
    domain_age_days            UInt16                  DEFAULT 0,
    registrar_name             LowCardinality(String)  DEFAULT '',
    backlink_count             UInt32                  DEFAULT 0,
    referring_domains_count    UInt16                  DEFAULT 0,
    majestic_backlinks         UInt32                  DEFAULT 0,
    majestic_ref_domains       UInt16                  DEFAULT 0,
    semrush_backlinks          UInt32                  DEFAULT 0,
    semrush_ref_domains        UInt16                  DEFAULT 0,
    auction_type_name          LowCardinality(String)  DEFAULT '',
    category_name              LowCardinality(String)  DEFAULT '',
    buy_it_now_price           Float64                 DEFAULT 0.0,
    is_featured                UInt8                   DEFAULT 0,
    gd_transfer                UInt8                   DEFAULT 0,
    on_sale_rate               Float32                 DEFAULT 0.0,
    starting_bid               Float64                 DEFAULT 0.0,
    category_id                Int32                   DEFAULT 0
) ENGINE = ReplacingMergeTree(updated_at)
PARTITION BY toYYYYMM(listed_at)
ORDER BY (auction_id)
TTL toDateTime(listed_at) + INTERVAL 120 DAY"""

# Behavioral analytics signal store (conversion funnels, buyer heatmaps).
_FEEDBACK_SIGNALS = """\
CREATE TABLE IF NOT EXISTS signals_platform_cln.feedback_signals (
    signal_id       String,
    request_id      String,
    signal_type     LowCardinality(String),
    payload         String   DEFAULT '',
    signal_origin   LowCardinality(String) DEFAULT 'unknown',
    created_at      DateTime64(3)
) ENGINE = MergeTree()
ORDER BY (signal_type, created_at)
TTL toDateTime(created_at) + INTERVAL 90 DAY"""

# Historical point-in-time snapshots — proxy substrate for past-tense
# ("sold last week") queries served by analytics/snapshot_port.py.
_DOMAIN_SNAPSHOTS = """\
CREATE TABLE IF NOT EXISTS signals_platform_cln.domain_snapshots (
    snapshot_date   Date,
    domain_name     String,
    tld             LowCardinality(String),
    current_price              Float64                DEFAULT 0.0,
    govalue_score              Float64                DEFAULT 0.0,
    bid_count                  Int32                  DEFAULT 0,
    auction_type_id            Int32                  DEFAULT 0,
    snapped_at                 DateTime64(3)          DEFAULT now64(3),
    sold_flag                  UInt8                  DEFAULT 0,
    domain_authority           Float32                DEFAULT 0.0,
    monthly_traffic            UInt32                 DEFAULT 0,
    domain_age_days            UInt16                 DEFAULT 0,
    registrar_name             LowCardinality(String) DEFAULT '',
    backlink_count             UInt32                 DEFAULT 0,
    referring_domains_count    UInt16                 DEFAULT 0,
    majestic_backlinks         UInt32                 DEFAULT 0,
    majestic_ref_domains       UInt16                 DEFAULT 0,
    semrush_backlinks          UInt32                 DEFAULT 0,
    semrush_ref_domains        UInt16                 DEFAULT 0,
    auction_type_name          LowCardinality(String) DEFAULT '',
    category_name              LowCardinality(String) DEFAULT ''
) ENGINE = ReplacingMergeTree(snapped_at)
ORDER BY (snapshot_date, domain_name)
TTL snapshot_date + INTERVAL 365 DAY"""

# Domain transaction ledger — one row per confirmed buy/sell event.
# Populated externally (bid settlement / payment confirmation pipeline).
# Powers investor analytics: flip profit margin (Q37), ROI by category (Q38),
# long-hold appreciation (Q39), best investment category (Q41).
# Uses MergeTree (not ReplacingMergeTree) — each transaction is a unique event.
# 2-year TTL; partition by month for efficient time-range scans.
_DOMAIN_TRANSACTIONS = """\
CREATE TABLE IF NOT EXISTS signals_platform_cln.domain_transactions (
    transaction_id     String,
    domain_name        String,
    tld                LowCardinality(String)  DEFAULT '',
    sale_price         Float64                 DEFAULT 0.0,
    listed_price       Float64                 DEFAULT 0.0,
    buyer_user_id      Int64                   DEFAULT 0,
    auction_type_id    Int32                   DEFAULT 0,
    auction_type_name  LowCardinality(String)  DEFAULT '',
    category_name      LowCardinality(String)  DEFAULT '',
    buyer_segment      LowCardinality(String)  DEFAULT '',
    domain_authority   Float32                 DEFAULT 0.0,
    monthly_traffic    UInt32                  DEFAULT 0,
    domain_age_days    UInt16                  DEFAULT 0,
    govalue_score      Float64                 DEFAULT 0.0,
    backlink_count     UInt32                  DEFAULT 0,
    sold_at            DateTime64(3),
    listed_at          DateTime64(3)           DEFAULT now64(3),
    created_at         DateTime64(3)           DEFAULT now64(3)
) ENGINE = MergeTree()
ORDER BY (sold_at, tld)
PARTITION BY toYYYYMM(sold_at)
TTL toDateTime(sold_at) + INTERVAL 730 DAY"""

_CREATE_EXPIRY_INDEX = """\
ALTER TABLE signals_platform_cln.auction_audit_cln ADD INDEX IF NOT EXISTS idx_expiry_lookup (registrar_name, expiry_status, toDate(ends_at)) TYPE minmax GRANULARITY 8192"""

# New analytical MVs — auction type name, expiry lifecycle, hold time, weekly sold.
# All are CREATE ... IF NOT EXISTS, safe to re-run.

# Expiry lifecycle by day and status bucket — powers Q33-Q36 expiry signals_platform_cln.
# Depends on expiry_status column being populated (via mutation or DEFAULT expr).
_MV_EXPIRY_LIFECYCLE_DAY = """\
CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_expiry_lifecycle_day
ENGINE = AggregatingMergeTree()
ORDER BY (event_day, expiry_status)
AS SELECT
    toDate(ends_at) AS event_day,
    expiry_status,
    countState(*)               AS count_state,
    sumState(sold_flag)         AS sold_count_state,
    avgState(current_price)     AS avg_price_state,
    minState(current_price)     AS min_price_state,
    maxState(current_price)     AS max_price_state,
    sumState(current_price)     AS sum_price_state,
    avgState(bid_count)         AS avg_bids_state,
    avgState(domain_authority)  AS avg_authority_state
FROM signals_platform_cln.auction_audit_cln
GROUP BY event_day, expiry_status"""

# Sell-through rate by auction type name — mirrors mv_sell_through_by_tld_day
# but groups by auction_type_name (human label) instead of tld.
# Powers closeout vs buynow vs expiry auction sell-through comparisons.
_MV_SELL_THROUGH_BY_TYPE_DAY = """\
CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_sell_through_by_type_day
ENGINE = AggregatingMergeTree()
ORDER BY (event_day, auction_type_name)
AS SELECT
    toDate(ends_at) AS event_day,
    auction_type_name,
    countState(*)               AS count_state,
    sumState(sold_flag)         AS sold_count_state,
    avgState(current_price)     AS avg_price_state,
    sumState(current_price)     AS sum_price_state,
    avgState(bid_count)         AS avg_bids_state
FROM signals_platform_cln.auction_audit_cln
GROUP BY event_day, auction_type_name"""

# Full stats by auction type name per day — all standard aggregate states
# plus hold_days and sold counts. Powers auction type performance signals_platform_cln.
_MV_AUCTIONS_BY_TYPE_NAME_DAY = """\
CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_auctions_by_type_name_day
ENGINE = AggregatingMergeTree()
ORDER BY (event_day, auction_type_name)
AS SELECT
    toDate(ends_at) AS event_day,
    auction_type_name,
    countState(*)                   AS count_state,
    avgState(current_price)         AS avg_price_state,
    minState(current_price)         AS min_price_state,
    maxState(current_price)         AS max_price_state,
    medianState(current_price)      AS median_price_state,
    sumState(current_price)         AS sum_price_state,
    avgState(monthly_traffic)       AS avg_traffic_state,
    avgState(domain_authority)      AS avg_authority_state,
    avgState(backlink_count)        AS avg_backlinks_state,
    sumState(bid_count)             AS sum_bids_state,
    avgState(bid_count)             AS avg_bids_state,
    avgState(govalue_score)         AS avg_govalue_state,
    avgState(domain_age_days)       AS avg_age_state,
    sumState(sold_flag)             AS sold_count_state,
    avgState(hold_days)             AS avg_hold_days_state
FROM signals_platform_cln.auction_audit_cln
GROUP BY event_day, auction_type_name"""

# Hold time distribution for sold domains by (event_day, auction_type_name, tld).
# WHERE sold_flag=1 to include only completed sales; time bucket uses sold_at.
# Powers hold time analytics: how long domains were listed before selling.
_MV_HOLD_TIME_BY_TYPE_DAY = """\
CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_hold_time_by_type_day
ENGINE = AggregatingMergeTree()
ORDER BY (event_day, auction_type_name, tld)
AS SELECT
    toDate(assumeNotNull(sold_at))  AS event_day,
    auction_type_name,
    tld,
    countState(*)                   AS count_state,
    avgState(hold_days)             AS avg_hold_days_state,
    minState(hold_days)             AS min_hold_days_state,
    maxState(hold_days)             AS max_hold_days_state,
    avgState(current_price)         AS avg_price_state,
    sumState(current_price)         AS sum_price_state
FROM signals_platform_cln.auction_audit_cln
WHERE sold_flag = 1 AND sold_at IS NOT NULL
GROUP BY event_day, auction_type_name, tld"""

# Weekly sold aggregation by TLD — coarser grain than mv_sold_by_tld_day.
# Powers weekly trend analytics (Q60-series) without needing date_trunc in query.
_MV_SOLD_BY_TLD_WEEK = """\
CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_sold_by_tld_week
ENGINE = AggregatingMergeTree()
ORDER BY (event_week, tld)
AS SELECT
    toStartOfWeek(assumeNotNull(sold_at)) AS event_week,
    tld,
    countState(*)               AS count_state,
    avgState(current_price)     AS avg_price_state,
    minState(current_price)     AS min_price_state,
    maxState(current_price)     AS max_price_state,
    medianState(current_price)  AS median_price_state,
    sumState(current_price)     AS sum_price_state,
    avgState(domain_authority)  AS avg_authority_state,
    avgState(monthly_traffic)   AS avg_traffic_state,
    sumState(bid_count)         AS sum_bids_state,
    avgState(bid_count)         AS avg_bids_state,
    avgState(govalue_score)     AS avg_govalue_state,
    avgState(domain_age_days)   AS avg_age_state
FROM signals_platform_cln.auction_audit_cln
WHERE sold_flag = 1 AND sold_at IS NOT NULL
GROUP BY event_week, tld"""


# Buyer-segment daily MV — activity + sell-through by buyer profile label.
# Powers Q43-Q60: Beginner / Professional / Advanced / Investor analytics.
# WHERE buyer_segment != '' skips un-classified rows (avoids polluting aggregates).
_MV_AUCTIONS_BY_BUYER_SEGMENT_DAY = """\
CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_auctions_by_buyer_segment_day
ENGINE = AggregatingMergeTree()
ORDER BY (event_day, buyer_segment)
AS SELECT
    toDate(ends_at)              AS event_day,
    buyer_segment,
    countState(*)                AS count_state,
    sumState(sold_flag)          AS sold_count_state,
    avgState(current_price)      AS avg_price_state,
    minState(current_price)      AS min_price_state,
    maxState(current_price)      AS max_price_state,
    sumState(current_price)      AS sum_price_state,
    avgState(bid_count)          AS avg_bids_state,
    avgState(domain_authority)   AS avg_authority_state,
    avgState(monthly_traffic)    AS avg_traffic_state
FROM signals_platform_cln.auction_audit_cln
WHERE buyer_segment != ''
GROUP BY event_day, buyer_segment"""

# Buyer-segment weekly MV — 12-week rolling trend queries.
_MV_AUCTIONS_BY_BUYER_SEGMENT_WEEK = """\
CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_auctions_by_buyer_segment_week
ENGINE = AggregatingMergeTree()
ORDER BY (event_week, buyer_segment)
AS SELECT
    toStartOfWeek(ends_at)       AS event_week,
    buyer_segment,
    countState(*)                AS count_state,
    sumState(sold_flag)          AS sold_count_state,
    avgState(current_price)      AS avg_price_state,
    minState(current_price)      AS min_price_state,
    maxState(current_price)      AS max_price_state,
    sumState(current_price)      AS sum_price_state,
    avgState(bid_count)          AS avg_bids_state,
    avgState(domain_authority)   AS avg_authority_state,
    avgState(monthly_traffic)    AS avg_traffic_state
FROM signals_platform_cln.auction_audit_cln
WHERE buyer_segment != ''
GROUP BY event_week, buyer_segment"""


# ---------------------------------------------------------------------------
# Bid events — from the_resale_place.item_bids_cln (Phase 1.5 ingest target).
# auction_id populated via item_winning_bids_cln join (0 when unmatched).
# Human-bid filter: buy_it_now_flag = 0; confirm bid_source value via SQL diagnostic #B1.
# ---------------------------------------------------------------------------
_BID_EVENTS = """\
CREATE TABLE IF NOT EXISTS signals_platform_cln.bid_events (
    bid_event_id     String,
    member_item_id   Int64,
    auction_id       Int64                    DEFAULT 0,
    bidder_id        Int64                    DEFAULT 0,
    seller_id        Int64                    DEFAULT 0,
    bid_usd_amount   Float64                  DEFAULT 0.0,
    bid_source       LowCardinality(String)   DEFAULT '',
    buy_it_now_flag  UInt8                    DEFAULT 0,
    bid_accepted     UInt8                    DEFAULT 0,
    counter_offer    UInt8                    DEFAULT 0,
    bid_start_at     DateTime64(3),
    bid_end_at       Nullable(DateTime64(3))  DEFAULT NULL,
    event_utc_ts     DateTime64(3),
    received_at      DateTime64(3)            DEFAULT now64(3)
) ENGINE = ReplacingMergeTree(received_at)
PARTITION BY toYYYYMM(event_utc_ts)
ORDER BY (member_item_id, bid_start_at)
TTL toDateTime(event_utc_ts) + INTERVAL 120 DAY"""

# Bid velocity by auction_id per hour — trending-now MV.
# auction_id > 0 filter excludes rows where the winning-bids join was incomplete.
_MV_BID_VELOCITY_BY_AUCTION_HOUR = """\
CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_bid_velocity_by_auction_hour
ENGINE = AggregatingMergeTree()
ORDER BY (auction_id, event_hour)
AS SELECT
    auction_id,
    toStartOfHour(event_utc_ts)  AS event_hour,
    countState()                 AS bid_count_state,
    sumState(bid_usd_amount)     AS sum_bid_state,
    maxState(bid_usd_amount)     AS max_bid_state,
    uniqState(bidder_id)         AS unique_bidders_state
FROM signals_platform_cln.bid_events
WHERE auction_id > 0
  AND buy_it_now_flag = 0
GROUP BY auction_id, event_hour"""

# Bid velocity by member_item_id per hour — fallback when auction_id join is incomplete.
# member_item_id is always populated from the source stream.
_MV_BID_VELOCITY_BY_ITEM_HOUR = """\
CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_bid_velocity_by_item_hour
ENGINE = AggregatingMergeTree()
ORDER BY (member_item_id, event_hour)
AS SELECT
    member_item_id,
    toStartOfHour(event_utc_ts)  AS event_hour,
    countState()                 AS bid_count_state,
    sumState(bid_usd_amount)     AS sum_bid_state,
    maxState(bid_usd_amount)     AS max_bid_state,
    uniqState(bidder_id)         AS unique_bidders_state
FROM signals_platform_cln.bid_events
WHERE buy_it_now_flag = 0
GROUP BY member_item_id, event_hour"""

# ---------------------------------------------------------------------------
# Watch events — from the_resale_place.member_items_watch_cln (Phase 1.5).
# watch_type_label populated from member_items_watch_types_cln at ingest time.
# ---------------------------------------------------------------------------
_WATCH_EVENTS = """\
CREATE TABLE IF NOT EXISTS signals_platform_cln.watch_events (
    watch_event_id    String,
    member_item_id    Int64,
    member_id         Int64                    DEFAULT 0,
    watch_type        Int32                    DEFAULT 0,
    watch_type_label  LowCardinality(String)   DEFAULT '',
    is_deleted        UInt8                    DEFAULT 0,
    created_at        DateTime64(3),
    modified_at       Nullable(DateTime64(3))  DEFAULT NULL,
    event_utc_ts      DateTime64(3),
    received_at       DateTime64(3)            DEFAULT now64(3)
) ENGINE = ReplacingMergeTree(received_at)
PARTITION BY toYYYYMM(event_utc_ts)
ORDER BY (member_item_id, watch_event_id)
TTL toDateTime(event_utc_ts) + INTERVAL 120 DAY"""

# Watch density per listing per day — complementary engagement signal alongside bid velocity.
# active_watch_state counts only non-deleted watch records (net watchlist adds).
# unique_watchers_state guards against member_id = 0 (field not populated in source stream).
_MV_WATCH_DENSITY_BY_ITEM_DAY = """\
CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_watch_density_by_item_day
ENGINE = AggregatingMergeTree()
ORDER BY (member_item_id, event_day)
AS SELECT
    member_item_id,
    toDate(event_utc_ts)                        AS event_day,
    countStateIf(is_deleted = 0)                AS active_watch_state,
    uniqStateIf(member_id, member_id > 0)       AS unique_watchers_state
FROM signals_platform_cln.watch_events
WHERE watch_type IN (1, 9)
GROUP BY member_item_id, event_day"""

# Bidder-intent watch density — type 9 ("Items I am bidding on") only.
# Stronger demand signal than passive watches (type 1): bidder added domain to their
# active-bid tracking list, indicating purchase intent beyond casual interest.
# Separate MV avoids mixing intent strengths in active_watch_state.
_MV_BIDDER_WATCH_DENSITY_BY_ITEM_DAY = """\
CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_bidder_watch_density_by_item_day
ENGINE = AggregatingMergeTree()
ORDER BY (member_item_id, event_day)
AS SELECT
    member_item_id,
    toDate(event_utc_ts)                        AS event_day,
    countStateIf(is_deleted = 0)                AS bidder_watch_state,
    uniqStateIf(member_id, member_id > 0)       AS unique_bidder_watchers_state
FROM signals_platform_cln.watch_events
WHERE watch_type = 9
GROUP BY member_item_id, event_day"""


# Competitive-depth signal: counts distinct human bidders per listing per hour.
# Keyed on member_item_id (not auction_id) — the bid_events bridge resolves to
# auction_id at query time.  Filters to bid_source = 'auc-bidding' to exclude
# automated pricing bots (same filter as mv_bid_velocity_by_item_hour).
_MV_UNIQUE_BIDDER_COUNT_BY_ITEM = """\
CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_unique_bidder_count_by_item
ENGINE = AggregatingMergeTree()
ORDER BY (member_item_id, event_hour)
AS SELECT
    member_item_id,
    toStartOfHour(event_utc_ts)                          AS event_hour,
    uniqStateIf(bidder_id, bid_source = 'auc-bidding')   AS unique_bidder_state
FROM signals_platform_cln.bid_events
GROUP BY member_item_id, event_hour"""


# ---------------------------------------------------------------------------
# Enriched feature tables — Python ETL (EnrichedTablesBuilder) populates these
# by reading from the base tables, applying EnrichmentPipeline steps, and
# writing computed feature columns back to ClickHouse.  Analytics engine methods
# query these tables for richer, more diverse analytics questions.
# ---------------------------------------------------------------------------

# Per-TLD weekly enriched aggregation — growth rates, anomaly flags, momentum,
# sell-through tiers, and linear forecast computed outside CH and stored here.
_ENRICHED_TLD_FEATURES = """\
CREATE TABLE IF NOT EXISTS signals_platform_cln.enriched_tld_features (
    tld                  LowCardinality(String),
    event_week           Date,
    avg_price            Float64                DEFAULT 0.0,
    total_auctions       UInt32                 DEFAULT 0,
    sold_auctions        UInt32                 DEFAULT 0,
    sell_through_rate    Float32                DEFAULT 0.0,
    avg_bids             Float32                DEFAULT 0.0,
    avg_govalue          Float32                DEFAULT 0.0,
    growth_pct           Nullable(Float32)      DEFAULT NULL,
    forecast_value       Nullable(Float32)      DEFAULT NULL,
    z_score              Nullable(Float32)      DEFAULT NULL,
    is_anomaly           UInt8                  DEFAULT 0,
    tld_momentum_pct     Nullable(Float32)      DEFAULT NULL,
    tld_momentum_label   LowCardinality(String) DEFAULT '',
    sell_through_tier    LowCardinality(String) DEFAULT '',
    refreshed_at         DateTime64(3)          DEFAULT now64(3)
) ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (tld, event_week)
TTL event_week + INTERVAL 1 YEAR"""

# Per-auction enriched features — heat score, time urgency, lifecycle stage,
# fair value bands, sell-through probability, composite rank, and engagement
# signals (watcher_count, unique_bidder_count) computed by the Python builder.
_ENRICHED_AUCTION_FEATURES = """\
CREATE TABLE IF NOT EXISTS signals_platform_cln.enriched_auction_features (
    auction_id           Int64,
    domain_name          String,
    tld                  LowCardinality(String)  DEFAULT '',
    current_price        Float64                 DEFAULT 0.0,
    bid_count            Int32                   DEFAULT 0,
    ends_at              DateTime64(3),
    govalue_score        Float64                 DEFAULT 0.0,
    domain_age_days      UInt16                  DEFAULT 0,
    monthly_traffic      UInt32                  DEFAULT 0,
    heat_score           LowCardinality(String)  DEFAULT '',
    time_urgency_label   LowCardinality(String)  DEFAULT '',
    lifecycle_stage      LowCardinality(String)  DEFAULT '',
    fair_value_p25       Nullable(Float64)       DEFAULT NULL,
    fair_value_p75       Nullable(Float64)       DEFAULT NULL,
    sell_through_prob    Nullable(Float32)       DEFAULT NULL,
    sell_through_tier    LowCardinality(String)  DEFAULT '',
    composite_score      Float64                 DEFAULT 0.0,
    tld_rank             UInt32                  DEFAULT 0,
    watcher_count        UInt32                  DEFAULT 0,
    unique_bidder_count  UInt32                  DEFAULT 0,
    category_name        LowCardinality(String)  DEFAULT '',
    refreshed_at         DateTime64(3)           DEFAULT now64(3)
) ENGINE = ReplacingMergeTree(refreshed_at)
ORDER BY (auction_id)
TTL toDateTime(ends_at) + INTERVAL 30 DAY"""


_ALTER_SNAP_COLS: List[Tuple[str, str, str]] = [
    ("sold_flag",               "UInt8",                   "0"),
    ("listed_at",               "DateTime64(3)",            "now64(3)"),
    ("sold_at",                 "Nullable(DateTime64(3))",  "NULL"),
    ("domain_authority",        "Float32",                  "0.0"),
    ("govalue_score",           "Float64",                  "0.0"),
    ("semrush_authority_score", "Float32",                  "0.0"),
    ("monthly_traffic",         "UInt32",                   "0"),
    ("domain_age_days",         "UInt16",                   "0"),
    ("registrar_name",          "LowCardinality(String)",   "''"),
    ("backlink_count",          "UInt32",                   "0"),
    ("referring_domains_count", "UInt16",                   "0"),
    ("majestic_backlinks",      "UInt32",                   "0"),
    ("majestic_ref_domains",    "UInt16",                   "0"),
    ("semrush_backlinks",       "UInt32",                   "0"),
    ("semrush_ref_domains",     "UInt16",                   "0"),
    ("auction_type_name",       "LowCardinality(String)",   "''"),
    ("category_name",           "LowCardinality(String)",   "''"),
    ("user_id",                 "Int64",                    "0"),
    # Expiry lifecycle status — 'active' | 'pending_delete' | 'grace_period' | 'expired'.
    # Powers Q33 (pending delete count), Q34 (grace period recovery), Q35 (backorder rate).
    ("expiry_status",           "LowCardinality(String)",   "''"),
    # Buyer segment classification — 'beginner' | 'professional' | 'advanced' | 'investor'.
    # Populated by buyer-profile pipeline. Powers Q43-60 (segment analytics).
    ("buyer_segment",           "LowCardinality(String)",   "''"),
    # Auction feature columns from auction_audit_cln.
    ("buy_it_now_price",        "Float64",                  "0.0"),
    ("is_featured",             "UInt8",                    "0"),
    ("gd_transfer",             "UInt8",                    "0"),
    ("on_sale_rate",            "Float32",                  "0.0"),
    ("starting_bid",            "Float64",                  "0.0"),
    ("category_id",             "Int32",                    "0"),
]

# MATERIALIZED columns — computed at INSERT/merge time, stored in parts.
# ADD COLUMN IF NOT EXISTS is safe to re-run; existing installs get the column.
# _DAM_AUCTION_SNAP does not include these so fresh installs also need this ALTER.
# Tuple: (column_name, type, materialized_expression)
_ALTER_SNAP_MATERIALIZED: List[Tuple[str, str, str]] = [
    # Listing duration in days: how long domain has been offered before ends_at.
    # Powers hold-time analytics and investor listing-age queries.
    ("hold_days", "UInt16",
     "toUInt16(greatest(0, dateDiff('day', listed_at, ends_at)))"),
    # Character count of domain_name (full label including TLD, e.g. "example.com" = 11).
    # Powers length-filter queries (short/long domain analytics).
    ("domain_length", "UInt16", "toUInt16(length(domain_name))"),
]

# ---------------------------------------------------------------------------
# Column registry — parsed from the DDL constants above at import time.
# DomainAnalyticsEngine reads this at construction to validate configured and
# hardcoded column names before the first query fires.
# Only tables present in this dict are checked; unknown table names are skipped
# so staging environments and custom installs do not fail at boot.
# ---------------------------------------------------------------------------
_COL_BODY_RE = _re.compile(r'\(\s*\n(.*?)\n\s*\)\s+ENGINE', _re.DOTALL)
_COL_DEF_RE = _re.compile(r'^\s{4}(\w+)\s', _re.MULTILINE)


def _parse_columns_from_ddl(ddl: str) -> frozenset:
    """Return column names found in a CREATE TABLE DDL body."""
    m = _COL_BODY_RE.search(ddl)
    if not m:
        return frozenset()
    return frozenset(cm.group(1) for cm in _COL_DEF_RE.finditer(m.group(1)))


KNOWN_COLUMNS: Dict[str, frozenset] = {
    'signals_platform_cln.auction_audit_cln': (
        _parse_columns_from_ddl(_DAM_AUCTION_SNAP)
        | {col for col, _, _ in _ALTER_SNAP_COLS}
        | {col for col, _, _ in _ALTER_SNAP_MATERIALIZED}
    ),
    'signals_platform_cln.feedback_signals': _parse_columns_from_ddl(_FEEDBACK_SIGNALS),
    'signals_platform_cln.domain_snapshots': _parse_columns_from_ddl(_DOMAIN_SNAPSHOTS),
    'signals_platform_cln.domain_transactions': (
        _parse_columns_from_ddl(_DOMAIN_TRANSACTIONS)
        | {'hold_days'}
    ),
    'signals_platform_cln.bid_events': _parse_columns_from_ddl(_BID_EVENTS),
    'signals_platform_cln.watch_events': _parse_columns_from_ddl(_WATCH_EVENTS),
    'signals_platform_cln.enriched_tld_features': _parse_columns_from_ddl(_ENRICHED_TLD_FEATURES),
    'signals_platform_cln.enriched_auction_features': _parse_columns_from_ddl(_ENRICHED_AUCTION_FEATURES),
}


# MODIFY COLUMN DEFAULT — change DEFAULT expression for existing columns so that
# new rows receive a derived value instead of '' (the original ADD COLUMN default).
# Tuple: (column_name, type, default_expression)
# Existing rows already present in CH are backfilled via MIGRATION_STATEMENTS below.
def _get_modify_default_cols() -> List[Tuple[str, str, str]]:
    return [
        ("expiry_status", "LowCardinality(String)", _expiry_status_expr()),
        ("auction_type_name", "LowCardinality(String)", _auction_type_transform_expr()),
        ("category_name", "LowCardinality(String)", _category_name_expr()),
    ]


def _build_ddl() -> List[str]:
    """Assemble the ordered, idempotent DDL statement list."""
    statements: List[str] = [
        "CREATE DATABASE IF NOT EXISTS signals_platform_cln",
        _DAM_AUCTION_SNAP,
    ]
    # Idempotent column additions to base table (new installs already have these
    # from _DAM_AUCTION_SNAP; existing installs get them via ALTER).
    for col, col_type, default in _ALTER_SNAP_COLS:
        statements.append(
            f"ALTER TABLE {_SOURCE_TABLE} ADD COLUMN IF NOT EXISTS {col} {col_type} DEFAULT {default}"
        )
    # MATERIALIZED computed columns (stored, always derived from expression at insert/merge).
    for col, col_type, expr in _ALTER_SNAP_MATERIALIZED:
        statements.append(
            f"ALTER TABLE {_SOURCE_TABLE} ADD COLUMN IF NOT EXISTS {col} {col_type} MATERIALIZED {expr}"
        )
    # Update DEFAULT expressions for expiry_status and auction_type_name so new rows
    # get derived values automatically (existing rows backfilled via MIGRATION_STATEMENTS).
    for col, col_type, default_expr in _get_modify_default_cols():
        statements.append(
            f"ALTER TABLE {_SOURCE_TABLE} MODIFY COLUMN IF EXISTS {col} {col_type} DEFAULT {default_expr}"
        )
    # Index on expiry queries (Q31-Q33) - speeds registrar + expiry_status scans 20s+ -> 2s.
    statements.append(_CREATE_EXPIRY_INDEX)
    # Drop the mv_auctions_by_category_day that stored auction_type_id under the
    # alias 'category' (router grain mismatch). CREATE IF NOT EXISTS in _MV_SPECS
    # recreates it with the corrected auction_type_id column name.
    statements.append("DROP TABLE IF EXISTS signals_platform_cln.mv_auctions_by_category_day")
    # Standard MVs (use ends_at time column, no sold_flag filter).
    statements.extend(_mv_ddl(*spec) for spec in _MV_SPECS)
    # NOTE: ALTER TABLE ADD COLUMN is not supported on MaterializedView storage in
    # ClickHouse (Code: 48). New state columns are included in _AGG_STATES above;
    # fresh installs get them via CREATE MV. Existing installs require DROP+recreate.
    # Sold-specific MVs (use sold_at time column, WHERE sold_flag=1).
    statements.extend(_sold_mv_ddl(*spec) for spec in _MV_SOLD_SPECS)
    # Sell-through, registrar, and category-name MVs (raw DDL — non-standard columns).
    statements.append(_MV_SELL_THROUGH_BY_TLD_DAY)
    statements.append(_MV_AUCTIONS_BY_REGISTRAR_DAY)
    statements.append(_MV_AUCTIONS_BY_CATEGORY_NAME_DAY)
    # New analytical MVs: expiry lifecycle, sell-through by type, type name stats,
    # hold time by type, weekly sold by TLD.
    statements.append(_MV_EXPIRY_LIFECYCLE_DAY)
    statements.append(_MV_SELL_THROUGH_BY_TYPE_DAY)
    statements.append(_MV_AUCTIONS_BY_TYPE_NAME_DAY)
    statements.append(_MV_HOLD_TIME_BY_TYPE_DAY)
    statements.append(_MV_SOLD_BY_TLD_WEEK)
    # Buyer-segment MVs — powers Q43-Q60 (beginner/professional/advanced/investor analytics).
    statements.append(_MV_AUCTIONS_BY_BUYER_SEGMENT_DAY)
    statements.append(_MV_AUCTIONS_BY_BUYER_SEGMENT_WEEK)
    # Real-time bid + watch event tables and their MVs (Phase 1.5 ingest).
    statements.append(_BID_EVENTS)
    statements.append(_MV_BID_VELOCITY_BY_AUCTION_HOUR)
    statements.append(_MV_BID_VELOCITY_BY_ITEM_HOUR)
    statements.append(_WATCH_EVENTS)
    statements.append(_MV_WATCH_DENSITY_BY_ITEM_DAY)
    statements.append(_MV_BIDDER_WATCH_DENSITY_BY_ITEM_DAY)
    statements.append(_MV_UNIQUE_BIDDER_COUNT_BY_ITEM)
    # Auxiliary tables.
    statements.append(_FEEDBACK_SIGNALS)
    statements.append(_DOMAIN_SNAPSHOTS)
    statements.append(_DOMAIN_TRANSACTIONS)
    # hold_days must come after _DOMAIN_TRANSACTIONS is created.
    statements.append(
        "ALTER TABLE signals_platform_cln.domain_transactions"
        " ADD COLUMN IF NOT EXISTS hold_days UInt16 MATERIALIZED"
        " toUInt16(greatest(0, dateDiff('day', listed_at, sold_at)))"
    )
    # Enriched feature tables — populated by EnrichedTablesBuilder (Python ETL).
    statements.append(_ENRICHED_TLD_FEATURES)
    statements.append(_ENRICHED_AUCTION_FEATURES)
    return statements


def _build_migration_statements() -> List[str]:
    """Async ClickHouse mutations to backfill existing rows that predate DEFAULT expressions.

    These run in the background — ClickHouse executes them asynchronously.
    WHERE guards make each mutation idempotent: re-running is safe.
    """
    return [
        # Populate auction_type_name from auction_type_id for rows where it was inserted as ''.
        (
            f"ALTER TABLE {_SOURCE_TABLE} UPDATE"
            f" auction_type_name = {_auction_type_transform_expr()}"
            " WHERE auction_type_name = ''"
        ),
        # Populate expiry_status from ends_at + sold_flag for rows still at the empty default.
        (
            f"ALTER TABLE {_SOURCE_TABLE} UPDATE"
            f" expiry_status = {_expiry_status_expr()}"
            " WHERE expiry_status = ''"
        ),
        # Derive category_name for all existing rows from domain_name keyword patterns.
        (
            f"ALTER TABLE {_SOURCE_TABLE} UPDATE"
            f" category_name = {_category_name_expr()}"
            " WHERE category_name = ''"
        ),
        # Backfill mv_auctions_by_category_day - dropped and recreated with auction_type_id grain.
        (  # nosec B608 - config-driven schema/table names, no user input
            f"INSERT INTO signals_platform_cln.mv_auctions_by_category_day"
            f" SELECT toDate(ends_at) AS event_day, auction_type_id, tld,\n{_AGG_STATES}"
            f"\nFROM {_SOURCE_TABLE} WHERE ends_at IS NOT NULL"
            f" GROUP BY event_day, auction_type_id, tld"
        ),
        # Backfill mv_new_listings_by_tld_day - time axis is listed_at.
        (  # nosec B608 - config-driven schema/table names, no user input
            f"INSERT INTO signals_platform_cln.mv_new_listings_by_tld_day"
            f" SELECT toDate(listed_at) AS listing_day, tld,\n{_AGG_STATES}"
            f"\nFROM {_SOURCE_TABLE} WHERE listed_at IS NOT NULL"
            f" GROUP BY listing_day, tld"
        ),
        # Backfill mv_new_listings_by_category_name_day - derive category_name inline so the MV
        # gets labelled data immediately without waiting for the async ALTER UPDATE above.
        (  # nosec B608 - config-driven schema/table names, no user input
            f"INSERT INTO signals_platform_cln.mv_new_listings_by_category_name_day"
            f" SELECT toDate(listed_at) AS listing_day,"
            f" {_category_name_expr()} AS category_name,\n{_AGG_STATES}"
            f"\nFROM {_SOURCE_TABLE} WHERE listed_at IS NOT NULL AND {_category_name_expr()} != ''"
            f" GROUP BY listing_day, category_name"
        ),
        # Backfill mv_auctions_by_tld_category_name_day - derive category_name inline.
        (  # nosec B608 - config-driven schema/table names, no user input
            f"INSERT INTO signals_platform_cln.mv_auctions_by_tld_category_name_day"
            f" SELECT toDate(ends_at) AS event_day, tld,"
            f" {_category_name_expr()} AS category_name,\n{_AGG_STATES}"
            f"\nFROM {_SOURCE_TABLE} WHERE ends_at IS NOT NULL AND {_category_name_expr()} != ''"
            f" GROUP BY event_day, tld, category_name"
        ),
        # Backfill mv_auctions_by_expiry_registrar_day — expiry_status × registrar cross-dim.
        (
            f"INSERT INTO signals_platform_cln.mv_auctions_by_expiry_registrar_day"
            f" SELECT toDate(ends_at) AS event_day, expiry_status, registrar_name,\n{_AGG_STATES}"
            f"\nFROM {_SOURCE_TABLE} WHERE ends_at IS NOT NULL AND expiry_status != ''"
            f" GROUP BY event_day, expiry_status, registrar_name"
        ),
    ]


# Ordered DDL: database -> base table -> columns -> MVs -> auxiliary tables. Every
# statement is CREATE ... IF NOT EXISTS or ALTER ... IF NOT EXISTS,
# so the list is safe to re-run.
DDL_STATEMENTS: List[str] = _build_ddl()

# Async ClickHouse mutations for existing-row backfill. Run via
# scripts/init_clickhouse.py --migrate after DDL_STATEMENTS complete.
MIGRATION_STATEMENTS: List[str] = _build_migration_statements()

# Minimal DDL for the real-time event ingest tables only (bid_events,
# watch_events, and their MVs). Safe to run at driver startup — all
# statements are CREATE IF NOT EXISTS, so re-runs are no-ops.
EVENT_TABLE_DDL_STATEMENTS: List[str] = [
    "CREATE DATABASE IF NOT EXISTS signals_platform_cln",
    _BID_EVENTS,
    _MV_BID_VELOCITY_BY_AUCTION_HOUR,
    _MV_BID_VELOCITY_BY_ITEM_HOUR,
    _MV_UNIQUE_BIDDER_COUNT_BY_ITEM,
    _WATCH_EVENTS,
    _MV_WATCH_DENSITY_BY_ITEM_DAY,
    _MV_BIDDER_WATCH_DENSITY_BY_ITEM_DAY,
]


def get_ddl_statements(source_table: str) -> List[str]:
    """Return DDL statements with ``source_table`` as the base table name.

    When ``source_table`` matches the module default (``signals_platform_cln.auction_audit_cln``),
    returns the pre-built list directly. Otherwise performs a string-level
    substitution so callers can inject the table name from config rather than
    relying on the hardcoded module constant.

    :param source_table: str - Fully-qualified ClickHouse table name (e.g. ``signals_platform_cln.auction_audit_cln``)
    :return: List[str] - Ordered, idempotent DDL statement list
    """
    if source_table == _SOURCE_TABLE:
        return list(DDL_STATEMENTS)
    return [stmt.replace(_SOURCE_TABLE, source_table) for stmt in DDL_STATEMENTS]


def get_migration_statements(source_table: str) -> List[str]:
    """Return migration (ALTER … UPDATE) statements with ``source_table`` substituted.

    :param source_table: str - Fully-qualified ClickHouse table name
    :return: List[str] - Mutation statements for existing-row backfill
    """
    if source_table == _SOURCE_TABLE:
        return list(MIGRATION_STATEMENTS)
    return [stmt.replace(_SOURCE_TABLE, source_table) for stmt in MIGRATION_STATEMENTS]


ENRICHED_TABLE_DDL_STATEMENTS: List[str] = [
    "CREATE DATABASE IF NOT EXISTS signals_platform_cln",
    _ENRICHED_TLD_FEATURES,
    _ENRICHED_AUCTION_FEATURES,
]


__all__ = [
    "DDL_STATEMENTS",
    "ENRICHED_TABLE_DDL_STATEMENTS",
    "KNOWN_COLUMNS",
    "MIGRATION_STATEMENTS",
    "EVENT_TABLE_DDL_STATEMENTS",
    "get_ddl_statements",
    "get_migration_statements",
]
