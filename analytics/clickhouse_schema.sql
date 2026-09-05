-- ============================================================
-- Analytics ClickHouse schema
-- Database: signals_platform_cln
-- Base table: auction_audit_cln
-- Materialized views: per grain (tld, type, category, registrar, expiry, buyer)
-- Run once per environment; safe to re-run (IF NOT EXISTS guards).
-- Populate base table via scripts/seed_clickhouse_from_athena
-- or the /data/build API endpoint.
-- ============================================================

CREATE DATABASE IF NOT EXISTS signals_platform_cln;

-- ============================================================
-- Base table
-- ============================================================
CREATE TABLE IF NOT EXISTS signals_platform_cln.auction_audit_cln
(
    -- Identity
    auction_id              UInt64,
    domain_name             String,
    tld                     LowCardinality(String),
    auction_type_id         UInt8,
    auction_type_name       LowCardinality(String),

    -- Price / activity
    current_price           Float64,
    bid_count               UInt32,
    govalue_score           Float32,

    -- Timestamps
    ends_at                 Nullable(DateTime('UTC')),
    created_at              DateTime('UTC'),
    listed_at               Nullable(DateTime('UTC')),
    sold_at                 Nullable(DateTime('UTC')),
    updated_at              DateTime('UTC') DEFAULT now(),

    -- Sale outcome
    sold_flag               UInt8,                          -- 1 = sold, 0 = active/unsold
    hold_days               Float64 DEFAULT 0,              -- days between listed and sold

    -- Classification
    category_name           LowCardinality(String),
    registrar_name          LowCardinality(String),
    expiry_status           LowCardinality(String),         -- active | pending_delete | dropped | sold
    buyer_segment           LowCardinality(String),         -- beginner | professional | advanced | investor

    -- SEO / authority
    domain_authority        Float32 DEFAULT 0,
    monthly_traffic         Float64 DEFAULT 0,
    domain_age_days         UInt32  DEFAULT 0,
    backlink_count          UInt64  DEFAULT 0,
    referring_domains_count UInt32  DEFAULT 0,

    -- User attribution (optional; 0 = anonymous)
    user_id                 UInt64  DEFAULT 0,

    -- Derived metrics (MATERIALIZED — stored, auto-computed from domain_name at insert)
    domain_length           UInt16        MATERIALIZED toUInt16(length(domain_name)),

    -- Computed time partitions (materialized for MV keys)
    event_day               Date          MATERIALIZED toDate(created_at),
    event_hour              DateTime      MATERIALIZED toStartOfHour(created_at),
    event_week              Date          MATERIALIZED toMonday(created_at)
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(created_at)
ORDER BY (tld, auction_type_id, created_at)
SETTINGS index_granularity = 8192;


-- ============================================================
-- Feedback / interaction signals
-- ============================================================
CREATE TABLE IF NOT EXISTS signals_platform_cln.feedback_signals
(
    signal_id   UUID          DEFAULT generateUUIDv4(),
    domain_name String,
    signal_type LowCardinality(String),   -- click | bid | watch | search | purchase
    user_id     UInt64 DEFAULT 0,
    occurred_at DateTime('UTC') DEFAULT now(),
    metadata    String DEFAULT '{}'       -- JSON blob, optional
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(occurred_at)
ORDER BY (domain_name, occurred_at);


-- ============================================================
-- Historical snapshots (point-in-time domain state)
-- ============================================================
CREATE TABLE IF NOT EXISTS signals_platform_cln.domain_snapshots
(
    snapshot_date   Date,
    domain_name     String,
    tld             LowCardinality(String),
    current_price   Float64,
    bid_count       UInt32,
    domain_authority Float32 DEFAULT 0,
    monthly_traffic  Float64 DEFAULT 0,
    sold_flag        UInt8   DEFAULT 0
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(snapshot_date)
ORDER BY (snapshot_date, tld, domain_name);


-- ============================================================
-- MV backing tables (AggregatingMergeTree)
-- ============================================================

-- mv_auctions_by_tld_day  — daily TLD aggregates
CREATE TABLE IF NOT EXISTS signals_platform_cln.mv_auctions_by_tld_day
(
    tld                 LowCardinality(String),
    event_day           Date,
    count_state         AggregateFunction(count),
    avg_price_state     AggregateFunction(avg, Float64),
    min_price_state     AggregateFunction(min, Float64),
    max_price_state     AggregateFunction(max, Float64),
    sum_price_state     AggregateFunction(sum, Float64),
    median_price_state  AggregateFunction(quantiles(0.5), Float64),
    avg_bids_state      AggregateFunction(avg, Float64),
    sum_bids_state      AggregateFunction(sum, Float64),
    avg_govalue_state   AggregateFunction(avg, Float32),
    avg_age_state       AggregateFunction(avg, Float64)
)
ENGINE = AggregatingMergeTree()
ORDER BY (tld, event_day);

CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_auctions_by_tld_day_mv
TO signals_platform_cln.mv_auctions_by_tld_day
AS SELECT
    tld,
    toDate(created_at)                      AS event_day,
    countState()                            AS count_state,
    avgState(current_price)                 AS avg_price_state,
    minState(current_price)                 AS min_price_state,
    maxState(current_price)                 AS max_price_state,
    sumState(current_price)                 AS sum_price_state,
    quantilesState(0.5)(current_price)      AS median_price_state,
    avgState(toFloat64(bid_count))          AS avg_bids_state,
    sumState(toFloat64(bid_count))          AS sum_bids_state,
    avgState(toFloat64(govalue_score))      AS avg_govalue_state,
    avgState(toFloat64(domain_age_days))    AS avg_age_state
FROM signals_platform_cln.auction_audit_cln;


-- mv_auctions_by_tld_hour  — hourly TLD aggregates (real-time trending)
CREATE TABLE IF NOT EXISTS signals_platform_cln.mv_auctions_by_tld_hour
(
    tld                 LowCardinality(String),
    event_hour          DateTime,
    count_state         AggregateFunction(count),
    avg_price_state     AggregateFunction(avg, Float64),
    min_price_state     AggregateFunction(min, Float64),
    max_price_state     AggregateFunction(max, Float64),
    sum_price_state     AggregateFunction(sum, Float64),
    median_price_state  AggregateFunction(quantiles(0.5), Float64),
    avg_bids_state      AggregateFunction(avg, Float64),
    sum_bids_state      AggregateFunction(sum, Float64),
    avg_govalue_state   AggregateFunction(avg, Float32),
    avg_age_state       AggregateFunction(avg, Float64)
)
ENGINE = AggregatingMergeTree()
ORDER BY (tld, event_hour);

CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_auctions_by_tld_hour_mv
TO signals_platform_cln.mv_auctions_by_tld_hour
AS SELECT
    tld,
    toStartOfHour(created_at)               AS event_hour,
    countState()                            AS count_state,
    avgState(current_price)                 AS avg_price_state,
    minState(current_price)                 AS min_price_state,
    maxState(current_price)                 AS max_price_state,
    sumState(current_price)                 AS sum_price_state,
    quantilesState(0.5)(current_price)      AS median_price_state,
    avgState(toFloat64(bid_count))          AS avg_bids_state,
    sumState(toFloat64(bid_count))          AS sum_bids_state,
    avgState(toFloat64(govalue_score))      AS avg_govalue_state,
    avgState(toFloat64(domain_age_days))    AS avg_age_state
FROM signals_platform_cln.auction_audit_cln;


-- mv_auctions_by_tld_week  — weekly TLD aggregates
CREATE TABLE IF NOT EXISTS signals_platform_cln.mv_auctions_by_tld_week
(
    tld                 LowCardinality(String),
    event_week          Date,
    count_state         AggregateFunction(count),
    avg_price_state     AggregateFunction(avg, Float64),
    min_price_state     AggregateFunction(min, Float64),
    max_price_state     AggregateFunction(max, Float64),
    sum_price_state     AggregateFunction(sum, Float64),
    median_price_state  AggregateFunction(quantiles(0.5), Float64),
    avg_bids_state      AggregateFunction(avg, Float64),
    sum_bids_state      AggregateFunction(sum, Float64),
    avg_govalue_state   AggregateFunction(avg, Float32),
    avg_age_state       AggregateFunction(avg, Float64)
)
ENGINE = AggregatingMergeTree()
ORDER BY (tld, event_week);

CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_auctions_by_tld_week_mv
TO signals_platform_cln.mv_auctions_by_tld_week
AS SELECT
    tld,
    toMonday(created_at)                    AS event_week,
    countState()                            AS count_state,
    avgState(current_price)                 AS avg_price_state,
    minState(current_price)                 AS min_price_state,
    maxState(current_price)                 AS max_price_state,
    sumState(current_price)                 AS sum_price_state,
    quantilesState(0.5)(current_price)      AS median_price_state,
    avgState(toFloat64(bid_count))          AS avg_bids_state,
    sumState(toFloat64(bid_count))          AS sum_bids_state,
    avgState(toFloat64(govalue_score))      AS avg_govalue_state,
    avgState(toFloat64(domain_age_days))    AS avg_age_state
FROM signals_platform_cln.auction_audit_cln;


-- mv_auctions_by_type_day  — daily auction-type aggregates
CREATE TABLE IF NOT EXISTS signals_platform_cln.mv_auctions_by_type_day
(
    auction_type_id     UInt8,
    event_day           Date,
    count_state         AggregateFunction(count),
    avg_price_state     AggregateFunction(avg, Float64),
    min_price_state     AggregateFunction(min, Float64),
    max_price_state     AggregateFunction(max, Float64),
    sum_price_state     AggregateFunction(sum, Float64),
    median_price_state  AggregateFunction(quantiles(0.5), Float64),
    avg_bids_state      AggregateFunction(avg, Float64),
    sum_bids_state      AggregateFunction(sum, Float64),
    avg_govalue_state   AggregateFunction(avg, Float32),
    avg_age_state       AggregateFunction(avg, Float64)
)
ENGINE = AggregatingMergeTree()
ORDER BY (auction_type_id, event_day);

CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_auctions_by_type_day_mv
TO signals_platform_cln.mv_auctions_by_type_day
AS SELECT
    auction_type_id,
    toDate(created_at)                      AS event_day,
    countState()                            AS count_state,
    avgState(current_price)                 AS avg_price_state,
    minState(current_price)                 AS min_price_state,
    maxState(current_price)                 AS max_price_state,
    sumState(current_price)                 AS sum_price_state,
    quantilesState(0.5)(current_price)      AS median_price_state,
    avgState(toFloat64(bid_count))          AS avg_bids_state,
    sumState(toFloat64(bid_count))          AS sum_bids_state,
    avgState(toFloat64(govalue_score))      AS avg_govalue_state,
    avgState(toFloat64(domain_age_days))    AS avg_age_state
FROM signals_platform_cln.auction_audit_cln;


-- mv_auctions_by_type_hour  — hourly auction-type aggregates
CREATE TABLE IF NOT EXISTS signals_platform_cln.mv_auctions_by_type_hour
(
    auction_type_id     UInt8,
    event_hour          DateTime,
    count_state         AggregateFunction(count),
    avg_price_state     AggregateFunction(avg, Float64),
    min_price_state     AggregateFunction(min, Float64),
    max_price_state     AggregateFunction(max, Float64),
    sum_price_state     AggregateFunction(sum, Float64),
    median_price_state  AggregateFunction(quantiles(0.5), Float64),
    avg_bids_state      AggregateFunction(avg, Float64),
    sum_bids_state      AggregateFunction(sum, Float64),
    avg_govalue_state   AggregateFunction(avg, Float32),
    avg_age_state       AggregateFunction(avg, Float64)
)
ENGINE = AggregatingMergeTree()
ORDER BY (auction_type_id, event_hour);

CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_auctions_by_type_hour_mv
TO signals_platform_cln.mv_auctions_by_type_hour
AS SELECT
    auction_type_id,
    toStartOfHour(created_at)               AS event_hour,
    countState()                            AS count_state,
    avgState(current_price)                 AS avg_price_state,
    minState(current_price)                 AS min_price_state,
    maxState(current_price)                 AS max_price_state,
    sumState(current_price)                 AS sum_price_state,
    quantilesState(0.5)(current_price)      AS median_price_state,
    avgState(toFloat64(bid_count))          AS avg_bids_state,
    sumState(toFloat64(bid_count))          AS sum_bids_state,
    avgState(toFloat64(govalue_score))      AS avg_govalue_state,
    avgState(toFloat64(domain_age_days))    AS avg_age_state
FROM signals_platform_cln.auction_audit_cln;


-- mv_auctions_by_type_week  — weekly auction-type aggregates
CREATE TABLE IF NOT EXISTS signals_platform_cln.mv_auctions_by_type_week
(
    auction_type_id     UInt8,
    event_week          Date,
    count_state         AggregateFunction(count),
    avg_price_state     AggregateFunction(avg, Float64),
    min_price_state     AggregateFunction(min, Float64),
    max_price_state     AggregateFunction(max, Float64),
    sum_price_state     AggregateFunction(sum, Float64),
    median_price_state  AggregateFunction(quantiles(0.5), Float64),
    avg_bids_state      AggregateFunction(avg, Float64),
    sum_bids_state      AggregateFunction(sum, Float64),
    avg_govalue_state   AggregateFunction(avg, Float32),
    avg_age_state       AggregateFunction(avg, Float64)
)
ENGINE = AggregatingMergeTree()
ORDER BY (auction_type_id, event_week);

CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_auctions_by_type_week_mv
TO signals_platform_cln.mv_auctions_by_type_week
AS SELECT
    auction_type_id,
    toMonday(created_at)                    AS event_week,
    countState()                            AS count_state,
    avgState(current_price)                 AS avg_price_state,
    minState(current_price)                 AS min_price_state,
    maxState(current_price)                 AS max_price_state,
    sumState(current_price)                 AS sum_price_state,
    quantilesState(0.5)(current_price)      AS median_price_state,
    avgState(toFloat64(bid_count))          AS avg_bids_state,
    sumState(toFloat64(bid_count))          AS sum_bids_state,
    avgState(toFloat64(govalue_score))      AS avg_govalue_state,
    avgState(toFloat64(domain_age_days))    AS avg_age_state
FROM signals_platform_cln.auction_audit_cln;


-- mv_auctions_by_type_tld_hour  — hourly (tld x type) composite
CREATE TABLE IF NOT EXISTS signals_platform_cln.mv_auctions_by_type_tld_hour
(
    tld                 LowCardinality(String),
    auction_type_id     UInt8,
    event_hour          DateTime,
    count_state         AggregateFunction(count),
    avg_price_state     AggregateFunction(avg, Float64),
    min_price_state     AggregateFunction(min, Float64),
    max_price_state     AggregateFunction(max, Float64),
    sum_price_state     AggregateFunction(sum, Float64),
    median_price_state  AggregateFunction(quantiles(0.5), Float64),
    avg_bids_state      AggregateFunction(avg, Float64),
    sum_bids_state      AggregateFunction(sum, Float64),
    avg_govalue_state   AggregateFunction(avg, Float32),
    avg_age_state       AggregateFunction(avg, Float64)
)
ENGINE = AggregatingMergeTree()
ORDER BY (tld, auction_type_id, event_hour);

CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_auctions_by_type_tld_hour_mv
TO signals_platform_cln.mv_auctions_by_type_tld_hour
AS SELECT
    tld, auction_type_id,
    toStartOfHour(created_at)               AS event_hour,
    countState()                            AS count_state,
    avgState(current_price)                 AS avg_price_state,
    minState(current_price)                 AS min_price_state,
    maxState(current_price)                 AS max_price_state,
    sumState(current_price)                 AS sum_price_state,
    quantilesState(0.5)(current_price)      AS median_price_state,
    avgState(toFloat64(bid_count))          AS avg_bids_state,
    sumState(toFloat64(bid_count))          AS sum_bids_state,
    avgState(toFloat64(govalue_score))      AS avg_govalue_state,
    avgState(toFloat64(domain_age_days))    AS avg_age_state
FROM signals_platform_cln.auction_audit_cln;


-- mv_auctions_by_type_tld_day  — daily (tld x type) composite
CREATE TABLE IF NOT EXISTS signals_platform_cln.mv_auctions_by_type_tld_day
(
    tld                 LowCardinality(String),
    auction_type_id     UInt8,
    event_day           Date,
    count_state         AggregateFunction(count),
    avg_price_state     AggregateFunction(avg, Float64),
    min_price_state     AggregateFunction(min, Float64),
    max_price_state     AggregateFunction(max, Float64),
    sum_price_state     AggregateFunction(sum, Float64),
    median_price_state  AggregateFunction(quantiles(0.5), Float64),
    avg_bids_state      AggregateFunction(avg, Float64),
    sum_bids_state      AggregateFunction(sum, Float64),
    avg_govalue_state   AggregateFunction(avg, Float32),
    avg_age_state       AggregateFunction(avg, Float64)
)
ENGINE = AggregatingMergeTree()
ORDER BY (tld, auction_type_id, event_day);

CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_auctions_by_type_tld_day_mv
TO signals_platform_cln.mv_auctions_by_type_tld_day
AS SELECT
    tld, auction_type_id,
    toDate(created_at)                      AS event_day,
    countState()                            AS count_state,
    avgState(current_price)                 AS avg_price_state,
    minState(current_price)                 AS min_price_state,
    maxState(current_price)                 AS max_price_state,
    sumState(current_price)                 AS sum_price_state,
    quantilesState(0.5)(current_price)      AS median_price_state,
    avgState(toFloat64(bid_count))          AS avg_bids_state,
    sumState(toFloat64(bid_count))          AS sum_bids_state,
    avgState(toFloat64(govalue_score))      AS avg_govalue_state,
    avgState(toFloat64(domain_age_days))    AS avg_age_state
FROM signals_platform_cln.auction_audit_cln;


-- mv_auctions_growth_week  — weekly (tld x type) growth/trend
CREATE TABLE IF NOT EXISTS signals_platform_cln.mv_auctions_growth_week
(
    tld                 LowCardinality(String),
    auction_type_id     UInt8,
    event_week          Date,
    count_state         AggregateFunction(count),
    avg_price_state     AggregateFunction(avg, Float64),
    min_price_state     AggregateFunction(min, Float64),
    max_price_state     AggregateFunction(max, Float64),
    sum_price_state     AggregateFunction(sum, Float64),
    median_price_state  AggregateFunction(quantiles(0.5), Float64),
    avg_bids_state      AggregateFunction(avg, Float64),
    sum_bids_state      AggregateFunction(sum, Float64),
    avg_govalue_state   AggregateFunction(avg, Float32),
    avg_age_state       AggregateFunction(avg, Float64)
)
ENGINE = AggregatingMergeTree()
ORDER BY (tld, auction_type_id, event_week);

CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_auctions_growth_week_mv
TO signals_platform_cln.mv_auctions_growth_week
AS SELECT
    tld, auction_type_id,
    toMonday(created_at)                    AS event_week,
    countState()                            AS count_state,
    avgState(current_price)                 AS avg_price_state,
    minState(current_price)                 AS min_price_state,
    maxState(current_price)                 AS max_price_state,
    sumState(current_price)                 AS sum_price_state,
    quantilesState(0.5)(current_price)      AS median_price_state,
    avgState(toFloat64(bid_count))          AS avg_bids_state,
    sumState(toFloat64(bid_count))          AS sum_bids_state,
    avgState(toFloat64(govalue_score))      AS avg_govalue_state,
    avgState(toFloat64(domain_age_days))    AS avg_age_state
FROM signals_platform_cln.auction_audit_cln;


-- mv_auctions_by_category_day  — daily (category integer type x tld)
CREATE TABLE IF NOT EXISTS signals_platform_cln.mv_auctions_by_category_day
(
    category            UInt8,
    tld                 LowCardinality(String),
    event_day           Date,
    count_state         AggregateFunction(count),
    avg_price_state     AggregateFunction(avg, Float64),
    min_price_state     AggregateFunction(min, Float64),
    max_price_state     AggregateFunction(max, Float64),
    sum_price_state     AggregateFunction(sum, Float64),
    median_price_state  AggregateFunction(quantiles(0.5), Float64),
    avg_bids_state      AggregateFunction(avg, Float64),
    sum_bids_state      AggregateFunction(sum, Float64),
    avg_govalue_state   AggregateFunction(avg, Float32),
    avg_age_state       AggregateFunction(avg, Float64)
)
ENGINE = AggregatingMergeTree()
ORDER BY (category, tld, event_day);

CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_auctions_by_category_day_mv
TO signals_platform_cln.mv_auctions_by_category_day
AS SELECT
    auction_type_id                         AS category,
    tld,
    toDate(created_at)                      AS event_day,
    countState()                            AS count_state,
    avgState(current_price)                 AS avg_price_state,
    minState(current_price)                 AS min_price_state,
    maxState(current_price)                 AS max_price_state,
    sumState(current_price)                 AS sum_price_state,
    quantilesState(0.5)(current_price)      AS median_price_state,
    avgState(toFloat64(bid_count))          AS avg_bids_state,
    sumState(toFloat64(bid_count))          AS sum_bids_state,
    avgState(toFloat64(govalue_score))      AS avg_govalue_state,
    avgState(toFloat64(domain_age_days))    AS avg_age_state
FROM signals_platform_cln.auction_audit_cln;


-- mv_auctions_by_category_name_day  — daily human-readable category aggregates
CREATE TABLE IF NOT EXISTS signals_platform_cln.mv_auctions_by_category_name_day
(
    category_name       LowCardinality(String),
    event_day           Date,
    count_state         AggregateFunction(count),
    avg_price_state     AggregateFunction(avg, Float64),
    min_price_state     AggregateFunction(min, Float64),
    max_price_state     AggregateFunction(max, Float64),
    sum_price_state     AggregateFunction(sum, Float64),
    median_price_state  AggregateFunction(quantiles(0.5), Float64),
    avg_bids_state      AggregateFunction(avg, Float64),
    sum_bids_state      AggregateFunction(sum, Float64),
    avg_authority_state AggregateFunction(avg, Float32),
    avg_traffic_state   AggregateFunction(avg, Float64)
)
ENGINE = AggregatingMergeTree()
ORDER BY (category_name, event_day);

CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_auctions_by_category_name_day_mv
TO signals_platform_cln.mv_auctions_by_category_name_day
AS SELECT
    category_name,
    toDate(created_at)                          AS event_day,
    countState()                                AS count_state,
    avgState(current_price)                     AS avg_price_state,
    minState(current_price)                     AS min_price_state,
    maxState(current_price)                     AS max_price_state,
    sumState(current_price)                     AS sum_price_state,
    quantilesState(0.5)(current_price)          AS median_price_state,
    avgState(toFloat64(bid_count))              AS avg_bids_state,
    sumState(toFloat64(bid_count))              AS sum_bids_state,
    avgState(toFloat64(domain_authority))       AS avg_authority_state,
    avgState(monthly_traffic)                   AS avg_traffic_state
FROM signals_platform_cln.auction_audit_cln;


-- mv_auctions_by_category_name_week  — weekly human-readable category aggregates
CREATE TABLE IF NOT EXISTS signals_platform_cln.mv_auctions_by_category_name_week
(
    category_name       LowCardinality(String),
    event_week          Date,
    count_state         AggregateFunction(count),
    avg_price_state     AggregateFunction(avg, Float64),
    min_price_state     AggregateFunction(min, Float64),
    max_price_state     AggregateFunction(max, Float64),
    sum_price_state     AggregateFunction(sum, Float64),
    median_price_state  AggregateFunction(quantiles(0.5), Float64),
    avg_bids_state      AggregateFunction(avg, Float64),
    sum_bids_state      AggregateFunction(sum, Float64),
    avg_authority_state AggregateFunction(avg, Float32),
    avg_traffic_state   AggregateFunction(avg, Float64),
    avg_age_state       AggregateFunction(avg, Float64)
)
ENGINE = AggregatingMergeTree()
ORDER BY (category_name, event_week);

CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_auctions_by_category_name_week_mv
TO signals_platform_cln.mv_auctions_by_category_name_week
AS SELECT
    category_name,
    toMonday(created_at)                         AS event_week,
    countState()                                 AS count_state,
    avgState(current_price)                      AS avg_price_state,
    minState(current_price)                      AS min_price_state,
    maxState(current_price)                      AS max_price_state,
    sumState(current_price)                      AS sum_price_state,
    quantilesState(0.5)(current_price)           AS median_price_state,
    avgState(toFloat64(bid_count))               AS avg_bids_state,
    sumState(toFloat64(bid_count))               AS sum_bids_state,
    avgState(toFloat64(domain_authority))        AS avg_authority_state,
    avgState(monthly_traffic)                    AS avg_traffic_state,
    avgState(toFloat64(domain_age_days))         AS avg_age_state
FROM signals_platform_cln.auction_audit_cln;

-- mv_sold_by_category_name_week  — weekly sold-only category aggregates
CREATE TABLE IF NOT EXISTS signals_platform_cln.mv_sold_by_category_name_week
(
    category_name       LowCardinality(String),
    event_week          Date,
    count_state         AggregateFunction(count),
    avg_price_state     AggregateFunction(avg, Float64),
    min_price_state     AggregateFunction(min, Float64),
    max_price_state     AggregateFunction(max, Float64),
    sum_price_state     AggregateFunction(sum, Float64),
    median_price_state  AggregateFunction(quantiles(0.5), Float64),
    avg_authority_state AggregateFunction(avg, Float32),
    avg_traffic_state   AggregateFunction(avg, Float64),
    avg_bids_state      AggregateFunction(avg, Float64),
    sum_bids_state      AggregateFunction(sum, Float64),
    avg_govalue_state   AggregateFunction(avg, Float32),
    avg_age_state       AggregateFunction(avg, Float64)
)
ENGINE = AggregatingMergeTree()
ORDER BY (category_name, event_week);

CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_sold_by_category_name_week_mv
TO signals_platform_cln.mv_sold_by_category_name_week
AS SELECT
    category_name,
    toMonday(sold_at)                            AS event_week,
    countState()                                 AS count_state,
    avgState(current_price)                      AS avg_price_state,
    minState(current_price)                      AS min_price_state,
    maxState(current_price)                      AS max_price_state,
    sumState(current_price)                      AS sum_price_state,
    quantilesState(0.5)(current_price)           AS median_price_state,
    avgState(toFloat64(domain_authority))        AS avg_authority_state,
    avgState(monthly_traffic)                    AS avg_traffic_state,
    avgState(toFloat64(bid_count))               AS avg_bids_state,
    sumState(toFloat64(bid_count))               AS sum_bids_state,
    avgState(toFloat64(govalue_score))           AS avg_govalue_state,
    avgState(toFloat64(domain_age_days))         AS avg_age_state
FROM signals_platform_cln.auction_audit_cln
WHERE sold_flag = 1 AND sold_at IS NOT NULL;


-- mv_sell_through_by_tld_day  — TLD sell-through rate (total + sold counts)
CREATE TABLE IF NOT EXISTS signals_platform_cln.mv_sell_through_by_tld_day
(
    tld                 LowCardinality(String),
    event_day           Date,
    count_state         AggregateFunction(count),
    avg_price_state     AggregateFunction(avg, Float64),
    sum_price_state     AggregateFunction(sum, Float64),
    avg_bids_state      AggregateFunction(avg, Float64)
)
ENGINE = AggregatingMergeTree()
ORDER BY (tld, event_day);

CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_sell_through_by_tld_day_mv
TO signals_platform_cln.mv_sell_through_by_tld_day
AS SELECT
    tld,
    toDate(created_at)                      AS event_day,
    countState()                            AS count_state,
    avgState(current_price)                 AS avg_price_state,
    sumState(current_price)                 AS sum_price_state,
    avgState(toFloat64(bid_count))          AS avg_bids_state
FROM signals_platform_cln.auction_audit_cln;


-- mv_auctions_by_registrar_day  — registrar volume and authority
CREATE TABLE IF NOT EXISTS signals_platform_cln.mv_auctions_by_registrar_day
(
    registrar_name      LowCardinality(String),
    event_day           Date,
    count_state         AggregateFunction(count),
    avg_price_state     AggregateFunction(avg, Float64),
    sum_price_state     AggregateFunction(sum, Float64),
    avg_authority_state AggregateFunction(avg, Float32),
    avg_bids_state      AggregateFunction(avg, Float64)
)
ENGINE = AggregatingMergeTree()
ORDER BY (registrar_name, event_day);

CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_auctions_by_registrar_day_mv
TO signals_platform_cln.mv_auctions_by_registrar_day
AS SELECT
    registrar_name,
    toDate(created_at)                          AS event_day,
    countState()                                AS count_state,
    avgState(current_price)                     AS avg_price_state,
    sumState(current_price)                     AS sum_price_state,
    avgState(toFloat64(domain_authority))       AS avg_authority_state,
    avgState(toFloat64(bid_count))              AS avg_bids_state
FROM signals_platform_cln.auction_audit_cln;


-- mv_expiry_lifecycle_day  — expiry status breakdown by day
CREATE TABLE IF NOT EXISTS signals_platform_cln.mv_expiry_lifecycle_day
(
    expiry_status       LowCardinality(String),
    event_day           Date,
    count_state         AggregateFunction(count),
    avg_price_state     AggregateFunction(avg, Float64),
    min_price_state     AggregateFunction(min, Float64),
    max_price_state     AggregateFunction(max, Float64),
    sum_price_state     AggregateFunction(sum, Float64),
    avg_bids_state      AggregateFunction(avg, Float64),
    avg_authority_state AggregateFunction(avg, Float32)
)
ENGINE = AggregatingMergeTree()
ORDER BY (expiry_status, event_day);

CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_expiry_lifecycle_day_mv
TO signals_platform_cln.mv_expiry_lifecycle_day
AS SELECT
    expiry_status,
    toDate(created_at)                          AS event_day,
    countState()                                AS count_state,
    avgState(current_price)                     AS avg_price_state,
    minState(current_price)                     AS min_price_state,
    maxState(current_price)                     AS max_price_state,
    sumState(current_price)                     AS sum_price_state,
    avgState(toFloat64(bid_count))              AS avg_bids_state,
    avgState(toFloat64(domain_authority))       AS avg_authority_state
FROM signals_platform_cln.auction_audit_cln;


-- mv_sell_through_by_type_day  — sell-through by auction type name
CREATE TABLE IF NOT EXISTS signals_platform_cln.mv_sell_through_by_type_day
(
    auction_type_name   LowCardinality(String),
    event_day           Date,
    count_state         AggregateFunction(count),
    avg_price_state     AggregateFunction(avg, Float64),
    sum_price_state     AggregateFunction(sum, Float64),
    avg_bids_state      AggregateFunction(avg, Float64)
)
ENGINE = AggregatingMergeTree()
ORDER BY (auction_type_name, event_day);

CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_sell_through_by_type_day_mv
TO signals_platform_cln.mv_sell_through_by_type_day
AS SELECT
    auction_type_name,
    toDate(created_at)                      AS event_day,
    countState()                            AS count_state,
    avgState(current_price)                 AS avg_price_state,
    sumState(current_price)                 AS sum_price_state,
    avgState(toFloat64(bid_count))          AS avg_bids_state
FROM signals_platform_cln.auction_audit_cln;


-- mv_auctions_by_type_name_day  — full stats by auction type name per day
CREATE TABLE IF NOT EXISTS signals_platform_cln.mv_auctions_by_type_name_day
(
    auction_type_name       LowCardinality(String),
    event_day               Date,
    count_state             AggregateFunction(count),
    avg_price_state         AggregateFunction(avg, Float64),
    min_price_state         AggregateFunction(min, Float64),
    max_price_state         AggregateFunction(max, Float64),
    sum_price_state         AggregateFunction(sum, Float64),
    median_price_state      AggregateFunction(quantiles(0.5), Float64),
    avg_traffic_state       AggregateFunction(avg, Float64),
    avg_authority_state     AggregateFunction(avg, Float32),
    avg_backlinks_state     AggregateFunction(avg, Float64),
    avg_bids_state          AggregateFunction(avg, Float64),
    sum_bids_state          AggregateFunction(sum, Float64),
    avg_govalue_state       AggregateFunction(avg, Float32),
    avg_age_state           AggregateFunction(avg, Float64),
    avg_hold_days_state     AggregateFunction(avg, Float64)
)
ENGINE = AggregatingMergeTree()
ORDER BY (auction_type_name, event_day);

CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_auctions_by_type_name_day_mv
TO signals_platform_cln.mv_auctions_by_type_name_day
AS SELECT
    auction_type_name,
    toDate(created_at)                          AS event_day,
    countState()                                AS count_state,
    avgState(current_price)                     AS avg_price_state,
    minState(current_price)                     AS min_price_state,
    maxState(current_price)                     AS max_price_state,
    sumState(current_price)                     AS sum_price_state,
    quantilesState(0.5)(current_price)          AS median_price_state,
    avgState(monthly_traffic)                   AS avg_traffic_state,
    avgState(toFloat64(domain_authority))       AS avg_authority_state,
    avgState(toFloat64(backlink_count))         AS avg_backlinks_state,
    avgState(toFloat64(bid_count))              AS avg_bids_state,
    sumState(toFloat64(bid_count))              AS sum_bids_state,
    avgState(toFloat64(govalue_score))          AS avg_govalue_state,
    avgState(toFloat64(domain_age_days))        AS avg_age_state,
    avgState(hold_days)                         AS avg_hold_days_state
FROM signals_platform_cln.auction_audit_cln;


-- mv_hold_time_by_type_day  — hold time for sold domains (auction_type_name x tld)
CREATE TABLE IF NOT EXISTS signals_platform_cln.mv_hold_time_by_type_day
(
    auction_type_name   LowCardinality(String),
    tld                 LowCardinality(String),
    event_day           Date,
    count_state         AggregateFunction(count),
    avg_hold_days_state AggregateFunction(avg, Float64),
    min_hold_days_state AggregateFunction(min, Float64),
    max_hold_days_state AggregateFunction(max, Float64),
    avg_price_state     AggregateFunction(avg, Float64),
    sum_price_state     AggregateFunction(sum, Float64)
)
ENGINE = AggregatingMergeTree()
ORDER BY (auction_type_name, tld, event_day);

CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_hold_time_by_type_day_mv
TO signals_platform_cln.mv_hold_time_by_type_day
AS SELECT
    auction_type_name,
    tld,
    toDate(sold_at)                         AS event_day,
    countState()                            AS count_state,
    avgState(hold_days)                     AS avg_hold_days_state,
    minState(hold_days)                     AS min_hold_days_state,
    maxState(hold_days)                     AS max_hold_days_state,
    avgState(current_price)                 AS avg_price_state,
    sumState(current_price)                 AS sum_price_state
FROM signals_platform_cln.auction_audit_cln
WHERE sold_flag = 1 AND sold_at IS NOT NULL;


-- mv_sold_by_tld_week  — weekly sold aggregation by TLD
CREATE TABLE IF NOT EXISTS signals_platform_cln.mv_sold_by_tld_week
(
    tld                 LowCardinality(String),
    event_week          Date,
    count_state         AggregateFunction(count),
    avg_price_state     AggregateFunction(avg, Float64),
    min_price_state     AggregateFunction(min, Float64),
    max_price_state     AggregateFunction(max, Float64),
    sum_price_state     AggregateFunction(sum, Float64),
    median_price_state  AggregateFunction(quantiles(0.5), Float64),
    avg_authority_state AggregateFunction(avg, Float32),
    avg_traffic_state   AggregateFunction(avg, Float64),
    avg_bids_state      AggregateFunction(avg, Float64),
    sum_bids_state      AggregateFunction(sum, Float64),
    avg_govalue_state   AggregateFunction(avg, Float32),
    avg_age_state       AggregateFunction(avg, Float64)
)
ENGINE = AggregatingMergeTree()
ORDER BY (tld, event_week);

CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_sold_by_tld_week_mv
TO signals_platform_cln.mv_sold_by_tld_week
AS SELECT
    tld,
    toMonday(sold_at)                           AS event_week,
    countState()                                AS count_state,
    avgState(current_price)                     AS avg_price_state,
    minState(current_price)                     AS min_price_state,
    maxState(current_price)                     AS max_price_state,
    sumState(current_price)                     AS sum_price_state,
    quantilesState(0.5)(current_price)          AS median_price_state,
    avgState(toFloat64(domain_authority))       AS avg_authority_state,
    avgState(monthly_traffic)                   AS avg_traffic_state,
    avgState(toFloat64(bid_count))              AS avg_bids_state,
    sumState(toFloat64(bid_count))              AS sum_bids_state,
    avgState(toFloat64(govalue_score))          AS avg_govalue_state,
    avgState(toFloat64(domain_age_days))        AS avg_age_state
FROM signals_platform_cln.auction_audit_cln
WHERE sold_flag = 1 AND sold_at IS NOT NULL;


-- ============================================================
-- mv_auctions_by_buyer_segment_day  — daily auction activity + sell-through by buyer segment
-- Covers Beginner / Professional / Advanced / Investor analytics (Q43-Q60).
-- Requires buyer_segment column populated by the buyer-profile pipeline.
CREATE TABLE IF NOT EXISTS signals_platform_cln.mv_auctions_by_buyer_segment_day
(
    buyer_segment       LowCardinality(String),
    event_day           Date,
    count_state         AggregateFunction(count),
    sold_count_state    AggregateFunction(sum, Float64),
    avg_price_state     AggregateFunction(avg, Float64),
    min_price_state     AggregateFunction(min, Float64),
    max_price_state     AggregateFunction(max, Float64),
    sum_price_state     AggregateFunction(sum, Float64),
    avg_bids_state      AggregateFunction(avg, Float64),
    avg_authority_state AggregateFunction(avg, Float32),
    avg_traffic_state   AggregateFunction(avg, Float64)
)
ENGINE = AggregatingMergeTree()
ORDER BY (buyer_segment, event_day);

CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_auctions_by_buyer_segment_day_mv
TO signals_platform_cln.mv_auctions_by_buyer_segment_day
AS SELECT
    buyer_segment,
    toDate(created_at)                          AS event_day,
    countState()                                AS count_state,
    sumState(toFloat64(sold_flag))              AS sold_count_state,
    avgState(current_price)                     AS avg_price_state,
    minState(current_price)                     AS min_price_state,
    maxState(current_price)                     AS max_price_state,
    sumState(current_price)                     AS sum_price_state,
    avgState(toFloat64(bid_count))              AS avg_bids_state,
    avgState(toFloat64(domain_authority))       AS avg_authority_state,
    avgState(monthly_traffic)                   AS avg_traffic_state
FROM signals_platform_cln.auction_audit_cln
WHERE buyer_segment != '';


-- mv_auctions_by_buyer_segment_week  — weekly rollup (powers 12-week trend queries)
CREATE TABLE IF NOT EXISTS signals_platform_cln.mv_auctions_by_buyer_segment_week
(
    buyer_segment       LowCardinality(String),
    event_week          Date,
    count_state         AggregateFunction(count),
    sold_count_state    AggregateFunction(sum, Float64),
    avg_price_state     AggregateFunction(avg, Float64),
    min_price_state     AggregateFunction(min, Float64),
    max_price_state     AggregateFunction(max, Float64),
    sum_price_state     AggregateFunction(sum, Float64),
    avg_bids_state      AggregateFunction(avg, Float64),
    avg_authority_state AggregateFunction(avg, Float32),
    avg_traffic_state   AggregateFunction(avg, Float64)
)
ENGINE = AggregatingMergeTree()
ORDER BY (buyer_segment, event_week);

CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_auctions_by_buyer_segment_week_mv
TO signals_platform_cln.mv_auctions_by_buyer_segment_week
AS SELECT
    buyer_segment,
    toMonday(created_at)                        AS event_week,
    countState()                                AS count_state,
    sumState(toFloat64(sold_flag))              AS sold_count_state,
    avgState(current_price)                     AS avg_price_state,
    minState(current_price)                     AS min_price_state,
    maxState(current_price)                     AS max_price_state,
    sumState(current_price)                     AS sum_price_state,
    avgState(toFloat64(bid_count))              AS avg_bids_state,
    avgState(toFloat64(domain_authority))       AS avg_authority_state,
    avgState(monthly_traffic)                   AS avg_traffic_state
FROM signals_platform_cln.auction_audit_cln
WHERE buyer_segment != '';


-- ============================================================
-- Bid events (from the_resale_place.item_bids_cln)
-- Source: ~97M rows real-time EventBus stream; Phase 1.5 ingest target.
-- auction_id populated via item_winning_bids_cln join (best-effort; 0 when unmatched).
-- Human-bid filter: bid_source = 'auc-bidding' (confirm value via SQL diagnostic #B1).
-- ============================================================
CREATE TABLE IF NOT EXISTS signals_platform_cln.bid_events
(
    bid_event_id     String,
    member_item_id   Int64,
    auction_id       Int64                      DEFAULT 0,
    bidder_id        Int64                      DEFAULT 0,
    seller_id        Int64                      DEFAULT 0,
    bid_usd_amount   Float64                    DEFAULT 0.0,
    bid_source       LowCardinality(String)     DEFAULT '',
    buy_it_now_flag  UInt8                      DEFAULT 0,
    bid_accepted     UInt8                      DEFAULT 0,
    counter_offer    UInt8                      DEFAULT 0,
    bid_start_at     DateTime64(3),
    bid_end_at       Nullable(DateTime64(3))    DEFAULT NULL,
    event_utc_ts     DateTime64(3),
    received_at      DateTime64(3)              DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(received_at)
PARTITION BY toYYYYMM(event_utc_ts)
ORDER BY (member_item_id, bid_start_at)
TTL toDateTime(event_utc_ts) + INTERVAL 120 DAY;


-- mv_bid_velocity_by_auction_hour  — hourly bid velocity per auction (trending-now MV)
-- Keys on auction_id when populated; used by trending-now explore rail and ending-soon ranker.
-- Filter: buy_it_now_flag = 0 AND bid_source = 'auc-bidding' for human auction bids only.
CREATE TABLE IF NOT EXISTS signals_platform_cln.mv_bid_velocity_by_auction_hour
(
    auction_id          Int64,
    event_hour          DateTime,
    bid_count_state     AggregateFunction(count),
    sum_bid_state       AggregateFunction(sum, Float64),
    max_bid_state       AggregateFunction(max, Float64),
    unique_bidders_state AggregateFunction(uniq, Int64)
)
ENGINE = AggregatingMergeTree()
ORDER BY (auction_id, event_hour);

CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_bid_velocity_by_auction_hour_mv
TO signals_platform_cln.mv_bid_velocity_by_auction_hour
AS SELECT
    auction_id,
    toStartOfHour(event_utc_ts)      AS event_hour,
    countState()                     AS bid_count_state,
    sumState(bid_usd_amount)         AS sum_bid_state,
    maxState(bid_usd_amount)         AS max_bid_state,
    uniqState(bidder_id)             AS unique_bidders_state
FROM signals_platform_cln.bid_events
WHERE auction_id > 0
  AND buy_it_now_flag = 0;


-- mv_bid_velocity_by_item_hour  — hourly bid velocity per member_item_id
-- Keys on member_item_id (always populated, even when auction_id join is 0).
-- Powers trending-now velocity when auction_id bridge is incomplete.
CREATE TABLE IF NOT EXISTS signals_platform_cln.mv_bid_velocity_by_item_hour
(
    member_item_id      Int64,
    event_hour          DateTime,
    bid_count_state     AggregateFunction(count),
    sum_bid_state       AggregateFunction(sum, Float64),
    max_bid_state       AggregateFunction(max, Float64),
    unique_bidders_state AggregateFunction(uniq, Int64)
)
ENGINE = AggregatingMergeTree()
ORDER BY (member_item_id, event_hour);

CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_bid_velocity_by_item_hour_mv
TO signals_platform_cln.mv_bid_velocity_by_item_hour
AS SELECT
    member_item_id,
    toStartOfHour(event_utc_ts)      AS event_hour,
    countState()                     AS bid_count_state,
    sumState(bid_usd_amount)         AS sum_bid_state,
    maxState(bid_usd_amount)         AS max_bid_state,
    uniqState(bidder_id)             AS unique_bidders_state
FROM signals_platform_cln.bid_events
WHERE buy_it_now_flag = 0;


-- ============================================================
-- Watch events (from the_resale_place.member_items_watch_cln)
-- Source: real-time EventBus stream; engagement signal for ranking.
-- watch_type_label populated from member_items_watch_types_cln dimension at ingest time.
-- ============================================================
CREATE TABLE IF NOT EXISTS signals_platform_cln.watch_events
(
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
)
ENGINE = ReplacingMergeTree(received_at)
PARTITION BY toYYYYMM(event_utc_ts)
ORDER BY (member_item_id, watch_event_id)
TTL toDateTime(event_utc_ts) + INTERVAL 120 DAY;


-- mv_watch_density_by_item_day  — daily active-watch count per listing
-- active_watch_state = watches where is_deleted = 0 (net watchlist adds).
-- unique_watchers_state = distinct member_id where member_id > 0 (guards against unpopulated field).
-- Used as engagement ranking signal alongside bid velocity.
CREATE TABLE IF NOT EXISTS signals_platform_cln.mv_watch_density_by_item_day
(
    member_item_id          Int64,
    event_day               Date,
    active_watch_state      AggregateFunction(count),
    unique_watchers_state   AggregateFunction(uniq, Int64)
)
ENGINE = AggregatingMergeTree()
ORDER BY (member_item_id, event_day);

CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_watch_density_by_item_day_mv
TO signals_platform_cln.mv_watch_density_by_item_day
AS SELECT
    member_item_id,
    toDate(event_utc_ts)             AS event_day,
    countStateIf(is_deleted = 0)     AS active_watch_state,
    uniqStateIf(member_id, member_id > 0) AS unique_watchers_state
FROM signals_platform_cln.watch_events
WHERE watch_type IN (1, 9)
GROUP BY member_item_id, event_day;


-- mv_bidder_watch_density_by_item_day — type-9 ("Items I am bidding on") watches only.
-- Stronger purchase-intent signal than passive type-1 watches.
-- Kept separate so consumers can weight bidder intent independently from passive interest.
CREATE TABLE IF NOT EXISTS signals_platform_cln.mv_bidder_watch_density_by_item_day
(
    member_item_id              Int64,
    event_day                   Date,
    bidder_watch_state          AggregateFunction(count),
    unique_bidder_watchers_state AggregateFunction(uniq, Int64)
)
ENGINE = AggregatingMergeTree()
ORDER BY (member_item_id, event_day);

CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_bidder_watch_density_by_item_day_mv
TO signals_platform_cln.mv_bidder_watch_density_by_item_day
AS SELECT
    member_item_id,
    toDate(event_utc_ts)                        AS event_day,
    countStateIf(is_deleted = 0)                AS bidder_watch_state,
    uniqStateIf(member_id, member_id > 0)       AS unique_bidder_watchers_state
FROM signals_platform_cln.watch_events
WHERE watch_type = 9
GROUP BY member_item_id, event_day;


-- mv_unique_bidder_count_by_item — competitive depth: distinct human bidders per listing per hour.
-- Filters bid_source = 'auc-bidding' to exclude automated pricing bots.
-- Bridge to auction_id happens at query time via bid_events.
CREATE TABLE IF NOT EXISTS signals_platform_cln.mv_unique_bidder_count_by_item
(
    member_item_id       Int64,
    event_hour           DateTime,
    unique_bidder_state  AggregateFunction(uniq, Int64)
)
ENGINE = AggregatingMergeTree()
ORDER BY (member_item_id, event_hour);

CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_unique_bidder_count_by_item_mv
TO signals_platform_cln.mv_unique_bidder_count_by_item
AS SELECT
    member_item_id,
    toStartOfHour(event_utc_ts)                          AS event_hour,
    uniqStateIf(bidder_id, bid_source = 'auc-bidding')   AS unique_bidder_state
FROM signals_platform_cln.bid_events
GROUP BY member_item_id, event_hour;


-- mv_user_engagement_day  — daily user engagement (DAU / depth)
CREATE TABLE IF NOT EXISTS signals_platform_cln.mv_user_engagement_day
(
    user_id             UInt64,
    event_day           Date,
    count_state         AggregateFunction(count),
    avg_price_state     AggregateFunction(avg, Float64),
    sum_price_state     AggregateFunction(sum, Float64),
    avg_bids_state      AggregateFunction(avg, Float64),
    avg_govalue_state   AggregateFunction(avg, Float32),
    avg_age_state       AggregateFunction(avg, Float64)
)
ENGINE = AggregatingMergeTree()
ORDER BY (user_id, event_day);

CREATE MATERIALIZED VIEW IF NOT EXISTS signals_platform_cln.mv_user_engagement_day_mv
TO signals_platform_cln.mv_user_engagement_day
AS SELECT
    user_id,
    toDate(created_at)                      AS event_day,
    countState()                            AS count_state,
    avgState(current_price)                 AS avg_price_state,
    sumState(current_price)                 AS sum_price_state,
    avgState(toFloat64(bid_count))          AS avg_bids_state,
    avgState(toFloat64(govalue_score))      AS avg_govalue_state,
    avgState(toFloat64(domain_age_days))    AS avg_age_state
FROM signals_platform_cln.auction_audit_cln
WHERE user_id > 0;
