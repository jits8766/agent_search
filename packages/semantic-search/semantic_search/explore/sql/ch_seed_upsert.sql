-- ch_seed_upsert.sql
--
-- Reference DDL/INSERT for the ClickHouse side of the seed pipeline. Style
-- mirrors offline_eval/offline/src/data_pipeline/conversation_metrics/sql/conv_trace_metrics.sql
-- and this package's own vectorization/sql/seed_merge_semsearch.sql.
--
-- This file is a human-reviewable, non-executing mirror of what
-- explore/ch_seed_writer.py actually runs (schema DDL sourced verbatim from
-- analytics/ch_schema.py's get_ddl_statements(), INSERT built by
-- _build_insert_payload()). It documents ONLY the seed-upsert path — the
-- full analytics schema (materialized views, feedback_signals, etc.) lives
-- separately in analytics/clickhouse_schema.sql and is not duplicated here.
--
-- All join/enrichment logic happens upstream, on the Athena side (see
-- vectorization/sql/seed_merge_semsearch.sql). By the time rows reach
-- ClickHouse they are already fully computed — this is a plain upsert of
-- <TMP_DB>.final_<TABLE>_<RUN>'s pages, no joins here.
--
-- Placeholders below (<...>) are filled in at runtime:
--   <TARGET_TABLE>    - _DEFAULT_TARGET_TABLE, signals_platform_cln.auction_audit_cln
--                        (overridable per-call; source_table param to get_ddl_statements)
--   <SNAPSHOT_TABLE>  - _DEFAULT_SNAPSHOT_TABLE, signals_platform_cln.domain_snapshots
--   <ROWS>            - one JSON line per doc, from _doc_to_ch_row(), batched
--                        BATCH_SIZE=500 docs per INSERT round-trip
--   <NOW_MS>          - integer epoch milliseconds, shared by every row in the
--                        batch (ReplacingMergeTree(updated_at) version column)

-- ============================================================================
-- SECTION 0: SCHEMA (idempotent, re-run before every seed write; non-fatal on
-- failure so an already-initialised schema is not blocked by a stale MV DDL)
-- ============================================================================

CREATE DATABASE IF NOT EXISTS signals_platform_cln;

CREATE TABLE IF NOT EXISTS <TARGET_TABLE> (
    auction_id                 Int64,
    domain_name                String,
    tld                        LowCardinality(String),
    auction_type_id            Int32,
    user_id                    Int64                   DEFAULT 0,
    current_price              Float64                 DEFAULT 0.0,
    govalue_score               Float64                 DEFAULT 0.0,
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
TTL toDateTime(listed_at) + INTERVAL 120 DAY;
-- ReplacingMergeTree(updated_at): re-running the build UPSERTS by auction_id
-- rather than duplicating rows (dedup is eventually consistent via ClickHouse
-- background merge — acceptable for explore rails).

-- (remaining CREATE DATABASE/TABLE/MATERIALIZED VIEW IF NOT EXISTS statements
-- from analytics/ch_schema.py's get_ddl_statements() run here too, in order;
-- omitted from this reference file since they belong to the broader
-- analytics schema, not the seed-upsert path.)

-- ============================================================================
-- SECTION 1: UPSERT (one INSERT per batch of <=500 docs; only columns already
-- computed by the Athena merge — no joins, no enrichment)
-- ============================================================================

INSERT INTO <TARGET_TABLE> (
    auction_id, domain_name, tld, auction_type_id, current_price, govalue_score,
    ends_at, bid_count, monthly_traffic, listed_at, user_id, buy_it_now_price,
    is_featured, gd_transfer, on_sale_rate, starting_bid, category_id, updated_at
) FORMAT JSONEachRow
<ROWS>
-- Each <ROWS> line is one JSON object, e.g.:
-- {"auction_id":123,"domain_name":"example.com","tld":"com","auction_type_id":16,
--  "current_price":42.5,"govalue_score":4200.0,"ends_at":<NOW_MS>,"bid_count":3,
--  "monthly_traffic":0,"listed_at":<NOW_MS>,"user_id":0,"buy_it_now_price":0.0,
--  "is_featured":0,"gd_transfer":0,"on_sale_rate":0.0,"starting_bid":0.0,
--  "category_id":0,"updated_at":<NOW_MS>}

-- ============================================================================
-- SECTION 2: PROXY SNAPSHOT BACKFILL (best-effort, scoped to this run's rows
-- only via updated_at = <NOW_MS> so it never full-scans the base table)
-- ============================================================================

INSERT INTO <SNAPSHOT_TABLE>
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
FROM <TARGET_TABLE>
WHERE domain_name != '' AND updated_at = <NOW_MS>;
-- This is a PROXY for completed-sales history ("listings active during the
-- period", not confirmed sales) — lets past-tense queries return data
-- instead of an empty set. ReplacingMergeTree(snapped_at) on
-- (snapshot_date, domain_name) dedupes on re-seed, so it's idempotent.
