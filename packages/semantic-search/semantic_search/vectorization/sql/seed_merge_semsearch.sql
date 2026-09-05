-- seed_merge_semsearch.sql
--
-- Reference DDL for the Athena-side seed merge. Style mirrors
-- offline_eval/offline/src/data_pipeline/conversation_metrics/sql/conv_trace_metrics.sql:
-- DROP TABLE IF EXISTS immediately before every CREATE TABLE ... WITH (...) AS SELECT.
--
-- This file is a human-reviewable, manually-runnable copy of what
-- semantic_search/vectorization/seed_merge.py generates and executes
-- statement-by-statement at runtime (Athena's API takes one statement per
-- call — this file is documentation, not what actually executes).
--
-- Placeholders below (<...>) are filled in at runtime:
--   <TMP_DB>          - from .env's ATHENA_TMP_DB, via db_cfg.merge_database
--   <ATHENA_TMP_DB_LOC> - from .env's ATHENA_TMP_DB_LOC, via db_cfg.merge_database_location
--   <RUN>             - _sanitize_identifier(run_token), e.g. "seed" or "backfill"
--   <TABLE>           - each configured SeedTableConfig.table_name, e.g. auction_audit_cln
--   <SEED_DB>         - the configured seed source database (SeedDatabaseConfig.name)
--   <MAJESTIC_DB>.<MAJESTIC_TABLE>   - SeedMajesticConfig.database / .table_name
--   <ROLLUP_DB>.<ROLLUP_TABLE>       - SeedSearchRollupConfig.database / .table_name (omitted entirely when not configured)
--
-- <TMP_DB> is a disposable scratch database: everything under it is dropped
-- once a run's Qdrant and ClickHouse upserts (SECTION 4) both finish.
--
-- Tables created here have NO external_location: the scratch database itself
-- carries a LOCATION, so every CTAS is "managed" under it, and DROP TABLE
-- deletes both the Glue catalog entry and the underlying S3 data — no
-- separate S3-prefix-clearing step is needed.

-- ============================================================================
-- SECTION 0: SCRATCH DATABASE (idempotent, created once per run)
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS <TMP_DB> WITH (location = '<ATHENA_TMP_DB_LOC>');

-- ============================================================================
-- STAGING TABLE DEPENDENCY HIERARCHY
--
--   stg_base_<TABLE>_<RUN>          (Phase 1, sequential, one per table_cfg)
--     |
--     +-- stg_bid_offer_<TABLE>_<RUN>   (Phase 2, parallel; JOINs stg_base_<TABLE>_<RUN>)
--     +-- stg_majestic_<RUN>            (Phase 2, parallel, shared; JOINs UNION of all stg_base_*)
--     +-- stg_search_rollup_<RUN>       (Phase 2, parallel, shared, optional; JOINs UNION of all stg_base_*)
--     |
--     +-- final_<TABLE>_<RUN>           (Phase 3, sequential, one per table_cfg;
--                                        LEFT JOINs stg_base_<TABLE>_<RUN> with its
--                                        stg_bid_offer_<TABLE>_<RUN>, the shared
--                                        stg_majestic_<RUN>, the shared stg_search_rollup_<RUN>;
--                                        bakes in ROW_NUMBER()/_page_num for paging)
--
-- Qdrant/ClickHouse upsert reads pages from final_<TABLE>_<RUN> only.
-- ============================================================================

-- ============================================================================
-- SECTION 1: PHASE 1 - BASE SEED TABLES (sequential, one per SeedTableConfig)
-- ============================================================================

DROP TABLE IF EXISTS <TMP_DB>.stg_base_<TABLE>_<RUN>;

CREATE TABLE <TMP_DB>.stg_base_<TABLE>_<RUN>
WITH (format = 'PARQUET')
AS
-- Exactly today's db_seed_source._build_query(<SEED_DB>, table_cfg) output:
-- SELECT <all seed columns> FROM <SEED_DB>.<TABLE>
-- WHERE <datewise: auctionstarttime lookback | count: none> [AND auctionendtime > CURRENT_TIMESTAMP]
-- ORDER BY auctionstarttime DESC LIMIT <max_records>
SELECT * FROM <SEED_DB>.<TABLE>; -- placeholder; real SELECT rendered by _build_query at runtime

-- ============================================================================
-- SECTION 2: PHASE 2 - ENRICHMENT STAGING (parallel; each depends only on Phase 1)
-- ============================================================================

-- 2a. Bid-offer times: one per table_cfg, JOINed to its own stg_base (was an IN-list per page).
DROP TABLE IF EXISTS <TMP_DB>.stg_bid_offer_<TABLE>_<RUN>;

CREATE TABLE <TMP_DB>.stg_bid_offer_<TABLE>_<RUN>
WITH (format = 'PARQUET')
AS
SELECT
    CAST(iwb.auction_id_num AS VARCHAR) AS auction_id,
    CAST(MAX(b.bid_start_date_utc_ts) AS VARCHAR) AS last_bid_offer_dtm
FROM the_resale_place.item_bids_cln b
JOIN the_resale_place.item_winning_bids_cln iwb
    ON b.item_bid_id_num = iwb.item_bid_id_num
JOIN <TMP_DB>.stg_base_<TABLE>_<RUN> base
    ON CAST(iwb.auction_id_num AS VARCHAR) = base.auction_id
GROUP BY iwb.auction_id_num;

-- 2b. Majestic metrics: ONE shared table across all table_cfgs, domain-scoped to the
--     union of every stg_base_*.domain_name (was an IN-list of the current page's domains).
DROP TABLE IF EXISTS <TMP_DB>.stg_majestic_<RUN>;

CREATE TABLE <TMP_DB>.stg_majestic_<RUN>
WITH (format = 'PARQUET')
AS
SELECT domain_name, majestic_ext_back_links, majestic_ref_domains_fm,
       majestic_citation_flow_score, majestic_trust_flow_score, majestic_metric_exists
FROM (
    SELECT
        LOWER(m.domain_name) AS domain_name,
        CAST(m.ext_back_link_cnt AS BIGINT) AS majestic_ext_back_links,
        CAST(m.ref_domain_cnt AS BIGINT) AS majestic_ref_domains_fm,
        CAST(m.citation_flow_score AS INTEGER) AS majestic_citation_flow_score,
        CAST(m.trust_flow_score AS INTEGER) AS majestic_trust_flow_score,
        1 AS majestic_metric_exists,
        ROW_NUMBER() OVER (
            PARTITION BY LOWER(m.domain_name)
            ORDER BY m.latest_scrape_utc_date DESC
        ) AS _rn
    FROM <MAJESTIC_DB>.<MAJESTIC_TABLE> m
    WHERE m.status_label = 'Found'
      AND LOWER(m.domain_name) IN (
          SELECT domain_name FROM <TMP_DB>.stg_base_<TABLE>_<RUN>
          -- UNION SELECT domain_name FROM <TMP_DB>.stg_base_<OTHER_TABLE>_<RUN> ...
      )
) ranked
WHERE _rn = 1;

-- 2c. Search rollup: ONE shared table across all table_cfgs (only when
--     SeedSearchRollupConfig is configured; omitted entirely otherwise).
DROP TABLE IF EXISTS <TMP_DB>.stg_search_rollup_<RUN>;

CREATE TABLE <TMP_DB>.stg_search_rollup_<RUN>
WITH (format = 'PARQUET')
AS
SELECT LOWER(domain_name) AS domain_name,
       COUNT(DISTINCT customer_id) AS unique_search_count
FROM <ROLLUP_DB>.<ROLLUP_TABLE>
WHERE log_date >= date_format(date_add('day', -<ROLLUP_LOOKBACK_DAYS>, current_date), '%Y-%m-%d')
  AND LOWER(domain_name) IN (
      SELECT domain_name FROM <TMP_DB>.stg_base_<TABLE>_<RUN>
      -- UNION SELECT domain_name FROM <TMP_DB>.stg_base_<OTHER_TABLE>_<RUN> ...
  )
GROUP BY LOWER(domain_name);

-- ============================================================================
-- SECTION 3: PHASE 3 - FINAL MERGED + PAGED TABLE (sequential, one per table_cfg)
-- ============================================================================

DROP TABLE IF EXISTS <TMP_DB>.final_<TABLE>_<RUN>;

CREATE TABLE <TMP_DB>.final_<TABLE>_<RUN>
WITH (format = 'PARQUET')
AS
SELECT numbered.*, CAST((_merge_rn - 1) / 50000 AS INTEGER) AS _page_num
FROM (
    SELECT merged.*, ROW_NUMBER() OVER (ORDER BY auction_id) AS _merge_rn
    FROM (
        SELECT
            base.*,
            bo.last_bid_offer_dtm,
            mj.majestic_ext_back_links,
            mj.majestic_ref_domains_fm,
            mj.majestic_citation_flow_score,
            mj.majestic_trust_flow_score,
            COALESCE(mj.majestic_metric_exists, 0) AS majestic_metric_exists,
            sr.unique_search_count -- CAST(NULL AS BIGINT) AS unique_search_count when rollup not configured
        FROM <TMP_DB>.stg_base_<TABLE>_<RUN> base
        LEFT JOIN <TMP_DB>.stg_bid_offer_<TABLE>_<RUN> bo ON bo.auction_id = base.auction_id
        LEFT JOIN <TMP_DB>.stg_majestic_<RUN> mj ON mj.domain_name = base.domain_name
        LEFT JOIN <TMP_DB>.stg_search_rollup_<RUN> sr ON sr.domain_name = base.domain_name -- omitted when rollup not configured
    ) merged
) numbered;

-- `is_gem` is NOT computed here: it is derived in Python (db_seed_source._row_to_doc)
-- from this table's own unique_search_count + govalue_score columns, exactly as before —
-- moving it into SQL would just duplicate logic that's already correct once
-- unique_search_count carries a real joined value instead of always being NULL.

-- Qdrant/ClickHouse upsert pages this table via:
--   SELECT MAX(_page_num) FROM <TMP_DB>.final_<TABLE>_<RUN>
--   SELECT * FROM <TMP_DB>.final_<TABLE>_<RUN> WHERE _page_num = <N>

-- ============================================================================
-- SECTION 4: CLEANUP - drop every table this run created (also re-run in a
-- `finally` block by seed_merge.cleanup_seed_merge, so cleanup happens on
-- success, error, or cancellation)
-- ============================================================================

DROP TABLE IF EXISTS <TMP_DB>.stg_base_<TABLE>_<RUN>;
DROP TABLE IF EXISTS <TMP_DB>.stg_bid_offer_<TABLE>_<RUN>;
DROP TABLE IF EXISTS <TMP_DB>.stg_majestic_<RUN>;
DROP TABLE IF EXISTS <TMP_DB>.stg_search_rollup_<RUN>;
DROP TABLE IF EXISTS <TMP_DB>.final_<TABLE>_<RUN>;
