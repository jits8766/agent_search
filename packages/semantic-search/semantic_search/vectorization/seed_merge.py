"""Athena-side seed merge: joins bid-offer, Majestic, and search-rollup
enrichment onto each configured seed table via Athena CTAS, producing one
paged final table per ``SeedTableConfig`` in the configured
``db_cfg.merge_database`` scratch database.

Supersedes the old per-page Python-side enrichment fan-out
(``_fetch_last_bid_offer_times`` / ``_fetch_majestic_metrics`` /
``_fetch_unique_search_counts``) and the Option-B per-source-table CTAS
staging mechanism — both replaced by one Athena-side merge, executed here.

Phases (mirrors ``sql/seed_merge_semsearch.sql``):
  Phase 1 - ``stg_base_*``: one CTAS per ``SeedTableConfig``, sequential
            (each independently retries on COLUMN_NOT_FOUND).
  Phase 2 - ``stg_bid_offer_*`` (per table), ``stg_majestic``,
            ``stg_search_rollup`` (shared): all depend only on Phase 1's
            output, run concurrently via ``asyncio.gather``.
  Phase 3 - ``final_*``: one CTAS per table, joining Phase 1 + Phase 2
            output and baking in the ``_page_num`` paging column.

Layer rules: imports stdlib + ``core`` only. Never imports registry,
orchestrator, retrieval, or ``db_seed_source`` (``db_seed_source`` imports
this module, not the reverse).
"""
import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)

_IDENTIFIER_RE = re.compile(r"[^A-Za-z0-9_]")


def _sanitize_identifier(raw: str) -> str:
    """Collapse any char outside ``[A-Za-z0-9_]`` to ``_`` for safe unquoted DDL interpolation."""
    return _IDENTIFIER_RE.sub("_", str(raw))


# Matches Athena's COLUMN_NOT_FOUND error message to extract the column name.
_COLUMN_NOT_FOUND_RE = re.compile(
    r"[Cc]olumn\s+'?(\w+)'?\s+cannot\s+be\s+resolved",
    re.IGNORECASE,
)

# Tier-1: enrichment columns from auction_audit_cln that may be absent in
# other configured tables. Nulled only when the column itself is missing.
_ENRICHMENT_FALLBACKS_T1: Dict[str, str] = {
    'bid_cnt': 'NULL',
    'monthly_traffic_cnt': 'NULL',
    'domain_create_utc_dt': 'NULL',
    'last_14day_traffic_cnt': 'NULL',
    'valuation_rank': 'NULL',
    'parking_revenue_usd_amt': 'NULL',
    'include_in_search_result_flag': 'NULL',
    'hide_flag': 'NULL',
    'buy_it_now_flag': 'NULL',
    'buy_it_now_usd_amt': 'NULL',
    'reserve_price_flag': 'NULL',
    'reserve_price_usd_amt': 'NULL',
    'feature_listing_flag': 'NULL',
    'gd_transfer_flag': 'NULL',
    'adult_listing_flag': 'NULL',
    'status_code_id': 'NULL',
    'category_id': 'NULL',
    'auction_list_utc_ts': 'NULL',
    'member_id': 'NULL',
    'on_sale_rate': 'NULL',
    'starting_bid_usd_amt': 'NULL',
    'traffic_cnt': 'NULL',
    'current_price_usd_amt': 'NULL',
    'website_include_flag': 'NULL',
    'item_description': 'NULL',
    'vendor_id': 'NULL',
    'domain_extension_id': 'NULL',
    'highest_bidder_id': 'NULL',
    'bid_accept_flag': 'NULL',
    'display_in_category_listing_flag': 'NULL',
    'subcategory_feature_listing_flag': 'NULL',
    'additional_category_listing_flag': 'NULL',
    'update_utc_ts': 'NULL',
}

# Tier-2: not used for auction_audit_cln (no SEO columns available).
_ENRICHMENT_FALLBACKS_T2: Dict[str, str] = {}

_ENRICHMENT_FALLBACKS: Dict[str, str] = {**_ENRICHMENT_FALLBACKS_T1, **_ENRICHMENT_FALLBACKS_T2}


def _patch_query(query: str, col_name: str) -> str:
    """Replace CAST(col_name AS TYPE) with NULL for a known enrichment column."""
    if col_name not in _ENRICHMENT_FALLBACKS:
        return query
    pattern = re.compile(
        r'CAST\s*\(\s*' + re.escape(col_name) + r'\s+AS\s+\w+\s*\)',
        re.IGNORECASE,
    )
    return pattern.sub('NULL', query)


def _build_t2_safe_query(base_query: str) -> str:
    q = base_query
    for col in _ENRICHMENT_FALLBACKS_T2:
        q = _patch_query(q, col)
    return q


def _build_safe_query(base_query: str) -> str:
    q = base_query
    for col in _ENRICHMENT_FALLBACKS:
        q = _patch_query(q, col)
    return q


@dataclass(frozen=True)
class SeedMergeStatement:
    """One DROP+CREATE pair in the merge plan.

    :param table: str - Unquoted table name (unqualified, already sanitized)
    :param drop_sql: str - ``DROP TABLE IF EXISTS {database}.{table}``
    :param create_sql: str - ``CREATE TABLE {database}.{table} ... AS SELECT ...``
    :param base_select: Optional[str] - For Phase 1 (``stg_base_*``) statements
        only: the un-wrapped seed SELECT, kept so a COLUMN_NOT_FOUND retry can
        re-patch it and rebuild ``create_sql`` without re-deriving the CTAS wrapper.
    :param fallback_create_sql: Optional[str] - For Phase 2 statements only: a
        zero-row stub CTAS with the same output schema, sourced from a literal
        ``VALUES`` row (no dependency on any external table). Used when
        ``create_sql`` fails at runtime so the optional enrichment source
        degrades to NULL columns instead of aborting the whole merge.
    :param source_label: Optional[str] - For Phase 2 statements only: a
        human-readable enrichment-source name for logging (e.g. ``"estibot"``,
        ``"bid_offer:auction_audit_cln"``).
    """
    table: str
    drop_sql: str
    create_sql: str
    base_select: Optional[str] = None
    fallback_create_sql: Optional[str] = None
    source_label: Optional[str] = None


@dataclass(frozen=True)
class SeedMergePlan:
    """Ordered plan of Athena DDL statements for one seed-merge run.

    :param database: str - From ``db_cfg.merge_database`` (env ``ATHENA_TMP_DB``; no hardcoded name)
    :param create_database_sql: str - Idempotent ``CREATE DATABASE IF NOT EXISTS
        {database}\nLOCATION '{s3_root}/'`` (unquoted name — Athena's DDL parser
        rejects the ``SCHEMA`` keyword and a quoted identifier here)
    :param phase1: List[SeedMergeStatement] - ``stg_base_*``, sequential
    :param phase2: List[SeedMergeStatement] - ``stg_bid_offer_*`` / ``stg_majestic`` /
        ``stg_search_rollup``, run concurrently
    :param phase3: List[SeedMergeStatement] - ``final_*``, sequential
    :param stg_base_tables: Dict[str, str] - table_cfg.table_name -> stg_base table name
    :param final_tables: Dict[str, str] - table_cfg.table_name -> final table name
    :param all_statements: List[SeedMergeStatement] - phase1+phase2+phase3, for cleanup
    """
    database: str
    create_database_sql: str
    phase1: List[SeedMergeStatement]
    phase2: List[SeedMergeStatement]
    phase3: List[SeedMergeStatement]
    stg_base_tables: Dict[str, str]
    final_tables: Dict[str, str]
    all_statements: List[SeedMergeStatement] = field(default_factory=list)


def _quoted(database: str, table: Optional[str] = None) -> str:
    """Unquoted table/schema reference. Both quoting styles were tried and
    both broke Athena's DDL parser: backquotes raise
    ``InvalidRequestException: backquoted identifiers are not supported; use
    double quotes``, and double quotes raise ``mismatched input '"<db>"'``
    (its DROP/CREATE TABLE grammar path doesn't accept ``QUOTED_IDENTIFIER``
    there). Plain unquoted works — confirmed by ``create_database_sql``
    already using an unquoted name successfully — and Trino/Presto's
    ``IDENTIFIER`` token allows a leading ``_`` natively, so quoting was
    never required for that case either. ``_sanitize_identifier`` already
    guarantees every table name is ``[A-Za-z0-9_]``-only before it reaches
    here."""
    if table is None:
        return database
    return f'{database}.{table}'


def _build_stg_base_ctas(database: str, table: str, select_sql: str) -> str:
    return f"CREATE TABLE {_quoted(database, table)}\nWITH (format = 'PARQUET')\nAS\n{select_sql}"


def _build_empty_stub_ctas(database: str, table: str, columns: List[Tuple[str, str]]) -> str:
    """Build a zero-row CTAS with the given ``(name, sql_type)`` schema, sourced
    from a literal ``VALUES`` row instead of any external table.

    Used as the runtime fallback for a Phase 2 statement whose real
    ``create_sql`` failed (e.g. a corrupt upstream partition) - has no
    dependency on the source that just failed, so it cannot fail for the same
    reason, and gives Phase 3's joins the same NULL-filled shape as when the
    enrichment source is simply not configured.

    :param columns: List[Tuple[str, str]] - ``[(column_name, athena_type), ...]``
    :return: str - ``CREATE TABLE ... AS SELECT * FROM (VALUES (...)) AS t(...) WHERE 1=0``
    """
    values = ", ".join(f"CAST(NULL AS {sql_type})" for _name, sql_type in columns)
    col_names = ", ".join(name for name, _sql_type in columns)
    return (
        f"CREATE TABLE {_quoted(database, table)}\n"
        f"WITH (format = 'PARQUET')\n"
        f"AS\n"
        f"SELECT * FROM (VALUES ({values})) AS t({col_names})\n"
        f"WHERE 1=0"
    )


def build_seed_merge_plan(
    db_cfg: Any,
    run_token: str,
    s3_root: str,
    base_queries: Dict[str, str],
) -> SeedMergePlan:
    """Build the full merge plan for one seed run. Pure function, no I/O.

    :param db_cfg: SeedDatabaseConfig
    :param run_token: str - Caller-supplied opaque identifier folded into every
        table name so concurrent/successive runs never collide
    :param s3_root: str - S3 URI root for the scratch database's default location
        (``db_cfg.merge_database_location``, env ``ATHENA_TMP_DB_LOC``). DB name
        itself is ``db_cfg.merge_database`` from env ``ATHENA_TMP_DB``.
    :param base_queries: Dict[str, str] - table_cfg.table_name -> rendered base
        seed SELECT (output of ``db_seed_source._build_query``)
    :return: SeedMergePlan
    """
    run_suffix = _sanitize_identifier(run_token)
    database = str(db_cfg.merge_database).strip().lower()
    create_database_sql = (
        f"CREATE DATABASE IF NOT EXISTS {database}\n"
        f"LOCATION '{s3_root.rstrip('/')}/'"
    )

    phase1: List[SeedMergeStatement] = []
    stg_base_tables: Dict[str, str] = {}
    for table_cfg in db_cfg.tables:
        stg_base = _sanitize_identifier(f"stg_base_{table_cfg.table_name}_{run_suffix}")
        stg_base_tables[table_cfg.table_name] = stg_base
        select_sql = base_queries[table_cfg.table_name]
        phase1.append(SeedMergeStatement(
            table=stg_base,
            drop_sql=f"DROP TABLE IF EXISTS {_quoted(database, stg_base)}",
            create_sql=_build_stg_base_ctas(database, stg_base, select_sql),
            base_select=select_sql,
        ))

    domain_scope_sql = " UNION ".join(
        f"SELECT domain_name FROM {_quoted(database, name)}" for name in stg_base_tables.values()
    )

    phase2: List[SeedMergeStatement] = []
    stg_bid_offer_tables: Dict[str, str] = {}
    for table_cfg in db_cfg.tables:
        stg_base = stg_base_tables[table_cfg.table_name]
        stg_bid = _sanitize_identifier(f"stg_bid_offer_{table_cfg.table_name}_{run_suffix}")
        stg_bid_offer_tables[table_cfg.table_name] = stg_bid
        create_sql = (
            f"CREATE TABLE {_quoted(database, stg_bid)}\n"
            f"WITH (format = 'PARQUET')\n"
            f"AS\n"
            f"SELECT\n"
            f"    CAST(iwb.auction_id_num AS VARCHAR) AS auction_id,\n"
            f"    CAST(MAX(b.bid_start_date_utc_ts) AS VARCHAR) AS last_bid_offer_dtm\n"
            f"FROM {db_cfg.bid_source_database}.{db_cfg.bid_source_table} b\n"
            f"JOIN {db_cfg.bid_source_database}.{db_cfg.bid_winning_table} iwb\n"
            f"    ON b.item_bid_id_num = iwb.item_bid_id_num\n"
            f"JOIN {_quoted(database, stg_base)} base\n"
            f"    ON CAST(iwb.auction_id_num AS VARCHAR) = base.auction_id\n"
            f"GROUP BY iwb.auction_id_num"
        )
        phase2.append(SeedMergeStatement(
            table=stg_bid,
            drop_sql=f"DROP TABLE IF EXISTS {_quoted(database, stg_bid)}",
            create_sql=create_sql,
            fallback_create_sql=_build_empty_stub_ctas(
                database, stg_bid,
                [("auction_id", "VARCHAR"), ("last_bid_offer_dtm", "VARCHAR")],
            ),
            source_label=f"bid_offer:{table_cfg.table_name}",
        ))

    stg_majestic = _sanitize_identifier(f"stg_majestic_{run_suffix}")
    majestic_create = (
        f"CREATE TABLE {_quoted(database, stg_majestic)}\n"
        f"WITH (format = 'PARQUET')\n"
        f"AS\n"
        f"SELECT domain_name, majestic_ext_back_links, majestic_ref_domains_fm,\n"
        f"       majestic_citation_flow_score, majestic_trust_flow_score, majestic_metric_exists\n"
        f"FROM (\n"
        f"    SELECT\n"
        f"        LOWER(m.domain_name) AS domain_name,\n"
        f"        CAST(m.ext_back_link_cnt AS BIGINT) AS majestic_ext_back_links,\n"
        f"        CAST(m.ref_domain_cnt AS BIGINT) AS majestic_ref_domains_fm,\n"
        f"        CAST(m.citation_flow_score AS INTEGER) AS majestic_citation_flow_score,\n"
        f"        CAST(m.trust_flow_score AS INTEGER) AS majestic_trust_flow_score,\n"
        f"        1 AS majestic_metric_exists,\n"
        f"        ROW_NUMBER() OVER (\n"
        f"            PARTITION BY LOWER(m.domain_name)\n"
        f"            ORDER BY m.latest_scrape_utc_date DESC\n"
        f"        ) AS _rn\n"
        f"    FROM {db_cfg.majestic.database}.{db_cfg.majestic.table_name} m\n"
        f"    WHERE m.status_label = 'Found'\n"
        f"      AND LOWER(m.domain_name) IN ({domain_scope_sql})\n"
        f") ranked\n"
        f"WHERE _rn = 1"
    )
    phase2.append(SeedMergeStatement(
        table=stg_majestic,
        drop_sql=f"DROP TABLE IF EXISTS {_quoted(database, stg_majestic)}",
        create_sql=majestic_create,
        fallback_create_sql=_build_empty_stub_ctas(
            database, stg_majestic,
            [
                ("domain_name", "VARCHAR"),
                ("majestic_ext_back_links", "BIGINT"),
                ("majestic_ref_domains_fm", "BIGINT"),
                ("majestic_citation_flow_score", "INTEGER"),
                ("majestic_trust_flow_score", "INTEGER"),
                ("majestic_metric_exists", "INTEGER"),
            ],
        ),
        source_label="majestic",
    ))

    rollup_cfg = getattr(db_cfg, "search_rollup", None)
    stg_rollup: Optional[str] = None
    if rollup_cfg is not None:
        stg_rollup = _sanitize_identifier(f"stg_search_rollup_{run_suffix}")
        rollup_create = (
            f"CREATE TABLE {_quoted(database, stg_rollup)}\n"
            f"WITH (format = 'PARQUET')\n"
            f"AS\n"
            f"SELECT LOWER(domain_name) AS domain_name,\n"
            f"       COUNT(DISTINCT customer_id) AS unique_search_count\n"
            f"FROM {rollup_cfg.database}.{rollup_cfg.table_name}\n"
            f"WHERE log_date >= date_format(date_add('day', -{rollup_cfg.lookback_days}, current_date), '%Y-%m-%d')\n"
            f"  AND LOWER(domain_name) IN ({domain_scope_sql})\n"
            f"GROUP BY LOWER(domain_name)"
        )
        phase2.append(SeedMergeStatement(
            table=stg_rollup,
            drop_sql=f"DROP TABLE IF EXISTS {_quoted(database, stg_rollup)}",
            create_sql=rollup_create,
            fallback_create_sql=_build_empty_stub_ctas(
                database, stg_rollup,
                [("domain_name", "VARCHAR"), ("unique_search_count", "BIGINT")],
            ),
            source_label="search_rollup",
        ))

    semrush_cfg = getattr(db_cfg, "semrush", None)
    stg_semrush: Optional[str] = None
    if semrush_cfg is not None:
        stg_semrush = _sanitize_identifier(f"stg_semrush_{run_suffix}")
        semrush_create = (
            f"CREATE TABLE {_quoted(database, stg_semrush)}\n"
            f"WITH (format = 'PARQUET')\n"
            f"AS\n"
            f"SELECT domain_name, semrush_ascore, semrush_total, semrush_domains_num,\n"
            f"       semrush_urls_num, semrush_keyword, semrush_search_volume,\n"
            f"       semrush_cpc, semrush_refdomains\n"
            f"FROM (\n"
            f"    SELECT\n"
            f"        LOWER(s.domain_name) AS domain_name,\n"
            f"        CAST(s.authority_score AS DOUBLE) AS semrush_ascore,\n"
            f"        CAST(s.total_backlink AS BIGINT) AS semrush_total,\n"
            f"        CAST(s.referring_domain_num AS BIGINT) AS semrush_domains_num,\n"
            f"        CAST(s.referring_url_num AS BIGINT) AS semrush_urls_num,\n"
            f"        s.keyword AS semrush_keyword,\n"
            f"        CAST(s.search_volume_monthly AS BIGINT) AS semrush_search_volume,\n"
            f"        CAST(s.cpc_usd_amt AS DOUBLE) AS semrush_cpc,\n"
            f"        CAST(json_format(CAST(s.referring_domain_array AS JSON)) AS VARCHAR) AS semrush_refdomains,\n"
            f"        ROW_NUMBER() OVER (\n"
            f"            PARTITION BY LOWER(s.domain_name)\n"
            f"            ORDER BY s.etl_build_utc_ts DESC\n"
            f"        ) AS _rn\n"
            f"    FROM {semrush_cfg.database}.{semrush_cfg.table_name} s\n"
            f"    WHERE LOWER(s.domain_name) IN ({domain_scope_sql})\n"
            f") ranked\n"
            f"WHERE _rn = 1"
        )
        phase2.append(SeedMergeStatement(
            table=stg_semrush,
            drop_sql=f"DROP TABLE IF EXISTS {_quoted(database, stg_semrush)}",
            create_sql=semrush_create,
            fallback_create_sql=_build_empty_stub_ctas(
                database, stg_semrush,
                [
                    ("domain_name", "VARCHAR"),
                    ("semrush_ascore", "DOUBLE"),
                    ("semrush_total", "BIGINT"),
                    ("semrush_domains_num", "BIGINT"),
                    ("semrush_urls_num", "BIGINT"),
                    ("semrush_keyword", "VARCHAR"),
                    ("semrush_search_volume", "BIGINT"),
                    ("semrush_cpc", "DOUBLE"),
                    ("semrush_refdomains", "VARCHAR"),
                ],
            ),
            source_label="semrush",
        ))

    estibot_cfg = getattr(db_cfg, "estibot", None)
    stg_estibot: Optional[str] = None
    if estibot_cfg is not None:
        stg_estibot = _sanitize_identifier(f"stg_estibot_{run_suffix}")
        estibot_create = (
            f"CREATE TABLE {_quoted(database, stg_estibot)}\n"
            f"WITH (format = 'PARQUET')\n"
            f"AS\n"
            f"SELECT domain_name, estibot_domain_count, estibot_domain_count_dev,\n"
            f"       estibot_ext_count, estibot_ext_count_dev\n"
            f"FROM (\n"
            f"    SELECT\n"
            f"        LOWER(e.domain_name) AS domain_name,\n"
            f"        CAST(e.domain_count AS BIGINT) AS estibot_domain_count,\n"
            f"        CAST(e.domain_count_deviation_num AS DOUBLE) AS estibot_domain_count_dev,\n"
            f"        CAST(e.extension_count AS BIGINT) AS estibot_ext_count,\n"
            f"        CAST(e.extension_count_deviation_num AS DOUBLE) AS estibot_ext_count_dev,\n"
            f"        ROW_NUMBER() OVER (\n"
            f"            PARTITION BY LOWER(e.domain_name)\n"
            f"            ORDER BY e.etl_build_utc_ts DESC\n"
            f"        ) AS _rn\n"
            f"    FROM {estibot_cfg.database}.{estibot_cfg.table_name} e\n"
            f"    WHERE LOWER(e.domain_name) IN ({domain_scope_sql})\n"
            f") ranked\n"
            f"WHERE _rn = 1"
        )
        phase2.append(SeedMergeStatement(
            table=stg_estibot,
            drop_sql=f"DROP TABLE IF EXISTS {_quoted(database, stg_estibot)}",
            create_sql=estibot_create,
            fallback_create_sql=_build_empty_stub_ctas(
                database, stg_estibot,
                [
                    ("domain_name", "VARCHAR"),
                    ("estibot_domain_count", "BIGINT"),
                    ("estibot_domain_count_dev", "DOUBLE"),
                    ("estibot_ext_count", "BIGINT"),
                    ("estibot_ext_count_dev", "DOUBLE"),
                ],
            ),
            source_label="estibot",
        ))

    boost_cfg = getattr(db_cfg, "aftermarket_boost", None)
    stg_aftermarket: Optional[str] = None
    if boost_cfg is not None:
        stg_aftermarket = _sanitize_identifier(f"stg_aftermarket_{run_suffix}")
        quoted_tiers = ", ".join(f"'{t}'" for t in boost_cfg.boosted_tier_values)
        aftermarket_create = (
            f"CREATE TABLE {_quoted(database, stg_aftermarket)}\n"
            f"WITH (format = 'PARQUET')\n"
            f"AS\n"
            f"SELECT LOWER({boost_cfg.domain_name_column}) AS domain_name,\n"
            f"       CAST(bool_or({boost_cfg.tier_column} IN ({quoted_tiers})) AS INTEGER) AS is_boosted_aftermarket\n"
            f"FROM {boost_cfg.database}.{boost_cfg.table_name}\n"
            f"WHERE LOWER({boost_cfg.domain_name_column}) IN ({domain_scope_sql})\n"
            f"GROUP BY LOWER({boost_cfg.domain_name_column})"
        )
        phase2.append(SeedMergeStatement(
            table=stg_aftermarket,
            drop_sql=f"DROP TABLE IF EXISTS {_quoted(database, stg_aftermarket)}",
            create_sql=aftermarket_create,
            fallback_create_sql=_build_empty_stub_ctas(
                database, stg_aftermarket,
                [("domain_name", "VARCHAR"), ("is_boosted_aftermarket", "INTEGER")],
            ),
            source_label="aftermarket_boost",
        ))

    phase3: List[SeedMergeStatement] = []
    final_tables: Dict[str, str] = {}
    for table_cfg in db_cfg.tables:
        stg_base = stg_base_tables[table_cfg.table_name]
        stg_bid = stg_bid_offer_tables[table_cfg.table_name]
        final_table = _sanitize_identifier(f"final_{table_cfg.table_name}_{run_suffix}")
        final_tables[table_cfg.table_name] = final_table

        if stg_rollup is not None:
            rollup_join = f"        LEFT JOIN {_quoted(database, stg_rollup)} sr ON sr.domain_name = base.domain_name\n"
            rollup_select = "        sr.unique_search_count,\n"
        else:
            rollup_join = ""
            rollup_select = "        CAST(NULL AS BIGINT) AS unique_search_count,\n"

        if stg_semrush is not None:
            semrush_join = f"        LEFT JOIN {_quoted(database, stg_semrush)} sm ON sm.domain_name = base.domain_name\n"
            semrush_select = (
                "        sm.semrush_ascore,\n"
                "        sm.semrush_total,\n"
                "        sm.semrush_domains_num,\n"
                "        sm.semrush_urls_num,\n"
                "        sm.semrush_keyword,\n"
                "        sm.semrush_search_volume,\n"
                "        sm.semrush_cpc,\n"
                "        sm.semrush_refdomains,\n"
            )
        else:
            semrush_join = ""
            semrush_select = (
                "        CAST(NULL AS DOUBLE) AS semrush_ascore,\n"
                "        CAST(NULL AS BIGINT) AS semrush_total,\n"
                "        CAST(NULL AS BIGINT) AS semrush_domains_num,\n"
                "        CAST(NULL AS BIGINT) AS semrush_urls_num,\n"
                "        CAST(NULL AS VARCHAR) AS semrush_keyword,\n"
                "        CAST(NULL AS BIGINT) AS semrush_search_volume,\n"
                "        CAST(NULL AS DOUBLE) AS semrush_cpc,\n"
                "        CAST(NULL AS VARCHAR) AS semrush_refdomains,\n"
            )

        if stg_estibot is not None:
            estibot_join = f"        LEFT JOIN {_quoted(database, stg_estibot)} eb ON eb.domain_name = base.domain_name\n"
            estibot_select = (
                "        eb.estibot_domain_count,\n"
                "        eb.estibot_domain_count_dev,\n"
                "        eb.estibot_ext_count,\n"
                "        eb.estibot_ext_count_dev,\n"
            )
        else:
            estibot_join = ""
            estibot_select = (
                "        CAST(NULL AS BIGINT) AS estibot_domain_count,\n"
                "        CAST(NULL AS DOUBLE) AS estibot_domain_count_dev,\n"
                "        CAST(NULL AS BIGINT) AS estibot_ext_count,\n"
                "        CAST(NULL AS DOUBLE) AS estibot_ext_count_dev,\n"
            )

        if stg_aftermarket is not None:
            aftermarket_join = f"        LEFT JOIN {_quoted(database, stg_aftermarket)} am ON am.domain_name = base.domain_name\n"
            aftermarket_select = "        COALESCE(am.is_boosted_aftermarket, 0) AS is_boosted_aftermarket\n"
        else:
            aftermarket_join = ""
            aftermarket_select = "        CAST(NULL AS INTEGER) AS is_boosted_aftermarket\n"

        merged_select = (
            f"        base.*,\n"
            f"        bo.last_bid_offer_dtm,\n"
            f"        mj.majestic_ext_back_links,\n"
            f"        mj.majestic_ref_domains_fm,\n"
            f"        mj.majestic_citation_flow_score,\n"
            f"        mj.majestic_trust_flow_score,\n"
            f"        COALESCE(mj.majestic_metric_exists, 0) AS majestic_metric_exists,\n"
            f"{rollup_select}"
            f"{semrush_select}"
            f"{estibot_select}"
            f"{aftermarket_select}"
        )
        create_sql = (
            f"CREATE TABLE {_quoted(database, final_table)}\n"
            f"WITH (format = 'PARQUET')\n"
            f"AS\n"
            f"SELECT numbered.*, CAST((_merge_rn - 1) / {db_cfg.merge_page_size} AS INTEGER) AS _page_num\n"
            f"FROM (\n"
            f"    SELECT merged.*, ROW_NUMBER() OVER (ORDER BY auction_id) AS _merge_rn\n"
            f"    FROM (\n"
            f"        SELECT\n"
            f"{merged_select}"
            f"        FROM {_quoted(database, stg_base)} base\n"
            f"        LEFT JOIN {_quoted(database, stg_bid)} bo ON bo.auction_id = base.auction_id\n"
            f"        LEFT JOIN {_quoted(database, stg_majestic)} mj ON mj.domain_name = base.domain_name\n"
            f"{rollup_join}"
            f"{semrush_join}"
            f"{estibot_join}"
            f"{aftermarket_join}"
            f"    ) merged\n"
            f") numbered"
        )
        phase3.append(SeedMergeStatement(
            table=final_table,
            drop_sql=f"DROP TABLE IF EXISTS {_quoted(database, final_table)}",
            create_sql=create_sql,
        ))

    return SeedMergePlan(
        database=database,
        create_database_sql=create_database_sql,
        phase1=phase1,
        phase2=phase2,
        phase3=phase3,
        stg_base_tables=stg_base_tables,
        final_tables=final_tables,
        all_statements=phase1 + phase2 + phase3,
    )


async def _create_stg_base_with_fallback(
    athena_client: Any,
    database: str,
    stmt: SeedMergeStatement,
    ddl_timeout_seconds: float,
) -> List[str]:
    """Run one Phase-1 ``stg_base_*`` CTAS with the 3-tier COLUMN_NOT_FOUND retry.

    :return: List[str] - missing enrichment column names (empty on full success)
    """
    try:
        await athena_client.execute_ddl(stmt.create_sql, ddl_timeout_seconds)
        return []
    except (RuntimeError, TimeoutError, OSError, ValueError, TypeError, KeyError) as exc:
        m = _COLUMN_NOT_FOUND_RE.search(str(exc))
        first_missing = m.group(1).lower() if m else None

        if first_missing and first_missing in _ENRICHMENT_FALLBACKS_T2:
            t2_safe = _build_t2_safe_query(stmt.base_select)
            missing_cols = list(_ENRICHMENT_FALLBACKS_T2.keys())
            logger.warning(
                f"seed_merge_column_missing table={stmt.table} first_missing_column={first_missing} "
                f"action=replacing_tier2_enrichment_with_null tier1_preserved=True retrying=True"
            )
            try:
                await athena_client.execute_ddl(_build_stg_base_ctas(database, stmt.table, t2_safe), ddl_timeout_seconds)
                return missing_cols
            except (RuntimeError, TimeoutError, OSError, ValueError, TypeError, KeyError) as exc2:
                m2 = _COLUMN_NOT_FOUND_RE.search(str(exc2))
                if m2 and m2.group(1).lower() in _ENRICHMENT_FALLBACKS_T1:
                    all_safe = _build_safe_query(stmt.base_select)
                    missing_cols = list(_ENRICHMENT_FALLBACKS.keys())
                    logger.warning(
                        f"seed_merge_column_missing table={stmt.table} first_missing_column={m2.group(1).lower()} "
                        f"action=replacing_all_enrichment_with_null retrying=True"
                    )
                    await athena_client.execute_ddl(_build_stg_base_ctas(database, stmt.table, all_safe), ddl_timeout_seconds)
                    return missing_cols
                raise
        elif first_missing and first_missing in _ENRICHMENT_FALLBACKS_T1:
            all_safe = _build_safe_query(stmt.base_select)
            missing_cols = list(_ENRICHMENT_FALLBACKS.keys())
            logger.warning(
                f"seed_merge_column_missing table={stmt.table} first_missing_column={first_missing} "
                f"action=replacing_all_enrichment_with_null retrying=True"
            )
            await athena_client.execute_ddl(_build_stg_base_ctas(database, stmt.table, all_safe), ddl_timeout_seconds)
            return missing_cols
        raise


async def run_seed_merge(
    athena_client: Any,
    plan: SeedMergePlan,
    ddl_timeout_seconds: float,
    timing: Any,
) -> Dict[str, List[str]]:
    """Execute the full merge plan: Phase 1 (sequential) -> Phase 2 (parallel) -> Phase 3 (sequential).

    :param timing: StageTimingSession - Config-driven stage timer (required)
    :return: Dict[str, List[str]] - table_cfg.table_name -> missing enrichment columns
    """
    merge_gate = bool(timing.config.log_merge_phases)
    with timing.stage("athena_seed_merge", gate=True, database=plan.database):
        with timing.stage("athena_merge_create_database", gate=merge_gate, database=plan.database):
            # Glue GetDatabase first — CREATE DATABASE (even IF NOT EXISTS) still
            # requires glue:CreateDatabase. Reuse a pre-created scratch DB when present.
            if await athena_client.database_exists(plan.database):
                logger.info(
                    f"seed_merge_database_reuse database={plan.database} "
                    f"action=skip_create reason=already_exists"
                )
            else:
                logger.info(
                    f"seed_merge_database_create database={plan.database} "
                    f"action=create_database_if_not_exists"
                )
                await athena_client.execute_ddl(plan.create_database_sql, ddl_timeout_seconds)

        missing_by_stg_base: Dict[str, List[str]] = {}
        with timing.stage(
            "athena_merge_phase1",
            gate=merge_gate,
            database=plan.database,
            statements=len(plan.phase1),
        ):
            for stmt in plan.phase1:
                await athena_client.execute_ddl(stmt.drop_sql, ddl_timeout_seconds)
                missing_by_stg_base[stmt.table] = await _create_stg_base_with_fallback(
                    athena_client, plan.database, stmt, ddl_timeout_seconds
                )

        async def _run_phase2(stmt: SeedMergeStatement) -> None:
            """Run one Phase 2 CTAS; on failure, fall back to a dependency-free empty
            stub so this source degrades to NULL columns instead of aborting the run."""
            await athena_client.execute_ddl(stmt.drop_sql, ddl_timeout_seconds)
            try:
                await athena_client.execute_ddl(stmt.create_sql, ddl_timeout_seconds)
            except (RuntimeError, TimeoutError, OSError, ValueError, TypeError, KeyError) as exc:
                logger.warning(
                    f"seed_merge_phase2_source_degraded table={stmt.table} source={stmt.source_label} "
                    f"error_type={type(exc).__name__} error={str(exc)[:300]}"
                )
                await athena_client.execute_ddl(stmt.fallback_create_sql, ddl_timeout_seconds)

        with timing.stage(
            "athena_merge_phase2",
            gate=merge_gate,
            database=plan.database,
            statements=len(plan.phase2),
        ):
            await asyncio.gather(*(_run_phase2(stmt) for stmt in plan.phase2))

        with timing.stage(
            "athena_merge_phase3",
            gate=merge_gate,
            database=plan.database,
            statements=len(plan.phase3),
        ):
            for stmt in plan.phase3:
                await athena_client.execute_ddl(stmt.drop_sql, ddl_timeout_seconds)
                await athena_client.execute_ddl(stmt.create_sql, ddl_timeout_seconds)

    return {
        table_name: missing_by_stg_base.get(stg_base, [])
        for table_name, stg_base in plan.stg_base_tables.items()
    }


async def cleanup_seed_merge(
    athena_client: Any,
    plan: SeedMergePlan,
    ddl_timeout_seconds: float,
) -> None:
    """Drop every table the plan created. Best-effort - one failure doesn't block the rest."""

    async def _drop(stmt: SeedMergeStatement) -> None:
        try:
            await athena_client.execute_ddl(stmt.drop_sql, ddl_timeout_seconds)
        except Exception:
            logger.exception(f"seed_merge_cleanup_drop_failed table={stmt.table}")

    await asyncio.gather(*(_drop(stmt) for stmt in plan.all_statements))
