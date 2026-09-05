"""Athena-backed enrichment source for seed-time-only domain enrichment fields.

Queries the majestic / semrush / estibot / search_rollup Athena tables — the
same source tables joined at seed time by ``seed_merge.py`` — for rows newer
than a cursor timestamp, deduplicates to one row per domain using
ROW_NUMBER(), and returns payload-update dicts keyed by lowercased
``domain_name`` (NOT ``auction_id``). ``domain_name`` is the Qdrant filter
key ``EnrichmentRefreshDriver`` patches on, since one domain backs multiple
concurrent Qdrant points (concurrent live auctions for that domain).

Column mapping mirrors the CAST/rename already proven in
``seed_merge.py``'s staging CTAS statements — see ``stg_majestic``,
``stg_semrush``, ``stg_estibot``, ``stg_rollup`` there for the reference
join this module re-derives incrementally:

  majestic  (m.ext_back_link_cnt, m.ref_domain_cnt, m.citation_flow_score,
             m.trust_flow_score) -> majestic_ext_back_links,
             majestic_ref_domains_fm, majestic_citation_flow_score,
             majestic_trust_flow_score, majestic_metric_exists
  semrush   (s.authority_score, s.total_backlink, s.referring_domain_num,
             s.referring_url_num, s.keyword, s.search_volume_monthly,
             s.cpc_usd_amt, s.referring_domain_array) -> semrush_ascore,
             semrush_total, semrush_domains_num, semrush_urls_num,
             semrush_keyword, semrush_search_volume, semrush_cpc,
             semrush_refdomains
  estibot   (e.domain_count, e.domain_count_deviation_num,
             e.extension_count, e.extension_count_deviation_num) ->
             estibot_domain_count, estibot_domain_count_dev,
             estibot_ext_count, estibot_ext_count_dev

Cursor columns (majestic: latest_scrape_utc_date, semrush/estibot:
etl_build_utc_ts) have [Uncertain] native Athena types — no live catalog
access confirmed them, and a prior hand-written freshness query against
these same tables failed against real Athena. Every cursor comparison and
ORDER BY here therefore uses TRY_CAST(... AS TIMESTAMP), which returns NULL
on a cast failure (Presto/Trino semantics) instead of erroring the whole
query, so a wrong assumed type degrades to "row excluded" rather than a
hard failure.

search_rollup has no incremental cursor at all: unique_search_count is a
rolling lookback_days-window aggregate (COUNT(DISTINCT customer_id)), so
every cycle recomputes the full window rather than filtering by a since-ts.

Unlike ``delta_source.py``'s auction_audit_cln query, none of these four
source tables have confirmed Athena partition columns (no evidence found in
seed_merge.py), so no partition-pruning predicate is applied here — cycles
scan the since/now window (or, for search_rollup, the full lookback
window) directly. This is a known cost tradeoff of moving from a
domain-scoped seed-time join to a periodically-polled table scan.

Layer rules: imports stdlib + core + nl_to_sql.athena_client + config.models.
Never imports registry, orchestrator, or retrieval code.
"""

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from semantic_search.config.models import (
    SeedEstibotConfig,
    SeedMajesticConfig,
    SeedSearchRollupConfig,
    SeedSemrushConfig,
)
from semantic_search.core.logging_utils import get_logger
from semantic_search.nl_to_sql.athena_client import AthenaClient

logger = get_logger(__name__)


def _ts_to_str(ts: float) -> str:
    """Format a Unix timestamp as the Athena TIMESTAMP literal string."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _safe_optional_int(raw: Any) -> Optional[int]:
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return int(float(raw))
    except (ValueError, TypeError):
        return None


def _safe_optional_float(raw: Any) -> Optional[float]:
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None


_MAJESTIC_SQL_TEMPLATE = (
    "WITH ranked AS ("
    " SELECT LOWER(m.domain_name) AS domain_name,"
    " CAST(m.ext_back_link_cnt AS BIGINT) AS majestic_ext_back_links,"
    " CAST(m.ref_domain_cnt AS BIGINT) AS majestic_ref_domains_fm,"
    " CAST(m.citation_flow_score AS INTEGER) AS majestic_citation_flow_score,"
    " CAST(m.trust_flow_score AS INTEGER) AS majestic_trust_flow_score,"
    " ROW_NUMBER() OVER (PARTITION BY LOWER(m.domain_name)"
    " ORDER BY TRY_CAST(m.latest_scrape_utc_date AS TIMESTAMP) DESC) AS rn"
    " FROM {database}.{table_name} m"
    " WHERE m.status_label = 'Found'"
    " AND TRY_CAST(m.latest_scrape_utc_date AS TIMESTAMP) >= TIMESTAMP '{since_ts}'"
    " AND TRY_CAST(m.latest_scrape_utc_date AS TIMESTAMP) < TIMESTAMP '{now_ts}'"
    ")"
    " SELECT domain_name, majestic_ext_back_links, majestic_ref_domains_fm,"
    " majestic_citation_flow_score, majestic_trust_flow_score"
    " FROM ranked WHERE rn = 1"
    " LIMIT {batch_size}"
)

_SEMRUSH_SQL_TEMPLATE = (
    "WITH ranked AS ("
    " SELECT LOWER(s.domain_name) AS domain_name,"
    " CAST(s.authority_score AS DOUBLE) AS semrush_ascore,"
    " CAST(s.total_backlink AS BIGINT) AS semrush_total,"
    " CAST(s.referring_domain_num AS BIGINT) AS semrush_domains_num,"
    " CAST(s.referring_url_num AS BIGINT) AS semrush_urls_num,"
    " s.keyword AS semrush_keyword,"
    " CAST(s.search_volume_monthly AS BIGINT) AS semrush_search_volume,"
    " CAST(s.cpc_usd_amt AS DOUBLE) AS semrush_cpc,"
    " ROW_NUMBER() OVER (PARTITION BY LOWER(s.domain_name)"
    " ORDER BY TRY_CAST(s.etl_build_utc_ts AS TIMESTAMP) DESC) AS rn"
    " FROM {database}.{table_name} s"
    " WHERE TRY_CAST(s.etl_build_utc_ts AS TIMESTAMP) >= TIMESTAMP '{since_ts}'"
    " AND TRY_CAST(s.etl_build_utc_ts AS TIMESTAMP) < TIMESTAMP '{now_ts}'"
    ")"
    " SELECT domain_name, semrush_ascore, semrush_total, semrush_domains_num,"
    " semrush_urls_num, semrush_keyword, semrush_search_volume, semrush_cpc"
    " FROM ranked WHERE rn = 1"
    " LIMIT {batch_size}"
)

_ESTIBOT_SQL_TEMPLATE = (
    "WITH ranked AS ("
    " SELECT LOWER(e.domain_name) AS domain_name,"
    " CAST(e.domain_count AS BIGINT) AS estibot_domain_count,"
    " CAST(e.domain_count_deviation_num AS DOUBLE) AS estibot_domain_count_dev,"
    " CAST(e.extension_count AS BIGINT) AS estibot_ext_count,"
    " CAST(e.extension_count_deviation_num AS DOUBLE) AS estibot_ext_count_dev,"
    " ROW_NUMBER() OVER (PARTITION BY LOWER(e.domain_name)"
    " ORDER BY TRY_CAST(e.etl_build_utc_ts AS TIMESTAMP) DESC) AS rn"
    " FROM {database}.{table_name} e"
    " WHERE TRY_CAST(e.etl_build_utc_ts AS TIMESTAMP) >= TIMESTAMP '{since_ts}'"
    " AND TRY_CAST(e.etl_build_utc_ts AS TIMESTAMP) < TIMESTAMP '{now_ts}'"
    ")"
    " SELECT domain_name, estibot_domain_count, estibot_domain_count_dev,"
    " estibot_ext_count, estibot_ext_count_dev"
    " FROM ranked WHERE rn = 1"
    " LIMIT {batch_size}"
)

_ROLLUP_SQL_TEMPLATE = (
    "SELECT LOWER(domain_name) AS domain_name,"
    " COUNT(DISTINCT customer_id) AS unique_search_count"
    " FROM {database}.{table_name}"
    " WHERE log_date >= date_format(date_add('day', -{lookback_days}, current_date), '%Y-%m-%d')"
    " GROUP BY LOWER(domain_name)"
    " LIMIT {batch_size}"
)


async def _fetch_generic(
    athena_client: AthenaClient,
    sql: str,
    timeout_seconds: float,
    row_to_update: Any,
    source_name: str,
    batch_size: int,
) -> List[Tuple[str, Dict[str, Any]]]:
    if not athena_client.credentials_available:
        raise RuntimeError(f"enrichment_source athena_client credentials unavailable source={source_name}")
    t0 = time.monotonic()
    rows, _cols, latency_ms = await athena_client.fetch_sql_async(sql, timeout_seconds)
    out: List[Tuple[str, Dict[str, Any]]] = []
    for row in rows or []:
        domain_name, updates = row_to_update(row)
        if domain_name and updates:
            out.append((domain_name, updates))
    elapsed = (time.monotonic() - t0) * 1000.0
    if len(rows or []) >= batch_size:
        logger.warning(
            f"enrichment_source_batch_truncated source={source_name} rows={len(rows or [])} batch_size={batch_size}"
        )
    logger.info(
        f"enrichment_source_fetch source={source_name} rows={len(rows or [])} updates={len(out)} "
        f"athena_ms={latency_ms:.1f} total_ms={elapsed:.1f}"
    )
    return out


def _majestic_row_to_update(row: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    domain_name = str(row.get("domain_name", "")).strip()
    updates: Dict[str, Any] = {}
    ext_back_links = _safe_optional_int(row.get("majestic_ext_back_links"))
    ref_domains = _safe_optional_int(row.get("majestic_ref_domains_fm"))
    citation_flow = _safe_optional_int(row.get("majestic_citation_flow_score"))
    trust_flow = _safe_optional_int(row.get("majestic_trust_flow_score"))
    if ext_back_links is not None:
        updates["majestic_ext_back_links"] = ext_back_links
    if ref_domains is not None:
        updates["majestic_ref_domains_fm"] = ref_domains
    if citation_flow is not None:
        updates["majestic_citation_flow_score"] = citation_flow
    if trust_flow is not None:
        updates["majestic_trust_flow_score"] = trust_flow
    if updates:
        updates["majestic_metric_exists"] = 1
    return domain_name, updates


def _semrush_row_to_update(row: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    domain_name = str(row.get("domain_name", "")).strip()
    updates: Dict[str, Any] = {}
    ascore = _safe_optional_float(row.get("semrush_ascore"))
    total = _safe_optional_int(row.get("semrush_total"))
    domains_num = _safe_optional_int(row.get("semrush_domains_num"))
    urls_num = _safe_optional_int(row.get("semrush_urls_num"))
    keyword = row.get("semrush_keyword")
    search_volume = _safe_optional_int(row.get("semrush_search_volume"))
    cpc = _safe_optional_float(row.get("semrush_cpc"))
    if ascore is not None:
        updates["semrush_ascore"] = ascore
    if total is not None:
        updates["semrush_total"] = total
    if domains_num is not None:
        updates["semrush_domains_num"] = domains_num
    if urls_num is not None:
        updates["semrush_urls_num"] = urls_num
    if keyword is not None and str(keyword).strip():
        updates["semrush_keyword"] = str(keyword).strip()
    if search_volume is not None:
        updates["semrush_search_volume"] = search_volume
    if cpc is not None:
        updates["semrush_cpc"] = cpc
    return domain_name, updates


def _estibot_row_to_update(row: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    domain_name = str(row.get("domain_name", "")).strip()
    updates: Dict[str, Any] = {}
    domain_count = _safe_optional_int(row.get("estibot_domain_count"))
    domain_count_dev = _safe_optional_float(row.get("estibot_domain_count_dev"))
    ext_count = _safe_optional_int(row.get("estibot_ext_count"))
    ext_count_dev = _safe_optional_float(row.get("estibot_ext_count_dev"))
    if domain_count is not None:
        updates["estibot_domain_count"] = domain_count
    if domain_count_dev is not None:
        updates["estibot_domain_count_dev"] = domain_count_dev
    if ext_count is not None:
        updates["estibot_ext_count"] = ext_count
    if ext_count_dev is not None:
        updates["estibot_ext_count_dev"] = ext_count_dev
    return domain_name, updates


def _rollup_row_to_update(row: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    domain_name = str(row.get("domain_name", "")).strip()
    count = _safe_optional_int(row.get("unique_search_count"))
    updates: Dict[str, Any] = {"unique_search_count": count} if count is not None else {}
    return domain_name, updates


async def fetch_majestic_delta(
    athena_client: AthenaClient,
    majestic_cfg: SeedMajesticConfig,
    since_ts: float,
    now_ts: float,
    batch_size: int,
    timeout_seconds: float,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Query majestic feature-mart snapshot for rows scraped in (since_ts, now_ts)."""
    sql = _MAJESTIC_SQL_TEMPLATE.format(
        database=majestic_cfg.database,
        table_name=majestic_cfg.table_name,
        since_ts=_ts_to_str(since_ts),
        now_ts=_ts_to_str(now_ts),
        batch_size=batch_size,
    )
    return await _fetch_generic(athena_client, sql, timeout_seconds, _majestic_row_to_update, "majestic", batch_size)


async def fetch_semrush_delta(
    athena_client: AthenaClient,
    semrush_cfg: SeedSemrushConfig,
    since_ts: float,
    now_ts: float,
    batch_size: int,
    timeout_seconds: float,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Query semrush_domain_enrichments for rows built in (since_ts, now_ts)."""
    sql = _SEMRUSH_SQL_TEMPLATE.format(
        database=semrush_cfg.database,
        table_name=semrush_cfg.table_name,
        since_ts=_ts_to_str(since_ts),
        now_ts=_ts_to_str(now_ts),
        batch_size=batch_size,
    )
    return await _fetch_generic(athena_client, sql, timeout_seconds, _semrush_row_to_update, "semrush", batch_size)


async def fetch_estibot_delta(
    athena_client: AthenaClient,
    estibot_cfg: SeedEstibotConfig,
    since_ts: float,
    now_ts: float,
    batch_size: int,
    timeout_seconds: float,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Query estibot_domain_enrichments for rows built in (since_ts, now_ts)."""
    sql = _ESTIBOT_SQL_TEMPLATE.format(
        database=estibot_cfg.database,
        table_name=estibot_cfg.table_name,
        since_ts=_ts_to_str(since_ts),
        now_ts=_ts_to_str(now_ts),
        batch_size=batch_size,
    )
    return await _fetch_generic(athena_client, sql, timeout_seconds, _estibot_row_to_update, "estibot", batch_size)


async def fetch_rollup_full(
    athena_client: AthenaClient,
    rollup_cfg: SeedSearchRollupConfig,
    batch_size: int,
    timeout_seconds: float,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Recompute the full rolling lookback_days unique_search_count window.

    No incremental since-cursor: unique_search_count is itself a rolling
    aggregate, so every cycle is a full pass over the configured window.
    """
    sql = _ROLLUP_SQL_TEMPLATE.format(
        database=rollup_cfg.database,
        table_name=rollup_cfg.table_name,
        lookback_days=rollup_cfg.lookback_days,
        batch_size=batch_size,
    )
    return await _fetch_generic(athena_client, sql, timeout_seconds, _rollup_row_to_update, "search_rollup", batch_size)


__all__ = [
    "fetch_majestic_delta",
    "fetch_semrush_delta",
    "fetch_estibot_delta",
    "fetch_rollup_full",
]
