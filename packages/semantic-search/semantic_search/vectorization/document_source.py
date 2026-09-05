"""Athena-backed document source factory for the ``OfflineIndexer``.

Queries ``signals_platform_cln.auction_audit_cln`` for in-scope auction types:
16 (GoDaddy AutoExtend), 20 (GoDaddy BuyNow/Closeout),
38 (Partner AutoExtend), 39 (Partner Closeout).
Projects the payload fields the indexer and structured retriever expect,
and yields one dict per row.

Column mapping from ``auction_audit_cln`` source columns to query aliases:
  - ``auction_id``                        → ``auction_id``        (VARCHAR)
  - ``domain_name``                       → ``domain_name``       (string)
  - ``auction_type_id``                   → ``auction_type``      (VARCHAR)
  - ``current_price_usd_amt``             → ``current_price``     (DOUBLE)
  - ``valuation_usd_amt``                 → ``govalue_score``     (DOUBLE)
  - ``auction_end_utc_ts``               → ``end_time``          (VARCHAR)
  - ``bid_cnt``                           → ``bid_count``         (INTEGER)
  - ``monthly_traffic_cnt``              → ``monthly_traffic``   (BIGINT)

Columns NOT present in ``auction_audit_cln`` (domain_age_years, majestic_*,
tlf_*, semrush_*) are absent from the query; ``row.get()`` returns None for
those fields and the ``or 0`` consumer fallbacks produce zero values.

The factory callable conforms to ``DocumentSourceFactory`` (from
``refresh_driver``): ``async (snapshot_version: int) -> List[Dict]``.

Usage in ``registry.py`` (after ``build_subsystems`` returns)::

    from semantic_search.vectorization.document_source import athena_document_source_factory
    factory = athena_document_source_factory(athena_client, config)
    subsystems.vector_refresh_driver._source_factory = factory

Or wire it directly during registry construction by replacing the
``_default_doc_source_factory`` closure.
"""

import math as _math
import re
from typing import Any, Callable, Awaitable, Dict, List, Optional

from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)

_ATHENA_QUERY = """\
SELECT
    CAST(auction_id AS VARCHAR)                           AS auction_id,
    domain_name,
    LOWER(SUBSTRING(domain_name FROM POSITION('.' IN domain_name) + 1))
                                                          AS tld,
    CAST(auction_type_id AS VARCHAR)                      AS auction_type,
    CAST(current_price_usd_amt AS DOUBLE)                 AS current_price,
    COALESCE(CAST(valuation_usd_amt AS DOUBLE), 0.0)      AS govalue_score,
    CAST(auction_end_utc_ts AS VARCHAR)                   AS end_time,
    COALESCE(CAST(bid_cnt AS INTEGER), 0)                 AS bid_count,
    COALESCE(CAST(monthly_traffic_cnt AS BIGINT), 0)      AS monthly_traffic
FROM {table}
WHERE auction_type_id IN (16, 20, 38, 39)
  AND domain_name IS NOT NULL
  AND LENGTH(domain_name) > 0
"""

def athena_document_source_factory(athena_client: Any, table: str, timeout_seconds: float) -> Callable[[int], Awaitable[List[Dict[str, Any]]]]:
    """Build a document-source factory backed by an Athena query.
    :param athena_client: AthenaClient - Constructed Athena client
    :param table: str - Athena table name (e.g. ``auction_audit_cln``)
    :param timeout_seconds: float - Athena query timeout (required, no default)
    :return: Callable[[int], Awaitable[List[Dict]]] - DocumentSourceFactory
    """
    query = _ATHENA_QUERY.format(table=table)

    async def _factory(snapshot_version: int) -> List[Dict[str, Any]]:
        """Fetch auction domains from Athena for the given snapshot version.
        :param snapshot_version: int - Snapshot version (logged, not used in query)
        :return: List[Dict[str, Any]] - Payload dicts for the OfflineIndexer
        """
        if not athena_client.credentials_available:
            logger.warning(f"athena_doc_source_skipped reason=credentials_unavailable snapshot_version={snapshot_version}")
            return []
        rows, columns, latency_ms = await athena_client.fetch_sql_async(query=query, timeout_seconds=timeout_seconds)
        logger.info(f"athena_doc_source_fetched rows={len(rows)} latency_ms={latency_ms:.1f} snapshot_version={snapshot_version}")
        docs: List[Dict[str, Any]] = []
        for row in rows:
            domain = row.get("domain_name")
            if not domain or not isinstance(domain, str) or not domain.strip():
                continue
            cleaned = domain.strip().lower()
            _raw_price = row.get("current_price")
            _price: Optional[float] = float(_raw_price) if _raw_price is not None else None
            _govalue = _safe_float(row.get("govalue_score"))
            _quality = min(max(_govalue / 100.0, 0.0), 1.0)
            sld = cleaned.split(".")[0] if "." in cleaned else cleaned
            doc: Dict[str, Any] = {
                "domain": cleaned,
                "domain_name": cleaned,
                "item_id": str(row.get("auction_id", "")),
                "auction_id": str(row.get("auction_id", "")),
                "tld": str(row.get("tld", "")).lower(),
                "auction_type": str(row.get("auction_type", "")),
                "price": _price,
                "name_length": len(sld),
                "quality": _quality,
                "score": _quality,
                "govalue_score": _govalue,
                "ends_at": str(row.get("end_time", "")),
                # Character analysis
                "sld": sld,
                "has_hyphen": int("-" in sld),
                "has_number": int(bool(re.search(r"\d", sld))),
                # Punycode (xn-- prefix) is ASCII but IS an IDN domain; check both.
                "is_idn": int(sld.startswith("xn--") or not sld.isascii()),
                # Auction enrichment
                "bid_count": int(row.get("bid_count") or 0),
                "domain_age_years": int(row.get("domain_age_years") or 0),
                "monthly_traffic": int(row.get("monthly_traffic") or 0),
                # Majestic
                "majestic_tf": int(row.get("majestic_tf") or 0),
                "majestic_cf": int(row.get("majestic_cf") or 0),
                "majestic_backlinks": int(row.get("majestic_backlinks") or 0),
                "majestic_ref_domains": int(row.get("majestic_ref_domains") or 0),
                # TLF Insights
                "tlf_exact_match": int(row.get("tlf_exact_match") or 0),
                "tlf_keyword_regs": int(row.get("tlf_keyword_regs") or 0),
                "tlf_developed": int(row.get("tlf_developed") or 0),
                # SEMrush
                "semrush_backlinks": int(row.get("semrush_backlinks") or 0),
                "semrush_indexed_pages": int(row.get("semrush_indexed_pages") or 0),
                "semrush_ref_domains": int(row.get("semrush_ref_domains") or 0),
                "semrush_authority_score": float(row.get("semrush_authority_score") or 0.0),
                "semrush_search_volume": int(row.get("semrush_search_volume") or 0),
                "semrush_cpc": float(row.get("semrush_cpc") or 0.0),
                # Traffic proxy features
                **_compute_traffic_features_athena(
                    monthly_traffic=int(row.get("monthly_traffic") or 0),
                    semrush_search_volume=int(row.get("semrush_search_volume") or 0),
                    semrush_authority_score=float(row.get("semrush_authority_score") or 0.0),
                    semrush_indexed_pages=int(row.get("semrush_indexed_pages") or 0),
                    majestic_tf=int(row.get("majestic_tf") or 0),
                ),
                "_snapshot_version": snapshot_version,
            }
            docs.append(doc)
        return docs

    return _factory


def _compute_traffic_features_athena(
    monthly_traffic: int,
    semrush_search_volume: int,
    semrush_authority_score: float,
    semrush_indexed_pages: int,
    majestic_tf: int,
) -> Dict[str, Any]:
    """Derive traffic proxy features for Athena-sourced documents."""
    proxy = (
        _math.log1p(monthly_traffic) * 0.40
        + _math.log1p(semrush_search_volume) * 0.25
        + _math.log1p(semrush_authority_score * 10.0) * 0.15
        + _math.log1p(semrush_indexed_pages) * 0.10
        + _math.log1p(majestic_tf) * 0.05
    )
    has_signal = int(monthly_traffic > 0 or semrush_search_volume > 0 or semrush_indexed_pages > 100 or majestic_tf > 0)
    if monthly_traffic >= 10000:
        tier = 4
    elif monthly_traffic >= 1000:
        tier = 3
    elif monthly_traffic >= 100:
        tier = 2
    elif monthly_traffic >= 1 or semrush_search_volume > 0 or majestic_tf > 0:
        tier = 1
    else:
        tier = 0
    return {
        "traffic_proxy_score": round(proxy, 4),
        "has_web_traffic_signal": has_signal,
        "estimated_traffic_tier": tier,
    }


def _safe_float(value: Any) -> float:
    """Coerce to float; return 0.0 on failure.
    :param value: Any - Value to coerce
    :return: float - Coerced value or 0.0
    """
    if value is None:
        return 0.0
    try:
        f = float(value)
        if f != f:
            return 0.0
        return f
    except (ValueError, TypeError):
        return 0.0
