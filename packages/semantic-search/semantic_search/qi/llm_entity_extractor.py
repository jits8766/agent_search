"""API↔internal slot maps and post-extract keyword scrubs.

Production L0 LLM extract is ``L0LLMFilterExtractor`` only. This module is not
an extractor — it holds name maps used by ``classify_async`` mapping and
keyword scrub helpers used by engine merge / tests.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List

from semantic_search.contracts import Entity
from semantic_search.core.logging_utils import get_logger
from semantic_search.qi.entity_reconcile import KEYWORD_MATCH_MODE_STANDALONE
from semantic_search.qi.keyword_meta import KEYWORD_META_BLOCKLIST

logger = get_logger(__name__)

# Slot names that map to backend API parameters (find_api_params.json allowlist).
# Any entity whose name is NOT in this set is dropped before returning to the engine.
# Names here are INTERNAL engine slot names — see _API_TO_INTERNAL for the
# API-name -> internal-name translation applied before this filter.
VALID_ENTITY_NAMES: frozenset = frozenset({
    # TLD
    "tld",
    "tldExcludeList",
    # Price
    "price_min",
    "price_max",
    "filterPriceCurrency",
    # Auction type
    "auction_type",
    "typeExcludeList",
    "isExtended",
    "isBidAccepted",
    # Bids
    "bids_min",
    "bids_max",
    # Domain name constraints
    "has_hyphen",
    "has_number",
    "is_idn",
    "name_length_min",
    "name_length_max",
    "excludeLetters",
    "minDigits",
    "minLetters",
    "charPattern",
    "isGemDomain",
    # Owner/seller
    "ownerMemberIncludeList",
    "ownerMemberExcludeList",
    # Domain age
    "domain_age_min",
    "domain_age_max",
    "domain_age_is_unknown",
    # Traffic
    "traffic_min",
    "traffic_max",
    "traffic_is_unknown",
    # Traffic proxy features
    "traffic_proxy_min",
    "traffic_proxy_max",
    "has_web_traffic_signal",
    "price_below_market",
    "buy_it_now",
    "estimated_traffic_tier_min",
    "estimated_traffic_tier_max",
    # GoValue / valuation
    "govalue_min",
    "govalue_max",
    # Majestic
    "majestic_tf_min",
    "majestic_tf_max",
    "majestic_cf_min",
    "majestic_cf_max",
    "majestic_backlinks_min",
    "majestic_backlinks_max",
    "majestic_ref_domains_min",
    "majestic_ref_domains_max",
    # SEMrush
    "semrush_authority_min",
    "semrush_authority_max",
    "semrush_search_volume_min",
    "semrush_search_volume_max",
    "semrush_cpc_min",
    "semrush_cpc_max",
    "semrush_backlinks_min",
    "semrush_backlinks_max",
    "semrush_indexed_pages_min",
    "semrush_indexed_pages_max",
    "semrush_ref_domains_min",
    "semrush_ref_domains_max",
    # Unique searches
    "minUniqueSearches",
    "maxUniqueSearches",
    # Estibot
    "minEstibotDomainCount",
    "maxEstibotDomainCount",
    "minEstibotDomainCountDev",
    "maxEstibotDomainCountDev",
    "minEstibotExtCount",
    "maxEstibotExtCount",
    "minEstibotExtCountDev",
    "maxEstibotExtCountDev",
    # Time (ISO dates from explicit dates; relative time -> time_remaining_max / days_listed_max)
    "endTimeBefore",
    "endTimeAfter",
    "startTimeBefore",
    "startTimeAfter",
    # Relative temporal (computed by L0, integer seconds / days)
    "time_remaining_max",
    "days_listed_max",
    "days_listed_min",
    # Keyword / text search
    "keyword_contains",
    "keyword_starts_with",
    "keyword_ends_with",
    "keyword_contains_exclude",
    "keyword_match_mode",
    "keyword_phrase",
    # Semantic similarity
    "similar_to",
    # Topic / industry
    "topic_include",
    "topic_exclude",
    # Domain word count
    "word_count_min",
    "word_count_max",
    # Lifecycle
    "lifecycle_state",
    "lifecycle_disjunction",
    # Auction feature flags backed by local Qdrant payload (no FIND API param)
    "has_reserve_price",
    "gd_transfer",
})

# Translates find_api_params.json camelCase API names -> internal engine slot names.
# Prompts use API names; this map is applied before VALID_ENTITY_NAMES filtering so
# all downstream engine/grounding code remains unchanged.
_API_TO_INTERNAL: Dict[str, str] = {
    # TLD
    "tldIncludeList": "tld",
    # Auction type
    "typeIncludeList": "auction_type",
    # Price
    "minPrice": "price_min",
    "maxPrice": "price_max",
    # Bids
    "minBids": "bids_min",
    "maxBids": "bids_max",
    # Domain name length
    "minSldLen": "name_length_min",
    "maxSldLen": "name_length_max",
    # Domain age (years)
    "minAge": "domain_age_min",
    "maxAge": "domain_age_max",
    # Traffic (monthly visitors)
    "minTraffic": "traffic_min",
    "maxTraffic": "traffic_max",
    # Traffic proxy features
    "minTrafficProxyScore": "traffic_proxy_min",
    "maxTrafficProxyScore": "traffic_proxy_max",
    "hasWebTrafficSignal": "has_web_traffic_signal",
    "has_web_traffic_signal": "has_web_traffic_signal",
    "minEstimatedTrafficTier": "estimated_traffic_tier_min",
    "maxEstimatedTrafficTier": "estimated_traffic_tier_max",
    # Char constraints (FIND exclude* = invert of internal has_*)
    "excludeHyphens": "has_hyphen",
    "excludeDigits": "has_number",
    "excludeLetters": "excludeLetters",
    # Local bools (identity)
    "domain_age_is_unknown": "domain_age_is_unknown",
    "traffic_is_unknown": "traffic_is_unknown",
    "price_below_market": "price_below_market",
    "buy_it_now": "buy_it_now",
    "gd_transfer": "gd_transfer",
    "has_reserve_price": "has_reserve_price",
    # FIND isIdn → internal is_idn
    "isIdn": "is_idn",
    # GoValue / valuation
    "minValuationPrice": "govalue_min",
    "maxValuationPrice": "govalue_max",
    # Majestic
    "minMajesticTrustFlowScore": "majestic_tf_min",
    "maxMajesticTrustFlowScore": "majestic_tf_max",
    "minMajesticCitationFlowScore": "majestic_cf_min",
    "maxMajesticCitationFlowScore": "majestic_cf_max",
    "minMajesticBackLinks": "majestic_backlinks_min",
    "maxMajesticBackLinks": "majestic_backlinks_max",
    "minMajesticRefDomains": "majestic_ref_domains_min",
    "maxMajesticRefDomains": "majestic_ref_domains_max",
    # SEMrush
    "minSemrushAScore": "semrush_authority_min",
    "maxSemrushAScore": "semrush_authority_max",
    "minSemrushLinksTotal": "semrush_backlinks_min",
    "maxSemrushLinksTotal": "semrush_backlinks_max",
    "minSemrushDomainsNum": "semrush_ref_domains_min",
    "maxSemrushDomainsNum": "semrush_ref_domains_max",
    "minSemrushUrlsNum": "semrush_indexed_pages_min",
    "maxSemrushUrlsNum": "semrush_indexed_pages_max",
    "minSemrushSearchVolume": "semrush_search_volume_min",
    "maxSemrushSearchVolume": "semrush_search_volume_max",
    "minSemrushCostPerClick": "semrush_cpc_min",
    "maxSemrushCostPerClick": "semrush_cpc_max",
}

# FIND exclude* bools mean the opposite of internal has_* slots.
_API_INVERT_BOOL: frozenset = frozenset({"excludeHyphens", "excludeDigits"})


def _reconcile_keyword_topic(entities: List[Entity]) -> List[Entity]:
    """Drop literal keyword tokens that merely echo a topic seed.

    A keyword token that is a strict substring of a topic word (equal tokens
    are a genuine dual signal and kept) is removed; a keyword slot emptied of
    every token is dropped.
    """
    topic_words = {
        str(t).lower()
        for e in entities
        if e.name == "topic_include" and isinstance(e.value, list)
        for t in e.value
        if isinstance(t, str)
    }
    if not topic_words:
        return entities
    keyword_slots = ("keyword_contains", "keyword_starts_with", "keyword_ends_with")
    kept: List[Entity] = []
    for e in entities:
        if e.name not in keyword_slots or not isinstance(e.value, list):
            kept.append(e)
            continue
        filtered = [
            tok for tok in e.value
            if not (
                isinstance(tok, str)
                and any(tok.lower() != tw and tok.lower() in tw for tw in topic_words)
            )
        ]
        if len(filtered) != len(e.value):
            dropped = [tok for tok in e.value if tok not in filtered]
            logger.info(
                f"llm_entity_keyword_topic_reconciled name={e.name} dropped={dropped} "
                f"reason=substring_of_topic_seed"
            )
        if not filtered:
            continue
        kept.append(e if filtered == e.value else replace(e, value=filtered))
    return kept


def _reconcile_dangling_match_mode(entities: List[Entity]) -> List[Entity]:
    """Drop keyword_match_mode when fewer than two keyword tokens remain.

    Standalone values (config: keyword_match_mode_standalone_values, e.g. 'exact')
    and presence of keyword_phrase keep the mode even with 0–1 keyword tokens.
    """
    keyword_slots = ("keyword_contains", "keyword_starts_with", "keyword_ends_with")
    token_count = sum(
        len(e.value) if isinstance(e.value, list) else 1
        for e in entities
        if e.name in keyword_slots
    )
    if token_count >= 2:
        return entities
    if any(e.name == "keyword_phrase" for e in entities):
        return entities
    mode_vals = {
        str(e.value).lower()
        for e in entities
        if e.name == "keyword_match_mode"
    }
    if mode_vals & KEYWORD_MATCH_MODE_STANDALONE:
        return entities
    kept = [e for e in entities if e.name != "keyword_match_mode"]
    if len(kept) != len(entities):
        logger.info(
            f"llm_entity_match_mode_dropped keyword_tokens={token_count} "
            f"reason=insufficient_keywords"
        )
    return kept


def _reconcile_keyword_meta_blocklist(entities: List[Entity]) -> List[Entity]:
    """Drop structural meta-tokens from keyword slots (keyword_meta_blocklist.json)."""
    if not KEYWORD_META_BLOCKLIST:
        return entities
    keyword_slots = (
        'keyword_contains', 'keyword_starts_with', 'keyword_ends_with',
        'keyword_contains_exclude',
    )
    kept: List[Entity] = []
    for e in entities:
        if e.name not in keyword_slots:
            kept.append(e)
            continue
        if isinstance(e.value, list):
            valid: Any = [
                v for v in e.value
                if not (isinstance(v, str) and v.lower().strip() in KEYWORD_META_BLOCKLIST)
            ]
        elif isinstance(e.value, str):
            valid = [] if e.value.lower().strip() in KEYWORD_META_BLOCKLIST else e.value
        else:
            kept.append(e)
            continue
        if isinstance(e.value, list):
            if len(valid) != len(e.value):
                dropped = [v for v in e.value if v not in valid]
                logger.info(
                    f"llm_entity_keyword_meta_dropped name={e.name} dropped={dropped} "
                    f"reason=keyword_meta_blocklist"
                )
            if not valid:
                continue
            kept.append(e if valid == e.value else replace(e, value=valid))
        else:
            if valid == []:
                logger.info(
                    f"llm_entity_keyword_meta_dropped name={e.name} dropped=[{e.value!r}] "
                    f"reason=keyword_meta_blocklist"
                )
                continue
            kept.append(e)
    return kept
