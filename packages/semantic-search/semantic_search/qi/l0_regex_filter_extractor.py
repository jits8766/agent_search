"""Regex L0 filter extractor - offline fallback for qie_only and full-search.

Reuses the same ``RegexEntityExtractor`` parsers as full-search. Output allowlist
matches L0 LLM / extract-script grounding: FIND-63 **plus** soft/local chips
(``FILTERABLE_PARAMS``). Soft chips are included (not FIND-only).

After regex parse, runs ``apply_post_merge_reconcile`` (same scrub full-search
applies when LLM is unavailable) so qie_only fallback stays slot-aligned.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from semantic_search.core.exceptions import ConfigurationError
from semantic_search.core.logging_utils import get_logger
from semantic_search.qi.entity_reconcile import apply_post_merge_reconcile
from semantic_search.qi.l0_llm_filter_extractor import (
    FILTERABLE_PARAMS,
    _FILTERABLE_PARAM_SET,
    filters_to_identified,
)
from semantic_search.qi.advisory_patterns import is_strong_advisory
from semantic_search.qi.regex_entity_extractor import RegexEntityExtractor
from semantic_search.qi.slot_to_api_param import (
    FIND_FILTERABLE_API_PARAMS,
    get_find_api_param_name,
    transform_slot_value,
)

logger = get_logger(__name__)

_REGEX_KW_STOP: frozenset = frozenset({
    # stop words
    "i", "a", "an", "the", "to", "and", "or", "but", "if", "in", "on", "at",
    "of", "for", "with", "by", "from", "is", "are", "was", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "not", "no", "nor", "so", "yet", "as", "it",
    "its", "that", "this", "these", "those", "my", "your", "our", "their",
    "what", "which", "who", "when", "where", "how", "all", "any", "both",
    "some", "more", "most", "than", "into", "up", "out", "about", "over",
    "just", "only", "also", "can", "must", "me", "we", "they", "you", "he",
    "she", "him", "his", "her", "us", "them", "go", "get", "too",
    # domain-context terms
    "domain", "domains", "website", "site", "url", "tld", "extension",
    "com", "net", "org", "io", "ai", "co", "app", "dev", "xyz", "info",
    "biz", "name", "names",
    # generic search intent / action words
    "buy", "sell", "find", "search", "register", "want", "need",
    "look", "looking", "show", "make", "give", "take", "want",
    # filler / qualifier words
    "good", "great", "nice", "best", "like", "new", "old", "high", "low",
    "big", "small", "short", "long", "under", "over", "above", "below",
    "price", "priced", "cost", "cheap", "expensive", "affordable",
    "available", "listing", "listings", "auction", "auctions",
})

_REGEX_TOKEN_RE = re.compile(r"[a-zA-Z]{2,}")


def _extract_topical_keywords(
    query: str,
    identified: List[Dict[str, Any]],
    *,
    probability: float = 0.5,
) -> List[Dict[str, Any]]:
    """Extract topical keywords from query text (stopword / TLD / slot-value stripped).

    ``probability`` is caller-supplied: regex offline path uses 0.5 (no confidence
    signal); LLM empty-keyword fill uses ``keyword_min_probability`` so terms pass
    the configured threshold.
    """
    if not 0.0 <= float(probability) <= 1.0:
        raise ConfigurationError(
            "_extract_topical_keywords probability must be in [0,1]"
        )
    already_in_slots: set = set()
    for item in identified:
        val = item.get("value")
        if isinstance(val, str):
            for part in re.split(r"[|,\s]+", val.lower()):
                p = part.strip()
                if p:
                    already_in_slots.add(p)
        elif isinstance(val, list):
            for v in val:
                for part in re.split(r"[|,\s]+", str(v).lower()):
                    p = part.strip()
                    if p:
                        already_in_slots.add(p)
    tokens = _REGEX_TOKEN_RE.findall(query.lower())
    seen: set = set()
    keywords: List[Dict[str, Any]] = []
    prob = float(probability)
    for tok in tokens:
        if tok in _REGEX_KW_STOP or tok in already_in_slots or tok in seen:
            continue
        seen.add(tok)
        keywords.append({"term": tok, "probability": prob})
    return keywords


def _resolve_filterable_param(slot_or_param: str) -> Optional[str]:
    """Map internal slot or already-catalog name to FILTERABLE_PARAMS entry."""
    if not slot_or_param:
        return None
    if slot_or_param in _FILTERABLE_PARAM_SET:
        return slot_or_param
    if slot_or_param in FIND_FILTERABLE_API_PARAMS:
        return slot_or_param
    return get_find_api_param_name(slot_or_param)


class L0RegexFilterExtractor:
    """Offline FIND-63 + soft/local extract; same allowlist as L0 LLM grounding."""

    def __init__(self, regex_extractor: RegexEntityExtractor) -> None:
        if regex_extractor is None:
            raise ConfigurationError("L0RegexFilterExtractor requires regex_extractor")
        if not FILTERABLE_PARAMS:
            raise ConfigurationError(
                "L0RegexFilterExtractor: FILTERABLE_PARAMS empty - "
                "check l0_llm_filter_extractor / find_api_params.json"
            )
        self._regex = regex_extractor
        self._config = regex_extractor._config

    async def extract_priced(
        self, query: str
    ) -> Tuple[List[Dict[str, Any]], float, List[Dict[str, Any]]]:
        """Like extract() but returns (identified, 0.0, keywords) for interface parity with L0LLMFilterExtractor."""
        identified = await self.extract(query)
        keywords = _extract_topical_keywords(query, identified)
        return identified, 0.0, keywords

    async def extract(self, query: str) -> List[Dict[str, Any]]:
        """Extract FIND + soft/local filters; same shape as L0LLMFilterExtractor."""
        if not isinstance(query, str) or not query.strip():
            return []
        if not self._config.enabled:
            return []
        q = query.strip()
        # Strong advisory/analytics yields [] (do not let reconcile re-inject filters).
        if is_strong_advisory(q):
            return []
        slice_ = await self._regex.classify_async(q)
        combined: List[Any] = []
        if slice_ is not None:
            combined = list(slice_.entities or []) + list(
                getattr(slice_, 'soft_entities', None) or []
            )
        # Parity with full-search regex fallback: post-merge inject-if-absent scrub.
        # Run even when regex slots are empty so soft lexicon cues (high estibot,
        # topic injectors, etc.) still materialize filters.
        hard_names = getattr(self._regex, '_hard_names', None)
        if isinstance(hard_names, frozenset) and hard_names:
            combined = apply_post_merge_reconcile(q, combined, hard_names)
        if not combined:
            return []
        filters: List[Dict[str, Any]] = []
        seen: set = set()
        cap = int(self._config.max_entities)
        for ent in combined:
            raw_name = str(getattr(ent, 'name', '') or '')
            api_param = _resolve_filterable_param(raw_name)
            if api_param is None or api_param not in _FILTERABLE_PARAM_SET or api_param in seen:
                continue
            raw_value = getattr(ent, 'value', None)
            if raw_value is None:
                continue
            # Prefer LLM-style relative offsets for listing age / end urgency
            # (prompt: startTimeAfter="-Nd"/"-Nh"; endTimeBefore="-1d" for tonight/soon).
            if raw_name == 'startTimeAfter' and isinstance(raw_value, str) and raw_value.startswith('-'):
                api_param = 'startTimeAfter'
                value = raw_value
            elif raw_name == 'days_listed_max' and isinstance(raw_value, (int, float)):
                days = int(raw_value)
                if days > 0:
                    api_param = 'startTimeAfter'
                    value = f"-{days}d"
                else:
                    value = transform_slot_value(raw_name, raw_value)
            elif raw_name == 'days_listed_min' and isinstance(raw_value, (int, float)):
                # "stale over 2 weeks" maps to startTimeBefore="-14d" (LLM relative, not ISO).
                days = int(raw_value)
                if days > 0:
                    api_param = 'startTimeBefore'
                    value = f"-{days}d"
                else:
                    value = transform_slot_value(raw_name, raw_value)
            elif raw_name == 'time_remaining_max' and isinstance(raw_value, (int, float)):
                secs = int(raw_value)
                api_param = 'endTimeBefore'
                if secs == 86_400:
                    value = '-1d'
                elif secs == 259_200:
                    value = '-3d'
                elif secs == 604_800:
                    value = '-7d'
                elif 0 < secs < 86_400:
                    # Sub-day horizon: emit -Nh (e.g. 21600s -> -6h, 3600s -> -1h).
                    hours = max(1, secs // 3600)
                    value = f'-{hours}h'
                elif secs <= 259_200:
                    value = '-3d'
                elif secs <= 604_800:
                    value = '-7d'
                else:
                    value = transform_slot_value(raw_name, raw_value)
            elif raw_name in _FILTERABLE_PARAM_SET or raw_name in FIND_FILTERABLE_API_PARAMS:
                value = raw_value
            else:
                value = transform_slot_value(raw_name, raw_value)
            if value is None:
                continue
            seen.add(api_param)
            filters.append({'param': api_param, 'value': value})
            if len(filters) >= cap:
                break
        identified = filters_to_identified(
            filters,
            source=str(self._config.source_tag),
            soft_slot_names=self._regex._soft_slots,
            confidence=float(self._config.confidence),
            query=q,
        )
        logger.info(
            f"l0_regex_filter_extract_done query_len={len(query)} "
            f"filters={len(identified)} allowlist={len(_FILTERABLE_PARAM_SET)}"
        )
        return identified
