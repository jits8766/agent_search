"""qie_only helpers: L0 filter cache, grounding shape, FIND wire attach, ops metrics.

Kept out of ``app.py`` so the FastAPI module owns request lifecycle only.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from semantic_search.cache.lru_ttl import LRUTTLCache
from semantic_search.config.loader import load_config
from semantic_search.config.models import AgentSearchConfig
from semantic_search.core.exceptions import ConfigurationError
from semantic_search.core.logging_utils import get_logger
from semantic_search.qi.entity_reconcile import apply_post_merge_reconcile
from semantic_search.qi.find_query_params import build_find_wire_payload
from semantic_search.qi.grounding import (
    ground_identified_filters,
    sanitize_identified_filters,
)
from semantic_search.qi.keyword_threshold import min_probability_fraction
from semantic_search.qi.l0_llm_filter_extractor import entities_to_identified

logger = get_logger(__name__)

# Value shape: {"identified": [...], "keywords": [...], "query_transform": dict|None}.
_QIE_L0_FILTER_CACHE: Optional[LRUTTLCache[Any]] = None


def _get_qie_l0_filter_cache() -> LRUTTLCache[Any]:
    """Return the qie_only L0 filter cache, constructing from config on first use."""
    global _QIE_L0_FILTER_CACHE
    if _QIE_L0_FILTER_CACHE is None:
        cfg = AgentSearchConfig.from_dict(
            load_config()
        ).general.search.qie_l0_filter_cache
        _QIE_L0_FILTER_CACHE = LRUTTLCache(
            max_entries=int(cfg.max_entries),
            ttl_seconds=int(cfg.ttl_seconds),
        )
    return _QIE_L0_FILTER_CACHE


def _clear_qie_l0_filter_cache() -> int:
    """Drop every qie_only L0 filter-cache entry. Used by POST /cache/clear.

    :return: int - Entries dropped (0 when never constructed or already empty)
    """
    global _QIE_L0_FILTER_CACHE
    if _QIE_L0_FILTER_CACHE is None:
        return 0
    dropped = len(_QIE_L0_FILTER_CACHE)
    _QIE_L0_FILTER_CACHE.clear()
    if dropped > 0:
        logger.info(f"qie_l0_filter_cache_cleared entries_dropped={dropped}")
    return dropped


def _qie_cache_envelope(
    identified: List[Dict[str, Any]],
    keywords: List[Dict[str, Any]],
    query_transform: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the qie_only L0 cache value (filters + keywords + optional transform)."""
    return {
        "identified": list(identified or []),
        "keywords": list(keywords or []),
        "query_transform": dict(query_transform) if isinstance(query_transform, dict) else None,
    }


def _unpack_qie_cache_entry(
    cached: Any,
) -> Optional[Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[Dict[str, Any]]]]:
    """Unpack envelope cache entry -> (identified, keywords, query_transform).

    List-only / non-dict entries are rejected (no legacy shape).
    """
    if cached is None:
        return None
    if not isinstance(cached, dict):
        return None
    identified = list(cached.get("identified") or [])
    keywords = list(cached.get("keywords") or [])
    qt = cached.get("query_transform")
    if not isinstance(qt, dict):
        qt = None
    if not identified and not keywords and qt is None:
        return None
    return identified, keywords, qt


def _qie_grounding_context(
    sub: Any,
) -> Tuple[Any, frozenset, frozenset, Any]:
    """Resolve (entity_extractor, hard_names, soft_names, grounder) for reconcile+ground."""
    _qi_engine = getattr(sub, "qi_engine", None)
    grounder = (
        getattr(_qi_engine, "_grounder", None) if _qi_engine is not None else None
    )
    extractor = (
        getattr(_qi_engine, "_entity_extractor", None)
        if _qi_engine is not None
        else None
    )
    soft_names: frozenset = frozenset()
    hard_names: frozenset = frozenset()
    if _qi_engine is not None:
        try:
            soft_names, hard_names = _qi_engine._slot_sets()
        except (ConfigurationError, ValueError, TypeError) as _slot_exc:
            logger.warning(
                f"qie_reconcile_slot_sets_unavailable error_type={type(_slot_exc).__name__}"
            )
    return extractor, hard_names, soft_names, grounder


def _qie_identified_from_intent(
    intent: Any,
    soft_names: frozenset,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """Map ``QueryIntent`` slices to API-shaped identified lists.

    :return: (grounded_identified, pre_ground_identified, grounded_drop_count)
    """
    pre_hard: List[Any] = []
    grounded_hard: List[Any] = []
    soft_all: List[Any] = []
    for s in getattr(intent, "slices", None) or []:
        pre = getattr(s, "pre_ground_entities", None)
        if pre is not None:
            pre_hard.extend(list(pre))
        else:
            pre_hard.extend(list(getattr(s, "entities", None) or []))
        grounded_hard.extend(list(getattr(s, "entities", None) or []))
        soft_all.extend(list(getattr(s, "soft_entities", None) or []))
    pre_identified = entities_to_identified(list(pre_hard) + list(soft_all), soft_names)
    grounded_identified = entities_to_identified(
        list(grounded_hard) + list(soft_all), soft_names,
    )
    drop_count = max(0, len(pre_hard) - len(grounded_hard))
    return grounded_identified, pre_identified, drop_count


def reconcile_and_ground_identified(
    identified: List[Dict[str, Any]],
    q_norm: str,
    *,
    extractor: Any,
    hard_names: frozenset,
    soft_names: frozenset,
    grounder: Any,
) -> Tuple[List[Dict[str, Any]], int]:
    """Cue-reconcile then inventory-ground a raw L0 filter list.

    Always runs ``apply_post_merge_reconcile`` when hard slot names are available —
    including on an empty extract — so inject-if-absent enrichers (listing age,
    ending soon, etc.) stay aligned between ``qie_only`` (IntentSlice cue reconcile)
    and ``/internal/l0_ground`` (LLMJ / regex offline arms). Skipping reconcile on
    empty identified previously left offline arms under-injected vs live QIE.
    """
    if extractor is not None and hard_names:
        try:
            _combined: List[Any] = []
            if identified:
                _slice = extractor._identified_to_intent_slice(identified, q_norm)
                if _slice is not None:
                    _combined = list(_slice.entities or []) + list(
                        _slice.soft_entities or []
                    )
            _reconciled = apply_post_merge_reconcile(
                q_norm, _combined, hard_names
            )
            identified = entities_to_identified(_reconciled, soft_names)
        except Exception as _rec_exc:  # noqa: BLE001 - best-effort; never fail the request
            logger.warning(f"qie_reconcile_failed error_type={type(_rec_exc).__name__}")
    if grounder is not None:
        identified, drop_count = ground_identified_filters(identified, grounder)
    else:
        identified = sanitize_identified_filters(identified)
        drop_count = 0
    return identified, drop_count


def _attach_find_wire_fields(
    body: Dict[str, Any],
    identified: List[Dict[str, Any]],
    keywords: Optional[List[Dict[str, Any]]] = None,
    *,
    get_subsystems: Callable[[], Any],
) -> Dict[str, Any]:
    """Attach FIND auction/recommend wire fields to a qie_only response body.

    Keyword probability gate comes from ``qi.l0_llm_entity.keyword_min_probability``.
    """
    sub = get_subsystems()
    min_frac = min_probability_fraction(sub.config.qi.l0_llm_entity)
    wire = build_find_wire_payload(
        identified,
        wire=sub.config.general.search.find_wire,
        keywords=keywords,
        min_keyword_probability=min_frac,
    )
    body["find_query_params"] = wire["find_query_params"]
    body["find_query_string"] = wire["find_query_string"]
    body["soft_chips"] = wire["soft_chips"]
    body["find_skipped"] = wire["find_skipped"]
    return body


def _qie_only_ops_metrics(body: Dict[str, Any]) -> Dict[str, Any]:
    """FIND-wire ops metrics for logs + Phase 1 launch stats."""
    skipped = body.get("find_skipped") or []
    if not isinstance(skipped, list):
        skipped = []
    soft = body.get("soft_chips") or []
    if not isinstance(soft, list):
        soft = []
    params = body.get("find_query_params") or {}
    if not isinstance(params, dict):
        params = {}
    hard_keys = [k for k in params.keys() if k != "query"]
    reasons: List[str] = []
    seen: Set[str] = set()
    for item in skipped:
        if not isinstance(item, dict):
            continue
        reason = str(item.get("reason") or "").strip()
        if not reason or reason in seen:
            continue
        seen.add(reason)
        reasons.append(reason)
    return {
        "find_skipped_count": len(skipped),
        "find_skipped_reasons": ",".join(reasons) if reasons else "-",
        "soft_chip_count": len(soft),
        "hard_params_empty": 0 if hard_keys else 1,
    }


def _qie_only_ops_log_fields(body: Dict[str, Any]) -> str:
    """Compact FIND-wire ops fields for ``qie_only_complete`` CloudWatch logs."""
    m = _qie_only_ops_metrics(body)
    return (
        f"find_skipped_count={m['find_skipped_count']} "
        f"find_skipped_reasons={m['find_skipped_reasons']} "
        f"soft_chip_count={m['soft_chip_count']} "
        f"hard_params_empty={m['hard_params_empty']}"
    )


__all__ = [
    "_get_qie_l0_filter_cache",
    "_clear_qie_l0_filter_cache",
    "_qie_cache_envelope",
    "_unpack_qie_cache_entry",
    "_qie_grounding_context",
    "_qie_identified_from_intent",
    "reconcile_and_ground_identified",
    "_attach_find_wire_fields",
    "_qie_only_ops_metrics",
    "_qie_only_ops_log_fields",
]
