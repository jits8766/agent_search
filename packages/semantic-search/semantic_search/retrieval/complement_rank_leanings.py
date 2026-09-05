"""Bounded reorder of ranked_results from analytics / guidance aggregates."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence

from semantic_search.config.models import ComplementRankLeaningsConfig
from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)

_PERIOD_ORDER = ('last_30d', 'last_7d', 'last_24h', 'last_1h')
_COUNT_KEYS = frozenset({
    'total_auctions', 'auctions_with_bids', 'total_bids', 'count', 'row_count',
    'listings', 'auctions', 'n',
})
_PRICE_KEYS = frozenset({'avg_price', 'mean_price', 'median_price', 'max_price'})


def _norm_tld(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip().lower().lstrip('.')
    return s or None


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _minmax_normalize(raw: Mapping[str, float]) -> Dict[str, float]:
    if not raw:
        return {}
    vals = list(raw.values())
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span <= 0.0:
        return {k: 1.0 for k in raw}
    return {k: (v - lo) / span for k, v in raw.items()}


def _row_activity(row: Mapping[str, Any]) -> Optional[float]:
    total = 0.0
    seen = False
    for key in _COUNT_KEYS:
        if key in row:
            f = _as_float(row.get(key))
            if f is not None and f >= 0.0:
                total += f
                seen = True
    for key in _PRICE_KEYS:
        if key in row:
            f = _as_float(row.get(key))
            if f is not None and f >= 0.0:
                total += f
                seen = True
    return total if seen else None


def _scores_from_rows(rows: Sequence[Any]) -> Dict[str, float]:
    raw: Dict[str, float] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        tld = _norm_tld(row.get('tld'))
        if tld is None:
            continue
        activity = _row_activity(row)
        if activity is None:
            continue
        raw[tld] = max(raw.get(tld, 0.0), activity)
    return _minmax_normalize(raw)


def extract_guidance_tld_scores(body: Any) -> Dict[str, float]:
    """Parse guidance snapshot body into per-TLD activity scores in [0, 1]."""
    try:
        if body is None:
            return {}
        if isinstance(body, str):
            text = body.strip()
            if not text:
                return {}
            parsed = json.loads(text)
        else:
            parsed = body
        if isinstance(parsed, list):
            return _scores_from_rows(parsed)
        if isinstance(parsed, Mapping) and isinstance(parsed.get('rows'), list):
            return _scores_from_rows(parsed['rows'])
        return {}
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.debug(f"guidance_tld_scores_parse_failed error_type={type(exc).__name__} error={exc}")
        return {}


def _analytics_row_lists(analytics_data: Mapping[str, Any]) -> List[Sequence[Any]]:
    lists: List[Sequence[Any]] = []
    periods = analytics_data.get('periods')
    if isinstance(periods, Mapping):
        for key in _PERIOD_ORDER:
            block = periods.get(key)
            if isinstance(block, Mapping) and isinstance(block.get('rows'), list) and block['rows']:
                lists.append(block['rows'])
                break
        if not lists:
            for block in periods.values():
                if isinstance(block, Mapping) and isinstance(block.get('rows'), list) and block['rows']:
                    lists.append(block['rows'])
                    break
    for key in ('rows', 'result_rows'):
        rows = analytics_data.get(key)
        if isinstance(rows, list) and rows:
            lists.append(rows)
    execution = analytics_data.get('execution')
    if isinstance(execution, Mapping) and isinstance(execution.get('rows'), list) and execution['rows']:
        lists.append(execution['rows'])
    return lists


def extract_analytics_cohort_scores(analytics_data: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    """Parse analytics payload into per-TLD cohort scores in [0, 1]."""
    try:
        if not isinstance(analytics_data, Mapping) or not analytics_data:
            return {}
        for rows in _analytics_row_lists(analytics_data):
            scores = _scores_from_rows(rows)
            if scores:
                return scores
        return {}
    except (TypeError, ValueError, KeyError) as exc:
        logger.debug(f"analytics_cohort_scores_parse_failed error_type={type(exc).__name__} error={exc}")
        return {}


def apply_complement_rank_leanings(
    ranked: List[Dict[str, Any]],
    *,
    query_type: str,
    cfg: ComplementRankLeaningsConfig,
    guidance_body: Any = None,
    analytics_data: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Reorder serialized ranked_results using bounded complement leanings.

    Does not mutate coherence_score. Re-stamps rank. No-op when disabled,
    query_type not configured, empty pool, or no usable cohort scores.
    """
    if cfg is None or not cfg.enabled or len(ranked) < 2:
        return ranked
    if query_type not in frozenset(cfg.query_types):
        return ranked

    g_scores: Dict[str, float] = {}
    a_scores: Dict[str, float] = {}
    if query_type == 'guidance' or guidance_body is not None:
        g_scores = extract_guidance_tld_scores(guidance_body)
    if query_type == 'analytics' or analytics_data is not None:
        a_scores = extract_analytics_cohort_scores(analytics_data)
    if not g_scores and not a_scores:
        return ranked

    w_g = float(cfg.weight_guidance)
    w_a = float(cfg.weight_analytics)
    bonus_cap = float(cfg.max_bonus)

    def _cohort(r: Dict[str, Any]) -> float:
        tld = _norm_tld(r.get('tld'))
        if tld is None:
            return 0.0
        g = g_scores.get(tld, 0.0)
        a = a_scores.get(tld, 0.0)
        return w_g * g + w_a * a

    def _coherence(r: Dict[str, Any]) -> float:
        f = _as_float(r.get('coherence_score'))
        return 0.0 if f is None else f

    keyed = [
        (_coherence(r) + bonus_cap * _cohort(r), -i, r)
        for i, r in enumerate(ranked)
    ]
    keyed.sort(key=lambda t: (t[0], t[1]), reverse=True)
    out = [t[2] for t in keyed]
    for new_rank, row in enumerate(out, start=1):
        row['rank'] = new_rank
    logger.debug(
        f"complement_rank_leanings_applied query_type={query_type} "
        f"items={len(out)} guidance_tlds={len(g_scores)} analytics_tlds={len(a_scores)}"
    )
    return out


__all__ = [
    'extract_guidance_tld_scores',
    'extract_analytics_cohort_scores',
    'apply_complement_rank_leanings',
]
