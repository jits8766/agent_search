"""Soft-keyword apply subcomponent — post-QI rank boost unit.

:class:`SoftKeywordApplier` is the public surface (same shape as
``BrandabilityScorer`` / ``FuzzyLexicalReranker``). Wired on
``SearchOrchestrator`` from ``qi.entity_slots`` +
``retrieval.structured.keyword_match_mode``.

``soft_apply_mode: rank`` — soft on ``IntentSlice.soft_entities`` only;
FIND-hard drives Qdrant payload filters; soft keywords/topics boost
fused_score on the post-diversify pool before truncate. ``off`` disables
the boost (config kill-switch).
"""
from __future__ import annotations

import dataclasses
import re
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

from semantic_search.config.models import QIEntitySlotsConfig
from semantic_search.contracts import Entity, IntentSlice, QueryIntent, RankedItem, RankedResults
from semantic_search.core.logging_utils import get_logger
from semantic_search.retrieval.structured_retriever import (
    derive_sld_from_payload,
    keyword_value_matches,
)

logger = get_logger(__name__)

_SOFT_APPLY_MODES = frozenset({'rank', 'off'})


def _normalize_soft_apply_mode(raw: Any) -> str:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raise ValueError(
            f"soft_apply_mode is required; must be one of {sorted(_SOFT_APPLY_MODES)}"
        )
    mode = str(raw).strip().lower()
    if mode not in _SOFT_APPLY_MODES:
        raise ValueError(f"soft_apply_mode must be one of {sorted(_SOFT_APPLY_MODES)}, got {raw!r}")
    return mode


def _collect_soft_keyword_entities(
    intent: QueryIntent,
    rank_slot_names: FrozenSet[str],
) -> List[Entity]:
    """Collect rank-boost soft chips from L0 ``soft_entities`` (canonical).

    Soft arrives here from LLM L0, or from regex L0 when LLM was unavailable.
    Hard ``entities`` are not scanned — soft never lives there after prepare.
    """
    if not rank_slot_names:
        raise ValueError("_collect_soft_keyword_entities requires non-empty rank_slot_names")
    by_name: Dict[str, Entity] = {}
    for s in intent.slices or []:
        for ent in list(getattr(s, 'soft_entities', None) or []):
            if ent.name in rank_slot_names and ent.name not in by_name:
                by_name[ent.name] = ent
    return list(by_name.values())


def _strip_soft_keyword_slots_from_intent(
    intent: QueryIntent,
    soft_slot_names: FrozenSet[str],
) -> QueryIntent:
    """Remove soft slots from hard ``entities``; migrate onto ``soft_entities``.

    Soft mis-bucketed onto hard entities must not be discarded — move them so
    soft_signals / rank-boost still see L0 soft chips (grounding/qie parity).
    """
    if not soft_slot_names:
        raise ValueError("_strip_soft_keyword_slots_from_intent requires non-empty soft_slot_names")
    new_slices: List[IntentSlice] = []
    stripped_n = 0
    migrated_n = 0
    for s in intent.slices or []:
        soft = list(getattr(s, 'soft_entities', None) or [])
        seen_soft = {e.name for e in soft}
        kept: List[Entity] = []
        for e in s.entities or []:
            if e.name in soft_slot_names:
                stripped_n += 1
                if e.name not in seen_soft:
                    soft.append(dataclasses.replace(e, chip_kind='soft'))
                    seen_soft.add(e.name)
                    migrated_n += 1
                continue
            kept.append(e)
        _pre = getattr(s, 'pre_ground_entities', None)
        new_slices.append(IntentSlice(
            query_type=s.query_type,
            entities=kept,
            confidence=s.confidence,
            raw_text=s.raw_text,
            slice_id=getattr(s, 'slice_id', '') or '',
            soft_entities=soft,
            # Keep inventory pre-ground snapshot for response identified filters.
            pre_ground_entities=list(_pre) if _pre is not None else None,
            keywords=list(getattr(s, 'keywords', None) or []),
        ))
    if stripped_n:
        logger.info(
            f"soft_keyword_slots_stripped request_id={intent.request_id} "
            f"stripped_n={stripped_n} migrated_to_soft_n={migrated_n}"
        )
    return dataclasses.replace(intent, slices=new_slices)


def _soft_map(
    soft_entities: Sequence[Entity],
    rank_slot_names: FrozenSet[str],
) -> Dict[str, Any]:
    return {e.name: e.value for e in soft_entities if e.name in rank_slot_names}


def _word_count(sld: str) -> int:
    if not sld:
        return 0
    if '-' in sld:
        return max(1, len([p for p in sld.split('-') if p]))
    parts = re.findall(r'[a-z]+|[0-9]+', sld.lower())
    return max(1, len(parts)) if parts else 1


def _theme_terms(raw: Any) -> List[str]:
    """Expand topic_include / topic_exclude values into SLD substring tokens."""
    vals = [raw] if isinstance(raw, str) else list(raw or [])
    terms: List[str] = []
    seen: set = set()
    for v in vals:
        t = str(v).lower().strip().replace('_', ' ').replace('-', ' ')
        if not t:
            continue
        compact = t.replace(' ', '')
        if compact and compact not in seen:
            seen.add(compact)
            terms.append(compact)
        for tok in re.findall(r'[a-z0-9]+', t):
            if len(tok) >= 3 and tok not in seen:
                seen.add(tok)
                terms.append(tok)
    return terms


def _sld_theme_hit(sld: str, terms: Sequence[str]) -> bool:
    sld_l = (sld or '').lower()
    compact = sld_l.replace('-', '')
    return any(bool(t) and (t in compact or t in sld_l) for t in terms)


def _normalize_keyword_entries(
    keywords: Sequence[Any] = (),
) -> List[Tuple[str, float]]:
    """Normalize keyword inputs to ``(term, probability)`` sorted by prob desc.

    Accepts ``[{"term", "probability"}, ...]`` (L0 shape) or bare term strings
    (legacy / tests — treated as probability 1.0). No term-name hardcoding.
    """
    entries: List[Tuple[str, float]] = []
    seen: set = set()
    for item in keywords or ():
        term = ""
        prob = 1.0
        if isinstance(item, dict):
            term = str(item.get("term") or "").strip()
            raw_prob = item.get("probability")
            try:
                prob = float(raw_prob) if raw_prob is not None else 1.0
            except (TypeError, ValueError):
                continue
        elif isinstance(item, str):
            term = item.strip()
        else:
            continue
        if not term or prob != prob:  # NaN
            continue
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        entries.append((term, float(prob)))
    entries.sort(key=lambda tp: (-tp[1], tp[0].casefold()))
    return entries


def _keyword_hit_quality(
    sld: str,
    keyword_entries: Sequence[Tuple[str, float]],
) -> Tuple[int, float, List[Tuple[str, float]]]:
    """Count keyword terms present on an SLD (substring / concat, any order).

    :return: (hit_count, best_hit_probability, hit_entries) where hit_count is
        in ``[0, N]`` and is the recall numerator over ``keyword_entries``.
    """
    if not keyword_entries:
        return 0, 0.0, []
    hits: List[Tuple[str, float]] = [
        (term, prob)
        for term, prob in keyword_entries
        if _sld_theme_hit(sld, [term])
    ]
    if not hits:
        return 0, 0.0, []
    best_prob = max(p for _, p in hits)
    return len(hits), best_prob, hits


def _keyword_coverage_precision(
    sld: str,
    hits: Sequence[Tuple[str, float]],
) -> float:
    """Fraction of compact SLD characters covered by hit keyword terms.

    Matching is order-independent. Returns a value in ``[0, 1]``.
    """
    compact = re.sub(r'[^a-z0-9]', '', (sld or '').lower())
    if not compact or not hits:
        return 0.0
    remaining = compact
    covered = 0
    for term, _prob in sorted(hits, key=lambda tp: (-len(tp[0]), tp[0].casefold())):
        tok = re.sub(r'[^a-z0-9]', '', str(term).lower())
        if not tok:
            continue
        idx = remaining.find(tok)
        if idx < 0:
            continue
        covered += len(tok)
        remaining = remaining[:idx] + remaining[idx + len(tok):]
    return min(1.0, float(covered) / float(len(compact)))


def _keyword_f1(recall: float, precision: float) -> float:
    """Harmonic mean of recall and precision; 0 when both are 0."""
    r = float(recall)
    p = float(precision)
    if r <= 0.0 and p <= 0.0:
        return 0.0
    denom = r + p
    if denom <= 0.0:
        return 0.0
    return 2.0 * r * p / denom


def _primary_keyword_term(
    hits: Sequence[Tuple[str, float]],
) -> Optional[str]:
    """Highest-probability hit term (original spelling preserved)."""
    if not hits:
        return None
    return max(hits, key=lambda tp: (tp[1], tp[0].casefold()))[0]


def _reorder_by_keyword_metrics(
    scored_rows: Sequence[Tuple[int, float, float, int, RankedItem, str]],
    keyword_entries: Sequence[Tuple[str, float]],
) -> List[RankedItem]:
    """Order items by keyword F1, then hit_count on ties.

    Sort key (descending):
      1. F1(recall, precision) — recall = hit_count / N
      2. hit_count (more matched terms wins when F1 ties)
      3. precision, then recall
      4. fused score, then stable index

    Term order inside the SLD is ignored. Row shape:
    ``(hit_count, precision, score, neg_idx, item, primary_term)``.
    """
    n_kw = len(keyword_entries)
    if n_kw < 2:
        # Soft-chip-only or single keyword: score order (hit/precision as minor keys).
        ordered = sorted(
            scored_rows, key=lambda t: (t[0], t[1], t[2], t[3]), reverse=True,
        )
        return [t[4] for t in ordered]

    decorated: List[Tuple[float, int, float, float, float, int, RankedItem]] = []
    for hit_count, precision, score, neg_idx, item, _primary in scored_rows:
        recall = float(hit_count) / float(n_kw)
        f1 = _keyword_f1(recall, float(precision))
        decorated.append(
            (f1, int(hit_count), float(precision), recall, float(score), int(neg_idx), item)
        )
    decorated.sort(key=lambda t: (t[0], t[1], t[2], t[3], t[4], t[5]), reverse=True)
    return [t[6] for t in decorated]


def _similar_to_hit(sld: str, raw: Any) -> bool:
    """True when SLD shares a seed brand stem from similar_to soft chip."""
    seeds = [raw] if isinstance(raw, str) else list(raw or [])
    compact = (sld or '').lower().replace('-', '')
    if not compact:
        return False
    for seed in seeds:
        seed_sld = str(seed).lower().strip().split('.')[0].replace('-', '')
        if len(seed_sld) < 3:
            continue
        if seed_sld in compact or compact in seed_sld:
            return True
        if len(seed_sld) >= 4 and compact.startswith(seed_sld[:4]):
            return True
    return False


def _soft_keyword_match_score(
    sld: str,
    soft_entities: Sequence[Entity],
    *,
    rank_slot_names: FrozenSet[str],
    weight: float,
    miss_penalty_ratio: float,
    partial_boost_ratio: float,
    default_keyword_match_mode: str,
    keyword_terms: Sequence[Any] = (),
) -> float:
    if weight <= 0:
        raise ValueError("_soft_keyword_match_score weight must be positive")
    if not (0.0 <= miss_penalty_ratio <= 1.0):
        raise ValueError("miss_penalty_ratio must be in [0, 1]")
    if not (0.0 <= partial_boost_ratio <= 1.0):
        raise ValueError("partial_boost_ratio must be in [0, 1]")
    if default_keyword_match_mode not in ('any', 'all'):
        raise ValueError(
            f"default_keyword_match_mode must be 'any' or 'all', got {default_keyword_match_mode!r}"
        )
    m = _soft_map(soft_entities, rank_slot_names)
    w = float(weight)
    miss = w * float(miss_penalty_ratio)
    partial = w * float(partial_boost_ratio)
    delta = 0.0
    kw_entries = _normalize_keyword_entries(keyword_terms)
    if kw_entries:
        # Boost-only for L0 keywords (no miss penalty). Magnitude scales with
        # hit-probability mass and F1(recall, precision); term order in the SLD
        # is ignored. Explicit soft chips below may still penalize.
        hit_count, _best_prob, hits = _keyword_hit_quality(sld, kw_entries)
        if hit_count > 0:
            precision = _keyword_coverage_precision(sld, hits)
            recall = float(hit_count) / float(len(kw_entries))
            f1 = _keyword_f1(recall, precision)
            delta += w * sum(p for _, p in hits) * (0.5 + 0.5 * f1)
    if not m:
        return delta
    mode = str(m['keyword_match_mode']) if 'keyword_match_mode' in m else default_keyword_match_mode
    if mode not in ('any', 'all'):
        mode = default_keyword_match_mode
    if 'keyword_contains' in m:
        if keyword_value_matches(sld, m['keyword_contains'], 'contains', mode):
            delta += w
        else:
            delta -= miss
    if 'keyword_starts_with' in m:
        if keyword_value_matches(sld, m['keyword_starts_with'], 'starts_with', mode):
            delta += w
        else:
            delta -= miss
    if 'keyword_ends_with' in m:
        if keyword_value_matches(sld, m['keyword_ends_with'], 'ends_with', mode):
            delta += w
        else:
            delta -= miss
    if 'keyword_phrase' in m:
        phrase = str(m['keyword_phrase']).lower().replace(' ', '').replace('-', '')
        if phrase and phrase in sld.replace('-', ''):
            delta += w
        elif phrase:
            delta -= miss
    if 'keyword_contains_exclude' in m:
        raw_exc = m['keyword_contains_exclude']
        exclude_terms = [raw_exc] if isinstance(raw_exc, str) else list(raw_exc)
        hit = False
        for term in exclude_terms:
            t = str(term).lower()
            if not t:
                continue
            if len(t) <= 5:
                for seg in sld.split('-'):
                    if seg.startswith(t) or seg.endswith(t):
                        hit = True
                        break
            elif t in sld:
                hit = True
            if hit:
                break
        if hit:
            delta -= w
    if 'topic_include' in m:
        terms = _theme_terms(m['topic_include'])
        if terms:
            if _sld_theme_hit(sld, terms):
                delta += w
            else:
                delta -= miss
    if 'topic_exclude' in m:
        terms = _theme_terms(m['topic_exclude'])
        if terms and _sld_theme_hit(sld, terms):
            delta -= w
    if 'similar_to' in m:
        if _similar_to_hit(sld, m['similar_to']):
            delta += w
        else:
            delta -= partial
    wc_min = m.get('word_count_min')
    wc_max = m.get('word_count_max')
    if wc_min is not None or wc_max is not None:
        wc = _word_count(sld)
        ok = True
        try:
            if wc_min is not None and wc < int(wc_min):
                ok = False
            if wc_max is not None and wc > int(wc_max):
                ok = False
        except (TypeError, ValueError):
            ok = True
        delta += partial if ok else (-miss)
    return delta


def _apply_soft_keyword_rank_boost(
    results: RankedResults,
    soft_entities: Sequence[Entity],
    *,
    rank_slot_names: FrozenSet[str],
    weight: float,
    miss_penalty_ratio: float,
    partial_boost_ratio: float,
    default_keyword_match_mode: str,
    keyword_terms: Sequence[Any] = (),
) -> RankedResults:
    if not results.items or not (soft_entities or keyword_terms):
        return results
    soft_kw = [e for e in soft_entities if e.name in rank_slot_names]
    kw_entries = _normalize_keyword_entries(keyword_terms)
    if not soft_kw and not kw_entries:
        return results
    # Per-item metrics for multi-keyword reorder (F1 primary, hit_count tiebreak).
    scored: List[Tuple[int, float, float, int, RankedItem, str]] = []
    for i, item in enumerate(results.items):
        sld = derive_sld_from_payload(item.payload or {}).lower()
        boost = _soft_keyword_match_score(
            sld,
            soft_kw,
            rank_slot_names=rank_slot_names,
            weight=weight,
            miss_penalty_ratio=miss_penalty_ratio,
            partial_boost_ratio=partial_boost_ratio,
            default_keyword_match_mode=default_keyword_match_mode,
            keyword_terms=kw_entries,
        )
        hit_count, _best_prob, hits = _keyword_hit_quality(sld, kw_entries)
        precision = _keyword_coverage_precision(sld, hits) if hit_count > 0 else 0.0
        primary = ''
        if hit_count == 1:
            primary = _primary_keyword_term(hits) or ''
        # Clamp: RankedItem.fused_score must be >= 0; miss penalties can
        # push a low base score below zero when soft chips survive to rank.
        new_score = max(0.0, float(item.fused_score) + boost)
        if new_score != float(item.fused_score):
            item = dataclasses.replace(item, fused_score=new_score)
        scored.append((hit_count, precision, new_score, -i, item, primary))
    new_items = _reorder_by_keyword_metrics(scored, kw_entries)
    # Stamp strictly decreasing fused_score by metric order so later
    # fused_score sorts keep the F1 / hit_count ranking.
    if len(kw_entries) >= 2 and new_items:
        n = len(new_items)
        new_items = [
            dataclasses.replace(item, fused_score=float(n - rank_i))
            for rank_i, item in enumerate(new_items)
        ]
    logger.info(
        f"soft_keyword_rank_boost_applied request_id={results.request_id} "
        f"items={len(new_items)} soft_slots={[e.name for e in soft_kw]} "
        f"kw_terms_n={len(kw_entries)} metric_rank={len(kw_entries) >= 2} "
        f"weight={weight}"
    )
    return dataclasses.replace(results, items=new_items)


class SoftKeywordApplier:
    """Independent soft-signal rank-boost unit for the full-search pipeline.

    Soft chips (``qi.entity_slots.soft_slot_names``) never become FIND/hard
    filters. Rank-boost uses ``soft_rank_slot_names`` only
    (``soft_apply_mode: rank``). Soft entities come from sequential L0 LLM
    extract, with regex only when LLM is unavailable (engine gate — no merge).

    Orchestrator calls :meth:`prepare_intent` pre-retrieve and
    :meth:`apply_rank_boost` post-truncate when mode is ``rank``.
    """

    def __init__(
        self,
        entity_slots: QIEntitySlotsConfig,
        default_keyword_match_mode: str,
    ) -> None:
        if entity_slots is None:
            raise ValueError("SoftKeywordApplier requires QIEntitySlotsConfig")
        if default_keyword_match_mode not in ('any', 'all'):
            raise ValueError(
                f"default_keyword_match_mode must be 'any' or 'all', "
                f"got {default_keyword_match_mode!r}"
            )
        self._all_soft_slots = entity_slots.soft_slot_set
        self._rank_slots = entity_slots.soft_rank_slot_set
        self._weight = float(entity_slots.soft_rank_boost_weight)
        self._miss_ratio = float(entity_slots.soft_rank_miss_penalty_ratio)
        self._partial_ratio = float(entity_slots.soft_rank_partial_boost_ratio)
        self._kw_mode = str(default_keyword_match_mode)
        self._mode = _normalize_soft_apply_mode(entity_slots.soft_apply_mode)

    @property
    def mode(self) -> str:
        """Configured soft_apply_mode from YAML (``rank`` or ``off``)."""
        return self._mode

    @property
    def rank_slot_names(self) -> frozenset:
        return self._rank_slots

    def collect(self, intent: QueryIntent) -> List[Entity]:
        """Collect soft rank-slot entities used for fused_score boost."""
        return _collect_soft_keyword_entities(intent, self._rank_slots)

    def prepare_intent(self, intent: QueryIntent) -> Tuple[QueryIntent, List[Entity]]:
        """Migrate/strip soft slots off hard entities; return rank-boost soft chips.

        :return: (intent, soft_rank_entities)
        """
        # Migrate first so collect sees soft that L0 parked on hard entities.
        intent = _strip_soft_keyword_slots_from_intent(intent, self._all_soft_slots)
        soft_kw = self.collect(intent)
        return intent, soft_kw

    def match_score(
        self,
        sld: str,
        soft_entities: Sequence[Entity],
        keyword_terms: Sequence[Any] = (),
    ) -> float:
        """Signed boost for one SLD."""
        return _soft_keyword_match_score(
            sld,
            soft_entities,
            rank_slot_names=self._rank_slots,
            weight=self._weight,
            miss_penalty_ratio=self._miss_ratio,
            partial_boost_ratio=self._partial_ratio,
            default_keyword_match_mode=self._kw_mode,
            keyword_terms=keyword_terms,
        )

    def apply_rank_boost(
        self,
        results: RankedResults,
        soft_entities: Sequence[Entity],
        keyword_terms: Sequence[Any] = (),
    ) -> RankedResults:
        """Reorder by soft theme/keyword score when mode=rank. No-op when off/empty.

        Multi-keyword order: higher F1(recall, precision), then higher hit_count
        when F1 ties, then precision / recall / fused score. Term order inside
        the SLD is ignored.
        """
        if self._mode != 'rank':
            return results
        return _apply_soft_keyword_rank_boost(
            results,
            soft_entities,
            rank_slot_names=self._rank_slots,
            weight=self._weight,
            miss_penalty_ratio=self._miss_ratio,
            partial_boost_ratio=self._partial_ratio,
            default_keyword_match_mode=self._kw_mode,
            keyword_terms=keyword_terms,
        )


__all__ = ['SoftKeywordApplier']
