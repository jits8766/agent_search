"""Search orchestrator — ties QI + caches + retrievers + fusion + external eRanker
into a single pipeline.
Pipeline:
  1) Normalize the query (via QIEngine.classify which performs normalization).
  2) Check the exact cache (sha256(normalized_query)) -> return on hit.
  3) Check the semantic cache (cosine over embedding) -> return on hit.
  4) Classify the query through the QI cascade (L0 -> L1 -> L2).
  5) Hybrid-first ranked_results: rewrite retrieve intent to hybrid for every QI
     type; fan out vector + structured (+ sql when price filter present).
     Analytics SQL / explore rails / guidance snapshot complement (app + merge),
     they do not replace hybrid ranks. ClickHouse down => temporal strip + hybrid only.
  6) Fuse via RRF.
  7) Call external eRanker (Layer 4) with latency budget; ``noop`` pass-through or ``http`` JSON client; shadow + A/B + health skip wired in ``_apply_eranker``.
  8) Post-eRanker diversification (MMR) then truncate.
  9) Populate exact + semantic caches with the pre-eRanker fused results so
     the cache stays user-agnostic and shareable across callers.
  10) Record a history entry for authenticated callers (skipped when opted out).
"""
import asyncio
import contextvars
import dataclasses
import math
import re
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

from semantic_search.analytics.pipeline_router import AnalyticsRouter
from semantic_search.cache.exact_cache import ExactCache
from semantic_search.cache.keys import intent_plan_key
from semantic_search.cache.intent_plan_cache import IntentPlanCache
from semantic_search.cache.redis_payload_tier import RedisPayloadTier
from semantic_search.cache.structured_cache import StructuredCache
from semantic_search.config.models import AgentSearchConfig, AnalyticsRateLimitConfig
from semantic_search.middleware import SlidingWindowRateLimiter
from semantic_search.core.exceptions import CostBudgetExceeded, DiversityError, HistoryError, LLMError, QdrantQueryError, QdrantUnavailableError, QueryCostBudgetExceeded, RetrievalError, ValidationError
from semantic_search.core.llm_client import reset_request_cost_observer, set_request_cost_observer
from semantic_search.core.logging_utils import get_logger
from semantic_search.cost.fleet_budget import FleetCostBudget
from semantic_search.cost.query_budget import NoOpQueryCostBudget, QueryCostBudget
from semantic_search.cost.request_cost_gate import RequestCostGate
from semantic_search.explore.composer import ExploreComposer, _rrf_fuse
from semantic_search.explore.zero_result_guard import ZeroResultGuard
from semantic_search.guidance.guidance_service import GuidanceService
from semantic_search.signal_store import SignalStore
from semantic_search.safety.layer_zero_sanitizer import LayerZeroSanitizer
from semantic_search.qi.spell_corrector import SymSpellCorrector
from semantic_search.qi.query_transformer import QueryTransformer, QueryTransformResult
from semantic_search.history.store import UserSearchHistoryStore
from semantic_search.measurement.store import MeasurementStore
from semantic_search.nl_to_sql.contracts import AnalyticsResult
from semantic_search.qi.engine import QIEngine, normalize_query
from semantic_search.qi.residual_extractor import (
    build_semantic_encode_text,
    extract_residual,
    listing_concept_encode_text,
)
from semantic_search.qi.l0_llm_filter_extractor import L0CombinedExtractOutcome, L0LLMFilterExtractor
from semantic_search.qi.keyword_threshold import filter_keywords, keyword_terms, min_probability_fraction
from semantic_search.qi.multi_intent_splitter import rank_sub_intents
from semantic_search.resilience.degradation import DegradationPlanner
from semantic_search.resilience.health import BackendHealthRegistry
from semantic_search.retrieval.base import Retriever
from semantic_search.retrieval.diversifier_base import Diversifier
from semantic_search.retrieval.eranker_client import ERankerClient
from semantic_search.retrieval.fusion import RRFFuser
from semantic_search.retrieval.brandability_scorer import BrandabilityScorer
from semantic_search.retrieval.fuzzy_lexical_reranker import FuzzyLexicalReranker
from semantic_search.retrieval.sql_retriever import SqlRetriever
from semantic_search.retrieval.soft_keyword_apply import SoftKeywordApplier
from semantic_search.retrieval.structured_retriever import (
    InMemoryStructuredIndex,
    detect_filter_conflicts,
    derive_sld_from_payload,
    extract_filters_from_intent,
    extract_hard_filters_from_intent,
    normalize_tld,
)
from semantic_search.safety.egress_contracts import EgressGuardOutcome
from semantic_search.safety.egress_guard import EgressGuard, NoOpEgressGuard
from semantic_search.contracts import QUERY_TYPES, CachedSearchPayload, CandidateSet, Entity, ERankerOutcome, FeedbackSignal, IntentChipGroup, IntentSlice, MultiIntentChipStrip, OverflowChip
from semantic_search.contracts import QueryIntent, RankedItem, RankedResults, ReasoningTrace, SUB_INTENT_MATCH_KINDS, SearchObservation, SpellCorrection, UserContext, ZeroResultGuardOutcome
from semantic_search.qi.chip_format import build_card_title, chip_labels_for_slice
from semantic_search.qi.slot_to_api_param import is_temporal_entity_slot

logger = get_logger(__name__)

# Per-task cost gate. Response stamping reads ContextVar or costs already on
# QueryIntent / AnalyticsResult — not the shared ``_last_cost_budget`` attribute.
_REQUEST_COST_GATE: contextvars.ContextVar[Optional[RequestCostGate]] = contextvars.ContextVar(
    'semantic_search_request_cost_gate', default=None,
)


def _stamp_intent_decision_cost(intent: Any, total: float) -> Any:
    """Set ``decision_cost_usd`` on intent; no-op-safe for mocks / non-dataclasses."""
    if intent is None:
        return intent
    try:
        current = float(getattr(intent, 'decision_cost_usd', 0.0) or 0.0)
    except (TypeError, ValueError):
        current = 0.0
    if abs(total - current) < 1e-12:
        return intent
    if dataclasses.is_dataclass(intent) and not isinstance(intent, type):
        return dataclasses.replace(intent, decision_cost_usd=total)
    try:
        intent.decision_cost_usd = total
    except (AttributeError, TypeError):
        pass
    return intent

# Retriever fan-out is always driven by a hybrid retrieve_intent for ranked_results
# (see hybrid_retrieve_intent). These sets still gate backends when query_type is
# already hybrid / explore / guidance without rewrite.
_QUERY_TYPES_USING_VECTOR = frozenset({'hybrid', 'explore', 'guidance'})
_QUERY_TYPES_USING_STRUCTURED = frozenset({'hybrid', 'explore', 'guidance'})
_QUERY_TYPES_USING_SQL = frozenset({'hybrid'})
# Zero-result guard applies to every QI type that fills ranked_results via hybrid.
_QUERY_TYPES_USING_ZERO_RESULT_GUARD = frozenset({'hybrid', 'explore', 'guidance', 'analytics'})
# Entity slots that select an explore rail rather than filter items within it.
# Excluded from post-fusion hard filtering in _explore_primary_retrieve so that
# e.g. "ending soon this week" activates the ending_soon rail via horizon_override
# instead of dropping rail items whose payload lacks ends_at.
# Explore-rail *selectors* (activate a rail) vs post-fusion item filters.
# Lifecycle flags + temporal slots that pick trending/ending-soon/latest rails.
_EXPLORE_RAIL_SELECTOR_SLOTS = frozenset({
    'lifecycle_state',
    'lifecycle_disjunction',
}) | frozenset(
    name for name in ('time_remaining_max', 'startTimeAfter')
    if is_temporal_entity_slot(name)
)
# Any non-hybrid QI intent uses a CH-backed path (analytics / explore rails /
# guidance snapshot). Derived from contracts.QUERY_TYPES — not a hand list —
# so new archetypes auto-degrade when ClickHouse is unavailable.
_CH_DEGRADE_QUERY_TYPES = frozenset(qt for qt in QUERY_TYPES if qt != 'hybrid')

# Concrete exception set for defensive retrieval/async guards. Excludes
# BaseException/CancelledError/KeyboardInterrupt so cancellation and
# interrupts propagate; catches realistic backend/parse failures.
_GUARD_EXC = (RuntimeError, ValueError, TypeError, KeyError, IndexError, AttributeError, OSError, asyncio.TimeoutError)
# Fallback / semantic-only paths: also swallow RetrievalError (Qdrant down,
# missing collection) so callers get [] instead of HTTP 500. Do NOT merge into
# _GUARD_EXC - primary retrieve fan-out must keep RetrievalError visible where
# gather already isolates failures.
_FALLBACK_RETRIEVE_EXC = _GUARD_EXC + (RetrievalError,)


def strip_time_filters_from_intent(intent: QueryIntent) -> QueryIntent:
    """Drop temporal entity slots from all slices (CH->hybrid degrade).

    Temporal detection is catalog-driven via ``is_temporal_entity_slot``
    (entity_slot_to_api_param.json transforms/API params) — not a fixed slot list.
    Preserves non-time filters (tld, price, …) and original query_type.
    """
    if intent is None or not intent.slices:
        return intent
    new_slices: List[IntentSlice] = []
    dropped_any = False
    for sl in intent.slices:
        kept = [e for e in (sl.entities or []) if not is_temporal_entity_slot(e.name)]
        if len(kept) != len(sl.entities or []):
            dropped_any = True
            new_slices.append(dataclasses.replace(sl, entities=kept))
        else:
            new_slices.append(sl)
    if not dropped_any:
        return intent
    return dataclasses.replace(intent, slices=new_slices)


def _concept_entities_from_slices(slices: Optional[List['IntentSlice']]) -> List[Entity]:
    """Hard + soft entities for listing-concept encode (theme survives prepare_intent)."""
    out: List[Entity] = []
    for sl in slices or []:
        out.extend(sl.entities or [])
        out.extend(getattr(sl, 'soft_entities', None) or [])
    return out


def _has_listing_concept(slices: Optional[List['IntentSlice']]) -> bool:
    """True when hard/soft concept entities produce non-empty listing encode text."""
    return bool(listing_concept_encode_text(_concept_entities_from_slices(slices)))


def _should_restore_rewrite_encode(query_type: str, *, has_listing_concept: bool) -> bool:
    """Whether concept-empty retrieve may restore ANN text from query rewrite.

    CH-backed archetypes (analytics / explore / guidance) use ranked_results as a
    filter-scoped complement. Restoring rewrite residual turns meta-intent words
    into vector queries. Hybrid listing search may still restore rewrite when
    concept-empty.
    """
    if has_listing_concept:
        return False
    return query_type not in _CH_DEGRADE_QUERY_TYPES


def hybrid_retrieve_intent(intent: QueryIntent, *, strip_temporal: bool) -> QueryIntent:
    """Rewrite a QI intent for hybrid ranked_results retrieval.

    Sets ``query_type='hybrid'`` on the intent and slices. Optionally strips
    catalog-temporal entity slots. Sets encode fields from keyword and
    soft-concept entities (hard + soft_entities) only; when none are present,
    sets an empty encode text and ``residual_kind='empty'``. The caller retains
    the original intent for the response envelope.
    """
    base = strip_time_filters_from_intent(intent) if strip_temporal else intent
    new_slices = [
        dataclasses.replace(sl, query_type='hybrid')
        for sl in (base.slices or [])
    ]
    concept = listing_concept_encode_text(_concept_entities_from_slices(new_slices))
    if concept:
        return dataclasses.replace(
            base,
            query_type='hybrid',
            slices=new_slices,
            semantic_query=concept,
            semantic_encode_text=concept,
            residual_kind='semantic',
        )
    return dataclasses.replace(
        base,
        query_type='hybrid',
        slices=new_slices,
        semantic_query=None,
        semantic_encode_text='',
        residual_kind='empty',
    )


def hybrid_degrade_intent(intent: QueryIntent) -> QueryIntent:
    """Rewrite for hybrid retrieve with temporal slots stripped."""
    return hybrid_retrieve_intent(intent, strip_temporal=True)


def _hard_entity_fingerprint_from_slice(slc: Optional['IntentSlice']) -> frozenset:
    """Return frozenset of (name, normalized_value) for hard-chip entities in one slice."""
    if slc is None:
        return frozenset()
    result = set()
    for e in slc.entities:
        if getattr(e, 'chip_kind', 'hard') != 'hard':
            continue
        val = e.value
        if isinstance(val, list):
            val = tuple(sorted(str(v) for v in val))
        result.add((e.name, val))
    return frozenset(result)


def _hard_entity_fingerprint(intent: QueryIntent) -> frozenset:
    """Return frozenset of (name, normalized_value) for all hard-chip entities across slices."""
    result: set = set()
    for s in intent.slices:
        result |= _hard_entity_fingerprint_from_slice(s)
    return frozenset(result)


def _intent_has_hard_chip(intent: QueryIntent) -> bool:
    """Return True iff any slice's entity is a hard chip (deterministic filter).

    Used by the residual-aware dispatch policy: dropping the vector backend is
    only safe when at least one hard chip will satisfy the recall floor on its
    own. Soft chips carry aspirational / rerank-only signal and cannot be
    relied on as a result-set guarantee.
    """
    for s in intent.slices:
        for e in s.entities:
            if e.chip_kind == 'hard':
                return True
    return False


def _should_drop_sql_for_semantic_residual(residual_kind: Optional[str], candidate_sets: List[CandidateSet]) -> bool:
    """True when the SQL price-band leg should be discarded post-gather.

    The pre-gather ``residual_dispatch`` policy only suppresses SQL for
    filter-dominant residual kinds (``empty`` / ``navigational``). A query that
    carries real thematic content (``residual_kind == 'semantic'``) still
    unconditionally triggers SQL whenever a price filter is present (taxonomy
    collapse into ``hybrid``), even though ``SqlRetriever`` ranks purely by a
    static item ``score`` with zero query-relevance. When vector already
    returned candidates for a semantic residual, SQL contributes nothing but
    RRF-equal-weighted noise that backfills the post-hard-gate survivor pool
    with off-theme rows once the price filter thins the vector pool. Drop it.
    """
    if residual_kind != 'semantic':
        return False
    return any(cs.source == 'vector' and cs.candidates for cs in candidate_sets)


# Numeric range constraint slots enforced post-fusion by the hard-chip gate.
# When any of these are present as hard chips, the ZeroResultGuard should only
# fire on truly empty results (len == 0), not on the soft min_results threshold.
# Rationale: the gate may reduce a large candidate set to fewer than the
# threshold, but the remaining results ARE valid — the guard must not replace
# them with constraint-violating results fetched from a relaxed intent.
_HARD_NUMERIC_GATE_SLOTS: frozenset = frozenset({
    'price_min', 'price_max', 'name_length_min', 'name_length_max',
})


def _intent_has_hard_numeric_constraint(intent: QueryIntent) -> bool:
    """Return True iff any slice carries a hard numeric range constraint.

    When True, the ZeroResultGuard uses a stricter threshold (fire only on
    0 results) to avoid replacing valid but scarce price/name-length results
    with constraint-violating alternatives from a relaxed intent.
    """
    for s in intent.slices:
        for e in s.entities:
            if e.chip_kind == 'hard' and e.name in _HARD_NUMERIC_GATE_SLOTS:
                return True
    return False


def _apply_hard_chip_gate(
    items: List[RankedItem],
    intent: QueryIntent,
    drop_on_missing_field: bool,
    *,
    keyword_match_mode: Optional[str],
) -> List[RankedItem]:
    """Remove fused items that violate hard-chip categorical constraints.

    The SQL/structured backend applies hard-chip filters at query time, but the
    vector/semantic backend performs ANN search without payload filtering — so
    type-38 domains can appear in results even when the user asked for type-16
    only.  This gate enforces tld and auction_type hard chips post-fusion so
    no backend can leak results that contradict an explicit categorical
    constraint.

    Policy:
    - Only list-valued categorical entities with chip_kind='hard' are checked
      (tld, auction_type).  Numeric range filters (price_min/max) are excluded
      because the payload price field may be absent on vector-only results.
    - If the payload carries the field AND the value is not in the allowed set,
      the item is removed.
    - If the payload does NOT carry the constrained field, the item is removed
      when ``drop_on_missing_field`` is True (precision-first: an unverifiable
      item cannot satisfy an explicit hard constraint) and kept when False
      (recall-first). Driven by
      ``retrieval.enforce_categorical_gate_on_missing_payload``.

    :param items: List[RankedItem] - Fused candidates to gate
    :param intent: QueryIntent - Carries the hard-chip categorical entities
    :param drop_on_missing_field: bool - Drop items whose payload lacks a
        hard-constrained field (True = precision-first)
    :param keyword_match_mode: Optional[str] - From
        ``retrieval.structured.keyword_match_mode``; required when hard chips
        include mode-governed keyword slots and intent did not set the mode
    :return: List[RankedItem] - Items satisfying every hard categorical chip
    """
    # Precision-first (default): enforce EVERY payload-verifiable hard chip, not
    # just tld/auction_type/price/name_length. Delegates to the structured
    # index's _matches predicate so the gate's coverage always tracks the
    # structured backend's filter semantics (keyword_*, bids_*, domain_age_*,
    # majestic_*, semrush_*, exclusion lists, character flags, ...). Without this
    # the vector leg leaked any hard filter outside the original 6-slot set.
    # Rail-selector slots (which pick an explore rail, not filter items) and
    # vector-only slots (ANN-driving, not payload fields) are excluded.
    if drop_on_missing_field:
        _hard_filters = extract_hard_filters_from_intent(intent)
        if not _hard_filters:
            return items
        if any(
            k in _hard_filters
            for k in ('keyword_contains', 'keyword_starts_with', 'keyword_ends_with')
        ) and 'keyword_match_mode' not in _hard_filters:
            if keyword_match_mode not in ('any', 'all'):
                raise RetrievalError(
                    "keyword_match_mode must be 'any' or 'all' when keyword slots "
                    f"are active; got {keyword_match_mode!r}"
                )
            _hard_filters = dict(_hard_filters)
            _hard_filters['keyword_match_mode'] = keyword_match_mode
        return [it for it in items if InMemoryStructuredIndex._matches(it.payload, _hard_filters)]  # noqa: SLF001

    # Recall-first (enforce_categorical_gate_on_missing_payload=False): keep the
    # legacy present-and-mismatch check on categorical + numeric slots only, so
    # unverifiable items are retained by design.
    hard_tlds: Optional[set] = None
    hard_auction_types: Optional[set] = None
    hard_price_min: Optional[float] = None
    hard_price_max: Optional[float] = None
    hard_name_len_min: Optional[int] = None
    hard_name_len_max: Optional[int] = None
    for s in intent.slices:
        for e in s.entities:
            if e.chip_kind != 'hard':
                continue
            if e.name == 'tld' and isinstance(e.value, list):
                vals = {normalize_tld(v) for v in e.value}
                hard_tlds = (hard_tlds | vals) if hard_tlds is not None else vals
            elif e.name == 'auction_type' and isinstance(e.value, list):
                vals = {str(v).lower() for v in e.value}
                hard_auction_types = (hard_auction_types | vals) if hard_auction_types is not None else vals
            elif e.name == 'price_min':
                try:
                    v = float(e.value)
                    hard_price_min = max(hard_price_min, v) if hard_price_min is not None else v
                except (TypeError, ValueError):
                    pass
            elif e.name == 'price_max':
                try:
                    v = float(e.value)
                    hard_price_max = min(hard_price_max, v) if hard_price_max is not None else v
                except (TypeError, ValueError):
                    pass
            elif e.name == 'name_length_min':
                try:
                    v = int(e.value)
                    hard_name_len_min = max(hard_name_len_min, v) if hard_name_len_min is not None else v
                except (TypeError, ValueError):
                    pass
            elif e.name == 'name_length_max':
                try:
                    v = int(e.value)
                    hard_name_len_max = min(hard_name_len_max, v) if hard_name_len_max is not None else v
                except (TypeError, ValueError):
                    pass
    if hard_tlds is None and hard_auction_types is None and hard_price_min is None and hard_price_max is None and hard_name_len_min is None and hard_name_len_max is None:
        return items
    out: List[RankedItem] = []
    for item in items:
        p = item.payload
        if hard_tlds is not None:
            if 'tld' in p:
                if normalize_tld(p['tld']) not in hard_tlds:
                    continue
            elif drop_on_missing_field:
                continue
        if hard_auction_types is not None:
            if 'auction_type' in p:
                if str(p['auction_type']).lower() not in hard_auction_types:
                    continue
            elif drop_on_missing_field:
                continue
        if hard_price_min is not None or hard_price_max is not None:
            price_val = p.get('price')
            if price_val is not None:
                try:
                    price_f = float(price_val)
                    if hard_price_min is not None and price_f < hard_price_min:
                        continue
                    if hard_price_max is not None and price_f > hard_price_max:
                        continue
                except (TypeError, ValueError):
                    pass
            elif drop_on_missing_field:
                continue
        if hard_name_len_min is not None or hard_name_len_max is not None:
            nl_val = p.get('name_length')
            if nl_val is None:
                _sld = derive_sld_from_payload(p)
                nl_val = len(_sld) if _sld else None
            if nl_val is None and drop_on_missing_field:
                continue
            if nl_val is not None:
                try:
                    nl_i = int(nl_val)
                    if hard_name_len_min is not None and nl_i < hard_name_len_min:
                        continue
                    if hard_name_len_max is not None and nl_i > hard_name_len_max:
                        continue
                except (TypeError, ValueError):
                    pass
        out.append(item)
    return out


def _dedup_by_domain_name(ranked: 'RankedResults') -> 'RankedResults':
    """Remove duplicate items that share the same normalised domain_name.

    Items arrive in score-descending order from RRF fusion; first-seen wins.
    Handles case-normalisation mismatches (e.g. "Stripe.com" vs "stripe.com")
    and multi-slice collision where the same domain surfaces in two sub-intents.
    """
    seen: 'Dict[str, None]' = {}
    deduped = []
    for item in ranked.items:
        key = (item.item_id or '').lower()
        if key and key in seen:
            continue
        if key:
            seen[key] = None
        deduped.append(item)
    if len(deduped) == len(ranked.items):
        return ranked
    return dataclasses.replace(ranked, items=deduped)


class RankedItemBuilder:
    """Mutable accumulator used during multi-intent merge.

    A single item may surface in multiple sub-intent fused result sets. We
    accumulate the maximum per-slice fused_score, the union of contributing
    sources, and the set of slice_ids the item matched. The cross-intent bonus
    is applied once at `build()` time so we don't double-apply when an item
    appears in 3+ slices (the bonus is a property of "matched >1", not a
    per-additional-match multiplier).
    """

    def __init__(self, item_id: str, fused_score: float, contributing_sources: List[str], payload: Dict, slice_ids: List[str]):
        self.item_id = item_id
        self.fused_score = float(fused_score)
        self.sources: List[str] = list(contributing_sources)
        self.payload: Dict = dict(payload) if payload else {}
        self.slice_ids: List[str] = list(slice_ids)
        self.match_kinds: Dict[str, str] = {}

    @classmethod
    def from_item(cls, item: RankedItem) -> 'RankedItemBuilder':
        """Seed a builder from a single sub-intent's RankedItem."""
        b = cls(item_id=item.item_id, fused_score=item.fused_score, contributing_sources=item.contributing_sources, payload=item.payload, slice_ids=[])
        for k, v in (item.sub_intent_match_kinds or {}).items():
            b.match_kinds[k] = v
        return b

    def merge_from_item(self, item: RankedItem) -> None:
        """Merge another sub-intent's view of the same item.

        Take the maximum fused_score across slices (winner-takes-all per slice
        is more truthful than averaging — an item that one slice ranked highly
        deserves to keep that signal), and union contributing_sources. Match
        kinds union with hard winning over soft on conflict (an item that
        hard-matched at least one slice is hard for that slice — never
        downgraded by a later soft view).
        """
        if item.item_id != self.item_id:
            raise ValidationError(f"merge_from_item: item_id mismatch {self.item_id} vs {item.item_id}")
        if item.fused_score > self.fused_score:
            self.fused_score = float(item.fused_score)
        for src in item.contributing_sources:
            if src not in self.sources:
                self.sources.append(src)
        for k, v in (item.payload or {}).items():
            if k not in self.payload:
                self.payload[k] = v
        for k, v in (item.sub_intent_match_kinds or {}).items():
            existing = self.match_kinds.get(k)
            if existing == 'hard':
                continue
            self.match_kinds[k] = v

    def add_slice(self, slice_id: str, match_kind: str = 'hard') -> None:
        """Record that this item was returned by a given sub-intent slice.

        :param slice_id: str - The sub-intent's stable slice id
        :param match_kind: str - One of ``SUB_INTENT_MATCH_KINDS``. Defaults
            to ``'hard'`` because the per-slice retriever's grounded filters
            (TLD/price/auction-type) constitute a hard match by construction.
            Soft matches surface only when the orchestrator runs a relaxed
            second pass for empty hard slices.
        """
        if not slice_id:
            return
        if match_kind not in SUB_INTENT_MATCH_KINDS:
            raise ValidationError(f"add_slice: match_kind must be one of {sorted(SUB_INTENT_MATCH_KINDS)}")
        if slice_id not in self.slice_ids:
            self.slice_ids.append(slice_id)
        existing = self.match_kinds.get(slice_id)
        if existing == 'hard':
            # Hard never downgrades.
            return
        self.match_kinds[slice_id] = match_kind

    def build(self, cross_intent_bonus: float) -> RankedItem:
        """Produce the final RankedItem, applying the cross-intent bonus once."""
        score = self.fused_score
        if len(self.slice_ids) >= 2 and cross_intent_bonus > 1.0:
            score = score * cross_intent_bonus
        # Filter match_kinds to only slices we actually recorded; defends
        # against a producer that seeded `from_item` with stale kinds whose
        # slice_id never re-appeared in any merge.
        slice_id_set = set(self.slice_ids)
        kinds = {k: v for k, v in self.match_kinds.items() if k in slice_id_set}
        return RankedItem(item_id=self.item_id, fused_score=score, contributing_sources=self.sources, payload=self.payload, sub_intent_ids=self.slice_ids, sub_intent_match_kinds=kinds)


def _rrf_contribution_match_kind_for_slice(item: RankedItem, slice_id: str) -> str:
    """Resolve per-slice match kind for one RRF input row (defaults to hard when unset)."""
    kinds = item.sub_intent_match_kinds or {}
    if slice_id in kinds:
        return kinds[slice_id]
    return 'hard'


def _rrf_accumulate_slice_match_kind(bucket: Dict[str, str], slice_id: str, new_kind: str) -> None:
    """Merge one slice's match kind into the RRF accumulator; hard wins over soft (same rule as RankedItemBuilder)."""
    if new_kind not in SUB_INTENT_MATCH_KINDS:
        raise ValidationError(f"rrf_merge: match_kind must be one of {sorted(SUB_INTENT_MATCH_KINDS)}")
    existing = bucket.get(slice_id)
    if existing == 'hard':
        return
    if new_kind == 'hard':
        bucket[slice_id] = 'hard'
        return
    if existing is None:
        bucket[slice_id] = new_kind
        return
    bucket[slice_id] = new_kind


class SearchOrchestrator:
    """End-to-end search pipeline.
    :param config: AgentSearchConfig - Top-level service config
    :param qi_engine: QIEngine - QI cascade
    :param vector_retriever: Retriever - Vector backend (in-memory or
        Qdrant-backed; reports `source='vector'`). When the unified Qdrant
        index is wired in hybrid mode, this slot is occupied by the
        single-round-trip `QdrantHybridRetriever`.
    :param structured_retriever: Retriever - Structured backend (in-memory
        or Qdrant payload-filter scan; reports `source='structured'`). In
        hybrid mode this slot is a no-op so the orchestrator's fan-out does
        not double-issue Qdrant calls.
    :param sql_retriever: SqlRetriever - SQL price-band fallback
    :param fuser: RRFFuser - RRF fusion engine
    :param eranker_client: ERankerClient - External Layer-4 ranking port
    :param exact_cache: ExactCache - Tier 1 cache
    :param structured_cache: StructuredCache - Structured retriever intermediate cache
    :param intent_plan_cache: IntentPlanCache - Tier-3 structural-intent cache (post-QI)
    :param redis_payload_tier: Optional[RedisPayloadTier] - Shared Redis/Dragonfly payload tier
    :param inventory_snapshot_version_fn: Optional[Callable[[], int]] - Live snapshot version for Redis stale reads
    """

    def __init__(
        self,
        config: AgentSearchConfig,
        qi_engine: QIEngine,
        vector_retriever: Retriever,
        structured_retriever: Retriever,
        sql_retriever: SqlRetriever,
        fuser: RRFFuser,
        eranker_client: ERankerClient,
        diversifier: Diversifier,
        exact_cache: ExactCache,
        structured_cache: StructuredCache,
        intent_plan_cache: IntentPlanCache,
        history_store: UserSearchHistoryStore,
        health_registry: BackendHealthRegistry,
        degradation_planner: DegradationPlanner,
        measurement_store: MeasurementStore,
        analytics_router: Optional[AnalyticsRouter],
        zero_result_guard: Optional[ZeroResultGuard],
        sanitizer: Optional[LayerZeroSanitizer] = None,
        egress_guard: Optional[Union[EgressGuard, NoOpEgressGuard]] = None,
        cost_budget_factory: Optional[Callable[[Optional[str]], Union[QueryCostBudget, NoOpQueryCostBudget]]] = None,
        fleet_cost_budget: Optional[FleetCostBudget] = None,
        spell_corrector: Optional[SymSpellCorrector] = None,
        query_transformer: Optional[QueryTransformer] = None,
        redis_payload_tier: Optional[RedisPayloadTier] = None,
        inventory_snapshot_version_fn: Optional[Callable[[], int]] = None,
        guidance_service: Optional[GuidanceService] = None,
        signal_store: Optional[SignalStore] = None,
        analytics_rate_limiter: Optional['SlidingWindowRateLimiter'] = None,
        analytics_rate_limit_config: Optional['AnalyticsRateLimitConfig'] = None,
        explore_composer: Optional[ExploreComposer] = None,
        entity_extractor: Optional[L0LLMFilterExtractor] = None,
    ):
        """Initialize SearchOrchestrator with all subsystem dependencies."""
        if health_registry is None:
            raise ValidationError("SearchOrchestrator requires a BackendHealthRegistry instance")
        if degradation_planner is None:
            raise ValidationError("SearchOrchestrator requires a DegradationPlanner instance")
        if measurement_store is None:
            raise ValidationError("SearchOrchestrator requires a MeasurementStore instance")
        self._config = config
        self._qi = qi_engine
        self._entity_extractor = entity_extractor
        self._vector = vector_retriever
        self._structured = structured_retriever
        self._sql = sql_retriever
        self._fuser = fuser
        if eranker_client is None:
            raise ValidationError("SearchOrchestrator requires an ERankerClient instance")
        self._eranker_client = eranker_client
        self._eranker_budget_s = float(config.retrieval.eranker.latency_budget_ms) / 1000.0
        if diversifier is None:
            raise ValidationError("SearchOrchestrator requires a Diversifier instance; wire NoOpDiversifier when retrieval.diversity.enabled=false")
        self._diversifier = diversifier
        self._diversifier_budget_s = float(config.retrieval.diversity.latency_budget_ms) / 1000.0
        self._diversifier_top_n = int(config.retrieval.diversity.top_n)
        self._diversifier_output_n = int(config.retrieval.diversity.output_n)
        self._diversifier_enabled = bool(config.retrieval.diversity.enabled)
        # Fuzzy lexical reranker — pure-config stdlib component (no external deps to
        # inject), built here from config and applied over the over-fetched fused pool
        # so SLD typo / near-matches survive the final top_k truncation.
        self._fuzzy_reranker: Optional[FuzzyLexicalReranker] = None
        self._fuzzy_rerank_max_candidates = 0
        _frc = config.retrieval.fuzzy_rerank
        if _frc is not None and _frc.enabled:
            self._fuzzy_reranker = FuzzyLexicalReranker(_frc)
            self._fuzzy_rerank_max_candidates = int(_frc.max_candidates)
            logger.info(f"fuzzy_lexical_reranker_enabled max_edit_distance={_frc.max_edit_distance} boost_weight={_frc.boost_weight} max_candidates={_frc.max_candidates}")
        # Soft-keyword apply — independent post-QI unit (rank boost / filter attach).
        # Built from qi.entity_slots + retrieval.structured.keyword_match_mode.
        self._soft_keyword_applier: Optional[SoftKeywordApplier] = None
        _entity_slots = config.qi.entity_slots
        if _entity_slots is not None:
            self._soft_keyword_applier = SoftKeywordApplier(
                _entity_slots,
                default_keyword_match_mode=config.retrieval.structured.keyword_match_mode,
            )
            logger.info(
                f"soft_keyword_applier_enabled mode={self._soft_keyword_applier.mode} "
                f"rank_slots={len(self._soft_keyword_applier.rank_slot_names)} "
                f"boost_weight={_entity_slots.soft_rank_boost_weight}"
            )
        if intent_plan_cache is None:
            raise ValidationError("SearchOrchestrator requires an IntentPlanCache instance")
        self._exact_cache = exact_cache
        self._structured_cache = structured_cache
        self._intent_plan_cache = intent_plan_cache
        self._redis_payload_tier = redis_payload_tier
        self._inventory_snapshot_version_fn = inventory_snapshot_version_fn
        self._history = history_store
        self._health = health_registry
        self._degradation = degradation_planner
        self._measurement = measurement_store
        # Analytics path — Optional: when None, ``analytics()`` raises and
        # the ``/search`` routing hint reports ``analytics_available=False``.
        # The substrate router (NL-SQL exact cache -> ClickHouse MV ->
        # events_raw scan) lives behind this single handle.
        self._analytics_router = analytics_router
        self._signal_store = signal_store
        # Per-tenant analytics rate limiter (gap 4). When None or
        # config.enabled=False, ``analytics()`` skips the bucket check.
        self._analytics_rate_limiter = analytics_rate_limiter
        self._analytics_rate_limit_config = analytics_rate_limit_config
        # Zero-Result Guard — Optional: when None or disabled the orchestrator
        # returns the empty result without trying the relax -> semantic-only ->
        # explore ladder.
        self._zero_result_guard = zero_result_guard
        # Explore-primary composer — Optional: when wired, explore-classified
        # queries call the trending/ending_soon rails BEFORE vector retrieval.
        # Falls back to the vector path when the composer returns empty results.
        self._explore_composer = explore_composer
        self._guidance_service = guidance_service
        # User-input sanitizer at the API ingress.
        # Optional: when None the orchestrator skips the gate (test paths that
        # don't wire a sanitizer remain functional). When wired AND
        # `applies_to_llm_ingress` is True (the same gate the LLM router
        # consults), every raw query passes through length-cap / blocklist /
        # PII-pattern checks BEFORE QI classification, cache lookup, or any
        # downstream call. A reject raises ValidationError with categorised
        # reason codes only — never echoes the input — so the API maps it to
        # HTTP 422 without leaking PII / injection markers per
        # `responsible-ai.mdc`.
        self._sanitizer = sanitizer
        # Output-side egress guardrails. Optional so test paths that don't
        # wire one continue to function. When wired (production via the
        # registry), the gate runs as the FINAL step on every return path —
        # including cache-hit paths, so a poisoned payload that landed in
        # cache is still scrubbed on every read. The orchestrator wraps the
        # (sync) gate call in ``asyncio.to_thread`` for event-loop courtesy.
        # The gate is NOT latency-gated (safety, not quality) — a per-item
        # exception in the gate fails closed (drops the item) rather than
        # leaking unscrubbed content.
        self._egress_guard = egress_guard
        # Per-request LLM cost-budget factory + optional shared fleet budget.
        # Bound as ``RequestCostGate`` (query + fleet). Breach inside
        # ``call_structured`` becomes ``LLMError`` so QI degrades to regex/L1
        # instead of rejecting the user query.
        self._cost_budget_factory = cost_budget_factory
        self._fleet_cost_budget = fleet_cost_budget
        # Fallback when ContextVar unset (direct unit-test access).
        self._last_cost_budget: Optional[RequestCostGate] = None
        # Per-request_id spend across search + analytics phases. Readable from the
        # parent task after ``asyncio.wait_for`` cancels a child (ContextVar is not).
        self._accumulated_llm_cost_usd: Dict[str, float] = {}
        self._active_cost_gate_by_rid: Dict[str, RequestCostGate] = {}
        # Tier-0 spell-correct + 'did you mean'. Optional: when None
        # the orchestrator skips the corrector entirely (zero overhead).
        # When wired the corrector runs AFTER normalize and BEFORE the cache
        # lookup so a successful auto-apply correction reaches the cache key.
        # The correction is attached to ``QueryIntent.did_you_mean`` on the
        # returned envelope — that is the single source of truth callers
        # read from. The orchestrator does not hold a singleton accessor
        # because two concurrent search() calls would clobber it.
        self._spell_corrector = spell_corrector
        self._spell_auto_apply = bool(spell_corrector._config.auto_apply) if spell_corrector is not None else False  # noqa: SLF001 — read-only access to validated config
        # DistilBERT query transformer — rewrites verbose queries before QI.
        self._query_transformer = query_transformer
        # Most-recent egress outcome — exposed as a public read-only accessor
        # so tests + future SRE dashboards can introspect the gate's per-call
        # decisions without re-running the pipeline. Initialised to None
        # because no call has happened yet; replaced after every search/refine
        # that flowed through `_apply_egress_guard`.
        self._last_egress_outcome: Optional[EgressGuardOutcome] = None
        self._noop_guard_outcome = ZeroResultGuardOutcome(fired=False, ladder_step='none', original_filter_count=0, relaxed_filter_count=0, relaxation_reason='')
        self._cost_budget_eranker_outcome = ERankerOutcome(applied=False, client=self._eranker_client.name, skipped_reason='cost_budget_breach')
        self._empty_refine_eranker_outcome = ERankerOutcome(applied=False, client=self._eranker_client.name, skipped_reason='empty_candidates')
        self._explore_fallback_eranker_outcome = ERankerOutcome(applied=False, client=self._eranker_client.name, skipped_reason='explore_fallback')
        self._legacy_bare_cache_eranker_outcome = ERankerOutcome(applied=False, client=self._eranker_client.name, skipped_reason='legacy_bare_cache')
        self._last_eranker_outcome = ERankerOutcome(applied=False, client=self._eranker_client.name, skipped_reason='not_yet_invoked')
        self._last_ranking_stages: Dict[str, Any] = {}
        self._stage_scratch: Dict[str, Any] = {}
        # reasoning-trace toggle. Read once at construction so the
        # hot path doesn't touch the config object on every call. When the
        # block is absent (additive default) or `enabled=False`, the
        # orchestrator never constructs a `ReasoningTrace` (zero overhead);
        # when enabled, every search() call gets one trace bounded by
        # `max_steps_per_trace` and attached to `QueryIntent.reasoning_trace`.
        rt_cfg = getattr(config, 'reasoning_trace', None)
        self._reasoning_trace_enabled = bool(rt_cfg.enabled) if rt_cfg is not None else False
        self._reasoning_trace_max_steps = int(rt_cfg.max_steps_per_trace) if (rt_cfg is not None and rt_cfg.enabled) else 0
        self._fb_result_cache: Optional[Tuple[RankedResults, float]] = None
        # Per-request explore-rail prewarm: search() registers; timeout/ZRG consume.
        self._explore_prewarm_by_rid: Dict[str, 'asyncio.Task'] = {}
        self._explore_prewarm_gate_by_rid: Dict[str, asyncio.Event] = {}
        # Last RetrievalError per backend name from _safe_retrieve (cleared each search).
        # Used so empty hybrid does not look like inventory_empty when Qdrant is down.
        self._last_retrieve_errors: Dict[str, BaseException] = {}

    def _new_trace(self) -> Optional[ReasoningTrace]:
        """construct a fresh per-call ``ReasoningTrace`` or None.

        Returns ``None`` when the global toggle is disabled so the call sites
        can guard a single ``if trace is not None`` check before emitting.
        Bounded by the configured ``max_steps_per_trace``.
        """
        if not self._reasoning_trace_enabled:
            return None
        return ReasoningTrace(max_steps=self._reasoning_trace_max_steps)

    def _truncate(self, results: RankedResults, limit: Optional[int] = None) -> RankedResults:
        """Cap returned items to ``limit`` (defaulting to general.max_results).

        Preserves ``multi_intent_envelope`` because the strip describes the
        intent-level fan-out (which doesn't change when we truncate the merged
        item list) — the strip is over the search, not over the visible items.
        """
        cap = min(limit, self._config.general.max_results) if limit is not None else self._config.general.max_results
        if len(results.items) <= cap:
            return results
        return RankedResults(
            request_id=results.request_id,
            items=results.items[:cap],
            total_candidates=results.total_candidates,
            fusion_latency_ms=results.fusion_latency_ms,
            cache_hit=results.cache_hit,
            multi_intent_envelope=results.multi_intent_envelope,
            failure_mode=results.failure_mode,
            query_intent=results.query_intent,
        )

    @property
    def last_egress_outcome(self) -> Optional[EgressGuardOutcome]:
        """Read-only handle on the most-recent egress-guard outcome.

        Returns the per-call audit envelope (counters + per-item decisions
        + latency) from the last call to ``search`` / ``refine`` that flowed
        through the egress guard. ``None`` when the guard is not wired or
        no call has happened yet. Intended for SRE drilldowns + test
        assertions; not part of the response envelope returned to clients.
        """
        return self._last_egress_outcome

    def _eranker_bucket_user_key(self, user_context: Optional[UserContext], request_id: str) -> str:
        """Stable key for eRanker A/B bucketing (authenticated user_id else session_id else request_id)."""
        if user_context is None:
            return request_id
        uid = user_context.user_id
        if isinstance(uid, str) and uid.strip():
            return uid
        return user_context.session_id

    async def _maybe_emit_eranker_feedback(self, request_id: str, intent: QueryIntent, outcome: ERankerOutcome, shadow_would_reorder: Optional[bool]) -> None:
        """Append ``eranker_applied`` signal when the signal store is wired."""
        if self._signal_store is None:
            return
        try:
            sig = FeedbackSignal(
                signal_id=FeedbackSignal.new_signal_id(),
                request_id=request_id,
                signal_type='eranker_applied',
                payload={
                    'applied': bool(outcome.applied),
                    'skipped_reason': str(outcome.skipped_reason or ''),
                    'latency_ms': float(outcome.latency_ms) if outcome.latency_ms is not None else 0.0,
                    'http_status': int(outcome.http_status) if outcome.http_status is not None else 0,
                    'shadow_would_reorder': bool(shadow_would_reorder) if shadow_would_reorder is not None else False,
                },
                intent_record_id=intent.intent_record_id,
                signal_origin='eranker',
            )
            await self._signal_store.record_async(sig)
        except ValidationError as e:
            logger.warning(f"eranker_feedback_signal_skipped request_id={request_id} error_type={type(e).__name__} error={str(e)}")

    async def _apply_eranker(self, request_id: str, intent: QueryIntent, ranked: RankedResults, user_context: Optional[UserContext]) -> Tuple[RankedResults, ERankerOutcome]:
        """Call external eRanker (Layer 4) with latency budget; personalization signals live in eRanker."""
        cfg = self._config.retrieval.eranker
        client_name = self._eranker_client.name
        if not cfg.enabled:
            return ranked, ERankerOutcome(applied=False, client=client_name, skipped_reason='disabled')
        if not ranked.items:
            return ranked, ERankerOutcome(applied=False, client=client_name, skipped_reason='empty_results')
        if cfg.backend == 'http' and cfg.skip_when_backend_unhealthy and not self._health.is_healthy('eranker'):
            oc = ERankerOutcome(applied=False, client=client_name, skipped_reason='backend_unhealthy')
            await self._maybe_emit_eranker_feedback(request_id, intent, oc, None)
            return ranked, oc
        t0 = time.monotonic()
        try:
            out = await asyncio.wait_for(self._eranker_client.rank(request_id, intent, ranked, user_context), timeout=self._eranker_budget_s)
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            self._health.record('eranker', False)
            http_st = self._eranker_client.last_http_status
            oc = ERankerOutcome(applied=False, client=client_name, skipped_reason='timeout', latency_ms=elapsed_ms, http_status=http_st)
            await self._maybe_emit_eranker_feedback(request_id, intent, oc, None)
            logger.warning(f"eranker_timeout request_id={request_id} client={client_name} budget_ms={self._eranker_budget_s * 1000.0:.1f} elapsed_ms={elapsed_ms:.1f}")
            return ranked, oc
        except _GUARD_EXC as e:
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            self._health.record('eranker', False)
            http_st = self._eranker_client.last_http_status
            oc = ERankerOutcome(applied=False, client=client_name, skipped_reason='error', latency_ms=elapsed_ms, http_status=http_st)
            await self._maybe_emit_eranker_feedback(request_id, intent, oc, None)
            logger.warning(f"eranker_failed request_id={request_id} client={client_name} error_type={type(e).__name__} error={str(e)}")
            return ranked, oc
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        self._health.record('eranker', True)
        http_st = self._eranker_client.last_http_status
        same_order = len(ranked.items) == len(out.items) and all(a.item_id == b.item_id for a, b in zip(ranked.items, out.items))
        shadow_serve = bool(cfg.shadow_enabled and cfg.shadow_serve_fused)
        if shadow_serve:
            would_reorder = not same_order
            oc = ERankerOutcome(applied=False, client=client_name, skipped_reason='shadow_serve_fused', latency_ms=elapsed_ms, http_status=http_st)
            await self._maybe_emit_eranker_feedback(request_id, intent, oc, would_reorder)
            return ranked, oc
        if same_order:
            oc = ERankerOutcome(applied=False, client=client_name, skipped_reason='pass_through', latency_ms=elapsed_ms, http_status=http_st)
            await self._maybe_emit_eranker_feedback(request_id, intent, oc, False)
            return out, oc
        oc = ERankerOutcome(applied=True, client=client_name, latency_ms=elapsed_ms, http_status=http_st)
        await self._maybe_emit_eranker_feedback(request_id, intent, oc, None)
        return out, oc

    async def _apply_egress_guard(self, results: RankedResults) -> RankedResults:
        """Final output-side scrub on every return path.

        Wraps the (sync) ``EgressGuard.apply`` call in ``asyncio.to_thread``
        so the event loop stays responsive on large result sets. The guard
        is NOT latency-gated (safety, not quality) — exceptions inside the
        gate fail closed per item (drop with reason) rather than leaking.

        Behaviour:
        - When the guard is None (test paths), returns ``results`` verbatim.
        - When the guard is :class:`NoOpEgressGuard`, returns ``results``
          verbatim with a "kept-everything" outcome stamped on
          ``last_egress_outcome``.
        - When the guard is :class:`EgressGuard`, returns the scrubbed
          ``RankedResults`` and stamps the outcome on ``last_egress_outcome``.

        :param results: RankedResults - Final ranked results, post-truncation
        :return: RankedResults - Possibly-scrubbed results (a NEW instance
            when any item was mutated; the same instance when nothing changed)
        """
        if self._egress_guard is None:
            # Reset the accessor so a stale outcome from a previous call doesn't
            # leak into a test that checks the current call.
            self._last_egress_outcome = None
            return results
        scrubbed, outcome = await asyncio.to_thread(self._egress_guard.apply, results)
        self._last_egress_outcome = outcome
        return scrubbed

    async def _retrieve_structured(self, intent: QueryIntent, top_k: int) -> CandidateSet:
        """Run the structured retriever with the intermediate cache in front."""
        filters = extract_filters_from_intent(intent)
        if filters:
            cached = self._structured_cache.get(intent.query_type, filters)
            if cached is not None:
                return cached
        result = await self._structured.retrieve(intent, top_k)
        if filters and result.candidates:
            self._structured_cache.put(intent.query_type, filters, result)
        return result

    async def _explore_primary_retrieve(self, intent: QueryIntent, request_id: str, user_id: Optional[str], max_per_rail: int) -> Optional[RankedResults]:
        """Call ExploreComposer rails for an explore-classified query.

        CH rail fan-out and semantic vector retrieval run concurrently via
        semantic_items_fut so neither blocks the other.

        Returns the RankedResults when rails yield at least one item; returns
        None when rails are empty so the caller can fall back to vector search.
        """
        sem_top_k = int(self._config.explore.zero_result_guard.semantic_fallback_top_k)
        sem_task: Optional[asyncio.Task] = None
        if self._vector is not None:
            try:
                sem_task = asyncio.create_task(self._quick_semantic_retrieve(intent, sem_top_k))
            except _GUARD_EXC:
                sem_task = None
        try:
            ranked, _rail_response = await self._explore_composer.compose_fallback(
                intent=intent,
                max_per_rail_override=max_per_rail,
                user_id=user_id,
                semantic_items_fut=sem_task,
                exclude_hard_filter_slots=_EXPLORE_RAIL_SELECTOR_SLOTS,
            )
        except _GUARD_EXC as e:
            if sem_task is not None and not sem_task.done():
                sem_task.cancel()
            logger.warning(f"explore_primary_retrieve_failed request_id={request_id} error_type={type(e).__name__} error={str(e)} falling_back_to_vector")
            return None
        if not ranked.items:
            # CH rails empty (or all soft-failed) — Qdrant filter_only_rails keeps
            # explore-quality ranking under the same hard filters when enabled.
            tf_cfg = self._config.general.search.timeout_fallback
            if bool(tf_cfg.qdrant_rails_when_ch_empty):
                qdrant_items = await self._qdrant_filter_only_rails_as_ranked(intent, int(max_per_rail))
                if qdrant_items:
                    logger.info(
                        f"explore_primary_retrieve_qdrant_rails request_id={request_id} "
                        f"items={len(qdrant_items)}"
                    )
                    return RankedResults(
                        request_id=request_id,
                        items=qdrant_items,
                        total_candidates=len(qdrant_items),
                        fusion_latency_ms=0.0,
                        cache_hit=None,
                        failure_mode='explore_fallback_rail',
                    )
            logger.info(f"explore_primary_retrieve_empty request_id={request_id} falling_back_to_vector")
            return None
        return ranked

    def _explore_rail_substrate_ready(self, ch_healthy: bool) -> bool:
        """True when CH composer or Qdrant rail ladder can build explore rails."""
        return (
            (bool(ch_healthy) and self._explore_composer is not None)
            or bool(self._config.general.search.timeout_fallback.qdrant_rails_when_ch_empty)
        )

    def _should_start_explore_rails_prewarm(
        self,
        intent: QueryIntent,
        *,
        ch_healthy: bool,
        want_rail_merge: bool,
    ) -> bool:
        """Start rails when merge and/or zero-result-guard may consume them."""
        if not self._explore_rail_substrate_ready(ch_healthy):
            return False
        if want_rail_merge:
            return True
        return (
            self._zero_result_guard is not None
            and self._zero_result_guard.enabled
            and intent.query_type in _QUERY_TYPES_USING_ZERO_RESULT_GUARD
        )

    def _ensure_explore_prewarm_maps(self) -> None:
        """Lazy-init per-request explore prewarm maps (supports __new__ test harnesses)."""
        if getattr(self, '_explore_prewarm_by_rid', None) is None:
            self._explore_prewarm_by_rid = {}
        if getattr(self, '_explore_prewarm_gate_by_rid', None) is None:
            self._explore_prewarm_gate_by_rid = {}

    def _prune_finished_explore_prewarm(self) -> None:
        """Drop finished request_id entries so cancel paths cannot grow maps unboundedly."""
        self._ensure_explore_prewarm_maps()
        for _rid, _task in list(self._explore_prewarm_by_rid.items()):
            if _task.done():
                self._explore_prewarm_by_rid.pop(_rid, None)
                self._explore_prewarm_gate_by_rid.pop(_rid, None)

    def _register_explore_prewarm(self, request_id: str, task: 'asyncio.Task') -> None:
        """Publish in-flight explore rail task for timeout / ZRG consumers."""
        self._prune_finished_explore_prewarm()
        self._explore_prewarm_by_rid[request_id] = task
        gate = self._explore_prewarm_gate_by_rid.get(request_id)
        if gate is None:
            gate = asyncio.Event()
            self._explore_prewarm_gate_by_rid[request_id] = gate
        gate.set()

    def _release_explore_prewarm(self, request_id: str, *, cancel: bool) -> None:
        """Drop registry entry; optionally cancel an unfinished task."""
        self._ensure_explore_prewarm_maps()
        task = self._explore_prewarm_by_rid.pop(request_id, None)
        self._explore_prewarm_gate_by_rid.pop(request_id, None)
        if cancel and task is not None and not task.done():
            task.cancel()

    def _start_explore_rails_prewarm_task(
        self,
        *,
        intent: QueryIntent,
        request_id: str,
        user_id: Optional[str],
        ch_healthy: bool,
    ) -> asyncio.Task:
        """Create + register explore rail task (CH composer or Qdrant filter_only_rails)."""
        max_per_rail = int(self._config.explore.zero_result_guard.explore_fallback_max_per_rail)
        if ch_healthy and self._explore_composer is not None:
            task: asyncio.Task = asyncio.create_task(
                self._explore_primary_retrieve(
                    intent=intent,
                    request_id=request_id,
                    user_id=user_id,
                    max_per_rail=max_per_rail,
                )
            )
            logger.info(
                f"explore_rails_prewarm_started request_id={request_id} "
                f"query_type={intent.query_type} ch_healthy={ch_healthy}"
            )
        else:
            async def _qdrant_rails_prewarm() -> Optional[RankedResults]:
                items = await self._qdrant_filter_only_rails_as_ranked(intent, max_per_rail)
                if not items:
                    return None
                return RankedResults(
                    request_id=request_id,
                    items=items,
                    total_candidates=len(items),
                    fusion_latency_ms=0.0,
                    cache_hit=None,
                    failure_mode='explore_fallback_rail',
                )

            task = asyncio.create_task(_qdrant_rails_prewarm())
            logger.info(
                f"explore_rails_prewarm_started request_id={request_id} "
                f"query_type={intent.query_type} ch_healthy={ch_healthy} "
                f"source=qdrant_filter_only_rails"
            )
        self._register_explore_prewarm(request_id, task)
        return task

    async def _await_registered_explore_prewarm(
        self,
        request_id: str,
    ) -> Optional[RankedResults]:
        """Wait for search()-registered explore rails, then return RankedResults or None."""
        tf_cfg = self._config.general.search.timeout_fallback
        if not bool(tf_cfg.consume_search_explore_prewarm):
            return None
        wait_s = float(tf_cfg.search_explore_prewarm_wait_seconds)
        self._ensure_explore_prewarm_maps()
        gate = self._explore_prewarm_gate_by_rid.get(request_id)
        if gate is None:
            gate = asyncio.Event()
            self._explore_prewarm_gate_by_rid[request_id] = gate
        try:
            await asyncio.wait_for(gate.wait(), timeout=wait_s)
        except asyncio.TimeoutError:
            logger.debug(
                f"explore_rails_prewarm_wait_timeout request_id={request_id} "
                f"wait_s={wait_s}"
            )
            return None
        task = self._explore_prewarm_by_rid.get(request_id)
        if task is None:
            return None
        try:
            ranked = await asyncio.shield(task)
        except _GUARD_EXC as err:
            logger.warning(
                f"explore_rails_prewarm_consume_failed request_id={request_id} "
                f"error_type={type(err).__name__} error={err}"
            )
            return None
        if ranked is None or not ranked.items:
            return None
        return ranked

    async def _gather_candidates(self, intent: QueryIntent, top_k: Optional[int] = None) -> List[CandidateSet]:
        """Run the appropriate retrievers in parallel and return their candidate sets.
        When a `DegradationPlanner` is wired, unhealthy backends are dropped and
        replaced by their configured fallback chain before tasks are scheduled.
        Per-backend success / failure is recorded on the `BackendHealthRegistry`.
        ``top_k`` overrides the per-backend config floor when the caller requests
        more candidates than the config default (e.g. top_k=100 with config=50).
        """
        requested: List[str] = []
        if self._config.retrieval.vector.enabled and intent.query_type in _QUERY_TYPES_USING_VECTOR:
            requested.append('vector')
        if self._config.retrieval.structured.enabled and intent.query_type in _QUERY_TYPES_USING_STRUCTURED:
            requested.append('structured')
        if self._config.retrieval.sql.enabled and intent.query_type in _QUERY_TYPES_USING_SQL:
            requested.append('sql')

        # Residual-aware modality dispatch. When residual is filter-dominant
        # (``empty`` / ``navigational``) and a hard chip exists, non-hybrid
        # deployments drop the dense vector leg (structured owns filters).
        # Under Qdrant unified-hybrid the structured leg is NoOp, so the hybrid
        # vector backend IS the filter path — only drop it for kinds listed in
        # ``hybrid_mode_residual_kinds`` (production keeps that list empty).
        # Empty encode text is handled inside QdrantHybridRetriever via scroll.
        residual_dispatch_cfg = self._config.retrieval.residual_dispatch
        qdrant_cfg = self._config.retrieval.qdrant
        hybrid_active = qdrant_cfg is not None and qdrant_cfg.hybrid.enabled
        if (residual_dispatch_cfg is not None and residual_dispatch_cfg.enabled and 'vector' in requested and intent.residual_kind in residual_dispatch_cfg.skip_vector_when.residual_kinds):
            chip_ok = (not residual_dispatch_cfg.skip_vector_when.require_hard_chip_coverage) or _intent_has_hard_chip(intent)
            # Hybrid: drop only when residual_kind ∈ hybrid_mode_residual_kinds.
            # Non-hybrid: drop for any kind in residual_kinds (legacy dual-backend).
            hybrid_drop_ok = ((not hybrid_active) or intent.residual_kind in residual_dispatch_cfg.skip_vector_when.hybrid_mode_residual_kinds)
            if chip_ok and hybrid_drop_ok:
                requested.remove('vector')
                logger.info(
                    f"retrieval_modality_dispatch reason=residual_filter_dominant "
                    f"request_id={intent.request_id} residual_kind={intent.residual_kind} "
                    f"hybrid_active={hybrid_active} dropped_backends=['vector'] active={requested}"
                )

        # Taxonomy collapse: pure-filter queries now classify as
        # ``hybrid`` and therefore unconditionally trigger the SQL retriever
        # (``hybrid`` is in ``_QUERY_TYPES_USING_SQL``). The price-band SQL
        # leg has no useful signal when the residual is filter-dominant —
        # the structured retriever already handles the hard chips, and SQL
        # fanout returns generic price-band candidates that pollute the
        # fusion. Drop ``sql`` whenever the residual-dispatch policy would
        # also drop the dense vector leg (same conditions, same intent).
        # Exception: when structured returns 0 candidates (empty index / no seed
        # loaded), SQL is rescued post-gather as a filter-aware fallback so
        # filter-only queries don't silently return zero results.
        _sql_suppressed_by_residual = False
        if (residual_dispatch_cfg is not None and residual_dispatch_cfg.enabled and 'sql' in requested and intent.residual_kind in residual_dispatch_cfg.skip_vector_when.residual_kinds):
            chip_ok_sql = (not residual_dispatch_cfg.skip_vector_when.require_hard_chip_coverage) or _intent_has_hard_chip(intent)
            if chip_ok_sql:
                requested.remove('sql')
                _sql_suppressed_by_residual = True
                logger.info(
                    f"retrieval_modality_dispatch reason=residual_filter_dominant "
                    f"request_id={intent.request_id} residual_kind={intent.residual_kind} "
                    f"dropped_backends=['sql'] active={requested}"
                )

        # Semantic residual + vector active: SQL is dropped post-gather whenever
        # vector returns candidates (see _should_drop_sql_for_semantic_residual).
        # Scheduling SQL anyway forces gather to wait on a ~1.5s ClickHouse price
        # fan-out that is then discarded — dominant happy-path latency regressor.
        # Skip pre-gather; keep SQL only when vector is not in the plan (fallback).
        if (
            'sql' in requested
            and 'vector' in requested
            and intent.residual_kind == 'semantic'
        ):
            requested.remove('sql')
            logger.info(
                f"retrieval_modality_dispatch reason=semantic_residual_sql_skip "
                f"request_id={intent.request_id} residual_kind={intent.residual_kind} "
                f"dropped_backends=['sql'] active={requested}"
            )

        if not requested:
            return []
        plan = self._degradation.plan(requested)
        _m = self._config.retrieval.over_fetch_multiplier
        tasks = []
        sources_used: List[str] = []
        for backend in plan.active_backends:
            if backend == 'vector':
                vec_k = int(max(self._config.retrieval.vector.top_k, top_k) * _m) if top_k else int(self._config.retrieval.vector.top_k * _m)
                tasks.append(self._safe_retrieve('vector', self._vector.retrieve(intent, vec_k)))
                sources_used.append('vector')
            elif backend == 'structured':
                str_k = int(max(self._config.retrieval.structured.top_k, top_k) * _m) if top_k else int(self._config.retrieval.structured.top_k * _m)
                tasks.append(self._safe_retrieve('structured', self._retrieve_structured(intent, str_k)))
                sources_used.append('structured')
            elif backend == 'sql':
                sql_k = int(max(self._config.retrieval.sql.top_k, top_k) * _m) if top_k else int(self._config.retrieval.sql.top_k * _m)
                tasks.append(self._safe_retrieve('sql', self._sql.retrieve(intent, sql_k)))
                sources_used.append('sql')
        if not tasks:
            logger.warning(f"retrieval_no_backends_available request_id={intent.request_id} requested={requested} mode={plan.mode}")
            return []
        results = await asyncio.gather(*tasks)
        results = [r for r in results if r is not None]
        if plan.dropped_backends:
            logger.warning(f"retrieval_degraded request_id={intent.request_id} mode={plan.mode} active={sources_used} dropped={plan.dropped_backends} notes={plan.notes}")
        _sizes = [len(cs.candidates) for cs in results]
        logger.info(f"retrieval_gathered request_id={intent.request_id} query_type={intent.query_type} sources={sources_used} sizes={_sizes}")
        if self._config.measurement.ranking_stage_attribution.enabled:
            self._stage_scratch['retrieve_sources'] = list(sources_used)
            self._stage_scratch['retrieve_sizes'] = list(_sizes)

        # Semantic-residual SQL demotion: ``residual_dispatch`` above only covers
        # filter-dominant residual kinds (empty/navigational). A themed query
        # ("coffee pizza domain under $100") classifies as residual_kind=semantic
        # yet still unconditionally triggers SQL via the hybrid query_type collapse.
        # SqlRetriever ranks purely by a static item score with zero query
        # relevance, so once vector already has a live pool to rank the concept
        # against, SQL only contributes RRF-equal-weighted noise that backfills
        # the post-hard-gate survivor pool with off-theme rows. Drop it.
        if _should_drop_sql_for_semantic_residual(intent.residual_kind, results):
            _dropped_sql = [cs for cs in results if cs.source == 'sql' and cs.candidates]
            if _dropped_sql:
                results = [cs for cs in results if cs.source != 'sql']
                logger.info(
                    f"retrieval_sql_dropped_semantic_residual request_id={intent.request_id} "
                    f"reason=vector_has_candidates dropped_candidates={sum(len(cs.candidates) for cs in _dropped_sql)}"
                )

        # SQL rescue: when SQL was suppressed by residual_dispatch (structured
        # expected to own filter-only queries) but structured returned 0 candidates
        # (empty in-memory index / seed not yet loaded), re-run SQL with the
        # original filtered intent so filter-only queries aren't silently empty.
        # Skip when vector already returned candidates — under Qdrant hybrid the
        # vector leg is filter_only_rails (structured is NoOp), so rescuing SQL
        # would re-pollute fusion with bid_count-ordered junk after we just
        # suppressed SQL for residual_filter_dominant.
        if (
            _sql_suppressed_by_residual
            and self._sql is not None
            and self._config.retrieval.sql.enabled
            and intent.query_type in _QUERY_TYPES_USING_SQL
            and sum(len(cs.candidates) for cs in results if cs.source == 'structured') == 0
            and sum(len(cs.candidates) for cs in results if cs.source == 'vector') == 0
        ):
            sql_k = int(max(self._config.retrieval.sql.top_k, top_k) * _m) if top_k else int(self._config.retrieval.sql.top_k * _m)
            sql_rescue = await self._safe_retrieve('sql', self._sql.retrieve(intent, sql_k))
            if sql_rescue is not None and sql_rescue.candidates:
                results.append(sql_rescue)
                logger.info(f"retrieval_sql_rescue request_id={intent.request_id} reason=structured_and_vector_empty candidates={len(sql_rescue.candidates)}")

        return list(results)

    async def _safe_retrieve(self, backend: str, awaitable: Awaitable[Optional[CandidateSet]]) -> Optional[CandidateSet]:
        """Await a retriever, record health, and convert failures to None (caller filters).

        Stores the last ``RetrievalError`` on ``_last_retrieve_errors[backend]`` so
        empty fusion can be reported as ``qdrant_unavailable`` instead of a false
        ``inventory_empty`` / filter-relax notice when Qdrant is down.
        """
        try:
            result = await awaitable
        except RetrievalError as e:
            logger.warning(f"retrieval_backend_failed backend={backend} error_type={type(e).__name__} error={str(e)}")
            self._health.record(backend, success=False)
            self._last_retrieve_errors[backend] = e
            return None
        self._health.record(backend, success=True)
        self._last_retrieve_errors.pop(backend, None)
        return result

    def _combine_rewrite_with_l0_extract_enabled(self) -> bool:
        qt = getattr(self._config.qi, 'query_transformer', None)
        if qt is None:
            return False
        val = getattr(qt, 'combine_rewrite_with_l0_extract', None)
        return isinstance(val, bool) and val

    def _on_rewrite_reject_reextract_enabled(self) -> bool:
        qt = getattr(self._config.qi, 'query_transformer', None)
        if qt is None:
            return False
        val = getattr(qt, 'on_rewrite_reject_reextract', None)
        return isinstance(val, bool) and val

    async def _run_spell_and_transform(
        self, normalized: str, rid: str, trace: Optional[ReasoningTrace],
    ) -> Tuple[Optional[SpellCorrection], Optional[QueryTransformResult], Optional[str]]:
        """Parallel spell-correct + query rewrite on ``normalized`` (shared by overlap path).

        When ``combine_rewrite_with_l0_extract`` is on, skips standalone transform LLM
        (rewrite comes from the combined L0 call in ``_preprocess_query``).
        """
        _pre_qi_tasks = []
        _pre_qi_keys: List[str] = []
        if self._spell_corrector is not None:
            _pre_qi_tasks.append(asyncio.to_thread(self._spell_corrector.correct, normalized))
            _pre_qi_keys.append('spell')
        _skip_standalone_transform = self._combine_rewrite_with_l0_extract_enabled()
        if self._query_transformer is not None and not _skip_standalone_transform:
            _pre_qi_tasks.append(self._query_transformer.transform(normalized))
            _pre_qi_keys.append('transform')
        _pre_qi_raw: list = await asyncio.gather(*_pre_qi_tasks, return_exceptions=True) if _pre_qi_tasks else []
        _pre_qi: dict = {}
        for _pk, _pr in zip(_pre_qi_keys, _pre_qi_raw):
            if isinstance(_pr, BaseException):
                logger.warning(f"pre_qi_task_failed request_id={rid} task={_pk} error_type={type(_pr).__name__} error={_pr}")
                _pre_qi[_pk] = None
            else:
                _pre_qi[_pk] = _pr
        transform_result: Optional[QueryTransformResult] = _pre_qi.get('transform')
        spell_correction: Optional[SpellCorrection] = _pre_qi.get('spell')
        if trace is not None and self._spell_corrector is not None:
            trace.add(
                'spell_correct',
                f"corrections={0 if spell_correction is None else len(spell_correction.corrections)} "
                f"applied={'true' if (spell_correction is not None and self._spell_auto_apply) else 'false'}",
            )
        _corrected_transform_query: Optional[str] = None
        if transform_result is not None and transform_result.transformed:
            _corrected_transform_query = transform_result.query
            logger.info(
                f"query_transform_applied request_id={rid} mode={transform_result.mode} "
                f"original_len={len(normalized)} transformed_len={len(transform_result.query)}"
            )
            if trace is not None:
                trace.add(
                    'query_transform',
                    f"mode={transform_result.mode} original_len={len(normalized)} "
                    f"transformed_len={len(transform_result.query)}",
                )
        elif spell_correction is not None and self._spell_auto_apply:
            logger.info(
                f"spell_correct_applied request_id={rid} corrections={len(spell_correction.corrections)} "
                f"original_normalized_len={len(normalized)} "
                f"corrected_normalized_len={len(spell_correction.corrected_query)}"
            )
        elif spell_correction is not None:
            logger.info(
                f"spell_correct_suggested request_id={rid} corrections={len(spell_correction.corrections)} "
                f"original_normalized_len={len(normalized)} "
                f"suggested_normalized_len={len(spell_correction.corrected_query)}"
            )
        return spell_correction, transform_result, _corrected_transform_query

    async def _run_combined_or_passthrough_extract(
        self,
        gate_text: str,
        rid: str,
        trace: Optional[ReasoningTrace],
    ) -> Tuple[
        Optional[QueryTransformResult], Optional[str], Optional[IntentSlice],
        float, bool, List[Dict[str, Any]], List[Dict[str, Any]],
    ]:
        """Token-gated combined rewrite+extract or leave extract to QI (passthrough).

        Route A (token_count > rewrite_threshold): one combined LLM call.
        Route B (≤ threshold): passthrough transform; extract stays in QI ensemble.
        LLM fail: no transform; pre_l0 empty so engine regex runs on raw/normalized.
        """
        empty_ids: List[Dict[str, Any]] = []
        empty_kws: List[Dict[str, Any]] = []
        qt = self._query_transformer
        extractor = getattr(self._qi, '_entity_extractor', None)
        if qt is None or extractor is None:
            return None, None, None, 0.0, False, empty_ids, empty_kws
        if not qt.needs_rewrite(gate_text):
            transform_result = qt.passthrough_result(gate_text)
            if trace is not None:
                trace.add('query_transform', 'mode=passthrough combined=skipped reason=token_gate')
            return transform_result, None, None, 0.0, False, empty_ids, empty_kws
        try:
            outcome: L0CombinedExtractOutcome = await extractor.extract_priced_combined(gate_text)
        except LLMError as exc:
            logger.warning(
                f"combined_l0_llm_failed request_id={rid} error_type={type(exc).__name__} error={exc}"
            )
            if trace is not None:
                trace.add('query_transform', 'mode=passthrough combined=llm_fail')
            return qt.passthrough_result(gate_text), None, None, 0.0, False, empty_ids, empty_kws
        accepted = qt.accept_rewrite(gate_text, outcome.rewritten_query)
        if accepted is not None:
            transform_result = QueryTransformResult(
                query=accepted,
                original_query=gate_text,
                mode='llm_rewrite_combined',
                transformed=True,
                engine=str(getattr(extractor, 'combined_prompt_tag', '') or ''),
            )
            logger.info(
                f"query_transform_applied request_id={rid} mode={transform_result.mode} "
                f"original_len={len(gate_text)} transformed_len={len(accepted)}"
            )
            if trace is not None:
                trace.add(
                    'query_transform',
                    f"mode={transform_result.mode} original_len={len(gate_text)} "
                    f"transformed_len={len(accepted)}",
                )
            return (
                transform_result,
                accepted,
                outcome.intent_slice,
                float(outcome.cost_usd),
                bool(outcome.llm_completed),
                list(outcome.identified),
                list(outcome.keywords),
            )
        # Rewrite rejected by echo/signal guards — discard filters from rejected rewrite.
        logger.info(
            f"combined_rewrite_rejected request_id={rid} original_len={len(gate_text)} "
            f"candidate_len={len(outcome.rewritten_query)} "
            f"reextract={self._on_rewrite_reject_reextract_enabled()}"
        )
        if trace is not None:
            trace.add('query_transform', 'mode=passthrough combined=rewrite_rejected')
        transform_result = qt.passthrough_result(gate_text)
        if self._on_rewrite_reject_reextract_enabled():
            try:
                identified, cost_usd, keywords = await extractor.extract_priced(gate_text)
                slice_ = extractor._identified_to_intent_slice(identified, gate_text, keywords)
                return (
                    transform_result, None, slice_,
                    float(cost_usd) + float(outcome.cost_usd), True,
                    list(identified), list(keywords),
                )
            except LLMError as exc:
                logger.warning(
                    f"combined_reject_reextract_failed request_id={rid} "
                    f"error_type={type(exc).__name__} error={exc}"
                )
                return transform_result, None, None, float(outcome.cost_usd), False, empty_ids, empty_kws
        return (
            transform_result, None, None, float(outcome.cost_usd),
            bool(outcome.llm_completed), empty_ids, empty_kws,
        )

    async def _preprocess_query(
        self, raw_query: str, rid: str, trace: Optional[ReasoningTrace]
    ) -> Tuple[
        str, str, str, Optional[SpellCorrection], Optional[QueryTransformResult], Optional[str],
        Optional[IntentSlice], float, bool, List[Dict[str, Any]], List[Dict[str, Any]],
    ]:
        """Run the pre-QI pipeline: normalize -> spell (+ rewrite or combined L0).

        Shared by ``search``, ``classify_query``, and ``extract_filters_only``.
        Returns (normalized, effective_normalized, effective_raw_query,
        spell_correction, transform_result, corrected_transform_query,
        pre_l0_slice, pre_l0_cost_usd, pre_l0_llm_completed,
        pre_l0_identified, pre_l0_keywords).

        ``normalized`` = pre-transform text (exact-cache key).
        ``effective_*`` = transformed text when rewrite accepted; else spell-corrected
        when auto-apply; else original. When
        ``qi.query_transformer.classify_on_transformed_query`` is true, QI uses
        ``effective_normalized``; otherwise QI uses ``normalized``.
        """
        normalized = self._qi_normalize(raw_query)
        spell_correction, transform_result, _corrected_transform_query = await self._run_spell_and_transform(
            normalized, rid, trace,
        )
        pre_l0_slice: Optional[IntentSlice] = None
        pre_l0_cost_usd = 0.0
        pre_l0_llm_completed = False
        pre_l0_identified: List[Dict[str, Any]] = []
        pre_l0_keywords: List[Dict[str, Any]] = []
        # Spell-corrected text is the gate input when auto-apply; else normalized.
        if spell_correction is not None and self._spell_auto_apply:
            gate_text = spell_correction.corrected_query
        else:
            gate_text = normalized
        if self._combine_rewrite_with_l0_extract_enabled():
            (
                transform_result,
                _corrected_transform_query,
                pre_l0_slice,
                pre_l0_cost_usd,
                pre_l0_llm_completed,
                pre_l0_identified,
                pre_l0_keywords,
            ) = await self._run_combined_or_passthrough_extract(gate_text, rid, trace)
        if transform_result is not None and transform_result.transformed:
            effective_normalized = transform_result.query
            effective_raw_query = transform_result.query
        elif spell_correction is not None and self._spell_auto_apply:
            effective_normalized = spell_correction.corrected_query
            effective_raw_query = spell_correction.corrected_query
        else:
            effective_normalized = normalized
            effective_raw_query = raw_query
        return (
            normalized, effective_normalized, effective_raw_query,
            spell_correction, transform_result, _corrected_transform_query,
            pre_l0_slice, pre_l0_cost_usd, pre_l0_llm_completed,
            pre_l0_identified, pre_l0_keywords,
        )

    def _classify_on_transformed_query_enabled(self) -> bool:
        qt = getattr(self._config.qi, 'query_transformer', None)
        if qt is None:
            return False
        val = getattr(qt, 'classify_on_transformed_query', None)
        return isinstance(val, bool) and val

    def _pre_normalized_for_qi(
        self,
        *,
        normalized: str,
        effective_normalized: str,
    ) -> str:
        """Text passed to QIEngine as ``pre_normalized`` (split / classify / L0)."""
        if self._classify_on_transformed_query_enabled():
            return effective_normalized
        return normalized

    def _encode_from_rewrite_enabled(self) -> bool:
        qt = getattr(self._config.qi, 'query_transformer', None)
        if qt is None:
            return False
        val = getattr(qt, 'encode_from_rewrite', None)
        return isinstance(val, bool) and val

    def _apply_encode_from_rewrite(
        self,
        intent: QueryIntent,
        transform_result: Optional[QueryTransformResult],
    ) -> QueryIntent:
        """Rebuild residual/encode from rewritten text when rewrite was accepted.

        When ``transformed=True``, vector ANN uses rewrite text only — never raw.
        """
        if transform_result is None or not transform_result.transformed:
            return intent
        if not self._encode_from_rewrite_enabled():
            return intent
        rewrite = transform_result.query
        if not isinstance(rewrite, str) or not rewrite.strip():
            return intent
        all_entities = [e for s in (intent.slices or []) for e in (s.entities or [])]
        residual_cfg = getattr(self._config.qi, 'residual', None)
        if residual_cfg is not None and getattr(residual_cfg, 'enabled', False):
            sem_q, res_kind = extract_residual(rewrite, all_entities, residual_cfg)
            nav = frozenset(getattr(residual_cfg, 'navigational_tokens', ()) or ())
            encode_text = build_semantic_encode_text(rewrite, all_entities, sem_q, nav)
        else:
            sem_q = rewrite
            res_kind = 'semantic'
            encode_text = rewrite
        logger.info(
            f"encode_from_rewrite_applied request_id={intent.request_id} "
            f"rewrite_len={len(rewrite)} encode_len={len(encode_text)} residual_kind={res_kind}"
        )
        return dataclasses.replace(
            intent,
            semantic_query=sem_q,
            semantic_encode_text=encode_text,
            residual_kind=res_kind,
        )

    def _attach_pre_qi(
        self,
        intent: QueryIntent,
        spell_correction: Optional[SpellCorrection],
        transform_result: Optional[QueryTransformResult],
        corrected_transform_query: Optional[str],
        trace: Optional[ReasoningTrace],
    ) -> QueryIntent:
        """Attach pre-QI outcomes (did_you_mean, query_transform, trace) to the intent.

        Shared by ``search`` and ``classify_query``. ``QueryIntent`` is logically
        immutable, so we rebuild rather than mutate.
        """
        if spell_correction is not None or trace is not None:
            intent = dataclasses.replace(
                intent,
                did_you_mean=spell_correction if spell_correction is not None else intent.did_you_mean,
                reasoning_trace=trace if trace is not None else intent.reasoning_trace,
            )
        # Attach the pre-QI transform outcome so HTTP surfaces can report which
        # transformer fired and the query text it produced. Done via replace so it
        # lands whether or not the rebuild above ran, and is preserved by every
        # later dataclasses.replace on this intent.
        if transform_result is not None:
            intent = dataclasses.replace(intent, query_transform={
                'mode': transform_result.mode,
                'engine': transform_result.engine,
                'transformed': transform_result.transformed,
                'transformed_query': corrected_transform_query if corrected_transform_query is not None else transform_result.query,
            })
        return intent

    async def classify_query(
        self, raw_query: str, request_id: Optional[str] = None, intent_record_id: Optional[str] = None
    ) -> QueryIntent:
        """Classify via the same ``_preprocess_query`` path ``search`` / ``qie_only`` use.

        Sanitize -> preprocess (spell + token-gated rewrite+extract) -> QI on effective
        text. Stops before retrieval.
        """
        if raw_query is None or not isinstance(raw_query, str):
            raise ValidationError("classify_query requires a non-null string query")
        rid = request_id if (request_id and isinstance(request_id, str)) else QueryIntent.new_request_id()
        gate = self._bind_request_cost_budget(rid)
        token = set_request_cost_observer(gate)
        try:
            self._sanitize_user_input(raw_query, rid, surface='search')
            trace = self._new_trace()
            (
                normalized,
                effective_normalized,
                _effective_raw_query,
                spell_correction,
                transform_result,
                _corrected_transform_query,
                pre_l0_slice,
                pre_l0_cost_usd,
                pre_l0_llm_completed,
                _pre_l0_identified,
                _pre_l0_keywords,
            ) = await self._preprocess_query(raw_query, rid, trace)
            intent = await self._qi.classify(
                raw_query=raw_query, request_id=rid,
                intent_record_id=intent_record_id,
                pre_normalized=self._pre_normalized_for_qi(
                    normalized=normalized, effective_normalized=effective_normalized,
                ),
                pre_l0_slice=pre_l0_slice,
                pre_l0_cost_usd=pre_l0_cost_usd,
                pre_l0_llm_completed=pre_l0_llm_completed,
            )
            if trace is not None:
                trace.add('qi_classify', f"tier={intent.decision_tier} type={intent.query_type} confidence={intent.confidence:.3f} slices={len(intent.slices)}")
            intent = self._attach_pre_qi(intent, spell_correction, transform_result, _corrected_transform_query, trace)
            intent = self._apply_encode_from_rewrite(intent, transform_result)
            return _stamp_intent_decision_cost(intent, float(gate.running_total_usd))
        finally:
            reset_request_cost_observer(token)

    async def extract_filters_only(
        self, raw_query: str, request_id: Optional[str] = None, intent_record_id: Optional[str] = None
    ) -> QueryIntent:
        """L0 filter extract + ground only — no L1/L2.

        Shares ``_preprocess_query`` with ``search`` / ``classify_query``
        (spell + token-gated combined rewrite+extract or extract-only), then
        ``QIEngine.extract_l0_filters`` (reuses ``pre_l0`` when combined already ran).
        Used by ``qie_only_mode``.
        """
        if raw_query is None or not isinstance(raw_query, str):
            raise ValidationError("extract_filters_only requires a non-null string query")
        rid = request_id if (request_id and isinstance(request_id, str)) else QueryIntent.new_request_id()
        gate = self._bind_request_cost_budget(rid)
        token = set_request_cost_observer(gate)
        try:
            self._sanitize_user_input(raw_query, rid, surface='search')
            trace = self._new_trace()
            (
                normalized,
                effective_normalized,
                _effective_raw_query,
                spell_correction,
                transform_result,
                _corrected_transform_query,
                pre_l0_slice,
                pre_l0_cost_usd,
                pre_l0_llm_completed,
                _pre_l0_identified,
                _pre_l0_keywords,
            ) = await self._preprocess_query(raw_query, rid, trace)
            intent = await self._qi.extract_l0_filters(
                raw_query=raw_query, request_id=rid,
                intent_record_id=intent_record_id,
                pre_normalized=self._pre_normalized_for_qi(
                    normalized=normalized, effective_normalized=effective_normalized,
                ),
                pre_l0_slice=pre_l0_slice,
                pre_l0_cost_usd=pre_l0_cost_usd,
                pre_l0_llm_completed=pre_l0_llm_completed,
            )
            if trace is not None:
                n_ents = sum(len(s.entities) for s in intent.slices) if intent.slices else 0
                trace.add('qi_l0_filters', f"tier={intent.decision_tier} entities={n_ents} slices={len(intent.slices)}")
            intent = self._attach_pre_qi(intent, spell_correction, transform_result, _corrected_transform_query, trace)
            intent = self._apply_encode_from_rewrite(intent, transform_result)
            return _stamp_intent_decision_cost(intent, float(gate.running_total_usd))
        finally:
            reset_request_cost_observer(token)

    def _stamp_results_llm_cost(
        self,
        results: RankedResults,
        budget: Union[RequestCostGate, QueryCostBudget, NoOpQueryCostBudget],
    ) -> RankedResults:
        """Overwrite ``query_intent.decision_cost_usd`` with all LLM spend this request.

        Budget observer records every successful ``call_structured`` (L0 extract,
        L2 classify, query rewrite, NL-SQL, …). Engine's field may only carry
        QI L0+L2; stamping makes the response / measurement reflect actual spend.
        """
        intent = results.query_intent
        if intent is None:
            return results
        stamped = _stamp_intent_decision_cost(intent, float(budget.running_total_usd))
        if stamped is intent:
            return results
        return dataclasses.replace(results, query_intent=stamped)

    def _ensure_request_cost_maps(self) -> None:
        """Init per-request cost maps when ``__new__`` test shells skip ``__init__``."""
        if not hasattr(self, '_accumulated_llm_cost_usd') or self._accumulated_llm_cost_usd is None:
            self._accumulated_llm_cost_usd = {}
        if not hasattr(self, '_active_cost_gate_by_rid') or self._active_cost_gate_by_rid is None:
            self._active_cost_gate_by_rid = {}

    def _bind_request_cost_budget(self, request_id: Optional[str]) -> RequestCostGate:
        """Build request-scoped ``RequestCostGate`` (query + optional fleet)."""
        self._ensure_request_cost_maps()
        factory = getattr(self, '_cost_budget_factory', None)
        if factory is not None:
            query_budget: Union[QueryCostBudget, NoOpQueryCostBudget] = factory(request_id)
        else:
            query_budget = NoOpQueryCostBudget(request_id=request_id)
        fleet = getattr(self, '_fleet_cost_budget', None)
        gate = RequestCostGate(query_budget=query_budget, fleet_budget=fleet)
        _REQUEST_COST_GATE.set(gate)
        self._last_cost_budget = gate
        if request_id:
            self._active_cost_gate_by_rid[request_id] = gate
        return gate

    def _finalize_request_cost_phase(self, request_id: Optional[str], gate: RequestCostGate) -> None:
        """Merge phase gate spend into the request_id rollup and clear the active gate."""
        if not request_id:
            return
        self._ensure_request_cost_maps()
        spend = float(gate.running_total_usd)
        prev = float(self._accumulated_llm_cost_usd.get(request_id, 0.0))
        self._accumulated_llm_cost_usd[request_id] = prev + spend
        if self._active_cost_gate_by_rid.get(request_id) is gate:
            self._active_cost_gate_by_rid.pop(request_id, None)

    def request_llm_cost_usd(self, request_id: Optional[str]) -> float:
        """Sum finalized phase spend plus any still-active gate for ``request_id``."""
        if not request_id:
            return 0.0
        self._ensure_request_cost_maps()
        total = float(self._accumulated_llm_cost_usd.get(request_id, 0.0))
        active = self._active_cost_gate_by_rid.get(request_id)
        if active is not None:
            total += float(active.running_total_usd)
        return total

    def clear_request_llm_cost(self, request_id: Optional[str]) -> None:
        """Release request_id rollup entries after the /search response is built."""
        if not request_id:
            return
        self._ensure_request_cost_maps()
        self._accumulated_llm_cost_usd.pop(request_id, None)
        self._active_cost_gate_by_rid.pop(request_id, None)

    async def search(
        self,
        raw_query: str,
        request_id: Optional[str] = None,
        user_context: Optional[UserContext] = None,
        intent_record_id: Optional[str] = None,
        top_k: Optional[int] = None,
        diversity_lambda: Optional[float] = None,
    ) -> Tuple[RankedResults, ERankerOutcome, ZeroResultGuardOutcome]:
        """Run the full search pipeline with per-query + fleet LLM cost tracking.

        Binds a ``RequestCostGate`` so every LLM call admits against query/fleet
        caps. On breach, ``call_structured`` raises ``LLMError`` — QI L0/L2
        catch that and degrade to regex / L1. Search never rejects the query
        with empty results for spend policy.

        :raises ValidationError: When raw_query is invalid (sanitizer / shape)
        """
        rid_pre = request_id if (request_id and isinstance(request_id, str)) else None
        gate = self._bind_request_cost_budget(rid_pre)
        token = set_request_cost_observer(gate)
        try:
            results, eranker_outcome, guard_outcome = await self._search_inner(
                raw_query, request_id, user_context, intent_record_id,
                top_k=top_k, diversity_lambda=diversity_lambda,
            )
            return self._stamp_results_llm_cost(results, gate), eranker_outcome, guard_outcome
        finally:
            reset_request_cost_observer(token)
            self._finalize_request_cost_phase(rid_pre, gate)

    async def _search_inner(
        self,
        raw_query: str,
        request_id: Optional[str] = None,
        user_context: Optional[UserContext] = None,
        intent_record_id: Optional[str] = None,
        top_k: Optional[int] = None,
        diversity_lambda: Optional[float] = None,
    ) -> Tuple[RankedResults, ERankerOutcome, ZeroResultGuardOutcome]:
        """Run the full search pipeline.

        :param raw_query: str - Query text from the API boundary
        :param request_id: Optional[str] - Caller-provided correlation id (auto-generated when None)
        :param user_context: Optional[UserContext] - When supplied, drives history capture for authenticated callers
        :param intent_record_id: Optional[str] - Caller-supplied intent
            record id. When None on a fresh ``/search`` we mint one;
            on a refine turn the API echoes the prior id so ``intent_record_id`` stays
            consistent across the refine sequence. Distinct from ``request_id``,
            which is per-call. Cache-hit branches reuse the cached envelope's
            id (the original intent stays the source of truth).
        :return: Tuple[RankedResults, ERankerOutcome, ZeroResultGuardOutcome] -
            The third element is the Zero-Result Guard outcome (``fired=False``
            and ``ladder_step='none'`` on the happy path).
        :raises ValidationError: When raw_query is invalid
        """
        if raw_query is None or not isinstance(raw_query, str):
            raise ValidationError("search requires a non-null string query")
        _soft_applier = self._soft_keyword_applier
        if _soft_applier is None:
            raise ValidationError("qi.entity_slots config is required for soft apply")
        rid = request_id if (request_id and isinstance(request_id, str)) else QueryIntent.new_request_id()
        self._last_retrieve_errors = {}
        self._reset_stage_scratch()
        # Sanitize the raw user input BEFORE normalize / cache lookup /
        # classification. Rejecting here means injection markers and PII
        # never enter the regex classifier, the cache key, or the QI logs.
        self._sanitize_user_input(raw_query, rid, surface='search')
        t0 = time.monotonic()
        # fresh per-call reasoning trace (None when toggle disabled).
        # Carried through the rest of the inner pipeline; attached to the final
        # ``QueryIntent`` so the trace lives on every consumer surface (cache
        # entry, response composition, audit log) without a separate handle.
        # Carries only structured ``key=value`` details — never raw query text.
        trace = self._new_trace()
        (
            normalized,
            effective_normalized,
            _effective_raw_query,
            spell_correction,
            transform_result,
            _corrected_transform_query,
            pre_l0_slice,
            pre_l0_cost_usd,
            pre_l0_llm_completed,
            _pre_l0_identified,
            _pre_l0_keywords,
        ) = await self._preprocess_query(raw_query, rid, trace)
        _use_exact_cache = self._config.cache.enabled
        if _use_exact_cache:
            # Key on pre-rewrite ``normalized`` (same key as put) so a lossy rewrite
            # cannot alias distinct filter queries onto one cache entry.
            exact_payload = self._exact_cache.get_payload(normalized)
            if exact_payload is not None:
                logger.info(f"search_cache_hit request_id={rid} tier=exact total_latency_ms={(time.monotonic()-t0)*1000.0:.1f}")
                return await self._serve_cache_hit(rid=rid, payload=exact_payload, tier='exact', user_context=user_context, t0=t0, top_k=top_k)
            # Bare-result entry — no intent available; eRanker not re-run.
            exact_hit = self._exact_cache.get(normalized)
            if exact_hit is not None:
                logger.info(f"search_cache_hit request_id={rid} tier=exact total_latency_ms={(time.monotonic()-t0)*1000.0:.1f}")
                exact_truncated = self._truncate(self._with_request_id(exact_hit, rid, 'exact'), limit=top_k)
                exact_scrubbed = await self._apply_egress_guard(exact_truncated)
                self._publish_ranking_stages(
                    request_id=rid,
                    cache_hit='exact',
                    fusion_latency_ms=float(exact_scrubbed.fusion_latency_ms or 0.0),
                    total_candidates=int(exact_scrubbed.total_candidates),
                    result_count=len(exact_scrubbed.items),
                    eranker_outcome=self._legacy_bare_cache_eranker_outcome,
                    guard_outcome=self._noop_guard_outcome,
                )
                return exact_scrubbed, self._legacy_bare_cache_eranker_outcome, self._noop_guard_outcome
        if self._config.cache.enabled:
            logger.debug(f"cache_lookup_miss request_id={rid} tier=exact")
            if trace is not None:
                trace.add('cache_lookup', "hit=miss")
        # QI on effective text (transformed when rewrite accepted).
        _qi_pre = self._pre_normalized_for_qi(
            normalized=normalized, effective_normalized=effective_normalized,
        )
        intent = await self._qi.classify(
            raw_query=raw_query, request_id=rid, intent_record_id=intent_record_id,
            pre_normalized=_qi_pre,
            pre_l0_slice=pre_l0_slice,
            pre_l0_cost_usd=pre_l0_cost_usd,
            pre_l0_llm_completed=pre_l0_llm_completed,
        )
        # Never log raw or normalized query text here (PII safety per responsible-ai.mdc).
        logger.debug(
            f"qi_classify_complete request_id={rid} tier={intent.decision_tier} "
            f"type={intent.query_type} confidence={intent.confidence:.3f} slices={len(intent.slices)}"
        )
        if trace is not None:
            trace.add('qi_classify', f"tier={intent.decision_tier} type={intent.query_type} confidence={intent.confidence:.3f} slices={len(intent.slices)}")
        intent = self._attach_pre_qi(intent, spell_correction, transform_result, _corrected_transform_query, trace)
        # When rewrite accepted, ANN encode from rewrite only — never raw.
        intent = self._apply_encode_from_rewrite(intent, transform_result)
        intent, _soft_kw_entities = _soft_applier.prepare_intent(intent)
        _kw_min = min_probability_fraction(self._config.qi.l0_llm_entity)
        # Keep {term, probability} for multi-keyword preference ranking; bare
        # terms still derived for encode/trace length logging.
        _kw_entries = filter_keywords(intent.keywords, _kw_min)
        _kw_terms = keyword_terms(intent.keywords, _kw_min)
        logger.debug(
            f"soft_keyword_apply_complete request_id={rid} mode={_soft_applier.mode} "
            f"soft_kw_n={len(_soft_kw_entities)} kw_terms_n={len(_kw_terms)}"
        )
        if trace is not None:
            trace.add('soft_keyword_apply', f"mode={_soft_applier.mode} soft_kw_n={len(_soft_kw_entities)} kw_terms_n={len(_kw_terms)}")
        # Filter-conflict short-circuit: when extracted slots contradict (e.g.
        # price_min > price_max, or an impossible character-length range), return
        # an explainer instead of an empty result set the user cannot interpret.
        _conflict_cfg = self._config.retrieval.conflict
        if _conflict_cfg is not None and _conflict_cfg.enabled:
            _conflicts = detect_filter_conflicts(
                extract_filters_from_intent(intent),
                _conflict_cfg.messages,
                getattr(_conflict_cfg, 'qualitative_conflict_rules', None),
            )
            if _conflicts:
                intent = dataclasses.replace(intent, conflicts=_conflicts)
                logger.info(f"filter_conflict_shortcircuit request_id={rid} kinds={[c.kind for c in _conflicts]} slots={[c.slots for c in _conflicts]}")
                _fc_ranked = RankedResults(request_id=rid, items=[], total_candidates=0, fusion_latency_ms=0.0, cache_hit=None, failure_mode='filter_conflict', query_intent=intent)
                if self._zero_result_guard is not None and self._zero_result_guard.enabled and intent.query_type in _QUERY_TYPES_USING_ZERO_RESULT_GUARD:
                    _fb_timeout_s = float(self._config.general.search.explore_fallback_timeout_seconds)
                    _fc_ch_healthy = bool(self.analytics_available) and self._health.is_healthy('clickhouse')
                    _fc_prewarm: Optional[asyncio.Task] = None
                    if self._should_start_explore_rails_prewarm(
                        intent, ch_healthy=_fc_ch_healthy, want_rail_merge=False,
                    ):
                        _fc_prewarm = self._start_explore_rails_prewarm_task(
                            intent=intent,
                            request_id=rid,
                            user_id=user_context.user_id if user_context else None,
                            ch_healthy=_fc_ch_healthy,
                        )
                    try:
                        _fc_guard = await self._zero_result_guard.run(
                            intent=intent,
                            retrieve_fn=self._retrieve_and_rank,
                            user_context=user_context,
                            semantic_retrieve_fn=self._quick_semantic_retrieve,
                            explore_fallback_timeout_s=_fb_timeout_s,
                            explore_prewarm_task=_fc_prewarm,
                        )
                    finally:
                        self._release_explore_prewarm(rid, cancel=True)
                    return RankedResults(
                        request_id=rid,
                        items=list(_fc_guard.results.items),
                        total_candidates=_fc_guard.results.total_candidates,
                        fusion_latency_ms=_fc_guard.results.fusion_latency_ms,
                        cache_hit=None,
                        failure_mode='filter_conflict',
                        query_intent=intent,
                    ), self._legacy_bare_cache_eranker_outcome, _fc_guard.outcome
                return _fc_ranked, self._legacy_bare_cache_eranker_outcome, self._noop_guard_outcome
        # Fire market snapshot concurrently with retrieval for genuine guidance queries.
        # Fires AFTER rerouting (lines 1134-1143) so only true guidance queries (not
        # rerouted-to-hybrid) create a task. Fires AFTER filter-conflict so we skip it
        # on the conflict short-circuit path. _with_guidance_envelope awaits this task
        # with snapshot_timeout_seconds instead of issuing a new serial call.
        _guidance_snapshot_task: Optional[asyncio.Task] = None
        if intent.query_type == 'guidance' and self._guidance_service is not None:
            _gs_tld = [
                str(v)
                for sl in (intent.slices or [])
                for ent in (sl.entities or [])
                if ent.name == 'tld' and getattr(ent, 'chip_kind', 'hard') == 'hard' and ent.value is not None
                for v in (ent.value if isinstance(ent.value, list) else [ent.value])
                if v
            ]
            _guidance_snapshot_task = asyncio.create_task(
                self._guidance_service.build_snapshot(request_id=intent.request_id, tld_filter=_gs_tld or None)
            )
        _is_fallback_intent = intent.decision_tier == 'fallback'
        # Pre-warm hybrid retrieval: execute 2 sample queries against Qdrant in parallel
        # to warm HNSW cache + populate result buffers before primary retrieval.
        # Degrades gracefully on Qdrant unavailability (try/except logs + continues).
        async def _run_hybrid_prewarm() -> None:
            if self._vector is None or self._vector.source != 'vector':
                return
            sample_queries = [
                'ai startup domain under 1500 with some traffic',
                'domains like stripe or plaid',
                # Misspell residual — warms char-ngram Prefetch alongside BM42.
                'hiigh rentalls',
            ]
            warm_tasks = []
            for idx, sq in enumerate(sample_queries):
                try:
                    _nq = normalize_query(
                        sq,
                        self._config.general.max_query_length,
                        normalize=self._config.qi.normalize,
                    )
                    warm_intent = dataclasses.replace(
                        intent,
                        request_id=f"{rid}_prewarm_{idx}",
                        raw_query=sq,
                        normalized_query=_nq,
                        decision_tier='prewarm',
                        decision_cost_usd=0.0,
                        did_you_mean=None,
                        reasoning_trace=None,
                        conflicts=[],
                        # Non-empty encode text forces dense + BM42 + ngram Prefetch
                        # (empty residual would skip to filter-only scroll).
                        semantic_encode_text=_nq,
                        semantic_query=_nq,
                    )
                    warm_tasks.append(self._vector.retrieve(warm_intent, top_k=5))
                except _GUARD_EXC as e:
                    logger.warning(f"hybrid_prewarm_intent_build_failed sq='{sq}' error={str(e)}")
            if warm_tasks:
                try:
                    await asyncio.gather(*warm_tasks, return_exceptions=True)
                    logger.debug(f"hybrid_prewarm_completed request_id={rid} samples={len(warm_tasks)}")
                except _GUARD_EXC as e:
                    logger.warning(f"hybrid_prewarm_failed request_id={rid} error={str(e)}")

        _prewarm_hybrid_task: Optional[asyncio.Task] = None
        if intent.query_type == 'hybrid' and self._config.general.search.hybrid_prewarm_enabled:
            _prewarm_hybrid_task = asyncio.create_task(_run_hybrid_prewarm())
            logger.debug(f"hybrid_prewarm_started request_id={rid}")
        # ClickHouse substrate gate. Wire+enabled AND health-registry not unhealthy.
        # Skip CH rail fan-out when unhealthy — no timeouts / exception burn.
        _ch_up = self.analytics_available
        _ch_healthy = bool(_ch_up) and self._health.is_healthy('clickhouse')
        _comp = self._config.general.search.ranked_results_complement
        _merge_rail_types = frozenset(_comp.merge_explore_rails_query_types)
        _want_rail_merge = (
            _comp.enabled
            and _comp.merge_explore_rails
            and intent.query_type in _merge_rail_types
            and self._explore_rail_substrate_ready(_ch_healthy)
        )
        # Defer explore-rail work until after hybrid retrieve. Prefiring 6+ CH
        # rail queries during gather contended with SqlRetriever (~1.5s) and was
        # discarded on the common ``vector_has_enough`` path. Start rails only
        # when complement-merge actually needs them (short primary) or when ZRG
        # later requests a prewarm for empty results.
        _prewarm_explore_task: Optional[asyncio.Task] = None
        # Hybrid-first retrieve for every QI type that would otherwise own a CH path.
        # Original ``intent`` stays on the response (classified_intent unchanged).
        retrieve_intent = intent
        if _comp.enabled and intent.query_type in _CH_DEGRADE_QUERY_TYPES:
            # Concept-empty complement: keep temporal so period cues still scope
            # listing examples. When a listing concept exists, CH owns the window.
            _has_concept = _has_listing_concept(intent.slices)
            _strip_temporal = (not _ch_up) or (
                bool(_comp.strip_temporal_when_clickhouse_available) and _has_concept
            )
            retrieve_intent = hybrid_retrieve_intent(intent, strip_temporal=_strip_temporal)
            # Keep entity-driven listing concept when present; restore rewrite
            # encode only for hybrid listing search (not CH archetype complements).
            if _should_restore_rewrite_encode(intent.query_type, has_listing_concept=_has_concept):
                retrieve_intent = self._apply_encode_from_rewrite(retrieve_intent, transform_result)
            logger.info(
                f"hybrid_first_retrieve request_id={rid} from_type={intent.query_type} "
                f"ch_up={_ch_up} ch_healthy={_ch_healthy} strip_temporal={_strip_temporal} "
                f"has_listing_concept={_has_concept} "
                f"encode_text={retrieve_intent.semantic_encode_text!r} "
                f"residual_kind={retrieve_intent.residual_kind}"
            )
        elif intent.query_type in _CH_DEGRADE_QUERY_TYPES and not _ch_up:
            retrieve_intent = hybrid_degrade_intent(intent)
            _has_concept = _has_listing_concept(retrieve_intent.slices)
            if _should_restore_rewrite_encode(intent.query_type, has_listing_concept=_has_concept):
                retrieve_intent = self._apply_encode_from_rewrite(retrieve_intent, transform_result)
            logger.info(
                f"ch_unavailable_hybrid_degrade request_id={rid} from_type={intent.query_type} "
                f"has_listing_concept={_has_concept} "
                f"encode_text={retrieve_intent.semantic_encode_text!r} "
                f"residual_kind={retrieve_intent.residual_kind}"
            )
        ranked = await self._retrieve_and_rank(retrieve_intent, top_k=top_k)
        # Explore rails complement — only when primary is short of the enough threshold.
        _enough_frac = float(_comp.merge_explore_rails_primary_enough_fraction)
        _enough_n = (
            max(1, int(math.ceil(float(top_k) * _enough_frac)))
            if top_k is not None
            else None
        )
        _vector_has_enough = _enough_n is not None and len(ranked.items) >= int(_enough_n)
        _defer_rails = bool(_comp.merge_explore_rails_only_when_primary_short)
        if (
            _want_rail_merge
            and (not _defer_rails or not _vector_has_enough)
            and self._should_start_explore_rails_prewarm(
                intent, ch_healthy=_ch_healthy, want_rail_merge=True,
            )
        ):
            _prewarm_explore_task = self._start_explore_rails_prewarm_task(
                intent=intent,
                request_id=rid,
                user_id=user_context.user_id if user_context else None,
                ch_healthy=_ch_healthy,
            )
        if _want_rail_merge and _prewarm_explore_task is not None:
            _rail_ranked: Optional[RankedResults] = None
            try:
                _rail_ranked = await _prewarm_explore_task
            except _GUARD_EXC as _rail_err:
                logger.warning(
                    f"explore_rails_prewarm_failed request_id={rid} "
                    f"error_type={type(_rail_err).__name__} error={_rail_err}"
                )
                _rail_ranked = None
            # Rail cards carry zero query-relevance signal (recency/popularity feeds,
            # composer.py always tags them contributing_sources=['sql']). Merging them
            # via RRF against an already-sufficient real result set lets a rail card's
            # rank-position artifact bump a genuinely relevant vector match out of the
            # top_k — pollution, not backfill. Only merge when the primary path came up
            # short of the enough threshold (_vector_has_enough computed pre-await).
            _may_merge_rails = (not _defer_rails) or (not _vector_has_enough)
            if _rail_ranked is not None and _rail_ranked.items and _may_merge_rails:
                _lists: List[Tuple[str, List[RankedItem]]] = []
                if ranked.items:
                    _lists.append(('hybrid', list(ranked.items)))
                _lists.append(('explore', list(_rail_ranked.items)))
                _merged = _rrf_fuse(_lists, k=int(_comp.merge_rrf_k))
                _limit = int(top_k) if top_k is not None else len(_merged)
                ranked = RankedResults(
                    request_id=rid,
                    items=_merged[:_limit],
                    total_candidates=max(ranked.total_candidates, _rail_ranked.total_candidates, len(_merged)),
                    fusion_latency_ms=ranked.fusion_latency_ms,
                    cache_hit=None,
                    multi_intent_envelope=ranked.multi_intent_envelope,
                    failure_mode='hybrid_explore_complement',
                    query_intent=intent,
                )
                logger.info(
                    f"hybrid_explore_complement_merge request_id={rid} "
                    f"query_type={intent.query_type} items={len(ranked.items)}"
                )
            elif _rail_ranked is not None and _rail_ranked.items and _vector_has_enough:
                logger.info(
                    f"hybrid_explore_complement_skipped request_id={rid} "
                    f"reason=vector_has_enough ranked_items={len(ranked.items)} top_k={top_k}"
                )
        logger.debug(
            f"retrieve_complete request_id={rid} items={len(ranked.items)} "
            f"candidates={ranked.total_candidates}"
        )
        logger.debug(f"fuse_complete request_id={rid} fusion_latency_ms={ranked.fusion_latency_ms:.1f}")
        if trace is not None:
            trace.add('retrieve', f"items={len(ranked.items)} candidates={ranked.total_candidates}")
            trace.add('fuse', f"fusion_latency_ms={ranked.fusion_latency_ms:.1f}")
        guard_outcome = self._noop_guard_outcome
        # Qdrant down + empty hybrid: stamp qdrant_unavailable and skip ZRG.
        # Otherwise ZRG relaxes filters and app reports inventory_empty / filter_relaxed.
        _vec_err = self._last_retrieve_errors.get('vector')
        _qdrant_down = isinstance(_vec_err, (QdrantUnavailableError, QdrantQueryError))
        if _qdrant_down and not ranked.items:
            ranked = RankedResults(
                request_id=rid,
                items=[],
                total_candidates=0,
                fusion_latency_ms=ranked.fusion_latency_ms,
                cache_hit=None,
                multi_intent_envelope=ranked.multi_intent_envelope,
                failure_mode='qdrant_unavailable',
                query_intent=intent,
            )
            logger.warning(
                f"retrieval_qdrant_unavailable request_id={rid} "
                f"error_type={type(_vec_err).__name__} error={_vec_err}"
            )
            self._release_explore_prewarm(rid, cancel=True)
            _prewarm_explore_task = None
            # Fall through to eRanker/egress with empty items + typed failure_mode.
        # Zero-Result Guard on the hybrid retrieve intent (includes analytics under hybrid-first).
        _guard_threshold = (
            1 if _intent_has_hard_numeric_constraint(retrieve_intent)
            else self._zero_result_guard._config.min_results_before_relax
        ) if self._zero_result_guard is not None else 0
        _original_filter_count = (
            len(ZeroResultGuard._collect_filter_names(retrieve_intent))  # noqa: SLF001
            if self._zero_result_guard is not None else 0
        )
        if (
            ranked.failure_mode != 'qdrant_unavailable'
            and self._zero_result_guard is not None
            and self._zero_result_guard.enabled
            and intent.query_type in _QUERY_TYPES_USING_ZERO_RESULT_GUARD
            and len(ranked.items) < _guard_threshold
            and (_original_filter_count > 0 or intent.query_type in ('explore', 'analytics', 'guidance'))
        ):
            _fb_timeout_s = float(self._config.general.search.explore_fallback_timeout_seconds)
            _zrg_intent = retrieve_intent
            guard_result = await self._zero_result_guard.run(
                intent=_zrg_intent,
                retrieve_fn=self._retrieve_and_rank,
                user_context=user_context,
                semantic_retrieve_fn=self._quick_semantic_retrieve,
                explore_fallback_timeout_s=_fb_timeout_s,
                explore_prewarm_task=_prewarm_explore_task,
            )
            guard_items = list(guard_result.results.items)
            reapply_gate = getattr(self._config.retrieval, 'guard_widen_reapply_hard_gate', True)
            if guard_result.outcome.fired and guard_result.outcome.ladder_step == 'widen_filters' and reapply_gate:
                _re_gated = _apply_hard_chip_gate(
                    guard_items,
                    _zrg_intent,
                    self._config.retrieval.enforce_categorical_gate_on_missing_payload,
                    keyword_match_mode=self._config.retrieval.structured.keyword_match_mode,
                )
                if _re_gated:
                    guard_items = _re_gated
                    logger.info(f"hard_chip_gate_post_guard request_id={rid} guard_items={len(list(guard_result.results.items))} re_gated={len(_re_gated)}")
            ranked = RankedResults(
                request_id=rid,
                items=guard_items,
                total_candidates=guard_result.results.total_candidates,
                fusion_latency_ms=guard_result.results.fusion_latency_ms,
                cache_hit=None,
                multi_intent_envelope=guard_result.results.multi_intent_envelope,
                failure_mode=guard_result.results.failure_mode,
                query_intent=guard_result.results.query_intent,
            )
            guard_outcome = guard_result.outcome
        # Nonempty ladder: vector-only semantic when hybrid (+ rails merge + guard) still empty.
        if (
            ranked.failure_mode != 'qdrant_unavailable'
            and _comp.enabled
            and _comp.ensure_nonempty
            and _comp.force_semantic_when_empty
            and not ranked.items
        ):
            _sem_top = int(_comp.force_semantic_top_k)
            _sem_items = await self._quick_semantic_retrieve(retrieve_intent, _sem_top)
            if _sem_items:
                _limit = int(top_k) if top_k is not None else len(_sem_items)
                ranked = RankedResults(
                    request_id=rid,
                    items=list(_sem_items)[:_limit],
                    total_candidates=len(_sem_items),
                    fusion_latency_ms=0.0,
                    cache_hit=None,
                    failure_mode='force_semantic_nonempty',
                    query_intent=intent,
                )
                logger.info(
                    f"ranked_results_force_semantic_nonempty request_id={rid} "
                    f"query_type={intent.query_type} items={len(ranked.items)}"
                )
        # SLD-level dedup: keep only the highest-scored item per SLD when the same
        # domain name (domain_name field) appears multiple times (e.g. duplicate index
        # entries, multi-slice merge collisions, or case-normalisation mismatches).
        # Items are already in score-descending order from RRF fusion, so first-seen wins.
        self._release_explore_prewarm(rid, cancel=True)
        ranked = _dedup_by_domain_name(ranked)
        # Cache pre-eRanker fused results (cache stays user-agnostic).
        # Zero-result-guard outputs are NOT cached (query-specific relaxations).
        if _use_exact_cache and ranked.items and not guard_outcome.fired and not _is_fallback_intent:
            self._exact_cache.put(normalized, ranked, intent=intent)
        # explore-fallback skips eRanker by default; set apply_eranker_on_explore_fallback=true
        # to enable eRanker on fallback results.
        _fallback_fired = guard_outcome.fired and guard_outcome.ladder_step == 'explore_fallback'
        _skip_eranker_on_fallback = _fallback_fired and not self._config.explore.zero_result_guard.apply_eranker_on_explore_fallback
        if _skip_eranker_on_fallback:
            eranked = ranked
            outcome = self._explore_fallback_eranker_outcome
        else:
            eranked, outcome = await self._apply_eranker(rid, intent, ranked, user_context)
        self._last_eranker_outcome = outcome
        logger.debug(
            f"erank_complete request_id={rid} applied={outcome.applied} "
            f"client={outcome.client} skipped_reason={outcome.skipped_reason or 'none'}"
        )
        if trace is not None:
            trace.add('erank', f"applied={'true' if outcome.applied else 'false'} client={outcome.client} skipped_reason={outcome.skipped_reason or 'none'}")
        self._maybe_record_history(user_context, intent, eranked)
        diversified = await self._apply_diversifier(eranked, intent.normalized_query, lambda_override=diversity_lambda)
        logger.debug(
            f"diversify_complete request_id={rid} input_items={len(eranked.items)} "
            f"output_items={len(diversified.items)}"
        )
        if trace is not None:
            trace.add('diversify', f"input_items={len(eranked.items)} output_items={len(diversified.items)}")
        # Soft theme/keyword boost before truncate so theme matches can surface
        # from the deeper post-diversify pool (config: qi.entity_slots.soft_apply_mode).
        ranked_for_truncate = diversified
        if _soft_kw_entities or _kw_entries:
            ranked_for_truncate = _soft_applier.apply_rank_boost(
                diversified, _soft_kw_entities, _kw_entries
            )
            if self._config.measurement.ranking_stage_attribution.enabled:
                self._stage_scratch['soft_boost_applied'] = True
        truncated = self._truncate(ranked_for_truncate)
        if top_k is not None and top_k > 0 and len(truncated.items) > top_k:
            truncated = dataclasses.replace(truncated, items=list(truncated.items[:top_k]))
        logger.debug(
            f"truncate_complete request_id={rid} input_items={len(ranked_for_truncate.items)} "
            f"output_items={len(truncated.items)} max_results={self._config.general.max_results}"
        )
        if trace is not None:
            trace.add('truncate', f"input_items={len(ranked_for_truncate.items)} output_items={len(truncated.items)} max_results={self._config.general.max_results}")
        # Egress guard is the FINAL stage before we return to the API.
        # Runs after ``_truncate`` so the gate's per-item cost is bounded
        # by ``max_results``, not by the (potentially much larger)
        # un-truncated candidate set. Observation + log capture the post-gate
        # surface so SRE drilldowns see exactly what the user saw.
        truncated = await self._apply_egress_guard(truncated)
        truncated = RankedResults(
            request_id=truncated.request_id,
            items=truncated.items,
            total_candidates=truncated.total_candidates,
            fusion_latency_ms=truncated.fusion_latency_ms,
            cache_hit=truncated.cache_hit,
            multi_intent_envelope=truncated.multi_intent_envelope,
            failure_mode=truncated.failure_mode,
            query_intent=intent,
        )
        truncated = await self._with_guidance_envelope(intent, truncated, snapshot_task=_guidance_snapshot_task)
        if self._egress_guard is not None:
            outcome_obj = self._last_egress_outcome
            if outcome_obj is not None:
                # Total "modified" = pii_masked + explanation_scrubbed +
                # moderation_masked (each item is at most ONE of these
                # because the gate's decision is single-action). Logged as
                # a single rolled-up counter so the trace stays compact.
                modified = int(outcome_obj.items_pii_masked) + int(outcome_obj.items_explanation_scrubbed) + int(outcome_obj.items_moderation_masked)
                logger.debug(
                    f"egress_guard_complete request_id={rid} items_in={outcome_obj.items_in} "
                    f"items_kept={outcome_obj.items_kept} items_dropped={outcome_obj.items_dropped} "
                    f"items_modified={modified}"
                )
                if trace is not None:
                    trace.add('egress_guard', f"items_in={outcome_obj.items_in} items_kept={outcome_obj.items_kept} items_dropped={outcome_obj.items_dropped} items_modified={modified}")
        total_ms = (time.monotonic() - t0) * 1000.0
        # Observation captured only on QI-routed (non-cache) paths.
        # Cache-hit-rate is computed independently from ``cache_stats()`` so
        # denominator truth for per-search rates stays "QI-routed searches",
        # not "all calls".
        self._record_observation(rid=rid, user_context=user_context, intent=intent, results=truncated, total_ms=total_ms, cache_hit=None, eranker_outcome=outcome)
        self._publish_ranking_stages(
            request_id=rid,
            cache_hit=None,
            fusion_latency_ms=float(truncated.fusion_latency_ms or 0.0),
            total_candidates=int(truncated.total_candidates),
            result_count=len(truncated.items),
            eranker_outcome=outcome,
            guard_outcome=guard_outcome,
        )
        logger.info(
            f"search_completed request_id={rid} query_type={intent.query_type} items={len(eranked.items)} "
            f"eranker_applied={outcome.applied} eranker_client={outcome.client} eranker_skip={outcome.skipped_reason or 'none'} "
            f"guard_fired={guard_outcome.fired} guard_step={guard_outcome.ladder_step} "
            f"total_latency_ms={total_ms:.1f}"
        )
        return truncated, outcome, guard_outcome

    async def _retrieve_and_rank_single(self, intent: QueryIntent, top_k: Optional[int] = None) -> RankedResults:
        """Gather + fuse one intent (no multi-intent slice fan-out)."""
        candidate_sets = await self._gather_candidates(intent, top_k=top_k)
        if not candidate_sets:
            # Fully degraded conversational fallback to FIND: empty RankedResults
            # with failure_mode='find_fallback' so the UI can render regex chips.
            if self._is_fully_degraded():
                return self._emit_find_fallback(intent)
            fused = RankedResults(request_id=intent.request_id, items=[], total_candidates=0, fusion_latency_ms=0.0, cache_hit=None)
        else:
            _m = self._config.retrieval.over_fetch_multiplier
            top_n_override = int(top_k * _m) if top_k is not None else None
            fused = self._fuser.fuse(request_id=intent.request_id, candidate_sets=candidate_sets, cache_hit=None, residual_kind=intent.residual_kind, top_n_override=top_n_override)
            gated_items = _apply_hard_chip_gate(
                fused.items,
                intent,
                self._config.retrieval.enforce_categorical_gate_on_missing_payload,
                keyword_match_mode=self._config.retrieval.structured.keyword_match_mode,
            )
            if self._config.measurement.ranking_stage_attribution.enabled:
                self._stage_scratch['hard_gate_before'] = len(fused.items)
                self._stage_scratch['hard_gate_after'] = len(gated_items)
            if len(gated_items) != len(fused.items):
                logger.info(f"hard_chip_gate request_id={intent.request_id} before={len(fused.items)} after={len(gated_items)} dropped={len(fused.items) - len(gated_items)}")
                fused = RankedResults(
                    request_id=fused.request_id,
                    items=gated_items,
                    total_candidates=fused.total_candidates,
                    fusion_latency_ms=fused.fusion_latency_ms,
                    cache_hit=fused.cache_hit,
                    multi_intent_envelope=fused.multi_intent_envelope,
                    failure_mode=fused.failure_mode,
                    query_intent=fused.query_intent,
                )
        return fused

    async def _retrieve_and_rank(self, intent: QueryIntent, top_k: Optional[int] = None) -> RankedResults:
        """Single-shot retrieval + fusion for a given intent (eRanker runs outside).

        Used both by the primary ``search()`` path and by the Zero-Result Guard
        (which calls it after each filter-relaxation step). Pure-compute on
        candidates returned from the gather; never raises on empty backends —
        an empty result is a valid outcome.

        Multi-intent:
        When the intent carries >1 slice each with a `slice_id`, dispatches to
        the multi-intent path: per-sub-intent pre-screen -> 5-cap weighted
        ranking -> parallel retrieval per surviving sub-intent -> RRF merge with
        cross-intent bonus -> sub_intent_ids stamped on each RankedItem.
        Single-intent intents use gather+fuse directly.
        """
        if self._is_multi_intent(intent):
            self._ensure_slice_ids(intent)
            fused = await self._retrieve_and_rank_multi(intent)
        else:
            fused = await self._retrieve_and_rank_single(intent, top_k=top_k)
        return self._apply_brandability_rerank(intent, self._apply_engagement_boost(self._apply_fuzzy_rerank(intent, fused)))

    def _apply_engagement_boost(self, fused: RankedResults) -> RankedResults:
        """Apply real-time engagement signal boost post-RRF (config-gated).

        Reads live Qdrant payload fields populated by EventIngestDriver and lifts
        fused_score by up to max_boost * normalised_engagement.  Always a no-op when
        engagement_boost is disabled or when no item carries any engagement payload.

        Signals used (all normalised to [0, 1] against their configured caps):
          bid_velocity_1h         — bids in the last hour (trending signal)
          watch_density_1d        — passive watchers today (audience signal)
          bidder_watch_density_1d — type-9 (bidder-intent) watchers (commitment)
          unique_bidder_count_4h  — distinct bidders in last 4h (competitive depth)
        """
        cfg = self._config.retrieval.engagement_boost
        if cfg is None or not cfg.enabled or not fused.items:
            return fused
        w_sum = (cfg.bid_velocity_weight + cfg.watch_density_weight
                 + cfg.bidder_watch_weight + cfg.unique_bidder_weight)
        boosted: List[RankedItem] = []
        for item in fused.items:
            p = item.payload or {}
            bv  = min(float(p.get('bid_velocity_1h', 0) or 0)        / cfg.bid_velocity_cap,  1.0)
            wd  = min(float(p.get('watch_density_1d', 0) or 0)        / cfg.watch_density_cap, 1.0)
            bwd = min(float(p.get('bidder_watch_density_1d', 0) or 0) / cfg.bidder_watch_cap,  1.0)
            ub  = min(float(p.get('unique_bidder_count_4h', 0) or 0)  / cfg.unique_bidder_cap, 1.0)
            engagement = (
                cfg.bid_velocity_weight  * bv +
                cfg.watch_density_weight * wd +
                cfg.bidder_watch_weight  * bwd +
                cfg.unique_bidder_weight * ub
            ) / w_sum
            boosted_score = item.fused_score * (1.0 + cfg.max_boost * engagement)
            boosted.append(dataclasses.replace(item, fused_score=boosted_score))
        boosted.sort(key=lambda it: it.fused_score, reverse=True)
        return dataclasses.replace(fused, items=boosted)

    def _apply_brandability_rerank(self, intent: QueryIntent, fused: RankedResults) -> RankedResults:
        """Reorder the pool by brandability score (config-gated).

        Fires when brandability is enabled and either the query matches a
        trigger term/regex ('brandable', 'memorableable', ...), or residual is
        empty/navigational with a hard chip (filter-only browse quality).

        When intent.residual_kind is in cfg.pure_sort_residual_kinds, items are sorted
        purely by brandability score. Otherwise items are sorted by
        ``fused_score * (1 + boost_weight * brandability(sld))``.

        :param intent: QueryIntent - Carries ``normalized_query`` and ``residual_kind``
        :param fused: RankedResults - Pool to reorder
        :return: RankedResults - Reordered pool; identical when disabled / not triggered
        """
        cfg = self._config.retrieval.brandability
        if cfg is None or not cfg.enabled or not fused.items:
            return fused
        nq = (intent.normalized_query or '').lower()
        term_hit = any(t in nq for t in cfg.trigger_terms)
        regex_hit = (not term_hit and cfg.trigger_regex is not None and bool(re.search(cfg.trigger_regex, nq, re.IGNORECASE | re.VERBOSE)))
        # Filter-only / concept-empty browse: demote digit-heavy junk via
        # brandability even without "brandable" wording (includes CH complements
        # that keep residual empty after skipping rewrite encode).
        empty_browse = (
            (intent.residual_kind or '') in ('empty', 'navigational')
            and _intent_has_hard_chip(intent)
        )
        concept_empty_browse = (
            not empty_browse
            and _intent_has_hard_chip(intent)
            and not _has_listing_concept(getattr(intent, 'slices', None))
            and not (intent.semantic_encode_text or '').strip()
        )
        if not term_hit and not regex_hit and not empty_browse and not concept_empty_browse:
            return fused
        scorer = BrandabilityScorer(cfg)
        _pure = bool(cfg.pure_sort_residual_kinds) and (intent.residual_kind or '') in cfg.pure_sort_residual_kinds

        def _key_pure(it: RankedItem) -> float:
            return scorer.score(str((it.payload or {}).get('sld', it.item_id)).lower())

        def _key_boosted(it: RankedItem) -> float:
            sld = str((it.payload or {}).get('sld', it.item_id)).lower()
            return float(it.fused_score) * (1.0 + cfg.boost_weight * scorer.score(sld))

        new_items = sorted(fused.items, key=_key_pure if _pure else _key_boosted, reverse=True)
        logger.info(f"brandability_rerank request_id={intent.request_id} items={len(new_items)} pure_sort={_pure}")
        return RankedResults(
            request_id=fused.request_id,
            items=new_items,
            total_candidates=fused.total_candidates,
            fusion_latency_ms=fused.fusion_latency_ms,
            cache_hit=fused.cache_hit,
            multi_intent_envelope=fused.multi_intent_envelope,
            failure_mode=fused.failure_mode,
            query_intent=fused.query_intent,
        )

    def _apply_fuzzy_rerank(self, intent: QueryIntent, fused: RankedResults) -> RankedResults:
        """Reorder the fused pool to surface SLD typo / near-matches (config-gated).

        No-op when the reranker is disabled or the pool is empty. Reorders the first
        ``max_candidates`` items by ``fused_score + boost_weight * fuzzy_coverage`` and
        concatenates the untouched tail. Pure-compute.

        The reranker's boosted score is written back into each reranked item's
        ``fused_score`` (via ``dataclasses.replace``) so the boost SURVIVES the
        downstream ``_apply_engagement_boost`` / ``_apply_brandability_rerank``
        stages, both of which re-sort by ``fused_score``. Without the write-back
        the fuzzy reorder is silently discarded by the next fused-score sort.

        :param intent: QueryIntent - Carries ``normalized_query`` used for tokenization
        :param fused: RankedResults - Post-fusion (post-gate) pool to reorder
        :return: RankedResults - Same items, near-matches boosted; identical when disabled
        """
        if self._fuzzy_reranker is None or not fused.items:
            return fused
        n = min(len(fused.items), self._fuzzy_rerank_max_candidates)
        reranked = self._fuzzy_reranker.rerank(intent.normalized_query, fused.items, n)
        new_items = [dataclasses.replace(r.item, fused_score=r.rerank_score) for r in reranked] + list(fused.items[n:])
        return RankedResults(
            request_id=fused.request_id,
            items=new_items,
            total_candidates=fused.total_candidates,
            fusion_latency_ms=fused.fusion_latency_ms,
            cache_hit=fused.cache_hit,
            multi_intent_envelope=fused.multi_intent_envelope,
            failure_mode=fused.failure_mode,
            query_intent=fused.query_intent,
        )

    def _is_fully_degraded(self) -> bool:
        """True iff every retrieval backend (vector + structured + sql) is
        unhealthy. Used by ``_retrieve_and_rank`` to detect the worst-case
        path and route to the FIND-only conversational fallback (Wave 9
        Gap 14). Pure-read against ``BackendHealthRegistry``; no side
        effects.
        """
        retrieval_backends = ('vector', 'structured', 'sql')
        return all(not self._health.is_healthy(b) for b in retrieval_backends)

    def _emit_find_fallback(self, intent: QueryIntent) -> RankedResults:
        """Build the FIND-fallback ``RankedResults`` envelope (Wave 9 Gap 14).

        Carries the regex-extracted entities (chips) from the QI verdict
        as a single FIND-shaped payload item so the conversational UI can
        re-render them as Advanced-surface filter controls. The ``items``
        list is empty (degraded path produces no candidates) and
        ``failure_mode='find_fallback'`` is set so consumers can branch
        on the typed envelope rather than sniffing for empty results.

        Pure-compute; never raises. Logs a single WARNING with the
        intent's ``request_id`` for ops visibility.
        """
        find_filters: List[Dict[str, Any]] = []
        for slc in intent.slices:
            for ent in slc.entities:
                find_filters.append({
                    'name': ent.name,
                    'value': ent.value,
                    'confidence': float(ent.confidence),
                    'source': ent.source,
                    'chip_kind': ent.chip_kind,
                    'slice_id': slc.slice_id,
                })
        logger.warning(f"search_find_fallback request_id={intent.request_id} query_type={intent.query_type} chip_count={len(find_filters)} reason=all_retrieval_backends_unhealthy")
        return RankedResults(request_id=intent.request_id, items=[], total_candidates=0, fusion_latency_ms=0.0, cache_hit=None, failure_mode='find_fallback')

    async def _apply_diversifier(self, ranked: RankedResults, query_text: str, lambda_override: Optional[float] = None) -> RankedResults:
        """Re-shuffle the head for diversity.

        Latency-gated: the diversifier call runs in ``asyncio.to_thread`` (sync
        ``Diversifier`` contract; offloading keeps the event loop free) wrapped in
        ``asyncio.wait_for`` with the configured budget. On timeout /
        ``DiversityError`` / any unexpected ``Exception`` the input
        ``RankedResults`` is returned unchanged — the user-facing path is
        never blocked by diversifier degradation.

        Slot in the pipeline: AFTER eRanker, BEFORE ``_truncate``. Items at
        positions ``> top_n`` keep their post-eRanker order; the diversified head
        ``output_n`` is followed by the un-selected head remainder (in input order)
        and then the un-touched tail.

        :param ranked: RankedResults - The post-eRanker ranked list
        :param query_text: str - Normalized query (``QueryIntent.normalized_query``)
        :param lambda_override: Optional[float] - Per-request MMR lambda; uses
            config value when None
        :return: RankedResults - Diversified head + un-selected head
            remainder + un-touched tail, OR the input on the disabled /
            empty / timeout / error path
        """
        _attr_on = self._config.measurement.ranking_stage_attribution.enabled
        if not self._diversifier_enabled:
            if _attr_on:
                self._stage_scratch['diversify_applied'] = False
                self._stage_scratch['diversify_skipped_reason'] = 'disabled'
            return ranked
        if not ranked.items:
            if _attr_on:
                self._stage_scratch['diversify_applied'] = False
                self._stage_scratch['diversify_skipped_reason'] = 'empty_results'
            return ranked
        head_n = min(self._diversifier_top_n, len(ranked.items))
        if head_n <= 0:
            if _attr_on:
                self._stage_scratch['diversify_applied'] = False
                self._stage_scratch['diversify_skipped_reason'] = 'empty_head'
            return ranked
        out_n = min(self._diversifier_output_n, head_n)
        if out_n <= 0:
            if _attr_on:
                self._stage_scratch['diversify_applied'] = False
                self._stage_scratch['diversify_skipped_reason'] = 'empty_output'
            return ranked
        t0 = time.monotonic()
        try:
            selection = await asyncio.wait_for(asyncio.to_thread(self._diversifier.diversify, query_text or '', ranked.items, head_n, out_n, lambda_override), timeout=self._diversifier_budget_s)
        except asyncio.TimeoutError:
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            if _attr_on:
                self._stage_scratch['diversify_applied'] = False
                self._stage_scratch['diversify_skipped_reason'] = 'timeout'
                self._stage_scratch['diversify_latency_ms'] = round(elapsed_ms, 3)
            logger.warning(
                f"diversifier_timeout request_id={ranked.request_id} backend={self._diversifier.name} "
                f"top_n={head_n} output_n={out_n} budget_ms={self._diversifier_budget_s * 1000.0:.1f} "
                f"elapsed_ms={elapsed_ms:.1f}"
            )
            return ranked
        except DiversityError as e:
            if _attr_on:
                self._stage_scratch['diversify_applied'] = False
                self._stage_scratch['diversify_skipped_reason'] = 'misconfigured'
            logger.warning(f"diversifier_misconfigured request_id={ranked.request_id} backend={self._diversifier.name} top_n={head_n} output_n={out_n} error={str(e)}")
            return ranked
        except _GUARD_EXC as e:
            if _attr_on:
                self._stage_scratch['diversify_applied'] = False
                self._stage_scratch['diversify_skipped_reason'] = 'error'
            logger.warning(f"diversifier_failed request_id={ranked.request_id} backend={self._diversifier.name} top_n={head_n} output_n={out_n} error_type={type(e).__name__} error={str(e)}")
            return ranked
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        if not selection:
            if _attr_on:
                self._stage_scratch['diversify_applied'] = False
                self._stage_scratch['diversify_skipped_reason'] = 'empty_selection'
                self._stage_scratch['diversify_latency_ms'] = round(elapsed_ms, 3)
            logger.warning(f"diversifier_returned_empty request_id={ranked.request_id} backend={self._diversifier.name} top_n={head_n} output_n={out_n} elapsed_ms={elapsed_ms:.1f}")
            return ranked
        if len(selection) != out_n:
            if _attr_on:
                self._stage_scratch['diversify_applied'] = False
                self._stage_scratch['diversify_skipped_reason'] = 'size_mismatch'
                self._stage_scratch['diversify_latency_ms'] = round(elapsed_ms, 3)
            logger.warning(f"diversifier_size_mismatch request_id={ranked.request_id} backend={self._diversifier.name} expected={out_n} actual={len(selection)} elapsed_ms={elapsed_ms:.1f}")
            return ranked
        # Re-stitch: selected items in selection order, then head remainder
        # (head items not selected, in their input order), then untouched tail.
        selected_items = [d.item for d in selection]
        selected_ids = {d.item.item_id for d in selection}
        head_slice = ranked.items[:head_n]
        head_remainder = [it for it in head_slice if it.item_id not in selected_ids]
        tail = list(ranked.items[head_n:])
        new_items = selected_items + head_remainder + tail
        # Defence-in-depth: re-stitch length must equal input length.
        if len(new_items) != len(ranked.items):
            if _attr_on:
                self._stage_scratch['diversify_applied'] = False
                self._stage_scratch['diversify_skipped_reason'] = 'restitch_mismatch'
                self._stage_scratch['diversify_latency_ms'] = round(elapsed_ms, 3)
            logger.warning(f"diversifier_restitch_length_mismatch request_id={ranked.request_id} backend={self._diversifier.name} expected={len(ranked.items)} actual={len(new_items)}")
            return ranked
        if _attr_on:
            self._stage_scratch['diversify_applied'] = True
            self._stage_scratch['diversify_skipped_reason'] = None
            self._stage_scratch['diversify_latency_ms'] = round(elapsed_ms, 3)
        logger.info(f"diversifier_applied request_id={ranked.request_id} backend={self._diversifier.name} top_n={head_n} output_n={out_n} tail_n={len(tail)} elapsed_ms={elapsed_ms:.1f}")
        return RankedResults(
            request_id=ranked.request_id,
            items=new_items,
            total_candidates=ranked.total_candidates,
            fusion_latency_ms=ranked.fusion_latency_ms,
            cache_hit=ranked.cache_hit,
            multi_intent_envelope=ranked.multi_intent_envelope,
            failure_mode=ranked.failure_mode,
            query_intent=ranked.query_intent,
        )

    @staticmethod
    def _is_multi_intent(intent: QueryIntent) -> bool:
        """A multi-intent intent carries >1 slice. slice_id is guaranteed at
        dispatch by ``_ensure_slice_ids`` — the predicate no longer requires it.

        Requiring a non-empty slice_id here silently demoted genuine multi-slice
        intents to the single path whenever any slice lacked an id (e.g.
        ``_classify_single`` yields >1 slice without minting ids). On the single
        path ``extract_filters_from_intent`` collapses every slice into one filter
        dict (last-slice-wins), dropping earlier slices' filters.
        """
        if intent is None or len(intent.slices) <= 1:
            return False
        return True

    @staticmethod
    def _ensure_slice_ids(intent: QueryIntent) -> None:
        """Fill any empty ``slice_id`` so per-slice RRF merge keys never collide.

        Idempotent: only empty ids are minted, so re-invoking on a cached intent
        is a no-op. Mutates in place (``IntentSlice`` is a non-frozen dataclass).
        """
        for s in intent.slices:
            if not s.slice_id:
                s.slice_id = IntentSlice.new_slice_id()

    async def _retrieve_and_rank_multi(self, intent: QueryIntent) -> RankedResults:
        """Multi-intent retrieval path.

        Steps:
          1. Pre-screen: per slice, run the structured retriever to estimate
             ``expected_results``. Drop slices with 0 expected when
             ``multi_intent.drop_zero_expected`` is true.
          2. 5-cap weighted ranking via ``rank_sub_intents`` keeps top-N by
             ``0.5 * confidence + 0.3 * expected_results_norm + 0.2 * specificity``.
          3. Fan retrieval per surviving slice in parallel via ``gather``.
          4. Cross-slice merge dispatched on ``multi_intent.merge_strategy``:
             - ``"rrf"`` (default): standard Reciprocal Rank Fusion across
               slice rankings (``sum_over_slices(1 / (rrf_k + rank_in_slice))``).
               Items appearing in multiple slices accumulate higher fused_score
               naturally; rank-driven so it tolerates fused_score scale drift
               between slices.
             - ``"max_score"``: legacy max-fused-score with a one-shot
               ``cross_intent_bonus`` multiplier for items appearing in 2+
               slices.
          5. Each ``RankedItem.sub_intent_ids`` carries the slice_ids it
             matched so downstream UI can render per-intent badges.

        Returns a single ``RankedResults`` envelope identical in shape to
        the single-intent path so downstream code (cache, eRanker,
        diversifier) does not branch on multi-intent.
        """
        cfg = self._config.multi_intent
        # Step 1: pre-screen counts (parallel).
        prescreen_tasks = [self._prescreen_structured_count_for_slice(s) for s in intent.slices]
        counts = await asyncio.gather(*prescreen_tasks, return_exceptions=True)
        candidates: List[tuple] = []
        kept_slices: List[IntentSlice] = []
        _prescreen_reliable = bool(
            getattr(self._structured, 'provides_inventory_estimate', True)
        )
        _exempt_qtypes = frozenset(cfg.drop_zero_expected_exempt_query_types)
        for slc, count in zip(intent.slices, counts):
            if isinstance(count, BaseException):
                logger.warning(f"multi_intent_prescreen_failed request_id={intent.request_id} slice_id={slc.slice_id} error={str(count)}")
                expected = 0
            else:
                expected = int(count)
            slice_has_hard_filters = any(getattr(e, 'chip_kind', 'soft') == 'hard' for e in slc.entities)
            requires_hard = bool(cfg.drop_zero_expected_requires_hard_filters)
            _skip_unreliable = (
                bool(cfg.drop_zero_expected_skip_unreliable_prescreen)
                and not _prescreen_reliable
            )
            _exempt_type = slc.query_type in _exempt_qtypes
            if (
                cfg.drop_zero_expected
                and expected == 0
                and (not requires_hard or slice_has_hard_filters)
                and not _skip_unreliable
                and not _exempt_type
            ):
                logger.info(
                    f"multi_intent_dropped_zero_results request_id={intent.request_id} "
                    f"slice_id={slc.slice_id} query_type={slc.query_type} "
                    f"has_hard_filters={slice_has_hard_filters}"
                )
                continue
            if expected == 0 and (_skip_unreliable or _exempt_type):
                logger.info(
                    f"multi_intent_zero_expected_kept request_id={intent.request_id} "
                    f"slice_id={slc.slice_id} query_type={slc.query_type} "
                    f"prescreen_reliable={_prescreen_reliable} exempt_type={_exempt_type}"
                )
            candidates.append((slc.slice_id, float(slc.confidence), expected, len(slc.entities)))
            kept_slices.append(slc)

        if not kept_slices:
            logger.warning(
                f"multi_intent_all_slices_dropped request_id={intent.request_id} "
                f"reason=zero_expected_results "
                f"fallback_single={bool(cfg.all_slices_dropped_fallback_to_single)}"
            )
            if bool(cfg.all_slices_dropped_fallback_to_single):
                return await self._retrieve_and_rank_single(intent, top_k=None)
            return RankedResults(request_id=intent.request_id, items=[], total_candidates=0, fusion_latency_ms=0.0, cache_hit=None)

        # Step 2: 5-cap weighted ranking.
        ranked = rank_sub_intents(candidates, cfg)
        ranked_slice_ids = {row[0] for row in ranked}
        surviving_slices = [s for s in kept_slices if s.slice_id in ranked_slice_ids]
        if len(surviving_slices) < len(kept_slices):
            dropped = [s.slice_id for s in kept_slices if s.slice_id not in ranked_slice_ids]
            logger.info(f"multi_intent_5cap_dropped request_id={intent.request_id} kept={len(surviving_slices)} dropped_slice_ids={dropped}")

        # Collect the sub-queries the 5-cap dropped (NOT the
        # zero-expected drops; those carry no useful signal to the user). The
        # surviving slices are the ones we actually retrieved against; the
        # cap-dropped ones become the overflow chip below.
        cap_dropped_sub_queries: List[str] = [
            (s.raw_text or '').strip()
            for s in kept_slices
            if s.slice_id not in ranked_slice_ids and (s.raw_text or '').strip()
        ]

        # Step 3: per-slice retrieval + fusion, bounded concurrency.
        # Each slice runs a full vector+structured+sql fan-out; firing every
        # surviving slice at once (3 comparators -> 9 backend calls) saturates the
        # shared Qdrant HNSW + ClickHouse pool and inflates wall-clock past the
        # search SLA. A semaphore caps in-flight slices; the rest queue.
        _slice_sema = asyncio.Semaphore(max(1, int(cfg.max_concurrent_slices)))

        async def _retrieve_slice_bounded(s: IntentSlice) -> RankedResults:
            async with _slice_sema:
                return await self._retrieve_and_fuse_slice(intent, s)

        per_slice_tasks = [_retrieve_slice_bounded(s) for s in surviving_slices]
        per_slice_results = await asyncio.gather(*per_slice_tasks, return_exceptions=True)

        # Step 4: cross-slice merge.
        # Two strategies, dispatched by ``multi_intent.merge_strategy``:
        #
        #   "rrf" (default) — treat each surviving slice's per-slice ranking
        #       as an input list and accumulate scores via the standard RRF
        #       formula 1/(k + rank_in_slice). Items appearing in multiple
        #       slices naturally accumulate higher fused_score without a
        #       per-merge tuning knob; rank-driven so it tolerates
        #       fused_score scale drift across slices.
        #
        #   "max_score" (legacy) — take the max per-slice fused_score and
        #       multiply once by ``cross_intent_bonus`` for items appearing
        #       in 2+ slices. Kept for back-compat and for callers that need
        #       the score-scale carried through unmodified.
        total_candidates = 0
        total_fusion_latency_ms = 0.0
        per_slice_result_counts: Dict[str, int] = {}
        # First pass: bookkeeping used by both strategies.
        successful_slice_results: List[Tuple[IntentSlice, RankedResults]] = []
        for slc, result in zip(surviving_slices, per_slice_results):
            if isinstance(result, BaseException):
                logger.warning(f"multi_intent_slice_retrieval_failed request_id={intent.request_id} slice_id={slc.slice_id} error={str(result)}")
                per_slice_result_counts[slc.slice_id] = 0
                continue
            sub_results: RankedResults = result
            total_candidates += sub_results.total_candidates
            total_fusion_latency_ms += sub_results.fusion_latency_ms
            per_slice_result_counts[slc.slice_id] = len(sub_results.items)
            successful_slice_results.append((slc, sub_results))

        # Joint transformed-query retrieve alongside per-keyword legs when
        # enough L0 keywords pass the probability gate. Outside the 5-cap.
        _kw_min = min_probability_fraction(self._config.qi.l0_llm_entity)
        _joint_kw = filter_keywords(intent.keywords, _kw_min)
        if (
            len(_joint_kw) >= int(cfg.split_on_l0_keywords_min_terms)
            and bool(cfg.split_on_l0_keywords)
        ):
            joint_slice, joint_wrapper = self._build_joint_transformed_intent(intent, _joint_kw)
            try:
                async with _slice_sema:
                    joint_ranked = await self._retrieve_and_rank_single(joint_wrapper, top_k=None)
                total_candidates += joint_ranked.total_candidates
                total_fusion_latency_ms += joint_ranked.fusion_latency_ms
                per_slice_result_counts[joint_slice.slice_id] = len(joint_ranked.items)
                successful_slice_results.append((joint_slice, joint_ranked))
                logger.info(
                    f"multi_intent_joint_transformed_retrieve request_id={intent.request_id} "
                    f"kw_terms_n={len(_joint_kw)} items={len(joint_ranked.items)} "
                    f"encode_len={len(str(joint_wrapper.semantic_encode_text or ''))}"
                )
            except (ValidationError, RetrievalError, QdrantUnavailableError, QdrantQueryError) as e:
                logger.warning(
                    f"multi_intent_joint_transformed_failed request_id={intent.request_id} "
                    f"error_type={type(e).__name__} error={str(e)}"
                )

        if cfg.merge_strategy == 'rrf':
            final_items = self._merge_slices_rrf(intent.request_id, successful_slice_results, cfg)
        else:
            final_items = self._merge_slices_max_score(intent.request_id, successful_slice_results, cfg)

        # Server-side per-intent chip strip envelope.
        # Built once per multi-intent search (only when surviving_slices is non-empty);
        # cache-hit paths leave RankedResults.multi_intent_envelope=None because the
        # cache is user-agnostic and per-intent counts would be stale.
        envelope = self._build_multi_intent_envelope(surviving_slices=surviving_slices, per_slice_result_counts=per_slice_result_counts, cap_dropped_sub_queries=cap_dropped_sub_queries)
        merged = RankedResults(
            request_id=intent.request_id,
            items=final_items,
            total_candidates=total_candidates,
            fusion_latency_ms=total_fusion_latency_ms,
            cache_hit=None,
            multi_intent_envelope=envelope,
        )
        return merged

    def _merge_slices_rrf(self, request_id: str, slice_results: List[Tuple[IntentSlice, RankedResults]], cfg: 'MultiIntentConfig') -> List[RankedItem]:
        """Multi-intent merge via Reciprocal Rank Fusion across slice rankings.

        Treats each slice's per-slice ranked list as an RRF input list. An
        item that appears in multiple slice rankings accumulates score from
        every list it participates in via the standard RRF contribution
        ``1 / (rrf_k + rank_in_slice)`` (rank is 1-indexed). Contributing
        sources are unioned across slices; ``sub_intent_ids`` records the
        slices the item matched. ``sub_intent_match_kinds`` mirrors
        ``RankedItemBuilder`` / max-score merge: per-slice hard vs soft with
        hard winning on conflict (Layer 5 badge contract).

        No additional ``cross_intent_bonus``
        multiplier is applied because the formula is already additive
        across slices — items hit by N slices accumulate N contributions
        and naturally outrank single-slice items at comparable per-slice
        ranks.

        :param request_id: str - Correlation id (used in the log line)
        :param slice_results: List[Tuple[IntentSlice, RankedResults]] - One
            entry per slice that produced results without raising
        :param cfg: MultiIntentConfig - Carries ``merge_rrf_k``
        :return: List[RankedItem] - Sorted desc by fused_score; ties broken
            on item_id for determinism
        """
        k = float(cfg.merge_rrf_k)
        scores: Dict[str, float] = {}
        sources_per_item: Dict[str, List[str]] = {}
        payload_per_item: Dict[str, Dict] = {}
        slice_ids_per_item: Dict[str, List[str]] = {}
        match_kinds_per_item: Dict[str, Dict[str, str]] = {}
        for slc, sub_results in slice_results:
            for rank_idx, item in enumerate(sub_results.items):
                contribution = 1.0 / (k + float(rank_idx + 1))
                scores[item.item_id] = scores.get(item.item_id, 0.0) + contribution
                bucket = sources_per_item.setdefault(item.item_id, [])
                for src in item.contributing_sources:
                    if src not in bucket:
                        bucket.append(src)
                payload_bucket = payload_per_item.setdefault(item.item_id, {})
                for key, val in (item.payload or {}).items():
                    if key not in payload_bucket:
                        payload_bucket[key] = val
                slice_bucket = slice_ids_per_item.setdefault(item.item_id, [])
                if slc.slice_id not in slice_bucket:
                    slice_bucket.append(slc.slice_id)
                mk_bucket = match_kinds_per_item.setdefault(item.item_id, {})
                contrib = _rrf_contribution_match_kind_for_slice(item, slc.slice_id)
                _rrf_accumulate_slice_match_kind(mk_bucket, slc.slice_id, contrib)

        ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        final_items: List[RankedItem] = []
        for item_id, fused_score in ordered:
            slice_ids = slice_ids_per_item.get(item_id, [])
            slice_set = set(slice_ids)
            raw_mk = match_kinds_per_item.get(item_id, {})
            kinds = {k: v for k, v in raw_mk.items() if k in slice_set}
            for sid in slice_ids:
                if sid not in kinds:
                    kinds[sid] = 'hard'
            final_items.append(RankedItem(
                item_id=item_id,
                fused_score=float(fused_score),
                contributing_sources=list(sources_per_item.get(item_id, [])),
                payload=payload_per_item.get(item_id, {}),
                sub_intent_ids=list(slice_ids),
                sub_intent_match_kinds=kinds,
            ))
        if final_items:
            multi_match = sum(1 for it in final_items if len(it.sub_intent_ids) > 1)
            if multi_match > 0:
                logger.info(f"multi_intent_rrf_merge request_id={request_id} items_with_multi_match={multi_match} total_items={len(final_items)} k={int(k)}")
        return final_items

    def _merge_slices_max_score(self, request_id: str, slice_results: List[Tuple[IntentSlice, RankedResults]], cfg: 'MultiIntentConfig') -> List[RankedItem]:
        """Legacy max-score merge with cross-intent bonus.

        Takes the maximum per-slice fused_score for each item and applies
        ``cross_intent_bonus`` once to items that appeared in 2+ slices.
        Preserved for back-compat; ``rrf`` is the recommended strategy.

        :param request_id: str - Correlation id (used in the log line)
        :param slice_results: List[Tuple[IntentSlice, RankedResults]] - One
            entry per slice that produced results without raising
        :param cfg: MultiIntentConfig - Carries ``cross_intent_bonus``
        :return: List[RankedItem] - Sorted desc by fused_score; ties broken
            on item_id for determinism
        """
        merged: Dict[str, RankedItemBuilder] = {}
        for slc, sub_results in slice_results:
            for item in sub_results.items:
                builder = merged.get(item.item_id)
                if builder is None:
                    builder = RankedItemBuilder.from_item(item)
                    merged[item.item_id] = builder
                else:
                    builder.merge_from_item(item)
                builder.add_slice(slc.slice_id)

        bonus = float(cfg.cross_intent_bonus)
        final_items: List[RankedItem] = [b.build(cross_intent_bonus=bonus) for b in merged.values()]
        final_items.sort(key=lambda it: (-it.fused_score, it.item_id))
        if final_items:
            multi_match = sum(1 for it in final_items if len(it.sub_intent_ids) > 1)
            if multi_match > 0:
                logger.info(f"multi_intent_max_score_merge request_id={request_id} items_with_multi_match={multi_match} total_items={len(final_items)} cross_intent_bonus={bonus}")
        return final_items

    def _build_multi_intent_envelope(self, surviving_slices: List[IntentSlice], per_slice_result_counts: Dict[str, int], cap_dropped_sub_queries: List[str]) -> Optional[MultiIntentChipStrip]:
        """Build the chip-strip envelope from the multi-intent fan-out.

        :param surviving_slices: List[IntentSlice] - Slices that survived the
            zero-expected drop + 5-cap (matches what was retrieved against)
        :param per_slice_result_counts: Dict[str, int] - slice_id -> retrieved
            item count (post-RRF per-slice fuse, pre-cross-intent merge)
        :param cap_dropped_sub_queries: List[str] - Sub-query texts the 5-cap
            dropped (zero-expected drops are excluded — they carry no signal)
        :return: Optional[MultiIntentChipStrip] - None when nothing survived
            (caller will return a no-envelope RankedResults); otherwise a fully
            populated strip with optional overflow chip.
        """
        if not surviving_slices:
            return None
        groups: List[IntentChipGroup] = []
        for slc in surviving_slices:
            groups.append(IntentChipGroup(
                slice_id=slc.slice_id,
                title=build_card_title(slc),
                chips=chip_labels_for_slice(slc),
                count=int(per_slice_result_counts.get(slc.slice_id, 0)),
                raw_text=slc.raw_text or '',
            ))
        overflow: Optional[OverflowChip] = None
        dropped_count = len(cap_dropped_sub_queries)
        if dropped_count > 0:
            overflow = OverflowChip(
                label=f"+ {dropped_count} more concept{'s' if dropped_count != 1 else ''} — refine to see them",
                dropped_sub_queries=list(cap_dropped_sub_queries),
                count=dropped_count,
            )
        return MultiIntentChipStrip(per_intent_chips=groups, overflow_chip=overflow, total_kept=len(groups), total_dropped=dropped_count)

    def _build_joint_transformed_intent(
        self,
        parent_intent: QueryIntent,
        keyword_entries: List[Dict[str, Any]],
    ) -> Tuple[IntentSlice, QueryIntent]:
        """Build a single-slice intent for joint transformed-query retrieve.

        Encode text comes from the parent semantic encode (or keyword join).
        Hard and soft entities are the union across parent slices.
        """
        terms = [
            str(k.get('term') or '').strip()
            for k in keyword_entries
            if isinstance(k, dict) and str(k.get('term') or '').strip()
        ]
        if len(terms) < 2:
            raise ValidationError("_build_joint_transformed_intent requires >=2 keyword terms")
        _nav = frozenset(self._config.qi.residual.navigational_tokens) if (self._config.qi.residual is not None) else frozenset()
        # Union hard entities across slices (price etc. already broadcast/propagated).
        by_name: Dict[str, Entity] = {}
        soft: List[Entity] = []
        soft_seen: set = set()
        for s in parent_intent.slices or []:
            for e in s.entities or []:
                if e.name not in by_name:
                    by_name[e.name] = e
            for e in list(getattr(s, 'soft_entities', None) or []):
                if e.name not in soft_seen:
                    soft.append(e)
                    soft_seen.add(e.name)
        entities = list(by_name.values())
        encode_text = str(parent_intent.semantic_encode_text or '').strip()
        if not encode_text:
            encode_text = build_semantic_encode_text(
                parent_intent.normalized_query,
                entities + soft,
                parent_intent.semantic_query,
                _nav,
            )
        if not encode_text:
            encode_text = ' '.join(terms)
        joint_raw = str(parent_intent.normalized_query or encode_text).strip()
        joint_slice = IntentSlice(
            query_type=parent_intent.query_type,
            entities=entities,
            confidence=float(parent_intent.confidence),
            raw_text=joint_raw,
            slice_id='joint_transformed',
            soft_entities=soft,
            keywords=list(keyword_entries),
        )
        wrapper = QueryIntent(
            request_id=parent_intent.request_id,
            raw_query=parent_intent.raw_query,
            normalized_query=joint_raw,
            query_type=parent_intent.query_type,
            confidence=float(parent_intent.confidence),
            decision_tier=parent_intent.decision_tier,
            slices=[joint_slice],
            decision_cost_usd=0.0,
            intent_record_id=parent_intent.intent_record_id,
            residual_kind='semantic' if encode_text else 'empty',
            semantic_encode_text=encode_text,
            semantic_query=encode_text or None,
            keywords=list(keyword_entries),
        )
        return joint_slice, wrapper

    def _build_single_intent_for_slice(self, parent_intent: QueryIntent, slc: IntentSlice) -> QueryIntent:
        """Build a single-query QueryIntent for one multi-intent slice.

        Applies sibling slot propagate from config, slice encode text, and the same
        hybrid-first rewrite used on the default single-query route when
        ranked_results_complement is enabled for CH-backed query types.
        """
        _slice_nav = frozenset(self._config.qi.residual.navigational_tokens) if (self._config.qi.residual is not None) else frozenset()
        _slice_normalized = slc.raw_text or parent_intent.normalized_query
        _propagate_slots = frozenset(self._config.multi_intent.cross_slice_retrieve_propagate_slots)
        _slice_entity_names = {e.name for e in slc.entities}
        _propagated = [
            e for s in parent_intent.slices
            for e in s.entities
            if e.name in _propagate_slots and e.name not in _slice_entity_names
        ]
        _effective_entities = list(slc.entities) + _propagated
        _soft = list(getattr(slc, 'soft_entities', None) or [])
        _kw = list(getattr(slc, 'keywords', None) or [])
        _slice_encode_text = build_semantic_encode_text(
            _slice_normalized, _effective_entities + _soft, parent_intent.semantic_query, _slice_nav,
        )
        if not str(_slice_encode_text or '').strip() and _kw:
            _slice_encode_text = ' '.join(
                str(k.get('term') or '').strip() for k in _kw if k.get('term')
            ).strip()
        if bool(self._config.multi_intent.slice_encode_blend_parent_concept):
            _parent_concept = str(parent_intent.semantic_encode_text or '').strip()
            if not _parent_concept:
                _parent_ents = [
                    e
                    for s in (parent_intent.slices or [])
                    for e in list(s.entities or []) + list(getattr(s, 'soft_entities', None) or [])
                ]
                _parent_concept = build_semantic_encode_text(
                    parent_intent.normalized_query,
                    _parent_ents,
                    parent_intent.semantic_query,
                    _slice_nav,
                )
            if not _parent_concept and parent_intent.keywords:
                _parent_concept = ' '.join(
                    str(k.get('term') or '').strip()
                    for k in parent_intent.keywords
                    if k.get('term')
                ).strip()
            if _parent_concept:
                _base_tokens = set(re.findall(r"[a-z0-9]+", str(_slice_encode_text or '').lower()))
                _extra = [
                    tok for tok in re.findall(r"[a-z0-9]+", _parent_concept.lower())
                    if tok not in _base_tokens
                ]
                if _extra:
                    _slice_encode_text = (
                        f"{_slice_encode_text} {' '.join(_extra)}".strip()
                        if str(_slice_encode_text or '').strip()
                        else ' '.join(_extra)
                    )
        _residual = 'semantic' if str(_slice_encode_text or '').strip() else 'empty'
        wrapper = QueryIntent(
            request_id=parent_intent.request_id,
            raw_query=slc.raw_text or parent_intent.raw_query,
            normalized_query=_slice_normalized,
            query_type=slc.query_type,
            confidence=float(slc.confidence),
            decision_tier=parent_intent.decision_tier,
            slices=[IntentSlice(
                query_type=slc.query_type,
                entities=_effective_entities,
                confidence=slc.confidence,
                raw_text=slc.raw_text,
                slice_id=slc.slice_id or '',
                soft_entities=_soft,
                keywords=_kw,
            )],
            decision_cost_usd=0.0,
            intent_record_id=parent_intent.intent_record_id,
            residual_kind=_residual,
            semantic_encode_text=_slice_encode_text,
            semantic_query=_slice_encode_text or None,
            keywords=_kw,
        )
        _comp = self._config.general.search.ranked_results_complement
        _ch_up = self.analytics_available
        if _comp.enabled and wrapper.query_type in _CH_DEGRADE_QUERY_TYPES:
            _has_concept = _has_listing_concept(wrapper.slices)
            _strip_temporal = (not _ch_up) or (
                bool(_comp.strip_temporal_when_clickhouse_available) and _has_concept
            )
            wrapper = hybrid_retrieve_intent(wrapper, strip_temporal=_strip_temporal)
            if _should_restore_rewrite_encode(slc.query_type, has_listing_concept=_has_concept):
                # No per-slice transform_result; keep slice encode when concept-empty.
                if not str(wrapper.semantic_encode_text or '').strip() and str(_slice_encode_text or '').strip():
                    wrapper = dataclasses.replace(
                        wrapper,
                        semantic_encode_text=_slice_encode_text,
                        semantic_query=_slice_encode_text,
                        residual_kind='semantic',
                    )
        elif wrapper.query_type in _CH_DEGRADE_QUERY_TYPES and not _ch_up:
            wrapper = hybrid_degrade_intent(wrapper)
        return wrapper

    async def _retrieve_and_fuse_slice(self, parent_intent: QueryIntent, slc: IntentSlice) -> RankedResults:
        """Run one multi-intent slice through the default single-query retrieve route.

        Builds a single-slice intent, applies the same hybrid-first rewrite as
        ``_search_inner`` for CH-backed types, then calls ``_retrieve_and_rank_single``.
        Cross-slice aggregation stays in ``_retrieve_and_rank_multi``.
        """
        wrapper = self._build_single_intent_for_slice(parent_intent, slc)
        logger.info(
            f"multi_intent_slice_single_route request_id={parent_intent.request_id} "
            f"slice_id={slc.slice_id} query_type={wrapper.query_type} "
            f"encode_len={len(str(wrapper.semantic_encode_text or ''))} "
            f"residual_kind={wrapper.residual_kind}"
        )
        return await self._retrieve_and_rank_single(wrapper, top_k=None)

    async def _prescreen_structured_count_for_slice(self, slc: IntentSlice) -> int:
        """Structured candidate count for one slice (multi-intent pre-screen).

        Uses the structured retriever bounded at general.max_results so a count for an
        unbounded slice (no filters) cannot exhaust memory. Returns 0 when the slice
        carries no structured filters or when the retriever has no candidates.
        """
        wrapper_intent = QueryIntent(
            request_id=QueryIntent.new_request_id(),
            raw_query=slc.raw_text or 'preview',
            normalized_query=(slc.raw_text or 'preview').lower(),
            query_type=slc.query_type,
            confidence=float(slc.confidence),
            decision_tier='L2_llm',
            slices=[slc],
            decision_cost_usd=0.0,
        )
        try:
            cs = await self._structured.retrieve(wrapper_intent, top_k=int(self._config.general.max_results))
            return len(cs.candidates)
        except (ValidationError, RetrievalError) as e:
            logger.warning(f"multi_intent_prescreen_count_failed slice_query_type={slc.query_type} error={str(e)}")
            return 0

    def _record_observation(
        self,
        rid: str,
        user_context: Optional[UserContext],
        intent: QueryIntent,
        results: RankedResults,
        total_ms: float,
        cache_hit: Optional[str],
        eranker_outcome: Optional[ERankerOutcome] = None,
    ) -> None:
        """Capture a `SearchObservation` for the measurement subsystem.

        Never raises — measurement is an observability concern, not a request
        path concern. Failures are logged + swallowed so the user-visible
        response is unaffected.
        """
        try:
            session_id = user_context.session_id if user_context is not None else ''
            distinct = 0
            if results.items:
                distinct = len({it.item_id for it in results.items})
            distinct_ratio = (distinct / len(results.items)) if results.items else 1.0
            # Active request gate when present; else intent.decision_cost_usd.
            _budget = self.last_cost_budget
            _cost = (
                float(_budget.running_total_usd)
                if _budget is not None
                else float(intent.decision_cost_usd)
            )
            obs = SearchObservation(
                request_id=rid,
                session_id=session_id,
                query_type=intent.query_type,
                decision_tier=intent.decision_tier,
                confidence=float(intent.confidence),
                decision_cost_usd=_cost,
                result_count=len(results.items),
                distinct_item_ratio=distinct_ratio,
                total_latency_ms=total_ms,
                cache_hit=cache_hit,
                intent_record_id=intent.intent_record_id,
                eranker_applied=eranker_outcome.applied if eranker_outcome is not None else None,
                eranker_latency_ms=eranker_outcome.latency_ms if eranker_outcome is not None else None,
                eranker_skipped_reason=eranker_outcome.skipped_reason if eranker_outcome is not None else None,
            )
            self._measurement.record(obs)
        except (ValidationError, ValueError) as e:
            logger.warning(f"measurement_record_failed request_id={rid} error_type={type(e).__name__} error={str(e)}")

    def _maybe_record_history(self, user_context: Optional[UserContext], intent: QueryIntent, results: RankedResults) -> None:
        """Record a history entry for authenticated callers; safe no-op otherwise."""
        if user_context is None or not user_context.is_authenticated or user_context.user_id is None:
            return
        if not self._history.is_enabled():
            return
        top_ids = [it.item_id for it in results.items[: self._config.general.max_results]]
        try:
            self._history.record(user_id=user_context.user_id, normalized_query=intent.normalized_query, query_type=intent.query_type, top_item_ids=top_ids, intent_record_id=intent.intent_record_id)
        except HistoryError as e:
            logger.warning(f"history_record_skipped request_id={intent.request_id} error_type={type(e).__name__} error={str(e)}")

    def _qi_normalize(self, raw_query: str) -> str:
        """Apply the same normalization the QI engine uses (no LLM cost)."""
        return normalize_query(
            raw_query,
            self._config.general.max_query_length,
            normalize=self._config.qi.normalize,
        )

    def _sanitize_user_input(self, raw_query: str, request_id: str, surface: str) -> None:
        """Gate raw user input through the Layer-0 sanitizer.

        Runs BEFORE QI classification, cache lookup, and any downstream call so
        prompt-injection markers / PII / over-length payloads are rejected at
        the API ingress (single chokepoint shared with the LLM ingress gate so
        counters in `LayerZeroSanitizer.stats()` reflect both surfaces).

        :param raw_query: str - The raw user query to evaluate
        :param request_id: str - Correlation id used in the warning log
        :param surface: str - Surface identifier for the log (`search` / `analytics`)
        :raises ValidationError: When the sanitizer rejects the input. The
            exception message carries categorical reason codes only — never the
            raw payload — per `responsible-ai.mdc` (the rejected text may be
            adversarial). The API maps this to HTTP 422 cleanly without leaking.
        """
        if self._sanitizer is None:
            return
        if not self._sanitizer.applies_to_llm_ingress:
            # Same toggle as the LLM ingress gate — when ops disable one they
            # disable both. This keeps the security posture coherent: a single
            # config flip turns the entire defence layer on or off rather than
            # leaving the API ingress gated while the LLM ingress is open
            # (or vice versa).
            return
        verdict = self._sanitizer.sanitize(raw_query)
        if verdict.passed:
            return
        logger.warning(f"user_input_blocked_by_sanitizer surface={surface} request_id={request_id} reasons={verdict.reasons}")
        raise ValidationError(f"user_input_blocked_by_sanitizer surface={surface} reasons={verdict.reasons}")

    @staticmethod
    def _with_request_id(results: RankedResults, request_id: str, cache_hit: str) -> RankedResults:
        """Return a copy of `results` stamped with the caller's request_id and cache_hit tier."""
        return RankedResults(
            request_id=request_id,
            items=list(results.items),
            total_candidates=results.total_candidates,
            fusion_latency_ms=results.fusion_latency_ms,
            cache_hit=cache_hit,
            multi_intent_envelope=results.multi_intent_envelope,
            failure_mode=results.failure_mode,
            query_intent=results.query_intent,
        )

    async def _serve_cache_hit(
        self,
        rid: str,
        payload: 'CachedSearchPayload',
        tier: str,
        user_context: Optional[UserContext],
        t0: float,
        top_k: Optional[int] = None,
    ) -> Tuple[RankedResults, ERankerOutcome, ZeroResultGuardOutcome]:
        """Cache-hit path: re-run eRanker on cached fused results without QI.

        The cache stores ``CachedSearchPayload(intent, results)`` so eRanker
        receives the same ``QueryIntent`` as the priming request.

        :param rid: str - The current request's correlation id (NOT the cached one)
        :param payload: CachedSearchPayload - The cache entry (intent + results)
        :param tier: str - 'exact' or 'semantic' — stamped onto cache_hit
        :param user_context: Optional[UserContext] - Forwarded to eRanker client
        :param t0: float - Search start time for observation latency
        :param top_k: Optional[int] - Per-request result cap; respected up to max_results
        :return: Tuple matching the ``search()`` return signature
        """
        cached_intent = payload.intent
        cached_ranked = self._with_request_id(payload.results, rid, tier)
        eranked, outcome = await self._apply_eranker(rid, cached_intent, cached_ranked, user_context)
        self._last_eranker_outcome = outcome
        self._maybe_record_history(user_context, cached_intent, eranked)
        truncated = self._truncate(eranked, limit=top_k)
        scrubbed = await self._apply_egress_guard(truncated)
        scrubbed = RankedResults(
            request_id=scrubbed.request_id,
            items=scrubbed.items,
            total_candidates=scrubbed.total_candidates,
            fusion_latency_ms=scrubbed.fusion_latency_ms,
            cache_hit=scrubbed.cache_hit,
            multi_intent_envelope=scrubbed.multi_intent_envelope,
            failure_mode=scrubbed.failure_mode,
            query_intent=cached_intent,
        )
        scrubbed = await self._with_guidance_envelope(cached_intent, scrubbed)
        total_ms = (time.monotonic() - t0) * 1000.0
        self._record_observation(rid=rid, user_context=user_context, intent=cached_intent, results=scrubbed, total_ms=total_ms, cache_hit=tier, eranker_outcome=outcome)
        self._publish_ranking_stages(
            request_id=rid,
            cache_hit=tier,
            fusion_latency_ms=float(scrubbed.fusion_latency_ms or 0.0),
            total_candidates=int(scrubbed.total_candidates),
            result_count=len(scrubbed.items),
            eranker_outcome=outcome,
            guard_outcome=self._noop_guard_outcome,
        )
        return scrubbed, outcome, self._noop_guard_outcome

    async def _serve_intent_plan_hit(
        self,
        rid: str,
        intent: QueryIntent,
        payload: CachedSearchPayload,
        user_context: Optional[UserContext],
        t0: float,
    ) -> Tuple[RankedResults, ERankerOutcome, ZeroResultGuardOutcome]:
        """Intent-structure cache hit: skip retrieval; use fresh intent wording with cached fused results."""
        ranked_stamped = self._with_request_id(payload.results, rid, 'intent_plan')
        eranked, outcome = await self._apply_eranker(rid, intent, ranked_stamped, user_context)
        self._last_eranker_outcome = outcome
        self._maybe_record_history(user_context, intent, eranked)
        truncated = self._truncate(eranked)
        scrubbed = await self._apply_egress_guard(truncated)
        scrubbed = RankedResults(
            request_id=scrubbed.request_id,
            items=scrubbed.items,
            total_candidates=scrubbed.total_candidates,
            fusion_latency_ms=scrubbed.fusion_latency_ms,
            cache_hit=scrubbed.cache_hit,
            multi_intent_envelope=scrubbed.multi_intent_envelope,
            failure_mode=scrubbed.failure_mode,
            query_intent=intent,
        )
        scrubbed = await self._with_guidance_envelope(intent, scrubbed)
        total_ms = (time.monotonic() - t0) * 1000.0
        self._record_observation(rid=rid, user_context=user_context, intent=intent, results=scrubbed, total_ms=total_ms, cache_hit='intent_plan', eranker_outcome=outcome)
        self._publish_ranking_stages(
            request_id=rid,
            cache_hit='intent_plan',
            fusion_latency_ms=float(scrubbed.fusion_latency_ms or 0.0),
            total_candidates=int(scrubbed.total_candidates),
            result_count=len(scrubbed.items),
            eranker_outcome=outcome,
            guard_outcome=self._noop_guard_outcome,
        )
        return scrubbed, outcome, self._noop_guard_outcome

    async def _with_guidance_envelope(self, intent: QueryIntent, ranked: RankedResults, snapshot_task: Optional[asyncio.Task] = None) -> RankedResults:
        """Attach ClickHouse market snapshot when query_type is guidance and the service is wired.

        When snapshot_task is provided, the caller has already fired build_snapshot()
        concurrently with retrieval — await it with the configured timeout instead of
        issuing a new call. Cache-hit paths pass snapshot_task=None and fall through to
        the direct await on cache-hit paths.
        """
        if intent.query_type != 'guidance' or self._guidance_service is None:
            if snapshot_task is not None and not snapshot_task.done():
                snapshot_task.cancel()
            return ranked
        if snapshot_task is not None:
            try:
                ge = await asyncio.wait_for(snapshot_task, timeout=float(self._config.guidance.snapshot_timeout_seconds))
            except (asyncio.TimeoutError, RuntimeError, ValueError, TypeError, KeyError, OSError) as _snap_err:
                logger.warning(f"guidance_snapshot_task_failed request_id={intent.request_id} error_type={type(_snap_err).__name__} error={_snap_err}")
                ge = None
        else:
            # Collect hard tld entities from the intent so the snapshot is scoped to the
            # TLDs the user asked about (e.g. "what .com domains" -> filter to .com only).
            tld_filter = []
            for sl in (intent.slices or []):
                for ent in (sl.entities or []):
                    if ent.name == 'tld' and getattr(ent, 'chip_kind', 'hard') == 'hard' and ent.value is not None:
                        vals = ent.value if isinstance(ent.value, list) else [ent.value]
                        tld_filter.extend(str(v) for v in vals if v)
            ge = await self._guidance_service.build_snapshot(request_id=intent.request_id, tld_filter=tld_filter or None)
        if ge is None:
            return ranked
        return dataclasses.replace(ranked, guidance_envelope=ge)

    async def get_timeout_fallback(self, request_id: str, query: str, top_k: int, intent: Optional[QueryIntent] = None, bypass_cache: bool = False) -> 'RankedResults':
        """Return RRF-fused mix of explore rails + semantic results for timeout/SLA-breach paths.
        :param request_id: str - Correlation id for the originating request
        :param query: str - Raw query text used for semantic search
        :param top_k: int - Result cap
        :param intent: Optional[QueryIntent] - Original classified intent; when provided, its slices
            carry hard filters (tld, auction_type, etc.) applied post-fusion so fallback results
            match the user's original query context
        :param bypass_cache: bool - When True, skip the in-memory result cache so entity slices
            from the provided intent are applied without returning a cached unscoped result
        :return: RankedResults - RRF-fused mix capped to top_k
        """
        fallback_timeout = float(self._config.general.search.explore_fallback_timeout_seconds)
        _cache_ttl = float(self._config.general.search.explore_fallback_cache_ttl_seconds)
        _now = time.monotonic()
        if not bypass_cache and self._fb_result_cache is not None:
            _cached, _cached_at = self._fb_result_cache
            if _now - _cached_at < _cache_ttl:
                logger.debug(f"get_timeout_fallback_cache_hit request_id={request_id} age_s={_now - _cached_at:.1f}")
                return _cached
        empty = RankedResults(request_id=request_id, items=[], total_candidates=0, fusion_latency_ms=0.0, cache_hit=None, failure_mode='explore_fallback_rail')

        # Copy slices from the original intent so compose_fallback can apply hard filters
        # (e.g. tld=com) post-fusion, returning contextually relevant results.
        # Fall back to a generic explore slice when no intent is provided (pre-warmed path).
        _fb_slices = intent.slices if intent is not None and intent.slices else [IntentSlice(query_type='explore', entities=[], confidence=1.0, raw_text='')]
        fallback_intent = QueryIntent(
            request_id=request_id,
            raw_query=query if query.strip() else 'timeout_fallback',
            normalized_query=query.strip().lower() if query.strip() else 'timeout_fallback',
            query_type='explore',
            confidence=1.0,
            decision_tier='fallback',
            slices=_fb_slices,
            decision_cost_usd=0.0,
            intent_record_id=QueryIntent.new_intent_record_id(),
            alternative_interpretations=[],
            routing_mode='auto_execute',
        )

        # Prefer search()-registered explore rails (same request_id); else compose.
        # Semantic retrieve runs in parallel so timeouts return meaningful listings.
        semantic_top_k = int(self._config.explore.zero_result_guard.semantic_fallback_top_k)
        rrf_k = int(self._config.explore.zero_result_guard.rrf_k)
        tf_cfg = self._config.general.search.timeout_fallback
        _max_per_rail = int(self._config.explore.zero_result_guard.explore_fallback_max_per_rail)

        async def _resolve_explore_leg() -> List[RankedItem]:
            _prewarm_ranked = await self._await_registered_explore_prewarm(request_id)
            if _prewarm_ranked is not None:
                logger.info(
                    f"get_timeout_fallback_prewarm_hit request_id={request_id} "
                    f"items={len(_prewarm_ranked.items)}"
                )
                return list(_prewarm_ranked.items)
            if self._explore_composer is not None and self._explore_composer.enabled:
                ranked_explore, _ = await self._explore_composer.compose_fallback(
                    intent=fallback_intent,
                    max_per_rail_override=_max_per_rail,
                    user_id=None,
                    semantic_items=None,
                )
                return list(ranked_explore.items) if ranked_explore.items else []
            return []

        explore_task = asyncio.create_task(_resolve_explore_leg())
        semantic_task = asyncio.create_task(self._quick_semantic_retrieve(fallback_intent, semantic_top_k))
        gather_tasks = [explore_task, semantic_task]
        try:
            raw_results = await asyncio.wait_for(asyncio.gather(*gather_tasks, return_exceptions=True), timeout=fallback_timeout)
        except asyncio.TimeoutError:
            logger.warning(f"get_timeout_fallback_timed_out request_id={request_id} timeout={fallback_timeout}")
            for _t in gather_tasks:
                _t.cancel()
            if bool(tf_cfg.qdrant_rails_when_ch_empty):
                qdrant_items = await self._qdrant_filter_only_rails_as_ranked(fallback_intent, top_k)
                if qdrant_items:
                    result = RankedResults(
                        request_id=request_id,
                        items=qdrant_items[:top_k],
                        total_candidates=len(qdrant_items),
                        fusion_latency_ms=0.0,
                        cache_hit=None,
                        failure_mode='explore_fallback_rail',
                    )
                    self._fb_result_cache = (result, time.monotonic())
                    return result
            return empty
        except _GUARD_EXC as _e:
            logger.warning(f"get_timeout_fallback_failed request_id={request_id} error={_e}")
            if bool(tf_cfg.qdrant_rails_when_ch_empty):
                qdrant_items = await self._qdrant_filter_only_rails_as_ranked(fallback_intent, top_k)
                if qdrant_items:
                    result = RankedResults(
                        request_id=request_id,
                        items=qdrant_items[:top_k],
                        total_candidates=len(qdrant_items),
                        fusion_latency_ms=0.0,
                        cache_hit=None,
                        failure_mode='explore_fallback_rail',
                    )
                    self._fb_result_cache = (result, time.monotonic())
                    return result
            return empty

        explore_result = raw_results[0]
        semantic_items = raw_results[1]

        if isinstance(explore_result, BaseException):
            logger.warning(f"get_timeout_fallback_explore_error request_id={request_id} error={explore_result}")
            explore_items: List[RankedItem] = []
        else:
            explore_items = list(explore_result) if explore_result else []
        if isinstance(semantic_items, BaseException):
            logger.warning(f"get_timeout_fallback_semantic_error request_id={request_id} error={semantic_items}")
            semantic_items = []

        # A: CH rails empty -> Qdrant filter_only_rails multi-scroll (explore-like).
        if not explore_items and bool(tf_cfg.qdrant_rails_when_ch_empty):
            explore_items = await self._qdrant_filter_only_rails_as_ranked(fallback_intent, top_k)
            if explore_items:
                logger.info(
                    f"get_timeout_fallback_qdrant_rails request_id={request_id} "
                    f"items={len(explore_items)}"
                )

        # B: rail_first — do not discard explore/Qdrant rail cards for semantic-only.
        # Semantic quality gate: when the semantic leg returns enough high-confidence results,
        # return them directly without diluting with generic explore-rail domains.
        _sem_min = int(self._config.general.search.explore_fallback_semantic_prefer_min_results)
        _sem_score = float(self._config.general.search.explore_fallback_semantic_prefer_min_score)
        _skip_semantic_prefer = bool(tf_cfg.rail_first) and bool(explore_items)
        if (
            not _skip_semantic_prefer
            and _sem_min >= 1
            and isinstance(semantic_items, list)
            and len(semantic_items) >= _sem_min
        ):
            _avg_score = sum(getattr(i, 'fused_score', 0.0) for i in semantic_items[:_sem_min]) / _sem_min
            if _avg_score >= _sem_score:
                logger.debug(f"get_timeout_fallback_semantic_prefer request_id={request_id} count={len(semantic_items)} avg_score={_avg_score:.3f}")
                sem_top_k = semantic_items[:top_k]
                return RankedResults(request_id=request_id, items=sem_top_k, total_candidates=len(semantic_items), fusion_latency_ms=0.0, cache_hit=None, failure_mode='explore_fallback_rail')
        elif _skip_semantic_prefer:
            logger.debug(
                f"get_timeout_fallback_rail_first request_id={request_id} "
                f"explore_items={len(explore_items)} skipped_semantic_prefer=true"
            )
        # Re-fuse explore items with semantic items via RRF.
        all_lists = []
        if explore_items:
            all_lists.append(('explore', explore_items))
        if semantic_items:
            all_lists.append(('semantic', semantic_items))
        fused = _rrf_fuse(all_lists, k=rrf_k) if all_lists else []
        fused_top_k = fused[:top_k]
        result = RankedResults(request_id=request_id, items=fused_top_k, total_candidates=len(fused), fusion_latency_ms=0.0, cache_hit=None, failure_mode='explore_fallback_rail')
        if fused_top_k:
            self._fb_result_cache = (result, time.monotonic())
        return result

    async def _qdrant_filter_only_rails_as_ranked(self, intent: 'QueryIntent', top_k: int) -> List['RankedItem']:
        """Qdrant multi-scroll RRF ladder for timeout fallback when CH rails are empty.

        Requires a vector retriever exposing ``retrieve_filter_only_rails``. Bounded by
        ``timeout_fallback.qdrant_rails_timeout_seconds``. Returns [] on any failure.
        """
        retr = self._vector
        method = getattr(retr, 'retrieve_filter_only_rails', None) if retr is not None else None
        if method is None:
            return []
        timeout_s = float(self._config.general.search.timeout_fallback.qdrant_rails_timeout_seconds)
        try:
            candidate_set = await asyncio.wait_for(method(intent, top_k), timeout=timeout_s)
        except asyncio.TimeoutError:
            logger.warning(
                f"qdrant_timeout_rails_timed_out request_id={intent.request_id} "
                f"timeout_s={timeout_s:.2f}"
            )
            return []
        except _FALLBACK_RETRIEVE_EXC as e:
            # Missing/down Qdrant degrades to [] (no 500 on timeout-fallback).
            logger.warning(
                f"qdrant_timeout_rails_failed request_id={intent.request_id} "
                f"error_type={type(e).__name__} error={e}"
            )
            return []
        return [
            RankedItem(
                item_id=c.item_id,
                fused_score=float(c.score),
                contributing_sources=[c.source],
                payload=dict(c.payload or {}),
            )
            for c in (candidate_set.candidates or [])
        ]

    async def _quick_semantic_retrieve(self, intent: 'QueryIntent', top_k: int) -> List['RankedItem']:
        """Run a fast vector-only search for the fallback path. Returns empty list on any failure.

        Records a health observation on the BackendHealthRegistry so the zero-result guard's
        implicit probe can unblock a stuck-open vector circuit breaker.
        """
        try:
            if self._vector is None:
                return []
            candidate_set = await self._vector.retrieve(intent, top_k)
            self._health.record('vector', success=True)
            return [
                RankedItem(
                    item_id=c.item_id,
                    fused_score=float(c.score),
                    contributing_sources=[c.source],
                    payload=dict(c.payload or {}),
                )
                for c in (candidate_set.candidates or [])
            ]
        except _FALLBACK_RETRIEVE_EXC as _e:
            # Qdrant down / connection refused returns [] (doc: any failure).
            logger.debug(f"quick_semantic_retrieve_failed error_type={type(_e).__name__} error={_e}")
            self._health.record('vector', success=False)
            return []

    @property
    def last_eranker_outcome(self) -> ERankerOutcome:
        """Most recent eRanker (Layer 4) audit outcome."""
        return self._last_eranker_outcome

    @property
    def last_ranking_stages(self) -> Dict[str, Any]:
        """Most recent ranking-stage attribution counters for this process."""
        return dict(self._last_ranking_stages)

    def _reset_stage_scratch(self) -> None:
        """Clear per-request ranking stage scratch (no-op when attribution disabled)."""
        attr = self._config.measurement.ranking_stage_attribution
        if not attr.enabled:
            self._stage_scratch = {}
            return
        self._stage_scratch = {
            'retrieve_sources': [],
            'retrieve_sizes': [],
            'hard_gate_before': None,
            'hard_gate_after': None,
            'soft_boost_applied': False,
            'diversify_applied': False,
            'diversify_skipped_reason': 'not_run',
            'diversify_latency_ms': None,
        }

    def _publish_ranking_stages(
        self,
        *,
        request_id: str,
        cache_hit: Optional[str],
        fusion_latency_ms: float,
        total_candidates: int,
        result_count: int,
        eranker_outcome: ERankerOutcome,
        guard_outcome: ZeroResultGuardOutcome,
    ) -> None:
        """Assemble, store, and optionally log ranking stage attribution."""
        attr = self._config.measurement.ranking_stage_attribution
        if not attr.enabled:
            self._last_ranking_stages = {}
            return
        scratch = self._stage_scratch or {}
        before = scratch.get('hard_gate_before')
        after = scratch.get('hard_gate_after')
        dropped = None
        if isinstance(before, int) and isinstance(after, int):
            dropped = max(0, before - after)
        stages: Dict[str, Any] = {
            'cache_hit': cache_hit,
            'retrieve_sources': list(scratch.get('retrieve_sources') or []),
            'retrieve_sizes': list(scratch.get('retrieve_sizes') or []),
            'fusion_latency_ms': round(float(fusion_latency_ms), 3),
            'hard_gate_before': before,
            'hard_gate_after': after,
            'hard_gate_dropped': dropped,
            'soft_boost_applied': bool(scratch.get('soft_boost_applied')),
            'eranker_applied': bool(eranker_outcome.applied),
            'eranker_client': eranker_outcome.client,
            'eranker_skipped_reason': eranker_outcome.skipped_reason,
            'eranker_latency_ms': (
                round(float(eranker_outcome.latency_ms), 3)
                if eranker_outcome.latency_ms is not None
                else None
            ),
            'diversify_applied': bool(scratch.get('diversify_applied')),
            'diversify_skipped_reason': scratch.get('diversify_skipped_reason'),
            'diversify_latency_ms': scratch.get('diversify_latency_ms'),
            'zero_result_fired': bool(guard_outcome.fired),
            'zero_result_ladder_step': guard_outcome.ladder_step,
            'total_candidates': int(total_candidates),
            'result_count': int(result_count),
        }
        self._last_ranking_stages = stages
        if attr.log_event:
            logger.info(
                f"ranking_stage_attribution request_id={request_id} "
                f"cache_hit={cache_hit or 'none'} "
                f"retrieve_sources={stages['retrieve_sources']} "
                f"retrieve_sizes={stages['retrieve_sizes']} "
                f"fusion_latency_ms={stages['fusion_latency_ms']} "
                f"hard_gate_dropped={dropped if dropped is not None else 'n/a'} "
                f"soft_boost_applied={stages['soft_boost_applied']} "
                f"eranker_applied={stages['eranker_applied']} "
                f"eranker_skip={stages['eranker_skipped_reason'] or 'none'} "
                f"eranker_latency_ms={stages['eranker_latency_ms'] if stages['eranker_latency_ms'] is not None else 'n/a'} "
                f"diversify_applied={stages['diversify_applied']} "
                f"diversify_skip={stages['diversify_skipped_reason'] or 'none'} "
                f"zero_result_fired={stages['zero_result_fired']} "
                f"zero_result_step={stages['zero_result_ladder_step']} "
                f"total_candidates={stages['total_candidates']} "
                f"result_count={stages['result_count']}"
            )

    @property
    def last_vector_retrieve_error(self) -> Optional[BaseException]:
        """Last vector-backend RetrievalError from this request (None if vector succeeded)."""
        return self._last_retrieve_errors.get('vector')

    @property
    def analytics_available(self) -> bool:
        """True iff the analytics router is wired AND its config-side enabled flag is on.

        The router itself reports `enabled` based on `(config.enabled AND
        executor_credentials_available)` — the orchestrator does not second-guess
        that. False here means analytics queries cannot be served end-to-end and
        callers should expect `analytics()` to raise.
        """
        return self._analytics_router is not None and self._analytics_router.enabled

    async def analytics(
        self,
        question: str,
        sql_hint: str,
        request_id: Optional[str] = None,
        intent_record_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> AnalyticsResult:
        """Run a natural-language analytics question through the analytics path.

        The substrate ladder (NL-SQL exact cache -> ClickHouse MV ->
        ``events_raw`` scan) is fully owned by
        ``AnalyticsRouter``. This method is a thin orchestrator-level entry
        point so callers see one place where every query reaches the system.

        When ``cost_budget_factory`` is wired, instantiates ONE
        ``QueryCostBudget`` per call and binds it to the request-scoped
        ContextVar so the NL-SQL generator + verifier cost flows into a
        single per-question ceiling. A breach surfaces as
        ``QueryCostBudgetExceeded`` which we convert to a typed
        ``AnalyticsResult.failure_mode='cost_budget_exceeded'`` so the
        caller doesn't 5xx.

        :param question: str - Natural-language analytics question
        :param sql_hint: str - Optional structured hint piped from QI
        :param request_id: Optional[str] - Correlation id (auto-generated when None)
        :param intent_record_id: Optional[str] - IntentRecord id from the
            originating QI verdict. Plumbed onto the returned
            ``AnalyticsResult.intent_record_id`` so the
            ``analytics_failure`` FeedbackSignal carries the same join key
            the search-side ``IntentRecord`` exposes. Empty string
            when caller has no IntentRecord context.
        :return: AnalyticsResult - Typed result; `success=True` on verifier-passing rows
        :raises RetrievalError: When the analytics router is not wired or disabled
        :raises ValidationError: When `question` is None / not a string
        """
        if question is None or not isinstance(question, str):
            raise ValidationError("analytics requires a non-null string question")
        if sql_hint is None or not isinstance(sql_hint, str):
            raise ValidationError("analytics requires a string sql_hint (use '' when none)")
        # Sanitize both the natural-language question AND the SQL hint
        # BEFORE the substrate ladder fires. The hint is caller-supplied
        # free text; without sanitization it would land directly in the
        # NL-to-SQL prompt on a cache miss. Two separate sanitize calls so
        # the rejection log identifies which side failed.
        rid = request_id if (request_id and isinstance(request_id, str)) else AnalyticsResult.new_request_id()
        self._sanitize_user_input(question, rid, surface='analytics_question')
        if sql_hint:
            self._sanitize_user_input(sql_hint, rid, surface='analytics_sql_hint')
        if self._analytics_router is None:
            raise RetrievalError("analytics_router_not_wired analytics path disabled at boot (no LLM router or analytics.enabled=false)")
        if not self._analytics_router.enabled:
            raise RetrievalError("analytics_router_disabled clickhouse executor unavailable or analytics.enabled=false in config")
        # Per-tenant rate limit gate. When wired AND enabled, the limiter is
        # consulted before any expensive work (LLM, ClickHouse) so
        # capacity-shedding happens cheaply. Bucket key is chosen from the
        # configured strategy with conservative fallbacks: anonymous users
        # collapse onto session_id, and unbound sessions fall back to the
        # request_id (process-local — same caller in the same process won't
        # share a bucket across requests, which is the safe default).
        if (self._analytics_rate_limiter is not None and self._analytics_rate_limit_config is not None and self._analytics_rate_limit_config.enabled):
            strategy = self._analytics_rate_limit_config.key_strategy
            if strategy == 'user_id':
                bucket_key = user_id or session_id or rid
            else:
                bucket_key = session_id or user_id or rid
            if not self._analytics_rate_limiter.check_and_record(bucket_key):
                retry = self._analytics_rate_limiter.retry_after_seconds(bucket_key)
                logger.warning(f"analytics_rate_limited request_id={rid} bucket_key={bucket_key} strategy={strategy} retry_after_s={retry}")
                return AnalyticsResult(
                    request_id=rid,
                    question=question,
                    sql_hint=sql_hint,
                    success=False,
                    failure_mode='rate_limited',
                    failure_reason=f'analytics rate limit exceeded; retry in {retry}s',
                    pruned_schema=None,
                    generation=None,
                    validation=None,
                    execution=None,
                    verifier=None,
                    total_latency_ms=0.0,
                )
        # Bind cost tracking around the NL-SQL ladder (enforcing budget or NoOp).
        _analytics_timeout = float(self._config.general.search.analytics_timeout_seconds)
        gate = self._bind_request_cost_budget(rid)
        token = set_request_cost_observer(gate)

        def _with_analytics_llm_cost(result: AnalyticsResult) -> AnalyticsResult:
            """Stamp analytics-phase LLM spend onto the result (request-local gate)."""
            cost = float(gate.running_total_usd)
            if abs(cost - float(getattr(result, 'llm_cost_usd', 0.0) or 0.0)) < 1e-12:
                return result
            return dataclasses.replace(result, llm_cost_usd=cost)

        try:
            try:
                result = await asyncio.wait_for(self._analytics_router.run(question=question, sql_hint=sql_hint, request_id=rid), timeout=_analytics_timeout)
            except asyncio.TimeoutError:
                logger.warning(f"analytics_timeout request_id={rid} timeout_s={_analytics_timeout}")
                return _with_analytics_llm_cost(AnalyticsResult(
                    request_id=rid,
                    question=question,
                    sql_hint=sql_hint,
                    success=False,
                    failure_mode='timeout',
                    failure_reason=f'analytics pipeline exceeded {_analytics_timeout}s budget',
                    pruned_schema=None,
                    generation=None,
                    validation=None,
                    execution=None,
                    verifier=None,
                    total_latency_ms=_analytics_timeout * 1000.0,
                ))
            except CostBudgetExceeded as e:
                # Safety net: call_structured maps budget -> LLMError for degrade
                # paths; if a CostBudgetExceeded still escapes analytics, return
                # typed failure (no regex substrate for NL-SQL).
                snap = gate.snapshot()
                logger.warning(
                    f"analytics_cost_budget_breach request_id={rid} "
                    f"running_total_usd={snap.get('running_total_usd', 0.0):.6f} "
                    f"max_cost_usd={snap.get('max_cost_usd_per_query', 0.0):.6f} "
                    f"call_count={snap.get('call_count', 0)} error={str(e)}"
                )
                return _with_analytics_llm_cost(AnalyticsResult(
                    request_id=rid,
                    question=question,
                    sql_hint=sql_hint,
                    success=False,
                    failure_mode='cost_budget_exceeded',
                    failure_reason=str(e),
                    pruned_schema=None,
                    generation=None,
                    validation=None,
                    execution=None,
                    verifier=None,
                    total_latency_ms=0.0,
                ))
            # Plumb intent_record_id so the analytics_failure FeedbackSignal
            # (emitted by app.py on result.success=False) carries the same
            # join key the search-side IntentRecord exposes. Use
            # dataclasses.replace so we don't depend on AnalyticsRouter
            # threading the id internally.
            if intent_record_id and isinstance(intent_record_id, str):
                result = dataclasses.replace(result, intent_record_id=intent_record_id)
            result = _with_analytics_llm_cost(result)
            logger.info(
                f"analytics_completed request_id={rid} success={result.success} "
                f"failure_mode={result.failure_mode or 'none'} total_latency_ms={result.total_latency_ms:.1f} "
                f"intent_record_id={result.intent_record_id or 'none'} "
                f"llm_cost_usd={float(result.llm_cost_usd):.6f}"
            )
            return result
        finally:
            reset_request_cost_observer(token)
            self._finalize_request_cost_phase(rid, gate)

    @property
    def last_cost_budget(self) -> Optional[RequestCostGate]:
        """Active ``RequestCostGate`` from the task ContextVar, else ``_last_cost_budget``.

        ``running_total_usd`` is the active phase total (search or analytics).
        """
        ctx = _REQUEST_COST_GATE.get()
        if ctx is not None:
            return ctx
        return self._last_cost_budget

    def cache_stats(self) -> Dict[str, Dict[str, int]]:
        """Return per-tier hit/miss counters for observability.
        :return: Dict[str, Dict[str, int]]
        """
        stats = {
            'exact': {'hits': self._exact_cache.hits, 'misses': self._exact_cache.misses},
            'structured': {'hits': self._structured_cache.hits, 'misses': self._structured_cache.misses},
            'intent_plan': {'hits': self._intent_plan_cache.hits, 'misses': self._intent_plan_cache.misses},
        }
        stats.update(self._qi.cache_stats())
        return stats

    def clear_caches(self) -> Dict[str, int]:
        """Invalidate all in-process search cache tiers (full-search path).

        Covers exact / structured / intent_plan result caches, QI intent
        (exact + semantic), NL-SQL, and the explore timeout-fallback slot.
        Does not touch Redis (not provisioned) or the module-level qie_only
        L0 filter cache (cleared separately by POST /cache/clear).

        :return: Dict[str, int] - Per-tier eviction counts
        """
        fb_dropped = 0
        if self._fb_result_cache is not None:
            self._fb_result_cache = None
            fb_dropped = 1
        result = {
            'exact': self._exact_cache.invalidate_all(),
            'structured': self._structured_cache.invalidate_all(),
            'intent_plan': self._intent_plan_cache.invalidate_all(),
            'qi_intent': self._qi.clear_cache(),
            'nl_sql_semantic': (
                self._analytics_router.clear_nl_sql_cache()
                if self._analytics_router is not None
                else 0
            ),
            'explore_fallback': fb_dropped,
        }
        return result
