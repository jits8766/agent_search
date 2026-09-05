"""ZeroResultGuard — four-step relax ladder.

When the orchestrator's primary retrieval returns zero candidates, the guard
walks the deterministic ladder:

    1. widen_filters       (optional) for each eligible numeric range filter,
                           scale the bound by configured multipliers (e.g.
                           price_max=100 -> 200, 500, 1000). First multiplier
                           that yields non-empty wins. Disabled by config flag.
    2. relax_filters       drop one filter at a time per the configured drop
                           priority; re-run retrieval after each drop until
                           non-empty or every filter has been dropped.
    3. semantic_only       drop ALL filters; run vector retrieval only.
    4. explore_fallback    compose the public explore rails into a flat
                           RankedResults so the user never sees a dead end.

The guard owns the ladder state machine; it does NOT own retrieval, fusion, or
the composer — those are injected. This keeps the guard pure-compute (easy to
test) and lets the orchestrator decide whether each step is worth running for a
given query type.
"""
import asyncio
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, List, Optional, Tuple

from semantic_search.config.models import ZeroResultGuardConfig
from semantic_search.contracts import Entity, ExploreRail, IntentSlice, LandingRailResponse, QueryIntent, RankedItem, RankedResults, UserContext, ZeroResultGuardOutcome
from semantic_search.core.exceptions import ValidationError
from semantic_search.core.logging_utils import get_logger
from semantic_search.explore.composer import ExploreComposer
from semantic_search.retrieval.structured_retriever import _FILTER_ENTITY_NAMES

logger = get_logger(__name__)

# "min" suffix slots are tightened by the user as a lower bound; widening
# loosens the bound by DIVIDING by the multiplier (e.g. price_min=100 with
# multiplier 2.0 becomes price_min=50 — a wider net). "max" suffix slots
# (price_max, name_length_max, time_remaining_max, ...) are upper bounds and
# widen by MULTIPLYING. The dispatch is purely string-suffix based since the
# slot vocabulary is disciplined this way.
def _widen_value(slot: str, value: float, multiplier: float) -> float:
    """Return the widened bound for ``slot`` at ``multiplier``.

    :param slot: str - Filter slot name; suffix '_min' or '_max' decides direction
    :param value: float - Current bound
    :param multiplier: float - Widening factor (must be > 1.0)
    :return: float - New bound
    :raises ValidationError: When multiplier is not > 1.0 or slot has no _min/_max suffix
    """
    if multiplier <= 1.0:
        raise ValidationError(f"_widen_value: multiplier must be > 1.0, got {multiplier}")
    if slot.endswith('_max'):
        return float(value) * float(multiplier)
    if slot.endswith('_min'):
        return float(value) / float(multiplier)
    raise ValidationError(f"_widen_value: slot '{slot}' must end in '_min' or '_max'")

# Retrieval callback type: takes a (possibly relaxed) intent and returns its
# ranked-result envelope. The orchestrator wraps its existing pipeline (gather
# → fuse → rank) into a single coroutine matching this signature.
RetrieveFn = Callable[[QueryIntent], Awaitable[RankedResults]]
# Semantic (vector-only) retrieval callback: takes a (filter-stripped) intent and a
# top-k cap, returns query-similarity RankedItems. The orchestrator wraps its
# existing ``_quick_semantic_retrieve`` into this signature so the explore-fallback
# step can fuse "semantic similar to user query" hits with the fresh/urgent rails.
SemanticRetrieveFn = Callable[[QueryIntent, int], Awaitable[List[RankedItem]]]


@dataclass
class ZeroResultGuardResult:
    """Full guard execution outcome.

    The orchestrator stamps :attr:`outcome` onto the response envelope and
    surfaces :attr:`results` to the caller. :attr:`rail_response` is non-None
    iff the explore-fallback step ran (None on relax/semantic success).
    """
    results: RankedResults
    outcome: ZeroResultGuardOutcome
    rail_response: Optional[LandingRailResponse]


class ZeroResultGuard:
    """Three-step relax ladder runner.

    :param config: ZeroResultGuardConfig - Guard config (drop priority, top-k caps)
    :param composer: ExploreComposer - Used for the explore-fallback step
    """

    def __init__(self, config: ZeroResultGuardConfig, composer: ExploreComposer):
        if config is None:
            raise ValidationError("ZeroResultGuard requires a ZeroResultGuardConfig")
        if composer is None:
            raise ValidationError("ZeroResultGuard requires an ExploreComposer")
        self._config = config
        self._composer = composer

    @property
    def enabled(self) -> bool:
        """Master toggle (mirrors `explore.zero_result_guard.enabled`)."""
        return bool(self._config.enabled)

    async def run(self, intent: QueryIntent, retrieve_fn: RetrieveFn, user_context: Optional[UserContext] = None, semantic_retrieve_fn: Optional[SemanticRetrieveFn] = None, explore_fallback_timeout_s: Optional[float] = None, explore_prewarm_task: Optional[asyncio.Task] = None) -> ZeroResultGuardResult:
        """Run the full ladder for an intent whose primary retrieval was empty.

        :param intent: QueryIntent - The original intent (the one whose retrieve
            returned []). Must contain at least one slice.
        :param retrieve_fn: RetrieveFn - Async callable that takes a QueryIntent
            (possibly relaxed) and returns its RankedResults
        :param user_context: Optional[UserContext] - When the caller has an
            authenticated user, the explore-fallback step (Step 3) threads
            ``user_context.user_id`` into the personalized rail so the user
            sees their own signals instead of the anonymous mix. Anonymous
            callers may omit this arg without behavior change.
        :param semantic_retrieve_fn: Optional[SemanticRetrieveFn] - Vector-only
            retrieval callback. When provided, the explore-fallback step (Step 3)
            runs it on the filter-stripped intent for ``semantic_fallback_top_k``
            items and fuses those query-similarity hits into the rail RRF (the
            ``'semantic'`` source). Omitted -> rails-only fallback (unchanged).
        :param explore_fallback_timeout_s: Optional[float] - Hard cap (seconds) for Step 3
            (compose_fallback + semantic_retrieve_fn). When None, falls back to
            ``self._config.explore_fallback_timeout_seconds`` if present, else raises
            ValidationError. Must be > 0.
        :param explore_prewarm_task: Optional[asyncio.Task] - Pre-warmed background task
            returning Optional[RankedResults] (same type as _explore_primary_retrieve).
            When provided and the task result is non-empty, Step 3 consumes it directly
            instead of calling compose_fallback, avoiding redundant CH rail fan-out.
        :return: ZeroResultGuardResult
        :raises ValidationError: When inputs are invalid or the guard is disabled
        """
        if not self.enabled:
            raise ValidationError("ZeroResultGuard is disabled (set explore.zero_result_guard.enabled=true)")
        if intent is None:
            raise ValidationError("ZeroResultGuard.run requires a QueryIntent")
        if retrieve_fn is None:
            raise ValidationError("ZeroResultGuard.run requires a retrieve_fn callable")
        if not intent.slices:
            raise ValidationError("ZeroResultGuard.run requires an intent with at least one slice")
        original_filters = self._collect_filter_names(intent)
        original_count = len(original_filters)
        t0 = time.monotonic()
        # ---- Step 0: widen numeric range filters (when enabled) ------------
        # Walks each eligible numeric slot in the active filter set and tries
        # the configured multipliers in order. Returns on the first widened
        # combination that yields a non-empty result. If no widening rescues
        # the query, falls through to the drop ladder unchanged.
        if (self._config.widen_filters_enabled and self._config.widen_filters_multipliers and original_filters and not intent.conflicts):
            widen_slots = [s for s in self._config.widen_filters_slots if s in original_filters]
            # Build all (slot, multiplier, widened_intent) triples upfront; skip non-wideable.
            _widen_combos: List[tuple] = []
            for _wslot in widen_slots:
                for _wmult in self._config.widen_filters_multipliers:
                    _widened = self._widen_filter(intent, _wslot, float(_wmult))
                    if _widened is not None:
                        _widen_combos.append((_wslot, float(_wmult), _widened))
            if _widen_combos:
                # Fan out all retrievals concurrently; pick first non-empty in original order
                # so the winner is deterministic (smallest slot index, smallest multiplier).
                _widen_tasks = [asyncio.create_task(retrieve_fn(w)) for _, _, w in _widen_combos]
                _widen_results = await asyncio.gather(*_widen_tasks, return_exceptions=True)
                for (_wslot, _wmult, _), _wres in zip(_widen_combos, _widen_results):
                    if isinstance(_wres, BaseException):
                        logger.warning(f"zero_result_guard_widen_retrieve_failed request_id={intent.request_id} slot={_wslot} mult={_wmult:g} error_type={type(_wres).__name__}")
                        continue
                    if len(_wres.items) >= self._config.min_results_before_relax:
                        elapsed_ms = (time.monotonic() - t0) * 1000.0
                        outcome = ZeroResultGuardOutcome(
                            fired=True,
                            ladder_step='widen_filters',
                            original_filter_count=original_count,
                            relaxed_filter_count=original_count,  # widened not dropped
                            relaxation_reason=f'widened:{_wslot}:x{_wmult:g}',
                            dropped_filter_names=[],
                        )
                        logger.info(
                            f"zero_result_guard_rescued request_id={intent.request_id} step=widen_filters "
                            f"slot={_wslot} multiplier={_wmult:g} items={len(_wres.items)} latency_ms={elapsed_ms:.1f}"
                        )
                        return ZeroResultGuardResult(results=_wres, outcome=outcome, rail_response=None)
        # ---- Step 1: relax filters -----------------------------------------
        if original_filters:
            relaxed_intent, dropped, current_filters = self._relax_until_nonempty_prep(intent)
            _protected = frozenset(self._config.relax_filters_protected_slots)
            # Walk the drop priority list; after each drop run retrieval and
            # short-circuit on the first non-empty result.
            # Protected slots are skipped here — they survive until Step 2 (semantic_only).
            for slot in list(self._config.relax_filters_drop_priority):
                if slot not in current_filters:
                    continue
                if slot in _protected:
                    continue
                relaxed_intent = self._drop_filter(relaxed_intent, slot)
                dropped.append(slot)
                current_filters = self._collect_filter_names(relaxed_intent)
                results = await retrieve_fn(relaxed_intent)
                if len(results.items) >= self._config.min_results_before_relax:
                    elapsed_ms = (time.monotonic() - t0) * 1000.0
                    outcome = ZeroResultGuardOutcome(
                        fired=True,
                        ladder_step='relax_filters',
                        original_filter_count=original_count,
                        relaxed_filter_count=len(current_filters),
                        relaxation_reason='zero_after_initial_retrieve',
                        dropped_filter_names=list(dropped),
                    )
                    logger.info(
                        f"zero_result_guard_rescued request_id={intent.request_id} step=relax_filters "
                        f"dropped={dropped} remaining={sorted(current_filters)} items={len(results.items)} latency_ms={elapsed_ms:.1f}"
                    )
                    return ZeroResultGuardResult(results=results, outcome=outcome, rail_response=None)
        # ---- Step 2: semantic-only -----------------------------------------
        semantic_intent = self._drop_all_filters(intent)
        # Step 2 is meaningful only when there were filters to drop (otherwise
        # we'd be re-running the same retrieval the orchestrator already ran).
        if original_count > 0:
            results = await retrieve_fn(semantic_intent)
            if len(results.items) >= self._config.min_results_before_relax:
                elapsed_ms = (time.monotonic() - t0) * 1000.0
                outcome = ZeroResultGuardOutcome(
                    fired=True,
                    ladder_step='semantic_only',
                    original_filter_count=original_count,
                    relaxed_filter_count=0,
                    relaxation_reason='still_zero_after_relax',
                    dropped_filter_names=sorted(original_filters),
                )
                logger.info(f"zero_result_guard_rescued request_id={intent.request_id} step=semantic_only items={len(results.items)} latency_ms={elapsed_ms:.1f}")
                return ZeroResultGuardResult(results=results, outcome=outcome, rail_response=None)
        # ---- Step 3: explore fallback --------------------------------------
        # Forward the authenticated user's id (when present) so the personalized
        # rail composes from their signals; anonymous callers fall through to
        # composer's user-agnostic mix unchanged.
        fallback_user_id: Optional[str] = None
        if user_context is not None and user_context.is_authenticated and user_context.user_id:
            fallback_user_id = user_context.user_id
        # Start semantic retrieval as a concurrent task (when caller provided the fn).
        # compose_fallback will gather rails + asyncio.shield(sem_task) in parallel so
        # neither blocks the other. Both are covered by the single _fb_budget cap.
        _fb_budget = float(explore_fallback_timeout_s) if explore_fallback_timeout_s is not None else float(self._config.explore_fallback_timeout_seconds)
        if _fb_budget <= 0.0:
            raise ValidationError(f"ZeroResultGuard.run: explore_fallback_timeout_s must be > 0, got {_fb_budget}")
        reason = 'still_zero_after_semantic_only' if original_count > 0 else 'zero_after_initial_retrieve'
        if explore_prewarm_task is not None:
            _prewarm_ranked: Optional[RankedResults] = None
            try:
                _prewarm_ranked = await asyncio.shield(explore_prewarm_task)
            except Exception as _pw_e:
                logger.debug(f"zero_result_guard_prewarm_error request_id={intent.request_id} error_type={type(_pw_e).__name__}")
            if _prewarm_ranked is not None and _prewarm_ranked.items:
                elapsed_ms = (time.monotonic() - t0) * 1000.0
                _pw_rail = ExploreRail(rail_id='fallback', title=self._composer.fallback_source_title, cards=[], explanation='', latency_ms=0.0)
                _pw_lr = LandingRailResponse(request_id=intent.request_id, user_id=fallback_user_id, rails=[_pw_rail], generated_at=time.time(), source='zero_result_guard')
                outcome = ZeroResultGuardOutcome(
                    fired=True,
                    ladder_step='explore_fallback',
                    original_filter_count=original_count,
                    relaxed_filter_count=0,
                    relaxation_reason=reason,
                    dropped_filter_names=sorted(original_filters),
                )
                logger.info(
                    f"zero_result_guard_rescued request_id={intent.request_id} step=explore_fallback "
                    f"items={len(_prewarm_ranked.items)} source=prewarm latency_ms={elapsed_ms:.1f}"
                )
                return ZeroResultGuardResult(results=_prewarm_ranked, outcome=outcome, rail_response=_pw_lr)
        _sem_task: Optional[asyncio.Task] = None
        if semantic_retrieve_fn is not None:
            _sem_task = asyncio.create_task(
                semantic_retrieve_fn(semantic_intent, int(self._config.semantic_fallback_top_k))
            )
        # Pass the filter-stripped intent so compose_fallback doesn't re-apply
        # hard filters the entire ladder just exhausted. The explore_fallback
        # step is the last resort; honoring a filter that produced zero results
        # at every prior step defeats the purpose of the fallback.
        try:
            ranked, rail_response = await asyncio.wait_for(
                self._composer.compose_fallback(intent=semantic_intent, max_per_rail_override=int(self._config.explore_fallback_max_per_rail), user_id=fallback_user_id, semantic_items_fut=_sem_task),
                timeout=_fb_budget,
            )
        except asyncio.TimeoutError:
            if _sem_task is not None and not _sem_task.done():
                _sem_task.cancel()
                try:
                    await _sem_task
                except Exception:  # noqa: BLE001 - absorb CancelledError + any backend error
                    pass
            logger.warning(f"zero_result_guard_compose_fallback_timeout request_id={intent.request_id} timeout_s={_fb_budget:.2f}")
            _deg_timeout = float(self._config.explore_fallback_degraded_timeout_seconds)
            try:
                ranked, rail_response = await asyncio.wait_for(self._composer.compose_fallback(intent=semantic_intent, max_per_rail_override=0, user_id=None, semantic_items=None), timeout=_deg_timeout)
            except Exception as _deg_e:  # noqa: BLE001 - last-resort fallback must never raise
                logger.warning(f"zero_result_guard_compose_fallback_degraded_failed request_id={intent.request_id} error_type={type(_deg_e).__name__}")
                _empty_rail = ExploreRail(rail_id='fallback', title=self._composer.fallback_source_title, cards=[], explanation='', latency_ms=0.0)
                rail_response = LandingRailResponse(request_id=intent.request_id, user_id=None, rails=[_empty_rail], generated_at=time.time(), source='zero_result_guard')
                ranked = RankedResults(request_id=intent.request_id, items=[], total_candidates=0, fusion_latency_ms=0.0, cache_hit=None, failure_mode='explore_fallback_timeout')
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        outcome = ZeroResultGuardOutcome(
            fired=True,
            ladder_step='explore_fallback',
            original_filter_count=original_count,
            relaxed_filter_count=0,
            relaxation_reason=reason,
            dropped_filter_names=sorted(original_filters),
        )
        logger.info(
            f"zero_result_guard_rescued request_id={intent.request_id} step=explore_fallback "
            f"items={len(ranked.items)} rails={[r.rail_id for r in rail_response.rails]} latency_ms={elapsed_ms:.1f}"
        )
        return ZeroResultGuardResult(results=ranked, outcome=outcome, rail_response=rail_response)

    # -- Helpers ---------------------------------------------------------------

    @staticmethod
    def _collect_filter_names(intent: QueryIntent) -> set:
        """Return the set of filter slot names currently active across all slices."""
        names: set = set()
        for s in intent.slices:
            for ent in s.entities:
                if ent.name in _FILTER_ENTITY_NAMES:
                    names.add(ent.name)
        return names

    @staticmethod
    def _relax_until_nonempty_prep(intent: QueryIntent) -> Tuple[QueryIntent, List[str], set]:
        """Return (mutable copy of intent, dropped list, current filter set)."""
        return intent, [], ZeroResultGuard._collect_filter_names(intent)

    @staticmethod
    def _widen_filter(intent: QueryIntent, slot: str, multiplier: float) -> Optional[QueryIntent]:
        """Return a copy of ``intent`` with the named slot's numeric bound widened.

        Only acts on slots ending in '_min' or '_max' (per
        :func:`_widen_value`). When the slot's value is non-numeric the widen
        is skipped (returns None) — the caller falls through to the next
        multiplier or the next ladder step.

        :param intent: QueryIntent - Source intent
        :param slot: str - Filter slot to widen
        :param multiplier: float - Factor to widen by (>1.0)
        :return: Optional[QueryIntent] - Widened intent, or None when the slot's
            value is non-numeric (cannot be widened)
        """
        new_slices: List[IntentSlice] = []
        any_widened = False
        for s in intent.slices:
            new_entities: List[Entity] = []
            for ent in s.entities:
                if ent.name != slot:
                    new_entities.append(ent)
                    continue
                try:
                    cur = float(ent.value)
                except (TypeError, ValueError):
                    return None
                widened_val = _widen_value(slot, cur, multiplier)
                # Round to int for slots that are conventionally integer-valued
                # (price in dollars, name_length in chars, bids count, age in
                # years). The retriever's int() coercion would otherwise drop
                # the fractional part silently.
                if isinstance(ent.value, int) or slot.endswith(('_min', '_max')) and 'price' in slot:
                    widened_val = int(round(widened_val))
                new_entities.append(Entity(name=ent.name, value=widened_val, confidence=ent.confidence, source=ent.source, chip_kind=ent.chip_kind))
                any_widened = True
            new_slices.append(IntentSlice(query_type=s.query_type, entities=new_entities, confidence=s.confidence, raw_text=s.raw_text))
        if not any_widened:
            return None
        return QueryIntent(
            request_id=intent.request_id,
            raw_query=intent.raw_query,
            normalized_query=intent.normalized_query,
            query_type=intent.query_type,
            confidence=intent.confidence,
            decision_tier=intent.decision_tier,
            slices=new_slices,
            decision_cost_usd=intent.decision_cost_usd,
            intent_record_id=intent.intent_record_id,
            semantic_query=intent.semantic_query,
            residual_kind=intent.residual_kind,
        )

    @staticmethod
    def _drop_filter(intent: QueryIntent, slot: str) -> QueryIntent:
        """Return a copy of `intent` with every entity named `slot` removed across all slices."""
        new_slices: List[IntentSlice] = []
        for s in intent.slices:
            kept = [e for e in s.entities if e.name != slot]
            new_slices.append(IntentSlice(query_type=s.query_type, entities=kept, confidence=s.confidence, raw_text=s.raw_text))
        return QueryIntent(
            request_id=intent.request_id,
            raw_query=intent.raw_query,
            normalized_query=intent.normalized_query,
            query_type=intent.query_type,
            confidence=intent.confidence,
            decision_tier=intent.decision_tier,
            slices=new_slices,
            decision_cost_usd=intent.decision_cost_usd,
            intent_record_id=intent.intent_record_id,
            semantic_query=intent.semantic_query,
            residual_kind=intent.residual_kind,
        )

    @staticmethod
    def _drop_all_filters(intent: QueryIntent) -> QueryIntent:
        """Return a copy of `intent` with every filter-slot entity removed."""
        new_slices: List[IntentSlice] = []
        for s in intent.slices:
            kept = [e for e in s.entities if e.name not in _FILTER_ENTITY_NAMES]
            new_slices.append(IntentSlice(query_type=s.query_type, entities=kept, confidence=s.confidence, raw_text=s.raw_text))
        return QueryIntent(
            request_id=intent.request_id,
            raw_query=intent.raw_query,
            normalized_query=intent.normalized_query,
            query_type=intent.query_type,
            confidence=intent.confidence,
            decision_tier=intent.decision_tier,
            slices=new_slices,
            decision_cost_usd=intent.decision_cost_usd,
            intent_record_id=intent.intent_record_id,
            semantic_query=intent.semantic_query,
            residual_kind=intent.residual_kind,
        )


__all__ = ['ZeroResultGuard', 'ZeroResultGuardResult', 'RetrieveFn']
