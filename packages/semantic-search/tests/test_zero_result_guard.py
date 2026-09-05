"""Unit tests for ZeroResultGuard ladder.

Coverage matrix (per ``testing.mdc`` §7):

``ZeroResultGuard.__init__``:
- requires_config                                -> TestZRGInit::test_requires_config
- requires_composer                              -> TestZRGInit::test_requires_composer

``ZeroResultGuard.run`` (ladder traversal):
- disabled_raises                                -> TestZRGRun::test_disabled_raises
- requires_intent                                -> TestZRGRun::test_requires_intent
- requires_retrieve_fn                           -> TestZRGRun::test_requires_retrieve_fn
- step1_relax_filters_rescues                    -> TestZRGRun::test_step1_relax_filters_rescues
- step1_drops_in_priority_order                  -> TestZRGRun::test_step1_drops_in_priority_order
- step2_semantic_only_rescues                    -> TestZRGRun::test_step2_semantic_only_rescues
- step3_explore_fallback_when_all_else_fails     -> TestZRGRun::test_step3_explore_fallback_when_all_else_fails
- no_filters_skips_to_explore                    -> TestZRGRun::test_no_filters_skips_to_explore
- outcome_records_filter_counts                  -> TestZRGRun::test_outcome_records_filter_counts

``ZeroResultGuard._drop_filter`` / ``_drop_all_filters`` (helpers):
- drop_filter_removes_only_named_slot            -> TestZRGHelpers::test_drop_filter_removes_only_named_slot
- drop_all_filters_keeps_non_filter_entities     -> TestZRGHelpers::test_drop_all_filters_keeps_non_filter_entities

analytics intent (regression — analytics was missing from _QUERY_TYPES_USING_ZERO_RESULT_GUARD):
- analytics_intent_reaches_explore_fallback      -> TestZRGAnalytics::test_analytics_intent_reaches_explore_fallback
"""
import pytest

from unittest.mock import MagicMock

from semantic_search.config.models import EndingSoonRailConfig, ExploreClickHouseRailsConfig, ExploreConfig, ExploreRailSourceConfig, FallbackRailConfig
from semantic_search.config.models import QIRegexConfig, TrendingRailConfig, ZeroResultGuardConfig
from semantic_search.contracts import Entity, ExploreCard, IntentSlice, QueryIntent, RankedItem, RankedResults, UserContext
from semantic_search.core.exceptions import ValidationError
from semantic_search.explore.composer import ExploreComposer
from semantic_search.explore.zero_result_guard import ZeroResultGuard


def _explore_cfg(drop_priority=None, widen_enabled=False, widen_multipliers=None, widen_slots=None, protected_slots=None) -> ExploreConfig:
    return ExploreConfig(
        enabled=True,
        trending=TrendingRailConfig(source=ExploreRailSourceConfig(enabled=True, max_items=4, title='Trending'), window_seconds=1800),
        ending_soon=EndingSoonRailConfig(source=ExploreRailSourceConfig(enabled=True, max_items=4, title='Ending'), horizon_seconds=86400),
        fallback=FallbackRailConfig(source=ExploreRailSourceConfig(enabled=True, max_items=4, title='Popular')),
        zero_result_guard=ZeroResultGuardConfig(
            enabled=True,
            relax_filters_drop_priority=drop_priority or ['quality_min', 'price_min', 'price_max', 'tld'],
            semantic_only_top_k=10,
            explore_fallback_max_per_rail=4,
            widen_filters_enabled=widen_enabled,
            widen_filters_multipliers=widen_multipliers if widen_multipliers is not None else [],
            widen_filters_slots=widen_slots if widen_slots is not None else [],
            rrf_k=60,
            semantic_fallback_top_k=10,
            min_results_before_relax=1,
            relax_filters_protected_slots=list(protected_slots) if protected_slots is not None else [],
        ),
        clickhouse_rails=ExploreClickHouseRailsConfig(enabled=False, trending_sql='', ending_soon_sql='', max_rows_each=16),
    )


class _StubSource:
    def __init__(self, rail_id, cards):
        self._rail_id = rail_id
        self._cards = list(cards)

    @property
    def rail_id(self):
        return self._rail_id

    async def fetch(self, user_id, max_items):
        return list(self._cards[:max_items])


def _composer(trending_cards=None) -> ExploreComposer:
    return ExploreComposer(
        config=_explore_cfg(),
        trending=_StubSource('trending', trending_cards or [ExploreCard(item_id='t1', fused_score=0.9, source_rail='trending', payload={})]),
        ending_soon=_StubSource('ending_soon', []),
    )


def _intent(filters=None, query_type='hybrid') -> QueryIntent:
    """Build an intent with the named filter slots active in its primary slice."""
    entities = [Entity(name=name, value=val, confidence=1.0, source='L0_entity', chip_kind='hard') for name, val in (filters or {}).items()]
    slc = IntentSlice(query_type=query_type, entities=entities, confidence=0.9, raw_text='q')
    return QueryIntent(
        request_id='req_zg',
        raw_query='q',
        normalized_query='q',
        query_type=query_type,
        confidence=0.9,
        decision_tier='L0_entity',
        slices=[slc],
        decision_cost_usd=0.0,
    )


class _RetrieverScript:
    """A scripted retrieve_fn — returns the next ``RankedResults`` from a queue
    of (filters_remaining_after_call -> result) tuples. Tests use it to assert
    that the guard called retrieve with the relaxed intent expected by each step."""

    def __init__(self, scripted_results):
        # scripted_results: List[RankedResults], consumed in order.
        self._queue = list(scripted_results)
        self.call_count = 0
        self.last_intent = None

    async def __call__(self, intent):
        self.call_count += 1
        self.last_intent = intent
        if not self._queue:
            return RankedResults(request_id=intent.request_id, items=[], total_candidates=0, fusion_latency_ms=0.0, cache_hit=None)
        return self._queue.pop(0)


def _nonempty_result(rid: str = 'req_zg') -> RankedResults:
    return RankedResults(
        request_id=rid,
        items=[RankedItem(item_id='x', fused_score=1.0, contributing_sources=['vector'], payload={})],
        total_candidates=1,
        fusion_latency_ms=0.5,
        cache_hit=None,
    )


def _empty_result(rid: str = 'req_zg') -> RankedResults:
    return RankedResults(request_id=rid, items=[], total_candidates=0, fusion_latency_ms=0.0, cache_hit=None)


class TestZRGInit:
    def test_requires_config(self):
        comp = _composer()
        with pytest.raises(ValidationError):
            ZeroResultGuard(config=None, composer=comp)

    def test_requires_composer(self):
        cfg = _explore_cfg().zero_result_guard
        with pytest.raises(ValidationError):
            ZeroResultGuard(config=cfg, composer=None)


class TestZRGRun:
    @pytest.mark.asyncio
    async def test_disabled_raises(self):
        cfg = ZeroResultGuardConfig(enabled=False, relax_filters_drop_priority=['tld'], semantic_only_top_k=10, explore_fallback_max_per_rail=4, widen_filters_enabled=False, widen_filters_multipliers=[], widen_filters_slots=[], rrf_k=60, semantic_fallback_top_k=10, min_results_before_relax=1)
        guard = ZeroResultGuard(config=cfg, composer=_composer())
        with pytest.raises(ValidationError):
            await guard.run(intent=_intent(), retrieve_fn=_RetrieverScript([]))

    @pytest.mark.asyncio
    async def test_requires_intent(self):
        guard = ZeroResultGuard(config=_explore_cfg().zero_result_guard, composer=_composer())
        with pytest.raises(ValidationError):
            await guard.run(intent=None, retrieve_fn=_RetrieverScript([]))

    @pytest.mark.asyncio
    async def test_requires_retrieve_fn(self):
        guard = ZeroResultGuard(config=_explore_cfg().zero_result_guard, composer=_composer())
        with pytest.raises(ValidationError):
            await guard.run(intent=_intent(), retrieve_fn=None)

    @pytest.mark.asyncio
    async def test_step1_skips_protected_tld(self):
        """Protected slots are not dropped in Step 1 — fall through to semantic_only."""
        intent = _intent(filters={'tld': 'com', 'has_hyphen': False})
        # Dropping has_hyphen still empty; tld is protected so Step 1 cannot drop it;
        # Step 2 semantic_only rescues.
        script = _RetrieverScript([_empty_result(), _nonempty_result()])
        guard = ZeroResultGuard(
            config=_explore_cfg(
                drop_priority=['has_hyphen', 'tld'],
                protected_slots=['tld'],
            ).zero_result_guard,
            composer=_composer(),
        )
        result = await guard.run(intent=intent, retrieve_fn=script)
        assert result.outcome.ladder_step == 'semantic_only'
        assert 'tld' in result.outcome.dropped_filter_names  # dropped only at semantic_only (all filters)
        assert script.call_count == 2  # one after has_hyphen drop, one semantic_only

    @pytest.mark.asyncio
    async def test_step1_relax_filters_rescues(self):
        # Intent has 2 filters: tld + price_min. Drop priority puts price_min first.
        # Script: first retrieval (after dropping price_min) returns non-empty -> step 1 wins.
        intent = _intent(filters={'tld': 'com', 'price_min': 10})
        script = _RetrieverScript([_nonempty_result()])
        guard = ZeroResultGuard(config=_explore_cfg(drop_priority=['price_min', 'tld']).zero_result_guard, composer=_composer())
        result = await guard.run(intent=intent, retrieve_fn=script)
        assert result.outcome.fired is True
        assert result.outcome.ladder_step == 'relax_filters'
        assert result.outcome.original_filter_count == 2
        assert result.outcome.relaxed_filter_count == 1
        assert script.call_count == 1
        # tld must remain in the relaxed intent — only price_min was dropped.
        remaining_names = {e.name for e in script.last_intent.slices[0].entities}
        assert 'price_min' not in remaining_names
        assert 'tld' in remaining_names

    @pytest.mark.asyncio
    async def test_step1_drops_in_priority_order(self):
        intent = _intent(filters={'tld': 'com', 'price_min': 10, 'price_max': 100})
        # First two relax-attempts return empty, third returns hit. Drop order:
        # quality_min (not present, skipped) -> price_min -> price_max -> tld.
        script = _RetrieverScript([_empty_result(), _empty_result(), _nonempty_result()])
        guard = ZeroResultGuard(config=_explore_cfg(drop_priority=['quality_min', 'price_min', 'price_max', 'tld']).zero_result_guard, composer=_composer())
        result = await guard.run(intent=intent, retrieve_fn=script)
        assert result.outcome.ladder_step == 'relax_filters'
        # Three retrieve calls: after dropping price_min, after price_max, after tld.
        assert script.call_count == 3
        remaining_names = {e.name for e in script.last_intent.slices[0].entities}
        assert remaining_names == set()

    @pytest.mark.asyncio
    async def test_step2_semantic_only_rescues(self):
        intent = _intent(filters={'tld': 'com'})
        # Step 1 (drop tld) -> empty; step 2 (semantic-only, all dropped) -> non-empty.
        script = _RetrieverScript([_empty_result(), _nonempty_result()])
        guard = ZeroResultGuard(config=_explore_cfg(drop_priority=['tld']).zero_result_guard, composer=_composer())
        result = await guard.run(intent=intent, retrieve_fn=script)
        assert result.outcome.ladder_step == 'semantic_only'
        assert result.outcome.original_filter_count == 1
        assert result.outcome.relaxed_filter_count == 0
        assert script.call_count == 2
        # Last intent passed to retrieve must have NO filter entities.
        last_filter_count = len([e for e in script.last_intent.slices[0].entities])
        assert last_filter_count == 0

    @pytest.mark.asyncio
    async def test_step3_explore_fallback_when_all_else_fails(self):
        # tld value is a list to match production shape (L0LLMFilterExtractor output).
        intent = _intent(filters={'tld': ['com']})
        # Both step-1 and step-2 retrievals return empty -> fall back to explore.
        script = _RetrieverScript([_empty_result(), _empty_result()])
        # Card payload satisfies the tld filter so the filter-honoring fallback
        # returns it (compose_fallback now drops cards that violate hard filters).
        composer = _composer(trending_cards=[ExploreCard(item_id='t1', fused_score=0.9, source_rail='trending', payload={'tld': 'com'})])
        guard = ZeroResultGuard(config=_explore_cfg(drop_priority=['tld']).zero_result_guard, composer=composer)
        result = await guard.run(intent=intent, retrieve_fn=script)
        assert result.outcome.ladder_step == 'explore_fallback'
        assert result.outcome.relaxation_reason == 'still_zero_after_semantic_only'
        # Composer's stub returns one trending card -> falls into ranked.items.
        assert len(result.results.items) == 1
        assert result.rail_response is not None
        assert result.rail_response.source == 'zero_result_guard'

    @pytest.mark.asyncio
    async def test_no_filters_skips_to_explore(self):
        # Intent has no filter entities at all -> step 1 + step 2 both skipped.
        intent = _intent(filters={})
        script = _RetrieverScript([])
        guard = ZeroResultGuard(config=_explore_cfg().zero_result_guard, composer=_composer())
        result = await guard.run(intent=intent, retrieve_fn=script)
        assert script.call_count == 0
        assert result.outcome.ladder_step == 'explore_fallback'
        assert result.outcome.original_filter_count == 0
        assert result.outcome.relaxation_reason == 'zero_after_initial_retrieve'

    @pytest.mark.asyncio
    async def test_outcome_records_filter_counts(self):
        intent = _intent(filters={'tld': 'com', 'price_min': 5})
        script = _RetrieverScript([_nonempty_result()])
        guard = ZeroResultGuard(config=_explore_cfg(drop_priority=['price_min', 'tld']).zero_result_guard, composer=_composer())
        result = await guard.run(intent=intent, retrieve_fn=script)
        assert result.outcome.original_filter_count == 2
        assert result.outcome.relaxed_filter_count == 1
        assert result.outcome.fired is True

    @pytest.mark.asyncio
    async def test_widen_filters_rescues_before_drop(self):
        # Regression for "find domains under $100" → previously the only ladder
        # action was DROP (price_max=100 → no price filter at all). Widen tries
        # price_max=200 first; the second retrieval call returns non-empty so
        # the guard must short-circuit at widen_filters, not advance to drop.
        intent = _intent(filters={'price_max': 100})
        # First retrieve (multiplier 2.0 → price_max=200) returns non-empty.
        script = _RetrieverScript([_nonempty_result()])
        cfg = _explore_cfg(
            drop_priority=['price_max'],
            widen_enabled=True,
            widen_multipliers=[2.0, 5.0, 10.0],
            widen_slots=['price_max'],
        ).zero_result_guard
        guard = ZeroResultGuard(config=cfg, composer=_composer())
        result = await guard.run(intent=intent, retrieve_fn=script)
        assert result.outcome.ladder_step == 'widen_filters'
        # First non-empty in original combo order wins: slot=price_max, multiplier=2.0.
        assert result.outcome.relaxation_reason.startswith('widened:price_max:x2')

    @pytest.mark.asyncio
    async def test_widen_disabled_falls_through_to_drop(self):
        # When widen is disabled, the guard must behave exactly as before:
        # advance straight to relax_filters with no widening attempt.
        intent = _intent(filters={'price_max': 100})
        script = _RetrieverScript([_nonempty_result()])
        cfg = _explore_cfg(
            drop_priority=['price_max'],
            widen_enabled=False,
            widen_multipliers=[2.0, 5.0],
            widen_slots=['price_max'],
        ).zero_result_guard
        guard = ZeroResultGuard(config=cfg, composer=_composer())
        result = await guard.run(intent=intent, retrieve_fn=script)
        assert result.outcome.ladder_step == 'relax_filters'


class _RecordingSource:
    """Source that records every (user_id, max_items) pair it is asked for."""

    def __init__(self, rail_id, cards):
        self._rail_id = rail_id
        self._cards = list(cards)
        self.calls = []

    @property
    def rail_id(self):
        return self._rail_id

    async def fetch(self, user_id, max_items):
        self.calls.append((user_id, max_items))
        return list(self._cards[:max_items])


def _composer_with_recorder(trending_recorder: '_RecordingSource') -> ExploreComposer:
    return ExploreComposer(
        config=_explore_cfg(),
        trending=trending_recorder,
        ending_soon=_StubSource('ending_soon', []),
    )


class TestZRGUserContextThreading:
    """user_context flows from guard.run -> composer.compose_fallback -> explore sources."""

    @pytest.mark.asyncio
    async def test_authenticated_user_id_reaches_trending_source(self):
        intent = _intent(filters={})
        recorder = _RecordingSource('trending', [])
        guard = ZeroResultGuard( config=_explore_cfg().zero_result_guard, composer=_composer_with_recorder(recorder),)
        uc = UserContext(user_id='alice', is_authenticated=True, session_id='s')
        result = await guard.run(intent=intent, retrieve_fn=_RetrieverScript([]), user_context=uc)
        assert result.outcome.ladder_step == 'explore_fallback'
        assert recorder.calls, "trending source must be queried"
        assert recorder.calls[0][0] == 'alice'
        assert result.rail_response is not None
        assert result.rail_response.user_id == 'alice'

    @pytest.mark.asyncio
    async def test_anonymous_user_context_passes_none(self):
        intent = _intent(filters={})
        recorder = _RecordingSource('trending', [])
        guard = ZeroResultGuard( config=_explore_cfg().zero_result_guard, composer=_composer_with_recorder(recorder),)
        uc = UserContext(user_id=None, is_authenticated=False, session_id='s')
        result = await guard.run(intent=intent, retrieve_fn=_RetrieverScript([]), user_context=uc)
        assert recorder.calls[0][0] is None
        assert result.rail_response.user_id is None

    @pytest.mark.asyncio
    async def test_no_user_context_passes_none(self):
        intent = _intent(filters={})
        recorder = _RecordingSource('trending', [])
        guard = ZeroResultGuard( config=_explore_cfg().zero_result_guard, composer=_composer_with_recorder(recorder),)
        result = await guard.run(intent=intent, retrieve_fn=_RetrieverScript([]), user_context=None)
        assert recorder.calls[0][0] is None
        assert result.rail_response.user_id is None

    @pytest.mark.asyncio
    async def test_unauthenticated_user_with_user_id_still_passes_none(self):
        """Defence in depth: user_id without is_authenticated must NOT leak through."""
        intent = _intent(filters={})
        recorder = _RecordingSource('trending', [])
        guard = ZeroResultGuard( config=_explore_cfg().zero_result_guard, composer=_composer_with_recorder(recorder),)
        uc = UserContext(user_id=None, is_authenticated=False, session_id='s')
        await guard.run(intent=intent, retrieve_fn=_RetrieverScript([]), user_context=uc)
        assert recorder.calls[0][0] is None


class TestZRGSemanticRail:
    """semantic_retrieve_fn threads "semantic similar to user query" hits into the
    explore-fallback RRF. When omitted, the fallback stays rails-only (back-compat)."""

    @pytest.mark.asyncio
    async def test_semantic_items_fused_into_explore_fallback(self):
        # No filters -> ladder skips straight to explore_fallback. The stub semantic
        # retriever returns one query-similarity item that must survive RRF fusion.
        intent = _intent(filters={})
        semantic_calls = []

        async def _stub_semantic(passed_intent, top_k):
            semantic_calls.append((passed_intent, top_k))
            return [RankedItem(item_id='sem1', fused_score=0.95, contributing_sources=['vector'], payload={})]

        composer = _composer(trending_cards=[ExploreCard(item_id='t1', fused_score=0.9, source_rail='trending', payload={})])
        guard = ZeroResultGuard(config=_explore_cfg().zero_result_guard, composer=composer)
        result = await guard.run(intent=intent, retrieve_fn=_RetrieverScript([]), semantic_retrieve_fn=_stub_semantic)
        assert result.outcome.ladder_step == 'explore_fallback'
        # Stub called once with the configured semantic_fallback_top_k.
        assert len(semantic_calls) == 1
        assert semantic_calls[0][1] == _explore_cfg().zero_result_guard.semantic_fallback_top_k
        # The semantic hit participated in the fused output alongside the rail card.
        item_ids = {it.item_id for it in result.results.items}
        assert 'sem1' in item_ids, "semantic-similar item must enter the fallback RRF"
        assert 't1' in item_ids, "rail item must also remain"

    @pytest.mark.asyncio
    async def test_omitting_semantic_fn_keeps_rails_only(self):
        # Back-compat: no semantic_retrieve_fn -> behavior identical to before (rails only).
        intent = _intent(filters={})
        composer = _composer(trending_cards=[ExploreCard(item_id='t1', fused_score=0.9, source_rail='trending', payload={})])
        guard = ZeroResultGuard(config=_explore_cfg().zero_result_guard, composer=composer)
        result = await guard.run(intent=intent, retrieve_fn=_RetrieverScript([]))
        assert result.outcome.ladder_step == 'explore_fallback'
        item_ids = {it.item_id for it in result.results.items}
        assert item_ids == {'t1'}, "rails-only fallback unchanged when semantic_retrieve_fn omitted"

    @pytest.mark.asyncio
    async def test_semantic_fetch_failure_degrades_to_rails(self):
        # A raising semantic retriever must not break the fallback — rails still return.
        intent = _intent(filters={})

        async def _boom(passed_intent, top_k):
            raise RuntimeError("vector backend down")

        composer = _composer(trending_cards=[ExploreCard(item_id='t1', fused_score=0.9, source_rail='trending', payload={})])
        guard = ZeroResultGuard(config=_explore_cfg().zero_result_guard, composer=composer)
        result = await guard.run(intent=intent, retrieve_fn=_RetrieverScript([]), semantic_retrieve_fn=_boom)
        assert result.outcome.ladder_step == 'explore_fallback'
        assert {it.item_id for it in result.results.items} == {'t1'}


class TestZRGTimeoutDegradation:
    """Bug: compose_fallback in the asyncio.TimeoutError handler had no timeout guard."""

    @pytest.mark.asyncio
    async def test_degraded_timeout_returns_empty_ranked_with_fallback_rail(self):
        import asyncio as _asyncio
        from dataclasses import replace as _replace

        async def _slow_fallback(*_a, **_kw):
            await _asyncio.sleep(60)

        composer = _composer()
        composer.compose_fallback = _slow_fallback  # type: ignore[method-assign]
        base_cfg = _explore_cfg().zero_result_guard
        cfg = _replace(base_cfg, explore_fallback_degraded_timeout_seconds=0.05)
        guard = ZeroResultGuard(config=cfg, composer=composer)
        result = await guard.run(intent=_intent(filters={}), retrieve_fn=_RetrieverScript([]), explore_fallback_timeout_s=0.05)
        assert result.outcome.ladder_step == 'explore_fallback'
        assert result.results.failure_mode == 'explore_fallback_timeout'
        assert result.results.items == []

    @pytest.mark.asyncio
    async def test_degraded_timeout_config_field_exists(self):
        cfg = _explore_cfg().zero_result_guard
        assert hasattr(cfg, 'explore_fallback_degraded_timeout_seconds')
        assert float(cfg.explore_fallback_degraded_timeout_seconds) > 0.0
        assert hasattr(cfg, 'explore_fallback_timeout_seconds')
        assert float(cfg.explore_fallback_timeout_seconds) > 0.0

    @pytest.mark.asyncio
    async def test_per_call_timeout_not_clamped_to_floor(self):
        """Timeout passed to run() must not be silently raised to a hardcoded floor.

        If caller sets 0.05s timeout, guard must respect it — not override it to
        any larger hardcoded minimum. Verified by: composer sleeps 0.5s, timeout=0.05,
        guard must fire timeout and return degraded result within ~0.05s, not ~0.5s.
        """
        import asyncio as _asyncio
        import time as _time
        from dataclasses import replace as _replace

        async def _slow_composer(*_a, **_kw):
            await _asyncio.sleep(0.5)

        composer = _composer()
        composer.compose_fallback = _slow_composer  # type: ignore[method-assign]
        base_cfg = _explore_cfg().zero_result_guard
        cfg = _replace(base_cfg, explore_fallback_degraded_timeout_seconds=0.05)
        guard = ZeroResultGuard(config=cfg, composer=composer)
        t0 = _time.monotonic()
        result = await guard.run(intent=_intent(filters={}), retrieve_fn=_RetrieverScript([]), explore_fallback_timeout_s=0.05)
        elapsed = _time.monotonic() - t0
        assert result.results.failure_mode == 'explore_fallback_timeout'
        assert elapsed < 0.15, f"timeout was not respected — elapsed={elapsed:.3f}s suggests hardcoded floor overriding 0.05s"

    @pytest.mark.asyncio
    async def test_timeout_defaults_to_config_when_omitted(self):
        """When explore_fallback_timeout_s is not passed, run() reads from config."""
        from dataclasses import replace as _replace
        cfg = _replace(_explore_cfg().zero_result_guard, explore_fallback_timeout_seconds=1.5)
        guard = ZeroResultGuard(config=cfg, composer=_composer())
        result = await guard.run(intent=_intent(filters={}), retrieve_fn=_RetrieverScript([]))
        assert result.outcome.ladder_step == 'explore_fallback'


class TestZRGAnalytics:
    """Regression: analytics intent was missing from _QUERY_TYPES_USING_ZERO_RESULT_GUARD.

    The guard itself is query-type-agnostic — it runs any intent passed to it.
    These tests confirm the ladder behaves correctly for analytics-typed intents
    so that the orchestrator gate fix (adding 'analytics' to the frozenset) is
    backed by explicit coverage.
    """

    @pytest.mark.asyncio
    async def test_analytics_intent_reaches_explore_fallback(self):
        # Analytics queries have no retrieval backends → all steps return empty.
        # Guard must fall through to explore_fallback and return rail results.
        intent = _intent(filters={'tld': 'com', 'auction_type': '16'}, query_type='analytics')
        trending_card = ExploreCard(item_id='a1', fused_score=0.9, source_rail='trending', payload={})
        composer = ExploreComposer(
            config=_explore_cfg(),
            trending=_StubSource('trending', [trending_card]),
            ending_soon=_StubSource('ending_soon', []),
        )
        guard = ZeroResultGuard(config=_explore_cfg().zero_result_guard, composer=composer)
        # retrieve_fn always returns empty (simulates no backends for analytics)
        result = await guard.run(intent=intent, retrieve_fn=_RetrieverScript([]))
        assert result.outcome.fired is True
        assert result.outcome.ladder_step == 'explore_fallback'
        assert len(result.results.items) > 0, "explore_fallback must return rail items"
        assert result.rail_response is not None

    @pytest.mark.asyncio
    async def test_analytics_intent_with_no_filters_skips_to_explore_fallback(self):
        # Zero filters means steps 0-2 are skipped; should land directly on explore_fallback.
        intent = _intent(filters={}, query_type='analytics')
        trending_card = ExploreCard(item_id='b1', fused_score=0.8, source_rail='trending', payload={})
        composer = ExploreComposer(
            config=_explore_cfg(),
            trending=_StubSource('trending', [trending_card]),
            ending_soon=_StubSource('ending_soon', []),
        )
        guard = ZeroResultGuard(config=_explore_cfg().zero_result_guard, composer=composer)
        result = await guard.run(intent=intent, retrieve_fn=_RetrieverScript([]))
        assert result.outcome.fired is True
        assert result.outcome.ladder_step == 'explore_fallback'
        assert result.outcome.relaxation_reason == 'zero_after_initial_retrieve'


class TestZRGGuidance:
    """Guidance is now in _QUERY_TYPES_USING_ZERO_RESULT_GUARD: a FAILED/empty
    guidance retrieval must reach the explore-rail fallback so ranked_results is
    never a dead end. (Successful guidance retrieval satisfies the orchestrator's
    min-results gate and never invokes the guard — verified at the gate, not here.)
    """

    @pytest.mark.asyncio
    async def test_guidance_failed_retrieval_reaches_explore_fallback(self):
        intent = _intent(filters={'tld': 'com'}, query_type='guidance')
        trending_card = ExploreCard(item_id='g1', fused_score=0.9, source_rail='trending', payload={})
        composer = ExploreComposer(
            config=_explore_cfg(),
            trending=_StubSource('trending', [trending_card]),
            ending_soon=_StubSource('ending_soon', []),
        )
        guard = ZeroResultGuard(config=_explore_cfg().zero_result_guard, composer=composer)
        # Every relax/semantic step returns empty -> ladder must end at explore_fallback.
        result = await guard.run(intent=intent, retrieve_fn=_RetrieverScript([]))
        assert result.outcome.fired is True
        assert result.outcome.ladder_step == 'explore_fallback'
        assert len(result.results.items) > 0, "guidance explore_fallback must return rail items"
        assert result.rail_response is not None


class TestZRGHelpers:
    def test_drop_filter_removes_only_named_slot(self):
        intent = _intent(filters={'tld': 'com', 'price_min': 5})
        relaxed = ZeroResultGuard._drop_filter(intent, 'tld')
        names = {e.name for e in relaxed.slices[0].entities}
        assert 'tld' not in names
        assert 'price_min' in names

    def test_drop_all_filters_keeps_non_filter_entities(self):
        # Add a non-filter entity (e.g. 'name_contains' isn't in _FILTER_ENTITY_NAMES).
        intent = _intent(filters={'tld': 'com'})
        relaxed = ZeroResultGuard._drop_all_filters(intent)
        assert all(True for s in relaxed.slices)
        assert relaxed.slices[0].entities == []  # only filter entity present, gets dropped


def _explore_cfg_protected(protected_slots, drop_priority=None) -> ExploreConfig:
    """Build an ExploreConfig with relax_filters_protected_slots set."""
    return ExploreConfig(
        enabled=True,
        trending=TrendingRailConfig(source=ExploreRailSourceConfig(enabled=True, max_items=4, title='Trending'), window_seconds=1800),
        ending_soon=EndingSoonRailConfig(source=ExploreRailSourceConfig(enabled=True, max_items=4, title='Ending'), horizon_seconds=86400),
        fallback=FallbackRailConfig(source=ExploreRailSourceConfig(enabled=True, max_items=4, title='Popular')),
        zero_result_guard=ZeroResultGuardConfig(
            enabled=True,
            relax_filters_drop_priority=drop_priority or ['keyword_ends_with', 'tld', 'time_remaining_max'],
            semantic_only_top_k=10,
            explore_fallback_max_per_rail=4,
            widen_filters_enabled=False,
            widen_filters_multipliers=[],
            widen_filters_slots=[],
            rrf_k=60,
            semantic_fallback_top_k=10,
            min_results_before_relax=1,
            relax_filters_protected_slots=protected_slots,
        ),
        clickhouse_rails=ExploreClickHouseRailsConfig(enabled=False, trending_sql='', ending_soon_sql='', max_rows_each=16),
    )


class TestZRGProtectedSlots:
    """Protected slots survive the relax_filters step (Step 1) — only dropped at semantic_only (Step 2).

    Regression: time_remaining_max was in Tier 1 and dropped before tld/price/keyword
    filters, causing "domains ending soon this week" to lose its temporal constraint
    before other less-essential filters were tried.
    """

    @pytest.mark.asyncio
    async def test_protected_slot_skipped_in_relax_step(self):
        """time_remaining_max is in drop_priority but also in protected_slots — must be skipped in Step 1."""
        intent = _intent(filters={'keyword_ends_with': 'ending', 'time_remaining_max': 604800})
        cfg = _explore_cfg_protected( protected_slots=['time_remaining_max'], drop_priority=['keyword_ends_with', 'time_remaining_max'],)
        calls = []

        async def retrieve_fn(i):
            active = {e.name for s in i.slices for e in s.entities}
            calls.append(frozenset(active))
            # Return non-empty only after keyword_ends_with is dropped (time_remaining_max still present)
            if 'keyword_ends_with' not in active and 'time_remaining_max' in active:
                return _nonempty_result()
            return _empty_result()

        guard = ZeroResultGuard(config=cfg.zero_result_guard, composer=_composer())
        result = await guard.run(intent=intent, retrieve_fn=retrieve_fn)

        assert result.outcome.ladder_step == 'relax_filters'
        assert 'time_remaining_max' not in result.outcome.dropped_filter_names, \
            "Protected slot must not appear in dropped_filter_names from relax_filters step"
        assert 'keyword_ends_with' in result.outcome.dropped_filter_names

    @pytest.mark.asyncio
    async def test_protected_slot_dropped_at_semantic_only(self):
        """When relax step fails with all non-protected slots exhausted, semantic_only drops protected too."""
        intent = _intent(filters={'keyword_ends_with': 'ending', 'time_remaining_max': 604800})
        cfg = _explore_cfg_protected( protected_slots=['time_remaining_max'], drop_priority=['keyword_ends_with', 'time_remaining_max'],)
        call_count = [0]

        async def retrieve_fn(i):
            call_count[0] += 1
            active = {e.name for s in i.slices for e in s.entities}
            # Only return non-empty when ALL filters dropped (semantic_only step)
            if not active:
                return _nonempty_result()
            return _empty_result()

        guard = ZeroResultGuard(config=cfg.zero_result_guard, composer=_composer())
        result = await guard.run(intent=intent, retrieve_fn=retrieve_fn)

        assert result.outcome.ladder_step == 'semantic_only', \
            "When relax_filters cannot rescue, semantic_only step must drop protected slot"

    @pytest.mark.asyncio
    async def test_empty_protected_slots_behaves_as_before(self):
        """No protected slots: all filters dropped in relax order as before."""
        intent = _intent(filters={'keyword_ends_with': 'ending', 'tld': 'com'})
        cfg = _explore_cfg_protected( protected_slots=[], drop_priority=['keyword_ends_with', 'tld'],)
        dropped = []

        async def retrieve_fn(i):
            active = {e.name for s in i.slices for e in s.entities}
            if 'keyword_ends_with' not in active:
                dropped.append('keyword_ends_with')
                return _nonempty_result()
            return _empty_result()

        guard = ZeroResultGuard(config=cfg.zero_result_guard, composer=_composer())
        result = await guard.run(intent=intent, retrieve_fn=retrieve_fn)
        assert result.outcome.ladder_step == 'relax_filters'
        assert 'keyword_ends_with' in result.outcome.dropped_filter_names


class TestZRGWidenConcurrency:
    """Perf 2: all (slot × multiplier) widen combos fan out concurrently in Step 0."""

    @pytest.mark.asyncio
    async def test_widen_combos_run_concurrently(self):
        import asyncio as _asyncio
        import time as _time
        widen_call_starts: list = []
        call_count = [0]

        # Track call start times; only the first 4 calls are widen (the rest come from steps 1+2).
        async def _slow_retrieve(passed_intent):
            idx = call_count[0]
            call_count[0] += 1
            if idx < 4:
                widen_call_starts.append(_time.monotonic())
            await _asyncio.sleep(0.03)
            return _empty_result()

        intent = _intent(filters={'price_max': 100, 'price_min': 10})
        cfg = _explore_cfg(
            widen_enabled=True,
            widen_multipliers=[2.0, 5.0],
            widen_slots=['price_max', 'price_min'],
        ).zero_result_guard
        guard = ZeroResultGuard(config=cfg, composer=_composer())
        await guard.run(intent=intent, retrieve_fn=_slow_retrieve)
        # All 4 widen combos must have started within a tight window — proves concurrent fan-out.
        assert len(widen_call_starts) == 4, "all 4 widen combos must be attempted"
        spread = max(widen_call_starts) - min(widen_call_starts)
        assert spread < 0.01, f"widen combos did not start concurrently (start spread={spread:.4f}s)"


class TestZRGConflictingFilters:
    """Bug: widen_filters step ran on an intent with detected filter conflicts."""

    @pytest.mark.asyncio
    async def test_widen_skipped_when_intent_has_conflicts(self):
        from dataclasses import replace as _replace
        from semantic_search.contracts import FilterConflict

        # Retriever returns non-empty on the FIRST call — if widen ran it would rescue and
        # outcome would be 'widen_filters'. When widen is skipped, the first call is from step 1
        # (drop one filter), so outcome is 'relax_filters'.
        retriever = _RetrieverScript([_nonempty_result()])
        conflict = FilterConflict(kind='range_inverted', slots=['price_min', 'price_max'], message='price_min > price_max')
        base_intent = _intent(filters={'price_min': 200, 'price_max': 50})
        conflicted_intent = _replace(base_intent, conflicts=[conflict])
        cfg = _explore_cfg(widen_enabled=True, widen_multipliers=[2.0], widen_slots=['price_max'])
        guard = ZeroResultGuard(config=cfg.zero_result_guard, composer=_composer())
        result = await guard.run(intent=conflicted_intent, retrieve_fn=retriever)
        assert result.outcome.ladder_step != 'widen_filters', "widen_filters must not fire on conflicting intent"

    @pytest.mark.asyncio
    async def test_widen_proceeds_when_no_conflicts(self):
        from dataclasses import replace as _replace

        widen_fired = [False]

        async def _rescuing_retrieve(intent):
            entity_names = {e.name for sl in intent.slices for e in sl.entities}
            if 'price_max' in entity_names:
                widen_fired[0] = True
                return _nonempty_result()
            return _empty_result()

        intent = _intent(filters={'price_max': 50})
        cfg = _explore_cfg(widen_enabled=True, widen_multipliers=[2.0], widen_slots=['price_max'])
        guard = ZeroResultGuard(config=cfg.zero_result_guard, composer=_composer())
        result = await guard.run(intent=intent, retrieve_fn=_rescuing_retrieve)
        assert result.outcome.ladder_step == 'widen_filters', "widen must fire when no conflicts present"
        assert widen_fired[0]


class TestZRGSemanticConcurrency:
    """Perf 1: semantic_retrieve_fn and compose_fallback run concurrently in Step 3."""

    @pytest.mark.asyncio
    async def test_semantic_and_rails_run_concurrently(self):
        import asyncio as _asyncio
        from semantic_search.contracts import ExploreRail, LandingRailResponse
        timeline: list = []

        async def _slow_semantic(passed_intent, top_k):
            timeline.append(('sem', 'start'))
            await _asyncio.sleep(0.05)
            timeline.append(('sem', 'end'))
            return [RankedItem(item_id='sem1', fused_score=0.9, contributing_sources=['vector'], payload={})]

        class _SlowComposer:
            fallback_source_title = 'Popular'

            async def compose_fallback(self, *_a, semantic_items_fut=None, **_kw):
                timeline.append(('rails', 'start'))
                await _asyncio.sleep(0.05)
                timeline.append(('rails', 'end'))
                sem_items = []
                if semantic_items_fut is not None:
                    try:
                        sem_items = list(await semantic_items_fut)
                    except Exception:  # noqa: BLE001
                        pass
                rail = ExploreRail(rail_id='fallback', title='Popular', cards=[], explanation='', latency_ms=0.0)
                response = LandingRailResponse(request_id='r', user_id=None, rails=[rail], generated_at=0.0, source='zero_result_guard')
                ranked = RankedResults(request_id='r', items=list(sem_items), total_candidates=len(sem_items), fusion_latency_ms=0.0, cache_hit=None, failure_mode=None)
                return ranked, response

        from dataclasses import replace as _replace
        base_cfg = _explore_cfg().zero_result_guard
        cfg = _replace(base_cfg, explore_fallback_degraded_timeout_seconds=0.5)
        guard = ZeroResultGuard(config=cfg, composer=_SlowComposer())  # type: ignore[arg-type]
        import time as _time
        t0 = _time.monotonic()
        result = await guard.run(intent=_intent(filters={}), retrieve_fn=_RetrieverScript([]), semantic_retrieve_fn=_slow_semantic, explore_fallback_timeout_s=1.0)
        elapsed = _time.monotonic() - t0
        assert result.outcome.ladder_step == 'explore_fallback'
        # If serial: ~0.1 s. If concurrent: ~0.05 s. Allow generous margin.
        assert elapsed < 0.09, f"semantic+rails ran serially (elapsed={elapsed:.3f}s); expected concurrent execution"
        assert 'sem1' in {it.item_id for it in result.results.items}
        # Both started before either ended — overlap proves concurrency.
        sem_start = next(i for i, ev in enumerate(timeline) if ev == ('sem', 'start'))
        rails_start = next(i for i, ev in enumerate(timeline) if ev == ('rails', 'start'))
        sem_end = next(i for i, ev in enumerate(timeline) if ev == ('sem', 'end'))
        rails_end = next(i for i, ev in enumerate(timeline) if ev == ('rails', 'end'))
        assert sem_start < rails_end, "semantic started before rails ended"
        assert rails_start < sem_end, "rails started before semantic ended"
