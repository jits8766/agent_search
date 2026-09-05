"""Hybrid-first ranked_results + ClickHouse complement / degrade."""
from __future__ import annotations

from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from semantic_search.contracts import Entity, IntentSlice, QueryIntent, RankedItem, RankedResults
from semantic_search.contracts import QUERY_TYPES
from semantic_search.orchestrator import (
    _CH_DEGRADE_QUERY_TYPES,
    _should_restore_rewrite_encode,
    hybrid_degrade_intent,
    hybrid_retrieve_intent,
    strip_time_filters_from_intent,
)
from semantic_search.qi.residual_extractor import semantic_encode_text_for
from semantic_search.qi.slot_to_api_param import is_temporal_entity_slot


def _entity(name: str, value, source: str = 'L0_llm', chip_kind: str = 'hard') -> Entity:
    return Entity(name=name, value=value, confidence=0.9, source=source, chip_kind=chip_kind)


def _intent(query_type: str, entities: Optional[List[Entity]] = None, query: str = 'compare .com and .net for last week') -> QueryIntent:
    ents = entities if entities is not None else [
        _entity('tld', ['com', 'net']),
        _entity('startTimeAfter', '2025-01-08T00:00:00Z'),
        _entity('days_listed_max', 7),
    ]
    return QueryIntent(
        request_id=QueryIntent.new_request_id(),
        raw_query=query,
        normalized_query=query.lower(),
        query_type=query_type,
        confidence=0.95,
        decision_tier='L0_entity',
        slices=[IntentSlice(query_type=query_type, entities=ents, confidence=0.95, raw_text=query)],
        decision_cost_usd=0.0,
        intent_record_id=QueryIntent.new_intent_record_id(),
        alternative_interpretations=[],
        routing_mode='auto_execute',
        semantic_query='compare last week',
        semantic_encode_text='compare last week',
        residual_kind='semantic',
    )


def _complement_cfg(*, merge_rails: bool = True, strip_when_ch_up: bool = True) -> MagicMock:
    cfg = MagicMock()
    cfg.enabled = True
    cfg.merge_explore_rails = merge_rails
    cfg.merge_explore_rails_query_types = ['analytics', 'explore', 'guidance']
    cfg.merge_rrf_k = 60
    cfg.merge_explore_rails_only_when_primary_short = True
    cfg.merge_explore_rails_primary_enough_fraction = 1.0
    cfg.strip_temporal_when_clickhouse_available = strip_when_ch_up
    cfg.ensure_nonempty = True
    cfg.force_semantic_when_empty = True
    cfg.force_semantic_top_k = 10
    return cfg


def test_ch_degrade_types_are_all_non_hybrid_query_types():
    assert _CH_DEGRADE_QUERY_TYPES == (QUERY_TYPES - {'hybrid'})
    assert _CH_DEGRADE_QUERY_TYPES == frozenset({'analytics', 'explore', 'guidance'})


def test_strip_time_filters_keeps_tld():
    intent = _intent('analytics')
    out = strip_time_filters_from_intent(intent)
    names = {e.name for e in out.slices[0].entities}
    assert 'tld' in names
    assert not any(is_temporal_entity_slot(n) for n in names)
    assert out.query_type == 'analytics'


def test_hybrid_retrieve_intent_optional_temporal_strip():
    intent = _intent('analytics')
    kept = hybrid_retrieve_intent(intent, strip_temporal=False)
    assert kept.query_type == 'hybrid'
    assert any(e.name == 'startTimeAfter' for e in kept.slices[0].entities)
    stripped = hybrid_retrieve_intent(intent, strip_temporal=True)
    assert stripped.query_type == 'hybrid'
    assert {e.name for e in stripped.slices[0].entities} == {'tld'}


def test_hybrid_degrade_intent_rewrites_type_and_strips_time():
    intent = _intent('explore')
    out = hybrid_degrade_intent(intent)
    assert out.query_type == 'hybrid'
    assert out.slices[0].query_type == 'hybrid'
    names = {e.name for e in out.slices[0].entities}
    assert names == {'tld'}
    assert intent.query_type == 'explore'
    assert any(e.name == 'startTimeAfter' for e in intent.slices[0].entities)


def test_strip_time_noop_when_no_time_slots():
    intent = _intent('guidance', entities=[_entity('tld', 'com')])
    assert strip_time_filters_from_intent(intent) is intent


@pytest.mark.parametrize('query_type', ['analytics', 'explore', 'guidance'])
def test_ch_down_degrade_contract_per_type(query_type):
    intent = _intent(query_type)
    degrade = hybrid_degrade_intent(intent)
    assert degrade.query_type == 'hybrid'
    assert degrade.slices[0].query_type == 'hybrid'
    assert {e.name for e in degrade.slices[0].entities} == {'tld'}
    assert intent.query_type == query_type


def test_hybrid_retrieve_encode_empty_without_concept_entities():
    intent = _intent('analytics')
    assert semantic_encode_text_for(intent) == 'compare last week'
    out = hybrid_retrieve_intent(intent, strip_temporal=True)
    assert semantic_encode_text_for(out) == ''
    assert out.residual_kind == 'empty'
    assert out.semantic_query is None
    assert intent.semantic_encode_text == 'compare last week'
    assert {e.name for e in out.slices[0].entities} == {'tld'}


def test_hybrid_retrieve_encode_from_concept_entities():
    intent = _intent(
        'analytics',
        entities=[
            _entity('tld', ['com']),
            _entity('days_listed_max', 7),
            _entity('keyword_contains', 'coffee shop', chip_kind='hard'),
        ],
        query='coffee shop .com domains last week',
    )
    out = hybrid_retrieve_intent(intent, strip_temporal=True)
    encode = semantic_encode_text_for(out)
    assert encode == 'coffee shop'
    assert out.residual_kind == 'semantic'
    assert {e.name for e in out.slices[0].entities} == {'tld', 'keyword_contains'}


def test_should_restore_rewrite_encode_archetype_gate() -> None:
    """CH complements skip rewrite encode when concept-empty; hybrid may restore."""
    assert _should_restore_rewrite_encode('hybrid', has_listing_concept=False) is True
    assert _should_restore_rewrite_encode('analytics', has_listing_concept=False) is False
    assert _should_restore_rewrite_encode('explore', has_listing_concept=False) is False
    assert _should_restore_rewrite_encode('guidance', has_listing_concept=False) is False
    assert _should_restore_rewrite_encode('analytics', has_listing_concept=True) is False


def test_hybrid_retrieve_encode_from_soft_entities() -> None:
    """Soft topic/similar_to must drive ANN encode after prepare parks them on soft_entities."""
    intent = QueryIntent(
        request_id=QueryIntent.new_request_id(),
        raw_query='cheap climate tech .com',
        normalized_query='cheap climate tech .com',
        query_type='hybrid',
        confidence=0.95,
        decision_tier='L0_entity',
        slices=[IntentSlice(
            query_type='hybrid',
            entities=[_entity('tld', ['com']), _entity('price_max', 50)],
            confidence=0.95,
            raw_text='cheap climate tech .com',
            soft_entities=[
                _entity('topic_include', ['climate_tech'], chip_kind='soft'),
                _entity('similar_to', ['stripe'], chip_kind='soft'),
            ],
        )],
        decision_cost_usd=0.0,
        intent_record_id=QueryIntent.new_intent_record_id(),
        alternative_interpretations=[],
        routing_mode='auto_execute',
        semantic_query=None,
        semantic_encode_text='',
        residual_kind='empty',
    )
    out = hybrid_retrieve_intent(intent, strip_temporal=False)
    encode = semantic_encode_text_for(out)
    assert 'climate tech' in encode
    assert 'stripe' in encode
    assert out.residual_kind == 'semantic'


def _build_orch(*, intent: QueryIntent, ch_up: bool, hybrid_result: RankedResults, rail_result: Optional[RankedResults] = None):
    from semantic_search.orchestrator import SearchOrchestrator

    cfg = MagicMock()
    cfg.general.max_query_length = 500
    cfg.general.max_results = 50
    cfg.general.search.hybrid_prewarm_enabled = False
    cfg.general.search.explore_fallback_timeout_seconds = 2.0
    cfg.general.search.ranked_results_complement = _complement_cfg(merge_rails=True)
    cfg.cache.enabled = False
    cfg.guidance.enabled = False
    cfg.guidance.snapshot_timeout_seconds = 1.0
    cfg.explore.zero_result_guard.explore_fallback_max_per_rail = 5
    cfg.explore.zero_result_guard.apply_eranker_on_explore_fallback = False
    cfg.retrieval.guard_widen_reapply_hard_gate = True
    cfg.retrieval.enforce_categorical_gate_on_missing_payload = False
    cfg.qi.residual = None

    qi = MagicMock()
    qi.classify = AsyncMock(return_value=intent)

    router = MagicMock()
    router.enabled = ch_up

    called = {}

    async def _retrieve_and_rank(ri, top_k=None):
        called['ri'] = ri
        called['top_k'] = top_k
        return hybrid_result

    orch = SearchOrchestrator.__new__(SearchOrchestrator)
    orch._config = cfg
    orch._qi = qi
    orch._analytics_router = router
    orch._explore_composer = MagicMock()
    orch._explore_composer.enabled = True
    orch._guidance_service = None
    orch._zero_result_guard = None
    orch._exact_cache = MagicMock()
    orch._exact_cache.get.return_value = None
    orch._semantic_cache = None
    orch._structured_cache = None
    orch._intent_plan_cache = None
    orch._spell_corrector = None
    orch._spell_auto_apply = False
    orch._query_transformer = None
    orch._sanitizer = None
    orch._cost_budget_factory = None
    orch._last_cost_budget = None
    orch._last_eranker_outcome = MagicMock()
    orch._last_egress_outcome = None
    orch._egress_guard = None
    orch._noop_guard_outcome = MagicMock(fired=False, ladder_step='none')
    orch._explore_fallback_eranker_outcome = MagicMock(applied=False, client='noop', skipped_reason='explore')
    orch._legacy_bare_cache_eranker_outcome = MagicMock()
    orch._measurement = None
    orch._history_store = None
    orch._signal_store = None
    orch._diversifier = None
    orch._eranker = None
    orch._health = MagicMock()
    orch._degradation = MagicMock()
    orch._vector = None
    orch._structured = None
    orch._sql = None
    orch._fuser = MagicMock()
    orch._brandability = None
    orch._fuzzy = None
    orch._fb_result_cache = None
    # Soft apply required on search(); identity stub for unit harness.
    soft = MagicMock()
    soft.mode = 'off'
    soft.prepare_intent = MagicMock(side_effect=lambda intent: (intent, []))
    soft.apply_rank_boost = MagicMock(side_effect=lambda results, soft_ents: results)
    orch._soft_keyword_applier = soft
    orch._retrieve_and_rank = _retrieve_and_rank
    orch._explore_primary_retrieve = AsyncMock(return_value=rail_result)
    orch._quick_semantic_retrieve = AsyncMock(return_value=[])
    orch._apply_eranker = AsyncMock(side_effect=lambda rid, i, r, u: (r, orch._explore_fallback_eranker_outcome))
    orch._apply_diversifier = AsyncMock(side_effect=lambda r, q, lambda_override=None: r)
    orch._truncate = lambda r, limit=None: r
    orch._apply_egress_guard = AsyncMock(side_effect=lambda r: r)
    orch._with_guidance_envelope = AsyncMock(side_effect=lambda i, r, snapshot_task=None: r)
    orch._maybe_record_history = MagicMock()
    orch._record_observation = MagicMock()
    orch._new_trace = MagicMock(return_value=None)
    orch._sanitize_user_input = MagicMock()
    orch._qi_normalize = lambda q: q.lower()
    orch._preprocess_query = AsyncMock(return_value=(
        intent.normalized_query, intent.normalized_query, intent.raw_query,
        None, None, None, None, 0.0, False, [], [],
    ))
    orch._attach_pre_qi = lambda i, *a: i
    orch._is_multi_intent = lambda i: False
    return orch, called


@pytest.mark.asyncio
async def test_orchestrator_search_analytics_ch_down_calls_retrieve():
    intent = _intent('analytics')
    item = RankedItem(item_id='x1', fused_score=0.8, contributing_sources=['vector'], payload={'domain_name': 'foo.com'})
    hybrid_result = RankedResults(
        request_id=intent.request_id, items=[item], total_candidates=1,
        fusion_latency_ms=0.0, cache_hit=None,
    )
    orch, called = _build_orch(intent=intent, ch_up=False, hybrid_result=hybrid_result)
    ranked, _er, _g = await orch.search(raw_query=intent.raw_query, request_id=intent.request_id, top_k=5)
    assert 'ri' in called
    assert called['ri'].query_type == 'hybrid'
    assert not any(is_temporal_entity_slot(e.name) for e in called['ri'].slices[0].entities)
    assert semantic_encode_text_for(called['ri']) == ''
    assert called['ri'].residual_kind == 'empty'
    assert len(ranked.items) == 1
    assert ranked.query_intent is not None
    assert ranked.query_intent.query_type == 'analytics'
    assert ranked.query_intent.semantic_encode_text == 'compare last week'
    orch._explore_primary_retrieve.assert_not_awaited()


@pytest.mark.asyncio
async def test_orchestrator_search_analytics_ch_up_still_hybrid_retrieve():
    """Analytics retrieve uses hybrid ranked_results when ClickHouse is available."""
    intent = _intent('analytics')
    item = RankedItem(item_id='x1', fused_score=0.8, contributing_sources=['vector'], payload={'domain_name': 'foo.com'})
    hybrid_result = RankedResults(
        request_id=intent.request_id, items=[item], total_candidates=1,
        fusion_latency_ms=0.0, cache_hit=None,
    )
    rail_item = RankedItem(item_id='r1', fused_score=0.5, contributing_sources=['structured'], payload={'domain_name': 'rail.com'})
    rail_result = RankedResults(
        request_id=intent.request_id, items=[rail_item], total_candidates=1,
        fusion_latency_ms=0.0, cache_hit=None,
    )
    orch, called = _build_orch(intent=intent, ch_up=True, hybrid_result=hybrid_result, rail_result=rail_result)
    ranked, _er, _g = await orch.search(raw_query=intent.raw_query, request_id=intent.request_id, top_k=5)
    assert called['ri'].query_type == 'hybrid'
    assert len(ranked.items) >= 1
    assert ranked.failure_mode == 'hybrid_explore_complement'
    assert ranked.query_intent.query_type == 'analytics'
    # Concept-empty complement keeps temporal so period cues scope listing examples.
    assert any(is_temporal_entity_slot(e.name) for e in called['ri'].slices[0].entities)
    assert semantic_encode_text_for(called['ri']) == ''
    assert called['ri'].residual_kind == 'empty'
    orch._explore_primary_retrieve.assert_awaited()


@pytest.mark.asyncio
async def test_orchestrator_explore_ch_up_merges_not_replaces():
    intent = _intent('explore')
    hyb = RankedItem(item_id='h1', fused_score=0.9, contributing_sources=['vector'], payload={'domain_name': 'hybrid.com'})
    rail = RankedItem(item_id='r1', fused_score=0.4, contributing_sources=['structured'], payload={'domain_name': 'rail.com'})
    hybrid_result = RankedResults(request_id=intent.request_id, items=[hyb], total_candidates=1, fusion_latency_ms=0.0, cache_hit=None)
    rail_result = RankedResults(request_id=intent.request_id, items=[rail], total_candidates=1, fusion_latency_ms=0.0, cache_hit=None)
    orch, called = _build_orch(intent=intent, ch_up=True, hybrid_result=hybrid_result, rail_result=rail_result)
    ranked, _er, _g = await orch.search(raw_query=intent.raw_query, request_id=intent.request_id, top_k=5)
    assert called['ri'].query_type == 'hybrid'
    ids = {i.item_id for i in ranked.items}
    assert 'h1' in ids
    assert 'r1' in ids
    assert ranked.failure_mode == 'hybrid_explore_complement'


@pytest.mark.asyncio
async def test_orchestrator_explore_skips_rail_merge_when_vector_has_enough():
    """Rail cards carry no query relevance; merging them once the primary path
    already meets top_k lets a rail-rank artifact bump a real match out. The
    merge must skip entirely when ``len(ranked.items) >= top_k``."""
    intent = _intent('explore')
    hybrid_items = [
        RankedItem(item_id=f'h{i}', fused_score=0.9 - i * 0.01, contributing_sources=['vector'], payload={'domain_name': f'hybrid{i}.com'})
        for i in range(5)
    ]
    rail = RankedItem(item_id='r1', fused_score=0.99, contributing_sources=['sql'], payload={'domain_name': 'rail.com'})
    hybrid_result = RankedResults(request_id=intent.request_id, items=hybrid_items, total_candidates=5, fusion_latency_ms=0.0, cache_hit=None)
    rail_result = RankedResults(request_id=intent.request_id, items=[rail], total_candidates=1, fusion_latency_ms=0.0, cache_hit=None)
    orch, called = _build_orch(intent=intent, ch_up=True, hybrid_result=hybrid_result, rail_result=rail_result)
    ranked, _er, _g = await orch.search(raw_query=intent.raw_query, request_id=intent.request_id, top_k=5)
    ids = {i.item_id for i in ranked.items}
    assert ids == {f'h{i}' for i in range(5)}
    assert 'r1' not in ids
    assert ranked.failure_mode != 'hybrid_explore_complement'


@pytest.mark.asyncio
async def test_ensure_nonempty_force_semantic_when_hybrid_empty():
    intent = _intent('guidance')
    empty = RankedResults(request_id=intent.request_id, items=[], total_candidates=0, fusion_latency_ms=0.0, cache_hit=None)
    sem = RankedItem(item_id='s1', fused_score=0.7, contributing_sources=['vector'], payload={'domain_name': 'sem.com'})
    orch, called = _build_orch(intent=intent, ch_up=False, hybrid_result=empty, rail_result=None)
    orch._quick_semantic_retrieve = AsyncMock(return_value=[sem])
    ranked, _er, _g = await orch.search(raw_query=intent.raw_query, request_id=intent.request_id, top_k=5)
    assert len(ranked.items) == 1
    assert ranked.items[0].item_id == 's1'
    assert ranked.failure_mode == 'force_semantic_nonempty'
