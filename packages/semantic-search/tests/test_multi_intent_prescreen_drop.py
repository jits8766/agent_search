"""Multi-intent zero-expected drop vs unreliable structured pre-screen."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from semantic_search.config.models import MultiIntentConfig
from semantic_search.contracts import Entity, IntentSlice, QueryIntent, RankedItem, RankedResults
from semantic_search.retrieval.qdrant_adapter import QdrantNoOpStructuredRetriever


def _entity(name: str, value, chip_kind: str = 'hard') -> Entity:
    return Entity(name=name, value=value, confidence=0.9, source='L0_llm', chip_kind=chip_kind)


def _mi_dict(**overrides) -> dict:
    base = dict(
        enabled=True,
        max_sub_intents=5,
        max_split_candidates=10,
        min_sub_query_chars=3,
        weight_confidence=0.5,
        weight_expected_results=0.3,
        weight_specificity=0.2,
        expected_results_norm_cap=100,
        cross_intent_bonus=1.2,
        drop_zero_expected=True,
        sub_intent_failure_policy='fail_soft',
        drop_zero_expected_requires_hard_filters=True,
        drop_zero_expected_skip_unreliable_prescreen=True,
        drop_zero_expected_exempt_query_types=['hybrid'],
        all_slices_dropped_fallback_to_single=True,
        merge_strategy='rrf',
        merge_rrf_k=60,
        collapse_threshold=6,
        top_k_after_collapse=3,
        max_concurrent_slices=2,
        cross_slice_retrieve_propagate_slots=[
            'price_min', 'price_max', 'bids_min', 'bids_max',
            'time_remaining_max', 'days_listed_max',
        ],
        slice_encode_blend_parent_concept=True,
        split_on_l0_keywords=True,
        split_on_l0_keywords_min_terms=2,
    )
    base.update(overrides)
    return base


def _multi_intent() -> QueryIntent:
    q = 'I want to make coffee pizza and sell it under $100'
    slices = [
        IntentSlice(
            query_type='hybrid',
            entities=[_entity('price_max', 99)],
            confidence=0.9,
            raw_text='make coffee pizza',
            slice_id='slc_hybrid',
        ),
        IntentSlice(
            query_type='guidance',
            entities=[_entity('price_max', 99)],
            confidence=0.85,
            raw_text='domain must be below 100',
            slice_id='slc_guidance',
        ),
    ]
    return QueryIntent(
        request_id=QueryIntent.new_request_id(),
        raw_query=q,
        normalized_query=q.lower(),
        query_type='hybrid',
        confidence=0.91,
        decision_tier='L0_multi_intent',
        slices=slices,
        decision_cost_usd=0.0,
        intent_record_id=QueryIntent.new_intent_record_id(),
        residual_kind='semantic',
        semantic_encode_text='coffee pizza',
        semantic_query='coffee pizza',
    )


@pytest.mark.asyncio
async def test_noop_structured_prescreen_keeps_hard_filter_slices():
    """NoOp structured always counts 0 — must not drop hybrid/hard slices."""
    intent = _multi_intent()
    item = RankedItem(
        item_id='coffee1',
        fused_score=0.9,
        contributing_sources=['vector'],
        payload={'domain_name': 'coffeepizza.com'},
    )
    slice_result = RankedResults(
        request_id=intent.request_id,
        items=[item],
        total_candidates=1,
        fusion_latency_ms=0.0,
        cache_hit=None,
    )

    orch = MagicMock()
    orch._config = MagicMock()
    orch._config.multi_intent = MultiIntentConfig.from_dict(_mi_dict())
    orch._config.general.max_results = 50
    orch._structured = QdrantNoOpStructuredRetriever()
    orch._prescreen_structured_count_for_slice = AsyncMock(return_value=0)
    orch._retrieve_and_fuse_slice = AsyncMock(return_value=slice_result)
    orch._merge_slices_rrf = lambda rid, results, cfg: [item]
    orch._build_multi_intent_envelope = MagicMock(return_value=None)
    orch._retrieve_and_rank_single = AsyncMock(
        return_value=RankedResults(
            request_id=intent.request_id, items=[], total_candidates=0, fusion_latency_ms=0.0, cache_hit=None,
        )
    )

    from semantic_search.orchestrator import SearchOrchestrator
    ranked = await SearchOrchestrator._retrieve_and_rank_multi(orch, intent)
    assert len(ranked.items) == 1
    assert ranked.items[0].item_id == 'coffee1'
    orch._retrieve_and_rank_single.assert_not_awaited()
    assert orch._retrieve_and_fuse_slice.await_count >= 1


@pytest.mark.asyncio
async def test_all_slices_dropped_falls_back_to_single_when_configured():
    intent = _multi_intent()
    fallback_item = RankedItem(
        item_id='fb1',
        fused_score=0.8,
        contributing_sources=['vector'],
        payload={'domain_name': 'fallbackcoffee.com'},
    )
    orch = MagicMock()
    orch._config = MagicMock()
    orch._config.multi_intent = MultiIntentConfig.from_dict(
        _mi_dict(
            drop_zero_expected_skip_unreliable_prescreen=False,
            drop_zero_expected_exempt_query_types=[],
            all_slices_dropped_fallback_to_single=True,
        )
    )
    orch._structured = MagicMock()
    orch._structured.provides_inventory_estimate = True
    orch._prescreen_structured_count_for_slice = AsyncMock(return_value=0)
    orch._retrieve_and_fuse_slice = AsyncMock()
    orch._retrieve_and_rank_single = AsyncMock(
        return_value=RankedResults(
            request_id=intent.request_id,
            items=[fallback_item],
            total_candidates=1,
            fusion_latency_ms=0.0,
            cache_hit=None,
        )
    )

    from semantic_search.orchestrator import SearchOrchestrator
    ranked = await SearchOrchestrator._retrieve_and_rank_multi(orch, intent)
    assert len(ranked.items) == 1
    assert ranked.items[0].item_id == 'fb1'
    orch._retrieve_and_rank_single.assert_awaited_once()
    orch._retrieve_and_fuse_slice.assert_not_awaited()


def test_noop_structured_reports_no_inventory_estimate():
    assert QdrantNoOpStructuredRetriever().provides_inventory_estimate is False


def test_build_single_intent_for_slice_propagates_price_and_encode():
    """Each slice wrapper carries sibling price + slice encode for single route."""
    from semantic_search.orchestrator import SearchOrchestrator

    parent = _multi_intent()
    # Price only on guidance slice — hybrid slice should receive propagate.
    parent.slices[0].entities = []
    parent.slices[1].entities = [_entity('price_max', 99)]
    orch = MagicMock()
    orch._config = MagicMock()
    orch._config.multi_intent = MultiIntentConfig.from_dict(_mi_dict())
    orch._config.qi.residual = MagicMock()
    orch._config.qi.residual.navigational_tokens = []
    orch._config.general.search.ranked_results_complement = MagicMock(
        enabled=True,
        strip_temporal_when_clickhouse_available=True,
    )
    orch.analytics_available = True
    wrapper = SearchOrchestrator._build_single_intent_for_slice(orch, parent, parent.slices[0])
    names = {e.name for e in wrapper.slices[0].entities}
    assert 'price_max' in names
    assert 'coffee' in str(wrapper.semantic_encode_text or '').lower() or 'pizza' in str(wrapper.semantic_encode_text or '').lower()
    assert wrapper.query_type == 'hybrid'


def test_slice_encode_blends_parent_concept_into_fragment():
    """Fragment slice ('sell it…') still gets coffee/pizza from parent residual."""
    from semantic_search.orchestrator import SearchOrchestrator

    parent = _multi_intent()
    fragment = IntentSlice(
        query_type='hybrid',
        entities=[_entity('price_max', 99)],
        confidence=0.9,
        raw_text='sell it to everyone',
        slice_id='slc_sell',
    )
    parent.slices = [fragment, parent.slices[1]]
    orch = MagicMock()
    orch._config = MagicMock()
    orch._config.multi_intent = MultiIntentConfig.from_dict(_mi_dict())
    orch._config.qi.residual = MagicMock()
    orch._config.qi.residual.navigational_tokens = []
    orch._config.general.search.ranked_results_complement = MagicMock(
        enabled=True,
        strip_temporal_when_clickhouse_available=True,
    )
    orch.analytics_available = True
    wrapper = SearchOrchestrator._build_single_intent_for_slice(orch, parent, fragment)
    enc = str(wrapper.semantic_encode_text or '').lower()
    assert 'coffee' in enc or 'pizza' in enc


@pytest.mark.asyncio
async def test_retrieve_and_fuse_slice_calls_single_route():
    from semantic_search.orchestrator import SearchOrchestrator

    intent = _multi_intent()
    item = RankedItem(
        item_id='coffee1',
        fused_score=0.9,
        contributing_sources=['vector'],
        payload={'domain_name': 'coffeepizza.com'},
    )
    single_result = RankedResults(
        request_id=intent.request_id,
        items=[item],
        total_candidates=1,
        fusion_latency_ms=0.0,
        cache_hit=None,
    )

    class _Orch:
        pass

    orch = _Orch()
    orch._config = MagicMock()
    orch._config.multi_intent = MultiIntentConfig.from_dict(_mi_dict())
    orch._config.qi.residual = MagicMock()
    orch._config.qi.residual.navigational_tokens = []
    orch._config.general.search.ranked_results_complement = MagicMock(
        enabled=True,
        strip_temporal_when_clickhouse_available=True,
    )
    orch.analytics_available = True
    orch._retrieve_and_rank_single = AsyncMock(return_value=single_result)
    orch._build_single_intent_for_slice = lambda parent, slc: SearchOrchestrator._build_single_intent_for_slice(
        orch, parent, slc,
    )

    ranked = await SearchOrchestrator._retrieve_and_fuse_slice(orch, intent, intent.slices[0])
    assert ranked.items[0].item_id == 'coffee1'
    orch._retrieve_and_rank_single.assert_awaited_once()
    called_intent = orch._retrieve_and_rank_single.await_args.args[0]
    assert called_intent.query_type == 'hybrid'
    assert len(called_intent.slices) == 1
