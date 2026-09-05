"""Model (a): filters stay on original; ANN encode swaps to rewrite when transformed."""
from __future__ import annotations

from unittest.mock import MagicMock

from semantic_search.contracts import Entity, IntentSlice, QueryIntent
from semantic_search.qi.query_transformer import QueryTransformResult


def _intent_with_entities(encode: str = 'messy long original query text here') -> QueryIntent:
    ents = [
        Entity(name='tld', value='com', confidence=0.95, source='L0_llm', chip_kind='hard'),
        Entity(name='price_max', value=50, confidence=0.9, source='L0_llm', chip_kind='hard'),
    ]
    return QueryIntent(
        request_id='req-1',
        raw_query='i really want cheap .com under 50 please',
        normalized_query='i really want cheap .com under 50 please',
        query_type='hybrid',
        confidence=0.9,
        decision_tier='L0_entity',
        slices=[IntentSlice(query_type='hybrid', entities=ents, confidence=0.9, raw_text='i really want cheap .com under 50 please')],
        decision_cost_usd=0.0,
        intent_record_id='ir-1',
        semantic_query=encode,
        semantic_encode_text=encode,
        residual_kind='semantic',
    )


def _orch(
    *,
    encode_from_rewrite: bool = True,
    residual_enabled: bool = False,
    classify_on_transformed_query: bool = False,
):
    """Minimal orchestrator stub exposing rewrite / QI-text helpers."""
    from semantic_search.orchestrator import SearchOrchestrator

    orch = MagicMock(spec=SearchOrchestrator)
    qt = MagicMock()
    qt.encode_from_rewrite = encode_from_rewrite
    qt.classify_on_transformed_query = classify_on_transformed_query
    residual = MagicMock()
    residual.enabled = residual_enabled
    residual.navigational_tokens = []
    qi = MagicMock()
    qi.query_transformer = qt
    qi.residual = residual if residual_enabled else None
    cfg = MagicMock()
    cfg.qi = qi
    orch._config = cfg
    orch._encode_from_rewrite_enabled = SearchOrchestrator._encode_from_rewrite_enabled.__get__(orch)
    orch._apply_encode_from_rewrite = SearchOrchestrator._apply_encode_from_rewrite.__get__(orch)
    orch._classify_on_transformed_query_enabled = (
        SearchOrchestrator._classify_on_transformed_query_enabled.__get__(orch)
    )
    orch._pre_normalized_for_qi = SearchOrchestrator._pre_normalized_for_qi.__get__(orch)
    return orch


class TestEncodeFromRewrite:
    def test_transformed_swaps_encode_keeps_entities(self):
        orch = _orch(encode_from_rewrite=True, residual_enabled=False)
        intent = _intent_with_entities('i really want cheap .com under 50 please')
        tr = QueryTransformResult(
            query='cheap com under 50',
            original_query=intent.normalized_query,
            mode='llm_rewrite',
            transformed=True,
            engine='test',
        )
        out = orch._apply_encode_from_rewrite(intent, tr)
        assert out.semantic_encode_text == 'cheap com under 50'
        assert out.semantic_query == 'cheap com under 50'
        assert [e.name for e in out.slices[0].entities] == ['tld', 'price_max']
        assert out.slices[0].entities[0].value == 'com'
        assert out.slices[0].entities[1].value == 50

    def test_passthrough_leaves_encode(self):
        orch = _orch(encode_from_rewrite=True)
        intent = _intent_with_entities('short query')
        tr = QueryTransformResult(
            query='short query',
            original_query='short query',
            mode='passthrough',
            transformed=False,
            engine='',
        )
        out = orch._apply_encode_from_rewrite(intent, tr)
        assert out.semantic_encode_text == 'short query'

    def test_flag_off_no_swap(self):
        orch = _orch(encode_from_rewrite=False)
        intent = _intent_with_entities('messy original long query')
        tr = QueryTransformResult(
            query='clean keywords',
            original_query=intent.normalized_query,
            mode='llm_rewrite',
            transformed=True,
            engine='test',
        )
        out = orch._apply_encode_from_rewrite(intent, tr)
        assert out.semantic_encode_text == 'messy original long query'


class TestClassifyOnTransformedQuery:
    def test_flag_on_uses_effective(self):
        orch = _orch(classify_on_transformed_query=True)
        assert orch._pre_normalized_for_qi(
            normalized='long messy original',
            effective_normalized='coffee pizza under 100',
        ) == 'coffee pizza under 100'

    def test_flag_off_uses_normalized(self):
        orch = _orch(classify_on_transformed_query=False)
        assert orch._pre_normalized_for_qi(
            normalized='long messy original',
            effective_normalized='coffee pizza under 100',
        ) == 'long messy original'

    def test_passthrough_effective_equals_normalized(self):
        orch = _orch(classify_on_transformed_query=True)
        assert orch._pre_normalized_for_qi(
            normalized='coffee pizza under 100',
            effective_normalized='coffee pizza under 100',
        ) == 'coffee pizza under 100'
