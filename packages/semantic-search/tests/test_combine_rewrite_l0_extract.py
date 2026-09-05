"""Combined rewrite+L0 extract: schema, token gate, reject/reextract, extract-before-classify."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from semantic_search.config.loader import load_config
from semantic_search.config.models import AgentSearchConfig, QIEnsembleConfig, QIEnsembleRoutingConfig
from semantic_search.contracts import Entity, IntentSlice
from semantic_search.core.exceptions import LLMError
from semantic_search.qi.l0_llm_filter_extractor import (
    L0LLMFilterExtractor,
    _L0CombinedBatchResponse,
    _L0CombinedResultEntry,
    _L0Filter,
    _L0Keyword,
)
from semantic_search.qi.query_transformer import QueryTransformer


def _qi():
    return AgentSearchConfig.from_dict(load_config()).qi


def _l0_extractor(router) -> L0LLMFilterExtractor:
    qi = _qi()
    cfg = qi.l0_llm_entity
    return L0LLMFilterExtractor(
        call_router=router,
        task_type=cfg.task_type,
        entity_slots=qi.entity_slots,
        source_tag=cfg.source_tag,
        max_entities=cfg.max_entities,
        confidence=cfg.confidence,
        enabled=cfg.enabled,
        prompt_tag=cfg.prompt_tag,
        keyword_min_probability=cfg.keyword_min_probability,
        combined_prompt_tag=cfg.combined_prompt_tag,
    )


def test_config_requires_combine_and_ensemble_flags() -> None:
    qi = _qi()
    assert qi.query_transformer is not None
    assert qi.query_transformer.combine_rewrite_with_l0_extract is True
    assert qi.query_transformer.on_rewrite_reject_reextract is True
    assert qi.l0_llm_entity is not None
    assert qi.l0_llm_entity.combined_prompt_tag
    assert qi.l0_llm_entity.combined_schema_version
    assert qi.ensemble is not None
    assert qi.ensemble.extract_before_classify is True


def test_needs_rewrite_token_gate() -> None:
    qi = _qi()
    qt = QueryTransformer.__new__(QueryTransformer)
    qt._config = qi.query_transformer
    qt._signal_res = []
    assert qt.needs_rewrite('a b c d e f g h') is False  # 8 tokens, threshold 8
    assert qt.needs_rewrite('a b c d e f g h i') is True  # 9 tokens


@pytest.mark.asyncio
async def test_extract_priced_combined_parses_rewrite_and_filters() -> None:
    payload = _L0CombinedBatchResponse(
        results=[
            _L0CombinedResultEntry(
                idx=1,
                rewritten_query='coffee pizza under 100',
                transformed=True,
                filters=[_L0Filter(param='maxPrice', value=99)],
                keywords=[_L0Keyword(term='coffee', probability=0.9)],
            )
        ]
    )
    router = MagicMock()
    router.call_structured = AsyncMock(return_value=(payload, {'model': 'test', 'usage': {}}))
    ext = _l0_extractor(router)
    out = await ext.extract_priced_combined(
        'i want to make coffee pizza and sell it under one hundred dollars please'
    )
    assert out.llm_completed is True
    assert out.rewritten_query == 'coffee pizza under 100'
    assert out.model_transformed is True
    assert out.intent_slice is not None
    assert any(e.name == 'price_max' for e in out.intent_slice.entities)
    assert out.keywords and out.keywords[0]['term'] == 'coffee'
    call_kwargs = router.call_structured.await_args.kwargs
    assert call_kwargs['prompt_tag'] == _qi().l0_llm_entity.combined_prompt_tag
    assert call_kwargs['response_schema'] is _L0CombinedBatchResponse


@pytest.mark.asyncio
async def test_extract_priced_combined_raises_llm_error() -> None:
    router = MagicMock()
    router.call_structured = AsyncMock(side_effect=LLMError('boom'))
    ext = _l0_extractor(router)
    with pytest.raises(LLMError):
        await ext.extract_priced_combined('a b c d e f g h i j long enough query')


@pytest.mark.asyncio
async def test_ensemble_extract_before_classify_awaits_l0_before_l1() -> None:
    from semantic_search.qi.engine import QIEngine
    from semantic_search.qi.ensemble_resolver import EnsembleResolver

    order: list[str] = []

    class _L0:
        async def classify_async_priced(self, text: str):
            order.append('l0')
            await asyncio.sleep(0.01)
            return IntentSlice(
                query_type='hybrid',
                entities=[Entity(name='price_max', value=99, confidence=0.9, source='L0_llm')],
                confidence=1.0,
                raw_text=text,
                soft_entities=[],
                keywords=[{'term': 'coffee', 'probability': 0.9}],
            ), 0.001

    class _Router:
        def classify(self, text: str):
            order.append('l1')
            return MagicMock(query_type='hybrid', confidence=0.95)

    from semantic_search.config.models import QIEnsembleVoterConfig
    ens = EnsembleResolver(
        QIEnsembleConfig(
            voters=[
                QIEnsembleVoterConfig(
                    voter_id='semantic', weight=1.5, has_veto=False,
                    abstain_on_no_signal=False, timeout_ms=200,
                ),
            ],
            routing=QIEnsembleRoutingConfig(
                auto_execute_agreement_min=0.70,
                auto_execute_confidence_min=0.75,
                suggest_agreement_min=0.50,
                suggest_confidence_min=0.55,
            ),
            consensus_cancel_l2=False,
            consensus_cancel_threshold=0.80,
            extract_before_classify=True,
            fallback_archetype='hybrid',
        ),
        frozenset({'hybrid', 'explore', 'guidance', 'analytics'}),
    )

    qi_cfg = _qi()
    engine = QIEngine.__new__(QIEngine)
    engine._config = qi_cfg
    engine._entity_extractor = _L0()
    engine._regex_entity_extractor = None
    engine._semantic_router = _Router()
    engine._llm = None
    engine._circuit_breaker = MagicMock(allow_request=MagicMock(return_value=True))
    engine._aggregation_gate = None
    engine._ngram_pre_gate = None
    engine._ensemble_resolver = ens
    engine._entity_type_voter = None
    engine._term_disambiguator = None
    engine._grounder = MagicMock(ground=lambda e: (e, 0))
    # Slot sets helper
    engine._slot_sets = lambda: (
        qi_cfg.entity_slots.soft_slot_set,
        qi_cfg.entity_slots.hard_entity_set,
    )
    engine._should_run_regex_l0_fallback = lambda **k: False
    engine._ensemble_consensus_cancel_l2 = lambda: (False, 1.0)
    engine._ensemble_extract_before_classify = lambda: True
    engine._l2_classify = AsyncMock(return_value=None)
    engine._ground_slices = lambda slices: slices
    engine._log_decision = MagicMock()

    grounded, tier, cost, _alts = await engine._classify_ensemble(
        'coffee under 100', 'coffee under 100', 'rid-1',
    )
    assert order == ['l0', 'l1']
    assert grounded
    assert grounded[0].entities
