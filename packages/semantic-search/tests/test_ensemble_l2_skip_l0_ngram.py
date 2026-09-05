"""Phase A: skip L2 when L0 entity voter + ngram agree (consensus_cancel_l2)."""
from __future__ import annotations

import asyncio
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock

from semantic_search.config.models import (
    QIEnsembleConfig,
    QIEnsembleRoutingConfig,
    QIEnsembleVoterConfig,
)
from semantic_search.contracts import Entity, IntentSlice
from semantic_search.qi.ensemble_resolver import EnsembleResolver, Vote
from semantic_search.qi.engine import QIEngine


def _make_minimal_config():
    routing = MagicMock()
    routing.fallback_confidence = 0.30
    routing.l0_fallback_force_hybrid_slots = ['tld', 'price_max', 'auction_type']
    routing.accept_confidence = 0.85
    routing.routing_auto_execute_min = 0.85
    routing.routing_suggest_min = 0.55

    llm_cfg = MagicMock()
    llm_cfg.max_concurrent_l2 = 2
    llm_cfg.tier_3_timeout_seconds = 10.0
    llm_cfg.classify_timeout_seconds = 30.0
    llm_cfg.l1_skip_l2_confidence_threshold = 0.90
    llm_cfg.prompt_tag = 'qi.classify.v19'
    llm_cfg.schema_version = '1'

    regex_cfg = MagicMock()
    regex_cfg.tld_context_match_max_chars = 5
    regex_cfg.tld_word_bare_exclusions = []
    regex_cfg.suppress_for_query_types = ['explore']
    regex_cfg.paired_direction_slots = []
    regex_cfg.known_auction_types = []

    from semantic_search.config.loader import load_config
    from semantic_search.config.models import AgentSearchConfig

    real_qi = AgentSearchConfig.from_dict(load_config()).qi

    cfg = MagicMock()
    cfg.enabled = True
    cfg.default_query_type = 'hybrid'
    cfg.routing = routing
    cfg.llm = llm_cfg
    cfg.regex = regex_cfg
    cfg.residual = None
    cfg.entity_slots = real_qi.entity_slots
    cfg.normalize = real_qi.normalize
    return cfg


def _ensemble(cancel_l2: bool = True, threshold: float = 0.80) -> EnsembleResolver:
    voters = [
        QIEnsembleVoterConfig(voter_id='ngram_gate', weight=2.0, has_veto=False, abstain_on_no_signal=True, timeout_ms=0),
        QIEnsembleVoterConfig(voter_id='entity', weight=2.0, has_veto=False, abstain_on_no_signal=True, timeout_ms=0),
        QIEnsembleVoterConfig(voter_id='semantic', weight=1.5, has_veto=False, abstain_on_no_signal=False, timeout_ms=200),
        QIEnsembleVoterConfig(voter_id='llm', weight=1.0, has_veto=False, abstain_on_no_signal=True, timeout_ms=3000),
    ]
    return EnsembleResolver(
        QIEnsembleConfig(
            voters=voters,
            routing=QIEnsembleRoutingConfig(
                auto_execute_agreement_min=0.70,
                auto_execute_confidence_min=0.75,
                suggest_agreement_min=0.50,
                suggest_confidence_min=0.55,
            ),
            consensus_cancel_l2=cancel_l2,
            consensus_cancel_threshold=threshold,
            extract_before_classify=True,
            fallback_archetype='hybrid',
        ),
        frozenset({'hybrid', 'explore', 'guidance', 'analytics'}),
    )


def _make_engine(
    *,
    ensemble: EnsembleResolver,
    llm: MagicMock,
    ngram_intent: Optional[str],
    entity_vote: Optional[Vote],
    l1_confidence: float = 0.5,
):
    from semantic_search.qi.grounding import EntityGrounder

    cfg = _make_minimal_config()
    grounder = MagicMock(spec=EntityGrounder)
    grounder.ground.side_effect = lambda entities: entities
    circuit_breaker = MagicMock()
    circuit_breaker.allow_request.return_value = True

    engine = QIEngine(
        config=cfg,
        entity_grounder=grounder,
        max_query_length=512,
        circuit_breaker=circuit_breaker,
        llm_classifier=llm,
        ensemble_resolver=ensemble,
    )

    l0_slice = MagicMock()
    l0_slice.entities = [Entity(name='tld', value='com', confidence=0.95, source='L0_llm', chip_kind='hard')]
    l0_slice.soft_entities = []
    engine._entity_extractor = MagicMock()
    engine._entity_extractor.classify_async = AsyncMock(return_value=l0_slice)
    engine._regex_entity_extractor = None

    ngram = MagicMock()
    ngram.classify.return_value = ngram_intent
    engine._ngram_pre_gate = ngram

    entity_voter = MagicMock()
    entity_voter.voter_id = 'entity'
    entity_voter.classify.return_value = entity_vote
    engine._entity_type_voter = entity_voter

    semantic = MagicMock()
    l1 = MagicMock()
    l1.query_type = 'explore'
    l1.confidence = l1_confidence
    semantic.classify.return_value = l1
    engine._semantic_router = semantic
    engine._aggregation_gate = None
    return engine


class TestL0NgramAgreeSkipL2:
    def test_agree_skips_l2(self):
        llm = MagicMock()
        llm.classify = AsyncMock(return_value=None)
        engine = _make_engine(
            ensemble=_ensemble(cancel_l2=True),
            llm=llm,
            ngram_intent='hybrid',
            entity_vote=Vote(voter_id='entity', archetype='hybrid', confidence=0.95),
            l1_confidence=0.5,
        )
        asyncio.run(engine._classify_ensemble('cheap .com domains', 'cheap .com domains', 'req-skip'))
        llm.classify.assert_not_called()

    def test_disagree_runs_l2(self):
        llm = MagicMock()
        llm.classify = AsyncMock(return_value=None)
        engine = _make_engine(
            ensemble=_ensemble(cancel_l2=True),
            llm=llm,
            ngram_intent='explore',
            entity_vote=Vote(voter_id='entity', archetype='hybrid', confidence=0.95),
            l1_confidence=0.5,
        )
        asyncio.run(engine._classify_ensemble('browse domains', 'browse domains', 'req-run'))
        llm.classify.assert_called_once()

    def test_cancel_disabled_runs_l2_even_when_agree(self):
        llm = MagicMock()
        llm.classify = AsyncMock(return_value=None)
        engine = _make_engine(
            ensemble=_ensemble(cancel_l2=False),
            llm=llm,
            ngram_intent='hybrid',
            entity_vote=Vote(voter_id='entity', archetype='hybrid', confidence=0.95),
            l1_confidence=0.5,
        )
        asyncio.run(engine._classify_ensemble('cheap .com', 'cheap .com', 'req-off'))
        llm.classify.assert_called_once()

    def test_helper_agreement_gate(self):
        agree = Vote(voter_id='entity', archetype='hybrid', confidence=0.9)
        assert QIEngine._l0_ngram_agree_skip_l2('hybrid', agree, 0.80) is True
        assert QIEngine._l0_ngram_agree_skip_l2('explore', agree, 0.80) is False
        assert QIEngine._l0_ngram_agree_skip_l2('hybrid', None, 0.80) is False
        assert QIEngine._l0_ngram_agree_skip_l2(None, agree, 0.80) is False
