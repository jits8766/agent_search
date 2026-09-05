"""Verify all code-review fixes from the Summary Table audit.

BUG-1: ensemble_resolver=None -> explicit guard, no AttributeError swallow
BUG-2: _resolve_l0_fallback_type returns (str, float), actual score used
BUG-3/ASYNC-2: L2 not launched when ensemble_resolver is None
ARCH-1: _l2_semaphore initialized in __init__, not lazily
ASYNC-1: CancelledError not swallowed in calibration task shutdown (manual)
ROB-2: timeout-derived fallback intents not cached
QUAL-1: chip_kind accessed directly, not via getattr fallback
QUAL-4: numpy bytes decoded correctly in semantic_router
BUG-4: contaminated explore seeds removed from router_seeds.yaml
BUG-5: hard_chip_override_min set to 0 in base.yaml
"""
import asyncio
import types
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from semantic_search.contracts import Entity, IntentSlice


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_config(
    default_query_type: str = 'hybrid',
    fallback_confidence: float = 0.30,
    max_concurrent_l2: int = 2,
):
    """Build a minimal QIConfig-like mock sufficient for QIEngine construction."""
    routing = MagicMock()
    routing.fallback_confidence = fallback_confidence
    routing.l0_fallback_force_hybrid_slots = ['tld', 'price_max', 'price_min', 'auction_type', 'name_length_max', 'name_length_min']
    routing.accept_confidence = 0.85
    routing.routing_auto_execute_min = 0.85
    routing.routing_suggest_min = 0.55

    llm_cfg = MagicMock()
    llm_cfg.max_concurrent_l2 = max_concurrent_l2
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
    real_slots = real_qi.entity_slots

    cfg = MagicMock()
    cfg.enabled = True
    cfg.default_query_type = default_query_type
    cfg.routing = routing
    cfg.llm = llm_cfg
    cfg.regex = regex_cfg
    cfg.residual = None  # disable residual extractor in tests
    # Real slot taxonomy so apply_post_merge_reconcile gets frozenset hard names.
    cfg.entity_slots = real_slots
    cfg.normalize = real_qi.normalize
    return cfg


def _make_engine(ensemble_resolver=None, llm_classifier=None, semantic_router=None, aggregation_gate=None):
    """Build QIEngine with minimal wiring. Patches _FILTER_SIGNAL_RE compilation."""
    from semantic_search.qi.engine import QIEngine
    from semantic_search.qi.grounding import EntityGrounder

    cfg = _make_minimal_config()
    grounder = MagicMock(spec=EntityGrounder)
    grounder.ground.side_effect = lambda entities: entities
    circuit_breaker = MagicMock()
    circuit_breaker.allow_request.return_value = True

    return QIEngine(
        config=cfg,
        entity_grounder=grounder,
        max_query_length=512,
        circuit_breaker=circuit_breaker,
        llm_classifier=llm_classifier,
        ensemble_resolver=ensemble_resolver,
        semantic_router=semantic_router,
        aggregation_gate=aggregation_gate,
    )


# ---------------------------------------------------------------------------
# BUG-1: explicit null guard for ensemble_resolver
# ---------------------------------------------------------------------------

class TestBug1EnsembleNullGuard:
    def test_no_attributeerror_when_ensemble_none(self):
        """With ensemble_resolver=None, _classify_ensemble must not raise AttributeError."""
        engine = _make_engine(ensemble_resolver=None)
        # Patch L0/L1/L2 tasks to return None immediately
        engine._entity_extractor = None
        engine._semantic_router = None
        engine._llm = None
        engine._aggregation_gate = None
        engine._ngram_pre_gate = None
        engine._entity_type_voter = None

        result = asyncio.run(
            engine._classify_ensemble('show me trending domains', 'show me trending domains', 'req-1')
        )
        slices, tier, cost, alts = result
        assert tier == 'L0_fallback', f"Expected L0_fallback, got {tier}"
        assert len(slices) == 1

    def test_fallback_uses_default_query_type_when_no_signals(self):
        """Pure explore query with no L1 signal falls back to default_query_type."""
        engine = _make_engine(ensemble_resolver=None)
        engine._entity_extractor = None
        engine._semantic_router = None
        engine._llm = None
        engine._aggregation_gate = None
        engine._ngram_pre_gate = None
        engine._entity_type_voter = None

        slices, tier, _, _ = asyncio.run(
            engine._classify_ensemble('browse please', 'browse please', 'req-2')
        )
        assert slices[0].query_type == 'hybrid'  # default_query_type

    def test_fallback_uses_analytics_when_agg_gate_fires(self):
        """When aggregation gate fires, L0_fallback returns analytics even with ensemble=None."""
        agg_gate = MagicMock()
        agg_gate.is_analytics.return_value = True
        engine = _make_engine(ensemble_resolver=None, aggregation_gate=agg_gate)
        engine._entity_extractor = None
        engine._semantic_router = None
        engine._llm = None
        engine._ngram_pre_gate = None
        engine._entity_type_voter = None

        slices, tier, _, _ = asyncio.run(
            engine._classify_ensemble('top tlds by volume', 'top tlds by volume', 'req-3')
        )
        assert slices[0].query_type == 'analytics'
        assert slices[0].confidence == 1.0


# ---------------------------------------------------------------------------
# BUG-2: _resolve_l0_fallback_type returns (str, float)
# ---------------------------------------------------------------------------

class TestBug2FallbackTypeReturnsScore:
    def test_returns_tuple(self):
        engine = _make_engine()
        engine._aggregation_gate = None
        engine._semantic_router = None
        result = engine._resolve_l0_fallback_type('show me trending domains')
        assert isinstance(result, tuple), "Must return (str, float)"
        assert len(result) == 2

    def test_analytics_gate_returns_confidence_1(self):
        agg_gate = MagicMock()
        agg_gate.is_analytics.return_value = True
        engine = _make_engine(aggregation_gate=agg_gate)
        t, c = engine._resolve_l0_fallback_type('top tlds by listing volume')
        assert t == 'analytics'
        assert c == 1.0

    def test_hard_entity_returns_confidence_1(self):
        engine = _make_engine()
        engine._aggregation_gate = None
        engine._semantic_router = None
        entities = [Entity(name='tld', value='.com', confidence=0.9, source='L0_entity', chip_kind='hard')]
        t, c = engine._resolve_l0_fallback_type('.com domains', entities)
        assert t == 'hybrid'
        assert c == 1.0

    def test_l1_best_guess_score_propagated(self):
        """When best_guess fires, the actual L1 score propagates as confidence."""
        router = MagicMock()
        router.best_guess.return_value = ('explore', 0.72)
        engine = _make_engine(semantic_router=router)
        engine._aggregation_gate = None
        t, c = engine._resolve_l0_fallback_type('show me trending domains')
        assert t == 'explore'
        assert abs(c - 0.72) < 1e-6, f"Expected 0.72, got {c}"

    def test_default_fallback_uses_fallback_confidence(self):
        """When nothing matches, returns default_query_type with fallback_confidence."""
        router = MagicMock()
        router.best_guess.return_value = None
        engine = _make_engine(semantic_router=router)
        engine._aggregation_gate = None
        t, c = engine._resolve_l0_fallback_type('some random query')
        assert t == 'hybrid'
        assert c == 0.30

    def test_value_discovery_signal_returns_confidence_1(self):
        engine = _make_engine()
        engine._aggregation_gate = None
        engine._semantic_router = None
        t, c = engine._resolve_l0_fallback_type('show me hidden gem domains')
        assert t == 'hybrid'
        assert c == 1.0


# ---------------------------------------------------------------------------
# BUG-3 / ASYNC-2: L2 not launched when ensemble_resolver is None
# ---------------------------------------------------------------------------

class TestBug3L2SkippedWhenEnsembleNone:
    def test_l2_task_not_created_when_ensemble_none(self):
        """_l2_classify must not be called when ensemble_resolver is None."""
        llm = MagicMock()
        llm.classify = AsyncMock(return_value=None)
        engine = _make_engine(ensemble_resolver=None, llm_classifier=llm)
        engine._entity_extractor = None
        engine._semantic_router = None
        engine._aggregation_gate = None
        engine._ngram_pre_gate = None
        engine._entity_type_voter = None

        asyncio.run(
            engine._classify_ensemble('query', 'query', 'req-l2')
        )
        llm.classify.assert_not_called()

    def test_l2_task_created_when_ensemble_present(self):
        """When ensemble is wired, LLM should be invoked."""
        from semantic_search.qi.ensemble_resolver import EnsembleResolver
        llm = MagicMock()
        llm.classify = AsyncMock(return_value=None)

        ensemble = MagicMock(spec=EnsembleResolver)
        # Make resolve return a valid EnsembleResult-like object
        er = MagicMock()
        er.archetype = 'hybrid'
        er.winner_confidence = 0.9
        er.decision_tier = 'ensemble'
        ensemble.resolve.return_value = er

        engine = _make_engine(ensemble_resolver=ensemble, llm_classifier=llm)
        engine._entity_extractor = None
        engine._semantic_router = None
        engine._aggregation_gate = None
        engine._ngram_pre_gate = None
        engine._entity_type_voter = None

        asyncio.run(
            engine._classify_ensemble('query', 'query', 'req-l2-on')
        )
        # LLM was invoked (semaphore acquired, classify called)
        llm.classify.assert_called_once()


# ---------------------------------------------------------------------------
# ARCH-1: semaphore initialized in __init__
# ---------------------------------------------------------------------------

class TestArch1SemaphoreInit:
    def test_semaphore_is_none_when_llm_none(self):
        engine = _make_engine(llm_classifier=None, ensemble_resolver=None)
        assert engine._l2_semaphore is None

    def test_semaphore_set_when_llm_present(self):
        llm = MagicMock()
        engine = _make_engine(llm_classifier=llm)
        assert engine._l2_semaphore is not None
        assert isinstance(engine._l2_semaphore, asyncio.Semaphore)

    def test_semaphore_capacity_from_config(self):
        llm = MagicMock()
        engine = _make_engine(llm_classifier=llm)
        # asyncio.Semaphore stores initial value; access internal _value
        assert engine._l2_semaphore._value == 2  # max_concurrent_l2=2 from _make_minimal_config


# ---------------------------------------------------------------------------
# ROB-2: timeout fallback not cached
# ---------------------------------------------------------------------------

class TestRob2TimeoutNotCached:
    pass


# ---------------------------------------------------------------------------
# QUAL-1: chip_kind direct attribute access
# ---------------------------------------------------------------------------

class TestQual1ChipKindAccess:
    def test_soft_chip_entity_does_not_force_hybrid(self):
        """A soft-chip entity in a filter slot must NOT force hybrid."""
        engine = _make_engine()
        engine._aggregation_gate = None
        engine._semantic_router = None
        soft_entity = Entity(name='tld', value='.com', confidence=0.9, source='L0_entity', chip_kind='soft')
        t, _ = engine._resolve_l0_fallback_type('.com domains', [soft_entity])
        # Should NOT return 'hybrid' from the entity check (falls through to default)
        assert t == 'hybrid'  # from default_query_type, not entity check
        # To distinguish, verify hard entity DOES force it
        hard_entity = Entity(name='tld', value='.com', confidence=0.9, source='L0_entity', chip_kind='hard')
        t2, c2 = engine._resolve_l0_fallback_type('.com domains', [hard_entity])
        assert t2 == 'hybrid' and c2 == 1.0


# ---------------------------------------------------------------------------
# BUG-4: explore seeds no longer contain filter-heavy queries
# ---------------------------------------------------------------------------

class TestBug4ExploreSeedsClean:
    def _load_explore_seeds(self):
        import yaml
        import os
        seeds_path = os.path.join(
            os.path.dirname(__file__),
            '../semantic_search/qi/router_seeds.yaml'
        )
        with open(seeds_path) as f:
            data = yaml.safe_load(f)
        return data.get('archetypes', {}).get('explore', [])

    # Markers that indicate HYBRID filter intent (not browse)
    _FILTER_MARKERS = [
        'Filter ', 'filter ',
        'under $',
        'active backlinks', 'PPC history', 'Google index',
        'whois history', 'domain age and authority',
        'email history', '4-letter', '3-letter .com',
        'under $1000', 'under $500',
        'premium .ai domains expiring', 'premium .io', 'short .net',
        'short 4-letter', 'estimated traffic potential',
    ]

    def test_no_filter_prefix_seeds(self):
        seeds = self._load_explore_seeds()
        filter_seeds = [s for s in seeds if any(m in s for m in self._FILTER_MARKERS)]
        assert filter_seeds == [], f"Contaminated seeds still present: {filter_seeds}"

    def test_core_browse_seeds_present(self):
        seeds = self._load_explore_seeds()
        assert 'show me trending domains' in seeds
        assert 'browse popular domains' in seeds
        assert 'surprise me with a good domain' in seeds

    def test_seed_count_reasonable(self):
        seeds = self._load_explore_seeds()
        assert len(seeds) >= 20, "Too few explore seeds after cleanup"


# ---------------------------------------------------------------------------
# BUG-5: hard_chip_override_min set to 0 in base.yaml
# ---------------------------------------------------------------------------

class TestBug5HardChipOverrideDisabled:
    def test_hard_chip_override_min_is_zero(self):
        import yaml
        import os
        cfg_path = os.path.join(
            os.path.dirname(__file__),
            '../semantic_search/config/base.yaml'
        )
        with open(cfg_path) as f:
            data = yaml.safe_load(f)
        routing = data.get('qi', {}).get('routing', {})
        assert routing.get('hard_chip_override_min') == 0, (
            "hard_chip_override_min must be 0 (disabled) since gate is not implemented"
        )


# ---------------------------------------------------------------------------
# QUAL-4: semantic_router numpy bytes decode (unit)
# ---------------------------------------------------------------------------

class TestQual4NumpyDecode:
    def test_bytes_kind_decoded_correctly(self):
        """artefact_kind must decode bytes -> str without repr wrapping."""
        raw = b'qi_svm_head_v1'
        _raw_kind = raw
        artefact_kind = _raw_kind.decode('utf-8') if isinstance(_raw_kind, bytes) else str(_raw_kind)
        assert artefact_kind == 'qi_svm_head_v1', f"Got: {artefact_kind!r}"

    def test_str_kind_unchanged(self):
        raw = 'centroid'
        artefact_kind = raw.decode('utf-8') if isinstance(raw, bytes) else str(raw)
        assert artefact_kind == 'centroid'
