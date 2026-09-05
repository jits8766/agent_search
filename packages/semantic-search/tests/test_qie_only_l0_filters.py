"""qie_only_mode: L0 extract + ground only - no L1/L2 intent classification.

Guarantees:
- extract_l0_filters never touches L1 semantic router or L2 intent LLM
  (100% of L1 + L2 latency saved on qie_only path).
- Full-search _classify_ensemble still runs L1/L2 (unchanged legs).
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from semantic_search.config.loader import load_config
from semantic_search.config.models import AgentSearchConfig, QIEntitySlotsConfig
from semantic_search.contracts import Entity, IntentSlice
from semantic_search.qi import qie_only
from semantic_search.qi.l0_llm_filter_extractor import L0LLMFilterExtractor


def _l0_extractor(call_router):
    qi = AgentSearchConfig.from_dict(load_config()).qi
    cfg = qi.l0_llm_entity
    return L0LLMFilterExtractor(
        call_router=call_router,
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


def _search_request():
    """Minimal Starlette-like request for direct ``app.search`` unit calls."""
    req = MagicMock()
    req.headers = {}
    return req


def _wire_qie_config(sub):
    """Attach real qi.normalize / l0 version tags / max_query_length onto a MagicMock sub."""
    cfg = AgentSearchConfig.from_dict(load_config())
    sub.config.general.max_query_length = cfg.general.max_query_length
    sub.config.qi.normalize = cfg.qi.normalize
    sub.config.identity = cfg.identity
    # Unit tests assert L0 extract on the Form query string; keep transform-before-QI off
    # here so mocks need not implement orchestrator._preprocess_query.
    qt = MagicMock()
    qt.classify_on_transformed_query = False
    sub.config.qi.query_transformer = qt
    search = sub.config.general.search
    search.find_wire = cfg.general.search.find_wire
    l0 = cfg.qi.l0_llm_entity
    sub.config.qi.l0_llm_entity = MagicMock(
        enabled=True,
        task_type=l0.task_type,
        prompt_tag=l0.prompt_tag,
        schema_version=l0.schema_version,
        combined_prompt_tag=l0.combined_prompt_tag,
        combined_schema_version=l0.combined_schema_version,
    )
    return l0


def _make_minimal_config():
    routing = MagicMock()
    routing.fallback_confidence = 0.30
    routing.l0_fallback_force_hybrid_slots = ['tld', 'price_max']
    routing.accept_confidence = 0.85
    routing.routing_auto_execute_min = 0.85
    routing.routing_suggest_min = 0.55

    llm_cfg = MagicMock()
    llm_cfg.max_concurrent_l2 = 2
    llm_cfg.tier_3_timeout_seconds = 10.0
    llm_cfg.classify_timeout_seconds = 30.0
    llm_cfg.l1_skip_l2_confidence_threshold = 0.9
    llm_cfg.prompt_tag = 'qi.classify.v19'
    llm_cfg.schema_version = '1'

    regex_cfg = MagicMock()
    regex_cfg.tld_context_match_max_chars = 5
    regex_cfg.tld_word_bare_exclusions = []
    regex_cfg.suppress_for_query_types = []
    regex_cfg.paired_direction_slots = []
    regex_cfg.known_auction_types = []

    real_qi = AgentSearchConfig.from_dict(load_config()).qi
    cfg = MagicMock()
    cfg.enabled = True
    cfg.default_query_type = 'hybrid'
    cfg.routing = routing
    cfg.llm = llm_cfg
    cfg.regex = regex_cfg
    cfg.residual = None
    cfg.normalize = real_qi.normalize
    cfg.entity_slots = QIEntitySlotsConfig(
        soft_slot_names=['keyword_contains', 'topic_primary'],
        hard_entity_names=['tld', 'price_max', 'buy_it_now'],
        soft_response_key='soft_signals',
        soft_group_tag='SOFT',
        soft_apply_mode='rank',
        soft_rank_boost_weight=0.15,
        soft_rank_slot_names=['keyword_contains'],
        soft_rank_miss_penalty_ratio=0.25,
        soft_rank_partial_boost_ratio=0.5,
    )
    return cfg


def _entity(name: str, value, source: str = 'L0_llm') -> Entity:
    return Entity(name=name, value=value, confidence=0.9, source=source, chip_kind='hard')


def _attach_regex_cfg(regex_extractor, *, fallback_only_when_llm_unavailable: bool = True):
    """MagicMock regex extractors need a real-shaped _config for fallback gating."""
    if regex_extractor is None:
        return None
    cfg = MagicMock()
    cfg.enabled = True
    cfg.fallback_only_when_llm_unavailable = fallback_only_when_llm_unavailable
    cfg.max_entities = 80
    cfg.confidence = 0.9
    cfg.source_tag = 'L0_regex'
    regex_extractor._config = cfg
    return regex_extractor


def _make_engine(
    *,
    entity_extractor=None,
    regex_entity_extractor=None,
    semantic_router=None,
    llm_classifier=None,
    ensemble_resolver=None,
    fallback_only_when_llm_unavailable: bool = True,
):
    from semantic_search.qi.engine import QIEngine
    from semantic_search.qi.grounding import EntityGrounder

    grounder = MagicMock(spec=EntityGrounder)
    grounder.ground.side_effect = lambda entities: list(entities)
    circuit_breaker = MagicMock()
    circuit_breaker.allow_request.return_value = True
    regex_entity_extractor = _attach_regex_cfg(
        regex_entity_extractor,
        fallback_only_when_llm_unavailable=fallback_only_when_llm_unavailable,
    )

    return QIEngine(
        config=_make_minimal_config(),
        entity_grounder=grounder,
        max_query_length=512,
        circuit_breaker=circuit_breaker,
        llm_classifier=llm_classifier,
        semantic_router=semantic_router,
        entity_extractor=entity_extractor,
        regex_entity_extractor=regex_entity_extractor,
        ensemble_resolver=ensemble_resolver,
    )


class TestExtractL0Filters:
    def test_runs_l0_and_regex_grounds_skips_l1_l2(self):
        llm_slice = IntentSlice(
            query_type='hybrid',
            entities=[_entity('tld', ['io'], 'L0_llm'), _entity('price_max', 100, 'L0_llm')],
            confidence=0.9,
            raw_text='cheap .io under 100',
        )
        regex_slice = IntentSlice(
            query_type='hybrid',
            entities=[_entity('tld', ['com'], 'L0_regex')],
            confidence=0.8,
            raw_text='cheap .io under 100',
        )
        entity_extractor = MagicMock()
        entity_extractor.classify_async = AsyncMock(return_value=llm_slice)
        regex_extractor = MagicMock()
        regex_extractor.classify_async = AsyncMock(return_value=regex_slice)

        semantic_router = MagicMock()
        semantic_router.classify = MagicMock(return_value=MagicMock(query_type='explore', confidence=0.99))
        llm_classifier = MagicMock()
        llm_classifier.classify = AsyncMock(side_effect=AssertionError('L2 must not run'))

        engine = _make_engine(
            entity_extractor=entity_extractor,
            regex_entity_extractor=regex_extractor,
            semantic_router=semantic_router,
            llm_classifier=llm_classifier,
        )
        # Fail loud if extract_l0_filters ever delegates to classify / ensemble.
        engine.classify = AsyncMock(side_effect=AssertionError('classify must not run'))
        engine._classify_ensemble = AsyncMock(side_effect=AssertionError('_classify_ensemble must not run'))
        engine._l2_classify = AsyncMock(side_effect=AssertionError('_l2_classify must not run'))
        engine.quick_classify = MagicMock(side_effect=AssertionError('quick_classify must not run'))

        intent = asyncio.run(engine.extract_l0_filters(
            raw_query='cheap .io under 100',
            request_id='req-qie-only',
        ))

        assert intent.decision_tier == 'L0_entity'
        names = {e.name for s in intent.slices for e in s.entities}
        assert 'tld' in names
        assert 'price_max' in names
        tld = next(e for s in intent.slices for e in s.entities if e.name == 'tld')
        assert tld.value == ['io']
        assert tld.source == 'L0_llm'

        entity_extractor.classify_async.assert_awaited()
        # LLM nonempty -> regex must NOT run (sequential fallback only).
        regex_extractor.classify_async.assert_not_awaited()
        semantic_router.classify.assert_not_called()
        llm_classifier.classify.assert_not_called()
        engine.classify.assert_not_awaited()
        engine._classify_ensemble.assert_not_awaited()
        engine._l2_classify.assert_not_awaited()
        engine.quick_classify.assert_not_called()
        engine._grounder.ground.assert_called()
        # Snapshot A: pre-inventory hard chips (response identified).
        # Snapshot B: grounded entities (retrieval + applied_filters).
        for s in intent.slices:
            assert s.pre_ground_entities is not None
            assert {e.name for e in s.pre_ground_entities} >= {'tld', 'price_max'}

    def test_keeps_l0_keywords_without_pre_l0_slice(self):
        """Fresh L0 extract (no combined pre_l0) must keep term/probability keywords.

        Regression: ``_run_l0_extractors`` used to drop ``IntentSlice.keywords``, so
        qie_only reported ``[]`` while full-search ``_classify_ensemble`` kept them.
        Thresholding stays in ``L0LLMFilterExtractor`` (base.yaml 70) — engine copies.
        """
        llm_slice = IntentSlice(
            query_type='hybrid',
            entities=[_entity('tld', ['com'], 'L0_llm')],
            confidence=0.9,
            raw_text='coffee pizza .com under 100',
            keywords=[
                {'term': 'coffee', 'probability': 0.95},
                {'term': 'pizza', 'probability': 0.93},
            ],
        )
        entity_extractor = MagicMock()
        entity_extractor.classify_async = AsyncMock(return_value=llm_slice)

        engine = _make_engine(entity_extractor=entity_extractor)
        intent = asyncio.run(engine.extract_l0_filters(
            raw_query='coffee pizza .com under 100',
            request_id='req-qie-keywords',
        ))

        assert intent.decision_tier == 'L0_entity'
        by_term = {k['term']: k['probability'] for k in intent.keywords}
        assert by_term == {'coffee': 0.95, 'pizza': 0.93}
        for s in intent.slices:
            slice_terms = {k['term']: k['probability'] for k in (s.keywords or [])}
            assert slice_terms == by_term

    def test_pre_l0_keywords_preferred_when_combined_ran(self):
        """When preprocess already ran combined extract, reuse pre_l0 keywords."""
        pre = IntentSlice(
            query_type='hybrid',
            entities=[_entity('tld', ['io'], 'L0_llm')],
            confidence=1.0,
            raw_text='fintech .io',
            keywords=[{'term': 'fintech', 'probability': 0.88}],
        )
        entity_extractor = MagicMock()
        entity_extractor.classify_async = AsyncMock(
            side_effect=AssertionError('fresh L0 must not re-run when pre_l0 completed'),
        )
        engine = _make_engine(entity_extractor=entity_extractor)
        intent = asyncio.run(engine.extract_l0_filters(
            raw_query='verbose fintech domains on .io please',
            request_id='req-pre-l0-kw',
            pre_l0_slice=pre,
            pre_l0_cost_usd=0.01,
            pre_l0_llm_completed=True,
        ))
        assert intent.keywords == [{'term': 'fintech', 'probability': 0.88}]
        entity_extractor.classify_async.assert_not_awaited()

    def test_no_regex_when_llm_empty_success(self):
        """LLM completed with no entities -> regex must NOT run (unavailable-only).

        Cue reconcile may still inject non-regex entities (e.g. price_max from
        "under N") — that is post-merge scrub, not L0_regex pollution.
        """
        entity_extractor = MagicMock()
        entity_extractor.classify_async = AsyncMock(return_value=None)
        regex_extractor = MagicMock()
        regex_extractor.classify_async = AsyncMock(return_value=IntentSlice(
            query_type='hybrid',
            entities=[_entity('tld', ['com'], 'L0_regex'), _entity('price_max', 50, 'L0_regex')],
            confidence=0.85,
            raw_text='.com under 50',
        ))

        engine = _make_engine(
            entity_extractor=entity_extractor,
            regex_entity_extractor=regex_extractor,
            fallback_only_when_llm_unavailable=True,
        )
        intent = asyncio.run(engine.extract_l0_filters(
            raw_query='.com under 50',
            request_id='req-no-regex-empty',
        ))
        assert intent.decision_tier == 'L0_entity'
        regex_extractor.classify_async.assert_not_awaited()
        ents = [e for s in intent.slices for e in s.entities]
        assert all(e.source != 'L0_regex' for e in ents)
        # Regex would have contributed tld; reconcile does not invent tld here.
        assert 'tld' not in {e.name for e in ents}

    def test_regex_fallback_when_llm_unavailable(self):
        """LLM missing/raises -> sequential regex FIND fallback runs."""
        entity_extractor = MagicMock()
        entity_extractor.classify_async = AsyncMock(side_effect=RuntimeError('llm down'))
        regex_extractor = MagicMock()
        regex_extractor.classify_async = AsyncMock(return_value=IntentSlice(
            query_type='hybrid',
            entities=[_entity('tld', ['com'], 'L0_regex'), _entity('price_max', 50, 'L0_regex')],
            confidence=0.85,
            raw_text='.com under 50',
        ))

        engine = _make_engine(
            entity_extractor=entity_extractor,
            regex_entity_extractor=regex_extractor,
            fallback_only_when_llm_unavailable=True,
        )
        intent = asyncio.run(engine.extract_l0_filters(
            raw_query='.com under 50',
            request_id='req-regex-fb',
        ))
        assert intent.decision_tier == 'L0_entity'
        by_name = {e.name: e for s in intent.slices for e in s.entities}
        assert by_name['tld'].source == 'L0_regex'
        assert by_name['price_max'].value == 50
        regex_extractor.classify_async.assert_awaited()

    def test_does_not_consult_intent_caches(self):
        cache = MagicMock()
        engine = _make_engine()
        engine._intent_result_cache = cache
        engine._semantic_intent_cache = cache
        engine._entity_extractor = None
        engine._regex_entity_extractor = None

        intent = asyncio.run(engine.extract_l0_filters(
            raw_query='anything',
            request_id='req-nocache',
        ))
        cache.get.assert_not_called()
        cache.put.assert_not_called()
        assert intent.decision_tier == 'L0_entity'


class TestFullSearchEnsembleUnchanged:
    """Normal classify path still runs L1 (+ L2 when threshold unmet)."""

    def test_ensemble_still_invokes_l1(self):
        semantic_router = MagicMock()
        semantic_router.classify = MagicMock(
            return_value=MagicMock(query_type='hybrid', confidence=0.95),
        )
        ensemble = MagicMock()
        er = MagicMock()
        er.archetype = 'hybrid'
        er.winner_confidence = 0.95
        er.decision_tier = 'L1_semantic'
        ensemble.resolve.return_value = er

        engine = _make_engine(
            semantic_router=semantic_router,
            ensemble_resolver=ensemble,
            llm_classifier=None,
        )
        engine._entity_extractor = None
        engine._regex_entity_extractor = None
        engine._aggregation_gate = None
        engine._ngram_pre_gate = None
        engine._entity_type_voter = None

        slices, tier, _, _ = asyncio.run(
            engine._classify_ensemble('cheap .io', 'cheap .io', 'req-full'),
        )
        semantic_router.classify.assert_called_once_with('cheap .io')
        assert tier == 'L1_semantic'
        assert len(slices) == 1


class TestOrchestratorExtractFiltersOnly:
    def test_extract_filters_only_does_not_call_classify(self):
        from semantic_search.orchestrator import SearchOrchestrator

        orch = SearchOrchestrator.__new__(SearchOrchestrator)
        orch._sanitize_user_input = MagicMock()
        orch._new_trace = MagicMock(return_value=None)
        orch._qi_normalize = MagicMock(return_value='normalized')
        orch._config = MagicMock()
        orch._config.general.search.overlap_preprocess_with_classify = False
        orch._preprocess_query = AsyncMock(return_value=(
            'normalized', 'normalized', 'raw', None, None, None,
            None, 0.0, False, [], [],
        ))
        orch._attach_pre_qi = MagicMock(side_effect=lambda intent, *a, **k: intent)
        orch._apply_encode_from_rewrite = MagicMock(side_effect=lambda intent, *a, **k: intent)
        qi = MagicMock()
        qi.extract_l0_filters = AsyncMock(return_value=MagicMock(
            decision_tier='L0_entity', slices=[],
        ))
        qi.classify = AsyncMock(side_effect=AssertionError('classify must not run on qie_only'))
        orch._qi = qi

        intent = asyncio.run(orch.extract_filters_only(
            raw_query='.io under 100', request_id='rid-1',
        ))

        qi.extract_l0_filters.assert_awaited_once()
        qi.classify.assert_not_awaited()
        assert intent.decision_tier == 'L0_entity'


# Slim qie_only JSON contract — keep in sync with app.search qie_only branch.
_QIE_ONLY_ALLOWED_KEYS = frozenset({
    'query',
    'request_id',
    'search_id',
    'answer_mode',
    'latency_ms',
    'decision_tier',
    'decision_cost_usd',
    'identified_filters',
    'pre_ground_identified',
    'keywords',
    'grounded_drop_count',
    'prompt_tag',
    'schema_version',
    'find_query_params',
    'find_query_string',
    'soft_chips',
    'find_skipped',
    'query_transform',
})
_QIE_ONLY_FORBIDDEN_KEYS = frozenset({
    'applied_filters',
    'grounding_applied',
    'query_intelligence',
    'pipeline_trace',
    'ranked_results',
    'retrieval_metrics',
    'analytics',
    'guidance',
    'guard_notice',
    'did_you_mean',
})


def _qie_intent_from_entities(
    *,
    entities,
    soft_entities=None,
    keywords=None,
    query_transform=None,
    decision_cost_usd: float = 0.01,
    raw_text: str = '',
):
    """Build a QueryIntent shaped like extract_filters_only output."""
    from semantic_search.contracts import QueryIntent

    soft_entities = soft_entities or []
    keywords = keywords or []
    slice_ = IntentSlice(
        query_type='hybrid',
        entities=list(entities),
        confidence=0.9,
        raw_text=raw_text,
        soft_entities=list(soft_entities),
        pre_ground_entities=list(entities),
        keywords=list(keywords),
    )
    return QueryIntent(
        request_id='rid-qie',
        raw_query=raw_text,
        normalized_query=raw_text,
        query_type='hybrid',
        confidence=0.9,
        decision_tier='L0_entity',
        slices=[slice_],
        decision_cost_usd=decision_cost_usd,
        keywords=list(keywords),
        query_transform=query_transform,
    )


class TestQieOnlyResponseJson:
    def test_slim_json_keys_l0_llm_extract(self):
        """qie_only_mode=true uses shared extract_filters_only pipeline."""
        from semantic_search import app as app_module

        qie_only._get_qie_l0_filter_cache().clear()
        soft_kw = Entity(
            name='keyword_contains', value='io', confidence=0.9,
            source='L0_llm', chip_kind='soft',
        )
        intent = _qie_intent_from_entities(
            entities=[
                _entity('tld', ['io'], 'L0_llm'),
                # price_max 100 -> public maxPrice after entities_to_identified
                # (exclusive under cues are applied upstream; unit uses grounded 99)
                _entity('price_max', 99, 'L0_llm'),
            ],
            soft_entities=[soft_kw],
            raw_text='cheap .io under 100',
            decision_cost_usd=0.012,
        )

        orch = MagicMock()
        orch.extract_filters_only = AsyncMock(return_value=intent)
        orch.search = AsyncMock(side_effect=AssertionError('full search must not run'))

        search_cfg = MagicMock()
        search_cfg.qie_only_mode = False
        search_cfg.top_k_cap = 100

        regex_cfg = MagicMock()
        regex_cfg.enabled = True
        regex_cfg.fallback_only_when_llm_unavailable = True

        sub = MagicMock()
        sub.config.general.search = search_cfg
        _wire_qie_config(sub)
        sub.config.qi.l0_regex_entity = regex_cfg
        sub.sanitizer = None
        sub.call_router = MagicMock()
        sub.orchestrator = orch
        sub.qi_engine = MagicMock()
        sub.qi_engine._entity_extractor = MagicMock()
        sub.qi_engine._regex_entity_extractor = None
        sub.qi_engine._grounder = None
        sub.qi_engine.quick_classify = MagicMock(
            side_effect=AssertionError('L1 preview must not run on qie_only'),
        )

        with patch.object(app_module, '_require_subsystems', return_value=sub):
            body = asyncio.run(app_module.search(
                request=_search_request(),
                query='cheap .io under 100',
                top_k=50,
                diversity_lambda=0.9,
                relevance_threshold=0.7,
                qie_only_mode=True,
                x_session_id=None,
            ))

        assert body['answer_mode'] == 'qie_only'
        assert body['decision_tier'] == 'L0_entity'
        assert 'query_transform' not in body
        assert isinstance(body['identified_filters'], list)
        assert isinstance(body.get('pre_ground_identified'), list)
        by_name = {e['name']: e['value'] for e in body['identified_filters']}
        assert by_name.get('tldIncludeList') == ['io']
        assert by_name.get('maxPrice') == 99
        assert by_name.get('keyword_contains') == 'io'
        assert all(e.get('source') == 'L0_llm' for e in body['identified_filters'])
        assert body.get('prompt_tag') == sub.config.qi.l0_llm_entity.prompt_tag
        assert body.get('schema_version') == sub.config.qi.l0_llm_entity.schema_version
        assert isinstance(body.get('grounded_drop_count'), int)
        assert isinstance(body.get('find_query_params'), dict)
        assert isinstance(body.get('find_query_string'), str)
        assert set(body.keys()) <= _QIE_ONLY_ALLOWED_KEYS
        for forbidden in _QIE_ONLY_FORBIDDEN_KEYS:
            assert forbidden not in body, f'qie_only JSON must not include {forbidden}'
        orch.extract_filters_only.assert_awaited_once()
        orch.search.assert_not_awaited()
        sub.qi_engine.quick_classify.assert_not_called()

    def test_regex_fallback_when_llm_router_unavailable(self):
        """extract_filters_only returns regex-sourced entities; no full search."""
        from semantic_search import app as app_module

        qie_only._get_qie_l0_filter_cache().clear()
        intent = _qie_intent_from_entities(
            entities=[
                _entity('tld', ['io'], 'L0_regex'),
                _entity('price_max', 100, 'L0_regex'),
            ],
            raw_text='.io under 100',
            decision_cost_usd=0.0,
        )
        orch = MagicMock()
        orch.extract_filters_only = AsyncMock(return_value=intent)
        orch.search = AsyncMock(side_effect=AssertionError('full search must not run'))
        search_cfg = MagicMock(qie_only_mode=False, top_k_cap=100)
        regex_cfg = MagicMock(enabled=True, fallback_only_when_llm_unavailable=True)

        sub = MagicMock()
        sub.config.general.search = search_cfg
        _wire_qie_config(sub)
        sub.config.qi.l0_regex_entity = regex_cfg
        sub.sanitizer = None
        sub.call_router = None
        sub.orchestrator = orch
        sub.qi_engine = MagicMock()
        sub.qi_engine._entity_extractor = None
        sub.qi_engine._regex_entity_extractor = None
        sub.qi_engine._grounder = None

        with patch.object(app_module, '_require_subsystems', return_value=sub):
            body = asyncio.run(app_module.search(
                request=_search_request(),
                query='.io under 100',
                top_k=50,
                diversity_lambda=0.9,
                relevance_threshold=0.7,
                qie_only_mode=True,
                x_session_id=None,
            ))

        assert body['answer_mode'] == 'qie_only'
        by_name = {e['name']: e for e in body['identified_filters']}
        assert 'tldIncludeList' in by_name
        assert by_name['tldIncludeList']['source'] == 'L0_regex'
        assert 'maxPrice' in by_name
        orch.extract_filters_only.assert_awaited_once()
        orch.search.assert_not_awaited()

    def test_no_regex_when_llm_returns_empty_filters(self):
        """LLM ok with [] -> identified_filters empty; inventory regex must not run."""
        from semantic_search import app as app_module

        qie_only._get_qie_l0_filter_cache().clear()
        intent = _qie_intent_from_entities(
            entities=[], raw_text='hello world', decision_cost_usd=0.01,
        )
        orch = MagicMock()
        orch.extract_filters_only = AsyncMock(return_value=intent)
        orch.search = AsyncMock(side_effect=AssertionError('full search must not run'))
        search_cfg = MagicMock(qie_only_mode=False, top_k_cap=100)
        regex_cfg = MagicMock(enabled=True, fallback_only_when_llm_unavailable=True)
        regex_extractor = MagicMock()
        regex_extractor.classify_async = AsyncMock(
            side_effect=AssertionError('regex must not run on LLM empty success'),
        )

        sub = MagicMock()
        sub.config.general.search = search_cfg
        _wire_qie_config(sub)
        sub.config.qi.l0_regex_entity = regex_cfg
        sub.sanitizer = None
        sub.call_router = MagicMock()
        sub.orchestrator = orch
        sub.qi_engine = MagicMock()
        sub.qi_engine._entity_extractor = MagicMock()
        sub.qi_engine._regex_entity_extractor = regex_extractor
        sub.qi_engine._grounder = None

        with patch.object(app_module, '_require_subsystems', return_value=sub):
            body = asyncio.run(app_module.search(
                request=_search_request(),
                query='hello world',
                top_k=50,
                diversity_lambda=0.9,
                relevance_threshold=0.7,
                qie_only_mode=True,
                x_session_id=None,
            ))

        assert body['identified_filters'] == []
        regex_extractor.classify_async.assert_not_awaited()
        orch.extract_filters_only.assert_awaited_once()

    def test_empty_cache_entry_ignored_reextracts(self):
        """Poisoned empty cache entry must not sticky-serve filter_count=0."""
        from semantic_search import app as app_module
        from semantic_search.cache.keys import exact_query_key, versioned_query_key
        from semantic_search.qi.engine import normalize_query

        qie_only._get_qie_l0_filter_cache().clear()
        cfg = AgentSearchConfig.from_dict(load_config())
        l0 = cfg.qi.l0_llm_entity
        query = 'find .net domains under $50'
        cache_key = exact_query_key(
            versioned_query_key(
                normalize_query(query, cfg.general.max_query_length, normalize=cfg.qi.normalize),
                prompt_tag=l0.prompt_tag,
                schema_version=l0.schema_version,
            )
        )
        qie_only._get_qie_l0_filter_cache().put(cache_key, [])

        intent = _qie_intent_from_entities(
            entities=[
                _entity('tld', ['net'], 'L0_llm'),
                _entity('price_max', 49, 'L0_llm'),
            ],
            raw_text=query,
            decision_cost_usd=0.01,
        )
        orch = MagicMock()
        orch.extract_filters_only = AsyncMock(return_value=intent)
        orch.search = AsyncMock(side_effect=AssertionError('full search must not run'))
        search_cfg = MagicMock(qie_only_mode=False, top_k_cap=100)
        regex_cfg = MagicMock(enabled=True, fallback_only_when_llm_unavailable=True)

        sub = MagicMock()
        sub.config.general.search = search_cfg
        _wire_qie_config(sub)
        sub.config.qi.l0_regex_entity = regex_cfg
        sub.sanitizer = None
        sub.call_router = MagicMock()
        sub.orchestrator = orch
        sub.qi_engine = MagicMock()
        sub.qi_engine._entity_extractor = MagicMock()
        sub.qi_engine._regex_entity_extractor = None
        sub.qi_engine._grounder = None

        with patch.object(app_module, '_require_subsystems', return_value=sub):
            body = asyncio.run(app_module.search(
                request=_search_request(),
                query=query,
                top_k=50,
                diversity_lambda=0.9,
                relevance_threshold=0.7,
                qie_only_mode=True,
                x_session_id=None,
            ))

        by_name = {e['name']: e['value'] for e in body['identified_filters']}
        assert by_name.get('tldIncludeList') == ['net']
        assert by_name.get('maxPrice') == 49
        orch.extract_filters_only.assert_awaited_once()
        cached = qie_only._get_qie_l0_filter_cache().get(cache_key)
        assert cached is not None
        unpacked = qie_only._unpack_qie_cache_entry(cached)
        assert unpacked is not None
        assert unpacked[0]  # identified non-empty

    def test_inventory_bound_empty_llm_falls_back_to_regex(self):
        """Empty extract_filters_only on inventory-bound query -> app regex recovery."""
        from semantic_search import app as app_module
        from semantic_search.qi.l0_regex_filter_extractor import L0RegexFilterExtractor

        qie_only._get_qie_l0_filter_cache().clear()
        empty_intent = _qie_intent_from_entities(
            entities=[], raw_text='find .net domains under $50', decision_cost_usd=0.01,
        )
        orch = MagicMock()
        orch.extract_filters_only = AsyncMock(return_value=empty_intent)
        orch.search = AsyncMock(side_effect=AssertionError('full search must not run'))

        qi = AgentSearchConfig.from_dict(load_config()).qi
        regex_extractor = MagicMock()
        regex_extractor.classify_async = AsyncMock(return_value=IntentSlice(
            query_type='hybrid',
            entities=[
                _entity('tld', ['net'], 'L0_regex'),
                _entity('price_max', 50, 'L0_regex'),
            ],
            confidence=qi.l0_regex_entity.confidence,
            raw_text='find .net domains under $50',
        ))
        regex_extractor._soft_slots = qi.entity_slots.soft_slot_set
        regex_extractor._hard_names = qi.entity_slots.hard_entity_set
        regex_extractor._config = MagicMock(
            enabled=True,
            max_entities=qi.l0_regex_entity.max_entities,
            confidence=qi.l0_regex_entity.confidence,
            source_tag=qi.l0_regex_entity.source_tag,
            fallback_only_when_llm_unavailable=True,
        )

        search_cfg = MagicMock(qie_only_mode=False, top_k_cap=100)
        regex_cfg = MagicMock(enabled=True, fallback_only_when_llm_unavailable=True)
        sub = MagicMock()
        sub.config.general.search = search_cfg
        _wire_qie_config(sub)
        sub.config.qi.l0_regex_entity = regex_cfg
        sub.sanitizer = None
        sub.call_router = MagicMock()
        sub.orchestrator = orch
        sub.qi_engine = MagicMock()
        sub.qi_engine._entity_extractor = MagicMock()
        sub.qi_engine._regex_entity_extractor = regex_extractor
        sub.qi_engine._grounder = None

        async def _regex_priced(query: str):
            from semantic_search.qi.l0_llm_filter_extractor import filters_to_identified
            return filters_to_identified(
                [
                    {'param': 'tldIncludeList', 'value': ['net']},
                    {'param': 'maxPrice', 'value': 50},
                ],
                source='L0_regex',
                soft_slot_names=qi.entity_slots.soft_slot_set,
                confidence=0.9,
                query=query,
            ), 0.0, []

        with patch.object(app_module, '_require_subsystems', return_value=sub), \
             patch.object(L0RegexFilterExtractor, 'extract_priced', AsyncMock(side_effect=_regex_priced)):
            body = asyncio.run(app_module.search(
                request=_search_request(),
                query='find .net domains under $50',
                top_k=50,
                diversity_lambda=0.9,
                relevance_threshold=0.7,
                qie_only_mode=True,
                x_session_id=None,
            ))

        by_name = {e['name']: e for e in body['identified_filters']}
        assert 'tldIncludeList' in by_name
        assert by_name['tldIncludeList']['source'] == 'L0_regex'
        assert 'maxPrice' in by_name
        orch.extract_filters_only.assert_awaited_once()

    def test_second_identical_query_hits_l0_cache(self):
        """Second identical qie_only query skips extract_filters_only; cost 0."""
        from semantic_search import app as app_module

        qie_only._get_qie_l0_filter_cache().clear()
        intent = _qie_intent_from_entities(
            entities=[
                _entity('tld', ['io'], 'L0_llm'),
                _entity('price_max', 99, 'L0_llm'),
            ],
            raw_text='cache me .io under 50',
            decision_cost_usd=0.02,
        )
        orch = MagicMock()
        orch.extract_filters_only = AsyncMock(return_value=intent)
        orch.search = AsyncMock(side_effect=AssertionError('full search must not run'))
        search_cfg = MagicMock(qie_only_mode=False, top_k_cap=100)
        regex_cfg = MagicMock(enabled=True, fallback_only_when_llm_unavailable=True)

        sub = MagicMock()
        sub.config.general.search = search_cfg
        _wire_qie_config(sub)
        sub.config.qi.l0_regex_entity = regex_cfg
        sub.sanitizer = None
        sub.call_router = MagicMock()
        sub.orchestrator = orch
        sub.qi_engine = MagicMock()
        sub.qi_engine._entity_extractor = MagicMock()
        sub.qi_engine._regex_entity_extractor = None
        sub.qi_engine._grounder = None

        kwargs = dict(
            request=_search_request(),
            query='cache me .io under 50',
            top_k=50,
            diversity_lambda=0.9,
            relevance_threshold=0.7,
            qie_only_mode=True,
            x_session_id=None,
        )
        with patch.object(app_module, '_require_subsystems', return_value=sub):
            first = asyncio.run(app_module.search(**kwargs))
            second = asyncio.run(app_module.search(**kwargs))

        assert first['answer_mode'] == 'qie_only'
        assert second['answer_mode'] == 'qie_only'
        assert second['decision_cost_usd'] == 0.0
        assert first['identified_filters'] == second['identified_filters']
        assert first['request_id'] != second['request_id']
        assert orch.extract_filters_only.await_count == 1
        orch.search.assert_not_awaited()
