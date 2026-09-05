"""L0LLMFilterExtractor.classify_async — shared L0 path for full-search IntentSlice."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

from semantic_search.config.loader import load_config
from semantic_search.config.models import AgentSearchConfig
from semantic_search.qi.l0_llm_filter_extractor import (
    L0LLMFilterExtractor,
    _L0BatchResponse,
    _L0Filter,
    _L0ResultEntry,
)


def _qi():
    qi = AgentSearchConfig.from_dict(load_config()).qi
    assert qi.entity_slots is not None
    assert qi.l0_llm_entity is not None
    return qi


def _l0_extractor(router, **overrides):
    qi = _qi()
    cfg = qi.l0_llm_entity
    kwargs = dict(
        entity_slots=qi.entity_slots,
        source_tag=cfg.source_tag,
        max_entities=cfg.max_entities,
        confidence=cfg.confidence,
        enabled=cfg.enabled,
        prompt_tag=cfg.prompt_tag,
        keyword_min_probability=cfg.keyword_min_probability,
        combined_prompt_tag=cfg.combined_prompt_tag,
    )
    kwargs.update(overrides)
    return L0LLMFilterExtractor(router, cfg.task_type, **kwargs)


def test_classify_async_maps_api_and_soft_to_intent_slice():
    payload = _L0BatchResponse(
        results=[
            _L0ResultEntry(
                idx=1,
                filters=[
                    _L0Filter(param='tldIncludeList', value='io|app'),
                    _L0Filter(param='maxPrice', value=500),
                    _L0Filter(param='keyword_contains', value=['cloud']),
                    _L0Filter(param='excludeDigits', value=True),
                ],
            )
        ]
    )
    router = MagicMock()
    router.call_structured = AsyncMock(return_value=(payload, {'model': 'test'}))
    ext = _l0_extractor(router)
    slice_ = asyncio.run(ext.classify_async('cloud .io under 500 letters only'))
    assert slice_ is not None
    hard = {e.name: e.value for e in slice_.entities}
    soft = {e.name: e.value for e in slice_.soft_entities}
    assert hard['tld'] == ['io', 'app']
    assert hard['price_max'] == 499
    assert hard['has_number'] is False  # excludeDigits inverted
    # Multi-value slots always lists (single token or pipe-joined).
    assert soft['keyword_contains'] == ['cloud']
    router.call_structured.assert_awaited_once()


def test_coerce_single_token_multi_slots_are_lists():
    """Bare tld/auction_type tokens must be lists — char-iteration breaks MatchAny/gate."""
    from semantic_search.qi.l0_llm_filter_extractor import _coerce_l0_filter_value

    assert _coerce_l0_filter_value('tld', 'com') == ['com']
    assert _coerce_l0_filter_value('auction_type', '16') == ['16']
    assert _coerce_l0_filter_value('tld', 'io|app') == ['io', 'app']
    assert _coerce_l0_filter_value('keyword_contains', 'cloud') == ['cloud']


def test_classify_async_single_tld_and_type_are_lists():
    payload = _L0BatchResponse(
        results=[
            _L0ResultEntry(
                idx=1,
                filters=[
                    _L0Filter(param='tldIncludeList', value='com'),
                    _L0Filter(param='typeIncludeList', value='16'),
                    _L0Filter(param='maxPrice', value=50),
                ],
            )
        ]
    )
    router = MagicMock()
    router.call_structured = AsyncMock(return_value=(payload, {'model': 'test'}))
    ext = _l0_extractor(router)
    slice_ = asyncio.run(ext.classify_async('find .com domains for auction type 16 under $50'))
    hard = {e.name: e.value for e in slice_.entities}
    assert hard['tld'] == ['com']
    assert hard['auction_type'] == ['16']
    assert hard['price_max'] == 49


def test_extract_and_classify_share_same_llm_call_shape():
    """qie_only extract() and full-search classify_async() use identical prompt_tag."""
    payload = _L0BatchResponse(results=[_L0ResultEntry(idx=1, filters=[])])
    router = MagicMock()
    router.call_structured = AsyncMock(return_value=(payload, {'model': 'test'}))
    ext = _l0_extractor(router)
    asyncio.run(ext.extract('cheap .com'))
    kwargs = router.call_structured.await_args.kwargs
    assert kwargs['prompt_tag'] == _qi().l0_llm_entity.prompt_tag
    assert kwargs['task_type'] == 'l0_entity_extraction'


def test_classify_sync_for_calibration_boot_no_llm():
    """Registry calibration producers call .classify - hybrid stub, no LLM."""
    router = MagicMock()
    router.call_structured = AsyncMock(side_effect=AssertionError('classify must not call LLM'))
    ext = _l0_extractor(router)
    slice_ = ext.classify('pending delete domains')
    assert slice_ is not None
    assert slice_.query_type == 'hybrid'
    assert slice_.confidence == 1.0
    assert slice_.entities == []
    assert slice_.soft_entities == []
    router.call_structured.assert_not_called()


def _identified_kwargs():
    qi = _qi()
    return dict(
        source=qi.l0_llm_entity.source_tag,
        soft_slot_names=qi.entity_slots.soft_slot_set,
        confidence=qi.l0_llm_entity.confidence,
    )


def test_filters_to_identified_drops_qualitative_sentinel_none():
    """L0 rule 4 - \"high cpc\" must not emit minSemrushCostPerClick:none."""
    from semantic_search.qi.l0_llm_filter_extractor import filters_to_identified

    out = filters_to_identified([
        {'param': 'minSemrushCostPerClick', 'value': 'none'},
        {'param': 'maxPrice', 'value': 3000},
        {'param': 'topic_include', 'value': 'finance|legal'},
    ], **_identified_kwargs())
    names = {e['name'] for e in out}
    assert 'minSemrushCostPerClick' not in names
    assert 'maxPrice' in names
    assert 'topic_include' in names


def test_filters_to_identified_emits_chip_kind_and_confidence():
    """Hard FIND vs soft keyword/topic chips from qi.entity_slots + l0_llm_entity.confidence."""
    from semantic_search.qi.l0_llm_filter_extractor import filters_to_identified

    qi = _qi()
    out = filters_to_identified([
        {'param': 'tldIncludeList', 'value': ['com']},
        {'param': 'maxPrice', 'value': 100},
        {'param': 'keyword_contains', 'value': 'ai'},
        {'param': 'topic_include', 'value': 'finance'},
        {'param': 'price_below_market', 'value': True},
    ], **_identified_kwargs())
    by_name = {e['name']: e for e in out}
    assert by_name['tldIncludeList']['chip_kind'] == 'hard'
    assert by_name['maxPrice']['chip_kind'] == 'hard'
    assert by_name['keyword_contains']['chip_kind'] == 'soft'
    assert by_name['topic_include']['chip_kind'] == 'soft'
    # Local but hard in entity_slots - not soft chip.
    assert by_name['price_below_market']['chip_kind'] == 'hard'
    assert all(e['confidence'] == qi.l0_llm_entity.confidence for e in out)


def test_expand_gd_to_godaddy_all_paths():
    """Bare gd -> godaddy; gd transfer becomes godaddy transfer (gd_transfer cue)."""
    from semantic_search.config.loader import load_config
    from semantic_search.config.models import QINormalizeConfig
    from semantic_search.qi.engine import normalize_query
    from semantic_search.qi.l0_llm_filter_extractor import (
        build_l0_filter_user_prompt,
        expand_gd_to_godaddy,
    )

    normalize = QINormalizeConfig.from_dict(load_config()['qi']['normalize'])
    assert expand_gd_to_godaddy('gd auctions under 75') == 'godaddy auctions under 75'
    assert expand_gd_to_godaddy('GD transfer eligible') == 'godaddy transfer eligible'
    assert expand_gd_to_godaddy('godaddy auctions') == 'godaddy auctions'
    assert normalize_query('gd com under 500', 500, normalize=normalize) == 'godaddy com under 500'
    assert normalize_query('cheap .com domains?', 500, normalize=normalize) == 'cheap .com domains'
    prompt = build_l0_filter_user_prompt([(1, 'gd auctions under 75')])
    assert 'godaddy auctions under 75' in prompt
    assert '1. gd auctions' not in prompt


def test_filters_to_identified_budget_prefix_in_dollar():
    """qie_only path: ``in $50`` injects inclusive maxPrice when LLM omits it."""
    from semantic_search.qi.l0_llm_filter_extractor import filters_to_identified

    out = filters_to_identified(
        [{'param': 'tldIncludeList', 'value': 'net'}],
        query='find .net domains in $50',
        **_identified_kwargs(),
    )
    by_name = {e['name']: e['value'] for e in out}
    assert by_name.get('tldIncludeList') == 'net'
    assert by_name.get('maxPrice') == 50


def test_filters_to_identified_budget_prefix_for_and_around():
    from semantic_search.qi.l0_llm_filter_extractor import filters_to_identified

    kw = _identified_kwargs()
    for_q = filters_to_identified([], query='domains for $100', **kw)
    assert any(e['name'] == 'maxPrice' and e['value'] == 100 for e in for_q)
    around = filters_to_identified([], query='domains around $75', **kw)
    by_name = {e['name']: e['value'] for e in around}
    assert by_name.get('minPrice') == 75
    assert by_name.get('maxPrice') == 75


def test_filters_to_identified_budget_prefix_fp_guards():
    from semantic_search.qi.l0_llm_filter_extractor import filters_to_identified

    kw = _identified_kwargs()
    assert not any(e['name'] == 'maxPrice' for e in filters_to_identified([], query='domains in 50', **kw))
    assert not any(e['name'] == 'maxPrice' for e in filters_to_identified([], query='ending in 50', **kw))
    assert not any(e['name'] == 'maxPrice' for e in filters_to_identified([], query='in 16', **kw))


def test_filters_to_identified_under_eur_injects_maxprice_and_currency():
    """ISO currency before amount: under EUR 80 -> maxPrice=79 + filterPriceCurrency=EUR."""
    from semantic_search.qi.l0_llm_filter_extractor import filters_to_identified

    out = filters_to_identified(
        [],
        query='brandable coffee domains under EUR 80 .com',
        **_identified_kwargs(),
    )
    by_name = {e['name']: e['value'] for e in out}
    assert by_name.get('maxPrice') == 79
    assert by_name.get('filterPriceCurrency') == 'EUR'


def test_filters_to_identified_under_gbp_injects_maxprice_and_currency():
    from semantic_search.qi.l0_llm_filter_extractor import filters_to_identified

    out = filters_to_identified(
        [],
        query='pizza restaurant domains under GBP 95 .io',
        **_identified_kwargs(),
    )
    by_name = {e['name']: e['value'] for e in out}
    assert by_name.get('maxPrice') == 94
    assert by_name.get('filterPriceCurrency') == 'GBP'


def test_keyword_threshold_fraction_and_terms():
    from types import SimpleNamespace

    from semantic_search.qi.keyword_threshold import (
        filter_keywords,
        keyword_terms,
        min_probability_fraction,
    )

    frac = min_probability_fraction(SimpleNamespace(keyword_min_probability=70))
    assert frac == 0.7
    kws = [
        {'term': 'coffee', 'probability': 0.95},
        {'term': 'weak', 'probability': 0.5},
        {'term': 'pizza', 'probability': 0.7},
    ]
    assert filter_keywords(kws, frac) == [
        {'term': 'coffee', 'probability': 0.95},
        {'term': 'pizza', 'probability': 0.7},
    ]
    assert keyword_terms(kws, frac) == ['coffee', 'pizza']
