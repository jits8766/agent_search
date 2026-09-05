"""L0RegexFilterExtractor maps regex slots → FIND + soft/local (same as L0 LLM)."""
import asyncio

from semantic_search.config.loader import load_config
from semantic_search.config.models import AgentSearchConfig, QIL0RegexEntityConfig
from semantic_search.contracts import Entity, IntentSlice
from semantic_search.qi.l0_llm_filter_extractor import FILTERABLE_PARAMS
from semantic_search.qi.l0_regex_filter_extractor import L0RegexFilterExtractor
from semantic_search.qi.regex_entity_extractor import RegexEntityExtractor
from semantic_search.qi.slot_to_api_param import FIND_FILTERABLE_API_PARAMS


def test_find_filterable_allowlist_is_63() -> None:
    assert len(FIND_FILTERABLE_API_PARAMS) == 63


def test_filterable_params_include_soft() -> None:
    assert 'keyword_contains' in FILTERABLE_PARAMS
    assert 'maxPrice' in FILTERABLE_PARAMS


def test_maps_tld_and_price_to_find_params() -> None:
    conf = AgentSearchConfig.from_dict(load_config())
    slots = conf.qi.entity_slots
    regex_cfg = conf.qi.l0_regex_entity
    assert slots is not None and regex_cfg is not None
    rex = RegexEntityExtractor(
        QIL0RegexEntityConfig(
            enabled=True,
            max_entities=regex_cfg.max_entities,
            confidence=regex_cfg.confidence,
            source_tag=regex_cfg.source_tag,
            fallback_only_when_llm_unavailable=regex_cfg.fallback_only_when_llm_unavailable,
        ),
        hard_entity_names=slots.hard_entity_set,
        soft_slot_names=slots.soft_slot_set,
        known_tlds=frozenset(conf.qi.regex.known_tlds),
    )
    identified = asyncio.run(L0RegexFilterExtractor(rex).extract('.io under 100'))
    by_name = {e['name']: e for e in identified}
    assert 'tldIncludeList' in by_name
    assert 'maxPrice' in by_name
    assert by_name['tldIncludeList']['source'] == regex_cfg.source_tag
    assert by_name['maxPrice']['source'] == regex_cfg.source_tag
    for e in identified:
        assert e['name'] in FILTERABLE_PARAMS


def test_keeps_soft_slots_same_as_llm_catalog(monkeypatch) -> None:
    conf = AgentSearchConfig.from_dict(load_config())
    slots = conf.qi.entity_slots
    regex_cfg = conf.qi.l0_regex_entity
    assert slots is not None and regex_cfg is not None
    rex = RegexEntityExtractor(
        QIL0RegexEntityConfig(
            enabled=True,
            max_entities=regex_cfg.max_entities,
            confidence=regex_cfg.confidence,
            source_tag=regex_cfg.source_tag,
            fallback_only_when_llm_unavailable=regex_cfg.fallback_only_when_llm_unavailable,
        ),
        hard_entity_names=slots.hard_entity_set,
        soft_slot_names=slots.soft_slot_set,
        known_tlds=frozenset(conf.qi.regex.known_tlds),
    )

    async def _fake_classify(query: str):
        return IntentSlice(
            query_type='hybrid',
            entities=[
                Entity(name='tld', value=['com'], confidence=0.9, source='L0_regex', chip_kind='hard'),
                Entity(name='buy_it_now', value=True, confidence=0.9, source='L0_regex', chip_kind='hard'),
            ],
            confidence=0.9,
            raw_text=query,
            soft_entities=[
                Entity(
                    name='topic_include',
                    value=['fintech'],
                    confidence=0.9,
                    source='L0_regex',
                    chip_kind='soft',
                ),
            ],
        )

    monkeypatch.setattr(rex, 'classify_async', _fake_classify)
    identified = asyncio.run(L0RegexFilterExtractor(rex).extract('fintech .com buy now'))
    names = {e['name'] for e in identified}
    assert 'tldIncludeList' in names
    assert 'topic_include' in names
    assert 'buy_it_now' in names
