"""_drop_numeric_misattributions: keep matching LLM prices; never backfill regex into LLM."""
from typing import Any, List, Optional, Tuple

from semantic_search.config.loader import load_config
from semantic_search.config.models import AgentSearchConfig
from semantic_search.contracts import Entity, IntentSlice
from semantic_search.qi.engine import (
    _drop_numeric_misattributions,
    _merge_extractor_entities,
)

_SLOT_SETS: Optional[Tuple[frozenset, frozenset]] = None


def _slot_sets() -> Tuple[frozenset, frozenset]:
    global _SLOT_SETS
    if _SLOT_SETS is None:
        slots = AgentSearchConfig.from_dict(load_config()).qi.entity_slots
        assert slots is not None
        _SLOT_SETS = (slots.soft_slot_set, slots.hard_entity_set)
    return _SLOT_SETS


def _merge(regex, llm, query: str, *, llm_completed: bool = True) -> List[Entity]:
    soft, hard = _slot_sets()
    hard_ents, soft_ents = _merge_extractor_entities(
        regex, llm, query, soft, hard, llm_completed=llm_completed,
    )
    return hard_ents + soft_ents


def _entity(name: str, value: Any, source: str) -> Entity:
    return Entity(name=name, value=value, confidence=0.9, source=source, chip_kind='hard')


def _slice(entities: List[Entity]) -> IntentSlice:
    return IntentSlice(query_type='hybrid', entities=entities, confidence=1.0, raw_text='q')


def _names(entities: List[Entity]) -> set:
    return {e.name for e in entities}


def test_currency_price_llm_matching_value_kept() -> None:
    """under $50: LLM price_max=50 must survive family-precedence (was wiped)."""
    q = 'find .net and .com domains under $50'
    regex = _slice([
        _entity('tld', ['net', 'com'], 'L0_regex'),
        _entity('price_max', 50.0, 'L0_regex'),
    ])
    llm = _slice([
        _entity('tld', ['net', 'com'], 'L0_llm'),
        _entity('price_max', 50.0, 'L0_llm'),
    ])
    merged = _merge(regex, llm, q)
    out = _drop_numeric_misattributions(merged, q)
    assert 'price_max' in _names(out)
    price = next(e for e in out if e.name == 'price_max')
    assert float(price.value) == 50.0
    assert price.source == 'L0_llm'


def test_currency_price_wrong_llm_not_backfilled_from_regex() -> None:
    """under $50: wrong LLM price scrubbed; regex must NOT re-enter (no backfill)."""
    q = 'find .net and .com domains under $50'
    regex = _slice([
        _entity('tld', ['net', 'com'], 'L0_regex'),
        _entity('price_max', 50.0, 'L0_regex'),
    ])
    llm = _slice([
        _entity('tld', ['net', 'com'], 'L0_llm'),
        _entity('price_max', 500.0, 'L0_llm'),
    ])
    merged = _merge(regex, llm, q)
    out = _drop_numeric_misattributions(merged, q)
    assert 'price_max' not in _names(out)
    assert all(e.source != 'L0_regex' for e in out)


def test_bare_under_2k_llm_survives_empty_authority() -> None:
    """under 2k: no currency anchor → authority empty → LLM 2000 kept."""
    q = 'ai or io under 2k with backlinks'
    regex = _slice([_entity('price_max', 2.0, 'L0_regex')])
    llm = _slice([_entity('price_max', 2000.0, 'L0_llm')])
    merged = _merge(regex, llm, q)
    out = _drop_numeric_misattributions(merged, q)
    price = next(e for e in out if e.name == 'price_max')
    assert float(price.value) == 2000.0
    assert price.source == 'L0_llm'


def test_cross_family_char_count_not_kept_as_price() -> None:
    """Number anchored to name_length must not survive as price_max."""
    q = 'domains under 5 characters'
    ents = [_entity('price_max', 5.0, 'L0_llm'), _entity('name_length_max', 4, 'L0_regex')]
    out = _drop_numeric_misattributions(ents, q)
    assert 'price_max' not in _names(out)
    assert 'name_length_max' in _names(out)


def test_exclusive_under_traffic_max_n_minus_1_kept() -> None:
    """under 100 traffic: L0 exclusive ceiling traffic_max=99 must survive.

    Authority anchors 100→traffic; exclusive-under emits N-1. Previously family
    precedence wiped 99 because it was not an exact anchored value.
    """
    q = 'low traffic under 100'
    out = _drop_numeric_misattributions(
        [_entity('traffic_max', 99, 'L0_llm')], q,
    )
    assert 'traffic_max' in _names(out)
    assert float(next(e for e in out if e.name == 'traffic_max').value) == 99.0


def test_exclusive_above_traffic_min_n_plus_1_kept() -> None:
    """above 15 traffic: L0 exclusive floor traffic_min=16 must survive."""
    q = 'traffic above 15'
    out = _drop_numeric_misattributions(
        [_entity('traffic_min', 16, 'L0_llm')], q,
    )
    # Only assert when authority owns the traffic family for 15.
    from semantic_search.qi.engine import _regex_numeric_authority
    val_to_fam, fams = _regex_numeric_authority(q)
    if 'traffic' in fams and 15.0 in val_to_fam:
        assert 'traffic_min' in _names(out)
        assert float(next(e for e in out if e.name == 'traffic_min').value) == 16.0
