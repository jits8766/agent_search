"""_merge_extractor_entities: LLM completed → LLM only; regex only when LLM unavailable.

No per-slot gap-fill. Regex must not pollute a successful LLM extract (incl. empty).
"""
from typing import Any, List, Optional, Tuple

from semantic_search.config.loader import load_config
from semantic_search.config.models import AgentSearchConfig
from semantic_search.contracts import Entity, IntentSlice
from semantic_search.qi.engine import _merge_extractor_entities

_SLOT_SETS: Optional[Tuple[frozenset, frozenset]] = None


def _slot_sets() -> Tuple[frozenset, frozenset]:
    """(soft_slot_names, hard_entity_names) from qi.entity_slots YAML."""
    global _SLOT_SETS
    if _SLOT_SETS is None:
        slots = AgentSearchConfig.from_dict(load_config()).qi.entity_slots
        assert slots is not None
        _SLOT_SETS = (slots.soft_slot_set, slots.hard_entity_set)
    return _SLOT_SETS


def _merge(regex, llm, query: str = '', *, llm_completed: bool = True) -> List[Entity]:
    soft, hard = _slot_sets()
    hard_ents, soft_ents = _merge_extractor_entities(
        regex, llm, query, soft, hard, llm_completed=llm_completed,
    )
    return hard_ents + soft_ents


def _slice(entities: List[Entity]) -> IntentSlice:
    return IntentSlice(query_type='hybrid', entities=entities, confidence=1.0, raw_text='q')


def _entity(name: str, value: Any, source: str) -> Entity:
    chip_kind = 'soft' if name.startswith('topic') else 'hard'
    return Entity(name=name, value=value, confidence=0.9, source=source, chip_kind=chip_kind)


def _by_name(entities: List[Entity], name: str) -> Entity:
    return next(e for e in entities if e.name == name)


def _names(entities: List[Entity]) -> set:
    return {e.name for e in entities}


def test_llm_keyword_wins_over_regex() -> None:
    """When both extractors emit keyword_contains, the LLM value wins."""
    regex = _slice([_entity('keyword_contains', ['fin'], 'L0_regex')])
    llm = _slice([_entity('keyword_contains', ['finance'], 'L0_llm')])
    merged = _merge(regex, llm)
    kc = _by_name(merged, 'keyword_contains')
    assert kc.value == ['finance']
    assert kc.source == 'L0_llm'


def test_regex_keyword_used_as_llm_off_fallback() -> None:
    """When LLM unavailable, the regex keyword is retained."""
    regex = _slice([_entity('keyword_contains', ['shop'], 'L0_regex')])
    merged = _merge(regex, None, llm_completed=False)
    kc = _by_name(merged, 'keyword_contains')
    assert kc.value == ['shop']
    assert kc.source == 'L0_regex'


def test_llm_wins_hard_filters() -> None:
    """Hard filters (tld/price): nonempty LLM discards regex entirely."""
    regex = _slice([_entity('tld', ['net'], 'L0_regex'), _entity('price_max', 2, 'L0_regex')])
    llm = _slice([_entity('tld', ['ai', 'io'], 'L0_llm'), _entity('price_max', 2000, 'L0_llm')])
    merged = _merge(regex, llm, 'ai or io under 2k with backlinks')
    assert _by_name(merged, 'tld').value == ['ai', 'io']
    assert _by_name(merged, 'tld').source == 'L0_llm'
    assert _by_name(merged, 'price_max').value == 2000
    assert _by_name(merged, 'price_max').source == 'L0_llm'


def test_regex_hard_filter_fallback_when_llm_unavailable() -> None:
    """When LLM did not complete, regex hard filters are the whole result."""
    regex = _slice([_entity('tld', ['com'], 'L0_regex'), _entity('price_max', 500, 'L0_regex')])
    merged = _merge(regex, None, 'buy now .com under 500', llm_completed=False)
    assert _by_name(merged, 'tld').source == 'L0_regex'
    assert _by_name(merged, 'price_max').value == 500


def test_no_regex_pollution_when_llm_empty_success() -> None:
    """LLM completed with zero entities → regex slots discarded (no L0_regex pollution).

    Cue reconcile may still inject non-regex entities from query text (e.g.
    price_max via reconcile_under_price_inject) — that is not regex merge.
    """
    regex = _slice([_entity('tld', ['com'], 'L0_regex'), _entity('price_max', 500, 'L0_regex')])
    merged = _merge(regex, _slice([]), 'buy now .com under 500', llm_completed=True)
    assert all(e.source != 'L0_regex' for e in merged)
    # Regex-only tld must not leak; reconcile may still add price_max as L0_entity.
    assert 'tld' not in _names(merged)
    if 'price_max' in _names(merged):
        assert _by_name(merged, 'price_max').source != 'L0_regex'


def test_llm_wins_tld_exclude_no_regex_pollution() -> None:
    """Nonempty LLM → regex tld includes discarded (no gap-fill)."""
    regex = _slice([
        _entity('tldExcludeList', ['xyz'], 'L0_regex'),
        _entity('tld', ['info', 'club'], 'L0_regex'),
    ])
    llm = _slice([_entity('tldExcludeList', ['xyz', 'info', 'club'], 'L0_llm')])
    merged = _merge(
        regex, llm, 'anything good just not .xyz .info or .club',
    )
    assert _by_name(merged, 'tldExcludeList').value == ['xyz', 'info', 'club']
    assert _by_name(merged, 'tldExcludeList').source == 'L0_llm'
    assert 'tld' not in _names(merged)


def test_mixed_query_llm_hard_and_keyword() -> None:
    """LLM owns tld + keyword; regex discarded when LLM nonempty."""
    regex = _slice([_entity('tld', ['net'], 'L0_regex'), _entity('keyword_contains', ['fin'], 'L0_regex')])
    llm = _slice([
        _entity('tld', ['app', 'dev'], 'L0_llm'),
        _entity('price_max', 300, 'L0_llm'),
        _entity('keyword_contains', ['finance'], 'L0_llm'),
        _entity('topic_include', ['finance'], 'L0_llm'),
    ])
    merged = _merge(regex, llm, 'app or dev extension under 300')
    assert _by_name(merged, 'tld').value == ['app', 'dev']
    assert _by_name(merged, 'tld').source == 'L0_llm'
    assert _by_name(merged, 'price_max').value == 300
    assert _by_name(merged, 'keyword_contains').source == 'L0_llm'
    assert _by_name(merged, 'topic_include').source == 'L0_llm'


def test_regex_buy_it_now_not_gap_filled_when_llm_nonempty() -> None:
    """Nonempty LLM → regex entities discarded (no gap-fill). Cue reconcile may still inject."""
    regex = _slice([_entity('buy_it_now', True, 'L0_regex'), _entity('price_max', 999, 'L0_regex')])
    llm = _slice([_entity('price_max', 200, 'L0_llm')])
    merged = _merge(regex, llm, 'buy it now under 200')
    assert _by_name(merged, 'price_max').value == 200
    assert _by_name(merged, 'price_max').source == 'L0_llm'
    # Regex must not contribute — any buy_it_now is cue-reconcile, not L0_regex.
    assert all(e.source != 'L0_regex' for e in merged)


def test_regex_buy_it_now_when_llm_unavailable() -> None:
    """LLM unavailable → regex buy_it_now kept as whole-result fallback."""
    regex = _slice([_entity('buy_it_now', True, 'L0_regex'), _entity('price_max', 200, 'L0_regex')])
    merged = _merge(regex, None, 'buy it now under 200', llm_completed=False)
    assert _by_name(merged, 'buy_it_now').value is True
    assert _by_name(merged, 'buy_it_now').source == 'L0_regex'
