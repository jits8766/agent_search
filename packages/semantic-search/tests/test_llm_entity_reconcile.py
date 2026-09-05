"""Keyword scrub guards used after L0 extract (maps module, not a second extractor).

Covers:
- _reconcile_keyword_topic: a literal keyword token that is a strict substring of a
  topic_include seed (abbreviation the LLM also expanded to a topic) is dropped, while
  an equal token (genuine dual signal) is kept.
- _reconcile_dangling_match_mode: keyword_match_mode is meaningless with fewer than two
  keyword tokens and is removed.
"""
from typing import Any, List

from semantic_search.contracts import Entity
from semantic_search.qi.llm_entity_extractor import (
    _reconcile_dangling_match_mode,
    _reconcile_keyword_meta_blocklist,
    _reconcile_keyword_topic,
)


def _entity(name: str, value: Any) -> Entity:
    """Build an Entity with a chip_kind consistent with the slot family."""
    chip_kind = 'soft' if name.startswith('topic') else 'hard'
    return Entity(name=name, value=value, confidence=0.9, source='L0_llm', chip_kind=chip_kind)


def _names(entities: List[Entity]) -> set:
    return {e.name for e in entities}


def test_keyword_substring_of_topic_is_dropped() -> None:
    """'fin' keyword echoing topic 'finance' is removed; the topic + other slots stay."""
    ents = [_entity('tld', ['net']), _entity('keyword_contains', ['fin']), _entity('topic_include', ['finance'])]
    out = _reconcile_keyword_topic(ents)
    assert 'keyword_contains' not in _names(out)
    assert 'topic_include' in _names(out)
    assert 'tld' in _names(out)


def test_keyword_equal_to_topic_is_kept() -> None:
    """An exact keyword==topic token is a genuine dual signal and is preserved."""
    ents = [_entity('keyword_contains', ['tech']), _entity('topic_include', ['tech'])]
    out = _reconcile_keyword_topic(ents)
    assert 'keyword_contains' in _names(out)


def test_keyword_partial_drop_keeps_non_topic_tokens() -> None:
    """Only the topic-echo token is dropped; unrelated keyword tokens survive."""
    ents = [_entity('keyword_contains', ['ai', 'fin']), _entity('topic_include', ['finance'])]
    out = _reconcile_keyword_topic(ents)
    kc = next(e for e in out if e.name == 'keyword_contains')
    assert kc.value == ['ai']


def test_keyword_topic_noop_without_topic() -> None:
    """With no topic_include present, keyword slots are untouched."""
    ents = [_entity('keyword_contains', ['fin'])]
    out = _reconcile_keyword_topic(ents)
    assert out == ents


def test_dangling_match_mode_dropped_with_single_keyword() -> None:
    """keyword_match_mode is removed when fewer than two keyword tokens exist."""
    ents = [_entity('keyword_contains', ['ai']), _entity('keyword_match_mode', 'any')]
    out = _reconcile_dangling_match_mode(ents)
    assert 'keyword_match_mode' not in _names(out)


def test_exact_match_mode_kept_without_keywords() -> None:
    """Standalone 'exact' mode survives dangling reconcile with zero keyword tokens."""
    ents = [_entity('keyword_match_mode', 'exact')]
    out = _reconcile_dangling_match_mode(ents)
    assert 'keyword_match_mode' in _names(out)


def test_match_mode_kept_with_two_keywords() -> None:
    """keyword_match_mode is retained when two or more keyword tokens exist."""
    ents = [_entity('keyword_contains', ['ai', 'cloud']), _entity('keyword_match_mode', 'all')]
    out = _reconcile_dangling_match_mode(ents)
    assert 'keyword_match_mode' in _names(out)


def test_keyword_meta_blocklist_drops_numbers_exclude() -> None:
    """Structural 'numbers'/'hyphens' tokens are stripped from keyword_contains_exclude."""
    ents = [
        _entity('keyword_contains_exclude', ['numbers', 'hyphens']),
        _entity('keyword_contains', ['cloud']),
    ]
    out = _reconcile_keyword_meta_blocklist(ents)
    assert 'keyword_contains_exclude' not in _names(out)
    assert 'keyword_contains' in _names(out)


def test_keyword_meta_blocklist_keeps_real_exclude() -> None:
    """Non-meta exclude tokens survive; only blocklisted tokens are removed."""
    ents = [_entity('keyword_contains_exclude', ['crypto', 'numbers'])]
    out = _reconcile_keyword_meta_blocklist(ents)
    kc = next(e for e in out if e.name == 'keyword_contains_exclude')
    assert kc.value == ['crypto']
