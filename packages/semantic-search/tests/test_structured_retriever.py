"""Tests for structured_retriever: crash-guard, exclusion filters, keyword_phrase, term_in_compound."""
import logging
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from semantic_search.contracts import Entity, IntentSlice, QueryIntent
from semantic_search.retrieval.structured_retriever import (
    InMemoryStructuredIndex,
    _term_in_compound,
    extract_filters_from_intent,
    item_matches_filters,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_intent(entities: List[Dict[str, Any]]) -> QueryIntent:
    """Build a minimal QueryIntent with the given entities on one slice."""
    ent_objs = [
        Entity(name=e['name'], value=e['value'], confidence=0.9, source='L0_entity', chip_kind='hard')
        for e in entities
    ]
    sl = IntentSlice(query_type='hybrid', entities=ent_objs, confidence=0.9, raw_text='test')
    return QueryIntent(
        request_id='test-req-1',
        raw_query='test query',
        normalized_query='test query',
        query_type='hybrid',
        confidence=0.9,
        decision_tier='L0_entity',
        slices=[sl],
        decision_cost_usd=0.0,
    )


def _make_item(**kwargs) -> Dict[str, Any]:
    """Build a minimal index item with defaults."""
    base = {'item_id': 'x1', 'score': 0.8, 'tld': 'com', 'sld': 'example', 'price': 100, 'auction_type': '16'}
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# FIX 1 — crash-guard: non-numeric values on numeric slots are dropped
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('slot,bad_value', [
    ('price_min', 'cheapest'),
    ('price_min', None),
    ('price_max', 'premium'),
    ('price_max', 'very expensive'),
    ('traffic_min', 'significant traffic'),
    ('traffic_max', 'lots'),
    ('name_length_max', 'short'),
    ('name_length_min', 'long'),
    ('bids_min', 'many'),
    ('domain_age_min', 'old'),
    ('majestic_tf_min', 'high'),
    ('semrush_backlinks_min', 'plenty'),
    ('quality_min', 'good'),
    ('govalue_min', 'valuable'),
])
def test_crash_guard_non_numeric_slot_dropped(slot, bad_value):
    """Non-coercible value on a numeric slot must be silently dropped — no exception."""
    intent = _make_intent([{'name': slot, 'value': bad_value}])
    filters = extract_filters_from_intent(intent)
    assert slot not in filters, f"slot={slot!r} with bad_value={bad_value!r} must be dropped"


@pytest.mark.parametrize('slot,bad_value', [
    ('price_min', 'cheapest'),
    ('traffic_min', 'significant traffic'),
    ('quality_min', 'good'),
])
def test_crash_guard_logs_warning(slot, bad_value):
    """Dropped numeric slot must call logger.warning with slot_dropped in the message."""
    import semantic_search.retrieval.structured_retriever as sr_mod
    intent = _make_intent([{'name': slot, 'value': bad_value}])
    with patch.object(sr_mod.logger, 'warning') as mock_warn:
        extract_filters_from_intent(intent)
    assert mock_warn.called, f"logger.warning not called for slot={slot!r}"
    call_args = mock_warn.call_args[0][0]
    assert 'slot_dropped' in call_args, f"'slot_dropped' missing from warning: {call_args!r}"
    assert slot in call_args, f"slot={slot!r} missing from warning: {call_args!r}"


def test_crash_guard_valid_numeric_passes():
    """A numeric slot with a coercible value is kept and applied without error."""
    intent = _make_intent([{'name': 'price_min', 'value': '100'}])
    filters = extract_filters_from_intent(intent)
    assert filters.get('price_min') == 100


def test_crash_guard_string_slots_unaffected():
    """Boolean/list/string slots are not touched by the numeric guard."""
    intent = _make_intent([
        {'name': 'tld', 'value': ['com', 'io']},
        {'name': 'keyword_contains', 'value': 'cloud'},
    ])
    filters = extract_filters_from_intent(intent)
    assert filters.get('tld') == ['com', 'io']
    # List-valued slots always projected as lists (scalar str wrapped).
    assert filters.get('keyword_contains') == ['cloud']


def test_extract_filters_wraps_scalar_tld_and_auction_type():
    """Bare str tld/auction_type must become lists — char-iteration zeros MatchAny/gate."""
    intent = _make_intent([
        {'name': 'tld', 'value': 'com'},
        {'name': 'auction_type', 'value': '16'},
        {'name': 'price_max', 'value': 49},
    ])
    filters = extract_filters_from_intent(intent)
    assert filters.get('tld') == ['com']
    assert filters.get('auction_type') == ['16']
    assert filters.get('price_max') == 49
    item = _make_item(tld='com', auction_type='16', price=25)
    assert item_matches_filters(item, filters) is True


def test_crash_guard_no_exception_in_matches():
    """_matches must not raise when filter value is already a non-numeric word (belt-and-suspenders)."""
    item = _make_item(price=500)
    # If crash-guard works, price_min='cheapest' is dropped before _matches sees it.
    # To also verify _matches itself survives (belt-and-suspenders), pass a pre-built filter
    # with a valid int value produced by extract_filters_from_intent.
    intent = _make_intent([{'name': 'price_min', 'value': 'cheapest'}])
    filters = extract_filters_from_intent(intent)
    # No exception here
    result = item_matches_filters(item, filters)
    assert result is True  # no price_min filter applied, so item passes


def test_has_hyphen_derived_from_sld_when_payload_missing():
    """Config derive_when_missing: contains_hyphen from SLD when field absent."""
    plain = _make_item(domain_name='foo.com', sld='foo', tld='com')
    plain.pop('has_hyphen', None)
    hyph = _make_item(domain_name='foo-bar.com', sld='foo-bar', tld='com')
    hyph.pop('has_hyphen', None)
    assert item_matches_filters(plain, {'has_hyphen': False})
    assert not item_matches_filters(hyph, {'has_hyphen': False})
    assert item_matches_filters(hyph, {'has_hyphen': True})


# ---------------------------------------------------------------------------
# FIX 2 — exclusion filters
# ---------------------------------------------------------------------------

class TestTldExcludeList:
    def test_excludes_matching_tld(self):
        item = _make_item(tld='com')
        assert not item_matches_filters(item, {'tldExcludeList': ['com']})

    def test_excludes_with_leading_dot(self):
        item = _make_item(tld='com')
        assert not item_matches_filters(item, {'tldExcludeList': ['.com']})

    def test_excludes_case_insensitive(self):
        item = _make_item(tld='COM')
        assert not item_matches_filters(item, {'tldExcludeList': ['com']})

    def test_keeps_non_excluded_tld(self):
        item = _make_item(tld='io')
        assert item_matches_filters(item, {'tldExcludeList': ['com', 'net']})

    def test_empty_list_keeps_all(self):
        item = _make_item(tld='com')
        assert item_matches_filters(item, {'tldExcludeList': []})

    @pytest.mark.parametrize('tld', ['com', '.com', 'COM', '.COM'])
    def test_various_tld_forms_excluded(self, tld):
        item = _make_item(tld='com')
        assert not item_matches_filters(item, {'tldExcludeList': [tld]})


class TestTypeExcludeList:
    def test_excludes_matching_type(self):
        item = _make_item(auction_type='16')
        assert not item_matches_filters(item, {'typeExcludeList': ['16']})

    def test_keeps_non_excluded_type(self):
        item = _make_item(auction_type='20')
        assert item_matches_filters(item, {'typeExcludeList': ['16', '39']})

    def test_case_insensitive(self):
        item = _make_item(auction_type='AUCTION')
        assert not item_matches_filters(item, {'typeExcludeList': ['auction']})

    def test_empty_list_keeps_all(self):
        item = _make_item(auction_type='16')
        assert item_matches_filters(item, {'typeExcludeList': []})


class TestKeywordContainsExclude:
    def test_excludes_sld_containing_term_str(self):
        item = _make_item(sld='spammy')
        assert not item_matches_filters(item, {'keyword_contains_exclude': 'spam'})

    def test_excludes_sld_containing_any_term_list(self):
        item = _make_item(sld='spammy')
        assert not item_matches_filters(item, {'keyword_contains_exclude': ['hello', 'spam']})

    def test_keeps_sld_not_containing_term(self):
        item = _make_item(sld='clean')
        assert item_matches_filters(item, {'keyword_contains_exclude': 'spam'})

    def test_case_insensitive(self):
        item = _make_item(sld='SpAmmy')
        assert not item_matches_filters(item, {'keyword_contains_exclude': 'spam'})

    def test_list_all_non_matching_keeps(self):
        item = _make_item(sld='clean')
        assert item_matches_filters(item, {'keyword_contains_exclude': ['spam', 'junk']})

    def test_empty_string_term_matches_everything(self):
        # Empty string is always a substring — item excluded
        item = _make_item(sld='anything')
        assert not item_matches_filters(item, {'keyword_contains_exclude': ''})

    @pytest.mark.parametrize('value', [['spam'], 'spam'])
    def test_str_and_list_behave_same(self, value):
        item = _make_item(sld='spammy')
        assert not item_matches_filters(item, {'keyword_contains_exclude': value})


# ---------------------------------------------------------------------------
# FIX 3 — keyword_phrase
# ---------------------------------------------------------------------------

class TestKeywordPhrase:
    def test_exact_phrase_matches(self):
        item = _make_item(sld='cloudhosting')
        assert item_matches_filters(item, {'keyword_phrase': 'cloud hosting'})

    def test_phrase_with_hyphen_matches(self):
        item = _make_item(sld='cloudhosting')
        assert item_matches_filters(item, {'keyword_phrase': 'cloud-hosting'})

    def test_phrase_not_in_sld_rejected(self):
        item = _make_item(sld='randomword')
        assert not item_matches_filters(item, {'keyword_phrase': 'cloud hosting'})

    def test_phrase_case_insensitive(self):
        item = _make_item(sld='CloudHosting')
        assert item_matches_filters(item, {'keyword_phrase': 'CLOUD HOSTING'})

    def test_phrase_substring_match(self):
        # phrase is a substring of sld
        item = _make_item(sld='bestcloudhosting')
        assert item_matches_filters(item, {'keyword_phrase': 'cloud hosting'})

    def test_phrase_sld_with_hyphens_matches(self):
        # sld itself has hyphens; they are stripped before comparison
        item = _make_item(sld='cloud-hosting')
        assert item_matches_filters(item, {'keyword_phrase': 'cloud hosting'})

    def test_phrase_no_match_different_words(self):
        item = _make_item(sld='webdesign')
        assert not item_matches_filters(item, {'keyword_phrase': 'cloud hosting'})

    def test_phrase_added_to_filter_entity_names(self):
        from semantic_search.retrieval.structured_retriever import _FILTER_ENTITY_NAMES
        assert 'keyword_phrase' in _FILTER_ENTITY_NAMES

    def test_phrase_extracted_from_intent(self):
        intent = _make_intent([{'name': 'keyword_phrase', 'value': 'cloud hosting'}])
        filters = extract_filters_from_intent(intent)
        assert filters.get('keyword_phrase') == 'cloud hosting'


# ---------------------------------------------------------------------------
# _term_in_compound — leading boundary guard
# ---------------------------------------------------------------------------

class TestTermInCompound:
    def test_word_at_start_matches(self):
        assert _term_in_compound("rent", "rentals") is True

    def test_word_after_hyphen_matches(self):
        assert _term_in_compound("rent", "hi-rentals") is True

    def test_infix_in_unrelated_word_rejected(self):
        # 'rent' must not match 'parent' (p-a-r-ent) — no leading boundary
        assert _term_in_compound("rent", "parent") is False

    def test_infix_in_long_compound_rejected(self):
        # 'age' must not match 'manage' — appears as infix after 'm-an-'
        assert _term_in_compound("age", "manage") is False

    def test_exact_sld_match(self):
        assert _term_in_compound("shop", "shop") is True

    def test_consonant_doubling_trailing_rejected(self):
        # 'shop' should not match 'shopping' (trailing consonant-doubling)
        assert _term_in_compound("shop", "shopping") is False

    def test_consonant_doubling_after_hyphen_rejected(self):
        assert _term_in_compound("run", "top-running") is False

    def test_compound_prefix_matches(self):
        # 'eco' is a genuine leading component of 'ecorentals'
        assert _term_in_compound("eco", "ecorentals") is True

    def test_second_component_after_hyphen_matches(self):
        assert _term_in_compound("rentals", "eco-rentals") is True


# ---------------------------------------------------------------------------
# Exclusion slot names registered in _FILTER_ENTITY_NAMES
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('slot', ['tldExcludeList', 'typeExcludeList', 'keyword_contains_exclude', 'keyword_phrase'])
def test_new_slots_in_filter_entity_names(slot):
    from semantic_search.retrieval.structured_retriever import _FILTER_ENTITY_NAMES
    assert slot in _FILTER_ENTITY_NAMES, f"slot={slot!r} missing from _FILTER_ENTITY_NAMES"


# ---------------------------------------------------------------------------
# None / empty / wrong-type robustness for new filter slots
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('filters,sld,expected', [
    ({'tldExcludeList': []}, 'example', True),
    ({'typeExcludeList': []}, 'example', True),
    ({'keyword_contains_exclude': []}, 'example', True),
    ({'keyword_phrase': ''}, 'example', True),   # empty phrase → '' in sld → matches all, True
    ({'tldExcludeList': [None]}, 'example', True),  # None in list → normalize_tld(None)='none', tld='com' not in {'none'}
])
def test_robustness_edge_cases(filters, sld, expected):
    item = _make_item(sld=sld, tld='com')
    assert item_matches_filters(item, filters) == expected


def test_tld_exclude_list_extracted_from_intent():
    intent = _make_intent([{'name': 'tldExcludeList', 'value': ['com', 'net']}])
    filters = extract_filters_from_intent(intent)
    assert filters.get('tldExcludeList') == ['com', 'net']


def test_type_exclude_list_extracted_from_intent():
    intent = _make_intent([{'name': 'typeExcludeList', 'value': ['16']}])
    filters = extract_filters_from_intent(intent)
    assert filters.get('typeExcludeList') == ['16']


def test_keyword_contains_exclude_extracted_from_intent():
    intent = _make_intent([{'name': 'keyword_contains_exclude', 'value': 'spam'}])
    filters = extract_filters_from_intent(intent)
    assert filters.get('keyword_contains_exclude') == ['spam']


# ---------------------------------------------------------------------------
# Word-count filter — StructuredRetriever with stub segmenter
# ---------------------------------------------------------------------------

import asyncio
from semantic_search.config.models import StructuredRetrievalConfig
from semantic_search.retrieval.structured_retriever import StructuredRetriever


def _make_structured_config(word_count_filter_enabled: bool = True) -> StructuredRetrievalConfig:
    return StructuredRetrievalConfig(enabled=True, top_k=10, backend='memory', word_count_filter_enabled=word_count_filter_enabled, keyword_match_mode='any', unknown_selectable_fields={'traffic_is_unknown': 'monthly_traffic'}, lifecycle_auction_type_map={}, traffic_signal_fields=[])


class _StubSegmenter:
    """Stub that returns a fixed token list per SLD (ignores the domain arg)."""

    def __init__(self, tokens_by_sld: Dict[str, List[str]]) -> None:
        self._tokens = tokens_by_sld

    def segment(self, domain: str) -> Any:
        sld = domain.replace('.placeholder', '')
        tokens = self._tokens.get(sld, [sld, 'placeholder'])

        class _Result:
            pass

        r = _Result()
        r.tokens = tuple(tokens)
        return r


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestWordCountFilter:
    def _make_index_with_items(self, items):
        idx = InMemoryStructuredIndex()
        for item in items:
            idx.add(item)
        return idx

    def test_one_word_candidate_passes_word_count_max_1(self):
        """A 1-word SLD passes word_count_max=1."""
        seg = _StubSegmenter({'shop': ['shop', 'placeholder']})
        idx = self._make_index_with_items([_make_item(sld='shop', score=0.9)])
        retr = StructuredRetriever(_make_structured_config(), idx, word_segmenter=seg)
        intent = _make_intent([{'name': 'word_count_max', 'value': 1}])
        result = _run(retr.retrieve(intent, top_k=10))
        assert len(result.candidates) == 1
        assert result.candidates[0].item_id == 'x1'

    def test_two_word_candidate_excluded_by_word_count_max_1(self):
        """A 2-word SLD is excluded when word_count_max=1."""
        seg = _StubSegmenter({'techshop': ['tech', 'shop', 'placeholder']})
        idx = self._make_index_with_items([_make_item(sld='techshop', score=0.9)])
        retr = StructuredRetriever(_make_structured_config(), idx, word_segmenter=seg)
        intent = _make_intent([{'name': 'word_count_max', 'value': 1}])
        result = _run(retr.retrieve(intent, top_k=10))
        assert len(result.candidates) == 0

    def test_word_count_min_filters_out_short_sld(self):
        """A 1-word SLD fails word_count_min=2."""
        seg = _StubSegmenter({'shop': ['shop', 'placeholder']})
        idx = self._make_index_with_items([_make_item(sld='shop', score=0.9)])
        retr = StructuredRetriever(_make_structured_config(), idx, word_segmenter=seg)
        intent = _make_intent([{'name': 'word_count_min', 'value': 2}])
        result = _run(retr.retrieve(intent, top_k=10))
        assert len(result.candidates) == 0

    def test_segmenter_none_skips_filter_and_warns(self):
        """When segmenter is None, word-count filter is skipped and a WARNING is logged."""
        import semantic_search.retrieval.structured_retriever as _sr_mod
        idx = self._make_index_with_items([_make_item(sld='shop', score=0.9)])
        retr = StructuredRetriever(_make_structured_config(), idx, word_segmenter=None)
        intent = _make_intent([{'name': 'word_count_max', 'value': 1}])
        with patch.object(_sr_mod.logger, 'warning') as mock_warn:
            result = _run(retr.retrieve(intent, top_k=10))
        # Filter skipped → candidate passes through
        assert len(result.candidates) == 1
        assert mock_warn.called
        warn_msg = mock_warn.call_args[0][0]
        assert 'word_count_filter_skipped' in warn_msg

    def test_toggle_off_skips_word_count_filter(self):
        """word_count_filter_enabled=false means the filter is not applied."""
        seg = _StubSegmenter({'techshop': ['tech', 'shop', 'placeholder']})
        idx = self._make_index_with_items([_make_item(sld='techshop', score=0.9)])
        retr = StructuredRetriever(_make_structured_config(word_count_filter_enabled=False), idx, word_segmenter=seg)
        intent = _make_intent([{'name': 'word_count_max', 'value': 1}])
        result = _run(retr.retrieve(intent, top_k=10))
        # Toggle off → filter not applied → 2-word SLD passes
        assert len(result.candidates) == 1

    @pytest.mark.parametrize('slot,bad_value', [
        ('word_count_min', 'few'),
        ('word_count_max', 'many'),
    ])
    def test_crash_guard_word_count_bad_value_dropped(self, slot, bad_value):
        """Non-numeric word_count_* value is dropped by crash-guard."""
        intent = _make_intent([{'name': slot, 'value': bad_value}])
        filters = extract_filters_from_intent(intent)
        assert slot not in filters


def test_word_count_slots_in_filter_entity_names():
    from semantic_search.retrieval.structured_retriever import _FILTER_ENTITY_NAMES
    assert 'word_count_min' in _FILTER_ENTITY_NAMES
    assert 'word_count_max' in _FILTER_ENTITY_NAMES


def test_word_count_slots_in_int_slots():
    from semantic_search.retrieval.structured_retriever import _INT_SLOTS
    assert 'word_count_min' in _INT_SLOTS
    assert 'word_count_max' in _INT_SLOTS


# ---------------------------------------------------------------------------
# startTimeAfter recency filter
# ---------------------------------------------------------------------------

import time as _time


def _make_item_with_listed_at(listed_at_offset_seconds: Optional[float]) -> Dict[str, Any]:
    """Build an item with listed_at = now + offset (None means field absent)."""
    item = _make_item()
    if listed_at_offset_seconds is not None:
        item['listed_at'] = _time.time() + listed_at_offset_seconds
    return item


_CUTOFF_ISO = '2024-01-15T00:00:00Z'
_CUTOFF_EPOCH = 1705276800.0  # 2024-01-15T00:00:00 UTC


@pytest.mark.parametrize('listed_at_epoch,expected', [
    # listed_at is after the cutoff → passes
    (_CUTOFF_EPOCH + 86400, True),
    # listed_at exactly at the cutoff → passes (>= semantics)
    (_CUTOFF_EPOCH, True),
    # listed_at is before the cutoff → excluded
    (_CUTOFF_EPOCH - 1, False),
    (_CUTOFF_EPOCH - 86400, False),
])
def test_startTimeAfter_listed_at_present(listed_at_epoch, expected):
    """Filter applied correctly when listed_at field is present."""
    item = _make_item(listed_at=listed_at_epoch)
    warned: set = set()
    result = item_matches_filters(item, {'startTimeAfter': _CUTOFF_ISO, '_warned': warned})
    assert result == expected


def test_startTimeAfter_no_listed_at_passes():
    """Candidate without listed_at is NOT excluded; WARNING logged once via _warned set."""
    import semantic_search.retrieval.structured_retriever as _sr_mod
    item = _make_item()  # no listed_at field
    warned: set = set()
    with patch.object(_sr_mod.logger, 'warning') as mock_warn:
        result = item_matches_filters(item, {'startTimeAfter': _CUTOFF_ISO, '_warned': warned})
    assert result is True, "candidate with absent listed_at must pass (filter skipped)"
    assert 'startTimeAfter_filter_skipped' in warned
    mock_warn.assert_called_once()
    call_msg = mock_warn.call_args[0][0]
    assert 'startTimeAfter_filter_skipped' in call_msg
    assert 'listed_at_field_absent' in call_msg


def test_startTimeAfter_warning_logged_once_per_retrieve_call():
    """WARNING emitted exactly once when multiple candidates all lack listed_at."""
    import semantic_search.retrieval.structured_retriever as _sr_mod
    items = [_make_item(item_id=f'id{i}') for i in range(5)]  # none have listed_at
    warned: set = set()
    call_count = 0
    original_warning = _sr_mod.logger.warning

    def counting_warning(msg, *args, **kwargs):
        nonlocal call_count
        if 'startTimeAfter_filter_skipped' in str(msg) and 'listed_at_field_absent' in str(msg):
            call_count += 1
        return original_warning(msg, *args, **kwargs)

    with patch.object(_sr_mod.logger, 'warning', side_effect=counting_warning):
        matched = [it for it in items if item_matches_filters(it, {'startTimeAfter': _CUTOFF_ISO, '_warned': warned})]
    assert len(matched) == 5
    assert call_count == 1, f"expected exactly 1 skip warning, got {call_count}"


def test_startTimeAfter_warning_once_via_retrieve():
    """retrieve() path: WARNING logged at most once for absent listed_at across all candidates."""
    import semantic_search.retrieval.structured_retriever as _sr_mod
    items = [_make_item(item_id=f'id{i}', score=0.9 - i * 0.1) for i in range(3)]
    idx = InMemoryStructuredIndex()
    for it in items:
        idx.add(it)
    cfg = _make_structured_config()
    retr = StructuredRetriever(cfg, idx)
    intent = _make_intent([{'name': 'startTimeAfter', 'value': _CUTOFF_ISO}])
    call_count = 0
    original_warning = _sr_mod.logger.warning

    def counting_warning(msg, *args, **kwargs):
        nonlocal call_count
        if 'startTimeAfter_filter_skipped' in str(msg) and 'listed_at_field_absent' in str(msg):
            call_count += 1
        return original_warning(msg, *args, **kwargs)

    with patch.object(_sr_mod.logger, 'warning', side_effect=counting_warning):
        result = _run(retr.retrieve(intent, top_k=10))
    assert len(result.candidates) == 3
    assert call_count == 1, f"expected 1 skip warning via retrieve(), got {call_count}"


@pytest.mark.parametrize('bad_cutoff', [
    'not-a-date',
    '32-99-9999',
    '',
    12345,  # integer — fromisoformat will reject
    'yesterday',
])
def test_startTimeAfter_malformed_cutoff_skips_no_crash(bad_cutoff):
    """Malformed startTimeAfter causes filter skip + WARNING; no exception raised."""
    import semantic_search.retrieval.structured_retriever as _sr_mod
    item = _make_item(listed_at=_CUTOFF_EPOCH + 9999)
    warned: set = set()
    with patch.object(_sr_mod.logger, 'warning') as mock_warn:
        result = item_matches_filters(item, {'startTimeAfter': bad_cutoff, '_warned': warned})
    assert result is True, "malformed cutoff must skip filter, not exclude candidate"
    assert mock_warn.called
    call_msg = mock_warn.call_args[0][0]
    assert 'malformed_cutoff' in call_msg


def test_startTimeAfter_not_in_int_slots():
    """startTimeAfter must NOT appear in _INT_SLOTS."""
    from semantic_search.retrieval.structured_retriever import _INT_SLOTS
    assert 'startTimeAfter' not in _INT_SLOTS


def test_startTimeAfter_not_in_float_slots():
    """startTimeAfter must NOT appear in _FLOAT_SLOTS."""
    from semantic_search.retrieval.structured_retriever import _FLOAT_SLOTS
    assert 'startTimeAfter' not in _FLOAT_SLOTS


def test_startTimeAfter_in_filter_entity_names():
    """startTimeAfter must appear in _FILTER_ENTITY_NAMES."""
    from semantic_search.retrieval.structured_retriever import _FILTER_ENTITY_NAMES
    assert 'startTimeAfter' in _FILTER_ENTITY_NAMES


def test_startTimeAfter_extracted_from_intent_as_string():
    """startTimeAfter passes through extract_filters_from_intent unchanged (not coerced)."""
    intent = _make_intent([{'name': 'startTimeAfter', 'value': _CUTOFF_ISO}])
    filters = extract_filters_from_intent(intent)
    assert filters.get('startTimeAfter') == _CUTOFF_ISO


def test_startTimeAfter_z_suffix_parsed_as_utc():
    """Trailing Z is treated as UTC; a listed_at 1 second after the cutoff passes."""
    cutoff = '2024-01-15T00:00:00Z'
    item = _make_item(listed_at=_CUTOFF_EPOCH + 1)
    assert item_matches_filters(item, {'startTimeAfter': cutoff})


def test_startTimeAfter_with_offset_timezone():
    """An ISO string with explicit UTC offset is parsed correctly."""
    cutoff = '2024-01-15T00:00:00+00:00'
    item_after = _make_item(listed_at=_CUTOFF_EPOCH + 1)
    item_before = _make_item(listed_at=_CUTOFF_EPOCH - 1)
    assert item_matches_filters(item_after, {'startTimeAfter': cutoff})
    assert not item_matches_filters(item_before, {'startTimeAfter': cutoff})


# ---------------------------------------------------------------------------
# StructuredRetrievalConfig validation
# ---------------------------------------------------------------------------

class TestStructuredRetrievalConfigValidation:
    def test_valid_config_construction(self):
        cfg = _make_structured_config()
        assert cfg.word_count_filter_enabled is True

    def test_toggle_false_valid(self):
        cfg = _make_structured_config(word_count_filter_enabled=False)
        assert cfg.word_count_filter_enabled is False

    def test_missing_word_count_filter_enabled_raises(self):
        from semantic_search.config.models import StructuredRetrievalConfig
        with pytest.raises(TypeError):
            StructuredRetrievalConfig(enabled=True, top_k=10, backend='memory')
