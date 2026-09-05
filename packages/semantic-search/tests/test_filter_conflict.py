"""Tests for filter-conflict detection + the FilterConflict contract (item 5)."""
import pytest

from semantic_search.contracts import FilterConflict, QueryIntent, IntentSlice
from semantic_search.core.exceptions import ValidationError
from semantic_search.retrieval.structured_retriever import detect_filter_conflicts

_MSGS = {'range_inverted': '{field}: minimum {min} exceeds maximum {max}.'}


class TestDetectConflicts:
    def test_inverted_price_range(self):
        conflicts = detect_filter_conflicts({'price_min': 500, 'price_max': 10}, _MSGS)
        assert len(conflicts) == 1
        assert conflicts[0].kind == 'range_inverted'
        assert set(conflicts[0].slots) == {'price_min', 'price_max'}
        assert '500' in conflicts[0].message and '10' in conflicts[0].message

    def test_inverted_length_range(self):
        # Query 22: "under 5 chars and at least 15 chars long".
        conflicts = detect_filter_conflicts({'name_length_min': 15, 'name_length_max': 5}, _MSGS)
        assert len(conflicts) == 1
        assert set(conflicts[0].slots) == {'name_length_min', 'name_length_max'}

    def test_valid_range_no_conflict(self):
        assert detect_filter_conflicts({'price_min': 10, 'price_max': 500}, _MSGS) == []

    def test_non_numeric_skipped(self):
        assert detect_filter_conflicts({'price_min': 'x', 'price_max': 5}, _MSGS) == []

    def test_only_min_present_no_conflict(self):
        assert detect_filter_conflicts({'traffic_min': 1000}, _MSGS) == []


class TestFilterConflictContract:
    def test_from_dict_roundtrip(self):
        fc = FilterConflict.from_dict({'kind': 'range_inverted', 'slots': ['price_min', 'price_max'], 'message': 'bad'})
        assert fc.kind == 'range_inverted'

    def test_from_dict_missing_key_rejected(self):
        with pytest.raises(ValidationError):
            FilterConflict.from_dict({'kind': 'range_inverted', 'slots': ['price_min']})

    def test_bad_kind_rejected(self):
        with pytest.raises(ValidationError):
            FilterConflict(kind='nope', slots=['a'], message='m')

    def test_empty_slots_rejected(self):
        with pytest.raises(ValidationError):
            FilterConflict(kind='range_inverted', slots=[], message='m')

    def test_query_intent_carries_conflicts(self):
        qi = QueryIntent(
            request_id='req_x', raw_query='q', normalized_query='q', query_type='hybrid',
            confidence=0.9, decision_tier='L0_entity',
            slices=[IntentSlice(query_type='hybrid', entities=[], confidence=0.9, raw_text='q')],
            decision_cost_usd=0.0,
            conflicts=[FilterConflict(kind='range_inverted', slots=['price_min', 'price_max'], message='m')],
        )
        assert len(qi.conflicts) == 1

    def test_query_intent_rejects_bad_conflicts(self):
        with pytest.raises(ValidationError):
            QueryIntent(
                request_id='req_x', raw_query='q', normalized_query='q', query_type='hybrid',
                confidence=0.9, decision_tier='L0_entity',
                slices=[IntentSlice(query_type='hybrid', entities=[], confidence=0.9, raw_text='q')],
                decision_cost_usd=0.0, conflicts=['not-a-conflict'],
            )
