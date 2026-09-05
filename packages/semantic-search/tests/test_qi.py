"""Unit tests for qi/engine.py dedup helpers.

Coverage matrix (per testing.mdc §7):

_entity_norm_key:
- list_value_sorted_lowercased              -> TestEntityNormKey::test_list_value_sorted_lowercased
- scalar_value_lowercased                   -> TestEntityNormKey::test_scalar_value_lowercased
- int_scalar_coerced_to_str                 -> TestEntityNormKey::test_int_scalar_coerced_to_str

_dedup_entities (single-slice helper):
- empty_list_returns_empty                  -> TestDedupEntities::test_empty_list_returns_empty
- no_duplicates_unchanged                   -> TestDedupEntities::test_no_duplicates_unchanged
- identical_entities_deduped               -> TestDedupEntities::test_identical_entities_deduped
- case_insensitive_dedup                    -> TestDedupEntities::test_case_insensitive_dedup
- first_occurrence_kept                     -> TestDedupEntities::test_first_occurrence_kept
- different_values_both_kept               -> TestDedupEntities::test_different_values_both_kept

_deconflict_numeric_auction_type_price:
- empty_numeric_ids_passthrough            -> TestDeconflictNumericAuctionTypePrice::test_empty_numeric_ids_passthrough
- no_auction_type_passthrough              -> TestDeconflictNumericAuctionTypePrice::test_no_auction_type_passthrough
- non_numeric_auction_type_passthrough     -> TestDeconflictNumericAuctionTypePrice::test_non_numeric_auction_type_passthrough
- price_max_equals_numeric_id_removed      -> TestDeconflictNumericAuctionTypePrice::test_price_max_equals_numeric_id_removed
- price_min_equals_numeric_id_removed      -> TestDeconflictNumericAuctionTypePrice::test_price_min_equals_numeric_id_removed
- price_not_matching_id_kept               -> TestDeconflictNumericAuctionTypePrice::test_price_not_matching_id_kept
- float_price_matching_id_removed          -> TestDeconflictNumericAuctionTypePrice::test_float_price_matching_id_removed
- multiple_entities_only_matching_removed  -> TestDeconflictNumericAuctionTypePrice::test_multiple_entities_only_matching_removed

_inject_keyword_match_mode_all:
- no_both_pattern_passthrough              -> TestInjectKeywordMatchModeAll::test_no_both_pattern_passthrough
- no_multi_kw_passthrough                  -> TestInjectKeywordMatchModeAll::test_no_multi_kw_passthrough
- mode_already_present_passthrough         -> TestInjectKeywordMatchModeAll::test_mode_already_present_passthrough
- injects_all_mode_on_kw_slice             -> TestInjectKeywordMatchModeAll::test_injects_all_mode_on_kw_slice
- injection_only_on_first_kw_slice         -> TestInjectKeywordMatchModeAll::test_injection_only_on_first_kw_slice

_dedup_merged_slice_entities (cross-slice dedup):
- two_slices_identical_tld_deduped         -> TestDedupMergedSliceEntities::test_two_slices_identical_tld_deduped
- two_slices_differing_values_both_kept    -> TestDedupMergedSliceEntities::test_two_slices_differing_values_both_kept
- differing_case_treated_as_duplicate      -> TestDedupMergedSliceEntities::test_differing_case_treated_as_duplicate
- order_preserved_first_slice_wins         -> TestDedupMergedSliceEntities::test_order_preserved_first_slice_wins
- single_slice_unchanged                   -> TestDedupMergedSliceEntities::test_single_slice_unchanged
- empty_entities_slice_preserved           -> TestDedupMergedSliceEntities::test_empty_entities_slice_preserved
- slice_ids_preserved_after_dedup          -> TestDedupMergedSliceEntities::test_slice_ids_preserved_after_dedup
- scalar_duplicate_across_slices           -> TestDedupMergedSliceEntities::test_scalar_duplicate_across_slices
- list_duplicate_different_order_deduped   -> TestDedupMergedSliceEntities::test_list_duplicate_different_order_deduped

_dedup_range_slots (most-restrictive wins):
- range_max_slot_takes_minimum_value        -> TestRangeSlotDedup::test_range_max_slot_takes_minimum_value
- range_max_slot_first_wins_when_lower      -> TestRangeSlotDedup::test_range_max_slot_first_wins_when_lower
- range_min_slot_takes_maximum_value        -> TestRangeSlotDedup::test_range_min_slot_takes_maximum_value
- range_min_slot_first_wins_when_higher     -> TestRangeSlotDedup::test_range_min_slot_first_wins_when_higher
- non_range_slot_first_occurrence_wins      -> TestRangeSlotDedup::test_non_range_slot_first_occurrence_wins
- cross_slice_range_max_most_restrictive    -> TestRangeSlotDedup::test_cross_slice_range_max_most_restrictive
- cross_slice_range_min_most_restrictive    -> TestRangeSlotDedup::test_cross_slice_range_min_most_restrictive
- name_length_max_most_restrictive          -> TestRangeSlotDedup::test_name_length_max_most_restrictive
- traffic_min_most_restrictive              -> TestRangeSlotDedup::test_traffic_min_most_restrictive
"""
import pytest

from semantic_search.contracts import Entity, IntentSlice
from semantic_search.qi.engine import (
    _deconflict_numeric_auction_type_price,
    _dedup_entities,
    _dedup_merged_slice_entities,
    _entity_norm_key,
    _inject_keyword_match_mode_all,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entity(name: str, value, confidence: float = 0.9, source: str = 'L2_llm', chip_kind: str = 'hard') -> Entity:
    return Entity(name=name, value=value, confidence=confidence, source=source, chip_kind=chip_kind)


def _slice(entities, query_type: str = 'hybrid', confidence: float = 0.85, raw_text: str = 'q', slice_id: str = 'slc_test') -> IntentSlice:
    return IntentSlice(query_type=query_type, entities=list(entities), confidence=confidence, raw_text=raw_text, slice_id=slice_id)


# ---------------------------------------------------------------------------
# _entity_norm_key
# ---------------------------------------------------------------------------

class TestEntityNormKey:
    def test_list_value_sorted_lowercased(self) -> None:
        e = _entity('tld', ['COM', 'io'])
        name, val = _entity_norm_key(e)
        assert name == 'tld'
        assert val == ('com', 'io')

    def test_scalar_value_lowercased(self) -> None:
        e = _entity('tld', 'COM')
        name, val = _entity_norm_key(e)
        assert name == 'tld'
        assert val == 'com'

    def test_int_scalar_coerced_to_str(self) -> None:
        e = _entity('traffic_min', 1000)
        name, val = _entity_norm_key(e)
        assert name == 'traffic_min'
        assert val == '1000'


# ---------------------------------------------------------------------------
# _dedup_entities (single-list helper)
# ---------------------------------------------------------------------------

class TestDedupEntities:
    def test_empty_list_returns_empty(self) -> None:
        assert _dedup_entities([]) == []

    def test_no_duplicates_unchanged(self) -> None:
        entities = [_entity('tld', 'com'), _entity('price_max', 500)]
        result = _dedup_entities(entities)
        assert len(result) == 2
        assert result[0].name == 'tld'
        assert result[1].name == 'price_max'

    def test_identical_entities_deduped(self) -> None:
        entities = [_entity('tld', 'com', confidence=0.9), _entity('tld', 'com', confidence=0.5)]
        result = _dedup_entities(entities)
        assert len(result) == 1
        assert result[0].confidence == 0.9

    def test_case_insensitive_dedup(self) -> None:
        entities = [_entity('tld', 'COM'), _entity('tld', 'com')]
        result = _dedup_entities(entities)
        assert len(result) == 1
        assert result[0].value == 'COM'

    def test_first_occurrence_kept(self) -> None:
        e1 = _entity('tld', 'com', confidence=0.95)
        e2 = _entity('tld', 'com', confidence=0.4)
        result = _dedup_entities([e1, e2])
        assert result[0] is e1

    def test_different_values_both_kept(self) -> None:
        entities = [_entity('tld', 'com'), _entity('tld', 'io')]
        result = _dedup_entities(entities)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# _dedup_merged_slice_entities (cross-slice dedup)
# ---------------------------------------------------------------------------

class TestDedupMergedSliceEntities:
    def test_two_slices_identical_tld_deduped(self) -> None:
        s1 = _slice([_entity('tld', ['com'])], slice_id='slc_1')
        s2 = _slice([_entity('tld', ['com'])], slice_id='slc_2')
        result = _dedup_merged_slice_entities([s1, s2])
        assert len(result) == 2
        assert len(result[0].entities) == 1
        assert len(result[1].entities) == 0

    def test_two_slices_differing_values_both_kept(self) -> None:
        s1 = _slice([_entity('tld', ['com'])], slice_id='slc_1')
        s2 = _slice([_entity('tld', ['io'])], slice_id='slc_2')
        result = _dedup_merged_slice_entities([s1, s2])
        assert len(result[0].entities) == 1
        assert len(result[1].entities) == 1

    def test_differing_case_treated_as_duplicate(self) -> None:
        s1 = _slice([_entity('tld', 'COM')], slice_id='slc_1')
        s2 = _slice([_entity('tld', 'com')], slice_id='slc_2')
        result = _dedup_merged_slice_entities([s1, s2])
        assert len(result[0].entities) == 1
        assert result[0].entities[0].value == 'COM'
        assert len(result[1].entities) == 0

    def test_order_preserved_first_slice_wins(self) -> None:
        e1 = _entity('traffic_min', 1000, confidence=0.95)
        e2 = _entity('traffic_min', 1000, confidence=0.3)
        s1 = _slice([e1], slice_id='slc_1')
        s2 = _slice([e2], slice_id='slc_2')
        result = _dedup_merged_slice_entities([s1, s2])
        assert result[0].entities[0] is e1
        assert len(result[1].entities) == 0

    def test_single_slice_unchanged(self) -> None:
        e = _entity('tld', 'com')
        s = _slice([e], slice_id='slc_only')
        result = _dedup_merged_slice_entities([s])
        assert len(result) == 1
        assert len(result[0].entities) == 1
        assert result[0].entities[0] is e

    def test_empty_entities_slice_preserved(self) -> None:
        s = _slice([], slice_id='slc_empty')
        result = _dedup_merged_slice_entities([s])
        assert len(result) == 1
        assert result[0].entities == []

    def test_slice_ids_preserved_after_dedup(self) -> None:
        s1 = _slice([_entity('tld', 'com')], slice_id='slc_a')
        s2 = _slice([_entity('tld', 'com')], slice_id='slc_b')
        result = _dedup_merged_slice_entities([s1, s2])
        assert result[0].slice_id == 'slc_a'
        assert result[1].slice_id == 'slc_b'

    def test_scalar_duplicate_across_slices(self) -> None:
        s1 = _slice([_entity('traffic_min', 1000)], slice_id='slc_1')
        s2 = _slice([_entity('traffic_min', 1000)], slice_id='slc_2')
        result = _dedup_merged_slice_entities([s1, s2])
        assert len(result[0].entities) == 1
        assert len(result[1].entities) == 0

    def test_list_duplicate_different_order_deduped(self) -> None:
        # ['io', 'com'] and ['com', 'io'] normalize to same sorted tuple
        s1 = _slice([_entity('tld', ['io', 'com'])], slice_id='slc_1')
        s2 = _slice([_entity('tld', ['com', 'io'])], slice_id='slc_2')
        result = _dedup_merged_slice_entities([s1, s2])
        assert len(result[0].entities) == 1
        assert len(result[1].entities) == 0

    @pytest.mark.parametrize("val1,val2,expect_dedup", [
        ('com', 'COM', True),
        ('com', 'io', False),
        (1000, 1000, True),
        (1000, 2000, False),
        (['com', 'io'], ['IO', 'COM'], True),
        (['com'], ['io'], False),
    ])
    def test_parametrized_dedup_cases(self, val1, val2, expect_dedup: bool) -> None:
        s1 = _slice([_entity('tld', val1)], slice_id='slc_1')
        s2 = _slice([_entity('tld', val2)], slice_id='slc_2')
        result = _dedup_merged_slice_entities([s1, s2])
        if expect_dedup:
            assert len(result[1].entities) == 0, f"expected {val2!r} deduped against {val1!r}"
        else:
            assert len(result[1].entities) == 1, f"expected {val2!r} kept alongside {val1!r}"


# ---------------------------------------------------------------------------
# _deconflict_numeric_auction_type_price
# ---------------------------------------------------------------------------

_NUMERIC_IDS = frozenset({'16', '20', '38', '39'})


class TestDeconflictNumericAuctionTypePrice:
    def test_empty_numeric_ids_passthrough(self) -> None:
        s = _slice([_entity('auction_type', ['16']), _entity('price_max', 16)])
        result = _deconflict_numeric_auction_type_price([s], frozenset())
        assert len(result[0].entities) == 2

    def test_no_auction_type_passthrough(self) -> None:
        s = _slice([_entity('price_max', 16)])
        result = _deconflict_numeric_auction_type_price([s], _NUMERIC_IDS)
        assert len(result[0].entities) == 1
        assert result[0].entities[0].name == 'price_max'

    def test_non_numeric_auction_type_passthrough(self) -> None:
        s = _slice([_entity('auction_type', ['expiry']), _entity('price_max', 16)])
        result = _deconflict_numeric_auction_type_price([s], _NUMERIC_IDS)
        assert len(result[0].entities) == 2

    def test_price_max_equals_numeric_id_removed(self) -> None:
        s = _slice([_entity('auction_type', ['16']), _entity('price_max', 16)])
        result = _deconflict_numeric_auction_type_price([s], _NUMERIC_IDS)
        names = [e.name for e in result[0].entities]
        assert 'price_max' not in names
        assert 'auction_type' in names

    def test_price_min_equals_numeric_id_removed(self) -> None:
        s = _slice([_entity('auction_type', ['20']), _entity('price_min', 20)])
        result = _deconflict_numeric_auction_type_price([s], _NUMERIC_IDS)
        names = [e.name for e in result[0].entities]
        assert 'price_min' not in names

    def test_price_not_matching_id_kept(self) -> None:
        s = _slice([_entity('auction_type', ['16']), _entity('price_max', 500)])
        result = _deconflict_numeric_auction_type_price([s], _NUMERIC_IDS)
        names = [e.name for e in result[0].entities]
        assert 'price_max' in names

    def test_float_price_matching_id_removed(self) -> None:
        s = _slice([_entity('auction_type', ['38']), _entity('price_max', 38.0)])
        result = _deconflict_numeric_auction_type_price([s], _NUMERIC_IDS)
        names = [e.name for e in result[0].entities]
        assert 'price_max' not in names

    def test_multiple_entities_only_matching_removed(self) -> None:
        s = _slice([_entity('auction_type', ['16']), _entity('price_max', 16), _entity('tld', ['com']), _entity('price_max', 500)])
        result = _deconflict_numeric_auction_type_price([s], _NUMERIC_IDS)
        price_entities = [e for e in result[0].entities if e.name == 'price_max']
        assert len(price_entities) == 1
        assert price_entities[0].value == 500


# ---------------------------------------------------------------------------
# _inject_keyword_match_mode_all
# ---------------------------------------------------------------------------

class TestInjectKeywordMatchModeAll:
    def test_no_both_pattern_passthrough(self) -> None:
        s = _slice([_entity('keyword_contains', ['pay', 'fast'])])
        result = _inject_keyword_match_mode_all([s], 'domains containing pay or fast')
        assert not any(e.name == 'keyword_match_mode' for e in result[0].entities)

    def test_no_multi_kw_passthrough(self) -> None:
        s = _slice([_entity('keyword_contains', 'pay')])
        result = _inject_keyword_match_mode_all([s], 'domains containing both pay and fast')
        assert not any(e.name == 'keyword_match_mode' for e in result[0].entities)

    def test_mode_already_present_passthrough(self) -> None:
        s = _slice([_entity('keyword_contains', ['pay', 'fast']), _entity('keyword_match_mode', 'any')])
        result = _inject_keyword_match_mode_all([s], 'domains containing both pay and fast')
        mode_entities = [e for e in result[0].entities if e.name == 'keyword_match_mode']
        assert len(mode_entities) == 1
        assert mode_entities[0].value == 'any'

    def test_injects_all_mode_on_kw_slice(self) -> None:
        s = _slice([_entity('keyword_contains', ['pay', 'fast'])])
        result = _inject_keyword_match_mode_all([s], 'domains containing both pay and fast')
        mode_entities = [e for e in result[0].entities if e.name == 'keyword_match_mode']
        assert len(mode_entities) == 1
        assert mode_entities[0].value == 'all'
        assert mode_entities[0].source == 'fallback'

    def test_injection_only_on_first_kw_slice(self) -> None:
        s1 = _slice([_entity('keyword_contains', ['pay', 'fast'])], slice_id='slc_1')
        s2 = _slice([_entity('keyword_contains', ['pay', 'fast'])], slice_id='slc_2')
        result = _inject_keyword_match_mode_all([s1, s2], 'domains containing both pay and fast')
        assert any(e.name == 'keyword_match_mode' for e in result[0].entities)
        assert not any(e.name == 'keyword_match_mode' for e in result[1].entities)


# ---------------------------------------------------------------------------
# Range slot dedup (most-restrictive wins)
# ---------------------------------------------------------------------------

class TestRangeSlotDedup:
    def test_range_max_slot_takes_minimum_value(self) -> None:
        # Two price_max constraints: keep the tighter (lower) cap
        entities = [_entity('price_max', 1000), _entity('price_max', 500)]
        result = _dedup_entities(entities)
        assert len(result) == 1
        assert result[0].value == 500

    def test_range_max_slot_first_wins_when_lower(self) -> None:
        entities = [_entity('price_max', 300), _entity('price_max', 900)]
        result = _dedup_entities(entities)
        assert len(result) == 1
        assert result[0].value == 300

    def test_range_min_slot_takes_maximum_value(self) -> None:
        # Two price_min constraints: keep the tighter (higher) floor
        entities = [_entity('price_min', 100), _entity('price_min', 500)]
        result = _dedup_entities(entities)
        assert len(result) == 1
        assert result[0].value == 500

    def test_range_min_slot_first_wins_when_higher(self) -> None:
        entities = [_entity('price_min', 800), _entity('price_min', 200)]
        result = _dedup_entities(entities)
        assert len(result) == 1
        assert result[0].value == 800

    def test_non_range_slot_first_occurrence_wins(self) -> None:
        # For categorical slots (e.g. tld), first-wins still applies
        e1 = _entity('tld', 'com', confidence=0.9)
        e2 = _entity('tld', 'com', confidence=0.4)
        result = _dedup_entities([e1, e2])
        assert result[0] is e1

    def test_name_length_max_most_restrictive(self) -> None:
        entities = [_entity('name_length_max', 12), _entity('name_length_max', 6)]
        result = _dedup_entities(entities)
        assert result[0].value == 6

    def test_traffic_min_most_restrictive(self) -> None:
        entities = [_entity('traffic_min', 500), _entity('traffic_min', 2000)]
        result = _dedup_entities(entities)
        assert result[0].value == 2000

    def test_cross_slice_range_max_most_restrictive(self) -> None:
        # price_max=1000 in slice 1 and price_max=500 in slice 2 → slice 1 updated to 500
        s1 = _slice([_entity('price_max', 1000)], slice_id='slc_1')
        s2 = _slice([_entity('price_max', 500)], slice_id='slc_2')
        result = _dedup_merged_slice_entities([s1, s2])
        # s2 entity should be deduplicated out (already represented in s1)
        all_price = [e for slc in result for e in slc.entities if e.name == 'price_max']
        assert len(all_price) == 1
        assert all_price[0].value == 500

    def test_cross_slice_range_min_most_restrictive(self) -> None:
        # price_min=200 in slice 1 and price_min=800 in slice 2 → result value is 800
        s1 = _slice([_entity('price_min', 200)], slice_id='slc_1')
        s2 = _slice([_entity('price_min', 800)], slice_id='slc_2')
        result = _dedup_merged_slice_entities([s1, s2])
        all_price = [e for slc in result for e in slc.entities if e.name == 'price_min']
        assert len(all_price) == 1
        assert all_price[0].value == 800
