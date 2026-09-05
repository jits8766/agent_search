"""Entity grounding tests — static + live inventory contracts and adapt-on-miss.

Coverage matrix (per ``testing.mdc`` §7) for ``semantic_search.qi.grounding``:

``StaticInventoryContract`` (config-fed allowlist):
- known_tlds_lowercased                 -> TestStaticInventoryContract::test_known_tlds_lowercased
- known_auction_types_lowercased        -> TestStaticInventoryContract::test_known_auction_types_lowercased

``LiveInventoryContract`` (live grounding):
- empty_payload_source_falls_back       -> TestLiveInventoryContract::test_empty_payload_source_falls_back
- live_tlds_returned_when_present       -> TestLiveInventoryContract::test_live_tlds_returned_when_present
- ttl_caches_scan                       -> TestLiveInventoryContract::test_ttl_caches_scan
- invalidate_drops_cache                -> TestLiveInventoryContract::test_invalidate_drops_cache
- scan_failure_falls_back               -> TestLiveInventoryContract::test_scan_failure_falls_back
- non_string_tld_skipped                -> TestLiveInventoryContract::test_non_string_tld_skipped
- auction_types_passthrough             -> TestLiveInventoryContract::test_auction_types_passthrough
- ttl_zero_rejected                     -> TestLiveInventoryContract::test_ttl_zero_rejected
- null_source_rejected                  -> TestLiveInventoryContract::test_null_source_rejected

``EntityGrounder.ground`` (existing path + adapt-on-miss):
- pass_through_known_tld                -> TestEntityGrounder::test_pass_through_known_tld
- drop_unknown_tld_with_substitution    -> TestEntityGrounder::test_drop_unknown_tld_with_substitution
- drop_unknown_tld_no_substitution      -> TestEntityGrounder::test_drop_unknown_tld_no_substitution
- partial_drop_keeps_known              -> TestEntityGrounder::test_partial_drop_keeps_known
- last_dropped_resets_per_call          -> TestEntityGrounder::test_last_dropped_resets_per_call
- empty_input_returns_empty             -> TestEntityGrounder::test_empty_input_returns_empty
- non_list_value_passes_through         -> TestEntityGrounder::test_non_list_value_passes_through
- adapt_on_miss_lookup                  -> TestEntityGrounder::test_adapt_on_miss_lookup
- adapt_on_miss_handles_none            -> TestEntityGrounder::test_adapt_on_miss_handles_none
- substitution_keys_lowercased          -> TestEntityGrounder::test_substitution_keys_lowercased
"""
import time

import pytest

from semantic_search.contracts import Entity
from semantic_search.qi.grounding import (
    EntityGrounder,
    LiveInventoryContract,
    StaticInventoryContract,
    ground_hard_entities,
    ground_identified_filters,
    sanitize_identified_filters,
)
from semantic_search.qi.l0_llm_filter_extractor import identified_to_ground_entities


def _ent(name: str, value, source: str = 'L0_entity', chip: str = 'hard') -> Entity:
    return Entity(name=name, value=value, confidence=0.9, source=source, chip_kind=chip)


class TestStaticInventoryContract:
    def test_known_tlds_lowercased(self):
        ic = StaticInventoryContract(tlds=['COM', 'Net', 'io'], auction_types=['expiry'])
        assert ic.known_tlds() == frozenset({'com', 'net', 'io'})

    def test_known_auction_types_lowercased(self):
        ic = StaticInventoryContract(tlds=['com'], auction_types=['Expiry', 'CLOSEOUT'])
        assert ic.known_auction_types() == frozenset({'expiry', 'closeout'})


class TestLiveInventoryContract:
    def test_empty_payload_source_falls_back(self):
        # Cold scan returns empty -> static fallback wins.
        fallback = StaticInventoryContract(tlds=['com', 'net'], auction_types=['expiry'])
        live = LiveInventoryContract(payload_source=lambda: [], fallback=fallback, ttl_seconds=1.0)
        assert live.known_tlds() == frozenset({'com', 'net'})

    def test_live_tlds_returned_when_present(self):
        # Live source has entries -> derived set wins; static is ignored.
        fallback = StaticInventoryContract(tlds=['com'], auction_types=['expiry'])
        live = LiveInventoryContract(
            payload_source=lambda: [{'tld': 'io'}, {'tld': 'AI'}, {'tld': 'io'}],
            fallback=fallback,
            ttl_seconds=10.0,
        )
        # Lowercased + deduped.
        assert live.known_tlds() == frozenset({'io', 'ai'})

    def test_ttl_caches_scan(self):
        # Counter pattern proves the scan is called once and cached for the
        # TTL window. A second call within TTL must NOT re-invoke the source.
        call_count = {'n': 0}

        def source():
            call_count['n'] += 1
            return [{'tld': 'io'}]

        fallback = StaticInventoryContract(tlds=['com'], auction_types=['expiry'])
        live = LiveInventoryContract(payload_source=source, fallback=fallback, ttl_seconds=10.0)
        live.known_tlds()
        live.known_tlds()
        live.known_tlds()
        assert call_count['n'] == 1

    def test_invalidate_drops_cache(self):
        # invalidate() clears the cache so the next call re-scans even
        # within TTL — this is what the snapshot bump hook calls.
        call_count = {'n': 0}

        def source():
            call_count['n'] += 1
            return [{'tld': 'io'}]

        fallback = StaticInventoryContract(tlds=['com'], auction_types=['expiry'])
        live = LiveInventoryContract(payload_source=source, fallback=fallback, ttl_seconds=60.0)
        live.known_tlds()
        live.invalidate()
        live.known_tlds()
        assert call_count['n'] == 2

    def test_scan_failure_falls_back(self):
        # Source raises -> contract MUST NOT propagate the exception (would
        # break the QI cascade). It logs a warning and falls back to static.
        def boom():
            raise RuntimeError("kaboom")

        fallback = StaticInventoryContract(tlds=['com'], auction_types=['expiry'])
        live = LiveInventoryContract(payload_source=boom, fallback=fallback, ttl_seconds=1.0)
        assert live.known_tlds() == frozenset({'com'})

    def test_non_string_tld_skipped(self):
        # Live payloads from production sources may carry None / int / dict
        # values for the tld field — the scan must reject these gracefully.
        fallback = StaticInventoryContract(tlds=['com'], auction_types=['expiry'])
        live = LiveInventoryContract(
            payload_source=lambda: [{'tld': None}, {'tld': 42}, {'tld': 'io'}, {'no_tld_field': True}],
            fallback=fallback,
            ttl_seconds=1.0,
        )
        assert live.known_tlds() == frozenset({'io'})

    def test_auction_types_passthrough(self):
        # Auction types are a closed set — live contract delegates to static.
        fallback = StaticInventoryContract(tlds=['com'], auction_types=['expiry', 'closeout'])
        live = LiveInventoryContract(payload_source=lambda: [{'tld': 'io'}], fallback=fallback, ttl_seconds=1.0)
        assert live.known_auction_types() == frozenset({'expiry', 'closeout'})

    def test_ttl_zero_rejected(self):
        fallback = StaticInventoryContract(tlds=['com'], auction_types=['expiry'])
        with pytest.raises(ValueError, match="ttl_seconds must be a positive number"):
            LiveInventoryContract(payload_source=lambda: [], fallback=fallback, ttl_seconds=0)

    def test_null_source_rejected(self):
        fallback = StaticInventoryContract(tlds=['com'], auction_types=['expiry'])
        with pytest.raises(ValueError, match="non-null payload_source"):
            LiveInventoryContract(payload_source=None, fallback=fallback, ttl_seconds=1.0)


class TestEntityGrounder:
    def test_pass_through_known_tld(self):
        ic = StaticInventoryContract(tlds=['com', 'io'], auction_types=['expiry'])
        g = EntityGrounder(inventory=ic)
        out = g.ground([_ent('tld', ['com', 'io'])])
        assert len(out) == 1
        assert out[0].value == ['com', 'io']
        assert g.last_dropped == []

    def test_drop_unknown_tld_with_substitution(self):
        # `.ai` not in inventory -> substitutions surface.
        ic = StaticInventoryContract(tlds=['com', 'io', 'tech'], auction_types=['expiry'])
        g = EntityGrounder(inventory=ic, tld_substitutions={'ai': ['io', 'tech']})
        out = g.ground([_ent('tld', ['ai'])])
        # Entire entity is dropped because no values survived.
        assert out == []
        # last_dropped exposes the substitution suggestions.
        assert g.last_dropped == [('tld', 'ai', ('io', 'tech'))]

    def test_drop_unknown_tld_no_substitution(self):
        # No mapping configured -> entity dropped, last_dropped records empty
        # alts so the surface knows the drop happened but has no chips to show.
        ic = StaticInventoryContract(tlds=['com'], auction_types=['expiry'])
        g = EntityGrounder(inventory=ic)
        out = g.ground([_ent('tld', ['xyz'])])
        assert out == []
        assert g.last_dropped == [('tld', 'xyz', ())]

    def test_partial_drop_keeps_known(self):
        # Mixed list: known values kept, unknowns dropped + recorded.
        ic = StaticInventoryContract(tlds=['com', 'io'], auction_types=['expiry'])
        g = EntityGrounder(inventory=ic, tld_substitutions={'ai': ['io', 'tech']})
        out = g.ground([_ent('tld', ['com', 'ai', 'xyz'])])
        assert len(out) == 1
        assert out[0].value == ['com']
        # Both unknowns recorded; .ai gets substitutions, .xyz gets ().
        names_dropped = {(d[0], d[1]) for d in g.last_dropped}
        assert names_dropped == {('tld', 'ai'), ('tld', 'xyz')}

    def test_last_dropped_resets_per_call(self):
        # Calling ground() twice MUST NOT accumulate drops across calls
        # (callers read this as the most-recent classification's drop set).
        ic = StaticInventoryContract(tlds=['com'], auction_types=['expiry'])
        g = EntityGrounder(inventory=ic)
        g.ground([_ent('tld', ['xyz'])])
        assert len(g.last_dropped) == 1
        g.ground([_ent('tld', ['com'])])
        assert g.last_dropped == []

    def test_empty_input_returns_empty(self):
        ic = StaticInventoryContract(tlds=['com'], auction_types=['expiry'])
        g = EntityGrounder(inventory=ic)
        assert g.ground([]) == []
        assert g.last_dropped == []

    def test_non_list_value_passes_through(self):
        # `price_max` is a scalar — grounder must not touch non-list entities.
        ic = StaticInventoryContract(tlds=['com'], auction_types=['expiry'])
        g = EntityGrounder(inventory=ic)
        out = g.ground([_ent('price_max', 100)])
        assert len(out) == 1
        assert out[0].value == 100

    def test_adapt_on_miss_lookup(self):
        ic = StaticInventoryContract(tlds=['com'], auction_types=['expiry'])
        g = EntityGrounder(inventory=ic, tld_substitutions={'ai': ['io', 'tech']})
        # Case-insensitive lookup — accepts 'AI' and returns lowercased alts.
        assert g.adapt_on_miss('AI') == ('io', 'tech')
        assert g.adapt_on_miss('unknown') == ()

    def test_adapt_on_miss_handles_none(self):
        # Robustness — None / empty / non-string inputs return empty tuple
        # (never raise).
        ic = StaticInventoryContract(tlds=['com'], auction_types=['expiry'])
        g = EntityGrounder(inventory=ic, tld_substitutions={'ai': ['io']})
        assert g.adapt_on_miss(None) == ()
        assert g.adapt_on_miss('') == ()
        assert g.adapt_on_miss(42) == ()

    def test_substitution_keys_lowercased(self):
        # Construction must lowercase keys + values so production data with
        # mixed case ('AI' -> ['IO', 'Tech']) still hits the lookup.
        ic = StaticInventoryContract(tlds=['com'], auction_types=['expiry'])
        g = EntityGrounder(inventory=ic, tld_substitutions={'AI': ['IO', 'Tech']})
        assert g.adapt_on_miss('ai') == ('io', 'tech')

    def test_backorder_expands_to_25(self):
        """backorder / drop catch -> type 25 (Drop Catch Private Backorder)."""
        ic = StaticInventoryContract(
            tlds=['com'],
            auction_types=['expiry', 'closeout', 'backorder', 'dropcatch', '25', '39'],
        )
        g = EntityGrounder(inventory=ic)
        out = g.ground([_ent('auction_type', ['backorder'])])
        assert out[0].value == ['25']
        out2 = g.ground([_ent('auction_type', ['dropcatch'])])
        assert out2[0].value == ['25']
        out3 = g.ground([_ent('auction_type', ['closeout'])])
        assert out3[0].value == ['39']

    def test_partner_expands_to_38_39_not_16(self):
        """partner auction -> 38|39 only; generic auction co-label must not reintroduce 16."""
        ic = StaticInventoryContract(
            tlds=['com'],
            auction_types=['partner', 'godaddy', 'auction', 'expiry', '16', '20', '38', '39'],
        )
        g = EntityGrounder(inventory=ic)
        out = g.ground([_ent('auction_type', ['partner', 'auction', '16'])])
        assert out[0].value == ['38', '39']
        out_gd = g.ground([_ent('auction_type', ['godaddy', 'auction', '38'])])
        assert out_gd[0].value == ['16', '20']

    def test_firehose_expands_to_37(self):
        ic = StaticInventoryContract(
            tlds=['com'],
            auction_types=['firehose', 'preregistration', '37'],
        )
        g = EntityGrounder(inventory=ic)
        assert g.ground([_ent('auction_type', ['firehose'])])[0].value == ['37']
        assert g.ground([_ent('auction_type', ['preregistration'])])[0].value == ['37']


class TestLiveInventoryContractIntegration:
    def test_live_then_invalidated_recovers_new_tld(self):
        # End-to-end: a brand-new TLD shows up in live data only after the
        # snapshot invalidation hook bumps the cache. This is the integration
        # contract the registry wires.
        items = [{'tld': 'io'}]
        fallback = StaticInventoryContract(tlds=['com'], auction_types=['expiry'])
        live = LiveInventoryContract(payload_source=lambda: list(items), fallback=fallback, ttl_seconds=60.0)
        assert live.known_tlds() == frozenset({'io'})
        # New listing event arrives - without invalidate, the cache hides it.
        items.append({'tld': 'ai'})
        assert live.known_tlds() == frozenset({'io'})
        live.invalidate()
        assert live.known_tlds() == frozenset({'io', 'ai'})


class TestGroundIdentifiedFilters:
    """Phase 1 qie_only: static-inventory ground on FIND TLD/type params only."""

    def test_drops_unknown_tld_keeps_known_and_passthrough(self):
        g = EntityGrounder(
            inventory=StaticInventoryContract(tlds=['io', 'com'], auction_types=['premium', '16']),
        )
        identified = [
            {'name': 'tldIncludeList', 'value': ['io', 'zz'], 'source': 'L0_llm'},
            {'name': 'maxPrice', 'value': 500, 'source': 'L0_llm'},
        ]
        out, drops = ground_identified_filters(identified, g)
        by_name = {e['name']: e['value'] for e in out}
        assert by_name['tldIncludeList'] == ['io']
        assert by_name['maxPrice'] == 500
        assert drops >= 1

    def test_drops_entire_tld_filter_when_all_unknown(self):
        g = EntityGrounder(
            inventory=StaticInventoryContract(tlds=['com'], auction_types=['premium']),
        )
        out, drops = ground_identified_filters(
            [{'name': 'tldIncludeList', 'value': ['zz'], 'source': 'L0_llm'}],
            g,
        )
        assert out == []
        assert drops >= 1

    def test_type_include_grounded_to_public_label(self):
        g = EntityGrounder(
            inventory=StaticInventoryContract(
                tlds=['com'],
                auction_types=['premium', '16'],
            ),
        )
        out, drops = ground_identified_filters(
            [{'name': 'typeIncludeList', 'value': ['premium'], 'source': 'L0_llm'}],
            g,
        )
        assert len(out) == 1
        assert out[0]['name'] == 'typeIncludeList'
        # Expand premium -> IDs then normalize_type_include_list_for_public collapses back.
        assert out[0]['value'] == 'premium'
        assert drops == 0

    def test_drops_unknown_tld_exclude_keeps_known(self):
        g = EntityGrounder(
            inventory=StaticInventoryContract(tlds=['io', 'com'], auction_types=['premium']),
        )
        out, drops = ground_identified_filters(
            [
                {'name': 'tldExcludeList', 'value': ['io', 'zz'], 'source': 'L0_llm'},
                {'name': 'maxPrice', 'value': 100, 'source': 'L0_llm'},
            ],
            g,
        )
        by_name = {e['name']: e['value'] for e in out}
        assert by_name['tldExcludeList'] == ['io']
        assert by_name['maxPrice'] == 100
        assert drops >= 1

    def test_tld_include_exclude_intersection_stripped_both_sides(self):
        g = EntityGrounder(
            inventory=StaticInventoryContract(tlds=['com', 'io'], auction_types=['premium']),
        )
        out, _drops = ground_identified_filters(
            [
                {'name': 'tldIncludeList', 'value': ['com', 'io'], 'source': 'L0_llm'},
                {'name': 'tldExcludeList', 'value': ['com'], 'source': 'L0_llm'},
            ],
            g,
        )
        by_name = {e['name']: e['value'] for e in out}
        assert by_name['tldIncludeList'] == ['io']
        assert 'tldExcludeList' not in by_name

    def test_tld_full_contradiction_omits_both_slots(self):
        g = EntityGrounder(
            inventory=StaticInventoryContract(tlds=['com'], auction_types=['premium']),
        )
        out, _drops = ground_identified_filters(
            [
                {'name': 'tldIncludeList', 'value': ['com'], 'source': 'L0_llm'},
                {'name': 'tldExcludeList', 'value': ['com'], 'source': 'L0_llm'},
                {'name': 'maxPrice', 'value': 50, 'source': 'L0_llm'},
            ],
            g,
        )
        names = {e['name'] for e in out}
        assert 'tldIncludeList' not in names
        assert 'tldExcludeList' not in names
        assert names == {'maxPrice'}

    def test_type_include_exclude_full_contradiction_omits_both(self):
        g = EntityGrounder(
            inventory=StaticInventoryContract(
                tlds=['com'],
                auction_types=['premium', '16'],
            ),
        )
        out, _drops = ground_identified_filters(
            [
                {'name': 'typeIncludeList', 'value': ['premium'], 'source': 'L0_llm'},
                {'name': 'typeExcludeList', 'value': ['premium'], 'source': 'L0_llm'},
                {'name': 'maxPrice', 'value': 50, 'source': 'L0_llm'},
            ],
            g,
        )
        names = {e['name'] for e in out}
        assert 'typeIncludeList' not in names
        assert 'typeExcludeList' not in names
        assert names == {'maxPrice'}

    def test_keyword_include_exclude_intersection_stripped(self):
        g = EntityGrounder(
            inventory=StaticInventoryContract(tlds=['com'], auction_types=['premium']),
        )
        out, _drops = ground_identified_filters(
            [
                {'name': 'keyword_contains', 'value': 'ai|crypto', 'source': 'L0_llm'},
                {'name': 'keyword_contains_exclude', 'value': 'ai', 'source': 'L0_llm'},
            ],
            g,
        )
        by_name = {e['name']: e['value'] for e in out}
        assert by_name['keyword_contains'] == 'crypto'
        assert 'keyword_contains_exclude' not in by_name

    def test_inverted_min_max_price_swapped(self):
        g = EntityGrounder(
            inventory=StaticInventoryContract(tlds=['com'], auction_types=['premium']),
        )
        out, drops = ground_identified_filters(
            [
                {'name': 'minPrice', 'value': 500, 'source': 'L0_llm'},
                {'name': 'maxPrice', 'value': 100, 'source': 'L0_llm'},
            ],
            g,
        )
        by_name = {e['name']: e['value'] for e in out}
        assert by_name['minPrice'] == 100
        assert by_name['maxPrice'] == 500
        assert drops == 0

    def test_sanitize_without_grounder_strips_and_swaps_price(self):
        """No inventory grounder: still strip contradictions + fix inverted prices."""
        out = sanitize_identified_filters(
            [
                {'name': 'tldIncludeList', 'value': ['com'], 'source': 'L0_llm'},
                {'name': 'tldExcludeList', 'value': ['com'], 'source': 'L0_llm'},
                {'name': 'minPrice', 'value': 900, 'source': 'L0_llm'},
                {'name': 'maxPrice', 'value': 100, 'source': 'L0_llm'},
            ],
        )
        by_name = {e['name']: e['value'] for e in out}
        assert 'tldIncludeList' not in by_name
        assert 'tldExcludeList' not in by_name
        assert by_name['minPrice'] == 100
        assert by_name['maxPrice'] == 900

    def test_grounding_preserves_chip_kind_and_confidence(self):
        g = EntityGrounder(
            inventory=StaticInventoryContract(tlds=['io', 'com'], auction_types=['premium']),
        )
        out, _drops = ground_identified_filters(
            [
                {
                    'name': 'tldIncludeList',
                    'value': ['io'],
                    'source': 'L0_llm',
                    'chip_kind': 'hard',
                    'confidence': 0.9,
                },
                {
                    'name': 'keyword_contains',
                    'value': 'ai',
                    'source': 'L0_llm',
                    'chip_kind': 'soft',
                    'confidence': 0.9,
                },
            ],
            g,
        )
        by_name = {e['name']: e for e in out}
        assert by_name['tldIncludeList']['chip_kind'] == 'hard'
        assert by_name['tldIncludeList']['confidence'] == 0.9
        assert by_name['keyword_contains']['chip_kind'] == 'soft'


class TestGroundHardEntities:
    """Entity-native core shared by qie_only (via ground_identified_filters) and
    full-search (QIEngine._ground_slices calls this directly)."""

    def test_drops_unknown_tld_exclude_value(self):
        g = EntityGrounder(
            inventory=StaticInventoryContract(tlds=['io', 'com'], auction_types=['premium']),
        )
        out, drops = ground_hard_entities(
            [_ent('tldExcludeList', ['io', 'zz'])],
            g,
        )
        by_name = {e.name: e.value for e in out}
        assert by_name['tldExcludeList'] == ['io']
        assert drops >= 1

    def test_drops_unknown_type_exclude_value(self):
        # Partial auction_type drops don't populate last_dropped (only a full-slot
        # drop does, per EntityGrounder.ground()) -- assert the value-level filtering.
        g = EntityGrounder(
            inventory=StaticInventoryContract(tlds=['com'], auction_types=['premium', '16']),
        )
        out, _drops = ground_hard_entities(
            [_ent('typeExcludeList', ['premium', 'not_a_type'])],
            g,
        )
        by_name = {e.name: e.value for e in out}
        assert 'typeExcludeList' in by_name
        assert 'not_a_type' not in by_name['typeExcludeList']

    def test_drops_entire_type_exclude_when_all_unknown(self):
        g = EntityGrounder(
            inventory=StaticInventoryContract(tlds=['com'], auction_types=['premium']),
        )
        out, drops = ground_hard_entities(
            [_ent('typeExcludeList', ['not_a_type'])],
            g,
        )
        names = {e.name for e in out}
        assert 'typeExcludeList' not in names
        assert drops >= 1

    def test_strips_tld_include_exclude_conflict(self):
        g = EntityGrounder(
            inventory=StaticInventoryContract(tlds=['com', 'io'], auction_types=['premium']),
        )
        out, _drops = ground_hard_entities(
            [_ent('tld', ['com', 'io']), _ent('tldExcludeList', ['com'])],
            g,
        )
        by_name = {e.name: e.value for e in out}
        assert by_name['tld'] == ['io']
        assert 'tldExcludeList' not in by_name

    def test_fixes_inverted_price_bounds(self):
        g = EntityGrounder(
            inventory=StaticInventoryContract(tlds=['com'], auction_types=['premium']),
        )
        out, drops = ground_hard_entities(
            [_ent('price_min', 500), _ent('price_max', 100)],
            g,
        )
        by_name = {e.name: e.value for e in out}
        assert by_name['price_min'] == 100
        assert by_name['price_max'] == 500
        assert drops == 0

    def test_empty_input_returns_empty(self):
        g = EntityGrounder(
            inventory=StaticInventoryContract(tlds=['com'], auction_types=['premium']),
        )
        out, drops = ground_hard_entities([], g)
        assert out == []
        assert drops == 0

    def test_null_grounder_raises(self):
        with pytest.raises(ValueError):
            ground_hard_entities([_ent('tld', ['com'])], None)


class TestIdentifiedToGroundEntities:
    """Dict -> Entity direction feeding ground_hard_entities from ground_identified_filters."""

    def test_round_trip_preserves_exclude_and_price_slots(self):
        identified = [
            {'name': 'tldExcludeList', 'value': ['io'], 'source': 'L0_llm', 'confidence': 0.8, 'chip_kind': 'hard'},
            {'name': 'typeExcludeList', 'value': ['premium'], 'source': 'L0_llm', 'confidence': 0.7, 'chip_kind': 'hard'},
            {'name': 'minPrice', 'value': 100, 'source': 'L0_llm', 'confidence': 0.9, 'chip_kind': 'hard'},
            {'name': 'maxPrice', 'value': 500, 'source': 'L0_llm', 'confidence': 0.9, 'chip_kind': 'hard'},
        ]
        entities = identified_to_ground_entities(identified)
        by_name = {e.name: e for e in entities}
        assert by_name['tldExcludeList'].value == ['io']
        assert by_name['typeExcludeList'].value == ['premium']
        assert by_name['price_min'].value == 100
        assert by_name['price_max'].value == 500
        assert by_name['tldExcludeList'].confidence == 0.8
        assert by_name['typeExcludeList'].source == 'L0_llm'

    def test_drops_qualitative_sentinel_values(self):
        entities = identified_to_ground_entities(
            [{'name': 'tldIncludeList', 'value': None, 'source': 'L0_llm'}],
        )
        assert entities == []
