"""Unit tests for config-driven entity_reconcile (Fix E/F cue scrub)."""
from typing import Any, Iterator, List, Optional, Tuple

import pytest

from semantic_search.config.loader import load_config
from semantic_search.config.models import AgentSearchConfig
from semantic_search.contracts import Entity, IntentSlice
from semantic_search.qi.entity_reconcile import (
    apply_post_merge_reconcile,
    hard_entity_names_context,
    reconcile_exact_match_mode,
    reconcile_lifecycle_disjunction,
    reconcile_metric_family_overrides,
    reconcile_stale_listing_polarity,
)
from semantic_search.qi.engine import _merge_extractor_entities
from semantic_search.qi.llm_entity_extractor import _reconcile_dangling_match_mode

_HARD_NAMES: Optional[frozenset] = None
_SLOT_SETS: Optional[Tuple[frozenset, frozenset]] = None


def _hard_names() -> frozenset:
    """qi.entity_slots.hard_entity_names from YAML — no in-test hardcoded taxonomy."""
    global _HARD_NAMES
    if _HARD_NAMES is None:
        slots = AgentSearchConfig.from_dict(load_config()).qi.entity_slots
        assert slots is not None
        _HARD_NAMES = slots.hard_entity_set
    return _HARD_NAMES


def _slot_sets() -> Tuple[frozenset, frozenset]:
    global _SLOT_SETS
    if _SLOT_SETS is None:
        slots = AgentSearchConfig.from_dict(load_config()).qi.entity_slots
        assert slots is not None
        _SLOT_SETS = (slots.soft_slot_set, slots.hard_entity_set)
    return _SLOT_SETS


@pytest.fixture(autouse=True)
def _bind_hard_entity_names() -> Iterator[None]:
    """Bind YAML hard_entity_names so individual reconcile_* helpers can emit entities."""
    with hard_entity_names_context(_hard_names()):
        yield


def _merge(regex, llm, query: str = '', *, llm_completed: bool = True) -> List[Entity]:
    soft, hard = _slot_sets()
    hard_ents, soft_ents = _merge_extractor_entities(
        regex, llm, query, soft, hard, llm_completed=llm_completed,
    )
    return hard_ents + soft_ents


def _entity(name: str, value: Any, source: str = 'L0_llm') -> Entity:
    chip_kind = 'soft' if name.startswith('topic') else 'hard'
    return Entity(name=name, value=value, confidence=0.9, source=source, chip_kind=chip_kind)


def _names(entities: List[Entity]) -> set:
    return {e.name for e in entities}


def _slice(entities: List[Entity]) -> IntentSlice:
    return IntentSlice(query_type='hybrid', entities=entities, confidence=1.0, raw_text='q')


def test_exact_mode_injected_on_bare_cue() -> None:
    out = reconcile_exact_match_mode('keyword match exact', [])
    assert any(e.name == 'keyword_match_mode' and e.value == 'exact' for e in out)


def test_exact_mode_kept_by_dangling_reconcile() -> None:
    ents = [_entity('keyword_match_mode', 'exact')]
    out = _reconcile_dangling_match_mode(ents)
    assert 'keyword_match_mode' in _names(out)


def test_any_mode_still_dropped_with_single_keyword() -> None:
    ents = [_entity('keyword_contains', ['ai']), _entity('keyword_match_mode', 'any')]
    out = _reconcile_dangling_match_mode(ents)
    assert 'keyword_match_mode' not in _names(out)


def test_lifecycle_disjunction_injected_on_or_cue() -> None:
    ents = [_entity('lifecycle_state', 'pending_delete')]
    out = reconcile_lifecycle_disjunction('pending delete or expiring status', ents)
    assert any(e.name == 'lifecycle_disjunction' and e.value is True for e in out)


def test_lifecycle_disjunction_normalized_from_string() -> None:
    ents = [
        _entity('lifecycle_state', 'active'),
        _entity('lifecycle_disjunction', 'pending_delete'),
    ]
    out = reconcile_lifecycle_disjunction(
        'show both active and pending delete status together', ents
    )
    disj = next(e for e in out if e.name == 'lifecycle_disjunction')
    assert disj.value is True


def test_stale_listing_injects_days_listed_min() -> None:
    out = reconcile_stale_listing_polarity('listed more than 10 days ago maybe overlooked', [])
    assert any(e.name == 'days_listed_min' and e.value == 10 for e in out)


def test_stale_listing_weeks_to_days() -> None:
    out = reconcile_stale_listing_polarity('stale over 2 weeks old', [])
    assert any(e.name == 'days_listed_min' and e.value == 14 for e in out)


def test_stale_listing_drops_wrong_polarity_max() -> None:
    ents = [_entity('days_listed_max', 14), _entity('startTimeAfter', '2025-01-01T00:00:00Z')]
    out = reconcile_stale_listing_polarity('stale over 2 weeks old', ents)
    assert 'days_listed_max' not in _names(out)
    assert 'startTimeAfter' not in _names(out)
    assert any(e.name == 'days_listed_min' and e.value == 14 for e in out)


def test_stale_listing_drops_age_and_price_misreads() -> None:
    """'stale over 2 weeks old' must not keep minAge≈weeks/years or minPrice=2."""
    ents = [
        _entity('days_listed_min', 14),
        _entity('domain_age_min', 0.033),
        _entity('price_min', 2.0),
    ]
    out = reconcile_stale_listing_polarity('stale over 2 weeks old', ents)
    assert any(e.name == 'days_listed_min' and e.value == 14 for e in out)
    assert 'domain_age_min' not in _names(out)
    assert 'price_min' not in _names(out)

    # Separate price ceiling survives when value ≠ duration count.
    ents2 = [_entity('days_listed_min', 14), _entity('price_max', 300)]
    out2 = reconcile_stale_listing_polarity('stale over 2 weeks under 300', ents2)
    assert any(e.name == 'price_max' and e.value == 300 for e in out2)


def test_metric_family_majestic_remaps_semrush_ref_domains() -> None:
    ents = [_entity('semrush_ref_domains_max', 20, 'L0_regex')]
    out = reconcile_metric_family_overrides('majestic ref domains under 20', ents)
    assert 'semrush_ref_domains_max' not in _names(out)
    assert any(e.name == 'majestic_ref_domains_max' and e.value == 20 for e in out)


def test_metric_bound_injected_when_no_semrush_slot() -> None:
    out = reconcile_metric_family_overrides('majestic ref domains under 20', [])
    assert any(e.name == 'majestic_ref_domains_max' and e.value == 20 for e in out)


def test_char_letters_only_template() -> None:
    from semantic_search.qi.entity_reconcile import reconcile_char_constraints
    for q in ('domains with letters only', 'pure alpha brands', '4-letter names only'):
        out = reconcile_char_constraints(q, [])
        assert any(e.name == 'has_number' and e.value is False for e in out), q


def test_char_clean_quality_pair_hyphen() -> None:
    from semantic_search.qi.entity_reconcile import reconcile_char_constraints
    for q in ('short and clean', 'clean brandable names', 'brandable and clean'):
        out = reconcile_char_constraints(q, [])
        assert any(e.name == 'has_hyphen' and e.value is False for e in out), q


def test_metric_range_grammar_between_and_to() -> None:
    from semantic_search.qi.entity_reconcile import reconcile_metric_ranges
    out = reconcile_metric_ranges(
        'trust flow between 10 and 30',
        [_entity('bids_min', 10), _entity('bids_max', 30), _entity('traffic_min', 10)],
    )
    assert any(e.name == 'majestic_tf_min' and e.value == 10 for e in out)
    assert any(e.name == 'majestic_tf_max' and e.value == 30 for e in out)
    assert 'bids_min' not in _names(out) and 'traffic_min' not in _names(out)

    out2 = reconcile_metric_ranges('citation flow from 5 to 15', [_entity('majestic_cf_min', 5)])
    assert any(e.name == 'majestic_cf_max' and e.value == 15 for e in out2)


def test_metric_max_only_drops_wrong_polarity_min() -> None:
    from semantic_search.qi.entity_reconcile import reconcile_metric_ranges
    out = reconcile_metric_ranges(
        'cf below 25',
        [_entity('majestic_cf_min', 25)],
    )
    assert 'majestic_cf_min' not in _names(out)
    assert any(e.name == 'majestic_cf_max' and e.value == 25 for e in out)


def test_metric_dual_bound_with_k_suffix() -> None:
    from semantic_search.qi.entity_reconcile import reconcile_metric_ranges
    out = reconcile_metric_ranges(
        'backlinks above 500 and under 2k',
        [],
    )
    assert any(e.name == 'semrush_backlinks_min' and e.value == 501 for e in out)
    assert any(e.name == 'semrush_backlinks_max' and e.value == 2000 for e in out)


def test_metric_dual_does_not_bind_trailing_price_under() -> None:
    """\"cf above 15 under 2k\" — 2k is price, not CF max."""
    from semantic_search.qi.entity_reconcile import reconcile_metric_ranges
    out = reconcile_metric_ranges(
        'tf above 20 and cf above 15 under 2k',
        [_entity('price_max', 2000), _entity('majestic_tf_min', 21), _entity('majestic_cf_min', 16)],
    )
    assert 'majestic_cf_max' not in _names(out)
    assert any(e.name == 'majestic_tf_min' and e.value == 21 for e in out)
    assert any(e.name == 'majestic_cf_min' and e.value == 16 for e in out)


def test_exact_mode_not_injected_when_keyword_phrase_present() -> None:
    ents = [_entity('keyword_phrase', 'fintech')]
    out = reconcile_exact_match_mode('exact phrase fintech in the name', ents)
    assert 'keyword_match_mode' not in _names(out)


def test_soft_estibot_skipped_when_l0_min_present() -> None:
    from semantic_search.qi.entity_reconcile import reconcile_metric_ranges
    out = reconcile_metric_ranges(
        'new keyword low estibot count',
        [_entity('minEstibotDomainCount', 1)],
    )
    assert 'maxEstibotDomainCount' not in _names(out)
    assert any(e.name == 'minEstibotDomainCount' and e.value == 1 for e in out)


def test_ref_domains_plus_keeps_l0_majestic_family() -> None:
    from semantic_search.qi.entity_reconcile import reconcile_metric_ranges
    out = reconcile_metric_ranges(
        'ref domains 50 plus',
        [_entity('majestic_ref_domains_min', 50)],
    )
    assert any(e.name == 'majestic_ref_domains_min' and e.value == 50 for e in out)
    assert 'semrush_ref_domains_min' not in _names(out)


def test_ref_domains_plus_defaults_to_semrush() -> None:
    """Bare \"ref domains N plus\" -> Semrush (prod lexicon default; Majestic when cued)."""
    from semantic_search.qi.entity_reconcile import reconcile_metric_ranges
    out = reconcile_metric_ranges('ref domains 50 plus', [])
    assert any(e.name == 'semrush_ref_domains_min' and e.value == 50 for e in out)
    assert 'majestic_ref_domains_min' not in _names(out)


def test_semrush_ref_domains_above_injects_semrush_exclusive() -> None:
    from semantic_search.qi.entity_reconcile import reconcile_metric_ranges
    out = reconcile_metric_ranges('semrush ref domains above 100', [])
    assert any(e.name == 'semrush_ref_domains_min' and e.value == 101 for e in out)


def test_exclusive_above_bumps_l0_inclusive_floor() -> None:
    from semantic_search.qi.entity_reconcile import reconcile_metric_ranges
    out = reconcile_metric_ranges(
        'tf above 20 and cf above 15 under 2k',
        [_entity('majestic_tf_min', 20), _entity('majestic_cf_min', 15), _entity('price_max', 2000)],
    )
    assert any(e.name == 'majestic_tf_min' and e.value == 21 for e in out)
    assert any(e.name == 'majestic_cf_min' and e.value == 16 for e in out)
    assert 'majestic_cf_max' not in _names(out)


def test_metric_soft_ceiling_from_lexicon() -> None:
    from semantic_search.qi.entity_reconcile import reconcile_metric_ranges
    out = reconcile_metric_ranges('looking for low cf domains', [])
    assert any(e.name == 'majestic_cf_max' and e.value == 10 for e in out)


def test_estibot_ext_bound_and_misattr_drop() -> None:
    from semantic_search.qi.entity_reconcile import reconcile_metric_ranges
    ents = [_entity('bids_min', 50), _entity('minUniqueSearches', 50)]
    out = reconcile_metric_ranges('extension saturation above 30', ents)
    assert any(e.name == 'minEstibotExtCount' and e.value == 31 for e in out)
    assert 'bids_min' not in _names(out)
    assert 'minUniqueSearches' not in _names(out)


def test_dev_ext_count_maps_to_estibot_ext() -> None:
    """\"dev ext count\" -> Dev-namespace Estibot (L0 grounding); not bare tld=dev."""
    from semantic_search.qi.entity_reconcile import reconcile_metric_ranges
    ents = [_entity('tld', 'dev'), _entity('price_max', 5)]
    out = reconcile_metric_ranges('dev ext count above 30', ents)
    assert any(e.name == 'minEstibotExtCountDev' and e.value == 31 for e in out)
    assert 'minEstibotExtCount' not in _names(out)


def test_estibot_soft_high_low() -> None:
    from semantic_search.qi.entity_reconcile import reconcile_metric_ranges
    hi = reconcile_metric_ranges('high estibot ext count', [])
    # Soft qualitative floor (lexicon soft_min=1); no invented large stand-in.
    assert any(e.name == 'minEstibotExtCount' and e.value == 1 for e in hi)
    lo = reconcile_metric_ranges('low estibot count', [])
    # soft_low_min presence floor (not soft_max ceiling).
    assert any(e.name == 'minEstibotDomainCount' and e.value == 0 for e in lo)


def test_estibot_domain_drops_govalue_misattr() -> None:
    from semantic_search.qi.entity_reconcile import reconcile_metric_ranges
    ents = [_entity('govalue_min', 100)]
    out = reconcile_metric_ranges('estibot domain count above 100', ents)
    assert any(e.name == 'minEstibotDomainCount' and e.value == 101 for e in out)
    assert 'govalue_min' not in _names(out)


def test_ungrounded_has_hyphen_dropped_without_cue() -> None:
    from semantic_search.qi.entity_reconcile import reconcile_ungrounded_char_slots
    ents = [_entity('has_hyphen', False), _entity('price_max', 50)]
    out = reconcile_ungrounded_char_slots('find .net and .io domains under $50', ents)
    assert 'has_hyphen' not in _names(out)
    assert any(e.name == 'price_max' for e in out)


def test_grounded_has_hyphen_kept_with_cue() -> None:
    from semantic_search.qi.entity_reconcile import reconcile_ungrounded_char_slots
    ents = [_entity('has_hyphen', False)]
    out = reconcile_ungrounded_char_slots('no hyphens under 100', ents)
    assert any(e.name == 'has_hyphen' and e.value is False for e in out)


def test_cross_slice_broadcast_price_onto_tld_peers() -> None:
    from semantic_search.qi.engine import _broadcast_slots_across_peer_slices
    s1 = _slice([_entity('tld', ['net'], 'L0_regex')])
    s2 = _slice([_entity('tld', ['io'], 'L0_regex'), _entity('price_max', 50, 'L0_regex')])
    out = _broadcast_slots_across_peer_slices(
        [s1, s2], frozenset({'price_max'}), frozenset({'tld'}),
    )
    assert any(e.name == 'price_max' and e.value == 50 for e in out[0].entities)


def test_digit_exclude_noise_scrubbed_with_has_number_false() -> None:
    from semantic_search.qi.entity_reconcile import reconcile_digit_char_exclude_noise
    ents = [
        _entity('has_number', False),
        _entity('keyword_contains_exclude', ['0', '1', '2', 'crypto']),
    ]
    out = reconcile_digit_char_exclude_noise(ents)
    kc = next(e for e in out if e.name == 'keyword_contains_exclude')
    assert kc.value == ['crypto']


def test_metric_family_noop_without_majestic_cue() -> None:
    ents = [_entity('semrush_ref_domains_max', 20, 'L0_regex')]
    out = reconcile_metric_family_overrides('ref domains under 20', ents)
    assert any(e.name == 'semrush_ref_domains_max' for e in out)


def test_merge_applies_metric_remap_with_query() -> None:
    regex = _slice([_entity('semrush_ref_domains_max', 20, 'L0_regex')])
    merged = _merge(regex, None, 'majestic ref domains under 20', llm_completed=False)
    assert any(e.name == 'majestic_ref_domains_max' and e.value == 20 for e in merged)


def test_merge_injects_exact_mode_with_query() -> None:
    merged = _merge(_slice([]), _slice([]), 'keyword match exact')
    assert any(e.name == 'keyword_match_mode' and e.value == 'exact' for e in merged)


def test_reconcile_numeric_preserves_soft_entities() -> None:
    """Single-intent path always runs numeric reconcile — must not wipe soft_entities."""
    from semantic_search.qi.engine import _reconcile_numeric_misattribution_slices

    soft = Entity(
        name='keyword_contains', value='cloud', confidence=0.9,
        source='L0_llm', chip_kind='soft',
    )
    hard = Entity(
        name='tld', value=['com'], confidence=0.9,
        source='L0_llm', chip_kind='hard',
    )
    slices = [
        IntentSlice(
            query_type='hybrid',
            entities=[hard],
            confidence=1.0,
            raw_text='domains containing cloud',
            soft_entities=[soft],
        )
    ]
    out = _reconcile_numeric_misattribution_slices(slices, 'domains containing cloud')
    assert len(out) == 1
    assert [e.name for e in out[0].soft_entities] == ['keyword_contains']
    assert out[0].soft_entities[0].value == 'cloud'
    assert [e.name for e in out[0].entities] == ['tld']


def test_keep_hard_on_suppress() -> None:
    from semantic_search.qi.engine import _keep_hard_entities_on_suppress
    slices = [
        IntentSlice(
            query_type='guidance',
            entities=[
                _entity('keyword_match_mode', 'exact'),
                _entity('topic_include', ['finance'], 'L0_llm'),
            ],
            confidence=0.8,
            raw_text='q',
            soft_entities=[
                Entity(
                    name='lifecycle_state',
                    value='pending_delete',
                    confidence=0.9,
                    source='L0_llm',
                    chip_kind='soft',
                ),
            ],
        )
    ]
    # force soft chip on topic (mis-bucketed onto entities)
    slices[0].entities[1] = Entity(
        name='topic_include', value=['finance'], confidence=0.9, source='L0_llm', chip_kind='soft'
    )
    out = _keep_hard_entities_on_suppress(slices, frozenset({'guidance', 'explore', 'analytics'}))
    names = {e.name for e in out[0].entities}
    assert 'keyword_match_mode' in names
    assert 'topic_include' not in names
    soft_names = {e.name for e in out[0].soft_entities}
    assert soft_names == {'lifecycle_state', 'topic_include'}


def test_apply_post_merge_chain() -> None:
    ents = [_entity('semrush_ref_domains_max', 20, 'L0_regex')]
    out = apply_post_merge_reconcile(
        'majestic ref domains under 20 keyword match exact',
        ents,
        _hard_names(),
    )
    assert any(e.name == 'majestic_ref_domains_max' for e in out)
    assert any(e.name == 'keyword_match_mode' and e.value == 'exact' for e in out)


def test_buy_now_strips_auction_type_20() -> None:
    """Buy-now cue + LLM typeIncludeList/20 -> drop auction_type; keep buy_it_now."""
    from semantic_search.qi.entity_reconcile import reconcile_buy_now_vs_auction_type

    ents = [
        _entity('buy_it_now', True, 'L0_regex'),
        _entity('auction_type', ['20'], 'L0_llm'),
        _entity('tld', ['com'], 'L0_regex'),
        _entity('price_max', 500, 'L0_regex'),
    ]
    out = reconcile_buy_now_vs_auction_type('buy now .com under 500', ents)
    assert 'auction_type' not in _names(out)
    assert 'buy_it_now' in _names(out)
    assert 'tld' in _names(out)


def test_buy_it_now_strips_buynow_label() -> None:
    from semantic_search.qi.entity_reconcile import reconcile_buy_now_vs_auction_type

    ents = [_entity('auction_type', ['buynow'], 'L0_llm'), _entity('buy_it_now', True, 'L0_regex')]
    out = reconcile_buy_now_vs_auction_type('buy it now under 200', ents)
    assert 'auction_type' not in _names(out)


def test_bid_accepted_forces_true_over_llm_false() -> None:
    """LLM isBidAccepted=False must not erase a bid-accepted cue (dropped as inactive otherwise)."""
    from semantic_search.qi.entity_reconcile import reconcile_bid_accepted

    q = 'bid accepted .com or .io under 2k worth jumping in'
    out = reconcile_bid_accepted(q, [_entity('isBidAccepted', False), _entity('tld', ['com', 'io'])])
    assert any(e.name == 'isBidAccepted' and e.value is True for e in out)

    out_miss = reconcile_bid_accepted(q, [_entity('tld', ['com', 'io']), _entity('price_max', 2000)])
    assert any(e.name == 'isBidAccepted' and e.value is True for e in out_miss)

    # End-to-end merge: LLM False + regex True -> post-merge True
    regex = _slice([
        _entity('isBidAccepted', True, 'L0_regex'),
        _entity('tld', ['com', 'io'], 'L0_regex'),
        _entity('price_max', 2.0, 'L0_regex'),
    ])
    llm = _slice([
        _entity('isBidAccepted', False),
        _entity('tld', ['com', 'io']),
        _entity('price_max', 2000.0),
    ])
    merged = _merge(regex, llm, q)
    assert any(e.name == 'isBidAccepted' and e.value is True for e in merged)


def test_zero_bids_and_nobody_bidding_force_exact_empty() -> None:
    """Exact-empty bid count needs BOTH min=0 and max=0; keep price ceiling."""
    from semantic_search.qi.entity_reconcile import reconcile_bid_count_bounds

    # LLM often emits only minBids=0
    out = reconcile_bid_count_bounds(
        'zero bids yet',
        [_entity('bids_min', 0)],
    )
    assert any(e.name == 'bids_min' and e.value == 0 for e in out)
    assert any(e.name == 'bids_max' and e.value == 0 for e in out)

    out2 = reconcile_bid_count_bounds(
        'nobody bidding yet under 300',
        [_entity('bids_min', 0), _entity('price_max', 300)],
    )
    # "nobody bidding" -> minBids=0 only (not the zero/no-bids max band).
    assert any(e.name == 'bids_min' and e.value == 0 for e in out2)
    assert 'bids_max' not in _names(out2)
    assert any(e.name == 'price_max' and e.value == 300 for e in out2)

    out3 = reconcile_bid_count_bounds('10 or more bids', [])
    assert any(e.name == 'bids_min' and e.value == 10 for e in out3)
    assert 'bids_max' not in _names(out3)


def test_exclusive_bid_bounds_more_than_fewer_than() -> None:
    """more than N -> min=N+1; fewer than N -> max=N-1 (LLM often emits inclusive N)."""
    from semantic_search.qi.entity_reconcile import reconcile_bid_count_bounds

    out = reconcile_bid_count_bounds('more than 15 bids', [_entity('bids_min', 15)])
    assert any(e.name == 'bids_min' and e.value == 16 for e in out)
    assert 'bids_max' not in _names(out)

    out2 = reconcile_bid_count_bounds('fewer than 5 bids', [_entity('bids_max', 5)])
    assert any(e.name == 'bids_max' and e.value == 4 for e in out2)
    assert 'bids_min' not in _names(out2)

    # Inject when LLM missed entirely
    out3 = reconcile_bid_count_bounds('more than 15 bids', [])
    assert any(e.name == 'bids_min' and e.value == 16 for e in out3)
    out4 = reconcile_bid_count_bounds('fewer than 5 bids', [])
    assert any(e.name == 'bids_max' and e.value == 4 for e in out4)


def test_buy_now_keeps_expiry_auction_type() -> None:
    """Mixed auction_type: strip only BuyNow values, keep expiry."""
    from semantic_search.qi.entity_reconcile import reconcile_buy_now_vs_auction_type

    ents = [_entity('auction_type', ['buynow', 'expiry'], 'L0_llm')]
    out = reconcile_buy_now_vs_auction_type('buy now only expiry', ents)
    at = next(e for e in out if e.name == 'auction_type')
    assert at.value == ['expiry']


def test_buy_now_noop_without_cue() -> None:
    """Explicit buynow type without buy-now phrasing is left alone."""
    from semantic_search.qi.entity_reconcile import reconcile_buy_now_vs_auction_type

    ents = [_entity('auction_type', ['20'], 'L0_llm')]
    out = reconcile_buy_now_vs_auction_type('fixed price listings under 100', ents)
    assert out == ents


def test_minimum_letters_drops_sld_len() -> None:
    """'minimum 6 letters' -> minLetters only, not minSldLen."""
    from semantic_search.qi.entity_reconcile import reconcile_letters_vs_sld_len

    q = 'minimum 6 letters'
    # Regex maps letters->name_length; LLM may dual-emit both.
    out = reconcile_letters_vs_sld_len(
        q,
        [
            _entity('name_length_min', 6, 'L0_regex'),
            _entity('name_length_max', 6, 'L0_regex'),
            _entity('minLetters', 6),
        ],
    )
    assert any(e.name == 'minLetters' and e.value == 6 for e in out)
    assert 'name_length_min' not in _names(out)
    assert 'name_length_max' not in _names(out)

    # Inject when LLM missed minLetters but regex wrong-family'd to length.
    out2 = reconcile_letters_vs_sld_len(q, [_entity('name_length_min', 6, 'L0_regex')])
    assert any(e.name == 'minLetters' and e.value == 6 for e in out2)
    assert 'name_length_min' not in _names(out2)


def test_chars_plus_keeps_floor_only() -> None:
    """'longer names 10 chars plus' -> minSldLen floor only, no exact max."""
    from semantic_search.qi.entity_reconcile import reconcile_letters_vs_sld_len

    q = 'longer names 10 chars plus'
    out = reconcile_letters_vs_sld_len(
        q,
        [
            _entity('name_length_min', 10, 'L0_regex'),
            _entity('name_length_max', 10, 'L0_regex'),
            _entity('minLetters', 10),
        ],
    )
    assert any(e.name == 'name_length_min' and e.value == 10 for e in out)
    assert 'name_length_max' not in _names(out)
    assert 'minLetters' not in _names(out)


def test_young_unknown_age_under_price_drops_age_and_sld() -> None:
    """'young or unknown age under 500' -> keep price; drop invented maxAge/maxSldLen."""
    from semantic_search.qi.entity_reconcile import apply_post_merge_reconcile

    q = 'young or unknown age under 500'
    out = apply_post_merge_reconcile(
        q,
        [
            _entity('price_max', 500),
            _entity('domain_age_max', 500),
            _entity('name_length_max', 499),
        ],
        _hard_names(),
    )
    assert any(e.name == 'price_max' and e.value == 500 for e in out)
    assert 'domain_age_max' not in _names(out)
    assert 'name_length_max' not in _names(out)


def test_aged_years_keeps_domain_age_with_price() -> None:
    """Years unit present -> domain_age kept even when price ceiling also present."""
    from semantic_search.qi.entity_reconcile import reconcile_bare_price_not_domain_age

    q = 'aged com 15 years min under 3k'
    out = reconcile_bare_price_not_domain_age(
        q,
        [_entity('domain_age_min', 15), _entity('price_max', 3000)],
    )
    assert any(e.name == 'domain_age_min' and e.value == 15 for e in out)
    assert any(e.name == 'price_max' and e.value == 3000 for e in out)


def test_short_cue_keeps_sld_len_with_bare_price() -> None:
    """'short … under 1k' must keep maxSldLen (not drop as bare-price misattr)."""
    from semantic_search.qi.entity_reconcile import reconcile_bare_price_not_sld_len

    q = 'new this week short com or io under 1k'
    out = reconcile_bare_price_not_sld_len(
        q,
        [_entity('name_length_max', 5), _entity('price_max', 1000)],
    )
    assert any(e.name == 'name_length_max' and e.value == 5 for e in out)
    assert any(e.name == 'price_max' and e.value == 1000 for e in out)


def test_govalue_vs_injects_min_zero() -> None:
    """underpriced vs govalue + under N -> inject govalue_min=0 (live LLMJ/qie)."""
    from semantic_search.qi.entity_reconcile import reconcile_govalue_vs_price_under

    q = 'underpriced vs govalue com or io under 2k'
    out = reconcile_govalue_vs_price_under(
        q,
        [
            _entity('price_below_market', True),
            _entity('price_max', 1999),
            _entity('tld', ['com', 'io']),
        ],
    )
    assert any(e.name == 'govalue_min' and int(e.value) == 0 for e in out)
    assert any(e.name == 'price_max' and e.value == 1999 for e in out)


def test_traffic_visitor_soft_inject_if_absent() -> None:
    from semantic_search.qi.entity_reconcile import reconcile_traffic_visitor_soft

    out = reconcile_traffic_visitor_soft('premium ai domain with real visitors', [])
    assert any(e.name == 'has_web_traffic_signal' and e.value is True for e in out)
    # Never overwrite existing traffic slot.
    kept = reconcile_traffic_visitor_soft(
        'premium ai domain with real visitors',
        [_entity('traffic_min', 500)],
    )
    assert not any(e.name == 'has_web_traffic_signal' for e in kept)


def test_exact_letter_length_inject() -> None:
    from semantic_search.qi.entity_reconcile import reconcile_exact_letter_length

    out = reconcile_exact_letter_length('four letter .com no numbers', [])
    assert any(e.name == 'name_length_min' and e.value == 4 for e in out)
    assert any(e.name == 'name_length_max' and e.value == 4 for e in out)


def test_price_between_inject() -> None:
    from semantic_search.qi.entity_reconcile import reconcile_price_between

    out = reconcile_price_between('domains between 500 and 1000', [])
    assert any(e.name == 'price_min' and e.value == 500 for e in out)
    assert any(e.name == 'price_max' and e.value == 1000 for e in out)


def test_budget_prefixed_price_inject_in_dollar() -> None:
    from semantic_search.qi.entity_reconcile import reconcile_budget_prefixed_price_inject

    out = reconcile_budget_prefixed_price_inject('find .net domains in $50', [])
    assert any(e.name == 'price_max' and e.value == 50 for e in out)
    assert not any(e.name == 'price_min' for e in out)


def test_budget_prefixed_price_inject_around_band() -> None:
    from semantic_search.qi.entity_reconcile import reconcile_budget_prefixed_price_inject

    out = reconcile_budget_prefixed_price_inject('domains around $75', [])
    assert any(e.name == 'price_min' and e.value == 75 for e in out)
    assert any(e.name == 'price_max' and e.value == 75 for e in out)


def test_budget_prefixed_price_post_merge_parity() -> None:
    """Full-search post-merge must surface maxPrice for ``in $50`` (qie_only parity)."""
    out = apply_post_merge_reconcile(
        'find .net domains in $50', [], _hard_names(),
    )
    assert any(e.name == 'price_max' and e.value == 50 for e in out)


def test_budget_prefixed_price_skips_bare_in_without_dollar() -> None:
    from semantic_search.qi.entity_reconcile import reconcile_budget_prefixed_price_inject

    assert reconcile_budget_prefixed_price_inject('domains in 50', []) == []
    assert reconcile_budget_prefixed_price_inject('ending in 50', []) == []
    assert reconcile_budget_prefixed_price_inject('auction type in 16', []) == []


def test_ending_tonight_inject() -> None:
    from semantic_search.qi.entity_reconcile import reconcile_ending_urgency

    out = reconcile_ending_urgency('auction domains ending tonight under 1k', [])
    assert any(e.name == 'time_remaining_max' and e.value == 86400 for e in out)


def test_similar_to_brands_inject() -> None:
    from semantic_search.qi.entity_reconcile import reconcile_similar_to_brands

    out = reconcile_similar_to_brands('domains like stripe or plaid', [])
    ent = next(e for e in out if e.name == 'similar_to')
    assert set(ent.value) == {'stripe', 'plaid'}


def test_investor_below_market_inject() -> None:
    from semantic_search.qi.entity_reconcile import reconcile_investor_below_market

    out = reconcile_investor_below_market('hidden gem domains', [])
    assert any(e.name == 'price_below_market' and e.value is True for e in out)


def test_no_vibe_exclude_inject() -> None:
    from semantic_search.qi.entity_reconcile import reconcile_no_vibe_exclude

    out = reconcile_no_vibe_exclude('fintech under 2k no crypto vibe', [])
    ent = next(e for e in out if e.name == 'keyword_contains_exclude')
    assert ent.value == ['crypto']


def test_word_count_phrase_inject() -> None:
    from semantic_search.qi.entity_reconcile import reconcile_word_count_phrase

    out = reconcile_word_count_phrase('one word .io under 3k', [])
    assert any(e.name == 'word_count_min' and e.value == 1 for e in out)
    assert any(e.name == 'word_count_max' and e.value == 1 for e in out)


def test_pending_delete_lifecycle_inject() -> None:
    from semantic_search.qi.entity_reconcile import reconcile_pending_delete_lifecycle

    out = reconcile_pending_delete_lifecycle('pending delete domains worth grabbing', [])
    assert any(e.name == 'lifecycle_state' and e.value == 'pending_delete' for e in out)


def test_added_recently_inject() -> None:
    from semantic_search.qi.entity_reconcile import reconcile_added_recently

    out = reconcile_added_recently('climate tech domains added recently', [])
    assert any(e.name == 'startTimeAfter' and e.value == '-7d' for e in out)


def test_calendar_window_inject_without_listed_verb() -> None:
    """Lookback phrases inject startTimeAfter without requiring listed/added verbs."""
    from semantic_search.qi.entity_reconcile import reconcile_added_recently

    out = reconcile_added_recently('tld performance in the last week under 50', [])
    assert any(e.name == 'startTimeAfter' and e.value == '-7d' for e in out)
    out_num = reconcile_added_recently('volume over the past 14 days', [])
    assert any(e.name == 'startTimeAfter' and e.value == '-14d' for e in out_num)


def test_suite_enrichers_do_not_overwrite() -> None:
    """Inject-if-absent: existing L0 values win."""
    from semantic_search.qi.entity_reconcile import apply_post_merge_reconcile

    existing = [
        _entity('has_web_traffic_signal', True),
        _entity('name_length_min', 5),
        _entity('name_length_max', 5),
        _entity('price_min', 100),
        _entity('price_max', 200),
        _entity('similar_to', ['notion']),
        _entity('price_below_market', True),
    ]
    q = (
        'four letter domains between 500 and 1000 like stripe or plaid '
        'with real visitors hidden gem ending tonight'
    )
    out = apply_post_merge_reconcile(q, existing, _hard_names())
    assert next(e.value for e in out if e.name == 'name_length_min') == 5
    assert next(e.value for e in out if e.name == 'price_min') == 100
    assert next(e.value for e in out if e.name == 'similar_to') == ['notion']


def test_l0_prompt_has_additive_rules_and_fewshots() -> None:
    from semantic_search.qi.l0_llm_filter_extractor import L0_FILTER_BATCH_PROMPT

    assert '10) Visitor language' in L0_FILTER_BATCH_PROMPT
    assert '18) Conditional price' in L0_FILTER_BATCH_PROMPT
    assert 'similar_to: brand/name similarity' in L0_FILTER_BATCH_PROMPT
    assert 'domains like stripe or plaid' in L0_FILTER_BATCH_PROMPT


def test_merge_buy_now_drops_llm_type_include() -> None:
    """Nonempty LLM: regex discarded; buy-now cue scrubs auction_type=20; may inject buy_it_now."""
    regex = _slice([_entity('buy_it_now', True, 'L0_regex'), _entity('price_max', 999, 'L0_regex')])
    llm = _slice([_entity('auction_type', ['20'], 'L0_llm'), _entity('price_max', 200, 'L0_llm')])
    merged = _merge(regex, llm, 'buy it now under 200')
    assert 'auction_type' not in _names(merged)
    assert all(e.source != 'L0_regex' for e in merged)
    assert any(e.name == 'price_max' and e.value == 200 for e in merged)


def test_merge_buy_now_regex_when_llm_unavailable() -> None:
    """LLM unavailable -> regex buy_it_now + price kept as whole-result fallback."""
    regex = _slice([_entity('buy_it_now', True, 'L0_regex'), _entity('price_max', 200, 'L0_regex')])
    merged = _merge(regex, None, 'buy it now under 200', llm_completed=False)
    assert any(e.name == 'buy_it_now' and e.value is True for e in merged)
    assert any(e.name == 'price_max' and e.value == 200 for e in merged)


def test_merge_soft_topic_include_across_slices() -> None:
    from semantic_search.contracts import IntentSlice
    from semantic_search.qi.engine import _merge_list_valued_entities

    soft_a = Entity(
        name='topic_include', value='finance', confidence=0.9,
        source='L0_llm', chip_kind='soft',
    )
    soft_b = Entity(
        name='topic_include', value='legal', confidence=0.9,
        source='L0_llm', chip_kind='soft',
    )
    slices = [
        IntentSlice(query_type='hybrid', entities=[], confidence=0.9, raw_text='a', soft_entities=[soft_a]),
        IntentSlice(query_type='hybrid', entities=[], confidence=0.9, raw_text='b', soft_entities=[soft_b]),
    ]
    out = _merge_list_valued_entities(slices, frozenset({'topic_include'}))
    soft_vals = [e.value for s in out for e in (s.soft_entities or []) if e.name == 'topic_include']
    assert soft_vals == [['finance', 'legal']]


def test_drop_keyword_overlapping_topics() -> None:
    from semantic_search.contracts import IntentSlice
    from semantic_search.qi.engine import _drop_keyword_overlapping_topics

    soft = [
        Entity(name='keyword_contains', value='legal', confidence=0.9, source='L0_llm', chip_kind='soft'),
        Entity(name='topic_include', value=['finance', 'legal'], confidence=0.9, source='L0_llm', chip_kind='soft'),
    ]
    slices = [IntentSlice(query_type='hybrid', entities=[], confidence=0.9, raw_text='q', soft_entities=soft)]
    out = _drop_keyword_overlapping_topics(slices)
    names = {e.name for e in (out[0].soft_entities or [])}
    assert 'keyword_contains' not in names
    assert 'topic_include' in names


def test_leading_niche_topic_seo_domains() -> None:
    from semantic_search.qi.entity_reconcile import reconcile_leading_niche_topic

    out = reconcile_leading_niche_topic('good seo domains cheap', [])
    assert any(e.name == 'topic_include' and 'seo' in e.value for e in out)


def test_b2b_saas_or_fintech_merges_topics() -> None:
    from semantic_search.qi.entity_reconcile import reconcile_b2b_saas_or_fintech

    q = 'looking for b2b saas or maybe fintech domain under 2k'
    out = reconcile_b2b_saas_or_fintech(q, [_entity('topic_include', ['b2b_saas'])])
    ti = next(e for e in out if e.name == 'topic_include')
    assert set(ti.value) == {'b2b_saas', 'fintech'}


def test_ai_and_fintech_co_topic() -> None:
    from semantic_search.qi.entity_reconcile import reconcile_ai_or_fintech_topic

    q = 'browse ai and fintech domains together'
    out = reconcile_ai_or_fintech_topic(q, [_entity('topic_include', ['ai'])])
    ti = next(e for e in out if e.name == 'topic_include')
    assert set(ti.value) == {'ai', 'fintech'}


def test_lifecycle_scrub_dropping_backorder() -> None:
    from semantic_search.qi.entity_reconcile import reconcile_scrub_lifecycle_spurious_backorder

    q = 'find me expired or dropping domain that has traffic'
    ents = [
        _entity('lifecycle_state', ['expired', 'pending_delete']),
        _entity('lifecycle_disjunction', True),
        _entity('auction_type', ['backorder']),
        _entity('has_web_traffic_signal', True),
    ]
    out = reconcile_scrub_lifecycle_spurious_backorder(q, ents)
    assert not any(e.name == 'auction_type' for e in out)


def test_available_now_skips_topic_only_fluff() -> None:
    from semantic_search.qi.entity_reconcile import reconcile_available_now_lifecycle

    out = reconcile_available_now_lifecycle(
        'cool startup names available now',
        [_entity('topic_include', ['startup'])],
    )
    assert not any(e.name == 'lifecycle_state' for e in out)

