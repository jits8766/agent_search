"""Public typeIncludeList label normalization for full-search parity."""
from __future__ import annotations

from semantic_search.config.loader import load_config
from semantic_search.config.models import AgentSearchConfig
from semantic_search.qi.l0_llm_filter_extractor import filters_to_identified
from semantic_search.qi.slot_to_api_param import (
    normalize_type_include_list_for_public,
    transform_slot_value,
)


def test_normalize_mixed_id_and_label_backorder() -> None:
    assert normalize_type_include_list_for_public('25|backorder') == 'backorder'
    assert normalize_type_include_list_for_public(['25', 'backorder']) == 'backorder'


def test_normalize_drops_redundant_auction_when_expiry_present() -> None:
    assert normalize_type_include_list_for_public('auction|expiry') == 'expiry'
    assert normalize_type_include_list_for_public('auction|expiry|listed') == 'expiry|listed'


def test_normalize_strips_partner_standard_ids() -> None:
    out = normalize_type_include_list_for_public('38|39|partner|standard')
    assert out == 'partner|standard'


def test_transform_slot_value_auction_type_uses_normalizer() -> None:
    # Exact singleton ID set {25} -> preferred label among backorder/dropcatch.
    assert transform_slot_value('auction_type', ['25']) == 'backorder'
    assert transform_slot_value('auction_type', '39|closeout') == 'closeout'
    # Orphan ID with no exact label set stays numeric.
    assert transform_slot_value('auction_type', ['16']) == '16'


def test_normalize_keeps_orphan_id_16() -> None:
    assert normalize_type_include_list_for_public('16') == '16'
    assert normalize_type_include_list_for_public(['16']) == '16'


def test_normalize_exact_grounded_id_set_to_premium() -> None:
    assert normalize_type_include_list_for_public('16|38|39') == 'premium'


def test_filters_to_identified_keeps_type_include_list_id() -> None:
    qi = AgentSearchConfig.from_dict(load_config()).qi
    out = filters_to_identified(
        [{'param': 'typeIncludeList', 'value': '16'}, {'param': 'maxPrice', 'value': 49}],
        source=qi.l0_llm_entity.source_tag,
        soft_slot_names=qi.entity_slots.soft_slot_set,
        confidence=qi.l0_llm_entity.confidence,
    )
    by_name = {e['name']: e['value'] for e in out}
    assert by_name['typeIncludeList'] == '16'
    assert by_name['maxPrice'] == 49
