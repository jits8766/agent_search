"""Tests for SearchParamsConfig and app._slim_result projection."""
import types

import pytest

from semantic_search import app
from semantic_search.config.models import SearchParamsConfig
from semantic_search.core.exceptions import ConfigurationError


def _complement_dict() -> dict:
    return {
        'enabled': True,
        'merge_explore_rails': True,
        'merge_explore_rails_query_types': ['analytics', 'explore', 'guidance'],
        'merge_rrf_k': 60,
        'merge_explore_rails_only_when_primary_short': True,
        'merge_explore_rails_primary_enough_fraction': 1.0,
        'strip_temporal_when_clickhouse_available': True,
        'ensure_nonempty': True,
        'force_semantic_when_empty': True,
        'force_semantic_top_k': 50,
        'rank_leanings': {
            'enabled': True,
            'max_bonus': 0.08,
            'weight_guidance': 0.5,
            'weight_analytics': 0.5,
            'query_types': ['analytics', 'guidance'],
        },
    }


def _find_listing_dict() -> dict:
    return {
        'iso_timestamp_fields': ['ends_at', 'auction_end_time', 'end_time', 'data_update_time'],
        'iso_fallback_source': 'ends_at',
        'iso_passthrough_string_fields': ['data_update_time'],
        'money_display_currency_prefix': '$',
        'money_display_template': '{value}',
        'money_display_value_placeholder': '{value}',
        'money_display_number_format': ',.2f',
        'money_display_when_null': '',
        'domain_name_payload_keys': ['fqdn', 'domain_name'],
        'money_display_pairs': [
            {'source': 'auction_price', 'targets': ['auction_price_display', 'auction_price_display_usd']},
        ],
    }


def _find_wire_dict() -> dict:
    return {
        'prefer_keywords_for_query': True,
        'empty_query_fallback': '*',
        'max_keyword_terms': 3,
        'keyword_term_separator': ' ',
        'set_use_semantic_search_when_keywords': True,
        'use_semantic_search_param': 'useSemanticSearch',
        'use_semantic_search_value': 'true',
    }


def _search_params_dict() -> dict:
    return {
        'top_k_cap': 20,
        'diversity_lambda': 0.3,
        'search_timeout_seconds': 8.0,
        'analytics_timeout_seconds': 12.0,
        'analytics_total_budget_seconds': 20.0,
        'explore_fallback_timeout_seconds': 3.0,
        'speculative_analytics_start': True,
        'speculative_analytics_l1_confidence_threshold': 0.75,
        'analytics_failure_explore_fallback': True,
        'guidance_analytics_crosstype_enabled': True,
        'guidance_analytics_crosstype_timeout_seconds': 3.0,
        'result_fields': ['tld', 'sld', 'ends_at'],
        'find_listing': _find_listing_dict(),
        'find_wire': _find_wire_dict(),
        'ranked_results_complement': _complement_dict(),
        'timeout_fallback': {
            'rail_first': True,
            'qdrant_rails_when_ch_empty': True,
            'qdrant_rails_timeout_seconds': 2.0,
            'consume_search_explore_prewarm': True,
            'search_explore_prewarm_wait_seconds': 2.0,
        },
        'explore_fallback_cache_ttl_seconds': 300.0,
        'explore_fallback_semantic_prefer_min_results': 5,
        'explore_fallback_semantic_prefer_min_score': 0.45,
        'analytics_skip_uncorroborated_detour': True,
        'analytics_budget_keywords': ['average', 'compare'],
        'permanently_unavailable_columns': [],
        'qie_only_mode': False,
        'qie_l0_filter_cache': {'max_entries': 256, 'ttl_seconds': 600},
        'hybrid_prewarm_enabled': False,
        'overlap_preprocess_with_classify': True,
        'auction_tiebreak': {'enabled': False, 'max_bonus': 0.06, 'urgency_horizon_hours': 72.0, 'weight_urgency': 0.5, 'weight_low_competition': 0.2, 'weight_value': 0.3},
        'analytics_unavailable_hybrid_notice': 'Analytics unavailable — hybrid results.',
        'analytics_unavailable_explore_notice': 'Analytics unavailable — explore results.',
        'analytics_timeout_hybrid_notice': 'Analytics timeout — hybrid results.',
        'analytics_timeout_explore_notice': 'Analytics timeout — explore results.',
        'analytics_detour_skip_notice': 'Detour skipped — explore results.',
        'analytics_connection_unavailable_notice': 'Analytics connection unavailable.',
        'analytics_failure_with_explore_notice': 'Analytics failed — explore results.',
        'analytics_failure_empty_explore_notice': 'Analytics failed — empty explore.',
    }


def _search_params_kwargs(**overrides):
    from semantic_search.config.models import (
        AuctionTiebreakConfig,
        FindListingConfig,
        FindWireConfig,
        QieL0FilterCacheConfig,
        RankedResultsComplementConfig,
        TimeoutFallbackConfig,
    )
    base = dict(
        top_k_cap=10,
        diversity_lambda=0.5,
        search_timeout_seconds=5.0,
        analytics_timeout_seconds=6.0,
        analytics_total_budget_seconds=12.0,
        explore_fallback_timeout_seconds=4.5,
        speculative_analytics_start=True,
        speculative_analytics_l1_confidence_threshold=0.7,
        analytics_failure_explore_fallback=True,
        guidance_analytics_crosstype_enabled=True,
        guidance_analytics_crosstype_timeout_seconds=3.0,
        result_fields=['tld'],
        find_listing=FindListingConfig.from_dict(_find_listing_dict()),
        find_wire=FindWireConfig.from_dict(_find_wire_dict()),
        ranked_results_complement=RankedResultsComplementConfig.from_dict(_complement_dict()),
        timeout_fallback=TimeoutFallbackConfig(
            rail_first=True,
            qdrant_rails_when_ch_empty=True,
            qdrant_rails_timeout_seconds=2.0,
            consume_search_explore_prewarm=True,
            search_explore_prewarm_wait_seconds=2.0,
        ),
        explore_fallback_cache_ttl_seconds=300.0,
        explore_fallback_semantic_prefer_min_results=5,
        explore_fallback_semantic_prefer_min_score=0.45,
        analytics_skip_uncorroborated_detour=True,
        analytics_budget_keywords=['average'],
        permanently_unavailable_columns=[],
        qie_only_mode=False,
        qie_l0_filter_cache=QieL0FilterCacheConfig(max_entries=256, ttl_seconds=600),
        hybrid_prewarm_enabled=False,
        overlap_preprocess_with_classify=True,
        auction_tiebreak=AuctionTiebreakConfig.from_dict({'enabled': False}),
        analytics_unavailable_hybrid_notice='Analytics unavailable — hybrid results.',
        analytics_unavailable_explore_notice='Analytics unavailable — explore results.',
        analytics_timeout_hybrid_notice='Analytics timeout — hybrid results.',
        analytics_timeout_explore_notice='Analytics timeout — explore results.',
        analytics_detour_skip_notice='Detour skipped — explore results.',
        analytics_connection_unavailable_notice='Analytics connection unavailable.',
        analytics_failure_with_explore_notice='Analytics failed — explore results.',
        analytics_failure_empty_explore_notice='Analytics failed — empty explore.',
    )
    base.update(overrides)
    return base


def _slim_cfg(result_fields):
    return SearchParamsConfig(**_search_params_kwargs(result_fields=result_fields))


def _fake_item(item_id: str = 'fallback.example', payload=None, contributing_sources=None):
    return types.SimpleNamespace(
        item_id=item_id,
        payload=payload,
        contributing_sources=contributing_sources if contributing_sources is not None else ['vector', 'structured'],
    )


class TestSearchParamsConfig:
    def test_from_dict_round_trip_preserves_result_fields_order(self):
        d = _search_params_dict()
        cfg = SearchParamsConfig.from_dict(d)
        assert cfg.result_fields == ['tld', 'sld', 'ends_at']
        assert cfg.top_k_cap == 20
        assert cfg.speculative_analytics_start is True

    def test_post_init_raises_on_empty_result_fields(self):
        with pytest.raises(ConfigurationError, match='result_fields'):
            SearchParamsConfig(**_search_params_kwargs(result_fields=[]))

    def test_post_init_raises_on_empty_string_in_result_fields(self):
        with pytest.raises(ConfigurationError, match='result_fields'):
            SearchParamsConfig(**_search_params_kwargs(result_fields=['tld', '']))

    def test_post_init_raises_on_non_string_in_result_fields(self):
        with pytest.raises(ConfigurationError, match='result_fields'):
            SearchParamsConfig(**_search_params_kwargs(result_fields=['tld', 42]))


class TestSlimResult:
    def test_slim_result_envelope_keys(self):
        item = _fake_item(payload={'domain_name': 'a.com'}, contributing_sources=['bm25'])
        result = app._slim_result(item, rank=2, coherence_score=0.42, search_cfg=_slim_cfg(['tld']))
        assert 'rank' in result
        assert 'domain_name' in result
        assert 'coherence_score' in result
        assert 'matched_by' in result
        assert result['rank'] == 2
        assert result['coherence_score'] == 0.42
        assert result['matched_by'] == ['bm25']

    def test_slim_result_projects_fields_in_order(self):
        item = _fake_item(payload={'tld': 'com', 'price': 99.0, 'sld': 'shop', 'domain_name': 'shop.com'})
        result = app._slim_result(item, rank=1, coherence_score=1.0, search_cfg=_slim_cfg(['tld', 'price', 'sld']))
        assert list(result.keys()) == ['rank', 'domain_name', 'coherence_score', 'matched_by', 'tld', 'price', 'sld']
        assert result['tld'] == 'com'
        assert result['price'] == 99.0
        assert result['sld'] == 'shop'

    def test_slim_result_absent_payload_key_is_none(self):
        item = _fake_item(payload={'tld': 'net'})
        result = app._slim_result(item, rank=1, coherence_score=0.5, search_cfg=_slim_cfg(['semrush_cpc']))
        assert result['semrush_cpc'] is None

    def test_slim_result_domain_name_fallback_to_item_id(self):
        item = _fake_item(item_id='id-only.example', payload={'tld': 'com'})
        result = app._slim_result(item, rank=1, coherence_score=0.9, search_cfg=_slim_cfg(['tld']))
        assert result['domain_name'] == 'id-only.example'

    def test_slim_result_ends_at_formatted(self):
        ts = 1700000000.0
        expected = app._format_ends_at(ts)
        item = _fake_item(payload={'ends_at': ts})
        result = app._slim_result(item, rank=1, coherence_score=0.8, search_cfg=_slim_cfg(['ends_at']))
        assert result['ends_at'] == expected
        assert result['ends_at'] == '2023-11-14T22:13:20Z'

    def test_slim_result_ends_at_none_when_absent(self):
        item = _fake_item(payload={'tld': 'org'})
        result = app._slim_result(item, rank=1, coherence_score=0.6, search_cfg=_slim_cfg(['ends_at']))
        assert result['ends_at'] is None

    def test_slim_result_none_payload_does_not_raise(self):
        item = _fake_item(item_id='none-payload.example', payload=None)
        result = app._slim_result(item, rank=1, coherence_score=0.3, search_cfg=_slim_cfg(['tld']))
        assert result['domain_name'] == 'none-payload.example'
        assert result['tld'] is None

    def test_slim_result_appraised_value_from_payload(self):
        # appraised_value is dual-written into the payload at seed time from
        # govalue_score (see find_payload_aliases in config/base.yaml) — _slim_result
        # just projects the already-aliased key, same as any other result field.
        item = _fake_item(payload={'appraised_value': 4200.0})
        result = app._slim_result(item, rank=1, coherence_score=0.7, search_cfg=_slim_cfg(['appraised_value']))
        assert result['appraised_value'] == 4200.0

    def test_slim_result_appraised_value_none_when_absent(self):
        item = _fake_item(payload={'tld': 'com'})
        result = app._slim_result(item, rank=1, coherence_score=0.7, search_cfg=_slim_cfg(['appraised_value']))
        assert result['appraised_value'] is None
