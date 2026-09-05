"""Unit tests for complement rank leanings (analytics + guidance)."""
import json

from semantic_search.config.models import ComplementRankLeaningsConfig
from semantic_search.retrieval.complement_rank_leanings import (
    apply_complement_rank_leanings,
    extract_analytics_cohort_scores,
    extract_guidance_tld_scores,
)


def _cfg(**overrides) -> ComplementRankLeaningsConfig:
    base = dict(
        enabled=True,
        max_bonus=0.08,
        weight_guidance=0.5,
        weight_analytics=0.5,
        query_types=['analytics', 'guidance'],
    )
    base.update(overrides)
    return ComplementRankLeaningsConfig(**base)


def _row(tld: str, coherence: float, rank: int = 1) -> dict:
    return {
        'rank': rank,
        'tld': tld,
        'coherence_score': coherence,
        'domain_name': f'example.{tld}',
    }


def test_extract_guidance_tld_scores_minmax():
    body = json.dumps([
        {'tld': 'com', 'total_auctions': 100, 'total_bids': 50},
        {'tld': 'net', 'total_auctions': 10, 'total_bids': 5},
    ])
    scores = extract_guidance_tld_scores(body)
    assert scores['com'] == 1.0
    assert scores['net'] == 0.0


def test_extract_guidance_malformed_empty():
    assert extract_guidance_tld_scores('not-json{') == {}
    assert extract_guidance_tld_scores(None) == {}
    assert extract_guidance_tld_scores('') == {}


def test_extract_analytics_periods_latest_window():
    data = {
        'periods': {
            'last_30d': {
                'rows': [
                    {'tld': 'com', 'avg_price': 200.0, 'count': 40},
                    {'tld': 'io', 'avg_price': 50.0, 'count': 5},
                ],
            },
            'last_7d': {
                'rows': [
                    {'tld': 'net', 'avg_price': 999.0, 'count': 99},
                ],
            },
        },
    }
    scores = extract_analytics_cohort_scores(data)
    assert 'com' in scores and 'io' in scores
    assert 'net' not in scores  # last_30d preferred over last_7d
    assert scores['com'] > scores['io']


def test_guidance_higher_activity_tld_rises_among_near_equal():
    ranked = [_row('net', 0.90, 1), _row('com', 0.89, 2)]
    body = [
        {'tld': 'com', 'total_auctions': 100, 'total_bids': 80},
        {'tld': 'net', 'total_auctions': 5, 'total_bids': 1},
    ]
    out = apply_complement_rank_leanings(
        ranked,
        query_type='guidance',
        cfg=_cfg(max_bonus=0.08, weight_guidance=1.0, weight_analytics=0.0),
        guidance_body=body,
    )
    assert out[0]['tld'] == 'com'
    assert out[0]['rank'] == 1
    assert out[0]['coherence_score'] == 0.89


def test_analytics_moves_near_tied_preserves_large_gap():
    near = [_row('net', 0.80, 1), _row('com', 0.79, 2)]
    analytics = {
        'rows': [
            {'tld': 'com', 'avg_price': 300.0, 'count': 50},
            {'tld': 'net', 'avg_price': 10.0, 'count': 2},
        ],
    }
    out_near = apply_complement_rank_leanings(
        near,
        query_type='analytics',
        cfg=_cfg(max_bonus=0.08, weight_guidance=0.0, weight_analytics=1.0),
        analytics_data=analytics,
    )
    assert out_near[0]['tld'] == 'com'

    wide = [_row('net', 0.95, 1), _row('com', 0.70, 2)]
    out_wide = apply_complement_rank_leanings(
        wide,
        query_type='analytics',
        cfg=_cfg(max_bonus=0.08, weight_guidance=0.0, weight_analytics=1.0),
        analytics_data=analytics,
    )
    assert out_wide[0]['tld'] == 'net'
    assert out_wide[0]['coherence_score'] == 0.95


def test_empty_payload_identity():
    ranked = [_row('com', 0.9, 1), _row('net', 0.8, 2)]
    out = apply_complement_rank_leanings(
        ranked,
        query_type='guidance',
        cfg=_cfg(),
        guidance_body=[],
    )
    assert [r['tld'] for r in out] == ['com', 'net']


def test_disabled_config_identity():
    ranked = [_row('net', 0.90, 1), _row('com', 0.89, 2)]
    body = [{'tld': 'com', 'total_auctions': 100}, {'tld': 'net', 'total_auctions': 1}]
    out = apply_complement_rank_leanings(
        ranked,
        query_type='guidance',
        cfg=_cfg(enabled=False),
        guidance_body=body,
    )
    assert [r['tld'] for r in out] == ['net', 'com']


def test_query_type_not_configured_identity():
    ranked = [_row('net', 0.90, 1), _row('com', 0.89, 2)]
    body = [{'tld': 'com', 'total_auctions': 100}, {'tld': 'net', 'total_auctions': 1}]
    out = apply_complement_rank_leanings(
        ranked,
        query_type='hybrid',
        cfg=_cfg(query_types=['analytics', 'guidance']),
        guidance_body=body,
    )
    assert [r['tld'] for r in out] == ['net', 'com']
