"""Tests for the Ranker — composite weights, runtime adaptation, demotion, tiebreaks."""
from typing import Any, Dict, List, Optional

import pytest

from llm_core.ranker import Ranker, RankWeights


_BUCKETS = [500, 1500, 3000, 6000]
_DEFAULT_WEIGHTS = RankWeights(capability=0.4, cost=0.4, latency=0.2)
_DEFAULT_PROVIDER_TIEBREAK = [
    'anthropic',
    'openai',
    'google',
    'xai',
    'deepseek',
    'qwen',
    'zhipu',
    'unknown',
]


def _make_ranker(
    error_demotion_threshold: float = 0.25,
    runtime_min_samples: int = 20,
    provider_tiebreak_priority: Optional[List[str]] = None,
) -> Ranker:
    return Ranker(
        latency_buckets_ms=list(_BUCKETS),
        error_demotion_threshold=error_demotion_threshold,
        runtime_min_samples=runtime_min_samples,
        provider_tiebreak_priority=list(provider_tiebreak_priority or _DEFAULT_PROVIDER_TIEBREAK),
    )


def _registry() -> Dict[str, Dict[str, Any]]:
    return {
        'fast-cheap': {'capability_tier': 2, 'cost_tier': 1, 'latency_tier': 1, 'total_cost': 1.0},
        'mid': {'capability_tier': 3, 'cost_tier': 3, 'latency_tier': 3, 'total_cost': 5.0},
        'smart-slow': {'capability_tier': 5, 'cost_tier': 5, 'latency_tier': 5, 'total_cost': 20.0},
    }


def test_weights_normalize_from_dict() -> None:
    w = RankWeights.from_dict({'capability': 2, 'cost': 2, 'latency': 1})
    assert abs(w.capability + w.cost + w.latency - 1.0) < 1e-9
    assert abs(w.latency - 0.2) < 1e-9


def test_weights_from_dict_requires_all_keys() -> None:
    with pytest.raises(ValueError, match='capability'):
        RankWeights.from_dict({'cost': 0.5, 'latency': 0.5})


def test_weights_from_dict_rejects_zero_sum() -> None:
    with pytest.raises(ValueError, match='> 0'):
        RankWeights.from_dict({'capability': 0, 'cost': 0, 'latency': 0})


def test_ranker_requires_four_buckets() -> None:
    with pytest.raises(ValueError, match='4 thresholds'):
        Ranker(
            latency_buckets_ms=[100, 200],
            error_demotion_threshold=0.25,
            runtime_min_samples=20,
            provider_tiebreak_priority=list(_DEFAULT_PROVIDER_TIEBREAK),
        )


def test_ranker_requires_provider_tiebreak_priority() -> None:
    with pytest.raises(ValueError, match='provider_tiebreak_priority'):
        Ranker(
            latency_buckets_ms=list(_BUCKETS),
            error_demotion_threshold=0.25,
            runtime_min_samples=20,
            provider_tiebreak_priority=[],
        )


def test_rank_capability_priority_picks_smartest() -> None:
    r = _make_ranker()
    out = r.rank(list(_registry()), _registry(), RankWeights(capability=1.0, cost=0.0, latency=0.0))
    assert out[0][0] == 'smart-slow'


def test_rank_cost_priority_picks_cheapest() -> None:
    r = _make_ranker()
    out = r.rank(list(_registry()), _registry(), RankWeights(capability=0.0, cost=1.0, latency=0.0))
    assert out[0][0] == 'fast-cheap'


def test_rank_latency_priority_picks_fastest() -> None:
    r = _make_ranker()
    out = r.rank(list(_registry()), _registry(), RankWeights(capability=0.0, cost=0.0, latency=1.0))
    assert out[0][0] == 'fast-cheap'


def test_runtime_latency_demotes_slow_observed_model() -> None:
    """Mid-tier static, but observed p50 is awful — should rank below faster observation."""
    r = _make_ranker(runtime_min_samples=10)
    reg = _registry()
    stats = {
        'mid': {'p50_latency_ms': 8000, 'error_rate': 0.0, 'sample_count': 50, 'quality_score': None},
        'fast-cheap': {'p50_latency_ms': 200, 'error_rate': 0.0, 'sample_count': 50, 'quality_score': None},
    }
    weights = RankWeights(capability=0.0, cost=0.0, latency=1.0)
    out = r.rank(list(reg), reg, weights, stats)
    assert out[0][0] == 'fast-cheap'
    mid_idx = [m for m, _ in out].index('mid')
    smart_idx = [m for m, _ in out].index('smart-slow')
    assert mid_idx > 0 and smart_idx > 0


def test_runtime_min_samples_gate() -> None:
    """Insufficient samples must NOT override static tier."""
    r = _make_ranker(runtime_min_samples=100)
    reg = _registry()
    stats = {'mid': {'p50_latency_ms': 50, 'error_rate': 0.0, 'sample_count': 5, 'quality_score': None}}
    out = r.rank(list(reg), reg, RankWeights(capability=0.0, cost=0.0, latency=1.0), stats)
    assert out[0][0] == 'fast-cheap'


def test_high_error_rate_demotes_to_end() -> None:
    r = _make_ranker(error_demotion_threshold=0.2, runtime_min_samples=10)
    reg = _registry()
    stats = {'fast-cheap': {'p50_latency_ms': 100, 'error_rate': 0.9, 'sample_count': 50, 'quality_score': None}}
    out = r.rank(list(reg), reg, RankWeights(capability=0.0, cost=1.0, latency=0.0), stats)
    assert out[-1][0] == 'fast-cheap'


def test_total_cost_tiebreak_picks_cheaper() -> None:
    """Equal scores: cheaper (lower total_cost) wins."""
    reg = {
        'a-cheap': {'capability_tier': 3, 'cost_tier': 3, 'latency_tier': 3, 'total_cost': 1.0},
        'b-pricey': {'capability_tier': 3, 'cost_tier': 3, 'latency_tier': 3, 'total_cost': 9.0},
    }
    r = _make_ranker()
    out = r.rank(list(reg), reg, _DEFAULT_WEIGHTS)
    assert out[0][0] == 'a-cheap'


def test_same_family_recency_tiebreak_prefers_latest() -> None:
    """Equal score + equal price: newer SKU in the same family wins."""
    reg = {
        'grok-4.3': {'capability_tier': 5, 'cost_tier': 3, 'latency_tier': 3, 'total_cost': 3.75},
        'grok-4.5': {'capability_tier': 5, 'cost_tier': 3, 'latency_tier': 3, 'total_cost': 3.75},
    }
    r = _make_ranker()
    # Older id listed first — recency must still put 4.5 ahead.
    out = r.rank(['grok-4.3', 'grok-4.5'], reg, _DEFAULT_WEIGHTS)
    assert [m for m, _ in out] == ['grok-4.5', 'grok-4.3']


def test_cross_family_recency_does_not_displace_other_vendor() -> None:
    """Latest grok must not jump Claude on version alone; provider priority picks Claude."""
    reg = {
        'grok-4.5': {'capability_tier': 5, 'cost_tier': 3, 'latency_tier': 3, 'total_cost': 8.0},
        'claude-sonnet-4-6': {'capability_tier': 5, 'cost_tier': 3, 'latency_tier': 3, 'total_cost': 8.0},
    }
    r = _make_ranker()
    out = r.rank(['claude-sonnet-4-6', 'grok-4.5'], reg, _DEFAULT_WEIGHTS)
    assert [m for m, _ in out] == ['claude-sonnet-4-6', 'grok-4.5']
    # Grok discovered first — anthropic still wins on provider tiebreak (not discovery order).
    out_rev = r.rank(['grok-4.5', 'claude-sonnet-4-6'], reg, _DEFAULT_WEIGHTS)
    assert [m for m, _ in out_rev] == ['claude-sonnet-4-6', 'grok-4.5']


def test_cross_provider_tiebreak_order_anthropic_openai_google() -> None:
    """Equal score+cost across vendors follows provider_tiebreak_priority."""
    reg = {
        'gemini-2.5-flash': {'capability_tier': 3, 'cost_tier': 2, 'latency_tier': 2, 'total_cost': 1.0},
        'gpt-4o-mini': {'capability_tier': 3, 'cost_tier': 2, 'latency_tier': 2, 'total_cost': 1.0},
        'claude-3-haiku-20240307': {'capability_tier': 3, 'cost_tier': 2, 'latency_tier': 2, 'total_cost': 1.0},
    }
    r = _make_ranker()
    out = r.rank(['gemini-2.5-flash', 'gpt-4o-mini', 'claude-3-haiku-20240307'], reg, _DEFAULT_WEIGHTS)
    assert [m for m, _ in out] == ['claude-3-haiku-20240307', 'gpt-4o-mini', 'gemini-2.5-flash']


def test_cheaper_openai_still_beats_anthropic() -> None:
    """Provider tiebreak must not override cheaper total_cost."""
    reg = {
        'gpt-4o-mini': {'capability_tier': 3, 'cost_tier': 2, 'latency_tier': 2, 'total_cost': 0.5},
        'claude-3-haiku-20240307': {'capability_tier': 3, 'cost_tier': 2, 'latency_tier': 2, 'total_cost': 1.5},
    }
    r = _make_ranker()
    out = r.rank(['claude-3-haiku-20240307', 'gpt-4o-mini'], reg, _DEFAULT_WEIGHTS)
    assert out[0][0] == 'gpt-4o-mini'


def test_empty_models() -> None:
    assert _make_ranker().rank([], {}, _DEFAULT_WEIGHTS) == []


def test_unregistered_model_uses_neutral_tiers() -> None:
    """Models missing from registry must be scored (logged) rather than crash."""
    r = _make_ranker()
    out = r.rank(['ghost-model', 'fast-cheap'], _registry(), RankWeights(capability=0.0, cost=1.0, latency=0.0))
    names = [m for m, _ in out]
    assert 'ghost-model' in names
    assert names[0] == 'fast-cheap'


def test_same_cost_measured_latency_prefers_fastest() -> None:
    """Equal score+cost: faster startup/runtime p50 wins before provider/recency."""
    reg = {
        'gpt-4.1-nano': {'capability_tier': 2, 'cost_tier': 1, 'latency_tier': 3, 'total_cost': 0.1},
        'gpt-5-nano': {'capability_tier': 2, 'cost_tier': 1, 'latency_tier': 3, 'total_cost': 0.1},
    }
    stats = {
        'gpt-4.1-nano': {'p50_latency_ms': 1100, 'error_rate': 0.0, 'sample_count': 20, 'quality_score': None},
        'gpt-5-nano': {'p50_latency_ms': 8000, 'error_rate': 0.0, 'sample_count': 20, 'quality_score': None},
    }
    # latency weight 0 so only the measured-latency *tiebreak* decides (same score+cost).
    r = _make_ranker(runtime_min_samples=20)
    out = r.rank(['gpt-5-nano', 'gpt-4.1-nano'], reg, RankWeights(capability=0.0, cost=1.0, latency=0.0), stats)
    assert [m for m, _ in out] == ['gpt-4.1-nano', 'gpt-5-nano']


def test_same_cost_same_latency_prefers_latest_in_family() -> None:
    """Equal score+cost+measured latency: newer SKU in the same family wins."""
    reg = {
        'grok-4.3': {'capability_tier': 5, 'cost_tier': 3, 'latency_tier': 3, 'total_cost': 3.75},
        'grok-4.5': {'capability_tier': 5, 'cost_tier': 3, 'latency_tier': 3, 'total_cost': 3.75},
    }
    stats = {
        'grok-4.3': {'p50_latency_ms': 500, 'error_rate': 0.0, 'sample_count': 20, 'quality_score': None},
        'grok-4.5': {'p50_latency_ms': 500, 'error_rate': 0.0, 'sample_count': 20, 'quality_score': None},
    }
    r = _make_ranker(runtime_min_samples=20)
    out = r.rank(['grok-4.3', 'grok-4.5'], reg, RankWeights(capability=0.0, cost=1.0, latency=0.0), stats)
    assert [m for m, _ in out] == ['grok-4.5', 'grok-4.3']
