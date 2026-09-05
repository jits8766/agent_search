"""Tests for bundled pricing reference and selection exclusion merge."""
import pytest

from llm_core.pricing import (
    infer_capability_tier,
    infer_model_family,
    infer_model_recency,
    load_selection_exclusions,
    reset_pricing_cache,
)


@pytest.fixture(autouse=True)
def reset_pricing_between_tests():
    reset_pricing_cache()
    yield
    reset_pricing_cache()


def test_selection_exclusions_include_output_cap_violations():
    """Priced SKUs with output USD/M strictly above selection_max_output_usd_per_million must be excluded."""
    excl = load_selection_exclusions()
    assert 'gpt-5.2-pro' in excl
    assert 'gpt-5.5-pro' in excl


def test_selection_exclusions_allow_output_at_cap():
    """Output rate equal to the cap must not auto-exclude (policy is strict >)."""
    excl = load_selection_exclusions()
    assert 'claude-opus-4-5-20251101' not in excl
    assert 'claude-opus-4-6' not in excl


def test_infer_model_family_groups_lineage_not_cross_vendor():
    assert infer_model_family('grok-4.5') == 'grok'
    assert infer_model_family('grok-4.3') == 'grok'
    assert infer_model_family('claude-sonnet-4-6') == 'claude-sonnet'
    assert infer_model_family('claude-opus-4-8') == 'claude-opus'
    assert infer_model_family('glm-5.2') == 'glm'
    assert infer_model_family('deepseek-v4-pro') == 'deepseek'
    assert infer_model_family('qwen3.7-max') == 'qwen'
    assert infer_model_family('grok-4.5') != infer_model_family('claude-sonnet-4-6')


def test_infer_model_recency_same_family_ordering():
    assert infer_model_recency('grok-4.5') > infer_model_recency('grok-4.3')
    assert infer_model_recency('grok-4.3') > infer_model_recency('grok-4.20-0309-reasoning')
    assert infer_model_recency('claude-opus-4-8') > infer_model_recency('claude-opus-4-6')
    assert infer_model_recency('glm-5.2') > infer_model_recency('glm-4.7')


def test_infer_capability_tier_covers_grok_glm_qwen_deepseek():
    assert infer_capability_tier('grok-4.5') == 5
    assert infer_capability_tier('glm-5.2') == 5
    assert infer_capability_tier('qwen3.7-max') == 5
    assert infer_capability_tier('deepseek-v4-pro') == 5
    assert infer_capability_tier('deepseek-v4-flash') == 3
    assert infer_capability_tier('glm-4.7-flash') == 3
    assert infer_capability_tier('grok-4.1-fast') == 3
