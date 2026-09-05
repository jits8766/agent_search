"""Tests for explicit composite weights and capability/latency overrides at provider level."""
from llm_core import LLMProvider


_FAKE_DISCOVERY = ['claude-3-5-haiku-20241022', 'claude-3-5-sonnet-20241022', 'gpt-4o-mini', 'gpt-4o']


def _seed(provider, models=_FAKE_DISCOVERY):
    """Inject a fake-discovery model set; mirrors what `validate_api_keys` would produce in prod."""
    provider._available_models = list(models)
    provider._registry.update_models(provider._available_models)


def test_explicit_weights_drive_ranking(weighted_config):
    """Provider must use the configured weights triple end-to-end (no legacy cost_weight knob)."""
    provider = LLMProvider(weighted_config)
    _seed(provider)
    provider.build_model_registry()
    ranked = provider.select_models_for_task('healing_analysis')
    assert set(ranked) == set(_FAKE_DISCOVERY)


def test_capability_override_promotes_model(weighted_config):
    """sonnet/gpt-4o get capability=5; should appear in the top half."""
    provider = LLMProvider(weighted_config)
    _seed(provider)
    provider.build_model_registry()
    ranked = provider.select_models_for_task('healing_analysis')
    top_two = set(ranked[:2])
    assert top_two & {'claude-3-5-sonnet-20241022', 'gpt-4o'}


def test_latency_override_picks_fastest_when_lat_dominant(base_config):
    cfg = {**base_config}
    cfg['model_selection_strategy'] = {
        **base_config['model_selection_strategy'],
        'weights': {'capability': 0.0, 'cost': 0.0, 'latency': 1.0},
    }
    cfg['model_latency_overrides'] = {
        'claude-3-5-haiku-20241022': 1,
        'gpt-4o-mini': 2,
        'gpt-4o': 4,
        'claude-3-5-sonnet-20241022': 5,
    }
    provider = LLMProvider(cfg)
    _seed(provider)
    provider.build_model_registry()
    ranked = provider.select_models_for_task('healing_analysis')
    assert ranked[0] == 'claude-3-5-haiku-20241022'
    assert ranked[-1] == 'claude-3-5-sonnet-20241022'
