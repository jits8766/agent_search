"""Shared fixtures for llm_core tests.

Post-refactor (2026-04) the LLMProvider config carries:
    * NO `llm_token_pricing` (sourced from `llm_core/pricing.yaml` instead)
    * NO `models:` field under `llm_api_keys` entries (sourced from runtime
      discovery via `client.models.list()` instead)
    * NO `auto_discover_models` flag (discovery is now mandatory)
The fixture below mirrors that minimal contract.
"""
from typing import Any, Dict, List

import pytest


_DEFAULT_RUNTIME_ADAPTATION = {
    'enabled': False,
    'refresh_interval_s': 300,
    'min_samples': 20,
    'latency_window_s': 3600,
    'quality_window_s': 86400,
    'error_demotion_threshold': 0.25,
    'latency_buckets_ms': [500, 1500, 3000, 6000],
    'max_entries_per_type': 5000,
}

_DEFAULT_WEIGHTS = {'capability': 0.4, 'cost': 0.4, 'latency': 0.2}

_DEFAULT_PROVIDER_TIEBREAK_PRIORITY = [
    'anthropic',
    'openai',
    'google',
    'xai',
    'deepseek',
    'qwen',
    'zhipu',
    'unknown',
]

_DEFAULT_STARTUP_INFERENCE_PROBE = {
    'enabled': False,
    'max_models': 8,
    'prefer_cost_tier_max': 2,
    'timeout_seconds': 5.0,
    'max_tokens': 16,
    'prompt': 'Reply with the single word OK.',
    'always_include_models': [],
}


@pytest.fixture
def base_config() -> Dict[str, Any]:
    """Minimal valid LLMProvider config — every required key present, no defaults relied on."""
    return {
        'llm_models': {'temperature': 0.0, 'max_tokens': 1024, 'max_tokens_validation': 16},
        'llm_api_keys': {
            'anthropic': [{'key_env_var': 'TEST_ANTHROPIC_KEY'}],
            'openai': [{'key_env_var': 'TEST_OPENAI_KEY'}],
        },
        'llm_base_url': '',
        'llm_client_settings': {
            'validation_timeout': 5.0, 'client_timeout': 60.0, 'max_retries': 3,
            'min_api_key_length': 8, 'validation_max_tokens': 16,
        },
        'model_selection_strategy': {
            'enabled': True,
            'weights': dict(_DEFAULT_WEIGHTS),
            'weights_by_task_type': {},
            'runtime_adaptation': dict(_DEFAULT_RUNTIME_ADAPTATION),
            'task_type_preferences': {'healing_analysis': 0.6, 'action_judgment': 0.4},
            'component_task_types': {'llm_judge': 'action_judgment'},
            'dimension_task_types': {},
            'provider_tiebreak_priority': list(_DEFAULT_PROVIDER_TIEBREAK_PRIORITY),
            'startup_inference_probe': dict(_DEFAULT_STARTUP_INFERENCE_PROBE),
            'model_overrides': {},
            'default_component_task_type': 'healing_analysis',
        },
        'model_capability_overrides': {},
        'model_latency_overrides': {},
    }


@pytest.fixture
def discovered_models() -> List[str]:
    """Canonical fake-discovery payload used to seed `available_models` directly in tests.

    Keeps tests deterministic without monkeypatching the SDK clients. Ranker tests
    that need a model set just call `provider._available_models = list(discovered_models)`
    followed by `provider.build_model_registry()`.
    """
    return [
        'claude-3-5-haiku-20241022',
        'claude-3-5-sonnet-20241022',
        'gpt-4o-mini',
        'gpt-4o',
    ]


@pytest.fixture
def weighted_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Config exercising explicit per-axis weights and capability/latency overrides."""
    cfg = {**base_config}
    cfg['model_selection_strategy'] = {
        **base_config['model_selection_strategy'],
        'weights': {'capability': 0.5, 'cost': 0.3, 'latency': 0.2},
    }
    cfg['model_capability_overrides'] = {
        'claude-3-5-sonnet-20241022': 5,
        'gpt-4o': 5,
        'claude-3-5-haiku-20241022': 2,
        'gpt-4o-mini': 1,
    }
    cfg['model_latency_overrides'] = {
        'claude-3-5-haiku-20241022': 1,
        'gpt-4o-mini': 1,
        'claude-3-5-sonnet-20241022': 4,
        'gpt-4o': 3,
    }
    return cfg
