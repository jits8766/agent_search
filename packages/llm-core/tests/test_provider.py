"""Tests for LLMProvider — discovery-driven contract, registry, ranking, validation, clients, fallbacks."""
import pytest

from llm_core import LLMProvider, _detect_provider, _is_chat_model
from llm_core.exceptions import ConfigurationError
from llm_core.provider import _extract_error_message, _is_output_length_only_failure


def test_extract_error_message_returns_raw_when_no_message_key():
    assert _extract_error_message('connection refused') == 'connection refused'
    assert _extract_error_message('') == ''


def test_extract_error_message_picks_deepest_message():
    """Gateway exceptions wrap upstream errors 2-3 layers deep; deepest message is the actual reason."""
    raw = (
        "Error code: 400 - {'error': {'message': 'litellm.BadRequestError: OpenAIException - "
        "{\"error\": {\"message\": \"Model gpt-5.2-codex requires the responses endpoint\", "
        "\"type\": \"invalid_request_error\"}}', 'type': 'BadRequestError'}}"
    )
    assert _extract_error_message(raw) == 'Model gpt-5.2-codex requires the responses endpoint'


def test_extract_error_message_handles_single_layer():
    assert _extract_error_message('{"message": "rate_limited"}') == 'rate_limited'


def test_is_output_length_only_failure_detects_token_budget_rejections():
    """Reasoning-model probe responses must be classified as reachable, not dropped."""
    assert _is_output_length_only_failure(
        'Could not finish the message because max_tokens or model output limit was reached.'
    )
    assert _is_output_length_only_failure(
        "Invalid 'max_output_tokens': integer below minimum value. Expected a value >= 16, but got 5 instead."
    )
    assert _is_output_length_only_failure('maximum context length exceeded')


def test_is_output_length_only_failure_does_not_swallow_real_failures():
    """Auth, routing, and not-found errors must still mark the model as unreachable."""
    assert not _is_output_length_only_failure('Invalid API key provided')
    assert not _is_output_length_only_failure('The model `gpt-9` does not exist')
    assert not _is_output_length_only_failure('You do not have access to this model')
    assert not _is_output_length_only_failure('')
    assert not _is_output_length_only_failure('Use the responses endpoint for this model')


def test_provider_init_requires_top_level_keys(base_config):
    """Pricing is no longer a service-config concern; only the surviving top-level keys are required."""
    for drop in [
        'llm_api_keys', 'llm_models', 'llm_client_settings',
        'llm_base_url', 'model_selection_strategy', 'model_capability_overrides', 'model_latency_overrides',
    ]:
        bad = {k: v for k, v in base_config.items() if k != drop}
        with pytest.raises(ConfigurationError, match=drop):
            LLMProvider(bad)


def test_provider_init_no_longer_requires_llm_token_pricing(base_config):
    """`llm_token_pricing` is sourced from llm_core/pricing.yaml; service config must NOT need it."""
    cfg = {k: v for k, v in base_config.items() if k != 'llm_token_pricing'}
    LLMProvider(cfg)


def test_provider_init_requires_strategy_subkeys(base_config):
    for drop in ['enabled', 'weights', 'weights_by_task_type', 'runtime_adaptation', 'task_type_preferences', 'component_task_types', 'dimension_task_types', 'provider_tiebreak_priority', 'startup_inference_probe']:
        bad = {**base_config}
        bad['model_selection_strategy'] = {k: v for k, v in base_config['model_selection_strategy'].items() if k != drop}
        with pytest.raises(ConfigurationError, match=drop):
            LLMProvider(bad)


def test_provider_init_requires_runtime_adaptation_keys(base_config):
    for drop in ['enabled', 'refresh_interval_s', 'min_samples', 'latency_window_s', 'quality_window_s', 'error_demotion_threshold', 'latency_buckets_ms', 'max_entries_per_type']:
        bad = {**base_config}
        bad['model_selection_strategy'] = {**base_config['model_selection_strategy']}
        bad['model_selection_strategy']['runtime_adaptation'] = {k: v for k, v in base_config['model_selection_strategy']['runtime_adaptation'].items() if k != drop}
        with pytest.raises(ConfigurationError, match=drop):
            LLMProvider(bad)


def test_provider_starts_with_no_available_models(base_config):
    """No declared list anymore — provider has zero models until validate_api_keys discovers them."""
    provider = LLMProvider(base_config)
    assert provider.available_models == []


def _seed_discovery(provider, models):
    """Inject a fake-discovery model set in the same shape as validate_api_keys would produce."""
    provider._available_models = list(models)
    provider._registry.update_models(provider._available_models)


def test_build_model_registry_assigns_tiers(base_config, discovered_models):
    provider = LLMProvider(base_config)
    _seed_discovery(provider, discovered_models)
    provider.build_model_registry()
    assert len(provider.model_registry) == len(discovered_models)
    for _model, meta in provider.model_registry.items():
        assert 1 <= meta['capability_tier'] <= 5
        assert 1 <= meta['cost_tier'] <= 5
        assert 1 <= meta['latency_tier'] <= 5


def test_build_model_registry_unpriced_model_gets_neutral_tiers(base_config):
    """A model the bundled pricing.yaml has never heard of must still be registered + rankable."""
    provider = LLMProvider(base_config)
    _seed_discovery(provider, ['gpt-future-unreleased', 'gpt-4o'])
    provider.build_model_registry()
    assert 'gpt-future-unreleased' in provider.model_registry
    meta = provider.model_registry['gpt-future-unreleased']
    assert meta['cost_tier'] == 3
    assert meta['total_cost'] == 0.0
    ranked = provider.select_models_for_task('healing_analysis')
    assert 'gpt-future-unreleased' in ranked


def test_capability_inferred_from_name(base_config):
    """Without overrides, capability tier is derived from the model name heuristic."""
    provider = LLMProvider(base_config)
    _seed_discovery(provider, [
        'claude-3-5-haiku-20241022',
        'claude-opus-4-20250514',
        'gpt-5.1',
        'gpt-4o',
        'grok-4.5',
        'glm-5.2',
        'deepseek-v4-flash',
        'qwen3.7-max',
    ])
    provider.build_model_registry()
    reg = provider.model_registry
    assert reg['claude-3-5-haiku-20241022']['capability_tier'] == 2
    assert reg['claude-opus-4-20250514']['capability_tier'] == 5
    assert reg['gpt-5.1']['capability_tier'] == 5
    assert reg['gpt-4o']['capability_tier'] == 4
    assert reg['grok-4.5']['capability_tier'] == 5
    assert reg['glm-5.2']['capability_tier'] == 5
    assert reg['qwen3.7-max']['capability_tier'] == 5
    assert reg['deepseek-v4-flash']['capability_tier'] == 3
    assert reg['grok-4.5']['provider'] == 'xai'
    assert reg['glm-5.2']['provider'] == 'zhipu'


def test_select_models_for_task_returns_ranked_list(base_config, discovered_models):
    provider = LLMProvider(base_config)
    _seed_discovery(provider, discovered_models)
    provider.build_model_registry()
    ranked = provider.select_models_for_task('healing_analysis')
    assert set(ranked) == set(discovered_models)


def test_get_ranked_models_returns_scores(base_config, discovered_models):
    provider = LLMProvider(base_config)
    _seed_discovery(provider, discovered_models)
    provider.build_model_registry()
    scored = provider.get_ranked_models_for_task('healing_analysis')
    assert all(isinstance(t, tuple) and len(t) == 2 for t in scored)
    scores = [s for _, s in scored]
    assert scores == sorted(scores, reverse=True)


def test_rank_candidates_subset(base_config, discovered_models):
    provider = LLMProvider(base_config)
    _seed_discovery(provider, discovered_models)
    provider.build_model_registry()
    out = provider.rank_candidates(['gpt-4o-mini', 'claude-3-5-sonnet-20241022'], 'healing_analysis')
    assert sorted(out) == sorted(['gpt-4o-mini', 'claude-3-5-sonnet-20241022'])


@pytest.mark.asyncio
async def test_validate_api_keys_drops_selection_exclusions(base_config, monkeypatch):
    """Models in pricing.yaml selection_exclusions must not appear in available_models after discovery."""
    monkeypatch.delenv('TEST_ANTHROPIC_KEY', raising=False)
    monkeypatch.setenv('TEST_OPENAI_KEY', 'sk-' + 'x' * 24)

    async def discover(provider, api_key, base_url, timeout):
        if provider == 'openai':
            return ['gpt-5.2-pro', 'gpt-4o-mini']
        return []

    monkeypatch.setattr(LLMProvider, '_discover_provider_models', staticmethod(discover))

    provider = LLMProvider(base_config)
    validated = await provider.validate_api_keys()
    assert 'openai' in validated
    assert provider.available_models == ['gpt-4o-mini']


def test_rank_candidates_filters_selection_exclusions(base_config):
    provider = LLMProvider(base_config)
    provider._available_models = ['gpt-4o-mini']
    provider.build_model_registry()
    out = provider.rank_candidates(['gpt-5.2-pro', 'gpt-4o-mini'], 'healing_analysis')
    assert out == ['gpt-4o-mini']


def test_build_model_registry_drops_selection_exclusions(base_config):
    provider = LLMProvider(base_config)
    _seed_discovery(provider, ['gpt-5.2-pro', 'gpt-4o'])
    provider.build_model_registry()
    assert 'gpt-5.2-pro' not in provider.model_registry
    assert 'gpt-4o' in provider.model_registry


def test_build_task_fallbacks(base_config, discovered_models):
    provider = LLMProvider(base_config)
    _seed_discovery(provider, discovered_models)
    provider.build_model_registry()
    provider.build_task_fallbacks()
    assert 'healing_analysis' in provider.model_fallbacks
    assert 'action_judgment' in provider.model_fallbacks
    assert 'llm_judge' in provider.model_fallbacks
    assert len(provider.model_fallbacks['healing_analysis']) == len(discovered_models)


@pytest.mark.asyncio
async def test_validate_api_keys_no_env(base_config, monkeypatch):
    """Without env vars, no provider is validated and there are no available models."""
    monkeypatch.delenv('TEST_ANTHROPIC_KEY', raising=False)
    monkeypatch.delenv('TEST_OPENAI_KEY', raising=False)
    provider = LLMProvider(base_config)
    validated = await provider.validate_api_keys()
    assert validated == {}
    assert provider.available_models == []


@pytest.mark.asyncio
async def test_validate_api_keys_uses_discovery(base_config, monkeypatch):
    """Discovery is the source of truth — whatever models.list() returns becomes available_models."""
    monkeypatch.delenv('TEST_OPENAI_KEY', raising=False)
    monkeypatch.setenv('TEST_ANTHROPIC_KEY', 'sk-ant-' + 'x' * 16)

    async def fake_discover(provider, api_key, base_url, timeout):
        if provider == 'anthropic':
            return ['claude-haiku-4-5-20251001', 'claude-opus-4-5-20251101']
        return []

    monkeypatch.setattr(LLMProvider, '_discover_provider_models', staticmethod(fake_discover))

    provider = LLMProvider(base_config)
    validated = await provider.validate_api_keys()
    assert 'anthropic' in validated
    assert provider.available_models == ['claude-haiku-4-5-20251001', 'claude-opus-4-5-20251101']


@pytest.mark.asyncio
async def test_validate_api_keys_discovery_empty_skips_provider(base_config, monkeypatch):
    """If discovery returns nothing (auth failure, empty list, or gateway error), do not record the provider as validated."""
    monkeypatch.delenv('TEST_OPENAI_KEY', raising=False)
    monkeypatch.setenv('TEST_ANTHROPIC_KEY', 'sk-ant-' + 'x' * 16)

    async def empty_discover(provider, api_key, base_url, timeout):
        return []

    monkeypatch.setattr(LLMProvider, '_discover_provider_models', staticmethod(empty_discover))

    provider = LLMProvider(base_config)
    validated = await provider.validate_api_keys()
    assert 'anthropic' not in validated
    assert validated == {}
    assert provider.available_models == []


@pytest.mark.asyncio
async def test_live_check_drops_unreachable_discovered_models(base_config, monkeypatch):
    """live_check=True must remove discovered models the gateway lists but cannot route, without dropping the whole provider."""
    monkeypatch.delenv('TEST_ANTHROPIC_KEY', raising=False)
    monkeypatch.setenv('TEST_OPENAI_KEY', 'sk-' + 'x' * 24)

    async def discover(provider, api_key, base_url, timeout):
        if provider == 'openai':
            return ['gpt-5.1', 'gpt-4o-mini']
        return []

    async def probe(provider, api_key, model, base_url, timeout, max_tokens):
        if model == 'gpt-4o-mini':
            return False, 'BadRequestError: model_not_supported'
        return True, None

    monkeypatch.setattr(LLMProvider, '_discover_provider_models', staticmethod(discover))
    monkeypatch.setattr(LLMProvider, '_live_check_model', staticmethod(probe))

    provider = LLMProvider(base_config)
    validated = await provider.validate_api_keys(live_check=True)
    assert 'openai' in validated
    assert provider.available_models == ['gpt-5.1']


@pytest.mark.asyncio
async def test_live_check_drops_provider_when_no_model_accessible(base_config, monkeypatch):
    """If every probe fails for a provider, that provider key is dropped from validated map."""
    monkeypatch.setenv('TEST_OPENAI_KEY', 'sk-' + 'x' * 24)
    monkeypatch.setenv('TEST_ANTHROPIC_KEY', 'sk-ant-' + 'x' * 24)

    async def discover(provider, api_key, base_url, timeout):
        if provider == 'openai':
            return ['gpt-5.1']
        if provider == 'anthropic':
            return ['claude-haiku-4-5-20251001']
        return []

    async def fail_openai(provider, api_key, model, base_url, timeout, max_tokens):
        if provider == 'openai':
            return False, 'BadRequestError: gateway_blocked'
        return True, None

    monkeypatch.setattr(LLMProvider, '_discover_provider_models', staticmethod(discover))
    monkeypatch.setattr(LLMProvider, '_live_check_model', staticmethod(fail_openai))

    provider = LLMProvider(base_config)
    validated = await provider.validate_api_keys(live_check=True)
    assert 'openai' not in validated
    assert 'anthropic' in validated
    assert provider.available_models == ['claude-haiku-4-5-20251001']


@pytest.mark.asyncio
async def test_initialize_cached_clients_no_keys(base_config, monkeypatch):
    monkeypatch.delenv('TEST_ANTHROPIC_KEY', raising=False)
    monkeypatch.delenv('TEST_OPENAI_KEY', raising=False)
    provider = LLMProvider(base_config)
    await provider.validate_api_keys()
    provider.initialize_cached_clients()
    assert provider.cached_clients == {}


def test_get_client_for_model_not_found(base_config):
    provider = LLMProvider(base_config)
    provider.build_model_registry()
    with pytest.raises(ConfigurationError, match='No LLM client available'):
        provider.get_client_for_model('gpt-4o-mini')


def test_get_default_client_none(base_config):
    provider = LLMProvider(base_config)
    assert provider.get_default_client() is None


def test_get_fallback_chain_known_task(base_config, discovered_models):
    provider = LLMProvider(base_config)
    _seed_discovery(provider, discovered_models)
    provider.build_model_registry()
    provider.build_task_fallbacks()
    chain = provider.get_fallback_chain('healing_analysis')
    assert isinstance(chain, list)
    assert set(chain) == set(discovered_models)


def test_get_fallback_chain_unknown_task(base_config, discovered_models):
    provider = LLMProvider(base_config)
    _seed_discovery(provider, discovered_models)
    chain = provider.get_fallback_chain('unknown_task_xyz')
    assert chain == provider.available_models


def test_get_summary(base_config, discovered_models):
    provider = LLMProvider(base_config)
    _seed_discovery(provider, discovered_models)
    provider.build_model_registry()
    summary = provider.get_summary()
    assert summary['available_models'] == len(discovered_models)
    assert summary['registry_size'] == len(discovered_models)
    assert summary['runtime_adaptation_enabled'] is False
    assert 'runtime_stats_models' in summary


def test_detect_provider_function():
    assert _detect_provider('claude-3-opus') == 'anthropic'
    assert _detect_provider('gpt-4o') == 'openai'
    assert _detect_provider('o1-mini') == 'openai'
    assert _detect_provider('gemini-pro') == 'google'
    assert _detect_provider('unknown-model') == 'unknown'


def test_is_chat_model_function():
    assert _is_chat_model('gpt-4o') is True
    assert _is_chat_model('text-embedding-3-small') is False
    assert _is_chat_model('dall-e-3') is False


def test_dimension_task_types_supported(base_config, discovered_models):
    """dimension_task_types must be honored in build_task_fallbacks."""
    cfg = {**base_config}
    cfg['model_selection_strategy'] = {
        **base_config['model_selection_strategy'],
        'dimension_task_types': {'jtbd_completion': 'healing_analysis'},
    }
    provider = LLMProvider(cfg)
    _seed_discovery(provider, discovered_models)
    provider.build_model_registry()
    provider.build_task_fallbacks()
    assert 'jtbd_completion' in provider.model_fallbacks
    assert len(provider.model_fallbacks['jtbd_completion']) == len(discovered_models)


def test_total_cost_tiebreaker_for_equal_scores(base_config):
    """When scores tie, prefer cheaper model. Pricing comes from the bundled pricing.yaml."""
    cfg = {**base_config}
    cfg['model_selection_strategy'] = {
        **base_config['model_selection_strategy'],
        'weights': {'capability': 0.0, 'cost': 1.0, 'latency': 0.0},
    }
    provider = LLMProvider(cfg)
    _seed_discovery(provider, ['gpt-4o-mini', 'gpt-4o', 'gpt-4-turbo'])
    provider.build_model_registry()
    ranked = provider.select_models_for_task('healing_analysis')
    assert ranked[0] == 'gpt-4o-mini'


def test_select_diverse_models_for_task_provider_diversity(base_config, discovered_models):
    """select_diverse_models_for_task must return one model per provider when require_provider_diversity=True."""
    provider = LLMProvider(base_config)
    _seed_discovery(provider, discovered_models)
    provider.build_model_registry()
    picked = provider.select_diverse_models_for_task('healing_analysis', count=2, require_provider_diversity=True)
    assert len(picked) == 2
    providers = {_detect_provider(m) for m in picked}
    assert providers == {'anthropic', 'openai'}


def test_select_diverse_models_for_task_no_diversity(base_config, discovered_models):
    """When require_provider_diversity=False, picks top-N from ranked list as-is."""
    provider = LLMProvider(base_config)
    _seed_discovery(provider, discovered_models)
    provider.build_model_registry()
    picked = provider.select_diverse_models_for_task('healing_analysis', count=3, require_provider_diversity=False)
    assert len(picked) == 3
    assert picked == provider.select_models_for_task('healing_analysis')[:3]


@pytest.mark.asyncio
async def test_probe_startup_inference_disabled_skips(base_config, discovered_models):
    provider = LLMProvider(base_config)
    _seed_discovery(provider, discovered_models)
    provider.build_model_registry()
    seeded = await provider.probe_startup_inference()
    assert seeded == {}
    assert provider.runtime_stats == {}


@pytest.mark.asyncio
async def test_probe_startup_inference_seeds_latency_and_prefers_fastest(base_config, monkeypatch):
    """Boot probe seeds p50; equal-cost ranking prefers the faster measured model."""
    cfg = {**base_config}
    cfg['model_selection_strategy'] = {
        **base_config['model_selection_strategy'],
        'weights': {'capability': 0.0, 'cost': 1.0, 'latency': 0.0},
        'startup_inference_probe': {
            'enabled': True,
            'max_models': 8,
            'prefer_cost_tier_max': 5,
            'timeout_seconds': 5.0,
            'max_tokens': 16,
            'prompt': 'OK',
            'always_include_models': [],
        },
    }
    provider = LLMProvider(cfg)
    # Distinct families so probe selection keeps both (latest-per-family).
    models = ['gpt-4o-mini', 'claude-3-5-haiku-20241022']
    _seed_discovery(provider, models)
    provider._validated_providers = {
        'openai': 'sk-' + 'x' * 24,
        'anthropic': 'sk-ant-' + 'x' * 24,
    }
    provider.build_model_registry()
    # Equal cost so measured-latency tiebreak (not cheaper total_cost) decides.
    for m in models:
        provider._registry.registry[m]['total_cost'] = 0.1
        provider._registry.registry[m]['cost_tier'] = 1

    async def fake_probe(self, provider_name, api_key, model, base_url, timeout_s, max_tokens, prompt):
        lat = 900.0 if model == 'gpt-4o-mini' else 4000.0
        return True, lat, None

    monkeypatch.setattr(LLMProvider, '_timed_inference_probe', fake_probe)
    seeded = await provider.probe_startup_inference()
    assert set(seeded) == set(models)
    assert seeded['gpt-4o-mini']['p50_latency_ms'] == 900.0
    assert seeded['gpt-4o-mini']['sample_count'] >= cfg['model_selection_strategy']['runtime_adaptation']['min_samples']
    provider.build_task_fallbacks()
    ranked = provider.select_models_for_task('healing_analysis')
    assert ranked[0] == 'gpt-4o-mini'


def test_select_startup_probe_models_keeps_latest_per_family(base_config):
    """Probe set keeps newest SKU within a family under the cost-tier cap."""
    cfg = {**base_config}
    cfg['model_selection_strategy'] = {
        **base_config['model_selection_strategy'],
        'startup_inference_probe': {
            'enabled': True,
            'max_models': 8,
            'prefer_cost_tier_max': 5,
            'timeout_seconds': 5.0,
            'max_tokens': 16,
            'prompt': 'OK',
            'always_include_models': [],
        },
    }
    provider = LLMProvider(cfg)
    _seed_discovery(provider, ['grok-4.3', 'grok-4.5', 'gpt-4o-mini'])
    provider.build_model_registry()
    picked = provider._select_startup_probe_models()
    assert 'grok-4.5' in picked
    assert 'grok-4.3' not in picked
    assert 'gpt-4o-mini' in picked


def test_select_startup_probe_models_always_includes_configured(base_config):
    """always_include_models forces gpt-5-mini into the probe set even above cost cap."""
    cfg = {**base_config}
    cfg['model_selection_strategy'] = {
        **base_config['model_selection_strategy'],
        'startup_inference_probe': {
            'enabled': True,
            'max_models': 2,
            'prefer_cost_tier_max': 1,
            'timeout_seconds': 5.0,
            'max_tokens': 128,
            'prompt': 'OK',
            'always_include_models': ['gpt-5-mini'],
        },
    }
    provider = LLMProvider(cfg)
    # Cheap haiku-class + gpt-5-mini; force high cost_tier on mini so cap would skip it.
    models = ['claude-3-5-haiku-20241022', 'gpt-4o-mini', 'gpt-5-mini', 'gpt-4o']
    _seed_discovery(provider, models)
    provider.build_model_registry()
    provider._registry.registry['gpt-5-mini']['cost_tier'] = 5
    picked = provider._select_startup_probe_models()
    assert 'gpt-5-mini' in picked
    # Cap still applies to the non-forced set; always-include is additive.
    assert len(picked) <= 3  # max_models(2) + always_include(1) when not already selected


def test_select_startup_probe_models_includes_task_allowlist_fallbacks(base_config):
    """With allowlists: probe only selected task models ∩ available — not discovery catalog."""
    cfg = {**base_config}
    cfg['model_selection_strategy'] = {
        **base_config['model_selection_strategy'],
        'startup_inference_probe': {
            'enabled': True,
            'max_models': 16,  # ignored when allowlists present
            'prefer_cost_tier_max': 1,
            'timeout_seconds': 5.0,
            'max_tokens': 128,
            'prompt': 'OK',
            'always_include_models': [],
        },
        'task_model_allowlists': {
            'l0_entity_extraction': [
                'gpt-5-mini',
                'gemini-2.5-flash',
                'claude-haiku-4-5-20251001',
                'gpt-4o-mini',
            ],
            'query_intent_classification': [
                'gpt-5-mini',
                'claude-haiku-4-5-20251001',
            ],
        },
    }
    provider = LLMProvider(cfg)
    models = [
        'gpt-5-mini',
        'gemini-2.5-flash',
        'claude-haiku-4-5-20251001',
        'gpt-4o-mini',
        'o3',  # discovered but not on allowlist — must not be probed
    ]
    _seed_discovery(provider, models)
    provider.build_model_registry()
    picked = provider._select_startup_probe_models()
    assert set(picked) == {
        'gpt-5-mini',
        'gemini-2.5-flash',
        'claude-haiku-4-5-20251001',
        'gpt-4o-mini',
    }
    assert 'o3' not in picked


def test_task_latency_slo_prefers_models_under_budget(base_config):
    """Classify/extract SLO: under-budget measured models lead; over-SLO last-resort."""
    cfg = {**base_config}
    cfg['model_selection_strategy'] = {
        **base_config['model_selection_strategy'],
        'weights': {'capability': 0.5, 'cost': 0.3, 'latency': 0.2},
        'task_type_preferences': {'l0_entity_extraction': 0.6},
        'component_task_types': {},
        'task_latency_slos': {
            'l0_entity_extraction': {'max_p50_latency_ms': 2000},
        },
    }
    provider = LLMProvider(cfg)
    models = ['gpt-4o-mini', 'claude-3-5-haiku-20241022', 'gpt-4o']
    _seed_discovery(provider, models)
    provider.build_model_registry()
    for m in models:
        provider._registry.registry[m]['cost_tier'] = 1
        provider._registry.registry[m]['total_cost'] = 0.1
    # Seed startup-style stats: mini fast, haiku slow, gpt-4o unprobed.
    provider._runtime_stats = {
        'gpt-4o-mini': {
            'p50_latency_ms': 900.0,
            'error_rate': 0.0,
            'sample_count': 20,
            'quality_score': None,
            'source': 'startup_inference_probe',
        },
        'claude-3-5-haiku-20241022': {
            'p50_latency_ms': 8000.0,
            'error_rate': 0.0,
            'sample_count': 20,
            'quality_score': None,
            'source': 'startup_inference_probe',
        },
    }
    provider.build_task_fallbacks()
    chain = provider.model_fallbacks['l0_entity_extraction']
    assert chain[0] == 'gpt-4o-mini'
    assert 'claude-3-5-haiku-20241022' in chain
    assert chain.index('gpt-4o-mini') < chain.index('claude-3-5-haiku-20241022')
    # Unprobed stays last-resort (after preferred under-SLO).
    assert chain.index('gpt-4o-mini') < chain.index('gpt-4o')


def test_task_latency_slo_keeps_ranked_when_none_meet(base_config):
    """If every model misses the SLO, keep ranked order (never empty preferred)."""
    cfg = {**base_config}
    cfg['model_selection_strategy'] = {
        **base_config['model_selection_strategy'],
        'weights': {'capability': 0.0, 'cost': 1.0, 'latency': 0.0},
        'task_type_preferences': {'l0_entity_extraction': 0.6},
        'component_task_types': {},
        'task_latency_slos': {
            'l0_entity_extraction': {'max_p50_latency_ms': 2000},
        },
    }
    provider = LLMProvider(cfg)
    models = ['gpt-4o-mini', 'gpt-4o']
    _seed_discovery(provider, models)
    provider.build_model_registry()
    provider._runtime_stats = {
        'gpt-4o-mini': {'p50_latency_ms': 9000.0, 'error_rate': 0.0, 'sample_count': 20, 'quality_score': None},
        'gpt-4o': {'p50_latency_ms': 10000.0, 'error_rate': 0.0, 'sample_count': 20, 'quality_score': None},
    }
    before = provider.select_models_for_task('l0_entity_extraction')
    provider.build_task_fallbacks()
    assert provider.model_fallbacks['l0_entity_extraction'] == before


def test_task_latency_slos_invalid_raises(base_config):
    from llm_core.exceptions import ConfigurationError

    cfg = {**base_config}
    cfg['model_selection_strategy'] = {
        **base_config['model_selection_strategy'],
        'task_latency_slos': {'l0_entity_extraction': {'max_p50_latency_ms': 0}},
    }
    with pytest.raises(ConfigurationError, match='max_p50_latency_ms'):
        LLMProvider(cfg)


def test_task_model_allowlist_prefers_primary_excludes_others(base_config):
    """Allowlist order = preference; models not listed (o3) never enter the chain."""
    cfg = {**base_config}
    cfg['model_selection_strategy'] = {
        **base_config['model_selection_strategy'],
        'task_type_preferences': {
            'query_intent_classification': 0.6,
            'l0_entity_extraction': 0.6,
        },
        'component_task_types': {
            'qi_llm_classifier': 'query_intent_classification',
        },
        'task_model_allowlists': {
            'query_intent_classification': [
                'gpt-5-mini',
                'claude-haiku-4-5-20251001',
                'gpt-4o-mini',
            ],
            'l0_entity_extraction': [
                'gpt-5-mini',
                'claude-haiku-4-5-20251001',
                'gpt-4o-mini',
            ],
        },
    }
    provider = LLMProvider(cfg)
    models = [
        'claude-haiku-4-5-20251001',
        'gpt-5-mini',
        'o3',
        'gpt-4o-mini',
        'claude-3-5-sonnet-20241022',
    ]
    _seed_discovery(provider, models)
    provider.build_model_registry()
    provider.build_task_fallbacks()

    for task in ('query_intent_classification', 'l0_entity_extraction', 'qi_llm_classifier'):
        chain = provider.get_fallback_chain(task)
        assert chain[0] == 'gpt-5-mini', f'{task} chain={chain}'
        assert chain == [
            'gpt-5-mini',
            'claude-haiku-4-5-20251001',
            'gpt-4o-mini',
        ], f'{task} chain={chain}'
        assert 'o3' not in chain
        assert 'claude-3-5-sonnet-20241022' not in chain


def test_task_model_allowlist_falls_back_when_primary_missing(base_config):
    """If gpt-5-mini absent, next allowlisted model wins — still never o3."""
    cfg = {**base_config}
    cfg['model_selection_strategy'] = {
        **base_config['model_selection_strategy'],
        'task_type_preferences': {'l0_entity_extraction': 0.6},
        'component_task_types': {},
        'task_model_allowlists': {
            'l0_entity_extraction': [
                'gpt-5-mini',
                'claude-haiku-4-5-20251001',
                'gpt-4o-mini',
            ],
        },
    }
    provider = LLMProvider(cfg)
    _seed_discovery(provider, ['claude-haiku-4-5-20251001', 'o3', 'gpt-4o-mini'])
    provider.build_model_registry()
    provider.build_task_fallbacks()
    chain = provider.get_fallback_chain('l0_entity_extraction')
    assert chain == ['claude-haiku-4-5-20251001', 'gpt-4o-mini']
    assert 'o3' not in chain


def test_task_model_allowlist_empty_when_none_available(base_config):
    """If no allowlisted model is discovered, chain is empty (fail loud)."""
    cfg = {**base_config}
    cfg['model_selection_strategy'] = {
        **base_config['model_selection_strategy'],
        'task_type_preferences': {'l0_entity_extraction': 0.6},
        'component_task_types': {},
        'task_model_allowlists': {
            'l0_entity_extraction': ['gpt-5-mini', 'gpt-4o-mini'],
        },
    }
    provider = LLMProvider(cfg)
    _seed_discovery(provider, ['claude-haiku-4-5-20251001', 'o3'])
    provider.build_model_registry()
    provider.build_task_fallbacks()
    assert provider.get_fallback_chain('l0_entity_extraction') == []


def test_task_model_allowlists_invalid_raises(base_config):
    cfg = {**base_config}
    cfg['model_selection_strategy'] = {
        **base_config['model_selection_strategy'],
        'task_model_allowlists': {'l0_entity_extraction': []},
    }
    with pytest.raises(ConfigurationError, match='task_model_allowlists'):
        LLMProvider(cfg)


def test_get_primary_model_returns_allowlist_head(base_config):
    cfg = {**base_config}
    cfg['model_selection_strategy'] = {
        **base_config['model_selection_strategy'],
        'task_type_preferences': {'l0_entity_extraction': 0.6},
        'component_task_types': {},
        'task_model_allowlists': {
            'l0_entity_extraction': [
                'gemini-2.5-flash-lite',
                'gemini-2.5-flash',
                'gpt-4o-mini',
            ],
        },
    }
    provider = LLMProvider(cfg)
    _seed_discovery(provider, ['gpt-4o-mini', 'gemini-2.5-flash', 'o3'])
    provider.build_model_registry()
    provider.build_task_fallbacks()
    assert provider.get_primary_model('l0_entity_extraction') == 'gemini-2.5-flash'


def test_get_primary_model_empty_chain_raises(base_config):
    cfg = {**base_config}
    cfg['model_selection_strategy'] = {
        **base_config['model_selection_strategy'],
        'task_type_preferences': {'l0_entity_extraction': 0.6},
        'component_task_types': {},
        'task_model_allowlists': {
            'l0_entity_extraction': ['gemini-2.5-flash-lite'],
        },
    }
    provider = LLMProvider(cfg)
    _seed_discovery(provider, ['gpt-4o-mini'])
    provider.build_model_registry()
    provider.build_task_fallbacks()
    with pytest.raises(ConfigurationError, match='No available model'):
        provider.get_primary_model('l0_entity_extraction')


def test_api_key_for_model_google_dual_path_openai(base_config):
    """Gemini models use GOOGLE key when present; else OPENAI dual-path."""
    provider = LLMProvider(base_config)
    provider._validated_providers = {'openai': 'sk-openai-test-key'}
    key, probe_provider = provider._api_key_for_model('gemini-2.5-flash-lite')
    assert key == 'sk-openai-test-key'
    assert probe_provider == 'openai'
    provider._validated_providers = {
        'google': 'sk-google-test-key',
        'openai': 'sk-openai-test-key',
    }
    key, probe_provider = provider._api_key_for_model('gemini-2.5-flash-lite')
    assert key == 'sk-google-test-key'
    assert probe_provider == 'google'
