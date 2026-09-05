"""Tests for LLMProvider.refresh_rankings + start/stop background refresh lifecycle."""
import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from llm_core import LLMProvider


class _FakeStore:
    def __init__(self):
        self.entries = {'latency': [], 'error': [], 'eval_quality_score': []}

    def get_entries(self, signal_type=None, limit=None):
        return list(self.entries.get(signal_type, []))

    def push_latency(self, model, value):
        self.entries['latency'].append(SimpleNamespace(
            timestamp=datetime.now(timezone.utc).isoformat(),
            source='svc', signal_type='latency', value=value,
            metadata={'model': model}, origin_system='svc',
        ))

    def push_error(self, model):
        self.entries['error'].append(SimpleNamespace(
            timestamp=datetime.now(timezone.utc).isoformat(),
            source='svc', signal_type='error', value=1.0,
            metadata={'model': model}, origin_system='svc',
        ))


_FAKE_DISCOVERY = ['claude-3-5-haiku-20241022', 'claude-3-5-sonnet-20241022', 'gpt-4o-mini', 'gpt-4o']


def _build_provider_with_runtime(base_config, store):
    cfg = {**base_config}
    cfg['model_selection_strategy'] = {
        **base_config['model_selection_strategy'],
        'runtime_adaptation': {
            'enabled': True,
            'refresh_interval_s': 60,
            'min_samples': 5,
            'latency_window_s': 3600,
            'quality_window_s': 86400,
            'error_demotion_threshold': 0.5,
            'latency_buckets_ms': [500, 1500, 3000, 6000],
            'max_entries_per_type': 5000,
        },
    }
    provider = LLMProvider(cfg, feedback_store=store)
    # Discovery is the source of truth in production; tests inject a fixed model
    # set so ranker behaviour is deterministic without mocking the SDK.
    provider._available_models = list(_FAKE_DISCOVERY)
    provider._registry.update_models(provider._available_models)
    provider.build_model_registry()
    provider.build_task_fallbacks()
    return provider


def test_refresh_noop_without_store(base_config):
    provider = LLMProvider(base_config)
    provider.build_model_registry()
    provider.build_task_fallbacks()
    before = provider.model_fallbacks
    provider.refresh_rankings()
    assert provider.model_fallbacks == before


def test_refresh_demotes_high_error_rate(base_config):
    store = _FakeStore()
    provider = _build_provider_with_runtime(base_config, store)
    target = 'claude-3-5-sonnet-20241022'
    for _ in range(10):
        store.push_error(target)
    provider.refresh_rankings()
    chain = provider.get_fallback_chain('healing_analysis')
    assert chain[-1] == target


def test_refresh_atomic_swap(base_config):
    store = _FakeStore()
    provider = _build_provider_with_runtime(base_config, store)
    chain_before = provider.get_fallback_chain('healing_analysis')
    assert len(chain_before) == 4
    for _ in range(10):
        store.push_latency('gpt-4o-mini', 50.0)
    provider.refresh_rankings()
    chain_after = provider.get_fallback_chain('healing_analysis')
    assert len(chain_after) == 4
    assert set(chain_after) == set(chain_before)


@pytest.mark.asyncio
async def test_background_refresh_lifecycle(base_config):
    store = _FakeStore()
    provider = _build_provider_with_runtime(base_config, store)
    await provider.start_background_refresh(interval_s=1)
    assert provider._refresh_task is not None
    await asyncio.sleep(0.05)
    await provider.stop_background_refresh()
    assert provider._refresh_task is None


@pytest.mark.asyncio
async def test_background_refresh_skipped_when_disabled(base_config):
    """If runtime_adaptation.enabled is False, start_background_refresh must no-op."""
    store = _FakeStore()
    provider = LLMProvider(base_config, feedback_store=store)
    provider.build_model_registry()
    provider.build_task_fallbacks()
    await provider.start_background_refresh(interval_s=1)
    assert provider._refresh_task is None


@pytest.mark.asyncio
async def test_background_refresh_skipped_without_store(base_config):
    provider = LLMProvider(base_config)
    await provider.start_background_refresh(interval_s=1)
    assert provider._refresh_task is None
