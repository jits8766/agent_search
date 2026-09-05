"""ClickHouse-off degrade contracts: readiness probe, master lever, path smokes."""
from __future__ import annotations

import copy
import logging
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

from semantic_search.analytics.clickhouse_client import ClickHouseClient
from semantic_search.config.analytics_models import ClickHouseClientConfig
from semantic_search.config.clickhouse_lever import apply_clickhouse_lever
from semantic_search.config.models import AgentSearchConfig
from semantic_search.contracts import IntentSlice, QueryIntent
from semantic_search.guidance.guidance_service import GuidanceService
from semantic_search.registry import _build_price_band_store, build_subsystems
from semantic_search.retrieval.sql_retriever import InMemoryPriceBandStore


def _apply_memory_backends(raw: Dict[str, Any]) -> Dict[str, Any]:
    qi_enc = raw.setdefault('qi', {}).setdefault('encoder', {})
    qi_enc['backend'] = 'hashing'
    sem_dim = raw.get('qi', {}).get('semantic', {}).get('embedding_dim', 128)
    qi_enc['dim'] = sem_dim
    qi_enc.setdefault('cascade', {})['enabled'] = False
    raw.setdefault('qi', {}).setdefault('semantic', {})['learned_head'] = None
    raw.setdefault('retrieval', {}).setdefault('vector', {})['backend'] = 'memory'
    raw['retrieval']['vector']['embedding_dim'] = sem_dim
    raw.setdefault('retrieval', {}).setdefault('structured', {})['backend'] = 'memory'
    raw.setdefault('retrieval', {}).setdefault('qdrant', {}).setdefault('hybrid', {})['enabled'] = False
    raw.setdefault('clickhouse', {})['enabled'] = True
    return raw


def _lever_off_nested_on(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Master lever off while nested CH feature flags stay true (lever must win)."""
    raw = _apply_memory_backends(copy.deepcopy(raw))
    raw['clickhouse'] = {'enabled': False}
    raw['nl_to_sql']['analytics']['enabled'] = True
    raw['explore']['clickhouse_rails']['enabled'] = True
    raw['retrieval']['sql']['clickhouse_adapter']['enabled'] = True
    raw['guidance']['enabled'] = True
    raw['guidance'].setdefault(
        'snapshot_unavailable_notice',
        'Market snapshot is unavailable — showing best-match domain recommendations for your query instead.',
    )
    return raw


def _make_intent(query_type: str, query: str = 'test query') -> QueryIntent:
    return QueryIntent(
        request_id=QueryIntent.new_request_id(),
        raw_query=query,
        normalized_query=query.lower(),
        query_type=query_type,
        confidence=0.99,
        decision_tier='L1_semantic',
        slices=[IntentSlice(query_type=query_type, entities=[], confidence=0.99, raw_text=query)],
        decision_cost_usd=0.0,
        intent_record_id=QueryIntent.new_intent_record_id(),
        alternative_interpretations=[],
        routing_mode='auto_execute',
    )


def test_readiness_probe_config_required(config_dict):
    ch = copy.deepcopy(config_dict['nl_to_sql']['analytics']['clickhouse'])
    ch.pop('readiness_probe', None)
    with pytest.raises(Exception):
        ClickHouseClientConfig.from_dict(ch)


def test_readiness_probe_disabled_marks_available_without_live_ch(config):
    cfg = copy.deepcopy(config.nl_to_sql.analytics.clickhouse)
    d = {
        'host': cfg.host,
        'port': cfg.port,
        'database': cfg.database,
        'user': cfg.user,
        'secure': cfg.secure,
        'connect_timeout_seconds': cfg.connect_timeout_seconds,
        'read_timeout_seconds': cfg.read_timeout_seconds,
        'max_retries': cfg.max_retries,
        'query_max_execution_time_seconds': cfg.query_max_execution_time_seconds,
        'execution_timeout_seconds': cfg.execution_timeout_seconds,
        'mv_freshness_probe': {
            'enabled': False,
            'interval_successful_queries': 0,
            'lag_sql': '',
            'max_lag_seconds': 1.0,
            'query_timeout_seconds': 1.0,
        },
        'readiness_probe': {
            'enabled': False,
            'sql': 'SELECT 1',
            'timeout_seconds': 1.0,
            'warm_timeout_seconds': 2.0,
        },
    }
    client = ClickHouseClient(config=ClickHouseClientConfig.from_dict(d), password=None, transport=None, health_registry=None)
    assert client.available is True


def test_clickhouse_lever_config_required(config_dict):
    raw = copy.deepcopy(config_dict)
    raw.pop('clickhouse', None)
    with pytest.raises(Exception):
        AgentSearchConfig.from_dict(raw)


def test_apply_clickhouse_lever_forces_nested_flags(config_dict):
    raw = _lever_off_nested_on(config_dict)
    cfg = AgentSearchConfig.from_dict(raw)
    assert cfg.clickhouse.enabled is False
    assert cfg.nl_to_sql.analytics.enabled is True
    assert cfg.explore.clickhouse_rails.enabled is True
    assert cfg.retrieval.sql.clickhouse_adapter.enabled is True
    applied = apply_clickhouse_lever(cfg)
    assert applied.clickhouse.enabled is False
    assert applied.nl_to_sql.analytics.enabled is False
    assert applied.explore.clickhouse_rails.enabled is False
    assert applied.retrieval.sql.clickhouse_adapter.enabled is False


def test_apply_clickhouse_lever_noop_when_enabled(config_dict):
    raw = _apply_memory_backends(copy.deepcopy(config_dict))
    raw['clickhouse'] = {'enabled': True}
    raw['nl_to_sql']['analytics']['enabled'] = True
    raw['explore']['clickhouse_rails']['enabled'] = True
    raw['retrieval']['sql']['clickhouse_adapter']['enabled'] = True
    cfg = AgentSearchConfig.from_dict(raw)
    applied = apply_clickhouse_lever(cfg)
    assert applied.clickhouse.enabled is True
    assert applied.nl_to_sql.analytics.enabled is True
    assert applied.explore.clickhouse_rails.enabled is True
    assert applied.retrieval.sql.clickhouse_adapter.enabled is True
    assert applied is cfg


def test_build_subsystems_lever_on_keeps_nested_flags(config_dict):
    """Lever true leaves nested CH flags as declared (wiring still needs LLM for router)."""
    raw = _apply_memory_backends(copy.deepcopy(config_dict))
    raw['clickhouse'] = {'enabled': True}
    raw['nl_to_sql']['analytics']['enabled'] = True
    raw['explore']['clickhouse_rails']['enabled'] = True
    raw['retrieval']['sql']['clickhouse_adapter']['enabled'] = True
    sub = build_subsystems(raw, llm_provider=None)
    assert sub.config.clickhouse.enabled is True
    assert sub.config.nl_to_sql.analytics.enabled is True
    assert sub.config.explore.clickhouse_rails.enabled is True
    assert sub.config.retrieval.sql.clickhouse_adapter.enabled is True
    # No LLM → analytics router still not wired; lever on does not invent a router.
    assert sub.analytics_router is None
    assert sub.orchestrator.analytics_available is False


def test_build_subsystems_lever_off_despite_nested_true(config_dict):
    raw = _lever_off_nested_on(config_dict)
    sub = build_subsystems(raw, llm_provider=None)
    assert sub.config.clickhouse.enabled is False
    assert sub.config.nl_to_sql.analytics.enabled is False
    assert sub.config.explore.clickhouse_rails.enabled is False
    assert sub.config.retrieval.sql.clickhouse_adapter.enabled is False
    assert sub.analytics_router is None
    assert sub.orchestrator.analytics_available is False
    assert sub.guidance_service is not None
    store = _build_price_band_store(config=sub.config, backend_health=None)
    assert isinstance(store, InMemoryPriceBandStore)


def test_build_subsystems_analytics_off_never_wires_router(config_dict):
    raw = _apply_memory_backends(copy.deepcopy(config_dict))
    raw['clickhouse'] = {'enabled': True}
    raw['nl_to_sql']['analytics']['enabled'] = False
    raw['explore']['clickhouse_rails']['enabled'] = False
    raw['retrieval']['sql']['clickhouse_adapter']['enabled'] = False
    raw['guidance']['enabled'] = True
    sub = build_subsystems(raw, llm_provider=None)
    assert sub.analytics_router is None
    assert sub.orchestrator.analytics_available is False
    assert sub.guidance_service is not None
    store = _build_price_band_store(config=sub.config, backend_health=None)
    assert isinstance(store, InMemoryPriceBandStore)


@pytest.mark.asyncio
async def test_guidance_skip_logs_and_returns_none_when_no_ch(config, caplog):
    caplog.set_level(logging.WARNING)
    gs = GuidanceService(config=config.guidance, ch_executor=None)
    gs._config.enabled = True
    out = await gs.build_snapshot(request_id='rid-skip')
    assert out is None
    assert any('guidance_snapshot_skipped' in r.message and 'no_ch_executor' in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_guidance_skip_when_credentials_unavailable(config, caplog):
    caplog.set_level(logging.WARNING)
    fake = MagicMock()
    fake.credentials_available = False
    gs = GuidanceService(config=config.guidance, ch_executor=fake)
    gs._config.enabled = True
    out = await gs.build_snapshot(request_id='rid-creds')
    assert out is None
    fake.execute.assert_not_called()
    assert any('credentials_unavailable' in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_explore_ch_rail_logs_skip(config, caplog):
    from semantic_search.explore.ch_clickhouse_sources import ClickHouseTrendingExploreSource
    caplog.set_level(logging.WARNING)
    fake_ch = MagicMock()
    fake_ch.credentials_available = False
    src = ClickHouseTrendingExploreSource(
        trending=config.explore.trending,
        rails_ch=config.explore.clickhouse_rails,
        ch_executor=fake_ch,
    )
    src._rails.enabled = True
    src._trending.source.enabled = True
    cards = await src.fetch(user_id=None, max_items=5)
    assert cards == []
    assert any('explore_ch_rail_skipped' in r.message and 'credentials_unavailable' in r.message for r in caplog.records)


@pytest.mark.asyncio
@pytest.mark.parametrize('query_type,query', [
    ('hybrid', 'cheap .com domains under 500'),
    ('explore', 'trending domains'),
    ('guidance', 'should I bid on this auction'),
    ('analytics', 'average price of .com auctions by auction type'),
])
async def test_query_types_survive_clickhouse_lever_off(config_dict, query_type, query):
    """With lever off, hybrid/explore/guidance/analytics search paths must not raise."""
    raw = _lever_off_nested_on(config_dict)
    sub = build_subsystems(raw, llm_provider=None)
    assert sub.orchestrator.analytics_available is False

    intent = _make_intent(query_type, query)

    async def _fake_classify(**_kwargs):
        return intent

    sub.qi_engine.classify = _fake_classify  # type: ignore[method-assign]
    ranked, _eranker, _guard = await sub.orchestrator.search(
        raw_query=query,
        request_id=intent.request_id,
        top_k=5,
    )
    assert ranked is not None
    assert ranked.request_id == intent.request_id
    # Analytics with no router falls through to domain retrieve (items may be empty
    # on a cold memory index). Guidance may omit envelope when CH is down.
    if query_type == 'analytics':
        assert sub.analytics_router is None
    if query_type == 'guidance':
        snap = await sub.guidance_service.build_snapshot(request_id=intent.request_id)
        assert snap is None
