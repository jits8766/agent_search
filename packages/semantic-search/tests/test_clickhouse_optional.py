"""Robustness: the service must boot and serve hybrid search without ClickHouse.

Covers two invariants the hybrid path relies on when the analytics/CH side is
absent or unreachable:

1. Boot never crashes — when the price-band adapter is enabled but no CH client
   config exists, `_build_price_band_store` degrades to the in-memory store
   instead of raising.
2. Graceful per-call degradation — `ClickHousePriceBandStore.lookup` returns an
   empty list (never raises) when the underlying client is unavailable, so the
   SQL leg drops out of fusion cleanly rather than breaking hybrid search.
"""
import copy

import pytest

from semantic_search.analytics.clickhouse_client import ClickHouseClient
from semantic_search.registry import _build_price_band_store
from semantic_search.retrieval.clickhouse_price_band_store import ClickHousePriceBandStore
from semantic_search.retrieval.sql_retriever import InMemoryPriceBandStore


def test_price_band_store_degrades_to_in_memory_when_analytics_config_missing(config):
    """Adapter enabled + no nl_to_sql.analytics → InMemory store, no crash."""
    cfg = copy.deepcopy(config)
    assert cfg.retrieval.sql.clickhouse_adapter is not None
    cfg.retrieval.sql.clickhouse_adapter.enabled = True
    cfg.nl_to_sql.analytics = None  # CH client config absent
    store = _build_price_band_store(config=cfg, backend_health=None)
    assert isinstance(store, InMemoryPriceBandStore)


def test_price_band_store_disabled_adapter_uses_in_memory(config):
    """Adapter explicitly disabled → InMemory store (baseline path)."""
    cfg = copy.deepcopy(config)
    if cfg.retrieval.sql.clickhouse_adapter is not None:
        cfg.retrieval.sql.clickhouse_adapter.enabled = False
    store = _build_price_band_store(config=cfg, backend_health=None)
    assert isinstance(store, InMemoryPriceBandStore)


@pytest.mark.asyncio
async def test_price_band_lookup_returns_empty_when_clickhouse_unavailable(config):
    """CH client unavailable → lookup yields [] (never raises), so hybrid survives."""
    adapter_cfg = config.retrieval.sql.clickhouse_adapter
    assert adapter_cfg is not None
    ch_client = ClickHouseClient(
        config=config.nl_to_sql.analytics.clickhouse,
        password=None,
        transport=None,
        health_registry=None,
    )
    # Simulate ClickHouse being unreachable after construction.
    ch_client._available = False  # noqa: SLF001 — test forces the down state
    store = ClickHousePriceBandStore(config=adapter_cfg, client=ch_client)
    rows = await store.lookup({'price_min': 10, 'price_max': 100}, top_k=20)
    assert rows == []
