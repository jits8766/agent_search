"""EnrichmentRefreshDriver — per-source polling, domain_name-keyed Qdrant patch, Qdrant-only scope."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from semantic_search.config.models import (
    EnrichmentRefreshConfig,
    SeedDatabaseConfig,
    SeedEstibotConfig,
    SeedMajesticConfig,
    SeedSearchRollupConfig,
    SeedSemrushConfig,
)
from semantic_search.nl_to_sql.athena_client import AthenaClient
from semantic_search.retrieval.qdrant_adapter import QdrantClientFactory
from semantic_search.vectorization.enrichment_refresh_driver import EnrichmentRefreshDriver


class _FakeAthena(AthenaClient):
    def __init__(self) -> None:  # noqa: D107
        self.credentials_available = True


class _FakeQdrant(QdrantClientFactory):
    def __init__(self) -> None:  # noqa: D107
        self._available = True
        self._client = AsyncMock()
        self._config = MagicMock()

    @property
    def available(self) -> bool:
        return self._available

    @property
    def client(self) -> Any:
        return self._client

    @property
    def config(self) -> Any:
        return self._config


class _FakeSeedDatabase(SeedDatabaseConfig):
    """Bypasses SeedDatabaseConfig.__post_init__ (tables/merge_database/etc. are
    unrelated to this driver) while still satisfying the isinstance check in
    EnrichmentRefreshDriver.__init__ — mirrors _FakeAthena/_FakeQdrant above."""

    def __init__(self, majestic, semrush=None, estibot=None, search_rollup=None) -> None:  # noqa: D107
        self.majestic = majestic
        self.semrush = semrush
        self.estibot = estibot
        self.search_rollup = search_rollup


def _majestic_cfg() -> SeedMajesticConfig:
    return SeedMajesticConfig(database="domain_feature_mart", table_name="domain_majestic_metric_snap", timeout_seconds=120.0)


def _semrush_cfg() -> SeedSemrushConfig:
    return SeedSemrushConfig(database="domain_auction_mart", table_name="semrush_domain_enrichments", timeout_seconds=120.0)


def _estibot_cfg() -> SeedEstibotConfig:
    return SeedEstibotConfig(database="domain_auction_mart", table_name="estibot_domain_enrichments", timeout_seconds=120.0)


def _rollup_cfg() -> SeedSearchRollupConfig:
    return SeedSearchRollupConfig(database="domain_search", table_name="domain_search_rollup", lookback_days=30, timeout_seconds=120.0)


def _driver_cfg(**overrides: Any) -> EnrichmentRefreshConfig:
    base = dict(
        enabled=True,
        interval_seconds=900.0,
        max_consecutive_failures=3,
        lookback_minutes=120,
        batch_size=50000,
        timeout_seconds=120.0,
        chunk_minutes=120,
    )
    base.update(overrides)
    return EnrichmentRefreshConfig(**base)


def _make_driver(**seed_kwargs: Any) -> EnrichmentRefreshDriver:
    seed_kwargs.setdefault("majestic", _majestic_cfg())
    seed_db = _FakeSeedDatabase(**seed_kwargs)
    return EnrichmentRefreshDriver(_driver_cfg(), seed_db, _FakeAthena(), _FakeQdrant())


@pytest.mark.asyncio
async def test_patch_qdrant_filters_by_domain_name_not_payload_id_field():
    driver = _make_driver(semrush=_semrush_cfg())
    pairs: List[Tuple[str, Dict[str, Any]]] = [("example.com", {"semrush_ascore": 42.0})]
    patched, skipped = await driver._patch_qdrant(pairs)
    assert patched == 1
    assert skipped == 0
    client = driver._qdrant.client
    client.set_payload.assert_awaited_once()
    kwargs = client.set_payload.await_args.kwargs
    assert kwargs["payload"] == {"semrush_ascore": 42.0}
    points_filter = kwargs["points"]
    condition = points_filter.must[0]
    assert condition.key == "domain_name"
    assert condition.match.value == "example.com"


@pytest.mark.asyncio
async def test_source_skipped_when_seed_config_absent():
    driver = _make_driver(semrush=None, estibot=None, search_rollup=None)
    assert "semrush" not in driver._states
    assert "estibot" not in driver._states
    assert "search_rollup" not in driver._states
    assert "majestic" in driver._states


@pytest.mark.asyncio
async def test_majestic_cycle_chunks_since_now_and_advances_cursor():
    driver = _make_driver()
    driver._states["majestic"].last_polled_at = 1_000.0
    calls: List[Tuple[float, float]] = []

    async def _fake_fetch(athena_client, cfg, since_ts, now_ts, batch_size, timeout_seconds):
        calls.append((since_ts, now_ts))
        return [("a.com", {"majestic_ext_back_links": 5})]

    with patch("semantic_search.vectorization.enrichment_refresh_driver.fetch_majestic_delta", new=_fake_fetch), \
         patch("semantic_search.vectorization.enrichment_refresh_driver.time.time", return_value=1_000.0 + 60.0):
        await driver._run_cycle("majestic", driver._states["majestic"])
    assert len(calls) == 1
    assert calls[0] == (1_000.0, 1_060.0)
    assert driver._states["majestic"].last_polled_at == 1_060.0
    assert driver._states["majestic"].consecutive_failures == 0
    driver._qdrant.client.set_payload.assert_awaited_once()


@pytest.mark.asyncio
async def test_rollup_cycle_has_no_since_now_params():
    driver = _make_driver(search_rollup=_rollup_cfg())
    fetch_mock = AsyncMock(return_value=[("b.com", {"unique_search_count": 3})])
    with patch("semantic_search.vectorization.enrichment_refresh_driver.fetch_rollup_full", new=fetch_mock):
        await driver._run_cycle("search_rollup", driver._states["search_rollup"])
    fetch_mock.assert_awaited_once()
    args = fetch_mock.await_args.args
    assert args[1] is driver._states["search_rollup"].cfg
    assert len(args) == 4  # (athena_client, cfg, batch_size, timeout_seconds) — no since_ts/now_ts
    assert driver._cycles_success == 1


@pytest.mark.asyncio
async def test_source_failure_pauses_only_that_source():
    driver = _make_driver(semrush=_semrush_cfg())
    driver._config.max_consecutive_failures = 1

    async def _boom(*args, **kwargs):
        raise RuntimeError("athena down")

    async def _ok(*args, **kwargs):
        return []

    with patch("semantic_search.vectorization.enrichment_refresh_driver.fetch_majestic_delta", new=_boom), \
         patch("semantic_search.vectorization.enrichment_refresh_driver.fetch_semrush_delta", new=_ok):
        await driver._run_cycle("majestic", driver._states["majestic"])
        await driver._run_cycle("semrush", driver._states["semrush"])
    assert driver._states["majestic"].paused is True
    assert driver._states["majestic"].consecutive_failures == 1
    assert driver._states["semrush"].paused is False
    assert driver._states["semrush"].consecutive_failures == 0


@pytest.mark.asyncio
async def test_no_clickhouse_write_path_exists():
    driver = _make_driver()
    assert not hasattr(driver, "set_ch_executor")
    assert not hasattr(driver, "_ch_executor")
    assert not hasattr(driver, "_write_clickhouse")
