"""DeltaRefreshDriver → ClickHouse mutable-field patch wiring."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from semantic_search.config.models import DeltaRefreshConfig
from semantic_search.explore.ch_delta_analytics import ChDeltaWriteSummary
from semantic_search.nl_to_sql.athena_client import AthenaClient
from semantic_search.retrieval.qdrant_adapter import QdrantClientFactory
from semantic_search.vectorization.delta_driver import DeltaRefreshDriver


class _FakeAthena(AthenaClient):
    def __init__(self) -> None:  # noqa: D107
        self.credentials_available = True


class _FakeQdrant(QdrantClientFactory):
    def __init__(self) -> None:  # noqa: D107
        self._available = False
        self._client = None
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


def _delta_cfg(**overrides: Any) -> DeltaRefreshConfig:
    base = dict(
        enabled=True,
        interval_seconds=30.0,
        max_consecutive_failures=3,
        source_table="auction_audit_cln",
        source_database="signals_platform_cln",
        lookback_minutes=60,
        batch_size=1000,
        timeout_seconds=30.0,
        mutable_fields=["price", "bid_count", "auction_type", "ends_at"],
        chunk_minutes=60,
        find_payload_aliases={
            "price": ["auction_price", "auction_price_usd"],
            "bid_count": ["bids"],
            "ends_at": ["auction_end_time", "end_time"],
        },
        find_bool_aliases={},
    )
    base.update(overrides)
    return DeltaRefreshConfig(**base)


def _make_driver() -> DeltaRefreshDriver:
    return DeltaRefreshDriver(_delta_cfg(), _FakeAthena(), _FakeQdrant())


@pytest.mark.asyncio
async def test_run_cycle_writes_clickhouse_when_executor_wired():
    driver = _make_driver()
    ch = MagicMock()
    driver.set_ch_executor(ch)
    pairs: List[Tuple[str, Dict[str, Any]]] = [
        ("12345", {"price": 10.0, "bid_count": 2, "auction_type": "16", "ends_at": 1_700_000_000.0}),
    ]
    with patch(
        "semantic_search.vectorization.delta_driver.fetch_delta",
        new=AsyncMock(return_value=pairs),
    ), patch(
        "semantic_search.vectorization.delta_driver.write_delta_to_clickhouse",
        new=AsyncMock(return_value=ChDeltaWriteSummary(rows_written=1, errors=0, elapsed_ms=1.0)),
    ) as write_ch:
        await driver._run_cycle()
    write_ch.assert_awaited()
    assert write_ch.await_args.args[0] == pairs
    assert write_ch.await_args.args[1] is ch
    assert driver._cycles_success == 1


@pytest.mark.asyncio
async def test_run_cycle_skips_clickhouse_when_executor_unwired():
    driver = _make_driver()
    pairs = [("99", {"price": 1.0})]
    with patch(
        "semantic_search.vectorization.delta_driver.fetch_delta",
        new=AsyncMock(return_value=pairs),
    ), patch(
        "semantic_search.vectorization.delta_driver.write_delta_to_clickhouse",
        new=AsyncMock(),
    ) as write_ch:
        await driver._run_cycle()
    write_ch.assert_not_awaited()
    assert driver._cycles_success == 1


@pytest.mark.asyncio
async def test_ch_write_failure_does_not_fail_cycle():
    driver = _make_driver()
    driver.set_ch_executor(MagicMock())
    pairs = [("77", {"price": 5.0, "bid_count": 1})]
    with patch(
        "semantic_search.vectorization.delta_driver.fetch_delta",
        new=AsyncMock(return_value=pairs),
    ), patch(
        "semantic_search.vectorization.delta_driver.write_delta_to_clickhouse",
        new=AsyncMock(side_effect=RuntimeError("ch down")),
    ):
        await driver._run_cycle()
    assert driver._cycles_success == 1
    assert driver._paused_after_failures is False


@pytest.mark.asyncio
async def test_write_clickhouse_helper_returns_zero_when_unwired():
    driver = _make_driver()
    n = await driver._write_clickhouse([("1", {"price": 1.0})])
    assert n == 0
