"""Tests for the 13 new DomainAnalyticsEngine methods."""
import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from semantic_search.analytics.clickhouse_executor import ClickHouseExecutor
from semantic_search.analytics.domain_analytics_engine import DomainAnalyticsEngine
from semantic_search.config.analytics_models import DomainAnalyticsConfig
from semantic_search.core.exceptions import RetrievalError
from semantic_search.nl_to_sql.contracts import SqlExecutionResult


_ROWS = [{'col': 1}]


def _make_config(enabled: bool = True, **extra) -> DomainAnalyticsConfig:
    base = {
        'enabled': enabled,
        'source_table': 'dam_auction_snap',
        'domain_column': 'domain_name',
        'tld_column': 'tld',
        'price_column': 'current_price',
        'bid_count_column': 'bid_count',
        'auction_end_column': 'ends_at',
        'score_column': 'govalue_score',
        'category_column': 'auction_type_id',
        'created_column': 'created_at',
        'signals_user_column': 'user_id',
        'signals_event_column': 'signal_type',
        'signals_time_column': 'created_at',
        'signals_origin_column': 'signal_origin',
        'signals_table': 'analytics.feedback_signals',
        'signals_batch_size': 50,
        'signals_insert_timeout_seconds': 5.0,
        'sold_flag_column': 'is_sold',
        'listed_at_column': 'listed_at',
        'sold_at_column': 'sold_at',
        'registrar_column': 'registrar',
        'transactions_table': 'analytics.transactions',
        'transactions_sold_at_column': 'sold_at',
        'transactions_sale_price_column': 'sale_price',
        'comparable_min_sales': 5,
    }
    base.update(extra)
    cfg = DomainAnalyticsConfig.from_dict(base)
    for k, v in extra.items():
        if hasattr(cfg, k):
            object.__setattr__(cfg, k, v)
    return cfg


def _stub_executor(rows: List[Dict[str, Any]] = None) -> MagicMock:
    ex = MagicMock(spec=ClickHouseExecutor)
    ex.credentials_available = True
    ex.execute = AsyncMock(return_value=SqlExecutionResult(
        sql='SELECT 1',
        rows=rows if rows is not None else _ROWS,
        column_names=list((rows if rows is not None else _ROWS)[0].keys()),
        row_count=len(rows if rows is not None else _ROWS),
        latency_ms=5.0,
        truncated=False,
    ))
    ex.execute_insert = AsyncMock()
    return ex


def _engine(config: DomainAnalyticsConfig = None, rows: List[Dict[str, Any]] = None) -> DomainAnalyticsEngine:
    cfg = config if config is not None else _make_config()
    return DomainAnalyticsEngine(executor=_stub_executor(rows), config=cfg)


def _engine_with_mv_config(**mv_kwargs) -> DomainAnalyticsEngine:
    cfg = _make_config(**mv_kwargs)
    return DomainAnalyticsEngine(executor=_stub_executor(), config=cfg)


class TestBidVelocityPerAuction:
    def test_happy_path_returns_list(self):
        engine = _engine_with_mv_config(bid_velocity_mv='analytics.mv_bid_velocity')
        result = asyncio.run(engine.bid_velocity_per_auction(window_hours=2))
        assert isinstance(result, list)

    def test_missing_mv_raises_retrieval_error(self):
        engine = _engine(_make_config())
        with pytest.raises(RetrievalError, match='bid_velocity_mv'):
            asyncio.run(engine.bid_velocity_per_auction())


class TestWatchDensityAnalytics:
    def test_happy_path_returns_list(self):
        engine = _engine_with_mv_config(
            watch_density_mv='analytics.mv_watch_density',
            watch_mv_item_column='member_item_id',
            watch_mv_day_column='event_day',
            watch_mv_count_state_column='total_watches',
            watch_mv_unique_state_column='unique_watchers'
        )
        result = asyncio.run(engine.watch_density_analytics())
        assert isinstance(result, list)

    def test_missing_mv_raises_retrieval_error(self):
        engine = _engine(_make_config())
        with pytest.raises(RetrievalError, match='watch_density_mv'):
            asyncio.run(engine.watch_density_analytics())


class TestHoldTimeDistribution:
    def test_happy_path_returns_list(self):
        engine = _engine_with_mv_config(hold_time_mv='analytics.mv_hold_time')
        result = asyncio.run(engine.hold_time_distribution())
        assert isinstance(result, list)

    def test_missing_mv_raises_retrieval_error(self):
        engine = _engine(_make_config())
        with pytest.raises(RetrievalError, match='hold_time_mv'):
            asyncio.run(engine.hold_time_distribution())


class TestWatchBidConversion:
    def test_happy_path_returns_list(self):
        engine = _engine_with_mv_config(watch_density_mv='analytics.mv_watch', bid_velocity_item_mv='analytics.mv_bids')
        result = asyncio.run(engine.watch_bid_conversion())
        assert isinstance(result, list)

    def test_missing_watch_mv_raises_retrieval_error(self):
        engine = _engine_with_mv_config(bid_velocity_item_mv='analytics.mv_bids')
        with pytest.raises(RetrievalError):
            asyncio.run(engine.watch_bid_conversion())

    def test_missing_bid_mv_raises_retrieval_error(self):
        engine = _engine_with_mv_config(watch_density_mv='analytics.mv_watch')
        with pytest.raises(RetrievalError):
            asyncio.run(engine.watch_bid_conversion())


class TestCounterOfferAnalytics:
    def test_happy_path_returns_list(self):
        engine = _engine_with_mv_config(bid_events_table='analytics.bid_events')
        result = asyncio.run(engine.counter_offer_analytics())
        assert isinstance(result, list)

    def test_missing_table_raises_retrieval_error(self):
        engine = _engine(_make_config())
        with pytest.raises(RetrievalError, match='bid_events_table'):
            asyncio.run(engine.counter_offer_analytics())


class TestPriceRealizationRate:
    def test_happy_path_returns_list(self):
        engine = _engine_with_mv_config(transactions_table='analytics.domain_transactions')
        result = asyncio.run(engine.price_realization_rate())
        assert isinstance(result, list)

    def test_missing_table_raises_retrieval_error(self):
        cfg = _make_config(transactions_table=None)
        engine = _engine(cfg)
        with pytest.raises(RetrievalError, match='transactions_table'):
            asyncio.run(engine.price_realization_rate())


class TestBuyerRetentionCohorts:
    def test_happy_path_returns_list(self):
        engine = _engine_with_mv_config(transactions_table='analytics.domain_transactions')
        result = asyncio.run(engine.buyer_retention_cohorts())
        assert isinstance(result, list)

    def test_missing_table_raises_retrieval_error(self):
        cfg = _make_config(transactions_table=None)
        engine = _engine(cfg)
        with pytest.raises(RetrievalError, match='transactions_table'):
            asyncio.run(engine.buyer_retention_cohorts())


class TestAuctionTimingPatterns:
    def test_happy_path_returns_list(self):
        result = asyncio.run(_engine().auction_timing_patterns())
        assert isinstance(result, list)

    def test_disabled_engine_raises(self):
        engine = _engine(_make_config(enabled=False))
        with pytest.raises(RetrievalError, match='enabled=false'):
            asyncio.run(engine.auction_timing_patterns())


class TestNameStructureAnalytics:
    def test_happy_path_returns_list(self):
        result = asyncio.run(_engine().name_structure_analytics())
        assert isinstance(result, list)

    def test_disabled_engine_raises(self):
        engine = _engine(_make_config(enabled=False))
        with pytest.raises(RetrievalError, match='enabled=false'):
            asyncio.run(engine.name_structure_analytics())


class TestCrossTldPriceSpread:
    def test_happy_path_returns_list(self):
        result = asyncio.run(_engine().cross_tld_price_spread())
        assert isinstance(result, list)

    def test_disabled_engine_raises(self):
        engine = _engine(_make_config(enabled=False))
        with pytest.raises(RetrievalError, match='enabled=false'):
            asyncio.run(engine.cross_tld_price_spread())


class TestSearchAttribution:
    def test_happy_path_returns_list(self):
        engine = _engine_with_mv_config(transactions_table='analytics.domain_transactions')
        result = asyncio.run(engine.search_attribution())
        assert isinstance(result, list)

    def test_missing_table_raises_retrieval_error(self):
        cfg = _make_config(transactions_table=None)
        engine = _engine(cfg)
        with pytest.raises(RetrievalError, match='transactions_table'):
            asyncio.run(engine.search_attribution())


class TestRegistrarHhi:
    def test_happy_path_returns_list(self):
        result = asyncio.run(_engine().registrar_hhi())
        assert isinstance(result, list)

    def test_disabled_engine_raises(self):
        engine = _engine(_make_config(enabled=False))
        with pytest.raises(RetrievalError, match='enabled=false'):
            asyncio.run(engine.registrar_hhi())


class TestComparableSalePrice:
    def test_happy_path_returns_list(self):
        engine = _engine_with_mv_config(transactions_table='analytics.domain_transactions')
        result = asyncio.run(engine.comparable_sale_price())
        assert isinstance(result, list)

    def test_missing_table_raises_retrieval_error(self):
        cfg = _make_config(transactions_table=None)
        engine = _engine(cfg)
        with pytest.raises(RetrievalError, match='transactions_table'):
            asyncio.run(engine.comparable_sale_price())
