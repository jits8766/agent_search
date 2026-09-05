"""Tests for DomainAnalyticsEngine and FeedbackSignalWriter.

Coverage matrix:

DomainAnalyticsEngine — constructor guards:
    - rejects_non_executor                                  TestDomainAnalyticsEngineInit::test_rejects_non_executor
- rejects_non_config                                    TestDomainAnalyticsEngineInit::test_rejects_non_config
- rejects_non_capabilities                              TestDomainAnalyticsEngineInit::test_rejects_non_capabilities

DomainAnalyticsEngine — disabled guard:
    - disabled_engine_raises_on_every_method                TestDomainAnalyticsEngineDisabled::test_disabled_raises

DomainAnalyticsEngine — method→template wiring (SQL keyword checks):
    - tld_popularity_trends_calls_category_trend            TestDomainAnalyticsEngineMethods::test_tld_popularity_trends
- price_movement_calls_time_window                      TestDomainAnalyticsEngineMethods::test_price_movement
- price_movement_adds_tld_filter                        TestDomainAnalyticsEngineMethods::test_price_movement_with_tld
- auction_competitiveness_calls_ranking_score           TestDomainAnalyticsEngineMethods::test_auction_competitiveness
- expiry_pipeline_forecast_calls_forecast               TestDomainAnalyticsEngineMethods::test_expiry_pipeline_forecast
- realtime_auction_activity_calls_realtime_trending     TestDomainAnalyticsEngineMethods::test_realtime_auction_activity
- historical_sales_comparison_delegates_correctly       TestDomainAnalyticsEngineMethods::test_historical_sales_comparison
- seasonal_trends_calls_time_window_month_grain         TestDomainAnalyticsEngineMethods::test_seasonal_trends
- domain_quality_score_calls_ranking_score              TestDomainAnalyticsEngineMethods::test_domain_quality_score
- domain_liquidity_calls_growth_rate                    TestDomainAnalyticsEngineMethods::test_domain_liquidity
- supply_demand_calls_time_window_two_metrics           TestDomainAnalyticsEngineMethods::test_supply_demand_analytics
- keyword_trends_calls_keyword_trend_sql                TestDomainAnalyticsEngineMethods::test_keyword_trends
- emerging_niche_calls_anomaly_detection                TestDomainAnalyticsEngineMethods::test_emerging_niche_detection
- multidim_tld_category_calls_multidim_aggregation      TestDomainAnalyticsEngineMethods::test_multidim_tld_category
- conversion_funnel_uses_signals_table                  TestDomainAnalyticsEngineMethods::test_conversion_funnel
- buyer_interest_heatmap_uses_signals_table             TestDomainAnalyticsEngineMethods::test_buyer_interest_heatmap

DomainAnalyticsEngine — executor failure propagates as RetrievalError:
    - retrieval_error_bubbles                               TestDomainAnalyticsEngineError::test_retrieval_error_bubbles

FeedbackSignalWriter — constructor guards:
    - rejects_non_executor                                  TestFeedbackSignalWriterInit::test_rejects_non_executor
- rejects_non_config                                    TestFeedbackSignalWriterInit::test_rejects_non_config

FeedbackSignalWriter — write():
    - write_calls_execute_insert_once                       TestFeedbackSignalWriterWrite::test_write_calls_execute_insert
- write_sql_contains_signal_fields                      TestFeedbackSignalWriterWrite::test_write_sql_content
- write_disabled_is_noop                                TestFeedbackSignalWriterWrite::test_write_disabled_noop
- write_error_is_swallowed                              TestFeedbackSignalWriterWrite::test_write_error_swallowed

FeedbackSignalWriter — write_batch():
    - write_batch_chunks_at_batch_size                      TestFeedbackSignalWriterBatch::test_batch_chunks
- write_batch_empty_is_noop                             TestFeedbackSignalWriterBatch::test_batch_empty_noop
- write_batch_non_signal_items_dropped                  TestFeedbackSignalWriterBatch::test_batch_drops_non_signals
- write_batch_error_is_swallowed                        TestFeedbackSignalWriterBatch::test_batch_error_swallowed

DomainAnalyticsConfig invariants:
    - batch_size_zero_rejected                              TestDomainAnalyticsConfig::test_batch_size_zero_rejected
- insert_timeout_zero_rejected                          TestDomainAnalyticsConfig::test_insert_timeout_zero_rejected
- empty_source_table_rejected                           TestDomainAnalyticsConfig::test_empty_source_table_rejected
"""
import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from semantic_search.analytics.clickhouse_executor import ClickHouseExecutor
from semantic_search.analytics.domain_analytics_engine import DomainAnalyticsEngine
from semantic_search.analytics.query_templates import TimeGrain
from semantic_search.analytics.signal_writer import FeedbackSignalWriter
from semantic_search.config.analytics_models import AnalyticsCapabilitiesConfig, AnomalyDetectionConfig, DomainAnalyticsConfig, ForecastingConfig, GrowthTrackingConfig
from semantic_search.config.analytics_models import HistoricalSnapshotConfig, RankingScoringConfig, SignalWriterConfig, TimeWindowAnalyticsConfig
from semantic_search.contracts import FeedbackSignal
from semantic_search.core.exceptions import ConfigurationError, RetrievalError
from semantic_search.nl_to_sql.contracts import SqlExecutionResult


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _da_config(enabled: bool = True) -> DomainAnalyticsConfig:
    return DomainAnalyticsConfig.from_dict({
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
    })


def _sw_config(enabled: bool = True, batch_size: int = 10) -> SignalWriterConfig:
    return SignalWriterConfig.from_dict({
        'enabled': enabled,
        'signals_table': 'analytics.feedback_signals',
        'batch_size': batch_size,
        'insert_timeout_seconds': 5.0,
    })


def _empty_result(sql: str = 'SELECT 1') -> SqlExecutionResult:
    return SqlExecutionResult( sql=sql, rows=[], column_names=[], row_count=0, latency_ms=1.0, truncated=False)


def _stub_executor(rows: List[Dict[str, Any]] = None) -> MagicMock:
    """Return a MagicMock ClickHouseExecutor that returns the given rows from execute()."""
    ex = MagicMock(spec=ClickHouseExecutor)
    ex.credentials_available = True
    ex.execute = AsyncMock(return_value=SqlExecutionResult(
        sql='SELECT 1',
        rows=rows or [],
        column_names=list((rows or [{}])[0].keys()) if rows else [],
        row_count=len(rows or []),
        latency_ms=5.0,
        truncated=False,
    ))
    ex.execute_insert = AsyncMock()
    return ex


def _make_signal(idx: int = 0) -> FeedbackSignal:
    return FeedbackSignal(
        signal_id=f'sig-{idx:04d}',
        request_id=f'req-{idx:04d}',
        signal_type='result_click',
        payload={'rank': idx},
        signal_origin='frontend',
    )


# ---------------------------------------------------------------------------
# DomainAnalyticsEngine — constructor
# ---------------------------------------------------------------------------

class TestDomainAnalyticsEngineInit:
    def test_rejects_non_executor(self):
        with pytest.raises(ConfigurationError):
            DomainAnalyticsEngine(executor="bad", config=_da_config())

    def test_rejects_non_config(self):
        ex = _stub_executor()
        with pytest.raises(ConfigurationError):
            DomainAnalyticsEngine(executor=ex, config="bad")  # type: ignore[arg-type]

    def test_rejects_non_capabilities(self):
        ex = _stub_executor()
        with pytest.raises(ConfigurationError):
            DomainAnalyticsEngine(executor=ex, config=_da_config(), capabilities="bad")  # type: ignore[arg-type]

    def test_valid_construction(self):
        engine = DomainAnalyticsEngine(executor=_stub_executor(), config=_da_config())
        assert engine._enabled()


# ---------------------------------------------------------------------------
# DomainAnalyticsEngine — disabled guard
# ---------------------------------------------------------------------------

class TestDomainAnalyticsEngineDisabled:
    @pytest.mark.parametrize("method_name,kwargs", [
        ("tld_popularity_trends", {}),
        ("price_movement", {}),
        ("auction_competitiveness", {}),
        ("expiry_pipeline_forecast", {}),
        ("realtime_auction_activity", {}),
        ("historical_sales_comparison", {
            "current_start": "2025-01-01", "current_end": "2025-01-31",
            "prior_start": "2024-01-01", "prior_end": "2024-01-31",
        }),
        ("seasonal_trends", {}),
        ("domain_quality_score", {}),
        ("domain_liquidity", {}),
        ("supply_demand_analytics", {}),
        ("keyword_trends", {}),
        ("emerging_niche_detection", {}),
        ("multidim_tld_category", {}),
        ("conversion_funnel", {}),
        ("buyer_interest_heatmap", {}),
    ])
    def test_disabled_raises(self, method_name: str, kwargs: dict):
        engine = DomainAnalyticsEngine(executor=_stub_executor(), config=_da_config(enabled=False))
        with pytest.raises(RetrievalError, match="domain_analytics.enabled=false"):
            asyncio.run(getattr(engine, method_name)(**kwargs))


# ---------------------------------------------------------------------------
# DomainAnalyticsEngine — method→template wiring
# ---------------------------------------------------------------------------

class TestDomainAnalyticsEngineMethods:
    def setup_method(self):
        self.ex = _stub_executor()
        self.engine = DomainAnalyticsEngine(executor=self.ex, config=_da_config())

    def _last_sql(self) -> str:
        return self.ex.execute.call_args[0][0]

    def test_tld_popularity_trends(self):
        asyncio.run(self.engine.tld_popularity_trends())
        sql = self._last_sql()
        assert 'dam_auction_snap' in sql
        assert 'tld' in sql
        assert 'current_price' in sql
        assert 'GROUP BY' in sql

    def test_price_movement(self):
        asyncio.run(self.engine.price_movement())
        sql = self._last_sql()
        assert 'dam_auction_snap' in sql
        assert 'current_price' in sql
        assert 'bid_count' in sql

    def test_price_movement_with_tld(self):
        asyncio.run(self.engine.price_movement(tld='.com'))
        sql = self._last_sql()
        assert ".com" in sql

    def test_auction_competitiveness(self):
        asyncio.run(self.engine.auction_competitiveness(limit=25))
        sql = self._last_sql()
        assert 'composite_score' in sql
        assert 'LIMIT 25' in sql
        assert 'bid_count' in sql

    def test_expiry_pipeline_forecast(self):
        asyncio.run(self.engine.expiry_pipeline_forecast(history_periods=14, forecast_periods=7))
        sql = self._last_sql()
        assert 'forecast_value' in sql
        assert 'tld' in sql
        assert 'bid_count' in sql

    def test_realtime_auction_activity(self):
        asyncio.run(self.engine.realtime_auction_activity(window_minutes=30))
        sql = self._last_sql()
        assert 'INTERVAL 30 MINUTE' in sql
        assert 'domain_name' in sql

    def test_historical_sales_comparison(self):
        asyncio.run(self.engine.historical_sales_comparison( current_start='2025-01-01', current_end='2025-01-31', prior_start='2024-01-01', prior_end='2024-01-31',))
        sql = self._last_sql()
        assert '2025-01-01' in sql
        assert '2024-01-01' in sql
        assert 'absolute_delta' in sql

    def test_seasonal_trends(self):
        asyncio.run(self.engine.seasonal_trends(grain=TimeGrain.MONTH))
        sql = self._last_sql()
        assert 'toStartOfMonth' in sql
        assert 'dam_auction_snap' in sql

    def test_domain_quality_score(self):
        asyncio.run(self.engine.domain_quality_score(limit=20))
        sql = self._last_sql()
        assert 'composite_score' in sql
        assert 'govalue_score' in sql
        assert 'LIMIT 20' in sql

    def test_domain_liquidity(self):
        asyncio.run(self.engine.domain_liquidity())
        sql = self._last_sql()
        assert 'growth_pct' in sql
        assert 'bid_count' in sql

    def test_supply_demand_analytics(self):
        asyncio.run(self.engine.supply_demand_analytics())
        sql = self._last_sql()
        assert 'bid_count' in sql
        assert 'current_price' in sql
        assert 'dam_auction_snap' in sql

    def test_keyword_trends(self):
        asyncio.run(self.engine.keyword_trends(min_frequency=5))
        sql = self._last_sql()
        assert 'ARRAY JOIN' in sql
        assert 'splitByChar' in sql
        assert 'HAVING frequency >= 5' in sql

    def test_emerging_niche_detection(self):
        asyncio.run(self.engine.emerging_niche_detection(z_threshold=2.0, window_periods=10))
        sql = self._last_sql()
        assert 'z_score' in sql
        assert 'is_anomaly' in sql
        assert 'tld' in sql

    def test_multidim_tld_category(self):
        asyncio.run(self.engine.multidim_tld_category())
        sql = self._last_sql()
        assert 'WITH ROLLUP' in sql
        assert 'tld' in sql
        assert 'auction_type_id' in sql

    def test_conversion_funnel(self):
        asyncio.run(self.engine.conversion_funnel())
        sql = self._last_sql()
        assert 'analytics.feedback_signals' in sql
        assert 'active_users' in sql

    def test_buyer_interest_heatmap(self):
        asyncio.run(self.engine.buyer_interest_heatmap())
        sql = self._last_sql()
        assert 'analytics.feedback_signals' in sql
        assert 'signal_type' in sql


# ---------------------------------------------------------------------------
# DomainAnalyticsEngine — executor failure propagation
# ---------------------------------------------------------------------------

class TestDomainAnalyticsEngineError:
    def test_retrieval_error_bubbles(self):
        ex = MagicMock(spec=ClickHouseExecutor)
        ex.credentials_available = True
        ex.execute = AsyncMock(side_effect=RetrievalError("CH down"))
        engine = DomainAnalyticsEngine(executor=ex, config=_da_config())
        with pytest.raises(RetrievalError):
            asyncio.run(engine.tld_popularity_trends())


# ---------------------------------------------------------------------------
# FeedbackSignalWriter — constructor
# ---------------------------------------------------------------------------

class TestFeedbackSignalWriterInit:
    def test_rejects_non_executor(self):
        with pytest.raises(ConfigurationError):
            FeedbackSignalWriter(executor="bad", config=_sw_config())  # type: ignore[arg-type]

    def test_rejects_non_config(self):
        with pytest.raises(ConfigurationError):
            FeedbackSignalWriter(executor=_stub_executor(), config="bad")  # type: ignore[arg-type]

    def test_valid_construction(self):
        writer = FeedbackSignalWriter(executor=_stub_executor(), config=_sw_config())
        assert writer.enabled


# ---------------------------------------------------------------------------
# FeedbackSignalWriter — write()
# ---------------------------------------------------------------------------

class TestFeedbackSignalWriterWrite:
    def test_write_calls_execute_insert(self):
        ex = _stub_executor()
        writer = FeedbackSignalWriter(executor=ex, config=_sw_config())
        asyncio.run(writer.write(_make_signal()))
        assert ex.execute_insert.call_count == 1

    def test_write_sql_content(self):
        ex = _stub_executor()
        writer = FeedbackSignalWriter(executor=ex, config=_sw_config())
        sig = _make_signal(42)
        asyncio.run(writer.write(sig))
        sql = ex.execute_insert.call_args[0][0]
        assert 'INSERT INTO analytics.feedback_signals' in sql
        assert 'sig-0042' in sql
        assert 'result_click' in sql
        assert 'frontend' in sql
        assert 'fromUnixTimestamp64Milli' in sql

    def test_write_disabled_noop(self):
        ex = _stub_executor()
        writer = FeedbackSignalWriter(executor=ex, config=_sw_config(enabled=False))
        asyncio.run(writer.write(_make_signal()))
        assert ex.execute_insert.call_count == 0

    def test_write_error_swallowed(self):
        ex = _stub_executor()
        ex.execute_insert = AsyncMock(side_effect=Exception("CH down"))
        writer = FeedbackSignalWriter(executor=ex, config=_sw_config())
        asyncio.run(writer.write(_make_signal()))  # must not raise

    def test_write_payload_with_single_quote_escaped(self):
        ex = _stub_executor()
        writer = FeedbackSignalWriter(executor=ex, config=_sw_config())
        sig = FeedbackSignal(
            signal_id='sig-q', request_id='req-q',
            signal_type='result_click', payload={"q": "it's"},
            signal_origin='frontend',
        )
        asyncio.run(writer.write(sig))
        sql = ex.execute_insert.call_args[0][0]
        assert "it''s" in sql


# ---------------------------------------------------------------------------
# FeedbackSignalWriter — write_batch()
# ---------------------------------------------------------------------------

class TestFeedbackSignalWriterBatch:
    def test_batch_chunks(self):
        ex = _stub_executor()
        writer = FeedbackSignalWriter(executor=ex, config=_sw_config(batch_size=3))
        sigs = [_make_signal(i) for i in range(7)]
        asyncio.run(writer.write_batch(sigs))
        # ceil(7/3) = 3 INSERT calls
        assert ex.execute_insert.call_count == 3

    def test_batch_empty_noop(self):
        ex = _stub_executor()
        writer = FeedbackSignalWriter(executor=ex, config=_sw_config())
        asyncio.run(writer.write_batch([]))
        assert ex.execute_insert.call_count == 0

    def test_batch_drops_non_signals(self):
        ex = _stub_executor()
        writer = FeedbackSignalWriter(executor=ex, config=_sw_config())
        items = [_make_signal(0), "not-a-signal", _make_signal(1)]
        asyncio.run(writer.write_batch(items))  # type: ignore[arg-type]
        # 2 valid signals in one batch
        assert ex.execute_insert.call_count == 1

    def test_batch_error_swallowed(self):
        ex = _stub_executor()
        ex.execute_insert = AsyncMock(side_effect=Exception("CH down"))
        writer = FeedbackSignalWriter(executor=ex, config=_sw_config(batch_size=2))
        sigs = [_make_signal(i) for i in range(4)]
        asyncio.run(writer.write_batch(sigs))  # must not raise


# ---------------------------------------------------------------------------
# DomainAnalyticsConfig invariants
# ---------------------------------------------------------------------------

class TestDomainAnalyticsConfig:
    def _base(self) -> dict:
        return {
            'enabled': True,
            'source_table': 'dam_auction_snap',
            'domain_column': 'domain_name',
            'tld_column': 'tld',
            'price_column': 'current_price',
            'bid_count_column': 'bid_count',
            'auction_end_column': 'ends_at',
            'score_column': 'govalue_score',
            'category_column': 'auction_type_id',
            'created_column': 'created_at',
            'signals_table': 'analytics.feedback_signals',
            'signals_batch_size': 50,
            'signals_insert_timeout_seconds': 5.0,
        }

    def test_batch_size_zero_rejected(self):
        d = self._base()
        d['signals_batch_size'] = 0
        with pytest.raises(ConfigurationError):
            DomainAnalyticsConfig.from_dict(d)

    def test_insert_timeout_zero_rejected(self):
        d = self._base()
        d['signals_insert_timeout_seconds'] = 0.0
        with pytest.raises(ConfigurationError):
            DomainAnalyticsConfig.from_dict(d)

    def test_empty_source_table_rejected(self):
        d = self._base()
        d['source_table'] = ''
        with pytest.raises(ConfigurationError):
            DomainAnalyticsConfig.from_dict(d)

    def test_missing_required_key_rejected(self):
        d = self._base()
        del d['tld_column']
        with pytest.raises(ConfigurationError):
            DomainAnalyticsConfig.from_dict(d)
