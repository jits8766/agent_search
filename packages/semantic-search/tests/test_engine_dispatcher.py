"""Tests for AnalyticsEngineDispatcher — keyword routing to DomainAnalyticsEngine methods."""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from semantic_search.analytics.engine_dispatcher import AnalyticsEngineDispatcher, _coerce_params
from semantic_search.analytics.query_templates import TimeGrain
from semantic_search.config.analytics_models import EngineDispatchConfig, EngineRouteConfig
from semantic_search.core.exceptions import RetrievalError
from semantic_search.nl_to_sql.contracts import ANALYTICS_SUBSTRATE_LABELS, AnalyticsResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _route(method: str, keywords: list, params: dict) -> EngineRouteConfig:
    return EngineRouteConfig(method=method, keywords=keywords, params=params)


def _dispatch_cfg(routes: list, enabled: bool = True, timeout: float = 3.0) -> EngineDispatchConfig:
    return EngineDispatchConfig(enabled=enabled, timeout_seconds=timeout, routes=routes)


def _stub_engine(method_name: str, return_rows: list = None) -> MagicMock:
    engine = MagicMock()
    rows = return_rows if return_rows is not None else [{'tld': 'com', 'count': 10}]
    setattr(engine, method_name, AsyncMock(return_value=rows))
    return engine


# ---------------------------------------------------------------------------
# _coerce_params
# ---------------------------------------------------------------------------

def test_coerce_params_grain_day():
    out = _coerce_params({'grain': 'day', 'lookback_periods': 30})
    assert out['grain'] == TimeGrain.DAY
    assert out['lookback_periods'] == 30


def test_coerce_params_grain_week():
    assert _coerce_params({'grain': 'week'})['grain'] == TimeGrain.WEEK


def test_coerce_params_grain_month():
    assert _coerce_params({'grain': 'month'})['grain'] == TimeGrain.MONTH


def test_coerce_params_grain_hour():
    assert _coerce_params({'grain': 'hour'})['grain'] == TimeGrain.HOUR


def test_coerce_params_unknown_grain_raises():
    with pytest.raises(ValueError, match="unknown grain value"):
        _coerce_params({'grain': 'fortnight'})


def test_coerce_params_non_grain_pass_through():
    out = _coerce_params({'lookback_days': 30, 'limit': 50})
    assert out == {'lookback_days': 30, 'limit': 50}


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_dispatcher_construction_ok():
    routes = [_route('market_summary', ['market summary'], {'lookback_days': 30})]
    engine = _stub_engine('market_summary')
    cfg = _dispatch_cfg(routes)
    dispatcher = AnalyticsEngineDispatcher(engine=engine, config=cfg)
    assert dispatcher.enabled is True


def test_dispatcher_construction_no_engine_raises():
    cfg = _dispatch_cfg([])
    with pytest.raises(ValueError, match="requires a DomainAnalyticsEngine"):
        AnalyticsEngineDispatcher(engine=None, config=cfg)


def test_dispatcher_construction_no_config_raises():
    engine = MagicMock()
    with pytest.raises(ValueError, match="requires an EngineDispatchConfig"):
        AnalyticsEngineDispatcher(engine=engine, config=None)


def test_dispatcher_construction_bad_grain_raises():
    routes = [_route('sell_through_rate', ['sell through'], {'grain': 'yearly'})]
    cfg = _dispatch_cfg(routes)
    engine = MagicMock()
    with pytest.raises(ValueError, match="bad route config"):
        AnalyticsEngineDispatcher(engine=engine, config=cfg)


# ---------------------------------------------------------------------------
# _match
# ---------------------------------------------------------------------------

def test_match_hits_sell_through_rate():
    routes = [_route('sell_through_rate', ['sell through rate', 'sold rate'], {'grain': 'day', 'lookback_periods': 30})]
    engine = _stub_engine('sell_through_rate')
    dispatcher = AnalyticsEngineDispatcher(engine=engine, config=_dispatch_cfg(routes))
    result = dispatcher._match('what is the sell through rate by tld last 30 days')
    assert result is not None
    method_name, params = result
    assert method_name == 'sell_through_rate'
    assert params['grain'] == TimeGrain.DAY


def test_match_no_match_returns_none():
    routes = [_route('market_summary', ['market summary'], {'lookback_days': 30})]
    engine = _stub_engine('market_summary')
    dispatcher = AnalyticsEngineDispatcher(engine=engine, config=_dispatch_cfg(routes))
    assert dispatcher._match('find me domains with .com ending') is None


def test_match_case_insensitive():
    routes = [_route('market_summary', ['Market Summary'], {'lookback_days': 30})]
    engine = _stub_engine('market_summary')
    dispatcher = AnalyticsEngineDispatcher(engine=engine, config=_dispatch_cfg(routes))
    assert dispatcher._match('Give me the MARKET SUMMARY for today') is not None


def test_match_first_route_wins():
    routes = [
        _route('method_a', ['trend'], {'grain': 'day', 'lookback_periods': 7}),
        _route('method_b', ['trend'], {'grain': 'week', 'lookback_periods': 4}),
    ]
    engine = MagicMock()
    setattr(engine, 'method_a', AsyncMock(return_value=[]))
    setattr(engine, 'method_b', AsyncMock(return_value=[]))
    dispatcher = AnalyticsEngineDispatcher(engine=engine, config=_dispatch_cfg(routes))
    result = dispatcher._match('show trend over last week')
    assert result[0] == 'method_a'


# ---------------------------------------------------------------------------
# try_dispatch — success
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dispatch_success_returns_analytics_result():
    rows = [{'tld': 'com', 'count': 42}, {'tld': 'net', 'count': 10}]
    routes = [_route('sell_through_rate', ['sell through rate'], {'grain': 'day', 'lookback_periods': 30, 'limit': 50})]
    engine = _stub_engine('sell_through_rate', return_rows=rows)
    dispatcher = AnalyticsEngineDispatcher(engine=engine, config=_dispatch_cfg(routes))

    result = await dispatcher.try_dispatch(
        question='what is the sell through rate by tld',
        sql_hint='',
        request_id='test-rid',
        t0=time.monotonic(),
    )

    assert result is not None
    assert isinstance(result, AnalyticsResult)
    assert result.success is True
    assert result.analytics_substrate == 'domain_engine'
    assert result.analytics_substrate in ANALYTICS_SUBSTRATE_LABELS
    assert result.execution is not None
    assert result.execution.rows == rows
    assert result.execution.row_count == 2
    assert result.execution.sql == '-- domain_engine:sell_through_rate'
    assert result.execution.column_names == ['tld', 'count']
    assert result.failure_mode is None


@pytest.mark.asyncio
async def test_dispatch_success_empty_rows():
    routes = [_route('market_summary', ['market summary'], {'lookback_days': 30})]
    engine = _stub_engine('market_summary', return_rows=[])
    dispatcher = AnalyticsEngineDispatcher(engine=engine, config=_dispatch_cfg(routes))

    result = await dispatcher.try_dispatch('market summary stats', '', 'rid', time.monotonic())
    assert result is not None
    assert result.success is True
    assert result.execution.rows == []
    assert result.execution.row_count == 0
    assert result.execution.column_names == []


# ---------------------------------------------------------------------------
# try_dispatch — fallthrough cases (all return None)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dispatch_disabled_returns_none():
    routes = [_route('market_summary', ['market summary'], {'lookback_days': 30})]
    engine = _stub_engine('market_summary')
    dispatcher = AnalyticsEngineDispatcher(engine=engine, config=_dispatch_cfg(routes, enabled=False))
    result = await dispatcher.try_dispatch('market summary', '', 'rid', time.monotonic())
    assert result is None


@pytest.mark.asyncio
async def test_dispatch_no_match_returns_none():
    routes = [_route('market_summary', ['market summary'], {'lookback_days': 30})]
    engine = _stub_engine('market_summary')
    dispatcher = AnalyticsEngineDispatcher(engine=engine, config=_dispatch_cfg(routes))
    result = await dispatcher.try_dispatch('find cheap .com domains', '', 'rid', time.monotonic())
    assert result is None


@pytest.mark.asyncio
async def test_dispatch_timeout_returns_none():
    async def slow_method(**kwargs):
        await asyncio.sleep(10)
        return []

    routes = [_route('sell_through_rate', ['sell through'], {'grain': 'day', 'lookback_periods': 30, 'limit': 50})]
    engine = MagicMock()
    engine.sell_through_rate = slow_method
    dispatcher = AnalyticsEngineDispatcher(engine=engine, config=_dispatch_cfg(routes, timeout=0.01))

    result = await dispatcher.try_dispatch('sell through rate last month', '', 'rid', time.monotonic())
    assert result is None


@pytest.mark.asyncio
async def test_dispatch_retrieval_error_returns_none():
    routes = [_route('sell_through_rate', ['sell through'], {'grain': 'day', 'lookback_periods': 30, 'limit': 50})]
    engine = MagicMock()
    engine.sell_through_rate = AsyncMock(side_effect=RetrievalError("CH unavailable"))
    dispatcher = AnalyticsEngineDispatcher(engine=engine, config=_dispatch_cfg(routes))

    result = await dispatcher.try_dispatch('sell through rate last week', '', 'rid', time.monotonic())
    assert result is None


@pytest.mark.asyncio
async def test_dispatch_generic_exception_returns_none():
    routes = [_route('bid_analytics', ['bid analytics'], {'grain': 'day', 'lookback_periods': 30})]
    engine = MagicMock()
    engine.bid_analytics = AsyncMock(side_effect=RuntimeError("unexpected failure"))
    dispatcher = AnalyticsEngineDispatcher(engine=engine, config=_dispatch_cfg(routes))

    result = await dispatcher.try_dispatch('bid analytics this week', '', 'rid', time.monotonic())
    assert result is None


@pytest.mark.asyncio
async def test_dispatch_unknown_method_name_returns_none():
    routes = [_route('nonexistent_method', ['magic phrase'], {})]
    engine = MagicMock(spec=[])  # no attributes
    dispatcher = AnalyticsEngineDispatcher(engine=engine, config=_dispatch_cfg(routes))

    result = await dispatcher.try_dispatch('magic phrase in query', '', 'rid', time.monotonic())
    assert result is None
