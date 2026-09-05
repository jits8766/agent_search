"""Unit tests for the analytics path components.

Coverage matrix preserved as `pytest.param(id=...)` ids — recoverable via
`pytest --collect-only -q`.
"""
import asyncio
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from semantic_search.analytics.clickhouse_client import ClickHouseClient, ClickHouseQueryError, ClickHouseUnavailableError
from semantic_search.analytics.clickhouse_executor import ClickHouseExecutor
from semantic_search.analytics.mv_router import MVRewriteDecision, MVRouter
from semantic_search.analytics.pipeline_router import AnalyticsRouter
from semantic_search.analytics.nl_sql_exact_cache import CachedSqlTemplate, NLSqlExactCache, canonicalize_question
from semantic_search.config.analytics_models import AnalyticsConfig, ClickHouseClientConfig, ClickHouseMvFreshnessProbeConfig, MVRouterConfig, MaterializedViewConfig, NLSqlExactCacheConfig
from semantic_search.config.models import BackendHealthConfig
from semantic_search.config.nl_to_sql_models import SqlExecutionConfig
from semantic_search.core.exceptions import CacheError, ConfigurationError, RetrievalError
from semantic_search.nl_to_sql.contracts import AnalyticsResult, SqlExecutionResult, SqlValidationResult, VerifierVerdict
from semantic_search.resilience.health import BackendHealthRegistry
from semantic_search.nl_to_sql.pipeline import GenAndValidate


# ---------------------------------------------------------------------------
# Config dict factories
# ---------------------------------------------------------------------------

def _ch_client_dict(**overrides: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = dict(
        host='clickhouse.local', port=8123, database='analytics', user='default',
        secure=False, connect_timeout_seconds=2.0, read_timeout_seconds=10.0,
        max_retries=1, execution_timeout_seconds=30.0,
        mv_freshness_probe={
            'enabled': False,
            'interval_successful_queries': 10,
            'lag_sql': 'SELECT 0 AS lag_seconds',
            'max_lag_seconds': 3600.0,
            'query_timeout_seconds': 2.0,
        },
        readiness_probe={
            'enabled': False,
            'sql': 'SELECT 1',
            'timeout_seconds': 2.0,
            'warm_timeout_seconds': 15.0,
        },
    )
    base.update(overrides)
    return base


def _mv_dict(**overrides: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = dict(
        name='analytics.mv_by_tld',
        source_table='events_raw',
        grain_columns=['tld', 'event_day'],
        aggregate_columns={'avg_price_state': 'price', 'count_state': '*'},
        time_column='event_day',
        freshness_lag_seconds=3.0,
    )
    base.update(overrides)
    return base


def _exact_cache_dict(**overrides: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = dict(
        enabled=True, ttl_seconds=300, max_entries=128,
        min_verified_count=1,
        remote={'enabled': False, 'redis_url_env_var': 'TEST_NL_SQL_REDIS', 'key_prefix': 't:', 'socket_timeout_seconds': 1.0},
    )
    base.update(overrides)
    return base


def _analytics_dict(**overrides: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = dict(
        enabled=True, freshness_target_seconds=5.0,
        clickhouse=_ch_client_dict(),
        mv_router={'enabled': True, 'materialized_views': [_mv_dict()]},
        exact_cache=_exact_cache_dict(),
        clickhouse_on_miss_enabled=True,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Stub transports + client fixtures
# ---------------------------------------------------------------------------

async def _stub_transport(query: str, timeout: float) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Stub transport returning a single deterministic row."""
    return ([{'tld': 'com', 'avg_price': 12.34}], ['tld', 'avg_price'])


@pytest.fixture
def stub_ch_client() -> ClickHouseClient:
    """A `ClickHouseClient` wired to the stub transport above."""
    cfg = ClickHouseClientConfig.from_dict(_ch_client_dict())
    return ClickHouseClient(cfg, transport=_stub_transport)


# ---------------------------------------------------------------------------
# Config invariants
# ---------------------------------------------------------------------------

class TestClickHouseClientConfig:
    def test_happy_path(self) -> None:
        cfg = ClickHouseClientConfig.from_dict(_ch_client_dict())
        assert cfg.host == 'clickhouse.local'
        assert cfg.port == 8123

    @pytest.mark.parametrize('field,value', [
        ('port', 0),
        ('port', 70000),
        ('connect_timeout_seconds', 0.0),
        ('read_timeout_seconds', -1.0),
        ('max_retries', -1),
        ('database', ''),
    ])
    def test_invariants(self, field: str, value: Any) -> None:
        with pytest.raises(ConfigurationError):
            ClickHouseClientConfig.from_dict(_ch_client_dict(**{field: value}))


class TestClickHouseMvFreshnessProbeConfig:
    def test_disabled_allows_zero_interval(self) -> None:
        p = ClickHouseMvFreshnessProbeConfig.from_dict( {'enabled': False, 'interval_successful_queries': 0, 'lag_sql': '', 'max_lag_seconds': 1.0, 'query_timeout_seconds': 1.0},)
        assert p.enabled is False

    def test_enabled_requires_positive_interval(self) -> None:
        with pytest.raises(ConfigurationError, match='interval_successful_queries'):
            ClickHouseMvFreshnessProbeConfig.from_dict( {'enabled': True, 'interval_successful_queries': 0, 'lag_sql': 'SELECT 1', 'max_lag_seconds': 1.0, 'query_timeout_seconds': 1.0},)


class TestClickHouseMvFreshnessProbeRuntime:
    @pytest.mark.asyncio
    async def test_lag_breach_records_health_failure(self) -> None:
        calls = {'n': 0}

        async def transport(query: str, timeout: float) -> Tuple[List[Dict[str, Any]], List[str]]:
            calls['n'] += 1
            if calls['n'] == 1:
                return ([{'v': 1}], ['v'])
            return ([{'lag_seconds': 999.0}], ['lag_seconds'])

        cfg = ClickHouseClientConfig.from_dict(
            _ch_client_dict(
                mv_freshness_probe={
                    'enabled': True,
                    'interval_successful_queries': 1,
                    'lag_sql': 'SELECT 999 AS lag_seconds',
                    'max_lag_seconds': 10.0,
                    'query_timeout_seconds': 2.0,
                },
            ),
        )
        reg = BackendHealthRegistry(
        BackendHealthConfig( enabled=True, failure_rate_threshold=0.5, rolling_window_seconds=300.0, min_observations=1, recovery_probe_seconds=0.0, force_unhealthy_backends=[],)
        )
        client = ClickHouseClient(cfg, transport=transport, health_registry=reg)
        await client.execute_query('SELECT 1', 2.0)
        assert reg.state('clickhouse') in ('degraded', 'unhealthy')


class TestMaterializedViewConfig:
    def test_happy_path(self) -> None:
        mv = MaterializedViewConfig.from_dict(_mv_dict())
        assert mv.name == 'analytics.mv_by_tld'
        assert mv.grain_columns == ['tld', 'event_day']
        assert mv.aggregate_columns['avg_price_state'] == 'price'

    @pytest.mark.parametrize('overrides', [
        pytest.param({'name': 'mv_by_tld'},      id='unqualified_name'),
        pytest.param({'grain_columns': []},      id='empty_grain'),
        pytest.param({'aggregate_columns': {}},  id='empty_agg_columns'),
    ])
    def test_invariants(self, overrides: Dict[str, Any]) -> None:
        with pytest.raises(ConfigurationError):
            MaterializedViewConfig.from_dict(_mv_dict(**overrides))


class TestMVRouterConfig:
    def test_empty_catalog_allowed(self) -> None:
        cfg = MVRouterConfig.from_dict({'enabled': True, 'materialized_views': []})
        assert cfg.materialized_views == []

    def test_duplicate_mv_names_rejected(self) -> None:
        with pytest.raises(ConfigurationError):
            MVRouterConfig.from_dict({
                'enabled': True,
                'materialized_views': [_mv_dict(), _mv_dict()],
            })


class TestNLSqlExactCacheConfig:
    def test_happy_path(self) -> None:
        cfg = NLSqlExactCacheConfig.from_dict(_exact_cache_dict())
        assert cfg.enabled is True
        assert cfg.min_verified_count == 1

    @pytest.mark.parametrize('field,value', [
        ('min_verified_count', -1),
        ('ttl_seconds', 0),
        ('max_entries', 0),
    ])
    def test_invariants(self, field: str, value: Any) -> None:
        with pytest.raises(ConfigurationError):
            NLSqlExactCacheConfig.from_dict(_exact_cache_dict(**{field: value}))


class TestAnalyticsConfig:
    def test_full_bundle(self) -> None:
        cfg = AnalyticsConfig.from_dict(_analytics_dict())
        assert cfg.enabled is True
        assert cfg.clickhouse.host == 'clickhouse.local'
        assert cfg.mv_router.materialized_views[0].name == 'analytics.mv_by_tld'

    def test_freshness_target_must_be_positive(self) -> None:
        with pytest.raises(ConfigurationError):
            AnalyticsConfig.from_dict(_analytics_dict(freshness_target_seconds=0.0))


# ---------------------------------------------------------------------------
# ClickHouse client + executor
# ---------------------------------------------------------------------------

class TestClickHouseClient:
    def test_construction_with_stub_transport(self, stub_ch_client: ClickHouseClient) -> None:
        assert stub_ch_client.available is True
        assert stub_ch_client.database == 'analytics'

    def test_execute_via_stub(self, stub_ch_client: ClickHouseClient) -> None:
        rows, cols, latency = asyncio.run(stub_ch_client.execute_query('SELECT 1', timeout_seconds=2.0))
        assert rows == [{'tld': 'com', 'avg_price': 12.34}]
        assert cols == ['tld', 'avg_price']
        assert latency >= 0.0

    @pytest.mark.parametrize('sql,timeout', [
        pytest.param('   ',      2.0, id='empty_sql'),
        pytest.param('SELECT 1', 0.0, id='non_positive_timeout'),
    ])
    def test_execute_rejects_invalid( self, stub_ch_client: ClickHouseClient, sql: str, timeout: float, ) -> None:
        with pytest.raises(RetrievalError):
            asyncio.run(stub_ch_client.execute_query(sql, timeout_seconds=timeout))

    def test_unavailable_when_transport_missing_and_httpx_init_skipped( self, stub_ch_client: ClickHouseClient, ) -> None:
        # Forcing unavailable: simulate no working transport at execute time.
        stub_ch_client._available = False  # type: ignore[attr-defined]
        with pytest.raises(ClickHouseUnavailableError):
            asyncio.run(stub_ch_client.execute_query('SELECT 1', timeout_seconds=2.0))


class TestClickHouseClientPostQueryParams:
    """Verify that _post_query sends the correct HTTP params per statement type.

    CREATE DATABASE DDL must omit the `database` param so ClickHouse accepts
    it before the target database exists (the 404-on-schema-init bug).
    All other statements must include `database=<config.database>`.
    """

    def _make_client(self, mock_post):
        cfg = ClickHouseClientConfig.from_dict(_ch_client_dict())
        client = ClickHouseClient(cfg)
        # Replace the internal httpx AsyncClient with a mock whose .post is controlled.
        mock_async_client = mock_post
        client._client = mock_async_client
        client._available = True
        return client

    @pytest.mark.asyncio
    @pytest.mark.parametrize('sql,expect_database_param', [
        pytest.param('CREATE DATABASE IF NOT EXISTS analytics', False, id='create_database_omits_param'),
        pytest.param('create database analytics', False, id='create_database_lowercase_omits_param'),
        pytest.param('  CREATE DATABASE foo', False, id='create_database_leading_whitespace_omits_param'),
        pytest.param('SELECT 1', True, id='select_includes_param'),
        pytest.param('INSERT INTO analytics.t FORMAT JSONEachRow\n{}', True, id='insert_includes_param'),
        pytest.param('CREATE TABLE IF NOT EXISTS analytics.t (id Int64) ENGINE=Memory', True, id='create_table_includes_param'),
    ])
    async def test_database_param_presence(self, sql: str, expect_database_param: bool) -> None:
        captured = {}

        async def fake_post(path, *, params=None, content=None, headers=None):
            captured['params'] = dict(params or {})
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = ''
            return mock_response

        cfg = ClickHouseClientConfig.from_dict(_ch_client_dict())
        client = ClickHouseClient(cfg)
        client._available = True

        mock_http = MagicMock()
        mock_http.post = fake_post
        client._client = mock_http

        await client.execute_query(sql, timeout_seconds=5.0, _skip_mv_freshness_probe=True)

        if expect_database_param:
            assert 'database' in captured['params'], f"Expected 'database' in params for: {sql!r}"
            assert captured['params']['database'] == 'analytics'
        else:
            assert 'database' not in captured['params'], f"Expected no 'database' param for DDL: {sql!r}"


class TestClickHouseClientHealthWiring:
    """ClickHouseClient records every call outcome on the
    injected `BackendHealthRegistry` (`backend='clickhouse'`).
    """

    def _registry(self):
        cfg = BackendHealthConfig(
            enabled=True,
            failure_rate_threshold=0.5,
            rolling_window_seconds=60.0,
            min_observations=2,
            recovery_probe_seconds=0.0,
            force_unhealthy_backends=[],
        )
        return BackendHealthRegistry(cfg)

    def test_success_records_clickhouse_outcome(self) -> None:
        reg = self._registry()
        cfg = ClickHouseClientConfig.from_dict(_ch_client_dict())
        client = ClickHouseClient(cfg, transport=_stub_transport, health_registry=reg)
        # Successful query -> True outcome on `clickhouse`.
        asyncio.run(client.execute_query('SELECT 1', timeout_seconds=2.0))
        snap = {b.backend: b for b in reg.snapshot()}
        ch = snap['clickhouse']
        assert ch.observation_count == 1
        assert ch.failure_rate == 0.0

    def test_failure_records_clickhouse_outcome(self) -> None:
        reg = self._registry()
        cfg = ClickHouseClientConfig.from_dict(_ch_client_dict())

        async def failing_transport(query: str, timeout: float):
            raise ClickHouseQueryError("synthetic CH 500")

        client = ClickHouseClient(cfg, transport=failing_transport, health_registry=reg)
        with pytest.raises(ClickHouseQueryError):
            asyncio.run(client.execute_query('SELECT 1', timeout_seconds=2.0))
        snap = {b.backend: b for b in reg.snapshot()}
        ch = snap['clickhouse']
        assert ch.observation_count == 1
        assert ch.failure_rate == 1.0

    def test_no_registry_is_safe_noop(self, stub_ch_client: ClickHouseClient) -> None:
        # Default construction without a registry must not raise on the
        # internal record path — the helper short-circuits when None.
        rows, cols, _ = asyncio.run(stub_ch_client.execute_query('SELECT 1', timeout_seconds=2.0))
        assert rows[0]['tld'] == 'com'


class TestClickHouseExecutor:
    def _executor(self, max_rows: int = 100) -> ClickHouseExecutor:
        ch_cfg = ClickHouseClientConfig.from_dict(_ch_client_dict())
        client = ClickHouseClient(ch_cfg, transport=_stub_transport)
        exec_cfg = SqlExecutionConfig.from_dict({'max_rows': max_rows, 'timeout_seconds': 2.0})
        return ClickHouseExecutor(exec_cfg, client)

    def test_execute_returns_typed_result(self) -> None:
        executor = self._executor()
        result = asyncio.run(executor.execute('SELECT 1'))
        assert result.row_count == 1
        assert result.column_names == ['tld', 'avg_price']
        assert result.truncated is False

    def test_truncation_when_over_max_rows(self) -> None:
        async def big_transport(q: str, t: float) -> Tuple[List[Dict[str, Any]], List[str]]:
            rows = [{'tld': str(i)} for i in range(50)]
            return rows, ['tld']

        ch_cfg = ClickHouseClientConfig.from_dict(_ch_client_dict())
        client = ClickHouseClient(ch_cfg, transport=big_transport)
        exec_cfg = SqlExecutionConfig.from_dict({'max_rows': 5, 'timeout_seconds': 2.0})
        executor = ClickHouseExecutor(exec_cfg, client)
        result = asyncio.run(executor.execute('SELECT tld FROM x'))
        assert result.row_count == 5
        assert result.truncated is True


# ---------------------------------------------------------------------------
# MV router
# ---------------------------------------------------------------------------

class TestMVRouter:
    def _router(self, mv_dicts: List[Dict[str, Any]], enabled: bool = True) -> MVRouter:
        cfg = MVRouterConfig.from_dict({'enabled': enabled, 'materialized_views': mv_dicts})
        return MVRouter(cfg, dialect='clickhouse')

    def test_disabled_router_passthrough(self) -> None:
        router = self._router([_mv_dict()], enabled=False)
        decision = router.route('SELECT tld, avg(price) FROM events_raw GROUP BY tld')
        assert decision.matched is False
        assert decision.reason == 'router_disabled_or_empty_catalog'

    def test_empty_catalog_passthrough(self) -> None:
        router = self._router([], enabled=True)
        decision = router.route('SELECT tld, avg(price) FROM events_raw GROUP BY tld')
        assert decision.matched is False

    def test_match_simple_avg(self) -> None:
        router = self._router([_mv_dict()])
        decision = router.route('SELECT tld, avg(price) FROM events_raw GROUP BY tld')
        assert decision.matched is True
        assert decision.mv_name == 'analytics.mv_by_tld'
        assert 'avgmerge' in decision.rewritten_sql.lower()
        assert 'avg_price_state' in decision.rewritten_sql.lower()
        assert decision.freshness_lag_seconds == 3.0

    @pytest.mark.parametrize('sql', [
        pytest.param( 'SELECT tld, avg(price) FROM other_table GROUP BY tld', id='source_table_differs',),
        pytest.param( 'SELECT auction_type, avg(price) FROM events_raw GROUP BY auction_type', id='group_by_outside_grain',),
        pytest.param( 'SELECT tld, avg(price) FROM events_raw WHERE bids > 5 GROUP BY tld', id='where_on_non_grain_column',),
        pytest.param( 'SELECT * FROM events_raw a JOIN x ON a.id = x.id', id='unsupported_query_shape',),
    ])
    def test_no_match_paths(self, sql: str) -> None:
        router = self._router([_mv_dict()])
        decision = router.route(sql)
        assert decision.matched is False

    def test_tie_break_picks_smaller_grain(self) -> None:
        small = _mv_dict(name='analytics.mv_small', grain_columns=['tld'])
        big = _mv_dict(name='analytics.mv_big', grain_columns=['tld', 'event_day', 'auction_type'])
        router = self._router([big, small])
        decision = router.route('SELECT tld, avg(price) FROM events_raw GROUP BY tld')
        assert decision.matched is True
        assert decision.mv_name == 'analytics.mv_small'


# ---------------------------------------------------------------------------
# Exact NL-to-SQL cache
# ---------------------------------------------------------------------------

def _build_cache(min_verified: int = 1) -> NLSqlExactCache:
    cfg = NLSqlExactCacheConfig.from_dict(_exact_cache_dict(min_verified_count=min_verified))
    return NLSqlExactCache(cfg)


class TestExactCache:
    def test_canonicalize_question_strips_punct_and_whitespace(self) -> None:
        assert canonicalize_question("  Average  PRICE for .COM??  ") == "average price for com"
        assert canonicalize_question("") == ""
        assert canonicalize_question(None) == ""  # type: ignore[arg-type]

    def test_lookup_empty_cache_returns_miss(self) -> None:
        cache = _build_cache()
        result = cache.lookup("anything")
        assert result.hit is False
        assert result.entry is None

    def test_upsert_and_verified_lookup(self) -> None:
        cache = _build_cache(min_verified=1)
        cache.upsert(
            question="average price for com",
            sql_template="SELECT avg(price) FROM events_raw WHERE tld='com'",
            mv_used='', verifier_passed=True,
        )
        result = cache.lookup("average price for com")
        assert result.hit is True
        assert result.verified is True
        assert result.entry is not None
        assert result.entry.verified_count == 1

    def test_lookup_is_case_insensitive(self) -> None:
        cache = _build_cache(min_verified=1)
        cache.upsert(
            question="Average Price For COM",
            sql_template="SELECT avg(price) FROM events_raw WHERE tld='com'",
            mv_used='', verifier_passed=True,
        )
        result = cache.lookup("average  price for com")
        assert result.hit is True
        assert result.verified is True

    def test_numerically_distinct_questions_do_not_share_entry(self) -> None:
        cache = _build_cache(min_verified=1)
        cache.upsert(
            question="find .net domains under $100",
            sql_template="SELECT domain FROM events_raw WHERE tld='net' AND price < 100",
            mv_used='', verifier_passed=True,
        )
        # A different price must MISS — the embedding-similarity conflation is gone.
        result = cache.lookup("find .net domains under $200")
        assert result.hit is False
        assert result.entry is None
        assert cache.lookup("find .net domains under $100").hit is True

    def test_unverified_hit_when_below_min_verified_count(self) -> None:
        cache = _build_cache(min_verified=3)
        cache.upsert(
            question="average price for com",
            sql_template="SELECT avg(price) FROM events_raw WHERE tld='com'",
            mv_used='', verifier_passed=True,
        )
        result = cache.lookup("average price for com")
        assert result.hit is True
        assert result.verified is False  # only 1 verifier-pass, below min=3

    def test_repeated_upsert_increments_verified_count(self) -> None:
        cache = _build_cache()
        for _ in range(3):
            cache.upsert(
                question="how many .com listings today",
                sql_template="SELECT count() FROM events_raw WHERE tld='com'",
                mv_used='', verifier_passed=True,
            )
        result = cache.lookup("how many .com listings today")
        assert result.entry is not None
        assert result.entry.verified_count == 3

    def test_disabled_cache_is_noop(self) -> None:
        cfg = NLSqlExactCacheConfig.from_dict(_exact_cache_dict(enabled=False))
        cache = NLSqlExactCache(cfg)
        cache.upsert("x", "SELECT 1", '', True)
        assert cache.size == 0
        assert cache.lookup("x").hit is False

    def test_requires_typed_config(self) -> None:
        with pytest.raises(CacheError):
            NLSqlExactCache(object())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# AnalyticsRouter end-to-end (cache miss -> pipeline fallback path)
# ---------------------------------------------------------------------------
# We exercise the AnalyticsRouter against a stubbed NLToSQLPipeline + verifier
# + executor so the test verifies the orchestration logic without touching the
# real LLM router or ClickHouse.

class _StubPipeline:
    """Minimal NLToSQLPipeline stand-in for AnalyticsRouter unit tests.

    ``gv_factory`` is optional. When provided, the stub also implements
    ``generate_and_validate`` so tests can exercise the ClickHouse-on-miss
    code path (which calls the new pipeline method). When omitted, the
    method is missing — letting tests verify the AttributeError fallback
    that defends pre-refactor pipelines.
    """

    def __init__(self, result_factory, gv_factory=None) -> None:
        self._factory = result_factory
        self._gv_factory = gv_factory
        self.calls: List[str] = []
        self.gv_calls: List[str] = []

    async def run(self, question: str, sql_hint: str = "", request_id=None) -> AnalyticsResult:
        self.calls.append(question)
        return self._factory(question, sql_hint, request_id)

    async def generate_and_validate(self, question: str, sql_hint: str, request_id: str, database_override: Optional[str] = None):
        if self._gv_factory is None:
            raise AttributeError("generate_and_validate not configured on this stub")
        self.gv_calls.append(question)
        return self._gv_factory(question, sql_hint, request_id)


class _StubVerifier:
    """Verifier stand-in returning a pre-canned verdict."""

    def __init__(self, sufficient: bool) -> None:
        self._sufficient = sufficient

    async def verify(self, question: str, execution: SqlExecutionResult, request_id: str = "") -> VerifierVerdict:
        if self._sufficient:
            return VerifierVerdict( sufficient=True, failure_mode='ok', confidence=0.99, model='stub', latency_ms=1.0, notes='stub-pass',)
        return VerifierVerdict( sufficient=False, failure_mode='empty_result', confidence=0.99, model='stub', latency_ms=1.0, notes='stub-fail',)


class _StubSecurityValidator:
    """Always-pass security validator stand-in."""

    def validate(self, sql: str) -> SqlValidationResult:
        return SqlValidationResult( is_valid=True, sql=sql, failure_mode=None, failure_reasons=[], mutations=[], estimated_rows=None, latency_ms=0.5,)


def _stub_executor() -> ClickHouseExecutor:
    async def transport(q: str, t: float) -> Tuple[List[Dict[str, Any]], List[str]]:
        return ([{'tld': 'com', 'avg_price': 9.99}], ['tld', 'avg_price'])

    cfg = ClickHouseClientConfig.from_dict(_ch_client_dict())
    client = ClickHouseClient(cfg, transport=transport)
    return ClickHouseExecutor(SqlExecutionConfig.from_dict({'max_rows': 100, 'timeout_seconds': 2.0}), client)


def _make_router(
    pipeline_factory,
    *,
    verifier_pass: bool = True,
    cache_min_verified: int = 1,
    mv_enabled: bool = True,
    analytics_enabled: bool = True,
    gv_factory=None,
    clickhouse_on_miss_enabled: bool = True,
    ch_executor=None,
) -> Tuple[AnalyticsRouter, _StubPipeline, NLSqlExactCache]:
    a_cfg = AnalyticsConfig.from_dict(_analytics_dict(
        enabled=analytics_enabled,
        exact_cache=_exact_cache_dict(min_verified_count=cache_min_verified),
        mv_router={'enabled': mv_enabled, 'materialized_views': [_mv_dict()]},
        clickhouse_on_miss_enabled=clickhouse_on_miss_enabled,
    ))
    cache = NLSqlExactCache(a_cfg.exact_cache)
    mv_router = MVRouter(a_cfg.mv_router, dialect='clickhouse')
    pipeline = _StubPipeline(pipeline_factory, gv_factory=gv_factory)
    router = AnalyticsRouter(
        config=a_cfg, pipeline=pipeline, cache=cache, mv_router=mv_router,
        security_validator=_StubSecurityValidator(),
        ch_executor=ch_executor or _stub_executor(),
        verifier=_StubVerifier(sufficient=verifier_pass),
    )
    return router, pipeline, cache


def _success_result(
    q: str, hint: str, rid,
    *,
    sql: str = 'SELECT tld, avg(price) FROM events_raw GROUP BY tld',
    rows: List[Dict[str, Any]] = None,
    column_names: List[str] = None,
) -> AnalyticsResult:
    """Build a successful `AnalyticsResult` — used as the pipeline factory return."""
    rows = rows if rows is not None else [{'tld': 'com', 'avg_price': 9.99}]
    column_names = column_names if column_names is not None else ['tld', 'avg_price']
    execution = SqlExecutionResult( sql=sql, rows=rows, column_names=column_names, row_count=len(rows), latency_ms=10.0, truncated=False,)
    verdict = VerifierVerdict( sufficient=True, failure_mode='ok', confidence=0.99, model='stub', latency_ms=1.0, notes='ok',)
    return AnalyticsResult(
        request_id=rid or AnalyticsResult.new_request_id(),
        question=q, sql_hint=hint, success=True, failure_mode=None,
        failure_reason="", pruned_schema=None, generation=None,
        validation=None, execution=execution, verifier=verdict,
        total_latency_ms=15.0,
    )


def _seed_verified_template(cache: NLSqlExactCache, *, question: str, sql: str) -> None:
    """Pre-seed the exact cache with a verified template for fast-path tests."""
    cache.upsert(question=question, sql_template=sql, mv_used='', verifier_passed=True)


def _unused_pipeline_factory(q: str, hint: str, rid) -> AnalyticsResult:
    """Pipeline factory that fails the test if the pipeline is invoked."""
    raise AssertionError("pipeline must NOT be called on a verified cache hit")


class TestAnalyticsRouter:
    def test_cache_miss_falls_back_to_pipeline_and_upserts_on_success(self) -> None:
        router, pipeline, cache = _make_router(_success_result, gv_factory=_gv_factory_success())
        result = asyncio.run(router.run(question="average price for com"))
        assert result.success is True
        assert pipeline.gv_calls == ["average price for com"]
        # The pipeline-success path should have upserted the cache.
        assert cache.size == 1

    def test_verified_cache_hit_serves_fast_path(self) -> None:
        router, pipeline, cache = _make_router( _unused_pipeline_factory, cache_min_verified=1, mv_enabled=False,)
        _seed_verified_template(
            cache,
            question="average price for com",
            sql="SELECT tld, avg(price) FROM events_raw GROUP BY tld",
        )
        result = asyncio.run(router.run(question="average price for com"))
        assert result.success is True
        assert result.execution is not None
        assert pipeline.calls == []  # fast path served it

    def test_fast_path_verifier_rejection_returns_failure(self) -> None:
        router, _pipeline, cache = _make_router( _unused_pipeline_factory, verifier_pass=False, cache_min_verified=1, mv_enabled=False,)
        _seed_verified_template(
            cache,
            question="average price for com",
            sql="SELECT tld, avg(price) FROM events_raw GROUP BY tld",
        )
        result = asyncio.run(router.run(question="average price for com"))
        assert result.success is False
        assert result.failure_mode == 'verifier'

    def test_disabled_router_delegates_to_pipeline(self) -> None:
        # analytics.enabled=False -> straight delegation to pipeline.
        router, pipeline, _cache = _make_router(_success_result, analytics_enabled=False)
        assert router.enabled is False
        result = asyncio.run(router.run(question="x"))
        assert result.success is True
        assert pipeline.calls == ["x"]

    def test_empty_question_returns_typed_failure(self) -> None:
        router, pipeline, _cache = _make_router(lambda q, h, r: None)  # type: ignore[arg-type]
        result = asyncio.run(router.run(question="   "))
        assert result.success is False
        assert result.failure_mode == 'generation'
        assert pipeline.calls == []


class TestAnalyticsRouterFreshness:
    """Per-path as_of derivation by ``AnalyticsRouter._freshness_lag_for``."""

    @pytest.mark.parametrize('mv_used,expected_lag', [
        # Known MV in the catalog uses the MV's own freshness_lag_seconds.
        pytest.param('analytics.mv_by_tld', 3.0, id='known_mv_uses_mv_lag'),
        # Unknown MV name falls back to analytics.freshness_target_seconds.
        pytest.param('not_in_catalog', 5.0, id='unknown_mv_uses_freshness_target'),
        # Empty mv_used (raw-table query) also falls back to freshness_target.
        pytest.param('', 5.0, id='raw_table_uses_freshness_target'),
    ])
    def test_freshness_lag_resolution(self, mv_used: str, expected_lag: float) -> None:
        router, _pipe, _cache = _make_router(lambda q, h, r: None, mv_enabled=True)
        assert router._freshness_lag_for(mv_used) == expected_lag

    def test_fast_path_success_carries_as_of(self) -> None:
        # Verified cache hit -> fast path -> success result must carry as_of.
        router, _pipe, cache = _make_router( _unused_pipeline_factory, cache_min_verified=1, mv_enabled=False,)
        _seed_verified_template(
            cache,
            question="average price for com",
            sql="SELECT tld, avg(price) FROM events_raw GROUP BY tld",
        )
        result = asyncio.run(router.run(question="average price for com"))
        assert result.success is True
        assert result.as_of is not None
        assert result.freshness_lag_seconds is not None
        # Raw-table fast path uses freshness_target_seconds from analytics config
        assert result.freshness_lag_seconds == pytest.approx(5.0, abs=1.0)


# ---------------------------------------------------------------------------
# AnalyticsRouter cache-miss -> ClickHouse-on-miss path
# ---------------------------------------------------------------------------
# When the semantic cache misses, the router should prefer ClickHouse (with
# optional MV rewrite) over the legacy Athena pipeline. Tests stub
# `_StubPipeline.generate_and_validate` so the new path can run end-to-end
# without a real LLM router.

def _gv_factory_success(sql: str = "SELECT tld, avg(price) FROM events_raw GROUP BY tld"):
    """Build a ``GenAndValidate`` factory that succeeds with the given SQL."""
    def factory(question: str, sql_hint: str, request_id: str) -> GenAndValidate:
        validation = SqlValidationResult(is_valid=True, sql=sql, failure_mode=None, failure_reasons=[], mutations=[], estimated_rows=None, latency_ms=0.5)
        return GenAndValidate(success=True, pruned=None, generation=None, validation=validation)
    return factory


def _gv_factory_failure(failure_mode: str = 'security'):
    """Factory that returns a deterministic gen+validate failure."""
    def factory(question: str, sql_hint: str, request_id: str) -> GenAndValidate:
        return GenAndValidate( success=False, pruned=None, generation=None, validation=None, failure_mode=failure_mode, failure_reason=f"stub {failure_mode} failure",)
    return factory


def _failing_executor() -> ClickHouseExecutor:
    """ClickHouse executor that always raises on execute."""
    async def transport(q: str, t: float):
        raise ClickHouseUnavailableError("simulated clickhouse outage")
    cfg = ClickHouseClientConfig.from_dict(_ch_client_dict())
    client = ClickHouseClient(cfg, transport=transport)
    return ClickHouseExecutor(SqlExecutionConfig.from_dict({'max_rows': 100, 'timeout_seconds': 2.0}), client)


class TestAnalyticsRouterClickHouseOnMiss:
    """Cache-miss path prefers ClickHouse before delegating to the legacy pipeline."""

    def test_miss_path_runs_on_clickhouse_when_eligible(self) -> None:
        router, pipeline, cache = _make_router( _success_result, gv_factory=_gv_factory_success(),)
        result = asyncio.run(router.run(question="average price for com"))
        assert result.success is True
        # ClickHouse-on-miss path served the request — legacy `run` was NOT called.
        assert pipeline.calls == []
        assert pipeline.gv_calls == ["average price for com"]
        # Verified SQL was upserted into the semantic cache for future hits.
        assert cache.size == 1

    def test_miss_path_returns_failure_when_ch_disabled(self) -> None:
        router, pipeline, _cache = _make_router( _success_result, gv_factory=_gv_factory_success(), clickhouse_on_miss_enabled=False,)
        result = asyncio.run(router.run(question="average price for com"))
        assert result.success is False
        assert result.failure_mode == 'no_substrate_available'
        assert pipeline.calls == []

    def test_miss_path_returns_failure_when_clickhouse_execution_fails(self) -> None:
        router, pipeline, _cache = _make_router(
            _success_result,
            gv_factory=_gv_factory_success(),
            ch_executor=_failing_executor(),
        )
        result = asyncio.run(router.run(question="average price for com"))
        assert result.success is False
        assert result.failure_mode == 'no_substrate_available'

    def test_miss_path_returns_typed_failure_on_validation_reject(self) -> None:
        router, pipeline, _cache = _make_router( _success_result, gv_factory=_gv_factory_failure(failure_mode='security'),)
        result = asyncio.run(router.run(question="drop table secrets"))
        assert result.success is False
        assert result.failure_mode == 'security'
        # Validation deterministically fails — do NOT fall back to Athena (it
        # would hit the same failure). The router returns the typed failure.
        assert pipeline.calls == []

    def test_miss_path_returns_failure_on_verifier_reject(self) -> None:
        router, pipeline, _cache = _make_router( _success_result, gv_factory=_gv_factory_success(), verifier_pass=False,)
        result = asyncio.run(router.run(question="average price for com"))
        assert result.success is False
        assert result.failure_mode == 'verifier'
        # Verifier rejected — do NOT fall back to Athena (CH already produced
        # rows; Athena would need a fresh gen+validate+execute roundtrip).
        assert pipeline.calls == []

    def test_miss_path_caches_verified_sql_for_subsequent_hit(self) -> None:
        router, pipeline, cache = _make_router( _success_result, gv_factory=_gv_factory_success(),)
        # First request — runs the miss path, caches the verified SQL.
        first = asyncio.run(router.run(question="average price for com"))
        assert first.success is True
        assert pipeline.gv_calls == ["average price for com"]

        # Second request with the same question should hit the fast path
        # without re-invoking generate_and_validate.
        second = asyncio.run(router.run(question="average price for com"))
        assert second.success is True
        assert pipeline.gv_calls == ["average price for com"]  # unchanged
        assert pipeline.calls == []  # legacy pipeline never invoked
        assert cache.size == 1

    def test_miss_path_returns_failure_when_pipeline_lacks_method(self) -> None:
        # When generate_and_validate raises AttributeError, router catches it
        # and returns no_substrate_available (no Athena fallback).
        router, pipeline, _cache = _make_router(_success_result, gv_factory=None)
        result = asyncio.run(router.run(question="average price for com"))
        assert result.success is False
        assert result.failure_mode == 'no_substrate_available'
