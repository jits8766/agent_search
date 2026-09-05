"""Typed configuration dataclasses for the real-time Analytics path.

Contains ClickHouse + MV router + exact NL-to-SQL cache configuration.
ClickHouse is the sole execution backend.

Every field is required (no silent defaults). Cross-cutting invariants:
- `clickhouse.port` must be in [1, 65535]
- `materialized_views[*].grain_columns` must be non-empty (rewrite needs a key)
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from semantic_search.core.exceptions import ConfigurationError

_ALLOWED_BACKENDS = frozenset({'clickhouse'})


def _require(d: Dict[str, Any], keys: List[str], context: str) -> None:
    """Raise ConfigurationError if any required key is missing from `d`."""
    if not isinstance(d, dict):
        raise ConfigurationError(f"{context}: expected dict, got {type(d).__name__}")
    for key in keys:
        if key not in d:
            raise ConfigurationError(f"{context}.{key} is required")


def _coerce_bool(value: Any) -> bool:
    """Coerce a config value to bool, accepting env-interpolated strings.

    Env interpolation (``${CLICKHOUSE_SECURE:-false}``) yields a string, so a
    plain ``bool("false")`` would wrongly evaluate truthy. Parse string forms
    explicitly; pass through real bools unchanged.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


@dataclass
class ClickHouseReadinessProbeConfig:
    """Boot-time reachability probe that gates ``ClickHouseClient.available``.

    When ``enabled`` is True the client issues ``sql`` over HTTP after the
    transport is constructed. ``available`` is True only when the probe
    succeeds. When ``enabled`` is False, ``available`` follows transport
    construction alone (no live round-trip).

    :param enabled: bool - Master toggle for the live reachability check
    :param sql: str - Read-only SQL used for the probe (and lifespan connection warm)
    :param timeout_seconds: float - Wall timeout for the boot-time probe round-trip
    :param warm_timeout_seconds: float - Wall timeout for the lifespan connection-warm query
    """
    enabled: bool
    sql: str
    timeout_seconds: float
    warm_timeout_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("nl_to_sql.clickhouse.readiness_probe.enabled must be a bool")
        if not isinstance(self.sql, str) or not str(self.sql).strip():
            raise ConfigurationError("nl_to_sql.clickhouse.readiness_probe.sql must be a non-empty string")
        if float(self.timeout_seconds) <= 0.0:
            raise ConfigurationError("nl_to_sql.clickhouse.readiness_probe.timeout_seconds must be > 0")
        if float(self.warm_timeout_seconds) <= 0.0:
            raise ConfigurationError("nl_to_sql.clickhouse.readiness_probe.warm_timeout_seconds must be > 0")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ClickHouseReadinessProbeConfig':
        _require(d, ['enabled', 'sql', 'timeout_seconds', 'warm_timeout_seconds'], 'nl_to_sql.clickhouse.readiness_probe')
        return cls(
            enabled=bool(d['enabled']),
            sql=str(d['sql']),
            timeout_seconds=float(d['timeout_seconds']),
            warm_timeout_seconds=float(d['warm_timeout_seconds']),
        )


@dataclass
class ClickHouseMvFreshnessProbeConfig:
    """Optional MV / replication lag probe executed after successful CH queries.

    When ``enabled`` is True the client runs ``lag_sql`` every
    ``interval_successful_queries`` successful primary round-trips and records
    a failure on the ``clickhouse`` health slot when the scalar lag exceeds
    ``max_lag_seconds`` or the probe cannot be parsed.

    :param enabled: bool - Master toggle
    :param interval_successful_queries: int - Run probe every N successful ``execute_query`` calls (>= 1 when enabled)
    :param lag_sql: str - ClickHouse SQL returning >=1 row with a numeric ``lag_seconds`` column (required when enabled)
    :param max_lag_seconds: float - Breach threshold (lag > max records health failure)
    :param query_timeout_seconds: float - Wall timeout for the probe query alone
    """
    enabled: bool
    interval_successful_queries: int
    lag_sql: str
    max_lag_seconds: float
    query_timeout_seconds: float

    def __post_init__(self) -> None:
        if bool(self.enabled):
            if int(self.interval_successful_queries) < 1:
                raise ConfigurationError("nl_to_sql.clickhouse.mv_freshness_probe.interval_successful_queries must be >= 1 when enabled")
            if not isinstance(self.lag_sql, str) or not str(self.lag_sql).strip():
                raise ConfigurationError("nl_to_sql.clickhouse.mv_freshness_probe.lag_sql must be a non-empty string when enabled")
            if float(self.max_lag_seconds) <= 0.0:
                raise ConfigurationError("nl_to_sql.clickhouse.mv_freshness_probe.max_lag_seconds must be > 0 when enabled")
            if float(self.query_timeout_seconds) <= 0.0:
                raise ConfigurationError("nl_to_sql.clickhouse.mv_freshness_probe.query_timeout_seconds must be > 0 when enabled")
        else:
            if int(self.interval_successful_queries) < 0:
                raise ConfigurationError("nl_to_sql.clickhouse.mv_freshness_probe.interval_successful_queries must be >= 0")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ClickHouseMvFreshnessProbeConfig':
        _require(d, ['enabled', 'interval_successful_queries', 'lag_sql', 'max_lag_seconds', 'query_timeout_seconds'], 'nl_to_sql.clickhouse.mv_freshness_probe')
        return cls(
            enabled=bool(d['enabled']),
            interval_successful_queries=int(d['interval_successful_queries']),
            lag_sql=str(d['lag_sql']),
            max_lag_seconds=float(d['max_lag_seconds']),
            query_timeout_seconds=float(d['query_timeout_seconds']),
        )


@dataclass
class ClickHouseClientConfig:
    """ClickHouse HTTP client settings for the real-time analytics path.

    The HTTP interface is used (port 8123 by default) so the dependency surface
    stays a single `httpx`-style client without a native protocol driver. All
    timeouts, retries, and pool sizes are explicit.
    """
    host: str
    port: int
    database: str
    user: str
    secure: bool
    connect_timeout_seconds: float
    read_timeout_seconds: float
    max_retries: int
    query_max_execution_time_seconds: float
    execution_timeout_seconds: float
    mv_freshness_probe: ClickHouseMvFreshnessProbeConfig
    readiness_probe: ClickHouseReadinessProbeConfig

    def __post_init__(self) -> None:
        if not isinstance(self.host, str) or not self.host:
            raise ConfigurationError("nl_to_sql.clickhouse.host must be a non-empty string")
        if not 1 <= int(self.port) <= 65535:
            raise ConfigurationError("nl_to_sql.clickhouse.port must be in [1, 65535]")
        if not isinstance(self.database, str) or not self.database:
            raise ConfigurationError("nl_to_sql.clickhouse.database must be a non-empty string")
        if not isinstance(self.user, str):
            raise ConfigurationError("nl_to_sql.clickhouse.user must be a string")
        if not isinstance(self.secure, bool):
            raise ConfigurationError("nl_to_sql.clickhouse.secure must be a bool")
        if float(self.connect_timeout_seconds) <= 0.0:
            raise ConfigurationError("nl_to_sql.clickhouse.connect_timeout_seconds must be > 0")
        if float(self.read_timeout_seconds) <= 0.0:
            raise ConfigurationError("nl_to_sql.clickhouse.read_timeout_seconds must be > 0")
        if int(self.max_retries) < 0:
            raise ConfigurationError("nl_to_sql.clickhouse.max_retries must be >= 0")
        if float(self.query_max_execution_time_seconds) < 0.0:
            raise ConfigurationError("nl_to_sql.clickhouse.query_max_execution_time_seconds must be >= 0")
        if float(self.execution_timeout_seconds) <= 0.0:
            raise ConfigurationError("nl_to_sql.clickhouse.execution_timeout_seconds must be > 0")
        if not isinstance(self.mv_freshness_probe, ClickHouseMvFreshnessProbeConfig):
            raise ConfigurationError("nl_to_sql.clickhouse.mv_freshness_probe must be a ClickHouseMvFreshnessProbeConfig")
        if not isinstance(self.readiness_probe, ClickHouseReadinessProbeConfig):
            raise ConfigurationError("nl_to_sql.clickhouse.readiness_probe must be a ClickHouseReadinessProbeConfig")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ClickHouseClientConfig':
        _require(d, ['host', 'port', 'database', 'user', 'secure', 'connect_timeout_seconds', 'read_timeout_seconds', 'max_retries', 'execution_timeout_seconds', 'mv_freshness_probe', 'readiness_probe'], 'nl_to_sql.clickhouse')
        return cls(
            host=str(d['host']),
            port=int(d['port']),
            database=str(d['database']),
            user=str(d['user']),
            secure=_coerce_bool(d['secure']),
            connect_timeout_seconds=float(d['connect_timeout_seconds']),
            read_timeout_seconds=float(d['read_timeout_seconds']),
            max_retries=int(d['max_retries']),
            query_max_execution_time_seconds=float(d.get('query_max_execution_time_seconds', 0.0)),
            execution_timeout_seconds=float(d['execution_timeout_seconds']),
            mv_freshness_probe=ClickHouseMvFreshnessProbeConfig.from_dict(d['mv_freshness_probe']),
            readiness_probe=ClickHouseReadinessProbeConfig.from_dict(d['readiness_probe']),
        )


@dataclass
class MaterializedViewConfig:
    """One incremental Materialized View definition the router can rewrite into.

    Catalog-driven (no MV name is hardcoded in code). Adding a new MV is a YAML
    edit alone.

    :param name: str - Fully-qualified MV name (e.g. `signals_platform_cln.mv_by_tld`)
    :param source_table: str - Raw table the MV aggregates from (e.g. `events_raw`)
    :param grain_columns: List[str] - GROUP BY keys the MV is partitioned on
    :param aggregate_columns: Dict[str, str] - {state_column: source_expr}
        (e.g. `{"avg_price_state": "price"}` — used for `*Merge`/`*State` rewriting)
    :param time_column: str - Day/hour bucket column the MV stores (e.g. `event_day`)
    :param freshness_lag_seconds: float - Worst-case Kinesis→queryable lag (audit
        only; emitted as a proxy signal so dashboards can warn on breach)
    :param fixed_filter_columns: List[str] - WHERE columns whose values are baked into
        the MV DDL (e.g. `sold_flag`, `sold_at` for sold-only MVs). The router counts
        these as satisfied when matching and removes them from the rewritten WHERE
        clause — the MV table does not carry those columns as physical columns.
    """
    name: str
    source_table: str
    grain_columns: List[str]
    aggregate_columns: Dict[str, str]
    time_column: str
    freshness_lag_seconds: float
    fixed_filter_columns: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or '.' not in self.name:
            raise ConfigurationError("nl_to_sql.analytics.materialized_views[*].name must be a fully-qualified 'db.table' string")
        if not isinstance(self.source_table, str) or not self.source_table:
            raise ConfigurationError("nl_to_sql.analytics.materialized_views[*].source_table must be a non-empty string")
        if not isinstance(self.grain_columns, list) or not self.grain_columns:
            raise ConfigurationError("nl_to_sql.analytics.materialized_views[*].grain_columns must be a non-empty list")
        for c in self.grain_columns:
            if not isinstance(c, str) or not c:
                raise ConfigurationError("nl_to_sql.analytics.materialized_views[*].grain_columns entries must be non-empty strings")
        if not isinstance(self.aggregate_columns, dict) or not self.aggregate_columns:
            raise ConfigurationError("nl_to_sql.analytics.materialized_views[*].aggregate_columns must be a non-empty dict")
        for state_col, src in self.aggregate_columns.items():
            if not isinstance(state_col, str) or not state_col:
                raise ConfigurationError("nl_to_sql.analytics.materialized_views[*].aggregate_columns keys must be non-empty strings")
            if not isinstance(src, str) or not src:
                raise ConfigurationError("nl_to_sql.analytics.materialized_views[*].aggregate_columns values must be non-empty strings")
        if not isinstance(self.time_column, str) or not self.time_column:
            raise ConfigurationError("nl_to_sql.analytics.materialized_views[*].time_column must be a non-empty string")
        if float(self.freshness_lag_seconds) < 0.0:
            raise ConfigurationError("nl_to_sql.analytics.materialized_views[*].freshness_lag_seconds must be >= 0")
        if not isinstance(self.fixed_filter_columns, list):
            raise ConfigurationError("nl_to_sql.analytics.materialized_views[*].fixed_filter_columns must be a list")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'MaterializedViewConfig':
        _require(d, ['name', 'source_table', 'grain_columns', 'aggregate_columns', 'time_column', 'freshness_lag_seconds'], 'nl_to_sql.analytics.materialized_views[*]')
        agg_in = d['aggregate_columns']
        if not isinstance(agg_in, dict):
            raise ConfigurationError("nl_to_sql.analytics.materialized_views[*].aggregate_columns must be a dict")
        return cls(
            name=str(d['name']),
            source_table=str(d['source_table']),
            grain_columns=[str(c) for c in d['grain_columns']],
            aggregate_columns={str(k): str(v) for k, v in agg_in.items()},
            time_column=str(d['time_column']),
            freshness_lag_seconds=float(d['freshness_lag_seconds']),
            fixed_filter_columns=[str(c) for c in d.get('fixed_filter_columns', [])],
        )


@dataclass
class MVRouterConfig:
    """MV-aware query router config.

    Owns the catalog of MVs the router is allowed to rewrite into. When the
    catalog is empty the router becomes a pass-through (no rewrite, no MV path);
    this lets the analytics module ship enabled-but-empty until ETL provisions
    the first MV.
    """
    enabled: bool
    materialized_views: List[MaterializedViewConfig]

    def __post_init__(self) -> None:
        if not isinstance(self.materialized_views, list):
            raise ConfigurationError("nl_to_sql.analytics.mv_router.materialized_views must be a list (empty allowed)")
        seen_names = set()
        for mv in self.materialized_views:
            if not isinstance(mv, MaterializedViewConfig):
                raise ConfigurationError("nl_to_sql.analytics.mv_router.materialized_views entries must be MaterializedViewConfig")
            if mv.name in seen_names:
                raise ConfigurationError(f"nl_to_sql.analytics.mv_router.materialized_views: duplicate MV name '{mv.name}'")
            seen_names.add(mv.name)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'MVRouterConfig':
        _require(d, ['enabled', 'materialized_views'], 'nl_to_sql.analytics.mv_router')
        mv_list = d['materialized_views']
        if not isinstance(mv_list, list):
            raise ConfigurationError("nl_to_sql.analytics.mv_router.materialized_views must be a list")
        return cls(enabled=bool(d['enabled']), materialized_views=[MaterializedViewConfig.from_dict(m) for m in mv_list])


@dataclass
class RemoteCacheLayerConfig:
    """Redis/Dragonfly-compatible payload tier (shared across replicas).

    Secret material never appears in YAML — only the name of an env var that
    holds the URL at runtime (same pattern as ClickHouse password env).
    """

    enabled: bool
    redis_url_env_var: str
    key_prefix: str
    socket_timeout_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.redis_url_env_var, str) or not self.redis_url_env_var.strip():
            raise ConfigurationError("RemoteCacheLayerConfig.redis_url_env_var must be a non-empty string")
        if not isinstance(self.key_prefix, str) or not self.key_prefix:
            raise ConfigurationError("RemoteCacheLayerConfig.key_prefix must be a non-empty string")
        if float(self.socket_timeout_seconds) <= 0.0:
            raise ConfigurationError("RemoteCacheLayerConfig.socket_timeout_seconds must be > 0")

    @classmethod
    def from_dict(cls, d: Dict[str, Any], context: str) -> 'RemoteCacheLayerConfig':
        _require(d, ['enabled', 'redis_url_env_var', 'key_prefix', 'socket_timeout_seconds'], context)
        return cls(enabled=bool(d['enabled']), redis_url_env_var=str(d['redis_url_env_var']), key_prefix=str(d['key_prefix']), socket_timeout_seconds=float(d['socket_timeout_seconds']))


@dataclass
class NLSqlExactCacheConfig:
    """Exact NL-to-SQL cache config (analytics-side question cache).

    Distinct from the search-side result caches (``cache.exact`` / ``cache.structured``,
    which cache search ``RankedResults``).
    This cache stores ``(question_template, sql_template)`` pairs keyed by the
    exact case-insensitive canonical form of the question. A hit lets the
    analytics path skip the LLM SQL-generation stage entirely. Matching is
    exact (no embedding similarity), so numerically distinct questions never
    share a cached SQL.

    :param enabled: bool - When False, cache is a no-op (lookups return None,
        writes are dropped) — the analytics pipeline still works
    :param ttl_seconds: int - Per-entry TTL
    :param max_entries: int - LRU cap before eviction
    :param min_verified_count: int - Minimum prior verifier-pass count before an
        entry is allowed to short-circuit the LLM. Guards cold templates that
        haven't been confirmed by the post-execution verifier yet.
    :param remote: RemoteCacheLayerConfig - Optional replica-shared tier (exact
        canonical-question key mirror).
    """
    enabled: bool
    ttl_seconds: int
    max_entries: int
    min_verified_count: int
    remote: RemoteCacheLayerConfig

    def __post_init__(self) -> None:
        if int(self.ttl_seconds) < 1:
            raise ConfigurationError("nl_to_sql.analytics.exact_cache.ttl_seconds must be >= 1")
        if int(self.max_entries) < 1:
            raise ConfigurationError("nl_to_sql.analytics.exact_cache.max_entries must be >= 1")
        if int(self.min_verified_count) < 0:
            raise ConfigurationError("nl_to_sql.analytics.exact_cache.min_verified_count must be >= 0")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'NLSqlExactCacheConfig':
        _require(d, ['enabled', 'ttl_seconds', 'max_entries', 'min_verified_count', 'remote'], 'nl_to_sql.analytics.exact_cache')
        return cls(
            enabled=bool(d['enabled']),
            ttl_seconds=int(d['ttl_seconds']),
            max_entries=int(d['max_entries']),
            min_verified_count=int(d['min_verified_count']),
            remote=RemoteCacheLayerConfig.from_dict(d['remote'], context='nl_to_sql.analytics.exact_cache.remote'),
        )


@dataclass
class MultiPeriodConfig:
    """Config for the multi-period aggregation handler.

    :param enabled: bool - Toggle the handler on/off without code changes
    :param query_timeout_seconds: float - Per-period ClickHouse query timeout
    :param max_rows_per_period: int - LIMIT applied to each period query
    :param hint_key_aliases: Dict[str, str] - Maps sql_hint keys to actual
        ClickHouse grain column names when they differ (e.g. 'auction_type' →
        'auction_type_id'). Keys not in this map are used verbatim.
    """
    enabled: bool = True
    query_timeout_seconds: float = 3.0
    max_rows_per_period: int = 100
    hint_key_aliases: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("nl_to_sql.analytics.multi_period.enabled must be a bool")
        if float(self.query_timeout_seconds) <= 0.0:
            raise ConfigurationError("nl_to_sql.analytics.multi_period.query_timeout_seconds must be > 0")
        if int(self.max_rows_per_period) <= 0:
            raise ConfigurationError("nl_to_sql.analytics.multi_period.max_rows_per_period must be > 0")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'MultiPeriodConfig':
        return cls(
            enabled=bool(d.get('enabled', True)),
            query_timeout_seconds=float(d.get('query_timeout_seconds', 3.0)),
            max_rows_per_period=int(d.get('max_rows_per_period', 100)),
            hint_key_aliases={str(k): str(v) for k, v in (d.get('hint_key_aliases') or {}).items()},
        )


@dataclass
class AnalyticsConfig:
    """Real-time analytics config bundle (ClickHouse + MV router + exact cache).

    All three sub-blocks are required so a YAML edit cannot accidentally enable
    one path while leaving another mis-configured.

    :param enabled: bool - When False, the analytics path is fully disabled and
        the existing Athena pipeline serves analytics requests untouched
    :param freshness_target_seconds: float - Acceptable Kinesis→queryable lag
        (used as the proxy-signal threshold; 5.0s in the canonical YAML)
    :param clickhouse: ClickHouseClientConfig - Connection settings
    :param mv_router: MVRouterConfig - MV catalog + rewrite toggle
    :param exact_cache: NLSqlExactCacheConfig - Exact-cache settings
    :param clickhouse_on_miss_enabled: bool - When True (default) and the
        exact cache misses, the router executes the validated SQL on ClickHouse
        (with MV rewrite). When False, every miss returns a no_substrate_available error.
    :param capabilities: Optional[AnalyticsCapabilitiesConfig] - Extended analytics
        capability config (growth, anomaly, forecasting, etc.); None when the
        ``capabilities`` YAML block is omitted (disables all extended capabilities)
    """
    enabled: bool
    freshness_target_seconds: float
    clickhouse: ClickHouseClientConfig
    mv_router: MVRouterConfig
    exact_cache: NLSqlExactCacheConfig
    seed_target_table: str
    seed_snapshot_table: str
    clickhouse_on_miss_enabled: bool = True
    capabilities: Optional['AnalyticsCapabilitiesConfig'] = None
    multi_period: Optional[MultiPeriodConfig] = None
    engine_dispatch: Optional['EngineDispatchConfig'] = None

    def __post_init__(self) -> None:
        if float(self.freshness_target_seconds) <= 0.0:
            raise ConfigurationError("nl_to_sql.analytics.freshness_target_seconds must be > 0")
        if not isinstance(self.clickhouse_on_miss_enabled, bool):
            raise ConfigurationError("nl_to_sql.analytics.clickhouse_on_miss_enabled must be a bool")
        if not self.seed_target_table or not isinstance(self.seed_target_table, str):
            raise ConfigurationError("nl_to_sql.analytics.seed_target_table must be a non-empty string")
        if not self.seed_snapshot_table or not isinstance(self.seed_snapshot_table, str):
            raise ConfigurationError("nl_to_sql.analytics.seed_snapshot_table must be a non-empty string")
        if self.capabilities is not None and not isinstance(self.capabilities, AnalyticsCapabilitiesConfig):
            raise ConfigurationError("nl_to_sql.analytics.capabilities must be an AnalyticsCapabilitiesConfig or omitted")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'AnalyticsConfig':
        _require(d, ['enabled', 'freshness_target_seconds', 'clickhouse', 'mv_router', 'exact_cache', 'clickhouse_on_miss_enabled'], 'nl_to_sql.analytics')
        capabilities: Optional[AnalyticsCapabilitiesConfig] = None
        if 'capabilities' in d and d['capabilities'] is not None:
            capabilities = AnalyticsCapabilitiesConfig.from_dict(d['capabilities'])
        multi_period: Optional[MultiPeriodConfig] = None
        if 'multi_period' in d and d['multi_period'] is not None:
            multi_period = MultiPeriodConfig.from_dict(d['multi_period'])
        engine_dispatch: Optional[EngineDispatchConfig] = None
        if 'engine_dispatch' in d and d['engine_dispatch'] is not None:
            engine_dispatch = EngineDispatchConfig.from_dict(d['engine_dispatch'])
        return cls(
            enabled=bool(d['enabled']),
            freshness_target_seconds=float(d['freshness_target_seconds']),
            clickhouse=ClickHouseClientConfig.from_dict(d['clickhouse']),
            mv_router=MVRouterConfig.from_dict(d['mv_router']),
            exact_cache=NLSqlExactCacheConfig.from_dict(d['exact_cache']),
            seed_target_table=str(d.get('seed_target_table', 'signals_platform_cln.auction_audit_cln')),
            seed_snapshot_table=str(d.get('seed_snapshot_table', 'signals_platform_cln.domain_snapshots')),
            clickhouse_on_miss_enabled=bool(d['clickhouse_on_miss_enabled']),
            capabilities=capabilities,
            multi_period=multi_period,
            engine_dispatch=engine_dispatch,
        )


_ALLOWED_GRAINS = frozenset({'hour', 'day', 'week', 'month'})


@dataclass
class TimeWindowAnalyticsConfig:
    """Config for time-window analytics (hourly / daily / weekly / monthly aggregations).

    :param enabled: bool - Master toggle
    :param supported_grains: List[str] - Allowed grain values (subset of hour/day/week/month)
    :param default_grain: str - Grain used when the caller omits an explicit grain
    :param default_lookback_periods: int - Default number of periods to include in a window query
    """
    enabled: bool
    supported_grains: List[str]
    default_grain: str
    default_lookback_periods: int

    def __post_init__(self) -> None:
        if not isinstance(self.supported_grains, list) or not self.supported_grains:
            raise ConfigurationError('signals_platform_cln.capabilities.time_windows.supported_grains must be a non-empty list')
        for g in self.supported_grains:
            if g not in _ALLOWED_GRAINS:
                raise ConfigurationError(f"signals_platform_cln.capabilities.time_windows.supported_grains entry {g!r} must be one of {sorted(_ALLOWED_GRAINS)}")
        if self.default_grain not in _ALLOWED_GRAINS:
            raise ConfigurationError(f"signals_platform_cln.capabilities.time_windows.default_grain {self.default_grain!r} must be one of {sorted(_ALLOWED_GRAINS)}")
        if int(self.default_lookback_periods) < 1:
            raise ConfigurationError('signals_platform_cln.capabilities.time_windows.default_lookback_periods must be >= 1')

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TimeWindowAnalyticsConfig':
        _require(d, ['enabled', 'supported_grains', 'default_grain', 'default_lookback_periods'], 'signals_platform_cln.capabilities.time_windows')
        return cls(
            enabled=bool(d['enabled']),
            supported_grains=[str(g) for g in d['supported_grains']],
            default_grain=str(d['default_grain']),
            default_lookback_periods=int(d['default_lookback_periods']),
        )


@dataclass
class AnomalyDetectionConfig:
    """Config for spike and anomaly detection via rolling Z-score.

    :param enabled: bool - Master toggle
    :param z_score_threshold: float - Standard-deviation multiplier for anomaly gate (> 0)
    :param min_window_periods: int - Rolling window size; minimum periods before Z-score is valid (>= 2)
    :param max_results: int - Row cap returned by anomaly queries (>= 1)
    """
    enabled: bool
    z_score_threshold: float
    min_window_periods: int
    max_results: int

    def __post_init__(self) -> None:
        if float(self.z_score_threshold) <= 0.0:
            raise ConfigurationError('signals_platform_cln.capabilities.anomaly_detection.z_score_threshold must be > 0')
        if int(self.min_window_periods) < 2:
            raise ConfigurationError('signals_platform_cln.capabilities.anomaly_detection.min_window_periods must be >= 2')
        if int(self.max_results) < 1:
            raise ConfigurationError('signals_platform_cln.capabilities.anomaly_detection.max_results must be >= 1')

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'AnomalyDetectionConfig':
        _require(d, ['enabled', 'z_score_threshold', 'min_window_periods', 'max_results'], 'signals_platform_cln.capabilities.anomaly_detection')
        return cls(
            enabled=bool(d['enabled']),
            z_score_threshold=float(d['z_score_threshold']),
            min_window_periods=int(d['min_window_periods']),
            max_results=int(d['max_results']),
        )


@dataclass
class GrowthTrackingConfig:
    """Config for fastest-growing / declining domain signals_platform_cln.

    :param enabled: bool - Master toggle
    :param default_lookback_periods: int - Default period window for growth comparison (>= 1)
    :param snapshot_table: str - Table that holds domain metric snapshots (non-empty)
    """
    enabled: bool
    default_lookback_periods: int
    snapshot_table: str

    def __post_init__(self) -> None:
        if int(self.default_lookback_periods) < 1:
            raise ConfigurationError('signals_platform_cln.capabilities.growth_tracking.default_lookback_periods must be >= 1')
        if not isinstance(self.snapshot_table, str) or not self.snapshot_table:
            raise ConfigurationError('signals_platform_cln.capabilities.growth_tracking.snapshot_table must be a non-empty string')

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'GrowthTrackingConfig':
        _require(d, ['enabled', 'default_lookback_periods', 'snapshot_table'], 'signals_platform_cln.capabilities.growth_tracking')
        return cls(
            enabled=bool(d['enabled']),
            default_lookback_periods=int(d['default_lookback_periods']),
            snapshot_table=str(d['snapshot_table']),
        )


@dataclass
class ForecastingConfig:
    """Config for linear-trend forecasting / predictive insights.

    :param enabled: bool - Master toggle
    :param max_history_periods: int - Max past periods fed into the regression (>= 7)
    :param max_forecast_periods: int - Max periods to project ahead (>= 1)
    :param supported_grains: List[str] - Grains on which forecasting is available
    """
    enabled: bool
    max_history_periods: int
    max_forecast_periods: int
    supported_grains: List[str]

    def __post_init__(self) -> None:
        if int(self.max_history_periods) < 7:
            raise ConfigurationError('signals_platform_cln.capabilities.forecasting.max_history_periods must be >= 7')
        if int(self.max_forecast_periods) < 1:
            raise ConfigurationError('signals_platform_cln.capabilities.forecasting.max_forecast_periods must be >= 1')
        if not isinstance(self.supported_grains, list) or not self.supported_grains:
            raise ConfigurationError('signals_platform_cln.capabilities.forecasting.supported_grains must be a non-empty list')
        for g in self.supported_grains:
            if g not in _ALLOWED_GRAINS:
                raise ConfigurationError(f"signals_platform_cln.capabilities.forecasting.supported_grains entry {g!r} must be one of {sorted(_ALLOWED_GRAINS)}")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ForecastingConfig':
        _require(d, ['enabled', 'max_history_periods', 'max_forecast_periods', 'supported_grains'], 'signals_platform_cln.capabilities.forecasting')
        return cls(
            enabled=bool(d['enabled']),
            max_history_periods=int(d['max_history_periods']),
            max_forecast_periods=int(d['max_forecast_periods']),
            supported_grains=[str(g) for g in d['supported_grains']],
        )


@dataclass
class RankingScoringConfig:
    """Config for custom ranking and composite scoring queries.

    :param enabled: bool - Master toggle
    :param default_score_weights: Dict[str, float] - Default per-metric weights applied
        when the caller does not supply explicit weights (empty = caller must always supply)
    :param max_candidates: int - Row cap before applying weighted ranking (>= 1)
    """
    enabled: bool
    default_score_weights: Dict[str, float]
    max_candidates: int

    def __post_init__(self) -> None:
        if not isinstance(self.default_score_weights, dict):
            raise ConfigurationError('signals_platform_cln.capabilities.ranking_scoring.default_score_weights must be a dict')
        for k, v in self.default_score_weights.items():
            if not isinstance(k, str) or not k:
                raise ConfigurationError('signals_platform_cln.capabilities.ranking_scoring.default_score_weights keys must be non-empty strings')
            if float(v) <= 0.0:
                raise ConfigurationError(f"signals_platform_cln.capabilities.ranking_scoring.default_score_weights[{k!r}] must be > 0")
        if int(self.max_candidates) < 1:
            raise ConfigurationError('signals_platform_cln.capabilities.ranking_scoring.max_candidates must be >= 1')

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'RankingScoringConfig':
        _require(d, ['enabled', 'default_score_weights', 'max_candidates'], 'signals_platform_cln.capabilities.ranking_scoring')
        raw_weights = d['default_score_weights']
        if not isinstance(raw_weights, dict):
            raise ConfigurationError('signals_platform_cln.capabilities.ranking_scoring.default_score_weights must be a dict')
        return cls(
            enabled=bool(d['enabled']),
            default_score_weights={str(k): float(v) for k, v in raw_weights.items()},
            max_candidates=int(d['max_candidates']),
        )


@dataclass
class HistoricalSnapshotConfig:
    """Config for maintaining and querying historical domain snapshots.

    :param enabled: bool - Master toggle
    :param snapshot_table: str - Fully-qualified or bare table name for snapshots
    :param retention_days: int - How many days of snapshots to keep (>= 1)
    :param snapshot_grains: List[str] - Granularities at which snapshots are maintained
    :param sale_signal_terms: List[str] - Lowercase keywords that indicate a historical-sales query (e.g. "sold", "completed"); empty list disables detection
    :param default_window_days: int - Fallback time window in days when question has no explicit period (>= 1)
    :param query_template: str - SQL template with {snapshot_table}, {window_days}, {filter_clauses} placeholders
    :param window_patterns: List[Dict[str, Any]] - Named-period to days mappings; each entry must have 'term' (str) and 'days' (int >= 1)
    """
    enabled: bool
    snapshot_table: str
    retention_days: int
    snapshot_grains: List[str]
    sale_signal_terms: List[str] = field(default_factory=list)
    default_window_days: int = 7
    query_template: str = ''
    window_patterns: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot_table, str) or not self.snapshot_table:
            raise ConfigurationError('signals_platform_cln.capabilities.historical_snapshots.snapshot_table must be a non-empty string')
        if int(self.retention_days) < 1:
            raise ConfigurationError('signals_platform_cln.capabilities.historical_snapshots.retention_days must be >= 1')
        if not isinstance(self.snapshot_grains, list) or not self.snapshot_grains:
            raise ConfigurationError('signals_platform_cln.capabilities.historical_snapshots.snapshot_grains must be a non-empty list')
        for g in self.snapshot_grains:
            if g not in _ALLOWED_GRAINS:
                raise ConfigurationError(f"signals_platform_cln.capabilities.historical_snapshots.snapshot_grains entry {g!r} must be one of {sorted(_ALLOWED_GRAINS)}")
        if not isinstance(self.sale_signal_terms, list):
            raise ConfigurationError('signals_platform_cln.capabilities.historical_snapshots.sale_signal_terms must be a list')
        if int(self.default_window_days) < 1:
            raise ConfigurationError('signals_platform_cln.capabilities.historical_snapshots.default_window_days must be >= 1')
        for wp in self.window_patterns:
            if 'term' not in wp or 'days' not in wp:
                raise ConfigurationError('signals_platform_cln.capabilities.historical_snapshots.window_patterns entries must have term and days')
            if int(wp['days']) < 1:
                raise ConfigurationError('signals_platform_cln.capabilities.historical_snapshots.window_patterns.days must be >= 1')

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'HistoricalSnapshotConfig':
        _require(d, ['enabled', 'snapshot_table', 'retention_days', 'snapshot_grains'], 'signals_platform_cln.capabilities.historical_snapshots')
        return cls(
            enabled=bool(d['enabled']),
            snapshot_table=str(d['snapshot_table']),
            retention_days=int(d['retention_days']),
            snapshot_grains=[str(g) for g in d['snapshot_grains']],
            sale_signal_terms=[str(t) for t in d.get('sale_signal_terms', [])],
            default_window_days=int(d.get('default_window_days', 7)),
            query_template=str(d.get('query_template', '')),
            window_patterns=[dict(p) for p in d.get('window_patterns', [])],
        )


@dataclass
class DomainAnalyticsConfig:
    """Column-mapping config for the domain-specific analytics engine.

    Maps logical field names to the physical column names of the auction snap
    table so the engine stays table-agnostic: changing the column names is a
    YAML edit, not a code change.

    :param enabled: bool - Master toggle; when False DomainAnalyticsEngine returns empty lists
    :param source_table: str - Physical auction table (e.g. ``auction_audit_cln``)
    :param domain_column: str - Fully-qualified domain name column
    :param tld_column: str - Top-level domain column (low-cardinality)
    :param price_column: str - Current auction price column
    :param bid_count_column: str - Bid count column (proxy for demand)
    :param auction_end_column: str - Auction end/expiry timestamp column
    :param score_column: str - Pre-computed domain quality/value score column
    :param category_column: str - Auction category / type column
    :param created_column: str - Row creation timestamp column
    :param signals_table: str - Fully-qualified ClickHouse table for FeedbackSignal writes
    :param signals_batch_size: int - Max signals per bulk INSERT (>= 1)
    :param signals_insert_timeout_seconds: float - Hard timeout for INSERT ops (> 0)
    :param sold_flag_column: Optional[str] - Column for sold status flag (UInt8 0/1); None = not tracked
    :param listed_at_column: Optional[str] - Column for listing timestamp; None = not tracked
    :param sold_at_column: Optional[str] - Column for sold timestamp; None = not tracked
    :param domain_authority_column: Optional[str] - Column for domain authority score; None = not tracked
    :param traffic_column: Optional[str] - Column for monthly organic traffic; None = not tracked
    :param domain_age_column: Optional[str] - Column for domain age in days; None = not tracked
    :param registrar_column: Optional[str] - Column for registrar name; None = not tracked
    :param backlink_column: Optional[str] - Column for total backlink count; None = not tracked
    :param referring_domains_column: Optional[str] - Column for referring domains count; None = not tracked
    :param auction_type_name_column: Optional[str] - Column for human-readable auction type label; None = not tracked
    :param category_name_column: Optional[str] - Column for human-readable category label; None = not tracked
    :param hold_days_column: Optional[str] - Column for listing duration in days (ends_at - listed_at); None = not tracked
    :param expiry_status_column: Optional[str] - Column for expiry lifecycle status bucket; None = not tracked
    :param bid_events_table: Optional[str] - Fully-qualified bid events table; None = bid velocity methods unavailable
    :param watch_events_table: Optional[str] - Fully-qualified watch events table; None = watch analytics unavailable
    :param transactions_table: Optional[str] - Fully-qualified domain transactions table; None = transaction analytics unavailable
    :param bid_velocity_mv: Optional[str] - AggregatingMergeTree MV for bid velocity by auction/hour; None = acceleration unavailable
    :param bid_velocity_item_mv: Optional[str] - AggregatingMergeTree MV for bid velocity by item/hour; None = watch conversion unavailable
    :param watch_density_mv: Optional[str] - AggregatingMergeTree MV for watch density by item/day; None = watch analytics unavailable
    :param unique_bidder_mv: Optional[str] - AggregatingMergeTree MV for unique bidder counts by item; None = competition depth unavailable
    :param hold_time_mv: Optional[str] - AggregatingMergeTree MV for hold time by type/day; None = hold time distribution unavailable
    :param price_quantiles: Optional[List[float]] - Fractile list for price distribution (e.g. [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]); None = uses [0.1,0.25,0.5,0.75,0.9,0.99]
    :param bid_velocity_window_hours: Optional[int] - Recent window hours for bid acceleration; None = method-level default applies
    :param top_k_limit: Optional[int] - k for topK() keyword queries; None = method-level default applies
    :param hhi_min_sales: Optional[int] - Minimum sold count for HHI registrar inclusion; None = method-level default applies
    :param comparable_min_sales: Optional[int] - Minimum sales for comparable price cohort; None = method-level default applies
    :param funnel_window_seconds: Optional[int] - windowFunnel time window in seconds; None = method-level default applies
    """
    enabled: bool
    source_table: str
    domain_column: str
    tld_column: str
    price_column: str
    bid_count_column: str
    auction_end_column: str
    score_column: str
    category_column: str
    created_column: str
    signals_user_column: str
    signals_event_column: str
    signals_time_column: str
    signals_origin_column: str
    signals_table: str
    signals_batch_size: int
    signals_insert_timeout_seconds: float
    sold_flag_column: Optional[str] = None
    listed_at_column: Optional[str] = None
    sold_at_column: Optional[str] = None
    domain_authority_column: Optional[str] = None
    traffic_column: Optional[str] = None
    domain_age_column: Optional[str] = None
    registrar_column: Optional[str] = None
    backlink_column: Optional[str] = None
    referring_domains_column: Optional[str] = None
    auction_type_name_column: Optional[str] = None
    category_name_column: Optional[str] = None
    hold_days_column: Optional[str] = None
    expiry_status_column: Optional[str] = None
    bid_events_table: Optional[str] = None
    watch_events_table: Optional[str] = None
    transactions_table: Optional[str] = None
    bid_velocity_mv: Optional[str] = None
    bid_velocity_item_mv: Optional[str] = None
    watch_density_mv: Optional[str] = None
    unique_bidder_mv: Optional[str] = None
    hold_time_mv: Optional[str] = None
    price_quantiles: Optional[List[float]] = None
    bid_velocity_window_hours: Optional[int] = None
    top_k_limit: Optional[int] = None
    hhi_min_sales: Optional[int] = None
    comparable_min_sales: Optional[int] = None
    funnel_window_seconds: Optional[int] = None
    buyer_segment_column: Optional[str] = None
    watch_mv_item_column: Optional[str] = None
    watch_mv_day_column: Optional[str] = None
    watch_mv_count_state_column: Optional[str] = None
    watch_mv_unique_state_column: Optional[str] = None
    transactions_category_column: Optional[str] = None
    transactions_sold_at_column: Optional[str] = None
    transactions_sale_price_column: Optional[str] = None
    transactions_listed_price_column: Optional[str] = None
    transactions_govalue_column: Optional[str] = None
    lifecycle_new_days: Optional[int] = None
    lifecycle_dormant_days: Optional[int] = None
    moving_average_window_size: Optional[int] = None
    max_lookback_periods: int = 90

    def __post_init__(self) -> None:
        for attr in (
            'source_table', 'domain_column', 'tld_column', 'price_column',
            'bid_count_column', 'auction_end_column', 'score_column',
            'category_column', 'created_column',
            'signals_user_column', 'signals_event_column',
            'signals_time_column', 'signals_origin_column',
            'signals_table',
        ):
            val = getattr(self, attr)
            if not isinstance(val, str) or not val:
                raise ConfigurationError(f"signals_platform_cln.capabilities.domain_signals_platform_cln.{attr} must be a non-empty string")
        if int(self.signals_batch_size) < 1:
            raise ConfigurationError('signals_platform_cln.capabilities.domain_signals_platform_cln.signals_batch_size must be >= 1')
        if float(self.signals_insert_timeout_seconds) <= 0.0:
            raise ConfigurationError('signals_platform_cln.capabilities.domain_signals_platform_cln.signals_insert_timeout_seconds must be > 0')
        if int(self.max_lookback_periods) < 1:
            raise ConfigurationError('signals_platform_cln.capabilities.domain_analytics.max_lookback_periods must be >= 1')

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DomainAnalyticsConfig':
        _require(
            d,
            [
                'enabled', 'source_table', 'domain_column', 'tld_column',
                'price_column', 'bid_count_column', 'auction_end_column',
                'score_column', 'category_column', 'created_column',
                'signals_user_column', 'signals_event_column',
                'signals_time_column', 'signals_origin_column',
                'signals_table', 'signals_batch_size', 'signals_insert_timeout_seconds',
            ],
            'signals_platform_cln.capabilities.domain_analytics',
        )
        return cls(
            enabled=bool(d['enabled']),
            source_table=str(d['source_table']),
            domain_column=str(d['domain_column']),
            tld_column=str(d['tld_column']),
            price_column=str(d['price_column']),
            bid_count_column=str(d['bid_count_column']),
            auction_end_column=str(d['auction_end_column']),
            score_column=str(d['score_column']),
            category_column=str(d['category_column']),
            created_column=str(d['created_column']),
            signals_user_column=str(d['signals_user_column']),
            signals_event_column=str(d['signals_event_column']),
            signals_time_column=str(d['signals_time_column']),
            signals_origin_column=str(d['signals_origin_column']),
            signals_table=str(d['signals_table']),
            signals_batch_size=int(d['signals_batch_size']),
            signals_insert_timeout_seconds=float(d['signals_insert_timeout_seconds']),
            sold_flag_column=str(d['sold_flag_column']) if d.get('sold_flag_column') else None,
            listed_at_column=str(d['listed_at_column']) if d.get('listed_at_column') else None,
            sold_at_column=str(d['sold_at_column']) if d.get('sold_at_column') else None,
            domain_authority_column=str(d['domain_authority_column']) if d.get('domain_authority_column') else None,
            traffic_column=str(d['traffic_column']) if d.get('traffic_column') else None,
            domain_age_column=str(d['domain_age_column']) if d.get('domain_age_column') else None,
            registrar_column=str(d['registrar_column']) if d.get('registrar_column') else None,
            backlink_column=str(d['backlink_column']) if d.get('backlink_column') else None,
            referring_domains_column=str(d['referring_domains_column']) if d.get('referring_domains_column') else None,
            auction_type_name_column=str(d['auction_type_name_column']) if d.get('auction_type_name_column') else None,
            category_name_column=str(d['category_name_column']) if d.get('category_name_column') else None,
            hold_days_column=str(d['hold_days_column']) if d.get('hold_days_column') else None,
            expiry_status_column=str(d['expiry_status_column']) if d.get('expiry_status_column') else None,
            bid_events_table=str(d['bid_events_table']) if d.get('bid_events_table') else None,
            watch_events_table=str(d['watch_events_table']) if d.get('watch_events_table') else None,
            transactions_table=str(d['transactions_table']) if d.get('transactions_table') else None,
            bid_velocity_mv=str(d['bid_velocity_mv']) if d.get('bid_velocity_mv') else None,
            bid_velocity_item_mv=str(d['bid_velocity_item_mv']) if d.get('bid_velocity_item_mv') else None,
            watch_density_mv=str(d['watch_density_mv']) if d.get('watch_density_mv') else None,
            unique_bidder_mv=str(d['unique_bidder_mv']) if d.get('unique_bidder_mv') else None,
            hold_time_mv=str(d['hold_time_mv']) if d.get('hold_time_mv') else None,
            price_quantiles=[float(q) for q in d['price_quantiles']] if d.get('price_quantiles') else None,
            bid_velocity_window_hours=int(d['bid_velocity_window_hours']) if d.get('bid_velocity_window_hours') is not None else None,
            top_k_limit=int(d['top_k_limit']) if d.get('top_k_limit') is not None else None,
            hhi_min_sales=int(d['hhi_min_sales']) if d.get('hhi_min_sales') is not None else None,
            comparable_min_sales=int(d['comparable_min_sales']) if d.get('comparable_min_sales') is not None else None,
            funnel_window_seconds=int(d['funnel_window_seconds']) if d.get('funnel_window_seconds') is not None else None,
            max_lookback_periods=int(d['max_lookback_periods']) if d.get('max_lookback_periods') is not None else 90,
            buyer_segment_column=str(d['buyer_segment_column']) if d.get('buyer_segment_column') else None,
            watch_mv_item_column=str(d['watch_mv_item_column']) if d.get('watch_mv_item_column') else None,
            watch_mv_day_column=str(d['watch_mv_day_column']) if d.get('watch_mv_day_column') else None,
            watch_mv_count_state_column=str(d['watch_mv_count_state_column']) if d.get('watch_mv_count_state_column') else None,
            watch_mv_unique_state_column=str(d['watch_mv_unique_state_column']) if d.get('watch_mv_unique_state_column') else None,
            transactions_category_column=str(d['transactions_category_column']) if d.get('transactions_category_column') else None,
            transactions_sold_at_column=str(d['transactions_sold_at_column']) if d.get('transactions_sold_at_column') else None,
            transactions_sale_price_column=str(d['transactions_sale_price_column']) if d.get('transactions_sale_price_column') else None,
            transactions_listed_price_column=str(d['transactions_listed_price_column']) if d.get('transactions_listed_price_column') else None,
            transactions_govalue_column=str(d['transactions_govalue_column']) if d.get('transactions_govalue_column') else None,
            lifecycle_new_days=int(d['lifecycle_new_days']) if d.get('lifecycle_new_days') is not None else None,
            lifecycle_dormant_days=int(d['lifecycle_dormant_days']) if d.get('lifecycle_dormant_days') is not None else None,
            moving_average_window_size=int(d['moving_average_window_size']) if d.get('moving_average_window_size') is not None else None,
        )


@dataclass
class SignalWriterConfig:
    """Config for FeedbackSignalWriter — ClickHouse signal persistence tier.

    :param enabled: bool - When False writes are no-ops (no CH round-trip)
    :param signals_table: str - Fully-qualified table for INSERT (e.g. ``signals_platform_cln.feedback_signals``)
    :param batch_size: int - Max signals per bulk INSERT VALUES call (>= 1)
    :param insert_timeout_seconds: float - Hard timeout per INSERT (> 0)
    """
    enabled: bool
    signals_table: str
    batch_size: int
    insert_timeout_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.signals_table, str) or not self.signals_table:
            raise ConfigurationError('signals_platform_cln.signal_writer.signals_table must be a non-empty string')
        if int(self.batch_size) < 1:
            raise ConfigurationError('signals_platform_cln.signal_writer.batch_size must be >= 1')
        if float(self.insert_timeout_seconds) <= 0.0:
            raise ConfigurationError('signals_platform_cln.signal_writer.insert_timeout_seconds must be > 0')

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SignalWriterConfig':
        _require(d, ['enabled', 'signals_table', 'batch_size', 'insert_timeout_seconds'], 'signals_platform_cln.signal_writer')
        return cls(
            enabled=bool(d['enabled']),
            signals_table=str(d['signals_table']),
            batch_size=int(d['batch_size']),
            insert_timeout_seconds=float(d['insert_timeout_seconds']),
        )


@dataclass
class FilterBridgeConfig:
    """Config for converting QI sql_hint key-value slots to typed CH FilterSpec objects.

    The hint string (e.g. ``price_min=500 tld=com,net traffic_max=10000``) is
    parsed by :func:`~semantic_search.analytics.filter_bridge.sql_hint_to_filters`.
    Slot → column resolution uses ``slot_to_column_attr`` (slot name →
    DomainAnalyticsConfig attribute name), so no physical column names appear here.

    :param enabled: bool - When False, no filters are built from sql_hint
    :param slot_to_column_attr: Dict[str, str] - Maps sql_hint slot name to
        DomainAnalyticsConfig attribute name (e.g. ``'price_min': 'price_column'``)
    :param min_suffix: str - Suffix marking minimum-bound slots (e.g. '_min')
    :param max_suffix: str - Suffix marking maximum-bound slots (e.g. '_max')
    """
    enabled: bool
    slot_to_column_attr: Dict[str, str]
    min_suffix: str
    max_suffix: str

    def __post_init__(self) -> None:
        if not isinstance(self.slot_to_column_attr, dict):
            raise ConfigurationError('engine_dispatch.filter_bridge.slot_to_column_attr must be a dict')
        if not isinstance(self.min_suffix, str) or not self.min_suffix:
            raise ConfigurationError('engine_dispatch.filter_bridge.min_suffix must be a non-empty string')
        if not isinstance(self.max_suffix, str) or not self.max_suffix:
            raise ConfigurationError('engine_dispatch.filter_bridge.max_suffix must be a non-empty string')

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FilterBridgeConfig':
        _require(d, ['enabled', 'slot_to_column_attr', 'min_suffix', 'max_suffix'], 'engine_dispatch.filter_bridge')
        raw_map = d['slot_to_column_attr']
        if not isinstance(raw_map, dict):
            raise ConfigurationError('engine_dispatch.filter_bridge.slot_to_column_attr must be a dict')
        return cls(
            enabled=bool(d['enabled']),
            slot_to_column_attr={str(k): str(v) for k, v in raw_map.items()},
            min_suffix=str(d['min_suffix']),
            max_suffix=str(d['max_suffix']),
        )


@dataclass
class EngineRouteConfig:
    """Config for a single keyword-to-method route in the engine dispatcher.

    :param method: str - DomainAnalyticsEngine method name to call on match
    :param keywords: List[str] - Any substring match triggers this route (lowercased)
    :param params: Dict[str, Any] - Passed directly to the method as **kwargs;
        grain values ('day', 'week', etc.) are coerced to TimeGrain at dispatcher construction time
    """
    method: str
    keywords: List[str]
    params: Dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.method, str) or not self.method:
            raise ConfigurationError('engine_dispatch route.method must be a non-empty string')
        if not isinstance(self.keywords, list) or not self.keywords:
            raise ConfigurationError('engine_dispatch route.keywords must be a non-empty list')
        if not isinstance(self.params, dict):
            raise ConfigurationError('engine_dispatch route.params must be a dict')

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'EngineRouteConfig':
        _require(d, ['method', 'keywords', 'params'], 'engine_dispatch.routes[*]')
        return cls(
            method=str(d['method']),
            keywords=[str(k) for k in d['keywords']],
            params=dict(d['params']),
        )


@dataclass
class EngineDispatchConfig:
    """Config block for keyword-driven dispatch to DomainAnalyticsEngine methods.

    :param enabled: bool - Master toggle; when False dispatcher is a no-op
    :param timeout_seconds: float - Per-call timeout for engine method calls (> 0)
    :param routes: List[EngineRouteConfig] - Ordered list of keyword-to-method routes
    :param filter_bridge: Optional[FilterBridgeConfig] - When set, QI sql_hint slots
        are converted to typed FilterSpec objects and injected into dispatched method calls
    """
    enabled: bool
    timeout_seconds: float
    routes: List[EngineRouteConfig]
    filter_bridge: Optional[FilterBridgeConfig] = None

    def __post_init__(self) -> None:
        if float(self.timeout_seconds) <= 0.0:
            raise ConfigurationError('engine_dispatch.timeout_seconds must be > 0')
        if not isinstance(self.routes, list):
            raise ConfigurationError('engine_dispatch.routes must be a list')
        if self.filter_bridge is not None and not isinstance(self.filter_bridge, FilterBridgeConfig):
            raise ConfigurationError('engine_dispatch.filter_bridge must be a FilterBridgeConfig or omitted')

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'EngineDispatchConfig':
        _require(d, ['enabled', 'timeout_seconds', 'routes'], 'signals_platform_cln.engine_dispatch')
        routes = [EngineRouteConfig.from_dict(r) for r in (d['routes'] or [])]
        filter_bridge: Optional[FilterBridgeConfig] = None
        if d.get('filter_bridge') is not None:
            filter_bridge = FilterBridgeConfig.from_dict(d['filter_bridge'])
        return cls(
            enabled=bool(d['enabled']),
            timeout_seconds=float(d['timeout_seconds']),
            routes=routes,
            filter_bridge=filter_bridge,
        )


@dataclass
class AnalyticsCapabilitiesConfig:
    """Top-level bundle for all extended analytics capability configs.

    Each sub-config has its own ``enabled`` toggle so individual capabilities
    can be rolled out independently without touching the parent YAML path.

    :param time_windows: TimeWindowAnalyticsConfig
    :param anomaly_detection: AnomalyDetectionConfig
    :param growth_tracking: GrowthTrackingConfig
    :param forecasting: ForecastingConfig
    :param ranking_scoring: RankingScoringConfig
    :param historical_snapshots: HistoricalSnapshotConfig
    :param domain_analytics: Optional[DomainAnalyticsConfig] - Domain-specific engine config;
        None when the ``domain_analytics`` YAML block is omitted (engine disabled)
    """
    time_windows: TimeWindowAnalyticsConfig
    anomaly_detection: AnomalyDetectionConfig
    growth_tracking: GrowthTrackingConfig
    forecasting: ForecastingConfig
    ranking_scoring: RankingScoringConfig
    historical_snapshots: HistoricalSnapshotConfig
    domain_analytics: Optional['DomainAnalyticsConfig'] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'AnalyticsCapabilitiesConfig':
        _require(d, ['time_windows', 'anomaly_detection', 'growth_tracking', 'forecasting', 'ranking_scoring', 'historical_snapshots'], 'signals_platform_cln.capabilities')
        domain_analytics: Optional[DomainAnalyticsConfig] = None
        if 'domain_analytics' in d and d['domain_analytics'] is not None:
            domain_analytics = DomainAnalyticsConfig.from_dict(d['domain_analytics'])
        return cls(
            time_windows=TimeWindowAnalyticsConfig.from_dict(d['time_windows']),
            anomaly_detection=AnomalyDetectionConfig.from_dict(d['anomaly_detection']),
            growth_tracking=GrowthTrackingConfig.from_dict(d['growth_tracking']),
            forecasting=ForecastingConfig.from_dict(d['forecasting']),
            ranking_scoring=RankingScoringConfig.from_dict(d['ranking_scoring']),
            historical_snapshots=HistoricalSnapshotConfig.from_dict(d['historical_snapshots']),
            domain_analytics=domain_analytics,
        )


__all__ = [
    'AnalyticsCapabilitiesConfig',
    'AnalyticsConfig',
    'AnomalyDetectionConfig',
    'ClickHouseClientConfig',
    'ClickHouseMvFreshnessProbeConfig',
    'ClickHouseReadinessProbeConfig',
    'DomainAnalyticsConfig',
    'EngineDispatchConfig',
    'EngineRouteConfig',
    'FilterBridgeConfig',
    'ForecastingConfig',
    'GrowthTrackingConfig',
    'HistoricalSnapshotConfig',
    'MVRouterConfig',
    'MaterializedViewConfig',
    'MultiPeriodConfig',
    'NLSqlExactCacheConfig',
    'RankingScoringConfig',
    'SignalWriterConfig',
    'TimeWindowAnalyticsConfig',
]

