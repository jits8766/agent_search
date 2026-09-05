"""Parameterized SQL template functions for ClickHouse analytics.

Each function returns a single SQL string that can be passed directly into
the existing AstSecurityValidator → MVRouter → ClickHouseExecutor pipeline
unchanged.  Templates never accept raw user strings for identifiers — all
column and table names are validated with ``_validate_identifier`` before
interpolation.  Filter *values* for dates and numbers are type-validated;
equality / IN values are restricted to printable ASCII without SQL special
characters.

ClickHouse-specific functions used throughout:
  toStartOfHour / toStartOfDay / toStartOfWeek / toStartOfMonth — time grain
  lagInFrame                                                    — window lag
  stddevPop / avg                                               — rolling stats
  uniqExact                                                     — exact unique count
  simpleLinearRegression                                        — forecast slope/intercept
  multiIf                                                       — multi-branch conditional
  dateDiff                                                      — age in time units
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

__all__ = [
    'TimeGrain',
    'FilterSpec',
    'DateRangeFilter',
    'EqualityFilter',
    'NumericRangeFilter',
    'InFilter',
    'growth_rate_sql',
    'anomaly_detection_sql',
    'category_trend_sql',
    'historical_comparison_sql',
    'time_window_sql',
    'user_engagement_sql',
    'realtime_trending_sql',
    'lifecycle_sql',
    'comparative_cohort_sql',
    'ranking_score_sql',
    'forecast_sql',
    'keyword_trend_sql',
    'multidim_aggregation_sql',
    'sell_through_rate_sql',
    'seo_bucket_sql',
    'bid_distribution_sql',
    'market_summary_sql',
    'price_distribution_sql',
    'window_funnel_sql',
    'top_k_sql',
    'moving_average_sql',
    'bid_velocity_acceleration_sql',
    'watch_bid_conversion_sql',
    'hold_time_distribution_sql',
    'counter_offer_sql',
    'price_realization_sql',
    'buyer_retention_cohorts_sql',
    'auction_timing_patterns_sql',
    'name_structure_sql',
    'cross_tld_spread_sql',
    'search_attribution_sql',
    'registrar_hhi_sql',
    'comparable_sale_price_sql',
]

_IDENTIFIER_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_.]*$')
_SAFE_VALUE_RE = re.compile(r"^[A-Za-z0-9 _\-.:/@]+$")
_DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?$')


def _validate_identifier(name: str, param: str) -> str:
    """Raise ValueError if ``name`` is not a safe SQL identifier."""
    if not isinstance(name, str) or not _IDENTIFIER_RE.match(name):
        raise ValueError(f"{param} must be a valid SQL identifier (letters/digits/underscores/dots), got {name!r}")
    return name


def _validate_date(value: str, param: str) -> str:
    """Raise ValueError if ``value`` is not a recognisable date/datetime literal."""
    if not isinstance(value, str) or not _DATE_RE.match(value.strip()):
        raise ValueError(f"{param} must be a date string like 'YYYY-MM-DD', got {value!r}")
    return value.strip()


def _validate_safe_value(value: str, param: str) -> str:
    """Raise ValueError if ``value`` contains characters that could escape a SQL string literal."""
    if not isinstance(value, str) or not _SAFE_VALUE_RE.match(value):
        raise ValueError(f"{param} contains unsafe characters. Only letters, digits, spaces, and _-.:/@ are allowed, got {value!r}")
    return value


# ---------------------------------------------------------------------------
# TimeGrain
# ---------------------------------------------------------------------------

class TimeGrain(str, Enum):
    """Supported time aggregation granularities."""
    HOUR = 'hour'
    DAY = 'day'
    WEEK = 'week'
    MONTH = 'month'


_GRAIN_FN: Dict[TimeGrain, str] = {
    TimeGrain.HOUR: 'toStartOfHour',
    TimeGrain.DAY: 'toStartOfDay',
    TimeGrain.WEEK: 'toStartOfWeek',
    TimeGrain.MONTH: 'toStartOfMonth',
}


# ---------------------------------------------------------------------------
# FilterSpec hierarchy — typed, injection-safe filter representations
# ---------------------------------------------------------------------------

class FilterSpec:
    """Base class for typed SQL filter fragments."""
    def to_sql(self) -> str:  # pragma: no cover
        raise NotImplementedError


@dataclass
class DateRangeFilter(FilterSpec):
    """``column >= start AND column <= end`` using validated date literals."""
    column: str
    start: str
    end: str

    def __post_init__(self) -> None:
        _validate_identifier(self.column, 'DateRangeFilter.column')
        _validate_date(self.start, 'DateRangeFilter.start')
        _validate_date(self.end, 'DateRangeFilter.end')

    def to_sql(self) -> str:
        return f"{self.column} >= '{self.start}' AND {self.column} <= '{self.end}'"


@dataclass
class EqualityFilter(FilterSpec):
    """``column = 'value'`` using a safe printable-ASCII value."""
    column: str
    value: str

    def __post_init__(self) -> None:
        _validate_identifier(self.column, 'EqualityFilter.column')
        _validate_safe_value(self.value, 'EqualityFilter.value')

    def to_sql(self) -> str:
        return f"{self.column} = '{self.value}'"


@dataclass
class NumericRangeFilter(FilterSpec):
    """Numeric bound filter: BETWEEN (both bounds) or >= / <= (one bound).

    At least one of min_value / max_value must be provided.
    """
    column: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    def __post_init__(self) -> None:
        _validate_identifier(self.column, 'NumericRangeFilter.column')
        if self.min_value is None and self.max_value is None:
            raise ValueError('NumericRangeFilter requires at least one of min_value or max_value')
        if self.min_value is not None and not isinstance(self.min_value, (int, float)):
            raise ValueError('NumericRangeFilter.min_value must be numeric')
        if self.max_value is not None and not isinstance(self.max_value, (int, float)):
            raise ValueError('NumericRangeFilter.max_value must be numeric')
        if self.min_value is not None and self.max_value is not None and float(self.min_value) > float(self.max_value):
            raise ValueError('NumericRangeFilter.min_value must be <= max_value')

    def to_sql(self) -> str:
        if self.min_value is not None and self.max_value is not None:
            return f"{self.column} BETWEEN {self.min_value!r} AND {self.max_value!r}"
        if self.min_value is not None:
            return f"{self.column} >= {self.min_value!r}"
        return f"{self.column} <= {self.max_value!r}"


@dataclass
class InFilter(FilterSpec):
    """``column IN ('a', 'b', ...)`` using a validated value list."""
    column: str
    values: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _validate_identifier(self.column, 'InFilter.column')
        if not self.values:
            raise ValueError('InFilter.values must be non-empty')
        for v in self.values:
            _validate_safe_value(v, 'InFilter.values entry')

    def to_sql(self) -> str:
        quoted = ', '.join(f"'{v}'" for v in self.values)
        return f"{self.column} IN ({quoted})"


def _build_where(filters: Optional[List[FilterSpec]]) -> str:
    """Render a WHERE clause from a filter list; empty string when None/empty."""
    if not filters:
        return ''
    parts = [f.to_sql() for f in filters]
    return 'WHERE ' + ' AND '.join(parts)


# ---------------------------------------------------------------------------
# Template functions
# ---------------------------------------------------------------------------

def growth_rate_sql(table: str, metric_column: str, entity_column: str, time_column: str, grain: TimeGrain = TimeGrain.DAY, lookback_periods: int = 7, limit: int = 50, filters: Optional[List[FilterSpec]] = None, descending: bool = True) -> str:
    """Fastest growing / declining entities by period-over-period metric change.

    Uses ``lagInFrame`` window function to compare each row's value against the
    immediately preceding period within the same entity partition.

    :param table: Source table name (fully-qualified allowed, e.g. ``db.tbl``)
    :param metric_column: Numeric column to aggregate (``sum``)
    :param entity_column: Column that identifies the entity (e.g. domain / tld)
    :param time_column: Timestamp or date column for time bucketing
    :param grain: Time aggregation granularity
    :param lookback_periods: Minimum number of periods required (used in comment)
    :param limit: Row cap on the result
    :param filters: Optional typed filter list applied before aggregation
    :param descending: True = top growers first; False = top decliners first
    :return: ClickHouse SQL string
    """
    _validate_identifier(table, 'table')
    _validate_identifier(metric_column, 'metric_column')
    _validate_identifier(entity_column, 'entity_column')
    _validate_identifier(time_column, 'time_column')
    if int(limit) < 1:
        raise ValueError('limit must be >= 1')
    grain_fn = _GRAIN_FN[grain]
    where = _build_where(filters)
    order_dir = 'DESC' if descending else 'ASC'
    return f"""\
WITH base AS (
    SELECT
        {entity_column},
        {grain_fn}({time_column}) AS period,
        sum({metric_column}) AS metric_value
    FROM {table}
    {where}
    GROUP BY {entity_column}, period
),
lagged AS (
    SELECT
        {entity_column},
        period,
        metric_value,
        lagInFrame(metric_value, 1, 0) OVER (
            PARTITION BY {entity_column}
            ORDER BY period ASC
        ) AS prev_value
    FROM base
)
SELECT
    {entity_column},
    period,
    metric_value AS current_value,
    prev_value AS previous_value,
    if(
        prev_value > 0,
        (metric_value - prev_value) / prev_value * 100.0,
        NULL
    ) AS growth_pct
FROM lagged
WHERE prev_value > 0
ORDER BY growth_pct {order_dir} NULLS LAST
LIMIT {limit}"""


def anomaly_detection_sql(table: str, metric_column: str, entity_column: str, time_column: str, grain: TimeGrain = TimeGrain.DAY, z_threshold: float = 3.0, window_periods: int = 30, limit: int = 200, filters: Optional[List[FilterSpec]] = None) -> str:
    """Spike and anomaly detection using a rolling Z-score.

    Computes ``rolling_avg`` and ``rolling_std`` over the preceding
    ``window_periods`` rows (per entity) then flags rows where
    ``|z_score| > z_threshold``.

    :param z_threshold: Standard-deviation multiplier for the anomaly gate (> 0)
    :param window_periods: Rolling window size in periods (>= 2)
    """
    _validate_identifier(table, 'table')
    _validate_identifier(metric_column, 'metric_column')
    _validate_identifier(entity_column, 'entity_column')
    _validate_identifier(time_column, 'time_column')
    if float(z_threshold) <= 0:
        raise ValueError('z_threshold must be > 0')
    if int(window_periods) < 2:
        raise ValueError('window_periods must be >= 2')
    grain_fn = _GRAIN_FN[grain]
    where = _build_where(filters)
    return f"""\
WITH aggregated AS (
    SELECT
        {entity_column},
        {grain_fn}({time_column}) AS period,
        sum({metric_column}) AS metric_value
    FROM {table}
    {where}
    GROUP BY {entity_column}, period
),
with_stats AS (
    SELECT
        {entity_column},
        period,
        metric_value,
        avg(metric_value) OVER (
            PARTITION BY {entity_column}
            ORDER BY period ASC
            ROWS BETWEEN {int(window_periods)} PRECEDING AND CURRENT ROW
        ) AS rolling_avg,
        stddevPop(metric_value) OVER (
            PARTITION BY {entity_column}
            ORDER BY period ASC
            ROWS BETWEEN {int(window_periods)} PRECEDING AND CURRENT ROW
        ) AS rolling_std
    FROM aggregated
),
scored AS (
    SELECT
        {entity_column},
        period,
        metric_value,
        rolling_avg,
        rolling_std,
        if(rolling_std > 0, (metric_value - rolling_avg) / rolling_std, 0.0) AS z_score
    FROM with_stats
)
SELECT
    {entity_column},
    period,
    metric_value,
    rolling_avg,
    rolling_std,
    z_score,
    abs(z_score) > {float(z_threshold)!r} AS is_anomaly
FROM scored
WHERE abs(z_score) > {float(z_threshold)!r}
ORDER BY abs(z_score) DESC
LIMIT {limit}"""


def category_trend_sql(table: str, metric_column: str, category_column: str, time_column: str, grain: TimeGrain = TimeGrain.DAY, lookback_periods: int = 30, extra_group_columns: Optional[List[str]] = None, limit: int = 500, filters: Optional[List[FilterSpec]] = None) -> str:
    """Category / region-wise trend aggregation over time windows.

    Groups by ``category_column`` + optional ``extra_group_columns`` (e.g. tld,
    region) and a time bucket to produce a multi-dimensional trend table.

    :param extra_group_columns: Additional dimension columns to include in GROUP BY
    """
    _validate_identifier(table, 'table')
    _validate_identifier(metric_column, 'metric_column')
    _validate_identifier(category_column, 'category_column')
    _validate_identifier(time_column, 'time_column')
    extra_cols: List[str] = []
    for col in (extra_group_columns or []):
        extra_cols.append(_validate_identifier(col, 'extra_group_columns entry'))
    grain_fn = _GRAIN_FN[grain]
    where = _build_where(filters)
    all_dims = [category_column] + extra_cols
    select_dims = ',\n        '.join(all_dims)
    group_dims = ', '.join(all_dims)
    return f"""\
SELECT
    {select_dims},
    {grain_fn}({time_column}) AS period,
    count() AS event_count,
    sum({metric_column}) AS total_metric,
    avg({metric_column}) AS avg_metric,
    max({metric_column}) AS max_metric,
    min({metric_column}) AS min_metric,
    uniqExact({category_column}) AS unique_categories
FROM {table}
{where}
GROUP BY {group_dims}, period
ORDER BY {category_column} ASC, period ASC
LIMIT {limit}"""


def historical_comparison_sql(table: str, metric_column: str, group_column: str, time_column: str, current_start: str, current_end: str, prior_start: str, prior_end: str, limit: int = 100, filters: Optional[List[FilterSpec]] = None) -> str:
    """Compare a metric between a current period and a historical reference period.

    Returns one row per ``group_column`` value with both period aggregates
    and an absolute / relative delta.

    :param current_start: Start of current window ('YYYY-MM-DD')
    :param current_end: End of current window ('YYYY-MM-DD')
    :param prior_start: Start of prior window ('YYYY-MM-DD')
    :param prior_end: End of prior window ('YYYY-MM-DD')
    """
    _validate_identifier(table, 'table')
    _validate_identifier(metric_column, 'metric_column')
    _validate_identifier(group_column, 'group_column')
    _validate_identifier(time_column, 'time_column')
    _validate_date(current_start, 'current_start')
    _validate_date(current_end, 'current_end')
    _validate_date(prior_start, 'prior_start')
    _validate_date(prior_end, 'prior_end')
    extra_where = (' AND ' + _build_where(filters).replace('WHERE ', '', 1)) if filters else ''
    return f"""\
WITH current_period AS (
    SELECT
        {group_column},
        sum({metric_column}) AS current_metric,
        count() AS current_count
    FROM {table}
    WHERE {time_column} >= '{current_start}' AND {time_column} <= '{current_end}'{extra_where}
    GROUP BY {group_column}
),
prior_period AS (
    SELECT
        {group_column},
        sum({metric_column}) AS prior_metric,
        count() AS prior_count
    FROM {table}
    WHERE {time_column} >= '{prior_start}' AND {time_column} <= '{prior_end}'{extra_where}
    GROUP BY {group_column}
)
SELECT
    coalesce(c.{group_column}, p.{group_column}) AS {group_column},
    c.current_metric,
    p.prior_metric,
    c.current_count,
    p.prior_count,
    c.current_metric - p.prior_metric AS absolute_delta,
    if(
        p.prior_metric > 0,
        (c.current_metric - p.prior_metric) / p.prior_metric * 100.0,
        NULL
    ) AS relative_delta_pct
FROM current_period AS c
FULL OUTER JOIN prior_period AS p
    ON c.{group_column} = p.{group_column}
ORDER BY abs(absolute_delta) DESC NULLS LAST
LIMIT {limit}"""


def time_window_sql(table: str, metric_columns: List[str], group_columns: List[str], time_column: str, grain: TimeGrain = TimeGrain.DAY, lookback_periods: int = 7, limit: int = 500, filters: Optional[List[FilterSpec]] = None) -> str:
    """Multi-level aggregation over configurable time windows.

    Supports arbitrary GROUP BY dimensions and multiple metric columns,
    each producing ``sum_``, ``avg_``, and ``max_`` projections.

    :param metric_columns: One or more numeric columns to aggregate
    :param group_columns: Dimension columns for GROUP BY
    :param lookback_periods: Informational (used in comment only — add a
        DateRangeFilter to actually constrain the window)
    """
    if not metric_columns:
        raise ValueError('time_window_sql requires at least one metric_column')
    if not group_columns:
        raise ValueError('time_window_sql requires at least one group_column')
    _validate_identifier(table, 'table')
    _validate_identifier(time_column, 'time_column')
    validated_metrics = [_validate_identifier(c, 'metric_columns entry') for c in metric_columns]
    validated_groups = [_validate_identifier(c, 'group_columns entry') for c in group_columns]
    grain_fn = _GRAIN_FN[grain]
    where = _build_where(filters)
    metric_projections = []
    for col in validated_metrics:
        metric_projections += [
            f'sum({col}) AS sum_{col}',
            f'avg({col}) AS avg_{col}',
            f'max({col}) AS max_{col}',
        ]
    select_dims = ',\n    '.join(validated_groups)
    select_metrics = ',\n    '.join(metric_projections)
    group_dims = ', '.join(validated_groups)
    return f"""\
SELECT
    {select_dims},
    {grain_fn}({time_column}) AS period,
    count() AS event_count,
    {select_metrics}
FROM {table}
{where}
GROUP BY {group_dims}, period
ORDER BY period DESC, event_count DESC
LIMIT {limit}"""


def user_engagement_sql(table: str, user_column: str, event_column: str, time_column: str, grain: TimeGrain = TimeGrain.DAY, lookback_periods: int = 30, limit: int = 200, filters: Optional[List[FilterSpec]] = None) -> str:
    """User behavior and engagement insights: active users, event depth, session count.

    Produces one row per time bucket with active user count (DAU/WAU), total
    events, events-per-user, and unique event type count.

    :param user_column: Column identifying the user (e.g. ``user_id``)
    :param event_column: Column carrying the event / action type
    """
    _validate_identifier(table, 'table')
    _validate_identifier(user_column, 'user_column')
    _validate_identifier(event_column, 'event_column')
    _validate_identifier(time_column, 'time_column')
    grain_fn = _GRAIN_FN[grain]
    where = _build_where(filters)
    return f"""\
SELECT
    {grain_fn}({time_column}) AS period,
    uniqExact({user_column}) AS active_users,
    count() AS total_events,
    count() / uniqExact({user_column}) AS events_per_user,
    uniqExact({event_column}) AS unique_event_types,
    avg(count()) OVER (
        ORDER BY {grain_fn}({time_column}) ASC
        ROWS BETWEEN {int(lookback_periods)} PRECEDING AND CURRENT ROW
    ) AS rolling_avg_events
FROM {table}
{where}
GROUP BY period
ORDER BY period DESC
LIMIT {limit}"""


def realtime_trending_sql(table: str, metric_column: str, entity_column: str, time_column: str, window_minutes: int = 60, limit: int = 20, filters: Optional[List[FilterSpec]] = None) -> str:
    """Real-time trending: entities ranked by activity in the last N minutes.

    Scans only the hot window (``now() - INTERVAL window_minutes MINUTE``) so
    the query is cheap enough for sub-second latency on a small ClickHouse MV.

    :param window_minutes: Lookback window in minutes (>= 1)
    """
    _validate_identifier(table, 'table')
    _validate_identifier(metric_column, 'metric_column')
    _validate_identifier(entity_column, 'entity_column')
    _validate_identifier(time_column, 'time_column')
    if int(window_minutes) < 1:
        raise ValueError('window_minutes must be >= 1')
    where = _build_where(filters)
    hot_filter = f"{time_column} >= now() - INTERVAL {int(window_minutes)} MINUTE"
    combined_where = (
        f'WHERE {hot_filter}'
        if not filters
        else f'{_build_where(filters)} AND {hot_filter}'
    )
    return f"""\
SELECT
    {entity_column},
    count() AS event_count,
    sum({metric_column}) AS total_metric,
    max({metric_column}) AS peak_metric,
    min({time_column}) AS first_seen,
    max({time_column}) AS last_seen,
    dateDiff('second', min({time_column}), max({time_column})) AS activity_span_seconds
FROM {table}
{combined_where}
GROUP BY {entity_column}
ORDER BY event_count DESC
LIMIT {limit}"""


def lifecycle_sql(table: str, entity_column: str, created_column: str, activity_column: str, time_column: str, new_days: int = 7, dormant_days: int = 30, grain: TimeGrain = TimeGrain.DAY, limit: int = 500, filters: Optional[List[FilterSpec]] = None) -> str:
    """Lifecycle analytics: entity age distribution and phase classification.

    Classifies each entity as ``new`` (seen <= new_days ago), ``active``
    (recent activity), ``dormant`` (no activity in dormant_days), or
    ``churned`` (no activity beyond 2x dormant_days).

    :param new_days: Age threshold (days) for the 'new' phase
    :param dormant_days: Inactivity threshold (days) for the 'dormant' phase
    """
    _validate_identifier(table, 'table')
    _validate_identifier(entity_column, 'entity_column')
    _validate_identifier(created_column, 'created_column')
    _validate_identifier(activity_column, 'activity_column')
    _validate_identifier(time_column, 'time_column')
    grain_fn = _GRAIN_FN[grain]
    where = _build_where(filters)
    churned_days = int(dormant_days) * 2
    return f"""\
WITH entity_stats AS (
    SELECT
        {entity_column},
        min({created_column}) AS first_seen,
        max({activity_column}) AS last_activity,
        count() AS total_events,
        dateDiff('day', min({created_column}), now()) AS age_days,
        dateDiff('day', max({activity_column}), now()) AS days_since_active
    FROM {table}
    {where}
    GROUP BY {entity_column}
)
SELECT
    {entity_column},
    first_seen,
    last_activity,
    total_events,
    age_days,
    days_since_active,
    multiIf(
        age_days <= {int(new_days)}, 'new',
        days_since_active <= {int(new_days)}, 'active',
        days_since_active <= {int(dormant_days)}, 'dormant',
        days_since_active <= {churned_days}, 'at_risk',
        'churned'
    ) AS lifecycle_stage,
    {grain_fn}(last_activity) AS last_active_bucket
FROM entity_stats
ORDER BY age_days ASC, last_activity DESC
LIMIT {limit}"""


def comparative_cohort_sql(table: str, metric_column: str, cohort_column: str, time_column: str, cohort_a: str, cohort_b: str, grain: TimeGrain = TimeGrain.DAY, lookback_periods: int = 30, filters: Optional[List[FilterSpec]] = None) -> str:
    """Side-by-side metric comparison between two named cohorts over time.

    Returns one row per period with columns for each cohort's ``sum`` and
    ``count``, plus absolute and relative deltas (A − B).

    :param cohort_a: Value of ``cohort_column`` for cohort A
    :param cohort_b: Value of ``cohort_column`` for cohort B
    """
    _validate_identifier(table, 'table')
    _validate_identifier(metric_column, 'metric_column')
    _validate_identifier(cohort_column, 'cohort_column')
    _validate_identifier(time_column, 'time_column')
    _validate_safe_value(cohort_a, 'cohort_a')
    _validate_safe_value(cohort_b, 'cohort_b')
    grain_fn = _GRAIN_FN[grain]
    extra_where = (' AND ' + _build_where(filters).replace('WHERE ', '', 1)) if filters else ''
    return f"""\
WITH cohort_a AS (
    SELECT
        {grain_fn}({time_column}) AS period,
        sum({metric_column}) AS metric_a,
        count() AS count_a
    FROM {table}
    WHERE {cohort_column} = '{cohort_a}'{extra_where}
    GROUP BY period
),
cohort_b AS (
    SELECT
        {grain_fn}({time_column}) AS period,
        sum({metric_column}) AS metric_b,
        count() AS count_b
    FROM {table}
    WHERE {cohort_column} = '{cohort_b}'{extra_where}
    GROUP BY period
)
SELECT
    coalesce(a.period, b.period) AS period,
    a.metric_a,
    b.metric_b,
    a.count_a,
    b.count_b,
    a.metric_a - b.metric_b AS absolute_delta,
    if(
        b.metric_b > 0,
        (a.metric_a - b.metric_b) / b.metric_b * 100.0,
        NULL
    ) AS relative_delta_pct
FROM cohort_a AS a
FULL OUTER JOIN cohort_b AS b ON a.period = b.period
ORDER BY period ASC"""


def ranking_score_sql(table: str, entity_column: str, score_weights: Dict[str, float], time_column: str, grain: TimeGrain = TimeGrain.DAY, lookback_periods: int = 7, limit: int = 100, filters: Optional[List[FilterSpec]] = None) -> str:
    """Custom composite ranking with per-metric weights.

    Each column in ``score_weights`` is normalised within the result set using
    min-max scaling before the weighted sum is computed, so metrics with
    different natural scales contribute proportionally to their weight.

    :param score_weights: ``{metric_column: weight}`` — all columns validated,
        all weights must be > 0, total need not equal 1
    """
    _validate_identifier(table, 'table')
    _validate_identifier(entity_column, 'entity_column')
    _validate_identifier(time_column, 'time_column')
    if not score_weights:
        raise ValueError('ranking_score_sql requires at least one score_weight entry')
    validated_weights: List[Tuple[str, float]] = []
    for col, w in score_weights.items():
        validated_weights.append((_validate_identifier(col, f'score_weights key {col!r}'), float(w)))
        if float(w) <= 0:
            raise ValueError(f'score_weights[{col!r}] must be > 0, got {w}')
    grain_fn = _GRAIN_FN[grain]
    where = _build_where(filters)

    agg_select = ',\n        '.join(f'sum({col}) AS raw_{col}' for col, _ in validated_weights)
    norm_select_parts = []
    for col, _ in validated_weights:
        norm_select_parts.append(
            f'if(max(raw_{col}) OVER () - min(raw_{col}) OVER () > 0, '
            f'(raw_{col} - min(raw_{col}) OVER ()) / (max(raw_{col}) OVER () - min(raw_{col}) OVER ()), '
            f'0.0) AS norm_{col}'
        )
    norm_select = ',\n        '.join(norm_select_parts)

    score_terms = ' + '.join(f'{w!r} * norm_{col}' for col, w in validated_weights)
    return f"""\
WITH aggregated AS (
    SELECT
        {entity_column},
        {grain_fn}({time_column}) AS period,
        {agg_select}
    FROM {table}
    {where}
    GROUP BY {entity_column}, period
),
normalised AS (
    SELECT
        {entity_column},
        period,
        {', '.join(f'raw_{col}' for col, _ in validated_weights)},
        {norm_select}
    FROM aggregated
)
SELECT
    {entity_column},
    period,
    {', '.join(f'raw_{col}' for col, _ in validated_weights)},
    {score_terms} AS composite_score
FROM normalised
ORDER BY composite_score DESC
LIMIT {limit}"""


def forecast_sql(table: str, metric_column: str, entity_column: str, time_column: str, grain: TimeGrain = TimeGrain.DAY, history_periods: int = 30, forecast_periods: int = 7, limit: int = 100, filters: Optional[List[FilterSpec]] = None) -> str:
    """Simple linear trend forecast using ClickHouse regression functions.

    Fits a linear model ``y = slope * t + intercept`` to the historical data
    (one row per entity) and returns the projected value ``forecast_periods``
    periods ahead as ``forecast_value``.  The ``forecast_quality`` column
    (Pearson correlation) indicates model fit.

    :param history_periods: Number of past periods included in the regression
    :param forecast_periods: How many periods to project forward
    """
    _validate_identifier(table, 'table')
    _validate_identifier(metric_column, 'metric_column')
    _validate_identifier(entity_column, 'entity_column')
    _validate_identifier(time_column, 'time_column')
    if int(history_periods) < 7:
        raise ValueError('history_periods must be >= 7 for a meaningful regression')
    if int(forecast_periods) < 1:
        raise ValueError('forecast_periods must be >= 1')
    grain_fn = _GRAIN_FN[grain]
    where = _build_where(filters)
    return f"""\
WITH history AS (
    SELECT
        {entity_column},
        {grain_fn}({time_column}) AS period,
        sum({metric_column}) AS metric_value
    FROM {table}
    {where}
    GROUP BY {entity_column}, period
    ORDER BY {entity_column} ASC, period ASC
    LIMIT {int(history_periods)} BY {entity_column}
),
regression AS (
    SELECT
        {entity_column},
        count() AS data_points,
        simpleLinearRegression(toUnixTimestamp(period), metric_value) AS model,
        max(period) AS latest_period,
        corr(toUnixTimestamp(period), metric_value) AS forecast_quality
    FROM history
    GROUP BY {entity_column}
    HAVING data_points >= 3
)
SELECT
    {entity_column},
    latest_period,
    data_points,
    round(forecast_quality, 4) AS forecast_quality,
    model.1 AS slope,
    model.2 AS intercept,
    round(
        model.1 * toUnixTimestamp(
            latest_period + INTERVAL {int(forecast_periods)} {grain.value.upper()}
        ) + model.2,
        2
    ) AS forecast_value
FROM regression
ORDER BY abs(forecast_quality) DESC
LIMIT {limit}"""


def keyword_trend_sql(table: str, domain_column: str, time_column: str, grain: TimeGrain = TimeGrain.WEEK, min_frequency: int = 3, lookback_periods: int = 52, limit: int = 100, filters: Optional[List[FilterSpec]] = None) -> str:
    """Keyword frequency trends extracted from domain names via ARRAY JOIN.

    Splits the SLD portion of each domain name on ``-`` to yield compound-word
    tokens, then aggregates by token + time bucket.  Tokens shorter than 3
    characters are dropped.

    Example: ``'cool-brand.com'`` → tokens ``['cool', 'brand']``

    Uses ClickHouse ``ARRAY JOIN`` for row-level expansion — the result has one
    row per (keyword, period) combination, not one row per domain.

    :param domain_column: Column containing fully-qualified domain names
    :param min_frequency: Minimum occurrence count per (keyword, period) bucket
    :param lookback_periods: Informational only; add a DateRangeFilter to constrain
    """
    _validate_identifier(table, 'table')
    _validate_identifier(domain_column, 'domain_column')
    _validate_identifier(time_column, 'time_column')
    if int(min_frequency) < 1:
        raise ValueError('min_frequency must be >= 1')
    if int(limit) < 1:
        raise ValueError('limit must be >= 1')
    grain_fn = _GRAIN_FN[grain]
    where = _build_where(filters)
    # Extract SLD: arrayElement(splitByChar('.', col), -2) gives second-to-last label.
    # Then split SLD on '-' to get compound-word tokens via ARRAY JOIN.
    sld_expr = f"arrayElement(splitByChar('.', lower({domain_column})), -2)"
    return f"""\
SELECT
    keyword,
    {grain_fn}({time_column}) AS period,
    count() AS frequency,
    uniqExact({domain_column}) AS unique_domains,
    avg(current_price) AS avg_price
FROM {table}
{where}
ARRAY JOIN splitByChar('-', {sld_expr}) AS keyword
WHERE length(keyword) >= 3
GROUP BY keyword, period
HAVING frequency >= {int(min_frequency)}
ORDER BY frequency DESC
LIMIT {limit}"""


def sell_through_rate_sql(table: str, dimension_column: str, sold_flag_column: str, time_column: str, price_column: str, grain: TimeGrain = TimeGrain.DAY, lookback_periods: int = 30, limit: int = 500, filters: Optional[List[FilterSpec]] = None) -> str:
    """Sell-through rate (sold / total) with avg sale price grouped by dimension and time period.

    Answers Q6 (overall sell-through), Q12 (TLD sell-through comparison),
    Q29 (auction success rate). Also used for registrar and category-name breakdowns.

    :param dimension_column: Column to group by (e.g. tld, registrar_name, category_name)
    :param sold_flag_column: UInt8 column that is 1 when sold (e.g. sold_flag)
    :param price_column: Price column for avg/sum aggregates
    :return: Rows with dimension, period, total_auctions, sold_auctions, sell_through_rate,
        avg_sale_price, total_sale_revenue
    """
    _validate_identifier(table, 'table')
    _validate_identifier(dimension_column, 'dimension_column')
    _validate_identifier(sold_flag_column, 'sold_flag_column')
    _validate_identifier(time_column, 'time_column')
    _validate_identifier(price_column, 'price_column')
    grain_fn = _GRAIN_FN[grain]
    where = _build_where(filters)
    return f"""\
SELECT
    {dimension_column},
    {grain_fn}({time_column}) AS period,
    count() AS total_auctions,
    countIf({sold_flag_column} = 1) AS sold_auctions,
    if(count() > 0, countIf({sold_flag_column} = 1) / count(), 0.0) AS sell_through_rate,
    avgIf({price_column}, {sold_flag_column} = 1) AS avg_sale_price,
    sumIf({price_column}, {sold_flag_column} = 1) AS total_sale_revenue
FROM {table}
{where}
GROUP BY {dimension_column}, period
HAVING total_auctions > 0
ORDER BY sell_through_rate DESC, total_auctions DESC
LIMIT {int(limit)}"""


def seo_bucket_sql(table: str, da_column: str, traffic_column: str, price_column: str, sold_flag_column: str, time_column: str, grain: TimeGrain = TimeGrain.MONTH, lookback_periods: int = 3, filters: Optional[List[FilterSpec]] = None) -> str:
    """Domain Authority and traffic bucket vs sale price correlation analysis.

    Buckets domain_authority into DA ranges and monthly_traffic into volume ranges,
    then computes avg listing price, avg sale price, and sell-through rate per
    (da_bucket, traffic_bucket, period) combination.

    Answers Q19 (traffic vs sale price), Q20 (DA vs sale price), Q21 (avg DA of sold),
    Q22 (zero-traffic sold above threshold), Q23 (backlinks vs value), Q24 (SEO trend).

    :return: Rows with da_bucket, traffic_bucket, period, domain_count, sold_count,
        sell_through_rate, avg_listing_price, avg_sale_price, max_price
    """
    _validate_identifier(table, 'table')
    _validate_identifier(da_column, 'da_column')
    _validate_identifier(traffic_column, 'traffic_column')
    _validate_identifier(price_column, 'price_column')
    _validate_identifier(sold_flag_column, 'sold_flag_column')
    _validate_identifier(time_column, 'time_column')
    grain_fn = _GRAIN_FN[grain]
    where = _build_where(filters)
    return f"""\
WITH bucketed AS (
    SELECT
        multiIf(
            {da_column} = 0,    'DA-0',
            {da_column} < 10,   'DA-1-9',
            {da_column} < 30,   'DA-10-29',
            {da_column} < 50,   'DA-30-49',
            {da_column} < 70,   'DA-50-69',
                                'DA-70+'
        ) AS da_bucket,
        multiIf(
            {traffic_column} = 0,       'traffic-0',
            {traffic_column} < 100,     'traffic-1-99',
            {traffic_column} < 1000,    'traffic-100-999',
            {traffic_column} < 10000,   'traffic-1K-9K',
                                        'traffic-10K+'
        ) AS traffic_bucket,
        {grain_fn}({time_column}) AS period,
        {price_column},
        {sold_flag_column}
    FROM {table}
    {where}
)
SELECT
    da_bucket,
    traffic_bucket,
    period,
    count() AS domain_count,
    countIf({sold_flag_column} = 1) AS sold_count,
    if(count() > 0, countIf({sold_flag_column} = 1) / count(), 0.0) AS sell_through_rate,
    avg({price_column}) AS avg_listing_price,
    avgIf({price_column}, {sold_flag_column} = 1) AS avg_sale_price,
    max({price_column}) AS max_price
FROM bucketed
GROUP BY da_bucket, traffic_bucket, period
ORDER BY da_bucket ASC, traffic_bucket ASC, period DESC
LIMIT 1000"""


def bid_distribution_sql(table: str, bid_count_column: str, price_column: str, sold_flag_column: str, time_column: str, grain: TimeGrain = TimeGrain.DAY, lookback_periods: int = 30, filters: Optional[List[FilterSpec]] = None) -> str:
    """Bid count distribution analysis: auctions bucketed by bid count, with win rate and price.

    Answers Q25 (avg bids per auction), Q26 (zero-bid auctions), Q27 (bid count vs
    final price correlation), Q30 (avg winning bid amount).

    :return: Rows with bid_bucket, period, auction_count, sold_count, win_rate,
        avg_bids_in_bucket, avg_listing_price, avg_winning_price, max_price
    """
    _validate_identifier(table, 'table')
    _validate_identifier(bid_count_column, 'bid_count_column')
    _validate_identifier(price_column, 'price_column')
    _validate_identifier(sold_flag_column, 'sold_flag_column')
    _validate_identifier(time_column, 'time_column')
    grain_fn = _GRAIN_FN[grain]
    where = _build_where(filters)
    return f"""\
WITH bucketed AS (
    SELECT
        multiIf(
            {bid_count_column} = 0,     '0-bids',
            {bid_count_column} = 1,     '1-bid',
            {bid_count_column} <= 3,    '2-3-bids',
            {bid_count_column} <= 10,   '4-10-bids',
            {bid_count_column} <= 25,   '11-25-bids',
                                        '26-plus-bids'
        ) AS bid_bucket,
        {bid_count_column},
        {price_column},
        {sold_flag_column},
        {grain_fn}({time_column}) AS period
    FROM {table}
    {where}
)
SELECT
    bid_bucket,
    period,
    count() AS auction_count,
    countIf({sold_flag_column} = 1) AS sold_count,
    if(count() > 0, countIf({sold_flag_column} = 1) / count(), 0.0) AS win_rate,
    round(avg({bid_count_column}), 2) AS avg_bids_in_bucket,
    avg({price_column}) AS avg_listing_price,
    avgIf({price_column}, {sold_flag_column} = 1) AS avg_winning_price,
    max({price_column}) AS max_price
FROM bucketed
GROUP BY bid_bucket, period
ORDER BY avg({bid_count_column}) ASC, period DESC
LIMIT 500"""


def market_summary_sql(table: str, price_column: str, bid_count_column: str, sold_flag_column: str, domain_authority_column: str, traffic_column: str, time_column: str, lookback_days: int = 30, filters: Optional[List[FilterSpec]] = None) -> str:
    """Consolidated market KPI summary — single-row overview of all top-level metrics.

    Answers Q82 (show me market stats summary). Also useful as a health-check
    widget that surfaces the most important marketplace numbers in one query.

    :param lookback_days: Number of past days to include (applied as WHERE time_column >= today() - N)
    :return: One row with total_listings, total_sold, overall_sell_through_rate,
        avg/median/max sale price, total sale volume, avg/max bids, zero_bid_auctions,
        avg DA, avg sold DA, avg traffic, zero_traffic_listings, as_of timestamp
    """
    _validate_identifier(table, 'table')
    _validate_identifier(price_column, 'price_column')
    _validate_identifier(bid_count_column, 'bid_count_column')
    _validate_identifier(sold_flag_column, 'sold_flag_column')
    _validate_identifier(domain_authority_column, 'domain_authority_column')
    _validate_identifier(traffic_column, 'traffic_column')
    _validate_identifier(time_column, 'time_column')
    if int(lookback_days) < 1:
        raise ValueError('lookback_days must be >= 1')
    extra_where = (' AND ' + _build_where(filters).replace('WHERE ', '', 1)) if filters else ''
    return f"""\
SELECT
    count() AS total_listings,
    countIf({sold_flag_column} = 1) AS total_sold,
    if(count() > 0, countIf({sold_flag_column} = 1) / count(), 0.0) AS overall_sell_through_rate,
    avgIf({price_column}, {sold_flag_column} = 1) AS avg_sale_price,
    medianIf({price_column}, {sold_flag_column} = 1) AS median_sale_price,
    maxIf({price_column}, {sold_flag_column} = 1) AS max_sale_price,
    sumIf({price_column}, {sold_flag_column} = 1) AS total_sale_volume,
    round(avg({bid_count_column}), 2) AS avg_bids_per_auction,
    countIf({bid_count_column} = 0) AS zero_bid_auctions,
    max({bid_count_column}) AS max_bids_on_single_auction,
    round(avg({domain_authority_column}), 1) AS avg_domain_authority,
    round(avgIf({domain_authority_column}, {sold_flag_column} = 1), 1) AS avg_sold_domain_authority,
    round(avg({traffic_column}), 0) AS avg_monthly_traffic,
    countIf({traffic_column} = 0) AS zero_traffic_listings,
    now() AS as_of
FROM {table}
WHERE {time_column} >= today() - {int(lookback_days)}{extra_where}"""


def multidim_aggregation_sql(table: str, metric_column: str, dimension_columns: List[str], time_column: str, grain: TimeGrain = TimeGrain.DAY, limit: int = 1000, filters: Optional[List[FilterSpec]] = None) -> str:
    """Multi-dimensional ROLLUP aggregation producing subtotals for all dimension subsets.

    Uses ClickHouse ``GROUP BY ... WITH ROLLUP`` so the result includes:
    - One row per full (dim1, dim2, …, period) combination
    - One subtotal row per prefix subset (dim1, period), (period,), ()

    NULL in a dimension column means "rolled up across that dimension".

    :param dimension_columns: One or more dimension columns; at least 1 required
    :param metric_column: Numeric column to aggregate (sum + avg + count)
    """
    _validate_identifier(table, 'table')
    _validate_identifier(metric_column, 'metric_column')
    _validate_identifier(time_column, 'time_column')
    if not dimension_columns:
        raise ValueError('multidim_aggregation_sql requires at least one dimension_column')
    validated_dims = [_validate_identifier(c, 'dimension_columns entry') for c in dimension_columns]
    grain_fn = _GRAIN_FN[grain]
    where = _build_where(filters)
    select_dims = ',\n    '.join(validated_dims)
    group_dims = ', '.join(validated_dims)
    return f"""\
SELECT
    {select_dims},
    {grain_fn}({time_column}) AS period,
    count() AS event_count,
    sum({metric_column}) AS total_metric,
    avg({metric_column}) AS avg_metric,
    max({metric_column}) AS max_metric
FROM {table}
{where}
GROUP BY {group_dims}, period WITH ROLLUP
ORDER BY {validated_dims[0]} ASC NULLS LAST, period ASC NULLS LAST
LIMIT {limit}"""


def price_distribution_sql(table: str, metric_column: str, group_column: str, time_column: str, grain: TimeGrain = TimeGrain.DAY, lookback_periods: int = 30, quantiles: Optional[List[float]] = None, limit: int = 500, filters: Optional[List[FilterSpec]] = None) -> str:
    """Quantile price distribution per group and time period using quantileExact.

    :param quantiles: Fractile list; each value generates a separate quantileExact column (e.g. [0.1,0.5,0.9])
    :return: Rows with group_column, period, total_count, avg/min/max_metric, pN columns per quantile
    """
    _validate_identifier(table, 'table')
    _validate_identifier(metric_column, 'metric_column')
    _validate_identifier(group_column, 'group_column')
    _validate_identifier(time_column, 'time_column')
    grain_fn = _GRAIN_FN[grain]
    where = _build_where(filters)
    _quantiles = quantiles if quantiles is not None else [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    quant_lines = ',\n    '.join(f'quantileExact({q:.2f})({metric_column}) AS p{int(round(q * 100))}' for q in _quantiles)
    return f"""\
SELECT
    {group_column},
    {grain_fn}({time_column}) AS period,
    count() AS total_count,
    avg({metric_column}) AS avg_metric,
    min({metric_column}) AS min_metric,
    max({metric_column}) AS max_metric,
    {quant_lines}
FROM {table}
{where}
GROUP BY {group_column}, period
ORDER BY period DESC, {group_column} ASC
LIMIT {int(limit)}"""


def window_funnel_sql(table: str, entity_column: str, time_column: str, event_column: str, funnel_values: List[str], window_seconds: int, grain: TimeGrain = TimeGrain.DAY, lookback_periods: int = 30, filters: Optional[List[FilterSpec]] = None) -> str:
    """Ordered conversion funnel using ClickHouse windowFunnel with a time constraint.

    :param funnel_values: Ordered list of event_column values representing funnel steps (e.g. ['search','bid','sold'])
    :param window_seconds: Max seconds between first and last step to count as conversion
    :return: Rows with period, reached_step_N counts per funnel level
    """
    _validate_identifier(table, 'table')
    _validate_identifier(entity_column, 'entity_column')
    _validate_identifier(time_column, 'time_column')
    _validate_identifier(event_column, 'event_column')
    grain_fn = _GRAIN_FN[grain]
    where = _build_where(filters)
    conds = ', '.join(f"{event_column} = '{_validate_safe_value(v, 'funnel_values entry')}'" for v in funnel_values)
    step_counts = ',\n    '.join(f'countIf(funnel_level >= {i + 1}) AS reached_step_{i + 1}' for i in range(len(funnel_values)))
    return f"""\
SELECT
    period,
    {step_counts}
FROM (
    SELECT
        {grain_fn}({time_column}) AS period,
        windowFunnel({int(window_seconds)})(toUnixTimestamp64Milli({time_column}), {conds}) AS funnel_level
    FROM {table}
    {where}
    GROUP BY {entity_column}, period
)
GROUP BY period
ORDER BY period DESC"""


def top_k_sql(table: str, keyword_column: str, time_column: str, k: int = 50, lookback_periods: int = 30, grain: TimeGrain = TimeGrain.DAY, filters: Optional[List[FilterSpec]] = None) -> str:
    """Probabilistic top-K values using ClickHouse topK aggregate — O(k) memory, sub-second on large tables.

    :param k: Maximum distinct values to return per period (topK sketch size)
    :return: Rows with period, top_k_values (Array of strings)
    """
    _validate_identifier(table, 'table')
    _validate_identifier(keyword_column, 'keyword_column')
    _validate_identifier(time_column, 'time_column')
    grain_fn = _GRAIN_FN[grain]
    where = _build_where(filters)
    return f"""\
SELECT
    {grain_fn}({time_column}) AS period,
    topK({int(k)})({keyword_column}) AS top_k_values,
    count() AS total_rows
FROM {table}
{where}
GROUP BY period
ORDER BY period DESC
LIMIT {int(lookback_periods)}"""


def moving_average_sql(table: str, metric_column: str, entity_column: str, time_column: str, grain: TimeGrain = TimeGrain.DAY, window_size: int = 7, lookback_periods: int = 30, limit: int = 500, filters: Optional[List[FilterSpec]] = None) -> str:
    """Moving average over a rolling row window using ClickHouse window functions.

    :param window_size: Number of preceding periods to include in the average (ROWS BETWEEN window_size-1 PRECEDING AND CURRENT ROW)
    :return: Rows with entity_column, period, period_avg, moving_avg
    """
    _validate_identifier(table, 'table')
    _validate_identifier(metric_column, 'metric_column')
    _validate_identifier(entity_column, 'entity_column')
    _validate_identifier(time_column, 'time_column')
    grain_fn = _GRAIN_FN[grain]
    where = _build_where(filters)
    preceding = int(window_size) - 1
    return f"""\
SELECT
    {entity_column},
    period,
    period_avg,
    avg(period_avg) OVER (PARTITION BY {entity_column} ORDER BY period ROWS BETWEEN {preceding} PRECEDING AND CURRENT ROW) AS moving_avg
FROM (
    SELECT
        {entity_column},
        {grain_fn}({time_column}) AS period,
        avg({metric_column}) AS period_avg
    FROM {table}
    {where}
    GROUP BY {entity_column}, period
)
ORDER BY {entity_column} ASC, period ASC
LIMIT {int(limit)}"""


def bid_velocity_acceleration_sql(bid_velocity_mv: str, window_hours: int = 2, prior_window_hours: int = 4, min_bids: int = 1, limit: int = 100) -> str:
    """Bid velocity acceleration: recent N-hour bid rate vs prior N-hour rate per auction.

    Reads from an AggregatingMergeTree MV with countState bid_count_state, uniqState unique_bidders_state.

    :param bid_velocity_mv: Fully-qualified AggregatingMergeTree MV name (e.g. 'analytics.mv_bid_velocity_by_auction_hour')
    :param window_hours: Hours in the recent window
    :param prior_window_hours: Total lookback hours (recent + prior combined); must be > window_hours
    :return: Rows with auction_id, recent_bids, prior_bids, velocity_delta, velocity_ratio, unique_bidders
    """
    _validate_identifier(bid_velocity_mv, 'bid_velocity_mv')
    return f"""\
WITH
    recent AS (
        SELECT auction_id, countMerge(bid_count_state) AS recent_bids, uniqMerge(unique_bidders_state) AS unique_bidders
        FROM {bid_velocity_mv}
        WHERE event_hour >= now() - INTERVAL {int(window_hours)} HOUR
        GROUP BY auction_id
        HAVING recent_bids >= {int(min_bids)}
    ),
    prior AS (
        SELECT auction_id, countMerge(bid_count_state) AS prior_bids
        FROM {bid_velocity_mv}
        WHERE event_hour >= now() - INTERVAL {int(prior_window_hours)} HOUR
          AND event_hour < now() - INTERVAL {int(window_hours)} HOUR
        GROUP BY auction_id
    )
SELECT
    r.auction_id,
    r.recent_bids,
    coalesce(p.prior_bids, 0) AS prior_bids,
    r.recent_bids - coalesce(p.prior_bids, 0) AS velocity_delta,
    CASE WHEN coalesce(p.prior_bids, 0) = 0 THEN toFloat64(r.recent_bids) ELSE toFloat64(r.recent_bids) / toFloat64(p.prior_bids) END AS velocity_ratio,
    r.unique_bidders
FROM recent r
LEFT JOIN prior p ON r.auction_id = p.auction_id
ORDER BY velocity_delta DESC
LIMIT {int(limit)}"""


def watch_bid_conversion_sql(watch_density_mv: str, bid_velocity_item_mv: str, lookback_days: int = 7, min_watches: int = 1, limit: int = 200) -> str:
    """Watch-to-bid conversion rate per listing by joining watch density and bid velocity MVs.

    Both MVs must be AggregatingMergeTree keyed on (member_item_id, event_day/event_hour).

    :param watch_density_mv: MV with active_watch_state (countState) + unique_watchers_state (uniqState)
    :param bid_velocity_item_mv: MV with bid_count_state (countState) + unique_bidders_state (uniqState)
    :return: Rows with member_item_id, total_watches, unique_watchers, total_bids, unique_bidders, watch_to_bid_rate
    """
    _validate_identifier(watch_density_mv, 'watch_density_mv')
    _validate_identifier(bid_velocity_item_mv, 'bid_velocity_item_mv')
    return f"""\
WITH
    watches AS (
        SELECT member_item_id, countMerge(active_watch_state) AS total_watches, uniqMerge(unique_watchers_state) AS unique_watchers
        FROM {watch_density_mv}
        WHERE event_day >= today() - {int(lookback_days)}
        GROUP BY member_item_id
        HAVING total_watches >= {int(min_watches)}
    ),
    bids AS (
        SELECT member_item_id, countMerge(bid_count_state) AS total_bids, uniqMerge(unique_bidders_state) AS unique_bidders
        FROM {bid_velocity_item_mv}
        WHERE event_hour >= now() - INTERVAL {int(lookback_days)} DAY
        GROUP BY member_item_id
    )
SELECT
    w.member_item_id,
    w.total_watches,
    w.unique_watchers,
    coalesce(b.total_bids, 0) AS total_bids,
    coalesce(b.unique_bidders, 0) AS unique_bidders,
    CASE WHEN w.unique_watchers = 0 THEN 0.0 ELSE toFloat64(coalesce(b.unique_bidders, 0)) / toFloat64(w.unique_watchers) END AS watch_to_bid_rate
FROM watches w
LEFT JOIN bids b ON w.member_item_id = b.member_item_id
ORDER BY watch_to_bid_rate DESC
LIMIT {int(limit)}"""


def hold_time_distribution_sql(hold_time_mv: str, group_column: str, lookback_days: int = 90, limit: int = 200) -> str:
    """Hold time distribution (days-to-sell) per group from AggregatingMergeTree hold time MV.

    MV must expose: count_state, avg_hold_days_state, min_hold_days_state, max_hold_days_state, avg_price_state, sum_price_state.

    :param hold_time_mv: Fully-qualified MV name (e.g. 'analytics.mv_hold_time_by_type_day')
    :param group_column: Dimension column in the MV to group by (e.g. 'auction_type_name', 'tld')
    :return: Rows with group_column, total_sold, avg/min/max_hold_days, avg/total sale price
    """
    _validate_identifier(hold_time_mv, 'hold_time_mv')
    _validate_identifier(group_column, 'group_column')
    return f"""\
SELECT
    {group_column},
    sumMerge(count_state) AS total_sold,
    avgMerge(avg_hold_days_state) AS avg_hold_days,
    minMerge(min_hold_days_state) AS min_hold_days,
    maxMerge(max_hold_days_state) AS max_hold_days,
    avgMerge(avg_price_state) AS avg_sale_price,
    sumMerge(sum_price_state) AS total_sale_revenue
FROM {hold_time_mv}
WHERE event_day >= today() - {int(lookback_days)}
GROUP BY {group_column}
HAVING total_sold > 0
ORDER BY total_sold DESC
LIMIT {int(limit)}"""


def counter_offer_sql(bid_events_table: str, time_column: str, grain: TimeGrain = TimeGrain.DAY, lookback_periods: int = 30, limit: int = 200, filters: Optional[List[FilterSpec]] = None) -> str:
    """Counter-offer rate and price delta from bid events table.

    :param bid_events_table: Fully-qualified bid events table (e.g. 'analytics.bid_events')
    :return: Rows with period, total_bids, counter_offer_count, counter_offer_rate, avg_bid_usd, avg_counter_offer_usd, price_delta
    """
    _validate_identifier(bid_events_table, 'bid_events_table')
    _validate_identifier(time_column, 'time_column')
    grain_fn = _GRAIN_FN[grain]
    where = _build_where(filters)
    return f"""\
SELECT
    {grain_fn}({time_column}) AS period,
    count() AS total_bids,
    countIf(counter_offer = 1) AS counter_offer_count,
    countIf(counter_offer = 1) / count() AS counter_offer_rate,
    avg(bid_usd_amount) AS avg_bid_usd,
    avgIf(bid_usd_amount, counter_offer = 1) AS avg_counter_offer_usd,
    avgIf(bid_usd_amount, counter_offer = 1) - avg(bid_usd_amount) AS price_delta
FROM {bid_events_table}
WHERE buy_it_now_flag = 0
  AND {time_column} >= now() - INTERVAL {int(lookback_periods)} DAY
{where}
GROUP BY period
ORDER BY period DESC
LIMIT {int(limit)}"""


def price_realization_sql(transactions_table: str, group_column: str, time_column: str, grain: TimeGrain = TimeGrain.DAY, lookback_periods: int = 30, limit: int = 100, filters: Optional[List[FilterSpec]] = None) -> str:
    """Price realization rate (sale_price / listed_price) per group over time.

    :param transactions_table: Fully-qualified domain_transactions table
    :return: Rows with group_column, period, total_transactions, avg_sale_price, avg_listed_price, avg_realization_rate, p50_realization_rate
    """
    _validate_identifier(transactions_table, 'transactions_table')
    _validate_identifier(group_column, 'group_column')
    _validate_identifier(time_column, 'time_column')
    grain_fn = _GRAIN_FN[grain]
    where = _build_where(filters)
    return f"""\
SELECT
    {group_column},
    {grain_fn}({time_column}) AS period,
    count() AS total_transactions,
    avg(sale_price) AS avg_sale_price,
    avg(listed_price) AS avg_listed_price,
    avgIf(sale_price / listed_price, listed_price > 0) AS avg_realization_rate,
    quantileExact(0.50)(CASE WHEN listed_price > 0 THEN sale_price / listed_price ELSE NULL END) AS p50_realization_rate,
    quantileExact(0.25)(CASE WHEN listed_price > 0 THEN sale_price / listed_price ELSE NULL END) AS p25_realization_rate,
    quantileExact(0.75)(CASE WHEN listed_price > 0 THEN sale_price / listed_price ELSE NULL END) AS p75_realization_rate
FROM {transactions_table}
WHERE listed_price > 0
  AND {time_column} >= now() - INTERVAL {int(lookback_periods)} DAY
{where}
GROUP BY {group_column}, period
ORDER BY period DESC, avg_realization_rate DESC
LIMIT {int(limit)}"""


def buyer_retention_cohorts_sql(transactions_table: str, buyer_column: str, time_column: str, cohort_periods: int = 6, lookback_months: int = 12) -> str:
    """Month-cohort buyer retention: of buyers who first purchased in month M, how many returned in M+1..M+N.

    :param transactions_table: Fully-qualified domain_transactions table
    :param buyer_column: Column identifying the buyer (e.g. 'buyer_user_id')
    :param cohort_periods: Number of retention periods (months) to track after first purchase
    :param lookback_months: How many cohort months to include
    :return: Rows with cohort_month, retention_period, cohort_size, retained_buyers, retention_rate
    """
    _validate_identifier(transactions_table, 'transactions_table')
    _validate_identifier(buyer_column, 'buyer_column')
    _validate_identifier(time_column, 'time_column')
    return f"""\
WITH
    first_purchase AS (
        SELECT {buyer_column}, toStartOfMonth(min({time_column})) AS cohort_month
        FROM {transactions_table}
        WHERE {buyer_column} > 0
          AND {time_column} >= now() - INTERVAL {int(lookback_months)} MONTH
        GROUP BY {buyer_column}
    ),
    purchases AS (
        SELECT t.{buyer_column}, f.cohort_month,
            dateDiff('month', f.cohort_month, toStartOfMonth(t.{time_column})) AS retention_period
        FROM {transactions_table} t
        INNER JOIN first_purchase f ON t.{buyer_column} = f.{buyer_column}
        WHERE t.{buyer_column} > 0
          AND retention_period BETWEEN 0 AND {int(cohort_periods)}
    ),
    cohort_sizes AS (
        SELECT cohort_month, count(DISTINCT {buyer_column}) AS cohort_size
        FROM first_purchase
        GROUP BY cohort_month
    )
SELECT
    p.cohort_month,
    p.retention_period,
    cs.cohort_size,
    count(DISTINCT p.{buyer_column}) AS retained_buyers,
    toFloat64(count(DISTINCT p.{buyer_column})) / toFloat64(cs.cohort_size) AS retention_rate
FROM purchases p
INNER JOIN cohort_sizes cs ON p.cohort_month = cs.cohort_month
GROUP BY p.cohort_month, p.retention_period, cs.cohort_size
ORDER BY p.cohort_month ASC, p.retention_period ASC"""


def auction_timing_patterns_sql(table: str, time_column: str, metric_column: str, sold_flag_column: str, lookback_days: int = 90, filters: Optional[List[FilterSpec]] = None) -> str:
    """Hour-of-day x day-of-week heatmap showing bid volume and sell-through by close time.

    :param table: Source auction table
    :param time_column: Auction close/end timestamp column
    :param metric_column: Numeric metric to average per time slot (e.g. bid_count)
    :return: Rows with close_hour (0-23), close_dow (1=Mon..7=Sun), auction_count, sold_count, sell_through_rate, avg_metric
    """
    _validate_identifier(table, 'table')
    _validate_identifier(time_column, 'time_column')
    _validate_identifier(metric_column, 'metric_column')
    _validate_identifier(sold_flag_column, 'sold_flag_column')
    where = _build_where(filters)
    return f"""\
SELECT
    toHour({time_column}) AS close_hour,
    toDayOfWeek({time_column}) AS close_dow,
    count() AS auction_count,
    countIf({sold_flag_column} = 1) AS sold_count,
    countIf({sold_flag_column} = 1) / count() AS sell_through_rate,
    avg({metric_column}) AS avg_metric,
    sum({metric_column}) AS total_metric
FROM {table}
WHERE {time_column} >= now() - INTERVAL {int(lookback_days)} DAY
{where}
GROUP BY close_hour, close_dow
ORDER BY close_dow ASC, close_hour ASC"""


def name_structure_sql(table: str, domain_column: str, price_column: str, sold_flag_column: str, time_column: str, lookback_days: int = 90, min_count: int = 5, limit: int = 100, filters: Optional[List[FilterSpec]] = None) -> str:
    """Domain name structural features correlated with price and sell-through.

    Buckets by: domain part character length (before the first dot), hyphen count, character class (numeric/alpha/mixed).

    :return: Rows with name_length_bucket, char_class, hyphen_count, auction_count, sold_count, sell_through_rate, avg_price, p50_price
    """
    _validate_identifier(table, 'table')
    _validate_identifier(domain_column, 'domain_column')
    _validate_identifier(price_column, 'price_column')
    _validate_identifier(sold_flag_column, 'sold_flag_column')
    _validate_identifier(time_column, 'time_column')
    where = _build_where(filters)
    return f"""\
SELECT
    multiIf(domain_part_len <= 3, '1-3', domain_part_len <= 5, '4-5', domain_part_len <= 8, '6-8', domain_part_len <= 12, '9-12', '13+') AS name_length_bucket,
    multiIf(match(domain_part, '^[0-9]+$'), 'numeric', match(domain_part, '^[a-zA-Z]+$'), 'alpha', 'mixed') AS char_class,
    countSubstrings(domain_part, '-') AS hyphen_count,
    count() AS auction_count,
    countIf({sold_flag_column} = 1) AS sold_count,
    countIf({sold_flag_column} = 1) / count() AS sell_through_rate,
    avg({price_column}) AS avg_price,
    quantileExact(0.50)({price_column}) AS p50_price,
    max({price_column}) AS max_price
FROM (
    SELECT *, substring({domain_column}, 1, position({domain_column}, '.') - 1) AS domain_part,
        length(substring({domain_column}, 1, position({domain_column}, '.') - 1)) AS domain_part_len
    FROM {table}
    WHERE {time_column} >= now() - INTERVAL {int(lookback_days)} DAY
    {where}
)
GROUP BY name_length_bucket, char_class, hyphen_count
HAVING auction_count >= {int(min_count)}
ORDER BY sell_through_rate DESC, auction_count DESC
LIMIT {int(limit)}"""


def cross_tld_spread_sql(table: str, domain_column: str, tld_column: str, price_column: str, sold_flag_column: str, time_column: str, reference_tld: str, lookback_days: int = 90, min_sales: int = 3, limit: int = 100, filters: Optional[List[FilterSpec]] = None) -> str:
    """Cross-TLD price spread: same keyword across different TLDs vs a reference TLD.

    Extracts the domain label (before first dot), groups by keyword + TLD, then joins to reference_tld rows.

    :param reference_tld: TLD to use as price baseline (e.g. 'com')
    :return: Rows with keyword, tld, avg_price, reference_avg_price, price_spread_ratio, sold_count
    """
    _validate_identifier(table, 'table')
    _validate_identifier(domain_column, 'domain_column')
    _validate_identifier(tld_column, 'tld_column')
    _validate_identifier(price_column, 'price_column')
    _validate_identifier(sold_flag_column, 'sold_flag_column')
    _validate_identifier(time_column, 'time_column')
    _validate_safe_value(reference_tld, 'reference_tld')
    where = _build_where(filters)
    return f"""\
WITH
    by_keyword_tld AS (
        SELECT
            substring({domain_column}, 1, position({domain_column}, '.') - 1) AS keyword,
            {tld_column},
            countIf({sold_flag_column} = 1) AS sold_count,
            avgIf({price_column}, {sold_flag_column} = 1) AS avg_price
        FROM {table}
        WHERE {time_column} >= now() - INTERVAL {int(lookback_days)} DAY
        {where}
        GROUP BY keyword, {tld_column}
        HAVING sold_count >= {int(min_sales)}
    ),
    ref AS (
        SELECT keyword, avg_price AS ref_avg_price
        FROM by_keyword_tld
        WHERE {tld_column} = '{reference_tld}'
    )
SELECT
    b.keyword,
    b.{tld_column},
    b.avg_price,
    b.sold_count,
    r.ref_avg_price,
    CASE WHEN r.ref_avg_price > 0 THEN b.avg_price / r.ref_avg_price ELSE NULL END AS price_spread_ratio
FROM by_keyword_tld b
INNER JOIN ref r ON b.keyword = r.keyword
WHERE b.{tld_column} != '{reference_tld}'
ORDER BY price_spread_ratio DESC NULLS LAST
LIMIT {int(limit)}"""


def search_attribution_sql(signals_table: str, transactions_table: str, signal_time_column: str, transaction_time_column: str, attribution_window_hours: int = 24, lookback_days: int = 30, min_signals: int = 1, limit: int = 100) -> str:
    """Search-to-sale attribution: feedback signals joined to domain transactions within a time window.

    :param signals_table: Fully-qualified feedback_signals table
    :param transactions_table: Fully-qualified domain_transactions table
    :param attribution_window_hours: Max hours between signal and confirmed sale to count as attributed
    :return: Rows with signal_type, attributed_sales, total_revenue, avg_sale_price, signal_count, attribution_rate
    """
    _validate_identifier(signals_table, 'signals_table')
    _validate_identifier(transactions_table, 'transactions_table')
    _validate_identifier(signal_time_column, 'signal_time_column')
    _validate_identifier(transaction_time_column, 'transaction_time_column')
    return f"""\
SELECT
    s.signal_type,
    count(DISTINCT t.transaction_id) AS attributed_sales,
    sum(t.sale_price) AS total_revenue,
    avg(t.sale_price) AS avg_sale_price,
    count(DISTINCT s.request_id) AS signal_count,
    toFloat64(count(DISTINCT t.transaction_id)) / toFloat64(count(DISTINCT s.request_id)) AS attribution_rate
FROM {signals_table} s
INNER JOIN {transactions_table} t
    ON s.payload LIKE concat('%', t.domain_name, '%')
    AND t.{transaction_time_column} >= s.{signal_time_column}
    AND t.{transaction_time_column} <= s.{signal_time_column} + INTERVAL {int(attribution_window_hours)} HOUR
WHERE s.{signal_time_column} >= now() - INTERVAL {int(lookback_days)} DAY
GROUP BY s.signal_type
HAVING signal_count >= {int(min_signals)}
ORDER BY attributed_sales DESC
LIMIT {int(limit)}"""


def registrar_hhi_sql(table: str, registrar_column: str, sold_flag_column: str, time_column: str, lookback_days: int = 90, filters: Optional[List[FilterSpec]] = None) -> str:
    """Herfindahl-Hirschman Index (HHI) for registrar market concentration among sold domains.

    HHI = sum(market_share_i ^ 2) where market_share_i = sold_count_i / total_sold.
    HHI close to 0 = competitive; HHI close to 1 = monopoly.

    :return: Rows with registrar_column, sold_count, market_share, hhi_contribution, and a single total_hhi row
    """
    _validate_identifier(table, 'table')
    _validate_identifier(registrar_column, 'registrar_column')
    _validate_identifier(sold_flag_column, 'sold_flag_column')
    _validate_identifier(time_column, 'time_column')
    where = _build_where(filters)
    return f"""\
WITH
    totals AS (
        SELECT countIf({sold_flag_column} = 1) AS total_sold
        FROM {table}
        WHERE {time_column} >= now() - INTERVAL {int(lookback_days)} DAY
        {where}
    ),
    by_registrar AS (
        SELECT {registrar_column},
            countIf({sold_flag_column} = 1) AS sold_count,
            toFloat64(countIf({sold_flag_column} = 1)) / toFloat64((SELECT total_sold FROM totals)) AS market_share
        FROM {table}
        WHERE {time_column} >= now() - INTERVAL {int(lookback_days)} DAY
        {where}
        GROUP BY {registrar_column}
        HAVING sold_count > 0
    )
SELECT
    {registrar_column},
    sold_count,
    market_share,
    market_share * market_share AS hhi_contribution
FROM by_registrar
ORDER BY sold_count DESC"""


def comparable_sale_price_sql(transactions_table: str, tld_column: str, price_column: str, domain_column: str, time_column: str, lookback_days: int = 365, min_sales: int = 5, limit: int = 200, filters: Optional[List[FilterSpec]] = None) -> str:
    """Comparable sale price cohorts: median price by TLD x name-length bucket for fair-value estimation.

    :param transactions_table: Fully-qualified domain_transactions table
    :return: Rows with tld_column, name_length_bucket, cohort_size, p25/p50/p75 sale price, avg_sale_price
    """
    _validate_identifier(transactions_table, 'transactions_table')
    _validate_identifier(tld_column, 'tld_column')
    _validate_identifier(price_column, 'price_column')
    _validate_identifier(domain_column, 'domain_column')
    _validate_identifier(time_column, 'time_column')
    where = _build_where(filters)
    return f"""\
SELECT
    {tld_column},
    multiIf(domain_part_len <= 3, '1-3', domain_part_len <= 5, '4-5', domain_part_len <= 8, '6-8', '9+') AS name_length_bucket,
    count() AS cohort_size,
    quantileExact(0.25)({price_column}) AS p25_price,
    quantileExact(0.50)({price_column}) AS p50_price,
    quantileExact(0.75)({price_column}) AS p75_price,
    avg({price_column}) AS avg_sale_price,
    min({price_column}) AS min_price,
    max({price_column}) AS max_price
FROM (
    SELECT *, length(substring({domain_column}, 1, position({domain_column}, '.') - 1)) AS domain_part_len
    FROM {transactions_table}
    WHERE {time_column} >= now() - INTERVAL {int(lookback_days)} DAY
    {where}
)
GROUP BY {tld_column}, name_length_bucket
HAVING cohort_size >= {int(min_sales)}
ORDER BY {tld_column} ASC, name_length_bucket ASC
LIMIT {int(limit)}"""
