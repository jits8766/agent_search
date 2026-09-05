"""Domain-specific analytics facade over ClickHouseExecutor + query_templates.

Provides high-level analytics methods for the domain auction marketplace,
mapping each analytics use case to the appropriate SQL template + physical
column names from :class:`DomainAnalyticsConfig`.

All methods are async and return ``List[Dict[str, Any]]`` rows directly so
callers can compose with :class:`EnrichmentPipeline` as needed.  Rows are
truncated by ``ClickHouseExecutor`` to ``SqlExecutionConfig.max_rows``.

Design constraints:
  - No MV router / security validator on this path — templates parameterise
    identifiers with ``_validate_identifier``; SQL injection is prevented
    at the template layer.
  - Constructor raises ``ConfigurationError`` on type mismatches; methods
    raise ``RetrievalError`` on ClickHouse failure (callers degrade gracefully).
  - Every capability method checks its ``AnalyticsCapabilitiesConfig`` toggle
    before executing; disabled capabilities raise ``RetrievalError`` immediately
    so callers can log and fall back rather than wait for an empty result.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from semantic_search.analytics.ch_schema import KNOWN_COLUMNS as _SCHEMA_KNOWN_COLUMNS
from semantic_search.analytics.clickhouse_executor import ClickHouseExecutor
from semantic_search.analytics.query_templates import DateRangeFilter, EqualityFilter, FilterSpec, InFilter, NumericRangeFilter, TimeGrain, anomaly_detection_sql, auction_timing_patterns_sql, bid_distribution_sql, bid_velocity_acceleration_sql, buyer_retention_cohorts_sql, category_trend_sql, comparable_sale_price_sql, comparative_cohort_sql, counter_offer_sql, cross_tld_spread_sql, forecast_sql, growth_rate_sql, hold_time_distribution_sql, historical_comparison_sql, keyword_trend_sql, lifecycle_sql, market_summary_sql, moving_average_sql, multidim_aggregation_sql, name_structure_sql, price_distribution_sql, price_realization_sql, ranking_score_sql, realtime_trending_sql, registrar_hhi_sql, search_attribution_sql, sell_through_rate_sql, seo_bucket_sql, time_window_sql, top_k_sql, user_engagement_sql, watch_bid_conversion_sql, window_funnel_sql
from semantic_search.config.analytics_models import AnalyticsCapabilitiesConfig, DomainAnalyticsConfig
from semantic_search.core.exceptions import ConfigurationError, RetrievalError
from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)


def _validate_engine_columns(
    table: str,
    required: Dict[str, str],
    optional: Dict[str, Optional[str]],
) -> None:
    """Validate column names against the known DDL schema for ``table``.

    Skips silently when ``table`` is absent from ``_SCHEMA_KNOWN_COLUMNS`` so
    custom table names and test doubles do not fail at construction.
    """
    schema = _SCHEMA_KNOWN_COLUMNS.get(table)
    if not schema:
        return
    bad: Dict[str, str] = {}
    for field, col in required.items():
        if col not in schema:
            bad[field] = col
    for field, col in optional.items():
        if col and col not in schema:
            bad[field] = col
    if bad:
        raise ConfigurationError(
            f"DomainAnalyticsEngine: column(s) not in {table!r}: "
            + ", ".join(f"{f}={c!r}" for f, c in sorted(bad.items()))
        )


class DomainAnalyticsEngine:
    """Facade providing domain-marketplace analytics via ClickHouse query templates.

    :param executor: ClickHouseExecutor - Shared CH executor (same instance as AnalyticsRouter)
    :param config: DomainAnalyticsConfig - Column-mapping config (table + column names)
    :param capabilities: Optional[AnalyticsCapabilitiesConfig] - Capability toggles; when None
        all methods are enabled with default parameters
    """

    def __init__(self, executor: ClickHouseExecutor, config: DomainAnalyticsConfig, capabilities: Optional[AnalyticsCapabilitiesConfig] = None) -> None:
        if not isinstance(executor, ClickHouseExecutor):
            raise ConfigurationError("DomainAnalyticsEngine requires a ClickHouseExecutor")
        if not isinstance(config, DomainAnalyticsConfig):
            raise ConfigurationError("DomainAnalyticsEngine requires a DomainAnalyticsConfig")
        if capabilities is not None and not isinstance(capabilities, AnalyticsCapabilitiesConfig):
            raise ConfigurationError("DomainAnalyticsEngine capabilities must be AnalyticsCapabilitiesConfig or None")
        self._executor = executor
        self._cfg = config
        self._caps = capabilities
        _validate_engine_columns(
            config.source_table,
            required={
                'domain_column': config.domain_column,
                'tld_column': config.tld_column,
                'price_column': config.price_column,
                'bid_count_column': config.bid_count_column,
                'auction_end_column': config.auction_end_column,
                'score_column': config.score_column,
                'category_column': config.category_column,
                'created_column': config.created_column,
            },
            optional={
                'sold_flag_column': config.sold_flag_column,
                'listed_at_column': config.listed_at_column,
                'sold_at_column': config.sold_at_column,
                'domain_authority_column': config.domain_authority_column,
                'traffic_column': config.traffic_column,
                'domain_age_column': config.domain_age_column,
                'registrar_column': config.registrar_column,
                'backlink_column': config.backlink_column,
                'referring_domains_column': config.referring_domains_column,
                'auction_type_name_column': config.auction_type_name_column,
                'category_name_column': config.category_name_column,
                'hold_days_column': config.hold_days_column,
                'expiry_status_column': config.expiry_status_column,
            },
        )
        _validate_engine_columns(
            config.signals_table,
            required={
                'signals_user_column': config.signals_user_column,
                'signals_event_column': config.signals_event_column,
                'signals_time_column': config.signals_time_column,
                'signals_origin_column': config.signals_origin_column,
            },
            optional={},
        )
        if config.bid_events_table:
            _validate_engine_columns(
                config.bid_events_table,
                required={'event_utc_ts': 'event_utc_ts'},
                optional={},
            )
        if config.transactions_table:
            _validate_engine_columns(
                config.transactions_table,
                required={'buyer_user_id': 'buyer_user_id'},
                optional={
                    'transactions_sold_at_column': config.transactions_sold_at_column,
                    'transactions_sale_price_column': config.transactions_sale_price_column,
                    'transactions_category_column': config.transactions_category_column,
                    'transactions_listed_price_column': config.transactions_listed_price_column,
                    'transactions_govalue_column': config.transactions_govalue_column,
                },
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enabled(self) -> bool:
        return bool(self._cfg.enabled)

    def _check_enabled(self, method_name: str) -> None:
        if not self._enabled():
            raise RetrievalError(f"DomainAnalyticsEngine.{method_name}: domain_analytics.enabled=false")

    def _cap_lookback(self, n: int) -> int:
        return min(int(n), int(self._cfg.max_lookback_periods))

    async def _run(self, sql: str, method_name: str) -> List[Dict[str, Any]]:
        """Execute SQL and return rows; logs method context on failure."""
        try:
            await self._executor.explain(sql)
        except RetrievalError as exc:
            raise RetrievalError(f"DomainAnalyticsEngine.{method_name} invalid SQL: {exc}") from exc
        except Exception as exc:
            raise RetrievalError(f"DomainAnalyticsEngine.{method_name} explain failed: {exc}") from exc
        try:
            result = await self._executor.execute(sql)
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(f"DomainAnalyticsEngine.{method_name} execution failed: {exc}") from exc
        return list(result.rows)

    # ------------------------------------------------------------------
    # Group 1 — Already wired, needs query template wiring
    # ------------------------------------------------------------------

    async def conversion_funnel(self, grain: TimeGrain = TimeGrain.DAY, lookback_periods: int = 30, filters: Optional[List[FilterSpec]] = None) -> List[Dict[str, Any]]:
        """Search → engagement funnel via FeedbackSignal events in ClickHouse.

        Aggregates daily active users, total events, and events-per-user from
        the ``analytics.feedback_signals`` table.  Requires the signals table
        to be populated via :class:`FeedbackSignalWriter`.

        :param grain: Time aggregation granularity
        :param lookback_periods: Rolling window size for avg_events smoothing
        :param filters: Optional typed filter list (e.g. DateRangeFilter on created_at)
        :return: List of rows with period, active_users, total_events, events_per_user
        """
        self._check_enabled('conversion_funnel')
        sql = user_engagement_sql(
            table=self._cfg.signals_table,
            user_column=self._cfg.signals_user_column,
            event_column=self._cfg.signals_event_column,
            time_column=self._cfg.signals_time_column,
            grain=grain,
            lookback_periods=self._cap_lookback(lookback_periods),
            filters=filters,
        )
        logger.info(f"domain_analytics_conversion_funnel grain={grain.value}")
        return await self._run(sql, 'conversion_funnel')

    async def buyer_interest_heatmap(self, grain: TimeGrain = TimeGrain.DAY, lookback_periods: int = 30, filters: Optional[List[FilterSpec]] = None) -> List[Dict[str, Any]]:
        """Click and engagement signal counts grouped by signal_type over time.

        Aggregates ``result_click``, ``chip_promoted``, and ``resume_clicked``
        signals from ``analytics.feedback_signals`` to show which signal types
        peak at what time.

        :param grain: Time aggregation granularity
        :param lookback_periods: Used as informational lookback; use DateRangeFilter to constrain
        :param filters: Optional typed filter list
        :return: Rows with signal_type, period, event_count, total_metric, avg_metric
        """
        self._check_enabled('buyer_interest_heatmap')
        sql = category_trend_sql(
            table=self._cfg.signals_table,
            metric_column=self._cfg.signals_time_column,
            category_column=self._cfg.signals_event_column,
            time_column=self._cfg.signals_time_column,
            grain=grain,
            lookback_periods=self._cap_lookback(lookback_periods),
            filters=filters,
        )
        logger.info(f"domain_analytics_buyer_interest_heatmap grain={grain.value}")
        return await self._run(sql, 'buyer_interest_heatmap')

    async def cache_hit_analytics(self, grain: TimeGrain = TimeGrain.DAY, lookback_periods: int = 30, filters: Optional[List[FilterSpec]] = None) -> List[Dict[str, Any]]:
        """Signal-origin distribution over time (shows orchestrator vs cache vs QI ratios).

        Groups signals by ``signal_origin`` + time bucket to surface which subsystem
        emits the most signals — a proxy for cache hit/miss and QI tier distribution.

        :return: Rows with signal_origin, period, event_count, total_metric, avg_metric
        """
        self._check_enabled('cache_hit_analytics')
        sql = category_trend_sql(
            table=self._cfg.signals_table,
            metric_column=self._cfg.signals_time_column,
            category_column=self._cfg.signals_origin_column,
            time_column=self._cfg.signals_time_column,
            grain=grain,
            lookback_periods=self._cap_lookback(lookback_periods),
            filters=filters,
        )
        logger.info(f"domain_analytics_cache_hit_analytics grain={grain.value}")
        return await self._run(sql, 'cache_hit_analytics')

    # ------------------------------------------------------------------
    # Group 2 — Domain data aggregation analytics
    # ------------------------------------------------------------------

    async def tld_popularity_trends(self, grain: TimeGrain = TimeGrain.DAY, lookback_periods: int = 30, filters: Optional[List[FilterSpec]] = None) -> List[Dict[str, Any]]:
        """TLD-level auction volume and price trends over time.

        :param grain: Time aggregation granularity
        :param lookback_periods: Number of periods; use DateRangeFilter to constrain
        :param filters: Optional typed filter list (e.g. EqualityFilter on tld)
        :return: Rows with tld, period, event_count, total_metric, avg_metric, max_metric, min_metric
        """
        self._check_enabled('tld_popularity_trends')
        sql = category_trend_sql(
            table=self._cfg.source_table,
            metric_column=self._cfg.price_column,
            category_column=self._cfg.tld_column,
            time_column=self._cfg.auction_end_column,
            grain=grain,
            lookback_periods=self._cap_lookback(lookback_periods),
            filters=filters,
        )
        logger.info(f"domain_analytics_tld_popularity_trends grain={grain.value}")
        return await self._run(sql, 'tld_popularity_trends')

    async def price_movement(self, grain: TimeGrain = TimeGrain.DAY, lookback_periods: int = 30, tld: Optional[str] = None) -> List[Dict[str, Any]]:
        """Price movement over time, optionally filtered to a single TLD.

        :param grain: Time aggregation granularity
        :param lookback_periods: Number of periods; add a DateRangeFilter via the filters param
        :param tld: When set, restricts to domains matching this TLD value
        :return: Rows with tld, period, event_count, sum/avg/max for current_price and bid_count
        """
        self._check_enabled('price_movement')
        filters: List[FilterSpec] = []
        if tld is not None:
            filters.append(EqualityFilter(column=self._cfg.tld_column, value=tld))
        sql = time_window_sql(
            table=self._cfg.source_table,
            metric_columns=[self._cfg.price_column, self._cfg.bid_count_column],
            group_columns=[self._cfg.tld_column],
            time_column=self._cfg.auction_end_column,
            grain=grain,
            lookback_periods=self._cap_lookback(lookback_periods),
            filters=filters or None,
        )
        logger.info(f"domain_analytics_price_movement grain={grain.value} tld={tld!r}")
        return await self._run(sql, 'price_movement')

    async def auction_competitiveness(self, grain: TimeGrain = TimeGrain.DAY, limit: int = 50, filters: Optional[List[FilterSpec]] = None) -> List[Dict[str, Any]]:
        """Top domains ranked by composite competitiveness score (bid_count + price + govalue).

        Uses min-max normalised weighted sum across the three signals so metrics
        with different natural scales contribute proportionally.

        :param grain: Time aggregation granularity
        :param limit: Maximum domains returned
        :return: Rows with domain_name, period, raw_* columns, composite_score
        """
        self._check_enabled('auction_competitiveness')
        caps = self._caps
        weights = (
            caps.ranking_scoring.default_score_weights
            if caps and caps.ranking_scoring.enabled and caps.ranking_scoring.default_score_weights
            else {}
        )
        if not weights:
            weights = {
                self._cfg.bid_count_column: 0.5,
                self._cfg.price_column: 0.3,
                self._cfg.score_column: 0.2,
            }
        sql = ranking_score_sql(
            table=self._cfg.source_table,
            entity_column=self._cfg.domain_column,
            score_weights=weights,
            time_column=self._cfg.auction_end_column,
            grain=grain,
            limit=limit,
            filters=filters,
        )
        logger.info(f"domain_analytics_auction_competitiveness limit={limit}")
        return await self._run(sql, 'auction_competitiveness')

    async def expiry_pipeline_forecast(self, grain: TimeGrain = TimeGrain.DAY, history_periods: int = 30, forecast_periods: int = 7, filters: Optional[List[FilterSpec]] = None) -> List[Dict[str, Any]]:
        """Linear-trend forecast of auction volume (bid_count) per TLD.

        Uses ``simpleLinearRegression`` in ClickHouse — no external model needed.

        :param history_periods: Past periods fed into the regression (>= 7)
        :param forecast_periods: Periods ahead to project (>= 1)
        :return: Rows with tld, latest_period, forecast_value, forecast_quality (Pearson r)
        """
        self._check_enabled('expiry_pipeline_forecast')
        caps = self._caps
        if caps and caps.forecasting.enabled:
            history_periods = min(history_periods, caps.forecasting.max_history_periods)
            forecast_periods = min(forecast_periods, caps.forecasting.max_forecast_periods)
        sql = forecast_sql(
            table=self._cfg.source_table,
            metric_column=self._cfg.bid_count_column,
            entity_column=self._cfg.tld_column,
            time_column=self._cfg.auction_end_column,
            grain=grain,
            history_periods=history_periods,
            forecast_periods=forecast_periods,
            filters=filters,
        )
        logger.info(f"domain_analytics_expiry_pipeline_forecast history={history_periods} horizon={forecast_periods}")
        return await self._run(sql, 'expiry_pipeline_forecast')

    async def realtime_auction_activity(self, window_minutes: int = 60, limit: int = 20, filters: Optional[List[FilterSpec]] = None) -> List[Dict[str, Any]]:
        """Domains ranked by activity in the last N minutes (real-time hot list).

        :param window_minutes: Lookback window in minutes (>= 1)
        :param limit: Maximum domains returned
        :return: Rows with domain_name, event_count, total_metric, first_seen, last_seen
        """
        self._check_enabled('realtime_auction_activity')
        sql = realtime_trending_sql(
            table=self._cfg.source_table,
            metric_column=self._cfg.bid_count_column,
            entity_column=self._cfg.domain_column,
            time_column=self._cfg.auction_end_column,
            window_minutes=window_minutes,
            limit=limit,
            filters=filters,
        )
        logger.info(f"domain_analytics_realtime_auction_activity window_minutes={window_minutes}")
        return await self._run(sql, 'realtime_auction_activity')

    async def historical_sales_comparison(self, current_start: str, current_end: str, prior_start: str, prior_end: str, group_by: str = 'tld', limit: int = 100, filters: Optional[List[FilterSpec]] = None) -> List[Dict[str, Any]]:
        """Compare auction price between a current and a historical reference period.

        :param current_start: Start of current window ('YYYY-MM-DD')
        :param current_end: End of current window ('YYYY-MM-DD')
        :param prior_start: Start of prior window ('YYYY-MM-DD')
        :param prior_end: End of prior window ('YYYY-MM-DD')
        :param group_by: Dimension column to group by ('tld' or 'auction_type_id')
        :return: Rows with group_by, current/prior metric, absolute_delta, relative_delta_pct
        """
        self._check_enabled('historical_sales_comparison')
        group_column = self._cfg.tld_column if group_by == 'tld' else self._cfg.category_column
        sql = historical_comparison_sql(
            table=self._cfg.source_table,
            metric_column=self._cfg.price_column,
            group_column=group_column,
            time_column=self._cfg.auction_end_column,
            current_start=current_start,
            current_end=current_end,
            prior_start=prior_start,
            prior_end=prior_end,
            limit=limit,
            filters=filters,
        )
        logger.info(f"domain_analytics_historical_sales_comparison current={current_start}/{current_end} prior={prior_start}/{prior_end}")
        return await self._run(sql, 'historical_sales_comparison')

    async def seasonal_trends(self, grain: TimeGrain = TimeGrain.MONTH, lookback_periods: int = 52, filters: Optional[List[FilterSpec]] = None) -> List[Dict[str, Any]]:
        """Multi-metric seasonal aggregation (price + bid_count) grouped by TLD.

        :param grain: Typically MONTH or WEEK to surface seasonality
        :param lookback_periods: Number of periods; add a DateRangeFilter to constrain
        :return: Rows with tld, period, event_count, sum/avg/max for price and bid_count
        """
        self._check_enabled('seasonal_trends')
        sql = time_window_sql(
            table=self._cfg.source_table,
            metric_columns=[self._cfg.price_column, self._cfg.bid_count_column],
            group_columns=[self._cfg.tld_column],
            time_column=self._cfg.auction_end_column,
            grain=grain,
            lookback_periods=self._cap_lookback(lookback_periods),
            filters=filters,
        )
        logger.info(f"domain_analytics_seasonal_trends grain={grain.value}")
        return await self._run(sql, 'seasonal_trends')

    async def domain_quality_score(self, grain: TimeGrain = TimeGrain.DAY, limit: int = 100, filters: Optional[List[FilterSpec]] = None) -> List[Dict[str, Any]]:
        """Domains ranked by composite quality score (govalue_score + bid_count + price).

        :param limit: Maximum domains returned
        :return: Rows with domain_name, period, raw metric columns, composite_score
        """
        self._check_enabled('domain_quality_score')
        caps = self._caps
        dq_weights = (caps.ranking_scoring.default_score_weights if caps and caps.ranking_scoring.enabled and caps.ranking_scoring.default_score_weights else {self._cfg.score_column: 0.5, self._cfg.bid_count_column: 0.3, self._cfg.price_column: 0.2})
        sql = ranking_score_sql(
            table=self._cfg.source_table,
            entity_column=self._cfg.domain_column,
            score_weights=dq_weights,
            time_column=self._cfg.auction_end_column,
            grain=grain,
            limit=limit,
            filters=filters,
        )
        logger.info(f"domain_analytics_domain_quality_score limit={limit}")
        return await self._run(sql, 'domain_quality_score')

    async def domain_liquidity(self, grain: TimeGrain = TimeGrain.WEEK, lookback_periods: int = 12, limit: int = 50, filters: Optional[List[FilterSpec]] = None) -> List[Dict[str, Any]]:
        """TLDs ranked by bid velocity growth — highest bid-count growth rate first.

        Bid count growth rate is a proxy for domain liquidity: fast-growing bid
        counts indicate active buyer demand and easier resale.

        :param grain: Typically WEEK or DAY
        :param lookback_periods: Minimum periods for a valid lag comparison
        :return: Rows with tld, period, current_value, previous_value, growth_pct
        """
        self._check_enabled('domain_liquidity')
        sql = growth_rate_sql(
            table=self._cfg.source_table,
            metric_column=self._cfg.bid_count_column,
            entity_column=self._cfg.tld_column,
            time_column=self._cfg.auction_end_column,
            grain=grain,
            lookback_periods=self._cap_lookback(lookback_periods),
            limit=limit,
            filters=filters,
            descending=True,
        )
        logger.info(f"domain_analytics_domain_liquidity grain={grain.value}")
        return await self._run(sql, 'domain_liquidity')

    async def supply_demand_analytics(self, grain: TimeGrain = TimeGrain.DAY, lookback_periods: int = 30, filters: Optional[List[FilterSpec]] = None) -> List[Dict[str, Any]]:
        """Supply (auction count) vs demand (bid count) aggregated by TLD over time.

        Both metrics share the same row so operators can compare them in a single
        query instead of joining two separate aggregations.

        :return: Rows with tld, period, event_count (supply proxy), sum/avg/max bid_count + price
        """
        self._check_enabled('supply_demand_analytics')
        sql = time_window_sql(
            table=self._cfg.source_table,
            metric_columns=[self._cfg.bid_count_column, self._cfg.price_column],
            group_columns=[self._cfg.tld_column],
            time_column=self._cfg.auction_end_column,
            grain=grain,
            lookback_periods=self._cap_lookback(lookback_periods),
            filters=filters,
        )
        logger.info(f"domain_analytics_supply_demand grain={grain.value}")
        return await self._run(sql, 'supply_demand_analytics')

    async def keyword_trends(self, grain: TimeGrain = TimeGrain.WEEK, lookback_periods: int = 52, min_frequency: int = 5, limit: int = 100, filters: Optional[List[FilterSpec]] = None) -> List[Dict[str, Any]]:
        """Keyword frequency trends extracted from domain SLD tokens.

        Splits each domain name on ``-`` to yield compound-word tokens (e.g.
        ``'cool-brand.com'`` → ``['cool', 'brand']``), then aggregates by
        keyword + time bucket.

        :param grain: Typically WEEK to surface mid-term trends
        :param min_frequency: Minimum occurrence count per (keyword, period) bucket
        :param limit: Maximum (keyword, period) rows returned
        :return: Rows with keyword, period, frequency, unique_domains, avg_price
        """
        self._check_enabled('keyword_trends')
        sql = keyword_trend_sql(
            table=self._cfg.source_table,
            domain_column=self._cfg.domain_column,
            time_column=self._cfg.auction_end_column,
            grain=grain,
            min_frequency=min_frequency,
            lookback_periods=self._cap_lookback(lookback_periods),
            limit=limit,
            filters=filters,
        )
        logger.info(f"domain_analytics_keyword_trends grain={grain.value} min_frequency={min_frequency}")
        return await self._run(sql, 'keyword_trends')

    async def emerging_niche_detection(self, grain: TimeGrain = TimeGrain.WEEK, window_periods: int = 30, z_threshold: float = 2.5, limit: int = 100, filters: Optional[List[FilterSpec]] = None) -> List[Dict[str, Any]]:
        """Detect TLDs with anomalously high bid-count spikes (emerging niches).

        Uses rolling Z-score — a TLD is flagged when its bid count deviates more
        than ``z_threshold`` standard deviations from the rolling mean.

        :param window_periods: Rolling window size for Z-score baseline (>= 2)
        :param z_threshold: Anomaly gate multiplier (> 0)
        :param limit: Maximum flagged rows returned
        :return: Rows with tld, period, metric_value, rolling_avg, z_score, is_anomaly=True
        """
        self._check_enabled('emerging_niche_detection')
        caps = self._caps
        if caps and caps.anomaly_detection.enabled:
            z_threshold = caps.anomaly_detection.z_score_threshold
            window_periods = max(window_periods, caps.anomaly_detection.min_window_periods)
            limit = min(limit, caps.anomaly_detection.max_results)
        sql = anomaly_detection_sql(
            table=self._cfg.source_table,
            metric_column=self._cfg.bid_count_column,
            entity_column=self._cfg.tld_column,
            time_column=self._cfg.auction_end_column,
            grain=grain,
            z_threshold=z_threshold,
            window_periods=window_periods,
            limit=limit,
            filters=filters,
        )
        logger.info(f"domain_analytics_emerging_niche_detection z_threshold={z_threshold} window={window_periods}")
        return await self._run(sql, 'emerging_niche_detection')

    # ------------------------------------------------------------------
    # Group 3 — Sold-domain, sell-through, SEO, bid, registrar analytics
    # ------------------------------------------------------------------

    async def sold_domain_analytics(self, grain: TimeGrain = TimeGrain.DAY, lookback_periods: int = 30, filters: Optional[List[FilterSpec]] = None) -> List[Dict[str, Any]]:
        """Sold domain volume, revenue, and avg price grouped by TLD over time.

        Queries the base table with sold_flag=1 filter using sold_at as the time
        column so results are bucketed by when domains actually sold, not when
        auctions ended.

        Covers Q1 (domains sold this week), Q2 (total sales volume), Q3 (avg sale
        price), Q4 (sales trend 90 days), Q7-9 (TLD sold analytics), Q30 (avg
        winning bid), Q61 (top sales this month), Q64 (top domains by traffic sold).

        :return: Rows with tld, period, event_count, sum/avg/max for current_price and bid_count
        """
        self._check_enabled('sold_domain_analytics')
        if not self._cfg.sold_flag_column:
            raise RetrievalError('sold_domain_analytics requires sold_flag_column in config')
        if not self._cfg.sold_at_column:
            raise RetrievalError('sold_domain_analytics requires sold_at_column in config')
        sold_flag = self._cfg.sold_flag_column
        sold_at = self._cfg.sold_at_column
        extra_filters: List[FilterSpec] = list(filters or [])
        extra_filters.append(EqualityFilter(column=sold_flag, value='1'))
        sql = time_window_sql(
            table=self._cfg.source_table,
            metric_columns=[self._cfg.price_column, self._cfg.bid_count_column],
            group_columns=[self._cfg.tld_column],
            time_column=sold_at,
            grain=grain,
            lookback_periods=self._cap_lookback(lookback_periods),
            filters=extra_filters,
        )
        logger.info(f"domain_analytics_sold_domain_analytics grain={grain.value}")
        return await self._run(sql, 'sold_domain_analytics')

    async def sell_through_rate(self, grain: TimeGrain = TimeGrain.DAY, lookback_periods: int = 30, limit: int = 50, filters: Optional[List[FilterSpec]] = None) -> List[Dict[str, Any]]:
        """Sell-through rate (sold / total) and avg sale price grouped by TLD.

        Covers Q6 (overall sell-through rate), Q12 (TLD sell-through comparison),
        Q29 (auction success rate).

        :return: Rows with tld, period, total_auctions, sold_auctions, sell_through_rate,
            avg_sale_price, total_sale_revenue
        """
        self._check_enabled('sell_through_rate')
        if not self._cfg.sold_flag_column:
            raise RetrievalError('sell_through_rate requires sold_flag_column in config')
        sql = sell_through_rate_sql(
            table=self._cfg.source_table,
            dimension_column=self._cfg.tld_column,
            sold_flag_column=self._cfg.sold_flag_column,
            time_column=self._cfg.auction_end_column,
            price_column=self._cfg.price_column,
            grain=grain,
            lookback_periods=self._cap_lookback(lookback_periods),
            limit=limit,
            filters=filters,
        )
        logger.info(f"domain_analytics_sell_through_rate grain={grain.value}")
        return await self._run(sql, 'sell_through_rate')

    async def seo_price_correlation(self, grain: TimeGrain = TimeGrain.MONTH, lookback_periods: int = 3, filters: Optional[List[FilterSpec]] = None) -> List[Dict[str, Any]]:
        """Domain Authority and traffic bucket correlation with listing and sale prices.

        Buckets DA into 6 ranges (DA-0, DA-1-9, DA-10-29, DA-30-49, DA-50-69, DA-70+)
        and traffic into 5 ranges, then computes avg listing price, avg sale price, and
        sell-through rate per bucket.

        Covers Q19 (traffic vs sale price), Q20 (DA vs sale price), Q21 (avg DA of sold
        domains), Q22 (zero-traffic domains sold above $1000), Q23 (backlinks vs value),
        Q24 (SEO domain performance trend).

        :return: Rows with da_bucket, traffic_bucket, period, domain_count, sold_count,
            sell_through_rate, avg_listing_price, avg_sale_price
        """
        self._check_enabled('seo_price_correlation')
        if not self._cfg.domain_authority_column:
            raise RetrievalError('seo_price_correlation requires domain_authority_column in config')
        if not self._cfg.traffic_column:
            raise RetrievalError('seo_price_correlation requires traffic_column in config')
        if not self._cfg.sold_flag_column:
            raise RetrievalError('seo_price_correlation requires sold_flag_column in config')
        sql = seo_bucket_sql(
            table=self._cfg.source_table,
            da_column=self._cfg.domain_authority_column,
            traffic_column=self._cfg.traffic_column,
            price_column=self._cfg.price_column,
            sold_flag_column=self._cfg.sold_flag_column,
            time_column=self._cfg.auction_end_column,
            grain=grain,
            lookback_periods=self._cap_lookback(lookback_periods),
            filters=filters,
        )
        logger.info(f"domain_analytics_seo_price_correlation grain={grain.value}")
        return await self._run(sql, 'seo_price_correlation')

    async def bid_analytics(self, grain: TimeGrain = TimeGrain.DAY, lookback_periods: int = 30, filters: Optional[List[FilterSpec]] = None) -> List[Dict[str, Any]]:
        """Bid count distribution with win rate and price per bid bucket.

        Covers Q25 (avg bids per auction), Q26 (auctions ending with no bids), Q27
        (bid count vs final price), Q30 (avg winning bid amount).

        :return: Rows with bid_bucket, period, auction_count, sold_count, win_rate,
            avg_bids_in_bucket, avg_listing_price, avg_winning_price
        """
        self._check_enabled('bid_analytics')
        if not self._cfg.sold_flag_column:
            raise RetrievalError('bid_analytics requires sold_flag_column in config')
        sql = bid_distribution_sql(
            table=self._cfg.source_table,
            bid_count_column=self._cfg.bid_count_column,
            price_column=self._cfg.price_column,
            sold_flag_column=self._cfg.sold_flag_column,
            time_column=self._cfg.auction_end_column,
            grain=grain,
            lookback_periods=self._cap_lookback(lookback_periods),
            filters=filters,
        )
        logger.info(f"domain_analytics_bid_analytics grain={grain.value}")
        return await self._run(sql, 'bid_analytics')

    async def registrar_analytics(self, grain: TimeGrain = TimeGrain.DAY, lookback_periods: int = 30, limit: int = 50, filters: Optional[List[FilterSpec]] = None) -> List[Dict[str, Any]]:
        """Domain volume and sell-through rate grouped by registrar.

        Covers Q32 (registrar drop volume), Q36 (expiry sales trend by registrar).

        :return: Rows with registrar_name, period, total_auctions, sold_auctions,
            sell_through_rate, avg_sale_price, total_sale_revenue
        """
        self._check_enabled('registrar_analytics')
        if not self._cfg.registrar_column:
            raise RetrievalError('registrar_analytics requires registrar_column in config')
        if not self._cfg.sold_flag_column:
            raise RetrievalError('registrar_analytics requires sold_flag_column in config')
        sql = sell_through_rate_sql(
            table=self._cfg.source_table,
            dimension_column=self._cfg.registrar_column,
            sold_flag_column=self._cfg.sold_flag_column,
            time_column=self._cfg.auction_end_column,
            price_column=self._cfg.price_column,
            grain=grain,
            lookback_periods=self._cap_lookback(lookback_periods),
            limit=limit,
            filters=filters,
        )
        logger.info(f"domain_analytics_registrar_analytics grain={grain.value}")
        return await self._run(sql, 'registrar_analytics')

    async def expiry_lifecycle_analytics(self, grain: TimeGrain = TimeGrain.DAY, lookback_periods: int = 30, filters: Optional[List[FilterSpec]] = None) -> List[Dict[str, Any]]:
        """Auction count and pricing broken down by expiry lifecycle stage.

        Requires the ``expiry_status`` column to be populated by the upstream
        expiry-event pipeline ('active' | 'pending_delete' | 'grace_period' | 'expired').

        Covers Q33 (pending delete count today), Q34 (grace period recovery rate),
        Q35 (backorder success rate), Q36 (expiry sales trend).

        :return: Rows with expiry_status, period, event_count, sum/avg/max price
        """
        self._check_enabled('expiry_lifecycle_analytics')
        if not self._cfg.expiry_status_column:
            raise RetrievalError('expiry_lifecycle_analytics requires expiry_status_column in config')
        sql = time_window_sql(
            table=self._cfg.source_table,
            metric_columns=[self._cfg.price_column],
            group_columns=[self._cfg.expiry_status_column],
            time_column=self._cfg.auction_end_column,
            grain=grain,
            lookback_periods=self._cap_lookback(lookback_periods),
            filters=filters,
        )
        logger.info(f"domain_analytics_expiry_lifecycle_analytics grain={grain.value}")
        return await self._run(sql, 'expiry_lifecycle_analytics')

    async def category_name_analytics(self, grain: TimeGrain = TimeGrain.DAY, lookback_periods: int = 30, limit: int = 200, filters: Optional[List[FilterSpec]] = None) -> List[Dict[str, Any]]:
        """Sell-through rate and revenue grouped by human-readable category name.

        Uses ``category_name`` (e.g. 'ai', 'fintech', 'healthcare') rather than the
        integer ``auction_type_id``. Backed by mv_auctions_by_category_name_day.

        Covers Q13 (ai domain sales), Q14 (fintech revenue trend), Q15 (healthcare
        trend), Q16 (hottest category), Q17 (category with most bids), Q18 (category
        revenue breakdown), Q62 (top categories by revenue).

        :return: Rows with category_name, period, total_auctions, sold_auctions,
            sell_through_rate, avg_sale_price, total_sale_revenue
        """
        self._check_enabled('category_name_analytics')
        if not self._cfg.category_name_column:
            raise RetrievalError('category_name_analytics requires category_name_column in config')
        if not self._cfg.sold_flag_column:
            raise RetrievalError('category_name_analytics requires sold_flag_column in config')
        sql = sell_through_rate_sql(
            table=self._cfg.source_table,
            dimension_column=self._cfg.category_name_column,
            sold_flag_column=self._cfg.sold_flag_column,
            time_column=self._cfg.auction_end_column,
            price_column=self._cfg.price_column,
            grain=grain,
            lookback_periods=self._cap_lookback(lookback_periods),
            limit=limit,
            filters=filters,
        )
        logger.info(f"domain_analytics_category_name_analytics grain={grain.value}")
        return await self._run(sql, 'category_name_analytics')

    async def investor_analytics(self, grain: TimeGrain = TimeGrain.MONTH, lookback_periods: int = 12, filters: Optional[List[FilterSpec]] = None) -> List[Dict[str, Any]]:
        """Flip profit margin and ROI from the domain_transactions ledger.

        Requires ``analytics.domain_transactions`` to be populated by the bid-settlement
        pipeline. Returns empty list if the table is empty or not yet seeded.

        Covers Q37 (avg flip profit margin), Q38 (investor ROI by category), Q39 (long
        hold appreciation), Q40 (underpriced frequency via govalue spread), Q41 (best
        investment category), Q42 (portfolio yield proxy).

        :return: Rows with category_name, period, transaction_count, avg_sale_price,
            avg_listed_price, avg_profit, avg_roi_pct, max_profit
        """
        self._check_enabled('investor_analytics')
        if not self._cfg.transactions_table:
            raise RetrievalError('investor_analytics requires transactions_table in config')
        if not self._cfg.transactions_category_column:
            raise RetrievalError('investor_analytics requires transactions_category_column in config')
        if not self._cfg.transactions_sold_at_column:
            raise RetrievalError('investor_analytics requires transactions_sold_at_column in config')
        if not self._cfg.transactions_sale_price_column:
            raise RetrievalError('investor_analytics requires transactions_sale_price_column in config')
        if not self._cfg.transactions_listed_price_column:
            raise RetrievalError('investor_analytics requires transactions_listed_price_column in config')
        if not self._cfg.transactions_govalue_column:
            raise RetrievalError('investor_analytics requires transactions_govalue_column in config')
        _GRAIN_FN_MAP = {
            TimeGrain.HOUR: 'toStartOfHour',
            TimeGrain.DAY: 'toStartOfDay',
            TimeGrain.WEEK: 'toStartOfWeek',
            TimeGrain.MONTH: 'toStartOfMonth',
        }
        grain_fn = _GRAIN_FN_MAP[grain]
        cat_col = self._cfg.transactions_category_column
        sold_at_col = self._cfg.transactions_sold_at_column
        sale_col = self._cfg.transactions_sale_price_column
        listed_col = self._cfg.transactions_listed_price_column
        gv_col = self._cfg.transactions_govalue_column
        filter_clause = ''
        if filters:
            parts = [f.to_sql() for f in filters]
            filter_clause = ' AND ' + ' AND '.join(parts)
        sql = f"""\
SELECT
    {cat_col},
    {grain_fn}({sold_at_col}) AS period,
    count() AS transaction_count,
    round(avg({sale_col}), 2) AS avg_sale_price,
    round(avg({listed_col}), 2) AS avg_listed_price,
    round(avg({sale_col} - {listed_col}), 2) AS avg_profit,
    round(avg(if({listed_col} > 0, ({sale_col} - {listed_col}) / {listed_col} * 100.0, NULL)), 2) AS avg_roi_pct,
    round(max({sale_col} - {listed_col}), 2) AS max_profit,
    round(avg({gv_col}), 2) AS avg_govalue_score
FROM {self._cfg.transactions_table}
WHERE {sale_col} > 0{filter_clause}
GROUP BY {cat_col}, period
ORDER BY avg_roi_pct DESC NULLS LAST, period DESC
LIMIT 200"""
        logger.info(f"domain_analytics_investor_analytics grain={grain.value}")
        return await self._run(sql, 'investor_analytics')

    async def buyer_segment_analytics(self, grain: TimeGrain = TimeGrain.WEEK, lookback_periods: int = 12, filters: Optional[List[FilterSpec]] = None) -> List[Dict[str, Any]]:
        """Auction activity and sell-through rate by buyer segment.

        Requires the ``buyer_segment`` column to be populated by the buyer-profile
        pipeline ('beginner' | 'professional' | 'advanced' | 'investor').

        Covers Q43-48 (beginner analytics), Q49-54 (professional analytics), Q55-60
        (advanced buyer analytics).

        :return: Rows with buyer_segment, period, total_auctions, sold_auctions,
            sell_through_rate, avg_sale_price, total_sale_revenue
        """
        self._check_enabled('buyer_segment_analytics')
        if not self._cfg.buyer_segment_column:
            raise RetrievalError('buyer_segment_analytics requires buyer_segment_column in config')
        if not self._cfg.sold_flag_column:
            raise RetrievalError('buyer_segment_analytics requires sold_flag_column in config')
        sql = sell_through_rate_sql(
            table=self._cfg.source_table,
            dimension_column=self._cfg.buyer_segment_column,
            sold_flag_column=self._cfg.sold_flag_column,
            time_column=self._cfg.auction_end_column,
            price_column=self._cfg.price_column,
            grain=grain,
            lookback_periods=self._cap_lookback(lookback_periods),
            limit=200,
            filters=filters,
        )
        logger.info(f"domain_analytics_buyer_segment_analytics grain={grain.value}")
        return await self._run(sql, 'buyer_segment_analytics')

    async def market_summary(self, lookback_days: int = 30, filters: Optional[List[FilterSpec]] = None) -> List[Dict[str, Any]]:
        """Consolidated marketplace KPI summary — single row of top-level metrics.

        Covers Q82 (show me market stats summary). Returns one row containing:
        total listings, total sold, sell-through rate, avg/median/max sale price,
        total sale volume, avg/max bids, zero-bid count, avg DA, avg traffic.

        :param lookback_days: Rolling window in calendar days (default 30)
        :return: List with exactly one summary row
        """
        self._check_enabled('market_summary')
        if not self._cfg.sold_flag_column:
            raise RetrievalError('market_summary requires sold_flag_column in config')
        if not self._cfg.domain_authority_column:
            raise RetrievalError('market_summary requires domain_authority_column in config')
        if not self._cfg.traffic_column:
            raise RetrievalError('market_summary requires traffic_column in config')
        sql = market_summary_sql(
            table=self._cfg.source_table,
            price_column=self._cfg.price_column,
            bid_count_column=self._cfg.bid_count_column,
            sold_flag_column=self._cfg.sold_flag_column,
            domain_authority_column=self._cfg.domain_authority_column,
            traffic_column=self._cfg.traffic_column,
            time_column=self._cfg.auction_end_column,
            lookback_days=lookback_days,
            filters=filters,
        )
        logger.info(f"domain_analytics_market_summary lookback_days={lookback_days}")
        return await self._run(sql, 'market_summary')

    # ------------------------------------------------------------------
    # Group 4 — Platform infra methods (multi-dim aggregation)
    # ------------------------------------------------------------------

    async def multidim_tld_category(self, grain: TimeGrain = TimeGrain.DAY, limit: int = 2000, filters: Optional[List[FilterSpec]] = None) -> List[Dict[str, Any]]:
        """ROLLUP aggregation across tld + category with all subtotals.

        Produces rows for every (tld, category, period) combination plus
        subtotal rows where NULL indicates a rolled-up dimension.  Useful for
        multi-dimensional dashboards that need both per-TLD and per-category
        totals without multiple queries.

        :param limit: Maximum rows returned (ROLLUP expands cardinality significantly)
        :return: Rows with tld (nullable), category (nullable), period, event_count, total/avg/max price
        """
        self._check_enabled('multidim_tld_category')
        sql = multidim_aggregation_sql(
            table=self._cfg.source_table,
            metric_column=self._cfg.price_column,
            dimension_columns=[self._cfg.tld_column, self._cfg.category_column],
            time_column=self._cfg.auction_end_column,
            grain=grain,
            limit=limit,
            filters=filters,
        )
        logger.info(f"domain_analytics_multidim_tld_category grain={grain.value}")
        return await self._run(sql, 'multidim_tld_category')

    # ------------------------------------------------------------------
    # Group 5 — Bid velocity, watch, hold time, transaction analytics
    # ------------------------------------------------------------------

    async def bid_velocity_per_auction(self, window_hours: Optional[int] = None, prior_window_hours: Optional[int] = None, min_bids: int = 1, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Bid velocity acceleration per auction: recent N-hour bid rate vs prior N-hour rate from bid velocity MV.

        :param window_hours: Recent window size in hours; None uses bid_velocity_window_hours from config
        :param prior_window_hours: Total lookback hours (recent + prior); None uses 2× window_hours
        :param min_bids: Minimum bids in recent window to include auction
        :param limit: Max rows; None uses top_k_limit from config
        :return: Rows with auction_id, recent_bids, prior_bids, velocity_delta, velocity_ratio, unique_bidders
        """
        self._check_enabled('bid_velocity_per_auction')
        if not self._cfg.bid_velocity_mv:
            raise RetrievalError("bid_velocity_per_auction requires bid_velocity_mv in domain_analytics config")
        _window = window_hours if window_hours is not None else (self._cfg.bid_velocity_window_hours or 2)
        _prior = prior_window_hours if prior_window_hours is not None else (_window * 2)
        _limit = limit if limit is not None else (self._cfg.top_k_limit or 100)
        sql = bid_velocity_acceleration_sql(bid_velocity_mv=self._cfg.bid_velocity_mv, window_hours=_window, prior_window_hours=_prior, min_bids=min_bids, limit=_limit)
        logger.info(f"domain_analytics_bid_velocity_per_auction window_hours={_window}")
        return await self._run(sql, 'bid_velocity_per_auction')

    async def watch_density_analytics(self, lookback_days: int = 7, limit: int = 200) -> List[Dict[str, Any]]:
        """Watch density per listing: total watches and unique watchers from watch density MV.

        :param lookback_days: Trailing days to include; references the event_day column of the MV
        :return: Rows with member_item_id, event_day, total_watches, unique_watchers
        """
        self._check_enabled('watch_density_analytics')
        if not self._cfg.watch_density_mv:
            raise RetrievalError("watch_density_analytics requires watch_density_mv in domain_analytics config")
        if not self._cfg.watch_mv_item_column:
            raise RetrievalError("watch_density_analytics requires watch_mv_item_column in config")
        if not self._cfg.watch_mv_day_column:
            raise RetrievalError("watch_density_analytics requires watch_mv_day_column in config")
        if not self._cfg.watch_mv_count_state_column:
            raise RetrievalError("watch_density_analytics requires watch_mv_count_state_column in config")
        if not self._cfg.watch_mv_unique_state_column:
            raise RetrievalError("watch_density_analytics requires watch_mv_unique_state_column in config")
        _mv = self._cfg.watch_density_mv
        _item = self._cfg.watch_mv_item_column
        _day = self._cfg.watch_mv_day_column
        _cnt = self._cfg.watch_mv_count_state_column
        _uniq = self._cfg.watch_mv_unique_state_column
        sql = f"SELECT {_item}, {_day}, countMerge({_cnt}) AS total_watches, uniqMerge({_uniq}) AS unique_watchers FROM {_mv} WHERE {_day} >= today() - {int(lookback_days)} GROUP BY {_item}, {_day} HAVING total_watches > 0 ORDER BY total_watches DESC LIMIT {int(limit)}"
        logger.info(f"domain_analytics_watch_density_analytics lookback_days={lookback_days}")
        return await self._run(sql, 'watch_density_analytics')

    async def hold_time_distribution(self, group_column: str = 'auction_type_name', lookback_days: int = 90, limit: int = 100) -> List[Dict[str, Any]]:
        """Hold time (days-to-sell) distribution from the hold time AggregatingMergeTree MV.

        :param group_column: Dimension to group by; must be a column in the hold time MV (e.g. 'auction_type_name', 'tld')
        :return: Rows with group_column, total_sold, avg/min/max_hold_days, avg/total sale price
        """
        self._check_enabled('hold_time_distribution')
        if not self._cfg.hold_time_mv:
            raise RetrievalError("hold_time_distribution requires hold_time_mv in domain_analytics config")
        sql = hold_time_distribution_sql(hold_time_mv=self._cfg.hold_time_mv, group_column=group_column, lookback_days=lookback_days, limit=limit)
        logger.info(f"domain_analytics_hold_time_distribution group_column={group_column} lookback_days={lookback_days}")
        return await self._run(sql, 'hold_time_distribution')

    async def watch_bid_conversion(self, lookback_days: int = 7, min_watches: int = 1, limit: int = 200) -> List[Dict[str, Any]]:
        """Watch-to-bid conversion rate per listing joining watch density and bid velocity MVs.

        :return: Rows with member_item_id, total_watches, unique_watchers, total_bids, unique_bidders, watch_to_bid_rate
        """
        self._check_enabled('watch_bid_conversion')
        if not self._cfg.watch_density_mv or not self._cfg.bid_velocity_item_mv:
            raise RetrievalError("watch_bid_conversion requires watch_density_mv and bid_velocity_item_mv in domain_analytics config")
        sql = watch_bid_conversion_sql(watch_density_mv=self._cfg.watch_density_mv, bid_velocity_item_mv=self._cfg.bid_velocity_item_mv, lookback_days=lookback_days, min_watches=min_watches, limit=limit)
        logger.info(f"domain_analytics_watch_bid_conversion lookback_days={lookback_days}")
        return await self._run(sql, 'watch_bid_conversion')

    async def counter_offer_analytics(self, grain: TimeGrain = TimeGrain.DAY, lookback_periods: int = 30, limit: int = 200, filters: Optional[List[FilterSpec]] = None) -> List[Dict[str, Any]]:
        """Counter-offer rate and price delta from bid events over time.

        :return: Rows with period, total_bids, counter_offer_count, counter_offer_rate, avg_bid_usd, avg_counter_offer_usd, price_delta
        """
        self._check_enabled('counter_offer_analytics')
        if not self._cfg.bid_events_table:
            raise RetrievalError("counter_offer_analytics requires bid_events_table in domain_analytics config")
        sql = counter_offer_sql(bid_events_table=self._cfg.bid_events_table, time_column='event_utc_ts', grain=grain, lookback_periods=self._cap_lookback(lookback_periods), limit=limit, filters=filters)
        logger.info(f"domain_analytics_counter_offer_analytics grain={grain.value}")
        return await self._run(sql, 'counter_offer_analytics')

    async def price_realization_rate(self, grain: TimeGrain = TimeGrain.MONTH, lookback_periods: int = 3, limit: int = 100, filters: Optional[List[FilterSpec]] = None) -> List[Dict[str, Any]]:
        """Price realization rate (sale_price / listed_price) by TLD over time from domain transactions.

        :return: Rows with tld, period, total_transactions, avg_sale_price, avg_listed_price, avg/p25/p50/p75_realization_rate
        """
        self._check_enabled('price_realization_rate')
        if not self._cfg.transactions_table:
            raise RetrievalError("price_realization_rate requires transactions_table in domain_analytics config")
        sql = price_realization_sql(transactions_table=self._cfg.transactions_table, group_column=self._cfg.tld_column, time_column='sold_at', grain=grain, lookback_periods=self._cap_lookback(lookback_periods), limit=limit, filters=filters)
        logger.info(f"domain_analytics_price_realization_rate grain={grain.value}")
        return await self._run(sql, 'price_realization_rate')

    async def buyer_retention_cohorts(self, cohort_periods: int = 6, lookback_months: int = 12) -> List[Dict[str, Any]]:
        """Month-cohort buyer retention from domain transactions.

        :param cohort_periods: Number of retention months to track after first purchase
        :param lookback_months: How many cohort months to include
        :return: Rows with cohort_month, retention_period, cohort_size, retained_buyers, retention_rate
        """
        self._check_enabled('buyer_retention_cohorts')
        if not self._cfg.transactions_table:
            raise RetrievalError("buyer_retention_cohorts requires transactions_table in domain_analytics config")
        sql = buyer_retention_cohorts_sql(transactions_table=self._cfg.transactions_table, buyer_column='buyer_user_id', time_column='sold_at', cohort_periods=cohort_periods, lookback_months=lookback_months)
        logger.info(f"domain_analytics_buyer_retention_cohorts cohort_periods={cohort_periods}")
        return await self._run(sql, 'buyer_retention_cohorts')

    async def auction_timing_patterns(self, lookback_days: int = 90, filters: Optional[List[FilterSpec]] = None) -> List[Dict[str, Any]]:
        """Hour-of-day × day-of-week heatmap of auction close time vs bid volume and sell-through.

        :return: Rows with close_hour (0-23), close_dow (1=Mon..7=Sun), auction_count, sold_count, sell_through_rate, avg/total bid_count
        """
        self._check_enabled('auction_timing_patterns')
        if not self._cfg.sold_flag_column:
            raise RetrievalError('auction_timing_patterns requires sold_flag_column in config')
        sql = auction_timing_patterns_sql(table=self._cfg.source_table, time_column=self._cfg.auction_end_column, metric_column=self._cfg.bid_count_column, sold_flag_column=self._cfg.sold_flag_column, lookback_days=lookback_days, filters=filters)
        logger.info(f"domain_analytics_auction_timing_patterns lookback_days={lookback_days}")
        return await self._run(sql, 'auction_timing_patterns')

    async def name_structure_analytics(self, lookback_days: int = 90, min_count: int = 5, limit: int = 100, filters: Optional[List[FilterSpec]] = None) -> List[Dict[str, Any]]:
        """Domain name structural features (length, char class, hyphens) vs price and sell-through.

        :return: Rows with name_length_bucket, char_class, hyphen_count, auction_count, sold_count, sell_through_rate, avg/p50/max price
        """
        self._check_enabled('name_structure_analytics')
        if not self._cfg.sold_flag_column:
            raise RetrievalError('name_structure_analytics requires sold_flag_column in config')
        sql = name_structure_sql(table=self._cfg.source_table, domain_column=self._cfg.domain_column, price_column=self._cfg.price_column, sold_flag_column=self._cfg.sold_flag_column, time_column=self._cfg.auction_end_column, lookback_days=lookback_days, min_count=min_count, limit=limit, filters=filters)
        logger.info(f"domain_analytics_name_structure_analytics lookback_days={lookback_days}")
        return await self._run(sql, 'name_structure_analytics')

    async def cross_tld_price_spread(self, reference_tld: str = 'com', lookback_days: int = 90, min_sales: Optional[int] = None, limit: int = 100, filters: Optional[List[FilterSpec]] = None) -> List[Dict[str, Any]]:
        """Cross-TLD price spread: same keyword across TLDs vs reference TLD baseline.

        :param reference_tld: TLD used as price baseline (default 'com')
        :return: Rows with keyword, tld, avg_price, sold_count, ref_avg_price, price_spread_ratio
        """
        self._check_enabled('cross_tld_price_spread')
        if not self._cfg.sold_flag_column:
            raise RetrievalError('cross_tld_price_spread requires sold_flag_column in config')
        _min_sales = min_sales if min_sales is not None else self._cfg.comparable_min_sales
        if _min_sales is None:
            raise RetrievalError('cross_tld_price_spread requires min_sales param or comparable_min_sales in config')
        sql = cross_tld_spread_sql(table=self._cfg.source_table, domain_column=self._cfg.domain_column, tld_column=self._cfg.tld_column, price_column=self._cfg.price_column, sold_flag_column=self._cfg.sold_flag_column, time_column=self._cfg.auction_end_column, reference_tld=reference_tld, lookback_days=lookback_days, min_sales=_min_sales, limit=limit, filters=filters)
        logger.info(f"domain_analytics_cross_tld_price_spread reference_tld={reference_tld}")
        return await self._run(sql, 'cross_tld_price_spread')

    async def search_attribution(self, attribution_window_hours: int = 24, lookback_days: int = 30, min_signals: int = 1, limit: int = 100) -> List[Dict[str, Any]]:
        """Search signal to sale attribution within a time window.

        :return: Rows with signal_type, attributed_sales, total_revenue, avg_sale_price, signal_count, attribution_rate
        """
        self._check_enabled('search_attribution')
        if not self._cfg.transactions_table:
            raise RetrievalError("search_attribution requires transactions_table in domain_analytics config")
        if not self._cfg.transactions_sold_at_column:
            raise RetrievalError("search_attribution requires transactions_sold_at_column in config")
        sql = search_attribution_sql(signals_table=self._cfg.signals_table, transactions_table=self._cfg.transactions_table, signal_time_column=self._cfg.signals_time_column, transaction_time_column=self._cfg.transactions_sold_at_column, attribution_window_hours=attribution_window_hours, lookback_days=lookback_days, min_signals=min_signals, limit=limit)
        logger.info(f"domain_analytics_search_attribution lookback_days={lookback_days}")
        return await self._run(sql, 'search_attribution')

    async def registrar_hhi(self, lookback_days: int = 90, filters: Optional[List[FilterSpec]] = None) -> List[Dict[str, Any]]:
        """Herfindahl-Hirschman Index for registrar market concentration among sold domains.

        :return: Rows with registrar_name, sold_count, market_share, hhi_contribution (sum = total HHI)
        """
        self._check_enabled('registrar_hhi')
        if not self._cfg.registrar_column:
            raise RetrievalError('registrar_hhi requires registrar_column in config')
        if not self._cfg.sold_flag_column:
            raise RetrievalError('registrar_hhi requires sold_flag_column in config')
        sql = registrar_hhi_sql(table=self._cfg.source_table, registrar_column=self._cfg.registrar_column, sold_flag_column=self._cfg.sold_flag_column, time_column=self._cfg.auction_end_column, lookback_days=lookback_days, filters=filters)
        logger.info(f"domain_analytics_registrar_hhi lookback_days={lookback_days}")
        return await self._run(sql, 'registrar_hhi')

    async def comparable_sale_price(self, lookback_days: Optional[int] = None, min_sales: Optional[int] = None, limit: int = 200, filters: Optional[List[FilterSpec]] = None) -> List[Dict[str, Any]]:
        """Comparable sale price cohorts: p25/p50/p75 price by TLD × name-length bucket.

        :param lookback_days: Trailing days for transaction lookback; None uses comparable_min_sales-driven default
        :return: Rows with tld, name_length_bucket, cohort_size, p25/p50/p75/avg/min/max sale price
        """
        self._check_enabled('comparable_sale_price')
        if not self._cfg.transactions_table:
            raise RetrievalError("comparable_sale_price requires transactions_table in domain_analytics config")
        _days = lookback_days if lookback_days is not None else 365
        _min = min_sales if min_sales is not None else (self._cfg.comparable_min_sales or 5)
        sql = comparable_sale_price_sql(transactions_table=self._cfg.transactions_table, tld_column=self._cfg.tld_column, price_column='sale_price', domain_column=self._cfg.domain_column, time_column='sold_at', lookback_days=_days, min_sales=_min, limit=limit, filters=filters)
        logger.info(f"domain_analytics_comparable_sale_price lookback_days={_days}")
        return await self._run(sql, 'comparable_sale_price')


    # ------------------------------------------------------------------
    # Group 6 — Lifecycle, cohort, distribution, funnel, smoothing
    # ------------------------------------------------------------------

    async def domain_lifecycle_stages(
        self,
        new_days: Optional[int] = None,
        dormant_days: Optional[int] = None,
        grain: TimeGrain = TimeGrain.DAY,
        limit: int = 500,
        filters: Optional[List[FilterSpec]] = None,
    ) -> List[Dict[str, Any]]:
        """Classify each domain as new/active/dormant/at_risk/churned by age and recency.

        :param new_days: Age threshold (days) for the 'new' lifecycle stage; uses lifecycle_new_days from config
        :param dormant_days: Inactivity threshold (days) for the 'dormant' stage; uses lifecycle_dormant_days from config
        :param grain: Time grain for last_active_bucket column
        :param limit: Maximum rows returned
        :return: Rows with domain_name, first_seen, last_activity, total_events,
            age_days, days_since_active, lifecycle_stage, last_active_bucket
        """
        self._check_enabled('domain_lifecycle_stages')
        _new_days = new_days if new_days is not None else self._cfg.lifecycle_new_days
        _dormant_days = dormant_days if dormant_days is not None else self._cfg.lifecycle_dormant_days
        if _new_days is None:
            raise RetrievalError('domain_lifecycle_stages requires new_days param or lifecycle_new_days in config')
        if _dormant_days is None:
            raise RetrievalError('domain_lifecycle_stages requires dormant_days param or lifecycle_dormant_days in config')
        sql = lifecycle_sql(
            table=self._cfg.source_table,
            entity_column=self._cfg.domain_column,
            created_column=self._cfg.created_column,
            activity_column=self._cfg.auction_end_column,
            time_column=self._cfg.auction_end_column,
            new_days=_new_days,
            dormant_days=_dormant_days,
            grain=grain,
            limit=limit,
            filters=filters,
        )
        logger.info(f"domain_analytics_domain_lifecycle_stages new_days={_new_days} dormant_days={_dormant_days}")
        return await self._run(sql, 'domain_lifecycle_stages')

    async def cohort_price_comparison(
        self,
        cohort_a: str,
        cohort_b: str,
        cohort_column: str = 'tld',
        grain: TimeGrain = TimeGrain.DAY,
        lookback_periods: int = 30,
        filters: Optional[List[FilterSpec]] = None,
    ) -> List[Dict[str, Any]]:
        """Side-by-side price comparison between two named cohorts over time.

        :param cohort_a: Value of cohort_column for cohort A (e.g. 'com')
        :param cohort_b: Value of cohort_column for cohort B (e.g. 'net')
        :param cohort_column: Column to split on ('tld', 'category_name', 'buyer_segment')
        :return: Rows with period, metric_a, metric_b, count_a, count_b,
            absolute_delta, relative_delta_pct
        """
        self._check_enabled('cohort_price_comparison')
        col_map: Dict[str, Optional[str]] = {
            'tld': self._cfg.tld_column,
            'category_name': self._cfg.category_name_column,
            'buyer_segment': self._cfg.buyer_segment_column,
        }
        if cohort_column in col_map:
            resolved_col = col_map[cohort_column]
            if resolved_col is None:
                raise RetrievalError(
                    f'cohort_price_comparison: cohort_column={cohort_column!r} requires '
                    f'the corresponding config column to be set'
                )
        else:
            resolved_col = cohort_column
        sql = comparative_cohort_sql(
            table=self._cfg.source_table,
            metric_column=self._cfg.price_column,
            cohort_column=resolved_col,
            time_column=self._cfg.auction_end_column,
            cohort_a=cohort_a,
            cohort_b=cohort_b,
            grain=grain,
            lookback_periods=self._cap_lookback(lookback_periods),
            filters=filters,
        )
        logger.info(f"domain_analytics_cohort_price_comparison cohort_a={cohort_a!r} cohort_b={cohort_b!r} col={cohort_column!r}")
        return await self._run(sql, 'cohort_price_comparison')

    async def price_quantile_distribution(
        self,
        group_column: str = 'tld',
        grain: TimeGrain = TimeGrain.DAY,
        lookback_periods: int = 30,
        quantiles: Optional[List[float]] = None,
        limit: int = 500,
        filters: Optional[List[FilterSpec]] = None,
    ) -> List[Dict[str, Any]]:
        """Quantile price distribution (p10/p25/p50/p75/p90/p99) per group and time period.

        :param group_column: Dimension to group by ('tld', 'category_name')
        :param quantiles: Custom quantile list; defaults to [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
        :return: Rows with group_column, period, total_count, avg/min/max_metric, pN columns
        """
        self._check_enabled('price_quantile_distribution')
        col_map_pqd: Dict[str, Optional[str]] = {
            'tld': self._cfg.tld_column,
            'category_name': self._cfg.category_name_column,
        }
        if group_column in col_map_pqd:
            resolved_col = col_map_pqd[group_column]
            if resolved_col is None:
                raise RetrievalError(
                    f'price_quantile_distribution: group_column={group_column!r} requires '
                    f'the corresponding config column to be set'
                )
        else:
            resolved_col = group_column
        sql = price_distribution_sql(
            table=self._cfg.source_table,
            metric_column=self._cfg.price_column,
            group_column=resolved_col,
            time_column=self._cfg.auction_end_column,
            grain=grain,
            lookback_periods=self._cap_lookback(lookback_periods),
            quantiles=quantiles,
            limit=limit,
            filters=filters,
        )
        logger.info(f"domain_analytics_price_quantile_distribution group_column={group_column} grain={grain.value}")
        return await self._run(sql, 'price_quantile_distribution')

    async def auction_window_funnel(
        self,
        funnel_values: Optional[List[str]] = None,
        window_seconds: Optional[int] = None,
        grain: TimeGrain = TimeGrain.DAY,
        lookback_periods: int = 30,
        filters: Optional[List[FilterSpec]] = None,
    ) -> List[Dict[str, Any]]:
        """Ordered conversion funnel using ClickHouse windowFunnel on signal events.

        :param funnel_values: Ordered signal_type values for funnel steps
        :param window_seconds: Max seconds between first and last step; uses funnel_window_seconds from config
        :param grain: Time grain for period bucketing
        :return: Rows with period, reached_step_N counts for each funnel level
        """
        self._check_enabled('auction_window_funnel')
        _window = window_seconds if window_seconds is not None else self._cfg.funnel_window_seconds
        if _window is None:
            raise RetrievalError('auction_window_funnel requires window_seconds param or funnel_window_seconds in config')
        _funnel = funnel_values if funnel_values is not None else None
        if _funnel is None:
            raise RetrievalError('auction_window_funnel requires funnel_values param')
        sql = window_funnel_sql(
            table=self._cfg.signals_table,
            entity_column=self._cfg.signals_user_column,
            time_column=self._cfg.signals_time_column,
            event_column=self._cfg.signals_event_column,
            funnel_values=_funnel,
            window_seconds=_window,
            grain=grain,
            lookback_periods=self._cap_lookback(lookback_periods),
            filters=filters,
        )
        logger.info(f"domain_analytics_auction_window_funnel funnel_steps={len(_funnel)} window_seconds={_window}")
        return await self._run(sql, 'auction_window_funnel')

    async def price_moving_average(
        self,
        entity_column: str = 'tld',
        window_size: Optional[int] = None,
        grain: TimeGrain = TimeGrain.DAY,
        lookback_periods: int = 30,
        limit: int = 500,
        filters: Optional[List[FilterSpec]] = None,
    ) -> List[Dict[str, Any]]:
        """Rolling N-period moving average of price per entity (TLD or domain).

        :param entity_column: Dimension to partition by ('tld' or 'domain')
        :param window_size: Number of preceding periods in the rolling window; uses moving_average_window_size from config
        :param grain: Time grain for period bucketing
        :return: Rows with entity_column, period, period_avg, moving_avg
        """
        self._check_enabled('price_moving_average')
        _window = window_size if window_size is not None else self._cfg.moving_average_window_size
        if _window is None:
            raise RetrievalError('price_moving_average requires window_size param or moving_average_window_size in config')
        col_map_pma = {
            'tld': self._cfg.tld_column,
            'domain': self._cfg.domain_column,
        }
        resolved_col = col_map_pma.get(entity_column, entity_column)
        sql = moving_average_sql(
            table=self._cfg.source_table,
            metric_column=self._cfg.price_column,
            entity_column=resolved_col,
            time_column=self._cfg.auction_end_column,
            grain=grain,
            window_size=_window,
            lookback_periods=self._cap_lookback(lookback_periods),
            limit=limit,
            filters=filters,
        )
        logger.info(f"domain_analytics_price_moving_average entity_column={entity_column} window_size={_window}")
        return await self._run(sql, 'price_moving_average')

    # ------------------------------------------------------------------
    # Enriched table queries — powered by EnrichedTablesBuilder ETL output
    # ------------------------------------------------------------------

    async def enriched_tld_analysis(
        self,
        lookback_weeks: int = 12,
        anomaly_only: bool = False,
        momentum_labels: Optional[List[str]] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Query enriched_tld_features for trend, anomaly, momentum, and forecast data.

        Returns richer TLD-level analytics than the raw MV queries because the
        enriched table carries Python-computed columns (growth_pct, forecast_value,
        is_anomaly, tld_momentum_label, sell_through_tier) that cannot be expressed
        as CH aggregate functions.

        :param lookback_weeks: Rolling week window
        :param anomaly_only: When True, restrict to rows where is_anomaly=1
        :param momentum_labels: Optional list of momentum labels to filter by
            (e.g. ['rising', 'stable'])
        :param limit: Maximum rows returned
        :return: Rows from enriched_tld_features
        """
        self._check_enabled('enriched_tld_analysis')
        where_parts = [f"event_week >= toStartOfWeek(now()) - INTERVAL {int(lookback_weeks)} WEEK"]
        if anomaly_only:
            where_parts.append("is_anomaly = 1")
        if momentum_labels:
            safe_labels = [str(lbl).replace("'", "") for lbl in momentum_labels]
            labels_sql = ', '.join(f"'{lbl}'" for lbl in safe_labels)
            where_parts.append(f"tld_momentum_label IN ({labels_sql})")
        where_clause = ' AND '.join(where_parts)
        sql = (
            f"SELECT tld, event_week, avg_price, total_auctions, sold_auctions, "
            f"sell_through_rate, avg_bids, avg_govalue, growth_pct, forecast_value, "
            f"z_score, is_anomaly, tld_momentum_pct, tld_momentum_label, sell_through_tier "
            f"FROM signals_platform_cln.enriched_tld_features "
            f"WHERE {where_clause} "
            f"ORDER BY tld ASC, event_week DESC "
            f"LIMIT {int(limit)}"
        )
        logger.info(f"domain_analytics_enriched_tld_analysis anomaly_only={anomaly_only} lookback_weeks={lookback_weeks}")
        return await self._run(sql, 'enriched_tld_analysis')

    async def enriched_hot_auctions(
        self,
        limit: int = 100,
        urgency_labels: Optional[List[str]] = None,
        heat_scores: Optional[List[str]] = None,
        min_composite_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Query enriched_auction_features for hot, urgent, or high-demand listings.

        Combines heat_score (bid velocity tier), time_urgency_label (hours to close),
        composite_score (weighted bid+price+govalue rank), watcher_count, and
        unique_bidder_count — data that would require multiple CH joins to assemble
        at query time is pre-joined by the Python builder.

        :param limit: Maximum rows returned
        :param urgency_labels: Optional filter list (e.g. ['critical', 'high'])
        :param heat_scores: Optional filter list (e.g. ['high', 'medium'])
        :param min_composite_score: Minimum composite score gate (0.0–1.0)
        :return: Rows from enriched_auction_features
        """
        self._check_enabled('enriched_hot_auctions')
        where_parts = ["ends_at > now()"]
        if urgency_labels:
            safe = [str(lbl).replace("'", "") for lbl in urgency_labels]
            where_parts.append(f"time_urgency_label IN ({', '.join(repr(l) for l in safe)})")
        if heat_scores:
            safe = [str(h).replace("'", "") for h in heat_scores]
            where_parts.append(f"heat_score IN ({', '.join(repr(h) for h in safe)})")
        if min_composite_score > 0.0:
            where_parts.append(f"composite_score >= {float(min_composite_score)}")
        where_clause = ' AND '.join(where_parts)
        sql = (
            f"SELECT auction_id, domain_name, tld, current_price, bid_count, ends_at, "
            f"govalue_score, heat_score, time_urgency_label, lifecycle_stage, "
            f"fair_value_p25, fair_value_p75, sell_through_prob, sell_through_tier, "
            f"composite_score, tld_rank, watcher_count, unique_bidder_count, category_name "
            f"FROM signals_platform_cln.enriched_auction_features "
            f"WHERE {where_clause} "
            f"ORDER BY composite_score DESC, watcher_count DESC "
            f"LIMIT {int(limit)}"
        )
        logger.info(f"domain_analytics_enriched_hot_auctions urgency_labels={urgency_labels} heat_scores={heat_scores} limit={limit}")
        return await self._run(sql, 'enriched_hot_auctions')

    async def enriched_domain_ranking(
        self,
        tld: Optional[str] = None,
        category_name: Optional[str] = None,
        top_n: int = 50,
    ) -> List[Dict[str, Any]]:
        """Top-ranked domains within a TLD or category from enriched_auction_features.

        Uses the pre-computed tld_rank (dense rank by composite_score within TLD)
        so the result is sorted by competitive standing without a scan-time aggregation.

        :param tld: Optional TLD filter (e.g. 'com')
        :param category_name: Optional category name filter (e.g. 'ai')
        :param top_n: Maximum rows returned
        :return: Rows sorted by tld_rank ASC (rank 1 = highest composite score in TLD)
        """
        self._check_enabled('enriched_domain_ranking')
        where_parts = ["ends_at > now()"]
        if tld:
            safe_tld = str(tld).replace("'", "")
            where_parts.append(f"tld = '{safe_tld}'")
        if category_name:
            safe_cat = str(category_name).replace("'", "")
            where_parts.append(f"category_name = '{safe_cat}'")
        where_clause = ' AND '.join(where_parts)
        sql = (
            f"SELECT auction_id, domain_name, tld, current_price, bid_count, ends_at, "
            f"govalue_score, composite_score, tld_rank, heat_score, time_urgency_label, "
            f"fair_value_p25, fair_value_p75, sell_through_tier, watcher_count, "
            f"unique_bidder_count, category_name "
            f"FROM signals_platform_cln.enriched_auction_features "
            f"WHERE {where_clause} "
            f"ORDER BY tld_rank ASC "
            f"LIMIT {int(top_n)}"
        )
        logger.info(f"domain_analytics_enriched_domain_ranking tld={tld!r} category_name={category_name!r} top_n={top_n}")
        return await self._run(sql, 'enriched_domain_ranking')


__all__ = ['DomainAnalyticsEngine']
