"""Periodic ETL that populates enriched CH tables via EnrichmentPipeline.

Reads aggregated data from the base auction tables, applies Python-side
enrichment steps (growth rates, anomaly flags, heat scores, urgency labels,
fair-value bands, sell-through tiers, percentile ranks), and writes the
enriched rows back to:
  - ``signals_platform_cln.enriched_tld_features``  (per-TLD weekly)
  - ``signals_platform_cln.enriched_auction_features``  (per-auction)

Run periodically via ``EnrichedTablesBuilder.loop_forever(interval_hours=6)``
or on-demand via ``build_all()``.  ClickHouse is the only substrate; a failed
CH read/write logs a warning and returns without raising so the service stays
up when CH is temporarily unavailable.
"""
from __future__ import annotations

import asyncio
import functools
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from semantic_search.analytics.enrichment_pipeline import (
    EnrichmentContext,
    EnrichmentPipeline,
    enrich_anomaly_flags,
    enrich_fair_value_band,
    enrich_forecast_values,
    enrich_growth_rates,
    enrich_heat_score,
    enrich_lifecycle_stage,
    enrich_percentile_ranks,
    enrich_sell_through_prob,
    enrich_time_urgency,
    enrich_tld_momentum,
    enrich_unique_bidders,
    enrich_watcher_count,
)
from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)

_SOURCE_TABLE = 'signals_platform_cln.auction_audit_cln'
_TLD_FEATURES_TABLE = 'signals_platform_cln.enriched_tld_features'
_AUCTION_FEATURES_TABLE = 'signals_platform_cln.enriched_auction_features'

_INSERT_BATCH = 2000


@dataclass
class BuildResult:
    table: str
    rows_written: int
    elapsed_ms: float
    errors: List[str] = field(default_factory=list)


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _norm_col(rows: List[Dict], col: str) -> Dict[Any, float]:
    """Return per-row min-max normalised values keyed by row index."""
    vals = [_safe_float(r.get(col)) for r in rows]
    clean = [v for v in vals if v is not None]
    if not clean:
        return {i: 0.0 for i in range(len(rows))}
    lo, hi = min(clean), max(clean)
    span = hi - lo if hi != lo else 1.0
    return {i: (v - lo) / span if v is not None else 0.0 for i, v in enumerate(vals)}


def _composite_score(rows: List[Dict], bid_col: str = 'bid_count', price_col: str = 'current_price', govalue_col: str = 'govalue_score') -> List[Dict]:
    """Add composite_score = 0.5*norm_bids + 0.3*norm_price + 0.2*norm_govalue."""
    nb = _norm_col(rows, bid_col)
    np_ = _norm_col(rows, price_col)
    ng = _norm_col(rows, govalue_col)
    result = []
    for i, row in enumerate(rows):
        score = round(0.5 * nb[i] + 0.3 * np_[i] + 0.2 * ng[i], 6)
        result.append({**row, 'composite_score': score})
    return result


def _format_value(v: Any) -> str:
    """Format a Python value as a ClickHouse SQL literal."""
    if v is None:
        return 'NULL'
    if isinstance(v, bool):
        return '1' if v else '0'
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v).replace("'", "\\'")
    return f"'{s}'"


def _build_insert_sql(table: str, cols: List[str], rows: List[Dict]) -> str:
    col_list = ', '.join(cols)
    value_rows = []
    for row in rows:
        vals = ', '.join(_format_value(row.get(c)) for c in cols)
        value_rows.append(f'({vals})')
    return f"INSERT INTO {table} ({col_list}) VALUES {', '.join(value_rows)}"


class EnrichedTablesBuilder:
    """Reads CH aggregates, applies EnrichmentPipeline, writes enriched tables.

    :param executor: ClickHouseExecutor — shared handle (same as analytics router)
    :param watch_density_mv: Optional table name for watch density MV
    :param bid_velocity_item_mv: Optional table name for bid velocity by item MV
    :param transactions_table: Optional domain transactions table for fair value bands
    :param sell_through_mv: Optional sell-through MV for sell-through probability
    """

    def __init__(
        self,
        executor: Any,
        watch_density_mv: Optional[str] = None,
        bid_velocity_item_mv: Optional[str] = None,
        transactions_table: Optional[str] = None,
        sell_through_mv: Optional[str] = 'signals_platform_cln.mv_sell_through_by_tld_day',
    ) -> None:
        self._executor = executor
        self._watch_density_mv = watch_density_mv
        self._bid_velocity_item_mv = bid_velocity_item_mv
        self._transactions_table = transactions_table
        self._sell_through_mv = sell_through_mv

    async def _query(self, sql: str, timeout: float = 60.0) -> List[Dict]:
        try:
            result = await self._executor.execute(sql)
            return list(result.rows)
        except Exception as e:
            logger.warning(f"enriched_builder_query_error error_type={type(e).__name__} error={e}")
            return []

    async def _insert(self, table: str, cols: List[str], rows: List[Dict], timeout: float = 120.0) -> int:
        if not rows:
            return 0
        written = 0
        for i in range(0, len(rows), _INSERT_BATCH):
            batch = rows[i: i + _INSERT_BATCH]
            sql = _build_insert_sql(table, cols, batch)
            try:
                await self._executor.execute_insert(sql, timeout_seconds=timeout)
                written += len(batch)
            except Exception as e:
                logger.warning(f"enriched_builder_insert_error table={table} batch_start={i} error_type={type(e).__name__} error={e}")
        return written

    # ------------------------------------------------------------------
    # TLD features build
    # ------------------------------------------------------------------

    async def build_tld_features(self, lookback_weeks: int = 13) -> BuildResult:
        """Fetch TLD weekly aggregates, enrich, write to enriched_tld_features."""
        t0 = time.monotonic()
        errors: List[str] = []

        # Fetch TLD weekly aggregates with rolling stddev for z-score computation.
        sql = f"""\
SELECT
    tld,
    toStartOfWeek(ends_at)               AS event_week,
    round(avg(current_price), 4)         AS avg_price,
    count()                              AS total_auctions,
    countIf(sold_flag = 1)               AS sold_auctions,
    round(countIf(sold_flag = 1) / count(), 4)  AS sell_through_rate,
    round(avg(bid_count), 4)             AS avg_bids,
    round(avg(govalue_score), 4)         AS avg_govalue,
    round(
        (avg(current_price) - avgIf(current_price, toStartOfWeek(ends_at) < toStartOfWeek(now())))
        / nullIf(stddevPop(current_price), 0), 4
    ) AS z_score
FROM {_SOURCE_TABLE}
WHERE ends_at >= toStartOfWeek(now()) - INTERVAL {int(lookback_weeks)} WEEK
  AND ends_at < now()
  AND tld != ''
GROUP BY tld, event_week
ORDER BY tld ASC, event_week ASC
LIMIT 50000"""

        rows = await self._query(sql)
        if not rows:
            return BuildResult(table=_TLD_FEATURES_TABLE, rows_written=0, elapsed_ms=(time.monotonic() - t0) * 1000)

        # Build sell-through lookup for enrich_sell_through_prob
        sell_through_data: Optional[Dict[str, float]] = None
        if self._sell_through_mv:
            # sell_through_data keyed by tld → fraction sold
            st_sql = (
                f"SELECT tld, "
                f"round(sumMerge(sold_count_state) / greatest(1, countMerge(count_state)), 4) AS st_rate "
                f"FROM {self._sell_through_mv} WHERE event_day >= today() - 90 "
                f"GROUP BY tld LIMIT 5000"
            )
            st_data_rows = await self._query(st_sql)
            if st_data_rows:
                sell_through_data = {str(r.get('tld', '')): float(r.get('st_rate') or 0) for r in st_data_rows if r.get('tld')}

        # Build momentum lookup from the rows themselves (WoW change per TLD)
        momentum_data: Optional[Dict[str, float]] = None
        if rows:
            by_tld: Dict[str, List] = {}
            for r in rows:
                by_tld.setdefault(str(r.get('tld', '')), []).append(r)
            mom: Dict[str, float] = {}
            for tld, tld_rows in by_tld.items():
                sorted_rows = sorted(tld_rows, key=lambda x: str(x.get('event_week', '')))
                if len(sorted_rows) >= 2:
                    prev_p = _safe_float(sorted_rows[-2].get('avg_price'))
                    curr_p = _safe_float(sorted_rows[-1].get('avg_price'))
                    if prev_p and prev_p != 0 and curr_p is not None:
                        mom[tld] = round((curr_p - prev_p) / prev_p * 100.0, 4)
            if mom:
                momentum_data = mom

        ctx = EnrichmentContext(
            request_id='enriched_tld_build',
            entity_column='tld',
            time_column='event_week',
            metric_column='avg_price',
        )

        pipeline = EnrichmentPipeline()
        pipeline = pipeline.then(functools.partial(enrich_growth_rates, value_col='avg_price', entity_col='tld', period_col='event_week'))
        pipeline = pipeline.then(functools.partial(enrich_anomaly_flags, z_col='z_score', threshold=2.5))
        pipeline = pipeline.then(functools.partial(enrich_forecast_values, value_col='avg_price', entity_col='tld', period_col='event_week'))
        if momentum_data is not None:
            pipeline = pipeline.then(functools.partial(enrich_tld_momentum, tld_col='tld', momentum_data=momentum_data))
        if sell_through_data is not None:
            pipeline = pipeline.then(functools.partial(enrich_sell_through_prob, tld_col='tld', sell_through_data=sell_through_data))

        result = pipeline.run(rows, ctx)
        enriched = result.rows
        logger.info(
            f"enriched_tld_pipeline steps_applied={result.steps_applied} "
            f"skipped={result.skipped_steps} errors={result.errors} rows={len(enriched)}"
        )
        if result.errors:
            errors.extend(result.errors)

        # Map to table columns
        _TLD_COLS = [
            'tld', 'event_week', 'avg_price', 'total_auctions', 'sold_auctions',
            'sell_through_rate', 'avg_bids', 'avg_govalue', 'growth_pct', 'forecast_value',
            'z_score', 'is_anomaly', 'tld_momentum_pct', 'tld_momentum_label',
            'sell_through_tier',
        ]

        def _to_tld_row(r: Dict) -> Dict:
            return {
                'tld': str(r.get('tld', '')),
                'event_week': str(r.get('event_week', '')),
                'avg_price': _safe_float(r.get('avg_price')) or 0.0,
                'total_auctions': int(r.get('total_auctions') or 0),
                'sold_auctions': int(r.get('sold_auctions') or 0),
                'sell_through_rate': _safe_float(r.get('sell_through_rate')) or 0.0,
                'avg_bids': _safe_float(r.get('avg_bids')) or 0.0,
                'avg_govalue': _safe_float(r.get('avg_govalue')) or 0.0,
                'growth_pct': _safe_float(r.get('growth_pct')),
                'forecast_value': _safe_float(r.get('forecast_value')),
                'z_score': _safe_float(r.get('z_score')),
                'is_anomaly': 1 if r.get('is_anomaly') else 0,
                'tld_momentum_pct': _safe_float(r.get('tld_momentum_pct')),
                'tld_momentum_label': str(r.get('tld_momentum_label') or ''),
                'sell_through_tier': str(r.get('sell_through_tier') or ''),
            }

        mapped = [_to_tld_row(r) for r in enriched]
        written = await self._insert(_TLD_FEATURES_TABLE, _TLD_COLS, mapped)
        elapsed = (time.monotonic() - t0) * 1000.0
        logger.info(f"enriched_tld_features_built rows_written={written} elapsed_ms={elapsed:.1f}")
        return BuildResult(table=_TLD_FEATURES_TABLE, rows_written=written, elapsed_ms=elapsed, errors=errors)

    # ------------------------------------------------------------------
    # Auction features build
    # ------------------------------------------------------------------

    async def build_auction_features(self, limit: int = 50000) -> BuildResult:
        """Fetch active auctions, enrich, write to enriched_auction_features."""
        t0 = time.monotonic()
        errors: List[str] = []

        sql = f"""\
SELECT
    auction_id,
    domain_name,
    tld,
    current_price,
    bid_count,
    ends_at,
    govalue_score,
    domain_age_days,
    monthly_traffic,
    category_name
FROM {_SOURCE_TABLE}
WHERE ends_at > now()
  AND sold_flag = 0
ORDER BY bid_count DESC, current_price DESC
LIMIT {int(limit)}"""

        rows = await self._query(sql, timeout=90.0)
        if not rows:
            return BuildResult(table=_AUCTION_FEATURES_TABLE, rows_written=0, elapsed_ms=(time.monotonic() - t0) * 1000)

        # Watcher counts lookup
        watcher_data: Optional[Dict[int, int]] = None
        if self._watch_density_mv:
            auction_ids = [int(r['auction_id']) for r in rows if r.get('auction_id')]
            if auction_ids:
                watch_sql = (
                    f"SELECT member_item_id, toUInt32(countMerge(active_watch_state)) AS total_watches "
                    f"FROM {self._watch_density_mv} "
                    f"WHERE event_day >= today() - 7 "
                    f"GROUP BY member_item_id LIMIT 100000"
                )
                watch_rows = await self._query(watch_sql)
                if watch_rows:
                    watcher_data = {int(r['member_item_id']): int(r.get('total_watches') or 0) for r in watch_rows if r.get('member_item_id')}

        # Unique bidder counts lookup
        bidder_data: Optional[Dict[int, int]] = None
        if self._bid_velocity_item_mv:
            bid_sql = (
                f"SELECT member_item_id, toUInt32(uniqMerge(unique_bidders_state)) AS unique_bidders "
                f"FROM {self._bid_velocity_item_mv} "
                f"WHERE event_hour >= now() - INTERVAL 48 HOUR "
                f"GROUP BY member_item_id LIMIT 100000"
            )
            bid_rows = await self._query(bid_sql)
            if bid_rows:
                bidder_data = {int(r['member_item_id']): int(r.get('unique_bidders') or 0) for r in bid_rows if r.get('member_item_id')}

        # Fair value bands from transactions table (p25/p75 by TLD)
        comparable_data: Optional[Dict[str, Dict[str, float]]] = None
        if self._transactions_table:
            fv_sql = (
                f"SELECT tld, "
                f"round(quantile(0.25)(sale_price), 2) AS p25, "
                f"round(quantile(0.75)(sale_price), 2) AS p75 "
                f"FROM {self._transactions_table} "
                f"WHERE sold_at >= today() - 365 AND sale_price > 0 "
                f"GROUP BY tld HAVING count() >= 3 LIMIT 2000"
            )
            fv_rows = await self._query(fv_sql)
            if fv_rows:
                comparable_data = {
                    str(r['tld']): {'p25': float(r.get('p25') or 0), 'p75': float(r.get('p75') or 0)}
                    for r in fv_rows if r.get('tld')
                }

        # Sell-through probability lookup by TLD
        sell_through_data: Optional[Dict[str, float]] = None
        if self._sell_through_mv:
            st_sql = (
                f"SELECT tld, "
                f"round(sumMerge(sold_count_state) / greatest(1, countMerge(count_state)), 4) AS st_rate "
                f"FROM {self._sell_through_mv} WHERE event_day >= today() - 90 "
                f"GROUP BY tld LIMIT 5000"
            )
            st_rows = await self._query(st_sql)
            if st_rows:
                sell_through_data = {str(r['tld']): float(r.get('st_rate') or 0) for r in st_rows if r.get('tld')}

        # Compute composite score first (needed for percentile rank)
        rows = _composite_score(rows)

        ctx = EnrichmentContext(
            request_id='enriched_auction_build',
            entity_column='auction_id',
            time_column='ends_at',
            metric_column='composite_score',
        )

        pipeline = EnrichmentPipeline()
        pipeline = pipeline.then(functools.partial(enrich_heat_score, velocity_delta_col='bid_count'))
        pipeline = pipeline.then(functools.partial(enrich_time_urgency, ends_at_col='ends_at'))
        pipeline = pipeline.then(functools.partial(
            enrich_lifecycle_stage,
            age_col='domain_age_days',
            activity_col='bid_count',
            new_threshold=90,
            active_threshold=1,
            dormant_threshold=0,
        ))
        pipeline = pipeline.then(functools.partial(enrich_percentile_ranks, sort_col='composite_score', partition_col='tld', output_col='tld_rank'))
        if watcher_data is not None:
            pipeline = pipeline.then(functools.partial(
                enrich_watcher_count,
                item_id_col='auction_id',
                watcher_data=watcher_data,
            ))
        if bidder_data is not None:
            pipeline = pipeline.then(functools.partial(
                enrich_unique_bidders,
                auction_id_col='auction_id',
                bidder_data=bidder_data,
            ))
        if comparable_data is not None:
            pipeline = pipeline.then(functools.partial(
                enrich_fair_value_band,
                tld_col='tld',
                comparable_data=comparable_data,
            ))
        if sell_through_data is not None:
            pipeline = pipeline.then(functools.partial(
                enrich_sell_through_prob,
                tld_col='tld',
                sell_through_data=sell_through_data,
            ))

        result = pipeline.run(rows, ctx)
        enriched = result.rows
        logger.info(
            f"enriched_auction_pipeline steps_applied={result.steps_applied} "
            f"skipped={result.skipped_steps} errors={result.errors} rows={len(enriched)}"
        )
        if result.errors:
            errors.extend(result.errors)

        _AUCTION_COLS = [
            'auction_id', 'domain_name', 'tld', 'current_price', 'bid_count',
            'ends_at', 'govalue_score', 'domain_age_days', 'monthly_traffic',
            'heat_score', 'time_urgency_label', 'lifecycle_stage',
            'fair_value_p25', 'fair_value_p75', 'sell_through_prob', 'sell_through_tier',
            'composite_score', 'tld_rank', 'watcher_count', 'unique_bidder_count',
            'category_name',
        ]

        def _to_auction_row(r: Dict) -> Dict:
            ends_raw = r.get('ends_at')
            ends_str = str(ends_raw) if ends_raw is not None else '1970-01-01 00:00:00'
            return {
                'auction_id': int(r.get('auction_id') or 0),
                'domain_name': str(r.get('domain_name') or ''),
                'tld': str(r.get('tld') or ''),
                'current_price': _safe_float(r.get('current_price')) or 0.0,
                'bid_count': int(r.get('bid_count') or 0),
                'ends_at': ends_str,
                'govalue_score': _safe_float(r.get('govalue_score')) or 0.0,
                'domain_age_days': int(r.get('domain_age_days') or 0),
                'monthly_traffic': int(r.get('monthly_traffic') or 0),
                'heat_score': str(r.get('heat_score') or ''),
                'time_urgency_label': str(r.get('time_urgency_label') or ''),
                'lifecycle_stage': str(r.get('lifecycle_stage') or ''),
                'fair_value_p25': _safe_float(r.get('fair_value_p25')),
                'fair_value_p75': _safe_float(r.get('fair_value_p75')),
                'sell_through_prob': _safe_float(r.get('sell_through_prob')),
                'sell_through_tier': str(r.get('sell_through_tier') or ''),
                'composite_score': _safe_float(r.get('composite_score')) or 0.0,
                'tld_rank': int(r.get('tld_rank') or 0),
                'watcher_count': int(r.get('watcher_count') or 0),
                'unique_bidder_count': int(r.get('unique_bidder_count') or 0),
                'category_name': str(r.get('category_name') or ''),
            }

        mapped = [_to_auction_row(r) for r in enriched]
        written = await self._insert(_AUCTION_FEATURES_TABLE, _AUCTION_COLS, mapped)
        elapsed = (time.monotonic() - t0) * 1000.0
        logger.info(f"enriched_auction_features_built rows_written={written} elapsed_ms={elapsed:.1f}")
        return BuildResult(table=_AUCTION_FEATURES_TABLE, rows_written=written, elapsed_ms=elapsed, errors=errors)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    async def build_all(self, lookback_weeks: int = 13, auction_limit: int = 50000) -> Dict[str, BuildResult]:
        """Run all enrichment builds concurrently."""
        tld_result, auction_result = await asyncio.gather(
            self.build_tld_features(lookback_weeks=lookback_weeks),
            self.build_auction_features(limit=auction_limit),
            return_exceptions=False,
        )
        return {
            'enriched_tld_features': tld_result,
            'enriched_auction_features': auction_result,
        }

    async def loop_forever(self, interval_hours: float = 6.0) -> None:
        """Periodic build loop — runs until cancelled."""
        interval_s = float(interval_hours) * 3600.0
        logger.info(f"enriched_tables_builder_loop_start interval_hours={interval_hours:.1f}")
        while True:
            try:
                results = await self.build_all()
                for table, res in results.items():
                    logger.info(
                        f"enriched_tables_build_cycle table={table} "
                        f"rows_written={res.rows_written} elapsed_ms={res.elapsed_ms:.1f} "
                        f"errors={res.errors}"
                    )
            except asyncio.CancelledError:
                logger.info("enriched_tables_builder_loop_cancelled")
                return
            except Exception as e:
                logger.warning(f"enriched_tables_build_cycle_error error_type={type(e).__name__} error={e}")
            await asyncio.sleep(interval_s)


__all__ = ['EnrichedTablesBuilder', 'BuildResult']
