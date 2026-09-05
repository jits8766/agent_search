"""Composable in-memory enrichment pipeline for post-query row processing.

Enrichment steps are pure functions that transform a list of row dicts returned
by ClickHouseExecutor.execute(), adding computed columns without any additional
database round-trips.  Steps are composed into an ``EnrichmentPipeline`` via the
``.then()`` builder:

    pipeline = (
        EnrichmentPipeline()
        .then(enrich_growth_rates)
        .then(enrich_anomaly_flags)
        .then(enrich_percentile_ranks)
    )
    enriched_rows = pipeline.run(rows, ctx)

Each step receives the full row list and an ``EnrichmentContext`` carrying
shared metadata (entity column name, time column name, etc.).  Steps must never
mutate their input list — they should return a fresh list.

Design constraints:
  - All steps are O(N) in the number of rows unless documented otherwise.
  - Steps raise ValueError on bad configuration, not silently drop rows.
  - Steps that require a minimum number of rows to produce meaningful output
    return the input unchanged when the row count is insufficient.
"""
from __future__ import annotations

import datetime
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = [
    'EnrichmentContext',
    'EnrichmentResult',
    'EnrichmentPipeline',
    'enrich_growth_rates',
    'enrich_anomaly_flags',
    'enrich_percentile_ranks',
    'enrich_forecast_values',
    'enrich_lifecycle_stage',
    'enrich_heat_score',
    'enrich_watcher_count',
    'enrich_unique_bidders',
    'enrich_fair_value_band',
    'enrich_tld_momentum',
    'enrich_sell_through_prob',
    'enrich_time_urgency',
]

# An enrichment step signature
EnrichmentStep = Callable[['list[dict]', 'EnrichmentContext'], 'list[dict]']


@dataclass
class EnrichmentContext:
    """Shared metadata passed to every enrichment step in a pipeline run.

    :param request_id: Correlation id for structured logging
    :param entity_column: Name of the primary entity dimension column in the rows
    :param time_column: Name of the time/period column (string or datetime value)
    :param metric_column: Primary numeric metric column (optional — steps may accept
        an explicit override via their own keyword arguments)
    :param metadata: Arbitrary caller-supplied key/value pairs (e.g. grain, table)
    """
    request_id: str
    entity_column: str = 'entity'
    time_column: str = 'period'
    metric_column: str = 'metric_value'
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnrichmentResult:
    """Outcome of a full ``EnrichmentPipeline.run()`` call.

    :param rows: Final enriched row list
    :param steps_applied: Number of steps successfully applied
    :param skipped_steps: Steps that were no-ops (e.g. empty input or insufficient data)
    :param errors: Step names that raised and were caught (row list unchanged for that step)
    """
    rows: List[Dict[str, Any]]
    steps_applied: int
    skipped_steps: int
    errors: List[str]


class EnrichmentPipeline:
    """Immutable composable pipeline of enrichment steps.

    Usage::

        pipeline = EnrichmentPipeline()                        # empty
        pipeline = pipeline.then(enrich_growth_rates)          # grow
        result = pipeline.run(rows, ctx)                       # execute

    ``run`` returns an ``EnrichmentResult`` rather than a bare list so callers
    can distinguish a genuine empty result from a pipeline that was never applied.
    Errors in individual steps are caught, logged, and counted — they never
    propagate out of ``run``.
    """

    def __init__(self, steps: Optional[List[EnrichmentStep]] = None) -> None:
        self._steps: List[EnrichmentStep] = list(steps or [])

    def then(self, step: EnrichmentStep) -> 'EnrichmentPipeline':
        """Return a NEW pipeline with ``step`` appended (original is unchanged)."""
        if not callable(step):
            raise ValueError(f'EnrichmentPipeline.then requires a callable, got {type(step).__name__}')
        return EnrichmentPipeline(self._steps + [step])

    def run(self, rows: List[Dict[str, Any]], ctx: EnrichmentContext) -> EnrichmentResult:
        """Apply all steps sequentially.  Errors are caught and counted."""
        current = list(rows)
        applied = 0
        skipped = 0
        errors: List[str] = []
        for step in self._steps:
            step_name = getattr(step, '__name__', repr(step))
            if not current:
                skipped += 1
                continue
            try:
                result = step(current, ctx)
                if result is current:
                    skipped += 1
                else:
                    current = list(result)
                    applied += 1
            except (ValueError, TypeError) as exc:
                logger.warning(f"enrichment_step_error request_id={ctx.request_id} step={step_name} error_type={type(exc).__name__} error={exc}")
                errors.append(step_name)
        return EnrichmentResult(rows=current, steps_applied=applied, skipped_steps=skipped, errors=errors)


# ---------------------------------------------------------------------------
# Built-in enrichment steps
# ---------------------------------------------------------------------------

def enrich_growth_rates(rows: List[Dict[str, Any]], ctx: EnrichmentContext, value_col: str = 'metric_value', output_col: str = 'growth_pct', entity_col: Optional[str] = None, period_col: Optional[str] = None) -> List[Dict[str, Any]]:
    """Add ``growth_pct`` computed from consecutive rows per entity.

    Rows must be sorted by ``(entity, period)`` ascending for the lag to be
    meaningful.  Returns the input list unchanged when fewer than 2 rows are
    present.

    :param value_col: Column carrying the numeric metric to diff
    :param output_col: Column name for the computed growth percentage
    :param entity_col: Entity dimension column (defaults to ``ctx.entity_column``)
    :param period_col: Time period column (defaults to ``ctx.time_column``)
    """
    if len(rows) < 2:
        return rows
    ent_col = entity_col or ctx.entity_column
    per_col = period_col or ctx.time_column
    result: List[Dict[str, Any]] = []
    prev_by_entity: Dict[Any, float] = {}
    for row in rows:
        new_row = dict(row)
        entity_key = row.get(ent_col)
        current_val = row.get(value_col)
        if current_val is None:
            new_row[output_col] = None
        else:
            try:
                current_f = float(current_val)
            except (TypeError, ValueError):
                new_row[output_col] = None
                result.append(new_row)
                prev_by_entity[entity_key] = current_val
                continue
            prev_f = prev_by_entity.get(entity_key)
            if prev_f is None or prev_f == 0.0:
                new_row[output_col] = None
            else:
                new_row[output_col] = round((current_f - prev_f) / prev_f * 100.0, 4)
            prev_by_entity[entity_key] = current_f
        result.append(new_row)
    return result


def enrich_anomaly_flags(rows: List[Dict[str, Any]], ctx: EnrichmentContext, z_col: str = 'z_score', threshold: float = 3.0, output_col: str = 'is_anomaly') -> List[Dict[str, Any]]:
    """Add ``is_anomaly`` boolean flag based on an existing z_score column.

    Rows that already carry a ``z_score`` from the DB (e.g. from
    ``anomaly_detection_sql``) are enriched in-memory to avoid a second
    templated query.

    :param z_col: Column carrying the pre-computed Z-score
    :param threshold: Absolute Z-score threshold for anomaly classification
    :param output_col: Output boolean column name
    """
    if float(threshold) <= 0:
        raise ValueError('enrich_anomaly_flags: threshold must be > 0')
    result: List[Dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        z = row.get(z_col)
        if z is None:
            new_row[output_col] = None
        else:
            try:
                new_row[output_col] = abs(float(z)) > threshold
            except (TypeError, ValueError):
                new_row[output_col] = None
        result.append(new_row)
    return result


def enrich_percentile_ranks(rows: List[Dict[str, Any]], ctx: EnrichmentContext, sort_col: str = 'composite_score', output_col: str = 'rank', partition_col: Optional[str] = None) -> List[Dict[str, Any]]:
    """Add a dense rank column within an optional partition.

    Rows with equal ``sort_col`` values receive the same rank.  The result list
    is returned in the original order; only the rank column is added.

    :param sort_col: Column to rank by (descending — highest value = rank 1)
    :param output_col: Output rank column name
    :param partition_col: If set, rank resets within each value of this column
    """
    if not rows:
        return rows

    def _rank_within(subset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        indexed = list(enumerate(subset))
        safe_key = lambda r: (r[1].get(sort_col) is None, -(float(r[1].get(sort_col) or 0)))
        indexed.sort(key=safe_key)
        ranked: Dict[int, int] = {}
        current_rank = 1
        prev_val = object()
        for pos, (orig_idx, row) in enumerate(indexed):
            val = row.get(sort_col)
            if val != prev_val:
                current_rank = pos + 1
                prev_val = val
            ranked[orig_idx] = current_rank
        return [dict(row, **{output_col: ranked[i]}) for i, row in enumerate(subset)]

    if partition_col is None:
        return _rank_within(rows)

    partitions: Dict[Any, List[tuple]] = {}
    for i, row in enumerate(rows):
        key = row.get(partition_col)
        partitions.setdefault(key, []).append((i, row))

    result_map: Dict[int, Dict[str, Any]] = {}
    for key, indexed_rows in partitions.items():
        original_indices = [idx for idx, _ in indexed_rows]
        subset = [row for _, row in indexed_rows]
        ranked_subset = _rank_within(subset)
        for orig_idx, ranked_row in zip(original_indices, ranked_subset):
            result_map[orig_idx] = ranked_row

    return [result_map[i] for i in range(len(rows))]


def enrich_forecast_values(rows: List[Dict[str, Any]], ctx: EnrichmentContext, period_col: str = 'period', value_col: str = 'metric_value', horizon: int = 1, output_col: str = 'forecast_value', entity_col: Optional[str] = None) -> List[Dict[str, Any]]:
    """Add a linear-extrapolation forecast column per entity.

    Fits a simple linear model to the rows for each entity (using their numeric
    period index within the entity's row sequence) and appends a
    ``forecast_value`` column holding the extrapolated value ``horizon`` steps
    beyond the last observed row.

    Requires at least 3 rows per entity for a meaningful fit; entities with
    fewer rows receive ``None``.

    :param period_col: Column carrying the time bucket (used as row order, not value)
    :param value_col: Numeric metric to forecast
    :param horizon: Steps ahead to project (>= 1)
    :param output_col: Output column name for the forecast
    :param entity_col: Entity dimension (defaults to ``ctx.entity_column``)
    """
    if int(horizon) < 1:
        raise ValueError('enrich_forecast_values: horizon must be >= 1')
    ent_col = entity_col or ctx.entity_column

    # Group rows by entity preserving order
    entity_indices: Dict[Any, List[int]] = {}
    for i, row in enumerate(rows):
        entity_indices.setdefault(row.get(ent_col), []).append(i)

    forecast_by_index: Dict[int, Optional[float]] = {}
    for entity_key, indices in entity_indices.items():
        vals: List[Optional[float]] = []
        for idx in indices:
            v = rows[idx].get(value_col)
            try:
                vals.append(float(v) if v is not None else None)
            except (TypeError, ValueError):
                vals.append(None)

        clean_vals = [(i, v) for i, v in enumerate(vals) if v is not None]
        if len(clean_vals) < 3:
            for idx in indices:
                forecast_by_index[idx] = None
            continue

        xs = [float(i) for i, _ in clean_vals]
        ys = [v for _, v in clean_vals]
        try:
            slope, intercept = _linear_regression(xs, ys)
            next_x = xs[-1] + float(horizon)
            predicted = round(slope * next_x + intercept, 4)
        except (ValueError, TypeError, ZeroDivisionError):
            predicted = None

        for idx in indices:
            forecast_by_index[idx] = predicted

    result: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        new_row = dict(row)
        new_row[output_col] = forecast_by_index.get(i)
        result.append(new_row)
    return result


def enrich_lifecycle_stage(rows: List[Dict[str, Any]], ctx: EnrichmentContext, age_col: str = 'age_days', activity_col: str = 'days_since_active', output_col: str = 'lifecycle_stage', new_threshold: int = 7, active_threshold: int = 7, dormant_threshold: int = 30) -> List[Dict[str, Any]]:
    """Add a lifecycle stage label based on entity age and recency of activity.

    Classification rules (evaluated in order):
      1. ``age_days <= new_threshold``             → ``'new'``
      2. ``days_since_active <= active_threshold`` → ``'active'``
      3. ``days_since_active <= dormant_threshold``→ ``'dormant'``
      4. ``days_since_active <= dormant_threshold * 2`` → ``'at_risk'``
      5. otherwise                                 → ``'churned'``

    When either column value is missing the stage is ``'unknown'``.

    :param age_col: Column carrying entity age in days
    :param activity_col: Column carrying days since last activity
    :param output_col: Output stage label column
    :param new_threshold: Max age (days) to classify as 'new'
    :param active_threshold: Max inactivity (days) to classify as 'active'
    :param dormant_threshold: Max inactivity (days) to classify as 'dormant'
    """
    result: List[Dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        age = row.get(age_col)
        inactive = row.get(activity_col)
        if age is None or inactive is None:
            new_row[output_col] = 'unknown'
        else:
            try:
                age_f = float(age)
                inactive_f = float(inactive)
            except (TypeError, ValueError):
                new_row[output_col] = 'unknown'
                result.append(new_row)
                continue
            if age_f <= new_threshold:
                stage = 'new'
            elif inactive_f <= active_threshold:
                stage = 'active'
            elif inactive_f <= dormant_threshold:
                stage = 'dormant'
            elif inactive_f <= dormant_threshold * 2:
                stage = 'at_risk'
            else:
                stage = 'churned'
            new_row[output_col] = stage
        result.append(new_row)
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _linear_regression(xs: List[float], ys: List[float]) -> tuple:
    """Return (slope, intercept) for the OLS line through (xs, ys).

    Uses stdlib ``statistics`` to avoid a numpy dependency.  Raises
    ``statistics.StatisticsError`` when variance is zero (all x equal).
    """
    n = len(xs)
    if n < 2:
        raise statistics.StatisticsError('Need at least 2 data points')
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    ss_xx = sum((x - mean_x) ** 2 for x in xs)
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    if ss_xx == 0:
        raise statistics.StatisticsError('Zero variance in x — cannot fit regression')
    slope = ss_xy / ss_xx
    intercept = mean_y - slope * mean_x
    return slope, intercept


def enrich_heat_score(rows: List[Dict[str, Any]], ctx: EnrichmentContext, velocity_delta_col: str = 'velocity_delta', output_col: str = 'heat_score', high_threshold: float = 5.0, medium_threshold: float = 1.0) -> List[Dict[str, Any]]:
    """Classify bid acceleration into heat tiers: 'high', 'medium', 'low' based on velocity_delta.

    :param velocity_delta_col: Column carrying recent_bids minus prior_bids
    :param output_col: Column name for the tier label
    :param high_threshold: velocity_delta >= this → 'high'
    :param medium_threshold: velocity_delta >= this (but < high_threshold) → 'medium'
    """
    if not rows:
        return rows
    result: List[Dict[str, Any]] = []
    for row in rows:
        delta = row.get(velocity_delta_col)
        if delta is None:
            tier = 'unknown'
        elif float(delta) >= float(high_threshold):
            tier = 'high'
        elif float(delta) >= float(medium_threshold):
            tier = 'medium'
        else:
            tier = 'low'
        result.append({**row, output_col: tier})
    logger.info(f"enrich_heat_score request_id={ctx.request_id} rows={len(result)} high_threshold={high_threshold}")
    return result


def enrich_watcher_count(rows: List[Dict[str, Any]], ctx: EnrichmentContext, item_id_col: str = 'member_item_id', watcher_data: Optional[Dict[int, int]] = None, output_col: str = 'watcher_count') -> List[Dict[str, Any]]:
    """Annotate rows with a watcher count looked up from a pre-fetched dict.

    :param watcher_data: Mapping of item_id → watcher_count; rows without a match get 0
    :param output_col: Column name for the annotated count
    """
    if not rows or watcher_data is None:
        return rows
    result: List[Dict[str, Any]] = []
    for row in rows:
        item_id = row.get(item_id_col)
        count = watcher_data.get(int(item_id), 0) if item_id is not None else 0
        result.append({**row, output_col: count})
    logger.info(f"enrich_watcher_count request_id={ctx.request_id} rows={len(result)} lookup_size={len(watcher_data)}")
    return result


def enrich_unique_bidders(rows: List[Dict[str, Any]], ctx: EnrichmentContext, auction_id_col: str = 'auction_id', bidder_data: Optional[Dict[int, int]] = None, output_col: str = 'unique_bidder_count') -> List[Dict[str, Any]]:
    """Annotate rows with a unique bidder count looked up from a pre-fetched dict.

    :param bidder_data: Mapping of auction_id → unique_bidder_count; unmatched rows get 0
    :param output_col: Column name for the annotated count
    """
    if not rows or bidder_data is None:
        return rows
    result: List[Dict[str, Any]] = []
    for row in rows:
        auction_id = row.get(auction_id_col)
        count = bidder_data.get(int(auction_id), 0) if auction_id is not None else 0
        result.append({**row, output_col: count})
    logger.info(f"enrich_unique_bidders request_id={ctx.request_id} rows={len(result)} lookup_size={len(bidder_data)}")
    return result


def enrich_fair_value_band(rows: List[Dict[str, Any]], ctx: EnrichmentContext, tld_col: str = 'tld', price_col: str = 'current_price', comparable_data: Optional[Dict[str, Dict[str, float]]] = None, p25_output: str = 'fair_value_p25', p75_output: str = 'fair_value_p75') -> List[Dict[str, Any]]:
    """Annotate rows with p25/p75 fair value band from a pre-fetched TLD comparable price dict.

    :param comparable_data: Mapping of tld → {'p25': float, 'p75': float}; unmatched rows get None
    :param p25_output: Column name for the p25 lower bound
    :param p75_output: Column name for the p75 upper bound
    """
    if not rows or comparable_data is None:
        return rows
    result: List[Dict[str, Any]] = []
    for row in rows:
        tld = row.get(tld_col, '')
        band = comparable_data.get(str(tld), {})
        result.append({**row, p25_output: band.get('p25'), p75_output: band.get('p75')})
    logger.info(f"enrich_fair_value_band request_id={ctx.request_id} rows={len(result)} tld_coverage={len(comparable_data)}")
    return result


def enrich_tld_momentum(rows: List[Dict[str, Any]], ctx: EnrichmentContext, tld_col: str = 'tld', momentum_data: Optional[Dict[str, float]] = None, output_col: str = 'tld_momentum_pct', positive_threshold: float = 10.0, negative_threshold: float = -10.0) -> List[Dict[str, Any]]:
    """Annotate rows with TLD WoW sales momentum (percent change) and a direction label.

    :param momentum_data: Mapping of tld → WoW percent change (e.g. {'com': 15.2, 'net': -5.1})
    :param positive_threshold: WoW % above this → momentum_label = 'rising'
    :param negative_threshold: WoW % below this → momentum_label = 'falling'
    """
    if not rows or momentum_data is None:
        return rows
    result: List[Dict[str, Any]] = []
    for row in rows:
        tld = row.get(tld_col, '')
        pct = momentum_data.get(str(tld))
        if pct is None:
            label = 'unknown'
        elif float(pct) >= float(positive_threshold):
            label = 'rising'
        elif float(pct) <= float(negative_threshold):
            label = 'falling'
        else:
            label = 'stable'
        result.append({**row, output_col: pct, 'tld_momentum_label': label})
    logger.info(f"enrich_tld_momentum request_id={ctx.request_id} rows={len(result)}")
    return result


def enrich_sell_through_prob(rows: List[Dict[str, Any]], ctx: EnrichmentContext, tld_col: str = 'tld', sell_through_data: Optional[Dict[str, float]] = None, output_col: str = 'sell_through_prob', high_threshold: float = 0.6, low_threshold: float = 0.3) -> List[Dict[str, Any]]:
    """Annotate rows with historical sell-through probability from a pre-fetched TLD lookup.

    :param sell_through_data: Mapping of tld → sell-through rate (0.0–1.0)
    :param high_threshold: Rate >= this → sell_through_tier = 'high'
    :param low_threshold: Rate < this → sell_through_tier = 'low'
    """
    if not rows or sell_through_data is None:
        return rows
    result: List[Dict[str, Any]] = []
    for row in rows:
        tld = row.get(tld_col, '')
        rate = sell_through_data.get(str(tld))
        if rate is None:
            tier = 'unknown'
        elif float(rate) >= float(high_threshold):
            tier = 'high'
        elif float(rate) < float(low_threshold):
            tier = 'low'
        else:
            tier = 'medium'
        result.append({**row, output_col: rate, 'sell_through_tier': tier})
    logger.info(f"enrich_sell_through_prob request_id={ctx.request_id} rows={len(result)}")
    return result


def enrich_time_urgency(rows: List[Dict[str, Any]], ctx: EnrichmentContext, ends_at_col: str = 'ends_at', output_col: str = 'time_urgency_label', critical_hours: float = 1.0, high_hours: float = 6.0, medium_hours: float = 24.0) -> List[Dict[str, Any]]:
    """Annotate rows with a time urgency label based on hours remaining until auction close.

    Parses ends_at as ISO-8601 or unix timestamp float. Rows with unparseable ends_at get 'unknown'.

    :param critical_hours: Hours remaining < this → 'critical'
    :param high_hours: Hours remaining < this → 'high'
    :param medium_hours: Hours remaining < this → 'medium'; else 'low'
    """
    if not rows:
        return rows
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    result: List[Dict[str, Any]] = []
    for row in rows:
        raw = row.get(ends_at_col)
        label = 'unknown'
        if raw is not None:
            try:
                if isinstance(raw, (int, float)):
                    ends = datetime.datetime.utcfromtimestamp(float(raw))
                elif isinstance(raw, datetime.datetime):
                    ends = raw.replace(tzinfo=None) if raw.tzinfo else raw
                else:
                    ends = datetime.datetime.fromisoformat(str(raw).replace('Z', '+00:00')).replace(tzinfo=None)
                hours_left = (ends - now).total_seconds() / 3600.0
                if hours_left < float(critical_hours):
                    label = 'critical'
                elif hours_left < float(high_hours):
                    label = 'high'
                elif hours_left < float(medium_hours):
                    label = 'medium'
                else:
                    label = 'low'
            except (ValueError, TypeError, KeyError):
                label = 'unknown'
        result.append({**row, output_col: label})
    logger.info(f"enrich_time_urgency request_id={ctx.request_id} rows={len(result)}")
    return result
