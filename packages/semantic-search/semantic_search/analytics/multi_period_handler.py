"""Multi-period aggregation handler for dimensional analytics questions.

Fires 8 parallel ClickHouse queries (last_5m → last_30d) using only
pre-aggregated Materialized Views (AggregatingMergeTree). No raw scans.

Multi-dim GROUP BY:
  When sql_hint carries multiple dimension keys that are ALL present as grain
  columns in a composite MV, the handler groups by ALL dimensions simultaneously
  (e.g. ``tld + auction_type`` → ``GROUP BY tld, auction_type_id``).
  This gives a richer cross-breakdown without a second query.

  When no composite MV covers all requested dims for a given window, the
  handler falls back to the best single-dim + WHERE-filter approach for
  that window only. Windows with no MV at any level are omitted entirely.

Config-driven:
  - dimension → column aliases come from ``MultiPeriodConfig.hint_key_aliases``
  - timeout and row cap from ``MultiPeriodConfig``
  - all MV definitions and grain columns from ``MVRouterConfig.materialized_views``
  - no values hardcoded in this module

Returns ``None`` when no dimension is detected in sql_hint or all windows
return 0 rows (caller continues to the NL-to-SQL pipeline).
"""
import asyncio
import re
import time
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from semantic_search.analytics.clickhouse_executor import ClickHouseExecutor
from semantic_search.config.analytics_models import MaterializedViewConfig, MultiPeriodConfig, MVRouterConfig
from semantic_search.core.logging_utils import get_logger
from semantic_search.nl_to_sql.contracts import AnalyticsResult

logger = get_logger(__name__)

# Hint keys expressing a raw price range. AggregatingMergeTree MVs store price
# only as pre-computed aggregate states, so these cannot be applied as a
# row-level filter here — their presence makes the handler decline (defer to
# NL-to-SQL on the raw table).
_PRICE_RANGE_KEYS = frozenset({'price_max', 'price_min'})

# Reserved hint key carrying explicit GROUP BY grain columns (e.g.
# "group_by=auction_type_id,tld"). When present it overrides the legacy
# heuristic that infers dimensions from concrete filter keys — this is how a
# grouping dimension ("by auction type") is told apart from a value filter
# ("auction type = 16"), which a bare key=value hint cannot express.
_GROUP_BY_KEY = 'group_by'

# Ordered ascending (smallest window first) so the response reads chronologically.
# Each entry: (label, grain_preference_order)
_WINDOWS: List[Tuple[str, List[str]]] = [
    ('last_5m',  ['hour', 'day']),
    ('last_30m', ['hour', 'day']),
    ('last_1h',  ['hour', 'day']),
    ('last_5h',  ['hour', 'day']),
    ('last_24h', ['hour', 'day']),
    ('last_7d',  ['day', 'week']),
    ('last_14d', ['week', 'day']),
    ('last_15d', ['day', 'week']),
    ('last_30d', ['day', 'week']),
    ('last_90d', ['day', 'week']),
]

# (window_label, grain) → time predicate template; {tc} replaced with MV time_column.
_TIME_PRED: Dict[Tuple[str, str], str] = {
    ('last_5m',  'hour'): '{tc} >= now() - INTERVAL 5 MINUTE',
    ('last_30m', 'hour'): '{tc} >= now() - INTERVAL 30 MINUTE',
    ('last_1h',  'hour'): '{tc} >= now() - INTERVAL 1 HOUR',
    ('last_5h',  'hour'): '{tc} >= now() - INTERVAL 5 HOUR',
    ('last_24h', 'hour'): '{tc} >= now() - INTERVAL 24 HOUR',
    ('last_7d',  'day'):  '{tc} >= today() - 7',
    ('last_14d', 'day'):  '{tc} >= today() - 14',
    ('last_15d', 'day'):  '{tc} >= today() - 15',
    ('last_30d', 'day'):  '{tc} >= today() - 30',
    ('last_90d', 'day'):  '{tc} >= today() - 90',
    ('last_7d',  'week'): '{tc} >= toStartOfWeek(today() - 7)',
    ('last_14d', 'week'): '{tc} >= toStartOfWeek(today() - 14)',
    ('last_15d', 'week'): '{tc} >= toStartOfWeek(today() - 15)',
    ('last_30d', 'week'): '{tc} >= toStartOfWeek(today() - 30)',
    ('last_90d', 'week'): '{tc} >= toStartOfWeek(today() - 90)',
    ('last_5m',  'day'):  '{tc} >= today()',
    ('last_30m', 'day'):  '{tc} >= today()',
    ('last_1h',  'day'):  '{tc} >= today()',
    ('last_5h',  'day'):  '{tc} >= today()',
    ('last_24h', 'day'):  '{tc} >= today() - 1',
}

_GRAIN_FRAGMENTS: List[Tuple[str, str]] = [
    ('hour', 'hour'), ('day', 'day'), ('week', 'week'), ('month', 'month'),
]

_AGG_PREFIX_TO_FN: List[Tuple[str, str]] = [
    ('avg',    'avgMerge'),
    ('min',    'minMerge'),
    ('max',    'maxMerge'),
    ('median', 'medianMerge'),
    ('sum',    'sumMerge'),
    ('count',  'countMerge'),
    ('uniq',   'uniqMerge'),
]


def _merge_periods(dims: List[str], periods: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Pivot 8 separate period result arrays into one dimension-keyed merged table.

    Input:  {period_label: {rows: [{dim_col: val, metric: val, ...}], ...}}
    Output: [{dim_col: val, ..., last_5m: {metric: val, ...}, last_30m: {...}, ...}]

    Each output row represents one unique dimension-value combination across all
    time windows — frontend can directly render a multi-period breakdown table
    without joining anything.  Ordered by first dimension value ascending.
    """
    dim_set = set(dims)
    index: Dict[tuple, Dict[str, Any]] = {}

    for label, period_data in periods.items():
        for row in (period_data.get('rows') or []):
            dim_key = tuple(row.get(d) for d in dims)
            if dim_key not in index:
                index[dim_key] = {d: row[d] for d in dims if d in row}
            metrics = {k: v for k, v in row.items() if k not in dim_set}
            if metrics:
                index[dim_key][label] = metrics

    result = list(index.values())
    result.sort(key=lambda r: tuple(str(r.get(d, '')) for d in dims))
    return result


def _infer_grain(time_column: str) -> Optional[str]:
    tc = time_column.lower()
    for fragment, grain in _GRAIN_FRAGMENTS:
        if fragment in tc:
            return grain
    return None


def _parse_hint(sql_hint: str) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for match in re.finditer(r'(\w+)=([\w,.\-]+)', sql_hint or ''):
        key = match.group(1)
        vals = [v.strip() for v in match.group(2).split(',') if v.strip()]
        if vals:
            result[key] = vals
    return result


def _resolve_col(key: str, aliases: Dict[str, str]) -> str:
    return aliases.get(key, key)


def _build_merge_projection(
    mv: MaterializedViewConfig, group_by_cols: List[str]
) -> str:
    """SELECT projection: all GROUP BY cols first, then *Merge aggregates."""
    parts: List[str] = list(group_by_cols)
    for state_col in mv.aggregate_columns:
        lc = state_col.lower()
        fn = next((f for pfx, f in _AGG_PREFIX_TO_FN if lc.startswith(pfx)), None)
        if fn is None:
            continue
        alias = re.sub(r'_state$', '', state_col)
        if alias == state_col:
            alias = state_col + '_v'
        parts.append(f"{fn}({state_col}) AS {alias}")
    return ',\n       '.join(parts)


def _mv_query(mv: MaterializedViewConfig, group_by_cols: List[str], window_label: str, grain: str, extra_where: List[str], max_rows: int) -> str:
    proj = _build_merge_projection(mv, group_by_cols)
    time_pred = _TIME_PRED.get((window_label, grain), '').format(tc=mv.time_column)
    all_preds = ([time_pred] if time_pred else []) + extra_where
    where = ' AND '.join(all_preds) if all_preds else '1=1'
    group_clause = ', '.join(group_by_cols)
    return (
        f"SELECT {proj}\n"
        f"FROM {mv.name}\n"
        f"WHERE {where}\n"
        f"GROUP BY {group_clause}\n"
        f"ORDER BY {len(group_by_cols) + 1} DESC\n"
        f"LIMIT {max_rows}"
    )


class _DimIndex:
    """MV lookup index built from the catalog at construction time.

    Two sub-indexes:
      _single:    {dim_col: {grain: MV}}
      _composite: {frozenset(grain_cols): {grain: MV}}
    """

    def __init__(self, mv_configs: List[MaterializedViewConfig]) -> None:
        self._single: Dict[str, Dict[str, MaterializedViewConfig]] = {}
        self._composite: Dict[FrozenSet[str], Dict[str, MaterializedViewConfig]] = {}
        for mv in mv_configs:
            grain = _infer_grain(mv.time_column)
            if grain is None:
                continue
            key = frozenset(mv.grain_columns)
            # Single-grain index
            if len(mv.grain_columns) == 1:
                col = mv.grain_columns[0]
                self._single.setdefault(col, {})
                if grain not in self._single[col]:
                    self._single[col][grain] = mv
            # Composite index (includes single-grain too for superset search)
            self._composite.setdefault(key, {})
            if grain not in self._composite[key]:
                self._composite[key][grain] = mv

    def known_cols(self) -> Set[str]:
        """All grain columns present in any MV."""
        return set(self._single)

    def resolve_dims(
        self, hint_parsed: Dict[str, List[str]], aliases: Dict[str, str]
    ) -> List[str]:
        """Return all hint keys whose resolved column is a known grain column.

        Aliased keys (explicit in config) come first; plain hint keys secondary.
        Returns at least one entry or an empty list.
        """
        ordered = [k for k in aliases if k in hint_parsed] + [
            k for k in hint_parsed if k not in aliases
        ]
        dims: List[str] = []
        seen: Set[str] = set()
        for key in ordered:
            col = _resolve_col(key, aliases)
            if col in self._single and col not in seen:
                dims.append(col)
                seen.add(col)
        return dims

    def find_mv(self, group_by_cols: List[str], extra_filter_cols: Set[str], window_label: str, grain_prefs: List[str]) -> Optional[Tuple[MaterializedViewConfig, str]]:
        """Return (mv, grain) for the best MV, or None.

        Considers multi-dim GROUP BY: when group_by_cols has >1 entry the MV
        grain must cover ALL of them. Extra filter cols also need grain coverage
        (otherwise the WHERE clause is invalid on an AggregatingMergeTree MV).

        Priority:
          1. Exact-grain composite covering all group_by + filter cols
          2. Fallback-grain composite covering all group_by + filter cols
          3. (single-dim fallback) exact-grain for group_by_cols[0], filter dropped
          4. (single-dim fallback) fallback-grain for group_by_cols[0], filter dropped
        """
        needed: FrozenSet[str] = frozenset(group_by_cols) | extra_filter_cols
        needed_min: FrozenSet[str] = frozenset(group_by_cols)  # without filter cols

        composite_full = [
            (gset, g2mv) for gset, g2mv in self._composite.items()
            if needed <= gset
        ]
        composite_min = [
            (gset, g2mv) for gset, g2mv in self._composite.items()
            if needed_min <= gset and needed > gset
        ]

        def _pick(candidates, grain):
            for gset, g2mv in sorted(candidates, key=lambda x: len(x[0])):
                if grain in g2mv:
                    return g2mv[grain], grain
            return None

        # 1 + 2: full coverage (group_by + filters in grain)
        for grain in grain_prefs:
            result = _pick(composite_full, grain)
            if result:
                return result

        # 3 + 4: partial coverage — group_by cols in grain, filter cols dropped
        # (only safe when group_by has a single col; multi-dim still needs all cols)
        if len(group_by_cols) == 1:
            for grain in grain_prefs:
                result = _pick(composite_min, grain)
                if result:
                    logger.info(f"multi_period_filter_dropped dim={group_by_cols[0]} dropped_filter_cols={sorted(extra_filter_cols)} grain={grain}")
                    return result

        return None


class MultiPeriodHandler:
    """Fire 8 parallel ClickHouse MV queries for dimensional aggregate questions.

    Supports multi-dim GROUP BY: when all hint dimensions are covered by a
    composite MV grain, all dims are included in GROUP BY simultaneously.

    :param executor: ClickHouseExecutor — query transport
    :param mv_config: MVRouterConfig — MV catalog (sole source of grain metadata)
    :param config: MultiPeriodConfig — timeout / row cap / hint_key_aliases
    """

    def __init__(self, executor: ClickHouseExecutor, mv_config: MVRouterConfig, config: MultiPeriodConfig) -> None:
        self._executor = executor
        self._timeout = config.query_timeout_seconds
        self._max_rows = config.max_rows_per_period
        self._aliases = config.hint_key_aliases
        self._idx = _DimIndex(mv_config.materialized_views)

    def can_handle(self, sql_hint: str) -> bool:
        parsed = _parse_hint(sql_hint)
        # Pre-aggregated MVs hold price only as aggregate states, so a raw
        # price-range predicate (price_max / price_min) cannot be applied here.
        # Decline so the request falls through to the NL-to-SQL path (which
        # filters price on the raw table) instead of silently dropping the
        # filter and returning a count that ignores the price constraint.
        if any(k in _PRICE_RANGE_KEYS for k in parsed):
            return False
        return bool(self._resolve_group_dims(parsed))

    def _resolve_group_dims(self, parsed: Dict[str, List[str]]) -> List[str]:
        """Resolve the GROUP BY dimensions for a parsed hint.

        Prefers an explicit ``group_by=`` token (filter-vs-dimension is then
        unambiguous); falls back to the legacy heuristic that treats concrete
        grain-column keys as dimensions when no token is present.
        """
        explicit = [c for c in parsed.get(_GROUP_BY_KEY, []) if c in self._idx.known_cols()]
        if explicit:
            return explicit
        return self._idx.resolve_dims(parsed, self._aliases)

    def _build_period_queries(self, dims: List[str], hint_parsed: Dict[str, List[str]]) -> Dict[str, Tuple[str, str]]:
        """Return {period_label: (sql, mv_name)} for all servable windows."""
        # cols that appear in hint but are NOT part of the GROUP BY (value filters)
        hint_cols: Set[str] = {_resolve_col(k, self._aliases) for k in hint_parsed}
        extra_filter_cols: Set[str] = hint_cols - set(dims)

        # WHERE clauses for value-filter columns
        def _value_filters(group_by_cols: List[str]) -> List[str]:
            clauses: List[str] = []
            for key, vals in hint_parsed.items():
                col = _resolve_col(key, self._aliases)
                if col in group_by_cols:
                    continue
                quoted = ', '.join(f"'{v}'" for v in vals)
                clauses.append(f"{col} IN ({quoted})")
            return clauses

        queries: Dict[str, Tuple[str, str]] = {}
        for label, grain_prefs in _WINDOWS:
            match = self._idx.find_mv(dims, extra_filter_cols, label, grain_prefs)
            if match is None:
                logger.info(f"multi_period_no_mv label={label} dims={dims} filter_cols={sorted(extra_filter_cols)} — skipped")
                continue
            mv, grain = match

            # Effective GROUP BY: intersection of requested dims and MV grain.
            # When filter cols were dropped (partial match), use only group_by dims
            # that the MV grain actually covers.
            eff_group_by = [d for d in dims if d in frozenset(mv.grain_columns)]
            # Recalculate filter clauses: exclude any col NOT in the MV grain
            # (those were already dropped by find_mv's partial-match path).
            eff_filter_cols = [
                c for c in extra_filter_cols if c in frozenset(mv.grain_columns)
            ]
            value_where = _value_filters(eff_group_by)
            # Only keep value-filters for cols in the MV grain
            value_where = [
                clause for clause in value_where
                if any(
                    clause.startswith(f"{c} IN") for c in frozenset(mv.grain_columns)
                )
            ]

            sql = _mv_query(mv, eff_group_by, label, grain, value_where, self._max_rows)
            queries[label] = (sql, mv.name)

        return queries

    async def _run_one(self, label: str, sql: str, mv_name: str) -> Tuple[str, Dict[str, Any]]:
        t0 = time.monotonic()
        try:
            rows, _cols, latency_ms = await self._executor._client.execute_query(sql, timeout_seconds=self._timeout, _skip_mv_freshness_probe=True)
            return label, {
                'rows': rows,
                'latency_ms': round(latency_ms, 1),
                '_row_count': len(rows),
            }
        except Exception as e:
            latency_ms = (time.monotonic() - t0) * 1000.0
            logger.warning(f"multi_period_query_failed period={label} mv={mv_name} error_type={type(e).__name__} latency_ms={latency_ms:.1f}")
            return label, {
                'rows': [],
                'latency_ms': round(latency_ms, 1),
                'error': type(e).__name__,
                '_row_count': 0,
            }

    async def handle(self, question: str, sql_hint: str, request_id: str) -> Optional[AnalyticsResult]:
        """Run parallel MV queries across all 8 time windows.

        Multi-dim GROUP BY fires automatically when multiple hint dimensions
        map to a composite MV grain. Returns None on no match or all-empty.
        """
        hint_parsed = _parse_hint(sql_hint)
        dims = self._resolve_group_dims(hint_parsed)
        if not dims:
            return None

        # Drop the group_by control key before filter parsing so it is never
        # mistaken for a value filter; concrete grain keys NOT in `dims` then
        # become value filters (e.g. group_by=auction_type_id + tld=com →
        # GROUP BY auction_type_id WHERE tld IN ('com')).
        filter_hint = {k: v for k, v in hint_parsed.items() if k != _GROUP_BY_KEY}

        t0 = time.monotonic()
        period_queries = self._build_period_queries(dims, filter_hint)

        if not period_queries:
            logger.warning(f"multi_period_no_mv_any_window request_id={request_id} dims={dims} — falling through to NL-to-SQL")
            return None

        logger.info(f"multi_period_start request_id={request_id} dims={dims} windows={list(period_queries)} known_dims={sorted(self._idx.known_cols())}")

        # All 8 queries fire concurrently; asyncio.gather returns only after
        # the slowest completes — wall-clock = max(individual latencies), not sum.
        raw_results: List[Tuple[str, Dict[str, Any]]] = await asyncio.gather(*(
            self._run_one(label, sql, mv_name)
            for label, (sql, mv_name) in period_queries.items()
        ))

        # raw_results arrive in the same order as period_queries.items()
        # (asyncio.gather preserves input order).
        periods_raw: Dict[str, Any] = dict(raw_results)
        total_ms = (time.monotonic() - t0) * 1000.0

        # Drain internal counters before any early-return.
        total_rows = sum(p.pop('_row_count', 0) for p in periods_raw.values())

        logger.info(f"multi_period_done request_id={request_id} dims={dims} total_rows={total_rows} latency_ms={total_ms:.1f}")

        if not total_rows:
            logger.warning(f"multi_period_all_empty request_id={request_id} dims={dims}")
            return None

        # ── Post-gather merge ────────────────────────────────────────────────
        # Pivot all period result arrays by dimension key combo so the caller
        # receives one unified row per dimension value with all windows nested.
        # This runs synchronously — it is pure dict manipulation, O(total_rows).
        merged = _merge_periods(dims, periods_raw)

        # Build per-period latency map (strip raw rows — merged is canonical).
        period_latency: Dict[str, Any] = {
            label: {
                'latency_ms': p.get('latency_ms'),
                **({'error': p['error']} if 'error' in p else {}),
            }
            for label, p in periods_raw.items()
        }

        value_filters = {
            k: v for k, v in hint_parsed.items()
            if _resolve_col(k, self._aliases) not in set(dims)
        }

        return AnalyticsResult(
            request_id=request_id,
            question=question,
            sql_hint=sql_hint,
            success=True,
            failure_mode=None,
            failure_reason='',
            pruned_schema=None,
            generation=None,
            validation=None,
            execution=None,
            verifier=None,
            total_latency_ms=total_ms,
            analytics_substrate='clickhouse_hot',
            periods={
                'dimensions': dims,
                'value_filters': value_filters,
                'merged': merged,
                'period_latency': period_latency,
            },
        )
