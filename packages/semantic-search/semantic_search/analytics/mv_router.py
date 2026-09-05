"""MV-aware SQL router (stage 5a).

Sits between SQL validation and execution on the analytics path. Walks the
post-validation AST with `sqlglot` and asks each catalog MV "can I serve this
query?". If exactly one MV can, the SQL is rewritten to read from the MV's
pre-aggregated columns (`avgMerge(state)`, `quantilesMerge(state)`) instead of
scanning the raw events table — typically 5-50× faster.

Match contract (an MV can serve the query iff ALL hold):

1. The query reads exactly one table and that table equals `mv.source_table`.
2. Every GROUP BY column appears in `mv.grain_columns` (subset = OK; the MV may
   carry more grain columns than the query needs).
3. Every projected aggregate is `(agg_fn(col)|count())` where:
   - `count()` always matches (every AggregatingMergeTree carries `count`)
   - `agg_fn(col)` matches an MV `aggregate_columns` entry whose value equals
     the source expression (e.g. `avg(price)` matches `{"avg_price_state":
     "price"}` when the query agg fn is `avg`).
4. Every WHERE-clause column is a grain column (filtering on non-grain columns
   would require scanning raw rows).

When zero MVs match the router returns the SQL untouched (graceful pass-through
to raw-table execution). When more than one MV matches we pick the MV with the
smallest grain set (highest aggregation level) to minimize bytes scanned —
tied MVs fall back to alphabetical name order for determinism.

The rewrite NEVER drops a WHERE/HAVING/ORDER BY clause; it only swaps the
FROM table and rewrites the projection aggregate functions to their `*Merge`
counterparts (`avg(price)` → `avgMerge(avg_price_state)`).

Defence in depth: the rewriter is purely optional — the post-rewrite SQL goes
back through `AstSecurityValidator` so a bug here cannot bypass the security
gate. The router itself only RAISES on construction-time misconfiguration.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import sqlglot
from sqlglot import exp

from semantic_search.config.analytics_models import MaterializedViewConfig, MVRouterConfig
from semantic_search.core.exceptions import RetrievalError
from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)


_AGG_FN_TO_MERGE = {
    'avg': 'avgMerge',
    'sum': 'sumMerge',
    'min': 'minMerge',
    'max': 'maxMerge',
    'count': 'countMerge',
    'uniq': 'uniqMerge',
    'quantile': 'quantileMerge',
    'quantiles': 'quantilesMerge',
}

# Maps state-column name prefix → the aggregate function that produced it.
# Used by _can_serve to build a collision-free (fn, source_expr) reverse map
# so that multiple state columns with the same source (e.g. avg_price_state,
# median_price_state both derived from current_price) resolve to the correct
# Merge counterpart without dict last-write-wins stomping earlier entries.
_STATE_COL_PREFIX_TO_FN: Dict[str, str] = {
    'avg_': 'avg',
    'sum_': 'sum',
    'min_': 'min',
    'max_': 'max',
    'count': 'count',
    'uniq_': 'uniq',
    'quantiles_': 'quantiles',
    'quantile_': 'quantile',
    'median_': 'quantile',
}


@dataclass
class MVRewriteDecision:
    """Outcome of one router pass.

    :param matched: bool - True iff an MV was selected and the SQL rewritten
    :param mv_name: str - Selected MV name (empty when matched=False)
    :param rewritten_sql: str - The SQL the executor should run (== input SQL when matched=False)
    :param reason: str - Human-readable reason for matched=False (empty when matched=True)
    :param freshness_lag_seconds: float - Selected MV's worst-case lag (0.0 when matched=False)
    """
    matched: bool
    mv_name: str
    rewritten_sql: str
    reason: str
    freshness_lag_seconds: float


class MVRouter:
    """Catalog-driven MV rewriter.

    :param config: MVRouterConfig - Catalog of MVs the router may rewrite into
    :param dialect: str - sqlglot dialect for parsing (default 'clickhouse')
    """

    def __init__(self, config: MVRouterConfig, dialect: str = 'clickhouse'):
        if not isinstance(config, MVRouterConfig):
            raise RetrievalError("MVRouter requires a typed MVRouterConfig")
        if not isinstance(dialect, str) or not dialect:
            raise RetrievalError("MVRouter requires a non-empty dialect")
        self._config = config
        self._dialect = dialect
        self._invalid_mvs: Set[str] = set()

    def mark_invalid(self, mv_name: str) -> None:
        """Permanently skip mv_name on future route() calls after a CH execution failure."""
        if mv_name:
            self._invalid_mvs.add(mv_name)
            logger.warning(f"mv_router_invalidated mv={mv_name}")

    @property
    def catalog_size(self) -> int:
        """Number of MVs in the rewrite catalog."""
        return len(self._config.materialized_views)

    @property
    def enabled(self) -> bool:
        """True iff config-enabled AND catalog non-empty."""
        return bool(self._config.enabled) and self.catalog_size > 0

    def route(self, sql: str) -> MVRewriteDecision:
        """Try to rewrite `sql` to read from an MV; return a typed decision.

        :param sql: str - Validated SQL (post-security, post-logic)
        :return: MVRewriteDecision - Always typed; pass-through on no match
        """
        if not isinstance(sql, str) or not sql.strip():
            raise RetrievalError("MVRouter.route requires a non-empty SQL string")
        if not self.enabled:
            return MVRewriteDecision(matched=False, mv_name='', rewritten_sql=sql, reason='router_disabled_or_empty_catalog', freshness_lag_seconds=0.0)

        try:
            tree = sqlglot.parse_one(sql, dialect=self._dialect)
        except Exception as e:
            return MVRewriteDecision(matched=False, mv_name='', rewritten_sql=sql, reason=f'parse_failed:{type(e).__name__}', freshness_lag_seconds=0.0)
        if not isinstance(tree, exp.Select):
            return MVRewriteDecision(matched=False, mv_name='', rewritten_sql=sql, reason='not_a_select', freshness_lag_seconds=0.0)
        analysis = self._analyze(tree)
        if analysis is None:
            return MVRewriteDecision(matched=False, mv_name='', rewritten_sql=sql, reason='unsupported_query_shape', freshness_lag_seconds=0.0)
        source_table, group_cols, where_cols, projections = analysis

        candidates: List[Tuple[MaterializedViewConfig, Dict[str, Tuple[str, str]]]] = []
        for mv in self._config.materialized_views:
            if mv.name in self._invalid_mvs:
                continue
            mapping = self._can_serve(mv, source_table, group_cols, where_cols, projections)
            if mapping is not None:
                candidates.append((mv, mapping))

        if not candidates:
            return MVRewriteDecision(matched=False, mv_name='', rewritten_sql=sql, reason='no_mv_candidate', freshness_lag_seconds=0.0)

        candidates.sort(key=lambda item: (len(item[0].grain_columns), item[0].name))
        chosen_mv, projection_mapping = candidates[0]

        rewritten = self._rewrite(tree, chosen_mv, projection_mapping)
        rewritten_sql = rewritten.sql(dialect=self._dialect)
        logger.info(f"mv_router_match mv={chosen_mv.name} grain_cols={chosen_mv.grain_columns} freshness_lag_s={chosen_mv.freshness_lag_seconds:.1f}")
        return MVRewriteDecision(matched=True, mv_name=chosen_mv.name, rewritten_sql=rewritten_sql, reason='', freshness_lag_seconds=chosen_mv.freshness_lag_seconds)

    def _analyze(self, select: 'exp.Select') -> Optional[Tuple[str, Set[str], Set[str], List[Tuple[str, Optional[str], 'exp.Expression']]]]:  # type: ignore[name-defined]
        """Pull (source_table, group_cols, where_cols, projections) out of a SELECT.

        Returns None when the query shape isn't supported (multi-table, joins,
        subqueries in FROM, set ops, …). Projections is a list of
        `(agg_fn_lower, col_name_lower_or_None, original_alias_or_expr)` tuples;
        `agg_fn_lower==''` for non-aggregate projections (which we route to
        grain columns), and `col_name_lower=None` for `count(*)`.
        """
        # FROM must be a single physical table — joins / subqueries / lateral
        # views all pre-empt rewrite.
        froms = list(select.find_all(exp.From))
        if len(froms) != 1:
            return None
        from_node = froms[0]
        if list(select.find_all(exp.Join)):
            return None
        from_tables = [t for t in from_node.find_all(exp.Table) if t.parent is from_node]
        if len(from_tables) != 1:
            return None
        table = from_tables[0]
        if table.args.get('expressions'):
            return None
        source_table = table.name.lower()
        if not source_table:
            return None

        # GROUP BY columns
        group_cols: Set[str] = set()
        group_node = select.args.get('group')
        if group_node is not None:
            for g in group_node.expressions:
                col = self._as_column_name(g)
                if col is None:
                    return None
                group_cols.add(col.lower())

        # WHERE columns (every column reference under WHERE)
        where_cols: Set[str] = set()
        where_node = select.args.get('where')
        if where_node is not None:
            for col in where_node.find_all(exp.Column):
                where_cols.add(col.name.lower())

        # Projections — each must be either:
        #   (a) a column reference that is in group_cols (grain projection)
        #   (b) an aggregate function `agg(col)` or `count(*)`
        projections: List[Tuple[str, Optional[str], 'exp.Expression']] = []
        for proj in select.expressions:
            target = proj.unalias() if isinstance(proj, exp.Alias) else proj
            if isinstance(target, exp.Count) and self._is_count_star(target):
                projections.append(('count', None, proj))
                continue
            if isinstance(target, exp.AggFunc):
                fn = target.sql_name().lower()
                if fn not in _AGG_FN_TO_MERGE:
                    return None
                inner_args = target.args.get('this')
                col = self._as_column_name(inner_args)
                if col is None:
                    return None
                projections.append((fn, col.lower(), proj))
                continue
            if isinstance(target, exp.Column):
                col = target.name.lower()
                if col not in group_cols:
                    return None
                projections.append(('', col, proj))
                continue
            return None

        return source_table, group_cols, where_cols, projections

    @staticmethod
    def _as_column_name(node) -> Optional[str]:
        """Return the column name iff `node` is a bare Column; else None."""
        if node is None:
            return None
        if isinstance(node, exp.Column):
            return node.name
        return None

    @staticmethod
    def _is_count_star(count_node) -> bool:
        """True iff the Count node is `count(*)` (no specific column)."""
        inner = count_node.args.get('this')
        if inner is None:
            return True
        if isinstance(inner, exp.Star):
            return True
        return False

    @staticmethod
    def _can_serve(mv: MaterializedViewConfig, source_table: str, group_cols: Set[str], where_cols: Set[str], projections: List[Tuple[str, Optional[str], 'object']]) -> Optional[Dict[str, Tuple[str, str]]]:
        """Return a {projection_id: (merge_fn, state_column)} mapping iff `mv` can serve the query.

        `projection_id` is the `id(proj_expression)` so the rewriter can match
        the AST node by identity (not value) when swapping. Returns None when
        any contract bullet from the module docstring fails.
        """
        if mv.source_table.lower() != source_table:
            return None
        mv_grain_lower = {g.lower() for g in mv.grain_columns}
        fixed_lower = {c.lower() for c in mv.fixed_filter_columns}
        if not group_cols.issubset(mv_grain_lower):
            return None
        # WHERE cols must be served by grain OR be fixed-filter columns baked into the MV DDL.
        if not where_cols.issubset(mv_grain_lower | fixed_lower):
            return None
        # Reverse map: (inferred_fn, source_expr.lower()) -> state_column_name.
        # Keying on (fn, source_expr) avoids dict last-write-wins stomping when
        # multiple state columns share the same source expression (e.g.
        # avg_price_state and median_price_state both derived from current_price).
        # The function is inferred from the state column name prefix via
        # _STATE_COL_PREFIX_TO_FN so the config stays declarative.
        agg_reverse: Dict[Tuple[str, str], str] = {}
        count_state_col: Optional[str] = None
        for state_col, source_expr in mv.aggregate_columns.items():
            state_col_lower = state_col.lower()
            source_lower = source_expr.lower()
            for prefix, fn in _STATE_COL_PREFIX_TO_FN.items():
                if state_col_lower.startswith(prefix):
                    agg_reverse[(fn, source_lower)] = state_col
                    break
            if source_lower in ('*', 'count(*)'):
                count_state_col = state_col
        mapping: Dict[str, Tuple[str, str]] = {}
        for fn, col, proj in projections:
            proj_id = str(id(proj))
            if fn == '':
                # Grain-column passthrough — must be a grain col on the MV.
                if col is None or col not in mv_grain_lower:
                    return None
                continue
            if fn == 'count' and col is None:
                # `count(*)` is universally available on AggregatingMergeTree
                # MVs as `countMerge(count_state)` if the MV defines one.
                state_col = count_state_col
                if state_col is None:
                    return None
                mapping[proj_id] = ('countMerge', state_col)
                continue
            if col is None:
                return None
            state_col = agg_reverse.get((fn, col))
            if state_col is None:
                return None
            merge_fn = _AGG_FN_TO_MERGE.get(fn)
            if merge_fn is None:
                return None
            mapping[proj_id] = (merge_fn, state_col)
        return mapping

    @staticmethod
    def _strip_fixed_filters(where_node: 'exp.Where', fixed_cols: Set[str]) -> Optional['exp.Expression']:  # type: ignore[name-defined]
        """Remove AND branches whose column references are all in fixed_cols.

        Returns the pruned condition expression, or None if all branches were stripped.
        The caller should set WHERE to None when this returns None.
        """
        def _collect_ands(node: 'exp.Expression') -> List['exp.Expression']:  # type: ignore[name-defined]
            if isinstance(node, exp.And):
                return _collect_ands(node.left) + _collect_ands(node.right)
            return [node]

        branches = _collect_ands(where_node.this)
        remaining = []
        for branch in branches:
            cols = {c.name.lower() for c in branch.find_all(exp.Column)}
            if cols and cols.issubset(fixed_cols):
                continue
            remaining.append(branch)

        if not remaining:
            return None
        result = remaining[0]
        for branch in remaining[1:]:
            result = exp.And(this=result, expression=branch)
        return result

    def _rewrite(self, select: 'exp.Select', mv: MaterializedViewConfig, projection_mapping: Dict[str, Tuple[str, str]]) -> 'exp.Select':  # type: ignore[name-defined]
        """Apply the rewrite: swap FROM table and aggregate fns; return a new tree.

        We mutate a deep copy so the caller's tree is unchanged (the security
        validator re-parses the rewritten SQL string anyway, but keeping the
        contract pure makes the unit tests trivial).
        """
        new_tree = select.copy()

        # 1) Replace FROM table. Use find_all(exp.From) — newer sqlglot stores the
        #    FROM clause under the key 'from_' (not 'from'), so args.get('from') is None.
        from_nodes = list(new_tree.find_all(exp.From))
        if from_nodes:
            mv_table = exp.to_table(mv.name, dialect=self._dialect)
            from_nodes[0].set('this', mv_table)

        # 2) Strip fixed-filter WHERE branches — these columns are baked into the
        #    MV DDL and don't exist as physical columns in the MV table.
        if mv.fixed_filter_columns:
            fixed_lower = {c.lower() for c in mv.fixed_filter_columns}
            where_node = new_tree.args.get('where')
            if where_node is not None:
                new_cond = self._strip_fixed_filters(where_node, fixed_lower)
                if new_cond is None:
                    new_tree.set('where', None)
                else:
                    where_node.set('this', new_cond)

        # 3) Rewrite projection aggregates by AST identity. We re-walk the new
        #    tree and match by structural shape since `id()` doesn't survive a
        #    `.copy()`. Strategy: for every projection in the new tree, look it
        #    up by *position* against the original `select.expressions` list so
        #    the projection_mapping (keyed on the original `id`) can be applied.
        original_projs = select.expressions
        new_projs = new_tree.expressions
        if len(original_projs) != len(new_projs):
            return new_tree
        for orig, new in zip(original_projs, new_projs):
            mapping = projection_mapping.get(str(id(orig)))
            if mapping is None:
                continue
            merge_fn, state_col = mapping
            # Build merge-fn call preserving any alias the caller provided.
            replacement = exp.Anonymous(
                this=merge_fn,
                expressions=[exp.column(state_col)],
            )
            if isinstance(new, exp.Alias):
                new.set('this', replacement)
            else:
                new.replace(replacement)
        return new_tree


__all__ = ['MVRouter', 'MVRewriteDecision']
