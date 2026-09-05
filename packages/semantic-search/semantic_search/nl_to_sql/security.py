"""AST-based SQL security validator (stage 3).

Parses the generated SQL with `sqlglot` (Trino dialect by default) and walks
the AST to enforce:

1. SELECT-only: reject any DML/DDL/utility statement (INSERT, UPDATE, DELETE,
   MERGE, CREATE, DROP, ALTER, GRANT, TRUNCATE, CALL, USE, SET, …).
2. Single-statement: reject multi-statement payloads (SQL injection vector).
3. Table allowlist: every referenced table name must be in the configured
   allowlist (so a malicious model can never join a forbidden table).
4. PII column block: every projected/filtered column is checked against the
   configured PII column list (case-insensitive) and the query is rejected if
   any PII column is touched.
5. Bounded result: the query MUST contain a WHERE clause OR a LIMIT clause.
6. LIMIT clamp: if a LIMIT exists but exceeds `security.max_limit`, it is
   auto-clamped (not rejected) and the mutation is recorded for audit.

We deliberately use AST inspection rather than regex/string matching because
regex can be bypassed by quoting / comments / case / unicode normalization.
"""
import time
from typing import List, Optional, Tuple

from sqlglot import exp, parse, parse_one
from sqlglot.errors import ParseError

from semantic_search.config.nl_to_sql_models import SqlSecurityConfig
from semantic_search.core.exceptions import ValidationError as AgentValidationError
from semantic_search.core.logging_utils import get_logger
from semantic_search.nl_to_sql.contracts import SqlValidationResult
from semantic_search.nl_to_sql.schema import SchemaCatalog

logger = get_logger(__name__)


_FORBIDDEN_AST_NAMES = frozenset({
    'Insert', 'Update', 'Delete', 'Merge', 'AlterTable', 'AlterColumn',
    'AlterSchema', 'Create', 'Drop', 'TruncateTable', 'Grant', 'Revoke',
    'Use', 'Set', 'SetItem', 'Command', 'Pragma',
})

# Baseline PII column block — operators can extend via SqlSecurityConfig.pii_columns,
# but never silently undercut. The merged set is computed once in __init__.
# Intentionally narrow: only column names that are PII regardless of table context.
_BASELINE_PII_COLUMNS = frozenset({
    'email',
    'phone',
    'phone_number',
    'ip_address',
    'visitor_id',
    'shopper_id',
    'user_agent',
    'session_id',
    'cookie',
    'device_id',
    'first_name',
    'last_name',
    'full_name',
})


class AstSecurityValidator:
    """Stage-3 AST security gate. Returns a `SqlValidationResult` (never raises).

    The pipeline treats the typed result as the contract — an exception here
    would skip the structured failure_mode and break observability. Only
    construction-time misconfiguration raises (caught at boot).

    :param config: SqlSecurityConfig - Allowed tables, PII column list, limit cap
    :param dialect: str - sqlglot dialect (e.g. 'trino', 'presto', 'athena')
    :param schema_catalog: Optional[SchemaCatalog] - When supplied, ``SELECT *``
        projections are auto-expanded to the explicit column list from the
        catalog (so the PII gate can subsequently catch sensitive columns and
        schema drift cannot silently widen what's returned). When ``None``,
        ``SELECT *`` is hard-rejected with ``select_star_unexpandable`` —
        we never silently pass an unbounded projection.
    """

    def __init__(self, config: SqlSecurityConfig, dialect: str, schema_catalog: Optional[SchemaCatalog] = None):
        if not isinstance(config, SqlSecurityConfig):
            raise AgentValidationError("AstSecurityValidator requires a SqlSecurityConfig")
        if not isinstance(dialect, str) or not dialect:
            raise AgentValidationError("AstSecurityValidator requires a non-empty dialect")
        if schema_catalog is not None and not isinstance(schema_catalog, SchemaCatalog):
            raise AgentValidationError("AstSecurityValidator schema_catalog must be a SchemaCatalog instance or None")
        self._config = config
        self._dialect = dialect
        self._schema_catalog = schema_catalog
        self._allowed_tables_lc = frozenset(t.lower() for t in config.allowed_tables)
        # Merge baseline PII set with operator-supplied list. Operators can
        # extend the blocklist but never silently undercut the baseline.
        self._pii_columns_lc = frozenset(
            c.lower() for c in (_BASELINE_PII_COLUMNS | set(config.pii_columns))
        )

    def validate(self, sql: str) -> SqlValidationResult:
        """Validate `sql` and return a typed verdict.

        :param sql: str - Raw SQL produced by the generator
        :return: SqlValidationResult - is_valid, possibly clamped sql, failure_mode, audit
        """
        t0 = time.monotonic()
        if not isinstance(sql, str) or not sql.strip():
            return self._fail('syntax', "empty SQL", "_empty_", t0)
        try:
            statements = parse(sql, dialect=self._dialect)
        except ParseError as e:
            return self._fail('syntax', f"sqlglot parse error: {e}", sql, t0)
        statements = [s for s in statements if s is not None]
        if not statements:
            return self._fail('syntax', "no parseable statement", sql, t0)
        if len(statements) > 1:
            return self._fail(
                'security',
                f"multiple statements rejected (count={len(statements)})",
                sql,
                t0,
            )
        ast = statements[0]
        type_name = type(ast).__name__
        if type_name in _FORBIDDEN_AST_NAMES:
            return self._fail('security', f"forbidden statement type {type_name}", sql, t0)
        if not isinstance(ast, exp.Select) and not (
            isinstance(ast, (exp.Subqueryable, exp.Query)) if hasattr(exp, 'Subqueryable') else False
        ):
            try:
                ast.find(exp.Select)
            except Exception as e:
                logger.warning(f"security_ast_validation_skipped error_type={type(e).__name__}")
                return self._fail('security', f"non-SELECT root: {type_name}", sql, t0)
            if ast.find(exp.Select) is None:
                return self._fail('security', f"non-SELECT root: {type_name}", sql, t0)
        for forbidden in _FORBIDDEN_AST_NAMES:
            forbidden_cls = getattr(exp, forbidden, None)
            if forbidden_cls is None:
                continue
            if ast.find(forbidden_cls):
                return self._fail(
                    'security', f"forbidden node type {forbidden} in subtree", sql, t0
                )
        table_violations = self._check_tables(ast, exp)
        if table_violations:
            return self._fail(
                'security',
                "table allowlist violation: " + ", ".join(table_violations),
                sql,
                t0,
            )
        # Expand SELECT * BEFORE the PII column check so the PII gate can
        # see the explicit column list. _expand_star_projections accumulates
        # mutations that we forward into the final result.
        star_mutations, star_failure = self._expand_star_projections(ast, exp)
        if star_failure is not None:
            return self._fail('security', star_failure, sql, t0)
        pii_violations = self._check_pii_columns(ast, exp)
        if pii_violations:
            return self._fail(
                'security',
                "PII column reference: " + ", ".join(pii_violations),
                sql,
                t0,
            )
        catalog_violations = self._check_columns_in_catalog(ast, exp)
        if catalog_violations:
            return self._fail(
                'security',
                "column not in catalog: " + ", ".join(catalog_violations),
                sql,
                t0,
            )
        mutated_sql, limit_mutations, clamp_failure = self._enforce_limit(ast, exp, sql)
        if clamp_failure is not None:
            return self._fail('security', clamp_failure, sql, t0)
        # Run AFTER limit enforcement so the time-window predicate is the last
        # AST mutation and the re-rendered SQL is what's returned.
        time_window_mutations, time_window_failure = self._inject_time_window(ast, exp)
        if time_window_failure is not None:
            return self._fail('beyond_data_window', time_window_failure, sql, t0)
        if self._config.require_where_or_limit and not self._has_where_or_limit(ast, exp):
            return self._fail(
                'security',
                "query must contain a WHERE clause or a LIMIT clause",
                sql,
                t0,
            )
        # If the AST was mutated by star expansion or time-window injection
        # but _enforce_limit didn't re-render (because no LIMIT clamp fired),
        # re-render now so all mutations are reflected in the returned SQL.
        if (star_mutations or time_window_mutations) and not limit_mutations:
            try:
                mutated_sql = ast.sql(dialect=self._dialect)
            except Exception as e:  # pragma: no cover — defensive
                return self._fail('security', f"ast re-render failed: {e}", sql, t0)
        elif time_window_mutations and limit_mutations:
            # Both fired — re-render again so the time-window predicate is included.
            try:
                mutated_sql = ast.sql(dialect=self._dialect)
            except Exception as e:  # pragma: no cover — defensive
                return self._fail('security', f"ast re-render failed: {e}", sql, t0)
        latency_ms = (time.monotonic() - t0) * 1000.0
        return SqlValidationResult(
            is_valid=True,
            sql=mutated_sql,
            failure_mode=None,
            failure_reasons=[],
            mutations=star_mutations + limit_mutations + time_window_mutations,
            estimated_rows=None,
            latency_ms=latency_ms,
        )

    def _check_tables(self, ast, exp_module) -> List[str]:
        """Return a list of disallowed fully-qualified table names found in the AST."""
        violations: List[str] = []
        for table in ast.find_all(exp_module.Table):
            name = (table.name or "").lower()
            if not name:
                continue
            if name not in self._allowed_tables_lc:
                qualified = self._qualified_table_name(table).lower()
                if qualified not in self._allowed_tables_lc and qualified.split('.')[-1] not in self._allowed_tables_lc:
                    violations.append(self._qualified_table_name(table))
        return violations

    @staticmethod
    def _qualified_table_name(table) -> str:
        """Produce 'db.table' or 'catalog.db.table' from a sqlglot Table node."""
        parts: List[str] = []
        for piece in (getattr(table, 'catalog', None), getattr(table, 'db', None), getattr(table, 'name', None)):
            if piece:
                parts.append(str(piece))
        return ".".join(parts) if parts else (table.name or "_unknown_")

    def _check_pii_columns(self, ast, exp_module) -> List[str]:
        """Return a list of PII column names found in any projection / predicate."""
        if not self._pii_columns_lc:
            return []
        violations: List[str] = []
        for column in ast.find_all(exp_module.Column):
            col_name = (column.name or "").lower()
            if col_name and col_name in self._pii_columns_lc:
                violations.append(column.name)
        return violations

    def _check_columns_in_catalog(self, ast, exp_module) -> List[str]:
        """Return a list of column names not found in the schema catalog.

        Only active when ``config.validate_columns_against_catalog=True`` AND
        a ``schema_catalog`` is wired at construction time. When either condition
        is absent, returns an empty list (no-op).

        Walks every ``Column`` node in the AST and validates each name against
        the union of known column names across catalog-covered tables referenced
        in the query. Tables absent from the catalog (e.g. materialized-view
        tables not yet catalogued) are skipped — their columns are not validated
        rather than treated as violations. PII columns are excluded here because
        they are already caught by ``_check_pii_columns``.
        """
        if not self._config.validate_columns_against_catalog or self._schema_catalog is None:
            return []
        table_columns: dict = {}
        for table in ast.find_all(exp_module.Table):
            name = (table.name or '').lower()
            if name and name not in table_columns:
                cols = self._schema_catalog.columns_for(name)
                if cols:
                    table_columns[name] = frozenset(c.name.lower() for c in cols)
        if not table_columns:
            # No catalog-covered tables in this query — skip column validation.
            return []
        all_known = frozenset().union(*table_columns.values())

        # Collect aliases defined within this query (SELECT expression aliases,
        # CTE names, subquery aliases) so outer-query Column references to derived
        # names (e.g. ORDER BY avg_bid, SELECT count FROM cte) aren't flagged —
        # they're not source-table columns.
        defined_aliases: set = set()
        for select in ast.find_all(exp_module.Select):
            for expr in select.expressions:
                alias = getattr(expr, 'alias', None)
                if alias:
                    defined_aliases.add(str(alias).lower())
        with_clause = ast.args.get('with')
        if with_clause is not None:
            for cte in (with_clause.expressions or []):
                cte_alias = getattr(cte, 'alias', None) or getattr(cte, 'name', None)
                if cte_alias:
                    defined_aliases.add(str(cte_alias).lower())
        for subquery in ast.find_all(exp_module.Subquery):
            sq_alias = getattr(subquery, 'alias', None)
            if sq_alias:
                defined_aliases.add(str(sq_alias).lower())

        violations: List[str] = []
        seen: set = set()
        for column in ast.find_all(exp_module.Column):
            col_name = (column.name or '').lower()
            if not col_name or col_name in self._pii_columns_lc:
                continue
            if col_name in defined_aliases:
                continue
            # Column references a non-base-table qualifier (CTE or subquery alias).
            table_qualifier = str(getattr(column, 'table', '') or '').lower()
            if table_qualifier and table_qualifier not in table_columns and table_qualifier in defined_aliases:
                continue
            if col_name not in all_known and col_name not in seen:
                seen.add(col_name)
                violations.append(column.name)
        return violations

    def _expand_star_projections(self, ast, exp_module) -> Tuple[List[str], Optional[str]]:
        """Replace ``SELECT *`` with explicit column lists from the schema catalog.

        Runs BEFORE _check_pii_columns so the PII gate can subsequently catch
        sensitive columns the catalog reveals. When SELECT * cannot be safely
        expanded (no catalog wired, or table not in catalog, or multiple
        tables in FROM with unqualified ``*``), returns a hard-reject reason —
        we never silently pass an unbounded projection.

        :return: (mutations, failure_reason). failure_reason None on success.
        """
        mutations: List[str] = []
        for select in list(ast.find_all(exp_module.Select)):
            star_indexes: List[int] = []
            qualified_stars: List[Tuple[int, str]] = []
            for idx, projection in enumerate(list(select.expressions)):
                if isinstance(projection, exp_module.Star):
                    star_indexes.append(idx)
                elif isinstance(projection, exp_module.Column) and isinstance(projection.this, exp_module.Star):
                    qualified_stars.append((idx, str(projection.table) if projection.table else ''))
            if not star_indexes and not qualified_stars:
                continue
            if self._schema_catalog is None:
                return [], "select_star_unexpandable: no schema catalog wired"
            from_clause = select.args.get('from')
            tables: List[exp_module.Table] = list(select.find_all(exp_module.Table)) if from_clause is not None else []
            # Resolve which table to expand against for unqualified `*`.
            if star_indexes:
                if len(tables) != 1:
                    return [], (
                        f"select_star_unexpandable: unqualified '*' with {len(tables)} tables in FROM"
                    )
                base_table = tables[0]
                base_name = (base_table.name or '').lower()
                cols = self._schema_catalog.columns_for(base_name) if base_name else []
                if not cols:
                    return [], f"select_star_unexpandable: catalog has no columns for table '{base_name}'"
                replacement_columns = [exp_module.Column(this=exp_module.to_identifier(c.name)) for c in cols]
                # Replace each unqualified * with the explicit column list.
                # Iterate from highest index so list mutation doesn't shift others.
                expressions = list(select.expressions)
                for idx in sorted(star_indexes, reverse=True):
                    expressions[idx:idx + 1] = [exp_module.Column(this=exp_module.to_identifier(c.name)) for c in cols]
                select.set('expressions', expressions)
                mutations.append(f"select_star_expanded table={base_name} columns={len(cols)}")
            for idx, table_alias in qualified_stars:
                # Find the table that the qualifier refers to (alias-aware).
                target_name: Optional[str] = None
                for t in tables:
                    alias = ''
                    try:
                        alias_node = t.args.get('alias')
                        if alias_node is not None and hasattr(alias_node, 'name'):
                            alias = str(alias_node.name)
                    except Exception:
                        alias = ''
                    if alias and alias.lower() == table_alias.lower():
                        target_name = (t.name or '').lower()
                        break
                    if (t.name or '').lower() == table_alias.lower():
                        target_name = (t.name or '').lower()
                        break
                if not target_name:
                    return [], f"select_star_unexpandable: qualifier '{table_alias}' not bound to a FROM table"
                cols = self._schema_catalog.columns_for(target_name)
                if not cols:
                    return [], f"select_star_unexpandable: catalog has no columns for table '{target_name}'"
                expressions = list(select.expressions)
                replacement = [
                    exp_module.Column(
                        this=exp_module.to_identifier(c.name),
                        table=exp_module.to_identifier(table_alias) if table_alias else None,
                    )
                    for c in cols
                ]
                expressions[idx:idx + 1] = replacement
                select.set('expressions', expressions)
                mutations.append(f"select_star_expanded table={target_name} alias={table_alias} columns={len(cols)}")
        return mutations, None

    def _enforce_limit(self, ast, exp_module, original_sql: str) -> Tuple[str, List[str], Optional[str]]:
        """If a LIMIT exists and exceeds the cap, clamp it. Returns (sql, mutations, fail_msg).

        We never *add* a LIMIT here because that would change semantics for
        aggregate queries (e.g. SELECT COUNT(*) ... LIMIT 1000 is a no-op).
        Whether a LIMIT is REQUIRED is gated by `_has_where_or_limit` instead.
        """
        mutations: List[str] = []
        max_limit = int(self._config.max_limit)
        for limit_node in list(ast.find_all(exp_module.Limit)):
            expression = limit_node.expression
            if expression is None:
                continue
            value: Optional[int] = None
            try:
                if isinstance(expression, exp_module.Literal) and expression.is_int:
                    value = int(expression.this)
            except Exception as e:
                logger.warning(f"limit_clamp_parse_failed error_type={type(e).__name__}")
                value = None
            if value is None:
                continue
            if value > max_limit:
                mutations.append(f"limit_clamped from={value} to={max_limit}")
                limit_node.set('expression', exp_module.Literal.number(max_limit))
        if mutations:
            try:
                rewritten = ast.sql(dialect=self._dialect)
            except Exception as e:
                return original_sql, [], f"limit clamp re-render failed: {e}"
            return rewritten, mutations, None
        return original_sql, mutations, None

    def _inject_time_window(self, ast, exp_module) -> Tuple[List[str], Optional[str]]:
        """Inject / validate time-window predicates per ``BulkTimeWindowConfig``.

        For every configured (table, column, max_days, default_days):
          - When the query references the table and has NO predicate on the
            column → inject ``column >= now() - INTERVAL <default_days> DAY``.
          - When a predicate exists with a literal ``INTERVAL N DAY`` and
            ``N <= max_days`` → leave alone.
          - When the existing predicate exceeds ``max_days`` → reject.

        Lookups are case-insensitive on table and column names. Predicates
        deeper than the top-level WHERE (subqueries) are not inspected; the
        outer scan is what matters for cost.

        :return: (mutations, failure_reason). failure_reason is None on success.
        """
        if not self._config.bulk_time_windows:
            return [], None
        mutations: List[str] = []
        # Build {lower(table_name): config} for quick lookup.
        windows = {tw.table_name.lower(): tw for tw in self._config.bulk_time_windows}
        # Walk every Select in the AST. For multi-table joins, apply each
        # configured window if its target table appears in the FROM list.
        for select in list(ast.find_all(exp_module.Select)):
            tables_in_from = list(select.find_all(exp_module.Table))
            for tbl in tables_in_from:
                tname = (tbl.name or '').lower()
                if tname not in windows:
                    continue
                tw = windows[tname]
                col_lower = tw.column.lower()
                where = select.args.get('where')
                existing_days = self._extract_interval_days(where, exp_module, col_lower) if where is not None else None
                if existing_days is not None:
                    if existing_days > tw.max_days:
                        return [], (
                            f"time_window_exceeds_max table={tname} column={tw.column} "
                            f"requested_days={existing_days} max_days={tw.max_days}"
                        )
                    continue
                # No predicate on the configured column — inject the default.
                injected = self._build_time_window_predicate(exp_module, tw.column, tw.default_days)
                if where is None:
                    select.set('where', exp_module.Where(this=injected))
                else:
                    new_where_expr = exp_module.And(this=where.this, expression=injected)
                    where.set('this', new_where_expr)
                mutations.append(
                    f"time_window_injected table={tname} column={tw.column} default_days={tw.default_days}"
                )
        return mutations, None

    @staticmethod
    def _extract_interval_days(where, exp_module, target_col_lower: str) -> Optional[int]:
        """Pull the largest ``INTERVAL N DAY`` literal that compares to ``target_col``.

        Returns the day count when found, ``None`` when no relevant predicate
        exists. We use the largest interval (most permissive) so the max_days
        gate enforces the WORST case rather than the best.
        """
        largest: Optional[int] = None
        for cmp_node in where.find_all(exp_module.GTE, exp_module.GT, exp_module.LT, exp_module.LTE, exp_module.EQ):
            left = cmp_node.this
            if not isinstance(left, exp_module.Column):
                continue
            if (left.name or '').lower() != target_col_lower:
                continue
            for interval in cmp_node.find_all(exp_module.Interval):
                value = interval.this
                unit = interval.args.get('unit')
                if value is None or unit is None:
                    continue
                unit_name = ''
                try:
                    unit_name = str(unit.name).upper() if hasattr(unit, 'name') else str(unit).upper()
                except Exception:
                    unit_name = ''
                if 'DAY' not in unit_name:
                    continue
                try:
                    n = int(str(value.this) if hasattr(value, 'this') else str(value))
                except Exception:
                    continue
                if largest is None or n > largest:
                    largest = n
        return largest

    @staticmethod
    def _build_time_window_predicate(exp_module, column: str, days: int):
        """Build ``column >= now() - INTERVAL <days> DAY`` as a sqlglot AST node."""
        col_node = exp_module.Column(this=exp_module.to_identifier(column))
        now_node = exp_module.Anonymous(this='now')
        interval_node = exp_module.Interval(
            this=exp_module.Literal.number(int(days)),
            unit=exp_module.Var(this='DAY'),
        )
        sub_node = exp_module.Sub(this=now_node, expression=interval_node)
        return exp_module.GTE(this=col_node, expression=sub_node)

    @staticmethod
    def _has_where_or_limit(ast, exp_module) -> bool:
        """True iff the top-level SELECT carries a WHERE clause OR a LIMIT clause.

        Subquery WHERE/LIMIT do not satisfy this gate — the OUTER query must
        bound the result set.
        """
        if isinstance(ast, exp_module.Select):
            select = ast
        else:
            select = ast.find(exp_module.Select)
        if select is None:
            return False
        if select.args.get('where') is not None:
            return True
        if select.args.get('limit') is not None:
            return True
        return False

    @staticmethod
    def _fail(failure_mode: str, reason: str, sql: str, t0: float) -> SqlValidationResult:
        """Build a failed `SqlValidationResult` and emit a structured warning."""
        latency_ms = (time.monotonic() - t0) * 1000.0
        logger.warning(f"ast_security_failed mode={failure_mode} reason={reason!r} latency_ms={latency_ms:.1f}")
        sql_for_record = sql if (isinstance(sql, str) and sql.strip()) else "_empty_"
        return SqlValidationResult(
            is_valid=False,
            sql=sql_for_record,
            failure_mode=failure_mode,
            failure_reasons=[reason],
            mutations=[],
            estimated_rows=None,
            latency_ms=latency_ms,
        )


__all__ = ['AstSecurityValidator']
