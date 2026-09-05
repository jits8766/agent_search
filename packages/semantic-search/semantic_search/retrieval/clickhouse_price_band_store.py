"""ClickHouse-backed `PriceBandStore` for the real-time price fan-out.

When a query needs a real-time price check, the pipeline fans out to SQL
and merges the results. The substrates table specifies ClickHouse
``events_raw`` as the live backend. This adapter satisfies the
``PriceBandStore`` async contract by composing a parametrized SELECT against
a configured table and forwarding it to a shared `ClickHouseClient`.

Security model (boot-time + per-call):
- The target table identifier is whitelisted at boot via
  `ClickHousePriceBandAdapterConfig.__post_init__` (regex on
  ``[A-Za-z_][A-Za-z0-9_]{0,63}(\\.[A-Za-z_][A-Za-z0-9_]{0,63})?``).
- The per-row score expression is whitelisted at boot (rejects ``;``, SQL
  comments, non-printable chars).
- Per-call: every filter value is re-validated against a strict per-column
  type/charset rule BEFORE composition. Any value that fails validation
  causes the column to be silently dropped (logged at WARNING). Strings
  are emitted as ClickHouse single-quoted literals with backslash + quote
  escaping; integers are emitted as bare digit tokens. There is NO string
  concatenation of unvalidated user input into the SQL.

Failure semantics:
- ``ClickHouseUnavailableError`` / ``ClickHouseQueryError`` / generic
  ``RetrievalError`` → return ``[]`` (graceful degradation; the upstream
  ``BackendHealthRegistry`` already has the per-call health signal recorded
  by `ClickHouseClient` for the breaker).
- The error category is logged WITHOUT the SQL payload (per
  `responsible-ai.mdc` — don't echo potentially-sensitive query bodies).

Schema contract for the target table (validated by ClickHouse at query time,
NOT by this module):
- Required columns: ``item_id`` (String) and the operator-supplied
  ``score_expr`` resolves to a numeric column or expression.
- Optional columns: ``price`` (Int64), ``tld`` (LowCardinality(String)),
  ``auction_type`` (LowCardinality(String)), ``name_length`` (Int32). When
  the corresponding filter is present and the column is missing the query
  errors out, which surfaces as a single WARNING + an empty result (no
  cascade failure).
"""
import re
from typing import Any, Dict, List, Optional, Tuple

from semantic_search.analytics.clickhouse_client import ClickHouseClient, ClickHouseQueryError, ClickHouseUnavailableError
from semantic_search.config.models import ClickHousePriceBandAdapterConfig
from semantic_search.core.exceptions import RetrievalError
from semantic_search.core.logging_utils import get_logger
from semantic_search.retrieval.sql_retriever import PriceBandStore

logger = get_logger(__name__)


_STRING_FILTER_RE = re.compile(r'^[A-Za-z0-9_.\-]{1,32}$')
_LIST_STRING_FILTERS = frozenset({'tld', 'auction_type'})
_INT_FILTERS = frozenset({'price_min', 'price_max', 'name_length_max'})


def _escape_ch_string(value: str) -> str:
    """Single-quote a string for ClickHouse SQL literals.

    Escapes backslash and single-quote per ClickHouse syntax. Caller MUST
    have already validated the value against the allowlist regex — this
    function only handles the literal-quoting concern (defence-in-depth).
    """
    escaped = value.replace('\\', '\\\\').replace("'", "\\'")
    return f"'{escaped}'"


def _normalize_string_value(value: Any) -> Optional[List[str]]:
    """Coerce a filter value to a list of validated string tokens.

    Returns None when the value is empty or contains any token that fails
    the allowlist regex — the caller drops the whole column on None.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
    else:
        items = [value]
    if not items:
        return None
    out: List[str] = []
    for it in items:
        if not isinstance(it, str):
            return None
        if not _STRING_FILTER_RE.match(it):
            return None
        out.append(it)
    return out


def _normalize_int_value(value: Any) -> Optional[int]:
    """Coerce a filter value to an int. Returns None on any failure."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class ClickHousePriceBandStore(PriceBandStore):
    """`PriceBandStore` backed by ClickHouse ``events_raw`` (or aliased table).

    :param config: ClickHousePriceBandAdapterConfig - Adapter config (table,
        score_expr, timeout, max_rows). Identifier guards run at boot via
        the config dataclass `__post_init__`.
    :param client: ClickHouseClient - Shared CH HTTP client (typically
        reused from the analytics router; constructed dedicated when
        analytics is disabled but the price fan-out is enabled)
    """

    def __init__(self, config: ClickHousePriceBandAdapterConfig, client: ClickHouseClient):
        if not isinstance(config, ClickHousePriceBandAdapterConfig):
            raise RetrievalError("ClickHousePriceBandStore requires a ClickHousePriceBandAdapterConfig")
        if not isinstance(client, ClickHouseClient):
            raise RetrievalError("ClickHousePriceBandStore requires a ClickHouseClient")
        self._config = config
        self._client = client
        logger.info(
            f"clickhouse_price_band_store_ready table={self._config.table} "
            f"score_expr_len={len(self._config.score_expr)} timeout_s={self._config.timeout_seconds:.2f} "
            f"max_rows={self._config.max_rows}"
        )

    async def lookup(self, filters: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
        if top_k < 1:
            return []
        sql = self._compose_sql(filters, top_k)
        if sql is None:
            return []
        try:
            rows, _columns, _latency_ms = await self._client.execute_query(query=sql, timeout_seconds=float(self._config.timeout_seconds))
        except (ClickHouseUnavailableError, ClickHouseQueryError, RetrievalError) as e:
            logger.warning(f"clickhouse_price_band_lookup_failed error_type={type(e).__name__} table={self._config.table} error={e}")
            return []
        return self._project_rows(rows)

    def _compose_sql(self, filters: Dict[str, Any], top_k: int) -> Optional[str]:
        """Compose a parametrized SELECT against the configured table.

        Returns None when no usable predicate could be assembled (the caller
        should treat that as an empty result, not an error — there is
        nothing meaningful to ask the warehouse).
        """
        where_clauses: List[str] = []
        for col in _LIST_STRING_FILTERS:
            if col not in filters:
                continue
            normalized = _normalize_string_value(filters[col])
            if normalized is None:
                logger.warning(f"clickhouse_price_band_filter_rejected column={col} reason=invalid_value")
                continue
            literals = ', '.join(_escape_ch_string(v) for v in normalized)
            # Use filter_column_expressions to get the physical SQL expression for
            # columns that are SELECT aliases rather than physical columns (e.g.
            # auction_type is CAST(auction_type_id AS VARCHAR) AS auction_type in
            # SELECT but auction_type is not a physical column ClickHouse can use
            # in WHERE).
            col_expr = self._config.filter_column_expressions.get(col, col)
            where_clauses.append(f"{col_expr} IN ({literals})")
        for col in _INT_FILTERS:
            if col not in filters:
                continue
            normalized_int = _normalize_int_value(filters[col])
            if normalized_int is None:
                logger.warning(f"clickhouse_price_band_filter_rejected column={col} reason=invalid_value")
                continue
            if col == 'price_min':
                where_clauses.append(f"{self._config.price_column} >= {normalized_int}")
            elif col == 'price_max':
                where_clauses.append(f"{self._config.price_column} <= {normalized_int}")
            elif col == 'name_length_max':
                where_clauses.append(f"length({self._config.item_id_column}) <= {normalized_int}")
        if self._config.base_where:
            where_clauses.insert(0, self._config.base_where)
        if not where_clauses:
            return None
        effective_limit = min(int(top_k), int(self._config.max_rows))
        if effective_limit < 1:
            return None
        id_col = self._config.item_id_column
        extra = (', ' + ', '.join(self._config.extra_columns)) if self._config.extra_columns else ''
        return (
            f"SELECT {id_col} AS item_id, ({self._config.score_expr}) AS score{extra} "
            f"FROM {self._config.table} "
            f"WHERE {' AND '.join(where_clauses)} "
            f"ORDER BY score DESC "
            f"LIMIT {effective_limit} "
            f"FORMAT JSONEachRow"
        )

    def _project_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Project ClickHouse rows to the `PriceBandStore` row shape.

        Drops rows missing ``item_id`` or with a non-numeric ``score``. The
        upstream contract guarantees ``item_id`` + ``score`` per row — the
        rest of the dict carries any additional columns the warehouse
        returned (kept under the ``payload`` slot by `SqlRetriever`).
        """
        out: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            item_id = row.get('item_id')
            score_raw = row.get('score')
            if item_id is None or score_raw is None:
                continue
            try:
                score = float(score_raw)
            except (TypeError, ValueError):
                continue
            projected = {'item_id': str(item_id), 'score': score}
            for k, v in row.items():
                if k in {'item_id', 'score'}:
                    continue
                projected[k] = v
            out.append(projected)
        # Normalise scores to [0, 1] within the batch so downstream Candidate
        # validation is satisfied regardless of what score_expr returns.
        # Dividing by the batch max preserves relative ranking.
        if out:
            max_score = max(r['score'] for r in out)
            if max_score > 0.0:
                for r in out:
                    r['score'] = r['score'] / max_score
            else:
                for r in out:
                    r['score'] = 0.0
        return out
