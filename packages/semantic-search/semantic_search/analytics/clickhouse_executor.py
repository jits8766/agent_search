"""ClickHouse-backed SQL executor (analytics path equivalent of `SqlExecutor`).

Wraps `ClickHouseClient` with the same `SqlExecutionResult` contract the
existing NL-to-SQL pipeline already understands. Keeping the contract identical
means downstream consumers (verifier, telemetry, surface) need zero changes
when traffic flips from Athena → ClickHouse.

Construction is dependency-injected (no implicit network IO at boot).
"""
import re as _re
import time

from semantic_search.analytics.clickhouse_client import ClickHouseClient

_LIMIT_RE = _re.compile(r'\bLIMIT\b', _re.IGNORECASE)
from semantic_search.config.nl_to_sql_models import SqlExecutionConfig
from semantic_search.core.exceptions import RetrievalError
from semantic_search.core.logging_utils import get_logger
from semantic_search.nl_to_sql.contracts import SqlExecutionResult

logger = get_logger(__name__)


class ClickHouseExecutor:
    """ClickHouse executor (analytics path, SqlExecutionResult contract)."""

    def __init__(self, config: SqlExecutionConfig, client: ClickHouseClient):
        if not isinstance(config, SqlExecutionConfig):
            raise RetrievalError("ClickHouseExecutor requires a SqlExecutionConfig")
        if not isinstance(client, ClickHouseClient):
            raise RetrievalError("ClickHouseExecutor requires a ClickHouseClient")
        self._config = config
        self._client = client

    @property
    def credentials_available(self) -> bool:
        """Mirrors `SqlExecutor.credentials_available` so callers stay polymorphic."""
        return self._client.available

    @property
    def database(self) -> str:
        """ClickHouse database (used as database_override in analytics miss path)."""
        return self._client.database

    async def execute(self, sql: str) -> SqlExecutionResult:
        """Execute SQL, return result (inject LIMIT if missing, truncate if needed)."""
        if not isinstance(sql, str) or not sql.strip():
            raise RetrievalError("ClickHouseExecutor.execute requires a non-empty SQL string")
        effective_sql = sql
        if not _LIMIT_RE.search(sql):
            effective_sql = f"{sql.rstrip().rstrip(';')} LIMIT {self._config.max_rows}"
            logger.info(f"clickhouse_limit_injected max_rows={self._config.max_rows}")
        t0 = time.monotonic()
        rows, columns, _ch_latency_ms = await self._client.execute_query(
            query=effective_sql,
            timeout_seconds=float(self._config.timeout_seconds),
        )
        truncated = False
        if len(rows) > self._config.max_rows:
            logger.warning(f"clickhouse_execution_row_truncation rows={len(rows)} max_rows={self._config.max_rows}")
            rows = rows[: self._config.max_rows]
            truncated = True
        latency_ms = (time.monotonic() - t0) * 1000.0
        return SqlExecutionResult(sql=sql, rows=rows, column_names=columns, row_count=len(rows), latency_ms=latency_ms, truncated=truncated)

    async def explain(self, sql: str, timeout_seconds: float = 10.0) -> None:
        """EXPLAIN SQL (validates syntax without executing)."""
        if not isinstance(sql, str) or not sql.strip():
            raise RetrievalError("ClickHouseExecutor.explain requires a non-empty SQL string")
        await self._client.execute_query(
            query=f"EXPLAIN {sql}",
            timeout_seconds=float(timeout_seconds),
            _skip_mv_freshness_probe=True,
        )

    async def execute_insert(self, sql: str, timeout_seconds: float = 60.0) -> None:
        """Execute INSERT/DDL (skip MV probe, no response parsing)."""
        if not isinstance(sql, str) or not sql.strip():
            raise RetrievalError("ClickHouseExecutor.execute_insert requires a non-empty SQL string")
        await self._client.execute_query(
            query=sql,
            timeout_seconds=float(timeout_seconds),
            _skip_mv_freshness_probe=True,
        )


__all__ = ['ClickHouseExecutor']
