"""SQL executor (stage 5).

Thin wrapper that delegates to a ClickHouseExecutor-compatible executor.
Duck-typed on the two properties the pipeline consumes so tests can
substitute fakes without subclassing.

Interface required from the injected executor:
  - async execute(sql: str) -> SqlExecutionResult
  - credentials_available: bool  (property)
"""
import time
from typing import Any

from semantic_search.config.nl_to_sql_models import SqlExecutionConfig
from semantic_search.core.exceptions import RetrievalError
from semantic_search.core.logging_utils import get_logger
from semantic_search.nl_to_sql.contracts import SqlExecutionResult

logger = get_logger(__name__)


class SqlExecutor:
    """Stage-5 executor backed by any executor exposing execute(sql) and credentials_available.

    :param config: SqlExecutionConfig - max_rows + timeout_seconds (kept for interface compat)
    :param ch_executor: Any - Executor with async execute() and credentials_available
    """

    def __init__(self, config: SqlExecutionConfig, ch_executor: Any) -> None:
        if not isinstance(config, SqlExecutionConfig):
            raise RetrievalError("SqlExecutor requires a SqlExecutionConfig")
        if not callable(getattr(ch_executor, 'execute', None)):
            raise RetrievalError("SqlExecutor requires an executor with an async execute() method")
        self._config = config
        self._ch = ch_executor

    @property
    def credentials_available(self) -> bool:
        """Forward the underlying executor's credential availability flag."""
        return bool(self._ch.credentials_available)

    async def execute(self, sql: str) -> SqlExecutionResult:
        """Execute `sql` and return a typed result.

        :param sql: str - Validated SQL (post-security, post-logic)
        :return: SqlExecutionResult - rows + columns + metadata
        :raises RetrievalError: On execution failure
        """
        if not isinstance(sql, str) or not sql.strip():
            raise RetrievalError("SqlExecutor.execute requires a non-empty SQL string")
        return await self._ch.execute(sql)


__all__ = ['SqlExecutor']
