"""FeedbackSignal to ClickHouse persistence tier.

Writes :class:`~semantic_search.contracts.FeedbackSignal` events to the
``analytics.feedback_signals`` table via
:class:`~semantic_search.analytics.clickhouse_executor.ClickHouseExecutor`.

Design constraints:
  - ``write()`` is fire-and-forget: ClickHouse errors are logged as warnings
    and swallowed so a CH outage never blocks the main request path.
  - ``write_batch()`` batches up to ``SignalWriterConfig.batch_size`` signals
    in a single INSERT VALUES statement; oversized lists are chunked automatically.
  - ``payload`` is serialised as a JSON string (stored in a ClickHouse ``String``
    column) — no schema migration needed to add new payload keys.
  - When ``enabled=False`` both methods are no-ops with no CH round-trip.
  - The class is NOT responsible for reading signals back; use
    :class:`DomainAnalyticsEngine` for that.
"""
from __future__ import annotations

import json
from typing import List

from semantic_search.analytics.clickhouse_executor import ClickHouseExecutor
from semantic_search.config.analytics_models import SignalWriterConfig
from semantic_search.contracts import FeedbackSignal
from semantic_search.core.exceptions import ConfigurationError
from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)

# Escape single-quotes as '' (SQL standard) for INSERT VALUES
_ESCAPE = str.maketrans({"'": "''"})

def _escape(value: str) -> str:
    """Escape single-quotes for ClickHouse literal embedding."""
    return value.translate(_ESCAPE)


def _signal_to_values_row(signal: FeedbackSignal) -> str:
    """Render FeedbackSignal as VALUES row (order: id, request_id, type, payload, origin, created_at).

    ``search_id`` is stored inside the JSON ``payload`` column for ClickHouse joins.
    """
    payload = dict(signal.payload) if isinstance(signal.payload, dict) else {}
    if signal.search_id:
        payload.setdefault("search_id", signal.search_id)
    payload_json = _escape(json.dumps(payload, ensure_ascii=True))
    return (
        f"('{_escape(signal.signal_id)}', "
        f"'{_escape(signal.request_id)}', "
        f"'{_escape(signal.signal_type)}', "
        f"'{payload_json}', "
        f"'{_escape(signal.signal_origin)}', "
        f"fromUnixTimestamp64Milli({int(signal.created_at * 1000)}))"
    )


class FeedbackSignalWriter:
    """Persist FeedbackSignal events to ClickHouse (fire-and-forget, logs errors)."""

    def __init__(self, executor: ClickHouseExecutor, config: SignalWriterConfig) -> None:
        if not isinstance(executor, ClickHouseExecutor):
            raise ConfigurationError("FeedbackSignalWriter requires a ClickHouseExecutor")
        if not isinstance(config, SignalWriterConfig):
            raise ConfigurationError("FeedbackSignalWriter requires a SignalWriterConfig")
        self._executor = executor
        self._config = config

    @property
    def enabled(self) -> bool:
        """True iff writer is configured on and the executor has credentials."""
        return bool(self._config.enabled) and self._executor.credentials_available

    async def write(self, signal: FeedbackSignal) -> None:
        """Insert single signal (never raises, logs CH errors)."""
        if not self.enabled:
            return
        if not isinstance(signal, FeedbackSignal):
            logger.warning("signal_writer_write_skipped reason=not_a_FeedbackSignal")
            return
        sql = (
            f"INSERT INTO {self._config.signals_table} "
            f"(signal_id, request_id, signal_type, payload, signal_origin, created_at) VALUES "
            f"{_signal_to_values_row(signal)}"
        )
        try:
            await self._executor.execute_insert(sql, timeout_seconds=float(self._config.insert_timeout_seconds))
        except Exception as exc:
            logger.warning(f"signal_writer_insert_failed signal_type={signal.signal_type} error_type={type(exc).__name__} error={exc}")

    async def write_batch(self, signals: List[FeedbackSignal]) -> None:
        """Bulk insert signals in chunks (drops non-FeedbackSignal, never raises, logs errors)."""
        if not self.enabled:
            return
        valid = [s for s in signals if isinstance(s, FeedbackSignal)]
        dropped = len(signals) - len(valid)
        if dropped:
            logger.warning(f"signal_writer_batch_dropped_non_signals count={dropped}")
        if not valid:
            return
        batch_size = int(self._config.batch_size)
        for offset in range(0, len(valid), batch_size):
            chunk = valid[offset: offset + batch_size]
            rows_sql = ',\n'.join(_signal_to_values_row(s) for s in chunk)
            sql = (
                f"INSERT INTO {self._config.signals_table} "
                f"(signal_id, request_id, signal_type, payload, signal_origin, created_at) VALUES\n"
                f"{rows_sql}"
            )
            try:
                await self._executor.execute_insert(sql, timeout_seconds=float(self._config.insert_timeout_seconds))
                logger.info(f"signal_writer_batch_inserted count={len(chunk)} table={self._config.signals_table}")
            except Exception as exc:
                logger.warning(f"signal_writer_batch_insert_failed chunk_size={len(chunk)} error_type={type(exc).__name__} error={exc}")


__all__ = ['FeedbackSignalWriter']
