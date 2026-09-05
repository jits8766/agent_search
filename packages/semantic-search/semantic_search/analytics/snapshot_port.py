"""ClickHouse-backed historical snapshot substrate for past-tense analytics queries.

Queries ``analytics.domain_snapshots`` (365-day rolling point-in-time table) to
proxy historical-sales questions such as "how many .net domains sold last week".
The table tracks active-auction metrics at snapshot time — not completed sales —
so answers are labelled as proxy data (listings active during the period), not
confirmed sale records.

All SQL, table names, time windows, and signal terms are config-driven. No
literals are embedded in logic; everything flows from ``HistoricalSnapshotConfig``.
"""
import re
import time
from typing import Any, Dict, List, Optional

from semantic_search.analytics.clickhouse_executor import ClickHouseExecutor
from semantic_search.config.analytics_models import HistoricalSnapshotConfig
from semantic_search.core.logging_utils import get_logger
from semantic_search.nl_to_sql.contracts import AnalyticsResult

logger = get_logger(__name__)

# Extracts explicit "N days" / "N day" from question text (module-level, not config-driven
# because it is structural parsing syntax, not a tuneable threshold).
_EXPLICIT_DAYS_RE = re.compile(r'\b(\d+)\s+days?\b')

# hint key → WHERE clause template; {col} replaced at render time.
_HINT_FILTER_TEMPLATES: Dict[str, str] = {
    'tld': "tld = '{val}'",
    'price_max': 'current_price <= {val}',
    'price_min': 'current_price >= {val}',
}

# hint key extracted as list (multi-value) vs scalar
_HINT_LIST_KEYS = frozenset({'tld'})


def _parse_hint(sql_hint: str) -> Dict[str, List[str]]:
    """Extract key=value pairs from sql_hint string."""
    result: Dict[str, List[str]] = {}
    for m in re.finditer(r'(\w+)=([\w,.\-]+)', sql_hint or ''):
        key = m.group(1)
        vals = [v.strip() for v in m.group(2).split(',') if v.strip()]
        if vals:
            result[key] = vals
    return result


_NUMERIC_KEYS = frozenset({'price_max', 'price_min'})
_TLD_RE = re.compile(r'^[\w.\-]{1,64}$')


def _sanitize_val(key: str, val: str) -> str:
    """Validate and return a safe scalar value for SQL template substitution.

    Numeric keys are cast to float then re-stringified — this prevents any
    injection regardless of upstream parsing. TLD values are validated against
    a strict allowlist regex. Unknown keys whose values slip through _parse_hint
    are dropped via the template lookup above, so no extra case is needed here.
    """
    if key in _NUMERIC_KEYS:
        return str(float(val))
    if not _TLD_RE.match(val):
        raise ValueError(f"Invalid filter value for key {key!r}: {val!r}")
    return val


def _build_filter_clauses(hint_kv: Dict[str, List[str]], templates: Dict[str, str], list_keys: frozenset) -> str:
    """Build SQL WHERE fragment from parsed hint key-value pairs."""
    parts: List[str] = []
    for key, vals in hint_kv.items():
        tpl = templates.get(key)
        if tpl is None:
            continue
        try:
            safe_vals = [_sanitize_val(key, v) for v in vals]
        except ValueError:
            continue
        if key in list_keys:
            per_val = [tpl.format(val=v) for v in safe_vals]
            parts.append(f"({' OR '.join(per_val)})" if len(per_val) > 1 else per_val[0])
        else:
            parts.append(tpl.format(val=safe_vals[0]))
    return (' AND ' + ' AND '.join(parts)) if parts else ''


def _resolve_window_days(question: str, window_patterns: List[Dict[str, Any]], default_days: int) -> int:
    """Resolve the time window in days from the question text.

    Priority:
    1. Config-driven named patterns (e.g. "last week" → 7)
    2. Regex-extracted explicit numeric days ("14 days" → 14)
    3. Config default_window_days
    """
    q = question.lower()
    for wp in window_patterns:
        if str(wp.get('term', '')) in q:
            return int(wp['days'])
    m = _EXPLICIT_DAYS_RE.search(q)
    if m:
        n = int(m.group(1))
        return n if n >= 1 else default_days
    return default_days


class HistoricalSnapshotAnalyticsPort:
    """Query ``analytics.domain_snapshots`` as a proxy for historical-sales questions.

    :param config: HistoricalSnapshotConfig - Table name, SQL template, window patterns
    :param ch_executor: ClickHouseExecutor - Shared ClickHouse executor
    """

    def __init__(self, config: HistoricalSnapshotConfig, ch_executor: ClickHouseExecutor) -> None:
        if not isinstance(config, HistoricalSnapshotConfig):
            raise ValueError("HistoricalSnapshotAnalyticsPort requires a HistoricalSnapshotConfig")
        if not callable(getattr(ch_executor, 'execute', None)):
            raise ValueError("HistoricalSnapshotAnalyticsPort requires an executor with an async execute() method")
        self._config = config
        self._executor = ch_executor

    async def try_answer(self, *, question: str, sql_hint: str, request_id: str) -> Optional[AnalyticsResult]:
        """Execute a snapshot query and return a typed result, or None to fall back.

        :param question: str - Original NL question
        :param sql_hint: str - Structured filter hint from QI engine
        :param request_id: str - Correlation id
        :return: Optional[AnalyticsResult] - Typed result on success, None on any failure
        """
        if not self._config.query_template:
            logger.warning(f"analytics_snapshot_no_template request_id={request_id}")
            return None
        t0 = time.monotonic()
        hint_kv = _parse_hint(sql_hint)
        filter_clauses = _build_filter_clauses(hint_kv, _HINT_FILTER_TEMPLATES, _HINT_LIST_KEYS)
        window_days = _resolve_window_days(question, self._config.window_patterns, self._config.default_window_days)
        sql = self._config.query_template.format(
            snapshot_table=self._config.snapshot_table,
            window_days=window_days,
            filter_clauses=filter_clauses,
        )
        logger.info(f"analytics_snapshot_query_fired request_id={request_id} window_days={window_days} filters={filter_clauses!r}")
        try:
            execution = await self._executor.execute(sql)
        except Exception as e:
            logger.warning(f"analytics_snapshot_execute_error request_id={request_id} error_type={type(e).__name__} error={e}")
            return None
        total_ms = (time.monotonic() - t0) * 1000.0
        logger.info(f"analytics_snapshot_result request_id={request_id} rows={execution.row_count} latency_ms={total_ms:.1f}")
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
            execution=execution,
            verifier=None,
            total_latency_ms=total_ms,
            analytics_substrate='clickhouse_snapshot',
        )
