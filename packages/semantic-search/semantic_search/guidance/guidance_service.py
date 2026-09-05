"""ClickHouse-backed market snapshot for guidance-classified queries."""
import json
import re
import time
from typing import List, Optional

from semantic_search.analytics.clickhouse_executor import ClickHouseExecutor
from semantic_search.config.models import GuidanceConfig
from semantic_search.contracts import GuidanceEnvelope
from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)

# Regex to locate the WHERE clause end (before GROUP BY / ORDER BY / LIMIT).
# Used to inject a tld IN (...) predicate when the user specifies a TLD filter.
_WHERE_RE = re.compile(r'\bWHERE\b', re.IGNORECASE)
_GROUP_ORDER_LIMIT_RE = re.compile(r'\b(GROUP\s+BY|ORDER\s+BY|LIMIT)\b', re.IGNORECASE)


def _inject_tld_filter(sql: str, tld_values: List[str]) -> str:
    """Inject AND tld IN (...) into WHERE (or prepend WHERE if absent); strip leading dots."""
    clean = [v.lstrip('.').lower() for v in tld_values if v]
    if not clean:
        return sql
    quoted = ', '.join(f"'{t}'" for t in clean)
    predicate = f"tld IN ({quoted})"

    where_match = _WHERE_RE.search(sql)
    if where_match:
        # Find GROUP BY / ORDER BY / LIMIT after WHERE
        tail_match = _GROUP_ORDER_LIMIT_RE.search(sql, where_match.end())
        if tail_match:
            insert_pos = tail_match.start()
            return sql[:insert_pos] + f"AND {predicate} " + sql[insert_pos:]
        # No GROUP/ORDER/LIMIT after WHERE — append to end of SQL.
        return sql.rstrip() + f" AND {predicate}"

    # No WHERE clause at all — insert before the first GROUP/ORDER/LIMIT.
    tail_match = _GROUP_ORDER_LIMIT_RE.search(sql)
    if tail_match:
        insert_pos = tail_match.start()
        return sql[:insert_pos] + f"WHERE {predicate} " + sql[insert_pos:]
    return sql.rstrip() + f" WHERE {predicate}"


class GuidanceService:
    """Execute market_snapshot_sql + inject TLD filter (if provided) + build GuidanceEnvelope."""

    def __init__(self, config: GuidanceConfig, ch_executor: Optional[ClickHouseExecutor]) -> None:
        self._config = config
        self._ch = ch_executor

    async def build_snapshot(self, *, request_id: str, tld_filter: Optional[List[str]] = None) -> Optional[GuidanceEnvelope]:
        """Execute SQL (returns None if disabled, no CH, or empty SQL)."""
        if not self._config.enabled:
            logger.info(f"guidance_snapshot_skipped request_id={request_id} reason=guidance_disabled")
            return None
        if self._ch is None:
            logger.warning(f"guidance_snapshot_skipped request_id={request_id} reason=no_ch_executor")
            return None
        if not self._ch.credentials_available:
            logger.warning(f"guidance_snapshot_skipped request_id={request_id} reason=credentials_unavailable")
            return None
        sql = self._config.market_snapshot_sql.strip()
        if not sql:
            logger.warning(f"guidance_snapshot_skipped request_id={request_id} reason=empty_market_snapshot_sql")
            return None
        if tld_filter:
            sql = _inject_tld_filter(sql, tld_filter)
            logger.info(f"guidance_snapshot_tld_filter request_id={request_id} tlds={tld_filter}")
        try:
            execution = await self._ch.execute(sql)
        except Exception as e:
            logger.warning(f"guidance_snapshot_failed request_id={request_id} error_type={type(e).__name__} error={str(e)}")
            return None
        cap = int(self._config.max_payload_chars)
        body = json.dumps(execution.rows)[:cap]
        as_hot = time.time()
        return GuidanceEnvelope(headline='Market snapshot', body=body, substrate='clickhouse_hot', as_of_hot=as_hot)
