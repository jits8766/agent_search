"""ClickHouse-backed explore rails (trending / ending-soon)."""
import math
import re
import time
from typing import Any, Dict, List, Optional, Sequence

from semantic_search.analytics.clickhouse_executor import ClickHouseExecutor

_MIN_HORIZON_OVERRIDE_SECONDS = 60.0
_LAST_HOUR_HORIZON_SECONDS = 3600.0
_LATEST_BID_COUNT_NORM = 100.0
_LATEST_BID_RECENCY_WEIGHT = 0.5
_LATEST_BID_COUNT_WEIGHT = 0.5
_SAFE_SQL_COLUMN = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
_SAFE_SQL_TOKEN = re.compile(r'^[A-Za-z0-9._-]+$')
from semantic_search.config.models import (  # noqa: E402
    EndingSoonRailConfig,
    ExploreClickHouseRailsConfig,
    HardFilterPushdownConfig,
    TrendingRailConfig,
)
from semantic_search.contracts import ExploreCard  # noqa: E402
from semantic_search.core.exceptions import ValidationError  # noqa: E402
from semantic_search.core.logging_utils import get_logger  # noqa: E402
from semantic_search.core.validation import safe_float  # noqa: E402

logger = get_logger(__name__)


def _limited_sql(sql: str, cap: int) -> str:
    """Wrap arbitrary SELECT in an outer LIMIT without mutating the inner query."""
    inner = sql.strip().rstrip(';')
    if not inner:
        return ''
    return f"SELECT * FROM ({inner}) AS _explore_limited LIMIT {int(cap)}"  # noqa: S608


def _sql_quote_token(value: Any) -> Optional[str]:
    """Return a single-quoted SQL string literal, or None when unsafe."""
    token = str(value).strip()
    if not token or not _SAFE_SQL_TOKEN.match(token):
        return None
    return f"'{token}'"


def _sql_quote_number(value: Any) -> Optional[str]:
    """Return a finite numeric literal, or None when unparseable."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    if num.is_integer():
        return str(int(num))
    return repr(num)


def apply_hard_filter_pushdown(
    sql: str,
    filters: Dict[str, Any],
    pushdown: Optional[HardFilterPushdownConfig],
) -> str:
    """Wrap ``sql`` with outer WHERE clauses from config-driven hard-filter pushdown.

    Column names and ops come only from ``pushdown.slots`` (YAML). Soft slots must
    not appear in that map. Unsafe values are skipped (fail closed for that clause).
    """
    if pushdown is None or not pushdown.enabled or not filters or not sql.strip():
        return sql
    clauses: List[str] = []
    for slot, spec in pushdown.slots.items():
        if slot not in filters:
            continue
        if not _SAFE_SQL_COLUMN.match(spec.column):
            logger.warning(
                f"explore_hard_filter_pushdown_skipped slot={slot} "
                f"reason=unsafe_column column={spec.column!r}"
            )
            continue
        raw = filters[slot]
        if spec.op == 'in':
            values: Sequence[Any] = raw if isinstance(raw, (list, tuple, set)) else [raw]
            literals = [_sql_quote_token(v) for v in values]
            safe = [lit for lit in literals if lit is not None]
            if not safe:
                continue
            clauses.append(f"{spec.column} IN ({', '.join(safe)})")
        elif spec.op in ('gte', 'lte'):
            lit = _sql_quote_number(raw)
            if lit is None:
                continue
            op_sql = '>=' if spec.op == 'gte' else '<='
            clauses.append(f"{spec.column} {op_sql} {lit}")
    if not clauses:
        return sql
    inner = sql.strip().rstrip(';')
    where = ' AND '.join(clauses)
    return f"SELECT * FROM ({inner}) AS _explore_hf_push WHERE {where}"  # noqa: S608


def _rail_sql(
    sql: str,
    cap: int,
    rails: ExploreClickHouseRailsConfig,
    hard_filters: Optional[Dict[str, Any]] = None,
) -> str:
    """Apply optional hard-filter pushdown then LIMIT wrap."""
    pushed = apply_hard_filter_pushdown(sql, hard_filters or {}, rails.hard_filter_pushdown)
    return _limited_sql(pushed, cap)


class ClickHouseTrendingExploreSource:
    """Trending rail backed by configured ClickHouse SQL.

    :param trending: TrendingRailConfig - In-rail caps + enable flag
    :param rails_ch: ExploreClickHouseRailsConfig - SQL + row cap
    :param ch_executor: ClickHouseExecutor - Shared analytics executor
    """

    def __init__(self, trending: TrendingRailConfig, rails_ch: ExploreClickHouseRailsConfig, ch_executor: ClickHouseExecutor) -> None:
        if trending is None:
            raise ValidationError("ClickHouseTrendingExploreSource requires TrendingRailConfig")
        if rails_ch is None:
            raise ValidationError("ClickHouseTrendingExploreSource requires ExploreClickHouseRailsConfig")
        if ch_executor is None:
            raise ValidationError("ClickHouseTrendingExploreSource requires ClickHouseExecutor")
        self._trending = trending
        self._rails = rails_ch
        self._ch = ch_executor

    @property
    def rail_id(self) -> str:
        return 'trending'

    async def fetch(
        self,
        user_id: Optional[str],
        max_items: int,
        hard_filters: Optional[Dict[str, Any]] = None,
    ) -> List[ExploreCard]:
        if int(max_items) < 1:
            return []
        if not self._trending.source.enabled or not self._rails.enabled:
            return []
        if not self._ch.credentials_available:
            logger.warning(f"explore_ch_rail_skipped rail={self.rail_id} reason=credentials_unavailable")
            return []
        cap = min(int(max_items), int(self._trending.source.max_items), int(self._rails.max_rows_each))
        wrapped = _rail_sql(self._rails.trending_sql, cap, self._rails, hard_filters)
        if not wrapped:
            return []
        try:
            execution = await self._ch.execute(wrapped)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"explore_ch_trending_failed error_type={type(e).__name__} error={str(e)}")
            return []
        cards: List[ExploreCard] = []
        for row in execution.rows:
            if not isinstance(row, dict):
                continue
            raw_id = row.get('item_id')
            if raw_id is None or str(raw_id).strip() == '':
                continue
            score = safe_float(row.get('trending_score'), 0.0)
            payload: Dict[str, Any] = {k: v for k, v in row.items() if k not in ('item_id', 'trending_score')}
            payload['trending_score'] = float(score)
            cards.append(ExploreCard(item_id=str(raw_id), fused_score=float(score), source_rail='trending', payload=payload))
        return cards


class ClickHouseEndingSoonExploreSource:
    """Ending-soon rail backed by configured ClickHouse SQL.

    :param ending_soon: EndingSoonRailConfig - Horizon + caps
    :param rails_ch: ExploreClickHouseRailsConfig - SQL + row cap
    :param ch_executor: ClickHouseExecutor - Shared analytics executor
    """

    def __init__(self, ending_soon: EndingSoonRailConfig, rails_ch: ExploreClickHouseRailsConfig, ch_executor: ClickHouseExecutor) -> None:
        if ending_soon is None:
            raise ValidationError("ClickHouseEndingSoonExploreSource requires EndingSoonRailConfig")
        if rails_ch is None:
            raise ValidationError("ClickHouseEndingSoonExploreSource requires ExploreClickHouseRailsConfig")
        if ch_executor is None:
            raise ValidationError("ClickHouseEndingSoonExploreSource requires ClickHouseExecutor")
        self._ending = ending_soon
        self._rails = rails_ch
        self._ch = ch_executor

    @property
    def rail_id(self) -> str:
        return 'ending_soon'

    async def fetch(
        self,
        user_id: Optional[str],
        max_items: int,
        horizon_override: Optional[float] = None,
        hard_filters: Optional[Dict[str, Any]] = None,
    ) -> List[ExploreCard]:
        if int(max_items) < 1:
            return []
        if not self._ending.source.enabled or not self._rails.enabled:
            return []
        if not self._ch.credentials_available:
            logger.warning(f"explore_ch_rail_skipped rail={self.rail_id} reason=credentials_unavailable")
            return []
        cap = min(int(max_items), int(self._ending.source.max_items), int(self._rails.max_rows_each))
        wrapped = _rail_sql(self._rails.ending_soon_sql, cap, self._rails, hard_filters)
        if not wrapped:
            return []
        try:
            execution = await self._ch.execute(wrapped)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"explore_ch_ending_soon_failed error_type={type(e).__name__} error={str(e)}")
            return []
        effective_horizon = (
            min(float(horizon_override), float(self._ending.max_horizon_seconds))
            if horizon_override is not None and float(horizon_override) >= _MIN_HORIZON_OVERRIDE_SECONDS
            else float(self._ending.horizon_seconds)
        )
        cards: List[ExploreCard] = []
        for row in execution.rows:
            if not isinstance(row, dict):
                continue
            raw_id = row.get('item_id')
            if raw_id is None or str(raw_id).strip() == '':
                continue
            ends_at = safe_float(row.get('ends_at'), -1.0)
            if ends_at < 0.0:
                continue
            payload: Dict[str, Any] = {k: v for k, v in row.items() if k not in ('item_id', 'ends_at')}
            payload['ends_at'] = float(ends_at)
            seconds_until = max(0.0, float(ends_at) - time.time())
            if seconds_until > effective_horizon:
                continue
            payload['seconds_until_end'] = seconds_until
            urgency = 1.0 - (seconds_until / effective_horizon) if effective_horizon > 0.0 else 0.0
            urgency = max(0.0, min(1.0, urgency))
            cards.append(ExploreCard(item_id=str(raw_id), fused_score=float(urgency), source_rail='ending_soon', payload=payload))
        cards.sort(key=lambda c: (safe_float(c.payload.get('ends_at'), 0.0), c.item_id))
        return cards


class ClickHouseLastHourExploreSource:
    """Domains ending in the next 1 hour — highest-urgency rail for the zero-result/timeout fallback."""

    def __init__(self, rails_ch: ExploreClickHouseRailsConfig, ch_executor: ClickHouseExecutor) -> None:
        if rails_ch is None:
            raise ValidationError("ClickHouseLastHourExploreSource requires ExploreClickHouseRailsConfig")
        if ch_executor is None:
            raise ValidationError("ClickHouseLastHourExploreSource requires ClickHouseExecutor")
        self._rails = rails_ch
        self._ch = ch_executor

    @property
    def rail_id(self) -> str:
        return 'last_hour'

    async def fetch(
        self,
        user_id: Optional[str],
        max_items: int,
        hard_filters: Optional[Dict[str, Any]] = None,
    ) -> List[ExploreCard]:
        if int(max_items) < 1:
            return []
        if not self._rails.enabled or not self._rails.last_hour_sql.strip():
            return []
        if not self._ch.credentials_available:
            logger.warning(f"explore_ch_rail_skipped rail={self.rail_id} reason=credentials_unavailable")
            return []
        cap = min(int(max_items), int(self._rails.max_rows_each))
        wrapped = _rail_sql(self._rails.last_hour_sql, cap, self._rails, hard_filters)
        if not wrapped:
            return []
        try:
            execution = await self._ch.execute(wrapped)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"explore_ch_last_hour_failed error_type={type(e).__name__} error={str(e)}")
            return []
        horizon = _LAST_HOUR_HORIZON_SECONDS
        cards: List[ExploreCard] = []
        for row in execution.rows:
            if not isinstance(row, dict):
                continue
            raw_id = row.get('item_id')
            if raw_id is None or str(raw_id).strip() == '':
                continue
            ends_at = safe_float(row.get('ends_at'), -1.0)
            if ends_at < 0.0:
                continue
            payload: Dict[str, Any] = {k: v for k, v in row.items() if k not in ('item_id', 'ends_at')}
            payload['ends_at'] = float(ends_at)
            seconds_until = max(0.0, float(ends_at) - time.time())
            payload['seconds_until_end'] = seconds_until
            urgency = 1.0 - (seconds_until / horizon) if horizon > 0.0 else 1.0
            urgency = max(0.0, min(1.0, urgency))
            cards.append(ExploreCard(item_id=str(raw_id), fused_score=float(urgency), source_rail='last_hour', payload=payload))
        cards.sort(key=lambda c: safe_float(c.payload.get('ends_at'), 0.0))
        return cards


class ClickHouseLatestExploreSource:
    """Domains expiring in the next 7 days ordered by newest expiry — broad recency rail."""

    def __init__(self, rails_ch: ExploreClickHouseRailsConfig, ch_executor: ClickHouseExecutor) -> None:
        if rails_ch is None:
            raise ValidationError("ClickHouseLatestExploreSource requires ExploreClickHouseRailsConfig")
        if ch_executor is None:
            raise ValidationError("ClickHouseLatestExploreSource requires ClickHouseExecutor")
        self._rails = rails_ch
        self._ch = ch_executor

    @property
    def rail_id(self) -> str:
        return 'latest'

    async def fetch(
        self,
        user_id: Optional[str],
        max_items: int,
        hard_filters: Optional[Dict[str, Any]] = None,
    ) -> List[ExploreCard]:
        if int(max_items) < 1:
            return []
        if not self._rails.enabled or not self._rails.latest_sql.strip():
            return []
        if not self._ch.credentials_available:
            logger.warning(f"explore_ch_rail_skipped rail={self.rail_id} reason=credentials_unavailable")
            return []
        cap = min(int(max_items), int(self._rails.max_rows_each))
        wrapped = _rail_sql(self._rails.latest_sql, cap, self._rails, hard_filters)
        if not wrapped:
            return []
        try:
            execution = await self._ch.execute(wrapped)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"explore_ch_latest_failed error_type={type(e).__name__} error={str(e)}")
            return []
        cards: List[ExploreCard] = []
        for row in execution.rows:
            if not isinstance(row, dict):
                continue
            raw_id = row.get('item_id')
            if raw_id is None or str(raw_id).strip() == '':
                continue
            ends_at = safe_float(row.get('ends_at'), -1.0)
            if ends_at < 0.0:
                continue
            payload: Dict[str, Any] = {k: v for k, v in row.items() if k not in ('item_id', 'ends_at')}
            payload['ends_at'] = float(ends_at)
            bid_count_score = safe_float(row.get('bid_count'), 0.0)
            last_bid_dt = safe_float(row.get('last_bid_offer_dt'), -1.0)
            if last_bid_dt > 0.0:
                horizon = float(self._rails.bid_recency_horizon_seconds)
                time_since = max(0.0, time.time() - last_bid_dt)
                recency = max(0.0, 1.0 - (time_since / horizon)) if horizon > 0.0 else 0.0
                bid_count_norm = min(1.0, bid_count_score / _LATEST_BID_COUNT_NORM)
                score = _LATEST_BID_RECENCY_WEIGHT * recency + _LATEST_BID_COUNT_WEIGHT * bid_count_norm
            else:
                score = bid_count_score
            cards.append(ExploreCard(item_id=str(raw_id), fused_score=float(score), source_rail='latest', payload=payload))
        return cards


class ClickHouseHighVolumeExploreSource:
    """High-bid-activity rail — domains with >= 5 bids, ordered by bid_count DESC."""

    def __init__(self, rails_ch: ExploreClickHouseRailsConfig, ch_executor: ClickHouseExecutor) -> None:
        if rails_ch is None:
            raise ValidationError("ClickHouseHighVolumeExploreSource requires ExploreClickHouseRailsConfig")
        if ch_executor is None:
            raise ValidationError("ClickHouseHighVolumeExploreSource requires ClickHouseExecutor")
        self._rails = rails_ch
        self._ch = ch_executor

    @property
    def rail_id(self) -> str:
        return 'high_volume'

    async def fetch(
        self,
        user_id: Optional[str],
        max_items: int,
        hard_filters: Optional[Dict[str, Any]] = None,
    ) -> List[ExploreCard]:
        """Fetch high-bid-volume domains scored by normalised bid_count.
        :param user_id: Optional[str] - User id (unused; present for protocol compliance)
        :param max_items: int - Maximum cards to return
        :param hard_filters: Optional[Dict[str, Any]] - Identified hard filters for SQL pushdown
        :return: List[ExploreCard] - Cards scored in [0, 1] by bid volume
        """
        if int(max_items) < 1:
            return []
        if not self._rails.enabled or not self._rails.high_volume_sql.strip():
            return []
        if not self._ch.credentials_available:
            logger.warning(f"explore_ch_rail_skipped rail={self.rail_id} reason=credentials_unavailable")
            return []
        cap = min(int(max_items), int(self._rails.max_rows_each))
        wrapped = _rail_sql(self._rails.high_volume_sql, cap, self._rails, hard_filters)
        if not wrapped:
            return []
        try:
            execution = await self._ch.execute(wrapped)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"explore_ch_high_volume_failed error_type={type(e).__name__} error={str(e)}")
            return []
        norm = float(self._rails.high_volume_score_norm)
        cards: List[ExploreCard] = []
        for row in execution.rows:
            if not isinstance(row, dict):
                continue
            raw_id = row.get('item_id')
            if raw_id is None or str(raw_id).strip() == '':
                continue
            volume_score = safe_float(row.get('volume_score', row.get('bid_count', 0.0)), 0.0)
            score = min(1.0, volume_score / norm) if volume_score > 0.0 else 0.0
            payload: Dict[str, Any] = {k: v for k, v in row.items() if k not in ('item_id', 'volume_score')}
            payload['volume_score'] = float(volume_score)
            cards.append(ExploreCard(item_id=str(raw_id), fused_score=float(score), source_rail='high_volume', payload=payload))
        return cards


class ClickHouseFreshExploreSource:
    """Zero-bid fresh-arrival rail — domains with no bids yet, scored by time remaining."""

    def __init__(self, rails_ch: ExploreClickHouseRailsConfig, ch_executor: ClickHouseExecutor) -> None:
        if rails_ch is None:
            raise ValidationError("ClickHouseFreshExploreSource requires ExploreClickHouseRailsConfig")
        if ch_executor is None:
            raise ValidationError("ClickHouseFreshExploreSource requires ClickHouseExecutor")
        self._rails = rails_ch
        self._ch = ch_executor

    @property
    def rail_id(self) -> str:
        return 'fresh'

    async def fetch(
        self,
        user_id: Optional[str],
        max_items: int,
        hard_filters: Optional[Dict[str, Any]] = None,
    ) -> List[ExploreCard]:
        """Fetch zero-bid domains scored by remaining time normalised to fresh_time_horizon_seconds.
        :param user_id: Optional[str] - User id (unused; present for protocol compliance)
        :param max_items: int - Maximum cards to return
        :param hard_filters: Optional[Dict[str, Any]] - Identified hard filters for SQL pushdown
        :return: List[ExploreCard] - Cards scored in [0, 1] by recency
        """
        if int(max_items) < 1:
            return []
        if not self._rails.enabled or not self._rails.fresh_sql.strip():
            return []
        if not self._ch.credentials_available:
            logger.warning(f"explore_ch_rail_skipped rail={self.rail_id} reason=credentials_unavailable")
            return []
        cap = min(int(max_items), int(self._rails.max_rows_each))
        wrapped = _rail_sql(self._rails.fresh_sql, cap, self._rails, hard_filters)
        if not wrapped:
            return []
        try:
            execution = await self._ch.execute(wrapped)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"explore_ch_fresh_failed error_type={type(e).__name__} error={str(e)}")
            return []
        horizon = float(self._rails.fresh_time_horizon_seconds)
        cards: List[ExploreCard] = []
        for row in execution.rows:
            if not isinstance(row, dict):
                continue
            raw_id = row.get('item_id')
            if raw_id is None or str(raw_id).strip() == '':
                continue
            ends_at = safe_float(row.get('ends_at'), -1.0)
            if ends_at < 0.0:
                continue
            payload: Dict[str, Any] = {k: v for k, v in row.items() if k not in ('item_id', 'ends_at')}
            payload['ends_at'] = float(ends_at)
            seconds_remaining = max(0.0, float(ends_at) - time.time())
            score = min(1.0, seconds_remaining / horizon) if horizon > 0.0 else 0.5
            cards.append(ExploreCard(item_id=str(raw_id), fused_score=float(score), source_rail='fresh', payload=payload))
        return cards


class ClickHouseLastWeekExploreSource:
    """Long-duration auction rail — domains with > 7 days remaining (just started), ordered by ends_at ASC."""

    def __init__(self, rails_ch: ExploreClickHouseRailsConfig, ch_executor: ClickHouseExecutor) -> None:
        if rails_ch is None:
            raise ValidationError("ClickHouseLastWeekExploreSource requires ExploreClickHouseRailsConfig")
        if ch_executor is None:
            raise ValidationError("ClickHouseLastWeekExploreSource requires ClickHouseExecutor")
        self._rails = rails_ch
        self._ch = ch_executor

    @property
    def rail_id(self) -> str:
        return 'last_week'

    async def fetch(
        self,
        user_id: Optional[str],
        max_items: int,
        hard_filters: Optional[Dict[str, Any]] = None,
    ) -> List[ExploreCard]:
        """Fetch long-duration auctions scored by inverse time-remaining (lower remaining = higher urgency).
        :param user_id: Optional[str] - User id (unused; present for protocol compliance)
        :param max_items: int - Maximum cards to return
        :param hard_filters: Optional[Dict[str, Any]] - Identified hard filters for SQL pushdown
        :return: List[ExploreCard] - Cards scored in [0, 1]
        """
        if int(max_items) < 1:
            return []
        if not self._rails.enabled or not self._rails.last_week_sql.strip():
            return []
        if not self._ch.credentials_available:
            logger.warning(f"explore_ch_rail_skipped rail={self.rail_id} reason=credentials_unavailable")
            return []
        cap = min(int(max_items), int(self._rails.max_rows_each))
        wrapped = _rail_sql(self._rails.last_week_sql, cap, self._rails, hard_filters)
        if not wrapped:
            return []
        try:
            execution = await self._ch.execute(wrapped)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"explore_ch_last_week_failed error_type={type(e).__name__} error={str(e)}")
            return []
        cards: List[ExploreCard] = []
        for row in execution.rows:
            if not isinstance(row, dict):
                continue
            raw_id = row.get('item_id')
            if raw_id is None or str(raw_id).strip() == '':
                continue
            ends_at = safe_float(row.get('ends_at'), -1.0)
            if ends_at < 0.0:
                continue
            payload: Dict[str, Any] = {k: v for k, v in row.items() if k not in ('item_id', 'ends_at')}
            payload['ends_at'] = float(ends_at)
            bid_count_score = safe_float(row.get('bid_count'), 0.0)
            score = min(1.0, bid_count_score / float(self._rails.high_volume_score_norm))
            cards.append(ExploreCard(item_id=str(raw_id), fused_score=float(score), source_rail='last_week', payload=payload))
        return cards


class ClickHouseWatchDensityExploreSource:
    """Passive-audience rail — domains ordered by today's watcher count."""

    def __init__(self, rails_ch: ExploreClickHouseRailsConfig, ch_executor: ClickHouseExecutor) -> None:
        if rails_ch is None:
            raise ValidationError("ClickHouseWatchDensityExploreSource requires ExploreClickHouseRailsConfig")
        if ch_executor is None:
            raise ValidationError("ClickHouseWatchDensityExploreSource requires ClickHouseExecutor")
        self._rails = rails_ch
        self._ch = ch_executor

    @property
    def rail_id(self) -> str:
        return 'watch_density'

    async def fetch(
        self,
        user_id: Optional[str],
        max_items: int,
        hard_filters: Optional[Dict[str, Any]] = None,
    ) -> List[ExploreCard]:
        if int(max_items) < 1:
            return []
        if not self._rails.enabled or not self._rails.watch_density_sql.strip():
            return []
        if not self._ch.credentials_available:
            logger.warning(f"explore_ch_rail_skipped rail={self.rail_id} reason=credentials_unavailable")
            return []
        cap = min(int(max_items), int(self._rails.max_rows_each))
        wrapped = _rail_sql(self._rails.watch_density_sql, cap, self._rails, hard_filters)
        if not wrapped:
            return []
        try:
            execution = await self._ch.execute(wrapped)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"explore_ch_watch_density_failed error_type={type(e).__name__} error={str(e)}")
            return []
        norm = float(self._rails.watch_density_score_norm)
        cards: List[ExploreCard] = []
        for row in execution.rows:
            if not isinstance(row, dict):
                continue
            raw_id = row.get('item_id')
            if raw_id is None or str(raw_id).strip() == '':
                continue
            watch_score = safe_float(row.get('watch_score', 0.0), 0.0)
            score = min(1.0, watch_score / norm) if watch_score > 0.0 and norm > 0.0 else 0.0
            payload: Dict[str, Any] = {k: v for k, v in row.items() if k not in ('item_id', 'watch_score')}
            payload['watch_score'] = float(watch_score)
            cards.append(ExploreCard(item_id=str(raw_id), fused_score=float(score), source_rail='watch_density', payload=payload))
        return cards


class ClickHouseHighTrafficExploreSource:
    """Combined bid-velocity + watch-density rail scored by config-weighted traffic signal."""

    def __init__(self, rails_ch: ExploreClickHouseRailsConfig, ch_executor: ClickHouseExecutor) -> None:
        if rails_ch is None:
            raise ValidationError("ClickHouseHighTrafficExploreSource requires ExploreClickHouseRailsConfig")
        if ch_executor is None:
            raise ValidationError("ClickHouseHighTrafficExploreSource requires ClickHouseExecutor")
        self._rails = rails_ch
        self._ch = ch_executor

    @property
    def rail_id(self) -> str:
        return 'high_traffic'

    async def fetch(
        self,
        user_id: Optional[str],
        max_items: int,
        hard_filters: Optional[Dict[str, Any]] = None,
    ) -> List[ExploreCard]:
        if int(max_items) < 1:
            return []
        if not self._rails.enabled or not self._rails.high_traffic_sql.strip():
            return []
        if not self._ch.credentials_available:
            logger.warning(f"explore_ch_rail_skipped rail={self.rail_id} reason=credentials_unavailable")
            return []
        cap = min(int(max_items), int(self._rails.max_rows_each))
        wrapped = _rail_sql(self._rails.high_traffic_sql, cap, self._rails, hard_filters)
        if not wrapped:
            return []
        try:
            execution = await self._ch.execute(wrapped)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"explore_ch_high_traffic_failed error_type={type(e).__name__} error={str(e)}")
            return []
        norm = float(self._rails.high_traffic_score_norm)
        cards: List[ExploreCard] = []
        for row in execution.rows:
            if not isinstance(row, dict):
                continue
            raw_id = row.get('item_id')
            if raw_id is None or str(raw_id).strip() == '':
                continue
            traffic_score = safe_float(row.get('traffic_score', 0.0), 0.0)
            score = min(1.0, traffic_score / norm) if traffic_score > 0.0 and norm > 0.0 else 0.0
            payload: Dict[str, Any] = {k: v for k, v in row.items() if k not in ('item_id', 'traffic_score')}
            payload['traffic_score'] = float(traffic_score)
            cards.append(ExploreCard(item_id=str(raw_id), fused_score=float(score), source_rail='high_traffic', payload=payload))
        return cards


__all__ = [
    'ClickHouseTrendingExploreSource',
    'ClickHouseEndingSoonExploreSource',
    'ClickHouseLastHourExploreSource',
    'ClickHouseLatestExploreSource',
    'ClickHouseHighVolumeExploreSource',
    'ClickHouseFreshExploreSource',
    'ClickHouseLastWeekExploreSource',
    'ClickHouseWatchDensityExploreSource',
    'ClickHouseHighTrafficExploreSource',
    'apply_hard_filter_pushdown',
]
