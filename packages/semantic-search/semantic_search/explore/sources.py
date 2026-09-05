"""Explore-rail data sources.

Each rail is fed by a typed source. In-memory implementations ship for tests +
local bootstrapping; production swaps in:
  - TrendingSource     -> bid-velocity Materialized View
  - EndingSoonSource   -> ClickHouse `ending_soon_*` MV

The protocol is intentionally tight: each source returns a list of
``ExploreCard`` for a given (user, max_items) pair; the composer is responsible
for ordering the rails and dedup across them.
"""
import threading
import time
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from semantic_search.config.models import EndingSoonRailConfig, TrendingRailConfig
from semantic_search.contracts import ExploreCard
from semantic_search.core.exceptions import ValidationError
from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)

_MIN_HORIZON_OVERRIDE_SECONDS = 60.0


@runtime_checkable
class ExploreSource(Protocol):
    """Common protocol every explore source implements.

    Implementations MUST be safe to call without a `user_id` (trending and
    ending-soon rails are user-agnostic).
    """

    @property
    def rail_id(self) -> str:
        """One of EXPLORE_RAIL_KINDS."""
        ...

    async def fetch(self, user_id: Optional[str], max_items: int) -> List[ExploreCard]:  # pragma: no cover - protocol
        """Return up to `max_items` cards for this rail.

        :param user_id: Optional[str] - Authenticated user id (None for anonymous)
        :param max_items: int - Hard cap (>= 1)
        :return: List[ExploreCard] - Cards in rail-native order; may be []
        """
        ...


@runtime_checkable
class TrendingSource(ExploreSource, Protocol):
    """Trending rail source — bid-velocity / engagement signal."""
    pass


@runtime_checkable
class EndingSoonSource(ExploreSource, Protocol):
    """Ending-soon rail source — auctions ending inside the configured horizon."""

    async def fetch(self, user_id: Optional[str], max_items: int, horizon_override: Optional[float] = None) -> List[ExploreCard]:  # pragma: no cover - protocol
        """Return up to `max_items` ending-soon cards.

        :param user_id: Optional[str] - Authenticated user id (None for anonymous)
        :param max_items: int - Hard cap (>= 1)
        :param horizon_override: Optional[float] - When set, overrides the config horizon_seconds (seconds from now).
            Capped at config.max_horizon_seconds. Values < 60 are ignored (config default used).
        :return: List[ExploreCard] - Cards ordered by ends_at ascending; may be []
        """
        ...


def _now_ts() -> float:
    """Wall-clock reader; isolated so tests can monkeypatch deterministically."""
    return time.time()


class InMemoryTrendingSource:
    """In-memory trending source for tests + local bootstrap.

    Items added via ``add()`` carry a ``created_at`` timestamp. ``fetch()`` keeps
    items inside the configured trending window and returns the top
    ``max_items`` ranked by their ``trending_score`` payload field
    (descending). Items without a ``trending_score`` fall to the back.
    """

    def __init__(self, config: TrendingRailConfig):
        if config is None:
            raise ValidationError("InMemoryTrendingSource requires a TrendingRailConfig")
        self._config = config
        self._lock = threading.Lock()
        self._items: List[Dict[str, Any]] = []

    @property
    def rail_id(self) -> str:
        return 'trending'

    def add(self, item_id: str, trending_score: float, payload: Optional[Dict[str, Any]] = None, created_at: Optional[float] = None) -> None:
        """Add a candidate to the trending pool.

        :param item_id: str - Stable item id
        :param trending_score: float - Per-item trending strength (>= 0)
        :param payload: Optional[Dict[str, Any]] - Additional payload merged into ExploreCard.payload
        :param created_at: Optional[float] - Unix timestamp (default: now)
        :raises ValidationError: When item_id is empty or trending_score is negative
        """
        if not item_id:
            raise ValidationError("InMemoryTrendingSource.add requires non-empty item_id")
        if float(trending_score) < 0.0:
            raise ValidationError("InMemoryTrendingSource.add trending_score must be >= 0")
        ts = float(created_at) if created_at is not None else _now_ts()
        record = {'item_id': str(item_id), 'trending_score': float(trending_score), 'created_at': ts, 'payload': dict(payload or {})}
        with self._lock:
            self._items.append(record)

    def clear(self) -> None:
        """Drop every item (test helper)."""
        with self._lock:
            self._items.clear()

    async def fetch(self, user_id: Optional[str], max_items: int) -> List[ExploreCard]:
        if int(max_items) < 1:
            return []
        if not self._config.source.enabled:
            return []
        cap = min(int(max_items), int(self._config.source.max_items))
        cutoff = _now_ts() - float(self._config.window_seconds)
        with self._lock:
            in_window = [r for r in self._items if r['created_at'] >= cutoff]
        in_window.sort(key=lambda r: (-float(r['trending_score']), r['item_id']))
        cards: List[ExploreCard] = []
        for r in in_window[:cap]:
            payload = dict(r['payload'])
            payload['trending_score'] = float(r['trending_score'])
            payload['rail_added_at'] = float(r['created_at'])
            cards.append(ExploreCard(item_id=r['item_id'], fused_score=float(r['trending_score']), source_rail='trending', payload=payload))
        return cards

    async def fetch_all(self, max_items: int) -> List[ExploreCard]:
        """Return top items by score ignoring the trending window.

        Used by the explore-fallback rail so the last-resort response is
        never empty just because items were indexed before the window cutoff.
        """
        if int(max_items) < 1:
            return []
        with self._lock:
            candidates = list(self._items)
        candidates.sort(key=lambda r: (-float(r['trending_score']), r['item_id']))
        cards: List[ExploreCard] = []
        for r in candidates[:max_items]:
            payload = dict(r['payload'])
            payload['trending_score'] = float(r['trending_score'])
            payload['rail_added_at'] = float(r['created_at'])
            cards.append(ExploreCard(item_id=r['item_id'], fused_score=float(r['trending_score']), source_rail='fallback', payload=payload))
        return cards


class InMemoryEndingSoonSource:
    """In-memory ending-soon source for tests + local bootstrap.

    Items added via ``add()`` carry an ``ends_at`` timestamp. ``fetch()`` keeps
    items whose ``ends_at`` falls inside ``[now, now + horizon_seconds]`` and
    returns the top ``max_items`` ordered by ``ends_at`` ascending (soonest
    first). Items already past their ``ends_at`` are excluded.
    """

    def __init__(self, config: EndingSoonRailConfig):
        if config is None:
            raise ValidationError("InMemoryEndingSoonSource requires an EndingSoonRailConfig")
        self._config = config
        self._lock = threading.Lock()
        self._items: List[Dict[str, Any]] = []

    @property
    def rail_id(self) -> str:
        return 'ending_soon'

    def add(self, item_id: str, ends_at: float, payload: Optional[Dict[str, Any]] = None) -> None:
        """Add a candidate to the ending-soon pool.

        :param item_id: str - Stable item id
        :param ends_at: float - Unix timestamp when the auction closes
        :param payload: Optional[Dict[str, Any]] - Additional payload merged into ExploreCard.payload
        :raises ValidationError: When item_id is empty
        """
        if not item_id:
            raise ValidationError("InMemoryEndingSoonSource.add requires non-empty item_id")
        record = {'item_id': str(item_id), 'ends_at': float(ends_at), 'payload': dict(payload or {})}
        with self._lock:
            self._items.append(record)

    def clear(self) -> None:
        """Drop every item (test helper)."""
        with self._lock:
            self._items.clear()

    async def fetch(self, user_id: Optional[str], max_items: int, horizon_override: Optional[float] = None) -> List[ExploreCard]:
        if int(max_items) < 1:
            return []
        if not self._config.source.enabled:
            return []
        cap = min(int(max_items), int(self._config.source.max_items))
        now = _now_ts()
        effective_horizon = min(float(horizon_override), float(self._config.max_horizon_seconds)) if horizon_override is not None and float(horizon_override) >= _MIN_HORIZON_OVERRIDE_SECONDS else float(self._config.horizon_seconds)
        horizon = now + effective_horizon
        with self._lock:
            in_window = [r for r in self._items if now <= r['ends_at'] <= horizon]
        in_window.sort(key=lambda r: (float(r['ends_at']), r['item_id']))
        cards: List[ExploreCard] = []
        for r in in_window[:cap]:
            payload = dict(r['payload'])
            payload['ends_at'] = float(r['ends_at'])
            seconds_until = max(0.0, float(r['ends_at']) - now)
            payload['seconds_until_end'] = seconds_until
            # fused_score is a normalized "urgency" — higher = ends sooner. We
            # compute it inside the source so the composer can dedup by max
            # score across rails without re-deriving urgency.
            urgency = 1.0 - (seconds_until / effective_horizon) if effective_horizon > 0.0 else 0.0
            urgency = max(0.0, min(1.0, urgency))
            cards.append(ExploreCard(item_id=r['item_id'], fused_score=float(urgency), source_rail='ending_soon', payload=payload))
        return cards


__all__ = [
    'ExploreSource',
    'TrendingSource',
    'EndingSoonSource',
    'InMemoryTrendingSource',
    'InMemoryEndingSoonSource',
]
