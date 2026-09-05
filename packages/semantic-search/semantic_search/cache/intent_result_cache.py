"""Tier-0 QI result cache: normalized_query → QueryIntent (in-memory LRU+TTL).

Auto-scales from warm-up capacity to a larger store when cumulative hits reach
`scale_at_hit_count`, proving the cache is effective before committing RAM.
Migration uses `LRUTTLCache.items_snapshot()` to carry live entries into the
new store without re-classifying them.

Thread safety: asyncio event loop is single-threaded; cache ops run in the
coroutine body (no await), so _maybe_scale() executes in one thread with no
interleaving. No locking is needed.
"""
from typing import Optional

from semantic_search.cache.lru_ttl import LRUTTLCache
from semantic_search.config.models import QIIntentResultCacheConfig
from semantic_search.contracts import QueryIntent
from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)


class QIIntentResultCache:
    """LRU+TTL store keyed by normalized_query string → QueryIntent.

    :param config: QIIntentResultCacheConfig - All sizing and TTL parameters
    """

    def __init__(self, config: QIIntentResultCacheConfig):
        self._config = config
        self._scaled = False
        self._store: LRUTTLCache[QueryIntent] = LRUTTLCache(
            max_entries=config.max_entries,
            ttl_seconds=config.ttl_seconds,
        )

    def get(self, normalized_query: str) -> Optional[QueryIntent]:
        """Return a cached QueryIntent for the normalized query, or None on miss.

        :param normalized_query: str - Lowercased, whitespace-collapsed query
        :return: Optional[QueryIntent] - Cached intent or None
        """
        if not self._config.enabled or not normalized_query:
            return None
        hit = self._store.get(normalized_query)
        if hit is not None:
            logger.info(f"qi_intent_cache_hit scaled={self._scaled} query_len={len(normalized_query)}")
            self._maybe_scale()
        return hit

    def put(self, normalized_query: str, intent: QueryIntent) -> None:
        """Store a QueryIntent keyed by normalized query.

        :param normalized_query: str - Cache key
        :param intent: QueryIntent - Classification result to store
        """
        if not self._config.enabled or not normalized_query:
            return
        self._store.put(normalized_query, intent)

    def _maybe_scale(self) -> None:
        """Promote to scaled capacity when cumulative hits cross the threshold."""
        if self._scaled:
            return
        if self._store.hits < self._config.scale_at_hit_count:
            return
        old_items = list(self._store.items_snapshot())
        self._store = LRUTTLCache(
            max_entries=self._config.max_entries_scaled,
            ttl_seconds=self._config.ttl_seconds_scaled,
        )
        for key, value in old_items:
            self._store.put(key, value)
        self._scaled = True
        logger.info(f"qi_intent_cache_scaled_up entries_migrated={len(old_items)} new_max_entries={self._config.max_entries_scaled} new_ttl_seconds={self._config.ttl_seconds_scaled}")

    @property
    def hits(self) -> int:
        """Total cache hits since last scale-up or construction."""
        return self._store.hits

    @property
    def misses(self) -> int:
        """Total cache misses since last scale-up or construction."""
        return self._store.misses

    def invalidate_all(self) -> int:
        """Drop every entry. Returns entries dropped.
        :return: int - Entries dropped (0 when already empty or disabled)
        """
        if not self._config.enabled:
            return 0
        dropped = len(self._store)
        self._store = LRUTTLCache(max_entries=self._config.max_entries, ttl_seconds=self._config.ttl_seconds)
        self._scaled = False
        if dropped > 0:
            logger.info(f"qi_intent_cache_invalidated entries_dropped={dropped}")
        return dropped

    @property
    def scaled(self) -> bool:
        """True once the cache has promoted to scaled capacity."""
        return self._scaled


__all__ = ['QIIntentResultCache']
