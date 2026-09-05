"""Bounded TTL+LRU cache primitive.
Shared by every cache tier so eviction logic lives in one place. Insertion order
is preserved via `OrderedDict`; reads bump items to the most-recent slot. Items
past their TTL are evicted on access.
"""
import random
import sys
import time
from collections import OrderedDict
from typing import Generic, Optional, Tuple, TypeVar

from semantic_search.core.exceptions import CacheError

V = TypeVar('V')


class LRUTTLCache(Generic[V]):
    """Bounded LRU + per-item TTL + soft byte cap + TTL jitter (thundering-herd guard)."""

    def __init__(self, max_entries: int, ttl_seconds: int, *, max_bytes: Optional[int] = None, ttl_jitter_seconds: int = 0):
        if max_entries < 1:
            raise CacheError("LRUTTLCache.max_entries must be >= 1")
        if ttl_seconds < 1:
            raise CacheError("LRUTTLCache.ttl_seconds must be >= 1")
        if max_bytes is not None and max_bytes < 1:
            raise CacheError("LRUTTLCache.max_bytes must be >= 1 when set")
        if ttl_jitter_seconds < 0:
            raise CacheError("LRUTTLCache.ttl_jitter_seconds must be >= 0")
        self._max_entries = int(max_entries)
        self._ttl_seconds = int(ttl_seconds)
        self._max_bytes = max_bytes
        self._ttl_jitter_seconds = int(ttl_jitter_seconds)
        self._items: 'OrderedDict[str, Tuple[V, float]]' = OrderedDict()
        # Byte estimates (populated only when max_bytes set)
        self._entry_bytes: 'OrderedDict[str, int]' = OrderedDict()
        self._current_bytes: int = 0
        self._hits = 0
        self._misses = 0

    @property
    def hits(self) -> int:
        """Total successful lookups."""
        return self._hits

    @property
    def misses(self) -> int:
        """Total misses / expired-on-access."""
        return self._misses

    @property
    def current_bytes(self) -> int:
        """Shallow byte estimate (0 if max_bytes unset)."""
        return self._current_bytes

    def __len__(self) -> int:
        return len(self._items)

    def get(self, key: str) -> Optional[V]:
        """Get value if present and not expired, else None (bump on hit)."""
        if key not in self._items:
            self._misses += 1
            return None
        value, expires_at = self._items[key]
        if time.time() >= expires_at:
            del self._items[key]
            if self._max_bytes is not None and key in self._entry_bytes:
                self._current_bytes -= self._entry_bytes.pop(key)
            self._misses += 1
            return None
        self._items.move_to_end(key)
        self._hits += 1
        return value

    def put(self, key: str, value: V) -> None:
        """Insert / replace the value for `key` and evict LRU if over capacity."""
        if not key:
            raise CacheError("LRUTTLCache.put requires non-empty key")
        jitter = random.uniform(0, self._ttl_jitter_seconds) if self._ttl_jitter_seconds > 0 else 0.0  # nosec B311
        expires_at = time.time() + self._ttl_seconds + jitter
        # If replacing an existing key, remove its old byte estimate first.
        if self._max_bytes is not None and key in self._entry_bytes:
            self._current_bytes -= self._entry_bytes.pop(key)
        if key in self._items:
            self._items.move_to_end(key)
        self._items[key] = (value, expires_at)
        if self._max_bytes is not None:
            entry_size = sys.getsizeof(value)
            self._entry_bytes[key] = entry_size
            self._current_bytes += entry_size
        # Evict LRU entries until both entry-count and byte caps are satisfied.
        while len(self._items) > self._max_entries or (
            self._max_bytes is not None and self._current_bytes > self._max_bytes and len(self._items) > 1
        ):
            evicted_key, _ = self._items.popitem(last=False)
            if self._max_bytes is not None and evicted_key in self._entry_bytes:
                self._current_bytes -= self._entry_bytes.pop(evicted_key)

    def items_snapshot(self):
        """Yield (key, value) for unexpired entries — used by sibling caches.
        :yields: Tuple[str, V]
        """
        now = time.time()
        for key, (value, expires_at) in list(self._items.items()):
            if now >= expires_at:
                continue
            yield key, value

    def invalidate(self, key: str) -> bool:
        """Drop a single entry by key. Counters are not adjusted (the read
        that detected the staleness already increments hits/misses as it
        sees fit).
        :param key: str - The cache key to drop
        :return: bool - True when the key was present and removed, False when absent
        """
        if not key:
            return False
        if key in self._items:
            del self._items[key]
            if self._max_bytes is not None and key in self._entry_bytes:
                self._current_bytes -= self._entry_bytes.pop(key)
            return True
        return False

    def clear(self) -> None:
        """Drop all entries (resets hit/miss counters and byte tracking)."""
        self._items.clear()
        self._entry_bytes.clear()
        self._current_bytes = 0
        self._hits = 0
        self._misses = 0
