"""Tier 1 — exact-hash cache keyed by sha256(normalized_query)."""
from typing import Callable, Optional, Union

from semantic_search.cache.keys import exact_query_key
from semantic_search.cache.lru_ttl import LRUTTLCache
from semantic_search.config.models import ExactCacheConfig
from semantic_search.core.logging_utils import get_logger
from semantic_search.contracts import CachedSearchPayload, QueryIntent, RankedResults

logger = get_logger(__name__)


# Per-entry snapshot-version reader. Wired from the registry so the cache
# stays decoupled from ``SnapshotVersionRegistry`` while still being able
# to silently expire entries primed against an older snapshot (Wave 9
# Gap 13). When None, versioning is OFF — every payload entry is served
# regardless of its tag (preserves legacy callers + tests that don't wire
# the registry).
SnapshotVersionFn = Callable[[], int]


class ExactCache:
    """Stores ``CachedSearchPayload`` (intent + results) keyed by the exact
    normalized query string.

    Two write modes:

    - ``put(query, results, intent=...)`` writes a payload (intent + results)
      so cache-hit branches can re-run eRanker (and downstream diversity) without
      re-running QI.
    - Legacy ``put(query, results)`` (no intent) keeps test fixtures and
      pre-payload callers working — the entry is stored as a bare
      ``RankedResults`` and ``get`` returns the same shape so existing
      behavior is preserved.

    :param config: ExactCacheConfig - Tier-specific config (TTL, max entries, enabled)
    :param snapshot_version_fn: Optional[SnapshotVersionFn] - Callable
        returning the live inventory snapshot version. When provided,
        ``put`` tags every payload entry with the current version and
        ``get_payload`` silently treats stale-tagged entries as misses
        (Wave 9 Gap 13). None disables per-entry versioning entirely.
    """

    def __init__(self, config: ExactCacheConfig, snapshot_version_fn: Optional[SnapshotVersionFn] = None):
        self._config = config
        self._snapshot_version_fn = snapshot_version_fn
        # Stale-on-read counter for observability; reset on ``invalidate_all``
        # to align with the existing tier-wide invalidation lifecycle.
        self._stale_skips = 0
        # Value is a union — either a payload (preferred for hot path) or a
        # bare RankedResults (legacy contract). The orchestrator uses
        # ``get_payload`` to opt into the payload-aware path.
        self._store: LRUTTLCache[Union[CachedSearchPayload, RankedResults]] = LRUTTLCache(
            max_entries=config.max_entries,
            ttl_seconds=config.ttl_seconds,
            max_bytes=config.max_bytes,
            ttl_jitter_seconds=config.ttl_jitter_seconds,
        )

    @property
    def hits(self) -> int:
        return self._store.hits

    @property
    def misses(self) -> int:
        return self._store.misses

    def get(self, normalized_query: str) -> Optional[RankedResults]:
        """Legacy accessor — returns the cached ``RankedResults`` regardless of
        whether the entry was stored as a payload or bare results.
        """
        payload_or_results = self._raw_get(normalized_query)
        if payload_or_results is None:
            return None
        if isinstance(payload_or_results, CachedSearchPayload):
            return payload_or_results.results
        return payload_or_results

    def get_payload(self, normalized_query: str) -> Optional[CachedSearchPayload]:
        """Payload-aware accessor used by the orchestrator's cache-hit branches.

        Returns ``None`` for both miss and legacy entries (legacy entries
        carry no intent and therefore can't drive on-cache-hit eRanker replay;
        callers fall back to the bare ``get`` in that case).

        When ``snapshot_version_fn`` is wired (Wave 9 Gap 13), payload
        entries whose ``snapshot_version`` is below the live registry
        version are silently treated as misses — old entries expire on
        access without requiring tier-wide ``invalidate_all`` on every
        snapshot bump.
        """
        payload_or_results = self._raw_get(normalized_query)
        if isinstance(payload_or_results, CachedSearchPayload):
            if self._snapshot_version_fn is not None:
                current = int(self._snapshot_version_fn())
                if not payload_or_results.is_fresh_for(current):
                    self._stale_skips += 1
                    logger.info(f"cache_exact_stale_skip key_preview={exact_query_key(normalized_query)[:12]} entry_version={payload_or_results.snapshot_version} current_version={current}")
                    return None
            return payload_or_results
        return None

    def _raw_get(self, normalized_query: str) -> Optional[Union[CachedSearchPayload, RankedResults]]:
        if not self._config.enabled:
            return None
        if not normalized_query:
            return None
        key = exact_query_key(normalized_query)
        value = self._store.get(key)
        if value is not None:
            logger.info(f"cache_exact hit=true key_preview={key[:12]}")
        return value

    def put(self, normalized_query: str, results: RankedResults, intent: Optional[QueryIntent] = None) -> None:
        """Insert a cache entry.

        :param normalized_query: str - Cache key (post-sanitizer normalized query)
        :param results: RankedResults - Pre-eRanker fused retrieval output (same as orchestrator cache write)
        :param intent: Optional[QueryIntent] - When provided, the entry is
            stored as a ``CachedSearchPayload`` so cache-hit branches can
            re-run eRanker. Omit to preserve legacy bare-result
            semantics for callers that don't have an intent on hand.
        """
        if not self._config.enabled:
            return
        if not normalized_query:
            return
        key = exact_query_key(normalized_query)
        if intent is not None:
            version = int(self._snapshot_version_fn()) if self._snapshot_version_fn is not None else 0
            self._store.put(key, CachedSearchPayload(intent=intent, results=results, snapshot_version=version))
        else:
            self._store.put(key, results)

    @property
    def stale_skips(self) -> int:
        """Per-entry-versioning observability: payload entries silently
        treated as misses on read because their ``snapshot_version`` was
        below the live registry version. Reset on ``invalidate_all``.
        """
        return self._stale_skips

    def invalidate_all(self) -> int:
        """drop every entry. Wired as a snapshot-version hook in the
        registry so an inventory advance silently discards stale cached results.
        :return: int - Entries dropped (0 when already empty / disabled)
        """
        if not self._config.enabled:
            return 0
        dropped = len(self._store)
        self._store.clear()
        self._stale_skips = 0
        if dropped > 0:
            logger.info(f"cache_exact_invalidated entries_dropped={dropped}")
        return dropped
