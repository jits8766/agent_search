"""Tier 3 — structured intermediate cache keyed by (query_type, filter dict).
Caches the structured-retriever's `CandidateSet` so refining filters via the
search surface short-circuits the structured backend lookup.

Per-entry snapshot-version tagging. When the inventory snapshot
advances (``SnapshotVersionRegistry.bump``), legacy invalidation drops
the entire cache. Per-entry tagging lets the cache silently SKIP a stale
entry on read instead of preemptively blowing every entry on write — old
entries simply expire on access while still-fresh entries (those tagged
with the current version) keep serving traffic. The cache asks the
caller-supplied ``snapshot_version_provider`` for the live version on
every read; entries tagged below that version are evicted on read and
counted as misses.
"""
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from semantic_search.cache.keys import structured_intermediate_key
from semantic_search.cache.lru_ttl import LRUTTLCache
from semantic_search.config.models import StructuredCacheConfig
from semantic_search.contracts import CandidateSet
from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class _VersionedEntry:
    """Internal per-entry container that pairs the cached payload with the
    snapshot version under which it was written. Stored values without a version
    tag are wrapped on read with ``snapshot_version=0`` so they naturally expire
    the next time the registry version advances.
    """
    candidate_set: CandidateSet
    snapshot_version: int


class StructuredCache:
    """Stores `CandidateSet`s from the structured retriever keyed by (query_type, filters).

    :param config: StructuredCacheConfig - Tier-specific config
    :param snapshot_version_provider: Optional[Callable[[], int]] - Returns
        the live snapshot version. When None, every read accepts the cached
        entry regardless of its tag (legacy behaviour). Wired in
        ``registry.py`` to ``SnapshotVersionRegistry.version`` so the cache
        always sees the current version without taking a hard dependency on
        the registry instance itself.
    """

    def __init__(self, config: StructuredCacheConfig, snapshot_version_provider: Optional[Callable[[], int]] = None):
        self._config = config
        self._store: LRUTTLCache[_VersionedEntry] = LRUTTLCache(
            max_entries=config.max_entries,
            ttl_seconds=config.ttl_seconds,
            max_bytes=config.max_bytes,
            ttl_jitter_seconds=config.ttl_jitter_seconds,
        )
        self._snapshot_version_provider = snapshot_version_provider
        self._stale_evictions = 0

    @property
    def hits(self) -> int:
        # Underlying LRU counts every key-present read as a hit. We subtract
        # stale evictions so the public ``hits`` metric reflects only reads
        # that returned a payload to the caller — keeping hit-rate proxies
        # honest when snapshot bumps cause many tagged-stale evictions.
        return self._store.hits - self._stale_evictions

    @property
    def misses(self) -> int:
        # Stale evictions are reported via ``stale_evictions`` only — they
        # are NOT folded into misses to avoid double-counting in
        # downstream alarms (e.g. cache miss-storm detector).
        return self._store.misses

    @property
    def stale_evictions(self) -> int:
        """Per-entry stale evictions caused by snapshot version advance.

        Distinct from ``misses`` because a stale eviction means the cache
        had a hit on the key but the entry was rejected by the
        snapshot-version check. Operators use this to confirm the
        per-entry tagging is doing useful work (high evictions immediately
        after a ``SnapshotVersionRegistry.bump`` is the expected pattern).
        """
        return self._stale_evictions

    def _current_snapshot_version(self) -> int:
        """Resolve the live snapshot version (0 when no provider is wired)."""
        if self._snapshot_version_provider is None:
            return 0
        try:
            return int(self._snapshot_version_provider())
        except Exception as e:
            # Defensive: the registry should never raise here, but if it
            # does we degrade to "accept everything" rather than blowing
            # the whole cache tier — matches the existing soft-fail
            # philosophy elsewhere in the cache stack.
            logger.warning(f"cache_structured_snapshot_provider_failed error_type={type(e).__name__} error={str(e)}")
            return 0

    def get(self, query_type: str, filters: Dict[str, Any]) -> Optional[CandidateSet]:
        if not self._config.enabled:
            return None
        if not query_type or not filters:
            return None
        key = structured_intermediate_key(query_type, filters)
        entry = self._store.get(key)
        if entry is None:
            return None
        current_version = self._current_snapshot_version()
        if entry.snapshot_version < current_version:
            # Stale entry — eject it so the next read is a clean miss and
            # the next put rewrites with the live version. Counted as a
            # stale eviction (NOT a regular miss) so dashboards can
            # distinguish "no entry" from "stale entry rejected".
            self._store.invalidate(key)
            self._stale_evictions += 1
            logger.info(f"cache_structured stale_evicted=true key_preview={key[:12]} entry_version={entry.snapshot_version} current_version={current_version}")
            return None
        logger.info(f"cache_structured hit=true key_preview={key[:12]} candidates={len(entry.candidate_set.candidates)} version={entry.snapshot_version}")
        return entry.candidate_set

    def put(self, query_type: str, filters: Dict[str, Any], candidate_set: CandidateSet) -> None:
        if not self._config.enabled:
            return
        if not query_type or not filters:
            return
        key = structured_intermediate_key(query_type, filters)
        version = self._current_snapshot_version()
        self._store.put(key, _VersionedEntry(candidate_set=candidate_set, snapshot_version=version))

    def invalidate_all(self) -> int:
        """drop every entry. Retained as a fallback for catastrophic schema
        changes (e.g. payload field rename) where per-entry expiry is too
        slow. Per-entry snapshot-version tagging is the preferred
        path on routine inventory advances.

        :return: int - Entries dropped (0 when already empty / disabled)
        """
        if not self._config.enabled:
            return 0
        dropped = len(self._store)
        self._store.clear()
        if dropped > 0:
            logger.info(f"cache_structured_invalidated entries_dropped={dropped}")
        return dropped
