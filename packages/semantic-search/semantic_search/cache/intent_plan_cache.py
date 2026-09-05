"""Tier 3 (plan) — intent-structure cache: fingerprint(QueryIntent) → CachedSearchPayload.

Consulted after QI ``classify`` and exact misses; skips retrieval when the
same structured intent was served recently. Per-entry snapshot versioning matches
``ExactCache`` / ``StructuredCache`` (Wave 9 Gap 13).
"""
from typing import Callable, Optional, Union

from semantic_search.cache.lru_ttl import LRUTTLCache
from semantic_search.config.models import IntentPlanCacheConfig
from semantic_search.contracts import CachedSearchPayload, QueryIntent, RankedResults
from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)

SnapshotVersionFn = Callable[[], int]


class IntentPlanCache:
    """LRU+TTL store keyed by ``intent_plan_fingerprint(intent)`` hex string."""

    def __init__(self, config: IntentPlanCacheConfig, snapshot_version_fn: Optional[SnapshotVersionFn] = None):
        self._config = config
        self._snapshot_version_fn = snapshot_version_fn
        self._stale_skips = 0
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

    def get(self, fingerprint_hex: str) -> Optional[RankedResults]:
        payload_or_results = self._raw_get(fingerprint_hex)
        if payload_or_results is None:
            return None
        if isinstance(payload_or_results, CachedSearchPayload):
            return payload_or_results.results
        return payload_or_results

    def get_payload(self, fingerprint_hex: str) -> Optional[CachedSearchPayload]:
        payload_or_results = self._raw_get(fingerprint_hex)
        if isinstance(payload_or_results, CachedSearchPayload):
            if self._snapshot_version_fn is not None:
                current = int(self._snapshot_version_fn())
                if not payload_or_results.is_fresh_for(current):
                    self._stale_skips += 1
                    logger.info(f"cache_intent_plan_stale_skip key_preview={fingerprint_hex[:12]} entry_version={payload_or_results.snapshot_version} current_version={current}")
                    return None
            return payload_or_results
        return None

    def _raw_get(self, fingerprint_hex: str) -> Optional[Union[CachedSearchPayload, RankedResults]]:
        if not self._config.enabled:
            return None
        if not fingerprint_hex:
            return None
        value = self._store.get(fingerprint_hex)
        if value is not None:
            logger.info(f"cache_intent_plan hit=true key_preview={fingerprint_hex[:12]}")
        return value

    def put(self, fingerprint_hex: str, results: RankedResults, intent: Optional[QueryIntent] = None) -> None:
        if not self._config.enabled:
            return
        if not fingerprint_hex:
            return
        if intent is not None:
            version = int(self._snapshot_version_fn()) if self._snapshot_version_fn is not None else 0
            self._store.put(fingerprint_hex, CachedSearchPayload(intent=intent, results=results, snapshot_version=version))
        else:
            self._store.put(fingerprint_hex, results)

    @property
    def stale_skips(self) -> int:
        return self._stale_skips

    def invalidate_all(self) -> int:
        if not self._config.enabled:
            return 0
        dropped = len(self._store)
        self._store.clear()
        self._stale_skips = 0
        if dropped > 0:
            logger.info(f"cache_intent_plan_invalidated entries_dropped={dropped}")
        return dropped


__all__ = ['IntentPlanCache']
