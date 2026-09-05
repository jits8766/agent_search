"""Tier-0.5 QI semantic intent cache: query embedding to QueryIntent.

On lookup, the incoming query is encoded and compared by cosine similarity against
every stored embedding. A cached QueryIntent is returned when the best match meets
or exceeds the configured similarity_threshold. This catches rephrased queries that
miss the exact normalized-query cache without re-running L1/L2 classification.

The stored vectors are downcast to packed float32 arrays (~4x smaller than
list[float] in CPython). cosine_similarity accepts any numeric iterable so the
downcast is transparent to callers.

Thread safety: asyncio event loop is single-threaded; cache ops run in the
coroutine body (no await), so all mutations execute in one thread with no
interleaving. No locking is needed.
"""
import array as _array
from typing import Optional, Tuple

from semantic_search.cache.keys import versioned_query_key
from semantic_search.cache.lru_ttl import LRUTTLCache
from semantic_search.config.models import QISemanticIntentCacheConfig
from semantic_search.contracts import QueryIntent
from semantic_search.core.exceptions import CacheError
from semantic_search.core.logging_utils import get_logger
from semantic_search.qi.encoder import Encoder, cosine_similarity

logger = get_logger(__name__)


class QISemanticIntentCache:
    """LRU+TTL store keyed by query embedding for fuzzy QueryIntent lookup.

    :param config: QISemanticIntentCacheConfig - Sizing, TTL, and threshold parameters
    :param encoder: Encoder - Shared encoder (dim must equal config.embedding_dim)
    """

    def __init__(self, config: QISemanticIntentCacheConfig, encoder: Encoder):
        if encoder.dim != config.embedding_dim:
            raise CacheError(f"QISemanticIntentCache encoder.dim={encoder.dim} != config.embedding_dim={config.embedding_dim}")
        self._config = config
        self._encoder = encoder
        self._store: LRUTTLCache[Tuple[_array.array, QueryIntent]] = LRUTTLCache(
            max_entries=config.max_entries,
            ttl_seconds=config.ttl_seconds,
        )

    def get(
        self,
        normalized_query: str,
        *,
        prompt_tag: str,
        schema_version: str,
    ) -> Optional[QueryIntent]:
        """Return a cached QueryIntent when a semantically similar query was seen before.

        Encodes the incoming query and scans stored embeddings for cosine similarity.
        Returns the most-similar QueryIntent iff that similarity meets the threshold
        and the entry's prompt/schema versions match the caller versions.

        :param normalized_query: str - Lowercased, whitespace-collapsed query
        :param prompt_tag: str - Classify prompt version from config
        :param schema_version: str - Classify schema version from config
        :return: Optional[QueryIntent] - Cached intent on hit, None on miss or disabled
        """
        if not self._config.enabled or not normalized_query:
            return None
        if not isinstance(prompt_tag, str) or not prompt_tag.strip():
            raise CacheError("QISemanticIntentCache.get requires non-empty prompt_tag")
        if not isinstance(schema_version, str) or not schema_version.strip():
            raise CacheError("QISemanticIntentCache.get requires non-empty schema_version")
        query_vec = self._encoder.encode(normalized_query)
        if all(x == 0.0 for x in query_vec):
            return None
        best_score = -1.0
        best_intent: Optional[QueryIntent] = None
        for _key, (stored_vec, intent) in self._store.items_snapshot():
            if intent.prompt_tag != prompt_tag or intent.schema_version != schema_version:
                continue
            sim = cosine_similarity(query_vec, stored_vec)
            if sim > best_score:
                best_score = sim
                best_intent = intent
        _effective_threshold = self._config.similarity_threshold
        if best_intent is not None and self._config.type_specific_similarity_thresholds:
            _primary_type = best_intent.slices[0].query_type if best_intent.slices else None
            if _primary_type and _primary_type in self._config.type_specific_similarity_thresholds:
                _effective_threshold = self._config.type_specific_similarity_thresholds[_primary_type]
        if best_intent is not None and best_score >= _effective_threshold:
            logger.info(f"qi_semantic_intent_cache_hit similarity={best_score:.4f} threshold={_effective_threshold} query_len={len(normalized_query)}")
            return best_intent
        logger.debug(f"qi_semantic_intent_cache_miss best_similarity={best_score:.4f} threshold={_effective_threshold} query_len={len(normalized_query)}")
        return None

    def put(
        self,
        normalized_query: str,
        intent: QueryIntent,
        *,
        prompt_tag: str,
        schema_version: str,
    ) -> None:
        """Store a QueryIntent keyed by the query embedding.

        :param normalized_query: str - Text to encode for similarity lookup
        :param intent: QueryIntent - Classification result to store
        :param prompt_tag: str - Classify prompt version from config
        :param schema_version: str - Classify schema version from config
        """
        if not self._config.enabled or not normalized_query:
            return
        if not isinstance(prompt_tag, str) or not prompt_tag.strip():
            raise CacheError("QISemanticIntentCache.put requires non-empty prompt_tag")
        if not isinstance(schema_version, str) or not schema_version.strip():
            raise CacheError("QISemanticIntentCache.put requires non-empty schema_version")
        query_vec = self._encoder.encode(normalized_query)
        if all(x == 0.0 for x in query_vec):
            return
        store_key = versioned_query_key(
            normalized_query,
            prompt_tag=prompt_tag,
            schema_version=schema_version,
        )
        stored_vec = _array.array('f', query_vec)
        self._store.put(store_key, (stored_vec, intent))

    @property
    def hits(self) -> int:
        """Total cache hits since construction."""
        return self._store.hits

    @property
    def misses(self) -> int:
        """Total cache misses since construction."""
        return self._store.misses

    def invalidate_all(self) -> int:
        """Drop every cached intent. Returns entries removed."""
        dropped = len(self._store)
        self._store.clear()
        if dropped:
            logger.info(f"qi_semantic_intent_cache_invalidated entries_dropped={dropped}")
        return dropped


__all__ = ['QISemanticIntentCache']
