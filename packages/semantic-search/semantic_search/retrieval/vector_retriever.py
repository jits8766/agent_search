"""Vector retriever — cosine similarity over an encoder-derived index.
Ships a `VectorIndex` interface plus an in-memory implementation. Production
swaps the same interface for a Qdrant-backed implementation by injecting a
different adapter; no caller changes.
"""
import asyncio
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from semantic_search.config.models import VectorRetrievalConfig
from semantic_search.core.exceptions import RetrievalError
from semantic_search.core.logging_utils import get_logger
from semantic_search.qi.encoder import Encoder, cosine_similarity
from semantic_search.qi.residual_extractor import semantic_encode_text_for
from semantic_search.retrieval.base import Retriever, slice_candidates
from semantic_search.retrieval.structured_retriever import InMemoryStructuredIndex, extract_hard_filters_from_intent
from semantic_search.retrieval.topic_negation import build_category_centroids, subtract_topics
from semantic_search.contracts import Candidate, CandidateSet, QueryIntent

logger = get_logger(__name__)

# When hard filters are present the ANN top_k is over-fetched by this factor
# before payload filtering, so precision filtering does not starve the fused
# result set. The in-memory index is a full scan, so a generous pool is cheap.
_FILTER_OVERFETCH = 25


class VectorIndex:
    """Vector store interface."""

    @property
    def dim(self) -> int:
        """Embedding dimension stored in this index."""
        raise NotImplementedError

    def search(self, query_vec: Sequence[float], top_k: int) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Return up to top_k (item_id, similarity, payload) tuples in score-descending order."""
        raise NotImplementedError


class InMemoryVectorIndex(VectorIndex):
    """In-memory cosine-similarity index for tests and local bootstrapping."""

    def __init__(self, dim: int):
        if dim < 4:
            raise RetrievalError("InMemoryVectorIndex.dim must be >= 4")
        self._dim = int(dim)
        self._items: Dict[str, Tuple[List[float], Dict[str, Any]]] = {}

    @property
    def dim(self) -> int:
        return self._dim

    def add(self, item_id: str, vector: Sequence[float], payload: Dict[str, Any]) -> None:
        """Insert / replace an item.
        :param item_id: str - Stable item id
        :param vector: Sequence[float] - Embedding (length must equal `dim`)
        :param payload: Dict[str, Any] - Item attributes returned with the candidate
        :raises RetrievalError: If vector length doesn't match index dim
        """
        if not item_id:
            raise RetrievalError("InMemoryVectorIndex.add requires non-empty item_id")
        if len(vector) != self._dim:
            raise RetrievalError(f"InMemoryVectorIndex vector dim mismatch expected={self._dim} got={len(vector)}")
        self._items[item_id] = (list(vector), dict(payload))

    def search(self, query_vec: Sequence[float], top_k: int) -> List[Tuple[str, float, Dict[str, Any]]]:
        if len(query_vec) != self._dim:
            raise RetrievalError(f"InMemoryVectorIndex query dim mismatch expected={self._dim} got={len(query_vec)}")
        if top_k < 1:
            return []
        scored: List[Tuple[str, float, Dict[str, Any]]] = []
        for item_id, (vec, payload) in self._items.items():
            sim = cosine_similarity(query_vec, vec)
            normalized = max(0.0, min(1.0, (sim + 1.0) / 2.0))
            scored.append((item_id, normalized, payload))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored[:top_k]


class VectorRetriever(Retriever):
    """Encodes the query and pulls top-K candidates from a `VectorIndex`.
    :param config: VectorRetrievalConfig - Vector retriever config
    :param encoder: Encoder - Shared encoder (must match index dim)
    :param index: VectorIndex - Backing vector store
    :param query_preprocessor: Optional[Callable[[str], str]] - Query-side
        transform applied to the encode text before encoding (e.g. compound
        splitting). ``None`` embeds the text verbatim.
    """

    def __init__(
        self,
        config: VectorRetrievalConfig,
        encoder: Encoder,
        index: VectorIndex,
        query_preprocessor: Optional[Callable[[str], str]] = None,
    ):
        if encoder.dim != config.embedding_dim:
            raise RetrievalError(f"VectorRetriever encoder.dim={encoder.dim} != config.embedding_dim={config.embedding_dim}")
        if index.dim != config.embedding_dim:
            raise RetrievalError(f"VectorRetriever index.dim={index.dim} != config.embedding_dim={config.embedding_dim}")
        self._config = config
        self._encoder = encoder
        self._index = index
        self._query_preprocessor = query_preprocessor
        # Lazily-built per-topic centroids for topic negation (None until first use).
        self._topic_centroids: Optional[Dict[str, Any]] = None

    @property
    def source(self) -> str:
        return 'vector'

    def _maybe_negate_topics(self, intent: QueryIntent, query_vec: Sequence[float]) -> Sequence[float]:
        """Subtract excluded-topic centroids from the query vector (config-gated).

        No-op unless topic negation is enabled, the intent carries a ``topic_exclude``
        entity, and at least one excluded topic has a centroid. Centroids are built
        once (lazily) from the configured seed phrases via the shared encoder.

        :param intent: QueryIntent - Carries ``topic_exclude`` entities
        :param query_vec: Sequence[float] - Encoded query vector
        :return: Sequence[float] - Adjusted vector, or the input when not applied
        """
        cfg = self._config.topic_negation
        if cfg is None or not cfg.enabled or cfg.alpha <= 0.0 or not cfg.category_seeds:
            return query_vec
        excluded: List[str] = []
        for s in intent.slices:
            for e in s.entities:
                if e.name == 'topic_exclude' and isinstance(e.value, list):
                    excluded.extend(str(t).lower() for t in e.value)
        if not excluded:
            return query_vec
        if self._topic_centroids is None:
            try:
                self._topic_centroids = build_category_centroids(cfg.category_seeds, self._encoder.encode_batch)
            except (RuntimeError, ValueError, TypeError, KeyError, IndexError, AttributeError, OSError) as exc:
                logger.warning(f"topic_negation_centroid_build_failed error_type={type(exc).__name__} error={exc}")
                self._topic_centroids = {}
        if not self._topic_centroids:
            return query_vec
        adjusted = subtract_topics(query_vec, excluded, self._topic_centroids, cfg.alpha)
        logger.info(f"topic_negation_applied request_id={intent.request_id} excluded={excluded}")
        return adjusted.tolist()

    async def retrieve(self, intent: QueryIntent, top_k: int) -> CandidateSet:
        if not self._config.enabled:
            return CandidateSet(source='vector', candidates=[], latency_ms=0.0)
        if top_k < 1:
            return CandidateSet(source='vector', candidates=[], latency_ms=0.0)
        t0 = time.monotonic()
        # Encode the TLD-safe text (residual concept when present, otherwise the
        # normalized query with TLD literals stripped). The shared accessor
        # guarantees the TLD literal never reaches the encoder, so TLD is matched
        # only as an exact structured filter — never via cosine similarity.
        encode_text = semantic_encode_text_for(intent)
        if self._query_preprocessor is not None:
            encode_text = self._query_preprocessor(encode_text)
        using_residual = encode_text != intent.normalized_query
        query_vec = await self._encoder.encode_async(encode_text)
        query_vec = self._maybe_negate_topics(intent, query_vec)
        # Enforce hard-chip filters on the ANN result. The vector index performs
        # pure similarity search with no payload filtering, so without this it
        # leaks items that contradict an explicit hard constraint (e.g. type/tld/
        # keyword/enrichment range) into the fused set. When hard filters exist,
        # over-fetch then payload-filter via the shared structured predicate so
        # the filtered pool still yields ~top_k survivors. (Qdrant-backed indexes
        # filter at query time and use their own adapter; this path is the
        # in-memory / local index.)
        hard_filters = extract_hard_filters_from_intent(intent)
        fetch_k = top_k * _FILTER_OVERFETCH if hard_filters else top_k
        raw = await asyncio.to_thread(self._index.search, query_vec, fetch_k)
        candidates: List[Candidate] = []
        for item_id, score, payload in raw:
            if score < self._config.min_similarity:
                continue
            if hard_filters and not InMemoryStructuredIndex._matches(payload, hard_filters):  # noqa: SLF001
                continue
            # Stamp the vector similarity into the payload so RRF preserves it
            # for the Layer 4 deterministic ranker (LAYER 4 primary signal).
            # Use a copy so we never mutate the caller's payload dict.
            enriched_payload = dict(payload)
            enriched_payload['vector_score'] = float(score)
            candidates.append(Candidate(item_id=item_id, score=float(score), source='vector', payload=enriched_payload))
        candidates = slice_candidates(candidates, top_k)
        latency_ms = (time.monotonic() - t0) * 1000.0
        logger.info(f"vector_retrieval request_id={intent.request_id} candidates={len(candidates)} using_semantic_residual={using_residual} latency_ms={latency_ms:.1f}")
        return CandidateSet(source='vector', candidates=candidates, latency_ms=latency_ms)
