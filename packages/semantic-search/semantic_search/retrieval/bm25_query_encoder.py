"""BM25-style sparse query encoder for the Qdrant hybrid path.

Hybrid retrieval against the unified Qdrant index requires the registry to
hand a `bm25_query_fn(text: str) -> qdrant_client.models.SparseVector` to
``QdrantHybridRetriever`` whenever ``retrieval.qdrant.hybrid.bm25_enabled=true``.

This module provides the canonical implementation of that callable:
``BM25QueryEncoder``. It is intentionally:

- **Stdlib-only** — deterministic across runs, trivially mockable in tests, and
  zero new runtime dependencies for the semantic_search package boot path
  (`qdrant-client` itself remains the only optional dep, imported lazily on the
  first call so module import never fails).
- **Query-side only** — the corpus-side BM25 IDF + term frequencies are baked
  into the sparse vectors at ingest (see Qdrant docs on sparse vectors). The
  query-side encoder must produce *term weights* that line up with the indexed
  vocabulary on the same hashing scheme, so this class is the single source of
  truth for tokenization + hashing + sub-linear TF on the query path.
- **Hash-based vocabulary** — terms are mapped to integer ids via a stable
  ``sha256`` hash mod ``vocab_size``. The same scheme MUST be used at ingest. We
  intentionally do not ship a learned vocabulary file because the auctions
  corpus mints new TLDs and SLDs daily; a fixed-size hashing vocabulary
  (256 K–1 M buckets) survives churn at the cost of a small, bounded collision
  rate which the BM25 sparse-fusion stage tolerates well in practice.

Shape contract returned by ``__call__``:

- ``qm.SparseVector(indices=List[int], values=List[float])`` where ``len(indices)
  == len(values)``, ``indices`` are unique and sorted ascending, and all values
  are positive floats.

Math:

- TF weighting is sub-linear ``1.0 + log(1 + tf)`` (matches Lucene's classic
  BM25 query-side term boosting; Qdrant fuses the per-term ``query_weight ×
  indexed_term_weight`` server-side via Reciprocal Rank Fusion or DBSF).

Robustness:

- ``None`` / empty / whitespace-only input → empty SparseVector (no crash).
- After tokenization a query may yield zero terms (all stopwords / too short)
  → empty SparseVector. The hybrid retriever degrades to dense-only behaviour
  for that single query (Qdrant ignores empty prefetches).

Layer placement: ``retrieval/`` (Option A — alongside
``qdrant_adapter.py``). Per ``architecture.mdc``: imports stdlib + ``core`` only.
``qdrant_client`` is imported lazily inside ``__call__`` to keep the module
test-friendly when the package is missing.
"""
import hashlib
import math
from typing import Any, List, Optional, Sequence, Tuple

try:
    from qdrant_client import models as _qm
except ImportError:
    _qm = None

from semantic_search.config.models import BM25QueryEncoderConfig
from semantic_search.core.exceptions import RetrievalError
from semantic_search.core.logging_utils import get_logger
from semantic_search.retrieval.dynamic_synonym_store import DynamicSynonymStore
from semantic_search.retrieval.lexical_tokenizer import tokenize_lexical
from semantic_search.retrieval.synonym_expander import SynonymExpander

logger = get_logger(__name__)


class BM25QueryEncoder:
    """Stable hash sparse encoder for Qdrant hybrid (query-side BM25 term weights)."""

    def __init__(self, config: BM25QueryEncoderConfig, dynamic_store: Optional[DynamicSynonymStore] = None):
        if not isinstance(config, BM25QueryEncoderConfig):
            raise RetrievalError("BM25QueryEncoder requires a BM25QueryEncoderConfig")
        self._vocab_size = int(config.vocab_size)
        self._min_term_length = int(config.min_term_length)
        self._max_terms = int(config.max_terms)
        self._stopwords = frozenset(s.lower() for s in config.stopwords)
        # Synonym expander (None if disabled; zero overhead when off)
        self._expander: Optional[SynonymExpander] = None
        if config.synonyms is not None and config.synonyms.enabled:
            self._expander = SynonymExpander(config.synonyms)
            logger.info(
                f"bm25_synonym_expander_enabled map_size={self._expander.map_size} "
                f"expansion_weight={config.synonyms.expansion_weight} "
                f"max_synonyms_per_token={config.synonyms.max_synonyms_per_token} "
                f"max_tokens_to_expand={config.synonyms.max_tokens_to_expand}"
            )
        # Dynamic synonym store — caller may inject a pre-built instance (allows
        # the registry to share one store between multiple encoder references).
        # When not injected and the config block is present, build one inline.
        self._dynamic_store: Optional[DynamicSynonymStore] = None
        if dynamic_store is not None and isinstance(dynamic_store, DynamicSynonymStore):
            self._dynamic_store = dynamic_store
        elif config.dynamic_synonyms is not None and config.dynamic_synonyms.enabled:
            self._dynamic_store = DynamicSynonymStore(config.dynamic_synonyms)
        if self._dynamic_store is not None:
            self._dyn_weight = float(config.dynamic_synonyms.expansion_weight) if config.dynamic_synonyms else 0.4
            self._dyn_max_synonyms = int(config.dynamic_synonyms.max_synonyms_per_token) if config.dynamic_synonyms else 3
            logger.info(f"bm25_dynamic_synonym_store_wired map_size={self._dynamic_store.map_size} expansion_weight={self._dyn_weight}")

    @property
    def vocab_size(self) -> int:
        """Hashing vocabulary cardinality (mirrors corpus-side ingest config)."""
        return self._vocab_size

    @property
    def dynamic_store(self) -> Optional[DynamicSynonymStore]:
        """The dynamic synonym store, or None when not wired."""
        return self._dynamic_store

    def _tokenize(self, text: str) -> List[str]:
        """Lowercase, extract ``[a-z0-9]+`` runs, drop short tokens + stopwords.

        Thin instance-method wrapper over the shared ``tokenize_lexical`` helper
        — the lexical reranker calls the same helper so the BM25 sparse leg and
        the reranker score on the SAME token universe.

        :param text: str - Raw query text (already-normalized OK)
        :return: List[str] - Surviving tokens preserving first-seen order
        """
        return tokenize_lexical(text=text, min_term_length=self._min_term_length, max_terms=self._max_terms, stopwords=self._stopwords)

    @staticmethod
    def _hash_term(term: str, vocab_size: int) -> int:
        """Stable, language-agnostic term -> bucket id in ``[0, vocab_size)``.

        Uses sha256 (collision-resistant for our purposes — we tolerate the
        small mod-bucket collision rate at the BM25 fusion stage).
        """
        digest = hashlib.sha256(term.encode('utf-8')).digest()
        return int.from_bytes(digest[:8], byteorder='big', signed=False) % vocab_size

    def _aggregate_term_weights(self, tokens: Sequence[str]) -> List[tuple]:
        """Compute (bucket_id, weight) pairs with sub-linear TF + collision-safe sum.

        Distinct terms that collide on the same bucket id have their weights
        summed (BM25-fusion-friendly; matches the indexer side).

        Synonym sources (both apply when wired):
          - Static ``SynonymExpander`` (bidirectional, from ``config.synonyms``)
          - Dynamic ``DynamicSynonymStore`` (bidirectional, persisted SQLite map
            that grows as the system handles more queries)

        Per-term weight: ``multiplier × (1.0 + log(1.0 + tf))`` where:
          - originals carry ``multiplier = 1.0`` and ``tf`` = input count
          - synonyms carry ``multiplier = expansion_weight ∈ (0, 1]`` and
            ``tf = 1`` (synonyms are emitted at most once per query)

        Tokens with no synonym in either source are recorded as misses so the
        background ``QueryDrivenExpander`` can schedule LLM expansion.

        :param tokens: Sequence[str] - Tokens after tokenization
        :return: List[Tuple[int,float]] - Sorted-ascending pairs (bucket_id, weight)
        """
        original_tf: dict = {}
        for tok in tokens:
            original_tf[tok] = original_tf.get(tok, 0) + 1
        # Static synonym pairs from the static expander.
        static_synonyms: List[Tuple[str, float]] = []
        if self._expander is not None:
            expanded = self._expander.expand(tokens)
            static_synonyms = [(t, m) for t, m in expanded if m != 1.0]
        # Dynamic synonym pairs from the persistent store. Also record misses
        # for tokens not covered by either source so they get queued for LLM
        # expansion. Dedup against originals + static synonyms to prevent
        # double-counting a term that appears in both sources.
        dyn_synonyms: List[Tuple[str, float]] = []
        if self._dynamic_store is not None:
            emitted: set = set(original_tf.keys()) | {t for t, _ in static_synonyms}
            for tok in original_tf.keys():
                dyn_syns = self._dynamic_store.get(tok)
                if dyn_syns:
                    added = 0
                    for syn in dyn_syns:
                        if added >= self._dyn_max_synonyms:
                            break
                        if syn not in emitted:
                            dyn_synonyms.append((syn, self._dyn_weight))
                            emitted.add(syn)
                            added += 1
                elif self._expander is None or not self._expander._bidir_map.get(tok):
                    # Token has no synonyms in either static or dynamic map — queue it.
                    self._dynamic_store.record_miss(tok)
        # Build bucket weights from originals + all synonym pairs.
        bucket_weights: dict = {}
        for term, count in original_tf.items():
            bid = self._hash_term(term, self._vocab_size)
            w = 1.0 + math.log(1.0 + float(count))
            bucket_weights[bid] = bucket_weights.get(bid, 0.0) + w
        for term, multiplier in static_synonyms + dyn_synonyms:
            bid = self._hash_term(term, self._vocab_size)
            w = float(multiplier) * (1.0 + math.log(2.0))
            bucket_weights[bid] = bucket_weights.get(bid, 0.0) + w
        return sorted(bucket_weights.items(), key=lambda kv: kv[0])

    def __call__(self, text: str) -> Any:
        """Encode query text into a Qdrant ``SparseVector``.

        :param text: str - Raw or normalized query text
        :return: qdrant_client.models.SparseVector - Empty when no terms survive
        :raises RetrievalError: When ``qdrant-client`` is not installed
        """
        if _qm is None:
            raise RetrievalError("BM25QueryEncoder requires qdrant-client to be installed; install it or disable retrieval.qdrant.hybrid.bm25_enabled")
        qm = _qm
        tokens = self._tokenize(text or '')
        if not tokens:
            return qm.SparseVector(indices=[], values=[])
        pairs = self._aggregate_term_weights(tokens)
        indices = [int(b) for b, _ in pairs]
        values = [float(w) for _, w in pairs]
        return qm.SparseVector(indices=indices, values=values)
