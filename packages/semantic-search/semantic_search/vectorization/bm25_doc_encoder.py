"""BM25-style sparse encoder for the corpus side of hybrid retrieval.

Companion to :class:`semantic_search.retrieval.bm25_query_encoder.BM25QueryEncoder`.
The two MUST share ``vocab_size`` and the ``sha256`` hashing scheme so that
indexed sparse vectors and runtime query sparse vectors line up bucket-for-bucket
inside Qdrant. This module is the single source of truth for the corpus-side
encoder; all offline indexer jobs and ad-hoc backfills go through it.

Key differences from the query-side encoder:

- Operates on already-segmented token streams (the offline indexer runs
  :class:`semantic_search.vectorization.segmenter.DomainNameSegmenter` first,
  then optionally expands via the same :class:`SynonymExpander` the query
  side uses, then hands the resulting tokens here). The encoder does NOT
  re-tokenise — it trusts its caller for tokenisation so the same token
  universe is used at index and query time.
- Uses the canonical BM25 doc-side TF saturation: ``tf / (tf + k1)``
  with ``k1`` from config. This matches Lucene's BM25Similarity for the
  corpus side and pairs naturally with the query-side
  ``1.0 + log(1.0 + tf)`` weighting (Qdrant's sparse fusion multiplies
  the two element-wise).
- Per-document length-normalisation is OFF by default (the ``b``
  parameter is exposed in config; setting it to 0 yields the
  unnormalised BM25 commonly used for short-text corpora like domain
  names where document length variance is low).

Stdlib-only. ``qdrant_client`` is imported lazily inside ``__call__`` so
the module imports successfully even when the optional dep is absent.
"""
import hashlib
from typing import Any, Dict, List, Sequence, Tuple

from semantic_search.core.exceptions import RetrievalError, ValidationError
from semantic_search.core.logging_utils import get_logger

try:
    from qdrant_client import models as _qm_bm25
except ImportError:
    _qm_bm25 = None

logger = get_logger(__name__)


class BM25DocEncoder:
    """Corpus-side BM25 sparse encoder (companion to query-side BM25QueryEncoder)."""

    def __init__(self, vocab_size: int, k1: float = 1.2, b: float = 0.0, avg_doc_length: float = 1.0):
        if int(vocab_size) < 1:
            raise ValidationError("BM25DocEncoder.vocab_size must be >= 1")
        if float(k1) <= 0.0:
            raise ValidationError("BM25DocEncoder.k1 must be > 0")
        if not 0.0 <= float(b) <= 1.0:
            raise ValidationError("BM25DocEncoder.b must be in [0.0, 1.0]")
        if float(avg_doc_length) <= 0.0:
            raise ValidationError("BM25DocEncoder.avg_doc_length must be > 0")
        self._vocab_size = int(vocab_size)
        self._k1 = float(k1)
        self._b = float(b)
        self._avg_doc_length = float(avg_doc_length)

    @property
    def vocab_size(self) -> int:
        """Vocab cardinality (must match query-side encoder)."""
        return self._vocab_size

    @staticmethod
    def _hash_term(term: str, vocab_size: int) -> int:
        """Stable term -> bucket id in ``[0, vocab_size)``.

        IDENTICAL hashing to ``BM25QueryEncoder._hash_term``: ``sha256`` of
        the UTF-8 bytes, take the first 8 bytes as a big-endian unsigned
        int, mod ``vocab_size``. Any change here must be mirrored on the
        query side or retrieval silently breaks.
        """
        digest = hashlib.sha256(term.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False) % vocab_size

    def _length_norm(self, doc_length: int) -> float:
        """Compute the BM25 length-normalisation factor.

        Formula: ``1 - b + b * (doc_length / avg_doc_length)``. When
        ``b=0`` the factor collapses to 1.0 and length normalisation is a
        no-op (the recommended setting for short-text corpora).
        """
        if self._b == 0.0:
            return 1.0
        return 1.0 - self._b + self._b * (float(doc_length) / self._avg_doc_length)

    def _aggregate(self, tokens: Sequence[str]) -> List[Tuple[int, float]]:
        """Compute (bucket_id, weight) pairs with BM25 doc-side TF saturation.

        Two surface tokens that hash into the same bucket have their
        weights summed (collision-safe; matches the query-side encoder's
        bucket-collision policy).
        """
        if not tokens:
            return []
        tf: Dict[str, int] = {}
        for tok in tokens:
            if not isinstance(tok, str) or not tok:
                continue
            tf[tok] = tf.get(tok, 0) + 1
        if not tf:
            return []
        doc_length = sum(tf.values())
        norm = self._length_norm(doc_length)
        bucket_weights: Dict[int, float] = {}
        for term, count in tf.items():
            bid = self._hash_term(term, self._vocab_size)
            # Canonical BM25 doc-side TF saturation. The (k1 + 1) factor
            # is conventional but optional (it scales every weight by the
            # same constant; Qdrant fuses with the query side, so the
            # constant cancels out at fusion time). We include it for
            # parity with the standard formulation.
            tf_sat = ((self._k1 + 1.0) * float(count)) / (
                self._k1 * norm + float(count)
            )
            bucket_weights[bid] = bucket_weights.get(bid, 0.0) + tf_sat
        return sorted(bucket_weights.items(), key=lambda kv: kv[0])

    def encode(self, tokens: Sequence[str]) -> Tuple[List[int], List[float]]:
        """Encode a token sequence to ``(indices, values)`` lists.

        Pure-stdlib helper that returns plain lists so callers without
        ``qdrant-client`` (e.g. backfill scripts emitting Parquet) can
        still produce the canonical sparse representation.

        :param tokens: Sequence[str] - Tokens emitted by the segmenter
            (and optionally expanded by ``SynonymExpander.expand``)
        :return: Tuple[List[int], List[float]] - Sorted-ascending parallel
            lists; both empty when no token survives
        :raises ValidationError: When ``tokens`` is None
        """
        if tokens is None:
            raise ValidationError("BM25DocEncoder.encode requires a non-None token sequence")
        pairs = self._aggregate(tokens)
        indices = [int(b) for b, _ in pairs]
        values = [float(w) for _, w in pairs]
        return indices, values

    def __call__(self, tokens: Sequence[str]) -> Any:
        """Encode tokens into a Qdrant ``SparseVector``.

        :param tokens: Sequence[str] - Tokens emitted by the segmenter
        :return: qdrant_client.models.SparseVector - Empty when no terms survive
        :raises ValidationError: When ``tokens`` is None
        :raises RetrievalError: When ``qdrant-client`` is not installed
        """
        if _qm_bm25 is None:
            raise RetrievalError(
                "BM25DocEncoder requires qdrant-client; install it or use "
                ".encode() to obtain plain (indices, values) lists"
            )
        indices, values = self.encode(tokens)
        return _qm_bm25.SparseVector(indices=indices, values=values)
