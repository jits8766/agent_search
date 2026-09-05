"""Character n-gram sparse encoder — recall-side fuzzy / misspell channel.

Feeds a dedicated sparse vector (``ngram``) into the Qdrant hybrid query so
near-miss and misspelled domain queries retrieve lexically-close candidates the
dense and BM42 token-sparse legs miss (query ``high rentals`` surfacing
``hi-rentals`` / ``rent.high``). Both sides hash boundary-marked character
n-grams into the SAME bucket space via ``sha256 mod vocab_size``, so a shared
character substring lands in the same bucket on the doc side (offline indexer)
and the query side (hybrid retriever) and contributes to the server-side RRF
fusion stage.

One class serves both sides (mirrors ``BM42SparseEncoder``):

- ``encode(text) -> (indices, values)`` — doc side, for the offline indexer
- ``__call__(text) -> SparseVector``     — query side, for the hybrid retriever

Both sides compute from the same boundary-marked n-gram aggregation, so the
indexed and query sparse vectors line up bucket-for-bucket. Intra-leg ranking
comes from sub-linear term frequency (``1 + log(1 + tf)``); the RRF stage fuses
by rank, so no per-leg weight scalar is required.

Stdlib-only (``hashlib`` + ``math`` + ``re``). ``qdrant_client`` is imported
lazily so the module imports cleanly when the optional dep is absent; the
doc-side ``encode`` returns plain lists and never needs it.
"""
import hashlib
import math
import re
from typing import Any, Dict, List, Tuple

try:
    from qdrant_client import models as _qm
except ImportError:
    _qm = None

from semantic_search.core.exceptions import RetrievalError, ValidationError
from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class CharNgramSparseEncoder:
    """Hashing character-n-gram sparse encoder for fuzzy lexical recall.
    :param vocab_size: int - Hashing bucket cardinality; doc + query MUST share it
    :param min_n: int - Minimum character n-gram length (>= 1)
    :param max_n: int - Maximum character n-gram length (>= min_n)
    :raises ValidationError: When parameters are out of range
    """

    def __init__(self, vocab_size: int, min_n: int, max_n: int):
        if int(vocab_size) < 1:
            raise ValidationError("CharNgramSparseEncoder.vocab_size must be >= 1")
        if int(min_n) < 1:
            raise ValidationError("CharNgramSparseEncoder.min_n must be >= 1")
        if int(max_n) < int(min_n):
            raise ValidationError("CharNgramSparseEncoder.max_n must be >= min_n")
        self._vocab_size = int(vocab_size)
        self._min_n = int(min_n)
        self._max_n = int(max_n)

    @property
    def vocab_size(self) -> int:
        """Hashing bucket cardinality (mirrors the other side)."""
        return self._vocab_size

    @staticmethod
    def _hash_gram(gram: str, vocab_size: int) -> int:
        """Stable gram -> bucket id in ``[0, vocab_size)`` via sha256 (doc + query share it)."""
        digest = hashlib.sha256(gram.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False) % vocab_size

    def _grams(self, text: str) -> List[str]:
        """Boundary-marked character n-grams across each ``[a-z0-9]+`` token.
        Each token is wrapped as ``^token$`` before slicing so prefix and suffix
        grams stay distinct from interior grams (standard fuzzy-match practice).
        :param text: str - Raw text (lowercased internally)
        :return: List[str] - All n-grams for n in [min_n, max_n]; empty when no token
        """
        if not text:
            return []
        grams: List[str] = []
        for match in _TOKEN_RE.finditer(str(text).lower()):
            marked = f"^{match.group(0)}$"
            length = len(marked)
            for n in range(self._min_n, self._max_n + 1):
                if n > length:
                    break
                for i in range(0, length - n + 1):
                    grams.append(marked[i:i + n])
        return grams

    def _aggregate(self, text: str) -> List[Tuple[int, float]]:
        """Compute sorted ``(bucket_id, weight)`` pairs with sub-linear TF.
        Grams colliding on the same bucket have their weights summed
        (collision-safe; identical policy on doc and query sides).
        """
        grams = self._grams(text)
        if not grams:
            return []
        tf: Dict[str, int] = {}
        for gram in grams:
            tf[gram] = tf.get(gram, 0) + 1
        bucket_weights: Dict[int, float] = {}
        for gram, count in tf.items():
            bid = self._hash_gram(gram, self._vocab_size)
            weight = 1.0 + math.log(1.0 + float(count))
            bucket_weights[bid] = bucket_weights.get(bid, 0.0) + weight
        return sorted(bucket_weights.items(), key=lambda kv: kv[0])

    def encode(self, text: str) -> Tuple[List[int], List[float]]:
        """Doc-side: character n-gram sparse ``(indices, values)`` for ``text``.
        :param text: str - Raw domain string (e.g. ``hi-rentals.io``)
        :return: Tuple[List[int], List[float]] - Sorted-ascending parallel lists; both empty when no gram survives
        """
        pairs = self._aggregate(text or "")
        return [int(b) for b, _ in pairs], [float(w) for _, w in pairs]

    def __call__(self, text: str) -> Any:
        """Query-side: encode ``text`` into a Qdrant ``SparseVector``.
        :param text: str - Raw or normalized query text
        :return: qdrant_client.models.SparseVector - Empty when no gram survives
        :raises RetrievalError: When qdrant-client is not installed
        """
        if _qm is None:
            raise RetrievalError("CharNgramSparseEncoder requires qdrant-client; install it or disable retrieval.qdrant.hybrid.ngram.enabled")
        indices, values = self.encode(text)
        return _qm.SparseVector(indices=indices, values=values)
