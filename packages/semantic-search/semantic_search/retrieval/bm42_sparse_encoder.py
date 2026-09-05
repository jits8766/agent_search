"""BM42-backed sparse encoder — drop-in upgrade for the hash-BM25 sparse leg.

Wraps FastEmbed ``SparseTextEmbedding`` (Qdrant/bm42-all-minilm-l6-v2-attentions)
and exposes the same two call-sites the existing BM25 encoders serve:

- ``__call__(text) -> SparseVector``   — query side, mirrors ``BM25QueryEncoder.__call__``
- ``encode(tokens) -> (indices, values)`` — doc side, mirrors ``BM25DocEncoder.encode``

Both sides read the model from a pre-downloaded local directory; no network
access is ever attempted. The model is loaded eagerly on construction so any
missing-file error surfaces at startup, not on the first query.

Fallback contract: if the local path is absent or the ONNX runtime fails to
initialise, the caller (registry) catches ``ConfigurationError`` and falls
back to the existing hash-BM25 pair — the service keeps running unchanged.
"""
import os
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple

try:
    from fastembed import SparseTextEmbedding as _SparseTextEmbedding
except ImportError:
    _SparseTextEmbedding = None

try:
    from qdrant_client import models as _qm
except ImportError:
    _qm = None

from llm_core.logging_utils import mask_path

from semantic_search.core.exceptions import ConfigurationError
from semantic_search.core.logging_utils import get_logger

if TYPE_CHECKING:  # avoid runtime import cycle (config.models <-> retrieval)
    from semantic_search.config.models import BM42QueryStopListConfig

logger = get_logger(__name__)


class BM42SparseEncoder:
    """Attention-weighted sparse encoder backed by FastEmbed BM42.

    :param model_name: str - FastEmbed registry name (e.g. ``Qdrant/bm42-all-minilm-l6-v2-attentions``)
    :param local_model_path: str - Absolute path to the pre-downloaded model directory.
    :param threads: Optional[int] - ONNX-Runtime intra-op thread count; None = library default.
    :param log_local_path_at_info: bool - When True, log masked path at INFO; when False omit path line
    :param query_stop_list: Optional[BM42QueryStopListConfig] - Query-side
        navigational stop list. When set and ``enabled=true``, tokens whose
        lowercase form is in ``tokens`` are removed from the query string
        BEFORE it is passed to BM42's ``query_embed``. Doc-side encoding is
        unchanged so corpus integrity is preserved. ``None`` (or
        ``enabled=false``) preserves the legacy behavior.
    :raises ConfigurationError: When ``local_model_path`` is missing or model init fails.
    """

    def __init__(self, model_name: str, local_model_path: str, log_local_path_at_info: bool, threads: Optional[int] = None, query_stop_list: Optional['BM42QueryStopListConfig'] = None):
        local_model_path = os.path.expanduser(local_model_path) if local_model_path else local_model_path
        if not local_model_path:
            raise ConfigurationError("BM42SparseEncoder: sparse_encoder.local_model_path must be set in YAML config. No network download will be attempted.")
        if not os.path.isdir(local_model_path):
            raise ConfigurationError(
                f"BM42SparseEncoder: local_model_path directory not found: {mask_path(local_model_path)!r}. "
                "Pre-download the model to that path before starting the service. "
                "No network download will be attempted."
            )
        if _SparseTextEmbedding is None:
            raise ConfigurationError("BM42SparseEncoder requires the 'fastembed' package; install via 'uv pip install fastembed'")
        self._model_name = str(model_name)
        self._local_model_path = local_model_path
        # Query-side navigational stop list. Stored as a frozenset of
        # lowercased tokens for O(1) lookup. ``None`` means filtering is OFF
        # (legacy behavior). Doc-side ``encode`` is NEVER consulted against
        # this set so corpus integrity is preserved.
        self._query_stop_set: Optional[frozenset] = None
        if query_stop_list is not None and query_stop_list.enabled:
            self._query_stop_set = frozenset(t.lower() for t in query_stop_list.tokens)
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            self._model = _SparseTextEmbedding(model_name=self._model_name, threads=threads, lazy_load=False, specific_model_path=self._local_model_path, local_files_only=True)
        except (RuntimeError, OSError, ValueError) as exc:
            raise ConfigurationError(f"BM42SparseEncoder failed to initialise model={self._model_name} path={mask_path(self._local_model_path)!r} error={exc}") from exc
        if log_local_path_at_info:
            logger.info(f"bm42_sparse_encoder_initialised model={self._model_name} path={mask_path(self._local_model_path)!r}")

    # ------------------------------------------------------------------
    # Query side — drop-in for BM25QueryEncoder.__call__(text)
    # ------------------------------------------------------------------

    def __call__(self, text: str) -> Any:
        """Encode query text into a Qdrant ``SparseVector`` using BM42 attention weights.

        :param text: str - Raw or normalised query text
        :return: qdrant_client.models.SparseVector - Empty when no terms survive
        :raises ConfigurationError: When qdrant-client is not installed
        """
        if _qm is None:
            raise ConfigurationError("BM42SparseEncoder requires qdrant-client; install it or disable retrieval.qdrant.hybrid.bm25_enabled")
        if not text or not str(text).strip():
            return _qm.SparseVector(indices=[], values=[])
        query_text = str(text).strip()
        # Query-side stop list filter. When configured, navigational tokens
        # (e.g. "find", "search", "domain") are removed BEFORE BM42 sees the
        # text — they would otherwise leak attention mass into the sparse
        # vector and dilute the truly content-bearing tokens. Doc encoding is
        # untouched so the corpus index stays a faithful representation.
        if self._query_stop_set is not None:
            kept = [tok for tok in query_text.split() if tok.lower() not in self._query_stop_set]
            if not kept:
                return _qm.SparseVector(indices=[], values=[])
            query_text = " ".join(kept)
        result = next(iter(self._model.query_embed(query_text)))
        return _qm.SparseVector(indices=result.indices.tolist(), values=result.values.tolist())

    # ------------------------------------------------------------------
    # Doc side — drop-in for BM25DocEncoder.encode(tokens)
    # ------------------------------------------------------------------

    def encode(self, tokens: Sequence[str]) -> Tuple[List[int], List[float]]:
        """Encode a pre-tokenised token sequence into ``(indices, values)``.

        Joins the tokens into a single string (mirrors what the segmenter
        produced) and passes it through the BM42 passage encoder.

        :param tokens: Sequence[str] - Tokens from the domain-name segmenter
        :return: Tuple[List[int], List[float]] - Sorted-ascending parallel lists;
            both empty when the token sequence is empty.
        """
        if not tokens:
            return [], []
        text = " ".join(t for t in tokens if t and str(t).strip())
        if not text.strip():
            return [], []
        result = next(iter(self._model.embed(text)))
        return result.indices.tolist(), result.values.tolist()

    def encode_texts(
        self, texts: Sequence[str], *, batch_size: int
    ) -> List[Tuple[List[int], List[float]]]:
        """Encode already-joined passage texts via one batched BM42 ``embed`` call.

        Empty / whitespace-only entries return empty sparse pairs in place.
        ``batch_size`` is required (FastEmbed internal batch width); callers
        must pass the YAML-configured indexer value.

        :param texts: Sequence[str] - Passage strings (space-joined tokens)
        :param batch_size: int - Texts per FastEmbed embed micro-batch (>= 1)
        :return: List[Tuple[List[int], List[float]]] - One pair per input text
        :raises ConfigurationError: When ``batch_size`` is invalid
        """
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ConfigurationError("BM42SparseEncoder.encode_texts batch_size must be int >= 1")
        if not texts:
            return []
        out: List[Tuple[List[int], List[float]]] = [([], []) for _ in texts]
        non_empty_idx: List[int] = []
        non_empty_texts: List[str] = []
        for i, raw in enumerate(texts):
            text = str(raw).strip() if raw is not None else ""
            if text:
                non_empty_idx.append(i)
                non_empty_texts.append(text)
        if not non_empty_texts:
            return out
        for dest_i, result in zip(
            non_empty_idx,
            self._model.embed(non_empty_texts, batch_size=batch_size),
        ):
            out[dest_i] = (result.indices.tolist(), result.values.tolist())
        return out
