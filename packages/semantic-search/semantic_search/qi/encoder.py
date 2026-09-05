"""Text encoders for QI / vector retrieval / analytics semantic cache.

Three concrete implementations share the same `Encoder` protocol:

- `HashingEncoder` — deterministic, dependency-free, used for tests and the
  degraded-mode fallback.
- `FastEmbedEncoder` — production encoder. Supports any FastEmbed-compatible
  Matryoshka model configured via `qi.encoder.model_name`. The Matryoshka
  cascade is implemented by truncating the native vector to the configured
  `dim` and re-normalising. Supports a `query_prefix` for task-type separation
  (e.g. `"search_query: "` at query time, `"search_document: "` at index time).
- `BatchingEncoder` — async coalescing wrapper; see class docstring.

Composition-root selection:

The registry chooses the implementation based on `qi.encoder.backend`
(`fastembed` or `hashing`). Either way, every consumer
(`SemanticRouter`, `InMemoryVectorIndex`,
`QISemanticIntentCache`, `QdrantVectorIndex`) receives an object whose
public surface is `encode(text)` + `encode_batch(texts)` returning
`List[float]` with L2-normalised output. No caller changes.
"""
import asyncio
import hashlib
import math
import os
import random
import re
from typing import List, Optional, Sequence, Tuple

try:
    import numpy as _np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

try:
    from fastembed import TextEmbedding as _TextEmbedding
except ImportError:
    _TextEmbedding = None

from llm_core.logging_utils import mask_path

from semantic_search.core.exceptions import ConfigurationError, ValidationError
from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")


class Encoder:
    """Encode text → L2-normalized vector."""

    def encode(self, text: str) -> List[float]:
        """Single text → vector."""
        raise NotImplementedError

    def encode_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """Batch texts → vectors (sequential by default)."""
        return [self.encode(t) for t in texts]

    async def encode_async(self, text: str) -> List[float]:
        """Async encode (thread pool default; override in BatchingEncoder for coalescing)."""
        return await asyncio.to_thread(self.encode, text)


class HashingEncoder(Encoder):
    """Token bag → sha256-bucketed vector (deterministic, seed-based)."""

    def __init__(self, dim: int, seed: int):
        if dim < 4:
            raise ValidationError("HashingEncoder.dim must be >= 4")
        self._dim = int(dim)
        self._seed = int(seed)

    @property
    def dim(self) -> int:
        """Output vector dimension."""
        return self._dim

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Lowercase alphanumeric tokens."""
        if text is None:
            return []
        return [t.lower() for t in _TOKEN_PATTERN.findall(text)]

    def _bucket(self, token: str) -> int:
        """Token → bucket index [0, dim)."""
        digest = hashlib.sha256(f"{self._seed}|{token}".encode('utf-8')).hexdigest()
        return int(digest[:8], 16) % self._dim

    def encode(self, text: str) -> List[float]:
        """Text → L2-normalized vector (empty/None → zero vector)."""
        vec = [0.0] * self._dim
        tokens = self._tokenize(text)
        if not tokens:
            return vec
        for tok in tokens:
            vec[self._bucket(tok)] += 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0.0:
            return vec
        return [x / norm for x in vec]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Two vectors → cosine similarity [-1,1] (raises on len mismatch)."""
    if len(a) != len(b):
        raise ValidationError(f"cosine_similarity dimension mismatch len_a={len(a)} len_b={len(b)}")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def centroid(vectors: Sequence[Sequence[float]]) -> List[float]:
    """Compute the L2-normalized mean vector of a set of vectors.
    :param vectors: Sequence[Sequence[float]] - Input vectors (must share length)
    :return: List[float] - Normalized centroid (zero vector when input empty)
    :raises ValidationError: When vectors have inconsistent lengths
    """
    if len(vectors) == 0:
        return []
    dim = len(vectors[0])
    acc = [0.0] * dim
    for v in vectors:
        if len(v) != dim:
            raise ValidationError(f"centroid dimension mismatch expected={dim} got={len(v)}")
        for i, x in enumerate(v):
            acc[i] += x
    n = float(len(vectors))
    mean = [x / n for x in acc]
    norm = math.sqrt(sum(x * x for x in mean))
    if norm == 0.0:
        return mean
    return [x / norm for x in mean]


def _l2_normalize(vec: List[float]) -> List[float]:
    """L2-normalise a single vector. Zero vectors are returned unchanged."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]



def kmeans_spherical(vectors: List[List[float]], k: int, seed: int = 0, max_iter: int = 50) -> List[List[float]]:
    """Spherical K-means with k-means++ initialisation on L2-normalised vectors.

    Each centroid update is L2-normalised so assignment always uses cosine
    similarity — correct for unit-sphere embeddings.  When k >= len(vectors)
    or k <= 1 the degenerate single-centroid mean is returned; callers can
    treat `len(result) == 1` as the single-centroid path.

    :param vectors: List[List[float]] - L2-normalised input vectors (same dim)
    :param k: int - Desired number of sub-centroids
    :param seed: int - RNG seed for reproducibility
    :param max_iter: int - Maximum EM iterations
    :return: List[List[float]] - Up to k L2-normalised centroids
    """
    if not vectors:
        return []
    if k <= 1 or len(vectors) <= k:
        return [centroid(vectors)]

    rng = random.Random(seed)

    # k-means++ initialisation — first centre uniform, rest D² weighted.
    centres: List[List[float]] = [list(vectors[rng.randrange(len(vectors))])]
    while len(centres) < k:
        if _NUMPY_AVAILABLE and len(vectors) > 100:
            v_arr = _np.array(vectors, dtype=_np.float32)
            c_arr = _np.array(centres, dtype=_np.float32)
            sims = v_arr @ c_arr.T
            best_sims = _np.max(sims, axis=1)
            distances = _np.maximum(0.0, 1.0 - best_sims).astype(_np.float32)
        else:
            distances: List[float] = []
            for v in vectors:
                best = max(cosine_similarity(v, c) for c in centres)
                distances.append(max(0.0, 1.0 - best))
            distances = _np.array(distances, dtype=_np.float32) if _NUMPY_AVAILABLE else distances
        total = float(_np.sum(distances)) if isinstance(distances, _np.ndarray) else sum(distances)
        if total == 0.0:
            break
        r = rng.random() * total
        cumulative = 0.0
        chosen = vectors[-1]
        if isinstance(distances, _np.ndarray):
            cumsum = _np.cumsum(distances)
            idx = int(_np.searchsorted(cumsum, r))
            idx = min(idx, len(vectors) - 1)
            chosen = vectors[idx]
        else:
            for v, d in zip(vectors, distances):
                cumulative += d
                if cumulative >= r:
                    chosen = v
                    break
        centres.append(list(chosen))

    # EM iterations.
    for _ in range(max_iter):
        clusters: List[List[List[float]]] = [[] for _ in range(len(centres))]
        if _NUMPY_AVAILABLE and len(vectors) > 100:
            v_arr = _np.array(vectors, dtype=_np.float32)
            c_arr = _np.array(centres, dtype=_np.float32)
            sims = v_arr @ c_arr.T
            best_idxs = _np.argmax(sims, axis=1)
            for v_idx, c_idx in enumerate(best_idxs):
                clusters[int(c_idx)].append(vectors[v_idx])
        else:
            for v in vectors:
                best_idx = max(range(len(centres)), key=lambda i: cosine_similarity(v, centres[i]))
                clusters[best_idx].append(v)

        next_centres: List[List[float]] = []
        changed = False
        for i, cluster in enumerate(clusters):
            if not cluster:
                next_centres.append(centres[i])
                continue
            next_c = centroid(cluster)
            if any(abs(a - b) > 1e-8 for a, b in zip(next_c, centres[i])):
                changed = True
            next_centres.append(next_c)
        centres = next_centres
        if not changed:
            break

    return centres


# Native output dimension shared by supported models; Matryoshka truncation
# to the configured dim is implemented by slice + L2-renormalize.
_FASTEMBED_NATIVE_DIM = 1024


class FastEmbedEncoder(Encoder):
    """ONNX-Runtime encoder backed by FastEmbed.

    Supports any FastEmbed-compatible Matryoshka model. The model is selected
    via ``qi.encoder.model_name`` in config. The Matryoshka cascade is
    implemented by truncating the native vector to the configured ``dim``
    and re-L2-normalising.

    :param model_name: str - HuggingFace model id
    :param dim: int - Output vector dimension (must be one of {64, 128, 256, 384, 768})
    :param query_prefix: str - Text prepended to every input before encoding.
        Use ``"search_query: "`` at query/routing time;
        use ``"search_document: "`` during vectorization/data-rebuild runs.
        Empty string (default) disables prefixing.
    :param cache_dir: Optional[str] - FastEmbed model cache directory; None = default user cache
    :param threads: Optional[int] - ONNX-Runtime intra-op thread count; None = library default
    :param max_length: int - Token budget for the pre-embed raw-text cap (``max_length * 8`` code points); tokenizer truncation follows the model bundle
    :param batch_size: int - Internal FastEmbed batch size for `embed()` calls
    :param log_local_path_at_info: bool - When True, include masked local model path in INFO init log; when False, omit path
    :raises ConfigurationError: When fastembed is not installed or `dim` is not a Matryoshka step
    """

    # Allowed Matryoshka cascade steps (64 = minimum supported, 768 = native).
    # Out-of-list dims are rejected so we never silently produce an untrained projection.
    _ALLOWED_DIMS = frozenset({64, 128, 256, 384, 768, 1024})

    def __init__(self, model_name: str, dim: int, local_model_path: str, threads: Optional[int], max_length: int, batch_size: int, log_local_path_at_info: bool, query_prefix: str = ""):
        if dim not in self._ALLOWED_DIMS:
            raise ConfigurationError(f"FastEmbedEncoder.dim must be one of {sorted(self._ALLOWED_DIMS)}; got {dim}")
        if dim > _FASTEMBED_NATIVE_DIM:
            raise ConfigurationError(f"FastEmbedEncoder.dim {dim} exceeds native dim {_FASTEMBED_NATIVE_DIM}")
        if max_length < 1:
            raise ConfigurationError(f"FastEmbedEncoder.max_length must be >= 1; got {max_length}")
        if batch_size < 1:
            raise ConfigurationError(f"FastEmbedEncoder.batch_size must be >= 1; got {batch_size}")
        if _TextEmbedding is None:
            raise ConfigurationError("FastEmbedEncoder requires the 'fastembed' package; install via 'uv pip install fastembed'")
        self._model_name = str(model_name)
        self._model_name_lower = self._model_name.lower()
        self._dim = int(dim)
        self._local_model_path = os.path.expanduser(str(local_model_path)) if local_model_path else ""
        self._threads = threads
        self._max_length = int(max_length)
        self._batch_size = int(batch_size)
        self._query_prefix = str(query_prefix)
        # nomic-embed-text-v1.5 requires layer_norm across the full native vector
        # before any Matryoshka slice. MatryoshkaCascadeEncoder reads this via getattr.
        self._layer_norm_before_truncation = 'nomic' in self._model_name_lower
        self.requires_layer_norm_truncation = self._layer_norm_before_truncation
        if not self._local_model_path:
            raise ConfigurationError("FastEmbedEncoder: qi.encoder.local_model_path must be set in the YAML config. No network download will be attempted.")
        if not os.path.isdir(self._local_model_path):
            raise ConfigurationError(
                f"FastEmbedEncoder: local_model_path directory not found: {mask_path(self._local_model_path)!r}. "
                "Pre-download the model to that path before starting the service. "
                "No network download will be attempted."
            )
        # Block all HuggingFace Hub network calls for this process. Combined
        # with specific_model_path + local_files_only this gives three
        # independent layers; any one of them is sufficient to prevent a
        # download, all three together make the guarantee unconditional.
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            self._model = _TextEmbedding(model_name=self._model_name, threads=self._threads, lazy_load=False, specific_model_path=self._local_model_path, local_files_only=True)
        except (RuntimeError, OSError, ValueError) as e:
            raise ConfigurationError(f"FastEmbedEncoder failed to initialise model={self._model_name} path={mask_path(self._local_model_path)!r} error={str(e)}") from e
        _path_part = f"path={mask_path(self._local_model_path)!r} " if log_local_path_at_info else ""
        logger.info(
            f"fastembed_encoder_initialised model={self._model_name} {_path_part}"
            f"dim={self._dim} native_dim={_FASTEMBED_NATIVE_DIM} max_length={self._max_length} "
            f"batch_size={self._batch_size} query_prefix={self._query_prefix!r}"
        )

    @property
    def dim(self) -> int:
        """Output vector dimension after Matryoshka truncation."""
        return self._dim

    @property
    def model_name(self) -> str:
        """Underlying HuggingFace model id."""
        return self._model_name

    @property
    def query_prefix(self) -> str:
        """Configured task-type prefix prepended to every input."""
        return self._query_prefix

    def _truncate_and_renormalize(self, native_vec) -> List[float]:
        """Truncate native vector to `self._dim` and L2-renormalise.

        Accepts a numpy array (FastEmbed output) or any sequence.
        """
        if _NUMPY_AVAILABLE:
            arr = _np.asarray(native_vec, dtype=_np.float32)
            if self._dim < arr.shape[0]:
                arr = arr[:self._dim]
            norm = float(_np.linalg.norm(arr))
            if norm > 0.0:
                arr = arr / norm
            return arr.tolist()
        vec = [float(x) for x in native_vec]
        if self._dim < len(vec):
            vec = vec[: self._dim]
        return _l2_normalize(vec)

    def encode(self, text: str) -> List[float]:
        """Encode a single text into a Matryoshka-truncated, L2-normalised vector.
        :param text: str - Input text (None / empty returns a zero vector of length `dim`)
        :return: List[float] - Embedding of length `dim`
        """
        if text is None:
            return [0.0] * self._dim
        clean = str(text).strip()
        if not clean:
            return [0.0] * self._dim
        if self._query_prefix:
            clean = self._query_prefix + clean
        if len(clean) > self._max_length * 8:
            # Hard pre-truncation guard — FastEmbed truncates at the tokenizer
            # per the ONNX bundle; we cap raw text here to bound tokenizer-side
            # memory on adversarially long inputs.
            clean = clean[: self._max_length * 8]
        try:
            results = list(self._model.embed([clean], batch_size=1))
        except (RuntimeError, ValueError) as e:
            raise ValidationError(f"FastEmbedEncoder.encode failed model={self._model_name} error={str(e)}") from e
        if len(results) != 1:
            raise ValidationError(f"FastEmbedEncoder.encode expected 1 result; got {len(results)}")
        return self._truncate_and_renormalize(results[0])

    def encode_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """Encode a batch of texts deterministically in input order.
        :param texts: Sequence[str] - Input texts (None / empty entries map to zero vectors)
        :return: List[List[float]] - Embeddings, length-aligned with input
        """
        if texts is None:
            return []
        out: List[List[float]] = []
        # Partition into encodable (non-empty) and zero (None / empty) so the
        # result preserves input order without invoking the model on degenerate
        # rows (avoids tokenizer warnings on empty strings + saves compute).
        encodable_indices: List[int] = []
        encodable_texts: List[str] = []
        for i, t in enumerate(texts):
            if t is None:
                continue
            clean = str(t).strip()
            if not clean:
                continue
            if self._query_prefix:
                clean = self._query_prefix + clean
            if len(clean) > self._max_length * 8:
                clean = clean[: self._max_length * 8]
            encodable_indices.append(i)
            encodable_texts.append(clean)
        # Allocate result with zero-vector defaults for None / empty inputs.
        for _ in range(len(texts)):
            out.append([0.0] * self._dim)
        if not encodable_texts:
            return out
        try:
            results = list(self._model.embed(encodable_texts, batch_size=self._batch_size))
        except (RuntimeError, ValueError) as e:
            raise ValidationError(f"FastEmbedEncoder.encode_batch failed model={self._model_name} error={str(e)}") from e
        if len(results) != len(encodable_texts):
            raise ValidationError(f"FastEmbedEncoder.encode_batch result count mismatch expected={len(encodable_texts)} got={len(results)}")
        for idx, vec in zip(encodable_indices, results):
            out[idx] = self._truncate_and_renormalize(vec)
        return out


class PrefixedEncoder(Encoder):
    """Wraps any Encoder and prepends a fixed task-type prefix to every input.

    Enables model-agnostic prefix separation (``"search_query: "`` vs
    ``"search_document: "``) without loading the underlying model twice.
    The search pipeline injects one prefix; the vectorization pipeline
    injects another — both share the same ONNX runtime instance.

    :param base: Encoder - The underlying encoder (HashingEncoder or FastEmbedEncoder).
    :param prefix: str - Task-type prefix prepended to non-empty, non-None inputs.
        Empty string is a no-op (all calls pass through directly).
    :raises ConfigurationError: When ``prefix`` is not a string.
    """

    def __init__(self, base: Encoder, prefix: str):
        if not isinstance(prefix, str):
            raise ConfigurationError("PrefixedEncoder.prefix must be a string")
        self._base = base
        self._prefix = prefix

    @property
    def dim(self) -> int:
        """Output dimension delegated to the base encoder."""
        return getattr(self._base, 'dim', 0)

    @property
    def prefix(self) -> str:
        """Task-type prefix applied to every non-empty input."""
        return self._prefix

    def encode(self, text: str) -> List[float]:
        """Encode a single text, prepending the configured prefix when non-empty.
        :param text: str - Input text; None / empty bypasses prefix logic.
        :return: List[float] - L2-normalised embedding from the base encoder.
        """
        if not self._prefix or text is None:
            return self._base.encode(text)
        clean = str(text).strip()
        if not clean:
            return self._base.encode(text)
        return self._base.encode(self._prefix + clean)

    def encode_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """Encode a batch, prepending the configured prefix to every non-empty entry.
        :param texts: Sequence[str] - Input texts; None / empty entries bypass prefix.
        :return: List[List[float]] - Embeddings in input order.
        """
        if not self._prefix or texts is None:
            return self._base.encode_batch(texts)
        prefixed = []
        for t in texts:
            if t is None:
                prefixed.append(t)
            else:
                clean = str(t).strip()
                prefixed.append((self._prefix + clean) if clean else t)
        return self._base.encode_batch(prefixed)


class BatchingEncoder(Encoder):
    """Coalesces concurrent async encode calls into encode_batch within a time window.

    Multiple concurrent callers each call encode_async(); requests arriving
    within window_ms are batched together and dispatched as a single
    encode_batch() call on the underlying encoder. Reduces ONNX inference
    overhead under concurrent load from N separate single-item calls to
    one batch call.

    The drain loop starts lazily on the first encode_async call and restarts
    automatically if the task completes (e.g. after event-loop recreation).

    :param base: Encoder - Underlying encoder (typically FastEmbedEncoder).
    :param window_ms: float - Batching window in milliseconds. Must be > 0.
    :raises ConfigurationError: When window_ms is not positive.
    """

    def __init__(self, base: Encoder, window_ms: float):
        if window_ms <= 0:
            raise ConfigurationError(f"BatchingEncoder.window_ms must be > 0; got {window_ms}")
        self._base = base
        self._window = window_ms / 1000.0
        self._queue: asyncio.Queue = asyncio.Queue()
        self._drain_task: Optional[asyncio.Task] = None

    @property
    def dim(self) -> int:
        return getattr(self._base, 'dim', 0)

    def encode(self, text: str) -> List[float]:
        """Sync encode — delegates to base for SemanticRouter compatibility."""
        return self._base.encode(text)

    def encode_batch(self, texts: Sequence[str]) -> List[List[float]]:
        return self._base.encode_batch(texts)

    async def encode_async(self, text: str) -> List[float]:
        """Async encode — coalesces into a batch within window_ms.
        :param text: str - Input text
        :return: List[float] - L2-normalised embedding
        """
        if self._drain_task is None or self._drain_task.done():
            self._drain_task = asyncio.create_task(self._drain_loop())
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        await self._queue.put((text, fut))
        return await fut

    async def _drain_loop(self) -> None:
        """Background loop: collect items within window_ms then batch-encode."""
        while True:
            first_text, first_fut = await self._queue.get()
            batch: List[Tuple[str, asyncio.Future]] = [(first_text, first_fut)]
            deadline = asyncio.get_event_loop().time() + self._window
            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break
            texts = [t for t, _ in batch]
            try:
                vecs = await asyncio.to_thread(self._base.encode_batch, texts)
                for (_, fut), vec in zip(batch, vecs):
                    if not fut.done():
                        fut.set_result(vec)
            except Exception as exc:
                for _, fut in batch:
                    if not fut.done():
                        fut.set_exception(exc)
