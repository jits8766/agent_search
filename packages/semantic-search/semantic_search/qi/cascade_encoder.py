"""Matryoshka embedding cascade — one native vector, three serving lengths.

Serves the same logical
embedding at three sizes — 256 (router), 512 (shortlist), 768
(rerank) — from a single underlying inference call. The Matryoshka
training of `nomic-ai/nomic-embed-text-v1.5` makes this safe: any prefix
of the native 768-dim vector is itself a usable embedding, and
re-normalising in the truncated subspace preserves cosine semantics.

This module provides ``MatryoshkaCascadeEncoder`` — a thin wrapper over
any base ``Encoder`` that exposes ``encode_at_dim(text, dim)`` and
``encode_batch_at_dims(texts, dims)``. The wrapper:

- For a base ``FastEmbedEncoder`` configured at ``dim=768`` (native):
  calls ``base.encode(text)`` once to get the full 768-dim vector, then
  slices+renormalises into each requested dim — three call sites pay one
  inference cost with zero private-attribute access.
- For a base ``HashingEncoder`` (test / degraded mode): the cascade
  collapses — every requested dim returns the encoder's fixed-dim
  vector unchanged. A WARNING is logged once per (encoder, requested_dim)
  pair so test harnesses notice the silent fallback.

The wrapper does NOT replace the base ``Encoder`` interface; existing
callers that use ``encoder.encode(text)`` / ``encoder.encode_batch(texts)``
keep working with their configured dim. The cascade is purely additive
— consumers that want per-stage dimension control inject the wrapper
explicitly.

Layer rules: stdlib + ``core`` + sibling QI primitives only. Never
imports retrieval, orchestration, or registry code.
"""
import math
import threading
from typing import Dict, FrozenSet, List, Sequence

from semantic_search.core.exceptions import ValidationError
from semantic_search.core.logging_utils import get_logger
from semantic_search.qi.encoder import Encoder, FastEmbedEncoder

logger = get_logger(__name__)


def _l2_normalize(vec: Sequence[float]) -> List[float]:
    """L2-normalise an arbitrary-length vector. Zero vector returns itself."""
    norm = math.sqrt(sum(float(x) * float(x) for x in vec))
    if norm == 0.0:
        return [float(x) for x in vec]
    return [float(x) / norm for x in vec]


def _layer_norm_vec(vec: Sequence[float]) -> List[float]:
    """Layer-normalise (required for Matryoshka truncation in nomic models)."""
    fvec = [float(x) for x in vec]
    n = len(fvec)
    if n == 0:
        return fvec
    mean = sum(fvec) / n
    var = sum((x - mean) ** 2 for x in fvec) / n
    std = math.sqrt(var + 1e-5)
    return [(x - mean) / std for x in fvec]


class MatryoshkaCascadeEncoder:
    """One native vector, three serving lengths — Matryoshka cascade wrapper.

    :param base_encoder: Encoder - The underlying encoder. When it is a
        ``FastEmbedEncoder`` configured at ``dim=768`` (native), the
        cascade calls ``base.encode()`` once and slices into each
        requested dim. For any other encoder, the cascade collapses to
        the base encoder's fixed dim with a one-time WARNING log per
        requested dim mismatch.
    :param supported_dims: FrozenSet[int] - The dims callers are
        allowed to request. Must be non-empty and every entry must be
        ``>= 1``. Typical: ``frozenset({128, 256, 384, 768})``. Requesting
        a dim outside this set raises ``ValidationError`` so misconfigured
        downstream callers fail loudly rather than silently retrieving
        against the wrong-dim index.
    :raises ValidationError: When ``base_encoder`` is None or
        ``supported_dims`` is empty / contains invalid entries.
    """

    # Module-level dedup so the silent-fallback warning fires at most
    # once per (encoder_id, dim) tuple per process. Without this the log
    # gets spammed on every encode call in the hashing-encoder path.
    _warned: set = set()
    _warned_lock = threading.Lock()

    def __init__(self, base_encoder: Encoder, supported_dims: FrozenSet[int]):
        if base_encoder is None or not isinstance(base_encoder, Encoder):
            raise ValidationError("MatryoshkaCascadeEncoder requires a non-None base Encoder")
        if not isinstance(supported_dims, (frozenset, set, list, tuple)) or not supported_dims:
            raise ValidationError("MatryoshkaCascadeEncoder requires a non-empty supported_dims set")
        normalised: List[int] = []
        for d in supported_dims:
            if not isinstance(d, int) or d < 1:
                raise ValidationError("MatryoshkaCascadeEncoder.supported_dims entries must be int >= 1")
            normalised.append(int(d))
        self._base = base_encoder
        self._supported_dims: FrozenSet[int] = frozenset(normalised)
        # Native dim is the base encoder's configured dim. For FastEmbedEncoder
        # configured at dim=768, this is the full Matryoshka output — every
        # requested dim <= 768 can be sliced from a single encode() call.
        try:
            base_dim = int(getattr(base_encoder, 'dim'))
        except (AttributeError, TypeError, ValueError):
            base_dim = 0
        self._native_dim = base_dim if base_dim > 0 else 0
        self._is_native_cascade = isinstance(base_encoder, FastEmbedEncoder) and self._native_dim > 0
        # Propagate the layer_norm-before-truncation requirement from the base encoder.
        # nomic-embed-text-v1.5 needs layer_norm on the full native vector before any
        # Matryoshka slice; Snowflake arctic and others use plain slice+L2-normalize.
        self._layer_norm_before_truncation: bool = bool(getattr(base_encoder, 'requires_layer_norm_truncation', False))

    @property
    def supported_dims(self) -> FrozenSet[int]:
        """Dims callers may request via ``encode_at_dim``."""
        return self._supported_dims

    @property
    def is_native_cascade(self) -> bool:
        """True iff the base encoder produces a native vector we can slice.

        ``False`` indicates the wrapper is operating in collapse mode
        (e.g. a ``HashingEncoder`` base) — every requested dim returns
        the base encoder's fixed-dim output. Useful for ops dashboards
        + integration tests.
        """
        return self._is_native_cascade

    @property
    def native_dim(self) -> int:
        """Underlying native dimension (768 for FastEmbed, base.dim otherwise)."""
        return self._native_dim

    def _truncate_to(self, native_vec: Sequence[float], dim: int) -> List[float]:
        """Slice to dim and L2-renormalise (layer_norm if required by base encoder)."""
        if dim >= len(native_vec):
            return _l2_normalize(native_vec)
        vec: Sequence[float] = native_vec
        if self._layer_norm_before_truncation:
            vec = _layer_norm_vec(native_vec)
        return _l2_normalize(vec[:dim])

    def _warn_collapse_once(self, dim: int) -> None:
        """Emit silent-fallback WARNING once per (encoder, dim)."""
        key = (id(self._base), dim)
        with self._warned_lock:
            if key in self._warned:
                return
            self._warned.add(key)
        logger.warning(f"matryoshka_cascade_collapsed encoder={type(self._base).__name__} requested_dim={dim} base_dim={self._native_dim} reason=base_encoder_not_truncatable")

    def encode_at_dim(self, text: str, dim: int) -> List[float]:
        """Return the embedding of ``text`` at the requested ``dim``.
        :param text: str - Input text. ``None`` / empty returns a zero vector of length ``dim``.
        :param dim: int - Requested dimension. Must be in ``supported_dims``.
        :return: List[float] - L2-normalised embedding of length ``dim``
        :raises ValidationError: When ``dim`` is not in ``supported_dims``.
        """
        if dim not in self._supported_dims:
            raise ValidationError(f"MatryoshkaCascadeEncoder.encode_at_dim requested dim={dim} not in supported_dims={sorted(self._supported_dims)}")
        if not self._is_native_cascade or self._native_dim < dim:
            self._warn_collapse_once(dim)
            return list(self._base.encode(text))
        native_vec = list(self._base.encode(text))
        return self._truncate_to(native_vec, dim)

    def encode_batch_at_dims(self, texts: Sequence[str], dims: Sequence[int]) -> Dict[int, List[List[float]]]:
        """Encode ``texts`` once and return one list per requested dim.
        :param texts: Sequence[str] - Input texts. Must not be None. None / empty entries map to zero vectors.
        :param dims: Sequence[int] - Dims to materialise. Each MUST be in ``supported_dims``.
        :return: Dict[int, List[List[float]]] - dim -> list of vectors in input order.
        :raises ValidationError: When ``texts`` / ``dims`` is None / empty, or any dim is out-of-range.
        """
        if texts is None:
            raise ValidationError("encode_batch_at_dims requires a non-None texts sequence")
        if dims is None or not list(dims):
            raise ValidationError("encode_batch_at_dims requires at least one dim")
        seen_dims: List[int] = []
        for d in dims:
            if d not in self._supported_dims:
                raise ValidationError(f"encode_batch_at_dims dim={d} not in supported_dims={sorted(self._supported_dims)}")
            if d not in seen_dims:
                seen_dims.append(int(d))

        if not self._is_native_cascade:
            for d in seen_dims:
                self._warn_collapse_once(d)
            base_batch = list(self._base.encode_batch(list(texts)))
            return {d: [list(v) for v in base_batch] for d in seen_dims}

        native_batch = list(self._base.encode_batch(list(texts)))
        out: Dict[int, List[List[float]]] = {}
        for d in seen_dims:
            out[d] = [self._truncate_to(v, d) for v in native_batch]
        return out
