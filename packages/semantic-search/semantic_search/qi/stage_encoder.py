"""Per-stage Matryoshka encoder adapter for MRL dim cascade.

One underlying inference served at three lengths picked per stage:

    L1 router        -> 256
    Listing shortlist -> 512
    Re-rank head     -> 768

The ``MatryoshkaCascadeEncoder`` (``qi/cascade_encoder.py``) already
exposes ``encode_at_dim(text, dim)`` for one-call truncation. What was
missing — and what this module closes — is a thin adapter that lets
existing consumers (``SemanticRouter``, ``VectorRetriever``,
``QdrantHybridRetriever``) keep their ``Encoder``-protocol call sites
unchanged while transparently asking the cascade for the stage's
configured dim.

``StageEncoder`` implements the full ``Encoder`` protocol surface
(``dim`` property + ``encode`` + ``encode_batch``) but its ``dim`` is
pinned to one stage dim and every ``encode`` / ``encode_batch`` call
delegates to ``MatryoshkaCascadeEncoder.encode_at_dim`` /
``encode_batch_at_dims`` at that dim. Three side benefits:

1. Existing consumers see an ``Encoder``; their construction-time
   ``encoder.dim != config.embedding_dim`` check still applies — and
   it now enforces the stage-dim contract end-to-end.
2. The orchestrator's existing ``await asyncio.to_thread(encoder.encode, ...)``
   call sites do not move; the cascade just changes the answer's length.
3. Test mode (``cascade.enabled=false`` OR ``cascade.stage_dims`` absent)
   bypasses ``StageEncoder`` entirely — the registry wires the base
   encoder as today and consumers behave identically.

Layer rules: stdlib + ``core`` + sibling QI primitives only. Never
imports retrieval, orchestration, or registry code.
"""
from typing import List, Sequence

from semantic_search.core.exceptions import ValidationError
from semantic_search.core.logging_utils import get_logger
from semantic_search.qi.cascade_encoder import MatryoshkaCascadeEncoder
from semantic_search.qi.encoder import Encoder

logger = get_logger(__name__)

_STAGE_KEYS = frozenset({'router', 'shortlist', 'rerank'})


class StageEncoder(Encoder):
    """Encoder-protocol adapter pinned to one Matryoshka cascade dim.

    :param cascade: MatryoshkaCascadeEncoder - The shared cascade
        wrapper produced at registry boot. MUST already include the
        requested ``stage_dim`` in its ``supported_dims`` (the cascade
        constructor enforces this; passing a non-supported dim here
        will surface as ``ValidationError`` on the first encode call).
    :param stage_dim: int - The dim this adapter serves. MUST be a
        member of ``cascade.supported_dims`` AND ``>= 1``.
    :param stage_name: str - One of ``{'router', 'shortlist', 'rerank'}``.
        Logged on construction and surfaced via the ``stage_name`` property
        for ops dashboards / introspection.
    :param emit_init_log: bool - When True, emit ``stage_encoder_initialized`` at INFO (detailed startup); when False, skip (compact startup; registry still logs ``stage_encoders_wired``)
    :raises ValidationError: When ``cascade`` is not a
        ``MatryoshkaCascadeEncoder``, ``stage_dim`` is not int >= 1 or
        is outside ``cascade.supported_dims``, or ``stage_name`` is not
        in the allowed set.
    """

    def __init__(self, cascade: MatryoshkaCascadeEncoder, stage_dim: int, stage_name: str, emit_init_log: bool):
        if cascade is None or not isinstance(cascade, MatryoshkaCascadeEncoder):
            raise ValidationError("StageEncoder requires a non-None MatryoshkaCascadeEncoder cascade")
        if not isinstance(stage_dim, int) or isinstance(stage_dim, bool) or stage_dim < 1:
            raise ValidationError(f"StageEncoder.stage_dim must be int >= 1; got {stage_dim!r}")
        if stage_dim not in cascade.supported_dims:
            raise ValidationError(f"StageEncoder.stage_dim={stage_dim} not in cascade.supported_dims={sorted(cascade.supported_dims)}")
        if not isinstance(stage_name, str) or stage_name not in _STAGE_KEYS:
            raise ValidationError(f"StageEncoder.stage_name must be one of {sorted(_STAGE_KEYS)}; got {stage_name!r}")
        self._cascade = cascade
        self._stage_dim = int(stage_dim)
        self._stage_name = stage_name
        if emit_init_log:
            logger.info(f"stage_encoder_initialized stage={stage_name} dim={stage_dim} native_cascade={cascade.is_native_cascade}")

    @property
    def dim(self) -> int:
        """Pinned stage dim; equals ``cascade.encode_at_dim`` output length."""
        return self._stage_dim

    @property
    def stage_name(self) -> str:
        """Stage label this encoder serves (``router`` | ``shortlist`` | ``rerank``)."""
        return self._stage_name

    @property
    def cascade(self) -> MatryoshkaCascadeEncoder:
        """Underlying cascade (introspection / tests only — do NOT mutate)."""
        return self._cascade

    def encode(self, text: str) -> List[float]:
        """Encode one text at the pinned stage dim (delegates to ``encode_at_dim``).

        :param text: str - Input text. ``None`` / empty maps to a zero
            vector at ``stage_dim`` (delegated to the cascade).
        :return: List[float] - L2-normalised vector of length
            ``stage_dim`` (or, in cascade collapse mode, the base
            encoder's fixed-dim output — the cascade emits the
            one-time WARNING in that case).
        """
        return list(self._cascade.encode_at_dim(text, self._stage_dim))

    def encode_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """Encode a batch at the pinned stage dim (one inference per text).

        Delegates to ``MatryoshkaCascadeEncoder.encode_batch_at_dims`` with
        a single-element ``dims`` list so model invocations are batched
        even though only one dim is materialised. ``None`` input raises
        ``ValidationError`` per the cascade's contract; empty list
        returns an empty list (the cascade itself rejects empty
        ``dims``, never empty ``texts``).

        :param texts: Sequence[str] - Input texts. ``None`` raises;
            empty returns ``[]``. ``None`` / empty entries inside the
            sequence map to per-entry zero vectors at ``stage_dim``.
        :return: List[List[float]] - One vector per input text,
            preserving order; each of length ``stage_dim``.
        """
        if texts is None:
            raise ValidationError("StageEncoder.encode_batch requires a non-None texts sequence")
        materialised = list(texts)
        if not materialised:
            return []
        per_dim = self._cascade.encode_batch_at_dims(materialised, [self._stage_dim])
        return [list(v) for v in per_dim[self._stage_dim]]
