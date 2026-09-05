"""L1 semantic router — embedding centroids per query archetype.

Pre-computes a centroid per archetype from a `RouterSeedDataset` at construction.
At query time, encodes the query, scores cosine similarity vs every centroid, and
returns the top archetype iff top similarity >= confidence_threshold.
Otherwise returns None and the cascade falls through to L2 (LLM).

Learned-head dispatch:
- When ``config.learned_head`` is None or ``kind == 'centroid'``, the router
  uses the K-means centroid scorer (``_score_centroid``).
- For any other ``kind`` (``auto`` recommended), the router loads the trained
  artifact at ``model_path``. The concrete head type is detected at construction
  time by peeking the ``kind`` tag inside the .npz — NOT from ``config.kind`` — so
  the same config loads a logistic / svm / gradient_boosting / deep artefact and
  survives a retrain that swaps the algorithm. All heads implement the same
  ``score_archetypes(query_vec)`` interface; the confidence gate is identical.
  If ``config.kind`` names a specific algorithm that disagrees with the artefact,
  the router still loads the artefact and logs a mismatch warning.
  Reversible by flipping config back to ``centroid``.

Split of responsibilities:
- `RouterSeedLoader` (qi/seed_loader.py) owns *where* the seeds come from.
- `SemanticRouter` (this file) owns *how* the seeds become centroids /
  feed the learned head and *how* the per-archetype distribution is scored.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from semantic_search.config.models import QISemanticConfig
from semantic_search.contracts import IntentSlice, RouterSeedDataset
from semantic_search.core.exceptions import ConfigurationError
from semantic_search.core.logging_utils import get_logger
from llm_core.logging_utils import mask_path
from semantic_search.qi.encoder import Encoder, centroid, cosine_similarity, kmeans_spherical
from semantic_search.qi.training.deep_head_trainer import _ARTIFACT_KIND_DEEP, DeepHead
from semantic_search.qi.training.head_trainer import (
    _ARTIFACT_KIND,
    _ARTIFACT_KIND_GB,
    _ARTIFACT_KIND_SVM,
    _SKLEARN_HEAD_KINDS,
    LearnedHead,
    SklearnHead,
)

logger = get_logger(__name__)

# Maps the algorithm tag embedded in a head .npz artefact to its config-facing
# ``learned_head.kind`` name. Drives the startup cross-check that warns when the
# configured kind disagrees with the artefact actually on disk. Head dispatch
# itself never consults config.kind — it always reads this tag from the artefact.
_ARTEFACT_KIND_TO_CONFIG_NAME: Dict[str, str] = {
    _ARTIFACT_KIND: 'logistic',
    _ARTIFACT_KIND_SVM: 'svm',
    _ARTIFACT_KIND_GB: 'gradient_boosting',
    _ARTIFACT_KIND_DEEP: 'deep',
}


class SemanticRouter:
    """Route query → archetype via embedding-centroid similarity (L1 scorer)."""

    def __init__(self, config: QISemanticConfig, encoder: Encoder, seed_dataset: RouterSeedDataset):
        if encoder.dim != config.embedding_dim:
            raise ConfigurationError(f"SemanticRouter encoder.dim={encoder.dim} != config.embedding_dim={config.embedding_dim}")
        self._config = config
        self._encoder = encoder
        self._seed_dataset = seed_dataset
        self._centroid_exclusions: frozenset[str] = frozenset(config.centroid_exclusions)
        self._sub_centroids: Dict[str, Any] = self._build_centroids()
        if self._config.enabled and len(self._sub_centroids) == 0:
            raise ConfigurationError("SemanticRouter has no centroids after centroid_exclusions — relax qi.semantic.centroid_exclusions")
        self._learned_head = self._maybe_load_learned_head()
        _k = config.num_sub_centroids
        _actual_k = max(v.shape[0] for v in self._sub_centroids.values()) if self._sub_centroids else 0
        scoring_mode = 'learned' if self._learned_head is not None else 'centroid'
        logger.info(
            f"semantic_router_initialized archetypes={len(self._sub_centroids)} "
            f"dim={config.embedding_dim} k_requested={_k} k_actual={_actual_k} "
            f"scoring_mode={scoring_mode}"
        )

    def _maybe_load_learned_head(self):
        """Load LearnedHead if configured & valid, else None (fallback to centroid).

        When ``learned_head.strict`` is set and a head is configured (kind !=
        'centroid'), any degrade reason raises ConfigurationError instead of
        silently falling back — prevents prod from running the weak centroid L1
        path unnoticed.
        """
        learned_head_cfg = self._config.learned_head
        if learned_head_cfg is None or learned_head_cfg.kind == 'centroid':
            return None
        strict = bool(getattr(learned_head_cfg, 'strict', False))

        def _degrade(reason: str):
            """Raise when strict, else warn and fall back to centroid scorer."""
            if strict:
                raise ConfigurationError(
                    f"semantic_router_learned_head_unavailable reason={reason} "
                    f"kind={learned_head_cfg.kind} — strict mode forbids centroid fallback; "
                    f"fix qi.semantic.learned_head or set strict: false"
                )
            logger.warning(f"semantic_router_learned_head_{reason} — falling back to centroid scorer")
            return None

        # Collapsed cascade (non-Matryoshka): encoder.dim reports pinned dim but
        # builds its centroids from the same encoder and is dimensionally
        # self-consistent regardless of collapse.
        cascade = getattr(self._encoder, 'cascade', None)
        if cascade is not None and not getattr(cascade, 'is_native_cascade', True):
            logger.warning(f"semantic_router_learned_head_skipped reason=cascade_collapsed encoder_dim={self._encoder.dim} — base encoder is not Matryoshka-truncatable")
            return _degrade('skipped_cascade_collapsed')
        path = Path(learned_head_cfg.model_path)
        if not path.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            path = (project_root / path).resolve()
        if not path.exists():
            logger.warning(f"semantic_router_learned_head_missing path={mask_path(str(path))} kind={learned_head_cfg.kind}")
            return _degrade('missing')
        try:
            _peek = np.load(path, allow_pickle=False)
            _raw_kind = _peek['kind'].item() if hasattr(_peek['kind'], 'item') else _peek['kind']
            artefact_kind = _raw_kind.decode('utf-8') if isinstance(_raw_kind, bytes) else str(_raw_kind)
            if artefact_kind == _ARTIFACT_KIND_DEEP:
                head = DeepHead.load(path)
            elif artefact_kind in _SKLEARN_HEAD_KINDS:
                head = SklearnHead.load(path)
            else:
                head = LearnedHead.load(path)
        except Exception as exc:  # noqa: BLE001 — defensive boundary
            logger.warning(f"semantic_router_learned_head_load_failed path={mask_path(str(path))} error_type={type(exc).__name__} error={exc}")
            return _degrade('load_failed')
        # Cross-check: config.kind never drives dispatch (the artefact tag above does),
        # but if config names a specific algorithm that disagrees with the artefact on
        # disk, surface it — the config is stale/misleading even though load succeeded.
        _detected = _ARTEFACT_KIND_TO_CONFIG_NAME.get(artefact_kind, artefact_kind)
        _cfg_kind = learned_head_cfg.kind
        if _cfg_kind not in ('auto', _detected):
            logger.warning(
                f"semantic_router_learned_head_kind_mismatch configured={_cfg_kind} "
                f"artefact={_detected} — dispatch follows the artefact; set kind: auto to silence"
            )
        if head.encoder_dim != self._encoder.dim:
            logger.warning(f"semantic_router_learned_head_dim_mismatch head_dim={head.encoder_dim} encoder_dim={self._encoder.dim}")
            return _degrade('dim_mismatch')
        if head.metadata.total_samples < learned_head_cfg.min_seed_count:
            logger.warning(f"semantic_router_learned_head_below_min total_samples={head.metadata.total_samples} min_seed_count={learned_head_cfg.min_seed_count}")
            return _degrade('below_min')
        logger.info(
            f"semantic_router_learned_head_loaded path={mask_path(str(path))} "
            f"detected_kind={_detected} classes={head.classes} "
            f"f1_macro_holdout={head.metadata.f1_macro_holdout:.4f}"
        )
        return head

    def _build_centroids(self) -> Dict[str, "np.ndarray"]:
        """Encode prototype texts per archetype and compute K-means sub-centroids.

        Returns centroids as numpy arrays (shape: k × dim) for vectorized scoring.
        """
        sub_centroids: Dict[str, "np.ndarray"] = {}
        k = self._config.num_sub_centroids
        seed = self._config.encoder_seed
        for archetype in self._seed_dataset.archetypes():
            if archetype in self._centroid_exclusions:
                continue
            prototypes = self._seed_dataset.texts(archetype)
            vectors = self._encoder.encode_batch(prototypes)
            non_zero = [v for v in vectors if any(abs(x) > 0.0 for x in v)]
            if not non_zero:
                raise ConfigurationError(f"SemanticRouter archetype '{archetype}' produced only zero vectors — check seeds")
            centroid_list = kmeans_spherical(non_zero, k=k, seed=seed)
            sub_centroids[archetype] = np.array(centroid_list, dtype=np.float32)
        return sub_centroids

    @property
    def archetypes(self) -> List[str]:
        """Configured archetype names."""
        return list(self._sub_centroids.keys())

    @property
    def confidence_threshold(self) -> float:
        """Active confidence threshold from config."""
        return float(self._config.confidence_threshold)

    def swap_centroids(self, new_sub_centroids: Dict[str, "np.ndarray"]) -> None:
        """Replace live centroids for archetypes present in new_sub_centroids.

        Partial swap (subset of archetypes) is allowed; archetypes absent from
        new_sub_centroids retain their current centroids. Atomic under the CPython
        GIL — dict reference replacement is a single STORE_ATTR bytecode;
        ThreadPoolExecutor callers see either the old or new reference, never a partial state.

        :param new_sub_centroids: Dict[str, np.ndarray] - shape (k, dim) per archetype
        :raises ConfigurationError: unknown archetype or embedding_dim mismatch
        """
        merged = dict(self._sub_centroids)
        for arch, c in new_sub_centroids.items():
            if arch not in merged:
                raise ConfigurationError(f"SemanticRouter.swap_centroids archetype '{arch}' not in current archetypes")
            if c.shape[-1] != self._config.embedding_dim:
                raise ConfigurationError(f"SemanticRouter.swap_centroids dim mismatch arch={arch} got={c.shape[-1]} expected={self._config.embedding_dim}")
            merged[arch] = c
        self._sub_centroids = merged
        logger.info(f"semantic_router_centroids_swapped archetypes={sorted(new_sub_centroids.keys())} total_archetypes={len(merged)}")

    def _score_centroid(self, query_vec: List[float]) -> List[Tuple[str, float]]:
        """Vectorized scorer: max cosine over per-archetype K-means sub-centroids.

        Centroids are stored as (k, dim) numpy arrays. A single matrix-vector
        multiply replaces 16 pure-Python cosine-similarity loops per query.
        Vectors are already L2-normalised so dot product == cosine similarity.
        """
        q = np.asarray(query_vec, dtype=np.float32)
        scored: List[Tuple[str, float]] = []
        for archetype, centroids_np in self._sub_centroids.items():
            best = float((centroids_np @ q).max())
            scored.append((archetype, best))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored

    def _score_learned(self, query_vec: List[float]) -> List[Tuple[str, float]]:
        """Learned-head scorer: multinomial-logistic ``predict_proba``.

        Output is filtered through ``centroid_exclusions`` so the gating
        semantics match the centroid path exactly (an excluded archetype
        such as ``analytics`` is never returned as ``top_archetype``).
        """
        if self._learned_head is None:
            raise ConfigurationError("SemanticRouter._score_learned called without a learned head")
        raw = self._learned_head.score_archetypes(query_vec)
        filtered = [(a, s) for (a, s) in raw if a not in self._centroid_exclusions]
        return filtered

    def _score(self, query_vec: List[float]) -> List[Tuple[str, float]]:
        """Dispatch: learned head when available, otherwise centroids."""
        if self._learned_head is not None:
            return self._score_learned(query_vec)
        return self._score_centroid(query_vec)

    def best_guess(self, query: str) -> Optional[Tuple[str, float]]:
        """Return the top-scored archetype without applying the confidence gate.

        Used by QIEngine when L2 is unavailable and the L0_fallback fires:
        a sub-threshold L1 score is more informative than the configured
        default_query_type. Returns None when the router is disabled or the
        query encodes to a zero vector (degraded encoder path).

        :param query: str - Normalized query text.
        :return: Optional[Tuple[str, float]] - (archetype, score) for the top archetype, or None.
        """
        if not self._config.enabled or not query:
            return None
        vec = self._encoder.encode(query)
        if all(x == 0.0 for x in vec):
            return None
        scored = self._score(vec)
        if not scored:
            return None
        return scored[0]

    def classify(self, query: str) -> Optional[IntentSlice]:
        """Classify a query via embedding centroids.

        :param query: str - Normalized query text
        :return: Optional[IntentSlice] - Slice when confidence gate passes
        """
        result = self.classify_with_distribution(query)
        if result is None:
            return None
        slc, _scored = result
        return slc

    def classify_with_distribution(self, query: str) -> Optional[Tuple[IntentSlice, List[Tuple[str, float]]]]:
        """Classify a query and return the per-archetype score distribution.

        :param query: str - Normalized query text
        :return: Optional[(IntentSlice, List[(archetype, score)])] - Slice +
            descending-sorted score distribution when top_score >= confidence_threshold; None otherwise.
        """
        if not self._config.enabled:
            return None
        if not query:
            return None
        vec = self._encoder.encode(query)
        if all(x == 0.0 for x in vec):
            return None
        scored = self._score(vec)
        if not scored:
            return None
        top_archetype, top_score = scored[0]
        if top_score < self._config.confidence_threshold:
            return None
        confidence = max(0.0, min(1.0, top_score))
        slc = IntentSlice(query_type=top_archetype, entities=[], confidence=confidence, raw_text=query)
        return slc, scored
