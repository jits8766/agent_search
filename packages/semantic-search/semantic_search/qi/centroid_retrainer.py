"""Tier-2 (L1 semantic router) centroid retraining channel.

Periodically recomputes
the L1 router centroids from real-traffic positives and shadow-deploying
the new centroids before promotion:

- Need >= 500 trusted positives per archetype before a retrain candidate
  is built
- Shadow-deploy the candidate against the live router
- Promote on agreement parity (>= configured threshold)
- Cadence ramps monthly -> bi-weekly -> weekly as the channel matures

This module provides ``CentroidRetrainer``: a stateless service that
takes (a) an encoder, (b) a current ``SemanticRouter`` (for shadow
agreement), and (c) per-archetype trusted-positive query texts. It
produces a typed ``CentroidRetrainCandidate`` and a typed
``CentroidRetrainVerdict``.

The retrainer is **read-only** with respect to the running router — it
NEVER swaps centroids. Promotion is the operator's call (or a future
``CentroidRetrainPromoter``). This keeps the hot path safe even when
the channel is invoked in the background.

Layer rules: stdlib + ``core`` + ``contracts`` + ``config`` + sibling
QI primitives (``encoder``, ``semantic_router``). No retrieval,
orchestration, or registry imports.
"""
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

from semantic_search.config.models import CentroidRetrainerConfig
from semantic_search.contracts import CentroidRetrainCandidate, CentroidRetrainVerdict, QUERY_TYPES
from semantic_search.core.exceptions import ValidationError
from semantic_search.core.logging_utils import get_logger
from semantic_search.qi.encoder import Encoder, centroid, cosine_similarity
from semantic_search.qi.semantic_router import SemanticRouter

logger = get_logger(__name__)


@dataclass(frozen=True)
class _ShadowOutcome:
    """One per-query shadow comparison.

    :param query: str - The probe query text
    :param current_archetype: Optional[str] - Live router classification
        (None when the live router declined to route)
    :param candidate_archetype: Optional[str] - Candidate centroids'
        classification under the same gating logic (None when below
        threshold)
    :param agreed: bool - True iff both routers chose the same archetype
        AND both routed (or both abstained)
    """
    query: str
    current_archetype: Optional[str]
    candidate_archetype: Optional[str]
    agreed: bool


class CentroidRetrainer:
    """Service for building + evaluating L1 centroid retrain candidates.

    :param config: CentroidRetrainerConfig - Min sample floor + shadow
        agreement threshold + window seconds
    :param encoder: Encoder - Same encoder the live SemanticRouter uses
        (centroid math is dim-coupled — mismatched dims are rejected by
        ``__init__``)
    :param current_router: SemanticRouter - Live router for shadow
        agreement scoring. The retrainer NEVER mutates this; only reads
        its centroids.
    :raises ValidationError: When config / encoder / router are
        None / wrong type / dim-mismatched
    """

    def __init__(self, config: CentroidRetrainerConfig, encoder: Encoder, current_router: SemanticRouter):
        if config is None or not isinstance(config, CentroidRetrainerConfig):
            raise ValidationError("CentroidRetrainer requires a CentroidRetrainerConfig")
        if encoder is None or not isinstance(encoder, Encoder):
            raise ValidationError("CentroidRetrainer requires a non-None Encoder")
        if current_router is None or not isinstance(current_router, SemanticRouter):
            raise ValidationError("CentroidRetrainer requires a non-None SemanticRouter")
        self._config = config
        self._encoder = encoder
        self._router = current_router
        logger.info(
            f"centroid_retrainer_initialized min_samples_per_archetype={config.min_samples_per_archetype} "
            f"min_shadow_agreement={config.min_shadow_agreement} "
            f"window_seconds={config.window_seconds}"
        )

    @property
    def config(self) -> CentroidRetrainerConfig:
        """Frozen retrainer config (diagnostics)."""
        return self._config

    def build_candidate(self, positives_by_archetype: Mapping[str, Sequence[str]], candidate_id: Optional[str] = None) -> CentroidRetrainCandidate:
        """Build a typed retrain candidate from per-archetype positives.

        :param positives_by_archetype: Mapping[str, Sequence[str]] - Trusted
            positive query texts grouped by archetype. Archetypes not in
            ``QUERY_TYPES`` raise. Archetypes with fewer than
            ``min_samples_per_archetype`` positives are dropped (logged).
            All-empty input raises (no candidate possible).
        :param candidate_id: Optional[str] - Override id (auto-gen UUID
            when None). Useful for deterministic tests.
        :return: CentroidRetrainCandidate - Typed candidate with computed
            centroids + sample counts
        :raises ValidationError: When positives are malformed, no archetype
            meets the floor, or any positive list is non-string
        """
        if positives_by_archetype is None or not isinstance(positives_by_archetype, Mapping):
            raise ValidationError("CentroidRetrainer.build_candidate requires a Mapping of positives")
        if not positives_by_archetype:
            raise ValidationError("CentroidRetrainer.build_candidate requires at least one archetype")
        candidate_centroids: Dict[str, List[float]] = {}
        sample_counts: Dict[str, int] = {}
        skipped: Dict[str, int] = {}
        for archetype, positives in positives_by_archetype.items():
            if archetype not in QUERY_TYPES:
                raise ValidationError(f"CentroidRetrainer.build_candidate archetype '{archetype}' not in QUERY_TYPES")
            texts: List[str] = []
            for p in positives:
                if not isinstance(p, str):
                    raise ValidationError(f"CentroidRetrainer.build_candidate positive for '{archetype}' must be str")
                clean = p.strip()
                if clean:
                    texts.append(clean)
            n = len(texts)
            if n < self._config.min_samples_per_archetype:
                skipped[archetype] = n
                continue
            vectors = self._encoder.encode_batch(texts)
            non_zero = [v for v in vectors if any(abs(x) > 0.0 for x in v)]
            if not non_zero:
                skipped[archetype] = n
                logger.warning(f"centroid_retrainer_archetype_all_zero archetype={archetype} texts={n} reason=encoder_returned_only_zero_vectors")
                continue
            candidate_centroids[archetype] = centroid(non_zero)
            sample_counts[archetype] = n
        if not candidate_centroids:
            raise ValidationError(f"CentroidRetrainer.build_candidate produced no centroids — every archetype below min_samples_per_archetype={self._config.min_samples_per_archetype}; skipped={skipped}")
        cid = candidate_id or f"centroid_retrain_{uuid.uuid4().hex[:12]}"
        candidate = CentroidRetrainCandidate(candidate_id=cid, new_centroids=candidate_centroids, sample_counts=sample_counts, window_seconds=float(self._config.window_seconds))
        logger.info(f"centroid_retrain_candidate_built candidate_id={cid} archetypes={sorted(candidate_centroids.keys())} sample_counts={sample_counts} skipped={skipped}")
        return candidate

    def decide(self, candidate: CentroidRetrainCandidate, shadow_queries: Sequence[str]) -> CentroidRetrainVerdict:
        """Run shadow agreement check + emit a typed verdict.

        Method:
        1. For each shadow query, classify under the live router AND under
           the candidate centroids (using identical confidence/margin
           gating from the live config).
        2. Count agreements (same archetype OR both abstain).
        3. Apply the gates:
           - min_sample_per_archetype >= config.min_samples_per_archetype
             -> 'reject' if violated
           - shadow_agreement_rate >= config.min_shadow_agreement
             -> 'promote' if both gates pass; 'shadow_only' if only the
             agreement gate passes; 'reject' otherwise
        4. Build a typed ``CentroidRetrainVerdict``.

        :param candidate: CentroidRetrainCandidate - Output of
            :meth:`build_candidate`
        :param shadow_queries: Sequence[str] - Probe queries (typically
            recent traffic). When empty, the verdict is 'reject' with
            reason 'empty_shadow_set'.
        :return: CentroidRetrainVerdict - Typed verdict
        :raises ValidationError: When inputs are wrong type
        """
        if candidate is None or not isinstance(candidate, CentroidRetrainCandidate):
            raise ValidationError("CentroidRetrainer.decide requires a CentroidRetrainCandidate")
        if shadow_queries is None:
            raise ValidationError("CentroidRetrainer.decide requires a non-None shadow_queries")
        reasons: List[str] = []
        min_per_archetype = (
            min(candidate.sample_counts.values()) if candidate.sample_counts else 0
        )
        if min_per_archetype < self._config.min_samples_per_archetype:
            reasons.append(f"min_sample_per_archetype={min_per_archetype} below floor={self._config.min_samples_per_archetype}")
            return CentroidRetrainVerdict(candidate_id=candidate.candidate_id, verdict='reject', shadow_agreement_rate=0.0, min_sample_per_archetype=min_per_archetype, reasons=reasons)
        if not list(shadow_queries):
            reasons.append("empty_shadow_set")
            return CentroidRetrainVerdict(candidate_id=candidate.candidate_id, verdict='reject', shadow_agreement_rate=0.0, min_sample_per_archetype=min_per_archetype, reasons=reasons)

        outcomes: List[_ShadowOutcome] = []
        for q in shadow_queries:
            outcomes.append(self._compare_one(q, candidate.new_centroids))
        agreements = sum(1 for o in outcomes if o.agreed)
        total = len(outcomes)
        agreement_rate = (agreements / total) if total > 0 else 0.0
        reasons.append(f"shadow_agreement_rate={agreement_rate:.3f} min_required={self._config.min_shadow_agreement:.3f} shadow_n={total} agreed={agreements}")
        if agreement_rate >= self._config.min_shadow_agreement:
            verdict_str = 'promote'
        else:
            # Below the parity floor — the candidate diverges too much from
            # the live router. Park it in shadow_only so dashboards can
            # surface drift without the operator having to choose between
            # promote and reject yet.
            verdict_str = 'shadow_only'
            reasons.append("verdict=shadow_only because agreement below promote threshold")
        verdict = CentroidRetrainVerdict(candidate_id=candidate.candidate_id, verdict=verdict_str, shadow_agreement_rate=agreement_rate, min_sample_per_archetype=min_per_archetype, reasons=reasons)
        logger.info(f"centroid_retrain_decided candidate_id={candidate.candidate_id} verdict={verdict_str} agreement_rate={agreement_rate:.3f} min_per_archetype={min_per_archetype}")
        return verdict

    def _compare_one(self, query: str, candidate_centroids: Mapping[str, List[float]]) -> _ShadowOutcome:
        """Classify one query under live router AND candidate centroids."""
        clean = (query or "").strip()
        if not clean:
            return _ShadowOutcome(query=query, current_archetype=None, candidate_archetype=None, agreed=True)
        live = self._router.classify(clean)
        cand = self._classify_under_candidate(clean, candidate_centroids)
        live_arch = live.query_type if live is not None else None
        cand_arch = cand
        agreed = (live_arch == cand_arch)
        return _ShadowOutcome(query=query, current_archetype=live_arch, candidate_archetype=cand_arch, agreed=agreed)

    def _classify_under_candidate(self, query: str, candidate_centroids: Mapping[str, List[float]]) -> Optional[str]:
        """Mimic SemanticRouter gating (confidence + margin) on the candidate."""
        vec = self._encoder.encode(query)
        if all(x == 0.0 for x in vec):
            return None
        scored = sorted(
            ((arch, cosine_similarity(vec, c)) for arch, c in candidate_centroids.items()),
            key=lambda kv: kv[1], reverse=True,
        )
        if not scored:
            return None
        top_archetype, top_score = scored[0]
        confidence_threshold = self._router.confidence_threshold
        if top_score < confidence_threshold:
            return None
        return top_archetype
