"""Reciprocal Rank Fusion (RRF) — combines candidates from multiple retrievers.
Score: fused(item) = sum_over_sources(weight_for_source * 1 / (k + rank_in_source))
Items contributed by more sources naturally rise.

When ``FusionConfig.weights`` is configured and enabled, ``weight_for_source`` is
resolved from the per-residual-kind profile keyed by ``QueryIntent.residual_kind``
(falling back to the ``default`` profile when the kind is None / unknown). When
weights are disabled or absent, every weight is 1.0 — byte-identical to the
classic symmetric RRF that this fuser shipped with.
"""
import time
from typing import Dict, List, Optional, Sequence, Tuple

from semantic_search.config.models import FusionConfig, FusionWeightsProfile
from semantic_search.core.exceptions import RetrievalError
from semantic_search.core.logging_utils import get_logger
from semantic_search.contracts import CandidateSet, RankedItem, RankedResults

logger = get_logger(__name__)


class RRFFuser:
    """RRF fuser: combines candidates (1/(k+rank) weighted per residual_kind)."""

    def __init__(self, config: FusionConfig):
        self._config = config

    def _resolve_profile(self, residual_kind: Optional[str]) -> Optional[FusionWeightsProfile]:
        """Resolve weight profile (falls back to 'default' on None/unknown)."""
        weights_cfg = self._config.weights
        if weights_cfg is None or not weights_cfg.enabled:
            return None
        if residual_kind is not None and residual_kind in weights_cfg.profiles:
            return weights_cfg.profiles[residual_kind]
        return weights_cfg.profiles['default']

    @staticmethod
    def _weight_for_source(profile: FusionWeightsProfile, source: str) -> float:
        """Map source → weight (unknown defaults 1.0)."""
        if source == 'vector':
            return float(profile.vector)
        if source == 'sparse':
            return float(profile.sparse)
        if source == 'structured':
            return float(profile.structured)
        if source == 'sql':
            return float(profile.sql)
        logger.warning(f"rrf_fusion_unknown_source source={source!r} defaulting weight=1.0 (profile has no field for this source)")
        return 1.0

    def fuse(self, request_id: str, candidate_sets: Sequence[CandidateSet], cache_hit: Optional[str], residual_kind: Optional[str] = None, top_n_override: Optional[int] = None) -> RankedResults:
        """Fuse candidates, return top-N ranked results (residual_kind selects profile)."""
        if not request_id:
            raise RetrievalError("RRFFuser.fuse requires non-empty request_id")
        t0 = time.monotonic()
        profile = self._resolve_profile(residual_kind)
        effective_top_n = top_n_override if top_n_override is not None else self._config.top_n

        # Single-source short-circuit. When exactly one candidate set contributed
        # any candidates, the RRF dict accumulation is wasted work — the rank
        # order is already correct and the formula 1/(k+rank) is monotonic in
        # rank for any positive weight. We materialize RankedItems directly so
        # the contract (fused_score value when weights are off, top_n cap,
        # contributing_sources list) is byte-identical to the general path;
        # only the intermediate dict allocations are skipped. When weights are
        # on, the single source's weight scales every contribution uniformly,
        # so the rank order is preserved (only absolute scores change).
        non_empty = [cs for cs in candidate_sets if cs.candidates]
        if len(non_empty) == 1:
            cs = non_empty[0]
            single_weight = self._weight_for_source(profile, cs.source) if profile is not None else 1.0
            total_candidates = len(cs.candidates)
            top_slice = cs.candidates[: effective_top_n]
            items_short: List[RankedItem] = []
            for rank_idx, cand in enumerate(top_slice):
                fused_score = single_weight * (1.0 / (float(self._config.rrf_k) + float(rank_idx + 1)))
                items_short.append(RankedItem(item_id=cand.item_id, fused_score=fused_score, contributing_sources=[cs.source], payload=dict(cand.payload)))
            latency_ms = (time.monotonic() - t0) * 1000.0
            logger.info(
                f"rrf_fusion request_id={request_id} sources=[{cs.source!r}] "
                f"fused_items={len(items_short)} total_in={total_candidates} "
                f"latency_ms={latency_ms:.2f} short_circuit=single_source "
                f"weighted_fusion={profile is not None} residual_kind={residual_kind}"
            )
            return RankedResults(request_id=request_id, items=items_short, total_candidates=total_candidates, fusion_latency_ms=latency_ms, cache_hit=cache_hit)

        scores: Dict[str, float] = {}
        sources_per_item: Dict[str, List[str]] = {}
        payload_per_item: Dict[str, Dict] = {}
        total_candidates = 0
        for cs in candidate_sets:
            source_weight = self._weight_for_source(profile, cs.source) if profile is not None else 1.0
            total_candidates += len(cs.candidates)
            for rank_idx, cand in enumerate(cs.candidates):
                contribution = source_weight * (1.0 / (float(self._config.rrf_k) + float(rank_idx + 1)))
                scores[cand.item_id] = scores.get(cand.item_id, 0.0) + contribution
                if cand.item_id not in sources_per_item:
                    sources_per_item[cand.item_id] = []
                if cs.source not in sources_per_item[cand.item_id]:
                    sources_per_item[cand.item_id].append(cs.source)
                if cand.item_id not in payload_per_item:
                    payload_per_item[cand.item_id] = {}
                payload_per_item[cand.item_id].update(cand.payload)
        ordered: List[Tuple[str, float]] = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        ordered = ordered[: effective_top_n]
        items: List[RankedItem] = []
        for item_id, fused_score in ordered:
            items.append(RankedItem(item_id=item_id, fused_score=float(fused_score), contributing_sources=list(sources_per_item.get(item_id, [])), payload=payload_per_item.get(item_id, {})))
        latency_ms = (time.monotonic() - t0) * 1000.0
        logger.info(
            f"rrf_fusion request_id={request_id} sources={[cs.source for cs in candidate_sets]} "
            f"fused_items={len(items)} total_in={total_candidates} latency_ms={latency_ms:.2f} "
            f"weighted_fusion={profile is not None} residual_kind={residual_kind}"
        )
        return RankedResults(request_id=request_id, items=items, total_candidates=total_candidates, fusion_latency_ms=latency_ms, cache_hit=cache_hit)
