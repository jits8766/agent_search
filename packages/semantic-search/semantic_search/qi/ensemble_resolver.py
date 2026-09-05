"""Ensemble voter resolver for query intent classification.

Multiple independent classifiers each cast a Vote for an archetype.
The resolver tallies weighted votes, applies optional veto-power overrides,
and derives a routing_mode from agreement ratio and winner confidence —
replacing single-classifier confidence bands.

All thresholds, weights, and voter identities come from QIEnsembleConfig;
no numeric literals appear in resolver logic.

Invariants:
- resolve() is pure: no I/O, no async, fully synchronous.
- All abstaining voters are excluded from both the tally and the agreement ratio.
- When all voters abstain the caller receives the configured fallback_archetype
  at routing_mode='explore'.
- Veto-voter archetype wins regardless of tally margin; highest-confidence
  veto voter wins when multiple veto voters disagree with the tally winner.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from semantic_search.config.models import QIEnsembleConfig
from semantic_search.core.exceptions import QueryIntelligenceError
from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Vote:
    """Ballot: voter_id + archetype + confidence (abstained → excluded from tally)."""
    voter_id: str
    archetype: str
    confidence: float
    abstained: bool = False

    def __post_init__(self) -> None:
        if not self.abstained:
            if not isinstance(self.voter_id, str) or not self.voter_id:
                raise QueryIntelligenceError("Vote.voter_id must be a non-empty string")
            if not isinstance(self.archetype, str) or not self.archetype:
                raise QueryIntelligenceError("Vote.archetype must be a non-empty string when not abstained")
            if not isinstance(self.confidence, (int, float)) or not 0.0 <= float(self.confidence) <= 1.0:
                raise QueryIntelligenceError(f"Vote.confidence must be in [0, 1]; got {self.confidence!r}")


def abstain(voter_id: str) -> Vote:
    """Abstaining ballot for voter_id."""
    return Vote(voter_id=voter_id, archetype='', confidence=0.0, abstained=True)


@dataclass(frozen=True)
class EnsembleResult:
    """Winner archetype + routing_mode + weight sums + agreement_ratio (veto override flagged)."""
    archetype: str
    routing_mode: str
    agreement_ratio: float
    winner_weight_sum: float
    total_weight_sum: float
    winner_confidence: float
    veto_applied: bool
    veto_voter_id: Optional[str]
    active_voter_ids: List[str]
    decision_tier: str


class EnsembleResolver:
    """Tally weighted votes; apply veto override; derive routing_mode from agreement."""

    def __init__(self, config: QIEnsembleConfig, query_types: 'frozenset[str]') -> None:
        if config is None:
            raise QueryIntelligenceError("EnsembleResolver requires a QIEnsembleConfig instance")
        if not query_types:
            raise QueryIntelligenceError("EnsembleResolver requires a non-empty query_types frozenset")
        self._config = config
        self._query_types = query_types
        # Index voter configs by voter_id for O(1) weight/veto lookup.
        self._voter_cfg: Dict[str, object] = {v.voter_id: v for v in config.voters}
        logger.info(
            f"ensemble_resolver_initialized voters={[v.voter_id for v in config.voters]} "
            f"veto_voters={[v.voter_id for v in config.voters if v.has_veto]} "
            f"consensus_cancel_l2={config.consensus_cancel_l2}"
        )

    def resolve(self, votes: List[Vote]) -> EnsembleResult:
        """Resolve a list of voter ballots into a single archetype decision.

        :param votes: List[Vote] - Ballots from all voters; may include abstentions.
        :return: EnsembleResult - Winner, routing_mode, agreement, veto metadata.
        :raises QueryIntelligenceError: When a non-abstaining vote carries an
            archetype not in query_types.
        """
        active = [v for v in votes if not v.abstained]

        # Validate all archetypes before any tally work.
        for v in active:
            if v.archetype not in self._query_types:
                raise QueryIntelligenceError(
                    f"ensemble_resolver_unknown_archetype voter_id={v.voter_id!r} "
                    f"archetype={v.archetype!r} valid={sorted(self._query_types)}"
                )

        if not active:
            logger.info(
                f"ensemble_resolver_all_abstain fallback_archetype={self._config.fallback_archetype}"
            )
            return EnsembleResult(
                archetype=self._config.fallback_archetype,
                routing_mode='explore',
                agreement_ratio=0.0,
                winner_weight_sum=0.0,
                total_weight_sum=0.0,
                winner_confidence=0.0,
                veto_applied=False,
                veto_voter_id=None,
                active_voter_ids=[],
                decision_tier='ensemble_all_abstain',
            )

        # Build weight map from config; voters absent from config get weight 1.0.
        weight_map: Dict[str, float] = {}
        for v in active:
            cfg = self._voter_cfg.get(v.voter_id)
            weight_map[v.voter_id] = float(cfg.weight) if cfg is not None else 1.0  # type: ignore[union-attr]

        total_weight = sum(weight_map[v.voter_id] for v in active)

        # Weighted vote tally per archetype.
        tally: Dict[str, float] = {}
        for v in active:
            w = weight_map[v.voter_id]
            tally[v.archetype] = tally.get(v.archetype, 0.0) + w * float(v.confidence)

        tally_winner = max(tally, key=lambda a: tally[a])
        veto_applied = False
        veto_voter_id: Optional[str] = None

        # Veto check: find veto-power voters whose ballot differs from tally winner.
        veto_candidates = [
            v for v in active
            if v.archetype != tally_winner
            and self._voter_cfg.get(v.voter_id) is not None
            and getattr(self._voter_cfg[v.voter_id], 'has_veto', False)
        ]
        if veto_candidates:
            # Highest confidence among disagreeing veto voters overrides.
            best_veto = max(veto_candidates, key=lambda v: float(v.confidence))
            winner = best_veto.archetype
            veto_applied = True
            veto_voter_id = best_veto.voter_id
            logger.info(
                f"qi_ensemble_veto voter_id={veto_voter_id!r} "
                f"winner_before={tally_winner!r} winner_after={winner!r} "
                f"veto_confidence={best_veto.confidence:.3f}"
            )
        else:
            winner = tally_winner

        # Agreement ratio: fraction of total active weight that voted for the winner.
        winner_weight_sum = sum(
            weight_map[v.voter_id] for v in active if v.archetype == winner
        )
        agreement_ratio = winner_weight_sum / total_weight if total_weight > 0.0 else 0.0

        # Winner confidence: weighted average confidence of winner-side voters.
        winner_voters = [v for v in active if v.archetype == winner]
        winner_w_total = sum(weight_map[v.voter_id] for v in winner_voters)
        if winner_w_total > 0.0:
            winner_confidence = sum(
                weight_map[v.voter_id] * float(v.confidence) for v in winner_voters
            ) / winner_w_total
        else:
            winner_confidence = 0.0

        routing_mode = self._derive_routing_mode(agreement_ratio, winner_confidence)
        active_voter_ids = [v.voter_id for v in active]
        _winner_voter_ids = {v.voter_id for v in active if v.archetype == winner}
        if 'llm' in _winner_voter_ids:
            decision_tier = 'L2_llm'
        elif 'semantic' in _winner_voter_ids:
            decision_tier = 'L1_semantic'
        else:
            decision_tier = 'L0_entity'

        logger.info(
            f"qi_ensemble_resolved winner={winner!r} routing_mode={routing_mode} "
            f"agreement={agreement_ratio:.3f} winner_confidence={winner_confidence:.3f} "
            f"veto_applied={veto_applied} active_voters={active_voter_ids} "
            f"tier={decision_tier}"
        )

        return EnsembleResult(
            archetype=winner,
            routing_mode=routing_mode,
            agreement_ratio=agreement_ratio,
            winner_weight_sum=winner_weight_sum,
            total_weight_sum=total_weight,
            winner_confidence=winner_confidence,
            veto_applied=veto_applied,
            veto_voter_id=veto_voter_id,
            active_voter_ids=active_voter_ids,
            decision_tier=decision_tier,
        )

    def _derive_routing_mode(self, agreement: float, winner_confidence: float) -> str:
        """Map agreement ratio and winner confidence to a routing mode.

        Both axes are checked; the most permissive band that either satisfies wins.
        All thresholds come from QIEnsembleRoutingConfig.

        :param agreement: float - Vote-weight agreement ratio in [0, 1].
        :param winner_confidence: float - Weighted average confidence of winner voters.
        :return: str - One of 'auto_execute', 'suggest', 'explore'.
        """
        r = self._config.routing
        if (agreement >= float(r.auto_execute_agreement_min)
                or winner_confidence >= float(r.auto_execute_confidence_min)):
            return 'auto_execute'
        if (agreement >= float(r.suggest_agreement_min)
                or winner_confidence >= float(r.suggest_confidence_min)):
            return 'suggest'
        return 'explore'
