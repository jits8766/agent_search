"""Unit tests for EnsembleResolver and Vote.

Coverage matrix:

EnsembleResolver.resolve:
- unanimous_vote                  -> TestEnsembleResolver::test_unanimous_vote
- majority_vote_wins              -> TestEnsembleResolver::test_majority_vote_wins
- veto_overrides_majority         -> TestEnsembleResolver::test_veto_overrides_majority
- abstain_excluded_from_tally     -> TestEnsembleResolver::test_abstain_excluded_from_tally
- all_abstain_returns_fallback    -> TestEnsembleResolver::test_all_abstain_returns_fallback
- unknown_archetype_raises        -> TestEnsembleResolver::test_unknown_archetype_raises

EnsembleResolver._derive_routing_mode:
- routing_mode_auto_execute       -> TestEnsembleResolver::test_routing_mode_auto_execute
- routing_mode_suggest            -> TestEnsembleResolver::test_routing_mode_suggest
- routing_mode_explore            -> TestEnsembleResolver::test_routing_mode_explore
- routing_mode_confidence_axis    -> TestEnsembleResolver::test_routing_mode_confidence_axis

Vote:
- vote_validation_rejects_empty   -> TestVote::test_vote_validation_rejects_empty_voter_id
- abstain_helper                  -> TestVote::test_abstain_helper
"""
import pytest

from semantic_search.config.models import (
    QIEnsembleConfig,
    QIEnsembleRoutingConfig,
    QIEnsembleVoterConfig,
)
from semantic_search.core.exceptions import QueryIntelligenceError
from semantic_search.qi.ensemble_resolver import EnsembleResolver, Vote, abstain

# ---------------------------------------------------------------------------
# Shared query_types
# ---------------------------------------------------------------------------

_QUERY_TYPES: frozenset = frozenset({'hybrid', 'guidance', 'explore', 'analytics'})


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _routing(
    auto_exec_agr: float = 0.75,
    auto_exec_conf: float = 0.80,
    suggest_agr: float = 0.50,
    suggest_conf: float = 0.60,
) -> QIEnsembleRoutingConfig:
    return QIEnsembleRoutingConfig(
        auto_execute_agreement_min=auto_exec_agr,
        auto_execute_confidence_min=auto_exec_conf,
        suggest_agreement_min=suggest_agr,
        suggest_confidence_min=suggest_conf,
    )


def _voter(voter_id: str, weight: float = 1.0, has_veto: bool = False) -> QIEnsembleVoterConfig:
    return QIEnsembleVoterConfig(
        voter_id=voter_id,
        weight=weight,
        has_veto=has_veto,
        abstain_on_no_signal=True,
        timeout_ms=0.0,
    )


def _cfg(
    voters,
    routing=None,
    fallback: str = 'explore',
    cancel_l2: bool = False,
    cancel_threshold: float = 1.0,
) -> QIEnsembleConfig:
    return QIEnsembleConfig(
        voters=voters,
        routing=routing or _routing(),
        consensus_cancel_l2=cancel_l2,
        consensus_cancel_threshold=cancel_threshold,
        extract_before_classify=True,
        fallback_archetype=fallback,
    )


def _resolver(voters=None, routing=None, fallback: str = 'explore') -> EnsembleResolver:
    v = voters or [_voter('semantic'), _voter('entity'), _voter('llm')]
    return EnsembleResolver(_cfg(v, routing=routing, fallback=fallback), _QUERY_TYPES)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVote:
    def test_vote_validation_rejects_empty_voter_id(self):
        with pytest.raises(QueryIntelligenceError):
            Vote(voter_id='', archetype='hybrid', confidence=0.9)

    def test_vote_validation_rejects_empty_archetype(self):
        with pytest.raises(QueryIntelligenceError):
            Vote(voter_id='v1', archetype='', confidence=0.9)

    def test_vote_validation_rejects_confidence_out_of_range(self):
        with pytest.raises(QueryIntelligenceError):
            Vote(voter_id='v1', archetype='hybrid', confidence=1.5)

    def test_abstain_helper(self):
        v = abstain('v1')
        assert v.abstained is True
        assert v.voter_id == 'v1'
        assert v.archetype == ''
        assert v.confidence == 0.0


class TestEnsembleResolver:
    def test_unanimous_vote(self):
        resolver = _resolver()
        votes = [
            Vote(voter_id='semantic', archetype='hybrid', confidence=0.95),
            Vote(voter_id='entity', archetype='hybrid', confidence=0.90),
            Vote(voter_id='llm', archetype='hybrid', confidence=0.88),
        ]
        result = resolver.resolve(votes)
        assert result.archetype == 'hybrid'
        assert result.agreement_ratio == 1.0
        assert result.veto_applied is False
        assert set(result.active_voter_ids) == {'semantic', 'entity', 'llm'}

    def test_majority_vote_wins(self):
        resolver = _resolver([_voter('v1'), _voter('v2'), _voter('v3'), _voter('v4')])
        votes = [
            Vote(voter_id='v1', archetype='hybrid', confidence=0.90),
            Vote(voter_id='v2', archetype='hybrid', confidence=0.85),
            Vote(voter_id='v3', archetype='hybrid', confidence=0.80),
            Vote(voter_id='v4', archetype='guidance', confidence=0.70),
        ]
        result = resolver.resolve(votes)
        assert result.archetype == 'hybrid'
        assert result.veto_applied is False
        assert result.agreement_ratio == pytest.approx(0.75)

    def test_veto_overrides_majority(self):
        voters = [
            _voter('v1', weight=1.0, has_veto=False),
            _voter('v2', weight=1.0, has_veto=False),
            _voter('v3', weight=1.0, has_veto=False),
            _voter('veto', weight=1.0, has_veto=True),
        ]
        resolver = _resolver(voters)
        votes = [
            Vote(voter_id='v1', archetype='guidance', confidence=0.90),
            Vote(voter_id='v2', archetype='guidance', confidence=0.88),
            Vote(voter_id='v3', archetype='guidance', confidence=0.85),
            Vote(voter_id='veto', archetype='hybrid', confidence=0.92),
        ]
        result = resolver.resolve(votes)
        assert result.archetype == 'hybrid'
        assert result.veto_applied is True
        assert result.veto_voter_id == 'veto'

    def test_abstain_excluded_from_tally(self):
        resolver = _resolver([_voter('v1'), _voter('v2'), _voter('v3')])
        votes = [
            Vote(voter_id='v1', archetype='hybrid', confidence=0.90),
            abstain('v2'),
            Vote(voter_id='v3', archetype='hybrid', confidence=0.85),
        ]
        result = resolver.resolve(votes)
        assert result.archetype == 'hybrid'
        assert 'v2' not in result.active_voter_ids
        assert result.total_weight_sum == pytest.approx(2.0)
        assert result.agreement_ratio == pytest.approx(1.0)

    def test_all_abstain_returns_fallback(self):
        resolver = _resolver(fallback='explore')
        votes = [abstain('semantic'), abstain('entity'), abstain('llm')]
        result = resolver.resolve(votes)
        assert result.archetype == 'explore'
        assert result.routing_mode == 'explore'
        assert result.agreement_ratio == 0.0
        assert result.active_voter_ids == []
        assert result.decision_tier == 'ensemble_all_abstain'

    def test_unknown_archetype_raises(self):
        resolver = _resolver()
        votes = [Vote(voter_id='semantic', archetype='unknown_type', confidence=0.9)]
        with pytest.raises(QueryIntelligenceError):
            resolver.resolve(votes)

    def test_routing_mode_auto_execute(self):
        routing = _routing(auto_exec_agr=0.75, auto_exec_conf=0.80,
                           suggest_agr=0.50, suggest_conf=0.60)
        resolver = _resolver(routing=routing)
        votes = [
            Vote(voter_id='semantic', archetype='hybrid', confidence=0.95),
            Vote(voter_id='entity', archetype='hybrid', confidence=0.90),
            Vote(voter_id='llm', archetype='hybrid', confidence=0.85),
        ]
        result = resolver.resolve(votes)
        assert result.routing_mode == 'auto_execute'

    def test_routing_mode_suggest(self):
        routing = _routing(auto_exec_agr=0.90, auto_exec_conf=0.92,
                           suggest_agr=0.50, suggest_conf=0.60)
        resolver = _resolver([_voter('v1'), _voter('v2')], routing=routing)
        votes = [
            Vote(voter_id='v1', archetype='hybrid', confidence=0.65),
            Vote(voter_id='v2', archetype='guidance', confidence=0.70),
        ]
        result = resolver.resolve(votes)
        assert result.routing_mode == 'suggest'

    def test_routing_mode_explore(self):
        routing = _routing(auto_exec_agr=0.90, auto_exec_conf=0.92,
                           suggest_agr=0.80, suggest_conf=0.85)
        resolver = _resolver([_voter('v1'), _voter('v2'), _voter('v3')], routing=routing)
        votes = [
            Vote(voter_id='v1', archetype='hybrid', confidence=0.55),
            Vote(voter_id='v2', archetype='guidance', confidence=0.60),
            Vote(voter_id='v3', archetype='explore', confidence=0.50),
        ]
        result = resolver.resolve(votes)
        assert result.routing_mode == 'explore'

    def test_routing_mode_confidence_axis(self):
        # High confidence can trigger auto_execute even with moderate agreement.
        routing = _routing(auto_exec_agr=0.90, auto_exec_conf=0.80,
                           suggest_agr=0.50, suggest_conf=0.60)
        resolver = _resolver([_voter('v1'), _voter('v2')], routing=routing)
        votes = [
            Vote(voter_id='v1', archetype='hybrid', confidence=0.95),
            Vote(voter_id='v2', archetype='guidance', confidence=0.40),
        ]
        result = resolver.resolve(votes)
        assert result.routing_mode == 'auto_execute'

    def test_weighted_voters_influence_tally(self):
        voters = [_voter('heavy', weight=3.0), _voter('light', weight=1.0)]
        resolver = _resolver(voters)
        votes = [
            Vote(voter_id='heavy', archetype='guidance', confidence=0.80),
            Vote(voter_id='light', archetype='hybrid', confidence=0.95),
        ]
        result = resolver.resolve(votes)
        assert result.archetype == 'guidance'
        assert result.agreement_ratio == pytest.approx(0.75)

    def test_veto_highest_confidence_wins_tie(self):
        voters = [
            _voter('v1', has_veto=True),
            _voter('v2', has_veto=True),
            _voter('v3'),
        ]
        resolver = _resolver(voters)
        votes = [
            Vote(voter_id='v1', archetype='guidance', confidence=0.70),
            Vote(voter_id='v2', archetype='explore', confidence=0.85),
            Vote(voter_id='v3', archetype='hybrid', confidence=0.95),
        ]
        result = resolver.resolve(votes)
        assert result.veto_applied is True
        assert result.veto_voter_id == 'v2'
        assert result.archetype == 'explore'
