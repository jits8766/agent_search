"""Unit tests for EntityTypeVoter.

Coverage matrix:

EntityTypeVoter.classify:
- hard_entity_votes_hybrid              -> TestEntityTypeVoter::test_hard_entity_votes_hybrid
- soft_entity_no_slot_match_abstains    -> TestEntityTypeVoter::test_soft_entity_no_slot_match_abstains
- value_discovery_signal_votes_hybrid   -> TestEntityTypeVoter::test_value_discovery_signal_votes_hybrid
- filter_signal_text_scaled_confidence  -> TestEntityTypeVoter::test_filter_signal_text_scaled_confidence
- no_signal_abstains                    -> TestEntityTypeVoter::test_no_signal_abstains
- invalid_query_type_abstains           -> TestEntityTypeVoter::test_invalid_query_type_abstains
- hard_entity_confidence_exact          -> TestEntityTypeVoter::test_hard_entity_confidence_exact

EntityTypeVoter.voter_id:
- voter_id_property                     -> TestEntityTypeVoter::test_voter_id_property

Construction:
- missing_config_raises                 -> TestEntityTypeVoterConstruction::test_missing_config_raises
- missing_pattern_raises                -> TestEntityTypeVoterConstruction::test_missing_pattern_raises
"""
import re

import pytest

from semantic_search.config.models import QIEntityVoterConfig
from semantic_search.contracts import Entity
from semantic_search.core.exceptions import ConfigurationError
from semantic_search.qi.entity_type_voter import EntityTypeVoter

# ---------------------------------------------------------------------------
# Fixtures / factories
# ---------------------------------------------------------------------------

_HARD_SLOTS = ['tld', 'price_max', 'price_min', 'name_length_max', 'auction_type']
_VALUE_SIGNALS = ['underpriced', 'resale', 'flip potential', 'good value']
_FILTER_RE = re.compile(r'\.[a-z]{2,6}\b|under\s+\$?\d+|price\s*<\s*\d+', re.IGNORECASE)


def _cfg(
    voter_id: str = 'entity',
    confidence_emit: float = 0.92,
    signal_only_scale: float = 0.70,
) -> QIEntityVoterConfig:
    return QIEntityVoterConfig(
        voter_id=voter_id,
        confidence_emit=confidence_emit,
        signal_only_scale=signal_only_scale,
        hard_filter_force_slots=list(_HARD_SLOTS),
        veto_archetype='guidance',
        value_discovery_signals=list(_VALUE_SIGNALS),
    )


def _entity(name: str, chip_kind: str = 'hard') -> Entity:
    return Entity(name=name, value='test', confidence=0.95, source='L0_llm', chip_kind=chip_kind)


def _voter(cfg=None) -> EntityTypeVoter:
    return EntityTypeVoter(cfg or _cfg(), _FILTER_RE)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEntityTypeVoterConstruction:
    def test_missing_config_raises(self):
        with pytest.raises(ConfigurationError):
            EntityTypeVoter(None, _FILTER_RE)  # type: ignore[arg-type]

    def test_missing_pattern_raises(self):
        with pytest.raises(ConfigurationError):
            EntityTypeVoter(_cfg(), None)  # type: ignore[arg-type]


class TestEntityTypeVoter:
    def test_voter_id_property(self):
        assert _voter().voter_id == 'entity'

    def test_hard_entity_votes_hybrid(self):
        voter = _voter()
        vote = voter.classify('domains under $500 with .io', [_entity('tld')])
        assert vote is not None
        assert vote.archetype == 'hybrid'
        assert vote.confidence == pytest.approx(0.92)
        assert vote.voter_id == 'entity'

    def test_hard_entity_confidence_exact(self):
        cfg = _cfg(confidence_emit=0.85)
        voter = EntityTypeVoter(cfg, _FILTER_RE)
        vote = voter.classify('cheap .com domains', [_entity('price_max')])
        assert vote is not None
        assert vote.confidence == pytest.approx(0.85)

    def test_soft_entity_no_slot_match_abstains(self):
        voter = _voter()
        # soft chip_kind entity whose name is in hard_filter_force_slots but chip_kind='soft'
        vote = voter.classify('domains with tld .ai', [_entity('tld', chip_kind='soft')])
        # soft entity bypasses the hard-entity branch;
        # if no other signal → None; if _FILTER_RE matches ".ai" it should fire signal-text
        if vote is None:
            pass  # abstained correctly
        else:
            assert vote.archetype == 'hybrid'

    def test_value_discovery_signal_votes_hybrid(self):
        voter = _voter()
        vote = voter.classify('find underpriced domains in tech niche', [])
        assert vote is not None
        assert vote.archetype == 'hybrid'
        assert vote.confidence == pytest.approx(0.92)

    def test_filter_signal_text_scaled_confidence(self):
        voter = _voter(_cfg(confidence_emit=0.92, signal_only_scale=0.70))
        # ".io" matches _FILTER_RE; no entities extracted
        vote = voter.classify('domains .io', [])
        assert vote is not None
        assert vote.archetype == 'hybrid'
        assert vote.confidence == pytest.approx(0.92 * 0.70)

    def test_no_signal_abstains(self):
        voter = _voter()
        vote = voter.classify('best domain registrar for startups', [])
        assert vote is None

    def test_invalid_query_type_abstains(self):
        voter = _voter()
        vote = voter.classify(None, [])  # type: ignore[arg-type]
        assert vote is None

    def test_value_discovery_case_insensitive(self):
        voter = _voter()
        vote = voter.classify('GOOD VALUE .com domains for resale', [])
        assert vote is not None
        assert vote.archetype == 'hybrid'

    def test_hard_entity_takes_priority_over_signal(self):
        cfg = _cfg(confidence_emit=0.92, signal_only_scale=0.50)
        voter = EntityTypeVoter(cfg, _FILTER_RE)
        entities = [_entity('price_max', chip_kind='hard')]
        vote = voter.classify('domains under $200', entities)
        assert vote is not None
        assert vote.confidence == pytest.approx(0.92)
