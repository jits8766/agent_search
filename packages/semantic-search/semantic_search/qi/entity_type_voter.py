"""Entity-signal voter for the ensemble resolver.

Infers query archetype from L0-extracted entities and raw query text signals.
Carries veto power because structured filter constraints (price, TLD, domain
length) are deterministic signals that semantic centroid similarity cannot
reliably distinguish from advisory phrasing.

All slot names, confidence levels, and signal vocabulary come from
QIEntityVoterConfig; no literals appear in the classification logic.
"""
from __future__ import annotations

import re
from typing import List, Optional

from semantic_search.config.models import QIEntityVoterConfig
from semantic_search.contracts import Entity
from semantic_search.core.exceptions import ConfigurationError
from semantic_search.core.logging_utils import get_logger
from semantic_search.qi.ensemble_resolver import Vote, abstain

logger = get_logger(__name__)


class EntityTypeVoter:
    """Vote archetype from L0 entities and text signals (filters, value-discovery terms)."""

    def __init__(
        self,
        config: QIEntityVoterConfig,
        filter_signal_re: re.Pattern,
    ) -> None:
        if config is None:
            raise ConfigurationError("EntityTypeVoter requires a QIEntityVoterConfig instance")
        if filter_signal_re is None:
            raise ConfigurationError("EntityTypeVoter requires a compiled filter_signal_re pattern")
        self._config = config
        self._filter_signal_re = filter_signal_re
        self._hard_filter_force_slots: frozenset = frozenset(config.hard_filter_force_slots)
        self._value_discovery_signals: frozenset = frozenset(config.value_discovery_signals)
        logger.info(
            f"entity_type_voter_initialized voter_id={config.voter_id!r} "
            f"hard_filter_slots={len(self._hard_filter_force_slots)} "
            f"value_discovery_signals={len(self._value_discovery_signals)} "
            f"confidence_emit={config.confidence_emit:.3f} "
            f"signal_only_scale={config.signal_only_scale:.3f}"
        )

    @property
    def voter_id(self) -> str:
        """Voter identifier matching the ensemble config entry."""
        return self._config.voter_id

    def classify(self, query: str, l0_entities: List[Entity]) -> Optional[Vote]:
        """Infer archetype: hard entity > value-discovery vocab > filter signals, else abstain."""
        if not isinstance(query, str):
            logger.debug(f"entity_type_voter_abstain voter_id={self._config.voter_id!r} reason=invalid_query_type")
            return None

        # Hard entity signal (strongest, deterministic)
        for entity in l0_entities:
            if (getattr(entity, 'chip_kind', 'hard') == 'hard'
                    and entity.name in self._hard_filter_force_slots):
                logger.debug(
                    f"entity_type_voter_hard_entity voter_id={self._config.voter_id!r} "
                    f"slot={entity.name!r} archetype=hybrid"
                )
                return Vote(
                    voter_id=self._config.voter_id,
                    archetype='hybrid',
                    confidence=float(self._config.confidence_emit),
                )

        # Value-discovery vocabulary: investor/resale signals in raw query text.
        q_lower = query.lower()
        if any(sig in q_lower for sig in self._value_discovery_signals):
            logger.debug(
                f"entity_type_voter_value_discovery voter_id={self._config.voter_id!r} archetype=hybrid"
            )
            return Vote(
                voter_id=self._config.voter_id,
                archetype='hybrid',
                confidence=float(self._config.confidence_emit),
            )

        # Filter signal pattern: structural signals in raw text when no entities extracted.
        if self._filter_signal_re.search(query):
            scaled_conf = float(self._config.confidence_emit) * float(self._config.signal_only_scale)
            logger.debug(
                f"entity_type_voter_signal_text voter_id={self._config.voter_id!r} "
                f"archetype=hybrid confidence={scaled_conf:.3f}"
            )
            return Vote(
                voter_id=self._config.voter_id,
                archetype='hybrid',
                confidence=scaled_conf,
            )

        # No signal found — abstain.
        logger.debug(f"entity_type_voter_abstain voter_id={self._config.voter_id!r} reason=no_signal")
        return None
