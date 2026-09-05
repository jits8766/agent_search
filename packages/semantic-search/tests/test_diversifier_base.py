"""Contract tests for the Diversifier protocol surface.

Covers ``DiversifiedItem`` validation, the ``Diversifier`` ABC's
NotImplementedError contract, and the protocol's argument-shape promises.
Mirrors ``tests/test_reranker_base.py`` so the two protocols evolve in
lock-step.
"""
import math

import pytest

from semantic_search.contracts import RankedItem
from semantic_search.core.exceptions import ValidationError
from semantic_search.retrieval.diversifier_base import DiversifiedItem, Diversifier


def _make_item(item_id: str = 'a', fused_score: float = 1.0) -> RankedItem:
    return RankedItem(
        item_id=item_id,
        fused_score=float(fused_score),
        contributing_sources=['vector'],
        payload={'title': f'Item {item_id}'},
        sub_intent_ids=[],
    )


class TestDiversifiedItemContract:
    """`DiversifiedItem.__post_init__` validates its four invariants."""

    def test_valid_construction_succeeds(self):
        item = _make_item()
        d = DiversifiedItem(item=item, base_score=1.0, diversity_penalty=0.0, mmr_score=0.7)
        assert d.item is item
        assert d.base_score == 1.0
        assert d.diversity_penalty == 0.0
        assert d.mmr_score == 0.7

    def test_item_must_be_ranked_item(self):
        with pytest.raises(ValidationError, match="must be a RankedItem"):
            DiversifiedItem(item="not_a_ranked_item", base_score=0.5, diversity_penalty=0.0, mmr_score=0.5)

    def test_base_score_below_zero_rejected(self):
        with pytest.raises(ValidationError, match="base_score must be in"):
            DiversifiedItem(item=_make_item(), base_score=-0.1, diversity_penalty=0.0, mmr_score=0.0)

    def test_base_score_above_one_rejected(self):
        with pytest.raises(ValidationError, match="base_score must be in"):
            DiversifiedItem(item=_make_item(), base_score=1.1, diversity_penalty=0.0, mmr_score=0.0)

    def test_base_score_at_boundary_zero_accepted(self):
        DiversifiedItem(item=_make_item(), base_score=0.0, diversity_penalty=0.0, mmr_score=0.0)

    def test_base_score_at_boundary_one_accepted(self):
        DiversifiedItem(item=_make_item(), base_score=1.0, diversity_penalty=0.0, mmr_score=1.0)

    def test_diversity_penalty_below_zero_rejected(self):
        with pytest.raises(ValidationError, match="diversity_penalty must be in"):
            DiversifiedItem(item=_make_item(), base_score=0.5, diversity_penalty=-0.1, mmr_score=0.0)

    def test_diversity_penalty_above_one_rejected(self):
        with pytest.raises(ValidationError, match="diversity_penalty must be in"):
            DiversifiedItem(item=_make_item(), base_score=0.5, diversity_penalty=1.1, mmr_score=0.0)

    def test_mmr_score_below_minus_one_rejected(self):
        with pytest.raises(ValidationError, match="mmr_score must be"):
            DiversifiedItem(item=_make_item(), base_score=0.5, diversity_penalty=1.0, mmr_score=-1.5)

    def test_mmr_score_above_one_rejected(self):
        with pytest.raises(ValidationError, match="mmr_score must be"):
            DiversifiedItem(item=_make_item(), base_score=1.0, diversity_penalty=0.0, mmr_score=1.5)

    def test_mmr_score_nan_rejected(self):
        with pytest.raises(ValidationError, match="mmr_score must be"):
            DiversifiedItem(item=_make_item(), base_score=0.5, diversity_penalty=0.0, mmr_score=float('nan'))

    def test_mmr_score_at_boundaries_accepted(self):
        DiversifiedItem(item=_make_item(), base_score=0.0, diversity_penalty=1.0, mmr_score=-1.0)
        DiversifiedItem(item=_make_item(), base_score=1.0, diversity_penalty=0.0, mmr_score=1.0)


class TestDiversifierABC:
    """The ``Diversifier`` ABC requires concrete subclasses to override both members."""

    def test_name_property_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            _ = Diversifier().name

    def test_diversify_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            Diversifier().diversify(query='q', items=[_make_item()], top_n=1, output_n=1)
