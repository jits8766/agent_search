"""Tests for ``semantic_search.retrieval.reranker_base``.

Covers:

* ``RerankedItem`` __post_init__ validation
* ``stable_sort_descending`` ordering + tie-stability
* ``Reranker`` ABC default behaviour (every method raises NotImplementedError)
"""
import pytest

from semantic_search.contracts import RankedItem
from semantic_search.core.exceptions import ValidationError
from semantic_search.retrieval.reranker_base import Reranker, RerankedItem, stable_sort_descending


def _ri(item_id: str, fused: float = 1.0) -> RankedItem:
    return RankedItem( item_id=item_id, fused_score=fused, contributing_sources=['vector'], payload={})


class TestRerankedItem:
    def test_constructs_with_valid_inputs(self):
        item = _ri('a')
        r = RerankedItem(item=item, rerank_score=0.5)
        assert r.item is item
        assert r.rerank_score == 0.5

    def test_zero_score_is_valid(self):
        r = RerankedItem(item=_ri('a'), rerank_score=0.0)
        assert r.rerank_score == 0.0

    def test_rejects_negative_score(self):
        with pytest.raises(ValidationError, match="rerank_score must be >= 0"):
            RerankedItem(item=_ri('a'), rerank_score=-0.0001)

    def test_rejects_non_ranked_item(self):
        with pytest.raises(ValidationError, match="must be a RankedItem"):
            RerankedItem(item={'item_id': 'a'}, rerank_score=0.5)  # type: ignore[arg-type]

    def test_frozen_dataclass(self):
        r = RerankedItem(item=_ri('a'), rerank_score=0.5)
        with pytest.raises(Exception):
            r.rerank_score = 1.0  # type: ignore[misc]


class TestStableSortDescending:
    def test_empty_returns_empty(self):
        assert stable_sort_descending([]) == []

    def test_single_returns_single(self):
        r = RerankedItem(item=_ri('a'), rerank_score=0.5)
        assert stable_sort_descending([r]) == [r]

    def test_orders_by_score_desc(self):
        a = RerankedItem(item=_ri('a'), rerank_score=0.1)
        b = RerankedItem(item=_ri('b'), rerank_score=0.9)
        c = RerankedItem(item=_ri('c'), rerank_score=0.5)
        out = stable_sort_descending([a, b, c])
        assert [x.item.item_id for x in out] == ['b', 'c', 'a']

    def test_stable_on_ties_preserves_input_order(self):
        # Three items all tied at 0.5 — must come back in input order
        a = RerankedItem(item=_ri('a'), rerank_score=0.5)
        b = RerankedItem(item=_ri('b'), rerank_score=0.5)
        c = RerankedItem(item=_ri('c'), rerank_score=0.5)
        out = stable_sort_descending([a, b, c])
        assert [x.item.item_id for x in out] == ['a', 'b', 'c']

    def test_stable_with_partial_ties(self):
        # b and d tied at 0.7 (b before d in input → b before d in output)
        # c is highest, a is lowest
        a = RerankedItem(item=_ri('a'), rerank_score=0.1)
        b = RerankedItem(item=_ri('b'), rerank_score=0.7)
        c = RerankedItem(item=_ri('c'), rerank_score=0.9)
        d = RerankedItem(item=_ri('d'), rerank_score=0.7)
        out = stable_sort_descending([a, b, c, d])
        assert [x.item.item_id for x in out] == ['c', 'b', 'd', 'a']


class TestRerankerABC:
    def test_name_property_raises_not_implemented(self):
        r = Reranker()
        with pytest.raises(NotImplementedError):
            _ = r.name

    def test_rerank_method_raises_not_implemented(self):
        r = Reranker()
        with pytest.raises(NotImplementedError):
            r.rerank('q', [], 5)
