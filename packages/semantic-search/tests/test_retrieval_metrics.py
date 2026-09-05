"""Tests for ranking-quality metrics.

Coverage matrix for ``dcg_at_k``:
- empty_ranked_returns_zero                 -> test_dcg_empty_ranked_returns_zero
- empty_gains_returns_zero                  -> test_dcg_empty_gains_returns_zero
- closed_form_known_ranking                 -> test_dcg_known_closed_form
- ignores_gain_zero                         -> test_dcg_ignores_zero_gain_items
- truncates_at_k                            -> test_dcg_truncates_at_k
- discount_log2_correct                     -> test_dcg_discount_uses_log2_i_plus_1
- gain_2_to_the_g_minus_1                   -> test_dcg_gain_uses_2_to_g_minus_1
- k_below_one_raises                        -> test_dcg_k_below_one_raises
- k_wrong_type_raises                       -> test_dcg_k_wrong_type_raises
- ranked_none_raises                        -> test_dcg_ranked_none_raises
- ranked_entry_wrong_type_raises            -> test_dcg_ranked_entry_wrong_type_raises
- gains_none_raises                         -> test_dcg_gains_none_raises
- gains_wrong_type_raises                   -> test_dcg_gains_wrong_type_raises

Coverage matrix for ``ndcg_at_k``:
- perfect_ordering_equals_one               -> test_ndcg_perfect_ordering_returns_one
- inverted_ordering_below_one               -> test_ndcg_inverted_ordering_below_perfect
- empty_ideal_returns_zero                  -> test_ndcg_empty_ideal_returns_zero
- known_closed_form                         -> test_ndcg_known_closed_form
- range_within_zero_one                     -> test_ndcg_range_within_zero_one
- truncates_idcg_at_k                       -> test_ndcg_idcg_uses_top_k_of_ideal
- k_below_one_raises                        -> test_ndcg_k_below_one_raises

Coverage matrix for ``recall_at_k``:
- all_relevant_in_top_k_returns_one         -> test_recall_all_in_top_k_returns_one
- none_relevant_returns_zero                -> test_recall_none_in_top_k_returns_zero
- partial_returns_fraction                  -> test_recall_partial_returns_fraction
- empty_relevant_returns_zero               -> test_recall_empty_relevant_returns_zero
- relevant_outside_top_k_excluded           -> test_recall_relevant_outside_top_k_excluded
- relevant_none_raises                      -> test_recall_relevant_none_raises
- k_below_one_raises                        -> test_recall_k_below_one_raises

Coverage matrix for ``precision_at_k``:
- all_top_k_relevant_returns_one            -> test_precision_all_top_k_relevant_returns_one
- none_relevant_returns_zero                -> test_precision_none_relevant_returns_zero
- divisor_is_k_not_returned_count           -> test_precision_divisor_is_k_not_returned_count
- empty_relevant_returns_zero               -> test_precision_empty_relevant_returns_zero
- relevant_none_raises                      -> test_precision_relevant_none_raises

Coverage matrix for ``mrr_at_k``:
- first_relevant_at_position_one_returns_one -> test_mrr_first_at_position_one
- relevant_at_position_n_returns_one_over_n  -> test_mrr_at_position_n_returns_one_over_n
- no_relevant_in_top_k_returns_zero          -> test_mrr_no_relevant_in_top_k_returns_zero
- relevant_only_outside_top_k_returns_zero   -> test_mrr_relevant_only_outside_top_k_returns_zero
- empty_relevant_returns_zero                -> test_mrr_empty_relevant_returns_zero
- relevant_none_raises                       -> test_mrr_relevant_none_raises
"""
import math

import pytest

from semantic_search.eval.retrieval_metrics import dcg_at_k, mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k


class TestDCG:
    """DCG@k correctness + math + contract tests."""

    def test_dcg_empty_ranked_returns_zero(self):
        assert dcg_at_k([], {'a': 3}, 10) == 0.0

    def test_dcg_empty_gains_returns_zero(self):
        assert dcg_at_k(['a', 'b', 'c'], {}, 10) == 0.0

    def test_dcg_known_closed_form(self):
        # ranking = [a(g=3), b(g=2), c(g=0)]
        # DCG = (2^3 - 1)/log2(2) + (2^2 - 1)/log2(3) + 0
        #     = 7/1 + 3/log2(3) ≈ 7 + 3/1.5849625 ≈ 7 + 1.8927892
        gains = {'a': 3, 'b': 2, 'c': 0}
        expected = 7.0 / math.log2(2) + 3.0 / math.log2(3)
        assert dcg_at_k(['a', 'b', 'c'], gains, 3) == pytest.approx(expected, rel=1e-12)

    def test_dcg_ignores_zero_gain_items(self):
        # An item with gain=0 contributes 0 to DCG; ranking pos still consumed.
        gains = {'a': 1, 'b': 0, 'c': 1}
        # DCG = (2^1-1)/log2(2) + 0 + (2^1-1)/log2(4)
        #     = 1.0 + 0 + 1.0/2.0 = 1.5
        assert dcg_at_k(['a', 'b', 'c'], gains, 3) == pytest.approx(1.5, rel=1e-12)

    def test_dcg_truncates_at_k(self):
        gains = {'a': 4, 'b': 4, 'c': 4, 'd': 4}
        full = dcg_at_k(['a', 'b', 'c', 'd'], gains, 4)
        truncated = dcg_at_k(['a', 'b', 'c', 'd'], gains, 2)
        assert truncated < full
        # DCG@2 = (2^4-1)/log2(2) + (2^4-1)/log2(3) = 15/1 + 15/log2(3)
        expected = 15.0 / 1.0 + 15.0 / math.log2(3)
        assert truncated == pytest.approx(expected, rel=1e-12)

    def test_dcg_discount_uses_log2_i_plus_1(self):
        # Single relevant item at position 1: discount = log2(2) = 1.0,
        # so DCG@1 = (2^g - 1) exactly.
        for g in (1, 2, 3, 4):
            assert dcg_at_k(['a'], {'a': g}, 1) == pytest.approx(2.0 ** g - 1.0, rel=1e-12)

    def test_dcg_gain_uses_2_to_g_minus_1(self):
        # Pin gain mapping per position 1: g=1 -> 1, g=2 -> 3, g=3 -> 7, g=4 -> 15
        for g, expected in [(1, 1.0), (2, 3.0), (3, 7.0), (4, 15.0)]:
            assert dcg_at_k(['x'], {'x': g}, 1) == pytest.approx(expected, rel=1e-12)

    def test_dcg_k_below_one_raises(self):
        with pytest.raises(ValueError, match="k must be a positive int"):
            dcg_at_k(['a'], {'a': 1}, 0)
        with pytest.raises(ValueError, match="k must be a positive int"):
            dcg_at_k(['a'], {'a': 1}, -1)

    def test_dcg_k_wrong_type_raises(self):
        with pytest.raises(ValueError, match="k must be a positive int"):
            dcg_at_k(['a'], {'a': 1}, "10")
        with pytest.raises(ValueError, match="k must be a positive int"):
            dcg_at_k(['a'], {'a': 1}, True)

    def test_dcg_ranked_none_raises(self):
        with pytest.raises(ValueError, match="ranked_item_ids must not be None"):
            dcg_at_k(None, {'a': 1}, 10)

    def test_dcg_ranked_entry_wrong_type_raises(self):
        with pytest.raises(ValueError, match="ranked_item_ids entries must be non-empty strings"):
            dcg_at_k(['a', '', 'c'], {'a': 1}, 10)
        with pytest.raises(ValueError, match="ranked_item_ids entries must be non-empty strings"):
            dcg_at_k(['a', 5, 'c'], {'a': 1}, 10)

    def test_dcg_gains_none_raises(self):
        with pytest.raises(ValueError, match="gain_by_item_id must not be None"):
            dcg_at_k(['a'], None, 10)

    def test_dcg_gains_wrong_type_raises(self):
        with pytest.raises(ValueError, match="gain_by_item_id must be a dict"):
            dcg_at_k(['a'], [('a', 1)], 10)


class TestNDCG:
    """NDCG@k correctness — proves ratio + ideal-ordering math."""

    def test_ndcg_perfect_ordering_returns_one(self):
        gains = {'a': 4, 'b': 3, 'c': 2, 'd': 1, 'e': 0}
        ranked = ['a', 'b', 'c', 'd', 'e']
        assert ndcg_at_k(ranked, gains, 5) == pytest.approx(1.0, rel=1e-12)

    def test_ndcg_inverted_ordering_below_perfect(self):
        gains = {'a': 4, 'b': 3, 'c': 2, 'd': 1}
        worst = ndcg_at_k(['d', 'c', 'b', 'a'], gains, 4)
        assert 0.0 < worst < 1.0

    def test_ndcg_empty_ideal_returns_zero(self):
        # No item has gain >= 1 -> IDCG = 0 -> NDCG = 0.0 (graceful, not div-by-zero)
        assert ndcg_at_k(['a', 'b'], {'a': 0, 'b': 0}, 5) == 0.0

    def test_ndcg_known_closed_form(self):
        # ranking = [b(g=2), a(g=3), c(g=1)]; ideal = [a(3), b(2), c(1)]
        # DCG  = 3/log2(2) + 7/log2(3) + 1/log2(4) = 3 + 7/1.585 + 1/2
        # IDCG = 7/log2(2) + 3/log2(3) + 1/log2(4) = 7 + 3/1.585 + 1/2
        gains = {'a': 3, 'b': 2, 'c': 1}
        dcg = 3.0 + 7.0 / math.log2(3) + 1.0 / math.log2(4)
        idcg = 7.0 + 3.0 / math.log2(3) + 1.0 / math.log2(4)
        expected = dcg / idcg
        assert ndcg_at_k(['b', 'a', 'c'], gains, 3) == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("ranked,gains,k", [
        (['a', 'b', 'c'], {'a': 1, 'b': 1, 'c': 1}, 3),
        (['c', 'b', 'a'], {'a': 4, 'b': 2, 'c': 0}, 3),
        (['x'], {'x': 4}, 1),
        (['a', 'b'], {'a': 0, 'b': 1}, 2),
    ])
    def test_ndcg_range_within_zero_one(self, ranked, gains, k):
        v = ndcg_at_k(ranked, gains, k)
        assert 0.0 <= v <= 1.0

    def test_ndcg_idcg_uses_top_k_of_ideal(self):
        # Many relevant items beyond k: IDCG@k uses only the top-k ideal slots.
        gains = {f"x{i}": 4 for i in range(20)}
        # Perfect retrieval of top-3 -> NDCG@3 = 1.0 (IDCG@3 also uses top-3 ideal)
        ranked = list(gains.keys())
        assert ndcg_at_k(ranked, gains, 3) == pytest.approx(1.0, rel=1e-12)

    def test_ndcg_k_below_one_raises(self):
        with pytest.raises(ValueError, match="k must be a positive int"):
            ndcg_at_k(['a'], {'a': 1}, 0)


class TestRecall:
    """Recall@k correctness."""

    def test_recall_all_in_top_k_returns_one(self):
        assert recall_at_k(['a', 'b', 'c'], ['a', 'b'], 3) == 1.0

    def test_recall_none_in_top_k_returns_zero(self):
        assert recall_at_k(['x', 'y', 'z'], ['a', 'b'], 3) == 0.0

    def test_recall_partial_returns_fraction(self):
        # 2 of 4 relevant present in top-3 -> 0.5
        assert recall_at_k(['a', 'x', 'b', 'y'], ['a', 'b', 'c', 'd'], 3) == pytest.approx(0.5, rel=1e-12)

    def test_recall_empty_relevant_returns_zero(self):
        assert recall_at_k(['a'], [], 5) == 0.0

    def test_recall_relevant_outside_top_k_excluded(self):
        # Relevant item at rank 4, k=3 -> not counted.
        assert recall_at_k(['x', 'y', 'z', 'a'], ['a'], 3) == 0.0

    def test_recall_relevant_none_raises(self):
        with pytest.raises(ValueError, match="relevant_item_ids must not be None"):
            recall_at_k(['a'], None, 5)

    def test_recall_k_below_one_raises(self):
        with pytest.raises(ValueError, match="k must be a positive int"):
            recall_at_k(['a'], ['a'], 0)


class TestPrecision:
    """Precision@k correctness — divisor MUST be k, not min(k, returned_count)."""

    def test_precision_all_top_k_relevant_returns_one(self):
        assert precision_at_k(['a', 'b', 'c'], ['a', 'b', 'c'], 3) == 1.0

    def test_precision_none_relevant_returns_zero(self):
        assert precision_at_k(['x', 'y', 'z'], ['a', 'b'], 3) == 0.0

    def test_precision_divisor_is_k_not_returned_count(self):
        # 1 relevant returned, but k=10 -> precision = 0.1 (NOT 1.0).
        # Standard convention: under-retrieval is penalised, not masked.
        assert precision_at_k(['a'], ['a'], 10) == pytest.approx(0.1, rel=1e-12)

    def test_precision_empty_relevant_returns_zero(self):
        assert precision_at_k(['a'], [], 5) == 0.0

    def test_precision_relevant_none_raises(self):
        with pytest.raises(ValueError, match="relevant_item_ids must not be None"):
            precision_at_k(['a'], None, 5)


class TestMRR:
    """MRR@k correctness (per-query reciprocal rank)."""

    def test_mrr_first_at_position_one(self):
        assert mrr_at_k(['a', 'b'], ['a'], 5) == 1.0

    @pytest.mark.parametrize("pos,expected", [
        (2, 0.5),
        (3, 1.0 / 3.0),
        (4, 0.25),
        (10, 0.1),
    ])
    def test_mrr_at_position_n_returns_one_over_n(self, pos, expected):
        ranking = [f"x{i}" for i in range(pos - 1)] + ['target']
        assert mrr_at_k(ranking, ['target'], pos) == pytest.approx(expected, rel=1e-12)

    def test_mrr_no_relevant_in_top_k_returns_zero(self):
        assert mrr_at_k(['x', 'y', 'z'], ['a'], 3) == 0.0

    def test_mrr_relevant_only_outside_top_k_returns_zero(self):
        # Target at position 5, k=3 -> 0.0
        ranking = ['x', 'y', 'z', 'w', 'target']
        assert mrr_at_k(ranking, ['target'], 3) == 0.0

    def test_mrr_empty_relevant_returns_zero(self):
        assert mrr_at_k(['a'], [], 5) == 0.0

    def test_mrr_relevant_none_raises(self):
        with pytest.raises(ValueError, match="relevant_item_ids must not be None"):
            mrr_at_k(['a'], None, 5)
