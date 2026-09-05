"""Pure ranking-quality metrics for the retrieval-quality evaluator.

Stdlib-only (``math.log2`` is the only non-trivial import). Every function is
a pure mathematical transform over (ranked_item_ids, gain_by_item_id, k):

- ``dcg_at_k``       : Discounted Cumulative Gain at k (graded relevance)
- ``ndcg_at_k``      : Normalized DCG (DCG / IDCG)
- ``recall_at_k``    : |relevant ∩ top-k| / |relevant|
- ``precision_at_k`` : |relevant ∩ top-k| / k
- ``mrr_at_k``       : 1 / rank of the first relevant item in top-k (else 0)

All metrics return ``float`` in ``[0.0, 1.0]`` (NDCG / Recall / Precision / MRR
are normalized; DCG is unbounded but we expose it for diagnostics).

NDCG formula (standard graded-gain formulation):
    DCG_k  = sum_{i=1..k} (2^gain_i - 1) / log2(i + 1)
    IDCG_k = DCG_k computed over the ideal ordering of the judgment set
    NDCG_k = DCG_k / IDCG_k  (0.0 when IDCG_k = 0)

All functions raise ``ValueError`` on ``k < 1`` or ``None`` inputs — the evaluator
catches these centrally and emits ``ValidationError`` with structured context.
"""
import math
from typing import Dict, List, Sequence


def _validate_k(k: int) -> None:
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive int")


def _validate_ranked(ranked_item_ids: Sequence[str]) -> None:
    if ranked_item_ids is None:
        raise ValueError("ranked_item_ids must not be None")
    for it in ranked_item_ids:
        if not isinstance(it, str) or not it:
            raise ValueError("ranked_item_ids entries must be non-empty strings")


def _validate_gains(gain_by_item_id: Dict[str, int]) -> None:
    if gain_by_item_id is None:
        raise ValueError("gain_by_item_id must not be None")
    if not isinstance(gain_by_item_id, dict):
        raise ValueError("gain_by_item_id must be a dict[str, int]")


def dcg_at_k(ranked_item_ids: Sequence[str], gain_by_item_id: Dict[str, int], k: int) -> float:
    """DCG@k: sum (2^gain - 1) / log2(rank + 1) for top-k (closed-world: missing = 0)."""
    _validate_k(k)
    _validate_ranked(ranked_item_ids)
    _validate_gains(gain_by_item_id)
    total = 0.0
    for i, item_id in enumerate(ranked_item_ids[:k], start=1):
        gain = int(gain_by_item_id.get(item_id, 0))
        if gain <= 0:
            continue
        # Rank discount: i=1 → log2(2)=1.0
        total += (2.0 ** gain - 1.0) / math.log2(i + 1)
    return total


def ndcg_at_k(ranked_item_ids: Sequence[str], gain_by_item_id: Dict[str, int], k: int) -> float:
    """NDCG@k: DCG@k / IDCG@k (ideal ordering, descending gains; returns 0 if IDCG=0).
    :param k: int - Cutoff rank (>= 1)
    :return: float - NDCG value in [0.0, 1.0]
    :raises ValueError: When k < 1 or any input is malformed
    """
    _validate_k(k)
    _validate_ranked(ranked_item_ids)
    _validate_gains(gain_by_item_id)
    dcg = dcg_at_k(ranked_item_ids, gain_by_item_id, k)
    ideal_gains = sorted((g for g in gain_by_item_id.values() if g > 0), reverse=True)
    idcg = 0.0
    for i, gain in enumerate(ideal_gains[:k], start=1):
        idcg += (2.0 ** int(gain) - 1.0) / math.log2(i + 1)
    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def recall_at_k(ranked_item_ids: Sequence[str], relevant_item_ids: Sequence[str], k: int) -> float:
    """Recall at k (binary relevance — any judgment with gain >= 1 is relevant).

    Formula: Recall_k = |relevant ∩ top-k| / |relevant|
    Returns 0.0 when the relevant set is empty (caller should exclude such
    queries upstream — ``RelevanceJudgedQuery`` enforces this invariant).

    :param ranked_item_ids: Sequence[str] - Item ids ordered best -> worst
    :param relevant_item_ids: Sequence[str] - Ids with gain >= 1
    :param k: int - Cutoff rank (>= 1)
    :return: float - Recall in [0.0, 1.0]
    :raises ValueError: When k < 1 or any input is malformed
    """
    _validate_k(k)
    _validate_ranked(ranked_item_ids)
    if relevant_item_ids is None:
        raise ValueError("relevant_item_ids must not be None")
    relevant = set(relevant_item_ids)
    if not relevant:
        return 0.0
    top_k = set(ranked_item_ids[:k])
    hits = len(top_k & relevant)
    return float(hits) / float(len(relevant))


def precision_at_k(ranked_item_ids: Sequence[str], relevant_item_ids: Sequence[str], k: int) -> float:
    """Precision at k (binary relevance).

    Formula: Precision_k = |relevant ∩ top-k| / k
    Note: the divisor is ``k``, NOT ``min(k, len(ranked_item_ids))`` — when the
    retriever returns fewer than k items, precision drops accordingly (this is
    the standard convention; reporting "P@10 = 1.0 because 1/1 returned" would
    mask under-retrieval).

    :param ranked_item_ids: Sequence[str] - Item ids ordered best -> worst
    :param relevant_item_ids: Sequence[str] - Ids with gain >= 1
    :param k: int - Cutoff rank (>= 1)
    :return: float - Precision in [0.0, 1.0]
    :raises ValueError: When k < 1 or any input is malformed
    """
    _validate_k(k)
    _validate_ranked(ranked_item_ids)
    if relevant_item_ids is None:
        raise ValueError("relevant_item_ids must not be None")
    relevant = set(relevant_item_ids)
    if not relevant:
        return 0.0
    top_k = set(ranked_item_ids[:k])
    hits = len(top_k & relevant)
    return float(hits) / float(k)


def mrr_at_k(ranked_item_ids: Sequence[str], relevant_item_ids: Sequence[str], k: int) -> float:
    """Mean Reciprocal Rank at k (single-query reciprocal rank, not yet meaned).

    Formula: rr_k = 1 / rank_of_first_relevant_in_top_k  (0.0 if none in top-k)
    The "Mean" in MRR is taken across queries by the evaluator; this function
    returns the per-query reciprocal rank.

    :param ranked_item_ids: Sequence[str] - Item ids ordered best -> worst
    :param relevant_item_ids: Sequence[str] - Ids with gain >= 1
    :param k: int - Cutoff rank (>= 1)
    :return: float - Reciprocal rank in [0.0, 1.0]
    :raises ValueError: When k < 1 or any input is malformed
    """
    _validate_k(k)
    _validate_ranked(ranked_item_ids)
    if relevant_item_ids is None:
        raise ValueError("relevant_item_ids must not be None")
    relevant = set(relevant_item_ids)
    if not relevant:
        return 0.0
    for i, item_id in enumerate(ranked_item_ids[:k], start=1):
        if item_id in relevant:
            return 1.0 / float(i)
    return 0.0
