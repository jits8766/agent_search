"""Retriever interface contracts.
Every retrieval backend implements `Retriever.retrieve(intent, top_k) -> CandidateSet`
and reports its `source` (vector / structured / sql) on the candidate set so the
fusion layer can apply per-source weighting.
"""
from typing import Iterable

from semantic_search.contracts import CandidateSet, QueryIntent


class Retriever:
    """Abstract retriever protocol surface."""

    @property
    def source(self) -> str:
        """Return the candidate-source label this retriever produces."""
        raise NotImplementedError

    @property
    def provides_inventory_estimate(self) -> bool:
        """True when ``retrieve`` counts reflect real inventory for pre-screen.

        Multi-intent zero-expected drop uses structured pre-screen counts. No-op
        structured legs (hybrid Qdrant) always return empty sets and must report
        False so slices are not dropped as empty inventory.
        """
        return True

    async def retrieve(self, intent: QueryIntent, top_k: int) -> CandidateSet:
        """Retrieve up to `top_k` candidates for the given intent.
        :param intent: QueryIntent - Classified query intent
        :param top_k: int - Maximum number of candidates to return
        :return: CandidateSet - Backend candidates with source + latency_ms
        """
        raise NotImplementedError


def slice_candidates(items: Iterable, top_k: int) -> list:
    """Truncate an iterable to top-K. Centralized to avoid duplicate `[:k]` literals.
    :param items: Iterable - Source items in score-descending order
    :param top_k: int - Desired length cap
    :return: list - Truncated list
    """
    if top_k < 1:
        return []
    return list(items)[:top_k]
