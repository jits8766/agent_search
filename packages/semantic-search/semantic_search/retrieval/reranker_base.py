"""Reranker protocol surface (LAYER 4 extension / — Cross-encoder reranker).

Slots BETWEEN the Layer-4 ``DeterministicRanker`` (when enabled in a retrieval path) and downstream eRanker /
zero-result-guard / cache-write. The contract is:

* Input: the post-ranker ``RankedResults`` (already in fused-score-descending order)
  + the query string + a ``top_n`` cap. Items at positions ``> top_n`` are NOT
  reranked — the long tail keeps its order.
* Output: a list of ``RerankedItem`` carrying the reranker's relevance score plus the
  original ``RankedItem`` (no mutation; the orchestrator re-sorts the top-N slice in
  place using the reranker score and concatenates the untouched tail).

Latency gating is the orchestrator's responsibility (``asyncio.wait_for`` around
``Reranker.rerank``); on timeout / exception the orchestrator MUST keep the
deterministic-ranker order. The reranker itself is therefore allowed to raise —
it should not implement its own circuit breaker.

This module is a contract-only module — stdlib + typing only. The concrete
backend is ``fuzzy_lexical_reranker.FuzzyLexicalReranker`` (deterministic,
stdlib-only) and an extension seam exists for a future model-backed
``CrossEncoderReranker`` (FlagReranker / sentence-transformers; not wired
because ``fastembed==0.8.0`` does not expose a cross-encoder API and we do not
introduce a new heavy dependency without explicit approval).
"""
from dataclasses import dataclass
from typing import List, Sequence

from semantic_search.contracts import RankedItem
from semantic_search.core.exceptions import ValidationError


@dataclass(frozen=True)
class RerankedItem:
    """A reranked item: the original ``RankedItem`` plus the reranker's score.

    :param item: RankedItem - The unchanged Layer-4 item (id, fused_score,
        contributing_sources, payload, sub_intent_ids stay verbatim — the
        orchestrator re-sorts this list and never mutates the underlying object)
    :param rerank_score: float - The reranker's relevance score; HIGHER MEANS MORE
        RELEVANT for ALL conforming backends (the orchestrator sorts descending).
        No upper bound is enforced (a model-backed cross-encoder may emit a
        sigmoid-of-logit in [0, 1]; a lexical reranker may emit an unbounded
        non-negative score; both are valid). The lower bound is 0.0.
    """
    item: RankedItem
    rerank_score: float

    def __post_init__(self) -> None:
        if not isinstance(self.item, RankedItem):
            raise ValidationError("RerankedItem.item must be a RankedItem")
        if float(self.rerank_score) < 0.0:
            raise ValidationError("RerankedItem.rerank_score must be >= 0.0")


class Reranker:
    """Abstract reranker protocol.

    Implementations score (query, item) pairs and return them in
    rerank-score-descending order. The contract is intentionally NOT async —
    every shipped backend in this PR is pure-CPU and runs in microseconds; the
    orchestrator wraps the call in ``asyncio.to_thread`` if it ever blocks the
    loop. Model-backed backends (future) MUST also expose the same sync surface
    and rely on the orchestrator's ``to_thread`` wrapper for off-loop execution.
    """

    @property
    def name(self) -> str:
        """Return a stable backend label (logged + used for ablation tagging)."""
        raise NotImplementedError

    def rerank(self, query: str, items: Sequence[RankedItem], top_n: int) -> List[RerankedItem]:
        """Score and reorder the top-N slice of an already-ranked list.

        :param query: str - The normalized query text
            (``QueryIntent.normalized_query`` at the call site)
        :param items: Sequence[RankedItem] - The full Layer-4 ranked list in
            fused-score-descending order. The reranker SCORES ONLY THE FIRST
            ``top_n`` items; items beyond ``top_n`` are NOT in the returned
            list (the orchestrator handles the tail concat).
        :param top_n: int - How many head items to score (>= 0). When 0 or
            ``items`` is empty, the implementation MUST return ``[]``.
        :return: List[RerankedItem] - Length ``min(top_n, len(items))``,
            ordered by ``rerank_score`` descending. Stable tie-break on the
            input order (i.e. on ``fused_score`` descending).
        :raises ValidationError: When inputs violate the contract
        """
        raise NotImplementedError


def stable_sort_descending(scored: Sequence[RerankedItem]) -> List[RerankedItem]:
    """Stable-sort ``RerankedItem`` by ``rerank_score`` descending.

    Centralized here so every ``Reranker`` backend gets the same tie-break
    semantics: equal ``rerank_score`` keeps the input order (which the
    orchestrator hands in as fused-score-descending), so a reranker that
    produces a flat score across the head degrades gracefully to the
    deterministic-ranker order.

    :param scored: Sequence[RerankedItem] - Scored items in any order
    :return: List[RerankedItem] - Same items, descending by rerank_score, stable
    """
    return sorted(scored, key=lambda r: r.rerank_score, reverse=True)
