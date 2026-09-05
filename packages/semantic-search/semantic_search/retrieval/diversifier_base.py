"""Diversifier protocol surface (LAYER 4 extension / — MMR / diversity on the final top-K).

Slots AFTER external eRanker (Layer 4 ranking) and BEFORE the orchestrator's
``_truncate`` cut to ``general.max_results``. The contract is:

* Input: the post-eRanker ``RankedResults`` (already in fused-score-then-eRanker
  order) + the query string + a
  ``top_n`` cap (the head we will actually re-shuffle for diversity) + an
  ``output_n`` cap (the number of items the diversifier should select with
  MMR — must be ``<= top_n``). Items at positions ``> top_n`` are NOT
  re-shuffled — the long tail keeps its order.
* Output: a list of ``DiversifiedItem`` carrying the diversifier's per-item
  base score + diversity penalty + final MMR score, plus the original
  ``RankedItem`` (no mutation; the orchestrator re-stitches the diversified
  head onto the un-touched tail and the un-selected head remainder).

Latency gating is the orchestrator's responsibility (``asyncio.wait_for``
around ``Diversifier.diversify``); on timeout / exception the orchestrator
MUST keep the post-eRanker order. The diversifier itself is
therefore allowed to raise — it should not implement its own circuit breaker.

This module is a contract-only module — stdlib + typing only. Concrete
backends live in ``lexical_jaccard_diversifier.py`` (deterministic,
stdlib-only, ships in this PR) and an extension seam exists for a future
embedding-backed ``EmbeddingMMRDiversifier`` (uses the package
``semantic_search.qi.encoder.Encoder`` to compute pairwise cosine similarity;
not in this PR because the per-item embedding encode would dominate the
latency budget on the default lexical backend's typical workload, and we
do not introduce that cost without explicit approval).
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence

from semantic_search.contracts import RankedItem
from semantic_search.core.exceptions import ValidationError


@dataclass(frozen=True)
class DiversifiedItem:
    """A diversified item: the original ``RankedItem`` plus the MMR scoring trace.

    :param item: RankedItem - The unchanged post-eRanker item (id,
        fused_score, contributing_sources, payload, sub_intent_ids stay
        verbatim — the orchestrator re-stitches this list and never mutates
        the underlying object)
    :param base_score: float - The diversifier's per-item relevance proxy in
        ``[0, 1]``. Computed by the backend from input rank (so the
        diversifier does not depend on the absolute scale of fused_score,
        which can drift across eRanker reordering).
    :param diversity_penalty: float - Maximum similarity to any
        already-selected item in ``[0, 1]``. ``0.0`` for the first selected
        item (no prior selection to compare against).
    :param mmr_score: float - The MMR-blended score the greedy selector
        used to pick this item:
        ``lambda_relevance * base_score - (1 - lambda_relevance) * diversity_penalty``.
        Stamped on the result so the audit log shows exactly why each
        position was picked.
    """
    item: RankedItem
    base_score: float
    diversity_penalty: float
    mmr_score: float

    def __post_init__(self) -> None:
        if not isinstance(self.item, RankedItem):
            raise ValidationError("DiversifiedItem.item must be a RankedItem")
        if not (0.0 <= float(self.base_score) <= 1.0):
            raise ValidationError("DiversifiedItem.base_score must be in [0, 1]")
        if not (0.0 <= float(self.diversity_penalty) <= 1.0):
            raise ValidationError("DiversifiedItem.diversity_penalty must be in [0, 1]")
        # mmr_score is `lambda * base - (1-lambda) * penalty`; with both
        # operands in [0,1] and lambda in [0,1] the result is bounded in
        # [-1, 1]. Enforce that range so a backend that returns NaN / inf
        # is rejected at the contract boundary.
        s = float(self.mmr_score)
        if s != s or s < -1.0 or s > 1.0:
            raise ValidationError("DiversifiedItem.mmr_score must be a finite number in [-1, 1]")


class Diversifier:
    """Abstract diversifier protocol.

    Implementations re-shuffle the head ``top_n`` items of an already-ranked
    list using a relevance-vs-novelty trade-off (MMR by default) and return
    the first ``output_n`` items in selection order. The contract is
    intentionally NOT async — every shipped backend in this PR is pure-CPU
    and runs in microseconds; the orchestrator wraps the call in
    ``asyncio.to_thread`` if it ever blocks the loop. Model-backed
    backends (future) MUST also expose the same sync surface and rely on
    the orchestrator's ``to_thread`` wrapper for off-loop execution.
    """

    @property
    def name(self) -> str:
        """Return a stable backend label (logged + used for ablation tagging)."""
        raise NotImplementedError

    def diversify(self, query: str, items: Sequence[RankedItem], top_n: int, output_n: int, lambda_override: Optional[float] = None) -> List[DiversifiedItem]:
        """Greedy-MMR-select ``output_n`` items from the head ``top_n`` slice.

        :param query: str - The normalized query text
            (``QueryIntent.normalized_query`` at the call site). Some
            backends (lexical-jaccard) ignore the query; future
            embedding-backed backends use it to compute query→item
            relevance instead of the rank-prior fallback.
        :param items: Sequence[RankedItem] - The post-eRanker
            ranked list in score-descending order. The diversifier
            considers ONLY the first ``top_n`` items as MMR candidates;
            items beyond ``top_n`` are NOT in the returned list (the
            orchestrator handles the tail concat).
        :param top_n: int - How many head items participate in MMR
            selection (>= 0). The diversifier is allowed to look across
            the full top_n to maximise diversity even if it only
            selects ``output_n``.
        :param output_n: int - How many items to actually select (0 <=
            output_n <= top_n). When 0 or ``items`` is empty, the
            implementation MUST return ``[]``. When
            ``output_n >= top_n``, the entire head slice is returned
            in selection order (which may differ from input order).
        :return: List[DiversifiedItem] - Length ``min(output_n, len(items))``,
            in selection order (so position 0 is the highest-MMR pick at
            step 1, position 1 is the highest-MMR pick at step 2 against
            the partially-built selection set, etc.). Items are NOT
            sorted by ``mmr_score`` after selection — selection order IS
            the output order.
        :raises ValidationError: When inputs violate the contract
        """
        raise NotImplementedError
