"""Lexical-Jaccard MMR diversifier (LAYER 4 extension / — MMR diversity, lexical backend).

Stdlib-only, deterministic ``Diversifier`` backend that re-shuffles the
head ``top_n`` items of a ``RankedResults`` to improve novelty in the
final ``output_n`` slice. Ships in this PR as the optimized-for default
backend; a future embedding-backed ``EmbeddingMMRDiversifier`` (uses the
package ``Encoder``) is a drop-in swap because both implement the same
``Diversifier`` protocol.

Math (Carbonell & Goldstein 1998, MMR):

    MMR(item_i) = lambda * relevance(item_i)
                  - (1 - lambda) * max_{j in S} sim(item_i, item_j)

where ``S`` is the already-selected set. ``lambda in [0, 1]`` is the
relevance-vs-novelty knob; ``lambda=1.0`` reduces to identity (pure
relevance, no diversity), ``lambda=0.0`` is pure novelty after the first
pick. Production defaults sit in ``[0.5, 0.8]``.

Per-pair similarity:

* Tokenize each item's payload by concatenating the configured payload
  fields (``payload_fields`` config, e.g. ``['title', 'description', 'tld']``)
  with the SAME tokenizer used by BM25 + the lexical reranker
  (``retrieval/lexical_tokenizer.py``) — guarantees the diversifier scores
  on the same token universe as BM25 and the reranker.
* Use the SET (not bag) of surviving tokens — Jaccard is set-defined.
* Similarity = ``|A & B| / |A | B|`` in ``[0, 1]``. When either set is
  empty, similarity = 0 (an empty payload cannot duplicate anything).

Per-item base relevance:

* Use the input rank: ``base = 1.0 / (1 + input_rank)``, bounded in
  ``(0, 1]``. We deliberately do NOT use ``RankedItem.fused_score``
  because its absolute scale drifts after upstream reordering stages —
  using rank keeps the MMR blend numerically
  comparable across requests and lambda values.

Greedy selection:

* Step 1: pick the item with the highest ``base`` (= rank 0 — equivalent
  to input order on the first step). Stamp ``diversity_penalty=0`` and
  ``mmr_score = lambda * base``.
* Step k>1: for each remaining candidate compute
  ``penalty = max(sim(candidate, selected_j) for j in S)`` and
  ``mmr = lambda * base - (1-lambda) * penalty``; pick the
  ``argmax(mmr)``. Stable tie-break on lower input rank (= higher
  base relevance), so equal-MMR ties keep the eRanker input order.

Robustness:

* Empty ``items`` or ``output_n=0`` → return ``[]``.
* ``output_n > top_n`` → cap to ``top_n`` (defensive — the orchestrator
  also caps).
* All payloads empty → every similarity = 0 → greedy reduces to
  base-relevance order (= input order). The diversifier is therefore a
  safe no-op for items with no text payload.
* Lambda = 1.0 → diversity term zero out → output = input order on the
  head ``output_n``.
* Lambda = 0.0 → relevance term zero out (after first pick) → output
  is the most-diverse permutation of the head; first pick falls back to
  input order via the tie-break.

Layer placement: ``retrieval/`` (sits beside ``ranker.py``,
``fuzzy_lexical_reranker.py``). Per ``architecture.mdc``: imports
stdlib + ``contracts`` + ``core.exceptions`` + sibling ``retrieval/``
modules only. Imports NO higher layer.
"""
from typing import List, Optional, Sequence, Set, Tuple

from semantic_search.config.models import LexicalDiversityConfig
from semantic_search.contracts import RankedItem
from semantic_search.core.exceptions import DiversityError, ValidationError
from semantic_search.core.logging_utils import get_logger
from semantic_search.retrieval.diversifier_base import Diversifier, DiversifiedItem
from semantic_search.retrieval.lexical_tokenizer import tokenize_lexical

logger = get_logger(__name__)


class LexicalJaccardDiversifier(Diversifier):
    """Stdlib-only Jaccard MMR diversifier; default backend behind the ``Diversifier`` protocol.

    See module docstring for the contract, math, and rationale.

    :param config: LexicalDiversityConfig - Validated config bundle (lambda,
        payload fields, tokenizer params)
    """

    _NAME = "lexical_jaccard_mmr"

    def __init__(self, config: LexicalDiversityConfig):
        if not isinstance(config, LexicalDiversityConfig):
            raise DiversityError("LexicalJaccardDiversifier requires a LexicalDiversityConfig")
        self._lambda = float(config.lambda_relevance)
        self._payload_fields = tuple(config.payload_fields)
        self._min_term_length = int(config.min_term_length)
        self._max_terms = int(config.max_terms)
        self._stopwords = frozenset(s.lower() for s in config.stopwords)

    @property
    def name(self) -> str:
        """Stable backend label (``"lexical_jaccard_mmr"``) for logs + ablation tagging."""
        return self._NAME

    def _extract_doc_text(self, item: RankedItem) -> str:
        """Concatenate the configured payload fields into one document string.

        Mirrors ``LexicalOverlapReranker._extract_doc_text`` so the two
        backends score on identical doc representations. Missing fields are
        skipped silently; non-string values are coerced via ``str()``;
        list/tuple values are flattened (each element coerced separately).
        """
        parts: List[str] = []
        for fld in self._payload_fields:
            value = item.payload.get(fld)
            if value is None:
                continue
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, (list, tuple)):
                for elem in value:
                    if elem is None:
                        continue
                    parts.append(str(elem))
            else:
                parts.append(str(value))
        return ' '.join(parts)

    def _token_set(self, text: str) -> Set[str]:
        """Tokenize then deduplicate to a set (Jaccard is set-defined)."""
        return set(tokenize_lexical(text=text, min_term_length=self._min_term_length, max_terms=self._max_terms, stopwords=self._stopwords))

    @staticmethod
    def _jaccard(a: Set[str], b: Set[str]) -> float:
        """Jaccard set-similarity in ``[0, 1]``; 0 when either set is empty.

        :param a: Set[str] - Token set A
        :param b: Set[str] - Token set B
        :return: float - ``|A ∩ B| / |A ∪ B|``; 0.0 when either is empty
        """
        if not a or not b:
            return 0.0
        union_size = len(a | b)
        if union_size == 0:
            return 0.0
        return float(len(a & b)) / float(union_size)

    def diversify(self, query: str, items: Sequence[RankedItem], top_n: int, output_n: int, lambda_override: Optional[float] = None) -> List[DiversifiedItem]:
        """Greedy-MMR select ``output_n`` items from the head ``top_n`` slice.

        Implements the ``Diversifier.diversify`` contract; see ``diversifier_base.py``.
        ``lambda_override`` replaces the config-set lambda for this call only.
        """
        if query is not None and not isinstance(query, str):
            raise ValidationError("LexicalJaccardDiversifier.diversify requires query to be str or None")
        if not isinstance(items, (list, tuple)):
            raise ValidationError("LexicalJaccardDiversifier.diversify requires items to be a list/tuple")
        if int(top_n) < 0:
            raise ValidationError("LexicalJaccardDiversifier.diversify requires top_n >= 0")
        if int(output_n) < 0:
            raise ValidationError("LexicalJaccardDiversifier.diversify requires output_n >= 0")
        if output_n == 0 or top_n == 0 or not items:
            return []

        head_n = min(int(top_n), len(items))
        out_n = min(int(output_n), head_n)
        head = items[:head_n]
        lam = float(lambda_override) if lambda_override is not None else self._lambda

        # Pre-compute token sets once per item (head_n tokenizations, not
        # head_n^2). Pairwise similarity is then a set-op on cached sets.
        token_sets: List[Set[str]] = [self._token_set(self._extract_doc_text(it)) for it in head]
        # Pre-compute base relevance per input rank.
        base_relevance: List[float] = [1.0 / (1.0 + float(rank)) for rank in range(head_n)]

        selected_indices: List[int] = []
        # Per-candidate cached "max similarity to selected so far". Updated
        # incrementally as each new item joins ``selected_indices`` so the
        # total work is O(head_n * out_n) similarity comparisons, not
        # O(head_n * out_n^2).
        max_sim_to_selected: List[float] = [0.0] * head_n

        results: List[DiversifiedItem] = []
        remaining: Set[int] = set(range(head_n))

        for step in range(out_n):
            # Score every remaining candidate at this step.
            best_idx = -1
            best_mmr = float('-inf')
            best_rank = head_n + 1  # tie-break: lower input rank wins
            for idx in remaining:
                base = base_relevance[idx]
                penalty = max_sim_to_selected[idx] if step > 0 else 0.0
                mmr = lam * base - (1.0 - lam) * penalty
                # Greedy argmax with deterministic tie-break: lower input
                # rank (= earlier in the personalized order) wins.
                if mmr > best_mmr or (mmr == best_mmr and idx < best_rank):
                    best_idx = idx
                    best_mmr = mmr
                    best_rank = idx
            if best_idx < 0:
                # Defensive: should not happen since `remaining` is non-empty
                # while step < out_n <= head_n. Documented as defence-in-depth
                # — if it ever fires, the orchestrator's soft-fail path
                # preserves the input order.
                break
            chosen_penalty = max_sim_to_selected[best_idx] if step > 0 else 0.0
            chosen_base = base_relevance[best_idx]
            results.append(DiversifiedItem(item=head[best_idx], base_score=chosen_base, diversity_penalty=chosen_penalty, mmr_score=best_mmr))
            selected_indices.append(best_idx)
            remaining.discard(best_idx)
            # Update each remaining candidate's max similarity against the
            # newly-selected item. New max = max(old max, sim(candidate, new)).
            chosen_tokens = token_sets[best_idx]
            for idx in remaining:
                new_sim = self._jaccard(token_sets[idx], chosen_tokens)
                if new_sim > max_sim_to_selected[idx]:
                    max_sim_to_selected[idx] = new_sim
        return results


class NoOpDiversifier(Diversifier):
    """Identity diversifier used when ``retrieval.diversity.enabled=false``.

    Returns the head ``output_n`` items in their input order with
    ``base_score = 1/(1+rank)``, ``diversity_penalty=0``,
    ``mmr_score = base_score`` (i.e. lambda=1.0 implicitly). This backend
    exists so the orchestrator can always call ``diversifier.diversify``
    without a ``None`` check — the disabled-path is a typed no-op, not a
    missing component.
    """

    _NAME = "noop"

    @property
    def name(self) -> str:
        return self._NAME

    def diversify(self, query: str, items: Sequence[RankedItem], top_n: int, output_n: int, lambda_override: Optional[float] = None) -> List[DiversifiedItem]:
        if query is not None and not isinstance(query, str):
            raise ValidationError("NoOpDiversifier.diversify requires query to be str or None")
        if not isinstance(items, (list, tuple)):
            raise ValidationError("NoOpDiversifier.diversify requires items to be a list/tuple")
        if int(top_n) < 0:
            raise ValidationError("NoOpDiversifier.diversify requires top_n >= 0")
        if int(output_n) < 0:
            raise ValidationError("NoOpDiversifier.diversify requires output_n >= 0")
        if output_n == 0 or top_n == 0 or not items:
            return []
        head_n = min(int(top_n), len(items))
        out_n = min(int(output_n), head_n)
        head = items[:out_n]
        return [DiversifiedItem(item=it, base_score=1.0 / (1.0 + float(rank)), diversity_penalty=0.0, mmr_score=1.0 / (1.0 + float(rank))) for rank, it in enumerate(head)]
