"""Fuzzy lexical reranker — typo / near-match surfacing over the fused pool.

Reorders an already-fused, over-fetched ``RankedResults`` pool so SLD near-matches
rank above the final top_k truncation. Example: query ``high rentals`` boosts the
candidate ``hi-rentals`` (and ``rent.high``) even when its fused score sat below
the cut, so the end user actually sees the near-match.

Scoring (per candidate):
  1. Tokenize the query (shared ``tokenize_lexical`` — same token universe as the
     BM25 sparse leg).
  2. Tokenize the candidate's ``payload['sld']`` into ``[a-z0-9]+`` runs
     (``hi-rentals`` -> ``[hi, rentals]``).
  3. For each query token, take the best per-pair score against the SLD tokens:
     exact substring containment scores 1.0; otherwise normalized
     Damerau-Levenshtein similarity (``core.text_distance.similarity_ratio``).
  4. ``coverage`` = mean of the best per-query-token scores that clear
     ``min_similarity`` (others contribute 0).
  5. ``rerank_score = fused_score + boost_weight * coverage`` — additive so a
     zero-coverage query preserves the fused order exactly (graceful no-op).

Stdlib-only, no re-index, no external model. Works on both the in-memory and the
Qdrant hybrid paths because it operates on the post-fusion ``RankedItem`` list.
"""
import re
from typing import List, Sequence

from semantic_search.config.models import FuzzyRerankConfig
from semantic_search.contracts import RankedItem
from semantic_search.core.exceptions import RetrievalError, ValidationError
from semantic_search.core.logging_utils import get_logger
from semantic_search.core.text_distance import similarity_ratio
from semantic_search.retrieval.lexical_tokenizer import tokenize_lexical
from semantic_search.retrieval.reranker_base import Reranker, RerankedItem, stable_sort_descending

logger = get_logger(__name__)

_SLD_TOKEN_RE = re.compile(r"[a-z0-9]+")


class FuzzyLexicalReranker(Reranker):
    """Deterministic fuzzy lexical reranker keyed on candidate SLD tokens.

    :param config: FuzzyRerankConfig - Validated config (edit distance, weights, caps)
    :raises RetrievalError: When ``config`` is not a FuzzyRerankConfig
    """

    def __init__(self, config: FuzzyRerankConfig) -> None:
        if not isinstance(config, FuzzyRerankConfig):
            raise RetrievalError("FuzzyLexicalReranker requires a FuzzyRerankConfig")
        self._config = config
        self._stopwords = frozenset(s.lower() for s in config.stopwords)

    @property
    def name(self) -> str:
        """Stable backend label for logs / ablation tagging."""
        return 'fuzzy_lexical'

    def _sld_tokens(self, payload: dict) -> List[str]:
        """Extract ``[a-z0-9]+`` runs from the candidate's SLD payload field."""
        sld = payload.get('sld') if isinstance(payload, dict) else None
        if not sld:
            return []
        return _SLD_TOKEN_RE.findall(str(sld).lower())

    def _pair_score(self, q_tok: str, sld_tok: str) -> float:
        """Best similarity for one (query token, sld token) pair in [0, 1].
        Exact substring containment (either direction) scores 1.0; otherwise a
        normalized Damerau-Levenshtein ratio capped at ``max_edit_distance``."""
        if q_tok == sld_tok:
            return 1.0
        if q_tok in sld_tok or sld_tok in q_tok:
            return 1.0
        return similarity_ratio(q_tok, sld_tok, self._config.max_edit_distance)

    def _coverage(self, query_tokens: Sequence[str], sld_tokens: Sequence[str]) -> float:
        """Matched-token fraction scaled by the best per-token score.

        coverage = best_matched_score * (matched_count / total_count)

        Breadth (fraction of query tokens that matched) and precision (quality of
        the strongest match) are combined multiplicatively.  A domain matching all
        query tokens with exact scores produces coverage 1.0; one matching only one
        token produces proportionally less, regardless of whether that single token
        was exact or fuzzy.  When all matched scores are 1.0 this equals the old
        mean formula exactly; the difference appears only on fuzzy (< 1.0) scores,
        where best_matched_score > average_matched_score raises coverage slightly.
        """
        if not query_tokens or not sld_tokens:
            return 0.0
        total = float(len(query_tokens))
        matched_count = 0
        best_matched = 0.0
        for qt in query_tokens:
            best = 0.0
            for st in sld_tokens:
                s = self._pair_score(qt, st)
                if s > best:
                    best = s
                if best >= 1.0:
                    break
            if best >= self._config.min_similarity:
                matched_count += 1
                if best > best_matched:
                    best_matched = best
        if matched_count == 0:
            return 0.0
        return best_matched * (matched_count / total)

    def rerank(self, query: str, items: Sequence[RankedItem], top_n: int) -> List[RerankedItem]:
        """Score and reorder the first ``top_n`` items by fused_score + fuzzy boost.

        :param query: str - Normalized query text
        :param items: Sequence[RankedItem] - Fused pool, fused-score-descending
        :param top_n: int - Number of head items to rescore (>= 0)
        :return: List[RerankedItem] - Length min(top_n, len(items)), rerank-score-descending
        :raises ValidationError: When ``top_n`` is negative
        """
        if top_n < 0:
            raise ValidationError("FuzzyLexicalReranker.rerank top_n must be >= 0")
        if top_n == 0 or not items:
            return []
        head = list(items[:top_n])
        query_tokens = tokenize_lexical(text=query or '', min_term_length=self._config.min_token_length, max_terms=self._config.max_terms, stopwords=self._stopwords)
        scored: List[RerankedItem] = []
        for item in head:
            coverage = self._coverage(query_tokens, self._sld_tokens(item.payload))
            rerank_score = float(item.fused_score) + (self._config.boost_weight * coverage)
            scored.append(RerankedItem(item=item, rerank_score=rerank_score))
        return stable_sort_descending(scored)
