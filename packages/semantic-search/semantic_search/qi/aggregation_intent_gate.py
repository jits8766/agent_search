"""Structural gate for aggregate-question intent detection at the L0_fallback decision point.

Fires when BOTH conditions hold:
  A) Query names a marketplace object (listing, auction, tld, ...).
  B) Query carries an aggregate operator — either a strong operator on its own
     (count, average, distribution, ...) or a weak/popularity operator (top,
     most, trend, ...) paired with a metric/grouping companion (by, per,
     volume, ...). The companion requirement separates analytics leaderboards
     ('top tlds by listing volume') from explore popularity browses
     ('most viewed domains').

All term lists and the word-boundary length threshold come from config; the
class holds no literal vocabulary. Designed as a standalone injectable so it can
be tested independently of the QI cascade and swapped or disabled via config
without touching engine logic.
"""
import re
from typing import List, Optional, Pattern, Tuple

from semantic_search.config.models import QIAggregationConfig
from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)


class AggregationIntentGate:
    """Detect aggregate questions: marketplace noun + strong/weak operator (+ companion)."""

    def __init__(self, config: QIAggregationConfig) -> None:
        if config is None:
            raise ValueError("AggregationIntentGate requires a QIAggregationConfig instance")
        self._enabled: bool = config.enabled
        self._wb_max_length: int = config.word_boundary_max_length
        # Marketplace nouns stay raw substrings so singular/plural forms both
        # match. Operators and companions are pre-compiled into (literal,
        # boundary-pattern) matchers per the configured length threshold.
        self._nouns: List[str] = list(config.marketplace_nouns)
        self._strong: List[Tuple[str, Optional[Pattern[str]]]] = self._compile(config.strong_operators)
        self._weak: List[Tuple[str, Optional[Pattern[str]]]] = self._compile(config.weak_operators)
        self._companions: List[Tuple[str, Optional[Pattern[str]]]] = self._compile(config.weak_operator_companions)

    def _compile(self, terms: List[str]) -> List[Tuple[str, Optional[Pattern[str]]]]:
        """Compile terms: word-boundary pattern if <= max_length, else raw substring."""
        out: List[Tuple[str, Optional[Pattern[str]]]] = []
        for t in terms:
            if len(t) <= self._wb_max_length:
                out.append((t, re.compile(r"\b" + re.escape(t) + r"\b")))
            else:
                out.append((t, None))
        return out

    @property
    def enabled(self) -> bool:
        """True when the gate is active."""
        return self._enabled

    def _first_noun(self, q: str) -> Optional[str]:
        """Leftmost-occurring marketplace noun, or None."""
        best: Optional[str] = None
        best_idx = len(q) + 1
        for n in self._nouns:
            idx = q.find(n)
            if idx != -1 and idx < best_idx:
                best_idx, best = idx, n
        return best

    def _first_match(self, q: str, matchers: List[Tuple[str, Optional[Pattern[str]]]]) -> Optional[str]:
        """Leftmost-occurring operator/companion (boundary-aware if pattern available)."""
        best: Optional[str] = None
        best_idx = len(q) + 1
        for term, pat in matchers:
            if pat is not None:
                m = pat.search(q)
                idx = m.start() if m is not None else -1
            else:
                idx = q.find(term)
            if idx != -1 and idx < best_idx:
                best_idx, best = idx, term
        return best

    def is_analytics(self, query: str) -> bool:
        """Return True when the query structurally forms an aggregate question over marketplace data.

        Requires a marketplace noun AND an aggregate operator (a strong operator
        alone, or a weak operator paired with a companion term). A marketplace
        noun alone, or an operator with no marketplace noun, returns False.

        :param query: str - Lowercased, normalized query text.
        :return: bool - True when both structural signals are detected.
        """
        if not self._enabled or not query:
            return False
        q = query.lower()
        if self._first_noun(q) is None:
            return False
        if self._first_match(q, self._strong) is not None:
            return True
        if self._first_match(q, self._weak) is not None and self._first_match(q, self._companions) is not None:
            return True
        return False

    def explain(self, query: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Return (fired, matched_operator, matched_noun) for diagnostics and test assertions.

        The operator is the strong operator when one matched, otherwise the weak
        operator that fired alongside a companion. Both terms are the
        leftmost-occurring matches so the diagnostic is deterministic.

        :param query: str - Normalized query text.
        :return: Tuple[bool, Optional[str], Optional[str]] - (fired, operator_that_matched, noun_that_matched)
        """
        if not self._enabled or not query:
            return False, None, None
        q = query.lower()
        noun = self._first_noun(q)
        if noun is None:
            return False, None, None
        strong = self._first_match(q, self._strong)
        if strong is not None:
            logger.debug(f"aggregation_gate query_len={len(query)} operator={strong} noun={noun} tier=strong")
            return True, strong, noun
        weak = self._first_match(q, self._weak)
        if weak is not None and self._first_match(q, self._companions) is not None:
            logger.debug(f"aggregation_gate query_len={len(query)} operator={weak} noun={noun} tier=weak")
            return True, weak, noun
        return False, None, noun
