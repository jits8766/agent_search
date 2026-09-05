"""Deterministic brandability scorer (assessment item 4).

Scores a domain SLD in ``[0, 1]`` from config-weighted, language-agnostic
features — name length, vowel/consonant balance, pronounceability (penalising
long consonant runs), and digit/hyphen penalties. The score is a ranking signal
for "rank by brandability"/"memorable name" queries, where the prior pipeline had
no brandability feature at all (brandability was a soft chip with no scorer).

A learned head can later replace ``score`` by loading weights trained on
LLM-judged brandability labels (see plan); this module is the deterministic
default and the reversible fallback. All thresholds/weights come from
``BrandabilityConfig`` — no magic numbers in logic.
"""
from typing import Dict, List

from semantic_search.config.models import BrandabilityConfig

_VOWELS = frozenset('aeiou')


class BrandabilityScorer:
    """Compute a deterministic brandability score for a domain SLD.

    :param config: BrandabilityConfig - Feature weights, penalties, and bounds
    """

    def __init__(self, config: BrandabilityConfig) -> None:
        self._config = config

    @staticmethod
    def _clamp01(value: float) -> float:
        """Clamp a float to ``[0, 1]``.
        :param value: float - Raw value
        :return: float - Clamped value
        """
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    def _length_score(self, sld: str) -> float:
        """Higher for SLDs near the short end of ``[min_length, max_length]``.
        :param sld: str - Second-level domain label
        :return: float - Length sub-score in ``[0, 1]``
        """
        cfg = self._config
        n = len(sld)
        if n <= cfg.min_length:
            return 1.0
        if n >= cfg.max_length:
            return 0.0
        span = float(cfg.max_length - cfg.min_length)
        return self._clamp01(1.0 - (n - cfg.min_length) / span)

    def _vowel_balance_score(self, sld: str) -> float:
        """Higher when the vowel ratio is near ``ideal_vowel_ratio``.
        :param sld: str - Second-level domain label
        :return: float - Vowel-balance sub-score in ``[0, 1]``
        """
        letters = [c for c in sld if c.isalpha()]
        if not letters:
            return 0.0
        ratio = sum(1 for c in letters if c in _VOWELS) / float(len(letters))
        ideal = self._config.ideal_vowel_ratio
        return self._clamp01(1.0 - abs(ratio - ideal) / ideal)

    def _pronounceability_score(self, sld: str) -> float:
        """Penalise consonant runs longer than ``max_consonant_run``.
        :param sld: str - Second-level domain label
        :return: float - Pronounceability sub-score in ``[0, 1]``
        """
        letters = [c for c in sld if c.isalpha()]
        if not letters:
            return 0.0
        longest_run = 0
        run = 0
        for c in letters:
            if c in _VOWELS:
                run = 0
            else:
                run += 1
                longest_run = max(longest_run, run)
        excess = max(0, longest_run - self._config.max_consonant_run)
        return self._clamp01(1.0 - excess / float(len(letters)))

    def score(self, sld: str) -> float:
        """Return the brandability score in ``[0, 1]`` for ``sld``.
        :param sld: str - Second-level domain label (TLD excluded)
        :return: float - Weighted brandability score
        """
        cfg = self._config
        s = str(sld).lower()
        if not s:
            return 0.0
        sub = (
            cfg.weight_length * self._length_score(s)
            + cfg.weight_vowel_balance * self._vowel_balance_score(s)
            + cfg.weight_pronounceability * self._pronounceability_score(s)
        )
        weight_sum = cfg.weight_length + cfg.weight_vowel_balance + cfg.weight_pronounceability
        base = sub / weight_sum if weight_sum > 0.0 else 0.0
        if any(c.isdigit() for c in s):
            base -= cfg.digit_penalty
        if '-' in s:
            base -= cfg.hyphen_penalty
        return self._clamp01(base)

    def boosted_scores(self, sld_to_fused: Dict[str, float]) -> Dict[str, float]:
        """Return ``fused * (1 + boost_weight * brandability)`` per SLD.

        Multiplicative boost preserves the fused ranking's scale while promoting
        more brandable names; used by the orchestrator's brandability rerank stage.

        :param sld_to_fused: Dict[str, float] - SLD -> fused score
        :return: Dict[str, float] - SLD -> boosted score
        """
        bw = self._config.boost_weight
        return {sld: fused * (1.0 + bw * self.score(sld)) for sld, fused in sld_to_fused.items()}
