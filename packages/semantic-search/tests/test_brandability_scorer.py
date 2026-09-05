"""Tests for the deterministic brandability scorer (assessment item 4)."""
import pytest

from semantic_search.config.models import BrandabilityConfig
from semantic_search.retrieval.brandability_scorer import BrandabilityScorer


def _cfg(**over):
    base = dict(enabled=True, weight_length=1.0, weight_vowel_balance=1.0, weight_pronounceability=1.0,
                ideal_vowel_ratio=0.4, min_length=3, max_length=15, max_consonant_run=3,
                digit_penalty=0.3, hyphen_penalty=0.3, boost_weight=0.25, trigger_terms=['brandable'])
    base.update(over)
    return BrandabilityConfig(**base)


class TestScoreRange:
    def test_score_in_unit_interval(self):
        s = BrandabilityScorer(_cfg())
        for sld in ('stripe', 'zoom', 'x7y9q2zzz', 'my-long-hyphen-domain-9', ''):
            assert 0.0 <= s.score(sld) <= 1.0

    def test_empty_is_zero(self):
        assert BrandabilityScorer(_cfg()).score('') == 0.0


class TestRelativeOrdering:
    def test_brandable_beats_unpronounceable(self):
        s = BrandabilityScorer(_cfg())
        assert s.score('stripe') > s.score('xkcdzz')

    def test_digit_penalised(self):
        s = BrandabilityScorer(_cfg())
        assert s.score('zooma') > s.score('zoom4a')

    def test_hyphen_penalised(self):
        s = BrandabilityScorer(_cfg())
        assert s.score('payfast') > s.score('pay-fast')

    def test_short_beats_long(self):
        s = BrandabilityScorer(_cfg())
        assert s.score('nova') > s.score('novanovanova')


class TestBoost:
    def test_boost_preserves_or_promotes(self):
        s = BrandabilityScorer(_cfg(boost_weight=0.5))
        out = s.boosted_scores({'stripe': 1.0, 'xkcdzz': 1.0})
        assert out['stripe'] > out['xkcdzz']

    def test_zero_boost_weight_is_identity(self):
        s = BrandabilityScorer(_cfg(boost_weight=0.0))
        out = s.boosted_scores({'stripe': 2.0})
        assert out['stripe'] == 2.0


class TestConfigValidation:
    def test_bad_vowel_ratio_rejected(self):
        with pytest.raises(Exception):
            _cfg(ideal_vowel_ratio=0.0)

    def test_bad_length_band_rejected(self):
        with pytest.raises(Exception):
            _cfg(min_length=10, max_length=5)
