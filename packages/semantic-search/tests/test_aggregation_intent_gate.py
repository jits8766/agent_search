"""Unit tests for AggregationIntentGate and related L0_fallback changes.

Coverage matrix (per testing.mdc §7):

AggregationIntentGate.is_analytics:
- noun_plus_strong_operator_fires           -> TestAggregationIntentGate::test_noun_plus_strong_operator_fires
- noun_plus_weak_operator_with_companion_fires -> TestAggregationIntentGate::test_noun_plus_weak_operator_with_companion_fires
- weak_operator_without_companion_returns_false -> TestAggregationIntentGate::test_weak_operator_without_companion_returns_false
- operator_without_noun_returns_false        -> TestAggregationIntentGate::test_operator_without_noun_returns_false
- noun_without_operator_returns_false        -> TestAggregationIntentGate::test_noun_without_operator_returns_false
- disabled_gate_always_returns_false         -> TestAggregationIntentGate::test_disabled_gate_always_returns_false
- empty_query_returns_false                  -> TestAggregationIntentGate::test_empty_query_returns_false
- case_insensitive_match                     -> TestAggregationIntentGate::test_case_insensitive_match
- short_operator_word_boundary_no_substring  -> TestAggregationIntentGate::test_short_operator_word_boundary_no_substring
- plural_noun_substring_match                -> TestAggregationIntentGate::test_plural_noun_substring_match

AggregationIntentGate.explain:
- explain_returns_operator_and_noun          -> TestAggregationIntentGate::test_explain_returns_operator_and_noun
- explain_weak_tier_returns_weak_operator     -> TestAggregationIntentGate::test_explain_weak_tier_returns_weak_operator
- explain_returns_false_on_no_noun           -> TestAggregationIntentGate::test_explain_returns_false_on_no_noun
- explain_disabled_returns_false             -> TestAggregationIntentGate::test_explain_disabled_returns_false

QIAggregationConfig validation:
- empty_nouns_rejected                       -> TestQIAggregationConfig::test_empty_nouns_rejected
- empty_strong_operators_rejected            -> TestQIAggregationConfig::test_empty_strong_operators_rejected
- non_positive_boundary_length_rejected      -> TestQIAggregationConfig::test_non_positive_boundary_length_rejected
- from_dict_requires_all_keys                -> TestQIAggregationConfig::test_from_dict_requires_all_keys
- from_dict_lowercases_terms                 -> TestQIAggregationConfig::test_from_dict_lowercases_terms

_GUIDANCE_VETO_RE (veto regex):
- how_many_not_vetoed                        -> TestGuidanceVetoRegex::test_how_many_not_vetoed
- how_much_not_vetoed                        -> TestGuidanceVetoRegex::test_how_much_not_vetoed
- how_do_still_vetoed                        -> TestGuidanceVetoRegex::test_how_do_still_vetoed
- should_i_still_vetoed                      -> TestGuidanceVetoRegex::test_should_i_still_vetoed
"""
import pytest

from semantic_search.config.models import QIAggregationConfig
from semantic_search.core.exceptions import ConfigurationError
from semantic_search.qi.aggregation_intent_gate import AggregationIntentGate
from semantic_search.qi.engine import _GUIDANCE_VETO_RE

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_NOUNS = ["listing", "auction", "bid", "domain", "tld", "price"]
_STRONG = ["count", "average", "distribution", "how many", "correlat"]
_WEAK = ["top", "most", "trend", "vs"]
_COMPANIONS = ["by ", "per ", "volume", "current bid"]
_WB_MAX_LENGTH = 5


def _gate(enabled: bool = True) -> AggregationIntentGate:
    cfg = QIAggregationConfig(
        enabled=enabled,
        marketplace_nouns=_NOUNS,
        strong_operators=_STRONG,
        weak_operators=_WEAK,
        weak_operator_companions=_COMPANIONS,
        word_boundary_max_length=_WB_MAX_LENGTH,
    )
    return AggregationIntentGate(cfg)


# ---------------------------------------------------------------------------
# AggregationIntentGate.is_analytics
# ---------------------------------------------------------------------------

class TestAggregationIntentGate:
    def test_noun_plus_strong_operator_fires(self) -> None:
        gate = _gate()
        assert gate.is_analytics("average current bid across active auctions") is True

    def test_noun_plus_weak_operator_with_companion_fires(self) -> None:
        gate = _gate()
        assert gate.is_analytics("top tlds by listing volume") is True

    def test_weak_operator_without_companion_returns_false(self) -> None:
        gate = _gate()
        # 'most viewed domains' is an explore popularity browse, not analytics.
        assert gate.is_analytics("most viewed domains") is False

    def test_operator_without_noun_returns_false(self) -> None:
        gate = _gate()
        assert gate.is_analytics("how many letters are there") is False

    def test_noun_without_operator_returns_false(self) -> None:
        gate = _gate()
        assert gate.is_analytics("ai domains under 1500 with traffic") is False

    def test_disabled_gate_always_returns_false(self) -> None:
        gate = _gate(enabled=False)
        assert gate.is_analytics("average current bid across active auctions") is False

    def test_empty_query_returns_false(self) -> None:
        gate = _gate()
        assert gate.is_analytics("") is False

    def test_case_insensitive_match(self) -> None:
        gate = _gate()
        assert gate.is_analytics("AVERAGE CURRENT BID ACROSS ACTIVE AUCTIONS") is True

    def test_short_operator_word_boundary_no_substring(self) -> None:
        gate = _gate()
        # 'count' must not match inside 'accountant'; this query has no real operator.
        assert gate.is_analytics("best domain for lawyer accountant advisor") is False

    def test_plural_noun_substring_match(self) -> None:
        gate = _gate()
        # 'listing' noun matches the plural 'listings' via substring.
        assert gate.is_analytics("how many listings added") is True

    def test_how_many_count_query_fires(self) -> None:
        gate = _gate()
        assert gate.is_analytics("how many auctions are ending this week") is True

    def test_correlation_query_fires(self) -> None:
        gate = _gate()
        assert gate.is_analytics("does traffic correlate with current bid price") is True

    # AggregationIntentGate.explain
    def test_explain_returns_operator_and_noun(self) -> None:
        gate = _gate()
        fired, operator, noun = gate.explain("average current bid across active auctions")
        assert fired is True
        assert operator == "average"
        assert noun == "bid"

    def test_explain_weak_tier_returns_weak_operator(self) -> None:
        gate = _gate()
        fired, operator, noun = gate.explain("top tlds by listing volume")
        assert fired is True
        assert operator == "top"
        assert noun in _NOUNS

    def test_explain_returns_false_on_no_noun(self) -> None:
        gate = _gate()
        fired, operator, noun = gate.explain("how many letters are there")
        assert fired is False
        assert operator is None
        assert noun is None

    def test_explain_disabled_returns_false(self) -> None:
        gate = _gate(enabled=False)
        fired, operator, noun = gate.explain("average bid across auctions")
        assert fired is False
        assert operator is None
        assert noun is None


# ---------------------------------------------------------------------------
# QIAggregationConfig validation
# ---------------------------------------------------------------------------

class TestQIAggregationConfig:
    def _kwargs(self, **overrides):
        base = dict(
            enabled=True,
            marketplace_nouns=["auction"],
            strong_operators=["count"],
            weak_operators=["top"],
            weak_operator_companions=["by "],
            word_boundary_max_length=5,
        )
        base.update(overrides)
        return base

    def test_empty_nouns_rejected(self) -> None:
        with pytest.raises(ConfigurationError):
            QIAggregationConfig(**self._kwargs(marketplace_nouns=[]))

    def test_empty_strong_operators_rejected(self) -> None:
        with pytest.raises(ConfigurationError):
            QIAggregationConfig(**self._kwargs(strong_operators=[]))

    def test_non_positive_boundary_length_rejected(self) -> None:
        with pytest.raises(ConfigurationError):
            QIAggregationConfig(**self._kwargs(word_boundary_max_length=0))

    def test_from_dict_requires_all_keys(self) -> None:
        with pytest.raises(ConfigurationError):
            QIAggregationConfig.from_dict({"enabled": True, "marketplace_nouns": ["auction"]})

    def test_from_dict_lowercases_terms(self) -> None:
        cfg = QIAggregationConfig.from_dict({
            "enabled": True,
            "marketplace_nouns": ["AUCTION"],
            "strong_operators": ["COUNT"],
            "weak_operators": ["TOP"],
            "weak_operator_companions": ["BY "],
            "word_boundary_max_length": 5,
        })
        assert "auction" in cfg.marketplace_nouns
        assert "count" in cfg.strong_operators
        assert "top" in cfg.weak_operators
        assert "by " in cfg.weak_operator_companions

    def test_valid_config_constructs(self) -> None:
        cfg = QIAggregationConfig(**self._kwargs())
        assert cfg.enabled is True
        assert "count" in cfg.strong_operators

    def test_null_config_raises(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            AggregationIntentGate(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _GUIDANCE_VETO_RE — veto regex verification
# ---------------------------------------------------------------------------

class TestGuidanceVetoRegex:
    def test_how_many_not_vetoed(self) -> None:
        assert not _GUIDANCE_VETO_RE.search("how many .net domains sold last week")

    def test_how_much_not_vetoed(self) -> None:
        assert not _GUIDANCE_VETO_RE.search("how much did .com domains sell for last month")

    def test_how_do_still_vetoed(self) -> None:
        assert _GUIDANCE_VETO_RE.search("how do I buy domains on GoDaddy")

    def test_should_i_still_vetoed(self) -> None:
        assert _GUIDANCE_VETO_RE.search("should I bid on this .io domain")

    def test_explain_still_vetoed(self) -> None:
        assert _GUIDANCE_VETO_RE.search("explain the difference between closeout and expiry")

    def test_approach_still_vetoed(self) -> None:
        assert _GUIDANCE_VETO_RE.search("best approach for buying .com domains")

    def test_how_often_still_vetoed(self) -> None:
        assert _GUIDANCE_VETO_RE.search("how often do .io domains expire")

    def test_how_long_still_vetoed(self) -> None:
        assert _GUIDANCE_VETO_RE.search("how long does a GoDaddy auction last")
