"""Tests for TermDisambiguator — polysemous TLD-stem resolution before L0/L1 routing."""
import pytest

from semantic_search.config.models import TermDisambiguationRule, TermDisambiguatorConfig
from semantic_search.qi.term_disambiguator import TermDisambiguator


def _rule(term, tld_form, tld_ctx=None, topic_ctx=None, default_to_tld=False) -> TermDisambiguationRule:
    return TermDisambiguationRule(
        term=term,
        tld_form=tld_form,
        tld_context_tokens=tld_ctx or [],
        topic_context_tokens=topic_ctx or [],
        default_to_tld=default_to_tld,
    )


def _cfg(rules=None, context_window=3, enabled=True) -> TermDisambiguatorConfig:
    if rules is None:
        rules = [
            _rule("ai", ".ai", tld_ctx=["domains", "names", "cheap"], topic_ctx=["startup", "machine", "learning"], default_to_tld=False),
            _rule("io", ".io", tld_ctx=["startup", "saas", "dev"], topic_ctx=["input", "output"], default_to_tld=True),
            _rule("co", ".co", tld_ctx=["domains", "short"], topic_ctx=["company", "colorado"], default_to_tld=False),
        ]
    return TermDisambiguatorConfig(enabled=enabled, context_window=context_window, rules=rules)


class TestTermDisambiguator:
    def test_tld_context_rewrites_token(self):
        result = TermDisambiguator(_cfg()).disambiguate("cheap ai domains")
        assert ".ai" in result
        assert "ai" not in result.split()

    def test_topic_context_preserves_token(self):
        result = TermDisambiguator(_cfg()).disambiguate("ai startup names")
        assert result == "ai startup names"

    def test_topic_wins_over_tld_context(self):
        # Both tld_ctx ("domains") and topic_ctx ("startup") present → topic wins.
        result = TermDisambiguator(_cfg()).disambiguate("ai startup domains")
        assert result == "ai startup domains"

    def test_default_to_tld_true_rewrites_without_context(self):
        # "io" alone → default_to_tld=True → rewrite.
        result = TermDisambiguator(_cfg()).disambiguate("io")
        assert result == ".io"

    def test_default_to_tld_false_keeps_without_context(self):
        # "ai" alone → default_to_tld=False → keep.
        result = TermDisambiguator(_cfg()).disambiguate("ai")
        assert result == "ai"

    def test_already_dot_notation_skipped(self):
        result = TermDisambiguator(_cfg()).disambiguate(".ai domains")
        assert result == ".ai domains"

    def test_unknown_token_unchanged(self):
        result = TermDisambiguator(_cfg()).disambiguate("blockchain domains")
        assert result == "blockchain domains"

    def test_disabled_returns_original(self):
        result = TermDisambiguator(_cfg(enabled=False)).disambiguate("cheap ai domains")
        assert result == "cheap ai domains"

    def test_empty_rules_returns_original(self):
        result = TermDisambiguator(_cfg(rules=[])).disambiguate("cheap ai domains")
        assert result == "cheap ai domains"

    def test_multiple_terms_in_one_query(self):
        result = TermDisambiguator(_cfg()).disambiguate("ai io startup")
        # "ai" has topic_ctx "startup" in window → kept; "io" has tld_ctx "startup" → rewritten.
        assert "ai" in result.split()
        assert ".io" in result

    def test_io_with_topic_context_preserved(self):
        result = TermDisambiguator(_cfg()).disambiguate("input output io protocol")
        # "input"/"output" are topic ctx → keep "io".
        assert "io" in result.split()
        assert ".io" not in result

    def test_context_window_respected(self):
        # Token far outside window should not influence decision.
        cfg = _cfg(context_window=1)
        # "startup" is 3 positions away → outside window of 1 → no topic ctx → default fires.
        result = TermDisambiguator(cfg).disambiguate("io domains startup names something")
        # "domains" IS within window(1) of "io" → tld_ctx → rewrite expected.
        assert ".io" in result

    def test_original_string_returned_when_no_change(self):
        q = "crypto domain for sale"
        result = TermDisambiguator(_cfg()).disambiguate(q)
        assert result is q or result == q

    def test_rewritten_query_preserves_other_tokens(self):
        result = TermDisambiguator(_cfg()).disambiguate("cheap ai names under 500")
        parts = result.split()
        assert parts[0] == "cheap"
        assert parts[1] == ".ai"
        assert "names" in parts
        assert "under" in parts


class TestTermDisambiguatorConfig:
    def test_from_dict_parses_rule_list(self):
        d = {
            "enabled": True,
            "context_window": 3,
            "rules": [
                {
                    "term": "ai",
                    "tld_form": ".ai",
                    "tld_context_tokens": ["domains"],
                    "topic_context_tokens": ["startup"],
                    "default_to_tld": False,
                }
            ],
        }
        cfg = TermDisambiguatorConfig.from_dict(d)
        assert cfg.enabled is True
        assert cfg.context_window == 3
        assert len(cfg.rules) == 1
        assert cfg.rules[0].term == "ai"
        assert cfg.rules[0].tld_form == ".ai"

    def test_from_dict_missing_required_raises(self):
        from semantic_search.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError):
            TermDisambiguatorConfig.from_dict({"enabled": True})

    def test_rule_tld_form_must_start_with_dot(self):
        from semantic_search.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError):
            TermDisambiguationRule(term="ai", tld_form="ai", tld_context_tokens=[], topic_context_tokens=[], default_to_tld=False)

    def test_config_rejects_duplicate_terms(self):
        from semantic_search.core.exceptions import ConfigurationError
        r1 = _rule("ai", ".ai")
        r2 = _rule("ai", ".ai")
        with pytest.raises(ConfigurationError):
            TermDisambiguatorConfig(enabled=True, context_window=3, rules=[r1, r2])

    def test_config_rejects_zero_context_window(self):
        from semantic_search.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError):
            TermDisambiguatorConfig(enabled=True, context_window=0, rules=[])
