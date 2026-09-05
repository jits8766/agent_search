"""Tests for VagueQuantifierResolver — vague phrase → concrete Entity injection."""
import pytest

from semantic_search.config.models import VagueQuantifierConfig
from semantic_search.contracts import Entity
from semantic_search.qi.vague_quantifier_resolver import VagueQuantifierResolver


def _cfg(**overrides) -> VagueQuantifierConfig:
    defaults = dict(
        enabled=True,
        decent_traffic_floor=500,
        strong_traffic_floor=2000,
        affordable_price_cap=1000,
        expensive_price_floor=5000,
        premium_govalue_floor=1000,
        expiring_soon_seconds=259200,
        domain_age_mature_years=3,
        short_name_max_chars=8,
        long_name_min_chars=15,
        high_authority_min=30,
        strong_backlinks_tf_min=15,
    )
    defaults.update(overrides)
    return VagueQuantifierConfig(**defaults)


def _entity(name: str, value) -> Entity:
    return Entity(name=name, value=value, confidence=0.9, source="L2_llm", chip_kind="hard")


class TestVagueQuantifierResolver:
    def test_decent_traffic_omitted_without_numeric_cue(self):
        """qualitative_floor_scrub — qualitative traffic alone must not invent floor."""
        resolver = VagueQuantifierResolver(_cfg())
        result = resolver.resolve([], "show me domains with decent traffic under $500")
        assert not any(e.name == "traffic_min" for e in result)

    def test_strong_traffic_omitted_without_numeric_cue(self):
        resolver = VagueQuantifierResolver(_cfg())
        result = resolver.resolve([], "com domains with high traffic")
        assert not any(e.name == "traffic_min" for e in result)

    def test_traffic_injected_with_numeric_cue(self):
        resolver = VagueQuantifierResolver(_cfg())
        result = resolver.resolve([], "domains with decent traffic over 5000 visitors")
        assert any(e.name == "traffic_min" and e.value == 500 for e in result)

    def test_traffic_not_overridden_when_set(self):
        resolver = VagueQuantifierResolver(_cfg())
        existing = [_entity("traffic_min", 10000)]
        result = resolver.resolve(existing, "domains with decent traffic")
        assert not any(e.name == "traffic_min" for e in result)

    def test_affordable_price_omitted_without_numeric_cue(self):
        """L0 rule 4 parity — bare affordable/cheap must not invent price_max."""
        resolver = VagueQuantifierResolver(_cfg())
        result = resolver.resolve([], "find affordable .io domains")
        assert not any(e.name == "price_max" for e in result)

    def test_affordable_price_injected_with_numeric_cue(self):
        resolver = VagueQuantifierResolver(_cfg())
        result = resolver.resolve([], "affordable domains under 2000")
        assert any(e.name == "price_max" and e.value == 1000 for e in result)

    def test_cheapest_price_omitted_without_numeric_cue(self):
        resolver = VagueQuantifierResolver(_cfg())
        result = resolver.resolve([], "cheapest brandable domains")
        assert not any(e.name == "price_max" for e in result)

    def test_expensive_price_injected(self):
        resolver = VagueQuantifierResolver(_cfg())
        result = resolver.resolve([], "expensive high-end .com domains")
        assert any(e.name == "price_min" and e.value == 5000 for e in result)

    def test_price_not_overridden_when_set(self):
        resolver = VagueQuantifierResolver(_cfg())
        existing = [_entity("price_max", 200)]
        result = resolver.resolve(existing, "affordable domains")
        assert not any(e.name == "price_max" for e in result)

    def test_expiring_soon_injected(self):
        resolver = VagueQuantifierResolver(_cfg())
        result = resolver.resolve([], "domains expiring soon")
        assert any(e.name == "time_remaining_max" and e.value == 259200 for e in result)

    def test_ending_soon_injected(self):
        resolver = VagueQuantifierResolver(_cfg())
        result = resolver.resolve([], "short domains ending soon")
        assert any(e.name == "time_remaining_max" for e in result)

    def test_mature_domain_age_omitted_without_years_cue(self):
        """qualitative_floor_scrub — mature/aged without N years must not invent age floor."""
        resolver = VagueQuantifierResolver(_cfg())
        result = resolver.resolve([], "mature established domains under $2000")
        assert not any(e.name == "domain_age_min" for e in result)

    def test_mature_domain_age_injected_with_years_cue(self):
        resolver = VagueQuantifierResolver(_cfg())
        result = resolver.resolve([], "mature domains over 5 years under $2000")
        assert any(e.name == "domain_age_min" and e.value == 3 for e in result)

    def test_short_domain_length_injected(self):
        resolver = VagueQuantifierResolver(_cfg())
        result = resolver.resolve([], "short domain names in .com")
        assert any(e.name == "name_length_max" and e.value == 8 for e in result)

    def test_long_domain_length_injected(self):
        resolver = VagueQuantifierResolver(_cfg())
        result = resolver.resolve([], "long domain names for blogs")
        assert any(e.name == "name_length_min" and e.value == 15 for e in result)

    def test_premium_govalue_injected_with_valuation_cue(self):
        resolver = VagueQuantifierResolver(_cfg())
        result = resolver.resolve([], "premium domain names govalue above 1000 under $5000")
        assert any(e.name == "govalue_min" and e.value == 1000 for e in result)

    def test_premium_govalue_omitted_without_valuation_cue(self):
        resolver = VagueQuantifierResolver(_cfg())
        result = resolver.resolve([], "premium ai domain with real visitors")
        assert not any(e.name == "govalue_min" for e in result)

    def test_disabled_resolver_returns_empty(self):
        resolver = VagueQuantifierResolver(_cfg(enabled=False))
        result = resolver.resolve([], "affordable domains with decent traffic")
        assert result == []

    def test_soft_chip_kind_on_injected_entity(self):
        resolver = VagueQuantifierResolver(_cfg())
        result = resolver.resolve([], "domains expiring soon")
        assert result
        assert all(e.chip_kind == "soft" for e in result)

    def test_confidence_on_injected_entity(self):
        resolver = VagueQuantifierResolver(_cfg())
        result = resolver.resolve([], "domains expiring soon")
        assert result
        assert all(e.confidence == 0.75 for e in result)

    def test_mid_budget_omits_price_max(self):
        resolver = VagueQuantifierResolver(_cfg())
        result = resolver.resolve([], "real estate domain decent traffic mid budget")
        assert not any(e.name == "price_max" for e in result)

    def test_no_duplicate_slot_from_multiple_patterns(self):
        resolver = VagueQuantifierResolver(_cfg())
        result = resolver.resolve([], "domains with decent traffic and high traffic over 10000")
        traffic_entities = [e for e in result if e.name == "traffic_min"]
        assert len(traffic_entities) == 1

    def test_no_injection_on_unrelated_query(self):
        resolver = VagueQuantifierResolver(_cfg())
        result = resolver.resolve([], "brandable tech domains .io under $500")
        assert result == []

    # ------------------------------------------------------------------
    # SEO authority and backlink qualitative → quantitative resolution
    # ------------------------------------------------------------------

    def test_high_authority_omitted_without_numeric_cue(self):
        resolver = VagueQuantifierResolver(_cfg(high_authority_min=30))
        result = resolver.resolve([], "domains with high authority")
        assert not any(e.name == "semrush_authority_min" for e in result)

    def test_strong_authority_omitted_without_numeric_cue(self):
        resolver = VagueQuantifierResolver(_cfg(high_authority_min=30))
        result = resolver.resolve([], "strong authority domain under $2000")
        assert not any(e.name == "semrush_authority_min" for e in result)

    def test_authority_score_omitted_without_numeric_cue(self):
        resolver = VagueQuantifierResolver(_cfg(high_authority_min=40))
        result = resolver.resolve([], "authority score finance domain")
        assert not any(e.name == "semrush_authority_min" for e in result)

    def test_seo_authority_omitted_without_numeric_cue(self):
        resolver = VagueQuantifierResolver(_cfg(high_authority_min=30))
        result = resolver.resolve([], "seo authority domain .com")
        assert not any(e.name == "semrush_authority_min" for e in result)

    def test_da_score_phrase_fires_with_numeric_cue(self):
        resolver = VagueQuantifierResolver(_cfg(high_authority_min=30))
        result = resolver.resolve([], "da score over 30 brandable domain")
        assert any(e.name == "semrush_authority_min" for e in result)

    def test_authority_not_overridden_when_set(self):
        resolver = VagueQuantifierResolver(_cfg(high_authority_min=30))
        existing = [_entity("semrush_authority_min", 60)]
        result = resolver.resolve(existing, "domains with high authority")
        assert not any(e.name == "semrush_authority_min" for e in result)

    def test_backlink_juice_omitted_without_numeric_cue(self):
        resolver = VagueQuantifierResolver(_cfg(strong_backlinks_tf_min=15))
        result = resolver.resolve([], "domain with backlink juice")
        assert not any(e.name == "majestic_tf_min" for e in result)

    def test_strong_backlinks_omitted_without_numeric_cue(self):
        resolver = VagueQuantifierResolver(_cfg(strong_backlinks_tf_min=15))
        result = resolver.resolve([], "strong backlinks tech domain")
        assert not any(e.name == "majestic_tf_min" for e in result)

    def test_strong_backlinks_injected_with_tf_numeric_cue(self):
        resolver = VagueQuantifierResolver(_cfg(strong_backlinks_tf_min=15))
        result = resolver.resolve([], "strong backlinks tf 20 tech domain")
        assert any(e.name == "majestic_tf_min" and e.value == 15 for e in result)

    def test_good_seo_does_not_inject_majestic_tf_min(self):
        """good seo → topic_include (L0), not TF invent (LLMJ / Full parity)."""
        resolver = VagueQuantifierResolver(_cfg(strong_backlinks_tf_min=20))
        result = resolver.resolve([], "finance domain good seo affordable")
        assert not any(e.name == "majestic_tf_min" for e in result)

    def test_tf_skipped_when_topic_include_present(self):
        resolver = VagueQuantifierResolver(_cfg(strong_backlinks_tf_min=15))
        existing = [_entity("topic_include", ["seo"])]
        result = resolver.resolve(existing, "good seo domains with strong backlinks")
        assert not any(e.name == "majestic_tf_min" for e in result)

    def test_link_equity_omitted_without_numeric_cue(self):
        resolver = VagueQuantifierResolver(_cfg(strong_backlinks_tf_min=15))
        result = resolver.resolve([], "link equity brandable .com")
        assert not any(e.name == "majestic_tf_min" for e in result)

    def test_backlinks_profile_omitted_without_numeric_cue(self):
        resolver = VagueQuantifierResolver(_cfg(strong_backlinks_tf_min=15))
        result = resolver.resolve([], "domains with backlinks profile and authority score")
        assert not any(e.name == "majestic_tf_min" for e in result)

    def test_seo_and_authority_omitted_without_numeric_cues(self):
        resolver = VagueQuantifierResolver(_cfg(high_authority_min=30, strong_backlinks_tf_min=15))
        result = resolver.resolve([], "high authority domain with strong backlinks")
        names = {e.name for e in result}
        assert "semrush_authority_min" not in names
        assert "majestic_tf_min" not in names

    def test_majestic_tf_not_overridden_when_set(self):
        resolver = VagueQuantifierResolver(_cfg(strong_backlinks_tf_min=15))
        existing = [_entity("majestic_tf_min", 50)]
        result = resolver.resolve(existing, "domain with backlink juice")
        assert not any(e.name == "majestic_tf_min" for e in result)

    def test_link_profile_skipped_when_backlinks_min_present(self):
        """L0 minMajesticBackLinks already set — do not stack vague TF."""
        resolver = VagueQuantifierResolver(_cfg(strong_backlinks_tf_min=15))
        existing = [_entity("majestic_backlinks_min", 1)]
        result = resolver.resolve(existing, "strong link profile")
        assert not any(e.name == "majestic_tf_min" for e in result)

    def test_seo_threshold_reads_from_config(self):
        resolver = VagueQuantifierResolver(_cfg(high_authority_min=99))
        result = resolver.resolve([], "domains with high authority da 40")
        assert any(e.name == "semrush_authority_min" and e.value == 99 for e in result)

    def test_backlink_threshold_reads_from_config(self):
        resolver = VagueQuantifierResolver(_cfg(strong_backlinks_tf_min=77))
        result = resolver.resolve([], "backlink juice tf 20 cheap domain")
        assert any(e.name == "majestic_tf_min" and e.value == 77 for e in result)

    def test_some_traffic_skipped_when_has_web_traffic_signal(self):
        """L0 soft traffic flag already set — do not stack vague traffic_min:500."""
        resolver = VagueQuantifierResolver(_cfg())
        existing = [_entity("has_web_traffic_signal", True)]
        result = resolver.resolve(existing, "budget 1500 max with some traffic")
        assert not any(e.name == "traffic_min" for e in result)

    def test_cheap_skipped_when_price_below_market(self):
        """Underpriced / below-market ≠ affordable price_max inject."""
        resolver = VagueQuantifierResolver(_cfg())
        existing = [_entity("price_below_market", True)]
        result = resolver.resolve(existing, "cheap domain")
        assert not any(e.name == "price_max" for e in result)
