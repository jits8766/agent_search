"""Regression tests for pipeline behaviour fixes.

Coverage matrix (per testing.mdc §7):

MultiIntentConfig.drop_zero_expected_requires_hard_filters:
- field_defaults_true                              -> TestMultiIntentConfig::test_field_defaults_true
- field_overridable_to_false                       -> TestMultiIntentConfig::test_field_overridable_to_false
- from_dict_reads_field                            -> TestMultiIntentConfig::test_from_dict_reads_field
- from_dict_defaults_when_absent                   -> TestMultiIntentConfig::test_from_dict_defaults_when_absent
- validation_rejects_non_bool                      -> TestMultiIntentConfig::test_validation_rejects_non_bool

BrandabilityConfig.trigger_regex field:
- trigger_regex_default_none                       -> TestBrandabilityConfig::test_trigger_regex_default_none
- trigger_regex_set_as_string                      -> TestBrandabilityConfig::test_trigger_regex_set_as_string
- trigger_regex_from_dict_none_when_absent         -> TestBrandabilityConfig::test_trigger_regex_from_dict_none_when_absent
- trigger_regex_from_dict_reads_value              -> TestBrandabilityConfig::test_trigger_regex_from_dict_reads_value

ZeroResultGuardConfig.apply_eranker_on_explore_fallback:
- default_false                                    -> TestZeroResultGuardConfigFix7::test_default_false
- overridable_to_true                              -> TestZeroResultGuardConfigFix7::test_overridable_to_true
- from_dict_reads_field                            -> TestZeroResultGuardConfigFix7::test_from_dict_reads_field
- from_dict_defaults_when_absent                   -> TestZeroResultGuardConfigFix7::test_from_dict_defaults_when_absent
- validation_rejects_non_bool                      -> TestZeroResultGuardConfigFix7::test_validation_rejects_non_bool
"""
import pytest

from semantic_search.config.models import (
    BrandabilityConfig,
    ConfigurationError,
    MultiIntentConfig,
    ZeroResultGuardConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zrg_dict(**overrides) -> dict:
    base = dict(
        enabled=True,
        relax_filters_drop_priority=['price_max'],
        semantic_only_top_k=10,
        explore_fallback_max_per_rail=4,
        widen_filters_enabled=False,
        widen_filters_multipliers=[],
        widen_filters_slots=[],
        rrf_k=60,
        semantic_fallback_top_k=10,
    )
    base.update(overrides)
    return base


def _multi_intent_dict(**overrides) -> dict:
    base = dict(
        enabled=True,
        max_sub_intents=5,
        max_split_candidates=10,
        min_sub_query_chars=3,
        weight_confidence=0.5,
        weight_expected_results=0.3,
        weight_specificity=0.2,
        expected_results_norm_cap=100,
        cross_intent_bonus=1.2,
        drop_zero_expected=True,
        sub_intent_failure_policy='fail_soft',
        drop_zero_expected_skip_unreliable_prescreen=True,
        drop_zero_expected_exempt_query_types=['hybrid'],
        all_slices_dropped_fallback_to_single=True,
        cross_slice_retrieve_propagate_slots=[
            'price_min', 'price_max', 'bids_min', 'bids_max',
            'time_remaining_max', 'days_listed_max',
        ],
        slice_encode_blend_parent_concept=True,
        split_on_l0_keywords=True,
        split_on_l0_keywords_min_terms=2,
    )
    base.update(overrides)
    return base


def _brandability_dict(**overrides) -> dict:
    base = dict(
        enabled=True,
        weight_length=0.4,
        weight_vowel_balance=0.3,
        weight_pronounceability=0.3,
        ideal_vowel_ratio=0.4,
        min_length=4,
        max_length=12,
        max_consonant_run=3,
        digit_penalty=0.2,
        hyphen_penalty=0.2,
        boost_weight=0.15,
        trigger_terms=['brandable', 'catchy', 'memorable', 'creative'],
        pure_sort_residual_kinds=[],
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# MultiIntentConfig.drop_zero_expected_requires_hard_filters
# ---------------------------------------------------------------------------

class TestMultiIntentConfig:
    def test_field_defaults_true(self) -> None:
        cfg = MultiIntentConfig.from_dict(_multi_intent_dict())
        assert cfg.drop_zero_expected_requires_hard_filters is True

    def test_field_overridable_to_false(self) -> None:
        cfg = MultiIntentConfig.from_dict(_multi_intent_dict(drop_zero_expected_requires_hard_filters=False))
        assert cfg.drop_zero_expected_requires_hard_filters is False

    def test_from_dict_reads_field(self) -> None:
        for val in (True, False):
            cfg = MultiIntentConfig.from_dict(_multi_intent_dict(drop_zero_expected_requires_hard_filters=val))
            assert cfg.drop_zero_expected_requires_hard_filters is val

    def test_from_dict_defaults_when_absent(self) -> None:
        d = _multi_intent_dict()
        d.pop('drop_zero_expected_requires_hard_filters', None)
        cfg = MultiIntentConfig.from_dict(d)
        assert cfg.drop_zero_expected_requires_hard_filters is True

    def test_validation_rejects_non_bool(self) -> None:
        with pytest.raises(ConfigurationError):
            MultiIntentConfig(
                enabled=True,
                max_sub_intents=5,
                max_split_candidates=10,
                min_sub_query_chars=3,
                weight_confidence=0.5,
                weight_expected_results=0.3,
                weight_specificity=0.2,
                expected_results_norm_cap=100,
                cross_intent_bonus=1.2,
                drop_zero_expected=True,
                sub_intent_failure_policy='fail_soft',
                drop_zero_expected_requires_hard_filters="yes",  # type: ignore[arg-type]
                split_on_l0_keywords=True,
                split_on_l0_keywords_min_terms=2,
            )


# ---------------------------------------------------------------------------
# BrandabilityConfig.trigger_regex
# ---------------------------------------------------------------------------

class TestBrandabilityConfig:
    def test_trigger_regex_default_none(self) -> None:
        cfg = BrandabilityConfig.from_dict(_brandability_dict())
        assert cfg.trigger_regex is None

    def test_trigger_regex_set_as_string(self) -> None:
        cfg = BrandabilityConfig.from_dict(_brandability_dict(trigger_regex=r"\btrust\b"))
        assert cfg.trigger_regex == r"\btrust\b"

    def test_trigger_regex_from_dict_none_when_absent(self) -> None:
        d = _brandability_dict()
        d.pop('trigger_regex', None)
        cfg = BrandabilityConfig.from_dict(d)
        assert cfg.trigger_regex is None

    def test_trigger_regex_from_dict_reads_value(self) -> None:
        cfg = BrandabilityConfig.from_dict(_brandability_dict(trigger_regex=r"\binvest\b"))
        assert cfg.trigger_regex == r"\binvest\b"


# ---------------------------------------------------------------------------
# ZeroResultGuardConfig.apply_eranker_on_explore_fallback
# ---------------------------------------------------------------------------

class TestZeroResultGuardConfigFix7:
    def test_default_false(self) -> None:
        cfg = ZeroResultGuardConfig.from_dict(_zrg_dict())
        assert cfg.apply_eranker_on_explore_fallback is False

    def test_overridable_to_true(self) -> None:
        cfg = ZeroResultGuardConfig.from_dict(_zrg_dict(apply_eranker_on_explore_fallback=True))
        assert cfg.apply_eranker_on_explore_fallback is True

    def test_from_dict_reads_field(self) -> None:
        for val in (True, False):
            cfg = ZeroResultGuardConfig.from_dict(_zrg_dict(apply_eranker_on_explore_fallback=val))
            assert cfg.apply_eranker_on_explore_fallback is val

    def test_from_dict_defaults_when_absent(self) -> None:
        d = _zrg_dict()
        d.pop('apply_eranker_on_explore_fallback', None)
        cfg = ZeroResultGuardConfig.from_dict(d)
        assert cfg.apply_eranker_on_explore_fallback is False

    def test_validation_rejects_non_bool(self) -> None:
        with pytest.raises(ConfigurationError):
            ZeroResultGuardConfig(
                enabled=True,
                relax_filters_drop_priority=['price_max'],
                semantic_only_top_k=10,
                explore_fallback_max_per_rail=4,
                widen_filters_enabled=False,
                widen_filters_multipliers=[],
                widen_filters_slots=[],
                rrf_k=60,
                semantic_fallback_top_k=10,
                apply_eranker_on_explore_fallback="yes",  # type: ignore[arg-type]
            )
