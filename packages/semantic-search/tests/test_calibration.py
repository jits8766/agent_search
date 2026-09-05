"""Tests for temperature-scaling calibrator.

Coverage matrix (per ``testing.mdc`` §7):

``temperature_scale`` (math primitive):
- identity_returns_input                          -> TestTemperatureScale::test_identity_returns_input
- T_above_one_pulls_toward_half                   -> TestTemperatureScale::test_T_above_one_pulls_toward_half
- T_below_one_pushes_to_extremes                  -> TestTemperatureScale::test_T_below_one_pushes_to_extremes
- monotonic_in_p                                  -> TestTemperatureScale::test_monotonic_in_p
- output_in_unit_interval                         -> TestTemperatureScale::test_output_in_unit_interval

``expected_calibration_error`` (eval metric):
- perfectly_calibrated_returns_zero               -> TestECE::test_perfectly_calibrated_returns_zero
- empty_returns_zero                              -> TestECE::test_empty_returns_zero
- skewed_returns_positive                         -> TestECE::test_skewed_returns_positive

``fit_temperature`` (golden-section search):
- recovers_planted_T                              -> TestFitTemperature::test_recovers_planted_T
- empty_samples_returns_one                       -> TestFitTemperature::test_empty_samples_returns_one
- all_correct_returns_one                         -> TestFitTemperature::test_all_correct_returns_one
- all_wrong_returns_one                           -> TestFitTemperature::test_all_wrong_returns_one
- bounds_respected                                -> TestFitTemperature::test_bounds_respected
- min_t_zero_raises                               -> TestFitTemperature::test_min_t_zero_raises
- max_t_le_min_raises                             -> TestFitTemperature::test_max_t_le_min_raises
- post_fit_nll_le_pre_fit                         -> TestFitTemperature::test_post_fit_nll_le_pre_fit

``fit_tier`` (per-tier driver):
- below_min_samples_returns_identity              -> TestFitTier::test_below_min_samples_returns_identity
- above_min_samples_fits                          -> TestFitTier::test_above_min_samples_fits
- bad_source_raises                               -> TestFitTier::test_bad_source_raises

``CalibratorRegistry`` (storage + per-tier dispatch):
- known_tiers_returns_constructor_arg             -> TestCalibratorRegistry::test_known_tiers_returns_constructor_arg
- calibrate_unknown_tier_passthrough              -> TestCalibratorRegistry::test_calibrate_unknown_tier_passthrough
- register_fit_replaces                           -> TestCalibratorRegistry::test_register_fit_replaces
- register_fit_below_hot_swap_min_raises          -> TestCalibratorRegistry::test_register_fit_below_hot_swap_min_raises
- empty_tier_keys_raises                          -> TestCalibratorRegistry::test_empty_tier_keys_raises

``fit_from_golden_seeds`` (end-to-end driver):
- happy_path_fits_only_present_tiers              -> TestFitFromGoldenSeeds::test_happy_path_fits_only_present_tiers
- producer_returning_none_skipped                 -> TestFitFromGoldenSeeds::test_producer_returning_none_skipped
- producer_raising_does_not_block_other_tiers     -> TestFitFromGoldenSeeds::test_producer_raising_does_not_block_other_tiers
- non_dict_case_skipped                           -> TestFitFromGoldenSeeds::test_non_dict_case_skipped
- missing_input_query_skipped                     -> TestFitFromGoldenSeeds::test_missing_input_query_skipped
- fit_on_load_disabled_no_op                      -> TestFitFromGoldenSeeds::test_fit_on_load_disabled_no_op

``QIEngine`` integration:
- engine_calibration_pass_through_when_registry_none -> TestEngineIntegration::test_engine_calibration_pass_through_when_registry_none
- engine_applies_calibration_to_l0_result         -> TestEngineIntegration::test_engine_applies_calibration_to_l0_result
"""
import math
import random

import pytest

from semantic_search.calibration.calibrator import CalibrationFit, CalibratorRegistry, _logit, _mean_nll, _sigmoid, expected_calibration_error, temperature_scale
from semantic_search.calibration.fit import collect_samples_from_golden_seeds, fit_from_golden_seeds, fit_temperature, fit_tier
from semantic_search.config.models import CalibrationConfig
from semantic_search.contracts import IntentSlice
from semantic_search.core.exceptions import ConfigurationError, ValidationError
from semantic_search.qi.engine import QIEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _calibration_config(min_samples: int = 5, hot_swap_min_samples: int = 5, fit_on_load: bool = True) -> CalibrationConfig:
    # CalibrationConfig.__post_init__ enforces:
    #   min_samples_per_tier >= 5  AND  hot_swap_min_samples >= min_samples_per_tier
    # Helper clamps callers to those floors so test intent (small N to exercise
    # the "below" branch) still works correctly.
    eff_min = max(int(min_samples), 5)
    eff_hot = max(int(hot_swap_min_samples), eff_min)
    return CalibrationConfig(
        enabled=True,
        fit_on_load=fit_on_load,
        min_samples_per_tier=eff_min,
        min_temperature=0.1,
        max_temperature=20.0,
        tolerance=1e-4,
        max_iterations=100,
        tier_keys=['L0_entity', 'L1_semantic', 'L2_llm'],
        hot_swap_min_samples=eff_hot,
    )


def _planted_samples(t_true: float, n: int = 200) -> list:
    """Generate samples whose raw confidence has a known temperature distortion.

    For each sample we draw a true probability ``p_true`` uniform on (0,1),
    compute the *distorted* raw probability via ``temperature_scale`` with
    ``T = 1/t_true`` (the *inverse* of what the calibrator should recover),
    then sample a Bernoulli outcome with probability ``p_true`` (the actual
    correctness rate). Fitting on these samples should recover T close to
    ``t_true``.
    """
    rng = random.Random(42)
    out = []
    for _ in range(n):
        p_true = rng.uniform(0.05, 0.95)
        # Distort: producer reports raw_p = sigmoid(logit(p_true) * t_true).
        # The calibrator should learn T ≈ t_true to undo it.
        raw_p = _sigmoid(_logit(p_true) * t_true)
        correct = 1 if rng.random() < p_true else 0
        out.append((raw_p, correct))
    return out


# ---------------------------------------------------------------------------
# temperature_scale
# ---------------------------------------------------------------------------
class TestTemperatureScale:
    def test_identity_returns_input(self):
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert temperature_scale(p, 1.0) == pytest.approx(p, abs=1e-9)

    def test_T_above_one_pulls_toward_half(self):
        # Higher T flattens — moves probabilities toward 0.5.
        p = 0.9
        assert temperature_scale(p, 5.0) < p
        assert temperature_scale(p, 5.0) > 0.5
        p = 0.1
        assert temperature_scale(p, 5.0) > p
        assert temperature_scale(p, 5.0) < 0.5

    def test_T_below_one_pushes_to_extremes(self):
        # Lower T sharpens — pushes probabilities toward 0/1.
        p = 0.7
        assert temperature_scale(p, 0.5) > p
        p = 0.3
        assert temperature_scale(p, 0.5) < p

    def test_monotonic_in_p(self):
        # Same T, increasing p must produce increasing output.
        prev = -1.0
        for p in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
            cur = temperature_scale(p, 2.0)
            assert cur > prev
            prev = cur

    def test_output_in_unit_interval(self):
        # Outputs in [0, 1] — float saturation can hit 0.0/1.0 exactly at
        # extreme p with sharpening (T<<1), so we assert closed bounds. The
        # internal logit/sigmoid is always finite (`_clip_prob` enforces the
        # eps margin); saturation is a float-precision artefact, not divergence.
        for p in [0.0, 1e-12, 0.001, 0.5, 0.999, 1.0]:
            for t in [0.1, 1.0, 20.0]:
                out = temperature_scale(p, t)
                assert 0.0 <= out <= 1.0


# ---------------------------------------------------------------------------
# expected_calibration_error
# ---------------------------------------------------------------------------
class TestECE:
    def test_perfectly_calibrated_returns_zero(self):
        # Synthetic perfectly calibrated: in each bin, accuracy == predicted.
        samples = []
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for _ in range(100):
                samples.append((p, 1 if (len(samples) % 100) < int(p * 100) else 0))
        # Allow a small slack — bin boundaries don't align perfectly with sample distribution.
        assert expected_calibration_error(samples, temperature=1.0) < 0.05

    def test_empty_returns_zero(self):
        assert expected_calibration_error([], temperature=1.0) == 0.0

    def test_skewed_returns_positive(self):
        # All "I'm 0.9 confident" but actually right 10% of the time.
        samples = [(0.9, 1) if i < 10 else (0.9, 0) for i in range(100)]
        ece = expected_calibration_error(samples, temperature=1.0)
        # Gap is ~0.8 in one bin, that bin holds 100% of mass.
        assert ece > 0.7


# ---------------------------------------------------------------------------
# fit_temperature
# ---------------------------------------------------------------------------
class TestFitTemperature:
    def test_recovers_planted_T(self):
        # Plant T=4.0 distortion, verify fit recovers it within 30%.
        samples = _planted_samples(t_true=4.0, n=400)
        t_fit = fit_temperature(samples, min_t=0.1, max_t=20.0, tolerance=1e-4, max_iterations=100)
        # Fit on Bernoulli-noisy samples won't be exact; ±30% is a reasonable bound.
        assert 2.8 <= t_fit <= 5.2, f"planted T=4.0, recovered T={t_fit:.3f}"

    def test_empty_samples_returns_one(self):
        assert fit_temperature([], 0.1, 20.0, 1e-4, 100) == 1.0

    def test_all_correct_returns_one(self):
        samples = [(0.7, 1), (0.8, 1), (0.9, 1)]
        assert fit_temperature(samples, 0.1, 20.0, 1e-4, 100) == 1.0

    def test_all_wrong_returns_one(self):
        samples = [(0.7, 0), (0.8, 0), (0.9, 0)]
        assert fit_temperature(samples, 0.1, 20.0, 1e-4, 100) == 1.0

    def test_bounds_respected(self):
        # Very high T regime (lots of overconfidence) should NOT exceed max_t.
        samples = _planted_samples(t_true=15.0, n=300)
        t_fit = fit_temperature(samples, min_t=0.1, max_t=10.0, tolerance=1e-4, max_iterations=100)
        assert 0.1 <= t_fit <= 10.0

    def test_min_t_zero_raises(self):
        with pytest.raises(ValidationError, match="min_t must be > 0"):
            fit_temperature([(0.5, 1)], 0.0, 20.0, 1e-4, 100)

    def test_max_t_le_min_raises(self):
        with pytest.raises(ValidationError, match="max_t must be > min_t"):
            fit_temperature([(0.5, 1)], 1.0, 0.5, 1e-4, 100)

    def test_post_fit_nll_le_pre_fit(self):
        # Fitting can never make calibration worse than identity on the fit set.
        samples = _planted_samples(t_true=3.0, n=300)
        pre = _mean_nll(samples, 1.0)
        t_fit = fit_temperature(samples, 0.1, 20.0, 1e-4, 100)
        post = _mean_nll(samples, t_fit)
        assert post <= pre + 1e-9


# ---------------------------------------------------------------------------
# fit_tier
# ---------------------------------------------------------------------------
class TestFitTier:
    def test_below_min_samples_returns_identity(self):
        cfg = _calibration_config(min_samples=10)
        fit = fit_tier('L0_entity', [(0.5, 1), (0.6, 0)], cfg, source='golden_seeds')
        assert fit.fitted is False
        assert fit.temperature == 1.0
        assert fit.source == 'identity'

    def test_above_min_samples_fits(self):
        cfg = _calibration_config(min_samples=5)
        samples = _planted_samples(t_true=3.0, n=50)
        fit = fit_tier('L0_entity', samples, cfg, source='golden_seeds')
        assert fit.fitted is True
        assert fit.temperature != 1.0
        assert fit.n_samples == 50

    def test_bad_source_raises(self):
        cfg = _calibration_config(min_samples=2)
        with pytest.raises(ValidationError, match="source must be golden_seeds or traffic_hot_swap"):
            fit_tier('L0_entity', [(0.5, 1), (0.6, 0)], cfg, source='bogus')


# ---------------------------------------------------------------------------
# CalibratorRegistry
# ---------------------------------------------------------------------------
class TestCalibratorRegistry:
    def test_known_tiers_returns_constructor_arg(self):
        reg = CalibratorRegistry(tier_keys=['L0_entity', 'L1_semantic'], hot_swap_min_samples=5, startup_log_detail=True)
        assert set(reg.known_tiers()) == {'L0_entity', 'L1_semantic'}

    def test_calibrate_unknown_tier_passthrough(self):
        # Unknown tier is identity — never raise on stale tier names.
        reg = CalibratorRegistry(tier_keys=['L0_entity'], hot_swap_min_samples=5, startup_log_detail=True)
        assert reg.calibrate('unknown_tier', 0.7) == 0.7

    def test_register_fit_replaces(self):
        reg = CalibratorRegistry(tier_keys=['L0_entity'], hot_swap_min_samples=5, startup_log_detail=True)
        # Identity by default.
        assert reg.calibrate('L0_entity', 0.7) == pytest.approx(0.7, abs=1e-9)
        # Plant a fit.
        fit = CalibrationFit(
            tier='L0_entity', temperature=4.0, n_samples=20,
            pre_nll=1.0, post_nll=0.5, accuracy=0.5,
            ece_pre=0.3, ece_post=0.05, fitted=True, source='golden_seeds',
        )
        reg.register_fit(fit)
        # Now 0.7 must be pulled toward 0.5.
        cal = reg.calibrate('L0_entity', 0.7)
        assert 0.5 < cal < 0.7

    def test_register_fit_below_hot_swap_min_raises(self):
        # Hot-swap fits with N below the configured floor are rejected loudly
        # via ValidationError. The registry must not
        # absorb a too-thin traffic-derived fit silently.
        reg = CalibratorRegistry(tier_keys=['L0_entity'], hot_swap_min_samples=100, startup_log_detail=True)
        fit = CalibrationFit(
            tier='L0_entity', temperature=5.0, n_samples=10,  # below hot_swap_min=100
            pre_nll=1.0, post_nll=0.5, accuracy=0.5,
            ece_pre=0.3, ece_post=0.05, fitted=True, source='traffic_hot_swap',
        )
        with pytest.raises(ValidationError, match="hot-swap requires"):
            reg.register_fit(fit)
        # Boot-time golden_seeds fits are NOT subject to the hot-swap floor.
        golden_fit = CalibrationFit(
            tier='L0_entity', temperature=5.0, n_samples=10,
            pre_nll=1.0, post_nll=0.5, accuracy=0.5,
            ece_pre=0.3, ece_post=0.05, fitted=True, source='golden_seeds',
        )
        reg.register_fit(golden_fit)  # must not raise

    def test_empty_tier_keys_raises(self):
        with pytest.raises(ValidationError):
            CalibratorRegistry(tier_keys=[], hot_swap_min_samples=5, startup_log_detail=True)


# ---------------------------------------------------------------------------
# fit_from_golden_seeds
# ---------------------------------------------------------------------------
class TestFitFromGoldenSeeds:
    def _registry(self) -> CalibratorRegistry:
        return CalibratorRegistry(tier_keys=['L0_entity', 'L1_semantic'], hot_swap_min_samples=2, startup_log_detail=True)

    def _producer(self, decision_tier: str, qt_for_query):
        def fn(query: str):
            qt = qt_for_query(query)
            if qt is None:
                return None
            return (decision_tier, qt, 0.85)
        return fn

    def test_happy_path_fits_only_present_tiers(self):
        # min_samples=5 (config floor); we feed 6 cases so above-floor branch fires.
        cfg = _calibration_config(min_samples=5)
        reg = self._registry()
        cases = [
            {'input_query': 'q1', 'expected_query_type': 'hybrid'},
            {'input_query': 'q2', 'expected_query_type': 'explore'},
            {'input_query': 'q3', 'expected_query_type': 'hybrid'},
            {'input_query': 'q4', 'expected_query_type': 'hybrid'},
            {'input_query': 'q5', 'expected_query_type': 'hybrid'},
            {'input_query': 'q6', 'expected_query_type': 'explore'},
        ]
        # L0 always emits 'hybrid' → 4/6 correct.
        # L1 always emits 'explore' → 2/6 correct.
        producers = {
            'L0_entity': self._producer('L0_entity', lambda q: 'hybrid'),
            'L1_semantic': self._producer('L1_semantic', lambda q: 'explore'),
        }
        fits = fit_from_golden_seeds(reg, cases, producers, cfg)
        assert fits['L0_entity'].n_samples == 6
        assert fits['L1_semantic'].n_samples == 6
        assert fits['L0_entity'].accuracy == pytest.approx(4.0 / 6.0)
        assert fits['L1_semantic'].accuracy == pytest.approx(2.0 / 6.0)

    def test_producer_returning_none_skipped(self):
        cfg = _calibration_config(min_samples=5)
        reg = self._registry()
        cases = [{'input_query': f'q{i}', 'expected_query_type': 'hybrid'} for i in range(5)]
        producers = {
            'L0_entity': lambda q: None,                                  # never fires
            'L1_semantic': self._producer('L1_semantic', lambda q: 'hybrid'),
        }
        fits = fit_from_golden_seeds(reg, cases, producers, cfg)
        assert fits['L0_entity'].n_samples == 0
        assert fits['L0_entity'].fitted is False
        assert fits['L1_semantic'].n_samples == 5

    def test_producer_raising_does_not_block_other_tiers(self):
        cfg = _calibration_config(min_samples=5)
        reg = self._registry()
        cases = [{'input_query': f'q{i}', 'expected_query_type': 'hybrid'} for i in range(5)]
        def bad_producer(q):
            raise RuntimeError("boom")
        producers = {
            'L0_entity': bad_producer,
            'L1_semantic': self._producer('L1_semantic', lambda q: 'hybrid'),
        }
        fits = fit_from_golden_seeds(reg, cases, producers, cfg)
        # L0 had every sample suppressed → identity. L1 still fits.
        assert fits['L0_entity'].n_samples == 0
        assert fits['L1_semantic'].n_samples == 5

    def test_non_dict_case_skipped(self):
        producers = {'L0_entity': self._producer('L0_entity', lambda q: 'hybrid')}
        samples = collect_samples_from_golden_seeds([None, "string-not-dict", 42, {'input_query': 'q', 'expected_query_type': 'hybrid'}], producers)
        assert len(samples['L0_entity']) == 1

    def test_missing_input_query_skipped(self):
        producers = {'L0_entity': self._producer('L0_entity', lambda q: 'hybrid')}
        samples = collect_samples_from_golden_seeds([{'expected_query_type': 'hybrid'}, {'input_query': 'q', 'expected_query_type': 'hybrid'}], producers)
        assert len(samples['L0_entity']) == 1

    def test_fit_on_load_disabled_no_op(self):
        cfg = _calibration_config(min_samples=5, fit_on_load=False)
        reg = self._registry()
        cases = [{'input_query': 'q1', 'expected_query_type': 'hybrid'}]
        producers = {'L0_entity': self._producer('L0_entity', lambda q: 'hybrid')}
        fits = fit_from_golden_seeds(reg, cases, producers, cfg)
        assert all(not f.fitted for f in fits.values())


# ---------------------------------------------------------------------------
# QIEngine integration — calibrator pass-through and application
# ---------------------------------------------------------------------------
class TestEngineIntegration:
    def test_engine_calibration_pass_through_when_registry_none(self):
        # When no registry is wired, _calibrate is identity (the input list is
        # returned unchanged so no allocation cost is incurred on the QI hot
        # path).
        eng = QIEngine.__new__(QIEngine)
        eng._calibrators = None
        slc = IntentSlice(query_type='hybrid', entities=[], confidence=0.7, raw_text='q')
        out = eng._calibrate([slc], 'L0_entity')
        assert out == [slc]
        assert out[0].confidence == 0.7

    def test_engine_applies_calibration_to_l0_result(self):
        reg = CalibratorRegistry(tier_keys=['L0_entity'], hot_swap_min_samples=5, startup_log_detail=True)
        fit = CalibrationFit(
            tier='L0_entity', temperature=5.0, n_samples=20,
            pre_nll=1.0, post_nll=0.4, accuracy=0.6,
            ece_pre=0.3, ece_post=0.05, fitted=True, source='golden_seeds',
        )
        reg.register_fit(fit)
        eng = QIEngine.__new__(QIEngine)
        eng._calibrators = reg
        slc = IntentSlice(query_type='hybrid', entities=[], confidence=0.95, raw_text='q')
        out = eng._calibrate([slc], 'L0_entity')
        # T=5 must pull 0.95 down toward 0.5 but never below it (sigmoid is
        # monotone in p_raw and 0.95 > 0.5).
        assert out[0].confidence < 0.95
        assert out[0].confidence > 0.5
        assert out[0].query_type == 'hybrid'
        assert out[0].raw_text == 'q'
