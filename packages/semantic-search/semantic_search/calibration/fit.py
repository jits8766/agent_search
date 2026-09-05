"""Temperature calibrator fit routines.

``fit_temperature`` implements golden-section search on log(T) over the
configured [min_T, max_T] range, minimising mean negative log-likelihood
(NLL) over a sequence of (raw_confidence, correct) pairs. Golden-section is
chosen over Newton-Raphson because:

1. NLL as a function of log(T) is unimodal but not always strictly convex on
   small samples (golden seeds can produce a near-flat plateau when many
   slices fire at the same raw confidence). Bracketing methods cope with
   plateaux gracefully; gradient-based methods stall.
2. No derivatives — pure stdlib, deterministic, dependency-free.
3. Linear convergence with golden ratio (φ ≈ 1.618) gives ≤ 30 iterations to
   reach 1e-4 precision over a 100× range — well under any realistic budget.

``fit_from_golden_seeds`` drives the per-tier fit by replaying the QI cascade
classifiers against the bootstrapped golden seed dataset and labelling each
slice as correct iff its query_type matches the case's expected_query_type.
The driver is synchronous and side-effect-free except for the registry
``register_fit`` call.
"""
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from semantic_search.calibration.calibrator import CalibrationFit, CalibratorRegistry, expected_calibration_error, _mean_nll
from semantic_search.calibration.probe import ProbeFit, ProbeRegistry, fit_probe_for_tier
from semantic_search.config.models import CalibrationConfig, CalibrationProbeConfig
from semantic_search.core.exceptions import ValidationError
from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)

# Reciprocal golden ratio — the bracket-shrinkage factor for golden-section search.
_INV_PHI = (math.sqrt(5.0) - 1.0) / 2.0  # ≈ 0.6180


def fit_temperature(samples: Sequence[Tuple[float, int]], min_t: float, max_t: float, tolerance: float, max_iterations: int) -> float:
    """Find the T in [min_t, max_t] that minimises mean NLL on `samples`.

    Operates in log-T space so the search is scale-invariant: doubling
    `max_t` does not change the geometry of the bracket. The returned T is
    bounded by [min_t, max_t] inclusive; degenerate inputs (no samples,
    all-correct, all-wrong) return T=1.0 (identity) with no error.

    :param samples: Sequence[(raw_conf, correct)] - Pairs to fit on
    :param min_t: float - Lower bound (must be > 0)
    :param max_t: float - Upper bound (must be > min_t)
    :param tolerance: float - Convergence tolerance on T (not log-T)
    :param max_iterations: int - Hard cap on bracket-shrink steps
    :return: float - Fitted T in [min_t, max_t]
    :raises ValidationError: When bounds are inconsistent
    """
    if min_t <= 0.0:
        raise ValidationError(f"fit_temperature.min_t must be > 0, got {min_t}")
    if max_t <= min_t:
        raise ValidationError(f"fit_temperature.max_t must be > min_t, got {max_t} <= {min_t}")
    sample_list = list(samples)
    if not sample_list:
        return 1.0
    # Degenerate: every sample carries the same correctness label → NLL is
    # monotone in T and the optimum sits at one of the bounds. Snap to T=1.0
    # rather than collapse to an extreme that would explode the calibrator.
    correct_count = sum(1 for _, c in sample_list if c)
    if correct_count == 0 or correct_count == len(sample_list):
        return 1.0

    a = math.log(min_t)
    b = math.log(max_t)
    log_tol = math.log(1.0 + tolerance / max(min_t, 1e-9))

    def f(log_t: float) -> float:
        return _mean_nll(sample_list, math.exp(log_t))

    c = b - _INV_PHI * (b - a)
    d = a + _INV_PHI * (b - a)
    fc = f(c)
    fd = f(d)
    iterations = 0
    while (b - a) > log_tol and iterations < max_iterations:
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - _INV_PHI * (b - a)
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + _INV_PHI * (b - a)
            fd = f(d)
        iterations += 1
    log_t_opt = (a + b) / 2.0
    t_opt = math.exp(log_t_opt)
    # Clip back into the explicit [min_t, max_t] range to guard against
    # floating-point drift at the bracket edges.
    if t_opt < min_t:
        t_opt = min_t
    if t_opt > max_t:
        t_opt = max_t
    return t_opt


def _accuracy(samples: Sequence[Tuple[float, int]]) -> float:
    """Fraction of `samples` with correct=1 — sanity signal beside NLL."""
    if not samples:
        return 0.0
    return sum(1 for _, c in samples if c) / float(len(samples))


def fit_tier(tier: str, samples: Sequence[Tuple[float, int]], config: CalibrationConfig, source: str) -> CalibrationFit:
    """Fit the calibrator for one tier and return a `CalibrationFit` record.

    Tier samples below ``config.min_samples_per_tier`` collapse to the
    identity fit so the registry stays consistent — never silently fits on
    too-small N.

    :param tier: str - Tier identifier
    :param samples: Sequence[(raw_conf, correct)] - (raw probability, correctness) pairs
    :param config: CalibrationConfig - Bounds, tolerance, iteration cap
    :param source: str - 'golden_seeds' | 'traffic_hot_swap'
    :return: CalibrationFit - Fit record (T, NLL pre/post, ECE pre/post, accuracy)
    """
    if source not in {'golden_seeds', 'traffic_hot_swap'}:
        raise ValidationError(f"fit_tier.source must be golden_seeds or traffic_hot_swap, got {source!r}")
    sample_list = list(samples)
    if len(sample_list) < config.min_samples_per_tier:
        return CalibrationFit.identity(tier)
    pre_nll = _mean_nll(sample_list, 1.0)
    pre_ece = expected_calibration_error(sample_list, 1.0)
    t = fit_temperature(sample_list, config.min_temperature, config.max_temperature, config.tolerance, config.max_iterations)
    post_nll = _mean_nll(sample_list, t)
    post_ece = expected_calibration_error(sample_list, t)
    return CalibrationFit(
        tier=tier,
        temperature=t,
        n_samples=len(sample_list),
        pre_nll=pre_nll,
        post_nll=post_nll,
        accuracy=_accuracy(sample_list),
        ece_pre=pre_ece,
        ece_post=post_ece,
        fitted=True,
        source=source,
    )


# ---------------------------------------------------------------------------
# Golden seed → (raw_conf, correct) sample assembly
# ---------------------------------------------------------------------------

# Tier producer signature — given a normalized query, return (decision_tier,
# emitted query_type, raw_confidence). Producers may return None to skip
# (tier did not fire on the case). The fit driver records (raw_conf,
# correct=1 if emitted_qt == expected_qt else 0) per non-None producer.
TierProducer = Callable[[str], Optional[Tuple[str, str, float]]]


def _normalize_query(q: str) -> str:
    """Match `qi.engine.normalize_query` (lower + whitespace collapse) without
    importing from the QI hot path. Avoids circular wiring at boot time.
    """
    return ' '.join(q.lower().split()).strip()


def collect_samples_from_golden_seeds(cases: Sequence[Dict[str, Any]], producers: Dict[str, TierProducer]) -> Dict[str, List[Tuple[float, int]]]:
    """Replay each golden case through every tier producer and collect samples.

    A producer returns ``(decision_tier, emitted_qt, raw_conf)`` when the tier
    fired or ``None`` when it short-circuited (e.g. L0 regex on a pure
    semantic query). Cases where the producer returned None are dropped from
    that tier's sample set — they would inject zero signal into the fit.

    :param cases: Sequence[Dict] - Golden seed cases (input_query + expected_query_type)
    :param producers: Dict[tier, TierProducer] - Per-tier replay function
    :return: Dict[tier, List[(raw_conf, correct)]] - Sample bundles per tier
    """
    samples: Dict[str, List[Tuple[float, int]]] = {tier: [] for tier in producers}
    for case in cases:
        if not isinstance(case, dict):
            continue
        raw_query = case.get('input_query')
        expected_qt = case.get('expected_query_type')
        if not isinstance(raw_query, str) or not isinstance(expected_qt, str):
            continue
        norm = _normalize_query(raw_query)
        if not norm:
            continue
        for tier, producer in producers.items():
            try:
                emit = producer(norm)
            except Exception as e:
                # A misbehaving producer must NOT block the rest of the fit.
                # Log + skip; the tier just gets fewer samples.
                logger.warning(f"calibration_producer_failed tier={tier} query={raw_query[:80]!r} error_type={type(e).__name__} error={str(e)}")
                continue
            if emit is None:
                continue
            decision_tier, emitted_qt, raw_conf = emit
            # Producers may return their own decision_tier (e.g. L0_entity).
            # We trust their tier name for log fidelity but always bucket
            # samples under the dict key the caller registered.
            correct = 1 if emitted_qt == expected_qt else 0
            samples[tier].append((float(raw_conf), int(correct)))
            if decision_tier and decision_tier != tier:
                logger.warning(f"calibration_producer_tier_mismatch declared={tier} producer_returned={decision_tier} query={raw_query[:80]!r}")
    return samples


def fit_from_golden_seeds(registry: CalibratorRegistry, cases: Sequence[Dict[str, Any]], producers: Dict[str, TierProducer], config: CalibrationConfig) -> Dict[str, CalibrationFit]:
    """End-to-end driver: collect samples, fit per tier, register every fit.

    :param registry: CalibratorRegistry - Target registry (mutated in place)
    :param cases: Sequence[Dict] - Golden seed cases
    :param producers: Dict[tier, TierProducer] - Per-tier replay function
    :param config: CalibrationConfig - Fit bounds + tolerance
    :return: Dict[tier, CalibrationFit] - Fits installed in the registry
    """
    if not config.fit_on_load:
        return {tier: CalibrationFit.identity(tier) for tier in registry.known_tiers()}
    samples_by_tier = collect_samples_from_golden_seeds(cases, producers)
    fits: Dict[str, CalibrationFit] = {}
    for tier in registry.known_tiers():
        tier_samples = samples_by_tier.get(tier, [])
        fit = fit_tier(tier, tier_samples, config, source='golden_seeds')
        registry.register_fit(fit)
        fits[tier] = fit
    return fits


# ---------------------------------------------------------------------------
# Probe (correctness-probe) fit driver — second leg of calibrated confidence.
# Mirrors the temperature-fit driver: per-tier producers replay the cascade
# and emit (decision_tier, emitted_qt, raw_conf, entropy_normalized). Tiers
# without a distribution (L0 regex, L2 LLM today) skip the probe fit entirely
# — the registry keeps their identity probe and combined ≡ T-scaled.
# ---------------------------------------------------------------------------

# Producer signature for the probe fit: (decision_tier, emitted_qt, raw_conf,
# entropy_normalized). A return of None means the tier did not fire on the
# case (skip). Producers that don't emit a distribution should NOT be
# registered for the probe fit — the driver tolerates absence (keeps
# identity probe) but registering them with a synthetic ``H_norm=1.0`` would
# pollute the fit with neutral signals that bias beta toward 0.
ProbeProducer = Callable[[str], Optional[Tuple[str, str, float, float]]]


def collect_probe_samples_from_golden_seeds(cases: Sequence[Dict[str, Any]], producers: Dict[str, ProbeProducer]) -> Dict[str, List[Tuple[float, int]]]:
    """Replay each golden case through every probe producer; collect (H_norm, correct).

    Identical contract to ``collect_samples_from_golden_seeds`` except the
    sample shape carries normalized entropy (not raw confidence).

    :param cases: Sequence[Dict] - Golden seed cases (input_query + expected_query_type)
    :param producers: Dict[tier, ProbeProducer] - Per-tier replay function
    :return: Dict[tier, List[(entropy_normalized, correct)]] - Sample bundles per tier
    """
    samples: Dict[str, List[Tuple[float, int]]] = {tier: [] for tier in producers}
    for case in cases:
        if not isinstance(case, dict):
            continue
        raw_query = case.get('input_query')
        expected_qt = case.get('expected_query_type')
        if not isinstance(raw_query, str) or not isinstance(expected_qt, str):
            continue
        norm = _normalize_query(raw_query)
        if not norm:
            continue
        for tier, producer in producers.items():
            try:
                emit = producer(norm)
            except Exception as e:
                logger.warning(f"probe_producer_failed tier={tier} query={raw_query[:80]!r} error_type={type(e).__name__} error={str(e)}")
                continue
            if emit is None:
                continue
            decision_tier, emitted_qt, _raw_conf, entropy_normalized = emit
            correct = 1 if emitted_qt == expected_qt else 0
            samples[tier].append((float(entropy_normalized), int(correct)))
            if decision_tier and decision_tier != tier:
                logger.warning(f"probe_producer_tier_mismatch declared={tier} producer_returned={decision_tier} query={raw_query[:80]!r}")
    return samples


def fit_probes_from_golden_seeds(probe_registry: ProbeRegistry, cases: Sequence[Dict[str, Any]], producers: Dict[str, ProbeProducer], probe_config: CalibrationProbeConfig) -> Dict[str, ProbeFit]:
    """End-to-end probe driver — collect (H_norm, correct), fit per tier, register.

    Tiers in ``probe_registry.known_tiers()`` but NOT in ``producers`` keep
    their identity probe (correct: not every cascade tier emits a
    distribution). Tiers whose sample count is below
    ``probe_config.min_samples_per_tier`` also keep identity (handled in
    ``fit_probe_for_tier``).

    :param probe_registry: ProbeRegistry - Target registry (mutated in place)
    :param cases: Sequence[Dict] - Golden seed cases
    :param producers: Dict[tier, ProbeProducer] - Per-tier distribution replay
    :param probe_config: CalibrationProbeConfig - Fit bounds + tolerance + alpha
    :return: Dict[tier, ProbeFit] - Fits installed in the registry
    """
    if not probe_config.enabled:
        return {tier: ProbeFit.identity(tier, alpha=probe_config.alpha) for tier in probe_registry.known_tiers()}
    samples_by_tier = collect_probe_samples_from_golden_seeds(cases, producers)
    fits: Dict[str, ProbeFit] = {}
    for tier in probe_registry.known_tiers():
        tier_samples = samples_by_tier.get(tier, [])
        fit = fit_probe_for_tier(
            tier=tier,
            samples=tier_samples,
            alpha=probe_config.alpha,
            min_beta=probe_config.min_beta,
            max_beta=probe_config.max_beta,
            tolerance=probe_config.tolerance,
            max_iterations=probe_config.max_iterations,
            min_samples=probe_config.min_samples_per_tier,
            source='golden_seeds',
        )
        probe_registry.register_fit(fit)
        fits[tier] = fit
    return fits
