"""Correctness probe — second leg of calibrated confidence.

Temperature scaling (`calibrator.py`) corrects systematic over/under-confidence
of a classifier's raw top-1 probability. It does NOT correct for the *shape*
of the underlying score distribution: a classifier emitting confidence=0.7
with a near-uniform distribution across 6 classes is much less reliable than
one emitting 0.7 with a sharp peak and the next-best class at 0.05.

The correctness probe captures that distributional signal. Per the plan:

    "Calibrated confidence combines entropy with a correctness probe trained
     on the golden dataset; raw LLM scores are never used directly for
     routing." — plan_agentic_search.md §1, line 299

Design — minimum viable, low-N stable:

* Single feature by default: ``H_norm = entropy / ln(N)`` ∈ [0, 1] where N is
  the number of classes. ``score_margin = top1 - top2`` is available as an
  optional second feature (config-gated) for when golden seed counts grow.
* Functional form: ``p_correct = sigmoid(alpha * (1 - H_norm) + beta)``.
  ``alpha`` is a config-driven slope (default 4.0) — fixing it leaves a
  single intercept parameter ``beta`` to fit per tier. One free parameter on
  ~30 cases is statistically defensible; two would overfit.
* Combination operator: weighted geometric mean of T-scaled raw confidence
  and probe-predicted correctness probability. Geometric mean preserves
  monotonicity in raw confidence (cascade ordering survives) and degrades
  gracefully: a flat distribution drives combined → 0 regardless of raw, a
  perfectly peaked distribution lets combined ≡ T-scaled raw.

Both the probe transform and the fit are pure-stdlib, deterministic, and
allocation-cheap so the QI hot path stays sub-millisecond per call.
"""
import math
import threading
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from semantic_search.calibration.calibrator import _clip_prob, _sigmoid, combine_calibrated
from semantic_search.core.exceptions import ValidationError
from semantic_search.core.logging_utils import get_logger
from semantic_search.core.validation import safe_float

logger = get_logger(__name__)

# Reciprocal golden ratio (shared search semantics with fit_temperature)
_INV_PHI = (math.sqrt(5.0) - 1.0) / 2.0
_IDENTITY_PROBE_BETA = 30.0


def compute_normalized_entropy(scores: Sequence[float]) -> float:
    """Normalised Shannon entropy [0,1] (softmax-normalized; 0=peaked, 1=uniform)."""
    n = 0
    for _ in scores:
        n += 1
    if n < 2:
        return 0.0
    score_list = [safe_float(s, 0.0) for s in scores]
    max_s = max(score_list)
    exps = [math.exp(s - max_s) for s in score_list]
    z = sum(exps)
    if z <= 0.0:
        return 1.0
    probs = [e / z for e in exps]
    h = 0.0
    for p in probs:
        if p > 0.0:
            h -= p * math.log(p)
    h_max = math.log(float(n))
    if h_max <= 0.0:
        return 0.0
    h_norm = h / h_max
    if h_norm < 0.0:
        return 0.0
    if h_norm > 1.0:
        return 1.0
    return h_norm


@dataclass(frozen=True)
class ProbeFit:
    """Result of fitting one tier's correctness probe.

    :param tier: str - Tier identifier (matches CalibrationConfig.tier_keys)
    :param alpha: float - Fixed slope from config (entropy responsiveness)
    :param beta: float - Fitted intercept that minimises mean NLL of correctness
    :param n_samples: int - Number of (entropy_normalized, correct) pairs used
    :param pre_nll: float - Mean NLL of constant-0.5 baseline (sanity floor)
    :param post_nll: float - Mean NLL at fitted (alpha, beta)
    :param accuracy: float - Empirical accuracy in fit set (sanity signal)
    :param fitted: bool - False when below min_samples / disabled (identity probe)
    :param source: str - 'golden_seeds' | 'traffic_hot_swap' | 'identity'
    """
    tier: str
    alpha: float
    beta: float
    n_samples: int
    pre_nll: float
    post_nll: float
    accuracy: float
    fitted: bool
    source: str

    def __post_init__(self) -> None:
        if not isinstance(self.tier, str) or not self.tier:
            raise ValidationError("ProbeFit.tier must be non-empty string")
        if self.alpha <= 0.0:
            raise ValidationError(f"ProbeFit.alpha must be > 0, got {self.alpha}")
        if self.n_samples < 0:
            raise ValidationError(f"ProbeFit.n_samples must be >= 0, got {self.n_samples}")
        if self.source not in {'golden_seeds', 'traffic_hot_swap', 'identity'}:
            raise ValidationError(f"ProbeFit.source unknown: {self.source!r}")

    @staticmethod
    def identity(tier: str, alpha: float = 4.0) -> 'ProbeFit':
        """Identity probe: alpha+beta tuned so probe(*) = 1.0 for all entropy.

        Concretely: with beta = +30 the sigmoid floor is ~1.0 in [0,1] entropy
        space regardless of alpha. Used (a) when calibration disabled, (b) when
        a tier has fewer than ``min_samples`` correctness pairs, or (c) at
        registry boot before any fit. The combined-calibration operator then
        collapses to pure temperature scaling — preserving today's behaviour.
        """
        return ProbeFit(
            tier=tier,
            alpha=float(alpha),
            beta=_IDENTITY_PROBE_BETA,
            n_samples=0,
            pre_nll=0.0,
            post_nll=0.0,
            accuracy=0.0,
            fitted=False,
            source='identity',
        )


def probe_correctness(entropy_normalized: float, alpha: float, beta: float) -> float:
    """Predict P(correct | distribution) from normalized entropy via the probe.

    Functional form: ``sigmoid(alpha * (1 - H_norm) + beta)``. With alpha > 0:
    - H_norm = 0 (peaked) → output = sigmoid(alpha + beta) (highest)
    - H_norm = 1 (uniform) → output = sigmoid(beta) (lowest)
    - Strictly monotone-decreasing in H_norm.

    :param entropy_normalized: float - H / ln(N) in [0, 1]
    :param alpha: float - Slope (responsiveness to entropy); must be > 0
    :param beta: float - Intercept (fit per tier)
    :return: float - Predicted P(correct) in (eps, 1-eps)
    :raises ValidationError: when alpha <= 0
    """
    if alpha <= 0.0:
        raise ValidationError(f"probe_correctness.alpha must be > 0, got {alpha}")
    h = safe_float(entropy_normalized, 1.0)
    if h < 0.0:
        h = 0.0
    elif h > 1.0:
        h = 1.0
    return _sigmoid(float(alpha) * (1.0 - h) + float(beta))


def _probe_mean_nll(samples: Iterable[Tuple[float, int]], alpha: float, beta: float) -> float:
    """Mean negative log-likelihood of (H_norm, correct) pairs at (alpha, beta)."""
    total = 0.0
    n = 0
    for h_norm, correct in samples:
        n += 1
        p = probe_correctness(safe_float(h_norm, 1.0), alpha, beta)
        if correct:
            total += -math.log(_clip_prob(p))
        else:
            total += -math.log(_clip_prob(1.0 - p))
    if n == 0:
        return 0.0
    return total / float(n)


def fit_correctness_probe(samples: Sequence[Tuple[float, int]], alpha: float, min_beta: float, max_beta: float, tolerance: float, max_iterations: int) -> float:
    """Find beta in [min_beta, max_beta] that minimises mean NLL on `samples`.

    Mirrors the temperature fit's golden-section search exactly so the two
    fits share search semantics. Degenerate inputs (no samples, all-correct,
    all-wrong) return ``beta=0.0`` so the identity-bias is preserved.

    :param samples: Sequence[(entropy_normalized, correct)] - Pairs to fit on
    :param alpha: float - Fixed slope (must be > 0)
    :param min_beta: float - Lower bound for beta search
    :param max_beta: float - Upper bound for beta search (must be > min_beta)
    :param tolerance: float - Convergence tolerance on beta
    :param max_iterations: int - Hard cap on bracket-shrink steps
    :return: float - Fitted beta in [min_beta, max_beta]
    :raises ValidationError: when bounds inconsistent or alpha <= 0
    """
    if alpha <= 0.0:
        raise ValidationError(f"fit_correctness_probe.alpha must be > 0, got {alpha}")
    if max_beta <= min_beta:
        raise ValidationError(f"fit_correctness_probe.max_beta must be > min_beta, got {max_beta} <= {min_beta}")
    if tolerance <= 0.0:
        raise ValidationError(f"fit_correctness_probe.tolerance must be > 0, got {tolerance}")
    if max_iterations < 5:
        raise ValidationError(f"fit_correctness_probe.max_iterations must be >= 5, got {max_iterations}")
    sample_list = list(samples)
    if not sample_list:
        return 0.0
    correct_count = sum(1 for _, c in sample_list if c)
    if correct_count == 0 or correct_count == len(sample_list):
        return 0.0

    a = float(min_beta)
    b = float(max_beta)

    def f(beta: float) -> float:
        return _probe_mean_nll(sample_list, alpha, beta)

    c = b - _INV_PHI * (b - a)
    d = a + _INV_PHI * (b - a)
    fc = f(c)
    fd = f(d)
    iterations = 0
    while (b - a) > tolerance and iterations < max_iterations:
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
    beta_opt = (a + b) / 2.0
    if beta_opt < min_beta:
        beta_opt = min_beta
    if beta_opt > max_beta:
        beta_opt = max_beta
    return beta_opt


def _probe_accuracy(samples: Sequence[Tuple[float, int]]) -> float:
    """Empirical accuracy — sanity signal beside NLL."""
    if not samples:
        return 0.0
    return sum(1 for _, c in samples if c) / float(len(samples))


def fit_probe_for_tier(
    tier: str,
    samples: Sequence[Tuple[float, int]],
    alpha: float,
    min_beta: float,
    max_beta: float,
    tolerance: float,
    max_iterations: int,
    min_samples: int,
    source: str,
) -> ProbeFit:
    """Fit the probe for one tier and return a ``ProbeFit`` record.

    Tier samples below ``min_samples`` collapse to the identity probe so the
    combined-calibration operator stays in identity-passthrough mode for
    that tier — never silently fits on too-small N.

    :param tier: str - Tier identifier
    :param samples: Sequence[(entropy_normalized, correct)] - Fit pairs
    :param alpha: float - Fixed slope
    :param min_beta: float - Lower bound for beta search
    :param max_beta: float - Upper bound for beta search
    :param tolerance: float - Convergence tolerance on beta
    :param max_iterations: int - Hard cap on bracket-shrink steps
    :param min_samples: int - Minimum sample floor before fitting
    :param source: str - 'golden_seeds' | 'traffic_hot_swap'
    :return: ProbeFit - Fit record
    """
    if source not in {'golden_seeds', 'traffic_hot_swap'}:
        raise ValidationError(f"fit_probe_for_tier.source must be golden_seeds or traffic_hot_swap, got {source!r}")
    sample_list = list(samples)
    if len(sample_list) < min_samples:
        return ProbeFit.identity(tier, alpha=alpha)
    pre_nll = _probe_mean_nll(sample_list, alpha, 0.0)
    beta = fit_correctness_probe(sample_list, alpha, min_beta, max_beta, tolerance, max_iterations)
    post_nll = _probe_mean_nll(sample_list, alpha, beta)
    return ProbeFit(
        tier=tier,
        alpha=float(alpha),
        beta=float(beta),
        n_samples=len(sample_list),
        pre_nll=pre_nll,
        post_nll=post_nll,
        accuracy=_probe_accuracy(sample_list),
        fitted=True,
        source=source,
    )


class CorrectnessProbe:
    """Per-tier correctness probe (immutable per fit) with thread-safe lookup.

    Mirrors ``TemperatureCalibrator`` in ergonomics: construct via
    ``CorrectnessProbe(tier, fit)`` or install via the registry. The
    transform is pure: given normalized entropy ``H_norm`` it returns
    ``sigmoid(alpha * (1 - H_norm) + beta)``.

    :param tier: str - Stable tier identifier
    :param fit: ProbeFit - Fit record (alpha + beta + provenance)
    """

    def __init__(self, tier: str, fit: ProbeFit):
        if tier != fit.tier:
            raise ValidationError(f"tier mismatch: probe tier={tier!r} fit.tier={fit.tier!r}")
        self._tier = tier
        self._fit = fit

    @property
    def tier(self) -> str:
        return self._tier

    @property
    def fit(self) -> ProbeFit:
        return self._fit

    def predict(self, entropy_normalized: float) -> float:
        """Map normalized entropy to predicted P(correct).

        :param entropy_normalized: float - H / ln(N) in [0, 1]
        :return: float - Predicted P(correct) in (eps, 1-eps)
        """
        return probe_correctness(entropy_normalized, self._fit.alpha, self._fit.beta)


class ProbeRegistry:
    """Per-tier registry of ``CorrectnessProbe``s with thread-safe hot-swap.

    Bootstrapped with identity probes for every declared tier so the
    combined-calibration hot path is never None-checked. Mirrors
    ``CalibratorRegistry``'s atomic-replace semantics; the lock guarantees
    that the boot fit and the future hot-swap path cannot race.

    :param tier_keys: List[str] - Stable tier identifiers from config
    :param alpha: float - Fixed slope baked into identity probes
    :param hot_swap_min_samples: int - Floor before a hot-swap fit is accepted
    """

    def __init__(self, tier_keys: List[str], alpha: float, hot_swap_min_samples: int, startup_log_detail: bool):
        if not tier_keys:
            raise ValidationError("ProbeRegistry requires non-empty tier_keys")
        if alpha <= 0.0:
            raise ValidationError(f"ProbeRegistry.alpha must be > 0, got {alpha}")
        if hot_swap_min_samples < 1:
            raise ValidationError("ProbeRegistry.hot_swap_min_samples must be >= 1")
        self._tier_keys = list(tier_keys)
        self._alpha = float(alpha)
        self._hot_swap_min = int(hot_swap_min_samples)
        self._startup_log_detail = bool(startup_log_detail)
        self._lock = threading.Lock()
        self._probes: Dict[str, CorrectnessProbe] = {
            t: CorrectnessProbe(t, ProbeFit.identity(t, alpha=alpha)) for t in self._tier_keys
        }

    def known_tiers(self) -> List[str]:
        return list(self._tier_keys)

    @property
    def alpha(self) -> float:
        return self._alpha

    def register_fit(self, fit: ProbeFit) -> None:
        """Install a fit for a tier; replaces any existing probe atomically."""
        if fit.tier not in self._tier_keys:
            raise ValidationError(f"ProbeRegistry.register_fit unknown tier {fit.tier!r}; expected one of {self._tier_keys}")
        if fit.source == 'traffic_hot_swap' and fit.n_samples < self._hot_swap_min:
            raise ValidationError(f"ProbeRegistry.register_fit hot-swap requires >= {self._hot_swap_min} samples, got {fit.n_samples}")
        with self._lock:
            self._probes[fit.tier] = CorrectnessProbe(fit.tier, fit)
        msg = f"probe_fit_registered tier={fit.tier} alpha={fit.alpha:.3f} beta={fit.beta:.3f} n={fit.n_samples} pre_nll={fit.pre_nll:.4f} post_nll={fit.post_nll:.4f} source={fit.source}"
        if self._startup_log_detail or fit.source != 'identity':
            logger.info(msg)
        else:
            logger.debug(msg)

    def get(self, tier: str) -> CorrectnessProbe:
        """Look up a probe by tier; identity for unknown tiers (graceful)."""
        with self._lock:
            probe = self._probes.get(tier)
        if probe is not None:
            return probe
        return CorrectnessProbe(tier, ProbeFit.identity(tier, alpha=self._alpha))

    def predict(self, tier: str, entropy_normalized: float) -> float:
        """Hot-path shortcut — predict P(correct) for a tier and entropy."""
        return self.get(tier).predict(entropy_normalized)

    def all_fits(self) -> Dict[str, ProbeFit]:
        """Snapshot every registered fit (defensive copy)."""
        with self._lock:
            return {tier: probe.fit for tier, probe in self._probes.items()}


