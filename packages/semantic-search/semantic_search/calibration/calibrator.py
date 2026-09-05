"""Temperature-scaling calibrator + per-tier registry.

A calibrator is a 1-parameter monotonic transform on classifier confidence:

    p_calibrated = sigmoid(logit(p_raw) / T),  T > 0

T = 1.0 is the identity (no calibration). T > 1 softens overconfident scores
(pulls toward 0.5); T < 1 sharpens underconfident scores. The single parameter
is stable under low-N fits, which is what the golden seed dataset gives us
(~30 cases per archetype). Larger calibrators (Platt with 2 params,
isotonic) need more data than we have at boot.

Monotonicity (T > 0 ⇒ sigmoid(logit(p)/T) is strictly increasing in p) means
the relative ordering of slices is preserved — the cascade's "highest
confidence wins" semantics survive calibration unchanged.

Both ``TemperatureCalibrator`` and ``CalibratorRegistry`` are deliberately
synchronous and side-effect-free: they're called inside the QI hot path so
allocation and CPU cost matter; nothing here logs at INFO per call.
"""
import math
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple

from semantic_search.core.exceptions import ValidationError
from semantic_search.core.logging_utils import get_logger
from semantic_search.core.validation import safe_float

if TYPE_CHECKING:
    from semantic_search.calibration.probe import ProbeRegistry
    from semantic_search.contracts import ConfidenceSignals

logger = get_logger(__name__)


# Avoid log/exp overflow in `logit` and `sigmoid` for probabilities pinned at
# the extremes. The clip is symmetric so identity (T=1) is preserved exactly
# in the open interval (eps, 1-eps) — it only differs at p ∈ {0, 1}.
_PROB_EPS = 1e-6
_DEFAULT_PROBE_WEIGHT_RAW = 1.0


def _clip_prob(p: float) -> float:
    """Clip into (eps, 1-eps) to keep logit/sigmoid finite."""
    if p < _PROB_EPS:
        return _PROB_EPS
    if p > 1.0 - _PROB_EPS:
        return 1.0 - _PROB_EPS
    return p


def _logit(p: float) -> float:
    """Logit: ln(p / (1-p)), bijection (0,1) → ℝ."""
    pp = _clip_prob(p)
    return math.log(pp / (1.0 - pp))


def _sigmoid(z: float) -> float:
    """Numerically stable sigmoid (avoids overflow)."""
    if z >= 0.0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def temperature_scale(p_raw: float, temperature: float) -> float:
    """Apply temperature scaling (T>0: 1.0=identity, >1=soften, <1=sharpen)."""
    if temperature <= 0.0:
        raise ValidationError(f"temperature must be > 0, got {temperature}")
    return _sigmoid(_logit(p_raw) / float(temperature))


@dataclass(frozen=True)
class CalibrationFit:
    """Result of fitting one tier's temperature calibrator.

    Carried in the registry so observability (logs / future health endpoint)
    can report exactly *what* tuned when calibration is enabled. Frozen so
    a fit cannot be mutated mid-flight; hot-swap replaces the whole record.

    :param tier: str - Tier identifier (matches `CalibrationConfig.tier_keys`)
    :param temperature: float - Fitted T (1.0 when below `min_samples_per_tier`)
    :param n_samples: int - Number of (raw_conf, correct) pairs used in the fit
    :param pre_nll: float - Mean negative log-likelihood at T=1 (identity)
    :param post_nll: float - Mean NLL at the fitted T
    :param accuracy: float - Fraction correct in the fit set (sanity signal)
    :param ece_pre: float - Expected calibration error before fitting (10 bins)
    :param ece_post: float - Expected calibration error after fitting (10 bins)
    :param fitted: bool - False when too few samples / disabled (T=1.0 enforced)
    :param source: str - 'golden_seeds' | 'traffic_hot_swap' | 'identity'
    """
    tier: str
    temperature: float
    n_samples: int
    pre_nll: float
    post_nll: float
    accuracy: float
    ece_pre: float
    ece_post: float
    fitted: bool
    source: str

    def __post_init__(self) -> None:
        if not isinstance(self.tier, str) or not self.tier:
            raise ValidationError("CalibrationFit.tier must be non-empty string")
        if self.temperature <= 0:
            raise ValidationError(f"CalibrationFit.temperature must be > 0, got {self.temperature}")
        if self.n_samples < 0:
            raise ValidationError(f"CalibrationFit.n_samples must be >= 0, got {self.n_samples}")
        if self.source not in {'golden_seeds', 'traffic_hot_swap', 'identity'}:
            raise ValidationError(f"CalibrationFit.source unknown: {self.source!r}")

    @staticmethod
    def identity(tier: str) -> 'CalibrationFit':
        """Return an identity fit (T=1.0) for a tier — no calibration applied.

        Used (a) when calibration is disabled in config, (b) when a tier has
        fewer than ``min_samples_per_tier`` golden-seed cases, or (c) at
        registry boot before any fit runs.
        """
        return CalibrationFit(tier=tier, temperature=1.0, n_samples=0, pre_nll=0.0, post_nll=0.0, accuracy=0.0, ece_pre=0.0, ece_post=0.0, fitted=False, source='identity')


def _mean_nll(samples: Iterable[Tuple[float, int]], temperature: float) -> float:
    """Mean negative log-likelihood over (raw_conf, correct) pairs at T.

    NLL is the canonical loss for probabilistic calibration: minimising NLL
    forces the predicted probability to match the empirical correctness rate
    in each confidence bin. Ties are broken by the lower-T solution (sharper).
    """
    total = 0.0
    n = 0
    for p_raw, correct in samples:
        n += 1
        p = temperature_scale(safe_float(p_raw, 0.5), temperature)
        if correct:
            total += -math.log(_clip_prob(p))
        else:
            total += -math.log(_clip_prob(1.0 - p))
    if n == 0:
        return 0.0
    return total / float(n)


def expected_calibration_error(samples: Iterable[Tuple[float, int]], temperature: float, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) over equal-width bins.

    ECE = Σ_bin (|bin| / N) * |confidence_in_bin - accuracy_in_bin|.

    Standard definition (Guo et al. 2017). 10 bins is conventional and what
    the plan's §3 calibration TCE metric is benchmarked against ("TCE < 0.05").

    :param samples: Iterable[(raw_conf, correct)] - Same shape as fit input
    :param temperature: float - T to apply before binning (1.0 = pre-cal ECE)
    :param n_bins: int - Number of equal-width [0, 1] bins
    :return: float - ECE in [0, 1] (lower = tighter calibration)
    """
    if n_bins < 2:
        raise ValidationError(f"expected_calibration_error.n_bins must be >= 2, got {n_bins}")
    bin_conf = [0.0] * n_bins
    bin_acc = [0.0] * n_bins
    bin_count = [0] * n_bins
    total = 0
    for p_raw, correct in samples:
        p = temperature_scale(safe_float(p_raw, 0.5), temperature)
        idx = min(int(p * n_bins), n_bins - 1)
        bin_conf[idx] += p
        bin_acc[idx] += 1 if correct else 0
        bin_count[idx] += 1
        total += 1
    if total == 0:
        return 0.0
    ece = 0.0
    for i in range(n_bins):
        if bin_count[i] == 0:
            continue
        avg_conf = bin_conf[i] / float(bin_count[i])
        avg_acc = bin_acc[i] / float(bin_count[i])
        ece += (bin_count[i] / float(total)) * abs(avg_conf - avg_acc)
    return ece


def combine_calibrated(p_temp_scaled: float, p_correct: float, weight_raw: float) -> float:
    """Weighted geometric mean of T-scaled raw probability and probe output.
    :param p_temp_scaled: float - Temperature-scaled raw probability in (eps, 1-eps)
    :param p_correct: float - Probe-predicted P(correct) in (eps, 1-eps)
    :param weight_raw: float - Weight on the temperature-scaled leg in [0, 1]
    :return: float - Combined calibrated confidence in (eps, 1-eps)
    :raises ValidationError: when weight_raw is outside [0, 1]
    """
    if not 0.0 <= float(weight_raw) <= 1.0:
        raise ValidationError(f"combine_calibrated.weight_raw must be in [0,1], got {weight_raw}")
    p1 = _clip_prob(safe_float(p_temp_scaled, 0.5))
    p2 = _clip_prob(safe_float(p_correct, 0.5))
    w = float(weight_raw)
    log_combined = w * math.log(p1) + (1.0 - w) * math.log(p2)
    return math.exp(log_combined)


class TemperatureCalibrator:
    """A single-tier temperature-scaling calibrator (immutable per fit).

    Construct via ``TemperatureCalibrator(tier, fit)`` or use the
    registry's `register_fit` to install one. The transform is pure: given
    raw confidence ``p_raw`` it returns ``sigmoid(logit(p_raw) / T)``.

    :param tier: str - Stable tier identifier
    :param fit: CalibrationFit - Fit record (T + provenance metadata)
    """

    def __init__(self, tier: str, fit: CalibrationFit):
        if tier != fit.tier:
            raise ValidationError(f"tier mismatch: calibrator tier={tier!r} fit.tier={fit.tier!r}")
        self._tier = tier
        self._fit = fit

    @property
    def tier(self) -> str:
        return self._tier

    @property
    def fit(self) -> CalibrationFit:
        return self._fit

    @property
    def temperature(self) -> float:
        return self._fit.temperature

    def calibrate(self, raw_confidence: float) -> float:
        """Map raw classifier confidence to calibrated confidence.
        :param raw_confidence: float - Raw probability in [0, 1]
        :return: float - Calibrated probability in (eps, 1-eps); identity at T=1
        """
        return temperature_scale(safe_float(raw_confidence, 0.5), self._fit.temperature)


class CalibratorRegistry:
    """Per-tier registry of `TemperatureCalibrator`s + optional ``ProbeRegistry``.

    The QI engine looks up a calibrator by ``decision_tier`` after a slice is
    classified. Tiers without a registered fit fall back to the identity
    calibrator (T=1.0) so absence is never a hard failure — disabling is
    always graceful.

    Calibrated confidence combines temperature scaling with a correctness probe
    (entropy-driven). When a
    ``ProbeRegistry`` and ``CalibrationProbeConfig`` are attached via
    ``attach_probe_registry``, ``calibrate_combined(tier, signals)`` returns
    the geometric-mean combination of the T-scaled raw probability and the
    probe's P(correct | distribution_shape). Otherwise the combined path
    collapses to pure temperature scaling — i.e. legacy behaviour.

    :param tier_keys: List[str] - Stable tier identifiers from config
    :param hot_swap_min_samples: int - Floor before a hot-swap fit is accepted
        (calibrator tightens with traffic data)
    """

    def __init__(self, tier_keys: List[str], hot_swap_min_samples: int, startup_log_detail: bool):
        if not tier_keys:
            raise ValidationError("CalibratorRegistry requires non-empty tier_keys")
        if hot_swap_min_samples < 1:
            raise ValidationError("CalibratorRegistry.hot_swap_min_samples must be >= 1")
        self._tier_keys = list(tier_keys)
        self._hot_swap_min = int(hot_swap_min_samples)
        self._startup_log_detail = bool(startup_log_detail)
        self._lock = threading.Lock()
        # Bootstrap with identity calibrators for every declared tier so the
        # `calibrate(tier, p)` hot path is never None-checked.
        self._calibrators: Dict[str, TemperatureCalibrator] = {
            t: TemperatureCalibrator(t, CalibrationFit.identity(t)) for t in self._tier_keys
        }
        # Probe registry + weight are attached lazily (post-construction) so
        # the temperature-only legacy path keeps working when the probe
        # sub-system is disabled in config. None ⇒ combined ≡ T-scaled.
        self._probe_registry: Optional['ProbeRegistry'] = None
        self._probe_weight_raw: float = _DEFAULT_PROBE_WEIGHT_RAW

    def known_tiers(self) -> List[str]:
        return list(self._tier_keys)

    def register_fit(self, fit: CalibrationFit) -> None:
        """Install a fit for a tier. Replaces any existing calibrator atomically.

        Thread-safe so the boot-time fit driver and the future hot-swap path
        (consuming feedback-store labels) cannot race on the same tier.
        """
        if fit.tier not in self._tier_keys:
            raise ValidationError(f"CalibratorRegistry.register_fit unknown tier {fit.tier!r}; expected one of {self._tier_keys}")
        if fit.source == 'traffic_hot_swap' and fit.n_samples < self._hot_swap_min:
            raise ValidationError(f"CalibratorRegistry.register_fit hot-swap requires >= {self._hot_swap_min} samples, got {fit.n_samples}")
        with self._lock:
            self._calibrators[fit.tier] = TemperatureCalibrator(fit.tier, fit)
        msg = f"calibration_fit_registered tier={fit.tier} temperature={fit.temperature:.3f} n={fit.n_samples} pre_nll={fit.pre_nll:.4f} post_nll={fit.post_nll:.4f} ece_pre={fit.ece_pre:.4f} ece_post={fit.ece_post:.4f} source={fit.source}"
        if self._startup_log_detail or fit.source != 'identity':
            logger.info(msg)
        else:
            logger.debug(msg)

    def get(self, tier: str) -> TemperatureCalibrator:
        """Look up a calibrator by tier; returns identity for unknown tiers.

        Unknown tiers are treated as identity rather than raising so a new
        decision_tier introduced by a future cascade change does not crash
        the hot path before its config is updated.
        """
        with self._lock:
            cal = self._calibrators.get(tier)
        if cal is not None:
            return cal
        # Synthesize an identity calibrator on demand for unknown tiers; do not
        # store it (the registry is the source of truth for declared tiers).
        return TemperatureCalibrator(tier, CalibrationFit.identity(tier))

    def calibrate(self, tier: str, raw_confidence: float) -> float:
        """Hot-path shortcut — map raw confidence by the tier's calibrator.

        Pure temperature scaling. Use ``calibrate_combined`` to get the
        plan-mandated entropy + probe combination when distribution data is
        available.
        """
        return self.get(tier).calibrate(raw_confidence)

    def attach_probe_registry(self, probe_registry: 'ProbeRegistry', weight_raw: float) -> None:
        """Attach a probe registry + combination weight (combined-calibration).

        After this call ``calibrate_combined`` returns the weighted
        geometric mean of the T-scaled raw probability and the probe's
        P(correct). Before this call ``calibrate_combined`` is identical
        to ``calibrate`` — pure temperature scaling.

        :param probe_registry: ProbeRegistry - Per-tier correctness probes
        :param weight_raw: float - Weight on the T-scaled leg in [0, 1]
        :raises ValidationError: when weight_raw is outside [0, 1]
        """
        if not 0.0 <= float(weight_raw) <= 1.0:
            raise ValidationError(f"CalibratorRegistry.attach_probe_registry weight_raw must be in [0,1], got {weight_raw}")
        with self._lock:
            self._probe_registry = probe_registry
            self._probe_weight_raw = float(weight_raw)

    def has_probe(self) -> bool:
        """True iff a probe registry has been attached."""
        with self._lock:
            return self._probe_registry is not None

    @property
    def probe_weight_raw(self) -> float:
        """Current weight on the temperature-scaled leg (default 1.0 = no probe)."""
        with self._lock:
            return self._probe_weight_raw

    def calibrate_combined(self, tier: str, signals: 'ConfidenceSignals') -> float:
        """Map ConfidenceSignals → combined calibrated confidence.

        When a probe registry is attached, returns the weighted geometric
        mean of the T-scaled raw probability and the probe's P(correct |
        H_norm). When no probe is attached, returns the pure T-scaled raw
        probability — preserving today's behaviour exactly.

        :param tier: str - decision_tier identifier (matches tier_keys)
        :param signals: ConfidenceSignals - Raw conf + entropy + margin
        :return: float - Combined calibrated confidence in (eps, 1-eps)
        """
        p_temp = self.get(tier).calibrate(signals.raw_confidence)
        with self._lock:
            probe_registry = self._probe_registry
            weight_raw = self._probe_weight_raw
        if probe_registry is None:
            return p_temp
        p_correct = probe_registry.predict(tier, signals.entropy_normalized)
        return combine_calibrated(p_temp, p_correct, weight_raw)

    def all_fits(self) -> Dict[str, CalibrationFit]:
        """Snapshot every registered fit (defensive copy)."""
        with self._lock:
            return {tier: cal.fit for tier, cal in self._calibrators.items()}
