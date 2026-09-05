"""Confidence-calibration package.

Public surface:
- ``TemperatureCalibrator`` — single-tier 1-parameter Platt-style sigmoid scaling
  ``p_calibrated = sigmoid(logit(p_raw) / T)``. Identity when not fit.
- ``CalibratorRegistry`` — per-tier registry that the QI engine consults to map
  raw classifier confidence into calibrated confidence before downstream gating.
  Supports combined calibration (temperature × correctness probe)
  via ``calibrate_combined`` once a ``ProbeRegistry`` is attached.
- ``fit_temperature`` / ``fit_from_golden_seeds`` — pure-stdlib fit routines
  (golden-section search on log-T against negative log-likelihood).
- ``CalibrationFit`` — result dataclass returned by every fit (T, n_samples,
  pre-NLL, post-NLL, accuracy, ECE).
- ``CorrectnessProbe`` / ``ProbeRegistry`` / ``ProbeFit`` /
  ``fit_probe_for_tier`` / ``compute_normalized_entropy`` /
  ``combine_calibrated`` — second leg of calibrated confidence; the
  entropy-driven probe that combines (geometric mean) with the
  temperature-scaled raw probability so raw classifier scores are never
  used directly for routing.

The package has zero dependencies on the QI cascade — calibrators are pure
data transforms that consumers wire in. The only inbound coupling is on
``contracts.IntentSlice`` / ``contracts.ConfidenceSignals`` (read-only) and
``config.models.CalibrationConfig``.
"""
from semantic_search.calibration.calibrator import CalibrationFit, CalibratorRegistry, TemperatureCalibrator
from semantic_search.calibration.fit import fit_from_golden_seeds, fit_probes_from_golden_seeds, fit_temperature
from semantic_search.calibration.seed_loader import load_calibration_boot_cases
from semantic_search.calibration.probe import CorrectnessProbe, ProbeFit, ProbeRegistry, combine_calibrated, compute_normalized_entropy, fit_correctness_probe, fit_probe_for_tier, probe_correctness

__all__ = [
    'CalibrationFit',
    'CalibratorRegistry',
    'TemperatureCalibrator',
    'fit_from_golden_seeds',
    'fit_probes_from_golden_seeds',
    'fit_temperature',
    'load_calibration_boot_cases',
    'CorrectnessProbe',
    'ProbeFit',
    'ProbeRegistry',
    'combine_calibrated',
    'compute_normalized_entropy',
    'fit_correctness_probe',
    'fit_probe_for_tier',
    'probe_correctness',
]
