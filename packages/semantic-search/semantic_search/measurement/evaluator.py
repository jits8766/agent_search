"""Computes the proxy-signal report.

Every measured signal maps to one ``ProxySignal`` row. Signals the platform
cannot yet observe (offline metrics, surface affordances not yet shipped)
are emitted with ``status='not_instrumented'`` rather than fabricating a
number - this keeps the dashboard honest.

Inputs are sourced from real subsystem state, never inferred:
- Per-search facts: `MeasurementStore` rolling window
- Cache hit/miss counters: `SearchOrchestrator.cache_stats()`
- Feedback signal counts: `SignalStore.stats()`
- Sanitizer rejection counters: `LayerZeroSanitizer.stats()`
- Circuit breaker activations: `CircuitBreaker.snapshot().total_open_transitions`
- Provider prompt-cache counters: ``LLMCallRouter.prompt_cache_stats()``
  via the ``prompt_cache_stats_fn`` injected callable (None when no LLM tier
  is wired - the signal degrades to ``not_instrumented``).
- Analytics verifier-skip counters: ``AnalyticsRouter.verifier_skip_stats()``
  via the ``verifier_skip_stats_fn`` injected callable (None when the
  analytics router is disabled - the signal degrades to ``not_instrumented``).

Two additional surfaces, both additive and default-disabled:
- ``evaluate_sliced()``: per-intent-bucket variant of every observation-level
  rate signal. The slice key is configured via
  ``MeasurementConfig.slicing.slice_by`` (today: ``query_type``). Each
  per-bucket row is named ``<metric>__<bucket>`` and gated by
  ``slicing.min_per_bucket_sample`` (NOT the global ``min_sample_size`` -
  per-bucket analysis is the whole point of the slicing surface).
- ``search_assisted_conversion_rate``: an instrumented signal computed
  from joining ``SearchObservation.session_id`` against feedback signals
  whose ``signal_type`` is in the configured positive-signal vocabulary.
  Numerator: searches with >=1 follow-up positive signal within
  ``assisted_conversion.attribution_window_seconds``. Denominator: searches
  with a non-empty session_id. Emits a per-bucket variant in
  ``evaluate_sliced()`` so operators can see "did analytics queries stop
  converting?" not just "global conversion dipped".
"""
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from semantic_search.config.models import MeasurementConfig, MeasurementThresholdsConfig
from semantic_search.contracts import ProxySignal, SearchObservation
from semantic_search.core.exceptions import ValidationError
from semantic_search.core.logging_utils import get_logger
from semantic_search.signal_store import SignalStore
from semantic_search.safety.layer_zero_sanitizer import LayerZeroSanitizer
from semantic_search.measurement.store import MeasurementStore
from semantic_search.resilience.circuit_breaker import CircuitBreaker

logger = get_logger(__name__)

_SECONDS_PER_DAY = 86_400.0
_SECONDS_PER_WEEK = 604_800.0

# Sliceable rate signals (excludes global counters: cache, sanitizer, feedback, breaker, LLM)
_SLICEABLE_RATE_SIGNALS = (
    ('zero_result_rate', 'zero_result_rate_max', 'lower_is_better'),
    ('high_confidence_rate', 'high_confidence_rate_min', 'higher_is_better'),
    ('regex_short_circuit_rate', 'regex_short_circuit_rate_min', 'higher_is_better'),
    ('multi_intent_duplicate_rate', 'multi_intent_duplicate_rate_max', 'lower_is_better'),
)

# Latency percentile signals that ``evaluate_sliced`` partitions by
# intent bucket. Each entry is (signal_name, threshold_attr, percentile).
_SLICEABLE_LATENCY_SIGNALS = (
    ('p50_latency_ms', 'p50_latency_ms_max', 0.50),
    ('p99_latency_ms', 'p99_latency_ms_max', 0.99),
)

# Cap on the number of feedback signals scanned per
# ``search_assisted_conversion_rate`` evaluation. The detector pulls this many
# most-recent signals from SignalStore and joins them against the observation
# window. Bounded to keep evaluation O(window_size + cap) instead of O(window
# * total_signals) - without it, a long-running process accumulating signals
# would make every dashboard refresh slower.
_ASSISTED_CONVERSION_SIGNAL_SCAN_CAP = 20_000
# rows the platform cannot yet observe in production. Emitting them as
# `not_instrumented` is intentional - it reminds the dashboard owner that the
# corresponding feature (clarification surface, NL-to-SQL, generative facets,
# etc.) hasn't shipped, instead of letting silence look like "all green".
_NOT_INSTRUMENTED_SIGNALS = (
    ('entity_grounding_adaptation_rate', 'informational'),
    ('query_assistance_ctr', 'higher_is_better'),
    ('qdrant_recall_quality', 'higher_is_better'),
    ('nl_to_sql_validation_failure_rate', 'lower_is_better'),
    ('semantic_router_accuracy_post_retrain', 'higher_is_better'),
    ('hard_filter_chip_lock_rate', 'higher_is_better'),
    ('soft_filter_chip_promotion_rate', 'higher_is_better'),
    ('calibration_error_tce', 'lower_is_better'),
    ('ast_validation_reject_rate', 'lower_is_better'),
    ('generated_facet_ctr', 'higher_is_better'),
    ('inventory_percentile_freshness_lag', 'lower_is_better'),
    ('verifier_gate_reject_rate_analytics', 'lower_is_better'),
    ('llm_token_amplification_ratio', 'informational'),
    ('cost_per_high_confidence_query_baseline_lift', 'informational'),
)


@dataclass
class ProxySignalReport:
    """Bundled output of the evaluator.
    :param signals: List[ProxySignal] - One row per signal (instrumented + not)
    :param window_size: int - Observations currently in the rolling window
    :param min_sample_size: int - Configured sample-size guard
    :param generated_at: float - Unix timestamp at evaluation
    """
    signals: List[ProxySignal]
    window_size: int
    min_sample_size: int
    generated_at: float = field(default_factory=time.time)

    def by_name(self) -> Dict[str, ProxySignal]:
        """Return signals indexed by name (stable for dashboard wiring)."""
        return {s.name: s for s in self.signals}


def _insufficient(
    name: str,
    threshold: Optional[float],
    direction: str,
    sample_size: int,
    details: Dict[str, Any] = None,
) -> ProxySignal:
    """Build an `insufficient_data` ProxySignal row (centralised to avoid duplication)."""
    return ProxySignal(name=name, value=None, threshold=threshold, direction=direction, status='insufficient_data', sample_size=sample_size, details=details if details is not None else {})


def _emit(
    name: str,
    value: float,
    threshold: Optional[float],
    direction: str,
    sample_size: int,
    details: Dict[str, Any] = None,
) -> ProxySignal:
    """Build an instrumented ProxySignal row, deriving status from value vs threshold."""
    if direction == 'lower_is_better' and threshold is not None:
        status = 'ok' if value <= threshold else 'breach'
    elif direction == 'higher_is_better' and threshold is not None:
        status = 'ok' if value >= threshold else 'breach'
    else:
        status = 'ok'
    return ProxySignal(name=name, value=value, threshold=threshold, direction=direction, status=status, sample_size=sample_size, details=details if details is not None else {})


class ProxySignalEvaluator:
    """Computes the proxy-signal table on demand.
    :param config: MeasurementConfig - Window + thresholds bundle
    :param store: MeasurementStore - Rolling window of `SearchObservation`
    :param signal_store: SignalStore - Source of feedback signal counters
    :param sanitizer: LayerZeroSanitizer - Source of sanitizer rejection counters
    :param circuit_breaker: CircuitBreaker - Source of circuit activation counter
    :param cache_stats_fn: Callable[[], Dict[str, Dict[str, int]]] - Returns per-tier hit/miss
    """

    def __init__(
        self,
        config: MeasurementConfig,
        store: MeasurementStore,
        signal_store: SignalStore,
        sanitizer: LayerZeroSanitizer,
        circuit_breaker: CircuitBreaker,
        cache_stats_fn: Callable[[], Dict[str, Dict[str, int]]],
        prompt_cache_stats_fn: Optional[Callable[[], Dict[str, int]]] = None,
        verifier_skip_stats_fn: Optional[Callable[[], Dict[str, int]]] = None,
    ):
        self._config = config
        self._store = store
        self._signals = signal_store
        self._sanitizer = sanitizer
        self._breaker = circuit_breaker
        self._cache_stats_fn = cache_stats_fn
        # `prompt_cache_stats_fn` returns the LLMCallRouter
        # snapshot (cached/total/calls). None means no LLM tier is wired,
        # so the signal degrades to `not_instrumented` automatically.
        self._prompt_cache_stats_fn = prompt_cache_stats_fn
        # `verifier_skip_stats_fn` returns the AnalyticsRouter
        # snapshot (skips/invocations/total). None means the analytics router
        # is disabled, so the signal degrades to `not_instrumented`.
        self._verifier_skip_stats_fn = verifier_skip_stats_fn

    def evaluate(self) -> ProxySignalReport:
        """Build a fresh `ProxySignalReport` from current subsystem state."""
        observations = self._store.snapshot()
        thresholds = self._config.thresholds
        signals: List[ProxySignal] = []
        signals.append(self._signal_zero_result_rate(observations, thresholds))
        signals.append(self._signal_filter_override_rate(observations, thresholds))
        signals.append(self._signal_query_to_click_rate(observations, thresholds))
        signals.append(self._signal_cache_hit_rate(thresholds))
        signals.append(self._signal_high_confidence_rate(observations, thresholds))
        signals.append(self._signal_intent_classification_accuracy())
        signals.append(self._signal_p50_latency(observations, thresholds))
        signals.append(self._signal_p99_latency(observations, thresholds))
        signals.append(self._signal_query_type_distribution(observations))
        signals.append(self._signal_feedback_signals_per_day(thresholds))
        signals.append(self._signal_circuit_breaker_activations(thresholds))
        signals.append(self._signal_regex_short_circuit_rate(observations, thresholds))
        signals.append(self._signal_sanitizer_rejection_rate(thresholds))
        signals.append(self._signal_multi_intent_duplicate_rate(observations, thresholds))
        signals.append(self._signal_explore_to_search_conversion(observations))
        signals.append(self._signal_cost_per_high_confidence_query(observations, thresholds))
        signals.append(self._signal_cost_per_correct_intent_query(observations, thresholds))
        signals.append(self._signal_avg_decision_cost_usd(observations))
        signals.append(self._signal_sla_breach_rate(observations, thresholds))
        # Provider prompt-cache and analytics verifier-skip rate signals are
        # instrumented when their stats functions are wired, otherwise emit
        # `not_instrumented` (with the captured threshold + direction).
        signals.append(self._signal_prompt_cache_hit_rate(thresholds))
        signals.append(self._signal_verifier_skip_rate(thresholds))
        # instrumented only when measurement.assisted_conversion is
        # enabled in YAML; otherwise emits `not_instrumented` so the dashboard
        # row is present and the absence of wiring is visible.
        signals.append(self._signal_search_assisted_conversion_rate(observations))
        for name, direction in _NOT_INSTRUMENTED_SIGNALS:
            signals.append(ProxySignal(name=name, value=None, threshold=None, direction=direction, status='not_instrumented', sample_size=0))
        return ProxySignalReport(signals=signals, window_size=len(observations), min_sample_size=self._store.min_sample_size())

    def evaluate_sliced(self) -> ProxySignalReport:
        """per-intent-bucket variant of every observation-level rate signal.

        Returns a `ProxySignalReport` whose signals are exclusively per-bucket
        rows named ``<metric>__<bucket>`` (e.g. ``zero_result_rate__filter``).
        Each row uses the configured ``min_per_bucket_sample`` floor - NOT the
        global ``min_sample_size`` - because per-bucket analysis is the whole
        point of this surface.

        Raises ConfigurationError-equivalent ValidationError when slicing is
        not configured. Operators that want the unsliced report should call
        :meth:`evaluate`. The per-bucket assisted-conversion rows are included
        only when ``measurement.assisted_conversion`` is also enabled.

        :return: ProxySignalReport - One row per (rate_signal, bucket) pair.
        :raises ValidationError: If `measurement.slicing` is None or disabled.
        """
        if self._config.slicing is None or not self._config.slicing.enabled:
            raise ValidationError("ProxySignalEvaluator.evaluate_sliced requires measurement.slicing.enabled=true")
        slice_by = self._config.slicing.slice_by
        per_bucket_floor = self._config.slicing.min_per_bucket_sample
        observations = self._store.snapshot()
        thresholds = self._config.thresholds
        # Group observations once; every per-bucket signal reuses the partition.
        partition: Dict[str, List[SearchObservation]] = self._partition_observations(observations, slice_by)
        signals: List[ProxySignal] = []
        # Stable bucket ordering for deterministic output (responsible-ai
        # explainability - same inputs => same dashboard rows).
        bucket_keys = sorted(partition.keys())
        for sig_name, threshold_attr, direction in _SLICEABLE_RATE_SIGNALS:
            threshold = getattr(thresholds, threshold_attr)
            for bucket in bucket_keys:
                signals.append(self._sliced_rate_signal(base_name=sig_name, bucket=bucket, bucket_obs=partition[bucket], threshold=threshold, direction=direction, per_bucket_floor=per_bucket_floor))
        # Latency percentiles get their own helper because the compute path
        # differs (sorted percentile vs. counted ratio).
        for sig_name, threshold_attr, percentile in _SLICEABLE_LATENCY_SIGNALS:
            threshold = getattr(thresholds, threshold_attr)
            for bucket in bucket_keys:
                signals.append(self._sliced_latency_signal(
                    base_name=sig_name, bucket=bucket, bucket_obs=partition[bucket],
                    threshold=threshold, percentile=percentile, per_bucket_floor=per_bucket_floor,
                ))
        # Per-bucket query_to_click uses a click-by-request_id join because the
        # signal store does not carry a query_type column directly.
        click_request_ids = self._click_request_ids()
        for bucket in bucket_keys:
            signals.append(self._sliced_query_to_click_signal(
                bucket=bucket, bucket_obs=partition[bucket],
                click_request_ids=click_request_ids, threshold=thresholds.query_to_click_rate_min,
                per_bucket_floor=per_bucket_floor,
            ))
        # Per-bucket assisted-conversion rows - only emitted when the
        # signal is also enabled (matches global behaviour).
        if self._config.assisted_conversion is not None and self._config.assisted_conversion.enabled:
            session_positives = self._collect_session_positives()
            for bucket in bucket_keys:
                signals.append(self._sliced_assisted_conversion_signal(bucket=bucket, bucket_obs=partition[bucket], session_positives=session_positives, per_bucket_floor=per_bucket_floor))
        return ProxySignalReport(signals=signals, window_size=len(observations), min_sample_size=per_bucket_floor)

    def _has_min_sample(self, count: int) -> bool:
        """True when the rolling window has met the configured minimum."""
        return count >= self._store.min_sample_size()

    def _signal_zero_result_rate(self, obs: List[SearchObservation], t: MeasurementThresholdsConfig) -> ProxySignal:
        threshold = t.zero_result_rate_max
        if not self._has_min_sample(len(obs)):
            return _insufficient('zero_result_rate', threshold, 'lower_is_better', len(obs))
        zeros = sum(1 for o in obs if o.result_count == 0)
        rate = zeros / len(obs)
        return _emit('zero_result_rate', rate, threshold, 'lower_is_better', len(obs))

    def _signal_filter_override_rate(self, obs: List[SearchObservation], t: MeasurementThresholdsConfig) -> ProxySignal:
        threshold = t.filter_override_rate_max
        if not self._has_min_sample(len(obs)):
            return _insufficient('filter_override_rate', threshold, 'lower_is_better', len(obs))
        # SignalStore.stats() pre-seeds every allowed signal_type to 0 (see signal_store.py:77),
        # so direct dict access is safe and a missing key would correctly raise.
        overrides = self._signals.stats()['filter_override']
        rate = min(1.0, overrides / len(obs))
        return _emit('filter_override_rate', rate, threshold, 'lower_is_better', len(obs))

    def _signal_query_to_click_rate(self, obs: List[SearchObservation], t: MeasurementThresholdsConfig) -> ProxySignal:
        threshold = t.query_to_click_rate_min
        if not self._has_min_sample(len(obs)):
            return _insufficient('query_to_click_rate', threshold, 'higher_is_better', len(obs))
        clicks = self._signals.stats()['result_click']
        rate = min(1.0, clicks / len(obs))
        return _emit('query_to_click_rate', rate, threshold, 'higher_is_better', len(obs))

    def _signal_cache_hit_rate(self, t: MeasurementThresholdsConfig) -> ProxySignal:
        threshold = t.cache_hit_rate_min
        # SearchOrchestrator.cache_stats() guarantees a {tier: {'hits': int, 'misses': int}}
        # shape for every tier in measurement.thresholds.cache_hit_rate_tiers.
        cache_stats = self._cache_stats_fn()
        tiers = t.cache_hit_rate_tiers
        hits = sum(int(cache_stats[tier]['hits']) for tier in tiers)
        misses = sum(int(cache_stats[tier]['misses']) for tier in tiers)
        total = hits + misses
        if not self._has_min_sample(total):
            return _insufficient('cache_hit_rate', threshold, 'higher_is_better', total)
        rate = hits / total
        return _emit('cache_hit_rate', rate, threshold, 'higher_is_better', total, details={'hits': hits, 'misses': misses})

    def _signal_high_confidence_rate(self, obs: List[SearchObservation], t: MeasurementThresholdsConfig) -> ProxySignal:
        threshold = t.high_confidence_rate_min
        if not self._has_min_sample(len(obs)):
            return _insufficient('high_confidence_rate', threshold, 'higher_is_better', len(obs))
        band = t.high_confidence_band_min
        high = sum(1 for o in obs if o.confidence >= band)
        rate = high / len(obs)
        return _emit('high_confidence_rate', rate, threshold, 'higher_is_better', len(obs), details={'band_min': band})

    def _signal_intent_classification_accuracy(self) -> ProxySignal:
        """Accuracy from offline/UAT ``calibration_label`` feedback signals.

        This is intentionally evaluated from SignalStore on demand, not during
        search, so it cannot add hot-path latency. Payload shape:
        ``{"is_correct": bool, "tier": "...", ...}``.
        """
        labels = self._calibration_label_rows()
        total = len(labels)
        correct_n = sum(1 for row in labels if bool(row['is_correct']))
        by_tier: Dict[str, Dict[str, int]] = {}
        for row in labels:
            tier = str(row['tier'])
            bucket = by_tier.setdefault(tier, {'correct': 0, 'total': 0})
            bucket['total'] += 1
            if bool(row['is_correct']):
                bucket['correct'] += 1
        details = {
            'source_signal': 'calibration_label',
            'correct': correct_n,
            'by_tier': by_tier,
        }
        if not self._has_min_sample(total):
            return _insufficient('intent_classification_accuracy', None, 'higher_is_better', total, details=details)
        return _emit('intent_classification_accuracy', correct_n / total, None, 'higher_is_better', total, details=details)

    def _signal_cost_per_correct_intent_query(self, obs: List[SearchObservation], t: MeasurementThresholdsConfig) -> ProxySignal:
        """Cost/query for labeled-correct high-confidence intent decisions.

        Joins ``calibration_label`` feedback to ``SearchObservation`` by
        request_id. Evaluated only on `/measurement/signals`, not on search.
        """
        observations_by_request = {o.request_id: o for o in obs if o.request_id}
        labels = self._calibration_label_rows()
        matched_labels = 0
        correct_high_conf: List[SearchObservation] = []
        confidence_min = float(t.correct_intent_confidence_min)
        for row in labels:
            observation = observations_by_request.get(str(row['request_id']))
            if observation is None:
                continue
            matched_labels += 1
            if not bool(row['is_correct']):
                continue
            if float(observation.confidence) < confidence_min:
                continue
            correct_high_conf.append(observation)
        details = {
            'source_signal': 'calibration_label',
            'confidence_min': confidence_min,
            'matched_labels': matched_labels,
            'correct_high_confidence': len(correct_high_conf),
        }
        if not self._has_min_sample(len(correct_high_conf)):
            return _insufficient('cost_per_correct_intent_query_usd', None, 'informational', len(correct_high_conf), details=details)
        value = sum(float(o.decision_cost_usd) for o in correct_high_conf) / len(correct_high_conf)
        return _emit('cost_per_correct_intent_query_usd', value, None, 'informational', len(correct_high_conf), details=details)

    def _calibration_label_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for sig in self._signals.recent(_ASSISTED_CONVERSION_SIGNAL_SCAN_CAP):
            if sig.signal_type != 'calibration_label':
                continue
            payload = sig.payload if isinstance(sig.payload, dict) else {}
            raw = payload.get('is_correct')
            if isinstance(raw, bool):
                is_correct = raw
            elif isinstance(raw, (int, float)) and raw in (0, 1):
                is_correct = bool(raw)
            elif isinstance(raw, str) and raw.strip().lower() in ('true', 'false'):
                is_correct = raw.strip().lower() == 'true'
            else:
                continue
            rows.append({
                'request_id': sig.request_id,
                'is_correct': is_correct,
                'tier': str(payload.get('tier') or payload.get('query_type') or 'unknown'),
            })
        return rows

    def _signal_p50_latency(self, obs: List[SearchObservation], t: MeasurementThresholdsConfig) -> ProxySignal:
        return self._latency_percentile_signal(obs, t.p50_latency_ms_max, percentile=0.50, name='p50_latency_ms')

    def _signal_p99_latency(self, obs: List[SearchObservation], t: MeasurementThresholdsConfig) -> ProxySignal:
        return self._latency_percentile_signal(obs, t.p99_latency_ms_max, percentile=0.99, name='p99_latency_ms')

    def _latency_percentile_signal(self, obs: List[SearchObservation], threshold_ms: float, percentile: float, name: str) -> ProxySignal:
        if not self._has_min_sample(len(obs)):
            return _insufficient(name, threshold_ms, 'lower_is_better', len(obs))
        values = sorted(o.total_latency_ms for o in obs)
        # Nearest-rank percentile: index = ceil(p * N) - 1 clamped to [0, N-1].
        # Matches the convention used by Grafana / Datadog so dashboards built
        # off the same observation series produce identical numbers.
        idx = max(0, min(len(values) - 1, int(math.ceil(percentile * len(values))) - 1))
        value = values[idx]
        return _emit(name, value, threshold_ms, 'lower_is_better', len(obs))

    def _signal_query_type_distribution(self, obs: List[SearchObservation]) -> ProxySignal:
        # Informational signal - no threshold. We still gate on min_sample_size so
        # an empty / under-populated window doesn't masquerade as a "tracked" row.
        if not self._has_min_sample(len(obs)):
            return _insufficient('query_type_distribution', None, 'informational', len(obs), details={'counts': {}})
        breakdown: Dict[str, int] = {}
        for o in obs:
            breakdown[o.query_type] = breakdown.get(o.query_type, 0) + 1
        return _emit('query_type_distribution', float(len(obs)), None, 'informational', len(obs), details={'counts': breakdown})

    def _signal_feedback_signals_per_day(self, t: MeasurementThresholdsConfig) -> ProxySignal:
        threshold = float(t.feedback_signals_per_day_min)
        # Sample size = total signals seen across all types. Gating on it prevents
        # noisy per-day extrapolation from tiny windows (oversight.mdc).
        total = sum(int(v) for v in self._signals.stats().values())
        first_obs = self._store.first_observation_time()
        if first_obs is None or not self._has_min_sample(total):
            return _insufficient('feedback_signals_per_day', threshold, 'higher_is_better', total)
        elapsed_seconds = max(1.0, time.time() - first_obs)
        per_day = total * (_SECONDS_PER_DAY / elapsed_seconds)
        return _emit('feedback_signals_per_day', per_day, threshold, 'higher_is_better', total, details={'elapsed_seconds': elapsed_seconds})

    def _signal_circuit_breaker_activations(self, t: MeasurementThresholdsConfig) -> ProxySignal:
        threshold = float(t.circuit_breaker_activations_per_week_max)
        snap = self._breaker.snapshot()
        first_obs = self._store.first_observation_time()
        # Sample size for this signal is the size of the observation window - we use it
        # to suppress extrapolating "per-week" rates from a handful of recent calls.
        window = self._store.size()
        details = {'state': snap.state}
        if first_obs is None or not self._has_min_sample(window):
            return _insufficient('circuit_breaker_activations_per_week', threshold, 'lower_is_better', snap.total_open_transitions, details=details)
        elapsed_seconds = max(1.0, time.time() - first_obs)
        per_week = snap.total_open_transitions * (_SECONDS_PER_WEEK / elapsed_seconds)
        return _emit('circuit_breaker_activations_per_week', per_week, threshold, 'lower_is_better', snap.total_open_transitions, details=details)

    def _signal_regex_short_circuit_rate(self, obs: List[SearchObservation], t: MeasurementThresholdsConfig) -> ProxySignal:
        threshold = t.regex_short_circuit_rate_min
        if not self._has_min_sample(len(obs)):
            return _insufficient('regex_short_circuit_rate', threshold, 'higher_is_better', len(obs))
        l0 = sum(1 for o in obs if o.decision_tier == 'L0_entity')
        rate = l0 / len(obs)
        return _emit('regex_short_circuit_rate', rate, threshold, 'higher_is_better', len(obs))

    def _signal_sanitizer_rejection_rate(self, t: MeasurementThresholdsConfig) -> ProxySignal:
        threshold = t.sanitizer_rejection_rate_max
        # LayerZeroSanitizer.stats() guarantees both keys (see sanitizer.py).
        s = self._sanitizer.stats()
        evaluated = int(s['total_evaluated'])
        rejected = int(s['total_rejected'])
        if evaluated == 0:
            return _insufficient('sanitizer_rejection_rate', threshold, 'lower_is_better', 0)
        rate = rejected / evaluated
        details = {'rejected': rejected, 'evaluated': evaluated}
        return _emit('sanitizer_rejection_rate', rate, threshold, 'lower_is_better', evaluated, details=details)

    def _signal_multi_intent_duplicate_rate(self, obs: List[SearchObservation], t: MeasurementThresholdsConfig) -> ProxySignal:
        threshold = t.multi_intent_duplicate_rate_max
        if not self._has_min_sample(len(obs)):
            return _insufficient('multi_intent_duplicate_rate', threshold, 'lower_is_better', len(obs))
        # Per-search duplicate ratio: 1 - distinct_item_ratio when results exist;
        # observations with zero results contribute nothing (no duplicates possible).
        dup_total = 0.0
        scored = 0
        for o in obs:
            if o.result_count <= 0:
                continue
            dup_total += max(0.0, 1.0 - o.distinct_item_ratio)
            scored += 1
        if scored == 0:
            return _insufficient('multi_intent_duplicate_rate', threshold, 'lower_is_better', 0)
        rate = dup_total / scored
        return _emit('multi_intent_duplicate_rate', rate, threshold, 'lower_is_better', scored)

    def _signal_explore_to_search_conversion(self, obs: List[SearchObservation]) -> ProxySignal:
        # An `explore` observation "converts" when the same session_id issues
        # any non-explore search within `explore_followup_window_seconds`. We
        # apply the same min_sample_size gate as the other rate signals so a
        # single explore doesn't produce a 0% / 100% rate that looks tracked.
        explores = [o for o in obs if o.query_type == 'explore' and o.session_id]
        if not self._has_min_sample(len(explores)):
            return _insufficient('explore_to_search_conversion_rate', None, 'higher_is_better', len(explores))
        window = self._config.explore_followup_window_seconds
        followups: Dict[str, List[float]] = {}
        for o in obs:
            if o.query_type != 'explore' and o.session_id:
                followups.setdefault(o.session_id, []).append(o.created_at)
        converted = 0
        for e in explores:
            ts_list = followups.get(e.session_id)
            if ts_list is None:
                continue
            if any(0 < (ts - e.created_at) <= window for ts in ts_list):
                converted += 1
        rate = converted / len(explores)
        details = {'window_seconds': window, 'converted': converted}
        return _emit('explore_to_search_conversion_rate', rate, None, 'higher_is_better', len(explores), details=details)

    def _signal_cost_per_high_confidence_query(self, obs: List[SearchObservation], t: MeasurementThresholdsConfig) -> ProxySignal:
        band = t.high_confidence_band_min
        relevant = [o for o in obs if o.confidence >= band]
        if not self._has_min_sample(len(relevant)):
            return _insufficient('cost_per_high_confidence_query_usd', None, 'informational', len(relevant), details={'band_min': band})
        avg_cost = sum(float(o.decision_cost_usd) for o in relevant) / len(relevant)
        details = {'band_min': band, 'alert_multiplier': t.cost_per_high_confidence_query_alert_multiplier}
        return _emit('cost_per_high_confidence_query_usd', avg_cost, None, 'informational', len(relevant), details=details)

    def _signal_avg_decision_cost_usd(self, obs: List[SearchObservation]) -> ProxySignal:
        if not self._has_min_sample(len(obs)):
            return _insufficient('avg_decision_cost_usd', None, 'informational', len(obs))
        avg = sum(float(o.decision_cost_usd) for o in obs) / len(obs)
        return _emit('avg_decision_cost_usd', avg, None, 'informational', len(obs))

    def _signal_sla_breach_rate(self, obs: List[SearchObservation], t: MeasurementThresholdsConfig) -> ProxySignal:
        # Wall = p99_latency_ms_max (the configured hard ceiling). Fraction of
        # queries that exceeded it is the SLA breach rate - mirrors eval_global's
        # per-suite SLA breach counter but computed from the live observation window.
        if not self._has_min_sample(len(obs)):
            return _insufficient('sla_breach_rate', None, 'informational', len(obs))
        wall_ms = t.p99_latency_ms_max
        breaches = sum(1 for o in obs if o.total_latency_ms > wall_ms)
        rate = breaches / len(obs)
        return _emit('sla_breach_rate', rate, None, 'informational', len(obs), details={'wall_ms': wall_ms, 'breaches': breaches})

    def _signal_prompt_cache_hit_rate(self, t: MeasurementThresholdsConfig) -> ProxySignal:
        """prompt-cache-hit-rate launch-blocking KPI >= 0.50.

        Numerator: ``cached_input_tokens`` aggregated across all successful
        ``LLMCallRouter.call_structured`` calls. Denominator: total
        ``prompt_tokens`` from the same set. Sample size guard uses the
        denominator (input tokens) so a handful of cheap calls cannot
        masquerade as a ratified rate.
        """
        threshold = t.prompt_cache_hit_rate_min
        if self._prompt_cache_stats_fn is None:
            return ProxySignal(name='prompt_cache_hit_rate', value=None, threshold=threshold, direction='higher_is_better', status='not_instrumented', sample_size=0)
        stats = self._prompt_cache_stats_fn() or {}
        cached = int(stats.get('cached_input_tokens', 0) or 0)
        total = int(stats.get('prompt_input_tokens', 0) or 0)
        calls = int(stats.get('calls', 0) or 0)
        details = {'cached_input_tokens': cached, 'prompt_input_tokens': total, 'calls': calls}
        if total <= 0 or not self._has_min_sample(calls):
            # Gate on call count (not token count) because min_sample_size is
            # calibrated against per-search counts.
            return _insufficient('prompt_cache_hit_rate', threshold, 'higher_is_better', calls, details=details)
        # Cap at 1.0 (defence against provider reporting bugs where cached > total).
        rate = min(1.0, cached / total)
        return _emit('prompt_cache_hit_rate', rate, threshold, 'higher_is_better', calls, details=details)

    # ---------------------------------------------------------------------
    # per-bucket helpers + assisted-conversion signal
    # ---------------------------------------------------------------------

    def _partition_observations(self, obs: List[SearchObservation], slice_by: str) -> Dict[str, List[SearchObservation]]:
        """Partition observations by ``slice_by`` (today only ``query_type``).

        Buckets that have no observations in the window are not represented
        in the returned dict - emitting `insufficient_data` for every
        unobserved bucket would balloon the report and obscure real coverage
        gaps. Buckets that DO appear, even briefly, are represented and
        gated by the per-bucket floor downstream.
        """
        if slice_by != 'query_type':
            # Defensive - config validation already enforces this; unreachable
            # in normal flow but raised explicitly to fail loud if a future
            # contracts.MEASUREMENT_SLICE_KEYS expansion is added without
            # extending this method.
            raise ValidationError(f"_partition_observations: unsupported slice_by={slice_by!r}")
        out: Dict[str, List[SearchObservation]] = {}
        for o in obs:
            out.setdefault(o.query_type, []).append(o)
        return out

    def _sliced_rate_signal(
        self,
        base_name: str,
        bucket: str,
        bucket_obs: List[SearchObservation],
        threshold: Optional[float],
        direction: str,
        per_bucket_floor: int,
    ) -> ProxySignal:
        """Per-bucket variant of zero_result / high_confidence / regex_short_circuit / multi_intent_duplicate."""
        signal_name = f'{base_name}__{bucket}'
        details = {'bucket': bucket, 'bucket_size': len(bucket_obs)}
        if len(bucket_obs) < per_bucket_floor:
            return _insufficient(signal_name, threshold, direction, len(bucket_obs), details=details)
        if base_name == 'zero_result_rate':
            value = sum(1 for o in bucket_obs if o.result_count == 0) / len(bucket_obs)
        elif base_name == 'high_confidence_rate':
            band = self._config.thresholds.high_confidence_band_min
            value = sum(1 for o in bucket_obs if o.confidence >= band) / len(bucket_obs)
            details['band_min'] = band
        elif base_name == 'regex_short_circuit_rate':
            value = sum(1 for o in bucket_obs if o.decision_tier == 'L0_entity') / len(bucket_obs)
        elif base_name == 'multi_intent_duplicate_rate':
            scored = [o for o in bucket_obs if o.result_count > 0]
            if len(scored) == 0:
                # Within-bucket all-zero-result corner case - no duplicates
                # are even definable, so emit insufficient_data with a zero
                # count rather than a misleading 0.0 rate.
                return _insufficient(signal_name, threshold, direction, 0, details=details)
            value = sum(max(0.0, 1.0 - o.distinct_item_ratio) for o in scored) / len(scored)
            details['scored'] = len(scored)
            return _emit(signal_name, value, threshold, direction, len(scored), details=details)
        else:
            # Unreachable - the dispatch above covers every entry in
            # _SLICEABLE_RATE_SIGNALS. Raised explicitly so a future entry
            # added without a matching branch fails loud, not silent.
            raise ValidationError(f"_sliced_rate_signal: unsupported base_name={base_name!r}")
        return _emit(signal_name, value, threshold, direction, len(bucket_obs), details=details)

    def _sliced_latency_signal(
        self,
        base_name: str,
        bucket: str,
        bucket_obs: List[SearchObservation],
        threshold: Optional[float],
        percentile: float,
        per_bucket_floor: int,
    ) -> ProxySignal:
        """Per-bucket nearest-rank percentile (matches the global helper's convention)."""
        signal_name = f'{base_name}__{bucket}'
        details = {'bucket': bucket, 'bucket_size': len(bucket_obs)}
        if len(bucket_obs) < per_bucket_floor:
            return _insufficient(signal_name, threshold, 'lower_is_better', len(bucket_obs), details=details)
        values = sorted(o.total_latency_ms for o in bucket_obs)
        idx = max(0, min(len(values) - 1, int(math.ceil(percentile * len(values))) - 1))
        return _emit(signal_name, values[idx], threshold, 'lower_is_better', len(bucket_obs), details=details)

    def _click_request_ids(self) -> set:
        """Return the set of `request_id` values carried on `result_click` signals.

        Bounded by the SignalStore's in-memory ring (`recent(_..._SCAN_CAP)`).
        Used by per-bucket query_to_click to attribute clicks back to the
        originating SearchObservation's bucket. Falsy `request_id` values are
        skipped (defensive - FeedbackSignal.__post_init__ already rejects
        empty `request_id`, but the ring may contain test-only entries from
        legacy fixtures).
        """
        out: set = set()
        for sig in self._signals.recent(_ASSISTED_CONVERSION_SIGNAL_SCAN_CAP):
            if sig.signal_type == 'result_click' and sig.request_id:
                out.add(sig.request_id)
        return out

    def _sliced_query_to_click_signal(
        self,
        bucket: str,
        bucket_obs: List[SearchObservation],
        click_request_ids: set,
        threshold: Optional[float],
        per_bucket_floor: int,
    ) -> ProxySignal:
        """Per-bucket query_to_click_rate via request_id join.

        Numerator: bucket searches whose request_id appears in the
        click-signal ring. Denominator: bucket search count. The detector
        joins on `request_id` (not `session_id`) because a session may issue
        many searches across multiple buckets and a click attaches to one
        specific search.
        """
        signal_name = f'query_to_click_rate__{bucket}'
        details = {'bucket': bucket, 'bucket_size': len(bucket_obs)}
        if len(bucket_obs) < per_bucket_floor:
            return _insufficient(signal_name, threshold, 'higher_is_better', len(bucket_obs), details=details)
        clicked = sum(1 for o in bucket_obs if o.request_id in click_request_ids)
        rate = min(1.0, clicked / len(bucket_obs))
        details['clicked'] = clicked
        return _emit(signal_name, rate, threshold, 'higher_is_better', len(bucket_obs), details=details)

    def _collect_session_positives(self) -> Dict[str, List[float]]:
        """Per-session sorted timestamps of positive feedback signals.

        Returns ``{session_id: [created_at, ...]}`` for each session that has
        at least one signal of a configured ``positive_signal_types``. The
        timestamps are sorted ascending so the per-observation lookup can
        binary-search the join window. ``payload['session_id']`` is the join
        key (matches the convention used by /chips endpoints - see app.py).
        """
        ac = self._config.assisted_conversion
        if ac is None:
            # Defensive - the only call sites already gate on this; raising
            # would be safe but returning an empty dict keeps the surface
            # symmetrical with the "no positives" path.
            return {}
        allowed_types = set(ac.positive_signal_types)
        per_session: Dict[str, List[float]] = {}
        for sig in self._signals.recent(_ASSISTED_CONVERSION_SIGNAL_SCAN_CAP):
            if sig.signal_type not in allowed_types:
                continue
            session_id = sig.payload.get('session_id') if isinstance(sig.payload, dict) else None
            if not isinstance(session_id, str) or not session_id:
                continue
            per_session.setdefault(session_id, []).append(float(sig.created_at))
        for ts_list in per_session.values():
            ts_list.sort()
        return per_session

    def _converted(self, obs: SearchObservation, session_positives: Dict[str, List[float]], window_seconds: float) -> bool:
        """True when `obs.session_id` has at least one positive signal in
        ``(obs.created_at, obs.created_at + window_seconds]``.

        The window is OPEN on the lower bound and CLOSED on the upper bound,
        matching the convention used by ``_signal_explore_to_search_conversion``
        (`0 < (ts - created_at) <= window`). A signal at or before the search
        cannot be a follow-up; a signal at exactly +window is the last
        moment that counts.
        """
        if not obs.session_id:
            return False
        ts_list = session_positives.get(obs.session_id)
        if not ts_list:
            return False
        lower = obs.created_at
        upper = obs.created_at + window_seconds
        for ts in ts_list:
            # ts_list is sorted asc; the first ts > lower decides - if it's
            # also <= upper, conversion. Otherwise no later ts can be in
            # range either (sort invariant).
            if ts <= lower:
                continue
            return ts <= upper
        return False

    def _signal_search_assisted_conversion_rate(self, observations: List[SearchObservation]) -> ProxySignal:
        """global search_assisted_conversion_rate signal.

        Numerator: searches with non-empty session_id that have at least one
        positive feedback signal arriving within
        ``assisted_conversion.attribution_window_seconds`` AFTER the search.
        Denominator: searches with non-empty session_id (anonymous searches
        cannot be attributed).

        Emits `not_instrumented` when the assisted_conversion sub-block is
        absent or disabled, matching how the prompt-cache and verifier-skip
        signals handle their wiring guards.
        """
        ac = self._config.assisted_conversion
        if ac is None or not ac.enabled:
            return ProxySignal(name='search_assisted_conversion_rate', value=None, threshold=None, direction='higher_is_better', status='not_instrumented', sample_size=0)
        threshold = ac.threshold_min
        attributable = [o for o in observations if o.session_id]
        if not self._has_min_sample(len(attributable)):
            return _insufficient('search_assisted_conversion_rate', threshold, 'higher_is_better', len(attributable), details={'attribution_window_seconds': ac.attribution_window_seconds})
        session_positives = self._collect_session_positives()
        converted = sum(1 for o in attributable if self._converted(o, session_positives, ac.attribution_window_seconds))
        rate = converted / len(attributable)
        return _emit(
            'search_assisted_conversion_rate', rate, threshold, 'higher_is_better',
            len(attributable),
            details={
                'attribution_window_seconds': ac.attribution_window_seconds,
                'converted': converted,
                'positive_signal_types': list(ac.positive_signal_types),
            },
        )

    def _sliced_assisted_conversion_signal(self, bucket: str, bucket_obs: List[SearchObservation], session_positives: Dict[str, List[float]], per_bucket_floor: int) -> ProxySignal:
        """Per-bucket variant of search_assisted_conversion_rate.

        Same numerator/denominator semantics as the global signal, restricted
        to observations in `bucket_obs`. The denominator is bucket searches
        with non-empty session_id (NOT all bucket searches) - anonymous
        searches cannot be attributed in the global signal either.
        """
        ac = self._config.assisted_conversion
        # Only call sites already gate on ac.enabled, but a defensive check
        # keeps the helper safe if it's reused later.
        if ac is None or not ac.enabled:
            raise ValidationError("_sliced_assisted_conversion_signal called without assisted_conversion enabled")
        signal_name = f'search_assisted_conversion_rate__{bucket}'
        threshold = ac.threshold_min
        attributable = [o for o in bucket_obs if o.session_id]
        details = {'bucket': bucket, 'bucket_size': len(bucket_obs), 'attributable_size': len(attributable), 'attribution_window_seconds': ac.attribution_window_seconds}
        if len(attributable) < per_bucket_floor:
            return _insufficient(signal_name, threshold, 'higher_is_better', len(attributable), details=details)
        converted = sum(1 for o in attributable if self._converted(o, session_positives, ac.attribution_window_seconds))
        rate = converted / len(attributable)
        details['converted'] = converted
        return _emit(signal_name, rate, threshold, 'higher_is_better', len(attributable), details=details)

    def _signal_verifier_skip_rate(self, t: MeasurementThresholdsConfig) -> ProxySignal:
        """verifier-skip-rate KPI >= 0.85 once warm.

        Numerator: ``AnalyticsRouter`` fast-path verifier auto-skips
        (``model='skipped_warm_cache'``). Denominator: skips + invocations
        on the fast path. Legacy pipeline verifier calls are excluded -
        this signal specifically tracks the *fast-path* skip rate (the
        warm-cache verifier-skip optimisation). The min-sample-size guard
        uses the denominator so an empty analytics path does not produce a
        0% rate that looks instrumented.
        """
        threshold = t.verifier_skip_rate_min
        if self._verifier_skip_stats_fn is None:
            return ProxySignal(name='verifier_skip_rate', value=None, threshold=threshold, direction='higher_is_better', status='not_instrumented', sample_size=0)
        stats = self._verifier_skip_stats_fn() or {}
        skips = int(stats.get('skips', 0) or 0)
        invocations = int(stats.get('invocations', 0) or 0)
        total = skips + invocations
        details = {'skips': skips, 'invocations': invocations}
        if total <= 0 or not self._has_min_sample(total):
            return _insufficient('verifier_skip_rate', threshold, 'higher_is_better', total, details=details)
        rate = skips / total
        return _emit('verifier_skip_rate', rate, threshold, 'higher_is_better', total, details=details)
