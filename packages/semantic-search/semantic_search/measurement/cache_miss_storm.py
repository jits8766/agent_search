"""Cache miss-storm alarm detector.

Fires when hit rate drops below the configured threshold for a sustained window;
cache continues serving — every miss falls through and re-populates on verifier-pass.

Why a separate detector instead of folding into ``ProxySignalEvaluator``:
the proxy report is computed lifetime-to-date over the ``MeasurementStore``
rolling window (typically 10–30 minutes). The miss-storm alarm needs a
**rolling 5-minute window keyed on cache traffic** (hits + misses), not on
search observations. The two windows differ in size, alignment, and
clearing semantics — a search may take 0 cache reads or 5, so search-rate
gating is the wrong gate for cache-rate alarms.

Design:
- Caller polls ``CacheMissStormDetector.poll(snapshot_hits, snapshot_misses)``
  with the current cumulative hit/miss counters (typically once per scheduler
  tick, e.g. every 30s in the existing measurement scheduler).
- The detector stores deltas vs. the previous poll keyed by timestamp,
  trims anything older than the window, and computes the rolling hit rate.
- Transitions into and out of the breached state emit a ``FeedbackSignal``
  with ``signal_type='cache_miss_storm'`` carrying the observed rate, the
  configured floor, and the window seconds on the payload.
- Edge case: a poll tick with **zero** new cache traffic does NOT clear the
  breach (the detector is dormant, not healed). The breach only clears on
  the first poll where the rolling hit rate >= floor.

This module deliberately does NOT own its scheduling — the registry wires
it into the same loop that ticks the proxy evaluator, which keeps cadence
configuration in one place (``measurement.scheduler``).
"""
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

from semantic_search.config.models import CacheMissStormConfig
from semantic_search.contracts import FeedbackSignal
from semantic_search.core.exceptions import ValidationError
from semantic_search.core.logging_utils import get_logger
from semantic_search.signal_store import SignalStore, schedule_feedback_signal_record

logger = get_logger(__name__)


@dataclass(frozen=True)
class CacheMissStormSample:
    """One detector observation: timestamp + delta hits + delta misses since prior poll.

    Frozen so accidental in-place mutation cannot corrupt the rolling window.
    """
    timestamp: float
    delta_hits: int
    delta_misses: int


@dataclass(frozen=True)
class CacheMissStormStatus:
    """Public snapshot of detector state for dashboards / tests.

    :param window_seconds: float - Rolling window the rate is computed over
    :param hit_rate: Optional[float] - Rolling hit rate; None when sample insufficient
    :param sample_size: int - Cache reads (hits + misses) in the window
    :param breached: bool - True iff hit_rate < floor and sample sufficient
    :param min_sample_size: int - Floor on (hits + misses) before evaluating
    :param hit_rate_min: float - Configured floor below which the breach fires
    """
    window_seconds: float
    hit_rate: Optional[float]
    sample_size: int
    breached: bool
    min_sample_size: int
    hit_rate_min: float


class CacheMissStormDetector:
    """Rolling-window cache hit-rate alarm.

    :param config: CacheMissStormConfig - Window, floor, sample-size guard
    :param signal_store: SignalStore - Sink for emitted ``cache_miss_storm`` signals

    Thread-safety: not thread-safe — caller is the measurement scheduler
    which holds a single tick at a time. Adding a lock would be safe but
    is unnecessary at the current call pattern.
    """

    def __init__(self, config: CacheMissStormConfig, signal_store: SignalStore):
        if not isinstance(config, CacheMissStormConfig):
            raise ValidationError("CacheMissStormDetector requires CacheMissStormConfig")
        if signal_store is None:
            raise ValidationError("CacheMissStormDetector requires a SignalStore")
        self._config = config
        self._signals = signal_store
        self._samples: Deque[CacheMissStormSample] = deque()
        # Anchor for delta computation. None on first poll → first poll
        # records (0, 0) and seeds the anchor. Subsequent polls compute
        # deltas against the anchor.
        self._last_cum_hits: Optional[int] = None
        self._last_cum_misses: Optional[int] = None
        # State for transition-edge signal emission. Starts ``False`` so a
        # cold start does not emit a "recovered" signal.
        self._currently_breached = False

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def poll(self, cumulative_hits: int, cumulative_misses: int) -> CacheMissStormStatus:
        """Record a poll and return the current detector status.

        Emits a ``FeedbackSignal('cache_miss_storm')`` on every state
        transition (closed→breached and breached→closed). The signal is
        emitted exactly once per transition — repeated polls in the same
        breached state do not re-emit.

        :param cumulative_hits: int - Lifetime cache hits across user-facing tiers
        :param cumulative_misses: int - Lifetime cache misses across user-facing tiers
        :return: CacheMissStormStatus - Current snapshot
        """
        if cumulative_hits < 0 or cumulative_misses < 0:
            raise ValidationError(
                f"CacheMissStormDetector.poll requires non-negative counters "
                f"(hits={cumulative_hits}, misses={cumulative_misses})"
            )
        if not self._config.enabled:
            # Detector disabled — return an unbreached status without
            # advancing internal state. Keeps the scheduler tick safe to
            # call when YAML disables the alarm.
            return self._build_status(hit_rate=None, sample_size=0, breached=False)
        now = time.time()
        delta_hits, delta_misses = self._compute_delta(cumulative_hits, cumulative_misses)
        # Always seed/refresh the anchor — the detector tracks deltas, not
        # cumulative values, so the anchor must follow every poll.
        self._last_cum_hits = cumulative_hits
        self._last_cum_misses = cumulative_misses
        if delta_hits > 0 or delta_misses > 0:
            self._samples.append(CacheMissStormSample(now, delta_hits, delta_misses))
        self._trim(now)
        return self._evaluate(now)

    def status(self) -> CacheMissStormStatus:
        """Return the last computed status without recording a poll.

        Useful for read-only dashboards that should not perturb the
        rolling window.
        """
        if not self._config.enabled:
            return self._build_status(hit_rate=None, sample_size=0, breached=False)
        return self._evaluate(time.time())

    def reset(self) -> None:
        """Drop all rolling samples and reset the anchor.

        Called by ops/tests after an intentional warm-cache invalidation
        (e.g. snapshot bump cascade) where the resulting miss spike is
        expected and should NOT alarm.
        """
        self._samples.clear()
        self._last_cum_hits = None
        self._last_cum_misses = None
        # Edge state stays unchanged — if we were breached, we stay
        # breached until a poll with sufficient sample says otherwise.

    def _compute_delta(self, cum_hits: int, cum_misses: int) -> Tuple[int, int]:
        """Return delta vs. the prior poll, defending against counter resets."""
        if self._last_cum_hits is None or self._last_cum_misses is None:
            return 0, 0
        # Defensive: if a process restart resets the counters, treat the
        # delta as zero (we cannot trust the new baseline against the old).
        if cum_hits < self._last_cum_hits or cum_misses < self._last_cum_misses:
            logger.warning(
                f"cache_miss_storm_counter_reset prev_hits={self._last_cum_hits} "
                f"prev_misses={self._last_cum_misses} cur_hits={cum_hits} cur_misses={cum_misses}"
            )
            return 0, 0
        return cum_hits - self._last_cum_hits, cum_misses - self._last_cum_misses

    def _trim(self, now: float) -> None:
        """Drop samples older than the configured window."""
        cutoff = now - self._config.window_seconds
        while self._samples and self._samples[0].timestamp < cutoff:
            self._samples.popleft()

    def _evaluate(self, now: float) -> CacheMissStormStatus:
        """Compute the rolling hit rate, transition state, and emit signals on edges."""
        # Trim once more in case ``status()`` is called between polls.
        self._trim(now)
        hits = sum(s.delta_hits for s in self._samples)
        misses = sum(s.delta_misses for s in self._samples)
        sample_size = hits + misses
        if sample_size < self._config.min_sample_size:
            # Insufficient sample → cannot evaluate. Do NOT clear an
            # existing breach; the breach should stay until we have
            # enough sample to prove recovery.
            return self._build_status(hit_rate=None, sample_size=sample_size, breached=self._currently_breached)
        hit_rate = hits / sample_size
        breached = hit_rate < self._config.hit_rate_min
        self._maybe_emit_transition(was_breached=self._currently_breached, now_breached=breached, hit_rate=hit_rate, sample_size=sample_size)
        self._currently_breached = breached
        return self._build_status(hit_rate=hit_rate, sample_size=sample_size, breached=breached)

    def _maybe_emit_transition(self, was_breached: bool, now_breached: bool, hit_rate: float, sample_size: int) -> None:
        """Emit a ``cache_miss_storm`` signal on the breach edge."""
        if was_breached == now_breached:
            return
        # Only emit on the breach EDGE — recovery is informational and
        # would double the signal volume; the absence of a follow-up
        # breach is the recovery indicator.
        if not now_breached:
            logger.info(
                f"cache_miss_storm_recovered hit_rate={hit_rate:.4f} sample={sample_size}"
            )
            return
        signal = FeedbackSignal(
            signal_id=f"miss_storm_{uuid.uuid4().hex[:12]}",
            request_id=f"miss_storm_{uuid.uuid4().hex[:12]}",
            signal_type='cache_miss_storm',
            signal_origin='cache',
            payload={
                'hit_rate': hit_rate,
                'hit_rate_min': self._config.hit_rate_min,
                'window_seconds': self._config.window_seconds,
                'sample_size': sample_size,
                'tiers': list(self._config.tiers),
            },
        )
        schedule_feedback_signal_record(self._signals, signal)
        logger.warning(
            f"cache_miss_storm_breach hit_rate={hit_rate:.4f} floor={self._config.hit_rate_min} "
            f"window_s={self._config.window_seconds} sample={sample_size}"
        )

    def _build_status(self, hit_rate: Optional[float], sample_size: int, breached: bool) -> CacheMissStormStatus:
        return CacheMissStormStatus(
            window_seconds=self._config.window_seconds,
            hit_rate=hit_rate,
            sample_size=sample_size,
            breached=breached,
            min_sample_size=self._config.min_sample_size,
            hit_rate_min=self._config.hit_rate_min,
        )
