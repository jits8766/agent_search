"""LLM circuit breaker — closed/open/half-open state machine (LLM row).

This is the **LLM-only** breaker. names three breakers (LLM, Qdrant,
ClickHouse) sharing the same three-state contract; only the LLM one lives
here today. The other two are implemented at the dependency boundary they
protect:
  - Qdrant     → `BackendHealthRegistry` (`backend='vector'` / `'structured'`)
                 + per-call exception handling in `qdrant_adapter.py`.
  - ClickHouse → `BackendHealthRegistry` (`backend='clickhouse'`) wired in
                 `analytics/clickhouse_client.py`.

States (LLM breaker — also the contract every breaker honours):
- CLOSED   : normal traffic; failures accumulate in a rolling window.
- OPEN     : tier disabled; calls short-circuit to `CircuitOpenError` and the
             caller dispatches the / §7 fallback (Tier 2 routing for LLM).
- HALF_OPEN: probe a small fraction of calls; promote to CLOSED on N consecutive
             successes; demote back to OPEN on a single failure.

The breaker exposes `allow_request()` for routers + `record_success`/`record_failure`
for callers (LLMCallRouter wrapper). All thresholds come from `CircuitBreakerConfig`
— no hardcoded numbers in code.
"""
import random
import threading
import time
from collections import deque
from typing import Callable, Deque, Optional, Tuple

from semantic_search.config.models import CircuitBreakerConfig
from semantic_search.core.exceptions import AgentSearchError
from semantic_search.core.logging_utils import get_logger
from semantic_search.contracts import CircuitBreakerState

logger = get_logger(__name__)

# Listener: (from_state, to_state, reason, observation_count); must not raise
TransitionListener = Callable[[str, str, str, int], None]


class CircuitOpenError(AgentSearchError):
    """Raised when the breaker rejects a call because the circuit is OPEN."""
    pass


class CircuitBreaker:
    """LLM closed/open/half_open state machine (LLM-only; Qdrant/CH breakers co-located)."""

    def __init__(self, config: CircuitBreakerConfig, rng: Optional[random.Random] = None, time_source=None, transition_listener: Optional[TransitionListener] = None):
        self._config = config
        self._lock = threading.RLock()
        self._state = 'closed'
        self._opened_at: Optional[float] = None
        self._consecutive_half_open_successes = 0
        self._total_open_transitions = 0
        self._events: Deque[Tuple[float, bool]] = deque()
        self._rng = rng if rng is not None else random.Random()  # nosec B311
        self._time = time_source if time_source is not None else time.monotonic
        self._transition_listener = transition_listener

    @property
    def state(self) -> str:
        """Current state: closed | open | half_open."""
        with self._lock:
            return self._state

    def snapshot(self) -> CircuitBreakerState:
        """Typed snapshot for resilience endpoint."""
        with self._lock:
            self._evict_old()
            successes = sum(1 for _, ok in self._events if ok)
            failures = sum(1 for _, ok in self._events if not ok)
            opened_at_wall: Optional[float] = None
            if self._opened_at is not None:
                opened_at_wall = time.time() - (self._time() - self._opened_at)
            return CircuitBreakerState(
                state=self._state,
                failure_count=failures,
                success_count=successes,
                consecutive_half_open_successes=self._consecutive_half_open_successes,
                opened_at=opened_at_wall,
                total_open_transitions=self._total_open_transitions,
            )

    def allow_request(self) -> bool:
        """Decide whether the next LLM call should be attempted.
        - CLOSED -> always True
        - OPEN -> True only after the cooldown has elapsed (transitions to HALF_OPEN)
        - HALF_OPEN -> True with probability `half_open_probe_ratio`; else False
        :return: bool
        """
        if not self._config.enabled:
            return True
        with self._lock:
            now = self._time()
            if self._state == 'open':
                if self._opened_at is not None and (now - self._opened_at) >= self._config.open_cooldown_seconds:
                    self._transition('half_open', reason='cooldown_elapsed')
                else:
                    return False
            if self._state == 'half_open':
                return self._rng.random() < self._config.half_open_probe_ratio
            return True

    def record_success(self) -> None:
        """Record a successful LLM call and update state."""
        if not self._config.enabled:
            return
        with self._lock:
            now = self._time()
            self._events.append((now, True))
            self._evict_old()
            if self._state == 'half_open':
                self._consecutive_half_open_successes += 1
                if self._consecutive_half_open_successes >= self._config.half_open_required_successes:
                    self._transition('closed', reason='probes_succeeded')
                    self._consecutive_half_open_successes = 0
                    self._opened_at = None

    def record_failure(self) -> None:
        """Record a failed LLM call and update state — may trip the breaker OPEN."""
        if not self._config.enabled:
            return
        with self._lock:
            now = self._time()
            self._events.append((now, False))
            self._evict_old()
            if self._state == 'half_open':
                self._transition('open', reason='probe_failed')
                self._opened_at = now
                self._consecutive_half_open_successes = 0
                return
            if self._state != 'closed':
                return
            total = len(self._events)
            if total < self._config.min_calls_before_trip:
                return
            failures = sum(1 for _, ok in self._events if not ok)
            failure_rate = failures / total if total else 0.0
            if failure_rate >= self._config.failure_rate_threshold:
                self._transition('open', reason=f"failure_rate={failure_rate:.3f}")
                self._opened_at = now

    def reset(self) -> None:
        """Force the breaker back to CLOSED (operator escape hatch)."""
        with self._lock:
            self._transition('closed', reason='manual_reset')
            self._events.clear()
            self._opened_at = None
            self._consecutive_half_open_successes = 0

    def _evict_old(self) -> None:
        """Drop events outside the rolling window (caller already holds the lock)."""
        cutoff = self._time() - self._config.rolling_window_seconds
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def _transition(self, target_state: str, reason: str) -> None:
        """Log + apply a state transition (caller already holds the lock)."""
        if target_state == self._state:
            return
        from_state = self._state
        observation_count = len(self._events)
        logger.info(f"circuit_breaker_transition from={from_state} to={target_state} reason={reason}")
        if target_state == 'open':
            # Counts every fresh trip to OPEN — drives the
            # "Circuit breaker activation count" proxy signal.
            self._total_open_transitions += 1
        self._state = target_state
        if self._transition_listener is not None:
            try:
                self._transition_listener(from_state, target_state, reason, observation_count)
            except Exception as e:
                # Best-effort — listener failure must not affect the breaker.
                logger.warning(f"circuit_breaker_listener_failed from={from_state} to={target_state} error_type={type(e).__name__} error={str(e)}")
