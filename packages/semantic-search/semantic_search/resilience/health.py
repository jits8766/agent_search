"""Backend health registry.
Tracks per-backend (vector / structured / sql / llm / clickhouse / eranker / bulk)
success/failure rates over a rolling window. Backends transition to `degraded` when failure-rate exceeds
`failure_rate_threshold` and to `unhealthy` when probes also fail. The registry
is thread-safe and consulted by the orchestrator before issuing a retrieval call.
"""
import threading
import time
from collections import defaultdict, deque
from typing import Callable, Deque, Dict, List, Optional, Tuple

from semantic_search.config.models import BackendHealthConfig
from semantic_search.core.exceptions import ValidationError
from semantic_search.core.logging_utils import get_logger
from semantic_search.contracts import BACKEND_HEALTH_BACKENDS, BackendHealth

logger = get_logger(__name__)

# Listener: (backend, from_state, to_state, observation_count); exceptions swallowed
HealthTransitionListener = Callable[[str, str, str, int], None]


class BackendHealthRegistry:
    """Per-backend rolling-window failure tracker (healthy|degraded|unhealthy)."""

    def __init__(self, config: BackendHealthConfig, transition_listener: Optional[HealthTransitionListener] = None):
        self._config = config
        self._lock = threading.RLock()
        self._events: Dict[str, Deque[Tuple[float, bool]]] = defaultdict(deque)
        self._states: Dict[str, str] = {b: 'healthy' for b in BACKEND_HEALTH_BACKENDS}
        self._unhealthy_since: Dict[str, float] = {}
        self._transition_listener = transition_listener

    def record(self, backend: str, success: bool) -> None:
        """Record event; update state on threshold cross."""
        if not self._config.enabled:
            return
        if backend not in BACKEND_HEALTH_BACKENDS:
            raise ValidationError(f"unknown backend: {backend}")
        with self._lock:
            now = time.monotonic()
            self._events[backend].append((now, success))
            self._evict_old(backend, now)
            self._reassess(backend, now)

    def is_healthy(self, backend: str) -> bool:
        """True iff healthy or degraded (still serves traffic)."""
        if not self._config.enabled:
            return True
        if backend not in BACKEND_HEALTH_BACKENDS:
            return True
        if backend in frozenset(self._config.force_unhealthy_backends):
            return False
        with self._lock:
            return self._states.get(backend, 'healthy') != 'unhealthy'

    def is_probe_eligible(self, backend: str) -> bool:
        """True iff unhealthy AND recovery_probe_seconds elapsed (half-open probe)."""
        if not self._config.enabled:
            return False
        if backend not in BACKEND_HEALTH_BACKENDS:
            return False
        with self._lock:
            if self._states.get(backend, 'healthy') != 'unhealthy':
                return False
            unhealthy_since = self._unhealthy_since.get(backend, 0.0)
            return (time.monotonic() - unhealthy_since) >= self._config.recovery_probe_seconds

    def state(self, backend: str) -> str:
        """Return the current state ('healthy' / 'degraded' / 'unhealthy')."""
        if not self._config.enabled:
            return 'healthy'
        if backend in frozenset(self._config.force_unhealthy_backends):
            return 'unhealthy'
        with self._lock:
            return self._states.get(backend, 'healthy')

    def snapshot(self) -> List[BackendHealth]:
        """Typed snapshot of every tracked backend (for /resilience/health)."""
        if not self._config.enabled:
            return [BackendHealth(backend=b, state='healthy', failure_rate=0.0, observation_count=0) for b in sorted(BACKEND_HEALTH_BACKENDS)]
        with self._lock:
            now = time.monotonic()
            out: List[BackendHealth] = []
            for backend in sorted(BACKEND_HEALTH_BACKENDS):
                self._evict_old(backend, now)
                events = list(self._events.get(backend, ()))
                total = len(events)
                failures = sum(1 for _, ok in events if not ok)
                rate = failures / total if total else 0.0
                out.append(BackendHealth(backend=backend, state=self._states.get(backend, 'healthy'), failure_rate=rate, observation_count=total))
            return out

    def reset(self, backend: str) -> None:
        """Force a backend back to 'healthy' (operator escape hatch)."""
        with self._lock:
            if backend in BACKEND_HEALTH_BACKENDS:
                self._states[backend] = 'healthy'
                self._events[backend].clear()
                self._unhealthy_since.pop(backend, None)
                logger.info(f"backend_health_reset backend={backend}")

    def _evict_old(self, backend: str, now: float) -> None:
        """Drop events outside the rolling window (caller already holds the lock)."""
        events = self._events.get(backend)
        if not events:
            return
        cutoff = now - self._config.rolling_window_seconds
        while events and events[0][0] < cutoff:
            events.popleft()

    def _reassess(self, backend: str, now: float) -> None:
        """Recompute state given current window contents (caller already holds the lock)."""
        events = self._events.get(backend, ())
        total = len(events)
        prior_state = self._states.get(backend, 'healthy')
        if total < self._config.min_observations:
            target_state = 'healthy' if (prior_state == 'unhealthy' and (now - self._unhealthy_since.get(backend, 0.0)) >= self._config.recovery_probe_seconds) else prior_state
        else:
            failures = sum(1 for _, ok in events if not ok)
            rate = failures / total if total else 0.0
            if rate >= self._config.failure_rate_threshold:
                if prior_state != 'unhealthy':
                    target_state = 'unhealthy'
                    self._unhealthy_since[backend] = now
                else:
                    target_state = 'unhealthy'
            else:
                if prior_state == 'unhealthy' and (now - self._unhealthy_since.get(backend, 0.0)) < self._config.recovery_probe_seconds:
                    target_state = 'unhealthy'
                else:
                    target_state = 'degraded' if rate > 0 else 'healthy'
                    if target_state == 'healthy':
                        self._unhealthy_since.pop(backend, None)
        if target_state != prior_state:
            logger.info(f"backend_health_transition backend={backend} from={prior_state} to={target_state} observations={total}")
            self._states[backend] = target_state
            if self._transition_listener is not None:
                try:
                    self._transition_listener(backend, prior_state, target_state, total)
                except Exception as e:
                    # Best-effort — listener failure must not affect the registry.
                    logger.warning(f"backend_health_listener_failed backend={backend} from={prior_state} to={target_state} error_type={type(e).__name__} error={str(e)}")
