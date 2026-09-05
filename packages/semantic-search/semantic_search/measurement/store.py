"""Thread-safe in-memory rolling window of `SearchObservation` instances.

The store is intentionally simple: a fixed-capacity deque protected by a single
lock. Reads (`recent`, `snapshot`) return a copy so callers never see the live
buffer. Writes are O(1) amortized.

The window survives only for the lifetime of the process — the measurement
subsystem needs trend visibility, not durability. A future change to add
JSONL persistence would wrap ``record`` without changing the read API.
"""
import threading
from collections import deque
from typing import Deque, List, Optional

from semantic_search.config.models import MeasurementConfig
from semantic_search.contracts import SearchObservation
from semantic_search.core.exceptions import ValidationError
from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)


class MeasurementStore:
    """Bounded in-memory ring buffer of `SearchObservation` records.

    :param config: MeasurementConfig - Capacity + sample-size guards
    """

    def __init__(self, config: MeasurementConfig):
        self._config = config
        self._lock = threading.Lock()
        self._buffer: Deque[SearchObservation] = deque(maxlen=config.window_size)

    def record(self, observation: SearchObservation) -> None:
        """Append one observation. No-op when measurement is disabled.
        :param observation: SearchObservation - Pre-validated observation
        :raises ValidationError: When `observation` is not a SearchObservation
        """
        if not self._config.enabled:
            return
        if not isinstance(observation, SearchObservation):
            raise ValidationError("MeasurementStore.record requires a SearchObservation")
        with self._lock:
            self._buffer.append(observation)

    def snapshot(self) -> List[SearchObservation]:
        """Return a defensive copy of the window's contents (oldest first).
        :return: List[SearchObservation]
        """
        with self._lock:
            return list(self._buffer)

    def recent(self, limit: int) -> List[SearchObservation]:
        """Return the most-recent N observations (newest last, oldest first slice).
        :param limit: int - Maximum number of observations to return (>=0)
        :return: List[SearchObservation]
        """
        if limit < 1:
            return []
        with self._lock:
            buf = list(self._buffer)
        if limit >= len(buf):
            return buf
        return buf[-limit:]

    def size(self) -> int:
        """Current observation count."""
        with self._lock:
            return len(self._buffer)

    def capacity(self) -> int:
        """Configured window capacity."""
        return self._config.window_size

    def min_sample_size(self) -> int:
        """Minimum sample size required before rate-based signals are published."""
        return self._config.min_sample_size

    def clear(self) -> None:
        """Drop every observation in the window (operator escape hatch / tests)."""
        with self._lock:
            self._buffer.clear()

    def first_observation_time(self) -> Optional[float]:
        """Wall-clock timestamp of the oldest observation in the window, or None."""
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer[0].created_at
