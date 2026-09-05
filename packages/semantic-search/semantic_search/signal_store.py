"""Append-only signal store for measurement and analytics consumers.
Writes one JSON object per line to a configured path and mirrors recent signals
in memory (FIFO trim above ``max_in_memory_signals``).
"""
import asyncio
import json
import os
import threading
from collections import deque
from dataclasses import asdict
from typing import Deque, Dict, List

from semantic_search.config.models import FeedbackConfig
from semantic_search.contracts import FeedbackSignal
from semantic_search.core.exceptions import ValidationError
from semantic_search.core.logging_utils import get_logger
from semantic_search.core.validation import sanitize_path_component
from llm_core.logging_utils import mask_path

logger = get_logger(__name__)


def schedule_feedback_signal_record(signal_store: "SignalStore", signal: FeedbackSignal) -> None:
    """Persist feedback signal non-blocking (async when loop running, sync otherwise)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        signal_store.record(signal)
        return

    async def _runner() -> None:
        try:
            await signal_store.record_async(signal)
        except Exception as e:
            logger.warning(
                f"feedback_signal_schedule_failed signal_type={signal.signal_type} "
                f"error_type={type(e).__name__} error={str(e)}"
            )

    loop.create_task(_runner())


class SignalStore:
    """Records FeedbackSignals to disk JSONL + in-memory ring buffer."""

    def __init__(self, config: FeedbackConfig):
        self._config = config
        self._allowed = frozenset(config.allowed_signal_types)
        self._lock = threading.Lock()
        self._memory: Deque[FeedbackSignal] = deque(maxlen=config.max_in_memory_signals)
        self._ensure_log_path()

    def _ensure_log_path(self) -> None:
        """Validate and create parent directory for JSONL log."""
        path = self._config.signal_log_path
        sanitize_path_component(os.path.basename(path), 'feedback.signal_log_path.basename')
        parent = os.path.dirname(path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

    def record(self, signal: FeedbackSignal) -> None:
        """Append signal to memory + disk (raises if signal_type not in allowlist)."""
        if not self._config.enabled:
            return
        if signal.signal_type not in self._allowed:
            raise ValidationError(f"feedback signal_type '{signal.signal_type}' not in allowed_signal_types")
        with self._lock:
            self._memory.append(signal)
            line = json.dumps(asdict(signal), default=str, sort_keys=True)
            try:
                with open(self._config.signal_log_path, 'a', encoding='utf-8') as f:
                    f.write(line + "\n")
            except (IOError, OSError) as e:
                logger.warning(f"feedback_signal_write_failed path={mask_path(self._config.signal_log_path)} error_type={type(e).__name__} error={str(e)}")
        logger.info(
            f"feedback_signal_recorded type={signal.signal_type} "
            f"search_id={signal.search_id} request_id={signal.request_id} "
            f"signal_id={signal.signal_id}"
        )

    async def record_async(self, signal: FeedbackSignal) -> None:
        """Append signal via worker thread (non-blocking I/O)."""
        await asyncio.to_thread(self.record, signal)

    def recent(self, limit: int) -> List[FeedbackSignal]:
        """Return most-recent N signals from memory."""
        if limit < 1:
            return []
        with self._lock:
            buf = list(self._memory)
        return buf[-limit:]

    def stats(self) -> Dict[str, int]:
        """Return signal_type counts from memory."""
        out: Dict[str, int] = {t: 0 for t in self._allowed}
        with self._lock:
            for sig in self._memory:
                out[sig.signal_type] = out.get(sig.signal_type, 0) + 1
        return out


__all__ = ["SignalStore", "schedule_feedback_signal_record"]
