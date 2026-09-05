"""Phase 1 launch stats for ``qie_only`` - in-process rates for go/no-go.

Structured logs (``qie_only_complete`` / ``qie_only_failed``) remain the durable
CloudWatch source. This module aggregates the same events into a rolling window
so decision makers can call ``GET /measurement/qie_only`` without log math.

Only QI-owned gates are evaluated here. FoS wiring, FIND Kinesis
``experimentInfo``, and funnel metrics stay external (status=``external``).
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from semantic_search.config.models import QieOnlyLaunchConfig
from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)


def _percentile_nearest_rank(sorted_vals: List[float], percentile: float) -> Optional[float]:
    if not sorted_vals:
        return None
    idx = max(0, min(len(sorted_vals) - 1, int(math.ceil(percentile * len(sorted_vals))) - 1))
    return float(sorted_vals[idx])


class QieOnlyLaunchStats:
    """Thread-safe rolling counters for Phase 1 QI launch gates."""

    def __init__(
        self,
        *,
        min_sample_size: int,
        fail_rate_max: float,
        p95_latency_ms_max: float,
        find_skipped_rate_max: float,
        hard_params_empty_rate_max: float,
        latency_window: int,
    ) -> None:
        self._min_sample = max(1, int(min_sample_size))
        self._fail_rate_max = float(fail_rate_max)
        self._p95_latency_ms_max = float(p95_latency_ms_max)
        self._find_skipped_rate_max = float(find_skipped_rate_max)
        self._hard_params_empty_rate_max = float(hard_params_empty_rate_max)
        self._lock = threading.Lock()
        self._complete = 0
        self._failed = 0
        self._failed_by_status: Dict[str, int] = {}
        self._find_skipped_nonzero = 0
        self._hard_params_empty = 0
        self._source_counts: Dict[str, int] = {}
        self._latencies_ms: Deque[float] = deque(maxlen=max(10, int(latency_window)))
        self._started_at = time.time()

    def record_complete(
        self,
        *,
        latency_ms: float,
        find_skipped_count: int,
        hard_params_empty: int,
        source: str,
    ) -> None:
        """Record one successful ``qie_only`` response."""
        src = (source or "-").strip() or "-"
        with self._lock:
            self._complete += 1
            self._latencies_ms.append(float(latency_ms))
            if int(find_skipped_count) > 0:
                self._find_skipped_nonzero += 1
            if int(hard_params_empty) == 1:
                self._hard_params_empty += 1
            self._source_counts[src] = self._source_counts.get(src, 0) + 1

    def record_failed(self, *, status: int, reason: str = "") -> None:
        """Record one ``qie_only`` 422/503 (or other) failure."""
        key = str(int(status))
        with self._lock:
            self._failed += 1
            self._failed_by_status[key] = self._failed_by_status.get(key, 0) + 1
        if reason:
            logger.debug(
                f"qie_only_launch_stats_failed status={key} reason={reason}"
            )

    def reset(self) -> None:
        """Test / operator escape hatch - clear the rolling window."""
        with self._lock:
            self._complete = 0
            self._failed = 0
            self._failed_by_status.clear()
            self._find_skipped_nonzero = 0
            self._hard_params_empty = 0
            self._source_counts.clear()
            self._latencies_ms.clear()
            self._started_at = time.time()

    def snapshot(self) -> Dict[str, Any]:
        """Rates + per-gate pass/fail for decision makers."""
        with self._lock:
            complete = self._complete
            failed = self._failed
            failed_by_status = dict(self._failed_by_status)
            skipped_nz = self._find_skipped_nonzero
            hard_empty = self._hard_params_empty
            source_counts = dict(self._source_counts)
            latencies = sorted(self._latencies_ms)
            started_at = self._started_at

        attempts = complete + failed
        fail_rate = (failed / attempts) if attempts else None
        skipped_rate = (skipped_nz / complete) if complete else None
        hard_empty_rate = (hard_empty / complete) if complete else None
        p50 = _percentile_nearest_rank(latencies, 0.50)
        p95 = _percentile_nearest_rank(latencies, 0.95)

        gates = {
            "qi_availability": self._rate_gate(
                name="qi_availability",
                value=fail_rate,
                threshold=self._fail_rate_max,
                sample=attempts,
                direction="lower_is_better",
                have=True,
                owner="ML",
                evidence="qie_only_failed / (complete+failed)",
            ),
            "qi_latency_p95_ms": self._rate_gate(
                name="qi_latency_p95_ms",
                value=p95,
                threshold=self._p95_latency_ms_max,
                sample=len(latencies),
                direction="lower_is_better",
                have=True,
                owner="ML",
                evidence="latency_ms on qie_only_complete",
            ),
            "find_skipped_rate": self._rate_gate(
                name="find_skipped_rate",
                value=skipped_rate,
                threshold=self._find_skipped_rate_max,
                sample=complete,
                direction="lower_is_better",
                have=True,
                owner="ML",
                evidence="find_skipped_count > 0 share",
            ),
            "hard_params_empty_rate": self._rate_gate(
                name="hard_params_empty_rate",
                value=hard_empty_rate,
                threshold=self._hard_params_empty_rate_max,
                sample=complete,
                direction="lower_is_better",
                have=True,
                owner="ML",
                evidence="hard_params_empty=1 share",
            ),
            "fos_wiring": self._external_gate(
                name="fos_wiring",
                owner="FoS",
                evidence="treatment qie_only -> FIND + experimentInfo; degrade on QI error",
            ),
            "fos_timeout_degrade_rate": self._external_gate(
                name="fos_timeout_degrade_rate",
                owner="FoS",
                evidence="FoS client timeout -> control FIND path",
            ),
            "find_experiment_attribution": self._external_gate(
                name="find_experiment_attribution",
                owner="FIND+FoS",
                evidence="Kinesis experimentInfo split + Pagination.Total",
            ),
            "safety_auth": self._external_gate(
                name="safety_auth",
                owner="FoS+FIND+ML",
                evidence="no PII in QI logs; FIND auth unchanged",
            ),
        }

        ml_statuses = [
            g["status"]
            for g in gates.values()
            if g.get("owner") == "ML" and g.get("have") is True
        ]
        if any(s == "fail" for s in ml_statuses):
            ml_overall = "fail"
        elif any(s == "insufficient_sample" for s in ml_statuses):
            ml_overall = "insufficient_sample"
        elif ml_statuses and all(s == "pass" for s in ml_statuses):
            ml_overall = "pass"
        else:
            ml_overall = "insufficient_sample"

        return {
            "surface": "qie_only_phase1_launch",
            "generated_at": time.time(),
            "window_started_at": started_at,
            "min_sample_size": self._min_sample,
            "counts": {
                "complete": complete,
                "failed": failed,
                "attempts": attempts,
                "find_skipped_nonzero": skipped_nz,
                "hard_params_empty": hard_empty,
                "failed_by_status": failed_by_status,
            },
            "rates": {
                "fail_rate": fail_rate,
                "find_skipped_rate": skipped_rate,
                "hard_params_empty_rate": hard_empty_rate,
                "p50_latency_ms": p50,
                "p95_latency_ms": p95,
            },
            "source_mix": source_counts,
            "thresholds": {
                "fail_rate_max": self._fail_rate_max,
                "p95_latency_ms_max": self._p95_latency_ms_max,
                "find_skipped_rate_max": self._find_skipped_rate_max,
                "hard_params_empty_rate_max": self._hard_params_empty_rate_max,
            },
            "gates": gates,
            "ml_owned_overall": ml_overall,
            "phase1_launch_overall": "pending_external_gates",
            "note": (
                "ml_owned_overall covers QI availability/latency/wire/empty-hard only. "
                "phase1_launch_overall stays pending_external_gates until FoS/FIND "
                "confirm wiring, experimentInfo attribution, and safety."
            ),
        }

    def _rate_gate(
        self,
        *,
        name: str,
        value: Optional[float],
        threshold: float,
        sample: int,
        direction: str,
        have: bool,
        owner: str,
        evidence: str,
    ) -> Dict[str, Any]:
        if sample < self._min_sample or value is None:
            status = "insufficient_sample"
        elif direction == "lower_is_better":
            status = "pass" if value <= threshold else "fail"
        else:
            status = "pass" if value >= threshold else "fail"
        return {
            "name": name,
            "status": status,
            "value": value,
            "threshold": threshold,
            "sample_size": sample,
            "direction": direction,
            "have": have,
            "owner": owner,
            "evidence": evidence,
        }

    @staticmethod
    def _external_gate(*, name: str, owner: str, evidence: str) -> Dict[str, Any]:
        return {
            "name": name,
            "status": "external",
            "value": None,
            "threshold": None,
            "sample_size": 0,
            "direction": "informational",
            "have": False,
            "owner": owner,
            "evidence": evidence,
        }


_STATS: Optional[QieOnlyLaunchStats] = None
_STATS_LOCK = threading.Lock()


def _stats_from_config(config: QieOnlyLaunchConfig) -> QieOnlyLaunchStats:
    return QieOnlyLaunchStats(
        min_sample_size=config.min_sample_size,
        fail_rate_max=config.fail_rate_max,
        p95_latency_ms_max=config.p95_latency_ms_max,
        find_skipped_rate_max=config.find_skipped_rate_max,
        hard_params_empty_rate_max=config.hard_params_empty_rate_max,
        latency_window=config.latency_window,
    )


def configure_qie_only_launch_stats(config: QieOnlyLaunchConfig) -> QieOnlyLaunchStats:
    """Initialize process-singleton launch stats from typed YAML config."""
    global _STATS
    with _STATS_LOCK:
        _STATS = _stats_from_config(config)
        return _STATS


def get_qie_only_launch_stats() -> QieOnlyLaunchStats:
    """Process-singleton launch stats (same lifetime as the API process)."""
    with _STATS_LOCK:
        if _STATS is None:
            raise RuntimeError("qie_only launch stats not configured")
        return _STATS


def reset_qie_only_launch_stats_for_tests(config: QieOnlyLaunchConfig) -> QieOnlyLaunchStats:
    """Replace singleton - tests only."""
    global _STATS
    with _STATS_LOCK:
        _STATS = _stats_from_config(config)
        return _STATS


__all__ = [
    "QieOnlyLaunchStats",
    "configure_qie_only_launch_stats",
    "get_qie_only_launch_stats",
    "reset_qie_only_launch_stats_for_tests",
]
