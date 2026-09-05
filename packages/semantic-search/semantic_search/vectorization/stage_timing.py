"""Config-driven stage timing spans for data-build seed pipelines.

Logs via ``get_logger(__name__)``. Knobs come from ``SeedStageTimingConfig``.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

from semantic_search.config.models import SeedStageTimingConfig
from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _fmt_elapsed_ms(elapsed_ms: float, decimals: int) -> float:
    return round(float(elapsed_ms), int(decimals))


def _fields_str(fields: Dict[str, Any]) -> str:
    if not fields:
        return ""
    parts = []
    for key in sorted(fields.keys()):
        val = fields[key]
        parts.append(f"{key}={val}")
    return " " + " ".join(parts)


class StageTimingSession:
    """Accumulate stage spans for logs and optional API response payload."""

    def __init__(self, config: SeedStageTimingConfig) -> None:
        if not isinstance(config, SeedStageTimingConfig):
            raise TypeError("StageTimingSession requires a SeedStageTimingConfig")
        self._config = config
        self._records: List[Dict[str, Any]] = []
        self._job_mono_start = time.monotonic()
        self._job_wall_start = _utc_now_iso()

    @property
    def config(self) -> SeedStageTimingConfig:
        return self._config

    @property
    def records(self) -> List[Dict[str, Any]]:
        return list(self._records)

    def summary(self) -> Dict[str, Any]:
        """Build response/history payload when ``include_in_response`` is set."""
        total_ms = _fmt_elapsed_ms(
            (time.monotonic() - self._job_mono_start) * 1000.0,
            self._config.elapsed_ms_decimals,
        )
        by_stage: Dict[str, float] = {}
        for rec in self._records:
            stage = str(rec["stage"])
            by_stage[stage] = by_stage.get(stage, 0.0) + float(rec["elapsed_ms"])
        for stage, ms in list(by_stage.items()):
            by_stage[stage] = _fmt_elapsed_ms(ms, self._config.elapsed_ms_decimals)
        return {
            "enabled": self._config.enabled,
            "job_started_at": self._job_wall_start,
            "job_ended_at": _utc_now_iso(),
            "job_elapsed_ms": total_ms,
            "stages": list(self._records),
            "elapsed_ms_by_stage": by_stage,
        }

    @contextmanager
    def stage(
        self,
        stage: str,
        *,
        gate: bool,
        **fields: Any,
    ) -> Iterator[None]:
        """Time one stage when ``config.enabled`` and ``gate`` are both True.

        ``gate`` selects detail flags from config (page / merge-phase / indexer).
        Top-level stages pass ``gate=True`` so only ``enabled`` controls them.
        """
        if not self._config.enabled or not gate:
            yield
            return
        started_at = _utc_now_iso()
        t0 = time.monotonic()
        logger.info(
            f"data_build_stage_start stage={stage} started_at={started_at}"
            f"{_fields_str(fields)}"
        )
        error_type: Optional[str] = None
        try:
            yield
        except BaseException as exc:
            error_type = type(exc).__name__
            raise
        finally:
            ended_at = _utc_now_iso()
            elapsed_ms = _fmt_elapsed_ms(
                (time.monotonic() - t0) * 1000.0,
                self._config.elapsed_ms_decimals,
            )
            rec: Dict[str, Any] = {
                "stage": stage,
                "started_at": started_at,
                "ended_at": ended_at,
                "elapsed_ms": elapsed_ms,
            }
            rec.update(fields)
            if error_type is not None:
                rec["error_type"] = error_type
            self._records.append(rec)
            err_part = f" error_type={error_type}" if error_type is not None else ""
            logger.info(
                f"data_build_stage_end stage={stage} started_at={started_at} "
                f"ended_at={ended_at} elapsed_ms={elapsed_ms}{_fields_str(fields)}{err_part}"
            )

    def record_elapsed(
        self,
        stage: str,
        *,
        gate: bool,
        elapsed_ms: float,
        **fields: Any,
    ) -> None:
        """Record a pre-measured span (e.g. indexer encode/upsert totals per page)."""
        if not self._config.enabled or not gate:
            return
        ended_at = _utc_now_iso()
        elapsed = _fmt_elapsed_ms(elapsed_ms, self._config.elapsed_ms_decimals)
        # Wall start is approximate (monotonic elapsed applied backward from end).
        started_at = ended_at
        rec: Dict[str, Any] = {
            "stage": stage,
            "started_at": started_at,
            "ended_at": ended_at,
            "elapsed_ms": elapsed,
        }
        rec.update(fields)
        self._records.append(rec)
        logger.info(
            f"data_build_stage_end stage={stage} started_at={started_at} "
            f"ended_at={ended_at} elapsed_ms={elapsed}{_fields_str(fields)}"
        )

    def log_job_complete(self, **fields: Any) -> None:
        """Emit job-level summary line when timing is enabled."""
        if not self._config.enabled:
            return
        summary = self.summary()
        logger.info(
            f"data_build_stage_summary job_started_at={summary['job_started_at']} "
            f"job_ended_at={summary['job_ended_at']} "
            f"job_elapsed_ms={summary['job_elapsed_ms']} "
            f"elapsed_ms_by_stage={summary['elapsed_ms_by_stage']}"
            f"{_fields_str(fields)}"
        )
