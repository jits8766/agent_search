"""Signal adapter — reads agent_feedback signals and aggregates per-model RuntimeStats.

Pure adapter. Safe no-op when store is None or agent_feedback is unavailable.
Decoupled via duck-typing: any store exposing `get_entries(signal_type=..., limit=...) -> List`
where each entry has `.timestamp`, `.value`, `.metadata` works.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from llm_core.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RuntimeStats:
    """Aggregated runtime stats for one model.
    :param p50_latency_ms: Optional[float] - Median observed latency
    :param error_rate: float - Failed calls / total calls in window
    :param quality_score: Optional[float] - Mean eval_quality_score in window
    :param sample_count: int - Total observations contributing to stats
    """
    p50_latency_ms: Optional[float] = None
    error_rate: float = 0.0
    quality_score: Optional[float] = None
    sample_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'p50_latency_ms': self.p50_latency_ms,
            'error_rate': self.error_rate,
            'quality_score': self.quality_score,
            'sample_count': self.sample_count,
        }


def _parse_ts(ts: Any) -> Optional[datetime]:
    """Best-effort ISO timestamp parse to aware UTC datetime."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    try:
        s = str(ts).replace('Z', '+00:00')
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _percentile(values: List[float], pct: float) -> Optional[float]:
    """Linear-interpolation percentile. Returns None for empty input."""
    if not values:
        return None
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * pct
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


class SignalAdapter:
    """Reads recent latency/error/quality signals from a FeedbackStore-like object.

    Aggregates per-model. Looks at metadata['model'] (preferred) and falls back to
    metadata['model_name'] for compatibility.
    """

    def __init__(self, store: Optional[Any], latency_window_s: int, quality_window_s: int, max_entries_per_type: int):
        """Initialize adapter. All sizing parameters are required (no silent defaults).
        :param store: Optional[Any] - FeedbackStore-like with get_entries(signal_type, limit); None disables
        :param latency_window_s: int - Latency/error window in seconds
        :param quality_window_s: int - Quality window in seconds
        :param max_entries_per_type: int - Cap entries pulled per signal type per refresh
        """
        self._store = store
        self._latency_window_s = int(latency_window_s)
        self._quality_window_s = int(quality_window_s)
        self._max_entries = int(max_entries_per_type)

    @property
    def enabled(self) -> bool:
        """True if a backing store is wired."""
        return self._store is not None

    def collect(self) -> Dict[str, Dict[str, Any]]:
        """Pull recent signals and return per-model stats dict.
        :return: Dict[str, Dict[str, Any]] - model_name -> RuntimeStats.to_dict(); empty when no store
        """
        if self._store is None:
            return {}
        try:
            now = datetime.now(timezone.utc)
            lat_cutoff = now - timedelta(seconds=self._latency_window_s)
            qual_cutoff = now - timedelta(seconds=self._quality_window_s)

            latency_by_model: Dict[str, List[float]] = {}
            errors_by_model: Dict[str, List[int]] = {}
            quality_by_model: Dict[str, List[float]] = {}

            for entry in self._safe_get_entries('latency'):
                ts = _parse_ts(getattr(entry, 'timestamp', None))
                if ts is None or ts < lat_cutoff:
                    continue
                model = self._extract_model(entry)
                if not model:
                    continue
                latency_by_model.setdefault(model, []).append(float(getattr(entry, 'value', 0.0)))

            for entry in self._safe_get_entries('error'):
                ts = _parse_ts(getattr(entry, 'timestamp', None))
                if ts is None or ts < lat_cutoff:
                    continue
                model = self._extract_model(entry)
                if not model:
                    continue
                errors_by_model.setdefault(model, []).append(1)

            for entry in self._safe_get_entries('eval_quality_score'):
                ts = _parse_ts(getattr(entry, 'timestamp', None))
                if ts is None or ts < qual_cutoff:
                    continue
                model = self._extract_model(entry)
                if not model:
                    continue
                value = getattr(entry, 'value', None)
                meta = getattr(entry, 'metadata', {}) or {}
                qval = value if value is not None else meta.get('quality_score')
                if qval is None:
                    continue
                try:
                    quality_by_model.setdefault(model, []).append(float(qval))
                except (TypeError, ValueError):
                    continue

            stats: Dict[str, Dict[str, Any]] = {}
            all_models = set(latency_by_model) | set(errors_by_model) | set(quality_by_model)
            for model in all_models:
                lat = latency_by_model.get(model, [])
                err_count = len(errors_by_model.get(model, []))
                qual = quality_by_model.get(model, [])
                total = len(lat) + err_count
                error_rate = (err_count / total) if total > 0 else 0.0
                stats[model] = RuntimeStats(p50_latency_ms=_percentile(lat, 0.5), error_rate=error_rate, quality_score=(sum(qual) / len(qual)) if qual else None, sample_count=total + len(qual)).to_dict()
            logger.info(f"signal_adapter_collected models={len(stats)}")
            return stats
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"signal_adapter_collect_failed error_type={type(e).__name__} error={str(e)}")
            return {}

    def _safe_get_entries(self, signal_type: str) -> List[Any]:
        """Resilient wrapper for store.get_entries; returns [] when the store cannot serve that signal.

        Catches `Exception` because the store is a duck-typed dependency from another package; we
        cannot enumerate its failure modes (DB errors, schema drift, unsupported signal types).
        Missing signals are an expected/recoverable condition for adaptive ranking.
        """
        try:
            return list(self._store.get_entries(signal_type=signal_type, limit=self._max_entries))
        except Exception as e:  # noqa: BLE001 — duck-typed external store
            logger.debug(f"signal_adapter_no_signal type={signal_type} error_type={type(e).__name__}")
            return []

    @staticmethod
    def _extract_model(entry: Any) -> Optional[str]:
        """Find model name in entry.metadata; supports 'model' or 'model_name'."""
        meta = getattr(entry, 'metadata', None)
        if not isinstance(meta, dict):
            return None
        model = meta.get('model') or meta.get('model_name')
        if not model or not isinstance(model, str):
            return None
        return model
