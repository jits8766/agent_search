"""Fleet-wide (cross-request) LLM USD budget.

Complements ``QueryCostBudget`` (per-request). Accumulates estimated LLM USD
across all requests in a process (``memory`` backend) or across replicas
(``redis`` backend). Caps are optional per-hour and/or per-day UTC buckets.

Design:
- Shared instance for the process lifetime (wired once in the registry).
- ``check_admit()`` raises when already exhausted — call BEFORE an LLM call
  so a hot fleet does not keep burning tokens.
- ``record_cost()`` adds after a successful call; may raise if the add tips
  the cap (last call already billed — same semantics as per-query budget).
- Redis failure fails open to local memory for that process so search stays
  available; logs warn so SRE can see split-brain risk.
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Protocol, Tuple

from semantic_search.core.exceptions import FleetCostBudgetExceeded, ValidationError
from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)


def _utc_bucket_keys(now: Optional[float] = None) -> Tuple[str, str]:
    """Return (hour_key, day_key) in UTC for wall-clock budget windows."""
    if now is None:
        dt = datetime.now(timezone.utc)
    else:
        dt = datetime.fromtimestamp(float(now), tz=timezone.utc)
    return dt.strftime('%Y%m%d%H'), dt.strftime('%Y%m%d')


class FleetCostStore(Protocol):
    """Backend that accumulates USD into hour/day buckets."""

    def get_totals(self, hour_key: str, day_key: str) -> Tuple[float, float]:
        """Return (hour_total_usd, day_total_usd) without mutating."""
        ...

    def add(self, cost_usd: float, hour_key: str, day_key: str) -> Tuple[float, float]:
        """Add cost; return new (hour_total_usd, day_total_usd)."""
        ...


class InMemoryFleetCostStore:
    """Process-local hour/day accumulators (multi-replica = per-pod caps)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hour: Dict[str, float] = {}
        self._day: Dict[str, float] = {}

    def get_totals(self, hour_key: str, day_key: str) -> Tuple[float, float]:
        with self._lock:
            return float(self._hour.get(hour_key, 0.0)), float(self._day.get(day_key, 0.0))

    def add(self, cost_usd: float, hour_key: str, day_key: str) -> Tuple[float, float]:
        with self._lock:
            # Drop stale buckets opportunistically (keep memory bounded).
            if len(self._hour) > 48:
                keep = {hour_key}
                self._hour = {k: v for k, v in self._hour.items() if k in keep}
            if len(self._day) > 14:
                keep_d = {day_key}
                self._day = {k: v for k, v in self._day.items() if k in keep_d}
            self._hour[hour_key] = float(self._hour.get(hour_key, 0.0)) + float(cost_usd)
            self._day[day_key] = float(self._day.get(day_key, 0.0)) + float(cost_usd)
            return float(self._hour[hour_key]), float(self._day[day_key])

    def reset(self) -> None:
        """Test-isolation hook — clear all buckets."""
        with self._lock:
            self._hour.clear()
            self._day.clear()


class RedisFleetCostStore:
    """Shared hour/day accumulators via Redis INCRBYFLOAT (+ TTL)."""

    def __init__(self, client: Any, key_prefix: str) -> None:
        if client is None:
            raise ValidationError("RedisFleetCostStore requires a redis client")
        if not isinstance(key_prefix, str) or not key_prefix:
            raise ValidationError("RedisFleetCostStore.key_prefix must be a non-empty str")
        self._client = client
        self._prefix = key_prefix
        self._fallback = InMemoryFleetCostStore()

    def _keys(self, hour_key: str, day_key: str) -> Tuple[str, str]:
        return f"{self._prefix}hour:{hour_key}", f"{self._prefix}day:{day_key}"

    def get_totals(self, hour_key: str, day_key: str) -> Tuple[float, float]:
        hk, dk = self._keys(hour_key, day_key)
        try:
            h_raw, d_raw = self._client.mget(hk, dk)
            h = float(h_raw) if h_raw is not None else 0.0
            d = float(d_raw) if d_raw is not None else 0.0
            return h, d
        except Exception as e:  # noqa: BLE001 — redis is optional infra
            logger.warning(
                f"fleet_cost_redis_get_failed error_type={type(e).__name__} error={str(e)} "
                f"falling_back_to=memory"
            )
            return self._fallback.get_totals(hour_key, day_key)

    def add(self, cost_usd: float, hour_key: str, day_key: str) -> Tuple[float, float]:
        hk, dk = self._keys(hour_key, day_key)
        amount = float(cost_usd)
        try:
            pipe = self._client.pipeline()
            pipe.incrbyfloat(hk, amount)
            pipe.expire(hk, 7200)  # 2h
            pipe.incrbyfloat(dk, amount)
            pipe.expire(dk, 172800)  # 48h
            results = pipe.execute()
            # results: [hour_total, True/1, day_total, True/1]
            return float(results[0]), float(results[2])
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"fleet_cost_redis_add_failed error_type={type(e).__name__} error={str(e)} "
                f"falling_back_to=memory"
            )
            return self._fallback.add(cost_usd, hour_key, day_key)


def build_fleet_cost_store(
    backend: str,
    redis_url_env_var: str,
    key_prefix: str,
    socket_timeout_seconds: float = 1.0,
) -> FleetCostStore:
    """Build store; redis unavailable → memory + warning."""
    backend_norm = (backend or 'memory').strip().lower()
    if backend_norm == 'memory':
        return InMemoryFleetCostStore()
    if backend_norm != 'redis':
        raise ValidationError(
            f"fleet cost backend must be 'memory' or 'redis'; got {backend!r}"
        )
    url = os.environ.get(redis_url_env_var, '')
    if not url:
        logger.warning(
            f"fleet_cost_redis_disabled reason=missing_env var={redis_url_env_var} "
            f"falling_back_to=memory"
        )
        return InMemoryFleetCostStore()
    try:
        import redis  # type: ignore
        client = redis.Redis.from_url(
            url,
            socket_timeout=float(socket_timeout_seconds),
            decode_responses=True,
        )
        client.ping()
        logger.info(f"fleet_cost_redis_connected prefix={key_prefix}")
        return RedisFleetCostStore(client=client, key_prefix=key_prefix)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            f"fleet_cost_redis_unavailable error_type={type(e).__name__} error={str(e)} "
            f"falling_back_to=memory"
        )
        return InMemoryFleetCostStore()


class FleetCostBudget:
    """Cross-request LLM USD cap (hour and/or day UTC windows)."""

    def __init__(
        self,
        store: FleetCostStore,
        max_cost_usd_per_hour: Optional[float] = None,
        max_cost_usd_per_day: Optional[float] = None,
        clock: Optional[Any] = None,
    ):
        if store is None:
            raise ValidationError("FleetCostBudget requires a store")
        hour_cap = None if max_cost_usd_per_hour is None else float(max_cost_usd_per_hour)
        day_cap = None if max_cost_usd_per_day is None else float(max_cost_usd_per_day)
        if hour_cap is not None and hour_cap <= 0.0:
            hour_cap = None
        if day_cap is not None and day_cap <= 0.0:
            day_cap = None
        if hour_cap is None and day_cap is None:
            raise ValidationError(
                "FleetCostBudget requires max_cost_usd_per_hour > 0 and/or max_cost_usd_per_day > 0"
            )
        self._store = store
        self._max_hour = hour_cap
        self._max_day = day_cap
        self._clock = clock if clock is not None else time.time
        self._breach_count = 0
        self._lock = threading.Lock()

    @property
    def is_enabled(self) -> bool:
        return True

    @property
    def max_cost_usd_per_hour(self) -> Optional[float]:
        return self._max_hour

    @property
    def max_cost_usd_per_day(self) -> Optional[float]:
        return self._max_day

    def _now(self) -> float:
        return float(self._clock())

    def current_totals(self) -> Tuple[float, float, str, str]:
        """Return (hour_usd, day_usd, hour_key, day_key)."""
        hour_key, day_key = _utc_bucket_keys(self._now())
        hour_usd, day_usd = self._store.get_totals(hour_key, day_key)
        return hour_usd, day_usd, hour_key, day_key

    def _raise_if_over(self, hour_usd: float, day_usd: float, last_call_usd: float = 0.0) -> None:
        if self._max_hour is not None and hour_usd > self._max_hour:
            with self._lock:
                self._breach_count += 1
            logger.warning(
                f"fleet_cost_budget_exceeded window=hour "
                f"running_total_usd={hour_usd:.6f} max_cost_usd={self._max_hour:.6f} "
                f"last_call_usd={last_call_usd:.6f}"
            )
            raise FleetCostBudgetExceeded(
                f"fleet LLM cost budget exceeded (hour): "
                f"running_total_usd={hour_usd:.6f} max_cost_usd={self._max_hour:.6f}"
            )
        if self._max_day is not None and day_usd > self._max_day:
            with self._lock:
                self._breach_count += 1
            logger.warning(
                f"fleet_cost_budget_exceeded window=day "
                f"running_total_usd={day_usd:.6f} max_cost_usd={self._max_day:.6f} "
                f"last_call_usd={last_call_usd:.6f}"
            )
            raise FleetCostBudgetExceeded(
                f"fleet LLM cost budget exceeded (day): "
                f"running_total_usd={day_usd:.6f} max_cost_usd={self._max_day:.6f}"
            )

    def check_admit(self) -> None:
        """Raise if fleet already exhausted (no LLM call should start)."""
        hour_usd, day_usd, _, _ = self.current_totals()
        # Admit uses >= so a window sitting exactly at the cap blocks further spend.
        if self._max_hour is not None and hour_usd >= self._max_hour:
            with self._lock:
                self._breach_count += 1
            logger.warning(
                f"fleet_cost_budget_admit_denied window=hour "
                f"running_total_usd={hour_usd:.6f} max_cost_usd={self._max_hour:.6f}"
            )
            raise FleetCostBudgetExceeded(
                f"fleet LLM cost budget exhausted (hour): "
                f"running_total_usd={hour_usd:.6f} max_cost_usd={self._max_hour:.6f}"
            )
        if self._max_day is not None and day_usd >= self._max_day:
            with self._lock:
                self._breach_count += 1
            logger.warning(
                f"fleet_cost_budget_admit_denied window=day "
                f"running_total_usd={day_usd:.6f} max_cost_usd={self._max_day:.6f}"
            )
            raise FleetCostBudgetExceeded(
                f"fleet LLM cost budget exhausted (day): "
                f"running_total_usd={day_usd:.6f} max_cost_usd={self._max_day:.6f}"
            )

    def record_cost(self, cost_usd: float) -> None:
        """Add realised call cost; raise when cumulative exceeds a window cap."""
        if not isinstance(cost_usd, (int, float)):
            raise ValidationError(
                f"FleetCostBudget.record_cost requires numeric cost_usd; got {type(cost_usd).__name__}"
            )
        if float(cost_usd) < 0.0:
            raise ValidationError(
                f"FleetCostBudget.record_cost rejects negative cost_usd; got {cost_usd}"
            )
        hour_key, day_key = _utc_bucket_keys(self._now())
        hour_usd, day_usd = self._store.add(float(cost_usd), hour_key, day_key)
        self._raise_if_over(hour_usd, day_usd, last_call_usd=float(cost_usd))

    def snapshot(self) -> dict:
        hour_usd, day_usd, hour_key, day_key = self.current_totals()
        with self._lock:
            breaches = int(self._breach_count)
        return {
            'max_cost_usd_per_hour': self._max_hour,
            'max_cost_usd_per_day': self._max_day,
            'hour_key': hour_key,
            'day_key': day_key,
            'hour_running_total_usd': float(hour_usd),
            'day_running_total_usd': float(day_usd),
            'breach_count': breaches,
        }


__all__ = [
    'FleetCostBudget',
    'FleetCostStore',
    'InMemoryFleetCostStore',
    'RedisFleetCostStore',
    'build_fleet_cost_store',
]
