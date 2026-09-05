"""PercentileResolver — live percentile-based value resolution.

Replaces hardcoded "cheap" / "expiring" thresholds with percentiles computed
off the structured-index payload columns:

  * "cheap" → ``inventory.cheap_percentile`` of the live ``price`` column
  * "expiring" → ``inventory.expiring_percentile`` of the live remaining-time
    column (``ends_at`` - now)

Percentile computation is gated by ``inventory.min_sample_size`` (so a tiny
freshly-rebuilt index cannot produce a misleading p25). When the gate fails
the resolver returns the configured prior — never silently returns None or
0, never crashes the call site.

Cached for ``inventory.refresh_interval_seconds`` so a busy QI cascade does
not pay the percentile-scan cost on every classification. The snapshot-version
hook (``invalidate_on_bump``) hard-resets the cache so a stream event
guarantees the next resolution sees fresh percentiles.
"""
import threading
import time
from typing import Optional

from semantic_search.config.models import InventoryConfig
from semantic_search.core.exceptions import ValidationError
from semantic_search.core.logging_utils import get_logger
from semantic_search.core.validation import safe_float
from semantic_search.retrieval.structured_retriever import StructuredIndex

logger = get_logger(__name__)


def _percentile(sorted_values, fraction: float) -> float:
    """Nearest-rank percentile of sorted sequence (matches measurement evaluator)."""
    if not sorted_values:
        raise ValidationError("_percentile requires a non-empty sequence")
    n = len(sorted_values)
    # idx = ceil(fraction * n) - 1, clamped to [0, n-1]
    rank = int((fraction * n) + 0.999999)
    idx = max(0, min(n - 1, rank - 1))
    return float(sorted_values[idx])


class PercentileResolver:
    """Resolve cheap/expiring percentiles from structured index (cached, cache-busted on event)."""

    def __init__(self, config: InventoryConfig, index: StructuredIndex):
        if config is None:
            raise ValidationError("PercentileResolver requires a non-null InventoryConfig")
        if index is None:
            raise ValidationError("PercentileResolver requires a non-null StructuredIndex")
        self._config = config
        self._index = index
        self._lock = threading.Lock()
        self._cached_cheap_max: Optional[float] = None
        self._cached_expiring_max_seconds: Optional[float] = None
        self._cached_at: float = 0.0
        self._cache_sample_size: int = 0
        # Per-TLD market price baselines + '_global' fallback (separate TTL cache).
        self._cached_market: Optional[dict] = None
        self._cached_market_at: float = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self._config.enabled)

    def invalidate_on_bump(self, snapshot_version: int) -> None:
        """Drop cache on snapshot bump (hook contract)."""
        with self._lock:
            self._cached_cheap_max = None
            self._cached_expiring_max_seconds = None
            self._cached_at = 0.0
            self._cache_sample_size = 0
        logger.info(f"inventory_percentile_cache_invalidated snapshot_version={snapshot_version}")

    def cheap_price_max(self) -> float:
        """Resolve the "cheap" price ceiling — p_cheap of live prices, or the prior.

        "cheap → p25 of live prices". Returns the configured
        prior when disabled, when the index is empty, or when the sample-size
        gate fails.
        """
        if not self._config.enabled:
            return float(self._config.cheap_price_max_prior)
        self._refresh_if_stale()
        with self._lock:
            cached = self._cached_cheap_max
        if cached is None:
            return float(self._config.cheap_price_max_prior)
        return float(cached)

    def expiring_seconds_max(self) -> int:
        """Resolve the "expiring" remaining-time ceiling — p_expiring of remaining
        seconds, or the prior.
        """
        if not self._config.enabled:
            return int(self._config.expiring_seconds_max_prior)
        self._refresh_if_stale()
        with self._lock:
            cached = self._cached_expiring_max_seconds
        if cached is None:
            return int(self._config.expiring_seconds_max_prior)
        return int(cached)

    def cache_sample_size(self) -> int:
        """Sample size of the most recent successful percentile compute (0 = never computed)."""
        with self._lock:
            return self._cache_sample_size

    def invalidate_market_on_bump(self, snapshot_version: int) -> None:
        """Drop the cached per-TLD market baselines after a stream event."""
        with self._lock:
            self._cached_market = None
            self._cached_market_at = 0.0
        logger.info(f"inventory_market_baseline_cache_invalidated snapshot_version={snapshot_version}")

    def market_price_baselines(self) -> dict:
        """Return per-TLD ``market_percentile`` price baselines + a '_global' key.

        Scans live ``price`` grouped by ``tld`` from the structured index, computes
        the configured ``market_percentile`` per group (and globally), and caches
        for ``refresh_interval_seconds``. Returns ``{}`` when disabled or below the
        sample floor — the caller then treats "below market" as a no-op rather than
        excluding everything.

        :return: Dict[str, float] - {tld: baseline_price, '_global': baseline_price}
        """
        if not self._config.enabled:
            return {}
        now = time.time()
        with self._lock:
            fresh = self._cached_market is not None and (now - self._cached_market_at) < float(self._config.refresh_interval_seconds)
            if fresh:
                return dict(self._cached_market)
            by_tld: dict = {}
            all_prices: list = []
            for payload in self._index.iter_payloads():
                if not isinstance(payload, dict):
                    continue
                price = safe_float(payload.get('price'), -1.0)
                if price <= 0.0:
                    continue
                all_prices.append(price)
                tld = str(payload.get('tld', '')).lower().lstrip('.')
                by_tld.setdefault(tld, []).append(price)
            if len(all_prices) < int(self._config.min_sample_size):
                self._cached_market = {}
                self._cached_market_at = now
                logger.info(f"inventory_market_baseline_below_sample_floor sample_size={len(all_prices)} min_required={self._config.min_sample_size}")
                return {}
            pct = float(self._config.market_percentile)
            baselines: dict = {}
            for tld, prices in by_tld.items():
                prices.sort()
                baselines[tld] = _percentile(prices, pct)
            all_prices.sort()
            baselines['_global'] = _percentile(all_prices, pct)
            self._cached_market = baselines
            self._cached_market_at = now
            logger.info(f"inventory_market_baseline_recomputed tlds={len(by_tld)} percentile={pct} global={baselines['_global']}")
            return dict(baselines)

    def _refresh_if_stale(self) -> None:
        """Recompute percentiles if the cache is older than the configured TTL.

        Pure-function on payload data — no external I/O — so we can hold the
        lock for the entire compute. If the sample is below the configured
        floor, leave the cache values as ``None`` so the caller falls back to
        the prior.
        """
        now = time.time()
        with self._lock:
            stale = (self._cached_at == 0.0) or ((now - self._cached_at) >= float(self._config.refresh_interval_seconds))
            if not stale:
                return
            prices = []
            remaining = []
            for payload in self._index.iter_payloads():
                if not isinstance(payload, dict):
                    continue
                price = safe_float(payload.get('price'), -1.0)
                if price >= 0.0:
                    prices.append(price)
                ends_at = safe_float(payload.get('ends_at'), -1.0)
                if ends_at > 0.0:
                    delta = ends_at - now
                    if delta >= 0.0:
                        remaining.append(delta)
            sample_size = max(len(prices), len(remaining))
            if sample_size < int(self._config.min_sample_size):
                self._cached_cheap_max = None
                self._cached_expiring_max_seconds = None
                self._cached_at = now
                self._cache_sample_size = sample_size
                if sample_size == 0:
                    logger.info(f"inventory_percentile_below_sample_floor sample_size={sample_size} min_required={self._config.min_sample_size} fallback=prior")
                else:
                    logger.warning(f"inventory_percentile_below_sample_floor sample_size={sample_size} min_required={self._config.min_sample_size} fallback=prior")
                return
            if prices:
                prices.sort()
                self._cached_cheap_max = _percentile(prices, float(self._config.cheap_percentile))
            else:
                self._cached_cheap_max = None
            if remaining:
                remaining.sort()
                self._cached_expiring_max_seconds = _percentile(remaining, float(self._config.expiring_percentile))
            else:
                self._cached_expiring_max_seconds = None
            self._cached_at = now
            self._cache_sample_size = sample_size
            logger.info(
                f"inventory_percentile_recomputed sample_size={sample_size} "
                f"cheap_p={self._config.cheap_percentile} cheap_max={self._cached_cheap_max} "
                f"expiring_p={self._config.expiring_percentile} expiring_max_seconds={self._cached_expiring_max_seconds}"
            )


__all__ = ['PercentileResolver']
