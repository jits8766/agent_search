"""Retrieval fallback planner (`DegradationPlanner`).

Given the current backend health, compute the set of retrieval backends the
orchestrator should attempt plus operator-facing notes. The planner is
pure: it consumes a `BackendHealthRegistry` snapshot and returns a `DegradationPlan`.
This separation keeps the orchestrator's wiring testable in isolation.

Fallback chains are config-driven (`DegradationConfig.fallback_chains`) so adding
a new retrieval backend requires only a YAML edit, not a code change.
"""
from typing import List, Optional, Set

from semantic_search.config.models import DegradationConfig
from semantic_search.contracts import AnalyticsSubstratePlan, DEGRADATION_RETRIEVAL_BACKENDS as _RETRIEVAL_BACKENDS, DegradationPlan
from semantic_search.core.logging_utils import get_logger
from semantic_search.resilience.health import BackendHealthRegistry

logger = get_logger(__name__)

# Terminal fallback markers: cache_only (serve cache/empty), none (no replacement)
_TERMINAL_FALLBACKS = frozenset({'cache_only', 'none'})

# Re-export under the legacy name so consumers that imported from this module
# continue to work without churn (`from semantic_search.resilience.degradation import DegradationPlan`).
__all__ = ['DegradationPlan', 'DegradationPlanner']


class DegradationPlanner:
    """Select retrieval mix given backend health (config-driven fallback chains)."""

    def __init__(self, config: DegradationConfig, health: BackendHealthRegistry):
        self._config = config
        self._health = health

    def plan(self, requested_backends: List[str]) -> DegradationPlan:
        """Compute degradation plan: active backends + drop reasons."""
        if not self._config.enabled:
            return DegradationPlan(active_backends=list(requested_backends))
        active: List[str] = []
        dropped: List[str] = []
        notes: List[str] = []
        for backend in requested_backends:
            if self._health.is_healthy(backend):
                active.append(backend)
                continue
            # Half-open probe: if recovery_probe_seconds elapsed, route one request
            if self._health.is_probe_eligible(backend):
                active.append(backend)
                notes.append(f"{backend}_unhealthy_half_open_probe")
                logger.info(f"degradation_half_open_probe backend={backend}")
                continue
            dropped.append(backend)
            replacement = self._first_healthy_fallback(backend)
            if replacement is None:
                notes.append(f"{backend}_unhealthy_no_replacement")
                continue
            if replacement in _TERMINAL_FALLBACKS:
                notes.append(f"{backend}_unhealthy_{replacement}")
                continue
            if replacement in _RETRIEVAL_BACKENDS and replacement not in active and replacement not in dropped:
                active.append(replacement)
                notes.append(f"{backend}_unhealthy_falls_back_to_{replacement}")
        seen: Set[str] = set()
        deduped: List[str] = []
        for b in active:
            if b in seen:
                continue
            seen.add(b)
            deduped.append(b)
        active = deduped
        if not active:
            mode = 'cache_only' if self._config.allow_empty_when_all_unhealthy else 'degraded'
        elif dropped:
            mode = 'degraded'
        else:
            mode = 'normal'
        if dropped:
            logger.warning(f"degradation_plan mode={mode} active={active} dropped={dropped} notes={notes}")
        return DegradationPlan(active_backends=active, dropped_backends=dropped, mode=mode, notes=notes)

    def plan_analytics_substrate(self, *, analytics_router_enabled: bool) -> AnalyticsSubstratePlan:
        """Summarize analytics substrate vs shared health for ops dashboards.
        :param analytics_router_enabled: bool - True when ``AnalyticsRouter.enabled`` (CH credentials + config)
        :return: AnalyticsSubstratePlan - Typed mode + cross-subsystem notes for ops dashboards
        """
        if not analytics_router_enabled:
            return AnalyticsSubstratePlan(mode='disabled', notes=['analytics_router_disabled_or_ch_unavailable'])
        notes: List[str] = []
        ch_ok = self._health.is_healthy('clickhouse')
        vec_ok = self._health.is_healthy('vector')
        if ch_ok:
            if not vec_ok:
                notes.append('vector_unhealthy_retrieval_may_be_degraded')
            return AnalyticsSubstratePlan(mode='clickhouse_primary', notes=notes)
        notes.append('clickhouse_unhealthy_analytics_disabled')
        if not vec_ok:
            notes.append('vector_unhealthy_sql_only_retrieval_possible')
        return AnalyticsSubstratePlan(mode='disabled', notes=notes)

    def _first_healthy_fallback(self, backend: str) -> Optional[str]:
        """Return the first healthy fallback backend (or `cache_only`/`none` marker, or None).

        Reads from `config.fallback_chains[backend]`. Backends without a configured
        chain return None so the planner can record an explicit reason.
        """
        chain: List[str] = self._config.fallback_chains.get(backend, [])
        for option in chain:
            if option in _TERMINAL_FALLBACKS:
                return option
            if option in _RETRIEVAL_BACKENDS and self._health.is_healthy(option):
                return option
        return None

    def llm_available(self) -> bool:
        """Convenience: True iff the LLM backend is currently healthy."""
        return self._health.is_healthy('llm')
