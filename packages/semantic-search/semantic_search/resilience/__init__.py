"""Resilience & graceful degradation.
Components:
- `circuit_breaker`: closed/open/half-open state machine for LLM calls.
- `health`: BackendHealthRegistry tracking vector/structured/sql/LLM availability.
- `degradation`: DegradationPlanner mapping unhealthy backends to fallback retrieval mixes.
"""
from semantic_search.resilience.circuit_breaker import CircuitBreaker, CircuitOpenError
from semantic_search.resilience.degradation import DegradationPlan, DegradationPlanner
from semantic_search.resilience.health import BackendHealthRegistry

__all__ = ['BackendHealthRegistry', 'CircuitBreaker', 'CircuitOpenError', 'DegradationPlan', 'DegradationPlanner']
