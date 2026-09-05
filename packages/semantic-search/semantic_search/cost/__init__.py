"""Per-query and fleet LLM cost-budget primitives.

- ``QueryCostBudget`` — per-request USD accumulator
- ``FleetCostBudget`` — cross-request hour/day USD accumulator
- ``RequestCostGate`` — composite observer bound per search/analytics call
"""
from semantic_search.cost.fleet_budget import (
    FleetCostBudget,
    InMemoryFleetCostStore,
    RedisFleetCostStore,
    build_fleet_cost_store,
)
from semantic_search.cost.query_budget import NoOpQueryCostBudget, QueryCostBudget
from semantic_search.cost.request_cost_gate import RequestCostGate

__all__ = [
    'QueryCostBudget',
    'NoOpQueryCostBudget',
    'FleetCostBudget',
    'InMemoryFleetCostStore',
    'RedisFleetCostStore',
    'build_fleet_cost_store',
    'RequestCostGate',
]
