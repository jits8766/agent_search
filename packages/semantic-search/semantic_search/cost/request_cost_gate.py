"""Per-request cost gate: query budget + optional fleet budget.

Bound as the LLM ``cost_observer`` for one search/analytics/classify call.
``check_admit`` runs before provider I/O; ``__call__`` records after success.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

from semantic_search.cost.fleet_budget import FleetCostBudget
from semantic_search.cost.query_budget import NoOpQueryCostBudget, QueryCostBudget


class RequestCostGate:
    """Composite observer: per-query ceiling + shared fleet ceiling."""

    def __init__(
        self,
        query_budget: Union[QueryCostBudget, NoOpQueryCostBudget],
        fleet_budget: Optional[FleetCostBudget] = None,
    ):
        self._query = query_budget
        self._fleet = fleet_budget

    @property
    def query_budget(self) -> Union[QueryCostBudget, NoOpQueryCostBudget]:
        return self._query

    @property
    def fleet_budget(self) -> Optional[FleetCostBudget]:
        return self._fleet

    @property
    def running_total_usd(self) -> float:
        """Per-request running total (used to stamp ``decision_cost_usd``)."""
        return float(self._query.running_total_usd)

    @property
    def call_count(self) -> int:
        return int(self._query.call_count)

    @property
    def is_enabled(self) -> bool:
        query_on = bool(getattr(self._query, 'is_enabled', False))
        fleet_on = self._fleet is not None and bool(getattr(self._fleet, 'is_enabled', False))
        return query_on or fleet_on

    def check_admit(self) -> None:
        """Block new LLM calls when query or fleet budget is already exhausted."""
        check = getattr(self._query, 'check_admit', None)
        if callable(check):
            check()
        if self._fleet is not None:
            self._fleet.check_admit()

    def __call__(self, model: str, cost_usd: float, usage: Dict[str, Any]) -> None:
        """Record realised call cost into query then fleet (query first)."""
        del model, usage  # signature matches CostObserver
        self._query.record_cost(cost_usd)
        if self._fleet is not None:
            self._fleet.record_cost(cost_usd)

    def snapshot(self) -> dict:
        snap = dict(self._query.snapshot())
        if self._fleet is not None:
            snap['fleet'] = self._fleet.snapshot()
        return snap


__all__ = ['RequestCostGate']
