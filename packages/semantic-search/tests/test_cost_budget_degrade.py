"""Query + fleet cost budgets degrade to LLMError -> regex path, not reject."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from semantic_search.contracts import Entity, IntentSlice
from semantic_search.core.exceptions import (
    CostBudgetExceeded,
    FleetCostBudgetExceeded,
    LLMError,
    QueryCostBudgetExceeded,
)
from semantic_search.cost.fleet_budget import FleetCostBudget, InMemoryFleetCostStore
from semantic_search.cost.query_budget import NoOpQueryCostBudget, QueryCostBudget
from semantic_search.cost.request_cost_gate import RequestCostGate
from semantic_search.qi.engine import QIEngine


def test_query_check_admit_blocks_when_exhausted() -> None:
    b = QueryCostBudget(max_cost_usd_per_query=0.01, request_id='r1')
    b.record_cost(0.01)
    with pytest.raises(QueryCostBudgetExceeded):
        b.check_admit()


def test_fleet_check_admit_and_record() -> None:
    store = InMemoryFleetCostStore()
    fleet = FleetCostBudget(store=store, max_cost_usd_per_day=0.02, max_cost_usd_per_hour=None)
    fleet.record_cost(0.01)
    fleet.check_admit()
    fleet.record_cost(0.01)
    with pytest.raises(FleetCostBudgetExceeded):
        fleet.check_admit()


def test_request_cost_gate_records_query_and_fleet() -> None:
    query = QueryCostBudget(max_cost_usd_per_query=1.0, request_id='r')
    store = InMemoryFleetCostStore()
    fleet = FleetCostBudget(store=store, max_cost_usd_per_day=0.05)
    gate = RequestCostGate(query, fleet)
    gate.check_admit()
    gate('model-x', 0.03, {'prompt_tokens': 1})
    assert gate.running_total_usd == pytest.approx(0.03)
    assert gate.snapshot()['fleet']['day_running_total_usd'] == pytest.approx(0.03)


def test_request_cost_gate_fleet_admit_denied() -> None:
    query = NoOpQueryCostBudget(request_id='r')
    store = InMemoryFleetCostStore()
    fleet = FleetCostBudget(store=store, max_cost_usd_per_hour=0.01)
    fleet.record_cost(0.01)
    gate = RequestCostGate(query, fleet)
    with pytest.raises(FleetCostBudgetExceeded):
        gate.check_admit()


@pytest.mark.asyncio
async def test_call_structured_admit_denied_is_llm_error_not_provider_call() -> None:
    from pydantic import BaseModel

    from semantic_search.core.llm_client import (
        LLMCallRouter,
        reset_request_cost_observer,
        set_request_cost_observer,
    )

    class _Tiny(BaseModel):
        ok: bool = True

    store = InMemoryFleetCostStore()
    fleet = FleetCostBudget(store=store, max_cost_usd_per_day=0.001)
    fleet.record_cost(0.001)
    gate = RequestCostGate(NoOpQueryCostBudget(), fleet)
    token = set_request_cost_observer(gate)
    try:
        router = object.__new__(LLMCallRouter)
        router._enforce_ingress_sanitizer = MagicMock()
        router._provider = MagicMock()
        router._provider.get_fallback_chain.return_value = ['m']
        router._circuit_breaker = MagicMock()
        router._circuit_breaker.allow_request.return_value = True
        router._structural_gate = None
        with pytest.raises(LLMError) as ei:
            await LLMCallRouter.call_structured(
                router,
                task_type='l0_entity_extraction',
                prompt_tag='t',
                system_prompt='s',
                user_prompt='u',
                response_schema=_Tiny,
            )
        assert 'llm_cost_budget_exhausted' in str(ei.value)
        router._provider.get_client_for_model.assert_not_called()
    finally:
        reset_request_cost_observer(token)


@pytest.mark.asyncio
async def test_l0_budget_llm_error_runs_regex_fallback() -> None:
    class _BudgetDeniedL0:
        async def classify_async_priced(self, _text: str):
            raise LLMError('llm_cost_budget_exhausted reason=FleetCostBudgetExceeded')

    class _RegexL0:
        def __init__(self) -> None:
            self._config = MagicMock(enabled=True, fallback_only_when_llm_unavailable=True)
            self.classify_async = AsyncMock(
                return_value=IntentSlice(
                    query_type='hybrid',
                    entities=[Entity(name='tld', value=['io'], confidence=0.9, source='L0_regex', chip_kind='hard')],
                    confidence=0.9,
                    raw_text='.io under 50',
                ),
            )

    engine = object.__new__(QIEngine)
    engine._entity_extractor = _BudgetDeniedL0()
    engine._regex_entity_extractor = _RegexL0()
    slots = MagicMock()
    slots.soft_slot_set = frozenset()
    slots.hard_entity_set = frozenset({'tld'})
    engine._config = MagicMock(entity_slots=slots)
    engine._slot_sets = QIEngine._slot_sets.__get__(engine, QIEngine)
    engine._should_run_regex_l0_fallback = QIEngine._should_run_regex_l0_fallback.__get__(engine, QIEngine)
    engine._call_l0_llm_priced = QIEngine._call_l0_llm_priced.__get__(engine, QIEngine)
    engine._run_l0_extractors = QIEngine._run_l0_extractors.__get__(engine, QIEngine)

    hard, soft, cost, keywords = await engine._run_l0_extractors(
        '.io under 50', 'req-budget',
    )
    assert cost == 0.0
    assert soft == []
    assert keywords == []
    assert any(e.name == 'tld' and e.source == 'L0_regex' for e in hard)
    engine._regex_entity_extractor.classify_async.assert_awaited()


def test_budget_exceptions_are_cost_policy() -> None:
    assert issubclass(QueryCostBudgetExceeded, CostBudgetExceeded)
    assert issubclass(FleetCostBudgetExceeded, CostBudgetExceeded)
