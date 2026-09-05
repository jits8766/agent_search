"""decision_cost_usd must include L0 extract + L2 classify (and budget stamp)."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple

import pytest

from llm_core.pricing import compute_call_cost_usd
from semantic_search.app import _add_llm_cost_usd
from semantic_search.config.loader import load_config
from semantic_search.config.models import AgentSearchConfig
from semantic_search.contracts import IntentSlice, QueryIntent, RankedResults
from semantic_search.cost.query_budget import NoOpQueryCostBudget
from semantic_search.orchestrator import SearchOrchestrator, _REQUEST_COST_GATE
from semantic_search.qi.l0_llm_filter_extractor import (
    L0LLMFilterExtractor,
    _L0BatchResponse,
    _L0Filter,
    _L0ResultEntry,
)


class _FakeRouter:
    """Minimal call_router that returns fixed usage for cost math."""

    def __init__(self, model: str, usage: Dict[str, Any], filters: Optional[List[Any]] = None) -> None:
        self._model = model
        self._usage = usage
        self._filters = filters or []

    async def call_structured(self, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        filters = [
            _L0Filter(param='tld', value=['com']),
        ]
        resp = _L0BatchResponse(results=[_L0ResultEntry(idx=1, filters=filters)])
        return resp, {'model': self._model, 'usage': dict(self._usage)}


def _l0_cfg_kwargs() -> Dict[str, Any]:
    qi = AgentSearchConfig.from_dict(load_config()).qi
    cfg = qi.l0_llm_entity
    return dict(
        task_type=cfg.task_type,
        entity_slots=qi.entity_slots,
        source_tag=cfg.source_tag,
        max_entities=cfg.max_entities,
        confidence=cfg.confidence,
        enabled=cfg.enabled,
        prompt_tag=cfg.prompt_tag,
        keyword_min_probability=cfg.keyword_min_probability,
        combined_prompt_tag=cfg.combined_prompt_tag,
    )


@pytest.mark.asyncio
async def test_extract_priced_returns_usage_based_cost() -> None:
    model = 'default'
    usage = {'prompt_tokens': 1_000_000, 'completion_tokens': 1_000_000}
    expected = compute_call_cost_usd(model, usage)
    extractor = L0LLMFilterExtractor(
        call_router=_FakeRouter(model, usage),
        **_l0_cfg_kwargs(),
    )
    _identified, cost, _keywords = await extractor.extract_priced('cheap .com domains')
    # Cost is computed from provider usage even when post-filters drop chips.
    assert cost == pytest.approx(expected, rel=1e-9)
    assert cost > 0.0


@pytest.mark.asyncio
async def test_classify_async_priced_matches_extract_cost() -> None:
    model = 'default'
    usage = {'prompt_tokens': 500_000, 'completion_tokens': 0}
    expected = compute_call_cost_usd(model, usage)
    extractor = L0LLMFilterExtractor(
        call_router=_FakeRouter(model, usage),
        **_l0_cfg_kwargs(),
    )
    slice_, cost = await extractor.classify_async_priced('cheap .com domains')
    assert slice_ is not None
    assert cost == pytest.approx(expected, rel=1e-9)


def test_stamp_results_llm_cost_overwrites_intent_field() -> None:
    intent = QueryIntent(
        request_id='r1',
        raw_query='q',
        normalized_query='q',
        query_type='hybrid',
        confidence=0.9,
        decision_tier='L2_llm',
        slices=[IntentSlice(query_type='hybrid', entities=[], confidence=0.9, raw_text='q')],
        decision_cost_usd=0.001,  # L2-only undercount
    )
    results = RankedResults(
        request_id='r1',
        items=[],
        total_candidates=0,
        fusion_latency_ms=0.0,
        query_intent=intent,
    )
    budget = NoOpQueryCostBudget(request_id='r1')
    budget.record_cost(0.004)  # L0
    budget.record_cost(0.001)  # L2
    # Bind minimal orchestrator shell for the helper (no __init__).
    orch = object.__new__(SearchOrchestrator)
    stamped = SearchOrchestrator._stamp_results_llm_cost(orch, results, budget)
    assert stamped.query_intent is not None
    assert stamped.query_intent.decision_cost_usd == pytest.approx(0.005, rel=1e-9)
    # Original results object left unchanged (replace returns new).
    assert results.query_intent.decision_cost_usd == pytest.approx(0.001, rel=1e-9)


def test_request_cost_gate_contextvar_isolates_concurrent_tasks() -> None:
    """Each asyncio task sees its own ContextVar cost gate."""
    orch = object.__new__(SearchOrchestrator)
    orch._cost_budget_factory = None
    orch._fleet_cost_budget = None
    orch._last_cost_budget = None
    orch._accumulated_llm_cost_usd = {}
    orch._active_cost_gate_by_rid = {}

    async def _phase(rid: str, amount: float) -> float:
        gate = orch._bind_request_cost_budget(rid)
        gate('default', amount, {})
        await asyncio.sleep(0)
        seen = orch.last_cost_budget
        assert seen is gate
        orch._finalize_request_cost_phase(rid, gate)
        return orch.request_llm_cost_usd(rid)

    async def _run() -> None:
        a, b = await asyncio.gather(_phase('ra', 0.01), _phase('rb', 0.02))
        assert a == pytest.approx(0.01, rel=1e-9)
        assert b == pytest.approx(0.02, rel=1e-9)
        assert orch.request_llm_cost_usd('ra') == pytest.approx(0.01, rel=1e-9)
        assert orch.request_llm_cost_usd('rb') == pytest.approx(0.02, rel=1e-9)
        assert _REQUEST_COST_GATE.get() is None

    asyncio.run(_run())


def test_add_llm_cost_usd_merges_analytics_phase() -> None:
    qi = {'decision_cost_usd': 0.003}
    _add_llm_cost_usd(qi, 0.004)
    assert qi['decision_cost_usd'] == pytest.approx(0.007, rel=1e-9)
    _add_llm_cost_usd(qi, 0.0)
    assert qi['decision_cost_usd'] == pytest.approx(0.007, rel=1e-9)
