"""Search SLA, get_timeout_fallback, and analytics timeout wiring.

Covers paths previously untested:
- app.search outer asyncio.wait_for SLA breach -> explore fallback body
- answer_mode mapping from L1 preview on SLA breach
- Orchestrator.get_timeout_fallback success / timeout / cache
- Orchestrator.analytics per-call timeout -> failure_mode='timeout'
"""
from __future__ import annotations

import asyncio
from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from semantic_search.contracts import RankedItem, RankedResults
from semantic_search.nl_to_sql.contracts import AnalyticsResult


def _item(item_id: str = 'fallback.example', score: float = 0.8) -> RankedItem:
    return RankedItem(
        item_id=item_id,
        fused_score=score,
        contributing_sources=['vector'],
        payload={'domain_name': item_id, 'tld': 'com'},
    )


def _ranked(items: Optional[List[RankedItem]] = None, *, request_id: str = 'rid-1') -> RankedResults:
    items = items if items is not None else [_item()]
    return RankedResults(
        request_id=request_id,
        items=items,
        total_candidates=len(items),
        fusion_latency_ms=1.0,
        cache_hit=None,
        failure_mode='explore_fallback_rail',
    )


def _search_cfg(**overrides: Any) -> MagicMock:
    cfg = MagicMock()
    cfg.qie_only_mode = False
    cfg.top_k_cap = 50
    cfg.search_timeout_seconds = 0.05
    cfg.analytics_timeout_seconds = 18.0
    cfg.analytics_total_budget_seconds = 20.0
    cfg.explore_fallback_timeout_seconds = 1.0
    cfg.speculative_analytics_start = False
    cfg.speculative_analytics_l1_confidence_threshold = 0.75
    cfg.analytics_budget_keywords = []
    cfg.result_fields = ['tld']
    cfg.auction_tiebreak = MagicMock(enabled=False)
    cfg.analytics_timeout_hybrid_notice = 'Analytics timeout — hybrid results.'
    cfg.analytics_timeout_explore_notice = 'Analytics timeout — explore results.'
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _search_request() -> MagicMock:
    """Minimal Starlette-like request for direct ``app.search`` unit calls."""
    req = MagicMock()
    req.headers = {}
    return req


def _subsystems(*, search_cfg: MagicMock, orch: MagicMock, l1_preview: Optional[tuple] = None) -> MagicMock:
    from semantic_search.config.loader import load_config
    from semantic_search.config.models import AgentSearchConfig

    sub = MagicMock()
    sub.config.general.search = search_cfg
    sub.config.retrieval.metrics.score_normalization = 'max'
    sub.config.identity = AgentSearchConfig.from_dict(load_config()).identity
    sub.sanitizer = None
    sub.call_router = None
    sub.orchestrator = orch
    sub.qi_engine = MagicMock()
    if l1_preview is None:
        sub.qi_engine.quick_classify = MagicMock(side_effect=RuntimeError('no l1'))
    else:
        sub.qi_engine.quick_classify = MagicMock(return_value=l1_preview)
    return sub


# ── app.search SLA breach ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_sla_breach_returns_explore_fallback_body():
    """Hanging orch.search -> timeout body with failure_mode=timeout + fallback items."""
    from semantic_search import app as app_module

    async def _hang(**_kwargs):
        await asyncio.sleep(10)

    fb = _ranked([_item('sla-fallback.com', 0.9)])
    orch = MagicMock()
    orch.analytics_available = False
    orch.search = AsyncMock(side_effect=_hang)
    orch.get_timeout_fallback = AsyncMock(return_value=fb)

    search_cfg = _search_cfg(search_timeout_seconds=0.05, explore_fallback_timeout_seconds=1.0)
    sub = _subsystems(search_cfg=search_cfg, orch=orch, l1_preview=None)

    with patch.object(app_module, '_require_subsystems', return_value=sub):
        body = await app_module.search(
            request=_search_request(),
            query='brandable domains',
            top_k=5,
            diversity_lambda=0.9,
            relevance_threshold=0.7,
            qie_only_mode=False,
            x_session_id=None,
        )

    assert body['retrieval_metrics']['failure_mode'] == 'timeout'
    assert body['answer_mode'] == 'explore_fallback'
    assert len(body['ranked_results']) == 1
    assert body['ranked_results'][0]['domain_name'] == 'sla-fallback.com'
    assert 'exceeded' in (body.get('guard_notice') or '').lower()
    orch.get_timeout_fallback.assert_awaited()


@pytest.mark.asyncio
async def test_search_sla_breach_hybrid_l1_maps_answer_mode_search():
    """L1=hybrid on SLA breach -> answer_mode='search' (not explore_fallback)."""
    from semantic_search import app as app_module

    async def _hang(**_kwargs):
        await asyncio.sleep(10)

    orch = MagicMock()
    orch.analytics_available = False
    orch.search = AsyncMock(side_effect=_hang)
    orch.get_timeout_fallback = AsyncMock(return_value=_ranked())

    search_cfg = _search_cfg(search_timeout_seconds=0.05)
    sub = _subsystems(search_cfg=search_cfg, orch=orch, l1_preview=('hybrid', 0.95))

    with patch.object(app_module, '_require_subsystems', return_value=sub):
        body = await app_module.search(
            request=_search_request(),
            query='short .com names',
            top_k=5,
            diversity_lambda=0.9,
            relevance_threshold=0.7,
            qie_only_mode=False,
            x_session_id=None,
        )

    assert body['answer_mode'] == 'search'
    assert body['retrieval_metrics']['failure_mode'] == 'timeout'


@pytest.mark.asyncio
async def test_search_sla_breach_explore_fallback_also_times_out():
    """When explore prefetch also times out -> empty ranked_results, failure_mode=timeout."""
    from semantic_search import app as app_module

    async def _hang(**_kwargs):
        await asyncio.sleep(10)

    async def _fb_hang(**_kwargs):
        await asyncio.sleep(10)

    orch = MagicMock()
    orch.analytics_available = False
    orch.search = AsyncMock(side_effect=_hang)
    orch.get_timeout_fallback = AsyncMock(side_effect=_fb_hang)

    search_cfg = _search_cfg(
        search_timeout_seconds=0.05,
        explore_fallback_timeout_seconds=0.05,
    )
    sub = _subsystems(search_cfg=search_cfg, orch=orch, l1_preview=('explore', 0.5))

    with patch.object(app_module, '_require_subsystems', return_value=sub):
        body = await app_module.search(
            request=_search_request(),
            query='trending domains',
            top_k=5,
            diversity_lambda=0.9,
            relevance_threshold=0.7,
            qie_only_mode=False,
            x_session_id=None,
        )

    # L1=explore -> parent tier keeps answer_mode='explore' even when rails time out.
    assert body['answer_mode'] == 'explore'
    assert body['retrieval_metrics']['failure_mode'] == 'timeout'
    assert body['ranked_results'] == []
    assert 'explore fallback also timed out' in (body.get('guard_notice') or '').lower()


# ── Orchestrator.get_timeout_fallback ─────────────────────────────────────────


def _orch_for_timeout_fallback(
    *,
    fallback_timeout: float = 1.0,
    cache_ttl: float = 300.0,
    semantic_prefer_min: int = 0,
    semantic_prefer_score: float = 0.45,
    explore_enabled: bool = True,
    rail_first: bool = True,
    qdrant_rails_when_ch_empty: bool = True,
    qdrant_rails_timeout_seconds: float = 1.0,
) -> Any:
    from semantic_search.orchestrator import SearchOrchestrator

    cfg = MagicMock()
    cfg.general.search.explore_fallback_timeout_seconds = fallback_timeout
    cfg.general.search.explore_fallback_cache_ttl_seconds = cache_ttl
    cfg.general.search.explore_fallback_semantic_prefer_min_results = semantic_prefer_min
    cfg.general.search.explore_fallback_semantic_prefer_min_score = semantic_prefer_score
    cfg.general.search.timeout_fallback.rail_first = rail_first
    cfg.general.search.timeout_fallback.qdrant_rails_when_ch_empty = qdrant_rails_when_ch_empty
    cfg.general.search.timeout_fallback.qdrant_rails_timeout_seconds = qdrant_rails_timeout_seconds
    cfg.general.search.timeout_fallback.consume_search_explore_prewarm = False
    cfg.general.search.timeout_fallback.search_explore_prewarm_wait_seconds = 0.05
    cfg.explore.zero_result_guard.semantic_fallback_top_k = 5
    cfg.explore.zero_result_guard.rrf_k = 60
    cfg.explore.zero_result_guard.explore_fallback_max_per_rail = 3

    orch = SearchOrchestrator.__new__(SearchOrchestrator)
    orch._config = cfg
    orch._fb_result_cache = None
    orch._explore_prewarm_by_rid = {}
    orch._explore_prewarm_gate_by_rid = {}
    orch._explore_composer = MagicMock()
    orch._explore_composer.enabled = explore_enabled
    orch._vector = MagicMock()
    # Default: no Qdrant rail ladder (tests that need A set AsyncMock explicitly).
    orch._vector.retrieve_filter_only_rails = None
    orch._health = MagicMock()
    return orch


@pytest.mark.asyncio
async def test_get_timeout_fallback_fuses_explore_and_semantic():
    orch = _orch_for_timeout_fallback(semantic_prefer_min=0)
    explore_ranked = _ranked([_item('rail.com', 0.7)], request_id='rid-fb')
    orch._explore_composer.compose_fallback = AsyncMock(return_value=(explore_ranked, MagicMock()))
    orch._quick_semantic_retrieve = AsyncMock(return_value=[_item('sem.com', 0.6)])

    out = await orch.get_timeout_fallback(request_id='rid-fb', query='domains', top_k=5)
    assert out.failure_mode == 'explore_fallback_rail'
    assert len(out.items) >= 1
    ids = {i.item_id for i in out.items}
    assert 'rail.com' in ids or 'sem.com' in ids


@pytest.mark.asyncio
async def test_get_timeout_fallback_consumes_search_explore_prewarm():
    """Timeout path uses search()-registered rail task; skips second compose_fallback."""
    orch = _orch_for_timeout_fallback(semantic_prefer_min=0)
    orch._config.general.search.timeout_fallback.consume_search_explore_prewarm = True
    orch._config.general.search.timeout_fallback.search_explore_prewarm_wait_seconds = 1.0
    prewarm_ranked = _ranked([_item('prewarm.com', 0.9)], request_id='rid-pw')

    async def _prewarm():
        return prewarm_ranked

    task = asyncio.create_task(_prewarm())
    orch._register_explore_prewarm('rid-pw', task)
    orch._explore_composer.compose_fallback = AsyncMock(
        side_effect=AssertionError('compose_fallback must not run when prewarm hits'),
    )
    orch._quick_semantic_retrieve = AsyncMock(return_value=[_item('sem.com', 0.5)])

    out = await orch.get_timeout_fallback(request_id='rid-pw', query='domains', top_k=5)
    assert out.failure_mode == 'explore_fallback_rail'
    assert any(i.item_id == 'prewarm.com' for i in out.items)
    orch._explore_composer.compose_fallback.assert_not_called()


@pytest.mark.asyncio
async def test_get_timeout_fallback_times_out_returns_empty():
    orch = _orch_for_timeout_fallback(fallback_timeout=0.05)

    async def _slow(**_k):
        await asyncio.sleep(2)
        return (_ranked(), MagicMock())

    orch._explore_composer.compose_fallback = AsyncMock(side_effect=_slow)
    orch._quick_semantic_retrieve = AsyncMock(side_effect=_slow)

    out = await orch.get_timeout_fallback(request_id='rid-to', query='slow', top_k=5)
    assert out.items == []
    assert out.failure_mode == 'explore_fallback_rail'
    assert out.total_candidates == 0


@pytest.mark.asyncio
async def test_get_timeout_fallback_cache_hit():
    orch = _orch_for_timeout_fallback(cache_ttl=300.0)
    cached = _ranked([_item('cached.com')], request_id='rid-cache')
    import time as _time
    orch._fb_result_cache = (cached, _time.monotonic())
    orch._explore_composer.compose_fallback = AsyncMock(
        side_effect=AssertionError('compose must not run on cache hit'),
    )
    orch._quick_semantic_retrieve = AsyncMock(
        side_effect=AssertionError('semantic must not run on cache hit'),
    )

    out = await orch.get_timeout_fallback(request_id='rid-cache', query='q', top_k=5)
    assert out is cached
    assert out.items[0].item_id == 'cached.com'


@pytest.mark.asyncio
async def test_get_timeout_fallback_semantic_prefer_skips_rrf():
    """High-score semantic leg returned directly when prefer gate fires and rails empty."""
    orch = _orch_for_timeout_fallback(
        semantic_prefer_min=1, semantic_prefer_score=0.4, rail_first=True,
    )
    orch._explore_composer.compose_fallback = AsyncMock(
        return_value=(_ranked([]), MagicMock()),
    )
    orch._quick_semantic_retrieve = AsyncMock(return_value=[_item('sem-prefer.com', 0.9)])

    out = await orch.get_timeout_fallback(request_id='rid-sem', query='business', top_k=5)
    assert len(out.items) == 1
    assert out.items[0].item_id == 'sem-prefer.com'


@pytest.mark.asyncio
async def test_get_timeout_fallback_rail_first_keeps_explore_over_semantic():
    """B: nonempty explore rails block semantic-only prefer."""
    orch = _orch_for_timeout_fallback(
        semantic_prefer_min=1, semantic_prefer_score=0.4, rail_first=True,
    )
    orch._explore_composer.compose_fallback = AsyncMock(
        return_value=(_ranked([_item('rail.com', 0.2)]), MagicMock()),
    )
    orch._quick_semantic_retrieve = AsyncMock(return_value=[_item('sem-prefer.com', 0.9)])

    out = await orch.get_timeout_fallback(request_id='rid-rail-first', query='business', top_k=5)
    ids = {i.item_id for i in out.items}
    assert 'rail.com' in ids
    # Prefer gate must not return semantic-only singleton when rails nonempty.
    assert not (len(out.items) == 1 and out.items[0].item_id == 'sem-prefer.com')


@pytest.mark.asyncio
async def test_qdrant_filter_only_rails_swallows_missing_collection():
    """Missing auctions_listings -> [] (no 500 / Task exception)."""
    from semantic_search.core.exceptions import QdrantQueryError
    from semantic_search.orchestrator import SearchOrchestrator

    orch = _orch_for_timeout_fallback(qdrant_rails_when_ch_empty=True)
    orch._vector.retrieve_filter_only_rails = AsyncMock(
        side_effect=QdrantQueryError(
            "qdrant hybrid filter-only rails all failed "
            "Collection `auctions_listings` doesn't exist!"
        ),
    )
    items = await SearchOrchestrator._qdrant_filter_only_rails_as_ranked(
        orch, intent=MagicMock(request_id='rid-no-coll'), top_k=5,
    )
    assert items == []


@pytest.mark.asyncio
async def test_quick_semantic_retrieve_swallows_qdrant_unavailable():
    """Qdrant connection refused -> [] (force_semantic_nonempty must not 500)."""
    from semantic_search.core.exceptions import QdrantQueryError
    from semantic_search.orchestrator import SearchOrchestrator

    orch = _orch_for_timeout_fallback()
    orch._vector.retrieve = AsyncMock(
        side_effect=QdrantQueryError(
            "qdrant hybrid query failed: connection refused [::1]:6334"
        ),
    )
    items = await SearchOrchestrator._quick_semantic_retrieve(
        orch, intent=MagicMock(request_id='rid-qd-down'), top_k=5,
    )
    assert items == []


@pytest.mark.asyncio
async def test_get_timeout_fallback_qdrant_rails_when_ch_empty():
    """A: empty CH rails -> Qdrant filter_only_rails ladder fills explore leg."""
    from semantic_search.contracts import Candidate, CandidateSet

    orch = _orch_for_timeout_fallback(
        semantic_prefer_min=0, rail_first=True, qdrant_rails_when_ch_empty=True,
    )
    orch._explore_composer.compose_fallback = AsyncMock(
        return_value=(_ranked([]), MagicMock()),
    )
    orch._quick_semantic_retrieve = AsyncMock(return_value=[])

    async def _rails(_intent, _top_k):
        return CandidateSet(
            source='vector',
            candidates=[
                Candidate(
                    item_id='ending-soon.com',
                    score=0.5,
                    source='vector',
                    payload={'filter_only_rails': ['ending_soon']},
                ),
            ],
            latency_ms=1.0,
        )

    orch._vector.retrieve_filter_only_rails = AsyncMock(side_effect=_rails)

    out = await orch.get_timeout_fallback(request_id='rid-qd', query='domains', top_k=5)
    assert len(out.items) == 1
    assert out.items[0].item_id == 'ending-soon.com'
    orch._vector.retrieve_filter_only_rails.assert_awaited()


@pytest.mark.asyncio
async def test_get_timeout_fallback_gather_timeout_uses_qdrant_rails():
    """A: gather timeout still returns Qdrant rail ladder when enabled."""
    from semantic_search.contracts import Candidate, CandidateSet

    orch = _orch_for_timeout_fallback(
        fallback_timeout=0.05,
        qdrant_rails_when_ch_empty=True,
        qdrant_rails_timeout_seconds=1.0,
    )

    async def _slow(**_k):
        await asyncio.sleep(2)
        return (_ranked(), MagicMock())

    orch._explore_composer.compose_fallback = AsyncMock(side_effect=_slow)
    orch._quick_semantic_retrieve = AsyncMock(side_effect=_slow)

    async def _rails(_intent, _top_k):
        return CandidateSet(
            source='vector',
            candidates=[Candidate(item_id='qd-rail.com', score=0.8, source='vector', payload={})],
            latency_ms=1.0,
        )

    orch._vector.retrieve_filter_only_rails = AsyncMock(side_effect=_rails)

    out = await orch.get_timeout_fallback(request_id='rid-to-qd', query='slow', top_k=5)
    assert len(out.items) == 1
    assert out.items[0].item_id == 'qd-rail.com'


# ── TimeoutFallbackConfig (required keys, no silent defaults) ─────────────────


def test_timeout_fallback_config_from_dict_requires_all_keys():
    from semantic_search.config.models import TimeoutFallbackConfig
    from semantic_search.core.exceptions import ConfigurationError

    full = {
        'rail_first': True,
        'qdrant_rails_when_ch_empty': True,
        'qdrant_rails_timeout_seconds': 2.0,
        'consume_search_explore_prewarm': True,
        'search_explore_prewarm_wait_seconds': 2.0,
    }
    cfg = TimeoutFallbackConfig.from_dict(full)
    assert cfg.rail_first is True
    assert cfg.qdrant_rails_when_ch_empty is True
    assert cfg.qdrant_rails_timeout_seconds == 2.0
    assert cfg.consume_search_explore_prewarm is True
    assert cfg.search_explore_prewarm_wait_seconds == 2.0

    for missing in (
        'rail_first',
        'qdrant_rails_when_ch_empty',
        'qdrant_rails_timeout_seconds',
        'consume_search_explore_prewarm',
        'search_explore_prewarm_wait_seconds',
    ):
        bad = dict(full)
        del bad[missing]
        with pytest.raises(ConfigurationError):
            TimeoutFallbackConfig.from_dict(bad)

    with pytest.raises(ConfigurationError):
        TimeoutFallbackConfig.from_dict({**full, 'qdrant_rails_timeout_seconds': 0.0})
    with pytest.raises(ConfigurationError):
        TimeoutFallbackConfig.from_dict({**full, 'search_explore_prewarm_wait_seconds': 0.0})


def test_timeout_fallback_loaded_from_base_yaml():
    from semantic_search.config.loader import load_config
    from semantic_search.config.models import AgentSearchConfig

    raw = load_config()
    cfg = AgentSearchConfig.from_dict(raw)
    tf = cfg.general.search.timeout_fallback
    assert tf.rail_first is True
    assert tf.qdrant_rails_when_ch_empty is True
    assert tf.qdrant_rails_timeout_seconds == 2.0
    assert tf.consume_search_explore_prewarm is True
    assert tf.search_explore_prewarm_wait_seconds == 2.0


# ── Orchestrator.analytics timeout ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_orchestrator_analytics_timeout_returns_failure_mode():
    from semantic_search.orchestrator import SearchOrchestrator

    cfg = MagicMock()
    cfg.general.search.analytics_timeout_seconds = 0.05

    async def _slow(**_k):
        await asyncio.sleep(2)
        return MagicMock()

    orch = SearchOrchestrator.__new__(SearchOrchestrator)
    orch._config = cfg
    orch._analytics_router = MagicMock()
    orch._analytics_router.enabled = True
    orch._analytics_router.run = AsyncMock(side_effect=_slow)
    orch._analytics_rate_limiter = None
    orch._analytics_rate_limit_config = None
    orch._cost_budget_factory = None
    orch._last_cost_budget = None
    orch._sanitize_user_input = MagicMock()

    result = await orch.analytics(question='average price of .com', sql_hint='', request_id='rid-an')
    assert isinstance(result, AnalyticsResult)
    assert result.success is False
    assert result.failure_mode == 'timeout'
    assert 'exceeded' in (result.failure_reason or '').lower()


# ── answer_mode mapping (parent tier wins over explore rails) ─────────────────


@pytest.mark.parametrize(
    'query_type,is_explore_fallback,expected',
    [
        # Rails / no-rails: parent tier owns label for every known archetype.
        ('hybrid', True, 'search'),
        ('hybrid', False, 'search'),
        ('explore', True, 'explore'),
        ('explore', False, 'explore'),
        ('guidance', True, 'guidance'),
        ('guidance', False, 'guidance'),
        ('analytics', True, 'analytics'),
        ('analytics', False, 'analytics'),
        # No trusted parent tier -> explore_fallback.
        ('unknown', True, 'explore_fallback'),
        (None, True, 'explore_fallback'),
    ],
)
def test_answer_mode_for_ranked_response(query_type, is_explore_fallback, expected):
    from semantic_search.app import _answer_mode_for_query_type, _answer_mode_for_ranked_response

    assert _answer_mode_for_query_type(query_type) == expected
    if query_type is not None:
        assert _answer_mode_for_ranked_response(
            query_type=query_type, is_explore_fallback=is_explore_fallback,
        ) == expected
