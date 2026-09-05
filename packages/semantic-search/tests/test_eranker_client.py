"""Tests for ``semantic_search.retrieval.eranker_client`` (Layer-4 wire + HTTP client)."""
import pytest

from semantic_search.config.models import ERankerConfig, ERankerHttpConfig
from semantic_search.contracts import IntentSlice, QueryIntent, RankedItem, RankedResults, UserContext
from semantic_search.core.exceptions import ConfigurationError, RetrievalError, ValidationError
from semantic_search.retrieval.eranker_client import HttpERankerClient, NoOpERankerClient, _reorder_by_item_ids, build_eranker_client


def _intent() -> QueryIntent:
    return QueryIntent(
        request_id='req_t',
        raw_query='q',
        normalized_query='q',
        query_type='hybrid',
        confidence=0.9,
        decision_tier='L0_entity',
        slices=[IntentSlice(query_type='hybrid', entities=[], confidence=0.9, raw_text='q')],
        decision_cost_usd=0.0,
        intent_record_id='ir_1',
    )


def _ranked() -> RankedResults:
    items = [
        RankedItem(item_id='a', fused_score=1.0, contributing_sources=['vector']),
        RankedItem(item_id='b', fused_score=0.9, contributing_sources=['vector']),
    ]
    return RankedResults(request_id='req_t', items=items, total_candidates=2, fusion_latency_ms=1.0, query_intent=_intent())


class TestReorderByItemIds:
    def test_reorder_and_tail_append_missing(self):
        r = _ranked()
        out = _reorder_by_item_ids(r, ['b', 'a'])
        assert [it.item_id for it in out.items] == ['b', 'a']

    def test_unknown_id_skipped_tail_preserves(self):
        r = _ranked()
        out = _reorder_by_item_ids(r, ['b', 'x', 'a'])
        assert [it.item_id for it in out.items] == ['b', 'a']


class TestBuildErankerClient:
    def test_http_dispatch(self):
        cfg = ERankerConfig(
            enabled=True,
            backend='http',
            latency_budget_ms=500.0,
            shadow_enabled=False,
            shadow_serve_fused=True,
            skip_when_backend_unhealthy=True,
            http=ERankerHttpConfig(base_url='http://localhost:9', timeout_seconds=0.4, rank_path='v1/rank', send_user_id=False),
        )
        c = build_eranker_client(cfg)
        assert c.name == 'http_eranker'

    def test_invalid_backend_rejected_by_config(self):
        with pytest.raises(ConfigurationError, match='backend must be one of'):
            ERankerConfig(
                enabled=True,
                backend='grpc',
                latency_budget_ms=200.0,
                shadow_enabled=False,
                shadow_serve_fused=True,
                    skip_when_backend_unhealthy=True,
            )


class TestERankerConfigValidation:
    def test_http_timeout_exceeds_budget_rejected(self):
        with pytest.raises(ConfigurationError, match='timeout_seconds'):
            ERankerConfig(
                enabled=True,
                backend='http',
                latency_budget_ms=200.0,
                shadow_enabled=False,
                shadow_serve_fused=True,
                    skip_when_backend_unhealthy=True,
                http=ERankerHttpConfig(base_url='http://localhost:9', timeout_seconds=0.3, rank_path='v1/rank', send_user_id=False),
            )


@pytest.mark.asyncio
class TestNoOpERankerClient:
    async def test_rank_accepts_user_context(self):
        cfg = ERankerConfig(
            enabled=True,
            backend='noop',
            latency_budget_ms=200.0,
            shadow_enabled=False,
            shadow_serve_fused=True,
            skip_when_backend_unhealthy=True,
        )
        c = NoOpERankerClient(cfg)
        r = _ranked()
        uc = UserContext(user_id='u1', is_authenticated=True, session_id='s1')
        out = await c.rank('rid', _intent(), r, uc)
        assert out.items == r.items


@pytest.mark.asyncio
class TestHttpERankerClientMocked:
    async def test_post_reorders(self, monkeypatch):
        cfg = ERankerConfig(
            enabled=True,
            backend='http',
            latency_budget_ms=500.0,
            shadow_enabled=False,
            shadow_serve_fused=True,
            skip_when_backend_unhealthy=True,
            http=ERankerHttpConfig(base_url='http://localhost:9', timeout_seconds=0.4, rank_path='v1/rank', send_user_id=False),
        )
        client = HttpERankerClient(cfg)

        class _Resp:
            status_code = 200

            def json(self):
                return {'item_ids': ['b', 'a']}

        class _AC:
            def __init__(self, *a, **k):
                pass

            async def post(self, url, json=None, headers=None):
                assert 'v1/rank' in url
                assert json['request_id'] == 'rid_x'
                return _Resp()

            async def aclose(self):
                return None

        monkeypatch.setattr('semantic_search.retrieval.eranker_client.httpx.AsyncClient', lambda **kw: _AC())
        out = await client.rank('rid_x', _intent(), _ranked(), None)
        assert [it.item_id for it in out.items] == ['b', 'a']
        assert client.last_http_status == 200

    async def test_http_error_raises_retrieval_error(self, monkeypatch):
        cfg = ERankerConfig(
            enabled=True,
            backend='http',
            latency_budget_ms=500.0,
            shadow_enabled=False,
            shadow_serve_fused=True,
            skip_when_backend_unhealthy=True,
            http=ERankerHttpConfig(base_url='http://localhost:9', timeout_seconds=0.4, rank_path='v1/rank', send_user_id=False),
        )
        client = HttpERankerClient(cfg)

        class _Resp:
            status_code = 500

            def json(self):
                return {}

        class _AC:
            def __init__(self, *a, **k):
                pass

            async def post(self, url, json=None, headers=None):
                return _Resp()

            async def aclose(self):
                return None

        monkeypatch.setattr('semantic_search.retrieval.eranker_client.httpx.AsyncClient', lambda **kw: _AC())
        with pytest.raises(RetrievalError, match='eranker_http_status'):
            await client.rank('rid_x', _intent(), _ranked(), None)
