"""Tests for the Qdrant retrieval adapters.

Coverage matrix preserved as `pytest.param(id=...)` ids — recoverable via
`pytest --collect-only -q`. Compact one-line-per-row matrices sit at the
top of each test class.

No live Qdrant cluster required — all tests use a stubbed `AsyncQdrantClient`
(injected on the factory after construction) and a `build_qdrant_filter`-only
path that exercises real `qm.Filter` objects.
"""
from dataclasses import replace
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest
from qdrant_client import models as qm

from semantic_search.config.models import (
    AgentSearchConfig,
    BM25QueryEncoderConfig,
    DiversityConfig,
    ERankerConfig,
    FilterOnlyRailLegConfig,
    FilterOnlyRailsConfig,
    FusionConfig,
    QdrantConfig,
    QdrantHybridConfig,
    RetrievalConfig,
)
from semantic_search.config.models import RetrievalMetricsConfig, SqlRetrievalConfig, StructuredRetrievalConfig, VectorRetrievalConfig
from semantic_search.contracts import Entity, IntentSlice, QueryIntent
from semantic_search.core.exceptions import ConfigurationError, QdrantQueryError, QdrantUnavailableError, RetrievalError
from semantic_search.qi.encoder import HashingEncoder
from semantic_search.registry import _build_retrieval_backends
from semantic_search.retrieval.qdrant_adapter import QdrantClientFactory, QdrantHybridRetriever, QdrantNoOpStructuredRetriever, QdrantStructuredIndex, QdrantVectorIndex, build_qdrant_filter
from semantic_search.retrieval.structured_retriever import InMemoryStructuredIndex, StructuredRetriever
from semantic_search.retrieval.vector_retriever import InMemoryVectorIndex, VectorRetriever
from ._contract_helpers import matrix_param


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _default_eranker_config() -> ERankerConfig:
    return ERankerConfig(
        enabled=False,
        backend='noop',
        latency_budget_ms=200.0,
        shadow_enabled=False,
        shadow_serve_fused=True,
        skip_when_backend_unhealthy=True,
    )


def _default_diversity_config():
    """Disabled-by-default noop diversifier (lexical diversifier has its own tests)."""
    return DiversityConfig(enabled=False, backend='noop', top_n=50, output_n=50, latency_budget_ms=15.0, lexical=None)


def _default_filter_only_rails(*, enabled: bool = False) -> FilterOnlyRailsConfig:
    """Explicit filter-only rails config for tests (no silent production defaults)."""
    return FilterOnlyRailsConfig(
        enabled=enabled,
        rrf_k=60,
        legs=[
            FilterOnlyRailLegConfig(
                rail_id='ending_soon',
                order_by_field='ends_at',
                order_by_direction='asc',
                limit_multiplier=1,
            ),
            FilterOnlyRailLegConfig(
                rail_id='trending',
                order_by_field='bid_count',
                order_by_direction='desc',
                limit_multiplier=1,
            ),
            FilterOnlyRailLegConfig(
                rail_id='value',
                order_by_field='govalue_score',
                order_by_direction='desc',
                limit_multiplier=1,
            ),
        ],
    )


def _make_qdrant_config(
    *,
    host: str = 'localhost',
    api_key_env_var: str = '',
    payload_id_field: str = 'domain_name',
    payload_score_field: str = '',
    hybrid_enabled: bool = False,
    bm25_enabled: bool = False,
    fusion_strategy: str = 'rrf',
    prefetch_limit: int = 100,
    filter_only_rails: Optional[FilterOnlyRailsConfig] = None,
) -> QdrantConfig:
    """Build a typed QdrantConfig with sensible test defaults."""
    return QdrantConfig(
        host=host, port=6333, grpc_port=6334, prefer_grpc=False, https=False,
        api_key_env_var=api_key_env_var,
        collection_name='test_listings',
        payload_id_field=payload_id_field,
        payload_score_field=payload_score_field,
        connect_timeout_seconds=0.5, read_timeout_seconds=1.0,
        hnsw_ef_search=64,
        check_compatibility=True,
        hybrid=QdrantHybridConfig(
            enabled=hybrid_enabled,
            fusion_strategy=fusion_strategy,
            bm25_enabled=bm25_enabled,
            bm25_vector_name='bm25' if bm25_enabled else '',
            dense_vector_name='',
            prefetch_limit=prefetch_limit,
            kw_post_oversample_factor=3,
            kw_prefix_oversample_factor=6,
            bm25_query_encoder=BM25QueryEncoderConfig( vocab_size=4096, min_term_length=2, max_terms=64, stopwords=[],) if bm25_enabled else None
        ),
        filter_only_rails=filter_only_rails if filter_only_rails is not None else _default_filter_only_rails(enabled=False),
    )


def _make_retrieval_config(
    *,
    vector_backend: str = 'qdrant',
    structured_backend: str = 'qdrant',
    vector_top_k: int = 10,
    structured_top_k: int = 10,
    embedding_dim: int = 32,
    qdrant: Optional[QdrantConfig] = None,
) -> RetrievalConfig:
    """Build a `RetrievalConfig` with defaults that satisfy `__post_init__`.

    `qdrant=None` lets tests exercise the "missing qdrant block" branch when
    one of the backends asks for `qdrant`.
    """
    return RetrievalConfig(
        vector=VectorRetrievalConfig(enabled=True, top_k=vector_top_k, min_similarity=0.0,
                                     embedding_dim=embedding_dim, backend=vector_backend),
        structured=StructuredRetrievalConfig(enabled=True, top_k=structured_top_k, backend=structured_backend, word_count_filter_enabled=True, keyword_match_mode='any', unknown_selectable_fields={}, lifecycle_auction_type_map={}, traffic_signal_fields=[]),
        sql=SqlRetrievalConfig(enabled=True, top_k=10, allowed_filter_columns=['tld']),
        fusion=FusionConfig(rrf_k=60, top_n=50),
        eranker=_default_eranker_config(),
        diversity=_default_diversity_config(),
        metrics=RetrievalMetricsConfig(relevance_threshold=0.5),
        qdrant=qdrant,
    )


def _intent(query_type: str = 'hybrid', entities=None, normalized: str = 'cheap .com domains') -> QueryIntent:
    return QueryIntent(
        request_id='req_qdrant_test',
        raw_query=normalized,
        normalized_query=normalized,
        query_type=query_type,
        confidence=0.95,
        decision_tier='L0_entity',
        slices=[IntentSlice(query_type=query_type, entities=list(entities or []), confidence=0.95, raw_text=normalized)],
        decision_cost_usd=0.0,
    )


class _FakePoint:
    """Minimal stand-in for `qdrant_client.http.models.ScoredPoint`."""

    def __init__(self, point_id: Any, score: float, payload: Optional[Dict[str, Any]] = None):
        self.id = point_id
        self.score = score
        self.payload = payload or {}


class _FakeQueryResponse:
    """Minimal stand-in for `qdrant_client.http.models.QueryResponse`."""

    def __init__(self, points: List[_FakePoint]):
        self.points = points


class _StubAsyncQdrantClient:
    """Async stub recording every call. Mirrors the subset of
    `AsyncQdrantClient` the adapters touch.
    """

    def __init__( self, query_points_return: Any = None, scroll_return: Any = None, raise_on: Optional[str] = None, ):
        self.query_points_calls: List[Dict[str, Any]] = []
        self.scroll_calls: List[Dict[str, Any]] = []
        self._query_points_return = query_points_return
        self._scroll_return = scroll_return
        self._raise_on = raise_on
        self.closed = False

    async def query_points(self, **kwargs) -> Any:
        self.query_points_calls.append(kwargs)
        if self._raise_on == 'query_points':
            raise RuntimeError('boom')
        if self._query_points_return is None:
            return _FakeQueryResponse(points=[])
        return self._query_points_return

    async def scroll(self, **kwargs) -> Any:
        self.scroll_calls.append(kwargs)
        if self._raise_on == 'scroll':
            raise RuntimeError('boom')
        if self._scroll_return is None:
            return ([], None)
        if callable(self._scroll_return):
            return self._scroll_return(kwargs)
        return self._scroll_return

    async def close(self) -> None:
        self.closed = True


def _factory_with_stub(stub: _StubAsyncQdrantClient, qcfg: Optional[QdrantConfig] = None) -> QdrantClientFactory:
    """Build a QdrantClientFactory wired to a stub client (no real connection)."""
    with patch('semantic_search.retrieval.qdrant_adapter._AsyncQdrantClient', return_value=stub):
        factory = QdrantClientFactory(qcfg or _make_qdrant_config())
    factory._available = True  # noqa: SLF001
    return factory


def _unavailable_factory(qcfg: Optional[QdrantConfig] = None) -> QdrantClientFactory:
    """Force ImportError → unavailable factory (no qdrant_client wiring)."""
    cfg = qcfg or _make_qdrant_config()
    with patch('semantic_search.retrieval.qdrant_adapter._AsyncQdrantClient', None):
        return QdrantClientFactory(cfg)


# ---------------------------------------------------------------------------
# Construction-safety + factory contract
# ---------------------------------------------------------------------------

class TestQdrantClientFactory:
    """Coverage matrix:
        - construction_safe_when_qdrant_client_missing -> test_factory_construction_safe_when_qdrant_client_missing
    - construction_safe_on_constructor_error      -> test_factory_construction_safe_on_constructor_error
    - api_key_resolution_from_env                 -> test_factory_api_key_resolution (parametrized: env_set / env_missing)
    - rejects_non_qdrant_config                   -> test_factory_rejects_non_qdrant_config
    - aclose_idempotent_and_safe                  -> test_factory_aclose_idempotent_and_safe
    """

    def test_factory_construction_safe_when_qdrant_client_missing(self):
        """ImportError of qdrant_client → available=False, no raise."""
        factory = _unavailable_factory()
        assert factory.available is False
        assert factory.client is None

    def test_factory_construction_safe_on_constructor_error(self):
        """AsyncQdrantClient ctor raising → available=False, no boot crash."""
        with patch('semantic_search.retrieval.qdrant_adapter._AsyncQdrantClient', side_effect=RuntimeError('nope')):
            factory = QdrantClientFactory(_make_qdrant_config())
        assert factory.available is False
        assert factory.client is None

    @pytest.mark.parametrize('env_var,env_set,expected_api_key', [
        matrix_param('env_var_set',     'QDRANT_TEST_KEY',    'sekret-123', 'sekret-123'),
        matrix_param('env_var_missing', 'QDRANT_MISSING_KEY', None,         None),
    ])
    def test_factory_api_key_resolution( self, env_var, env_set, expected_api_key, monkeypatch: pytest.MonkeyPatch, ):
        """API key sourced from OS env, never YAML (responsible-ai.mdc §secret-handling)."""
        cfg = _make_qdrant_config(api_key_env_var=env_var)
        if env_set is not None:
            monkeypatch.setenv(env_var, env_set)
        else:
            monkeypatch.delenv(env_var, raising=False)
        captured: Dict[str, Any] = {}

        def _fake_ctor(*args, **kwargs):
            captured.update(kwargs)
            return _StubAsyncQdrantClient()

        with patch('semantic_search.retrieval.qdrant_adapter._AsyncQdrantClient', side_effect=_fake_ctor):
            factory = QdrantClientFactory(cfg)
        assert factory.available is True
        assert captured.get('api_key') == expected_api_key

    def test_factory_rejects_non_qdrant_config(self):
        """Wrong config type → RetrievalError (typed contract)."""
        with pytest.raises(RetrievalError):
            QdrantClientFactory(config="not a QdrantConfig")  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_factory_aclose_idempotent_and_safe(self):
        """aclose() is idempotent and never raises."""
        stub = _StubAsyncQdrantClient()
        factory = _factory_with_stub(stub)
        await factory.aclose()
        await factory.aclose()  # second call is no-op
        assert stub.closed is True


# ---------------------------------------------------------------------------
# Filter translation
# ---------------------------------------------------------------------------

class TestBuildQdrantFilter:
    """Coverage matrix:
        - empty_filters_returns_none                  -> test_empty_filters_returns_none
    - single-slot translations (5 slots)          -> test_single_slot_translation (parametrized)
    - combined_filters_emit_multiple_conditions   -> test_combined_filters_emit_multiple_conditions
    """

    def test_empty_filters_returns_none(self):
        assert build_qdrant_filter({}) is None

    @pytest.mark.parametrize('filters,expected_key,assertion', [
        # tld values must be lowercased and emit a `match.any` against `tld`
        matrix_param(
            'tld_match_any',
            {'tld': ['com', 'IO']}, 'tld',
            lambda c: c.match.any == ['com', 'io'],
        ),
        # auction_type values must be lowercased and emit `match.any` against `auction_type`
        matrix_param(
            'auction_type_match_any',
            {'auction_type': ['Expired', 'CLOSEOUT']}, 'auction_type',
            lambda c: c.match.any == ['expired', 'closeout'],
        ),
        # price_min + price_max collapse into a single range over `price`
        matrix_param(
            'price_min_max_combined_range',
            {'price_min': 10, 'price_max': 100}, 'price',
            lambda c: c.range.gte == 10.0 and c.range.lte == 100.0,
        ),
        # name_length_max → range.lte against `name_length`
        matrix_param(
            'name_length_max_lte',
            {'name_length_max': 7}, 'name_length',
            lambda c: c.range.lte == 7.0,
        ),
        # quality_min → range.gte against `quality`
        matrix_param(
            'quality_min_gte',
            {'quality_min': 0.6}, 'quality',
            lambda c: c.range.gte == 0.6,
        ),
    ])
    def test_single_slot_translation(self, filters, expected_key, assertion):
        f = build_qdrant_filter(filters)
        assert isinstance(f, qm.Filter)
        # Find the condition for this slot (price_min/price_max collapse → one cond).
        conds = [c for c in f.must if c.key == expected_key]
        assert len(conds) == 1, f"expected 1 condition for {expected_key}, got {len(conds)}"
        assert assertion(conds[0]), f"assertion failed for {expected_key}: {conds[0]}"

    def test_combined_filters_emit_multiple_conditions(self):
        f = build_qdrant_filter({'tld': ['com'], 'price_max': 50, 'name_length_max': 8})
        assert isinstance(f, qm.Filter)
        keys = sorted(c.key for c in f.must)
        assert keys == ['name_length', 'price', 'tld']


# ---------------------------------------------------------------------------
# QdrantVectorIndex
# ---------------------------------------------------------------------------

class TestQdrantVectorIndex:
    """Coverage matrix:
        - unavailable_factory_raises_qdrant_unavailable -> test_unavailable_factory_raises_qdrant_unavailable
    - top_k_zero_returns_empty                      -> test_top_k_zero_returns_empty
    - set_dim_hint_validation                       -> test_set_dim_hint_validation
    - search_emits_query_points_with_hnsw_params    -> test_search_emits_query_points_with_hnsw_params
    """

    def test_unavailable_factory_raises_qdrant_unavailable(self):
        index = QdrantVectorIndex(_unavailable_factory())
        index.set_dim_hint(32)
        with pytest.raises(QdrantUnavailableError):
            index.search([0.1] * 32, top_k=5)

    def test_top_k_zero_returns_empty(self):
        stub = _StubAsyncQdrantClient()
        index = QdrantVectorIndex(_factory_with_stub(stub))
        index.set_dim_hint(32)
        assert index.search([0.1] * 32, top_k=0) == []
        assert stub.query_points_calls == []

    def test_set_dim_hint_validation(self):
        index = QdrantVectorIndex(_factory_with_stub(_StubAsyncQdrantClient()))
        with pytest.raises(RetrievalError):
            index.set_dim_hint(2)
        index.set_dim_hint(32)
        assert index.dim == 32

    def test_search_emits_query_points_with_hnsw_params(self):
        stub = _StubAsyncQdrantClient(
            query_points_return=_FakeQueryResponse(points=[
                _FakePoint(point_id=1, score=0.8, payload={'domain_name': 'example.com', 'tld': 'com'}),
                _FakePoint(point_id=2, score=0.6, payload={'domain_name': 'foo.com'}),
            ])
        )
        index = QdrantVectorIndex(_factory_with_stub(stub))
        index.set_dim_hint(32)
        results = index.search([0.1] * 32, top_k=2)
        assert len(stub.query_points_calls) == 1
        call = stub.query_points_calls[0]
        assert call['collection_name'] == 'test_listings'
        assert call['limit'] == 2
        assert call['with_payload'] is True
        assert call['search_params'].hnsw_ef == 64
        # Item id resolution comes from payload.domain_name (configured field)
        assert results[0][0] == 'example.com'
        assert 0.0 <= results[0][1] <= 1.0


# ---------------------------------------------------------------------------
# QdrantStructuredIndex
# ---------------------------------------------------------------------------

class TestQdrantStructuredIndex:
    """Coverage matrix:
        - empty_filters_short_circuits_no_call          -> test_empty_filters_short_circuits_no_call
    - scroll_returns_payload_id_when_field_set      -> test_scroll_returns_payload_id_when_field_configured
    - payload_score_field_overrides_rank_score      -> test_payload_score_field_overrides_rank_score
    - unavailable_factory_raises                    -> test_unavailable_factory_raises
    - scroll_transport_error_wrapped                -> test_scroll_transport_error_wrapped
    """

    def test_empty_filters_short_circuits_no_call(self):
        stub = _StubAsyncQdrantClient()
        index = QdrantStructuredIndex(_factory_with_stub(stub))
        assert index.search({}, top_k=10) == []
        assert stub.scroll_calls == []

    def test_scroll_returns_payload_id_when_field_configured(self):
        stub = _StubAsyncQdrantClient(
            scroll_return=(
                [
                    _FakePoint(point_id='pt-1', score=0.0, payload={'domain_name': 'a.com', 'tld': 'com', 'price': 50}),
                    _FakePoint(point_id='pt-2', score=0.0, payload={'domain_name': 'b.com', 'tld': 'com', 'price': 75}),
                ],
                None,
            )
        )
        index = QdrantStructuredIndex(_factory_with_stub(stub))
        rows = index.search({'tld': ['com']}, top_k=2)
        assert [r['item_id'] for r in rows] == ['a.com', 'b.com']
        # rank-derived score: r=0 → 1.0, r=1 → 0.5
        assert rows[0]['score'] == pytest.approx(1.0)
        assert rows[1]['score'] == pytest.approx(0.5)
        assert rows[0]['tld'] == 'com'

    def test_payload_score_field_overrides_rank_score(self):
        cfg = _make_qdrant_config(payload_score_field='govalue_score')
        stub = _StubAsyncQdrantClient(
        scroll_return=( [_FakePoint(point_id='x', score=0.0, payload={'domain_name': 'a.com', 'govalue_score': 0.42})], None,)
        )
        index = QdrantStructuredIndex(_factory_with_stub(stub, qcfg=cfg))
        rows = index.search({'tld': ['com']}, top_k=1)
        assert rows[0]['score'] == pytest.approx(0.42)

    def test_unavailable_factory_raises(self):
        index = QdrantStructuredIndex(_unavailable_factory())
        with pytest.raises(QdrantUnavailableError):
            index.search({'tld': ['com']}, top_k=5)

    def test_scroll_transport_error_wrapped(self):
        stub = _StubAsyncQdrantClient(raise_on='scroll')
        index = QdrantStructuredIndex(_factory_with_stub(stub))
        with pytest.raises(QdrantQueryError):
            index.search({'tld': ['com']}, top_k=5)


# ---------------------------------------------------------------------------
# QdrantHybridRetriever — Option A semantics
# ---------------------------------------------------------------------------

def _hybrid_retriever(
    *, bm25_enabled: bool = False, fusion_strategy: str = 'rrf', prefetch_limit: int = 100,
    min_similarity: float = 0.0, query_points_return=None, scroll_return=None, raise_on=None,
    bm25_query_fn=None, filter_only_rails: Optional[FilterOnlyRailsConfig] = None,
):
    """Construct a QdrantHybridRetriever wired to a stub client. Returns
    (retriever, stub) so tests can assert call shape after retrieve()."""
    cfg = _make_qdrant_config(
        hybrid_enabled=True,
        bm25_enabled=bm25_enabled,
        fusion_strategy=fusion_strategy,
        prefetch_limit=prefetch_limit,
        filter_only_rails=filter_only_rails,
    )
    stub = _StubAsyncQdrantClient(query_points_return=query_points_return, scroll_return=scroll_return, raise_on=raise_on)
    factory = _factory_with_stub(stub, qcfg=cfg)
    encoder = HashingEncoder(dim=32, seed=7)
    if bm25_enabled and bm25_query_fn is None:

        def _stub_bm25(text: str):
            return qm.SparseVector(indices=[1, 2, 3], values=[0.1, 0.2, 0.3])

        bm25_query_fn = _stub_bm25
    retr = QdrantHybridRetriever(factory=factory, encoder=encoder, embedding_dim=32, min_similarity=min_similarity, bm25_query_fn=bm25_query_fn, keyword_match_mode='any')
    return retr, stub


class TestQdrantHybridRetriever:
    """Coverage matrix:
        - source_is_vector_for_orchestrator_compat   -> test_source_is_vector_for_orchestrator_compat
    - constructor_rejects_dim_mismatch           -> test_constructor_rejects_dim_mismatch
    - constructor_requires_bm25_fn_when_enabled  -> test_constructor_requires_bm25_fn_when_bm25_enabled
    - dense_only_path_emits_single_call          -> test_dense_only_path_emits_single_call_with_filter
    - bm25_path_uses_server_side_fusion_query    -> test_bm25_path_uses_server_side_fusion_query
    - min_similarity_filters_low_score           -> test_min_similarity_filters_low_score_candidates
    - empty_response_returns_empty_candidate_set -> test_empty_response_returns_empty_candidate_set
    - query_transport_error_wrapped              -> test_query_transport_error_wrapped
    - empty_encode_scrolls_filter_only           -> test_empty_encode_text_uses_filter_only_scroll
    - empty_encode_no_filter_returns_empty       -> test_empty_encode_without_filter_returns_empty
    """

    def test_source_is_vector_for_orchestrator_compat(self):
        retr, _ = _hybrid_retriever()
        assert retr.source == 'vector'

    def test_constructor_rejects_dim_mismatch(self):
        cfg = _make_qdrant_config(hybrid_enabled=True)
        factory = _factory_with_stub(_StubAsyncQdrantClient(), qcfg=cfg)
        encoder = HashingEncoder(dim=16, seed=7)  # mismatched
        with pytest.raises(RetrievalError):
            QdrantHybridRetriever(factory=factory, encoder=encoder, embedding_dim=32, min_similarity=0.0, keyword_match_mode='any')

    def test_constructor_requires_bm25_fn_when_bm25_enabled(self):
        cfg = _make_qdrant_config(hybrid_enabled=True, bm25_enabled=True)
        factory = _factory_with_stub(_StubAsyncQdrantClient(), qcfg=cfg)
        encoder = HashingEncoder(dim=32, seed=7)
        with pytest.raises(RetrievalError):
            QdrantHybridRetriever( factory=factory, encoder=encoder, embedding_dim=32, min_similarity=0.0, bm25_query_fn=None, keyword_match_mode='any',)

    @pytest.mark.asyncio
    async def test_dense_only_path_emits_single_call_with_filter(self):
        retr, stub = _hybrid_retriever(query_points_return=_FakeQueryResponse(points=[
            _FakePoint(point_id=1, score=0.8, payload={'domain_name': 'a.com', 'tld': 'com'}),
        ]))
        intent = _intent( query_type='hybrid', entities=[Entity(name='tld', value=['com'], confidence=0.95, source='L0_entity', chip_kind='hard')],)
        cs = await retr.retrieve(intent, top_k=10)
        assert cs.source == 'vector'
        assert len(cs.candidates) == 1
        assert cs.candidates[0].source == 'vector'
        assert len(stub.query_points_calls) == 1
        call = stub.query_points_calls[0]
        # No prefetch / fusion when BM25 is off
        assert 'prefetch' not in call
        assert call['query_filter'] is not None
        assert isinstance(call['query'], list)  # dense vector
        assert len(call['query']) == 32

    @pytest.mark.asyncio
    async def test_bm25_path_uses_server_side_fusion_query(self):
        retr, stub = _hybrid_retriever( bm25_enabled=True, fusion_strategy='rrf', prefetch_limit=50, query_points_return=_FakeQueryResponse(points=[]),)
        await retr.retrieve(_intent(query_type='hybrid'), top_k=10)
        assert len(stub.query_points_calls) == 1
        call = stub.query_points_calls[0]
        assert 'prefetch' in call
        assert len(call['prefetch']) == 2
        assert isinstance(call['query'], qm.FusionQuery)
        assert call['query'].fusion == qm.Fusion.RRF

    @pytest.mark.asyncio
    async def test_all_legs_encode_tld_safe_text_filter_stays_exact(self):
        """Dense + BM25 legs encode the TLD-safe ``semantic_encode_text``; the
        TLD never reaches an encoder, yet the exact ``tld`` MatchAny filter is
        still built from the entity (TLD = exact filter, never semantic)."""
        captured = {'dense': [], 'bm25': []}

        class _RecEnc:
            dim = 32

            def __init__(self):
                self._inner = HashingEncoder(dim=32, seed=7)

            def encode(self, text):
                captured['dense'].append(text)
                return self._inner.encode(text)

        def _rec_bm25(text):
            captured['bm25'].append(text)
            return qm.SparseVector(indices=[1, 2, 3], values=[0.1, 0.2, 0.3])

        cfg = _make_qdrant_config(hybrid_enabled=True, bm25_enabled=True, prefetch_limit=50)
        stub = _StubAsyncQdrantClient(query_points_return=_FakeQueryResponse(points=[]))
        factory = _factory_with_stub(stub, qcfg=cfg)
        retr = QdrantHybridRetriever(factory=factory, encoder=_RecEnc(), embedding_dim=32, min_similarity=0.0, bm25_query_fn=_rec_bm25, keyword_match_mode='any')

        intent = _intent(
            query_type='hybrid',
            entities=[Entity(name='tld', value=['com'], confidence=0.95, source='L0_entity', chip_kind='hard')],
            normalized='cheap com domains',
        )
        intent = replace(intent, semantic_encode_text='coffee shop', semantic_query='coffee shop')
        await retr.retrieve(intent, top_k=10)

        # Every encode leg saw the TLD-safe text — no 'com' token leaked.
        assert captured['dense'] == ['coffee shop']
        assert captured['bm25'] == ['coffee shop']
        # The TLD is still applied as an exact filter.
        f = build_qdrant_filter({'tld': ['com']}, active_only=cfg.active_only_baseline, price_gt_zero=cfg.price_gt_zero_baseline)
        assert any(getattr(c, 'key', None) == 'tld' for c in f.must)

    @pytest.mark.asyncio
    async def test_min_similarity_filters_low_score_candidates(self):
        retr, _ = _hybrid_retriever(
            min_similarity=0.5,
            query_points_return=_FakeQueryResponse(points=[
                _FakePoint(point_id=1, score=0.9, payload={'domain_name': 'high.com'}),
                _FakePoint(point_id=2, score=0.05, payload={'domain_name': 'low.com'}),
            ]),
        )
        cs = await retr.retrieve(_intent(query_type='hybrid'), top_k=10)
        assert [c.item_id for c in cs.candidates] == ['high.com']

    @pytest.mark.asyncio
    async def test_empty_response_returns_empty_candidate_set(self):
        retr, _ = _hybrid_retriever(query_points_return=_FakeQueryResponse(points=[]))
        cs = await retr.retrieve(_intent(), top_k=10)
        assert cs.source == 'vector'
        assert cs.candidates == []

    @pytest.mark.asyncio
    async def test_query_transport_error_wrapped(self):
        retr, _ = _hybrid_retriever(raise_on='query_points')
        with pytest.raises(QdrantQueryError):
            await retr.retrieve(_intent(), top_k=10)

    @pytest.mark.asyncio
    async def test_empty_encode_text_uses_filter_only_scroll(self):
        """Pure-filter residual (empty encode) must scroll, not ANN zero-vector."""
        from dataclasses import replace

        scroll_return = (
            [
                _FakePoint(
                    point_id=1,
                    score=0.0,
                    payload={'domain_name': 'hongkongtales.com', 'tld': 'com', 'auction_type': '16', 'price': 0.0, 'govalue_score': 19.0},
                ),
            ],
            None,
        )
        retr, stub = _hybrid_retriever(scroll_return=scroll_return)
        intent = _intent(
            query_type='hybrid',
            entities=[
                Entity(name='tld', value=['com'], confidence=0.95, source='L0_entity', chip_kind='hard'),
                Entity(name='auction_type', value=['16'], confidence=0.95, source='L0_entity', chip_kind='hard'),
                Entity(name='price_max', value=49, confidence=0.95, source='L0_entity', chip_kind='hard'),
            ],
            normalized='find .com domains for auction type 16 under $50',
        )
        intent = replace(intent, semantic_encode_text='', semantic_query=None, residual_kind='empty')
        cs = await retr.retrieve(intent, top_k=10)
        assert cs.source == 'vector'
        assert len(cs.candidates) == 1
        assert cs.candidates[0].item_id == 'hongkongtales.com'
        assert stub.query_points_calls == []
        assert len(stub.scroll_calls) == 1
        assert stub.scroll_calls[0]['scroll_filter'] is not None

    @pytest.mark.asyncio
    async def test_empty_encode_without_filter_returns_empty(self):
        """Empty encode + no payload filter → empty set (do not scroll whole corpus)."""
        from dataclasses import replace

        retr, stub = _hybrid_retriever()
        # Test QdrantConfig defaults have baselines off → qfilter is None.
        intent = replace(_intent(entities=[]), semantic_encode_text='', semantic_query=None, residual_kind='empty')
        cs = await retr.retrieve(intent, top_k=10)
        assert cs.candidates == []
        assert stub.query_points_calls == []
        assert stub.scroll_calls == []

    @pytest.mark.asyncio
    async def test_filter_only_rails_multi_scroll_rrf_respects_order_by(self):
        """Enabled rails: one ordered scroll per leg; RRF prefers multi-leg hits."""
        from dataclasses import replace

        soon = _FakePoint(
            point_id=1, score=0.0,
            payload={'domain_name': 'soon.com', 'tld': 'com', 'price': 10.0, 'ends_at': 100.0, 'bid_count': 1, 'govalue_score': 5.0},
        )
        hot = _FakePoint(
            point_id=2, score=0.0,
            payload={'domain_name': 'hot.com', 'tld': 'com', 'price': 20.0, 'ends_at': 900.0, 'bid_count': 50, 'govalue_score': 8.0},
        )
        both = _FakePoint(
            point_id=3, score=0.0,
            payload={'domain_name': 'both.com', 'tld': 'com', 'price': 15.0, 'ends_at': 110.0, 'bid_count': 40, 'govalue_score': 9.0},
        )

        def _by_order(kwargs: Dict[str, Any]):
            order_by = kwargs.get('order_by')
            key = getattr(order_by, 'key', None)
            if key == 'ends_at':
                return ([soon, both, hot], None)
            if key == 'bid_count':
                return ([hot, both, soon], None)
            if key == 'govalue_score':
                return ([both, hot, soon], None)
            return ([], None)

        rails = _default_filter_only_rails(enabled=True)
        retr, stub = _hybrid_retriever(scroll_return=_by_order, filter_only_rails=rails)
        intent = _intent(
            query_type='hybrid',
            entities=[
                Entity(name='tld', value=['com'], confidence=0.95, source='L0_entity', chip_kind='hard'),
                Entity(name='price_max', value=49, confidence=0.95, source='L0_entity', chip_kind='hard'),
            ],
            normalized='.com under $50',
        )
        intent = replace(intent, semantic_encode_text='', semantic_query=None, residual_kind='empty')
        cs = await retr.retrieve(intent, top_k=10)
        assert stub.query_points_calls == []
        assert len(stub.scroll_calls) == 3
        for call in stub.scroll_calls:
            assert call['scroll_filter'] is not None
            assert call.get('order_by') is not None
        assert [c.item_id for c in cs.candidates[:3]] == ['both.com', 'hot.com', 'soon.com']
        assert 'ending_soon' in cs.candidates[0].payload.get('filter_only_rails', [])

    @pytest.mark.asyncio
    async def test_filter_only_rails_all_scrolls_share_hard_filter(self):
        """Every rail scroll must carry the same hard qfilter (no inventory widen)."""
        from dataclasses import replace

        point = _FakePoint(
            point_id=1, score=0.0,
            payload={'domain_name': 'cheap.com', 'tld': 'com', 'price': 12.0, 'ends_at': 50.0, 'bid_count': 2, 'govalue_score': 3.0},
        )
        rails = _default_filter_only_rails(enabled=True)
        retr, stub = _hybrid_retriever(
            scroll_return=([point], None),
            filter_only_rails=rails,
        )
        intent = replace(
            _intent(
                entities=[
                    Entity(name='price_max', value=49, confidence=0.95, source='L0_entity', chip_kind='hard'),
                ],
            ),
            semantic_encode_text='',
            semantic_query=None,
            residual_kind='empty',
        )
        await retr.retrieve(intent, top_k=5)
        assert len(stub.scroll_calls) == 3
        filters = [call['scroll_filter'] for call in stub.scroll_calls]
        assert all(f is not None for f in filters)
        # Same filter object shape across legs (price_max must be present on each).
        for f in filters:
            assert any(getattr(c, 'key', None) == 'price' for c in (f.must or []))


class TestFilterOnlyRailsConfig:
    def test_from_dict_requires_all_keys(self):
        with pytest.raises(ConfigurationError):
            FilterOnlyRailsConfig.from_dict({'enabled': True, 'rrf_k': 60})

    def test_leg_rejects_bad_direction(self):
        with pytest.raises(ConfigurationError):
            FilterOnlyRailLegConfig(
                rail_id='ending_soon',
                order_by_field='ends_at',
                order_by_direction='sideways',
                limit_multiplier=1,
            )

    def test_duplicate_rail_id_rejected(self):
        leg = FilterOnlyRailLegConfig(
            rail_id='ending_soon',
            order_by_field='ends_at',
            order_by_direction='asc',
            limit_multiplier=1,
        )
        with pytest.raises(ConfigurationError):
            FilterOnlyRailsConfig(enabled=True, rrf_k=60, legs=[leg, leg])

    def test_qdrant_from_dict_requires_filter_only_rails(self):
        raw = {
            'host': 'localhost', 'port': 6333, 'grpc_port': 6334,
            'prefer_grpc': False, 'https': False, 'api_key_env_var': '',
            'collection_name': 'c', 'payload_id_field': 'domain_name',
            'payload_score_field': 'govalue_score',
            'connect_timeout_seconds': 1.0, 'read_timeout_seconds': 1.0,
            'hnsw_ef_search': 64, 'check_compatibility': True,
            'hybrid': {
                'enabled': True, 'fusion_strategy': 'rrf', 'bm25_enabled': False,
                'bm25_vector_name': '', 'dense_vector_name': '', 'prefetch_limit': 10,
                'kw_post_oversample_factor': 3, 'kw_prefix_oversample_factor': 6,
            },
        }
        with pytest.raises(ConfigurationError):
            QdrantConfig.from_dict(raw)


# ---------------------------------------------------------------------------
# QdrantNoOpStructuredRetriever — Option A pair-mate
# ---------------------------------------------------------------------------

class TestQdrantNoOpStructuredRetriever:
    @pytest.mark.asyncio
    async def test_returns_empty_structured_candidate_set(self):
        retr = QdrantNoOpStructuredRetriever()
        cs = await retr.retrieve(_intent(), top_k=10)
        assert cs.source == 'structured'
        assert cs.candidates == []
        assert cs.latency_ms == 0.0


# ---------------------------------------------------------------------------
# Cross-field config invariants
# ---------------------------------------------------------------------------

class TestRetrievalConfigInvariants:
    """Coverage matrix:
        - qdrant_block_required_when_backend_qdrant -> test_invariant_violation (no_qdrant_block)
    - hybrid_requires_both_backends_qdrant      -> test_invariant_violation (hybrid_mixed_backends)
    - hybrid_prefetch_limit_must_cover_top_k    -> test_invariant_violation (prefetch_too_small)
    - invalid_backend_value_rejected            -> test_invalid_backend_value_rejected
    - bm25_requires_non_empty_vector_name       -> test_bm25_requires_non_empty_vector_name
    """

    @pytest.mark.parametrize('config_factory', [
        # No qdrant block while vector.backend='qdrant' → invariant fires.
        matrix_param( 'no_qdrant_block', lambda: _make_retrieval_config(vector_backend='qdrant', structured_backend='memory', qdrant=None),),
        # Hybrid requires BOTH backends to be qdrant; this mixes memory in.
        matrix_param(
            'hybrid_mixed_backends',
            lambda: _make_retrieval_config( vector_backend='qdrant', structured_backend='memory', qdrant=_make_qdrant_config(hybrid_enabled=True, prefetch_limit=100),)
        ),
        # Hybrid prefetch limit must cover top_k.
        matrix_param(
            'prefetch_too_small',
            lambda: _make_retrieval_config( vector_top_k=200, structured_top_k=200, qdrant=_make_qdrant_config(hybrid_enabled=True, prefetch_limit=50),)
        ),
    ])
    def test_invariant_violation(self, config_factory):
        with pytest.raises(ConfigurationError):
            config_factory()

    def test_invalid_backend_value_rejected(self):
        with pytest.raises(ConfigurationError):
            VectorRetrievalConfig( enabled=True, top_k=10, min_similarity=0.0, embedding_dim=32, backend='postgres',)

    def test_bm25_requires_non_empty_vector_name(self):
        with pytest.raises(ConfigurationError):
            QdrantHybridConfig( enabled=True, fusion_strategy='rrf', bm25_enabled=True, bm25_vector_name='', dense_vector_name='', prefetch_limit=100, kw_post_oversample_factor=3, kw_prefix_oversample_factor=6,)


# ---------------------------------------------------------------------------
# Registry branch matrix
# ---------------------------------------------------------------------------

def _replace_retrieval(config: AgentSearchConfig, *, hybrid_enabled: bool, prefetch_limit: int = 100) -> AgentSearchConfig:
    """Build a config with both backends switched to qdrant + a fresh QdrantConfig."""
    qcfg = _make_qdrant_config(hybrid_enabled=hybrid_enabled, prefetch_limit=prefetch_limit)
    new_vector = replace(config.retrieval.vector, backend='qdrant')
    new_struct = replace(config.retrieval.structured, backend='qdrant')
    new_retr = RetrievalConfig(
        vector=new_vector, structured=new_struct, sql=config.retrieval.sql,
        fusion=config.retrieval.fusion, eranker=config.retrieval.eranker,
        diversity=config.retrieval.diversity, metrics=config.retrieval.metrics, qdrant=qcfg,
    )
    new_vec_cfg = config.vectorization
    if new_vec_cfg is not None:
        new_vec_cfg = replace(new_vec_cfg, synonyms=None)
    return replace(config, retrieval=new_retr, vectorization=new_vec_cfg)


class TestRegistryBranching:
    """Coverage matrix:
        - memory_only_default_path                -> test_memory_only_default_path
    - qdrant_unavailable_falls_back_to_memory -> test_qdrant_unavailable_falls_back_to_memory
    - hybrid_path_wires_qdrant_hybrid_and_noop -> test_qdrant_path_wires_correct_backends (hybrid)
    - classic_qdrant_path_wires_dual_backends  -> test_qdrant_path_wires_correct_backends (classic)
    """

    def test_memory_only_default_path(self, config: AgentSearchConfig):
        """Default config (vector.backend=memory + structured.backend=memory) → in-memory backends, no factory."""
        dim = config.retrieval.vector.embedding_dim
        factory, vidx, sidx, vretr, sretr, *_ = _build_retrieval_backends(config=config, encoder=HashingEncoder(dim=dim, seed=7))
        assert factory is None
        assert isinstance(vidx, InMemoryVectorIndex)
        assert isinstance(sidx, InMemoryStructuredIndex)
        assert isinstance(vretr, VectorRetriever)
        assert isinstance(sretr, StructuredRetriever)

    def test_qdrant_unavailable_falls_back_to_memory(self, config: AgentSearchConfig):
        """Qdrant requested but factory.available=False → silent fallback to in-memory."""
        new_config = _replace_retrieval(config, hybrid_enabled=True, prefetch_limit=100)
        # Use an unreachable host AND patch the ctor to raise to deterministically
        # exercise the soft-failure branch.
        new_config = replace(new_config, retrieval=replace( new_config.retrieval, qdrant=replace(new_config.retrieval.qdrant, host='unreachable.invalid'),))
        dim = config.retrieval.vector.embedding_dim
        with patch('semantic_search.retrieval.qdrant_adapter._AsyncQdrantClient', side_effect=RuntimeError('host down')):
            factory, vidx, sidx, vretr, sretr, *_ = _build_retrieval_backends(config=new_config, encoder=HashingEncoder(dim=dim, seed=7))
        assert factory is None
        assert isinstance(vidx, InMemoryVectorIndex)
        assert isinstance(sidx, InMemoryStructuredIndex)
        assert isinstance(vretr, VectorRetriever)
        assert isinstance(sretr, StructuredRetriever)

    @pytest.mark.parametrize(
        'hybrid_enabled,expected_vretr,expected_sretr',
        [
            matrix_param('hybrid_path',  True,  QdrantHybridRetriever, QdrantNoOpStructuredRetriever),
            matrix_param('classic_path', False, VectorRetriever,       StructuredRetriever),
        ],
    )
    def test_qdrant_path_wires_correct_backends( self, hybrid_enabled, expected_vretr, expected_sretr, config: AgentSearchConfig, ):
        # hybrid path needs prefetch_limit >= max(top_k); classic path is unconstrained.
        prefetch_limit = max(config.retrieval.vector.top_k, config.retrieval.structured.top_k) if hybrid_enabled else 100
        new_config = _replace_retrieval(config, hybrid_enabled=hybrid_enabled, prefetch_limit=prefetch_limit)
        dim = config.retrieval.vector.embedding_dim
        with patch('semantic_search.retrieval.qdrant_adapter._AsyncQdrantClient', return_value=_StubAsyncQdrantClient()):
            factory, vidx, sidx, vretr, sretr, *_ = _build_retrieval_backends(config=new_config, encoder=HashingEncoder(dim=dim, seed=7))
        assert factory is not None
        assert factory.available is True
        # Both paths use Qdrant indexes (the difference is in the retriever wrappers).
        assert isinstance(vidx, QdrantVectorIndex)
        assert isinstance(sidx, QdrantStructuredIndex)
        assert isinstance(vretr, expected_vretr)
        assert isinstance(sretr, expected_sretr)
        if hybrid_enabled:
            assert vretr.source == 'vector'
            assert sretr.source == 'structured'

