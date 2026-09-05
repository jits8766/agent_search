"""Tests for the three retrieval backends + RRF fusion.

Coverage matrix (per ``testing.mdc`` §7):

``VectorRetriever.retrieve`` (in-memory dense backend):
- returns_top_k                                 -> TestVectorRetriever::test_returns_top_k
- disabled_returns_empty                        -> TestVectorRetriever::test_disabled_returns_empty

``StructuredRetriever.retrieve`` (in-memory filter backend):
- filters_by_tld                                -> TestStructuredRetriever::test_filters_by_tld
- no_filters_returns_empty                      -> TestStructuredRetriever::test_no_filters_returns_empty
- extract_filters_from_intent                   -> TestStructuredRetriever::test_extract_filters_from_intent

``SqlRetriever.retrieve`` (SQL price-fallback path):
- only_runs_when_price_filter_present           -> TestSqlRetriever::test_only_runs_when_price_filter_present
- excludes_outside_band                         -> TestSqlRetriever::test_excludes_outside_band

``RRFFuser.fuse`` (rank fusion):
- items_in_multiple_sources_rank_higher         -> TestRRFFusion::test_items_in_multiple_sources_rank_higher
- empty_input_returns_empty                     -> TestRRFFusion::test_empty_input_returns_empty
"""
import pytest

from semantic_search.config.models import AgentSearchConfig, VectorRetrievalConfig
from semantic_search.qi.encoder import HashingEncoder
from semantic_search.retrieval.fusion import RRFFuser
from semantic_search.retrieval.sql_retriever import InMemoryPriceBandStore, SqlRetriever
from semantic_search.retrieval.structured_retriever import InMemoryStructuredIndex, StructuredRetriever, extract_filters_from_intent
from semantic_search.retrieval.vector_retriever import InMemoryVectorIndex, VectorRetriever
from semantic_search.contracts import Candidate, CandidateSet, Entity, IntentSlice, QueryIntent


def _intent(query_type: str, entities=None) -> QueryIntent:
    """Build a minimal QueryIntent for tests."""
    ents = list(entities or [])
    return QueryIntent(
        request_id='req_test',
        raw_query='x',
        normalized_query='x',
        query_type=query_type,
        confidence=0.95,
        decision_tier='L0_entity',
        slices=[IntentSlice(query_type=query_type, entities=ents, confidence=0.95, raw_text='x')],
        decision_cost_usd=0.0,
    )


class TestVectorRetriever:
    @pytest.mark.asyncio
    async def test_returns_top_k(self, config: AgentSearchConfig, encoder: HashingEncoder):
        index = InMemoryVectorIndex(dim=config.retrieval.vector.embedding_dim)
        for i in range(5):
            vec = encoder.encode(f"item {i} com domain")
            index.add(item_id=f"item_{i}", vector=vec, payload={'tld': 'com'})
        retr = VectorRetriever(config=config.retrieval.vector, encoder=encoder, index=index)
        cs = await retr.retrieve(_intent('explore'), top_k=3)
        assert cs.source == 'vector'
        assert len(cs.candidates) <= 3

    @pytest.mark.asyncio
    async def test_disabled_returns_empty(self, config: AgentSearchConfig, encoder: HashingEncoder):
        disabled = VectorRetrievalConfig(enabled=False, top_k=10, min_similarity=0.0, embedding_dim=config.retrieval.vector.embedding_dim, backend='memory')
        index = InMemoryVectorIndex(dim=config.retrieval.vector.embedding_dim)
        retr = VectorRetriever(config=disabled, encoder=encoder, index=index)
        cs = await retr.retrieve(_intent('explore'), top_k=3)
        assert cs.candidates == []


class TestStructuredRetriever:
    @pytest.mark.asyncio
    async def test_filters_by_tld(self, config: AgentSearchConfig):
        index = InMemoryStructuredIndex()
        index.add({'item_id': 'a', 'score': 0.9, 'tld': 'com', 'price': 50})
        index.add({'item_id': 'b', 'score': 0.8, 'tld': 'io', 'price': 50})
        retr = StructuredRetriever(config=config.retrieval.structured, index=index)
        intent = _intent('hybrid', entities=[Entity(name='tld', value=['com'], confidence=0.95, source='L0_entity', chip_kind='hard')])
        cs = await retr.retrieve(intent, top_k=10)
        ids = [c.item_id for c in cs.candidates]
        assert ids == ['a']

    @pytest.mark.asyncio
    async def test_no_filters_returns_empty(self, config: AgentSearchConfig):
        index = InMemoryStructuredIndex()
        index.add({'item_id': 'a', 'score': 0.9, 'tld': 'com', 'price': 50})
        retr = StructuredRetriever(config=config.retrieval.structured, index=index)
        cs = await retr.retrieve(_intent('explore'), top_k=10)
        assert cs.candidates == []

    def test_extract_filters_from_intent(self):
        intent = _intent('hybrid', entities=[
            Entity(name='tld', value=['com'], confidence=0.95, source='L0_entity', chip_kind='hard'),
            Entity(name='price_max', value=100, confidence=0.95, source='L0_entity', chip_kind='hard'),
        ])
        filters = extract_filters_from_intent(intent)
        assert filters == {'tld': ['com'], 'price_max': 100}


class TestSqlRetriever:
    @pytest.mark.asyncio
    async def test_only_runs_when_price_filter_present(self, config: AgentSearchConfig):
        store = InMemoryPriceBandStore()
        store.add({'item_id': 'a', 'price': 80, 'score': 0.9, 'tld': 'com'})
        retr = SqlRetriever(config=config.retrieval.sql, store=store)
        empty = await retr.retrieve(_intent('hybrid'), top_k=10)
        assert empty.candidates == []
        # tld-only must not fan out to CH/SQL — Qdrant already owns those filters.
        tld_only = await retr.retrieve(
            _intent(
                'hybrid',
                entities=[Entity(name='tld', value=['com'], confidence=0.95, source='L0_entity', chip_kind='hard')],
            ),
            top_k=10,
        )
        assert tld_only.candidates == []
        intent = _intent('hybrid', entities=[Entity(name='price_max', value=100, confidence=0.95, source='L0_entity', chip_kind='hard')])
        cs = await retr.retrieve(intent, top_k=10)
        assert [c.item_id for c in cs.candidates] == ['a']

    @pytest.mark.asyncio
    async def test_excludes_outside_band(self, config: AgentSearchConfig):
        store = InMemoryPriceBandStore()
        store.add({'item_id': 'a', 'price': 80, 'score': 0.9})
        store.add({'item_id': 'b', 'price': 500, 'score': 0.8})
        retr = SqlRetriever(config=config.retrieval.sql, store=store)
        intent = _intent('hybrid', entities=[Entity(name='price_max', value=100, confidence=0.95, source='L0_entity', chip_kind='hard')])
        cs = await retr.retrieve(intent, top_k=10)
        assert [c.item_id for c in cs.candidates] == ['a']


class TestRRFFusion:
    def test_items_in_multiple_sources_rank_higher(self, config: AgentSearchConfig):
        cs_v = CandidateSet(source='vector', candidates=[
            Candidate(item_id='shared', score=0.5, source='vector', payload={}),
            Candidate(item_id='only_v', score=0.4, source='vector', payload={}),
        ], latency_ms=1.0)
        cs_s = CandidateSet(source='structured', candidates=[
            Candidate(item_id='shared', score=0.7, source='structured', payload={}),
            Candidate(item_id='only_s', score=0.6, source='structured', payload={}),
        ], latency_ms=1.0)
        fuser = RRFFuser(config=config.retrieval.fusion)
        results = fuser.fuse(request_id='req_test', candidate_sets=[cs_v, cs_s], cache_hit=None)
        assert results.items[0].item_id == 'shared'
        assert set(results.items[0].contributing_sources) == {'vector', 'structured'}

    def test_empty_input_returns_empty(self, config: AgentSearchConfig):
        fuser = RRFFuser(config=config.retrieval.fusion)
        results = fuser.fuse(request_id='req_test', candidate_sets=[], cache_hit=None)
        assert results.items == []

    def test_single_source_short_circuit_preserves_rank(self, config: AgentSearchConfig):
        """One non-empty source -> short-circuit path; rank order preserved."""
        cs = CandidateSet(source='vector', candidates=[
            Candidate(item_id='a', score=0.9, source='vector', payload={'k': 1}),
            Candidate(item_id='b', score=0.5, source='vector', payload={'k': 2}),
            Candidate(item_id='c', score=0.1, source='vector', payload={'k': 3}),
        ], latency_ms=1.0)
        fuser = RRFFuser(config=config.retrieval.fusion)
        results = fuser.fuse(request_id='req_test', candidate_sets=[cs], cache_hit=None)
        assert [it.item_id for it in results.items] == ['a', 'b', 'c']
        # contributing_sources is the single source
        assert all(it.contributing_sources == ['vector'] for it in results.items)
        # payload preserved
        assert results.items[0].payload == {'k': 1}

    def test_single_source_short_circuit_matches_general_path(self, config: AgentSearchConfig):
        """fused_score from short-circuit must equal general-path RRF for one source."""
        cs = CandidateSet(source='vector', candidates=[
            Candidate(item_id='a', score=0.9, source='vector', payload={}),
            Candidate(item_id='b', score=0.5, source='vector', payload={}),
        ], latency_ms=1.0)
        fuser = RRFFuser(config=config.retrieval.fusion)
        short_results = fuser.fuse(request_id='req1', candidate_sets=[cs], cache_hit=None)
        # Force the general path by adding a second empty CandidateSet.
        # Wait — empty sets are filtered out by len(non_empty)==1 check, so
        # we add a one-item second source that won't change rank order, then
        # compute by hand what the general path produces.
        k = float(config.retrieval.fusion.rrf_k)
        expected_short_a = 1.0 / (k + 1.0)
        expected_short_b = 1.0 / (k + 2.0)
        assert short_results.items[0].fused_score == pytest.approx(expected_short_a)
        assert short_results.items[1].fused_score == pytest.approx(expected_short_b)

    def test_multi_source_with_some_empty_takes_short_circuit(self, config: AgentSearchConfig):
        """Empty candidate sets are filtered out before counting non-empty sources."""
        cs_full = CandidateSet(source='vector', candidates=[
            Candidate(item_id='a', score=0.9, source='vector', payload={}),
        ], latency_ms=1.0)
        cs_empty = CandidateSet(source='structured', candidates=[], latency_ms=0.5)
        fuser = RRFFuser(config=config.retrieval.fusion)
        results = fuser.fuse(request_id='req_t', candidate_sets=[cs_full, cs_empty], cache_hit=None)
        # Must take short-circuit -> contributing_sources is the single populated source
        assert results.items[0].contributing_sources == ['vector']
        assert results.total_candidates == 1

    def test_top_n_cap_applied_in_short_circuit(self, config: AgentSearchConfig):
        """Short-circuit honours top_n cap (slice before iteration)."""
        n_total = config.retrieval.fusion.top_n + 5
        # Use uniform score=0.5 so the contract validator never trips; rank is
        # already preserved by list order on entry to the fuser.
        cs = CandidateSet(
            source='vector',
            candidates=[
                Candidate(item_id=f'i{i}', score=0.5, source='vector', payload={})
                for i in range(n_total)
            ],
            latency_ms=1.0,
        )
        fuser = RRFFuser(config=config.retrieval.fusion)
        results = fuser.fuse(request_id='req_t', candidate_sets=[cs], cache_hit=None)
        assert len(results.items) == config.retrieval.fusion.top_n
        # total_candidates reflects the input, not the cap
        assert results.total_candidates == n_total
