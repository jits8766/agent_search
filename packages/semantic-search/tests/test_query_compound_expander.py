"""Query-side compound expansion: expander unit behaviour, real splitter
integration, retriever wiring, and config parsing.

Guards query/document symmetry — the query encode text splits glued labels
(``techstartup`` -> ``tech startup``) via the SAME splitter the ingest pipeline
applied to document labels, so the sparse / ngram / dense legs match.
"""
import pytest

from semantic_search.config.models import CompoundSplitterConfig, RetrievalConfig, VectorRetrievalConfig
from semantic_search.contracts import Entity, IntentSlice, QueryIntent
from semantic_search.qi.encoder import HashingEncoder
from semantic_search.retrieval.query_compound_expander import QueryCompoundExpander
from semantic_search.retrieval.vector_retriever import InMemoryVectorIndex, VectorRetriever
from semantic_search.vectorization.compound_splitter import CompoundWordSplitter


def _intent(normalized, semantic_encode_text=None):
    return QueryIntent(
        request_id='req_qce',
        raw_query=normalized,
        normalized_query=normalized,
        query_type='hybrid',
        confidence=0.95,
        decision_tier='L0_entity',
        slices=[IntentSlice(query_type='hybrid', entities=[], confidence=0.95, raw_text=normalized)],
        decision_cost_usd=0.0,
        semantic_encode_text=semantic_encode_text,
    )


class _RecordingEncoder:
    def __init__(self, dim):
        self._inner = HashingEncoder(dim=dim, seed=7)
        self.texts = []

    @property
    def dim(self):
        return self._inner.dim

    def encode(self, text):
        self.texts.append(text)
        return self._inner.encode(text)

    async def encode_async(self, text):
        self.texts.append(text)
        return self._inner.encode(text)


# ---------------------------------------------------------------------------
# QueryCompoundExpander unit behaviour
# ---------------------------------------------------------------------------

class TestQueryCompoundExpander:
    def test_splits_glued_token_preserving_order(self):
        seg = {'techstartup': ('tech', 'startup')}
        exp = QueryCompoundExpander(lambda label: seg.get(label, (label,)))
        assert exp('techstartup coffee shop') == 'tech startup coffee shop'

    def test_passes_through_non_split_token(self):
        exp = QueryCompoundExpander(lambda label: (label,))
        assert exp('coffee shop') == 'coffee shop'

    def test_blank_input_unchanged(self):
        exp = QueryCompoundExpander(lambda label: (label,))
        assert exp('   ') == '   '
        assert exp('') == ''

    def test_segment_fn_exception_keeps_token_verbatim(self):
        def boom(label):
            raise RuntimeError('splitter glitch')
        exp = QueryCompoundExpander(boom)
        assert exp('alpha beta') == 'alpha beta'

    def test_empty_segments_keeps_token(self):
        exp = QueryCompoundExpander(lambda label: ())
        assert exp('alpha') == 'alpha'


# ---------------------------------------------------------------------------
# Real CompoundWordSplitter through the expander
# ---------------------------------------------------------------------------

class TestExpanderWithRealSplitter:
    def test_real_splitter_segments_glued_label(self):
        # min_segment_length=3 so 'tech'/'startup' qualify; oov cost high enough
        # that the in-dictionary two-word split beats the single OOV token.
        splitter = CompoundWordSplitter(
            dictionary={'tech': 100.0, 'startup': 100.0},
            min_segment_length=3,
            max_segments=4,
            oov_char_cost=20.0,
            length_penalty=0.0,
        )
        exp = QueryCompoundExpander(lambda label: splitter.split(label).segments)
        assert exp('techstartup').split() == ['tech', 'startup']


# ---------------------------------------------------------------------------
# Retriever applies the injected preprocessor
# ---------------------------------------------------------------------------

class TestRetrieverWiring:
    @pytest.mark.asyncio
    async def test_vector_retriever_applies_preprocessor_before_encode(self):
        enc = _RecordingEncoder(dim=32)
        cfg = VectorRetrievalConfig(enabled=True, top_k=5, min_similarity=0.0, embedding_dim=32, backend='memory')
        exp = QueryCompoundExpander(lambda label: {'techstartup': ('tech', 'startup')}.get(label, (label,)))
        retr = VectorRetriever(config=cfg, encoder=enc, index=InMemoryVectorIndex(dim=32), query_preprocessor=exp)
        await retr.retrieve(_intent('techstartup', semantic_encode_text='techstartup'), top_k=3)
        assert enc.texts == ['tech startup']

    @pytest.mark.asyncio
    async def test_none_preprocessor_embeds_verbatim(self):
        enc = _RecordingEncoder(dim=32)
        cfg = VectorRetrievalConfig(enabled=True, top_k=5, min_similarity=0.0, embedding_dim=32, backend='memory')
        retr = VectorRetriever(config=cfg, encoder=enc, index=InMemoryVectorIndex(dim=32), query_preprocessor=None)
        await retr.retrieve(_intent('techstartup', semantic_encode_text='techstartup'), top_k=3)
        assert enc.texts == ['techstartup']


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

def _retrieval_dict(extra=None):
    base = {
        'vector': {'enabled': True, 'top_k': 10, 'min_similarity': 0.0, 'embedding_dim': 32, 'backend': 'memory'},
        'structured': {'enabled': True, 'top_k': 10, 'backend': 'memory', 'word_count_filter_enabled': True, 'keyword_match_mode': 'any', 'unknown_selectable_fields': {}, 'lifecycle_auction_type_map': {}, 'traffic_signal_fields': []},
        'sql': {'enabled': False, 'top_k': 10, 'allowed_filter_columns': ['tld']},
        'fusion': {'rrf_k': 60, 'top_n': 50},
        'eranker': {'enabled': False, 'backend': 'noop', 'latency_budget_ms': 50, 'shadow_enabled': False, 'shadow_serve_fused': False, 'skip_when_backend_unhealthy': True},
        'diversity': {'enabled': False, 'backend': 'noop', 'top_n': 10, 'output_n': 10, 'latency_budget_ms': 50},
        'metrics': {'relevance_threshold': 0.5},
    }
    if extra:
        base.update(extra)
    return base


class TestConfigParsing:
    def test_absent_block_yields_none(self):
        cfg = RetrievalConfig.from_dict(_retrieval_dict())
        assert cfg.query_compound_split is None

    def test_present_block_parsed_as_compound_splitter_config(self):
        block = {
            'enabled': True,
            'dictionary_path': 'semantic_search/config/base.yaml',
            'min_segment_length': 2,
            'max_segments': 4,
            'oov_char_cost': 20.0,
            'length_penalty': 0.0,
        }
        cfg = RetrievalConfig.from_dict(_retrieval_dict({'query_compound_split': block}))
        assert isinstance(cfg.query_compound_split, CompoundSplitterConfig)
        assert cfg.query_compound_split.enabled is True
        assert cfg.query_compound_split.max_segments == 4
