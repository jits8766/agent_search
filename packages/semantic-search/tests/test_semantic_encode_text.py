"""TLD-safe semantic-encode-text: builder, accessor, contract field, retriever wiring.

Guards the invariant that the TLD literal is matched ONLY as an exact structured
filter and never reaches any embedding / sparse encode leg. See
``semantic_search.qi.residual_extractor`` (build_semantic_encode_text /
semantic_encode_text_for) and the two retrievers that consume the accessor.
"""
import pytest

from semantic_search.contracts import Entity, IntentSlice, QueryIntent
from semantic_search.core.exceptions import ValidationError
from semantic_search.qi.encoder import HashingEncoder
from semantic_search.qi.residual_extractor import build_semantic_encode_text, semantic_encode_text_for
from semantic_search.retrieval.vector_retriever import InMemoryVectorIndex, VectorRetriever
from semantic_search.config.models import VectorRetrievalConfig


def _tld(value):
    return Entity(name='tld', value=value, confidence=0.95, source='L0_entity', chip_kind='hard')


def _intent(normalized, entities=None, semantic_query=None, semantic_encode_text=None):
    ents = list(entities or [])
    return QueryIntent(
        request_id='req_set',
        raw_query=normalized,
        normalized_query=normalized,
        query_type='hybrid',
        confidence=0.95,
        decision_tier='L0_entity',
        slices=[IntentSlice(query_type='hybrid', entities=ents, confidence=0.95, raw_text=normalized)],
        decision_cost_usd=0.0,
        semantic_query=semantic_query,
        semantic_encode_text=semantic_encode_text,
    )


class _RecordingEncoder:
    """Wraps HashingEncoder, recording every text handed to encode/encode_async."""

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
# build_semantic_encode_text
# ---------------------------------------------------------------------------

class TestBuildSemanticEncodeText:
    def test_returns_residual_when_present(self):
        out = build_semantic_encode_text('coffee shop domains ending in io', [_tld(['io'])], 'coffee shop')
        assert out == 'coffee shop'

    def test_strips_tld_literal_from_normalized_fallback(self):
        out = build_semantic_encode_text('coffee shop domains ending in io', [_tld(['io'])], None)
        assert 'io' not in out.split()
        assert out == 'coffee shop domains ending in'

    def test_strips_dotted_and_excluded_tld_tokens(self):
        ents = [_tld(['io', 'com']), Entity(name='tldExcludeList', value=['net'], confidence=0.9, source='L0_entity', chip_kind='hard')]
        out = build_semantic_encode_text('tech io com names not net', ents, None)
        toks = out.split()
        assert 'io' not in toks and 'com' not in toks and 'net' not in toks
        assert 'tech' in toks and 'names' in toks

    def test_degenerate_tld_only_query_returns_normalized(self):
        # Nothing but the TLD token — no other signal to embed; returned verbatim.
        assert build_semantic_encode_text('io', [_tld(['io'])], None) == 'io'

    def test_empty_normalized_returns_empty(self):
        assert build_semantic_encode_text('', [_tld(['io'])], None) == ''

    def test_no_entities_returns_normalized_unchanged(self):
        assert build_semantic_encode_text('blue widgets', [], None) == 'blue widgets'


# ---------------------------------------------------------------------------
# semantic_encode_text_for accessor — precedence + fallback
# ---------------------------------------------------------------------------

class TestSemanticEncodeTextFor:
    def test_prefers_contract_field(self):
        intent = _intent('cheap io domains', [_tld(['io'])], semantic_query='ignored', semantic_encode_text='coffee shop')
        assert semantic_encode_text_for(intent) == 'coffee shop'

    def test_falls_back_to_semantic_query(self):
        intent = _intent('cheap io domains', [_tld(['io'])], semantic_query='coffee shop', semantic_encode_text=None)
        assert semantic_encode_text_for(intent) == 'coffee shop'

    def test_derives_tld_stripped_text_from_slices_when_field_absent(self):
        # Wrapper / cache / legacy intent: neither field set, but the TLD must
        # still never survive into the embed text.
        intent = _intent('cheap io domains', [_tld(['io'])], semantic_query=None, semantic_encode_text=None)
        out = semantic_encode_text_for(intent)
        assert 'io' not in out.split()
        assert out == 'cheap domains'


# ---------------------------------------------------------------------------
# Contract field validation
# ---------------------------------------------------------------------------

class TestContractField:
    def test_accepts_str_and_none(self):
        assert _intent('x', semantic_encode_text='ok').semantic_encode_text == 'ok'
        assert _intent('x', semantic_encode_text=None).semantic_encode_text is None

    def test_rejects_non_str(self):
        with pytest.raises(ValidationError):
            _intent('x', semantic_encode_text=123)


# ---------------------------------------------------------------------------
# VectorRetriever encodes the TLD-safe text
# ---------------------------------------------------------------------------

class TestVectorRetrieverEncodeText:
    @pytest.mark.asyncio
    async def test_encodes_contract_field_text(self):
        enc = _RecordingEncoder(dim=32)
        cfg = VectorRetrievalConfig(enabled=True, top_k=5, min_similarity=0.0, embedding_dim=32, backend='memory')
        retr = VectorRetriever(config=cfg, encoder=enc, index=InMemoryVectorIndex(dim=32))
        intent = _intent('coffee shop domains in io', [_tld(['io'])], semantic_encode_text='coffee shop')
        await retr.retrieve(intent, top_k=3)
        assert enc.texts == ['coffee shop']

    @pytest.mark.asyncio
    async def test_strips_tld_on_fallback_path(self):
        enc = _RecordingEncoder(dim=32)
        cfg = VectorRetrievalConfig(enabled=True, top_k=5, min_similarity=0.0, embedding_dim=32, backend='memory')
        retr = VectorRetriever(config=cfg, encoder=enc, index=InMemoryVectorIndex(dim=32))
        # No residual and no contract field → accessor derives from slices.
        intent = _intent('cheap io domains', [_tld(['io'])], semantic_query=None, semantic_encode_text=None)
        await retr.retrieve(intent, top_k=3)
        assert enc.texts and 'io' not in enc.texts[0].split()
