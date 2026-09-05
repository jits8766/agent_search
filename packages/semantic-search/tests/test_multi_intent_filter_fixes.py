"""Validation for the multi-intent filter-leak fixes.

Covers three fixes:
  F1  _is_multi_intent no longer requires pre-set slice_id; _ensure_slice_ids
      mints ids so a genuine multi-slice intent takes the parallel + RRF path
      instead of the single path that collapses all slices' filters into one
      dict (last-slice-wins).
  F2  _apply_hard_chip_gate (precision-first) enforces EVERY payload-verifiable
      hard chip, not just tld/auction_type/price/name_length -- so a slot like
      keyword_contains / bids_min no longer leaks vector-origin items post-RRF.
  F3  In-memory VectorRetriever applies hard-chip filters to the ANN result, so
      the dense leg stops leaking items that violate an explicit hard filter.
"""
from typing import Any, List, Optional, Sequence, Tuple

import pytest

from semantic_search.config.models import AgentSearchConfig
from semantic_search.qi.encoder import HashingEncoder
from semantic_search.contracts import Entity, IntentSlice, QueryIntent, RankedItem
from semantic_search.orchestrator import SearchOrchestrator, _apply_hard_chip_gate
from semantic_search.retrieval.structured_retriever import extract_hard_filters_from_intent
from semantic_search.retrieval.vector_retriever import InMemoryVectorIndex, VectorRetriever


def _hard(name: str, value: Any) -> Entity:
    return Entity(name=name, value=value, confidence=0.95, source='L0_entity', chip_kind='hard')


def _multi_intent(slice_specs: Sequence[Tuple[str, Sequence[Entity]]], slice_ids: Optional[List[str]] = None) -> QueryIntent:
    """Build a multi-slice QueryIntent. slice_ids defaults to '' (unset) to
    reproduce the pre-fix condition where the multi path was skipped."""
    slices = []
    for i, (qtype, ents) in enumerate(slice_specs):
        sid = (slice_ids[i] if slice_ids else '')
        slices.append(IntentSlice(query_type=qtype, entities=list(ents), confidence=0.9, raw_text=f'sub{i}', slice_id=sid))
    return QueryIntent(
        request_id='req_mi', raw_query='x', normalized_query='x', query_type=slices[0].query_type,
        confidence=0.9, decision_tier='L0_multi_intent', slices=slices, decision_cost_usd=0.0,
    )


def _item(item_id: str, payload: dict, source: str = 'vector') -> RankedItem:
    return RankedItem(item_id=item_id, fused_score=1.0, contributing_sources=[source], payload=payload)


# ---------------------------------------------------------------- F1

def test_f1_multi_slice_without_slice_ids_is_multi_intent() -> None:
    """Two slices, both slice_id='' -> still multi-intent (pre-fix returned False)."""
    intent = _multi_intent([
        ('hybrid', [_hard('tld', ['com'])]),
        ('hybrid', [_hard('tld', ['io'])]),
    ])
    assert SearchOrchestrator._is_multi_intent(intent) is True


def test_f1_ensure_slice_ids_fills_empties_uniquely() -> None:
    intent = _multi_intent([
        ('hybrid', [_hard('tld', ['com'])]),
        ('hybrid', [_hard('tld', ['io'])]),
    ])
    SearchOrchestrator._ensure_slice_ids(intent)
    ids = [s.slice_id for s in intent.slices]
    assert all(ids) and len(set(ids)) == len(ids)


def test_f1_single_slice_is_not_multi_intent() -> None:
    intent = _multi_intent([('hybrid', [_hard('tld', ['com'])])])
    assert SearchOrchestrator._is_multi_intent(intent) is False


# ---------------------------------------------------------------- F2

def test_f2_gate_enforces_keyword_contains_hard_chip() -> None:
    """keyword_contains is outside the legacy 6-slot set; precision-first gate
    must drop a vector-origin item whose SLD lacks the term."""
    intent = _multi_intent([('hybrid', [_hard('keyword_contains', 'shop')])])
    items = [
        _item('a', {'domain_name': 'shopmart.com', 'tld': 'com'}),   # contains 'shop'
        _item('b', {'domain_name': 'techzone.com', 'tld': 'com'}),   # does not
    ]
    kept = _apply_hard_chip_gate(
        items, intent, drop_on_missing_field=True, keyword_match_mode='any',
    )
    assert [it.item_id for it in kept] == ['a']


def test_f2_gate_enforces_keyword_contains_exclude_hard_chip() -> None:
    """Negation chip must drop matching SLDs (chip/list parity)."""
    intent = _multi_intent([('hybrid', [_hard('keyword_contains_exclude', ['crypto'])])])
    items = [
        _item('a', {'domain_name': 'cryptowallet.com', 'tld': 'com', 'sld': 'cryptowallet'}),
        _item('b', {'domain_name': 'techzone.com', 'tld': 'com', 'sld': 'techzone'}),
    ]
    kept = _apply_hard_chip_gate(
        items, intent, drop_on_missing_field=True, keyword_match_mode='any',
    )
    assert [it.item_id for it in kept] == ['b']


def test_f2_gate_enforces_tld_and_type_exclude() -> None:
    # Numeric type id — label 'auction' expands to {16,38} and would wipe both.
    intent = _multi_intent([('hybrid', [
        _hard('tldExcludeList', ['ai']),
        _hard('typeExcludeList', ['16']),
    ])])
    items = [
        _item('a', {'domain_name': 'x.ai', 'tld': 'ai', 'auction_type': '16'}),
        _item('b', {'domain_name': 'y.com', 'tld': 'com', 'auction_type': '38'}),
        _item('c', {'domain_name': 'z.com', 'tld': 'com', 'auction_type': '16'}),
    ]
    kept = _apply_hard_chip_gate(
        items, intent, drop_on_missing_field=True, keyword_match_mode='any',
    )
    assert [it.item_id for it in kept] == ['b']


def test_f2_gate_enforces_bids_min_hard_chip() -> None:
    intent = _multi_intent([('hybrid', [_hard('bids_min', 5)])])
    items = [
        _item('a', {'domain_name': 'x.com', 'tld': 'com', 'bid_count': 10}),
        _item('b', {'domain_name': 'y.com', 'tld': 'com', 'bid_count': 1}),
        _item('c', {'domain_name': 'z.com', 'tld': 'com'}),  # missing -> drop (precision-first)
    ]
    kept = _apply_hard_chip_gate(
        items, intent, drop_on_missing_field=True, keyword_match_mode='any',
    )
    assert [it.item_id for it in kept] == ['a']


def test_f2_gate_noop_without_hard_chips() -> None:
    intent = _multi_intent([('explore', [])])
    items = [_item('a', {'tld': 'com'}), _item('b', {'tld': 'io'})]
    kept = _apply_hard_chip_gate(
        items, intent, drop_on_missing_field=True, keyword_match_mode='any',
    )
    assert [it.item_id for it in kept] == ['a', 'b']


def test_f2_extract_hard_filters_excludes_soft_and_rail_slots() -> None:
    intent = _multi_intent([('hybrid', [
        _hard('tld', ['com']),
        _hard('lifecycle_state', 'active'),  # rail selector -- excluded
        Entity(name='keyword_contains', value='soft', confidence=0.9, source='L0_entity', chip_kind='soft'),  # soft -- excluded
    ])])
    hf = extract_hard_filters_from_intent(intent)
    assert hf == {'tld': ['com']}


# ---------------------------------------------------------------- F3

@pytest.mark.asyncio
async def test_f3_vector_retriever_applies_hard_filter(config: AgentSearchConfig, encoder: HashingEncoder) -> None:
    index = InMemoryVectorIndex(dim=config.retrieval.vector.embedding_dim)
    # Same text so similarity is comparable; tld differs.
    index.add(item_id='com1', vector=encoder.encode('alpha domain'), payload={'tld': 'com', 'domain_name': 'alpha.com'})
    index.add(item_id='io1', vector=encoder.encode('alpha domain'), payload={'tld': 'io', 'domain_name': 'alpha.io'})
    retr = VectorRetriever(config=config.retrieval.vector, encoder=encoder, index=index)
    intent = _multi_intent([('hybrid', [_hard('tld', ['com'])])])
    cs = await retr.retrieve(intent, top_k=10)
    tlds = {c.payload['tld'] for c in cs.candidates}
    ids = {c.item_id for c in cs.candidates}
    assert tlds == {'com'}
    assert 'io1' not in ids


@pytest.mark.asyncio
async def test_f3_vector_retriever_unfiltered_when_no_hard_chip(config: AgentSearchConfig, encoder: HashingEncoder) -> None:
    index = InMemoryVectorIndex(dim=config.retrieval.vector.embedding_dim)
    index.add(item_id='com1', vector=encoder.encode('alpha domain'), payload={'tld': 'com'})
    index.add(item_id='io1', vector=encoder.encode('alpha domain'), payload={'tld': 'io'})
    retr = VectorRetriever(config=config.retrieval.vector, encoder=encoder, index=index)
    cs = await retr.retrieve(_multi_intent([('explore', [])]), top_k=10)
    assert {c.item_id for c in cs.candidates} == {'com1', 'io1'}
