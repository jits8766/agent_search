"""Unit tests for SoftKeywordApplier — independent soft-signal apply subcomponent."""
from __future__ import annotations

from typing import Any, List, Optional

import pytest

from semantic_search.config.models import QIEntitySlotsConfig
from semantic_search.contracts import Entity, IntentSlice, QueryIntent, RankedItem, RankedResults
from semantic_search.retrieval.soft_keyword_apply import SoftKeywordApplier


def _slots(**overrides: Any) -> QIEntitySlotsConfig:
    base = dict(
        soft_slot_names=[
            'keyword_contains',
            'keyword_starts_with',
            'topic_include',
            'topic_exclude',
            'similar_to',
            'word_count_min',
        ],
        hard_entity_names=['tld', 'price_max'],
        soft_response_key='soft_signals',
        soft_group_tag='soft',
        soft_apply_mode='rank',
        soft_rank_boost_weight=0.15,
        soft_rank_slot_names=[
            'keyword_contains',
            'keyword_starts_with',
            'topic_include',
            'topic_exclude',
            'similar_to',
            'word_count_min',
        ],
        soft_rank_miss_penalty_ratio=0.25,
        soft_rank_partial_boost_ratio=0.5,
    )
    base.update(overrides)
    return QIEntitySlotsConfig(**base)


def _entity(name: str, value: Any, *, soft: bool = True) -> Entity:
    return Entity(
        name=name,
        value=value,
        confidence=0.9,
        source='L0_llm',
        chip_kind='soft' if soft else 'hard',
    )


def _intent(
    hard: Optional[List[Entity]] = None,
    soft: Optional[List[Entity]] = None,
) -> QueryIntent:
    return QueryIntent(
        request_id='req-test',
        intent_record_id='rec-test',
        raw_query='domains containing cloud',
        normalized_query='domains containing cloud',
        query_type='hybrid',
        confidence=0.9,
        decision_tier='L0_entity',
        decision_cost_usd=0.0,
        slices=[IntentSlice(
            query_type='hybrid',
            entities=list(hard or []),
            confidence=0.9,
            raw_text='domains containing cloud',
            soft_entities=list(soft or []),
        )],
    )


@pytest.fixture
def applier() -> SoftKeywordApplier:
    return SoftKeywordApplier(_slots(), default_keyword_match_mode='any')


def test_mode_and_rank_slots_from_config(applier: SoftKeywordApplier) -> None:
    assert applier.mode == 'rank'
    assert 'keyword_contains' in applier.rank_slot_names
    assert 'topic_include' in applier.rank_slot_names
    assert 'similar_to' in applier.rank_slot_names


def test_prepare_intent_strips_soft_from_entities(applier: SoftKeywordApplier) -> None:
    soft = [_entity('keyword_contains', 'cloud'), _entity('topic_include', ['tech'])]
    hard = [
        _entity('tld', ['com'], soft=False),
        _entity('keyword_contains', 'cloud'),
        _entity('topic_include', ['tech']),  # soft signal must not stay as hard filter
    ]
    intent, soft_kw = applier.prepare_intent(_intent(hard=hard, soft=soft))
    assert {e.name for e in soft_kw} == {'keyword_contains', 'topic_include'}
    hard_names = {e.name for e in intent.slices[0].entities}
    assert hard_names == {'tld'}
    assert 'topic_include' not in hard_names
    soft_names = {e.name for e in intent.slices[0].soft_entities}
    assert soft_names == {'keyword_contains', 'topic_include'}


def test_prepare_intent_preserves_pre_ground_entities(applier: SoftKeywordApplier) -> None:
    """Soft strip must keep IntentSlice.pre_ground_entities for identified filters."""
    pre = [_entity('tld', ['com'], soft=False), _entity('price_max', 100, soft=False)]
    hard = [
        _entity('tld', ['com'], soft=False),
        _entity('keyword_contains', 'cloud'),
    ]
    base = _intent(hard=hard, soft=[])
    base.slices[0] = IntentSlice(
        query_type='hybrid',
        entities=list(hard),
        confidence=0.9,
        raw_text=base.raw_query,
        soft_entities=[],
        pre_ground_entities=list(pre),
    )
    intent, _ = applier.prepare_intent(base)
    assert intent.slices[0].pre_ground_entities is not None
    assert [e.name for e in intent.slices[0].pre_ground_entities] == ['tld', 'price_max']
    assert {e.name for e in intent.slices[0].entities} == {'tld'}


def test_prepare_intent_migrates_soft_only_on_hard(applier: SoftKeywordApplier) -> None:
    """Soft parked only on hard entities must move to soft_entities (not dropped)."""
    hard = [
        _entity('tld', ['com'], soft=False),
        _entity('keyword_contains', 'cloud'),
        _entity('word_count_min', 1),
    ]
    intent, soft_kw = applier.prepare_intent(_intent(hard=hard, soft=[]))
    assert {e.name for e in intent.slices[0].entities} == {'tld'}
    soft_names = {e.name for e in intent.slices[0].soft_entities}
    assert soft_names == {'keyword_contains', 'word_count_min'}
    assert [e.name for e in soft_kw] == ['keyword_contains', 'word_count_min']


def test_match_score_boosts_contains(applier: SoftKeywordApplier) -> None:
    soft = [_entity('keyword_contains', 'cloud')]
    hit = applier.match_score('cloudhost', soft)
    miss = applier.match_score('finance', soft)
    assert hit > 0
    assert miss < 0


def test_match_score_topic_and_similar_to(applier: SoftKeywordApplier) -> None:
    topic = [_entity('topic_include', ['climate_tech'])]
    assert applier.match_score('climatehub', topic) > 0
    assert applier.match_score('finance', topic) < 0
    similar = [_entity('similar_to', ['stripe.com'])]
    assert applier.match_score('stripepay', similar) > 0
    assert applier.match_score('finance', similar) < 0


def test_apply_rank_boost_reorders(applier: SoftKeywordApplier) -> None:
    soft = [_entity('keyword_contains', 'cloud')]
    items = [
        RankedItem(
            item_id='a.com', fused_score=1.0, contributing_sources=['vector'],
            payload={'domain_name': 'finance.com', 'sld': 'finance'},
        ),
        RankedItem(
            item_id='b.com', fused_score=0.9, contributing_sources=['vector'],
            payload={'domain_name': 'cloudhost.com', 'sld': 'cloudhost'},
        ),
    ]
    results = RankedResults(request_id='r1', items=items, total_candidates=2, fusion_latency_ms=1.0)
    out = applier.apply_rank_boost(results, soft)
    assert out.items[0].item_id == 'b.com'
    assert out.items[0].fused_score > out.items[1].fused_score


def test_apply_rank_boost_clamps_negative_fused_score(applier: SoftKeywordApplier) -> None:
    """Miss penalty on low base score must not violate RankedItem.fused_score >= 0."""
    soft = [_entity('word_count_min', 1), _entity('word_count_max', 1)]
    items = [
        RankedItem(
            item_id='a.com', fused_score=0.01, contributing_sources=['vector'],
            payload={'domain_name': 'many-word-domain-name.com', 'sld': 'many-word-domain-name'},
        ),
    ]
    results = RankedResults(request_id='r1', items=items, total_candidates=1, fusion_latency_ms=1.0)
    out = applier.apply_rank_boost(results, soft)
    assert out.items[0].fused_score >= 0.0


def test_apply_rank_boost_noop_when_mode_off() -> None:
    applier = SoftKeywordApplier(_slots(soft_apply_mode='off'), default_keyword_match_mode='any')
    soft = [_entity('keyword_contains', 'cloud')]
    items = [
        RankedItem(
            item_id='a.com', fused_score=1.0, contributing_sources=['vector'],
            payload={'sld': 'finance'},
        ),
        RankedItem(
            item_id='b.com', fused_score=0.9, contributing_sources=['vector'],
            payload={'sld': 'cloudhost'},
        ),
    ]
    results = RankedResults(request_id='r1', items=items, total_candidates=2, fusion_latency_ms=1.0)
    out = applier.apply_rank_boost(results, soft)
    assert out.items[0].item_id == 'a.com'


def test_multi_keyword_f1_ranks_dense_combo_above_singles(
    applier: SoftKeywordApplier,
) -> None:
    """Dense dual-keyword SLD (high F1) ranks above single-term and miss."""
    keywords = [
        {'term': 'coffee', 'probability': 0.95},
        {'term': 'pizza', 'probability': 0.80},
    ]
    items = [
        RankedItem(
            item_id='none.com', fused_score=1.0, contributing_sources=['vector'],
            payload={'sld': 'finance'},
        ),
        RankedItem(
            item_id='pizza.com', fused_score=0.95, contributing_sources=['vector'],
            payload={'sld': 'pizzahub'},
        ),
        RankedItem(
            item_id='coffee.com', fused_score=0.90, contributing_sources=['vector'],
            payload={'sld': 'coffeeshop'},
        ),
        RankedItem(
            item_id='combo.com', fused_score=0.70, contributing_sources=['vector'],
            payload={'sld': 'coffeepizza'},
        ),
    ]
    results = RankedResults(
        request_id='r1', items=items, total_candidates=4, fusion_latency_ms=1.0,
    )
    out = applier.apply_rank_boost(results, [], keywords)
    ids = [it.item_id for it in out.items]
    assert ids[0] == 'combo.com'
    assert ids[-1] == 'none.com'
    assert set(ids[1:3]) == {'pizza.com', 'coffee.com'}


def test_multi_keyword_high_f1_single_beats_low_f1_dual(
    applier: SoftKeywordApplier,
) -> None:
    """Single with higher F1 ranks above a noisy dual-keyword SLD."""
    keywords = [
        {'term': 'crypto', 'probability': 0.95},
        {'term': 'app', 'probability': 0.90},
    ]
    items = [
        RankedItem(
            item_id='noisy_dual.com', fused_score=0.99, contributing_sources=['vector'],
            payload={'sld': 'bestcryptoappsmarketplaceextra'},
        ),
        RankedItem(
            item_id='exact_single.com', fused_score=0.40, contributing_sources=['vector'],
            payload={'sld': 'crypto'},
        ),
    ]
    results = RankedResults(
        request_id='r1', items=items, total_candidates=2, fusion_latency_ms=1.0,
    )
    out = applier.apply_rank_boost(results, [], keywords)
    assert [it.item_id for it in out.items] == ['exact_single.com', 'noisy_dual.com']


def test_multi_keyword_equal_f1_prefers_higher_hit_count(
    applier: SoftKeywordApplier,
) -> None:
    """When F1 ties, more matched keywords rank higher."""
    keywords = [
        {'term': 'crypto', 'probability': 0.95},
        {'term': 'app', 'probability': 0.90},
    ]
    # dual: R=1.0 P=0.5 -> F1=2/3; single: R=0.5 P=1.0 -> F1=2/3
    items = [
        RankedItem(
            item_id='single.com', fused_score=0.90, contributing_sources=['vector'],
            payload={'sld': 'crypto'},
        ),
        RankedItem(
            item_id='dual.com', fused_score=0.10, contributing_sources=['vector'],
            payload={'sld': 'cryptoappxx'},  # covered 9/11? crypto+app=9, len=11 -> P=9/11
        ),
    ]
    # Build an exact P=0.5 dual: compact length = 2 * covered => covered 6 of 12
    # "cryptoXXXXXX" with app not there - need both terms. "appcryptoXXXX" = 3+6=9 of 13
    # For P=0.5 with both: covered = len(crypto)+len(app)=9, need compact len=18
    items = [
        RankedItem(
            item_id='single.com', fused_score=0.90, contributing_sources=['vector'],
            payload={'sld': 'crypto'},
        ),
        RankedItem(
            item_id='dual.com', fused_score=0.10, contributing_sources=['vector'],
            payload={'sld': 'cryptoappxxxxxxxxx'},  # 9/18 = 0.5
        ),
    ]
    results = RankedResults(
        request_id='r1', items=items, total_candidates=2, fusion_latency_ms=1.0,
    )
    out = applier.apply_rank_boost(results, [], keywords)
    assert [it.item_id for it in out.items] == ['dual.com', 'single.com']


def test_multi_keyword_f1_orders_three_two_one(
    applier: SoftKeywordApplier,
) -> None:
    """Higher F1 from fuller keyword coverage ranks first."""
    keywords = [
        {'term': 'coffee', 'probability': 0.95},
        {'term': 'pizza', 'probability': 0.90},
        {'term': 'bagel', 'probability': 0.85},
    ]
    items = [
        RankedItem(
            item_id='none.com', fused_score=1.0, contributing_sources=['vector'],
            payload={'sld': 'finance'},
        ),
        RankedItem(
            item_id='one.com', fused_score=0.95, contributing_sources=['vector'],
            payload={'sld': 'coffeeshop'},
        ),
        RankedItem(
            item_id='two.com', fused_score=0.50, contributing_sources=['vector'],
            payload={'sld': 'pizzabagel'},
        ),
        RankedItem(
            item_id='three.com', fused_score=0.10, contributing_sources=['vector'],
            payload={'sld': 'bagelcoffeepizza'},
        ),
        RankedItem(
            item_id='two_rev.com', fused_score=0.40, contributing_sources=['vector'],
            payload={'sld': 'coffeepizza'},
        ),
    ]
    results = RankedResults(
        request_id='r1', items=items, total_candidates=5, fusion_latency_ms=1.0,
    )
    out = applier.apply_rank_boost(results, [], keywords)
    ids = [it.item_id for it in out.items]
    assert ids[0] == 'three.com'
    assert set(ids[1:3]) == {'two.com', 'two_rev.com'}
    assert ids[3] == 'one.com'
    assert ids[4] == 'none.com'


def test_multi_keyword_order_independent_and_precision_first(
    applier: SoftKeywordApplier,
) -> None:
    """Term order in SLD ignored; denser dual beats noisy dual and single."""
    keywords = [
        {'term': 'crypto', 'probability': 0.95},
        {'term': 'app', 'probability': 0.90},
    ]
    items = [
        RankedItem(
            item_id='noisy.com', fused_score=0.99, contributing_sources=['vector'],
            payload={'sld': 'bestcryptoappsmarketplaceextra'},
        ),
        RankedItem(
            item_id='rev.com', fused_score=0.40, contributing_sources=['vector'],
            payload={'sld': 'appcrypto'},
        ),
        RankedItem(
            item_id='fwd.com', fused_score=0.41, contributing_sources=['vector'],
            payload={'sld': 'cryptoapp'},
        ),
        RankedItem(
            item_id='single.com', fused_score=0.95, contributing_sources=['vector'],
            payload={'sld': 'cryptowallet'},
        ),
    ]
    results = RankedResults(
        request_id='r1', items=items, total_candidates=4, fusion_latency_ms=1.0,
    )
    out = applier.apply_rank_boost(results, [], keywords)
    ids = [it.item_id for it in out.items]
    assert ids[0] in ('fwd.com', 'rev.com')
    assert ids[1] in ('fwd.com', 'rev.com')
    assert set(ids[:2]) == {'fwd.com', 'rev.com'}
    # Noisy dual and single both below dense duals; relative order is F1-driven.
    assert 'noisy.com' in ids[2:]
    assert 'single.com' in ids[2:]
