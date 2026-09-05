"""Identified (pre-ground) vs applied_filters (grounded) response contract.

Full-search QI ``filters.identified`` reports as-identified hard chips.
``pipeline_trace.applied_filters`` reports inventory-grounded + available +
not guard-dropped chips (retrieval truth on ``IntentSlice.entities``).
"""
from __future__ import annotations

from semantic_search.app import (
    _build_applied_keywords,
    _build_filter_summary,
    _build_pipeline_trace,
    _build_query_intelligence,
)
from semantic_search.contracts import (
    Entity,
    IntentSlice,
    QueryIntent,
    SubIntentFilterSet,
    ZeroResultGuardOutcome,
)
from semantic_search.config.loader import load_config
from semantic_search.config.models import AgentSearchConfig
from semantic_search.qi.entity_reconcile import apply_post_merge_reconcile


def _ent(name: str, value: object, *, source: str = 'L0_llm', chip_kind: str = 'hard') -> Entity:
    return Entity(name=name, value=value, confidence=0.9, source=source, chip_kind=chip_kind)


def _intent(slice_: IntentSlice) -> QueryIntent:
    return QueryIntent(
        request_id='req_fs',
        raw_query='x',
        normalized_query='x',
        query_type=slice_.query_type,
        confidence=0.95,
        decision_tier='L2_llm',
        slices=[slice_],
        decision_cost_usd=0.0,
    )


_SOFT = frozenset({'keyword_contains', 'topic_primary'})
_SOFT_KEY = 'soft_signals'


class TestFilterSummaryGroundingSplit:
    def test_unknown_tld_stays_in_identified_absent_from_applied(self) -> None:
        """Inventory drop: identified keeps unknown TLD; applied does not."""
        pre = [_ent('tld', ['zz']), _ent('price_max', 100)]
        grounded = [_ent('price_max', 100)]  # tld dropped by EntityGrounder
        intent = _intent(IntentSlice(
            query_type='hybrid',
            entities=grounded,
            confidence=0.9,
            raw_text='.zz under 100',
            pre_ground_entities=pre,
        ))
        summary = _build_filter_summary(
            intent, soft_slot_names=_SOFT, soft_response_key=_SOFT_KEY,
        )
        id_names = {e['name'] for e in summary['identified']}
        assert 'tldIncludeList' in id_names
        assert 'maxPrice' in id_names
        applied_names = {e['name'] for e in summary['applied_filters']}
        assert 'tld' not in applied_names
        assert 'price_max' in applied_names
        assert 'grounding_applied' not in summary
        ungrounded = [e for e in summary['not_applied'] if e['reason'] == 'inventory_ungrounded']
        assert any(e['name'] == 'tld' for e in ungrounded)

        qi = _build_query_intelligence(intent, summary)
        assert set(qi['filters'].keys()) == {'identified', 'not_applied', 'soft_signals', 'keywords'}
        assert 'applied_filters' not in qi['filters']
        assert 'tldIncludeList' in {e['name'] for e in qi['filters']['identified']}
        assert all(set(e.keys()) <= {'name', 'value'} for e in qi['filters']['identified'])
        assert qi['multi_intent'] is False
        assert 'multi_intent_envelope' not in qi  # absent when no envelope passed
        assert 'sub_intent_filters' not in qi
        assert qi['decision_cost_usd'] == 0.0
        assert 'cache_hit' not in qi

        class _Out:
            applied = False
            skipped_reason = None

        class _Guard:
            fired = False
            ladder_step = 'none'

        trace = _build_pipeline_trace(summary, 'n/a', _Out(), _Guard(), result_count=0)
        assert set(trace.keys()) == {
            'applied_filters',
            'applied_keywords',
            'ranked_results_role',
        }
        assert trace['ranked_results_role'] == 'primary'
        assert {e['name'] for e in trace['applied_filters']} == {'price_max'}
        assert trace['applied_keywords'] == []
        assert all(isinstance(e.get('api_param'), (str, type(None))) for e in trace['applied_filters'])

    def test_pipeline_trace_includes_stages_when_configured(self) -> None:
        class _Attr:
            enabled = True
            include_in_response = True

        class _Meas:
            ranking_stage_attribution = _Attr()

        class _Cfg:
            measurement = _Meas()

        class _Orch:
            last_ranking_stages = {
                'cache_hit': None,
                'hard_gate_dropped': 2,
                'eranker_applied': False,
                'result_count': 3,
            }

        class _Sub:
            config = _Cfg()
            orchestrator = _Orch()

        class _Out:
            applied = False
            skipped_reason = None

        class _Guard:
            fired = False
            ladder_step = 'none'

        summary = {'applied_filters': []}
        trace = _build_pipeline_trace(
            summary, 'n/a', _Out(), _Guard(), result_count=3, sub=_Sub(),
        )
        assert 'stages' in trace
        assert trace['stages']['hard_gate_dropped'] == 2
        assert trace['stages']['result_count'] == 3

    def test_guard_drop_excluded_from_applied_not_from_identified(self) -> None:
        pre = [_ent('tld', ['io']), _ent('price_max', 50)]
        grounded = list(pre)  # both survived inventory
        intent = _intent(IntentSlice(
            query_type='hybrid',
            entities=grounded,
            confidence=0.9,
            raw_text='.io under 50',
            pre_ground_entities=pre,
        ))
        guard = ZeroResultGuardOutcome(
            fired=True,
            ladder_step='relax_filters',
            original_filter_count=2,
            relaxed_filter_count=1,
            relaxation_reason='zero_after_initial_retrieve',
            dropped_filter_names=['price_max'],
        )
        summary = _build_filter_summary(
            intent,
            soft_slot_names=_SOFT,
            soft_response_key=_SOFT_KEY,
            guard_outcome=guard,
        )
        id_names = {e['name'] for e in summary['identified']}
        assert 'tldIncludeList' in id_names
        assert 'maxPrice' in id_names
        applied_names = {e['name'] for e in summary['applied_filters']}
        assert applied_names == {'tld'}
        assert 'price_max' not in applied_names
        assert not any(e.get('guard_dropped') for e in summary['applied_filters'])
        assert any(
            e['name'] == 'price_max' and e['reason'] == 'filter_relaxed_by_guard'
            for e in summary['not_applied']
        )

    def test_backend_unsupported_not_in_applied(self) -> None:
        """Config-listed slots get not_applied reason backend_unsupported."""
        pre = [_ent('tld', ['com']), _ent('charPattern', 'cvcc')]
        grounded = list(pre)
        intent = _intent(IntentSlice(
            query_type='hybrid',
            entities=grounded,
            confidence=0.9,
            raw_text='cvcc .com',
            pre_ground_entities=pre,
        ))
        summary = _build_filter_summary(
            intent,
            soft_slot_names=_SOFT,
            soft_response_key=_SOFT_KEY,
            backend_unsupported_slots={'charPattern'},
        )
        assert 'tld' in {e['name'] for e in summary['applied_filters']}
        assert 'charPattern' not in {e['name'] for e in summary['applied_filters']}
        assert any(
            e['name'] == 'charPattern' and e['reason'] == 'backend_unsupported'
            for e in summary['not_applied']
        )

    def test_legacy_no_pre_ground_falls_back_to_entities(self) -> None:
        """Cache / legacy slices without snapshot: identified uses entities."""
        ents = [_ent('tld', ['com'])]
        intent = _intent(IntentSlice(
            query_type='hybrid',
            entities=ents,
            confidence=0.9,
            raw_text='.com',
            pre_ground_entities=None,
        ))
        summary = _build_filter_summary(
            intent, soft_slot_names=_SOFT, soft_response_key=_SOFT_KEY,
        )
        assert 'grounding_applied' not in summary
        assert {e['name'] for e in summary['identified']} == {'tldIncludeList'}
        assert {e['name'] for e in summary['applied_filters']} == {'tld'}

    def test_post_reconcile_has_hyphen_surfaces_as_exclude_hyphens_in_identified(self) -> None:
        """Cue reconcile after ground injects has_hyphen on entities — public identified must show it."""
        pre = [_ent('name_length_max', 5)]
        grounded = [_ent('name_length_max', 5), _ent('has_hyphen', False)]
        intent = _intent(IntentSlice(
            query_type='hybrid',
            entities=grounded,
            confidence=0.9,
            raw_text='short and clean',
            pre_ground_entities=pre,
        ))
        summary = _build_filter_summary(
            intent, soft_slot_names=_SOFT, soft_response_key=_SOFT_KEY,
        )
        identified = {e['name']: e['value'] for e in summary['identified']}
        assert identified['excludeHyphens'] is True
        assert identified['maxSldLen'] == 5

    def test_mixed_auction_type_ids_normalize_to_labels_in_identified(self) -> None:
        """Pre-ground label + grounded ID must not surface as ``25|backorder``."""
        pre = [_ent('auction_type', ['backorder'])]
        grounded = [_ent('auction_type', ['25'])]
        intent = _intent(IntentSlice(
            query_type='hybrid',
            entities=grounded,
            confidence=0.9,
            raw_text='backorder worthy domains',
            pre_ground_entities=pre,
        ))
        summary = _build_filter_summary(
            intent, soft_slot_names=_SOFT, soft_response_key=_SOFT_KEY,
        )
        identified = {e['name']: e['value'] for e in summary['identified']}
        assert identified['typeIncludeList'] == 'backorder'

    def test_short_brandable_injects_exclude_hyphens_via_char_rules(self) -> None:
        """letters-only / short+brandable cues -> excludeHyphens on public identified."""
        slots = AgentSearchConfig.from_dict(load_config()).qi.entity_slots
        assert slots is not None
        q = 'short brandable name not too pricey'
        out = apply_post_merge_reconcile(q, [], slots.hard_entity_set)
        names = {e.name for e in out}
        assert 'has_hyphen' in names
        assert 'has_number' in names
        intent = _intent(IntentSlice(
            query_type='hybrid',
            entities=[e for e in out if e.name in ('has_hyphen', 'has_number', 'name_length_max')],
            confidence=0.9,
            raw_text=q,
            pre_ground_entities=list(out),
        ))
        summary = _build_filter_summary(
            intent, soft_slot_names=_SOFT, soft_response_key=_SOFT_KEY,
        )
        identified = {e['name']: e['value'] for e in summary['identified']}
        assert identified.get('excludeHyphens') is True
        assert identified.get('excludeDigits') is True

    def test_multi_slice_identified_is_union_not_concat(self) -> None:
        """Same hard chips on 2 slices must not duplicate in top-level identified."""
        pre = [_ent('price_max', 99), _ent('filterPriceCurrency', 'USD')]
        grounded = list(pre)
        s1 = IntentSlice(
            query_type='hybrid',
            entities=grounded,
            confidence=0.9,
            raw_text='coffee below $100',
            slice_id='slc_a',
            pre_ground_entities=pre,
        )
        s2 = IntentSlice(
            query_type='hybrid',
            entities=list(grounded),
            confidence=0.9,
            raw_text='pizza below $100',
            slice_id='slc_b',
            pre_ground_entities=list(pre),
        )
        intent = QueryIntent(
            request_id='req_multi',
            raw_query='coffee and pizza below 100',
            normalized_query='coffee and pizza below 100',
            query_type='hybrid',
            confidence=0.94,
            decision_tier='L0_multi_intent',
            slices=[s1, s2],
            decision_cost_usd=0.0,
            sub_intent_filters=[
                SubIntentFilterSet(sub_query='coffee below $100', entities=list(pre)),
                SubIntentFilterSet(sub_query='pizza below $100', entities=list(pre)),
            ],
        )
        summary = _build_filter_summary(
            intent, soft_slot_names=_SOFT, soft_response_key=_SOFT_KEY,
            backend_unsupported_slots={'filterPriceCurrency'},
        )
        id_names = [e['name'] for e in summary['identified']]
        assert id_names.count('maxPrice') == 1
        assert id_names.count('filterPriceCurrency') == 1
        na = [e for e in summary['not_applied'] if e['name'] == 'filterPriceCurrency']
        assert len(na) == 1
        assert na[0]['reason'] == 'backend_unsupported'

        qi = _build_query_intelligence(
            intent, summary, soft_slot_names=set(_SOFT),
        )
        assert qi['multi_intent'] is True
        assert len(qi['sub_intent_filters']) == 2
        for sif in qi['sub_intent_filters']:
            assert set(sif.keys()) == {'sub_query', 'identified'}
            assert 'entities' not in sif
            assert 'maxPrice' in {e['name'] for e in sif['identified']}
        # Top-level remains the union (still single maxPrice).
        assert [e['name'] for e in qi['filters']['identified']].count('maxPrice') == 1

    def test_applied_keywords_excludes_soft_signal_dupes(self) -> None:
        """Keywords already listed as soft_signals values must not reappear."""
        kws = [
            {'term': 'coffee', 'probability': 0.95},
            {'term': 'pizza', 'probability': 0.93},
        ]
        soft = [{'name': 'keyword_contains', 'value': 'coffee'}]
        applied = _build_applied_keywords(kws, soft, soft_apply_mode='rank')
        terms = [e['term'] for e in applied]
        assert terms == ['pizza']
        assert applied[0]['roles'] == ['encode', 'soft_boost']
