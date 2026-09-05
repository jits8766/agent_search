"""Tests for typed contracts in semantic_search.contracts.

Validates that every dataclass enforces its invariants in ``__post_init__``.

Coverage matrix (per ``testing.mdc`` §7):

``Entity`` (``__post_init__``):
- valid_entity_constructs                       -> TestEntity::test_valid_entity
- empty_name_rejected                           -> TestEntity::test_empty_name
- invalid_confidence_rejected                   -> TestEntity::test_invalid_confidence
- invalid_source_rejected                       -> TestEntity::test_invalid_source

``IntentSlice`` (``__post_init__``):
- valid_slice_constructs                        -> TestIntentSlice::test_valid_slice
- invalid_query_type_rejected                   -> TestIntentSlice::test_invalid_query_type

``QueryIntent`` (``__post_init__`` + factory):
- valid_intent_constructs                       -> TestQueryIntent::test_valid_intent
- empty_slices_rejected                         -> TestQueryIntent::test_empty_slices_rejected
- negative_cost_rejected                        -> TestQueryIntent::test_negative_cost_rejected
- new_request_id_uniqueness                     -> TestQueryIntent::test_new_request_id_uniqueness

``Candidate`` (``__post_init__``):
- invalid_source_rejected                       -> TestCandidate::test_invalid_source
- score_out_of_range_rejected                   -> TestCandidate::test_score_out_of_range

``CandidateSet`` (``__post_init__``):
- source_mismatch_rejected                      -> TestCandidateSet::test_source_mismatch_rejected

``RankedItem`` (``__post_init__``):
- invalid_source_rejected                       -> TestRankedItem::test_invalid_source

``RankedResults`` (``__post_init__``):
- invalid_cache_tier_rejected                   -> TestRankedResults::test_invalid_cache_tier
- none_cache_tier_accepted                      -> TestRankedResults::test_none_cache_tier

``FeedbackSignal`` (``__post_init__`` + factory):
- invalid_signal_type_rejected                  -> TestFeedbackSignal::test_invalid_signal_type
- new_signal_id_format                          -> TestFeedbackSignal::test_new_signal_id_format
"""
import pytest

from semantic_search.contracts import Candidate, CandidateSet, Entity, FeedbackSignal, IntentSlice, PromptVersion, QueryIntent, RankedItem, RankedResults, infer_breaker_transition_fallback_kind
from semantic_search.core.exceptions import ValidationError
from semantic_search.nl_to_sql.contracts import AnalyticsResult


class TestEntity:
    def test_valid_entity(self):
        e = Entity(name='tld', value=['com'], confidence=0.9, source='L0_entity', chip_kind='hard')
        assert e.name == 'tld'
        assert e.chip_kind == 'hard'

    def test_invalid_confidence(self):
        with pytest.raises(ValidationError):
            Entity(name='tld', value=['com'], confidence=1.5, source='L0_entity', chip_kind='hard')

    def test_invalid_source(self):
        with pytest.raises(ValidationError):
            Entity(name='tld', value=['com'], confidence=0.9, source='not_a_tier', chip_kind='hard')

    def test_empty_name(self):
        with pytest.raises(ValidationError):
            Entity(name='', value=['com'], confidence=0.9, source='L0_entity', chip_kind='hard')

    def test_invalid_chip_kind_rejected(self):
        # Chip_kind must be one of CHIP_KINDS so downstream UX and the zero-result guard can branch deterministically on it.
        with pytest.raises(ValidationError, match="chip_kind"):
            Entity(name='tld', value=['com'], confidence=0.9, source='L0_entity', chip_kind='lukewarm')

    def test_soft_chip_kind_accepted(self):
        # Aspirational signals (brandable, quality, aesthetic) carry chip_kind='soft'.
        e = Entity(name='brandable', value=True, confidence=0.7, source='L2_llm', chip_kind='soft')
        assert e.chip_kind == 'soft'


class TestIntentSlice:
    def test_valid_slice(self):
        s = IntentSlice(query_type='hybrid', entities=[], confidence=0.95, raw_text='com')
        assert s.query_type == 'hybrid'

    def test_invalid_query_type(self):
        with pytest.raises(ValidationError):
            IntentSlice(query_type='nonsense', entities=[], confidence=0.95, raw_text='x')


class TestQueryIntent:
    def _slice(self):
        return IntentSlice(query_type='hybrid', entities=[], confidence=0.9, raw_text='x')

    def test_valid_intent(self):
        qi = QueryIntent(
            request_id='req_abc',
            raw_query='com',
            normalized_query='com',
            query_type='hybrid',
            confidence=0.9,
            decision_tier='L0_entity',
            slices=[self._slice()],
            decision_cost_usd=0.0,
        )
        assert qi.request_id == 'req_abc'

    def test_empty_slices_rejected(self):
        with pytest.raises(ValidationError):
            QueryIntent(
                request_id='req_abc',
                raw_query='x',
                normalized_query='x',
                query_type='hybrid',
                confidence=0.9,
                decision_tier='L0_entity',
                slices=[],
                decision_cost_usd=0.0,
            )

    def test_negative_cost_rejected(self):
        with pytest.raises(ValidationError):
            QueryIntent(
                request_id='req_abc',
                raw_query='x',
                normalized_query='x',
                query_type='hybrid',
                confidence=0.9,
                decision_tier='L0_entity',
                slices=[self._slice()],
                decision_cost_usd=-0.01,
            )

    def test_new_request_id_uniqueness(self):
        a = QueryIntent.new_request_id()
        b = QueryIntent.new_request_id()
        assert a != b
        assert a.startswith('req_')


class TestCandidate:
    def test_invalid_source(self):
        with pytest.raises(ValidationError):
            Candidate(item_id='x', score=0.5, source='not_real', payload={})

    def test_score_out_of_range(self):
        with pytest.raises(ValidationError):
            Candidate(item_id='x', score=1.5, source='vector', payload={})


class TestCandidateSet:
    def test_source_mismatch_rejected(self):
        c = Candidate(item_id='x', score=0.5, source='vector', payload={})
        with pytest.raises(ValidationError):
            CandidateSet(source='structured', candidates=[c], latency_ms=1.0)


class TestRankedItem:
    def test_invalid_source(self):
        with pytest.raises(ValidationError):
            RankedItem(item_id='x', fused_score=0.1, contributing_sources=['unknown'])


class TestRankedResults:
    def test_invalid_cache_tier(self):
        with pytest.raises(ValidationError):
            RankedResults(request_id='req', items=[], total_candidates=0, fusion_latency_ms=0.0, cache_hit='nonsense')

    def test_none_cache_tier(self):
        r = RankedResults(request_id='req', items=[], total_candidates=0, fusion_latency_ms=0.0, cache_hit=None)
        assert r.cache_hit is None

    @pytest.mark.parametrize('tier', ['exact', 'semantic', 'structured', 'intent_plan'])
    def test_valid_cache_hit_tiers(self, tier):
        r = RankedResults(request_id='req', items=[], total_candidates=0, fusion_latency_ms=0.0, cache_hit=tier)
        assert r.cache_hit == tier


class TestFeedbackSignal:
    def test_invalid_signal_type(self):
        with pytest.raises(ValidationError):
            FeedbackSignal(signal_id=FeedbackSignal.new_signal_id(), request_id='req', signal_type='bogus', payload={})

    def test_new_signal_id_format(self):
        sid = FeedbackSignal.new_signal_id()
        assert sid.startswith('sig_')

    def test_signal_origin_default_is_unknown(self):
        sig = FeedbackSignal(signal_id=FeedbackSignal.new_signal_id(), request_id='req', signal_type='filter_override')
        assert sig.signal_origin == 'unknown'

    def test_signal_origin_invalid_value_rejected(self):
        with pytest.raises(ValidationError, match="signal_origin must be one of"):
            FeedbackSignal(
                signal_id=FeedbackSignal.new_signal_id(),
                request_id='req',
                signal_type='filter_override',
                signal_origin='bogus_origin',
            )

    @pytest.mark.parametrize('origin', [
        'orchestrator', 'qi_engine', 'retrieval', 'analytics_router', 'cache',
        'eranker', 'circuit_breaker',
        'offline_eval', 'measurement', 'frontend', 'unknown',
    ])
    def test_signal_origin_accepts_every_enum_value(self, origin):
        sig = FeedbackSignal(
            signal_id=FeedbackSignal.new_signal_id(),
            request_id='req',
            signal_type='filter_override',
            signal_origin=origin,
        )
        assert sig.signal_origin == origin

    def test_infer_breaker_fallback_backend_unhealthy(self):
        assert infer_breaker_transition_fallback_kind('clickhouse', 'healthy', 'unhealthy', 'backend_health_transition') == 'fallback_active'

    def test_infer_breaker_fallback_circuit_half_open(self):
        assert infer_breaker_transition_fallback_kind('llm', 'open', 'half_open', 'cooldown_elapsed') == 'probe'

    def test_breaker_transition_rejects_unknown_fallback_kind(self):
        with pytest.raises(ValidationError, match='fallback_kind'):
            FeedbackSignal(
                signal_id=FeedbackSignal.new_signal_id(),
                request_id='req',
                signal_type='breaker_transition',
                payload={
                    'breaker': 'llm',
                    'from_state': 'closed',
                    'to_state': 'open',
                    'reason': 'x',
                    'observation_count': 1,
                    'fallback_kind': 'not_a_real_kind',
                },
                signal_origin='circuit_breaker',
            )

    def test_analytics_result_intent_record_id_default_empty(self):
        r = AnalyticsResult(
            request_id='ana_x', question='q', sql_hint='', success=True,
            failure_mode=None, failure_reason='', pruned_schema=None,
            generation=None, validation=None, execution=None, verifier=None,
            total_latency_ms=0.0,
        )
        assert r.intent_record_id == ''

    def test_prompt_version_baseline_scores_default_empty(self):
        v = PromptVersion(version_id='v.x', system_prompt='sys', user_template='usr')
        assert v.baseline_scores == {}
        assert v.ab_outcome == 'pending'
        assert v.rollback_target == ''

    def test_prompt_version_baseline_score_out_of_range_rejected(self):
        with pytest.raises(ValidationError, match="baseline_scores"):
            PromptVersion(version_id='v.x', system_prompt='sys', user_template='usr',
                          baseline_scores={'golden_pass_rate': 1.5})

    def test_prompt_version_ab_outcome_unknown_rejected(self):
        with pytest.raises(ValidationError, match="ab_outcome must be one of"):
            PromptVersion(version_id='v.x', system_prompt='sys', user_template='usr',
                          ab_outcome='bogus')

    def test_prompt_version_rollback_target_self_rejected(self):
        with pytest.raises(ValidationError, match="must not equal version_id"):
            PromptVersion(version_id='v.x', system_prompt='sys', user_template='usr',
                          rollback_target='v.x')

    def test_analytics_result_intent_record_id_validates_string_type(self):
        with pytest.raises(ValidationError, match="intent_record_id must be a string"):
            AnalyticsResult(
                request_id='ana_x', question='q', sql_hint='', success=True,
                failure_mode=None, failure_reason='', pruned_schema=None,
                generation=None, validation=None, execution=None, verifier=None,
                total_latency_ms=0.0,
                intent_record_id=12345,  # type: ignore[arg-type]
            )
