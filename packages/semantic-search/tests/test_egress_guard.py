"""Composition tests for EgressGuard + NoOpEgressGuard.

Covers:
    - Construction-time validation (None / wrong-type sub-deps rejected).
- ``NoOpEgressGuard`` preserves the input verbatim.
- PII scrub: hit / miss / oversize-truncated / nested list+dict shapes.
- Moderation: drop policy removes the item, mask policy redacts fields and
  keeps the item.
- Grounding check: ungrounded citations are masked at the span; grounded
  citations are preserved.
- Action precedence: dropped > moderation_masked > pii_masked >
  explanation_scrubbed > kept.
- Per-item exception is fail-closed (drops the item with a typed reason).
- Outcome counters always reconcile with the per-item decision list.
"""
import pytest

from semantic_search.config.models import EgressGuardConfig, GroundingCheckConfig, LexicalModeratorConfig, ModeratorConfig, PIIScrubConfig, SanitizerConfig
from semantic_search.contracts import RankedItem, RankedResults
from semantic_search.core.exceptions import EgressGuardError
from semantic_search.safety.layer_zero_sanitizer import LayerZeroSanitizer
from semantic_search.safety.egress_guard import EgressGuard, NoOpEgressGuard
from semantic_search.safety.lexical_blocklist_moderator import LexicalBlocklistModerator, Moderator, ModeratorVerdict, NoOpModerator


# --------------------------------------------------------------------------- #
# Helpers / fixtures                                                           #
# --------------------------------------------------------------------------- #

def _sanitizer(pii_patterns=None) -> LayerZeroSanitizer:
    cfg = SanitizerConfig(
        enabled=True,
        max_chars=4096,
        system_max_chars=50000,
        blocked_patterns=[],
        # Email + a fake "credit-card-style" 16-digit pattern. Both are real
        # regex patterns that the LayerZeroSanitizer will mask via subn().
        pii_patterns=pii_patterns if pii_patterns is not None else [
            r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',
            r'\b\d{16}\b',
        ],
        applies_to_llm_ingress=False,
    )
    return LayerZeroSanitizer(cfg)


def _lex_cfg(banned_terms=('spam',)) -> LexicalModeratorConfig:
    return LexicalModeratorConfig(
        payload_fields=['title', 'description'],
        banned_terms=list(banned_terms),
        min_term_length=3,
        max_terms=1000,
        stopwords=[],
    )


def _egress_cfg(
    enabled=True,
    pii_enabled=True,
    pii_fields=('title', 'description', 'explanation'),
    moderator_enabled=True,
    moderator_policy='drop',
    moderator_backend='lexical_blocklist',
    grounding_enabled=True,
) -> EgressGuardConfig:
    pii = PIIScrubConfig(
        enabled=pii_enabled,
        payload_fields=list(pii_fields),
        max_field_chars=64,
    )
    lex = _lex_cfg() if (moderator_enabled and moderator_backend == 'lexical_blocklist') else None
    mod = ModeratorConfig(
        enabled=moderator_enabled,
        backend=moderator_backend,
        policy=moderator_policy,
        lexical=lex,
    )
    ground = GroundingCheckConfig(
        enabled=grounding_enabled,
        explanation_field='explanation',
        item_id_pattern=r'\b(item[_-]?[A-Za-z0-9_-]+)\b',
        mask_token='[UNVERIFIED]',
    )
    return EgressGuardConfig(
        enabled=enabled,
        pii_scrub=pii,
        moderator=mod,
        grounding_check=ground,
    )


def _results(items) -> RankedResults:
    return RankedResults(
        request_id='req_test',
        items=items,
        total_candidates=len(items),
        fusion_latency_ms=0.0,
    )


def _item(item_id, payload, fused_score=1.0) -> RankedItem:
    return RankedItem(
        item_id=item_id,
        fused_score=fused_score,
        contributing_sources=['vector'],
        payload=dict(payload),
        sub_intent_ids=[],
    )


# --------------------------------------------------------------------------- #
# NoOpEgressGuard                                                              #
# --------------------------------------------------------------------------- #

class TestNoOpEgressGuard:
    def test_preserves_input(self):
        items = [_item('item_1', {'title': 'a'}), _item('item_2', {'title': 'b'})]
        results = _results(items)
        guard = NoOpEgressGuard()
        out, outcome = guard.apply(results)
        assert out.items == items
        assert outcome.items_in == 2
        assert outcome.items_kept == 2
        assert outcome.items_dropped == 0
        assert all(d.action.action == 'kept' for d in outcome.decisions)

    def test_empty_results(self):
        out, outcome = NoOpEgressGuard().apply(_results([]))
        assert out.items == []
        assert outcome.items_in == 0

    def test_rejects_non_results(self):
        with pytest.raises(EgressGuardError, match="requires a RankedResults"):
            NoOpEgressGuard().apply("not_a_results")  # type: ignore[arg-type]

    def test_name(self):
        assert NoOpEgressGuard().name == 'noop'


# --------------------------------------------------------------------------- #
# EgressGuard construction                                                     #
# --------------------------------------------------------------------------- #

class TestEgressGuardConstruction:
    def test_valid_construction(self):
        g = EgressGuard(
            config=_egress_cfg(),
            sanitizer=_sanitizer(),
            moderator=LexicalBlocklistModerator(_lex_cfg()),
        )
        assert g.name == 'egress_guard'

    @pytest.mark.parametrize('config_attr,sanitizer_attr,moderator_attr,match', [
        pytest.param('none',  'valid',         'noop',  "requires an EgressGuardConfig",  id='none_config'),
        pytest.param('valid', 'none',          'noop',  "requires a LayerZeroSanitizer",  id='none_sanitizer'),
        pytest.param('valid', 'valid',         'none',  "requires a Moderator",           id='none_moderator'),
        pytest.param('valid', 'wrong_type',    'noop',  "requires a LayerZeroSanitizer",  id='wrong_type_sanitizer'),
    ])
    def test_invalid_construction_rejected( self, config_attr: str, sanitizer_attr: str, moderator_attr: str, match: str, ):
        config = _egress_cfg() if config_attr == 'valid' else None
        if sanitizer_attr == 'valid':
            sanitizer = _sanitizer()
        elif sanitizer_attr == 'wrong_type':
            sanitizer = "not_a_sanitizer"
        else:
            sanitizer = None
        moderator = NoOpModerator() if moderator_attr == 'noop' else None
        with pytest.raises(EgressGuardError, match=match):
            EgressGuard(config=config, sanitizer=sanitizer, moderator=moderator)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Apply argument validation                                                    #
# --------------------------------------------------------------------------- #

class TestApplyArgumentValidation:
    def test_apply_rejects_non_results(self):
        g = EgressGuard(
            config=_egress_cfg(),
            sanitizer=_sanitizer(),
            moderator=LexicalBlocklistModerator(_lex_cfg()),
        )
        with pytest.raises(EgressGuardError, match="requires a RankedResults"):
            g.apply("not_a_results")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# PII scrub paths                                                              #
# --------------------------------------------------------------------------- #

class TestPIIScrub:
    def _build(self, **kw):
        return EgressGuard(
            config=_egress_cfg(moderator_enabled=False, grounding_enabled=False, **kw),
            sanitizer=_sanitizer(),
            moderator=NoOpModerator(),
        )

    def test_clean_payload_kept(self):
        g = self._build()
        results = _results([_item('item_1', {'title': 'premium domain'})])
        out, outcome = g.apply(results)
        assert outcome.items_pii_masked == 0
        assert outcome.items_kept == 1
        assert out.items[0].payload['title'] == 'premium domain'

    def test_email_in_payload_masked(self):
        g = self._build()
        results = _results([_item('item_1', {'title': 'contact bob@example.com'})])
        out, outcome = g.apply(results)
        assert outcome.items_pii_masked == 1
        assert '[REDACTED]' in out.items[0].payload['title']
        assert 'bob@example.com' not in out.items[0].payload['title']

    def test_pii_disabled_passthrough(self):
        g = self._build(pii_enabled=False)
        results = _results([_item('item_1', {'title': 'bob@example.com'})])
        out, outcome = g.apply(results)
        assert outcome.items_pii_masked == 0
        assert outcome.items_kept == 1
        # Disabled — payload preserved verbatim.
        assert out.items[0].payload['title'] == 'bob@example.com'

    def test_pii_in_list_value_masked(self):
        g = self._build()
        results = _results([_item('item_1', {
            'title': 'clean',
            'description': ['line1', 'reach me at bob@example.com', 'line3'],
        })])
        out, outcome = g.apply(results)
        assert outcome.items_pii_masked == 1
        desc = out.items[0].payload['description']
        assert isinstance(desc, list)
        assert '[REDACTED]' in desc[1]

    def test_pii_in_nested_dict_masked(self):
        g = self._build()
        results = _results([_item('item_1', {
            'title': 'clean',
            'description': {'extra': 'see bob@example.com', 'note': 'fine'},
        })])
        out, outcome = g.apply(results)
        assert outcome.items_pii_masked == 1
        assert '[REDACTED]' in out.items[0].payload['description']['extra']

    def test_oversize_field_truncated_then_scrubbed(self):
        # max_field_chars=64 in _egress_cfg; build a long string with PII
        # in the FIRST 64 chars so the scrub still hits.
        long_text = 'bob@example.com ' + 'x' * 100  # PII at offset 0
        g = self._build()
        results = _results([_item('item_1', {'title': long_text})])
        out, outcome = g.apply(results)
        assert outcome.items_pii_masked == 1
        # Reasons list should record the truncation event.
        decision = outcome.decisions[0]
        assert any('oversize_field_truncated=title' == r for r in decision.action.reasons)

    def test_pii_in_unconfigured_field_not_scrubbed(self):
        # Configure pii_fields=['title'] only — PII in description is left alone.
        g = self._build(pii_fields=['title'])
        results = _results([_item('item_1', {
            'title': 'clean',
            'description': 'reach bob@example.com',
        })])
        out, outcome = g.apply(results)
        assert outcome.items_pii_masked == 0
        assert out.items[0].payload['description'] == 'reach bob@example.com'

    def test_non_string_scalar_passes_through(self):
        g = self._build()
        results = _results([_item('item_1', {'title': 42, 'description': 'ok'})])
        out, outcome = g.apply(results)
        assert outcome.items_pii_masked == 0
        assert out.items[0].payload['title'] == 42


# --------------------------------------------------------------------------- #
# Moderation                                                                   #
# --------------------------------------------------------------------------- #

class TestModeration:
    def _build(self, policy='drop', **kw):
        return EgressGuard(
        config=_egress_cfg( pii_enabled=False, grounding_enabled=False, moderator_enabled=True, moderator_policy=policy, **kw,),
            sanitizer=_sanitizer(),
            moderator=LexicalBlocklistModerator(_lex_cfg()),
        )

    def test_drop_policy_removes_flagged_item(self):
        g = self._build(policy='drop')
        results = _results([
            _item('item_1', {'title': 'clean'}),
            _item('item_2', {'title': 'spam content'}),
            _item('item_3', {'title': 'also clean'}),
        ])
        out, outcome = g.apply(results)
        assert outcome.items_in == 3
        assert outcome.items_dropped == 1
        assert outcome.items_kept == 2
        assert {it.item_id for it in out.items} == {'item_1', 'item_3'}
        # Decision for the dropped item carries the matched-token reason.
        dropped_dec = next(d for d in outcome.decisions if d.item_id == 'item_2')
        assert dropped_dec.action.action == 'dropped_moderation'
        assert any('banned_token=spam' in r for r in dropped_dec.action.reasons)

    def test_mask_policy_keeps_item_redacts_fields(self):
        g = self._build(policy='mask')
        results = _results([_item('item_1', {'title': 'spam content', 'description': 'bad'})])
        out, outcome = g.apply(results)
        assert outcome.items_dropped == 0
        assert outcome.items_kept == 1
        assert outcome.items_moderation_masked == 1
        # Lexical fields are redacted.
        assert out.items[0].payload['title'] == '[REDACTED]'
        assert out.items[0].payload['description'] == '[REDACTED]'
        decision = outcome.decisions[0]
        assert decision.action.action == 'moderation_masked'
        assert 'moderator_mask_applied' in decision.action.reasons

    def test_moderator_disabled_passthrough(self):
        # When moderator_enabled=False the gate skips the check entirely.
        g = EgressGuard(
        config=_egress_cfg( pii_enabled=False, grounding_enabled=False, moderator_enabled=False, moderator_backend='noop', moderator_policy='drop',),
            sanitizer=_sanitizer(),
            moderator=NoOpModerator(),
        )
        results = _results([_item('item_1', {'title': 'spam content'})])
        out, outcome = g.apply(results)
        assert outcome.items_dropped == 0
        assert outcome.items_kept == 1
        assert out.items[0].payload['title'] == 'spam content'


# --------------------------------------------------------------------------- #
# Grounding check                                                              #
# --------------------------------------------------------------------------- #

class TestGroundingCheck:
    def _build(self):
        return EgressGuard(
            config=_egress_cfg(pii_enabled=False, moderator_enabled=False, grounding_enabled=True),
            sanitizer=_sanitizer(),
            moderator=NoOpModerator(),
        )

    def test_grounded_citation_preserved(self):
        g = self._build()
        results = _results([
            _item('item_1', {'title': 'a', 'explanation': 'best match for item_1'}),
            _item('item_2', {'title': 'b'}),
        ])
        out, outcome = g.apply(results)
        assert outcome.items_explanation_scrubbed == 0
        assert out.items[0].payload['explanation'] == 'best match for item_1'

    def test_ungrounded_citation_masked(self):
        g = self._build()
        results = _results([
            _item('item_1', {'title': 'a', 'explanation': 'similar to item_999 in shape'}),
        ])
        out, outcome = g.apply(results)
        assert outcome.items_explanation_scrubbed == 1
        # The ungrounded citation 'item_999' is replaced with the mask token.
        assert '[UNVERIFIED]' in out.items[0].payload['explanation']
        assert 'item_999' not in out.items[0].payload['explanation']

    def test_mixed_grounded_and_ungrounded(self):
        g = self._build()
        results = _results([
            _item('item_1', {
                'title': 'a',
                'explanation': 'compare item_1 with item_999 and item_42',
            }),
            _item('item_42', {'title': 'b'}),
        ])
        out, outcome = g.apply(results)
        assert outcome.items_explanation_scrubbed == 1
        expl = out.items[0].payload['explanation']
        # item_1 and item_42 are present — kept verbatim.
        assert 'item_1' in expl
        assert 'item_42' in expl
        # item_999 is ungrounded — replaced.
        assert 'item_999' not in expl
        assert '[UNVERIFIED]' in expl

    @pytest.mark.parametrize('payload', [
        pytest.param({'title': 'a'},                       id='no_explanation_field'),
        pytest.param({'title': 'a', 'explanation': ''},    id='empty_explanation'),
        pytest.param({'title': 'a', 'explanation': 42},    id='non_string_explanation'),
    ])
    def test_explanation_skipped_paths(self, payload):
        g = self._build()
        results = _results([_item('item_1', payload)])
        out, outcome = g.apply(results)
        assert outcome.items_explanation_scrubbed == 0


# --------------------------------------------------------------------------- #
# Action precedence                                                            #
# --------------------------------------------------------------------------- #

class TestActionPrecedence:
    def test_drop_wins_over_pii(self):
        # Item has BOTH banned token AND PII — drop policy wins (item removed).
        g = EgressGuard(
            config=_egress_cfg(moderator_policy='drop', grounding_enabled=False),
            sanitizer=_sanitizer(),
            moderator=LexicalBlocklistModerator(_lex_cfg()),
        )
        results = _results([_item('item_1', {
            'title': 'spam item', 'description': 'reach bob@example.com',
        })])
        out, outcome = g.apply(results)
        assert outcome.items_dropped == 1
        assert outcome.items_kept == 0
        assert outcome.items_pii_masked == 0  # Skipped — drop happened first.

    def test_mask_then_pii_then_explanation_action_is_moderation_masked(self):
        # Item has banned token (mask policy), PII (in a field NOT covered by
        # the moderator's lexical fields so it survives mask), AND ungrounded
        # citation. Precedence: moderation_masked > pii_masked >
        # explanation_scrubbed.
        # NB: when the moderator lexical fields overlap with the PII fields
        # AND the item is masked, the mask token replaces the field BEFORE
        # the PII pass runs, so PII would find nothing — that's the right
        # outcome, not a contradiction. To exercise both the moderation
        # mask AND a PII match in the same item, place the PII in
        # `explanation` (which the moderator does NOT mask but the PII pass
        # DOES walk).
        g = EgressGuard(
            config=_egress_cfg(moderator_policy='mask'),
            sanitizer=_sanitizer(),
            moderator=LexicalBlocklistModerator(_lex_cfg()),
        )
        results = _results([_item('item_1', {
            'title': 'spam payload',                  # moderator-masked
            'description': 'clean',                   # moderator-masked
            'explanation': 'reach bob@example.com cite item_999',  # PII + ungrounded citation
        })])
        out, outcome = g.apply(results)
        assert outcome.items_kept == 1
        assert outcome.items_dropped == 0
        assert outcome.items_moderation_masked == 1
        decision = outcome.decisions[0]
        assert decision.action.action == 'moderation_masked'
        reasons_text = ' '.join(decision.action.reasons)
        assert 'banned_token=spam' in reasons_text
        assert 'pii_matches' in reasons_text
        assert 'ungrounded_citations' in reasons_text
        # Both mask + PII + grounding mutations applied to the surviving item.
        assert out.items[0].payload['title'] == '[REDACTED]'
        assert '[REDACTED]' in out.items[0].payload['explanation']  # PII masked
        assert '[UNVERIFIED]' in out.items[0].payload['explanation']  # ungrounded citation masked

    def test_pii_wins_over_explanation_only(self):
        # Item has PII AND ungrounded citation — pii_masked wins.
        g = EgressGuard(
            config=_egress_cfg(moderator_enabled=False, moderator_backend='noop'),
            sanitizer=_sanitizer(),
            moderator=NoOpModerator(),
        )
        results = _results([_item('item_1', {
            'title': 'ok',
            'description': 'reach bob@example.com',
            'explanation': 'see item_999',
        })])
        out, outcome = g.apply(results)
        assert outcome.items_pii_masked == 1
        assert outcome.items_explanation_scrubbed == 0  # explanation_scrubbed counter loses to pii.
        decision = outcome.decisions[0]
        assert decision.action.action == 'pii_masked'
        reasons_text = ' '.join(decision.action.reasons)
        assert 'pii_matches' in reasons_text
        assert 'ungrounded_citations' in reasons_text  # Still appears in reasons.


# --------------------------------------------------------------------------- #
# Fail-soft on per-item exceptions                                             #
# --------------------------------------------------------------------------- #

class _ExplodingModerator:
    """Mock moderator that raises EgressGuardError on every call.

    Used to verify the gate fails-CLOSED — drops the offending item rather
    than letting an unscrubbed payload leak past the gate.
    """

    _NAME = 'exploding'

    @property
    def name(self) -> str:
        return self._NAME

    def moderate(self, payload):
        del payload
        raise EgressGuardError("simulated backend failure")


class TestFailSoftOnException:
    def test_per_item_exception_drops_item(self):
        g = EgressGuard(
            config=_egress_cfg(moderator_enabled=True, moderator_backend='noop', moderator_policy='drop',
                               pii_enabled=False, grounding_enabled=False),
            sanitizer=_sanitizer(),
            moderator=_ExplodingModerator(),
        )
        # Bypass `Moderator` Protocol isinstance check — `Moderator` is
        # `@runtime_checkable` so ducktyped backends pass.
        assert isinstance(_ExplodingModerator(), Moderator)
        results = _results([_item('item_1', {'title': 'anything'})])
        out, outcome = g.apply(results)
        assert outcome.items_dropped == 1
        assert outcome.items_kept == 0
        assert len(out.items) == 0
        decision = outcome.decisions[0]
        assert decision.action.action == 'dropped_moderation'
        assert any('guard_internal_error_EgressGuardError' in r for r in decision.action.reasons)


# --------------------------------------------------------------------------- #
# Outcome consistency on every path                                            #
# --------------------------------------------------------------------------- #

class TestOutcomeConsistency:
    def test_outcome_passes_post_init_for_kept_only(self):
        g = EgressGuard(
            config=_egress_cfg(),
            sanitizer=_sanitizer(),
            moderator=LexicalBlocklistModerator(_lex_cfg()),
        )
        out, outcome = g.apply(_results([_item(f'item_{i}', {'title': 'clean'}) for i in range(5)]))
        # Construction would have raised if counters mismatched.
        assert outcome.items_in == 5
        assert outcome.items_kept == 5
        assert len(outcome.decisions) == 5

    def test_outcome_consistency_under_mixed_actions(self):
        g = EgressGuard(
            config=_egress_cfg(moderator_policy='drop'),
            sanitizer=_sanitizer(),
            moderator=LexicalBlocklistModerator(_lex_cfg()),
        )
        results = _results([
            _item('item_a', {'title': 'clean'}),
            _item('item_b', {'title': 'spam dropped'}),
            _item('item_c', {'title': 'visit bob@example.com'}),
            _item('item_d', {'title': 'a', 'explanation': 'cite item_999'}),
        ])
        out, outcome = g.apply(results)
        # 1 kept, 1 dropped, 1 pii, 1 explanation -> 3 kept + 1 dropped.
        assert outcome.items_in == 4
        assert outcome.items_dropped == 1
        assert outcome.items_kept == 3
        assert outcome.items_pii_masked == 1
        assert outcome.items_explanation_scrubbed == 1
        # Output items match counters.
        assert len(out.items) == 3
        # Decisions list mirrors input order.
        assert [d.item_id for d in outcome.decisions] == ['item_a', 'item_b', 'item_c', 'item_d']

    def test_envelope_metadata_preserved(self):
        # request_id, total_candidates, fusion_latency_ms, cache_hit, multi_intent_envelope
        # must all be preserved on the scrubbed RankedResults.
        g = EgressGuard(
            config=_egress_cfg(moderator_enabled=False, moderator_backend='noop', pii_enabled=False, grounding_enabled=False),
            sanitizer=_sanitizer(),
            moderator=NoOpModerator(),
        )
        results = RankedResults(
            request_id='req_xyz',
            items=[_item('item_1', {'title': 'a'})],
            total_candidates=42,
            fusion_latency_ms=12.5,
            cache_hit='exact',
            multi_intent_envelope=None,
        )
        out, _outcome = g.apply(results)
        assert out.request_id == 'req_xyz'
        assert out.total_candidates == 42
        assert out.fusion_latency_ms == 12.5
        assert out.cache_hit == 'exact'

    def test_input_not_mutated(self):
        # Defense-in-depth: the gate must NEVER mutate the input list/items.
        g = EgressGuard(
            config=_egress_cfg(moderator_policy='drop'),
            sanitizer=_sanitizer(),
            moderator=LexicalBlocklistModerator(_lex_cfg()),
        )
        item_clean = _item('item_a', {'title': 'clean'})
        item_pii = _item('item_b', {'title': 'reach bob@example.com'})
        results = _results([item_clean, item_pii])
        original_pii_payload = dict(item_pii.payload)
        g.apply(results)
        # Original item must NOT have been mutated.
        assert item_pii.payload == original_pii_payload
        assert results.items == [item_clean, item_pii]
