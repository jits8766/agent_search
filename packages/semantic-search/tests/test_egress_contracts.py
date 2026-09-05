"""Contract tests for the egress-guard typed envelope.

Covers ``EgressItemAction``, ``EgressDecision``, and ``EgressGuardOutcome``
``__post_init__`` validation. The cross-counter consistency check on
``EgressGuardOutcome`` is the safety-critical invariant — counters and the
per-item decision list cannot drift out of sync without raising at
construction time.
"""
import pytest

from semantic_search.core.exceptions import ValidationError
from semantic_search.safety.egress_contracts import EGRESS_ITEM_ACTIONS, EgressDecision, EgressGuardOutcome, EgressItemAction


def _kept() -> EgressItemAction:
    return EgressItemAction(action='kept', reasons=[])


def _dropped() -> EgressItemAction:
    return EgressItemAction(action='dropped_moderation', reasons=['banned_token=foo'])


def _pii_masked() -> EgressItemAction:
    return EgressItemAction(action='pii_masked', reasons=['pii_field=description'])


def _expl_scrubbed() -> EgressItemAction:
    return EgressItemAction(action='explanation_scrubbed', reasons=['ungrounded_citations=1'])


def _mod_masked() -> EgressItemAction:
    return EgressItemAction(action='moderation_masked', reasons=['banned_token=foo', 'moderator_mask_applied'])


class TestEgressItemAction:
    """`EgressItemAction.__post_init__` validation."""

    def test_action_universe_is_closed(self):
        assert EGRESS_ITEM_ACTIONS == frozenset({
            'kept', 'explanation_scrubbed', 'pii_masked',
            'moderation_masked', 'dropped_moderation',
        })

    def test_unknown_action_rejected(self):
        with pytest.raises(ValidationError, match="action must be one of"):
            EgressItemAction(action='unknown', reasons=['x'])

    def test_kept_with_reasons_rejected(self):
        with pytest.raises(ValidationError, match="must be empty when action == 'kept'"):
            EgressItemAction(action='kept', reasons=['noise'])

    def test_dropped_without_reasons_rejected(self):
        with pytest.raises(ValidationError, match="must be non-empty when action == 'dropped_moderation'"):
            EgressItemAction(action='dropped_moderation', reasons=[])

    def test_pii_masked_without_reasons_rejected(self):
        with pytest.raises(ValidationError, match="must be non-empty when action == 'pii_masked'"):
            EgressItemAction(action='pii_masked', reasons=[])

    def test_reasons_must_be_list(self):
        with pytest.raises(ValidationError, match="reasons must be a list"):
            EgressItemAction(action='kept', reasons='not_a_list')  # type: ignore[arg-type]

    def test_reason_entries_must_be_non_empty_strings(self):
        with pytest.raises(ValidationError, match="reasons entries must be non-empty strings"):
            EgressItemAction(action='pii_masked', reasons=[''])
        with pytest.raises(ValidationError, match="reasons entries must be non-empty strings"):
            EgressItemAction(action='pii_masked', reasons=[123])  # type: ignore[list-item]

    def test_kept_default_reasons_accepted(self):
        a = EgressItemAction(action='kept')
        assert a.reasons == []

    @pytest.mark.parametrize('action', sorted(EGRESS_ITEM_ACTIONS - {'kept'}))
    def test_every_non_kept_action_round_trips(self, action):
        a = EgressItemAction(action=action, reasons=['some_reason'])
        assert a.action == action
        assert a.reasons == ['some_reason']


class TestEgressDecision:
    def test_valid(self):
        d = EgressDecision(item_id='item_1', action=_kept())
        assert d.item_id == 'item_1'
        assert d.action.action == 'kept'

    def test_empty_item_id_rejected(self):
        with pytest.raises(ValidationError, match="item_id must be a non-empty string"):
            EgressDecision(item_id='', action=_kept())

    def test_non_string_item_id_rejected(self):
        with pytest.raises(ValidationError, match="item_id must be a non-empty string"):
            EgressDecision(item_id=42, action=_kept())  # type: ignore[arg-type]

    def test_action_must_be_egress_item_action(self):
        with pytest.raises(ValidationError, match="action must be an EgressItemAction"):
            EgressDecision(item_id='item_1', action='kept')  # type: ignore[arg-type]


class TestEgressGuardOutcome:
    """`EgressGuardOutcome.__post_init__` cross-counter consistency."""

    def test_zero_items_valid(self):
        o = EgressGuardOutcome(
            items_in=0, items_kept=0, items_dropped=0,
            items_pii_masked=0, items_explanation_scrubbed=0,
            items_moderation_masked=0, decisions=[], latency_ms=0.0,
        )
        assert o.items_in == 0

    def test_kept_only(self):
        decs = [EgressDecision(item_id=f'i{i}', action=_kept()) for i in range(3)]
        o = EgressGuardOutcome(
            items_in=3, items_kept=3, items_dropped=0,
            items_pii_masked=0, items_explanation_scrubbed=0,
            items_moderation_masked=0, decisions=decs, latency_ms=0.0,
        )
        assert len(o.decisions) == 3

    def test_kept_plus_dropped_must_equal_in(self):
        decs = [
            EgressDecision(item_id='i0', action=_kept()),
            EgressDecision(item_id='i1', action=_dropped()),
        ]
        # 1 kept + 1 dropped = 2 in: valid.
        EgressGuardOutcome(
            items_in=2, items_kept=1, items_dropped=1,
            items_pii_masked=0, items_explanation_scrubbed=0,
            items_moderation_masked=0, decisions=decs, latency_ms=0.0,
        )
        # 1 kept + 0 dropped != 2 in: invalid.
        with pytest.raises(ValidationError, match="counter mismatch"):
            EgressGuardOutcome(
                items_in=2, items_kept=1, items_dropped=0,
                items_pii_masked=0, items_explanation_scrubbed=0,
                items_moderation_masked=0, decisions=decs, latency_ms=0.0,
            )

    def test_decisions_length_must_equal_items_in(self):
        decs = [EgressDecision(item_id='i0', action=_kept())]
        with pytest.raises(ValidationError, match="decisions length"):
            EgressGuardOutcome(
                items_in=2, items_kept=2, items_dropped=0,
                items_pii_masked=0, items_explanation_scrubbed=0,
                items_moderation_masked=0, decisions=decs, latency_ms=0.0,
            )

    def test_pii_counter_must_match_decisions(self):
        decs = [
            EgressDecision(item_id='i0', action=_pii_masked()),
            EgressDecision(item_id='i1', action=_kept()),
        ]
        with pytest.raises(ValidationError, match="items_pii_masked.*does not match decisions count"):
            EgressGuardOutcome(
                items_in=2, items_kept=2, items_dropped=0,
                items_pii_masked=0,  # Should be 1.
                items_explanation_scrubbed=0,
                items_moderation_masked=0, decisions=decs, latency_ms=0.0,
            )

    def test_explanation_counter_must_match_decisions(self):
        decs = [EgressDecision(item_id='i0', action=_expl_scrubbed())]
        with pytest.raises(ValidationError, match="items_explanation_scrubbed.*does not match"):
            EgressGuardOutcome(
                items_in=1, items_kept=1, items_dropped=0,
                items_pii_masked=0, items_explanation_scrubbed=0,  # Should be 1.
                items_moderation_masked=0, decisions=decs, latency_ms=0.0,
            )

    def test_moderation_masked_counter_must_match_decisions(self):
        decs = [EgressDecision(item_id='i0', action=_mod_masked())]
        # Correct counter passes.
        EgressGuardOutcome(
            items_in=1, items_kept=1, items_dropped=0,
            items_pii_masked=0, items_explanation_scrubbed=0,
            items_moderation_masked=1, decisions=decs, latency_ms=0.0,
        )
        # Mismatched counter fails.
        with pytest.raises(ValidationError, match="items_moderation_masked.*does not match"):
            EgressGuardOutcome(
                items_in=1, items_kept=1, items_dropped=0,
                items_pii_masked=0, items_explanation_scrubbed=0,
                items_moderation_masked=0, decisions=decs, latency_ms=0.0,
            )

    def test_dropped_counter_must_match_decisions(self):
        decs = [EgressDecision(item_id='i0', action=_dropped())]
        with pytest.raises(ValidationError, match="items_dropped.*does not match decisions count"):
            EgressGuardOutcome(
                items_in=1, items_kept=1, items_dropped=0,  # Should be dropped=1, kept=0.
                items_pii_masked=0, items_explanation_scrubbed=0,
                items_moderation_masked=0, decisions=decs, latency_ms=0.0,
            )

    def test_negative_counters_rejected(self):
        for kw in (
            'items_in', 'items_kept', 'items_dropped',
            'items_pii_masked', 'items_explanation_scrubbed',
            'items_moderation_masked',
        ):
            kwargs = dict(
                items_in=0, items_kept=0, items_dropped=0,
                items_pii_masked=0, items_explanation_scrubbed=0,
                items_moderation_masked=0, decisions=[], latency_ms=0.0,
            )
            kwargs[kw] = -1
            with pytest.raises(ValidationError, match=f"{kw} must be >= 0"):
                EgressGuardOutcome(**kwargs)

    def test_negative_latency_rejected(self):
        with pytest.raises(ValidationError, match="latency_ms must be >= 0"):
            EgressGuardOutcome(
                items_in=0, items_kept=0, items_dropped=0,
                items_pii_masked=0, items_explanation_scrubbed=0,
                items_moderation_masked=0, decisions=[], latency_ms=-0.1,
            )

    def test_decisions_must_be_list(self):
        with pytest.raises(ValidationError, match="decisions must be a list"):
            EgressGuardOutcome(
                items_in=0, items_kept=0, items_dropped=0,
                items_pii_masked=0, items_explanation_scrubbed=0,
                items_moderation_masked=0, decisions='not_a_list',  # type: ignore[arg-type]
                latency_ms=0.0,
            )

    def test_decisions_entries_must_be_egress_decision(self):
        with pytest.raises(ValidationError, match="decisions entries must be EgressDecision"):
            EgressGuardOutcome(
                items_in=1, items_kept=1, items_dropped=0,
                items_pii_masked=0, items_explanation_scrubbed=0,
                items_moderation_masked=0, decisions=['raw'],  # type: ignore[list-item]
                latency_ms=0.0,
            )

    def test_full_audit_envelope_valid(self):
        decs = [
            EgressDecision(item_id='i0', action=_kept()),
            EgressDecision(item_id='i1', action=_pii_masked()),
            EgressDecision(item_id='i2', action=_expl_scrubbed()),
            EgressDecision(item_id='i3', action=_mod_masked()),
            EgressDecision(item_id='i4', action=_dropped()),
        ]
        o = EgressGuardOutcome(
            items_in=5, items_kept=4, items_dropped=1,
            items_pii_masked=1, items_explanation_scrubbed=1,
            items_moderation_masked=1, decisions=decs, latency_ms=1.5,
        )
        assert o.items_kept + o.items_dropped == o.items_in
        assert o.items_pii_masked == 1
        assert o.items_explanation_scrubbed == 1
        assert o.items_moderation_masked == 1
        assert o.items_dropped == 1
