"""Typed contracts for the output-side egress guard.

Three dataclasses define the gate's audit envelope so callers (orchestrator,
tests, future SRE dashboards) never have to inspect raw dicts:

- :class:`EgressItemAction` — what the gate did to a single item
- :class:`EgressDecision` — the per-item record (item_id + action + reasons)
- :class:`EgressGuardOutcome` — the per-call summary attached to RankedResults
  (counters + the ordered per-item decision list)

All three follow the established ``__post_init__`` validation pattern from
``semantic_search.contracts`` so a malformed instance fails at construction.
"""
from dataclasses import dataclass, field
from typing import FrozenSet, List

from semantic_search.core.exceptions import ValidationError

# Per-item actions the gate can take. Closed enum (frozenset) so a typo at a
# call site fails ``__post_init__`` instead of silently producing an
# unrecognised action that downstream dashboards will drop.
# Precedence (most-severe wins; multi-issue detail kept in ``reasons``):
#     dropped_moderation > moderation_masked > pii_masked > explanation_scrubbed > kept
EGRESS_ITEM_ACTIONS: FrozenSet[str] = frozenset({
    'kept',                    # Item survived all checks unmodified
    'explanation_scrubbed',    # Item kept; ungrounded ``explanation`` cite spans masked
    'pii_masked',              # Item kept; one or more payload fields had PII redacted
    'moderation_masked',       # Item kept; moderator flagged in mask-policy mode (configured fields redacted)
    'dropped_moderation',      # Item dropped because moderator flagged in drop-policy mode
})


@dataclass(frozen=True)
class EgressItemAction:
    """A single action taken by the gate on one item, used inside :class:`EgressDecision`.

    :param action: str - One of ``EGRESS_ITEM_ACTIONS``
    :param reasons: List[str] - Per-check reason codes (e.g. ``['pii_email_match']``,
        ``['banned_token=fakeword']``); empty when ``action == 'kept'``
    """
    action: str
    reasons: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.action not in EGRESS_ITEM_ACTIONS:
            raise ValidationError(f"EgressItemAction.action must be one of {sorted(EGRESS_ITEM_ACTIONS)!r}, got {self.action!r}")
        if not isinstance(self.reasons, list):
            raise ValidationError("EgressItemAction.reasons must be a list")
        for r in self.reasons:
            if not isinstance(r, str) or not r:
                raise ValidationError("EgressItemAction.reasons entries must be non-empty strings")
        if self.action == 'kept' and len(self.reasons) > 0:
            raise ValidationError("EgressItemAction.reasons must be empty when action == 'kept'")
        if self.action != 'kept' and len(self.reasons) == 0:
            raise ValidationError(f"EgressItemAction.reasons must be non-empty when action == {self.action!r}")


@dataclass(frozen=True)
class EgressDecision:
    """Per-item audit record. The gate emits exactly one of these per input item.

    :param item_id: str - Stable item id (matches ``RankedItem.item_id``)
    :param action: EgressItemAction - What the gate did and why
    """
    item_id: str
    action: EgressItemAction

    def __post_init__(self) -> None:
        if not isinstance(self.item_id, str) or not self.item_id:
            raise ValidationError("EgressDecision.item_id must be a non-empty string")
        if not isinstance(self.action, EgressItemAction):
            raise ValidationError("EgressDecision.action must be an EgressItemAction instance")


@dataclass(frozen=True)
class EgressGuardOutcome:
    """The audit envelope returned alongside the scrubbed RankedResults.

    Counters give the SRE-dashboard / proxy-signal view; ``decisions`` gives
    the per-item drilldown for debugging a specific request.

    Invariants enforced in ``__post_init__``:

    - ``items_in == items_kept + items_dropped``
    - Every per-action counter equals the count of matching ``decisions``
      entries (single source of truth check — counters cannot drift from
      the per-item record)
    - ``items_kept`` counts every item whose action is NOT ``dropped_moderation``
      (an item that was masked or had its explanation scrubbed is still kept)

    :param items_in: int - Items the gate received (>= 0)
    :param items_kept: int - Items still in the output (>= 0, <= items_in)
    :param items_dropped: int - Items removed by moderation (>= 0)
    :param items_pii_masked: int - Items whose primary action was PII redaction (>= 0)
    :param items_explanation_scrubbed: int - Items whose primary action was
        ungrounded-citation masking (>= 0)
    :param items_moderation_masked: int - Items whose primary action was
        moderator-mask-policy redaction (>= 0)
    :param decisions: List[EgressDecision] - Per-item audit, in input order
    :param latency_ms: float - Wall time the gate spent (>= 0)
    """
    items_in: int
    items_kept: int
    items_dropped: int
    items_pii_masked: int
    items_explanation_scrubbed: int
    items_moderation_masked: int
    decisions: List[EgressDecision]
    latency_ms: float

    def __post_init__(self) -> None:
        if int(self.items_in) < 0:
            raise ValidationError("EgressGuardOutcome.items_in must be >= 0")
        if int(self.items_kept) < 0:
            raise ValidationError("EgressGuardOutcome.items_kept must be >= 0")
        if int(self.items_dropped) < 0:
            raise ValidationError("EgressGuardOutcome.items_dropped must be >= 0")
        if int(self.items_pii_masked) < 0:
            raise ValidationError("EgressGuardOutcome.items_pii_masked must be >= 0")
        if int(self.items_explanation_scrubbed) < 0:
            raise ValidationError("EgressGuardOutcome.items_explanation_scrubbed must be >= 0")
        if int(self.items_moderation_masked) < 0:
            raise ValidationError("EgressGuardOutcome.items_moderation_masked must be >= 0")
        if float(self.latency_ms) < 0.0:
            raise ValidationError("EgressGuardOutcome.latency_ms must be >= 0")
        if int(self.items_kept) + int(self.items_dropped) != int(self.items_in):
            raise ValidationError(f"EgressGuardOutcome counter mismatch: items_kept({self.items_kept}) + items_dropped({self.items_dropped}) " f"!= items_in({self.items_in})")
        if not isinstance(self.decisions, list):
            raise ValidationError("EgressGuardOutcome.decisions must be a list")
        if len(self.decisions) != int(self.items_in):
            raise ValidationError(f"EgressGuardOutcome.decisions length ({len(self.decisions)}) must equal items_in ({self.items_in})")
        for d in self.decisions:
            if not isinstance(d, EgressDecision):
                raise ValidationError("EgressGuardOutcome.decisions entries must be EgressDecision instances")
        # Cross-counter consistency — the per-item record is the single source of truth.
        actual_dropped = sum(1 for d in self.decisions if d.action.action == 'dropped_moderation')
        actual_pii = sum(1 for d in self.decisions if d.action.action == 'pii_masked')
        actual_expl = sum(1 for d in self.decisions if d.action.action == 'explanation_scrubbed')
        actual_mod_masked = sum(1 for d in self.decisions if d.action.action == 'moderation_masked')
        if actual_dropped != int(self.items_dropped):
            raise ValidationError(f"EgressGuardOutcome.items_dropped ({self.items_dropped}) does not match decisions count ({actual_dropped})")
        if actual_pii != int(self.items_pii_masked):
            raise ValidationError(f"EgressGuardOutcome.items_pii_masked ({self.items_pii_masked}) does not match decisions count ({actual_pii})")
        if actual_expl != int(self.items_explanation_scrubbed):
            raise ValidationError(f"EgressGuardOutcome.items_explanation_scrubbed ({self.items_explanation_scrubbed}) " f"does not match decisions count ({actual_expl})")
        if actual_mod_masked != int(self.items_moderation_masked):
            raise ValidationError(f"EgressGuardOutcome.items_moderation_masked ({self.items_moderation_masked}) " f"does not match decisions count ({actual_mod_masked})")
