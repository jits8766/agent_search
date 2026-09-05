"""Output-side safety package.

Three composed checks run on the FINAL ranked-results envelope BEFORE the
orchestrator returns to the API surface:

1. PII scrub — masks email / phone / SSN / credit-card patterns inside every
   string-typed payload field. Reuses the same regex pass already compiled in
   ``semantic_search.safety.layer_zero_sanitizer.LayerZeroSanitizer`` so a policy change in
   one place propagates to both the input-side gate and the output-side gate.
2. Lexical-blocklist moderation — drops items whose payload (concatenated
   configured fields, tokenised through the same ``tokenize_lexical`` shared
   utility BM25 + reranker + diversifier all use) contains any token in the
   configured banned-term set. Stdlib-only, deterministic; an extension seam
   is left for a future model-backed ``Moderator`` implementation.
3. Grounding check — when an item carries an LLM-generated ``explanation``
   payload field, every ``item_id`` cited in the explanation must exist in
   the result set; mismatched citations are masked at the span (the listing
   itself is preserved — only the unsafe explanation text is scrubbed).

The composed gate is exposed as :class:`EgressGuard` (or :class:`NoOpEgressGuard`
when fully disabled). Both implement ``apply(results) -> Tuple[RankedResults,
EgressGuardOutcome]`` so the orchestrator can call it unconditionally without
a ``None`` check, mirroring the / dispatch shape.

Public re-exports:
- :class:`EgressGuard` / :class:`NoOpEgressGuard` — the gate
- :class:`EgressGuardOutcome`, :class:`EgressDecision`, :class:`EgressItemAction`
- :class:`Moderator` (Protocol), :class:`LexicalBlocklistModerator`, :class:`NoOpModerator`
"""
from semantic_search.safety.egress_contracts import EGRESS_ITEM_ACTIONS, EgressDecision, EgressGuardOutcome, EgressItemAction
from semantic_search.safety.egress_guard import EgressGuard, NoOpEgressGuard
from semantic_search.safety.lexical_blocklist_moderator import LexicalBlocklistModerator, Moderator, ModeratorVerdict, NoOpModerator

__all__ = [
    'EGRESS_ITEM_ACTIONS',
    'EgressDecision',
    'EgressGuard',
    'EgressGuardOutcome',
    'EgressItemAction',
    'LexicalBlocklistModerator',
    'Moderator',
    'ModeratorVerdict',
    'NoOpEgressGuard',
    'NoOpModerator',
]
