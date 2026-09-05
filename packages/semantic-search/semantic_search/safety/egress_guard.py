"""EgressGuard — composes PII scrub + moderation + grounding check.

The gate is the FINAL step in the orchestrator's search/refine pipeline,
applied AFTER ``_truncate`` so we only scrub what the user actually sees
(bounded cost). It runs SYNCHRONOUSLY because every primitive (regex,
dict walks, lexical tokenisation) is CPU-bound; the orchestrator wraps
``EgressGuard.apply`` in ``asyncio.to_thread`` so the event loop stays
responsive without making the gate itself async.

Action precedence (mutual exclusion within ``EgressItemAction.action``):

    dropped_moderation > pii_masked > explanation_scrubbed > kept

Multi-issue detail (e.g. an item with both PII matches AND ungrounded
citations) is preserved in the per-item ``reasons`` list — the most-severe
action category wins so SRE counters stay simple, but the full picture
is recoverable from the audit envelope.

Failure mode: per-item exceptions are caught and converted to a
"drop with reason='guard_internal_error_<type>'" decision so a bug in
any sub-check cannot leak an unscrubbed payload past the gate.
Construction-time errors (malformed sub-config, bad sanitizer handle)
raise ``EgressGuardError`` — the registry refuses to boot rather than
silently shipping a half-wired gate.
"""
import re as _re
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

from semantic_search.config.models import EgressGuardConfig
from semantic_search.contracts import RankedItem, RankedResults
from semantic_search.core.exceptions import EgressGuardError
from semantic_search.core.logging_utils import get_logger
from semantic_search.safety.layer_zero_sanitizer import LayerZeroSanitizer
from semantic_search.safety.egress_contracts import EgressDecision, EgressGuardOutcome, EgressItemAction
from semantic_search.safety.lexical_blocklist_moderator import Moderator

logger = get_logger(__name__)

_MS_PER_S = 1000.0


class NoOpEgressGuard:
    """Identity gate.

    Wired by the registry when ``safety.egress_guard.enabled=false`` so
    the orchestrator can call ``apply`` unconditionally without a None
    check. Returns the input ``RankedResults`` verbatim alongside an
    audit envelope reporting "kept" for every item.

    The shape mirrors :class:`EgressGuard.apply` so the orchestrator's
    call site is structurally identical regardless of the gate flavor.
    """

    _NAME = 'noop'

    @property
    def name(self) -> str:
        return self._NAME

    def apply(self, results: RankedResults) -> Tuple[RankedResults, EgressGuardOutcome]:
        if results is None or not isinstance(results, RankedResults):
            raise EgressGuardError("NoOpEgressGuard.apply requires a RankedResults instance")
        t0 = time.monotonic()
        decisions = [
            EgressDecision(item_id=it.item_id, action=EgressItemAction(action='kept', reasons=[]))
            for it in results.items
        ]
        latency_ms = (time.monotonic() - t0) * _MS_PER_S
        outcome = EgressGuardOutcome(
            items_in=len(results.items),
            items_kept=len(results.items),
            items_dropped=0,
            items_pii_masked=0,
            items_explanation_scrubbed=0,
            items_moderation_masked=0,
            decisions=decisions,
            latency_ms=latency_ms,
        )
        return results, outcome


class EgressGuard:
    """Composing gate for the three checks.

    :param config: EgressGuardConfig - Already-validated config envelope
    :param sanitizer: LayerZeroSanitizer - Source of truth for PII regex
        (the egress gate borrows the input-side gate's compiled patterns
        so a policy change applies to both)
    :param moderator: Moderator - Backend implementation
        (LexicalBlocklistModerator / NoOpModerator / future model-backed)
    :raises EgressGuardError: When construction would produce a half-wired gate
    """

    _NAME = 'egress_guard'

    def __init__(self, config: EgressGuardConfig, sanitizer: LayerZeroSanitizer, moderator: Moderator):
        if config is None or not isinstance(config, EgressGuardConfig):
            raise EgressGuardError("EgressGuard requires an EgressGuardConfig instance")
        if sanitizer is None or not isinstance(sanitizer, LayerZeroSanitizer):
            # The PII pass borrows the LayerZeroSanitizer's regex set; refusing
            # to construct without one prevents a silent "no PII patterns
            # configured" deployment from masquerading as a working gate.
            raise EgressGuardError("EgressGuard requires a LayerZeroSanitizer instance " "(reused as the source of truth for PII patterns)")
        if moderator is None or not isinstance(moderator, Moderator):
            raise EgressGuardError("EgressGuard requires a Moderator instance " "(wire NoOpModerator when safety.egress_guard.moderator.enabled=false)")
        self._config = config
        self._sanitizer = sanitizer
        self._moderator = moderator
        # Pre-compile the grounding-check pattern once at construction.
        # GroundingCheckConfig.__post_init__ already validated the regex.
        self._explanation_field = config.grounding_check.explanation_field
        self._mask_token = config.grounding_check.mask_token
        self._citation_re = _re.compile(config.grounding_check.item_id_pattern)
        # Hot-path scalars copied off the dataclass to bound dict lookups.
        self._pii_enabled = bool(config.pii_scrub.enabled)
        self._pii_fields = list(config.pii_scrub.payload_fields)
        self._pii_max_chars = int(config.pii_scrub.max_field_chars)
        self._moderator_enabled = bool(config.moderator.enabled)
        self._moderator_policy = str(config.moderator.policy)
        self._grounding_enabled = bool(config.grounding_check.enabled)

    @property
    def name(self) -> str:
        return self._NAME

    def apply(self, results: RankedResults) -> Tuple[RankedResults, EgressGuardOutcome]:
        """Run all enabled checks against ``results.items`` and return the scrubbed envelope.

        :param results: RankedResults - Final ranked results, post-truncation
        :return: Tuple[RankedResults, EgressGuardOutcome] - Scrubbed results
            (a reconstructed ``RankedResults`` — the input is never mutated) plus the
            audit envelope. ``RankedResults.request_id`` /
            ``total_candidates`` / ``fusion_latency_ms`` / ``cache_hit`` /
            ``multi_intent_envelope`` are preserved verbatim;
            ``items`` is the surviving (possibly mutated) subset.
        :raises EgressGuardError: Only on argument-validation failure;
            per-item runtime errors are caught and converted to a
            drop-with-reason decision so the gate cannot leak.
        """
        if results is None or not isinstance(results, RankedResults):
            raise EgressGuardError("EgressGuard.apply requires a RankedResults instance")
        t0 = time.monotonic()
        items_in = len(results.items)
        scrubbed_items: List[RankedItem] = []
        decisions: List[EgressDecision] = []
        items_dropped = 0
        items_pii_masked = 0
        items_explanation_scrubbed = 0
        items_moderation_masked = 0
        # Snapshot of valid item_ids — used by the grounding check to verify
        # that any cited id in an explanation is actually in the result set.
        valid_item_ids = frozenset(it.item_id for it in results.items)
        for item in results.items:
            try:
                action, scrubbed_item = self._scrub_item(item, valid_item_ids)
            except EgressGuardError as e:
                # Fail-closed on any guard-internal error: drop the item rather
                # than risk leaking an unscrubbed payload.
                action = EgressItemAction(action='dropped_moderation', reasons=[f'guard_internal_error_{type(e).__name__}'],)
                scrubbed_item = None
                logger.warning(f"egress_guard_item_dropped item_id={item.item_id} " f"reason=guard_internal_error error_type={type(e).__name__}")
            decisions.append(EgressDecision(item_id=item.item_id, action=action))
            if action.action == 'dropped_moderation':
                items_dropped += 1
            elif action.action == 'moderation_masked':
                items_moderation_masked += 1
                assert scrubbed_item is not None
                scrubbed_items.append(scrubbed_item)
            elif action.action == 'pii_masked':
                items_pii_masked += 1
                assert scrubbed_item is not None
                scrubbed_items.append(scrubbed_item)
            elif action.action == 'explanation_scrubbed':
                items_explanation_scrubbed += 1
                assert scrubbed_item is not None
                scrubbed_items.append(scrubbed_item)
            else:  # 'kept'
                scrubbed_items.append(item)
        latency_ms = (time.monotonic() - t0) * _MS_PER_S
        outcome = EgressGuardOutcome(
            items_in=items_in,
            items_kept=items_in - items_dropped,
            items_dropped=items_dropped,
            items_pii_masked=items_pii_masked,
            items_explanation_scrubbed=items_explanation_scrubbed,
            items_moderation_masked=items_moderation_masked,
            decisions=decisions,
            latency_ms=latency_ms,
        )
        scrubbed_results = RankedResults(
            request_id=results.request_id,
            items=scrubbed_items,
            total_candidates=results.total_candidates,
            fusion_latency_ms=results.fusion_latency_ms,
            cache_hit=results.cache_hit,
            multi_intent_envelope=results.multi_intent_envelope,
            failure_mode=results.failure_mode,
            query_intent=results.query_intent,
        )
        # INFO log only when the gate actually mutated the result set; clean
        # passthroughs stay quiet to avoid log spam on healthy traffic.
        any_mutation = (items_dropped > 0 or items_pii_masked > 0 or items_explanation_scrubbed > 0 or items_moderation_masked > 0)
        if any_mutation:
            logger.info(
                f"egress_guard_applied request_id={results.request_id} items_in={items_in} "
                f"items_kept={items_in - items_dropped} items_dropped={items_dropped} "
                f"items_pii_masked={items_pii_masked} items_explanation_scrubbed={items_explanation_scrubbed} "
                f"items_moderation_masked={items_moderation_masked} "
                f"moderator={self._moderator.name} latency_ms={latency_ms:.2f}"
            )
        return scrubbed_results, outcome

    def _scrub_item(self, item: RankedItem, valid_item_ids: frozenset) -> Tuple[EgressItemAction, Optional[RankedItem]]:
        """Run all enabled checks against one item.

        Returns ``(action, new_item)`` where ``new_item`` is None iff the
        action is ``dropped_moderation``. The action category follows the
        precedence ``dropped_moderation > pii_masked > explanation_scrubbed > kept``.
        Multi-issue items have every triggering reason recorded in
        ``action.reasons``.
        """
        reasons: List[str] = []
        moderation_was_masked = False
        # Step 1 — moderation (runs first; a flagged item in drop-policy mode
        # is dropped immediately and the more-expensive PII / grounding work
        # is skipped. In mask-policy mode the moderator's fields are
        # redacted but the item survives so PII + grounding still apply).
        if self._moderator_enabled:
            verdict = self._moderator.moderate(item.payload)
            if verdict.flagged:
                # Truncated audit list is mirrored into the reasons surface so
                # SRE dashboards can detect partial-list matches without
                # having to introspect ModeratorVerdict directly.
                term_reasons = [f"banned_token={t}" for t in verdict.matched_terms]
                if verdict.truncated_matches:
                    term_reasons.append('matches_truncated')
                if self._moderator_policy == 'drop':
                    return (EgressItemAction(action='dropped_moderation', reasons=term_reasons), None,)
                # mask policy: redact the configured moderator payload fields
                # but keep the item visible. Falls through to step 2/3 so PII
                # + grounding checks still run on the surviving item.
                masked_payload = self._mask_moderator_fields(item.payload)
                item = self._with_payload(item, masked_payload)
                reasons.extend(term_reasons)
                reasons.append('moderator_mask_applied')
                moderation_was_masked = True
        # Step 2 — PII scrub.
        pii_was_masked = False
        if self._pii_enabled:
            new_payload, pii_match_count, pii_field_reasons = self._scrub_pii(item.payload)
            if pii_match_count > 0:
                pii_was_masked = True
                item = self._with_payload(item, new_payload)
                reasons.append(f'pii_matches={pii_match_count}')
                reasons.extend(pii_field_reasons)
        # Step 3 — grounding check (only when an explanation field is present).
        explanation_was_scrubbed = False
        if self._grounding_enabled and self._explanation_field in item.payload:
            new_payload, ungrounded_count = self._scrub_grounding(item.payload, valid_item_ids)
            if ungrounded_count > 0:
                explanation_was_scrubbed = True
                item = self._with_payload(item, new_payload)
                reasons.append(f'ungrounded_citations={ungrounded_count}')
        # Resolve the final action category by precedence:
        #     moderation_masked > pii_masked > explanation_scrubbed > kept
        # (dropped_moderation is handled inline above with an early return.)
        if moderation_was_masked:
            return EgressItemAction(action='moderation_masked', reasons=reasons), item
        if pii_was_masked:
            return EgressItemAction(action='pii_masked', reasons=reasons), item
        if explanation_was_scrubbed:
            return EgressItemAction(action='explanation_scrubbed', reasons=reasons), item
        # No mutation triggered by any enabled check — preserve the item.
        return EgressItemAction(action='kept', reasons=[]), item

    def _scrub_pii(self, payload: Mapping[str, Any]) -> Tuple[Dict[str, Any], int, List[str]]:
        """Walk configured payload fields, mask PII via the LayerZeroSanitizer regex set.

        Walks ``str`` values directly. For ``list``/``tuple`` values, walks
        each string element (non-string elements are left untouched). For
        ``dict`` values, recurses one level for string scalars (deeper
        nesting is intentionally NOT walked — payloads with deeply nested
        free text would cost more than they're worth to scrub here; if a
        payload schema needs deep walks, list each path explicitly in
        ``payload_fields``).

        :return: Tuple[Dict[str, Any], int, List[str]] - (new_payload, total_match_count, per-field reasons)
        """
        new_payload: Dict[str, Any] = dict(payload) if payload else {}
        total_matches = 0
        per_field_reasons: List[str] = []
        for fld in self._pii_fields:
            if fld not in new_payload:
                continue
            value = new_payload[fld]
            new_value, matches, oversize = self._scrub_value(value)
            if oversize:
                per_field_reasons.append(f"oversize_field_truncated={fld}")
            if matches > 0:
                new_payload[fld] = new_value
                total_matches += matches
                per_field_reasons.append(f"pii_field={fld}")
        return new_payload, total_matches, per_field_reasons

    def _scrub_value(self, value: Any) -> Tuple[Any, int, bool]:
        """Recursive helper: returns (new_value, match_count, oversize_truncated_flag).

        Only str / list / tuple / dict are walked. All other types pass through.
        """
        if isinstance(value, str):
            truncated = len(value) > self._pii_max_chars
            target = value[: self._pii_max_chars] if truncated else value
            verdict = self._sanitizer.sanitize(target)
            # We only care about masked_text; the verdict may report other
            # check failures (length / blocklist) but the PII pass is
            # specifically about replacing matched PII with the redaction
            # marker. The ``masked_text`` is always populated.
            masked = verdict.masked_text
            # Count matches by comparing pre/post — the sanitizer doesn't
            # expose a per-call PII counter, so we infer from the diff.
            # An exact equality check is sufficient because the regex pass
            # only mutates on a hit.
            match_count = 0 if masked == target else 1
            return masked, match_count, truncated
        if isinstance(value, list):
            new_list: List[Any] = []
            total = 0
            any_truncated = False
            for elem in value:
                new_elem, matches, trunc = self._scrub_value(elem)
                new_list.append(new_elem)
                total += matches
                any_truncated = any_truncated or trunc
            return new_list, total, any_truncated
        if isinstance(value, tuple):
            new_seq: List[Any] = []
            total = 0
            any_truncated = False
            for elem in value:
                new_elem, matches, trunc = self._scrub_value(elem)
                new_seq.append(new_elem)
                total += matches
                any_truncated = any_truncated or trunc
            return tuple(new_seq), total, any_truncated
        if isinstance(value, dict):
            new_dict: Dict[Any, Any] = {}
            total = 0
            any_truncated = False
            for k, v in value.items():
                new_v, matches, trunc = self._scrub_value(v)
                new_dict[k] = new_v
                total += matches
                any_truncated = any_truncated or trunc
            return new_dict, total, any_truncated
        # Pass-through for non-string scalars (int / float / bool / None).
        return value, 0, False

    def _scrub_grounding(self, payload: Mapping[str, Any], valid_item_ids: frozenset) -> Tuple[Dict[str, Any], int]:
        """Mask any item-id citation in the explanation field that is NOT in the result set.

        Returns ``(new_payload, ungrounded_count)``. Only the explanation
        field is mutated; the rest of the payload passes through.
        """
        explanation = payload[self._explanation_field]
        if not isinstance(explanation, str) or not explanation:
            return dict(payload), 0
        ungrounded = 0
        # Build the masked explanation by walking every citation match and
        # substituting unmasked spans verbatim, masked spans with the token.
        out_parts: List[str] = []
        last_end = 0
        for m in self._citation_re.finditer(explanation):
            cited_id = m.group(1)
            if cited_id in valid_item_ids:
                # Grounded — keep the original text intact.
                continue
            # Ungrounded citation — replace the entire match span with mask token.
            out_parts.append(explanation[last_end:m.start()])
            out_parts.append(self._mask_token)
            last_end = m.end()
            ungrounded += 1
        if ungrounded == 0:
            return dict(payload), 0
        out_parts.append(explanation[last_end:])
        new_payload = dict(payload)
        new_payload[self._explanation_field] = ''.join(out_parts)
        return new_payload, ungrounded

    def _mask_moderator_fields(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        """Replace each configured moderator payload field with ``[REDACTED]``.

        Only used on the ``policy='mask'`` path; the ``drop`` policy returns
        before reaching this helper. Non-existent fields are skipped silently.
        """
        new_payload: Dict[str, Any] = dict(payload) if payload else {}
        # The lexical sub-config is None on the noop backend; guard accordingly.
        lex_cfg = self._config.moderator.lexical
        fields = lex_cfg.payload_fields if lex_cfg is not None else []
        for fld in fields:
            if fld in new_payload:
                new_payload[fld] = '[REDACTED]'
        return new_payload

    @staticmethod
    def _with_payload(item: RankedItem, new_payload: Dict[str, Any]) -> RankedItem:
        """Return a reconstructed RankedItem identical to ``item`` but with ``new_payload``.

        The contracts dataclass is frozen-by-convention rather than
        ``frozen=True`` (it has list fields), so we re-construct rather
        than mutate in place. Re-construction also revalidates the item
        via ``RankedItem.__post_init__`` — a defense-in-depth check that
        a sub-step did not produce a malformed payload shape.
        """
        return RankedItem(item_id=item.item_id, fused_score=item.fused_score, contributing_sources=list(item.contributing_sources), payload=new_payload, sub_intent_ids=list(item.sub_intent_ids),)
