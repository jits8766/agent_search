"""Per-fragment Layer-0 sanitizer for retrieved content re-entering NL-SQL prompts.

The Layer-0 ingress sanitizer defines a 500-char-capped policy for *user inputs* (queries
re-entering Tier-3 LLM, refinement candidate payloads, etc.). The ingress
sanitizer at `LLMCallRouter._enforce_ingress_sanitizer` enforces that contract
end-to-end on full prompts — fail-closed on the first match (length cap,
blocked substring, PII pattern).

That fail-closed policy is correct at the LLM ingress (a single matched check
on a full prompt almost always means a real attack or a real malformed call),
but it is the WRONG policy for retrieved-content fragments. Three NL-SQL
producer sites read content from upstream stages and stitch it into LLM
prompts:

  1. ``render_schema_for_prompt(pruned)`` — column descriptions + sample
     values rendered from `information_schema` rows. Sample values can carry
     user-controlled text (e.g. a `tld_name`, `bio`, or `description`
     column).
  2. ``Verifier._render_sample(execution)`` — actual data rows from the
     executed SQL. **Highest-risk vector:** raw user-controlled DB content
     flowing directly into the verifier prompt. The verifier system prompt
     forbids echoing row data into the audit notes, but enforcement is
     hope-based without a structural sanitizer.
  3. ``SqlGenerator.generate(..., previous_error=...)`` — the verifier's
     own free-text `notes` from a prior failed attempt. LLM output is
     untrusted from a prompt-injection standpoint and must be sanitized
     before being fed back into the next generation prompt.

If we routed these fragments through the ingress sanitizer they would
either bust the 500-char cap (schema dumps are routinely >5 kB) or
fail-close the entire analytics request on a single poisoned cell. The
right policy is **mask + audit, not block**: replace any flagged fragment
with the configured `mask_replacement` token, emit a typed
`retrieved_content_sanitized` audit signal, and let the analytics request
continue with the redacted prompt. Operators investigate the source.

Layer rules (per `architecture.mdc`):

    stdlib + `core` + `contracts` + `signal_store` + `safety.layer_zero_sanitizer`.

This module is consumed by `nl_to_sql.{schema, verifier, generator}` and stays
cycle-free relative to the search orchestrator.
"""
from typing import List, Optional

from semantic_search.config.nl_to_sql_models import RetrievedContentSanitizerConfig
from semantic_search.contracts import FeedbackSignal
from semantic_search.core.exceptions import ValidationError
from semantic_search.core.logging_utils import get_logger
from semantic_search.signal_store import SignalStore, schedule_feedback_signal_record
from semantic_search.safety.layer_zero_sanitizer import LayerZeroSanitizer

logger = get_logger(__name__)


_FRAGMENT_KINDS = frozenset({
    'schema_description',
    'schema_sample_value',
    'verifier_sample_row',
    'previous_error',
})


class RetrievedContentSanitizer:
    """Mask-and-audit Layer-0 gate for NL-SQL retrieved content.

    Wraps `LayerZeroSanitizer` with NL-SQL-specific policy:

      - **Per-fragment max length** (`max_chars_per_fragment`) decoupled
        from the conversational `safety.ingress_sanitizer.max_chars` cap so
        schema dumps and sample rows can run far longer than 500 chars
        without bursting the ingress gate. Anything over the per-fragment
        cap is HARD-truncated with a `[TRUNCATED]` suffix and then
        re-checked.
      - **Mask-on-block** instead of fail-closed. If any check matches,
        the fragment is replaced wholesale with `mask_replacement` and a
        single `retrieved_content_sanitized` signal is emitted (when a
        `signal_store` is wired). The PII-mask path of the underlying
        `LayerZeroSanitizer` (which only masks PII regex hits, not
        blocked substrings) is intentionally NOT used here because we
        cannot leak a partial blocked-substring back into the prompt.

    The class is stateless beyond the cached config + delegate sanitizer
    + signal store — safe for concurrent use across pipeline tasks.

    :param config: RetrievedContentSanitizerConfig - Per-fragment caps,
        mask replacement, audit toggle
    :param sanitizer: LayerZeroSanitizer - Underlying check engine
    :param signal_store: Optional[SignalStore] - Audit channel for
        `retrieved_content_sanitized` signals. None = log-only.
    :raises ValidationError: When `config` / `sanitizer` are wrong type
    """

    def __init__(self, config: RetrievedContentSanitizerConfig, sanitizer: LayerZeroSanitizer, signal_store: Optional[SignalStore] = None):
        if not isinstance(config, RetrievedContentSanitizerConfig):
            raise ValidationError(
                "RetrievedContentSanitizer requires a RetrievedContentSanitizerConfig"
            )
        if not isinstance(sanitizer, LayerZeroSanitizer):
            raise ValidationError(
                "RetrievedContentSanitizer requires a LayerZeroSanitizer"
            )
        if signal_store is not None and not isinstance(signal_store, SignalStore):
            raise ValidationError(
                "RetrievedContentSanitizer.signal_store must be a SignalStore or None"
            )
        self._config = config
        self._sanitizer = sanitizer
        self._signal_store = signal_store
        self._mask = str(config.mask_replacement)
        self._max_chars = int(config.max_chars_per_fragment)
        self._emit_audit_signal = bool(config.emit_audit_signal)
        logger.info(
            f"retrieved_content_sanitizer_initialized "
            f"max_chars_per_fragment={self._max_chars} "
            f"signal_store_wired={signal_store is not None} "
            f"emit_audit_signal={self._emit_audit_signal}"
        )

    @property
    def enabled(self) -> bool:
        """Whether sanitization runs (mirrors the underlying config toggle)."""
        return bool(self._config.enabled and self._sanitizer is not None)

    def sanitize_fragment(self, fragment: str, kind: str, request_id: str = '') -> str:
        """Run the underlying Layer-0 sanitizer on `fragment` and mask-or-pass.

        Truncation rule: a fragment longer than `max_chars_per_fragment` is
        first hard-truncated to that length with a trailing `[TRUNCATED]`
        marker. The truncated fragment is then run through the underlying
        sanitizer's blocklist + PII checks. This guarantees the underlying
        sanitizer's own length check (which fires at `safety.ingress_sanitizer.
        max_chars` — typically 500) NEVER fires on a fragment that is
        over the conversational cap but under the per-fragment cap.

        :param fragment: str - Text to sanitize (None / non-string -> '')
        :param kind: str - One of `_FRAGMENT_KINDS` (audit metadata)
        :param request_id: str - Originating request id (audit metadata only)
        :return: str - Either the original (or hard-truncated) fragment
            when every check passed, or `mask_replacement` when any check
            blocked. Never raises on the hot path.
        """
        if kind not in _FRAGMENT_KINDS:
            raise ValidationError(
                f"RetrievedContentSanitizer.sanitize_fragment.kind must be "
                f"one of {sorted(_FRAGMENT_KINDS)}"
            )
        if not self.enabled:
            return fragment if isinstance(fragment, str) else ''
        if fragment is None or not isinstance(fragment, str):
            return ''
        # Hard-truncate above the per-fragment cap so the underlying
        # sanitizer's own length check (typically 500) never fires on a
        # legitimately-large schema dump or sample row. The truncation
        # marker is appended deterministically so downstream prompt
        # producers can spot it.
        if len(fragment) > self._max_chars:
            fragment = fragment[: self._max_chars] + ' [TRUNCATED]'
        verdict = self._sanitizer.sanitize(fragment, max_chars=self._max_chars)
        if verdict.passed:
            return fragment
        # MASK, do not block. A single poisoned cell must not sink the
        # whole analytics request — the right action is to neutralize and
        # let the operator investigate via the audit channel.
        reasons: List[str] = list(verdict.reasons)
        logger.warning(
            f"retrieved_content_sanitized kind={kind} reasons={reasons} "
            f"input_length={len(fragment)} request_id={request_id or 'unknown'}"
        )
        if self._emit_audit_signal and self._signal_store is not None:
            self._emit_signal(kind, reasons, len(fragment), request_id)
        return self._mask

    def _emit_signal(self, kind: str, reasons: List[str], input_length: int, request_id: str) -> None:
        """Emit a `retrieved_content_sanitized` signal for dashboards.

        Best-effort: a failure here MUST NOT prevent the masking from
        taking effect (the prompt is already redacted).
        """
        try:
            signal = FeedbackSignal(
                signal_id=FeedbackSignal.new_signal_id(),
                request_id=request_id or 'retrieved_content_sanitizer',
                signal_type='retrieved_content_sanitized',
                payload={
                    'kind': kind,
                    'reasons': reasons,
                    'input_length': int(input_length),
                    'mask_replacement': self._mask,
                },
                signal_origin='analytics_router',
            )
            schedule_feedback_signal_record(self._signal_store, signal)
        except Exception as e:
            # Per `responsible-ai.mdc` §"Explainability" — never lose the
            # audit trail on a downstream signal-store error.
            logger.error(
                f"retrieved_content_sanitized_signal_emission_failed "
                f"kind={kind} error_type={type(e).__name__} error={e}"
            )


__all__ = ['RetrievedContentSanitizer']
