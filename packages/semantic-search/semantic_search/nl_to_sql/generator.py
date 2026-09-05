"""LLM-driven SQL generator (stage 2).

Routes through `LLMCallRouter.call_structured` so model selection, token logging,
cost accounting, and circuit-breaker integration come for free.

Discriminated response schema:

The Pydantic envelope is a discriminated union — the model commits to ONE of:

  - ``kind='answer'`` — produces a SQL query + confidence + explanation.
  - ``kind='refuse'`` — explicitly states the question cannot be answered
    safely from the pruned schema, with a structured ``reason`` and an
    optional ``suggested_clarification`` for the UI.

Pydantic emits ``oneOf`` + ``discriminator: kind`` for this Union which
activates ``LLMCallRouter`` capability pruning via
``ModelStructuralCapabilityRegistry.filter_capable_chain`` so only models
that honour discriminated decoding stay in the routing chain. A model
that emits hybrid shapes (both branches' fields) fails Pydantic
validation and falls through the normal LLM fallback chain. A model that
explicitly chooses ``kind='refuse'`` is surfaced as a typed
``LLMRefusalError`` and the pipeline records it as
``failure_mode='llm_refused'`` (distinct from the soft 'generation'
mode). Eliminates a class of silent SQL hallucinations on
unanswerable questions, since the model has a structural escape hatch
and ``responsible-ai.mdc`` "guardrails as pipeline stages" is honoured at
the schema level rather than via post-hoc detection.

When a `RetrievedContentSanitizer` is wired, every retrieved-content
fragment that re-enters the prompt — schema descriptions, schema sample
values, and the verifier's free-text `notes` from a prior failed attempt
(threaded back as `previous_error`) — is mask-or-passed before the prompt
is assembled. The system + structural prompt template itself is NOT
sanitized (it is code-owned, not retrieved).
"""
import time
from typing import Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from semantic_search.config.nl_to_sql_models import SqlGenerationConfig
from semantic_search.core.exceptions import LLMError, LLMRefusalError
from semantic_search.core.llm_client import LLMCallRouter
from semantic_search.core.logging_utils import get_logger
from semantic_search.nl_to_sql.content_sanitizer import RetrievedContentSanitizer
from semantic_search.nl_to_sql.contracts import PrunedSchema, SqlGenerationResult
from semantic_search.nl_to_sql.prompts import GENERATION_PROMPT_TAG, GENERATION_SYSTEM_PROMPT, build_generation_user_prompt
from semantic_search.nl_to_sql.schema import render_schema_for_prompt
from semantic_search.contracts import FeedbackSignal
from semantic_search.signal_store import SignalStore, schedule_feedback_signal_record

logger = get_logger(__name__)


class _SqlAnswerBranch(BaseModel):
    """SQL answer branch: sql + confidence + explanation."""
    kind: Literal['answer']
    sql: str = Field(min_length=1, max_length=8192)
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(default="", max_length=2048)


class _SqlRefuseBranch(BaseModel):
    """Refuse branch: reason + suggested_clarification (ops/UI surfaces)."""
    kind: Literal['refuse']
    reason: str = Field(min_length=1, max_length=2048)
    suggested_clarification: str = Field(default="", max_length=1024)


class _SqlGenerationResponse(BaseModel):
    """SQL generation response: answer branch | refuse branch (discriminated union).

    Pydantic renders ``decision`` as ``oneOf`` + ``discriminator: kind`` so
    the structural-capability gate prunes models that don't honour
    discriminated decoding before the call is dispatched.
    """
    decision: Annotated[
        Union[_SqlAnswerBranch, _SqlRefuseBranch],
        Field(discriminator='kind'),
    ]


class SqlGenerator:
    """Generates SQL from a natural-language question + pruned schema.

    :param config: SqlGenerationConfig - LLM tier config (task type, prompt tag, etc.)
    :param call_router: LLMCallRouter - Shared structured-call router
    :param dialect: str - SQL dialect name forwarded to the LLM (e.g. 'trino')
    :param max_rows: int - Forwarded to the LLM as MAX RESULT ROWS guidance
    """

    def __init__(
        self,
        config: SqlGenerationConfig,
        call_router: LLMCallRouter,
        dialect: str,
        max_rows: int,
        content_sanitizer: Optional[RetrievedContentSanitizer] = None,
        signal_store: Optional['SignalStore'] = None,
    ):
        if not isinstance(config, SqlGenerationConfig):
            raise LLMError("SqlGenerator requires a typed SqlGenerationConfig")
        if call_router is None:
            raise LLMError("SqlGenerator requires an LLMCallRouter")
        if not isinstance(dialect, str) or not dialect:
            raise LLMError("SqlGenerator requires a non-empty dialect")
        if int(max_rows) < 1:
            raise LLMError("SqlGenerator requires max_rows >= 1")
        if content_sanitizer is not None and not isinstance(content_sanitizer, RetrievedContentSanitizer):
            raise LLMError(
                "SqlGenerator.content_sanitizer must be a RetrievedContentSanitizer or None"
            )
        # Defer SignalStore type check to runtime to avoid an import cycle
        # (signal_store -> contracts -> nl_to_sql is fine; the reverse would
        # not be). Validating via isinstance against the concrete class keeps
        # the contract explicit at construction time.
        if signal_store is not None:
            if not isinstance(signal_store, SignalStore):
                raise LLMError("SqlGenerator.signal_store must be a SignalStore or None")
        self._config = config
        self._call_router = call_router
        self._dialect = dialect
        self._max_rows = int(max_rows)
        self._content_sanitizer = content_sanitizer
        self._signal_store = signal_store

    async def generate(self, question: str, sql_hint: str, pruned: PrunedSchema, previous_sql: str = "", previous_error: str = "", attempt: int = 1, request_id: str = "") -> SqlGenerationResult:
        """Run a single generation attempt and return the typed result.

        Retry orchestration is the pipeline's job — this method runs ONE attempt
        so the caller decides when to stop given the validator's verdict.

        :param question: str - Natural-language analytics question
        :param sql_hint: str - Optional structured QI hint
        :param pruned: PrunedSchema - Output of stage 1
        :param previous_sql: str - Last failed SQL (empty on first attempt)
        :param previous_error: str - Validator failure reason (empty on first attempt)
        :param attempt: int - 1-indexed attempt counter (forwarded into the result)
        :return: SqlGenerationResult - Typed wrapper around the LLM output
        :raises LLMError: When the call_router exhausts its fallback chain or
                          confidence is below the configured floor
        """
        if not isinstance(question, str) or not question:
            raise LLMError("SqlGenerator.generate requires a non-empty question")
        if not isinstance(pruned, PrunedSchema):
            raise LLMError("SqlGenerator.generate requires a PrunedSchema")
        if int(attempt) < 1:
            raise LLMError("SqlGenerator.generate requires attempt >= 1")
        # Sanitize EVERY retrieved-content fragment that re-enters the
        # prompt. The system prompt + structural template are
        # code-owned and intentionally NOT sanitized; only fragments
        # whose source is upstream stages (catalog, prior verifier
        # output) flow through the mask-or-pass gate.
        schema_block = render_schema_for_prompt(
            pruned,
            sanitizer=self._content_sanitizer,
            request_id=request_id,
        )
        sanitized_previous_error = previous_error or ""
        if self._content_sanitizer is not None and sanitized_previous_error:
            # The verifier's free-text `notes` from the prior failed attempt
            # is LLM output — untrusted from a prompt-injection standpoint.
            sanitized_previous_error = self._content_sanitizer.sanitize_fragment(
                sanitized_previous_error,
                kind='previous_error',
                request_id=request_id,
            )
        user_prompt = build_generation_user_prompt(
            dialect=self._dialect,
            max_rows=self._max_rows,
            schema=schema_block,
            question=question,
            sql_hint=sql_hint or "",
            previous_sql=previous_sql or "",
            previous_error=sanitized_previous_error,
        )
        forced = self._config.model_override if self._config.model_override else None
        t0 = time.monotonic()
        response, metadata = await self._call_router.call_structured(
            task_type=self._config.task_type,
            prompt_tag=self._config.prompt_tag or GENERATION_PROMPT_TAG,
            system_prompt=GENERATION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_schema=_SqlGenerationResponse,
            model_override=forced,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        usage = metadata.get('usage') or {}
        prompt_tokens = int(usage.get('prompt_tokens', 0))
        completion_tokens = int(usage.get('completion_tokens', 0))
        cost_usd = float(metadata.get('cost_usd', 0.0))
        model = str(metadata.get('model', ''))
        branch = response.decision
        if isinstance(branch, _SqlRefuseBranch):
            # Hard refusal — no fallback chain will produce a different
            # answer for this question/schema pair. Surface as a typed
            # LLMRefusalError; the pipeline maps it to
            # failure_mode='llm_refused'. Best-effort signal
            # emission for dashboards.
            logger.warning(
                f"sql_generation_refused attempt={attempt} model={model} "
                f"input_tokens={prompt_tokens} output_tokens={completion_tokens} "
                f"latency_ms={elapsed_ms:.1f} request_id={request_id or 'unknown'} "
                f"reason_length={len(branch.reason)}"
            )
            self._emit_refusal_signal(
                request_id=request_id,
                reason=branch.reason,
                suggested_clarification=branch.suggested_clarification,
                model=model,
                attempt=int(attempt),
            )
            raise LLMRefusalError(
                f"sql_generation_refused attempt={attempt} model={model}",
                reason=branch.reason,
                suggested_clarification=branch.suggested_clarification,
            )
        # _SqlAnswerBranch — Pydantic guarantees sql/confidence/explanation are present.
        if branch.confidence < self._config.min_confidence:
            logger.warning(
                f"sql_generation_under_confident attempt={attempt} "
                f"confidence={branch.confidence:.3f} min={self._config.min_confidence}"
            )
            raise LLMError(
                f"sql_generation_under_confident confidence={branch.confidence:.3f} "
                f"min={self._config.min_confidence}"
            )
        logger.info(
            f"sql_generation_ok attempt={attempt} model={model} "
            f"confidence={branch.confidence:.3f} "
            f"input_tokens={prompt_tokens} output_tokens={completion_tokens} "
            f"latency_ms={elapsed_ms:.1f}"
        )
        return SqlGenerationResult(
            sql=branch.sql.strip(),
            confidence=float(branch.confidence),
            model=model,
            attempt_count=int(attempt),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            latency_ms=elapsed_ms,
        )

    def _emit_refusal_signal(self, request_id: str, reason: str, suggested_clarification: str, model: str, attempt: int) -> None:
        """Best-effort emission of an `llm_refused` FeedbackSignal.

        Best-effort: the SignalStore is optional (legacy construction passes
        None) and a failure to record MUST NOT prevent the LLMRefusalError
        from being raised — the user-visible recovery is the pipeline's
        failure_mode='llm_refused' surface, not the signal.
        """
        if self._signal_store is None:
            return
        try:
            payload = {
                'reason': str(reason)[:512],
                'suggested_clarification': str(suggested_clarification)[:512],
                'model': model,
                'attempt': int(attempt),
            }
            signal = FeedbackSignal(
                signal_id=FeedbackSignal.new_signal_id(),
                request_id=request_id or 'sql_generator_refusal',
                signal_type='llm_refused',
                payload=payload,
                signal_origin='analytics_router',
            )
            schedule_feedback_signal_record(self._signal_store, signal)
        except Exception as e:
            logger.warning(
                f"sql_generation_refusal_signal_emit_failed request_id={request_id or 'unknown'} "
                f"error_type={type(e).__name__} error={str(e)}"
            )


__all__ = ['SqlGenerator']
