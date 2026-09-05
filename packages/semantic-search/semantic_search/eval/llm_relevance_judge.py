"""LLM-as-judge for relevance grading.

Produces ``RelevanceJudgment`` instances by asking a separate LLM (different
``task_type`` from the retrieval-side classifier per ``ai-eval-testing.mdc``)
to grade each ``(query, item)`` pair on the existing 0-4 graded-relevance
scale. The output composes into ``RelevanceJudgedQuery`` instances which
``RetrievalQualityEvaluator`` consumes unchanged — there is no
parallel evaluation path to keep in sync.

Two layers:

- ``LLMRelevanceJudge`` — judges ONE ``(query, item)`` pair. Soft-fail
  contract: returns ``None`` on LLM error, schema failure, or below-confidence
  grade so a single bad call does not sink the evaluator. Reuses
  ``LLMCallRouter.call_structured`` so the L0 sanitizer, structural gate,
  prompt-cache observability, circuit breaker, and ``cost_observer`` budget
  hook all apply automatically (no parallel LLM client).

- ``LLMJudgedQueryBuilder`` — composes per-item grades into a single
  ``RelevanceJudgedQuery``. Bulk-judges via ``asyncio.gather`` bounded by
  ``max_concurrent_judgments``. Returns ``None`` when no item earns
  ``gain >= 1`` (the contract requires at least one relevant judgment),
  so degenerate queries are skipped rather than raising.

Pydantic schema (``RelevanceGradeOutput``) is enforced by
``LLMCallRouter.call_structured`` — invalid JSON / out-of-range gain is a
parse failure, NOT a silently-clipped result.

PII / responsible-ai posture:

- The user-supplied query and item text travel through the L0 sanitizer at
  the ingress chokepoint (``LLMCallRouter._enforce_ingress_sanitizer``) — no
  bypass.
- We log only ``query_id``, ``item_id``, the integer ``gain``, the rounded
  ``confidence``, the ``model`` selected by the router, and the ``latency_ms``.
  Raw query / item text and free-text reasoning are NEVER logged.
- ``reviewer_id`` on the produced ``RelevanceJudgedQuery`` is
  ``llm_judge:<task_type>`` so downstream consumers can distinguish synthetic
  judgments from human ones at audit time.
"""
import asyncio
import time
from typing import List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, field_validator

from semantic_search.config.models import LLMJudgeConfig
from semantic_search.contracts import RelevanceJudgedQuery, RelevanceJudgment
from semantic_search.core.exceptions import AgentSearchError, LLMError, ValidationError
from semantic_search.core.llm_client import LLMCallRouter
from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)

_MIN_GAIN = 0
_MAX_GAIN = 4


class RelevanceGradeOutput(BaseModel):
    """Judge output: gain [0-4] + confidence [0-1] + reasoning (out-of-range = parse failure)."""

    gain: int = Field(ge=_MIN_GAIN, le=_MAX_GAIN)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1, max_length=512)

    @field_validator('gain', mode='before')
    @classmethod
    def _gain_is_int_not_bool(cls, v):
        # Pydantic v2 silently coerces bool -> int (True -> 1, False -> 0)
        # because bool is a Python int subclass. We forbid bool explicitly so
        # an LLM that returns the JSON literal `true` cannot smuggle in a
        # silently-clipped gain=1.
        if isinstance(v, bool):
            raise ValueError(f"gain must be an int (got bool {v})")
        return v


class LLMRelevanceJudge:
    """Grade (query, item) pair via LLM (soft-fail returns None on error)."""

    def __init__(self, config: LLMJudgeConfig, call_router: LLMCallRouter):
        if config is None:
            raise ValidationError("LLMRelevanceJudge requires a LLMJudgeConfig")
        if call_router is None:
            raise ValidationError("LLMRelevanceJudge requires a LLMCallRouter")
        self._config = config
        self._call_router = call_router

    @property
    def config(self) -> LLMJudgeConfig:
        return self._config

    async def judge(self, query: str, item_id: str, item_text: str) -> Optional[RelevanceJudgment]:
        """Grade one (query, item) pair.

        Soft-fail contract: returns ``None`` on (a) LLM error / parse failure
        (``LLMError``), (b) self-reported ``confidence < min_confidence``, or
        (c) any unexpected validation error. Callers (the builder) treat
        ``None`` as ungraded -- never as ``gain=0``.

        :param query: str - Raw user query (sanitized by the router ingress)
        :param item_id: str - Stable item id (matches ``RankedItem.item_id``)
        :param item_text: str - Item text body to grade
        :return: Optional[RelevanceJudgment] - Validated judgment or None
        """
        if not isinstance(query, str) or not query:
            raise ValidationError("LLMRelevanceJudge.judge requires non-empty query")
        if not isinstance(item_id, str) or not item_id:
            raise ValidationError("LLMRelevanceJudge.judge requires non-empty item_id")
        if not isinstance(item_text, str):
            raise ValidationError("LLMRelevanceJudge.judge requires str item_text")
        user_prompt = self._config.user_prompt_template.format(
            query=query, item_id=item_id, item_text=item_text,
        )
        t0 = time.monotonic()
        try:
            response, metadata = await self._call_router.call_structured(
                task_type=self._config.task_type,
                prompt_tag=self._config.prompt_tag,
                system_prompt=self._config.system_prompt,
                user_prompt=user_prompt,
                response_schema=RelevanceGradeOutput,
            )
        except asyncio.CancelledError:
            raise
        except LLMError as e:
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            logger.warning(
                f"llm_judge_call_failed item_id={item_id} elapsed_ms={elapsed_ms:.1f} "
                f"error_type={type(e).__name__}"
            )
            return None
        if response.confidence < self._config.min_confidence:
            logger.info(
                f"llm_judge_under_confident item_id={item_id} "
                f"confidence={response.confidence:.3f} min={self._config.min_confidence} "
                f"model={metadata.get('model')}"
            )
            return None
        try:
            judgment = RelevanceJudgment(item_id=item_id, gain=int(response.gain))
        except ValidationError as e:
            logger.warning(
                f"llm_judge_invalid_judgment item_id={item_id} gain={response.gain} "
                f"error={str(e)}"
            )
            return None
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        logger.info(
            f"llm_judge_grade item_id={item_id} gain={judgment.gain} "
            f"confidence={response.confidence:.3f} model={metadata.get('model')} "
            f"latency_ms={elapsed_ms:.1f}"
        )
        return judgment


class LLMJudgedQueryBuilder:
    """Compose per-item grades into a ``RelevanceJudgedQuery``.

    :param judge: LLMRelevanceJudge - Single-pair grader
    :param config: LLMJudgeConfig - Judge configuration (controls fan-out
        cap and concurrency)
    """

    def __init__(self, judge: LLMRelevanceJudge, config: LLMJudgeConfig):
        if judge is None:
            raise ValidationError("LLMJudgedQueryBuilder requires a LLMRelevanceJudge")
        if config is None:
            raise ValidationError("LLMJudgedQueryBuilder requires a LLMJudgeConfig")
        self._judge = judge
        self._config = config

    @property
    def reviewer_id(self) -> str:
        """Synthetic reviewer id stamped on every produced query."""
        return f"llm_judge:{self._config.task_type}"

    async def build(self, query_id: str, input_query: str, candidates: Sequence[Tuple[str, str]], difficulty: str, edge_type: str) -> Optional[RelevanceJudgedQuery]:
        """Judge each candidate and assemble a ``RelevanceJudgedQuery``.

        Returns ``None`` when (a) the LLM produces zero successful grades or
        (b) every successful grade has ``gain == 0`` (the contract requires
        at least one relevant judgment). This lets callers iterate over a
        large candidate set and skip degenerate queries cleanly.

        :param query_id: str - Stable id (caller controls; e.g.
            ``RelevanceJudgedQuery.new_query_id()``)
        :param input_query: str - Raw user query
        :param candidates: Sequence[Tuple[str, str]] - ``(item_id, item_text)``
            pairs to grade. The builder caps at
            ``config.max_items_per_query`` and de-duplicates by ``item_id``
            (first occurrence wins) so callers can pass ranker output
            directly without pre-processing.
        :param difficulty: str - One of ``GOLDEN_DIFFICULTIES``
        :param edge_type: str - One of ``GOLDEN_EDGE_TYPES``
        :return: Optional[RelevanceJudgedQuery] - Judged query or None
            when no relevant items were found
        """
        if not isinstance(query_id, str) or not query_id:
            raise ValidationError("LLMJudgedQueryBuilder.build requires non-empty query_id")
        if not isinstance(input_query, str) or not input_query:
            raise ValidationError("LLMJudgedQueryBuilder.build requires non-empty input_query")
        if candidates is None:
            raise ValidationError("LLMJudgedQueryBuilder.build requires non-None candidates")
        seen_ids: set = set()
        deduped: List[Tuple[str, str]] = []
        for entry in candidates:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise ValidationError("LLMJudgedQueryBuilder.build candidates must be (item_id, item_text) tuples")
            iid, text = entry
            if not isinstance(iid, str) or not iid:
                raise ValidationError("LLMJudgedQueryBuilder.build candidate item_id must be a non-empty string")
            if not isinstance(text, str):
                raise ValidationError("LLMJudgedQueryBuilder.build candidate item_text must be a string")
            if iid in seen_ids:
                continue
            seen_ids.add(iid)
            deduped.append((iid, text))
            if len(deduped) >= self._config.max_items_per_query:
                break
        if not deduped:
            logger.info(f"llm_judge_build_skip query_id={query_id} reason=no_candidates")
            return None
        sem = asyncio.Semaphore(self._config.max_concurrent_judgments)

        async def _grade(iid: str, text: str) -> Optional[RelevanceJudgment]:
            async with sem:
                return await self._judge.judge(input_query, iid, text)

        results = await asyncio.gather(
            *(_grade(iid, text) for iid, text in deduped),
            return_exceptions=True,
        )
        judgments: List[RelevanceJudgment] = []
        unexpected_errors = 0
        for entry, outcome in zip(deduped, results):
            iid, _text = entry
            if isinstance(outcome, asyncio.CancelledError):
                raise outcome
            if isinstance(outcome, BaseException):
                unexpected_errors += 1
                logger.warning(
                    f"llm_judge_unexpected_error query_id={query_id} item_id={iid} "
                    f"error_type={type(outcome).__name__}"
                )
                continue
            if outcome is None:
                continue
            judgments.append(outcome)
        relevant_count = sum(1 for j in judgments if j.gain >= 1)
        if relevant_count == 0:
            logger.info(
                f"llm_judge_build_skip query_id={query_id} reason=no_relevant "
                f"graded={len(judgments)} candidates={len(deduped)} errors={unexpected_errors}"
            )
            return None
        try:
            judged = RelevanceJudgedQuery(
                query_id=query_id,
                input_query=input_query,
                judgments=judgments,
                difficulty=difficulty,
                edge_type=edge_type,
                reviewer_id=self.reviewer_id,
            )
        except ValidationError as e:
            logger.warning(f"llm_judge_build_failed query_id={query_id} error={str(e)}")
            return None
        logger.info(
            f"llm_judge_build_ok query_id={query_id} candidates={len(deduped)} "
            f"graded={len(judgments)} relevant={relevant_count} "
            f"errors={unexpected_errors} reviewer_id={self.reviewer_id}"
        )
        return judged
