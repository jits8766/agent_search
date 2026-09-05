"""LLM classifier — sole classification tier in the QI engine.
Uses `LLMCallRouter.call_structured` so all calls flow through `LLMProvider` and
are validated against a Pydantic schema.

Structural commitment:
    The model must commit STRUCTURALLY to one of two classification shapes via a
    discriminated union:

      - ``kind='single'`` — one primary intent slice, optional 2-3 distinct
        alternative readings. Multi-intent decomposition is NOT permitted here.
      - ``kind='multi'``  — 2-5 sibling intent slices. Alternatives are NOT
        permitted (chip-strip envelope handles multi-intent ambiguity).

    Pydantic emits ``oneOf`` + ``discriminator: kind`` which activates
    ``ModelStructuralCapabilityRegistry.filter_capable_chain`` so only models
    that honour discriminated decoding stay in the routing chain.

Hard timeout:
    ``classify_with_prompt`` wraps the LLM call in ``asyncio.wait_for``. On
    ``asyncio.TimeoutError`` the classifier raises ``LLMError`` and the
    QIEngine falls back to the configured fallback query_type. A typed
    ``llm_timeout`` FeedbackSignal is emitted when a SignalStore is wired.
"""
import asyncio
import datetime
import re
import time
import uuid

_UTC_ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
from typing import Any, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator
from typing_extensions import Annotated

from semantic_search.config.models import QILLMConfig
from semantic_search.contracts import QUERY_TYPES, FeedbackSignal, IntentSlice
from semantic_search.core.exceptions import LLMError
from semantic_search.core.llm_client import LLMCallRouter
from semantic_search.core.logging_utils import get_logger
from semantic_search.signal_store import SignalStore, schedule_feedback_signal_record
from semantic_search.qi.prompts import PROMPT_TAG, build_system_prompt, build_user_prompt

logger = get_logger(__name__)

# Suffix pattern for slot names that structurally resemble hard filter slots
# (e.g. _min / _max / _exact). Unknown names matching this pattern are treated
# as soft but warrant a warning — they are likely hallucinated filter slots.
_HARD_SLOT_SUFFIX_PATTERN = re.compile(
    r'_(min|max|exact|contains|starts_with|ends_with|phrase|exclude|mode|is_unknown|list|currency)$'
)


def infer_chip_kind(entity_name: str, hard_entity_names: frozenset) -> str:
    """Classify entity slot as 'hard' or 'soft' using config-driven hard name set.

    :param entity_name: str - Internal entity slot name
    :param hard_entity_names: frozenset - From qi.entity_slots.hard_entity_names (required)
    """
    if not isinstance(hard_entity_names, frozenset):
        raise TypeError("infer_chip_kind requires hard_entity_names as frozenset from qi.entity_slots")
    if entity_name in hard_entity_names:
        return 'hard'
    # Soft keyword/topic slots share some hard-looking suffixes (_contains/_phrase);
    # only warn for unknown min/max/list/currency names that likely belong on hard.
    if re.search(r'_(min|max|list|currency)$', entity_name):
        logger.warning(
            f"qi_entity_name_unknown_hard_pattern name={entity_name!r} chip_kind=soft"
        )
    return 'soft'


class _LLMSlice(BaseModel):
    """One intent slice (query_type + confidence)."""
    raw_text: str = Field(min_length=1, max_length=512)
    query_type: str
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator('query_type')
    @classmethod
    def _validate_qt(cls, v: str) -> str:
        if v not in QUERY_TYPES:
            raise ValueError(f"query_type must be one of {sorted(QUERY_TYPES)}")
        return v


class _SingleIntentBranch(BaseModel):
    """One primary slice + 0-5 alternative interpretations (same query, not sub-queries)."""
    kind: Literal['single']
    primary: _LLMSlice
    alternative_interpretations: List[_LLMSlice] = Field(default_factory=list, max_length=5)


class _MultiIntentBranch(BaseModel):
    """2-5 sibling slices (no alternatives; chip-strip handles multi-intent ambiguity)."""
    kind: Literal['multi']
    slices: List[_LLMSlice] = Field(min_length=2, max_length=5)


class QIClassificationResponse(BaseModel):
    """Discriminated union: single vs multi intent (pydantic oneOf + discriminator).

    The flat-shape predecessor (a sibling ``query_type`` enum + ``slices``
    list + ``alternative_interpretations``) is gone — the Pydantic-level
    discriminator gives us per-branch field validation that the flat shape
    could only enforce via post-hoc consistency checks.
    """
    decision: Annotated[
        Union[_SingleIntentBranch, _MultiIntentBranch],
        Field(discriminator='kind'),
    ]


class LLMClassifier:
    """L2 LLM-based query classifier.

    :param config: QILLMConfig - LLM tier config (task type, prompt tag, min confidence, entity cap)
    :param call_router: LLMCallRouter - Shared LLM call router that owns model selection + cost logging
    :param allowed_query_types: List[str] - Closed enum the LLM must choose from
    :param alt_band_high: float - Upper edge of the
        confidence band that triggers ``alternative_interpretations``. The model is
        instructed to emit alternatives only when its top-level confidence is below
        this value. Must be in (0, 1]; the registry passes ``qi.llm.alternative_band_high``.
    :param max_alternatives: int - Hard cap on alternatives propagated downstream
        (pre-dedupe). Set from ``qi.llm.max_alternative_interpretations`` to bound per-call allocation.
    """

    def __init__(self, config: QILLMConfig, call_router: LLMCallRouter, allowed_query_types: List[str], alt_band_high: float, max_alternatives: int, signal_store: Optional[SignalStore] = None):
        self._config = config
        self._call_router = call_router
        self._allowed_query_types = list(allowed_query_types)
        if not self._allowed_query_types:
            raise LLMError("LLMClassifier requires non-empty allowed_query_types")
        if not 0.0 < float(alt_band_high) <= 1.0:
            raise LLMError(f"LLMClassifier requires alt_band_high in (0, 1], got {alt_band_high}")
        if int(max_alternatives) < 0:
            raise LLMError(f"LLMClassifier requires max_alternatives >= 0, got {max_alternatives}")
        self._alt_band_high = float(alt_band_high)
        self._max_alternatives = int(max_alternatives)
        # Sink for `llm_timeout` FeedbackSignals. Optional —
        # legacy construction (eval pipelines / unit tests) passes None and
        # timeouts still fall back correctly, just without the dashboard signal.
        self._signal_store = signal_store
        # Counter for ops visibility (also exposed via property).
        self._timeout_count = 0

    def _slice_to_intent(self, s: _LLMSlice) -> IntentSlice:
        """Convert a single _LLMSlice into a typed IntentSlice. Entities empty — produced by L0."""
        return IntentSlice(query_type=s.query_type, entities=[], confidence=float(s.confidence), raw_text=s.raw_text)

    def _normalize_response(self, response: QIClassificationResponse) -> Tuple[List[IntentSlice], float, List[_LLMSlice]]:
        """Flatten the discriminated union into the legacy (slices, top_confidence, raw_alternatives) shape.

        The discriminator decides:
          - ``kind='single'`` → ``[primary]`` slice list, ``primary.confidence``
            top-level, raw ``alternative_interpretations`` forwarded to
            :meth:`_filter_alternatives`.
          - ``kind='multi'``  → all ``slices`` returned, max-confidence as
            top-level, alternatives forced empty (chip-strip envelope is the
            surface for multi-intent ambiguity, not Pattern-A cards).

        :param response: QIClassificationResponse - Pydantic-validated payload
        :return: (slices, top_confidence, raw_alternatives)
            - ``slices`` is non-empty (Pydantic ensures it via branch min_length).
            - ``top_confidence`` ∈ [0,1].
            - ``raw_alternatives`` is the unfiltered list from the single branch
              (empty for multi). Caller passes this to :meth:`_filter_alternatives`.
        """
        branch = response.decision
        if isinstance(branch, _SingleIntentBranch):
            primary = self._slice_to_intent(branch.primary)
            return [primary], float(branch.primary.confidence), list(branch.alternative_interpretations)
        # _MultiIntentBranch — Pydantic guarantees min_length=2.
        slices = [self._slice_to_intent(s) for s in branch.slices]
        top_confidence = max((s.confidence for s in slices), default=0.0)
        return slices, float(top_confidence), []

    def _filter_alternatives(self, raw_alternatives: List[_LLMSlice], top_confidence: float, primary_slices: List[IntentSlice]) -> List[IntentSlice]:
        """Filter + dedupe + cap alternative interpretations from a single-intent branch.

        Surfaced ONLY when:
          1. ``primary_slices`` carries exactly one slice (multi-intent never reaches here).
          2. ``top_confidence`` is in the low band (< ``alt_band_high``).
          3. Each alternative has a query_type distinct from the primary and all
             previously-kept alternatives. Near-duplicates are dropped.

        :param raw_alternatives: List[_LLMSlice] - Pre-filtered list from the single-intent branch
        :param top_confidence: float - The model's primary confidence (driver of band check)
        :param primary_slices: List[IntentSlice] - The slices already kept as the primary classification
        :return: List[IntentSlice] - At most ``self._max_alternatives`` distinct-query-type alternatives
        """
        if self._max_alternatives <= 0:
            return []
        if len(primary_slices) != 1:
            return []
        if top_confidence >= self._alt_band_high:
            return []
        if not raw_alternatives:
            return []

        seen_query_types = {primary_slices[0].query_type}
        kept: List[IntentSlice] = []
        for alt in raw_alternatives:
            if len(kept) >= self._max_alternatives:
                break
            if alt.query_type in seen_query_types:
                continue
            seen_query_types.add(alt.query_type)
            kept.append(IntentSlice(query_type=alt.query_type, entities=[], confidence=float(alt.confidence), raw_text=alt.raw_text, slice_id=IntentSlice.new_slice_id()))
        return kept

    async def classify(self, query: str, always_return: bool = False) -> Optional[Tuple[List[IntentSlice], float, str, Any, List[IntentSlice]]]:
        """Run the LLM classifier with the production prompt.
        :param query: str - Normalized query text
        :param always_return: bool - When True, return slices even when top_confidence
            is below min_confidence. Default False preserves the gate for eval callers.
        :return: Optional[Tuple[List[IntentSlice], float, str, Any, List[IntentSlice]]]
                 (slices, top_confidence, model_used, usage, alternative_interpretations)
                 when the response passes the minimum-confidence gate (or always_return=True).
                 None when the LLM is disabled or under-confident (and always_return=False).
        :raises LLMError: When the underlying LLM call exhausts its fallback chain
        """
        if not self._config.enabled:
            return None
        if not query:
            return None
        system_prompt = build_system_prompt(alt_band_high=self._alt_band_high, keyword_expansion_max_terms=self._config.keyword_expansion_max_terms)
        _utc_now = datetime.datetime.now(datetime.timezone.utc).strftime(_UTC_ISO_FORMAT)
        user_prompt = build_user_prompt(query=query, allowed_query_types=self._allowed_query_types, current_utc_iso=_utc_now)
        return await self.classify_with_prompt(system_prompt=system_prompt, user_prompt=user_prompt, prompt_tag=PROMPT_TAG, model_override=None, always_return=always_return)

    async def classify_with_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        prompt_tag: str,
        model_override: Optional[str],
        always_return: bool = False,
    ) -> Optional[Tuple[List[IntentSlice], float, str, Any, List[IntentSlice]]]:
        """Run the LLM classifier with a caller-supplied prompt pair (eval / refinement use).

        Public because components (refinement pipeline + baselines) need to
        score candidate prompts and forced model overrides without reaching into
        LLMClassifier's private state. The schema and confidence gate stay identical
        to `classify()` so eval results are comparable.

        :param system_prompt: str - System role content (must be non-empty)
        :param user_prompt: str - User role content (must be non-empty)
        :param prompt_tag: str - Stable identifier used in LLM logs
        :param model_override: Optional[str] - Forced model name (None = use task fallback chain).
                               Empty string is treated as None so YAML defaults can disable forcing.
        :return: Optional[Tuple[List[IntentSlice], float, str, Any, List[IntentSlice]]] -
                 Same shape as ``classify()``: (slices, top_confidence, model_used, usage,
                 alternative_interpretations). Eval pipelines that score Pattern-A coverage
                 read the 5th element directly without re-invoking the LLM.
        :raises LLMError: When the underlying LLM call exhausts its chain
        """
        if not self._config.enabled:
            return None
        if not isinstance(system_prompt, str) or not system_prompt:
            raise LLMError("classify_with_prompt requires a non-empty system_prompt")
        if not isinstance(user_prompt, str) or not user_prompt:
            raise LLMError("classify_with_prompt requires a non-empty user_prompt")
        if not isinstance(prompt_tag, str) or not prompt_tag:
            raise LLMError("classify_with_prompt requires a non-empty prompt_tag")
        forced = model_override if (isinstance(model_override, str) and model_override) else None
        timeout_seconds = float(self._config.tier_3_timeout_seconds)
        started = time.monotonic()
        try:
            response, metadata = await asyncio.wait_for(
                self._call_router.call_structured(
                    task_type=self._config.task_type,
                    prompt_tag=prompt_tag,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_schema=QIClassificationResponse,
                    model_override=forced,
                ),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - started
            self._timeout_count += 1
            self._emit_timeout_signal(prompt_tag=prompt_tag, elapsed_seconds=elapsed, model_override=forced)
            logger.warning(f"qi_llm_timeout prompt_tag={prompt_tag} elapsed_seconds={elapsed:.3f} timeout_seconds={timeout_seconds:.3f} model_override={forced or ''}")
            raise LLMError(f"llm_tier3_timeout prompt_tag={prompt_tag} elapsed_seconds={elapsed:.3f} timeout_seconds={timeout_seconds:.3f}")
        slices, top_confidence, raw_alternatives = self._normalize_response(response)
        if top_confidence < self._config.min_confidence:
            logger.warning(f"qi_llm_under_confident prompt_tag={prompt_tag} confidence={top_confidence:.3f} min={self._config.min_confidence} always_return={always_return}")
            if not always_return:
                return None
        if not slices:
            # Defensive — Pydantic branch min_length guarantees this is unreachable,
            # but the guard keeps the contract explicit and survives schema edits.
            return None
        alternatives = self._filter_alternatives(raw_alternatives=raw_alternatives, top_confidence=top_confidence, primary_slices=slices)
        if alternatives:
            logger.info(f"qi_llm_alternatives_emitted prompt_tag={prompt_tag} count={len(alternatives)} top_confidence={top_confidence:.3f}")
        model_used = str(metadata.get('model', ''))
        usage = metadata.get('usage', {})
        return slices, float(top_confidence), model_used, usage, alternatives

    def _emit_timeout_signal(self, prompt_tag: str, elapsed_seconds: float, model_override: Optional[str]) -> None:
        """Best-effort emission of a `llm_timeout` FeedbackSignal.

        :param prompt_tag: str - Stable prompt identifier the timeout fired against
        :param elapsed_seconds: float - Wall-clock time spent waiting before timeout fired
        :param model_override: Optional[str] - Forced model when the timeout originated
            from an eval/refinement call; None when the routing chain was used.

        Best-effort because the SignalStore is optional (legacy
        construction passes None) and a failure to record MUST NOT prevent
        the LLMError from being raised — the fallback path is the actual
        user-visible recovery, the signal is the dashboard breadcrumb.
        The emitting site has no request_id context (the engine's request_id
        does not propagate into the classifier today), so we synthesize a
        deterministic synthetic id with the `qi_llm_timeout-` prefix.
        ``signal_origin`` is set to
        ``'qi_engine'`` because the timeout fires inside the QI cascade
        regardless of which prompt_tag triggered it.
        """
        if self._signal_store is None:
            return
        try:
            payload = {
                'prompt_tag': prompt_tag,
                'timeout_seconds': float(self._config.tier_3_timeout_seconds),
                'elapsed_seconds': float(elapsed_seconds),
                'model_override': model_override or '',
                'task_type': self._config.task_type,
            }
            signal = FeedbackSignal(
                signal_id=FeedbackSignal.new_signal_id(),
                request_id=f"qi_llm_timeout-{uuid.uuid4().hex[:12]}",
                signal_type='llm_timeout',
                payload=payload,
                signal_origin='qi_engine',
            )
            schedule_feedback_signal_record(self._signal_store, signal)
        except Exception as e:
            logger.warning(f"qi_llm_timeout_signal_emit_failed prompt_tag={prompt_tag} error_type={type(e).__name__} error={str(e)}")

    @property
    def min_confidence(self) -> float:
        """Minimum query-type confidence gate (below this T2 is considered under-confident)."""
        return float(self._config.min_confidence)

    @property
    def timeout_count(self) -> int:
        """Total number of Tier-3 hard timeouts observed by this classifier."""
        return int(self._timeout_count)

    @property
    def task_type(self) -> str:
        """The task type configured for this classifier (used by eval pipelines)."""
        return self._config.task_type

    @property
    def allowed_query_types(self) -> List[str]:
        """The closed enum of query types this classifier is allowed to emit."""
        return list(self._allowed_query_types)

    @property
    def alt_band_high(self) -> float:
        """Upper edge of the alternative-interpretation band."""
        return self._alt_band_high

    @property
    def max_alternatives(self) -> int:
        """Hard cap on emitted alternatives (pre-dedupe)."""
        return self._max_alternatives


