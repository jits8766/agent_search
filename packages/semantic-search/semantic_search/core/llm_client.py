"""LLM call helpers for semantic_search.
All business code routes LLM calls through `LLMCallRouter.call_structured` which:
- selects a model via the configured task type (LLMProvider fallback chain),
- enforces a Pydantic schema on the response (constrained-decoding-friendly),
- logs token usage / cost, and
- raises `LLMError` on parse / schema / call failures.
- consults the optional CircuitBreaker before each call and records the outcome
  on both the breaker and the BackendHealthRegistry.
- consults the optional
  ``ModelStructuralCapabilityRegistry`` BEFORE the per-model loop when the
  response schema carries a discriminated union; non-capable models are pruned
  from the fallback chain via live probe + TTL cache. An empty pruned chain
  raises ``LLMError('llm_no_oneof_capable_models …')`` so the operator is
  forced to fix config rather than serve silently-degraded responses.
- when an optional ``LayerZeroSanitizer`` is wired
  in and ``applies_to_llm_ingress`` is true, every ``system_prompt`` and
  ``user_prompt`` is gated through the sanitizer BEFORE any provider call
  fires. A block raises ``LLMError('llm_input_blocked_by_sanitizer …')`` —
  fail fast, no silent passthrough. Closes the indirect prompt-injection
  vector at the single chokepoint shared by Tier-3, NL-SQL gen + verifier,
  and refinement.
- per-call ``cached_input_tokens`` are aggregated into a thread-safe
  counter so the ``prompt_cache_hit_rate`` proxy signal is computed from
  real provider metadata, not inferred from cost.
"""
import asyncio
import contextvars
import json
import threading
import time
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, Tuple, Type, TypeVar, Union, runtime_checkable

from pydantic import BaseModel, Field, ValidationError as PydanticValidationError
from typing_extensions import Annotated

from semantic_search.core.exceptions import LLMError
from semantic_search.core.logging_utils import get_logger
from semantic_search.core.llm_provider import LLMProvider
from semantic_search.core.structural_gate import ModelStructuralCapabilityRegistry, schema_requires_oneof
from semantic_search.resilience.circuit_breaker import CircuitBreaker, CircuitOpenError
from semantic_search.resilience.health import BackendHealthRegistry

_RETRYABLE_TRANSPORT_ERRORS: Tuple[type, ...] = (LLMError, asyncio.TimeoutError, TimeoutError, ConnectionError, OSError, RuntimeError, ValueError)

logger = get_logger(__name__)

T = TypeVar('T', bound=BaseModel)

# per-call cost callback. Receives the USD cost the
# router computed for the just-completed (and successfully validated) LLM
# call. Invoked AFTER schema validation succeeds so the caller never
# accumulates spend on a request that will be retried or rejected.
# Implementations MUST be:
#   - synchronous (no await),
#   - cheap (no network / disk),
#   - free to raise — the router lets the exception propagate so a
#     QueryCostBudget breach surfaces immediately to the orchestrator.
# When the callback raises, the LLM result is discarded BEFORE return:
# the caller can't accidentally consume an over-budget response.
CostObserver = Callable[[str, float, Dict[str, Any]], None]

# request-scoped cost observer. The orchestrator sets this
# ContextVar at the top of ``search()`` to the bound ``QueryCostBudget``
# observer; the router consults it on every successful call_structured
# return when no explicit ``cost_observer=`` was passed. Using a
# ContextVar (vs. plumbing the observer through every signature) means:
#   - Internal LLM call sites (LLMClassifier, NL-SQL generator/verifier)
#     stay untouched — no caller-side opt-in needed.
#   - asyncio tasks spawned from the orchestrator inherit the binding
#     automatically (PEP 567 semantics).
#   - The contextvar is per-Task, so concurrent requests CANNOT bleed
#     cost into each other's budgets even though the router is shared.
# When unset (default None) the router behaves exactly as before:
# zero-cost overhead, no cost tracking.
_REQUEST_COST_OBSERVER: contextvars.ContextVar[Optional[CostObserver]] = contextvars.ContextVar('semantic_search_llm_cost_observer', default=None)


def set_request_cost_observer(observer: Optional[CostObserver]) -> contextvars.Token:
    """Bind a request-scoped LLM cost observer in the current context.

    :param observer: Optional[CostObserver] - Callback invoked after every successful
        ``call_structured`` return. Pass ``None`` to clear the binding.
    :return: contextvars.Token - Pass to ``reset_request_cost_observer`` to restore
        the previous binding (use try/finally so a raised exception cleans up).
    """
    return _REQUEST_COST_OBSERVER.set(observer)


def reset_request_cost_observer(token: contextvars.Token) -> None:
    """Restore the previous cost-observer binding using a token from ``set_request_cost_observer``."""
    _REQUEST_COST_OBSERVER.reset(token)




# ----------------------------------------------------------------------
# Structural-gate probe payload.
# ----------------------------------------------------------------------
# A minimal *tagged-union* schema we use to probe each model. We deliberately
# keep this independent of the production schema so the probe is one well-known
# JSON shape per model (cache key = (model, production_schema_signature)) —
# the gate's job is to detect whether the model honours the ``oneOf`` +
# ``discriminator`` *contract*, not to redo the production decoding.
class _StructuralProbeOptionA(BaseModel):
    kind: Literal['option_a']
    value: str


class _StructuralProbeOptionB(BaseModel):
    kind: Literal['option_b']
    value: int


class _StructuralProbeEnvelope(BaseModel):
    """Tagged union with discriminator='kind'. A capable model returns ONE branch."""
    payload: Annotated[Union[_StructuralProbeOptionA, _StructuralProbeOptionB], Field(discriminator='kind')]


# ----------------------------------------------------------------------
# Sanitizer protocol — kept here as a
# `runtime_checkable` Protocol so `core/llm_client.py` does NOT import
# `semantic_search.safety.layer_zero_sanitizer`. The concrete `LayerZeroSanitizer`
# satisfies this Protocol structurally; tests can supply lightweight
# fakes without instantiating the full search stack.
# ----------------------------------------------------------------------
@runtime_checkable
class _SanitizerVerdictLike(Protocol):
    """Anything with ``passed`` (bool) + ``reasons`` (List[str])."""
    passed: bool
    reasons: List[str]


@runtime_checkable
class IngressSanitizer(Protocol):
    """Structural surface required for an LLM-ingress sanitizer.

    A concrete implementation must:
    - expose a boolean ``applies_to_llm_ingress`` field gating the gate, AND
    - expose a ``sanitize(text: str) -> _SanitizerVerdictLike`` method that
      returns an object with ``passed`` (bool) and ``reasons`` (list of strings).

    The verdict's ``masked_text`` is intentionally NOT consumed at the LLM
    ingress — masking would alter the payload, and the policy at the LLM
    boundary is fail-closed: any matched check raises immediately so the
    operator (or upstream input-cleaner) fixes the producer rather than
    seeing a silently-rewritten prompt.
    """

    applies_to_llm_ingress: bool
    system_max_chars: int

    def sanitize(self, text: str, *, max_chars: Optional[int] = None) -> _SanitizerVerdictLike:
        ...


class LLMCallRouter:
    """Routes LLM calls through `LLMProvider` and validates structured responses.
    :param provider: LLMProvider - Initialized LLMProvider instance
    :param max_tokens: int - Hard cap on completion tokens per call
    :param temperature: float - Generation temperature
    :param token_warn_threshold: int - WARN when total tokens exceed this on a single call
    :param token_pricing: Dict[str, Dict[str, float]] - input/output USD per 1M tokens by model name
    :param default_pricing: Dict[str, float] - Pricing applied when model is unpriced
    """

    def __init__(self, provider: LLMProvider, max_tokens: int, temperature: float, token_warn_threshold: int, token_pricing: Dict[str, Dict[str, float]], default_pricing: Dict[str, float], circuit_breaker: CircuitBreaker, health_registry: BackendHealthRegistry, structural_gate: Optional[ModelStructuralCapabilityRegistry] = None, sanitizer: Optional[IngressSanitizer] = None):
        if circuit_breaker is None:
            raise LLMError("LLMCallRouter requires a CircuitBreaker instance")
        if health_registry is None:
            raise LLMError("LLMCallRouter requires a BackendHealthRegistry instance")
        self._provider = provider
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._token_warn_threshold = token_warn_threshold
        self._token_pricing = token_pricing
        self._default_pricing = default_pricing
        self._circuit_breaker = circuit_breaker
        self._health = health_registry
        # Optional. None = gate disabled (legacy
        # behaviour: every model in the chain is tried regardless of schema
        # support). When wired, discriminator-bearing schemas filter the chain
        # before any production call fires.
        self._structural_gate = structural_gate
        # Optional. None = ingress gate disabled
        # (legacy behaviour). When wired AND the concrete sanitizer's
        # `applies_to_llm_ingress` is True, every system + user prompt is
        # gated through `sanitize()` before any provider call. A block raises
        # `LLMError("llm_input_blocked_by_sanitizer …")` so the caller (or the
        # operator) is forced to fix the producer rather than serve a
        # degraded / poisoned prompt.
        self._sanitizer = sanitizer
        # Prompt-cache hit-rate counters. We
        # aggregate cached_input_tokens vs total prompt_tokens across every
        # successful `call_structured` (Tier-3 + NL-SQL gen + verifier) so the proxy signal is computed from real
        # provider metadata.
        self._cache_lock = threading.Lock()
        self._cached_input_tokens_total = 0
        self._prompt_input_tokens_total = 0
        self._calls_with_usage_total = 0
        # Per-model counters consumed by the model_provider drift detector.
        # Each entry: {model: {'cached_input_tokens', 'prompt_input_tokens',
        # 'calls', 'total_cost_usd', 'total_latency_ms'}}. Same lock as the
        # global counters so the two views stay consistent under concurrent
        # `call_structured` returns.
        self._per_model_stats: Dict[str, Dict[str, float]] = {}

    async def warmup_structural_gate_for_schema(self, models: List[str], schema_cls: Type[T]) -> None:
        """Pre-probe all models for structural capability so the gate cache is warm.

        Call at startup (before the service accepts traffic). Probes run in
        parallel via ``filter_capable_chain``; results are TTL-cached so the
        first real classify request sees cache hits and pays zero probe latency.

        No-op when the structural gate is not wired.
        """
        if self._structural_gate is None or not models:
            return
        await self._structural_gate.filter_capable_chain(models, schema_cls, self._probe_model_for_oneof)
        logger.info(f"structural_gate_warmup_complete models={len(models)} schema={schema_cls.__name__}")

    def _resolve_pricing(self, model: str) -> Dict[str, float]:
        """Look up per-1M-token pricing for a model with default fallback."""
        if model in self._token_pricing:
            return self._token_pricing[model]
        return self._default_pricing

    def _compute_cost_usd(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Compute USD cost for a call given pricing in USD per 1M tokens."""
        pricing = self._resolve_pricing(model)
        in_rate = float(pricing['input'])
        out_rate = float(pricing['output'])
        return (prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000.0

    def _emit_metrics(self, task_type: str, prompt_tag: str, model: str, usage: Dict[str, Any], elapsed_ms: float) -> None:
        """Log per-call observability — model, tokens, cost, latency, prompt tag."""
        prompt_tokens = int(usage.get('prompt_tokens', 0)) if usage else 0
        completion_tokens = int(usage.get('completion_tokens', 0)) if usage else 0
        total_tokens = int(usage.get('total_tokens', prompt_tokens + completion_tokens)) if usage else 0
        cached_input_tokens = int(usage.get('cached_input_tokens', 0)) if usage else 0
        cost = self._compute_cost_usd(model, prompt_tokens, completion_tokens)
        # Emit cached_input_tokens on every successful call so the
        # measurement dashboard can plot the prompt-cache hit-rate from log
        # streams as well as from the in-memory counter (defence in depth
        # against counter-reset).
        logger.info(f"llm_call task={task_type} prompt={prompt_tag} model={model} input_tokens={prompt_tokens} cached_input_tokens={cached_input_tokens} output_tokens={completion_tokens} total_tokens={total_tokens} cost_usd={cost:.6f} "f"latency_ms={elapsed_ms:.1f}")
        if total_tokens > self._token_warn_threshold:
            logger.warning(f"llm_call_token_budget_exceeded task={task_type} prompt={prompt_tag} model={model} total_tokens={total_tokens} threshold={self._token_warn_threshold}")

    def _record_prompt_cache_observation(self, usage: Dict[str, Any], model: Optional[str] = None, elapsed_ms: float = 0.0) -> None:
        """Aggregate per-call counters for the hit-rate + per-model drift signals.

        Called only on successful `call_structured` returns (so failed /
        retried attempts do not skew the denominator). Counters are
        thread-safe; concurrent calls may interleave but the totals are
        eventually consistent — strictly monotonic, never decrement.

        :param usage: Dict[str, Any] - Provider usage payload
            ({prompt_tokens, completion_tokens, cached_input_tokens})
        :param model: Optional[str] - Model name for the per-model stats
            slice (consumed by ModelProviderDriftDetector). When None or
            empty the global counters still update; the per-model slice is
            skipped.
        :param elapsed_ms: float - Wall-clock latency for this call. Added
            to the per-model running latency total. >= 0.
        """
        if not usage:
            return
        prompt_tokens = int(usage.get('prompt_tokens', 0) or 0)
        cached = int(usage.get('cached_input_tokens', 0) or 0)
        if prompt_tokens <= 0:
            return
        # Cap cached at prompt_tokens (defence against provider reporting bugs).
        if cached > prompt_tokens:
            cached = prompt_tokens
        if cached < 0:
            cached = 0
        completion_tokens = int(usage.get('completion_tokens', 0) or 0)
        cost_usd = self._compute_cost_usd(model, prompt_tokens, completion_tokens) if model else 0.0
        latency = float(elapsed_ms) if elapsed_ms >= 0.0 else 0.0
        with self._cache_lock:
            self._prompt_input_tokens_total += prompt_tokens
            self._cached_input_tokens_total += cached
            self._calls_with_usage_total += 1
            if model:
                slot = self._per_model_stats.get(model)
                if slot is None:
                    slot = {'cached_input_tokens': 0.0, 'prompt_input_tokens': 0.0, 'calls': 0.0, 'total_cost_usd': 0.0, 'total_latency_ms': 0.0}
                    self._per_model_stats[model] = slot
                slot['cached_input_tokens'] += float(cached)
                slot['prompt_input_tokens'] += float(prompt_tokens)
                slot['calls'] += 1.0
                slot['total_cost_usd'] += float(cost_usd)
                slot['total_latency_ms'] += latency

    def per_model_stats(self) -> Dict[str, Dict[str, float]]:
        """Return aggregated per-model counters for ``ModelProviderDriftDetector``.

        :return: Dict[model, Dict[counter, value]] — Snapshot copy. Each
            entry holds five running totals: ``cached_input_tokens``,
            ``prompt_input_tokens``, ``calls``, ``total_cost_usd``,
            ``total_latency_ms``. The detector derives cache-hit rate, mean
            cost-per-call, and mean latency from these.
        """
        with self._cache_lock:
            return {model: dict(slot) for model, slot in self._per_model_stats.items()}

    def prompt_cache_stats(self) -> Dict[str, int]:
        """Return aggregated prompt-cache counters for the proxy signal.

        Denominator (`prompt_input_tokens`) is
        the total INPUT tokens reported by the provider across every
        successful `call_structured`; numerator (`cached_input_tokens`) is
        the subset that hit the provider prompt cache. `calls` tracks
        the number of successful calls (denominator-side observations) so
        the evaluator can apply a min-sample-size guard.

        :return: Dict[str, int] - {'cached_input_tokens', 'prompt_input_tokens', 'calls'}
        """
        with self._cache_lock:
            return {
                'cached_input_tokens': int(self._cached_input_tokens_total),
                'prompt_input_tokens': int(self._prompt_input_tokens_total),
                'calls': int(self._calls_with_usage_total),
            }

    @staticmethod
    def _extract_json_object(text: str) -> str:
        """Extract the first balanced JSON object from a text blob (model may wrap in fences/prose).
        :param text: str - Raw LLM output
        :return: str - Substring containing the JSON object
        :raises LLMError: If no JSON object can be located
        """
        if not text:
            raise LLMError("llm_response_empty")
        start = text.find('{')
        if start == -1:
            raise LLMError(f"llm_response_no_json text_preview={text[:120]!r}")
        depth = 0
        in_string = False
        escape_next = False
        for i, ch in enumerate(text[start:], start):
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        raise LLMError(f"llm_response_no_json text_preview={text[:120]!r}")

    async def _probe_model_for_oneof(self, model: str) -> Tuple[bool, Optional[str]]:
        """Send the structural-gate probe to ``model`` and report the verdict.

        A model is judged ``oneOf``-capable iff its response can be parsed as
        a JSON object that satisfies the ``_StructuralProbeEnvelope`` tagged
        union. This is the minimum contract Pydantic emits for
        ``Annotated[Union[...], Field(discriminator='kind')]`` — if the model
        cannot satisfy that, it cannot satisfy the production schema either.

        :param model: str - Model identifier from the LLMProvider chain
        :return: Tuple[bool, Optional[str]] - (capable, error_or_none)
        """
        if self._structural_gate is None:
            return False, "no_gate_configured"
        sys_prompt, user_prompt = self._structural_gate.probe_prompts()
        max_toks = self._structural_gate.probe_max_tokens()
        try:
            client, _provider = self._provider.get_client_for_model(model)
        except Exception as e:
            return False, f"client_unavailable:{type(e).__name__}:{str(e)[:120]}"
        try:
            text, _usage = await client.call(sys_prompt, user_prompt, max_toks, model, self._temperature)
        except asyncio.CancelledError:
            raise
        except _RETRYABLE_TRANSPORT_ERRORS as e:
            return False, f"call_failed:{type(e).__name__}:{str(e)[:120]}"
        if not text:
            return False, "empty_response"
        try:
            json_blob = self._extract_json_object(text)
            _StructuralProbeEnvelope.model_validate_json(json_blob)
        except (LLMError, PydanticValidationError, json.JSONDecodeError, ValueError) as e:
            # responsible-ai.mdc: probe error stored in the
            # capability cache must NOT echo the model's response body. Hostile
            # output (PII / prompt injection) would otherwise leak into the
            # cache and any diagnostic surface that reads it. Record only the
            # error CATEGORY (exception class + a short, body-free reason).
            return False, f"parse_failed:{type(e).__name__}"
        return True, None

    # Tasks where the user-role prompt is system-generated (schema dumps, structured
    # hints, retry context) — not raw user input. Apply system_max_chars so the
    # 500-char user-query cap does not block legitimate internal prompts.
    _SYSTEM_GENERATED_USER_PROMPT_TASKS: frozenset = frozenset({
        "nl_to_sql_generation",
        "nl_to_sql_verifier",
        # qie_only L0 filter extract: user role carries catalog + hints + query.
        "l0_entity_extraction",
    })

    def _enforce_ingress_sanitizer(self, task_type: str, prompt_tag: str, system_prompt: str, user_prompt: str) -> None:
        """Gate the LLM ingress through the optional sanitizer.

        System prompts use ``system_max_chars`` (high cap — developer-authored content).
        User prompts normally use ``max_chars`` (low cap — user-supplied query text),
        but tasks in ``_SYSTEM_GENERATED_USER_PROMPT_TASKS`` also use ``system_max_chars``
        because their user-role content is system-assembled (schema + hints), not raw input.
        Both sides are checked for blocklist patterns and PII regardless of length.
        Policy is fail-closed: any matched check raises immediately.

        :raises LLMError: When the sanitizer rejects either prompt half
        """
        if self._sanitizer is None or not self._sanitizer.applies_to_llm_ingress:
            return
        sys_verdict = self._sanitizer.sanitize(system_prompt, max_chars=self._sanitizer.system_max_chars)
        if not sys_verdict.passed:
            logger.warning(f"llm_input_blocked_by_sanitizer task={task_type} prompt={prompt_tag} side=system reasons={sys_verdict.reasons}")
            raise LLMError(f"llm_input_blocked_by_sanitizer task={task_type} prompt={prompt_tag} side=system reasons={sys_verdict.reasons}")
        user_max = self._sanitizer.system_max_chars if task_type in self._SYSTEM_GENERATED_USER_PROMPT_TASKS else None  # uses config max_chars (500) for user-supplied queries
        user_verdict = self._sanitizer.sanitize(user_prompt, max_chars=user_max)
        if not user_verdict.passed:
            logger.warning(f"llm_input_blocked_by_sanitizer task={task_type} prompt={prompt_tag} side=user reasons={user_verdict.reasons}")
            raise LLMError(f"llm_input_blocked_by_sanitizer task={task_type} prompt={prompt_tag} side=user reasons={user_verdict.reasons}")

    async def call_structured(self, task_type: str, prompt_tag: str, system_prompt: str, user_prompt: str, response_schema: Type[T], model_override: Optional[str] = None, cost_observer: Optional[CostObserver] = None) -> Tuple[T, Dict[str, Any]]:
        """Run an LLM call and validate the JSON response against a Pydantic schema.
        :param task_type: str - Task type used to pick a model from the fallback chain
        :param prompt_tag: str - Short stable identifier used in logs (prompt version/name)
        :param system_prompt: str - System role content
        :param user_prompt: str - User role content
        :param response_schema: Type[T] - Pydantic model the response must validate against
        :param model_override: Optional[str] - Bypass fallback chain (testing only)
        :param cost_observer: Optional[CostObserver] - Per-request callback receiving (model, cost_usd, usage)
            after a successful schema-validated call. Used by ``SearchOrchestrator`` to push the cost into a
            ``QueryCostBudget`` that may raise ``QueryCostBudgetExceeded`` synchronously.
            The router lets that exception propagate so the orchestrator can fail-fast on the same
            event-loop tick the budget was breached.
        :return: Tuple[T, Dict[str, Any]] - Parsed structured response and call metadata
        :raises LLMError: On call failure, JSON parse failure, schema validation failure,
                          or sanitizer-ingress block. The
                          sanitizer block fires BEFORE any provider call so a poisoned
                          prompt never reaches an LLM provider.
        :raises QueryCostBudgetExceeded: When the optional ``cost_observer`` raises it
                          on the just-recorded cost. Surfaces the breach to the caller
                          on the same tick — the LLM result is discarded.
        """
        # Layer-0 sanitizer at the LLM ingress.
        # Runs FIRST so a poisoned prompt cannot consume circuit-breaker
        # budget, structural-gate probes, or any provider tokens.
        self._enforce_ingress_sanitizer(task_type, prompt_tag, system_prompt, user_prompt)
        # Cost admit (query + fleet): deny BEFORE provider I/O so spend
        # pressure degrades to regex / L1 paths (LLMError) instead of
        # rejecting the user query at the orchestrator.
        from semantic_search.core.exceptions import CostBudgetExceeded
        effective_observer_pre = cost_observer if cost_observer is not None else _REQUEST_COST_OBSERVER.get()
        if effective_observer_pre is not None:
            check_admit = getattr(effective_observer_pre, 'check_admit', None)
            if callable(check_admit):
                try:
                    check_admit()
                except CostBudgetExceeded as e:
                    logger.warning(
                        f"llm_cost_budget_admit_denied task={task_type} prompt={prompt_tag} "
                        f"error_type={type(e).__name__} error={str(e)}"
                    )
                    raise LLMError(
                        f"llm_cost_budget_exhausted task={task_type} prompt={prompt_tag} "
                        f"reason={type(e).__name__}"
                    ) from e
        chain = self._provider.get_fallback_chain(task_type)
        if not chain and model_override is None:
            raise LLMError(f"llm_no_models_available task={task_type}")
        if not self._circuit_breaker.allow_request():
            logger.warning(f"llm_call_short_circuited task={task_type} prompt={prompt_tag} reason=circuit_open")
            raise CircuitOpenError(f"llm_circuit_open task={task_type} prompt={prompt_tag}")
        models_to_try: List[str] = [model_override] if model_override is not None else list(chain)
        # Structural capability gate. Bypassed
        # when (a) the gate isn't wired, (b) the caller forced a specific
        # model via override (test path), or (c) the schema does not require
        # ``oneOf`` discrimination (most schemas).
        if self._structural_gate is not None and model_override is None and schema_requires_oneof(response_schema.model_json_schema()):
            pre_count = len(models_to_try)
            models_to_try = await self._structural_gate.filter_capable_chain(models_to_try, response_schema, self._probe_model_for_oneof)
            logger.info(f"structural_gate_filter task={task_type} prompt={prompt_tag} pre_count={pre_count} post_count={len(models_to_try)} survivors={models_to_try}")
            if not models_to_try:
                # Fail fast. Operator must add a capable
                # model to the chain or remove the discriminator from the
                # schema; silent degradation is forbidden.
                raise LLMError(f"llm_no_oneof_capable_models task={task_type} prompt={prompt_tag} pre_count={pre_count}")
        last_error: Optional[Exception] = None
        for model in models_to_try:
            client, _provider_name = self._provider.get_client_for_model(model)
            t0 = time.monotonic()
            try:
                text, usage = await client.call(system_prompt, user_prompt, self._max_tokens, model, self._temperature)
            except asyncio.CancelledError:
                logger.warning(f"llm_call_cancelled task={task_type} prompt={prompt_tag} model={model}")
                raise
            except _RETRYABLE_TRANSPORT_ERRORS as e:
                last_error = e
                logger.warning(f"llm_call_failed task={task_type} prompt={prompt_tag} model={model} error_type={type(e).__name__} error={str(e)}")
                continue
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            self._emit_metrics(task_type, prompt_tag, model, usage or {}, elapsed_ms)
            if not text:
                last_error = LLMError(f"llm_response_empty task={task_type} model={model}")
                continue
            try:
                json_blob = self._extract_json_object(text)
                parsed: T = response_schema.model_validate_json(json_blob)
            except (LLMError, PydanticValidationError, json.JSONDecodeError, ValueError) as e:
                last_error = e
                logger.warning(f"llm_response_parse_failed task={task_type} prompt={prompt_tag} model={model} error_type={type(e).__name__} error={str(e)}")
                continue
            metadata: Dict[str, Any] = {'model': model, 'usage': usage or {}, 'latency_ms': elapsed_ms}
            self._circuit_breaker.record_success()
            self._health.record('llm', success=True)
            # Counter advances ONLY on full
            # success (after schema validation), so failed parses / retries do
            # not skew the prompt-cache hit-rate denominator. Model + elapsed_ms
            # are passed so the per-model counters consumed by
            # ModelProviderDriftDetector stay in lockstep with the global
            # prompt-cache counters.
            self._record_prompt_cache_observation(usage or {}, model=model, elapsed_ms=elapsed_ms)
            # Per-call cost callback. Computed from the SAME
            # usage dict the metrics emitter saw, so observability + budget
            # accounting are guaranteed consistent. Invoked AFTER schema
            # validation + breaker/health recording so a budget breach
            # discards a fully-validated result rather than a partial one.
            # If the observer raises (e.g. QueryCostBudgetExceeded), we let
            # it propagate — the result is intentionally NOT returned.
            # Observer resolution order:
            #   1. explicit `cost_observer=` parameter (per-call override),
            #   2. request-scoped ContextVar bound by SearchOrchestrator,
            #   3. None — no cost tracking (legacy path, zero overhead).
            effective_observer = cost_observer if cost_observer is not None else _REQUEST_COST_OBSERVER.get()
            if effective_observer is not None:
                u = usage or {}
                prompt_tokens = int(u.get('prompt_tokens', 0) or 0)
                completion_tokens = int(u.get('completion_tokens', 0) or 0)
                call_cost_usd = self._compute_cost_usd(model, prompt_tokens, completion_tokens)
                try:
                    effective_observer(model, call_cost_usd, u)
                except CostBudgetExceeded as e:
                    # Cost already counted; discard LLM result so callers
                    # (QI L0/L2) treat as LLM failure → regex / L1 degrade.
                    logger.warning(
                        f"llm_cost_budget_post_call task={task_type} prompt={prompt_tag} "
                        f"model={model} cost_usd={call_cost_usd:.6f} "
                        f"error_type={type(e).__name__} error={str(e)}"
                    )
                    raise LLMError(
                        f"llm_cost_budget_exceeded task={task_type} prompt={prompt_tag} "
                        f"model={model} reason={type(e).__name__}"
                    ) from e
            return parsed, metadata
        self._circuit_breaker.record_failure()
        self._health.record('llm', success=False)
        raise LLMError(f"llm_call_exhausted task={task_type} prompt={prompt_tag} attempts={len(models_to_try)} last_error_type={type(last_error).__name__ if last_error else 'none'}")
