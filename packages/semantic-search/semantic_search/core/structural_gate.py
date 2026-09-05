"""LLM oneOf-discriminator structural capability gate.

Background
----------
Several QI prompts emit a tagged-union response (e.g.
``Annotated[Union[ClassifyResult, DecomposeResult], Field(discriminator='kind')]``).
Pydantic renders such schemas with ``oneOf`` + ``discriminator``. Not every
provider/model honours strict ``oneOf`` discrimination — some emit hybrid
shapes that violate the discriminator, others omit the discriminator entirely
and the response then fails Pydantic validation downstream.

The contract binds us to *gate* the fallback chain on this capability so calls
never burn budget on a model that cannot satisfy the contract. This module
owns the gate.

Design (per the user's selected options)
----------------------------------------
- **Live probe + cache**: the first time a ``(model, schema_signature)`` pair
  is requested, we send a tiny one-token discrimination probe to the model and
  record pass/fail. Subsequent calls hit the cache for ``ttl`` seconds.
- **Raise loud on empty chain**: when the schema requires ``oneOf`` and pruning
  empties the model list, the gate raises ``LLMError('llm_no_oneof_capable_models …')``.
  Failing fast forces operator action rather than silently degrading to a
  non-discriminator schema.
- **No-op on non-discriminator schemas**: schemas without ``oneOf``/``anyOf+discriminator``
  bypass the gate entirely so we never probe (or block) on schemas that don't
  need the capability.

Threading
---------
The cache is protected by a single ``threading.Lock``. Probes themselves are
async (call into ``LLMClient.call``) and run one-at-a-time per
``(model, schema_signature)`` via per-key ``asyncio.Lock`` so concurrent
``call_structured`` invocations on the same hot path coalesce onto a single
probe instead of stampeding the provider.
"""
import asyncio
import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Type

from pydantic import BaseModel

from semantic_search.core.exceptions import LLMError, ValidationError
from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)


# Discriminator-bearing JSON-Schema constructs we must detect. Pydantic v2
# emits ``oneOf`` for ``Field(discriminator=...)`` on ``Union`` types and
# ``anyOf`` for unions without a discriminator. We treat both as "structural
# union" but the gate only fires when a discriminator is also present (since
# ``anyOf`` without a discriminator is satisfiable by any branch and does not
# need the structural-gate check).
_UNION_KEYS = ('oneOf', 'anyOf')


def schema_requires_oneof(schema: Dict[str, Any]) -> bool:
    """Detect discriminated union (oneOf/anyOf + discriminator key) recursively."""
    if not isinstance(schema, dict):
        return False
    # Direct check at this level.
    has_union = any(k in schema and isinstance(schema[k], list) for k in _UNION_KEYS)
    has_disc = 'discriminator' in schema
    if has_union and has_disc:
        return True
    # Recurse — order doesn't matter, we short-circuit on first hit.
    for v in schema.values():
        if isinstance(v, dict):
            if schema_requires_oneof(v):
                return True
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict) and schema_requires_oneof(item):
                    return True
    return False


def schema_signature(schema_cls: Type[BaseModel]) -> str:
    """SHA-256 (first 16 chars) of canonicalised JSON Schema (stable cache key)."""
    if not isinstance(schema_cls, type) or not issubclass(schema_cls, BaseModel):
        raise ValidationError("schema_signature requires a Pydantic BaseModel subclass")
    schema = schema_cls.model_json_schema()
    canon = json.dumps(schema, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canon.encode('utf-8')).hexdigest()[:16]


@dataclass
class CapabilityRecord:
    """Cached probe: (model, schema_sig) → capable + latency_ms + error."""
    model: str
    schema_sig: str
    capable: bool
    probed_at: float
    probe_latency_ms: float
    probe_error: Optional[str] = None
    schema_requires_oneof: bool = True

    def is_fresh(self, ttl_seconds: float, now: Optional[float] = None) -> bool:
        """Return True iff this record is still within the TTL window.

        TTL=0 means "never cache" — every lookup re-probes. Negative TTL is
        rejected at config-load (``LLMStructuralGateConfig.__post_init__``).
        """
        if ttl_seconds <= 0.0:
            return False
        n = now if now is not None else time.time()
        return (n - self.probed_at) <= ttl_seconds


class ModelStructuralCapabilityRegistry:
    """Live-probe + TTL cache of ``(model, schema_signature)`` capability.

    Public API:
    - ``filter_capable_chain(models, schema_cls, probe_fn)`` — async; returns
      the subset of ``models`` known/probed to be oneOf-capable for
      ``schema_cls``. Models with no record are probed once (per-key locked).
    - ``get(model, schema_sig)`` — sync; non-blocking cache read.
    - ``record(record)`` — sync; manual seeding (used by tests).
    - ``invalidate(model=None, schema_sig=None)`` — sync; targeted purge or full
      flush. Used by the snapshot-version registry on inventory bumps so a
      schema change does not get served stale capability verdicts.

    The probe payload is intentionally minimal so the warm-up cost is bounded:
    one short prompt per ``(model, schema)`` pair, charged against the same
    LLM budget surface as production calls.
    """

    def __init__(self, ttl_seconds: float, probe_timeout_seconds: float, probe_max_tokens: int, probe_system_prompt: str, probe_user_prompt: str, treat_unknown_as_capable: bool):
        if ttl_seconds < 0.0:
            raise ValidationError(f"ttl_seconds must be >= 0, got {ttl_seconds}")
        if probe_timeout_seconds <= 0.0:
            raise ValidationError(f"probe_timeout_seconds must be > 0, got {probe_timeout_seconds}")
        if probe_max_tokens <= 0:
            raise ValidationError(f"probe_max_tokens must be > 0, got {probe_max_tokens}")
        if not isinstance(probe_system_prompt, str) or not probe_system_prompt.strip():
            raise ValidationError("probe_system_prompt must be a non-empty string")
        if not isinstance(probe_user_prompt, str) or not probe_user_prompt.strip():
            raise ValidationError("probe_user_prompt must be a non-empty string")
        self._ttl = float(ttl_seconds)
        self._timeout = float(probe_timeout_seconds)
        self._probe_max_tokens = int(probe_max_tokens)
        self._probe_system_prompt = probe_system_prompt
        self._probe_user_prompt = probe_user_prompt
        self._treat_unknown_as_capable = bool(treat_unknown_as_capable)
        self._records: Dict[Tuple[str, str], CapabilityRecord] = {}
        self._lock = threading.Lock()
        # One asyncio.Lock per (model, schema_sig) so concurrent callers
        # asking for the same probe coalesce onto a single in-flight call.
        self._probe_locks: Dict[Tuple[str, str], asyncio.Lock] = {}
        self._probe_locks_guard = threading.Lock()

    # ------------------------------------------------------------------
    # Cache primitives
    # ------------------------------------------------------------------
    def get(self, model: str, schema_sig: str) -> Optional[CapabilityRecord]:
        """Return the fresh cached record (or None when absent / expired)."""
        with self._lock:
            rec = self._records.get((model, schema_sig))
        if rec is None:
            return None
        if not rec.is_fresh(self._ttl):
            return None
        return rec

    def record(self, record: CapabilityRecord) -> None:
        """Insert or overwrite the cached record."""
        if not isinstance(record, CapabilityRecord):
            raise ValidationError("record requires a CapabilityRecord instance")
        with self._lock:
            self._records[(record.model, record.schema_sig)] = record

    def invalidate(self, model: Optional[str] = None, schema_sig: Optional[str] = None) -> int:
        """Drop entries matching the filters. ``None`` matches anything.

        :return: int - Number of records evicted
        """
        with self._lock:
            if model is None and schema_sig is None:
                count = len(self._records)
                self._records.clear()
                return count
            to_drop = [k for k in self._records.keys() if (model is None or k[0] == model) and (schema_sig is None or k[1] == schema_sig)]
            for k in to_drop:
                del self._records[k]
            return len(to_drop)

    def snapshot(self) -> Dict[Tuple[str, str], CapabilityRecord]:
        """Return a shallow copy of the cache for diagnostics."""
        with self._lock:
            return dict(self._records)

    # ------------------------------------------------------------------
    # Probe coordination
    # ------------------------------------------------------------------
    def _get_probe_lock(self, key: Tuple[str, str]) -> asyncio.Lock:
        with self._probe_locks_guard:
            lock = self._probe_locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._probe_locks[key] = lock
            return lock

    async def ensure_probed(self, model: str, schema_sig: str, probe_fn) -> CapabilityRecord:
        """Return a (cached or freshly-probed) capability record for ``model``.

        ``probe_fn(model)`` is awaited only on cache miss / expiry. It must
        return ``Tuple[bool, Optional[str]]`` — ``(capable, error_or_none)``.

        Concurrent callers for the same key share the in-flight probe via a
        per-key asyncio.Lock; only one network call fires.

        Timeout failures are NOT persisted to the cache so a transient network
        delay at startup (cold-start) cannot blacklist a model for the full TTL.
        The record is still returned to the current caller with capable=False.
        All other failures (bad JSON, wrong discriminator shape) ARE cached so
        genuinely incapable models are not re-probed on every request.
        """
        cached = self.get(model, schema_sig)
        if cached is not None:
            return cached
        lock = self._get_probe_lock((model, schema_sig))
        async with lock:
            # Re-check inside the lock — another coroutine may have just
            # populated the cache while we were waiting.
            cached = self.get(model, schema_sig)
            if cached is not None:
                return cached
            t0 = time.monotonic()
            timed_out = False
            try:
                capable, err = await asyncio.wait_for(probe_fn(model), timeout=self._timeout)
            except asyncio.TimeoutError:
                capable, err = False, f"probe_timeout_after_{self._timeout}s"
                timed_out = True
            except asyncio.CancelledError:
                # Never swallow cancellation — re-raise per async-patterns.mdc.
                raise
            except Exception as e:  # broad on purpose: any probe failure = not capable
                capable, err = False, f"{type(e).__name__}:{str(e)[:160]}"
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            rec = CapabilityRecord(model=model, schema_sig=schema_sig, capable=bool(capable), probed_at=time.time(), probe_latency_ms=elapsed_ms, probe_error=err if not capable else None)
            if not timed_out:
                self.record(rec)
            logger.info(f"structural_probe model={model} schema_sig={schema_sig} capable={rec.capable} latency_ms={elapsed_ms:.1f} error={rec.probe_error or 'none'} cached={not timed_out}")
            return rec

    # ------------------------------------------------------------------
    # Public chain filter
    # ------------------------------------------------------------------
    async def filter_capable_chain(self, models, schema_cls: Type[BaseModel], probe_fn) -> list:
        """Return the subset of ``models`` capable of decoding ``schema_cls``.

        - When ``schema_cls`` doesn't require ``oneOf``, returns ``models``
          unchanged (the gate is a no-op).
        - When the schema does require ``oneOf``, every model is checked
          against the cache; misses trigger a single probe each.
        - Order is preserved (the LLM router relies on chain order = priority).

        :param models: Iterable[str] - Candidate models in priority order
        :param schema_cls: Type[BaseModel] - The Pydantic schema the call needs
        :param probe_fn: Callable[[str], Awaitable[Tuple[bool, Optional[str]]]]
        :return: List[str] - Models known capable (cache or fresh probe)
        """
        models_list = list(models)
        if not models_list:
            return []
        if not schema_requires_oneof(schema_cls.model_json_schema()):
            return models_list
        sig = schema_signature(schema_cls)
        # Probe every miss in parallel — keeps cold-start latency bounded by
        # the slowest single probe rather than N × probe latency.
        async def _decide(m: str) -> Tuple[str, bool]:
            cached = self.get(m, sig)
            if cached is not None:
                return m, cached.capable
            try:
                rec = await self.ensure_probed(m, sig, probe_fn)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # any unexpected error = not capable but logged
                logger.warning(f"structural_probe_unexpected_error model={m} schema_sig={sig} error_type={type(e).__name__} error={str(e)}")
                return m, self._treat_unknown_as_capable
            return m, rec.capable

        results = await asyncio.gather(*(_decide(m) for m in models_list))
        capable_set = {m for m, ok in results if ok}
        # Preserve chain order.
        return [m for m in models_list if m in capable_set]

    # ------------------------------------------------------------------
    # Probe payload helpers (consumed by LLMCallRouter)
    # ------------------------------------------------------------------
    def probe_prompts(self) -> Tuple[str, str]:
        """Return the configured ``(system, user)`` probe prompts."""
        return self._probe_system_prompt, self._probe_user_prompt

    def probe_max_tokens(self) -> int:
        return self._probe_max_tokens
