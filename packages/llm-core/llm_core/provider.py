"""Unified LLMProvider — discovers models from API keys, ranks, caches clients, fails over.

Design contract (post-refactor 2026-04):
  * Service config carries ZERO model lists and ZERO pricing. The only
    per-provider data in YAML is the env var name holding the API key.
  * The model universe is built dynamically. `validate_api_keys` calls
    `client.models.list()` against each validated key — whatever the vendor or
    corporate gateway returns is the available model set. No declared list
    means no risk of hand-maintained YAML drifting from gateway entitlements.
  * Pricing comes from `llm_core/pricing.yaml` (single source of truth, used
    by both the registry and `cost_analysis`). Models with no pricing entry
    still rank with neutral cost tier 3 — they remain selectable. Model ids
    in `selection_exclusions` plus any priced SKU whose output USD/M is strictly
    above `selection_max_output_usd_per_million` are omitted from discovery ranking
    and from `rank_candidates` (see `pricing.load_selection_exclusions`).
  * Capability tiers are inferred from model name; latency defaults to neutral.
    Both can be overridden per service via `model_capability_overrides` /
    `model_latency_overrides` for the rare case where the heuristic is wrong.

Lifecycle (typical FastAPI lifespan):
    provider = LLMProvider(config_dict, feedback_store=app_state.feedback_store)
    await provider.validate_api_keys(live_check=True)   # discovery happens here
    provider.build_model_registry()
    await provider.probe_startup_inference()            # timed generation into runtime_stats
    provider.build_task_fallbacks()                     # cost + measured latency + latest
    provider.initialize_cached_clients()
    await provider.start_background_refresh()
"""
import asyncio
import os
import re
import threading  # noqa: TID251
import time
from typing import Any, Dict, List, Optional, Tuple

from llm_core.config_schema import validate_provider_config
from llm_core.exceptions import ConfigurationError
from llm_core.llm_client import AnthropicLLMClient, OpenAILLMClient
from llm_core.logging_utils import get_logger
from llm_core.model_registry import ModelRegistry, _detect_provider, _is_chat_model
from llm_core.pricing import infer_model_family, infer_model_recency, load_selection_exclusions
from llm_core.ranker import Ranker, RankWeights
from llm_core.signal_adapter import RuntimeStats, SignalAdapter

try:
    from anthropic import Anthropic as _Anthropic
except ImportError:
    _Anthropic = None  # type: ignore[assignment,misc]

try:
    from openai import OpenAI as _OpenAI
except ImportError:
    _OpenAI = None  # type: ignore[assignment,misc]

# Transient probe failures — the request never reached a routing verdict (gateway
# slow / connection flapped), so the model is NOT provably inaccessible. Treated
# as reachable by `_live_check_model` so a slow cold probe doesn't permanently
# drop an entitled model at boot. Real calls use `client_timeout` (>> the probe
# budget), so a probe timeout does not predict downstream failure.
_TRANSIENT_PROBE_ERRORS: tuple = ()
for _mod_name in ('anthropic', 'openai'):
    try:
        _sdk = __import__(_mod_name)
        _TRANSIENT_PROBE_ERRORS += tuple(
            e for e in (getattr(_sdk, 'APITimeoutError', None), getattr(_sdk, 'APIConnectionError', None)) if e is not None
        )
    except ImportError:
        continue

_NEUTRAL_COST_TIER = 3
_REFRESH_STOP_TIMEOUT_S = 5.0
_LOG_ERROR_MAX_LEN = 200
# GoCaas / OpenAI-compat gateways: google (Gemini) uses the OpenAI SDK + LLM_BASE_URL.
_OPENAI_COMPAT_PROVIDERS = frozenset({'openai', 'google'})

logger = get_logger(__name__)

# Matches `'message': '...'` and `"message": "..."` (handles single + double quotes,
# escaped quotes within the value). Gateways like litellm wrap upstream errors in
# nested dict reprs, so a single regex pass over the stringified exception gives us
# the operator-readable message buried 2-3 layers deep.
_ERROR_MESSAGE_RE = re.compile(r"""['"]message['"]\s*:\s*['"]((?:\\.|[^'"\\])*)['"]""")


def _extract_error_message(raw: str) -> str:
    """Pull the deepest `message` field out of a stringified gateway exception.

    Returns the raw string unchanged if no `message` key is found, so callers always
    get *something* useful even when the SDK exception isn't a wrapped JSON body.
    """
    if not raw:
        return ''
    matches = _ERROR_MESSAGE_RE.findall(raw)
    if not matches:
        return raw
    # Deepest message is typically the most specific (upstream provider's reason);
    # outer layers just say "litellm.BadRequestError: OpenAIException - ...".
    return matches[-1].strip()


# Substrings that indicate the probe reached the model and was rejected for a
# token/output-length policy (not for routing/auth/entitlement). These are
# operationally "model is reachable" — see `_live_check_model` for context.
_OUTPUT_LENGTH_FAILURE_MARKERS = (
    'max_tokens or model output limit was reached',
    'max_output_tokens',
    'max output tokens',
    'maximum context length',
    'finish the message because max_tokens',
)


def _is_output_length_only_failure(msg: str) -> bool:
    """Decide whether an error message describes only a token-budget rejection.

    Used by `_live_check_model` to keep models that route correctly but reject
    our 1-token probe due to a per-model output-length floor or reasoning-token
    consumption. We do NOT widen this to all 400s — auth, model-not-found, and
    endpoint-mismatch errors must still drop the model.
    """
    if not msg:
        return False
    lo = msg.lower()
    return any(marker.lower() in lo for marker in _OUTPUT_LENGTH_FAILURE_MARKERS)


class LLMProvider:
    """Discovery-driven model provisioning with ranked selection and runtime adaptation."""

    def __init__(self, config: Dict[str, Any], feedback_store: Optional[Any] = None):
        """Initialize provider from config dict.

        :param config: Dict[str, Any] - Full service config (must include llm_* and model_selection_strategy)
        :param feedback_store: Optional[Any] - FeedbackStore instance for runtime adaptation; pass None to disable
        :raises ConfigurationError: If required config sections or keys are missing
        """
        validate_provider_config(config)
        self._config = config
        self._strategy = config['model_selection_strategy']
        self._weights_default = RankWeights.from_dict(self._strategy['weights'])

        runtime_cfg = self._strategy['runtime_adaptation']
        self._runtime_enabled = bool(runtime_cfg['enabled'])
        self._refresh_interval_s = int(runtime_cfg['refresh_interval_s'])
        self._runtime_min_samples = int(runtime_cfg['min_samples'])
        self._latency_window_s = int(runtime_cfg['latency_window_s'])
        self._quality_window_s = int(runtime_cfg['quality_window_s'])
        self._error_demotion_threshold = float(runtime_cfg['error_demotion_threshold'])
        self._latency_buckets_ms = list(runtime_cfg['latency_buckets_ms'])
        self._max_entries_per_type = int(runtime_cfg['max_entries_per_type'])

        self._cap_overrides = config['model_capability_overrides']
        self._lat_overrides = config['model_latency_overrides']

        self._available_models: List[str] = []
        self._validated_providers: Dict[str, str] = {}
        self._cached_clients: Dict[str, Any] = {}
        self._model_fallbacks: Dict[str, List[str]] = {}
        self._runtime_stats: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

        self._registry = ModelRegistry(
            models=[],
            capability_overrides=self._cap_overrides,
            latency_overrides=self._lat_overrides,
        )
        self._ranker = Ranker(
            latency_buckets_ms=self._latency_buckets_ms,
            error_demotion_threshold=self._error_demotion_threshold,
            runtime_min_samples=self._runtime_min_samples,
            provider_tiebreak_priority=list(self._strategy['provider_tiebreak_priority']),
        )
        self._signal_adapter = SignalAdapter(
            store=feedback_store,
            latency_window_s=self._latency_window_s,
            quality_window_s=self._quality_window_s,
            max_entries_per_type=self._max_entries_per_type,
        )

        self._refresh_task: Optional[asyncio.Task] = None
        self._refresh_stop: Optional[asyncio.Event] = None

    @property
    def available_models(self) -> List[str]:
        """Models discovered against validated API keys (empty until validate_api_keys runs)."""
        return list(self._available_models)

    @property
    def model_registry(self) -> Dict[str, Dict[str, Any]]:
        """Model registry with capability/cost/latency tiers."""
        return self._registry.registry

    @property
    def cached_clients(self) -> Dict[str, Any]:
        """Cached LLM client instances keyed by provider."""
        return dict(self._cached_clients)

    @property
    def model_fallbacks(self) -> Dict[str, List[str]]:
        """Model fallback chains keyed by task type or component name."""
        with self._lock:
            return {k: list(v) for k, v in self._model_fallbacks.items()}

    @property
    def runtime_stats(self) -> Dict[str, Dict[str, Any]]:
        """Most recent per-model runtime stats."""
        with self._lock:
            return {k: dict(v) for k, v in self._runtime_stats.items()}

    def build_model_registry(self) -> None:
        """Build model registry over the currently available (discovered) model set."""
        self._available_models = LLMProvider._without_selection_exclusions(self._available_models)
        self._registry.update_models(self._available_models)
        self._registry.build()

    def _resolve_weights(self, task_type: str) -> RankWeights:
        """Pick weights for a task: per-task override (if mapped) else default."""
        per_task = self._strategy['weights_by_task_type']
        if isinstance(per_task, dict) and task_type in per_task:
            return RankWeights.from_dict(per_task[task_type])
        return self._weights_default

    def select_models_for_task(self, task_type: str) -> List[str]:
        """Rank models for a task by composite score using configured weights.

        :param task_type: str - Task type identifier (e.g. 'evaluation', 'self_healing')
        :return: List[str] - Ranked models (first=primary, rest=fallbacks)
        """
        if not self._available_models:
            return []
        weights = self._resolve_weights(task_type)
        with self._lock:
            stats = dict(self._runtime_stats)
        ranked = self._ranker.rank(self._available_models, self._registry.registry, weights, stats)
        return [m for m, _ in ranked]

    def get_ranked_models_for_task(self, task_type: str) -> List[Tuple[str, float]]:
        """Return models ranked by score with composite scores."""
        if not self._available_models:
            return []
        weights = self._resolve_weights(task_type)
        with self._lock:
            stats = dict(self._runtime_stats)
        return self._ranker.rank(self._available_models, self._registry.registry, weights, stats)

    @staticmethod
    def _without_selection_exclusions(models: List[str]) -> List[str]:
        """Remove model ids listed in pricing.yaml `selection_exclusions`.
        :param models: list[str] - candidate model identifiers
        :return: list[str] - models not in the exclusion set
        """
        excl = load_selection_exclusions()
        if not excl:
            return list(models)
        return [m for m in models if m not in excl]

    def rank_candidates(self, candidates: List[str], task_type: str) -> List[str]:
        """Rank a caller-supplied candidate list (e.g. after a domain-specific reliability filter)."""
        candidates = LLMProvider._without_selection_exclusions(list(candidates))
        if not candidates:
            return []
        weights = self._resolve_weights(task_type)
        with self._lock:
            stats = dict(self._runtime_stats)
        ranked = self._ranker.rank(candidates, self._registry.registry, weights, stats)
        return [m for m, _ in ranked]

    def select_diverse_models_for_task(self, task_type: str, count: int, require_provider_diversity: bool) -> List[str]:
        """Pick top-N models for a task while honoring provider diversity for ensemble review."""
        if count <= 0:
            return []
        ranked = self.select_models_for_task(task_type)
        if not ranked:
            return []
        if not require_provider_diversity:
            return ranked[:count]
        selected: List[str] = []
        seen_providers: set = set()
        for model in ranked:
            if len(selected) >= count:
                break
            provider = _detect_provider(model)
            if provider in seen_providers:
                continue
            selected.append(model)
            seen_providers.add(provider)
        if len(selected) < count:
            for model in ranked:
                if model in selected:
                    continue
                selected.append(model)
                if len(selected) >= count:
                    break
        return selected

    def _build_capped_chain(self, ranked: List[str], max_cost_tier: int) -> List[str]:
        """Partition a ranked model list by cost cap.

        Models with cost_tier <= max_cost_tier come first (preferred pool).
        Models above the cap are appended afterwards as last-resort — the chain
        is never left empty so callers always have something to fall back to.
        """
        registry = self._registry.registry
        preferred: List[str] = []
        above_cap: List[str] = []
        for model in ranked:
            cost_tier = int((registry.get(model) or {}).get('cost_tier', _NEUTRAL_COST_TIER))
            if cost_tier <= max_cost_tier:
                preferred.append(model)
            else:
                above_cap.append(model)
        return preferred + above_cap

    def _model_meets_latency_slo(self, model: str, max_p50_ms: float) -> bool:
        """True when startup/runtime stats show measured p50 within the SLO."""
        with self._lock:
            stats = self._runtime_stats.get(model)
        if not stats:
            return False
        sample_count = int(stats.get('sample_count', 0))
        p50 = stats.get('p50_latency_ms')
        if sample_count < self._runtime_min_samples or p50 is None:
            return False
        return float(p50) <= float(max_p50_ms)

    def _build_latency_slo_chain(self, ranked: List[str], max_p50_ms: float) -> List[str]:
        """Partition ranked models by measured latency SLO.

        Preferred: models with enough samples and ``p50_latency_ms <= max_p50_ms``.
        Unprobed / over-SLO models append as last-resort (chain never emptied).
        When no model meets the SLO, the original ranked order is kept.
        """
        preferred: List[str] = []
        last_resort: List[str] = []
        for model in ranked:
            if self._model_meets_latency_slo(model, max_p50_ms):
                preferred.append(model)
            else:
                last_resort.append(model)
        if not preferred:
            logger.warning(
                f"task_latency_slo_no_preferred max_p50_ms={max_p50_ms} "
                f"candidates={len(ranked)} action=keep_ranked_order"
            )
            return ranked
        return preferred + last_resort

    @staticmethod
    def _apply_task_model_allowlist(ranked: List[str], allowlist: List[str]) -> List[str]:
        """Hard-filter ``ranked`` to allowlisted ids, preserving allowlist preference order.

        First listed model that is available becomes primary; later entries are
        fallbacks. Models not in the allowlist never enter the chain (unlike
        cost/latency caps, which only reorder). Empty result means none of the
        allowlisted models were discovered — callers fail loud at call time.
        """
        available = set(ranked)
        return [m for m in allowlist if m in available]

    def build_task_fallbacks(self) -> None:
        """Build fallback chains for all configured task types and components.

        Optional partitions:
          1. ``task_cost_caps`` — cost_tier above cap moved to end (never drops)
          2. ``task_latency_slos`` — measured p50 above SLO (or unprobed) moved to end
          3. ``task_model_allowlists`` — hard filter to named models in preference order
             (drops everything else; may yield empty chain if none available)

        Winner for classify/NER-style tasks is then the highest-ranked model that
        still meets the latency SLO (capability/cost/latency weights still apply
        inside the preferred pool), unless an allowlist overrides to preference order.
        """
        if not self._strategy['enabled']:
            return
        cost_caps: Dict[str, Any] = self._strategy.get('task_cost_caps') or {}
        latency_slos: Dict[str, Any] = self._strategy.get('task_latency_slos') or {}
        allowlists: Dict[str, Any] = self._strategy.get('task_model_allowlists') or {}

        def _chain_for(task_type: str) -> List[str]:
            ranked = self.select_models_for_task(task_type)
            cap_cfg = cost_caps.get(task_type)
            if cap_cfg and cap_cfg.get('max_cost_tier') is not None:
                ranked = self._build_capped_chain(ranked, int(cap_cfg['max_cost_tier']))
            slo_cfg = latency_slos.get(task_type)
            if slo_cfg and slo_cfg.get('max_p50_latency_ms') is not None:
                ranked = self._build_latency_slo_chain(ranked, float(slo_cfg['max_p50_latency_ms']))
            allow = allowlists.get(task_type)
            if allow:
                filtered = self._apply_task_model_allowlist(ranked, [str(m).strip() for m in allow])
                if not filtered:
                    logger.warning(
                        f"task_model_allowlist_empty task={task_type} "
                        f"allowlist={list(allow)} available={len(ranked)} "
                        f"action=empty_chain"
                    )
                return filtered
            return ranked

        fallbacks: Dict[str, List[str]] = {}
        for task_type in self._strategy['task_type_preferences'].keys():
            fallbacks[task_type] = _chain_for(task_type)
        for component, task_type in self._strategy['component_task_types'].items():
            fallbacks[component] = _chain_for(task_type)
        for dimension, task_type in self._strategy['dimension_task_types'].items():
            fallbacks[dimension] = _chain_for(task_type)
        with self._lock:
            self._model_fallbacks = fallbacks
        logger.debug(f"llm_provider_fallbacks_built chains={len(fallbacks)}")

    def _select_startup_probe_models(self) -> List[str]:
        """Pick models to time at boot (cold-start warm + latency seed).

        When ``task_model_allowlists`` is configured, probe **only** those
        selected task models that are actually discovered (allowlist ∩ available)
        — not the full discovery catalog (70+) and not a broad cheap-family sweep.
        That matches classify/extract fallback chains from
        ``task_model_allowlists`` (see ``base.yaml`` preference order).

        When no allowlists are set, fall back to cheap/latest-per-family selection
        capped by ``max_models`` / ``prefer_cost_tier_max``.

        ``always_include_models`` adds extra discovered ids after the above.
        """
        probe_cfg = self._strategy['startup_inference_probe']
        available = set(self._available_models)
        allowlists = self._strategy.get('task_model_allowlists') or {}

        picked: List[str] = []
        seen: set = set()

        def _add(name: str) -> None:
            if name and name in available and name not in seen:
                picked.append(name)
                seen.add(name)

        # Path A — target-task selected models only (preferred when allowlists exist).
        if isinstance(allowlists, dict) and allowlists:
            for allow in allowlists.values():
                if not isinstance(allow, list):
                    continue
                for mid in allow:
                    _add(str(mid).strip())
        else:
            # Path B — legacy cheap/latest family probe when no task allowlists.
            max_models = int(probe_cfg['max_models'])
            prefer_tier = int(probe_cfg['prefer_cost_tier_max'])
            registry = self._registry.registry
            by_family: Dict[str, Tuple[str, Tuple[Any, ...], int]] = {}
            for model in self._available_models:
                meta = registry.get(model) or {}
                cost_tier = int(meta.get('cost_tier', _NEUTRAL_COST_TIER))
                if cost_tier > prefer_tier:
                    continue
                family = infer_model_family(model) or model
                recency = infer_model_recency(model)
                prev = by_family.get(family)
                if prev is None or recency > prev[1]:
                    by_family[family] = (model, tuple(recency), cost_tier)
            candidates = list(by_family.values())
            candidates.sort(key=lambda t: (t[2], tuple(-x for x in t[1])))
            for m, _, _ in candidates[:max_models]:
                _add(m)

        for mid in (probe_cfg.get('always_include_models') or []):
            _add(str(mid).strip())
        return picked

    async def _timed_inference_probe(
        self,
        provider: str,
        api_key: str,
        model: str,
        base_url: Optional[str],
        timeout_s: float,
        max_tokens: int,
        prompt: str,
    ) -> Tuple[bool, Optional[float], Optional[str]]:
        """Run one short generation and return (ok, latency_ms, error_reason)."""
        t0 = time.monotonic()
        try:
            if provider == 'anthropic':
                from anthropic import Anthropic  # noqa: PLC0415
                client = Anthropic(api_key=api_key, base_url=base_url, timeout=timeout_s)
                await asyncio.to_thread(
                    client.messages.create,
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                )
                return True, (time.monotonic() - t0) * 1000.0, None
            if provider in _OPENAI_COMPAT_PROVIDERS:
                from openai import OpenAI  # noqa: PLC0415
                client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)
                await asyncio.to_thread(
                    client.chat.completions.create,
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                )
                return True, (time.monotonic() - t0) * 1000.0, None
            return False, None, f"unsupported_provider:{provider}"
        except _TRANSIENT_PROBE_ERRORS as e:
            # Transient: keep model selectable but do not seed a latency sample.
            logger.debug(
                f"startup_inference_probe_transient model={model} "
                f"error_type={type(e).__name__}"
            )
            return True, None, type(e).__name__
        except Exception as e:  # noqa: BLE001
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            raw = str(e).replace('\n', ' ').strip()
            extracted = _extract_error_message(raw)
            if _is_output_length_only_failure(extracted):
                # Routed successfully; record wall time (reasoning models often hit this).
                return True, elapsed_ms, None
            if len(extracted) > _LOG_ERROR_MAX_LEN:
                extracted = extracted[: _LOG_ERROR_MAX_LEN - 3] + '...'
            return False, None, f"{type(e).__name__}: {extracted}" if extracted else type(e).__name__

    def _api_key_for_model(self, model: str) -> Tuple[Optional[str], str]:
        """Resolve (api_key, probe_provider) for a discovered model id.

        Prefer the native provider key (``google`` for Gemini). Dual-path: when
        Gemini is only present via OpenAI-compat discovery under ``OPENAI_API_KEY``,
        fall back to the openai key and OpenAI-compat probe path.
        """
        provider = _detect_provider(model)
        key = self._validated_providers.get(provider)
        if key:
            return key, provider
        if provider == 'google' and 'openai' in self._validated_providers:
            return self._validated_providers['openai'], 'openai'
        if provider == 'unknown' and 'openai' in self._validated_providers:
            return self._validated_providers['openai'], 'openai'
        return None, provider

    async def probe_startup_inference(self) -> Dict[str, Dict[str, Any]]:
        """Time a short generation per selected model and seed ``runtime_stats``.

        Seeds ``p50_latency_ms`` with ``sample_count >= runtime_adaptation.min_samples``
        so the ranker applies measured latency immediately (cost preference still
        dominates via weights / ``task_cost_caps``; same-cost pairs prefer fastest
        measured inference, then latest SKU via recency tiebreak).

        :return: Dict[str, Dict[str, Any]] - model -> RuntimeStats.to_dict() for seeded models
        """
        probe_cfg = self._strategy['startup_inference_probe']
        if not bool(probe_cfg['enabled']):
            logger.info("startup_inference_probe_skipped reason=disabled")
            return {}
        if not self._available_models or not self._validated_providers:
            logger.warning("startup_inference_probe_skipped reason=no_models_or_keys")
            return {}
        models = self._select_startup_probe_models()
        if not models:
            logger.warning("startup_inference_probe_skipped reason=no_probe_candidates")
            return {}
        base_url_raw = self._config['llm_base_url']
        base_url = base_url_raw if base_url_raw else None
        timeout = float(probe_cfg['timeout_seconds'])
        max_tokens = int(probe_cfg['max_tokens'])
        prompt = str(probe_cfg['prompt']).strip()
        seed_samples = max(int(self._runtime_min_samples), 1)

        async def _one(model: str) -> Tuple[str, bool, Optional[float], Optional[str]]:
            api_key, provider = self._api_key_for_model(model)
            if not api_key:
                return model, False, None, 'no_api_key_for_provider'
            ok, lat_ms, err = await self._timed_inference_probe(
                provider, api_key, model, base_url, timeout, max_tokens, prompt,
            )
            return model, ok, lat_ms, err

        results = await asyncio.gather(*[_one(m) for m in models])
        seeded: Dict[str, Dict[str, Any]] = {}
        failed: Dict[str, str] = {}
        for model, ok, lat_ms, err in results:
            if ok and lat_ms is not None:
                seeded[model] = RuntimeStats(
                    p50_latency_ms=float(lat_ms),
                    error_rate=0.0,
                    quality_score=None,
                    sample_count=seed_samples,
                ).to_dict()
                seeded[model]['source'] = 'startup_inference_probe'
            elif not ok:
                failed[model] = err or 'unknown'
        with self._lock:
            merged = dict(self._runtime_stats)
            merged.update(seeded)
            self._runtime_stats = merged
        lat_parts = [
            f"{m}={float(seeded[m]['p50_latency_ms']):.0f}" for m in sorted(seeded)
        ]
        logger.info(
            f"startup_inference_probe_complete probed={len(models)} seeded={len(seeded)} "
            f"failed={len(failed)} seed_samples={seed_samples} "
            f"latencies_ms={{{', '.join(lat_parts)}}}"
            + (f" failed_models={failed}" if failed else "")
        )
        return seeded

    async def validate_api_keys(self, live_check: bool = False) -> Dict[str, str]:
        """Validate API keys and discover the model universe from the vendor/gateway.

        Discovery is the only source of available models. The flow per
        configured provider entry:
            1. Read the env var named in `key_env_var`. If absent or shorter
               than `min_api_key_length`, skip — that provider has no key.
            2. Call `client.models.list()` and keep every chat model returned.
               This is the model universe for that provider; corporate gateway
               entitlements are honoured automatically because the gateway
               only lists what the tenant can call.
            3. If `live_check=True`, additionally probe each discovered model
               with a one-token completion. Drop any that fail (catches
               models the gateway lists but does not actually route).

        If `models.list()` fails or returns no chat models for a provider (invalid
        key, network down, gateway misconfiguration), that provider is not added
        to the validated map — there is no discoverable model id to rank or cache.

        :param live_check: bool - If True, probe every discovered model individually
        :return: Dict[str, str] - Validated provider -> api_key mapping
        """
        api_keys_config = self._config['llm_api_keys']
        client_settings = self._config['llm_client_settings']
        min_len = int(client_settings['min_api_key_length'])
        validation_timeout = float(client_settings['validation_timeout'])
        validation_max_tokens = int(self._config['llm_models']['max_tokens_validation'])
        base_url_raw = self._config['llm_base_url']
        base_url = base_url_raw if base_url_raw else None

        validated: Dict[str, str] = {}
        models_by_provider: Dict[str, List[str]] = {}
        for provider_name, entries in api_keys_config.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                env_var = entry['key_env_var']
                key = os.getenv(env_var, '').strip()
                if not key or len(key) < min_len:
                    continue
                discovered = await self._discover_provider_models(
                    provider_name, key, base_url, validation_timeout
                )
                if not discovered:
                    logger.warning(f"llm_provider_discovery_empty provider={provider_name} env_var={env_var} action=verify_api_key_or_gateway_models_endpoint")
                    continue

                if live_check:
                    # Probe every discovered model concurrently. The serial loop previously
                    # added probes end-to-end (reasoning models can take 5-10s each), making
                    # startup O(N * slowest_probe). `gather` collapses it to ~max(probe_latency)
                    # for the provider. Order of `discovered` is preserved by zipping results
                    # back to their candidate name, so provider model precedence in the merged
                    # list is unchanged.
                    probe_results = await asyncio.gather(*[self._live_check_model(provider_name, key, candidate, base_url, validation_timeout, validation_max_tokens) for candidate in discovered])
                    accessible: List[str] = []
                    drop_reasons: Dict[str, str] = {}
                    for candidate, (ok, reason) in zip(discovered, probe_results):
                        if ok:
                            accessible.append(candidate)
                        else:
                            drop_reasons[candidate] = reason or 'unknown'
                    if drop_reasons:
                        logger.debug(f"llm_provider_partial_model_access provider={provider_name} discovered={len(discovered)} accessible={len(accessible)} dropped={len(drop_reasons)} dropped_models={drop_reasons}")  # noqa: E501
                    if not accessible:
                        logger.warning(f"llm_provider_no_model_accessible provider={provider_name} probed={len(discovered)}")
                        continue
                    discovered = accessible

                logger.info(f"llm_provider_discovery_ok provider={provider_name} models_found={len(discovered)} live_check={live_check}")
                validated[provider_name] = key
                models_by_provider[provider_name] = discovered
                break

        self._validated_providers = validated
        merged: List[str] = []
        for provider_name, models in models_by_provider.items():
            for model in models:
                if model not in merged:
                    merged.append(model)
        merged_before = list(merged)
        merged = LLMProvider._without_selection_exclusions(merged)
        if len(merged) < len(merged_before):
            dropped_models = sorted(set(merged_before) - set(merged))
            logger.info(f"llm_provider_selection_exclusions_applied dropped_count={len(merged_before) - len(merged)} dropped_models={dropped_models}")
        self._available_models = merged
        self._registry.update_models(self._available_models)
        # Rebuild eagerly: ranking and `get_meta` both read `self._registry.registry`
        # which is materialised only by `build()`. Without this, every `rank()` call
        # would see an empty dict and log `ranker_unregistered_model` for every model.
        self._registry.build()

        logger.debug(f"llm_provider_keys_validated providers={list(validated.keys())} available_models={len(self._available_models)} live_check={live_check}")
        return validated

    @staticmethod
    async def _live_check_model(provider: str, api_key: str, model: str, base_url: Optional[str], timeout: float, max_tokens: int) -> Tuple[bool, Optional[str]]:  # noqa: ASYNC109
        """Issue a one-token completion to confirm a single (provider, model) is usable.

        Catches `Exception` deliberately: this is an opportunistic liveness probe whose only
        job is to classify a model as accessible or not. Any exception (auth, rate limit,
        network, SDK, gateway entitlement) means the model is unusable right now. The caller
        aggregates the (model -> reason) map into a single summary line per provider; we
        return `(False, '<error_type>: <short message>')` so the operator can tell at a
        glance whether a drop was a routing issue, an auth failure, or rate limiting.
        Returning False removes only this model from `available_models` — the provider key
        remains valid for any other model that does succeed.

        :return: Tuple[bool, Optional[str]] - (accessible, drop_reason). reason=None when accessible.
        """
        try:
            if provider == 'anthropic':
                from anthropic import Anthropic  # noqa: PLC0415
                client = Anthropic(api_key=api_key, base_url=base_url, timeout=timeout)
                await asyncio.to_thread(
                    client.messages.create,
                    model=model,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=max_tokens,
                )
                return True, None
            if provider in _OPENAI_COMPAT_PROVIDERS:
                from openai import OpenAI  # noqa: PLC0415
                client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
                await asyncio.to_thread(
                    client.chat.completions.create,
                    model=model,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=max_tokens,
                )
                return True, None
            return False, f"unsupported_provider:{provider}"
        except _TRANSIENT_PROBE_ERRORS as e:
            # Timeout / connection failure: the request never reached a routing verdict, so the
            # model is not provably inaccessible. Dropping it here permanently removes an entitled
            # model at boot because a cold-gateway probe outran the tight `validation_timeout`
            # (real calls get `client_timeout`, far larger). Treat as reachable — the probe's job
            # is to catch entitlement/auth/routing failures, not to fail a slow gateway.
            logger.debug(f"llm_provider_probe_transient_kept provider={provider} model={model} error_type={type(e).__name__}")
            return True, None
        except Exception as e:  # noqa: BLE001
            # Vendor error bodies are nested JSON wrapped in repr() — extract the human-readable
            # `message` if we can find it, otherwise fall back to a bounded slice. This keeps the
            # aggregated `dropped_models={...}` line legible without dumping kilobytes per provider.
            raw = str(e).replace('\n', ' ').strip()
            extracted = _extract_error_message(raw)
            # Reachability semantics: the probe answers "did this (key, model) tuple route?",
            # not "did the model produce a useful response?". Reasoning-style models (gpt-5.x,
            # o-series) consume the entire `max_tokens` budget on internal thinking and the
            # gateway returns a `BadRequestError` saying the budget was exhausted — that error
            # *proves* routing worked. Same for the few SKUs that enforce a higher token-floor:
            # the request reached the model and was rejected for an output-length policy, not
            # for entitlement. Treat both as accessible so the registry doesn't silently lose
            # models that real downstream calls (with a normal `max_tokens`) would succeed on.
            if _is_output_length_only_failure(extracted):
                return True, None
            if len(extracted) > 200:
                extracted = extracted[:197] + '...'
            return False, f"{type(e).__name__}: {extracted}" if extracted else type(e).__name__

    @staticmethod
    async def _discover_provider_models(provider: str, api_key: str, base_url: Optional[str], timeout: float) -> List[str]:  # noqa: ASYNC109
        """List chat models the validated key can see via the provider's models endpoint.

        Returns an empty list on any error (network, auth, SDK incompatibility). Filters out
        non-chat artifacts (embeddings, image, audio) so the registry only ever holds models
        a chat-completion call can target. Catches `Exception` because discovery is best-effort
        — any failure should leave the provider with zero models, never raise.
        """
        try:
            if provider == 'anthropic':
                from anthropic import Anthropic  # noqa: PLC0415
                client = Anthropic(api_key=api_key, base_url=base_url, timeout=timeout)
                page = await asyncio.to_thread(client.models.list)
                ids = [getattr(m, 'id', '') for m in getattr(page, 'data', [])]
                return [m for m in ids if m and _is_chat_model(m)]
            if provider in _OPENAI_COMPAT_PROVIDERS:
                from openai import OpenAI  # noqa: PLC0415
                client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
                page = await asyncio.to_thread(client.models.list)
                ids = [getattr(m, 'id', '') for m in getattr(page, 'data', [])]
                return [m for m in ids if m and _is_chat_model(m)]
            return []
        except Exception as e:  # noqa: BLE001
            logger.warning(f"llm_provider_discovery_failed provider={provider} error_type={type(e).__name__} error={str(e)[:200]}")
            return []

    def initialize_cached_clients(self) -> None:
        """Create and cache one LLM client per validated provider."""
        try:
            from llm_core.llm_client import AnthropicLLMClient, OpenAILLMClient  # noqa: PLC0415
        except ImportError as e:
            logger.warning(f"llm_client_libraries_unavailable cached_clients=0 error_type={type(e).__name__}")
            return
        llm_settings = self._config['llm_client_settings']
        llm_models = self._config['llm_models']
        base_url_raw = self._config['llm_base_url']
        base_url = base_url_raw if base_url_raw else None
        timeout = int(llm_settings['client_timeout'])
        max_retries = int(llm_settings['max_retries'])
        temperature = float(llm_models['temperature'])
        no_temperature_models = frozenset(llm_models.get('temperature_unsupported_models') or [])
        prompt_caching_enabled = bool((llm_models.get('prompt_caching') or {}).get('enabled', False))
        provider_models: Dict[str, str] = {}
        for model in self._available_models:
            provider = _detect_provider(model)
            if provider not in provider_models and provider in self._validated_providers:
                provider_models[provider] = model
        # OpenAI-compat google key with only openai-labeled models still needs a client.
        for provider_name in self._validated_providers:
            if provider_name in _OPENAI_COMPAT_PROVIDERS and provider_name not in provider_models:
                for model in self._available_models:
                    if _detect_provider(model) in _OPENAI_COMPAT_PROVIDERS | {'unknown'}:
                        provider_models[provider_name] = model
                        break
        for provider, model in provider_models.items():
            api_key = self._validated_providers[provider]
            try:
                if provider == 'anthropic':
                    self._cached_clients['anthropic'] = AnthropicLLMClient(
                        api_key=api_key, model=model, temperature=temperature,
                        timeout=timeout, max_retries=max_retries, base_url=base_url,
                        no_temperature_models=no_temperature_models,
                        use_prompt_caching=prompt_caching_enabled,
                    )
                elif provider in _OPENAI_COMPAT_PROVIDERS:
                    self._cached_clients[provider] = OpenAILLMClient(
                        api_key=api_key, model=model, temperature=temperature,
                        timeout=timeout, max_retries=max_retries, base_url=base_url,
                        no_temperature_models=no_temperature_models,
                    )
                logger.debug(f"llm_client_cached provider={provider} model={model}")
            except (ImportError, ValueError, TypeError) as e:
                logger.warning(f"llm_client_cache_failed provider={provider} error_type={type(e).__name__} error={str(e)}")
        logger.debug(f"llm_provider_clients_initialized count={len(self._cached_clients)}")

    def get_client_for_model(self, model: str) -> Tuple[Any, str]:
        """Return cached client for a model.

        :param model: str - Model name
        :return: Tuple[LLMClient, str] - (client, provider_name)
        :raises ConfigurationError: If no client available for this model's provider
        """
        provider = _detect_provider(model)
        if provider == 'anthropic' and 'anthropic' in self._cached_clients:
            return self._cached_clients['anthropic'], 'anthropic'
        if provider == 'google' and 'google' in self._cached_clients:
            return self._cached_clients['google'], 'google'
        # Dual-path: Gemini discovered under OPENAI_API_KEY only.
        if provider == 'google' and 'openai' in self._cached_clients:
            return self._cached_clients['openai'], 'openai'
        if provider in ('openai', 'unknown') and 'openai' in self._cached_clients:
            return self._cached_clients['openai'], 'openai'
        if self._cached_clients:
            client_type = next(iter(self._cached_clients.keys()))
            return self._cached_clients[client_type], client_type
        raise ConfigurationError(f"No LLM client available for model={model}")

    def get_default_client(self) -> Optional[Any]:
        """Return first available cached client, or None."""
        if not self._cached_clients:
            return None
        return next(iter(self._cached_clients.values()))

    def get_fallback_chain(self, task_type: str) -> List[str]:
        """Return fallback chain for a task type or component name."""
        with self._lock:
            chain = self._model_fallbacks.get(task_type)
        if chain is not None:
            return list(chain)
        return list(self._available_models)

    def get_primary_model(self, task_type: str) -> str:
        """First model in the task fallback chain (allowlist ∩ discovered).

        :param task_type: str - Task type or component name (e.g. l0_entity_extraction)
        :return: str - Primary model id
        :raises ConfigurationError: If the chain is empty (none of the allowlisted
            models were discovered / live-checked)
        """
        chain = self.get_fallback_chain(task_type)
        if not chain:
            raise ConfigurationError(
                f"No available model for task_type={task_type!r} "
                f"(task_model_allowlists ∩ discovered empty)"
            )
        return chain[0]

    def refresh_rankings(self) -> None:
        """Pull fresh runtime stats from FeedbackStore (when wired) and rebuild fallback chains.

        Feedback observations overwrite per-model keys. Empty collect leaves
        startup-inference seeds intact so boot-measured latency is not wiped.
        """
        if not self._signal_adapter.enabled:
            return
        new_stats = self._signal_adapter.collect()
        with self._lock:
            if new_stats:
                merged = dict(self._runtime_stats)
                merged.update(new_stats)
                self._runtime_stats = merged
        self.build_task_fallbacks()
        logger.info(f"llm_provider_rankings_refreshed models_with_stats={len(new_stats)}")

    async def start_background_refresh(self, interval_s: Optional[int] = None) -> None:
        """Start a background asyncio task that periodically calls refresh_rankings."""
        if self._refresh_task is not None and not self._refresh_task.done():
            return
        if not self._signal_adapter.enabled:
            logger.debug("llm_provider_background_refresh_skipped reason=no_feedback_store")
            return
        if not self._runtime_enabled:
            logger.debug("llm_provider_background_refresh_skipped reason=runtime_adaptation_disabled")
            return
        interval = int(interval_s if interval_s is not None else self._refresh_interval_s)
        loop = asyncio.get_running_loop()
        self._refresh_stop = asyncio.Event()
        self._refresh_task = loop.create_task(self._refresh_loop(interval))
        logger.info(f"llm_provider_background_refresh_started interval_s={interval}")

    async def stop_background_refresh(self) -> None:
        """Stop the background refresh task; safe to call when not started."""
        if self._refresh_stop is not None:
            self._refresh_stop.set()
        if self._refresh_task is None:
            return
        try:
            await asyncio.wait_for(self._refresh_task, timeout=5.0)
        except asyncio.TimeoutError:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                logger.info("llm_provider_background_refresh_cancelled")
        finally:
            self._refresh_task = None
            self._refresh_stop = None
        logger.info("llm_provider_background_refresh_stopped")

    async def _refresh_loop(self, interval_s: int) -> None:
        """Internal loop body — calls refresh_rankings on cadence until stop signaled."""
        while True:
            try:
                self.refresh_rankings()
            except (RuntimeError, ValueError, TypeError, KeyError) as e:
                logger.warning(f"llm_provider_refresh_loop_error error_type={type(e).__name__} error={str(e)}")
            if self._refresh_stop is None:
                return
            try:
                await asyncio.wait_for(self._refresh_stop.wait(), timeout=interval_s)
                return
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                logger.info("llm_provider_refresh_loop_cancelled")
                raise

    def get_summary(self) -> Dict[str, Any]:
        """Return provider status summary."""
        with self._lock:
            chains_count = len(self._model_fallbacks)
            stats_count = len(self._runtime_stats)
        return {
            'available_models': len(self._available_models),
            'registry_size': len(self._registry.registry),
            'cached_clients': list(self._cached_clients.keys()),
            'fallback_chains': chains_count,
            'validated_providers': list(self._validated_providers.keys()),
            'runtime_adaptation_enabled': self._runtime_enabled,
            'runtime_stats_models': stats_count,
        }
