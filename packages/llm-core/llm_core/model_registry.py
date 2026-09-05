"""Model registry — derives capability/cost/latency tiers for the runtime model set.

The registry is now driven entirely by the runtime model set (the models the
LLMProvider has discovered against the validated API key). It does NOT take a
hand-curated declared list and does NOT take a per-service pricing dict — both
of those used to live in service `base.yaml` and have been moved to
`llm_core/pricing.yaml`.

Tier semantics:
  * `capability_tier` 1..5: higher = stronger model. Inferred from the model
    name via `pricing.infer_capability_tier`; overridable per service via
    `model_capability_overrides`.
  * `cost_tier` 1..5: higher = more expensive. Derived from the relative
    position of the model's `input + output` rate within the priced subset of
    the registry. Models with no pricing entry get neutral tier 3.
  * `latency_tier` 1..5: 1 = fastest. Defaults to neutral 3; overridable via
    `model_latency_overrides` (vendors do not expose latency either).
"""
from typing import Any, Dict, List, Optional

from llm_core.logging_utils import get_logger
from llm_core.pricing import infer_capability_tier, load_pricing

logger = get_logger(__name__)

_TIER_MIN = 1
_TIER_MAX = 5
_NEUTRAL_TIER = 3
_COST_TIER_BINS = 4


def _detect_provider(model_name: str) -> str:
    """Detect LLM provider from model name prefix.

    :param model_name: str - Model identifier
    :return: str - Provider name (anthropic, openai, google, xai, zhipu, deepseek, qwen, unknown)
    """
    name_lower = model_name.lower()
    if name_lower.startswith('claude'):
        return 'anthropic'
    if name_lower.startswith(('gpt', 'o1', 'o3', 'o4', 'dall-e', 'text-embedding')):
        return 'openai'
    if name_lower.startswith('gemini'):
        return 'google'
    if name_lower.startswith('grok'):
        return 'xai'
    if name_lower.startswith('glm'):
        return 'zhipu'
    if name_lower.startswith('deepseek'):
        return 'deepseek'
    if 'qwen' in name_lower:
        return 'qwen'
    return 'unknown'


def _is_chat_model(model_name: str) -> bool:
    """Check if model is a chat/completion model.

    :param model_name: str - Model identifier
    :return: bool - True if chat model
    """
    excluded = ('text-embedding', 'titan-embed', 'dall-e', 'gpt-image', 'imagen', 'sora', 'veo')
    return not model_name.lower().startswith(excluded)


class ModelRegistry:
    """Builds and owns model metadata for the runtime model set.

    The registry no longer takes a pricing argument — it consults the bundled
    `llm_core/pricing.yaml` via `pricing.load_pricing()`. Services pass the
    list of models discovered against their API key (or empty if discovery has
    not run yet) and optional override maps.
    """

    def __init__(self, models: List[str], capability_overrides: Optional[Dict[str, int]] = None, latency_overrides: Optional[Dict[str, int]] = None):
        """Initialize registry.

        :param models: List[str] - Model identifiers the LLMProvider has registered
        :param capability_overrides: Optional[Dict[str, int]] - Static capability tier 1..5 per model
        :param latency_overrides: Optional[Dict[str, int]] - Static latency tier 1..5 per model (1=fastest)
        """
        self._models = list(models or [])
        self._cap_overrides = dict(capability_overrides or {})
        self._lat_overrides = dict(latency_overrides or {})
        self._registry: Dict[str, Dict[str, Any]] = {}

    @property
    def registry(self) -> Dict[str, Dict[str, Any]]:
        """Return defensive copy of the registry."""
        return {m: dict(meta) for m, meta in self._registry.items()}

    def update_models(self, models: List[str]) -> None:
        """Replace the model set. Caller must invoke `build()` again afterwards."""
        self._models = list(models or [])

    def build(self) -> Dict[str, Dict[str, Any]]:
        """Build the registry from the runtime model set + bundled pricing reference.

        Models with a pricing entry get a cost-derived `cost_tier`. Models
        without one are still included with neutral cost tier 3 (a warning is
        logged so the developer can add the rate to `pricing.yaml`). Capability
        is inferred from the model name and may be overridden via config.

        :return: Dict[str, Dict[str, Any]] - The built registry
        """
        self._registry.clear()
        if not self._models:
            return self._registry
        pricing = load_pricing()
        priced: Dict[str, Dict[str, float]] = {}
        unpriced: List[str] = []
        for model in self._models:
            if not _is_chat_model(model):
                continue
            entry = pricing.get(model)
            if isinstance(entry, dict) and 'input' in entry and 'output' in entry:
                priced[model] = entry
            else:
                unpriced.append(model)
        if not priced and not unpriced:
            logger.warning(f"model_registry_no_chat_models registered_count={len(self._models)}")
            return self._registry
        if priced:
            costs = {m: float(p['input']) + float(p['output']) for m, p in priced.items()}
            sorted_costs = sorted(costs.items(), key=lambda kv: kv[1])
            min_cost = sorted_costs[0][1]
            max_cost = sorted_costs[-1][1]
            cost_range = max_cost - min_cost if max_cost > min_cost else 1.0
            for model, total_cost in sorted_costs:
                normalised = (total_cost - min_cost) / cost_range if cost_range > 0 else 0.5
                cost_tier = max(_TIER_MIN, min(_TIER_MAX, int(normalised * _COST_TIER_BINS) + 1))
                cap_tier = self._cap_overrides.get(model, infer_capability_tier(model))
                cap_tier = max(_TIER_MIN, min(_TIER_MAX, int(cap_tier)))
                lat_tier = max(_TIER_MIN, min(_TIER_MAX, int(self._lat_overrides.get(model, _NEUTRAL_TIER))))
                self._registry[model] = {
                    'provider': _detect_provider(model),
                    'capability_tier': cap_tier,
                    'cost_tier': cost_tier,
                    'latency_tier': lat_tier,
                    'total_cost': total_cost,
                    'input_cost': float(priced[model]['input']),
                    'output_cost': float(priced[model]['output']),
                }
        for model in unpriced:
            cap_tier = max(_TIER_MIN, min(_TIER_MAX, int(self._cap_overrides.get(model, infer_capability_tier(model)))))
            lat_tier = max(_TIER_MIN, min(_TIER_MAX, int(self._lat_overrides.get(model, _NEUTRAL_TIER))))
            self._registry[model] = {
                'provider': _detect_provider(model),
                'capability_tier': cap_tier,
                'cost_tier': _NEUTRAL_TIER,
                'latency_tier': lat_tier,
                'total_cost': 0.0,
                'input_cost': 0.0,
                'output_cost': 0.0,
            }
            logger.warning(f"model_registry_unpriced_model_neutralized model={model} action=add_entry_to_llm_core_pricing_yaml")
        logger.info(f"model_registry_built models={len(self._registry)} priced={len(priced)} unpriced={len(unpriced)}")
        return self._registry

    def get_meta(self, model: str) -> Dict[str, Any]:
        """Return metadata for model, with neutral defaults if missing.

        :param model: str - Model name
        :return: Dict[str, Any] - Metadata dict
        """
        if model in self._registry:
            return dict(self._registry[model])
        return {
            'provider': _detect_provider(model),
            'capability_tier': infer_capability_tier(model),
            'cost_tier': _NEUTRAL_TIER,
            'latency_tier': _NEUTRAL_TIER,
            'total_cost': 0.0,
            'input_cost': 0.0,
            'output_cost': 0.0,
        }
