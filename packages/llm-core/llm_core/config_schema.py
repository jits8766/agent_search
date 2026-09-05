"""Required-key validation for the unified LLMProvider config contract.

Every key referenced by `LLMProvider` MUST be present in config. No silent defaults.
Missing keys raise `ConfigurationError` so deployment errors fail loudly at startup.

Pricing is intentionally NOT a service-config concern — it ships with `llm_core` in
`pricing.yaml` and is loaded via `llm_core.pricing.load_pricing()`. Per-service
pricing dicts have been removed (2026-04 refactor) so a model rate change is one
edit in one file rather than seven.

Model lists are also NOT a service-config concern — `LLMProvider.validate_api_keys`
discovers them from the vendor / corporate gateway via `client.models.list()`.
Service config only declares the env var names that hold the API keys.
"""
from typing import Any, Dict, List

from llm_core.exceptions import ConfigurationError

REQUIRED_TOP_LEVEL_KEYS: List[str] = [
    'llm_api_keys',
    'llm_models',
    'llm_client_settings',
    'llm_base_url',
    'model_selection_strategy',
    'model_capability_overrides',
    'model_latency_overrides',
]

REQUIRED_LLM_MODELS_KEYS: List[str] = [
    'temperature',
    'max_tokens_validation',
]

REQUIRED_LLM_CLIENT_SETTINGS_KEYS: List[str] = [
    'client_timeout',
    'max_retries',
    'min_api_key_length',
    'validation_timeout',
    'validation_max_tokens',
]

REQUIRED_STRATEGY_KEYS: List[str] = [
    'enabled',
    'weights',
    'weights_by_task_type',
    'runtime_adaptation',
    'task_type_preferences',
    'component_task_types',
    'dimension_task_types',
    'provider_tiebreak_priority',
    'startup_inference_probe',
]

REQUIRED_STARTUP_INFERENCE_PROBE_KEYS: List[str] = [
    'enabled',
    'max_models',
    'prefer_cost_tier_max',
    'timeout_seconds',
    'max_tokens',
    'prompt',
]

# Labels must match model_registry._detect_provider() return values.
ALLOWED_PROVIDER_TIEBREAK_IDS = frozenset({
    'anthropic',
    'openai',
    'google',
    'xai',
    'deepseek',
    'qwen',
    'zhipu',
    'unknown',
})

REQUIRED_WEIGHTS_KEYS: List[str] = ['capability', 'cost', 'latency']

REQUIRED_RUNTIME_ADAPTATION_KEYS: List[str] = [
    'enabled',
    'refresh_interval_s',
    'min_samples',
    'latency_window_s',
    'quality_window_s',
    'error_demotion_threshold',
    'latency_buckets_ms',
    'max_entries_per_type',
]


def _require(section: Any, keys: List[str], section_name: str) -> None:
    """Raise ConfigurationError if any key missing from a dict section."""
    if not isinstance(section, dict):
        raise ConfigurationError(f"LLM provider config '{section_name}' must be a dict")
    for key in keys:
        if key not in section:
            raise ConfigurationError(f"LLM provider config '{section_name}.{key}' is required")


def validate_provider_tiebreak_priority(raw: Any) -> List[str]:
    """Validate and return provider_tiebreak_priority (first = highest preference).

    :param raw: Any - Expected non-empty list of distinct allowed provider ids
    :return: List[str] - Normalized provider ids
    :raises ConfigurationError: If shape/contents invalid
    """
    if not isinstance(raw, list) or not raw:
        raise ConfigurationError(
            "LLM provider config 'model_selection_strategy.provider_tiebreak_priority' "
            "must be a non-empty list"
        )
    out: List[str] = []
    seen: set = set()
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise ConfigurationError(
                "LLM provider config 'model_selection_strategy.provider_tiebreak_priority' "
                "entries must be non-empty strings"
            )
        pid = item.strip().lower()
        if pid not in ALLOWED_PROVIDER_TIEBREAK_IDS:
            raise ConfigurationError(
                f"LLM provider config 'model_selection_strategy.provider_tiebreak_priority' "
                f"has unknown provider id {pid!r}; allowed={sorted(ALLOWED_PROVIDER_TIEBREAK_IDS)}"
            )
        if pid in seen:
            raise ConfigurationError(
                f"LLM provider config 'model_selection_strategy.provider_tiebreak_priority' "
                f"has duplicate provider id {pid!r}"
            )
        seen.add(pid)
        out.append(pid)
    return out


def validate_startup_inference_probe(raw: Any) -> Dict[str, Any]:
    """Validate startup_inference_probe block (timed generation at app boot for ranking).

    :param raw: Any - Expected dict with REQUIRED_STARTUP_INFERENCE_PROBE_KEYS
    :return: Dict[str, Any] - The validated dict (unchanged reference)
    :raises ConfigurationError: If shape/contents invalid
    """
    _require(raw, REQUIRED_STARTUP_INFERENCE_PROBE_KEYS, 'model_selection_strategy.startup_inference_probe')
    assert isinstance(raw, dict)
    if not isinstance(raw['enabled'], bool):
        raise ConfigurationError(
            "LLM provider config 'model_selection_strategy.startup_inference_probe.enabled' "
            "must be a bool"
        )
    max_models = raw['max_models']
    if not isinstance(max_models, int) or isinstance(max_models, bool) or max_models < 1:
        raise ConfigurationError(
            "LLM provider config 'model_selection_strategy.startup_inference_probe.max_models' "
            "must be an int >= 1"
        )
    prefer_tier = raw['prefer_cost_tier_max']
    if not isinstance(prefer_tier, int) or isinstance(prefer_tier, bool) or prefer_tier < 1:
        raise ConfigurationError(
            "LLM provider config 'model_selection_strategy.startup_inference_probe.prefer_cost_tier_max' "
            "must be an int >= 1"
        )
    timeout_s = raw['timeout_seconds']
    if not isinstance(timeout_s, (int, float)) or isinstance(timeout_s, bool) or float(timeout_s) <= 0:
        raise ConfigurationError(
            "LLM provider config 'model_selection_strategy.startup_inference_probe.timeout_seconds' "
            "must be a number > 0"
        )
    max_tokens = raw['max_tokens']
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        raise ConfigurationError(
            "LLM provider config 'model_selection_strategy.startup_inference_probe.max_tokens' "
            "must be an int >= 1"
        )
    prompt = raw['prompt']
    if not isinstance(prompt, str) or not prompt.strip():
        raise ConfigurationError(
            "LLM provider config 'model_selection_strategy.startup_inference_probe.prompt' "
            "must be a non-empty string"
        )
    # Optional: force-probe these model ids at boot (cold-start warm) when discovered.
    if 'always_include_models' in raw:
        always = raw['always_include_models']
        if not isinstance(always, list):
            raise ConfigurationError(
                "LLM provider config 'model_selection_strategy.startup_inference_probe."
                "always_include_models' must be a list of model id strings"
            )
        seen: set = set()
        for item in always:
            if not isinstance(item, str) or not item.strip():
                raise ConfigurationError(
                    "LLM provider config 'model_selection_strategy.startup_inference_probe."
                    "always_include_models' entries must be non-empty strings"
                )
            mid = item.strip()
            if mid in seen:
                raise ConfigurationError(
                    "LLM provider config 'model_selection_strategy.startup_inference_probe."
                    f"always_include_models' has duplicate model id {mid!r}"
                )
            seen.add(mid)
    return raw


def validate_task_latency_slos(raw: Any) -> None:
    """Validate optional task_latency_slos map (task_type -> {max_p50_latency_ms}).

    Omitted or empty dict is fine. When present, every entry must declare a
    positive ``max_p50_latency_ms`` (milliseconds).
    """
    if raw is None:
        return
    if not isinstance(raw, dict):
        raise ConfigurationError(
            "LLM provider config 'model_selection_strategy.task_latency_slos' must be a dict"
        )
    for task_type, cfg in raw.items():
        if not isinstance(task_type, str) or not task_type.strip():
            raise ConfigurationError(
                "LLM provider config 'model_selection_strategy.task_latency_slos' "
                "keys must be non-empty task_type strings"
            )
        if not isinstance(cfg, dict) or 'max_p50_latency_ms' not in cfg:
            raise ConfigurationError(
                f"LLM provider config 'model_selection_strategy.task_latency_slos.{task_type}' "
                "must be a dict with max_p50_latency_ms"
            )
        val = cfg['max_p50_latency_ms']
        if not isinstance(val, (int, float)) or isinstance(val, bool) or float(val) <= 0:
            raise ConfigurationError(
                f"LLM provider config 'model_selection_strategy.task_latency_slos.{task_type}."
                "max_p50_latency_ms' must be a number > 0"
            )


def validate_task_model_allowlists(raw: Any) -> None:
    """Validate optional task_model_allowlists (task_type -> ordered model ids).

    Omitted or empty dict is fine. When present, each task maps to a non-empty
    list of distinct non-empty model id strings. List order is preference order
    (first available wins); models not listed never enter that task's chain.
    """
    if raw is None:
        return
    if not isinstance(raw, dict):
        raise ConfigurationError(
            "LLM provider config 'model_selection_strategy.task_model_allowlists' must be a dict"
        )
    for task_type, models in raw.items():
        if not isinstance(task_type, str) or not task_type.strip():
            raise ConfigurationError(
                "LLM provider config 'model_selection_strategy.task_model_allowlists' "
                "keys must be non-empty task_type strings"
            )
        if not isinstance(models, list) or not models:
            raise ConfigurationError(
                f"LLM provider config 'model_selection_strategy.task_model_allowlists."
                f"{task_type}' must be a non-empty list of model ids"
            )
        seen: set = set()
        for item in models:
            if not isinstance(item, str) or not item.strip():
                raise ConfigurationError(
                    f"LLM provider config 'model_selection_strategy.task_model_allowlists."
                    f"{task_type}' entries must be non-empty strings"
                )
            mid = item.strip()
            if mid in seen:
                raise ConfigurationError(
                    f"LLM provider config 'model_selection_strategy.task_model_allowlists."
                    f"{task_type}' has duplicate model id {mid!r}"
                )
            seen.add(mid)


def validate_provider_config(config: Dict[str, Any]) -> None:
    """Validate the full LLMProvider config contract.

    :param config: Dict[str, Any] - Service config dict
    :raises ConfigurationError: If any required key is missing
    """
    if not isinstance(config, dict):
        raise ConfigurationError("LLM provider config must be a dict")
    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in config:
            raise ConfigurationError(f"LLM provider requires '{key}' in config")
    _require(config['llm_models'], REQUIRED_LLM_MODELS_KEYS, 'llm_models')
    _require(config['llm_client_settings'], REQUIRED_LLM_CLIENT_SETTINGS_KEYS, 'llm_client_settings')
    _require(config['model_selection_strategy'], REQUIRED_STRATEGY_KEYS, 'model_selection_strategy')
    _require(config['model_selection_strategy']['weights'], REQUIRED_WEIGHTS_KEYS, 'model_selection_strategy.weights')
    _require(config['model_selection_strategy']['runtime_adaptation'], REQUIRED_RUNTIME_ADAPTATION_KEYS, 'model_selection_strategy.runtime_adaptation')
    validate_provider_tiebreak_priority(config['model_selection_strategy']['provider_tiebreak_priority'])
    validate_startup_inference_probe(config['model_selection_strategy']['startup_inference_probe'])
    if 'task_latency_slos' in config['model_selection_strategy']:
        validate_task_latency_slos(config['model_selection_strategy']['task_latency_slos'])
    if 'task_model_allowlists' in config['model_selection_strategy']:
        validate_task_model_allowlists(config['model_selection_strategy']['task_model_allowlists'])
