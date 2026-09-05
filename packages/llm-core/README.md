# llm_core

Shared Python package for LLM provider discovery, model registry construction, weighted model ranking, runtime adaptation, pricing metadata, and Anthropic/OpenAI client wrappers.

**Package:** `llm-core`
**Python:** `>=3.12,<3.13`

## What It Provides

- `LLMProvider` - Validates configured API-key environment variables, discovers available chat models through provider model-list APIs, builds model metadata, ranks models by task, caches provider clients, and optionally refreshes rankings from runtime signals.
- `ModelRegistry` - Builds per-model metadata for the discovered model set: provider, capability tier, cost tier, latency tier, and token pricing. Capability is inferred from model name unless overridden by service config.
- `Ranker` - Scores models using normalized `{capability, cost, latency}` weights. Runtime latency and error-rate signals can adjust ranking when enough samples are available.
- `SignalAdapter` - Reads `latency`, `error`, and `eval_quality_score` entries from a FeedbackStore-like object and aggregates per-model runtime stats.
- `pricing.yaml` and `pricing.py` - Provide bundled per-model token rates, fallback pricing, and model-selection exclusions.
- `LLMClient` implementations - Provide async Anthropic and OpenAI clients, plus factory and adapter helpers.

## Model Discovery Contract

Services do not declare model lists in config. `LLMProvider.validate_api_keys(...)` discovers available chat models from each configured provider.

Service config declares API-key environment variable names:

```yaml
llm_api_keys:
  anthropic:
    - key_env_var: ANTHROPIC_API_KEY
  openai:
    - key_env_var: OPENAI_API_KEY
```

Discovery behavior:

1. Read each `key_env_var`.
2. Skip missing keys or keys shorter than `llm_client_settings.min_api_key_length`.
3. Call the provider model-list API.
4. Keep chat models only.
5. If `live_check=True`, probe discovered models with a small completion request and drop models that fail non-transient routing or access checks.
6. Apply selection exclusions from `llm_core/pricing.yaml`.

Supported runtime client providers are Anthropic and OpenAI. Pricing metadata includes additional model families, but Gemini and other non-Anthropic/OpenAI chat clients are not implemented in `LLMClientFactory`.

## Pricing and Selection Exclusions

`llm_core/pricing.yaml` is the package-level pricing reference. It contains:

- Per-model input and output token prices in USD per 1M tokens.
- `default` fallback pricing for unknown models in cost computation.
- `selection_exclusions`, an explicit list of model IDs to exclude from ranking.
- `selection_max_output_usd_per_million`, which excludes priced chat SKUs whose output price is strictly greater than the configured cap.

Models missing from `pricing.yaml` remain rankable. `ModelRegistry` assigns them neutral cost tier `3`, zero price fields, and logs a warning.

## Typical FastAPI Lifespan Wiring

```python
from llm_core import LLMProvider


async def lifespan(app):
    config_dict = load_config()
    provider = LLMProvider(config_dict, feedback_store=app_state.feedback_store)

    await provider.validate_api_keys(live_check=True)
    provider.build_model_registry()
    provider.build_task_fallbacks()
    provider.initialize_cached_clients()
    await provider.start_background_refresh()

    app_state.llm_provider = provider
    try:
        yield
    finally:
        await provider.stop_background_refresh()
```

Consumers can request ranked models or clients:

```python
ranked = app_state.llm_provider.get_ranked_models_for_task("hypothesis_generation")
client, provider_name = app_state.llm_provider.get_client_for_model(ranked[0][0])
```

## Config Contract

`LLMProvider.__init__` validates required config keys through `validate_provider_config(...)` and raises `ConfigurationError` when required sections are missing.

Required top-level keys:

- `llm_api_keys`
- `llm_models`
- `llm_client_settings`
- `llm_base_url`
- `model_selection_strategy`
- `model_capability_overrides`
- `model_latency_overrides`

Required `llm_models` keys:

- `temperature`
- `max_tokens_validation`

Required `llm_client_settings` keys:

- `client_timeout`
- `max_retries`
- `min_api_key_length`
- `validation_timeout`
- `validation_max_tokens`

Required `model_selection_strategy` keys:

- `enabled`
- `weights`
- `weights_by_task_type`
- `runtime_adaptation`
- `task_type_preferences`
- `component_task_types`
- `dimension_task_types`

Required `model_selection_strategy.weights` keys:

- `capability`
- `cost`
- `latency`

Required `model_selection_strategy.runtime_adaptation` keys:

- `enabled`
- `refresh_interval_s`
- `min_samples`
- `latency_window_s`
- `quality_window_s`
- `error_demotion_threshold`
- `latency_buckets_ms`
- `max_entries_per_type`

`llm_token_pricing` is not required by `LLMProvider`.

## Ranking

`Ranker` computes:

```text
score =
  capability_weight * capability_tier
  + cost_weight * (6 - cost_tier)
  + latency_weight * (6 - effective_latency_tier)
```

Weights are normalized by `RankWeights.from_dict(...)`. All three keys are required, and the weight sum must be greater than zero.

Models with runtime error rate above `error_demotion_threshold` are moved after non-demoted models when enough samples are available.

`LLMProvider.build_task_fallbacks()` builds chains for configured task types, components, and dimensions. If `task_cost_caps` is present, models above a task's `max_cost_tier` are moved later in the chain rather than removed.

## Runtime Adaptation

Runtime adaptation is active only when both conditions are true:

- `model_selection_strategy.runtime_adaptation.enabled` is `true`
- A FeedbackStore-like object is passed to `LLMProvider`

When active, `start_background_refresh()` starts a task that periodically:

1. Calls `SignalAdapter.collect()`.
2. Stores per-model runtime stats.
3. Rebuilds task fallback chains.

`SignalAdapter` expects a duck-typed store with `get_entries(signal_type=..., limit=...)`. It reads:

- `latency`
- `error`
- `eval_quality_score`

Without a store, runtime adaptation is a no-op.

## Clients and Adapters

`LLMClientFactory.create(...)` creates clients by model prefix:

- `claude*` -> `AnthropicLLMClient`
- `gpt*`, `o1*`, `o3*`, `o4*` -> `OpenAILLMClient`

`AnthropicLLMClient` and `OpenAILLMClient` support:

- `call(...)`
- `call_with_messages(...)`
- temperature retry without the `temperature` parameter when the provider rejects it
- normalized token usage output

Adapters:

- `LLMJudgeClientAdapter` adapts async `LLMClient.call(...)` to return a string.
- `PromptLearnerSyncAdapter` wraps async calls behind a synchronous `generate(...)` method.

## Updating Pricing

Edit `llm_core/pricing.yaml`:

```yaml
my-new-model-id: {input: 1.5, output: 6.0}
```

Prices are USD per 1M tokens.

Restart the consuming service after pricing changes.

## Testing

```bash
pytest packages/llm-core/tests -v
```

The test suite covers provider config validation, discovery behavior, pricing exclusions, weighted ranking, runtime refresh, signal aggregation, logging utilities, and ranker behavior.
