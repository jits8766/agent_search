"""Offline-harness LLM model resolution (no hardcoded model ids).

Primary model = first entry of
``model_selection_strategy.task_model_allowlists.<task>`` that appears in
GoCaas discovery (same rule as production ``LLMProvider.get_fallback_chain``).

Override: env ``L0_GROUNDING_MODEL`` (HTTP harness injects extraction primary
when the form field is omitted).
"""
from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, Optional, Tuple

# Task key must match base.yaml model_selection_strategy.task_model_allowlists.
HARNESS_EXTRACTION_TASK = "l0_entity_extraction"
HARNESS_GROUNDING_MODEL_ENV = "L0_GROUNDING_MODEL"


def resolve_harness_grounding_model(
    *,
    llm_provider: Any = None,
    task_type: str = HARNESS_EXTRACTION_TASK,
    env_var: str = HARNESS_GROUNDING_MODEL_ENV,
) -> str:
    """Resolve LLMJ grounding model: env override, else task primary from provider/config.

    :param llm_provider: Optional live ``LLMProvider`` (preferred — already discovered)
    :param task_type: str - Allowlist task key (default: l0_entity_extraction)
    :param env_var: str - Override env var name
    :return: str - Non-empty model id
    :raises SystemExit: When no override and allowlist ∩ discovery is empty
    """
    override = (os.environ.get(env_var) or "").strip()
    if override:
        return override
    if llm_provider is not None:
        try:
            return str(llm_provider.get_primary_model(task_type)).strip()
        except Exception as exc:  # noqa: BLE001 — surface as harness exit
            raise SystemExit(
                f"ERROR: no primary model for task={task_type!r} "
                f"({type(exc).__name__}: {exc})"
            ) from exc
    return _bootstrap_primary_model(task_type)


def _bootstrap_primary_model(task_type: str) -> str:
    """CLI path: discover models from config keys, then pick task primary."""
    from llm_core.provider import LLMProvider  # noqa: PLC0415
    from semantic_search.config.loader import load_config  # noqa: PLC0415

    raw = load_config()
    provider_config = {
        "llm_api_keys": raw["llm_api_keys"],
        "llm_models": raw["llm_models"],
        "llm_client_settings": raw["llm_client_settings"],
        "llm_base_url": raw.get("llm_base_url") or "",
        "model_selection_strategy": raw["model_selection_strategy"],
        "model_capability_overrides": raw.get("model_capability_overrides") or {},
        "model_latency_overrides": raw.get("model_latency_overrides") or {},
    }
    provider = LLMProvider(config=provider_config, feedback_store=None)

    async def _discover() -> str:
        await provider.validate_api_keys(live_check=False)
        provider.build_model_registry()
        provider.build_task_fallbacks()
        return provider.get_primary_model(task_type)

    try:
        # amain() is already async — asyncio.run() fails inside a running loop.
        # Thread + fresh loop keeps discovery sync for callers in either context.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_discover()).strip()
        import concurrent.futures  # noqa: PLC0415

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(_discover())).result().strip()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"ERROR: harness model resolve failed task={task_type!r} "
            f"({type(exc).__name__}: {exc}). Set {HARNESS_GROUNDING_MODEL_ENV} "
            f"or ensure llm_api_keys env vars + LLM_BASE_URL discovery."
        ) from exc


def resolve_openai_compat_api_key_for_model(
    model: str,
    *,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Optional[str]]:
    """Pick OpenAI-compat API key for ``model`` from ``llm_api_keys`` env names.

    Gemini (``google``): ``GOOGLE_API_KEY`` if set, else ``OPENAI_API_KEY``.
    Other OpenAI-compat ids: ``OPENAI_API_KEY``.

    :return: Tuple[str, Optional[str]] - (api_key, base_url or None)
    :raises SystemExit: When no usable key is present
    """
    from llm_core.model_registry import _detect_provider  # noqa: PLC0415
    from semantic_search.config.loader import load_config  # noqa: PLC0415

    raw = config if config is not None else load_config()
    api_keys_cfg = raw["llm_api_keys"]
    base_url = (os.environ.get("LLM_BASE_URL") or str(raw.get("llm_base_url") or "")).strip() or None
    provider = _detect_provider(model)

    def _first_env(provider_name: str) -> str:
        entries = api_keys_cfg.get(provider_name) or []
        for entry in entries:
            env_name = str(entry.get("key_env_var") or "").strip()
            if not env_name:
                continue
            val = (os.environ.get(env_name) or "").strip()
            if val:
                return val
        return ""

    if provider == "google":
        key = _first_env("google") or _first_env("openai")
        if not key:
            raise SystemExit(
                "ERROR: grounding LLM unavailable "
                "(GOOGLE_API_KEY or OPENAI_API_KEY required for Gemini models)"
            )
        return key, base_url
    key = _first_env("openai")
    if not key:
        raise SystemExit("ERROR: grounding LLM unavailable (OPENAI_API_KEY?)")
    return key, base_url
