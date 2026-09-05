"""Background LLM expansion loop for the query-driven dynamic synonym store.

Reads miss tokens from ``DynamicSynonymStore.drain_miss_queue``, calls the
LLM for each batch and writes
results back via ``DynamicSynonymStore.put``. Runs as a long-lived asyncio
task — one iteration per ``poll_interval_seconds``.

Layer placement: ``vectorization/`` because this module imports ``llm_core``.
The ``retrieval/`` layer stays stdlib-only; this module bridges upward.

Usage:
    expander = QueryDrivenExpander(store, config)
    task = asyncio.create_task(expander.run_forever())
    # on shutdown:
    expander.stop()
    await task

Wire-up in ``app.py`` lifespan:
    if cfg.retrieval.qdrant.hybrid.bm25_query_encoder.dynamic_synonyms is not None:
        store = bm25_encoder.dynamic_store
        expander = QueryDrivenExpander(store, full_config)
        asyncio.create_task(expander.run_forever())
"""
import asyncio
import json
from typing import Dict, List, Optional, Tuple

from llm_core import LLMProvider, _detect_provider
from llm_core.llm_client import LLMClient

from semantic_search.core.exceptions import ValidationError
from semantic_search.core.logging_utils import get_logger
from semantic_search.retrieval.dynamic_synonym_store import DynamicSynonymStore

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a domain-name marketplace expert. "
    "Given a token extracted from a registered domain name, "
    "produce 2-3 closely related terms that a buyer would search for. "
    "Return ONLY a JSON object: {\"synonyms\": [\"term1\", \"term2\"]}. "
    "No markdown fences, no explanation."
)
_USER_PROMPT_TEMPLATE = "Token: {token}\nDomain: domain-name marketplace context.\nGive 2-3 search synonyms."

_TASK_NAME = "synonym_generation"
_MAX_TOKENS_PER_CALL = 128


async def _expand_token(token: str, llm_client: LLMClient, temperature: float) -> Optional[List[str]]:
    """Call LLM once to expand a single token into synonyms.

    :param token: str - Token to expand
    :param llm_client: LLMClient - Configured LLM client
    :param temperature: float - Sampling temperature
    :return: Optional[List[str]] - Synonym list, or None on failure
    """
    user_prompt = _USER_PROMPT_TEMPLATE.format(token=token)
    try:
        text, _ = await llm_client.call(system_prompt=_SYSTEM_PROMPT, user_prompt=user_prompt, max_tokens=_MAX_TOKENS_PER_CALL, temperature=temperature)
        if not text:
            return None
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        parsed = json.loads(cleaned)
        synonyms = parsed.get("synonyms", [])
        if isinstance(synonyms, list) and all(isinstance(s, str) for s in synonyms):
            return [s.lower().strip() for s in synonyms if s.strip()]
        return None
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning(f"query_driven_expand_failed token={token} error_type={type(exc).__name__}")
        return None
    except Exception as exc:
        logger.error(f"query_driven_expand_error token={token} error_type={type(exc).__name__}")
        return None


async def _expand_batch(tokens: List[str], llm_client: LLMClient, temperature: float, concurrency: int) -> Dict[str, List[str]]:
    """Expand a batch of tokens with bounded concurrency.

    :param tokens: List[str] - Tokens to expand
    :param llm_client: LLMClient - Configured LLM client
    :param temperature: float - Sampling temperature
    :param concurrency: int - Maximum parallel LLM calls
    :return: Dict[str, List[str]] - token → synonyms map for successful expansions
    """
    semaphore = asyncio.Semaphore(concurrency)
    results: Dict[str, List[str]] = {}

    async def _limited(tok: str) -> None:
        async with semaphore:
            syns = await _expand_token(tok, llm_client, temperature)
            if syns:
                results[tok] = syns

    await asyncio.gather(*[_limited(t) for t in tokens], return_exceptions=True)
    return results


class QueryDrivenExpander:
    """Background loop that drains the miss queue and writes LLM-generated synonyms to the store.

    :param store: DynamicSynonymStore - The target synonym store
    :param full_config: Dict - Full application config dict (requires llm_api_keys, llm_models,
        model_selection_strategy to build an LLM client)
    :param poll_interval_seconds: int - Seconds to sleep between drain cycles (from config)
    :param batch_size: int - Tokens to drain and expand per cycle (from config)
    :raises ValidationError: When store or config are malformed
    """

    def __init__(self, store: DynamicSynonymStore, full_config: Dict, poll_interval_seconds: int, batch_size: int):
        if store is None or not isinstance(store, DynamicSynonymStore):
            raise ValidationError("QueryDrivenExpander requires a DynamicSynonymStore")
        if not isinstance(full_config, dict):
            raise ValidationError("QueryDrivenExpander requires a dict config")
        if not isinstance(poll_interval_seconds, int) or poll_interval_seconds < 1:
            raise ValidationError("QueryDrivenExpander poll_interval_seconds must be int >= 1")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValidationError("QueryDrivenExpander batch_size must be int >= 1")
        self._store = store
        self._full_config = full_config
        self._poll_interval = poll_interval_seconds
        self._batch_size = batch_size
        self._stop_event = asyncio.Event()
        self._llm_client: Optional[LLMClient] = None
        self._temperature: float = 0.0

    async def _init_llm_client(self) -> None:
        """Build the LLM client from config. Called once on first run.

        :raises RuntimeError: When no suitable model is available for synonym_generation
        """
        provider = LLMProvider(self._full_config, feedback_store=None)
        await provider.validate_api_keys(live_check=False)
        provider.build_model_registry()
        candidates = provider.select_models_for_task(_TASK_NAME)
        if not candidates:
            raise RuntimeError(f"query_driven_expander: no models for task={_TASK_NAME}")
        selected = candidates[0]
        logger.info(f"query_driven_expander_llm model={selected} provider={_detect_provider(selected)}")
        llm_models_cfg = self._full_config["llm_models"]
        self._temperature = float(llm_models_cfg["temperature"])
        self._llm_client = provider.get_client_for_model(selected)

    async def _run_one_cycle(self) -> int:
        """Drain one batch from the miss queue, expand, and write back.

        :return: int - Number of tokens expanded this cycle
        """
        tokens = self._store.drain_miss_queue(self._batch_size)
        if not tokens:
            return 0
        if self._llm_client is None:
            await self._init_llm_client()
        dyn_cfg = self._store._config
        results = await _expand_batch(tokens, self._llm_client, self._temperature, int(dyn_cfg.expansion_concurrency))
        for token, synonyms in results.items():
            self._store.put(token, synonyms)
        expanded = len(results)
        if expanded:
            logger.info(f"query_driven_expand_cycle expanded={expanded} remaining_queue={self._store.queue_size}")
        return expanded

    def stop(self) -> None:
        """Signal the run loop to exit after the current cycle."""
        self._stop_event.set()

    async def run_forever(self) -> None:
        """Run expansion cycles until ``stop()`` is called.

        Soft-fail: any unhandled exception in a cycle is logged and the loop
        continues after the next poll interval so a transient LLM error does
        not crash the background task.
        """
        logger.info(f"query_driven_expander_started poll_interval={self._poll_interval}s batch_size={self._batch_size}")
        while not self._stop_event.is_set():
            try:
                await self._run_one_cycle()
            except Exception as exc:
                logger.error(f"query_driven_expander_cycle_error error_type={type(exc).__name__} error={exc}")
            try:
                await asyncio.wait_for(asyncio.shield(asyncio.ensure_future(self._stop_event.wait())), timeout=float(self._poll_interval))
            except asyncio.TimeoutError:
                pass
        logger.info("query_driven_expander_stopped")
