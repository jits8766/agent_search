"""Unified LLM client implementations: Anthropic, OpenAI, factory, and adapters.

Single source of truth for all packages.
"""
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from llm_core.logging_utils import get_logger

try:
    from anthropic import Anthropic as _Anthropic
except ImportError:
    _Anthropic = None  # type: ignore[assignment,misc]

try:
    from openai import OpenAI as _OpenAI
except ImportError:
    _OpenAI = None  # type: ignore[assignment,misc]

logger = get_logger(__name__)


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    async def call(self, system_prompt: str, user_prompt: str, max_tokens: int, model: Optional[str] = None, temperature: Optional[float] = None) -> Tuple[Optional[str], Dict[str, Any]]:
        """Call LLM with prompts.
        :param system_prompt: str - System prompt
        :param user_prompt: str - User prompt
        :param max_tokens: int - Max response tokens
        :param model: Optional[str] - Model override (uses instance model if None)
        :param temperature: Optional[float] - Temperature override
        :return: Tuple[Optional[str], Dict[str, Any]] - (Response, token_usage)
        """
        pass

    @abstractmethod
    async def call_with_messages(self, messages: list, max_tokens: int, model: Optional[str] = None, temperature: Optional[float] = None, response_format: Optional[Dict[str, str]] = None) -> Tuple[Optional[str], Dict[str, Any]]:
        """Call LLM with custom message array.
        :param messages: list - Message array with role and content
        :param max_tokens: int - Max response tokens
        :param model: Optional[str] - Model override
        :param temperature: Optional[float] - Temperature override
        :param response_format: Optional[Dict[str, str]] - Response format (OpenAI only)
        :return: Tuple[Optional[str], Dict[str, Any]] - (Response, token_usage)
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get current model name.
        :return: str - Model name
        """
        pass


def _is_temperature_error(exc: Exception) -> bool:
    """Return True when the provider rejected the request specifically because of the temperature parameter.

    Catches both Anthropic ("temperature is deprecated for this model") and OpenAI
    ("temperature is not supported", "unsupported parameter: temperature") error shapes
    so the retry logic in both clients stays provider-agnostic.
    """
    msg = str(exc).lower()
    return 'temperature' in msg and any(w in msg for w in ('deprecated', 'not supported', 'unsupported', 'invalid parameter'))


class AnthropicLLMClient(LLMClient):
    """Anthropic Claude LLM client."""

    def __init__(self, api_key: str, model: str, temperature: float, timeout: int, max_retries: int = 3, base_url: Optional[str] = None, no_temperature_models: frozenset = frozenset(), use_prompt_caching: bool = False):
        """Initialize Anthropic client.
        :param api_key: str - Anthropic API key
        :param model: str - Model name
        :param temperature: float - Generation temperature
        :param timeout: int - Timeout seconds
        :param max_retries: int - Max retries
        :param base_url: Optional[str] - API base URL
        :param no_temperature_models: frozenset - Pre-seeded set of model names that must not receive the ``temperature`` parameter.
        :param use_prompt_caching: bool - When True, wraps the system prompt in a cache_control block so Anthropic caches static prompt tokens across calls.
        """
        if _Anthropic is None:
            raise ImportError("anthropic package is required for AnthropicLLMClient")
        self.model = model
        self.temperature = temperature
        self._no_temperature_models: set = set(no_temperature_models)
        self._use_prompt_caching: bool = bool(use_prompt_caching)
        kwargs: Dict[str, Any] = {'api_key': api_key, 'timeout': timeout, 'max_retries': max_retries}
        if base_url:
            kwargs['base_url'] = base_url
        self.client = _Anthropic(**kwargs)

    async def call(self, system_prompt: str, user_prompt: str, max_tokens: int, model: Optional[str] = None, temperature: Optional[float] = None) -> Tuple[Optional[str], Dict[str, Any]]:
        try:
            model_name = model or self.model
            temp = temperature if temperature is not None else self.temperature
            messages = [{"role": "user", "content": user_prompt}]
            system_value: Any = system_prompt
            if self._use_prompt_caching and isinstance(system_prompt, str) and system_prompt:
                system_value = [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]
            kwargs: Dict[str, Any] = dict(model=model_name, system=system_value, messages=messages, max_tokens=max_tokens)
            if model_name not in self._no_temperature_models:
                kwargs['temperature'] = temp
            try:
                response = await asyncio.to_thread(self.client.messages.create, **kwargs)
            except Exception as first_exc:
                if 'temperature' in kwargs and _is_temperature_error(first_exc):
                    logger.warning(f"anthropic_temperature_unsupported model={model_name} retrying_without_temperature=True")
                    self._no_temperature_models.add(model_name)
                    kwargs.pop('temperature')
                    response = await asyncio.to_thread(self.client.messages.create, **kwargs)
                else:
                    raise
            content = response.content[0].text if response.content else None
            token_usage = self._token_usage_from_response(response, model_name)
            return content.strip() if content else None, token_usage
        except Exception as e:
            logger.error(f"anthropic_call_failed model={model or self.model} error_type={type(e).__name__}")
            raise

    async def call_with_messages(self, messages: list, max_tokens: int, model: Optional[str] = None, temperature: Optional[float] = None, response_format: Optional[Dict[str, str]] = None) -> Tuple[Optional[str], Dict[str, Any]]:
        try:
            model_name = model or self.model
            temp = temperature if temperature is not None else self.temperature
            system_content = next((m['content'] for m in messages if m['role'] == 'system'), '')
            user_messages = [{'role': m['role'], 'content': m['content']} for m in messages if m['role'] != 'system']
            system_value: Any = system_content
            if self._use_prompt_caching and isinstance(system_content, str) and system_content:
                system_value = [{"type": "text", "text": system_content, "cache_control": {"type": "ephemeral"}}]
            kwargs: Dict[str, Any] = dict(model=model_name, system=system_value, messages=user_messages, max_tokens=max_tokens)
            if model_name not in self._no_temperature_models:
                kwargs['temperature'] = temp
            try:
                response = await asyncio.to_thread(self.client.messages.create, **kwargs)
            except Exception as first_exc:
                if 'temperature' in kwargs and _is_temperature_error(first_exc):
                    logger.warning(f"anthropic_temperature_unsupported model={model_name} retrying_without_temperature=True")
                    self._no_temperature_models.add(model_name)
                    kwargs.pop('temperature')
                    response = await asyncio.to_thread(self.client.messages.create, **kwargs)
                else:
                    raise
            content = response.content[0].text if response.content else None
            token_usage = self._token_usage_from_response(response, model_name)
            return content.strip() if content else None, token_usage
        except Exception as e:
            logger.error(f"anthropic_call_with_messages_failed model={model or self.model} error_type={type(e).__name__}")
            raise

    @staticmethod
    def _token_usage_from_response(response: Any, model_name: str) -> Dict[str, Any]:
        """Extract a uniform token-usage dict from an Anthropic response.

        Wave 6 / A14 — we surface ``cached_input_tokens`` (Anthropic
        ``cache_read_input_tokens``) so ``LLMCallRouter`` can compute the
        prompt-cache hit-rate proxy signal directly from provider metadata
        rather than inferring it from cost. ``prompt_tokens`` stays the full
        input-token count; ``cached_input_tokens`` is the subset that came
        from the cache (always 0 ≤ cached ≤ prompt).
        """
        usage = response.usage
        cached = int(getattr(usage, 'cache_read_input_tokens', 0) or 0)
        return {
            'prompt_tokens': int(usage.input_tokens),
            'completion_tokens': int(usage.output_tokens),
            'total_tokens': int(usage.input_tokens) + int(usage.output_tokens),
            'cached_input_tokens': cached,
            'model': model_name,
        }

    def get_model_name(self) -> str:
        return self.model


class OpenAILLMClient(LLMClient):
    """OpenAI GPT LLM client."""

    def __init__(self, api_key: str, model: str, temperature: float, timeout: int, max_retries: int = 3, base_url: Optional[str] = None, no_temperature_models: frozenset = frozenset()):
        """Initialize OpenAI client.
        :param api_key: str - OpenAI API key
        :param model: str - Model name
        :param temperature: float - Generation temperature
        :param timeout: int - Timeout seconds
        :param max_retries: int - Max retries
        :param base_url: Optional[str] - API base URL
        :param no_temperature_models: frozenset - Pre-seeded set of model names that must not receive ``temperature``.
        """
        if _OpenAI is None:
            raise ImportError("openai package is required for OpenAILLMClient")
        self.model = model
        self.temperature = temperature
        self._no_temperature_models: set = set(no_temperature_models)
        kwargs: Dict[str, Any] = {'api_key': api_key, 'timeout': timeout, 'max_retries': max_retries}
        if base_url:
            kwargs['base_url'] = base_url
        self.client = _OpenAI(**kwargs)

    async def call(self, system_prompt: str, user_prompt: str, max_tokens: int, model: Optional[str] = None, temperature: Optional[float] = None) -> Tuple[Optional[str], Dict[str, Any]]:
        try:
            model_name = model or self.model
            temp = temperature if temperature is not None else self.temperature
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            call_kwargs: Dict[str, Any] = dict(model=model_name, messages=messages, max_tokens=max_tokens)
            if model_name not in self._no_temperature_models:
                call_kwargs['temperature'] = temp
            try:
                response = await asyncio.to_thread(self.client.chat.completions.create, **call_kwargs)
            except Exception as first_exc:
                if 'temperature' in call_kwargs and _is_temperature_error(first_exc):
                    logger.warning(f"openai_temperature_unsupported model={model_name} retrying_without_temperature=True")
                    self._no_temperature_models.add(model_name)
                    call_kwargs.pop('temperature')
                    response = await asyncio.to_thread(self.client.chat.completions.create, **call_kwargs)
                else:
                    raise
            content = response.choices[0].message.content if response.choices else None
            token_usage = self._token_usage_from_response(response, model_name)
            return content.strip() if content else None, token_usage
        except Exception as e:
            logger.error(f"openai_call_failed model={model or self.model} error_type={type(e).__name__} error={str(e)}")
            raise

    async def call_with_messages(self, messages: list, max_tokens: int, model: Optional[str] = None, temperature: Optional[float] = None, response_format: Optional[Dict[str, str]] = None) -> Tuple[Optional[str], Dict[str, Any]]:
        try:
            model_name = model or self.model
            temp = temperature if temperature is not None else self.temperature
            call_kwargs: Dict[str, Any] = {'model': model_name, 'messages': messages, 'max_tokens': max_tokens}
            if model_name not in self._no_temperature_models:
                call_kwargs['temperature'] = temp
            if response_format:
                call_kwargs['response_format'] = response_format
            try:
                response = await asyncio.to_thread(self.client.chat.completions.create, **call_kwargs)
            except Exception as first_exc:
                if 'temperature' in call_kwargs and _is_temperature_error(first_exc):
                    logger.warning(f"openai_temperature_unsupported model={model_name} retrying_without_temperature=True")
                    self._no_temperature_models.add(model_name)
                    call_kwargs.pop('temperature')
                    response = await asyncio.to_thread(self.client.chat.completions.create, **call_kwargs)
                else:
                    raise
            content = response.choices[0].message.content if response.choices else None
            token_usage = self._token_usage_from_response(response, model_name)
            return content.strip() if content else None, token_usage
        except Exception as e:
            logger.error(f"openai_call_with_messages_failed model={model or self.model} error_type={type(e).__name__} error={str(e)}")
            raise

    @staticmethod
    def _token_usage_from_response(response: Any, model_name: str) -> Dict[str, Any]:
        """Extract a uniform token-usage dict from an OpenAI response.

        Wave 6 / A14 — we surface ``cached_input_tokens`` (OpenAI
        ``response.usage.prompt_tokens_details.cached_tokens``) so the router
        can plot prompt-cache hit-rate from provider metadata directly.
        ``prompt_tokens`` stays the full input-token count; ``cached_input_tokens``
        is the subset that hit the cache (always 0 ≤ cached ≤ prompt).
        """
        if not hasattr(response, 'usage') or not response.usage:
            return {}
        usage = response.usage
        cached = 0
        details = getattr(usage, 'prompt_tokens_details', None)
        if details is not None:
            cached = int(getattr(details, 'cached_tokens', 0) or 0)
        return {
            'prompt_tokens': int(usage.prompt_tokens),
            'completion_tokens': int(usage.completion_tokens),
            'total_tokens': int(usage.total_tokens),
            'cached_input_tokens': cached,
            'model': model_name,
        }

    def get_model_name(self) -> str:
        return self.model


class LLMClientFactory:
    """Factory for creating LLM clients based on model name prefix."""

    @staticmethod
    def create(model: str, api_key: str, temperature: float, timeout: int, max_retries: int, base_url: Optional[str] = None, no_temperature_models: frozenset = frozenset()) -> LLMClient:
        """Create LLM client for the given model.
        :param model: str - Model name (prefix determines provider)
        :param api_key: str - API key for the provider
        :param temperature: float - Generation temperature
        :param timeout: int - Timeout seconds
        :param max_retries: int - Max retries
        :param base_url: Optional[str] - API base URL
        :param no_temperature_models: frozenset - Model names that must not receive ``temperature``.
            Passed through to the concrete client. See ``AnthropicLLMClient`` / ``OpenAILLMClient``.
        :return: LLMClient - Concrete client instance
        """
        if not api_key:
            raise ValueError(f"API key is required for model: {model}")
        if model.startswith('claude'):
            return AnthropicLLMClient(api_key=api_key, model=model, temperature=temperature, timeout=timeout, max_retries=max_retries, base_url=base_url, no_temperature_models=no_temperature_models)
        elif model.startswith(('gpt', 'o1', 'o3', 'o4')):
            return OpenAILLMClient(api_key=api_key, model=model, temperature=temperature, timeout=timeout, max_retries=max_retries, base_url=base_url, no_temperature_models=no_temperature_models)
        raise ValueError(f"Unsupported model prefix: {model}")


class LLMJudgeClientAdapter:
    """Adapts LLMClient async call() to consumers expecting async call returning str (e.g. healing LLMJudge)."""

    def __init__(self, client: LLMClient):
        self._client = client

    async def call(self, system_prompt: str, user_prompt: str, max_tokens: int, model: Optional[str] = None, temperature: Optional[float] = None) -> str:
        model_use = model if model is not None and str(model).strip() != '' else None
        text, _ = await self._client.call(system_prompt, user_prompt, max_tokens, model_use, temperature)
        return text if text else ''


class PromptLearnerSyncAdapter:
    """Adapts LLMClient to PromptLearner protocol (sync generate) via asyncio.run in thread context."""

    def __init__(self, client: LLMClient, max_tokens: int, temperature: float):
        self._client = client
        self._max_tokens = max_tokens
        self._temperature = temperature

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        async def _run() -> str:
            text, _ = await self._client.call(system_prompt, user_prompt, self._max_tokens, None, self._temperature)
            return text if text else ''
        return asyncio.run(_run())
