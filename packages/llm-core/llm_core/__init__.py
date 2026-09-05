"""llm_core — unified LLM provider, registry, ranker, and clients.

Public surface re-exported at package root for ergonomic imports.
"""
from llm_core.exceptions import LLMCoreError, ConfigurationError
from llm_core.logging_utils import get_logger
from llm_core.llm_client import LLMClient, AnthropicLLMClient, OpenAILLMClient, LLMClientFactory, LLMJudgeClientAdapter, PromptLearnerSyncAdapter
from llm_core.model_registry import ModelRegistry, _detect_provider, _is_chat_model
from llm_core.ranker import Ranker, RankWeights
from llm_core.signal_adapter import SignalAdapter, RuntimeStats
from llm_core.provider import LLMProvider

__version__ = "0.1.0"

__all__ = [
    "LLMCoreError",
    "ConfigurationError",
    "get_logger",
    "LLMClient",
    "AnthropicLLMClient",
    "OpenAILLMClient",
    "LLMClientFactory",
    "LLMJudgeClientAdapter",
    "PromptLearnerSyncAdapter",
    "ModelRegistry",
    "Ranker",
    "RankWeights",
    "SignalAdapter",
    "RuntimeStats",
    "LLMProvider",
    "_detect_provider",
    "_is_chat_model",
]
