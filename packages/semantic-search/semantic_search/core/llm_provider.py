"""LLMProvider shim for semantic_search.
Re-exports the unified `llm_core.LLMProvider` and translates `llm_core.ConfigurationError`
into `semantic_search.ConfigurationError` so consumers see a consistent exception hierarchy.
"""
from typing import Any, Dict, Optional

from llm_core import LLMProvider as _CoreLLMProvider, ConfigurationError as _CoreConfigurationError

from semantic_search.core.exceptions import ConfigurationError
from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)


class LLMProvider(_CoreLLMProvider):
    """Thin subclass that translates llm_core.ConfigurationError into semantic_search.ConfigurationError."""

    def __init__(self, config: Dict[str, Any], feedback_store: Optional[Any] = None):
        """Initialize provider — same contract as llm_core.LLMProvider.
        :param config: Dict[str, Any] - Full service config (must include llm_* + model_selection_strategy)
        :param feedback_store: Optional[Any] - FeedbackStore-like signal source (or None)
        :raises ConfigurationError: If required config sections or keys are missing
        """
        try:
            super().__init__(config=config, feedback_store=feedback_store)
        except _CoreConfigurationError as e:
            raise ConfigurationError(str(e)) from e
