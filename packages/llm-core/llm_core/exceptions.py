"""Exception hierarchy for llm_core."""


class LLMCoreError(Exception):
    """Base exception for llm_core errors."""
    pass


class ConfigurationError(LLMCoreError):
    """Raised when required configuration is missing or invalid."""
    pass
