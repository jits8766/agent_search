"""Standalone logging for semantic_search package.
Uses stdlib logging only — no external dependencies.
"""
import sys
import logging

_DEFAULT_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(funcName)s:%(lineno)d - %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"
_INITIALIZED_LOGGERS: set = set()
# Config-independent by design (config/loader.py imports get_logger — importing config
# here would cycle). Stays at INFO until apply_log_level_from_config() is called once at
# startup, after config is loaded.
_CURRENT_LEVEL: int = logging.INFO


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance.
    :param name: str - Logger name (typically __name__)
    :return: logging.Logger - Configured logger
    """
    if name in _INITIALIZED_LOGGERS:
        return logging.getLogger(name)
    logger = logging.getLogger(name)
    logger.setLevel(_CURRENT_LEVEL)
    logger.propagate = False
    if not logger.handlers:
        formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(_CURRENT_LEVEL)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    _INITIALIZED_LOGGERS.add(name)
    return logger


def apply_log_level_from_config(level_name: str) -> None:
    """Re-level all loggers created via get_logger() to `level_name`.

    Call once at startup, after config is loaded (`config/loader.py` itself calls
    `get_logger()`, so this module must stay config-independent — the level can only be
    applied after the fact, not read inside `get_logger()`). Unknown level names fall
    back to INFO (logged once as a warning).
    :param level_name: str - DEBUG | INFO | WARNING | ERROR | CRITICAL (case-insensitive)
    """
    global _CURRENT_LEVEL
    resolved = logging.getLevelName(str(level_name).strip().upper())
    if not isinstance(resolved, int):
        get_logger(__name__).warning(
            f"log_level_invalid value={level_name!r} — falling back to INFO"
        )
        resolved = logging.INFO
    _CURRENT_LEVEL = resolved
    for name in _INITIALIZED_LOGGERS:
        logger = logging.getLogger(name)
        logger.setLevel(resolved)
        for handler in logger.handlers:
            handler.setLevel(resolved)


def quiet_llm_core_child_loggers_warning() -> None:
    """Set llm_core child loggers to WARNING to reduce startup console noise."""
    for suffix in ("provider", "model_registry", "ranker", "llm_client", "signal_adapter", "pricing"):
        logging.getLogger(f"llm_core.{suffix}").setLevel(logging.WARNING)
