"""Standalone logging for llm_core. Stdlib only."""
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Union

_DEFAULT_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(funcName)s:%(lineno)d - %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"
_INITIALIZED_LOGGERS: set = set()
_WORKSPACE_ROOT_CACHE: Optional[Path] = None
_WORKSPACE_ROOT_RESOLVED: bool = False


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance.
    :param name: str - Logger name (typically __name__)
    :return: logging.Logger - Configured logger
    """
    if name in _INITIALIZED_LOGGERS:
        return logging.getLogger(name)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    _INITIALIZED_LOGGERS.add(name)
    return logger


_WORKSPACE_ROOT_MARKERS = (".git", ".cursor")


def _resolve_workspace_root() -> Optional[Path]:
    """Resolve workspace root once: env var > marker ancestor > package parent.

    Markers checked (any one is sufficient): ``.git``, ``.cursor``, ``pyproject.toml``,
    ``requirements.txt``. Falls back to ``llm_core``'s parent directory, which is by
    construction the workspace root in this monorepo.
    :return: Optional[Path] - Workspace root, or None if none of the strategies succeed.
    """
    global _WORKSPACE_ROOT_CACHE, _WORKSPACE_ROOT_RESOLVED
    if _WORKSPACE_ROOT_RESOLVED:
        return _WORKSPACE_ROOT_CACHE
    env_root = os.environ.get("WORKSPACE_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser()
        if candidate.is_dir():
            _WORKSPACE_ROOT_CACHE = candidate.resolve()
            _WORKSPACE_ROOT_RESOLVED = True
            return _WORKSPACE_ROOT_CACHE
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        if any((ancestor / marker).exists() for marker in _WORKSPACE_ROOT_MARKERS):
            _WORKSPACE_ROOT_CACHE = ancestor
            _WORKSPACE_ROOT_RESOLVED = True
            return _WORKSPACE_ROOT_CACHE
    package_parent = here.parent.parent
    if package_parent.is_dir():
        _WORKSPACE_ROOT_CACHE = package_parent
        _WORKSPACE_ROOT_RESOLVED = True
        return _WORKSPACE_ROOT_CACHE
    _WORKSPACE_ROOT_CACHE = None
    _WORKSPACE_ROOT_RESOLVED = True
    return None


def mask_path(value: Union[str, Path, None]) -> str:
    """Mask absolute filesystem paths for PII-safe logging.

    Resolution order:
      1. Workspace-root relative (e.g. ``llm_core/pricing.yaml``) when path is under root.
      2. ``~``-prefixed when path is under ``$HOME`` but outside workspace.
      3. Untouched string otherwise (already-relative, URLs, S3 URIs, etc.).

    Non-path inputs (None / empty) return ``"<none>"`` to keep log lines parseable.

    :param value: Union[str, Path, None] - Path-like value to mask.
    :return: str - PII-safe representation.
    """
    if value is None:
        return "<none>"
    text = str(value).strip()
    if not text:
        return "<none>"
    try:
        path = Path(text).expanduser()
    except (OSError, ValueError):
        return text
    if not path.is_absolute():
        return text
    try:
        resolved = path.resolve(strict=False)
    except (OSError, RuntimeError):
        resolved = path
    workspace_root = _resolve_workspace_root()
    if workspace_root is not None:
        try:
            return str(resolved.relative_to(workspace_root))
        except ValueError:
            pass
    home = Path(os.path.expanduser("~")).resolve() if os.path.expanduser("~") != "~" else None
    if home is not None:
        try:
            return f"~/{resolved.relative_to(home)}"
        except ValueError:
            pass
    return str(resolved)
