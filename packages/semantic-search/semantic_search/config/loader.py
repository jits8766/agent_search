"""Configuration loader for the semantic_search package.
Loads the canonical YAML and validates required-key invariants via the typed
dataclasses in `semantic_search.config.models`. No silent defaults, no merging
from environment overrides at this stage — overrides are passed explicitly to
the typed dataclass `from_dict` constructors.

Environment variable expansion: any string value in the loaded YAML that
contains ``${VAR_NAME}`` tokens is resolved against ``os.environ`` after load.
This is the single pick-up point for portable model paths. Models are baked into
the image at ``/app/pretrained`` (CI syncs them from S3 into ``./pretrained`` before
``docker build``). ``base.yaml`` references them via ``${LOCAL_PRETRAINED_DIR:-/app/pretrained}``,
which resolves to the baked path; set ``LOCAL_PRETRAINED_DIR`` only to point at a different
location for local runs. Missing models => startup logs a degraded-model warning.
"""
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from semantic_search.core.exceptions import ConfigurationError
from semantic_search.core.logging_utils import get_logger
from llm_core.logging_utils import mask_path

logger = get_logger(__name__)

_BASE_CONFIG_PATH = Path(__file__).parent / "base.yaml"
_ENV_VAR_RE = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}')


def _expand_env_vars(obj: Any) -> Any:
    """Recursively expand ``${VAR}`` and ``${VAR:-default}`` tokens in all string leaves of ``obj``.

    When a default is supplied via ``:-``, it is returned when the env var is unset — no warning.
    Unresolved tokens without a default are left as-is and a WARNING is emitted.
    """
    if isinstance(obj, str):
        def _replace(m: re.Match) -> str:
            var = m.group(1)
            default = m.group(2)
            val = os.environ.get(var)
            if val is None:
                if default is not None:
                    return default
                logger.warning(f"config_env_var_unresolved token=${{{var}}} — set the environment variable before starting the service")
                return m.group(0)
            return val
        return _ENV_VAR_RE.sub(_replace, obj)
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    return obj


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from a YAML file.
    :param config_path: Optional[str] - Path to YAML file (None = load base.yaml)
    :return: Dict[str, Any] - Parsed configuration with ``${VAR}`` tokens expanded
    :raises ConfigurationError: If the file is missing, empty, or malformed
    """
    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            raise ConfigurationError(f"Config file not found: {config_path}")
    else:
        path = _BASE_CONFIG_PATH
        if not path.exists():
            raise ConfigurationError(f"Base config not found: {_BASE_CONFIG_PATH}")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML in {path}: {e}") from e
    if config is None:
        raise ConfigurationError(f"Empty config file: {path}")
    config = _expand_env_vars(config)
    logger.info(f"config_loaded path={mask_path(path)}")
    return config
