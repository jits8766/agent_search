"""Load YAML golden seed cases for calibration boot (``fit_on_load``).

Schema matches ``semantic_search/config/golden_seeds.yaml`` top-level ``cases:``
list. Each case must include ``input_query`` and ``expected_query_type``;
other fields are ignored by the calibration fit drivers.
"""
import os
from typing import Any, Dict, List

import yaml
from llm_core.logging_utils import mask_path

from semantic_search.core.exceptions import ConfigurationError
from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)


def load_calibration_boot_cases(path: str) -> List[Dict[str, Any]]:
    """Load calibration seed cases from a YAML file under the workspace root.
    :param path: str - Relative or absolute path to YAML with ``cases:`` list
    :return: list[dict] - Dicts with at least ``input_query`` and ``expected_query_type``
    :raises ConfigurationError: When path escapes workspace, file missing, or schema invalid
    """
    if not isinstance(path, str) or not path.strip():
        raise ConfigurationError("calibration.boot_seeds_path must be a non-empty string")
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    stripped = os.path.expanduser(path.strip())
    if os.path.isabs(stripped):
        expanded = os.path.normpath(stripped)
    else:
        expanded = os.path.normpath(os.path.join(workspace_root, stripped))
    if not expanded.startswith(workspace_root + os.sep) and expanded != workspace_root:
        raise ConfigurationError(f"calibration.boot_seeds_path must be inside the workspace; got {path}")
    if not os.path.isfile(expanded):
        raise ConfigurationError(f"calibration.boot_seeds_path not found at {mask_path(expanded)}")
    with open(expanded, 'r', encoding='utf-8') as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ConfigurationError("calibration boot seeds YAML root must be a mapping")
    cases = raw.get('cases')
    if not isinstance(cases, list):
        raise ConfigurationError("calibration boot seeds YAML must contain a list key 'cases'")
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(cases):
        if not isinstance(row, dict):
            raise ConfigurationError(f"calibration boot seeds cases[{i}] must be a mapping")
        iq = row.get('input_query')
        eqt = row.get('expected_query_type')
        if not isinstance(iq, str) or not iq.strip():
            raise ConfigurationError(f"calibration boot seeds cases[{i}].input_query must be a non-empty string")
        if not isinstance(eqt, str) or not eqt.strip():
            raise ConfigurationError(f"calibration boot seeds cases[{i}].expected_query_type must be a non-empty string")
        out.append({'input_query': iq, 'expected_query_type': eqt})
    logger.info(f"calibration_boot_seeds_loaded path={mask_path(expanded)} cases={len(out)}")
    return out
