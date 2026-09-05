"""Persist and load fitted per-tier temperature calibrators.

Fitting replays the golden-seed corpus through the L0 entity extractor, which
issues one LLM call group per case — hundreds of calls on every boot. This
module lets the service fit once (cold boot), write the resulting
``CalibrationFit`` per tier plus a fingerprint to disk, and on subsequent boots
load the fits directly (zero LLM) whenever the fingerprint still matches.

The fingerprint invalidates the cache whenever any input that could change the
fitted temperatures changes: the seed corpus bytes, the L0 entity prompt tag,
the fit parameters, or a manual ``cache_version`` bump.
"""
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from semantic_search.calibration.calibrator import CalibrationFit
from semantic_search.core.exceptions import ValidationError
from semantic_search.core.logging_utils import get_logger
from llm_core.logging_utils import mask_path

logger = get_logger(__name__)

_SCHEMA_VERSION = 1


def compute_fingerprint(seeds_path: str, prompt_tag: str, fit_params: Dict[str, Any], cache_version: int) -> str:
    """Derive a stable cache key from every input that affects the fitted T.

    :param seeds_path: str - Path to the golden-seed YAML (hashed by content)
    :param prompt_tag: str - L0 entity-extractor prompt tag (e.g. 'qi.entity.v3')
    :param fit_params: Dict[str, Any] - Deterministic fit bounds/params
    :param cache_version: int - Manual bump to force invalidation
    :return: str - Hex SHA256 digest
    """
    h = hashlib.sha256()
    h.update(f"schema={_SCHEMA_VERSION}".encode())
    h.update(f"prompt={prompt_tag}".encode())
    h.update(f"cache_version={int(cache_version)}".encode())
    for key in sorted(fit_params):
        h.update(f"{key}={fit_params[key]}".encode())
    try:
        h.update(Path(seeds_path).read_bytes())
    except OSError:
        h.update(b"__seeds_unreadable__")
    return h.hexdigest()


def save_fits(path: str, fits: Dict[str, CalibrationFit], fingerprint: str) -> None:
    """Write fits + fingerprint to disk atomically. Logs and swallows errors."""
    try:
        p = Path(path)
        if p.parent and not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": _SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "fits": [asdict(f) for f in fits.values()],
        }
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(tmp, p)
        logger.info(f"calibration_fits_saved path={mask_path(str(p))} tiers={len(fits)}")
    except (OSError, TypeError, ValueError) as e:
        logger.warning(f"calibration_fits_save_failed path={mask_path(str(path))} error_type={type(e).__name__} error={e}")


def load_fits(path: str, expected_fingerprint: str) -> Optional[Dict[str, CalibrationFit]]:
    """Load fits when the on-disk fingerprint matches; else return None.

    Returns None (cache miss) on: file absent, unreadable/corrupt JSON,
    fingerprint mismatch, or any fit failing ``CalibrationFit`` validation.
    A None return is never fatal — the caller falls back to a live fit.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"calibration_fits_load_failed reason=unreadable path={mask_path(str(p))} error_type={type(e).__name__} error={e}")
        return None
    if not isinstance(payload, dict) or payload.get("fingerprint") != expected_fingerprint:
        logger.info(f"calibration_fits_cache_miss path={mask_path(str(p))} reason=fingerprint_mismatch")
        return None
    out: Dict[str, CalibrationFit] = {}
    try:
        for row in payload.get("fits", []):
            if not isinstance(row, dict):
                raise ValidationError("calibration fit row must be a dict")
            fit = CalibrationFit(**row)
            out[fit.tier] = fit
    except (TypeError, ValidationError) as e:
        logger.warning(f"calibration_fits_load_failed reason=invalid_fit path={mask_path(str(p))} error_type={type(e).__name__} error={e}")
        return None
    if not out:
        return None
    logger.info(f"calibration_fits_loaded path={mask_path(str(p))} tiers={len(out)}")
    return out


__all__ = ["compute_fingerprint", "save_fits", "load_fits"]
