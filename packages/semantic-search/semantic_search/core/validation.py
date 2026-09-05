"""Input validation and sanitization for security-critical data boundaries.
Shared validators for deserialization and filesystem-component safety.
"""
import re
from typing import Any

from semantic_search.core.exceptions import ValidationError

__all__ = ['validate_identifier', 'sanitize_path_component', 'safe_int', 'safe_float']

_PATH_TRAVERSAL_PATTERN = re.compile(r'(?:^|[/\\])\.\.(?:[/\\]|$)')
_SAFE_IDENTIFIER_PATTERN = re.compile(r'^[a-zA-Z][a-zA-Z0-9_-]{0,127}$')


def validate_identifier(value: str, field_name: str) -> str:
    """Identifier validation: [a-zA-Z][a-zA-Z0-9_-]{0,127}."""
    if value is None or not isinstance(value, str) or len(value) == 0:
        raise ValidationError(f"{field_name} must be a non-empty string")
    if not _SAFE_IDENTIFIER_PATTERN.match(value):
        raise ValidationError(f"{field_name} contains invalid characters: must match [a-zA-Z][a-zA-Z0-9_-]{{0,127}}")
    return value


def sanitize_path_component(value: str, field_name: str) -> str:
    """Path component validation: reject traversal (..) and null bytes."""
    if value is None or not isinstance(value, str) or len(value) == 0:
        raise ValidationError(f"{field_name} must be a non-empty string")
    if _PATH_TRAVERSAL_PATTERN.search(value):
        raise ValidationError(f"{field_name} contains path traversal sequence")
    if '\x00' in value:
        raise ValidationError(f"{field_name} contains null byte")
    return value


def safe_int(value: Any, fallback: int) -> int:
    """Convert value to int (fallback on error)."""
    if value is None:
        return fallback
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return fallback


def safe_float(value: Any, fallback: float) -> float:
    """Convert value to float (fallback on error; symmetric with safe_int)."""
    if value is None:
        return fallback
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback
