"""Search identity: request_id (trace) vs search_id (durable business join).

``request_id`` — ephemeral per HTTP hop (observability / retries).
``search_id`` — durable id for one search interaction; feedback and analysis join on it.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Mapping, Optional, Pattern

from semantic_search.config.subsystem_bundle import IdentityConfig
from semantic_search.core.exceptions import ValidationError


@dataclass(frozen=True)
class ResolvedIdentity:
    """Resolved pair for one search or feedback boundary call."""

    request_id: str
    search_id: str
    request_id_from_client: bool
    search_id_from_client: bool


def mint_prefixed_id(prefix: str, hex_length: int) -> str:
    """Mint ``{prefix}_{hex}`` from config prefix and hex length."""
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValidationError("identity mint prefix must be a non-empty string")
    if not isinstance(hex_length, int) or isinstance(hex_length, bool) or hex_length < 1:
        raise ValidationError("identity mint hex_length must be int >= 1")
    return f"{prefix.strip()}_{uuid.uuid4().hex[:hex_length]}"


def normalize_client_id(
    raw: Optional[str],
    *,
    max_id_chars: int,
    field_name: str,
    value_pattern: Pattern[str],
) -> Optional[str]:
    """Return stripped client id or None when absent/blank.

    Rejects overlong values, control characters, and values that fail
    ``identity.id_value_pattern`` (log-safe / storage-safe client ids).
    """
    if raw is None or not isinstance(raw, str):
        return None
    value = raw.strip()
    if not value:
        return None
    if len(value) > int(max_id_chars):
        raise ValidationError(
            f"{field_name} exceeds identity.max_id_chars={int(max_id_chars)}"
        )
    if any(ord(ch) < 32 for ch in value):
        raise ValidationError(f"{field_name} must not contain control characters")
    if value_pattern.fullmatch(value) is None:
        raise ValidationError(
            f"{field_name} must match identity.id_value_pattern"
        )
    return value


def _header_get(headers: Mapping[str, str], name: str) -> Optional[str]:
    """Case-insensitive header lookup for Starlette/FastAPI header maps."""
    if not name:
        return None
    target = name.lower()
    for key, value in headers.items():
        if str(key).lower() == target:
            return value if isinstance(value, str) else str(value)
    return None


def resolve_identity(
    cfg: IdentityConfig,
    *,
    headers: Optional[Mapping[str, str]] = None,
    form_request_id: Optional[str] = None,
    form_search_id: Optional[str] = None,
) -> ResolvedIdentity:
    """Resolve request_id + search_id from form, headers, or server mint.

    Precedence per id: non-empty form field, then header (when accept_client_*), then mint.
    """
    header_map = headers or {}
    value_pattern = re.compile(cfg.id_value_pattern)

    client_rid = normalize_client_id(
        form_request_id,
        max_id_chars=cfg.max_id_chars,
        field_name="request_id",
        value_pattern=value_pattern,
    )
    if client_rid is None and cfg.accept_client_request_id:
        client_rid = normalize_client_id(
            _header_get(header_map, cfg.request_id_header),
            max_id_chars=cfg.max_id_chars,
            field_name="request_id",
            value_pattern=value_pattern,
        )

    client_sid = normalize_client_id(
        form_search_id,
        max_id_chars=cfg.max_id_chars,
        field_name="search_id",
        value_pattern=value_pattern,
    )
    if client_sid is None and cfg.accept_client_search_id:
        client_sid = normalize_client_id(
            _header_get(header_map, cfg.search_id_header),
            max_id_chars=cfg.max_id_chars,
            field_name="search_id",
            value_pattern=value_pattern,
        )

    rid_from_client = client_rid is not None
    sid_from_client = client_sid is not None
    request_id = client_rid or mint_prefixed_id(cfg.request_id_prefix, cfg.id_hex_length)
    search_id = client_sid or mint_prefixed_id(cfg.search_id_prefix, cfg.id_hex_length)
    return ResolvedIdentity(
        request_id=request_id,
        search_id=search_id,
        request_id_from_client=rid_from_client,
        search_id_from_client=sid_from_client,
    )


def resolve_feedback_search_id(
    cfg: IdentityConfig,
    *,
    form_search_id: Optional[str],
) -> tuple[str, bool]:
    """Resolve durable search_id for feedback.

    Returns ``(search_id, correlated)`` where correlated means the client supplied it.
    Mode ``required`` raises ValidationError when missing; ``soft_generate`` mints.
    """
    client_sid = normalize_client_id(
        form_search_id,
        max_id_chars=cfg.max_id_chars,
        field_name="search_id",
        value_pattern=re.compile(cfg.id_value_pattern),
    )
    if client_sid is not None:
        return client_sid, True
    mode = cfg.feedback_search_id_mode
    if mode == "required":
        raise ValidationError("search_id is required on feedback")
    if mode != "soft_generate":
        raise ValidationError(
            f"identity.feedback_search_id_mode unsupported value={mode!r}"
        )
    return mint_prefixed_id(cfg.search_id_prefix, cfg.id_hex_length), False


__all__ = [
    "ResolvedIdentity",
    "mint_prefixed_id",
    "normalize_client_id",
    "resolve_identity",
    "resolve_feedback_search_id",
]
