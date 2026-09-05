"""Shared keyword meta-token blocklist loaded from keyword_meta_blocklist.json.

Used by LLM entity reconcile and regex/LLM merge scrub so structural words
(numbers, hyphens, letters, …) never become keyword_contains(_exclude) values.
"""
from __future__ import annotations

import json
import os
from typing import FrozenSet

from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)

_BLOCKLIST_PATH = os.path.join(os.path.dirname(__file__), 'keyword_meta_blocklist.json')


def load_keyword_meta_blocklist() -> FrozenSet[str]:
    """Return lowercase meta-token blocklist from config JSON; empty frozenset on load failure."""
    try:
        with open(_BLOCKLIST_PATH, encoding='utf-8') as fh:
            raw = json.load(fh)
        return frozenset(str(t).lower() for t in (raw.get('blocklist') or []) if str(t).strip())
    except (OSError, json.JSONDecodeError, TypeError, AttributeError) as exc:
        logger.warning(
            f"keyword_meta_blocklist_load_failed path={_BLOCKLIST_PATH!r} "
            f"error_type={type(exc).__name__} error={exc}"
        )
        return frozenset()


KEYWORD_META_BLOCKLIST: FrozenSet[str] = load_keyword_meta_blocklist()
