"""Chip-label and card-title formatters for multi-intent chip-strip envelopes.

Pure-compute, deterministic, no I/O — safe to import from any layer.
"""
from typing import Any

from semantic_search.contracts import Entity, IntentSlice


def format_chip_label(name: str, value: Any) -> str:
    """Render an Entity (name, value) into the locked chip label shown on the strip.

    :param name: str - Entity slot name
    :param value: Any - Entity value (typed by name)
    :return: str - Display label
    """
    if name == 'tld' and isinstance(value, list):
        return ','.join(f".{v}" for v in value)
    if name == 'tldExcludeList':
        vals = value if isinstance(value, list) else [value]
        return 'exclude ' + ','.join(f".{v}" for v in vals if str(v).strip())
    if name == 'typeExcludeList':
        vals = value if isinstance(value, list) else [value]
        return 'exclude type=' + ','.join(str(v) for v in vals if str(v).strip())
    if name == 'keyword_contains_exclude':
        vals = value if isinstance(value, list) else [value]
        joined = ','.join(str(v) for v in vals if str(v).strip())
        return f"does not contain {joined}" if joined else "does not contain"
    if name == 'price_max':
        return f"under ${value}"
    if name == 'price_min':
        return f"over ${value}"
    if name == 'auction_type' and isinstance(value, list):
        return 'auction=' + ','.join(str(v) for v in value)
    if name == 'name_length_max':
        return f"<= {value} chars"
    if name == 'time_remaining_max':
        return f"ending within {_humanize_seconds(value)}"
    return f"{name}={value}"


def _humanize_seconds(value: Any) -> str:
    """Render a seconds count as a coarse human-readable interval."""
    try:
        seconds = int(value)
    except (TypeError, ValueError):
        return str(value)
    if seconds < 0:
        return str(seconds)
    if seconds < 3600:
        return f"{seconds}s"
    if seconds < 86400:
        hours = seconds // 3600
        return f"{hours}h"
    days = seconds // 86400
    return f"{days}d"


def build_card_title(slc: IntentSlice) -> str:
    """Render an IntentSlice into a plain-English card title.

    :param slc: IntentSlice - Slice to title
    :return: str - Plain-English title
    """
    if not slc.entities:
        return f"Show: {slc.raw_text or slc.query_type}"
    parts = [format_chip_label(e.name, e.value) for e in slc.entities[:4]]
    return "Show " + ' '.join(parts)


def chip_labels_for_slice(slc: IntentSlice) -> list:
    """Return the full ordered list of chip labels for a slice's entities.

    :param slc: IntentSlice - Slice whose entities to render
    :return: List[str] - One chip label per entity, in entity order
    """
    return [format_chip_label(e.name, e.value) for e in slc.entities]
