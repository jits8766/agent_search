"""Bridge: parse sql_hint key=value string → List[FilterSpec] for engine dispatch."""
from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from semantic_search.config.analytics_models import DomainAnalyticsConfig, FilterBridgeConfig

from semantic_search.analytics.query_templates import FilterSpec, InFilter, NumericRangeFilter


def sql_hint_to_filters(
    sql_hint: str,
    bridge_cfg: 'FilterBridgeConfig',
    domain_cfg: 'DomainAnalyticsConfig',
) -> List[FilterSpec]:
    """Parse ``sql_hint`` into concrete FilterSpec objects.

    Format: space-separated ``key=value`` pairs, e.g.
    ``price_min=500 tld=com,net traffic_max=10000``

    Range slots (``_min`` / ``_max`` suffix) pair into NumericRangeFilter.
    Comma-separated values become InFilter.
    Unrecognised or unmapped slots are silently skipped.
    """
    if not bridge_cfg.enabled or not sql_hint:
        return []

    pairs: dict[str, str] = {}
    for token in sql_hint.split():
        if '=' not in token:
            continue
        k, _, v = token.partition('=')
        pairs[k.strip()] = v.strip()

    slot_map = bridge_cfg.slot_to_column_attr
    min_sfx = bridge_cfg.min_suffix   # e.g. '_min'
    max_sfx = bridge_cfg.max_suffix   # e.g. '_max'

    filters: List[FilterSpec] = []
    consumed: set[str] = set()

    for slot, value in pairs.items():
        if slot in consumed:
            continue

        if slot.endswith(min_sfx):
            base = slot[: -len(min_sfx)]
            max_slot = base + max_sfx
            attr = slot_map.get(base) or slot_map.get(slot)
            if attr is None:
                continue
            col = getattr(domain_cfg, attr, None)
            if col is None:
                continue
            min_val = _to_number(value)
            max_val = _to_number(pairs.get(max_slot))
            if min_val is None and max_val is None:
                continue
            filters.append(NumericRangeFilter(column=col, min_value=min_val, max_value=max_val))
            consumed.add(slot)
            consumed.add(max_slot)

        elif slot.endswith(max_sfx):
            base = slot[: -len(max_sfx)]
            min_slot = base + min_sfx
            if min_slot in pairs:
                continue  # already handled by _min branch
            attr = slot_map.get(base) or slot_map.get(slot)
            if attr is None:
                continue
            col = getattr(domain_cfg, attr, None)
            if col is None:
                continue
            max_val = _to_number(value)
            if max_val is None:
                continue
            filters.append(NumericRangeFilter(column=col, max_value=max_val))
            consumed.add(slot)

        else:
            attr = slot_map.get(slot)
            if attr is None:
                continue
            col = getattr(domain_cfg, attr, None)
            if col is None:
                continue
            values = [v.strip() for v in value.split(',') if v.strip()]
            if not values:
                continue
            filters.append(InFilter(column=col, values=values))
            consumed.add(slot)

    return filters


def _to_number(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
