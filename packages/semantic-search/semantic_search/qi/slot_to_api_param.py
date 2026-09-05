"""Maps extracted entity slot names to FIND API param names with value transforms.

Loaded once at import time from entity_slot_to_api_param.json (sibling file) and
find_api_params.json (repo root).

Public API:
  to_api_filters(entity_list, raw_query) -> Dict[str, Any]
  get_api_param_for_slot(slot)           -> Optional[Dict[str, Any]]
  is_find_api_filter_slot(slot)          -> bool
  get_find_api_param_name(slot)          -> Optional[str]
  FIND_FILTERABLE_API_PARAMS             -> FrozenSet[str] (63 FIND filters)

Transforms applied per the mapping:
  direct         — value passed as-is
  pass_through   — keyword_contains -> query (only used when raw_query is not provided)
  list_to_csv    — list -> comma-joined string (tld -> tldIncludeList)
  invert_bool    — negated bool (has_hyphen=False -> excludeHyphens=True)
  bool_to_str    — bool -> "true"/"false" string (is_idn -> isIdn)
  seconds_to_iso — int seconds -> ISO 8601 UTC timestamp (time_remaining_max -> endTimeBefore)
  prefix_query   — keyword_starts_with -> query (only used when raw_query is not provided)
  suffix_query   — keyword_ends_with -> query (only used when raw_query is not provided)
  no_api_param   — entry skipped (TLF slots have no auction/recommend param)

When raw_query is supplied, it is used verbatim as the `query` API param and all
keyword-slot aggregation (pass_through / prefix_query / suffix_query) is skipped.
This prevents TLD or price tokens extracted by QI from leaking into the query param
when the same token also appears as a structured filter (e.g. tld=io -> tldIncludeList).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional

from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)

_MAPPING_PATH = os.path.join(os.path.dirname(__file__), "entity_slot_to_api_param.json")

# Slot -> {api_param, transform} lookup built once at import time.
_SLOT_MAP: Dict[str, Dict[str, str]] = {}

# Local-only slots (no_api_param transform): sentinel dicts keyed by slot name.
# Non-None so get_api_param_for_slot() returns a truthy value, allowing these
# slots to pass the `api_param is not None` gate in _build_filter_summary.
_NO_API_PARAM_SLOTS: Dict[str, Dict[str, Any]] = {}

# Filterable FIND API catalog — every slot that maps to a real auction/recommend
# param (no_api_param slots, e.g. TLF, excluded). Order preserved from the JSON so
# response consumers get a stable slot ordering. Each entry: {slot, api_param, type}.
FIND_API_FILTER_CATALOG: List[Dict[str, Any]] = []

try:
    with open(_MAPPING_PATH, encoding="utf-8") as _f:
        _raw = json.load(_f)
    for _entry in _raw.get("mappings", []):
        _slot = _entry.get("entity_slot")
        _param = _entry.get("api_param")
        _transform = _entry.get("transform", "direct")
        if _slot and _param and _transform != "no_api_param":
            _SLOT_MAP[_slot] = {"api_param": _param, "transform": _transform}
            FIND_API_FILTER_CATALOG.append({"slot": _slot, "api_param": _param, "type": _entry.get("type")})
        elif _slot and _transform == "no_api_param":
            _NO_API_PARAM_SLOTS[_slot] = {"name": _slot, "transform": "no_api_param"}
except (OSError, json.JSONDecodeError, KeyError) as _e:
    logger.warning(f"slot_to_api_param_mapping_load_failed path={_MAPPING_PATH!r} error_type={type(_e).__name__} error={_e} — to_api_filters returns empty dict")

# ── FIND API param metadata ──────────────────────────────────────────────────
# Loaded from find_api_params.json (repo root). Keyed by API param name -> subset
# of param metadata (name, type, format, description) — the fields the UX needs for
# input rendering and tooltips. Built once at import; silently empty when absent.
_API_PARAM_LOOKUP: Dict[str, Dict[str, Any]] = {}
_FIND_API_PARAMS_PATH: Optional[Path] = None
try:
    _here = Path(__file__).resolve()
    _candidates = [_here.parents[i] / "find_api_params.json" for i in range(len(_here.parents))]
    _FIND_API_PARAMS_PATH = next((p for p in _candidates if p.exists()), None)
    if _FIND_API_PARAMS_PATH is None:
        raise OSError("find_api_params.json not found in any parent directory")
    with open(_FIND_API_PARAMS_PATH, encoding="utf-8") as _af:
        for _ap in json.load(_af):
            _pname = _ap.get("name")
            if _pname:
                _API_PARAM_LOOKUP[_pname] = {
                    "name": _pname,
                    "type": _ap.get("type"),
                    "format": _ap.get("format"),
                    "description": _ap.get("description"),
                }
except (OSError, json.JSONDecodeError, KeyError, IndexError) as _e:
    logger.warning(f"slot_to_api_param_find_params_load_failed path={str(_FIND_API_PARAMS_PATH)!r} error_type={type(_e).__name__} error={_e} — api param enrichment skipped")

# Slot -> API param metadata: pre-built so get_api_param_for_slot() is O(1).
# Maps each entity_slot to the downstream API param metadata (or None when the
# slot has no API param — e.g. TLF slots with no_api_param transform).
_SLOT_TO_API_META: Dict[str, Optional[Dict[str, Any]]] = {
    slot: _API_PARAM_LOOKUP.get(info["api_param"])
    for slot, info in _SLOT_MAP.items()
}
# Merge local-only slots so get_api_param_for_slot() returns a non-None sentinel
# for no_api_param entries — required for the `api_param is not None` gate in
# _build_filter_summary to pass these slots into applied_filters.
_SLOT_TO_API_META.update(_NO_API_PARAM_SLOTS)

# FIND filterable API param names (find_api_params.json ∩ slot mapping).
# Excludes meta (query/pagination/sort/…) and no_api_param / unmapped local hard.
# Single source of truth for qie_only L0 filter extract, full-search FIND63 LLM
# group, and offline FIND-63 regex fallback — do not re-list these elsewhere.
FIND_FILTERABLE_API_PARAMS: FrozenSet[str] = frozenset(
    info["api_param"]
    for info in _SLOT_MAP.values()
    if info.get("api_param") in _API_PARAM_LOOKUP
)

# Stable catalog order from find_api_params.json load order (prompt / allowlist CSV).
FIND_FILTERABLE_API_PARAMS_ORDERED: List[str] = [
    name for name in _API_PARAM_LOOKUP if name in FIND_FILTERABLE_API_PARAMS
]


def get_api_param_for_slot(slot: str) -> Optional[Dict[str, Any]]:
    """Return API param metadata for slot, or None if unmapped."""
    return _SLOT_TO_API_META.get(slot)


def is_find_api_filter_slot(slot: str) -> bool:
    """True when ``slot`` maps to a FIND filterable auction/recommend API param."""
    info = _SLOT_MAP.get(slot)
    if info is None:
        return False
    return info.get("api_param") in FIND_FILTERABLE_API_PARAMS


def get_find_api_param_name(slot: str) -> Optional[str]:
    """Return FIND api_param name for slot, or None when not FIND-filterable."""
    info = _SLOT_MAP.get(slot)
    if info is None:
        return None
    api_param = info.get("api_param")
    if api_param not in FIND_FILTERABLE_API_PARAMS:
        return None
    return str(api_param)


def get_slot_transform(slot: str) -> Optional[str]:
    """Return transform name for slot from entity_slot_to_api_param.json, or None if unmapped."""
    info = _SLOT_MAP.get(slot)
    if info is not None:
        return info.get("transform")
    no_api = _NO_API_PARAM_SLOTS.get(slot)
    if no_api is not None:
        return str(no_api.get("transform") or "no_api_param")
    return None


# Transforms where a boolean False entity value is still an active filter constraint
# (e.g. has_number=False -> excludeDigits=True). Derived from mapping JSON — not hardcoded slots.
_FALSE_ACTIVE_TRANSFORMS: FrozenSet[str] = frozenset({"invert_bool", "bool_to_str"})
# Slots where False is an active user intent even if transform lookup is stale/missing.
_FALSE_ACTIVE_SLOTS: FrozenSet[str] = frozenset({"has_reserve_price"})

# Temporal transforms / FIND API params — any new mapping that uses these is
# treated as a time window for CH→hybrid degrade (no hand-maintained slot list).
_TEMPORAL_TRANSFORMS: FrozenSet[str] = frozenset({"seconds_to_iso", "days_ago_to_iso"})
_TEMPORAL_API_PARAMS: FrozenSet[str] = frozenset({
    "endTimeAfter", "endTimeBefore", "startTimeAfter", "startTimeBefore",
})


def is_temporal_entity_slot(slot: str) -> bool:
    """True when ``slot`` is a listing/auction time-window filter.

    Driven by ``entity_slot_to_api_param.json`` (temporal transforms + time API
    params). Unknown slots matching relative-time name prefixes are included so
    local-only ages still strip on CH→hybrid degrade.
    """
    if not slot or not isinstance(slot, str):
        return False
    info = _SLOT_MAP.get(slot)
    if info is not None:
        if info.get("transform") in _TEMPORAL_TRANSFORMS:
            return True
        if info.get("api_param") in _TEMPORAL_API_PARAMS:
            return True
    # Unmapped / no_api_param relative ages still used as hard chips in retrieval.
    return slot.startswith(("days_listed_", "time_remaining"))


def slot_keeps_false_value(slot: str) -> bool:
    """True when False is a meaningful active value for this slot (config-driven via transform)."""
    if slot in _FALSE_ACTIVE_SLOTS:
        return True
    transform = get_slot_transform(slot)
    return transform in _FALSE_ACTIVE_TRANSFORMS if transform else False


# Prefer more specific FIND labels when multiple names share an ID set
# (auction/expiry both map to {16,38}).
_AUCTION_ID_SET_LABEL_PREFERENCE: tuple = (
    'premium', 'partner', 'godaddy', 'closeout', 'buynow', 'buy_now',
    'backorder', 'dropcatch', 'drop_catch', 'firehose', 'preregistration',
    'pre_registration', 'auction', 'expiry',
)


def _split_type_include_tokens(value: Any) -> List[str]:
    """Split pipe/comma/list auction type values into lowercase tokens."""
    if isinstance(value, list):
        raw = [str(v).strip().lower() for v in value if v not in ('', None)]
    elif isinstance(value, str):
        sep = '|' if '|' in value else ','
        raw = [p.strip().lower() for p in value.split(sep) if p.strip()]
    else:
        sv = str(value).strip().lower()
        return [sv] if sv else []
    return raw


def normalize_type_include_list_for_public(value: Any) -> Any:
    """Normalize public ``typeIncludeList`` for identified_filters.

    - Mixed ID+label: drop IDs already covered by a label (``25|backorder`` → ``backorder``).
    - Orphan numeric IDs stay numeric (``16`` → ``16``), never widen to a multi-ID label.
    - Digit-only sets that exactly equal one label's ID set collapse to that label
      (grounded ``premium`` → ``16|38|39`` → ``premium``).
    - Drop redundant generics (``auction|expiry`` → ``expiry``).
    """
    from semantic_search.contracts import AUCTION_TYPE_LABEL_TO_IDS, prefer_registrar_auction_values

    parts = _split_type_include_tokens(value)
    if not parts:
        return value

    labels: List[str] = []
    orphan_ids: List[str] = []
    for p in parts:
        if p.isdigit():
            orphan_ids.append(p)
        else:
            labels.append(p)

    covered_ids: set = set()
    for lab in labels:
        covered_ids |= {str(n).lower() for n in AUCTION_TYPE_LABEL_TO_IDS.get(lab, frozenset())}

    # Keep orphan IDs as IDs (do not map 16 → premium).
    for did in orphan_ids:
        if did in covered_ids:
            continue
        labels.append(did)
        covered_ids.add(did)

    # Grounded label→ID expand reverse: exact ID-set match → single FIND label.
    digit_only = [x for x in labels if str(x).isdigit()]
    non_digit = [x for x in labels if not str(x).isdigit()]
    if digit_only and not non_digit:
        id_set = frozenset(str(x) for x in digit_only)
        exact = [
            lab for lab, nids in AUCTION_TYPE_LABEL_TO_IDS.items()
            if frozenset(str(n).lower() for n in nids) == id_set
        ]
        if len(exact) == 1:
            labels = list(exact)
        elif len(exact) > 1:
            picked = None
            for preferred in _AUCTION_ID_SET_LABEL_PREFERENCE:
                if preferred in exact:
                    picked = preferred
                    break
            labels = [picked] if picked is not None else digit_only

    norm = prefer_registrar_auction_values(list(dict.fromkeys(labels)))
    if 'auction' in norm and 'expiry' in norm:
        norm = [x for x in norm if x != 'auction']
    if not norm:
        return value
    if len(norm) == 1:
        return norm[0]
    return '|'.join(sorted(norm))


def collapse_auction_ids_to_find_label(value: Any) -> Any:
    """Normalize auction type values for public ``typeIncludeList`` (IDs kept when orphan)."""
    return normalize_type_include_list_for_public(value)


def transform_slot_value(slot: str, value: Any) -> Any:
    """Apply the configured slot transform to value; return value unchanged when unmapped/no-op."""
    info = _SLOT_MAP.get(slot)
    if info is None:
        return value
    transformed = _apply_transform(value, info["transform"])
    out = value if transformed is None else transformed
    if slot == 'auction_type':
        return collapse_auction_ids_to_find_label(out)
    return out


# Keyword slots all map to the same `query` param — collect and merge them.
_QUERY_SLOT_TRANSFORMS = frozenset({"pass_through", "prefix_query", "suffix_query"})


def _apply_transform(value: Any, transform: str) -> Optional[Any]:
    """Transform entity value per specified rule (direct, list_to_csv, bool_to_str, etc)."""
    if transform in ("direct", "pass_through"):
        if isinstance(value, list):
            return ",".join(str(v) for v in value)
        return value
    if transform in ("prefix_query", "suffix_query"):
        if isinstance(value, list):
            return "|".join(str(v) for v in value)
        return value
    if transform == "list_to_csv":
        if isinstance(value, list):
            return ",".join(str(v) for v in value)
        return str(value)
    if transform == "invert_bool":
        return not bool(value)
    if transform == "bool_to_str":
        return "true" if value else "false"
    if transform == "seconds_to_iso":
        try:
            deadline = datetime.now(timezone.utc) + timedelta(seconds=int(value))
            return deadline.strftime("%Y-%m-%dT%H:%M:%SZ")
        except (TypeError, ValueError, OverflowError):
            return None
    if transform == "days_ago_to_iso":
        # N days ago -> ISO startTimeAfter timestamp
        try:
            start = datetime.now(timezone.utc) - timedelta(days=int(value))
            return start.strftime("%Y-%m-%dT%H:%M:%SZ")
        except (TypeError, ValueError, OverflowError):
            return None
    return None


def to_api_filters(entity_list: List[Dict[str, Any]], raw_query: Optional[str] = None) -> Dict[str, Any]:
    """Map entity slots to API params with transforms; use raw_query verbatim to prevent duplication."""
    if not _SLOT_MAP:
        return {}

    result: Dict[str, Any] = {}
    query_parts: List[str] = []
    use_raw_query = bool(raw_query and raw_query.strip())

    for ent in entity_list:
        name = ent.get("name")
        value = ent.get("value")
        if not name or value is None:
            continue
        mapping = _SLOT_MAP.get(name)
        if not mapping:
            continue

        transform = mapping["transform"]
        api_param = mapping["api_param"]

        if transform in _QUERY_SLOT_TRANSFORMS:
            if use_raw_query:
                # raw_query takes precedence; skip keyword slots to avoid duplication
                continue
            part = str(value) if not isinstance(value, list) else " ".join(str(v) for v in value)
            if part.strip():
                query_parts.append(part.strip())
            continue

        transformed = _apply_transform(value, transform)
        if transformed is None:
            continue
        result[api_param] = transformed

    if use_raw_query:
        result["query"] = raw_query.strip()  # type: ignore[union-attr]
    elif query_parts:
        result["query"] = " ".join(query_parts)

    return result
