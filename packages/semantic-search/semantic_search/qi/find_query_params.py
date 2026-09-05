"""Build FIND auction/recommend wire params from qie_only filters and keywords.

Produces FIND-ready ``find_query_params`` / ``find_query_string`` for
``GET /v4/aftermarket/find/auction/recommend?<find_query_string>``.

``identified_filters`` remain chip-friendly (labels, relative times). The wire
view applies:

- hard FIND-filterable params only (soft/local chips excluded)
- relative ``endTime*`` / ``startTime*`` to absolute UTC ISO
- ``typeIncludeList`` / ``typeExcludeList`` labels to numeric IDs, comma-joined
- keyword terms to FIND ``query`` per ``FindWireConfig``
- FIND semantic-search query param when keywords become ``query`` and config enables it
- all values stringified for query-string encoding
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlencode

from semantic_search.config.models import FindWireConfig
from semantic_search.contracts import AUCTION_TYPE_LABEL_TO_IDS
from semantic_search.qi.slot_to_api_param import FIND_FILTERABLE_API_PARAMS

# FIND ProcessAuctionRecReq ISOLayout = "2006-01-02T15:04:05Z"
_FIND_ISO_LAYOUT = "%Y-%m-%dT%H:%M:%SZ"

_RELATIVE_OFFSET_RE = re.compile(r"^-(\d+)([dh])$", re.IGNORECASE)

_TIME_PARAMS = frozenset(
    {
        "endTimeBefore",
        "endTimeAfter",
        "startTimeBefore",
        "startTimeAfter",
    }
)

_TYPE_LIST_PARAMS = frozenset({"typeIncludeList", "typeExcludeList"})

_LIST_CSV_PARAMS = frozenset(
    {
        "tldIncludeList",
        "tldExcludeList",
        "typeIncludeList",
        "typeExcludeList",
        "ownerMemberIncludeList",
        "ownerMemberExcludeList",
    }
)


def _split_tokens(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, bool):
        return ["true" if value else "false"]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and value.is_integer():
            return [str(int(value))]
        return [str(value)]
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            out.extend(_split_tokens(item))
        return out
    text = str(value).strip()
    if not text:
        return []
    if "|" in text:
        return [p.strip() for p in text.split("|") if p.strip()]
    if "," in text:
        return [p.strip() for p in text.split(",") if p.strip()]
    return [text]


def _normalize_time_value(param: str, value: Any, *, now: datetime) -> Tuple[Optional[str], Optional[str]]:
    """Return (iso_string, skip_reason). skip_reason set when unusable."""
    tokens = _split_tokens(value)
    if not tokens:
        return None, "empty_time"
    raw = tokens[0]
    # Already FIND ISO?
    try:
        datetime.strptime(raw, _FIND_ISO_LAYOUT)
        return raw, None
    except ValueError:
        pass
    # Accept ISO with timezone Z / offset, re-emit FIND layout.
    if raw.endswith("Z") and "T" in raw:
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return parsed.astimezone(timezone.utc).strftime(_FIND_ISO_LAYOUT), None
        except ValueError:
            pass

    match = _RELATIVE_OFFSET_RE.match(raw)
    if match is None:
        return None, "unparseable_time"

    amount = int(match.group(1))
    unit = match.group(2).lower()
    if amount <= 0:
        return None, "non_positive_relative_time"
    delta = timedelta(days=amount) if unit == "d" else timedelta(hours=amount)

    # Before + relative = horizon ahead of now (ending soon).
    # After + relative = lookback behind now (listed recently).
    if param.endswith("Before"):
        instant = now + delta
    else:
        instant = now - delta
    return instant.strftime(_FIND_ISO_LAYOUT), None


def _expand_type_list(value: Any) -> Tuple[Optional[str], Optional[str]]:
    """Labels/IDs -> comma-separated numeric IDs for FIND ES Atoi path.

    Unknown labels are ignored; if nothing expands, return skip reason.
    """
    tokens = _split_tokens(value)
    if not tokens:
        return None, "empty_type_list"
    ids: List[str] = []
    seen: set = set()
    unknown: List[str] = []
    for tok in tokens:
        key = tok.lower()
        if tok.isdigit():
            if tok not in seen:
                seen.add(tok)
                ids.append(tok)
            continue
        mapped = AUCTION_TYPE_LABEL_TO_IDS.get(key)
        if mapped is None:
            unknown.append(tok)
            continue
        for aid in sorted(mapped, key=lambda x: int(x)):
            if aid not in seen:
                seen.add(aid)
                ids.append(aid)
    if not ids:
        reason = (
            f"unknown_type_label:{','.join(unknown)}"
            if unknown
            else "empty_type_ids"
        )
        return None, reason
    return ",".join(ids), None


def _stringify_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _normalize_listish(param: str, value: Any) -> str:
    tokens = _split_tokens(value)
    if param in _LIST_CSV_PARAMS:
        return ",".join(tokens)
    if len(tokens) == 1:
        return tokens[0]
    return ",".join(tokens)


def _select_keyword_query(
    keywords: Optional[Sequence[Dict[str, Any]]],
    wire: FindWireConfig,
    *,
    min_keyword_probability: float,
) -> Optional[str]:
    """Select keyword terms by probability and cap; return joined FIND query or None.

    ``min_keyword_probability`` is a fraction from
    ``qi.l0_llm_entity.keyword_min_probability`` (percent/100).
    """
    if not 0.0 <= float(min_keyword_probability) <= 1.0:
        raise ValueError(
            f"min_keyword_probability must be in [0, 1], got {min_keyword_probability!r}"
        )
    if not keywords:
        return None
    scored: List[Tuple[float, str]] = []
    for item in keywords:
        if not isinstance(item, dict):
            continue
        term = str(item.get("term") or "").strip()
        if not term:
            continue
        raw_prob = item.get("probability")
        try:
            prob = float(raw_prob) if raw_prob is not None else float("nan")
        except (TypeError, ValueError):
            continue
        if prob != prob:  # NaN
            continue
        if prob < float(min_keyword_probability):
            continue
        scored.append((prob, term))
    if not scored:
        return None
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    terms = [term for _, term in scored[: wire.max_keyword_terms]]
    if not terms:
        return None
    return wire.keyword_term_separator.join(terms)


def build_find_wire_payload(
    identified: List[Dict[str, Any]],
    *,
    wire: FindWireConfig,
    keywords: Optional[Sequence[Dict[str, Any]]] = None,
    min_keyword_probability: float,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Map qie_only identified filters and keywords to FIND-ready wire fields.

    :param identified: chip-friendly identified_filters
    :param wire: FindWireConfig from ``general.search.find_wire``
    :param keywords: L0 keyword terms with probability
    :param min_keyword_probability: fraction gate from
        ``qi.l0_llm_entity.keyword_min_probability`` (percent/100)
    :param now: clock for relative time normalization
    :return: dict with find_query_params, find_query_string, soft_chips, find_skipped
    """
    if not isinstance(wire, FindWireConfig):
        raise TypeError("build_find_wire_payload requires FindWireConfig wire=")

    clock = now if now is not None else datetime.now(timezone.utc)
    if clock.tzinfo is None:
        clock = clock.replace(tzinfo=timezone.utc)
    else:
        clock = clock.astimezone(timezone.utc)

    params: Dict[str, str] = {}
    soft_chips: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []

    for item in identified or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        value = item.get("value")
        chip_kind = str(item.get("chip_kind") or "hard").strip().lower()

        # soft_chips = chip_kind soft only (rank/theme signals). Hard locals that
        # FIND cannot apply (e.g. keyword_contains_exclude) stay out of soft_chips
        # so qie_only clients do not misread a hard exclude as a soft hint —
        # they remain on identified with find_skipped reason not_find_filterable.
        if chip_kind == "soft":
            soft_chips.append(item)
            continue
        if name not in FIND_FILTERABLE_API_PARAMS:
            skipped.append({"name": name, "reason": "not_find_filterable"})
            continue

        if value is None:
            skipped.append({"name": name, "reason": "null_value"})
            continue

        if name in _TIME_PARAMS:
            iso, reason = _normalize_time_value(name, value, now=clock)
            if reason is not None or iso is None:
                skipped.append({"name": name, "reason": reason or "time_normalize_failed"})
                continue
            params[name] = iso
            continue

        if name in _TYPE_LIST_PARAMS:
            csv_ids, reason = _expand_type_list(value)
            if reason is not None or csv_ids is None:
                skipped.append({"name": name, "reason": reason or "type_expand_failed"})
                continue
            params[name] = csv_ids
            continue

        if isinstance(value, bool):
            params[name] = _stringify_scalar(value)
            continue

        if name in _LIST_CSV_PARAMS or isinstance(value, list) or (
            isinstance(value, str) and ("|" in value or "," in value)
        ):
            text = _normalize_listish(name, value)
            if not text:
                skipped.append({"name": name, "reason": "empty_list"})
                continue
            params[name] = text
            continue

        params[name] = _stringify_scalar(value)

    kw_query = _select_keyword_query(
        keywords, wire, min_keyword_probability=min_keyword_probability
    )
    hard_filter_present = any(k != "query" for k in params)

    if "query" not in params:
        if wire.prefer_keywords_for_query:
            params["query"] = kw_query if kw_query else wire.empty_query_fallback
        elif hard_filter_present:
            params["query"] = wire.empty_query_fallback
        else:
            params["query"] = kw_query if kw_query else wire.empty_query_fallback

    keywords_drove_query = bool(kw_query) and params.get("query") == kw_query
    if keywords_drove_query and wire.set_use_semantic_search_when_keywords:
        params[wire.use_semantic_search_param] = wire.use_semantic_search_value

    # Stable key order: query first, then catalog order, then leftovers.
    ordered_keys = ["query"] + [
        k for k in sorted(params) if k != "query"
    ]
    ordered = {k: params[k] for k in ordered_keys if k in params}
    query_string = urlencode(ordered, doseq=True)

    return {
        "find_query_params": ordered,
        "find_query_string": query_string,
        "soft_chips": soft_chips,
        "find_skipped": skipped,
    }


__all__ = ["build_find_wire_payload"]
