"""Config-driven entity reconcile helpers (entity_reconcile_rules.json).

Used by L0 merge scrub (engine + regex fallback packaging). Deterministic cue →
slot fixes for gaps regex must not own and LLM miss cases.
"""
from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import Any, Dict, Iterator, List, Optional, Pattern, Tuple

from semantic_search.contracts import Entity
from semantic_search.core.exceptions import ConfigurationError
from semantic_search.core.logging_utils import get_logger
from semantic_search.qi.advisory_patterns import (
    is_soft_advisory_no_inventory,
    is_strong_advisory,
)
from semantic_search.qi.budget_price_cues import parse_budget_prefixed_price
from semantic_search.qi.llm_classifier import infer_chip_kind

logger = get_logger(__name__)

# Set by apply_post_merge_reconcile / hard_entity_names_context from qi.entity_slots.
_HARD_ENTITY_NAMES_CTX: ContextVar[Optional[frozenset]] = ContextVar(
    'entity_reconcile_hard_names', default=None
)


@contextmanager
def hard_entity_names_context(hard_entity_names: frozenset) -> Iterator[None]:
    """Bind qi.entity_slots.hard_entity_names for reconcile helpers that call ``_entity``.

    Required for unit tests that invoke individual ``reconcile_*`` functions outside
    ``apply_post_merge_reconcile``. Production path always goes through apply_post_merge.
    """
    if not isinstance(hard_entity_names, frozenset):
        raise ConfigurationError(
            "hard_entity_names_context requires hard_entity_names frozenset from qi.entity_slots"
        )
    if not hard_entity_names:
        raise ConfigurationError("hard_entity_names_context requires non-empty hard_entity_names")
    token = _HARD_ENTITY_NAMES_CTX.set(hard_entity_names)
    try:
        yield
    finally:
        _HARD_ENTITY_NAMES_CTX.reset(token)

_RULES_PATH = os.path.join(os.path.dirname(__file__), 'entity_reconcile_rules.json')
_CHAR_RULES_PATH = os.path.join(os.path.dirname(__file__), 'char_constraint_rules.json')


def _load_rules() -> Dict[str, Any]:
    try:
        with open(_RULES_PATH, encoding='utf-8') as fh:
            raw = json.load(fh)
        return raw if isinstance(raw, dict) else {}
    except (OSError, json.JSONDecodeError, TypeError, AttributeError) as exc:
        logger.warning(
            f"entity_reconcile_rules_load_failed path={_RULES_PATH!r} "
            f"error_type={type(exc).__name__} error={exc}"
        )
        return {}


def _load_char_rules() -> Tuple[Tuple[str, Pattern[str], Any], ...]:
    """Load char_constraint_rules.json → (slot, compiled_re, value) tuples."""
    try:
        with open(_CHAR_RULES_PATH, encoding='utf-8') as fh:
            raw = json.load(fh)
        rules = raw.get('rules') if isinstance(raw, dict) else None
        out: List[Tuple[str, Pattern[str], Any]] = []
        for rule in rules or []:
            if not isinstance(rule, dict):
                continue
            slot = rule.get('slot')
            pat = rule.get('pattern')
            if not slot or not pat:
                continue
            out.append((str(slot), re.compile(str(pat), re.IGNORECASE), rule.get('value')))
        return tuple(out)
    except (OSError, json.JSONDecodeError, TypeError, AttributeError) as exc:
        logger.warning(
            f"char_constraint_rules_load_failed path={_CHAR_RULES_PATH!r} "
            f"error_type={type(exc).__name__} error={exc}"
        )
        return tuple()


_RULES: Dict[str, Any] = _load_rules()
_CHAR_RULES: Tuple[Tuple[str, Pattern[str], Any], ...] = _load_char_rules()

KEYWORD_MATCH_MODE_STANDALONE = frozenset(
    str(v).lower() for v in (_RULES.get('keyword_match_mode_standalone_values') or []) if str(v).strip()
)

_EXACT_MODE_RES: Tuple[Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (_RULES.get('keyword_match_mode_exact_patterns') or [])
    if isinstance(p, str) and p.strip()
)

_LIFECYCLE_DISJ_RES: Tuple[Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE | re.DOTALL)
    for p in (_RULES.get('lifecycle_disjunction_patterns') or [])
    if isinstance(p, str) and p.strip()
)

_STALE_SPECS: Tuple[Dict[str, Any], ...] = tuple(
    s for s in (_RULES.get('stale_listing_patterns') or []) if isinstance(s, dict) and s.get('pattern')
)
_STALE_COMPILED: Tuple[Tuple[Pattern[str], Dict[str, Any]], ...] = tuple(
    (re.compile(str(s['pattern']), re.IGNORECASE), s) for s in _STALE_SPECS
)

_METRIC_OVERRIDES: Tuple[Tuple[Pattern[str], Dict[str, str]], ...] = tuple(
    (
        re.compile(str(o['query_cue']), re.IGNORECASE),
        {str(k): str(v) for k, v in dict(o.get('remap') or {}).items()},
    )
    for o in (_RULES.get('metric_family_overrides') or [])
    if isinstance(o, dict) and o.get('query_cue') and o.get('remap')
)

_METRIC_INJECT: Tuple[Tuple[Pattern[str], Pattern[str], str], ...] = tuple(
    (
        re.compile(str(o['query_cue']), re.IGNORECASE),
        re.compile(str(o['pattern']), re.IGNORECASE),
        str(o['slot']),
    )
    for o in (_RULES.get('metric_bound_inject') or [])
    if isinstance(o, dict) and o.get('query_cue') and o.get('pattern') and o.get('slot')
)

def _build_metric_lexicon(
    raw: Any,
) -> Tuple[Tuple[Dict[str, Any], ...], Optional[Pattern[str]], Dict[str, Dict[str, Any]]]:
    """Build shared-grammar metric lexicon from config.

    Returns (entries, alias_alternation_re, alias_lower→entry). Aliases are regex
    fragments; grammar (between/to/under/above/soft) lives in code once.
    """
    entries: List[Dict[str, Any]] = []
    alias_to_entry: Dict[str, Dict[str, Any]] = {}
    parts: List[str] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        aliases = [str(a) for a in (item.get('aliases') or []) if str(a).strip()]
        if not aliases or not (item.get('min_slot') or item.get('max_slot')):
            continue
        entry = dict(item)
        entry['aliases'] = aliases
        entries.append(entry)
        for a in aliases:
            parts.append(f'(?:{a})')
            # Map bare text form (strip regex escapes loosely) for match→entry lookup
            # via the captured metric span lowercased against each alias search.
    if not parts:
        return tuple(), None, {}
    # Longest-first so "trust flow" wins over "tf" when both could apply in alt.
    parts_sorted = sorted(parts, key=len, reverse=True)
    alt = '|'.join(parts_sorted)
    try:
        metric_re = re.compile(rf'(?P<metric>\b(?:{alt})\b)', re.IGNORECASE)
    except re.error as exc:
        logger.warning(f"metric_lexicon_compile_failed error={exc}")
        return tuple(), None, {}
    return tuple(entries), metric_re, alias_to_entry


_METRIC_LEXICON, _METRIC_ALIAS_RE, _ = _build_metric_lexicon(_RULES.get('metric_lexicon'))


def _compile_qualitative_floor_scrub(raw: Any) -> Tuple[Dict[str, Any], ...]:
    """Compile qualitative_floor_scrub families from entity_reconcile_rules.json.

    Modes:
      - qual_without_numeric: drop slots when any qual_cue hits and no numeric_cue hits
      - ungrounded: drop slots when none of cues hit (e.g. invented currency)
    Extend JSON to cover new metric families — no code change.
    """
    out: List[Dict[str, Any]] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        mode = str(item.get('mode') or '').strip()
        drop_slots = frozenset(
            str(s) for s in (item.get('drop_slots') or []) if str(s).strip()
        )
        if not mode or not drop_slots:
            continue
        fam_id = str(item.get('id') or mode)
        if mode == 'qual_without_numeric':
            qual = tuple(
                re.compile(str(p), re.IGNORECASE)
                for p in (item.get('qual_cues') or [])
                if isinstance(p, str) and p.strip()
            )
            numeric = tuple(
                re.compile(str(p), re.IGNORECASE)
                for p in (item.get('numeric_cues') or [])
                if isinstance(p, str) and p.strip()
            )
            if not qual:
                logger.warning(
                    f"qualitative_floor_scrub_skip id={fam_id!r} reason=empty_qual_cues"
                )
                continue
            out.append({
                'id': fam_id,
                'mode': mode,
                'qual': qual,
                'numeric': numeric,
                'drop_slots': drop_slots,
            })
        elif mode == 'ungrounded':
            cues = tuple(
                re.compile(str(p), re.IGNORECASE)
                for p in (item.get('cues') or [])
                if isinstance(p, str) and p.strip()
            )
            if not cues:
                logger.warning(
                    f"qualitative_floor_scrub_skip id={fam_id!r} reason=empty_cues"
                )
                continue
            out.append({
                'id': fam_id,
                'mode': mode,
                'cues': cues,
                'drop_slots': drop_slots,
            })
        else:
            logger.warning(
                f"qualitative_floor_scrub_skip id={fam_id!r} reason=unknown_mode mode={mode!r}"
            )
    return tuple(out)


def _compile_calendar_window_guard(raw: Any) -> Dict[str, Tuple[Pattern[str], ...]]:
    """Compile calendar_window_guard — skip bare lookback when popularity speech."""
    if not isinstance(raw, dict):
        return {'skip': tuple(), 'except': tuple()}
    skip = tuple(
        re.compile(str(p), re.IGNORECASE)
        for p in (raw.get('skip_bare_window_if_any') or [])
        if isinstance(p, str) and p.strip()
    )
    except_ = tuple(
        re.compile(str(p), re.IGNORECASE)
        for p in (raw.get('except_if_any') or [])
        if isinstance(p, str) and p.strip()
    )
    return {'skip': skip, 'except': except_}


_QUAL_FLOOR_SCRUB: Tuple[Dict[str, Any], ...] = _compile_qualitative_floor_scrub(
    _RULES.get('qualitative_floor_scrub')
)
_CALENDAR_WINDOW_GUARD: Dict[str, Tuple[Pattern[str], ...]] = _compile_calendar_window_guard(
    _RULES.get('calendar_window_guard')
)


def qualitative_floor_scrub_blocks_slot(query: str, slot: str) -> bool:
    """True when config scrub would drop ``slot`` for ``query`` (shared with vague invent)."""
    if not query or not slot or not _QUAL_FLOOR_SCRUB:
        return False
    for fam in _QUAL_FLOOR_SCRUB:
        if slot not in fam['drop_slots']:
            continue
        mode = fam['mode']
        if mode == 'qual_without_numeric':
            if any(rx.search(query) for rx in fam['qual']) and not any(
                rx.search(query) for rx in fam['numeric']
            ):
                return True
        elif mode == 'ungrounded':
            if not any(rx.search(query) for rx in fam['cues']):
                return True
    return False


# Shared bound grammar — applied once per query against every lexicon metric.
_NUM = r'(?P<{g}>\d+)\s*(?P<{g}_k>k)?'
_LO = _NUM.format(g='lo')
_HI = _NUM.format(g='hi')
_MAX_OPS = r'(?:under|below|at\s+most|max(?:imum)?|less\s+than|no\s+more\s+than)'
_MIN_OPS = r'(?:over|above|at\s+least|min(?:imum)?|more\s+than|greater\s+than)'

_UNIT_DAYS = {'day': 1, 'days': 1, 'week': 7, 'weeks': 7, 'year': 365, 'years': 365}


def _resolve_metric_entry(metric_span: str) -> Optional[Dict[str, Any]]:
    """Map a matched metric span to its lexicon entry (first alias that matches)."""
    if not metric_span:
        return None
    for entry in _METRIC_LEXICON:
        for alias in entry['aliases']:
            try:
                if re.fullmatch(alias, metric_span, re.IGNORECASE):
                    return entry
            except re.error:
                continue
    return None


def _parse_k_num(m: re.Match[str], group: str) -> Optional[int]:
    try:
        raw = m.group(group)
    except (IndexError, TypeError):
        return None
    if raw is None:
        return None
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return None
    try:
        if m.group(f'{group}_k'):
            n *= 1000
    except (IndexError, TypeError):
        pass
    return n


def _metric_slots(entry: Dict[str, Any], query: str) -> Tuple[Optional[str], Optional[str]]:
    """Pick default vs family-cue alternate min/max slots.

    ``min_slot``/``max_slot`` are the default family. ``majestic_min_slot``/
    ``majestic_max_slot`` hold the alternate family (historical key names —
    may be Semrush when default is Majestic, e.g. bare ``ref domains``).
    ``family_cue`` in the query selects the alternate.
    """
    family = str(entry.get('family_cue') or '').strip()
    use_family = bool(family) and bool(re.search(rf'\b{re.escape(family)}\b', query, re.IGNORECASE))
    if use_family and entry.get('majestic_min_slot'):
        return (
            str(entry.get('majestic_min_slot') or '') or None,
            str(entry.get('majestic_max_slot') or entry.get('max_slot') or '') or None,
        )
    min_slot = entry.get('min_slot')
    max_slot = entry.get('max_slot')
    return (str(min_slot) if min_slot else None, str(max_slot) if max_slot else None)


def _active_hard_names() -> frozenset:
    names = _HARD_ENTITY_NAMES_CTX.get()
    if names is None:
        raise ConfigurationError(
            "entity_reconcile requires hard_entity_names via apply_post_merge_reconcile "
            "(qi.entity_slots.hard_entity_names)"
        )
    return names


def _entity(
    name: str,
    value: Any,
    source: str = 'L0_entity',
) -> Entity:
    return Entity(
        name=name,
        value=value,
        confidence=0.95,
        source=source,
        chip_kind=infer_chip_kind(name, _active_hard_names()),
    )


def _has(entities: List[Entity], name: str) -> bool:
    return any(e.name == name for e in entities)


def reconcile_exact_match_mode(query: str, entities: List[Entity]) -> List[Entity]:
    """Inject/normalize keyword_match_mode='exact' when query cues exact match mode.

    ``keyword_phrase`` already encodes \"exact phrase X\" — do not also inject
    match_mode (full-search would diverge from qie_only / grounding).
    """
    if not query or not _EXACT_MODE_RES:
        return entities
    if any(e.name == 'keyword_phrase' for e in entities):
        return entities
    if not any(rx.search(query) for rx in _EXACT_MODE_RES):
        return entities
    out: List[Entity] = []
    found = False
    for e in entities:
        if e.name == 'keyword_match_mode':
            found = True
            if str(e.value).lower() != 'exact':
                logger.info(
                    f"llm_entity_match_mode_normalized value={e.value!r} -> 'exact' reason=exact_cue"
                )
                out.append(replace(e, value='exact'))
            else:
                out.append(e)
        else:
            out.append(e)
    if not found:
        logger.info("llm_entity_match_mode_injected value='exact' reason=exact_cue")
        out.append(_entity('keyword_match_mode', 'exact'))
    return out


_ACTIVE_PENDING_PAIR_RE = re.compile(
    r"\bactive\b.{0,40}\bpending\s+delete\b|\bpending\s+delete\b.{0,40}\bactive\b",
    re.IGNORECASE,
)
_EXPIRED_DROPPING_PAIR_RE = re.compile(
    r"\bexpired(?:\s+domains?)?\b.{0,40}\b(?:or|and)\b.{0,40}\b(?:dropping|pending\s+delete)"
    r"|\b(?:dropping|pending\s+delete)\b.{0,40}\b(?:or|and)\b.{0,40}\bexpired(?:\s+domains?)?"
    r"|\bexpired\s+domains?\b.{0,40}\b(?:or|and)\b.{0,40}\bauction\b"
    r"|\bauction\b.{0,40}\b(?:or|and)\b.{0,40}\bexpired\s+domains?\b",
    re.IGNORECASE,
)
_LIFECYCLE_STATE_TOKENS = frozenset({
    'active', 'pending_delete', 'pending_deletion', 'expired', 'deleted',
    'dropping', 'recently_expired', 'expiring_soon',
})

_BACKORDER_AUCTION_TOKENS = frozenset({
    'backorder', 'dropcatch', 'drop_catch', '25',
})
_EXPLICIT_BACKORDER_CUE_RE = re.compile(
    r"\bbackorder\s+worthy\b|\b(?:want|need|find|show)\s+(?:me\s+)?backorders?\b",
    re.IGNORECASE,
)


def _is_lifecycle_owned_backorder_auction(ent: Entity) -> bool:
    """True when ``auction_type`` is bare backorder/dropcatch (label or ID 25)."""
    if ent.name not in ('auction_type', 'typeIncludeList'):
        return False
    raw = ent.value
    if isinstance(raw, (list, tuple)):
        vals = {str(v).strip().lower() for v in raw if v not in ('', None)}
        return bool(vals) and vals <= _BACKORDER_AUCTION_TOKENS
    sv = str(raw or '').strip().lower()
    if '|' in sv or ',' in sv:
        parts = {p.strip() for p in sv.replace(',', '|').split('|') if p.strip()}
        return bool(parts) and parts <= _BACKORDER_AUCTION_TOKENS
    return sv in _BACKORDER_AUCTION_TOKENS


def _lifecycle_states_from_entities(entities: List[Entity]) -> List[str]:
    states: List[str] = []
    for e in entities:
        if e.name != 'lifecycle_state':
            continue
        raw = e.value
        if isinstance(raw, (list, tuple)):
            parts = [str(v).strip().lower().replace(' ', '_') for v in raw if str(v).strip()]
        else:
            parts = [
                p.strip().lower().replace(' ', '_')
                for p in str(raw or '').replace(',', '|').split('|')
                if p.strip()
            ]
        for p in parts:
            if p == 'pending_deletion':
                p = 'pending_delete'
            if p == 'dropping':
                p = 'pending_delete'
            if p and p not in states:
                states.append(p)
    return states


def reconcile_lifecycle_disjunction(query: str, entities: List[Entity]) -> List[Entity]:
    """Inject lifecycle_disjunction=True when query ORs / pairs lifecycle statuses.

    Also merges split lifecycle chips (two ``lifecycle_state`` entities, or a
    state string misplaced on ``lifecycle_disjunction``) into one pipe/list value.
    Active+pending-delete cues force ``active|pending_delete``.
    Expired+dropping / expired-domain+auction cues force ``expired|pending_delete``.
    """
    if not query:
        return entities
    cue = any(rx.search(query) for rx in _LIFECYCLE_DISJ_RES) or bool(
        _ACTIVE_PENDING_PAIR_RE.search(query)
    ) or bool(_EXPIRED_DROPPING_PAIR_RE.search(query))
    if not cue:
        return entities

    states: List[str] = []
    rest: List[Entity] = []
    saw_disj = False
    for e in entities:
        if e.name == 'lifecycle_state':
            raw = e.value
            if isinstance(raw, (list, tuple)):
                parts = [str(v).strip().lower().replace(' ', '_') for v in raw if str(v).strip()]
            else:
                parts = [
                    p.strip().lower().replace(' ', '_')
                    for p in str(raw or '').replace(',', '|').split('|')
                    if p.strip()
                ]
            for p in parts:
                if p == 'pending_deletion':
                    p = 'pending_delete'
                if p == 'dropping':
                    p = 'pending_delete'
                if p and p not in states:
                    states.append(p)
            continue
        if e.name == 'lifecycle_disjunction':
            saw_disj = True
            # LLM sometimes parks the second state on the disjunction flag.
            if e.value is not True and e.value is not False:
                tok = str(e.value).strip().lower().replace(' ', '_')
                if tok == 'pending_deletion':
                    tok = 'pending_delete'
                if tok == 'dropping':
                    tok = 'pending_delete'
                if tok in _LIFECYCLE_STATE_TOKENS and tok not in states:
                    states.append(tok)
            continue
        rest.append(e)

    if _ACTIVE_PENDING_PAIR_RE.search(query):
        for tok in ('active', 'pending_delete'):
            if tok not in states:
                states.append(tok)
    if _EXPIRED_DROPPING_PAIR_RE.search(query):
        for tok in ('expired', 'pending_delete'):
            if tok not in states:
                states.append(tok)
        # Lifecycle OR owns intent — drop auction-type=backorder from bare "dropping".
        rest = [
            e for e in rest
            if not _is_lifecycle_owned_backorder_auction(e)
        ]

    if not states:
        return entities

    merged_val: Any = states[0] if len(states) == 1 else list(states)
    out = rest + [_entity('lifecycle_state', merged_val)]
    if len(states) > 1 or saw_disj or cue:
        out.append(_entity('lifecycle_disjunction', True))
        logger.info(
            f"llm_entity_lifecycle_disjunction_merged states={states} reason=or_both_cue"
        )
    return out


def _stale_info(query: str) -> Optional[Tuple[int, Optional[int]]]:
    """Return ``(days, source_num)`` for the first stale-listing cue, or None.

    ``source_num`` is the raw count from the cue (``2`` in ``stale over 2 weeks``);
    ``None`` when the pattern uses ``days_fixed``.
    """
    for rx, spec in _STALE_COMPILED:
        m = rx.search(query)
        if not m:
            continue
        if 'days_fixed' in spec:
            try:
                return int(spec['days_fixed']), None
            except (TypeError, ValueError):
                continue
        try:
            num = int(m.group('num'))
        except (IndexError, TypeError, ValueError):
            continue
        unit = 'days'
        try:
            unit = str(m.group('unit') or 'days').lower()
        except IndexError:
            pass
        return num * int(_UNIT_DAYS.get(unit, 1)), num
    return None


def _stale_days(query: str) -> Optional[int]:
    info = _stale_info(query)
    return info[0] if info is not None else None


# Slots that never belong on a listing-staleness cue (listing age ≠ domain age).
_STALE_CONFLICT_SLOTS = frozenset({
    'days_listed_max',
    'startTimeAfter',
    'domain_age_min',
    'domain_age_max',
})
_STALE_PRICE_SLOTS = frozenset({'price_min', 'price_max'})


def reconcile_stale_listing_polarity(query: str, entities: List[Entity]) -> List[Entity]:
    """Map 'listed more than N days ago' / stale cues → days_listed_min (→ startTimeBefore).

    Drops conflicting days_listed_max / startTimeAfter, and LLM misreads of the same
    duration as domain age (minAge) or price (minPrice from the bare ``2`` in
    ``stale over 2 weeks``). Genuine separate price ceilings (``under 300``) are kept
    when their value is not the stale duration count.
    """
    info = _stale_info(query) if query else None
    if info is None:
        return entities
    days, src_num = info
    if days <= 0:
        return entities
    out: List[Entity] = []
    saw_min = False
    for e in entities:
        if e.name == 'days_listed_min':
            saw_min = True
            out.append(e if e.value == days else replace(e, value=days))
            continue
        if e.name in _STALE_CONFLICT_SLOTS:
            logger.info(
                f"llm_entity_stale_listing_dropped name={e.name} value={e.value!r} "
                f"reason=stale_cue_days={days}"
            )
            continue
        if e.name in _STALE_PRICE_SLOTS:
            try:
                pval = float(e.value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                out.append(e)
                continue
            # Drop price that merely echoes the stale duration count ("over 2 weeks" → 2).
            if src_num is not None and pval == float(src_num):
                logger.info(
                    f"llm_entity_stale_listing_dropped name={e.name} value={e.value!r} "
                    f"reason=stale_duration_misread_as_price src_num={src_num}"
                )
                continue
            if pval == float(days):
                logger.info(
                    f"llm_entity_stale_listing_dropped name={e.name} value={e.value!r} "
                    f"reason=stale_days_misread_as_price days={days}"
                )
                continue
        out.append(e)
    if not saw_min:
        logger.info(f"llm_entity_days_listed_min_injected value={days} reason=stale_cue")
        out.append(_entity('days_listed_min', days))
    return out


def reconcile_metric_family_overrides(query: str, entities: List[Entity]) -> List[Entity]:
    """Remap Semrush metric slots → Majestic when query names Majestic (config remap table).

    Also injects Majestic min/max bounds from cue+number patterns when the slot is absent
    (covers regex family misroute that never emitted a Semrush slot to remap).
    """
    if not query:
        return entities
    out = list(entities)
    present = {e.name for e in out}

    remap: Dict[str, str] = {}
    for cue_rx, table in _METRIC_OVERRIDES:
        if cue_rx.search(query):
            remap.update(table)
    if remap:
        remapped: List[Entity] = []
        for e in out:
            target = remap.get(e.name)
            if not target:
                remapped.append(e)
                continue
            if target in present:
                logger.info(
                    f"llm_entity_metric_family_dropped name={e.name} value={e.value!r} "
                    f"reason=target_present target={target}"
                )
                continue
            logger.info(
                f"llm_entity_metric_family_remapped from={e.name} to={target} value={e.value!r}"
            )
            present.add(target)
            remapped.append(replace(e, name=target, chip_kind=infer_chip_kind(target, _active_hard_names())))
        out = remapped
        present = {e.name for e in out}

    for cue_rx, bound_rx, slot in _METRIC_INJECT:
        if slot in present:
            continue
        if not cue_rx.search(query):
            continue
        m = bound_rx.search(query)
        if not m:
            continue
        try:
            num = int(m.group('num'))
        except (IndexError, TypeError, ValueError):
            continue
        logger.info(f"llm_entity_metric_bound_injected name={slot} value={num} reason=majestic_cue_pattern")
        out.append(_entity(slot, num))
        present.add(slot)
    return out


def reconcile_char_constraints(query: str, entities: List[Entity]) -> List[Entity]:
    """Inject has_number / has_hyphen / is_idn from char_constraint_rules.json when absent.

    Linguistic templates only (letters-only, clean brand, hyphen-free). First rule per
    slot wins; existing slot values untouched.
    """
    if not query or not _CHAR_RULES:
        return entities
    present = {e.name for e in entities}
    injected: List[Entity] = []
    claimed: set = set()
    for slot, rx, value in _CHAR_RULES:
        if slot in present or slot in claimed:
            continue
        if not rx.search(query):
            continue
        logger.info(
            f"llm_entity_char_constraint_injected name={slot} value={value!r} reason=char_constraint_rules"
        )
        injected.append(_entity(slot, value))
        claimed.add(slot)
    if not injected:
        return entities
    return list(entities) + injected


def reconcile_ungrounded_char_slots(query: str, entities: List[Entity]) -> List[Entity]:
    """Drop char-constraint slots with no matching cue in char_constraint_rules.json.

    LLM often emits has_hyphen/has_number without an explicit cue. Keep a slot only
    when at least one config rule for that slot matches the query (regex/LLM inject
    from the same rule file already covers grounded cases).
    """
    if not entities or not _CHAR_RULES:
        return entities
    char_slots = {slot for slot, _rx, _val in _CHAR_RULES}
    grounded: set = set()
    if query:
        for slot, rx, _val in _CHAR_RULES:
            if slot in grounded:
                continue
            if rx.search(query):
                grounded.add(slot)
    out: List[Entity] = []
    for e in entities:
        if e.name not in char_slots:
            out.append(e)
            continue
        if e.name in grounded:
            out.append(e)
            continue
        logger.info(
            f"llm_entity_ungrounded_char_dropped name={e.name} value={e.value!r} "
            f"reason=no_char_constraint_cue"
        )
    return out


# Letter-count (FIND minLetters) vs SLD length (FIND minSldLen/maxSldLen).
_LETTER_COUNT_FLOOR_RE = re.compile(
    r"\b(?:min(?:imum)?|at\s+least)\s+(?P<n>\d+)\s+letters?\b"
    r"|\b(?P<n2>\d+)\s+letters?\s+(?:or\s+more|minimum|min|plus)\b",
    re.IGNORECASE,
)
_SLD_LEN_FLOOR_RE = re.compile(
    r"\b(?:longer(?:\s+names?)?|at\s+least|min(?:imum)?)\s+(?P<n>\d+)\s+(?:chars?|characters?)\b"
    r"|\b(?P<n2>\d+)\s+(?:chars?|characters?)\s+(?:plus|or\s+more|or\s+longer|and\s+above|minimum|min)\b"
    r"|\blonger\s+names?\s+(?P<n3>\d+)\s+(?:chars?|characters?)(?:\s+plus)?\b",
    re.IGNORECASE,
)


def reconcile_letters_vs_sld_len(query: str, entities: List[Entity]) -> List[Entity]:
    """Separate alphabetic letter-count from total SLD length.

    - 'minimum 6 letters' → minLetters=6; drop name_length_* (not minSldLen).
    - 'longer names 10 chars plus' → name_length_min=10; drop name_length_max
      when it was wrongly set equal (exact misread of a floor cue).
    """
    if not query:
        return entities
    m_let = _LETTER_COUNT_FLOOR_RE.search(query)
    if m_let is not None:
        raw = m_let.group('n') or m_let.group('n2')
        try:
            n = int(raw)
        except (TypeError, ValueError):
            n = None
        out: List[Entity] = []
        saw = False
        for e in entities:
            if e.name in ('name_length_min', 'name_length_max'):
                logger.info(
                    f"entity_reconcile_letters_drop_sld_len name={e.name} value={e.value!r} "
                    f"reason=letter_count_cue"
                )
                continue
            if e.name == 'minLetters':
                saw = True
                if n is not None and e.value != n:
                    out.append(_entity('minLetters', n))
                else:
                    out.append(e)
                continue
            out.append(e)
        if not saw and n is not None:
            logger.info(f"entity_reconcile_minLetters_injected value={n} reason=letter_count_cue")
            out.append(_entity('minLetters', n))
        return out

    m_sld = _SLD_LEN_FLOOR_RE.search(query)
    if m_sld is None:
        return entities
    raw = m_sld.group('n') or m_sld.group('n2') or m_sld.group('n3')
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return entities
    out = []
    saw_min = False
    max_val = None
    for e in entities:
        if e.name == 'minLetters':
            # Floor-on-chars cue owns SLD length, not alphabetic letter count.
            logger.info(
                f"entity_reconcile_sld_floor_drop_minLetters value={e.value!r} reason=sld_len_floor_cue"
            )
            continue
        if e.name == 'name_length_min':
            saw_min = True
            out.append(e if e.value == n else _entity('name_length_min', n))
            continue
        if e.name == 'name_length_max':
            max_val = e.value
            continue
        out.append(e)
    if not saw_min:
        logger.info(f"entity_reconcile_name_length_min_injected value={n} reason=sld_len_floor_cue")
        out.append(_entity('name_length_min', n))
    # Drop exact max when it merely duplicates the floor (chars plus / or more).
    if max_val is not None:
        try:
            if int(max_val) == n:
                logger.info(
                    f"entity_reconcile_name_length_max_dropped value={max_val!r} "
                    f"reason=sld_len_floor_not_exact"
                )
            else:
                out.append(_entity('name_length_max', max_val))
        except (TypeError, ValueError):
            out.append(_entity('name_length_max', max_val))
    return out


# Bare price ceilings must never invent SLD length (LLM over-applies exclusive under→N-1).
# Include short/long — "short com under 1k" is an intentional length cue, not a false maxSldLen.
_LENGTH_UNIT_CUE_RE = re.compile(
    r"\b(?:chars?|characters?|letters?|sld\s*len(?:gth)?|name\s*length|short|long)\b",
    re.IGNORECASE,
)
_BARE_PRICE_UNDER_RE = re.compile(
    r"\b(?:under|below|less\s+than|at\s+most|up\s+to|no\s+more\s+than|cheaper\s+than)\s+\$?\d",
    re.IGNORECASE,
)


def reconcile_bare_price_not_sld_len(query: str, entities: List[Entity]) -> List[Entity]:
    """Drop name_length_* when query has bare price 'under N' and no length unit.

    '.io names under 500' → maxPrice only. LLM sometimes also emits maxSldLen=499.
    Keeps length when query has short/long / chars / letters (parity with qie_only).
    """
    if not query or not entities:
        return entities
    if _LENGTH_UNIT_CUE_RE.search(query):
        return entities
    if not _BARE_PRICE_UNDER_RE.search(query):
        return entities
    out: List[Entity] = []
    for e in entities:
        if e.name in ('name_length_min', 'name_length_max'):
            logger.info(
                f"entity_reconcile_sld_len_dropped_bare_price name={e.name} "
                f"value={e.value!r} reason=price_under_no_length_unit"
            )
            continue
        out.append(e)
    return out


# "underpriced vs govalue … under 2k" — grounding/qie emit minValuationPrice with maxPrice.
# Multi-intent / LLM often keep price_below_market + price_max but drop govalue_min.
_GOVALUE_VS_CUE_RE = re.compile(
    r"\bvs\.?\s+(?:govalue|go\s*value|appraisal|appraised)\b"
    r"|\b(?:govalue|go\s*value|appraisal)\s+vs\.?\b"
    r"|\b(?:underpriced|undervalued|below\s+market)\b.{0,60}\bvs\.?\b",
    re.IGNORECASE,
)


def reconcile_govalue_vs_price_under(query: str, entities: List[Entity]) -> List[Entity]:
    """Inject govalue_min=0 when query is underpriced-vs-govalue (+ optional under N).

    Live grounding/qie emit minValuationPrice:0 (any positive appraisal) with
    maxPrice from the under-N cue — not a floor copied from price_max.
    """
    if not query or not entities:
        return entities
    if not _GOVALUE_VS_CUE_RE.search(query):
        return entities
    if any(e.name in ('govalue_min', 'govalue_max') for e in entities):
        return entities
    if not any(e.name == 'price_max' for e in entities):
        return entities
    logger.info("entity_reconcile_govalue_min_injected value=0 reason=vs_govalue_price_under")
    return list(entities) + [_entity('govalue_min', 0)]


_AGE_YEARS_UNIT_RE = re.compile(
    r"\b(?:years?|yrs?)\b",
    re.IGNORECASE,
)


def reconcile_bare_price_not_domain_age(query: str, entities: List[Entity]) -> List[Entity]:
    """Drop domain_age_* when bare 'under N' is price and no years unit present.

    'young or unknown age under 500' → maxPrice only; drop invented maxAge=500.
    """
    if not query or not entities:
        return entities
    if _AGE_YEARS_UNIT_RE.search(query):
        return entities
    if not _BARE_PRICE_UNDER_RE.search(query):
        return entities
    # Keep soft age floor for "new brand" / "brand new" (minAge=0).
    keep_new_brand_age = bool(re.search(r'\b(?:new\s+brand|brand\s+new)\b', query, re.I))
    # Keep qualitative age floor for "old/aged domain" (minAge>=1) with price under.
    keep_old_age = bool(re.search(
        r'\b(?:old|aged|aging)\s+domains?\b|\baged\s+domain\b|\bold\s+domain\b',
        query,
        re.I,
    ))
    out: List[Entity] = []
    for e in entities:
        if e.name in ('domain_age_min', 'domain_age_max'):
            if keep_new_brand_age and e.name == 'domain_age_min' and e.value in (0, '0'):
                out.append(e)
                continue
            if keep_old_age and e.name == 'domain_age_min':
                out.append(e)
                continue
            logger.info(
                f"entity_reconcile_domain_age_dropped_bare_price name={e.name} "
                f"value={e.value!r} reason=price_under_no_years_unit"
            )
            continue
        out.append(e)
    return out


def reconcile_metric_ranges(query: str, entities: List[Entity]) -> List[Entity]:
    """Apply shared bound grammar against metric_lexicon (between/to/under/above/soft).

    Config owns metric aliases + slots; code owns operators once. Drops misattributed
    slots when a lexicon entry declares misattr_drop / misattr_slot_values. Max-only
    cues also drop that metric's min_slot (wrong polarity). Soft high→soft_min,
    soft low→soft_max when configured on the entry.
    """
    if not query or not _METRIC_LEXICON:
        return entities

    parts = [f'(?:{a})' for entry in _METRIC_LEXICON for a in entry['aliases']]
    m_alt = '|'.join(sorted(set(parts), key=len, reverse=True))
    metric_cap = rf'(?P<metric>\b(?:{m_alt})\b)'

    grammars: List[Tuple[str, Pattern[str]]] = [
        (
            'dual',
            # Require "and" between min/max ops so "cf above 15 under 2k" (price)
            # does not bind 2k as citation_flow max. Use "above 500 and under 2k".
            re.compile(
                rf'{metric_cap}\s+{_MIN_OPS}\s+{_LO}\s+and\s+{_MAX_OPS}\s+{_HI}\b',
                re.IGNORECASE,
            ),
        ),
        (
            'between',
            re.compile(
                rf'{metric_cap}\s+between\s+{_LO}\s+and\s+{_HI}\b',
                re.IGNORECASE,
            ),
        ),
        (
            'to',
            re.compile(
                rf'{metric_cap}\s+(?:from\s+)?{_LO}\s+to\s+{_HI}\b',
                re.IGNORECASE,
            ),
        ),
        (
            'min_plus',
            re.compile(
                rf'{metric_cap}\s+{_LO}\s*(?:\+|plus)\b',
                re.IGNORECASE,
            ),
        ),
        (
            'max',
            re.compile(
                rf'{metric_cap}\s+{_MAX_OPS}\s+{_HI}\b',
                re.IGNORECASE,
            ),
        ),
        (
            'min',
            re.compile(
                rf'{metric_cap}\s+{_MIN_OPS}\s+{_LO}\b',
                re.IGNORECASE,
            ),
        ),
        (
            'soft_max',
            re.compile(rf'\blow\s+{metric_cap}\b', re.IGNORECASE),
        ),
        (
            'soft_max_trail',
            re.compile(rf'{metric_cap}\s+low\b', re.IGNORECASE),
        ),
        (
            'soft_min',
            re.compile(
                rf'\b(?:high|strong|lots\s+of|barely\s+any)\s+{metric_cap}\b',
                re.IGNORECASE,
            ),
        ),
        (
            'soft_min_trail',
            re.compile(rf'{metric_cap}\s+(?:high|strong|barely\s+any)\b', re.IGNORECASE),
        ),
    ]

    out = list(entities)
    present = {e.name for e in out}
    drop: set = set()
    value_drop: Dict[str, set] = {}
    handled_metrics: set = set()

    for kind, rx in grammars:
        for m in rx.finditer(query):
            entry = _resolve_metric_entry(m.group('metric'))
            if not entry:
                continue
            mid = str(entry.get('id') or m.group('metric').lower())
            if mid in handled_metrics:
                continue
            min_slot, max_slot = _metric_slots(entry, query)
            maj_min = str(entry.get('majestic_min_slot') or '') or None
            maj_max = str(entry.get('majestic_max_slot') or '') or None
            family_mins = frozenset(s for s in (min_slot, maj_min) if s)
            family_maxs = frozenset(s for s in (max_slot, maj_max) if s)
            family_slots = family_mins | family_maxs
            lo = hi = None
            if kind in ('soft_max', 'soft_max_trail'):
                # "low X" → soft_max ceiling when configured; else soft_low_min floor
                # (link/searcher families treat "low" as presence floor, not ceiling).
                soft = entry.get('soft_max')
                soft_lo = entry.get('soft_low_min')
                if soft is not None:
                    try:
                        hi = int(soft)
                    except (TypeError, ValueError):
                        continue
                elif soft_lo is not None:
                    try:
                        lo = int(soft_lo)
                    except (TypeError, ValueError):
                        continue
                else:
                    continue
            elif kind in ('soft_min', 'soft_min_trail'):
                soft = entry.get('soft_min')
                if soft is None:
                    continue
                try:
                    lo = int(soft)
                except (TypeError, ValueError):
                    continue
            elif kind in ('dual', 'between', 'to'):
                lo = _parse_k_num(m, 'lo')
                hi = _parse_k_num(m, 'hi')
            elif kind == 'max':
                hi = _parse_k_num(m, 'hi')
                # "backlinks under 2k" / "traffic under 1500" are price budgets, not
                # maxBacklinks/maxTraffic — skip when under/below + large/k amount.
                if hi is not None and re.search(
                    r'\b(?:under|below|less\s+than|fewer\s+than)\b',
                    m.group(0),
                    re.IGNORECASE,
                ):
                    mid_l = str(mid).lower()
                    if any(
                        tok in mid_l
                        for tok in (
                            'backlink', 'traffic', 'visitor', 'authority',
                            'ref_domain', 'refdomain', 'indexed',
                        )
                    ) and (hi >= 100 or bool(re.search(r'[kKmM]\b', m.group(0)))):
                        logger.info(
                            f"llm_entity_metric_range_skip_budget_as_price metric={mid} "
                            f"hi={hi} reason=under_large_is_price"
                        )
                        continue
                    # "barely any <estibot ext> under N" (N small) → soft floor, not max.
                    if (
                        str(mid).startswith('estibot')
                        and hi is not None
                        and hi < 50
                        and re.search(r'\bbarely\s+any\b', query, re.IGNORECASE)
                    ):
                        logger.info(
                            f"llm_entity_metric_range_skip_barely_any_tiny_max "
                            f"metric={mid} hi={hi}"
                        )
                        soft_floor = entry.get('soft_min')
                        if soft_floor is None:
                            soft_floor = entry.get('soft_low_min')
                        if soft_floor is not None and lo is None:
                            try:
                                lo = int(soft_floor)
                            except (TypeError, ValueError):
                                pass
                        hi = None
                    # Max-only cue: drop wrong-polarity mins (incl. L0 misread).
                # Keep soft-low floor when "low/barely any <metric>" is cued.
                soft_lo = entry.get('soft_low_min')
                window = query[max(0, m.start() - 24):m.end() + 8]
                keep_soft_low = soft_lo is not None and re.search(
                    r'\b(?:low|barely\s+any)\b',
                    window,
                    re.IGNORECASE,
                )
                keep_soft_min = bool(re.search(r'\bbarely\s+any\b', window, re.IGNORECASE))
                if keep_soft_min and lo is None:
                    soft_floor = entry.get('soft_min')
                    if soft_floor is None:
                        soft_floor = soft_lo
                    if soft_floor is not None:
                        try:
                            lo = int(soft_floor)
                        except (TypeError, ValueError):
                            pass
                if not keep_soft_low and not keep_soft_min and hi is not None:
                    for s in family_mins:
                        drop.add(s)
                elif keep_soft_low and min_slot and lo is None:
                    try:
                        lo = int(soft_lo)
                    except (TypeError, ValueError):
                        pass
            elif kind in ('min', 'min_plus'):
                lo = _parse_k_num(m, 'lo')

            for ds in entry.get('misattr_drop') or []:
                drop.add(str(ds))
            for slot, vals in dict(entry.get('misattr_slot_values') or {}).items():
                bucket = value_drop.setdefault(str(slot), set())
                for v in vals or []:
                    bucket.add(str(v).lower())

            # Soft bounds: if L0 already set any family slot, keep L0 (no soft fill).
            if kind.startswith('soft') and (family_slots & present):
                logger.info(
                    f"llm_entity_metric_range_skip_inject metric={mid} kind={kind} "
                    f"reason=l0_owns_family_slots"
                )
                handled_metrics.add(mid)
                continue
            # Soft estibot ceilings/floors are inventory chips — skip on "new keyword" /
            # "emerging" browse where grounding emits no estibot filter.
            if kind.startswith('soft') and str(mid).startswith('estibot') and re.search(
                r'\b(?:new\s+keyword|emerging)\b',
                query,
                re.IGNORECASE,
            ):
                logger.info(
                    f"llm_entity_metric_range_skip_inject metric={mid} kind={kind} "
                    f"reason=new_keyword_or_emerging_browse"
                )
                handled_metrics.add(mid)
                continue
            # Soft TF/CF ceilings on "clean start" / "starting clean" browse — L0 omits
            # qualitative low TF/CF (rule 4). Other soft metrics (semrush links/score) stay.
            if (
                kind.startswith('soft')
                and mid in ('trust_flow', 'citation_flow')
                and re.search(
                    r"\b(?:starting\s+clean|clean\s+start|i'?m\s+fine|im\s+fine)\b",
                    query,
                    re.IGNORECASE,
                )
            ):
                logger.info(
                    f"llm_entity_metric_range_skip_inject metric={mid} kind={kind} "
                    f"reason=clean_start_browse"
                )
                handled_metrics.add(mid)
                continue

            # Exclusive floor cues (above/over/more than) → N+1 so full-search
            # matches grounding/qie_only strict "above N" interpretation.
            exclusive_min = bool(
                kind in ('min', 'min_plus', 'dual')
                and re.search(
                    r'\b(?:over|above|more\s+than|greater\s+than)\b',
                    m.group(0),
                    re.IGNORECASE,
                )
            )
            # Estibot under-N only (live LLMJ: "estibot count under 20" → 19).
            # Other metrics stay inclusive on under/below.
            exclusive_max = bool(
                kind in ('max', 'dual')
                and str(mid).startswith('estibot')
                and re.search(
                    r'\b(?:under|below|less\s+than|fewer\s+than)\b',
                    m.group(0),
                    re.IGNORECASE,
                )
            )
            raw_lo = lo
            raw_hi = hi
            if exclusive_min and lo is not None:
                lo = lo + 1
                # Upgrade L0 inclusive floor that mirrors the cue number.
                if raw_lo is not None and (family_mins & present):
                    upgraded: List[Entity] = []
                    for e in out:
                        if e.name in family_mins:
                            try:
                                if int(e.value) == int(raw_lo):
                                    logger.info(
                                        f"llm_entity_metric_exclusive_min_bump "
                                        f"name={e.name} value={e.value!r}->{lo}"
                                    )
                                    upgraded.append(replace(e, value=lo))
                                    continue
                            except (TypeError, ValueError):
                                pass
                        upgraded.append(e)
                    out = upgraded
            if exclusive_max and hi is not None:
                hi = max(0, hi - 1)
                if raw_hi is not None and (family_maxs & present):
                    upgraded_max: List[Entity] = []
                    for e in out:
                        if e.name in family_maxs:
                            try:
                                if int(e.value) == int(raw_hi):
                                    logger.info(
                                        f"llm_entity_metric_exclusive_max_bump "
                                        f"name={e.name} value={e.value!r}->{hi}"
                                    )
                                    upgraded_max.append(replace(e, value=hi))
                                    continue
                            except (TypeError, ValueError):
                                pass
                        upgraded_max.append(e)
                    out = upgraded_max

            effective = present - drop
            # Do not stack Semrush min when Majestic min already present (or vice versa).
            if min_slot and lo is not None and not (family_mins & effective):
                logger.info(
                    f"llm_entity_metric_range_injected name={min_slot} value={lo} "
                    f"reason=lexicon_{kind}"
                )
                out.append(_entity(min_slot, lo))
                present.add(min_slot)
            if max_slot and hi is not None and not (family_maxs & effective):
                logger.info(
                    f"llm_entity_metric_range_injected name={max_slot} value={hi} "
                    f"reason=lexicon_{kind}"
                )
                out.append(_entity(max_slot, hi))
                present.add(max_slot)
            handled_metrics.add(mid)

    if drop or value_drop:
        cleaned: List[Entity] = []
        for e in out:
            if e.name in drop:
                logger.info(
                    f"llm_entity_metric_misattr_dropped name={e.name} value={e.value!r} "
                    f"reason=lexicon_drop"
                )
                continue
            bad_vals = value_drop.get(e.name)
            if bad_vals:
                vals = e.value if isinstance(e.value, list) else [e.value]
                kept = []
                dropped_any = False
                for v in vals:
                    raw = str(v).lower()
                    try:
                        as_num = str(int(float(v)))
                    except (TypeError, ValueError):
                        as_num = raw
                    if raw in bad_vals or as_num in bad_vals:
                        dropped_any = True
                        continue
                    kept.append(v)
                if dropped_any:
                    logger.info(
                        f"llm_entity_metric_misattr_value_dropped name={e.name} value={e.value!r} "
                        f"reason=lexicon_value_drop"
                    )
                    if not kept:
                        continue
                    if isinstance(e.value, list):
                        cleaned.append(replace(e, value=kept))
                        continue
                    # Scalar fully dropped above via empty kept.
                    continue
            cleaned.append(e)
        out = cleaned
    return out


def reconcile_digit_char_exclude_noise(entities: List[Entity]) -> List[Entity]:
    """Drop keyword_contains_exclude single-digit tokens when has_number=False.

    LLM sometimes emits digit chars as excludes instead of the char constraint.
    """
    if not any(e.name == 'has_number' and e.value is False for e in entities):
        return entities
    out: List[Entity] = []
    for e in entities:
        if e.name != 'keyword_contains_exclude' or not isinstance(e.value, list):
            out.append(e)
            continue
        kept = [v for v in e.value if not (isinstance(v, str) and len(v) == 1 and v.isdigit())]
        if len(kept) != len(e.value):
            logger.info(
                f"llm_entity_digit_exclude_noise_dropped dropped={[v for v in e.value if v not in kept]}"
            )
        if not kept:
            continue
        out.append(e if kept == e.value else replace(e, value=kept))
    return out


# Buy-now / BIN surface forms belong on the regex-owned ``buy_it_now`` bool slot,
# not on ``auction_type`` / typeIncludeList (ID 20). Matches regex_entity_extractor
# ``_BOOLEAN_SLOT_SIGNALS['buy_it_now']`` plus compact ``buynow`` / ``buyout``.
_BUY_NOW_CUE_RE = re.compile(
    r"\b(?:"
    r"buy[\s-]?it[\s-]?now|buy[\s-]?now|buynow|buyout|"
    r"bin(?:\s+(?:price|available|option))?|"
    r"(?:instant|immediate)\s+(?:buy|purchase|bin|buyout)|"
    r"buy\s+(?:instantly|immediately)|"
    # Soft BIN intent (no auction-bid framing).
    r"available\s+to\s+buy\s+right\s+now|"
    r"want\s+to\s+buy\s+something|"
    r"nobody\s+bought\s+yet|"
    r"no\s+one\s+bought\s+yet"
    r")\b",
    re.IGNORECASE,
)
_BUY_NOW_AUCTION_TYPE_VALUES = frozenset({'buynow', 'buy_now', '20'})


def reconcile_buy_now_vs_auction_type(query: str, entities: List[Entity]) -> List[Entity]:
    """Strip BuyNow auction-type values when the query uses buy-now / BIN language.

    Option B: buy-now phrases map to ``buy_it_now`` (regex) only. LLM often also
    emits ``auction_type`` / typeIncludeList as ``buynow`` / ``20``; that becomes a
    spurious FIND ``typeIncludeList=20`` alongside the correct bool flag. Drop those
    values here (post-merge) so regex ``buy_it_now`` remains the sole buy-now signal.
    Non-BuyNow auction types (expiry / closeout / …) are kept.

    Also injects ``buy_it_now=True`` when a buy-now cue is present but the slot is
    absent (covers LLM miss on 'instant buy' / 'immediate purchase').
    """
    if not query or not _BUY_NOW_CUE_RE.search(query):
        return entities
    out: List[Entity] = []
    has_buy_now = False
    for e in entities:
        if e.name == 'buy_it_now':
            has_buy_now = True
            out.append(e)
            continue
        if e.name not in ('auction_type', 'typeExcludeList'):
            out.append(e)
            continue
        if isinstance(e.value, list):
            kept = [v for v in e.value if str(v).lower() not in _BUY_NOW_AUCTION_TYPE_VALUES]
            if len(kept) != len(e.value):
                dropped = [v for v in e.value if v not in kept]
                logger.info(
                    f"entity_reconcile_buy_now_auction_type_dropped name={e.name} "
                    f"dropped={dropped} reason=buy_now_cue_owns_buy_it_now_slot"
                )
            if not kept:
                continue
            out.append(e if kept == e.value else replace(e, value=kept))
        elif str(e.value).lower() in _BUY_NOW_AUCTION_TYPE_VALUES:
            logger.info(
                f"entity_reconcile_buy_now_auction_type_dropped name={e.name} "
                f"dropped=[{e.value!r}] reason=buy_now_cue_owns_buy_it_now_slot"
            )
            continue
        else:
            out.append(e)
    if not has_buy_now:
        logger.info("entity_reconcile_buy_it_now_injected reason=buy_now_cue_present")
        out.append(_entity('buy_it_now', True))
    return out


# Bid-accepted surface forms — FIND ``isBidAccepted`` bool (regex owns primary extract).
_BID_ACCEPTED_CUE_RE = re.compile(
    r"\b(?:"
    r"bid\s+accepted|accepted\s+bid|accepted\s+offer|"
    r"has\s+(?:an?\s+)?accepted\s+(?:bid|offer)|"
    r"with\s+(?:an?\s+)?accepted\s+(?:bid|offer)|"
    r"(?:active\s+)?bid\s+interest|active\s+bidding|"
    r"with\s+(?:active\s+)?bids?\s+interest"
    r")\b",
    re.IGNORECASE,
)


def reconcile_bid_accepted(query: str, entities: List[Entity]) -> List[Entity]:
    """Force ``isBidAccepted=True`` when the query has a bid-accepted cue.

    LLM-first merge can emit ``isBidAccepted=False`` (or omit it). False then wins
    first-seen over regex ``True``, and ``_is_inactive_filter_value`` drops False
    from applied_filters — so the cue vanishes. Cue wins: replace False / inject True.
    """
    if not query or not _BID_ACCEPTED_CUE_RE.search(query):
        return entities
    out: List[Entity] = []
    seen = False
    for e in entities:
        if e.name != 'isBidAccepted':
            out.append(e)
            continue
        seen = True
        if e.value is True:
            out.append(e)
            continue
        logger.info(
            f"entity_reconcile_isBidAccepted_forced value={e.value!r} "
            f"reason=bid_accepted_cue"
        )
        out.append(_entity('isBidAccepted', True))
    if not seen:
        logger.info("entity_reconcile_isBidAccepted_injected reason=bid_accepted_cue_present")
        out.append(_entity('isBidAccepted', True))
    return out


# Bid-count floors / ceilings / exact-empty. Distinct from isBidAccepted (offer accepted flag).
_ZERO_BIDS_CUE_RE = re.compile(
    r"\b(?:zero|0|no|none|without)\s+bids?\b"
    r"|\bbids?\s*(?:count\s+)?(?:is\s+|=\s*)?(?:zero|0)\b"
    r"|\bno\s+bids?\s+yet\b"
    # Gerund / agent forms: "nobody bidding yet", "no one bidding", "no bidding yet"
    r"|\b(?:nobody|no\s*one|no\s*body)(?:'s)?\s+bidd(?:ing|in')\b"
    r"|\bno\s+bidd(?:ing|in')\s+yet\b"
    r"|\b(?:without|no)\s+(?:any\s+)?bidd(?:ing|ers?)\b",
    re.IGNORECASE,
)
# Inclusive floor (>= N).
_BIDS_FLOOR_INCLUSIVE_RE = re.compile(
    r"\b(?P<n>\d+)\s+(?:or\s+more|and\s+above|and\s+over|plus)\s+bids?\b"
    r"|\b(?:at\s+least|minimum|min)\s+(?P<n2>\d+)\s+bids?\b"
    r"|\b(?P<n3>\d+)\+\s*bids?\b"
    r"|\bbids?\s+(?:at\s+least|minimum|min(?:imum)?|>=)\s*(?P<n4>\d+)\b",
    re.IGNORECASE,
)
# Exclusive floor (> N) → minBids = N+1.
_BIDS_FLOOR_EXCLUSIVE_RE = re.compile(
    r"\b(?:more\s+than|greater\s+than|over)\s+(?P<n>\d+)\s+bids?\b"
    r"|\bbids?\s+(?:more\s+than|greater\s+than|over|>)\s*(?P<n2>\d+)\b",
    re.IGNORECASE,
)
# Inclusive ceiling (<= N).
_BIDS_CEIL_INCLUSIVE_RE = re.compile(
    r"\b(?:at\s+most|no\s+more\s+than|maximum|max)\s+(?P<n>\d+)\s+bids?\b"
    r"|\b(?P<n2>\d+)\s+bids?\s+or\s+(?:fewer|less)\b"
    r"|\b(?P<n3>\d+)\s+or\s+(?:fewer|less)\s+bids?\b"
    r"|\bbids?\s+(?:at\s+most|no\s+more\s+than|maximum|max|<=)\s*(?P<n4>\d+)\b",
    re.IGNORECASE,
)
# Exclusive ceiling (< N) → maxBids = N-1.
_BIDS_CEIL_EXCLUSIVE_RE = re.compile(
    r"\b(?:fewer\s+than|less\s+than|under|below)\s+(?P<n>\d+)\s+bids?\b"
    r"|\bbids?\s+(?:fewer\s+than|less\s+than|under|below|<)\s*(?P<n2>\d+)\b",
    re.IGNORECASE,
)


def reconcile_bid_count_bounds(query: str, entities: List[Entity]) -> List[Entity]:
    """Inject/fix auction bid-count bounds from cue phrases.

    Covers LLM misses / inclusive-vs-exclusive errors:
      - 'zero bids yet' / 'nobody bidding yet' → bids_min=0 + bids_max=0
      - '10 or more bids' → bids_min=10 (inclusive)
      - 'more than 15 bids' → bids_min=16 (exclusive N+1)
      - 'fewer than 5 bids' / 'under 5 bids' → bids_max=4 (exclusive N-1)
      - 'at most 5 bids' → bids_max=5 (inclusive)
    """
    if not query:
        return entities
    if _ZERO_BIDS_CUE_RE.search(query):
        out = [e for e in entities if e.name not in ('bids_min', 'bids_max')]
        # "nobody/no one bidding" → minBids=0; "zero/no bids" → maxBids=0 (+ min when
        # both bounds appear in grounding for "no bids" auction cues).
        if re.search(
            r"\b(?:nobody|no\s*one|no\s*body)(?:'s)?\s+bidd(?:ing|in')\b",
            query,
            re.IGNORECASE,
        ):
            logger.info("entity_reconcile_zero_bids reason=nobody_bidding_min")
            out.append(_entity('bids_min', 0))
        elif re.search(r"\b(?:no|zero)\s+bids?\b", query, re.IGNORECASE):
            logger.info("entity_reconcile_zero_bids reason=no_bids_band")
            out.append(_entity('bids_min', 0))
            out.append(_entity('bids_max', 0))
        else:
            logger.info("entity_reconcile_zero_bids reason=zero_bids_max")
            out.append(_entity('bids_max', 0))
        return out

    def _first_int(m: re.Match) -> Optional[int]:
        for g in m.groups():
            if g is not None:
                try:
                    return int(g)
                except (TypeError, ValueError):
                    return None
        return None

    def _force_bound(name: str, value: int, reason: str) -> List[Entity]:
        out: List[Entity] = []
        saw = False
        drop_other = 'bids_max' if name == 'bids_min' else 'bids_min'
        # Exclusive/inclusive single-bound cues own one side; drop the twin if equal misread.
        for e in entities:
            if e.name == drop_other:
                try:
                    if int(e.value) == value or (
                        name == 'bids_min' and int(e.value) == value - 1
                    ) or (
                        name == 'bids_max' and int(e.value) == value + 1
                    ):
                        logger.info(
                            f"entity_reconcile_bids_twin_dropped name={e.name} "
                            f"value={e.value!r} reason={reason}"
                        )
                        continue
                except (TypeError, ValueError):
                    pass
                out.append(e)
                continue
            if e.name == name:
                saw = True
                if isinstance(e.value, (int, float)) and int(e.value) == value:
                    out.append(e)
                else:
                    logger.info(
                        f"entity_reconcile_{name}_forced was={e.value!r} now={value} "
                        f"reason={reason}"
                    )
                    out.append(_entity(name, value))
                continue
            out.append(e)
        if not saw:
            logger.info(f"entity_reconcile_{name}_injected value={value} reason={reason}")
            out.append(_entity(name, value))
        return out

    m_excl_floor = _BIDS_FLOOR_EXCLUSIVE_RE.search(query)
    if m_excl_floor is not None:
        n = _first_int(m_excl_floor)
        if n is not None:
            return _force_bound('bids_min', n + 1, 'bids_floor_exclusive')

    m_excl_ceil = _BIDS_CEIL_EXCLUSIVE_RE.search(query)
    if m_excl_ceil is not None:
        n = _first_int(m_excl_ceil)
        if n is not None:
            return _force_bound('bids_max', max(0, n - 1), 'bids_ceil_exclusive')

    m_incl_floor = _BIDS_FLOOR_INCLUSIVE_RE.search(query)
    if m_incl_floor is not None:
        n = _first_int(m_incl_floor)
        if n is not None:
            return _force_bound('bids_min', n, 'bids_floor_inclusive')

    m_incl_ceil = _BIDS_CEIL_INCLUSIVE_RE.search(query)
    if m_incl_ceil is not None:
        n = _first_int(m_incl_ceil)
        if n is not None:
            return _force_bound('bids_max', n, 'bids_ceil_inclusive')

    return entities


# Currency ISO codes that must never surface as TLD filters (LLM often maps 'eur'→tld).
_CURRENCY_CODES_LOWER = frozenset({'usd', 'eur', 'gbp', 'inr', 'cad', 'aud', 'jpy'})
_CURRENCY_CODE_IN_QUERY_RE = re.compile(
    r"\b(?P<code>usd|eur|gbp|inr|cad|aud|jpy)\b"
    r"|\b(?P<word>dollars?|euros?|pounds?)\b"
    r"|(?P<sym>[$€£¥₹])",
    re.IGNORECASE,
)
_CURRENCY_WORD_TO_CODE = {
    'dollar': 'USD', 'dollars': 'USD',
    'euro': 'EUR', 'euros': 'EUR',
    'pound': 'GBP', 'pounds': 'GBP',
}
_CURRENCY_SYMBOL_TO_CODE = {
    '$': 'USD', '€': 'EUR', '£': 'GBP', '¥': 'JPY', '₹': 'INR',
}


def reconcile_currency_vs_tld(query: str, entities: List[Entity]) -> List[Entity]:
    """Drop TLD values that are currency codes; ensure filterPriceCurrency is set.

    'eur under 800' is price currency, not tldIncludeList=eur. When the query (or an
    existing filterPriceCurrency entity) names a currency, strip matching tld values
    and inject filterPriceCurrency if missing.
    """
    if not entities and not query:
        return entities
    currency_codes: set = set()
    for e in entities:
        if e.name == 'filterPriceCurrency' and e.value is not None:
            currency_codes.add(str(e.value).lower())
    if query:
        m = _CURRENCY_CODE_IN_QUERY_RE.search(query)
        if m is not None:
            if m.group('code'):
                currency_codes.add(m.group('code').lower())
            elif m.group('word'):
                mapped = _CURRENCY_WORD_TO_CODE.get(m.group('word').lower())
                if mapped:
                    currency_codes.add(mapped.lower())
            elif m.group('sym'):
                mapped = _CURRENCY_SYMBOL_TO_CODE.get(m.group('sym'))
                if mapped:
                    currency_codes.add(mapped.lower())
    if not currency_codes:
        return entities

    out: List[Entity] = []
    has_currency_slot = False
    for e in entities:
        if e.name == 'filterPriceCurrency':
            has_currency_slot = True
            out.append(e)
            continue
        if e.name != 'tld':
            out.append(e)
            continue
        if isinstance(e.value, list):
            kept = [v for v in e.value if str(v).lower() not in currency_codes]
            if len(kept) != len(e.value):
                dropped = [v for v in e.value if v not in kept]
                logger.info(
                    f"entity_reconcile_currency_tld_dropped dropped={dropped} "
                    f"reason=currency_code_not_tld"
                )
            if not kept:
                continue
            out.append(e if kept == e.value else replace(e, value=kept))
        elif str(e.value).lower() in currency_codes:
            logger.info(
                f"entity_reconcile_currency_tld_dropped dropped=[{e.value!r}] "
                f"reason=currency_code_not_tld"
            )
            continue
        else:
            out.append(e)
    if not has_currency_slot and currency_codes:
        # Prefer explicit non-USD when both appear; otherwise emit USD for $/dollar/usd.
        non_usd = [c for c in currency_codes if str(c).lower() != 'usd']
        code = (next(iter(non_usd)) if non_usd else next(iter(currency_codes))).upper()
        logger.info(f"entity_reconcile_currency_injected value={code!r}")
        out.append(_entity('filterPriceCurrency', code))
    return out


# Ceiling-only price cues: under/budget/max → price_max only (include $0 listings).
_PRICE_CEILING_CUE_RE = re.compile(
    r"\b(?:under|below|at\s+most|up\s+to|no\s+more\s+than|cheaper\s+than|"
    r"max(?:imum)?|budget|capped\s+at)\b",
    re.IGNORECASE,
)
_PRICE_FLOOR_CUE_RE = re.compile(
    r"\b(?:over|above|at\s+least|min(?:imum)?|more\s+than|greater\s+than|"
    r"floor|starting\s+(?:at|from)|from\s+\$?\d)\b",
    re.IGNORECASE,
)
_PRICE_RANGE_CUE_RE = re.compile(
    r"\bbetween\b|\b\d[\d,]*\s*(?:k)?\s+to\s+\$?\d|\bfrom\s+\$?\d[\d,]*\s+to\b",
    re.IGNORECASE,
)


def reconcile_price_ceiling_only(query: str, entities: List[Entity]) -> List[Entity]:
    """Drop price_min when the query is ceiling-only (under/budget/max, no floor/range).

    Stops regex/LLM from locking minPrice=maxPrice (excludes free/$0 domains).
    'budget 1500 max' / 'under 1k maybe 800' → price_max only.
    """
    if not query or not entities:
        return entities
    if not _PRICE_CEILING_CUE_RE.search(query):
        return entities
    if _PRICE_FLOOR_CUE_RE.search(query) or _PRICE_RANGE_CUE_RE.search(query):
        return entities
    out: List[Entity] = []
    for e in entities:
        if e.name == 'price_min':
            logger.info(
                f"entity_reconcile_price_min_dropped value={e.value!r} "
                f"reason=ceiling_only_query"
            )
            continue
        out.append(e)
    return out


# ---------------------------------------------------------------------------
# Additive inject-if-absent enrichers (search-suite coverage). Never overwrite
# an existing slot — only fill gaps left by L0 / regex.
# ---------------------------------------------------------------------------

_TRAFFIC_VISITOR_SOFT_RE = re.compile(
    r"\b(?:real\s+visitors?|existing\s+traffic|has\s+visitors?|with\s+visitors?|"
    r"has\s+traffic|with\s+(?:web\s+)?traffic|traffic\s+signal)\b",
    re.IGNORECASE,
)
_TRAFFIC_NUMERIC_RE = re.compile(
    r"\btraffic\s*(?:of\s*)?(?:above|over|more\s+than|at\s+least|>=|>)?\s*\$?\d"
    r"|\b\d[\d,]*\s*\+?\s*traffic\b"
    r"|\btraffic\s+\d",
    re.IGNORECASE,
)

_LETTER_LEN_WORD = {
    'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
    'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
}
_EXACT_LETTER_LEN_RE = re.compile(
    r"\b(?P<word>two|three|four|five|six|seven|eight|nine|ten)\s*[- ]?letters?\b"
    r"|\b(?P<num>\d{1,2})\s*[- ]?letters?\b",
    re.IGNORECASE,
)

_PRICE_BETWEEN_RE = re.compile(
    r"\bbetween\s+\$?(?P<lo>\d[\d,]*)\s*(?P<lo_k>k)?\s+and\s+\$?(?P<hi>\d[\d,]*)\s*(?P<hi_k>k)?\b"
    r"|\bfrom\s+\$?(?P<flo>\d[\d,]*)\s*(?P<flo_k>k)?\s+to\s+\$?(?P<fhi>\d[\d,]*)\s*(?P<fhi_k>k)?\b"
    # "1k to 2k" / "800 to 1200" / "around 1k"
    r"|\b\$?(?P<rlo>\d[\d,]*)\s*(?P<rlo_k>k)?\s+to\s+\$?(?P<rhi>\d[\d,]*)\s*(?P<rhi_k>k)?\b"
    r"|\baround\s+\$?(?P<alo>\d[\d,]*)\s*(?P<alo_k>k)?\b",
    re.IGNORECASE,
)

_ENDING_TONIGHT_RE = re.compile(
    r"\b(?:ending|closing|expiring)\s+tonight\b"
    r"|\bend(?:s|ing|ng)?\s+tonight\b",
    re.IGNORECASE,
)
_ENDING_THIS_WEEKEND_RE = re.compile(
    r"\b(?:end(?:ing|ng)|closing|expiring|ends?|closes?)\s+this\s+weekend\b",
    re.IGNORECASE,
)
_ENDING_THIS_WEEK_RE = re.compile(
    r"\b(?:end(?:ing|ng)|closing|expiring)\s+this\s+week\b",
    re.IGNORECASE,
)
# Tonight → 24h. Weekend → 3d. Do NOT also claim "ending soon" — vague_quantifier owns that
# (expiring_soon_seconds). This week → 7d.
_ENDING_TONIGHT_SECONDS = 86400
_ENDING_THIS_WEEKEND_SECONDS = 259200
_ENDING_THIS_WEEK_SECONDS = 604800

# Few-shot: "added recently" → startTimeAfter="-7d".
_ADDED_RECENTLY_RE = re.compile(
    r"\b(?:added|listed)\s+recently\b|\brecently\s+(?:added|listed)\b",
    re.IGNORECASE,
)
# Prompt: just listed / new today → "-1d". Bare "fresh listings" → recently (-7d).
_ADDED_TODAY_RE = re.compile(
    r"\bjust\s+listed\b|\bnew\s+today\b"
    r"|\blistings?\s+today\b|\blisted\s+today\b",
    re.IGNORECASE,
)
_FRESH_LISTINGS_RECENT_RE = re.compile(r"\bfresh\s+listings?\b", re.IGNORECASE)
_ADDED_LAST_HOUR_RE = re.compile(
    r"\b(?:added|listed)\s+(?:in\s+)?(?:the\s+)?last\s+hour\b"
    r"|\blast\s+hour\b",
    re.IGNORECASE,
)
_ADDED_THIS_WEEK_RE = re.compile(
    r"\b(?:added|listed|new)\s+this\s+week\b|\bthis\s+week\b",
    re.IGNORECASE,
)

_SIMILAR_TO_RE = re.compile(
    r"\b(?:(?:domains?|names?)\s+)?"
    r"(?:like|similar\s+to|sounds?\s+like)\s+"
    r"(?P<body>.+?)(?=\s+(?:but|for|not|kinda|style|naming|vibe|feel|type|"
    r"with|under|below|cheap|premium|short|clean|and\s+also)|$)",
    re.IGNORECASE,
)
_SIMILAR_STOP = frozenset({
    'a', 'an', 'the', 'my', 'our', 'real', 'brand', 'brands', 'name', 'names',
    'domain', 'domains', 'style', 'naming', 'vibe', 'feel', 'type',
    'kinda', 'something', 'that', 'sounds', 'sound', 'maybe', 'it', 'specifically',
    'they', 'them', 'these', 'those', 'this', 'could', 'would', 'should', 'be',
    'company', 'someday', 'available', 'look', 'looking', 'cool', 'also',
    'startup', 'product', 'way', 'go', 'less', 'more', 'around',
    'to', 'see', 'like', 'for', 'or', 'and', 'with', 'from', 'into',
    'you', 'on', 'would', 'will', 'can', 'demo', 'day', 'hunt', 'aged',
    # Keep 'app' — short style compounds ("linear app") are seeds.
})
_SIMILAR_NUMERIC_SEED_RE = re.compile(r'^\d+[km]?$', re.IGNORECASE)

# "no X vibe" patterns whose topic is NOT a brand seed (should not inject similar_to).
_NO_CRYPTO_VIBE_RE = re.compile(
    r"\bno\s+(?:crypto|blockchain|web3|nft|defi)\s+vibe\b",
    re.IGNORECASE,
)

_BELOW_MARKET_RE = re.compile(
    r"\b(?:below\s+market(?:\s+value)?|underpriced|undervalued|"
    r"hidden\s+gems?|gems?\s+only|gem\s+domains?|"
    r"sleeper(?:\s+picks?)?|flip\s+potential|but\s+cheaper|"
    r"cheaper\s+alternative|priced\s+way\s+below|under\s+budget|"
    r"resale\s+value|flip\s+later|good\s+deal|"
    r"fraction\s+of\s+what\s+(?:it'?s|its|they'?re)\s+worth|"
    r"selling\s+for\s+a\s+fraction|"
    r"bargain|overlooked|under\s+(?:the\s+)?radar|quick\s+flip)\b",
    re.IGNORECASE,
)

_NO_VIBE_RE = re.compile(
    r"\bno\s+(?P<tok>[a-z]{2,20})\s+vibe\b"
    r"|\bwithout\s+(?P<tok2>[a-z]{2,20})\s+words?\b",
    re.IGNORECASE,
)

_ONE_WORD_RE = re.compile(r"\b(?:one|1|single)[- ]words?\b", re.IGNORECASE)
_TWO_WORD_RE = re.compile(r"\b(?:two|2)[- ]words?\b", re.IGNORECASE)
_ONE_OR_TWO_WORD_RE = re.compile(
    r"\b(?:one|1)\s+words?\s+or\s+(?:two|2)\s+words?\b"
    r"|\b(?:one|1)\s+or\s+(?:two|2)\s+words?\b"
    r"|\b(?:one|1)[- ]word\s+or\s+(?:two|2)[- ]word\b",
    re.IGNORECASE,
)
_SYLLABLE_WORD_COUNT_RE = re.compile(
    r"\b(?P<w>one|two|three|four|five|six|1|2|3|4|5|6)\s*[- ]?syllables?\b",
    re.IGNORECASE,
)
_SYLLABLE_WORD_MAP = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
    '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
}
_PENDING_DELETE_RE = re.compile(
    r"\bpending\s+deletes?\b|\bpending\s+deletion\b",
    re.IGNORECASE,
)
_DROPPED_TODAY_RE = re.compile(
    r"\b(?:what\s+)?dropped\s+today\b",
    re.IGNORECASE,
)
_AVAILABLE_NOW_RE = re.compile(r"\bavailable\s+now\b", re.IGNORECASE)
_INVENTORY_BOUND_FOR_AVAILABLE_NOW_RE = re.compile(
    r"\b(?:under|below|over|above|less\s+than|more\s+than|between|around|"
    r"max|min|budget|cheap|affordable|\$\d|\.(?:com|io|ai|net|org|app|dev|co)\b|"
    r"tld|auctions?|listings?|browse|show\s+me|find\s+me|domains?\s+under)\b",
    re.IGNORECASE,
)
# Soft OR traffic|backlinks|authority|brandable — keep traffic signal;
# drop soft backlink/authority floors invented from an OR arm.
_SOFT_OR_LINK_TRAFFIC_RE = re.compile(
    r"\b(?:some\s+)?(?:traffic|backlinks?|authority)\s+or\s+"
    r"(?:(?:some\s+)?(?:traffic|backlinks?|authority)|"
    r"(?:at\s+least\s+)?(?:be\s+)?(?:really\s+|very\s+)?brandable)"
    r"(?:\s+or\s+(?:(?:some\s+)?(?:traffic|backlinks?|authority)|some\s+real\s+value|"
    r"(?:at\s+least\s+)?(?:be\s+)?(?:really\s+|very\s+)?brandable))?",
    re.IGNORECASE,
)
_SOFT_BACKLINK_FLOOR_NAMES = frozenset({
    'majestic_backlinks_min', 'minMajesticBackLinks',
    'semrush_backlinks_min', 'minSemrushLinksTotal',
    'majestic_tf_min', 'minMajesticTrustFlowScore',
    'semrush_authority_min', 'minSemrushAScore',
})
_LOOK_EXPENSIVE_MINPRICE_RE = re.compile(
    r"\b(?:look|looks|sound|sounds)\s+expensive\b"
    r"|\bexpensive[- ]looking\b",
    re.IGNORECASE,
)


def _parse_price_token(num: Optional[str], k_flag: Optional[str]) -> Optional[int]:
    if not num:
        return None
    try:
        n = int(num.replace(',', ''))
    except (TypeError, ValueError):
        return None
    if k_flag:
        n *= 1000
    return n if n > 0 else None


def reconcile_traffic_visitor_soft(query: str, entities: List[Entity]) -> List[Entity]:
    """Inject has_web_traffic_signal when visitor/traffic soft cue present and slot empty."""
    if not query:
        return entities
    if any(e.name in ('has_web_traffic_signal', 'traffic_min', 'traffic_max') for e in entities):
        return entities
    if _TRAFFIC_NUMERIC_RE.search(query):
        return entities
    if not _TRAFFIC_VISITOR_SOFT_RE.search(query):
        return entities
    logger.info("entity_reconcile_traffic_visitor_soft_injected reason=visitor_cue")
    return list(entities) + [_entity('has_web_traffic_signal', True)]


_SHORT_CUE_RE = re.compile(r"\bshort\b", re.IGNORECASE)


def reconcile_exact_letter_length(query: str, entities: List[Entity]) -> List[Entity]:
    """Inject name_length_min=max=N for 'four letter' / 'N-letter' when both slots empty.

    Ceiling cues (``N chars or less``) are max-only — drop a mistaken min twin.
    Bare ``short`` (no numeric length) → name_length_max=5 when length slots empty.
    """
    if not query:
        return entities
    # Ceiling cues ("N chars or less") are max-only — not exact length.
    if re.search(r'\b(?:or\s+less|or\s+fewer|and\s+under|and\s+below)\b', query, re.IGNORECASE):
        cleaned = [e for e in entities if e.name != 'name_length_min']
        if len(cleaned) != len(entities):
            logger.info("entity_reconcile_exact_letter_length_min_dropped reason=ceiling_cue")
        return cleaned
    if any(e.name in ('name_length_min', 'name_length_max') for e in entities):
        return entities
    # Floor cues own minLetters — never also invent exact SLD length.
    if _LETTER_COUNT_FLOOR_RE.search(query) or any(e.name == 'minLetters' for e in entities):
        return entities
    m = _EXACT_LETTER_LEN_RE.search(query)
    if m is not None:
        if m.group('word'):
            n = _LETTER_LEN_WORD.get(m.group('word').lower())
        else:
            try:
                n = int(m.group('num'))
            except (TypeError, ValueError):
                n = None
        if n is None or n < 1 or n > 63:
            return entities
        logger.info(f"entity_reconcile_exact_letter_length_injected value={n}")
        return list(entities) + [
            _entity('name_length_min', n),
            _entity('name_length_max', n),
        ]
    # Bare short → max length 5 (L0 / regex parity). Skip when explicit char count present.
    if (
        _SHORT_CUE_RE.search(query)
        and not re.search(r"\b\d+\s*(?:chars?|characters?|letters?)\b", query, re.IGNORECASE)
    ):
        logger.info("entity_reconcile_short_max_injected value=5")
        return list(entities) + [_entity('name_length_max', 5)]
    return entities


def reconcile_price_between(query: str, entities: List[Entity]) -> List[Entity]:
    """Inject price_min+price_max for 'between A and B' / 'from A to B' / 'A to B' when absent."""
    if not query:
        return entities
    m = _PRICE_BETWEEN_RE.search(query)
    if m is None:
        return entities
    # Skip SEO/metric ranges: "tf between 20 and 50", "citation flow 20 to 40".
    prefix = query[max(0, m.start() - 40):m.start()].lower()
    if re.search(
        r'\b(?:tf|cf|trust\s+flow|citation\s+flow|authority|backlinks?|'
        r'ref\s+domains?|search\s+volume|ascore|da|age|years?|bids?)\s*$',
        prefix,
    ):
        return entities
    # "around N" → exact band min=max when absent (live L0 parity).
    # "budget around 2k maybe less" → max-only soft ceiling.
    if m.groupdict().get('alo'):
        mid = _parse_price_token(m.group('alo'), m.group('alo_k'))
        if mid is None:
            return entities
        if re.search(r'\b(?:maybe\s+)?(?:less|under|below)\b', query, re.IGNORECASE):
            # "budget around 2k maybe less" → maxPrice only (exclusive N-1 when k).
            hi = mid - 1 if (m.group('alo_k') or mid >= 1000) else mid
            out = list(entities)
            if not any(e.name == 'price_max' for e in out):
                out.append(_entity('price_max', hi))
                logger.info(f"entity_reconcile_price_around_max_injected value={hi}")
            # Drop mistaken exact min from earlier passes.
            out = [e for e in out if e.name != 'price_min' or e.value != mid]
            return out
        out = list(entities)
        has_min = any(e.name == 'price_min' for e in out)
        has_max = any(e.name == 'price_max' for e in out)
        if has_min and has_max:
            return entities
        if not has_min:
            out.append(_entity('price_min', mid))
            logger.info(f"entity_reconcile_price_around_band_min_injected value={mid}")
        if not has_max:
            out.append(_entity('price_max', mid))
            logger.info(f"entity_reconcile_price_around_band_max_injected value={mid}")
        return out
    else:
        lo = _parse_price_token(
            m.group('lo') or m.group('flo') or m.group('rlo'),
            m.group('lo_k') or m.group('flo_k') or m.group('rlo_k'),
        )
        hi = _parse_price_token(
            m.group('hi') or m.group('fhi') or m.group('rhi'),
            m.group('hi_k') or m.group('fhi_k') or m.group('rhi_k'),
        )
    if lo is None or hi is None or lo > hi:
        return entities
    out = list(entities)
    has_min = any(e.name == 'price_min' for e in out)
    has_max = any(e.name == 'price_max' for e in out)
    if has_min and has_max:
        return entities
    if not has_min:
        out.append(_entity('price_min', lo))
        logger.info(f"entity_reconcile_price_between_min_injected value={lo}")
    if not has_max:
        out.append(_entity('price_max', hi))
        logger.info(f"entity_reconcile_price_between_max_injected value={hi}")
    return out


_ENDING_SOON_GAP_RE = re.compile(
    r"\b(?:end(?:ing|ng)|closing|expiring|ends?|closes?|expires?)"
    r"(?:\s+\w+){0,3}\s+soon\b",
    re.IGNORECASE,
)


def reconcile_ending_urgency(query: str, entities: List[Entity]) -> List[Entity]:
    """Inject/force time_remaining_max for ending tonight / weekend / this week / soon.

    Weekend/tonight/soon cues win over a wrong LLM ``endTimeBefore=-7d``.
    """
    if not query:
        return entities
    if _ENDING_TONIGHT_RE.search(query):
        secs, reason = _ENDING_TONIGHT_SECONDS, 'ending_tonight'
    elif _ENDING_THIS_WEEKEND_RE.search(query):
        secs, reason = _ENDING_THIS_WEEKEND_SECONDS, 'ending_this_weekend'
    elif _ENDING_THIS_WEEK_RE.search(query):
        secs, reason = _ENDING_THIS_WEEK_SECONDS, 'ending_this_week'
    elif _ENDING_SOON_GAP_RE.search(query):
        secs, reason = _ENDING_TONIGHT_SECONDS, 'ending_soon'
    else:
        return entities
    # Weekend/tonight/soon: rewrite wrong endTimeBefore offsets (e.g. LLM -7d).
    force = reason in ('ending_this_weekend', 'ending_tonight', 'ending_soon')
    out: List[Entity] = []
    saw = False
    for e in entities:
        if e.name in ('time_remaining_max', 'endTimeBefore', 'endTimeAfter'):
            if force:
                if not saw:
                    out.append(_entity('time_remaining_max', secs))
                    saw = True
                continue
            return entities
        out.append(e)
    if not saw:
        logger.info(f"entity_reconcile_ending_urgency_injected value={secs} reason={reason}")
        out.append(_entity('time_remaining_max', secs))
    elif force:
        logger.info(f"entity_reconcile_ending_urgency_forced value={secs} reason={reason}")
    return out


# Calendar / lookback windows (not requiring listed/added verbs). Shared alias
# table — archetypes (analytics complement, hybrid filter browse) all use it.
_WINDOW_ALIAS_OFFSETS: Dict[str, str] = {
    'last week': '-7d',
    'past week': '-7d',
    'this week': '-7d',
    'last month': '-30d',
    'past month': '-30d',
    'this month': '-30d',
}
_WINDOW_PERIOD_RE = re.compile(
    r"\b(?:in|over|during|for|across)\s+(?:the\s+)?"
    r"(?P<prepped>last\s+week|past\s+week|this\s+week|last\s+month|past\s+month|this\s+month|"
    r"last\s+\d+\s+days?|past\s+\d+\s+days?)\b"
    r"|\b(?P<bare>last\s+week|past\s+week|this\s+week|last\s+month|past\s+month|this\s+month)\b"
    r"|\b(?:last|past)\s+(?P<num>\d+)\s+days?\b",
    re.IGNORECASE,
)


def _window_offset_from_query(query: str) -> Optional[Tuple[str, str]]:
    """Return (startTimeAfter offset, reason) for calendar lookback phrases, else None."""
    m = _WINDOW_PERIOD_RE.search(query or '')
    if m is None:
        return None
    if m.group('num'):
        days = int(m.group('num'))
        if days < 1:
            return None
        return f'-{days}d', 'window_numeric_days'
    raw = (m.group('prepped') or m.group('bare') or '').lower()
    raw = re.sub(r'\s+', ' ', raw.strip())
    num_m = re.match(r'(?:last|past)\s+(\d+)\s+days?$', raw)
    if num_m is not None:
        days = int(num_m.group(1))
        if days < 1:
            return None
        return f'-{days}d', 'window_numeric_days'
    offset = _WINDOW_ALIAS_OFFSETS.get(raw)
    if offset is None:
        return None
    return offset, 'window_alias'


def reconcile_added_recently(query: str, entities: List[Entity]) -> List[Entity]:
    """Inject startTimeAfter for listing-recency and calendar lookback cues."""
    if not query:
        return entities
    if any(e.name in ('startTimeAfter', 'days_listed_max', 'days_listed_min') for e in entities):
        return entities
    if _ADDED_LAST_HOUR_RE.search(query):
        offset, reason = '-1h', 'last_hour'
    elif _ADDED_TODAY_RE.search(query):
        offset, reason = '-1d', 'listed_today'
    elif _FRESH_LISTINGS_RECENT_RE.search(query) or _ADDED_RECENTLY_RE.search(query):
        offset, reason = '-7d', 'added_recently'
    elif re.search(
        r"\b(?:added|listed|new)\s+this\s+week\b"
        r"|\bthis\s+week\s+(?:added|listed|new)\b"
        r"|\bfresh\s+listings?\s+this\s+week\b",
        query,
        re.IGNORECASE,
    ):
        # Listing-age "this week" only — not "fresh <topic> names this week".
        offset, reason = '-7d', 'this_week_listed'
    else:
        # Config: calendar_window_guard — popularity speech ≠ listing lookback.
        skip_rxs = _CALENDAR_WINDOW_GUARD.get('skip') or ()
        except_rxs = _CALENDAR_WINDOW_GUARD.get('except') or ()
        if skip_rxs and any(rx.search(query) for rx in skip_rxs):
            if not except_rxs or not any(rx.search(query) for rx in except_rxs):
                return entities
        window = _window_offset_from_query(query)
        if window is None:
            return entities
        offset, reason = window
    logger.info(f"entity_reconcile_added_recently_injected value={offset!r} reason={reason}")
    return list(entities) + [_entity('startTimeAfter', offset)]


_SOUNDS_LIKE_INDUSTRY_RE = re.compile(
    r"\b(?:sounds?\s+like|similar\s+to)\s+(?:a|an|the)\s+"
    r"(?P<topic>startup|saas|fintech|devtools|healthcare|crypto)\b",
    re.IGNORECASE,
)


def reconcile_similar_to_brands(query: str, entities: List[Entity]) -> List[Entity]:
    """Inject similar_to brand list from like/similar to/sounds like cues when absent."""
    if not query:
        return entities
    # "X meets Y" / "sounds like real brand" → not brand seeds.
    if re.search(
        r"\b(?:ai|fintech|saas|devtools|crypto|startup)\s+meets\s+"
        r"(?:ai|fintech|saas|devtools|crypto|startup)\b"
        r"|\b(?:sounds?\s+like|similar\s+to)\s+(?:a\s+)?(?:real\s+)?brands?\b",
        query,
        re.IGNORECASE,
    ):
        cleaned = [e for e in entities if e.name != 'similar_to']
        if len(cleaned) != len(entities):
            logger.info("entity_reconcile_similar_to_dropped reason=meets_or_real_brand")
        return cleaned
    # "sounds like a startup/saas/…" → topic owns; drop venue seeds (product hunt/yc).
    m_ind = _SOUNDS_LIKE_INDUSTRY_RE.search(query)
    if m_ind is not None:
        cleaned = [e for e in entities if e.name != 'similar_to']
        if len(cleaned) != len(entities):
            logger.info("entity_reconcile_similar_to_dropped reason=sounds_like_industry_topic")
        topic = (m_ind.group('topic') or '').strip().lower()
        if topic and not any(e.name == 'topic_include' for e in cleaned):
            logger.info(f"entity_reconcile_sounds_like_topic_injected topic={topic}")
            cleaned = list(cleaned) + [_entity('topic_include', [topic])]
        return cleaned
    if any(e.name == 'similar_to' for e in entities):
        return entities
    # "no crypto vibe" / "no X vibe" → keyword_contains_exclude, not similar_to.
    if _NO_CRYPTO_VIBE_RE.search(query):
        m_clean = _SIMILAR_TO_RE.search(re.sub(r'\bno\s+\w+\s+vibe\b', '', query, flags=re.IGNORECASE))
        if m_clean is None:
            return entities
        m = m_clean
    else:
        m = _SIMILAR_TO_RE.search(query)
    if m is None:
        return entities
    body = (m.group('body') or '').strip()
    if not body:
        return entities
    raw_parts = re.split(r"\s*(?:,|/|\bor\b|\band\b)\s*", body, flags=re.IGNORECASE)
    brands: List[str] = []
    seen: set = set()
    for part in raw_parts:
        words = re.findall(r"[a-z0-9][\w.-]*", (part or '').lower())
        if not words:
            continue
        # Short compound before style/naming only ("linear app") — else first token.
        keep_words = words
        if len(words) > 2 or not re.search(
            r'\b(?:style|naming|vibe)\b', query, re.IGNORECASE,
        ):
            keep_words = words[:1]
        for tok in keep_words:
            if (
                tok in _SIMILAR_STOP
                or len(tok) < 2
                or tok in seen
                or _SIMILAR_NUMERIC_SEED_RE.match(tok)
            ):
                continue
            seen.add(tok)
            brands.append(tok)
            if len(brands) >= 5:
                break
        if len(brands) >= 5:
            break
    if not brands:
        return entities
    logger.info(f"entity_reconcile_similar_to_injected brands={brands}")
    return list(entities) + [_entity('similar_to', brands)]


def reconcile_investor_below_market(query: str, entities: List[Entity]) -> List[Entity]:
    """Inject price_below_market for hidden gem / sleeper / below market when absent."""
    if not query:
        return entities
    if any(e.name == 'price_below_market' for e in entities):
        return entities
    if not _BELOW_MARKET_RE.search(query):
        return entities
    logger.info("entity_reconcile_investor_below_market_injected")
    return list(entities) + [_entity('price_below_market', True)]


def reconcile_no_vibe_exclude(query: str, entities: List[Entity]) -> List[Entity]:
    """Inject keyword_contains_exclude for 'no X vibe' / 'without X words' when absent."""
    if not query:
        return entities
    if any(e.name == 'keyword_contains_exclude' for e in entities):
        return entities
    m = _NO_VIBE_RE.search(query)
    if m is None:
        return entities
    tok = (m.group('tok') or m.group('tok2') or '').strip().lower()
    if not tok or tok in _SIMILAR_STOP:
        return entities
    logger.info(f"entity_reconcile_no_vibe_exclude_injected token={tok}")
    return list(entities) + [_entity('keyword_contains_exclude', [tok])]


_MAX_N_WORDS_RE = re.compile(
    r"\b(?:max(?:imum)?|at\s+most|up\s+to|no\s+more\s+than)\s+(?P<n>\d+)\s+words?\b",
    re.IGNORECASE,
)
_CHAR_PATTERN_OR_RE = re.compile(
    r"\b(?P<pats>[vcn]{2,8}(?:\s+or\s+[vcn]{2,8})+)\b",
    re.IGNORECASE,
)


def reconcile_word_count_phrase(query: str, entities: List[Entity]) -> List[Entity]:
    """Inject/fix word_count for one/two-word, syllable, and max-N-words phrases."""
    if not query:
        return entities
    # "max N words" → min=1, max=N (not min=N).
    m_max = _MAX_N_WORDS_RE.search(query)
    if m_max is not None:
        n = int(m_max.group('n'))
        out = [e for e in entities if e.name not in ('word_count_min', 'word_count_max')]
        logger.info(f"entity_reconcile_max_words_normalized max={n}")
        return out + [_entity('word_count_min', 1), _entity('word_count_max', n)]
    # "one word or two word" → range [1, 2] (overrides single-side invents).
    if _ONE_OR_TWO_WORD_RE.search(query):
        out = [e for e in entities if e.name not in ('word_count_min', 'word_count_max')]
        logger.info("entity_reconcile_one_or_two_words_injected min=1 max=2")
        return out + [_entity('word_count_min', 1), _entity('word_count_max', 2)]
    if any(e.name in ('word_count_min', 'word_count_max') for e in entities):
        return entities
    m_syl = _SYLLABLE_WORD_COUNT_RE.search(query)
    if m_syl is not None:
        n = _SYLLABLE_WORD_MAP.get(m_syl.group('w').lower())
        if n is not None:
            out = [e for e in entities if e.name not in ('word_count_min', 'word_count_max')]
            logger.info(f"entity_reconcile_syllable_word_count_injected value={n}")
            return out + [
                _entity('word_count_min', n),
                _entity('word_count_max', n),
            ]
    if _ONE_WORD_RE.search(query):
        n = 1
    elif _TWO_WORD_RE.search(query):
        n = 2
    else:
        return entities
    logger.info(f"entity_reconcile_word_count_phrase_injected value={n}")
    return list(entities) + [
        _entity('word_count_min', n),
        _entity('word_count_max', n),
    ]


def reconcile_char_pattern_or(query: str, entities: List[Entity]) -> List[Entity]:
    """Pipe-join coordinated charPattern mnemonics ('cvcv or vcvc' → cvcv|vcvc)."""
    if not query:
        return entities
    m = _CHAR_PATTERN_OR_RE.search(query)
    if m is None:
        return entities
    pats = [
        p.strip().lower()
        for p in re.split(r'\s+or\s+', m.group('pats'), flags=re.IGNORECASE)
        if re.fullmatch(r'[vcn]{2,8}', p.strip(), re.IGNORECASE)
    ]
    if len(pats) < 2:
        return entities
    joined = '|'.join(pats)
    out: List[Entity] = []
    saw = False
    for e in entities:
        if e.name != 'charPattern':
            out.append(e)
            continue
        out.append(_entity('charPattern', joined, source=e.source))
        saw = True
    if not saw:
        out.append(_entity('charPattern', joined))
    logger.info(f"entity_reconcile_char_pattern_or value={joined}")
    return out


def reconcile_unknown_age_with_price(query: str, entities: List[Entity]) -> List[Entity]:
    """Drop unknown-age / age bounds when 'young|unknown age under N' (price owns)."""
    if not query:
        return entities
    if not re.search(
        r"\b(?:young|unknown)\b.{0,40}\bage\b|\bunknown\s+age\b",
        query,
        re.IGNORECASE,
    ):
        return entities
    if not re.search(r"\b(?:under|below|less\s+than)\s+\$?\d", query, re.IGNORECASE):
        return entities
    if re.search(r"\byears?\b", query, re.IGNORECASE):
        return entities
    drop = {
        'domain_age_is_unknown', 'domain_age_min', 'domain_age_max',
        'minAge', 'maxAge',
    }
    return [e for e in entities if e.name not in drop]


def reconcile_scrub_lifecycle_spurious_backorder(
    query: str, entities: List[Entity],
) -> List[Entity]:
    """Drop bare backorder auction_type when lifecycle OR owns ``dropping`` cue.

    LLM entity path may keep ``backorder``/``25`` after lifecycle merge; L0 filter
    scrub already drops it for qie_only. Inject-if-absent inverse — remove only.
    """
    if not query or not entities or _EXPLICIT_BACKORDER_CUE_RE.search(query):
        return entities
    states = _lifecycle_states_from_entities(entities)
    if not states:
        return entities
    has_disj = any(e.name == 'lifecycle_disjunction' for e in entities)
    lifecycle_or = has_disj or len(states) > 1
    if not lifecycle_or:
        return entities
    if not (
        _EXPIRED_DROPPING_PAIR_RE.search(query)
        or re.search(r"\b(?:expired|dropping|pending\s+delete)\b", query, re.IGNORECASE)
    ):
        return entities
    out = [e for e in entities if not _is_lifecycle_owned_backorder_auction(e)]
    if len(out) != len(entities):
        logger.info("entity_reconcile_lifecycle_spurious_backorder_dropped")
    return out


def reconcile_pending_delete_lifecycle(query: str, entities: List[Entity]) -> List[Entity]:
    """Inject lifecycle_state=pending_delete when cue present and slot empty."""
    if not query:
        return entities
    if any(e.name == 'lifecycle_state' for e in entities):
        return entities
    if not _PENDING_DELETE_RE.search(query):
        return entities
    logger.info("entity_reconcile_pending_delete_injected")
    return list(entities) + [_entity('lifecycle_state', 'pending_delete')]


def reconcile_dropped_today_lifecycle(query: str, entities: List[Entity]) -> List[Entity]:
    """Inject lifecycle_state=deleted (+ startTimeAfter=-1d) for dropped-today cues."""
    if not query or not _DROPPED_TODAY_RE.search(query):
        return entities
    out = list(entities)
    # Contrastive "yesterday / nobody bought" → pending_delete (+ disjunction), not deleted.
    if re.search(
        r"\b(?:yesterday|nobody\s+bought|pending\s+delete)\b",
        query,
        re.IGNORECASE,
    ):
        rest = [e for e in out if e.name not in ('lifecycle_state', 'lifecycle_disjunction')]
        rest.append(_entity('lifecycle_state', 'pending_delete'))
        rest.append(_entity('lifecycle_disjunction', True))
        if not any(e.name in ('startTimeAfter', 'days_listed_max') for e in rest):
            rest.append(_entity('startTimeAfter', '-1d'))
        logger.info(
            "entity_reconcile_dropped_today_contrastive_pending_delete"
        )
        return rest
    if not any(e.name == 'lifecycle_state' for e in out):
        out.append(_entity('lifecycle_state', 'deleted'))
        logger.info("entity_reconcile_dropped_today_lifecycle_injected value=deleted")
    if not any(e.name in ('startTimeAfter', 'days_listed_max') for e in out):
        out.append(_entity('startTimeAfter', '-1d'))
        logger.info("entity_reconcile_dropped_today_start_injected value=-1d")
    return out


def reconcile_scrub_qualitative_metric_floors(
    query: str, entities: List[Entity],
) -> List[Entity]:
    """Drop invented metric floors using qualitative_floor_scrub from rules JSON.

    Family table in entity_reconcile_rules.json — add a family for new scenarios;
    no code change. Modes: qual_without_numeric, ungrounded.
    """
    if not query or not entities or not _QUAL_FLOOR_SCRUB:
        return entities
    drop_slots: set = set()
    hit_ids: List[str] = []
    for fam in _QUAL_FLOOR_SCRUB:
        mode = fam['mode']
        active = False
        if mode == 'qual_without_numeric':
            active = any(rx.search(query) for rx in fam['qual']) and not any(
                rx.search(query) for rx in fam['numeric']
            )
        elif mode == 'ungrounded':
            active = not any(rx.search(query) for rx in fam['cues'])
        if not active:
            continue
        drop_slots |= set(fam['drop_slots'])
        hit_ids.append(str(fam['id']))
    if not drop_slots:
        return entities
    out: List[Entity] = []
    dropped = False
    for e in entities:
        if e.name in drop_slots:
            dropped = True
            continue
        out.append(e)
    if dropped:
        logger.info(
            f"entity_reconcile_qualitative_metric_floors_scrubbed "
            f"families={hit_ids} drop_slots={sorted(drop_slots)}"
        )
    return out


def reconcile_available_now_lifecycle(query: str, entities: List[Entity]) -> List[Entity]:
    """Inject lifecycle_state=active when 'available now' + inventory bound."""
    if not query or not _AVAILABLE_NOW_RE.search(query):
        return entities
    if any(e.name == 'lifecycle_state' for e in entities):
        return entities
    # Mirror regex lifecycle guard: topic-only "startup names available now" is fluff.
    inv = {'tld', 'price_min', 'price_max', 'auction_type', 'typeExcludeList'}
    if not (
        any(e.name in inv for e in entities)
        or _INVENTORY_BOUND_FOR_AVAILABLE_NOW_RE.search(query)
    ):
        return entities
    logger.info("entity_reconcile_available_now_lifecycle_injected value=active")
    return list(entities) + [_entity('lifecycle_state', 'active')]


def reconcile_soft_or_link_traffic(query: str, entities: List[Entity]) -> List[Entity]:
    """Drop soft backlink floors when query is traffic|backlinks|authority OR.

    Live grounding keeps has_web_traffic_signal only; LLM often invents
    minMajesticBackLinks=1 from the backlinks arm of the soft OR.
    """
    if not query or not entities or not _SOFT_OR_LINK_TRAFFIC_RE.search(query):
        return entities
    out: List[Entity] = []
    for e in entities:
        if e.name in _SOFT_BACKLINK_FLOOR_NAMES:
            try:
                if int(e.value) <= 1:
                    logger.info(
                        f"entity_reconcile_soft_or_backlink_floor_dropped "
                        f"name={e.name} value={e.value!r}"
                    )
                    continue
            except (TypeError, ValueError):
                pass
        out.append(e)
    return out


def reconcile_look_expensive_not_min_price(query: str, entities: List[Entity]) -> List[Entity]:
    """Drop invented minPrice from 'look expensive' without a numeric floor cue."""
    if not query or not entities or not _LOOK_EXPENSIVE_MINPRICE_RE.search(query):
        return entities
    if re.search(
        r"\b(?:over|above|at\s+least|more\s+than|from|min(?:imum)?)\s+\$?\d",
        query,
        re.IGNORECASE,
    ):
        return entities
    out: List[Entity] = []
    for e in entities:
        if e.name in ('price_min', 'minPrice'):
            logger.info(
                f"entity_reconcile_look_expensive_min_price_dropped value={e.value!r}"
            )
            continue
        out.append(e)
    return out


_DOESNT_SOUND_LIKE_RE = re.compile(
    r"\b(?:doesn'?t|does\s+not|dont|don'?t)\s+sound\s+like\s+(?P<tok>[a-z][a-z0-9_-]{1,24})\b",
    re.IGNORECASE,
)


def reconcile_doesnt_sound_like_not_similar(query: str, entities: List[Entity]) -> List[Entity]:
    """Drop similar_to tokens that echo 'doesn't sound like X' (exclude owns X)."""
    if not query or not entities:
        return entities
    banned = {m.group('tok').lower() for m in _DOESNT_SOUND_LIKE_RE.finditer(query)}
    if not banned:
        return entities
    out: List[Entity] = []
    for e in entities:
        if e.name != 'similar_to':
            out.append(e)
            continue
        vals = e.value if isinstance(e.value, list) else [e.value]
        kept = [
            v for v in vals
            if str(v).lower().strip() not in banned
            and str(v).lower().strip() not in ('stays', 'stay', 'remains')
        ]
        if not kept:
            logger.info(
                f"entity_reconcile_doesnt_sound_like_similar_dropped dropped={vals}"
            )
            continue
        if len(kept) != len(vals):
            logger.info(
                f"entity_reconcile_doesnt_sound_like_similar_trimmed kept={kept}"
            )
            out.append(replace(e, value=kept))
        else:
            out.append(e)
    return out


_STANDARD_OR_PARTNER_RE = re.compile(
    r"\bstandard\s+or\s+partner\b|\bpartner\s+or\s+standard\b",
    re.IGNORECASE,
)


def reconcile_standard_or_partner(query: str, entities: List[Entity]) -> List[Entity]:
    """Force auction_type = standard + partner when both cued (LLM often keeps one)."""
    if not query or not _STANDARD_OR_PARTNER_RE.search(query):
        return entities
    out: List[Entity] = []
    saw = False
    for e in entities:
        if e.name != 'auction_type':
            out.append(e)
            continue
        vals: List[str] = []
        raw = e.value
        if isinstance(raw, (list, tuple)):
            vals = [str(v).strip().lower() for v in raw if str(v).strip()]
        elif raw is not None:
            vals = [
                p.strip().lower()
                for p in str(raw).replace(',', '|').split('|')
                if p.strip()
            ]
        # Canonical labels (not "listed" alias).
        vals = ['standard' if v == 'listed' else v for v in vals]
        for need in ('standard', 'partner'):
            if need not in vals:
                vals.append(need)
        saw = True
        out.append(replace(e, value=vals))
    if not saw:
        out.append(_entity('auction_type', ['standard', 'partner']))
        logger.info("entity_reconcile_standard_or_partner_injected")
    else:
        logger.info("entity_reconcile_standard_or_partner_forced")
    return out


_TOPICISH_BEFORE_TLD_RE = re.compile(
    r"\b(?:premium|extended|brandable|startup|fintech|saas|crypto|cloud|devtools|"
    r"edtech|healthcare|climate|gaming)\s+(?P<tok>ai|io|app|dev)\b"
    r"|\b(?P<tok2>ai)\s+(?:agent|names?|gems?|inventory|compan(?:y|ies)|products?|"
    r"startups?|brands?|tools?|niches?|meets)\b"
    r"|\b(?P<tok3>ai)\s+\.(?:com|net|org|io)\b"
    r"|\b(?P<tok4>ai)\s+or\s+(?:fintech|saas|startup|devtools)\b",
    re.IGNORECASE,
)


def reconcile_contrastive_preference_empty(
    query: str, entities: List[Entity],
) -> List[Entity]:
    """Wipe invents for contrastive preference ORs (aged|fresh, exact|brandable)."""
    if not query or not entities:
        return entities
    if re.search(
        r"\baged\s+domains?\s+or\s+fresh(?:\s+domains?)?\b"
        r"|\bfresh\s+domains?\s+or\s+aged(?:\s+domains?)?\b",
        query,
        re.IGNORECASE,
    ):
        drop = {
            'domain_age_min', 'domain_age_max', 'minAge', 'maxAge',
            'has_web_traffic_signal', 'topic_include',
        }
        cleaned = [e for e in entities if e.name not in drop]
        if len(cleaned) != len(entities):
            logger.info("entity_reconcile_aged_or_fresh_emptied")
        return cleaned
    if re.search(r"\bexact\s+match\s+or\s+brandable\b", query, re.IGNORECASE):
        out: List[Entity] = []
        for e in entities:
            if e.name != 'topic_include':
                out.append(e)
                continue
            raw = e.value
            if isinstance(raw, (list, tuple)):
                kept = [v for v in raw if str(v).lower() != 'seo']
                if not kept:
                    continue
                out.append(e if kept == list(raw) else replace(e, value=kept))
            elif str(raw).lower() == 'seo':
                continue
            else:
                out.append(e)
        if len(out) != len(entities):
            logger.info("entity_reconcile_exact_or_brandable_topic_scrubbed")
        return out
    if re.search(r"\bbetter\s+expired\s+or\s+auction\b", query, re.IGNORECASE):
        rest = [
            e for e in entities
            if e.name not in (
                'lifecycle_state', 'lifecycle_disjunction', 'auction_type',
            )
        ]
        rest.append(_entity('auction_type', ['expiry', 'listed']))
        logger.info("entity_reconcile_better_expired_or_auction_type")
        return rest
    return entities


def reconcile_or_under_inclusive(query: str, entities: List[Entity]) -> List[Entity]:
    """Force 'N or under' → inclusive price_max=N (not exclusive N-1)."""
    if not query:
        return entities
    m = re.search(
        r"\b(?P<n>\d[\d,]*)\s*(?P<k>k|thousand|million)?\s+or\s+(?:under|below|less)\b",
        query,
        re.IGNORECASE,
    )
    if m is None:
        return entities
    try:
        n = int(str(m.group('n')).replace(',', ''))
    except (TypeError, ValueError):
        return entities
    scale = (m.group('k') or '').lower()
    if scale in ('k', 'thousand'):
        n *= 1000
    elif scale == 'million':
        n *= 1_000_000
    out: List[Entity] = []
    saw = False
    for e in entities:
        if e.name != 'price_max':
            out.append(e)
            continue
        saw = True
        out.append(replace(e, value=n) if e.value != n else e)
    if not saw:
        out.append(_entity('price_max', n))
        logger.info(f"entity_reconcile_or_under_inclusive_injected value={n}")
    return out


def reconcile_under_price_inject(query: str, entities: List[Entity]) -> List[Entity]:
    """Inject exclusive price_max from under/below N when omitted."""
    if not query:
        return entities
    if any(e.name == 'price_max' for e in entities):
        return entities
    # Inclusive "N or under" owned elsewhere.
    if re.search(r"\b\d[\d,]*\s*(?:k|thousand)?\s+or\s+(?:under|below|less)\b", query, re.IGNORECASE):
        return entities
    m = re.search(
        r"\b(?:under|below|less\s+than)\s+\$?(?P<n>\d[\d,]*)\s*(?P<k>k|thousand|million)?\b",
        query,
        re.IGNORECASE,
    )
    if m is None:
        return entities
    try:
        n = int(str(m.group('n')).replace(',', ''))
    except (TypeError, ValueError):
        return entities
    scale = (m.group('k') or '').lower()
    if scale in ('k', 'thousand'):
        n *= 1000
    elif scale == 'million':
        n *= 1_000_000
    hi = max(0, n - 1)
    logger.info(f"entity_reconcile_under_price_injected value={hi}")
    return list(entities) + [_entity('price_max', hi)]


def reconcile_budget_prefixed_price_inject(
    query: str, entities: List[Entity],
) -> List[Entity]:
    """Inject inclusive price_max (band min+max for around/at) from ``in $50`` cues.

    Shares ``parse_budget_prefixed_price`` with qie_only ``filters_to_identified`` so
    full-search regex/L0_fallback cannot miss a cue the slim path already surfaces.
    """
    if not query:
        return entities
    has_max = any(e.name == 'price_max' for e in entities)
    has_min = any(e.name == 'price_min' for e in entities)
    if has_max and has_min:
        return entities
    parsed = parse_budget_prefixed_price(query)
    if parsed is None:
        return entities
    n, is_band = parsed
    out = list(entities)
    if is_band:
        if not has_min:
            out.append(_entity('price_min', n))
        if not has_max:
            out.append(_entity('price_max', n))
    elif not has_max:
        out.append(_entity('price_max', n))
    logger.info(
        f"entity_reconcile_budget_prefixed_price_injected value={n} band={is_band}"
    )
    return out


def reconcile_traffic_unknown_zero_band(
    query: str, entities: List[Entity],
) -> List[Entity]:
    """traffic unknown or zero → traffic_is_unknown + traffic_max=0; new brand → age 0."""
    if not query or not re.search(
        r"\btraffic\s+unknown\s+or\s+zero\b|\bunknown\s+or\s+zero\s+traffic\b",
        query,
        re.IGNORECASE,
    ):
        return entities
    out = list(entities)
    names = {e.name for e in out}
    if 'traffic_is_unknown' not in names:
        out.append(_entity('traffic_is_unknown', True))
    if 'traffic_max' not in names and 'maxTraffic' not in names:
        out.append(_entity('traffic_max', 0))
    if re.search(r"\b(?:new\s+brand|brand\s+new)\b", query, re.IGNORECASE):
        if 'domain_age_min' not in names and 'minAge' not in names:
            out.append(_entity('domain_age_min', 0))
    return out


def reconcile_premium_extension_slots(
    query: str, entities: List[Entity],
) -> List[Entity]:
    """premium extension → isExtended + auction_type=premium; keep under-N + short."""
    if not query or not re.search(r"\bpremium\s+extension\b", query, re.IGNORECASE):
        return entities
    out = list(entities)
    names = {e.name for e in out}
    if 'isExtended' not in names:
        out.append(_entity('isExtended', True))
    if 'auction_type' not in names:
        out.append(_entity('auction_type', 'premium'))
    return out


def reconcile_numeric_backlinks_majestic(
    query: str, entities: List[Entity],
) -> List[Entity]:
    """'N backlinks majestic' → majestic_backlinks_min=N when omitted."""
    if not query:
        return entities
    m = re.search(
        r"\b(?P<n>\d[\d,]*)\s*(?P<k>k)?\s+backlinks?\s+majestic\b"
        r"|\bmajestic\s+(?P<n2>\d[\d,]*)\s*(?P<k2>k)?\s+backlinks?\b"
        r"|\b(?P<n3>\d[\d,]*)\s*(?P<k3>k)?\s+majestic\s+backlinks?\b",
        query,
        re.IGNORECASE,
    )
    if m is None:
        return entities
    if any(
        e.name in (
            'majestic_backlinks_min', 'minMajesticBackLinks',
            'semrush_backlinks_min',
        )
        for e in entities
    ):
        return entities
    raw = m.group('n') or m.group('n2') or m.group('n3')
    scale = (m.group('k') or m.group('k2') or m.group('k3') or '').lower()
    try:
        n = int(str(raw).replace(',', ''))
    except (TypeError, ValueError):
        return entities
    if scale == 'k':
        n *= 1000
    logger.info(f"entity_reconcile_majestic_backlinks_injected value={n}")
    return list(entities) + [_entity('majestic_backlinks_min', n)]


def reconcile_leading_niche_topic(query: str, entities: List[Entity]) -> List[Entity]:
    """Inject topic_include for leading niche tokens (seo domains, fintech names, …)."""
    from semantic_search.qi.l0_llm_filter_extractor import (
        _AUDIENCE_TOPIC_RE,
        _INDUSTRY_TOPIC_LEXICON,
        _LEADING_NICHE_TOPIC_RE,
    )
    if not query or any(e.name == 'topic_include' for e in entities):
        return entities
    m = _LEADING_NICHE_TOPIC_RE.search(query) or _AUDIENCE_TOPIC_RE.search(query)
    if m is None:
        return entities
    tok = re.sub(r'\s+', '_', (m.group('topic') or '').strip().lower())
    if not tok or tok not in _INDUSTRY_TOPIC_LEXICON:
        return entities
    logger.info(f"entity_reconcile_leading_niche_topic_injected topic={tok}")
    return list(entities) + [_entity('topic_include', [tok])]


def reconcile_meets_cotopic(query: str, entities: List[Entity]) -> List[Entity]:
    """Ensure topic_include carries both sides of 'X meets Y' co-topic cue."""
    from semantic_search.qi.l0_llm_filter_extractor import _MEETS_COTOPIC_RE
    if not query:
        return entities
    m = _MEETS_COTOPIC_RE.search(query)
    if m is None:
        return entities
    need = sorted({
        (m.group('a') or '').strip().lower(),
        (m.group('b') or '').strip().lower(),
    } - {''})
    if not need:
        return entities
    out: List[Entity] = []
    saw = False
    for e in entities:
        if e.name != 'topic_include':
            out.append(e)
            continue
        saw = True
        raw = e.value
        if isinstance(raw, (list, tuple)):
            parts = [str(p).strip().lower() for p in raw if str(p).strip()]
        else:
            parts = [
                p.strip().lower()
                for p in str(raw or '').replace(',', '|').split('|')
                if p.strip()
            ]
        merged = sorted(set(parts) | set(need))
        out.append(_entity('topic_include', merged, source=e.source))
    if not saw:
        out.append(_entity('topic_include', need))
    return out


def reconcile_b2b_topic_shapes(query: str, entities: List[Entity]) -> List[Entity]:
    """Normalize b2b startup|tech co-topics vs venture-backed B2B software → b2b_saas."""
    if not query:
        return entities
    # "b2b startup or tech" → b2b|saas|tech
    if re.search(
        r"\bb2b\s+startup\b.{0,48}\bor\b.{0,48}\btech\b",
        query,
        re.IGNORECASE,
    ):
        out: List[Entity] = []
        saw = False
        for e in entities:
            if e.name != 'topic_include':
                out.append(e)
                continue
            saw = True
            out.append(_entity('topic_include', ['b2b', 'saas', 'tech'], source=e.source))
        if not saw:
            out.append(_entity('topic_include', ['b2b', 'saas', 'tech']))
        return out
    # Venture-backed B2B software / modern b2b saas → collapse to b2b_saas only.
    # Skip when disjunctive "b2b saas or (maybe) fintech" — that merge belongs to
    # reconcile_b2b_saas_or_fintech (must not early-return with b2b_saas alone).
    if _B2B_SAAS_OR_FINTECH_RE.search(query):
        return entities
    if re.search(
        r"\b(?:venture[- ]backed\s+)?b2b\s+software\b|\bb2b\s+saas\b",
        query,
        re.IGNORECASE,
    ):
        out = []
        saw = False
        for e in entities:
            if e.name != 'topic_include':
                out.append(e)
                continue
            saw = True
            out.append(_entity('topic_include', ['b2b_saas'], source=e.source))
        if not saw:
            out.append(_entity('topic_include', ['b2b_saas']))
        return out
    return entities


_B2B_SAAS_OR_FINTECH_RE = re.compile(
    r"\bb2b\s+saas\s+or\s+(?:maybe\s+)?fintech\b"
    r"|\bfintech\s+or\s+(?:maybe\s+)?b2b\s+saas\b"
    r"|\bb2b\s+saas\s+or\s+fintech\b",
    re.IGNORECASE,
)
_SAAS_STARTUP_NAME_RE = re.compile(
    r"\bsaas\s+start(?:up|p)\s+(?:name|names|domain|domains)\b",
    re.IGNORECASE,
)


def reconcile_b2b_saas_or_fintech(query: str, entities: List[Entity]) -> List[Entity]:
    """Force topic_include=b2b_saas|fintech for disjunctive B2B SaaS / fintech cues."""
    if not query or not _B2B_SAAS_OR_FINTECH_RE.search(query):
        return entities
    out: List[Entity] = []
    saw = False
    for e in entities:
        if e.name != 'topic_include':
            out.append(e)
            continue
        saw = True
        raw = e.value
        if isinstance(raw, (list, tuple)):
            parts = [str(p).strip().lower() for p in raw if str(p).strip()]
        else:
            parts = [
                p.strip().lower()
                for p in str(raw or '').replace(',', '|').split('|')
                if p.strip()
            ]
        merged = sorted(set(parts) | {'b2b_saas', 'fintech'})
        out.append(_entity('topic_include', merged, source=e.source))
    if not saw:
        out.append(_entity('topic_include', ['b2b_saas', 'fintech']))
    return out


def reconcile_saas_startup_compound_topic(query: str, entities: List[Entity]) -> List[Entity]:
    """``saas startp name`` → topic_include=saas|startup (typo-normalized upstream)."""
    if not query or not _SAAS_STARTUP_NAME_RE.search(query):
        return entities
    out: List[Entity] = []
    saw = False
    for e in entities:
        if e.name != 'topic_include':
            out.append(e)
            continue
        saw = True
        raw = e.value
        if isinstance(raw, (list, tuple)):
            parts = [str(p).strip().lower() for p in raw if str(p).strip()]
        else:
            parts = [
                p.strip().lower()
                for p in str(raw or '').replace(',', '|').split('|')
                if p.strip()
            ]
        merged = sorted(set(parts) | {'saas', 'startup'})
        out.append(_entity('topic_include', merged, source=e.source))
    if not saw:
        out.append(_entity('topic_include', ['saas', 'startup']))
    return out


def reconcile_ai_or_fintech_topic(query: str, entities: List[Entity]) -> List[Entity]:
    """Force topic_include=ai|fintech for ai-or-fintech co-topic; drop tld=ai steal."""
    if not query or not re.search(
        r"\bai\s+or\s+fintech\b|\bfintech\s+or\s+ai\b"
        r"|\b(?:either\s+)?ai\s+startup\s+or\s+fintech\b"
        r"|\bai\s+and\s+fintech\b|\bfintech\s+and\s+ai\b"
        r"|\bbrowse\s+ai\s+and\s+fintech\b",
        query,
        re.IGNORECASE,
    ):
        return entities
    out: List[Entity] = []
    saw_topic = False
    for e in entities:
        if e.name == 'topic_include':
            saw_topic = True
            raw = e.value
            parts = list(raw) if isinstance(raw, (list, tuple)) else (
                [p.strip().lower() for p in str(raw).replace(',', '|').split('|') if p.strip()]
            )
            # Drop startup when "ai startup or fintech" — co-topic owns ai|fintech.
            parts = [p for p in parts if p not in ('startup',)]
            merged = sorted(set(str(p).lower() for p in parts) | {'ai', 'fintech'})
            out.append(_entity('topic_include', merged, source=e.source))
            continue
        if e.name == 'tld':
            raw = e.value
            if isinstance(raw, (list, tuple)):
                kept = [v for v in raw if str(v).lower().lstrip('.') != 'ai']
                if not kept:
                    continue
                out.append(e if kept == list(raw) else replace(e, value=kept))
            elif str(raw).lower().lstrip('.') == 'ai':
                continue
            else:
                out.append(e)
            continue
        out.append(e)
    if not saw_topic:
        out.append(_entity('topic_include', ['ai', 'fintech']))
    return out


def reconcile_scrub_quality_char_pattern(
    query: str, entities: List[Entity],
) -> List[Entity]:
    """Drop charPattern=pronounceable/typeable quality words (not cvcv mnemonics)."""
    if not entities:
        return entities
    out: List[Entity] = []
    for e in entities:
        if e.name != 'charPattern':
            out.append(e)
            continue
        raw = str(e.value or '').lower()
        if raw in ('pronounceable', 'typeable', 'brandable', 'catchy') or (
            '|' not in raw and not re.fullmatch(r'[vcn]{2,16}', raw)
        ):
            logger.info(f"entity_reconcile_quality_char_pattern_dropped value={e.value!r}")
            continue
        out.append(e)
    return out


def reconcile_scrub_keyword_not(query: str, entities: List[Entity]) -> List[Entity]:
    """Drop keyword_contains=not from 'not keyword spam' polarity."""
    if not query or not entities:
        return entities
    if not re.search(r"\bnot\s+keyword\b", query, re.IGNORECASE):
        return entities
    out = []
    for e in entities:
        if e.name != 'keyword_contains':
            out.append(e)
            continue
        if e.value == 'not' or e.value == ['not']:
            continue
        if isinstance(e.value, list):
            kept = [v for v in e.value if str(v).lower() != 'not']
            if not kept:
                continue
            out.append(e if kept == e.value else replace(e, value=kept))
        else:
            out.append(e)
    return out


def reconcile_scrub_startup_co_topic(query: str, entities: List[Entity]) -> List[Entity]:
    """Drop leftover topic=startup when ai|fintech co-topic owns the OR."""
    if not query or not entities:
        return entities
    if not re.search(
        r"\b(?:ai\s+startup\s+or\s+fintech|fintech\s+or\s+ai\s+startup)\b",
        query,
        re.IGNORECASE,
    ):
        return entities
    out = []
    for e in entities:
        if e.name != 'topic_include':
            out.append(e)
            continue
        raw = e.value
        if isinstance(raw, (list, tuple)):
            kept = [v for v in raw if str(v).lower() != 'startup']
            if not kept:
                continue
            out.append(e if kept == list(raw) else replace(e, value=kept))
        elif str(raw).lower() == 'startup':
            continue
        else:
            out.append(e)
    return out


def reconcile_topicish_tld_vs_topic(query: str, entities: List[Entity]) -> List[Entity]:
    """Keep topic_include for niche-modified short tokens; drop matching lone tld chip.

    ``extended ai auction`` / ``premium ai domain`` / ``ai agent`` → topic_include=ai (not tld=ai).
    Matches L0 ``_normalize_tld_vs_topic`` topicish keep rule.
    """
    if not query:
        return entities
    keep = set()
    for m in _TOPICISH_BEFORE_TLD_RE.finditer(query):
        tok = m.group('tok') or m.group('tok2') or m.group('tok3') or m.group('tok4')
        if tok:
            keep.add(tok.lower())
        # Named groups may miss; pull ai from "ai .com" / "ai or fintech" blobs.
        blob = (m.group(0) or '').lower()
        if re.search(r'\bai\b', blob):
            keep.add('ai')
    if not keep:
        return entities
    out: List[Entity] = []
    has_topic = False
    for e in entities:
        if e.name == 'topic_include':
            has_topic = True
            out.append(e)
            continue
        if e.name != 'tld':
            out.append(e)
            continue
        raw = e.value
        if isinstance(raw, (list, tuple)):
            filtered = [
                v for v in raw
                if not (isinstance(v, str) and v.lower().lstrip('.') in keep)
            ]
        else:
            tok = str(raw or '').lower().lstrip('.')
            filtered = [] if tok in keep else [raw]
        if not filtered:
            logger.info(
                f"entity_reconcile_topicish_tld_dropped keep_topic={sorted(keep)}"
            )
            continue
        out.append(e if filtered == raw else replace(e, value=list(filtered)))
    # Ensure topic_include carries the topicish token when tld was the only home.
    if not has_topic:
        for tok in sorted(keep):
            out.append(_entity('topic_include', [tok]))
            logger.info(f"entity_reconcile_topicish_topic_injected value={tok}")
    return out


def apply_post_merge_reconcile(
    query: str,
    entities: List[Entity],
    hard_entity_names: frozenset,
) -> List[Entity]:
    """Full post-merge scrub chain (exact, lifecycle, stale, char, metric family/range).

    :param hard_entity_names: frozenset - From qi.entity_slots.hard_entity_names (required)
    """
    if not isinstance(hard_entity_names, frozenset):
        raise ConfigurationError(
            "apply_post_merge_reconcile requires hard_entity_names frozenset from qi.entity_slots"
        )
    with hard_entity_names_context(hard_entity_names):
        # Typo-normalize so co-topic / lexicon cues see canonical tokens (fintec→fintech).
        from semantic_search.qi.l0_llm_filter_extractor import _normalize_query_typos
        query = _normalize_query_typos(query or '')
        # Analytics / guidance → empty (do not let enrichers re-inject filters).
        if is_strong_advisory(query):
            logger.info("entity_reconcile_strong_advisory_empty")
            return []
        if is_soft_advisory_no_inventory(query):
            logger.info("entity_reconcile_soft_advisory_empty")
            return []
        # Dual qualitative "high X and high Y" with no numbers → empty (rule-4 omit).
        if (
            re.search(
                r"\bhigh\s+(?:go\s*value|govalue|domain\s+authority|da|authority|"
                r"traffic|backlinks?|valuation)\b"
                r".{0,48}\bhigh\s+(?:go\s*value|govalue|domain\s+authority|da|"
                r"authority|traffic|backlinks?|valuation)\b",
                query,
                re.IGNORECASE,
            )
            and not re.search(r"\d", query)
        ):
            logger.info("entity_reconcile_dual_high_qualitative_empty")
            return []
        ents = reconcile_exact_match_mode(query, entities)
        ents = reconcile_lifecycle_disjunction(query, ents)
        ents = reconcile_stale_listing_polarity(query, ents)
        ents = reconcile_ungrounded_char_slots(query, ents)
        ents = reconcile_char_constraints(query, ents)
        ents = reconcile_letters_vs_sld_len(query, ents)
        ents = reconcile_bare_price_not_sld_len(query, ents)
        ents = reconcile_bare_price_not_domain_age(query, ents)
        ents = reconcile_digit_char_exclude_noise(ents)
        ents = reconcile_buy_now_vs_auction_type(query, ents)
        ents = reconcile_bid_accepted(query, ents)
        ents = reconcile_bid_count_bounds(query, ents)
        ents = reconcile_metric_family_overrides(query, ents)
        ents = reconcile_metric_ranges(query, ents)
        ents = reconcile_currency_vs_tld(query, ents)
        ents = reconcile_price_ceiling_only(query, ents)
        ents = reconcile_govalue_vs_price_under(query, ents)
        # Additive suite enrichers — inject-if-absent only (never overwrite).
        ents = reconcile_traffic_visitor_soft(query, ents)
        ents = reconcile_exact_letter_length(query, ents)
        ents = reconcile_price_between(query, ents)
        ents = reconcile_ending_urgency(query, ents)
        ents = reconcile_added_recently(query, ents)
        ents = reconcile_similar_to_brands(query, ents)
        ents = reconcile_investor_below_market(query, ents)
        ents = reconcile_no_vibe_exclude(query, ents)
        ents = reconcile_word_count_phrase(query, ents)
        ents = reconcile_char_pattern_or(query, ents)
        ents = reconcile_unknown_age_with_price(query, ents)
        ents = reconcile_pending_delete_lifecycle(query, ents)
        ents = reconcile_dropped_today_lifecycle(query, ents)
        ents = reconcile_available_now_lifecycle(query, ents)
        ents = reconcile_soft_or_link_traffic(query, ents)
        ents = reconcile_look_expensive_not_min_price(query, ents)
        ents = reconcile_doesnt_sound_like_not_similar(query, ents)
        ents = reconcile_standard_or_partner(query, ents)
        ents = reconcile_topicish_tld_vs_topic(query, ents)
        ents = reconcile_contrastive_preference_empty(query, ents)
        ents = reconcile_or_under_inclusive(query, ents)
        ents = reconcile_under_price_inject(query, ents)
        ents = reconcile_budget_prefixed_price_inject(query, ents)
        ents = reconcile_traffic_unknown_zero_band(query, ents)
        ents = reconcile_premium_extension_slots(query, ents)
        ents = reconcile_numeric_backlinks_majestic(query, ents)
        ents = reconcile_leading_niche_topic(query, ents)
        ents = reconcile_meets_cotopic(query, ents)
        ents = reconcile_b2b_topic_shapes(query, ents)
        ents = reconcile_b2b_saas_or_fintech(query, ents)
        ents = reconcile_saas_startup_compound_topic(query, ents)
        ents = reconcile_ai_or_fintech_topic(query, ents)
        ents = reconcile_scrub_quality_char_pattern(query, ents)
        ents = reconcile_scrub_keyword_not(query, ents)
        ents = reconcile_scrub_startup_co_topic(query, ents)
        ents = reconcile_scrub_lifecycle_spurious_backorder(query, ents)
        ents = reconcile_scrub_qualitative_metric_floors(query, ents)
        return ents


__all__ = [
    'KEYWORD_MATCH_MODE_STANDALONE',
    'apply_post_merge_reconcile',
    'hard_entity_names_context',
    'qualitative_floor_scrub_blocks_slot',
    'reconcile_added_recently',
    'reconcile_scrub_qualitative_metric_floors',
    'reconcile_bid_accepted',
    'reconcile_bid_count_bounds',
    'reconcile_buy_now_vs_auction_type',
    'reconcile_bare_price_not_domain_age',
    'reconcile_bare_price_not_sld_len',
    'reconcile_budget_prefixed_price_inject',
    'reconcile_char_constraints',
    'reconcile_char_pattern_or',
    'reconcile_currency_vs_tld',
    'reconcile_digit_char_exclude_noise',
    'reconcile_ending_urgency',
    'reconcile_exact_letter_length',
    'reconcile_exact_match_mode',
    'reconcile_govalue_vs_price_under',
    'reconcile_investor_below_market',
    'reconcile_letters_vs_sld_len',
    'reconcile_lifecycle_disjunction',
    'reconcile_metric_family_overrides',
    'reconcile_metric_ranges',
    'reconcile_no_vibe_exclude',
    'reconcile_pending_delete_lifecycle',
    'reconcile_price_between',
    'reconcile_price_ceiling_only',
    'reconcile_similar_to_brands',
    'reconcile_standard_or_partner',
    'reconcile_stale_listing_polarity',
    'reconcile_topicish_tld_vs_topic',
    'reconcile_traffic_visitor_soft',
    'reconcile_ungrounded_char_slots',
    'reconcile_unknown_age_with_price',
    'reconcile_word_count_phrase',
]
