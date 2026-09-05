"""Structured retriever — payload-filter-then-score backend.
Ships a `StructuredIndex` interface plus an in-memory implementation that
applies the standard filter slots (tld, price_min, price_max, auction_type,
name_length_max) directly. Production swaps in a Qdrant payload-filter
adapter (filterable HNSW) behind the same interface — the unified Qdrant
listing index serves filter / semantic / hybrid in a single round-trip; see

"""
import asyncio, json, os, time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Protocol

import dataclasses
from semantic_search.config.models import StructuredRetrievalConfig
from semantic_search.core.exceptions import RetrievalError
from semantic_search.core.logging_utils import get_logger
from semantic_search.core.validation import safe_float, safe_int
from semantic_search.retrieval.base import Retriever, slice_candidates
from semantic_search.contracts import AUCTION_TYPE_LABEL_TO_IDS, Candidate, CandidateSet, FilterConflict, QueryIntent, prefer_registrar_auction_values

logger = get_logger(__name__)

_CHAR_DERIVE_PATH = os.path.join(os.path.dirname(__file__), '..', 'qi', 'char_constraint_derive.json')


def _load_char_derive_when_missing() -> Dict[str, str]:
    """Load slot -> derive-strategy map from char_constraint_derive.json."""
    try:
        with open(_CHAR_DERIVE_PATH, encoding='utf-8') as fh:
            raw = json.load(fh)
        mapping = raw.get('derive_when_missing') if isinstance(raw, dict) else None
        if not isinstance(mapping, dict):
            return {}
        return {str(k): str(v) for k, v in mapping.items() if str(k).strip() and str(v).strip()}
    except (OSError, json.JSONDecodeError, TypeError, AttributeError) as exc:
        logger.warning(
            f"char_constraint_derive_load_failed path={_CHAR_DERIVE_PATH!r} "
            f"error_type={type(exc).__name__} error={exc}"
        )
        return {}


_CHAR_DERIVE_WHEN_MISSING: Dict[str, str] = _load_char_derive_when_missing()


def _derive_char_field(strategy: str, sld: str) -> Optional[int]:
    """Apply a named derive strategy to an SLD. Unknown strategy -> None."""
    if strategy == 'contains_hyphen':
        return 1 if '-' in sld else 0
    if strategy == 'contains_digit':
        return 1 if any(ch.isdigit() for ch in sld) else 0
    return None


class WordSegmenter(Protocol):
    """Minimal protocol for a domain SLD segmenter.

    :meth:`segment` must accept a dot-separated domain string and return an
    object whose ``tokens`` attribute is a tuple/sequence of string tokens
    (the TLD is included as the last element; callers count only the non-TLD
    tokens to derive the word count for the SLD).
    """

    def segment(self, domain: str) -> Any:
        """Segment domain into tokens; result has a ``tokens`` attribute.
        :param domain: str - Domain string (sld + placeholder TLD, or full domain)
        :return: Any - Object with ``tokens: Tuple[str, ...]``
        """
        ...  # pragma: no cover


def normalize_tld(value: Any) -> str:
    """Canonical TLD comparison key: stringified, lowercased, leading dot stripped.
    Single normalizer shared by the structured filter, the Qdrant filter, and the
    orchestrator hard-chip gate so '.com', 'COM', and 'com' all compare equal.
    :param value: Any - Raw TLD value (entity value or payload field)
    :return: str - Normalized key ('com', 'io')
    """
    return str(value).strip().lower().lstrip('.')


def _field_int_or_none(item: Dict[str, Any], field: str) -> Optional[int]:
    """Return the integer value of ``field`` from ``item``, or None when absent/null.

    Used by ``_matches`` for enrichment fields that may be None when the source
    Athena column was absent at data-build time.  Callers treat a None return
    as "data unavailable — exclude item from this filter".

    :param item: Dict[str, Any] - Payload dict
    :param field: str - Field name to look up
    :return: Optional[int]
    """
    val = item.get(field)
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _field_float_or_none(item: Dict[str, Any], field: str) -> Optional[float]:
    """Return the float value of ``field`` from ``item``, or None when absent/null/NaN.

    :param item: Dict[str, Any] - Payload dict
    :param field: str - Field name to look up
    :return: Optional[float]
    """
    val = item.get(field)
    if val is None:
        return None
    try:
        f = float(val)
        return None if f != f else f  # NaN -> None
    except (ValueError, TypeError):
        return None


# Auction types are stored as numeric string IDs in the index payload
# (CAST(auction_type_id AS VARCHAR): "16", "20", "38", "39"). Canonical labels
# extracted by L0_entity or entered via filter chips expand via the shared
# AUCTION_TYPE_LABEL_TO_IDS map (contracts.py) before comparison.
# List-valued filter slots. Scalar str values (e.g. L0 bare ``\"com\"`` /
# ``\"16\"``) must be wrapped before ``for x in value`` — otherwise Python
# iterates characters and MatchAny / hard-chip gate match nothing.
_LIST_FILTER_SLOTS: frozenset = frozenset({
    'tld', 'auction_type', 'tldExcludeList', 'typeExcludeList',
    'keyword_contains', 'keyword_starts_with', 'keyword_ends_with',
    'keyword_contains_exclude', 'topic_include', 'topic_exclude',
})


def _as_filter_list(value: Any) -> List[Any]:
    """Coerce a filter value to a list (scalar / pipe / CSV -> one list)."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if '|' in text:
            return [p.strip() for p in text.split('|') if p.strip()]
        if ',' in text:
            return [p.strip() for p in text.split(',') if p.strip()]
        return [text]
    return [value]


def _expand_auction_type_filter(values: Any) -> frozenset:
    """Expand a list of canonical or numeric auction-type values into the full
    set of numeric string IDs that the index payload uses.

    A numeric string ('16', '38') passes through unchanged.  A canonical label
    ('auction', 'expiry', 'closeout', 'buynow', 'premium', 'partner', 'godaddy')
    expands via the shared :data:`semantic_search.contracts.AUCTION_TYPE_LABEL_TO_IDS`.
    Registrar-scoped labels win over generics / opposing IDs via
    :func:`prefer_registrar_auction_values`. Unknown values pass through as-is.
    Accepts a scalar str (wraps via ``_as_filter_list``) so char-iteration cannot
    turn ``\"16\"`` into ``{\"1\",\"6\"}``.
    """
    result: set = set()
    for v in prefer_registrar_auction_values([str(x) for x in _as_filter_list(values)]):
        sv = str(v).lower()
        if sv in AUCTION_TYPE_LABEL_TO_IDS:
            result.update(AUCTION_TYPE_LABEL_TO_IDS[sv])
        else:
            result.add(sv)
    return frozenset(result)


# Match modes for multi-term keyword slots. 'any' = OR (term list satisfied
# when at least one term matches); 'all' = AND (every term must match).
KEYWORD_MATCH_MODES = frozenset({'any', 'all'})


def _term_in_compound(term: str, sld: str) -> bool:
    """True if *term* appears in *sld* as a word-unit, not as a prefix of a longer word.

    Rejects matches where the immediately following character equals the last character
    of the term (English consonant-doubling morphology: shop->shopping, run->running).
    Accepts matches at end-of-string or when the next character differs.

    Also requires a leading word-boundary: the match must start at position 0 or be
    preceded by a non-alphanumeric character (e.g. a hyphen). Without this, short
    terms like 'rent' would incorrectly match infix positions in unrelated compounds
    such as 'parent' or 'current'.
    """
    pos = sld.find(term)
    while pos != -1:
        if pos == 0 or not sld[pos - 1].isalnum():
            end = pos + len(term)
            next_ch = sld[end:end + 1]
            if not next_ch or next_ch != term[-1]:
                return True
        pos = sld.find(term, pos + 1)
    return False


def keyword_value_matches(sld: str, value: Any, op: str, mode: str) -> bool:
    """Return True iff ``sld`` satisfies a keyword constraint.

    ``op`` is one of 'contains' / 'starts_with' / 'ends_with'. ``value`` is a
    single term (str) or a list of terms; ``mode`` ('any'/'all') governs how a
    multi-term list combines. A single-term value is mode-agnostic. Blank terms
    are dropped; an all-blank value matches trivially (no constraint).

    :param sld: str - Lowercased second-level domain label
    :param value: Any - Term (str) or list of terms
    :param op: str - 'contains' / 'starts_with' / 'ends_with'
    :param mode: str - 'any' or 'all'
    :return: bool - Whether the SLD satisfies the constraint
    """
    terms = [value] if isinstance(value, str) else list(value)
    terms = [str(t).lower() for t in terms if str(t).strip()]
    if not terms:
        return True
    if op == 'starts_with':
        checks = [sld.startswith(t) for t in terms]
    elif op == 'ends_with':
        checks = [sld.endswith(t) for t in terms]
    else:  # contains — segment-first: exact hyphen-segment match OR word-unit substring
        _segs = sld.split('-')
        checks = [t in _segs or _term_in_compound(t, sld) for t in terms]
    return all(checks) if mode == 'all' else any(checks)


_FILTER_ENTITY_NAMES = frozenset({
    # Core — already wired
    'tld', 'price_min', 'price_max', 'auction_type',
    'name_length_max', 'quality_min', 'time_remaining_max',
    # Keyword matching on SLD
    'keyword_contains', 'keyword_starts_with', 'keyword_ends_with',
    # Match mode ('any'/'all') for multi-term keyword slots (passthrough str)
    'keyword_match_mode',
    # Bid count
    'bids_min', 'bids_max',
    # Domain provenance
    'domain_age_min', 'domain_age_max',
    'traffic_min', 'traffic_max',
    # Estimated raw value (GoValue dollars)
    'govalue_min', 'govalue_max',
    # Character constraints
    'has_hyphen', 'has_number', 'is_idn',
    'name_length_min',
    # Letter/digit character-count constraints (computed from derived SLD)
    'minLetters', 'excludeLetters', 'minDigits', 'maxDigits',
    # Gem domain quality flag
    'isGemDomain',
    # Majestic
    'majestic_tf_min', 'majestic_tf_max',
    'majestic_cf_min', 'majestic_cf_max',
    'majestic_backlinks_min', 'majestic_backlinks_max',
    'majestic_ref_domains_min', 'majestic_ref_domains_max',
    # TLF Insights
    'tlf_exact_match', 'tlf_keyword_regs_min', 'tlf_developed',
    # SEMrush
    'semrush_backlinks_min', 'semrush_backlinks_max',
    'semrush_indexed_pages_min', 'semrush_indexed_pages_max',
    'semrush_ref_domains_min', 'semrush_ref_domains_max',
    'semrush_authority_min', 'semrush_authority_max',
    'semrush_search_volume_min', 'semrush_search_volume_max',
    'semrush_cpc_min', 'semrush_cpc_max',
    # Exclusion filters
    'tldExcludeList', 'typeExcludeList', 'keyword_contains_exclude',
    # Exact-phrase match on SLD
    'keyword_phrase',
    # Word-count constraints (segmenter-backed)
    'word_count_min', 'word_count_max',
    # Marketplace recency (ISO 8601 UTC string; applied against optional listed_at epoch field)
    'startTimeAfter',
    # Traffic presence signal — True = item must have at least one non-null/non-zero traffic field.
    'has_web_traffic_signal',
    # "Select unknown" slots — keep items whose mapped enrichment field is None/absent.
    'traffic_is_unknown', 'domain_age_is_unknown',
    # Marketplace lifecycle state — expands to a set of allowed auction_type ids.
    'lifecycle_state',
    # Disjunction flag — when True, time_remaining_max and lifecycle_state are ORed.
    'lifecycle_disjunction',
    # Relative price — keep items priced below the per-TLD market baseline.
    'price_below_market',
    # Vector-only slots: drive ANN retrieval, not structured filters.
    # Listed here to prevent "unsupported_filter_slot" warnings; _matches ignores them.
    'similar_to',
    'topic_include',
    'topic_exclude',
    # Auction feature filters backed by auction_audit_cln columns.
    'buy_it_now',
    'buy_it_now_min',
    'buy_it_now_max',
    'has_reserve_price',
    'gd_transfer',
    # Absolute timing (ISO 8601 UTC; passed to FIND API / Qdrant filter)
    'endTimeAfter', 'endTimeBefore', 'startTimeBefore',
    # Relative recency (days since listing -> startTimeAfter / startTimeBefore ISO conversion)
    'days_listed_max', 'days_listed_min',
    # Auction feature flags (passed to FIND API; not in local payload index)
    'isExtended', 'isBidAccepted',
    # Price currency context for minPrice/maxPrice interpretation
    'filterPriceCurrency',
    # SLD character pattern (e.g. vccnn)
    'charPattern',
    # Seller / owner membership filters
    'ownerMemberIncludeList', 'ownerMemberExcludeList',
    # Demand signals (unique search count)
    'minUniqueSearches', 'maxUniqueSearches',
    # Estibot domain / extension counts
    'minEstibotDomainCount', 'maxEstibotDomainCount',
    'minEstibotDomainCountDev', 'maxEstibotDomainCountDev',
    'minEstibotExtCount', 'maxEstibotExtCount',
    'minEstibotExtCountDev', 'maxEstibotExtCountDev',
})


def item_below_market(item: Dict[str, Any], baselines: Dict[str, float]) -> bool:
    """Return True iff the item's price is below its per-TLD market baseline.

    Falls back to the '_global' baseline when the item's TLD has none. Returns
    True (no-op pass) when ``baselines`` is empty (resolver below sample floor /
    disabled) so the predicate never empties an otherwise-valid result set.

    :param item: Dict[str, Any] - Candidate payload (needs 'price', 'tld')
    :param baselines: Dict[str, float] - Per-TLD baseline + '_global'
    :return: bool - Whether the item is below market
    """
    if not baselines:
        return True
    price = safe_int(item.get('price'), -1)
    if price <= 0:
        return False
    tld = str(item.get('tld', '')).lower().lstrip('.')
    baseline = baselines.get(tld, baselines.get('_global'))
    if baseline is None:
        return True
    return price < float(baseline)

# Maps a lifecycle state to the auction_type ids that represent it (reuses the
# marketplace's existing auction-type taxonomy). An empty set means "no auction-type
# restriction" — 'active' is enforced by the ends_at baseline, not a type filter.
_LIFECYCLE_AUCTION_TYPES: Dict[str, frozenset] = {
    'expired': frozenset({'39'}),
    'recently_expired': frozenset({'39', '16', '38'}),
    'dropping': frozenset({'39'}),
    'active': frozenset(),
    'pending_delete': frozenset({'39'}),
    'redemption_grace': frozenset({'39'}),
    'lapsed_renewal': frozenset({'39', '16', '38'}),
    'transfer_lock': frozenset(),
    'auto_renew_disabled': frozenset({'39', '16', '38'}),
    'backorder_eligible': frozenset({'39'}),
    'registrar_hold': frozenset({'39'}),
}

_SECONDS_PER_DAY: int = 86400

# Maps each "select unknown" slot to the payload field whose None/absence it selects.
# Mirrors retrieval.structured.unknown_selectable_fields; the retriever may override
# this from config, but the mapping is also applied directly by _matches for the
# in-memory backend and tests.
_UNKNOWN_SELECTABLE: Dict[str, str] = {
    'traffic_is_unknown': 'monthly_traffic',
    'domain_age_is_unknown': 'domain_age_years',
}


def derive_sld_from_payload(item: Dict[str, Any]) -> str:
    """Return the lowercased SLD for a payload item.

    Uses the 'sld' payload field when present and non-None; otherwise derives
    the SLD by stripping the TLD suffix from 'domain_name'. Returns '' on failure.

    :param item: Dict[str, Any] - Item payload (needs 'sld' or 'domain_name'+'tld')
    :return: str - Lowercased SLD string (empty string on failure)
    """
    raw = item.get('sld')
    if raw is not None:
        return str(raw).lower()
    domain = str(item.get('domain_name', '')).lower()
    tld = normalize_tld(item.get('tld', ''))
    suffix = '.' + tld if tld else ''
    if suffix and domain.endswith(suffix):
        return domain[: -len(suffix)]
    return domain


class StructuredIndex:
    """Structured filter store interface."""

    def search(self, filters: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
        """Return up to top_k items (each a dict containing 'item_id', 'score', payload fields)."""
        raise NotImplementedError

    def iter_payloads(self) -> Iterable[Dict[str, Any]]:
        """Iterate over every indexed payload — used by the inventory percentile
        resolver. Backends that cannot enumerate cheaply should
        return an empty iterator; the resolver then falls back to its prior.
        :yields: Dict[str, Any] - One payload per indexed item
        """
        return iter(())


class InMemoryStructuredIndex(StructuredIndex):
    """In-memory filter index for tests and local bootstrapping."""

    def __init__(self):
        self._items: List[Dict[str, Any]] = []

    def iter_payloads(self) -> Iterable[Dict[str, Any]]:
        """Yield a defensive copy of every indexed item (for the percentile resolver)."""
        return [dict(item) for item in self._items]

    def add(self, item: Dict[str, Any]) -> None:
        """Add an item dict — must contain at minimum 'item_id' and 'score' (in [0,1]).
        :param item: Dict[str, Any] - Item attributes (tld, price, auction_type, name_length, ...)
        :raises RetrievalError: If item is missing required fields or score is out of range
        """
        if 'item_id' not in item or not item['item_id']:
            raise RetrievalError("InMemoryStructuredIndex item missing 'item_id'")
        if 'score' not in item:
            raise RetrievalError("InMemoryStructuredIndex item missing 'score'")
        try:
            score = float(item['score'])
        except (TypeError, ValueError) as e:
            raise RetrievalError(f"InMemoryStructuredIndex item score not numeric: {e}") from e
        if not 0.0 <= score <= 1.0:
            raise RetrievalError("InMemoryStructuredIndex item score must be in [0,1]")
        self._items.append(dict(item))

    @staticmethod
    def _matches(item: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check whether an item satisfies all filters.

        Enrichment fields (bid_count, domain_age_years, monthly_traffic, all
        Majestic/TLF/SEMrush fields, govalue_score, quality) may be None when
        the source Athena column was absent at data-build time.  When a filter
        targets such a field and its payload value is None the item is EXCLUDED
        (unknown data cannot satisfy a constraint).

        The ``time_remaining_max`` slot is evaluated against the item's
        ``ends_at`` payload field (float Unix epoch).  The item passes when
        ``0 <= ends_at - now <= time_remaining_max``.  Items past their close
        time and items with a None/unparseable ``ends_at`` are excluded.
        """
        if 'tld' in filters:
            wanted = {normalize_tld(t) for t in _as_filter_list(filters['tld'])}
            if normalize_tld(item.get('tld', '')) not in wanted:
                return False
        if 'auction_type' in filters:
            wanted = _expand_auction_type_filter(filters['auction_type'])
            if str(item.get('auction_type', '')).lower() not in wanted:
                return False
        if 'price_min' in filters:
            # Ask price (price_usd_amt) is used for price_min; current_bid_price may be 0 when no bids.
            # Only reject missing/unparseable (price < 0) or below the stated floor.
            price = safe_int(item.get('price'), -1)
            if price < 0 or price < int(filters['price_min']):
                return False
        if 'price_max' in filters:
            # Include zero-price (no-bid) domains under a max-price ceiling.
            price = safe_int(item.get('price'), -1)
            if price < 0 or price > int(filters['price_max']):
                return False
        if 'name_length_max' in filters or 'name_length_min' in filters:
            name_len = safe_int(item.get('name_length'), -1)
            if name_len < 0:
                _derived_sld = derive_sld_from_payload(item)
                name_len = len(_derived_sld) if _derived_sld else -1
            if 'name_length_max' in filters and (name_len < 0 or name_len > int(filters['name_length_max'])):
                return False
            if 'name_length_min' in filters and (name_len < 0 or name_len < int(filters['name_length_min'])):
                return False
        if 'quality_min' in filters:
            quality = _field_float_or_none(item, 'quality')
            if quality is None or quality < float(filters['quality_min']):
                return False
        if filters.get('lifecycle_disjunction'):
            # OR semantics: item passes when EITHER the time-window condition OR the
            # lifecycle-state condition is satisfied. Evaluated here so the individual
            # checks below are skipped for queries with disjunction.
            _disj_pass = False
            if 'time_remaining_max' in filters:
                _ends_at = _field_float_or_none(item, 'ends_at')
                if _ends_at is not None:
                    _rem = _ends_at - time.time()
                    if 0.0 <= _rem <= float(filters['time_remaining_max']):
                        _disj_pass = True
            if not _disj_pass and 'lifecycle_state' in filters:
                _lc_map_d = filters.get('_lifecycle_map') or _LIFECYCLE_AUCTION_TYPES
                _allowed_d = _lc_map_d.get(str(filters['lifecycle_state']).lower())
                if not _allowed_d or str(item.get('auction_type', '')).lower() in _allowed_d:
                    _disj_pass = True
            if not _disj_pass:
                return False
        else:
            if 'time_remaining_max' in filters:
                ends_at = _field_float_or_none(item, 'ends_at')
                if ends_at is None:
                    return False
                remaining = ends_at - time.time()
                if remaining < 0.0 or remaining > float(filters['time_remaining_max']):
                    return False
        # Keyword matching on SLD — each slot may carry a single term or a term
        # list; 'keyword_match_mode' ('any'/'all') governs multi-term lists.
        # Mode is required only when mode-governed keyword slots are present
        # (hard-chip gate / explore post-filter call _matches with tld/price only).
        # derive_sld_from_payload falls back to domain_name when sld field is absent.
        sld = derive_sld_from_payload(item)
        _kw_mode_slots = (
            'keyword_contains' in filters
            or 'keyword_starts_with' in filters
            or 'keyword_ends_with' in filters
        )
        kw_mode = filters.get('keyword_match_mode')
        if _kw_mode_slots:
            if kw_mode not in ('any', 'all'):
                raise RetrievalError(
                    "keyword_match_mode must be 'any' or 'all' when keyword slots "
                    f"are active; got {kw_mode!r}"
                )
            if 'keyword_contains' in filters:
                if not keyword_value_matches(sld, filters['keyword_contains'], 'contains', kw_mode):
                    return False
            if 'keyword_starts_with' in filters:
                if not keyword_value_matches(sld, filters['keyword_starts_with'], 'starts_with', kw_mode):
                    return False
            if 'keyword_ends_with' in filters:
                if not keyword_value_matches(sld, filters['keyword_ends_with'], 'ends_with', kw_mode):
                    return False
        if 'keyword_phrase' in filters:
            phrase = str(filters['keyword_phrase']).lower().replace(' ', '').replace('-', '')
            if phrase not in sld.replace('-', ''):
                return False
        # Lifecycle state — require the item's auction_type to fall in the mapped set.
        # Skipped when lifecycle_disjunction=True (handled above with OR semantics).
        if 'lifecycle_state' in filters and not filters.get('lifecycle_disjunction'):
            _lc_map = filters.get('_lifecycle_map') or _LIFECYCLE_AUCTION_TYPES
            _allowed = _lc_map.get(str(filters['lifecycle_state']).lower())
            if _allowed:
                if str(item.get('auction_type', '')).lower() not in _allowed:
                    return False
        # "Select unknown" slots — keep the item only when the mapped enrichment
        # field is None/absent (inverse of the default "None excludes" rule).
        for _u_slot, _u_field in _UNKNOWN_SELECTABLE.items():
            if _u_slot in filters and bool(filters[_u_slot]):
                if item.get(_u_field) is not None:
                    return False
        # Exclusion filters on TLD, auction type, and keyword terms.
        if 'tldExcludeList' in filters:
            excluded_tlds = {normalize_tld(t) for t in _as_filter_list(filters['tldExcludeList'])}
            if normalize_tld(item.get('tld', '')) in excluded_tlds:
                return False
        if 'typeExcludeList' in filters:
            # Same label-to-ID expansion as includes so "exclude auction" matches
            # payload ids ('16') and label-shaped payloads ('AUCTION').
            excluded_types = _expand_auction_type_filter(filters['typeExcludeList'])
            item_types = _expand_auction_type_filter([item.get('auction_type', '')])
            if item_types & excluded_types:
                return False
        if 'keyword_contains_exclude' in filters:
            raw_exc = filters['keyword_contains_exclude']
            exclude_terms = [raw_exc] if isinstance(raw_exc, str) else list(raw_exc)
            for term in exclude_terms:
                t = str(term).lower()
                if len(t) <= 5:
                    for _seg in sld.split('-'):
                        if _seg.startswith(t) or _seg.endswith(t):
                            return False
                elif t in sld:
                    return False
        # Character constraints — prefer payload field; when missing, derive from
        # SLD via config strategies in char_constraint_derive.json (fail closed
        # when the slot has no strategy or derive returns None).
        for slot, field in (('has_hyphen', 'has_hyphen'), ('has_number', 'has_number'), ('is_idn', 'is_idn')):
            if slot not in filters:
                continue
            field_val = _field_int_or_none(item, field)
            if field_val is None:
                strategy = _CHAR_DERIVE_WHEN_MISSING.get(slot)
                if strategy:
                    field_val = _derive_char_field(strategy, derive_sld_from_payload(item))
            if field_val is None or field_val != int(bool(filters[slot])):
                return False
        # Bid count — None means column was absent; exclude item.
        if 'bids_min' in filters or 'bids_max' in filters:
            bids = _field_int_or_none(item, 'bid_count')
            if bids is None:
                return False
            if 'bids_min' in filters and bids < int(filters['bids_min']):
                return False
            if 'bids_max' in filters and bids > int(filters['bids_max']):
                return False
        # Domain age — None means column was absent; exclude item.
        if 'domain_age_min' in filters or 'domain_age_max' in filters:
            # Float comparison so sub-year ages (e.g. "older than 30 days" ≈ 0.082y) work.
            age = _field_float_or_none(item, 'domain_age_years')
            if age is None:
                return False
            if 'domain_age_min' in filters and age < float(filters['domain_age_min']):
                return False
            if 'domain_age_max' in filters and age > float(filters['domain_age_max']):
                return False
        # Traffic — None means column was absent; exclude item.
        # Skip numeric range check when traffic_is_unknown=1 (caller wants domains
        # with absent traffic data; numeric range and unknown-select are incompatible).
        if 'traffic_min' in filters or 'traffic_max' in filters:
            if not bool(filters.get('traffic_is_unknown')):
                traffic = _field_int_or_none(item, 'monthly_traffic')
                if traffic is None:
                    return False
                if 'traffic_min' in filters and traffic < int(filters['traffic_min']):
                    return False
                if 'traffic_max' in filters and traffic > int(filters['traffic_max']):
                    return False
        # Traffic presence signal — True = at least one traffic field must be non-null and > 0.
        # Fields checked are driven by config (_traffic_signal_fields injected by retrieve()).
        if 'has_web_traffic_signal' in filters and bool(filters['has_web_traffic_signal']):
            _ts_fields = filters.get('_traffic_signal_fields') or []
            _has_traffic = any(
                (_field_int_or_none(item, f) or 0) > 0 or (_field_float_or_none(item, f) or 0.0) > 0.0
                for f in _ts_fields
            )
            if not _has_traffic:
                return False
        # GoValue (raw estimated value) — None means valuationprice was NULL; exclude item.
        if 'govalue_min' in filters or 'govalue_max' in filters:
            gv = _field_float_or_none(item, 'govalue_score')
            if gv is None:
                return False
            if 'govalue_min' in filters and gv < float(filters['govalue_min']):
                return False
            if 'govalue_max' in filters and gv > float(filters['govalue_max']):
                return False
        # Majestic — None means column was absent; exclude item.
        for slot_min, slot_max, field in (
            ('majestic_tf_min', 'majestic_tf_max', 'majestic_tf'),
            ('majestic_cf_min', 'majestic_cf_max', 'majestic_cf'),
            ('majestic_backlinks_min', 'majestic_backlinks_max', 'majestic_backlinks'),
            ('majestic_ref_domains_min', 'majestic_ref_domains_max', 'majestic_ref_domains'),
        ):
            if slot_min in filters or slot_max in filters:
                val = _field_int_or_none(item, field)
                if val is None:
                    return False
                if slot_min in filters and val < int(filters[slot_min]):
                    return False
                if slot_max in filters and val > int(filters[slot_max]):
                    return False
        # TLF Insights — None means column was absent; exclude item.
        if 'tlf_exact_match' in filters:
            tlf_val = _field_int_or_none(item, 'tlf_exact_match')
            if tlf_val is None or tlf_val != int(bool(filters['tlf_exact_match'])):
                return False
        if 'tlf_developed' in filters:
            tlf_dev = _field_int_or_none(item, 'tlf_developed')
            if tlf_dev is None or tlf_dev != int(bool(filters['tlf_developed'])):
                return False
        if 'tlf_keyword_regs_min' in filters:
            tlf_regs = _field_int_or_none(item, 'tlf_keyword_regs')
            if tlf_regs is None or tlf_regs < int(filters['tlf_keyword_regs_min']):
                return False
        # SEMrush int fields — None means column was absent; exclude item.
        for slot_min, slot_max, field in (
            ('semrush_backlinks_min', 'semrush_backlinks_max', 'semrush_backlinks'),
            ('semrush_indexed_pages_min', 'semrush_indexed_pages_max', 'semrush_indexed_pages'),
            ('semrush_ref_domains_min', 'semrush_ref_domains_max', 'semrush_ref_domains'),
            ('semrush_search_volume_min', 'semrush_search_volume_max', 'semrush_search_volume'),
        ):
            if slot_min in filters or slot_max in filters:
                val = _field_int_or_none(item, field)
                if val is None:
                    return False
                if slot_min in filters and val < int(filters[slot_min]):
                    return False
                if slot_max in filters and val > int(filters[slot_max]):
                    return False
        # SEMrush float fields — None means column was absent; exclude item.
        for slot_min, slot_max, field in (('semrush_authority_min', 'semrush_authority_max', 'semrush_authority_score'), ('semrush_cpc_min', 'semrush_cpc_max', 'semrush_cpc')):
            if slot_min in filters or slot_max in filters:
                val = _field_float_or_none(item, field)
                if val is None:
                    return False
                if slot_min in filters and val < float(filters[slot_min]):
                    return False
                if slot_max in filters and val > float(filters[slot_max]):
                    return False
        # Character-count constraints — computed from derived SLD; skip when SLD is empty.
        if sld:
            if 'minLetters' in filters:
                letter_count = sum(1 for c in sld if c.isalpha())
                if letter_count < int(filters['minLetters']):
                    return False
            if 'excludeLetters' in filters:
                excluded_chars = set(str(filters['excludeLetters']).lower())
                if any(c in excluded_chars for c in sld):
                    return False
            if 'minDigits' in filters or 'maxDigits' in filters:
                digit_count = sum(1 for c in sld if c.isdigit())
                if 'minDigits' in filters and digit_count < int(filters['minDigits']):
                    return False
                if 'maxDigits' in filters and digit_count > int(filters['maxDigits']):
                    return False
        # Gem domain quality flag — payload field 'is_gem'; absent/None treated as False.
        if 'isGemDomain' in filters:
            want_gem = bool(filters['isGemDomain'])
            is_gem = bool(item.get('is_gem') or item.get('isGemDomain'))
            if want_gem and not is_gem:
                return False
        # Marketplace recency filter — listed_at is optional; SKIP (pass candidate through) when absent.
        # startTimeAfter is an ISO 8601 UTC string; listed_at is epoch seconds (float/int).
        # Skip-once tracking uses a mutable set injected by retrieve() as filters['_warned'].
        if 'startTimeAfter' in filters:
            listed_at = _field_float_or_none(item, 'listed_at')
            warned: Optional[set] = filters.get('_warned')  # type: ignore[assignment]
            if listed_at is None:
                if warned is not None and 'startTimeAfter_filter_skipped' not in warned:
                    logger.warning("startTimeAfter_filter_skipped reason=listed_at_field_absent")
                    warned.add('startTimeAfter_filter_skipped')
                # listed_at absent -> cannot apply constraint -> pass candidate through
            else:
                raw_cutoff = filters['startTimeAfter']
                try:
                    normalized = str(raw_cutoff).replace('Z', '+00:00')
                    cutoff_dt = datetime.fromisoformat(normalized)
                    if cutoff_dt.tzinfo is None:
                        cutoff_dt = cutoff_dt.replace(tzinfo=timezone.utc)
                    cutoff_epoch = cutoff_dt.timestamp()
                except (ValueError, TypeError) as exc:
                    if warned is not None and 'startTimeAfter_parse_failed' not in warned:
                        logger.warning(f"startTimeAfter_filter_skipped reason=malformed_cutoff value={raw_cutoff!r} error_type={type(exc).__name__}")
                        warned.add('startTimeAfter_parse_failed')
                else:
                    if listed_at < cutoff_epoch:
                        return False
        if 'days_listed_max' in filters:
            _dl_listed_at = _field_float_or_none(item, 'listed_at')
            if _dl_listed_at is not None:
                _dl_cutoff = datetime.now(timezone.utc).timestamp() - int(filters['days_listed_max']) * _SECONDS_PER_DAY
                if _dl_listed_at < _dl_cutoff:
                    return False
        if 'days_listed_min' in filters:
            # Listed at least N days ago -> listed_at must be on or before (now - N days).
            _dlmin_listed_at = _field_float_or_none(item, 'listed_at')
            if _dlmin_listed_at is not None:
                _dlmin_cutoff = datetime.now(timezone.utc).timestamp() - int(filters['days_listed_min']) * _SECONDS_PER_DAY
                if _dlmin_listed_at > _dlmin_cutoff:
                    return False
        if 'startTimeBefore' in filters:
            _stb_listed = _field_float_or_none(item, 'listed_at')
            if _stb_listed is not None:
                try:
                    _stb_raw = str(filters['startTimeBefore']).replace('Z', '+00:00')
                    _stb_dt = datetime.fromisoformat(_stb_raw)
                    if _stb_dt.tzinfo is None:
                        _stb_dt = _stb_dt.replace(tzinfo=timezone.utc)
                    if _stb_listed > _stb_dt.timestamp():
                        return False
                except (ValueError, TypeError):
                    pass
        if 'endTimeAfter' in filters or 'endTimeBefore' in filters:
            _ends = _field_float_or_none(item, 'ends_at')
            if _ends is None:
                return False
            if 'endTimeAfter' in filters:
                try:
                    _eta_raw = str(filters['endTimeAfter']).replace('Z', '+00:00')
                    _eta_dt = datetime.fromisoformat(_eta_raw)
                    if _eta_dt.tzinfo is None:
                        _eta_dt = _eta_dt.replace(tzinfo=timezone.utc)
                    if _ends < _eta_dt.timestamp():
                        return False
                except (ValueError, TypeError):
                    return False
            if 'endTimeBefore' in filters:
                try:
                    _etb_raw = str(filters['endTimeBefore']).replace('Z', '+00:00')
                    _etb_dt = datetime.fromisoformat(_etb_raw)
                    if _etb_dt.tzinfo is None:
                        _etb_dt = _etb_dt.replace(tzinfo=timezone.utc)
                    if _ends > _etb_dt.timestamp():
                        return False
                except (ValueError, TypeError):
                    return False
        # Boolean flag filters backed by auction_audit_cln columns.
        # None means the column was absent at data-build time -> exclude item.
        for _slot, _field in (
            ('buy_it_now', 'buy_it_now'),
            ('has_reserve_price', 'has_reserve_price'),
            ('gd_transfer', 'gd_transfer'),
        ):
            if _slot in filters:
                _fv = _field_int_or_none(item, _field)
                if _fv is None or _fv != int(bool(filters[_slot])):
                    return False
        # BuyItNow price range — requires the item to have a buy_it_now_price value.
        if 'buy_it_now_min' in filters or 'buy_it_now_max' in filters:
            _bin_price = _field_float_or_none(item, 'buy_it_now_price')
            if _bin_price is None:
                return False
            if 'buy_it_now_min' in filters and _bin_price < float(filters['buy_it_now_min']):
                return False
            if 'buy_it_now_max' in filters and _bin_price > float(filters['buy_it_now_max']):
                return False
        return True

    def search(self, filters: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
        if top_k < 1:
            return []
        matched = [item for item in self._items if self._matches(item, filters)]
        matched.sort(key=lambda it: float(it.get('score', 0.0)), reverse=True)
        return matched[:top_k]


def item_matches_filters(item: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """Public predicate for the canonical filter rules.

    Wraps :meth:`InMemoryStructuredIndex._matches` so callers outside this
    module (e.g. ExploreComposer's fallback path) can reuse the filter logic
    without depending on a private method.

    :param item: Dict[str, Any] - Inventory payload to test
    :param filters: Dict[str, Any] - Filter slot dict (output of
        :func:`extract_filters_from_intent`)
    :return: bool - True iff item satisfies every filter
    """
    return InMemoryStructuredIndex._matches(item, filters)


# Slots whose values must coerce to int before _matches can apply them.
_INT_SLOTS: frozenset = frozenset({
    'price_min', 'price_max',
    'name_length_max', 'name_length_min',
    'bids_min', 'bids_max',
    'traffic_min', 'traffic_max',
    'majestic_tf_min', 'majestic_tf_max',
    'majestic_cf_min', 'majestic_cf_max',
    'majestic_backlinks_min', 'majestic_backlinks_max',
    'majestic_ref_domains_min', 'majestic_ref_domains_max',
    'semrush_backlinks_min', 'semrush_backlinks_max',
    'semrush_indexed_pages_min', 'semrush_indexed_pages_max',
    'semrush_ref_domains_min', 'semrush_ref_domains_max',
    'semrush_search_volume_min', 'semrush_search_volume_max',
    'tlf_keyword_regs_min',
    'word_count_min', 'word_count_max',
    'minLetters', 'minDigits', 'maxDigits',
    'minUniqueSearches', 'maxUniqueSearches',
    'days_listed_max',
    'days_listed_min',
    'minEstibotDomainCount', 'maxEstibotDomainCount',
    'minEstibotDomainCountDev', 'maxEstibotDomainCountDev',
    'minEstibotExtCount', 'maxEstibotExtCount',
    'minEstibotExtCountDev', 'maxEstibotExtCountDev',
})

# Slots whose values must coerce to float before _matches can apply them.
_FLOAT_SLOTS: frozenset = frozenset({
    'quality_min',
    'time_remaining_max',
    'domain_age_min', 'domain_age_max',
    'govalue_min', 'govalue_max',
    'semrush_authority_min', 'semrush_authority_max',
    'semrush_cpc_min', 'semrush_cpc_max',
    'buy_it_now_min', 'buy_it_now_max',
})


def detect_filter_conflicts(
    filters: Dict[str, Any],
    messages: Dict[str, str],
    qualitative_conflict_rules: Optional[List[Dict[str, Any]]] = None,
) -> List[FilterConflict]:
    """Return contradictions between extracted filter slots.

    Detects (1) inverted numeric ranges — any ``<prefix>_min`` whose paired
    ``<prefix>_max`` is smaller; and (2) qualitative-quantitative conflicts — a
    qualitative signal slot (e.g. ``govalue_min`` from "premium-sounding") paired
    with an explicit ``price_max`` that falls below a config-defined threshold.

    The ``messages`` map supplies the explainer template per conflict kind;
    ``{field}`` / ``{min}`` / ``{max}`` (range) and ``{qualitative_slot}`` /
    ``{price_max}`` / ``{price_max_cap}`` (qualitative) are interpolated.
    Non-numeric slot values are skipped.

    Config-driven qualitative rules have shape:
      ``{qualitative_slot: str, price_max_cap: int}``
    A conflict fires when the slot is present in ``filters`` and the ``price_max``
    value is below ``price_max_cap``.

    :param filters: Dict[str, Any] - Output of `extract_filters_from_intent`
    :param messages: Dict[str, str] - Template per conflict kind
    :param qualitative_conflict_rules: Optional[List[Dict]] - Qualitative rules from config
    :return: List[FilterConflict] - One entry per detected contradiction
    """
    conflicts: List[FilterConflict] = []
    for key in sorted(filters.keys()):
        if not key.endswith('_min'):
            continue
        max_key = key[:-4] + '_max'
        if max_key not in filters:
            continue
        try:
            lo = float(filters[key])
            hi = float(filters[max_key])
        except (TypeError, ValueError):
            continue
        if lo > hi:
            field = key[:-4]
            template = messages.get('range_inverted') or "{field}: minimum {min} exceeds maximum {max}"
            msg = template.format(field=field, min=filters[key], max=filters[max_key])
            conflicts.append(FilterConflict(kind='range_inverted', slots=[key, max_key], message=msg))
    if qualitative_conflict_rules:
        try:
            price_max_num = float(filters['price_max'])
        except (KeyError, TypeError, ValueError):
            price_max_num = None
        if price_max_num is not None:
            for rule in qualitative_conflict_rules:
                q_slot = str(rule.get('qualitative_slot', ''))
                cap = rule.get('price_max_cap')
                if not q_slot or cap is None or q_slot not in filters:
                    continue
                try:
                    cap_num = float(cap)
                except (TypeError, ValueError):
                    continue
                if price_max_num < cap_num:
                    template = (
                        messages.get('qualitative_quantitative')
                        or "'{qualitative_slot}' signals premium value but price_max={price_max} is below the expected threshold of {price_max_cap}"
                    )
                    msg = template.format(
                        qualitative_slot=q_slot,
                        price_max=int(price_max_num),
                        price_max_cap=int(cap_num),
                    )
                    conflicts.append(FilterConflict(
                        kind='qualitative_quantitative',
                        slots=[q_slot, 'price_max'],
                        message=msg,
                    ))
    return conflicts


def extract_filters_from_intent(intent: QueryIntent) -> Dict[str, Any]:
    """Project the intent's grounded entities into a filter dict.
    Multi-intent queries collapse all slices' filter entities into one filter dict.
    Numeric slots whose entity value cannot be coerced to int/float are dropped
    with a WARNING — they must not propagate into _matches where bare int()/float()
    casts would raise ValueError.
    :param intent: QueryIntent - Classified intent
    :return: Dict[str, Any] - Filter dict consumable by `StructuredIndex.search`
    """
    filters: Dict[str, Any] = {}
    skipped: list = []
    for s in intent.slices:
        for ent in s.entities:
            if ent.name not in _FILTER_ENTITY_NAMES:
                skipped.append(ent.name)
                continue
            if ent.name in _INT_SLOTS:
                try:
                    filters[ent.name] = int(float(ent.value))
                except (ValueError, TypeError):
                    logger.warning(
                        f"extract_filters slot_dropped slot={ent.name} "
                        f"value={ent.value!r} reason=not_int request_id={intent.request_id}"
                    )
            elif ent.name in _FLOAT_SLOTS:
                try:
                    filters[ent.name] = float(ent.value)
                except (ValueError, TypeError):
                    logger.warning(
                        f"extract_filters slot_dropped slot={ent.name} "
                        f"value={ent.value!r} reason=not_float request_id={intent.request_id}"
                    )
            elif ent.name in _LIST_FILTER_SLOTS:
                filters[ent.name] = _as_filter_list(ent.value)
            else:
                filters[ent.name] = ent.value
    logger.debug(
        f"extract_filters request_id={intent.request_id} tier={intent.decision_tier} "
        f"total_entities={sum(len(s.entities) for s in intent.slices)} "
        f"filter_keys={sorted(filters.keys())} skipped_entities={skipped}"
    )
    return filters


# Hard-chip slots that select an explore rail (horizon/lifecycle) or drive ANN
# retrieval rather than name a payload field to match against. Excluded from
# hard-chip payload gating so a rail selector never drops rail items and a
# vector-only slot never triggers a payload comparison.
_NON_PAYLOAD_HARD_SLOTS: frozenset = frozenset({
    'time_remaining_max', 'startTimeAfter', 'lifecycle_state', 'lifecycle_disjunction',
    'similar_to', 'topic_include', 'topic_exclude',
})


def extract_hard_filters_from_intent(intent: QueryIntent) -> Dict[str, Any]:
    """Project only hard-chip, payload-verifiable entities into a filter dict.

    Same projection as `extract_filters_from_intent` but restricted to
    ``chip_kind == 'hard'`` and with rail-selector / vector-only slots removed.
    Used to enforce hard filters on backends that do NOT filter at query time —
    the orchestrator's post-fusion gate and the in-memory vector leg — so no
    backend leaks results that contradict an explicit hard constraint.

    :param intent: QueryIntent - Classified intent
    :return: Dict[str, Any] - Hard-chip filter dict consumable by `_matches`
    """
    hard_slices = [
        dataclasses.replace(
            s,
            entities=[e for e in s.entities if e.chip_kind == 'hard' and e.name not in _NON_PAYLOAD_HARD_SLOTS],
        )
        for s in intent.slices
    ]
    return extract_filters_from_intent(dataclasses.replace(intent, slices=hard_slices))


class StructuredRetriever(Retriever):
    """Filter-then-score retriever for `filter` and `hybrid` query types.

    :param config: StructuredRetrievalConfig - Tier-specific config
    :param index: StructuredIndex - Backing filter store
    :param word_segmenter: Optional[WordSegmenter] - Domain SLD segmenter used to
        apply word_count_min / word_count_max filters. When None, word-count
        filters are skipped with a WARNING (graceful degradation).
    """

    def __init__(self, config: StructuredRetrievalConfig, index: StructuredIndex, word_segmenter: Optional[WordSegmenter] = None, percentile_resolver: Optional[Any] = None) -> None:
        self._config = config
        self._index = index
        self._word_segmenter = word_segmenter
        # Duck-typed resolver exposing market_price_baselines(); injected by the
        # registry. Typed Any to avoid an import cycle (resolver imports this module).
        self._percentile_resolver = percentile_resolver

    def set_percentile_resolver(self, resolver: Any) -> None:
        """Inject the percentile resolver after construction (it depends on the index).
        :param resolver: Any - Object exposing ``market_price_baselines()``
        """
        self._percentile_resolver = resolver

    @property
    def source(self) -> str:
        return 'structured'

    def _count_sld_words(self, sld: str, _cache: Dict[str, int]) -> int:
        """Return the word count for an SLD string using the injected segmenter.

        Results are memoised in ``_cache`` (keyed on SLD) to avoid re-segmenting
        the same SLD multiple times within one retrieve call. The TLD token
        emitted by the segmenter is excluded from the count; only meaningful
        alphabetic / numeric word tokens are counted.

        :param sld: str - Second-level domain label (no TLD)
        :param _cache: Dict[str, int] - Per-call memoisation dict
        :return: int - Number of word tokens (0 on segmentation failure)
        """
        if sld in _cache:
            return _cache[sld]
        try:
            segmented = self._word_segmenter.segment(sld + ".placeholder")  # type: ignore[union-attr]
            # tokens[-1] is the TLD placeholder; all preceding tokens are SLD words.
            count = max(0, len(segmented.tokens) - 1)
        except (RuntimeError, ValueError, TypeError, KeyError, IndexError, AttributeError, OSError) as exc:
            logger.warning(f"word_count_segment_failed sld={sld!r} error_type={type(exc).__name__} error={exc}")
            count = 0
        _cache[sld] = count
        return count

    async def retrieve(self, intent: QueryIntent, top_k: int) -> CandidateSet:
        """Retrieve candidates matching the intent's filter slots.

        :param intent: QueryIntent - Classified intent
        :param top_k: int - Maximum candidates to return
        :return: CandidateSet - Filtered + scored candidates
        """
        if not self._config.enabled:
            return CandidateSet(source='structured', candidates=[], latency_ms=0.0)
        if top_k < 1:
            return CandidateSet(source='structured', candidates=[], latency_ms=0.0)
        filters = extract_filters_from_intent(intent)
        if not filters:
            return CandidateSet(source='structured', candidates=[], latency_ms=0.0)
        _detected = detect_filter_conflicts(filters, {})
        if _detected:
            intent.conflicts = list(intent.conflicts or []) + _detected
        t0 = time.monotonic()
        filters['_warned'] = set()
        filters['_lifecycle_map'] = {k: frozenset(v) for k, v in self._config.lifecycle_auction_type_map.items()}
        filters['_traffic_signal_fields'] = list(self._config.traffic_signal_fields)
        # Inject the config-default keyword combine mode when the intent did not
        # carry an explicit one, so multi-term keyword lists resolve consistently.
        if 'keyword_match_mode' not in filters and any(
            k in filters for k in ('keyword_contains', 'keyword_starts_with', 'keyword_ends_with')
        ):
            filters['keyword_match_mode'] = self._config.keyword_match_mode
        rows: List[Dict[str, Any]] = await asyncio.to_thread(self._index.search, filters, top_k)
        # Word-count post-filter — applied lazily only when the slots are present
        # AND the config toggle is on. Uses the injected segmenter; skipped with a
        # WARNING when the segmenter is not wired (graceful degradation).
        apply_wc = (
            self._config.word_count_filter_enabled
            and ('word_count_min' in filters or 'word_count_max' in filters)
        )
        if apply_wc:
            if self._word_segmenter is None:
                logger.warning(
                    f"word_count_filter_skipped reason=segmenter_not_wired "
                    f"request_id={intent.request_id}"
                )
            else:
                _seg_cache: Dict[str, int] = {}
                wc_min = int(filters['word_count_min']) if 'word_count_min' in filters else None
                wc_max = int(filters['word_count_max']) if 'word_count_max' in filters else None
                filtered_rows: List[Dict[str, Any]] = []
                for row in rows:
                    sld = str(row.get('sld', '')).lower()
                    wc = self._count_sld_words(sld, _seg_cache)
                    if wc_min is not None and wc < wc_min:
                        continue
                    if wc_max is not None and wc > wc_max:
                        continue
                    filtered_rows.append(row)
                rows = filtered_rows
        # Relative price post-filter — keep items priced below the per-TLD market
        # baseline. No-op when the slot is absent or the resolver is not wired.
        if 'price_below_market' in filters and bool(filters['price_below_market']) and self._percentile_resolver is not None:
            baselines = self._percentile_resolver.market_price_baselines()
            rows = [row for row in rows if item_below_market(row, baselines)]
        candidates: List[Candidate] = []
        for row in rows:
            payload = {k: v for k, v in row.items() if k not in {'item_id', 'score'}}
            candidates.append(Candidate(item_id=str(row['item_id']), score=float(row['score']), source='structured', payload=payload))
        candidates = slice_candidates(candidates, top_k)
        latency_ms = (time.monotonic() - t0) * 1000.0
        logger.info(f"structured_retrieval request_id={intent.request_id} candidates={len(candidates)} filters={sorted(filters.keys())} latency_ms={latency_ms:.1f}")
        return CandidateSet(source='structured', candidates=candidates, latency_ms=latency_ms)


__all__ = ['StructuredIndex', 'InMemoryStructuredIndex', 'StructuredRetriever', 'WordSegmenter', 'derive_sld_from_payload', 'extract_filters_from_intent', 'item_matches_filters']
