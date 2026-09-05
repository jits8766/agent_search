"""Qdrant adapters for the unified listing index.

Houses three concrete implementations behind the existing retrieval contracts:

- `QdrantVectorIndex`        implements `VectorIndex`        — vector ANN
- `QdrantStructuredIndex`    implements `StructuredIndex`    — payload-filter scan
- `QdrantHybridRetriever`    implements `Retriever`          — single round-trip
                                                              vector + payload
                                                              (+ optional BM25)
                                                              fusion via the
                                                              Qdrant Query API

Plus a `QdrantClientFactory` that mirrors `ClickHouseClient`'s construction
contract: the factory **never raises** when Qdrant is unreachable or the
`qdrant-client` package is missing — it logs a warning and exposes
`available=False`. The adapter methods then raise `QdrantUnavailableError`
which the `BackendHealthRegistry` + `DegradationPlanner` handle deterministically
(no boot-time crash, no orchestrator-level branching). This keeps the
production code path and the test path on a single orchestrator (testing.mdc
— no parallel test-only branches inside production logic).

Wiring (Option A):

When `retrieval.qdrant.hybrid.enabled=true` the registry constructs a
`QdrantHybridRetriever` and exposes it to `SearchOrchestrator` *as the vector
retriever* (`source='vector'`). The structured retriever is replaced by a
`QdrantNoOpStructuredRetriever` that returns an empty `CandidateSet` —
because the hybrid retriever has already applied the payload filter inline
with the vector traversal in a single round-trip. The orchestrator's
`_gather_candidates` fan-out is unchanged (existing contract preserved).

When `retrieval.qdrant.hybrid.enabled=false` the registry constructs the
classic dual-backend pair (`QdrantVectorIndex` + `QdrantStructuredIndex`) and
the orchestrator fans out exactly as it does for the in-memory backends.

All client calls run through `asyncio.to_thread` *only* when the underlying
client is sync. `qdrant-client>=1.10` ships an `AsyncQdrantClient` which is
natively awaitable; we await it directly to avoid the thread-pool hop.
"""
import asyncio
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

try:
    from qdrant_client import AsyncQdrantClient as _AsyncQdrantClient, models as _qm
except ImportError:
    _AsyncQdrantClient = None
    _qm = None

from semantic_search.config.models import FilterOnlyRailLegConfig, FilterOnlyRailsConfig, QdrantConfig
from semantic_search.core.exceptions import QdrantQueryError, QdrantUnavailableError, RetrievalError
from semantic_search.core.logging_utils import get_logger
from semantic_search.qi.encoder import Encoder
from semantic_search.retrieval.base import Retriever, slice_candidates
from semantic_search.qi.residual_extractor import semantic_encode_text_for
from semantic_search.retrieval.structured_retriever import (
    _LIFECYCLE_AUCTION_TYPES,
    _UNKNOWN_SELECTABLE,
    _as_filter_list,
    _expand_auction_type_filter,
    StructuredIndex,
    extract_filters_from_intent,
    keyword_value_matches,
    normalize_tld,
)
from semantic_search.retrieval.vector_retriever import VectorIndex
from semantic_search.contracts import Candidate, CandidateSet, QueryIntent

logger = get_logger(__name__)


_SECONDS_PER_DAY: int = 86400

# Boolean auction payload fields: slot name -> Qdrant payload key (0/1 integer fields).
# Add a row here to wire a new boolean auction column with no logic changes required.
_AUCTION_BOOLEAN_PAYLOAD_FIELDS: Tuple[Tuple[str, str], ...] = (
    ('buy_it_now', 'buy_it_now'),
    ('has_reserve_price', 'has_reserve_price'),
    ('gd_transfer', 'gd_transfer'),
)

# Filter translation: central, deterministic, fully testable (module scope).
# Unavailable keys: columns absent in indexed data (COALESCE to 0); updated by app.py.
_GLOBALLY_UNAVAILABLE_FILTER_KEYS: FrozenSet[str] = frozenset()


def set_unavailable_filter_keys(keys: Iterable[str]) -> None:
    """Update globally-unavailable filter keys (data-build calls after fetch)."""
    global _GLOBALLY_UNAVAILABLE_FILTER_KEYS
    _GLOBALLY_UNAVAILABLE_FILTER_KEYS = frozenset(keys)

def build_qdrant_filter(filters: Dict[str, Any], active_only: bool = False, price_gt_zero: bool = False, starting_bid_gt_zero: bool = False) -> Optional[Any]:
    """Translate entity filters to Qdrant Filter (mirrors InMemoryStructuredIndex._matches).
      - `time_remaining_max`-> Range(gte=now, lte=now+seconds) on payload['ends_at']
        (fuzzy "expiring" filter)

    Baseline flags (applied on top of user filters, cannot be dropped by ZeroResultGuard):
      - `active_only`       -> ends_at >= now() — excludes expired auctions
      - `price_gt_zero`     -> price > 0 when no explicit price filter is present
                               — excludes unpriced/zero-price items from fallback results.
                               NOTE: `price` is the CURRENT bid (0 = no bids yet), so this
                               also drops biddable no-bid auctions. Prefer `starting_bid_gt_zero`.
      - `starting_bid_gt_zero` -> starting_bid > 0 when no explicit price filter is present
                               — keeps biddable no-bid auctions (price=0) while still excluding
                               genuinely unpriced/parked items (starting_bid=0)

    Returns None when all conditions are empty so the caller can pass
    `query_filter=None` (Qdrant treats it as 'no filter').

    :param filters: Dict[str, Any] - Output of `extract_filters_from_intent`
    :param active_only: bool - When True, inject ends_at >= now() baseline
    :param price_gt_zero: bool - When True and no price filter present, inject price > 0
    :return: Optional[qm.Filter] - Filter object or None
    :raises RetrievalError: When `qdrant-client` is not installed (caller
        should treat this as a soft failure and route to the in-memory path)
    """
    # Drop filter keys whose source columns are absent in indexed data.
    if _GLOBALLY_UNAVAILABLE_FILTER_KEYS and filters:
        filters = {k: v for k, v in filters.items() if k not in _GLOBALLY_UNAVAILABLE_FILTER_KEYS}
    if not filters and not active_only and not price_gt_zero and not starting_bid_gt_zero:
        return None
    if _qm is None:
        raise RetrievalError("qdrant-client not installed; cannot build Qdrant filter")
    qm = _qm
    must: List[Any] = []
    must_not: List[Any] = []
    if 'tld' in filters:
        # Scalar str ("com") must not char-iterate — wrap via _as_filter_list.
        values = [normalize_tld(t) for t in _as_filter_list(filters['tld'])]
        must.append(qm.FieldCondition(key='tld', match=qm.MatchAny(any=values)))
    if 'auction_type' in filters:
        values = [str(a).lower() for a in _as_filter_list(filters['auction_type'])]
        must.append(qm.FieldCondition(key='auction_type', match=qm.MatchAny(any=values)))
    if 'price_min' in filters or 'price_max' in filters:
        rng_kwargs: Dict[str, Any] = {}
        if 'price_min' in filters:
            rng_kwargs['gte'] = float(filters['price_min'])
        else:
            # Keep biddable no-bid items (price=0); a Range still drops null/missing price.
            rng_kwargs['gte'] = 0.0
        if 'price_max' in filters:
            rng_kwargs['lte'] = float(filters['price_max'])
        must.append(qm.FieldCondition(key='price', range=qm.Range(**rng_kwargs)))
    else:
        # No user price filter present — apply the priced/biddable baseline(s).
        if price_gt_zero:
            # Legacy: exclude zero/null CURRENT-price items (also drops no-bid auctions).
            must.append(qm.FieldCondition(key='price', range=qm.Range(gt=0.0)))
        if starting_bid_gt_zero:
            # Keep biddable no-bid auctions (current price=0) while excluding
            # genuinely unpriced/parked items (starting_bid=0).
            must.append(qm.FieldCondition(key='starting_bid', range=qm.Range(gt=0.0)))
    if 'name_length_max' in filters:
        must.append(qm.FieldCondition(key='name_length', range=qm.Range(lte=float(filters['name_length_max']))))
    if 'name_length_min' in filters:
        must.append(qm.FieldCondition(key='name_length', range=qm.Range(gte=float(filters['name_length_min']))))
    if 'quality_min' in filters:
        must.append(qm.FieldCondition(key='quality', range=qm.Range(gte=float(filters['quality_min']))))
    if 'time_remaining_max' in filters:
        # "expiring" filter: now <= end_field <= now + seconds.
        # Compatibility OR: pre-reindex data uses 'end_time'; post-reindex uses 'ends_at'.
        now = float(time.time())
        lte = now + float(filters['time_remaining_max'])
        must.append(qm.Filter(should=[qm.FieldCondition(key='ends_at', range=qm.Range(gte=now, lte=lte)), qm.FieldCondition(key='end_time', range=qm.Range(gte=now, lte=lte))]))
    elif active_only:
        # Baseline: suppress expired auctions.
        # Compatibility OR: pre-reindex data uses 'end_time'; post-reindex uses 'ends_at'.
        now = float(time.time())
        must.append(qm.Filter(should=[qm.FieldCondition(key='ends_at', range=qm.Range(gte=now)), qm.FieldCondition(key='end_time', range=qm.Range(gte=now))]))
    # listed_at range: ISO startTimeAfter/Before + relative days_listed_min/max.
    if (
        'startTimeAfter' in filters
        or 'startTimeBefore' in filters
        or 'days_listed_max' in filters
        or 'days_listed_min' in filters
    ):
        _la_rng: Dict[str, Any] = {}
        if 'startTimeAfter' in filters:
            _sta_raw = str(filters['startTimeAfter']).replace('Z', '+00:00')
            try:
                _sta_dt = datetime.fromisoformat(_sta_raw)
                if _sta_dt.tzinfo is None:
                    _sta_dt = _sta_dt.replace(tzinfo=timezone.utc)
                _la_rng['gte'] = _sta_dt.timestamp()
            except (ValueError, TypeError):
                pass
        if 'startTimeBefore' in filters:
            _stb_raw = str(filters['startTimeBefore']).replace('Z', '+00:00')
            try:
                _stb_dt = datetime.fromisoformat(_stb_raw)
                if _stb_dt.tzinfo is None:
                    _stb_dt = _stb_dt.replace(tzinfo=timezone.utc)
                _la_rng['lte'] = _stb_dt.timestamp()
            except (ValueError, TypeError):
                pass
        if 'days_listed_max' in filters and 'gte' not in _la_rng:
            _la_rng['gte'] = datetime.now(timezone.utc).timestamp() - int(filters['days_listed_max']) * _SECONDS_PER_DAY
        if 'days_listed_min' in filters and 'lte' not in _la_rng:
            # Listed at least N days ago: listed_at <= now - N days.
            _la_rng['lte'] = datetime.now(timezone.utc).timestamp() - int(filters['days_listed_min']) * _SECONDS_PER_DAY
        if _la_rng:
            must.append(qm.FieldCondition(key='listed_at', range=qm.Range(**_la_rng)))
    # Absolute auction end window (ISO to ends_at epoch).
    if 'endTimeAfter' in filters or 'endTimeBefore' in filters:
        _et_rng: Dict[str, Any] = {}
        if 'endTimeAfter' in filters:
            _eta_raw = str(filters['endTimeAfter']).replace('Z', '+00:00')
            try:
                _eta_dt = datetime.fromisoformat(_eta_raw)
                if _eta_dt.tzinfo is None:
                    _eta_dt = _eta_dt.replace(tzinfo=timezone.utc)
                _et_rng['gte'] = _eta_dt.timestamp()
            except (ValueError, TypeError):
                pass
        if 'endTimeBefore' in filters:
            _etb_raw = str(filters['endTimeBefore']).replace('Z', '+00:00')
            try:
                _etb_dt = datetime.fromisoformat(_etb_raw)
                if _etb_dt.tzinfo is None:
                    _etb_dt = _etb_dt.replace(tzinfo=timezone.utc)
                _et_rng['lte'] = _etb_dt.timestamp()
            except (ValueError, TypeError):
                pass
        if _et_rng:
            must.append(qm.FieldCondition(key='ends_at', range=qm.Range(**_et_rng)))
    if 'isGemDomain' in filters:
        must.append(qm.FieldCondition(
            key='is_gem',
            match=qm.MatchValue(value=int(bool(filters['isGemDomain']))),
        ))
    # --- keyword and enrichment search filters ---
    # keyword_contains / keyword_starts_with / keyword_ends_with: applied as Python
    # substring post-filters via extract_keyword_post_filters. The `sld` field is a
    # KEYWORD index (init_qdrant_collection.py), which supports only exact MatchValue/
    # MatchAny — not MatchText substring matching — so contains/prefix/suffix must run
    # client-side over an oversampled pool to stay consistent with the in-memory backend.
    # Bid count
    if 'bids_min' in filters or 'bids_max' in filters:
        _bids_rng: Dict[str, Any] = {}
        if 'bids_min' in filters:
            _bids_rng['gte'] = float(filters['bids_min'])
        if 'bids_max' in filters:
            _bids_rng['lte'] = float(filters['bids_max'])
        must.append(qm.FieldCondition(key='bid_count', range=qm.Range(**_bids_rng)))
    # Domain age — skip numeric range when domain_age_is_unknown is set (caller wants null-valued items; IsNullCondition handles that via _UNKNOWN_SELECTABLE below).
    if ('domain_age_min' in filters or 'domain_age_max' in filters) and not bool(filters.get('domain_age_is_unknown')):
        _age_rng: Dict[str, Any] = {}
        if 'domain_age_min' in filters:
            _age_rng['gte'] = float(filters['domain_age_min'])
        if 'domain_age_max' in filters:
            _age_rng['lte'] = float(filters['domain_age_max'])
        must.append(qm.FieldCondition(key='domain_age_years', range=qm.Range(**_age_rng)))
    # Traffic — skip numeric range when traffic_is_unknown is set (mirrors StructuredRetriever._matches semantics).
    if ('traffic_min' in filters or 'traffic_max' in filters) and not bool(filters.get('traffic_is_unknown')):
        _tr_rng: Dict[str, Any] = {}
        if 'traffic_min' in filters:
            _tr_rng['gte'] = float(filters['traffic_min'])
        if 'traffic_max' in filters:
            _tr_rng['lte'] = float(filters['traffic_max'])
        must.append(qm.FieldCondition(key='monthly_traffic', range=qm.Range(**_tr_rng)))
    # Traffic proxy score (float composite)
    if 'traffic_proxy_min' in filters or 'traffic_proxy_max' in filters:
        _tp_rng: Dict[str, Any] = {}
        if 'traffic_proxy_min' in filters:
            _tp_rng['gte'] = float(filters['traffic_proxy_min'])
        if 'traffic_proxy_max' in filters:
            _tp_rng['lte'] = float(filters['traffic_proxy_max'])
        must.append(qm.FieldCondition(key='traffic_proxy_score', range=qm.Range(**_tp_rng)))
    # Has web traffic signal (0/1)
    if 'has_web_traffic_signal' in filters:
        must.append(qm.FieldCondition(key='has_web_traffic_signal', match=qm.MatchValue(value=int(bool(filters['has_web_traffic_signal'])))))
    # Estimated traffic tier (0–4 integer range)
    if 'estimated_traffic_tier_min' in filters or 'estimated_traffic_tier_max' in filters:
        _et_rng: Dict[str, Any] = {}
        if 'estimated_traffic_tier_min' in filters:
            _et_rng['gte'] = float(filters['estimated_traffic_tier_min'])
        if 'estimated_traffic_tier_max' in filters:
            _et_rng['lte'] = float(filters['estimated_traffic_tier_max'])
        must.append(qm.FieldCondition(key='estimated_traffic_tier', range=qm.Range(**_et_rng)))
    # GoValue (raw estimated value in dollars)
    if 'govalue_min' in filters or 'govalue_max' in filters:
        _gv_rng: Dict[str, Any] = {}
        if 'govalue_min' in filters:
            _gv_rng['gte'] = float(filters['govalue_min'])
        if 'govalue_max' in filters:
            _gv_rng['lte'] = float(filters['govalue_max'])
        must.append(qm.FieldCondition(key='govalue_score', range=qm.Range(**_gv_rng)))
    # Character constraints (0/1 integer payload fields)
    for _slot, _field in (('has_hyphen', 'has_hyphen'), ('has_number', 'has_number'), ('is_idn', 'is_idn')):
        if _slot in filters:
            must.append(qm.FieldCondition(key=_field, match=qm.MatchValue(value=int(bool(filters[_slot])))))
    # Auction boolean payload fields (0/1 integer columns from auction_audit_cln).
    for _slot, _field in _AUCTION_BOOLEAN_PAYLOAD_FIELDS:
        if _slot in filters:
            must.append(qm.FieldCondition(key=_field, match=qm.MatchValue(value=int(bool(filters[_slot])))))
    # BuyItNow price range — requires the item to have a buy_it_now_price value.
    if 'buy_it_now_min' in filters or 'buy_it_now_max' in filters:
        _bin_rng: Dict[str, Any] = {}
        if 'buy_it_now_min' in filters:
            _bin_rng['gte'] = float(filters['buy_it_now_min'])
        if 'buy_it_now_max' in filters:
            _bin_rng['lte'] = float(filters['buy_it_now_max'])
        must.append(qm.FieldCondition(key='buy_it_now_price', range=qm.Range(**_bin_rng)))
    # Majestic integer range pairs
    for _min_slot, _max_slot, _key in (
        ('majestic_tf_min', 'majestic_tf_max', 'majestic_tf'),
        ('majestic_cf_min', 'majestic_cf_max', 'majestic_cf'),
        ('majestic_backlinks_min', 'majestic_backlinks_max', 'majestic_backlinks'),
        ('majestic_ref_domains_min', 'majestic_ref_domains_max', 'majestic_ref_domains'),
    ):
        if _min_slot in filters or _max_slot in filters:
            _m_rng: Dict[str, Any] = {}
            if _min_slot in filters:
                _m_rng['gte'] = float(filters[_min_slot])
            if _max_slot in filters:
                _m_rng['lte'] = float(filters[_max_slot])
            must.append(qm.FieldCondition(key=_key, range=qm.Range(**_m_rng)))
    # TLF Insights
    if 'tlf_exact_match' in filters:
        must.append(qm.FieldCondition(key='tlf_exact_match', match=qm.MatchValue(value=int(bool(filters['tlf_exact_match'])))))
    if 'tlf_developed' in filters:
        must.append(qm.FieldCondition(key='tlf_developed', match=qm.MatchValue(value=int(bool(filters['tlf_developed'])))))
    if 'tlf_keyword_regs_min' in filters:
        must.append(qm.FieldCondition(key='tlf_keyword_regs', range=qm.Range(gte=float(filters['tlf_keyword_regs_min']))))
    # SEMrush integer range pairs
    for _min_slot, _max_slot, _key in (
        ('semrush_backlinks_min', 'semrush_backlinks_max', 'semrush_backlinks'),
        ('semrush_indexed_pages_min', 'semrush_indexed_pages_max', 'semrush_indexed_pages'),
        ('semrush_ref_domains_min', 'semrush_ref_domains_max', 'semrush_ref_domains'),
        ('semrush_search_volume_min', 'semrush_search_volume_max', 'semrush_search_volume'),
    ):
        if _min_slot in filters or _max_slot in filters:
            _s_rng: Dict[str, Any] = {}
            if _min_slot in filters:
                _s_rng['gte'] = float(filters[_min_slot])
            if _max_slot in filters:
                _s_rng['lte'] = float(filters[_max_slot])
            must.append(qm.FieldCondition(key=_key, range=qm.Range(**_s_rng)))
    # SEMrush float range pairs. semrush_authority_score is the index's only
    # authority metric — "domain authority"/"DA" queries alias onto it (NOT Moz DA).
    for _min_slot, _max_slot, _key in (('semrush_authority_min', 'semrush_authority_max', 'semrush_authority_score'), ('semrush_cpc_min', 'semrush_cpc_max', 'semrush_cpc')):
        if _min_slot in filters or _max_slot in filters:
            _sf_rng: Dict[str, Any] = {}
            if _min_slot in filters:
                _sf_rng['gte'] = float(filters[_min_slot])
            if _max_slot in filters:
                _sf_rng['lte'] = float(filters[_max_slot])
            must.append(qm.FieldCondition(key=_key, range=qm.Range(**_sf_rng)))
    # Lifecycle state — restrict auction_type to the mapped id set ('active' is empty
    # and imposes no type restriction; ends_at baseline handles it).
    if 'lifecycle_state' in filters:
        _lc_map_q = filters.get('_lifecycle_map') or _LIFECYCLE_AUCTION_TYPES
        _lc_allowed = _lc_map_q.get(str(filters['lifecycle_state']).lower())
        if _lc_allowed:
            must.append(qm.FieldCondition(key='auction_type', match=qm.MatchAny(any=[str(a) for a in sorted(_lc_allowed)])))
    # "Select unknown" slots — match items whose mapped field is null in the payload.
    for _u_slot, _u_field in _UNKNOWN_SELECTABLE.items():
        if _u_slot in filters and bool(filters[_u_slot]):
            must.append(qm.IsNullCondition(is_null=qm.PayloadField(key=_u_field)))
    # Exclusion filters: tld and auction_type exclude lists become Qdrant must_not conditions.
    # keyword_contains_exclude is handled as a Python post-filter (sld is a KEYWORD field; no server-side substring must_not).
    if 'tldExcludeList' in filters:
        _excl_tlds = [normalize_tld(t) for t in _as_filter_list(filters['tldExcludeList'])]
        if _excl_tlds:
            must_not.append(qm.FieldCondition(key='tld', match=qm.MatchAny(any=_excl_tlds)))
    if 'typeExcludeList' in filters:
        # Expand labels to numeric IDs (parity with typeIncludeList / structured _matches).
        _excl_types = sorted(_expand_auction_type_filter(filters['typeExcludeList']))
        if _excl_types:
            must_not.append(qm.FieldCondition(key='auction_type', match=qm.MatchAny(any=_excl_types)))
    if not must and not must_not:
        return None
    _fkw: Dict[str, Any] = {}
    if must:
        _fkw['must'] = must
    if must_not:
        _fkw['must_not'] = must_not
    return qm.Filter(**_fkw)


def extract_keyword_post_filters(
    filters: Dict[str, Any],
    *,
    default_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Return contains / starts_with / ends_with constraints Qdrant cannot express natively.

    The `sld` field is a KEYWORD index, so substring containment and prefix/suffix
    matching cannot run server-side. The caller applies these as Python-side
    post-filters via `keyword_value_matches` after retrieving an oversample from
    Qdrant — identical semantics to the in-memory structured backend. Raw slot
    values (str or list of terms) are passed through unchanged; the combine mode
    ('any'/'all') for multi-term lists travels under the 'mode' key.

    Mode resolution (no hardcoded default):
      1. ``filters['keyword_match_mode']`` when set
      2. else ``default_mode`` (caller must pass config-driven value)
      3. else raise — keyword post-filters require an explicit mode

    :param filters: Dict[str, Any] - Output of extract_filters_from_intent
    :param default_mode: Optional[str] - Config-driven fallback ('any'/'all')
    :return: Dict[str, Any] - Keys 'contains' / 'starts_with' / 'ends_with' (raw
        term or term list) plus 'mode' when any keyword slot is present
    """
    out: Dict[str, Any] = {}
    if 'keyword_contains' in filters:
        out['contains'] = filters['keyword_contains']
    if 'keyword_starts_with' in filters:
        out['starts_with'] = filters['keyword_starts_with']
    if 'keyword_ends_with' in filters:
        out['ends_with'] = filters['keyword_ends_with']
    if 'keyword_contains_exclude' in filters:
        out['contains_exclude'] = filters['keyword_contains_exclude']
    if 'keyword_phrase' in filters:
        out['phrase'] = filters['keyword_phrase']
    if out:
        mode = filters.get('keyword_match_mode')
        if mode is None:
            mode = default_mode
        if mode not in ('any', 'all'):
            raise RetrievalError(
                "keyword_match_mode must be 'any' or 'all' when keyword "
                f"post-filters are active; got {mode!r}"
            )
        out['mode'] = mode
    return out


# ---------------------------------------------------------------------------
# Construction-safe factory — mirrors semantic_search.analytics.ClickHouseClient
# ---------------------------------------------------------------------------

class QdrantClientFactory:
    """Construction-safe holder for an `AsyncQdrantClient`.

    Construction NEVER raises:
      - `qdrant-client` import failure sets `available=False`, warning logged
      - client construction error sets `available=False`, warning logged
      - subsequent adapter calls raise `QdrantUnavailableError`

    The factory exposes the live client via `client` (Optional) and the
    resolved API key (read from `config.api_key_env_var` at construction
    time, never from YAML — `responsible-ai.mdc` §secret-handling).

    :param config: QdrantConfig - Typed Qdrant config
    """

    def __init__(self, config: QdrantConfig) -> None:
        if not isinstance(config, QdrantConfig):
            raise RetrievalError("QdrantClientFactory requires a typed QdrantConfig")
        self._config = config
        self._client: Optional[Any] = None
        self._available = False
        self._api_key_present = False
        self._init_client()

    @property
    def available(self) -> bool:
        """True iff an `AsyncQdrantClient` was constructed successfully."""
        return self._available

    @property
    def client(self) -> Optional[Any]:
        """The wrapped `AsyncQdrantClient` (or None when unavailable)."""
        return self._client

    @property
    def collection_name(self) -> str:
        """Target collection name from config."""
        return self._config.collection_name

    @property
    def config(self) -> QdrantConfig:
        """The typed Qdrant config (read-only access for adapters)."""
        return self._config

    def _init_client(self) -> None:
        """Build the AsyncQdrantClient. Soft-fails so the registry can boot."""
        if _AsyncQdrantClient is None:
            logger.warning("qdrant_client_unavailable error_type=ImportError")
            return
        api_key: Optional[str] = None
        if self._config.api_key_env_var:
            raw = os.environ.get(self._config.api_key_env_var, '')
            api_key = raw if raw else None
            self._api_key_present = api_key is not None
        try:
            self._client = _AsyncQdrantClient(
                host=self._config.host,
                port=int(self._config.port),
                grpc_port=int(self._config.grpc_port),
                prefer_grpc=bool(self._config.prefer_grpc),
                https=bool(self._config.https),
                api_key=api_key,
                # gRPC channel deadline for all ops. Set large because:
                # - search SLA is enforced by asyncio.wait_for(read_timeout_seconds) in callers
                # - collection management (_ensure_collection) and bulk upserts need 120+ s
                # qdrant-client 1.17.x uses self._timeout as the gRPC deadline for every call
                # and ignores per-call timeout= at the gRPC level.
                timeout=120,
                check_compatibility=bool(self._config.check_compatibility),
            )
            self._available = True
            logger.info(
                f"qdrant_client_ready host={self._config.host} port={self._config.port} "
                f"grpc_port={self._config.grpc_port} prefer_grpc={self._config.prefer_grpc} "
                f"https={self._config.https} collection={self._config.collection_name} "
                f"api_key_present={self._api_key_present} check_compatibility={bool(self._config.check_compatibility)}"
            )
        except Exception as e:  # noqa: BLE001 — soft-fail factory boot
            logger.error(f"qdrant_init_failed error_type={type(e).__name__} error={str(e)}")

    async def aclose(self) -> None:
        """Close the underlying client (idempotent, never raises)."""
        if self._client is None:
            return
        close_fn = getattr(self._client, 'close', None)
        if close_fn is None:
            self._client = None
            return
        try:
            result = close_fn()
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:  # noqa: BLE001 — close must not raise
            logger.warning(f"qdrant_close_failed error_type={type(e).__name__} error={e}")
        self._client = None


# ---------------------------------------------------------------------------
# Vector + Structured + Hybrid adapters
# ---------------------------------------------------------------------------

def _extract_item_id(point: Any, payload_id_field: str) -> str:
    """Resolve `Candidate.item_id` from a Qdrant point.

    Priority:
      1. payload[payload_id_field] when payload_id_field is non-empty AND present
      2. str(point.id) — covers numeric/UUID surrogate ids

    :raises QdrantQueryError: When the resolved id is empty / falsy.
    """
    payload = getattr(point, 'payload', None) or {}
    if payload_id_field and isinstance(payload, dict):
        value = payload.get(payload_id_field)
        if value is not None and str(value):
            return str(value)
    raw_id = getattr(point, 'id', None)
    if raw_id is None:
        raise QdrantQueryError("Qdrant point missing both id and payload_id_field")
    text = str(raw_id)
    if not text:
        raise QdrantQueryError("Qdrant point id resolves to empty string")
    return text


def _normalize_score(raw: Any) -> float:
    """Clamp / normalize a Qdrant similarity score into [0, 1].

    Qdrant returns cosine in [-1, 1] for the `Cosine` distance and dot/euclid
    in unbounded ranges. We map cosine via (s+1)/2 (matches the in-memory
    backend), and clamp anything outside [0,1] from other distances so the
    `Candidate.score must be in [0,1]` invariant holds.
    """
    try:
        s = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if -1.0 <= s <= 1.0:
        s = (s + 1.0) / 2.0 if s < 0.0 else s
    if s < 0.0:
        return 0.0
    if s > 1.0:
        return 1.0
    return s


def _filter_payload(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a defensive copy of the payload dict (or an empty dict)."""
    if not isinstance(payload, dict):
        return {}
    return dict(payload)


class QdrantVectorIndex(VectorIndex):
    """Qdrant-backed vector index (filterable HNSW dense search).

    Honors the `VectorIndex` contract: `search(query_vec, top_k)` returns a
    list of `(item_id, score, payload)` tuples in score-descending order.

    Synchronous-on-an-async-API: the `VectorIndex` interface is defined as
    sync (matching the in-memory implementation). To keep that contract we
    stage the awaitable on a private event loop owned by this index. In
    production callers reach the Qdrant client through `QdrantHybridRetriever`
    / `QdrantStructuredRetriever` which are natively async — this sync
    `search` exists to satisfy `VectorRetriever`'s `asyncio.to_thread(...)`
    call, which already isolates blocking IO in a worker thread.
    """

    def __init__(self, factory: QdrantClientFactory):
        if not isinstance(factory, QdrantClientFactory):
            raise RetrievalError("QdrantVectorIndex requires a QdrantClientFactory")
        self._factory = factory
        # `dim` is intentionally not enforced client-side — Qdrant will reject
        # mismatched vectors at the server. We expose the configured embedding
        # dim to satisfy the `VectorRetriever` constructor's dim-check.
        self._dim_hint = -1

    @property
    def dim(self) -> int:
        """Configured embedding dim (set by the registry after construction)."""
        return self._dim_hint

    def set_dim_hint(self, dim: int) -> None:
        """Wire the configured embedding_dim so VectorRetriever's check passes."""
        if dim < 4:
            raise RetrievalError("QdrantVectorIndex.set_dim_hint dim must be >= 4")
        self._dim_hint = int(dim)

    def search(self, query_vec: Sequence[float], top_k: int) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Synchronous facade over `AsyncQdrantClient.query_points`.

        :raises QdrantUnavailableError: When the factory has no live client
        :raises QdrantQueryError: On a query-level Qdrant error
        """
        if top_k < 1:
            return []
        if not self._factory.available or self._factory.client is None:
            raise QdrantUnavailableError("Qdrant client unavailable")
        cfg = self._factory.config
        if _qm is None:
            raise QdrantUnavailableError("qdrant-client not installed")
        qm = _qm
        params = qm.SearchParams(hnsw_ef=int(cfg.hnsw_ef_search), exact=False)
        dense_name = cfg.hybrid.dense_vector_name or None
        coro = self._factory.client.query_points(
            collection_name=cfg.collection_name,
            query=list(query_vec),
            using=dense_name,
            search_params=params,
            limit=int(top_k),
            with_payload=True,
            with_vectors=False,
        )
        try:
            response = _run_coro_sync(coro, timeout=float(cfg.read_timeout_seconds))
        except asyncio.TimeoutError as e:
            raise QdrantQueryError(f"qdrant query exceeded timeout={cfg.read_timeout_seconds:.1f}s") from e
        except QdrantUnavailableError:
            raise
        except Exception as e:
            raise QdrantQueryError(f"qdrant query failed: {e}") from e
        return _points_to_tuples(response, cfg.payload_id_field)


class QdrantStructuredIndex(StructuredIndex):
    """Qdrant-backed structured filter index (payload-only scan, no vector).

    Implements the `StructuredIndex.search(filters, top_k) -> List[Dict]`
    contract by issuing a `scroll` (filter-only) call against the unified
    collection and mapping each point to a dict the existing
    `StructuredRetriever` already understands (`item_id`, `score`, payload
    fields).

    Score policy:
      - When `payload_score_field` is non-empty AND present on the point,
        the field's float value is clamped to [0,1] and used as `score`.
      - Otherwise `score = 1 - (rank / top_k)` so earlier rows rank higher
        and the [0,1] invariant on `Candidate.score` holds.
    """

    def __init__(self, factory: QdrantClientFactory):
        if not isinstance(factory, QdrantClientFactory):
            raise RetrievalError("QdrantStructuredIndex requires a QdrantClientFactory")
        self._factory = factory

    def search(self, filters: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
        """Filter-only payload scan via Qdrant `scroll`.

        :raises QdrantUnavailableError: When the factory has no live client
        :raises QdrantQueryError: On a query-level Qdrant error
        """
        if top_k < 1:
            return []
        if not filters:
            return []
        if not self._factory.available or self._factory.client is None:
            raise QdrantUnavailableError("Qdrant client unavailable")
        cfg = self._factory.config
        # contains/starts_with/ends_with run as Python post-filters (sld is a KEYWORD
        # index) — oversample the scroll so post-filtering does not under-deliver.
        # Mode must come from filters (StructuredRetriever injects structured.keyword_match_mode).
        kw_post = extract_keyword_post_filters(filters)
        qfilter = build_qdrant_filter(filters, active_only=cfg.active_only_baseline, price_gt_zero=cfg.price_gt_zero_baseline, starting_bid_gt_zero=cfg.starting_bid_gt_zero_baseline)
        if qfilter is None and not kw_post:
            return []
        _ovs = int(cfg.hybrid.kw_prefix_oversample_factor) if (
            kw_post and ('starts_with' in kw_post or 'ends_with' in kw_post)
        ) else int(cfg.hybrid.kw_post_oversample_factor)
        scroll_limit = int(top_k) * _ovs if kw_post else int(top_k)
        coro = self._factory.client.scroll(collection_name=cfg.collection_name, scroll_filter=qfilter, limit=scroll_limit, with_payload=True, with_vectors=False)
        try:
            scroll_result = _run_coro_sync(coro, timeout=float(cfg.read_timeout_seconds))
        except asyncio.TimeoutError as e:
            raise QdrantQueryError(f"qdrant scroll exceeded timeout={cfg.read_timeout_seconds:.1f}s") from e
        except QdrantUnavailableError:
            raise
        except Exception as e:
            raise QdrantQueryError(f"qdrant scroll failed: {e}") from e
        # `scroll` returns a (points, next_offset) tuple.
        points: Sequence[Any]
        if isinstance(scroll_result, tuple) and scroll_result:
            points = scroll_result[0]
        else:
            points = scroll_result or []
        if kw_post:
            kept_points: List[Any] = []
            for point in points:
                sld = str((getattr(point, 'payload', None) or {}).get('sld', '')).lower()
                _mode = kw_post['mode']
                if 'contains' in kw_post and not keyword_value_matches(sld, kw_post['contains'], 'contains', _mode):
                    continue
                if 'starts_with' in kw_post and not keyword_value_matches(sld, kw_post['starts_with'], 'starts_with', _mode):
                    continue
                if 'ends_with' in kw_post and not keyword_value_matches(sld, kw_post['ends_with'], 'ends_with', _mode):
                    continue
                kept_points.append(point)
            points = kept_points[: int(top_k)]
        rows: List[Dict[str, Any]] = []
        n = max(1, int(top_k))
        for rank, point in enumerate(points):
            payload = _filter_payload(getattr(point, 'payload', None))
            item_id = _extract_item_id(point, cfg.payload_id_field)
            score: float
            if cfg.payload_score_field and cfg.payload_score_field in payload:
                score = _normalize_score(payload[cfg.payload_score_field])
            else:
                score = max(0.0, 1.0 - (float(rank) / float(n)))
            row: Dict[str, Any] = {'item_id': item_id, 'score': float(score)}
            for k, v in payload.items():
                if k in row:
                    continue
                row[k] = v
            rows.append(row)
        return rows


class QdrantHybridRetriever(Retriever):
    """Single-round-trip vector + payload (+ optional BM25) retriever.

    Reports `source='vector'` so `SearchOrchestrator._gather_candidates` keeps
    its existing fan-out contract (Option A). The structured retriever wired
    alongside this one MUST be a no-op (the registry enforces this).

    Implementation:
      1. Encode the query with the shared `Encoder`.
      2. Build the standard filter dict from the intent and translate it
         into a Qdrant `Filter`.
      3. Issue a single `query_points(...)` call:
           - No sparse legs: dense `query=` + `query_filter=`.
           - With sparse legs (BM42 token-sparse when `bm25_enabled=true` and/or
             the character n-gram fuzzy-recall leg when `ngram.enabled=true`):
             dense + sparse prefetches fused server-side via
             `FusionQuery(fusion=...)` — no client-side RRF.
           - When a rerank encoder is wired: the first-stage query (fusion or
             dense) becomes a nested prefetch whose fused pool is reranked
             server-side by the higher-dim `dense_rerank` vector.
      4. Map response points to `Candidate(source='vector', ...)`.

    All work runs on the asyncio event loop directly because
    `AsyncQdrantClient` is natively awaitable.
    """

    def __init__(
        self,
        factory: QdrantClientFactory,
        encoder: Encoder,
        embedding_dim: int,
        min_similarity: float,
        bm25_query_fn: Optional[Any] = None,
        rerank_encoder: Optional[Encoder] = None,
        rerank_vector_name: Optional[str] = None,
        rerank_dim: Optional[int] = None,
        rerank_input_n: Optional[int] = None,
        ngram_query_fn: Optional[Any] = None,
        query_preprocessor: Optional[Callable[[str], str]] = None,
        keyword_match_mode: str = '',
    ):
        if not isinstance(factory, QdrantClientFactory):
            raise RetrievalError("QdrantHybridRetriever requires a QdrantClientFactory")
        if encoder is None:
            raise RetrievalError("QdrantHybridRetriever requires an Encoder")
        if encoder.dim != int(embedding_dim):
            raise RetrievalError(f"QdrantHybridRetriever encoder.dim={encoder.dim} != embedding_dim={embedding_dim}")
        if not 0.0 <= float(min_similarity) <= 1.0:
            raise RetrievalError("QdrantHybridRetriever min_similarity must be in [0,1]")
        if keyword_match_mode not in ('any', 'all'):
            raise RetrievalError(
                "QdrantHybridRetriever keyword_match_mode must be 'any' or 'all' "
                f"(from retrieval.structured.keyword_match_mode); got {keyword_match_mode!r}"
            )
        cfg = factory.config
        if cfg.hybrid.bm25_enabled and bm25_query_fn is None:
            raise RetrievalError("QdrantHybridRetriever bm25_enabled=true requires a bm25_query_fn callable (text -> SparseVector); registry must wire one when enabling BM25")
        ngram_cfg = cfg.hybrid.ngram
        ngram_enabled = ngram_cfg is not None and ngram_cfg.enabled
        if ngram_enabled and ngram_query_fn is None:
            raise RetrievalError("QdrantHybridRetriever ngram.enabled=true requires a ngram_query_fn callable (text -> SparseVector); registry must wire one when enabling the ngram channel")
        rerank_enabled = rerank_encoder is not None
        if rerank_enabled:
            if rerank_dim is None or rerank_encoder.dim != int(rerank_dim):
                raise RetrievalError(f"QdrantHybridRetriever rerank_encoder.dim={rerank_encoder.dim} != rerank_dim={rerank_dim}")
            if not isinstance(rerank_vector_name, str) or not rerank_vector_name:
                raise RetrievalError("QdrantHybridRetriever rerank_vector_name must be a non-empty string when rerank_encoder is set")
            if rerank_input_n is None or int(rerank_input_n) < 1:
                raise RetrievalError("QdrantHybridRetriever rerank_input_n must be int >= 1 when rerank_encoder is set")
        self._factory = factory
        self._encoder = encoder
        self._embedding_dim = int(embedding_dim)
        self._min_similarity = float(min_similarity)
        self._bm25_query_fn = bm25_query_fn
        self._ngram_enabled = ngram_enabled
        self._ngram_query_fn = ngram_query_fn
        self._ngram_vector_name = ngram_cfg.vector_name if ngram_enabled else None
        self._rerank_enabled = rerank_enabled
        self._rerank_encoder = rerank_encoder
        self._rerank_vector_name = rerank_vector_name
        self._rerank_dim = int(rerank_dim) if rerank_enabled else None
        self._rerank_input_n = int(rerank_input_n) if rerank_enabled else None
        self._query_preprocessor = query_preprocessor
        self._keyword_match_mode = keyword_match_mode

    @property
    def source(self) -> str:
        """Reports 'vector' so the existing orchestrator fan-out is preserved (Option A)."""
        return 'vector'

    def _kw_post_oversample_factor(self, kw_post: Dict[str, Any]) -> int:
        """Return config-driven oversample multiplier for keyword post-filters."""
        if not kw_post:
            return 1
        hybrid = self._factory.config.hybrid
        if 'starts_with' in kw_post or 'ends_with' in kw_post:
            return int(hybrid.kw_prefix_oversample_factor)
        return int(hybrid.kw_post_oversample_factor)

    def _kw_post_from_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build keyword post-filters using config-driven keyword_match_mode."""
        return extract_keyword_post_filters(
            filters, default_mode=self._keyword_match_mode,
        )

    def _apply_kw_post_filter(self, points: Sequence[Any], kw_post: Dict[str, Any], cap: int) -> List[Any]:
        """Python-side keyword post-filter; returns up to ``cap`` points."""
        if not kw_post:
            return list(points)[: int(cap)]
        kept: List[Any] = []
        _mode = kw_post['mode']
        for point in points:
            sld = str((getattr(point, 'payload', None) or {}).get('sld', '')).lower()
            if 'contains' in kw_post and not keyword_value_matches(sld, kw_post['contains'], 'contains', _mode):
                continue
            if 'starts_with' in kw_post and not keyword_value_matches(sld, kw_post['starts_with'], 'starts_with', _mode):
                continue
            if 'ends_with' in kw_post and not keyword_value_matches(sld, kw_post['ends_with'], 'ends_with', _mode):
                continue
            if 'contains_exclude' in kw_post and keyword_value_matches(sld, kw_post['contains_exclude'], 'contains', _mode):
                continue
            if 'phrase' in kw_post and str(kw_post['phrase']).lower() not in sld.replace('-', ''):
                continue
            kept.append(point)
            if len(kept) >= int(cap):
                break
        return kept

    def _points_to_ordered_candidates(
        self,
        points: Sequence[Any],
        *,
        rail_id: Optional[str],
        top_k: int,
    ) -> List[Candidate]:
        """Map scroll points to Candidates in scroll order (rank = explore-rail order)."""
        cfg = self._factory.config
        candidates: List[Candidate] = []
        n = max(1, int(top_k))
        for rank, point in enumerate(points):
            if len(candidates) >= int(top_k):
                break
            payload = _filter_payload(getattr(point, 'payload', None))
            item_id = _extract_item_id(point, cfg.payload_id_field)
            if cfg.payload_score_field and cfg.payload_score_field in payload:
                score = _normalize_score(payload[cfg.payload_score_field])
            else:
                score = max(0.0, 1.0 - (float(rank) / float(n)))
            enriched = dict(payload)
            enriched['vector_score'] = float(score)
            if rail_id:
                enriched['filter_only_rail'] = rail_id
            candidates.append(Candidate(item_id=item_id, score=float(score), source='vector', payload=enriched))
        return candidates

    @staticmethod
    def _rrf_fuse_filter_only_legs(
        leg_lists: List[Tuple[str, List[Candidate]]],
        rrf_k: int,
        top_k: int,
    ) -> List[Candidate]:
        """RRF across ordered filter-only scroll legs; score = fused RRF mass."""
        scores: Dict[str, float] = {}
        first_seen: Dict[str, Candidate] = {}
        rails_hit: Dict[str, List[str]] = {}
        for rail_id, cands in leg_lists:
            for rank, cand in enumerate(cands, start=1):
                iid = cand.item_id
                scores[iid] = scores.get(iid, 0.0) + 1.0 / (float(rrf_k) + float(rank))
                if iid not in first_seen:
                    first_seen[iid] = cand
                rails_hit.setdefault(iid, []).append(rail_id)
        ordered = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        out: List[Candidate] = []
        for iid in ordered[: int(top_k)]:
            base = first_seen[iid]
            payload = dict(base.payload or {})
            payload['vector_score'] = float(scores[iid])
            payload['filter_only_rails'] = list(rails_hit.get(iid, []))
            out.append(Candidate(item_id=iid, score=float(scores[iid]), source='vector', payload=payload))
        return out

    def _order_by_for_leg(self, leg: FilterOnlyRailLegConfig) -> Any:
        """Build Qdrant ``OrderBy`` from a config leg (no hardcoded directions)."""
        if _qm is None:
            raise QdrantUnavailableError("qdrant-client not installed")
        direction = _qm.Direction.ASC if leg.order_by_direction == 'asc' else _qm.Direction.DESC
        return _qm.OrderBy(key=leg.order_by_field, direction=direction)

    async def _scroll_filter_only_leg(
        self,
        *,
        qfilter: Any,
        kw_post: Dict[str, Any],
        top_k: int,
        leg: Optional[FilterOnlyRailLegConfig],
        scroll_limit: int,
    ) -> List[Candidate]:
        """One filter-constrained scroll; optional ``order_by`` from ``leg``."""
        cfg = self._factory.config
        kwargs: Dict[str, Any] = {
            'collection_name': cfg.collection_name,
            'scroll_filter': qfilter,
            'limit': int(scroll_limit),
            'with_payload': True,
            'with_vectors': False,
        }
        if leg is not None:
            kwargs['order_by'] = self._order_by_for_leg(leg)
        try:
            scroll_result = await asyncio.wait_for(
                self._factory.client.scroll(**kwargs),
                timeout=float(cfg.read_timeout_seconds),
            )
        except asyncio.TimeoutError as e:
            raise QdrantQueryError(
                f"qdrant hybrid filter-only scroll exceeded timeout={cfg.read_timeout_seconds:.1f}s"
                + (f" rail={leg.rail_id}" if leg is not None else "")
            ) from e
        except (QdrantUnavailableError, QdrantQueryError):
            raise
        except Exception as e:
            raise QdrantQueryError(
                f"qdrant hybrid filter-only scroll failed: {e}"
                + (f" rail={leg.rail_id}" if leg is not None else "")
            ) from e
        if isinstance(scroll_result, tuple) and scroll_result:
            points: Sequence[Any] = scroll_result[0]
        else:
            points = scroll_result or []
        filtered = self._apply_kw_post_filter(points, kw_post, cap=int(top_k))
        rail_id = leg.rail_id if leg is not None else None
        return self._points_to_ordered_candidates(filtered, rail_id=rail_id, top_k=int(top_k))

    async def retrieve_filter_only_rails(self, intent: QueryIntent, top_k: int) -> CandidateSet:
        """Forced multi-scroll RRF under hard qfilter (timeout / CH-empty explore ladder).

        Called dynamically via ``SearchOrchestrator._qdrant_filter_only_rails_as_ranked``
        (``getattr``) when ``timeout_fallback.qdrant_rails_when_ch_empty=true`` — timeout
        fallback, explore-primary CH-empty, and CH-unhealthy complement merge.

        Always merges ``filter_only_rails.legs`` (ignores ``filter_only_rails.enabled``,
        which only gates the normal empty-encode search path). Same hard filters as
        hybrid retrieve — inventory is never widened for ranking.
        """
        if top_k < 1:
            return CandidateSet(source='vector', candidates=[], latency_ms=0.0)
        if not self._factory.available or self._factory.client is None:
            raise QdrantUnavailableError("Qdrant client unavailable")
        if _qm is None:
            raise QdrantUnavailableError("qdrant-client not installed")
        cfg = self._factory.config
        t0 = time.monotonic()
        filters = extract_filters_from_intent(intent)
        kw_post = self._kw_post_from_filters(filters)
        qfilter = build_qdrant_filter(
            filters,
            active_only=cfg.active_only_baseline,
            price_gt_zero=cfg.price_gt_zero_baseline,
            starting_bid_gt_zero=cfg.starting_bid_gt_zero_baseline,
        )
        if qfilter is None and not kw_post:
            latency_ms = (time.monotonic() - t0) * 1000.0
            logger.info(
                f"qdrant_timeout_filter_only_rails_empty request_id={intent.request_id} "
                f"reason=no_filter latency_ms={latency_ms:.1f}"
            )
            return CandidateSet(source='vector', candidates=[], latency_ms=latency_ms)
        return await self._retrieve_filter_only_scroll(
            intent, filters, qfilter, kw_post, top_k, t0, use_rail_merge=True,
        )

    async def _retrieve_filter_only_scroll(
        self,
        intent: QueryIntent,
        filters: Dict[str, Any],
        qfilter: Any,
        kw_post: Dict[str, Any],
        top_k: int,
        t0: float,
        use_rail_merge: bool,
    ) -> CandidateSet:
        """Payload scroll for empty-residual / pure-filter queries.

        Hybrid mode pairs this retriever with ``QdrantNoOpStructuredRetriever``,
        so filter-only queries have no separate structured leg. ANN on a zero
        vector is noise; scroll returns inventory that matches ``qfilter``.

        When ``use_rail_merge`` is True, runs configured ordered scrolls in
        parallel under the same hard filter and RRF-merges (ending-soon /
        trending / value). When False, a single unordered scroll is used.
        """
        cfg = self._factory.config
        rails_cfg: FilterOnlyRailsConfig = cfg.filter_only_rails
        kw_ovs = self._kw_post_oversample_factor(kw_post)
        if use_rail_merge:
            leg_tasks = []
            for leg in rails_cfg.legs:
                scroll_limit = int(top_k) * int(leg.limit_multiplier) * int(kw_ovs)
                leg_tasks.append(
                    self._scroll_filter_only_leg(
                        qfilter=qfilter,
                        kw_post=kw_post,
                        top_k=int(top_k),
                        leg=leg,
                        scroll_limit=scroll_limit,
                    )
                )
            leg_results = await asyncio.gather(*leg_tasks, return_exceptions=True)
            leg_lists: List[Tuple[str, List[Candidate]]] = []
            leg_errors: List[BaseException] = []
            for leg, result in zip(rails_cfg.legs, leg_results):
                if isinstance(result, BaseException):
                    leg_errors.append(result)
                    logger.warning(
                        f"qdrant_filter_only_rail_failed request_id={intent.request_id} "
                        f"rail={leg.rail_id} error_type={type(result).__name__} error={result}"
                    )
                    continue
                if result:
                    leg_lists.append((leg.rail_id, result))
            if not leg_lists and leg_errors:
                raise QdrantQueryError(
                    f"qdrant hybrid filter-only rails all failed request_id={intent.request_id} "
                    f"error_type={type(leg_errors[0]).__name__} error={leg_errors[0]}"
                ) from leg_errors[0]
            candidates = self._rrf_fuse_filter_only_legs(leg_lists, int(rails_cfg.rrf_k), int(top_k))
            latency_ms = (time.monotonic() - t0) * 1000.0
            logger.info(
                f"qdrant_hybrid_filter_only_rails request_id={intent.request_id} "
                f"legs={len(leg_lists)}/{len(rails_cfg.legs)} candidates={len(candidates)} "
                f"filters={sorted(filters.keys()) if filters else []} latency_ms={latency_ms:.1f}"
            )
            return CandidateSet(source='vector', candidates=candidates, latency_ms=latency_ms)

        scroll_limit = int(top_k) * int(kw_ovs)
        candidates = await self._scroll_filter_only_leg(
            qfilter=qfilter,
            kw_post=kw_post,
            top_k=int(top_k),
            leg=None,
            scroll_limit=scroll_limit,
        )
        latency_ms = (time.monotonic() - t0) * 1000.0
        logger.info(
            f"qdrant_hybrid_filter_only_scroll request_id={intent.request_id} "
            f"candidates={len(candidates)} filters={sorted(filters.keys()) if filters else []} "
            f"latency_ms={latency_ms:.1f}"
        )
        return CandidateSet(source='vector', candidates=candidates, latency_ms=latency_ms)

    async def retrieve(self, intent: QueryIntent, top_k: int) -> CandidateSet:
        """Single round-trip hybrid query against the unified Qdrant index.

        :raises QdrantUnavailableError: When the factory has no live client
        :raises QdrantQueryError: On a query-level Qdrant error
        """
        if top_k < 1:
            return CandidateSet(source='vector', candidates=[], latency_ms=0.0)
        if not self._factory.available or self._factory.client is None:
            raise QdrantUnavailableError("Qdrant client unavailable")
        if _qm is None:
            raise QdrantUnavailableError("qdrant-client not installed")
        cfg = self._factory.config
        t0 = time.monotonic()
        # Per-leg query construction: every semantic + lexical leg (dense, BM25,
        # ngram, rerank) encodes the SAME TLD-safe, filter-stripped text. The
        # accessor returns the residual concept when present, else the normalized
        # query with TLD literals removed — so filter words ("under 100 dollars")
        # and the TLD never dilute SLD lexical recall, and TLD is matched only as
        # the exact ``tld`` MatchAny filter built below from extract_filters_from_intent.
        encode_text = semantic_encode_text_for(intent)
        if self._query_preprocessor is not None:
            encode_text = self._query_preprocessor(encode_text)
        filters = extract_filters_from_intent(intent)
        logger.debug(
            f"qdrant_filter_build request_id={intent.request_id} "
            f"filter_keys={sorted(filters.keys()) if filters else []} "
            f"filter_values={ {k: v for k, v in filters.items()} if filters else {} }"
        )
        kw_post = self._kw_post_from_filters(filters)
        qfilter = build_qdrant_filter(filters, active_only=cfg.active_only_baseline, price_gt_zero=cfg.price_gt_zero_baseline, starting_bid_gt_zero=cfg.starting_bid_gt_zero_baseline)
        # Pure-filter / empty residual: no ANN signal. Scroll payload matches
        # (structured is NoOp under hybrid, so this IS the filter path).
        if not str(encode_text or '').strip():
            if qfilter is None and not kw_post:
                latency_ms = (time.monotonic() - t0) * 1000.0
                logger.info(
                    f"qdrant_hybrid_filter_only_empty request_id={intent.request_id} "
                    f"reason=no_encode_text_no_filter latency_ms={latency_ms:.1f}"
                )
                return CandidateSet(source='vector', candidates=[], latency_ms=latency_ms)
            return await self._retrieve_filter_only_scroll(
                intent, filters, qfilter, kw_post, top_k, t0,
                use_rail_merge=bool(cfg.filter_only_rails.enabled),
            )
        # Encoding is CPU-bound but fast; isolate via to_thread so a slow
        # encoder cannot stall the event loop (consistency with VectorRetriever).
        # Encode dense (shortlist), optional sparse, and optional rerank-dim query
        # vectors in parallel — each is CPU-bound and isolated via to_thread.
        _want_sparse = cfg.hybrid.bm25_enabled and self._bm25_query_fn is not None
        _want_ngram = self._ngram_enabled and self._ngram_query_fn is not None
        _jobs = [asyncio.to_thread(self._encoder.encode, encode_text)]
        if _want_sparse:
            _jobs.append(asyncio.to_thread(self._bm25_query_fn, encode_text))
        if _want_ngram:
            _jobs.append(asyncio.to_thread(self._ngram_query_fn, encode_text))
        if self._rerank_enabled:
            _jobs.append(asyncio.to_thread(self._rerank_encoder.encode, encode_text))
        _enc_results = await asyncio.gather(*_jobs)
        query_vec = _enc_results[0]
        _next = 1
        sparse_vec = None
        if _want_sparse:
            sparse_vec = _enc_results[_next]
            _next += 1
        ngram_vec = None
        if _want_ngram:
            ngram_vec = _enc_results[_next]
            _next += 1
        rerank_query_vec = None
        if self._rerank_enabled:
            rerank_query_vec = _enc_results[_next]
        if len(query_vec) != self._embedding_dim:
            raise RetrievalError(f"QdrantHybridRetriever query_vec dim={len(query_vec)} != embedding_dim={self._embedding_dim}")
        if self._rerank_enabled and len(rerank_query_vec) != self._rerank_dim:
            raise RetrievalError(f"QdrantHybridRetriever rerank_query_vec dim={len(rerank_query_vec)} != rerank_dim={self._rerank_dim}")
        params = _qm.SearchParams(hnsw_ef=int(cfg.hnsw_ef_search), exact=False)
        dense_name = cfg.hybrid.dense_vector_name or None
        # Oversample when a Python-side keyword post-filter is active so the
        # post-filter can trim to top_k without under-delivering rare prefixes.
        # Scale BOTH the final limit AND per-leg prefetch — otherwise fusion
        # still draws from an unscaled pool and rare starts_with/ends_with miss.
        _kw_ovs = self._kw_post_oversample_factor(kw_post)
        qdrant_limit = top_k * _kw_ovs
        kwargs: Dict[str, Any] = {
            'collection_name': cfg.collection_name,
            'limit': int(qdrant_limit),
            'with_payload': True,
            'with_vectors': False,
            'search_params': params,
        }
        if qfilter is not None:
            kwargs['query_filter'] = qfilter
        prefetch_limit = int(cfg.hybrid.prefetch_limit) * int(_kw_ovs)
        # Build the first-stage query. Server-side RRF fuses the dense leg with
        # any sparse legs that are wired (BM42 token-sparse and/or the character
        # n-gram fuzzy-recall leg). When the rerank stage is wired this fused
        # query becomes a nested prefetch whose pool is reranked by the higher-dim
        # ``dense_rerank`` vector server-side. With no sparse legs it is dense-only.
        _fuse = _want_sparse or _want_ngram
        fusion_legs = None
        fusion_mode = None
        if _fuse:
            fusion_legs = [_qm.Prefetch(query=list(query_vec), using=dense_name, limit=prefetch_limit, params=params, filter=qfilter)]
            if _want_sparse:
                fusion_legs.append(_qm.Prefetch(query=sparse_vec, using=cfg.hybrid.bm25_vector_name, limit=prefetch_limit, params=params, filter=qfilter))
            if _want_ngram:
                fusion_legs.append(_qm.Prefetch(query=ngram_vec, using=self._ngram_vector_name, limit=prefetch_limit, params=params, filter=qfilter))
            fusion_mode = _qm.Fusion.RRF if cfg.hybrid.fusion_strategy == 'rrf' else _qm.Fusion.DBSF
        if self._rerank_enabled:
            # Pool fed into the rerank stage; never smaller than the parent's limit.
            pool_n = max(int(self._rerank_input_n), int(qdrant_limit))
            if _fuse:
                inner = _qm.Prefetch(prefetch=fusion_legs, query=_qm.FusionQuery(fusion=fusion_mode), limit=pool_n)
            else:
                inner = _qm.Prefetch(query=list(query_vec), using=dense_name, limit=pool_n, params=params, filter=qfilter)
            kwargs['prefetch'] = inner
            kwargs['query'] = list(rerank_query_vec)
            kwargs['using'] = self._rerank_vector_name
        elif _fuse:
            kwargs['prefetch'] = fusion_legs
            kwargs['query'] = _qm.FusionQuery(fusion=fusion_mode)
        else:
            kwargs['query'] = list(query_vec)
            kwargs['using'] = dense_name
        try:
            response = await asyncio.wait_for(self._factory.client.query_points(**kwargs), timeout=float(cfg.read_timeout_seconds))
        except asyncio.TimeoutError as e:
            raise QdrantQueryError(f"qdrant hybrid query exceeded timeout={cfg.read_timeout_seconds:.1f}s") from e
        except (QdrantUnavailableError, QdrantQueryError):
            raise
        except Exception as e:
            raise QdrantQueryError(f"qdrant hybrid query failed: {e}") from e
        candidates: List[Candidate] = []
        for item_id, score, payload in _points_to_tuples(response, cfg.payload_id_field):
            if score < self._min_similarity:
                continue
            # Stamp the vector similarity into the payload so RRF preserves it
            # for the Layer 4 deterministic ranker (LAYER 4 primary signal).
            enriched_payload = dict(payload) if payload is not None else {}
            enriched_payload['vector_score'] = float(score)
            candidates.append(Candidate(item_id=item_id, score=float(score), source='vector', payload=enriched_payload))
        # Apply Python-side keyword post-filters (contains / starts_with / ends_with on SLD)
        if kw_post:
            filtered: List[Candidate] = []
            _mode = kw_post['mode']
            for c in candidates:
                sld = str(c.payload.get('sld', '')).lower()
                if 'contains' in kw_post and not keyword_value_matches(sld, kw_post['contains'], 'contains', _mode):
                    continue
                if 'starts_with' in kw_post and not keyword_value_matches(sld, kw_post['starts_with'], 'starts_with', _mode):
                    continue
                if 'ends_with' in kw_post and not keyword_value_matches(sld, kw_post['ends_with'], 'ends_with', _mode):
                    continue
                if 'contains_exclude' in kw_post and keyword_value_matches(sld, kw_post['contains_exclude'], 'contains', _mode):
                    continue
                if 'phrase' in kw_post and str(kw_post['phrase']).lower() not in sld.replace('-', ''):
                    continue
                filtered.append(c)
            candidates = filtered
        candidates = slice_candidates(candidates, top_k)
        latency_ms = (time.monotonic() - t0) * 1000.0
        logger.info(
            f"qdrant_hybrid_retrieval request_id={intent.request_id} "
            f"candidates={len(candidates)} bm25={cfg.hybrid.bm25_enabled} ngram={self._ngram_enabled} rerank={self._rerank_enabled} "
            f"filters={sorted(filters.keys()) if filters else []} latency_ms={latency_ms:.1f}"
        )
        return CandidateSet(source='vector', candidates=candidates, latency_ms=latency_ms)


class QdrantNoOpStructuredRetriever(Retriever):
    """No-op structured retriever paired with `QdrantHybridRetriever`.

    When the unified Qdrant index serves vector + payload in a single
    round-trip, the structured leg of `SearchOrchestrator._gather_candidates`
    must not issue a second Qdrant call (would double the latency without
    new information). This retriever satisfies the orchestrator's contract
    by returning an empty `CandidateSet` with zero latency, while still
    reporting `source='structured'` so the fuser's per-source counters and
    the health registry's labels stay accurate.
    """

    def __init__(self) -> None:
        return

    @property
    def source(self) -> str:
        return 'structured'

    @property
    def provides_inventory_estimate(self) -> bool:
        """Empty no-op sets are not inventory estimates — never use for drop-zero."""
        return False

    async def retrieve(self, intent: QueryIntent, top_k: int) -> CandidateSet:
        return CandidateSet(source='structured', candidates=[], latency_ms=0.0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _points_to_tuples(response: Any, payload_id_field: str) -> List[Tuple[str, float, Dict[str, Any]]]:
    """Map a `QueryResponse` (or list of points) to (item_id, score, payload) tuples.

    Qdrant 1.10+ returns `QueryResponse` with a `.points` attribute holding
    `ScoredPoint`s; older / alternate paths may return a bare list. Both
    shapes are handled.
    """
    points: Sequence[Any]
    if response is None:
        return []
    if hasattr(response, 'points'):
        points = response.points or []
    elif isinstance(response, (list, tuple)):
        points = response
    else:
        points = []
    out: List[Tuple[str, float, Dict[str, Any]]] = []
    for point in points:
        item_id = _extract_item_id(point, payload_id_field)
        score = _normalize_score(getattr(point, 'score', 0.0))
        payload = _filter_payload(getattr(point, 'payload', None))
        out.append((item_id, score, payload))
    return out


def _run_coro_sync(coro: Any, timeout: float) -> Any:
    """Run `coro` to completion from a sync context with a wall-clock timeout.

    Used by `QdrantVectorIndex.search` and `QdrantStructuredIndex.search` to
    bridge the sync `VectorIndex` / `StructuredIndex` interface to the async
    Qdrant client. The caller (the existing in-memory-style retriever) is
    *already* invoked inside `asyncio.to_thread(...)`, so we are guaranteed
    to be on a worker thread with no running event loop — `asyncio.run` is
    safe and creates an isolated loop per call.

    :raises asyncio.TimeoutError: When the coroutine exceeds `timeout`
    :raises QdrantUnavailableError: When called from inside a running loop
        (programming error — adapter contracts forbid it)
    """
    if timeout <= 0.0:
        raise RetrievalError("_run_coro_sync requires timeout > 0")
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is not None:
        raise QdrantUnavailableError(
            "QdrantVectorIndex/QdrantStructuredIndex.search invoked from a running "
            "event loop; callers must wrap in asyncio.to_thread (the existing "
            "VectorRetriever/StructuredRetriever already do this)"
        )

    async def _waited() -> Any:
        return await asyncio.wait_for(coro, timeout=float(timeout))

    return asyncio.run(_waited())


__all__ = [
    'QdrantClientFactory',
    'QdrantVectorIndex',
    'QdrantStructuredIndex',
    'QdrantHybridRetriever',
    'QdrantNoOpStructuredRetriever',
    'build_qdrant_filter',
    'extract_keyword_post_filters',
    'set_unavailable_filter_keys',
]
