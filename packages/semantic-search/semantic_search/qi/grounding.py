"""Entity grounding — validate extracted entity slots against a live inventory contract.

Ships three pieces:
1. ``InventoryContract`` interface (production-swappable).
2. ``StaticInventoryContract`` — config-fed allowlist (test/bootstrap path).
3. ``LiveInventoryContract`` — derives ``known_tlds`` from a live
   ``StructuredIndex`` with TTL caching and a static fallback when the live
   source is empty. Replaces the static-only grounding path so dropped
   entities reflect actual inventory rather than a hardcoded allowlist.

The grounder also produces *adapt-on-miss* substitutions for dropped TLDs
(``.ai -> .io / .tech``). Substitutions are logged for
observability and exposed via ``EntityGrounder.last_dropped`` so callers
that want to surface them as chips (search surface)
can read the most-recent drop set without changing the public ``ground()``
contract.
"""
import asyncio
import time
from threading import RLock  # noqa: TID251 - sync TTL cache; inventory contracts are sync
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Optional, Tuple

from semantic_search.core.logging_utils import get_logger
from semantic_search.contracts import AUCTION_TYPE_LABEL_TO_IDS, Entity, prefer_registrar_auction_values
from semantic_search.qi.slot_to_api_param import normalize_type_include_list_for_public
from semantic_search.qi.l0_llm_filter_extractor import entities_to_identified, identified_to_ground_entities

logger = get_logger(__name__)


def _expand_auction_to_ids(values: List[str]) -> List[str]:
    """Expand canonical auction type labels to numeric index payload IDs.

    Numeric strings ('16', '38') pass through unchanged.  Canonical labels
    expand via the shared :data:`semantic_search.contracts.AUCTION_TYPE_LABEL_TO_IDS`.
    Deduplication preserves insertion order so the output list is deterministic.

    When the caller already supplies explicit numeric IDs, canonical label
    expansion is skipped entirely — prevents sibling IDs from leaking in
    (e.g. "auction" expanding to both 16 and 38 when only 16 was requested).

    Registrar-scoped labels (``partner`` / ``godaddy``) win over generic
    ``auction``/``expiry``/``premium`` so opposing registrar IDs are not mixed in.
    """
    values = prefer_registrar_auction_values(values)
    result: List[str] = []
    seen: set = set()
    # If any value is an explicit numeric ID (not a canonical label key),
    # skip label expansion so we don't add sibling type IDs.
    has_explicit_numeric = any(str(v).lower() not in AUCTION_TYPE_LABEL_TO_IDS for v in values)
    for v in values:
        sv = str(v).lower()
        if sv in AUCTION_TYPE_LABEL_TO_IDS:
            if not has_explicit_numeric:
                for nid in sorted(AUCTION_TYPE_LABEL_TO_IDS[sv]):
                    if nid not in seen:
                        seen.add(nid)
                        result.append(nid)
        else:
            if sv not in seen:
                seen.add(sv)
                result.append(sv)
    return result


class InventoryContract:
    """Interface for entity grounding against live inventory."""

    def known_tlds(self) -> FrozenSet[str]:
        """Return the set of TLDs that exist in current inventory."""
        raise NotImplementedError

    def known_auction_types(self) -> FrozenSet[str]:
        """Return the set of auction types that exist in current inventory."""
        raise NotImplementedError


class StaticInventoryContract(InventoryContract):
    """Inventory contract backed by a static config-supplied allowlist."""

    def __init__(self, tlds: List[str], auction_types: List[str]):
        self._tlds = frozenset(t.lower() for t in tlds)
        self._auction_types = frozenset(a.lower() for a in auction_types)

    def known_tlds(self) -> FrozenSet[str]:
        return self._tlds

    def known_auction_types(self) -> FrozenSet[str]:
        return self._auction_types


class LiveInventoryContract(InventoryContract):
    """Live inventory contract — derives ``known_tlds`` from an iterable
    payload source (typically ``StructuredIndex.iter_payloads()``) with
    TTL caching and a ``StaticInventoryContract`` fallback.

    Replaces the static-only grounder so
    a domain like ``.ai`` is grounded against the actual inventory; if the
    live source returns zero TLDs (cold cache, ingest paused, scan failure)
    we transparently fall back to the static config so the grounder never
    silently drops every entity.

    :param payload_source: Callable returning ``Iterable[Dict[str, Any]]``
        — typically ``StructuredIndex.iter_payloads``. Each payload may
        carry a ``tld`` field; missing / non-string values are skipped.
    :param fallback: ``StaticInventoryContract`` consulted when the live
        source yields zero TLDs.
    :param ttl_seconds: Cache TTL for the live aggregate. Higher values
        reduce scan overhead at the cost of staleness; the snapshot
        invalidation hook (registry wiring) calls ``invalidate()`` on
        every ingest bump so the TTL is the upper bound, not the typical
        refresh interval.
    """

    def __init__(
        self,
        payload_source: Callable[[], Iterable[Dict[str, Any]]],
        fallback: 'StaticInventoryContract',
        ttl_seconds: float,
    ):
        if payload_source is None:
            raise ValueError("LiveInventoryContract requires a non-null payload_source")
        if fallback is None:
            raise ValueError("LiveInventoryContract requires a static fallback")
        if not isinstance(ttl_seconds, (int, float)) or ttl_seconds <= 0:
            raise ValueError("LiveInventoryContract.ttl_seconds must be a positive number")
        self._payload_source = payload_source
        self._fallback = fallback
        self._ttl = float(ttl_seconds)
        self._lock = RLock()
        self._cached_tlds: Optional[FrozenSet[str]] = None
        self._cached_at: float = 0.0

    def known_tlds(self) -> FrozenSet[str]:
        """Return the live (cached) TLD set; falls back to static when empty."""
        with self._lock:
            now = time.monotonic()
            if self._cached_tlds is not None and (now - self._cached_at) < self._ttl:
                return self._cached_tlds
            live = self._scan()
            if live:
                self._cached_tlds = live
                self._cached_at = now
                return live
            # Live source empty — return the static fallback but DO NOT cache
            # it (so the next call re-attempts the live scan rather than
            # locking us into the fallback for the full TTL window).
            return self._fallback.known_tlds()

    def known_auction_types(self) -> FrozenSet[str]:
        """Auction types are a small closed set — pass through to the static
        fallback (no benefit to scanning live payloads for this attribute)."""
        return self._fallback.known_auction_types()

    def invalidate(self) -> None:
        """Drop the cached TLD set; next ``known_tlds()`` re-scans the live
        source. Wired as a ``SnapshotInvalidationHook`` against the ingest
        snapshot registry so a new listing event immediately surfaces."""
        with self._lock:
            self._cached_tlds = None
            self._cached_at = 0.0

    def _scan(self) -> FrozenSet[str]:
        """Walk the payload source and collect distinct TLD values with confirmed bid activity."""
        try:
            payloads: Iterable = self._payload_source() or ()
        except Exception as e:  # noqa: BLE001 — never fail the cascade on grounding source error
            logger.warning(f"live_inventory_scan_failed error_type={type(e).__name__} error={str(e)}")
            return frozenset()
        seen: set = set()
        for payload in payloads:
            if not isinstance(payload, dict):
                continue
            raw = payload.get('tld')
            if not (isinstance(raw, str) and raw):
                continue
            bc = payload.get('bid_count')
            if bc is not None and int(bc) <= 0:
                continue
            seen.add(raw.strip().lower())
        return frozenset(seen)


class ClickHouseInventoryContract(InventoryContract):

    def __init__(self, fallback: InventoryContract):
        if fallback is None:
            raise ValueError("ClickHouseInventoryContract requires a non-null fallback")
        self._fallback = fallback
        self._lock = RLock()
        self._cached_tlds: Optional[FrozenSet[str]] = None

    def known_tlds(self) -> FrozenSet[str]:
        with self._lock:
            if self._cached_tlds:
                return self._cached_tlds
        return self._fallback.known_tlds()

    def known_auction_types(self) -> FrozenSet[str]:
        return self._fallback.known_auction_types()

    def update(self, tlds: FrozenSet[str]) -> None:
        if not isinstance(tlds, frozenset):
            raise TypeError("ClickHouseInventoryContract.update requires a frozenset")
        with self._lock:
            self._cached_tlds = tlds
        logger.info(f"ch_tld_inventory_updated tld_count={len(tlds)}")


class ClickHouseTLDRefreshDriver:

    def __init__(
        self,
        contract: ClickHouseInventoryContract,
        client: Any,
        database: str,
        lookback_days: int,
        interval_seconds: float,
        tld_query_sql: str,
    ):
        if contract is None:
            raise ValueError("ClickHouseTLDRefreshDriver requires a non-null contract")
        if client is None:
            raise ValueError("ClickHouseTLDRefreshDriver requires a non-null client")
        if not database:
            raise ValueError("ClickHouseTLDRefreshDriver requires a non-empty database")
        if lookback_days < 1:
            raise ValueError("ClickHouseTLDRefreshDriver.lookback_days must be >= 1")
        if interval_seconds <= 0:
            raise ValueError("ClickHouseTLDRefreshDriver.interval_seconds must be > 0")
        if not tld_query_sql:
            raise ValueError("ClickHouseTLDRefreshDriver requires a non-empty tld_query_sql")
        self._contract = contract
        self._client = client
        self._database = database
        self._lookback_days = int(lookback_days)
        self._interval = float(interval_seconds)
        self._tld_query_sql = tld_query_sql
        self._task: Optional[asyncio.Task] = None

    async def _refresh_once(self) -> bool:
        sql = self._tld_query_sql.format(
            database=self._database,
            lookback_days=self._lookback_days,
        )
        try:
            rows, _cols, _latency_ms = await self._client.execute_query(
                query=sql,
                timeout_seconds=10.0,
                _skip_mv_freshness_probe=True,
            )
            tlds = frozenset(
                str(row.get('tld', '') or '').strip().lower()
                for row in rows
                if row.get('tld')
            )
            if tlds:
                self._contract.update(tlds)
                logger.info(f"ch_tld_refresh_ok tld_count={len(tlds)} lookback_days={self._lookback_days}")
                return True
            logger.warning(f"ch_tld_refresh_empty lookback_days={self._lookback_days} action=keep_prior_cache")
            return False
        except Exception as e:  # noqa: BLE001
            logger.warning(f"ch_tld_refresh_failed error_type={type(e).__name__} error={str(e)} action=keep_prior_cache")
            return False

    async def _run_loop(self) -> None:
        await self._refresh_once()
        while True:
            try:
                await asyncio.sleep(self._interval)
            except asyncio.CancelledError:
                logger.info("ch_tld_refresh_driver_stopped")
                return
            await self._refresh_once()

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            f"ch_tld_refresh_driver_started interval_seconds={self._interval} "
            f"lookback_days={self._lookback_days}"
        )

    async def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(self._task), timeout=2.0)
            except Exception:  # noqa: BLE001
                pass
            self._task = None


class EntityGrounder:
    """Filter entity values to those present in the live inventory contract,
    and emit *adapt-on-miss* substitutions for dropped TLDs.

    :param inventory: InventoryContract - Live inventory adapter
    :param tld_substitutions: Optional[Dict[str, List[str]]] - Adapt-on-miss
        substitution map (``.ai -> .io, .tech``). When a TLD value is dropped during grounding,
        the substitution map (if present) provides alternative TLDs the
        downstream surface can offer as chips. Keys are looked up
        case-insensitively.
    """

    def __init__(self, inventory: InventoryContract, tld_substitutions: Optional[Dict[str, List[str]]] = None):
        self._inventory = inventory
        # Normalize substitution keys + values to lowercase up-front so the
        # hot-path lookup is a single dict hit per dropped value.
        self._substitutions: Dict[str, Tuple[str, ...]] = {}
        if tld_substitutions:
            for k, v in tld_substitutions.items():
                if not isinstance(k, str) or not isinstance(v, list):
                    continue
                lowered = tuple(str(item).strip().lower() for item in v if isinstance(item, str) and item)
                if lowered:
                    self._substitutions[k.strip().lower()] = lowered
        # Most-recent drop record exposed for callers that surface chips.
        # Format: list of (entity_name, dropped_value, suggested_alts).
        # Reset at the start of every ``ground()`` call so callers always
        # read the latest classification's drop set, never a stale one.
        self.last_dropped: List[Tuple[str, str, Tuple[str, ...]]] = []

    def adapt_on_miss(self, value: str) -> Tuple[str, ...]:
        """Return the configured substitution list for a dropped TLD value
        (empty tuple when no mapping exists). Public so the search surface can request suggestions outside the grounding path.
        :param value: str - The dropped value to look up
        :return: Tuple[str, ...] - Suggested alternatives (lowercased)
        """
        if not isinstance(value, str) or not value:
            return ()
        return self._substitutions.get(value.strip().lower(), ())

    def ground(self, entities: List[Entity]) -> List[Entity]:
        """Drop unrecognized values from list-valued entities. Pass-through on others.

        Side effect: populates ``self.last_dropped`` with the (entity_name,
        dropped_value, substitution_alts) tuples for the current call. The
        list is reset at the top of every invocation so callers reading
        ``last_dropped`` after each ``ground_slices`` see only the most
        recent classification's drops (never accumulated).

        :param entities: List[Entity] - Raw extracted entities
        :return: List[Entity] - Grounded entities (entities with all values rejected are dropped)
        """
        self.last_dropped = []
        if not entities:
            return []
        out: List[Entity] = []
        known_tlds = self._inventory.known_tlds()
        known_auctions = self._inventory.known_auction_types()
        for ent in entities:
            if ent.name == 'tld' and isinstance(ent.value, list):
                kept: List = []
                for v in ent.value:
                    sv = str(v).lower()
                    if sv in known_tlds:
                        kept.append(v)
                    else:
                        alts = self.adapt_on_miss(sv)
                        self.last_dropped.append(('tld', sv, alts))
                        if alts:
                            logger.info(f"qi_grounding_substitution name=tld dropped={sv} suggested={list(alts)}")
                        else:
                            logger.warning(f"qi_grounding_dropped_entity name=tld dropped_value={sv} no_substitution=true")
                if not kept:
                    logger.warning(f"qi_grounding_dropped_entity name=tld dropped={len(ent.value)} suggestions={[s[2] for s in self.last_dropped if s[0] == 'tld']}")
                    continue
                out.append(Entity(name=ent.name, value=kept, confidence=ent.confidence, source=ent.source, chip_kind=ent.chip_kind))
            elif ent.name == 'auction_type' and isinstance(ent.value, list):
                kept = [v for v in ent.value if str(v).lower() in known_auctions]
                if not kept:
                    for v in ent.value:
                        self.last_dropped.append(('auction_type', str(v).lower(), ()))
                    logger.warning(f"qi_grounding_dropped_entity name=auction_type dropped={len(ent.value)}")
                    continue
                # Expand canonical labels -> numeric IDs so all retrieval backends
                # (Qdrant, SQL, InMemory) receive the storage-format IDs ('16', '38')
                # rather than canonical labels ('auction', 'expiry') that only the
                # InMemory backend could expand locally.  Numeric IDs pass through
                # unchanged; unknown values also pass through rather than being dropped.
                expanded = _expand_auction_to_ids(kept)
                out.append(Entity(name=ent.name, value=expanded, confidence=ent.confidence, source=ent.source, chip_kind=ent.chip_kind))
            else:
                out.append(ent)
        return out


def _as_filter_str_list(value: Any) -> List[str]:
    """Normalize identified_filters value to a non-empty string list."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if '|' in text:
            return [p.strip() for p in text.split('|') if p.strip()]
        if ',' in text:
            return [p.strip() for p in text.split(',') if p.strip()]
        return [text]
    text = str(value).strip()
    return [text] if text else []


# Include/exclude pairs stripped when the same value appears on both sides.
_INCLUDE_EXCLUDE_PAIRS: Tuple[Tuple[str, str], ...] = (
    ('tldIncludeList', 'tldExcludeList'),
    ('typeIncludeList', 'typeExcludeList'),
    ('keyword_contains', 'keyword_contains_exclude'),
    ('topic_include', 'topic_exclude'),
)
_TYPE_FILTER_SLOTS = frozenset({'typeIncludeList', 'typeExcludeList'})


def _norm_filter_token(value: str) -> str:
    return value.strip().lower()


def _repack_filter_list_value(original: Any, kept: List[str], *, is_type: bool) -> Any:
    """Rebuild a filter value after contradiction strip; preserve list vs scalar/pipe shape."""
    if is_type:
        return normalize_type_include_list_for_public(kept)
    if isinstance(original, list):
        return list(kept)
    if len(kept) == 1 and not (
        isinstance(original, str) and ('|' in original or ',' in original)
    ):
        return kept[0]
    if isinstance(original, str) and ',' in original and '|' not in original:
        return ','.join(kept)
    return '|'.join(kept)


def _strip_include_exclude_conflicts(
    identified: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Remove values that appear in both include and exclude for the same dimension.

    Intersection is dropped from both sides; empty slots are omitted. Prefer empty
    chips over contradictory ones for Phase 1 UX learning.
    """
    by_name = {
        str(item.get('name') or '').strip(): item
        for item in identified
        if str(item.get('name') or '').strip()
    }
    remove_by_slot: Dict[str, FrozenSet[str]] = {}
    for inc_name, exc_name in _INCLUDE_EXCLUDE_PAIRS:
        inc = by_name.get(inc_name)
        exc = by_name.get(exc_name)
        if inc is None or exc is None:
            continue
        inc_set = {_norm_filter_token(v) for v in _as_filter_str_list(inc.get('value'))}
        exc_set = {_norm_filter_token(v) for v in _as_filter_str_list(exc.get('value'))}
        overlap = inc_set & exc_set
        if not overlap:
            continue
        logger.info(
            f"qi_filter_contradiction_stripped include={inc_name} exclude={exc_name} "
            f"values={sorted(overlap)}"
        )
        frozen = frozenset(overlap)
        remove_by_slot[inc_name] = frozen
        remove_by_slot[exc_name] = frozen

    if not remove_by_slot:
        return identified

    out: List[Dict[str, Any]] = []
    for item in identified:
        name = str(item.get('name') or '').strip()
        remove = remove_by_slot.get(name)
        if remove is None:
            out.append(item)
            continue
        kept = [v for v in _as_filter_str_list(item.get('value')) if _norm_filter_token(v) not in remove]
        if not kept:
            continue
        new_item = dict(item)
        new_item['value'] = _repack_filter_list_value(
            item.get('value'), kept, is_type=name in _TYPE_FILTER_SLOTS,
        )
        out.append(new_item)
    return out


def _fix_inverted_price_bounds(
    identified: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Swap ``minPrice`` / ``maxPrice`` when min > max (LLM inversion hygiene)."""
    min_idx: Optional[int] = None
    max_idx: Optional[int] = None
    min_num: Optional[float] = None
    max_num: Optional[float] = None
    for i, item in enumerate(identified):
        name = str(item.get('name') or '').strip()
        if name == 'minPrice':
            try:
                min_num = float(item.get('value'))
                min_idx = i
            except (TypeError, ValueError):
                pass
        elif name == 'maxPrice':
            try:
                max_num = float(item.get('value'))
                max_idx = i
            except (TypeError, ValueError):
                pass
    if min_idx is None or max_idx is None or min_num is None or max_num is None:
        return identified
    if min_num <= max_num:
        return identified
    logger.info(
        f"qi_filter_price_bounds_swapped minPrice={min_num} maxPrice={max_num}"
    )
    out = [dict(item) for item in identified]
    min_orig = out[min_idx]['value']
    max_orig = out[max_idx]['value']
    out[min_idx]['value'] = max_orig
    out[max_idx]['value'] = min_orig
    return out


_ENTITY_INCLUDE_EXCLUDE_PAIRS: Tuple[Tuple[str, str], ...] = (
    ('tld', 'tldExcludeList'),
    ('auction_type', 'typeExcludeList'),
    ('keyword_contains', 'keyword_contains_exclude'),
    ('topic_include', 'topic_exclude'),
)

# Exclude-slot internal name -> the name EntityGrounder.ground() recognizes for
# inventory checking. Centralizes the relabel-for-grounding trick in one place
# so both qie_only and full-search route through the identical algorithm.
_EXCLUDE_TO_GROUND_NAME: Dict[str, str] = {'tldExcludeList': 'tld', 'typeExcludeList': 'auction_type'}
_GROUND_TO_EXCLUDE_NAME: Dict[str, str] = {v: k for k, v in _EXCLUDE_TO_GROUND_NAME.items()}


def _strip_entity_include_exclude_conflicts(entities: List[Entity]) -> List[Entity]:
    """Entity-native port of ``_strip_include_exclude_conflicts`` (dict version).

    Removes values present in both include and exclude for the same dimension
    (tld/auction_type/keyword_contains/topic); empties the entity when nothing
    remains. Entity.value for these slots is already a list by construction
    (``_coerce_l0_filter_value``), so no string-splitting fallback is needed.
    """
    by_name = {e.name: e for e in entities}
    remove_by_slot: Dict[str, FrozenSet[str]] = {}
    for inc_name, exc_name in _ENTITY_INCLUDE_EXCLUDE_PAIRS:
        inc = by_name.get(inc_name)
        exc = by_name.get(exc_name)
        if inc is None or exc is None:
            continue
        inc_set = {_norm_filter_token(v) for v in _as_filter_str_list(inc.value)}
        exc_set = {_norm_filter_token(v) for v in _as_filter_str_list(exc.value)}
        overlap = inc_set & exc_set
        if not overlap:
            continue
        logger.info(
            f"qi_entity_contradiction_stripped include={inc_name} exclude={exc_name} "
            f"values={sorted(overlap)}"
        )
        frozen = frozenset(overlap)
        remove_by_slot[inc_name] = frozen
        remove_by_slot[exc_name] = frozen

    if not remove_by_slot:
        return entities

    out: List[Entity] = []
    for ent in entities:
        remove = remove_by_slot.get(ent.name)
        if remove is None:
            out.append(ent)
            continue
        kept = [v for v in _as_filter_str_list(ent.value) if _norm_filter_token(v) not in remove]
        if not kept:
            continue
        new_value = kept if isinstance(ent.value, list) else kept[0]
        out.append(
            ent if new_value == ent.value else
            Entity(name=ent.name, value=new_value, confidence=ent.confidence, source=ent.source, chip_kind=ent.chip_kind)
        )
    return out


def _fix_entity_price_bounds(entities: List[Entity]) -> List[Entity]:
    """Entity-native port of ``_fix_inverted_price_bounds`` (dict version)."""
    min_idx: Optional[int] = None
    max_idx: Optional[int] = None
    min_num: Optional[float] = None
    max_num: Optional[float] = None
    for i, ent in enumerate(entities):
        if ent.name == 'price_min':
            try:
                min_num = float(ent.value)
                min_idx = i
            except (TypeError, ValueError):
                pass
        elif ent.name == 'price_max':
            try:
                max_num = float(ent.value)
                max_idx = i
            except (TypeError, ValueError):
                pass
    if min_idx is None or max_idx is None or min_num is None or max_num is None:
        return entities
    if min_num <= max_num:
        return entities
    logger.info(f"qi_entity_price_bounds_swapped price_min={min_num} price_max={max_num}")
    out = list(entities)
    min_ent, max_ent = out[min_idx], out[max_idx]
    out[min_idx] = Entity(name=min_ent.name, value=max_ent.value, confidence=min_ent.confidence, source=min_ent.source, chip_kind=min_ent.chip_kind)
    out[max_idx] = Entity(name=max_ent.name, value=min_ent.value, confidence=max_ent.confidence, source=max_ent.source, chip_kind=max_ent.chip_kind)
    return out


def ground_hard_entities(
    entities: List[Entity],
    grounder: 'EntityGrounder',
) -> Tuple[List[Entity], int]:
    """Entity-native grounding core shared by qie_only and full-search.

    The ONE place the algorithm lives: grounds ``tld``/``auction_type``
    (include) and ``tldExcludeList``/``typeExcludeList`` (exclude — relabeled
    to their ``ground()``-recognized name, then relabeled back) in two
    separate ``EntityGrounder.ground()`` passes so include and exclude never
    collide inside one call, then strips include/exclude contradictions and
    repairs inverted price bounds. Everything else passes through unchanged.

    Exclude-side auction-type values are left in whatever form ``ground()``
    produces (numeric storage IDs, per its label-expansion side effect) —
    that is the form ``retrieval.qdrant_adapter`` / ``structured_retriever``
    match against the payload's ``auction_type`` field for both include and
    exclude ``must_not`` conditions.

    :param entities: List[Entity] - hard entities, internal names, pre-ground
    :param grounder: EntityGrounder - live/static inventory grounder
    :return: Tuple[List[Entity], int] - (grounded entities, value-drop count)
    """
    if grounder is None:
        raise ValueError("ground_hard_entities requires a non-null EntityGrounder")
    if not entities:
        return [], 0

    include_src = [e for e in entities if e.name in ('tld', 'auction_type')]
    exclude_src = [e for e in entities if e.name in _EXCLUDE_TO_GROUND_NAME]
    other = [
        e for e in entities
        if e.name not in ('tld', 'auction_type') and e.name not in _EXCLUDE_TO_GROUND_NAME
    ]

    drop_count = 0
    grounded_include: List[Entity] = []
    if include_src:
        grounded_include = grounder.ground(include_src)
        drop_count += len(getattr(grounder, 'last_dropped', None) or [])

    grounded_exclude: List[Entity] = []
    if exclude_src:
        relabeled = [
            Entity(
                name=_EXCLUDE_TO_GROUND_NAME[e.name],
                value=e.value, confidence=e.confidence, source=e.source, chip_kind=e.chip_kind,
            )
            for e in exclude_src
        ]
        grounded_relabeled = grounder.ground(relabeled)
        drop_count += len(getattr(grounder, 'last_dropped', None) or [])
        grounded_exclude = [
            Entity(
                name=_GROUND_TO_EXCLUDE_NAME[e.name],
                value=e.value, confidence=e.confidence, source=e.source, chip_kind=e.chip_kind,
            )
            for e in grounded_relabeled
        ]

    out = other + grounded_include + grounded_exclude
    out = _strip_entity_include_exclude_conflicts(out)
    out = _fix_entity_price_bounds(out)
    return out, int(drop_count)


def sanitize_identified_filters(
    identified: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Contradiction strip + inverted price swap without inventory grounding.

    Use when ``EntityGrounder`` is unavailable so Phase 1 still drops
    include∩exclude conflicts and repairs ``minPrice``/``maxPrice`` inversion.
    """
    if not identified:
        return []
    return _fix_inverted_price_bounds(_strip_include_exclude_conflicts(identified))


def ground_identified_filters(
    identified: List[Dict[str, Any]],
    grounder: 'EntityGrounder',
) -> Tuple[List[Dict[str, Any]], int]:
    """Ground TLD/type include **and** exclude lists; strip contradictions; fix price bounds.

    Phase 1 ``qie_only`` path: thin dict<->Entity adapter around ``ground_hard_entities``,
    the same Entity-native grounding core ``QIEngine._ground_slices`` (full-search) calls
    directly — one algorithm, two callers, so both modes ground identically.

    :param identified: List[Dict] - ``[{name, value, source}, ...]`` from L0 extract
    :param grounder: EntityGrounder - Static or live inventory grounder
    :return: Tuple[List[Dict], int] - (grounded filters, value-level drop count)
    """
    if grounder is None:
        raise ValueError("ground_identified_filters requires a non-null EntityGrounder")
    if not identified:
        return [], 0

    entities = identified_to_ground_entities(identified)
    grounded, drop_count = ground_hard_entities(entities, grounder)
    out = entities_to_identified(grounded, frozenset())

    # Repack generic multi-value slots (keyword_contains, etc.) back to the
    # caller's original scalar/pipe/comma shape. typeIncludeList/typeExcludeList
    # are excluded — entities_to_identified already shapes those via
    # normalize_type_include_list_for_public.
    original_by_name = {
        str(item.get('name') or '').strip(): item.get('value') for item in identified
    }
    for item in out:
        name = item['name']
        if name in _TYPE_FILTER_SLOTS or not isinstance(item['value'], list):
            continue
        original_value = original_by_name.get(name)
        if original_value is not None and not isinstance(original_value, list):
            item['value'] = _repack_filter_list_value(original_value, item['value'], is_type=False)
    return out, int(drop_count)
