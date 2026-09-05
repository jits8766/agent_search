"""ExploreComposer — assembles the trending / ending-soon / last-hour / latest / high-volume / fresh / last-week rails (+ fallback).

Two surfaces share the composer:
  1. ``compose_landing(user_id, request_id)`` — the public landing-rail endpoint
  2. ``compose_fallback(intent, request_id)`` — the orchestrator's zero-result guard

Both return the same :class:`LandingRailResponse` shape; only ``source`` differs
so the dashboard can split usefulness metrics by call path.

RRF fusion: all rail item lists are fused via Reciprocal Rank Fusion (k from config) in ``compose_fallback``. Landing rail keeps ordered append (user-visible grouping).

Dedupe: item surfaces in multiple rails is kept ONLY in first rail per RAIL_ORDER (last_hour > trending > ending_soon > latest > high_volume > fresh > last_week > fallback).
"""
import asyncio
import inspect
import re
import time
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from semantic_search.cache.lru_ttl import LRUTTLCache
from semantic_search.config.models import ExploreConfig
from semantic_search.contracts import ExploreCard, ExploreRail, LandingRailResponse, QueryIntent, RankedItem, RankedResults
from semantic_search.core.exceptions import ValidationError
from semantic_search.core.logging_utils import get_logger
from semantic_search.explore.sources import EndingSoonSource, TrendingSource
from semantic_search.retrieval.structured_retriever import extract_filters_from_intent, item_matches_filters

_ENDING_SOON_RE = re.compile(
    r'\b(?:ending|closing|expiring|final\s+hours?|last\s+chance|pending\s+delete|closing\s+tonight)\b',
    re.IGNORECASE,
)
_TRENDING_RE = re.compile(
    r'\b(?:trending|popular|hot|viral|most\s+(?:bid|watched|viewed)|highest\s+bid|most\s+active)\b',
    re.IGNORECASE,
)
_FRESH_RE = re.compile(
    r'\b(?:fresh|just\s+listed|new\s+(?:arrival|listing|domain)|recently\s+(?:added|listed)|no\s+bids?\s+yet)\b',
    re.IGNORECASE,
)


def _rail_hint_from_intent(intent: QueryIntent) -> Optional[FrozenSet[str]]:
    """Infer rail hints from entities/signals (ending_soon -> {ending_soon, last_hour}, etc)."""
    q = intent.raw_query or ''
    entity_names = {getattr(e, 'name', '') for s in (intent.slices or []) for e in (s.entities or [])}
    if 'time_remaining_max' in entity_names or _ENDING_SOON_RE.search(q):
        return frozenset({'ending_soon', 'last_hour'})
    if _TRENDING_RE.search(q):
        return frozenset({'trending', 'high_volume'})
    if _FRESH_RE.search(q):
        return frozenset({'fresh', 'latest'})
    return None

logger = get_logger(__name__)

# Rails emitted in priority order. last_hour first (most urgent), fallback last.
RAIL_ORDER = ('last_hour', 'high_traffic', 'trending', 'ending_soon', 'latest', 'high_volume', 'watch_density', 'fresh', 'last_week', 'fallback')

# Per-rail TTL (seconds). Rail data is user-agnostic; shorter TTL for time-sensitive rails.
_RAIL_CACHE_TTL_S: Dict[str, int] = {
    'last_hour':   10,
    'trending':    60,
    'ending_soon': 15,
    'latest':      20,
    'high_volume':   60,
    'watch_density': 120,
    'high_traffic':  30,
    'fresh':         30,
    'last_week':     120,
}

ExploreSource = object  # protocol alias for type hints


def _rrf_fuse(
    rail_item_lists: List[Tuple[str, List[RankedItem]]],
    k: int,
) -> List[RankedItem]:
    """RRF fusion: 1/(k+rank) per item per rail, aggregate, sort descending."""
    scores: Dict[str, float] = {}
    first_seen: Dict[str, RankedItem] = {}
    for _rail_id, items in rail_item_lists:
        for rank, item in enumerate(items, start=1):
            iid = item.item_id
            scores[iid] = scores.get(iid, 0.0) + 1.0 / (k + rank)
            if iid not in first_seen:
                first_seen[iid] = item
    fused = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [RankedItem(item_id=iid, fused_score=scores[iid], contributing_sources=first_seen[iid].contributing_sources, payload=first_seen[iid].payload) for iid in fused]


class ExploreComposer:
    """Compose landing/fallback rails (trending, ending_soon, etc) via RRF + dedup.
    :param latest: Optional source - Domains expiring in next 7 days (newest-first)
    :param high_volume: Optional source - High-bid-activity domains (bid_count >= threshold)
    :param fresh_listings: Optional source - Zero-bid fresh-arrival domains
    :param last_week: Optional source - Long-duration auctions (> 7 days remaining)
    """

    def __init__(
        self,
        config: ExploreConfig,
        trending: TrendingSource,
        ending_soon: EndingSoonSource,
        last_hour: Optional[object] = None,
        latest: Optional[object] = None,
        high_volume: Optional[object] = None,
        fresh_listings: Optional[object] = None,
        last_week: Optional[object] = None,
        watch_density: Optional[object] = None,
        high_traffic: Optional[object] = None,
    ) -> None:
        if config is None:
            raise ValidationError("ExploreComposer requires an ExploreConfig")
        if trending is None:
            raise ValidationError("ExploreComposer requires a TrendingSource")
        if ending_soon is None:
            raise ValidationError("ExploreComposer requires an EndingSoonSource")
        self._config = config
        self._trending = trending
        self._ending_soon = ending_soon
        self._last_hour = last_hour
        self._latest = latest
        self._high_volume = high_volume
        self._fresh_listings = fresh_listings
        self._last_week = last_week
        self._watch_density = watch_density
        self._high_traffic = high_traffic
        self._rail_caches: Dict[str, LRUTTLCache] = {
            r: LRUTTLCache(max_entries=32, ttl_seconds=t, ttl_jitter_seconds=3)
            for r, t in _RAIL_CACHE_TTL_S.items()
        }

    @property
    def enabled(self) -> bool:
        """Master toggle (mirrors `explore.enabled`)."""
        return bool(self._config.enabled)

    @property
    def fallback_source_title(self) -> str:
        """Title for the fallback rail, as configured."""
        return str(self._config.fallback.source.title)

    async def compose_landing(
        self,
        user_id: Optional[str],
        request_id: Optional[str] = None,
        max_per_rail_override: Optional[int] = None,
        horizon_override: Optional[float] = None,
    ) -> LandingRailResponse:
        """Compose the public landing-rail response.

        :param user_id: Optional[str] - Authenticated user id (None permitted for anonymous landing)
        :param request_id: Optional[str] - Caller-provided correlation id
        :param max_per_rail_override: Optional[int] - When set, overrides the per-rail max_items cap
        :param horizon_override: Optional[float] - When set, overrides the ending-soon time window (seconds)
        :return: LandingRailResponse
        :raises ValidationError: When the composer is disabled
        """
        if not self.enabled:
            raise ValidationError("ExploreComposer is disabled (set explore.enabled=true)")
        rid = request_id if (request_id and isinstance(request_id, str)) else LandingRailResponse.new_request_id()
        _rail_timeout_landing = float(getattr(self._config.zero_result_guard, 'rail_timeout_seconds', 3.0))
        rails = await self._compose_rails(user_id=user_id, max_per_rail_override=max_per_rail_override, horizon_override=horizon_override, rail_timeout_s=_rail_timeout_landing)
        response = LandingRailResponse(request_id=rid, user_id=user_id, rails=rails, generated_at=time.time(), source='landing_rail_endpoint')
        logger.info(f"explore_landing_composed request_id={rid} user_id_present={'yes' if user_id else 'no'} rails={[r.rail_id for r in rails]} cards={[len(r.cards) for r in rails]}")
        return response

    async def compose_fallback(
        self,
        intent: QueryIntent,
        max_per_rail_override: Optional[int] = None,
        user_id: Optional[str] = None,
        semantic_items: Optional[List[RankedItem]] = None,
        semantic_items_fut: Optional['asyncio.Task[List[RankedItem]]'] = None,
        exclude_hard_filter_slots: Optional['frozenset[str]'] = None,
    ) -> Tuple[RankedResults, LandingRailResponse]:
        """Compose the zero-result fallback — RRF-fused across all rails + optional semantic results.

        Rails are fetched in parallel. Results are fused via Reciprocal Rank Fusion using
        ``config.zero_result_guard.rrf_k``. Hard filters from the original intent are applied
        after fusion so the fallback never violates explicit user constraints.

        When the intent carries a ``time_remaining_max`` entity (e.g. "ending soon this week"),
        the value is used as ``horizon_override`` on the ending-soon rail so the source window
        matches the user's stated time constraint. The same value participates in post-fusion
        hard-filter evaluation via ``extract_filters_from_intent``.

        :param intent: QueryIntent - The original (zero-result) classification
        :param max_per_rail_override: Optional[int] - Per-rail cap override
        :param user_id: Optional[str] - Authenticated user id
        :param semantic_items: Optional[List[RankedItem]] - Pre-computed vector search results
            to include in the RRF fusion (caller-supplied synchronously)
        :param semantic_items_fut: Optional[asyncio.Task] - In-flight semantic retrieval task.
            When provided, ``_compose_rails`` and this task run concurrently via
            ``asyncio.gather`` so neither blocks the other. ``semantic_items`` is ignored
            when ``semantic_items_fut`` is set.
        :param exclude_hard_filter_slots: Optional[frozenset[str]] - Entity slot names to
            exclude from post-fusion hard filtering. Use for slots that select a rail (e.g.
            ``time_remaining_max`` tunes the ending-soon rail via horizon_override) rather than
            filtering items within it. ``time_remaining_max`` is always used for horizon_override
            regardless of this exclusion.
        :return: (RankedResults, LandingRailResponse)
        :raises ValidationError: When the composer is disabled
        """
        if not self.enabled:
            raise ValidationError("ExploreComposer is disabled (set explore.enabled=true)")
        if intent is None:
            raise ValidationError("compose_fallback requires a QueryIntent")
        hard_filters = extract_filters_from_intent(intent)
        _trm = hard_filters.get('time_remaining_max')
        horizon_override: Optional[float] = float(_trm) if _trm is not None else None
        _post_filters = (
            {k: v for k, v in hard_filters.items() if k not in exclude_hard_filter_slots}
            if exclude_hard_filter_slots else hard_filters
        )
        _rail_timeout = float(getattr(self._config.zero_result_guard, 'rail_timeout_seconds', 3.0))
        _active_rails = _rail_hint_from_intent(intent)
        if semantic_items_fut is not None:
            _rails_res, _sem_res = await asyncio.gather(
                self._compose_rails(
                    user_id=user_id,
                    max_per_rail_override=max_per_rail_override,
                    horizon_override=horizon_override,
                    rail_timeout_s=_rail_timeout,
                    active_rails_override=_active_rails,
                    hard_filters=hard_filters,
                ),
                asyncio.shield(semantic_items_fut),
                return_exceptions=True,
            )
            if isinstance(_rails_res, BaseException):
                raise _rails_res
            rails = _rails_res
            if isinstance(_sem_res, BaseException):
                logger.warning(f"explore_fallback_semantic_fut_failed request_id={intent.request_id} error_type={type(_sem_res).__name__}")
                _resolved_semantic: Optional[List[RankedItem]] = None
            else:
                _resolved_semantic = list(_sem_res) if _sem_res else None
        else:
            rails = await self._compose_rails(
                user_id=user_id,
                max_per_rail_override=max_per_rail_override,
                horizon_override=horizon_override,
                rail_timeout_s=_rail_timeout,
                active_rails_override=_active_rails,
                hard_filters=hard_filters,
            )
            _resolved_semantic = semantic_items
        rail_response = LandingRailResponse(request_id=intent.request_id, user_id=user_id, rails=rails, generated_at=time.time(), source='zero_result_guard')

        # Build per-rail RankedItem lists for RRF input.
        rrf_k = int(self._config.zero_result_guard.rrf_k)
        rail_item_lists: List[Tuple[str, List[RankedItem]]] = []
        for rail in rails:
            rail_items: List[RankedItem] = []
            for c in rail.cards:
                payload = dict(c.payload)
                payload['source_rail'] = c.source_rail
                rail_items.append(RankedItem(item_id=c.item_id, fused_score=float(c.fused_score), contributing_sources=['sql'], payload=payload))
            if rail_items:
                rail_item_lists.append((rail.rail_id, rail_items))

        if _resolved_semantic:
            rail_item_lists.append(('semantic', list(_resolved_semantic)))

        fused = _rrf_fuse(rail_item_lists, k=rrf_k) if rail_item_lists else []

        # Apply hard filters post-fusion. Rail-selector slots (e.g. time_remaining_max,
        # startTimeAfter) are excluded via _post_filters when the caller signals they
        # drive rail selection rather than item-level filtering.
        dropped_by_filters = 0
        items: List[RankedItem] = []
        seen: set = set()
        for item in fused:
            if item.item_id in seen:
                continue
            seen.add(item.item_id)
            if _post_filters and not item_matches_filters(item.payload, _post_filters):
                dropped_by_filters += 1
                continue
            items.append(item)

        if items:
            failure_mode = None
        elif _post_filters and dropped_by_filters > 0:
            failure_mode = 'inventory_empty_under_filters'
        else:
            failure_mode = 'inventory_empty'

        ranked = RankedResults(request_id=intent.request_id, items=items, total_candidates=len(items), fusion_latency_ms=0.0, cache_hit=None, failure_mode=failure_mode)
        logger.info(
            f"explore_fallback_composed request_id={intent.request_id} "
            f"rails={[r.rail_id for r in rails]} cards={[len(r.cards) for r in rails]} "
            f"semantic_in={len(_resolved_semantic) if _resolved_semantic else 0} "
            f"total_fused={len(fused)} total_items={len(items)} rrf_k={rrf_k}"
        )
        return ranked, rail_response

    async def _compose_rails(
        self,
        user_id: Optional[str],
        max_per_rail_override: Optional[int],
        horizon_override: Optional[float] = None,
        rail_timeout_s: Optional[float] = None,
        active_rails_override: Optional['frozenset[str]'] = None,
        hard_filters: Optional[Dict[str, Any]] = None,
    ) -> List[ExploreRail]:
        """Fetch rail sources in parallel, dedupe across rails, return in canonical order.

        When active_rails_override is set, only rails in that set are built.
        ``hard_filters`` (compose_fallback only) push identified hard chips into
        CH rail SQL when ``clickhouse_rails.hard_filter_pushdown`` is enabled.
        """
        all_rail_ids = [r for r in RAIL_ORDER if r != 'fallback']
        rail_ids_to_build = [r for r in all_rail_ids if active_rails_override is None or r in active_rails_override]
        if active_rails_override is not None and len(rail_ids_to_build) < len(all_rail_ids):
            _skipped = [r for r in all_rail_ids if r not in active_rails_override]
            logger.debug(f"explore_rails_scoped active={rail_ids_to_build} skipped={_skipped}")
        if rail_timeout_s is not None and rail_timeout_s > 0:
            tasks = [
                asyncio.wait_for(
                    self._build_rail(
                        rail_id=rid,
                        user_id=user_id,
                        max_per_rail_override=max_per_rail_override,
                        seen=set(),
                        horizon_override=horizon_override,
                        hard_filters=hard_filters,
                    ),
                    timeout=rail_timeout_s,
                )
                for rid in rail_ids_to_build
            ]
        else:
            tasks = [
                self._build_rail(
                    rail_id=rid,
                    user_id=user_id,
                    max_per_rail_override=max_per_rail_override,
                    seen=set(),
                    horizon_override=horizon_override,
                    hard_filters=hard_filters,
                )
                for rid in rail_ids_to_build
            ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        seen_item_ids: set = set()
        rails: List[ExploreRail] = []
        for rail_id, result in zip(rail_ids_to_build, results):
            if isinstance(result, BaseException):
                logger.warning(f"explore_rail_fetch_error rail={rail_id} error_type={type(result).__name__} error={result}")
                continue
            if result is None:
                continue
            deduped_cards = [c for c in result.cards if c.item_id not in seen_item_ids]
            for c in deduped_cards:
                seen_item_ids.add(c.item_id)
            if deduped_cards:
                rails.append(ExploreRail(rail_id=result.rail_id, title=result.title, cards=deduped_cards, explanation=result.explanation, latency_ms=result.latency_ms))

        if not rails or all(len(r.cards) == 0 for r in rails):
            fallback = await self._build_fallback_rail(max_per_rail_override=max_per_rail_override, timeout_s=rail_timeout_s)
            if fallback is not None:
                rails = [fallback] if not rails else rails + [fallback]

        rails = [r for r in rails if r.cards or r.rail_id == 'fallback']
        if not rails:
            rails = [ExploreRail(rail_id='fallback', title=self._config.fallback.source.title, cards=[], explanation='No candidates available — please widen your search', latency_ms=0.0)]
        return rails

    async def _build_rail(
        self,
        rail_id: str,
        user_id: Optional[str],
        max_per_rail_override: Optional[int],
        seen: set,
        horizon_override: Optional[float] = None,
        hard_filters: Optional[Dict[str, Any]] = None,
    ) -> Optional[ExploreRail]:
        """Build a single rail by id; returns None when the rail is disabled or source unavailable."""
        if rail_id == 'trending':
            source = self._trending
            cfg_source = self._config.trending.source
            title = cfg_source.title
            explanation = 'Heating up across the marketplace'
        elif rail_id == 'ending_soon':
            source = self._ending_soon
            cfg_source = self._config.ending_soon.source
            title = cfg_source.title
            explanation = 'Closing soon'
        elif rail_id == 'last_hour':
            if self._last_hour is None:
                return None
            source = self._last_hour
            cfg_source = self._config.ending_soon.source
            title = 'Ending this hour'
            explanation = 'Expiring within the next 60 minutes'
        elif rail_id == 'latest':
            if self._latest is None:
                return None
            source = self._latest
            cfg_source = self._config.trending.source
            title = 'Recently listed'
            explanation = 'Newest domains on the marketplace'
        elif rail_id == 'high_volume':
            if self._high_volume is None:
                return None
            source = self._high_volume
            cfg_source = self._config.trending.source
            title = 'High activity'
            explanation = 'Hottest auctions by bid volume'
        elif rail_id == 'fresh':
            if self._fresh_listings is None:
                return None
            source = self._fresh_listings
            cfg_source = self._config.trending.source
            title = 'Fresh arrivals'
            explanation = 'Zero-bid domains just listed on the marketplace'
        elif rail_id == 'last_week':
            if self._last_week is None:
                return None
            source = self._last_week
            cfg_source = self._config.trending.source
            title = 'Long runway'
            explanation = 'Auctions with over 7 days remaining'
        elif rail_id == 'watch_density':
            if self._watch_density is None:
                return None
            source = self._watch_density
            cfg_source = self._config.trending.source
            title = 'Being watched'
            explanation = 'Domains with the most active watchers today'
        elif rail_id == 'high_traffic':
            if self._high_traffic is None:
                return None
            source = self._high_traffic
            cfg_source = self._config.trending.source
            title = 'High traffic'
            explanation = 'Most activity combining bids and watchers'
        else:
            return None

        if not cfg_source.enabled:
            return None

        max_items = int(max_per_rail_override) if max_per_rail_override is not None else int(cfg_source.max_items)
        if max_per_rail_override is not None:
            max_items = min(max_items, int(cfg_source.max_items))

        # Skip shared rail cache when hard filters are active — pushdown changes the result set.
        _hf_key = ''
        if hard_filters:
            _hf_key = ':' + ','.join(
                f"{k}={repr(hard_filters[k])}" for k in sorted(hard_filters.keys())
            )
        _ck = (
            f"{max_items}:{int((horizon_override or 0) // 1800)}{_hf_key}"
            if rail_id == 'ending_soon'
            else f"{max_items}{_hf_key}"
        )
        _cache = self._rail_caches.get(rail_id)
        if _cache is not None and not hard_filters:
            _hit = _cache.get(_ck)
            if _hit is not None:
                logger.debug(f"explore_rail_cache_hit rail={rail_id} n={len(_hit)}")
                return ExploreRail(rail_id=rail_id, title=title, cards=_hit, explanation=explanation, latency_ms=0.0)

        t0 = time.monotonic()
        _fetch_kwargs: Dict[str, Any] = {'user_id': user_id, 'max_items': max_items}
        if rail_id == 'ending_soon':
            _fetch_kwargs['horizon_override'] = horizon_override
        # Only sources that declare hard_filters (CH rails) receive pushdown kwargs.
        if hard_filters and 'hard_filters' in inspect.signature(source.fetch).parameters:
            _fetch_kwargs['hard_filters'] = hard_filters
        cards = await source.fetch(**_fetch_kwargs)
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        if _cache is not None and cards and not hard_filters:
            _cache.put(_ck, cards)
        deduped = [c for c in cards if c.item_id not in seen]
        return ExploreRail(rail_id=rail_id, title=title, cards=deduped, explanation=explanation, latency_ms=elapsed_ms)

    async def _build_fallback_rail(self, max_per_rail_override: Optional[int], timeout_s: Optional[float] = None) -> Optional[ExploreRail]:
        """Build the fallback rail from the trending corpus, bypassing the trending window."""
        if not self._config.fallback.source.enabled:
            return None
        max_items = int(max_per_rail_override) if max_per_rail_override is not None else int(self._config.fallback.source.max_items)
        max_items = min(max_items, int(self._config.fallback.source.max_items))
        t0 = time.monotonic()
        _coro = (
            self._trending.fetch_all(max_items=max_items)
            if hasattr(self._trending, 'fetch_all')
            else self._trending.fetch(user_id=None, max_items=max_items)
        )
        try:
            if timeout_s is not None and timeout_s > 0:
                raw = await asyncio.wait_for(_coro, timeout=timeout_s)
            else:
                raw = await _coro
        except asyncio.TimeoutError:
            logger.warning(f"explore_fallback_rail_fetch_timeout timeout_s={timeout_s}")
            return None
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        cards = [ExploreCard(item_id=c.item_id, fused_score=float(c.fused_score), source_rail='fallback', payload=dict(c.payload)) for c in raw]
        return ExploreRail(rail_id='fallback', title=self._config.fallback.source.title, cards=cards, explanation='Popular auctions while we look for more matches', latency_ms=elapsed_ms)


__all__ = ['ExploreComposer', 'RAIL_ORDER']
