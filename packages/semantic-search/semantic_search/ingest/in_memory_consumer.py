"""In-memory listing + bid event consumer stubs.

Replayable, thread-safe, and Protocol-conforming so the live Kinesis consumers
can be swapped in without touching downstream code. Each consumed event:

  1. is appended to a bounded ring buffer (``max_events_in_memory``)
  2. bumps a monotonic ``snapshot_version`` (when ``invalidate_caches_on_event``)
  3. fires every registered ``SnapshotInvalidationHook`` with the new version

The snapshot bump is the cache-invalidation cascade trigger; the three cache
tiers register a hook that calls ``invalidate_all()`` so stale results from
the previous snapshot are dropped on the spot.
"""
import asyncio
import threading
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from semantic_search.config.models import IngestConfig
from semantic_search.contracts import BidEvent, ListingEvent, SnapshotInvalidationHook
from semantic_search.core.exceptions import ValidationError
from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)


class SnapshotVersionRegistry:
    """Process-wide snapshot version + invalidation hooks (monotonic, fire on bump)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._version: int = 0
        self._hooks: List[SnapshotInvalidationHook] = []
        self._priority_hooks: List[SnapshotInvalidationHook] = []

    @property
    def version(self) -> int:
        """Current snapshot version (monotonically non-decreasing)."""
        with self._lock:
            return self._version

    def bump(self, priority: bool = False) -> int:
        """Increment version + fire hooks (priority hooks fire after regular if priority=True)."""
        with self._lock:
            self._version += 1
            new_version = self._version
            hooks = list(self._hooks)
            priority_hooks = list(self._priority_hooks) if priority else []
        for hook in hooks:
            try:
                hook(new_version)
            except (ValueError, RuntimeError) as e:
                # An invalidation hook failure must NEVER kill ingestion; log + continue.
                logger.warning(f"snapshot_invalidation_hook_failed new_version={new_version} hook={getattr(hook, '__qualname__', repr(hook))} error_type={type(e).__name__} error={str(e)}")
        for hook in priority_hooks:
            try:
                hook(new_version)
            except (ValueError, RuntimeError) as e:
                logger.warning(f"snapshot_priority_hook_failed new_version={new_version} hook={getattr(hook, '__qualname__', repr(hook))} error_type={type(e).__name__} error={str(e)}")
        return new_version

    def register(self, hook: SnapshotInvalidationHook) -> None:
        """Register hook (fired on each bump)."""
        if hook is None or not callable(hook):
            raise ValidationError("SnapshotVersionRegistry.register requires a callable hook")
        with self._lock:
            self._hooks.append(hook)

    def register_priority_hook(self, hook: SnapshotInvalidationHook) -> None:
        """Register a priority hook fired only on ``bump(priority=True)`` calls.

        Used by the vector refresh driver to schedule an accelerated re-index
        after auction auto-extension events.
        """
        if hook is None or not callable(hook):
            raise ValidationError("SnapshotVersionRegistry.register_priority_hook requires a callable hook")
        with self._lock:
            self._priority_hooks.append(hook)


class _BaseInMemoryConsumer:
    """Shared replay + invalidation plumbing for the listing + bid stubs."""

    def __init__(self, config: IngestConfig, registry: SnapshotVersionRegistry):
        if config is None:
            raise ValidationError("InMemoryConsumer requires a non-null IngestConfig")
        if registry is None:
            raise ValidationError("InMemoryConsumer requires a non-null SnapshotVersionRegistry")
        self._config = config
        self._registry = registry
        self._lock = threading.Lock()
        self._events: Deque[Any] = deque(maxlen=int(config.max_events_in_memory))

    @property
    def snapshot_version(self) -> int:
        """Current snapshot version (delegates to the shared registry)."""
        return self._registry.version

    def register_invalidation_hook(self, hook: SnapshotInvalidationHook) -> None:
        """Add a cache-invalidation callback fired on every event when enabled."""
        self._registry.register(hook)

    def replay(self) -> List[Any]:
        """Return a defensive copy of the buffered event log (oldest first)."""
        with self._lock:
            return list(self._events)

    def clear(self) -> None:
        """Drop the in-memory event log. Does not roll back the snapshot version."""
        with self._lock:
            self._events.clear()

    def _record(self, event: Any, priority: bool = False) -> int:
        """Persist an event in the replay buffer + bump snapshot version when enabled."""
        if not self._config.enabled:
            return self._registry.version
        with self._lock:
            self._events.append(event)
        if self._config.invalidate_caches_on_event:
            return self._registry.bump(priority=priority)
        return self._registry.version


def _is_auction_extension(event: ListingEvent) -> bool:
    """Return True when the event represents an auction auto-extension.

    An auto-extension is an 'updated' listing event where the upstream system
    set isextendedauction=True in the payload (bid arrived near end-time and
    auctionendtime was pushed forward). This triggers a priority bump so the
    vector index refreshes faster than the normal cadence.
    """
    return event.event_kind == 'updated' and bool(event.payload.get('isextendedauction')) and 'auctionendtime' in event.payload


class InMemoryListingConsumer(_BaseInMemoryConsumer):
    """In-memory implementation of ``semantic_search.contracts.ListingEventConsumer``.

    :param config: IngestConfig - Ingest config block
    :param registry: SnapshotVersionRegistry - Shared cross-stream version source
    """

    async def consume(self, event: ListingEvent) -> int:
        """Consume one listing event and return the snapshot version after the event.

        The live consumer's contract is identical (``async def consume(event)
        -> new_version``), so tests and the orchestrator can drive this in a
        loop with no awareness that the stream is in-memory.

        Auto-extension events (event_kind='updated', isextendedauction=True in
        payload) trigger a priority bump so the vector refresh driver accelerates
        its next cycle.

        :param event: ListingEvent - Validated event
        :return: int - Snapshot version after the event (incremented when invalidation enabled)
        :raises ValidationError: When the event is the wrong type
        """
        if not isinstance(event, ListingEvent):
            raise ValidationError(f"InMemoryListingConsumer.consume expected ListingEvent, got {type(event).__name__}")
        # No real I/O; await yield keeps the contract strictly async so the
        # caller is wired through ``await`` exactly like the live consumer.
        await asyncio.sleep(0)
        is_extension = _is_auction_extension(event)
        new_version = self._record(event, priority=is_extension)
        if is_extension:
            logger.info(f"auction_auto_extended event_id={event.event_id} item_id={event.item_id} new_auctionendtime={event.payload.get('auctionendtime')} snapshot_version={new_version}")
        logger.info(f"ingest_listing_event event_id={event.event_id} item_id={event.item_id} kind={event.event_kind} snapshot_version={new_version}")
        return new_version

    async def consume_batch(self, events: List[ListingEvent]) -> int:
        """Consume a batch sequentially, returning the snapshot version after the last event."""
        if not isinstance(events, list):
            raise ValidationError("InMemoryListingConsumer.consume_batch requires a list")
        last_version = self._registry.version
        for event in events:
            last_version = await self.consume(event)
        return last_version

    def seed_from_dict(self, raw_events: List[Dict[str, Any]]) -> int:
        """Seed the consumer synchronously from a list of dicts (test helper).

        Each dict is validated via ``ListingEvent.from_dict``; malformed entries
        raise ``ValidationError`` so seed-data drift fails fast.

        :param raw_events: List[Dict] - Raw event dicts
        :return: int - Snapshot version after the last seeded event
        """
        if not isinstance(raw_events, list):
            raise ValidationError("InMemoryListingConsumer.seed_from_dict requires a list")
        last_version = self._registry.version
        for raw in raw_events:
            event = ListingEvent.from_dict(raw)
            last_version = self._record(event)
        return last_version


class InMemoryBidConsumer(_BaseInMemoryConsumer):
    """In-memory implementation of ``semantic_search.contracts.BidEventConsumer``."""

    async def consume(self, event: BidEvent) -> int:
        if not isinstance(event, BidEvent):
            raise ValidationError(f"InMemoryBidConsumer.consume expected BidEvent, got {type(event).__name__}")
        await asyncio.sleep(0)
        new_version = self._record(event)
        logger.info(f"ingest_bid_event event_id={event.event_id} item_id={event.item_id} kind={event.event_kind} bid_amount_usd={event.bid_amount_usd:.2f} snapshot_version={new_version}")
        return new_version

    async def consume_batch(self, events: List[BidEvent]) -> int:
        if not isinstance(events, list):
            raise ValidationError("InMemoryBidConsumer.consume_batch requires a list")
        last_version = self._registry.version
        for event in events:
            last_version = await self.consume(event)
        return last_version

    def seed_from_dict(self, raw_events: List[Dict[str, Any]]) -> int:
        if not isinstance(raw_events, list):
            raise ValidationError("InMemoryBidConsumer.seed_from_dict requires a list")
        last_version = self._registry.version
        for raw in raw_events:
            event = BidEvent.from_dict(raw)
            last_version = self._record(event)
        return last_version


__all__ = ['SnapshotVersionRegistry', 'InMemoryListingConsumer', 'InMemoryBidConsumer', '_is_auction_extension']
