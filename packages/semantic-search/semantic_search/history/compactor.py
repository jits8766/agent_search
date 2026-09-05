"""History compactor — aggregates 90-day raw user history into a per-user feature vector.

Privacy discipline: 90-day raw retention then aggregation to a per-user feature
vector (raw text discarded). Search history is hashed at session-id level for
the feedback-loop join. Right-to-delete clears both raw rows and the rail
materialized view within one batch cycle (≤ 24 hr). Consumer surface: a
landing-page rail composed offline daily from the user's 90-day search history
— top TLDs, median price-max, dominant semantic themes, frequent auction type.

This module implements that compaction step end-to-end:

- ``UserFeatureVector`` is the typed boundary contract. Every persisted vector
  carries a hashed ``user_id_hash`` (sha256[:12]) — never the raw user_id.
- ``UserFeatureVectorStore`` is the protocol so an S3 / DB-backed
  implementation can be swapped in without touching the compactor.
- ``InMemoryUserFeatureVectorStore`` is the in-process reference implementation
  used in tests + local environments.
- ``HistoryCompactor`` is the pure compute step: given a ``user_id``, read the
  raw retention window, compute features, upsert. ``forget_user`` is the
  right-to-delete entry point (clears both raw + aggregated state).
- ``HistoryCompactorDriver`` is the periodic asyncio loop matching the
  canonical ``*Driver`` shape used elsewhere in the package
  (``VectorRefreshDriver``).

Layer rules: imports stdlib + ``core`` + ``contracts`` + sibling
``history.store``. Never imports orchestration or retrieval code.
"""
import asyncio
import hashlib
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple

from semantic_search.config.models import HistoryCompactorConfig
from semantic_search.contracts import HistoryEntry
from semantic_search.core.exceptions import ValidationError
from semantic_search.core.logging_utils import get_logger
from semantic_search.history.store import UserSearchHistoryStore

logger = get_logger(__name__)


_TLD_RE = re.compile(r'\b[a-z0-9-]{1,63}\.([a-z]{2,32})\b')


def _hash_user_id(user_id: str) -> str:
    """SHA-256[:12] of user_id (for feedback-loop join; never raw)."""
    if not isinstance(user_id, str) or not user_id:
        raise ValidationError("_hash_user_id requires a non-empty string user_id")
    return hashlib.sha256(user_id.encode('utf-8')).hexdigest()[:12]


@dataclass(frozen=True)
class UserFeatureVector:
    """90-day aggregated user vector (frozen): hashed user_id + top TLDs/themes + distributions.
        Empty until ``HistoryEntry`` carries entity values; reserved for future
        use so consumers (landing rail) can read a stable contract today.
    :param recency_seconds: float - ``computed_at - newest_entry.created_at``
    :param revisit_count: int - Number of entry pairs whose ``top_item_ids``
        overlap (a proxy for repeat-search behavior)
    """
    user_id_hash: str
    computed_at: float
    window_seconds: float
    entry_count: int
    top_tlds: Tuple[Tuple[str, int], ...]
    query_type_distribution: Dict[str, int]
    top_themes: Tuple[Tuple[str, int], ...]
    auction_type_distribution: Dict[str, int]
    recency_seconds: float
    revisit_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.user_id_hash, str) or len(self.user_id_hash) != 12:
            raise ValidationError("UserFeatureVector.user_id_hash must be a 12-char hex string")
        if not isinstance(self.computed_at, (int, float)) or self.computed_at <= 0:
            raise ValidationError("UserFeatureVector.computed_at must be a positive number")
        if not isinstance(self.window_seconds, (int, float)) or self.window_seconds <= 0:
            raise ValidationError("UserFeatureVector.window_seconds must be > 0")
        if not isinstance(self.entry_count, int) or self.entry_count < 0:
            raise ValidationError("UserFeatureVector.entry_count must be a non-negative int")
        if not isinstance(self.top_tlds, tuple):
            raise ValidationError("UserFeatureVector.top_tlds must be a tuple")
        if not isinstance(self.query_type_distribution, dict):
            raise ValidationError("UserFeatureVector.query_type_distribution must be a dict")
        if not isinstance(self.top_themes, tuple):
            raise ValidationError("UserFeatureVector.top_themes must be a tuple")
        if not isinstance(self.auction_type_distribution, dict):
            raise ValidationError("UserFeatureVector.auction_type_distribution must be a dict")
        if not isinstance(self.recency_seconds, (int, float)) or self.recency_seconds < 0:
            raise ValidationError("UserFeatureVector.recency_seconds must be a non-negative number")
        if not isinstance(self.revisit_count, int) or self.revisit_count < 0:
            raise ValidationError("UserFeatureVector.revisit_count must be a non-negative int")


class UserFeatureVectorStore(Protocol):
    """Async store contract for aggregated user feature vectors.

    Async by default (per ``async-patterns.mdc``) so a future S3 / DynamoDB
    backend can plug in without changing the compactor.
    """

    async def upsert(self, user_id_hash: str, vector: UserFeatureVector) -> None:
        ...

    async def get(self, user_id_hash: str) -> Optional[UserFeatureVector]:
        ...

    async def delete(self, user_id_hash: str) -> bool:
        ...

    async def list_recent(self, limit: int) -> List[UserFeatureVector]:
        ...


class InMemoryUserFeatureVectorStore:
    """Thread-safe in-memory ``UserFeatureVectorStore``. Reference impl + tests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._vectors: Dict[str, UserFeatureVector] = {}

    async def upsert(self, user_id_hash: str, vector: UserFeatureVector) -> None:
        if not isinstance(user_id_hash, str) or not user_id_hash:
            raise ValidationError("InMemoryUserFeatureVectorStore.upsert requires a non-empty user_id_hash")
        if not isinstance(vector, UserFeatureVector):
            raise ValidationError("InMemoryUserFeatureVectorStore.upsert requires a UserFeatureVector")
        if vector.user_id_hash != user_id_hash:
            raise ValidationError("InMemoryUserFeatureVectorStore.upsert: user_id_hash mismatch with vector.user_id_hash")
        with self._lock:
            self._vectors[user_id_hash] = vector

    async def get(self, user_id_hash: str) -> Optional[UserFeatureVector]:
        if not isinstance(user_id_hash, str) or not user_id_hash:
            return None
        with self._lock:
            return self._vectors.get(user_id_hash)

    async def delete(self, user_id_hash: str) -> bool:
        if not isinstance(user_id_hash, str) or not user_id_hash:
            return False
        with self._lock:
            return self._vectors.pop(user_id_hash, None) is not None

    async def list_recent(self, limit: int) -> List[UserFeatureVector]:
        if not isinstance(limit, int) or limit < 0:
            raise ValidationError("InMemoryUserFeatureVectorStore.list_recent requires a non-negative int limit")
        with self._lock:
            vectors = list(self._vectors.values())
        vectors.sort(key=lambda v: v.computed_at, reverse=True)
        if limit > 0:
            vectors = vectors[:limit]
        return vectors

    def size(self) -> int:
        """Operator/observability helper. Not part of the protocol."""
        with self._lock:
            return len(self._vectors)


class HistoryCompactor:
    """Pure compute step: raw entries → ``UserFeatureVector``. No scheduling.

    :param history_store: UserSearchHistoryStore - Source of raw entries
    :param vector_store: UserFeatureVectorStore - Sink for aggregated vectors
    :param config: HistoryCompactorConfig - Aggregation thresholds
    :raises ValidationError: When dependencies are missing/wrong-typed
    """

    def __init__(self, history_store: UserSearchHistoryStore, vector_store: UserFeatureVectorStore, config: HistoryCompactorConfig):
        if not isinstance(history_store, UserSearchHistoryStore):
            raise ValidationError("HistoryCompactor requires a UserSearchHistoryStore")
        if vector_store is None:
            raise ValidationError("HistoryCompactor requires a UserFeatureVectorStore")
        for method_name in ('upsert', 'get', 'delete', 'list_recent'):
            attr = getattr(vector_store, method_name, None)
            if not callable(attr):
                raise ValidationError(f"HistoryCompactor: vector_store missing required method '{method_name}'")
        if not isinstance(config, HistoryCompactorConfig):
            raise ValidationError("HistoryCompactor requires a HistoryCompactorConfig")
        self._history = history_store
        self._vectors = vector_store
        self._config = config
        self._window_seconds: float = float(history_store.retention_seconds)

    async def compact_user(self, user_id: str) -> Optional[UserFeatureVector]:
        """Compute + persist the aggregated vector for ``user_id``.

        Returns ``None`` when the user has no entries OR is opted out OR the
        history layer is disabled. Otherwise returns the persisted vector.

        :param user_id: str - Authenticated user id (non-empty)
        :return: Optional[UserFeatureVector]
        :raises ValidationError: When user_id is empty
        """
        if not isinstance(user_id, str) or not user_id:
            raise ValidationError("HistoryCompactor.compact_user requires a non-empty user_id")
        entries = self._history.list_entries(user_id)
        if not entries:
            return None
        vector = self._compute_vector(user_id, entries)
        await self._vectors.upsert(vector.user_id_hash, vector)
        logger.info(
            f"history_compaction_user user_id_hash={vector.user_id_hash} "
            f"entries={vector.entry_count} top_tlds={len(vector.top_tlds)} "
            f"top_themes={len(vector.top_themes)} recency_s={int(vector.recency_seconds)}"
        )
        return vector

    async def compact_all_users(self) -> List[UserFeatureVector]:
        """Compact every currently-tracked user up to ``max_users_per_cycle``.

        Iteration order is the snapshot order returned by
        ``UserSearchHistoryStore.iter_users()``. The cap exists so a single
        cycle on a very large user base cannot starve the event loop.

        :return: List[UserFeatureVector] - Vectors produced this cycle (skips opted-out / empty)
        """
        produced: List[UserFeatureVector] = []
        users = self._history.iter_users()
        cap = int(self._config.max_users_per_cycle)
        if len(users) > cap:
            logger.warning(f"history_compaction_user_cap_applied total_users={len(users)} cap={cap} dropped={len(users) - cap}")
            users = users[:cap]
        for user_id in users:
            vector = await self.compact_user(user_id)
            if vector is not None:
                produced.append(vector)
        logger.info(f"history_compaction_cycle users_seen={len(users)} vectors_produced={len(produced)}")
        return produced

    async def forget_user(self, user_id: str) -> bool:
        """Right-to-delete: clears both raw rows AND the aggregated vector.

        Right-to-delete clears both raw rows and the rail materialized view
        within one batch cycle (≤ 24 hr). Returns True when at least one of
        the two stores had data to delete.

        :param user_id: str - Authenticated user id (non-empty)
        :return: bool - True when raw rows OR aggregated vector were removed
        :raises ValidationError: When user_id is empty
        """
        if not isinstance(user_id, str) or not user_id:
            raise ValidationError("HistoryCompactor.forget_user requires a non-empty user_id")
        raw_removed = self._history.delete_user(user_id)
        vec_removed = await self._vectors.delete(_hash_user_id(user_id))
        any_removed = raw_removed > 0 or vec_removed
        if any_removed:
            logger.info(f"history_compaction_forget user_id_hash={_hash_user_id(user_id)} raw_removed={raw_removed} vector_removed={vec_removed}")
        return any_removed

    def _compute_vector(self, user_id: str, entries: List[HistoryEntry]) -> UserFeatureVector:
        """Pure aggregation. No IO. Deterministic given the same inputs."""
        now = time.time()
        tld_counts: Dict[str, int] = {}
        theme_counts: Dict[str, int] = {}
        qt_counts: Dict[str, int] = {}
        seen_items: List[set] = []
        revisit_pairs = 0
        newest_at = 0.0
        for entry in entries:
            if entry.created_at > newest_at:
                newest_at = entry.created_at
            qt_counts[entry.query_type] = qt_counts.get(entry.query_type, 0) + 1
            for tld in self._extract_tlds(entry.normalized_query):
                tld_counts[tld] = tld_counts.get(tld, 0) + 1
            stem = entry.normalized_query.strip().lower()
            if stem:
                theme_counts[stem] = theme_counts.get(stem, 0) + 1
            current_items = set(entry.top_item_ids or [])
            if current_items:
                for prior in seen_items:
                    if current_items & prior:
                        revisit_pairs += 1
                seen_items.append(current_items)
        top_tlds = self._top_n(tld_counts, self._config.top_tld_count)
        top_themes = self._top_n({k: v for k, v in theme_counts.items() if v >= self._config.min_theme_occurrences}, self._config.top_theme_count)
        recency = max(0.0, now - newest_at) if newest_at > 0 else 0.0
        return UserFeatureVector(
            user_id_hash=_hash_user_id(user_id),
            computed_at=now,
            window_seconds=self._window_seconds,
            entry_count=len(entries),
            top_tlds=top_tlds,
            query_type_distribution=qt_counts,
            top_themes=top_themes,
            auction_type_distribution={},
            recency_seconds=recency,
            revisit_count=revisit_pairs,
        )

    def tracked_user_count(self) -> int:
        """Number of users currently held in the raw history store.

        Public observability helper used by the driver summary so it does not
        need to reach into the compactor's private attributes.
        """
        return len(self._history.iter_users())

    @staticmethod
    def _extract_tlds(text: str) -> List[str]:
        """Extract dotted-name TLD segments from ``text`` (lowercased)."""
        if not isinstance(text, str) or not text:
            return []
        return [m.group(1) for m in _TLD_RE.finditer(text.lower())]

    @staticmethod
    def _top_n(counts: Dict[str, int], n: int) -> Tuple[Tuple[str, int], ...]:
        """Return top-``n`` (key, count) pairs sorted by count desc, key asc.

        Key-asc tiebreak guarantees deterministic ordering across runs even
        when two keys share the same count.
        """
        if n <= 0 or not counts:
            return tuple()
        items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return tuple(items[:n])


@dataclass
class CompactionCycleSummary:
    """Single compaction cycle outcome (kept in driver history)."""
    started_at: float
    duration_seconds: float
    users_seen: int
    vectors_produced: int
    success: bool
    error: Optional[str] = None


@dataclass
class CompactorDriverSummary:
    """Aggregate snapshot of driver state for ops endpoints."""
    enabled: bool
    running: bool
    paused_after_failures: bool
    interval_seconds: float
    consecutive_failures: int
    cycles_total: int
    cycles_success: int
    cycles_failed: int
    history: List[CompactionCycleSummary] = field(default_factory=list)


class HistoryCompactorDriver:
    """Periodic asyncio loop wrapping ``HistoryCompactor.compact_all_users``.

    Mirrors the canonical ``*Driver`` shape used elsewhere in the package
    (``VectorRefreshDriver``):

    - **Idempotent start/stop.** Safe to call multiple times.
    - **Drop-overlapping policy.** If a previous cycle is still in flight when
      the next tick comes around, the driver logs and skips that tick rather
      than queueing a parallel run.
    - **Bounded back-off.** ``max_consecutive_failures`` consecutive failed
      cycles pause the loop until ``stop()`` + ``start()`` is called.
    - **Bounded history.** The most recent ``_MAX_HISTORY`` cycles are kept
      for ops endpoints; older entries are discarded.

    :param config: HistoryCompactorConfig - Validated cadence + back-off
    :param compactor: HistoryCompactor - The compactor the driver invokes
    :raises ValidationError: When dependencies are missing/wrong-typed
    """

    _MAX_HISTORY = 100

    def __init__(self, config: HistoryCompactorConfig, compactor: HistoryCompactor):
        if not isinstance(config, HistoryCompactorConfig):
            raise ValidationError("HistoryCompactorDriver requires a HistoryCompactorConfig")
        if not isinstance(compactor, HistoryCompactor):
            raise ValidationError("HistoryCompactorDriver requires a HistoryCompactor")
        self._config = config
        self._compactor = compactor

        self._task: Optional[asyncio.Task[None]] = None
        self._consecutive_failures: int = 0
        self._paused_after_failures: bool = False
        self._cycles_total: int = 0
        self._cycles_success: int = 0
        self._cycles_failed: int = 0
        self._history: List[CompactionCycleSummary] = []
        self._cycle_lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self._config.enabled)

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        """Spawn the background polling task. Idempotent + safe when disabled."""
        if not self.enabled:
            logger.info("history_compactor_driver_disabled")
            return
        if self.running:
            return
        self._paused_after_failures = False
        self._consecutive_failures = 0
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(f"history_compactor_driver_started interval_seconds={self._config.interval_seconds} max_consecutive_failures={self._config.max_consecutive_failures}")

    async def stop(self) -> None:
        """Cancel + await the background task. Idempotent."""
        if self._task is None:
            return
        task = self._task
        self._task = None
        if not task.done():
            task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        logger.info("history_compactor_driver_stopped")

    async def run_once(self) -> CompactionCycleSummary:
        """Run a single compaction cycle synchronously. Used by tests + ops endpoints."""
        return await self._tick()

    async def _poll_loop(self) -> None:
        try:
            while True:
                if self._paused_after_failures:
                    return
                await self._tick()
                await asyncio.sleep(float(self._config.interval_seconds))
        except asyncio.CancelledError:
            raise

    async def _tick(self) -> CompactionCycleSummary:
        """One poll iteration — runs at most one compaction cycle."""
        if self._cycle_lock.locked():
            logger.warning("history_compactor_tick_skipped reason=overlapping_cycle")
            return CompactionCycleSummary(started_at=time.time(), duration_seconds=0.0, users_seen=0, vectors_produced=0, success=False, error="overlapping_cycle")
        async with self._cycle_lock:
            started = time.time()
            try:
                vectors = await self._compactor.compact_all_users()
                duration = time.time() - started
                summary = CompactionCycleSummary(started_at=started, duration_seconds=duration, users_seen=self._compactor.tracked_user_count(), vectors_produced=len(vectors), success=True)
                self._record_success(summary)
                return summary
            except (asyncio.CancelledError, Exception) as e:
                duration = time.time() - started
                err_tail = '' if isinstance(e, asyncio.CancelledError) else str(e)[:256]
                err_field = f"{type(e).__name__}: {err_tail}" if err_tail else f"{type(e).__name__}"
                summary = CompactionCycleSummary(started_at=started, duration_seconds=duration, users_seen=0, vectors_produced=0, success=False, error=err_field)
                self._record_failure(summary)
                if isinstance(e, asyncio.CancelledError):
                    raise
                return summary

    def _record_success(self, summary: CompactionCycleSummary) -> None:
        self._cycles_total += 1
        self._cycles_success += 1
        self._consecutive_failures = 0
        self._append_history(summary)

    def _record_failure(self, summary: CompactionCycleSummary) -> None:
        self._cycles_total += 1
        self._cycles_failed += 1
        self._consecutive_failures += 1
        self._append_history(summary)
        if self._consecutive_failures >= int(self._config.max_consecutive_failures):
            self._paused_after_failures = True
            logger.error(f"history_compactor_paused_after_failures consecutive={self._consecutive_failures} max_allowed={self._config.max_consecutive_failures}")

    def _append_history(self, summary: CompactionCycleSummary) -> None:
        self._history.append(summary)
        if len(self._history) > self._MAX_HISTORY:
            self._history = self._history[-self._MAX_HISTORY:]

    def get_summary(self) -> CompactorDriverSummary:
        """Snapshot driver state for ops endpoints."""
        return CompactorDriverSummary(
            enabled=self.enabled,
            running=self.running,
            paused_after_failures=self._paused_after_failures,
            interval_seconds=float(self._config.interval_seconds),
            consecutive_failures=self._consecutive_failures,
            cycles_total=self._cycles_total,
            cycles_success=self._cycles_success,
            cycles_failed=self._cycles_failed,
            history=list(self._history),
        )
