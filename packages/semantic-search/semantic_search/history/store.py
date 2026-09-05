"""User search history store — thread-safe in-memory ring buffer per user.

Honors:
  * `history.retention_days`        — entries older than the window are pruned at read time.
  * `history.max_entries_per_user`  — hard cap; oldest entries pruned on insert.
  * `history.max_query_length`      — defense-in-depth bound on stored query text.
  * Per-user opt-out                — when set, all writes are dropped and reads return [].
"""
import hashlib
import threading
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Set

from semantic_search.config.models import HistoryConfig
from semantic_search.contracts import HistoryEntry
from semantic_search.core.exceptions import HistoryError
from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)


class UserSearchHistoryStore:
    """Ring-buffer history (retention window + max_entries_per_user + opt-out)."""

    def __init__(self, config: HistoryConfig):
        self._config = config
        self._lock = threading.Lock()
        self._entries: Dict[str, Deque[HistoryEntry]] = {}
        self._opted_out: Set[str] = set()
        self._retention_seconds: float = float(config.retention_days) * 86400.0

    def is_enabled(self) -> bool:
        """Whether history is enabled (mirrors `history.enabled`)."""
        return bool(self._config.enabled)

    @property
    def retention_seconds(self) -> float:
        """Configured retention window in seconds (public for compactor)."""
        return self._retention_seconds

    def is_opted_out(self, user_id: str) -> bool:
        """Whether the given user has opted out of history capture."""
        if not user_id:
            return False
        with self._lock:
            return user_id in self._opted_out

    def opt_out(self, user_id: str) -> None:
        """Opt out user + purge their existing entries."""
        if not user_id:
            raise HistoryError("UserSearchHistoryStore.opt_out requires a non-empty user_id")
        with self._lock:
            self._opted_out.add(user_id)
            self._entries.pop(user_id, None)
        logger.info(f"history_user_opted_out user_id_hash={_hash_id(user_id)}")

    def opt_in(self, user_id: str) -> None:
        """Reverse opt_out (no historical replay)."""
        if not user_id:
            raise HistoryError("UserSearchHistoryStore.opt_in requires a non-empty user_id")
        with self._lock:
            self._opted_out.discard(user_id)
        logger.info(f"history_user_opted_in user_id_hash={_hash_id(user_id)}")

    def record(self, user_id: str, normalized_query: str, query_type: str, top_item_ids: List[str], intent_record_id: Optional[str] = None) -> Optional[HistoryEntry]:
        """Record a new history entry for `user_id`.

        Returns the persisted `HistoryEntry`, or None when history is disabled or
        the user has opted out.

        :param user_id: str - Authenticated user id
        :param normalized_query: str - Normalized query stem (used for repeat detection)
        :param query_type: str - QI-derived query type
        :param top_item_ids: List[str] - First N item ids returned (for delta detection)
        :param intent_record_id: Optional[str] - Propagates the QI-emitted
            ``intent_record_id`` so resume / repeat-search analytics can stitch a
            user's history rows to the same intent across refines.
        :return: Optional[HistoryEntry]
        :raises HistoryError: When inputs are invalid
        """
        if not self._config.enabled:
            return None
        if not user_id:
            raise HistoryError("UserSearchHistoryStore.record requires a non-empty user_id")
        if not isinstance(normalized_query, str) or not normalized_query:
            raise HistoryError("UserSearchHistoryStore.record requires a non-empty normalized_query")
        if len(normalized_query) > self._config.max_query_length:
            normalized_query = normalized_query[: self._config.max_query_length]
        if not isinstance(top_item_ids, list):
            raise HistoryError("UserSearchHistoryStore.record requires top_item_ids to be a list")

        with self._lock:
            if user_id in self._opted_out:
                return None
            buf = self._entries.get(user_id)
            if buf is None:
                buf = deque(maxlen=self._config.max_entries_per_user)
                self._entries[user_id] = buf
            entry = HistoryEntry(
                entry_id=HistoryEntry.new_entry_id(),
                user_id=user_id,
                normalized_query=normalized_query,
                query_type=query_type,
                top_item_ids=list(top_item_ids),
                intent_record_id=str(intent_record_id) if intent_record_id else '',
            )
            buf.append(entry)
        return entry

    def list_entries(self, user_id: str, limit: Optional[int] = None) -> List[HistoryEntry]:
        """Return the most recent retention-windowed entries for `user_id`.

        :param user_id: str - Authenticated user id
        :param limit: Optional[int] - Maximum entries returned; None = no extra cap
        :return: List[HistoryEntry] - Newest-first
        """
        if not self._config.enabled or not user_id:
            return []
        cutoff = time.time() - self._retention_seconds
        with self._lock:
            if user_id in self._opted_out:
                return []
            buf = self._entries.get(user_id)
            if not buf:
                return []
            kept = [e for e in buf if e.created_at >= cutoff]
            if len(kept) != len(buf):
                # Prune in-place so the deque stays bounded under repeated reads.
                self._entries[user_id] = deque(kept, maxlen=self._config.max_entries_per_user)
        kept.sort(key=lambda e: e.created_at, reverse=True)
        if limit is not None and limit >= 0:
            kept = kept[: int(limit)]
        return kept

    def delete_user(self, user_id: str) -> int:
        """Delete every history entry for `user_id`. Returns the number of entries removed."""
        if not user_id:
            raise HistoryError("UserSearchHistoryStore.delete_user requires a non-empty user_id")
        with self._lock:
            buf = self._entries.pop(user_id, None)
        removed = 0 if buf is None else len(buf)
        if removed > 0:
            logger.info(f"history_user_deleted user_id_hash={_hash_id(user_id)} removed={removed}")
        return removed

    def iter_users(self) -> List[str]:
        """Snapshot of currently-tracked user_ids (point-in-time copy under the lock).

        Used by the compactor to drive `compact_all_users()`. Returns an empty
        list when history is disabled. The snapshot is a copy — callers may
        iterate without holding the lock.

        :return: List[str] - User ids currently holding entries (any opt-out
            users have already been removed by `opt_out`)
        """
        if not self._config.enabled:
            return []
        with self._lock:
            return list(self._entries.keys())

    def repeat_query_count(self, user_id: str, normalized_query: str, window_seconds: float) -> int:
        """Count how many times `normalized_query` appears in the trailing window.

        :param user_id: str
        :param normalized_query: str
        :param window_seconds: float - Lookback window
        :return: int - Occurrence count (0 when opted out / disabled)
        """
        if not self._config.enabled or not user_id or not normalized_query or window_seconds <= 0:
            return 0
        cutoff = time.time() - float(window_seconds)
        with self._lock:
            if user_id in self._opted_out:
                return 0
            buf = self._entries.get(user_id)
            if not buf:
                return 0
            return sum(1 for e in buf if e.created_at >= cutoff and e.normalized_query == normalized_query)


def _hash_id(user_id: str) -> str:
    """Stable, low-cardinality identifier hash used in logs (no PII)."""
    return hashlib.sha256(user_id.encode('utf-8')).hexdigest()[:12]
