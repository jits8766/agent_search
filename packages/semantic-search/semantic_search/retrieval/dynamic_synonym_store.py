"""Persistent bidirectional synonym store for the BM25 query-driven expansion layer.

Sits between the static ``SynonymExpander`` and the BM25 aggregation stage.
Learned entries are stored permanently — the map grows as the system handles
more queries and is never evicted. On each restart the full map is reloaded
from SQLite so coverage is cumulative across deployments.

Bidirectionality:
Every ``put(token, synonyms)`` call mirrors every forward edge into an inverse
edge in the in-memory map: ``put("cloud", ["hosting", "aws"])`` produces
``cloud → {hosting, aws}``, ``hosting → {cloud}``, and ``aws → {cloud}``.
Inverse edges are NOT written to SQLite (only the canonical forward edge is
stored); they are reconstructed from scratch on every startup load so the
SQLite schema stays minimal and the in-memory map is always the authoritative
source for reads.

Thread safety:
A ``threading.Lock`` guards all mutations to ``_bidir`` and ``_miss_freq``.
Reads (``get``) acquire the lock too so a background expander writing new
entries does not race a query-path read.

SQLite WAL mode is enabled so the background expander thread can write without
blocking the read path.

No external dependencies — stdlib only (``sqlite3``, ``threading``,
``collections``, ``json``, ``pathlib``).
"""
import collections
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

from semantic_search.config.models import DynamicSynonymConfig
from semantic_search.core.exceptions import RetrievalError, ValidationError
from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)

_CREATE_TABLE_SQL = "CREATE TABLE IF NOT EXISTS token_synonyms (token TEXT PRIMARY KEY, synonyms TEXT NOT NULL, last_updated REAL NOT NULL DEFAULT 0)"
_UPSERT_SQL = "INSERT OR REPLACE INTO token_synonyms (token, synonyms, last_updated) VALUES (?, ?, ?)"
_SELECT_ALL_SQL = "SELECT token, synonyms FROM token_synonyms ORDER BY last_updated DESC"
_EVICT_SQL = "DELETE FROM token_synonyms WHERE token NOT IN (SELECT token FROM token_synonyms ORDER BY last_updated DESC LIMIT ?)"
_COUNT_SQL = "SELECT COUNT(*) FROM token_synonyms"


def _build_bidir(forward: Dict[str, List[str]]) -> Dict[str, Set[str]]:
    """Build an in-memory bidirectional map from a forward synonym map.

    Self-edges are dropped. Empty value lists are skipped.

    :param forward: Dict[str, List[str]] - Forward synonym map
    :return: Dict[str, Set[str]] - Bidirectional map
    :raises ValidationError: When inputs are malformed
    """
    if not isinstance(forward, dict):
        raise ValidationError("_build_bidir requires a dict")
    bidir: Dict[str, Set[str]] = {}
    for raw_key, raw_vals in forward.items():
        if not isinstance(raw_key, str) or not raw_key:
            raise ValidationError(f"synonym map key must be non-empty string, got {raw_key!r}")
        if not isinstance(raw_vals, list):
            raise ValidationError(f"synonym map value for '{raw_key}' must be a list")
        key = raw_key.lower().strip()
        if not key:
            continue
        for raw_v in raw_vals:
            if not isinstance(raw_v, str) or not raw_v:
                continue
            v = raw_v.lower().strip()
            if not v or v == key:
                continue
            bidir.setdefault(key, set()).add(v)
            bidir.setdefault(v, set()).add(key)
    return bidir


class DynamicSynonymStore:
    """Persistent bidirectional synonym store for query-driven BM25 expansion.

    :param config: DynamicSynonymConfig - Validated config bundle
    :raises ValidationError: When ``config`` is None or wrong-typed
    :raises RetrievalError: When the SQLite database cannot be opened or initialised
    """

    def __init__(self, config: DynamicSynonymConfig):
        if config is None or not isinstance(config, DynamicSynonymConfig):
            raise ValidationError("DynamicSynonymStore requires a DynamicSynonymConfig")
        self._config = config
        self._lock = threading.Lock()
        self._bidir: Dict[str, Set[str]] = {}
        self._miss_freq: collections.Counter = collections.Counter()
        self._miss_queue: collections.deque = collections.deque(maxlen=int(config.miss_queue_max))
        self._queued_tokens: Set[str] = set()
        self._conn: Optional[sqlite3.Connection] = None
        if config.enabled:
            self._open_db(str(config.db_path))
            self._load_from_db()
            logger.info(f"dynamic_synonym_store_ready db={config.db_path} entries={len(self._bidir)}")

    def _open_db(self, db_path: str) -> None:
        """Open (or create) the SQLite database and activate WAL mode.

        :param db_path: str - Resolved file path
        :raises RetrievalError: On any sqlite3 error
        """
        try:
            path = Path(db_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(_CREATE_TABLE_SQL)
            self._conn.commit()
        except sqlite3.Error as exc:
            raise RetrievalError(f"dynamic_synonym_store_db_open_failed path={db_path} error={exc}") from exc

    def _load_from_db(self) -> None:
        """Read all rows from SQLite and build the in-memory bidirectional map.

        :raises RetrievalError: On any sqlite3 error
        """
        if self._conn is None:
            return
        try:
            cursor = self._conn.execute(_SELECT_ALL_SQL)
            forward: Dict[str, List[str]] = {}
            for token, syns_json in cursor.fetchall():
                try:
                    syns = json.loads(syns_json)
                    if isinstance(syns, list):
                        forward[token] = [str(s) for s in syns if s]
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"dynamic_synonym_store_corrupt_row token={token}")
            self._bidir = _build_bidir(forward)
        except sqlite3.Error as exc:
            raise RetrievalError(f"dynamic_synonym_store_load_failed error={exc}") from exc

    @property
    def enabled(self) -> bool:
        """True when the store is active."""
        return bool(self._config.enabled)

    @property
    def map_size(self) -> int:
        """Number of token keys in the bidirectional map (diagnostics)."""
        with self._lock:
            return len(self._bidir)

    @property
    def queue_size(self) -> int:
        """Tokens currently queued for LLM expansion."""
        with self._lock:
            return len(self._miss_queue)

    def get(self, token: str) -> List[str]:
        """Return synonyms for ``token`` from the bidirectional map.

        Returns an empty list when the token has no known synonyms.
        Never raises — any internal error degrades to an empty list.

        :param token: str - Lowercased query token
        :return: List[str] - Synonym list (may be empty)
        """
        if not self._config.enabled or not isinstance(token, str) or not token:
            return []
        key = token.lower().strip()
        if not key:
            return []
        with self._lock:
            syns = self._bidir.get(key)
            return list(syns) if syns else []

    def record_miss(self, token: str) -> None:
        """Record a miss for ``token`` and enqueue it for LLM expansion if threshold met.

        No-op when the store is disabled, token is already in the map, or the
        miss queue is at capacity.

        :param token: str - Lowercased query token that had no synonyms
        """
        if not self._config.enabled or not isinstance(token, str) or not token:
            return
        key = token.lower().strip()
        if not key:
            return
        min_freq = int(self._config.min_freq_to_expand)
        with self._lock:
            if key in self._bidir:
                return
            if key in self._queued_tokens:
                return
            self._miss_freq[key] += 1
            if self._miss_freq[key] >= min_freq:
                self._miss_queue.append(key)
                self._queued_tokens.add(key)
                logger.debug(f"dynamic_synonym_queued token={key} freq={self._miss_freq[key]}")

    def put(self, token: str, synonyms: List[str]) -> None:
        """Write ``token → synonyms`` as permanent bidirectional entries.

        Both the in-memory map and the SQLite database are updated atomically
        from the caller's perspective (lock held for the memory write; SQLite
        write follows under the same lock for consistency).

        Inverse edges are added automatically. Self-edges are dropped.

        :param token: str - Canonical token
        :param synonyms: List[str] - Synonyms from LLM expansion
        :raises ValidationError: When ``token`` or ``synonyms`` are malformed
        """
        if not self._config.enabled:
            return
        if not isinstance(token, str) or not token:
            raise ValidationError("DynamicSynonymStore.put: token must be a non-empty string")
        if not isinstance(synonyms, list):
            raise ValidationError("DynamicSynonymStore.put: synonyms must be a list")
        key = token.lower().strip()
        if not key:
            raise ValidationError("DynamicSynonymStore.put: token must be non-empty after normalisation")
        clean_syns = [s.lower().strip() for s in synonyms if isinstance(s, str) and s.strip() and s.strip() != key]
        if not clean_syns:
            return
        with self._lock:
            self._bidir.setdefault(key, set()).update(clean_syns)
            for syn in clean_syns:
                self._bidir.setdefault(syn, set()).add(key)
            self._queued_tokens.discard(key)
            if self._conn is not None:
                try:
                    existing = list(self._bidir.get(key, set()) - {key})
                    self._conn.execute(_UPSERT_SQL, (key, json.dumps(existing), time.time()))
                    row = self._conn.execute(_COUNT_SQL).fetchone()
                    if row and row[0] > int(self._config.max_db_entries):
                        self._conn.execute(_EVICT_SQL, (int(self._config.max_db_entries),))
                        logger.info(f"dynamic_synonym_store_evicted cap={self._config.max_db_entries}")
                    self._conn.commit()
                except sqlite3.Error as exc:
                    logger.error(f"dynamic_synonym_store_write_failed token={key} error={exc}")
        logger.info(f"dynamic_synonym_stored token={key} synonyms={len(clean_syns)} total_keys={len(self._bidir)}")

    def drain_miss_queue(self, max_count: int) -> List[str]:
        """Pop up to ``max_count`` tokens from the miss queue for LLM expansion.

        Tokens remain in ``_queued_tokens`` until ``put`` is called so they
        are not re-enqueued while the LLM call is in-flight.

        :param max_count: int - Maximum tokens to return
        :return: List[str] - Tokens needing LLM expansion
        :raises ValidationError: When ``max_count < 1``
        """
        if not isinstance(max_count, int) or max_count < 1:
            raise ValidationError("drain_miss_queue: max_count must be int >= 1")
        batch: List[str] = []
        with self._lock:
            while self._miss_queue and len(batch) < max_count:
                batch.append(self._miss_queue.popleft())
        return batch

    def close(self) -> None:
        """Close the SQLite connection. Safe to call multiple times."""
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except sqlite3.Error:
                    pass
                self._conn = None
