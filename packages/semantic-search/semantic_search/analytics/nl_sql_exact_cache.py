"""Exact NL-to-SQL cache (Pattern 7 — analytics-side cache).

`AnalyticsRouter` runs ``lookup`` / ``upsert`` inside ``asyncio.to_thread`` so
sync Redis work does not block the event loop.

Distinct from the search-side result caches (`ExactCache` / `StructuredCache`,
which cache search `RankedResults`). This cache stores ``(question_template,
sql_template, verified_count)`` triples keyed by the exact case-insensitive
canonical form of the question. A hit lets the analytics path skip the LLM
SQL-generation stage entirely.

Exact-match only (case-insensitive): two questions hit the same entry iff their
canonical forms (lowercase + punctuation-stripped + whitespace-collapsed) are
identical. Numerically distinct questions like ``find .net domains under $100``
and ``find .net domains under $200`` canonicalize to different keys and never
share a cached SQL — the embedding cosine-similarity matching that conflated
them is gone.

Safety contract:

- A cached SQL is NEVER executed without going back through
  `AstSecurityValidator` (defence-in-depth). The cache only stores the SQL
  string and metadata; trust gates live in the pipeline layer.
- Cold templates (verified_count < min_verified_count) DO NOT short-circuit
  the LLM — they are returned with `verified=False` so the router can still
  consult them as a "warm start" for the LLM but won't bypass generation.
- TTL + LRU eviction prevent unbounded growth and stale-template drift.
- Cache lookups skip empty / whitespace-only questions.

The Redis-backed mirror (multi-replica, persistent) shares the same exact
canonical-question key so replicas read each other's verified templates. The
in-memory store ships with the package so tests + small deployments work
without external dependencies.
"""
import dataclasses
import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

from semantic_search.cache.lru_ttl import LRUTTLCache
from semantic_search.config.analytics_models import NLSqlExactCacheConfig
from semantic_search.core.exceptions import CacheError
from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def canonicalize_question(question: str) -> str:
    """Case-insensitive canonical key form: lowercase + strip punctuation + collapse whitespace.
    :param question: str - Raw NL question
    :return: str - Canonical key text ('' for non-string / empty input)
    """
    if not isinstance(question, str):
        return ""
    text = question.strip().lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


@dataclass
class CachedSqlTemplate:
    """One cached (question, sql) pair with verifier-based trust metadata.

    :param question_template: str - Canonicalized question (cache key)
    :param sql_template: str - SQL string the LLM produced for this question
    :param mv_used: str - MV name the SQL was rewritten into ('' if raw-table)
    :param verified_count: int - Number of times the post-execution verifier
        confirmed the SQL produced a sufficient result (incremented on each hit
        that passes the verifier downstream)
    :param last_used: float - Unix timestamp of the last successful retrieval
    :param schema_version: str - Schema version when the entry was created;
        invalidation hook for future schema migrations
    """
    question_template: str
    sql_template: str
    mv_used: str
    verified_count: int
    last_used: float
    schema_version: str = ''

    def __post_init__(self) -> None:
        if not isinstance(self.question_template, str) or not self.question_template:
            raise CacheError("CachedSqlTemplate.question_template must be non-empty")
        if not isinstance(self.sql_template, str) or not self.sql_template.strip():
            raise CacheError("CachedSqlTemplate.sql_template must be non-empty")
        if int(self.verified_count) < 0:
            raise CacheError("CachedSqlTemplate.verified_count must be >= 0")
        if float(self.last_used) <= 0.0:
            raise CacheError("CachedSqlTemplate.last_used must be > 0")


@dataclass
class NLSqlCacheLookup:
    """Result of an exact NL-to-SQL cache lookup.

    :param hit: bool - True iff the canonical question matched a stored entry
    :param verified: bool - True iff the matched entry's verified_count >= the
        min_verified_count gate (False entries do NOT bypass the LLM)
    :param entry: Optional[CachedSqlTemplate] - The matched template (None on miss)
    """
    hit: bool
    verified: bool
    entry: Optional[CachedSqlTemplate]


class NLSqlExactCache:
    """In-memory exact NL-to-SQL cache with verifier-gated short-circuit.

    :param config: NLSqlExactCacheConfig - Cache settings
    :param redis_client: Optional[Any] - Sync ``redis.Redis`` for replica-shared exact-question mirror
    """

    def __init__(self, config: NLSqlExactCacheConfig, redis_client: Any = None):
        if not isinstance(config, NLSqlExactCacheConfig):
            raise CacheError("NLSqlExactCache requires a typed NLSqlExactCacheConfig")
        self._config = config
        self._redis = redis_client if config.remote.enabled else None
        self._store: LRUTTLCache[CachedSqlTemplate] = LRUTTLCache(max_entries=config.max_entries, ttl_seconds=config.ttl_seconds)

    @property
    def hits(self) -> int:
        """Total successful exact lookups."""
        return self._store.hits

    @property
    def misses(self) -> int:
        """Total miss / expired-on-access lookups."""
        return self._store.misses

    @property
    def size(self) -> int:
        """Current entry count."""
        return len(self._store)

    @staticmethod
    def _key(question_template: str) -> str:
        """Stable cache key for a canonical question (used for LRU + Redis storage)."""
        return hashlib.sha256(question_template.encode('utf-8')).hexdigest()[:32]

    def lookup(self, question: str) -> NLSqlCacheLookup:
        """Look up the exact cached template for `question` (case-insensitive).

        Returns a typed `NLSqlCacheLookup` (never raises). Callers MUST check
        `verified` before bypassing the LLM — `hit=True, verified=False` means
        a template exists but hasn't earned trust yet.

        Sync API: call from ``asyncio.to_thread`` when invoked from async code.
        """
        miss = NLSqlCacheLookup(hit=False, verified=False, entry=None)
        if not self._config.enabled:
            return miss
        question_key = canonicalize_question(question)
        if not question_key:
            return miss
        key = self._key(question_key)
        if self._redis is not None:
            rkey = f"{self._config.remote.key_prefix}{key}"
            try:
                raw = self._redis.get(rkey)
            except Exception as e:
                logger.warning(f"nl_sql_redis_get_failed error_type={type(e).__name__}")
                raw = None
            if raw:
                try:
                    entry = CachedSqlTemplate(**json.loads(raw))
                except Exception as e:
                    logger.warning(f"nl_sql_cache_deserialize_failed error_type={type(e).__name__}")
                    entry = None
                if isinstance(entry, CachedSqlTemplate):
                    verified = int(entry.verified_count) >= int(self._config.min_verified_count)
                    logger.info(f"nl_sql_redis_hit verified={verified} verified_count={entry.verified_count}")
                    return NLSqlCacheLookup(hit=True, verified=verified, entry=entry)
        entry = self._store.get(key)
        if entry is None:
            return miss
        verified = int(entry.verified_count) >= int(self._config.min_verified_count)
        logger.info(f"nl_sql_cache_hit verified={verified} verified_count={entry.verified_count}")
        return NLSqlCacheLookup(hit=True, verified=verified, entry=entry)

    def upsert(self, question: str, sql_template: str, mv_used: str, verifier_passed: bool, schema_version: str = '') -> Optional[CachedSqlTemplate]:
        """Insert or update a cache entry for `question` → `sql_template`.

        :param question: str - Original NL question (canonicalized internally)
        :param sql_template: str - SQL the pipeline produced
        :param mv_used: str - MV name (empty when SQL ran on raw table)
        :param verifier_passed: bool - Whether the post-execution verifier said
            `sufficient=True`. Increments `verified_count` when True; never
            decrements (we treat verifier-pass as monotonic evidence).
        :param schema_version: str - Schema version stamp (audit trail)
        :return: Optional[CachedSqlTemplate] - The persisted entry (None when
            cache disabled or input invalid)

        Sync API: call from ``asyncio.to_thread`` when invoked from async code.
        """
        if not self._config.enabled:
            return None
        question_key = canonicalize_question(question)
        if not question_key or not isinstance(sql_template, str) or not sql_template.strip():
            return None
        key = self._key(question_key)
        existing = self._store.get(key)
        if existing is not None:
            new_count = existing.verified_count + (1 if verifier_passed else 0)
            entry = CachedSqlTemplate(
                question_template=question_key,
                sql_template=sql_template,
                mv_used=mv_used,
                verified_count=new_count,
                last_used=time.time(),
                schema_version=schema_version or existing.schema_version,
            )
        else:
            entry = CachedSqlTemplate(
                question_template=question_key,
                sql_template=sql_template,
                mv_used=mv_used,
                verified_count=1 if verifier_passed else 0,
                last_used=time.time(),
                schema_version=schema_version,
            )
        self._store.put(key, entry)
        if self._redis is not None:
            rkey = f"{self._config.remote.key_prefix}{key}"
            try:
                self._redis.setex(rkey, int(self._config.ttl_seconds), json.dumps(dataclasses.asdict(entry)))
            except Exception as e:
                logger.warning(f"nl_sql_redis_set_failed error_type={type(e).__name__}")
        logger.info(f"nl_sql_cache_upsert verifier_passed={verifier_passed} verified_count={entry.verified_count} mv_used={mv_used or 'raw'} size={self.size}")
        return entry

    def invalidate_all(self) -> int:
        """Drop every in-memory entry and flush Redis mirror if configured.
        :return: int - In-memory entries dropped (0 when already empty or disabled)
        """
        if not self._config.enabled:
            return 0
        dropped = len(self._store)
        self._store.clear()
        if self._redis is not None:
            try:
                pattern = f"{self._config.remote.key_prefix}*"
                keys = self._redis.keys(pattern)
                if keys:
                    self._redis.delete(*keys)
                    logger.info(f"nl_sql_redis_cache_flushed keys_deleted={len(keys)}")
            except Exception as e:
                logger.warning(f"nl_sql_redis_flush_failed error_type={type(e).__name__}")
        if dropped > 0:
            logger.info(f"nl_sql_cache_invalidated entries_dropped={dropped}")
        return dropped


__all__ = [
    'NLSqlExactCache',
    'NLSqlCacheLookup',
    'CachedSqlTemplate',
    'canonicalize_question',
]
