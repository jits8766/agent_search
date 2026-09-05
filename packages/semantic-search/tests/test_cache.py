"""Tests for the 3-tier cache layer + the underlying LRUTTL primitive.

Coverage matrix (per ``testing.mdc`` §7):

``LRUTTLCache``:
- basic_put_get                                 -> TestLRUTTLCache::test_basic_put_get
- invalid_construction                          -> TestLRUTTLCache::test_invalid_construction
- lru_eviction_at_capacity                      -> TestLRUTTLCache::test_lru_eviction
- ttl_expiry                                    -> TestLRUTTLCache::test_ttl_expiry
- put_empty_key_rejected                        -> TestLRUTTLCache::test_put_empty_key_rejected

``ExactCache``:
- disabled_returns_none                         -> TestExactCache::test_disabled_returns_none
- miss_on_unseen_query                          -> TestExactCache::test_miss_on_unseen_query
- round_trip                                    -> TestExactCache::test_round_trip
- payload_round_trip_carries_intent             -> TestExactCache::test_payload_round_trip_carries_intent
- bare_put_get_payload_returns_none             -> TestExactCache::test_bare_put_get_payload_returns_none
- payload_get_returns_results_for_legacy_caller -> TestExactCache::test_payload_get_returns_results_for_legacy_caller
- lookup_is_case_insensitive                    -> TestExactCache::test_lookup_is_case_insensitive
- numerically_distinct_queries_do_not_collide   -> TestExactCache::test_numerically_distinct_queries_do_not_collide

``StructuredCache``:
- empty_filters_returns_none                    -> TestStructuredCache::test_empty_filters_returns_none
- filter_order_independent                      -> TestStructuredCache::test_filter_order_independent
- round_trip                                    -> TestStructuredCache::test_round_trip
- snapshot_version_tagged_on_put                -> TestStructuredCacheSnapshotVersionTagging::test_entry_serves_at_same_version
- stale_entry_evicted_after_snapshot_advance    -> TestStructuredCacheSnapshotVersionTagging::test_stale_entry_evicted_after_advance
- stale_eviction_counted_separately             -> TestStructuredCacheSnapshotVersionTagging::test_stale_eviction_counter
- no_provider_means_legacy_behaviour            -> TestStructuredCacheSnapshotVersionTagging::test_no_provider_accepts_all
- provider_failure_degrades_safely              -> TestStructuredCacheSnapshotVersionTagging::test_provider_exception_degrades_to_zero
"""
import random as _rnd
import time

import pytest

from semantic_search.cache.exact_cache import ExactCache
from semantic_search.cache.intent_result_cache import QIIntentResultCache
from semantic_search.cache.keys import versioned_query_key
from semantic_search.cache.lru_ttl import LRUTTLCache
from semantic_search.cache.structured_cache import StructuredCache
from semantic_search.config.models import AgentSearchConfig, ExactCacheConfig, QIIntentResultCacheConfig
from semantic_search.core.exceptions import CacheError, ValidationError as ASValidationError
from semantic_search.contracts import CachedSearchPayload, Candidate, CandidateSet, IntentSlice, QueryIntent, RankedItem, RankedResults


def _ranked(item_count: int = 1) -> RankedResults:
    """Build a small RankedResults for cache tests."""
    items = [RankedItem(item_id=f"i{i}", fused_score=1.0 - i * 0.1, contributing_sources=['vector'], payload={}) for i in range(item_count)]
    return RankedResults(request_id='req_x', items=items, total_candidates=item_count, fusion_latency_ms=0.0, cache_hit=None)


def _intent(query_type: str = 'hybrid') -> QueryIntent:
    """Minimal QueryIntent for payload-aware cache tests (c4-cache-personalize)."""
    slc = IntentSlice(query_type=query_type, entities=[], confidence=0.9, raw_text='q')
    return QueryIntent(
        request_id='req_x', raw_query='q', normalized_query='q',
        query_type=query_type, confidence=0.9, decision_tier='L0_entity',
        slices=[slc], decision_cost_usd=0.0,
    )


class TestCachedSearchPayloadContract:
    """c4-cache-personalize — input contract tests for the payload dataclass.

    The payload is a trust-boundary type (it sits inside the cache and is
    consumed by the orchestrator's cache-hit branches), so __post_init__ must
    reject malformed values per ``data-contracts.mdc``.
    """

    def test_round_trip(self):
        intent = _intent('hybrid')
        results = _ranked(1)
        payload = CachedSearchPayload(intent=intent, results=results)
        assert payload.intent is intent
        assert payload.results is results

    def test_intent_must_be_query_intent(self):
        with pytest.raises(ASValidationError, match="CachedSearchPayload.intent"):
            CachedSearchPayload(intent=None, results=_ranked(1))
        with pytest.raises(ASValidationError, match="CachedSearchPayload.intent"):
            CachedSearchPayload(intent="not_an_intent", results=_ranked(1))

    def test_results_must_be_ranked_results(self):
        with pytest.raises(ASValidationError, match="CachedSearchPayload.results"):
            CachedSearchPayload(intent=_intent(), results=None)
        with pytest.raises(ASValidationError, match="CachedSearchPayload.results"):
            CachedSearchPayload(intent=_intent(), results=[])


class TestLRUTTLCache:
    def test_basic_put_get(self):
        c: LRUTTLCache[str] = LRUTTLCache(max_entries=2, ttl_seconds=10)
        c.put('k1', 'v1')
        assert c.get('k1') == 'v1'
        assert c.hits == 1

    def test_lru_eviction(self):
        c: LRUTTLCache[str] = LRUTTLCache(max_entries=2, ttl_seconds=10)
        c.put('k1', 'v1')
        c.put('k2', 'v2')
        c.put('k3', 'v3')
        assert c.get('k1') is None
        assert c.get('k2') == 'v2'
        assert c.get('k3') == 'v3'

    def test_ttl_expiry(self):
        c: LRUTTLCache[str] = LRUTTLCache(max_entries=10, ttl_seconds=1)
        c.put('k1', 'v1')
        time.sleep(1.1)
        assert c.get('k1') is None

    def test_invalid_construction(self):
        with pytest.raises(CacheError):
            LRUTTLCache(max_entries=0, ttl_seconds=10)
        with pytest.raises(CacheError):
            LRUTTLCache(max_entries=1, ttl_seconds=0)

    def test_put_empty_key_rejected(self):
        c: LRUTTLCache[str] = LRUTTLCache(max_entries=2, ttl_seconds=10)
        with pytest.raises(CacheError):
            c.put('', 'v')


class TestExactCache:
    def test_round_trip(self, exact_cache: ExactCache):
        r = _ranked(2)
        exact_cache.put('com domains', r)
        got = exact_cache.get('com domains')
        assert got is not None
        assert len(got.items) == 2

    def test_miss_on_unseen_query(self, exact_cache: ExactCache):
        assert exact_cache.get('never seen') is None

    def test_disabled_returns_none(self):
        cfg = ExactCacheConfig(enabled=False, ttl_seconds=10, max_entries=10)
        cache = ExactCache(cfg)
        cache.put('q', _ranked(1))
        assert cache.get('q') is None

    def test_payload_round_trip_carries_intent(self, exact_cache: ExactCache):
        """c4-cache-personalize — putting with intent stores a CachedSearchPayload.
        get_payload returns the full payload; get returns just the results.
        """
        r = _ranked(2)
        intent = _intent('hybrid')
        exact_cache.put('com domains', r, intent=intent)
        payload = exact_cache.get_payload('com domains')
        assert payload is not None
        assert isinstance(payload, CachedSearchPayload)
        assert payload.intent.query_type == 'hybrid'
        assert payload.intent.normalized_query == 'q'
        assert len(payload.results.items) == 2
        # Legacy accessor still returns just RankedResults from a payload entry
        bare = exact_cache.get('com domains')
        assert bare is not None
        assert len(bare.items) == 2

    def test_bare_put_get_payload_returns_none(self, exact_cache: ExactCache):
        """c4-cache-personalize — entries written without intent (legacy path)
        do NOT show up via get_payload. The orchestrator falls back to the
        bare get + cache_hit-marker path for those.
        """
        exact_cache.put('legacy q', _ranked(1))
        assert exact_cache.get_payload('legacy q') is None
        # Bare get still works for legacy entries
        assert exact_cache.get('legacy q') is not None

    def test_lookup_is_case_insensitive(self, exact_cache: ExactCache):
        """Exact result cache folds case at the key boundary: a mixed-case
        query hits an entry stored under a different case, and surrounding
        whitespace is stripped.
        """
        exact_cache.put('COM Domains', _ranked(2))
        assert exact_cache.get('com domains') is not None
        assert exact_cache.get('CoM dOmAiNs') is not None
        assert exact_cache.get('  com domains  ') is not None

    def test_numerically_distinct_queries_do_not_collide(self, exact_cache: ExactCache):
        """Numerically distinct queries map to distinct keys — exact matching
        never conflates `under $100` with `under $200`.
        """
        exact_cache.put('com domains under $100', _ranked(1))
        assert exact_cache.get('com domains under $100') is not None
        assert exact_cache.get('com domains under $200') is None


class TestStructuredCache:
    def test_round_trip(self, structured_cache: StructuredCache):
        cs = CandidateSet(source='structured', candidates=[Candidate(item_id='a', score=0.5, source='structured', payload={})], latency_ms=1.0)
        filters = {'tld': ['com'], 'price_max': 100}
        structured_cache.put('hybrid', filters, cs)
        got = structured_cache.get('hybrid', filters)
        assert got is not None
        assert got.candidates[0].item_id == 'a'

    def test_filter_order_independent(self, structured_cache: StructuredCache):
        cs = CandidateSet(source='structured', candidates=[Candidate(item_id='a', score=0.5, source='structured', payload={})], latency_ms=1.0)
        structured_cache.put('hybrid', {'tld': ['com'], 'price_max': 100}, cs)
        got = structured_cache.get('hybrid', {'price_max': 100, 'tld': ['com']})
        assert got is not None

    def test_empty_filters_returns_none(self, structured_cache: StructuredCache):
        assert structured_cache.get('hybrid', {}) is None


class TestStructuredCacheSnapshotVersionTagging:
    """Per-entry snapshot-version tagging.

    Cache entries primed at version N are silently treated as misses once the
    registry advances past N. Validated against the live
    ``snapshot_version_provider`` callable rather than a hard ``invalidate_all``
    cascade so warm entries continue to serve when only a subset of inventory
    is touched.
    """

    @staticmethod
    def _build(version_box, structured_cache_config):
        # Builds a fresh StructuredCache wired to a mutable version box so
        # tests can advance the snapshot version arbitrarily without
        # constructing a full SnapshotVersionRegistry.
        return StructuredCache(structured_cache_config, snapshot_version_provider=lambda: version_box[0])

    def test_entry_serves_at_same_version(self, config):
        version_box = [5]
        cache = self._build(version_box, config.cache.structured)
        cs = CandidateSet(source='structured', candidates=[Candidate(item_id='a', score=0.5, source='structured', payload={})], latency_ms=1.0)
        cache.put('hybrid', {'tld': ['com']}, cs)
        # Same version on read → hit.
        assert cache.get('hybrid', {'tld': ['com']}) is not None
        assert cache.stale_evictions == 0

    def test_stale_entry_evicted_after_advance(self, config):
        version_box = [5]
        cache = self._build(version_box, config.cache.structured)
        cs = CandidateSet(source='structured', candidates=[Candidate(item_id='a', score=0.5, source='structured', payload={})], latency_ms=1.0)
        cache.put('hybrid', {'tld': ['com']}, cs)
        # Advance the snapshot — the cached entry is now stale.
        version_box[0] = 6
        result = cache.get('hybrid', {'tld': ['com']})
        assert result is None
        assert cache.stale_evictions == 1
        # And it is gone — a second read is a true miss, not another eviction.
        result2 = cache.get('hybrid', {'tld': ['com']})
        assert result2 is None
        assert cache.stale_evictions == 1

    def test_stale_eviction_counter(self, config):
        version_box = [1]
        cache = self._build(version_box, config.cache.structured)
        cs = CandidateSet(source='structured', candidates=[Candidate(item_id='a', score=0.5, source='structured', payload={})], latency_ms=1.0)
        # Two distinct entries → two stale evictions on advance.
        cache.put('hybrid', {'tld': ['com']}, cs)
        cache.put('hybrid', {'tld': ['io']}, cs)
        version_box[0] = 2
        cache.get('hybrid', {'tld': ['com']})
        cache.get('hybrid', {'tld': ['io']})
        assert cache.stale_evictions == 2
        # ``hits`` excludes stale evictions so dashboards see honest hit rate.
        assert cache.hits == 0

    def test_no_provider_accepts_all(self, config):
        # No provider → snapshot version always 0 → every entry stays valid.
        cache = StructuredCache(config.cache.structured, snapshot_version_provider=None)
        cs = CandidateSet(source='structured', candidates=[Candidate(item_id='a', score=0.5, source='structured', payload={})], latency_ms=1.0)
        cache.put('hybrid', {'tld': ['com']}, cs)
        assert cache.get('hybrid', {'tld': ['com']}) is not None
        assert cache.stale_evictions == 0

    def test_provider_exception_degrades_to_zero(self, config):
        # Misbehaving provider must not blow the cache tier — degrade to
        # version=0 (which means "accept everything tagged at >= 0",
        # i.e. nothing is ever stale). Caller-side soft-fail philosophy.
        def boom():
            raise RuntimeError("registry exploded")
        cache = StructuredCache(config.cache.structured, snapshot_version_provider=boom)
        cs = CandidateSet(source='structured', candidates=[Candidate(item_id='a', score=0.5, source='structured', payload={})], latency_ms=1.0)
        cache.put('hybrid', {'tld': ['com']}, cs)
        # Provider keeps failing → entry stays valid, no stale eviction.
        assert cache.get('hybrid', {'tld': ['com']}) is not None
        assert cache.stale_evictions == 0


class TestVersionedQueryKey:
    def test_requires_all_components(self):
        with pytest.raises(CacheError):
            versioned_query_key('', prompt_tag='p', schema_version='1')
        with pytest.raises(CacheError):
            versioned_query_key('q', prompt_tag='', schema_version='1')
        with pytest.raises(CacheError):
            versioned_query_key('q', prompt_tag='p', schema_version='')

    def test_prompt_or_schema_bump_isolates_intent_result_entries(self):
        cfg = QIIntentResultCacheConfig(
            enabled=True,
            ttl_seconds=60,
            max_entries=10,
            scale_at_hit_count=100,
            max_entries_scaled=20,
            ttl_seconds_scaled=120,
        )
        cache = QIIntentResultCache(cfg)
        slice_ = IntentSlice(query_type='hybrid', entities=[], confidence=0.9, raw_text='q')
        intent = QueryIntent(
            request_id='r1',
            raw_query='q',
            normalized_query='q',
            query_type='hybrid',
            confidence=0.9,
            decision_tier='L2_llm',
            slices=[slice_],
            decision_cost_usd=0.0,
            prompt_tag='qi.classify.v19',
            schema_version='1',
        )
        key_v1 = versioned_query_key('q', prompt_tag='qi.classify.v19', schema_version='1')
        key_v2 = versioned_query_key('q', prompt_tag='qi.classify.v20', schema_version='1')
        key_s2 = versioned_query_key('q', prompt_tag='qi.classify.v19', schema_version='2')
        cache.put(key_v1, intent)
        assert cache.get(key_v1) is not None
        assert cache.get(key_v2) is None
        assert cache.get(key_s2) is None

