"""Backend-unavailable /search envelope: HTTP 503 + clear JSON (not Internal Server Error)."""
from __future__ import annotations

from semantic_search.analytics.clickhouse_client import ClickHouseUnavailableError
from semantic_search.app import (
    _backend_matches_query_primary,
    _backend_unavailable_envelope,
    _backend_unavailable_response,
    _classify_backend_unavailable,
    _is_analytics_primary_query,
)
from semantic_search.contracts import RankedResults
from semantic_search.core.exceptions import QdrantQueryError, QdrantUnavailableError


def test_classify_qdrant_connection_refused() -> None:
    backend, mode, msg = _classify_backend_unavailable(
        QdrantQueryError("failed to connect to all addresses; Connection refused [::1]:6334")
    )
    assert backend == 'qdrant'
    assert mode == 'qdrant_unavailable'
    assert 'not reachable' in msg.lower()
    assert '6334' in msg or 'Connection refused' in msg


def test_classify_qdrant_missing_collection() -> None:
    backend, mode, msg = _classify_backend_unavailable(
        QdrantQueryError("Collection `auctions_listings` doesn't exist!")
    )
    assert backend == 'qdrant'
    assert mode == 'qdrant_unavailable'
    assert 'collection is missing' in msg.lower()
    assert 'data-build' in msg.lower()


def test_classify_clickhouse_unavailable() -> None:
    backend, mode, msg = _classify_backend_unavailable(
        ClickHouseUnavailableError("connection refused to clickhouse:8123")
    )
    assert backend == 'clickhouse'
    assert mode == 'clickhouse_unavailable'
    assert 'clickhouse is unavailable' in msg.lower()
    assert 'host/port' in msg.lower()


def test_envelope_includes_error_and_request_id() -> None:
    exc = QdrantUnavailableError("Qdrant client unavailable")
    body = _backend_unavailable_envelope(
        query='find .com domains',
        request_id='rid-test-1',
        search_id='search-test-1',
        exc=exc,
        latency_ms=12.3,
    )
    assert body['ranked_results'] == []
    assert body['request_id'] == 'rid-test-1'
    assert body['search_id'] == 'search-test-1'
    assert body['retrieval_metrics']['failure_mode'] == 'qdrant_unavailable'
    assert body['retrieval_metrics']['error'] == {
        'backend': 'qdrant',
        'error_type': 'QdrantUnavailableError',
        'message': 'Qdrant client unavailable',
        'request_id': 'rid-test-1',
        'search_id': 'search-test-1',
    }
    assert 'request_id=rid-test-1' in body['guard_notice']
    assert 'search_id=search-test-1' in body['guard_notice']
    assert 'Qdrant is unavailable' in body['guard_notice']


def test_response_is_http_503() -> None:
    resp = _backend_unavailable_response(
        query='x',
        request_id='rid-503',
        search_id='search-503',
        exc=QdrantQueryError('connection refused'),
        latency_ms=1.0,
    )
    assert resp.status_code == 503
    assert resp.body  # JSON body present


def test_ranked_results_accepts_new_failure_modes() -> None:
    for mode in ('qdrant_unavailable', 'clickhouse_unavailable'):
        rr = RankedResults(
            request_id='rid',
            items=[],
            total_candidates=0,
            fusion_latency_ms=0.0,
            failure_mode=mode,
        )
        assert rr.failure_mode == mode


def test_guard_notice_not_inventory_empty_when_qdrant_down() -> None:
    """Qdrant-down envelope must name Qdrant — not filter-relax / inventory_empty."""
    exc = QdrantQueryError("failed to connect to all addresses; Connection refused [::1]:6334")
    body = _backend_unavailable_envelope(
        query='find .com domains for auction type 16 under $50',
        request_id='rid-qdrant-down',
        search_id='search-qdrant-down',
        exc=exc,
        latency_ms=100.0,
    )
    assert body['retrieval_metrics']['failure_mode'] == 'qdrant_unavailable'
    assert 'inventory_empty' not in str(body)
    assert 'filter_relaxed' not in str(body).lower()
    assert 'Qdrant is unavailable' in body['guard_notice']
    assert 'not reachable' in body['guard_notice'].lower() or 'Connection refused' in body['guard_notice']


def test_guard_notice_clear_when_clickhouse_down() -> None:
    exc = ClickHouseUnavailableError(
        "ClickHouse analytics is unavailable (router not wired, credentials missing, or ClickHouse down)."
    )
    body = _backend_unavailable_envelope(
        query='compare between .com and .net domain average for last week',
        request_id='rid-ch-down',
        search_id='search-ch-down',
        exc=exc,
        latency_ms=100.0,
        answer_mode='analytics',
    )
    assert body['answer_mode'] == 'analytics'
    assert body['retrieval_metrics']['failure_mode'] == 'clickhouse_unavailable'
    assert 'showing trending' not in body['guard_notice'].lower()
    assert 'ClickHouse is unavailable' in body['guard_notice']


def test_analytics_primary_detects_analytics_type() -> None:
    assert _is_analytics_primary_query(
        query_type='analytics', l1_type='hybrid', use_analytics_budget=False,
    )
    assert _is_analytics_primary_query(
        query_type=None, l1_type='analytics', use_analytics_budget=False,
    )
    assert _is_analytics_primary_query(
        query_type='hybrid', l1_type='hybrid', use_analytics_budget=True,
    )
    assert not _is_analytics_primary_query(
        query_type='hybrid', l1_type='hybrid', use_analytics_budget=False,
    )


def test_backend_match_analytics_ignores_qdrant() -> None:
    qd = QdrantQueryError('connection refused [::1]:6334')
    ch = ClickHouseUnavailableError('connection refused clickhouse:8123')
    assert not _backend_matches_query_primary(qd, analytics_primary=True)
    assert _backend_matches_query_primary(ch, analytics_primary=True)


def test_backend_match_listing_ignores_clickhouse() -> None:
    qd = QdrantUnavailableError('Qdrant client unavailable')
    ch = ClickHouseUnavailableError('connection refused clickhouse:8123')
    assert _backend_matches_query_primary(qd, analytics_primary=False)
    assert not _backend_matches_query_primary(ch, analytics_primary=False)
