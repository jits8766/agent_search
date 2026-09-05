"""Tests for post-gather SQL demotion on semantic-residual queries.

Coverage matrix (per ``testing.mdc`` §7):

``_should_drop_sql_for_semantic_residual`` (orchestrator.py):
- drops_when_semantic_and_vector_has_candidates  -> test_drops_when_semantic_and_vector_has_candidates
- keeps_when_vector_empty                        -> test_keeps_when_vector_empty
- keeps_when_no_vector_candidate_set              -> test_keeps_when_no_vector_candidate_set
- keeps_when_residual_navigational                -> test_keeps_when_residual_navigational
- keeps_when_residual_none                        -> test_keeps_when_residual_none
"""
from semantic_search.contracts import Candidate, CandidateSet
from semantic_search.orchestrator import _should_drop_sql_for_semantic_residual


def _cs(source: str, n_candidates: int) -> CandidateSet:
    candidates = [Candidate(item_id=f'{source}-{i}', score=1.0, source=source, payload={}) for i in range(n_candidates)]
    return CandidateSet(source=source, candidates=candidates, latency_ms=1.0)


def test_drops_when_semantic_and_vector_has_candidates():
    results = [_cs('vector', 3), _cs('sql', 5)]
    assert _should_drop_sql_for_semantic_residual('semantic', results) is True


def test_keeps_when_vector_empty():
    results = [_cs('vector', 0), _cs('sql', 5)]
    assert _should_drop_sql_for_semantic_residual('semantic', results) is False


def test_keeps_when_no_vector_candidate_set():
    results = [_cs('sql', 5)]
    assert _should_drop_sql_for_semantic_residual('semantic', results) is False


def test_keeps_when_residual_navigational():
    results = [_cs('vector', 3), _cs('sql', 5)]
    assert _should_drop_sql_for_semantic_residual('navigational', results) is False


def test_keeps_when_residual_none():
    results = [_cs('vector', 3), _cs('sql', 5)]
    assert _should_drop_sql_for_semantic_residual(None, results) is False
