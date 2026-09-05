"""Orchestrator-level test: the fuzzy lexical boost must be PERSISTED into
``fused_score`` so it survives the downstream engagement / brandability sorts.

Regression guard for the bug where ``_apply_fuzzy_rerank`` reordered the pool
but kept each item's original ``fused_score`` — the very next stage
(``_apply_engagement_boost``) re-sorts by ``fused_score`` and silently undid the
fuzzy reorder, so the lexical scorer had zero effect on the final ranking.
"""
from types import SimpleNamespace

import pytest

from semantic_search.config.models import FuzzyRerankConfig
from semantic_search.contracts import RankedItem, RankedResults
from semantic_search.orchestrator import SearchOrchestrator
from semantic_search.retrieval.fuzzy_lexical_reranker import FuzzyLexicalReranker


def _ranked(item_id: str, fused_score: float, sld: str) -> RankedItem:
    return RankedItem(
        item_id=item_id,
        fused_score=fused_score,
        contributing_sources=["vector"],
        payload={"sld": sld},
    )


def _orchestrator_with_fuzzy(boost_weight: float = 0.5, max_candidates: int = 200) -> SearchOrchestrator:
    """Build a bare SearchOrchestrator carrying only the two attrs
    ``_apply_fuzzy_rerank`` reads — avoids the heavy DI wiring."""
    cfg = FuzzyRerankConfig(
        enabled=True,
        max_edit_distance=2,
        min_token_length=2,
        max_terms=16,
        max_candidates=max_candidates,
        min_similarity=0.6,
        boost_weight=boost_weight,
        stopwords=["the", "a"],
    )
    orch = SearchOrchestrator.__new__(SearchOrchestrator)
    orch._fuzzy_reranker = FuzzyLexicalReranker(cfg)
    orch._fuzzy_rerank_max_candidates = max_candidates
    return orch


def _fused(*items: RankedItem) -> RankedResults:
    return RankedResults(
        request_id="req-1",
        items=list(items),
        total_candidates=len(items),
        fusion_latency_ms=0.0,
        cache_hit=None,
    )


def test_boost_is_persisted_into_fused_score():
    """A low-fused near-match outranks a higher-fused non-match AND carries the
    boosted score forward in ``fused_score`` (not a discarded side channel)."""
    orch = _orchestrator_with_fuzzy(boost_weight=0.5)
    intent = SimpleNamespace(normalized_query="high rentals")
    fused = _fused(
        _ranked("techhub", 0.40, "techhub"),      # no lexical match
        _ranked("hirent", 0.30, "hi-rentals"),     # full match -> coverage 1.0
    )

    out = orch._apply_fuzzy_rerank(intent, fused)

    by_id = {it.item_id: it for it in out.items}
    # boost (0.5 * 1.0) written back into fused_score, not dropped
    assert by_id["hirent"].fused_score == pytest.approx(0.80)
    # non-match is untouched
    assert by_id["techhub"].fused_score == pytest.approx(0.40)
    # and the boosted near-match now leads the pool
    assert out.items[0].item_id == "hirent"


def test_no_match_leaves_fused_score_unchanged():
    """Zero coverage => fused_score identical => stable input order preserved."""
    orch = _orchestrator_with_fuzzy(boost_weight=0.5)
    intent = SimpleNamespace(normalized_query="zzzz")
    fused = _fused(
        _ranked("a", 0.40, "techhub"),
        _ranked("b", 0.30, "hi-rentals"),
    )

    out = orch._apply_fuzzy_rerank(intent, fused)

    assert [it.item_id for it in out.items] == ["a", "b"]
    assert out.items[0].fused_score == pytest.approx(0.40)
    assert out.items[1].fused_score == pytest.approx(0.30)


def test_higher_boost_weight_yields_larger_persisted_boost():
    """boost_weight is the strength knob: a larger weight => larger fused_score
    delta for the same coverage (Item-2 tuning lever is now effective)."""
    intent = SimpleNamespace(normalized_query="high rentals")
    item = _ranked("hirent", 0.30, "hi-rentals")  # coverage 1.0

    low = _orchestrator_with_fuzzy(boost_weight=0.5)._apply_fuzzy_rerank(intent, _fused(item))
    high = _orchestrator_with_fuzzy(boost_weight=1.0)._apply_fuzzy_rerank(intent, _fused(item))

    assert low.items[0].fused_score == pytest.approx(0.80)
    assert high.items[0].fused_score == pytest.approx(1.30)
