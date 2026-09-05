"""Tests for ``semantic_search.retrieval.fuzzy_lexical_reranker``."""
import copy
import pytest

from semantic_search.config.models import FuzzyRerankConfig
from semantic_search.contracts import RankedItem
from semantic_search.core.exceptions import ConfigurationError, RetrievalError, ValidationError
from semantic_search.retrieval.fuzzy_lexical_reranker import FuzzyLexicalReranker
from semantic_search.retrieval.reranker_base import RerankedItem


def _valid_fuzzy_config(**overrides) -> FuzzyRerankConfig:
    """Build a valid FuzzyRerankConfig with optional field overrides."""
    base = dict(
        enabled=True,
        max_edit_distance=2,
        min_token_length=2,
        max_terms=16,
        max_candidates=200,
        min_similarity=0.6,
        boost_weight=0.5,
        stopwords=["the", "a"],
    )
    base.update(overrides)
    return FuzzyRerankConfig(**base)


def _ranked(item_id: str, fused_score: float, sld: str) -> RankedItem:
    return RankedItem(
        item_id=item_id,
        fused_score=fused_score,
        contributing_sources=["vector"],
        payload={"sld": sld},
    )


@pytest.fixture
def reranker() -> FuzzyLexicalReranker:
    return FuzzyLexicalReranker(_valid_fuzzy_config())


def test_constructor_rejects_non_config():
    """Wrong config type raises RetrievalError."""
    with pytest.raises(RetrievalError, match="FuzzyLexicalReranker requires a FuzzyRerankConfig"):
        FuzzyLexicalReranker(None)  # type: ignore[arg-type]
    with pytest.raises(RetrievalError, match="FuzzyLexicalReranker requires a FuzzyRerankConfig"):
        FuzzyLexicalReranker({"max_edit_distance": 2})  # type: ignore[arg-type]


def test_name_property(reranker: FuzzyLexicalReranker):
    assert reranker.name == "fuzzy_lexical"


def test_high_rentals_rerank_order_and_scores(reranker: FuzzyLexicalReranker):
    """Query ``high rentals`` boosts hi-rentals and rent over techhub."""
    items = [
        _ranked("1", 0.40, "techhub"),
        _ranked("2", 0.30, "hi-rentals"),
        _ranked("3", 0.35, "rent"),
    ]
    out = reranker.rerank("high rentals", items, top_n=3)
    assert [r.item.item_id for r in out] == ["2", "3", "1"]
    assert out[0].rerank_score == pytest.approx(0.80)
    assert out[1].rerank_score == pytest.approx(0.60)
    assert out[2].rerank_score == pytest.approx(0.40)


def test_typo_within_edit_distance_gets_boost(reranker: FuzzyLexicalReranker):
    """Near-match SLD token raises rerank_score above fused_score."""
    item = _ranked("r", 0.25, "rentls")
    out = reranker.rerank("rentals", [item], top_n=1)
    assert len(out) == 1
    assert out[0].rerank_score > item.fused_score


def test_below_threshold_no_boost(reranker: FuzzyLexicalReranker):
    """Unrelated query leaves fused_score unchanged."""
    item = _ranked("t", 0.42, "techhub")
    out = reranker.rerank("zzzz", [item], top_n=1)
    assert out[0].rerank_score == pytest.approx(item.fused_score)


def test_no_match_preserves_input_order(reranker: FuzzyLexicalReranker):
    """Zero-coverage query keeps the fused-score-descending order (orchestrator
    feeds the pool already sorted; zero coverage => rerank_score == fused_score
    so the stable score-descending sort reproduces the input order)."""
    items = [
        _ranked("1", 0.40, "techhub"),
        _ranked("2", 0.35, "hi-rentals"),
        _ranked("3", 0.30, "rent"),
    ]
    out = reranker.rerank("qqqq", items, top_n=3)
    assert [r.item.item_id for r in out] == ["1", "2", "3"]
    for r, src in zip(out, items):
        assert r.rerank_score == pytest.approx(src.fused_score)


def test_top_n_slices_head_only(reranker: FuzzyLexicalReranker):
    """Only the first top_n items are rescored."""
    items = [
        _ranked("1", 0.40, "techhub"),
        _ranked("2", 0.30, "hi-rentals"),
        _ranked("3", 0.35, "rent"),
    ]
    out = reranker.rerank("high rentals", items, top_n=2)
    assert len(out) == 2
    assert {r.item.item_id for r in out} == {"1", "2"}


def test_rerank_does_not_mutate_inputs(reranker: FuzzyLexicalReranker):
    """Input list and RankedItem objects are unchanged."""
    items = [
        _ranked("1", 0.40, "techhub"),
        _ranked("2", 0.30, "hi-rentals"),
    ]
    snapshot = copy.deepcopy(items)
    ids_before = [id(x) for x in items]
    reranker.rerank("high rentals", items, top_n=2)
    assert items == snapshot
    assert [id(x) for x in items] == ids_before


def test_top_n_zero_returns_empty(reranker: FuzzyLexicalReranker):
    assert reranker.rerank("q", [_ranked("1", 0.1, "x")], top_n=0) == []


def test_empty_items_returns_empty(reranker: FuzzyLexicalReranker):
    assert reranker.rerank("q", [], top_n=3) == []


def test_negative_top_n_raises_validation_error(reranker: FuzzyLexicalReranker):
    with pytest.raises(ValidationError, match="top_n must be >= 0"):
        reranker.rerank("q", [_ranked("1", 0.1, "x")], top_n=-1)


def test_missing_sld_key_coverage_zero(reranker: FuzzyLexicalReranker):
    item = RankedItem(
        item_id="m",
        fused_score=0.55,
        contributing_sources=["vector"],
        payload={},
    )
    out = reranker.rerank("high rentals", [item], top_n=1)
    assert out[0].rerank_score == pytest.approx(0.55)


def test_returned_items_are_reranked_item_with_original_reference(reranker: FuzzyLexicalReranker):
    items = [_ranked("1", 0.40, "techhub"), _ranked("2", 0.30, "hi-rentals")]
    out = reranker.rerank("high rentals", items, top_n=2)
    for r in out:
        assert isinstance(r, RerankedItem)
        assert r.rerank_score >= 0.0
        assert r.item in items


def test_coverage_breadth_beats_single_exact_match(reranker: FuzzyLexicalReranker):
    """Domain matching both query tokens ranks above one matching only one token,
    even when the two-match domain has one fuzzy (< 1.0) score.

    Query 'hi rents' (tokens ['hi', 'rents']):
      'hi-rental' → 'hi' exact (1.0) + 'rents'~'rental' fuzzy (≥0.6) → 2 matched, best=1.0
                    coverage = 1.0 * (2/2) = 1.0
      'hitech'    → 'hi' exact (1.0), 'rents' no match  → 1 matched, best=1.0
                    coverage = 1.0 * (1/2) = 0.5

    Two-token match domain must outscore single-token domain despite same fused_score.
    """
    items = [
        _ranked("both", 0.40, "hi-rental"),
        _ranked("one",  0.40, "hitech"),
    ]
    out = reranker.rerank("hi rents", items, top_n=2)
    both_score = next(r.rerank_score for r in out if r.item.item_id == "both")
    one_score  = next(r.rerank_score for r in out if r.item.item_id == "one")
    assert both_score > one_score
    assert out[0].item.item_id == "both"


def test_coverage_multi_word_query_proportional(reranker: FuzzyLexicalReranker):
    """coverage = best_matched * (matched/total), not mean over all tokens.

    Query 'eco friendly rentals' (3 tokens):
      'ecorentals' → 'eco' (1.0) + 'rentals' (1.0), 'friendly' no match → 2/3 matched
      coverage = 1.0 * (2/3) ≈ 0.667; rerank ≈ fused + 0.5 * 0.667
    """
    item = _ranked("er", 0.50, "ecorentals")
    out = reranker.rerank("eco friendly rentals", [item], top_n=1)
    expected = pytest.approx(0.50 + 0.5 * (2.0 / 3.0), rel=1e-3)
    assert out[0].rerank_score == expected


@pytest.mark.parametrize(
    "overrides,match_substr",
    [
        ({"max_edit_distance": 0}, "max_edit_distance"),
        ({"min_token_length": 0}, "min_token_length"),
        ({"max_terms": 0}, "max_terms"),
        ({"max_candidates": 0}, "max_candidates"),
        ({"min_similarity": 1.5}, "min_similarity"),
        ({"min_similarity": -0.1}, "min_similarity"),
        ({"boost_weight": -1.0}, "boost_weight"),
    ],
)
def test_fuzzy_rerank_config_validation_raises(overrides, match_substr):
    with pytest.raises(ConfigurationError, match=match_substr):
        _valid_fuzzy_config(**overrides)
