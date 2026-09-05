"""Tests for semantic_search.qi.residual_extractor — concept-slot flow.

Coverage matrix:
``extract_residual`` — Path A (concept slots including similar_to):
- similar_to_hard_chip_yields_semantic_query   -> TestExtractResidualPathA::test_similar_to_single_seed
- similar_to_multi_seed_joined                 -> TestExtractResidualPathA::test_similar_to_multi_seed
- keyword_contains_path_a                      -> TestExtractResidualPathA::test_keyword_contains_path_a
- soft_chip_path_a                             -> TestExtractResidualPathA::test_soft_chip_path_a
- similar_to_navigational_single_token         -> TestExtractResidualPathA::test_similar_to_navigational_single_token

``extract_residual`` — Path B:
- tld_only_query_navigational                  -> TestExtractResidualPathB::test_tld_only_navigational
- content_tokens_remain_semantic               -> TestExtractResidualPathB::test_content_tokens_semantic

``build_semantic_encode_text``:
- similar_to_concept_reaches_encode_text       -> TestBuildSemanticEncodeTextSimilarTo::test_similar_to_in_encode_text
- similar_to_with_tld_filter_stripped          -> TestBuildSemanticEncodeTextSimilarTo::test_similar_to_tld_stripped

``semantic_encode_text_for``:
- intent_with_similar_to_entity                -> TestSemanticEncodeTextForSimilarTo::test_intent_with_similar_to

Robustness:
- none_normalized_returns_empty_none           -> TestExtractResidualRobustness::test_none_normalized
- empty_normalized_returns_empty_none          -> TestExtractResidualRobustness::test_empty_normalized
- no_entities_path_b                           -> TestExtractResidualRobustness::test_no_entities

All tests mock the encoder — no model loading required.
"""
import pytest

from semantic_search.config.models import ResidualQIConfig
from semantic_search.contracts import Entity, IntentSlice, QueryIntent
from semantic_search.qi.residual_extractor import (
    build_semantic_encode_text,
    extract_residual,
    semantic_encode_text_for,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _cfg(min_tokens: int = 1) -> ResidualQIConfig:
    """Return a minimal ResidualQIConfig for tests."""
    return ResidualQIConfig(enabled=True, navigational_tokens=["domain", "domains", "auction"], min_content_tokens_for_semantic=min_tokens)


def _ent(name: str, value, chip_kind: str = "hard") -> Entity:
    return Entity(name=name, value=value, confidence=0.9, source="L0_entity", chip_kind=chip_kind)


def _intent(normalized: str, entities=None, semantic_query: str = None, semantic_encode_text: str = None) -> QueryIntent:
    ents = list(entities or [])
    return QueryIntent(
        request_id="req_test",
        raw_query=normalized,
        normalized_query=normalized,
        query_type="hybrid",
        confidence=0.9,
        decision_tier="L0_entity",
        slices=[IntentSlice(query_type="hybrid", entities=ents, confidence=0.9, raw_text=normalized)],
        decision_cost_usd=0.0,
        semantic_query=semantic_query,
        semantic_encode_text=semantic_encode_text,
    )


# ---------------------------------------------------------------------------
# extract_residual — Path A concept slots
# ---------------------------------------------------------------------------

class TestExtractResidualPathA:
    def test_similar_to_single_seed(self):
        """similar_to hard entity yields SLD as semantic_query via Path A."""
        ents = [_ent("similar_to", ["stripe"], chip_kind="hard")]
        sq, kind = extract_residual("similar to stripe com domains", ents, _cfg())
        assert sq == "stripe"
        assert kind == "semantic"

    def test_similar_to_multi_seed(self):
        """Multiple SLD seeds are joined with a space as semantic_query."""
        ents = [_ent("similar_to", ["stripe", "plaid"], chip_kind="hard")]
        sq, kind = extract_residual("comparable to stripe com or plaid com", ents, _cfg())
        assert sq is not None
        assert "stripe" in sq
        assert "plaid" in sq
        assert kind == "semantic"

    def test_keyword_contains_path_a(self):
        """keyword_contains hard entity yields its value as semantic_query."""
        ents = [_ent("keyword_contains", "fintech", chip_kind="hard")]
        sq, kind = extract_residual("fintech domains", ents, _cfg())
        assert sq == "fintech"
        assert kind == "semantic"

    def test_soft_chip_path_a(self):
        """Soft-chip entity also triggers Path A."""
        ents = [_ent("theme", "machine learning", chip_kind="soft")]
        sq, kind = extract_residual("machine learning domains", ents, _cfg())
        assert sq is not None
        assert kind == "semantic"

    def test_similar_to_navigational_single_token(self):
        """Single-token similar_to seed is still navigational when min_content_tokens=2."""
        ents = [_ent("similar_to", ["a"], chip_kind="hard")]
        sq, kind = extract_residual("similar to a", ents, _cfg(min_tokens=2))
        assert sq is None
        assert kind == "navigational"

    def test_similar_to_list_joined_properly(self):
        """Three seeds are joined; result contains all SLDs."""
        ents = [_ent("similar_to", ["stripe", "plaid", "square"], chip_kind="hard")]
        sq, kind = extract_residual("query text", ents, _cfg())
        assert sq is not None
        for sld in ["stripe", "plaid", "square"]:
            assert sld in sq


# ---------------------------------------------------------------------------
# extract_residual — Path B
# ---------------------------------------------------------------------------

class TestExtractResidualPathB:
    def test_tld_only_no_semantic_query(self):
        """After stripping the tld entity value and nav tokens, zero tokens remain → no semantic_query."""
        ents = [_ent("tld", ["io"], chip_kind="hard")]
        sq, kind = extract_residual("io domains", ents, _cfg())
        assert sq is None
        assert kind in ("empty", "navigational")

    def test_content_tokens_semantic(self):
        """Residual content tokens after filter stripping yield a semantic query."""
        ents = [_ent("tld", ["com"], chip_kind="hard")]
        sq, kind = extract_residual("fintech startup com domains", ents, _cfg())
        assert sq is not None
        assert "fintech" in sq
        assert "startup" in sq

    def test_empty_normalized(self):
        sq, kind = extract_residual("", [], _cfg())
        assert sq is None
        assert kind == "empty"


# ---------------------------------------------------------------------------
# build_semantic_encode_text — similar_to concept reaches encode text
# ---------------------------------------------------------------------------

class TestBuildSemanticEncodeTextSimilarTo:
    def test_similar_to_in_encode_text(self):
        """When semantic_query carries the SLD, it flows straight through."""
        out = build_semantic_encode_text("similar to stripe com domains", [], "stripe")
        assert out == "stripe"

    def test_similar_to_tld_stripped(self):
        """TLD literals are stripped from fallback even when similar_to is present via semantic_query."""
        tld_ent = _ent("tld", ["com"], chip_kind="hard")
        out = build_semantic_encode_text("similar to stripe com domains", [tld_ent], "stripe")
        assert out == "stripe"
        assert "com" not in out.split()

    def test_fallback_strips_tld_when_no_semantic_query(self):
        """Fallback path strips tld token from normalized."""
        tld_ent = _ent("tld", ["io"], chip_kind="hard")
        out = build_semantic_encode_text("io domains", [tld_ent], None)
        assert "io" not in out.split()


# ---------------------------------------------------------------------------
# semantic_encode_text_for — intent with similar_to entity
# ---------------------------------------------------------------------------

class TestSemanticEncodeTextForSimilarTo:
    def test_intent_with_similar_to_prefers_contract_field(self):
        """When semantic_encode_text is set on the intent, that wins."""
        intent = _intent("similar to stripe com", semantic_encode_text="stripe")
        assert semantic_encode_text_for(intent) == "stripe"

    def test_intent_with_similar_to_falls_back_to_semantic_query(self):
        """Falls back to semantic_query when contract field is absent."""
        intent = _intent("similar to stripe com", semantic_query="stripe", semantic_encode_text=None)
        assert semantic_encode_text_for(intent) == "stripe"

    def test_intent_derives_from_slices_when_both_absent(self):
        """Derives TLD-stripped text from slices when neither field is set."""
        ents = [_ent("tld", ["com"], chip_kind="hard")]
        intent = _intent("fintech com domains", ents, semantic_query=None, semantic_encode_text=None)
        out = semantic_encode_text_for(intent)
        assert "com" not in out.split()
        assert "fintech" in out


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

class TestExtractResidualRobustness:
    def test_none_normalized(self):
        sq, kind = extract_residual(None, [], _cfg())
        assert sq is None
        assert kind == "empty"

    def test_empty_entities_path_b(self):
        sq, kind = extract_residual("some semantic concept", [], _cfg())
        assert sq == "some semantic concept"
        assert kind == "semantic"

    def test_similar_to_empty_list_ignored(self):
        """An empty similar_to list value must not crash and falls through to Path B."""
        ents = [_ent("similar_to", [], chip_kind="hard")]
        sq, kind = extract_residual("similar to something", ents, _cfg())
        # Empty list joins to "" which is stripped; Path A produces empty string → Path B
        # OR Path A still runs but empty string concept falls through to navigational.
        assert kind in ("semantic", "navigational", "empty")

    def test_similar_to_none_value_ignored(self):
        """A None similar_to entity value must not crash."""
        ents = [_ent("similar_to", None, chip_kind="hard")]
        sq, kind = extract_residual("similar to something", ents, _cfg())
        assert kind in ("semantic", "navigational", "empty")


# ---------------------------------------------------------------------------
# Parametrized similar_to concept-slot flow
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("slds,query,expect_in_sq", [
    (["stripe"], "similar to stripe com", "stripe"),
    (["bitcoin", "wallet"], "comparable to bitcoinwallet com", "bitcoin"),
    (["openai"], "like openai com", "openai"),
])
def test_similar_to_concept_flow_parametrized(slds, query, expect_in_sq):
    ents = [_ent("similar_to", slds, chip_kind="hard")]
    sq, kind = extract_residual(query, ents, _cfg())
    assert sq is not None
    assert expect_in_sq in sq
    assert kind == "semantic"
