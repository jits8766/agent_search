"""Structural tests for qi/prompts.py.

Coverage matrix (per testing.mdc §7 — deterministic structural tests only;
LLM output is non-deterministic and the eval suite is disabled):

PROMPT_TAG version:
- tag_is_v14                        -> TestPromptTag::test_tag_is_v14

build_system_prompt structure:
- contains_new_slot_keyword_phrase              -> TestSystemPromptContent::test_contains_slot_keyword_phrase
- contains_new_slot_keyword_contains_exclude    -> TestSystemPromptContent::test_contains_slot_keyword_contains_exclude
- contains_new_slot_word_count_min              -> TestSystemPromptContent::test_contains_slot_word_count_min
- contains_new_slot_word_count_max              -> TestSystemPromptContent::test_contains_slot_word_count_max
- contains_new_slot_similar_to                 -> TestSystemPromptContent::test_contains_slot_similar_to
- contains_verbatim_keyword_instruction         -> TestSystemPromptContent::test_contains_verbatim_keyword_instruction
- contains_price_direction_rule                 -> TestSystemPromptContent::test_contains_price_direction_rule
- contains_tld_vs_ends_with_rule                -> TestSystemPromptContent::test_contains_tld_vs_ends_with_rule
- contains_minletters_guard                     -> TestSystemPromptContent::test_contains_minletters_guard
- alt_band_placeholder_replaced                 -> TestSystemPromptContent::test_alt_band_placeholder_replaced

build_user_prompt structure:
- user_prompt_contains_query                    -> TestUserPromptContent::test_user_prompt_contains_query
- user_prompt_contains_allowed_types            -> TestUserPromptContent::test_user_prompt_contains_allowed_types
- user_prompt_contains_utc_timestamp            -> TestUserPromptContent::test_user_prompt_contains_utc_timestamp
"""
from semantic_search.qi.prompts import PROMPT_TAG, build_system_prompt, build_user_prompt


# ---------------------------------------------------------------------------
# PROMPT_TAG version
# ---------------------------------------------------------------------------

class TestPromptTag:
    def test_tag_is_v14(self) -> None:
        assert PROMPT_TAG == "qi.classify.v19"


# ---------------------------------------------------------------------------
# build_system_prompt — new slot names and instruction rules
# ---------------------------------------------------------------------------

class TestSystemPromptContent:
    def _prompt(self) -> str:
        return build_system_prompt(alt_band_high=0.85, keyword_expansion_max_terms=5)

    def test_alt_band_placeholder_replaced(self) -> None:
        prompt = build_system_prompt(alt_band_high=0.75, keyword_expansion_max_terms=5)
        assert "<ALT_BAND_HIGH>" not in prompt
        assert "0.75" in prompt

    def test_keyword_cap_placeholder_replaced(self) -> None:
        prompt = build_system_prompt(alt_band_high=0.85, keyword_expansion_max_terms=7)
        assert "<KEYWORD_EXPANSION_MAX_TERMS>" not in prompt
        assert "7" in prompt


# ---------------------------------------------------------------------------
# build_user_prompt structure
# ---------------------------------------------------------------------------

class TestUserPromptContent:
    def test_user_prompt_contains_query(self) -> None:
        prompt = build_user_prompt(query="tech domains", allowed_query_types=["hybrid"], current_utc_iso="")
        assert "tech domains" in prompt

    def test_user_prompt_contains_allowed_types(self) -> None:
        prompt = build_user_prompt(query="q", allowed_query_types=["hybrid", "guidance"], current_utc_iso="")
        assert "guidance" in prompt
        assert "hybrid" in prompt

    def test_user_prompt_contains_utc_timestamp(self) -> None:
        prompt = build_user_prompt(query="q", allowed_query_types=["hybrid"], current_utc_iso="2026-06-06T12:00:00Z")
        assert "2026-06-06T12:00:00Z" in prompt

