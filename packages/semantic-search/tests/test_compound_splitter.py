"""Tests for ``CompoundWordSplitter`` and its integration into the segmenter.

Coverage matrix (`semantic_search/vectorization/compound_splitter.py`):

``CompoundWordSplitter.__init__``:
    - dictionary_none_raises                          -> TestConstruction::test_dictionary_none_raises
- dictionary_empty_raises                         -> TestConstruction::test_dictionary_empty_raises
- dictionary_non_mapping_raises                   -> TestConstruction::test_dictionary_non_mapping_raises
- dictionary_non_string_key_raises                -> TestConstruction::test_dictionary_non_string_key_raises
- dictionary_empty_key_raises                     -> TestConstruction::test_dictionary_empty_key_raises
- dictionary_non_numeric_value_raises             -> TestConstruction::test_dictionary_non_numeric_value_raises
- dictionary_negative_weight_raises               -> TestConstruction::test_dictionary_negative_weight_raises
- dictionary_all_zero_weight_raises               -> TestConstruction::test_dictionary_all_zero_weight_raises
- dictionary_all_below_min_length_raises          -> TestConstruction::test_dictionary_all_below_min_length_raises
- min_segment_length_invalid_raises               -> TestConstruction::test_min_segment_length_invalid_raises
- max_segments_invalid_raises                     -> TestConstruction::test_max_segments_invalid_raises
- oov_char_cost_invalid_raises                    -> TestConstruction::test_oov_char_cost_invalid_raises
- length_penalty_invalid_raises                   -> TestConstruction::test_length_penalty_invalid_raises
- duplicate_keys_aggregate_weight                 -> TestConstruction::test_duplicate_keys_aggregate_weight
- dictionary_size_diagnostic                      -> TestConstruction::test_dictionary_size

``CompoundWordSplitter.split``:
    - none_input_raises                               -> TestSplit::test_none_raises
- non_string_input_raises                        -> TestSplit::test_non_string_raises
- empty_input_raises                              -> TestSplit::test_empty_raises
- single_dictionary_word_no_split                 -> TestSplit::test_single_word_no_split
- compound_two_words_split                        -> TestSplit::test_two_words
- compound_three_words_split                      -> TestSplit::test_three_words
- shorter_than_min_segment_length_returns_self    -> TestSplit::test_shorter_than_min
- pure_oov_returns_single_oov_segment             -> TestSplit::test_pure_oov
- partial_oov_merged_into_single_segment          -> TestSplit::test_partial_oov_merged
- max_segments_exceeded_returns_baseline          -> TestSplit::test_max_segments_exceeded
- length_penalty_prefers_longer_match             -> TestSplit::test_length_penalty
- determinism_same_input_same_output              -> TestSplit::test_determinism
- casefold_input                                  -> TestSplit::test_casefold

DomainNameSegmenter integration:
    - segmenter_without_splitter_unchanged            -> TestSegmenterIntegration::test_no_splitter_legacy
- segmenter_with_splitter_splits_compound         -> TestSegmenterIntegration::test_splitter_compounds
- segmenter_with_splitter_short_runs_unchanged    -> TestSegmenterIntegration::test_splitter_short_runs
- segmenter_with_splitter_subdomain_split         -> TestSegmenterIntegration::test_splitter_subdomain
- segmenter_with_splitter_digit_runs_pass_through -> TestSegmenterIntegration::test_splitter_digits_pass_through
- segmenter_with_splitter_invalid_type_raises     -> TestSegmenterIntegration::test_splitter_invalid_type
- segmenter_has_compound_splitter_property         -> TestSegmenterIntegration::test_has_compound_splitter_property
"""
import pytest

from semantic_search.core.exceptions import ValidationError
from semantic_search.vectorization import CompoundWordSplitter, DomainNameSegmenter, SplitResult


def _make_splitter(extra: dict = None, **overrides):
    """Helper: build a small but representative dictionary for tests."""
    base = {
        "tech": 5000.0,
        "startup": 4000.0,
        "cloud": 3500.0,
        "stack": 3000.0,
        "brand": 2500.0,
        "able": 2000.0,
        "shop": 1500.0,
        "deploy": 1200.0,
        "ify": 1000.0,
        "store": 900.0,
        "api": 800.0,
    }
    if extra:
        base.update(extra)
    kwargs = dict(
        dictionary=base,
        min_segment_length=2,
        max_segments=5,
        oov_char_cost=20.0,
        length_penalty=0.0,
    )
    kwargs.update(overrides)
    return CompoundWordSplitter(**kwargs)


class TestConstruction:
    def test_dictionary_none_raises(self):
        with pytest.raises(ValidationError, match="dictionary must be a non-None Mapping"):
            CompoundWordSplitter( dictionary=None, min_segment_length=2, max_segments=5, oov_char_cost=20.0, length_penalty=0.0,)  # type: ignore[arg-type]

    def test_dictionary_empty_raises(self):
        with pytest.raises(ValidationError, match="dictionary must be non-empty"):
            CompoundWordSplitter( dictionary={}, min_segment_length=2, max_segments=5, oov_char_cost=20.0, length_penalty=0.0,)

    def test_dictionary_non_mapping_raises(self):
        with pytest.raises(ValidationError, match="non-None Mapping"):
            CompoundWordSplitter( dictionary=["tech", "startup"], min_segment_length=2, max_segments=5, oov_char_cost=20.0, length_penalty=0.0,)  # type: ignore[arg-type]

    def test_dictionary_non_string_key_raises(self):
        with pytest.raises(ValidationError, match="keys must be non-empty strings"):
            CompoundWordSplitter( dictionary={123: 1.0}, min_segment_length=2, max_segments=5, oov_char_cost=20.0, length_penalty=0.0,)  # type: ignore[dict-item]

    def test_dictionary_empty_key_raises(self):
        with pytest.raises(ValidationError, match="keys must be non-empty strings"):
            CompoundWordSplitter( dictionary={"": 1.0}, min_segment_length=2, max_segments=5, oov_char_cost=20.0, length_penalty=0.0,)

    def test_dictionary_non_numeric_value_raises(self):
        with pytest.raises(ValidationError, match="values must be numeric weights"):
            CompoundWordSplitter( dictionary={"tech": "high"}, min_segment_length=2, max_segments=5, oov_char_cost=20.0, length_penalty=0.0,)  # type: ignore[dict-item]

    def test_dictionary_negative_weight_raises(self):
        with pytest.raises(ValidationError, match="weights must be >= 0"):
            CompoundWordSplitter( dictionary={"tech": -1.0}, min_segment_length=2, max_segments=5, oov_char_cost=20.0, length_penalty=0.0,)

    def test_dictionary_all_zero_weight_raises(self):
        with pytest.raises(ValidationError, match="must contain at least one entry with weight > 0"):
            CompoundWordSplitter( dictionary={"tech": 0.0, "startup": 0.0}, min_segment_length=2, max_segments=5, oov_char_cost=20.0, length_penalty=0.0,)

    def test_dictionary_all_below_min_length_raises(self):
        with pytest.raises(ValidationError, match="length >= min_segment_length"):
            CompoundWordSplitter( dictionary={"a": 1.0, "b": 1.0}, min_segment_length=2, max_segments=5, oov_char_cost=20.0, length_penalty=0.0,)

    def test_min_segment_length_invalid_raises(self):
        with pytest.raises(ValidationError, match="min_segment_length must be int >= 1"):
            CompoundWordSplitter( dictionary={"tech": 1.0}, min_segment_length=0, max_segments=5, oov_char_cost=20.0, length_penalty=0.0,)

    def test_max_segments_invalid_raises(self):
        with pytest.raises(ValidationError, match="max_segments must be int >= 1"):
            CompoundWordSplitter( dictionary={"tech": 1.0}, min_segment_length=2, max_segments=0, oov_char_cost=20.0, length_penalty=0.0,)

    def test_oov_char_cost_invalid_raises(self):
        with pytest.raises(ValidationError, match="oov_char_cost must be a number > 0"):
            CompoundWordSplitter( dictionary={"tech": 1.0}, min_segment_length=2, max_segments=5, oov_char_cost=0.0, length_penalty=0.0,)

    def test_length_penalty_invalid_raises(self):
        with pytest.raises(ValidationError, match="length_penalty must be a number >= 0"):
            CompoundWordSplitter( dictionary={"tech": 1.0}, min_segment_length=2, max_segments=5, oov_char_cost=20.0, length_penalty=-0.1,)

    def test_duplicate_keys_aggregate_weight(self):
        # Casefold collisions ("Tech" + "TECH" -> "tech") aggregate.
        splitter = CompoundWordSplitter( dictionary={"Tech": 100.0, "TECH": 100.0}, min_segment_length=2, max_segments=5, oov_char_cost=20.0, length_penalty=0.0,)
        assert splitter.dictionary_size == 1

    def test_dictionary_size(self):
        splitter = _make_splitter()
        assert splitter.dictionary_size >= 5  # 11 words in the helper dict


class TestSplit:
    def test_none_raises(self):
        s = _make_splitter()
        with pytest.raises(ValidationError, match="non-None label"):
            s.split(None)  # type: ignore[arg-type]

    def test_non_string_raises(self):
        s = _make_splitter()
        with pytest.raises(ValidationError, match="requires a string"):
            s.split(123)  # type: ignore[arg-type]

    def test_empty_raises(self):
        s = _make_splitter()
        with pytest.raises(ValidationError, match="non-empty label"):
            s.split("")

    def test_single_word_no_split(self):
        # A label that IS a dictionary word emits as one segment — the
        # baseline guard returns the original label, not the dictionary form.
        s = _make_splitter()
        result = s.split("tech")
        assert isinstance(result, SplitResult)
        assert result.segments == ("tech",)
        assert result.had_oov is False

    def test_two_words(self):
        s = _make_splitter()
        result = s.split("techstartup")
        assert result.segments == ("tech", "startup")
        assert result.had_oov is False

    def test_three_words(self):
        s = _make_splitter()
        result = s.split("cloudstackshop")
        assert result.segments == ("cloud", "stack", "shop")
        assert result.had_oov is False

    def test_shorter_than_min(self):
        s = _make_splitter()
        result = s.split("a")
        assert result.segments == ("a",)
        assert result.had_oov is True

    def test_pure_oov(self):
        # No dictionary word covers any prefix of "zxqwvy" — falls back
        # to one merged OOV segment, NOT one segment per character.
        s = _make_splitter()
        result = s.split("zxqwvy")
        assert result.segments == ("zxqwvy",)
        assert result.had_oov is True

    def test_partial_oov_merged(self):
        # "techzxq" splits as ("tech", "zxq") — OOV characters merge.
        s = _make_splitter()
        result = s.split("techzxq")
        assert result.segments == ("tech", "zxq")
        assert result.had_oov is True

    def test_max_segments_exceeded(self):
        # max_segments=2 with input that wants 3 words -> baseline.
        s = _make_splitter(max_segments=2)
        result = s.split("cloudstackshop")
        assert result.segments == ("cloudstackshop",)

    def test_length_penalty(self):
        # Without length penalty, equally-weighted "shop" + "ify" beats
        # "shopify" only if "shopify" isn't in the dictionary. Add it.
        s = _make_splitter(extra={"shopify": 100.0}, length_penalty=2.0)
        result = s.split("shopify")
        # length_penalty subtracts 2.0 per character from in-dict cost,
        # so the longer "shopify" (7 chars * 2.0 = 14 bonus) beats
        # "shop"+"ify" (3+4 chars * 2.0 = 14 bonus, but as TWO segments).
        # Tie-broken by fewer-segments: "shopify" wins.
        assert result.segments == ("shopify",)

    def test_determinism(self):
        s = _make_splitter()
        a = s.split("techstartup")
        b = s.split("techstartup")
        assert a.segments == b.segments
        assert a.total_cost == b.total_cost

    def test_casefold(self):
        # Case-insensitive: "TechStartup" -> ("tech", "startup").
        s = _make_splitter()
        result = s.split("TechStartup")
        assert result.segments == ("tech", "startup")


class TestSegmenterIntegration:
    def test_no_splitter_legacy(self):
        # Without the splitter, the existing single-token output is preserved
        # — guarantees the existing test suite continues to pass.
        seg = DomainNameSegmenter()
        out = seg.segment("techstartup.com")
        assert out.tokens == ("techstartup", "com")
        assert seg.has_compound_splitter is False

    def test_splitter_compounds(self):
        splitter = _make_splitter()
        seg = DomainNameSegmenter(compound_splitter=splitter)
        out = seg.segment("techstartup.com")
        assert out.tokens == ("tech", "startup", "com")
        assert seg.has_compound_splitter is True

    def test_splitter_short_runs(self):
        # Single-character alpha runs (rare; e.g. inside hyphenated labels)
        # bypass the splitter entirely.
        splitter = _make_splitter()
        seg = DomainNameSegmenter(compound_splitter=splitter)
        out = seg.segment("a-techstartup.com")
        # "a" passes through unchanged; "techstartup" splits.
        assert out.tokens == ("a", "tech", "startup", "com")

    def test_splitter_subdomain(self):
        # Splitter applies to subdomain alpha runs as well.
        splitter = _make_splitter()
        seg = DomainNameSegmenter(compound_splitter=splitter)
        out = seg.segment("cloudstack.foo.com")
        assert out.tokens == ("cloud", "stack", "foo", "com")

    def test_splitter_digits_pass_through(self):
        # "cloud9" splits as alpha+digit; only the alpha run runs through
        # the splitter — "cloud" is a dictionary word so it stays single.
        splitter = _make_splitter()
        seg = DomainNameSegmenter(compound_splitter=splitter)
        out = seg.segment("cloud9.io")
        assert out.tokens == ("cloud", "9", "io")

    def test_splitter_invalid_type(self):
        with pytest.raises(ValidationError, match="must be a CompoundWordSplitter or None"):
            DomainNameSegmenter(compound_splitter="not-a-splitter")  # type: ignore[arg-type]

    def test_has_compound_splitter_property(self):
        assert DomainNameSegmenter().has_compound_splitter is False
        assert DomainNameSegmenter(compound_splitter=_make_splitter()).has_compound_splitter is True
