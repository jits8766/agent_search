"""Tests for the offline domain-name vectorization primitives.

Covers ``DomainNameSegmenter`` and ``BM25DocEncoder`` end-to-end so the
offline indexer can rely on shared invariants (TLD-aware token streams,
hashing-scheme parity with the query side, deterministic outputs).

Coverage matrix:

``DomainNameSegmenter.segment``:
- single_label_tld_returns_single_token            -> TestSegmenter::test_simple_com
- ascii_alpha_only_label                           -> TestSegmenter::test_ascii_alpha
- ascii_with_digits_splits                         -> TestSegmenter::test_ascii_with_digits
- hyphenated_label_splits                          -> TestSegmenter::test_hyphenated
- mixed_alpha_digit_hyphen                         -> TestSegmenter::test_mixed_alpha_digit_hyphen
- subdomain_preserved                              -> TestSegmenter::test_subdomain
- multi_subdomain_segmented                        -> TestSegmenter::test_multi_subdomain
- compound_tld_co_uk                               -> TestSegmenter::test_compound_co_uk
- compound_tld_unknown_falls_back                  -> TestSegmenter::test_compound_tld_unknown
- idn_alabel_decoded_to_ulabel                     -> TestSegmenter::test_idn_alabel_decoded
- idn_decode_failure_falls_back                    -> TestSegmenter::test_idn_decode_failure
- non_ascii_label_emits_single_token               -> TestSegmenter::test_non_ascii_label
- casefold_preserves_unicode                       -> TestSegmenter::test_casefold
- trailing_dot_stripped                            -> TestSegmenter::test_trailing_dot_stripped
- whitespace_stripped                              -> TestSegmenter::test_whitespace_stripped
- none_input_raises                                -> TestSegmenter::test_none_raises
- non_string_input_raises                          -> TestSegmenter::test_non_string_raises
- empty_input_raises                               -> TestSegmenter::test_empty_raises
- single_label_no_tld_raises                       -> TestSegmenter::test_no_tld_raises
- compound_tld_normalised                          -> TestSegmenter::test_compound_tld_normalised
- compound_tld_invalid_input_raises                -> TestSegmenter::test_compound_tld_invalid_raises
- tld_token_is_first_class                         -> TestSegmenter::test_tld_in_tokens
- token_order_preserved                            -> TestSegmenter::test_token_order
- compound_tld_count_diagnostic                    -> TestSegmenter::test_compound_tld_count

``BM25DocEncoder.encode`` / ``__call__``:
- indices_sorted_ascending                         -> TestBM25DocEncoder::test_sorted
- empty_tokens_returns_empty                       -> TestBM25DocEncoder::test_empty
- none_tokens_raises                               -> TestBM25DocEncoder::test_none_raises
- duplicate_tokens_aggregated                      -> TestBM25DocEncoder::test_duplicates_aggregated
- bucket_collision_sums                            -> TestBM25DocEncoder::test_bucket_collision_sums
- length_norm_disabled_when_b_zero                 -> TestBM25DocEncoder::test_length_norm_disabled
- length_norm_active_when_b_positive               -> TestBM25DocEncoder::test_length_norm_active
- k1_changes_saturation_curve                      -> TestBM25DocEncoder::test_k1_saturation
- vocab_size_invalid_raises                        -> TestBM25DocEncoder::test_vocab_size_invalid
- k1_invalid_raises                                -> TestBM25DocEncoder::test_k1_invalid
- b_out_of_range_raises                            -> TestBM25DocEncoder::test_b_out_of_range
- avg_doc_length_invalid_raises                    -> TestBM25DocEncoder::test_avg_doc_length_invalid
- non_string_tokens_skipped                        -> TestBM25DocEncoder::test_non_string_tokens_skipped
- hashing_matches_query_side                       -> TestBM25DocEncoder::test_hashing_matches_query_side
- sparse_vector_qdrant_call                        -> TestBM25DocEncoder::test_sparse_vector_call
"""
import math

import pytest

from semantic_search.config.models import BM25QueryEncoderConfig
from semantic_search.core.exceptions import RetrievalError, ValidationError
from semantic_search.retrieval.bm25_query_encoder import BM25QueryEncoder
from semantic_search.vectorization import BM25DocEncoder, DomainNameSegmenter, SegmentedDomain


class TestSegmenter:
    def test_simple_com(self):
        s = DomainNameSegmenter()
        out = s.segment("foo.com")
        assert out.registrable_label == "foo"
        assert out.subdomain_labels == ()
        assert out.tld == "com"
        assert out.tokens == ("foo", "com")
        assert out.had_idn_decode_failure is False
        assert out.original == "foo.com"

    def test_ascii_alpha(self):
        s = DomainNameSegmenter()
        out = s.segment("brandable.io")
        assert out.tokens == ("brandable", "io")

    def test_ascii_with_digits(self):
        s = DomainNameSegmenter()
        out = s.segment("cloud9.io")
        assert out.tokens == ("cloud", "9", "io")

    def test_hyphenated(self):
        s = DomainNameSegmenter()
        out = s.segment("foo-bar.com")
        assert out.tokens == ("foo", "bar", "com")
        assert out.registrable_label == "foo-bar"

    def test_mixed_alpha_digit_hyphen(self):
        s = DomainNameSegmenter()
        out = s.segment("web3-pro.io")
        assert out.tokens == ("web", "3", "pro", "io")

    def test_subdomain(self):
        s = DomainNameSegmenter()
        out = s.segment("shop.foo.com")
        assert out.subdomain_labels == ("shop",)
        assert out.registrable_label == "foo"
        assert out.tokens == ("shop", "foo", "com")

    def test_multi_subdomain(self):
        s = DomainNameSegmenter()
        out = s.segment("api-v2.shop.foo.io")
        assert out.subdomain_labels == ("api-v2", "shop")
        assert out.tokens == ("api", "v", "2", "shop", "foo", "io")

    def test_compound_co_uk(self):
        s = DomainNameSegmenter(known_compound_tlds=frozenset({"co.uk"}))
        out = s.segment("foo.co.uk")
        assert out.registrable_label == "foo"
        assert out.tld == "co.uk"
        # The TLD is emitted as one composite token.
        assert out.tokens == ("foo", "co.uk")

    def test_compound_tld_unknown(self):
        # Without registering co.uk, the segmenter falls back to single-component TLD.
        s = DomainNameSegmenter()
        out = s.segment("foo.co.uk")
        assert out.tld == "uk"
        assert out.registrable_label == "co"
        assert out.subdomain_labels == ("foo",)

    def test_idn_alabel_decoded(self):
        # Punycode for "中国.中国" -> "xn--fiqs8s.xn--fiqs8s"
        s = DomainNameSegmenter()
        out = s.segment("xn--fiqs8s.xn--fiqs8s")
        # IDN decode produces U-label form.
        assert out.tld == "中国"
        assert out.registrable_label == "中国"
        assert out.had_idn_decode_failure is False

    def test_idn_decode_failure(self):
        s = DomainNameSegmenter()
        # Malformed punycode → falls back to raw A-label, sets failure flag.
        out = s.segment("xn--invalid!.com")
        assert out.had_idn_decode_failure is True
        # Fallback: raw A-label preserved as the registrable label.
        assert "xn--invalid" in out.registrable_label or out.registrable_label.startswith("xn--")

    def test_non_ascii_label(self):
        s = DomainNameSegmenter()
        out = s.segment("café.com")
        # Non-ASCII label collapses to a single (NFKD-stripped) token.
        assert "com" in out.tokens
        # NFKD strips the accent: "café" -> "cafe".
        assert any(tok == "cafe" for tok in out.tokens)

    def test_casefold(self):
        s = DomainNameSegmenter()
        out = s.segment("FOO-BAR.COM")
        assert out.registrable_label == "foo-bar"
        assert out.tld == "com"
        assert out.tokens == ("foo", "bar", "com")

    def test_trailing_dot_stripped(self):
        s = DomainNameSegmenter()
        out = s.segment("foo.com.")
        assert out.tld == "com"

    def test_whitespace_stripped(self):
        s = DomainNameSegmenter()
        out = s.segment("  foo.com  ")
        assert out.registrable_label == "foo"

    def test_none_raises(self):
        s = DomainNameSegmenter()
        with pytest.raises(ValidationError):
            s.segment(None)

    def test_non_string_raises(self):
        s = DomainNameSegmenter()
        with pytest.raises(ValidationError):
            s.segment(123)  # type: ignore[arg-type]

    def test_empty_raises(self):
        s = DomainNameSegmenter()
        with pytest.raises(ValidationError):
            s.segment("")
        with pytest.raises(ValidationError):
            s.segment("   ")

    def test_no_tld_raises(self):
        s = DomainNameSegmenter()
        with pytest.raises(ValidationError):
            s.segment("foo")

    def test_compound_tld_normalised(self):
        # Leading dots and casing are normalised at construction.
        s = DomainNameSegmenter(known_compound_tlds=frozenset({".CO.UK"}))
        out = s.segment("foo.co.uk")
        assert out.tld == "co.uk"

    def test_compound_tld_invalid_raises(self):
        with pytest.raises(ValidationError):
            DomainNameSegmenter(known_compound_tlds={"valid", ""})  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            DomainNameSegmenter(known_compound_tlds="not-a-set")  # type: ignore[arg-type]

    def test_tld_in_tokens(self):
        s = DomainNameSegmenter()
        out = s.segment("foo.io")
        # TLD is the LAST token and a first-class member.
        assert out.tokens[-1] == "io"

    def test_token_order(self):
        # Subdomain tokens come first, then registrable, then TLD.
        s = DomainNameSegmenter()
        out = s.segment("api.foo-bar.com")
        assert out.tokens == ("api", "foo", "bar", "com")

    def test_compound_tld_count(self):
        s = DomainNameSegmenter(known_compound_tlds=frozenset({"co.uk", "com.br"}))
        assert s.compound_tld_count == 2


class TestBM25DocEncoder:
    def test_sorted(self):
        enc = BM25DocEncoder(vocab_size=1024)
        # Pick tokens that hash into different buckets — unlikely to collide
        # across this small vocab; the test asserts sorted order, not specific ids.
        indices, values = enc.encode(["foo", "bar", "baz", "qux"])
        assert indices == sorted(indices)
        assert len(indices) == len(values)
        assert all(v > 0 for v in values)

    def test_empty(self):
        enc = BM25DocEncoder(vocab_size=1024)
        assert enc.encode([]) == ([], [])

    def test_none_raises(self):
        enc = BM25DocEncoder(vocab_size=1024)
        with pytest.raises(ValidationError):
            enc.encode(None)  # type: ignore[arg-type]

    def test_duplicates_aggregated(self):
        enc = BM25DocEncoder(vocab_size=1024, k1=1.2, b=0.0)
        indices_one, values_one = enc.encode(["foo"])
        indices_two, values_two = enc.encode(["foo", "foo"])
        # Same bucket regardless of TF.
        assert indices_one == indices_two
        # Higher TF → higher saturated weight (but not 2x — it saturates).
        assert values_two[0] > values_one[0]
        assert values_two[0] < 2.0 * values_one[0]

    def test_bucket_collision_sums(self):
        # Force collision by using a tiny vocab where two distinct tokens
        # almost certainly land in the same bucket.
        enc = BM25DocEncoder(vocab_size=2)
        out_indices, out_values = enc.encode(["foo", "bar"])
        # Both tokens fit into 0 or 1 buckets; sum at most across 2 distinct
        # buckets, possibly into 1 if they collide.
        assert len(out_indices) <= 2
        assert sum(out_values) > 0

    def test_length_norm_disabled(self):
        enc = BM25DocEncoder(vocab_size=1024, b=0.0, avg_doc_length=10.0)
        assert enc._length_norm(5) == 1.0
        assert enc._length_norm(50) == 1.0

    def test_length_norm_active(self):
        enc = BM25DocEncoder(vocab_size=1024, b=0.5, avg_doc_length=10.0)
        # doc_length == avg → norm == 1
        assert enc._length_norm(10) == pytest.approx(1.0)
        # doc_length == 2 * avg → norm = 1 - 0.5 + 0.5*2 = 1.5
        assert enc._length_norm(20) == pytest.approx(1.5)
        # doc_length == 0.5 * avg → norm = 1 - 0.5 + 0.5*0.5 = 0.75
        assert enc._length_norm(5) == pytest.approx(0.75)

    def test_k1_saturation(self):
        # Higher k1 means TF saturates more slowly. At TF=10 the lower-k1
        # encoder should be closer to its asymptote than the higher-k1 one.
        low = BM25DocEncoder(vocab_size=1024, k1=0.5, b=0.0)
        high = BM25DocEncoder(vocab_size=1024, k1=4.0, b=0.0)
        _, vals_low = low.encode(["foo"] * 10)
        _, vals_high = high.encode(["foo"] * 10)
        # Both strictly positive.
        assert vals_low[0] > 0 and vals_high[0] > 0
        # (k1+1)*tf / (k1*norm + tf): with norm=1, low (k1=0.5) -> 1.5*10/10.5 = 1.43
        # high (k1=4.0) -> 5*10/14 = 3.57. So high yields a LARGER raw weight at same tf.
        assert vals_high[0] > vals_low[0]

    def test_vocab_size_invalid(self):
        with pytest.raises(ValidationError):
            BM25DocEncoder(vocab_size=0)
        with pytest.raises(ValidationError):
            BM25DocEncoder(vocab_size=-1)

    def test_k1_invalid(self):
        with pytest.raises(ValidationError):
            BM25DocEncoder(vocab_size=1024, k1=0.0)
        with pytest.raises(ValidationError):
            BM25DocEncoder(vocab_size=1024, k1=-1.0)

    def test_b_out_of_range(self):
        with pytest.raises(ValidationError):
            BM25DocEncoder(vocab_size=1024, b=-0.1)
        with pytest.raises(ValidationError):
            BM25DocEncoder(vocab_size=1024, b=1.1)

    def test_avg_doc_length_invalid(self):
        with pytest.raises(ValidationError):
            BM25DocEncoder(vocab_size=1024, avg_doc_length=0.0)
        with pytest.raises(ValidationError):
            BM25DocEncoder(vocab_size=1024, avg_doc_length=-5.0)

    def test_non_string_tokens_skipped(self):
        enc = BM25DocEncoder(vocab_size=1024)
        # Non-strings are silently skipped to keep the offline indexer
        # robust against minor schema drift in source documents.
        indices, _ = enc.encode(["foo", None, 42, "bar"])  # type: ignore[list-item]
        assert len(indices) == 2

    def test_hashing_matches_query_side(self):
        """The doc encoder MUST share hashing with the query encoder.

        Same vocab_size + same term -> same bucket id. This is the
        contract that makes the corpus + query sparse vectors line up
        inside Qdrant.
        """
        vocab_size = 8192
        # Build minimal query-side config — mirrors the doc encoder's vocab.
        query_cfg = BM25QueryEncoderConfig(
            vocab_size=vocab_size,
            min_term_length=1,
            max_terms=100,
            stopwords=[],
            synonyms=None,
        )
        q_enc = BM25QueryEncoder(query_cfg)
        d_enc = BM25DocEncoder(vocab_size=vocab_size)
        for term in ["foo", "bar", "cloud", "9", "io", "中国"]:
            assert q_enc._hash_term(term, vocab_size) == d_enc._hash_term(term, vocab_size)

    def test_sparse_vector_call(self):
        """``__call__`` returns a Qdrant SparseVector when the dep is present."""
        pytest.importorskip("qdrant_client")
        from qdrant_client import models as qm

        enc = BM25DocEncoder(vocab_size=1024)
        sv = enc(["foo", "bar"])
        assert isinstance(sv, qm.SparseVector)
        assert sv.indices == sorted(sv.indices)
        assert len(sv.indices) == len(sv.values)
