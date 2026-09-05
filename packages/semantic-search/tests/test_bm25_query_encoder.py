"""Tests for BM25QueryEncoder + BM25QueryEncoderConfig.

Coverage matrix for ``BM25QueryEncoderConfig``:
- vocab_size_below_floor_raises             -> test_config_vocab_size_below_floor_raises
- vocab_size_wrong_type_raises              -> test_config_vocab_size_wrong_type_raises
- min_term_length_below_floor_raises        -> test_config_min_term_length_below_floor_raises
- min_term_length_wrong_type_raises         -> test_config_min_term_length_wrong_type_raises
- max_terms_below_floor_raises              -> test_config_max_terms_below_floor_raises
- max_terms_wrong_type_raises               -> test_config_max_terms_wrong_type_raises
- stopwords_wrong_type_raises               -> test_config_stopwords_wrong_type_raises
- stopwords_entry_wrong_type_raises         -> test_config_stopwords_entry_wrong_type_raises
- from_dict_missing_required_raises         -> test_config_from_dict_missing_required_raises
- from_dict_happy_path                      -> test_config_from_dict_happy_path

Coverage matrix for ``BM25QueryEncoder``:
- constructor_wrong_type_raises             -> test_constructor_wrong_type_raises
- vocab_size_property                       -> test_vocab_size_property_matches_config
- tokenize_lowercases                       -> test_tokenize_lowercases_input
- tokenize_drops_short_tokens               -> test_tokenize_drops_short_tokens
- tokenize_drops_stopwords                  -> test_tokenize_drops_stopwords
- tokenize_caps_max_terms                   -> test_tokenize_caps_at_max_terms
- tokenize_extracts_alpha_num_runs          -> test_tokenize_extracts_alpha_num_runs
- tokenize_none_returns_empty               -> test_tokenize_none_input_returns_empty
- tokenize_empty_returns_empty              -> test_tokenize_empty_string_returns_empty
- tokenize_whitespace_only_returns_empty    -> test_tokenize_whitespace_only_returns_empty
- hash_term_in_range                        -> test_hash_term_within_vocab_range
- hash_term_deterministic                   -> test_hash_term_deterministic_across_calls
- hash_term_different_vocab_changes_id      -> test_hash_term_different_vocab_size_changes_bucket
- aggregate_sublinear_tf                    -> test_aggregate_uses_sublinear_tf_weighting
- aggregate_collision_sums                  -> test_aggregate_sums_weights_on_bucket_collision
- aggregate_returns_sorted_pairs            -> test_aggregate_returns_pairs_sorted_by_bucket_id
- call_returns_sparsevector_shape           -> test_call_returns_sparsevector_with_aligned_indices_values
- call_indices_unique_and_sorted            -> test_call_returned_indices_are_unique_and_sorted_ascending
- call_values_strictly_positive             -> test_call_returned_values_are_strictly_positive
- call_empty_text_returns_empty_sv          -> test_call_empty_text_returns_empty_sparsevector
- call_only_stopwords_returns_empty_sv      -> test_call_only_stopwords_returns_empty_sparsevector
- call_only_short_terms_returns_empty_sv    -> test_call_only_short_terms_returns_empty_sparsevector
- call_deterministic_across_invocations     -> test_call_deterministic_across_invocations
- call_repeated_term_higher_weight          -> test_call_repeated_term_yields_higher_weight_than_singleton
- call_does_not_mutate_input                -> test_call_does_not_mutate_input_string
- call_no_qdrant_raises_retrieval_error     -> test_call_without_qdrant_client_raises_retrieval_error
"""
import importlib
import math
import sys
from typing import List

import pytest

from semantic_search.config.models import BM25QueryEncoderConfig, QdrantHybridConfig
from semantic_search.core.exceptions import ConfigurationError, RetrievalError
from semantic_search.retrieval.bm25_query_encoder import BM25QueryEncoder


def _make_cfg( vocab_size: int = 4096, min_term_length: int = 2, max_terms: int = 16, stopwords: List[str] = None, ) -> BM25QueryEncoderConfig:
    """Test helper for valid encoder configs."""
    if stopwords is None:
        stopwords = ['the', 'a', 'an']
    return BM25QueryEncoderConfig(
        vocab_size=vocab_size,
        min_term_length=min_term_length,
        max_terms=max_terms,
        stopwords=stopwords,
    )


class TestBM25QueryEncoderConfig:
    """Input contract tests for BM25QueryEncoderConfig."""

    def test_config_vocab_size_below_floor_raises(self):
        with pytest.raises(ConfigurationError, match="vocab_size must be int >= 1024"):
            _make_cfg(vocab_size=512)

    def test_config_vocab_size_wrong_type_raises(self):
        with pytest.raises(ConfigurationError, match="vocab_size must be int >= 1024"):
            BM25QueryEncoderConfig(
                vocab_size="4096",
                min_term_length=2,
                max_terms=16,
                stopwords=[],
            )

    def test_config_min_term_length_below_floor_raises(self):
        with pytest.raises(ConfigurationError, match="min_term_length must be int >= 1"):
            _make_cfg(min_term_length=0)

    def test_config_min_term_length_wrong_type_raises(self):
        with pytest.raises(ConfigurationError, match="min_term_length must be int >= 1"):
            BM25QueryEncoderConfig(
                vocab_size=4096,
                min_term_length="2",
                max_terms=16,
                stopwords=[],
            )

    def test_config_max_terms_below_floor_raises(self):
        with pytest.raises(ConfigurationError, match="max_terms must be int >= 1"):
            _make_cfg(max_terms=0)

    def test_config_max_terms_wrong_type_raises(self):
        with pytest.raises(ConfigurationError, match="max_terms must be int >= 1"):
            BM25QueryEncoderConfig(
                vocab_size=4096,
                min_term_length=2,
                max_terms="16",
                stopwords=[],
            )

    def test_config_stopwords_wrong_type_raises(self):
        with pytest.raises(ConfigurationError, match="stopwords must be a list"):
            BM25QueryEncoderConfig(
                vocab_size=4096,
                min_term_length=2,
                max_terms=16,
                stopwords="the,a,an",
            )

    def test_config_stopwords_entry_wrong_type_raises(self):
        with pytest.raises(ConfigurationError, match="stopwords entries must be strings"):
            BM25QueryEncoderConfig(
                vocab_size=4096,
                min_term_length=2,
                max_terms=16,
                stopwords=['the', 5, 'an'],
            )

    def test_config_from_dict_missing_required_raises(self):
        with pytest.raises(ConfigurationError, match="vocab_size is required"):
            BM25QueryEncoderConfig.from_dict({
                'min_term_length': 2,
                'max_terms': 16,
                'stopwords': [],
            })

    def test_config_from_dict_happy_path(self):
        cfg = BM25QueryEncoderConfig.from_dict({
            'vocab_size': 4096,
            'min_term_length': 2,
            'max_terms': 16,
            'stopwords': ['x'],
        })
        assert cfg.vocab_size == 4096
        assert cfg.min_term_length == 2
        assert cfg.max_terms == 16
        assert cfg.stopwords == ['x']


class TestBM25QueryEncoderConstruction:
    """Constructor contract tests."""

    def test_constructor_wrong_type_raises(self):
        with pytest.raises(RetrievalError, match="requires a BM25QueryEncoderConfig"):
            BM25QueryEncoder(config={'vocab_size': 4096})

    def test_vocab_size_property_matches_config(self):
        enc = BM25QueryEncoder(_make_cfg(vocab_size=8192))
        assert enc.vocab_size == 8192


class TestBM25QueryEncoderTokenize:
    """Tokenization behavioural tests (private API used via _tokenize for math proof)."""

    def test_tokenize_lowercases_input(self):
        enc = BM25QueryEncoder(_make_cfg(min_term_length=1, stopwords=[]))
        assert enc._tokenize("HELLO World") == ["hello", "world"]

    def test_tokenize_drops_short_tokens(self):
        enc = BM25QueryEncoder(_make_cfg(min_term_length=3, stopwords=[]))
        assert enc._tokenize("a be cat dog") == ["cat", "dog"]

    def test_tokenize_drops_stopwords(self):
        enc = BM25QueryEncoder(_make_cfg(min_term_length=1, stopwords=['the', 'a']))
        assert enc._tokenize("the cat a dog") == ["cat", "dog"]

    def test_tokenize_caps_at_max_terms(self):
        enc = BM25QueryEncoder(_make_cfg(min_term_length=1, max_terms=3, stopwords=[]))
        assert enc._tokenize("aa bb cc dd ee ff") == ["aa", "bb", "cc"]

    def test_tokenize_extracts_alpha_num_runs(self):
        enc = BM25QueryEncoder(_make_cfg(min_term_length=1, stopwords=[]))
        assert enc._tokenize("foo-bar.baz_42 hi!") == ["foo", "bar", "baz", "42", "hi"]

    def test_tokenize_none_input_returns_empty(self):
        enc = BM25QueryEncoder(_make_cfg())
        assert enc._tokenize(None) == []

    def test_tokenize_empty_string_returns_empty(self):
        enc = BM25QueryEncoder(_make_cfg())
        assert enc._tokenize("") == []

    def test_tokenize_whitespace_only_returns_empty(self):
        enc = BM25QueryEncoder(_make_cfg())
        assert enc._tokenize("   \t\n  ") == []


class TestBM25QueryEncoderHashing:
    """Hashing scheme correctness."""

    def test_hash_term_within_vocab_range(self):
        for vocab in (1024, 4096, 65536, 262144):
            for term in ("alpha", "beta", "gamma-1", "long-token", "foo"):
                bid = BM25QueryEncoder._hash_term(term, vocab)
                assert 0 <= bid < vocab, f"bid {bid} out of [0,{vocab}) for {term}"

    def test_hash_term_deterministic_across_calls(self):
        term = "deterministic"
        vocab = 4096
        bids = {BM25QueryEncoder._hash_term(term, vocab) for _ in range(5)}
        assert len(bids) == 1

    def test_hash_term_different_vocab_size_changes_bucket(self):
        # Highly likely (not guaranteed) that the bucket changes between
        # vocabs of very different sizes. Test with multiple terms to ensure
        # at least one shifts (deterministic + repeatable across CI).
        terms = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
        bids_small = [BM25QueryEncoder._hash_term(t, 1024) for t in terms]
        bids_large = [BM25QueryEncoder._hash_term(t, 262144) for t in terms]
        differing = sum(1 for a, b in zip(bids_small, bids_large) if a != b)
        assert differing >= len(terms) - 1, f"only {differing}/{len(terms)} buckets differed"


class TestBM25QueryEncoderAggregate:
    """Term-weight aggregation math."""

    def test_aggregate_uses_sublinear_tf_weighting(self):
        # Single occurrence of one term -> weight = 1 + log(1 + 1) = 1 + log(2)
        enc = BM25QueryEncoder(_make_cfg())
        pairs = enc._aggregate_term_weights(["foo"])
        assert len(pairs) == 1
        bid, weight = pairs[0]
        expected = 1.0 + math.log(1.0 + 1.0)
        assert weight == pytest.approx(expected, rel=1e-9)

        # Two occurrences -> 1 + log(1+2) = 1 + log(3); MUST be > singleton weight
        # but NOT 2x (sub-linear).
        pairs2 = enc._aggregate_term_weights(["foo", "foo"])
        assert len(pairs2) == 1
        _, w2 = pairs2[0]
        expected2 = 1.0 + math.log(1.0 + 2.0)
        assert w2 == pytest.approx(expected2, rel=1e-9)
        assert expected < w2 < 2 * expected, "TF weighting MUST be sub-linear"

    def test_aggregate_sums_weights_on_bucket_collision(self):
        """Force a guaranteed sha1 collision via min-vocab pigeonhole.

        Config requires vocab_size >= 1024, so we can't use vocab=2 directly;
        instead, scan a wide candidate space at vocab_size=1024 to find a
        verified colliding pair (sha1 distributes uniformly enough that
        collisions are always present in 5000+ candidate words).
        """
        enc = BM25QueryEncoder(BM25QueryEncoderConfig(
            vocab_size=1024,
            min_term_length=1,
            max_terms=128,
            stopwords=[],
        ))
        # Generate a deterministic candidate pool large enough that collisions
        # are mathematically guaranteed (5000 candidates, 1024 buckets).
        candidates = [f"term{i}" for i in range(5000)]
        bucket_to_terms: dict = {}
        a, b = None, None
        for cand in candidates:
            bid = BM25QueryEncoder._hash_term(cand, 1024)
            if bid in bucket_to_terms:
                a, b = bucket_to_terms[bid], cand
                break
            bucket_to_terms[bid] = cand
        assert a is not None and b is not None, "sha1 distribution failed to produce a collision in 5000 candidates"
        assert a != b
        assert BM25QueryEncoder._hash_term(a, 1024) == BM25QueryEncoder._hash_term(b, 1024)
        pairs = enc._aggregate_term_weights([a, b])
        assert len(pairs) == 1, "colliding distinct terms must produce one merged bucket"
        _, summed = pairs[0]
        single_weight = 1.0 + math.log(1.0 + 1.0)
        assert summed == pytest.approx(2 * single_weight, rel=1e-9)

    def test_aggregate_returns_pairs_sorted_by_bucket_id(self):
        enc = BM25QueryEncoder(_make_cfg(min_term_length=1, stopwords=[], max_terms=64))
        # Many distinct terms -> nearly-uniform bucket distribution after sha1.
        tokens = [chr(ord('a') + i) + "x" for i in range(10)]
        pairs = enc._aggregate_term_weights(tokens)
        assert len(pairs) >= 1
        bids = [p[0] for p in pairs]
        assert bids == sorted(bids), "pairs MUST be sorted ascending by bucket id"


class TestBM25QueryEncoderCall:
    """End-to-end ``__call__`` contract tests (returns qm.SparseVector)."""

    def test_call_returns_sparsevector_with_aligned_indices_values(self):
        enc = BM25QueryEncoder(_make_cfg(min_term_length=1, stopwords=[]))
        sv = enc("hello world foo")
        assert hasattr(sv, 'indices')
        assert hasattr(sv, 'values')
        assert len(sv.indices) == len(sv.values)
        assert len(sv.indices) >= 1

    def test_call_returned_indices_are_unique_and_sorted_ascending(self):
        enc = BM25QueryEncoder(_make_cfg(min_term_length=1, stopwords=[]))
        sv = enc("a b c d e f g h i j k l m n o p")
        assert list(sv.indices) == sorted(set(sv.indices))

    def test_call_returned_values_are_strictly_positive(self):
        enc = BM25QueryEncoder(_make_cfg(min_term_length=1, stopwords=[]))
        sv = enc("alpha beta gamma")
        for v in sv.values:
            assert v > 0.0, f"BM25 query weights MUST be > 0; got {v}"

    def test_call_empty_text_returns_empty_sparsevector(self):
        enc = BM25QueryEncoder(_make_cfg())
        sv = enc("")
        assert list(sv.indices) == []
        assert list(sv.values) == []

    def test_call_only_stopwords_returns_empty_sparsevector(self):
        enc = BM25QueryEncoder(_make_cfg(min_term_length=1, stopwords=['the', 'and']))
        sv = enc("the and the and")
        assert list(sv.indices) == []
        assert list(sv.values) == []

    def test_call_only_short_terms_returns_empty_sparsevector(self):
        enc = BM25QueryEncoder(_make_cfg(min_term_length=4, stopwords=[]))
        sv = enc("a be cat dog")
        assert list(sv.indices) == []
        assert list(sv.values) == []

    def test_call_deterministic_across_invocations(self):
        enc = BM25QueryEncoder(_make_cfg(min_term_length=1, stopwords=[]))
        a = enc("foo bar baz")
        b = enc("foo bar baz")
        assert list(a.indices) == list(b.indices)
        assert list(a.values) == list(b.values)

    def test_call_repeated_term_yields_higher_weight_than_singleton(self):
        enc = BM25QueryEncoder(_make_cfg(min_term_length=1, stopwords=[]))
        single = enc("foo")
        doubled = enc("foo foo")
        # Same term -> same bucket id; doubled MUST out-weight single (sub-linear).
        assert list(single.indices) == list(doubled.indices)
        assert doubled.values[0] > single.values[0]

    def test_call_does_not_mutate_input_string(self):
        enc = BM25QueryEncoder(_make_cfg())
        text = "Foo Bar Baz"
        original = text
        enc(text)
        assert text == original

    def test_call_without_qdrant_client_raises_retrieval_error(self, monkeypatch):
        """When qdrant-client is unavailable, ``__call__`` MUST raise RetrievalError.

        Prod binds ``_qm`` at module import; patch it to None to simulate the
        missing optional dependency.
        """
        monkeypatch.setattr('semantic_search.retrieval.bm25_query_encoder._qm', None)
        enc = BM25QueryEncoder(_make_cfg(min_term_length=1, stopwords=[]))
        with pytest.raises(RetrievalError, match="requires qdrant-client"):
            enc("hello")


class TestQdrantHybridConfigBM25Wiring:
    """Cross-config invariants — bm25_query_encoder presence/absence rules."""

    def test_bm25_enabled_without_encoder_raises(self):
        with pytest.raises(ConfigurationError, match="bm25_query_encoder is required when bm25_enabled=true"):
            QdrantHybridConfig(
                enabled=True,
                fusion_strategy='rrf',
                bm25_enabled=True,
                bm25_vector_name='bm25',
                dense_vector_name='',
                prefetch_limit=100,
                kw_post_oversample_factor=3,
                kw_prefix_oversample_factor=6,
                bm25_query_encoder=None,
            )

    def test_bm25_disabled_with_encoder_is_allowed(self):
        cfg = QdrantHybridConfig(
            enabled=True,
            fusion_strategy='rrf',
            bm25_enabled=False,
            bm25_vector_name='',
            dense_vector_name='',
            prefetch_limit=100,
            kw_post_oversample_factor=3,
            kw_prefix_oversample_factor=6,
            bm25_query_encoder=_make_cfg(),
        )
        assert cfg.bm25_query_encoder is not None

    def test_bm25_query_encoder_wrong_type_raises(self):
        with pytest.raises(ConfigurationError, match="bm25_query_encoder must be a BM25QueryEncoderConfig"):
            QdrantHybridConfig(
                enabled=True,
                fusion_strategy='rrf',
                bm25_enabled=False,
                bm25_vector_name='',
                dense_vector_name='',
                prefetch_limit=100,
                kw_post_oversample_factor=3,
                kw_prefix_oversample_factor=6,
                bm25_query_encoder={'vocab_size': 4096},
            )

    def test_qdrant_hybrid_from_dict_round_trip_with_encoder(self):
        cfg = QdrantHybridConfig.from_dict({
            'enabled': True,
            'fusion_strategy': 'rrf',
            'bm25_enabled': True,
            'bm25_vector_name': 'bm25',
            'dense_vector_name': '',
            'prefetch_limit': 100,
            'kw_post_oversample_factor': 3,
            'kw_prefix_oversample_factor': 6,
            'bm25_query_encoder': {
                'vocab_size': 4096,
                'min_term_length': 2,
                'max_terms': 16,
                'stopwords': ['the'],
            },
        })
        assert cfg.bm25_enabled is True
        assert cfg.bm25_query_encoder is not None
        assert cfg.bm25_query_encoder.vocab_size == 4096

    def test_qdrant_hybrid_from_dict_omits_encoder_when_disabled(self):
        cfg = QdrantHybridConfig.from_dict({
            'enabled': True,
            'fusion_strategy': 'rrf',
            'bm25_enabled': False,
            'bm25_vector_name': '',
            'dense_vector_name': '',
            'prefetch_limit': 100,
            'kw_post_oversample_factor': 3,
            'kw_prefix_oversample_factor': 6,
        })
        assert cfg.bm25_enabled is False
        assert cfg.bm25_query_encoder is None
