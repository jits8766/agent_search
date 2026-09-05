"""Tests for CharNgramSparseEncoder and QdrantNgramConfig.

Coverage matrix for ``CharNgramSparseEncoder.__init__``:
- vocab_size_below_floor_raises       -> test_init_invalid_params_raise[vocab]
- min_n_below_floor_raises            -> test_init_invalid_params_raise[min_n]
- max_n_below_min_n_raises            -> test_init_invalid_params_raise[max_lt_min]
- valid_params_construct              -> test_init_valid_params_does_not_raise

Coverage matrix for ``CharNgramSparseEncoder.encode``:
- empty_string_returns_empty          -> test_encode_empty_and_none_return_empty_lists
- indices_values_length_aligned       -> test_encode_normal_input_shape_and_bounds
- indices_strictly_ascending_unique   -> test_encode_normal_input_shape_and_bounds
- determinism                         -> test_encode_deterministic_across_calls
- fuzzy_recall_overlap_near_miss      -> test_encode_fuzzy_recall_overlap_near_miss
- fuzzy_recall_far_less_overlap       -> test_encode_fuzzy_recall_far_less_overlap_than_near_miss

Coverage matrix for ``CharNgramSparseEncoder.__call__``:
- call_matches_encode_lists           -> test_call_sparse_vector_matches_encode

Coverage matrix for ``QdrantNgramConfig``:
- from_dict_round_trip                -> test_ngram_config_from_dict_round_trip
- enabled_empty_vector_name_raises    -> test_ngram_config_enabled_validation_raises[empty_name]
- enabled_vocab_zero_raises           -> test_ngram_config_enabled_validation_raises[vocab_zero]
- enabled_max_lt_min_raises           -> test_ngram_config_enabled_validation_raises[max_lt_min]
- disabled_empty_name_ok              -> test_ngram_config_disabled_empty_vector_name_ok

Coverage matrix for ``QdrantHybridConfig.from_dict`` ngram parsing:
- absent_ngram_key_is_none            -> test_hybrid_from_dict_without_ngram_key
- present_ngram_key_parsed            -> test_hybrid_from_dict_with_ngram_subdict

Coverage matrix for base.yaml defaults:
- base_config_ngram_enabled           -> test_base_config_ngram_enabled
"""
import pytest

from semantic_search.config.loader import load_config
from semantic_search.config.models import AgentSearchConfig, QdrantHybridConfig, QdrantNgramConfig
from semantic_search.core.exceptions import ConfigurationError, ValidationError
from semantic_search.retrieval.char_ngram_sparse_encoder import CharNgramSparseEncoder

_VALID_NGRAM_DICT = {
    'enabled': True,
    'vector_name': 'ngram',
    'vocab_size': 262144,
    'min_n': 3,
    'max_n': 4,
}

_MINIMAL_HYBRID_DICT = {
    'enabled': True,
    'fusion_strategy': 'rrf',
    'bm25_enabled': False,
    'bm25_vector_name': 'sparse',
    'dense_vector_name': 'dense',
    'prefetch_limit': 50,
    'kw_post_oversample_factor': 3,
    'kw_prefix_oversample_factor': 6,
}


def _overlap(indices_a, indices_b):
    """Count of shared bucket ids between two encode index lists."""
    return len(set(indices_a) & set(indices_b))


@pytest.fixture
def fuzzy_encoder():
    """Encoder sized for fuzzy near-miss recall overlap tests."""
    return CharNgramSparseEncoder(vocab_size=262144, min_n=3, max_n=4)


@pytest.mark.parametrize(
    "vocab_size,min_n,max_n,match_substr",
    [
        (0, 3, 4, "vocab_size must be >= 1"),
        (4096, 0, 4, "min_n must be >= 1"),
        (4096, 4, 3, "max_n must be >= min_n"),
    ],
)
def test_init_invalid_params_raise(vocab_size, min_n, max_n, match_substr):
    """Invalid constructor parameters raise ValidationError."""
    with pytest.raises(ValidationError, match=match_substr):
        CharNgramSparseEncoder(vocab_size=vocab_size, min_n=min_n, max_n=max_n)


def test_init_valid_params_does_not_raise():
    """Valid constructor parameters succeed."""
    enc = CharNgramSparseEncoder(vocab_size=4096, min_n=3, max_n=4)
    assert enc.vocab_size == 4096


@pytest.mark.parametrize("text", ["", None])
def test_encode_empty_and_none_return_empty_lists(text):
    """Empty or missing text yields empty parallel lists."""
    enc = CharNgramSparseEncoder(vocab_size=4096, min_n=3, max_n=4)
    assert enc.encode(text) == ([], [])


def test_encode_normal_input_shape_and_bounds():
    """Normal text yields sorted unique in-range indices and positive weights."""
    enc = CharNgramSparseEncoder(vocab_size=4096, min_n=3, max_n=4)
    indices, values = enc.encode("high rentals domain")
    assert len(indices) == len(values)
    assert indices == sorted(indices)
    assert len(set(indices)) == len(indices)
    assert all(0 <= i < enc.vocab_size for i in indices)
    assert all(v > 0.0 for v in values)


def test_encode_deterministic_across_calls():
    """Repeated encode on the same text returns identical lists."""
    enc = CharNgramSparseEncoder(vocab_size=4096, min_n=3, max_n=4)
    text = "hi-rentals.io"
    first = enc.encode(text)
    second = enc.encode(text)
    assert first == second


def test_encode_fuzzy_recall_overlap_near_miss(fuzzy_encoder):
    """Near-miss strings share at least one hashed n-gram bucket."""
    a_idx, _ = fuzzy_encoder.encode("high rentals")
    b_idx, _ = fuzzy_encoder.encode("hi-rentals.io")
    c_idx, _ = fuzzy_encoder.encode("rent.high")
    assert _overlap(a_idx, b_idx) > 0
    assert _overlap(a_idx, c_idx) > 0


def test_encode_fuzzy_recall_far_less_overlap_than_near_miss(fuzzy_encoder):
    """Unrelated text overlaps far fewer buckets than a fuzzy near-match."""
    a_idx, _ = fuzzy_encoder.encode("high rentals")
    near_idx, _ = fuzzy_encoder.encode("hi-rentals.io")
    far_idx, _ = fuzzy_encoder.encode("zzzzz qqqqq")
    assert _overlap(a_idx, near_idx) > _overlap(a_idx, far_idx)


def test_call_sparse_vector_matches_encode():
    """__call__ SparseVector lists match encode() output."""
    enc = CharNgramSparseEncoder(vocab_size=4096, min_n=3, max_n=4)
    text = "rent.high example"
    indices, values = enc.encode(text)
    sv = enc(text)
    assert list(sv.indices) == indices
    assert list(sv.values) == values


def test_ngram_config_from_dict_round_trip():
    """from_dict preserves all QdrantNgramConfig fields."""
    cfg = QdrantNgramConfig.from_dict(_VALID_NGRAM_DICT)
    assert cfg.enabled is True
    assert cfg.vector_name == "ngram"
    assert cfg.vocab_size == 262144
    assert cfg.min_n == 3
    assert cfg.max_n == 4


@pytest.mark.parametrize(
    "payload,match_substr",
    [
        ({**_VALID_NGRAM_DICT, 'vector_name': ''}, "vector_name must be non-empty"),
        ({**_VALID_NGRAM_DICT, 'vocab_size': 0}, "vocab_size must be int >= 1"),
        ({**_VALID_NGRAM_DICT, 'min_n': 4, 'max_n': 3}, "max_n must be int >= min_n"),
    ],
)
def test_ngram_config_enabled_validation_raises(payload, match_substr):
    """Enabled n-gram config rejects invalid vector_name and numeric bounds."""
    with pytest.raises(ConfigurationError, match=match_substr):
        QdrantNgramConfig.from_dict(payload)


def test_ngram_config_disabled_empty_vector_name_ok():
    """Disabled n-gram config skips enabled-only validation."""
    cfg = QdrantNgramConfig(enabled=False, vector_name='', vocab_size=0, min_n=0, max_n=0)
    assert cfg.enabled is False
    assert cfg.vector_name == ''


def test_hybrid_from_dict_without_ngram_key():
    """Hybrid config without ngram sub-dict leaves ngram None."""
    cfg = QdrantHybridConfig.from_dict(_MINIMAL_HYBRID_DICT)
    assert cfg.ngram is None


def test_hybrid_from_dict_with_ngram_subdict():
    """Hybrid config parses optional ngram sub-dict into QdrantNgramConfig."""
    hybrid_dict = {**_MINIMAL_HYBRID_DICT, 'ngram': _VALID_NGRAM_DICT}
    cfg = QdrantHybridConfig.from_dict(hybrid_dict)
    assert isinstance(cfg.ngram, QdrantNgramConfig)
    assert cfg.ngram.enabled is True
    assert cfg.ngram.vector_name == "ngram"
    assert cfg.ngram.vocab_size == 262144
    assert cfg.ngram.min_n == 3
    assert cfg.ngram.max_n == 4


def test_base_config_ngram_enabled():
    """base.yaml must ship with retrieval.qdrant.hybrid.ngram.enabled=true.

    Regression guard: catches any accidental edit that disables the ngram
    fuzzy-recall leg in the default config. A disabled ngram in base.yaml
    means reindex silently omits the ngram sparse vector and fuzzy recall
    is dead in all environments that inherit the default.
    """
    config_dict = load_config()
    cfg = AgentSearchConfig.from_dict(config_dict)
    ngram = cfg.retrieval.qdrant.hybrid.ngram
    assert ngram is not None, "retrieval.qdrant.hybrid.ngram block must be present in base.yaml"
    assert ngram.enabled is True, (
        "retrieval.qdrant.hybrid.ngram.enabled must be true in base.yaml — "
        "fuzzy-recall leg is permanently off when this is false"
    )
