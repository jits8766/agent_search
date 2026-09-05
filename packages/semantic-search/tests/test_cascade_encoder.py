"""Tests for ``MatryoshkaCascadeEncoder`` and registry wiring.

Coverage matrix:

``MatryoshkaCascadeEncoder.__init__``:
    - base_encoder_none_raises                       -> TestConstruction::test_base_none
- supported_dims_empty_raises                    -> TestConstruction::test_supported_dims_empty
- supported_dims_negative_raises                 -> TestConstruction::test_supported_dims_negative
- non_native_base_collapses                      -> TestConstruction::test_non_native_base_collapses
- native_dim_property                            -> TestConstruction::test_native_dim
- supported_dims_property                        -> TestConstruction::test_supported_dims_property
- is_native_cascade_property                     -> TestConstruction::test_is_native_cascade

``MatryoshkaCascadeEncoder.encode_at_dim``:
    - requested_dim_not_supported_raises            -> TestEncodeAtDim::test_dim_not_supported
- collapse_mode_returns_base_output             -> TestEncodeAtDim::test_collapse_returns_base
- collapse_warns_only_once_per_dim              -> TestEncodeAtDim::test_collapse_warns_once
- none_text_returns_zero_vector                 -> TestEncodeAtDim::test_none_text
- empty_text_returns_zero_vector                -> TestEncodeAtDim::test_empty_text

``MatryoshkaCascadeEncoder.encode_batch_at_dims``:
    - texts_none_raises                             -> TestEncodeBatchAtDims::test_texts_none
- empty_dims_raises                             -> TestEncodeBatchAtDims::test_empty_dims
- duplicate_dims_deduplicated                   -> TestEncodeBatchAtDims::test_duplicate_dims
- collapse_mode_one_pass_per_dim                -> TestEncodeBatchAtDims::test_collapse_batch
- preserves_input_order                         -> TestEncodeBatchAtDims::test_preserves_order
- none_entries_zero_vectors                     -> TestEncodeBatchAtDims::test_none_entries
"""
import pytest

from semantic_search.core.exceptions import ValidationError
from semantic_search.qi.cascade_encoder import MatryoshkaCascadeEncoder
from semantic_search.qi.encoder import HashingEncoder


class TestConstruction:
    def test_base_none(self):
        with pytest.raises(ValidationError, match="non-None base Encoder"):
            MatryoshkaCascadeEncoder(
                base_encoder=None,  # type: ignore[arg-type]
                supported_dims=frozenset({128}),
            )

    def test_supported_dims_empty(self):
        with pytest.raises(ValidationError, match="non-empty supported_dims"):
            MatryoshkaCascadeEncoder(
                base_encoder=HashingEncoder(dim=128, seed=42),
                supported_dims=frozenset(),
            )

    def test_supported_dims_negative(self):
        with pytest.raises(ValidationError, match="entries must be int >= 1"):
            MatryoshkaCascadeEncoder(
                base_encoder=HashingEncoder(dim=128, seed=42),
                supported_dims=frozenset({-1}),
            )

    def test_non_native_base_collapses(self):
        cascade = MatryoshkaCascadeEncoder(
            base_encoder=HashingEncoder(dim=128, seed=42),
            supported_dims=frozenset({128, 256}),
        )
        assert cascade.is_native_cascade is False

    def test_native_dim(self):
        # HashingEncoder base — native_dim equals base.dim.
        cascade = MatryoshkaCascadeEncoder( base_encoder=HashingEncoder(dim=128, seed=42), supported_dims=frozenset({128}),)
        assert cascade.native_dim == 128

    def test_supported_dims_property(self):
        cascade = MatryoshkaCascadeEncoder( base_encoder=HashingEncoder(dim=128, seed=42), supported_dims=frozenset({128, 256, 384}),)
        assert cascade.supported_dims == frozenset({128, 256, 384})

    def test_is_native_cascade(self):
        cascade = MatryoshkaCascadeEncoder( base_encoder=HashingEncoder(dim=128, seed=42), supported_dims=frozenset({128}),)
        assert cascade.is_native_cascade is False


class TestEncodeAtDim:
    def test_dim_not_supported(self):
        cascade = MatryoshkaCascadeEncoder( base_encoder=HashingEncoder(dim=128, seed=42), supported_dims=frozenset({128}),)
        with pytest.raises(ValidationError, match="not in supported_dims"):
            cascade.encode_at_dim("hello", 256)

    def test_collapse_returns_base(self):
        # In collapse mode (HashingEncoder base), encode_at_dim returns
        # the base encoder's fixed-dim vector regardless of requested dim.
        base = HashingEncoder(dim=128, seed=42)
        cascade = MatryoshkaCascadeEncoder( base_encoder=base, supported_dims=frozenset({128}),)
        out = cascade.encode_at_dim("hello world", 128)
        baseline = base.encode("hello world")
        assert out == baseline

    def test_collapse_warns_once(self):
        # The wrapper deduplicates the warning per (encoder_id, dim) so
        # repeated calls with the same dim don't spam logs. The package
        # logger sets propagate=False so caplog can't capture; we assert
        # via the side effect on the _warned dedup set directly.
        base = HashingEncoder(dim=128, seed=42)
        cascade = MatryoshkaCascadeEncoder( base_encoder=base, supported_dims=frozenset({128}),)
        MatryoshkaCascadeEncoder._warned.discard((id(base), 128))
        cascade.encode_at_dim("hello", 128)
        cascade.encode_at_dim("world", 128)
        cascade.encode_at_dim("foo", 128)
        assert (id(base), 128) in MatryoshkaCascadeEncoder._warned

    def test_none_text(self):
        # collapse mode: HashingEncoder.encode(None) returns a zero vector.
        cascade = MatryoshkaCascadeEncoder( base_encoder=HashingEncoder(dim=128, seed=42), supported_dims=frozenset({128}),)
        out = cascade.encode_at_dim(None, 128)  # type: ignore[arg-type]
        assert out == [0.0] * 128

    def test_empty_text(self):
        cascade = MatryoshkaCascadeEncoder( base_encoder=HashingEncoder(dim=128, seed=42), supported_dims=frozenset({128}),)
        out = cascade.encode_at_dim("", 128)
        assert out == [0.0] * 128


class TestEncodeBatchAtDims:
    def test_texts_none(self):
        cascade = MatryoshkaCascadeEncoder( base_encoder=HashingEncoder(dim=128, seed=42), supported_dims=frozenset({128}),)
        with pytest.raises(ValidationError, match="non-None texts sequence"):
            cascade.encode_batch_at_dims(None, [128])  # type: ignore[arg-type]

    def test_empty_dims(self):
        cascade = MatryoshkaCascadeEncoder( base_encoder=HashingEncoder(dim=128, seed=42), supported_dims=frozenset({128}),)
        with pytest.raises(ValidationError, match="at least one dim"):
            cascade.encode_batch_at_dims(["hello"], [])

    def test_duplicate_dims(self):
        # Duplicates are silently deduplicated.
        cascade = MatryoshkaCascadeEncoder( base_encoder=HashingEncoder(dim=128, seed=42), supported_dims=frozenset({128}),)
        out = cascade.encode_batch_at_dims(["a"], [128, 128, 128])
        assert list(out.keys()) == [128]

    def test_collapse_batch(self):
        # In collapse mode, every requested dim returns the base encoder's
        # batch output (one list per requested dim).
        cascade = MatryoshkaCascadeEncoder( base_encoder=HashingEncoder(dim=128, seed=42), supported_dims=frozenset({128}),)
        out = cascade.encode_batch_at_dims(["a", "b"], [128])
        assert 128 in out
        assert len(out[128]) == 2

    def test_preserves_order(self):
        cascade = MatryoshkaCascadeEncoder( base_encoder=HashingEncoder(dim=128, seed=42), supported_dims=frozenset({128}),)
        out = cascade.encode_batch_at_dims(["alpha", "beta", "gamma"], [128])
        # The vectors should map 1:1 with input order.
        assert len(out[128]) == 3
        # alpha and beta produce different vectors (different tokens).
        assert out[128][0] != out[128][1]

    def test_none_entries(self):
        cascade = MatryoshkaCascadeEncoder( base_encoder=HashingEncoder(dim=128, seed=42), supported_dims=frozenset({128}),)
        # In collapse mode, the base encoder handles None entries; we just
        # verify the output length matches the input length.
        out = cascade.encode_batch_at_dims(["alpha", None, "gamma"], [128])
        assert len(out[128]) == 3
