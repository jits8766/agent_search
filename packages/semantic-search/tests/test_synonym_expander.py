"""Tests for synonym/abbreviation expansion for the BM25 leg.

Coverage matrix for ``SynonymExpansionConfig`` — every row → at least one parametrized case:
- enabled_wrong_type_raises             -> TestSynonymExpansionConfigContract::test_config_rejections[enabled_wrong_type]
- expansion_weight_below_floor_raises   -> ...::test_config_rejections[weight_below_floor]
- expansion_weight_above_ceiling_raises -> ...::test_config_rejections[weight_above_ceiling]
- expansion_weight_wrong_type_raises    -> ...::test_config_rejections[weight_wrong_type]
- max_synonyms_below_floor_raises       -> ...::test_config_rejections[max_synonyms_below_floor]
- max_synonyms_wrong_type_raises        -> ...::test_config_rejections[max_synonyms_wrong_type]
- max_tokens_below_floor_raises         -> ...::test_config_rejections[max_tokens_below_floor]
- max_tokens_wrong_type_raises          -> ...::test_config_rejections[max_tokens_wrong_type]
- synonym_map_wrong_type_raises         -> ...::test_config_rejections[map_wrong_type]
- synonym_map_key_wrong_type_raises     -> ...::test_config_rejections[map_key_wrong_type]
- synonym_map_key_empty_raises          -> ...::test_config_rejections[map_key_empty]
- synonym_map_value_wrong_type_raises   -> ...::test_config_rejections[map_value_wrong_type]
- synonym_map_value_entry_empty_raises  -> ...::test_config_rejections[map_value_entry_empty]
- from_dict_missing_required_raises     -> TestSynonymExpansionConfigContract::test_config_from_dict_missing_required_raises
- from_dict_happy_path                  -> TestSynonymExpansionConfigContract::test_config_from_dict_happy_path
- bm25_config_synonyms_optional         -> TestBM25QueryEncoderConfigSynonymsField::test_bm25_synonyms_field_states[default_none|accepts_cfg|wrong_type_raises]
- bm25_from_dict_with_synonyms          -> TestBM25QueryEncoderConfigSynonymsField::test_bm25_from_dict_loads_synonyms_block
- bm25_from_dict_without_synonyms       -> TestBM25QueryEncoderConfigSynonymsField::test_bm25_from_dict_omits_synonyms_block

Coverage matrix for ``build_bidirectional_map``:
- mirrors_forward_edges                 -> TestBuildBidirectionalMap::test_build_map_mirrors_forward_edges
- dedupes_overlap                       -> ...::test_build_map_dedupes_overlap
- sorts_values_lexicographically        -> ...::test_build_map_sorts_values_lex
- self_edge_dropped                     -> ...::test_build_map_drops_self_edge
- empty_value_list_skipped              -> ...::test_build_map_skips_empty_values
- lowercases_keys_and_values            -> ...::test_build_map_lowercases_inputs
- non_dict_raises                       -> TestBuildBidirectionalMap::test_build_map_input_rejections[non_dict]
- malformed_value_list_raises           -> ...::test_build_map_input_rejections[malformed_value_list]

Coverage matrix for ``SynonymExpander``:
- constructor_wrong_type_raises         -> TestSynonymExpander::test_constructor_rejects_invalid[none|wrong_type]
- enabled_property                      -> TestSynonymExpander::test_enabled_property[true|false]
- map_size_property                     -> TestSynonymExpander::test_map_size_property
- expand_disabled_pass_through          -> TestSynonymExpander::test_expand_passthrough[disabled|empty_map]
- expand_empty_map_pass_through         -> TestSynonymExpander::test_expand_passthrough[disabled|empty_map]
- expand_emits_synonyms_with_weight     -> TestSynonymExpander::test_expand_emits_synonyms_with_correct_weight
- expand_caps_per_token                 -> TestSynonymExpander::test_expand_caps_synonyms_per_token
- expand_caps_consulted_tokens          -> TestSynonymExpander::test_expand_caps_consulted_tokens
- expand_dedupes_against_originals      -> TestSynonymExpander::test_expand_dedupes_against_originals_anywhere
- expand_dedupes_against_prior_synonyms -> TestSynonymExpander::test_expand_dedupes_against_prior_synonyms
- expand_dedupes_duplicate_originals    -> TestSynonymExpander::test_expand_dedupes_duplicate_originals
- expand_preserves_first_seen_order     -> TestSynonymExpander::test_expand_preserves_first_seen_token_order
- expand_token_with_no_synonyms         -> TestSynonymExpander::test_expand_token_without_synonyms_emits_only_original
- expand_lowercases_inputs              -> TestSynonymExpander::test_expand_lowercases_input_tokens
- expand_skips_non_string_tokens        -> TestSynonymExpander::test_expand_skips_non_string_tokens
- expand_none_raises                    -> TestSynonymExpander::test_expand_none_input_raises
- expand_empty_returns_empty            -> TestSynonymExpander::test_expand_empty_input_returns_empty
- expand_deterministic                  -> TestSynonymExpander::test_expand_deterministic_across_invocations

Coverage matrix for ``BM25QueryEncoder`` synonym integration:
- expander_not_built_when_synonyms_none -> TestBM25EncoderSynonymWiring::test_encoder_expander_state[no_synonyms|disabled|enabled]
- expander_not_built_when_disabled      -> ...::test_encoder_expander_state[no_synonyms|disabled|enabled]
- expander_built_when_enabled           -> ...::test_encoder_expander_state[no_synonyms|disabled|enabled]
- aggregate_no_expander_legacy_path     -> TestBM25EncoderSynonymWiring::test_aggregate_no_expander_matches_legacy_math
- aggregate_with_expander_emits_extras  -> TestBM25EncoderSynonymWiring::test_aggregate_with_expander_emits_extra_buckets
- aggregate_synonym_weight_math         -> TestBM25EncoderSynonymWiring::test_aggregate_synonym_weight_is_multiplier_times_log2
- aggregate_original_weight_math        -> TestBM25EncoderSynonymWiring::test_aggregate_original_weight_uses_input_tf
- aggregate_synonym_collision_sums      -> TestBM25EncoderSynonymWiring::test_aggregate_collisions_sum_across_originals_and_synonyms
- call_with_synonyms_more_buckets       -> TestBM25EncoderCallWithSynonyms::test_call_with_synonyms_emits_more_buckets_than_without
- call_dedup_propagates_to_sparse       -> TestBM25EncoderCallWithSynonyms::test_call_dedup_in_expander_propagates_to_sparsevector
- call_synonym_disabled_same_as_pre_r13 -> TestBM25EncoderCallWithSynonyms::test_call_synonyms_disabled_byte_for_byte_same_as_pre_rec13
- call_with_synonyms_deterministic      -> TestBM25EncoderCallWithSynonyms::test_call_with_synonyms_deterministic
"""
import math
from typing import Dict, List

import pytest

from semantic_search.config.models import BM25QueryEncoderConfig, SynonymExpansionConfig
from semantic_search.core.exceptions import ConfigurationError, RetrievalError, ValidationError
from semantic_search.retrieval.bm25_query_encoder import BM25QueryEncoder
from semantic_search.retrieval.synonym_expander import SynonymExpander, build_bidirectional_map

from ._contract_helpers import matrix_param


# ---------- Helpers (single-file, kept local because they encode this file's
# domain defaults — not generic enough to live in a shared factory module). ----

def _syn(**overrides) -> SynonymExpansionConfig:
    base = dict(enabled=True, expansion_weight=0.5, max_synonyms_per_token=3,
                max_tokens_to_expand=10, synonym_map={'ai': ['artificial', 'intelligence']})
    base.update(overrides)
    return SynonymExpansionConfig(**base)


def _bm25(**overrides) -> BM25QueryEncoderConfig:
    base = dict(vocab_size=4096, min_term_length=2, max_terms=64, stopwords=['the', 'a'], synonyms=None)
    base.update(overrides)
    return BM25QueryEncoderConfig(**base)


# ---------- 1. SynonymExpansionConfig — input contract ----------------------

class TestSynonymExpansionConfigContract:
    @pytest.mark.parametrize('overrides,match', [
        matrix_param('enabled_wrong_type',         dict(enabled='true'),                           r'synonyms\.enabled must be a bool'),
        matrix_param('weight_below_floor',         dict(expansion_weight=0.0),                    r'expansion_weight must be in'),
        matrix_param('weight_above_ceiling',       dict(expansion_weight=1.5),                    r'expansion_weight must be in'),
        matrix_param('weight_wrong_type',          dict(expansion_weight='0.5'),                  r'expansion_weight must be a number'),
        matrix_param('max_synonyms_below_floor',   dict(max_synonyms_per_token=0),                r'max_synonyms_per_token must be int >= 1'),
        matrix_param('max_synonyms_wrong_type',    dict(max_synonyms_per_token='3'),              r'max_synonyms_per_token must be int >= 1'),
        matrix_param('max_tokens_below_floor',     dict(max_tokens_to_expand=0),                  r'max_tokens_to_expand must be int >= 1'),
        matrix_param('max_tokens_wrong_type',      dict(max_tokens_to_expand=2.5),                r'max_tokens_to_expand must be int >= 1'),
        matrix_param('map_wrong_type',             dict(synonym_map='ai:artificial'),             r'synonym_map must be a dict'),
        matrix_param('map_key_wrong_type',         dict(synonym_map={123: ['x']}),                r'synonym_map keys must be non-empty strings'),
        matrix_param('map_key_empty',              dict(synonym_map={'': ['x']}),                 r'synonym_map keys must be non-empty strings'),
        matrix_param('map_value_wrong_type',       dict(synonym_map={'ai': 'artificial'}),        r"synonym_map\['ai'\] must be a list"),
        matrix_param('map_value_entry_empty',      dict(synonym_map={'ai': ['']}),                r"synonym_map\['ai'\] entries must be non-empty strings"),
    ])
    def test_config_rejections(self, overrides, match):
        with pytest.raises(ConfigurationError, match=match):
            _syn(**overrides)

    def test_config_from_dict_missing_required_raises(self):
        with pytest.raises(ConfigurationError, match='is required'):
            SynonymExpansionConfig.from_dict({'enabled': True})

    def test_config_from_dict_happy_path(self):
        cfg = SynonymExpansionConfig.from_dict({
            'enabled': True, 'expansion_weight': 0.5,
            'max_synonyms_per_token': 3, 'max_tokens_to_expand': 10,
            'synonym_map': {'ai': ['artificial']},
        })
        assert (cfg.enabled, cfg.expansion_weight, cfg.max_synonyms_per_token,
                cfg.max_tokens_to_expand, cfg.synonym_map) == (
                    True, 0.5, 3, 10, {'ai': ['artificial']})


class TestBM25QueryEncoderConfigSynonymsField:
    @pytest.mark.parametrize('build_kwargs,expect_type,expect_raises_match', [
        matrix_param('default_none',      dict(),                            type(None),               None),
        matrix_param('accepts_cfg',       dict(synonyms=_syn()),             SynonymExpansionConfig,   None),
        matrix_param('wrong_type_raises', dict(synonyms={'enabled': True}),  None,                     r'synonyms must be a SynonymExpansionConfig or None'),
    ])
    def test_bm25_synonyms_field_states(self, build_kwargs, expect_type, expect_raises_match):
        if expect_raises_match is not None:
            with pytest.raises(ConfigurationError, match=expect_raises_match):
                _bm25(**build_kwargs)
            return
        cfg = _bm25(**build_kwargs)
        assert isinstance(cfg.synonyms, expect_type)

    def test_bm25_from_dict_loads_synonyms_block(self):
        cfg = BM25QueryEncoderConfig.from_dict({
            'vocab_size': 4096, 'min_term_length': 2, 'max_terms': 16, 'stopwords': [],
            'synonyms': {
                'enabled': True, 'expansion_weight': 0.5, 'max_synonyms_per_token': 3,
                'max_tokens_to_expand': 10, 'synonym_map': {'ai': ['artificial']},
            },
        })
        assert cfg.synonyms is not None
        assert cfg.synonyms.enabled is True

    def test_bm25_from_dict_omits_synonyms_block(self):
        cfg = BM25QueryEncoderConfig.from_dict({
            'vocab_size': 4096, 'min_term_length': 2, 'max_terms': 16, 'stopwords': [],
        })
        assert cfg.synonyms is None


# ---------- 2. build_bidirectional_map --------------------------------------

class TestBuildBidirectionalMap:
    def test_build_map_mirrors_forward_edges(self):
        m = build_bidirectional_map({'ai': ['artificial', 'intelligence']})
        assert m['ai'] == ('artificial', 'intelligence')
        assert m['artificial'] == ('ai',)
        assert m['intelligence'] == ('ai',)

    def test_build_map_dedupes_overlap(self):
        m = build_bidirectional_map({'ai': ['ml'], 'machinelearning': ['ml']})
        assert set(m['ml']) == {'ai', 'machinelearning'} and len(m['ml']) == 2

    def test_build_map_sorts_values_lex(self):
        assert build_bidirectional_map({'x': ['c', 'a', 'b']})['x'] == ('a', 'b', 'c')

    def test_build_map_drops_self_edge(self):
        m = build_bidirectional_map({'foo': ['foo', 'bar']})
        assert 'foo' not in m.get('foo', ()) and m['foo'] == ('bar',)

    def test_build_map_skips_empty_values(self):
        assert build_bidirectional_map({'foo': []}) == {}

    def test_build_map_lowercases_inputs(self):
        m = build_bidirectional_map({'AI': ['Artificial']})
        assert 'ai' in m and 'artificial' in m and m['ai'] == ('artificial',)

    @pytest.mark.parametrize('bad_input', [
        matrix_param('non_dict',             ['ai', 'artificial']),
        matrix_param('malformed_value_list', {'ai': 'artificial'}),
    ])
    def test_build_map_input_rejections(self, bad_input):
        with pytest.raises(ValidationError):
            build_bidirectional_map(bad_input)


# ---------- 3. SynonymExpander ----------------------------------------------

class TestSynonymExpander:
    @pytest.mark.parametrize('bad_cfg', [
        matrix_param('none',       None),
        matrix_param('wrong_type', {'enabled': True}),
    ])
    def test_constructor_rejects_invalid(self, bad_cfg):
        with pytest.raises(ValidationError):
            SynonymExpander(bad_cfg)  # type: ignore[arg-type]

    @pytest.mark.parametrize('enabled', [True, False])
    def test_enabled_property(self, enabled):
        assert SynonymExpander(_syn(enabled=enabled)).enabled is enabled

    def test_map_size_property(self):
        # 'ai' → [artificial, intelligence] yields 3 keys after bidir mirror
        assert SynonymExpander(_syn(synonym_map={'ai': ['artificial', 'intelligence']})).map_size == 3

    @pytest.mark.parametrize('cfg_overrides', [
        matrix_param('disabled',  dict(enabled=False)),
        matrix_param('empty_map', dict(enabled=True, synonym_map={})),
    ])
    def test_expand_passthrough(self, cfg_overrides):
        exp = SynonymExpander(_syn(**cfg_overrides))
        assert exp.expand(['ai', 'brand']) == [('ai', 1.0), ('brand', 1.0)]

    def test_expand_emits_synonyms_with_correct_weight(self):
        exp = SynonymExpander(_syn(expansion_weight=0.7, synonym_map={'ai': ['artificial']}))
        assert exp.expand(['ai']) == [('ai', 1.0), ('artificial', 0.7)]

    def test_expand_caps_synonyms_per_token(self):
        exp = SynonymExpander(_syn(max_synonyms_per_token=1,
                                    synonym_map={'ai': ['artificial', 'intelligence', 'machine']}))
        synonyms = [t for t, w in exp.expand(['ai']) if w != 1.0]
        # bidir map sorts lex → first synonym is 'artificial'
        assert synonyms == ['artificial']

    def test_expand_caps_consulted_tokens(self):
        # cap_tokens=1 → 'ai' consults synonyms, 'tld' is over the consultation cap.
        exp = SynonymExpander(_syn(max_tokens_to_expand=1,
                                    synonym_map={'ai': ['artificial'], 'tld': ['domain']}))
        out = exp.expand(['ai', 'tld'])
        assert [t for t, w in out if w != 1.0] == ['artificial']
        assert [t for t, w in out if w == 1.0] == ['ai', 'tld']

    @pytest.mark.parametrize('tokens', [
        matrix_param('synonym_after_original',  ['ai', 'artificial']),
        matrix_param('synonym_before_original', ['artificial', 'ai']),
    ])
    def test_expand_dedupes_against_originals_anywhere(self, tokens):
        exp = SynonymExpander(_syn(synonym_map={'ai': ['artificial']}))
        out = exp.expand(tokens)
        # Original-anywhere dedup: 'artificial' must NOT appear at expansion_weight.
        assert all(w == 1.0 for _, w in out)
        assert sorted(t for t, _ in out) == ['ai', 'artificial']

    def test_expand_dedupes_against_prior_synonyms(self):
        exp = SynonymExpander(_syn(synonym_map={'ai': ['ml'], 'machinelearning': ['ml']}))
        assert sum(1 for t, _ in exp.expand(['ai', 'machinelearning']) if t == 'ml') == 1

    def test_expand_dedupes_duplicate_originals(self):
        exp = SynonymExpander(_syn(synonym_map={'foo': ['bar']}))
        out = exp.expand(['foo', 'foo', 'foo'])
        assert sum(1 for t, _ in out if t == 'foo') == 1
        assert sum(1 for t, _ in out if t == 'bar') == 1

    def test_expand_preserves_first_seen_token_order(self):
        exp = SynonymExpander(_syn(synonym_map={'a': ['x'], 'b': ['y'], 'c': ['z']}))
        assert [t for t, w in exp.expand(['c', 'a', 'b']) if w == 1.0] == ['c', 'a', 'b']

    def test_expand_token_without_synonyms_emits_only_original(self):
        exp = SynonymExpander(_syn(synonym_map={'ai': ['artificial']}))
        assert exp.expand(['unknown']) == [('unknown', 1.0)]

    def test_expand_lowercases_input_tokens(self):
        exp = SynonymExpander(_syn(synonym_map={'ai': ['artificial']}))
        out = exp.expand(['AI'])
        assert ('ai', 1.0) in out and any(t == 'artificial' for t, _ in out)

    def test_expand_skips_non_string_tokens(self):
        exp = SynonymExpander(_syn())
        out = exp.expand(['ai', 123, 'brand'])  # type: ignore[list-item]
        originals = [t for t, w in out if w == 1.0]
        assert 'ai' in originals and 'brand' in originals and 123 not in originals

    def test_expand_none_input_raises(self):
        with pytest.raises(ValidationError):
            SynonymExpander(_syn()).expand(None)  # type: ignore[arg-type]

    def test_expand_empty_input_returns_empty(self):
        assert SynonymExpander(_syn()).expand([]) == []

    def test_expand_deterministic_across_invocations(self):
        exp = SynonymExpander(_syn(synonym_map={'ai': ['artificial', 'intelligence', 'machine']}))
        assert exp.expand(['ai', 'brand']) == exp.expand(['ai', 'brand'])


# ---------- 4. BM25QueryEncoder synonym integration -------------------------

class TestBM25EncoderSynonymWiring:
    @pytest.mark.parametrize('synonyms,expect_built', [
        matrix_param('no_synonyms', None,                False),
        matrix_param('disabled',    _syn(enabled=False), False),
        matrix_param('enabled',     _syn(enabled=True),  True),
    ])
    def test_encoder_expander_state(self, synonyms, expect_built):
        enc = BM25QueryEncoder(_bm25(synonyms=synonyms))
        if expect_built:
            assert enc._expander is not None and enc._expander.enabled is True
        else:
            assert enc._expander is None

    def test_aggregate_no_expander_matches_legacy_math(self):
        enc = BM25QueryEncoder(_bm25(synonyms=None))
        pairs = enc._aggregate_term_weights(['hello', 'world'])
        # 2 distinct tokens → 2 buckets, each with weight 1 + log(2)
        expected_weight = 1.0 + math.log(2.0)
        assert len(pairs) == 2 and all(abs(w - expected_weight) < 1e-9 for _, w in pairs)

    def test_aggregate_with_expander_emits_extra_buckets(self):
        enc_no = BM25QueryEncoder(_bm25(synonyms=None))
        enc_yes = BM25QueryEncoder(_bm25( synonyms=_syn(enabled=True, synonym_map={'hello': ['hi', 'greetings']})))
        pairs_no = enc_no._aggregate_term_weights(['hello', 'world'])
        pairs_yes = enc_yes._aggregate_term_weights(['hello', 'world'])
        # 'yes' adds at least one extra synonym-derived bucket (collision improbable on 4096 buckets)
        assert len(pairs_yes) >= len(pairs_no) + 1

    def test_aggregate_synonym_weight_is_multiplier_times_log2(self):
        enc = BM25QueryEncoder(_bm25(synonyms=_syn( enabled=True, expansion_weight=0.5, synonym_map={'ai': ['artificial']})))
        # 'ai' (TF=1) → weight = 1 + log(2). 'artificial' (synonym, TF=1) → 0.5 * (1 + log(2)).
        weights = sorted(w for _, w in enc._aggregate_term_weights(['ai']))
        assert len(weights) == 2
        assert abs(weights[0] - 0.5 * (1.0 + math.log(2.0))) < 1e-9
        assert abs(weights[1] - (1.0 + math.log(2.0))) < 1e-9

    def test_aggregate_original_weight_uses_input_tf(self):
        # When 'ai' appears 3×, original-side weight = 1 + log(1+3) = 1 + log(4).
        enc = BM25QueryEncoder(_bm25(synonyms=_syn( enabled=True, synonym_map={'ai': ['artificial']})))
        weights = sorted(w for _, w in enc._aggregate_term_weights(['ai', 'ai', 'ai']))
        assert len(weights) == 2
        assert abs(weights[0] - 0.5 * (1.0 + math.log(2.0))) < 1e-9
        assert abs(weights[1] - (1.0 + math.log(4.0))) < 1e-9

    def test_aggregate_collisions_sum_across_originals_and_synonyms(self):
        # Sum of all bucket weights regardless of layout = original + synonym.
        enc = BM25QueryEncoder(_bm25(vocab_size=1024, synonyms=_syn( enabled=True, expansion_weight=0.5, synonym_map={'ai': ['artificial']})))
        total = sum(w for _, w in enc._aggregate_term_weights(['ai']))
        expected = (1.0 + math.log(2.0)) + 0.5 * (1.0 + math.log(2.0))
        assert abs(total - expected) < 1e-9


class TestBM25EncoderCallWithSynonyms:
    def test_call_with_synonyms_emits_more_buckets_than_without(self):
        cfg_no = _bm25(min_term_length=1, stopwords=[], synonyms=None)
        cfg_yes = _bm25(min_term_length=1, stopwords=[],
                         synonyms=_syn(enabled=True, synonym_map={'foo': ['bar', 'baz']}))
        # 2 originals → 2 buckets in 'no'; 'yes' adds bar+baz → 4 buckets (collision improbable).
        assert len(BM25QueryEncoder(cfg_yes)("foo qux").indices) > len(BM25QueryEncoder(cfg_no)("foo qux").indices)

    def test_call_dedup_in_expander_propagates_to_sparsevector(self):
        # 'artificial' is both an original and a synonym of 'ai' — original-anywhere dedup
        # collapses the synonym so the SparseVector sees ONE bucket per surface form.
        cfg = _bm25(min_term_length=1, stopwords=[],
                     synonyms=_syn(enabled=True, synonym_map={'ai': ['artificial']}))
        sv = BM25QueryEncoder(cfg)("ai artificial")
        expected_weight = 1.0 + math.log(2.0)
        assert len(sv.indices) == 2 and all(abs(v - expected_weight) < 1e-9 for v in sv.values)

    def test_call_synonyms_disabled_byte_for_byte_same_as_pre_rec13(self):
        # backwards-compat: synonyms=None and synonyms.enabled=False produce identical SparseVectors.
        cfg_pre = _bm25(stopwords=['the'], synonyms=None)
        cfg_disabled = _bm25(stopwords=['the'],
                              synonyms=_syn(enabled=False, synonym_map={'ai': ['artificial']}))
        sv_pre = BM25QueryEncoder(cfg_pre)("ai brand the search")
        sv_dis = BM25QueryEncoder(cfg_disabled)("ai brand the search")
        assert list(sv_pre.indices) == list(sv_dis.indices)
        assert list(sv_pre.values) == list(sv_dis.values)

    def test_call_with_synonyms_deterministic(self):
        cfg = _bm25(synonyms=_syn(enabled=True,
                                   synonym_map={'ai': ['artificial', 'intelligence', 'machine']}))
        enc = BM25QueryEncoder(cfg)
        sv_a, sv_b = enc("ai brand"), enc("ai brand")
        assert list(sv_a.indices) == list(sv_b.indices) and list(sv_a.values) == list(sv_b.values)
