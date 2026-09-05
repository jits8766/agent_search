"""Tests for `LexicalJaccardDiversifier` + `NoOpDiversifier` + their config.

Coverage:
    * `LexicalDiversityConfig.__post_init__` + `from_dict` validators.
* `DiversityConfig.__post_init__` + `from_dict` validators.
* `LexicalJaccardDiversifier` math with hand-calculated MMR scores.
* Edge cases: empty items, output_n=0, all-empty payloads, missing payload
  fields, list/non-string payload coercion, top_n > len(items).
* `NoOpDiversifier` identity behaviour + arg validation.
"""
import pytest

from semantic_search.config.models import DiversityConfig, LexicalDiversityConfig
from semantic_search.contracts import RankedItem
from semantic_search.core.exceptions import ConfigurationError, DiversityError, ValidationError
from semantic_search.retrieval.lexical_jaccard_diversifier import LexicalJaccardDiversifier, NoOpDiversifier
from ._contract_helpers import matrix_param


def _lex_kw(**overrides):
    base = dict(lambda_relevance=0.7, payload_fields=['title', 'description'],
                min_term_length=2, max_terms=64, stopwords=['the', 'a'])
    base.update(overrides)
    return base


def _div_kw(**overrides):
    base = dict(enabled=True, backend='lexical_jaccard_mmr', top_n=50,
                output_n=10, latency_budget_ms=15.0,
                lexical=LexicalDiversityConfig(**_lex_kw()))
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# LexicalDiversityConfig
# ---------------------------------------------------------------------------
class TestLexicalDiversityConfigValidation:
    def test_valid_construction(self):
        LexicalDiversityConfig(**_lex_kw())

    @pytest.mark.parametrize('lambda_relevance', [0.0, 1.0])
    def test_lambda_at_boundaries_accepted(self, lambda_relevance):
        LexicalDiversityConfig(**_lex_kw(lambda_relevance=lambda_relevance))

    @pytest.mark.parametrize('overrides,match', [
        matrix_param('lambda_below_zero',         dict(lambda_relevance=-0.1),         r'lambda_relevance'),
        matrix_param('lambda_above_one',          dict(lambda_relevance=1.1),          r'lambda_relevance'),
        matrix_param('payload_fields_empty',      dict(payload_fields=[]),             r'payload_fields'),
        matrix_param('payload_fields_int_entry',  dict(payload_fields=['title', 42]),  r'payload_fields entries'),
        matrix_param('payload_fields_empty_str',  dict(payload_fields=['title', '']),  r'payload_fields entries'),
        matrix_param('min_term_length_zero',      dict(min_term_length=0),             r'min_term_length'),
        matrix_param('max_terms_zero',            dict(max_terms=0),                   r'max_terms'),
        matrix_param('stopwords_not_list',        dict(stopwords='the'),               r'stopwords must be a list'),
        matrix_param('stopwords_int_entry',       dict(stopwords=['the', 99]),         r'stopwords entries'),
    ])
    def test_rejections(self, overrides, match):
        with pytest.raises(ConfigurationError, match=match):
            LexicalDiversityConfig(**_lex_kw(**overrides))


class TestLexicalDiversityConfigFromDict:
    def test_from_dict_happy_path(self):
        d = LexicalDiversityConfig.from_dict({
            'lambda_relevance': 0.5, 'payload_fields': ['title'],
            'min_term_length': 3, 'max_terms': 32, 'stopwords': ['x'],
        })
        assert (d.lambda_relevance, d.payload_fields, d.min_term_length, d.max_terms, d.stopwords) == \
               (0.5, ['title'], 3, 32, ['x'])

    @pytest.mark.parametrize('missing_key', ['lambda_relevance', 'payload_fields'])
    def test_from_dict_missing_required_rejected(self, missing_key):
        d = {'lambda_relevance': 0.5, 'payload_fields': ['t'],
             'min_term_length': 1, 'max_terms': 10, 'stopwords': []}
        d.pop(missing_key)
        with pytest.raises(ConfigurationError):
            LexicalDiversityConfig.from_dict(d)


# ---------------------------------------------------------------------------
# DiversityConfig
# ---------------------------------------------------------------------------
class TestDiversityConfigValidation:
    def test_valid_construction(self):
        DiversityConfig(**_div_kw())

    def test_output_n_equal_top_n_accepted(self):
        DiversityConfig(**_div_kw(top_n=10, output_n=10))

    @pytest.mark.parametrize('overrides,match', [
        matrix_param('enabled_not_bool',         dict(enabled=1),                                r'enabled must be a bool'),
        matrix_param('unknown_backend',          dict(backend='mystery'),                        r'backend must be one of'),
        matrix_param('top_n_zero',               dict(top_n=0),                                  r'top_n must be >= 1'),
        matrix_param('output_n_zero',            dict(output_n=0),                               r'output_n must be >= 1'),
        matrix_param('output_n_gt_top_n',        dict(top_n=10, output_n=20),                    r'output_n must be <= retrieval.diversity.top_n'),
        matrix_param('latency_budget_zero',      dict(latency_budget_ms=0.0),                    r'latency_budget_ms must be > 0'),
        matrix_param('latency_budget_negative',  dict(latency_budget_ms=-1.0),                   r'latency_budget_ms must be > 0'),
        matrix_param('lexical_required',         dict(lexical=None),                             r'lexical is required'),
        matrix_param('lexical_wrong_type',       dict(lexical='not_a_cfg'),                      r'must be a LexicalDiversityConfig'),
    ])
    def test_rejections(self, overrides, match):
        with pytest.raises(ConfigurationError, match=match):
            DiversityConfig(**_div_kw(**overrides))

    @pytest.mark.parametrize('backend', ['noop', 'lexical_jaccard_mmr'])
    def test_disabled_does_not_require_lexical(self, backend):
        DiversityConfig(enabled=False, backend=backend, top_n=50, output_n=50,
                        latency_budget_ms=15.0, lexical=None)


class TestDiversityConfigFromDict:
    def test_from_dict_with_lexical(self):
        d = DiversityConfig.from_dict({
            'enabled': True, 'backend': 'lexical_jaccard_mmr',
            'top_n': 30, 'output_n': 10, 'latency_budget_ms': 20.0,
            'lexical': {'lambda_relevance': 0.7, 'payload_fields': ['title'],
                        'min_term_length': 2, 'max_terms': 32, 'stopwords': []},
        })
        assert (d.enabled, d.backend, d.top_n, d.output_n, d.latency_budget_ms) == \
               (True, 'lexical_jaccard_mmr', 30, 10, 20.0)
        assert d.lexical is not None and d.lexical.lambda_relevance == 0.7

    def test_from_dict_disabled_no_lexical(self):
        d = DiversityConfig.from_dict({
            'enabled': False, 'backend': 'noop', 'top_n': 50,
            'output_n': 50, 'latency_budget_ms': 15.0,
        })
        assert d.enabled is False and d.lexical is None


# ---------------------------------------------------------------------------
# LexicalJaccardDiversifier
# ---------------------------------------------------------------------------
def _make_item(item_id: str, payload: dict) -> RankedItem:
    return RankedItem(item_id=item_id, fused_score=1.0, contributing_sources=['vector'],
                      payload=payload, sub_intent_ids=[])


def _lex_cfg(lambda_relevance: float = 0.7, fields=('title', 'tags')):
    return LexicalDiversityConfig(lambda_relevance=lambda_relevance,
                                   payload_fields=list(fields),
                                   min_term_length=1, max_terms=64, stopwords=[])


class TestLexicalJaccardDiversifierConstruction:
    def test_requires_lexical_diversity_config(self):
        with pytest.raises(DiversityError, match='LexicalDiversityConfig'):
            LexicalJaccardDiversifier(config='not_a_cfg')

    def test_name_is_stable(self):
        assert LexicalJaccardDiversifier(_lex_cfg()).name == 'lexical_jaccard_mmr'


class TestLexicalJaccardDiversifierMath:
    """Hand-calculated MMR scores for known token sets + lambda values."""

    def test_three_items_two_overlap_lambda_half(self):
        # Doc tokens: 0={a,b}, 1={c,d}, 2={a,b,e}
        # Step 1: pick item 0 (base=1.0, mmr = 0.5*1.0 = 0.50)
        # Step 2: sims to {0}:
        #   sim(1,0) = 0/4 = 0; sim(2,0) = 2/3 = 0.666...
        #   mmr(1) = 0.5*0.5 - 0.5*0 = 0.25
        #   mmr(2) = 0.5*(1/3) - 0.5*(2/3) = -0.1666 ⇒ pick 1.
        # Step 3: pick 2. penalty = max(2/3, 0) = 2/3 ⇒ mmr = -1/6
        items = [_make_item('0', {'title': 'a b'}), _make_item('1', {'title': 'c d'}),
                 _make_item('2', {'title': 'a b e'})]
        out = LexicalJaccardDiversifier(_lex_cfg(lambda_relevance=0.5)).diversify( query='', items=items, top_n=3, output_n=3)
        assert [r.item.item_id for r in out] == ['0', '1', '2']
        assert out[0].mmr_score == pytest.approx(0.5)
        assert out[0].diversity_penalty == pytest.approx(0.0)
        assert out[1].mmr_score == pytest.approx(0.25)
        assert out[2].mmr_score == pytest.approx(-1.0 / 6.0)
        assert out[2].diversity_penalty == pytest.approx(2.0 / 3.0)

    def test_lambda_one_pure_relevance_keeps_input_order(self):
        items = [_make_item('0', {'title': 'a'}), _make_item('1', {'title': 'a b'}),
                 _make_item('2', {'title': 'c'})]
        out = LexicalJaccardDiversifier(_lex_cfg(lambda_relevance=1.0)).diversify( query='', items=items, top_n=3, output_n=3)
        assert [r.item.item_id for r in out] == ['0', '1', '2']
        for rank, r in enumerate(out):
            assert r.mmr_score == pytest.approx(1.0 / (1.0 + rank))

    def test_lambda_zero_pure_novelty_promotes_diverse_item(self):
        # 0={a}, 1={a,b}, 2={c}; lambda=0 ⇒ novel item (2) wins step 2.
        items = [_make_item('0', {'title': 'a'}), _make_item('1', {'title': 'a b'}),
                 _make_item('2', {'title': 'c'})]
        out = LexicalJaccardDiversifier(_lex_cfg(lambda_relevance=0.0)).diversify( query='', items=items, top_n=3, output_n=3)
        assert [r.item.item_id for r in out] == ['0', '2', '1']
        assert out[0].mmr_score == pytest.approx(0.0)
        assert out[1].mmr_score == pytest.approx(0.0)
        assert out[2].mmr_score == pytest.approx(-0.5)


class TestLexicalJaccardDiversifierEdges:
    def test_empty_items_returns_empty(self):
        assert LexicalJaccardDiversifier(_lex_cfg()).diversify(query='', items=[], top_n=10, output_n=10) == []

    @pytest.mark.parametrize('top_n,output_n', [
        pytest.param(10, 0, id='output_n_zero'),
        pytest.param(0,  10, id='top_n_zero'),
    ])
    def test_zero_returns_empty(self, top_n, output_n):
        items = [_make_item('0', {'title': 'a'})]
        assert LexicalJaccardDiversifier(_lex_cfg()).diversify( query='', items=items, top_n=top_n, output_n=output_n) == []

    def test_output_n_capped_at_top_n(self):
        items = [_make_item(str(i), {'title': str(i)}) for i in range(5)]
        out = LexicalJaccardDiversifier(_lex_cfg()).diversify(query='', items=items, top_n=3, output_n=10)
        assert len(out) == 3

    def test_top_n_capped_at_len_items(self):
        items = [_make_item(str(i), {'title': str(i)}) for i in range(3)]
        out = LexicalJaccardDiversifier(_lex_cfg()).diversify(query='', items=items, top_n=10, output_n=10)
        assert {r.item.item_id for r in out} == {'0', '1', '2'}

    def test_all_empty_payloads_falls_back_to_input_order(self):
        items = [_make_item(str(i), {'unknown_field': 'x'}) for i in range(4)]
        out = LexicalJaccardDiversifier(_lex_cfg(fields=('title',))).diversify( query='', items=items, top_n=4, output_n=4)
        assert [r.item.item_id for r in out] == ['0', '1', '2', '3']

    def test_missing_payload_field_skipped_silently(self):
        items = [_make_item('0', {'title': 'apple'}), _make_item('1', {})]
        out = LexicalJaccardDiversifier(_lex_cfg(fields=('title',))).diversify( query='', items=items, top_n=2, output_n=2)
        assert [r.item.item_id for r in out] == ['0', '1']

    def test_list_value_in_payload_is_flattened(self):
        items = [_make_item('0', {'tags': ['python', 'web']}),
                 _make_item('1', {'tags': ['python']}),
                 _make_item('2', {'tags': ['rust']})]
        out = LexicalJaccardDiversifier(_lex_cfg(lambda_relevance=0.0, fields=('tags',))).diversify( query='', items=items, top_n=3, output_n=3)
        assert [r.item.item_id for r in out] == ['0', '2', '1']

    def test_non_string_payload_value_coerced(self):
        items = [_make_item('0', {'title': 599}), _make_item('1', {'title': 599}),
                 _make_item('2', {'title': 999})]
        out = LexicalJaccardDiversifier(_lex_cfg(lambda_relevance=0.0, fields=('title',))).diversify( query='', items=items, top_n=3, output_n=3)
        assert [r.item.item_id for r in out] == ['0', '2', '1']

    @pytest.mark.parametrize('kwargs,match', [
        matrix_param('query_wrong_type',  dict(query=42,             items=[_make_item('0', {})],   top_n=1, output_n=1), r'query'),
        matrix_param('items_not_seq',     dict(query='',             items={'not': 'list'},          top_n=1, output_n=1), r'items'),
        matrix_param('top_n_negative',    dict(query='',             items=[_make_item('0', {})],   top_n=-1, output_n=1), r'top_n'),
        matrix_param('output_n_negative', dict(query='',             items=[_make_item('0', {})],   top_n=1, output_n=-1), r'output_n'),
    ])
    def test_diversify_rejections(self, kwargs, match):
        with pytest.raises(ValidationError, match=match):
            LexicalJaccardDiversifier(_lex_cfg()).diversify(**kwargs)

    def test_tuple_input_accepted(self):
        items = (_make_item('0', {'title': 'a'}),)
        out = LexicalJaccardDiversifier(_lex_cfg()).diversify(query='', items=items, top_n=1, output_n=1)
        assert len(out) == 1


# ---------------------------------------------------------------------------
# NoOpDiversifier
# ---------------------------------------------------------------------------
class TestNoOpDiversifier:
    def test_name_is_stable(self):
        assert NoOpDiversifier().name == 'noop'

    def test_returns_head_in_input_order(self):
        items = [_make_item(str(i), {'title': str(i)}) for i in range(5)]
        out = NoOpDiversifier().diversify(query='q', items=items, top_n=3, output_n=3)
        assert [r.item.item_id for r in out] == ['0', '1', '2']

    def test_base_and_mmr_scores_decrease_with_rank(self):
        items = [_make_item(str(i), {}) for i in range(3)]
        out = NoOpDiversifier().diversify(query=None, items=items, top_n=3, output_n=3)
        for rank, r in enumerate(out):
            expected = 1.0 / (1.0 + rank)
            assert r.base_score == pytest.approx(expected)
            assert r.mmr_score == pytest.approx(expected)
            assert r.diversity_penalty == 0.0

    def test_empty_items_returns_empty(self):
        assert NoOpDiversifier().diversify(query='', items=[], top_n=10, output_n=10) == []

    @pytest.mark.parametrize('items,top_n,output_n,expected_len', [
        pytest.param([_make_item('0', {})],                          1,  0, 0, id='output_n_zero'),
        pytest.param([_make_item(str(i), {}) for i in range(5)],     2, 10, 2, id='output_n_capped'),
        pytest.param([_make_item(str(i), {}) for i in range(3)],    10, 10, 3, id='top_n_capped'),
    ])
    def test_size_caps(self, items, top_n, output_n, expected_len):
        out = NoOpDiversifier().diversify(query='', items=items, top_n=top_n, output_n=output_n)
        assert len(out) == expected_len

    @pytest.mark.parametrize('kwargs,match', [
        matrix_param('query_wrong_type',  dict(query=99, items=[],  top_n=1, output_n=1), r'query'),
        matrix_param('items_not_seq',     dict(query='', items={},   top_n=1, output_n=1), r'items'),
        matrix_param('top_n_negative',    dict(query='', items=[],   top_n=-1, output_n=1), r'top_n'),
        matrix_param('output_n_negative', dict(query='', items=[],   top_n=1, output_n=-1), r'output_n'),
    ])
    def test_rejections(self, kwargs, match):
        with pytest.raises(ValidationError, match=match):
            NoOpDiversifier().diversify(**kwargs)
