"""Tier-0 spell-correct and 'did you mean' formal regression suite.

Every behavioural row is preserved as a `pytest.param(id=...)` so the matrix
is recoverable via `pytest --collect-only -q`. A compact one-line-per-row
matrix sits at the top of each test class.
"""
import pathlib
from typing import Optional

import pytest

from semantic_search.config.models import AgentSearchConfig, SpellCorrectConfig
from semantic_search.contracts import DECISION_TIERS, IntentSlice, QueryIntent, SpellCorrection, TokenCorrection
from semantic_search.core.exceptions import ConfigurationError, ValidationError
from semantic_search.core.text_distance import damerau_levenshtein
from semantic_search.qi.spell_corrector import SymSpellCorrector
from ._contract_helpers import assert_dataclass_contract, assert_frozen, matrix_param


# ---------------------------------------------------------------------------
# Local helpers — minimum-viable factories so tests don't restate every field
# ---------------------------------------------------------------------------


def _config(**overrides) -> SpellCorrectConfig:
    """Build a valid SpellCorrectConfig with selective overrides."""
    base = dict(
        enabled=True,
        frequency_dict_path="symspellpy:frequency_dictionary_en_82_765.txt",
        max_edit_distance=2,
        prefix_length=7,
        min_token_length=3,
        max_tokens_to_correct=6,
        verbosity="CLOSEST",
        auto_apply=True,
        protected_tokens=[],
        protected_phrase_patterns=[],
    )
    base.update(overrides)
    return SpellCorrectConfig(**base)


def _intent_minimal(**overrides) -> QueryIntent:
    """Minimum-viable QueryIntent to exercise did_you_mean without
    restating every required field."""
    base = dict(
        request_id='req_xyz',
        raw_query='expring drives',
        normalized_query='expring drives',
        query_type='hybrid',
        confidence=0.9,
        decision_tier='L0_entity',
        slices=[IntentSlice(query_type='hybrid', entities=[], confidence=0.9, raw_text='expring drives')],
        decision_cost_usd=0.0,
    )
    base.update(overrides)
    return QueryIntent(**base)


# ---------------------------------------------------------------------------
# Contracts: TokenCorrection, SpellCorrection, QueryIntent.did_you_mean,
# DECISION_TIERS extension
# ---------------------------------------------------------------------------


class TestSpellCorrectionContracts:
    """Coverage matrix:
    - token_correction_minimal_ok            -> test_token_correction_happy_path
    - token_correction_field_rejections      -> test_token_correction_rejects_invalid (parametrized)
    - token_correction_frozen                -> test_token_correction_frozen
    - spell_correction_minimal_ok            -> test_spell_correction_happy_path
    - spell_correction_field_rejections      -> test_spell_correction_rejects_invalid (parametrized)
    - spell_correction_frozen                -> test_spell_correction_frozen
    - decision_tier_includes_l0_spell        -> test_decision_tier_includes_l0_spell_correct
    - query_intent_did_you_mean_default_none -> test_query_intent_did_you_mean_default_none
    - query_intent_did_you_mean_attached     -> test_query_intent_did_you_mean_attached
    - query_intent_did_you_mean_wrong_type   -> test_query_intent_did_you_mean_wrong_type_raises
    """

    def test_token_correction_happy_path(self):
        tc = TokenCorrection(original='expring', corrected='expiring', edit_distance=1)
        assert (tc.original, tc.corrected, tc.edit_distance) == ('expring', 'expiring', 1)

    def test_token_correction_rejects_invalid(self):
        assert_dataclass_contract(
            TokenCorrection,
            valid_kwargs=dict(original='expring', corrected='expiring', edit_distance=1),
            type_violations={
                'original': ('', 'original'),
                'corrected': ('', 'corrected'),
                'edit_distance': (0, 'positive'),
            },
            equality_violations={
                ('original', 'corrected'): ('same', 'same', 'differ'),
            },
            extra_violations={
                'negative_distance': (dict(edit_distance=-1), 'positive'),
            },
            exception_cls=ValidationError,
        )

    def test_token_correction_frozen(self):
        tc = TokenCorrection(original='a', corrected='b', edit_distance=1)
        assert_frozen(tc, 'original')

    def test_spell_correction_happy_path(self):
        tc = TokenCorrection(original='expring', corrected='expiring', edit_distance=1)
        sc = SpellCorrection(original_query='expring drives', corrected_query='expiring drives', corrections=[tc], applied=True)
        assert sc.original_query == 'expring drives'
        assert sc.corrected_query == 'expiring drives'
        assert sc.corrections == [tc]
        assert sc.applied is True

    def test_spell_correction_rejects_invalid(self):
        tc = TokenCorrection(original='a', corrected='b', edit_distance=1)
        assert_dataclass_contract(
            SpellCorrection,
            valid_kwargs=dict(original_query='x', corrected_query='y', corrections=[tc], applied=True),
            type_violations={
                'corrections': ([], 'non-empty list'),
                'applied': ('yes', 'applied must be a bool'),
            },
            equality_violations={
                ('original_query', 'corrected_query'): ('same', 'same', 'differ from original_query'),
            },
            extra_violations={
                'wrong_correction_type': (dict(corrections=['not-a-correction']), 'TokenCorrection'),
            },
            exception_cls=ValidationError,
        )

    def test_spell_correction_frozen(self):
        tc = TokenCorrection(original='a', corrected='b', edit_distance=1)
        sc = SpellCorrection(original_query='x', corrected_query='y', corrections=[tc], applied=True)
        assert_frozen(sc, 'applied')

    def test_query_intent_did_you_mean_default_none(self):
        assert _intent_minimal().did_you_mean is None

    def test_query_intent_did_you_mean_attached(self):
        tc = TokenCorrection(original='expring', corrected='expiring', edit_distance=1)
        sc = SpellCorrection(original_query='expring drives', corrected_query='expiring drives', corrections=[tc], applied=True)
        intent = _intent_minimal(did_you_mean=sc)
        assert intent.did_you_mean is sc
        assert intent.did_you_mean.corrections[0].corrected == 'expiring'

    def test_query_intent_did_you_mean_wrong_type_raises(self):
        with pytest.raises(ValidationError, match='SpellCorrection'):
            _intent_minimal(did_you_mean='not-a-correction')


# ---------------------------------------------------------------------------
# Config: SpellCorrectConfig boundary + from_dict + qi wiring
# ---------------------------------------------------------------------------


class TestSpellCorrectConfig:
    """Coverage matrix:
    - happy_path                  -> test_happy_path
    - field_rejections            -> test_config_rejects_invalid (parametrized)
    - from_dict_missing_required  -> test_from_dict_missing_required
    - qi_omitted_yields_none      -> test_qi_config_omitted_yields_none
    - qi_loads_block              -> test_qi_config_loads_block
    """

    def test_happy_path(self):
        cfg = _config()
        assert cfg.enabled is True
        assert cfg.frequency_dict_path == "symspellpy:frequency_dictionary_en_82_765.txt"
        assert cfg.max_edit_distance == 2
        assert cfg.prefix_length == 7
        assert cfg.min_token_length == 3
        assert cfg.max_tokens_to_correct == 6
        assert cfg.verbosity == "CLOSEST"
        assert cfg.auto_apply is True
        assert cfg.protected_tokens == []
        assert cfg.protected_phrase_patterns == []

    @pytest.mark.parametrize('overrides,error_match', [
        matrix_param('enabled_non_bool',            dict(enabled='yes'),                       'enabled must be a bool'),
        matrix_param('auto_apply_non_bool',         dict(auto_apply='no'),                     'auto_apply must be a bool'),
        matrix_param('frequency_dict_path_empty',   dict(frequency_dict_path=''),              'frequency_dict_path must be a non-empty'),
        matrix_param('prefix_length_zero',          dict(prefix_length=0),                     r'prefix_length must be >= 1'),
        matrix_param('min_token_length_too_small',  dict(min_token_length=1),                  'min_token_length must be >= 2'),
        matrix_param('max_edit_distance_zero',      dict(max_edit_distance=0),                 r'max_edit_distance must be in \[1, 3\]'),
        matrix_param('max_edit_distance_too_high',  dict(max_edit_distance=4),                 r'max_edit_distance must be in \[1, 3\]'),
        matrix_param('max_tokens_zero',             dict(max_tokens_to_correct=0),             'max_tokens_to_correct must be >= 1'),
        matrix_param('verbosity_invalid',           dict(verbosity='UNKNOWN'),                 'verbosity must be one of'),
        matrix_param('protected_phrase_bad_regex',  dict(protected_phrase_patterns=['[bad']),  'invalid regex'),
    ])
    def test_config_rejects_invalid(self, overrides, error_match):
        with pytest.raises(ConfigurationError, match=error_match):
            _config(**overrides)

    def test_from_dict_missing_required(self):
        with pytest.raises(ConfigurationError, match='qi.spell_correct'):
            SpellCorrectConfig.from_dict({'enabled': True})

    def test_qi_config_omitted_yields_none(self, config_dict):
        cfg_dict = dict(config_dict)
        cfg_dict['qi'] = dict(cfg_dict['qi'])
        cfg_dict['qi'].pop('spell_correct', None)
        assert AgentSearchConfig.from_dict(cfg_dict).qi.spell_correct is None

    def test_qi_config_loads_block(self, config_dict):
        cfg_dict = dict(config_dict)
        cfg_dict['qi'] = dict(cfg_dict['qi'])
        cfg_dict['qi']['spell_correct'] = {
            'enabled': True,
            'frequency_dict_path': 'symspellpy:frequency_dictionary_en_82_765.txt',
            'max_edit_distance': 2,
            'prefix_length': 7,
            'min_token_length': 3,
            'max_tokens_to_correct': 6,
            'verbosity': 'CLOSEST',
            'auto_apply': True,
            'protected_tokens': ['cpc', 'ctr'],
            'protected_phrase_patterns': [],
        }
        cfg = AgentSearchConfig.from_dict(cfg_dict)
        assert isinstance(cfg.qi.spell_correct, SpellCorrectConfig)
        assert cfg.qi.spell_correct.enabled is True
        assert cfg.qi.spell_correct.verbosity == 'CLOSEST'
        assert 'cpc' in cfg.qi.spell_correct.protected_tokens


# ---------------------------------------------------------------------------
# Damerau-Levenshtein algorithm
# ---------------------------------------------------------------------------


class TestDamerauLevenshtein:
    """Coverage matrix:
    Each ``pytest.param`` id is the matrix row. Distance values are
    hand-calculated and noted inline in the param table.
    """

    @pytest.mark.parametrize('a,b,max_d,expected', [
        matrix_param('equal_strings_zero',         'hello', 'hello', 2, 0),    # identity
        matrix_param('single_substitution',        'cat',   'bat',   2, 1),    # 1 sub
        matrix_param('single_insertion',           'cat',   'cats',  2, 1),    # 1 ins
        matrix_param('single_deletion',            'cats',  'cat',   2, 1),    # 1 del
        matrix_param('adjacent_transposition',     'expring', 'expirng', 2, 1),# Damerau swap = 1
        matrix_param('non_adjacent_transposition', 'abc',   'cba',   3, 2),    # only adjacent swap rewarded
        matrix_param('length_diff_short_circuit',  'a',     'abcdefghij', 2, 3), # |Δlen|=9 > max=2 → sentinel max+1
        matrix_param('early_exit_returns_sentinel','kitten','sitting',2, 3),    # true=3, max=2 → row-min bail
        matrix_param('empty_a',                    '',      'abc',   3, 3),    # 3 insertions
        matrix_param('empty_b',                    'abc',   '',      3, 3),    # 3 deletions
        matrix_param('known_two_edit_query',       'exiprng','expiring',3, 2), # 1 swap + 1 ins
    ])
    def test_distance(self, a, b, max_d, expected):
        assert damerau_levenshtein(a, b, max_d) == expected


# ---------------------------------------------------------------------------
# SymSpellCorrector — construction + runtime behaviour
# ---------------------------------------------------------------------------


@pytest.fixture()
def spell_dict_path(tmp_path) -> str:
    """Write a minimal tab-separated frequency dictionary to a temp file."""
    entries = [
        ("expiring", 1000), ("drives", 800), ("premium", 600),
        ("cheap", 500), ("brand", 400), ("brands", 400),
        ("budget", 300), ("names", 300), ("distribution", 200),
        ("average", 200),
    ]
    p = tmp_path / "test_freq_dict.txt"
    p.write_text("\n".join(f"{w} {c}" for w, c in entries), encoding="utf-8")
    return str(p)


@pytest.fixture()
def corrector(spell_dict_path) -> SymSpellCorrector:
    return SymSpellCorrector(config=_config(frequency_dict_path=spell_dict_path))


class TestSymSpellCorrector:
    """Coverage matrix:
    - construction_happy_path            -> test_construction_happy_path
    - bad_dict_path_raises               -> test_bad_dict_path_raises
    - single_typo_correction             -> test_single_typo_correction
    - in_vocab_returns_none              -> test_in_vocab_returns_none
    - short_token_skipped                -> test_short_token_skipped
    - max_tokens_to_correct_capped       -> test_max_tokens_to_correct_capped
    - repeated_typo_dedup                -> test_repeated_typo_dedup
    - non_word_chars_preserved           -> test_non_word_chars_preserved
    - rewrite_offsets_preserved          -> test_rewrite_offsets_preserved
    - protected_token_skipped            -> test_protected_token_skipped
    - applied_flag_propagated            -> test_applied_flag_propagated (parametrized)
    - soft_fail_internal_error           -> test_soft_fail_internal_error
    """

    def test_construction_happy_path(self, spell_dict_path):
        c = SymSpellCorrector(config=_config(frequency_dict_path=spell_dict_path))
        assert c is not None

    def test_bad_dict_path_raises(self):
        with pytest.raises(ValidationError, match='SymSpell'):
            SymSpellCorrector(config=_config(frequency_dict_path='/nonexistent/path/dict.txt'))

    def test_single_typo_correction(self, corrector):
        sc = corrector.correct('expring drives')
        assert sc is not None
        assert sc.original_query == 'expring drives'
        assert sc.corrected_query == 'expiring drives'
        assert len(sc.corrections) == 1
        tc = sc.corrections[0]
        assert tc.original == 'expring'
        assert tc.corrected == 'expiring'
        assert tc.edit_distance >= 1

    def test_in_vocab_returns_none(self, corrector):
        assert corrector.correct('expiring drives') is None

    def test_short_token_skipped(self, corrector):
        assert corrector.correct('ab') is None

    def test_max_tokens_to_correct_capped(self, spell_dict_path):
        c = SymSpellCorrector(config=_config(frequency_dict_path=spell_dict_path, max_tokens_to_correct=1))
        sc = c.correct('expring drvies')
        assert sc is not None and len(sc.corrections) == 1

    def test_repeated_typo_dedup(self, corrector):
        sc = corrector.correct('expring expring expring')
        assert sc is not None
        assert len(sc.corrections) == 1
        assert sc.corrected_query == 'expiring expiring expiring'

    def test_non_word_chars_preserved(self, corrector):
        sc = corrector.correct('expring  drives')
        assert sc is not None and sc.corrected_query == 'expiring  drives'

    def test_rewrite_offsets_preserved(self, corrector):
        sc = corrector.correct('cheap expring drvies')
        assert sc is not None and 'expiring' in sc.corrected_query

    def test_protected_token_skipped(self, spell_dict_path):
        c = SymSpellCorrector(config=_config(
            frequency_dict_path=spell_dict_path,
            protected_tokens=['expring'],
        ))
        assert c.correct('expring drives') is None

    @pytest.mark.parametrize('auto_apply,expected_applied', [
        matrix_param('auto_apply_true',  True,  True),
        matrix_param('auto_apply_false', False, False),
    ])
    def test_applied_flag_propagated(self, spell_dict_path, auto_apply, expected_applied):
        c = SymSpellCorrector(config=_config(frequency_dict_path=spell_dict_path, auto_apply=auto_apply))
        sc = c.correct('expring')
        assert sc is not None and sc.applied is expected_applied

    def test_soft_fail_internal_error(self, corrector, monkeypatch):
        monkeypatch.setattr(corrector, '_correct_inner', lambda _q: (_ for _ in ()).throw(RuntimeError('boom')))
        assert corrector.correct('expring drives') is None
