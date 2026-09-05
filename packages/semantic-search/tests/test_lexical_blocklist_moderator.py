"""Tests for LexicalBlocklistModerator + NoOpModerator + ModeratorVerdict.

Covers:
    - ``LexicalModeratorConfig`` and ``ModeratorConfig`` validators (config-side
  fail-fast contract).
- ``ModeratorVerdict`` self-consistency invariants.
- ``NoOpModerator`` always returns ``flagged=False`` regardless of payload.
- ``LexicalBlocklistModerator`` runtime behaviour: hit / miss / multi-hit /
  truncation / payload-shape edge cases.
"""
import pytest

from semantic_search.config.models import LexicalModeratorConfig, ModeratorConfig
from semantic_search.core.exceptions import ConfigurationError, EgressGuardError, ValidationError
from semantic_search.safety.lexical_blocklist_moderator import LexicalBlocklistModerator, Moderator, ModeratorVerdict, NoOpModerator


def _lex_cfg( payload_fields=('title', 'description'), banned_terms=('spam', 'fakeword'), min_term_length=3, max_terms=1000, stopwords=(), ) -> LexicalModeratorConfig:
    return LexicalModeratorConfig(
        payload_fields=list(payload_fields),
        banned_terms=list(banned_terms),
        min_term_length=int(min_term_length),
        max_terms=int(max_terms),
        stopwords=list(stopwords),
    )


# --------------------------------------------------------------------------- #
# LexicalModeratorConfig validation                                            #
# --------------------------------------------------------------------------- #

class TestLexicalModeratorConfig:
    def test_valid(self):
        cfg = _lex_cfg()
        assert cfg.banned_terms == ['spam', 'fakeword']

    def test_empty_payload_fields_rejected(self):
        with pytest.raises(ConfigurationError, match="payload_fields must be a non-empty list"):
            _lex_cfg(payload_fields=[])

    def test_payload_fields_must_be_strings(self):
        with pytest.raises(ConfigurationError, match="payload_fields entries must be non-empty strings"):
            _lex_cfg(payload_fields=[''])

    def test_empty_banned_terms_rejected(self):
        with pytest.raises(ConfigurationError, match="banned_terms must be a non-empty list"):
            _lex_cfg(banned_terms=[])

    def test_uppercase_banned_term_rejected(self):
        with pytest.raises(ConfigurationError, match="banned_terms entries must be lowercase"):
            _lex_cfg(banned_terms=['Spam'])

    def test_min_term_length_below_one_rejected(self):
        with pytest.raises(ConfigurationError, match="min_term_length must be >= 1"):
            _lex_cfg(min_term_length=0)

    def test_max_terms_below_one_rejected(self):
        with pytest.raises(ConfigurationError, match="max_terms must be >= 1"):
            _lex_cfg(max_terms=0)

    def test_banned_term_in_stopwords_rejected(self):
        # A banned term shadowed by a stopword would be silently un-banned.
        with pytest.raises(ConfigurationError, match="is also in stopwords"):
            _lex_cfg(banned_terms=['spam'], stopwords=['spam'])

    def test_from_dict_round_trip(self):
        d = {
            'payload_fields': ['title'],
            'banned_terms': ['spam'],
            'min_term_length': 3,
            'max_terms': 100,
            'stopwords': ['the'],
        }
        cfg = LexicalModeratorConfig.from_dict(d)
        assert cfg.payload_fields == ['title']
        assert cfg.banned_terms == ['spam']

    def test_from_dict_missing_field_rejected(self):
        d = {'payload_fields': ['title']}  # missing the rest
        with pytest.raises(ConfigurationError):
            LexicalModeratorConfig.from_dict(d)


# --------------------------------------------------------------------------- #
# ModeratorConfig validation                                                   #
# --------------------------------------------------------------------------- #

class TestModeratorConfig:
    def test_disabled_no_lexical_required(self):
        cfg = ModeratorConfig(enabled=False, backend='noop', policy='drop', lexical=None)
        assert cfg.backend == 'noop'

    def test_unknown_backend_rejected(self):
        with pytest.raises(ConfigurationError, match="backend must be one of"):
            ModeratorConfig(enabled=True, backend='bogus', policy='drop', lexical=None)

    def test_unknown_policy_rejected(self):
        with pytest.raises(ConfigurationError, match="policy must be one of"):
            ModeratorConfig(enabled=False, backend='noop', policy='ignore', lexical=None)

    def test_lexical_backend_requires_lexical_subcfg_when_enabled(self):
        with pytest.raises(ConfigurationError, match="lexical is required when enabled=true"):
            ModeratorConfig(enabled=True, backend='lexical_blocklist', policy='drop', lexical=None)

    def test_lexical_backend_disabled_does_not_require_lexical_subcfg(self):
        # Disabled lexical backend should NOT need the sub-config — registry
        # will wire NoOpModerator anyway.
        cfg = ModeratorConfig(enabled=False, backend='lexical_blocklist', policy='drop', lexical=None)
        assert cfg.lexical is None

    def test_lexical_field_must_be_dataclass(self):
        with pytest.raises(ConfigurationError, match="lexical must be a LexicalModeratorConfig"):
            ModeratorConfig(enabled=True, backend='lexical_blocklist', policy='drop', lexical='not_a_dataclass')  # type: ignore[arg-type]

    def test_from_dict_with_lexical(self):
        d = {
            'enabled': True,
            'backend': 'lexical_blocklist',
            'policy': 'mask',
            'lexical': {
                'payload_fields': ['title'],
                'banned_terms': ['spam'],
                'min_term_length': 3,
                'max_terms': 100,
                'stopwords': [],
            },
        }
        cfg = ModeratorConfig.from_dict(d)
        assert cfg.policy == 'mask'
        assert cfg.lexical is not None and cfg.lexical.banned_terms == ['spam']


# --------------------------------------------------------------------------- #
# ModeratorVerdict invariants                                                  #
# --------------------------------------------------------------------------- #

class TestModeratorVerdict:
    def test_unflagged_default(self):
        v = ModeratorVerdict(flagged=False)
        assert v.matched_terms == []
        assert v.truncated_matches is False

    def test_flagged_with_terms(self):
        v = ModeratorVerdict(flagged=True, matched_terms=['spam'])
        assert v.flagged is True
        assert v.matched_terms == ['spam']

    def test_unflagged_with_terms_rejected(self):
        with pytest.raises(ValidationError, match="matched_terms must be empty when flagged=False"):
            ModeratorVerdict(flagged=False, matched_terms=['spam'])

    def test_unflagged_with_truncation_rejected(self):
        with pytest.raises(ValidationError, match="truncated_matches cannot be True when flagged=False"):
            ModeratorVerdict(flagged=False, matched_terms=[], truncated_matches=True)

    def test_matched_terms_must_be_strings(self):
        with pytest.raises(ValidationError, match="matched_terms entries must be non-empty"):
            ModeratorVerdict(flagged=True, matched_terms=[''])


# --------------------------------------------------------------------------- #
# NoOpModerator                                                                #
# --------------------------------------------------------------------------- #

class TestNoOpModerator:
    def test_name(self):
        assert NoOpModerator().name == 'noop'

    def test_satisfies_protocol(self):
        # `Moderator` is `@runtime_checkable` so isinstance works on duck types.
        assert isinstance(NoOpModerator(), Moderator)

    @pytest.mark.parametrize('payload', [
        {},
        {'title': 'hello world'},
        {'title': 'spam fakeword evil'},
        {'unrelated': 42, 'list_field': ['a', 'b']},
    ])
    def test_always_unflagged(self, payload):
        v = NoOpModerator().moderate(payload)
        assert v.flagged is False
        assert v.matched_terms == []


# --------------------------------------------------------------------------- #
# LexicalBlocklistModerator                                                    #
# --------------------------------------------------------------------------- #

class TestLexicalBlocklistModeratorConstruction:
    def test_construction_with_valid_config(self):
        m = LexicalBlocklistModerator(_lex_cfg())
        assert m.name == 'lexical_blocklist'
        assert isinstance(m, Moderator)

    def test_none_config_rejected(self):
        with pytest.raises(EgressGuardError, match="requires a LexicalModeratorConfig"):
            LexicalBlocklistModerator(None)  # type: ignore[arg-type]

    def test_wrong_type_config_rejected(self):
        with pytest.raises(EgressGuardError, match="requires a LexicalModeratorConfig"):
            LexicalBlocklistModerator("not_a_config")  # type: ignore[arg-type]


class TestLexicalBlocklistModerate:
    def test_clean_payload_unflagged(self):
        m = LexicalBlocklistModerator(_lex_cfg())
        v = m.moderate({'title': 'premium domain', 'description': 'great brand opportunity'})
        assert v.flagged is False
        assert v.matched_terms == []

    def test_single_banned_token_flagged(self):
        m = LexicalBlocklistModerator(_lex_cfg())
        v = m.moderate({'title': 'this is spam', 'description': 'clean text'})
        assert v.flagged is True
        assert 'spam' in v.matched_terms

    def test_multiple_banned_tokens_collected(self):
        m = LexicalBlocklistModerator(_lex_cfg())
        v = m.moderate({'title': 'spam content', 'description': 'fakeword item'})
        assert v.flagged is True
        assert set(v.matched_terms) == {'spam', 'fakeword'}

    def test_duplicate_matches_deduplicated(self):
        m = LexicalBlocklistModerator(_lex_cfg())
        v = m.moderate({'title': 'spam spam spam', 'description': 'spam'})
        assert v.flagged is True
        assert v.matched_terms == ['spam']  # Single entry despite 4 hits.

    def test_short_token_below_min_length_filtered_out(self):
        # min_term_length=3 means a 2-char banned word never matches.
        cfg = _lex_cfg(banned_terms=['ad'], min_term_length=3)
        m = LexicalBlocklistModerator(cfg)
        v = m.moderate({'title': 'this is an ad'})
        assert v.flagged is False

    def test_stopword_filters_out_token(self):
        cfg = _lex_cfg(banned_terms=['spam'], stopwords=['the', 'is'])
        m = LexicalBlocklistModerator(cfg)
        v = m.moderate({'title': 'the spam is evil'})
        assert v.flagged is True  # 'spam' still survives the stopword filter.

    def test_missing_payload_field_skipped(self):
        m = LexicalBlocklistModerator(_lex_cfg(payload_fields=['title', 'missing_field']))
        v = m.moderate({'title': 'spam'})
        assert v.flagged is True
        assert v.matched_terms == ['spam']

    def test_none_field_value_skipped(self):
        m = LexicalBlocklistModerator(_lex_cfg())
        v = m.moderate({'title': None, 'description': 'spam content'})
        assert v.flagged is True

    def test_non_string_field_coerced_via_str(self):
        # The moderator coerces non-string scalars via str() so numerics
        # contribute tokens. With banned_terms=['42'] and a numeric payload,
        # we still want a hit. Tokenizer drops short tokens though
        # (min_term_length=3 default), so use a longer banned numeric.
        cfg = _lex_cfg(banned_terms=['9999'])
        m = LexicalBlocklistModerator(cfg)
        v = m.moderate({'title': 9999, 'description': 'clean'})
        assert v.flagged is True

    def test_empty_payload_unflagged(self):
        m = LexicalBlocklistModerator(_lex_cfg())
        v = m.moderate({})
        assert v.flagged is False

    def test_non_mapping_payload_raises_egress_guard_error(self):
        m = LexicalBlocklistModerator(_lex_cfg())
        with pytest.raises(EgressGuardError, match="payload must be a Mapping"):
            m.moderate(['not', 'a', 'mapping'])  # type: ignore[arg-type]

    def test_determinism(self):
        m = LexicalBlocklistModerator(_lex_cfg())
        payload = {'title': 'spam payload', 'description': 'fakeword too'}
        v1 = m.moderate(payload)
        v2 = m.moderate(payload)
        assert v1.flagged == v2.flagged
        assert v1.matched_terms == v2.matched_terms

    def test_truncation_flag_set_on_huge_match_set(self):
        # Build a banned-term list that exceeds the internal cap of 16 reported
        # matches and a payload that hits all of them. The verdict should
        # report truncated_matches=True.
        many_bans = [f'term{i:03d}' for i in range(30)]
        cfg = _lex_cfg(banned_terms=many_bans, max_terms=10000)
        m = LexicalBlocklistModerator(cfg)
        text = ' '.join(many_bans)
        v = m.moderate({'title': text, 'description': ''})
        assert v.flagged is True
        # Internal cap is 16; matched_terms must be capped at 16 with the
        # truncated flag set.
        assert len(v.matched_terms) == 16
        assert v.truncated_matches is True
