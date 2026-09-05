"""Unit tests for qi/llm_classifier.py — constrained generation guards.

Coverage matrix (per testing.mdc §7):

infer_chip_kind:
- known_hard_name_returns_hard          -> TestInferChipKind::test_known_hard_name_returns_hard
- unknown_name_returns_soft             -> TestInferChipKind::test_unknown_name_returns_soft
- unknown_hard_pattern_logs_warning     -> TestInferChipKind::test_unknown_hard_pattern_logs_warning
- unknown_no_pattern_no_warning         -> TestInferChipKind::test_unknown_no_pattern_no_warning
- requires_frozenset                    -> TestInferChipKind::test_requires_frozenset
"""
from unittest.mock import patch

import pytest

from semantic_search.qi.llm_classifier import infer_chip_kind

# Fixture: mirrors qi.entity_slots.hard_entity_names (subset used by these tests).
_HARD = frozenset({
    'price_min', 'price_max', 'tld', 'auction_type', 'domain_age_max', 'domain_age_min',
    'traffic_min', 'traffic_max',
})


class TestInferChipKind:
    def test_known_hard_name_returns_hard(self) -> None:
        assert infer_chip_kind('price_min', _HARD) == 'hard'
        assert infer_chip_kind('tld', _HARD) == 'hard'
        assert infer_chip_kind('auction_type', _HARD) == 'hard'
        assert infer_chip_kind('domain_age_max', _HARD) == 'hard'

    def test_unknown_name_returns_soft(self) -> None:
        assert infer_chip_kind('brandable', _HARD) == 'soft'
        assert infer_chip_kind('aesthetic', _HARD) == 'soft'
        # keyword_contains is soft when not in hard set (config-driven)
        assert infer_chip_kind('keyword_contains', _HARD) == 'soft'

    def test_unknown_hard_pattern_logs_warning(self) -> None:
        with patch('semantic_search.qi.llm_classifier.logger') as mock_log:
            result = infer_chip_kind('blockchain_era_min', _HARD)
        assert result == 'soft'
        warning_calls = [str(c) for c in mock_log.warning.call_args_list]
        assert any('qi_entity_name_unknown_hard_pattern' in c for c in warning_calls)
        assert any('blockchain_era_min' in c for c in warning_calls)

    def test_unknown_no_pattern_no_warning(self) -> None:
        with patch('semantic_search.qi.llm_classifier.logger') as mock_log:
            result = infer_chip_kind('brandable_domain', _HARD)
        assert result == 'soft'
        warning_calls = [str(c) for c in mock_log.warning.call_args_list]
        assert not any('qi_entity_name_unknown_hard_pattern' in c for c in warning_calls)

    def test_requires_frozenset(self) -> None:
        with pytest.raises(TypeError):
            infer_chip_kind('price_min', {'price_min'})  # type: ignore[arg-type]
