"""Unit tests for NgramPreGate and NgramTrainer.

Coverage matrix:

NgramPreGate.classify:
- fires_on_matching_class                 -> TestNgramPreGate::test_fires_on_matching_class
- returns_none_below_threshold            -> TestNgramPreGate::test_returns_none_below_threshold
- returns_none_when_disabled              -> TestNgramPreGate::test_returns_none_when_disabled
- returns_none_on_empty_query             -> TestNgramPreGate::test_returns_none_on_empty_query
- returns_none_with_empty_weights         -> TestNgramPreGate::test_returns_none_with_empty_weights
- bigram_scoring_adds_weight              -> TestNgramPreGate::test_bigram_scoring_adds_weight
- unigram_only_mode                       -> TestNgramPreGate::test_unigram_only_mode
- case_insensitive                        -> TestNgramPreGate::test_case_insensitive

NgramPreGate.explain:
- explain_returns_class_and_ngram         -> TestNgramPreGate::test_explain_returns_class_and_ngram
- explain_returns_none_when_below_thresh  -> TestNgramPreGate::test_explain_returns_none_when_below_thresh
- explain_disabled_returns_none           -> TestNgramPreGate::test_explain_disabled_returns_none

QINgramPreGateConfig validation:
- rejects_non_bool_enabled                -> TestQINgramPreGateConfig::test_rejects_non_bool_enabled
- rejects_empty_model_path                -> TestQINgramPreGateConfig::test_rejects_empty_model_path
- rejects_non_positive_threshold          -> TestQINgramPreGateConfig::test_rejects_non_positive_threshold
- rejects_invalid_max_ngram_order         -> TestQINgramPreGateConfig::test_rejects_invalid_max_ngram_order
- from_dict_requires_all_keys             -> TestQINgramPreGateConfig::test_from_dict_requires_all_keys
- from_dict_roundtrip                     -> TestQINgramPreGateConfig::test_from_dict_roundtrip

NgramTrainer.train:
- basic_two_class_training                -> TestNgramTrainer::test_basic_two_class_training
- retains_top_k                           -> TestNgramTrainer::test_retains_top_k
- rejects_single_class                    -> TestNgramTrainer::test_rejects_single_class
- rejects_insufficient_examples           -> TestNgramTrainer::test_rejects_insufficient_examples
- rejects_empty_input                     -> TestNgramTrainer::test_rejects_empty_input
- unigram_only_mode                       -> TestNgramTrainer::test_unigram_only_mode

NgramTrainer.save + _load_weights:
- save_and_load_roundtrip                 -> TestNgramTrainer::test_save_and_load_roundtrip
"""
import json
import os
import tempfile
from pathlib import Path
from typing import Dict
from unittest.mock import patch

import pytest

from semantic_search.config.models import QINgramPreGateConfig
from semantic_search.core.exceptions import ConfigurationError
from semantic_search.qi.ngram_pre_gate import NgramPreGate, _load_weights
from semantic_search.qi.training.ngram_trainer import NgramTrainer, NgramTrainerConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config(
    enabled: bool = True,
    model_path: str = "/nonexistent/weights.json",
    confidence_threshold: float = 1.0,
    max_ngram_order: int = 2,
) -> QINgramPreGateConfig:
    return QINgramPreGateConfig(
        enabled=enabled,
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        max_ngram_order=max_ngram_order,
    )


def _gate_with_weights(weights: Dict, threshold: float = 1.0, max_ngram_order: int = 2) -> NgramPreGate:
    cfg = _config(enabled=True, confidence_threshold=threshold, max_ngram_order=max_ngram_order)
    gate = NgramPreGate.__new__(NgramPreGate)
    gate._config = cfg
    gate._enabled = True
    gate._weights = weights
    gate._classes = sorted({cls for row in weights.values() for cls in row})
    return gate


def _trainer_cfg(top_k: int = 10, min_log_odds: float = 0.0, smoothing: float = 1.0, min_examples: int = 1, order: int = 2) -> NgramTrainerConfig:
    return NgramTrainerConfig(
        top_k_per_class=top_k,
        min_log_odds=min_log_odds,
        smoothing=smoothing,
        min_examples_per_class=min_examples,
        max_ngram_order=order,
    )


# ---------------------------------------------------------------------------
# NgramPreGate tests
# ---------------------------------------------------------------------------

class TestNgramPreGate:
    def test_fires_on_matching_class(self) -> None:
        weights = {"trending": {"explore": 2.0}, "domains": {"explore": 0.5}}
        gate = _gate_with_weights(weights, threshold=1.5)
        assert gate.classify("trending domains") == "explore"

    def test_returns_none_below_threshold(self) -> None:
        weights = {"trending": {"explore": 0.5}}
        gate = _gate_with_weights(weights, threshold=5.0)
        assert gate.classify("trending domains") is None

    def test_returns_none_when_disabled(self) -> None:
        gate = NgramPreGate.__new__(NgramPreGate)
        gate._config = _config(enabled=False)
        gate._enabled = False
        gate._weights = {"trending": {"explore": 99.0}}
        gate._classes = ["explore"]
        assert gate.classify("trending domains") is None

    def test_returns_none_on_empty_query(self) -> None:
        weights = {"trending": {"explore": 2.0}}
        gate = _gate_with_weights(weights, threshold=1.0)
        assert gate.classify("") is None

    def test_returns_none_with_empty_weights(self) -> None:
        gate = _gate_with_weights({}, threshold=1.0)
        assert gate.classify("trending domains") is None

    def test_bigram_scoring_adds_weight(self) -> None:
        weights = {
            "conversion rate": {"analytics": 3.0},
            "conversion": {"analytics": 0.5},
        }
        gate = _gate_with_weights(weights, threshold=2.0, max_ngram_order=2)
        assert gate.classify("conversion rate") == "analytics"

    def test_unigram_only_mode(self) -> None:
        weights = {
            "conversion rate": {"analytics": 3.0},
        }
        gate = _gate_with_weights(weights, threshold=1.0, max_ngram_order=1)
        assert gate.classify("conversion rate") is None

    def test_case_insensitive(self) -> None:
        weights = {"trending": {"explore": 2.0}}
        gate = _gate_with_weights(weights, threshold=1.5)
        assert gate.classify("TRENDING domains") == "explore"

    def test_explain_returns_class_and_ngram(self) -> None:
        weights = {"show tld": {"analytics": 3.0}, "domains": {"explore": 0.4}}
        gate = _gate_with_weights(weights, threshold=2.0, max_ngram_order=2)
        cls, ng = gate.explain("show tld domains")
        assert cls == "analytics"
        assert ng == "show tld"

    def test_explain_returns_none_when_below_thresh(self) -> None:
        weights = {"trending": {"explore": 0.1}}
        gate = _gate_with_weights(weights, threshold=5.0)
        cls, ng = gate.explain("trending domains")
        assert cls is None
        assert ng is None

    def test_explain_disabled_returns_none(self) -> None:
        gate = NgramPreGate.__new__(NgramPreGate)
        gate._config = _config(enabled=False)
        gate._enabled = False
        gate._weights = {}
        gate._classes = []
        cls, ng = gate.explain("trending domains")
        assert cls is None
        assert ng is None


# ---------------------------------------------------------------------------
# QINgramPreGateConfig validation tests
# ---------------------------------------------------------------------------

class TestQINgramPreGateConfig:
    def test_rejects_non_bool_enabled(self) -> None:
        with pytest.raises(ConfigurationError, match="enabled must be bool"):
            QINgramPreGateConfig(enabled="yes", model_path="/w.json", confidence_threshold=1.0, max_ngram_order=2)

    def test_rejects_empty_model_path(self) -> None:
        with pytest.raises(ConfigurationError, match="model_path"):
            QINgramPreGateConfig(enabled=True, model_path="", confidence_threshold=1.0, max_ngram_order=2)

    def test_rejects_non_positive_threshold(self) -> None:
        with pytest.raises(ConfigurationError, match="confidence_threshold"):
            QINgramPreGateConfig(enabled=True, model_path="/w.json", confidence_threshold=0.0, max_ngram_order=2)

    def test_rejects_invalid_max_ngram_order(self) -> None:
        with pytest.raises(ConfigurationError, match="max_ngram_order"):
            QINgramPreGateConfig(enabled=True, model_path="/w.json", confidence_threshold=1.0, max_ngram_order=3)

    def test_from_dict_requires_all_keys(self) -> None:
        with pytest.raises(ConfigurationError):
            QINgramPreGateConfig.from_dict({"enabled": True})

    def test_from_dict_roundtrip(self) -> None:
        cfg = QINgramPreGateConfig.from_dict({
            "enabled": False,
            "model_path": "/some/path.json",
            "confidence_threshold": 3.5,
            "max_ngram_order": 1,
        })
        assert cfg.enabled is False
        assert cfg.model_path == "/some/path.json"
        assert cfg.confidence_threshold == 3.5
        assert cfg.max_ngram_order == 1


# ---------------------------------------------------------------------------
# NgramTrainer tests
# ---------------------------------------------------------------------------

class TestNgramTrainer:
    def test_basic_two_class_training(self) -> None:
        examples = [
            ("conversion rate analytics", "analytics"),
            ("conversion rate data", "analytics"),
            ("trending domains explore", "explore"),
            ("explore trending domains", "explore"),
        ]
        trainer = NgramTrainer(_trainer_cfg())
        weights = trainer.train(examples)
        assert isinstance(weights, dict)
        assert len(weights) > 0
        analytics_keys = {ng for ng, row in weights.items() if "analytics" in row}
        explore_keys = {ng for ng, row in weights.items() if "explore" in row}
        assert len(analytics_keys) > 0
        assert len(explore_keys) > 0

    def test_retains_top_k(self) -> None:
        examples = [(f"word{i} stuff", "analytics") for i in range(20)]
        examples += [(f"browse{i} domains", "explore") for i in range(20)]
        trainer = NgramTrainer(_trainer_cfg(top_k=3))
        weights = trainer.train(examples)
        analytics_count = sum(1 for row in weights.values() if "analytics" in row)
        explore_count = sum(1 for row in weights.values() if "explore" in row)
        assert analytics_count <= 3
        assert explore_count <= 3

    def test_rejects_single_class(self) -> None:
        examples = [("query one", "analytics"), ("query two", "analytics")]
        trainer = NgramTrainer(_trainer_cfg())
        with pytest.raises(ConfigurationError, match="2 distinct class"):
            trainer.train(examples)

    def test_rejects_insufficient_examples(self) -> None:
        examples = [("query one", "analytics"), ("domain browse", "explore")]
        trainer = NgramTrainer(_trainer_cfg(min_examples=5))
        with pytest.raises(ConfigurationError, match="examples"):
            trainer.train(examples)

    def test_rejects_empty_input(self) -> None:
        trainer = NgramTrainer(_trainer_cfg())
        with pytest.raises(ConfigurationError, match="at least one example"):
            trainer.train([])

    def test_unigram_only_mode(self) -> None:
        examples = [
            ("conversion rate", "analytics"),
            ("conversion rate", "analytics"),
            ("trending domains", "explore"),
            ("trending domains", "explore"),
        ]
        trainer = NgramTrainer(_trainer_cfg(order=1))
        weights = trainer.train(examples)
        bigrams = [ng for ng in weights if " " in ng]
        assert len(bigrams) == 0

    def test_save_and_load_roundtrip(self) -> None:
        examples = [
            ("conversion rate analytics", "analytics"),
            ("conversion rate data", "analytics"),
            ("trending domains explore", "explore"),
            ("explore trending domains", "explore"),
        ]
        trainer = NgramTrainer(_trainer_cfg())
        weights = trainer.train(examples)
        classes = sorted({"analytics", "explore"})
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "weights.json")
            trainer.save(weights, classes, out)
            loaded_weights, loaded_classes = _load_weights(out)
        assert loaded_classes == classes
        assert set(loaded_weights.keys()) == set(weights.keys())
