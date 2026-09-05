"""N-gram log-odds pre-gate: replaces IntentPreGate phrase lists with a learned scorer.

Loads a weights file produced by NgramTrainer (JSON, offline).
At query time extracts unigrams + optional bigrams, sums per-class log-odds weights,
and returns the top class iff its score meets confidence_threshold.
Sub-millisecond — pure Python dict lookup, no model inference at query time.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from semantic_search.config.models import QINgramPreGateConfig
from semantic_search.core.exceptions import ConfigurationError
from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)


def _load_weights(model_path: str) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    path = Path(model_path)
    if not path.is_absolute():
        project_root = Path(__file__).resolve().parents[2]
        path = (project_root / path).resolve()
    if not path.exists():
        raise ConfigurationError(f"NgramPreGate weights file not found: {model_path}")
    with open(path, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    if data.get('format_version') != 1:
        raise ConfigurationError(f"NgramPreGate weights format_version must be 1, got {data.get('format_version')!r}")
    classes: List[str] = [str(c) for c in data['classes']]
    weights: Dict[str, Dict[str, float]] = {
        ng: {cls: float(w) for cls, w in row.items()}
        for ng, row in data['weights'].items()
    }
    return weights, classes


class NgramPreGate:
    """N-gram log-odds scorer (dict lookup + sum, sub-millisecond)."""

    def __init__(self, config: QINgramPreGateConfig) -> None:
        if config is None:
            raise ValueError("NgramPreGate requires a QINgramPreGateConfig instance")
        self._config = config
        self._enabled: bool = config.enabled
        self._weights: Dict[str, Dict[str, float]] = {}
        self._classes: List[str] = []
        if config.enabled:
            self._weights, self._classes = _load_weights(config.model_path)
            logger.info(
                f"ngram_pre_gate_loaded vocab_size={len(self._weights)} "
                f"classes={self._classes} "
                f"threshold={config.confidence_threshold} order={config.max_ngram_order}"
            )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _extract_ngrams(self, query: str) -> List[str]:
        tokens = query.lower().split()
        ngrams: List[str] = list(tokens)
        if self._config.max_ngram_order >= 2:
            ngrams += [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
        return ngrams

    def classify(self, query: str) -> Optional[str]:
        """Query → class label (if score >= confidence_threshold, else None)."""
        if not self._enabled or not query or not self._weights:
            return None
        class_scores: Dict[str, float] = {}
        for ng in self._extract_ngrams(query):
            row = self._weights.get(ng)
            if row is None:
                continue
            for cls, w in row.items():
                class_scores[cls] = class_scores.get(cls, 0.0) + w
        if not class_scores:
            return None
        best_cls = max(class_scores, key=lambda c: class_scores[c])
        best_score = class_scores[best_cls]
        if best_score < self._config.confidence_threshold:
            return None
        if self._config.margin_threshold > 0 and len(class_scores) >= 2:
            second_score = sorted(class_scores.values(), reverse=True)[1]
            if best_score - second_score < self._config.margin_threshold:
                return None
        logger.debug(f"ngram_pre_gate_hit class={best_cls} score={best_score:.3f} query_len={len(query)}")
        return best_cls

    def explain(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """Return (intent_class, top_contributing_ngram) for diagnostics.

        :param query: str - Normalized query text.
        :return: Tuple[Optional[str], Optional[str]] - (class, ngram) when gate fires, else (None, None).
        """
        if not self._enabled or not query or not self._weights:
            return None, None
        ngrams = self._extract_ngrams(query)
        class_scores: Dict[str, float] = {}
        ngram_contrib: Dict[str, Dict[str, float]] = {}
        for ng in ngrams:
            row = self._weights.get(ng)
            if row is None:
                continue
            ngram_contrib[ng] = row
            for cls, w in row.items():
                class_scores[cls] = class_scores.get(cls, 0.0) + w
        if not class_scores:
            return None, None
        best_cls = max(class_scores, key=lambda c: class_scores[c])
        best_score = class_scores[best_cls]
        if best_score < self._config.confidence_threshold:
            return None, None
        if self._config.margin_threshold > 0 and len(class_scores) >= 2:
            second_score = sorted(class_scores.values(), reverse=True)[1]
            if best_score - second_score < self._config.margin_threshold:
                return None, None
        best_ngram = max(
            (ng for ng in ngram_contrib if best_cls in ngram_contrib[ng]),
            key=lambda ng: ngram_contrib[ng].get(best_cls, 0.0),
            default=None,
        )
        return best_cls, best_ngram
