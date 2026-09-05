"""Offline trainer for NgramPreGate weights.

Reads a labeled query corpus, computes per-class log-odds weights for unigrams and
bigrams, and serializes the top-K discriminative n-grams per class to a JSON weights
file. Two interchangeable corpus sources are supported:

  * ``--seeds`` — the Semantic Router seed YAML (``qi/router_seeds.yaml``). Each
    archetype is a class label and every seed query is one example. This is the
    canonical, enriched source shared with the SemanticRouter; loaded via the same
    validated ``RouterSeedLoader``.
  * ``--input`` — a JSONL corpus of ``{"text": "...", "label": "..."}`` rows
    (weak-labeled from traffic logs: L2/LLM-classifier decisions at confidence
    >= 0.90). Legacy/alternate source.

Exactly one of ``--seeds`` / ``--input`` must be supplied.

Usage (CLI):
    python -m semantic_search.qi.training.ngram_trainer \\
        --seeds semantic_search/qi/router_seeds.yaml \\
        --output .pretrained/ngram_pregate/weights.json \\
        --top-k 200 \\
        --min-log-odds 0.3 \\
        --smoothing 1.0 \\
        --min-examples 50 \\
        --max-ngram-order 2

Input JSONL format (``--input`` mode):
    {"text": "show me trending domains", "label": "explore"}
    {"text": "what is the conversion rate", "label": "analytics"}
"""
import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from math import log
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from semantic_search.core.exceptions import ConfigurationError
from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)

# Default log-odds hyperparameters. Used by the CLI defaults and by callers that
# co-train the gate programmatically (e.g. the head trainer) so the standalone
# and co-training paths produce identical weights for the same corpus.
_DEFAULT_TOP_K = 200
_DEFAULT_MIN_LOG_ODDS = 0.3
_DEFAULT_SMOOTHING = 1.0
_DEFAULT_MIN_EXAMPLES = 50


@dataclass
class NgramTrainerConfig:
    """Config for the offline NgramTrainer.

    :param top_k_per_class: Max n-grams to retain per class by log-odds score.
    :param min_log_odds: Minimum log-odds for a (ngram, class) pair to be retained.
        Raise to keep only highly discriminative n-grams; lower for broader coverage.
    :param smoothing: Laplace smoothing constant applied to all count estimates.
        Prevents division by zero and reduces variance on rare n-grams.
    :param min_examples_per_class: Minimum labeled examples required per class before
        training proceeds. Raises ConfigurationError when violated.
    :param max_ngram_order: Maximum n-gram order to extract (1 = unigrams only,
        2 = unigrams + bigrams). Must match the NgramPreGate inference config.
    """
    top_k_per_class: int
    min_log_odds: float
    smoothing: float
    min_examples_per_class: int
    max_ngram_order: int

    def __post_init__(self) -> None:
        if not isinstance(self.top_k_per_class, int) or self.top_k_per_class < 1:
            raise ConfigurationError("NgramTrainerConfig.top_k_per_class must be a positive integer")
        if not isinstance(self.min_log_odds, (int, float)) or float(self.min_log_odds) < 0:
            raise ConfigurationError("NgramTrainerConfig.min_log_odds must be a non-negative number")
        if not isinstance(self.smoothing, (int, float)) or float(self.smoothing) <= 0:
            raise ConfigurationError("NgramTrainerConfig.smoothing must be a positive number")
        if not isinstance(self.min_examples_per_class, int) or self.min_examples_per_class < 1:
            raise ConfigurationError("NgramTrainerConfig.min_examples_per_class must be a positive integer")
        if not isinstance(self.max_ngram_order, int) or self.max_ngram_order not in (1, 2):
            raise ConfigurationError("NgramTrainerConfig.max_ngram_order must be 1 or 2")


class NgramTrainer:
    """Trains n-gram log-odds weights from a labeled query corpus.

    :param config: NgramTrainerConfig - Training hyperparameters.
    """

    def __init__(self, config: NgramTrainerConfig) -> None:
        self._config = config

    def _extract_ngrams(self, text: str) -> List[str]:
        tokens = text.lower().split()
        ngrams: List[str] = list(tokens)
        if self._config.max_ngram_order >= 2:
            ngrams += [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
        return ngrams

    def train(self, examples: List[Tuple[str, str]]) -> Dict[str, Dict[str, float]]:
        """Compute per-class log-odds weights from labeled examples.

        :param examples: List[Tuple[str, str]] - (query_text, class_label) pairs.
        :return: Dict[str, Dict[str, float]] - {ngram: {class: log_odds_weight}}.
        :raises ConfigurationError: When fewer than min_examples_per_class exist for any class.
        """
        if not examples:
            raise ConfigurationError("NgramTrainer.train requires at least one example")
        class_counts: Counter = Counter(label for _, label in examples)
        classes = sorted(class_counts.keys())
        if len(classes) < 2:
            raise ConfigurationError("NgramTrainer.train requires at least 2 distinct class labels")
        for cls in classes:
            if class_counts[cls] < self._config.min_examples_per_class:
                raise ConfigurationError(
                    f"NgramTrainer: class '{cls}' has {class_counts[cls]} examples, "
                    f"need >= {self._config.min_examples_per_class}"
                )
        ng_cls_count: Dict[str, Dict[str, int]] = {}
        for text, label in examples:
            for ng in set(self._extract_ngrams(text)):
                if ng not in ng_cls_count:
                    ng_cls_count[ng] = {}
                ng_cls_count[ng][label] = ng_cls_count[ng].get(label, 0) + 1
        total_per_cls = {cls: int(class_counts[cls]) for cls in classes}
        retained_per_cls: Dict[str, List[Tuple[str, float]]] = {}
        sm = float(self._config.smoothing)
        for cls in classes:
            n_cls = total_per_cls[cls]
            n_not_cls = sum(total_per_cls[c] for c in classes if c != cls)
            scored: List[Tuple[str, float]] = []
            for ng, counts in ng_cls_count.items():
                cnt_cls = counts.get(cls, 0)
                cnt_not = sum(counts.get(c, 0) for c in classes if c != cls)
                p_given_cls = (cnt_cls + sm) / (n_cls + sm * 2)
                p_given_not = (cnt_not + sm) / (n_not_cls + sm * 2)
                log_odds = log(p_given_cls / p_given_not)
                if log_odds >= self._config.min_log_odds:
                    scored.append((ng, log_odds))
            scored.sort(key=lambda x: x[1], reverse=True)
            retained_per_cls[cls] = scored[: self._config.top_k_per_class]
            logger.info(f"ngram_trainer class={cls} retained={len(retained_per_cls[cls])} top_ngram={scored[0][0] if scored else 'none'}")
        weights: Dict[str, Dict[str, float]] = {}
        for cls, entries in retained_per_cls.items():
            for ng, w in entries:
                if ng not in weights:
                    weights[ng] = {}
                weights[ng][cls] = round(float(w), 6)
        return weights

    def save(self, weights: Dict[str, Dict[str, float]], classes: List[str], output_path: str) -> None:
        """Serialize weights to JSON.

        :param weights: Dict[str, Dict[str, float]] - Output of train().
        :param classes: List[str] - Ordered class labels.
        :param output_path: str - Destination file path (parent dirs created if needed).
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'format_version': 1,
            'classes': classes,
            'weights': weights,
        }
        with open(out, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, ensure_ascii=False, separators=(',', ':'))
        logger.info(f"ngram_trainer_saved path={output_path} vocab_size={len(weights)} classes={classes}")


def _load_jsonl(path: str) -> List[Tuple[str, str]]:
    examples: List[Tuple[str, str]] = []
    with open(path, 'r', encoding='utf-8') as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConfigurationError(f"ngram_trainer: malformed JSON on line {line_no}: {exc}") from exc
            if 'text' not in obj or 'label' not in obj:
                raise ConfigurationError(f"ngram_trainer: line {line_no} missing 'text' or 'label' key")
            examples.append((str(obj['text']), str(obj['label'])))
    return examples


def examples_from_dataset(dataset: Any) -> List[Tuple[str, str]]:
    """Flatten a ``RouterSeedDataset`` into (query, archetype) example pairs.

    Each archetype name is the class label; each seed query is one example. Kept
    dataset-typed (not path-typed) so callers that already hold a loaded dataset
    — e.g. the head trainer co-training on the same in-memory corpus — reuse it
    without re-reading the YAML.

    :param dataset: RouterSeedDataset - Loaded seed dataset (duck-typed: needs
        ``archetypes()`` and ``texts(archetype)``).
    :return: List[Tuple[str, str]] - (query_text, archetype_label) pairs.
    """
    examples: List[Tuple[str, str]] = []
    for archetype in dataset.archetypes():
        examples.extend((text, archetype) for text in dataset.texts(archetype))
    return examples


def _load_router_seeds(seeds_path: str, min_seeds_per_archetype: int) -> List[Tuple[str, str]]:
    """Load (query, archetype) example pairs from the Semantic Router seed YAML.

    Reuses ``RouterSeedLoader`` so the ngram gate trains on the same validated,
    de-duplicated corpus as the SemanticRouter.

    :param seeds_path: str - Path to the router seed YAML (e.g. qi/router_seeds.yaml).
    :param min_seeds_per_archetype: int - Per-archetype floor enforced by the loader.
    :return: List[Tuple[str, str]] - (query_text, archetype_label) pairs.
    """
    from semantic_search.qi.seed_loader import RouterSeedLoader

    resolved = str(Path(seeds_path).resolve())
    dataset = RouterSeedLoader(resolved, min_seeds_per_archetype).load()
    return examples_from_dataset(dataset)


def train_and_save_from_examples(
    examples: List[Tuple[str, str]],
    output_path: str,
    *,
    top_k: int = _DEFAULT_TOP_K,
    min_log_odds: float = _DEFAULT_MIN_LOG_ODDS,
    smoothing: float = _DEFAULT_SMOOTHING,
    min_examples: int = _DEFAULT_MIN_EXAMPLES,
    max_ngram_order: int = 2,
) -> Dict[str, Any]:
    """Train log-odds weights from labeled examples and serialize to JSON.

    Single reusable entry point shared by the CLI and programmatic co-training
    callers (e.g. the head trainer) so both paths produce identical artifacts.

    :param examples: List[Tuple[str, str]] - (query_text, class_label) pairs.
    :param output_path: str - Destination weights JSON path.
    :return: Dict[str, Any] - Summary {'vocab_size', 'classes', 'output_path'}.
    :raises ConfigurationError: When the corpus fails NgramTrainer validation.
    """
    config = NgramTrainerConfig(
        top_k_per_class=top_k,
        min_log_odds=min_log_odds,
        smoothing=smoothing,
        min_examples_per_class=min_examples,
        max_ngram_order=max_ngram_order,
    )
    trainer = NgramTrainer(config)
    weights = trainer.train(examples)
    classes = sorted(set(label for _, label in examples))
    trainer.save(weights, classes, output_path)
    return {'vocab_size': len(weights), 'classes': classes, 'output_path': output_path}


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train NgramPreGate weights from a labeled corpus.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--seeds', help="Path to the router seed YAML (e.g. qi/router_seeds.yaml)")
    src.add_argument('--input', help="Path to labeled JSONL file ({'text','label'} per line)")
    p.add_argument('--min-seeds-per-archetype', type=int, default=30, help="Per-archetype floor enforced when loading --seeds (default: 30)")
    p.add_argument('--output', required=True, help="Path to write weights JSON")
    p.add_argument('--top-k', type=int, required=True, help="Top-K n-grams to retain per class")
    p.add_argument('--min-log-odds', type=float, required=True, help="Minimum log-odds to retain a (ngram, class) pair")
    p.add_argument('--smoothing', type=float, required=True, help="Laplace smoothing constant")
    p.add_argument('--min-examples', type=int, required=True, help="Minimum examples per class")
    p.add_argument('--max-ngram-order', type=int, required=True, choices=[1, 2], help="Max n-gram order (1 or 2)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.seeds:
        examples = _load_router_seeds(args.seeds, args.min_seeds_per_archetype)
        logger.info(f"ngram_trainer_loaded source=seeds path={args.seeds} examples={len(examples)}")
    else:
        examples = _load_jsonl(args.input)
        logger.info(f"ngram_trainer_loaded source=jsonl path={args.input} examples={len(examples)}")
    train_and_save_from_examples(
        examples,
        args.output,
        top_k=args.top_k,
        min_log_odds=args.min_log_odds,
        smoothing=args.smoothing,
        min_examples=args.min_examples,
        max_ngram_order=args.max_ngram_order,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
