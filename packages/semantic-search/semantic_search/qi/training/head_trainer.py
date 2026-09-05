"""QI learned-head trainer — logistic regression, SVM, gradient-boosting.

Layer rules (per ``architecture.mdc``): stdlib + ``numpy`` + ``core`` +
``contracts`` + ``config`` + sibling QI primitives. No retrieval /
orchestration / LLM imports.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from semantic_search.config.loader import load_config
from semantic_search.config.models import AgentSearchConfig
from semantic_search.contracts import QUERY_TYPES, RouterSeedDataset
from semantic_search.core.exceptions import ValidationError
from semantic_search.core.logging_utils import get_logger
from semantic_search.qi.cascade_encoder import MatryoshkaCascadeEncoder
from semantic_search.qi.encoder import Encoder
from semantic_search.qi.seed_loader import RouterSeedLoader, dataset_from_inline
from semantic_search.qi.stage_encoder import StageEncoder
from semantic_search.qi.training.data_schema import HardNegativeRow
from semantic_search.qi.training.hard_negative_miner import read_artifact_jsonl
from semantic_search.qi.training.ngram_trainer import (
    examples_from_dataset,
    train_and_save_from_examples,
)

try:
    import joblib as _joblib
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    _SKLEARN_AVAILABLE = True
except ImportError:
    _joblib = None
    GradientBoostingClassifier = None
    LogisticRegression = None
    SVC = None
    _SKLEARN_AVAILABLE = False

logger = get_logger(__name__)

_DEFAULT_LEARNING_RATE = 0.5
_DEFAULT_L2 = 1e-2  # Increased from 1e-4 for stronger regularization, prevents overfitting
_DEFAULT_MAX_EPOCHS = 400
_DEFAULT_TOLERANCE = 1e-5
_ARTIFACT_KIND = 'qi_logistic_head_v1'
_ARTIFACT_KIND_SVM = 'qi_svm_head_v1'
_ARTIFACT_KIND_GB = 'qi_gb_head_v1'
_SKLEARN_HEAD_KINDS = frozenset({_ARTIFACT_KIND_SVM, _ARTIFACT_KIND_GB})


@dataclass(frozen=True)
class HeadTrainerConfig:
    """Hyperparameters for one trainer run.

    :param l2_penalty: float - L2 regularisation strength (>= 0).
    :param learning_rate: float - Initial learning rate for batch gradient
        descent (> 0). Halved when the validation loss plateaus.
    :param max_epochs: int - Hard cap on training iterations.
    :param tolerance: float - Stop when ``|loss_t - loss_{t-1}| <
        tolerance`` for ``patience`` consecutive epochs.
    :param patience: int - Plateau detection window (>= 1).
    :param holdout_fraction: float - Held-out fraction in (0, 0.5] for the
        f1_macro report. Stratified by class label for balanced reporting.
    :param random_seed: int - Deterministic shuffle / split seed.
    """
    l2_penalty: float = _DEFAULT_L2
    learning_rate: float = _DEFAULT_LEARNING_RATE
    max_epochs: int = _DEFAULT_MAX_EPOCHS
    tolerance: float = _DEFAULT_TOLERANCE
    patience: int = 3  # Reduced from 5 for earlier stopping, catches overfitting faster
    holdout_fraction: float = 0.1
    random_seed: int = 0

    def __post_init__(self) -> None:
        if float(self.l2_penalty) < 0.0:
            raise ValidationError("HeadTrainerConfig.l2_penalty must be >= 0")
        if float(self.learning_rate) <= 0.0:
            raise ValidationError("HeadTrainerConfig.learning_rate must be > 0")
        if int(self.max_epochs) < 1:
            raise ValidationError("HeadTrainerConfig.max_epochs must be >= 1")
        if float(self.tolerance) < 0.0:
            raise ValidationError("HeadTrainerConfig.tolerance must be >= 0")
        if int(self.patience) < 1:
            raise ValidationError("HeadTrainerConfig.patience must be >= 1")
        if not 0.0 < float(self.holdout_fraction) <= 0.5:
            raise ValidationError("HeadTrainerConfig.holdout_fraction must be in (0, 0.5]")


@dataclass(frozen=True)
class HeadTrainingMetadata:
    """Provenance + quality metadata embedded alongside the head artefact."""
    encoder_dim: int
    classes: List[str]
    total_samples: int
    samples_per_class: Dict[str, int]
    holdout_fraction: float
    f1_macro_holdout: float
    f1_per_class_holdout: Dict[str, float]
    accuracy_holdout: float
    epochs_run: int
    final_loss: float
    l2_penalty: float
    learning_rate: float
    random_seed: int
    source_seeds_path: str
    source_hard_negatives_path: str
    created_at: float

    def __post_init__(self) -> None:
        if int(self.encoder_dim) < 4:
            raise ValidationError("HeadTrainingMetadata.encoder_dim must be >= 4")
        for c in self.classes:
            if c not in QUERY_TYPES:
                raise ValidationError(f"HeadTrainingMetadata.classes entry '{c}' not in QUERY_TYPES")
        if int(self.total_samples) < 1:
            raise ValidationError("HeadTrainingMetadata.total_samples must be >= 1")
        for k, v in self.samples_per_class.items():
            if k not in QUERY_TYPES:
                raise ValidationError(f"HeadTrainingMetadata.samples_per_class key '{k}' not in QUERY_TYPES")
            if int(v) < 0:
                raise ValidationError(f"HeadTrainingMetadata.samples_per_class['{k}']={v} must be >= 0")
        if not 0.0 <= float(self.f1_macro_holdout) <= 1.0:
            raise ValidationError("HeadTrainingMetadata.f1_macro_holdout out of [0,1]")
        if not 0.0 <= float(self.accuracy_holdout) <= 1.0:
            raise ValidationError("HeadTrainingMetadata.accuracy_holdout out of [0,1]")

    def to_dict(self) -> Dict[str, object]:
        return {
            'encoder_dim': int(self.encoder_dim),
            'classes': list(self.classes),
            'total_samples': int(self.total_samples),
            'samples_per_class': dict(self.samples_per_class),
            'holdout_fraction': float(self.holdout_fraction),
            'f1_macro_holdout': float(self.f1_macro_holdout),
            'f1_per_class_holdout': {k: float(v) for k, v in self.f1_per_class_holdout.items()},
            'accuracy_holdout': float(self.accuracy_holdout),
            'epochs_run': int(self.epochs_run),
            'final_loss': float(self.final_loss),
            'l2_penalty': float(self.l2_penalty),
            'learning_rate': float(self.learning_rate),
            'random_seed': int(self.random_seed),
            'source_seeds_path': str(self.source_seeds_path),
            'source_hard_negatives_path': str(self.source_hard_negatives_path),
            'created_at': float(self.created_at),
        }


# ---------------------------------------------------------------------------
# Inference object — used by SemanticRouter when learned_head.kind != 'centroid'
# ---------------------------------------------------------------------------

class LearnedHead:
    """Trained multinomial-logistic head over encoder embeddings.

    The model is ``softmax(query_vec @ W + b)`` where ``W`` is a (d, K)
    matrix and ``b`` is a (K,) bias. Classes are stored as an ordered list
    of strings so the output rows always map back to the same archetype
    names regardless of insertion order in the source dataset.

    :param weights: np.ndarray shape (d, K)
    :param bias: np.ndarray shape (K,)
    :param classes: List[str] - Length K, sorted, every entry in QUERY_TYPES.
    :param encoder_dim: int - Expected encoder dimension d.
    :param metadata: HeadTrainingMetadata - Audit trail.
    """

    def __init__(self, weights: np.ndarray, bias: np.ndarray, classes: List[str], encoder_dim: int, metadata: HeadTrainingMetadata):
        if not isinstance(weights, np.ndarray):
            raise ValidationError("LearnedHead.weights must be a numpy.ndarray")
        if not isinstance(bias, np.ndarray):
            raise ValidationError("LearnedHead.bias must be a numpy.ndarray")
        if weights.ndim != 2:
            raise ValidationError("LearnedHead.weights must be 2-D")
        if bias.ndim != 1 or bias.shape[0] != weights.shape[1]:
            raise ValidationError("LearnedHead.bias shape must equal (weights.shape[1],)")
        if weights.shape[0] != int(encoder_dim):
            raise ValidationError(f"LearnedHead.weights.shape[0]={weights.shape[0]} != encoder_dim={encoder_dim}")
        if len(classes) != weights.shape[1]:
            raise ValidationError(f"LearnedHead.classes len={len(classes)} != weights.shape[1]={weights.shape[1]}")
        for c in classes:
            if c not in QUERY_TYPES:
                raise ValidationError(f"LearnedHead.classes entry '{c}' not in QUERY_TYPES")
        if not isinstance(metadata, HeadTrainingMetadata):
            raise ValidationError("LearnedHead.metadata must be a HeadTrainingMetadata")
        self._weights = weights.astype(np.float64, copy=False)
        self._bias = bias.astype(np.float64, copy=False)
        self._classes = list(classes)
        self._encoder_dim = int(encoder_dim)
        self._metadata = metadata

    @property
    def classes(self) -> List[str]:
        return list(self._classes)

    @property
    def encoder_dim(self) -> int:
        return self._encoder_dim

    @property
    def metadata(self) -> HeadTrainingMetadata:
        return self._metadata

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities for a batch of query vectors.

        :param X: np.ndarray shape (n, d) or (d,)
        :return: np.ndarray shape (n, K) — probability per class, rows sum to 1.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self._encoder_dim:
            raise ValidationError(
                f"LearnedHead.predict_proba X.shape[1]={X.shape[1]} != encoder_dim={self._encoder_dim}"
            )
        logits = X @ self._weights + self._bias
        # Stable softmax: subtract per-row max before exp.
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def score_archetypes(self, query_vec: List[float]) -> List[Tuple[str, float]]:
        """Inference path the SemanticRouter calls — descending (class, prob)."""
        arr = np.asarray(query_vec, dtype=np.float64)
        if arr.ndim != 1 or arr.shape[0] != self._encoder_dim:
            raise ValidationError(
                f"LearnedHead.score_archetypes expects a 1-D vector of dim={self._encoder_dim}; "
                f"got shape={arr.shape}"
            )
        probs = self.predict_proba(arr)[0]
        scored = [(self._classes[i], float(probs[i])) for i in range(len(self._classes))]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored

    def save(self, path: Path) -> None:
        if not isinstance(path, Path):
            raise ValidationError("LearnedHead.save requires a pathlib.Path")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            kind=np.array(_ARTIFACT_KIND, dtype=str),
            weights=self._weights,
            bias=self._bias,
            classes=np.array(self._classes, dtype=str),
            encoder_dim=np.array(self._encoder_dim, dtype=np.int64),
            metadata_json=np.array(json.dumps(self._metadata.to_dict(), sort_keys=True), dtype=str),
        )
        logger.info(f"logistic_head_saved path={path} classes={self._classes} encoder_dim={self._encoder_dim}")

    @classmethod
    def load(cls, path: Path) -> 'LearnedHead':
        if not isinstance(path, Path):
            raise ValidationError("LearnedHead.load requires a pathlib.Path")
        if not path.exists():
            raise ValidationError(f"LearnedHead.load: path does not exist: {path}")
        data = np.load(path, allow_pickle=False)
        kind = str(data['kind'])
        if kind != _ARTIFACT_KIND:
            raise ValidationError(
                f"LearnedHead.load: artefact kind={kind!r} != expected {_ARTIFACT_KIND!r}"
            )
        weights = data['weights']
        bias = data['bias']
        classes = [str(c) for c in data['classes']]
        encoder_dim = int(data['encoder_dim'])
        meta_dict = json.loads(str(data['metadata_json']))
        metadata = HeadTrainingMetadata(
            encoder_dim=int(meta_dict['encoder_dim']),
            classes=[str(c) for c in meta_dict['classes']],
            total_samples=int(meta_dict['total_samples']),
            samples_per_class={str(k): int(v) for k, v in meta_dict['samples_per_class'].items()},
            holdout_fraction=float(meta_dict['holdout_fraction']),
            f1_macro_holdout=float(meta_dict['f1_macro_holdout']),
            f1_per_class_holdout={str(k): float(v) for k, v in meta_dict['f1_per_class_holdout'].items()},
            accuracy_holdout=float(meta_dict['accuracy_holdout']),
            epochs_run=int(meta_dict['epochs_run']),
            final_loss=float(meta_dict['final_loss']),
            l2_penalty=float(meta_dict['l2_penalty']),
            learning_rate=float(meta_dict['learning_rate']),
            random_seed=int(meta_dict['random_seed']),
            source_seeds_path=str(meta_dict['source_seeds_path']),
            source_hard_negatives_path=str(meta_dict['source_hard_negatives_path']),
            created_at=float(meta_dict['created_at']),
        )
        return cls(weights=weights, bias=bias, classes=classes, encoder_dim=encoder_dim, metadata=metadata)


# ---------------------------------------------------------------------------
# Sklearn head — wraps a single SVM or GradientBoosting sklearn model
# ---------------------------------------------------------------------------

class SklearnHead:
    """Joblib-serialised sklearn classifier head (svm or gradient_boosting).

    Identical inference interface to the logistic head so SemanticRouter
    can swap scorers without call-site changes.

    :param model_bytes: Joblib-serialised sklearn classifier bytes.
    :param artifact_kind: One of ``_SKLEARN_HEAD_KINDS`` — identifies model type in .npz.
    :param classes: Sorted archetype name list (every entry in QUERY_TYPES).
    :param encoder_dim: Expected encoder output dimension.
    :param metadata: Training provenance.
    """

    def __init__(self, model_bytes: bytes, artifact_kind: str, classes: List[str], encoder_dim: int, metadata: HeadTrainingMetadata) -> None:
        if not _SKLEARN_AVAILABLE:
            raise ValidationError("SklearnHead requires scikit-learn and joblib — install scikit-learn")
        if artifact_kind not in _SKLEARN_HEAD_KINDS:
            raise ValidationError(f"SklearnHead.artifact_kind must be in {_SKLEARN_HEAD_KINDS}; got {artifact_kind!r}")
        for c in classes:
            if c not in QUERY_TYPES:
                raise ValidationError(f"SklearnHead.classes entry '{c}' not in QUERY_TYPES")
        if int(encoder_dim) < 4:
            raise ValidationError("SklearnHead.encoder_dim must be >= 4")
        self._artifact_kind = artifact_kind
        self._model = _joblib.load(io.BytesIO(model_bytes))
        self._classes = list(classes)
        self._encoder_dim = int(encoder_dim)
        self._metadata = metadata

    @property
    def classes(self) -> List[str]: return list(self._classes)

    @property
    def encoder_dim(self) -> int: return self._encoder_dim

    @property
    def metadata(self) -> HeadTrainingMetadata: return self._metadata

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability matrix shape (n, K) from the sklearn model."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self._encoder_dim:
            raise ValidationError(f"SklearnHead.predict_proba X.shape[1]={X.shape[1]} != encoder_dim={self._encoder_dim}")
        return self._model.predict_proba(X)

    def score_archetypes(self, query_vec: List[float]) -> List[Tuple[str, float]]:
        """Return descending-sorted [(archetype, probability)] for a single query vector."""
        arr = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        probs = self.predict_proba(arr)[0]
        return sorted(zip(self._classes, probs.tolist()), key=lambda kv: kv[1], reverse=True)

    def save(self, path: Path) -> None:
        """Serialise to .npz — sklearn model as uint8 bytes blob."""
        if not isinstance(path, Path):
            raise ValidationError("SklearnHead.save requires a pathlib.Path")
        path.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        _joblib.dump(self._model, buf)
        np.savez(
            path,
            kind=np.array(self._artifact_kind, dtype=str),
            model_bytes=np.frombuffer(buf.getvalue(), dtype=np.uint8),
            classes=np.array(self._classes, dtype=str),
            encoder_dim=np.array(self._encoder_dim, dtype=np.int64),
            metadata_json=np.array(json.dumps(self._metadata.to_dict(), sort_keys=True), dtype=str),
        )
        _kind_label = 'svm_head_saved' if self._artifact_kind == _ARTIFACT_KIND_SVM else 'gb_head_saved'
        logger.info(f"{_kind_label} path={path} classes={self._classes} encoder_dim={self._encoder_dim}")

    @classmethod
    def load(cls, path: Path) -> 'SklearnHead':
        """Load from .npz produced by :meth:`save`."""
        if not _SKLEARN_AVAILABLE:
            raise ValidationError("SklearnHead.load requires scikit-learn and joblib")
        if not isinstance(path, Path):
            raise ValidationError("SklearnHead.load requires a pathlib.Path")
        if not path.exists():
            raise ValidationError(f"SklearnHead.load: path does not exist: {path}")
        data = np.load(path, allow_pickle=False)
        kind = str(data['kind'])
        if kind not in _SKLEARN_HEAD_KINDS:
            raise ValidationError(f"SklearnHead.load: kind={kind!r} not in {_SKLEARN_HEAD_KINDS}")
        classes = [str(c) for c in data['classes']]
        encoder_dim = int(data['encoder_dim'])
        model_bytes = data['model_bytes'].tobytes()
        meta_dict = json.loads(str(data['metadata_json']))
        metadata = HeadTrainingMetadata(
            encoder_dim=int(meta_dict['encoder_dim']),
            classes=[str(c) for c in meta_dict['classes']],
            total_samples=int(meta_dict['total_samples']),
            samples_per_class={str(k): int(v) for k, v in meta_dict['samples_per_class'].items()},
            holdout_fraction=float(meta_dict['holdout_fraction']),
            f1_macro_holdout=float(meta_dict['f1_macro_holdout']),
            f1_per_class_holdout={str(k): float(v) for k, v in meta_dict['f1_per_class_holdout'].items()},
            accuracy_holdout=float(meta_dict['accuracy_holdout']),
            epochs_run=int(meta_dict['epochs_run']),
            final_loss=float(meta_dict['final_loss']),
            l2_penalty=float(meta_dict['l2_penalty']),
            learning_rate=float(meta_dict['learning_rate']),
            random_seed=int(meta_dict['random_seed']),
            source_seeds_path=str(meta_dict['source_seeds_path']),
            source_hard_negatives_path=str(meta_dict['source_hard_negatives_path']),
            created_at=float(meta_dict['created_at']),
        )
        return cls(model_bytes=model_bytes, artifact_kind=kind, classes=classes, encoder_dim=encoder_dim, metadata=metadata)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

def _stratified_split(y: np.ndarray, holdout_fraction: float, random_seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified train/holdout indices — at least 1 holdout sample per class.

    Pure numpy (no sklearn). Returns (train_idx, holdout_idx).
    """
    rng = np.random.default_rng(random_seed)
    train_idx_list: List[int] = []
    holdout_idx_list: List[int] = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n_holdout = max(1, int(round(holdout_fraction * len(cls_idx))))
        holdout_idx_list.extend(cls_idx[:n_holdout].tolist())
        train_idx_list.extend(cls_idx[n_holdout:].tolist())
    train_idx = np.asarray(sorted(train_idx_list), dtype=np.int64)
    holdout_idx = np.asarray(sorted(holdout_idx_list), dtype=np.int64)
    return train_idx, holdout_idx


def _f1_per_class(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> Dict[int, float]:
    """Per-class F1 (numpy implementation)."""
    out: Dict[int, float] = {}
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if tp == 0 and fp == 0 and fn == 0:
            out[c] = 0.0
        else:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            out[c] = f1
    return out


class HeadTrainer:
    """Trainer for the multinomial logistic-regression head.

    :param config: HeadTrainerConfig
    :param encoder: Encoder - Same encoder type the live router uses.
    :param dataset: RouterSeedDataset - Curated seeds (positives).
    :param hard_negatives_path: Optional[Path] - JSONL emitted by the hard-negative miner.
        When provided, the trainer cross-checks that the seed corpus
        matches the labels in the file (defends against feeding a stale
        artefact). The hard-negative annotations themselves are NOT
        treated as wrong-class examples — the rows label every seed under
        its TRUE archetype. They appear as extra positives for the
        confusable archetype's centroid topology, but the multinomial
        loss already penalises misclassification, so no extra sample
        weight is applied. (A future cross-encoder reranker is the place to use
        confusable-pair structure explicitly.)
    """

    def __init__(self, config: HeadTrainerConfig, encoder: Encoder, dataset: RouterSeedDataset, hard_negatives_path: Optional[Path] = None):
        if config is None or not isinstance(config, HeadTrainerConfig):
            raise ValidationError("HeadTrainer requires a HeadTrainerConfig")
        if encoder is None or not isinstance(encoder, Encoder):
            raise ValidationError("HeadTrainer requires a non-None Encoder")
        if dataset is None or not isinstance(dataset, RouterSeedDataset):
            raise ValidationError("HeadTrainer requires a non-None RouterSeedDataset")
        if hard_negatives_path is not None and not isinstance(hard_negatives_path, Path):
            raise ValidationError("HeadTrainer.hard_negatives_path must be a pathlib.Path or None")
        self._config = config
        self._encoder = encoder
        self._dataset = dataset
        self._hard_negatives_path = hard_negatives_path

    def _build_xy(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Encode every seed; build (X, y, classes_sorted)."""
        classes_sorted = sorted(self._dataset.archetypes())
        class_to_idx = {c: i for i, c in enumerate(classes_sorted)}
        texts: List[str] = []
        labels: List[int] = []
        for archetype in classes_sorted:
            for q in self._dataset.texts(archetype):
                texts.append(q)
                labels.append(class_to_idx[archetype])
        # Optional cross-check: every (query, archetype) pair in the
        # hard-negatives file must align with the dataset.
        if self._hard_negatives_path is not None and self._hard_negatives_path.exists():
            _, hn_rows = read_artifact_jsonl(self._hard_negatives_path)
            hn_pairs = {(r.query.strip().lower(), r.archetype) for r in hn_rows}
            ds_pairs = {(t.strip().lower(), classes_sorted[labels[i]]) for i, t in enumerate(texts)}
            missing = ds_pairs - hn_pairs
            extra = hn_pairs - ds_pairs
            if missing or extra:
                logger.warning(f"head_trainer_hn_mismatch missing={len(missing)} extra={len(extra)} — regenerate hard_negatives.jsonl after editing seeds")
        vectors = self._encoder.encode_batch(texts)
        X = np.asarray(vectors, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64)
        # Defence-in-depth: drop zero vectors (the encoder failed on them).
        non_zero_mask = (np.abs(X).sum(axis=1) > 0.0)
        if not non_zero_mask.all():
            dropped = int((~non_zero_mask).sum())
            logger.warning(f"head_trainer_dropping_zero_vectors n={dropped}")
            X = X[non_zero_mask]
            y = y[non_zero_mask]
        return X, y, classes_sorted

    def _train_loop(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_classes: int,
        sample_weights: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, int, float]:
        """Batch gradient descent with L2 + plateau detection on validation loss/accuracy.

        Early stopping monitors validation loss and validation accuracy when
        X_val / y_val are supplied (the caller's holdout split). The loop
        stops when neither val_loss decreases nor val_accuracy improves by
        more than ``config.tolerance`` for ``config.patience`` consecutive
        epochs. When no validation data is supplied the loop falls back to
        monitoring training loss only (legacy behaviour, used by tests).
        """
        rng = np.random.default_rng(self._config.random_seed)
        d = X.shape[1]
        W = rng.normal(scale=0.01, size=(d, n_classes)).astype(np.float64)
        b = np.zeros(n_classes, dtype=np.float64)
        Y_one_hot = np.zeros((X.shape[0], n_classes), dtype=np.float64)
        Y_one_hot[np.arange(X.shape[0]), y] = 1.0
        sw = sample_weights.reshape(-1, 1)
        sw_total = float(sample_weights.sum())
        lr = float(self._config.learning_rate)
        l2 = float(self._config.l2_penalty)
        max_epochs = int(self._config.max_epochs)
        tol = float(self._config.tolerance)
        patience = int(self._config.patience)
        use_val = X_val is not None and y_val is not None and len(y_val) > 0
        best_val_loss = math.inf
        best_val_acc = -math.inf
        prev_train_loss = math.inf
        plateau_count = 0
        epoch = 0
        for epoch in range(1, max_epochs + 1):
            # ---- forward pass (training set) ---
            logits = X @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            log_probs = np.log(np.clip(probs, 1e-12, 1.0))
            ce = -(Y_one_hot * log_probs).sum(axis=1)
            train_loss = float((sample_weights * ce).sum() / sw_total + 0.5 * l2 * float((W * W).sum()))
            # ---- gradient step ---
            grad_logits = (probs - Y_one_hot) * sw / sw_total
            grad_W = X.T @ grad_logits + l2 * W
            grad_b = grad_logits.sum(axis=0)
            W = W - lr * grad_W
            b = b - lr * grad_b
            # ---- plateau detection on validation metrics ---
            if use_val:
                val_logits = X_val @ W + b
                val_logits -= val_logits.max(axis=1, keepdims=True)
                val_exp = np.exp(val_logits)
                val_probs = val_exp / val_exp.sum(axis=1, keepdims=True)
                val_log_probs = np.log(np.clip(val_probs, 1e-12, 1.0))
                Y_val_oh = np.zeros((len(y_val), n_classes), dtype=np.float64)
                Y_val_oh[np.arange(len(y_val)), y_val] = 1.0
                val_loss = float(-(Y_val_oh * val_log_probs).sum() / len(y_val))
                val_acc = float((val_probs.argmax(axis=1) == y_val).mean())
                # Overfitting detector: if train_loss < val_loss/3, stop early
                overfit_threshold = 0.35  # train_loss must be < 35% of val_loss to trigger
                if train_loss < overfit_threshold * val_loss and epoch > 10:
                    logger.info(f"head_trainer_overfitting_detected epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} ratio={train_loss/val_loss:.3f}")
                    plateau_count = patience
                loss_decreased = (best_val_loss - val_loss > tol) or (val_acc - best_val_acc > tol)
                if loss_decreased:
                    best_val_loss = min(best_val_loss, val_loss)
                    best_val_acc = max(best_val_acc, val_acc)
                    plateau_count = 0
                else:
                    plateau_count += 1
                    if plateau_count >= patience:
                        logger.info(f"head_trainer_converged epoch={epoch} val_loss={val_loss:.6f} val_acc={val_acc:.4f} train_loss={train_loss:.6f}")
                        break
            else:
                if abs(prev_train_loss - train_loss) < tol:
                    plateau_count += 1
                    if plateau_count >= patience:
                        logger.info(f"head_trainer_converged epoch={epoch} loss={train_loss:.6f}")
                        break
                else:
                    plateau_count = 0
            # Halve LR on repeated plateau.
            if plateau_count > 0 and plateau_count % max(1, patience // 2) == 0:
                lr = lr * 0.5
            prev_train_loss = train_loss
        final_loss = best_val_loss if (use_val and not math.isinf(best_val_loss)) else prev_train_loss
        return W, b, epoch, float(final_loss)

    def fit(self, source_seeds_path: Optional[str] = None) -> LearnedHead:
        """Train the head and return a `LearnedHead` with metadata."""
        X, y, classes_sorted = self._build_xy()
        n_classes = len(classes_sorted)
        if X.shape[0] < n_classes:
            raise ValidationError(
                f"HeadTrainer.fit: total samples={X.shape[0]} < n_classes={n_classes}"
            )
        # Class-balanced sample weights so a 295/75 hybrid/analytics split
        # doesn't drown out the smaller classes during gradient descent.
        class_counts = np.bincount(y, minlength=n_classes)
        class_weights = X.shape[0] / (n_classes * np.clip(class_counts, 1, None))
        sample_weights = class_weights[y].astype(np.float64)
        # Stratified holdout for the f1_macro report.
        train_idx, holdout_idx = _stratified_split(
            y, self._config.holdout_fraction, self._config.random_seed
        )
        X_train, y_train, sw_train = X[train_idx], y[train_idx], sample_weights[train_idx]
        X_hold, y_hold = X[holdout_idx], y[holdout_idx]
        # Train on training split; pass holdout for validation-loss early stopping.
        W, b, epochs_run, final_loss = self._train_loop(
            X_train, y_train, n_classes, sw_train,
            X_val=X_hold, y_val=y_hold,
        )
        # Holdout evaluation.
        logits_hold = X_hold @ W + b
        y_pred_hold = logits_hold.argmax(axis=1)
        f1_per_idx = _f1_per_class(y_hold, y_pred_hold, n_classes)
        f1_per_class = {classes_sorted[i]: float(f1_per_idx[i]) for i in range(n_classes)}
        f1_macro = float(np.mean(list(f1_per_class.values()))) if f1_per_class else 0.0
        accuracy = float((y_pred_hold == y_hold).mean()) if len(y_hold) > 0 else 0.0
        samples_per_class = {classes_sorted[i]: int(class_counts[i]) for i in range(n_classes)}
        metadata = HeadTrainingMetadata(
            encoder_dim=int(self._encoder.dim),
            classes=classes_sorted,
            total_samples=int(X.shape[0]),
            samples_per_class=samples_per_class,
            holdout_fraction=self._config.holdout_fraction,
            f1_macro_holdout=f1_macro,
            f1_per_class_holdout=f1_per_class,
            accuracy_holdout=accuracy,
            epochs_run=int(epochs_run),
            final_loss=float(final_loss),
            l2_penalty=float(self._config.l2_penalty),
            learning_rate=float(self._config.learning_rate),
            random_seed=int(self._config.random_seed),
            source_seeds_path=str(source_seeds_path or self._dataset.source_path),
            source_hard_negatives_path=str(self._hard_negatives_path) if self._hard_negatives_path else '',
            created_at=time.time(),
        )
        head = LearnedHead(
            weights=W, bias=b, classes=classes_sorted,
            encoder_dim=int(self._encoder.dim), metadata=metadata,
        )
        logger.info(f"head_trainer_fit_complete classes={classes_sorted} f1_macro={f1_macro:.4f} accuracy={accuracy:.4f} f1_per_class={f1_per_class} epochs_run={epochs_run} total_samples={X.shape[0]}")
        return head


# ---------------------------------------------------------------------------
# Random hyperparameter search helpers
# ---------------------------------------------------------------------------

def _random_search_trials(
    n_trials: int,
    encoder: Encoder,
    dataset: RouterSeedDataset,
    hn_path: Optional[Path],
    max_epochs: int,
    holdout_fraction: float,
    base_seed: int,
    max_time_seconds: Optional[int] = None,
    lr_min: float = 0.05,
    lr_max: float = 2.0,
    l2_min: float = 1e-4,
    l2_max: float = 5e-2,
    patience: int = 10,
    label: str = 'Logistic',
    log_interval: int = 5,
) -> 'LearnedHead':
    """Run N random (lr, l2) trials and return the best head by holdout f1_macro.

    :param label: Classifier label used in log lines (e.g. 'Logistic', 'RF', 'GB').
    :param log_interval: Emit a progress log every N trials.
    :param max_time_seconds: Wall-clock budget. Stops starting new trials once exceeded.
    """
    deadline = (time.monotonic() + max_time_seconds) if max_time_seconds is not None else None
    rng = np.random.default_rng(base_seed)
    lrs = np.exp(rng.uniform(np.log(lr_min), np.log(lr_max), size=n_trials)).tolist()
    l2s = np.exp(rng.uniform(np.log(l2_min), np.log(l2_max), size=n_trials)).tolist()
    best_head: Optional[LearnedHead] = None
    best_f1 = -1.0
    for i, (lr, l2) in enumerate(zip(lrs, l2s)):
        if deadline is not None and time.monotonic() >= deadline:
            logger.warning(f"{label}: time_budget_exceeded trial={i}/{n_trials}")
            break
        cfg = HeadTrainerConfig(
            learning_rate=float(lr),
            l2_penalty=float(l2),
            max_epochs=max_epochs,
            holdout_fraction=holdout_fraction,
            random_seed=base_seed + i,
            patience=patience,
        )
        trainer = HeadTrainer(cfg, encoder, dataset, hn_path if (hn_path and hn_path.exists()) else None)
        head = trainer.fit()
        f1 = head.metadata.f1_macro_holdout
        if (i + 1) % log_interval == 0 or i == 0:
            logger.info(f"{label}: trial={i + 1}/{n_trials} lr={lr:.4f} l2={l2:.6f} f1_macro={f1:.4f} best_so_far={max(best_f1, f1):.4f} epochs={head.metadata.epochs_run}")
        if f1 > best_f1:
            best_f1 = f1
            best_head = head
    if best_head is None:
        raise ValidationError(f"{label}: no trial completed — increase max_time_seconds or reduce max_epochs")
    logger.info(f"{label}: search_done best_f1={best_f1:.4f} lr={best_head.metadata.learning_rate:.4f} l2={best_head.metadata.l2_penalty:.6f}")
    return best_head


def _merge_golden_seeds(dataset: RouterSeedDataset, golden_path: Path) -> RouterSeedDataset:
    """Merge calibration golden seeds (golden_seeds.yaml) into router seed dataset.

    Golden seeds have `input_query` + `expected_query_type`; they are high-quality
    human-curated examples ideal as training positives. Deduplicates against existing
    router seeds before merging.
    """
    if not golden_path.exists():
        logger.warning(f"golden_seeds_not_found path={golden_path} — skipping merge")
        return dataset

    with open(golden_path) as fh:
        raw = yaml.safe_load(fh)

    cases = raw.get('cases', [])
    extra: Dict[str, List[str]] = {}
    for c in cases:
        qt = str(c.get('expected_query_type', '')).strip()
        q = str(c.get('input_query', '')).strip()
        if qt not in QUERY_TYPES or not q:
            continue
        if qt not in extra:
            extra[qt] = []
        extra[qt].append(q)

    if not extra:
        return dataset

    # Merge: existing seeds first, golden extras added if not already present.
    merged: Dict[str, List[str]] = {}
    for arch in dataset.archetypes():
        existing = list(dataset.texts(arch))
        existing_set = {t.lower().strip() for t in existing}
        for q in extra.get(arch, []):
            if q.lower().strip() not in existing_set:
                existing.append(q)
                existing_set.add(q.lower().strip())
        merged[arch] = existing

    # Include any golden archetypes not already in the router dataset.
    for arch, qs in extra.items():
        if arch not in merged:
            merged[arch] = qs

    total_added = sum(len([q for q in qs if q.lower().strip() not in {t.lower().strip() for t in dataset.texts(arch) if arch in dataset.archetypes()}]) for arch, qs in extra.items())
    logger.info(f"golden_seeds_merged golden_cases={len(cases)} added_to_training={total_added}")
    return dataset_from_inline(merged)


def _fit_sklearn_head(
    X_train: 'np.ndarray',
    y_train: 'np.ndarray',
    X_hold: 'np.ndarray',
    y_hold: 'np.ndarray',
    classes: List[str],
    classifier_type: str,
    n_trials: int,
    base_seed: int,
    max_time_seconds: Optional[int] = None,
    gb_n_estimators: Optional[List[int]] = None,
    gb_max_depths: Optional[List[int]] = None,
    gb_learning_rate: float = 0.1,
    svm_c_min: float = 0.1,  # Tightened from 0.01 (stronger L2-equiv regularization)
    svm_c_max: float = 10.0,  # Tightened from 100.0 (prevents overfitting on small datasets)
    svm_kernels: Optional[List[str]] = None,
    svm_gamma_opts: Optional[List[str]] = None,
    label: str = '',
    log_interval: int = 5,
) -> 'Tuple[bytes, float, float, Dict[str, float], str]':
    """Run random hyperparameter search for a sklearn tree or SVM classifier.

    Returns (model_bytes, f1_macro, accuracy, f1_per_class, classifier_kind).
    model_bytes is the joblib-serialised best fitted model.

    :param label: Classifier label for log lines (e.g. 'RF', 'GB').
    :param log_interval: Emit a progress log every N trials.
    :param max_time_seconds: Wall-clock budget; stops before new trials on expiry.
    :param gb_n_estimators: Candidate tree counts (from qi.training.gb_n_estimators).
    :param gb_max_depths: Candidate depths; 0 = unlimited (from qi.training.gb_max_depths).
    :param gb_learning_rate: Fixed learning rate for GradientBoostingClassifier.
    """
    if not _SKLEARN_AVAILABLE:
        raise ValidationError(
            f"scikit-learn is required for --classifier {classifier_type}. "
            "Install: pip install scikit-learn"
        )
    clf_label = label or classifier_type.upper()[:2]
    deadline = (time.monotonic() + max_time_seconds) if max_time_seconds is not None else None
    n_classes = len(classes)
    rng = np.random.default_rng(base_seed)
    _n_est_opts = gb_n_estimators if gb_n_estimators else [100, 200, 300, 500]
    _depth_opts = [(None if d == 0 else d) for d in (gb_max_depths if gb_max_depths else [3, 4, 5, 6])]
    _svm_kernels = svm_kernels if svm_kernels else ['rbf', 'poly', 'linear']
    _svm_gammas = svm_gamma_opts if svm_gamma_opts else ['scale', 'auto']
    best_clf = None
    best_f1 = -1.0
    best_acc: float = 0.0
    best_f1pc: Dict[str, float] = {}
    for trial_i in range(n_trials):
        if deadline is not None and time.monotonic() >= deadline:
            logger.warning(f"{clf_label}: time_budget_exceeded trial={trial_i}/{n_trials}")
            break
        seed_i = int(rng.integers(0, 10000))
        if classifier_type == 'svm':
            C = float(np.exp(rng.uniform(np.log(svm_c_min), np.log(svm_c_max))))
            kernel = str(rng.choice(_svm_kernels))
            gamma = str(rng.choice(_svm_gammas))
            degree = int(rng.integers(2, 5))
            clf = SVC(C=C, kernel=kernel, gamma=gamma, degree=degree, probability=True, random_state=seed_i)
        elif classifier_type == 'gradient_boosting':
            n_est = int(rng.choice(_n_est_opts))
            depth = rng.choice(_depth_opts)
            clf = GradientBoostingClassifier(
                n_estimators=n_est, max_depth=int(depth) if depth else 3,
                learning_rate=float(gb_learning_rate), random_state=seed_i,
            )
        else:
            raise ValidationError(f"{clf_label}: unknown classifier_type={classifier_type!r}")
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_hold)
        preds = proba.argmax(axis=1)
        f1pc = {classes[i]: float(_f1_per_class(y_hold, preds, n_classes)[i]) for i in range(n_classes)}
        f1 = float(np.mean(list(f1pc.values())))
        if (trial_i + 1) % log_interval == 0 or trial_i == 0:
            logger.info(f"{clf_label}: trial={trial_i + 1}/{n_trials} f1_macro={f1:.4f} best_so_far={max(best_f1, f1):.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_acc = float((preds == y_hold).mean())
            best_f1pc = f1pc
            best_clf = clf
    if best_clf is None:
        raise ValidationError(f"{clf_label}: no trial completed — increase max_time_seconds")
    logger.info(f"{clf_label}: search_done best_f1={best_f1:.4f}")
    buf = io.BytesIO()
    _joblib.dump(best_clf, buf)
    return buf.getvalue(), best_f1, best_acc, best_f1pc, classifier_type


def _train_best_sklearn(
    encoder: Encoder,
    dataset: RouterSeedDataset,
    hn_path: Optional[Path],
    X: 'np.ndarray',
    y: 'np.ndarray',
    X_hold: 'np.ndarray',
    y_hold: 'np.ndarray',
    classes: List[str],
    source_path: str,
    max_epochs: int,
    holdout_fraction: float,
    patience: int,
    base_seed: int,
    n_trials: int,
    per_budget: int,
    lr_min: float,
    lr_max: float,
    l2_min: float,
    l2_max: float,
    gb_n_estimators: List[int],
    gb_max_depths: List[int],
    gb_learning_rate: float,
    svm_c_min: float,
    svm_c_max: float,
    svm_kernels: List[str],
    svm_gamma_opts: List[str],
    log_interval: int,
    total_samples: int,
    samples_per_class: Dict[str, int],
) -> 'Any':
    """Train logistic, SVM, and GB sequentially; return best as individual head.

    Each model trains with its own per_budget wall-clock slice (= max_time // 3).
    Sequential execution avoids CPU contention — each model gets full core bandwidth during its turn.
    Only DeepHead (PyTorch) benefits from GPU/MPS; sklearn models are CPU-only regardless of device.
    Winner (highest holdout f1_macro) is returned as logistic / svm / gradient_boosting head.
    """
    X_tr, y_tr = X, y

    logger.info(f"sklearn_sequential_start classifiers=[Logistic, SVM, GB] per_budget={per_budget}s")
    logistic_head = _random_search_trials(
        n_trials=n_trials, encoder=encoder, dataset=dataset, hn_path=hn_path,
        max_epochs=max_epochs, holdout_fraction=holdout_fraction, base_seed=base_seed,
        max_time_seconds=per_budget, lr_min=lr_min, lr_max=lr_max,
        l2_min=l2_min, l2_max=l2_max, patience=patience,
        label='Logistic', log_interval=log_interval,
    )
    svm_bytes, f1_svm, acc_svm, f1pc_svm, _ = _fit_sklearn_head(
        X_tr, y_tr, X_hold, y_hold, classes, 'svm',
        n_trials=n_trials, base_seed=base_seed + 200,
        max_time_seconds=per_budget,
        gb_n_estimators=gb_n_estimators, gb_max_depths=gb_max_depths,
        gb_learning_rate=gb_learning_rate,
        svm_c_min=svm_c_min, svm_c_max=svm_c_max,
        svm_kernels=svm_kernels, svm_gamma_opts=svm_gamma_opts,
        label='SVM', log_interval=log_interval,
    )
    gb_bytes, f1_gb, acc_gb, f1pc_gb, _ = _fit_sklearn_head(
        X_tr, y_tr, X_hold, y_hold, classes, 'gradient_boosting',
        n_trials=n_trials, base_seed=base_seed + 400,
        max_time_seconds=per_budget,
        gb_n_estimators=gb_n_estimators, gb_max_depths=gb_max_depths,
        gb_learning_rate=gb_learning_rate, label='GB', log_interval=log_interval,
    )

    f1_log = logistic_head.metadata.f1_macro_holdout
    logger.info(f"sklearn_sequential_done logistic_f1={f1_log:.4f} svm_f1={f1_svm:.4f} gb_f1={f1_gb:.4f}")

    _scores = [('logistic', f1_log, logistic_head.metadata.accuracy_holdout, logistic_head.metadata.f1_per_class_holdout), ('svm', f1_svm, acc_svm, f1pc_svm), ('gb', f1_gb, acc_gb, f1pc_gb)]
    _best_name, _best_f1, _best_acc, _best_f1pc = max(_scores, key=lambda kv: kv[1])
    logger.info(f"sklearn_winner name={_best_name} f1_macro={_best_f1:.4f}")

    meta = HeadTrainingMetadata(
        encoder_dim=int(logistic_head.encoder_dim),
        classes=classes,
        total_samples=total_samples,
        samples_per_class=samples_per_class,
        holdout_fraction=holdout_fraction,
        f1_macro_holdout=_best_f1,
        f1_per_class_holdout=_best_f1pc,
        accuracy_holdout=_best_acc,
        epochs_run=logistic_head.metadata.epochs_run,
        final_loss=logistic_head.metadata.final_loss,
        l2_penalty=logistic_head.metadata.l2_penalty,
        learning_rate=logistic_head.metadata.learning_rate,
        random_seed=base_seed,
        source_seeds_path=source_path,
        source_hard_negatives_path=str(hn_path) if hn_path and hn_path.exists() else '',
        created_at=time.time(),
    )
    if _best_name == 'logistic':
        logistic_head._metadata = meta  # noqa: SLF001
        return logistic_head
    artifact_kind = _ARTIFACT_KIND_SVM if _best_name == 'svm' else _ARTIFACT_KIND_GB
    winner_bytes = svm_bytes if _best_name == 'svm' else gb_bytes
    return SklearnHead(model_bytes=winner_bytes, artifact_kind=artifact_kind, classes=classes, encoder_dim=int(logistic_head.encoder_dim), metadata=meta)


# ---------------------------------------------------------------------------
# Co-trained lexical n-gram pre-gate
# ---------------------------------------------------------------------------

def _cotrain_ngram_gate(
    cfg: AgentSearchConfig,
    dataset: RouterSeedDataset,
    project_root: Path,
) -> Optional[Dict[str, Any]]:
    """Train the log-odds n-gram pre-gate on the SAME seed corpus as the head.

    The gate is a fast, encoder-free Tier-0 classifier; co-training it here keeps
    its weights in lock-step with the SemanticRouter head whenever the head is
    retrained, off the same in-memory ``dataset`` (incl. any golden-seed merge).

    Best-effort by contract: any failure is logged and swallowed (returns None) so
    a gate problem never blocks the head training. The gate self-disables at
    startup when weights are absent.

    Output path + n-gram order come from the inference config
    (``qi.regex.ngram_pre_gate``) so the artifact lands exactly where the gate
    loads it. Relative paths resolve against the package root, matching
    ``NgramPreGate._load_weights``.

    :return: Dict | None - Training summary, or None when skipped/failed.
    """
    ng_cfg = cfg.qi.regex.ngram_pre_gate
    if ng_cfg is None or not ng_cfg.enabled:
        logger.info("ngram_cotrain_skipped reason=disabled_or_absent_in_config")
        return None
    try:
        out_path = Path(ng_cfg.model_path)
        if not out_path.is_absolute():
            out_path = (project_root / out_path).resolve()
        examples = examples_from_dataset(dataset)
        summary = train_and_save_from_examples(
            examples,
            str(out_path),
            max_ngram_order=ng_cfg.max_ngram_order,
        )
        logger.info(
            f"ngram_cotrain_done examples={len(examples)} "
            f"vocab_size={summary['vocab_size']} classes={summary['classes']} "
            f"path={out_path}"
        )
        return summary
    except Exception as exc:  # noqa: BLE001 — best-effort co-training, never fatal
        logger.warning(
            f"ngram_cotrain_failed reason=construction_failed "
            f"error_type={type(exc).__name__} error={exc}"
        )
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_main() -> int:
    """CLI: train a learned head from the live seed corpus + hard-negatives JSONL.

    Supports three modes:
      1. Default (single run): explicit --l2 + --learning-rate.
      2. --random-search: log-uniform random search over (lr, l2); keeps best.
      3. --classifier gradient_boosting|random_forest|sklearn_logistic: sklearn
         backend (requires scikit-learn installed in the training environment).

    Optional --golden-seeds merges the calibration golden_seeds.yaml into the
    training corpus before fitting.
    """
    from semantic_search.registry import _build_encoder
    parser = argparse.ArgumentParser(description="Train QI semantic router head (logistic / svm / gradient_boosting / best_sklearn).")
    parser.add_argument('--config-path', type=str, default=None, help='Path to base.yaml.')
    parser.add_argument('--hard-negatives', type=str, default='semantic_search/qi/training_data/hard_negatives.jsonl', help='Hard-negatives JSONL from the miner.')
    parser.add_argument('--output', type=str, default='semantic_search/qi/training_data/semantic_head_v1.npz', help='Output .npz path.')
    parser.add_argument('--l2', type=float, default=_DEFAULT_L2)
    parser.add_argument('--learning-rate', type=float, default=_DEFAULT_LEARNING_RATE)
    parser.add_argument('--max-epochs', type=int, default=_DEFAULT_MAX_EPOCHS)
    parser.add_argument('--holdout-fraction', type=float, default=0.1)
    parser.add_argument('--random-seed', type=int, default=0)
    # Random hyperparameter search
    parser.add_argument('--random-search', action='store_true', default=True, help='Run random search over (lr, l2); keeps best trial by holdout f1_macro. Enabled by default.')
    parser.add_argument('--no-random-search', dest='random_search', action='store_false', help='Disable random search and use --l2 / --learning-rate directly.')
    parser.add_argument('--random-search-trials', type=int, default=20, help='Number of (lr, l2) combinations to try (default: 20).')
    # These args use None as sentinel so we can distinguish "user passed a value"
    # from "user did not pass anything". Config is the source of truth; CLI overrides.
    parser.add_argument('--max-time', type=int, default=None, help='Wall-clock budget in seconds (default: qi.training.max_time_seconds from config).')
    # Alternative classifiers (require scikit-learn)
    parser.add_argument(
        '--classifier',
        choices=['logistic', 'sklearn_logistic', 'gradient_boosting', 'svm', 'ensemble'],
        default=None,
        help='Classifier backend (default: qi.training.classifier from config). best_sklearn trains logistic + SVM + GB in parallel and saves the winner.',
    )
    parser.add_argument('--log-interval', type=int, default=None, help='Log progress every N trials per classifier (default: qi.training.log_interval_trials from config).')
    # Golden seeds enrichment
    parser.add_argument('--golden-seeds', type=str, default='', help='Path to calibration golden_seeds.yaml; merged into training corpus.')
    args = parser.parse_args()

    raw = load_config(args.config_path)
    cfg = AgentSearchConfig.from_dict(raw)
    train_cfg = cfg.qi.training  # QIHeadTrainerConfig — all training defaults live here
    sem_cfg = cfg.qi.semantic

    # Resolve effective values: CLI arg wins over config when explicitly supplied.
    eff_max_epochs       = args.max_epochs       if args.max_epochs       != _DEFAULT_MAX_EPOCHS else (train_cfg.max_epochs       if train_cfg else args.max_epochs)
    eff_holdout          = args.holdout_fraction  if args.holdout_fraction != 0.1               else (train_cfg.holdout_fraction  if train_cfg else args.holdout_fraction)
    eff_trials           = args.random_search_trials                                             if args.random_search_trials != 20           else (train_cfg.random_search_trials if train_cfg else args.random_search_trials)
    eff_max_time         = args.max_time          if args.max_time         is not None           else (train_cfg.max_time_seconds  if train_cfg else 1800)
    eff_classifier       = args.classifier        if args.classifier       is not None           else (train_cfg.classifier        if train_cfg else 'best_sklearn')
    eff_patience         = train_cfg.patience if train_cfg else 10
    if not sem_cfg.seeds_path:
        raise ValidationError("qi.semantic.seeds_path must be set; the inline-prototype path is not supported by the head trainer CLI.")
    loader = RouterSeedLoader(seeds_path=sem_cfg.seeds_path, min_seeds_per_archetype=sem_cfg.min_seeds_per_archetype)
    dataset = loader.load()

    # Optionally merge calibration golden seeds into training corpus.
    if args.golden_seeds:
        golden_path = Path(args.golden_seeds)
        if not golden_path.is_absolute():
            golden_path = (Path(__file__).resolve().parents[3] / golden_path).resolve()
        dataset = _merge_golden_seeds(dataset, golden_path)

    base_encoder = _build_encoder(cfg)
    # Mirror the production wiring (registry.py): when the cascade is enabled,
    # the live SemanticRouter receives a router-stage StageEncoder sliced from
    # the Matryoshka native vector. Train at the *router stage dim* so the
    # learned head matches the dim that production inference will use.
    cascade_cfg = cfg.qi.encoder.cascade
    if cascade_cfg is not None and cascade_cfg.enabled and cascade_cfg.stage_dims is not None and cascade_cfg.stage_dims.router is not None:
        cascade = MatryoshkaCascadeEncoder(base_encoder=base_encoder, supported_dims=frozenset(cascade_cfg.supported_dims))
        encoder = StageEncoder(cascade=cascade, stage_dim=cascade_cfg.stage_dims.router, stage_name='router', emit_init_log=cfg.general.startup_log_detail)
    else:
        encoder = base_encoder
    project_root = Path(__file__).resolve().parents[3]
    hn_path = Path(args.hard_negatives)
    if not hn_path.is_absolute():
        hn_path = (project_root / hn_path).resolve()
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = (project_root / out_path).resolve()

    # ---- co-train the lexical n-gram pre-gate, in parallel -------------------
    # Runs on the SAME loaded dataset, concurrently with the (slower) embedding
    # head fit below, so retraining the router head keeps the gate weights in
    # lock-step at ~zero added wall-clock. _cotrain_ngram_gate swallows its own
    # errors, so joining never raises and never blocks head training.
    _ngram_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    _ngram_future = _ngram_pool.submit(_cotrain_ngram_gate, cfg, dataset, project_root)

    def _await_ngram() -> None:
        try:
            _ngram_future.result()
        finally:
            _ngram_pool.shutdown(wait=True)

    # ---- pre-encode seeds once (shared by all classifiers) -------------------
    _base_cfg = HeadTrainerConfig(l2_penalty=args.l2, learning_rate=args.learning_rate, max_epochs=eff_max_epochs, holdout_fraction=eff_holdout, random_seed=args.random_seed, patience=eff_patience)
    _base_trainer = HeadTrainer(_base_cfg, encoder, dataset, hn_path if hn_path.exists() else None)
    X, y, classes_sorted = _base_trainer._build_xy()  # noqa: SLF001
    train_idx, holdout_idx = _stratified_split(y, eff_holdout, args.random_seed)
    X_train, y_train = X[train_idx], y[train_idx]
    X_hold, y_hold = X[holdout_idx], y[holdout_idx]
    _spc = {classes_sorted[i]: int(np.sum(y == i)) for i in range(len(classes_sorted))}
    _log_int = train_cfg.log_interval_trials if train_cfg else 5

    # ---- best_sklearn: train logistic + SVM + GB in parallel, save winner ----
    if eff_classifier == 'best_sklearn':
        _per_budget = eff_max_time // 3
        head = _train_best_sklearn(
            encoder=encoder, dataset=dataset, hn_path=hn_path if hn_path.exists() else None,
            X=X_train, y=y_train, X_hold=X_hold, y_hold=y_hold,
            classes=classes_sorted, source_path=str(loader._resolve_path()),  # noqa: SLF001
            max_epochs=eff_max_epochs, holdout_fraction=eff_holdout,
            patience=eff_patience, base_seed=args.random_seed,
            n_trials=eff_trials, per_budget=_per_budget,
            lr_min=train_cfg.lr_min if train_cfg else 0.05,
            lr_max=train_cfg.lr_max if train_cfg else 2.0,
            l2_min=train_cfg.l2_min if train_cfg else 1e-4,
            l2_max=train_cfg.l2_max if train_cfg else 5e-2,
            gb_n_estimators=train_cfg.gb_n_estimators if train_cfg else [100, 200],
            gb_max_depths=train_cfg.gb_max_depths if train_cfg else [3, 5],
            gb_learning_rate=train_cfg.gb_learning_rate if train_cfg else 0.1,
            svm_c_min=train_cfg.svm_c_min if train_cfg else 0.01,
            svm_c_max=train_cfg.svm_c_max if train_cfg else 100.0,
            svm_kernels=train_cfg.svm_kernels if train_cfg else ['rbf', 'poly', 'linear'],
            svm_gamma_opts=train_cfg.svm_gamma_opts if train_cfg else ['scale', 'auto'],
            log_interval=_log_int,
            total_samples=int(X.shape[0]), samples_per_class=_spc,
        )
        head.save(out_path)
        print(f"best_sklearn head: classes={head.classes} encoder_dim={head.encoder_dim} f1_macro={head.metadata.f1_macro_holdout:.4f} accuracy={head.metadata.accuracy_holdout:.4f} path={out_path}")
        print(f"per_class_f1: {head.metadata.f1_per_class_holdout}")
        _await_ngram()
        return 0

    # ---- single sklearn classifier -------------------------------------------
    if eff_classifier in ('gradient_boosting', 'svm'):
        model_bytes, f1_macro, accuracy, f1_per_class, kind = _fit_sklearn_head(
            X_train, y_train, X_hold, y_hold,
            classes_sorted, eff_classifier, eff_trials, args.random_seed,
            max_time_seconds=eff_max_time,
            gb_n_estimators=train_cfg.gb_n_estimators if train_cfg else None,
            gb_max_depths=train_cfg.gb_max_depths if train_cfg else None,
            gb_learning_rate=train_cfg.gb_learning_rate if train_cfg else 0.1,
            svm_c_min=train_cfg.svm_c_min if train_cfg else 0.01,
            svm_c_max=train_cfg.svm_c_max if train_cfg else 100.0,
            svm_kernels=train_cfg.svm_kernels if train_cfg else ['rbf', 'poly', 'linear'],
            svm_gamma_opts=train_cfg.svm_gamma_opts if train_cfg else ['scale', 'auto'],
            label=eff_classifier[:2].upper(), log_interval=_log_int,
        )
        artifact_kind = _ARTIFACT_KIND_SVM if kind == 'svm' else _ARTIFACT_KIND_GB
        meta = HeadTrainingMetadata(
            encoder_dim=int(encoder.dim), classes=classes_sorted,
            total_samples=int(X.shape[0]), samples_per_class=_spc,
            holdout_fraction=eff_holdout, f1_macro_holdout=f1_macro,
            f1_per_class_holdout=f1_per_class, accuracy_holdout=accuracy,
            epochs_run=0, final_loss=0.0, l2_penalty=args.l2,
            learning_rate=args.learning_rate, random_seed=args.random_seed,
            source_seeds_path=str(loader._resolve_path()),  # noqa: SLF001
            source_hard_negatives_path=str(hn_path) if hn_path.exists() else '',
            created_at=time.time(),
        )
        sklearn_head = SklearnHead(model_bytes=model_bytes, artifact_kind=artifact_kind, classes=classes_sorted, encoder_dim=int(encoder.dim), metadata=meta)
        sklearn_head.save(out_path)
        print(f"trained head ({kind}): classes={sklearn_head.classes} encoder_dim={sklearn_head.encoder_dim} f1_macro={f1_macro:.4f} accuracy={accuracy:.4f} path={out_path}")
        print(f"per_class_f1: {f1_per_class}")
        _await_ngram()
        return 0

    # ---- pure-numpy logistic -------------------------------------------------
    if args.random_search:
        head = _random_search_trials(
            n_trials=eff_trials, encoder=encoder, dataset=dataset,
            hn_path=hn_path if hn_path.exists() else None,
            max_epochs=eff_max_epochs, holdout_fraction=eff_holdout,
            base_seed=args.random_seed, max_time_seconds=eff_max_time,
            lr_min=train_cfg.lr_min if train_cfg else 0.05,
            lr_max=train_cfg.lr_max if train_cfg else 2.0,
            l2_min=train_cfg.l2_min if train_cfg else 1e-4,
            l2_max=train_cfg.l2_max if train_cfg else 5e-2,
            patience=eff_patience, label='Logistic', log_interval=_log_int,
        )
    else:
        trainer_cfg = HeadTrainerConfig(
            l2_penalty=args.l2, learning_rate=args.learning_rate,
            max_epochs=eff_max_epochs, holdout_fraction=eff_holdout,
            random_seed=args.random_seed, patience=eff_patience,
        )
        trainer = HeadTrainer(config=trainer_cfg, encoder=encoder, dataset=dataset, hard_negatives_path=hn_path if hn_path.exists() else None)
        head = trainer.fit(source_seeds_path=str(loader._resolve_path()))  # noqa: SLF001

    head.save(out_path)
    print(f"trained head: classes={head.classes} encoder_dim={head.encoder_dim} f1_macro={head.metadata.f1_macro_holdout:.4f} accuracy={head.metadata.accuracy_holdout:.4f} path={out_path}")
    print(f"per_class_f1: {head.metadata.f1_per_class_holdout}")
    _await_ngram()
    if os.environ.get('HF_HUB_OFFLINE') != '1':
        print("warning: HF_HUB_OFFLINE != 1 — production runs should set HF_HUB_OFFLINE=1")
    return 0


if __name__ == '__main__':
    raise SystemExit(_cli_main())
