"""QI deep-learning head trainer — residual MLP on encoder embeddings.

Layer rules (per ``architecture.mdc``): stdlib + ``numpy`` + ``core`` +
``contracts`` + ``config`` + sibling QI primitives. No retrieval /
orchestration / LLM imports.
"""
from __future__ import annotations

import argparse, io, json, math, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

try:
    import torch, torch.nn as nn, torch.utils.data as data_utils
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    data_utils = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

try:
    from sklearn.metrics import f1_score, recall_score
    _SKLEARN_METRICS = True
except ImportError:
    f1_score = None  # type: ignore[assignment]
    recall_score = None  # type: ignore[assignment]
    _SKLEARN_METRICS = False

from semantic_search.config.loader import load_config
from semantic_search.config.models import AgentSearchConfig
from semantic_search.contracts import QUERY_TYPES
from semantic_search.core.exceptions import ValidationError
from semantic_search.core.logging_utils import get_logger
from semantic_search.qi.training.head_trainer import HeadTrainingMetadata

logger = get_logger(__name__)

_ARTIFACT_KIND_DEEP = 'qi_deep_head_v1'


# ---------------------------------------------------------------------------
# PyTorch model (only instantiated when torch is available)
# ---------------------------------------------------------------------------

def _build_deep_mlp(input_dim, hidden_dims, num_labels, dropout):
    """Build residual MLP: input → (hidden_block × n) → linear classifier."""
    if not _TORCH_AVAILABLE:
        raise ValidationError("DeepHead requires torch — install torch")

    class _MLPBlock(nn.Module):
        def __init__(self, in_d, out_d, dp):
            super().__init__()
            self.fc = nn.Linear(in_d, out_d)
            self.norm = nn.LayerNorm(out_d)
            self.act = nn.GELU()
            self.drop = nn.Dropout(dp)
            self.proj = nn.Linear(in_d, out_d, bias=False) if in_d != out_d else None

        def forward(self, x):
            h = self.drop(self.act(self.norm(self.fc(x))))
            res = self.proj(x) if self.proj is not None else x
            return h + res

    class _ResidualMLP(nn.Module):
        def __init__(self):
            super().__init__()
            dims = [input_dim] + list(hidden_dims)
            self.blocks = nn.ModuleList([_MLPBlock(dims[i], dims[i + 1], dropout) for i in range(len(dims) - 1)])
            self.head = nn.Linear(dims[-1], num_labels)

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return self.head(x)

    return _ResidualMLP()


# ---------------------------------------------------------------------------
# Inference object — used by SemanticRouter when learned_head.kind == 'deep'
# ---------------------------------------------------------------------------

class DeepHead:
    """Residual-MLP head over mean-pooled encoder embeddings.

    Implements the same interface as :class:`LearnedHead` and :class:`SklearnHead`
    so :class:`SemanticRouter` can swap scorers without any call-site changes.

    :param model_bytes: Torch state_dict serialised to bytes via ``torch.save``.
    :param classes: Sorted archetype name list (length K, every entry in QUERY_TYPES).
    :param encoder_dim: Expected encoder output dimension.
    :param hidden_dims: MLP block sizes (list of ints).
    :param dropout: Dropout probability used at training time (ignored at inference).
    :param metadata: Training provenance.
    """

    def __init__(self, model_bytes: bytes, classes: List[str], encoder_dim: int, hidden_dims: List[int], dropout: float, metadata: HeadTrainingMetadata) -> None:
        if not _TORCH_AVAILABLE:
            raise ValidationError("DeepHead requires torch — install torch")
        for c in classes:
            if c not in QUERY_TYPES:
                raise ValidationError(f"DeepHead.classes entry '{c}' not in QUERY_TYPES")
        if int(encoder_dim) < 4:
            raise ValidationError("DeepHead.encoder_dim must be >= 4")
        self._classes = list(classes)
        self._encoder_dim = int(encoder_dim)
        self._hidden_dims = list(hidden_dims)
        self._dropout = float(dropout)
        self._metadata = metadata
        self._model = _build_deep_mlp(encoder_dim, hidden_dims, len(classes), dropout)
        state = torch.load(io.BytesIO(model_bytes), map_location="cpu", weights_only=True)  # nosemgrep
        self._model.load_state_dict(state)
        self._model.eval()

    @property
    def classes(self) -> List[str]: return list(self._classes)

    @property
    def encoder_dim(self) -> int: return self._encoder_dim

    @property
    def metadata(self) -> HeadTrainingMetadata: return self._metadata

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities for a batch of query vectors.

        :param X: np.ndarray shape (n, d) or (d,)
        :return: np.ndarray shape (n, K) — probability per class, rows sum to 1.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self._encoder_dim:
            raise ValidationError(f"DeepHead.predict_proba X.shape[1]={X.shape[1]} != encoder_dim={self._encoder_dim}")
        with torch.no_grad():
            logits = self._model(torch.tensor(X, dtype=torch.float32))
            probs = torch.softmax(logits, dim=-1).numpy()
        return probs

    def score_archetypes(self, query_vec: List[float]) -> List[Tuple[str, float]]:
        """Return descending-sorted [(archetype, probability)] for a single query vector."""
        arr = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        probs = self.predict_proba(arr)[0]
        return sorted(zip(self._classes, probs.tolist()), key=lambda kv: kv[1], reverse=True)

    def save(self, path: Path) -> None:
        """Serialise to .npz — torch state_dict as bytes, config inline."""
        if not isinstance(path, Path):
            raise ValidationError("DeepHead.save requires a pathlib.Path")
        path.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        torch.save(self._model.state_dict(), buf)
        np.savez(
            path,
            kind=np.array(_ARTIFACT_KIND_DEEP, dtype=str),
            classes=np.array(self._classes, dtype=str),
            encoder_dim=np.array(self._encoder_dim, dtype=np.int64),
            hidden_dims=np.array(self._hidden_dims, dtype=np.int64),
            dropout=np.array(self._dropout, dtype=np.float64),
            model_bytes=np.frombuffer(buf.getvalue(), dtype=np.uint8),
            metadata_json=np.array(json.dumps(self._metadata.to_dict(), sort_keys=True), dtype=str),
        )
        logger.info(f"deep_head_saved path={path} classes={self._classes} encoder_dim={self._encoder_dim}")

    @classmethod
    def load(cls, path: Path) -> 'DeepHead':
        """Load from .npz produced by :meth:`save`."""
        if not isinstance(path, Path):
            raise ValidationError("DeepHead.load requires a pathlib.Path")
        if not path.exists():
            raise ValidationError(f"DeepHead.load: path does not exist: {path}")
        data = np.load(path, allow_pickle=False)
        kind = str(data['kind'])
        if kind != _ARTIFACT_KIND_DEEP:
            raise ValidationError(f"DeepHead.load: kind={kind!r} != {_ARTIFACT_KIND_DEEP!r}")
        classes = [str(c) for c in data['classes']]
        encoder_dim = int(data['encoder_dim'])
        hidden_dims = data['hidden_dims'].tolist()
        dropout = float(data['dropout'])
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
        return cls(model_bytes=model_bytes, classes=classes, encoder_dim=encoder_dim, hidden_dims=hidden_dims, dropout=dropout, metadata=metadata)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _get_device():
    if _TORCH_AVAILABLE and torch.backends.mps.is_available():
        return torch.device("mps")
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu") if _TORCH_AVAILABLE else None


def _hmean(a: float, b: float) -> float:
    return 2.0 * a * b / (a + b) if (a + b) > 0.0 else 0.0


def _val_metrics(model, X_va_t, y_va_int, classes_sorted, batch_size, device):
    """Compute macro_f1, min_recall, hmean score on validation set."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_va_t), batch_size):
            xb = X_va_t[i:i + batch_size].to(device)
            preds.append(model(xb).argmax(dim=-1).cpu().numpy())
    y_pred = np.concatenate(preds)
    id2lbl = {i: c for i, c in enumerate(classes_sorted)}
    y_true_str = [id2lbl[v] for v in y_va_int]
    y_pred_str = [id2lbl[v] for v in y_pred]
    mf1 = float(f1_score(y_true_str, y_pred_str, average="macro", labels=classes_sorted, zero_division=0))
    per_r = recall_score(y_true_str, y_pred_str, average=None, labels=classes_sorted, zero_division=0)
    min_r = float(np.asarray(per_r).min())
    return mf1, min_r, _hmean(mf1, min_r), float(np.mean(y_pred == y_va_int))


def train_deep_head(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray, classes_sorted: List[str], cfg: Dict[str, Any]) -> Tuple[DeepHead, HeadTrainingMetadata]:
    """Train a DeepHead on mean-pooled embeddings.

    :param X_tr: (N_train, encoder_dim) float32 embeddings
    :param y_tr: (N_train,) int64 class indices
    :param X_va: (N_val, encoder_dim) float32 embeddings
    :param y_va: (N_val,) int64 class indices
    :param classes_sorted: sorted archetype name list
    :param cfg: deep_training section from QIConfig (already-parsed dict or dataclass.to_dict())
    :return: (DeepHead, HeadTrainingMetadata)
    """
    if not _TORCH_AVAILABLE:
        raise ValidationError("train_deep_head requires torch")
    if not _SKLEARN_METRICS:
        raise ValidationError("train_deep_head requires scikit-learn for metrics")

    encoder_dim = X_tr.shape[1]
    num_labels = len(classes_sorted)
    hidden_dims = [int(d) for d in cfg["hidden_dims"]]
    dropout = float(cfg["dropout"])
    lr = float(cfg["lr"])
    weight_decay = float(cfg["weight_decay"])
    label_smoothing = float(cfg["label_smoothing"])
    batch_size = int(cfg["batch_size"])
    epochs = int(cfg["epochs"])
    patience = int(cfg["patience"])
    lr_patience = int(cfg["lr_patience"])
    lr_factor = float(cfg["lr_factor"])
    min_lr = float(cfg["min_lr"])
    max_grad_norm = float(cfg["max_grad_norm"])
    seed = int(cfg["seed"])

    torch.manual_seed(seed)
    device = _get_device()

    model = _build_deep_mlp(encoder_dim, hidden_dims, num_labels, dropout).to(device)

    counts = np.bincount(y_tr, minlength=num_labels).astype(float)
    cw = torch.tensor((counts.sum() / (num_labels * counts)).clip(0.1, 10.0), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=lr_patience, factor=lr_factor, min_lr=min_lr)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    X_va_t = torch.tensor(X_va, dtype=torch.float32)

    pin_mem = device.type == "cuda"
    loader = data_utils.DataLoader(data_utils.TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True, pin_memory=pin_mem, num_workers=0)

    best_score, best_ep, no_improve = -1.0, 0, 0
    best_state, best_loss = None, float("inf")

    for epoch in range(epochs):
        model.train()
        ep_loss = 0.0
        for xb, yb in loader:
            if device.type == "mps":
                torch.mps.empty_cache()
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            ep_loss += float(loss.item()) * len(xb)
        ep_loss /= len(X_tr)

        mf1, min_r, score, acc = _val_metrics(model, X_va_t, y_va, classes_sorted, batch_size, device)
        scheduler.step(1.0 - score)
        logger.info(f"deep_head_epoch epoch={epoch+1} score={score:.4f} f1={mf1:.4f} min_recall={min_r:.4f} acc={acc:.4f} loss={ep_loss:.4f}")

        if score > best_score:
            best_score, best_ep, no_improve, best_loss = score, epoch + 1, 0, ep_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"deep_head_early_stop epoch={epoch+1} best_epoch={best_ep}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    buf = io.BytesIO()
    torch.save(best_state or model.state_dict(), buf)  # nosemgrep
    model_bytes = buf.getvalue()

    mf1_final, _, _, acc_final = _val_metrics(model, X_va_t, y_va, classes_sorted, batch_size, device)
    id2lbl = {i: c for i, c in enumerate(classes_sorted)}
    y_va_str = [id2lbl[v] for v in y_va]
    X_va_cpu = X_va_t
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_va_cpu), batch_size):
            preds.append(model(X_va_cpu[i:i + batch_size].to(device)).argmax(dim=-1).cpu().numpy())
    y_pred = np.concatenate(preds)
    y_pred_str = [id2lbl[v] for v in y_pred]
    per_r = recall_score(y_va_str, y_pred_str, average=None, labels=classes_sorted, zero_division=0)
    f1_per = {c: float(f1_score(y_va_str, y_pred_str, average=None, labels=classes_sorted, zero_division=0)[i]) for i, c in enumerate(classes_sorted)}

    samples_per_class = {c: int((y_tr == i).sum()) for i, c in enumerate(classes_sorted)}
    metadata = HeadTrainingMetadata(
        encoder_dim=encoder_dim,
        classes=classes_sorted,
        total_samples=len(X_tr),
        samples_per_class=samples_per_class,
        holdout_fraction=float(len(X_va)) / (len(X_tr) + len(X_va)),
        f1_macro_holdout=round(mf1_final, 4),
        f1_per_class_holdout=f1_per,
        accuracy_holdout=round(acc_final, 4),
        epochs_run=best_ep,
        final_loss=round(best_loss, 6),
        l2_penalty=weight_decay,
        learning_rate=lr,
        random_seed=seed,
        source_seeds_path=str(cfg.get("seeds_path", "")),
        source_hard_negatives_path="",
        created_at=float(time.time()),
    )
    head = DeepHead(model_bytes=model_bytes, classes=classes_sorted, encoder_dim=encoder_dim, hidden_dims=hidden_dims, dropout=dropout, metadata=metadata)
    return head, metadata


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _load_seeds_as_arrays(seeds_path: Path, encoder, classes_sorted: List[str], holdout_fraction: float, seed: int):
    """Load router_seeds.yaml, encode all samples, return train/val arrays."""
    import random
    with open(seeds_path) as f:
        raw = yaml.safe_load(f)
    label2id = {c: i for i, c in enumerate(classes_sorted)}
    texts_by_class: Dict[str, List[str]] = {c: [] for c in classes_sorted}
    for label, items in raw.get("archetypes", {}).items():
        if label not in label2id:
            continue
        for item in (items or []):
            if isinstance(item, str) and item.strip():
                texts_by_class[label].append(item.strip())

    rng = random.Random(seed)
    tr_texts, tr_labels, va_texts, va_labels = [], [], [], []
    for label, texts in texts_by_class.items():
        shuffled = list(texts)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * holdout_fraction))
        va_texts.extend(shuffled[:n_val])
        va_labels.extend([label2id[label]] * n_val)
        tr_texts.extend(shuffled[n_val:])
        tr_labels.extend([label2id[label]] * (len(shuffled) - n_val))

    logger.info(f"deep_head_data train={len(tr_texts)} val={len(va_texts)}")
    all_texts = tr_texts + va_texts
    all_vecs = encoder.encode_batch(all_texts)
    all_arr = np.array(all_vecs, dtype=np.float32)
    X_tr = all_arr[:len(tr_texts)]
    X_va = all_arr[len(tr_texts):]
    return X_tr, np.array(tr_labels, dtype=np.int64), X_va, np.array(va_labels, dtype=np.int64)


def main() -> None:
    p = argparse.ArgumentParser(description="Train DeepHead on QI router seeds and save to pretrained/semantic_router/")
    p.add_argument("--config", default=None, help="Path to base.yaml; defaults to package config")
    p.add_argument("--pretrained-dir", default=None, help="Override LOCAL_PRETRAINED_DIR")
    p.add_argument("--seeds-path", default=None, help="Override qi.seeds_path")
    p.add_argument("--out-path", default=None, help="Explicit output .npz path (overrides pretrained-dir)")
    args = p.parse_args()

    cfg: AgentSearchConfig = load_config(config_path=args.config)
    qi_cfg = cfg.qi
    deep_cfg_raw = qi_cfg.deep_training
    if deep_cfg_raw is None:
        raise ValueError("qi.deep_training config section missing from base.yaml")
    deep_cfg = vars(deep_cfg_raw) if hasattr(deep_cfg_raw, '__dataclass_fields__') else dict(deep_cfg_raw)

    seeds_path = Path(args.seeds_path or qi_cfg.seeds_path)
    classes_sorted = sorted(str(c) for c in QUERY_TYPES)
    holdout_fraction = float(deep_cfg.get("holdout_fraction", 0.1))
    seed = int(deep_cfg.get("seed", 42))

    from semantic_search.registry import build_subsystems
    import asyncio
    sub = asyncio.run(build_subsystems(cfg))
    encoder = sub.qi_engine._router._encoder

    logger.info(f"deep_head_cli encoding seeds seeds_path={seeds_path}")
    X_tr, y_tr, X_va, y_va = _load_seeds_as_arrays(seeds_path, encoder, classes_sorted, holdout_fraction, seed)

    logger.info("deep_head_cli training")
    deep_cfg["seeds_path"] = str(seeds_path)
    head, meta = train_deep_head(X_tr, y_tr, X_va, y_va, classes_sorted, deep_cfg)

    pretrained_dir = Path(args.pretrained_dir or "").expanduser() if args.pretrained_dir else None
    if args.out_path:
        out_path = Path(args.out_path)
    elif pretrained_dir:
        out_path = pretrained_dir / "semantic_router" / "semantic_head_v1.npz"
    else:
        import os
        base_pretrained = os.environ.get("LOCAL_PRETRAINED_DIR", "/app/pretrained")
        out_path = Path(base_pretrained) / "semantic_router" / "semantic_head_v1.npz"

    head.save(out_path)
    logger.info(f"deep_head_cli saved score={_hmean(meta.f1_macro_holdout, float(min(meta.f1_per_class_holdout.values()))):.4f} path={out_path}")


if __name__ == "__main__":
    main()
