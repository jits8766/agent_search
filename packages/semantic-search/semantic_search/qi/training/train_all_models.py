#!/usr/bin/env python3
"""Train all semantic router head models with Bayesian hyperparameter optimisation.

Pipeline
--------
    1. python yaml_to_csv.py            → train.csv, valid.csv
    2. python train_all_models.py       → trains logistic / svm / gb / deep,
                                           compares, saves winner

All four models share the SAME X_train / X_valid embedding arrays (encoded once),
so holdout F1 scores are directly comparable.  Each model's hyperparameters are
found via Gaussian Process Bayesian Optimisation (GP+EI) using
``sklearn.gaussian_process`` + ``scipy.optimize`` — no Optuna required.

Deep head draws from /app/DLC/textcnn+ce:
  • Residual MLP architecture (already used by DeepHead)
  • Focal loss  (combats easy-example dominance)
  • ArcFace angular margin head  (tighter class boundaries)
  • Supervised Contrastive auxiliary loss  (cluster same-class embeddings)
  • AdamW + ReduceLROnPlateau + gradient clipping

Usage
-----
    python train_all_models.py
    python train_all_models.py --no-deep --n-trials 40 --max-time 1200
    python train_all_models.py --run-yaml-to-csv --seeds-path qi/router_seeds.yaml
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import math
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_all_models")

_TRAINING_DATA_DIR = Path(__file__).resolve().parent.parent / "training_data"
_DEFAULT_TRAIN_CSV = _TRAINING_DATA_DIR / "train.csv"
_DEFAULT_VALID_CSV = _TRAINING_DATA_DIR / "valid.csv"
_DEFAULT_OUTPUT = Path(os.environ.get("LOCAL_PRETRAINED_DIR", "/app/pretrained")) / "trained_semantic_router" / "semantic_head_v1.npz"
_DEFAULT_NGRAM_OUTPUT = Path(os.environ.get("LOCAL_PRETRAINED_DIR", "/app/pretrained")) / "ngram_pregate" / "weights.json"
_DEFAULT_N_TRIALS = 25
_DEFAULT_MAX_TIME = 720   # 12 min total → 4 × 3 min per model
_DEFAULT_HOLDOUT_FRACTION = 0.1
_DEFAULT_SEED = 42
_DEFAULT_HARD_NEGATIVES_PATH = _TRAINING_DATA_DIR / "hard_negatives.jsonl"
_DEFAULT_HARD_NEG_OVERSAMPLE = 2


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> Tuple[List[str], List[str]]:
    texts, labels = [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            texts.append(row["text"])
            labels.append(row["label"])
    return texts, labels


def build_label_maps(labels: List[str]) -> Tuple[List[str], Dict[str, int]]:
    classes = sorted(set(labels))
    return classes, {c: i for i, c in enumerate(classes)}


def encode_texts(encoder: Any, texts: List[str]) -> np.ndarray:
    return np.array(encoder.encode_batch(texts), dtype=np.float32)


# ---------------------------------------------------------------------------
# Shared metric helpers
# ---------------------------------------------------------------------------

def _f1_per_class(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if tp == fp == fn == 0:
            out[c] = 0.0
        else:
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            out[c] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return out


def _compute_metrics(
    X_va: np.ndarray,
    y_va: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    n_classes: int,
    classes: List[str],
) -> Tuple[float, float, Dict[str, float]]:
    preds = (X_va @ W + b).argmax(axis=1)
    f1_idx = _f1_per_class(y_va, preds, n_classes)
    return (
        float(np.mean(list(f1_idx.values()))),
        float((preds == y_va).mean()),
        {classes[i]: f1_idx[i] for i in range(n_classes)},
    )


# ---------------------------------------------------------------------------
# Gaussian Process Bayesian Optimisation (GP + Expected Improvement)
# ---------------------------------------------------------------------------

class BayesianOptimizer:
    """GP surrogate + EI acquisition — no Optuna required.

    Search space is a list of parameter specs:
        ('name', 'float',       lo, hi)          continuous in [lo, hi]
        ('name', 'log_float',   lo, hi)          log-scale continuous
        ('name', 'int',         lo, hi)          integer in [lo, hi]
        ('name', 'choice',      [v1, v2, ...])   categorical (index → value)

    All parameters are normalised to [0, 1]^d internally for the GP.
    """

    def __init__(self, space: List[Tuple], n_warmup: int = 5, xi: float = 0.01, seed: int = _DEFAULT_SEED):
        self.space = space
        self.n_warmup = n_warmup
        self.xi = xi
        self.rng = np.random.default_rng(seed)
        self._X: List[np.ndarray] = []
        self._y: List[float] = []
        self._gp = self._build_gp()

    @staticmethod
    def _build_gp():
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel
        kernel = Matern(nu=2.5, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-3)
        return GaussianProcessRegressor(
            kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=3,
        )

    # ---- normalise / denormalise ----

    def _to_unit(self, x: float, spec: Tuple) -> float:
        kind = spec[1]
        if kind == "float":
            lo, hi = spec[2], spec[3]
            return (x - lo) / (hi - lo)
        if kind == "log_float":
            lo, hi = math.log(spec[2]), math.log(spec[3])
            return (math.log(max(x, 1e-30)) - lo) / (hi - lo)
        if kind == "int":
            lo, hi = float(spec[2]), float(spec[3])
            return (x - lo) / (hi - lo)
        if kind == "choice":
            choices = spec[2]
            idx = choices.index(x) if x in choices else 0
            return idx / max(len(choices) - 1, 1)
        raise ValueError(f"Unknown param kind: {kind}")

    def _from_unit(self, u: float, spec: Tuple) -> Any:
        u = float(np.clip(u, 0.0, 1.0))
        kind = spec[1]
        if kind == "float":
            return spec[2] + u * (spec[3] - spec[2])
        if kind == "log_float":
            lo, hi = math.log(spec[2]), math.log(spec[3])
            return float(np.exp(lo + u * (hi - lo)))
        if kind == "int":
            lo, hi = spec[2], spec[3]
            return int(round(lo + u * (hi - lo)))
        if kind == "choice":
            choices = spec[2]
            idx = int(round(u * (len(choices) - 1)))
            return choices[min(idx, len(choices) - 1)]
        raise ValueError(f"Unknown param kind: {kind}")

    def _params_to_vec(self, params: Dict) -> np.ndarray:
        return np.array([self._to_unit(params[s[0]], s) for s in self.space])

    def _vec_to_params(self, vec: np.ndarray) -> Dict:
        return {s[0]: self._from_unit(vec[i], s) for i, s in enumerate(self.space)}

    def _random_params(self) -> Dict:
        return {s[0]: self._from_unit(float(self.rng.uniform()), s) for s in self.space}

    # ---- EI maximisation ----

    def _expected_improvement(self, X_cand: np.ndarray) -> np.ndarray:
        """Vectorised EI over a (n, d) candidate matrix."""
        from scipy.stats import norm
        if len(self._y) == 0:
            return np.zeros(len(X_cand))
        mu, sigma = self._gp.predict(X_cand, return_std=True)
        best = max(self._y)
        z = (mu - best - self.xi) / np.clip(sigma, 1e-9, None)
        ei = (mu - best - self.xi) * norm.cdf(z) + sigma * norm.pdf(z)
        ei[sigma < 1e-9] = 0.0
        return ei

    def _maximise_ei(self) -> Dict:
        from scipy.optimize import minimize
        d = len(self.space)
        best_ei, best_vec = -np.inf, self.rng.uniform(size=d)
        candidates = self.rng.uniform(size=(512, d))
        ei_vals = self._expected_improvement(candidates)
        top_idx = ei_vals.argsort()[-5:][::-1]

        for start in candidates[top_idx]:
            res = minimize(
                lambda x: -float(self._expected_improvement(x.reshape(1, -1))[0]),
                x0=start,
                method="L-BFGS-B",
                bounds=[(0.0, 1.0)] * d,
                options={"maxiter": 100},
            )
            if -res.fun > best_ei:
                best_ei = -res.fun
                best_vec = res.x
        return self._vec_to_params(np.clip(best_vec, 0.0, 1.0))

    # ---- public API ----

    def suggest(self) -> Dict:
        if len(self._X) < self.n_warmup:
            return self._random_params()
        return self._maximise_ei()

    def observe(self, params: Dict, score: float) -> None:
        self._X.append(self._params_to_vec(params))
        self._y.append(score)
        if len(self._X) >= self.n_warmup:
            X = np.array(self._X)
            y = np.array(self._y)
            self._gp.fit(X, y)

    @property
    def best(self) -> Tuple[Dict, float]:
        if not self._y:
            return {}, -np.inf
        best_idx = int(np.argmax(self._y))
        return self._vec_to_params(self._X[best_idx]), self._y[best_idx]


# ---------------------------------------------------------------------------
# Logistic regression — pure numpy, GP Bayesian search over (log_lr, log_l2)
# ---------------------------------------------------------------------------

_LOGISTIC_SPACE = [
    ("lr",  "log_float", 0.01,  5.0),
    ("l2",  "log_float", 1e-5,  0.1),
]


def train_logistic(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    classes: List[str],
    n_trials: int = _DEFAULT_N_TRIALS,
    base_seed: int = _DEFAULT_SEED,
    max_time: Optional[int] = None,
    max_epochs: int = 400,
    patience: int = 10,
) -> Dict[str, Any]:
    n_classes = len(classes)
    X_tr64 = X_tr.astype(np.float64)
    X_va64 = X_va.astype(np.float64)
    class_counts = np.bincount(y_tr, minlength=n_classes)
    class_weights = len(y_tr) / (n_classes * np.clip(class_counts, 1, None))
    sw = class_weights[y_tr].astype(np.float64)
    sw_total = float(sw.sum())
    Y_oh = np.zeros((len(y_tr), n_classes), dtype=np.float64)
    Y_oh[np.arange(len(y_tr)), y_tr] = 1.0

    bayes = BayesianOptimizer(_LOGISTIC_SPACE, n_warmup=min(5, n_trials), seed=base_seed)
    deadline = (time.monotonic() + max_time) if max_time else None
    best_meta: Optional[Dict] = None

    for trial_i in range(n_trials):
        if deadline and time.monotonic() >= deadline:
            logger.warning(f"Logistic: time budget at trial {trial_i}/{n_trials}")
            break

        hp = bayes.suggest()
        lr0 = float(hp["lr"])
        l2 = float(hp["l2"])
        trial_rng = np.random.default_rng(base_seed + trial_i)
        d = X_tr64.shape[1]
        W = trial_rng.normal(scale=0.01, size=(d, n_classes))
        b = np.zeros(n_classes)
        lr = lr0
        best_vl, no_improve, final_epoch = math.inf, 0, 0

        for epoch in range(1, max_epochs + 1):
            logits = X_tr64 @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            exp_l = np.exp(logits)
            probs = exp_l / exp_l.sum(axis=1, keepdims=True)
            g = (probs - Y_oh) * sw.reshape(-1, 1) / sw_total
            W -= lr * (X_tr64.T @ g + l2 * W)
            b -= lr * g.sum(axis=0)
            vl_logits = X_va64 @ W + b
            vl_logits -= vl_logits.max(axis=1, keepdims=True)
            vl_exp = np.exp(vl_logits)
            vl_probs = vl_exp / vl_exp.sum(axis=1, keepdims=True)
            Y_va_oh = np.zeros((len(y_va), n_classes), dtype=np.float64)
            Y_va_oh[np.arange(len(y_va)), y_va] = 1.0
            val_loss = float(-(Y_va_oh * np.log(np.clip(vl_probs, 1e-12, 1.0))).sum() / len(y_va))
            if best_vl - val_loss > 1e-5:
                best_vl = val_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
            if no_improve > 0 and no_improve % max(1, patience // 2) == 0:
                lr *= 0.5
            final_epoch = epoch

        f1_macro, acc, f1_named = _compute_metrics(X_va64, y_va, W, b, n_classes, classes)
        bayes.observe(hp, f1_macro)
        if (trial_i + 1) % 5 == 0 or trial_i == 0:
            _, best_f1 = bayes.best
            logger.info(f"Logistic: trial={trial_i+1}/{n_trials} lr={lr0:.4f} l2={l2:.6f} f1={f1_macro:.4f} best={best_f1:.4f} epochs={final_epoch}")
        if best_meta is None or f1_macro > best_meta["f1_macro"]:
            best_meta = dict(W=W.copy(), b=b.copy(), f1_macro=f1_macro, acc=acc, f1_per_class=f1_named, epochs_run=final_epoch, final_loss=best_vl, l2=l2, lr=lr0)

    if best_meta is None:
        raise RuntimeError("Logistic: no trial completed")
    logger.info(f"Logistic: best f1={best_meta['f1_macro']:.4f} acc={best_meta['acc']:.4f}")
    return best_meta


# ---------------------------------------------------------------------------
# SVM + Gradient Boosting — sklearn, GP Bayesian search
# ---------------------------------------------------------------------------

_SVM_SPACE = [
    ("log_C",    "float",  math.log(0.01), math.log(100.0)),
    ("kernel",   "choice", ["rbf", "linear", "poly"]),
    ("gamma",    "choice", ["scale", "auto"]),
]
_GB_SPACE = [
    ("n_estimators", "choice", [100, 200, 300, 500]),
    ("max_depth",    "int",    2, 8),
    ("lr",           "log_float", 0.01, 0.3),
    ("subsample",    "float",  0.5, 1.0),
]


def train_sklearn(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    classes: List[str],
    classifier_type: str,
    n_trials: int = _DEFAULT_N_TRIALS,
    base_seed: int = _DEFAULT_SEED,
    max_time: Optional[int] = None,
) -> Dict[str, Any]:
    try:
        import joblib as _joblib
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.svm import SVC
    except ImportError:
        raise RuntimeError(f"scikit-learn required for {classifier_type}. pip install scikit-learn")

    n_classes = len(classes)
    space = _SVM_SPACE if classifier_type == "svm" else _GB_SPACE
    bayes = BayesianOptimizer(space, n_warmup=min(5, n_trials), seed=base_seed)
    label_str = "SVM" if classifier_type == "svm" else "GB"
    deadline = (time.monotonic() + max_time) if max_time else None
    best_meta: Optional[Dict] = None

    for trial_i in range(n_trials):
        if deadline and time.monotonic() >= deadline:
            logger.warning(f"{label_str}: time budget at trial {trial_i}/{n_trials}")
            break
        hp = bayes.suggest()
        seed_i = int(np.random.default_rng(base_seed + trial_i).integers(0, 10000))
        if classifier_type == "svm":
            C = float(np.exp(float(hp["log_C"])))
            clf = SVC(C=C, kernel=hp["kernel"], gamma=hp["gamma"], probability=True, random_state=seed_i)
        else:
            clf = GradientBoostingClassifier(
                n_estimators=int(hp["n_estimators"]),
                max_depth=int(hp["max_depth"]),
                learning_rate=float(hp["lr"]),
                subsample=float(hp["subsample"]),
                random_state=seed_i,
            )
        clf.fit(X_tr, y_tr)
        preds = clf.predict_proba(X_va).argmax(axis=1)
        f1_idx = _f1_per_class(y_va, preds, n_classes)
        f1_macro = float(np.mean(list(f1_idx.values())))
        acc = float((preds == y_va).mean())
        f1_named = {classes[i]: f1_idx[i] for i in range(n_classes)}
        bayes.observe(hp, f1_macro)
        if (trial_i + 1) % 5 == 0 or trial_i == 0:
            _, best_f1 = bayes.best
            logger.info(f"{label_str}: trial={trial_i+1}/{n_trials} f1={f1_macro:.4f} best={best_f1:.4f}")
        if best_meta is None or f1_macro > best_meta["f1_macro"]:
            buf = io.BytesIO()
            _joblib.dump(clf, buf)
            best_meta = dict(model_bytes=buf.getvalue(), f1_macro=f1_macro, acc=acc, f1_per_class=f1_named, kind=classifier_type)

    if best_meta is None:
        raise RuntimeError(f"{label_str}: no trial completed")
    logger.info(f"{label_str}: best f1={best_meta['f1_macro']:.4f} acc={best_meta['acc']:.4f}")
    return best_meta


# ---------------------------------------------------------------------------
# Deep head — DLC-inspired ResidualMLP with focal loss, ArcFace, SupCon
# ---------------------------------------------------------------------------

_DEEP_SPACE = [
    ("hidden_dim",       "choice",    [256, 512, 1024]),
    ("num_blocks",       "int",       2, 6),
    ("dropout",          "float",     0.05, 0.5),
    ("lr",               "log_float", 1e-4, 5e-3),
    ("weight_decay",     "log_float", 1e-5, 5e-2),
    ("label_smoothing",  "float",     0.0, 0.15),
    ("focal_gamma",      "float",     0.0, 2.5),
    ("arc_m",            "float",     0.0, 0.5),   # 0 = disabled
    ("supcon_weight",    "float",     0.0, 0.5),   # 0 = disabled
    ("batch_size",       "choice",    [32, 64, 128]),
]


def _get_device():
    try:
        import torch
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    except Exception:
        return None


def _deep_trial(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    classes: List[str],
    hp: Dict,
    max_epochs: int = 60,
    patience: int = 8,
    seed: int = 42,
) -> Tuple[float, float, Dict[str, float], Any, int]:
    """Run one deep training trial. Returns (f1_macro, acc, f1_per_class, best_state_dict, best_epoch)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    device = _get_device() or torch.device("cpu")
    n_classes = len(classes)
    encoder_dim = X_tr.shape[1]
    hidden_dim = int(hp["hidden_dim"])
    num_blocks = int(hp["num_blocks"])
    dropout = float(hp["dropout"])
    lr = float(hp["lr"])
    wd = float(hp["weight_decay"])
    ls = float(hp["label_smoothing"])
    fg = float(hp["focal_gamma"])
    arc_m = float(hp["arc_m"])
    supcon_w = float(hp["supcon_weight"])
    batch_size = int(hp["batch_size"])
    use_arcface = arc_m > 0.05

    # ── build residual MLP (matches DeepHead architecture) ────────────────────
    from semantic_search.qi.training.deep_head_trainer import _build_deep_mlp
    dims = [hidden_dim] * num_blocks
    model = _build_deep_mlp(encoder_dim, dims, n_classes, dropout).to(device)

    # ── ArcFace head (DLC-inspired) ───────────────────────────────────────────
    arc_head = None
    if use_arcface:
        cos_m = math.cos(arc_m)
        sin_m = math.sin(arc_m)
        th = math.cos(math.pi - arc_m)
        mm = math.sin(math.pi - arc_m) * arc_m
        arc_W = nn.Parameter(torch.empty(n_classes, hidden_dim, device=device))
        nn.init.xavier_uniform_(arc_W.unsqueeze(0))
        arc_opt_params = [arc_W]
    else:
        arc_opt_params = []

    # ── SupCon loss (DLC-inspired) ────────────────────────────────────────────
    def _supcon_loss(feats: "torch.Tensor", labels: "torch.Tensor", temp: float = 0.07) -> "torch.Tensor":
        B = feats.size(0)
        feat = F.normalize(feats, dim=-1)
        sim = torch.mm(feat, feat.T) / temp
        mask_pos = (labels.unsqueeze(1) == labels.unsqueeze(0))
        mask_pos.fill_diagonal_(False)
        exp_sim = torch.exp(sim)
        eye = ~torch.eye(B, dtype=torch.bool, device=feats.device)
        log_prob = sim - torch.log((exp_sim * eye).sum(dim=1, keepdim=True) + 1e-9)
        n_pos = mask_pos.sum(dim=1).clamp(min=1)
        loss = -(log_prob * mask_pos).sum(dim=1) / n_pos
        return loss.mean()

    # ── class-balanced weights ─────────────────────────────────────────────────
    counts = np.bincount(y_tr, minlength=n_classes).astype(float)
    cw = torch.tensor((counts.sum() / (n_classes * counts)).clip(0.1, 10.0), dtype=torch.float32, device=device)

    # ── optimiser & scheduler ──────────────────────────────────────────────────
    opt = torch.optim.AdamW(list(model.parameters()) + arc_opt_params, lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3, min_lr=1e-6)

    # ── data loaders ──────────────────────────────────────────────────────────
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    y_va_t = torch.tensor(y_va, dtype=torch.long)
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)  # nosemgrep

    best_score, best_state, best_ep, no_improve = -1.0, None, 0, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        ep_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()

            if use_arcface:
                # get penultimate features (all blocks without final head)
                h = xb
                for blk in model.blocks:
                    h = blk(h)
                # ArcFace logits
                cos_theta = F.linear(F.normalize(h), F.normalize(arc_W))
                sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta ** 2, 1e-7, 1.0))
                phi = cos_theta * cos_m - sin_theta * sin_m
                phi = torch.where(cos_theta > th, phi, cos_theta - mm)
                one_hot = torch.zeros_like(cos_theta).scatter_(1, yb.unsqueeze(1), 1.0)
                logits = (one_hot * phi + (1 - one_hot) * cos_theta) * 32.0
                feats = h
            else:
                logits = model(xb)
                feats = None

            # focal loss (DLC-inspired)
            if fg > 0:
                ce = F.cross_entropy(logits, yb, weight=cw, label_smoothing=ls, reduction="none")
                p_t = torch.softmax(logits, -1).gather(1, yb.unsqueeze(1)).squeeze(1)
                loss = ((1 - p_t) ** fg * ce).mean()
            else:
                loss = F.cross_entropy(logits, yb, weight=cw, label_smoothing=ls)

            # SupCon auxiliary loss (DLC-inspired)
            if supcon_w > 0.0 and feats is not None:
                loss = loss + supcon_w * _supcon_loss(feats, yb)

            nn.utils.clip_grad_norm_(list(model.parameters()) + arc_opt_params, 1.0)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item()) * len(xb)
        ep_loss /= len(X_tr)

        # ── validation ────────────────────────────────────────────────────────
        model.eval()
        preds_list = []
        with torch.no_grad():
            for i in range(0, len(X_va_t), 256):
                xb = X_va_t[i: i + 256].to(device)
                if use_arcface:
                    h = xb
                    for blk in model.blocks:
                        h = blk(h)
                    logits = F.linear(F.normalize(h), F.normalize(arc_W)) * 32.0
                else:
                    logits = model(xb)
                preds_list.append(logits.argmax(dim=-1).cpu().numpy())
        preds = np.concatenate(preds_list)
        f1_idx = _f1_per_class(y_va, preds, n_classes)
        f1_macro = float(np.mean(list(f1_idx.values())))
        acc = float((preds == y_va).mean())
        sched.step(1.0 - f1_macro)
        logger.debug(f"Deep: epoch={epoch} f1={f1_macro:.4f} acc={acc:.4f} loss={ep_loss:.4f}")

        if f1_macro > best_score:
            best_score, best_ep, no_improve = f1_macro, epoch, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # restore best weights, final eval
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    preds_list = []
    with torch.no_grad():
        for i in range(0, len(X_va_t), 256):
            xb = X_va_t[i: i + 256].to(device)
            if use_arcface:
                h = xb
                for blk in model.blocks:
                    h = blk(h)
                logits = F.linear(F.normalize(h), F.normalize(arc_W)) * 32.0
            else:
                logits = model(xb)
            preds_list.append(logits.argmax(dim=-1).cpu().numpy())
    preds = np.concatenate(preds_list)
    f1_idx = _f1_per_class(y_va, preds, n_classes)
    f1_macro = float(np.mean(list(f1_idx.values())))
    acc = float((preds == y_va).mean())
    f1_named = {classes[i]: f1_idx[i] for i in range(n_classes)}
    return f1_macro, acc, f1_named, model.state_dict(), best_ep


def train_deep(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    classes: List[str],
    n_trials: int = _DEFAULT_N_TRIALS,
    base_seed: int = _DEFAULT_SEED,
    max_time: Optional[int] = None,
    max_epochs: int = 60,
    patience: int = 8,
) -> Dict[str, Any]:
    try:
        import torch
    except ImportError:
        raise RuntimeError("PyTorch required for deep head. pip install torch")

    encoder_dim = X_tr.shape[1]
    bayes = BayesianOptimizer(_DEEP_SPACE, n_warmup=min(5, n_trials), seed=base_seed)
    deadline = (time.monotonic() + max_time) if max_time else None
    best_meta: Optional[Dict] = None

    for trial_i in range(n_trials):
        if deadline and time.monotonic() >= deadline:
            logger.warning(f"Deep: time budget at trial {trial_i}/{n_trials}")
            break
        hp = bayes.suggest()
        try:
            f1_macro, acc, f1_named, state_dict, best_ep = _deep_trial(
                X_tr, y_tr, X_va, y_va, classes, hp,
                max_epochs=max_epochs, patience=patience,
                seed=base_seed + trial_i,
            )
        except Exception as exc:
            logger.warning(f"Deep: trial {trial_i} failed: {exc}")
            bayes.observe(hp, 0.0)
            continue
        bayes.observe(hp, f1_macro)
        _, best_f1 = bayes.best
        logger.info(
            f"Deep: trial={trial_i+1}/{n_trials} "
            f"hd={hp['hidden_dim']} nb={hp['num_blocks']} drop={hp['dropout']:.2f} "
            f"lr={hp['lr']:.2e} arc_m={hp['arc_m']:.2f} sc={hp['supcon_weight']:.2f} "
            f"f1={f1_macro:.4f} best={best_f1:.4f} ep={best_ep}"
        )
        if best_meta is None or f1_macro > best_meta["f1_macro"]:
            best_meta = dict(
                state_dict=state_dict, encoder_dim=encoder_dim,
                hidden_dims=[int(hp["hidden_dim"])] * int(hp["num_blocks"]),
                dropout=float(hp["dropout"]),
                f1_macro=f1_macro, acc=acc, f1_per_class=f1_named,
                epochs_run=best_ep, lr=float(hp["lr"]), weight_decay=float(hp["weight_decay"]),
                kind="deep",
            )

    if best_meta is None:
        raise RuntimeError("Deep: no trial completed")
    logger.info(f"Deep: best f1={best_meta['f1_macro']:.4f} acc={best_meta['acc']:.4f}")
    return best_meta


# ---------------------------------------------------------------------------
# Save helpers — produce artifacts compatible with SemanticRouter
# ---------------------------------------------------------------------------

def _save_logistic(meta: Dict, classes: List[str], encoder_dim: int, source: str, out: Path,
                   total_samples: int = 1, samples_per_class: Optional[Dict[str, int]] = None,
                   hard_negatives_path: str = "") -> None:
    from semantic_search.qi.training.head_trainer import LearnedHead, HeadTrainingMetadata
    md = HeadTrainingMetadata(
        encoder_dim=encoder_dim, classes=classes,
        total_samples=max(total_samples, 1),
        samples_per_class=samples_per_class or {c: 1 for c in classes},
        holdout_fraction=_DEFAULT_HOLDOUT_FRACTION,
        f1_macro_holdout=meta["f1_macro"], f1_per_class_holdout=meta["f1_per_class"],
        accuracy_holdout=meta["acc"], epochs_run=meta["epochs_run"],
        final_loss=meta["final_loss"], l2_penalty=meta["l2"],
        learning_rate=meta["lr"], random_seed=_DEFAULT_SEED,
        source_seeds_path=source, source_hard_negatives_path=hard_negatives_path,
        created_at=time.time(),
    )
    LearnedHead(weights=meta["W"].astype(np.float64), bias=meta["b"].astype(np.float64),
                classes=classes, encoder_dim=encoder_dim, metadata=md).save(out)


def _save_sklearn(meta: Dict, classes: List[str], encoder_dim: int, source: str, out: Path,
                  total_samples: int = 1, samples_per_class: Optional[Dict[str, int]] = None,
                  hard_negatives_path: str = "") -> None:
    from semantic_search.qi.training.head_trainer import (
        SklearnHead, HeadTrainingMetadata, _ARTIFACT_KIND_SVM, _ARTIFACT_KIND_GB,
    )
    artifact_kind = _ARTIFACT_KIND_SVM if meta["kind"] == "svm" else _ARTIFACT_KIND_GB
    md = HeadTrainingMetadata(
        encoder_dim=encoder_dim, classes=classes,
        total_samples=max(total_samples, 1),
        samples_per_class=samples_per_class or {c: 1 for c in classes},
        holdout_fraction=_DEFAULT_HOLDOUT_FRACTION,
        f1_macro_holdout=meta["f1_macro"], f1_per_class_holdout=meta["f1_per_class"],
        accuracy_holdout=meta["acc"], epochs_run=0, final_loss=0.0,
        l2_penalty=0.0, learning_rate=0.0, random_seed=_DEFAULT_SEED,
        source_seeds_path=source, source_hard_negatives_path=hard_negatives_path,
        created_at=time.time(),
    )
    SklearnHead(model_bytes=meta["model_bytes"], artifact_kind=artifact_kind,
                classes=classes, encoder_dim=encoder_dim, metadata=md).save(out)


def _save_deep(meta: Dict, classes: List[str], source: str, out: Path,
               total_samples: int = 1, samples_per_class: Optional[Dict[str, int]] = None,
               hard_negatives_path: str = "") -> None:
    import torch
    from semantic_search.qi.training.deep_head_trainer import DeepHead
    from semantic_search.qi.training.head_trainer import HeadTrainingMetadata
    md = HeadTrainingMetadata(
        encoder_dim=meta["encoder_dim"], classes=classes,
        total_samples=max(total_samples, 1),
        samples_per_class=samples_per_class or {c: 1 for c in classes},
        holdout_fraction=_DEFAULT_HOLDOUT_FRACTION,
        f1_macro_holdout=meta["f1_macro"], f1_per_class_holdout=meta["f1_per_class"],
        accuracy_holdout=meta["acc"], epochs_run=meta["epochs_run"],
        final_loss=0.0, l2_penalty=meta["weight_decay"],
        learning_rate=meta["lr"], random_seed=_DEFAULT_SEED,
        source_seeds_path=source, source_hard_negatives_path=hard_negatives_path,
        created_at=time.time(),
    )
    buf = io.BytesIO()
    torch.save(meta["state_dict"], buf)  # nosemgrep
    head = DeepHead(
        model_bytes=buf.getvalue(), classes=classes,
        encoder_dim=meta["encoder_dim"], hidden_dims=meta["hidden_dims"],
        dropout=meta["dropout"], metadata=md,
    )
    head.save(out)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    train_csv: Path,
    valid_csv: Path,
    out_path: Path,
    config_path: Optional[str],
    n_trials: int,
    max_time: int,
    include_deep: bool,
    deep_max_epochs: int = 60,
    deep_patience: int = 8,
    hard_negatives_path: Optional[str] = str(_DEFAULT_HARD_NEGATIVES_PATH),
    hard_neg_oversample: int = _DEFAULT_HARD_NEG_OVERSAMPLE,
    hard_neg_threshold: float = 0.65,
    force_mine_hard_negatives: bool = False,
    seeds_path: Optional[str] = None,
) -> Dict[str, Any]:
    logger.info("=" * 70)
    logger.info("train_all_models: START")
    logger.info(f"  train_csv={train_csv}  valid_csv={valid_csv}")
    logger.info(f"  output={out_path}")
    logger.info(f"  n_trials={n_trials}  total_budget={max_time}s  deep={include_deep}")
    logger.info("=" * 70)

    train_texts, train_labels = load_csv(train_csv)
    valid_texts, valid_labels = load_csv(valid_csv)
    classes, label2id = build_label_maps(train_labels + valid_labels)
    y_tr = np.array([label2id[l] for l in train_labels], dtype=np.int64)
    y_va = np.array([label2id[l] for l in valid_labels], dtype=np.int64)
    logger.info(f"data: train={len(train_texts)} valid={len(valid_texts)} classes={classes}")

    from semantic_search.config.loader import load_config
    from semantic_search.config.models import AgentSearchConfig
    from semantic_search.registry import _build_encoder
    raw = load_config(config_path)
    cfg = AgentSearchConfig.from_dict(raw) if isinstance(raw, dict) else raw
    encoder = _build_encoder(cfg)
    encoder_dim = encoder.dim
    logger.info(f"encoder: dim={encoder_dim}")

    logger.info("encoding train texts …")
    X_tr = encode_texts(encoder, train_texts)
    logger.info("encoding valid texts …")
    X_va = encode_texts(encoder, valid_texts)
    logger.info(f"encoded: X_tr={X_tr.shape}  X_va={X_va.shape}")

    # ---- mine / load hard negatives and oversample in training set ----
    hn_rows: List = []
    hn_path_str = ""
    if hard_negatives_path:
        from semantic_search.qi.training.hard_negative_miner import (
            HardNegativeMiner, write_artifact_jsonl, read_artifact_jsonl as _read_hn,
        )
        from semantic_search.qi.training.data_schema import HardNegativeMinerConfig
        from semantic_search.qi.seed_loader import RouterSeedLoader
        hn_out = Path(hard_negatives_path)
        sem_cfg = cfg.qi.semantic
        # Auto-provision: re-mine from the CURRENT seeds whenever forced (e.g. seeds
        # were just regenerated via --run-yaml-to-csv) or when the file is absent.
        # Prevents a stale hard_negatives.jsonl — built from older seeds — from
        # silently re-injecting removed/contaminated queries into training.
        if force_mine_hard_negatives or not hn_out.exists():
            loader = RouterSeedLoader(
                seeds_path=(seeds_path or sem_cfg.seeds_path),
                min_seeds_per_archetype=sem_cfg.min_seeds_per_archetype,
            )
            dataset = loader.load()
            miner_cfg = HardNegativeMinerConfig(
                threshold=hard_neg_threshold,
                num_sub_centroids=sem_cfg.num_sub_centroids,
                encoder_seed=sem_cfg.encoder_seed,
                min_seeds_per_archetype=sem_cfg.min_seeds_per_archetype,
            )
            artifact = HardNegativeMiner(config=miner_cfg, encoder=encoder, dataset=dataset).mine()
            hn_out.parent.mkdir(parents=True, exist_ok=True)
            write_artifact_jsonl(artifact, hn_out)
            logger.info(f"hard_negatives_mined rows={artifact.total_rows()} hn={artifact.total_hard_negatives()} path={hn_out}")
        _, hn_rows = _read_hn(hn_out)
        hn_path_str = str(hn_out)
        logger.info(f"hard_negatives_loaded path={hn_out} rows={len(hn_rows)}")

    if hn_rows and hard_neg_oversample > 0:
        train_set = {t.strip().lower() for t in train_texts}
        hn_confusable = [(r.query, r.label) for r in hn_rows if r.is_hard_negative_for]
        valid_pairs = [
            (t, l) for t, l in hn_confusable
            if l in label2id and t.strip().lower() not in train_set
        ]
        if valid_pairs:
            logger.info(f"encoding {len(valid_pairs)} hard-negative queries for injection …")
            hn_X = encode_texts(encoder, [t for t, _ in valid_pairs])
            hn_y = np.array([label2id[l] for _, l in valid_pairs], dtype=np.int64)
            extra_X = np.tile(hn_X, (hard_neg_oversample, 1))
            extra_y = np.tile(hn_y, hard_neg_oversample)
            X_tr = np.vstack([X_tr, extra_X])
            y_tr = np.concatenate([y_tr, extra_y])
            per_class = {classes[c]: int((hn_y == c).sum()) for c in range(len(classes)) if (hn_y == c).any()}
            logger.info(
                f"hard_neg_injected n_unique={len(valid_pairs)} oversample={hard_neg_oversample} "
                f"added={len(extra_X)} train_size={len(X_tr)} per_class={per_class}"
            )

    n_models = 4 if include_deep else 3
    per_budget = max_time // n_models
    results: Dict[str, Dict] = {}

    for model_name, trainer_fn, kwargs in [
        ("logistic", train_logistic, dict(n_trials=n_trials, max_time=per_budget)),
        ("svm",      train_sklearn,  dict(n_trials=n_trials, max_time=per_budget, classifier_type="svm")),
        ("gradient_boosting", train_sklearn, dict(n_trials=n_trials, max_time=per_budget, classifier_type="gradient_boosting")),
    ] + ([("deep", train_deep, dict(n_trials=n_trials, max_time=per_budget, max_epochs=deep_max_epochs, patience=deep_patience))] if include_deep else []):
        logger.info("-" * 50)
        logger.info(f"TRAINING: {model_name.upper()}")
        t0 = time.monotonic()
        try:
            results[model_name] = trainer_fn(X_tr, y_tr, X_va, y_va, classes, base_seed=_DEFAULT_SEED, **kwargs)
        except Exception as exc:
            logger.warning(f"{model_name} FAILED: {exc}")
        logger.info(f"{model_name}: elapsed={time.monotonic() - t0:.1f}s")

    # ---- comparison table ----
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON TABLE")
    logger.info(f"{'Model':<22} {'F1 Macro':>10} {'Accuracy':>10}  Per-Class F1")
    logger.info("-" * 70)
    winner_name, winner_f1 = None, -1.0
    for name, meta in results.items():
        f1pc = "  ".join(f"{k}:{v:.4f}" for k, v in meta["f1_per_class"].items())
        logger.info(f"{name:<22} {meta['f1_macro']:>10.4f} {meta['acc']:>10.4f}  {f1pc}")
        if meta["f1_macro"] > winner_f1:
            winner_f1, winner_name = meta["f1_macro"], name
    logger.info("=" * 70)

    if winner_name is None:
        raise RuntimeError("No model trained successfully")

    winner = results[winner_name]
    logger.info(f"\nWINNER: {winner_name.upper()}")
    logger.info(f"  f1_macro = {winner['f1_macro']:.4f}  accuracy = {winner['acc']:.4f}")
    for k, v in winner["f1_per_class"].items():
        logger.info(f"    {k:<18} {v:.4f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    source = str(train_csv)
    from collections import Counter as _Counter
    total_samples = int(len(X_tr))
    samples_per_class = {classes[i]: int(v) for i, v in sorted(
        _Counter(int(yi) for yi in y_tr.tolist()).items()
    )}
    if winner_name == "logistic":
        _save_logistic(winner, classes, encoder_dim, source, out_path,
                       total_samples, samples_per_class, hn_path_str)
    elif winner_name in ("svm", "gradient_boosting"):
        _save_sklearn(winner, classes, encoder_dim, source, out_path,
                      total_samples, samples_per_class, hn_path_str)
    elif winner_name == "deep":
        _save_deep(winner, classes, source, out_path,
                   total_samples, samples_per_class, hn_path_str)

    logger.info(f"\nSAVED winner ({winner_name}) → {out_path}")
    logger.info("train_all_models: DONE\n" + "=" * 70)

    return {
        "winner": winner_name,
        "winner_f1_macro": winner["f1_macro"],
        "winner_accuracy": winner["acc"],
        "winner_f1_per_class": winner["f1_per_class"],
        "all_scores": {k: {"f1_macro": v["f1_macro"], "accuracy": v["acc"]} for k, v in results.items()},
        "output": str(out_path),
        "encoder_dim": encoder_dim,
        "classes": classes,
        "train_samples": len(train_texts),
        "valid_samples": len(valid_texts),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="Train all semantic router heads (Bayesian HP search) and save winner.")
    p.add_argument("--train-csv",          default=None)
    p.add_argument("--valid-csv",          default=None)
    p.add_argument("--output",             default=str(_DEFAULT_OUTPUT))
    p.add_argument("--config-path",        default=None)
    p.add_argument("--n-trials",           type=int, default=_DEFAULT_N_TRIALS, help=f"Bayes trials per model (default: {_DEFAULT_N_TRIALS})")
    p.add_argument("--max-time",           type=int, default=_DEFAULT_MAX_TIME, help=f"Total wall-clock budget in seconds (default: {_DEFAULT_MAX_TIME})")
    p.add_argument("--no-deep",            dest="deep", action="store_false", default=True)
    p.add_argument("--deep-max-epochs",    type=int, default=60)
    p.add_argument("--deep-patience",      type=int, default=8)
    p.add_argument("--run-yaml-to-csv",    action="store_true", default=False, help="Auto-generate CSVs from router_seeds.yaml first")
    p.add_argument("--seeds-path",         default=None)
    p.add_argument("--csv-split",          type=float, default=0.1)
    p.add_argument("--ngram-output",           default=str(_DEFAULT_NGRAM_OUTPUT), help="Output path for ngram weights JSON (empty to skip)")
    p.add_argument("--skip-ngram",             action="store_true", default=False, help="Skip ngram scorer co-training")
    p.add_argument("--ngram-top-k",            type=int, default=200, help="Top-K n-grams per class for ngram trainer")
    p.add_argument("--ngram-min-log-odds",     type=float, default=0.3, help="Min log-odds threshold for ngram trainer")
    p.add_argument("--ngram-min-examples",     type=int, default=30, help="Min examples per class for ngram trainer")
    p.add_argument("--hard-negatives-path",    default=str(_DEFAULT_HARD_NEGATIVES_PATH), help="JSONL path for hard negatives (mined if missing)")
    p.add_argument("--hard-neg-oversample",    type=int, default=_DEFAULT_HARD_NEG_OVERSAMPLE, help="Oversample factor for hard negative examples")
    p.add_argument("--remine-hard-negatives",  action="store_true", default=False, help="Force re-mine hard negatives from current seeds (auto-on with --run-yaml-to-csv)")
    p.add_argument("--hard-neg-threshold",     type=float, default=0.65, help="Cosine threshold for hard-negative mining")
    args = p.parse_args()

    train_csv = Path(args.train_csv) if args.train_csv else _DEFAULT_TRAIN_CSV
    valid_csv = Path(args.valid_csv) if args.valid_csv else _DEFAULT_VALID_CSV

    from semantic_search.qi.training.yaml_to_csv import convert, _DEFAULT_SEEDS
    seeds = Path(args.seeds_path) if args.seeds_path else _DEFAULT_SEEDS

    # Positives (train/valid CSV) are ALWAYS rebuilt fresh from the seeds at train
    # time — never reused from a persisted file. Samples are ephemeral training
    # artifacts, not durable references (prevents stale-sample drift).
    logger.info(f"Generating CSVs from {seeds}")
    convert(seeds, train_csv.parent, split=args.csv_split)

    # ── ngram pre-gate co-train — runs IN PARALLEL with the head retrain ──────────
    # The ngram scorer trains from the same seeds via log-odds (no encoder, no head
    # dependency), so it runs concurrently with run_pipeline() instead of waiting for
    # the multi-minute head search. Errors are captured and surfaced after join.
    ngram_thread: Optional[threading.Thread] = None
    ngram_err: Dict[str, BaseException] = {}
    if not args.skip_ngram and args.ngram_output:
        def _ngram_cotrain() -> None:
            try:
                from semantic_search.qi.training.ngram_trainer import train_and_save_from_examples, _load_router_seeds
                logger.info(f"ngram_co_train_started (parallel) seeds={seeds} output={args.ngram_output}")
                examples = _load_router_seeds(str(seeds.resolve()), min_seeds_per_archetype=args.ngram_min_examples)
                result = train_and_save_from_examples(
                    examples, args.ngram_output, top_k=args.ngram_top_k,
                    min_log_odds=args.ngram_min_log_odds, smoothing=1.0,
                    min_examples=args.ngram_min_examples, max_ngram_order=2,
                )
                logger.info(f"ngram_co_train_complete vocab_size={result['vocab_size']} classes={result['classes']} output={result['output_path']}")
            except BaseException as e:  # noqa: BLE001 — surfaced after join
                ngram_err['e'] = e
                logger.error(f"ngram_co_train_failed error={e}")
        ngram_thread = threading.Thread(target=_ngram_cotrain, name="ngram-cotrain")
        ngram_thread.start()

    summary = run_pipeline(
        train_csv=train_csv, valid_csv=valid_csv,
        out_path=Path(args.output), config_path=args.config_path,
        n_trials=args.n_trials, max_time=args.max_time,
        include_deep=args.deep,
        deep_max_epochs=args.deep_max_epochs, deep_patience=args.deep_patience,
        hard_negatives_path=args.hard_negatives_path,
        hard_neg_oversample=args.hard_neg_oversample,
        hard_neg_threshold=args.hard_neg_threshold,
        force_mine_hard_negatives=True,   # negatives always re-mined fresh from current seeds
        seeds_path=str(seeds),
    )
    print("\n--- RESULT ---")
    print(json.dumps(summary, indent=2, default=str))

    if ngram_thread is not None:
        ngram_thread.join()
        if 'e' in ngram_err:
            raise ngram_err['e']

    # Ephemeral samples — built fresh each run, never kept as a later reference.
    # Only the trained artifacts (semantic_head_v1.npz + ngram weights.json) persist.
    for _p in (train_csv, valid_csv, Path(args.hard_negatives_path)):
        try:
            _p.unlink()
        except (FileNotFoundError, OSError):
            pass
    logger.info("training_samples_cleaned (positives + hard-negatives rebuilt each run, not retained)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
