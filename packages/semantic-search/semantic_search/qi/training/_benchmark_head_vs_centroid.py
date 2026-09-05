"""Learned-head verification gate: compare learned-head vs centroid macro-F1.

Reproduces the same stratified holdout split the trainer used (random_seed=0)
and scores both the learned head and the legacy centroid scorer on it. Used
to validate the promotion gate (>=+3 pp macro-F1, no per-class regression > 1 pp).
Not part of production code paths.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from semantic_search.config.loader import load_config
from semantic_search.config.models import AgentSearchConfig
from semantic_search.qi.cascade_encoder import MatryoshkaCascadeEncoder
from semantic_search.qi.encoder import centroid, cosine_similarity, kmeans_spherical
from semantic_search.qi.seed_loader import RouterSeedLoader
from semantic_search.qi.stage_encoder import StageEncoder
from semantic_search.qi.training.head_trainer import LearnedHead, _f1_per_class, _stratified_split
from semantic_search.registry import _build_encoder as registry_build_encoder


def _build_encoder(cfg: AgentSearchConfig):
    """Build the production router-stage encoder (cascade-sliced to router dim)."""
    base = registry_build_encoder(cfg)
    cascade_cfg = cfg.qi.encoder.cascade
    if cascade_cfg is not None and cascade_cfg.enabled and cascade_cfg.stage_dims is not None and cascade_cfg.stage_dims.router is not None:
        cascade = MatryoshkaCascadeEncoder(base_encoder=base, supported_dims=frozenset(cascade_cfg.supported_dims))
        return StageEncoder(cascade=cascade, stage_dim=cascade_cfg.stage_dims.router, stage_name='router', emit_init_log=cfg.general.startup_log_detail)
    return base


def _encode_dataset(seeds_by_arch: Dict[str, List[str]], encoder) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    classes = sorted(seeds_by_arch.keys())
    cls_index = {c: i for i, c in enumerate(classes)}
    rows: List[List[float]] = []
    labels: List[int] = []
    for cls in classes:
        for q in seeds_by_arch[cls]:
            rows.append(encoder.encode(q))
            labels.append(cls_index[cls])
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int64), classes


def _centroid_predict(X_train: np.ndarray, y_train: np.ndarray, X_holdout: np.ndarray, classes: List[str], k: int, seed: int) -> np.ndarray:
    """Cosine-max over per-archetype K-means sub-centroids — mirrors SemanticRouter."""
    sub_centroids: Dict[int, List[List[float]]] = {}
    for cls_idx, cls in enumerate(classes):
        vectors = X_train[y_train == cls_idx].tolist()
        non_zero = [v for v in vectors if any(abs(x) > 0.0 for x in v)]
        sub_centroids[cls_idx] = kmeans_spherical(non_zero, k=k, seed=seed)
    preds = np.zeros(X_holdout.shape[0], dtype=np.int64)
    for i, vec in enumerate(X_holdout.tolist()):
        best_score = -1.0
        best_cls = 0
        for cls_idx in range(len(classes)):
            s = max(cosine_similarity(vec, c) for c in sub_centroids[cls_idx])
            if s > best_score:
                best_score = s
                best_cls = cls_idx
        preds[i] = best_cls
    return preds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--head-path', default='semantic_search/qi/training_data/semantic_head_v1.npz')
    parser.add_argument('--config-path', default=None)
    parser.add_argument('--holdout-fraction', type=float, default=0.10)
    parser.add_argument('--random-seed', type=int, default=0)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    raw = load_config(args.config_path) if args.config_path else load_config()
    cfg = AgentSearchConfig.from_dict(raw)

    loader = RouterSeedLoader(seeds_path=cfg.qi.semantic.seeds_path, min_seeds_per_archetype=cfg.qi.semantic.min_seeds_per_archetype)
    dataset = loader.load()

    encoder = _build_encoder(cfg)
    seeds_by_arch = {a: dataset.texts(a) for a in dataset.archetypes()}
    X, y, classes = _encode_dataset(seeds_by_arch, encoder)

    train_idx, hold_idx = _stratified_split(y, args.holdout_fraction, args.random_seed)
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_ho, y_ho = X[hold_idx], y[hold_idx]

    head = LearnedHead.load(project_root / args.head_path)
    head_classes = head.classes
    assert head_classes == classes, f'class mismatch: head={head_classes} dataset={classes}'
    probs = head.predict_proba(X_ho)
    head_preds = np.argmax(probs, axis=1)
    head_f1_dict = _f1_per_class(y_ho, head_preds, len(classes))
    head_f1 = np.asarray([head_f1_dict[i] for i in range(len(classes))], dtype=np.float64)
    head_acc = float((head_preds == y_ho).mean())

    centroid_preds = _centroid_predict(
        X_tr, y_tr, X_ho, classes,
        k=cfg.qi.semantic.num_sub_centroids,
        seed=cfg.qi.semantic.encoder_seed,
    )
    centroid_f1_dict = _f1_per_class(y_ho, centroid_preds, len(classes))
    centroid_f1 = np.asarray([centroid_f1_dict[i] for i in range(len(classes))], dtype=np.float64)
    centroid_acc = float((centroid_preds == y_ho).mean())

    print('=== Learned-head verification: head vs centroid ===')
    print(f'holdout size: {len(y_ho)} (per-class: {dict(Counter(y_ho.tolist()))})')
    print(f'classes: {classes}')
    print(f'centroid    accuracy={centroid_acc:.4f}  f1_macro={float(np.mean(centroid_f1)):.4f}  per-class={[round(x,4) for x in centroid_f1.tolist()]}')
    print(f'learned     accuracy={head_acc:.4f}  f1_macro={float(np.mean(head_f1)):.4f}  per-class={[round(x,4) for x in head_f1.tolist()]}')
    delta = float(np.mean(head_f1)) - float(np.mean(centroid_f1))
    per_class_delta = (head_f1 - centroid_f1).tolist()
    print(f'delta_macro_f1: {delta:+.4f}  delta_per_class={[round(x,4) for x in per_class_delta]}')
    gate_macro = delta >= 0.03
    gate_no_regress = all(d >= -0.01 for d in per_class_delta)
    print(f'gate macro_f1>=+0.03: {"PASS" if gate_macro else "FAIL"}')
    print(f'gate per-class regression <=0.01: {"PASS" if gate_no_regress else "FAIL"}')
    print(f'phase4 verification: {"PASS" if (gate_macro and gate_no_regress) else "FAIL"}')


if __name__ == '__main__':
    main()
