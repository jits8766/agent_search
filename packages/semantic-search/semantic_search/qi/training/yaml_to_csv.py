#!/usr/bin/env python3
"""Convert router_seeds.yaml to stratified train.csv + valid.csv.

Usage:
    python yaml_to_csv.py
    python yaml_to_csv.py --seeds path/to/router_seeds.yaml --split 0.2 --out-dir /tmp/data
    python yaml_to_csv.py --split 0.15 --seed 99

Outputs two files: train.csv and valid.csv, each with columns: text,label
Default output directory is the same directory as the seeds file.
"""
import argparse
import csv
import random
from collections import Counter
from pathlib import Path

import yaml

_DEFAULT_SEEDS = Path(__file__).resolve().parents[1] / "router_seeds.yaml"
_DEFAULT_SPLIT = 0.2
_DEFAULT_SEED = 42


def convert(
    seeds_path: Path,
    out_dir: Path,
    split: float = _DEFAULT_SPLIT,
    seed: int = _DEFAULT_SEED,
) -> tuple:
    """Load seeds YAML and write train.csv + valid.csv.

    Returns (train_path, valid_path, train_count, valid_count).
    """
    if not seeds_path.exists():
        raise FileNotFoundError(f"Seeds file not found: {seeds_path}")
    if not 0.0 < split < 1.0:
        raise ValueError(f"split must be in (0, 1), got {split}")

    with open(seeds_path) as fh:
        data = yaml.safe_load(fh)

    archetypes = data.get("archetypes", {})
    if not archetypes:
        raise ValueError(f"No 'archetypes' key found in {seeds_path}")

    rng = random.Random(seed)
    train_rows: list = []
    valid_rows: list = []

    for label, raw_texts in archetypes.items():
        texts = [
            t.strip()
            for t in (raw_texts or [])
            if isinstance(t, str) and t.strip()
        ]
        if not texts:
            continue
        rng.shuffle(texts)
        n_valid = max(1, int(len(texts) * split))
        for t in texts[:n_valid]:
            valid_rows.append({"text": t, "label": label})
        for t in texts[n_valid:]:
            train_rows.append({"text": t, "label": label})

    rng.shuffle(train_rows)
    rng.shuffle(valid_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.csv"
    valid_path = out_dir / "valid.csv"

    for path, rows in [(train_path, train_rows), (valid_path, valid_rows)]:
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["text", "label"])
            writer.writeheader()
            writer.writerows(rows)

    return train_path, valid_path, len(train_rows), len(valid_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert router_seeds.yaml → train.csv + valid.csv")
    parser.add_argument("--seeds", default=None, help=f"Path to router_seeds.yaml (default: {_DEFAULT_SEEDS})")
    parser.add_argument("--split", type=float, default=_DEFAULT_SPLIT, help=f"Validation fraction (default: {_DEFAULT_SPLIT})")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: same as seeds file)")
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED, help=f"RNG seed (default: {_DEFAULT_SEED})")
    args = parser.parse_args()

    seeds_path = Path(args.seeds) if args.seeds else _DEFAULT_SEEDS
    out_dir = Path(args.out_dir) if args.out_dir else seeds_path.parent

    train_path, valid_path, n_train, n_valid = convert(seeds_path, out_dir, args.split, args.seed)

    with open(train_path) as fh:
        train_rows = list(csv.DictReader(fh))
    with open(valid_path) as fh:
        valid_rows = list(csv.DictReader(fh))

    train_counts = dict(Counter(r["label"] for r in train_rows))
    valid_counts = dict(Counter(r["label"] for r in valid_rows))

    print(f"yaml_to_csv: seeds={seeds_path} split={args.split} seed={args.seed}")
    print(f"  train → {train_path}  ({n_train} rows)")
    print(f"  valid → {valid_path}  ({n_valid} rows)")
    print(f"  train per-class: {train_counts}")
    print(f"  valid per-class: {valid_counts}")


if __name__ == "__main__":
    main()
