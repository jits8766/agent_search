#!/usr/bin/env python3
"""Harvest positive FeedbackSignals from ClickHouse into a JSONL ring buffer and router seeds.

Pipeline:
  ClickHouse signals_platform_cln.feedback_signals
      ↓  (filter: signal_type IN positive_signal_types, last N days)
      ↓  extract query_type + query_text from JSON payload
  Deduplicate against existing router_seeds.yaml
      ↓  Append to traffic_signals.jsonl  (FIFO ring buffer, max traffic_signals_max_rows lines)
      ↓  (--merge)  Merge qualifying archetypes into router_seeds.yaml
      ↓  (--train)  Run train_all_models.py on the updated seeds

Usage (standalone CLI):
    # Dry-run: see what would be harvested
    python signal_harvester.py

    # Harvest + append to ring buffer
    python signal_harvester.py

    # Harvest + merge qualifying archetypes into seeds + retrain
    python signal_harvester.py --merge --train

    # Drive entirely from base.yaml config (reads RouterRetrainingConfig)
    python signal_harvester.py --config-path /app/config/base.yaml --merge --train

    # Override window and floor
    python signal_harvester.py --lookback-days 7 --min-per-archetype 20 --merge --train

Programmatic entry point: harvest(cfg, seeds_path, merge, train)
  All harvest parameters come from RouterRetrainingConfig; no hardcoded values.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import httpx
import yaml

_HERE = Path(__file__).resolve().parent
_TRAIN_SCRIPT = _HERE / "train_all_models.py"

# No hardcoded connection defaults — all values must come from env vars, --config-path, or explicit CLI flags.


# ---------------------------------------------------------------------------
# Ring buffer helpers
# ---------------------------------------------------------------------------

def _append_ring_buffer(
    path: Path,
    new_rows: List[Dict],
    max_rows: int,
) -> int:
    """Append new_rows to a JSONL ring buffer at path, FIFO-evicting oldest lines so total <= max_rows.

    Returns total line count after write.
    Row format: {"query_type": str, "query_text": str, "ts": float}
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: List[str] = []
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            existing = [l for l in fh.readlines() if l.strip()]

    new_lines = [json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in new_rows]
    combined = existing + new_lines

    if len(combined) > max_rows:
        combined = combined[len(combined) - max_rows:]

    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(combined)

    return len(combined)


def _read_ring_buffer(path: Path) -> List[Dict]:
    """Read all rows from a JSONL ring buffer. Returns [] when file absent."""
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------

def _load_seeds(path: Path) -> Dict[str, List[str]]:
    """Load router_seeds.yaml → {archetype: [text, ...]}."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    raw = data.get("archetypes", {})
    return {k: [str(t) for t in (v or []) if str(t).strip()] for k, v in raw.items()}


def _normalized_existing(archetypes: Dict[str, List[str]]) -> Set[str]:
    """Lowercase-stripped set of every text in seed file (fast dedup)."""
    return {t.strip().lower() for texts in archetypes.values() for t in texts}


def _normalized_ring_buffer(rows: List[Dict]) -> Set[str]:
    """Lowercase-stripped set of every query_text already in the ring buffer."""
    return {str(r.get("query_text", "")).strip().lower() for r in rows if r.get("query_text")}


# ---------------------------------------------------------------------------
# ClickHouse fetch
# ---------------------------------------------------------------------------

def _build_ch_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def _fetch_from_clickhouse(
    ch_url: str,
    ch_user: str,
    ch_password: str,
    database: str,
    positive_types: Set[str],
    lookback_days: int,
    max_per_arch: int,
) -> Dict[str, List[str]]:
    """Query ClickHouse for positive signals → {archetype: [query_text, ...]} (deduped within response)."""
    types_literal = ", ".join(f"'{t}'" for t in sorted(positive_types))
    sql = f"""
SELECT
    JSONExtractString(payload, 'query_type') AS query_type,
    JSONExtractString(payload, 'query_text') AS query_text
FROM {database}.feedback_signals
WHERE signal_type IN ({types_literal})
  AND created_at >= now() - INTERVAL {lookback_days} DAY
  AND JSONExtractString(payload, 'query_type') != ''
  AND JSONExtractString(payload, 'query_text') != ''
ORDER BY created_at DESC
LIMIT {max_per_arch * len(positive_types) * 20}
FORMAT JSONEachRow
""".strip()

    params: Dict[str, str] = {"user": ch_user, "database": database}
    if ch_password:
        params["password"] = ch_password

    with httpx.Client(timeout=60.0) as client:
        resp = client.post(ch_url, content=sql, params=params)
        resp.raise_for_status()

    seen_per_arch: Dict[str, Set[str]] = defaultdict(set)
    result: Dict[str, List[str]] = defaultdict(list)

    for line in resp.text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        qt = str(row.get("query_type", "")).strip()
        txt = str(row.get("query_text", "")).strip()
        if not qt or not txt:
            continue
        key = txt.lower()
        if key not in seen_per_arch[qt] and len(result[qt]) < max_per_arch:
            seen_per_arch[qt].add(key)
            result[qt].append(txt)

    return dict(result)


def _count_clickhouse_signals_since(
    ch_url: str,
    ch_user: str,
    ch_password: str,
    database: str,
    positive_types: Set[str],
    since_ts: float,
) -> int:
    """Count positive signals in ClickHouse since a Unix timestamp. Returns 0 on any error."""
    types_literal = ", ".join(f"'{t}'" for t in sorted(positive_types))
    sql = f"""
SELECT count() AS cnt
FROM {database}.feedback_signals
WHERE signal_type IN ({types_literal})
  AND created_at >= fromUnixTimestamp64Milli({int(since_ts * 1000)})
FORMAT JSONEachRow
""".strip()
    params: Dict[str, str] = {"user": ch_user, "database": database}
    if ch_password:
        params["password"] = ch_password
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(ch_url, content=sql, params=params)
            resp.raise_for_status()
        for line in resp.text.splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            return int(row.get("cnt", 0))
    except Exception:
        return 0
    return 0


# ---------------------------------------------------------------------------
# Marker helpers
# ---------------------------------------------------------------------------

def _read_marker(path: Path) -> float:
    """Read last-retrain Unix timestamp from marker file. Returns 0.0 when absent."""
    try:
        return float(path.read_text(encoding="utf-8").strip())
    except Exception:
        return 0.0


def _write_marker(path: Path, ts: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(ts), encoding="utf-8")


# ---------------------------------------------------------------------------
# Core harvest pipeline
# ---------------------------------------------------------------------------

def harvest(
    ch_url: str,
    ch_user: str,
    ch_password: str,
    database: str,
    seeds_path: Path,
    traffic_signals_path: Path,
    traffic_signals_max_rows: int,
    positive_types: Set[str],
    lookback_days: int,
    min_per_arch: int,
    max_per_arch: int,
    merge: bool,
    train: bool,
    output_npz_path: Optional[str] = None,
    train_args: Optional[List[str]] = None,
    ngram_weights_output_path: Optional[str] = None,
    ngram_top_k: int = 200,
    ngram_min_log_odds: float = 0.3,
    hard_negatives_path: Optional[str] = None,
    hard_neg_oversample: int = 0,
    hard_neg_threshold: float = 0.65,
) -> Tuple[Dict[str, int], Dict[str, int], int]:
    """Full harvest pipeline.

    Returns (new_counts, merged_counts, ring_buffer_total):
      new_counts      — {archetype: n} of fresh texts appended to ring buffer
      merged_counts   — {archetype: n} of texts merged into router_seeds.yaml (empty if not --merge)
      ring_buffer_total — total lines in traffic_signals.jsonl after write
    """
    train_args = train_args or []

    existing = _load_seeds(seeds_path)
    existing_norm = _normalized_existing(existing)
    ring_rows = _read_ring_buffer(traffic_signals_path)
    ring_norm = _normalized_ring_buffer(ring_rows)
    already_seen = existing_norm | ring_norm

    print(f"signal_harvester: seeds={seeds_path} existing_seed_texts={sum(len(v) for v in existing.values())} ring_buffer_rows={len(ring_rows)}")
    print(f"signal_harvester: querying ClickHouse url={ch_url} db={database} lookback={lookback_days}d types={sorted(positive_types)}")

    raw = _fetch_from_clickhouse(ch_url, ch_user, ch_password, database, positive_types, lookback_days, max_per_arch)
    print(f"signal_harvester: ClickHouse raw per-arch: { {k: len(v) for k, v in raw.items()} }")

    ts_now = time.time()
    new_by_arch: Dict[str, List[str]] = {}
    ring_rows_to_add: List[Dict] = []

    for arch, texts in raw.items():
        fresh = [t for t in texts if t.strip().lower() not in already_seen]
        if fresh:
            new_by_arch[arch] = fresh
            for txt in fresh:
                ring_rows_to_add.append({"query_type": arch, "query_text": txt, "ts": ts_now})

    new_counts = {arch: len(texts) for arch, texts in new_by_arch.items()}
    total_new = sum(new_counts.values())
    print(f"signal_harvester: {total_new} new texts after dedup | per-arch: {new_counts}")

    ring_total = _append_ring_buffer(traffic_signals_path, ring_rows_to_add, traffic_signals_max_rows)
    print(f"signal_harvester: ring buffer → {traffic_signals_path}  total_rows={ring_total}/{traffic_signals_max_rows}")

    merged_counts: Dict[str, int] = {}
    if merge:
        qualifying = {arch: texts for arch, texts in new_by_arch.items() if len(texts) >= min_per_arch}
        if not qualifying:
            print(f"signal_harvester: skip merge — no archetype meets min_per_archetype={min_per_arch} (got: {new_counts})")
        else:
            merged = {arch: list(texts) for arch, texts in existing.items()}
            for arch, texts in qualifying.items():
                merged.setdefault(arch, [])
                merged[arch].extend(texts)
            with open(seeds_path, "w", encoding="utf-8") as fh:
                yaml.dump(
                    {"archetypes": merged},
                    fh,
                    allow_unicode=True,
                    default_flow_style=False,
                    sort_keys=True,
                )
            merged_counts = {arch: len(texts) for arch, texts in qualifying.items()}
            print(f"signal_harvester: merged into {seeds_path} archetypes={sorted(qualifying.keys())} counts={merged_counts}")

    if train:
        cmd = [sys.executable, str(_TRAIN_SCRIPT), "--run-yaml-to-csv", "--seeds-path", str(seeds_path)]
        if output_npz_path:
            cmd += ["--output", output_npz_path]
        if ngram_weights_output_path:
            cmd += ["--ngram-output", ngram_weights_output_path,
                    "--ngram-top-k", str(ngram_top_k),
                    "--ngram-min-log-odds", str(ngram_min_log_odds)]
        else:
            cmd += ["--skip-ngram"]
        if hard_negatives_path:
            cmd += ["--hard-negatives-path", hard_negatives_path,
                    "--hard-neg-oversample", str(hard_neg_oversample),
                    "--hard-neg-threshold", str(hard_neg_threshold)]
        cmd += train_args
        print(f"signal_harvester: launching training → {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    return new_counts, merged_counts, ring_total


# ---------------------------------------------------------------------------
# Auto-trigger entry point (called from app.py startup)
# ---------------------------------------------------------------------------

def maybe_trigger_harvest_and_train(
    ch_url: str,
    ch_user: str,
    ch_password: str,
    database: str,
    seeds_path: Path,
    traffic_signals_path: Path,
    traffic_signals_max_rows: int,
    positive_types: Set[str],
    lookback_days: int,
    min_per_arch: int,
    max_per_arch: int,
    new_signals_threshold: int,
    last_retrain_marker_path: Path,
    output_npz_path: str,
    ngram_weights_output_path: str = "",
    ngram_top_k: int = 200,
    ngram_min_log_odds: float = 0.3,
    hard_negatives_path: str = "",
    hard_neg_oversample: int = 0,
    hard_neg_threshold: float = 0.65,
) -> bool:
    """Check threshold and run harvest + train if enough new signals exist.

    Returns True when pipeline was triggered, False when threshold not met.
    Used by app.py lifespan startup hook.
    """
    last_ts = _read_marker(last_retrain_marker_path)
    count = _count_clickhouse_signals_since(ch_url, ch_user, ch_password, database, positive_types, last_ts)
    print(f"signal_harvester: new_signals_since_last_retrain={count} threshold={new_signals_threshold} last_retrain_ts={last_ts}")

    if count < new_signals_threshold:
        print(f"signal_harvester: threshold not met — skipping auto-retrain ({count} < {new_signals_threshold})")
        return False

    print(f"signal_harvester: threshold met ({count} >= {new_signals_threshold}) — starting harvest + train")
    new_counts, merged_counts, ring_total = harvest(
        ch_url=ch_url,
        ch_user=ch_user,
        ch_password=ch_password,
        database=database,
        seeds_path=seeds_path,
        traffic_signals_path=traffic_signals_path,
        traffic_signals_max_rows=traffic_signals_max_rows,
        positive_types=positive_types,
        lookback_days=lookback_days,
        min_per_arch=min_per_arch,
        max_per_arch=max_per_arch,
        merge=True,
        train=True,
        output_npz_path=output_npz_path,
        ngram_weights_output_path=ngram_weights_output_path,
        ngram_top_k=ngram_top_k,
        ngram_min_log_odds=ngram_min_log_odds,
        hard_negatives_path=hard_negatives_path or None,
        hard_neg_oversample=hard_neg_oversample,
        hard_neg_threshold=hard_neg_threshold,
    )
    _write_marker(last_retrain_marker_path, time.time())
    print(f"signal_harvester: auto-retrain complete new={new_counts} merged={merged_counts} ring_total={ring_total}")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_retraining_config_from_yaml(config_path: str):
    """Load RouterRetrainingConfig from base.yaml if available. Returns None on error."""
    try:
        from semantic_search.config.loader import load_config
        cfg = load_config(config_path=config_path)
        qi_raw = cfg.get("qi", {})
        rr_raw = qi_raw.get("router_retraining")
        if not isinstance(rr_raw, dict):
            return None
        from semantic_search.config.models import RouterRetrainingConfig
        return RouterRetrainingConfig.from_dict(rr_raw)
    except Exception as e:
        print(f"signal_harvester: warning — could not load RouterRetrainingConfig from {config_path}: {e}", file=sys.stderr)
        return None


def main() -> None:
    p = argparse.ArgumentParser(
        description="Harvest positive FeedbackSignals from ClickHouse → router seeds pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config-path", default=None,
                   help="Path to base.yaml — loads all defaults from qi.router_retraining (CLI flags override)")
    p.add_argument("--ch-host", default=None,
                   help="ClickHouse host — required; set via CLICKHOUSE_HOST env var or this flag")
    p.add_argument("--ch-port", type=int, default=None,
                   help="ClickHouse HTTP port — required; set via CLICKHOUSE_PORT env var or this flag")
    p.add_argument("--ch-user", default=None,
                   help="ClickHouse user — required; set via CLICKHOUSE_USER env var or this flag")
    p.add_argument("--ch-password", default=None,
                   help="ClickHouse password (env: CLICKHOUSE_PASSWORD; default: empty)")
    p.add_argument("--database", default=None, help="ClickHouse database for feedback_signals")
    p.add_argument("--seeds-path", default=None, help="Path to router_seeds.yaml")
    p.add_argument("--traffic-signals-path", default=None, help="Path to traffic_signals.jsonl ring buffer")
    p.add_argument("--traffic-signals-max-rows", type=int, default=None, help="Ring buffer capacity")
    p.add_argument("--positive-types", nargs="+", default=None,
                   help="Signal types treated as positives (space-separated)")
    p.add_argument("--lookback-days", type=int, default=None, help="Days of ClickHouse history to pull")
    p.add_argument("--min-per-archetype", type=int, default=None,
                   help="Min new texts per archetype before merging into seeds")
    p.add_argument("--max-per-archetype", type=int, default=None,
                   help="Cap on new texts per archetype from ClickHouse")
    p.add_argument("--output-npz-path", default=None, help="Output .npz path for trained head artifact")
    p.add_argument("--merge", action="store_true",
                   help="Merge qualifying new texts into router_seeds.yaml (permanent)")
    p.add_argument("--train", action="store_true",
                   help="Run train_all_models.py after harvest")
    p.add_argument("--train-args", nargs=argparse.REMAINDER, default=[],
                   help="Extra args forwarded verbatim to train_all_models.py (put after --)")
    args = p.parse_args()

    rr_cfg = _load_retraining_config_from_yaml(args.config_path) if args.config_path else None

    def _resolve(cli_val, cfg_attr, env_var=None, fallback=None):
        if cli_val is not None:
            return cli_val
        if rr_cfg is not None:
            v = getattr(rr_cfg, cfg_attr, None)
            if v is not None and v != "":
                return v
        if env_var:
            v = os.environ.get(env_var)
            if v:
                return v
        return fallback

    ch_host = _resolve(args.ch_host, "clickhouse_host", "CLICKHOUSE_HOST")
    ch_port = _resolve(args.ch_port, "clickhouse_port", "CLICKHOUSE_PORT")
    ch_user = _resolve(args.ch_user, "clickhouse_user", "CLICKHOUSE_USER")
    ch_password = _resolve(args.ch_password, None, "CLICKHOUSE_PASSWORD", "")
    database = _resolve(args.database, "clickhouse_database", None)
    seeds_raw = _resolve(args.seeds_path, "seeds_path", None)
    traffic_path_raw = _resolve(args.traffic_signals_path, "traffic_signals_path", None)
    max_rows = _resolve(args.traffic_signals_max_rows, "traffic_signals_max_rows", None)
    pos_types = _resolve(args.positive_types, "positive_signal_types", None)
    lookback = _resolve(args.lookback_days, "clickhouse_lookback_days", None)
    min_per_arch = _resolve(args.min_per_archetype, "min_new_signals_per_archetype", None)
    max_per_arch = _resolve(args.max_per_archetype, "max_signals_per_archetype", None)
    out_npz = _resolve(args.output_npz_path, "output_npz_path", None)

    missing = [name for name, val in [
        ("ch-host", ch_host), ("ch-port", ch_port), ("ch-user", ch_user),
        ("database", database), ("seeds-path", seeds_raw), ("traffic-signals-path", traffic_path_raw),
        ("traffic-signals-max-rows", max_rows), ("positive-types", pos_types),
        ("lookback-days", lookback), ("min-per-archetype", min_per_arch), ("max-per-archetype", max_per_arch),
    ] if val is None]
    if missing:
        p.error(f"Missing required values (set via --config-path, CLI flags, or env vars): {missing}")

    seeds_path = Path(str(seeds_raw)) if seeds_raw else (_HERE.parent / "router_seeds.yaml")
    traffic_signals_path = Path(str(traffic_path_raw))
    ch_url = _build_ch_url(str(ch_host), int(ch_port))

    new_counts, merged_counts, ring_total = harvest(
        ch_url=ch_url,
        ch_user=str(ch_user),
        ch_password=str(ch_password),
        database=str(database),
        seeds_path=seeds_path,
        traffic_signals_path=traffic_signals_path,
        traffic_signals_max_rows=int(max_rows),
        positive_types=set(pos_types),
        lookback_days=int(lookback),
        min_per_arch=int(min_per_arch),
        max_per_arch=int(max_per_arch),
        merge=args.merge,
        train=args.train,
        output_npz_path=str(out_npz) if out_npz else None,
        train_args=args.train_args,
    )

    print()
    print("=== signal_harvester summary ===")
    print(f"  New texts appended to ring buffer:  {new_counts}")
    print(f"  Ring buffer total rows:             {ring_total}/{int(max_rows)}")
    print(f"  Merged into router_seeds.yaml:      {merged_counts if merged_counts else '(not merged)'}")
    total = sum(new_counts.values())
    if not total:
        print("  (no new texts — ClickHouse empty or all texts already in seeds/ring buffer)")
    if new_counts and not args.merge:
        print()
        print("  Tip: run with --merge to add qualifying archetypes to seeds.")
        print("  Run with --merge --train to also retrain the learned head.")


if __name__ == "__main__":
    main()
