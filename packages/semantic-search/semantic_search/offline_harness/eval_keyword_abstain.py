"""Eval L0 keyword abstain quality vs Sonnet gold (false-positive focus).

Loads cases exported from phase1_validation_outputs.xlsx and re-runs the
current L0 filter+keyword prompt on gemini-2.5-flash-lite (harness grounding
model). Compares baseline sheet keywords vs new extract.

Usage (from auc-semantic-search):
  uv run python -m semantic_search.offline_harness.eval_keyword_abstain
"""
from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

_REPO = Path(__file__).resolve().parents[4]
_OUT = _REPO / "output" / "keyword_abstain_eval"
_APP_ROOT = Path(__file__).resolve().parents[5]


def _tok(terms: List[str]) -> Set[str]:
    bag: Set[str] = set()
    for t in terms:
        bag.update(p for p in re.split(r"[\s_]+", str(t).lower()) if p)
    return bag


def _extract_one(complete: Any, query: str, kw_min: float) -> Tuple[List[str], float]:
    from semantic_search.qi.l0_llm_filter_extractor import (  # noqa: PLC0415
        build_l0_filter_user_prompt,
        expand_gd_to_godaddy,
    )

    prompt = build_l0_filter_user_prompt([(1, expand_gd_to_godaddy(query))])
    t0 = time.monotonic()
    raw = complete(prompt)
    elapsed = time.monotonic() - t0
    terms: List[str] = []
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group())
            entry = next(
                (e for e in parsed.get("results", []) if e.get("idx") == 1), None
            )
            if entry:
                for kw in entry.get("keywords") or []:
                    if not isinstance(kw, dict):
                        continue
                    term = str(kw.get("term") or "").strip().lower()
                    if not term:
                        continue
                    try:
                        prob = float(kw.get("probability"))
                    except (TypeError, ValueError):
                        continue
                    if prob >= kw_min:
                        terms.append(term)
        except json.JSONDecodeError:
            terms = []
    return terms, elapsed


def _run_cases(
    cases: List[Dict[str, Any]],
    *,
    complete: Any,
    kw_min: float,
    workers: int,
    label: str,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    total = len(cases)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(_extract_one, complete, c["query"], kw_min): c for c in cases
        }
        for fut in as_completed(futs):
            c = futs[fut]
            done += 1
            try:
                terms, elapsed = fut.result()
                err = None
            except Exception as exc:  # noqa: BLE001
                terms, elapsed, err = [], 0.0, f"{type(exc).__name__}: {exc}"
            results.append(
                {
                    **c,
                    "new_keywords": terms,
                    "elapsed_s": round(elapsed, 3),
                    "error": err,
                }
            )
            if done % 25 == 0 or done == total:
                print(f"  [{label}] {done}/{total}", flush=True)
    return results


def _summarize_failed(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    baseline_empty = sum(1 for r in rows if not r.get("baseline_keywords"))
    errors = sum(1 for r in rows if r.get("error"))
    # All failed cases: Sonnet=[] and baseline non-empty. Success = new empty.
    success = sum(
        1 for r in rows if not r.get("new_keywords") and not r.get("error")
    )
    still_fp = [r for r in rows if r.get("new_keywords") and not r.get("error")]
    fp_terms: Dict[str, int] = {}
    for r in still_fp:
        for t in r["new_keywords"]:
            fp_terms[t] = fp_terms.get(t, 0) + 1
    top_fp = sorted(fp_terms.items(), key=lambda x: -x[1])[:20]
    scored = max(n - errors, 1)
    return {
        "n": n,
        "baseline_wrong_emit": n - baseline_empty,
        "new_abstain_correct": success,
        "new_abstain_rate": round(success / scored, 4),
        "still_false_positive": len(still_fp),
        "still_fp_rate": round(len(still_fp) / scored, 4),
        "errors": errors,
        "top_remaining_fp_terms": top_fp,
        "fp_reduction_vs_baseline": round(
            1.0 - (len(still_fp) / max(n - baseline_empty, 1)), 4
        ),
    }


def _summarize_holdout(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Sonnet-positive: measure recall/precision regression."""
    usable = [r for r in rows if not r.get("error")]
    n = len(usable)
    if n == 0:
        return {"n": 0}
    recalls: List[float] = []
    precs: List[float] = []
    empty_new = 0
    for r in usable:
        gold = _tok(r.get("sonnet_keyword") or [])
        pred = _tok(r.get("new_keywords") or [])
        if not pred:
            empty_new += 1
        if gold:
            recalls.append(len(gold & pred) / len(gold))
        if pred:
            precs.append(len(gold & pred) / len(pred))
        else:
            precs.append(0.0 if gold else 1.0)
    return {
        "n": n,
        "mean_recall_vs_sonnet": round(sum(recalls) / len(recalls), 4) if recalls else None,
        "mean_precision_vs_sonnet": round(sum(precs) / len(precs), 4) if precs else None,
        "new_empty_when_sonnet_has": empty_new,
        "new_empty_rate": round(empty_new / n, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failed-limit", type=int, default=0, help="0=all failed cases")
    parser.add_argument("--holdout-limit", type=int, default=80)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--out-suffix",
        type=str,
        default="",
        help="Optional suffix for result filenames (e.g. haiku, gpt54mini)",
    )
    args = parser.parse_args()

    from semantic_search.offline_harness.reground_filters_four_way import (  # noqa: PLC0415
        _build_grounding_complete,
        _fill_blank_env_from_dotenv,
        _grounding_model,
    )
    from semantic_search.config.loader import load_config  # noqa: PLC0415
    from semantic_search.config.models import AgentSearchConfig  # noqa: PLC0415

    for candidate in (_APP_ROOT / ".env", _REPO / ".env"):
        if candidate.exists():
            _fill_blank_env_from_dotenv(candidate)
            break

    failed_path = _OUT / "failed_abstain_cases.json"
    holdout_path = _OUT / "sonnet_positive_holdout.json"
    failed = json.loads(failed_path.read_text())
    holdout = json.loads(holdout_path.read_text())
    if args.failed_limit and args.failed_limit > 0:
        failed = failed[: args.failed_limit]
    if args.holdout_limit and args.holdout_limit > 0:
        holdout = holdout[: args.holdout_limit]

    cfg = AgentSearchConfig.from_dict(load_config())
    kw_min = float(cfg.qi.l0_llm_entity.keyword_min_probability) / 100.0
    model = _grounding_model()
    print(f"model={model} kw_min={kw_min} failed={len(failed)} holdout={len(holdout)}")
    complete = _build_grounding_complete()

    print("Running failed-abstain cases...")
    failed_rows = _run_cases(
        failed, complete=complete, kw_min=kw_min, workers=args.workers, label="failed"
    )
    print("Running sonnet-positive holdout...")
    holdout_rows = _run_cases(
        holdout, complete=complete, kw_min=kw_min, workers=args.workers, label="holdout"
    )

    failed_summary = _summarize_failed(failed_rows)
    holdout_summary = _summarize_holdout(holdout_rows)
    baseline = {
        "n": len(failed),
        "abstain_correct": 0,
        "abstain_rate": 0.0,
        "false_positive_rows": len(failed),
        "note": (
            "Sheet baseline: Sonnet=[] and identified_keywords non-empty "
            "for every failed case"
        ),
    }
    report = {
        "model": model,
        "prompt_tag": cfg.qi.l0_llm_entity.prompt_tag,
        "keyword_min_probability": cfg.qi.l0_llm_entity.keyword_min_probability,
        "baseline_failed_abstain": baseline,
        "improved_failed_abstain": failed_summary,
        "holdout_sonnet_positive": holdout_summary,
    }
    _OUT.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.out_suffix.strip()}" if args.out_suffix.strip() else ""
    (_OUT / f"failed_abstain_results{suffix}.json").write_text(
        json.dumps(failed_rows, indent=2)
    )
    (_OUT / f"holdout_results{suffix}.json").write_text(
        json.dumps(holdout_rows, indent=2)
    )
    (_OUT / f"abstain_eval_report{suffix}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
