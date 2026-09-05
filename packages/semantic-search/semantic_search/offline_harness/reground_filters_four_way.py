#!/usr/bin/env python3
"""Offline QA harness: LLM vs regex L0 filter extraction, grounded uniformly via :8085
(part of ``semantic_search.offline_harness``).

Purpose: for every query in the test suites, run L0 filter extraction 4 ways at once
and diff the results, so LLM-vs-regex disagreements surface *before* they reach prod.

Outcome: `results_*.json` (per_query source for --analysis-only) plus
**sheet-wise JSON** matching xlsx sheets (`pairwise_summary_*.json` with holdout +
keyword pairwise accuracy, `missing_extra_vs_llmj_*.json`, `grounding_drops_*.json`)
and `reground_*.xlsx`. No separate holdout_*.json / fail_only_*.json.

Keyword compare (term sets; probabilities kept on ``*_keywords_detail`` for audit):
LLMJ × QIE_Only_LLM × Full_Search_LLM — ``keyword_status`` OK/DIFF when all three
agree; pairwise exact/DIFF% in ``pairwise_summary["keywords"]``.

Four arms run **in parallel per query**:

  1. LLMJ — offline LLM extraction (``GROUNDING_MODEL`` = env ``L0_GROUNDING_MODEL``
          or primary from ``task_model_allowlists.l0_entity_extraction`` ∩ discovery;
          same L0 prompts as prod — plain or, above ``rewrite_threshold`` tokens, the
          combined rewrite+extract prompt, mirroring prod's token gate), then
          reconciled + live-inventory-grounded via ``POST :8085/internal/l0_ground``
          (the same ``reconcile_and_ground_identified`` pipeline qie_only uses in prod).
  2. QIE_Only_LLM — live qie_only LLM (POST :8085/search qie_only_mode=true). Exercises
          the actual Phase 1 filters-only code path end to end.
  3. Full_Search_LLM — live full-search LLM (POST :8085/search; hard
          ``pipeline_trace.applied_filters`` only — soft_signals excluded from filter DIFF).
  4. Regex — offline regex fallback (``L0RegexFilterExtractor`` over ``RegexEntityExtractor``,
          same parser qie_only/full-search use on LLM outage), then reconciled +
          live-inventory-grounded via the same ``/internal/l0_ground`` route.

No arm builds its own Qdrant/ClickHouse/config wiring — every arm's grounding runs
through the already-running :8085 service (single source of truth), so all 4 arms are
directly comparable. OK = all 4 arms produce the identical filter set. When they don't,
the full C(4,2)=6-pair table below shows exactly which arm(s) disagree:
LLMJ×QIE, LLMJ×Full, QIE×Full, LLMJ×Regex, QIE×Regex, Full×Regex.

Default suites: test_search_queries.md + test_filter_queries.md

Query concurrency defaults from config/base.yaml's
``offline_eval.retrieval_eval.max_concurrent_queries``, hard-capped at 2
(``COMPARE_CONCURRENCY`` env overrides, still capped at 2).

Run directly (python one-liner, from auc-semantic-search/, with :8085 already running):
    python -m semantic_search.offline_harness.reground_filters_four_way --seed \\
        --md test_search_queries.md --md test_filter_queries.md

    python -m semantic_search.offline_harness.reground_filters_four_way --fail-only
    python -m semantic_search.offline_harness.reground_filters_four_way --error-only   # retry status=ERROR only
    python -m semantic_search.offline_harness.reground_filters_four_way --analysis-only \\
        --results output/reground_four_way/results_YYYYMMDDTHHMMSSZ.json  # rebuild xlsx from JSON, no HTTP/LLM

    # Formal holdout (default on): stratified train/test focus pairs in
    # pairwise_summary_*.json["holdout"] + appended on pairwise_summary xlsx sheet
    # (no separate holdout sheet/json; no re-query).
    python -m semantic_search.offline_harness.reground_filters_four_way --analysis-only \\
        --results output/reground_four_way/results_YYYYMMDDTHHMMSSZ.json \\
        --holdout-frac 0.2 --holdout-seed 42

Run over HTTP (see ``POST /internal/harness/reground-four-way`` in app.py):
    curl -sS -X POST http://localhost:8085/internal/harness/reground-four-way \\
        -F seed=true -F limit=50
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import itertools
import json
import os
import re
import sys
import tempfile
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

import requests

# Transient HTTP statuses the API documents as retryable (rate limit + load cancel).
# RateLimitMiddleware -> 429; cancelled BaseHTTPMiddleware work -> 503
# ("request_cancelled: server busy under load, retry").
# Keep retries SHORT — prevent overload via concurrency/HTTP caps, not long backoff.
_RETRYABLE_HTTP = frozenset({429, 502, 503, 504})
# Exactly one retry (2 attempts total). Env cannot raise this.
_HTTP_MAX_ATTEMPTS = 2
_GROUND_MAX_ATTEMPTS = 2
# Hard cap: at most 2 queries in flight (prevents local uvicorn 503 under load).
_MAX_QUERY_CONCURRENCY = 2

# offline_harness/ -> semantic_search/ -> semantic-search/ -> packages/ -> auc-semantic-search/
_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MDS = [
    _REPO_ROOT / "test_search_queries.md",
    _REPO_ROOT / "test_filter_queries.md",
]
ENDPOINT = os.environ.get(
    "L0_COMPARE_ENDPOINT",
    os.environ.get("SEARCH_ENDPOINT", "http://localhost:8085/search"),
)
# Same host as ENDPOINT's /search — the additive, read-only reconcile+ground route
# (see app.py POST /internal/l0_ground) both offline arms (LLMJ, Regex) call for
# grounding, instead of duplicating Qdrant/ClickHouse/config wiring in this script.
L0_GROUND_ENDPOINT = os.environ.get(
    "L0_GROUND_ENDPOINT",
    re.sub(r"/search$", "/internal/l0_ground", ENDPOINT),
)
# At most one /search (or /internal/l0_ground) in flight globally.
_HTTP_INFLIGHT = max(1, min(2, int(os.environ.get("COMPARE_HTTP_INFLIGHT", "1"))))
PROGRESS_EVERY = max(1, int(os.environ.get("COMPARE_PROGRESS_EVERY", "25")))
# Wall-clock heartbeat, independent of PROGRESS_EVERY's completion-count trigger —
# fires even if no query has finished in the interval (e.g. one arm hanging).
PROGRESS_INTERVAL_SECONDS = max(
    10, int(os.environ.get("COMPARE_PROGRESS_SECONDS", "90"))
)

# Model-name prefixes whose API rejects an explicit temperature kwarg.
_NO_TEMPERATURE_MODEL_PREFIXES = tuple(
    os.environ.get("L0_NO_TEMPERATURE_MODEL_PREFIXES", "o1,o3,o4,gpt-5").split(",")
)

# Arm keys (stable order for reports).
ARM_LLMJ = "LLMJ"
ARM_QIE = "QIE_Only_LLM"
ARM_FULL = "Full_Search_LLM"
ARM_REGEX = "Regex"
ARMS = (ARM_LLMJ, ARM_QIE, ARM_FULL, ARM_REGEX)
# Full C(4,2)=6-pair comparison matrix — uniform grounding means every pair is fair.
ARM_PAIRS: Tuple[Tuple[str, str], ...] = tuple(itertools.combinations(ARMS, 2))
# Keyword accuracy focus: three LLM extract arms (Regex topical fill is not LLM keywords).
ARM_KEYWORD = (ARM_LLMJ, ARM_QIE, ARM_FULL)
ARM_KEYWORD_PAIRS: Tuple[Tuple[str, str], ...] = tuple(
    itertools.combinations(ARM_KEYWORD, 2)
)
ARM_LABEL = {
    ARM_LLMJ: "offline LLM extract + API-grounded",
    ARM_QIE: "qie_only LLM (live)",
    ARM_FULL: "full-search LLM (live)",
    ARM_REGEX: "offline regex fallback + API-grounded",
}


def _arm_set_key(arm: str) -> str:
    return f"{arm}_set"


def _arm_ms_key(arm: str) -> str:
    """Work latency: extract_ms + http_ms (excludes http_sem queue wait)."""
    return f"{arm}_ms"


def _arm_cost_key(arm: str) -> str:
    """Per-arm LLM spend in USD (Regex=0; QIE/Full from decision_cost_usd)."""
    return f"{arm}_cost_usd"


def _arm_extract_ms_key(arm: str) -> str:
    """Client-side extract only (LLMJ offline LLM / Regex local parse). 0 for live /search arms."""
    return f"{arm}_extract_ms"


def _arm_http_ms_key(arm: str) -> str:
    """HTTP round-trip from ``_retry_post`` (retries/backoff included; sem wait excluded)."""
    return f"{arm}_http_ms"


def _arm_queue_ms_key(arm: str) -> str:
    """Time waiting on ``http_sem`` before the HTTP call starts (harness contention, not arm cost)."""
    return f"{arm}_queue_ms"


def _arm_http_queue_ms_key(arm: str) -> str:
    """Aggregated HTTP round-trip + http_sem queue wait (ms)."""
    return f"{arm}_http_queue_ms"


def _latency_parts(
    *,
    extract_ms: float = 0.0,
    http_ms: float = 0.0,
    queue_ms: float = 0.0,
) -> Dict[str, float]:
    """Normalized per-arm latency breakdown.

    ``ms`` (work) = extract + http. Queue wait is recorded separately so contention
    under ``COMPARE_HTTP_INFLIGHT`` does not inflate arm-to-arm work comparisons.
    """
    e = float(extract_ms)
    h = float(http_ms)
    q = float(queue_ms)
    return {"extract_ms": e, "http_ms": h, "queue_ms": q, "ms": e + h}


def _arm_kw_set_key(arm: str) -> str:
    return f"{arm}_kw_set"


def _arm_kw_detail_key(arm: str) -> str:
    """Per-arm keyword list with probabilities (audit; not used for set equality)."""
    return f"{arm}_keywords_detail"


def _fill_blank_env_from_dotenv(path: Path) -> None:
    """Fill missing/blank os.environ keys from dotenv (override=False skips blanks)."""
    try:
        from dotenv import dotenv_values  # noqa: PLC0415 — soft optional dep, gated by try/except
    except ImportError:
        return
    if not path.is_file():
        return
    for key, val in (dotenv_values(path) or {}).items():
        if not key or val is None:
            continue
        cur = os.environ.get(key)
        if cur is None or not str(cur).strip():
            os.environ[key] = val


# Cursor/shell often set OPENAI_API_KEY=""; fill from module .env, then workspace .env.
_fill_blank_env_from_dotenv(_REPO_ROOT / ".env")
_fill_blank_env_from_dotenv(_REPO_ROOT.parent / ".env")

sys.path.insert(0, str(_REPO_ROOT / "packages" / "llm-core"))
sys.path.insert(0, str(_REPO_ROOT / "packages" / "semantic-search"))

from semantic_search.config.loader import load_config  # noqa: E402
from semantic_search.config.models import AgentSearchConfig  # noqa: E402
from semantic_search.config.offline_harness_defaults import (  # noqa: E402
    resolve_harness_grounding_model,
    resolve_openai_compat_api_key_for_model,
)
from semantic_search.qi.l0_llm_filter_extractor import (  # noqa: E402
    FILTERABLE_PARAMS,
    L0_FILTER_SYSTEM,
    build_l0_combined_user_prompt,
    build_l0_filter_user_prompt,
    expand_gd_to_godaddy,
    filters_to_identified,
)
from llm_core.pricing import compute_call_cost_usd  # noqa: E402

try:
    _CONFIG_MAX_CONCURRENCY = (
        AgentSearchConfig.from_dict(load_config())
        .offline_eval.retrieval_eval.max_concurrent_queries
    )
except Exception:  # noqa: BLE001 — config-sourced default only; hard cap below still applies
    _CONFIG_MAX_CONCURRENCY = _MAX_QUERY_CONCURRENCY
CONCURRENCY = max(
    1,
    min(
        _MAX_QUERY_CONCURRENCY,
        int(os.environ.get("COMPARE_CONCURRENCY", str(_CONFIG_MAX_CONCURRENCY))),
    ),
)

# Lazy: env L0_GROUNDING_MODEL, else allowlist primary (resolved on first LLM use).
# Module import must stay cheap for --seed / --analysis-only (no discovery at import).
_GROUNDING_MODEL_CACHE: Optional[str] = None


def _grounding_model() -> str:
    """Resolved LLMJ model id (env override or l0_entity_extraction primary)."""
    global _GROUNDING_MODEL_CACHE  # noqa: PLW0603
    if _GROUNDING_MODEL_CACHE is None:
        _GROUNDING_MODEL_CACHE = resolve_harness_grounding_model()
    return _GROUNDING_MODEL_CACHE

_QI_CONFIG = AgentSearchConfig.from_dict(load_config()).qi
if _QI_CONFIG.entity_slots is None or _QI_CONFIG.l0_llm_entity is None:
    raise SystemExit("ERROR: qi.entity_slots and qi.l0_llm_entity required in base.yaml")

# ── Filter-name canonicalization (own copy — no import from extract_test_filters_queries.py) ──

# Only this not_applied reason is merged into predicted applied_filters.
_NOT_APPLIED_INCLUDE_REASONS = frozenset({"filter_relaxed_by_guard"})

# Optional legacy -> canonical rewrite. Prefer exact names from find_api_params.json.
_ALIAS_PATH = _REPO_ROOT / "filter_param_aliases.json"
_PARAM_ALIASES: Dict[str, str] = {}
if _ALIAS_PATH.is_file():
    try:
        _PARAM_ALIASES = {
            str(k): str(v)
            for k, v in dict(
                json.loads(_ALIAS_PATH.read_text()).get("aliases") or {}
            ).items()
        }
    except (OSError, json.JSONDecodeError, TypeError, AttributeError) as _alias_err:
        print(f"WARN: filter_param_aliases load failed ({_alias_err}) — no alias rewrite")
        _PARAM_ALIASES = {}

# Case-insensitive lookup -> canonical spelling (FIND catalog wins over aliases).
_PARAM_ALIASES_LOWER: Dict[str, str] = {
    str(k).lower(): str(v) for k, v in _PARAM_ALIASES.items()
}
_FILTERABLE_LOWER: Dict[str, str] = {str(p).lower(): str(p) for p in FILTERABLE_PARAMS}

# Entity slot -> FIND api_param (for query_intelligence.filters.identified entries).
_SLOT_MAP_PATH = (
    _REPO_ROOT / "packages/semantic-search/semantic_search/qi/entity_slot_to_api_param.json"
)
try:
    _SLOT_TO_API: Dict[str, str] = {}
    for _m in json.loads(_SLOT_MAP_PATH.read_text()).get("mappings") or []:
        _slot = _m.get("entity_slot")
        if not _slot:
            continue
        _ap = _m.get("api_param")
        # no_api_param / null -> keep slot name when it is a known filterable param
        _SLOT_TO_API[str(_slot)] = str(_ap) if _ap else str(_slot)
except (OSError, json.JSONDecodeError, TypeError, AttributeError) as _slot_err:
    print(f"WARN: entity_slot_to_api_param load failed ({_slot_err}) — slot names used as-is")
    _SLOT_TO_API = {}


def _canonicalize_param(name: str) -> str:
    """Rewrite legacy name via aliases; case-insensitive -> canonical FIND spelling."""
    n = str(name or "").strip()
    if not n:
        return ""
    key = n.lower()
    if key in _PARAM_ALIASES_LOWER:
        return _PARAM_ALIASES_LOWER[key]
    if key in _FILTERABLE_LOWER:
        return _FILTERABLE_LOWER[key]
    return n


def _param_name_from_filter(entry: dict) -> str:
    """Resolve FIND api_param name from applied_filters / not_applied entry."""
    api_param = entry.get("api_param") or {}
    if isinstance(api_param, dict) and api_param.get("name"):
        return _canonicalize_param(str(api_param["name"]))
    if isinstance(api_param, str) and api_param.strip():
        return _canonicalize_param(api_param.strip())
    slot = str(entry.get("name") or "").strip()
    if not slot:
        return ""
    return _canonicalize_param(_SLOT_TO_API.get(slot, slot))


# Keyword filter slots whose *values* are lexical terms (not modes like keyword_match_mode).
# Matched via keyword sets alongside body["keywords"] / pipeline_trace.applied_keywords.
_KEYWORD_VALUE_PARAMS: frozenset = frozenset(
    {
        "keyword_contains",
        "keyword_starts_with",
        "keyword_ends_with",
        "keyword_phrase",
        "keyword_contains_exclude",
    }
)


def _collect_applied_filter_entries(response: dict) -> list:
    """identified ∪ applied filters ∪ not_applied(filter_relaxed_by_guard), deduped.

    qie_only: top-level ``identified_filters`` (no ``applied_filters`` on that path).
    Full search: ``pipeline_trace.applied_filters`` and/or top-level ``applied_filters``,
    plus ``query_intelligence.filters.identified`` when present. Sources are unioned
    (identified first, then applied) so keyword/soft terms on either side still match.

    Dedup key = FIND api_param name. Soft / unsupported / column_data_unavailable
    not_applied reasons are excluded.
    """
    applied: list = []
    seen_ids: set = set()
    for bucket in (
        response.get("identified_filters") or [],
        response.get("applied_filters") or [],
        (response.get("pipeline_trace") or {}).get("applied_filters") or [],
        ((response.get("query_intelligence") or {}).get("filters") or {}).get(
            "identified"
        )
        or [],
    ):
        for e in bucket:
            if not e or not isinstance(e, dict):
                continue
            key = (str(e.get("name") or e.get("param") or ""), repr(e.get("value")))
            if key in seen_ids:
                continue
            seen_ids.add(key)
            applied.append(e)

    qi = response.get("query_intelligence") or {}
    not_applied = (qi.get("filters") or {}).get("not_applied") or []
    relaxed = [
        e
        for e in not_applied
        if e and str(e.get("reason") or "") in _NOT_APPLIED_INCLUDE_REASONS
    ]

    by_param: dict = {}
    order: list = []
    for entry, origin in [(e, "applied") for e in applied] + [
        (e, "filter_relaxed_by_guard") for e in relaxed
    ]:
        param_name = _param_name_from_filter(entry)
        if not param_name or param_name in by_param:
            continue
        by_param[param_name] = {
            "param": param_name,
            "value": entry.get("value", ""),
            "origin": origin,
            "source": entry.get("source") or origin,
        }
        order.append(param_name)
    return [by_param[p] for p in order]


def _normalize_value(v: Any) -> str:
    """Normalize filter value: lowercase, split tokens, sort unique (order-invariant)."""
    if isinstance(v, list):
        tokens = [str(x).strip().lower() for x in v if str(x).strip()]
    else:
        s = str(v).strip()
        s = re.sub(r"[\[\]'\"]", "", s)
        s = re.sub(r",\s*", "|", s).strip("|").lower()
        tokens = [t.strip() for t in s.split("|") if t.strip()]
    return "|".join(sorted(set(tokens)))


def parse_md_queries(md_path: str) -> list:
    queries = []
    suite = "FILTER SEARCH"
    section = ""
    with open(md_path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("### "):
                section = line[4:].strip()
            elif re.match(r"^\d+\.\s+", line):
                query = re.sub(r"^\d+\.\s+", "", line).strip()
                if query:
                    queries.append((suite, section, query))
    return queries


def _suite_label(md_path: Path) -> str:
    name = md_path.name.lower()
    if "search" in name:
        return "SEARCH"
    if "filter" in name:
        return "FILTER"
    return md_path.stem.upper()


def seed_rows_from_mds(md_paths: List[Path]) -> List[dict]:
    """Build PENDING rows from numbered queries in markdown suites."""
    rows: List[dict] = []
    for md in md_paths:
        if not md.is_file():
            raise FileNotFoundError(f"MD not found: {md}")
        suite = _suite_label(md)
        parsed = parse_md_queries(str(md))
        for _suite, section, query in parsed:
            row = {
                "query_index": len(rows) + 1,
                "suite": suite,
                "source_md": md.name,
                "section": section,
                "query": query,
                "status": "PENDING",
            }
            for arm in ARMS:
                row[_arm_set_key(arm)] = []
            rows.append(row)
    return rows


def _param_of_token(tok: str) -> str:
    return tok.split(":", 1)[0] if ":" in tok else tok


def _canonicalize_token_set(tokens: Any) -> Set[str]:
    """Re-run stored ``param:value`` tokens through ``_entries_to_set`` norms.

    Lets analysis-only / older results pick up general value canonicalization
    (e.g. auction label <-> id) without re-hitting the live service. No
    query-specific rules — whatever ``_entries_to_set`` normalizes applies.
    """
    entries: List[Dict[str, Any]] = []
    for tok in tokens or []:
        s = str(tok or "").strip()
        if not s:
            continue
        if ":" in s:
            param, val = s.split(":", 1)
        else:
            param, val = s, ""
        entries.append({"param": param, "value": val})
    return set(_entries_to_set(entries))


def _arm_set(row: dict, arm: str) -> Set[str]:
    return _canonicalize_token_set(row.get(_arm_set_key(arm)) or [])


def _arm_drop_set(row: dict, key: str) -> Set[str]:
    return _canonicalize_token_set(row.get(key) or [])


def _arm_kw_set(row: dict, arm: str) -> Set[str]:
    return set(row.get(_arm_kw_set_key(arm)) or [])


def _pct(numer: int, denom: int) -> str:
    return f"{100.0 * numer / denom:.1f}%" if denom else "n/a"


def _mean_ms(vals: List[float]) -> Optional[float]:
    return round(sum(vals) / len(vals), 1) if vals else None


def _percentile_ms(vals: List[float], p: float) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return round(s[0], 1)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return round(s[f] + (s[c] - s[f]) * (k - f), 1)


def _arm_latency_stats(rows: List[dict]) -> Dict[str, Dict[str, Any]]:
    """Per-arm latency over OK/DIFF only (ERROR zeros excluded).

    Prefer split columns when present; fall back to legacy ``*_ms``-only rows
    (pre-split harness) so --analysis-only on old results still reports work_ms.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for arm in ARMS:
        extract: List[float] = []
        http: List[float] = []
        queue: List[float] = []
        http_queue: List[float] = []
        work: List[float] = []
        for r in rows:
            if r.get("status") not in ("OK", "DIFF"):
                continue
            e = r.get(_arm_extract_ms_key(arm))
            h = r.get(_arm_http_ms_key(arm))
            q = r.get(_arm_queue_ms_key(arm))
            hq = r.get(_arm_http_queue_ms_key(arm))
            m = r.get(_arm_ms_key(arm))
            if e is None and h is None and m is None:
                continue
            if e is not None:
                extract.append(float(e))
            if h is not None:
                http.append(float(h))
            if q is not None:
                queue.append(float(q))
            if hq is not None:
                http_queue.append(float(hq))
            elif h is not None or q is not None:
                http_queue.append(float(h or 0.0) + float(q or 0.0))
            if m is not None:
                work.append(float(m))
            else:
                work.append(float(e or 0.0) + float(h or 0.0))
        out[arm] = {
            "n": len(work),
            "work_ms_mean": _mean_ms(work),
            "work_ms_p50": _percentile_ms(work, 50.0),
            "work_ms_p95": _percentile_ms(work, 95.0),
            "extract_ms_mean": _mean_ms(extract),
            "http_ms_mean": _mean_ms(http),
            "queue_ms_mean": _mean_ms(queue),
            "http_queue_ms_mean": _mean_ms(http_queue),
            "http_queue_ms_p50": _percentile_ms(http_queue, 50.0),
            "http_queue_ms_p95": _percentile_ms(http_queue, 95.0),
        }
    return out


def _arm_cost_stats(rows: List[dict]) -> Dict[str, Dict[str, Any]]:
    """Per-arm LLM cost_usd over OK/DIFF only (ERROR rows excluded)."""
    out: Dict[str, Dict[str, Any]] = {}
    for arm in ARMS:
        vals: List[float] = []
        for r in rows:
            if r.get("status") not in ("OK", "DIFF"):
                continue
            c = r.get(_arm_cost_key(arm))
            if c is None:
                continue
            vals.append(float(c))
        out[arm] = {
            "n": len(vals),
            "sum_usd": round(sum(vals), 6) if vals else None,
            "mean_usd": round(sum(vals) / len(vals), 6) if vals else None,
            "p50_usd": (
                round(sorted(vals)[len(vals) // 2], 6) if vals else None
            ),
        }
    totals = [
        float(r["total_cost_usd"])
        for r in rows
        if r.get("status") in ("OK", "DIFF") and r.get("total_cost_usd") is not None
    ]
    out["_query_total"] = {
        "n": len(totals),
        "sum_usd": round(sum(totals), 6) if totals else None,
        "mean_usd": round(sum(totals) / len(totals), 6) if totals else None,
    }
    return out


def _cost_by_suite_section(rows: List[dict]) -> List[Dict[str, Any]]:
    """Suite x section cost rollup (OK/DIFF only)."""
    buckets: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in rows:
        if r.get("status") not in ("OK", "DIFF"):
            continue
        key = (str(r.get("suite") or ""), str(r.get("section") or ""))
        b = buckets.setdefault(
            key,
            {
                "suite": key[0],
                "section": key[1],
                "n": 0,
                "total_usd": 0.0,
                **{a: 0.0 for a in ARMS},
            },
        )
        b["n"] += 1
        for arm in ARMS:
            c = r.get(_arm_cost_key(arm))
            if c is not None:
                b[arm] = float(b[arm]) + float(c)
        t = r.get("total_cost_usd")
        if t is not None:
            b["total_usd"] = float(b["total_usd"]) + float(t)
    rows_out: List[Dict[str, Any]] = []
    for (_suite, _section), b in sorted(
        buckets.items(), key=lambda kv: (-kv[1]["total_usd"], kv[0])
    ):
        n = int(b["n"]) or 1
        rows_out.append(
            {
                "suite": b["suite"],
                "section": b["section"],
                "n": b["n"],
                "total_usd": round(float(b["total_usd"]), 6),
                "mean_total_usd": round(float(b["total_usd"]) / n, 6),
                **{arm: round(float(b[arm]), 6) for arm in ARMS},
            }
        )
    return rows_out


def _sets_agree_after_restoring_drops(
    set_a: Set[str],
    set_b: Set[str],
    *,
    drops_a: Set[str],
    drops_b: Set[str],
) -> bool:
    """True when restoring each side's drops of tokens the peer still has yields equal sets.

    General rule (no query-/param-specific branches): a drop on side S explains a
    mismatch token T only when T is in the peer's applied set and in S's drop list.
    Restores are computed from the original sets only (order-independent).
    """
    sa = set(set_a) | ((set_b - set_a) & drops_a)
    sb = set(set_b) | ((set_a - set_b) & drops_b)
    return sa == sb


def _not_applied_as_entry(e: dict) -> Dict[str, Any]:
    """Keep name + api_param so ``_entries_to_set`` canonicalizes the same as arm sets."""
    return {
        "name": e.get("name"),
        "param": e.get("param") or e.get("name"),
        "value": e.get("value"),
        "api_param": e.get("api_param"),
    }


def _results_beside_xlsx(xlsx: Path) -> Optional[Path]:
    """Map reground_{stamp}.xlsx to sibling results_{stamp}.json when present."""
    name = xlsx.name
    if not (name.startswith("reground_") and name.endswith(".xlsx")):
        return None
    stamp = name[len("reground_") : -len(".xlsx")]
    cand = xlsx.with_name(f"results_{stamp}.json")
    return cand if cand.is_file() else None


def _resolve_results_path(
    explicit: Optional[Path],
    out_dir: Path,
    *,
    from_xlsx: Optional[Path] = None,
) -> Path:
    """Resolve results_*.json for --analysis-only.

    Preference: --results, else sibling of --from-xlsx, else latest under out_dir.
    Excel alone is not enough (per_query omits {arm}_set lists); the matching
    results_{stamp}.json next to reground_{stamp}.xlsx is the recompute source.
    """
    if explicit is not None:
        path = explicit.resolve()
        if not path.is_file():
            raise SystemExit(f"--results not found: {path}")
        return path
    if from_xlsx is not None:
        xlsx = from_xlsx.resolve()
        if not xlsx.is_file():
            raise SystemExit(f"--from-xlsx not found: {xlsx}")
        sibling = _results_beside_xlsx(xlsx)
        if sibling is None:
            raise SystemExit(
                f"--from-xlsx {xlsx} has no sibling results_*.json "
                f"(expected results_{xlsx.stem.replace('reground_', '', 1)}.json); "
                f"pass --results explicitly"
            )
        return sibling
    candidates = sorted(out_dir.glob("results_*.json"))
    if not candidates:
        raise SystemExit(
            f"--analysis-only needs --results PATH, --from-xlsx PATH, "
            f"or a results_*.json under {out_dir}"
        )
    return candidates[-1]


def _pairwise_quality_rows(aggregates: dict) -> List[Dict[str, Any]]:
    """Build C(4,2) matrix rows: Prod exact vs Exact+grounded / Exact+gnd+pipeline /
    Exact+gnd+pipeline+soft.

    All agreement columns are **query-level** (same unit as Prod exact / N):
    a mismatched query counts toward Exact+grounded only when restoring grounding
    drops makes the two sets equal; Exact+gnd+pipeline when restoring grounding ∪
    pipeline drops makes them equal; Exact+gnd+pipeline+soft when grounding ∪
    pipeline ∪ soft-downgrade drops makes them equal. Partial token overlap that
    leaves a residual mismatch does **not** inflate agreement. Token counters
    remain separate for ranked param tables.
    """
    rows: List[Dict[str, Any]] = []
    for a, b in ARM_PAIRS:
        eq = aggregates["pair_eq"][(a, b)]
        n = aggregates["pair_n"][(a, b)]
        tot = n or 1
        mismatches = n - eq
        exact_g = aggregates["pair_eq_after_g"][(a, b)]
        exact_gp = aggregates["pair_eq_after_gp"][(a, b)]
        exact_gps = aggregates["pair_eq_after_gps"][(a, b)]
        g_explained = exact_g - eq
        p_explained = exact_gp - exact_g
        s_explained = exact_gps - exact_gp
        unexplained = n - exact_gps
        rows.append(
            {
                "pair": f"{a}={b}",
                "arm_a": a,
                "arm_a_label": ARM_LABEL[a],
                "arm_b": b,
                "arm_b_label": ARM_LABEL[b],
                "prod_exact": eq,
                "n": n,
                "prod_exact_pct": _pct(eq, tot),
                "exact_plus_grounded_dropped": exact_g,
                "exact_plus_grounded_dropped_pct": _pct(exact_g, tot),
                "exact_plus_gnd_pipeline": exact_gp,
                "exact_plus_gnd_pipeline_pct": _pct(exact_gp, tot),
                "exact_plus_gnd_pipeline_soft": exact_gps,
                "exact_plus_gnd_pipeline_soft_pct": _pct(exact_gps, tot),
                "mismatches": mismatches,
                "grounding_explained": g_explained,
                "pipeline_explained": p_explained,
                "soft_explained": s_explained,
                "unexplained": unexplained,
            }
        )
    return rows


# Formal holdout for MLS-Q1-style LLM vs regex baselines (analysis-only; no re-query).
DEFAULT_HOLDOUT_FRAC = 0.2
DEFAULT_HOLDOUT_SEED = 42
# Codebase capability pairs only: QIE / Full_Search vs Regex.
# LLMJ is an offline reference arm — not a prod capability; excluded from holdout focus.
HOLDOUT_FOCUS_PAIRS: Tuple[Tuple[str, str], ...] = (
    (ARM_QIE, ARM_REGEX),
    (ARM_FULL, ARM_REGEX),
)


def _holdout_u01(query: str, *, seed: int, stratum: str) -> float:
    """Deterministic U[0,1) from md5(seed|stratum|query)."""
    digest = hashlib.md5(
        f"{int(seed)}|{stratum}|{query}".encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()
    return int(digest[:8], 16) / float(0xFFFFFFFF)


def assign_holdout_splits(
    rows: List[dict],
    *,
    frac: float,
    seed: int,
) -> Optional[Dict[str, Any]]:
    """Stamp each row with ``holdout_split`` = ``train``|``test``.

    Stratified by ``suite`` (fallback ``source_md``) so SEARCH/FILTER each get
    ~``frac`` test. Returns meta dict, or None when holdout disabled
    (``frac <= 0`` or ``frac >= 1``).
    """
    if not (0.0 < float(frac) < 1.0):
        for r in rows:
            r.pop("holdout_split", None)
        return None
    frac_f = float(frac)
    seed_i = int(seed)
    by_suite: Dict[str, Dict[str, int]] = {}
    for r in rows:
        stratum = str(r.get("suite") or r.get("source_md") or "all")
        q = str(r.get("query") or "")
        split = "test" if _holdout_u01(q, seed=seed_i, stratum=stratum) < frac_f else "train"
        r["holdout_split"] = split
        bucket = by_suite.setdefault(stratum, {"n": 0, "train": 0, "test": 0})
        bucket["n"] += 1
        bucket[split] += 1
    n_train = sum(1 for r in rows if r.get("holdout_split") == "train")
    n_test = sum(1 for r in rows if r.get("holdout_split") == "test")
    return {
        "seed": seed_i,
        "frac": frac_f,
        "method": "md5(seed|suite|query) < frac; stratified by suite",
        "unit": "query",
        "metric": (
            "pairwise filter-set exact-match (arm agreement; "
            "not gold-label accuracy)"
        ),
        "n_all": len(rows),
        "n_train": n_train,
        "n_test": n_test,
        "by_suite": by_suite,
    }


def _pairwise_keyword_rows(aggregates: dict) -> List[Dict[str, Any]]:
    """C(3,2) keyword term-set accuracy for LLMJ × QIE × Full (exact / DIFF / Jaccard)."""
    rows: List[Dict[str, Any]] = []
    pair_kw_eq = aggregates.get("pair_kw_eq") or {}
    pair_kw_n = aggregates.get("pair_kw_n") or {}
    pair_kw_jaccard_sum = aggregates.get("pair_kw_jaccard_sum") or {}
    for a, b in ARM_KEYWORD_PAIRS:
        eq = int(pair_kw_eq.get((a, b), 0))
        n = int(pair_kw_n.get((a, b), 0))
        tot = n or 1
        diff = n - eq
        jacc_sum = float(pair_kw_jaccard_sum.get((a, b), 0.0))
        rows.append(
            {
                "pair": f"{a}={b}",
                "arm_a": a,
                "arm_a_label": ARM_LABEL[a],
                "arm_b": b,
                "arm_b_label": ARM_LABEL[b],
                "exact": eq,
                "n": n,
                "exact_pct": _pct(eq, tot),
                "diff": diff,
                "diff_pct": _pct(diff, tot),
                "jaccard_mean": round(jacc_sum / tot, 4) if n else None,
            }
        )
    return rows


def _collect_pairwise_counters(rows: List[dict]) -> Dict[str, Any]:
    """Query-level pairwise agreement counters (OK/DIFF rows only)."""
    pair_eq: Dict[Tuple[str, str], int] = {p: 0 for p in ARM_PAIRS}
    pair_n: Dict[Tuple[str, str], int] = {p: 0 for p in ARM_PAIRS}
    pair_eq_after_g: Dict[Tuple[str, str], int] = {p: 0 for p in ARM_PAIRS}
    pair_eq_after_gp: Dict[Tuple[str, str], int] = {p: 0 for p in ARM_PAIRS}
    pair_eq_after_gps: Dict[Tuple[str, str], int] = {p: 0 for p in ARM_PAIRS}
    pair_kw_eq: Dict[Tuple[str, str], int] = {p: 0 for p in ARM_KEYWORD_PAIRS}
    pair_kw_n: Dict[Tuple[str, str], int] = {p: 0 for p in ARM_KEYWORD_PAIRS}
    pair_kw_jaccard_sum: Dict[Tuple[str, str], float] = {
        p: 0.0 for p in ARM_KEYWORD_PAIRS
    }
    three_kw_exact = 0
    three_kw_n = 0
    for r in rows:
        if r.get("status") not in ("OK", "DIFF"):
            continue
        sets = {a: _arm_set(r, a) for a in ARMS}
        g_drops = {
            a: _arm_drop_set(r, _dropped_by_grounding_key(a))
            for a in GROUND_CALLER_ARMS
        }
        p_drops = {
            a: _arm_drop_set(r, _dropped_by_pipeline_key(a))
            for a in PIPELINE_ARMS
        }
        s_drops = {
            a: _arm_drop_set(r, _dropped_by_soft_key(a))
            for a in SOFT_DOWNGRADE_ARMS
        }
        for a, b in ARM_PAIRS:
            pair_n[(a, b)] += 1
            sa, sb = sets[a], sets[b]
            g_a = g_drops.get(a, set())
            g_b = g_drops.get(b, set())
            p_a = p_drops.get(a, set())
            p_b = p_drops.get(b, set())
            s_a = s_drops.get(a, set())
            s_b = s_drops.get(b, set())
            if sa == sb:
                pair_eq[(a, b)] += 1
                pair_eq_after_g[(a, b)] += 1
                pair_eq_after_gp[(a, b)] += 1
                pair_eq_after_gps[(a, b)] += 1
                continue
            if _sets_agree_after_restoring_drops(
                sa, sb, drops_a=g_a, drops_b=g_b
            ):
                pair_eq_after_g[(a, b)] += 1
            if _sets_agree_after_restoring_drops(
                sa, sb, drops_a=g_a | p_a, drops_b=g_b | p_b
            ):
                pair_eq_after_gp[(a, b)] += 1
            if _sets_agree_after_restoring_drops(
                sa, sb, drops_a=g_a | p_a | s_a, drops_b=g_b | p_b | s_b
            ):
                pair_eq_after_gps[(a, b)] += 1
        kw_sets = {a: _arm_kw_set(r, a) for a in ARM_KEYWORD}
        three_kw_n += 1
        if _keyword_three_arm_status(
            {a: sorted(kw_sets[a]) for a in ARM_KEYWORD}
        ) == "OK":
            three_kw_exact += 1
        for a, b in ARM_KEYWORD_PAIRS:
            pair_kw_n[(a, b)] += 1
            kwa, kwb = kw_sets[a], kw_sets[b]
            if kwa == kwb:
                pair_kw_eq[(a, b)] += 1
            union = kwa | kwb
            pair_kw_jaccard_sum[(a, b)] += (
                (len(kwa & kwb) / len(union)) if union else 1.0
            )
    return {
        "pair_eq": pair_eq,
        "pair_n": pair_n,
        "pair_eq_after_g": pair_eq_after_g,
        "pair_eq_after_gp": pair_eq_after_gp,
        "pair_eq_after_gps": pair_eq_after_gps,
        "pair_kw_eq": pair_kw_eq,
        "pair_kw_n": pair_kw_n,
        "pair_kw_jaccard_sum": pair_kw_jaccard_sum,
        "three_kw_exact": three_kw_exact,
        "three_kw_n": three_kw_n,
    }


def _pairwise_block_from_rows(rows: List[dict]) -> Dict[str, Any]:
    """Pairs matrix + aggregate summary for a row subset (train or test)."""
    counters = _collect_pairwise_counters(rows)
    matrix = _pairwise_quality_rows(counters)
    kw_matrix = _pairwise_keyword_rows(counters)
    agg_exact = sum(counters["pair_eq"].values())
    agg_n = sum(counters["pair_n"].values())
    agg_exact_g = sum(counters["pair_eq_after_g"].values())
    agg_exact_gp = sum(counters["pair_eq_after_gp"].values())
    agg_exact_gps = sum(counters["pair_eq_after_gps"].values())
    agg_mismatches = agg_n - agg_exact
    three_kw_exact = int(counters.get("three_kw_exact") or 0)
    three_kw_n = int(counters.get("three_kw_n") or 0)
    return {
        "pairs": matrix,
        "aggregate": {
            "unit": "query",
            "prod_exact": agg_exact,
            "n": agg_n,
            "prod_exact_pct": (
                round(100.0 * agg_exact / agg_n, 1) if agg_n else None
            ),
            "exact_plus_grounded_dropped": agg_exact_g,
            "exact_plus_grounded_dropped_pct": (
                round(100.0 * agg_exact_g / agg_n, 1) if agg_n else None
            ),
            "exact_plus_gnd_pipeline": agg_exact_gp,
            "exact_plus_gnd_pipeline_pct": (
                round(100.0 * agg_exact_gp / agg_n, 1) if agg_n else None
            ),
            "exact_plus_gnd_pipeline_soft": agg_exact_gps,
            "exact_plus_gnd_pipeline_soft_pct": (
                round(100.0 * agg_exact_gps / agg_n, 1) if agg_n else None
            ),
            "mismatches": agg_mismatches,
            "grounding_explained": agg_exact_g - agg_exact,
            "pipeline_explained": agg_exact_gp - agg_exact_g,
            "soft_explained": agg_exact_gps - agg_exact_gp,
            "unexplained": agg_n - agg_exact_gps,
        },
        "keywords": {
            "unit": "query",
            "arms": list(ARM_KEYWORD),
            "notes": (
                "Term-set exact match (casefold); probabilities on "
                "*_keywords_detail only. Threshold already applied in L0 extractor."
            ),
            "pairs": kw_matrix,
            "three_arm_exact": three_kw_exact,
            "three_arm_n": three_kw_n,
            "three_arm_exact_pct": (
                round(100.0 * three_kw_exact / three_kw_n, 1) if three_kw_n else None
            ),
            "three_arm_diff": three_kw_n - three_kw_exact,
        },
    }


def build_holdout_report(
    rows: List[dict],
    *,
    frac: float,
    seed: int,
) -> Optional[Dict[str, Any]]:
    """Assign splits and compute train/test pairwise matrices (+ MLS-Q1 focus)."""
    meta = assign_holdout_splits(rows, frac=frac, seed=seed)
    if meta is None:
        return None
    train_rows = [r for r in rows if r.get("holdout_split") == "train"]
    test_rows = [r for r in rows if r.get("holdout_split") == "test"]
    train_block = _pairwise_block_from_rows(train_rows)
    test_block = _pairwise_block_from_rows(test_rows)
    all_block = _pairwise_block_from_rows(rows)

    def _find_pair(matrix: List[Dict[str, Any]], a: str, b: str) -> Optional[Dict[str, Any]]:
        key = f"{a}={b}"
        for m in matrix:
            if m.get("pair") == key:
                return m
        return None

    # Per focus pair: train row, then test row, then all (sheet/console same order).
    _focus_splits = (
        ("train", train_block),
        ("test", test_block),
        ("all", all_block),
    )
    focus: List[Dict[str, Any]] = []
    for a, b in HOLDOUT_FOCUS_PAIRS:
        for split_name, block in _focus_splits:
            m = _find_pair(block["pairs"], a, b)
            if m is None:
                continue
            # Full_Search surface Prod exact undercounts vs Regex (pipeline/soft drops).
            # Headline for Full pairs = Exact+gnd+pipeline+soft; QIE stays Prod exact.
            involves_full = ARM_FULL in (a, b)
            focus.append(
                {
                    "split": split_name,
                    "pair": m["pair"],
                    "prod_exact": m["prod_exact"],
                    "n": m["n"],
                    "prod_exact_pct": m["prod_exact_pct"],
                    "exact_plus_grounded_dropped": m["exact_plus_grounded_dropped"],
                    "exact_plus_grounded_dropped_pct": m[
                        "exact_plus_grounded_dropped_pct"
                    ],
                    "exact_plus_gnd_pipeline": m["exact_plus_gnd_pipeline"],
                    "exact_plus_gnd_pipeline_pct": m["exact_plus_gnd_pipeline_pct"],
                    "exact_plus_gnd_pipeline_soft": m[
                        "exact_plus_gnd_pipeline_soft"
                    ],
                    "exact_plus_gnd_pipeline_soft_pct": m[
                        "exact_plus_gnd_pipeline_soft_pct"
                    ],
                    "mismatches": m["mismatches"],
                    "grounding_explained": m["grounding_explained"],
                    "pipeline_explained": m["pipeline_explained"],
                    "soft_explained": m["soft_explained"],
                    "unexplained": m["unexplained"],
                    "headline_metric": (
                        "exact_plus_gnd_pipeline_soft"
                        if involves_full
                        else "prod_exact"
                    ),
                }
            )
    return {
        **meta,
        "all": all_block,
        "train": train_block,
        "test": test_block,
        "focus": focus,
    }


def write_analysis(
    rows: List[dict],
    out_dir: Path,
    *,
    holdout_frac: float = DEFAULT_HOLDOUT_FRAC,
    holdout_seed: int = DEFAULT_HOLDOUT_SEED,
) -> Path:
    """Write pairwise / per-query / gap analysis (+ holdout when enabled)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_copy = out_dir / f"results_{stamp}.json"
    results_copy.write_text(json.dumps(rows, indent=2))

    # Pairwise exact (exclude ERROR/PENDING rows from denominator for that pair).
    pair_eq: Dict[Tuple[str, str], int] = {p: 0 for p in ARM_PAIRS}
    pair_n: Dict[Tuple[str, str], int] = {p: 0 for p in ARM_PAIRS}

    # Per-arm missing vs LLMJ (LLMJ − arm) and extras (arm − LLMJ). LLMJ is the
    # offline reference baseline: same reconcile+ground pipeline as the other
    # arms, so it's a fair comparison point for all three.
    _MISS_VS_LLMJ_ARMS = (ARM_QIE, ARM_FULL, ARM_REGEX)
    miss_vs_llmj: Dict[str, Counter] = {a: Counter() for a in _MISS_VS_LLMJ_ARMS}
    extra_vs_llmj: Dict[str, Counter] = {a: Counter() for a in _MISS_VS_LLMJ_ARMS}
    examples_miss: Dict[str, Dict[str, List[str]]] = {a: {} for a in _MISS_VS_LLMJ_ARMS}
    examples_extra: Dict[str, Dict[str, List[str]]] = {a: {} for a in _MISS_VS_LLMJ_ARMS}

    # Params identified pre-ground (raw LLMJ/Regex/QIE extract, or Full_Search_LLM's
    # reconstructed pre_ground_set) but removed by grounding enforcement (e.g. tld
    # "ai" not in live inventory) — not by the extractor. All four arms via
    # GROUND_CALLER_ARMS.
    dropped_param_ct: Dict[str, Counter] = {a: Counter() for a in GROUND_CALLER_ARMS}
    dropped_token_ct: Dict[str, Counter] = {a: Counter() for a in GROUND_CALLER_ARMS}
    dropped_examples: Dict[str, Dict[str, List[str]]] = {
        a: {} for a in GROUND_CALLER_ARMS
    }

    # Params Full_Search_LLM's applied_filters is missing for a full-search-only
    # pipeline reason (guard relaxation / backend capability / column availability)
    # — not extraction or grounding disagreement, since QIE_Only_LLM/LLMJ/Regex
    # never run this check (see PIPELINE_ARMS, _collect_full_pipeline_dropped).
    dropped_pipeline_param_ct: Dict[str, Counter] = {a: Counter() for a in PIPELINE_ARMS}
    dropped_pipeline_token_ct: Dict[str, Counter] = {a: Counter() for a in PIPELINE_ARMS}
    dropped_pipeline_examples: Dict[str, Dict[str, List[str]]] = {
        a: {} for a in PIPELINE_ARMS
    }

    # Params migrated from hard filters to rank-boost-only soft signals by
    # SoftKeywordApplier.prepare_intent before not_applied accounting even runs —
    # a distinct pipeline effect from PIPELINE_ARMS's not_applied reasons (see
    # SOFT_DOWNGRADE_ARMS, _collect_full_soft_downgraded). Only Full_Search_LLM.
    dropped_soft_param_ct: Dict[str, Counter] = {a: Counter() for a in SOFT_DOWNGRADE_ARMS}
    dropped_soft_token_ct: Dict[str, Counter] = {a: Counter() for a in SOFT_DOWNGRADE_ARMS}
    dropped_soft_examples: Dict[str, Dict[str, List[str]]] = {
        a: {} for a in SOFT_DOWNGRADE_ARMS
    }

    # Per pair (a, b): token occurrences in the symmetric diff that sit in one
    # side's own drop list (ranked param tables). Separate from query-level
    # agreement below — token counts must not be added to Prod exact.
    pair_grounding_ct: Dict[Tuple[str, str], Counter] = {p: Counter() for p in ARM_PAIRS}
    pair_pipeline_ct: Dict[Tuple[str, str], Counter] = {p: Counter() for p in ARM_PAIRS}
    pair_soft_ct: Dict[Tuple[str, str], Counter] = {p: Counter() for p in ARM_PAIRS}
    # Query-level agreement after restoring drops (same unit as pair_eq / pair_n).
    pair_eq_after_g: Dict[Tuple[str, str], int] = {p: 0 for p in ARM_PAIRS}
    pair_eq_after_gp: Dict[Tuple[str, str], int] = {p: 0 for p in ARM_PAIRS}
    pair_eq_after_gps: Dict[Tuple[str, str], int] = {p: 0 for p in ARM_PAIRS}

    # Keyword pairwise (LLMJ × QIE × Full only) — exact / DIFF / Jaccard.
    # Regex topical fill excluded: not an LLM keyword extract arm.
    pair_kw_eq: Dict[Tuple[str, str], int] = {p: 0 for p in ARM_KEYWORD_PAIRS}
    pair_kw_n: Dict[Tuple[str, str], int] = {p: 0 for p in ARM_KEYWORD_PAIRS}
    pair_kw_jaccard_sum: Dict[Tuple[str, str], float] = {
        p: 0.0 for p in ARM_KEYWORD_PAIRS
    }
    three_kw_exact = 0
    three_kw_n = 0
    _MISS_VS_LLMJ_KW_ARMS = (ARM_QIE, ARM_FULL)
    miss_kw_vs_llmj: Dict[str, Counter] = {a: Counter() for a in _MISS_VS_LLMJ_KW_ARMS}
    extra_kw_vs_llmj: Dict[str, Counter] = {a: Counter() for a in _MISS_VS_LLMJ_KW_ARMS}

    for r in rows:
        for a in GROUND_CALLER_ARMS:
            for tok in r.get(_dropped_by_grounding_key(a)) or []:
                dropped_param_ct[a][_param_of_token(tok)] += 1
                dropped_token_ct[a][tok] += 1
                bucket = dropped_examples[a].setdefault(tok, [])
                if len(bucket) < 3:
                    bucket.append(str(r.get("query") or "")[:80])
        for a in PIPELINE_ARMS:
            for tok in r.get(_dropped_by_pipeline_key(a)) or []:
                dropped_pipeline_param_ct[a][_param_of_token(tok)] += 1
                dropped_pipeline_token_ct[a][tok] += 1
                bucket = dropped_pipeline_examples[a].setdefault(tok, [])
                if len(bucket) < 3:
                    bucket.append(str(r.get("query") or "")[:80])
        for a in SOFT_DOWNGRADE_ARMS:
            for tok in r.get(_dropped_by_soft_key(a)) or []:
                dropped_soft_param_ct[a][_param_of_token(tok)] += 1
                dropped_soft_token_ct[a][tok] += 1
                bucket = dropped_soft_examples[a].setdefault(tok, [])
                if len(bucket) < 3:
                    bucket.append(str(r.get("query") or "")[:80])
        # Only completed arm runs enter pairwise / gap tables (skip PENDING empties).
        if r.get("status") not in ("OK", "DIFF"):
            continue
        sets = {a: _arm_set(r, a) for a in ARMS}
        g_drops = {
            a: _arm_drop_set(r, _dropped_by_grounding_key(a))
            for a in GROUND_CALLER_ARMS
        }
        p_drops = {
            a: _arm_drop_set(r, _dropped_by_pipeline_key(a))
            for a in PIPELINE_ARMS
        }
        s_drops = {
            a: _arm_drop_set(r, _dropped_by_soft_key(a))
            for a in SOFT_DOWNGRADE_ARMS
        }
        for a, b in ARM_PAIRS:
            pair_n[(a, b)] += 1
            sa, sb = sets[a], sets[b]
            g_a = g_drops.get(a, set())
            g_b = g_drops.get(b, set())
            p_a = p_drops.get(a, set())
            p_b = p_drops.get(b, set())
            s_a = s_drops.get(a, set())
            s_b = s_drops.get(b, set())
            if sa == sb:
                pair_eq[(a, b)] += 1
                pair_eq_after_g[(a, b)] += 1
                pair_eq_after_gp[(a, b)] += 1
                pair_eq_after_gps[(a, b)] += 1
                continue
            if _sets_agree_after_restoring_drops(
                sa, sb, drops_a=g_a, drops_b=g_b
            ):
                pair_eq_after_g[(a, b)] += 1
            if _sets_agree_after_restoring_drops(
                sa, sb, drops_a=g_a | p_a, drops_b=g_b | p_b
            ):
                pair_eq_after_gp[(a, b)] += 1
            if _sets_agree_after_restoring_drops(
                sa, sb, drops_a=g_a | p_a | s_a, drops_b=g_b | p_b | s_b
            ):
                pair_eq_after_gps[(a, b)] += 1
            # Token-level attribution for ranked param tables only.
            for tok in sorted((sb - sa) & g_a):
                pair_grounding_ct[(a, b)][tok] += 1
            for tok in sorted((sa - sb) & g_b):
                pair_grounding_ct[(a, b)][tok] += 1
            for tok in sorted((sb - sa) & p_a):
                pair_pipeline_ct[(a, b)][tok] += 1
            for tok in sorted((sa - sb) & p_b):
                pair_pipeline_ct[(a, b)][tok] += 1
            for tok in sorted((sb - sa) & s_a):
                pair_soft_ct[(a, b)][tok] += 1
            for tok in sorted((sa - sb) & s_b):
                pair_soft_ct[(a, b)][tok] += 1
        llmj = sets[ARM_LLMJ]
        # Example query text for miss/extra tables (dedupe; never leave blank when count>0).
        q_ex = (str(r.get("query") or "").strip()[:80]
                or f"query_index={r.get('query_index')}")
        for a in miss_vs_llmj:
            for tok in sorted(llmj - sets[a]):
                p = _param_of_token(tok)
                miss_vs_llmj[a][p] += 1
                bucket = examples_miss[a].setdefault(p, [])
                if len(bucket) < 5 and q_ex not in bucket:
                    bucket.append(q_ex)
            for tok in sorted(sets[a] - llmj):
                p = _param_of_token(tok)
                extra_vs_llmj[a][p] += 1
                bucket = examples_extra[a].setdefault(p, [])
                if len(bucket) < 5 and q_ex not in bucket:
                    bucket.append(q_ex)

        kw_sets = {a: _arm_kw_set(r, a) for a in ARM_KEYWORD}
        three_kw_n += 1
        kw_status = _keyword_three_arm_status(
            {a: sorted(kw_sets[a]) for a in ARM_KEYWORD}
        )
        r["keyword_status"] = kw_status
        if kw_status == "OK":
            three_kw_exact += 1
        for a, b in ARM_KEYWORD_PAIRS:
            pair_kw_n[(a, b)] += 1
            kwa, kwb = kw_sets[a], kw_sets[b]
            if kwa == kwb:
                pair_kw_eq[(a, b)] += 1
            union = kwa | kwb
            pair_kw_jaccard_sum[(a, b)] += (
                (len(kwa & kwb) / len(union)) if union else 1.0
            )
        llmj_kw = kw_sets[ARM_LLMJ]
        for a in miss_kw_vs_llmj:
            for term in llmj_kw - kw_sets[a]:
                miss_kw_vs_llmj[a][term] += 1
            for term in kw_sets[a] - llmj_kw:
                extra_kw_vs_llmj[a][term] += 1

    dropped_totals: Dict[str, int] = {
        a: sum(dropped_param_ct[a].values()) for a in GROUND_CALLER_ARMS
    }
    dropped_pipeline_totals: Dict[str, int] = {
        a: sum(dropped_pipeline_param_ct[a].values()) for a in PIPELINE_ARMS
    }
    dropped_soft_totals: Dict[str, int] = {
        a: sum(dropped_soft_param_ct[a].values()) for a in SOFT_DOWNGRADE_ARMS
    }

    agg_exact = sum(pair_eq.values())
    agg_n = sum(pair_n.values())
    agg_mismatches = agg_n - agg_exact
    agg_exact_g = sum(pair_eq_after_g.values())
    agg_exact_gp = sum(pair_eq_after_gp.values())
    agg_exact_gps = sum(pair_eq_after_gps.values())
    # Query-level explained buckets (summed over pairs).
    agg_grounding_explained_q = agg_exact_g - agg_exact
    agg_pipeline_explained_q = agg_exact_gp - agg_exact_g
    agg_soft_explained_q = agg_exact_gps - agg_exact_gp
    agg_unexplained_q = agg_n - agg_exact_gps
    # Token-level counters — ranked param tables only (not agreement %).
    agg_grounding_ct: Counter = Counter()
    for ct in pair_grounding_ct.values():
        agg_grounding_ct.update(ct)
    agg_pipeline_ct: Counter = Counter()
    for ct in pair_pipeline_ct.values():
        agg_pipeline_ct.update(ct)
    agg_soft_ct: Counter = Counter()
    for ct in pair_soft_ct.values():
        agg_soft_ct.update(ct)

    aggregates = {
        "grounding_model": _grounding_model(),
        "grounding_endpoint": L0_GROUND_ENDPOINT,
        "pair_eq": pair_eq,
        "pair_n": pair_n,
        "pair_eq_after_g": pair_eq_after_g,
        "pair_eq_after_gp": pair_eq_after_gp,
        "pair_eq_after_gps": pair_eq_after_gps,
        "pair_grounding_ct": pair_grounding_ct,
        "pair_pipeline_ct": pair_pipeline_ct,
        "pair_soft_ct": pair_soft_ct,
        "agg_exact": agg_exact,
        "agg_n": agg_n,
        "agg_mismatches": agg_mismatches,
        "agg_exact_g": agg_exact_g,
        "agg_exact_gp": agg_exact_gp,
        "agg_exact_gps": agg_exact_gps,
        "agg_grounding_ct": agg_grounding_ct,
        # Query-level explained (sheet / Exact+ columns). Legacy key names kept so
        # write_xlsx aggregate rows stay wired; values are query counts not tokens.
        "agg_grounding_dropped": agg_grounding_explained_q,
        "agg_pipeline_ct": agg_pipeline_ct,
        "agg_pipeline_dropped": agg_pipeline_explained_q,
        "agg_soft_ct": agg_soft_ct,
        "agg_soft_dropped": agg_soft_explained_q,
        "agg_unexplained": agg_unexplained_q,
        "miss_vs_llmj_arms": _MISS_VS_LLMJ_ARMS,
        "miss_vs_llmj": miss_vs_llmj,
        "extra_vs_llmj": extra_vs_llmj,
        "examples_miss": examples_miss,
        "examples_extra": examples_extra,
        "pair_kw_eq": pair_kw_eq,
        "pair_kw_n": pair_kw_n,
        "pair_kw_jaccard_sum": pair_kw_jaccard_sum,
        "three_kw_exact": three_kw_exact,
        "three_kw_n": three_kw_n,
        "miss_kw_vs_llmj_arms": _MISS_VS_LLMJ_KW_ARMS,
        "miss_kw_vs_llmj": miss_kw_vs_llmj,
        "extra_kw_vs_llmj": extra_kw_vs_llmj,
        "dropped_param_ct": dropped_param_ct,
        "dropped_token_ct": dropped_token_ct,
        "dropped_examples": dropped_examples,
        "dropped_totals": dropped_totals,
        "dropped_pipeline_param_ct": dropped_pipeline_param_ct,
        "dropped_pipeline_token_ct": dropped_pipeline_token_ct,
        "dropped_pipeline_examples": dropped_pipeline_examples,
        "dropped_pipeline_totals": dropped_pipeline_totals,
        "dropped_soft_param_ct": dropped_soft_param_ct,
        "dropped_soft_token_ct": dropped_soft_token_ct,
        "dropped_soft_examples": dropped_soft_examples,
        "dropped_soft_totals": dropped_soft_totals,
        "results_copy": results_copy,
    }
    # Pairwise quality rows: Prod exact (surface after grounding/pipeline) vs
    # Exact+grounded-dropped / Exact+gnd+pipeline (extraction-adjusted agreement).
    pairwise_matrix = _pairwise_quality_rows(aggregates)
    aggregates["pairwise_matrix"] = pairwise_matrix
    keyword_matrix = _pairwise_keyword_rows(aggregates)
    aggregates["keyword_matrix"] = keyword_matrix
    latency_stats = _arm_latency_stats(rows)
    aggregates["latency_stats"] = latency_stats
    cost_stats = _arm_cost_stats(rows)
    aggregates["cost_stats"] = cost_stats
    cost_by_section = _cost_by_suite_section(rows)
    aggregates["cost_by_suite_section"] = cost_by_section
    holdout_report = build_holdout_report(
        rows, frac=holdout_frac, seed=holdout_seed,
    )
    aggregates["holdout"] = holdout_report
    # Do not leave holdout_split on per-query results — holdout is only in
    # pairwise_summary_*.json["holdout"] + appended section on pairwise_summary sheet.
    for r in rows:
        r.pop("holdout_split", None)
    results_copy.write_text(json.dumps(rows, indent=2))
    pairwise_json_path = out_dir / f"pairwise_summary_{stamp}.json"
    pairwise_payload: Dict[str, Any] = {
        "grounding_model": _grounding_model(),
        "grounding_endpoint": L0_GROUND_ENDPOINT,
        "pairs": pairwise_matrix,
        "aggregate": {
            "unit": "query",
            "prod_exact": agg_exact,
            "n": agg_n,
            "prod_exact_pct": (
                round(100.0 * agg_exact / agg_n, 1) if agg_n else None
            ),
            "exact_plus_grounded_dropped": agg_exact_g,
            "exact_plus_grounded_dropped_pct": (
                round(100.0 * agg_exact_g / agg_n, 1) if agg_n else None
            ),
            "exact_plus_gnd_pipeline": agg_exact_gp,
            "exact_plus_gnd_pipeline_pct": (
                round(100.0 * agg_exact_gp / agg_n, 1) if agg_n else None
            ),
            "exact_plus_gnd_pipeline_soft": agg_exact_gps,
            "exact_plus_gnd_pipeline_soft_pct": (
                round(100.0 * agg_exact_gps / agg_n, 1) if agg_n else None
            ),
            "mismatches": agg_mismatches,
            "grounding_explained": agg_grounding_explained_q,
            "pipeline_explained": agg_pipeline_explained_q,
            "soft_explained": agg_soft_explained_q,
            "unexplained": agg_unexplained_q,
        },
        "keywords": {
            "unit": "query",
            "arms": list(ARM_KEYWORD),
            "notes": (
                "Term-set exact match (casefold); probabilities on "
                "*_keywords_detail only. Threshold already applied in L0 extractor."
            ),
            "pairs": keyword_matrix,
            "three_arm_exact": three_kw_exact,
            "three_arm_n": three_kw_n,
            "three_arm_exact_pct": (
                round(100.0 * three_kw_exact / three_kw_n, 1) if three_kw_n else None
            ),
            "three_arm_diff": three_kw_n - three_kw_exact,
            "miss_extra_vs_llmj": {
                a: {
                    "missing": _counter_top(miss_kw_vs_llmj[a]),
                    "extra": _counter_top(extra_kw_vs_llmj[a], 15),
                }
                for a in _MISS_VS_LLMJ_KW_ARMS
            },
        },
        "latency": {
            "unit": "ms",
            "notes": (
                "work_ms = extract_ms + http_ms (excludes http_sem queue). "
                "OK/DIFF rows only. QIE/Full extract_ms=0 (extract inside /search). "
                "Fair: QIE work vs Full work; LLMJ extract vs Regex extract; "
                "LLMJ http vs Regex http (both /internal/l0_ground)."
            ),
            "arms": latency_stats,
        },
        "cost": {
            "unit": "usd",
            "notes": (
                "Per-arm LLM spend on OK/DIFF only. LLMJ=offline extract usage; "
                "QIE/Full=decision_cost_usd from /search; Regex=0. "
                "total_cost_usd = sum of arm costs per query."
            ),
            "arms": {a: cost_stats.get(a) for a in ARMS},
            "query_total": cost_stats.get("_query_total"),
            "by_suite_section": cost_by_section,
        },
    }
    if holdout_report is not None:
        pairwise_payload["holdout"] = holdout_report
    pairwise_json_path.write_text(json.dumps(pairwise_payload, indent=2))
    aggregates["pairwise_json_path"] = pairwise_json_path
    # Sheet-wise JSON (one file per xlsx analysis sheet; results_*.json = per_query).
    sheet_paths = _write_reground_sheet_jsons(out_dir, stamp, aggregates, pairwise_payload)
    aggregates["sheet_json_paths"] = sheet_paths
    xlsx_path = write_xlsx(rows, out_dir, stamp, aggregates)
    # Per-arm, independently — never merged into one joint count (LLMJ, Regex, and
    # Full_Search_LLM each ground/reconstruct their pre-ground payload separately).
    for a in GROUND_CALLER_ARMS:
        n_arm = dropped_totals[a]
        if n_arm:
            dropped_list = ",".join(sorted(dropped_token_ct[a]))
            print(f"GROUNDING_DROP arm={a} n={n_arm} dropped=[{dropped_list}]")
        else:
            print(f"GROUNDING_DROP arm={a} n=0 (none dropped by grounding enforcement)")
    # Pipeline-only drops (guard/backend/column-availability) — Full_Search_LLM only.
    for a in PIPELINE_ARMS:
        n_arm = dropped_pipeline_totals[a]
        if n_arm:
            dropped_list = ",".join(sorted(dropped_pipeline_token_ct[a]))
            print(f"PIPELINE_DROP arm={a} n={n_arm} dropped=[{dropped_list}]")
        else:
            print(f"PIPELINE_DROP arm={a} n=0 (none dropped by a pipeline-only reason)")
    # Soft-downgrade drops (SoftKeywordApplier hard->rank-boost migration) — Full_Search_LLM only.
    for a in SOFT_DOWNGRADE_ARMS:
        n_arm = dropped_soft_totals[a]
        if n_arm:
            dropped_list = ",".join(sorted(dropped_soft_token_ct[a]))
            print(f"SOFT_DOWNGRADE arm={a} n={n_arm} dropped=[{dropped_list}]")
        else:
            print(f"SOFT_DOWNGRADE arm={a} n=0 (none downgraded to a soft signal)")
    # One line per pair: prod exact + extraction-adjusted (exact+drops) + explained buckets.
    for row in pairwise_matrix:
        print(
            f"PAIR {row['pair']} prod_exact={row['prod_exact']}/{row['n']} "
            f"({row['prod_exact_pct']}) "
            f"exact+grounded-dropped={row['exact_plus_grounded_dropped']}/{row['n']} "
            f"({row['exact_plus_grounded_dropped_pct']}) "
            f"exact+gnd+pipeline={row['exact_plus_gnd_pipeline']}/{row['n']} "
            f"({row['exact_plus_gnd_pipeline_pct']}) "
            f"exact+gnd+pipeline+soft={row['exact_plus_gnd_pipeline_soft']}/{row['n']} "
            f"({row['exact_plus_gnd_pipeline_soft_pct']}) "
            f"grounding_explained={row['grounding_explained']} "
            f"pipeline_explained={row['pipeline_explained']} "
            f"soft_explained={row['soft_explained']} "
            f"unexplained={row['unexplained']}"
        )
    for row in keyword_matrix:
        print(
            f"PAIR_KW {row['pair']} exact={row['exact']}/{row['n']} "
            f"({row['exact_pct']}) diff={row['diff']}/{row['n']} "
            f"({row['diff_pct']}) jaccard_mean={row['jaccard_mean']}"
        )
    print(
        f"KEYWORD_THREE exact={three_kw_exact}/{three_kw_n} "
        f"({round(100.0 * three_kw_exact / three_kw_n, 1) if three_kw_n else None}%) "
        f"diff={three_kw_n - three_kw_exact}"
    )
    print(
        f"AGGREGATE_ALL_PAIRS prod_exact={agg_exact}/{agg_n} "
        f"exact+grounded-dropped={agg_exact_g}/{agg_n} "
        f"exact+gnd+pipeline={agg_exact_gp}/{agg_n} "
        f"exact+gnd+pipeline+soft={agg_exact_gps}/{agg_n} "
        f"mismatches={agg_mismatches} "
        f"grounding_explained_queries={agg_grounding_explained_q} "
        f"pipeline_explained_queries={agg_pipeline_explained_q} "
        f"soft_explained_queries={agg_soft_explained_q} "
        f"unexplained_queries={agg_unexplained_q} "
        f"grounding_tokens=[{','.join(sorted(agg_grounding_ct)) if agg_grounding_ct else 'none'}] "
        f"pipeline_tokens=[{','.join(sorted(agg_pipeline_ct)) if agg_pipeline_ct else 'none'}] "
        f"soft_tokens=[{','.join(sorted(agg_soft_ct)) if agg_soft_ct else 'none'}]"
    )
    print(
        "LATENCY (OK/DIFF only; work=extract+http, queue=http_sem wait excluded from work)"
    )
    for arm in ARMS:
        st = latency_stats.get(arm) or {}
        print(
            f"LATENCY arm={arm} n={st.get('n', 0)} "
            f"work_mean={st.get('work_ms_mean')} "
            f"work_p50={st.get('work_ms_p50')} "
            f"work_p95={st.get('work_ms_p95')} "
            f"extract_mean={st.get('extract_ms_mean')} "
            f"http_mean={st.get('http_ms_mean')} "
            f"queue_mean={st.get('queue_ms_mean')}"
        )
    print("COST (OK/DIFF only; USD)")
    for arm in ARMS:
        st = cost_stats.get(arm) or {}
        print(
            f"COST arm={arm} n={st.get('n', 0)} "
            f"sum={st.get('sum_usd')} mean={st.get('mean_usd')} p50={st.get('p50_usd')}"
        )
    qt_cost = cost_stats.get("_query_total") or {}
    print(
        f"COST query_total n={qt_cost.get('n', 0)} "
        f"sum={qt_cost.get('sum_usd')} mean={qt_cost.get('mean_usd')}"
    )
    print(f"PAIRWISE_JSON wrote {pairwise_json_path}")
    for name, path in (aggregates.get("sheet_json_paths") or {}).items():
        if name != "pairwise_summary":
            print(f"SHEET_JSON {name} -> {path}")
    if holdout_report is not None:
        print(
            f"HOLDOUT seed={holdout_report['seed']} frac={holdout_report['frac']} "
            f"train={holdout_report['n_train']} test={holdout_report['n_test']} "
            f"method={holdout_report['method']}"
        )
        for row in holdout_report.get("focus") or []:
            print(
                f"HOLDOUT_FOCUS split={row['split']} pair={row['pair']} "
                f"prod_exact={row['prod_exact']}/{row['n']} ({row['prod_exact_pct']}) "
                f"exact+gnd={row['exact_plus_grounded_dropped']}/{row['n']} "
                f"({row['exact_plus_grounded_dropped_pct']}) "
                f"exact+gnd+pipe={row['exact_plus_gnd_pipeline']}/{row['n']} "
                f"({row['exact_plus_gnd_pipeline_pct']}) "
                f"exact+gnd+pipe+soft={row['exact_plus_gnd_pipeline_soft']}/{row['n']} "
                f"({row['exact_plus_gnd_pipeline_soft_pct']}) "
                f"headline={row.get('headline_metric')} "
                f"mismatches={row['mismatches']} "
                f"gnd_expl={row['grounding_explained']} "
                f"pipe_expl={row['pipeline_explained']} "
                f"soft_expl={row['soft_explained']} "
                f"unexplained={row['unexplained']}"
            )
    else:
        print(f"HOLDOUT disabled (holdout_frac={holdout_frac})")
    return xlsx_path


def _counter_top(ct: Counter, n: int = 25) -> List[Dict[str, Any]]:
    return [{"key": k, "count": int(v)} for k, v in ct.most_common(n)]


def _write_reground_sheet_jsons(
    out_dir: Path,
    stamp: str,
    aggregates: dict,
    pairwise_payload: Dict[str, Any],
) -> Dict[str, Path]:
    """Write one JSON per analysis xlsx sheet (sheet_name_{stamp}.json).

    ``results_{stamp}.json`` remains the per_query source for --analysis-only.
    No separate holdout_*.json — holdout lives under pairwise_summary["holdout"].
    """
    paths: Dict[str, Path] = {}
    # pairwise_summary already written by caller; record path.
    paths["pairwise_summary"] = out_dir / f"pairwise_summary_{stamp}.json"

    miss_payload: Dict[str, Any] = {"arms": {}}
    for a in aggregates["miss_vs_llmj_arms"]:
        miss_payload["arms"][a] = {
            "missing_vs_llmj": _counter_top(aggregates["miss_vs_llmj"][a]),
            "extra_vs_llmj": _counter_top(aggregates["extra_vs_llmj"][a], 15),
            "examples_miss": {
                p: examples[:3]
                for p, examples in (aggregates["examples_miss"][a] or {}).items()
            },
            "examples_extra": {
                p: examples[:3]
                for p, examples in (aggregates.get("examples_extra") or {}).get(a, {}).items()
            },
        }
    paths["missing_extra_vs_llmj"] = out_dir / f"missing_extra_vs_llmj_{stamp}.json"
    paths["missing_extra_vs_llmj"].write_text(json.dumps(miss_payload, indent=2))

    kw_block = pairwise_payload.get("keywords") or {}
    kw_payload: Dict[str, Any] = {
        "arms": list(ARM_KEYWORD),
        "notes": kw_block.get("notes"),
        "pairs": kw_block.get("pairs") or aggregates.get("keyword_matrix") or [],
        "three_arm_exact": kw_block.get("three_arm_exact", aggregates.get("three_kw_exact")),
        "three_arm_n": kw_block.get("three_arm_n", aggregates.get("three_kw_n")),
        "three_arm_exact_pct": kw_block.get("three_arm_exact_pct"),
        "three_arm_diff": kw_block.get("three_arm_diff"),
        "miss_extra_vs_llmj": kw_block.get("miss_extra_vs_llmj") or {
            a: {
                "missing": _counter_top(aggregates["miss_kw_vs_llmj"][a]),
                "extra": _counter_top(aggregates["extra_kw_vs_llmj"][a], 15),
            }
            for a in aggregates.get("miss_kw_vs_llmj_arms") or ()
        },
    }
    paths["keyword_overlap"] = out_dir / f"keyword_overlap_{stamp}.json"
    paths["keyword_overlap"].write_text(json.dumps(kw_payload, indent=2))

    gnd_payload: Dict[str, Any] = {
        "grounding": {},
        "pipeline": {},
        "soft_downgrade": {},
    }
    for a in GROUND_CALLER_ARMS:
        gnd_payload["grounding"][a] = {
            "total": aggregates["dropped_totals"][a],
            "by_token": _counter_top(aggregates["dropped_token_ct"][a]),
            "by_param": _counter_top(aggregates["dropped_param_ct"][a], 10),
        }
    for a in PIPELINE_ARMS:
        gnd_payload["pipeline"][a] = {
            "total": aggregates["dropped_pipeline_totals"][a],
            "by_token": _counter_top(aggregates["dropped_pipeline_token_ct"][a]),
            "by_param": _counter_top(aggregates["dropped_pipeline_param_ct"][a], 10),
        }
    for a in SOFT_DOWNGRADE_ARMS:
        gnd_payload["soft_downgrade"][a] = {
            "total": aggregates["dropped_soft_totals"][a],
            "by_token": _counter_top(aggregates["dropped_soft_token_ct"][a]),
            "by_param": _counter_top(aggregates["dropped_soft_param_ct"][a], 10),
        }
    paths["grounding_drops"] = out_dir / f"grounding_drops_{stamp}.json"
    paths["grounding_drops"].write_text(json.dumps(gnd_payload, indent=2))

    # Ensure pairwise file on disk matches payload (caller already wrote; no-op rewrite ok).
    paths["pairwise_summary"].write_text(json.dumps(pairwise_payload, indent=2))
    return paths


def write_xlsx(rows: List[dict], out_dir: Path, stamp: str, aggregates: dict) -> Path:
    """Write lean `reground_{stamp}.xlsx` for review (full detail stays in results_*.json).

    Sheets (order): pairwise_summary (latency/cost at top, then filter pair
    matrix, keyword pair matrix + three-arm exact, holdout), per_query
    (identified side-by-side, keyword_status + LLM keyword cols, LLMJ diffs,
    work_ms / http_queue_ms / cost_usd), missing_extra_vs_llmj, description.
    scorecard / diffs / grounding_drops sheets omitted (JSON still written).
    """
    out_path = out_dir / f"reground_{stamp}.xlsx"
    wb = Workbook()

    header_font = Font(bold=True, color="000000")
    blue_fill = PatternFill("solid", fgColor="D6E4F5")
    green_fill = PatternFill("solid", fgColor="D9EAD9")
    orange_fill = PatternFill("solid", fgColor="FCE4D6")
    purple_fill = PatternFill("solid", fgColor="E6D9F2")
    center = Alignment(horizontal="center")

    def _section(ws, row: int, title: str) -> int:
        cell = ws.cell(row=row, column=1, value=title)
        cell.font = Font(bold=True, size=12)
        return row + 1

    def _table(ws, row: int, headers: List[str], data_rows: List[list], fill) -> int:
        for c, name in enumerate(headers, 1):
            cell = ws.cell(row=row, column=c, value=name)
            cell.font = header_font
            cell.fill = fill
            cell.alignment = center
        row += 1
        for data in data_rows:
            for c, val in enumerate(data, 1):
                ws.cell(row=row, column=c, value=val)
            row += 1
        return row + 1

    def _autowidth(ws) -> None:
        for col_cells in ws.columns:
            max_len = max((len(str(c.value or "")) for c in col_cells), default=10)
            ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 2, 60)

    # --- pairwise_summary (first sheet): latency/cost then pair matrix + holdout ---
    ws4 = wb.active
    ws4.title = "pairwise_summary"
    latency_stats = aggregates.get("latency_stats") or {}
    row = _section(
        ws4, 1,
        "Latency (OK/DIFF only). work_ms = extract+http (queue excluded). "
        "http_queue_ms = http + http_sem wait (one aggregated column).",
    ) + 1
    lat_rows = [
        [
            arm,
            ARM_LABEL[arm],
            (latency_stats.get(arm) or {}).get("n"),
            (latency_stats.get(arm) or {}).get("work_ms_p50"),
            (latency_stats.get(arm) or {}).get("work_ms_p95"),
            (latency_stats.get(arm) or {}).get("http_queue_ms_p50"),
            (latency_stats.get(arm) or {}).get("http_queue_ms_p95"),
        ]
        for arm in ARMS
    ]
    row = _table(
        ws4, row,
        [
            "Arm", "Label", "N",
            "work_p50", "work_p95",
            "http_queue_p50", "http_queue_p95",
        ],
        lat_rows, purple_fill,
    )
    cost_stats = aggregates.get("cost_stats") or {}
    qt = cost_stats.get("_query_total") or {}
    row = _section(
        ws4, row,
        "LLM cost_usd (OK/DIFF only). LLMJ=offline extract; QIE/Full=decision_cost_usd; "
        "Regex=0. Query total = sum of arm costs.",
    ) + 1
    cost_rows = [
        [
            arm,
            ARM_LABEL[arm],
            (cost_stats.get(arm) or {}).get("n"),
            (cost_stats.get(arm) or {}).get("sum_usd"),
            (cost_stats.get(arm) or {}).get("p50_usd"),
        ]
        for arm in ARMS
    ]
    cost_rows.append(
        [
            "_query_total",
            "sum of arms per query",
            qt.get("n"),
            qt.get("sum_usd"),
            None,
        ]
    )
    row = _table(
        ws4, row,
        ["Arm", "Label", "N", "sum_usd", "p50_usd"],
        cost_rows, orange_fill,
    )
    row = _section(
        ws4, row,
        f"Pairwise quality - full C(4,2)=6 matrix "
        f"(grounding model: {aggregates['grounding_model']}). "
        f"Prod exact = surface agreement after grounding/pipeline. "
        f"Exact+grounded-dropped / Exact+gnd+pipeline = extraction-adjusted "
        f"(treat explained drops as agreement).",
    ) + 1
    matrix = aggregates.get("pairwise_matrix") or _pairwise_quality_rows(aggregates)
    pair_rows = [
        [
            m["pair"],
            m["prod_exact"],
            m["n"],
            m["prod_exact_pct"],
            m["exact_plus_grounded_dropped"],
            m["exact_plus_grounded_dropped_pct"],
            m["exact_plus_gnd_pipeline"],
            m["exact_plus_gnd_pipeline_pct"],
            m["exact_plus_gnd_pipeline_soft"],
            m["exact_plus_gnd_pipeline_soft_pct"],
            m["mismatches"],
            m["grounding_explained"],
            m["pipeline_explained"],
            m["soft_explained"],
            m["unexplained"],
        ]
        for m in matrix
    ]
    row = _table(
        ws4, row,
        [
            "Pair",
            "Prod exact", "N", "Prod exact%",
            "Exact+grounded-dropped", "Exact+grounded-dropped%",
            "Exact+gnd+pipeline", "Exact+gnd+pipeline%",
            "Exact+gnd+pipeline+soft", "Exact+gnd+pipeline+soft%",
            "Mismatches", "Grounding-explained", "Pipeline-explained",
            "Soft-downgrade-explained", "Unexplained",
        ],
        pair_rows, blue_fill,
    )
    kw_matrix = aggregates.get("keyword_matrix") or _pairwise_keyword_rows(aggregates)
    three_kw_exact = int(aggregates.get("three_kw_exact") or 0)
    three_kw_n = int(aggregates.get("three_kw_n") or 0)
    row = _section(
        ws4, row,
        "Keyword term-set accuracy — LLMJ × QIE_Only_LLM × Full_Search_LLM "
        "(casefold terms; probabilities excluded from equality). "
        f"Three-arm exact={three_kw_exact}/{three_kw_n} "
        f"({round(100.0 * three_kw_exact / three_kw_n, 1) if three_kw_n else 'n/a'}%); "
        f"DIFF={three_kw_n - three_kw_exact}. Regex topical fill excluded.",
    ) + 1
    kw_rows = [
        [
            m["pair"],
            m["exact"],
            m["n"],
            m["exact_pct"],
            m["diff"],
            m["diff_pct"],
            m["jaccard_mean"],
        ]
        for m in kw_matrix
    ]
    kw_rows.append(
        [
            "THREE_ARM (LLMJ=QIE=Full)",
            three_kw_exact,
            three_kw_n,
            (
                f"{round(100.0 * three_kw_exact / three_kw_n, 1)}%"
                if three_kw_n else None
            ),
            three_kw_n - three_kw_exact,
            (
                f"{round(100.0 * (three_kw_n - three_kw_exact) / three_kw_n, 1)}%"
                if three_kw_n else None
            ),
            None,
        ]
    )
    row = _table(
        ws4, row,
        [
            "Pair", "Exact", "N", "Exact%", "DIFF", "DIFF%", "Jaccard mean",
        ],
        kw_rows, green_fill,
    )
    # Latency/cost tables are at the top of this sheet.
    if aggregates["agg_grounding_ct"]:
        row = _section(ws4, row, "Params dropped by grounding (any pair), ranked by total occurrences") + 1
        drop_rows = [
            [i, tok, cnt]
            for i, (tok, cnt) in enumerate(aggregates["agg_grounding_ct"].most_common(25), 1)
        ]
        row = _table(ws4, row, ["Rank", "Param:Value", "Count (all pairs)"], drop_rows, orange_fill)
    if aggregates["agg_pipeline_ct"]:
        row = _section(
            ws4, row,
            "Params dropped by a full-search pipeline reason (guard/backend/column "
            "availability), ranked by total occurrences",
        ) + 1
        pdrop_rows = [
            [i, tok, cnt]
            for i, (tok, cnt) in enumerate(aggregates["agg_pipeline_ct"].most_common(25), 1)
        ]
        row = _table(ws4, row, ["Rank", "Param:Value", "Count (all pairs)"], pdrop_rows, orange_fill)
    if aggregates["agg_soft_ct"]:
        row = _section(
            ws4, row,
            "Params downgraded to a soft rank-boost signal by SoftKeywordApplier "
            "(Full_Search_LLM only), ranked by total occurrences",
        ) + 1
        sdrop_rows = [
            [i, tok, cnt]
            for i, (tok, cnt) in enumerate(aggregates["agg_soft_ct"].most_common(25), 1)
        ]
        row = _table(ws4, row, ["Rank", "Param:Value", "Count (all pairs)"], sdrop_rows, orange_fill)

    # Holdout train/test — appended on this same pairwise_summary sheet (no new sheet).
    # Focus = QIE vs Regex + Full_Search vs Regex only (prod capability arms). Not LLMJ vs Regex.
    holdout = aggregates.get("holdout")
    if holdout:
        focus_keys = {f"{a}={b}" for a, b in HOLDOUT_FOCUS_PAIRS}
        row = _section(
            ws4, row + 1,
            f"HOLDOUT train/test pairwise agreement "
            f"(seed={holdout['seed']}, frac={holdout['frac']}, "
            f"train={holdout['n_train']}, test={holdout['n_test']}). "
            f"Focus pairs: QIE_Only_LLM vs Regex and Full_Search_LLM vs Regex "
            f"(codebase capability vs regex baseline). LLMJ excluded. "
            f"For Full_Search vs Regex, Prod exact alone is misleading — Full drops "
            f"filters via grounding / pipeline not_applied / soft-downgrade; read "
            f"Exact+gnd+pipe+soft% (headline). QIE headline = Prod exact. "
            f"Not gold-label accuracy. Method: {holdout['method']}.",
        ) + 1
        focus_src = sorted(
            holdout.get("focus") or [],
            key=lambda f: (str(f.get("pair") or ""), str(f.get("split") or "")),
        )
        focus_rows = [
            [
                f.get("split"),
                f.get("pair"),
                f.get("headline_metric"),
                f.get("prod_exact"),
                f.get("n"),
                f.get("prod_exact_pct"),
                f.get("exact_plus_grounded_dropped_pct"),
                f.get("exact_plus_gnd_pipeline_pct"),
                f.get("exact_plus_gnd_pipeline_soft_pct"),
                f.get("grounding_explained"),
                f.get("pipeline_explained"),
                f.get("soft_explained"),
                f.get("unexplained"),
                f.get("mismatches"),
            ]
            for f in focus_src
        ]
        row = _table(
            ws4, row,
            [
                "Split", "Pair", "Headline", "Prod exact", "N", "Prod exact%",
                "Exact+gnd%", "Exact+gnd+pipe%", "Exact+gnd+pipe+soft%",
                "Gnd expl", "Pipe expl", "Soft expl", "Unexplained", "Mismatches",
            ],
            focus_rows, green_fill,
        )
        for split_name in ("train", "test"):
            block = holdout.get(split_name) or {}
            matrix_h = sorted(
                [
                    m for m in (block.get("pairs") or [])
                    if m.get("pair") in focus_keys
                ],
                key=lambda m: str(m.get("pair") or ""),
            )
            row = _section(
                ws4, row,
                f"HOLDOUT focus pairs — {split_name} "
                f"(n_queries≈{holdout.get('n_' + split_name)}; "
                f"QIE vs Regex + Full vs Regex only)",
            ) + 1
            split_pair_rows = [
                [
                    m["pair"],
                    m["prod_exact"],
                    m["n"],
                    m["prod_exact_pct"],
                    m["mismatches"],
                    m["exact_plus_grounded_dropped_pct"],
                    m["exact_plus_gnd_pipeline_pct"],
                    m["exact_plus_gnd_pipeline_soft_pct"],
                ]
                for m in matrix_h
            ]
            row = _table(
                ws4, row,
                [
                    "Pair", "Prod exact", "N", "Prod exact%", "Mismatches",
                    "Exact+gnd%", "Exact+gnd+pipe%", "Exact+gnd+pipe+soft%",
                ],
                split_pair_rows, blue_fill,
            )
        suite_rows = [
            [suite, counts.get("n"), counts.get("train"), counts.get("test")]
            for suite, counts in sorted((holdout.get("by_suite") or {}).items())
        ]
        row = _section(ws4, row, "HOLDOUT split counts by suite") + 1
        row = _table(
            ws4, row,
            ["Suite", "N", "Train", "Test"],
            suite_rows, orange_fill,
        )
    _autowidth(ws4)

    # --- per_query: filters, keywords (3 LLM arms), mismatch vs LLMJ, latency, cost ---
    ws = wb.create_sheet("per_query")
    base_cols = ["query_index", "suite", "section", "query", "status", "keyword_status"]
    identified_cols = list(ARMS)
    keyword_cols = [f"{a}_keywords" for a in ARM_KEYWORD]
    # QIE_Only FIND wire (info only — not part of status / keyword_status).
    find_cols = ["find_query_params", "find_query_string"]
    # Symmetric diffs vs LLMJ reference (match columns first; these = not-match).
    mismatch_cols = [
        c
        for arm in ARMS
        if arm != ARM_LLMJ
        for c in (f"{ARM_LLMJ}_minus_{arm}", f"{arm}_minus_{ARM_LLMJ}")
    ]
    kw_mismatch_cols = [
        c
        for arm in (ARM_QIE, ARM_FULL)
        for c in (f"{ARM_LLMJ}_kw_minus_{arm}", f"{arm}_kw_minus_{ARM_LLMJ}")
    ]
    work_ms_cols = [_arm_ms_key(a) for a in ARMS]
    http_queue_cols = [_arm_http_queue_ms_key(a) for a in ARMS]
    cost_cols = [_arm_cost_key(a) for a in ARMS]
    cols = (
        base_cols
        + identified_cols
        + keyword_cols
        + find_cols
        + mismatch_cols
        + kw_mismatch_cols
        + work_ms_cols
        + http_queue_cols
        + cost_cols
        + ["total_cost_usd"]
    )
    identified_set = set(identified_cols)
    keyword_set = set(keyword_cols) | {"keyword_status"}
    find_set = set(find_cols)
    mismatch_set = set(mismatch_cols) | set(kw_mismatch_cols)
    work_set = set(work_ms_cols)
    hq_set = set(http_queue_cols)
    cost_set = set(cost_cols) | {"total_cost_usd"}

    for c, name in enumerate(cols, 1):
        cell = ws.cell(row=1, column=c, value=name)
        cell.font = header_font
        if name in base_cols and name != "keyword_status":
            cell.fill = green_fill
        elif name in keyword_set or name in find_set:
            cell.fill = green_fill
        elif name in identified_set:
            cell.fill = blue_fill
        elif name in mismatch_set:
            cell.fill = orange_fill
        elif name in work_set or name in hq_set:
            cell.fill = purple_fill
        elif name in cost_set:
            cell.fill = orange_fill
        else:
            cell.fill = blue_fill
        cell.alignment = center
    ws.freeze_panes = "A2"

    for ri, rowdata in enumerate(rows, 2):
        for ci, name in enumerate(cols, 1):
            val = rowdata.get(name, "")
            if name == "find_query_params" and isinstance(val, dict):
                val = _format_find_query_params(val)
            elif isinstance(val, list):
                val = _disp(val)
            ws.cell(row=ri, column=ci, value=val)
    _autowidth(ws)

    ws5 = wb.create_sheet("missing_extra_vs_llmj")
    row = _section(
        ws5, 1,
        f"Uses {ARM_LLMJ} as the reference arm and ranks, per other arm, which params "
        f"it MISSES (LLMJ found the param, the other arm didn't — a recall gap in "
        f"that arm) and which params it adds EXTRA (the other arm found a param LLMJ "
        f"didn't — a precision gap / false positive in that arm). Outcome: the higher "
        f"a param ranks in a 'missing' table, the more consistently that arm fails to "
        f"extract it — fix that arm's prompt/route/regex for that param first. A param "
        f"high in an 'extra' table means that arm over-extracts it — tighten that "
        f"arm's extraction rule for it.",
    ) + 1
    for a in aggregates["miss_vs_llmj_arms"]:
        row = _section(ws5, row, f"{a} — {ARM_LABEL[a]}") + 1
        miss = aggregates["miss_vs_llmj"][a]
        examples = aggregates["examples_miss"][a]
        miss_rows = [
            [i, p, cnt, "; ".join(examples.get(p, [])[:3])]
            for i, (p, cnt) in enumerate(miss.most_common(25), 1)
        ] or [["—", "(none)", 0, "—"]]
        row = _table(
            ws5, row,
            ["Rank", "Param", f"Missing count — {ARM_LLMJ} found it, {a} didn't", "Example queries"],
            miss_rows, blue_fill,
        )
        extra = aggregates["extra_vs_llmj"][a]
        examples_x = (aggregates.get("examples_extra") or {}).get(a) or {}
        extra_rows = []
        for i, (p, cnt) in enumerate(extra.most_common(15), 1):
            ex = [e for e in (examples_x.get(p) or [])[:3] if e]
            extra_rows.append([i, p, cnt, "; ".join(ex) if ex else "(no example query)"])
        if not extra_rows:
            extra_rows = [["—", "(none)", 0, "—"]]
        row = _table(
            ws5, row,
            [
                "Rank", "Param",
                f"Extra count — {a} found it, {ARM_LLMJ} didn't",
                "Example queries",
            ],
            extra_rows, orange_fill,
        )
    _autowidth(ws5)

    # grounding_drops sheet omitted — grounding_drops_*.json + keyword_overlap_*.json written.

    ws8 = wb.create_sheet("description")
    wrap = Alignment(wrap_text=True, vertical="top")

    def _kv(ws, row: int, pairs: List[tuple], fill) -> int:
        for topic, text in pairs:
            c1 = ws.cell(row=row, column=1, value=topic)
            c1.font = Font(bold=True)
            c1.fill = fill
            c1.alignment = Alignment(vertical="top", wrap_text=True)
            c2 = ws.cell(row=row, column=2, value=text)
            c2.alignment = wrap
            row += 1
        return row + 1

    row = _section(ws8, 1, "How to read this workbook") + 1
    row = _kv(
        ws8, row,
        [
            (
                "Workbook sheets (order)",
                "1) pairwise_summary — latency + cost at top, filter pair matrix, "
                "keyword pair matrix (LLMJ/QIE/Full exact+DIFF) + three-arm row, "
                "ranked drop tables, HOLDOUT. "
                "2) per_query — filters + keyword_status + LLM keyword cols. "
                "3) missing_extra_vs_llmj — miss/extra vs LLMJ + example queries. "
                "4) description — this guide. "
                "Omitted xlsx sheets: scorecard, diffs, grounding_drops "
                "(grounding_drops_*.json + keyword_overlap_*.json still written).",
            ),
        ],
        orange_fill,
    )
    row = _kv(ws8, row, [(f"Arm: {a}", ARM_LABEL[a]) for a in ARMS], green_fill)
    row = _kv(
        ws8, row,
        [
            (
                "identified_filters (LLMJ, QIE_Only_LLM, Regex)",
                "Hard filters only, post-live-inventory-grounding. LLMJ/Regex: raw offline "
                "extraction reconciled+grounded via POST /internal/l0_ground. QIE_Only_LLM: "
                "body['identified_filters'] from live /search?qie_only_mode=true, which "
                "short-circuits before the SoftKeywordApplier soft-signal stage even runs.",
            ),
            (
                "Full_Search_LLM's comparison set",
                "pipeline_trace.applied_filters only (post-grounding hard filters). "
                "soft_signals are rank-boost-only in prod and are excluded from the filter "
                "DIFF set so Full is apples-to-apples with LLMJ/QIE_Only_LLM/Regex "
                "identified_filters. Soft lexical terms stay in results JSON keyword "
                "fields (_collect_keywords_full), not a separate xlsx sheet.",
            ),
            (
                "Why this matters for the pairwise matrix",
                "DIFF against Full_Search_LLM means hard-filter disagreement (extraction, "
                "grounding, full-search pipeline not_applied, or soft-signal downgrade) — not "
                "soft-signal shape. Check Grounding-explained / Pipeline-explained / "
                "Soft-downgrade-explained / Unexplained on pairwise_summary, and "
                "missing_extra_vs_llmj for which params diverge.",
            ),
            (
                "status column (per_query sheet)",
                "OK = all 4 arms' hard-filter sets are byte-identical after grounding "
                "(and after general value canonicalization such as auction type label<->id). "
                "DIFF = at least one arm differs on hard filters.",
            ),
            (
                "keyword_status column (per_query sheet)",
                "OK = LLMJ, QIE_Only_LLM, and Full_Search_LLM keyword term sets match "
                "(casefold; probabilities ignored). DIFF = at least one of those three "
                "differs. Regex topical fill is not part of keyword_status. Pairwise "
                "exact/DIFF% live under pairwise_summary['keywords'] and the Keyword "
                "table on the pairwise_summary sheet.",
            ),
            (
                "find_query_params / find_query_string (per_query sheet)",
                "Informational only — from the QIE_Only_LLM /search?qie_only_mode=true "
                "response (FIND wire). find_query_params is one cell listing key=value "
                "pairs (comma-separated) that would go to the FIND API; "
                "find_query_string is the urlencoded query string. Not used for "
                "status / keyword_status / pairwise exact.",
            ),
            (
                "Prod exact / Prod exact% (pairwise_summary sheet)",
                "Share of queries where both arms' *surfaced* filter sets match byte-for-byte "
                "after grounding and (for Full_Search_LLM) pipeline not_applied drops. This is "
                "the prod-quality / user-visible agreement number - what each path would emit. "
                "Pair column alone identifies the arm pair (no Arm A / Arm B columns).",
            ),
            (
                "Exact+grounded-dropped / % (pairwise_summary sheet)",
                "Query-level: count of queries where the two arms' sets are equal after "
                "restoring each side's grounding drops of tokens the peer still has. A "
                "partial token overlap that leaves any residual mismatch does NOT count. "
                "Same unit as Prod exact / N.",
            ),
            (
                "Exact+gnd+pipeline / % (pairwise_summary sheet)",
                "Query-level: equal after restoring grounding ∪ pipeline drops (pipeline "
                "reasons only exist on Full_Search_LLM). Extraction-quality view.",
            ),
            (
                "Exact+gnd+pipeline+soft / % (pairwise_summary sheet)",
                "Query-level: equal after restoring grounding ∪ pipeline ∪ soft-downgrade "
                "drops (soft-downgrade only exists on Full_Search_LLM — see "
                "SoftKeywordApplier.prepare_intent). Fullest extraction-quality view. "
                "Unexplained = N - Exact+gnd+pipeline+soft.",
            ),
            (
                "Grounding-explained (pairwise_summary sheet)",
                "Query count: Exact+grounded-dropped minus Prod exact. Ranked param:value "
                "tables later on this sheet still use token occurrence counts.",
            ),
            (
                "Pipeline-explained (pairwise_summary sheet)",
                "Query count: Exact+gnd+pipeline minus Exact+grounded-dropped (incremental "
                "queries rescued only when pipeline drops are also restored). Full_Search_LLM "
                "pairs only for the pipeline half.",
            ),
            (
                "Soft-downgrade-explained (pairwise_summary sheet)",
                "Query count: Exact+gnd+pipeline+soft minus Exact+gnd+pipeline (incremental "
                "queries rescued only when soft-downgraded params are also restored). "
                "Full_Search_LLM pairs only — the params SoftKeywordApplier moved to "
                "rank-boost-only before not_applied accounting even runs.",
            ),
        ],
        blue_fill,
    )
    row = _section(
        ws8, row,
        "CAVEAT 1 — how latency and cost are shown (xlsx columns)",
    ) + 1
    row = _kv(
        ws8, row,
        [
            (
                "pairwise_summary latency table",
                "Per arm: N, work_p50, work_p95, http_queue_p50, http_queue_p95. "
                "Means omitted from xlsx (p50/p95 only). OK/DIFF rows only.",
            ),
            (
                "{arm}_ms (work) on per_query",
                "extract_ms + http_ms. Queue excluded. "
                f"{ARM_LLMJ}: offline extract + /internal/l0_ground. "
                f"{ARM_REGEX}: local regex + /internal/l0_ground. "
                f"{ARM_QIE}/{ARM_FULL}: extract_ms=0 (extract inside /search); work≈http.",
            ),
            (
                "{arm}_http_queue_ms on per_query",
                "http_ms + http_sem queue_ms aggregated into one column "
                f"(COMPARE_HTTP_INFLIGHT default 1). Diagnostic of wait+HTTP, not pure arm CPU.",
            ),
            (
                "Fair latency comparisons",
                f"(1) {ARM_QIE}_ms vs {ARM_FULL}_ms — both single /search work. "
                f"(2) {ARM_LLMJ} extract vs {ARM_REGEX} extract (in results JSON split timers). "
                f"(3) Do NOT treat {ARM_LLMJ}_ms as the same shape as {ARM_QIE}_ms "
                "(LLMJ = client LLM + ground; QIE = full live qie_only path).",
            ),
            (
                "pairwise_summary / per_query cost",
                "LLM cost_usd: sum_usd + p50_usd per arm on pairwise_summary "
                "(mean_usd omitted from xlsx). Per-query: {arm}_cost_usd + total_cost_usd. "
                f"{ARM_LLMJ}=offline extract usage; {ARM_QIE}/{ARM_FULL}=decision_cost_usd; "
                f"{ARM_REGEX}=0. Old results without cost fields show blank/None until re-run.",
            ),
        ],
        orange_fill,
    )
    row = _section(
        ws8, row,
        f"CAVEAT 2 — why {ARM_QIE}_keywords != {ARM_FULL}_keywords despite the same LLM",
    ) + 1
    row = _kv(
        ws8, row,
        [
            (
                "Same LLM extractor, confirmed",
                f"Both {ARM_QIE} (/search?qie_only_mode=true) and {ARM_FULL} (/search) route "
                "through the identical production L0LLMFilterExtractor instance "
                "(sub.qi_engine._entity_extractor in app.py) — there is no separate model or "
                "prompt per arm.",
            ),
            (
                "Root cause: qie_only has a filter cache that returns keywords=[] on a hit",
                f"{ARM_QIE}'s dedicated cache (_get_qie_l0_filter_cache() in qi/qie_only.py, keyed by "
                "versioned_query_key/exact_query_key) does not store keywords alongside the "
                "cached filters. On a cache HIT it returns the cached filters with keywords "
                "hardcoded to an empty list — per the in-code comment: \"Keywords aren't "
                "cached alongside filters (LLM-usage-derived signal, same as "
                "decision_cost_usd/model_id on cache hits).\" The full-search path does not "
                "go through this qie_only-specific cache shortcut, so it more often carries "
                "through the real extracted keywords. This is a deterministic caching "
                "artifact, not LLM sampling randomness.",
            ),
            (
                "Keyword overlap not an xlsx sheet",
                "Keyword pairwise exact/Jaccard + miss/extra vs LLMJ stay in results JSON "
                "only (no keyword_overlap sheet).",
            ),
            (
                "What would actually fix it",
                f"Cache keywords alongside filters in _get_qie_l0_filter_cache(), or bypass "
                "the qie_only filter-cache path entirely when the caller needs keywords "
                "(e.g. this harness's LLMJ arm) — not a prompt/model change.",
            ),
        ],
        orange_fill,
    )
    row = _section(ws8, row, "not_applied handling for Full_Search_LLM's comparison set") + 1
    row = _kv(
        ws8, row,
        [
            (
                "Why filters.not_applied isn't folded into Full_Search_LLM's compared set",
                f"{ARM_QIE}/{ARM_LLMJ}/{ARM_REGEX} also drop grounding-ungrounded values "
                "before returning identified_filters (app.py: reconcile_and_ground_identified "
                "-> ground_identified_filters drops the same way pipeline_trace.applied_filters "
                "does) — so unioning not_applied into Full_Search_LLM's set would make it "
                "structurally WIDER than what any other arm could ever produce, manufacturing "
                "new mismatches instead of removing spurious ones.",
            ),
            (
                "What's done instead",
                "Compared sets are left unchanged; not_applied reasons are surfaced as the "
                "Pipeline-explained mismatch bucket on pairwise_summary, scoped to the 3 "
                "not_applied reasons (filter_relaxed_by_guard / backend_unsupported / "
                "column_data_unavailable) that only exist on the full-search path. A DIFF "
                "against Full_Search_LLM fully covered by Grounding-explained + "
                "Pipeline-explained + Soft-downgrade-explained is a pipeline/grounding/"
                "soft-signal artifact, not a real extraction disagreement — check the "
                "Unexplained column on pairwise_summary for the residual that is.",
            ),
            (
                "Soft-downgrade — a distinct bucket, not a not_applied reason",
                "SoftKeywordApplier.prepare_intent (retrieval/soft_keyword_apply.py) migrates "
                "entities whose slot name is in entity_slots.soft_slot_names (topic_include/"
                "exclude, similar_to, lifecycle_state, lifecycle_disjunction, keyword_* slots) "
                "off the hard entities list onto soft rank-boost signals BEFORE not_applied "
                "accounting runs — no not_applied entry is ever written for it, so it is "
                "structurally invisible to the Pipeline-explained bucket above. Read straight "
                "off query_intelligence.filters.soft_signals instead (see "
                "_collect_full_soft_downgraded) and surfaced as its own "
                "Soft-downgrade-explained bucket.",
            ),
        ],
        orange_fill,
    )
    row = _section(ws8, row, "Per-sheet guide — what each sheet shows, how it's computed, what to do") + 1
    row = _kv(
        ws8, row,
        [
            (
                "pairwise_summary (first sheet)",
                "Top: latency (work_p50/p95, http_queue_p50/p95) then LLM cost "
                "(sum_usd, p50_usd) per arm. Next: C(4,2)=6-pair quality matrix "
                "(Pair column only — no Arm A/B; no 'Aggregate across all 6 pairs' block). "
                "Then optional ranked param:value drop tables (grounding / pipeline / soft). "
                "HOLDOUT focus: per pair, rows ordered train then test then all "
                "(QIE vs Regex, Full vs Regex; LLMJ excluded). Full vs Regex headline = "
                "Exact+gnd+pipe+soft%; QIE vs Regex headline = Prod exact. "
                "JSON: pairwise_summary_*.json (+ holdout); grounding_drops_*.json "
                "(no grounding_drops xlsx sheet).",
            ),
            (
                "per_query",
                "One row per query. Column groups left-to-right: (1) identified filters "
                "for LLMJ/QIE/Full/Regex side-by-side, (2) not-match vs LLMJ "
                "(LLMJ_minus_* / *_minus_LLMJ), (3) work_ms per arm, (4) http_queue_ms "
                "per arm, (5) cost_usd per arm + total. status=OK = all 4 sets identical "
                "after grounding; DIFF = inspect minus columns / missing_extra_vs_llmj. "
                "Source for --analysis-only: results_*.json.",
            ),
            (
                "missing_extra_vs_llmj",
                f"Uses {ARM_LLMJ} as the reference. Per other arm: ranked 'Missing count' "
                f"and 'Extra count' param tables, each with Example queries (deduped). "
                "Fix highest-ranked missing (recall gap) / extra (precision gap) first. "
                "JSON: missing_extra_vs_llmj_*.json.",
            ),
            (
                "description",
                "This guide. Rebuild via --analysis-only after harness report changes so "
                "the xlsx description stays in sync with sheet layout.",
            ),
        ],
        blue_fill,
    )
    ws8.column_dimensions["A"].width = 44
    ws8.column_dimensions["B"].width = 110

    wb.save(out_path)
    print(f"XLSX: {out_path}")
    return out_path


def _norm_val(v: Any) -> str:
    if isinstance(v, float) and v == int(v):
        v = int(v)
    if isinstance(v, list):
        return "|".join(sorted({_norm_val(x) for x in v if x not in ("", None)}))
    s = _normalize_value(v)
    if re.fullmatch(r"\d+\.0+", s or ""):
        s = s.split(".", 1)[0]
    if s and "|" in s:
        s = "|".join(sorted({t for t in s.split("|") if t}))
    return s


_ISO_Z_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})t(\d{2}:\d{2}:\d{2})z$",
    re.IGNORECASE,
)


def _iso_to_relative(v: str) -> Optional[str]:
    """Map absolute Zulu timestamp -> nearest L0 relative offset (-1d/-3d/-7d/-14d)."""
    m = _ISO_Z_RE.fullmatch((v or "").strip())
    if not m:
        return None
    try:
        dt = datetime.fromisoformat(f"{m.group(1)}T{m.group(2)}+00:00")
    except ValueError:
        return None
    days = abs((dt - datetime.now(timezone.utc)).total_seconds()) / 86400.0
    if days < 1.5:
        return "-1d"
    if days < 4.5:
        return "-3d"
    if days < 10.0:
        return "-7d"
    if days < 18.0:
        return "-14d"
    return None


def _norm_time(param: str, v: str) -> str:
    if not v:
        return v
    low = v.lower().strip()
    if param in (
        "endTimeBefore",
        "endTimeAfter",
        "startTimeBefore",
        "startTimeAfter",
    ):
        if low in ("today", "tonight", "-24h"):
            return "-1d"
        if low.startswith("-") and re.fullmatch(r"-\d+[dh]", low):
            return low
        rel = _iso_to_relative(low)
        if rel:
            return rel
    return v


def _expand_auction_type_values(values: List[str]) -> List[str]:
    """Label/id-stable auction type tokens (same rules as qi.grounding expand).

    General: any label in ``AUCTION_TYPE_LABEL_TO_IDS`` expands to its numeric
    ids unless the value list already carries an explicit numeric id — then
    labels are dropped so sibling ids cannot leak in. Registrar preference via
    ``prefer_registrar_auction_values``. No query-specific branches.
    """
    # Soft import: analysis-only / seed paths should not require a live service,
    # but contracts is light and always available with the package.
    from semantic_search.contracts import (  # noqa: PLC0415
        AUCTION_TYPE_LABEL_TO_IDS,
        prefer_registrar_auction_values,
    )

    values = prefer_registrar_auction_values([str(v) for v in values if v not in ("", None)])
    result: List[str] = []
    seen: set = set()
    has_explicit_numeric = any(
        str(v).lower() not in AUCTION_TYPE_LABEL_TO_IDS for v in values
    )
    for v in values:
        sv = str(v).lower()
        if sv in AUCTION_TYPE_LABEL_TO_IDS:
            if not has_explicit_numeric:
                for nid in sorted(AUCTION_TYPE_LABEL_TO_IDS[sv]):
                    if nid not in seen:
                        seen.add(nid)
                        result.append(nid)
        else:
            if sv not in seen:
                seen.add(sv)
                result.append(sv)
    return result


def _norm_type_list_value(val: str) -> str:
    """Canonical ``|``-joined auction type list for cross-arm set equality."""
    parts = [p for p in str(val or "").split("|") if p]
    if not parts:
        return ""
    expanded = _expand_auction_type_values(parts)
    return "|".join(sorted(expanded))


def _entries_to_set(entries: List[Dict[str, Any]]) -> List[str]:
    """Canonical param->value set; merge multi lifecycle; prefer relative times."""
    by_param: Dict[str, str] = {}
    for e in entries or []:
        param = _param_name_from_filter(e) if isinstance(e, dict) else ""
        if not param:
            api = e.get("api_param") if isinstance(e, dict) else None
            if isinstance(api, dict) and api.get("name"):
                param = str(api["name"]).strip()
            elif isinstance(api, str) and api.strip():
                param = api.strip()
            else:
                param = str(e.get("param") or e.get("name") or "").strip()
            param = _canonicalize_param(_SLOT_TO_API.get(param, param))
        if not param:
            continue
        val = _norm_val(e.get("value"))
        val = _norm_time(param, val)
        if param in ("typeIncludeList", "typeExcludeList"):
            val = _norm_type_list_value(val)
        prev = by_param.get(param)
        if prev is None:
            by_param[param] = val
            continue
        if prev.startswith("-") and _ISO_Z_RE.fullmatch(val or ""):
            continue
        if _ISO_Z_RE.fullmatch(prev or "") and val.startswith("-"):
            by_param[param] = val
            continue
        if param in (
            "lifecycle_state",
            "tldIncludeList",
            "typeIncludeList",
            "typeExcludeList",
            "topic_include",
            "similar_to",
        ):
            toks = {t for t in (prev.split("|") + val.split("|")) if t}
            merged_val = "|".join(sorted(toks))
            if param in ("typeIncludeList", "typeExcludeList"):
                merged_val = _norm_type_list_value(merged_val)
            by_param[param] = merged_val
            continue
        by_param[param] = val
    ls = by_param.get("lifecycle_state", "")
    if {"active", "pending_delete"} <= set(ls.split("|")):
        by_param.setdefault("lifecycle_disjunction", "true")
    out: Set[str] = set()
    for param, val in by_param.items():
        out.add(f"{param}:{val}" if val not in ("", None) else param)
    return sorted(out)


def _disp(tokens: List[str]) -> str:
    return ", ".join(tokens) if tokens else "none"


def _keywords_to_set(keywords: List[Dict[str, Any]]) -> List[str]:
    """Canonical keyword-term set: lowercase term only, deduped, sorted.

    Probabilities are excluded from the comparison key — they vary run-to-run
    (LLM non-determinism) and would cause spurious cross-arm mismatches that
    have nothing to do with which terms were actually extracted.
    """
    terms = {
        str(k.get("term") or "").strip().lower()
        for k in (keywords or [])
        if isinstance(k, dict)
    }
    terms.discard("")
    return sorted(terms)


def _keywords_detail(keywords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Stable ``{term, probability?}`` list for results JSON / audit (not set equality)."""
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for kw in keywords or []:
        if not isinstance(kw, dict):
            continue
        term = str(kw.get("term") or "").strip()
        if not term:
            continue
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        entry: Dict[str, Any] = {"term": term}
        raw_prob = kw.get("probability")
        if raw_prob is not None:
            try:
                entry["probability"] = float(raw_prob)
            except (TypeError, ValueError):
                pass
        out.append(entry)
    out.sort(
        key=lambda e: (-float(e["probability"]) if "probability" in e else 0.0, e["term"].lower()),
    )
    return out


def _keyword_three_arm_status(kw_sets: Dict[str, List[str]]) -> str:
    """OK when LLMJ / QIE_Only_LLM / Full_Search_LLM term sets match; else DIFF."""
    three = [frozenset(kw_sets.get(a) or []) for a in ARM_KEYWORD]
    return "OK" if len(set(three)) == 1 else "DIFF"


def _collect_keywords_llmj(
    raw_identified: List[Dict[str, Any]],
    keywords: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """LLMJ keywords — same union shape as QIE/Full collectors (body keywords ∪ keyword_* chips)."""
    return _merge_keyword_dicts(
        list(keywords or []),
        _keyword_terms_from_filter_entries(list(raw_identified or [])),
    )


def _retry_sleep_seconds(resp: Optional[requests.Response], attempt: int) -> float:
    """Short bounded backoff (≤3s). Prefer load-shedding over waiting out 503 storms."""
    if resp is not None:
        ra = (resp.headers.get("Retry-After") or "").strip()
        if ra.isdigit():
            return min(3.0, float(ra))
        try:
            body = resp.json()
            if isinstance(body, dict) and body.get("retry_after_seconds") is not None:
                return min(3.0, float(body["retry_after_seconds"]))
        except (ValueError, TypeError, requests.JSONDecodeError):
            pass
    # 0.4s, 0.8s, 1.6s, … capped at 3s — never multi-minute waits.
    return min(3.0, 0.4 * (2**attempt))


def _retry_post(
    post_once: Callable[[str], requests.Response], *, label: str
) -> Tuple[dict, float]:
    """Shared retry loop for any POST to the :8085 service — retry 429/502/503/504 +
    transport errors. Used by both /search and /internal/l0_ground callers below so
    grounding gets the exact same retry policy as search (no separate policy invented).
    """
    t0 = time.perf_counter()
    session_id = f"l0-compare-{uuid.uuid4().hex}"
    last_exc: Optional[BaseException] = None
    resp: Optional[requests.Response] = None
    for attempt in range(_HTTP_MAX_ATTEMPTS):
        try:
            resp = post_once(session_id)
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
            time.sleep(_retry_sleep_seconds(None, attempt))
            continue
        if resp.status_code in _RETRYABLE_HTTP:
            last_exc = requests.HTTPError(
                f"{resp.status_code} Server Error for url: {resp.url}",
                response=resp,
            )
            time.sleep(_retry_sleep_seconds(resp, attempt))
            continue
        break
    else:
        if resp is not None:
            resp.raise_for_status()
        raise RuntimeError(
            f"{label}_failed after {_HTTP_MAX_ATTEMPTS} attempts: {last_exc}"
        )

    assert resp is not None
    resp.raise_for_status()
    return resp.json(), (time.perf_counter() - t0) * 1000.0


def _post_search_once(
    query: str,
    *,
    qie_only: bool,
    timeout: float,
    session_id: str,
) -> requests.Response:
    data = {"query": query, "top_k": "5"}
    if qie_only:
        data["qie_only_mode"] = "true"
    headers = {"X-Session-Id": session_id}
    return requests.post(ENDPOINT, data=data, headers=headers, timeout=timeout)


def _post_search(
    query: str, *, qie_only: bool, timeout: float = 120.0
) -> Tuple[dict, float]:
    """POST /search with retries on 429/502/503/504 and transport errors.

    Permanent: compare harness used to retry only 429; production middleware also
    returns 503 when Starlette cancels in-flight work under load
    (``request_cancelled: server busy under load, retry``). Without 503 retry,
    concurrent full-suite runs mark ~10% of rows ERROR with empty arm sets.
    Unique ``X-Session-Id`` avoids anonymous IP rate-limit bucket collisions.
    """
    return _retry_post(
        lambda session_id: _post_search_once(
            query, qie_only=qie_only, timeout=timeout, session_id=session_id
        ),
        label="search",
    )


def _post_ground(
    query: str, identified: List[Dict[str, Any]], *, timeout: float = 60.0
) -> Tuple[dict, float]:
    """POST /internal/l0_ground — reconcile + live-inventory-ground a raw filter list
    via the same pipeline production's qie_only route uses. Same retry policy as
    _post_search (429/502/503/504 + transport errors).
    """

    def post_once(session_id: str) -> requests.Response:
        headers = {"X-Session-Id": session_id}
        harness_key = os.environ.get("HARNESS_API_KEY", "").strip()
        if harness_key:
            headers["X-Harness-Key"] = harness_key
        return requests.post(
            L0_GROUND_ENDPOINT,
            json={"query": query, "identified": identified},
            headers=headers,
            timeout=timeout,
        )

    return _retry_post(post_once, label="l0_ground")


def _is_keyword_filter_param(param: str) -> bool:
    """True when param is a keyword_* slot (compared via keyword sets, not filter sets)."""
    p = _canonicalize_param(str(param or "").strip())
    return p in _KEYWORD_VALUE_PARAMS or p == "keyword_match_mode"


def _filter_qie_chip_entries(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop fallback invents + keyword_* chips from a qie-shaped entry list."""
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for e in raw:
        if not e:
            continue
        if str(e.get("source") or "").strip().lower() == "fallback":
            continue
        param = e.get("name") or e.get("param")
        if param and _is_keyword_filter_param(str(param)):
            continue
        key = (str(param), repr(e.get("value")))
        if key in seen:
            continue
        seen.add(key)
        out.append({"param": param, "value": e.get("value")})
    return out


def _collect_qie(body: dict) -> List[Dict[str, Any]]:
    """qie_only chips for filter-set match (keyword_* + vague fallback invents excluded).

    ``source=fallback`` chips come from VagueQuantifierResolver (rank soft invents).
    LLMJ / Full applied_filters do not emit them as extract agreement — counting them
    as QIE filter-set tokens creates false-positive DIFF vs LLMJ.
    """
    raw = list(body.get("identified_filters") or []) + list(body.get("soft_chips") or [])
    return _filter_qie_chip_entries(raw)


def _format_find_query_params(params: Any) -> str:
    """Single-column display for FIND wire ``find_query_params`` dict (info only)."""
    if not isinstance(params, dict) or not params:
        return ""
    return ", ".join(f"{k}={v}" for k, v in params.items())


def _collect_qie_find_wire(body: dict) -> Tuple[Dict[str, str], str]:
    """Pull qie_only FIND wire fields from /search body (not used in arm agreement)."""
    raw = body.get("find_query_params")
    params: Dict[str, str] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if k is None:
                continue
            params[str(k)] = "" if v is None else str(v)
    qs = body.get("find_query_string")
    return params, ("" if qs is None else str(qs))


def _collect_qie_pre_ground(body: dict) -> Optional[List[str]]:
    """Pre-ground token set from qie_only ``pre_ground_identified`` (None if absent)."""
    if "pre_ground_identified" not in body:
        return None
    entries = _filter_qie_chip_entries(list(body.get("pre_ground_identified") or []))
    # soft_chips are post-wire; pre-ground list is the inventory-ground input only.
    return _entries_to_set(entries)


def _keyword_terms_from_filter_entries(
    entries: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Pull lexical terms from keyword_* filter chips (applied / identified / soft)."""
    out: List[Dict[str, Any]] = []
    for e in entries or []:
        if not isinstance(e, dict):
            continue
        param = _param_name_from_filter(e) if e.get("api_param") else ""
        if not param:
            raw_name = str(e.get("name") or e.get("param") or "").strip()
            param = _canonicalize_param(_SLOT_TO_API.get(raw_name, raw_name))
        if param not in _KEYWORD_VALUE_PARAMS:
            continue
        raw = e.get("value")
        values = raw if isinstance(raw, list) else [raw]
        for v in values:
            if v is None:
                continue
            # Pipe/CSV multi-terms from LLM keyword_contains lists.
            parts = re.split(r"[|,]", str(v)) if not isinstance(v, list) else [str(v)]
            for part in parts:
                term = str(part).strip()
                if term:
                    out.append({"term": term})
    return out


def _merge_keyword_dicts(*groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Union keyword dicts by casefolded term; keep first-seen probability when present."""
    by_term: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for group in groups:
        for kw in group or []:
            if not isinstance(kw, dict):
                continue
            term = str(kw.get("term") or "").strip()
            if not term:
                continue
            key = term.casefold()
            if key not in by_term:
                by_term[key] = {"term": term}
                order.append(key)
            if kw.get("probability") is not None and "probability" not in by_term[key]:
                try:
                    by_term[key]["probability"] = float(kw["probability"])
                except (TypeError, ValueError):
                    pass
    return [by_term[k] for k in order]


def _collect_keywords_qie(body: dict) -> List[Dict[str, Any]]:
    """L0 keywords for qie_only: ``keywords`` ∪ keyword_* on identified/soft chips.

    Same L0 sources as ``_collect_keywords_full`` (no retrieval-side applied_keywords).
    """
    return _merge_keyword_dicts(
        list(body.get("keywords") or []),
        _keyword_terms_from_filter_entries(
            list(body.get("identified_filters") or [])
        ),
        _keyword_terms_from_filter_entries(list(body.get("soft_chips") or [])),
    )


def _collect_keywords_full(body: dict) -> List[Dict[str, Any]]:
    """L0 keywords for full-search — parity with qie_only collectors.

    Sources (union, deduped by term) — L0 only, not retrieval applied_keywords:
    - ``query_intelligence.filters.keywords`` (intent.keywords)
    - keyword_* on ``filters.identified`` / ``filters.soft_signals`` / soft_chips
    """
    qi = body.get("query_intelligence") or {}
    filt = qi.get("filters") or {}
    return _merge_keyword_dicts(
        list(filt.get("keywords") or []),
        _keyword_terms_from_filter_entries(list(filt.get("identified") or [])),
        _keyword_terms_from_filter_entries(list(filt.get("soft_signals") or [])),
        _keyword_terms_from_filter_entries(list(body.get("soft_chips") or [])),
    )


def _body_decision_cost_usd(body: dict) -> float:
    """LLM spend from live /search body (top-level or query_intelligence)."""
    qi = body.get("query_intelligence") or {}
    for src in (body.get("decision_cost_usd"), qi.get("decision_cost_usd")):
        if src is None:
            continue
        try:
            return float(src)
        except (TypeError, ValueError):
            continue
    return 0.0


def _collect_full(body: dict) -> List[Dict[str, Any]]:
    """Full-search hard filters only (``pipeline_trace.applied_filters``).

    Soft signals / soft chips are **not** unioned into the filter comparison set.
    They are rank-boost-only in prod (never hard filters) and structurally wider
    than LLMJ/QIE_Only_LLM/Regex ``identified_filters`` — folding them in created
    false-positive DIFFs that are shape artifacts, not extraction disagreements.
    Keywords from soft signals still flow through ``_collect_keywords_full``.

    ``query_intelligence.filters.identified`` is the pre-inventory-ground snapshot
    and is intentionally unused here (compare grounded-vs-grounded).
    """
    trace = body.get("pipeline_trace") or {}
    identified = list(trace.get("applied_filters") or [])
    merged: List[Dict[str, Any]] = []
    seen: set = set()
    for e in identified:
        if not e:
            continue
        param = _param_name_from_filter(e) if isinstance(e, dict) else ""
        if not param:
            param = e.get("name") or e.get("param")
            if param:
                param = _canonicalize_param(_SLOT_TO_API.get(str(param), str(param)))
        if not param or _is_keyword_filter_param(str(param)):
            # keyword_* terms matched via _collect_keywords_full, not filter sets
            continue
        key = (str(param), repr(e.get("value")))
        if key in seen:
            continue
        seen.add(key)
        merged.append(
            {"param": param, "value": e.get("value"), "api_param": e.get("api_param")}
        )
    if merged:
        return merged
    return [
        {"param": e["param"], "value": e.get("value")}
        for e in _collect_applied_filter_entries(body)
    ]


def _collect_full_grounding_dropped(body: dict) -> List[Dict[str, Any]]:
    """Full-search entries dropped by live-inventory grounding — read straight off
    ``query_intelligence.filters.not_applied`` (name/value/reason), no diffing needed.

    ``not_applied`` entries carry ``reason="inventory_ungrounded"`` exactly when
    ``app._build_filter_summary`` finds a hard, available, filterable entity that
    isn't in ``grounded_names`` (``pipeline_trace.applied_filters``) — i.e. the same
    condition LLMJ/Regex's own ``/internal/l0_ground`` raw-vs-grounded diff attributes
    to grounding enforcement. Other ``not_applied`` reasons (guard relaxation, backend
    unsupported, missing column data, soft signal) are not grounding drops and are
    excluded here.
    """
    qi = body.get("query_intelligence") or {}
    filt = qi.get("filters") or {}
    not_applied = list(filt.get("not_applied") or [])
    return [
        _not_applied_as_entry(e)
        for e in not_applied
        if e and e.get("reason") == "inventory_ungrounded"
    ]


# not_applied reasons that only exist on the full-search path - app.py:2988
# ("qie_only_mode uses a separate slim L0 path and does not call this
# [_build_filter_summary]") confirms QIE_Only_LLM/LLMJ/Regex never hit guard
# relaxation, backend-capability, or enrichment-column checks. "soft_signal_not_filter"
# is excluded: soft_signals are not part of the hard-filter comparison set.
# "inventory_ungrounded" is excluded: that's the grounding drop already handled by
# _collect_full_grounding_dropped.
_PIPELINE_ONLY_REASONS = frozenset(
    {"filter_relaxed_by_guard", "backend_unsupported", "column_data_unavailable"}
)


def _collect_full_pipeline_dropped(body: dict) -> List[Dict[str, Any]]:
    """Full-search entries missing from ``applied_filters`` for a full-search-pipeline
    reason (guard relaxation / backend capability / enrichment-column availability) —
    not an extraction or grounding disagreement, since LLMJ/QIE_Only_LLM/Regex never
    run this check at all. Read straight off ``query_intelligence.filters.not_applied``,
    same shape as ``_collect_full_grounding_dropped``, disjoint reason set.
    """
    qi = body.get("query_intelligence") or {}
    filt = qi.get("filters") or {}
    not_applied = list(filt.get("not_applied") or [])
    return [
        _not_applied_as_entry(e)
        for e in not_applied
        if e and e.get("reason") in _PIPELINE_ONLY_REASONS
    ]


def _collect_full_soft_downgraded(body: dict) -> List[Dict[str, Any]]:
    """Full-search entries migrated from hard filters to rank-boost-only soft
    signals by ``SoftKeywordApplier.prepare_intent`` (retrieval/soft_keyword_apply.py)
    before ``not_applied`` accounting even runs — any entity whose slot name is in
    ``entity_slots.soft_slot_names`` (topic_include/exclude, similar_to,
    lifecycle_state, lifecycle_disjunction, keyword_* slots) is stripped pre-retrieve
    on the full-search path only. LLMJ/QIE_Only_LLM/Regex never run this stage, so a
    mismatch fully explained by an entry here is a pipeline effect (soft downgrade),
    not an extraction gap — distinct from ``_PIPELINE_ONLY_REASONS``, which reads
    ``not_applied``; this reads ``query_intelligence.filters.<soft_response_key>``
    (default ``"soft_signals"``) directly, since the downgrade produces no
    ``not_applied`` entry at all.
    """
    qi = body.get("query_intelligence") or {}
    filt = qi.get("filters") or {}
    soft_response_key = str(
        getattr(_QI_CONFIG.entity_slots, "soft_response_key", "soft_signals")
        or "soft_signals"
    )
    soft_signals = list(filt.get(soft_response_key) or [])
    return [_not_applied_as_entry(e) for e in soft_signals if e]


def _build_regex_extractor() -> Any:
    """One ``L0RegexFilterExtractor`` over one ``RegexEntityExtractor`` — the same
    fallback parser qie_only/full-search use on LLM outage.
    """
    # Deferred: heavy semantic_search deps not needed for --seed/--analysis-only.
    from semantic_search.config.loader import load_config  # noqa: PLC0415
    from semantic_search.config.models import (  # noqa: PLC0415
        AgentSearchConfig,
        QIL0RegexEntityConfig,
    )
    from semantic_search.qi.l0_regex_filter_extractor import (  # noqa: PLC0415
        L0RegexFilterExtractor,
    )
    from semantic_search.qi.regex_entity_extractor import RegexEntityExtractor  # noqa: PLC0415

    conf = AgentSearchConfig.from_dict(load_config())
    slots = conf.qi.entity_slots
    regex_cfg = conf.qi.l0_regex_entity
    rex = RegexEntityExtractor(
        QIL0RegexEntityConfig(
            enabled=True,
            max_entities=regex_cfg.max_entities,
            confidence=regex_cfg.confidence,
            source_tag=regex_cfg.source_tag,
            fallback_only_when_llm_unavailable=regex_cfg.fallback_only_when_llm_unavailable,
        ),
        hard_entity_names=slots.hard_entity_set,
        soft_slot_names=slots.soft_slot_set,
        known_tlds=frozenset(conf.qi.regex.known_tlds),
    )
    return L0RegexFilterExtractor(rex)


def _build_grounding_complete() -> Callable[[str], str]:
    """OpenAI-compat complete_fn locked to resolved harness grounding model.

    Model: env ``L0_GROUNDING_MODEL`` or primary from
    ``task_model_allowlists.l0_entity_extraction`` ∩ discovery. Gemini prefers
    ``GOOGLE_API_KEY``, else ``OPENAI_API_KEY`` (``llm_api_keys`` in base.yaml).
    Reasoning-family SKUs reject ``temperature``; some want ``max_completion_tokens``
    instead of ``max_tokens`` — both probed and retried below.
    """
    # Deferred: heavy openai dep not needed for --seed/--analysis-only.
    from openai import OpenAI  # noqa: PLC0415

    model = _grounding_model()
    if not model:
        raise SystemExit(
            "ERROR: grounding model resolved empty "
            "(set L0_GROUNDING_MODEL or ensure task_model_allowlists ∩ discovery)"
        )
    api_key, base_url = resolve_openai_compat_api_key_for_model(model)
    kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    # Reasoning-family models reject an explicit temperature — match llm_core retry policy.
    omit_temperature = model.startswith(_NO_TEMPERATURE_MODEL_PREFIXES)

    def complete(prompt: str) -> str:
        messages = [
            {"role": "system", "content": L0_FILTER_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        call_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": 4096,
        }
        if not omit_temperature:
            call_kwargs["temperature"] = 0
        try:
            resp = client.chat.completions.create(**call_kwargs)
        except Exception as first_exc:  # noqa: BLE001 — probe alternate token kw
            msg = str(first_exc).lower()
            if "max_tokens" in msg and "max_completion_tokens" in msg:
                call_kwargs.pop("max_tokens", None)
                call_kwargs["max_completion_tokens"] = 4096
                resp = client.chat.completions.create(**call_kwargs)
            elif "temperature" in msg:
                call_kwargs.pop("temperature", None)
                resp = client.chat.completions.create(**call_kwargs)
            else:
                raise
        usage = getattr(resp, "usage", None)
        complete.last_model = model
        complete.last_usage = {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        } if usage is not None else {}
        content = resp.choices[0].message.content
        return (content or "").strip()

    complete.last_model = model
    complete.last_usage = {}
    return complete


def _needs_rewrite_like_prod(normalized_query: str) -> bool:
    """Mirror ``QueryTransformer.needs_rewrite`` (query_transformer.py) against the
    harness's already-loaded prod config (``_QI_CONFIG.query_transformer`` — same
    ``base.yaml`` source ``orchestrator.py`` reads). Same formula, no live
    ``QueryTransformer`` instantiation — that class's ``__init__`` loads a local
    seq2seq torch model as a side effect, which this offline-extract-only caller
    doesn't need."""
    qt_cfg = _QI_CONFIG.query_transformer
    if not qt_cfg.rewrite_enabled:
        return False
    if not normalized_query:
        return False
    return len(normalized_query.split()) > qt_cfg.rewrite_threshold


def _l0_llm_extract_raw(
    complete: Callable[[str], str], query: str
) -> Tuple[Any, List[Dict[str, Any]], float]:
    """One offline LLM L0 extract call for a single query — raw, pre-reconcile,
    pre-ground ``identified_filters``-shaped list (same prompt/parse production's
    L0 uses, up through ``filters_to_identified``). Reconcile + grounding happen
    afterward via the live ``/internal/l0_ground`` route (``_post_ground``), not
    here — this arm never grounds locally, so the reconcile/static-grounder branch
    ``extract_test_filters_queries.py::l0_llm_ground_batch`` has for its own
    ``ground=True`` default is dead weight for this caller and is not ported here.

    Token-gated like prod: when ``combine_rewrite_with_l0_extract`` is on and the
    query is over ``rewrite_threshold`` tokens (``_needs_rewrite_like_prod``), uses
    the combined rewrite+extract prompt (``build_l0_combined_user_prompt``) and
    grounds/extracts against the model's ``rewritten_query`` — matching
    ``L0LLMFilterExtractor.extract_priced_combined``'s ``query=rewritten`` call
    exactly (l0_llm_filter_extractor.py:2862). Otherwise falls back to the plain
    ``build_l0_filter_user_prompt`` path. Same ``L0_FILTER_SYSTEM`` system prompt
    either way (baked into ``complete()`` by ``_build_grounding_complete``).

    Returns ``(identified_list, keywords, cost_usd)`` on success, or
    ``("parse_error"|"json_error"|"error:<ExcName>", [], 0.0)`` on failure.
    """
    expanded_q = expand_gd_to_godaddy(query)
    qt_cfg = _QI_CONFIG.query_transformer
    use_combined = bool(qt_cfg.combine_rewrite_with_l0_extract) and _needs_rewrite_like_prod(
        expanded_q
    )
    prompt = (
        build_l0_combined_user_prompt([(1, expanded_q)])
        if use_combined
        else build_l0_filter_user_prompt([(1, expanded_q)])
    )
    try:
        raw = complete(prompt)
        cost_usd = float(
            compute_call_cost_usd(
                str(getattr(complete, "last_model", "") or ""),
                getattr(complete, "last_usage", None),
            )
        )
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            print(f"  L0 LLM parse error. Raw: {raw[:120]}")
            return "parse_error", [], cost_usd
        parsed = json.loads(json_match.group())
        entry = next((e for e in parsed.get("results", []) if e.get("idx") == 1), None)
        if entry is None:
            return [], [], cost_usd
        raw_filters = entry.get("filters", []) or []
        if use_combined:
            candidate = str(entry.get("rewritten_query") or "").strip()
            grounding_query = expand_gd_to_godaddy(candidate) if candidate else expanded_q
        else:
            grounding_query = expanded_q
        identified = filters_to_identified(
            [
                {"param": str(f.get("param") or "").strip(), "value": f.get("value")}
                for f in raw_filters
                if isinstance(f, dict)
            ],
            query=grounding_query,
            source=_QI_CONFIG.l0_llm_entity.source_tag,
            soft_slot_names=_QI_CONFIG.entity_slots.soft_slot_set,
            confidence=_QI_CONFIG.l0_llm_entity.confidence,
        )
        keywords: List[Dict[str, Any]] = []
        for kw in entry.get("keywords", []) or []:
            if not isinstance(kw, dict):
                continue
            term = str(kw.get("term") or "").strip()
            if not term:
                continue
            try:
                probability = float(kw.get("probability"))
            except (TypeError, ValueError):
                probability = 0.0
            keywords.append({"term": term, "probability": probability})
        kw_min_prob = _QI_CONFIG.l0_llm_entity.keyword_min_probability / 100.0
        keywords = [k for k in keywords if k["probability"] >= kw_min_prob]
        # Trust LLM abstain (keywords=[]) — do not residual-fill; matches
        # L0LLMFilterExtractor._fill_keywords_if_empty pass-through.
        keywords.sort(key=lambda k: k["probability"], reverse=True)
        return identified, keywords, cost_usd
    except json.JSONDecodeError as e:
        print(f"  L0 LLM JSON error: {e}")
        return "json_error", [], 0.0
    except Exception as e:  # noqa: BLE001
        print(f"  L0 LLM error: {type(e).__name__}: {e}")
        return f"error:{type(e).__name__}", [], 0.0


def _llmj_raw_identified(
    complete: Callable[[str], str], query: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
    """Raw (pre-reconcile, pre-ground) offline LLM L0 extract — one query per call
    (L0_LLM_BATCH_SIZE=1 parity). Retries transient ``parse_error``/``json_error``
    from the offline extractor (flaky model JSON) so one bad parse does not ERROR
    the whole row. Reconcile + grounding happen afterward via ``/internal/l0_ground``
    (``_post_ground``) — not here.

    Returns ``(raw_identified, keywords, cost_usd)``.
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(_GROUND_MAX_ATTEMPTS):
        raw, keywords, cost_usd = _l0_llm_extract_raw(complete, query)
        if isinstance(raw, list):
            return raw, keywords, cost_usd
        last_exc = RuntimeError(f"grounding_failed: {raw}")
        if raw not in ("parse_error", "json_error"):
            raise last_exc
        time.sleep(min(8.0, 0.75 * (2**attempt)))
    raise RuntimeError(
        f"grounding_failed after {_GROUND_MAX_ATTEMPTS} attempts: {last_exc}"
    )


def _dropped_by_grounding_key(arm: str) -> str:
    return f"{arm}_dropped_by_grounding"


def _dropped_by_pipeline_key(arm: str) -> str:
    return f"{arm}_dropped_by_pipeline"


def _dropped_by_soft_key(arm: str) -> str:
    return f"{arm}_dropped_by_soft"


def _pre_ground_set_key(arm: str) -> str:
    return f"{arm}_pre_ground_set"


# Arms where we can attribute a drop to grounding enforcement (e.g. tld not in
# live inventory), not extraction. LLMJ/Regex: raw extract vs /internal/l0_ground.
# QIE_Only_LLM: body.pre_ground_identified vs identified_filters (same grounder).
# Full_Search_LLM: post-ground set ∪ not_applied reason=inventory_ungrounded
# (see _collect_full_grounding_dropped).
GROUND_CALLER_ARMS = (ARM_LLMJ, ARM_QIE, ARM_FULL, ARM_REGEX)

# Arms with a not_applied reason bucket that only exists on their own pipeline
# (guard relaxation / backend capability / column availability — full-search only,
# see _collect_full_pipeline_dropped). Mismatches explained by these reasons are
# tallied separately from GROUND_CALLER_ARMS's grounding-explained bucket — they
# are a distinct cause, not a grounding drop, and mixing the two would mislabel
# GROUNDING_DROP reporting. Only Full_Search_LLM has this concept today.
PIPELINE_ARMS = (ARM_FULL,)

# Arms where entities can be migrated off the hard-filter set entirely, pre-retrieve,
# by SoftKeywordApplier.prepare_intent (retrieval/soft_keyword_apply.py) — a distinct
# cause from PIPELINE_ARMS's not_applied reasons: no not_applied entry is ever written
# for a soft downgrade, so it is invisible to PIPELINE_ARMS accounting unless tracked
# separately here. Only Full_Search_LLM has this concept today.
SOFT_DOWNGRADE_ARMS = (ARM_FULL,)


def _row_update(
    row: dict,
    *,
    sets: Dict[str, List[str]],
    latencies: Dict[str, Dict[str, float]],
    pre_ground: Optional[Dict[str, Optional[List[str]]]] = None,
    keywords: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    pipeline_dropped: Optional[Dict[str, Optional[List[str]]]] = None,
    soft_downgraded: Optional[Dict[str, Optional[List[str]]]] = None,
    costs: Optional[Dict[str, float]] = None,
    find_query_params: Optional[Dict[str, str]] = None,
    find_query_string: Optional[str] = None,
) -> dict:
    out = dict(row)
    for arm in ARMS:
        parts = latencies.get(arm) or _latency_parts()
        out[_arm_set_key(arm)] = sets[arm]
        e = round(float(parts.get("extract_ms", 0.0)), 1)
        h = round(float(parts.get("http_ms", 0.0)), 1)
        q = round(float(parts.get("queue_ms", 0.0)), 1)
        out[_arm_extract_ms_key(arm)] = e
        out[_arm_http_ms_key(arm)] = h
        out[_arm_queue_ms_key(arm)] = q
        # Work = extract + http (queue excluded — harness contention, not arm work).
        out[_arm_ms_key(arm)] = round(float(parts.get("ms", 0.0)), 1)
        # Single report column: HTTP round-trip + sem queue wait.
        out[_arm_http_queue_ms_key(arm)] = round(h + q, 1)
        out[_arm_cost_key(arm)] = round(float((costs or {}).get(arm, 0.0) or 0.0), 6)
        out[arm] = _disp(sets[arm])
    out["qie_source"] = "L0_live"
    out["offline"] = False
    out["grounding_model"] = _grounding_model()
    # Query-level total = sum of per-arm LLM spend (Regex contributes 0).
    out["total_cost_usd"] = round(
        sum(float(out.get(_arm_cost_key(a), 0.0) or 0.0) for a in ARMS), 6
    )

    for a, b in ARM_PAIRS:
        out[f"{a}_minus_{b}"] = sorted(set(sets[a]) - set(sets[b]))
        out[f"{b}_minus_{a}"] = sorted(set(sets[b]) - set(sets[a]))

    for arm in GROUND_CALLER_ARMS:
        raw = (pre_ground or {}).get(arm)
        if raw is None:
            continue
        raw_set = set(raw)
        out[_pre_ground_set_key(arm)] = sorted(raw_set)
        out[_dropped_by_grounding_key(arm)] = sorted(raw_set - set(sets[arm]))

    for arm in PIPELINE_ARMS:
        dropped = (pipeline_dropped or {}).get(arm)
        if dropped is None:
            continue
        out[_dropped_by_pipeline_key(arm)] = sorted(set(dropped))

    for arm in SOFT_DOWNGRADE_ARMS:
        dropped = (soft_downgraded or {}).get(arm)
        if dropped is None:
            continue
        out[_dropped_by_soft_key(arm)] = sorted(set(dropped))

    # Keyword sets — term-only (see _keywords_to_set). Regex uses topical fill.
    kw_raw: Dict[str, List[Dict[str, Any]]] = {
        arm: list((keywords or {}).get(arm) or []) for arm in ARMS
    }
    kw_sets: Dict[str, List[str]] = {
        arm: _keywords_to_set(kw_raw[arm]) for arm in ARMS
    }
    for arm in ARMS:
        out[_arm_kw_set_key(arm)] = kw_sets[arm]
        out[f"{arm}_keywords"] = _disp(kw_sets[arm])
        out[_arm_kw_detail_key(arm)] = _keywords_detail(kw_raw[arm])
    for a, b in ARM_PAIRS:
        out[f"{a}_kw_minus_{b}"] = sorted(set(kw_sets[a]) - set(kw_sets[b]))
        out[f"{b}_kw_minus_{a}"] = sorted(set(kw_sets[b]) - set(kw_sets[a]))

    ok = len({frozenset(sets[a]) for a in ARMS}) == 1
    out["status"] = "OK" if ok else "DIFF"
    # Three LLM extract arms only (Regex topical fill excluded from keyword_status).
    out["keyword_status"] = _keyword_three_arm_status(
        {a: kw_sets[a] for a in ARM_KEYWORD}
    )
    # QIE_Only FIND wire — informational (not part of filter/keyword OK/DIFF).
    if find_query_params is not None:
        out["find_query_params"] = dict(find_query_params)
    if find_query_string is not None:
        out["find_query_string"] = str(find_query_string)
    return out


async def amain() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "4-way L0 compare: LLMJ / QIE_Only_LLM / Full_Search_LLM / Regex "
            "(parallel, all grounded via :8085); OK = all 4 arms agree"
        ),
    )
    ap.add_argument(
        "--seed",
        action="store_true",
        help="Build/overwrite results from --md files (default: search+filter suites)",
    )
    ap.add_argument(
        "--md",
        type=Path,
        action="append",
        default=None,
        help="Markdown query suite (repeatable). Default: test_search + test_filter",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "output" / "reground_four_way",
        help="Directory for results / pairwise_summary / xlsx artifacts",
    )
    ap.add_argument(
        "--fail-only",
        action="store_true",
        help="Re-run rows that are not OK (DIFF + ERROR)",
    )
    ap.add_argument(
        "--error-only",
        action="store_true",
        help="Re-run only status=ERROR rows (transient HTTP / grounding failures)",
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--analysis-only",
        action="store_true",
        help=(
            "Skip HTTP/LLM; rebuild pairwise_summary xlsx + pairwise_summary_*.json "
            "from --results / --from-xlsx sibling JSON / latest results_*.json under --out-dir"
        ),
    )
    ap.add_argument(
        "--results",
        type=Path,
        default=None,
        help=(
            "Path to results_*.json (working file for a live run, or source for "
            "--analysis-only). Prefer output/reground_four_way/results_*.json."
        ),
    )
    ap.add_argument(
        "--from-xlsx",
        type=Path,
        default=None,
        help=(
            "With --analysis-only: use sibling results_{stamp}.json next to "
            "reground_{stamp}.xlsx (excel stamp maps to JSON; does not parse sheet cells)"
        ),
    )
    ap.add_argument(
        "--holdout-frac",
        type=float,
        default=DEFAULT_HOLDOUT_FRAC,
        help=(
            "Formal train/test holdout fraction for pairwise metrics "
            f"(default {DEFAULT_HOLDOUT_FRAC}; 0 disables). Stratified by suite."
        ),
    )
    ap.add_argument(
        "--holdout-seed",
        type=int,
        default=DEFAULT_HOLDOUT_SEED,
        help=f"Holdout split seed (default {DEFAULT_HOLDOUT_SEED}).",
    )
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    md_paths = [p.resolve() for p in (args.md or DEFAULT_MDS)]
    holdout_frac = float(args.holdout_frac)
    holdout_seed = int(args.holdout_seed)

    if args.analysis_only:
        results_path = _resolve_results_path(
            args.results, out_dir, from_xlsx=args.from_xlsx
        )
        text = await asyncio.to_thread(results_path.read_text)
        rows = json.loads(text)
        print(
            f"ANALYSIS_ONLY results={results_path} n={len(rows)} out_dir={out_dir} "
            f"holdout_frac={holdout_frac} holdout_seed={holdout_seed}"
        )
        write_analysis(
            rows, out_dir, holdout_frac=holdout_frac, holdout_seed=holdout_seed,
        )
        return 0

    results_path = (
        args.results.resolve()
        if args.results is not None
        else Path(tempfile.gettempdir())
        / f"reground_results_{uuid.uuid4().hex[:8]}.json"
    )

    if args.seed or not await asyncio.to_thread(results_path.is_file):
        rows = seed_rows_from_mds(md_paths)
        await asyncio.to_thread(results_path.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(results_path.write_text, json.dumps(rows, indent=2))
        print(f"SEEDED n={len(rows)} -> {results_path}")
        for md in md_paths:
            n_md = sum(1 for r in rows if r.get("source_md") == md.name)
            print(f"  {md.name}: {n_md}")
    else:
        text = await asyncio.to_thread(results_path.read_text)
        rows = json.loads(text)

    targets = list(range(len(rows)))
    if args.error_only:
        targets = [i for i, r in enumerate(rows) if r.get("status") == "ERROR"]
    elif args.fail_only:
        targets = [i for i, r in enumerate(rows) if r.get("status") != "OK"]
    if args.limit > 0:
        targets = targets[: args.limit]

    # Re-fill blanks (module imports above may have run with empty OPENAI_API_KEY).
    _fill_blank_env_from_dotenv(_REPO_ROOT / ".env")
    _fill_blank_env_from_dotenv(_REPO_ROOT.parent / ".env")
    complete = _build_grounding_complete()

    print(
        f"reground4 n={len(targets)}/{len(rows)} endpoint={ENDPOINT} "
        f"ground_endpoint={L0_GROUND_ENDPOINT} "
        f"concurrency={CONCURRENCY} (max={_MAX_QUERY_CONCURRENCY}) "
        f"http_inflight={_HTTP_INFLIGHT} "
        f"http_attempts={_HTTP_MAX_ATTEMPTS} "
        f"progress_every={PROGRESS_EVERY} progress_seconds={PROGRESS_INTERVAL_SECONDS} "
        f"grounding={_grounding_model()} "
        f"arms={','.join(ARMS)} "
        f"results={results_path}"
    )
    try:
        _post_search("ai domains under 100", qie_only=True)
        print("smoke ok (qie_only)")
    except Exception as exc:  # noqa: BLE001 — smoke probe must catch any backend failure
        print(f"ERROR: endpoint smoke failed: {exc}", file=sys.stderr)
        return 1

    l0r = _build_regex_extractor()
    sem = asyncio.Semaphore(CONCURRENCY)
    http_sem = asyncio.Semaphore(_HTTP_INFLIGHT)
    done = 0
    ok_n = 0
    err_n = 0
    t_all = time.perf_counter()
    lock = asyncio.Lock()

    async def one(idx: int) -> None:
        nonlocal done, ok_n, err_n
        row = rows[idx]
        q = row["query"]
        # Filled by arm_qie; must outlive `async with sem` for _row_update.
        qie_find_wire: Dict[str, Any] = {}
        async with sem:

            # 3rd element = pre-ground canonical set (None for arms that don't call
            # /internal/l0_ground with a raw payload we control) — lets _row_update
            # diff raw-vs-grounded to attribute drops to grounding enforcement.
            # 4th element = raw keyword dicts (term/probability) from that arm —
            # [] for ARM_REGEX, which has no keyword-extraction capability.
            # 5th element = pipeline-only not_applied drops (None for arms without
            # that concept — only ARM_FULL has one today, see PIPELINE_ARMS).
            # 6th element = soft-downgraded entities (migrated off the hard-filter
            # set pre-retrieve by SoftKeywordApplier — None for arms without that
            # concept — only ARM_FULL has one today, see SOFT_DOWNGRADE_ARMS).
            # 7th element = LLM cost_usd for that arm (Regex=0.0).
            async def arm_llmj() -> (
                Tuple[
                    List[str], Dict[str, float], Optional[List[str]], List[Dict[str, Any]],
                    Optional[List[str]], Optional[List[str]], float,
                ]
            ):
                t_ex0 = time.perf_counter()
                raw_identified, keywords, cost_usd = await asyncio.to_thread(
                    _llmj_raw_identified, complete, q
                )
                extract_ms = (time.perf_counter() - t_ex0) * 1000.0
                pre_ground_set = _entries_to_set(raw_identified)
                t_q0 = time.perf_counter()
                async with http_sem:
                    queue_ms = (time.perf_counter() - t_q0) * 1000.0
                    body, http_ms = await asyncio.to_thread(
                        _post_ground, q, raw_identified
                    )
                return (
                    _entries_to_set(_collect_qie(body)),
                    _latency_parts(
                        extract_ms=extract_ms, http_ms=http_ms, queue_ms=queue_ms
                    ),
                    pre_ground_set,
                    _collect_keywords_llmj(raw_identified, keywords),
                    None,
                    None,
                    float(cost_usd),
                )

            async def arm_qie() -> (
                Tuple[
                    List[str], Dict[str, float], Optional[List[str]], List[Dict[str, Any]],
                    Optional[List[str]], Optional[List[str]], float,
                ]
            ):
                t_q0 = time.perf_counter()
                async with http_sem:
                    queue_ms = (time.perf_counter() - t_q0) * 1000.0
                    body, http_ms = await asyncio.to_thread(
                        _post_search, q, qie_only=True
                    )
                fq_params, fq_string = _collect_qie_find_wire(body)
                qie_find_wire["find_query_params"] = fq_params
                qie_find_wire["find_query_string"] = fq_string
                return (
                    _entries_to_set(_collect_qie(body)),
                    # Live /search: extract happens inside the service — counted in http_ms.
                    _latency_parts(http_ms=http_ms, queue_ms=queue_ms),
                    _collect_qie_pre_ground(body),
                    _collect_keywords_qie(body),
                    None,
                    None,
                    _body_decision_cost_usd(body),
                )

            async def arm_full() -> (
                Tuple[
                    List[str], Dict[str, float], Optional[List[str]], List[Dict[str, Any]],
                    Optional[List[str]], Optional[List[str]], float,
                ]
            ):
                t_q0 = time.perf_counter()
                async with http_sem:
                    queue_ms = (time.perf_counter() - t_q0) * 1000.0
                    body, http_ms = await asyncio.to_thread(
                        _post_search, q, qie_only=False
                    )
                full_set = _entries_to_set(_collect_full(body))
                grounding_dropped_set = _entries_to_set(
                    _collect_full_grounding_dropped(body)
                )
                pipeline_dropped_set = _entries_to_set(
                    _collect_full_pipeline_dropped(body)
                )
                soft_downgraded_set = _entries_to_set(
                    _collect_full_soft_downgraded(body)
                )
                pre_ground_set = sorted(set(full_set) | set(grounding_dropped_set))
                return (
                    full_set,
                    _latency_parts(http_ms=http_ms, queue_ms=queue_ms),
                    pre_ground_set,
                    _collect_keywords_full(body),
                    sorted(pipeline_dropped_set),
                    sorted(soft_downgraded_set),
                    _body_decision_cost_usd(body),
                )

            async def arm_regex() -> (
                Tuple[
                    List[str], Dict[str, float], Optional[List[str]], List[Dict[str, Any]],
                    Optional[List[str]], Optional[List[str]], float,
                ]
            ):
                t_ex0 = time.perf_counter()
                raw_identified, _rex_cost, keywords = await l0r.extract_priced(q)
                extract_ms = (time.perf_counter() - t_ex0) * 1000.0
                pre_ground_set = _entries_to_set(raw_identified)
                t_q0 = time.perf_counter()
                async with http_sem:
                    queue_ms = (time.perf_counter() - t_q0) * 1000.0
                    body, http_ms = await asyncio.to_thread(
                        _post_ground, q, raw_identified
                    )
                return (
                    _entries_to_set(_collect_qie(body)),
                    _latency_parts(
                        extract_ms=extract_ms, http_ms=http_ms, queue_ms=queue_ms
                    ),
                    pre_ground_set,
                    keywords,
                    None,
                    None,
                    0.0,
                )

            # return_exceptions=True: one arm's transient failure (e.g. a 503 that
            # outlasts the retry budget) no longer cancels or discards the other
            # arms' already-completed work — only the failed arm(s) are marked ERROR.
            arm_results = await asyncio.gather(
                arm_llmj(),
                arm_qie(),
                arm_full(),
                arm_regex(),
                return_exceptions=True,
            )

        sets: Dict[str, List[str]] = {}
        arm_latencies: Dict[str, Dict[str, float]] = {}
        arm_pre_ground: Dict[str, Optional[List[str]]] = {}
        arm_keywords: Dict[str, List[Dict[str, Any]]] = {}
        arm_pipeline_dropped: Dict[str, Optional[List[str]]] = {}
        arm_soft_downgraded: Dict[str, Optional[List[str]]] = {}
        arm_costs: Dict[str, float] = {}
        arm_errors: Dict[str, str] = {}
        for arm, res in zip(ARMS, arm_results):
            if isinstance(res, BaseException):
                arm_errors[arm] = str(res)
                sets[arm] = []
                # Leave latency keys absent on ERROR — averages skip ERROR rows.
                arm_latencies[arm] = _latency_parts()
                arm_pre_ground[arm] = None
                arm_keywords[arm] = []
                arm_pipeline_dropped[arm] = None
                arm_soft_downgraded[arm] = None
                arm_costs[arm] = 0.0
            else:
                (
                    sets[arm], arm_latencies[arm], arm_pre_ground[arm], arm_keywords[arm],
                    arm_pipeline_dropped[arm], arm_soft_downgraded[arm], arm_costs[arm],
                ) = res

        updated = _row_update(
            row, sets=sets, latencies=arm_latencies, pre_ground=arm_pre_ground,
            keywords=arm_keywords, pipeline_dropped=arm_pipeline_dropped,
            soft_downgraded=arm_soft_downgraded, costs=arm_costs,
            find_query_params=qie_find_wire.get("find_query_params"),
            find_query_string=qie_find_wire.get("find_query_string"),
        )
        # Do not report zeroed latency/cost for failed arms (was biasing means to 0).
        if arm_errors:
            for arm in arm_errors:
                for key_fn in (
                    _arm_ms_key,
                    _arm_extract_ms_key,
                    _arm_http_ms_key,
                    _arm_queue_ms_key,
                    _arm_http_queue_ms_key,
                    _arm_cost_key,
                ):
                    updated.pop(key_fn(arm), None)
            updated.pop("total_cost_usd", None)
        updated.pop("error", None)
        if arm_errors:
            err_summary = "; ".join(f"{arm}: {msg}" for arm, msg in arm_errors.items())
            print(f"  FAIL arms idx={idx} q={q[:60]!r} err={err_summary}", flush=True)
            updated["status"] = "ERROR"
            updated["error"] = err_summary
            updated["arm_errors"] = arm_errors
            for arm in arm_errors:
                updated[arm] = f"ERROR: {arm_errors[arm]}"

        async with lock:
            rows[idx] = updated
            done += 1
            if updated["status"] == "OK":
                ok_n += 1
            elif updated["status"] == "ERROR":
                err_n += 1
            if done % PROGRESS_EVERY == 0 or done == len(targets):
                await asyncio.to_thread(
                    results_path.write_text, json.dumps(rows, indent=2)
                )
                elapsed = time.perf_counter() - t_all
                rate = done / elapsed if elapsed else 0
                eta = (len(targets) - done) / rate if rate else 0
                print(
                    f"  progress {done}/{len(targets)} ok={ok_n} err={err_n} "
                    f"rate={rate:.2f}/s elapsed={elapsed:.0f}s eta={eta:.0f}s "
                    f"last={q[:50]!r} status={updated['status']}",
                    flush=True,
                )

    async def heartbeat() -> None:
        """Wall-clock progress log every PROGRESS_INTERVAL_SECONDS (default 90s),
        regardless of per-query completion cadence — catches a hung/slow run that
        the completion-count-based `progress` line above wouldn't emit for a while.
        """
        while True:
            await asyncio.sleep(PROGRESS_INTERVAL_SECONDS)
            elapsed = time.perf_counter() - t_all
            rate = done / elapsed if elapsed else 0
            eta = (len(targets) - done) / rate if rate else 0
            print(
                f"  heartbeat {done}/{len(targets)} ok={ok_n} err={err_n} "
                f"rate={rate:.2f}/s elapsed={elapsed:.0f}s eta={eta:.0f}s "
                f"(every {PROGRESS_INTERVAL_SECONDS}s)",
                flush=True,
            )

    hb_task = asyncio.create_task(heartbeat())
    try:
        await asyncio.gather(*(one(i) for i in targets))
    finally:
        hb_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await hb_task

    await asyncio.to_thread(results_path.write_text, json.dumps(rows, indent=2))
    all_ok = sum(1 for r in rows if r.get("status") == "OK")
    print(
        f"DONE wrote {results_path} OK={all_ok}/{len(rows)} "
        f"({100.0 * all_ok / len(rows):.1f}%) "
        f"batch_ok={ok_n}/{len(targets)} err={err_n}",
        flush=True,
    )
    write_analysis(
        rows, out_dir, holdout_frac=holdout_frac, holdout_seed=holdout_seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))
