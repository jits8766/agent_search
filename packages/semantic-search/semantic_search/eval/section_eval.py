"""Section-driven /search evaluation harness — runs the curated query bank in
``test_search_queries.md`` against a live ``/search`` endpoint and writes a
Markdown report to ``_temp/eval_*.md``.

Unlike ``query_eval_runner.py`` (curated YAML cases with per-query expected
filters), this harness is intent-/segment-oriented:

  * Queries are parsed straight from ``test_search_queries.md``. The top-level
    banners (HYBRID / EXPLORE / GUIDANCE / ANALYTICS) are the *expected* intent
    label; each ``### Sub-heading`` is a sub-segment (e.g. "SEO + Authority").
  * Every query is fired at ``/search`` in parallel (bounded concurrency).
  * The report rolls metrics up **per sub-segment** and **per section** —
    classification/routing accuracy, latency & SLA breaches, errors, zero-result
    and explore-fallback rates, and label-free retrieval quality (NDCG/Coherence).
  * A heuristic "Gaps & Improvement Scope" section flags sub-segments that
    misroute, breach SLA, error, or return nothing — so issues read as
    "SEO queries in HYBRID misroute to guidance" rather than per-query noise.

No LLM is invoked by this script: it is a pure HTTP client + aggregator and can
be run any time, unattended.

Run:
    # all sections, default localhost endpoint
    python -m semantic_search.eval.section_eval

    # one section only, higher concurrency
    python -m semantic_search.eval.section_eval --section HYBRID --concurrency 12

    # filter to a single sub-segment substring
    python -m semantic_search.eval.section_eval --section ANALYTICS --subsegment "TLD"

Env overrides: SEARCH_ENDPOINT, SEARCH_QUERIES_FILE, SEARCH_EVAL_OUT_DIR.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import re
import statistics
import sys
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Reuse the SSRF-guarded synchronous POST so this harness shares the exact same
# endpoint allowlist as query_eval_runner (no duplicated security surface).
# Loaded by file path rather than `from semantic_search.eval...` so importing it
# never triggers the eval package __init__ (which pulls llm_core and the full LLM
# stack) — this keeps the harness runnable in a minimal env with the service up.
def _load_qe_runner():
    import importlib.util
    sibling = Path(__file__).resolve().parent / "query_eval_runner.py"
    spec = importlib.util.spec_from_file_location("_qe_runner_for_section_eval", sibling)
    if spec is None or spec.loader is None:  # pragma: no cover — defensive
        raise ImportError(f"cannot load {sibling}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_qe = _load_qe_runner()
_validate_endpoint_url = _qe._validate_endpoint_url


def _post(url: str, data: bytes, session_id: str) -> Dict[str, Any]:
    """SSRF-guarded POST that tags the request with a per-query ``X-Session-Id``.

    The server rate-limit middleware buckets by ``X-Session-Id`` (falling back
    to client IP). A unique id per query gives each its own bucket, so the
    eval sweep isn't throttled by the 10-req/60s per-session limit — every
    query models a distinct session, which is exactly what the limiter keys on.
    """
    import httpx
    _validate_endpoint_url(url)
    r = httpx.post(url, content=data, headers={"X-Session-Id": session_id}, timeout=45.0)
    r.raise_for_status()
    return r.json()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ──────────────────────────────────────────────────────────────────────────────
# Defaults — resolved relative to the repo so no machine-specific paths leak in.
# Repo layout: <repo>/auc-semantic-search/packages/semantic-search/semantic_search/eval/section_eval.py
# parents[4] == <repo>; test_search_queries.md and _temp/ both live at <repo> root.
# ──────────────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[5]
_DEFAULT_QUERIES_FILE = _REPO_ROOT / "test_search_queries.md"
_DEFAULT_OUT_DIR = _REPO_ROOT / "_temp"
_DEFAULT_ENDPOINT = "http://localhost:8085/search"

# Top-level banner → expected classified intent. The banner is the ground-truth
# routing label this harness scores classification accuracy against.
_SECTION_TO_INTENT: Dict[str, str] = {
    "HYBRID": "hybrid",
    "EXPLORE": "explore",
    "GUIDANCE": "guidance",
    "ANALYTICS": "analytics",
}

# SLA wall-clock ceilings (seconds) by expected intent — mirrors config base.yaml
# general.search.{search_timeout_seconds, analytics_total_budget_seconds}.
_SLA_SECONDS: Dict[str, float] = {
    "hybrid": 5.0,
    "explore": 5.0,
    "guidance": 5.0,
    "analytics": 7.0,
}

# failure_mode values the /search response sets when an SLA timeout fired.
_TIMEOUT_FAILURE_MODES = frozenset({"timeout", "analytics_timeout"})


# ──────────────────────────────────────────────────────────────────────────────
# Data models
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class QuerySpec:
    section: str          # HYBRID / EXPLORE / GUIDANCE / ANALYTICS
    subsegment: str       # "### " heading text
    index: int            # 1-based position within the file
    query: str
    expected_intent: str  # _SECTION_TO_INTENT[section]


@dataclass
class QueryResult:
    spec: QuerySpec
    classified_intent: Optional[str]
    intent_confidence: Optional[float]
    decision_tier: Optional[str]
    routing_applied: Optional[str]
    answer_mode: str
    n_results: int
    client_latency_ms: float          # measured round-trip from this harness
    server_latency_ms: Optional[float]  # response.latency_ms (endpoint-internal)
    failure_mode: Optional[str]
    metrics_valid: bool
    ndcg: Optional[float]
    coherence: Optional[float]
    total_candidates: Optional[int]
    backends_active: List[str]
    guard_fired: bool
    guard_notice: Optional[str]
    error: Optional[str] = None

    @property
    def intent_match(self) -> bool:
        return self.classified_intent == self.spec.expected_intent

    @property
    def sla_seconds(self) -> float:
        return _SLA_SECONDS.get(self.spec.expected_intent, 5.0)

    @property
    def sla_breach(self) -> bool:
        # Two independent breach signals: measured latency over the ceiling, OR
        # the endpoint itself reported a timeout failure_mode / explore fallback.
        if self.error:
            return False  # transport error tracked separately, not an SLA breach
        if self.client_latency_ms > self.sla_seconds * 1000.0:
            return True
        if self.failure_mode in _TIMEOUT_FAILURE_MODES:
            return True
        return False

    @property
    def is_timeout_fallback(self) -> bool:
        return self.answer_mode == "explore_fallback" and self.failure_mode in _TIMEOUT_FAILURE_MODES

    @property
    def zero_results(self) -> bool:
        # Analytics answers live in the analytics block, not ranked_results, so a
        # zero ranked_results count is only a retrieval gap for non-analytics paths.
        if self.error or self.spec.expected_intent == "analytics":
            return False
        return self.n_results == 0


# ──────────────────────────────────────────────────────────────────────────────
# Markdown query-bank parser
# ──────────────────────────────────────────────────────────────────────────────

_SECTION_RE = re.compile(r"^(HYBRID|EXPLORE|GUIDANCE|ANALYTICS)\s*$")
_SUBSEG_RE = re.compile(r"^#{2,4}\s+(.*\S)\s*$")
_QUERY_RE = re.compile(r"^\s*\d+\.\s+(.*\S)\s*$")


def parse_query_bank(path: Path) -> List[QuerySpec]:
    """Parse ``test_search_queries.md`` into ordered QuerySpec rows.

    Recognises the top-level banners as sections, ``### x`` lines as
    sub-segments, and ``N. query`` lines as queries. Anything before the first
    banner is ignored.
    """
    specs: List[QuerySpec] = []
    section: Optional[str] = None
    subsegment = "(unsegmented)"
    counter = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        m_sec = _SECTION_RE.match(line)
        if m_sec:
            section = m_sec.group(1)
            subsegment = "(unsegmented)"
            continue
        m_sub = _SUBSEG_RE.match(line)
        if m_sub:
            subsegment = m_sub.group(1)
            continue
        m_q = _QUERY_RE.match(line)
        if m_q and section is not None:
            counter += 1
            specs.append(QuerySpec(
                section=section,
                subsegment=subsegment,
                index=counter,
                query=m_q.group(1),
                expected_intent=_SECTION_TO_INTENT[section],
            ))
    return specs


# ──────────────────────────────────────────────────────────────────────────────
# Search driver
# ──────────────────────────────────────────────────────────────────────────────

async def _run_one(spec: QuerySpec, endpoint: str, top_k: int, sem: asyncio.Semaphore) -> QueryResult:
    async with sem:
        data = urllib.parse.urlencode({"query": spec.query, "top_k": top_k}).encode()
        session_id = f"eval-{spec.section.lower()}-{spec.index}"
        t0 = time.perf_counter()
        try:
            resp = await asyncio.to_thread(_post, endpoint, data, session_id)
        except Exception as exc:  # noqa: BLE001 — any transport/parse failure is a recorded error row
            client_ms = (time.perf_counter() - t0) * 1000.0
            logger.warning(f"query_failed section={spec.section} idx={spec.index} error={exc}")
            return QueryResult(
                spec=spec, classified_intent=None, intent_confidence=None,
                decision_tier=None, routing_applied=None, answer_mode="",
                n_results=0, client_latency_ms=client_ms, server_latency_ms=None,
                failure_mode=None, metrics_valid=False, ndcg=None, coherence=None,
                total_candidates=None, backends_active=[], guard_fired=False,
                guard_notice=None, error=str(exc),
            )
        client_ms = (time.perf_counter() - t0) * 1000.0

    qi = resp.get("query_intelligence") or {}
    rm = resp.get("retrieval_metrics") or {}
    pt = resp.get("pipeline_trace") or {}
    # Metric keys carry the @K suffix (NDCG@50, Coherence@50, …) — match by prefix.
    ndcg = _first_metric(rm, "NDCG@")
    coherence = _first_metric(rm, "Coherence@")
    return QueryResult(
        spec=spec,
        classified_intent=qi.get("classified_intent"),
        intent_confidence=qi.get("intent_confidence"),
        decision_tier=qi.get("decision_tier"),
        routing_applied=qi.get("routing_applied"),
        answer_mode=str(resp.get("answer_mode") or ""),
        n_results=len(resp.get("ranked_results") or []),
        client_latency_ms=client_ms,
        server_latency_ms=resp.get("latency_ms"),
        failure_mode=rm.get("failure_mode"),
        metrics_valid=bool(rm.get("metrics_valid", False)),
        ndcg=ndcg,
        coherence=coherence,
        total_candidates=rm.get("total_candidates"),
        backends_active=list(rm.get("backends_active") or []),
        guard_fired=bool(pt.get("zero_result_guard_fired", False)),
        guard_notice=resp.get("guard_notice"),
    )


def _first_metric(metrics: Dict[str, Any], prefix: str) -> Optional[float]:
    for k, v in metrics.items():
        if k.startswith(prefix) and isinstance(v, (int, float)):
            return float(v)
    return None


async def run_all(specs: List[QuerySpec], endpoint: str, top_k: int, concurrency: int) -> List[QueryResult]:
    sem = asyncio.Semaphore(max(1, concurrency))
    tasks = [asyncio.create_task(_run_one(s, endpoint, top_k, sem)) for s in specs]
    results: List[QueryResult] = []
    for fut in asyncio.as_completed(tasks):
        results.append(await fut)
    # Restore deterministic file order for the report.
    results.sort(key=lambda r: r.spec.index)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Agg:
    label: str
    n: int = 0
    errors: int = 0
    intent_hits: int = 0
    sla_breaches: int = 0
    timeouts: int = 0
    zero_results: int = 0
    fallbacks: int = 0
    guard_fired: int = 0
    latencies: List[float] = field(default_factory=list)
    ndcgs: List[float] = field(default_factory=list)
    classified_dist: Dict[str, int] = field(default_factory=dict)
    routing_dist: Dict[str, int] = field(default_factory=dict)

    def add(self, r: QueryResult) -> None:
        self.n += 1
        if r.error:
            self.errors += 1
            return
        if r.intent_match:
            self.intent_hits += 1
        if r.sla_breach:
            self.sla_breaches += 1
        if r.is_timeout_fallback:
            self.timeouts += 1
        if r.zero_results:
            self.zero_results += 1
        if r.answer_mode == "explore_fallback":
            self.fallbacks += 1
        if r.guard_fired:
            self.guard_fired += 1
        self.latencies.append(r.client_latency_ms)
        if r.metrics_valid and r.ndcg is not None:
            self.ndcgs.append(r.ndcg)
        ci = r.classified_intent or "(none)"
        self.classified_dist[ci] = self.classified_dist.get(ci, 0) + 1
        ra = r.routing_applied or "(none)"
        self.routing_dist[ra] = self.routing_dist.get(ra, 0) + 1

    @property
    def scored(self) -> int:
        return self.n - self.errors

    @property
    def intent_acc(self) -> float:
        return self.intent_hits / self.scored if self.scored else 0.0

    @property
    def avg_lat(self) -> float:
        return statistics.mean(self.latencies) if self.latencies else 0.0

    @property
    def p50_lat(self) -> float:
        return statistics.median(self.latencies) if self.latencies else 0.0

    @property
    def p95_lat(self) -> float:
        return _percentile(self.latencies, 95)

    @property
    def max_lat(self) -> float:
        return max(self.latencies) if self.latencies else 0.0

    @property
    def avg_ndcg(self) -> Optional[float]:
        return statistics.mean(self.ndcgs) if self.ndcgs else None

    def dominant_route(self) -> str:
        if not self.routing_dist:
            return "(none)"
        return max(self.routing_dist.items(), key=lambda kv: kv[1])[0]


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (pct / 100.0) * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    frac = rank - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def aggregate(results: List[QueryResult], key) -> Dict[str, Agg]:
    out: Dict[str, Agg] = {}
    for r in results:
        k = key(r)
        out.setdefault(k, Agg(label=k)).add(r)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Heuristic gap detection (sub-segment-wise, no LLM)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Gap:
    segment: str          # "HYBRID › SEO + Authority"
    severity: str         # HIGH / MED / LOW
    kind: str
    detail: str


def detect_gaps(results: List[QueryResult]) -> List[Gap]:
    """Flag sub-segment-level problems from the aggregated signals.

    Thresholds are deliberately conservative so a flag means a real pattern,
    not single-query noise. Each gap names the segment and the corrective lens.
    """
    gaps: List[Gap] = []
    by_seg = aggregate(results, key=lambda r: f"{r.spec.section} › {r.spec.subsegment}")
    # Per-segment misroute target needs the raw rows.
    rows_by_seg: Dict[str, List[QueryResult]] = {}
    for r in results:
        rows_by_seg.setdefault(f"{r.spec.section} › {r.spec.subsegment}", []).append(r)

    for seg, a in by_seg.items():
        if a.scored == 0 and a.errors > 0:
            gaps.append(Gap(seg, "HIGH", "endpoint_error",
                            f"all {a.errors}/{a.n} queries errored — endpoint unreachable or crashing on this segment"))
            continue
        if a.errors:
            sev = "HIGH" if a.errors / a.n >= 0.3 else "MED"
            gaps.append(Gap(seg, sev, "endpoint_error",
                            f"{a.errors}/{a.n} queries errored (transport/5xx) — inspect server logs"))

        # Misrouting: classification accuracy below 0.8 on a non-trivial segment.
        if a.scored >= 3 and a.intent_acc < 0.8:
            wrong = _misroute_targets(rows_by_seg[seg])
            sev = "HIGH" if a.intent_acc < 0.5 else "MED"
            gaps.append(Gap(seg, sev, "misrouting",
                            f"intent accuracy {a.intent_acc:.0%} ({a.intent_hits}/{a.scored}); "
                            f"classified as {wrong} instead of '{rows_by_seg[seg][0].spec.expected_intent}'"))

        # SLA: any breach in a segment is worth surfacing; >30% is high.
        if a.sla_breaches:
            rate = a.sla_breaches / max(1, a.scored)
            sev = "HIGH" if rate >= 0.3 else "MED"
            gaps.append(Gap(seg, sev, "sla_breach",
                            f"{a.sla_breaches}/{a.scored} over {_SLA_SECONDS.get(rows_by_seg[seg][0].spec.expected_intent, 5.0):.0f}s SLA "
                            f"(p95={a.p95_lat:.0f}ms, max={a.max_lat:.0f}ms, timeouts={a.timeouts})"))

        # Zero-result retrieval gaps (non-analytics only).
        if a.zero_results:
            rate = a.zero_results / max(1, a.scored)
            sev = "HIGH" if rate >= 0.4 else "MED"
            gaps.append(Gap(seg, sev, "zero_results",
                            f"{a.zero_results}/{a.scored} returned no ranked results — "
                            f"over-constrained filters or corpus gap for this segment"))

        # Explore-fallback (non-timeout) — degraded answer quality.
        non_timeout_fallbacks = a.fallbacks - a.timeouts
        if non_timeout_fallbacks > 0 and a.scored >= 3 and non_timeout_fallbacks / a.scored >= 0.4:
            gaps.append(Gap(seg, "MED", "fallback",
                            f"{non_timeout_fallbacks}/{a.scored} fell back to explore (non-timeout) — "
                            f"primary retrieval/analytics path not satisfying this segment"))

        # Ranking quality — low NDCG where metrics are valid.
        if a.avg_ndcg is not None and len(a.ndcgs) >= 3 and a.avg_ndcg < 0.5:
            gaps.append(Gap(seg, "LOW", "ranking_quality",
                            f"avg NDCG {a.avg_ndcg:.2f} over {len(a.ndcgs)} scored queries — weak rank sharpness"))

    sev_order = {"HIGH": 0, "MED": 1, "LOW": 2}
    gaps.sort(key=lambda g: (sev_order.get(g.severity, 9), g.segment))
    return gaps


def _misroute_targets(rows: List[QueryResult]) -> str:
    dist: Dict[str, int] = {}
    expected = rows[0].spec.expected_intent
    for r in rows:
        if r.error or r.intent_match:
            continue
        ci = r.classified_intent or "(none)"
        dist[ci] = dist.get(ci, 0) + 1
    if not dist:
        return "(n/a)"
    return ", ".join(f"{k}×{v}" for k, v in sorted(dist.items(), key=lambda kv: -kv[1]))


# ──────────────────────────────────────────────────────────────────────────────
# Markdown report
# ──────────────────────────────────────────────────────────────────────────────

def _pct(num: int, den: int) -> str:
    return f"{(num / den * 100):.0f}%" if den else "—"


def build_report(
    results: List[QueryResult],
    *,
    endpoint: str,
    top_k: int,
    concurrency: int,
    section_filter: str,
    subsegment_filter: Optional[str],
    wall_seconds: float,
    queries_file: Path,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    overall = Agg(label="ALL")
    for r in results:
        overall.add(r)
    by_section = aggregate(results, key=lambda r: r.spec.section)
    by_seg = aggregate(results, key=lambda r: f"{r.spec.section} › {r.spec.subsegment}")
    gaps = detect_gaps(results)

    L: List[str] = []
    L.append(f"# Search Evaluation Report")
    L.append("")
    L.append(f"- **Generated:** {now}")
    L.append(f"- **Endpoint:** `{endpoint}`")
    L.append(f"- **Query bank:** `{queries_file.name}`")
    L.append(f"- **Scope:** section=`{section_filter}`" + (f", subsegment~`{subsegment_filter}`" if subsegment_filter else ""))
    L.append(f"- **Params:** top_k={top_k}, concurrency={concurrency}")
    L.append(f"- **Wall time:** {wall_seconds:.1f}s for {overall.n} queries "
             f"({(overall.n / wall_seconds):.1f} q/s effective)")
    L.append("")

    # ── Executive summary ────────────────────────────────────────────────────
    L.append("## 1. Executive Summary")
    L.append("")
    L.append(f"- Queries run: **{overall.n}** · errors: **{overall.errors}** · scored: **{overall.scored}**")
    L.append(f"- Classification accuracy: **{overall.intent_acc:.0%}** ({overall.intent_hits}/{overall.scored})")
    L.append(f"- SLA breaches: **{overall.sla_breaches}** ({_pct(overall.sla_breaches, overall.scored)}) · "
             f"timeouts→fallback: **{overall.timeouts}**")
    L.append(f"- Explore fallbacks: **{overall.fallbacks}** · zero-result (non-analytics): **{overall.zero_results}** · "
             f"zero-result guard fired: **{overall.guard_fired}**")
    L.append(f"- Latency: p50 **{overall.p50_lat:.0f}ms** · p95 **{overall.p95_lat:.0f}ms** · max **{overall.max_lat:.0f}ms**")
    if overall.avg_ndcg is not None:
        L.append(f"- Avg NDCG (valid-metric queries): **{overall.avg_ndcg:.2f}** over {len(overall.ndcgs)} queries")
    L.append("")
    high = sum(1 for g in gaps if g.severity == "HIGH")
    med = sum(1 for g in gaps if g.severity == "MED")
    L.append(f"- **Gaps flagged:** {len(gaps)} (HIGH={high}, MED={med}, LOW={len(gaps) - high - med}) — see §6")
    L.append("")

    # ── Per-section rollup ───────────────────────────────────────────────────
    L.append("## 2. Per-Section Rollup")
    L.append("")
    L.append("| Section | N | Err | Intent acc | p50 ms | p95 ms | max ms | SLA breach | Timeouts | Fallback | Zero-res | Avg NDCG |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for sec in ("HYBRID", "EXPLORE", "GUIDANCE", "ANALYTICS"):
        a = by_section.get(sec)
        if a is None:
            continue
        ndcg = f"{a.avg_ndcg:.2f}" if a.avg_ndcg is not None else "—"
        L.append(f"| {sec} | {a.n} | {a.errors} | {a.intent_acc:.0%} | {a.p50_lat:.0f} | {a.p95_lat:.0f} | "
                 f"{a.max_lat:.0f} | {a.sla_breaches} | {a.timeouts} | {a.fallbacks} | {a.zero_results} | {ndcg} |")
    L.append("")

    # ── Classification confusion ─────────────────────────────────────────────
    L.append("## 3. Classification & Routing Accuracy")
    L.append("")
    L.append("Expected intent (section banner) vs. classified intent returned by QI.")
    L.append("")
    L.append("| Section (expected) | Classified distribution | Accuracy |")
    L.append("|---|---|--:|")
    for sec in ("HYBRID", "EXPLORE", "GUIDANCE", "ANALYTICS"):
        a = by_section.get(sec)
        if a is None:
            continue
        dist = ", ".join(f"`{k}`×{v}" for k, v in sorted(a.classified_dist.items(), key=lambda kv: -kv[1]))
        L.append(f"| {sec} | {dist or '—'} | {a.intent_acc:.0%} |")
    L.append("")

    # ── Latency & SLA ────────────────────────────────────────────────────────
    L.append("## 4. Latency & SLA")
    L.append("")
    L.append("SLA ceilings: hybrid/explore/guidance **5s**, analytics **7s**. "
             "A breach = measured latency over the ceiling *or* an endpoint timeout `failure_mode`.")
    L.append("")
    breached = [r for r in results if r.sla_breach]
    if breached:
        L.append("| Section › Sub-segment | Query | Latency ms | SLA s | failure_mode | answer_mode |")
        L.append("|---|---|--:|--:|---|---|")
        for r in sorted(breached, key=lambda r: -r.client_latency_ms):
            L.append(f"| {r.spec.section} › {r.spec.subsegment} | {_md(r.spec.query)} | "
                     f"{r.client_latency_ms:.0f} | {r.sla_seconds:.0f} | {r.failure_mode or '—'} | {r.answer_mode or '—'} |")
    else:
        L.append("_No SLA breaches._")
    L.append("")

    # ── Errors ───────────────────────────────────────────────────────────────
    L.append("## 5. Errors")
    L.append("")
    errs = [r for r in results if r.error]
    if errs:
        L.append("| Section › Sub-segment | Query | Error |")
        L.append("|---|---|---|")
        for r in errs:
            L.append(f"| {r.spec.section} › {r.spec.subsegment} | {_md(r.spec.query)} | {_md(str(r.error))} |")
    else:
        L.append("_No transport/endpoint errors._")
    L.append("")

    # ── Gaps & improvement scope (the core deliverable) ──────────────────────
    L.append("## 6. Gaps & Improvement Scope (sub-segment-wise)")
    L.append("")
    if gaps:
        L.append("| Severity | Segment | Kind | Detail |")
        L.append("|---|---|---|---|")
        for g in gaps:
            L.append(f"| {g.severity} | {g.segment} | `{g.kind}` | {_md(g.detail)} |")
    else:
        L.append("_No sub-segment-level gaps detected at current thresholds._")
    L.append("")

    # ── Per-sub-segment detail ───────────────────────────────────────────────
    L.append("## 7. Per-Sub-Segment Detail")
    L.append("")
    L.append("| Section › Sub-segment | N | Intent acc | avg ms | p95 ms | SLA brk | Zero-res | Fallback | Avg NDCG | Dominant route |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|---|")
    for seg in sorted(by_seg.keys()):
        a = by_seg[seg]
        ndcg = f"{a.avg_ndcg:.2f}" if a.avg_ndcg is not None else "—"
        L.append(f"| {seg} | {a.n} | {a.intent_acc:.0%} | {a.avg_lat:.0f} | {a.p95_lat:.0f} | "
                 f"{a.sla_breaches} | {a.zero_results} | {a.fallbacks} | {ndcg} | `{a.dominant_route()}` |")
    L.append("")

    # ── Appendix: per-query rows ─────────────────────────────────────────────
    L.append("## 8. Appendix — Per-Query Results")
    L.append("")
    L.append("<details><summary>Expand all queries</summary>")
    L.append("")
    L.append("| # | Section › Sub-segment | Query | Expected | Classified | ✓ | mode | ms | results | failure_mode |")
    L.append("|--:|---|---|---|---|:-:|---|--:|--:|---|")
    for r in results:
        tick = "✅" if (not r.error and r.intent_match) else ("⚠️" if r.error else "❌")
        cls = r.classified_intent or ("ERR" if r.error else "—")
        L.append(f"| {r.spec.index} | {r.spec.section} › {r.spec.subsegment} | {_md(r.spec.query)} | "
                 f"{r.spec.expected_intent} | {cls} | {tick} | {r.answer_mode or '—'} | "
                 f"{r.client_latency_ms:.0f} | {r.n_results} | {r.failure_mode or '—'} |")
    L.append("")
    L.append("</details>")
    L.append("")
    return "\n".join(L)


def _md(text: str) -> str:
    """Escape pipe characters so free text doesn't break Markdown tables."""
    return str(text).replace("|", "\\|").replace("\n", " ")


# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    import os
    p = argparse.ArgumentParser(description="Section-driven /search evaluation → Markdown report in _temp/")
    p.add_argument("--section", default="all",
                   help="HYBRID | EXPLORE | GUIDANCE | ANALYTICS | all (default: all)")
    p.add_argument("--subsegment", default=None,
                   help="Case-insensitive substring filter on the '### ' sub-heading (e.g. 'SEO')")
    p.add_argument("--endpoint", default=os.environ.get("SEARCH_ENDPOINT", _DEFAULT_ENDPOINT),
                   help=f"Search endpoint URL (default env SEARCH_ENDPOINT or {_DEFAULT_ENDPOINT})")
    p.add_argument("--queries-file", default=os.environ.get("SEARCH_QUERIES_FILE", str(_DEFAULT_QUERIES_FILE)),
                   help="Path to the Markdown query bank (default: repo test_search_queries.md)")
    p.add_argument("--out-dir", default=os.environ.get("SEARCH_EVAL_OUT_DIR", str(_DEFAULT_OUT_DIR)),
                   help="Directory for the eval_*.md report (default: repo _temp/)")
    p.add_argument("--top-k", type=int, default=10, help="top_k passed to /search (default: 10)")
    p.add_argument("--concurrency", type=int, default=8, help="Max in-flight requests (default: 8)")
    p.add_argument("--limit", type=int, default=0, help="Cap total queries (0 = no cap; for smoke tests)")
    return p


def _filter_specs(specs: List[QuerySpec], section: str, subsegment: Optional[str], limit: int) -> List[QuerySpec]:
    sec = section.strip().upper()
    out = specs
    if sec != "ALL":
        if sec not in _SECTION_TO_INTENT:
            raise SystemExit(f"--section must be one of {list(_SECTION_TO_INTENT)} or 'all'; got {section!r}")
        out = [s for s in out if s.section == sec]
    if subsegment:
        needle = subsegment.lower()
        out = [s for s in out if needle in s.subsegment.lower()]
    if limit and limit > 0:
        out = out[:limit]
    return out


async def _amain(args: argparse.Namespace) -> int:
    queries_file = Path(args.queries_file)
    if not queries_file.is_file():
        raise SystemExit(f"query bank not found: {queries_file}")
    specs = parse_query_bank(queries_file)
    specs = _filter_specs(specs, args.section, args.subsegment, args.limit)
    if not specs:
        raise SystemExit("no queries matched the section/subsegment filter")

    logger.info(f"running {len(specs)} queries against {args.endpoint} "
                f"(section={args.section}, concurrency={args.concurrency})")
    t0 = time.perf_counter()
    results = await run_all(specs, args.endpoint, args.top_k, args.concurrency)
    wall = time.perf_counter() - t0

    report = build_report(
        results, endpoint=args.endpoint, top_k=args.top_k, concurrency=args.concurrency,
        section_filter=args.section, subsegment_filter=args.subsegment,
        wall_seconds=wall, queries_file=queries_file,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    sec_tag = args.section.strip().lower()
    out_path = out_dir / f"eval_{sec_tag}_{stamp}.md"
    out_path.write_text(report, encoding="utf-8")

    # Console one-liner so the run is legible without opening the file.
    ov = Agg(label="ALL")
    for r in results:
        ov.add(r)
    logger.info(f"done: {ov.n} queries, intent_acc={ov.intent_acc:.0%}, "
                f"sla_breaches={ov.sla_breaches}, errors={ov.errors}, p95={ov.p95_lat:.0f}ms")
    print(f"\nReport written: {out_path}")
    return 0


def main() -> int:
    args = _build_arg_parser().parse_args()
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
