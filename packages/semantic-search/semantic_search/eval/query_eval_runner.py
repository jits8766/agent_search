"""Generic async eval harness for /search endpoint query validation.

Loads test cases from a YAML config, calls /search for each query, then:
  - Scores filter extraction accuracy (extracted vs expected_filters)
  - Checks ranked_result payload compliance per declared ops
  - Runs semantic relevance scoring via nomic-embed (or transformers fallback)

All parameters come from the YAML config — no hardcoded query-specific logic.

500-query eval optimisations (behaviour-preserving):
  - Parallel execution via asyncio.Semaphore(concurrency) — default 20 workers.
  - Unique X-Session-Id per case so each gets its own rate-limit bucket; the
    service's session_state_max: 10000 cap comfortably holds 500 buckets.
  - Automatic 429 retry with Retry-After backoff — guards against accidental
    bucket collisions without hiding real rate-limit violations in tests.
  - Fixed inter-request sleep removed; rate pacing is now the semaphore + retry.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import sys
import time
import urllib.parse
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

_DEFAULT_CASES_PATH = Path(__file__).parent / "query_eval_cases.yaml"

# ──────────────────────────────────────────────────────────────────────────────
# Data models
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ComplianceCheck:
    field: str
    op: str
    value: Any


@dataclass
class EvalCase:
    case_id: str
    query: str
    category: str
    expected_filters: Dict[str, Any]
    compliance_checks: List[ComplianceCheck]
    expected_query_types: List[str]
    semantic_topic_hints: List[str]
    notes: str


@dataclass
class FilterScore:
    slot: str
    expected: Any
    extracted: Any
    match: bool


@dataclass
class ComplianceResult:
    check: ComplianceCheck
    item_id: str
    domain_name: str
    actual_value: Any
    passed: bool
    reason: str


@dataclass
class CaseResult:
    case_id: str
    query: str
    category: str
    answer_mode: str
    decision_tier: str
    n_results: int
    filter_scores: List[FilterScore]
    compliance_results: List[ComplianceResult]
    semantic_score: Optional[float]
    latency_ms: float
    error: Optional[str] = None

    @property
    def filter_precision(self) -> float:
        if not self.filter_scores:
            return 1.0
        return sum(1 for s in self.filter_scores if s.match) / len(self.filter_scores)

    @property
    def compliance_pass_rate(self) -> float:
        if not self.compliance_results:
            return 1.0
        passed = sum(1 for r in self.compliance_results if r.passed)
        return passed / len(self.compliance_results)


# ──────────────────────────────────────────────────────────────────────────────
# Config loader
# ──────────────────────────────────────────────────────────────────────────────

def _expand_env(value: str) -> str:
    return re.sub(
        r"\$\{([^}:]+)(?::-([^}]*))?\}",
        lambda m: os.environ.get(m.group(1), m.group(2) or ""),
        value,
    )


def load_eval_config(path: Path) -> Tuple[Dict[str, Any], List[EvalCase]]:
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh)
    cfg = {
        "endpoint": raw.get("endpoint", "http://localhost:8085/search"),
        "top_k": int(raw.get("top_k", 10)),
        "embed_model_path": _expand_env(raw.get("embed_model_path", "")),
        "semantic_similarity_threshold": float(raw.get("semantic_similarity_threshold", 0.35)),
        "min_results_expected": int(raw.get("min_results_expected", 1)),
        "concurrency": int(raw.get("concurrency", 20)),
    }
    cases: List[EvalCase] = []
    for item in raw.get("cases", []):
        checks = [
            ComplianceCheck(
                field=c["field"],
                op=c["op"],
                value=c["value"],
            )
            for c in item.get("compliance_checks", [])
        ]
        cases.append(EvalCase(
            case_id=item["id"],
            query=item["query"],
            category=item.get("category", "structured"),
            expected_filters=item.get("expected_filters", {}),
            compliance_checks=checks,
            expected_query_types=item.get("expected_query_type", []),
            semantic_topic_hints=item.get("semantic_topic_hints", []),
            notes=item.get("notes", ""),
        ))
    return cfg, cases


# ──────────────────────────────────────────────────────────────────────────────
# Search client
# ──────────────────────────────────────────────────────────────────────────────

async def call_search(
    endpoint: str,
    query: str,
    top_k: int,
    session_id: Optional[str] = None,
    max_retries: int = 5,
) -> Tuple[Dict[str, Any], float]:
    """Call /search with optional session isolation and 429 retry.

    Each eval case passes its own unique session_id so it lands in a fresh
    rate-limit bucket, independent of other concurrent cases.  On a 429 the
    call waits Retry-After seconds (returned by the service) then retries —
    this covers the unlikely case where two cases share a bucket.
    """
    data = urllib.parse.urlencode({"query": query, "top_k": top_k}).encode()
    headers: Dict[str, str] = {}
    if session_id:
        headers["X-Session-Id"] = session_id

    for attempt in range(max_retries):
        t0 = time.perf_counter()
        try:
            response = await asyncio.to_thread(_sync_post, endpoint, data, headers)
            latency_ms = (time.perf_counter() - t0) * 1000
            return response, latency_ms
        except _RateLimitedError as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            if attempt == max_retries - 1:
                raise
            wait = exc.retry_after + 1
            logger.warning(
                f"rate_limited session={session_id} attempt={attempt + 1}/{max_retries} "
                f"retry_after={wait}s"
            )
            await asyncio.sleep(wait)

    # unreachable — last attempt raises inside the loop
    raise RuntimeError("call_search: exceeded max_retries")  # pragma: no cover


class _RateLimitedError(Exception):
    def __init__(self, retry_after: int) -> None:
        super().__init__(f"HTTP 429 retry_after={retry_after}")
        self.retry_after = retry_after


_ALLOWED_SCHEMES = {"http", "https"}
# Allowlist: eval harness only ever calls internal search endpoints
_ALLOWED_HOSTS = re.compile(
    r"^(localhost|127\.\d+\.\d+\.\d+|::1|[\w.-]+\.internal|[\w.-]+\.local)$"
)
# Explicit denylist — checked before allowlist; blocks cloud metadata even if host matches *.internal
_SSRF_DENYLIST = {
    "metadata.google.internal",  # GCP instance metadata
    "metadata.goog",
    "169.254.169.254",           # AWS/Azure/DO instance metadata
    "169.254.170.2",             # AWS ECS credential endpoint
}


def _validate_endpoint_url(url: str) -> None:
    """Reject non-http(s) schemes and hosts outside the trusted allowlist.

    Prevents file:// / ftp:// exploitation (dynamic-urllib) and limits
    SSRF surface to known internal hosts only.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"Disallowed URL scheme {parsed.scheme!r}; only http/https allowed")
    host = (parsed.hostname or "").lower()
    if host in _SSRF_DENYLIST:
        raise ValueError(
            f"Endpoint host {host!r} not in trusted allowlist; "
            "eval harness may only target internal/localhost endpoints"
        )
    if not _ALLOWED_HOSTS.match(host):
        raise ValueError(
            f"Endpoint host {host!r} not in trusted allowlist; "
            "eval harness may only target internal/localhost endpoints"
        )


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def validate_loopback_api_base(url: str) -> str:
    """Strict SSRF guard for HTTP-triggered offline harness: loopback only.

    Rejects non-http(s), URL userinfo, and any non-loopback host (incl. metadata
    and ``*.internal``). Returns stripped base without trailing slash.
    """
    raw = (url or "").strip()
    if not raw:
        raise ValueError("api base URL must be non-empty")
    parsed = urllib.parse.urlparse(raw)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(
            f"Disallowed URL scheme {parsed.scheme!r}; only http/https allowed"
        )
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("credentials in api URL are not allowed")
    host = (parsed.hostname or "").lower()
    if host not in _LOOPBACK_HOSTS:
        raise ValueError(
            f"Endpoint host {host!r} must be loopback "
            "(127.0.0.1, localhost, or ::1)"
        )
    # Rebuild without trailing slash / path noise for --api base.
    netloc = parsed.netloc
    return f"{parsed.scheme}://{netloc}".rstrip("/")


def _sync_post(url: str, data: bytes, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    import httpx
    _validate_endpoint_url(url)
    r = httpx.post(url, content=data, headers=headers or {}, timeout=45.0)
    if r.status_code == 429:
        try:
            body = r.json()
        except Exception:
            body = {}
        retry_after = int(body.get("retry_after_seconds", r.headers.get("Retry-After", 60)))
        raise _RateLimitedError(retry_after)
    r.raise_for_status()
    return r.json()


# ──────────────────────────────────────────────────────────────────────────────
# Filter extraction scorer
# ──────────────────────────────────────────────────────────────────────────────

def _normalise_filter_value(v: Any) -> Any:
    if isinstance(v, list):
        return sorted(str(x).lower() for x in v)
    if isinstance(v, str):
        return v.lower()
    return v


def score_filter_extraction(
    expected: Dict[str, Any],
    identified: List[Dict[str, Any]],
) -> List[FilterScore]:
    extracted: Dict[str, Any] = {f["name"]: f["value"] for f in identified}
    scores: List[FilterScore] = []
    for slot, exp_val in expected.items():
        ext_val = extracted.get(slot)
        exp_norm = _normalise_filter_value(exp_val)
        ext_norm = _normalise_filter_value(ext_val) if ext_val is not None else None
        if isinstance(exp_norm, list) and isinstance(ext_norm, list):
            match = all(e in ext_norm for e in exp_norm)
        elif isinstance(exp_norm, list) and isinstance(ext_norm, str):
            # scalar extracted where list expected: check if scalar in expected list
            match = ext_norm in exp_norm
        elif isinstance(exp_norm, list) and isinstance(ext_norm, (int, float)):
            match = str(int(ext_norm)) in exp_norm or str(ext_norm) in exp_norm
        elif isinstance(exp_norm, (int, float)) and isinstance(ext_norm, (int, float)):
            match = math.isclose(float(exp_norm), float(ext_norm), rel_tol=0.05)
        elif isinstance(exp_norm, bool):
            match = ext_norm == exp_norm
        else:
            match = exp_norm == ext_norm
        scores.append(FilterScore(slot=slot, expected=exp_val, extracted=ext_val, match=match))
    return scores


# ──────────────────────────────────────────────────────────────────────────────
# Result compliance checker
# ──────────────────────────────────────────────────────────────────────────────

def _derive_sld(domain_name: str) -> str:
    parts = str(domain_name).lower().split(".")
    return parts[0] if parts else domain_name.lower()


def _derive_has_hyphen(domain_name: str) -> bool:
    return "-" in _derive_sld(domain_name)


def _derive_has_number(domain_name: str) -> bool:
    return any(c.isdigit() for c in _derive_sld(domain_name))


def _resolve_field(item: Dict[str, Any], field_name: str) -> Any:
    domain_name = item.get("domain_name", "")
    payload = item.get("payload") or {}
    if field_name == "sld":
        return _derive_sld(domain_name)
    if field_name == "result_count":
        return None
    direct = item.get(field_name)
    if direct is not None:
        return direct
    return payload.get(field_name)


def _check_one(
    item: Dict[str, Any],
    check: ComplianceCheck,
    n_results: int,
) -> ComplianceResult:
    domain_name = item.get("domain_name", "unknown")
    item_id = item.get("item_id", domain_name)

    if check.field == "result_count":
        actual = n_results
        passed = _apply_op(actual, check.op, check.value)
        return ComplianceResult(
            check=check, item_id="__global__", domain_name="",
            actual_value=actual, passed=passed,
            reason="" if passed else f"result_count={actual} {check.op} {check.value} failed",
        )

    actual = _resolve_field(item, check.field)

    if check.op == "eq_or_derive":
        if actual is None:
            if check.field == "has_hyphen":
                actual = _derive_has_hyphen(domain_name)
            elif check.field == "has_number":
                actual = _derive_has_number(domain_name)
        passed = actual == check.value
    elif check.op == "starts_with_any":
        sld = _derive_sld(domain_name)
        passed = any(sld.startswith(v) for v in check.value)
        actual = sld
    elif check.op == "ends_with_any":
        sld = _derive_sld(domain_name)
        passed = any(sld.endswith(v) for v in check.value)
        actual = sld
    elif check.op == "contains_any":
        sld = _derive_sld(domain_name)
        passed = any(v in sld for v in check.value)
        actual = sld
    elif check.op == "contains_all":
        sld = _derive_sld(domain_name)
        passed = all(v in sld for v in check.value)
        actual = sld
    else:
        passed = _apply_op(actual, check.op, check.value)

    reason = ""
    if not passed:
        reason = f"{check.field}={actual!r} failed op={check.op} expected={check.value!r}"
    return ComplianceResult(
        check=check, item_id=str(item_id), domain_name=domain_name,
        actual_value=actual, passed=passed, reason=reason,
    )


def _apply_op(actual: Any, op: str, expected: Any) -> bool:
    if actual is None:
        return op in ("gte_or_null", "lte_or_null")
    try:
        if op in ("lte", "lte_or_null"):
            return float(actual) <= float(expected)
        if op in ("gte", "gte_or_null"):
            return float(actual) >= float(expected)
        if op == "eq":
            return actual == expected
        if op == "in":
            return str(actual).lower() in [str(v).lower() for v in expected]
        if op == "not_eq":
            return actual != expected
    except (TypeError, ValueError):
        pass
    return False


def check_compliance(
    items: List[Dict[str, Any]],
    checks: List[ComplianceCheck],
    n_results: int,
) -> List[ComplianceResult]:
    results: List[ComplianceResult] = []
    for check in checks:
        if check.field == "result_count":
            results.append(_check_one({}, check, n_results))
            continue
        for item in items:
            results.append(_check_one(item, check, n_results))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Semantic relevance scorer - nomic-embed-text -> transformers fallback
# ──────────────────────────────────────────────────────────────────────────────

class SemanticScorer:
    """Score semantic relevance between query and domain names via nomic-embed-text (or transformers fallback)."""

    def __init__(self, embed_model_path: str) -> None:
        self._embedder = self._load_embedder(embed_model_path)

    @staticmethod
    def _load_embedder(path: str) -> Any:
        if not path or not os.path.isdir(path):
            return None
        try:
            from nomic import embed
            mdl = embed.FlagModel(path, query_instruction_for_retrieval="", cache_dir=None)
            logger.info(f"nomic_embed_model loaded for eval scorer path={path!r}")
            return mdl
        except Exception:
            pass
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(path, local_files_only=True)  # nosec B615 - local_files_only=True blocks any Hub network fetch; no revision-pinning risk
            mdl = AutoModel.from_pretrained(path, local_files_only=True).eval()  # nosec B615 - local_files_only=True blocks any Hub network fetch; no revision-pinning risk
            logger.info(f"transformers_model loaded for eval scorer path={path!r}")
            return (tok, mdl)
        except Exception as exc:
            logger.info(f"embed_model_load_skipped reason={exc}")
            return None

    def score_embed(
        self,
        query: str,
        domain_names: List[str],
    ) -> Optional[float]:
        if self._embedder is None or not domain_names:
            return None
        try:
            import numpy as np
            texts = [query] + domain_names
            if isinstance(self._embedder, tuple):
                tok, mdl = self._embedder
                import torch
                with torch.no_grad():
                    enc = tok(texts, padding=True, truncation=True, return_tensors="pt")
                    out = mdl(**enc)
                    vecs = out.last_hidden_state[:, 0, :].numpy()
            else:
                result = self._embedder.encode(texts)
                vecs = result['embeddings'] if isinstance(result, dict) else result

            def _norm(v):
                n = np.linalg.norm(v)
                return v / n if n > 0 else v

            q_vec = _norm(vecs[0])
            scores = [float(np.dot(q_vec, _norm(v))) for v in vecs[1:]]
            return float(np.mean(scores)) if scores else None
        except Exception as exc:
            logger.debug(f"embed_score_failed error={exc}")
            return None

    def score(
        self,
        query: str,
        domain_names: List[str],
        topic_hints: List[str],
    ) -> Optional[float]:
        return self.score_embed(query, domain_names)


# ──────────────────────────────────────────────────────────────────────────────
# Per-case evaluator
# ──────────────────────────────────────────────────────────────────────────────

async def evaluate_case(
    case: EvalCase,
    cfg: Dict[str, Any],
    scorer: SemanticScorer,
    session_id: Optional[str] = None,
) -> CaseResult:
    try:
        response, latency_ms = await call_search(
            cfg["endpoint"], case.query, cfg["top_k"], session_id=session_id
        )
    except Exception as exc:
        return CaseResult(
            case_id=case.case_id, query=case.query, category=case.category,
            answer_mode="", decision_tier="", n_results=0,
            filter_scores=[], compliance_results=[], semantic_score=None,
            latency_ms=0.0, error=str(exc),
        )

    qi = response.get("query_intelligence") or {}
    filters_block = qi.get("filters") or {}
    identified = filters_block.get("identified") or []
    results = response.get("ranked_results") or []
    answer_mode = str(response.get("answer_mode") or "")
    decision_tier = str(qi.get("decision_tier") or "")

    filter_scores = score_filter_extraction(case.expected_filters, identified)
    compliance_results = check_compliance(results, case.compliance_checks, len(results))

    domain_names = [r.get("domain_name", "") for r in results if r.get("domain_name")]
    semantic_score: Optional[float] = None
    if case.category == "semantic" and domain_names:
        semantic_score = await asyncio.to_thread(
            scorer.score, case.query, domain_names, case.semantic_topic_hints
        )

    return CaseResult(
        case_id=case.case_id, query=case.query, category=case.category,
        answer_mode=answer_mode, decision_tier=decision_tier,
        n_results=len(results),
        filter_scores=filter_scores,
        compliance_results=compliance_results,
        semantic_score=semantic_score,
        latency_ms=latency_ms,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Report printer
# ──────────────────────────────────────────────────────────────────────────────

def _status(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def print_report(results: List[CaseResult], threshold: float) -> None:
    total = len(results)
    errors = sum(1 for r in results if r.error)
    filter_pass = sum(
        1 for r in results if not r.error and r.filter_precision == 1.0
    )
    compliance_pass = sum(
        1 for r in results if not r.error and r.compliance_pass_rate == 1.0
    )

    print("\n" + "=" * 72)
    print("QUERY EVAL REPORT")
    print("=" * 72)
    print(f"Total cases : {total}  Errors: {errors}")
    print(f"Filter extraction full-match : {filter_pass}/{total - errors}")
    print(f"Result compliance full-pass  : {compliance_pass}/{total - errors}")
    print("=" * 72)

    for r in results:
        prefix = f"[{r.case_id.upper()}]"
        if r.error:
            print(f"\n{prefix} ERROR — {r.error}")
            continue

        f_prec = r.filter_precision
        c_rate = r.compliance_pass_rate
        overall = "PASS" if f_prec == 1.0 and c_rate == 1.0 else "FAIL"
        sem_str = f"  sem={r.semantic_score:.2f}" if r.semantic_score is not None else ""
        print(
            f"\n{prefix} [{overall}] mode={r.answer_mode} tier={r.decision_tier}"
            f"  results={r.n_results}  lat={r.latency_ms:.0f}ms"
            f"  filter={f_prec:.0%}  comply={c_rate:.0%}{sem_str}"
        )
        print(f"  query: {r.query}")

        for fs in r.filter_scores:
            tick = "  ok" if fs.match else "  MISS"
            print(f"  {tick} filter[{fs.slot}] expected={fs.expected!r} got={fs.extracted!r}")

        fails = [cr for cr in r.compliance_results if not cr.passed]
        if fails:
            for cr in fails[:5]:
                label = f"{cr.domain_name}({cr.item_id})" if cr.domain_name else cr.item_id
                print(f"  FAIL comply: {label} — {cr.reason}")
        else:
            print("  all compliance checks passed")

    print("\n" + "=" * 72)


# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────

async def run_eval(
    cases_path: Optional[Path] = None,
    concurrency: Optional[int] = None,
) -> List[CaseResult]:
    """Run all eval cases in parallel, bounded by ``concurrency`` workers.

    Each case gets a unique X-Session-Id so it occupies its own rate-limit
    bucket.  The semaphore limits simultaneous in-flight HTTP requests to
    avoid overwhelming the service; LLM calls are still gated by the
    service-side max_concurrent_l2 semaphore and will queue naturally.
    """
    path = cases_path or _DEFAULT_CASES_PATH
    cfg, cases = load_eval_config(path)
    workers = concurrency if concurrency is not None else cfg["concurrency"]

    scorer = SemanticScorer(embed_model_path=cfg["embed_model_path"])

    sem = asyncio.Semaphore(workers)
    run_id = uuid.uuid4().hex[:8]

    async def _bounded(idx: int, case: EvalCase) -> CaseResult:
        session_id = f"eval-{run_id}-{idx}"
        async with sem:
            logger.info(f"evaluating [{idx + 1}/{len(cases)}] {case.case_id}: {case.query[:60]}")
            return await evaluate_case(case, cfg, scorer, session_id=session_id)

    results: List[CaseResult] = await asyncio.gather(
        *[_bounded(i, case) for i, case in enumerate(cases)]
    )

    # Restore original case order (gather preserves order, but be explicit)
    print_report(list(results), cfg["semantic_similarity_threshold"])
    return list(results)


if __name__ == "__main__":
    _path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    _concurrency = int(sys.argv[2]) if len(sys.argv) > 2 else None
    asyncio.run(run_eval(_path, _concurrency))
