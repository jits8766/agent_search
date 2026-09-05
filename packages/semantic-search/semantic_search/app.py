"""Agent Search FastAPI service.
Docs: /docs, /redoc, /openapi.json.
Data Build: POST /data-build/seed (Qdrant), /data-build/analytics-backfill (ClickHouse), /data-build/full (parallel).
Search: POST /search.
Status: GET /healthz, /capabilities, /data-build/status, /cache/stats, /resilience/health, /measurement/observations, /measurement/qie_only.
Freshness: DeltaRefreshDriver syncs mutable fields every 30s; Kinesis cutover is config-only.
Storage: Qdrant for search, ClickHouse for analytics; all limits are config-driven.
Telemetry: 9 write-only signals (cache_miss_storm, llm_refused, llm_timeout,
breaker_transition, eranker_applied, chip_demoted, uat_feedback,
analytics_failure, retrieved_content_sanitized) are recorded to
signals_platform_cln.feedback_signals (TTL 90 days) and the JSONL at
feedback.signal_log_path; available for future dashboarding, alerting, and
offline analysis.
Schedule: vectorization.seed.schedule; GET /data-build/status exposes effective
values; in-process loop runs when schedule.enabled and schedule.in_process.
Run: uvicorn semantic_search.app:app --host 0.0.0.0 --port 8085 --reload.
"""

import asyncio, copy, csv, hmac, io, json, math, os, re, sys, time, uuid, httpx
from collections.abc import AsyncIterator
from contextlib import aclosing, asynccontextmanager
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[assignment, misc]

from fastapi import FastAPI, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, RedirectResponse, Response

from llm_core.logging_utils import mask_path

from semantic_search.cache.keys import exact_query_key, versioned_query_key
from semantic_search.config.loader import load_config
from semantic_search.config.models import AgentSearchConfig
from semantic_search.contracts import (
    ERankerOutcome,
    FeedbackSignal,
    IntentSlice,
    QueryIntent,
    RankedResults,
    UserContext,
    ZeroResultGuardOutcome,
)
from semantic_search.analytics.clickhouse_client import (
    ClickHouseQueryError,
    ClickHouseUnavailableError,
)
from semantic_search.core.exceptions import (
    AgentSearchError,
    ConfigurationError,
    DataIngestInterruptedError,
    LLMError,
    QdrantQueryError,
    QdrantUnavailableError,
    RetrievalError,
    ValidationError,
)
from semantic_search.identity import (
    ResolvedIdentity,
    mint_prefixed_id,
    normalize_client_id,
    resolve_feedback_search_id,
    resolve_identity,
)
from semantic_search.core.llm_client import (
    reset_request_cost_observer,
    set_request_cost_observer,
)
from semantic_search.core.llm_provider import LLMProvider
from semantic_search.qi.llm_classifier import QIClassificationResponse
from semantic_search.core.logging_utils import (
    apply_log_level_from_config,
    get_logger,
    quiet_llm_core_child_loggers_warning,
)
from semantic_search.middleware.rate_limit import (
    RateLimitMiddleware,
    SlidingWindowRateLimiter,
)
from semantic_search.qi.encoder import HashingEncoder
from semantic_search.qi.advisory_patterns import (
    INVENTORY_BOUND_RE,
    is_soft_advisory_no_inventory,
    is_strong_advisory,
)
from semantic_search.qi.engine import normalize_query
from semantic_search.qi.l0_llm_filter_extractor import (
    FILTERABLE_PARAMS,
)
from semantic_search.qi.l0_regex_filter_extractor import L0RegexFilterExtractor
from semantic_search.qi.qie_only import (
    _attach_find_wire_fields,
    _clear_qie_l0_filter_cache,
    _get_qie_l0_filter_cache,
    _qie_cache_envelope,
    _qie_grounding_context,
    _qie_identified_from_intent,
    _qie_only_ops_metrics,
    _unpack_qie_cache_entry,
    reconcile_and_ground_identified,
)
from semantic_search.qi.slot_to_api_param import (
    get_api_param_for_slot,
    get_find_api_param_name,
    is_find_api_filter_slot,
    slot_keeps_false_value,
    transform_slot_value,
)

_FILTERABLE_PARAM_SET = frozenset(FILTERABLE_PARAMS)
from semantic_search.orchestrator import hybrid_degrade_intent  # noqa: E402
from semantic_search.registry import (  # noqa: E402
    Subsystems,
    build_subsystems,
    seed_boot_indexes,
    _resolve_seed_pages,
)
from semantic_search.retrieval.structured_retriever import (  # noqa: E402
    InMemoryStructuredIndex,
    _FILTER_ENTITY_NAMES,
)
from semantic_search.retrieval.qdrant_adapter import set_unavailable_filter_keys  # noqa: E402
from semantic_search.retrieval.vector_retriever import InMemoryVectorIndex  # noqa: E402
from semantic_search.retrieval.complement_rank_leanings import (  # noqa: E402
    apply_complement_rank_leanings,
)
from semantic_search.vectorization.boot_loader import (  # noqa: E402
    load_seed_into_indexes,
)
from semantic_search.vectorization.query_driven_expander import QueryDrivenExpander  # noqa: E402
from semantic_search.explore.ch_seed_writer import insert_seed_to_clickhouse  # noqa: E402
from semantic_search.nl_to_sql.athena_client import AthenaClient as _AthenaClient  # noqa: E402
from semantic_search.s3_feedback_uploader import (  # noqa: E402
    upload_feedback_csv,
    read_feedback_from_s3,
    _CSV_FIELDS as _FEEDBACK_CSV_FIELDS,
)
from semantic_search.s3_search_log_uploader import (  # noqa: E402
    is_katana_env,
    upload_search_result_json,
    ensure_search_logs_retention_policy,
)
from semantic_search.config.models import (  # noqa: E402
    SeedTableConfig as _SeedTableConfig,
    SeedDatabaseConfig as _SeedDatabaseConfig,
)
from semantic_search.vectorization.db_seed_source import (  # noqa: E402
    fetch_seed_pages_from_db,
    DbSeedSummary as _DbSeedSummary,
)
from semantic_search.vectorization.stage_timing import StageTimingSession  # noqa: E402

logger = get_logger(__name__)


def _load_dotenv_first() -> None:
    """Load .env (walk up from module -> CWD -> /app) before load_config()."""
    if load_dotenv is None:
        return
    candidates = [parent / ".env" for parent in Path(__file__).resolve().parents]
    candidates.extend((Path.cwd() / ".env", Path("/app/.env")))
    for path in candidates:
        if path.exists():
            load_dotenv(path, override=True)
            logger.info(f"dotenv_loaded path={mask_path(str(path))}")
            return


_load_dotenv_first()


_DOCS_PATHS = ("/docs", "/redoc", "/openapi.json")

_OPENAPI_TAGS = [
    {
        "name": "Data Build",
        "description": "Build and index the search corpus from source data.",
    },
    {
        "name": "Search",
        "description": "Unified search API for retrieval, analytics, ranking, and diagnostics.",
    },
    {
        "name": "Feedback",
        "description": "Collect and review UAT feedback signals. Nine write-only telemetry signals (cache_miss_storm, llm_refused, llm_timeout, breaker_transition, eranker_applied, chip_demoted, uat_feedback, analytics_failure, retrieved_content_sanitized) are also recorded here — persisted to ClickHouse and JSONL for future dashboarding and alerting.",  # noqa: E501
    },
    {
        "name": "Status",
        "description": "Health, metrics, cache, backend, and index observability.",
    },
]


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach security headers (docs use SAMEORIGIN for Swagger iframe)."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        try:
            response = await call_next(request)
        except RuntimeError as exc:
            # Starlette's BaseHTTPMiddleware raises RuntimeError("No response
            # returned.") when the inner request task is cancelled mid-flight
            # (work abandoned under load — e.g. ClickHouse stalls) and the
            # coroutine exits without emitting a response. Surface a graceful
            # 503 instead of a bare 500 so clients can retry the transient.
            if "No response returned" not in str(exc):
                raise
            logger.warning(f"request_cancelled_no_response path={request.url.path}")
            response = JSONResponse(
                status_code=503,
                content={"detail": "request_cancelled: server busy under load, retry"},
            )
        is_docs = any(request.url.path.startswith(p) for p in _DOCS_PATHS)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
        response.headers["X-Frame-Options"] = "SAMEORIGIN" if is_docs else "DENY"
        if not is_docs:
            response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response


class AppState:
    """Runtime container for initialized subsystems."""

    def __init__(self):
        self.subsystems: Optional[Subsystems] = None
        self._build_history: List[Dict[str, Any]] = []
        self._last_build_params: Optional[Dict[str, Any]] = None
        self._seed_task: Optional[asyncio.Task] = None
        self._calibration_task: Optional["asyncio.Future"] = None
        self._synonym_expander: Optional[QueryDrivenExpander] = None
        self._synonym_expansion_task: Optional[asyncio.Task] = None
        self._synonym_shutdown_grace_s: float = 5.0
        self._enriched_builder_task: Optional[asyncio.Task] = None
        # Enrichment columns absent from last data-build (skipped to prevent zero-result queries).
        self._missing_data_columns: Set[str] = set()
        self._last_ch_seed_result: Optional[Dict[str, Any]] = None
        # Live progress for in-flight /data-build (stage + counts) so ALB/cancel can report how far.
        self._ingest_progress: Optional[Dict[str, Any]] = None
        # True when FastEmbedEncoder failed and service uses HashingEncoder (degraded L1 recall).
        self.encoder_degraded: bool = False


app_state = AppState()

# Serialize concurrent calls to the same data-build endpoint — each endpoint's
# run_token (see below) namespaces its seed_merge tables in the shared
# scratch database (vectorization.seed.database.merge_database, override via
# ATHENA_TMP_DB), so two in-flight calls to the same endpoint would otherwise
# race on the identical stg_*/final_* table names. Also held across
# _clear_for_rebuild (see data_build_seed) so a second overlapping call can't
# delete/recreate the Qdrant collection while a first call is still upserting
# into it.
_SEED_BUILD_LOCK = asyncio.Lock()
_BACKFILL_BUILD_LOCK = asyncio.Lock()


def _require_subsystems() -> Subsystems:
    """Return initialized subsystems or raise 503."""
    if app_state.subsystems is None:
        raise HTTPException(
            status_code=503, detail="semantic_search subsystems not initialized"
        )
    return app_state.subsystems


def _is_analytics_primary_query(
    *,
    query_type: Optional[str],
    l1_type: Optional[str],
    use_analytics_budget: bool,
) -> bool:
    """True when request is analytics-primary (ClickHouse owns the answer).

    Listing / hybrid / explore / guidance are Qdrant-primary for ranked_results.
    """
    if query_type == "analytics" or l1_type == "analytics":
        return True
    return bool(use_analytics_budget)


def _backend_matches_query_primary(
    exc: BaseException, *, analytics_primary: bool
) -> bool:
    """True when the exception backend matches the query's primary store.

    - analytics-primary -> ClickHouse only (ignore Qdrant noise)
    - listing/hybrid-primary -> Qdrant only (ignore ClickHouse noise)
    """
    backend, _mode, _human = _classify_backend_unavailable(exc)
    if analytics_primary:
        return backend == "clickhouse"
    return backend == "qdrant"


def _classify_backend_unavailable(exc: BaseException) -> tuple[str, str, str]:
    """Map a backend exception to (backend, failure_mode, human_message).

    ``human_message`` states what failed and what to check — not a generic
    internal error. Callers stamp ``request_id`` into the response separately.
    """
    err_text = str(exc) or type(exc).__name__
    err_l = err_text.lower()
    if (
        isinstance(exc, (ClickHouseUnavailableError, ClickHouseQueryError))
        or "clickhouse" in err_l
    ):
        return (
            "clickhouse",
            "clickhouse_unavailable",
            (
                f"ClickHouse is unavailable: {err_text}. "
                "Analytics / explore rails / guidance that need CH cannot run. "
                "Verify ClickHouse host/port/credentials and that the service is up."
            ),
        )
    if isinstance(exc, (QdrantUnavailableError, QdrantQueryError)) or "qdrant" in err_l:
        if "doesn't exist" in err_l or ("not found" in err_l and "collection" in err_l):
            hint = "Qdrant collection is missing - run /data-build/seed (or full data-build)."
        elif (
            "connection refused" in err_l
            or "failed to connect" in err_l
            or "statuscode.unavailable" in err_l
        ):
            hint = "Qdrant is not reachable (connection refused) - start Qdrant or fix host/port."
        else:
            hint = "Qdrant query failed - check collection health and Qdrant logs."
        return (
            "qdrant",
            "qdrant_unavailable",
            f"Qdrant is unavailable: {err_text}. {hint}",
        )
    return (
        "retrieval",
        "qdrant_unavailable",
        f"Retrieval backend unavailable: {err_text}. Check Qdrant and ClickHouse.",
    )


def _apply_identity_fields(
    payload: Dict[str, Any], *, request_id: str, search_id: str
) -> Dict[str, Any]:
    """Stamp request_id (trace) and search_id (durable) onto a search/feedback JSON body."""
    payload["request_id"] = request_id
    payload["search_id"] = search_id
    return payload


def _backend_unavailable_envelope(
    *,
    query: str,
    request_id: str,
    search_id: str,
    exc: BaseException,
    latency_ms: float,
    answer_mode: str = "search",
    top_k: int = 50,
    diversity_lambda: float = 0.9,
    relevance_threshold: float = 0.7,
    decision_cost_usd: float = 0.0,
) -> Dict[str, Any]:
    """Search-shaped JSON body for backend-down paths (paired with HTTP 503)."""
    backend, failure_mode, human = _classify_backend_unavailable(exc)
    notice = f"{human} request_id={request_id} search_id={search_id}"
    body = {
        "query": query,
        "answer_mode": answer_mode,
        "latency_ms": round(float(latency_ms), 1),
        "query_intelligence": {"decision_cost_usd": round(float(decision_cost_usd), 4)},
        "pipeline_trace": {"applied_filters": []},
        "ranked_results": [],
        "retrieval_metrics": {
            "failure_mode": failure_mode,
            "total_candidates": 0,
            "metrics_valid": False,
            "backends_active": [],
            "params": {
                "top_k": int(top_k),
                "diversity_lambda": float(diversity_lambda),
                "relevance_threshold": float(relevance_threshold),
            },
            "error": {
                "backend": backend,
                "error_type": type(exc).__name__,
                "message": str(exc) or type(exc).__name__,
                "request_id": request_id,
                "search_id": search_id,
            },
        },
        "analytics": {},
        "guidance": {},
        "guard_notice": notice,
    }
    return _apply_identity_fields(body, request_id=request_id, search_id=search_id)


def _backend_unavailable_response(
    *,
    query: str,
    request_id: str,
    search_id: str,
    exc: BaseException,
    latency_ms: float,
    answer_mode: str = "search",
    top_k: int = 50,
    diversity_lambda: float = 0.9,
    relevance_threshold: float = 0.7,
    decision_cost_usd: float = 0.0,
) -> JSONResponse:
    """HTTP 503 + clear JSON (never opaque Internal Server Error)."""
    body = _backend_unavailable_envelope(
        query=query,
        request_id=request_id,
        search_id=search_id,
        exc=exc,
        latency_ms=latency_ms,
        answer_mode=answer_mode,
        top_k=top_k,
        diversity_lambda=diversity_lambda,
        relevance_threshold=relevance_threshold,
        decision_cost_usd=decision_cost_usd,
    )
    backend = body["retrieval_metrics"]["error"]["backend"]
    logger.error(
        f"backend_unavailable backend={backend} request_id={request_id} "
        f"search_id={search_id} error_type={type(exc).__name__} error={exc}"
    )
    return JSONResponse(status_code=503, content=body)


def _set_ingest_progress(
    stage: str, records_completed: int = 0, records_attempted: int = 0
) -> None:
    """Stamp live ingest progress for cancel/ALB interrupt reporting."""
    app_state._ingest_progress = {
        "stage": str(stage),
        "records_completed": int(records_completed),
        "records_attempted": int(records_attempted),
    }


def _data_ingest_interrupted_envelope(
    *,
    endpoint: str,
    exc: BaseException,
    started_at: str = "",
    partial: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """JSON body when data-build stops mid-ingest (ALB idle timeout, client disconnect, cancel)."""
    if isinstance(exc, DataIngestInterruptedError):
        stage = exc.stage
        completed = exc.records_completed
        attempted = exc.records_attempted
        reason = exc.reason
        detail = exc.detail or f"{type(exc).__name__}: {exc}"
        message = str(exc) or detail
    else:
        prog = app_state._ingest_progress or {}
        stage = str(prog.get("stage") or "unknown")
        completed = int(prog.get("records_completed") or 0)
        attempted = int(prog.get("records_attempted") or 0)
        reason = "cancelled" if isinstance(exc, asyncio.CancelledError) else "error"
        detail = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        message = (
            f"Data ingest interrupted at stage={stage} after {completed} of {attempted} records. "
            "Often ALB idle timeout or client disconnect."
        )
    body: Dict[str, Any] = {
        "status": "interrupted",
        "endpoint": endpoint,
        "started_at": started_at,
        "stage": stage,
        "records_completed": completed,
        "records_attempted": attempted,
        "reason": reason,
        "detail": detail,
        "message": message,
        "hint": (
            "Ingest stopped before completion. Partial data may remain in Qdrant/ClickHouse. "
            "Common cause: ALB idle timeout (raise idle_timeout) or client/proxy disconnect. "
            "Re-run the same /data-build endpoint to resume (append/rebuild as needed)."
        ),
    }
    if partial:
        body["partial"] = partial
    return body


def _data_ingest_interrupted_response(
    *,
    endpoint: str,
    exc: BaseException,
    started_at: str = "",
    partial: Optional[Dict[str, Any]] = None,
    history_extra: Optional[Dict[str, Any]] = None,
) -> JSONResponse:
    """HTTP 503 + clear interrupt JSON (records completed + why)."""
    body = _data_ingest_interrupted_envelope(
        endpoint=endpoint,
        exc=exc,
        started_at=started_at,
        partial=partial,
    )
    hist = {
        "build_id": str(uuid.uuid4())[:8],
        "triggered_by": "manual",
        "endpoint": endpoint,
        "started_at": started_at,
        "stage": body["stage"],
        "records_completed": body["records_completed"],
        "records_attempted": body["records_attempted"],
        "reason": body["reason"],
        "detail": body["detail"],
        "status": "interrupted",
    }
    if history_extra:
        hist.update(history_extra)
    app_state._build_history.append(hist)
    logger.error(
        f"data_ingest_interrupted endpoint={endpoint} stage={body['stage']} "
        f"records_completed={body['records_completed']} records_attempted={body['records_attempted']} "
        f"reason={body['reason']} detail={body['detail']}"
    )
    return JSONResponse(status_code=503, content=body)


async def _ping_qdrant(
    host: str,
    port: int,
    timeout: float = 2.0,  # noqa: ASYNC109
    secure: bool = False,
) -> bool:
    """Return True iff Qdrant responds on host:port within timeout seconds.

    ``secure`` selects https — required when Qdrant is fronted by a TLS listener
    (Katana/SSG serves it on 443 HTTPS); a plain-http probe to a TLS port fails.
    """
    scheme = "https" if secure else "http"
    try:
        async with httpx.AsyncClient(
            base_url=f"{scheme}://{host}:{port}", timeout=timeout
        ) as _c:
            r = await _c.get("/healthz")
            return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


async def _ping_clickhouse(
    host: str,
    port: int,
    timeout: float = 2.0,  # noqa: ASYNC109
    secure: bool = False,
) -> bool:
    """Return True iff ClickHouse HTTP interface responds on host:port within timeout seconds.

    ``secure`` selects https — required when ClickHouse is fronted by a TLS listener
    (Katana/SSG serves it on 443 HTTPS); a plain-http probe to a TLS port fails.
    """
    scheme = "https" if secure else "http"
    try:
        async with httpx.AsyncClient(
            base_url=f"{scheme}://{host}:{port}", timeout=timeout
        ) as _c:
            r = await _c.get("/ping")
            return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


async def _dependency_health_check(sub: "Subsystems") -> Dict[str, Any]:
    """Probe Qdrant/ClickHouse/LLM on startup. Failures logged but don't abort (degraded mode continues)."""
    status: Dict[str, Any] = {}

    # Qdrant probe
    qdrant_needed = sub.qdrant_factory is not None and sub.qdrant_factory.available
    if qdrant_needed:
        qdrant_cfg = sub.qdrant_factory.config  # type: ignore[union-attr]
        host = qdrant_cfg.host
        port = int(qdrant_cfg.port)
        reachable = await _ping_qdrant(host, port, secure=bool(qdrant_cfg.https))
        if reachable:
            logger.info(f"qdrant_reachable host={host} port={port}")
        else:
            logger.warning(
                f"qdrant_unavailable host={host} port={port} — service expected to be running externally (Katana-managed when deployed, standalone container locally via scripts/deploy_qdrant.sh)"
            )  # noqa: E501
        status["qdrant"] = "ok" if reachable else "unavailable"
    else:
        status["qdrant"] = "not_configured"

    # ClickHouse probe
    ch_analytics_cfg = (
        sub.config.nl_to_sql.analytics if sub.config.nl_to_sql is not None else None
    )
    ch_needed = ch_analytics_cfg is not None and ch_analytics_cfg.enabled
    if ch_needed:
        ch_client_cfg = ch_analytics_cfg.clickhouse  # type: ignore[union-attr]
        ch_host = ch_client_cfg.host
        ch_port = int(ch_client_cfg.port)
        ch_reachable = await _ping_clickhouse(
            ch_host, ch_port, secure=bool(ch_client_cfg.secure)
        )
        if ch_reachable:
            logger.info(f"clickhouse_reachable host={ch_host} port={ch_port}")
        else:
            logger.warning(
                f"clickhouse_unavailable host={ch_host} port={ch_port} — service expected to be running externally (Katana-managed when deployed, standalone container locally via scripts/deploy_clickhouse.sh)"  # noqa: E501
            )
        status["clickhouse"] = "ok" if ch_reachable else "unavailable"
    else:
        status["clickhouse"] = "not_configured"

    # LLM provider probe
    llm_ok = sub.llm_provider is not None
    if not llm_ok:
        logger.warning(
            "llm_provider_unavailable hint='set LLM API key env vars or LLM_BASE_URL for gateway'"
        )
    status["llm"] = "ok" if llm_ok else "unavailable"

    return status


def _validate_pretrained_models(config_dict: Dict[str, Any]) -> Dict[str, str]:
    """Validate pretrained models from LOCAL_PRETRAINED_DIR. Missing models logged; service boots in degraded mode."""
    status: Dict[str, str] = {}
    pretrained_root = os.environ.get("LOCAL_PRETRAINED_DIR")
    _root_display = (
        mask_path(pretrained_root)
        if pretrained_root
        else "(unset — base.yaml default /app/pretrained)"
    )
    logger.info(f"pretrained_dir_resolved root={_root_display!r}")

    enc_path = os.path.expanduser(
        str(config_dict.get("qi", {}).get("encoder", {}).get("local_model_path", ""))
    )
    if not enc_path:
        status["encoder"] = "not_configured"
    elif os.path.isdir(enc_path):
        status["encoder"] = "ok"
    else:
        status["encoder"] = "missing"
        logger.warning(
            f"embedding_model_not_found path={mask_path(enc_path)!r} "
            f"— service will start with HashingEncoder (degraded semantic quality); "
            f"pre-download the model to {mask_path(enc_path)!r} and restart to restore full recall"
        )

    sparse_cfg = (
        config_dict.get("retrieval", {})
        .get("qdrant", {})
        .get("hybrid", {})
        .get("sparse_encoder", {})
        or {}
    )
    sparse_path = os.path.expanduser(str(sparse_cfg.get("local_model_path", "") or ""))
    # Explicit false only — missing key keeps legacy warn+hash-BM25 path at registry.
    hash_bm25_fallback = sparse_cfg.get("hash_bm25_fallback")
    if not sparse_path:
        status["sparse"] = "not_configured"
    elif os.path.isdir(sparse_path):
        status["sparse"] = "ok"
    else:
        status["sparse"] = "missing"
        if hash_bm25_fallback is False:
            raise ConfigurationError(
                f"sparse_model_not_found path={mask_path(sparse_path)!r} "
                f"hash_bm25_fallback=false — set LOCAL_PRETRAINED_DIR to the pretrained "
                f"root that contains Qdrant/all_miniLM_L6_v2_with_attentions "
                f"(current LOCAL_PRETRAINED_DIR={mask_path(pretrained_root)!r}), "
                f"or set retrieval.qdrant.hybrid.sparse_encoder.hash_bm25_fallback=true"
            )
        logger.warning(
            f"sparse_model_not_found path={mask_path(sparse_path)!r} "
            f"hash_bm25_fallback={hash_bm25_fallback!r} — service will start with "
            f"hash-BM25 sparse encoder (reduced sparse precision); pre-download the "
            f"model to {mask_path(sparse_path)!r} and restart to use BM42"
        )

    _ng_cfg = config_dict.get("qi", {}).get("regex", {}).get("ngram_pre_gate", {})
    _ng_enabled = bool(_ng_cfg.get("enabled", False)) if _ng_cfg else False
    _ng_raw_path = str(_ng_cfg.get("model_path", "")) if _ng_cfg else ""
    if not _ng_enabled or not _ng_raw_path:
        status["ngram_pre_gate"] = "not_configured"
    else:
        _ng_path = Path(_ng_raw_path)
        if not _ng_path.is_absolute():
            import importlib.util  # noqa: PLC0415

            _ng_module = importlib.util.find_spec("semantic_search.qi.ngram_pre_gate")
            if _ng_module and _ng_module.origin:
                _ng_project_root = Path(_ng_module.origin).resolve().parents[2]
                _ng_path = (_ng_project_root / _ng_path).resolve()
        if _ng_path.is_file():
            status["ngram_pre_gate"] = "ok"
            logger.info(
                f"ngram_pre_gate_weights_found path={mask_path(str(_ng_path))!r}"
            )
        else:
            status["ngram_pre_gate"] = "missing"
            logger.warning(
                f"ngram_pre_gate_weights_not_found path={mask_path(str(_ng_path))!r} "
                f"— QI pre-gate disabled; train weights with: "
                f"python -m semantic_search.qi.training.ngram_trainer "
                f"--input .pretrained/ngram_pregate/training_data.jsonl "
                f"--output .pretrained/ngram_pregate/weights.json "
                f"--top-k 200 --min-log-odds 0.3 --smoothing 1.0 --min-examples 50 --max-ngram-order 2"
            )

    _sh_cfg = config_dict.get("qi", {}).get("semantic", {}).get("learned_head", {})
    _sh_kind = str(_sh_cfg.get("kind", "centroid")) if _sh_cfg else "centroid"
    _sh_raw_path = str(_sh_cfg.get("model_path", "")) if _sh_cfg else ""
    if _sh_kind == "centroid" or not _sh_raw_path:
        status["semantic_head"] = "not_configured"
    else:
        _pretrained_root = os.environ.get("LOCAL_PRETRAINED_DIR", "/app/pretrained")
        _sh_expanded = os.path.expandvars(
            _sh_raw_path.replace(
                "${LOCAL_PRETRAINED_DIR:-/app/pretrained}", _pretrained_root
            ).replace("${LOCAL_PRETRAINED_DIR}", _pretrained_root)
        )
        _sh_path = Path(os.path.expanduser(_sh_expanded))
        if _sh_path.is_file():
            status["semantic_head"] = "ok"
            logger.info(
                f"semantic_head_found kind={_sh_kind!r} path={mask_path(str(_sh_path))!r}"
            )
        else:
            status["semantic_head"] = "missing"
            logger.warning(
                f"semantic_head_missing kind={_sh_kind!r} path={mask_path(str(_sh_path))!r} "
                f"— L1 routing falls back to centroid scorer; train and deploy the head with: "
                f"python -m semantic_search.qi.training.deep_head_trainer (for deep) or "
                f"python -m semantic_search.qi.training.head_trainer --classifier best_sklearn"
            )

    _line = "  ".join(f"{k}={v}" for k, v in status.items())
    _missing = [k for k, v in status.items() if v == "missing"]
    if _missing:
        logger.warning(f"pretrained_models_degraded missing={_missing} {_line}")
    else:
        logger.info(f"pretrained_models_ok {_line}")
    return status


_HYBRID_PREWARM_QUERIES: List[str] = [
    # Semantic + price filter — canonical hybrid route (Qdrant vector + sparse fetch).
    "b2b saas domain under 3k some traffic",
    "clean four letter .com domain under 2k",
    # Misspelled residual — exercises char-ngram fuzzy Prefetch leg via E2E search.
    "hiigh rentalls brandable domain",
]

# Direct HybridRetriever.retrieve intents (bypass QI routing). Forces dense + BM42
# + ngram Prefetch/RRF even if orchestrator E2E routes oddly.
_HYBRID_SPARSE_LEG_PREWARM: List[Tuple[str, str]] = [
    ("saas marketplace brandable", "bm42"),
    ("hiigh rentalls", "ngram_fuzzy"),
]

_HYBRID_PREWARM_CONCURRENCY = 4
_HYBRID_PREWARM_TIMEOUT_S = 200.0

# Pre-warm input for the DistilBERT query rewriter. The input is long
# (> rewrite_threshold) so it exercises the rewrite path, warming the forward
# graph / tokenizer caches before the first live query.
_QT_PREWARM_REWRITE = "i want to open a small coffee shop in delhi and need a short brandable domain under fifty dollars please"


async def _prewarm_query_transformer(sub: Subsystems) -> None:
    """Warm the resident query rewriter so the first live query skips kernel/JIT init.

    Runs one rewrite input through the in-memory model.
    Soft-fails: a warm-up error is logged and never blocks startup.
    """
    qt = getattr(getattr(sub, "orchestrator", None), "_query_transformer", None)
    if qt is None:
        logger.info("prewarm_query_transformer_skipped reason=transformer_disabled")
        return
    t0 = time.monotonic()
    try:
        r_rewrite = await qt.transform(_QT_PREWARM_REWRITE)
        logger.info(
            f"prewarm_query_transformer_complete rewrite_mode={r_rewrite.mode} "
            f"elapsed_ms={(time.monotonic() - t0) * 1000.0:.1f}"
        )
    except Exception as _qt_e:  # noqa: BLE001 — warm-up must never abort boot
        logger.warning(
            f"prewarm_query_transformer_failed error_type={type(_qt_e).__name__} error={_qt_e}"
        )


def _prewarm_hybrid_intent(encode_text: str, request_id: str) -> QueryIntent:
    """Build a hybrid intent with non-empty semantic_encode_text for ANN+sparse legs."""
    return QueryIntent(
        request_id=request_id,
        raw_query=encode_text,
        normalized_query=encode_text,
        query_type="hybrid",
        confidence=1.0,
        decision_tier="prewarm",
        slices=[
            IntentSlice(
                query_type="hybrid",
                entities=[],
                confidence=1.0,
                raw_text=encode_text,
            )
        ],
        decision_cost_usd=0.0,
        semantic_encode_text=encode_text,
        semantic_query=encode_text,
        residual_kind="semantic",
    )


async def _prewarm_hybrid_queries(sub: Subsystems, qdrant_available: bool) -> None:
    """Background pre-warm: hybrid E2E + direct dense/BM42/ngram Prefetch path."""
    if not qdrant_available:
        logger.warning("prewarm_hybrid_skipped reason=qdrant_unavailable")
        return
    if getattr(sub, "orchestrator", None) is None:
        logger.warning("prewarm_hybrid_skipped reason=orchestrator_unavailable")
        return

    vr = getattr(sub, "vector_retriever", None)
    bm25_on = bool(getattr(vr, "_bm25_query_fn", None))
    ngram_on = bool(
        getattr(vr, "_ngram_enabled", False) and getattr(vr, "_ngram_query_fn", None)
    )
    logger.info(f"prewarm_hybrid_legs_ready bm25={bm25_on} ngram={ngram_on}")
    if not bm25_on or not ngram_on:
        logger.warning(
            f"prewarm_hybrid_sparse_incomplete bm25={bm25_on} ngram={ngram_on} "
            f"— sparse Prefetch warm may be partial"
        )

    sem = asyncio.Semaphore(_HYBRID_PREWARM_CONCURRENCY)
    # Serialize the LLM-bound e2e prewarm calls: running them concurrently makes
    # their L0-extraction + L2-classify calls compete for the same model capacity,
    # which can push a later query's classify_timeout_seconds (4s) past its budget.
    llm_sem = asyncio.Semaphore(1)
    failed = 0

    async def _run_e2e(query: str, idx: int) -> None:
        nonlocal failed
        async with sem, llm_sem:
            try:
                _res_tuple = await asyncio.wait_for(
                    sub.orchestrator.search(
                        raw_query=query,
                        request_id=f"prewarm_{idx:03d}",
                        user_context=None,
                        top_k=5,
                        diversity_lambda=0.5,
                    ),
                    timeout=_HYBRID_PREWARM_TIMEOUT_S,
                )
                _routed_as = getattr(
                    getattr(_res_tuple[0], "query_intent", None), "query_type", None
                )
                if _routed_as and _routed_as != "hybrid":
                    logger.warning(
                        f"prewarm_hybrid_wrong_route idx={idx} routed_as={_routed_as} query={query!r}"
                    )
            except asyncio.TimeoutError:
                failed += 1
                logger.warning(
                    f"prewarm_hybrid_query_failed idx={idx} reason=timeout query={query!r}"
                )
            except Exception as _pw_e:  # noqa: BLE001
                failed += 1
                logger.warning(
                    f"prewarm_hybrid_query_failed idx={idx} reason={type(_pw_e).__name__} error={_pw_e} query={query!r}"
                )

    async def _run_sparse_leg(encode_text: str, tag: str, idx: int) -> None:
        """Direct HybridRetriever.retrieve — must hit BM42 + ngram Prefetch when wired."""
        nonlocal failed
        if vr is None or not hasattr(vr, "retrieve"):
            return
        async with sem:
            try:
                intent = _prewarm_hybrid_intent(
                    encode_text, f"prewarm_sparse_{tag}_{idx:03d}"
                )
                cs = await asyncio.wait_for(
                    vr.retrieve(intent, top_k=5),
                    timeout=_HYBRID_PREWARM_TIMEOUT_S,
                )
                logger.info(
                    f"prewarm_hybrid_sparse_leg tag={tag} candidates={len(cs.candidates)} "
                    f"latency_ms={cs.latency_ms:.1f} bm25={bm25_on} ngram={ngram_on}"
                )
            except asyncio.TimeoutError:
                failed += 1
                logger.warning(
                    f"prewarm_hybrid_sparse_leg_failed tag={tag} reason=timeout"
                )
            except Exception as _pw_e:  # noqa: BLE001
                failed += 1
                logger.warning(
                    f"prewarm_hybrid_sparse_leg_failed tag={tag} "
                    f"reason={type(_pw_e).__name__} error={_pw_e}"
                )

    total = len(_HYBRID_PREWARM_QUERIES)
    logger.info(
        f"prewarm_hybrid_start query_count={total} concurrency={_HYBRID_PREWARM_CONCURRENCY}"
    )
    _t0 = time.monotonic()
    e2e_tasks = [_run_e2e(q, i) for i, q in enumerate(_HYBRID_PREWARM_QUERIES)]
    sparse_tasks = [
        _run_sparse_leg(text, tag, i)
        for i, (text, tag) in enumerate(_HYBRID_SPARSE_LEG_PREWARM)
    ]
    await asyncio.gather(*(e2e_tasks + sparse_tasks))
    _elapsed_ms = round((time.monotonic() - _t0) * 1000, 1)
    logger.info(
        f"prewarm_hybrid_complete total={total} sparse_legs={len(_HYBRID_SPARSE_LEG_PREWARM)} "
        f"succeeded={total + len(_HYBRID_SPARSE_LEG_PREWARM) - failed} failed={failed} "
        f"bm25={bm25_on} ngram={ngram_on} elapsed_ms={_elapsed_ms}"
    )


_ANALYTICS_PREWARM_QUERIES: List[str] = [
    # COUNT / AVG aggregates — canonical analytics route (ClickHouse MV fetch).
    "total active auction count right now",
    "average current bid on .io domains",
]

_ANALYTICS_PREWARM_CONCURRENCY = 4
_ANALYTICS_PREWARM_TIMEOUT_S = 300.0


async def _prewarm_analytics_queries(
    sub: Subsystems, clickhouse_available: bool
) -> None:
    """Background pre-warm: run analytics queries to seed ClickHouse connection pool and MV query cache."""
    if not clickhouse_available:
        logger.warning("prewarm_analytics_skipped reason=clickhouse_unavailable")
        return
    if getattr(sub, "orchestrator", None) is None:
        logger.warning("prewarm_analytics_skipped reason=orchestrator_unavailable")
        return

    sem = asyncio.Semaphore(_ANALYTICS_PREWARM_CONCURRENCY)
    failed = 0

    async def _run_one(query: str, idx: int) -> None:
        nonlocal failed
        async with sem:
            try:
                _res_tuple = await asyncio.wait_for(
                    sub.orchestrator.search(
                        raw_query=query,
                        request_id=f"prewarm_analytics_{idx:03d}",
                        user_context=None,
                        top_k=5,
                        diversity_lambda=0.5,
                    ),
                    timeout=_ANALYTICS_PREWARM_TIMEOUT_S,
                )
                _routed_as = getattr(
                    getattr(_res_tuple[0], "query_intent", None), "query_type", None
                )
                if _routed_as and _routed_as != "analytics":
                    logger.warning(
                        f"prewarm_analytics_wrong_route idx={idx} routed_as={_routed_as} query={query!r}"
                    )
            except asyncio.TimeoutError:
                failed += 1
                logger.warning(
                    f"prewarm_analytics_query_failed idx={idx} reason=timeout query={query!r}"
                )
            except Exception as _pw_e:  # noqa: BLE001
                failed += 1
                logger.warning(
                    f"prewarm_analytics_query_failed idx={idx} reason={type(_pw_e).__name__} error={_pw_e} query={query!r}"
                )

    total = len(_ANALYTICS_PREWARM_QUERIES)
    logger.info(
        f"prewarm_analytics_start query_count={total} concurrency={_ANALYTICS_PREWARM_CONCURRENCY}"
    )
    _t0 = time.monotonic()
    await asyncio.gather(
        *[_run_one(q, i) for i, q in enumerate(_ANALYTICS_PREWARM_QUERIES)]
    )
    _elapsed_ms = round((time.monotonic() - _t0) * 1000, 1)
    logger.info(
        f"prewarm_analytics_complete total={total} succeeded={total - failed} failed={failed} elapsed_ms={_elapsed_ms}"
    )


async def _warm_clickhouse_connection(
    sub: Subsystems, clickhouse_available: bool
) -> None:
    """Open the shared ClickHouse httpx connection pool before prewarm/traffic.

    Uses ``nl_to_sql.analytics.clickhouse.readiness_probe`` sql + warm_timeout_seconds
    from config. Pays TCP+session-auth setup once so subsequent prewarm and live
    queries keep their full latency budget for the actual read.
    """
    router = getattr(sub, "analytics_router", None)
    executor = (
        getattr(router, "clickhouse_executor", None) if router is not None else None
    )
    if (
        not clickhouse_available
        or executor is None
        or not executor.credentials_available
    ):
        logger.info(
            "clickhouse_connection_warm_skipped reason=clickhouse_unavailable_or_executor_missing"
        )
        return
    analytics_cfg = getattr(getattr(sub, "config", None), "nl_to_sql", None)
    analytics_cfg = (
        getattr(analytics_cfg, "analytics", None) if analytics_cfg is not None else None
    )
    ch_cfg = (
        getattr(analytics_cfg, "clickhouse", None)
        if analytics_cfg is not None
        else None
    )
    probe = getattr(ch_cfg, "readiness_probe", None) if ch_cfg is not None else None
    if probe is None:
        logger.warning(
            "clickhouse_connection_warm_skipped reason=readiness_probe_config_missing"
        )
        return
    warm_sql = str(probe.sql).strip()
    warm_timeout_s = float(probe.warm_timeout_seconds)
    try:
        await asyncio.wait_for(executor.execute(warm_sql), timeout=warm_timeout_s)
        logger.info(
            f"clickhouse_connection_warmed sql={warm_sql!r} timeout_s={warm_timeout_s}"
        )
    except asyncio.TimeoutError:
        logger.warning(
            f"clickhouse_connection_warm_failed reason=timeout timeout_s={warm_timeout_s}"
        )
    except Exception as _w_e:  # noqa: BLE001
        logger.warning(
            f"clickhouse_connection_warm_failed reason={type(_w_e).__name__} error={_w_e}"
        )


def _maybe_schedule_router_retraining(
    subsystems: "Subsystems", config_dict: Dict[str, Any]
) -> None:
    """If qi.router_retraining.enabled, schedule threshold check + harvest/train as background task."""
    rr = getattr(getattr(subsystems, "config", None), "qi", None)
    rr_cfg = getattr(rr, "router_retraining", None) if rr is not None else None
    if rr_cfg is None or not rr_cfg.enabled:
        return
    ch_raw = config_dict.get("retrieval", {}).get("clickhouse", {})
    ch_host = str(ch_raw.get("host", ""))
    ch_port = int(ch_raw.get("port", 0))
    ch_user = str(ch_raw.get("user", ""))
    ch_database = str(ch_raw.get("database", ""))
    ch_password = os.getenv("CLICKHOUSE_PASSWORD", "")
    if not ch_host or not ch_port or not ch_user or not ch_database:
        logger.warning(
            "router_retraining_skipped reason=clickhouse_connection_params_missing — retrieval.clickhouse must define host/port/user/database"
        )
        return
    ch_url = f"http{'s' if str(ch_raw.get('secure', 'false')).lower() in ('true', '1') else ''}://{ch_host}:{ch_port}"
    qi_cfg = getattr(subsystems, "config", None)
    seeds_path_raw = rr_cfg.seeds_path
    if not seeds_path_raw:
        seeds_path_raw = str(
            getattr(getattr(qi_cfg, "qi", None), "semantic", None)
            and getattr(getattr(qi_cfg, "qi", None).semantic, "seeds_path", "")
            or ""
        )
    asyncio.get_event_loop().create_task(
        _run_router_retraining_if_needed(
            rr_cfg, ch_url, ch_user, ch_password, ch_database, seeds_path_raw
        )
    )
    logger.info("router_retraining_check_scheduled")


async def _run_router_retraining_if_needed(
    rr_cfg: Any,
    ch_url: str,
    ch_user: str,
    ch_password: str,
    ch_database: str,
    seeds_path_raw: str,
) -> None:
    """Background task: check threshold and run harvest + train pipeline if met."""
    from semantic_search.qi.training.signal_harvester import (  # noqa: PLC0415
        maybe_trigger_harvest_and_train,
    )
    from pathlib import Path  # noqa: PLC0415

    try:
        triggered = await asyncio.to_thread(
            maybe_trigger_harvest_and_train,
            ch_url=ch_url,
            ch_user=ch_user,
            ch_password=ch_password,
            database=ch_database,
            seeds_path=Path(seeds_path_raw)
            if seeds_path_raw
            else Path(__file__).resolve().parents[0] / "qi" / "router_seeds.yaml",  # noqa: ASYNC240
            traffic_signals_path=Path(rr_cfg.traffic_signals_path),
            traffic_signals_max_rows=rr_cfg.traffic_signals_max_rows,
            positive_types=set(rr_cfg.positive_signal_types),
            lookback_days=rr_cfg.clickhouse_lookback_days,
            min_per_arch=rr_cfg.min_new_signals_per_archetype,
            max_per_arch=rr_cfg.max_signals_per_archetype,
            new_signals_threshold=rr_cfg.new_signals_threshold,
            last_retrain_marker_path=Path(rr_cfg.last_retrain_marker_path),
            output_npz_path=rr_cfg.output_npz_path,
            ngram_weights_output_path=rr_cfg.ngram_weights_output_path,
            ngram_top_k=rr_cfg.ngram_top_k,
            ngram_min_log_odds=rr_cfg.ngram_min_log_odds,
            hard_negatives_path=rr_cfg.hard_negatives_path,
            hard_neg_oversample=rr_cfg.hard_neg_oversample,
            hard_neg_threshold=rr_cfg.hard_neg_threshold,
        )
        if triggered:
            logger.info("router_retraining_pipeline_completed")
        else:
            logger.info("router_retraining_threshold_not_met — skipped")
    except Exception as e:  # noqa: BLE001
        logger.warning(
            f"router_retraining_pipeline_failed error_type={type(e).__name__} error={e}"
        )


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:  # noqa: ARG001
    """Initialize subsystems on startup; clean up on shutdown. LLM tier is best-effort — if it cannot initialize the service boots in degraded (no-LLM) mode."""
    config_dict = load_config()
    quiet_llm_core_child_loggers_warning()
    _configured_log_level = (config_dict.get("general") or {}).get("log_level", "INFO")
    _effective_log_level = os.environ.get("LOG_LEVEL", "").strip() or _configured_log_level
    apply_log_level_from_config(_effective_log_level)
    if is_katana_env():
        asyncio.create_task(asyncio.to_thread(ensure_search_logs_retention_policy))
    # LLM is best-effort. When the provider is missing/invalid the service still boots:
    # QI degrades to the L0_fallback path (ngram pre-gate + L1 semantic router + regex
    # aggregation gate), L0 entity extraction and L2 LLM classification are skipped, and
    # NL-to-SQL / LLM-judge wire themselves off (registry guards call_router is None).
    # We log loudly at the LLM stage so the degradation is visible, then continue with
    # llm_provider=None rather than aborting startup. (The DistilBERT query transformer
    # is separate and degrades independently.)
    llm_provider: Optional[LLMProvider] = None
    provider_config: Optional[Dict[str, Any]] = None
    _gocode_auth: Optional[Any] = None
    try:
        base_url_yaml = config_dict["llm_base_url"]
        base_url = (
            (base_url_yaml or "").strip() if isinstance(base_url_yaml, str) else ""
        )
        if not base_url:
            base_url = os.getenv("LLM_BASE_URL", "").strip()
        _jwt_auth_used = False
        _orig_anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        _orig_openai_key = os.environ.get("OPENAI_API_KEY", "")
        gocode_env = os.getenv("GOCODE_ENV", "").strip().lower()
        if gocode_env:
            try:
                from semantic_search.core.gocode_auth import GoCodeIAMAuth, GOCODE_URLS  # noqa: PLC0415

                _gocode_auth = GoCodeIAMAuth(gocode_env)
                _jwt = await _gocode_auth.get_token()
                os.environ["OPENAI_API_KEY"] = _jwt
                os.environ["ANTHROPIC_API_KEY"] = _jwt
                if not base_url:
                    base_url = GOCODE_URLS.get(gocode_env, "")
                _jwt_auth_used = True
                logger.info(
                    f"gocode_iam_jwt_minted env={gocode_env} base_url={base_url} exp_in_s={_gocode_auth.seconds_to_expiry}"
                )
            except Exception as _gce:  # noqa: BLE001
                logger.warning(
                    f"gocode_iam_jwt_mint_failed error_type={type(_gce).__name__} error={_gce} — falling back to env api keys"
                )
                _gocode_auth = None
        lm = config_dict["llm_models"]
        provider_config = {
            "llm_api_keys": config_dict["llm_api_keys"],
            "llm_models": {
                "temperature": lm["temperature"],
                "max_tokens": lm["max_tokens"],
                "max_tokens_validation": lm["max_tokens_validation"],
                "temperature_unsupported_models": lm.get(
                    "temperature_unsupported_models"
                )
                or [],
            },
            "llm_client_settings": config_dict["llm_client_settings"],
            "llm_base_url": base_url,
            "model_selection_strategy": config_dict["model_selection_strategy"],
            "model_capability_overrides": config_dict["model_capability_overrides"],
            "model_latency_overrides": config_dict["model_latency_overrides"],
        }
        provider = LLMProvider(config=provider_config, feedback_store=None)
        validated = await provider.validate_api_keys(live_check=True)
        live_check_used = True
        if not validated:
            validated = await provider.validate_api_keys(live_check=False)
            live_check_used = False
        if not validated and _jwt_auth_used:
            logger.warning(
                "gocode_iam_jwt_discovery_failed validated_providers=[] — "
                "restoring original env api keys and retrying. "
                f"orig_anthropic_key_present={bool(_orig_anthropic_key)} "
                f"orig_openai_key_present={bool(_orig_openai_key)}"
            )
            _gocode_auth = None
            if _orig_anthropic_key:
                os.environ["ANTHROPIC_API_KEY"] = _orig_anthropic_key
            else:
                os.environ.pop("ANTHROPIC_API_KEY", None)
            if _orig_openai_key:
                os.environ["OPENAI_API_KEY"] = _orig_openai_key
            else:
                os.environ.pop("OPENAI_API_KEY", None)
            validated = await provider.validate_api_keys(live_check=True)
            live_check_used = True
            if not validated:
                validated = await provider.validate_api_keys(live_check=False)
                live_check_used = False
            if validated:
                logger.info(
                    f"llm_provider_static_key_fallback_success providers={sorted(validated.keys())}"
                )
            else:
                logger.warning(
                    "llm_provider_static_key_fallback_failed no_providers_validated"
                )
        provider.build_model_registry()
        # Timed short generation for selected task models (task_model_allowlists ∩
        # discovered) — seeds runtime_stats AND cold-starts classify/extract routes.
        try:
            seeded = await provider.probe_startup_inference()
            allowlists = (config_dict.get("model_selection_strategy") or {}).get(
                "task_model_allowlists"
            ) or {}
            selected = sorted(
                {
                    str(m).strip()
                    for allow in (
                        allowlists.values() if isinstance(allowlists, dict) else []
                    )
                    if isinstance(allow, list)
                    for m in allow
                    if str(m).strip()
                }
            )
            missing_selected = [m for m in selected if m not in seeded]
            logger.info(
                f"llm_startup_inference_probe_done seeded={len(seeded)} "
                f"models={sorted(seeded.keys())} task_allowlist_selected={selected} "
                f"task_allowlist_warmed={[m for m in selected if m in seeded]} "
                f"task_allowlist_missing={missing_selected}"
            )
        except Exception as _probe_exc:  # noqa: BLE001
            # Best-effort: ranking falls back to static latency tiers.
            logger.warning(
                f"llm_startup_inference_probe_failed error_type={type(_probe_exc).__name__} "
                f"error={_probe_exc}"
            )
        provider.build_task_fallbacks()
        provider.initialize_cached_clients()
        await provider.start_background_refresh()
        summ = provider.get_summary()
        n_models = int(summ.get("available_models", 0))
        provs = summ.get("validated_providers") or []
        logger.info(
            f"llm_provider_ready providers={provs} models={n_models} live_probe={live_check_used}"
        )
        if n_models == 0:
            env_vars: List[str] = []
            raw_keys = config_dict["llm_api_keys"]
            if isinstance(raw_keys, dict):
                for _prov, entries in raw_keys.items():
                    if not isinstance(entries, list):
                        continue
                    for ent in entries:
                        if isinstance(ent, dict) and isinstance(
                            ent.get("key_env_var"), str
                        ):
                            env_vars.append(ent["key_env_var"])
            await provider.stop_background_refresh()
            logger.warning(
                "llm_provider_degraded reason=zero_usable_models — booting in no-LLM mode. "
                f"validated_providers={provs} key_env_vars={env_vars or ['(see llm_api_keys in config)']} "
                f"base_url_set={bool(base_url)}. QI uses L0_fallback (ngram + L1 + aggregation gate); "
                "L0 entity extraction, L2 LLM classification, NL-to-SQL and the LLM judge are off. "
                "Fix the API keys or set LLM_BASE_URL to a reachable gateway to restore the LLM tier."
            )
            llm_provider = None
        else:
            llm_provider = provider
    except (AgentSearchError, RuntimeError, OSError) as e:
        # Best-effort dependency — degrade to no-LLM mode instead of aborting startup.
        # Downstream (registry/QI engine) is None-safe: the LLM tiers wire themselves off
        # and QI falls back to the L0_fallback path.
        logger.warning(
            f"llm_provider_degraded reason=init_failed ({type(e).__name__}: {e}) — booting in no-LLM mode. "
            "QI uses L0_fallback (ngram + L1 + aggregation gate); L0 entity extraction, L2 LLM "
            "classification, NL-to-SQL and the LLM judge are off. Fix the API keys or set "
            "LLM_BASE_URL to a reachable gateway to restore the LLM tier."
        )
        llm_provider = None
    # Pre-flight: validate pretrained models (CI syncs from S3 -> /app/pretrained at build).
    _validate_pretrained_models(config_dict)

    app_state.subsystems = build_subsystems(config_dict, llm_provider)
    from semantic_search.measurement.qie_only_launch import configure_qie_only_launch_stats

    configure_qie_only_launch_stats(
        app_state.subsystems.config.measurement.qie_only_launch
    )

    # Warm the structural-capability-gate cache for the classify ensemble's
    # discriminated-union schema before traffic arrives. Without this, the
    # first classify() call after boot (or after a probe_ttl_seconds expiry)
    # pays the cold-probe round trip (observed 5-8s) inside its own tighter
    # qi.llm.classify_timeout_seconds budget (4s) and degrades to
    # _classify_timeout_regex_fallback. Awaited (not fire-and-forget) so the
    # cache is actually warm before _prewarm_sequential or live traffic runs.
    _call_router = app_state.subsystems.call_router
    if (
        _call_router is not None
        and app_state.subsystems.structural_gate is not None
        and llm_provider is not None
    ):
        _classify_task_type = app_state.subsystems.config.qi.llm.task_type
        _classify_models = llm_provider.get_fallback_chain(_classify_task_type)
        if _classify_models:
            try:
                await _call_router.warmup_structural_gate_for_schema(
                    _classify_models, QIClassificationResponse
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    f"structural_gate_warmup_failed task_type={_classify_task_type} "
                    f"error_type={type(e).__name__} error={e}"
                )
        else:
            logger.warning(
                f"structural_gate_warmup_skipped reason=empty_fallback_chain task_type={_classify_task_type}"
            )

    app_state.encoder_degraded = isinstance(
        app_state.subsystems.encoder, HashingEncoder
    )
    if app_state.encoder_degraded:
        logger.warning(
            "encoder_degraded_mode active — L1 semantic routing uses HashingEncoder; recall is lower than fastembed"
        )
    _startup_qi = getattr(app_state.subsystems, "qi_engine", None)
    _startup_ng = (
        getattr(_startup_qi, "_ngram_pre_gate", None)
        if _startup_qi is not None
        else None
    )
    if _startup_ng is not None and _startup_ng.enabled:
        logger.info(
            f"ngram_pre_gate_active vocab_size={len(_startup_ng._weights)} classes={_startup_ng._classes} threshold={_startup_ng._config.confidence_threshold}"
        )
    else:
        logger.warning(
            "ngram_pre_gate_inactive — QI pre-gate not loaded; all queries route through L1/LLM tiers"
        )
    # Pre-warm the DistilBERT query rewriter so the first
    # live query does not pay torch kernel/JIT init. Model is already resident
    # from build_subsystems; this only exercises the forward graphs.
    await _prewarm_query_transformer(app_state.subsystems)
    _ng_retrieval_cfg = getattr(
        getattr(
            getattr(app_state.subsystems.config, "retrieval", None), "qdrant", None
        ),
        "hybrid",
        None,
    )
    _ng_retrieval_cfg = (
        getattr(_ng_retrieval_cfg, "ngram", None)
        if _ng_retrieval_cfg is not None
        else None
    )
    if _ng_retrieval_cfg is None or not _ng_retrieval_cfg.enabled:
        logger.error(
            "retrieval_ngram_disabled — retrieval.qdrant.hybrid.ngram.enabled is false or absent; "
            "fuzzy-recall leg is off and reindex will not write the ngram sparse vector. "
            "Set retrieval.qdrant.hybrid.ngram.enabled: true and reindex to restore."
        )
    # Apply permanently unavailable columns at startup so filter skipping is active before first data build.
    _startup_perm_missing = set(
        app_state.subsystems.config.general.search.permanently_unavailable_columns
    )
    if _startup_perm_missing:
        app_state._missing_data_columns = _startup_perm_missing
        _startup_unavail_ents = _get_unavailable_filter_entities(_startup_perm_missing)
        set_unavailable_filter_keys(_startup_unavail_ents)
        logger.info(
            f"startup_permanent_unavailable_columns columns={sorted(_startup_perm_missing)} filter_entities_skipped={sorted(_startup_unavail_ents)}"
        )

    # Deferred calibration fit (background thread; service boots with T=1.0, updates in-place on completion).
    _calib_fn = app_state.subsystems.calibration_boot_fit_fn
    if _calib_fn is not None:
        app_state._calibration_task = asyncio.get_running_loop().run_in_executor(
            None, _calib_fn
        )
        logger.info("calibration_boot_fit_deferred background_thread=True")

    _evicted = app_state.subsystems.orchestrator.clear_caches()
    logger.info(f"lru_cache_cleared_at_startup evicted={_evicted}")

    # Probe external dependencies (Qdrant/CH/LLM).
    dep_status = await _dependency_health_check(app_state.subsystems)
    _dep_lines = "  ".join(f"{k}={v}" for k, v in dep_status.items())
    _dep_ok = all(v in ("ok", "not_configured") for v in dep_status.values())
    if _dep_ok:
        logger.info(f"dependency_health_ok {_dep_lines}")
    else:
        _failed = [k for k, v in dep_status.items() if v == "unavailable"]
        logger.warning(f"dependency_health_degraded failed={_failed} {_dep_lines}")

    seed_result = None
    try:
        seed_result = await seed_boot_indexes(app_state.subsystems)
        if seed_result is not None:
            vec_ct = seed_result["loading"]["vector_indexed"]
            str_ct = seed_result["loading"]["structured_indexed"]
            logger.info(f"boot_seed_loaded vector={vec_ct} structured={str_ct}")
            _seed_vec = app_state.subsystems.config.vectorization
            _seed_db = _seed_vec.seed.database if _seed_vec and _seed_vec.seed else None
            _seed_tbl0 = _seed_db.tables[0] if _seed_db and _seed_db.tables else None
            app_state._build_history.append(
                {
                    "build_id": str(uuid.uuid4())[:8],
                    "triggered_by": "boot",
                    "mode": "rebuild",
                    "source": "boot",
                    "strategy": _seed_tbl0.strategy if _seed_tbl0 else None,
                    "lookback_days": _seed_tbl0.lookback_days if _seed_tbl0 else None,
                    "max_records": _seed_tbl0.max_records if _seed_tbl0 else None,
                    "active_only": _seed_tbl0.active_only if _seed_tbl0 else None,
                    "started_at": None,
                    "documents_offered": seed_result["loading"].get(
                        "documents_offered"
                    ),
                    "vector_indexed": vec_ct,
                    "bm25_encoded": seed_result["loading"].get("bm25_encoded"),
                    "bm25_skipped": seed_result["loading"].get("bm25_skipped"),
                    "skipped": seed_result["loading"].get("skipped"),
                    "elapsed_ms": seed_result["loading"].get("elapsed_ms"),
                    "status": "success",
                }
            )
    except (AgentSearchError, RuntimeError, OSError) as e:
        logger.warning(
            f"boot_seed_failed error_type={type(e).__name__} error={str(e)} seed=off"
        )
    # Seed schedule: in-process loop when enabled and in_process; otherwise
    # schedule fields remain available on GET /data-build/status.
    _seed_sched = None
    if (
        app_state.subsystems.config.vectorization is not None
        and app_state.subsystems.config.vectorization.seed is not None
    ):
        _seed_sched = app_state.subsystems.config.vectorization.seed.schedule
    if (
        _seed_sched is not None
        and _seed_sched.enabled
        and _seed_sched.in_process
    ):
        app_state._seed_task = asyncio.create_task(_seed_background_task())
        logger.info(
            f"seed_build_task_started interval_hours={_seed_sched.interval_hours} "
            f"run_at_hour_utc={_seed_sched.run_at_hour_utc} in_process=true "
            f"run_on_deploy={_seed_sched.run_on_deploy}"
        )
    elif _seed_sched is not None and _seed_sched.enabled:
        logger.info(
            f"seed_build_schedule_status_only interval_hours={_seed_sched.interval_hours} "
            f"run_at_hour_utc={_seed_sched.run_at_hour_utc} in_process=false "
            f"run_on_deploy={_seed_sched.run_on_deploy}"
        )
    # Query-driven synonym expander (LLM expansion for cache misses; active when BM25 fallback used).
    _dyn_store = getattr(app_state.subsystems, "dynamic_synonym_store", None)
    if _dyn_store is not None and _dyn_store.enabled and llm_provider is not None:
        try:
            app_state._synonym_expander = QueryDrivenExpander(
                store=_dyn_store,
                full_config=provider_config,
                poll_interval_seconds=30,
                batch_size=32,
            )
            app_state._synonym_expansion_task = asyncio.create_task(
                app_state._synonym_expander.run_forever()
            )
            app_state._synonym_shutdown_grace_s = float(
                _dyn_store._config.shutdown_grace_timeout_seconds
            )
            logger.info(
                f"dynamic_synonym_expander_started store_size={_dyn_store.map_size}"
            )
        except Exception as _dse:  # noqa: BLE001
            logger.warning(
                f"dynamic_synonym_expander_start_failed error_type={type(_dse).__name__} error={_dse}"
            )
    if app_state.subsystems.enriched_tables_builder is not None:
        try:
            app_state._enriched_builder_task = asyncio.create_task(
                app_state.subsystems.enriched_tables_builder.loop_forever(
                    interval_hours=6.0
                )
            )
            logger.info("enriched_tables_builder_loop_started interval_hours=6.0")
        except Exception as _etb_e:  # noqa: BLE001
            logger.warning(
                f"enriched_tables_builder_loop_start_failed error_type={type(_etb_e).__name__} error={_etb_e}"
            )
    if app_state.subsystems.vector_refresh_driver is not None:
        try:
            await app_state.subsystems.vector_refresh_driver.start()
            logger.info("vector_refresh_driver_started driver=vector_refresh")
        except (AgentSearchError, RuntimeError, OSError) as e:
            logger.warning(
                f"vector_refresh_driver_start_failed error_type={type(e).__name__} "
                f"error={str(e)} vector_refresh=off"
            )
    # Delta refresh: patches mutable fields (price/bids/type/ends_at) via polling. Phase 1.5: cutover to Kinesis.
    if app_state.subsystems.delta_refresh_driver is not None:
        try:
            await app_state.subsystems.delta_refresh_driver.start()
            logger.info("delta_refresh_driver_started driver=delta_refresh")
        except (AgentSearchError, RuntimeError, OSError) as e:
            logger.warning(
                f"delta_refresh_driver_start_failed error_type={type(e).__name__} error={str(e)} delta_refresh=off"
            )
    if app_state.subsystems.event_ingest_driver is not None:
        try:
            await app_state.subsystems.event_ingest_driver.start()
            logger.info("event_ingest_driver_started driver=event_ingest")
        except (AgentSearchError, RuntimeError, OSError) as e:
            logger.warning(
                f"event_ingest_driver_start_failed error_type={type(e).__name__} error={str(e)} event_ingest=off"
            )
    # Periodic seed-time enrichment refresh (majestic/semrush/estibot/search_rollup), Qdrant-only.
    if app_state.subsystems.enrichment_refresh_driver is not None:
        try:
            await app_state.subsystems.enrichment_refresh_driver.start()
            logger.info("enrichment_refresh_driver_started driver=enrichment_refresh")
        except (AgentSearchError, RuntimeError, OSError) as e:
            logger.warning(
                f"enrichment_refresh_driver_start_failed error_type={type(e).__name__} error={str(e)} enrichment_refresh=off"
            )
    # 90-day compactor (best-effort; search/resume remain available even if driver fails; right-to-delete always available).
    if app_state.subsystems.history_compactor_driver is not None:
        try:
            await app_state.subsystems.history_compactor_driver.start()
            logger.info("history_compactor_driver_started driver=history_compactor")
        except (AgentSearchError, RuntimeError, OSError) as e:
            logger.warning(
                f"history_compactor_driver_start_failed error_type={type(e).__name__} "
                f"error={str(e)} history_compactor=off"
            )
    if app_state.subsystems.centroid_retrainer_driver is not None:
        try:
            await app_state.subsystems.centroid_retrainer_driver.start()
            logger.info("centroid_retrainer_driver_started")
        except (AgentSearchError, RuntimeError, OSError) as e:
            logger.warning(
                f"centroid_retrainer_driver_start_failed error_type={type(e).__name__} error={str(e)}"
            )
    if app_state.subsystems.ch_tld_refresh_driver is not None:
        try:
            await app_state.subsystems.ch_tld_refresh_driver.start()
            logger.info("ch_tld_refresh_driver_started")
        except (AgentSearchError, RuntimeError, OSError) as e:
            logger.warning(
                f"ch_tld_refresh_driver_start_failed error_type={type(e).__name__} error={str(e)}"
            )
    _maybe_schedule_router_retraining(app_state.subsystems, config_dict)
    # Warm the shared ClickHouse connection pool before prewarm fires so the first
    # guidance/explore-rail/analytics queries don't pay cold connection+auth cost on
    # top of their tight per-component timeouts (rail/snapshot TimeoutError noise).
    await _warm_clickhouse_connection(
        app_state.subsystems, dep_status.get("clickhouse") == "ok"
    )

    async def _prewarm_sequential() -> None:
        # Run hybrid then analytics back-to-back rather than concurrently so the
        # shared single-node Qdrant/ClickHouse stack sees peak concurrency 4
        # (one phase) instead of 8 (both phases overlapping). The overlapping
        # burst is what pushed rail/guidance queries past their per-component
        # timeouts and flipped the vector backend to degraded at boot.
        await _prewarm_hybrid_queries(
            app_state.subsystems, dep_status.get("qdrant") == "ok"
        )
        await _prewarm_analytics_queries(
            app_state.subsystems, dep_status.get("clickhouse") == "ok"
        )

    asyncio.create_task(_prewarm_sequential())
    _gocode_refresh_task: Optional[asyncio.Task] = None
    if _gocode_auth is not None and llm_provider is not None:
        _lp, _ga = llm_provider, _gocode_auth

        async def _refresh_gocode_jwt() -> None:
            while True:
                try:
                    await asyncio.sleep(60)
                    _new_jwt = await _ga.get_token()
                    os.environ["OPENAI_API_KEY"] = _new_jwt
                    os.environ["ANTHROPIC_API_KEY"] = _new_jwt
                    await _lp.validate_api_keys(live_check=False)
                    _lp.initialize_cached_clients()
                    logger.info(
                        f"gocode_iam_jwt_refreshed exp_in_s={_ga.seconds_to_expiry}"
                    )
                except asyncio.CancelledError:
                    break
                except Exception as _re:  # noqa: BLE001
                    logger.warning(
                        f"gocode_iam_jwt_refresh_failed error_type={type(_re).__name__} error={_re}"
                    )

        _gocode_refresh_task = asyncio.create_task(_refresh_gocode_jwt())
    logger.info("semantic_search_api_startup_complete event=startup")
    yield
    if _gocode_refresh_task is not None and not _gocode_refresh_task.done():
        _gocode_refresh_task.cancel()
        try:
            await _gocode_refresh_task
        except asyncio.CancelledError:
            pass
    if app_state._synonym_expander is not None:
        app_state._synonym_expander.stop()
    if (
        app_state._synonym_expansion_task is not None
        and not app_state._synonym_expansion_task.done()
    ):
        try:
            await asyncio.wait_for(
                app_state._synonym_expansion_task,
                timeout=app_state._synonym_shutdown_grace_s,
            )
        except (asyncio.CancelledError, asyncio.TimeoutError):
            app_state._synonym_expansion_task.cancel()
    if (
        app_state._calibration_task is not None
        and not app_state._calibration_task.done()
    ):
        app_state._calibration_task.cancel()
        try:
            await app_state._calibration_task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
    if app_state._seed_task is not None and not app_state._seed_task.done():
        app_state._seed_task.cancel()
        try:
            await app_state._seed_task
        except asyncio.CancelledError:
            pass
    if (
        app_state._enriched_builder_task is not None
        and not app_state._enriched_builder_task.done()
    ):
        app_state._enriched_builder_task.cancel()
        try:
            await app_state._enriched_builder_task
        except asyncio.CancelledError:
            pass
    if (
        app_state.subsystems is not None
        and app_state.subsystems.vector_refresh_driver is not None
    ):
        try:
            await app_state.subsystems.vector_refresh_driver.stop()
        except (AgentSearchError, RuntimeError, OSError) as e:
            logger.warning(
                f"vector_refresh_driver_stop_failed error_type={type(e).__name__} error={str(e)}"
            )
    if (
        app_state.subsystems is not None
        and app_state.subsystems.delta_refresh_driver is not None
    ):
        try:
            await app_state.subsystems.delta_refresh_driver.stop()
        except (AgentSearchError, RuntimeError, OSError) as e:
            logger.warning(
                f"delta_refresh_driver_stop_failed error_type={type(e).__name__} error={str(e)}"
            )
    if (
        app_state.subsystems is not None
        and app_state.subsystems.enrichment_refresh_driver is not None
    ):
        try:
            await app_state.subsystems.enrichment_refresh_driver.stop()
        except (AgentSearchError, RuntimeError, OSError) as e:
            logger.warning(
                f"enrichment_refresh_driver_stop_failed error_type={type(e).__name__} error={str(e)}"
            )
    if (
        app_state.subsystems is not None
        and app_state.subsystems.centroid_retrainer_driver is not None
    ):
        try:
            await app_state.subsystems.centroid_retrainer_driver.stop()
        except (AgentSearchError, RuntimeError, OSError) as e:
            logger.warning(
                f"centroid_retrainer_driver_stop_failed error_type={type(e).__name__} error={str(e)}"
            )
    if (
        app_state.subsystems is not None
        and app_state.subsystems.ch_tld_refresh_driver is not None
    ):
        try:
            await app_state.subsystems.ch_tld_refresh_driver.stop()
        except (AgentSearchError, RuntimeError, OSError) as e:
            logger.warning(
                f"ch_tld_refresh_driver_stop_failed error_type={type(e).__name__} error={str(e)}"
            )
    if (
        app_state.subsystems is not None
        and app_state.subsystems.history_compactor_driver is not None
    ):
        try:
            await app_state.subsystems.history_compactor_driver.stop()
        except (AgentSearchError, RuntimeError, OSError) as e:
            logger.warning(
                f"history_compactor_driver_stop_failed error_type={type(e).__name__} error={str(e)}"
            )
    if (
        app_state.subsystems is not None
        and app_state.subsystems.llm_provider is not None
    ):
        try:
            await app_state.subsystems.llm_provider.stop_background_refresh()
        except (AgentSearchError, RuntimeError, OSError) as e:
            logger.warning(
                f"llm_provider_stop_failed error_type={type(e).__name__} error={str(e)}"
            )
    logger.info("semantic_search_api_shutdown_complete event=shutdown")


app = FastAPI(
    title="Agent Search API",
    description=(
        "QI Engine, Hybrid Retrieval, Caching, eRanker, Analytics, A/B Bucketing. "
        "**SLA:** hybrid/explore/guidance ≤ 10 s; analytics ≤ 20 s (total from request start)."
    ),
    version="0.1.0",
    lifespan=lifespan,
    openapi_tags=_OPENAPI_TAGS,
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1,
        "docExpansion": "list",
        "filter": True,
        "tryItOutEnabled": True,
    },
)

app.add_middleware(SecurityHeadersMiddleware)
_cors_origins = [
    o.strip() for o in os.environ.get("CORS_ALLOWED_ORIGINS", "*").split(",")
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(QdrantUnavailableError)
@app.exception_handler(QdrantQueryError)
@app.exception_handler(ClickHouseUnavailableError)
@app.exception_handler(ClickHouseQueryError)
async def _backend_unavailable_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Global 503 JSON for Qdrant/CH down — clear message + identity for log join."""
    id_cfg = None
    sub = getattr(app_state, "subsystems", None)
    if sub is not None:
        id_cfg = sub.config.identity
    else:
        boot = globals().get("_bootstrap_config")
        if boot is not None:
            id_cfg = boot.identity
    if id_cfg is None:
        id_cfg = AgentSearchConfig.from_dict(load_config()).identity
    identity = resolve_identity(id_cfg, headers=request.headers)
    query = ""
    if request.method == "POST":
        try:
            form = await request.form()
            raw_q = form.get("query")
            if isinstance(raw_q, str):
                query = raw_q
        except (RuntimeError, ValueError, TypeError, OSError):
            query = ""
    return _backend_unavailable_response(
        query=query or request.url.path,
        request_id=identity.request_id,
        search_id=identity.search_id,
        exc=exc,
        latency_ms=0.0,
        answer_mode="search",
    )


# Per-session rate limit (config loaded once at import; middleware attached before lifespan).
rate_limiter: Optional[SlidingWindowRateLimiter] = None
"""Module-level rate limiter handle; test fixture resets between test functions."""
try:
    _bootstrap_config = AgentSearchConfig.from_dict(load_config())
    _rate_limit_cfg = _bootstrap_config.general.rate_limit
    if _rate_limit_cfg.enabled and _rate_limit_cfg.paths:
        rate_limiter = SlidingWindowRateLimiter(
            max_requests=_rate_limit_cfg.max_requests_per_window,
            window_seconds=_rate_limit_cfg.window_seconds,
            session_state_max=_rate_limit_cfg.session_state_max,
            burst_requests=int(getattr(_rate_limit_cfg, "burst_requests", 0) or 0),
            burst_window_seconds=int(
                getattr(_rate_limit_cfg, "burst_window_seconds", 10) or 10
            ),
        )
        app.add_middleware(
            RateLimitMiddleware, config=_rate_limit_cfg, limiter=rate_limiter
        )
        logger.info(
            f"rate_limit_middleware_wired max={_rate_limit_cfg.max_requests_per_window} window_s={_rate_limit_cfg.window_seconds} paths={_rate_limit_cfg.paths}"
        )
    else:
        logger.info("rate_limit_middleware_disabled enabled_or_paths_empty=true")
except (ConfigurationError, FileNotFoundError, KeyError) as _e:
    # Config may not load in some test rigs; skip middleware but input sanitizer still defends path.
    logger.warning(
        f"rate_limit_middleware_skipped reason=config_load_failed error={str(_e)}"
    )


class UserContextBody(BaseModel):
    """Optional user-context envelope for history scoping and eRanker request context."""

    user_id: Optional[str] = Field(
        None, description="Stable authenticated user id; null for anonymous traffic"
    )
    is_authenticated: bool = Field(
        False, description="True iff the caller is authenticated"
    )
    session_id: str = Field(
        ..., description="Session identifier (required for history scoping)"
    )
    explicit_mode: Optional[str] = Field(
        None,
        description="Caller-asserted UI mode override ('conversational' | 'advanced'). "
        "When set, mode-selector rule #1 honors it before signal-based or default policy.",
    )


def _user_context_from_body(body: Optional[UserContextBody]) -> Optional[UserContext]:
    """Translate UserContextBody -> UserContext contract (None if body omitted)."""
    if body is None:
        return None
    return UserContext(
        user_id=body.user_id,
        is_authenticated=bool(body.is_authenticated),
        session_id=body.session_id,
        explicit_mode=body.explicit_mode,
    )


_ROUTING_LABELS: Dict[str, str] = {
    "hybrid": "hybrid_dense_bm42_retrieval",
    "explore": "explore_retrieval",
    "guidance": "guidance_market_snapshot",
    "analytics": "nl_to_sql_analytics",
    "multi_intent": "multi_intent_rrf_retrieval",
}

# Hybrid queries with no semantic residual pay zero embedding cost (structured-only path). Residual_kind decides label.
_STRUCTURED_FILTER_LABEL = "structured_filter_retrieval"


def _routing_label(
    query_type: str, residual_kind: Optional[str] = None, n_slices: int = 1
) -> str:
    if n_slices > 1:
        return _ROUTING_LABELS["multi_intent"]
    if query_type == "hybrid" and residual_kind == "empty":
        return _STRUCTURED_FILTER_LABEL
    return _ROUTING_LABELS.get(query_type, f"hybrid_rrf_retrieval ({query_type})")


# Classified query_type -> response answer_mode. Explore rails / timeouts / empty
# inventory fill the body but must NOT rewrite this label — parent routing tier wins.
_QUERY_TYPE_TO_ANSWER_MODE: Dict[str, str] = {
    "hybrid": "search",
    "explore": "explore",
    "guidance": "guidance",
    "analytics": "analytics",
}


def _answer_mode_for_query_type(query_type: Optional[str]) -> str:
    """Map parent routing tier (classified intent) to ``answer_mode``.

    Unknown / missing type -> ``explore_fallback`` (no trusted parent tier).
    """
    if query_type is None:
        return "explore_fallback"
    return _QUERY_TYPE_TO_ANSWER_MODE.get(str(query_type), "explore_fallback")


def _answer_mode_for_ranked_response(
    *, query_type: str, is_explore_fallback: bool
) -> str:
    """Map classified intent to ``answer_mode`` for the ranked-results response.

    Zero-result-guard explore rails, SLA timeout rails, and analytics substrate
    fallbacks may fill ``ranked_results`` — that is inventory degradation, not a
    re-route. ``is_explore_fallback`` is call-site documentation only; the parent
    tier (``query_type``) always owns the label.
    """
    _ = is_explore_fallback
    return _answer_mode_for_query_type(query_type)


_RESPONSE_FLOAT_PRECISION = 4


def _round_floats(obj: Any, precision: int) -> Any:
    """Recursively round all floats in dict/list/scalar to precision places."""
    if isinstance(obj, float):
        return round(obj, precision)
    if isinstance(obj, dict):
        return {k: _round_floats(v, precision) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, precision) for v in obj]
    return obj


def _normalise_scores(items: List[Any], mode: str = "max") -> List[float]:
    """Normalize fused_scores to [0,1]; round to 4 dp. Ranking order is preserved.

    :param items: List[Any] - Score-sorted RankedItems (each exposes ``fused_score``)
    :param mode: str - 'max' -> ``s / max`` (legacy; floor floats high). 'minmax' ->
        ``(s - min) / (max - min)`` (sharper spread; the tail drops toward 0).
        When all scores are equal, minmax falls back to 1.0 for every item so a
        genuinely uniform ranking is not forced to a degenerate 0.
    :return: List[float] - Normalized scores aligned to ``items`` order
    """
    if not items:
        return []
    raw = [float(item.fused_score) for item in items]
    if mode == "minmax":
        max_score = max(raw)
        min_score = min(raw)
        span = max_score - min_score
        if span <= 0.0:
            return [1.0 if max_score > 0 else 0.0 for _ in raw]
        return [round((s - min_score) / span, 4) for s in raw]
    max_score = max(raw)
    return [round(s / max_score if max_score > 0 else 0.0, 4) for s in raw]


def _format_ends_at(ts: Any) -> Optional[str]:
    """Convert Unix epoch to ISO 8601 UTC string (None if missing/non-numeric)."""
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    except (TypeError, ValueError, OSError):
        return None


def _format_money_display(amount: Any, find_listing: Any) -> str:
    """Format a money amount using find_listing display template/number format."""
    if amount is None:
        return str(find_listing.money_display_when_null)
    try:
        numeric = float(amount)
    except (TypeError, ValueError):
        return str(find_listing.money_display_when_null)
    formatted = format(numeric, find_listing.money_display_number_format)
    body = str(find_listing.money_display_template).replace(
        str(find_listing.money_display_value_placeholder), formatted
    )
    return f"{find_listing.money_display_currency_prefix}{body}"


def _slim_result(
    item: Any, rank: int, coherence_score: float, search_cfg: Any
) -> Dict[str, Any]:
    """Project RankedItem to configured result_fields + find_listing enrichments."""
    payload = item.payload or {}
    result_fields = list(search_cfg.result_fields)
    find_listing = search_cfg.find_listing
    _domain = item.item_id
    for _key in find_listing.domain_name_payload_keys:
        _candidate = payload.get(_key)
        if _candidate is not None and str(_candidate).strip():
            _domain = _candidate
            break
    result: Dict[str, Any] = {
        "rank": rank,
        "domain_name": _domain,
        "coherence_score": coherence_score,
        "matched_by": item.contributing_sources,
    }
    _end_keys = frozenset(find_listing.iso_timestamp_fields)
    _passthrough_iso = frozenset(find_listing.iso_passthrough_string_fields)
    _iso_fallback = find_listing.iso_fallback_source
    _display_targets: Dict[str, str] = {}
    for pair in find_listing.money_display_pairs:
        for target in pair.targets:
            _display_targets[target] = pair.source
    for field_name in result_fields:
        if field_name in _end_keys:
            raw_end = payload.get(field_name)
            if raw_end is None and field_name != _iso_fallback:
                raw_end = payload.get(_iso_fallback)
            if (
                field_name in _passthrough_iso
                and isinstance(raw_end, str)
                and raw_end.strip()
            ):
                result[field_name] = raw_end
            else:
                result[field_name] = _format_ends_at(raw_end)
        elif field_name in _display_targets:
            result[field_name] = _format_money_display(
                payload.get(_display_targets[field_name]), find_listing
            )
        else:
            _value = payload.get(field_name)
            # unique_search_count is None when the domain had no rows in the
            # domain_search_rollup lookback window (see
            # vectorization/seed_merge.py's stg_search_rollup join) — omit
            # rather than emit a null that's indistinguishable from a
            # legitimately-absent value.
            if field_name == "unique_search_count" and _value is None:
                continue
            result[field_name] = _value
    return result


def _auction_tiebreak(
    ranked_results: List[Dict[str, Any]], cfg: Any
) -> List[Dict[str, Any]]:
    """Relevance-dominant auction-signal tie-break over serialized ranked_results.

    Nudges ending-soon / low-competition / high-value domains UP, but ONLY among
    near-equal-coherence items: the blended key is
    ``coherence + cfg.max_bonus * (weighted signals)``, so with a small max_bonus a
    real coherence gap is never crossed. coherence_score values are NOT mutated and
    retrieval metrics are computed upstream from the pre-tiebreak order.

    :param ranked_results: List[Dict] - Serialized results (carry coherence_score,
        time_to_end_hours, bids/bid_count, valuation_price/govalue_score;
        absent => that signal = 0)
    :param cfg: AuctionTiebreakConfig - Validated tie-break config
    :return: List[Dict] - Re-ordered (stable on coherence for equal blended score)
    """
    if cfg is None or not getattr(cfg, "enabled", False) or len(ranked_results) < 2:
        return ranked_results

    def _f(v: Any) -> Optional[float]:
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _bids_of(r: Dict[str, Any]) -> float:
        return max(
            0.0,
            _f(r.get("bids") if r.get("bids") is not None else r.get("bid_count"))
            or 0.0,
        )

    def _value_of(r: Dict[str, Any]) -> float:
        raw = r.get("valuation_price")
        if raw is None:
            raw = r.get("govalue_score")
        return max(0.0, _f(raw) or 0.0)

    horizon = float(cfg.urgency_horizon_hours)
    bids = [_bids_of(r) for r in ranked_results]
    max_bids = max(bids) or 1.0
    gv_log = [math.log1p(_value_of(r)) for r in ranked_results]
    gv_min, gv_max = min(gv_log), max(gv_log)
    gv_span = (gv_max - gv_min) or 1.0
    bonus_cap = float(cfg.max_bonus)
    w_u, w_c, w_v = (
        float(cfg.weight_urgency),
        float(cfg.weight_low_competition),
        float(cfg.weight_value),
    )

    def _bonus(i: int, r: Dict[str, Any]) -> float:
        tte = _f(r.get("time_to_end_hours"))
        urgency = 0.0 if tte is None else max(0.0, min(1.0, (horizon - tte) / horizon))
        low_comp = 1.0 - (bids[i] / max_bids)
        value = (gv_log[i] - gv_min) / gv_span
        return bonus_cap * (w_u * urgency + w_c * low_comp + w_v * value)

    # Stable: equal blended score keeps original (coherence-descending) order via
    # the -i secondary key under reverse sort.
    keyed = [
        ((_f(r.get("coherence_score")) or 0.0) + _bonus(i, r), -i, r)
        for i, r in enumerate(ranked_results)
    ]
    keyed.sort(key=lambda t: (t[0], t[1]), reverse=True)
    reordered = [t[2] for t in keyed]
    # Re-stamp 1-based rank to match the new order so callers need no separate loop.
    for _new_rank, _r in enumerate(reordered, start=1):
        _r["rank"] = _new_rank
    return reordered


# Enrichment column -> filter entities mapping. Absent columns skip their entities.
_ATHENA_COL_TO_FILTER_ENTITIES: Dict[str, List[str]] = {
    "bidcount": ["bids_min", "bids_max"],
    "domain_age_years": ["domain_age_min", "domain_age_max"],
    "monthly_traffic": ["traffic_min", "traffic_max"],
    "majestic_tf": ["majestic_tf_min", "majestic_tf_max"],
    "majestic_cf": ["majestic_cf_min", "majestic_cf_max"],
    "majestic_backlinks": ["majestic_backlinks_min", "majestic_backlinks_max"],
    "majestic_ref_domains": ["majestic_ref_domains_min", "majestic_ref_domains_max"],
    "tlf_exact_match": ["tlf_exact_match"],
    "tlf_keyword_regs": ["tlf_keyword_regs_min"],
    "tlf_developed": ["tlf_developed"],
    "semrush_backlinks": ["semrush_backlinks_min", "semrush_backlinks_max"],
    "semrush_indexed_pages": ["semrush_indexed_pages_min", "semrush_indexed_pages_max"],
    "semrush_ref_domains": ["semrush_ref_domains_min", "semrush_ref_domains_max"],
    "semrush_authority_score": ["semrush_authority_min", "semrush_authority_max"],
    "semrush_search_volume": ["semrush_search_volume_min", "semrush_search_volume_max"],
    "semrush_cpc": ["semrush_cpc_min", "semrush_cpc_max"],
    "traffic_proxy_score": ["traffic_proxy_min", "traffic_proxy_max"],
    "has_web_traffic_signal": ["has_web_traffic_signal"],
    "estimated_traffic_tier": [
        "estimated_traffic_tier_min",
        "estimated_traffic_tier_max",
    ],
    "buy_it_now_flag": ["buy_it_now"],
    "buy_it_now_usd_amt": ["buy_it_now_min", "buy_it_now_max"],
    "reserve_price_flag": ["has_reserve_price"],
    "gd_transfer_flag": ["gd_transfer"],
}


def _get_unavailable_filter_entities(missing_cols: Set[str]) -> Set[str]:
    """Return all filter entity names whose source column is missing."""
    unavailable: Set[str] = set()
    for col in missing_cols:
        unavailable.update(_ATHENA_COL_TO_FILTER_ENTITIES.get(col, []))
    return unavailable


_FILTER_CATEGORY_MAP: Dict[str, str] = {
    "keyword_contains": "keywords",
    "keyword_starts_with": "keywords",
    "keyword_ends_with": "keywords",
    "price_min": "price",
    "price_max": "price",
    "tld": "tld",
    "auction_type": "auction_type",
    "time_remaining_max": "time",
    "name_length_max": "name_length",
    "name_length_min": "name_length",
    "quality_min": "quality",
    "has_hyphen": "character",
    "has_number": "character",
    "is_idn": "character",
    "bids_min": "bids",
    "bids_max": "bids",
    "domain_age_min": "age",
    "domain_age_max": "age",
    "traffic_min": "traffic",
    "traffic_max": "traffic",
    "traffic_proxy_min": "traffic",
    "traffic_proxy_max": "traffic",
    "has_web_traffic_signal": "traffic",
    "estimated_traffic_tier_min": "traffic",
    "estimated_traffic_tier_max": "traffic",
    "govalue_min": "govalue",
    "govalue_max": "govalue",
    "majestic_tf_min": "majestic",
    "majestic_tf_max": "majestic",
    "majestic_cf_min": "majestic",
    "majestic_cf_max": "majestic",
    "majestic_backlinks_min": "majestic",
    "majestic_backlinks_max": "majestic",
    "majestic_ref_domains_min": "majestic",
    "majestic_ref_domains_max": "majestic",
    "tlf_exact_match": "tlf",
    "tlf_keyword_regs_min": "tlf",
    "tlf_developed": "tlf",
    "semrush_backlinks_min": "semrush",
    "semrush_backlinks_max": "semrush",
    "semrush_indexed_pages_min": "semrush",
    "semrush_indexed_pages_max": "semrush",
    "semrush_ref_domains_min": "semrush",
    "semrush_ref_domains_max": "semrush",
    "semrush_authority_min": "semrush",
    "semrush_authority_max": "semrush",
    "semrush_search_volume_min": "semrush",
    "semrush_search_volume_max": "semrush",
    "semrush_cpc_min": "semrush",
    "semrush_cpc_max": "semrush",
    "buy_it_now": "auction_type",
    "buy_it_now_min": "price",
    "buy_it_now_max": "price",
    "has_reserve_price": "auction_type",
    "gd_transfer": "auction_type",
    # Absolute timing
    "endTimeAfter": "time_to_end",
    "endTimeBefore": "time_to_end",
    "startTimeAfter": "listing_date",
    "startTimeBefore": "listing_date",
    "days_listed_max": "listing_date",
    "days_listed_min": "listing_date",
    # Auction feature flags
    "isExtended": "auction_type",
    "isBidAccepted": "auction_type",
    # Price extras
    "filterPriceCurrency": "price",
    # Character constraint extras
    "charPattern": "character",
    # Seller / owner
    "ownerMemberIncludeList": "seller",
    "ownerMemberExcludeList": "seller",
    # Demand
    "minUniqueSearches": "demand",
    "maxUniqueSearches": "demand",
    # Estibot
    "minEstibotDomainCount": "estibot",
    "maxEstibotDomainCount": "estibot",
    "minEstibotDomainCountDev": "estibot",
    "maxEstibotDomainCountDev": "estibot",
    "minEstibotExtCount": "estibot",
    "maxEstibotExtCount": "estibot",
    "minEstibotExtCountDev": "estibot",
    "maxEstibotExtCountDev": "estibot",
    # TLD / type exclusions
    "tldExcludeList": "tld",
    "typeExcludeList": "auction_type",
    # Keyword extras
    "keyword_contains_exclude": "keywords",
    "keyword_match_mode": "keywords",
    # Content / word count
    "word_count_min": "content",
    "word_count_max": "content",
    # Semantic (vector-only, no API param)
    "similar_to": "semantic",
    "topic_include": "semantic",
    "topic_exclude": "semantic",
    "lifecycle_state": "lifecycle",
}

# UI taxonomy for active_filters — entity name -> category + human-readable slot label.
_FILTER_UI_MAP: Dict[str, Dict[str, str]] = {
    # 1. Keywords
    "keyword_contains": {"category": "keywords", "label": "contains"},
    "keyword_starts_with": {"category": "keywords", "label": "starts with"},
    "keyword_ends_with": {"category": "keywords", "label": "ends with"},
    "keyword_contains_exclude": {"category": "keywords", "label": "does not contain"},
    "keyword_phrase": {"category": "keywords", "label": "phrase"},
    "keyword_match_mode": {"category": "keywords", "label": "keyword match mode"},
    # 2. Time to end
    "time_remaining_max": {"category": "time_to_end", "label": "max time remaining"},
    # 3. Auction type
    "auction_type": {"category": "type", "label": "auction type"},
    # 4. Price
    "price_min": {"category": "price", "label": "min price"},
    "price_max": {"category": "price", "label": "max price"},
    # 5. Bids
    "bids_min": {"category": "bids", "label": "min bids"},
    "bids_max": {"category": "bids", "label": "max bids"},
    # 6. Extensions (TLD)
    "tld": {"category": "extensions", "label": "extension"},
    # 7. Characters
    "has_hyphen": {"category": "characters", "label": "has hyphen"},
    "has_number": {"category": "characters", "label": "has number"},
    "is_idn": {"category": "characters", "label": "IDN"},
    "name_length_max": {"category": "characters", "label": "max length"},
    "name_length_min": {"category": "characters", "label": "min length"},
    # 8. Age
    "domain_age_min": {"category": "age", "label": "min age (years)"},
    "domain_age_max": {"category": "age", "label": "max age (years)"},
    # 9. Traffic
    "traffic_min": {"category": "traffic", "label": "min traffic"},
    "traffic_max": {"category": "traffic", "label": "max traffic"},
    "traffic_proxy_min": {"category": "traffic", "label": "min traffic score"},
    "traffic_proxy_max": {"category": "traffic", "label": "max traffic score"},
    "has_web_traffic_signal": {"category": "traffic", "label": "has traffic"},
    "estimated_traffic_tier_min": {"category": "traffic", "label": "min traffic tier"},
    "estimated_traffic_tier_max": {"category": "traffic", "label": "max traffic tier"},
    # 10. Estimated value
    "govalue_min": {"category": "estimated_value", "label": "min est. value"},
    "govalue_max": {"category": "estimated_value", "label": "max est. value"},
    # 11. Majestic
    "majestic_tf_min": {"category": "majestic", "label": "TF min"},
    "majestic_tf_max": {"category": "majestic", "label": "TF max"},
    "majestic_cf_min": {"category": "majestic", "label": "CF min"},
    "majestic_cf_max": {"category": "majestic", "label": "CF max"},
    "majestic_backlinks_min": {"category": "majestic", "label": "backlinks min"},
    "majestic_backlinks_max": {"category": "majestic", "label": "backlinks max"},
    "majestic_ref_domains_min": {"category": "majestic", "label": "ref domains min"},
    "majestic_ref_domains_max": {"category": "majestic", "label": "ref domains max"},
    # 12. TLF Insights
    "tlf_exact_match": {"category": "tlf_insights", "label": "exact match TLD"},
    "tlf_keyword_regs_min": {
        "category": "tlf_insights",
        "label": "keyword registrations min",
    },
    "tlf_developed": {"category": "tlf_insights", "label": "developed TLD"},
    # 13. SEMrush
    "semrush_backlinks_min": {"category": "semrush", "label": "backlinks min"},
    "semrush_backlinks_max": {"category": "semrush", "label": "backlinks max"},
    "semrush_indexed_pages_min": {"category": "semrush", "label": "indexed pages min"},
    "semrush_indexed_pages_max": {"category": "semrush", "label": "indexed pages max"},
    "semrush_ref_domains_min": {"category": "semrush", "label": "ref domains min"},
    "semrush_ref_domains_max": {"category": "semrush", "label": "ref domains max"},
    "semrush_authority_min": {"category": "semrush", "label": "authority score min"},
    "semrush_authority_max": {"category": "semrush", "label": "authority score max"},
    "semrush_search_volume_min": {"category": "semrush", "label": "search volume min"},
    "semrush_search_volume_max": {"category": "semrush", "label": "search volume max"},
    "semrush_cpc_min": {"category": "semrush", "label": "CPC min"},
    "semrush_cpc_max": {"category": "semrush", "label": "CPC max"},
    # Quality (internal, no panel equivalent)
    "quality_min": {"category": "quality", "label": "min quality"},
    # 14. Auction feature flags
    "buy_it_now": {"category": "type", "label": "buy it now"},
    "buy_it_now_min": {"category": "price", "label": "min buy-it-now price"},
    "buy_it_now_max": {"category": "price", "label": "max buy-it-now price"},
    "has_reserve_price": {"category": "type", "label": "has reserve price"},
    "gd_transfer": {"category": "type", "label": "GoDaddy transfer included"},
    # 15. Absolute timing (ISO 8601 UTC)
    "endTimeAfter": {"category": "time_to_end", "label": "ends after"},
    "endTimeBefore": {"category": "time_to_end", "label": "ends before"},
    "startTimeAfter": {"category": "listing_date", "label": "listed after"},
    "startTimeBefore": {"category": "listing_date", "label": "listed before"},
    "days_listed_max": {"category": "listing_date", "label": "listed within (days)"},
    "days_listed_min": {
        "category": "listing_date",
        "label": "listed at least (days ago)",
    },
    # 16. Auction type / price extras
    "typeExcludeList": {"category": "type", "label": "exclude auction type"},
    "isExtended": {"category": "type", "label": "extended auction"},
    "isBidAccepted": {"category": "type", "label": "bid accepted"},
    "filterPriceCurrency": {"category": "price", "label": "price currency"},
    # 17. Extensions exclude
    "tldExcludeList": {"category": "extensions", "label": "exclude extension"},
    # 18. Character extras
    "excludeLetters": {"category": "characters", "label": "digits only"},
    "minDigits": {"category": "characters", "label": "min digits"},
    "minLetters": {"category": "characters", "label": "min letters"},
    "charPattern": {"category": "characters", "label": "character pattern"},
    "isGemDomain": {"category": "quality", "label": "gem domain"},
    # 19. Seller / owner
    "ownerMemberIncludeList": {"category": "seller", "label": "seller"},
    "ownerMemberExcludeList": {"category": "seller", "label": "exclude seller"},
    # 20. Demand (unique searches)
    "minUniqueSearches": {"category": "demand", "label": "min unique searches"},
    "maxUniqueSearches": {"category": "demand", "label": "max unique searches"},
    # 21. Estibot counts
    "minEstibotDomainCount": {"category": "estibot", "label": "domain count min"},
    "maxEstibotDomainCount": {"category": "estibot", "label": "domain count max"},
    "minEstibotDomainCountDev": {
        "category": "estibot",
        "label": "domain count (dev) min",
    },
    "maxEstibotDomainCountDev": {
        "category": "estibot",
        "label": "domain count (dev) max",
    },
    "minEstibotExtCount": {"category": "estibot", "label": "extension count min"},
    "maxEstibotExtCount": {"category": "estibot", "label": "extension count max"},
    "minEstibotExtCountDev": {
        "category": "estibot",
        "label": "extension count (dev) min",
    },
    "maxEstibotExtCountDev": {
        "category": "estibot",
        "label": "extension count (dev) max",
    },
}


_SQL_HINT_MAX_CHARS = 300

# Exclude computed (non-column) entities from NL->SQL hints.
_SQL_HINT_EXCLUDE_ENTITIES: frozenset = frozenset({"time_remaining_max"})


# Maps grouping phrases ("by/per/across X") to analytics GROUP BY dimensions.
_GROUP_BY_DIM_PHRASES: List[tuple] = [
    ("auction type", "auction_type_id"),
    ("auction_type", "auction_type_id"),
    ("category", "auction_type_id"),
    ("categories", "auction_type_id"),
    ("extension", "tld"),
    ("tld", "tld"),
    ("user", "user_id"),
    ("buyer", "user_id"),
    ("seller", "user_id"),
]
_GROUP_BY_RE = re.compile(
    r"\b(?:by|per|across|grouped by|broken down by)\s+([a-z _]+)", re.IGNORECASE
)


def _detect_group_by_dims(query: str) -> List[str]:
    """Extract GROUP BY grain columns from 'by/per <dim>' phrases (ordered, de-duplicated)."""
    if not query:
        return []
    dims: List[str] = []
    for m in _GROUP_BY_RE.finditer(query.lower()):
        tail = m.group(1).strip()
        for phrase, col in _GROUP_BY_DIM_PHRASES:
            if tail.startswith(phrase) and col not in dims:
                dims.append(col)
                break
    return dims


def _build_sql_hint(intent: Optional[Any], query: str = "") -> str:
    """Serialize hard-chip entities into hint string (injected into NL-to-SQL prompt); append group_by if present."""
    if intent is None:
        return ""
    parts: List[str] = []
    for s in intent.slices or []:
        for ent in s.entities or []:
            if not ent.name or ent.value is None:
                continue
            if getattr(ent, "chip_kind", "soft") != "hard":
                continue
            # Only pass supported filter slots; exclude-list entities inject contradictory signals into SQL prompt.
            if ent.name not in _FILTER_ENTITY_NAMES:
                continue
            if ent.name in _SQL_HINT_EXCLUDE_ENTITIES:
                continue
            val = ent.value
            if isinstance(val, list):
                val = ",".join(str(v) for v in val)
            parts.append(f"{ent.name}={val}")
    group_by_dims = _detect_group_by_dims(query)
    if group_by_dims:
        parts.append(f"group_by={','.join(group_by_dims)}")
    hint = " ".join(parts)
    return hint[:_SQL_HINT_MAX_CHARS] if len(hint) > _SQL_HINT_MAX_CHARS else hint


def _is_inactive_filter_value(value: Any, slot_name: Optional[str] = None) -> bool:
    """True when an entity value is null/empty/false and should not count as an active filter.

    Only genuinely active (identified) filters belong in query_intelligence.filters.
    Drops None, boolean False, blank strings, and empty containers. Numeric zero (price_max=0)
    is kept — a zero-price ceiling is a real constraint.

    Boolean False is kept when ``slot_name`` uses an invert_bool transform in
    entity_slot_to_api_param.json (e.g. has_number=False -> excludeDigits=True).
    """
    if value is None:
        return True
    if value is False:
        if slot_name and slot_keeps_false_value(slot_name):
            return False
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0
    return False


def _entity_entry(ent: Any, unavailable_entities: Set[str]) -> Dict[str, Any]:
    """Build one filter-summary entry dict from an Entity."""
    is_unavailable = ent.name in unavailable_entities
    _ui = _FILTER_UI_MAP.get(ent.name, {})
    entry: Dict[str, Any] = {
        "name": ent.name,
        "value": ent.value,
        "source": getattr(ent, "source", None),
        "chip_kind": getattr(ent, "chip_kind", "soft"),
        "confidence": getattr(ent, "confidence", None),
        "category": _ui.get("category", _FILTER_CATEGORY_MAP.get(ent.name, "other")),
        "label": _ui.get("label", ent.name),
        "data_status": "unavailable" if is_unavailable else "available",
        "api_param": get_api_param_for_slot(ent.name),
    }
    if is_unavailable:
        entry["message"] = "Column not available in source data — filter skipped"
    return entry


# reconcile_ending_urgency (qi/entity_reconcile.py) may rewrite a raw
# endTimeBefore/endTimeAfter LLM guess to time_remaining_max for the same
# logical filter. pre_ground_entities snapshots the pre-reconcile name, so
# without treating these as one family, the stale name survives in
# ``identified`` alongside the renamed one (two conflicting entries).
_ENDING_URGENCY_SLOT_GROUP = frozenset(
    {"time_remaining_max", "endTimeBefore", "endTimeAfter"}
)


def _build_filter_summary(
    intent: Optional[Any],
    soft_slot_names: Set[str],
    soft_response_key: str,
    guard_outcome: Optional[Any] = None,
    missing_columns: Optional[Set[str]] = None,
    tld_substitutions: Optional[Dict[str, List[str]]] = None,
    backend_unsupported_slots: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Build rich filter summary (identified/soft/not_applied/relaxed/data_availability).

    Soft slots (qi.entity_slots.soft_slot_names) go under soft_response_key only —
    never into identified. Public ``identified`` is the cross-slice **union** of
    FIND filterable API params from **pre-inventory-ground** hard entities (same
    name+value across multi-intent legs is emitted once). Per-slice identified
    lives under ``query_intelligence.sub_intent_filters[].identified``.
    ``applied_filters`` is grounded hard entities that retrieval actually uses
    (available + not guard-dropped + not backend-unsupported). soft_slot_names +
    soft_response_key are required (config-driven).
    ``backend_unsupported_slots`` comes from
    ``retrieval.hard_filter_application.backend_unsupported_slots``.
    """
    if not isinstance(soft_slot_names, (set, frozenset)):
        raise TypeError(
            "_build_filter_summary requires soft_slot_names as set/frozenset from qi.entity_slots"
        )
    if not isinstance(soft_response_key, str) or not soft_response_key.strip():
        raise ValueError(
            "_build_filter_summary requires soft_response_key from qi.entity_slots"
        )
    soft_names = frozenset(soft_slot_names)
    unsupported_names = frozenset(backend_unsupported_slots or ())

    if intent is None:
        return {
            "identified": [],
            soft_response_key: [],
            "not_applied": [],
            "relaxed": None,
            "data_availability": None,
            "applied_filters": [],
            "keywords": [],
        }

    unavailable_entities = _get_unavailable_filter_entities(missing_columns or set())

    # Snapshot A — as identified (pre-ground hard + soft chips).
    # Top-level ``identified`` is a cross-slice union (deduped by name+value).
    # Per-slice identified is emitted only under ``sub_intent_filters``.
    identified: List[Dict[str, Any]] = []
    identified_seen: Set[Tuple[str, str]] = set()
    soft_entries: List[Dict[str, Any]] = []
    soft_seen: Set[Tuple[str, str]] = set()
    for s in intent.slices or []:
        pre = getattr(s, "pre_ground_entities", None)
        if pre is not None:
            hard_for_identified = list(pre)
        else:
            # Legacy / cache without snapshot — fall back to entities (may be grounded).
            hard_for_identified = list(s.entities or [])
        # Cue reconcile (e.g. reconcile_ending_urgency) can rewrite a raw
        # endTimeBefore/endTimeAfter LLM guess to time_remaining_max after
        # pre_ground_entities was snapshotted, appending the renamed entity
        # alongside the stale pre-reconcile one. Keep only the last (reconciled)
        # entry for the family so ``identified`` doesn't carry both.
        _urgency_idxs = [
            i
            for i, e in enumerate(hard_for_identified)
            if e.name in _ENDING_URGENCY_SLOT_GROUP
        ]
        if len(_urgency_idxs) > 1:
            _last = _urgency_idxs[-1]
            hard_for_identified = [
                e
                for i, e in enumerate(hard_for_identified)
                if e.name not in _ENDING_URGENCY_SLOT_GROUP or i == _last
            ]
        # Post-ground cue reconcile may inject hard chips (has_hyphen -> excludeHyphens)
        # onto ``entities`` after ``pre_ground_entities`` was snapshotted — union so
        # public ``filters.identified`` matches qie_only / offline grounding parity.
        _hard_ident_seen: Set[Tuple[str, str]] = {
            (e.name, repr(e.value))
            for e in hard_for_identified
            if getattr(e, "name", None)
        }
        for ent in list(s.entities or []):
            if (
                not ent.name
                or ent.name in soft_names
                or getattr(ent, "chip_kind", None) == "soft"
            ):
                continue
            key = (ent.name, repr(ent.value))
            if key in _hard_ident_seen:
                continue
            if _is_inactive_filter_value(ent.value, slot_name=ent.name):
                continue
            _hard_ident_seen.add(key)
            hard_for_identified.append(ent)
        # Production classify() never duplicates the urgency family inside
        # pre_ground_entities itself (reconcile mutates s.entities, leaving the
        # pre-reconcile snapshot's single stale entry untouched) — the duplicate
        # is introduced right above, when the corrected s.entities member is
        # unioned in under a different (name, value) key. Collapse again here,
        # after the union, keeping the last (corrected) entry for the family.
        _urgency_idxs2 = [
            i
            for i, e in enumerate(hard_for_identified)
            if e.name in _ENDING_URGENCY_SLOT_GROUP
        ]
        if len(_urgency_idxs2) > 1:
            _last2 = _urgency_idxs2[-1]
            hard_for_identified = [
                e
                for i, e in enumerate(hard_for_identified)
                if e.name not in _ENDING_URGENCY_SLOT_GROUP or i == _last2
            ]
        for ent in list(getattr(s, "soft_entities", None) or []) + hard_for_identified:
            if not ent.name or _is_inactive_filter_value(ent.value, slot_name=ent.name):
                continue
            is_soft = (
                ent.name in soft_names or getattr(ent, "chip_kind", None) == "soft"
            )
            if is_soft:
                key = (ent.name, repr(ent.value))
                if key in soft_seen:
                    continue
                soft_seen.add(key)
                soft_entries.append(_entity_entry(ent, unavailable_entities))
                continue
            # Top-level identified is a union across multi-intent slices —
            # same (name, value) from N legs must not repeat (per-slice detail
            # lives under query_intelligence.sub_intent_filters[].identified).
            key = (ent.name, repr(ent.value))
            if key in identified_seen:
                continue
            identified_seen.add(key)
            identified.append(_entity_entry(ent, unavailable_entities))

    # Snapshot B — grounded hard entities (retrieval truth on slice.entities).
    grounded_entries: List[Dict[str, Any]] = []
    grounded_seen: Set[Tuple[str, str]] = set()
    for s in intent.slices or []:
        for ent in list(s.entities or []):
            if not ent.name or _is_inactive_filter_value(ent.value, slot_name=ent.name):
                continue
            if ent.name in soft_names or getattr(ent, "chip_kind", None) == "soft":
                continue
            key = (ent.name, repr(ent.value))
            if key in grounded_seen:
                continue
            grounded_seen.add(key)
            grounded_entries.append(_entity_entry(ent, unavailable_entities))

    grounded_names = {e["name"] for e in grounded_entries}

    relaxed = None
    if guard_outcome is not None and getattr(guard_outcome, "fired", False):
        _dropped_names = list(getattr(guard_outcome, "dropped_filter_names", []))
        relaxed = {
            "guard_fired": True,
            "ladder_step": guard_outcome.ladder_step,
            "original_filter_count": guard_outcome.original_filter_count,
            "relaxed_filter_count": guard_outcome.relaxed_filter_count,
            "filters_dropped": guard_outcome.original_filter_count
            - guard_outcome.relaxed_filter_count,
            "dropped_filter_names": _dropped_names,
            "reason": guard_outcome.relaxation_reason,
        }

    # Data_availability block when enrichment columns missing.
    data_availability: Optional[Dict[str, Any]] = None
    if missing_columns:
        unavailable_cats: Set[str] = set()
        for col in missing_columns:
            for ent_name in _ATHENA_COL_TO_FILTER_ENTITIES.get(col, []):
                cat = _FILTER_CATEGORY_MAP.get(ent_name)
                if cat:
                    unavailable_cats.add(cat)
        data_availability = {
            "missing_source_columns": sorted(missing_columns),
            "unavailable_filter_categories": sorted(unavailable_cats),
            "message": (
                "Some enrichment columns are absent from the source data. "
                "Filters targeting these columns are automatically skipped to prevent zero results. "
                "Run a data-build after the columns become available in the source data."
            ),
        }

    # When guard exhausts full ladder, surface TLD substitutions for UI alternatives (e.g. .ai -> .io/.tech/.app).
    guard_fully_exhausted = (
        guard_outcome is not None
        and getattr(guard_outcome, "fired", False)
        and getattr(guard_outcome, "ladder_step", "") == "explore_fallback"
    )
    if guard_fully_exhausted and tld_substitutions:
        tld_suggestions: Dict[str, List[str]] = {}
        for e in identified:
            if e["name"] == "tld" and e["chip_kind"] == "hard":
                values = e["value"] if isinstance(e["value"], list) else [e["value"]]
                for tld_val in values:
                    sv = str(tld_val).lower()
                    alts = tld_substitutions.get(sv) or []
                    if alts:
                        tld_suggestions[sv] = list(alts)
        if tld_suggestions:
            block: Dict[str, Any] = dict(data_availability) if data_availability else {}
            block["tld_suggestions"] = tld_suggestions
            block.setdefault(
                "message",
                "No domains found for the requested TLD after exhaustive search. "
                "Consider trying the suggested alternative TLDs.",
            )
            data_availability = block

    guard_dropped: set = set((relaxed or {}).get("dropped_filter_names", []))

    # applied_filters = grounded + available + FIND slot + not guard-dropped
    # + not backend-unsupported. Slim shape: name / value / api_param (string).
    applied_filters: List[Dict[str, Any]] = []
    for e in grounded_entries:
        if e["name"] in guard_dropped:
            continue
        if e["name"] in unsupported_names:
            continue
        if (
            e.get("chip_kind") == "hard"
            and e.get("data_status") == "available"
            and e["name"] in _FILTER_ENTITY_NAMES
            and e.get("api_param") is not None
        ):
            _api = get_find_api_param_name(e["name"])
            if _api is None:
                _meta = e.get("api_param")
                if isinstance(_meta, dict):
                    _api = _meta.get("name")
                elif isinstance(_meta, str):
                    _api = _meta
            fe: Dict[str, Any] = {
                "name": e["name"],
                "value": transform_slot_value(e["name"], e["value"]),
            }
            if _api:
                fe["api_param"] = _api
            applied_filters.append(fe)

    applied_names = {e["name"] for e in applied_filters}

    not_applied: List[Dict[str, Any]] = []
    for e in identified:
        is_hard_filter = e["chip_kind"] == "hard" and e["name"] in _FILTER_ENTITY_NAMES
        is_available = e["data_status"] == "available"
        if e["name"] in applied_names:
            continue
        if e["name"] in guard_dropped:
            reason = "filter_relaxed_by_guard"
        elif e["name"] in unsupported_names:
            reason = "backend_unsupported"
        elif is_hard_filter and is_available and e["name"] not in grounded_names:
            reason = "inventory_ungrounded"
        elif e["data_status"] == "unavailable":
            reason = "column_data_unavailable"
        elif e["chip_kind"] == "soft":
            reason = "soft_signal_not_filter"
        else:
            reason = "unsupported_filter_slot"
        # Slim: name / value / reason only.
        not_applied.append(
            {
                "name": e["name"],
                "value": e["value"],
                "reason": reason,
            }
        )

    # Public identified = FIND API params + local hard chips in FILTERABLE_PARAMS.
    # Slim: name + value only (no source / confidence / chip metadata).
    public_identified = _entries_to_public_identified(identified)

    # Soft / relaxed / data_availability kept for internal callers; QI response
    # picks only identified + not_applied (see _build_query_intelligence).
    _STRIP_SOFT = frozenset(
        {"chip_kind", "confidence", "category", "label", "data_status", "api_param"}
    )
    public_soft = [
        {k: v for k, v in e.items() if k not in _STRIP_SOFT} for e in soft_entries
    ]

    return {
        "identified": public_identified,
        soft_response_key: public_soft,
        "not_applied": not_applied,
        "relaxed": relaxed,
        "data_availability": data_availability,
        "applied_filters": applied_filters,
        # Independent of soft_entries — never merged into identified/soft signals.
        "keywords": list(getattr(intent, "keywords", None) or []),
    }


def _slot_value_to_public_identified(
    slot_name: str, value: Any
) -> Optional[Dict[str, Any]]:
    """Map one hard slot/value to the public identified {name, value} shape."""
    if not isinstance(slot_name, str) or not slot_name:
        return None
    if _is_inactive_filter_value(value, slot_name=slot_name):
        return None
    if is_find_api_filter_slot(slot_name):
        api_name = get_find_api_param_name(slot_name)
        if not api_name:
            return None
        return {
            "name": api_name,
            "value": transform_slot_value(slot_name, value),
        }
    if slot_name in _FILTERABLE_PARAM_SET:
        return {"name": slot_name, "value": value}
    return None


def _entries_to_public_identified(
    entries: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Union-dedupe public identified entries from rich filter-summary rows."""
    public: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()
    for e in entries:
        item = _slot_value_to_public_identified(str(e.get("name") or ""), e.get("value"))
        if item is None:
            continue
        key = (str(item["name"]), repr(item["value"]))
        if key in seen:
            continue
        seen.add(key)
        public.append(item)
    return public


def _entities_to_public_identified(
    entities: List[Any],
    soft_slot_names: Set[str],
) -> List[Dict[str, Any]]:
    """Per-slice public identified from Entity list (hard chips only, union-deduped)."""
    soft_names = frozenset(soft_slot_names)
    public: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()
    for ent in entities or []:
        name = getattr(ent, "name", None)
        if not name or name in soft_names or getattr(ent, "chip_kind", None) == "soft":
            continue
        item = _slot_value_to_public_identified(str(name), getattr(ent, "value", None))
        if item is None:
            continue
        key = (str(item["name"]), repr(item["value"]))
        if key in seen:
            continue
        seen.add(key)
        public.append(item)
    return public


def _soft_signal_term_set(soft_signals: List[Dict[str, Any]]) -> Set[str]:
    """Collect casefolded term tokens already listed under soft_signals / soft boost."""
    terms: Set[str] = set()
    for entry in soft_signals or []:
        if not isinstance(entry, dict):
            continue
        raw = entry.get("value")
        values = raw if isinstance(raw, list) else [raw]
        for v in values:
            if v is None:
                continue
            tok = str(v).strip().casefold()
            if tok:
                terms.add(tok)
    return terms


def _build_applied_keywords(
    keywords: List[Dict[str, Any]],
    soft_signals: List[Dict[str, Any]],
    *,
    soft_apply_mode: str,
) -> List[Dict[str, Any]]:
    """Keywords applied via encode (+ soft boost when mode=rank), minus soft-signal dupes.

    Terms already present as soft_signals values (e.g. keyword_contains) are omitted
    so pipeline_trace does not double-list the same lexical constraint.
    """
    mode = str(soft_apply_mode or "").strip().lower()
    soft_terms = _soft_signal_term_set(soft_signals)
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for kw in keywords or []:
        if not isinstance(kw, dict):
            continue
        term = str(kw.get("term") or "").strip()
        if not term:
            continue
        key = term.casefold()
        if key in seen or key in soft_terms:
            continue
        seen.add(key)
        roles: List[str] = ["encode"]
        if mode == "rank":
            roles.append("soft_boost")
        entry: Dict[str, Any] = {"term": term, "roles": roles}
        raw_prob = kw.get("probability")
        if raw_prob is not None:
            try:
                entry["probability"] = float(raw_prob)
            except (TypeError, ValueError):
                pass
        out.append(entry)
    return out


# Fields meaningful only after successful pipeline run (gen -> exec -> verifier); omitted from failures.
_ANALYTICS_SUCCESS_ONLY_FIELDS: frozenset = frozenset(
    {
        "as_of",
        "freshness_lag_seconds",
        "mv_used",
        "as_of_hot",
        "as_of_analytics",
        "schema_version_at_gen",
        "schema_drift_detected",
        "intent_record_id",
        "analytics_substrate",
        "generation",
        "validation",
        "execution",
        "verifier",
    }
)

# Epoch float fields -> ISO 8601 UTC strings in response.
_ANALYTICS_EPOCH_FIELDS: frozenset = frozenset(
    {"created_at", "as_of", "as_of_hot", "as_of_analytics"}
)


def _format_analytics_timestamps(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Unix epoch floats -> ISO 8601 UTC strings."""
    out = dict(data)
    for key in _ANALYTICS_EPOCH_FIELDS:
        if key in out and out[key] is not None:
            out[key] = _format_ends_at(out[key])
    return out


def _slim_analytics_failure(data: Dict[str, Any]) -> Dict[str, Any]:
    """Strip success-only fields from a failed analytics result dict."""
    return {k: v for k, v in data.items() if k not in _ANALYTICS_SUCCESS_ONLY_FIELDS}


def _build_data_window_notice(failure_reason: str) -> Dict[str, Any]:
    """Parse time_window_exceeds_max failure_reason into user-facing data window block."""
    parts = {
        seg.split("=")[0]: seg.split("=")[1]
        for seg in failure_reason.split()
        if "=" in seg and len(seg.split("=")) == 2
    }
    req = int(parts["requested_days"]) if "requested_days" in parts else None
    avail = int(parts["max_days"]) if "max_days" in parts else None
    if req is not None and avail is not None:
        notice = f"Data available for up to {avail} days. Query requested {req} days — reduce the time range to {avail} days or fewer."
    else:
        notice = (
            "Query time range exceeds the available data window. Reduce the time range."
        )
    return {"notice": notice, "available_days": avail, "requested_days": req}


def _compute_retrieval_metrics(
    top_k_norm_scores: List[float],
    pool_norm_scores: List[float],
    top_k: int,
    relevance_threshold: float,
) -> Dict[str, Any]:
    """Compute label-free retrieval metrics (NDCG/Coherence/Recall/Precision) from normalized scores; all 4 dp, bounded [0,1]."""
    if not top_k_norm_scores:
        return {
            f"NDCG@{top_k}": 0.0,
            f"Recall@{top_k}": 0.0,
            f"Precision@{top_k}": 0.0,
            f"Coherence@{top_k}": 0.0,
        }
    ranks = list(range(1, len(top_k_norm_scores) + 1))
    dcg = sum(s / math.log2(r + 1) for s, r in zip(top_k_norm_scores, ranks))
    idcg = sum(
        s / math.log2(r + 1)
        for s, r in zip(sorted(top_k_norm_scores, reverse=True), ranks)
    )
    ndcg = round(dcg / idcg, 4) if idcg > 0 else 0.0
    coherence = round(sum(top_k_norm_scores) / len(top_k_norm_scores), 4)
    relevant_in_top_k = sum(1 for s in top_k_norm_scores if s >= relevance_threshold)
    relevant_in_pool = sum(1 for s in pool_norm_scores if s >= relevance_threshold)
    # Cap denominator to returned items; small corpus not unfairly penalized for unfillable slots.
    precision_denom = min(top_k, len(top_k_norm_scores))
    precision = (
        round(relevant_in_top_k / float(precision_denom), 4)
        if precision_denom > 0
        else 0.0
    )
    recall = (
        round(relevant_in_top_k / float(relevant_in_pool), 4)
        if relevant_in_pool > 0
        else 0.0
    )
    return {
        f"NDCG@{top_k}": ndcg,
        f"Recall@{top_k}": recall,
        f"Precision@{top_k}": precision,
        f"Coherence@{top_k}": coherence,
    }


def _add_llm_cost_usd(qi: Dict[str, Any], extra_usd: float) -> None:
    """Add phase LLM spend into ``qi['decision_cost_usd']`` (mutates ``qi``)."""
    try:
        extra = float(extra_usd or 0.0)
    except (TypeError, ValueError):
        return
    if extra <= 0.0:
        return
    try:
        base = float(qi.get("decision_cost_usd") or 0.0)
    except (TypeError, ValueError):
        base = 0.0
    qi["decision_cost_usd"] = base + extra


def _build_query_intelligence(
    intent: Optional[QueryIntent],
    filter_summary: Dict[str, Any],
    *,
    multi_intent_envelope: Optional[Any] = None,
    cache_hit: Optional[str] = None,
    soft_slot_names: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Build slim query_intelligence envelope for the search response.

    Kept: classified_intent, decision_tier, intent_confidence,
    decision_cost_usd (all LLM spend this request: L0 + L2 + rewrite + …),
    filters (identified + soft_signals + not_applied), query_transform when present,
    multi_intent_envelope + sub_intent_filters when multi-intent split ran.
    Dropped from response: relaxed, data_availability,
    cache_hit, routing_applied. ``applied_filters`` lives on pipeline_trace.
    Soft slots under ``filters.<soft_response_key>`` for L0 parity with qie_only.
    ``qie_only_mode`` uses a separate slim L0 path and does not call this.

    Top-level ``filters.identified`` is the cross-slice union. Per-slice public
    identified lives only under ``sub_intent_filters[].identified`` (no internal
    ``entities`` — that would duplicate the same slots under internal names).
    """
    _ = cache_hit  # call-site compat; omitted from slim body
    n_slices = len(intent.slices) if intent and intent.slices else 0
    qt = intent.query_type if intent else "unknown"
    out: Dict[str, Any] = {
        "classified_intent": qt,
        "intent_confidence": intent.confidence if intent else None,
        # Never emit a blank tier: missing intent / unset tier -> 'fallback'.
        "decision_tier": (
            intent.decision_tier if (intent and intent.decision_tier) else "fallback"
        ),
        "decision_cost_usd": float(intent.decision_cost_usd)
        if intent is not None
        else 0.0,
        "multi_intent": n_slices > 1,
        "filters": {
            "identified": list(filter_summary.get("identified") or []),
            "not_applied": list(filter_summary.get("not_applied") or []),
            **{
                k: list(v or [])
                for k, v in filter_summary.items()
                if k
                not in (
                    "identified",
                    "not_applied",
                    "relaxed",
                    "data_availability",
                    "applied_filters",
                )
                and isinstance(v, list)
            },
        },
    }
    if multi_intent_envelope is not None:
        out["multi_intent_envelope"] = {
            k: v
            for k, v in asdict(multi_intent_envelope).items()
            if k != "per_intent_chips"
        }
    if intent and intent.sub_intent_filters:
        # Per-sub-query public identified only (internal Entity list stays off-wire).
        _soft = set(soft_slot_names or ())
        out["sub_intent_filters"] = [
            {
                "sub_query": sif.sub_query,
                "identified": _entities_to_public_identified(
                    list(sif.entities or []), soft_slot_names=_soft
                ),
            }
            for sif in intent.sub_intent_filters
        ]
    _qt = intent.query_transform if intent else None
    if _qt:
        out["query_transform"] = _qt
    return out


def _ranked_results_role_for_query_type(query_type: Optional[str]) -> str:
    """CH archetypes use ranked_results as filter-scoped examples, not the primary answer."""
    if query_type in ("analytics", "guidance"):
        return "examples"
    return "primary"


def _build_pipeline_trace(
    filter_summary: Dict[str, Any],
    analytics_outcome: str,
    outcome: Any,
    guard_outcome: Any,
    result_count: Optional[int],
    query_type: Optional[str] = None,
    *,
    sub: Any = None,
) -> Dict[str, Any]:
    """Slim pipeline_trace — applied_filters + applied_keywords + ranked_results role.

    When ``measurement.ranking_stage_attribution`` is enabled and
    ``include_in_response`` is true, attach ``stages`` from the orchestrator
    (retrieve / fuse / hard-gate / soft / eRanker / diversify / zero-result).
    Extra args kept for call-site compat.
    """
    _ = analytics_outcome, outcome, guard_outcome, result_count
    soft_apply_mode = "off"
    soft_response_key = "soft_signals"
    if sub is not None:
        _slots = getattr(getattr(sub.config, "qi", None), "entity_slots", None)
        if _slots is not None:
            soft_apply_mode = str(getattr(_slots, "soft_apply_mode", "off") or "off")
            soft_response_key = str(
                getattr(_slots, "soft_response_key", "soft_signals") or "soft_signals"
            )
    soft_signals = list(filter_summary.get(soft_response_key) or [])
    applied_keywords = _build_applied_keywords(
        list(filter_summary.get("keywords") or []),
        soft_signals,
        soft_apply_mode=soft_apply_mode,
    )
    out: Dict[str, Any] = {
        "applied_filters": filter_summary.get("applied_filters", []),
        "applied_keywords": applied_keywords,
        "ranked_results_role": _ranked_results_role_for_query_type(query_type),
    }
    if sub is not None:
        attr = sub.config.measurement.ranking_stage_attribution
        if attr.enabled and attr.include_in_response:
            stages = sub.orchestrator.last_ranking_stages
            if stages:
                out["stages"] = stages
    return out


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Redirect the root path to the Swagger UI."""
    return RedirectResponse(url="/docs")


@app.get("/healthz", tags=["Status"])
@app.get("/health", tags=["Status"])
def healthz() -> Dict[str, Any]:
    """Liveness probe."""
    return {
        "status": "healthy",
        "service": "semantic_search",
        "encoder_backend": "hashing_degraded"
        if app_state.encoder_degraded
        else "fastembed",
    }


def _llm_escalation_available(sub: Subsystems) -> bool:
    """True when provider key validated and model registry non-empty."""
    prov = sub.llm_provider
    if prov is None:
        return False
    try:
        return int(prov.get_summary().get("available_models", 0)) > 0
    except (TypeError, ValueError, KeyError, AgentSearchError, RuntimeError, OSError):
        return False


def _log_qie_only_complete(
    *,
    request_id: str,
    query: str,
    latency_ms: float,
    filter_count: int,
    hard_filter_count: int,
    source: str,
    model_id: str,
    token_count: int,
    decision_cost_usd: float,
    grounded_drop_count: int,
    prompt_tag: str,
    schema_version: str,
    body: Dict[str, Any],
) -> None:
    """Emit structured ``qie_only_complete`` line + update launch stats."""
    ops_m = _qie_only_ops_metrics(body)
    ops = (
        f"find_skipped_count={ops_m['find_skipped_count']} "
        f"find_skipped_reasons={ops_m['find_skipped_reasons']} "
        f"soft_chip_count={ops_m['soft_chip_count']} "
        f"hard_params_empty={ops_m['hard_params_empty']}"
    )
    logger.info(
        f"qie_only_complete request_id={request_id} query_len={len(query)} "
        f"latency_ms={latency_ms} decision_tier=L0_entity "
        f"filter_count={filter_count} hard_filter_count={hard_filter_count} "
        f"source={source} "
        f"model_id={model_id or '-'} token_count={token_count} "
        f"decision_cost_usd={decision_cost_usd:.6f} "
        f"grounded_drop_count={grounded_drop_count} "
        f"prompt_tag={prompt_tag} schema_version={schema_version} "
        f"{ops}"
    )
    try:
        from semantic_search.measurement.qie_only_launch import get_qie_only_launch_stats

        get_qie_only_launch_stats().record_complete(
            latency_ms=float(latency_ms),
            find_skipped_count=int(ops_m["find_skipped_count"]),
            hard_params_empty=int(ops_m["hard_params_empty"]),
            source=str(source),
        )
    except Exception as exc:  # noqa: BLE001 — stats must never break response
        logger.warning(
            f"qie_only_launch_stats_record_failed request_id={request_id} "
            f"error_type={type(exc).__name__}"
        )


def _log_qie_only_failed(
    *,
    request_id: str,
    status: int,
    reason: str,
    error_type: str = "-",
) -> None:
    """Emit ``qie_only_failed`` + update launch stats (availability gate)."""
    logger.warning(
        f"qie_only_failed request_id={request_id} status={int(status)} "
        f"reason={reason} error_type={error_type or '-'}"
    )
    try:
        from semantic_search.measurement.qie_only_launch import get_qie_only_launch_stats

        get_qie_only_launch_stats().record_failed(
            status=int(status), reason=str(reason or "")
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"qie_only_launch_stats_failed_record_error request_id={request_id} "
            f"error_type={type(exc).__name__}"
        )


@app.get("/capabilities", tags=["Status"])
def capabilities() -> Dict[str, Any]:
    """Snapshot of which subsystems are wired and available."""
    sub = _require_subsystems()
    return {
        "qi_engine": True,
        "qi_llm_tier": _llm_escalation_available(sub),
        "vector_retriever": sub.config.retrieval.vector.enabled,
        "structured_retriever": sub.config.retrieval.structured.enabled,
        "sql_retriever": sub.config.retrieval.sql.enabled,
        "cache": sub.config.cache.enabled,
        "surface": sub.config.surface.enabled,
        "feedback": sub.config.feedback.enabled,
        "eranker": sub.config.retrieval.eranker.enabled,
        "history": sub.config.history.enabled,
        "ingress_sanitizer": sub.config.safety.ingress_sanitizer.enabled,
        "offline_eval": {
            "retrieval_eval": sub.config.offline_eval.retrieval_eval.enabled,
            "llm_judge": sub.config.offline_eval.llm_judge.enabled
            if sub.config.offline_eval.llm_judge is not None
            else False,
        },
        "resilience": {
            "circuit_breaker": sub.config.resilience.circuit_breaker.enabled,
            "backend_health": sub.config.resilience.backend_health.enabled,
            "degradation": sub.config.resilience.degradation.enabled,
        },
        "pipeline": {
            "post_search": True,
            "post_analytics": sub.orchestrator.analytics_available,
            "qi_classify": True,
            "guidance_wired": sub.guidance_service is not None,
            "explore_enabled": sub.config.explore.enabled,
            "vectorization_wired": sub.vector_refresh_driver is not None,
        },
    }


async def _search_impl(
    query: str = Form(
        ...,
        description="Search query or analytics question — QI auto-classifies and routes internally",
    ),
    top_k: int = Form(
        50,
        ge=1,
        le=100,
        description="Number of ranked results to return (1–100, default 50). Capped by general.search.top_k_cap in config (default 100). Not applied to analytics queries.",
    ),  # noqa: E501
    diversity_lambda: float = Form(
        0.9,
        ge=0.0,
        le=1.0,
        description="MMR lambda for result diversity (0=max diversity, 1=max relevance). Overrides config value when provided.",
    ),
    relevance_threshold: float = Form(
        0.7,
        ge=0.0,
        le=1.0,
        description="Score threshold for labelling a result relevant in metrics (Recall, Precision, HitRate). Overrides retrieval.metrics.relevance_threshold in config when provided.",
    ),  # noqa: E501
    qie_only_mode: Optional[bool] = Form(
        False,
        description="Default false. When true, runs L0 LLM filter extract only (no retrieval). Returns slim JSON: identified_filters, find_query_params/find_query_string (FIND wire), soft_chips, decision_tier=L0_entity, latency_ms.",
    ),  # noqa: E501
    x_session_id: Optional[str] = Header(
        None,
        alias="X-Session-Id",
        description="Per-session rate-limit key. Window: 10 req/60 s. Omit to fall back to ip-keyed bucket.",
    ),
    request_id: Optional[str] = None,
    search_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Unified search — one input box, fully autonomous routing.

    **SLA Guarantees:**

    - ``hybrid``, ``explore``, ``guidance`` — **10-second wall-clock SLA**
      (``search_timeout_seconds``) enforced via ``asyncio.wait_for``; on breach the
      pre-computed explore fallback (trending/ending-soon) is returned with
      ``retrieval_metrics.failure_mode='timeout'``. ``answer_mode`` follows the
      parent routing tier (L1 preview on outer SLA breach; classified intent
      otherwise): ``hybrid``->``search``, ``explore``->``explore``,
      ``guidance``->``guidance``, ``analytics``->``analytics``. Unknown / missing
      type -> ``explore_fallback``. Explore rails fill the body only — empty
      inventory or substrate failure is not a misroute.
    - ``analytics`` — **20-second total wall-clock budget** from request start
      (``analytics_total_budget_seconds``), covering QI classification + routing +
      ClickHouse execution.  The per-call analytics timeout (``analytics_timeout_seconds``)
      is 18 s, but the remaining budget after classification is used when it is shorter.
      On budget exhaustion: keep hybrid ``ranked_results`` when nonempty, else
      timeout ladder (semantic / explore) fills listings — both keep
      ``answer_mode='analytics'`` with ``failure_mode='analytics_timeout'``.

    **Query Intelligence (QI)** classifies every query through a single structured LLM call
    that returns query type, extracted entities, and confidence in one shot.

    **Routing** (hybrid-first ranked_results for every intent):
    - ``hybrid`` -> hybrid RRF (vector + structured + optional SQL price fan-out)
    - ``explore`` -> hybrid ranks + ClickHouse explore rails RRF-merged when CH up
    - ``guidance`` -> hybrid ranks + market snapshot envelope when CH up
    - ``analytics`` -> hybrid ranks + ClickHouse multi-period / NL-to-SQL in ``analytics`` block

    **Analytics — multi-period aggregation (no LLM required):**
    When QI classifies the query as ``analytics`` and ``sql_hint`` contains a recognisable
    dimension (e.g. ``auction_type``, ``tld``), the multi-period handler fires **4 parallel
    ClickHouse queries** — ``last_1h``, ``last_24h``, ``last_7d``, ``last_30d`` — using
    pre-validated SQL templates derived from the MV catalog.  No schema discovery, no LLM
    generation call.  Response includes ``analytics.periods`` with per-window rows, column
    names, row count, latency, and source (MV name or ``raw``).

    Example: *"average price of .com auctions by auction type"* ->
    ``analytics.periods.last_7d.rows = [{auction_type_id: 16, avg_price: 45.2, ...}, ...]``

    When no dimension is detected or all periods return 0 rows, the request falls through
    to the NL-to-SQL pipeline.

    **Response layout (full search, slim):**
    - ``query_intelligence`` — ``classified_intent``, ``intent_confidence``,
      ``decision_tier``, ``decision_cost_usd`` (all LLM spend this request), ``multi_intent``,
      ``filters.identified`` (as-identified, name/value),
      ``filters.not_applied`` (name/value/reason), optional
      ``multi_intent_envelope`` / ``sub_intent_filters`` (post-split slices),
      optional ``query_transform``
    - ``pipeline_trace.applied_filters`` — grounded filters used for retrieval
      (name/value/api_param string); optional ``pipeline_trace.stages`` when
      ``measurement.ranking_stage_attribution.include_in_response`` is true
    - ``ranked_results`` — hybrid-first top ``top_k`` (rank, domain_name,
      coherence_score, matched_by, plus ``general.search.result_fields``)
    - ``retrieval_metrics`` — total_candidates, failure_mode, params, NDCG@K /
      Coherence@K / Recall@K / Precision@K / HitRate@K
    - ``latency_ms`` — wall-clock request latency
    - ``analytics`` — when query_type is ``analytics`` and CH available

    **Response layout (``qie_only_mode=true``):** Same preprocess as full search
      (``extract_filters_only``: spell + token-gated combined rewrite+extract or
      extract-only), then L0 ground. No retrieval. Returns ``answer_mode='qie_only'``,
      ``identified_filters`` (post-ground), ``pre_ground_identified`` (pre-inventory
      ground — same shape as ``identified_filters``; harness diffs for drops),
      ``keywords`` (term/probability >=
      ``qi.l0_llm_entity.keyword_min_probability``), optional ``query_transform``
      when rewrite accepted, ``find_query_params`` / ``find_query_string``,
      ``soft_chips``, ``find_skipped``, ``decision_tier='L0_entity'``,
      ``latency_ms``, ``decision_cost_usd``, ``grounded_drop_count``,
      ``prompt_tag``.
		    """
    t_start = time.monotonic()
    sub = _require_subsystems()
    search_cfg = sub.config.general.search
    top_k = min(top_k, search_cfg.top_k_cap)
    effective_diversity_lambda = diversity_lambda
    effective_relevance_threshold = relevance_threshold
    # Score->[0,1] normalization mode for coherence_score and the threshold-based
    # quality metrics. Config-driven ('max' legacy | 'minmax' sharper); threaded
    # to every _normalise_scores call so all response paths stay consistent.
    _norm_mode = sub.config.retrieval.metrics.score_normalization
    user_ctx: Optional[UserContext] = None
    if x_session_id:
        user_ctx = UserContext(
            user_id=None, is_authenticated=False, session_id=x_session_id
        )
    # Speculative explore fallback: fire in parallel; ready before 5s SLA fires. Cancelled if search succeeds.
    # Identity: request_id = trace hop; search_id = durable search interaction (from search() wrapper).
    id_cfg = sub.config.identity
    if isinstance(request_id, str) and request_id.strip():
        _rid = request_id.strip()
    else:
        _rid = mint_prefixed_id(id_cfg.request_id_prefix, id_cfg.id_hex_length)
    if isinstance(search_id, str) and search_id.strip():
        _sid = search_id.strip()
    else:
        _sid = mint_prefixed_id(id_cfg.search_id_prefix, id_cfg.id_hex_length)
    # ── QIE-only mode (per-request, default false) ────────────────────────────
    # L0 LLM filter extract + static inventory ground (tld/type only) — no L1/L2 or retrieval.
    # None Form value falls back to config.
    _qie_only_effective = (
        search_cfg.qie_only_mode if qie_only_mode is None else qie_only_mode
    )
    if _qie_only_effective:
        # Same preprocess + L0 path as full search via extract_filters_only.
        # No duplicate transform/extract stack in this handler.
        _qie_sanitizer = getattr(sub, "sanitizer", None)
        if _qie_sanitizer is not None and _qie_sanitizer.applies_to_llm_ingress:
            _qie_verdict = _qie_sanitizer.sanitize(query)
            if not _qie_verdict.passed:
                logger.warning(
                    f"user_input_blocked_by_sanitizer surface=search_qie_only request_id={_rid} reasons={_qie_verdict.reasons}"
                )
                raise HTTPException(
                    status_code=422,
                    detail=f"user_input_blocked_by_sanitizer surface=search reasons={_qie_verdict.reasons}",
                )
        _qi_engine = getattr(sub, "qi_engine", None)
        (
            _qie_l0_extractor_for_reconcile,
            _qie_hard_names,
            _qie_soft_names,
            _qie_grounder,
        ) = _qie_grounding_context(sub)
        _qi_cfg = getattr(sub.config, "qi", None)
        _l0_cfg = (
            getattr(_qi_cfg, "l0_llm_entity", None) if _qi_cfg is not None else None
        )
        if _l0_cfg is None:
            raise HTTPException(
                status_code=503,
                detail="qi.l0_llm_entity required for qie_only_mode",
            )
        _qt_cfg = getattr(_qi_cfg, "query_transformer", None)
        _qie_normalized = normalize_query(
            query,
            sub.config.general.max_query_length,
            normalize=_qi_cfg.normalize,
        )
        # Token-gate cache tags (same gate as QueryTransformer.needs_rewrite).
        _combine_on = (
            _qt_cfg is not None
            and isinstance(
                getattr(_qt_cfg, "combine_rewrite_with_l0_extract", None), bool
            )
            and bool(_qt_cfg.combine_rewrite_with_l0_extract)
        )
        _rewrite_enabled = (
            _qt_cfg is not None
            and isinstance(getattr(_qt_cfg, "rewrite_enabled", None), bool)
            and bool(_qt_cfg.rewrite_enabled)
        )
        _rewrite_threshold = (
            int(getattr(_qt_cfg, "rewrite_threshold", 0) or 0)
            if _qt_cfg is not None
            else 0
        )
        _token_gate = (
            _combine_on
            and _rewrite_enabled
            and _rewrite_threshold >= 1
            and len(_qie_normalized.split()) > _rewrite_threshold
        )
        _qie_prompt_tag = (
            str(_l0_cfg.combined_prompt_tag) if _token_gate else str(_l0_cfg.prompt_tag)
        )
        _qie_schema_version = (
            str(_l0_cfg.combined_schema_version)
            if _token_gate
            else str(_l0_cfg.schema_version)
        )
        _qie_cache_key = exact_query_key(
            versioned_query_key(
                _qie_normalized,
                prompt_tag=_qie_prompt_tag,
                schema_version=_qie_schema_version,
            )
        )
        _qie_cached = _get_qie_l0_filter_cache().get(_qie_cache_key)
        _qie_unpacked = _unpack_qie_cache_entry(_qie_cached)
        if _qie_cached is not None and _qie_unpacked is None:
            logger.info(
                f"qie_only_l0_cache hit=false reason=empty_entry_ignored "
                f"key_preview={_qie_cache_key[:12]} request_id={_rid}"
            )
            _get_qie_l0_filter_cache().invalidate(_qie_cache_key)
        if _qie_unpacked is not None:
            _qie_identified, _qie_cached_keywords, _qie_cached_qt = (
                copy.deepcopy(_qie_unpacked[0]),
                copy.deepcopy(_qie_unpacked[1]),
                copy.deepcopy(_qie_unpacked[2]) if _qie_unpacked[2] is not None else None,
            )
            # Cache stores pre-ground extract; snapshot before reconcile+ground.
            _qie_pre_identified = copy.deepcopy(_qie_identified)
            _qie_identified, _grounded_drop_count = reconcile_and_ground_identified(
                _qie_identified,
                _qie_normalized,
                extractor=_qie_l0_extractor_for_reconcile,
                hard_names=_qie_hard_names,
                soft_names=_qie_soft_names,
                grounder=_qie_grounder,
            )
            _qie_elapsed = round((time.monotonic() - t_start) * 1000, 1)
            logger.info(
                f"qie_only_l0_cache hit=true key_preview={_qie_cache_key[:12]} "
                f"request_id={_rid} filter_count={len(_qie_identified)} "
                f"keywords={len(_qie_cached_keywords)} "
                f"transformed={bool((_qie_cached_qt or {}).get('transformed'))}"
            )
            _hard_n = sum(
                1 for f in _qie_identified if str(f.get("chip_kind") or "") == "hard"
            )
            if isinstance(_qie_cached_qt, dict) and bool(_qie_cached_qt.get("transformed")):
                _qie_prompt_tag = str(_l0_cfg.combined_prompt_tag)
                _qie_schema_version = str(_l0_cfg.combined_schema_version)
            _qie_cache_body = {
                "query": query,
                "request_id": _rid,
                "search_id": _sid,
                "answer_mode": "qie_only",
                "latency_ms": _qie_elapsed,
                "decision_tier": "L0_entity",
                "decision_cost_usd": 0.0,
                "identified_filters": _qie_identified,
                "pre_ground_identified": _qie_pre_identified,
                "keywords": _qie_cached_keywords,
                "grounded_drop_count": _grounded_drop_count,
                "prompt_tag": _qie_prompt_tag,
                "schema_version": _qie_schema_version,
            }
            if isinstance(_qie_cached_qt, dict) and bool(_qie_cached_qt.get("transformed")):
                _qie_cache_body["query_transform"] = {
                    "mode": str(_qie_cached_qt.get("mode") or ""),
                    "engine": str(_qie_cached_qt.get("engine") or ""),
                    "transformed": True,
                    "transformed_query": str(
                        _qie_cached_qt.get("transformed_query")
                        or _qie_cached_qt.get("query")
                        or ""
                    ),
                }
            _attach_find_wire_fields(
                _qie_cache_body,
                _qie_identified,
                keywords=_qie_cached_keywords,
                get_subsystems=_require_subsystems,
            )
            _log_qie_only_complete(
                request_id=_rid,
                query=query,
                latency_ms=_qie_elapsed,
                filter_count=len(_qie_identified),
                hard_filter_count=_hard_n,
                source="L0_llm_cache",
                model_id="-",
                token_count=0,
                decision_cost_usd=0.0,
                grounded_drop_count=_grounded_drop_count,
                prompt_tag=_qie_prompt_tag,
                schema_version=_qie_schema_version,
                body=_qie_cache_body,
            )
            return _round_floats(_qie_cache_body, _RESPONSE_FLOAT_PRECISION)

        try:
            _qie_intent = await sub.orchestrator.extract_filters_only(
                query, request_id=_rid,
            )
        except ConfigurationError as _qie_ce:
            _log_qie_only_failed(
                request_id=_rid,
                status=503,
                reason="configuration_error",
                error_type=type(_qie_ce).__name__,
            )
            raise HTTPException(status_code=503, detail=str(_qie_ce)) from _qie_ce
        except ValidationError as _qie_ve:
            _log_qie_only_failed(
                request_id=_rid,
                status=422,
                reason="validation_error",
                error_type=type(_qie_ve).__name__,
            )
            raise HTTPException(status_code=422, detail=str(_qie_ve)) from _qie_ve
        except AgentSearchError as _qie_ae:
            _log_qie_only_failed(
                request_id=_rid,
                status=503,
                reason="query_intelligence_unavailable",
                error_type=type(_qie_ae).__name__,
            )
            raise HTTPException(
                status_code=503,
                detail=f"query_intelligence_unavailable: {_qie_ae}",
            ) from _qie_ae

        _qie_qt = getattr(_qie_intent, "query_transform", None) or {}
        _qie_transformed = bool(_qie_qt.get("transformed")) if isinstance(_qie_qt, dict) else False
        if _qie_transformed:
            _qie_prompt_tag = str(_l0_cfg.combined_prompt_tag)
            _qie_schema_version = str(_l0_cfg.combined_schema_version)
            _qie_cache_key = exact_query_key(
                versioned_query_key(
                    _qie_normalized,
                    prompt_tag=_qie_prompt_tag,
                    schema_version=_qie_schema_version,
                )
            )
        _qie_identified, _qie_pre_identified, _grounded_drop_count = (
            _qie_identified_from_intent(_qie_intent, _qie_soft_names)
        )
        # Cue-reconcile + inventory-ground against the *original* normalized query,
        # starting from the pre-ground snapshot (same as cache-hit path). Grounding
        # already-grounded chips under-counts drops and diverges from /internal/l0_ground.
        _qie_identified, _grounded_drop_count = reconcile_and_ground_identified(
            _qie_pre_identified,
            _qie_normalized,
            extractor=_qie_l0_extractor_for_reconcile,
            hard_names=_qie_hard_names,
            soft_names=_qie_soft_names,
            grounder=_qie_grounder,
        )
        _qie_keywords = list(getattr(_qie_intent, "keywords", None) or [])
        _qie_cost_usd = float(getattr(_qie_intent, "decision_cost_usd", 0.0) or 0.0)
        _qie_source = "L0_llm"
        if _qie_identified:
            _src0 = str(_qie_identified[0].get("source") or "").strip()
            if _src0:
                _qie_source = _src0

        # Inventory-bound + empty L0: regex recovery (same contract as prior app path).
        # Do NOT recover when advisory/analytics speech acts intentionally wiped
        # filters — regex reinject of buy_it_now / lifecycle undoes that wipe.
        _regex_cfg = getattr(_qi_cfg, "l0_regex_entity", None)
        _rex = getattr(_qi_engine, "_regex_entity_extractor", None) if _qi_engine else None
        _inventory_bound = bool(INVENTORY_BOUND_RE.search(query or ""))
        _advisory_empty = is_strong_advisory(query or "") or is_soft_advisory_no_inventory(
            query or ""
        )
        if (
            not _qie_identified
            and _inventory_bound
            and not _advisory_empty
            and _regex_cfg is not None
            and bool(_regex_cfg.enabled)
            and _rex is not None
        ):
            _qie_extract_text = (
                str(_qie_qt.get("transformed_query") or "").strip()
                if _qie_transformed
                else _qie_normalized
            )
            if not _qie_extract_text:
                _qie_extract_text = _qie_normalized
            logger.warning(
                f"qie_only_l0_llm_empty_inventory_bound request_id={_rid} "
                f"action=regex_fallback query_len={len(query)}"
            )
            try:
                _qie_raw, _, _qie_keywords = await L0RegexFilterExtractor(
                    _rex
                ).extract_priced(_qie_extract_text)
                _qie_pre_identified = copy.deepcopy(_qie_raw)
                _qie_identified, _grounded_drop_count = reconcile_and_ground_identified(
                    _qie_raw,
                    _qie_extract_text,
                    extractor=_qie_l0_extractor_for_reconcile,
                    hard_names=_qie_hard_names,
                    soft_names=_qie_soft_names,
                    grounder=_qie_grounder,
                )
                _qie_source = "L0_regex"
                _qie_cost_usd = 0.0
            except (ConfigurationError, AgentSearchError) as _qie_rex_e:
                logger.warning(
                    f"qie_only_inventory_regex_failed request_id={_rid} "
                    f"error_type={type(_qie_rex_e).__name__}"
                )

        _llm_source = _qie_source.startswith("L0_llm")
        _qie_qt_for_cache: Optional[Dict[str, Any]] = None
        if _qie_transformed and isinstance(_qie_qt, dict):
            _qie_qt_for_cache = {
                "mode": str(_qie_qt.get("mode") or ""),
                "engine": str(_qie_qt.get("engine") or ""),
                "transformed": True,
                "transformed_query": str(
                    _qie_qt.get("transformed_query") or _qie_qt.get("query") or ""
                ),
            }
        if _llm_source and (_qie_pre_identified or _qie_keywords or _qie_qt_for_cache):
            _get_qie_l0_filter_cache().put(
                _qie_cache_key,
                _qie_cache_envelope(
                    copy.deepcopy(_qie_pre_identified),
                    copy.deepcopy(_qie_keywords),
                    copy.deepcopy(_qie_qt_for_cache) if _qie_qt_for_cache else None,
                ),
            )
        elif _llm_source and not _qie_pre_identified and not _qie_keywords:
            logger.info(
                f"qie_only_l0_cache put=skipped reason=empty_extract "
                f"key_preview={_qie_cache_key[:12]} request_id={_rid}"
            )

        _qie_elapsed = round((time.monotonic() - t_start) * 1000, 1)
        _hard_n = sum(
            1 for f in _qie_identified if str(f.get("chip_kind") or "") == "hard"
        )
        _qie_body: Dict[str, Any] = {
            "query": query,
            "request_id": _rid,
            "search_id": _sid,
            "answer_mode": "qie_only",
            "latency_ms": _qie_elapsed,
            "decision_tier": "L0_entity",
            "decision_cost_usd": _qie_cost_usd,
            "identified_filters": _qie_identified,
            "pre_ground_identified": list(_qie_pre_identified or []),
            "keywords": _qie_keywords,
            "grounded_drop_count": _grounded_drop_count,
            "prompt_tag": _qie_prompt_tag,
            "schema_version": _qie_schema_version,
        }
        if _qie_qt_for_cache is not None:
            _qie_body["query_transform"] = _qie_qt_for_cache
        _attach_find_wire_fields(
            _qie_body,
            _qie_identified,
            keywords=_qie_keywords,
            get_subsystems=_require_subsystems,
        )
        _log_qie_only_complete(
            request_id=_rid,
            query=query,
            latency_ms=_qie_elapsed,
            filter_count=len(_qie_identified),
            hard_filter_count=_hard_n,
            source=_qie_source,
            model_id="extract_filters_only",
            token_count=0,
            decision_cost_usd=_qie_cost_usd,
            grounded_drop_count=_grounded_drop_count,
            prompt_tag=_qie_prompt_tag,
            schema_version=_qie_schema_version,
            body=_qie_body,
        )
        return _round_floats(_qie_body, _RESPONSE_FLOAT_PRECISION)
    # L1 preview (fast cosine, ~5ms) for timeout selection and speculative analytics pre-launch.
    _spec_analytics_task: Optional[asyncio.Task] = None
    _l1_preview: Optional[tuple] = None
    _qi_engine_ref = getattr(sub, "qi_engine", None)
    if _qi_engine_ref is not None:
        try:
            _l1_preview = await asyncio.get_running_loop().run_in_executor(
                None, _qi_engine_ref.quick_classify, query
            )
        except Exception as _l1e:  # noqa: BLE001
            logger.warning(f"l1_preview_failed request_id={_rid} error={_l1e}")
        if (
            _l1_preview is not None
            and search_cfg.speculative_analytics_start
            and sub.orchestrator.analytics_available
            and _l1_preview[0] == "analytics"
            and _l1_preview[1]
            >= search_cfg.speculative_analytics_l1_confidence_threshold
        ):
            _spec_analytics_task = asyncio.create_task(
                sub.orchestrator.analytics(question=query, sql_hint="", request_id=_rid)
            )
            logger.info(
                f"speculative_analytics_started request_id={_rid} l1_type={_l1_preview[0]} l1_confidence={_l1_preview[1]:.3f}"
            )
    # Per-type outer timeout: analytics get extended budget. Two triggers:
    # 1. L1 classifies analytics with high confidence. 2. Query has aggregate keywords.
    _l1_type = _l1_preview[0] if _l1_preview is not None else None
    _l1_conf = _l1_preview[1] if _l1_preview is not None else 0.0
    _query_lower = query.lower()
    _has_aggregate_signal = any(
        kw in _query_lower for kw in search_cfg.analytics_budget_keywords
    )
    _l1_analytics_confident = (
        _l1_type == "analytics"
        and _l1_conf >= search_cfg.speculative_analytics_l1_confidence_threshold
    )
    _use_analytics_budget = _l1_analytics_confident or _has_aggregate_signal
    _outer_timeout = (
        float(search_cfg.analytics_total_budget_seconds)
        if _use_analytics_budget
        else float(search_cfg.search_timeout_seconds)
    )
    logger.debug(
        f"outer_timeout_selected request_id={_rid} l1_type={_l1_type} aggregate_signal={_has_aggregate_signal} timeout_s={_outer_timeout}"
    )
    # Do NOT speculative-prefire explore/timeout rails. Early CH rail fan-out
    # contends with SqlRetriever + explore-prewarm on the happy path (~1.5s),
    # and hybrid retrieve already has ranked_results_complement / ZRG paths.
    # Create the task lazily in the timeout handler only when the outer SLA
    # actually breaches (same path used when L1=explore confident).
    _l1_explore_confident = (
        _l1_type == "explore"
        and _l1_conf >= search_cfg.speculative_analytics_l1_confidence_threshold
    )
    _explore_task: Optional[asyncio.Task] = None
    logger.debug(
        f"explore_prefetch_deferred request_id={_rid} l1_type={_l1_type} "
        f"l1_conf={_l1_conf:.3f} l1_explore_confident={_l1_explore_confident} "
        f"reason=avoid_ch_contention_on_happy_path"
    )
    logger.debug(
        f"hybrid_retrieval_start request_id={_rid} top_k={top_k} "
        f"diversity_lambda={effective_diversity_lambda} timeout_s={_outer_timeout}"
    )
    try:
        results, outcome, guard_outcome = await asyncio.wait_for(
            sub.orchestrator.search(
                raw_query=query,
                request_id=_rid,
                user_context=user_ctx,
                top_k=top_k,
                diversity_lambda=effective_diversity_lambda,
            ),
            timeout=_outer_timeout,
        )
        # Don't cancel _explore_task; analytics need it as fallback. Non-analytics cancel after qt known.
    except asyncio.TimeoutError:
        _elapsed_pre = round((time.monotonic() - t_start) * 1000, 1)
        if _spec_analytics_task is not None and not _spec_analytics_task.done():
            _spec_analytics_task.cancel()
        logger.warning(
            f"search_timeout_sla_breach request_id={_rid} "
            f"elapsed_ms={_elapsed_pre} sla_seconds={_outer_timeout} l1_type={_l1_type}"
        )
        # Parent L1 tier owns answer_mode; explore rails fill body only.
        _timeout_answer_mode = _answer_mode_for_query_type(_l1_type)
        # Guard against event-loop starvation or CH TCP hangs after internal timeout cap.
        # When explore pre-fire was skipped (L1=explore confident), create the task now.
        if _explore_task is None:
            _explore_task = asyncio.create_task(
                sub.orchestrator.get_timeout_fallback(
                    request_id=_rid, query=query, top_k=top_k
                )
            )
        try:
            _fallback = await asyncio.wait_for(
                _explore_task,
                timeout=float(search_cfg.explore_fallback_timeout_seconds),
            )
        except (asyncio.TimeoutError, asyncio.CancelledError):
            if not _explore_task.done():
                _explore_task.cancel()
            logger.warning(
                f"explore_task_timeout_post_sla_breach request_id={_rid} elapsed_ms={_elapsed_pre}"
            )
            _elapsed = round((time.monotonic() - t_start) * 1000, 1)
            _timeout_qi = {
                "decision_cost_usd": float(sub.orchestrator.request_llm_cost_usd(_rid))
            }
            sub.orchestrator.clear_request_llm_cost(_rid)
            return {
                "query": query,
                "answer_mode": _timeout_answer_mode,
                "latency_ms": _elapsed,
                "query_intelligence": _timeout_qi,
                "pipeline_trace": {},
                "ranked_results": [],
                "retrieval_metrics": {
                    "failure_mode": "timeout",
                    "total_candidates": 0,
                    "metrics_valid": False,
                    "params": {
                        "top_k": top_k,
                        "diversity_lambda": effective_diversity_lambda,
                        "relevance_threshold": effective_relevance_threshold,
                    },
                },
                "analytics": {},
                "guidance": {},
                "guard_notice": (
                    f"Search exceeded the {_outer_timeout}s SLA — explore fallback also timed out."
                ),
            }
        _elapsed = round((time.monotonic() - t_start) * 1000, 1)
        _fallback_norm = _normalise_scores(_fallback.items, _norm_mode)
        _fallback_results = [
            _slim_result(item, rank, score, search_cfg)
            for rank, (item, score) in enumerate(
                zip(_fallback.items, _fallback_norm), start=1
            )
        ]
        _fallback_results = _auction_tiebreak(
            _fallback_results, search_cfg.auction_tiebreak
        )
        _timeout_qi = {
            "decision_cost_usd": float(sub.orchestrator.request_llm_cost_usd(_rid))
        }
        sub.orchestrator.clear_request_llm_cost(_rid)
        return {
            "query": query,
            "answer_mode": _timeout_answer_mode,
            "latency_ms": _elapsed,
            "query_intelligence": _timeout_qi,
            "pipeline_trace": {},
            "ranked_results": _fallback_results,
            "retrieval_metrics": {
                "failure_mode": "timeout",
                "total_candidates": _fallback.total_candidates,
                "metrics_valid": False,
                "params": {
                    "top_k": top_k,
                    "diversity_lambda": effective_diversity_lambda,
                    "relevance_threshold": effective_relevance_threshold,
                },
            },
            "analytics": {},
            "guidance": {},
            "guard_notice": (
                f"Search exceeded the {_outer_timeout}s SLA — "
                + (
                    "showing trending, ending-soon, and latest domains instead."
                    if _fallback_results
                    else "explore fallback also returned no results (index may be empty — run /data-build/seed)."
                )
            ),
        }
    except ValidationError as e:
        if _explore_task is not None:
            _explore_task.cancel()
        if _spec_analytics_task is not None and not _spec_analytics_task.done():
            _spec_analytics_task.cancel()
        raise HTTPException(status_code=422, detail=str(e)) from e
    except (
        QdrantUnavailableError,
        QdrantQueryError,
        ClickHouseUnavailableError,
        ClickHouseQueryError,
        RetrievalError,
    ) as e:
        # 503 only when the failed backend is the query's primary store.
        # Analytics + Qdrant error, or listing + ClickHouse error = wrong primary — ignore.
        _analytics_primary_exc = _is_analytics_primary_query(
            query_type=_l1_type,
            l1_type=_l1_type,
            use_analytics_budget=bool(_use_analytics_budget),
        )
        if _backend_matches_query_primary(e, analytics_primary=_analytics_primary_exc):
            if _explore_task is not None:
                _explore_task.cancel()
            if _spec_analytics_task is not None and not _spec_analytics_task.done():
                _spec_analytics_task.cancel()
            _elapsed_be = round((time.monotonic() - t_start) * 1000, 1)
            _cost_be = float(sub.orchestrator.request_llm_cost_usd(_rid))
            sub.orchestrator.clear_request_llm_cost(_rid)
            return _backend_unavailable_response(
                query=query,
                request_id=_rid,
                search_id=_sid,
                exc=e,
                latency_ms=_elapsed_be,
                answer_mode=_answer_mode_for_query_type(
                    "analytics" if _analytics_primary_exc else _l1_type
                ),
                top_k=top_k,
                diversity_lambda=effective_diversity_lambda,
                relevance_threshold=effective_relevance_threshold,
                decision_cost_usd=_cost_be,
            )
        logger.warning(
            f"backend_exc_ignored_wrong_primary request_id={_rid} "
            f"analytics_primary={_analytics_primary_exc} error_type={type(e).__name__} error={e}"
        )
        # Continue with empty hybrid so analytics (CH) or listing response can finish.
        results = RankedResults(
            request_id=_rid,
            items=[],
            total_candidates=0,
            fusion_latency_ms=0.0,
            cache_hit=None,
            failure_mode=None,
            search_id=_sid,
        )
        outcome = ERankerOutcome(
            applied=False, client="noop", skipped_reason="wrong_primary_backend_ignored"
        )
        guard_outcome = ZeroResultGuardOutcome(
            fired=False,
            ladder_step="none",
            original_filter_count=0,
            relaxed_filter_count=0,
            relaxation_reason="",
        )

    # Soft-fail retrieve (_safe_retrieve) swallows Qdrant errors into empty sets.
    # Orchestrator stamps failure_mode=qdrant_unavailable when that happens — surface
    # a clear 503 instead of inventory_empty + filter_relaxed_by_guard.
    # Exception: analytics queries need ClickHouse, not Qdrant. Hybrid-first still
    # hits Qdrant for ranked_results, but a Qdrant-down 503 here would lie about
    # the primary backend. Defer to the analytics route (CH 503 / analytics body).
    _qi_type_early = (
        results.query_intent.query_type if results.query_intent is not None else None
    )
    _is_analytics_query = _is_analytics_primary_query(
        query_type=_qi_type_early,
        l1_type=_l1_type,
        use_analytics_budget=bool(_use_analytics_budget),
    )
    _be_exc: Optional[BaseException] = None
    if _is_analytics_query:
        # Analytics-primary: only ClickHouse unavailability is user-facing here.
        if results.failure_mode == "clickhouse_unavailable":
            _be_exc = ClickHouseUnavailableError(
                "ClickHouse is unavailable (connection refused or credentials absent)"
            )
    else:
        # Listing/hybrid-primary: only Qdrant unavailability (never ClickHouse).
        if results.failure_mode == "qdrant_unavailable":
            _be_exc = (
                sub.orchestrator.last_vector_retrieve_error
                or QdrantUnavailableError(
                    "Qdrant is unavailable (connection refused or client not reachable)"
                )
            )
        elif not results.items:
            _vec_err = sub.orchestrator.last_vector_retrieve_error
            if isinstance(_vec_err, (QdrantUnavailableError, QdrantQueryError)):
                _be_exc = _vec_err
    if _be_exc is not None:
        if _explore_task is not None:
            _explore_task.cancel()
        if _spec_analytics_task is not None and not _spec_analytics_task.done():
            _spec_analytics_task.cancel()
        _elapsed_be2 = round((time.monotonic() - t_start) * 1000, 1)
        _cost_be2 = float(sub.orchestrator.request_llm_cost_usd(_rid))
        sub.orchestrator.clear_request_llm_cost(_rid)
        return _backend_unavailable_response(
            query=query,
            request_id=_rid,
            search_id=_sid,
            exc=_be_exc,
            latency_ms=_elapsed_be2,
            answer_mode=_answer_mode_for_query_type(
                results.query_intent.query_type
                if results.query_intent is not None
                else _l1_type
            ),
            top_k=top_k,
            diversity_lambda=effective_diversity_lambda,
            relevance_threshold=effective_relevance_threshold,
            decision_cost_usd=_cost_be2,
        )

    # Primary-backend hygiene: do not leak the other store's failure_mode.
    if _is_analytics_query and results.failure_mode == "qdrant_unavailable":
        results = replace(results, failure_mode=None)
    elif (not _is_analytics_query) and results.failure_mode == "clickhouse_unavailable":
        results = replace(results, failure_mode=None)

    # Resolve intent: re-classify on cache hit (stale intent) or when absent.
    intent_for_response: Optional[QueryIntent] = results.query_intent
    _internally_rerouted = (
        results.query_intent is not None
        and _l1_type is not None
        and results.query_intent.query_type != _l1_type
    )
    # For explore/guidance reroutes, results.query_intent already holds the accurate
    # full-QI classification. L1 imprecision on these types is expected (L1 often
    # returns 'hybrid' for explore/guidance queries). Re-classifying adds a serial
    # LLM call with no correctness benefit — the full-QI result inside search() is right.
    _final_type = (
        results.query_intent.query_type if results.query_intent is not None else None
    )
    _reroute_needs_reclassify = _internally_rerouted and _final_type not in (
        "explore",
        "guidance",
    )
    # A reroute reclassify re-runs the same _classify_single ensemble under the
    # same qi.llm.classify_timeout_seconds budget as the classification already
    # performed inside orchestrator.search() for this exact query — it can time
    # out and degrade to _classify_timeout_regex_fallback (decision_tier
    # 'L0_fallback'), which is regex-only and misses LLM-only slots. When that
    # happens, keep the original routing-consistent classification instead of
    # overwriting it with a strictly worse one for the same query.
    _degraded_tiers = frozenset({"L0_fallback", "fallback", "ensemble_all_abstain"})
    _can_prefer_original = (
        _reroute_needs_reclassify
        and results.cache_hit != "semantic"
        and results.query_intent is not None
    )
    if (
        results.cache_hit == "semantic"
        or (intent_for_response is None and results.cache_hit is None)
        or _reroute_needs_reclassify
    ):
        try:
            _reclassified = await sub.qi_engine.classify(
                raw_query=query, request_id=results.request_id
            )
            if (
                _can_prefer_original
                and _reclassified.decision_tier in _degraded_tiers
                and results.query_intent.decision_tier not in _degraded_tiers
            ):
                logger.warning(
                    f"qi_reclassify_degraded_kept_original request_id={results.request_id} "
                    f"reclassify_tier={_reclassified.decision_tier} "
                    f"original_tier={results.query_intent.decision_tier}"
                )
            else:
                intent_for_response = _reclassified
        except AgentSearchError as e:
            logger.warning(
                f"intent_recovery_classify_failed request_id={results.request_id} "
                f"error_type={type(e).__name__} error={str(e)}"
            )

    # ── Build query_intelligence block ────────────────────────────────────────
    qt = intent_for_response.query_type if intent_for_response else "unknown"
    # Analytics path keeps _explore_task alive as a timeout fallback.
    # Every other path doesn't need it — cancel now so it doesn't outlive this request.
    if qt != "analytics":
        if _explore_task is not None:
            _explore_task.cancel()
            try:
                await _explore_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        if _spec_analytics_task is not None and not _spec_analytics_task.done():
            _spec_analytics_task.cancel()
    _slots_cfg = sub.config.qi.entity_slots
    if _slots_cfg is None:
        raise HTTPException(
            status_code=503,
            detail="qi.entity_slots not configured — cannot shape identified_filters",
        )
    _hfa = getattr(sub.config.retrieval, "hard_filter_application", None)
    _backend_unsupported = (
        set(_hfa.backend_unsupported_set) if _hfa is not None else None
    )
    filter_summary = _build_filter_summary(
        intent_for_response,
        soft_slot_names=set(_slots_cfg.soft_slot_set),
        soft_response_key=_slots_cfg.soft_response_key,
        guard_outcome=guard_outcome,
        missing_columns=app_state._missing_data_columns or None,
        tld_substitutions=sub.config.qi.regex.tld_substitutions or None,
        backend_unsupported_slots=_backend_unsupported,
    )
    # Keep identified/soft filters on guidance — answer_mode may be guidance, but
    # extracted chips still belong in query_intelligence (parity with qie_only / L0).
    # Per-entity extraction confidence is on each filters.identified[].confidence.
    query_intelligence: Dict[str, Any] = _build_query_intelligence(
        intent_for_response,
        filter_summary,
        multi_intent_envelope=results.multi_intent_envelope,
        cache_hit=results.cache_hit,
        soft_slot_names=set(_slots_cfg.soft_slot_set),
    )

    # ── Analytics route ───────────────────────────────────────────────────────
    analytics_outcome: str
    analytics_data: Optional[Dict[str, Any]] = None
    _scoped_explore_task: Optional[asyncio.Task] = None
    if qt == "analytics":
        logger.debug(f"analytics_branch_start request_id={_rid} l1_type={_l1_type}")
        # _explore_task is None when L1 previewed explore-confident (pre-fire skipped).
        # Both analytics branches below consume it (unavailable: as the fallback source;
        # available: to supersede before the scoped task), so materialize it here.
        if _explore_task is None:
            _explore_task = asyncio.create_task(
                sub.orchestrator.get_timeout_fallback(
                    request_id=_rid, query=query, top_k=top_k
                )
            )
        if not sub.orchestrator.analytics_available:
            analytics_outcome = "unavailable: analytics_router not configured"
            logger.warning(
                f"analytics_unavailable request_id={_rid} reason=router_not_wired_or_executor_down"
            )
            if _spec_analytics_task is not None and not _spec_analytics_task.done():
                _spec_analytics_task.cancel()
            if _explore_task is not None and not _explore_task.done():
                _explore_task.cancel()
            _elapsed_ch = round((time.monotonic() - t_start) * 1000, 1)
            _cost_ch = float(sub.orchestrator.request_llm_cost_usd(_rid))
            sub.orchestrator.clear_request_llm_cost(_rid)
            return _backend_unavailable_response(
                query=query,
                request_id=_rid,
                search_id=_sid,
                exc=ClickHouseUnavailableError(
                    "ClickHouse analytics is unavailable (router not wired, credentials missing, "
                    "or ClickHouse down). Verify ClickHouse host/port/credentials and that "
                    "nl_to_sql.analytics is enabled."
                ),
                latency_ms=_elapsed_ch,
                answer_mode=_answer_mode_for_query_type("analytics"),
                top_k=top_k,
                diversity_lambda=effective_diversity_lambda,
                relevance_threshold=effective_relevance_threshold,
                decision_cost_usd=_cost_ch,
            )
        else:
            # Fire an entity-scoped explore task now that intent_for_response is known.
            # intent_for_response carries the classified slices (tld, auction_type, etc.)
            # so compose_fallback can apply hard filters post-fusion.
            # bypass_cache=True skips the 300 s in-memory cache that would otherwise
            # return the earlier unscoped result from the pre-fired _explore_task.
            _scoped_explore_task = asyncio.create_task(
                sub.orchestrator.get_timeout_fallback(
                    request_id=_rid,
                    query=query,
                    top_k=top_k,
                    intent=intent_for_response,
                    bypass_cache=True,
                )
            )
            # The earlier unscoped task is superseded — cancel it now.
            if not _explore_task.done():
                _explore_task.cancel()
                try:
                    await _explore_task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            # Trust the analytics classification only when an intent-understanding
            # signal corroborates it. Routing defers to the classifier's intent
            # legs rather than any reserved-keyword list:
            #   1. an intent-understanding leg decided analytics — the LLM tier
            #      (L2_llm) or the learned semantic router (L1_semantic); or
            #   2. the learned L1 preview agrees the query is analytics; or
            #   3. the classifier extracted a concrete analytics dimension/filter
            #      (a non-empty sql_hint).
            # When analytics was asserted only by a cheap entity/regex gate
            # (L0_*) with none of the above, it is treated as a likely misroute:
            # running the NL-to-SQL pipeline would burn ~5s of LLM SQL generation
            # for a query the intent legs do not believe is analytics, and it
            # falls back to explore anyway. Serve the parallel explore task now.
            _detour_sql_hint = _build_sql_hint(intent_for_response, query)
            _an_tier = (
                (intent_for_response.decision_tier or "") if intent_for_response else ""
            )
            _an_intent_leg_decided = _an_tier in ("L1_semantic", "L2_llm")
            _an_l1_agrees = _l1_type == "analytics"
            _an_corroborated = (
                _an_intent_leg_decided or _an_l1_agrees or bool(_detour_sql_hint)
            )
            if search_cfg.analytics_skip_uncorroborated_detour and not _an_corroborated:
                analytics_outcome = "skipped: uncorroborated_analytics_detour"
                logger.info(
                    f"analytics_detour_skipped request_id={results.request_id} "
                    f"reason=uncorroborated_intent decision_tier={_an_tier or 'none'} "
                    f"l1_type={_l1_type} intent_confidence={intent_for_response.confidence if intent_for_response else None}"
                )
                _sc_fb = None
                try:
                    _sc_fb = await asyncio.wait_for(
                        _scoped_explore_task,
                        timeout=float(search_cfg.explore_fallback_timeout_seconds),
                    )
                except (
                    asyncio.TimeoutError,
                    asyncio.CancelledError,
                    Exception,  # noqa: BLE001
                ) as _sc_err:
                    logger.warning(
                        f"analytics_detour_skip_explore_failed request_id={_rid} error_type={type(_sc_err).__name__} error={_sc_err}"
                    )
                    if not _scoped_explore_task.done():
                        _scoped_explore_task.cancel()
                if _spec_analytics_task is not None and not _spec_analytics_task.done():
                    _spec_analytics_task.cancel()
                # results.items already computed by search() (explore/hybrid primary) —
                # prefer the scoped explore task when it yielded items, else reuse those.
                if _sc_fb is not None and _sc_fb.items:
                    _sc_items = list(_sc_fb.items)
                    _sc_total = _sc_fb.total_candidates
                else:
                    _sc_items = list(results.items)
                    _sc_total = results.total_candidates
                _sc_norm = _normalise_scores(_sc_items, _norm_mode)
                _sc_results = [
                    _slim_result(item, rank, score, search_cfg)
                    for rank, (item, score) in enumerate(
                        zip(_sc_items, _sc_norm), start=1
                    )
                ]
                _sc_results = _auction_tiebreak(
                    _sc_results, search_cfg.auction_tiebreak
                )
                _elapsed_sc = round((time.monotonic() - t_start) * 1000, 1)
                sub.orchestrator.clear_request_llm_cost(_rid)
                return {
                    "query": query,
                    "answer_mode": _answer_mode_for_query_type("analytics"),
                    "latency_ms": _elapsed_sc,
                    "query_intelligence": _round_floats(
                        query_intelligence, _RESPONSE_FLOAT_PRECISION
                    ),
                    "pipeline_trace": _round_floats(
                        _build_pipeline_trace(
                            filter_summary,
                            analytics_outcome,
                            outcome,
                            guard_outcome,
                            len(_sc_results) or None,
                            query_type="analytics",
                            sub=sub,
                        ),
                        _RESPONSE_FLOAT_PRECISION,
                    ),
                    "ranked_results": _sc_results,
                    "retrieval_metrics": {
                        "failure_mode": "analytics_failure_fallback",
                        "total_candidates": _sc_total,
                        "metrics_valid": False,
                        "params": {
                            "top_k": top_k,
                            "diversity_lambda": effective_diversity_lambda,
                            "relevance_threshold": effective_relevance_threshold,
                        },
                        "backends_active": [],
                    },
                    "analytics": {},
                    "guidance": {},
                    "guard_notice": search_cfg.analytics_detour_skip_notice,
                }
            # Bound TOTAL analytics wall-clock from request start: the inline
            # analytics call gets whatever remains of analytics_total_budget_seconds
            # after classification + routing, capped by the per-call
            # analytics_timeout_seconds. This guarantees the end-to-end analytics
            # response never exceeds analytics_total_budget_seconds.
            _analytics_budget_remaining = float(
                search_cfg.analytics_total_budget_seconds
            ) - (time.monotonic() - t_start)
            _effective_analytics_timeout = min(
                float(search_cfg.analytics_timeout_seconds), _analytics_budget_remaining
            )
            try:
                if _effective_analytics_timeout <= 0.0:
                    logger.warning(
                        f"analytics_total_budget_exhausted_pre_call request_id={results.request_id} budget_s={float(search_cfg.analytics_total_budget_seconds):.1f}"
                    )
                    raise asyncio.TimeoutError
                _sql_hint = _build_sql_hint(intent_for_response, query)
                _use_spec = _spec_analytics_task is not None
                if _use_spec and _sql_hint:
                    if not _spec_analytics_task.done():
                        _spec_analytics_task.cancel()
                        try:
                            await _spec_analytics_task
                        except (asyncio.CancelledError, Exception):  # noqa: BLE001
                            pass
                        _use_spec = False
                        logger.info(
                            f"speculative_analytics_cancelled_hint_available request_id={_rid}"
                        )
                    else:
                        try:
                            _spec_check = _spec_analytics_task.result()
                            if not _spec_check.success:
                                _use_spec = False
                                logger.info(
                                    f"speculative_analytics_discarded_failed_hint_available request_id={_rid} failure_mode={_spec_check.failure_mode}"
                                )
                        except Exception:  # noqa: BLE001
                            _use_spec = False
                if _use_spec:
                    analytics_result = await asyncio.wait_for(
                        _spec_analytics_task, timeout=_effective_analytics_timeout
                    )
                else:
                    analytics_result = await asyncio.wait_for(
                        sub.orchestrator.analytics(
                            question=query,
                            sql_hint=_sql_hint,
                            request_id=results.request_id,
                        ),
                        timeout=_effective_analytics_timeout,
                    )
                analytics_outcome = (
                    f"applied: {analytics_result.analytics_substrate or 'completed'}"
                )
                _add_llm_cost_usd(
                    query_intelligence,
                    float(getattr(analytics_result, "llm_cost_usd", 0.0) or 0.0),
                )
                _raw = _format_analytics_timestamps(
                    _round_floats(asdict(analytics_result), _RESPONSE_FLOAT_PRECISION)
                )
                analytics_data = (
                    _raw if analytics_result.success else _slim_analytics_failure(_raw)
                )
                if (
                    not analytics_result.success
                    and analytics_result.failure_mode == "beyond_data_window"
                ):
                    analytics_data["data_window"] = _build_data_window_notice(
                        analytics_result.failure_reason
                    )
                    analytics_outcome = "failed: beyond_data_window"
                    logger.info(
                        f"analytics_beyond_data_window request_id={analytics_result.request_id} reason={analytics_result.failure_reason!r}"
                    )
                # Hybrid ranked_results already produced by orchestrator.search(); cancel
                # scoped explore unless analytics fails and hybrid is empty (fill below).
                if (
                    analytics_result.success
                    and _scoped_explore_task is not None
                    and not _scoped_explore_task.done()
                ):
                    _scoped_explore_task.cancel()
                    try:
                        await _scoped_explore_task
                    except (asyncio.CancelledError, Exception):  # noqa: BLE001
                        pass
                if not analytics_result.success and analytics_result.failure_mode:
                    try:
                        await sub.signal_store.record_async(
                            FeedbackSignal(
                                signal_id=FeedbackSignal.new_signal_id(),
                                request_id=analytics_result.request_id,
                                signal_type="analytics_failure",
                                payload={
                                    "failure_mode": analytics_result.failure_mode,
                                    "failure_reason": analytics_result.failure_reason,
                                    "total_latency_ms": analytics_result.total_latency_ms,
                                },
                                intent_record_id=analytics_result.intent_record_id,
                                signal_origin="analytics_router",
                            )
                        )
                    except (ValidationError, AgentSearchError) as _sig_err:
                        logger.error(
                            f"analytics_failure_signal_emit_failed request_id={analytics_result.request_id} error={_sig_err}"
                        )
            except asyncio.TimeoutError:
                analytics_outcome = "failed: analytics_timeout"
                _elapsed_analytics_to = round((time.monotonic() - t_start) * 1000, 1)
                logger.warning(
                    f"inline_analytics_timeout request_id={results.request_id} "
                    f"effective_timeout_s={_effective_analytics_timeout:.1f} "
                    f"total_elapsed_ms={_elapsed_analytics_to}"
                )
                query_intelligence["decision_cost_usd"] = float(
                    sub.orchestrator.request_llm_cost_usd(_rid)
                )
                # Hybrid pipeline already completed for this query — prefer those
                # results over the CH-backed explore rails (empty when ClickHouse is
                # down/slow). Matches the analytics-unavailable + analytics-failure paths.
                if results.items:
                    for _t in (_scoped_explore_task, _explore_task):
                        if _t is not None and not _t.done():
                            _t.cancel()
                            try:
                                await _t
                            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                                pass
                    analytics_outcome = (
                        "failed: analytics_timeout; served hybrid search results"
                    )
                    _hyb_to_items = list(results.items)
                    _hyb_to_norm = _normalise_scores(_hyb_to_items, _norm_mode)
                    _hyb_to_results = [
                        _slim_result(item, rank, score, search_cfg)
                        for rank, (item, score) in enumerate(
                            zip(_hyb_to_items, _hyb_to_norm), start=1
                        )
                    ]
                    _hyb_to_results = _auction_tiebreak(
                        _hyb_to_results, search_cfg.auction_tiebreak
                    )
                    _elapsed_hyb_to = round((time.monotonic() - t_start) * 1000, 1)
                    sub.orchestrator.clear_request_llm_cost(_rid)
                    return {
                        "query": query,
                        "answer_mode": _answer_mode_for_query_type("analytics"),
                        "latency_ms": _elapsed_hyb_to,
                        "query_intelligence": _round_floats(
                            query_intelligence, _RESPONSE_FLOAT_PRECISION
                        ),
                        "pipeline_trace": _round_floats(
                            _build_pipeline_trace(
                                filter_summary,
                                analytics_outcome,
                                outcome,
                                guard_outcome,
                                len(_hyb_to_results),
                                query_type="analytics",
                                sub=sub,
                            ),
                            _RESPONSE_FLOAT_PRECISION,
                        ),
                        "ranked_results": _hyb_to_results,
                        "retrieval_metrics": {
                            "failure_mode": "analytics_timeout",
                            "total_candidates": results.total_candidates,
                            "metrics_valid": False,
                            "params": {
                                "top_k": top_k,
                                "diversity_lambda": effective_diversity_lambda,
                                "relevance_threshold": effective_relevance_threshold,
                            },
                            "backends_active": [],
                        },
                        "analytics": {},
                        "guidance": {},
                        "guard_notice": search_cfg.analytics_timeout_hybrid_notice,
                    }
                # Scoped explore task has been running in parallel since analytics started.
                # Await with a short grace period — it should be complete by now.
                _timeout_explore = (
                    _scoped_explore_task
                    if _scoped_explore_task is not None
                    else _explore_task
                )
                try:
                    _analytics_fallback = await asyncio.wait_for(
                        _timeout_explore, timeout=0.5
                    )
                except (asyncio.TimeoutError, asyncio.CancelledError, Exception):  # noqa: BLE001
                    _analytics_fallback = None
                    _timeout_explore.cancel()
                # Pre-warmed fallback may be empty when explore_composer is disabled
                # or the parallel task had no data. Fire a fresh fallback that runs
                # trending + ending-soon + latest + semantic RRF in parallel.
                if _analytics_fallback is None or not _analytics_fallback.items:
                    try:
                        _analytics_fallback = await asyncio.wait_for(
                            sub.orchestrator.get_timeout_fallback(
                                request_id=_rid,
                                query=query,
                                top_k=top_k,
                                intent=intent_for_response,
                            ),
                            timeout=float(
                                getattr(
                                    search_cfg, "explore_fallback_timeout_seconds", 1.5
                                )
                            ),
                        )
                        logger.info(
                            f"analytics_timeout_refallback_ok request_id={_rid} "
                            f"items={len(_analytics_fallback.items) if _analytics_fallback else 0}"
                        )
                    except (asyncio.TimeoutError, Exception) as _rfb_err:  # noqa: BLE001
                        logger.warning(
                            f"analytics_timeout_refallback_failed request_id={_rid} error={_rfb_err}"
                        )
                        _analytics_fallback = None
                # Rails fill body; parent tier stays analytics (not a misroute).
                _elapsed_fb = round((time.monotonic() - t_start) * 1000, 1)
                _fb_items = (
                    _analytics_fallback.items
                    if (_analytics_fallback is not None and _analytics_fallback.items)
                    else []
                )
                _fb_total = (
                    _analytics_fallback.total_candidates
                    if _analytics_fallback is not None
                    else 0
                )
                _fb_norm = _normalise_scores(_fb_items, _norm_mode)
                _fb_results = [
                    _slim_result(item, rank, score, search_cfg)
                    for rank, (item, score) in enumerate(
                        zip(_fb_items, _fb_norm), start=1
                    )
                ]
                _fb_results = _auction_tiebreak(
                    _fb_results, search_cfg.auction_tiebreak
                )
                sub.orchestrator.clear_request_llm_cost(_rid)
                return {
                    "query": query,
                    "answer_mode": _answer_mode_for_query_type("analytics"),
                    "latency_ms": _elapsed_fb,
                    "query_intelligence": _round_floats(
                        query_intelligence, _RESPONSE_FLOAT_PRECISION
                    ),
                    "pipeline_trace": _round_floats(
                        _build_pipeline_trace(
                            filter_summary,
                            analytics_outcome,
                            outcome,
                            guard_outcome,
                            None,
                            query_type="analytics",
                            sub=sub,
                        ),
                        _RESPONSE_FLOAT_PRECISION,
                    ),
                    "ranked_results": _fb_results,
                    "retrieval_metrics": {
                        "failure_mode": "analytics_timeout",
                        "total_candidates": _fb_total,
                        "metrics_valid": False,
                        "params": {
                            "top_k": top_k,
                            "diversity_lambda": effective_diversity_lambda,
                            "relevance_threshold": effective_relevance_threshold,
                        },
                        "backends_active": [],
                    },
                    "analytics": {},
                    "guidance": {},
                    "guard_notice": search_cfg.analytics_timeout_explore_notice,
                }
            except (ValidationError, RetrievalError, AgentSearchError) as _ae:
                analytics_outcome = f"failed: {type(_ae).__name__}"
                logger.warning(
                    f"inline_analytics_failed request_id={results.request_id} "
                    f"error_type={type(_ae).__name__} error={str(_ae)} — falling back to search results"
                )
    else:
        analytics_outcome = f"not_applicable: classified as {qt}"

    # Final cleanup: ensure _explore_task, _scoped_explore_task, and _spec_analytics_task
    # are not left dangling. analytics-unavailable, analytics-error, and
    # analytics-timeout-with-no-items all reach here without having cancelled the tasks.
    if _explore_task is not None and not _explore_task.done():
        _explore_task.cancel()
        try:
            await _explore_task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
    if _scoped_explore_task is not None and not _scoped_explore_task.done():
        _scoped_explore_task.cancel()
        try:
            await _scoped_explore_task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
    if _spec_analytics_task is not None and not _spec_analytics_task.done():
        _spec_analytics_task.cancel()

    if analytics_data is not None:
        # Analytics answer lives in `analytics` block. ranked_results = hybrid from
        # orchestrator.search(); explore rails only fill when analytics failed and hybrid empty.
        _analytics_failed = not analytics_result.success
        _fb_items: List[Any] = list(results.items)
        _fb_failure_mode: Optional[str] = results.failure_mode
        if (
            _analytics_failed
            and search_cfg.analytics_failure_explore_fallback
            and not _fb_items
        ):
            try:
                _afb = await asyncio.wait_for(
                    sub.orchestrator.get_timeout_fallback(
                        request_id=_rid,
                        query=query,
                        top_k=top_k,
                        intent=intent_for_response,
                    ),
                    timeout=float(search_cfg.explore_fallback_timeout_seconds),
                )
                _fb_items = list(_afb.items) if _afb is not None else []
                _fb_failure_mode = (
                    _afb.failure_mode
                    if _afb is not None
                    else "analytics_failure_fallback"
                )
            except (asyncio.TimeoutError, Exception) as _afb_err:  # noqa: BLE001
                logger.warning(
                    f"analytics_failure_explore_fallback_failed request_id={_rid} error_type={type(_afb_err).__name__} error={_afb_err}"
                )
                _fb_items = []
        _fb_norm = _normalise_scores(_fb_items, _norm_mode)
        _fb_ranked = [
            _slim_result(item, rank, score, search_cfg)
            for rank, (item, score) in enumerate(zip(_fb_items, _fb_norm), start=1)
        ]
        _fb_ranked = apply_complement_rank_leanings(
            _fb_ranked,
            query_type="analytics",
            cfg=search_cfg.ranked_results_complement.rank_leanings,
            analytics_data=analytics_data,
        )
        _fb_ranked = _auction_tiebreak(_fb_ranked, search_cfg.auction_tiebreak)
        _fb_retrieval_metrics: Dict[str, Any] = {
            "failure_mode": _fb_failure_mode,
            "total_candidates": results.total_candidates
            if results.items
            else len(_fb_items),
            "metrics_valid": bool(results.items) and not _analytics_failed,
            "params": {
                "top_k": top_k,
                "diversity_lambda": effective_diversity_lambda,
                "relevance_threshold": effective_relevance_threshold,
            },
        }
        _analytics_guard_notice: Optional[str] = None
        if (
            _analytics_failed
            and search_cfg.analytics_failure_explore_fallback
            and not results.items
        ):
            _analytics_guard_notice = (
                search_cfg.analytics_failure_with_explore_notice
                if _fb_ranked
                else search_cfg.analytics_failure_empty_explore_notice
            )
        endpoint_latency_ms = round((time.monotonic() - t_start) * 1000, 1)
        sub.orchestrator.clear_request_llm_cost(_rid)
        return {
            "query": query,
            "answer_mode": _answer_mode_for_query_type("analytics"),
            "latency_ms": endpoint_latency_ms,
            "query_intelligence": _round_floats(
                query_intelligence, _RESPONSE_FLOAT_PRECISION
            ),
            "pipeline_trace": _round_floats(
                _build_pipeline_trace(
                    filter_summary,
                    analytics_outcome,
                    outcome,
                    guard_outcome,
                    len(_fb_ranked) if _fb_ranked else None,
                    query_type="analytics",
                    sub=sub,
                ),
                _RESPONSE_FLOAT_PRECISION,
            ),
            "ranked_results": _fb_ranked,
            "retrieval_metrics": _fb_retrieval_metrics,
            "analytics": analytics_data,
            "guidance": {},
            "guard_notice": _analytics_guard_notice,
        }

    # ── Compute ranked results + quality metrics ──────────────────────────────
    # Normalise once over the FULL candidate pool so Recall@K's denominator
    # (relevant items in the whole post-fusion list) is consistent with the
    # numerator (relevant items in the returned top-K). Because the pool is
    # already sorted by fused_score and the max sits at rank 1, the prefix
    # `pool_norm_scores[:top_k]` is identical to normalising the slice on its
    # own — so existing per-result `coherence_score` values are unchanged.
    # Domain-level dedup: a domain_name can appear in multiple simultaneous auction
    # records. Keep only the highest-score entry per domain (results.items is
    # score-sorted descending, so the first occurrence is the best one).
    _seen_dns: set = set()
    _items_deduped: list = []
    for _c in results.items:
        _dn = str(_c.payload.get("domain_name", "")).lower()
        if _dn and _dn in _seen_dns:
            continue
        if _dn:
            _seen_dns.add(_dn)
        _items_deduped.append(_c)
    pool_norm_scores = _normalise_scores(_items_deduped, _norm_mode)
    ranked_items = _items_deduped[:top_k]
    norm_scores = pool_norm_scores[:top_k]
    # explore_fallback items carry SQL-derived fused_scores (bid_count / urgency),
    # not semantic similarity scores, so NDCG/Coherence/Recall are not meaningful.
    # Pass empty scores so the metrics block is honestly zero rather than
    # misleadingly computed from unrelated SQL ranking signals.
    is_explore_fallback = (
        guard_outcome.fired and guard_outcome.ladder_step == "explore_fallback"
    )
    metric_scores = [] if is_explore_fallback else norm_scores
    metric_pool = [] if is_explore_fallback else pool_norm_scores
    ranked_results = [
        _slim_result(item, rank, score, search_cfg)
        for rank, (item, score) in enumerate(zip(ranked_items, norm_scores), start=1)
    ]
    metrics = _compute_retrieval_metrics(
        metric_scores, metric_pool, top_k, effective_relevance_threshold
    )
    # Complement leanings from guidance snapshot (before auction tie-break).
    if qt == "guidance" and results.guidance_envelope is not None:
        ranked_results = apply_complement_rank_leanings(
            ranked_results,
            query_type="guidance",
            cfg=search_cfg.ranked_results_complement.rank_leanings,
            guidance_body=results.guidance_envelope.body,
        )
    # Relevance-dominant auction-signal tie-break (bounded; metrics computed above
    # from the pre-tiebreak order, so coherence_score/NDCG stay honest).
    ranked_results = _auction_tiebreak(ranked_results, search_cfg.auction_tiebreak)
    for _new_rank, _r in enumerate(ranked_results, start=1):
        _r["rank"] = _new_rank
    # Suppress score metrics when the corpus is too small to be meaningful.
    # With fewer candidates than half of top_k, NDCG/Recall/Precision are
    # trivially perfect — they measure the system's ability to rank N items
    # that are all already returned, not whether it found the right ones.
    _metrics_unreliable = 0 < results.total_candidates < max(5, top_k // 2)
    if _metrics_unreliable:
        metrics = {k: None for k in metrics}
    endpoint_latency_ms = round((time.monotonic() - t_start) * 1000, 1)
    returned_count = len(ranked_items)
    result_budget_shortfall = (
        round((top_k - returned_count) / float(top_k), 4)
        if top_k > 0 and returned_count < top_k
        else 0.0
    )
    if result_budget_shortfall > 0.0:
        logger.warning(
            f"retrieval_candidate_shortfall request_id={results.request_id} "
            f"query_type={qt} top_k={top_k} total_candidates={results.total_candidates} "
            f"returned={returned_count} shortfall_pct={result_budget_shortfall:.3f}"
        )
    retrieval_metrics: Dict[str, Any] = {
        **metrics,
        "metrics_valid": not _metrics_unreliable,
        "total_candidates": results.total_candidates,
        "result_budget_shortfall": result_budget_shortfall,
        "failure_mode": results.failure_mode,
        "params": {
            "top_k": top_k,
            "diversity_lambda": effective_diversity_lambda,
            "relevance_threshold": effective_relevance_threshold,
        },
    }
    # ── Backend coverage ──────────────────────────────────────────────────────
    # Collect which retrieval backends actually contributed candidates so the
    # caller can detect when hybrid routing fell through to vector-only.
    backends_seen: set = set()
    for _item in results.items:
        for _src in _item.contributing_sources or []:
            backends_seen.add(_src)
    retrieval_metrics["backends_active"] = sorted(backends_seen)

    # ── Corpus-size warning ───────────────────────────────────────────────────
    # Quality metrics are trivially perfect on tiny corpora (e.g. 5 candidates
    # against top_k=50). Surface a note so dashboards don't misread a perfect
    # NDCG as evidence of genuine retrieval quality.
    if _metrics_unreliable:
        retrieval_metrics["metrics_note"] = (
            f"quality_metrics_nulled: only {results.total_candidates} candidates vs top_k={top_k}; "
            f"scores suppressed — trivially perfect on a small corpus"
        )

    # ── Guard notice ─────────────────────────────────────────────────────────
    guard_notice: Optional[str] = None
    if guard_outcome.fired:
        _dropped = list(getattr(guard_outcome, "dropped_filter_names", []))
        if qt == "analytics" and guard_outcome.ladder_step == "explore_fallback":
            guard_notice = search_cfg.analytics_connection_unavailable_notice
        elif _dropped:
            guard_notice = (
                f"Some filters were relaxed to return results — "
                f"dropped: {', '.join(_dropped)}. "
                f"Results may not satisfy all original constraints."
            )
        else:
            guard_notice = (
                f"No results found with original filters; showing broader results "
                f"(guard step: {guard_outcome.ladder_step})."
            )

    # ── Guidance route — dedicated response shape ─────────────────────────────
    if qt == "guidance":
        logger.debug(f"guidance_branch_start request_id={_rid} l1_type={_l1_type}")
        # when guidance+aggregate signal: run analytics in parallel; prefer analytics answer on success
        _ct_analytics_result = None
        if (
            search_cfg.guidance_analytics_crosstype_enabled
            and _has_aggregate_signal
            and sub.orchestrator.analytics_available
        ):
            _ct_sql_hint = _build_sql_hint(intent_for_response, query)
            try:
                _ct_analytics_result = await asyncio.wait_for(
                    sub.orchestrator.analytics(
                        question=query, sql_hint=_ct_sql_hint, request_id=_rid
                    ),
                    timeout=float(
                        search_cfg.guidance_analytics_crosstype_timeout_seconds
                    ),
                )
                logger.info(
                    f"guidance_crosstype_analytics_attempted request_id={_rid} success={_ct_analytics_result.success}"
                )
                _add_llm_cost_usd(
                    query_intelligence,
                    float(getattr(_ct_analytics_result, "llm_cost_usd", 0.0) or 0.0),
                )
            except asyncio.TimeoutError:
                logger.info(
                    f"guidance_crosstype_analytics_timeout request_id={_rid} timeout_s={search_cfg.guidance_analytics_crosstype_timeout_seconds}"
                )
                query_intelligence["decision_cost_usd"] = float(
                    sub.orchestrator.request_llm_cost_usd(_rid)
                )
            except Exception as _cte:  # noqa: BLE001
                logger.warning(
                    f"guidance_crosstype_analytics_error request_id={_rid} error_type={type(_cte).__name__} error={_cte}"
                )
                query_intelligence["decision_cost_usd"] = float(
                    sub.orchestrator.request_llm_cost_usd(_rid)
                )
        if _ct_analytics_result is not None and _ct_analytics_result.success:
            _ct_raw = _format_analytics_timestamps(
                _round_floats(asdict(_ct_analytics_result), _RESPONSE_FLOAT_PRECISION)
            )
            _ct_qi = dict(query_intelligence)
            _ct_qi["crosstype_analytics_applied"] = True
            _ct_substrate = _ct_analytics_result.analytics_substrate or "completed"
            _ct_pt = _round_floats(
                _build_pipeline_trace(
                    filter_summary,
                    f"crosstype:{_ct_substrate}",
                    outcome,
                    guard_outcome,
                    None,
                    query_type=qt,
                    sub=sub,
                ),
                _RESPONSE_FLOAT_PRECISION,
            )
            _ct_ranked = apply_complement_rank_leanings(
                list(ranked_results),
                query_type="analytics",
                cfg=search_cfg.ranked_results_complement.rank_leanings,
                analytics_data=_ct_raw,
            )
            _ct_ranked = _auction_tiebreak(_ct_ranked, search_cfg.auction_tiebreak)
            sub.orchestrator.clear_request_llm_cost(_rid)
            return {
                "query": query,
                "answer_mode": _answer_mode_for_query_type("analytics"),
                "latency_ms": round((time.monotonic() - t_start) * 1000, 1),
                "query_intelligence": _round_floats(_ct_qi, _RESPONSE_FLOAT_PRECISION),
                "pipeline_trace": _ct_pt,
                "ranked_results": _round_floats(_ct_ranked, _RESPONSE_FLOAT_PRECISION),
                "retrieval_metrics": _round_floats(
                    retrieval_metrics, _RESPONSE_FLOAT_PRECISION
                ),
                "analytics": _ct_raw,
                "guidance": {},
                "guard_notice": None,
            }
        if results.guidance_envelope is not None:
            _env_dict = asdict(results.guidance_envelope)
            # Decode body from raw JSON string to a parsed object so consumers
            # don't have to double-decode.
            _raw_body = _env_dict.get("body")
            if isinstance(_raw_body, str):
                try:
                    _env_dict["body"] = json.loads(_raw_body)
                except (json.JSONDecodeError, ValueError):
                    pass  # leave as string if it is not valid JSON
            # Convert raw Unix epoch to a human-readable ISO 8601 UTC string so
            # callers don't need to interpret floats.
            _hot_epoch = _env_dict.pop("as_of_hot", None)
            if _hot_epoch is not None:
                try:
                    _env_dict["as_of_utc"] = datetime.fromtimestamp(
                        float(_hot_epoch), tz=timezone.utc
                    ).strftime("%Y-%m-%dT%H:%M:%SZ")
                except (TypeError, ValueError, OSError):
                    _env_dict["as_of_utc"] = None
        else:
            _env_dict = {}
            if not guard_notice:
                guard_notice = sub.config.guidance.snapshot_unavailable_notice
                logger.warning(
                    f"guidance_snapshot_unavailable request_id={_rid} "
                    f"reason=envelope_missing ranked_results={len(ranked_results)}"
                )
        # When snapshot missing and ranked_results empty, force semantic best-match
        # with CH->hybrid degrade (strip temporal filters).
        _g_comp = search_cfg.ranked_results_complement
        if (not ranked_results) and bool(_g_comp.force_semantic_when_empty):
            _g_sem_top_k = int(_g_comp.force_semantic_top_k)
            _g_sem_intent = (
                hybrid_degrade_intent(intent_for_response)
                if intent_for_response is not None
                else intent_for_response
            )
            try:
                _g_sem = await asyncio.wait_for(
                    sub.orchestrator._quick_semantic_retrieve(
                        _g_sem_intent, _g_sem_top_k
                    ),
                    timeout=float(search_cfg.explore_fallback_timeout_seconds),
                )
            except Exception as _g_sem_err:  # noqa: BLE001
                logger.warning(
                    f"guidance_semantic_force_failed request_id={_rid} "
                    f"error_type={type(_g_sem_err).__name__} error={_g_sem_err}"
                )
                _g_sem = []
            if _g_sem:
                _g_norm = _normalise_scores(_g_sem, _norm_mode)
                ranked_results = [
                    _slim_result(item, rank, score, search_cfg)
                    for rank, (item, score) in enumerate(
                        zip(_g_sem[:top_k], _g_norm[:top_k]), start=1
                    )
                ]
                if _env_dict:
                    ranked_results = apply_complement_rank_leanings(
                        ranked_results,
                        query_type="guidance",
                        cfg=_g_comp.rank_leanings,
                        guidance_body=_env_dict.get("body"),
                    )
                ranked_results = _auction_tiebreak(
                    ranked_results, search_cfg.auction_tiebreak
                )
                logger.info(
                    f"guidance_semantic_force request_id={_rid} items={len(ranked_results)}"
                )
        sub.orchestrator.clear_request_llm_cost(_rid)
        return {
            "query": query,
            "answer_mode": _answer_mode_for_query_type("guidance"),
            "latency_ms": endpoint_latency_ms,
            "query_intelligence": _round_floats(
                query_intelligence, _RESPONSE_FLOAT_PRECISION
            ),
            "pipeline_trace": _round_floats(
                _build_pipeline_trace(
                    filter_summary,
                    analytics_outcome,
                    outcome,
                    guard_outcome,
                    len(ranked_results) or None,
                    query_type=qt,
                    sub=sub,
                ),
                _RESPONSE_FLOAT_PRECISION,
            ),
            "ranked_results": _round_floats(ranked_results, _RESPONSE_FLOAT_PRECISION),
            "retrieval_metrics": _round_floats(
                retrieval_metrics, _RESPONSE_FLOAT_PRECISION
            ),
            "analytics": {},
            "guidance": _round_floats(_env_dict, _RESPONSE_FLOAT_PRECISION),
            "guard_notice": guard_notice,
        }

    sub.orchestrator.clear_request_llm_cost(_rid)
    return {
        "query": query,
        "answer_mode": _answer_mode_for_ranked_response(
            query_type=qt,
            is_explore_fallback=is_explore_fallback,
        ),
        "latency_ms": endpoint_latency_ms,
        "query_intelligence": _round_floats(
            query_intelligence, _RESPONSE_FLOAT_PRECISION
        ),
        "pipeline_trace": _round_floats(
            _build_pipeline_trace(
                filter_summary,
                analytics_outcome,
                outcome,
                guard_outcome,
                len(ranked_results),
                query_type=qt,
                sub=sub,
            ),
            _RESPONSE_FLOAT_PRECISION,
        ),
        "ranked_results": _round_floats(ranked_results, _RESPONSE_FLOAT_PRECISION),
        "retrieval_metrics": _round_floats(
            retrieval_metrics, _RESPONSE_FLOAT_PRECISION
        ),
        "analytics": {},
        "guidance": {},
        "guard_notice": guard_notice,
    }


@app.post("/search", tags=["Search"])
async def search(
    request: Request,
    query: str = Form(
        ...,
        description="Search query or analytics question — QI auto-classifies and routes internally",
    ),
    top_k: int = Form(
        50,
        ge=1,
        le=100,
        description="Number of ranked results to return (1–100, default 50). Capped by general.search.top_k_cap in config (default 100). Not applied to analytics queries.",
    ),  # noqa: E501
    diversity_lambda: float = Form(
        0.9,
        ge=0.0,
        le=1.0,
        description="MMR lambda for result diversity (0=max diversity, 1=max relevance). Overrides config value when provided.",
    ),
    relevance_threshold: float = Form(
        0.7,
        ge=0.0,
        le=1.0,
        description="Score threshold for labelling a result relevant in metrics (Recall, Precision, HitRate). Overrides retrieval.metrics.relevance_threshold in config when provided.",
    ),  # noqa: E501
    qie_only_mode: Optional[bool] = Form(
        False,
        description="Default false. When true, runs L0 LLM filter extract only (no retrieval). Returns slim JSON: identified_filters, find_query_params/find_query_string (FIND wire), soft_chips, decision_tier=L0_entity, latency_ms.",
    ),  # noqa: E501
    x_session_id: Optional[str] = Header(
        None,
        alias="X-Session-Id",
        description="Per-session rate-limit key. Window: 10 req/60 s. Omit to fall back to ip-keyed bucket.",
    ),
    form_request_id: Optional[str] = Form(
        None,
        alias="request_id",
        description="Optional client trace id; else server mints from identity.request_id_prefix.",
    ),
    form_search_id: Optional[str] = Form(
        None,
        alias="search_id",
        description="Optional durable search_id for feedback joins; else server mints from identity.search_id_prefix.",
    ),
) -> Any:
    """Unified search — one input box, fully autonomous routing.

    Thin wrapper around `_search_impl`: resolves ``request_id`` (trace) +
    ``search_id`` (durable), stamps both on every response, logs, and fire-and-forget
    S3 dump. Never blocks on S3 (see `s3_search_log_uploader.py`).
    """
    sub = _require_subsystems()
    try:
        identity = resolve_identity(
            sub.config.identity,
            headers=request.headers,
            form_request_id=form_request_id,
            form_search_id=form_search_id,
        )
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=str(ve)) from ve
    _wrapper_started_at = time.time()
    logger.debug(
        f"search_request_received request_id={identity.request_id} "
        f"search_id={identity.search_id} "
        f"request_id_from_client={identity.request_id_from_client} "
        f"search_id_from_client={identity.search_id_from_client} "
        f"query_len={len(query)} top_k={top_k} qie_only_mode={bool(qie_only_mode)} "
        f"has_session={bool(x_session_id)}"
    )

    result = await _search_impl(
        query=query,
        top_k=top_k,
        diversity_lambda=diversity_lambda,
        relevance_threshold=relevance_threshold,
        qie_only_mode=qie_only_mode,
        x_session_id=x_session_id,
        request_id=identity.request_id,
        search_id=identity.search_id,
    )

    if isinstance(result, JSONResponse):
        try:
            fields = json.loads(bytes(result.body))
        except (ValueError, TypeError):
            fields = {}
        if isinstance(fields, dict):
            _apply_identity_fields(
                fields, request_id=identity.request_id, search_id=identity.search_id
            )
            result = JSONResponse(status_code=result.status_code, content=fields)
        http_status = result.status_code
    else:
        if isinstance(result, dict):
            _apply_identity_fields(
                result, request_id=identity.request_id, search_id=identity.search_id
            )
        fields = result if isinstance(result, dict) else {}
        http_status = 200

    answer_mode = fields.get("answer_mode", "unknown") if isinstance(fields, dict) else "unknown"
    latency_ms = fields.get("latency_ms") if isinstance(fields, dict) else None
    retrieval_metrics = (fields.get("retrieval_metrics") or {}) if isinstance(fields, dict) else {}
    failure_mode = retrieval_metrics.get("failure_mode", "none")
    result_count = len(fields.get("ranked_results") or []) if isinstance(fields, dict) else 0
    decision_cost_usd = (
        (fields.get("query_intelligence") or {}).get("decision_cost_usd", 0.0)
        if isinstance(fields, dict)
        else 0.0
    )

    logger.info(
        f"search_request_complete request_id={identity.request_id} "
        f"search_id={identity.search_id} "
        f"http_status={http_status} answer_mode={answer_mode} latency_ms={latency_ms} "
        f"result_count={result_count} failure_mode={failure_mode} "
        f"decision_cost_usd={decision_cost_usd}"
    )

    dashboard_record = {
        "request_id": identity.request_id,
        "search_id": identity.search_id,
        "query": query,
        "top_k": top_k,
        "diversity_lambda": diversity_lambda,
        "relevance_threshold": relevance_threshold,
        "qie_only_mode": bool(qie_only_mode),
        "http_status": http_status,
        "answer_mode": answer_mode,
        "latency_ms": latency_ms,
        "result_count": result_count,
        "failure_mode": failure_mode,
        "decision_cost_usd": decision_cost_usd,
    }
    asyncio.create_task(
        asyncio.to_thread(
            upload_search_result_json,
            dashboard_record,
            identity.request_id,
            _wrapper_started_at,
        )
    )
    return result


@app.post("/feedback", tags=["Feedback"])
async def submit_feedback(
    comment: str = Form(
        ..., description="Free-text feedback comment from the end user"
    ),
    query: Optional[str] = Form(
        None, description="Search query this feedback refers to"
    ),
    search_id: Optional[str] = Form(
        None,
        description="Durable search_id from POST /search (preferred business join key)",
    ),
    request_id: Optional[str] = Form(
        None,
        description="Optional trace request_id from POST /search (observability only)",
    ),
    x_feedback_key: Optional[str] = Header(
        None,
        alias="X-Feedback-Key",
        description="Shared secret required when feedback.uat_api_key_env_var is provisioned. Blocks anonymous/scanner writes.",
    ),
) -> Dict[str, Any]:
    """Record a UAT free-text comment tied to a durable ``search_id``.

    Signals are appended to ``feedback.signal_log_path`` (JSONL, one record per line) and
    kept in the in-memory ring buffer. Mount the parent directory as a Docker named volume
    to persist records across container restarts:

        volumes:
          - ./feedback_data:/app/feedback

    ``comment`` is stripped and capped at ``feedback.uat_max_comment_chars``.
    Prefer ``search_id`` from the preceding ``POST /search`` response. Mode
    ``identity.feedback_search_id_mode`` controls missing ``search_id``
    (``soft_generate`` or ``required``). ``request_id`` is optional trace only.
    """
    sub = _require_subsystems()
    fb_cfg = sub.config.feedback
    id_cfg = sub.config.identity
    if not fb_cfg.enabled:
        raise HTTPException(status_code=503, detail="feedback signal store not enabled")
    if fb_cfg.uat_api_key_env_var:
        expected_key = os.environ.get(fb_cfg.uat_api_key_env_var, "").strip()
        if not expected_key:
            logger.warning(
                f"uat_feedback_rejected reason=env_unset env_var={fb_cfg.uat_api_key_env_var}"
            )
            raise HTTPException(
                status_code=403,
                detail=f"feedback auth misconfigured: {fb_cfg.uat_api_key_env_var} not set",
            )
        if not x_feedback_key or not hmac.compare_digest(
            x_feedback_key.strip(), expected_key
        ):
            logger.warning("uat_feedback_rejected reason=invalid_api_key")
            raise HTTPException(
                status_code=401, detail="invalid or missing X-Feedback-Key"
            )
    comment_stripped = (comment or "").strip()
    if not comment_stripped:
        raise HTTPException(status_code=422, detail="comment must be non-empty")
    if len(comment_stripped) > fb_cfg.uat_max_comment_chars:
        raise HTTPException(
            status_code=422,
            detail=f"comment exceeds max length of {fb_cfg.uat_max_comment_chars} chars",
        )
    try:
        _sid, search_correlated = resolve_feedback_search_id(
            id_cfg,
            form_search_id=search_id if isinstance(search_id, str) else None,
        )
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=str(ve)) from ve
    if not search_correlated:
        logger.info(f"feedback_search_id_missing generated_search_id={_sid}")
    try:
        _rid_validated = normalize_client_id(
            request_id if isinstance(request_id, str) else None,
            max_id_chars=id_cfg.max_id_chars,
            field_name="request_id",
            value_pattern=re.compile(id_cfg.id_value_pattern),
        )
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=str(ve)) from ve
    if _rid_validated:
        _rid = _rid_validated
    else:
        # Trace id for this feedback HTTP hop (not the business join key).
        _rid = mint_prefixed_id(id_cfg.request_id_prefix, id_cfg.id_hex_length)
        logger.info(f"feedback_request_id_missing generated_request_id={_rid}")
    signal = FeedbackSignal(
        signal_id=FeedbackSignal.new_signal_id(),
        request_id=_rid,
        search_id=_sid,
        signal_type="uat_feedback",
        payload={
            "comment": comment_stripped,
            "query": (query or "").strip() or None,
            "search_id": _sid,
        },
        signal_origin="frontend",
    )
    await sub.signal_store.record_async(signal)
    asyncio.create_task(asyncio.to_thread(upload_feedback_csv, signal))
    logger.info(
        f"uat_feedback_recorded signal_id={signal.signal_id} "
        f"search_id={signal.search_id} request_id={signal.request_id} "
        f"search_correlated={search_correlated}"
    )
    return {
        "signal_id": signal.signal_id,
        "search_id": signal.search_id,
        "request_id": signal.request_id,
        "status": "recorded",
        "recorded_at": datetime.fromtimestamp(
            signal.created_at, tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


@app.post("/feedback/calibration-label", tags=["Feedback"])
async def submit_calibration_label(
    search_id: str = Form(
        ..., description="Durable search_id this correctness label applies to"
    ),
    is_correct: bool = Form(
        ..., description="Whether the predicted intent/filter decision was correct"
    ),
    request_id: Optional[str] = Form(
        None, description="Optional trace request_id from the originating search hop"
    ),
    tier: Optional[str] = Form(
        None, description="QI tier or route that produced the prediction"
    ),
    raw_confidence: Optional[float] = Form(
        None, description="Optional confidence emitted with the prediction"
    ),
    expected_intent: Optional[str] = Form(
        None, description="Optional reviewed/expected intent label"
    ),
    predicted_intent: Optional[str] = Form(
        None, description="Optional predicted intent label"
    ),
    x_feedback_key: Optional[str] = Header(
        None,
        alias="X-Feedback-Key",
        description="Shared secret when feedback.uat_api_key_env_var is provisioned",
    ),
) -> Dict[str, Any]:
    """Record an intent correctness label for measurement dashboards.

    This writes a ``calibration_label`` FeedbackSignal keyed by durable ``search_id``.
    It does not run during `/search`; `/measurement/signals` consumes these labels later.
    """
    sub = _require_subsystems()
    fb_cfg = sub.config.feedback
    id_cfg = sub.config.identity
    if not fb_cfg.enabled:
        raise HTTPException(status_code=503, detail="feedback signal store not enabled")
    if fb_cfg.uat_api_key_env_var:
        expected_key = os.environ.get(fb_cfg.uat_api_key_env_var, "").strip()
        if not expected_key:
            logger.warning(
                f"calibration_label_rejected reason=env_unset env_var={fb_cfg.uat_api_key_env_var}"
            )
            raise HTTPException(
                status_code=403,
                detail=f"feedback auth misconfigured: {fb_cfg.uat_api_key_env_var} not set",
            )
        if not x_feedback_key or not hmac.compare_digest(
            x_feedback_key.strip(), expected_key
        ):
            logger.warning("calibration_label_rejected reason=invalid_api_key")
            raise HTTPException(
                status_code=401, detail="invalid or missing X-Feedback-Key"
            )
    try:
        sid = normalize_client_id(
            search_id,
            max_id_chars=id_cfg.max_id_chars,
            field_name="search_id",
            value_pattern=re.compile(id_cfg.id_value_pattern),
        )
        rid_opt = normalize_client_id(
            request_id if isinstance(request_id, str) else None,
            max_id_chars=id_cfg.max_id_chars,
            field_name="request_id",
            value_pattern=re.compile(id_cfg.id_value_pattern),
        )
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=str(ve)) from ve
    if not sid:
        raise HTTPException(status_code=422, detail="search_id must be non-empty")
    rid = rid_opt or mint_prefixed_id(id_cfg.request_id_prefix, id_cfg.id_hex_length)
    payload: Dict[str, Any] = {
        "is_correct": bool(is_correct),
        "search_id": sid,
    }
    for key, value in {
        "tier": tier,
        "raw_confidence": raw_confidence,
        "expected_intent": expected_intent,
        "predicted_intent": predicted_intent,
    }.items():
        if value is None:
            continue
        if isinstance(value, str):
            text = value.strip()
            if not text:
                continue
            payload[key] = text
        else:
            payload[key] = value
    signal = FeedbackSignal(
        signal_id=FeedbackSignal.new_signal_id(),
        request_id=rid,
        search_id=sid,
        signal_type="calibration_label",
        payload=payload,
        signal_origin="frontend",
    )
    await sub.signal_store.record_async(signal)
    logger.info(
        f"calibration_label_recorded signal_id={signal.signal_id} "
        f"search_id={signal.search_id} request_id={signal.request_id} "
        f"is_correct={bool(is_correct)}"
    )
    return {
        "signal_id": signal.signal_id,
        "search_id": signal.search_id,
        "request_id": signal.request_id,
        "status": "recorded",
        "signal_type": "calibration_label",
        "recorded_at": datetime.fromtimestamp(
            signal.created_at, tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


@app.get("/feedback", tags=["Feedback"])
async def get_feedback(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> Response:
    """Download uat_feedback records from S3 as a merged CSV (default: last 7 days).

    ``date_from`` / ``date_to`` format: ``YYYY-MM-DD``.
    Returns ``Content-Disposition: attachment; filename=feedback_export.csv``.
    Returns 503 when S3 is not configured (``S3_PRETRAINED_DIR`` not an s3:// URI).
    """
    today = datetime.now(tz=timezone.utc).date()
    if date_to is None:
        date_to = today.isoformat()
    if date_from is None:
        date_from = (today - timedelta(days=7)).isoformat()

    try:
        rows = await asyncio.to_thread(read_feedback_from_s3, date_from, date_to)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_FEEDBACK_CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    filename = f"feedback_{date_from}_{date_to}.csv"
    return Response(
        content=buf.getvalue().encode("utf-8"),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


class GroundFiltersBody(BaseModel):
    query: str
    identified: List[Dict[str, Any]] = Field(default_factory=list)


@app.post("/internal/l0_ground", tags=["Internal"])
def l0_ground_filters(
    body: GroundFiltersBody,
    x_harness_key: Optional[str] = Header(
        None,
        alias="X-Harness-Key",
        description=(
            "Shared secret when HARNESS_API_KEY is set (required on Katana)"
        ),
    ),
) -> Dict[str, Any]:
    """Test/harness-only: reconcile + live-inventory-ground a pre-extracted filter list
    via the same pipeline the qie_only route uses. Read-only; no DB writes.

    Lets offline extraction harnesses (e.g. LLM-only or regex-only extractors that
    don't run through /search) get their raw filter list grounded against the same
    Qdrant/ClickHouse-backed inventory as production, instead of duplicating that
    wiring locally.
    """
    _require_harness_access(x_harness_key)
    sub = _require_subsystems()
    extractor, hard_names, soft_names, grounder = _qie_grounding_context(sub)
    q_norm = normalize_query(
        body.query,
        sub.config.general.max_query_length,
        normalize=sub.config.qi.normalize,
    )
    grounded, drop_count = reconcile_and_ground_identified(
        body.identified,
        q_norm,
        extractor=extractor,
        hard_names=hard_names,
        soft_names=soft_names,
        grounder=grounder,
    )
    return {"identified_filters": grounded, "grounded_drop_count": drop_count}


# ── Offline QA harness routes ────────────────────────────────────────────────
# Test-harness-only utility routes (same framing as POST /internal/l0_ground above).
# The three offline_harness/ scripts use asyncio.run()/blocking HTTP at their own
# entrypoint — calling that from a route already running inside uvicorn's event loop
# would raise "asyncio.run() cannot be called from a running event loop" — so each is
# spawned as a subprocess, reusing its own existing CLI arg handling verbatim.

def _resolve_repo_root(here: Path) -> Path:
    """auc-semantic-search/ in monorepo; /app in Docker (flattened COPY layout).

    Local:  .../auc-semantic-search/packages/semantic-search/semantic_search/app.py
            walk finds auc-semantic-search/ (has packages/semantic-search/).
    Docker: /app/semantic_search/app.py (WORKDIR=/app; see Dockerfile COPY)
            only parents[0..2] exist; parents[3] raises IndexError and kills boot.
    """
    for candidate in here.parents:
        if (candidate / "packages" / "semantic-search").is_dir():
            return candidate
    # Flattened image: package at /app/semantic_search/ -> root = /app
    return here.parents[1] if len(here.parents) > 1 else here.parent


_REPO_ROOT = _resolve_repo_root(Path(__file__).resolve())
_HARNESS_OUTPUT_ROOT = (_REPO_ROOT / "output").resolve()
_HARNESS_RUN_LOCKS: Dict[str, asyncio.Lock] = {
    "reground": asyncio.Lock(),
    "eval": asyncio.Lock(),
}
_HARNESS_RUNS_DIR = _REPO_ROOT / "output" / "harness_runs"


def _resolve_under_root(root: Path, raw: str, *, must_exist: bool) -> Path:
    """Resolve `raw` under `root`; reject any path that escapes it (e.g. `../../etc/passwd`)."""
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = root / p
    resolved = p.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"path must stay under {root}: {raw}")
    if must_exist and not resolved.is_file():
        raise HTTPException(status_code=404, detail=f"file not found: {resolved}")
    return resolved


def _require_harness_access(x_harness_key: Optional[str]) -> None:
    """Gate ``/internal/harness/*`` and ``/internal/l0_ground``.

    - Default (unset / empty): enabled.
    - ``HARNESS_ENABLED=false|0|no``: always 403.
    - Katana (``is_katana_env()``): require matching ``X-Harness-Key`` vs
      ``HARNESS_API_KEY`` (403 if key unset).
    - Local: open unless ``HARNESS_API_KEY`` is set (then key required).
    """
    enabled_raw = os.environ.get("HARNESS_ENABLED", "true").strip().lower()
    if enabled_raw in ("0", "false", "no"):
        raise HTTPException(status_code=403, detail="harness routes disabled")
    expected = os.environ.get("HARNESS_API_KEY", "").strip()
    if is_katana_env():
        if not expected:
            raise HTTPException(
                status_code=403,
                detail="harness auth misconfigured: HARNESS_API_KEY not set",
            )
        if not x_harness_key or not hmac.compare_digest(
            x_harness_key.strip(), expected
        ):
            raise HTTPException(
                status_code=401, detail="invalid or missing X-Harness-Key"
            )
        return
    if expected:
        if not x_harness_key or not hmac.compare_digest(
            x_harness_key.strip(), expected
        ):
            raise HTTPException(
                status_code=401, detail="invalid or missing X-Harness-Key"
            )


def _validate_harness_api_form(api: str) -> str:
    """Loopback-only SSRF guard for eval-search ``api`` form field."""
    from semantic_search.eval.query_eval_runner import validate_loopback_api_base

    try:
        return validate_loopback_api_base(api)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _run_harness_subprocess(
    argv: List[str], env_overrides: Dict[str, str], log_name: str
) -> Tuple[int, float, str, Path]:
    """Run `argv` as a subprocess from `_REPO_ROOT`, teeing combined stdout/stderr to
    `output/harness_runs/{log_name}_{stamp}.log` — this file **is** the "summary text"
    artifact, capturing each script's own progress/print_summary output verbatim, with
    zero changes needed to their stdout logic.

    Returns (exit_code, duration_s, stdout_tail (~last 80 lines), log_path).
    """
    _HARNESS_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = str(int(time.time()))
    log_path = _HARNESS_RUNS_DIR / f"{log_name}_{stamp}.log"
    env = {**os.environ, **env_overrides}
    t0 = time.perf_counter()
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(_REPO_ROOT),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    lines: List[str] = []
    with open(log_path, "wb") as log_f:
        while True:
            chunk = await proc.stdout.readline()
            if not chunk:
                break
            log_f.write(chunk)
            lines.append(chunk.decode("utf-8", errors="replace").rstrip("\n"))
    exit_code = await proc.wait()
    duration_s = round(time.perf_counter() - t0, 2)
    stdout_tail = "\n".join(lines[-80:])
    return exit_code, duration_s, stdout_tail, log_path


@app.post("/internal/harness/reground-four-way", tags=["Internal"])
async def harness_reground_four_way(
    seed: bool = Form(
        default=True,
        description=(
            "seed [Boolean, default: true] — build/overwrite results from --md files "
            "instead of resuming a partial results JSON"
        ),
    ),
    md: Optional[List[str]] = Form(
        default=None,
        description=(
            "md [String, optional, repeatable] — markdown query-suite path(s), relative "
            "to repo root (default: test_search_queries.md + test_filter_queries.md)"
        ),
    ),
    out_dir: Optional[str] = Form(
        default=None,
        description=(
            "out_dir [String, optional] — output dir, relative to repo root "
            "(default: output/reground_four_way/)"
        ),
    ),
    fail_only: bool = Form(
        default=False,
        description="fail_only [Boolean, default: false] — re-run only previously DIFF/ERROR rows",
    ),
    error_only: bool = Form(
        default=False,
        description="error_only [Boolean, default: false] — re-run only previously status=ERROR rows",
    ),
    limit: Optional[int] = Form(
        default=None, ge=1, description="limit [Integer, optional] — cap number of queries"
    ),
    analysis_only: bool = Form(
        default=False,
        description="analysis_only [Boolean, default: false] — skip HTTP/LLM, rebuild report+xlsx from existing results only",
    ),
    grounding_model: Optional[str] = Form(
        default=None,
        description=(
            "grounding_model [String, optional] — override LLMJ offline model "
            "(env L0_GROUNDING_MODEL). When omitted, uses primary from "
            "task_model_allowlists.l0_entity_extraction ∩ GoCaas discovery"
        ),
    ),
    holdout_frac: float = Form(
        default=0.2,
        ge=0.0,
        lt=1.0,
        description=(
            "holdout_frac [Float, default: 0.2] — formal train/test holdout fraction "
            "for pairwise metrics (0 disables). Stratified by suite."
        ),
    ),
    holdout_seed: int = Form(
        default=42,
        description="holdout_seed [Integer, default: 42] — deterministic holdout split seed",
    ),
    x_harness_key: Optional[str] = Header(
        None,
        alias="X-Harness-Key",
        description=(
            "Shared secret when HARNESS_API_KEY is set (required on Katana)"
        ),
    ),
) -> Dict[str, Any]:
    """Test-harness-only: run `offline_harness/reground_filters_four_way.py` (4-arm LLM vs
    regex L0 filter-extraction comparison, all arms grounded via this same running
    instance's `/internal/l0_ground` + `/search`) as a subprocess. Multi-minute runtime —
    clients should set a long HTTP timeout. `SEARCH_ENDPOINT`/`L0_GROUND_ENDPOINT` are left
    unset so the subprocess's own `localhost:8085` default resolves (same-host loopback,
    always correct regardless of how this request arrived).

    ``curl -sS -X POST http://localhost:8085/internal/harness/reground-four-way -F seed=true -F limit=5``
    """
    _require_harness_access(x_harness_key)
    async with _HARNESS_RUN_LOCKS["reground"]:
        # out_dir must stay under output/ (not whole repo root).
        resolved_out_dir = (
            _resolve_under_root(_HARNESS_OUTPUT_ROOT, out_dir, must_exist=False)
            if out_dir
            else _HARNESS_OUTPUT_ROOT / "reground_four_way"
        )
        argv = [
            sys.executable,
            "-m", "semantic_search.offline_harness.reground_filters_four_way",
            "--out-dir", str(resolved_out_dir),
            "--holdout-frac", str(holdout_frac),
            "--holdout-seed", str(holdout_seed),
        ]
        if seed:
            argv.append("--seed")
        if fail_only:
            argv.append("--fail-only")
        if error_only:
            argv.append("--error-only")
        if analysis_only:
            argv.append("--analysis-only")
        if limit:
            argv += ["--limit", str(limit)]
        for m in md or []:
            argv += ["--md", str(_resolve_under_root(_REPO_ROOT, m, must_exist=True))]
        resolved_grounding = (grounding_model or "").strip() or None
        if resolved_grounding is None and not analysis_only:
            from semantic_search.config.offline_harness_defaults import (  # noqa: PLC0415
                HARNESS_EXTRACTION_TASK,
            )
            prov = getattr(_require_subsystems(), "llm_provider", None)
            if prov is None:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "LLM provider unavailable — cannot resolve harness grounding "
                        "model; pass grounding_model form field or configure llm_api_keys"
                    ),
                )
            try:
                resolved_grounding = prov.get_primary_model(HARNESS_EXTRACTION_TASK)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"No primary model for {HARNESS_EXTRACTION_TASK}: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                ) from exc
        env_overrides = (
            {"L0_GROUNDING_MODEL": resolved_grounding} if resolved_grounding else {}
        )
        exit_code, duration_s, summary_text, log_path = await _run_harness_subprocess(
            argv, env_overrides, "reground_four_way"
        )

        def _latest(pattern: str) -> Optional[Path]:
            matches = sorted(resolved_out_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
            return matches[-1] if matches else None

        analysis_md_path = _latest("analysis_*.md") if resolved_out_dir.is_dir() else None
        return {
            "exit_code": exit_code,
            "duration_s": duration_s,
            "results_json_path": str(p) if (p := _latest("results_*.json")) else None,
            "analysis_md_path": str(analysis_md_path) if analysis_md_path else None,
            "analysis_markdown": analysis_md_path.read_text() if analysis_md_path else None,
            "xlsx_path": str(p) if (p := _latest("reground_*.xlsx")) else None,
            "summary_text": summary_text,
            "log_path": str(log_path),
            "grounding_model": resolved_grounding,
        }


@app.post("/internal/harness/eval-search", tags=["Internal"])
async def harness_eval_search(
    api: str = Form(
        default="http://127.0.0.1:8085",
        description=(
            "api [String, default: http://127.0.0.1:8085] — loopback API base URL only "
            "(127.0.0.1 / localhost / ::1); non-loopback rejected (SSRF guard)"
        ),
    ),
    suite: Optional[str] = Form(
        default=None,
        description=(
            "suite [String, optional, comma-separated] — HYBRID,EXPLORE,GUIDANCE,ANALYTICS "
            "(default: all four)"
        ),
    ),
    x_harness_key: Optional[str] = Header(
        None,
        alias="X-Harness-Key",
        description=(
            "Shared secret when HARNESS_API_KEY is set (required on Katana)"
        ),
    ),
) -> Dict[str, Any]:
    """Test-harness-only: run `offline_harness/eval_test_search_queries.py` (global HYBRID
    · EXPLORE · GUIDANCE · ANALYTICS eval suite) as a subprocess and return its markdown +
    XLSX report. Multi-minute runtime — clients should set a long HTTP timeout.

    ``curl -sS -X POST http://localhost:8085/internal/harness/eval-search -F api=http://127.0.0.1:8085 -F suite=HYBRID``
    """
    _require_harness_access(x_harness_key)
    api_base = _validate_harness_api_form(api)
    async with _HARNESS_RUN_LOCKS["eval"]:
        out_dir = _REPO_ROOT / "output" / "eval_search_queries"
        before = set(out_dir.glob("eval_report_*.md")) if out_dir.is_dir() else set()
        argv = [
            sys.executable,
            "-m", "semantic_search.offline_harness.eval_test_search_queries",
            "--api", api_base,
        ]
        if suite:
            argv += ["--suite", suite]
        exit_code, duration_s, summary_text, log_path = await _run_harness_subprocess(
            argv, {}, "eval_search"
        )
        after = set(out_dir.glob("eval_report_*.md")) if out_dir.is_dir() else set()
        new_reports = sorted(after - before, key=lambda p: p.stat().st_mtime)
        report_md_path = new_reports[-1] if new_reports else None
        report_xlsx_path = report_md_path.with_suffix(".xlsx") if report_md_path else None
        return {
            "exit_code": exit_code,
            "duration_s": duration_s,
            "report_md_path": str(report_md_path) if report_md_path else None,
            "report_markdown": report_md_path.read_text() if report_md_path else None,
            "xlsx_path": str(report_xlsx_path) if report_xlsx_path and report_xlsx_path.is_file() else None,
            "summary_text": summary_text,
            "log_path": str(log_path),
        }


@app.get("/internal/harness/artifact", tags=["Internal"])
def harness_artifact(
    path: str,
    x_harness_key: Optional[str] = Header(
        None,
        alias="X-Harness-Key",
        description=(
            "Shared secret when HARNESS_API_KEY is set (required on Katana)"
        ),
    ),
) -> FileResponse:
    """Test-harness-only: download an artifact (xlsx/json/md/log) under ``output/``.

    Path must stay under ``output/`` (not whole repo / config). Escape attempts
    (e.g. ``../../../etc/passwd``) are rejected with 400.

    ``curl "http://localhost:8085/internal/harness/artifact?path=output/eval_search_queries/eval_report_....xlsx" -o out.xlsx``
    """
    _require_harness_access(x_harness_key)
    # Accept either "output/..." or a path relative to output/.
    raw = path.strip().lstrip("/")
    if raw.startswith("output/"):
        raw = raw[len("output/") :]
    resolved = _resolve_under_root(_HARNESS_OUTPUT_ROOT, raw, must_exist=True)
    return FileResponse(str(resolved), filename=resolved.name)


@app.get("/cache/stats", tags=["Status"])
def cache_stats() -> Dict[str, Dict[str, int]]:
    """Per-tier cache hit/miss counters."""
    sub = _require_subsystems()
    stats = dict(sub.orchestrator.cache_stats())
    qie_cache = _get_qie_l0_filter_cache()
    stats["qie_l0"] = {"hits": int(qie_cache.hits), "misses": int(qie_cache.misses)}
    return stats


@app.post("/cache/clear", tags=["Status"])
def cache_clear() -> Dict[str, Any]:
    """Invalidate all in-process caches used by full search **and** qie_only.

    Full-search leg (orchestrator): exact, structured, intent_plan, QI intent
    (exact+semantic), NL-SQL, explore timeout-fallback slot.

    qie_only leg (module-level): L0 filter cache — not owned by the
    orchestrator; cleared here so both modes miss after this call.

    Redis/remote tiers are not provisioned and are not touched.

    Returns per-tier eviction counts.
    """
    sub = _require_subsystems()
    evicted = dict(sub.orchestrator.clear_caches())
    evicted["qie_l0"] = _clear_qie_l0_filter_cache()
    total = sum(int(v) for v in evicted.values())
    logger.info(f"cache_clear_all evicted={total} per_tier={evicted}")
    return {"evicted": evicted, "total_evicted": total}


@app.get("/resilience/health", tags=["Status"])
def resilience_health() -> Dict[str, Any]:
    """Per-backend health snapshot plus typed analytics substrate summary."""
    sub = _require_subsystems()
    analytics_en = sub.analytics_router is not None and bool(
        sub.analytics_router.enabled
    )
    substrate = sub.degradation_planner.plan_analytics_substrate(
        analytics_router_enabled=analytics_en
    )
    return {
        "backends": [asdict(b) for b in sub.backend_health.snapshot()],
        "analytics_substrate": asdict(substrate),
    }


@app.get("/measurement/observations", tags=["Status"])
def measurement_observations(limit: Optional[int] = None) -> Dict[str, Any]:
    """Most-recent SearchObservations in the rolling window (debug surface).

    `limit` defaults to general.max_results when omitted.
    """
    sub = _require_subsystems()
    n = (
        limit
        if (limit is not None and int(limit) > 0)
        else sub.config.general.max_results
    )
    obs = sub.measurement_store.recent(int(n))
    return {
        "count": len(obs),
        "window_size": sub.measurement_store.size(),
        "capacity": sub.measurement_store.capacity(),
        "observations": [asdict(o) for o in obs],
    }


@app.get("/measurement/signals", tags=["Status"])
def measurement_signals(sliced: bool = False) -> Dict[str, Any]:
    """Computed proxy-signal KPI report from the rolling observation window.

    Pass ``?sliced=true`` for per-intent-bucket breakdown (requires
    ``measurement.slicing.enabled=true`` in config; returns HTTP 400 otherwise).
    """
    sub = _require_subsystems()
    if sliced:
        try:
            report = sub.proxy_signal_evaluator.evaluate_sliced()
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        report = sub.proxy_signal_evaluator.evaluate()
    return asdict(report)


@app.get("/measurement/qie_only", tags=["Status"])
def measurement_qie_only() -> Dict[str, Any]:
    """Phase 1 ``qie_only`` launch gates for decision makers.

    Aggregates in-process rates from ``qie_only_complete`` / ``qie_only_failed``.
    Durable source remains CloudWatch structured logs. FoS/FIND gates are marked
    ``status=external`` (not computable inside this service).
    """
    _require_subsystems()
    from semantic_search.measurement.qie_only_launch import get_qie_only_launch_stats

    return get_qie_only_launch_stats().snapshot()


# ---------------------------------------------------------------------------
# Data Build — seed corpus generation + index population
# ---------------------------------------------------------------------------


def _resolve_mode(
    mode: str, last_params: Optional[Dict[str, Any]], strategy: str, lookback_days: int
) -> str:
    """Return the effective build mode for the current request.

    'auto' picks 'rebuild' when the window matches the last build (same
    strategy + lookback_days), otherwise 'append'.
    """
    if mode != "auto":
        return mode
    if last_params is None:
        return "rebuild"
    if (
        last_params.get("strategy") == strategy
        and last_params.get("lookback_days") == lookback_days
    ):
        return "rebuild"
    return "append"


async def _clear_for_rebuild(sub: Subsystems) -> None:
    """Wipe in-memory indexes and drop+recreate the Qdrant collection for a clean reload.

    Recreate is mandatory after delete: OfflineIndexer.upsert against a missing
    collection raises UnexpectedResponse 404, which FastAPI turns into opaque 500.
    Ensuring here (before Athena fetch) closes that gap even if the first page's
    ``_ensure_collection`` is skipped or fails later.
    """
    if isinstance(sub.vector_index, InMemoryVectorIndex):
        sub.vector_index._items.clear()  # noqa: SLF001
    if isinstance(sub.structured_index, InMemoryStructuredIndex):
        sub.structured_index._items.clear()  # noqa: SLF001
    deleted = False
    qclient = None
    coll_name = None
    if (
        sub.qdrant_factory is not None
        and sub.qdrant_factory.available
        and sub.qdrant_factory.client is not None
    ):
        qclient = sub.qdrant_factory.client
        coll_name = sub.qdrant_factory.collection_name
        try:
            exists = await qclient.collection_exists(coll_name)
            if exists:
                await qclient.delete_collection(coll_name, timeout=120)
                deleted = True
                logger.info(
                    f"seed_rebuild_collection_deleted collection={coll_name}"
                )
        except Exception as _e:  # noqa: BLE001
            logger.warning(f"seed_rebuild_delete_collection_failed error={_e}")
            raise
    if deleted or (
        qclient is not None
        and coll_name is not None
        and not await qclient.collection_exists(coll_name)
    ):
        if sub.offline_indexer is None:
            raise RuntimeError(
                "rebuild wiped Qdrant collection but offline_indexer is None; "
                "cannot recreate auctions_listings"
            )
        # Same schema path OfflineIndexer.run uses — must run before Athena fetch
        # so a long merge cannot race a missing collection on first upsert.
        await sub.offline_indexer._ensure_collection()  # noqa: SLF001
        if not await qclient.collection_exists(coll_name):
            raise RuntimeError(
                f"rebuild recreate failed: collection={coll_name} still missing "
                f"after OfflineIndexer._ensure_collection"
            )
        logger.info(
            f"seed_rebuild_collection_recreated "
            f"collection={sub.offline_indexer.collection_name}"
        )


async def _seed_background_task() -> None:
    """Background rebuild loop driven by `vectorization.seed.schedule` config."""
    sub = app_state.subsystems
    if sub is None:
        return
    vec_cfg = sub.config.vectorization
    if vec_cfg is None or vec_cfg.seed is None:
        return
    schedule = vec_cfg.seed.schedule
    if schedule is None or not schedule.enabled:
        return

    if schedule.run_at_hour_utc is not None:
        now = datetime.now(timezone.utc)
        next_run = now.replace(
            hour=schedule.run_at_hour_utc, minute=0, second=0, microsecond=0
        )
        if next_run <= now:
            next_run += timedelta(days=1)
        delay = (next_run - now).total_seconds()
    else:
        delay = schedule.interval_hours * 3600

    await asyncio.sleep(delay)

    while True:
        try:
            _tbl0 = (
                vec_cfg.seed.database.tables[0]
                if vec_cfg.seed.database.tables
                else None
            )
            _abf_cfg = vec_cfg.analytics_backfill
            if _tbl0 is None:
                raise RuntimeError(
                    "vectorization.seed.database.tables[0] required for scheduled data-build"
                )
            if _abf_cfg is None:
                raise RuntimeError(
                    "vectorization.analytics_backfill required for scheduled data-build"
                )
            result = await data_build_full(
                seed_lookback_days=_tbl0.lookback_days,
                analytics_lookback_days=_abf_cfg.lookback_days,
                seed_strategy=_tbl0.strategy,
                seed_mode=schedule.seed_mode,
            )
            _qdrant_res = result.get("qdrant") or {}
            _ch_res = result.get("clickhouse") or {}
            _qdrant_ok = "error" not in _qdrant_res
            _ch_ok = "error" not in _ch_res
            logger.info(
                f"seed_build_scheduled_complete triggered_by=scheduled "
                f"qdrant_ok={_qdrant_ok} "
                f"qdrant_indexed={_qdrant_res.get('qdrant_indexing', {}).get('points_upserted', 0)} "
                f"ch_ok={_ch_ok} "
                f"ch_rows={_ch_res.get('clickhouse', {}).get('rows_written', 0)}"
            )
        except asyncio.CancelledError:
            break
        except Exception as _e:  # noqa: BLE001
            logger.warning(
                f"seed_build_scheduled_failed error_type={type(_e).__name__} error={_e}"
            )

        try:
            await asyncio.sleep(schedule.interval_hours * 3600)
        except asyncio.CancelledError:
            break


@app.post("/data-build/seed", tags=["Data Build"])
async def data_build_seed(
    source: str = Form(
        default="daily_snapshot",
        description=(
            "source [String, default: daily_snapshot] — "
            "'daily_snapshot' queries the database at vectorization.seed.database.name (T+1 daily bulk snapshot); "
            "'realtime' queries the live replica configured in database.realtime_name "
            "(set via YAML overlay when a realtime feed is available)"
        ),
    ),
    strategy: str = Form(
        default="datewise",
        description=(
            "strategy [String, default: datewise] — "
            "'datewise' fetches records where auction_end_utc_ts >= NOW() - lookback_days (recommended); "
            "'count' fetches the latest max_records rows ordered by auction_end_utc_ts DESC"
        ),
    ),
    lookback_days: int = Form(
        default=14,
        ge=1,
        le=730,
        description=(
            "lookback_days [Integer, default: 14, range: 1–730] — "
            "Rolling window in days for the datewise strategy (auction_end_utc_ts >= NOW() - N days). "
            "Default 14 matches ClickHouse auction_audit_cln TTL (14 days): covers the 90-day analytics "
            "window with a 30-day safety buffer. Auction lifecycle max is 72 days (analysis_auction.md §2) "
            "so 14 days retains all historical auctions before cleanup."
        ),
    ),
    max_records: int = Form(
        default=1_000_000,
        ge=1,
        le=30_000_000,
        include_in_schema=False,
    ),
    mode: str = Form(
        default="auto",
        description=(
            "mode [String, default: auto] — "
            "'auto' wipes+reloads when strategy+lookback_days match the last build, appends otherwise; "
            "'rebuild' always wipes existing data and does a full reload; "
            "'append' always upserts without clearing existing data"
        ),
    ),
) -> Dict[str, Any]:
    """Vectorize domain records from ``{vectorization.seed.database.name}.auction_audit_cln``.

    Fetches all four in-scope auction types from the configured database instance
    and loads them into the vector + structured indexes:

    - Type 16 — GoDaddy AutoExtend (bid auction, ~10 days active, ~363K active records)
    - Type 20 — GoDaddy BuyNow / Closeouts (fixed price $50->$5, ~5 days, ~12K active records)
    - Type 38 — Partner AutoExtend (bid auction, ~10 days active, ~409K active records)
    - Type 39 — Partner Closeout (fixed price $11->$5, ~5 days, ~10K active records)

    Types 15 and 33 are excluded — no product definition; they represent ~84% of
    the raw Athena table volume and must never be indexed.

    **Column selection is automatic.** The fetch query selects all columns defined
    in ``db_seed_source.py`` (domain name, TLD, price, GoValue, bid count, auction
    type, SEO signals, etc.) from the source table. You do not specify columns —
    the query is fixed. If a column is absent in the source (e.g. GoValue not yet
    backfilled), it is detected as a missing column, defaulted to 0 in the indexed
    payload, and its corresponding filter entity is marked unavailable for the
    session. Run ``/data-build/status`` to see which columns defaulted.

    **Recommended data range (``datewise`` strategy):**
    Use ``lookback_days=120`` to align with the ClickHouse ``auction_audit_cln``
    TTL (120 days). This ensures:
    - 90-day analytics queries always have a full 30-day safety buffer.
    - Auction lifecycle max is 72 days (``analysis_auction.md §2``), so 120 days
      captures all historical auctions needed before TTL cleanup.
    - Active Qdrant inventory at any time is ~417K records (types 16+20+38+39
      combined), well within the 1M index ceiling.
    For the default ``count`` strategy, ``max_records`` caps the fetch regardless
    of date; use ``datewise`` + ``lookback_days=120`` for time-bounded rebuilds.

    ``source`` selects the database instance:
    - ``daily_snapshot`` — ``vectorization.seed.database.name`` (T+1, default)
    - ``realtime``        — live replica; configure ``database.realtime_name``
                           in YAML before using this option

    ``mode`` controls rebuild vs append behaviour:
    - ``auto``    — smart: rebuilds when the window matches the last run, appends otherwise
    - ``rebuild`` — always wipes indexes and collection before loading
    - ``append``  — always upserts without clearing

    **Athena-side merge** (``run_token="seed"``) builds a per-run set of
    ``stg_*``/``final_*`` tables in the configured scratch database
    (``vectorization.seed.database.merge_database``, override via
    ``ATHENA_TMP_DB`` — see ``vectorization.seed_merge``), pages the merged
    final table, and drops every table it created once this run's Qdrant and
    ClickHouse upserts have both finished (success or failure). Concurrent
    calls to this endpoint serialize on an internal lock rather than racing
    on that shared scratch database.
    """
    if source not in ("daily_snapshot", "realtime"):
        raise HTTPException(
            status_code=422, detail="source must be 'daily_snapshot' or 'realtime'"
        )
    if strategy not in ("datewise", "count"):
        raise HTTPException(
            status_code=422, detail="strategy must be 'datewise' or 'count'"
        )
    if mode not in ("auto", "rebuild", "append"):
        raise HTTPException(
            status_code=422, detail="mode must be 'auto', 'rebuild', or 'append'"
        )

    started_at_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    sub = _require_subsystems()
    vec_cfg = sub.config.vectorization
    if vec_cfg is None or vec_cfg.seed is None:
        raise HTTPException(
            status_code=503, detail="vectorization.seed block not configured"
        )

    # Guard: refuse to vectorize when Qdrant is the backend but is unreachable.
    # Silently succeeding with 0 points is worse than a clear error.
    if sub.qdrant_factory is not None and sub.qdrant_factory.available:
        _qhost = sub.qdrant_factory.config.host
        _qport = int(sub.qdrant_factory.config.port)
        if not await _ping_qdrant(
            _qhost, _qport, secure=bool(sub.qdrant_factory.config.https)
        ):
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Qdrant is not reachable at {_qhost}:{_qport}. "
                    f"Start it with: docker run -d -p {_qport}:6333 -p {int(sub.qdrant_factory.config.grpc_port)}:6334 "
                    f"-v ~/.qdrant/storage:/qdrant/storage qdrant/qdrant"
                ),
            )
        # Validate indexing encoder dim matches the Qdrant collection's dense_dim.
        # A mismatch after a model swap causes silent 0-point upserts; fail loudly here
        # before spending time on the data fetch.
        _idx_enc = (
            sub.indexing_encoder
            if sub.indexing_encoder is not None
            else (
                sub.shortlist_encoder
                if sub.shortlist_encoder is not None
                else sub.encoder
            )
        )
        _expected_dim = sub.config.retrieval.vector.embedding_dim
        _actual_dim = getattr(_idx_enc, "dim", None)
        if _actual_dim is not None and _actual_dim != _expected_dim:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"indexing_encoder.dim={_actual_dim} != "
                    f"retrieval.vector.embedding_dim={_expected_dim}; "
                    f"model swap requires re-initializing the Qdrant collection "
                    f"with dim={_actual_dim} and restarting"
                ),
            )

    seed_cfg = vec_cfg.seed
    db_cfg = seed_cfg.database

    # Enforce the YAML-configured max_records_cap before clamping to the hard cap.
    effective_max = min(max_records, db_cfg.max_records_cap)

    # Apply form overrides onto shallow copies — never mutate live config.
    updated_tables = [
        replace(
            t, strategy=strategy, lookback_days=lookback_days, max_records=effective_max
        )
        for t in db_cfg.tables
    ]
    seed_cfg = replace(
        seed_cfg, source=source, database=replace(db_cfg, tables=updated_tables)
    )

    effective_mode = _resolve_mode(
        mode, app_state._last_build_params, strategy, lookback_days
    )
    if effective_mode == "rebuild":
        # Hold the same lock the paging loop uses below: without this, a second
        # overlapping /data-build/seed call can delete+recreate the collection
        # while this call is still mid-upsert into it (404 on the next upsert).
        async with _SEED_BUILD_LOCK:
            try:
                await _clear_for_rebuild(sub)
            except HTTPException:
                raise
            except Exception as _wipe_exc:  # noqa: BLE001 — surface wipe/recreate as 503
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"rebuild wipe/recreate failed: "
                        f"{type(_wipe_exc).__name__}: {_wipe_exc}"
                    ),
                ) from _wipe_exc

    _history_extra = {
        "mode": effective_mode,
        "source": source,
        "strategy": strategy,
        "lookback_days": lookback_days,
    }

    # Only pass in-memory indexes — Qdrant-backed indexes have no .add()
    # and are populated exclusively via offline_indexer.run(page_docs) below.
    # Independent of fetched data, so resolved once before the page loop.
    _mem_vec = (
        sub.vector_index if isinstance(sub.vector_index, InMemoryVectorIndex) else None
    )
    _mem_str = (
        sub.structured_index if isinstance(sub.structured_index, InMemoryStructuredIndex) else None
    )
    # Qdrant-only path: both in-memory indexes are absent. load_seed_into_indexes
    # is still called (to populate in-memory explore sources as a ClickHouse
    # fallback) but with vector_index/structured_index/pipeline=None so it skips
    # dense/BM25 encoding that would otherwise be discarded — the offline indexer
    # is what actually seeds Qdrant.
    _qdrant_only = _mem_vec is None and _mem_str is None
    # ClickHouse executor intentionally unused here — seed is Qdrant-only.
    # CH writes go through data_build_analytics_backfill only.
    _ch_exec = None

    docs_total = 0
    fetch_meta: Dict[str, Any] = {}
    tables_seen: set = set()
    all_missing: List[str] = []
    total_tables_skipped = 0
    resolved_meta: Dict[str, Any] = {}
    load_totals = dict(
        documents_offered=0, vector_indexed=0, structured_indexed=0,
        bm25_encoded=0, bm25_skipped=0, skipped=0, elapsed_ms=0.0,
    )
    qdrant_totals = dict(points_upserted=0, points_skipped=0, points_seen=0, failures=0, bm25_encoded=0, bm25_skipped=0)
    qdrant_available_all = True
    qdrant_ran = False
    ch_totals = dict(rows_attempted=0, rows_written=0, batches=0, errors=0, elapsed_ms=0.0, first_error=None)
    ch_ran = False
    _ch_schema_pending = True

    def _partial() -> Dict[str, Any]:
        p: Dict[str, Any] = {"fetch": fetch_meta}
        if qdrant_ran:
            p["qdrant_indexing"] = {
                "points_upserted": qdrant_totals["points_upserted"],
                "points_skipped": qdrant_totals["points_skipped"],
                "qdrant_available": qdrant_available_all,
                "failures": qdrant_totals["failures"],
            }
        if ch_ran:
            p["clickhouse_seed"] = {
                "rows_written": ch_totals["rows_written"],
                "errors": ch_totals["errors"],
                "elapsed_ms": round(ch_totals["elapsed_ms"], 1),
            }
        return p

    _set_ingest_progress("fetch", 0, 0)
    run_token = "seed"
    fetch_elapsed_ms = 0.0
    _prev_ts = time.monotonic()
    timing = StageTimingSession(seed_cfg.stage_timing)
    page_gate = bool(timing.config.log_page_stages)
    async with _SEED_BUILD_LOCK, aclosing(
        _resolve_seed_pages(seed_cfg, sub.config, run_token=run_token, timing=timing)
    ) as pages:
        pages_iter = pages.__aiter__()
        while True:
            try:
                page_docs, missing_cols, table_name, base_meta = await pages_iter.__anext__()
            except StopAsyncIteration:
                break
            except DataIngestInterruptedError as _die:
                return _data_ingest_interrupted_response(
                    endpoint="/data-build/seed", exc=_die, started_at=started_at_iso,
                    partial=_partial(), history_extra=_history_extra,
                )
            except asyncio.CancelledError as _ce:
                return _data_ingest_interrupted_response(
                    endpoint="/data-build/seed",
                    exc=DataIngestInterruptedError(
                        f"Seed fetch cancelled after partial progress (docs={docs_total}).",
                        stage="fetch", records_completed=docs_total,
                        records_attempted=docs_total, reason="cancelled",
                        detail=type(_ce).__name__,
                    ),
                    started_at=started_at_iso, partial=_partial(), history_extra=_history_extra,
                )
            except (ValidationError, AgentSearchError) as e:
                raise HTTPException(status_code=422, detail=str(e)) from e
            except Exception as _fetch_exc:  # noqa: BLE001 — Athena/boto/SDK surface
                logger.warning(
                    f"data_build_seed_fetch_failed docs_so_far={docs_total} "
                    f"error_type={type(_fetch_exc).__name__} error={_fetch_exc}"
                )
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"seed fetch failed: "
                        f"{type(_fetch_exc).__name__}: {_fetch_exc}"
                    ),
                ) from _fetch_exc

            _now = time.monotonic()
            fetch_elapsed_ms += (_now - _prev_ts) * 1000.0
            resolved_meta = base_meta
            if table_name is None:
                total_tables_skipped = base_meta.get("tables_skipped", 0)
                _prev_ts = time.monotonic()
                continue
            tables_seen.add(table_name)
            for c in missing_cols:
                if c not in all_missing:
                    all_missing.append(c)
            docs_total += len(page_docs)
            _set_ingest_progress("fetch", docs_total, docs_total)

            try:
                with timing.stage(
                    "load_indexes",
                    gate=page_gate,
                    table=table_name,
                    docs=len(page_docs),
                ):
                    if not _qdrant_only:
                        page_load_summary = await load_seed_into_indexes(
                            documents=page_docs,
                            vector_index=_mem_vec,
                            structured_index=_mem_str,
                            encoder=sub.indexing_encoder
                            if sub.indexing_encoder is not None
                            else (sub.shortlist_encoder if sub.shortlist_encoder is not None else sub.encoder),
                            trending_source=sub.explore_trending_source,
                            ending_soon_source=sub.explore_ending_soon_source,
                            pipeline=sub.doc_vectorization_pipeline,
                            idempotency_key=vec_cfg.indexer.idempotency_key_field,
                            batch_yield_size=seed_cfg.batch_yield_size,
                            encode_batch_size=seed_cfg.encode_batch_size,
                        )
                    else:
                        page_load_summary = await load_seed_into_indexes(
                            documents=page_docs,
                            vector_index=None,
                            structured_index=None,
                            encoder=sub.indexing_encoder
                            if sub.indexing_encoder is not None
                            else (sub.shortlist_encoder if sub.shortlist_encoder is not None else sub.encoder),
                            trending_source=sub.explore_trending_source,
                            ending_soon_source=sub.explore_ending_soon_source,
                            pipeline=None,
                            idempotency_key=vec_cfg.indexer.idempotency_key_field,
                            batch_yield_size=seed_cfg.batch_yield_size,
                            encode_batch_size=seed_cfg.encode_batch_size,
                        )
                for _k in ("documents_offered", "vector_indexed", "structured_indexed", "bm25_encoded", "bm25_skipped", "skipped"):
                    load_totals[_k] += getattr(page_load_summary, _k)
                load_totals["elapsed_ms"] += page_load_summary.elapsed_ms
            except DataIngestInterruptedError as _die:
                return _data_ingest_interrupted_response(
                    endpoint="/data-build/seed", exc=_die, started_at=started_at_iso,
                    partial=_partial(), history_extra=_history_extra,
                )
            except asyncio.CancelledError as _ce:
                return _data_ingest_interrupted_response(
                    endpoint="/data-build/seed",
                    exc=DataIngestInterruptedError(
                        f"Seed load cancelled after partial progress (docs={docs_total}).",
                        stage="fetch", records_completed=docs_total,
                        records_attempted=docs_total, reason="cancelled",
                        detail=type(_ce).__name__,
                    ),
                    started_at=started_at_iso, partial=_partial(), history_extra=_history_extra,
                )
            except (ValidationError, AgentSearchError) as e:
                raise HTTPException(status_code=422, detail=str(e)) from e

            # Qdrant-only sink on /data-build/seed. ClickHouse population is
            # exclusively POST /data-build/analytics-backfill (GHA runs that
            # sequentially after seed success when clickhouse.enabled).
            _do_qdrant = sub.offline_indexer is not None
            _do_ch = False
            _ch_batch = int(seed_cfg.clickhouse_batch_size)
            _ch_target = ""
            _ch_snapshot = ""
            if _do_ch:
                _ch_target = sub.config.nl_to_sql.analytics.seed_target_table
                _ch_snapshot = sub.config.nl_to_sql.analytics.seed_snapshot_table

            async def _page_qdrant_index():
                _set_ingest_progress("qdrant_index", qdrant_totals["points_upserted"], docs_total)
                with timing.stage(
                    "qdrant_index",
                    gate=page_gate,
                    table=table_name,
                    docs=len(page_docs),
                ):
                    return await sub.offline_indexer.run(page_docs, timing=timing)

            async def _page_clickhouse_seed():
                nonlocal _ch_schema_pending
                _set_ingest_progress("clickhouse_seed", ch_totals["rows_written"], docs_total)
                _ensure = True
                if seed_cfg.clickhouse_ensure_schema_once:
                    _ensure = _ch_schema_pending
                    _ch_schema_pending = False
                with timing.stage(
                    "clickhouse_seed",
                    gate=page_gate,
                    table=table_name,
                    docs=len(page_docs),
                ):
                    return await insert_seed_to_clickhouse(
                        page_docs,
                        _ch_exec,
                        ensure_schema=_ensure,
                        batch_size=_ch_batch,
                        insert_timeout_seconds=seed_cfg.clickhouse_insert_timeout_seconds,
                        schema_timeout_seconds=seed_cfg.clickhouse_schema_timeout_seconds,
                        target_table=_ch_target,
                        snapshot_table=_ch_snapshot,
                    )

            def _apply_qdrant_summary(page_indexer_summary) -> None:
                nonlocal qdrant_available_all
                for _k in ("points_upserted", "points_skipped", "points_seen", "failures", "bm25_encoded", "bm25_skipped"):
                    qdrant_totals[_k] += getattr(page_indexer_summary, _k, 0)
                qdrant_available_all = qdrant_available_all and page_indexer_summary.qdrant_available
                _set_ingest_progress("qdrant_index", qdrant_totals["points_upserted"], qdrant_totals["points_seen"])
                if _qdrant_only:
                    # In-memory indexes absent: fold offline-indexer stats into load_totals.
                    load_totals["documents_offered"] += page_indexer_summary.points_seen - page_load_summary.documents_offered
                    load_totals["vector_indexed"] += page_indexer_summary.points_upserted - page_load_summary.vector_indexed
                    load_totals["bm25_encoded"] += page_indexer_summary.bm25_encoded - page_load_summary.bm25_encoded
                    load_totals["bm25_skipped"] += page_indexer_summary.bm25_skipped - page_load_summary.bm25_skipped
                    load_totals["skipped"] += page_indexer_summary.points_skipped - page_load_summary.skipped

            def _apply_ch_summary(_ch_write) -> None:
                ch_totals["rows_attempted"] += _ch_write.rows_attempted
                ch_totals["rows_written"] += _ch_write.rows_written
                ch_totals["batches"] += _ch_write.batches
                ch_totals["errors"] += _ch_write.errors
                ch_totals["elapsed_ms"] += _ch_write.elapsed_ms
                if _ch_write.first_error and ch_totals["first_error"] is None:
                    ch_totals["first_error"] = _ch_write.first_error
                logger.info(
                    f"data_build_ch_seed_complete table={table_name} rows_written={_ch_write.rows_written} "
                    f"errors={_ch_write.errors}"
                )

            def _handle_sink_result(res, *, sink: str):
                if isinstance(res, DataIngestInterruptedError):
                    return _data_ingest_interrupted_response(
                        endpoint="/data-build/seed", exc=res, started_at=started_at_iso,
                        partial=_partial(), history_extra=_history_extra,
                    )
                if isinstance(res, asyncio.CancelledError):
                    return _data_ingest_interrupted_response(
                        endpoint="/data-build/seed",
                        exc=DataIngestInterruptedError(
                            f"{sink} cancelled after partial progress (docs={docs_total}).",
                            stage=sink,
                            records_completed=int((app_state._ingest_progress or {}).get("records_completed") or 0),
                            records_attempted=docs_total, reason="cancelled",
                            detail=type(res).__name__,
                        ),
                        started_at=started_at_iso, partial=_partial(), history_extra=_history_extra,
                    )
                if isinstance(res, (RuntimeError, OSError, ValueError, TypeError, AgentSearchError)):
                    logger.warning(
                        f"data_build_{sink}_failed table={table_name} "
                        f"error_type={type(res).__name__} error={str(res)}"
                    )
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            f"{sink} failed table={table_name}: "
                            f"{type(res).__name__}: {res}"
                        ),
                    ) from res
                if isinstance(res, BaseException):
                    logger.warning(
                        f"data_build_{sink}_failed table={table_name} "
                        f"error_type={type(res).__name__} error={str(res)}"
                    )
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            f"{sink} failed table={table_name}: "
                            f"{type(res).__name__}: {res}"
                        ),
                    ) from res
                return res

            if _do_qdrant and _do_ch:
                qdrant_ran = True
                ch_ran = True
                _qi_res, _ch_res = await asyncio.gather(
                    _page_qdrant_index(),
                    _page_clickhouse_seed(),
                    return_exceptions=True,
                )
                _qi_out = _handle_sink_result(_qi_res, sink="qdrant_index")
                if isinstance(_qi_out, JSONResponse):
                    return _qi_out
                _ch_out = _handle_sink_result(_ch_res, sink="clickhouse_seed")
                if isinstance(_ch_out, JSONResponse):
                    return _ch_out
                if _qi_out is not None:
                    _apply_qdrant_summary(_qi_out)
                if _ch_out is not None:
                    _apply_ch_summary(_ch_out)
            elif _do_qdrant:
                qdrant_ran = True
                try:
                    _apply_qdrant_summary(await _page_qdrant_index())
                except DataIngestInterruptedError as _die:
                    return _data_ingest_interrupted_response(
                        endpoint="/data-build/seed", exc=_die, started_at=started_at_iso,
                        partial=_partial(), history_extra=_history_extra,
                    )
                except asyncio.CancelledError as _ce:
                    return _data_ingest_interrupted_response(
                        endpoint="/data-build/seed",
                        exc=DataIngestInterruptedError(
                            f"Qdrant indexing cancelled after partial progress (docs={docs_total}).",
                            stage="qdrant_index",
                            records_completed=int((app_state._ingest_progress or {}).get("records_completed") or 0),
                            records_attempted=docs_total, reason="cancelled",
                            detail=type(_ce).__name__,
                        ),
                        started_at=started_at_iso, partial=_partial(), history_extra=_history_extra,
                    )
                except (RuntimeError, OSError, ValueError, TypeError, AgentSearchError) as _ie:
                    logger.warning(
                        f"data_build_offline_indexer_failed table={table_name} error_type={type(_ie).__name__} error={str(_ie)}"
                    )
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            f"Qdrant indexing failed table={table_name}: "
                            f"{type(_ie).__name__}: {_ie}"
                        ),
                    ) from _ie
                except Exception as _ie:  # noqa: BLE001 — Qdrant SDK UnexpectedResponse etc.
                    logger.warning(
                        f"data_build_offline_indexer_failed table={table_name} "
                        f"error_type={type(_ie).__name__} error={str(_ie)}"
                    )
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            f"Qdrant indexing failed table={table_name}: "
                            f"{type(_ie).__name__}: {_ie}"
                        ),
                    ) from _ie
            elif _do_ch:
                ch_ran = True
                try:
                    _apply_ch_summary(await _page_clickhouse_seed())
                except DataIngestInterruptedError as _die:
                    return _data_ingest_interrupted_response(
                        endpoint="/data-build/seed", exc=_die, started_at=started_at_iso,
                        partial=_partial(), history_extra=_history_extra,
                    )
                except asyncio.CancelledError as _ce:
                    return _data_ingest_interrupted_response(
                        endpoint="/data-build/seed",
                        exc=DataIngestInterruptedError(
                            f"ClickHouse seed cancelled after partial progress (docs={docs_total}).",
                            stage="clickhouse_seed",
                            records_completed=int((app_state._ingest_progress or {}).get("records_completed") or 0),
                            records_attempted=docs_total, reason="cancelled",
                            detail=type(_ce).__name__,
                        ),
                        started_at=started_at_iso, partial=_partial(), history_extra=_history_extra,
                    )
                except (RuntimeError, OSError, ValueError, TypeError, AgentSearchError) as _ce:
                    logger.warning(
                        f"data_build_ch_seed_failed table={table_name} error_type={type(_ce).__name__} error={str(_ce)}"
                    )

            _prev_ts = time.monotonic()

    # Persist missing columns so filters targeting absent data are skipped.
    # Computed once, after all pages, as the union across every table's fallback
    # (mirrors _resolve_seed_docs's single-shot union for the unstaged path).
    _perm_missing = set(sub.config.general.search.permanently_unavailable_columns)
    app_state._missing_data_columns = set(all_missing) | _perm_missing
    _all_unavailable_ents = _get_unavailable_filter_entities(app_state._missing_data_columns)
    if _all_unavailable_ents:
        set_unavailable_filter_keys(_all_unavailable_ents)
    else:
        set_unavailable_filter_keys(set())
    if all_missing:
        logger.warning(
            f"data_build_missing_columns columns={sorted(all_missing)} "
            f"filter_entities_skipped={sorted(_all_unavailable_ents)}"
        )

    tables_queried = len(tables_seen)
    fetch_meta = {
        "source": resolved_meta.get("source", seed_cfg.source),
        "fetch": {
            "database": resolved_meta.get("database", seed_cfg.database.name),
            "documents": docs_total,
            "tables_queried": tables_queried,
            "tables_skipped": total_tables_skipped if tables_queried == 0 else len(seed_cfg.database.tables) - tables_queried,
            "elapsed_ms": round(fetch_elapsed_ms, 1),
            "missing_columns": all_missing,
        },
    }

    if _ch_exec is not None:
        app_state._last_ch_seed_result = {
            "rows_attempted": ch_totals["rows_attempted"],
            "rows_written": ch_totals["rows_written"],
            "batches": ch_totals["batches"],
            "errors": ch_totals["errors"],
            "elapsed_ms": round(ch_totals["elapsed_ms"], 1),
            "first_error": ch_totals["first_error"],
        }

    app_state._ingest_progress = None
    app_state._last_build_params = {
        "strategy": strategy,
        "lookback_days": lookback_days,
        "source": source,
    }
    timing.log_job_complete(
        endpoint="/data-build/seed",
        mode=effective_mode,
        source=source,
        documents=docs_total,
    )
    _timing_summary = timing.summary() if timing.config.include_in_response else None
    _history_entry: Dict[str, Any] = {
        "build_id": str(uuid.uuid4())[:8],
        "triggered_by": "manual",
        "mode": effective_mode,
        "source": source,
        "strategy": strategy,
        "lookback_days": lookback_days,
        "max_records": effective_max,
        "started_at": started_at_iso,
        "documents_offered": load_totals["documents_offered"],
        "vector_indexed": load_totals["vector_indexed"],
        "bm25_encoded": load_totals["bm25_encoded"],
        "bm25_skipped": load_totals["bm25_skipped"],
        "skipped": load_totals["skipped"],
        "elapsed_ms": round(load_totals["elapsed_ms"], 1),
        "status": "success",
    }
    if _timing_summary is not None:
        _history_entry["stage_timing"] = {
            "job_elapsed_ms": _timing_summary["job_elapsed_ms"],
            "elapsed_ms_by_stage": _timing_summary["elapsed_ms_by_stage"],
        }
    app_state._build_history.append(_history_entry)

    _result: Dict[str, Any] = {
        "status": "completed",
        "mode": effective_mode,
        **fetch_meta,
        "encoder": {
            "model_name": sub.config.qi.encoder.model_name,
            "dim": sub.config.retrieval.vector.embedding_dim,
            "doc_prefix": sub.config.vectorization.encoder_query_prefix
            if sub.config.vectorization is not None
            else "",
        },
        "loading": {**load_totals, "elapsed_ms": round(load_totals["elapsed_ms"], 1)},
        "qdrant_indexing": (
            {
                "points_upserted": qdrant_totals["points_upserted"],
                "points_skipped": qdrant_totals["points_skipped"],
                "qdrant_available": qdrant_available_all,
                "failures": qdrant_totals["failures"],
                "bm25_encoded": qdrant_totals["bm25_encoded"],
                "bm25_skipped": qdrant_totals["bm25_skipped"],
            }
            if qdrant_ran else None
        ),
        "clickhouse_seed": (
            {
                "rows_attempted": ch_totals["rows_attempted"],
                "rows_written": ch_totals["rows_written"],
                "batches": ch_totals["batches"],
                "errors": ch_totals["errors"],
                "elapsed_ms": round(ch_totals["elapsed_ms"], 1),
                "first_error": ch_totals["first_error"],
            }
            if ch_ran else None
        ),
    }
    if _timing_summary is not None:
        _result["stage_timing"] = _timing_summary
    return _result


@app.post("/data-build/analytics-backfill", tags=["Data Build"])
async def data_build_analytics_backfill(
    lookback_days: int = Form(
        default=180,
        ge=1,
        le=730,
        description=(
            "lookback_days [Integer, default: 180, range: 1–730] — "
            "Rolling window in days for the Athena query against auction_audit_cln. "
            "180 = two quarters (recommended; fits 22 GB ClickHouse budget at ~10–16 GB compressed). "
            "Default read from vectorization.analytics_backfill.lookback_days in base.yaml."
        ),
    ),
    max_records: int = Form(
        default=25_000_000,
        ge=1,
        le=50_000_000,
        description=(
            "max_records [Integer, default: 25 000 000, range: 1–50 000 000] — "
            "Row cap applied to the Athena query. "
            "Default read from vectorization.analytics_backfill.max_records in base.yaml."
        ),
    ),
) -> Dict[str, Any]:
    """Populate ClickHouse signals_platform_cln.auction_audit_cln with a historical window from auction_audit_cln.

    Decoupled from POST /data-build/seed so Qdrant (active-only, 14-day window)
    and ClickHouse (historical, up to 730-day window) are sized independently.

    Trigger paths:
    - Manual: POST this endpoint directly.
    - Boot: called from seed_boot_indexes when ClickHouse credentials are present.
    - Kinesis (Phase 1.5): wire the Kinesis consumer to call this endpoint or invoke
      insert_seed_to_clickhouse directly; set vectorization.analytics_backfill in
      base.yaml for the target window.
    - Delta driver: DeltaRefreshDriver writes mutable-field patches to ClickHouse on
      each cycle via write_delta_to_clickhouse (ch_delta_analytics.py); this endpoint
      handles full historical population rather than per-row field patches.

    Does NOT write to Qdrant. Does NOT affect the in-memory or BM25 indexes.
    active_only is always False so historical (ended) auctions are included.

    **Athena-side merge** (``run_token="backfill"``) builds a per-run set of
    ``stg_*``/``final_*`` tables in the configured scratch database
    (``vectorization.seed.database.merge_database``, override via
    ``ATHENA_TMP_DB`` — see ``vectorization.seed_merge``), pages the merged
    final table, and drops every table it created once this run's Qdrant and
    ClickHouse upserts have both finished (success or failure). Concurrent
    calls to this endpoint serialize on an internal lock rather than racing
    on that shared scratch database.

    :return: Dict with fetch summary and ClickHouse write summary.
    """
    sub = _require_subsystems()
    vec_cfg = sub.config.vectorization
    seed_cfg = vec_cfg.seed if vec_cfg is not None else None
    db_cfg = seed_cfg.database if seed_cfg is not None else None
    if db_cfg is None:
        raise HTTPException(
            status_code=503, detail="vectorization.seed.database not configured"
        )

    _ch_exec = (
        sub.analytics_router.clickhouse_executor
        if sub.analytics_router is not None
        and getattr(sub.analytics_router, "clickhouse_executor", None) is not None
        and sub.analytics_router.clickhouse_executor.credentials_available
        else None
    )
    if _ch_exec is None:
        raise HTTPException(
            status_code=503,
            detail="ClickHouse executor unavailable or credentials absent",
        )

    _abf_cfg = vec_cfg.analytics_backfill if vec_cfg is not None else None
    if _abf_cfg is None:
        raise HTTPException(
            status_code=503,
            detail="vectorization.analytics_backfill not configured",
        )
    _cfg_lookback = _abf_cfg.lookback_days
    _cfg_max = _abf_cfg.max_records
    _cfg_batch = _abf_cfg.batch_size
    _analytics = (
        sub.config.nl_to_sql.analytics
        if sub.config.nl_to_sql is not None
        else None
    )
    if _analytics is None:
        raise HTTPException(
            status_code=503,
            detail="nl_to_sql.analytics not configured",
        )
    _ch_target = _analytics.seed_target_table
    _ch_snapshot = _analytics.seed_snapshot_table
    _ch_schema_pending = True

    effective_lookback = lookback_days if lookback_days != 180 else _cfg_lookback
    # Analytics window may exceed Qdrant seed max_records_cap (1M); use hard cap.
    _hard_cap = _SeedDatabaseConfig._MAX_RECORDS_HARD_CAP
    effective_max = min(
        max_records if max_records != 25_000_000 else _cfg_max, _hard_cap
    )
    # SeedDatabaseConfig requires max_records_cap >= max_records.
    effective_cap = max(db_cfg.max_records_cap, effective_max)

    _tbl0 = db_cfg.tables[0] if db_cfg.tables else None
    if _tbl0 is None:
        raise HTTPException(
            status_code=503, detail="vectorization.seed.database.tables is empty"
        )

    _abf_tbl = _SeedTableConfig(
        table_name=_tbl0.table_name,
        strategy="datewise",
        lookback_days=effective_lookback,
        max_records=effective_max,
        active_only=False,
    )
    _abf_db = _SeedDatabaseConfig(
        name=db_cfg.name,
        realtime_name=db_cfg.realtime_name,
        tables=[_abf_tbl],
        max_records=effective_max,
        max_records_cap=effective_cap,
        timeout_seconds=db_cfg.timeout_seconds,
        find_payload_aliases=db_cfg.find_payload_aliases,
        find_bool_aliases=db_cfg.find_bool_aliases,
        majestic=db_cfg.majestic,
        merge_database=db_cfg.merge_database,
        merge_database_location=db_cfg.merge_database_location,
        bid_source_database=db_cfg.bid_source_database,
        bid_source_table=db_cfg.bid_source_table,
        bid_winning_table=db_cfg.bid_winning_table,
        merge_page_size=db_cfg.merge_page_size,
    )

    athena_cfg = (
        sub.config.nl_to_sql.athena if sub.config.nl_to_sql is not None else None
    )
    if athena_cfg is None:
        raise HTTPException(
            status_code=503,
            detail="Athena config absent — nl_to_sql.athena not configured",
        )

    athena_client = _AthenaClient(athena_cfg)
    if not athena_client.credentials_available:
        raise HTTPException(status_code=503, detail="Athena credentials unavailable")

    started_at_iso = datetime.now(timezone.utc).isoformat()
    _history_extra = {
        "lookback_days": effective_lookback,
        "max_records": effective_max,
    }
    _fetch_t0 = time.monotonic()
    docs_total = 0
    tables_seen: set = set()
    all_missing: List[str] = []
    ch_totals = dict(rows_attempted=0, rows_written=0, batches=0, errors=0, elapsed_ms=0.0, first_error=None)
    run_token = "backfill"

    def _summary(elapsed_ms: float) -> _DbSeedSummary:
        return _DbSeedSummary(
            source="daily_snapshot",
            database=_abf_db.name,
            documents=docs_total,
            tables_queried=len(tables_seen),
            tables_skipped=len(_abf_db.tables) - len(tables_seen),
            elapsed_ms=round(elapsed_ms, 1),
            missing_columns=all_missing,
        )

    def _partial() -> Dict[str, Any]:
        return {
            "fetch": _summary((time.monotonic() - _fetch_t0) * 1000).__dict__,
            "clickhouse": {
                "rows_written": ch_totals["rows_written"],
                "errors": ch_totals["errors"],
                "elapsed_ms": round(ch_totals["elapsed_ms"], 1),
            },
        }

    if seed_cfg is None:
        raise HTTPException(
            status_code=503, detail="vectorization.seed not configured"
        )
    timing = StageTimingSession(seed_cfg.stage_timing)
    page_gate = bool(timing.config.log_page_stages)
    try:
        _set_ingest_progress("analytics_backfill_fetch", 0, 0)
        async with _BACKFILL_BUILD_LOCK, aclosing(
            fetch_seed_pages_from_db(
                athena_client, _abf_db, "daily_snapshot",
                run_token=run_token, timing=timing,
            )
        ) as pages:
            async for page_docs, missing_cols, table_name in pages:
                tables_seen.add(table_name)
                for c in missing_cols:
                    if c not in all_missing:
                        all_missing.append(c)
                docs_total += len(page_docs)
                _set_ingest_progress("analytics_backfill_fetch", docs_total, docs_total)
                if not page_docs:
                    continue
                _set_ingest_progress("clickhouse_seed", ch_totals["rows_written"], docs_total)
                _ensure = True
                if _abf_cfg.ensure_schema_once:
                    _ensure = _ch_schema_pending
                    _ch_schema_pending = False
                with timing.stage(
                    "clickhouse_seed",
                    gate=page_gate,
                    table=table_name,
                    docs=len(page_docs),
                ):
                    _ch_write = await insert_seed_to_clickhouse(
                        page_docs,
                        _ch_exec,
                        ensure_schema=_ensure,
                        batch_size=_cfg_batch,
                        insert_timeout_seconds=_abf_cfg.insert_timeout_seconds,
                        schema_timeout_seconds=_abf_cfg.schema_timeout_seconds,
                        target_table=_ch_target,
                        snapshot_table=_ch_snapshot,
                    )
                ch_totals["rows_attempted"] += _ch_write.rows_attempted
                ch_totals["rows_written"] += _ch_write.rows_written
                ch_totals["batches"] += _ch_write.batches
                ch_totals["errors"] += _ch_write.errors
                ch_totals["elapsed_ms"] += _ch_write.elapsed_ms
                if _ch_write.first_error and ch_totals["first_error"] is None:
                    ch_totals["first_error"] = _ch_write.first_error

        elapsed_ms = (time.monotonic() - _fetch_t0) * 1000
        summary = _summary(elapsed_ms)
        timing.log_job_complete(
            endpoint="/data-build/analytics-backfill",
            lookback_days=effective_lookback,
            documents=docs_total,
        )
        logger.info(
            f"analytics_backfill_complete lookback_days={effective_lookback} "
            f"docs={docs_total} rows_written={ch_totals['rows_written']} "
            f"errors={ch_totals['errors']}"
        )
        app_state._ingest_progress = None
        _abf_result: Dict[str, Any] = {
            "started_at": started_at_iso,
            "lookback_days": effective_lookback,
            "max_records": effective_max,
            "fetch": summary,
            "clickhouse": {
                "rows_attempted": ch_totals["rows_attempted"],
                "rows_written": ch_totals["rows_written"],
                "batches": ch_totals["batches"],
                "errors": ch_totals["errors"],
                "elapsed_ms": round(ch_totals["elapsed_ms"], 1),
                "first_error": ch_totals["first_error"],
            },
        }
        if timing.config.include_in_response:
            _abf_result["stage_timing"] = timing.summary()
        return _abf_result
    except DataIngestInterruptedError as _die:
        return _data_ingest_interrupted_response(
            endpoint="/data-build/analytics-backfill",
            exc=_die,
            started_at=started_at_iso,
            partial=_partial(),
            history_extra=_history_extra,
        )
    except asyncio.CancelledError as _ce:
        return _data_ingest_interrupted_response(
            endpoint="/data-build/analytics-backfill",
            exc=DataIngestInterruptedError(
                f"Analytics backfill cancelled after partial progress (docs={docs_total}).",
                stage="clickhouse_seed" if ch_totals["rows_written"] else "fetch",
                records_completed=ch_totals["rows_written"] or docs_total,
                records_attempted=docs_total,
                reason="cancelled",
                detail=type(_ce).__name__,
            ),
            started_at=started_at_iso,
            partial=_partial(),
            history_extra=_history_extra,
        )
    except RuntimeError as _re:
        # AthenaClient raises RuntimeError on ExpiredToken + failed STS refresh
        # (base Okta session past -d window). Surface as 503 interrupt, not 500.
        _detail = str(_re)
        if "credential" not in _detail.lower() and "credentials_unavailable" not in _detail:
            raise
        return _data_ingest_interrupted_response(
            endpoint="/data-build/analytics-backfill",
            exc=DataIngestInterruptedError(
                (
                    f"Analytics backfill stopped after partial progress "
                    f"(docs={docs_total}, ch_rows={ch_totals['rows_written']}): {_detail}. "
                    f"Re-authenticate (aws-okta-processor) and re-run; temp Athena merge "
                    f"tables for run_token={run_token!r} may need manual DROP if cleanup failed."
                ),
                stage="fetch",
                records_completed=ch_totals["rows_written"] or docs_total,
                records_attempted=docs_total,
                reason="error",
                detail=_detail,
            ),
            started_at=started_at_iso,
            partial=_partial(),
            history_extra=_history_extra,
        )


@app.post("/data-build/full", tags=["Data Build"])
async def data_build_full(
    seed_lookback_days: int = Form(
        default=14,
        ge=1,
        le=730,
        description=(
            "seed_lookback_days [Integer, default: 14, range: 1–730] — "
            "lookback_days forwarded to POST /data-build/seed (Qdrant index window). "
            "Default matches vectorization.seed.database.tables[0].lookback_days in base.yaml."
        ),
    ),
    analytics_lookback_days: int = Form(
        default=180,
        ge=1,
        le=730,
        description=(
            "analytics_lookback_days [Integer, default: 180, range: 1–730] — "
            "lookback_days forwarded to POST /data-build/analytics-backfill (ClickHouse window). "
            "180 = two quarters; fits 22 GB ClickHouse budget. Use 730 for a two-year back-fill. "
            "Default matches vectorization.analytics_backfill.lookback_days."
        ),
    ),
    seed_strategy: str = Form(
        default="datewise",
        description="seed_strategy [String, default: datewise] — forwarded to POST /data-build/seed.",
    ),
    seed_mode: str = Form(
        default="auto",
        description="seed_mode [String, default: auto] — forwarded to POST /data-build/seed.",
    ),
) -> Dict[str, Any]:
    """Run Qdrant seed then ClickHouse analytics back-fill sequentially.

    Order is fixed: ``/data-build/seed`` first; ``/data-build/analytics-backfill``
    only after seed succeeds. Seed failure or interrupt skips ClickHouse.
    When ``clickhouse.enabled`` is false, only the Qdrant seed runs and
    ``clickhouse`` is reported as skipped.

    Prefer calling the single-purpose endpoints from CI (seed, then
    analytics-backfill). This combined route remains for manual/ops and the
    in-process schedule loop.

    :return: Dict with keys ``qdrant`` (seed result) and ``clickhouse`` (backfill or skip).
    """
    started_at_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _normalize_side(side: str, result: Any) -> Any:
        if isinstance(result, JSONResponse):
            try:
                return json.loads(result.body.decode("utf-8"))
            except (TypeError, ValueError, UnicodeDecodeError):
                return {"error": "interrupt_response_unreadable", "side": side}
        if isinstance(result, DataIngestInterruptedError):
            return _data_ingest_interrupted_envelope(
                endpoint=f"/data-build/full:{side}",
                exc=result,
                started_at=started_at_iso,
            )
        if isinstance(result, asyncio.CancelledError):
            return _data_ingest_interrupted_envelope(
                endpoint=f"/data-build/full:{side}",
                exc=result,
                started_at=started_at_iso,
            )
        if isinstance(result, Exception):
            return {"error": f"{type(result).__name__}: {result}"}
        return result

    def _side_failed(out: Any) -> bool:
        if not isinstance(out, dict):
            return True
        if out.get("status") == "interrupted":
            return True
        if "error" in out:
            return True
        return False

    try:
        qdrant_result = await data_build_seed(
            source="daily_snapshot",
            strategy=seed_strategy,
            lookback_days=seed_lookback_days,
            max_records=1_000_000,
            mode=seed_mode,
        )
    except asyncio.CancelledError as _ce:
        return _data_ingest_interrupted_response(
            endpoint="/data-build/full",
            exc=_ce,
            started_at=started_at_iso,
        )
    except Exception as _seed_exc:  # noqa: BLE001 — surface as qdrant error envelope
        qdrant_result = _seed_exc

    qdrant_out = _normalize_side("qdrant", qdrant_result)
    if _side_failed(qdrant_out):
        body = {
            "qdrant": qdrant_out,
            "clickhouse": {
                "skipped": True,
                "reason": "qdrant_seed_failed",
            },
        }
        if isinstance(qdrant_out, dict) and qdrant_out.get("status") == "interrupted":
            envelope = {
                "status": "interrupted",
                "endpoint": "/data-build/full",
                "started_at": started_at_iso,
                "stage": qdrant_out.get("stage", "unknown"),
                "records_completed": int(qdrant_out.get("records_completed") or 0),
                "records_attempted": int(qdrant_out.get("records_attempted") or 0),
                "reason": qdrant_out.get("reason", "cancelled"),
                "detail": qdrant_out.get("detail", ""),
                "message": qdrant_out.get("message")
                or (
                    f"Full data-build interrupted at stage={qdrant_out.get('stage')} "
                    f"after {qdrant_out.get('records_completed')} of "
                    f"{qdrant_out.get('records_attempted')} records."
                ),
                "hint": qdrant_out.get("hint")
                or (
                    "Ingest stopped before completion. Often ALB idle timeout or client disconnect. "
                    "Re-run /data-build/seed (or /data-build/full)."
                ),
                "partial": body,
            }
            logger.error(
                f"data_ingest_interrupted endpoint=/data-build/full stage={envelope['stage']} "
                f"records_completed={envelope['records_completed']} "
                f"records_attempted={envelope['records_attempted']} reason={envelope['reason']}"
            )
            return JSONResponse(status_code=503, content=envelope)
        return JSONResponse(status_code=503, content=body)

    sub = _require_subsystems()
    _ch_lever = getattr(sub.config, "clickhouse", None)
    ch_enabled = bool(_ch_lever is not None and _ch_lever.enabled)
    if not ch_enabled:
        return {
            "qdrant": qdrant_out,
            "clickhouse": {"skipped": True, "reason": "clickhouse_disabled"},
        }

    try:
        ch_result = await data_build_analytics_backfill(
            lookback_days=analytics_lookback_days,
            max_records=25_000_000,
        )
    except asyncio.CancelledError as _ce:
        return _data_ingest_interrupted_response(
            endpoint="/data-build/full",
            exc=_ce,
            started_at=started_at_iso,
        )
    except Exception as _ch_exc:  # noqa: BLE001
        ch_result = _ch_exc

    ch_out = _normalize_side("clickhouse", ch_result)
    body = {"qdrant": qdrant_out, "clickhouse": ch_out}
    if isinstance(ch_out, dict) and ch_out.get("status") == "interrupted":
        envelope = {
            "status": "interrupted",
            "endpoint": "/data-build/full",
            "started_at": started_at_iso,
            "stage": ch_out.get("stage", "unknown"),
            "records_completed": int(ch_out.get("records_completed") or 0),
            "records_attempted": int(ch_out.get("records_attempted") or 0),
            "reason": ch_out.get("reason", "cancelled"),
            "detail": ch_out.get("detail", ""),
            "message": ch_out.get("message")
            or (
                f"Full data-build interrupted at stage={ch_out.get('stage')} "
                f"after {ch_out.get('records_completed')} of "
                f"{ch_out.get('records_attempted')} records."
            ),
            "hint": ch_out.get("hint")
            or (
                "Ingest stopped before completion. Often ALB idle timeout or client disconnect. "
                "Re-run /data-build/analytics-backfill (or /data-build/full)."
            ),
            "partial": body,
        }
        logger.error(
            f"data_ingest_interrupted endpoint=/data-build/full stage={envelope['stage']} "
            f"records_completed={envelope['records_completed']} "
            f"records_attempted={envelope['records_attempted']} reason={envelope['reason']}"
        )
        return JSONResponse(status_code=503, content=envelope)
    if _side_failed(ch_out):
        return JSONResponse(status_code=503, content=body)
    return body


@app.get("/data-build/status", tags=["Status"])
async def data_build_status() -> Dict[str, Any]:
    """Return data-layer health: Qdrant, ClickHouse, index counts, vectorization pipeline, and column availability."""
    sub = _require_subsystems()

    # ── Qdrant ───────────────────────────────────────────────────────────────
    vector_count = 0
    structured_count = 0
    vector_backend = "unknown"
    structured_backend = "unknown"
    if isinstance(sub.vector_index, InMemoryVectorIndex):
        vector_count = len(sub.vector_index._items)  # noqa: SLF001
        vector_backend = "memory"
    else:
        vector_backend = "qdrant"
    if isinstance(sub.structured_index, InMemoryStructuredIndex):
        structured_count = len(sub.structured_index._items)  # noqa: SLF001
        structured_backend = "memory"
    else:
        structured_backend = "qdrant"

    qdrant_status: Dict[str, Any] = {"configured": False}
    if sub.qdrant_factory is not None and sub.qdrant_factory.available:
        _qcfg = sub.qdrant_factory.config
        _qhost = _qcfg.host
        _qport = int(_qcfg.port)
        _qreachable = await _ping_qdrant(_qhost, _qport, secure=bool(_qcfg.https))
        qdrant_status = {
            "configured": True,
            "reachable": _qreachable,
            "host": _qhost,
            "port": _qport,
            "collection": sub.qdrant_factory.collection_name,
            "document_count": None,
        }
        if _qreachable and sub.qdrant_factory.client is not None:
            try:
                _count_result = await sub.qdrant_factory.client.count(
                    collection_name=sub.qdrant_factory.collection_name,
                    exact=False,
                )
                qdrant_count = _count_result.count
                qdrant_status["document_count"] = qdrant_count
                if vector_backend == "qdrant":
                    vector_count = qdrant_count
                if structured_backend == "qdrant":
                    structured_count = qdrant_count
            except Exception as _ce:  # noqa: BLE001
                logger.warning(f"data_build_status_qdrant_count_failed error={_ce}")
                qdrant_status["count_error"] = str(_ce)
    elif vector_backend == "memory" or structured_backend == "memory":
        qdrant_status = {"configured": False, "note": "in-memory indexes active"}

    # ── ClickHouse ───────────────────────────────────────────────────────────
    ch_analytics_cfg = (
        sub.config.nl_to_sql.analytics if sub.config.nl_to_sql is not None else None
    )
    ch_status: Dict[str, Any] = {"configured": False}
    if ch_analytics_cfg is not None and ch_analytics_cfg.enabled:
        _ch_cfg = ch_analytics_cfg.clickhouse
        _ch_host = _ch_cfg.host
        _ch_port = int(_ch_cfg.port)
        _ch_reachable = await _ping_clickhouse(
            _ch_host, _ch_port, secure=bool(_ch_cfg.secure)
        )
        ch_status = {
            "configured": True,
            "reachable": _ch_reachable,
            "host": _ch_host,
            "port": _ch_port,
            "last_seed": app_state._last_ch_seed_result,
        }

    # ── Vectorization pipeline ───────────────────────────────────────────────
    pipeline_info: Optional[Dict[str, Any]] = None
    if sub.doc_vectorization_pipeline is not None:
        p = sub.doc_vectorization_pipeline
        syn_info: Optional[Dict[str, Any]] = None
        if p.has_expander:
            exp = p._expander  # noqa: SLF001
            syn_info = {"enabled": exp.enabled, "synonym_terms_loaded": exp.map_size}
        pipeline_info = {
            "bm25_vocab_size": getattr(p._doc_encoder, "vocab_size", None),  # noqa: SLF001
            "synonym_expander": syn_info,
        }

    # ── Column / filter availability ─────────────────────────────────────────
    _missing_cols = sorted(app_state._missing_data_columns)
    _unavailable_ents = sorted(_get_unavailable_filter_entities(set(_missing_cols)))

    # ── Seed / build history ─────────────────────────────────────────────────
    seed_enabled = False
    seed_schedule: Optional[Dict[str, Any]] = None
    seed_lookback_days: Optional[int] = None
    seed_strategy: Optional[str] = None
    analytics_lookback_days: Optional[int] = None
    if (
        sub.config.vectorization is not None
        and sub.config.vectorization.seed is not None
    ):
        seed_cfg = sub.config.vectorization.seed
        seed_enabled = seed_cfg.enabled
        if seed_cfg.schedule is not None:
            _sched = seed_cfg.schedule
            _ch_on = bool(
                getattr(sub.config, "clickhouse", None) is not None
                and sub.config.clickhouse.enabled
            )
            # Per-step budget; sequence doubles when CH backfill runs after seed.
            _step_s = int(_sched.max_runtime_seconds)
            seed_schedule = {
                "enabled": _sched.enabled,
                "in_process": _sched.in_process,
                "run_on_deploy": _sched.run_on_deploy,
                "interval_hours": _sched.interval_hours,
                "run_at_hour_utc": _sched.run_at_hour_utc,
                "max_runtime_seconds": _step_s,
                "sequence_max_runtime_seconds": _step_s * (2 if _ch_on else 1),
                "seed_mode": _sched.seed_mode,
            }
        if seed_cfg.database.tables:
            _tbl0 = seed_cfg.database.tables[0]
            seed_lookback_days = _tbl0.lookback_days
            seed_strategy = _tbl0.strategy
        _abf = sub.config.vectorization.analytics_backfill
        if _abf is not None:
            analytics_lookback_days = _abf.lookback_days
    last_build = app_state._build_history[-1] if app_state._build_history else None

    return {
        "qdrant": qdrant_status,
        "clickhouse": ch_status,
        "vector_index": {"backend": vector_backend, "document_count": vector_count},
        "structured_index": {
            "backend": structured_backend,
            "document_count": structured_count,
        },
        "vectorization_pipeline": pipeline_info
        if pipeline_info is not None
        else (sub.doc_vectorization_pipeline is not None),
        "indexing": {
            "missing_source_columns": _missing_cols,
            "unavailable_filter_entities": _unavailable_ents,
            "snapshot_version": sub.snapshot_registry.version,
            "encoder_degraded": app_state.encoder_degraded,
            "note": (
                "column_availability_unknown — no seed build run this session; call POST /data-build/seed to detect missing columns"
                if not app_state._build_history
                else None
            ),
        },
        "seed": {
            "enabled": seed_enabled,
            "scheduled_task_active": app_state._seed_task is not None
            and not app_state._seed_task.done(),
            "refresh_driver_active": sub.vector_refresh_driver is not None,
            "last_build": last_build,
            "total_builds_this_session": len(app_state._build_history),
            "schedule": seed_schedule,
            "lookback_days": seed_lookback_days,
            "strategy": seed_strategy,
            "analytics_lookback_days": analytics_lookback_days,
        },
    }
