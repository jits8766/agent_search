"""ClickHouse HTTP client for the real-time analytics path.

Mirrors the construction contract of `AthenaClient`:
- Construction NEVER raises when ClickHouse is unreachable; the analytics
  pipeline downgrades to `failure_mode='execution'` instead of crashing at
  import / boot.
- Every IO method gates on `available` and raises `ClickHouseUnavailableError`
  with a typed reason instead of an opaque httpx error.

Tests inject a stubbed transport via the `_transport` parameter (`async def
_transport(query: str, timeout: float) -> tuple[List[Dict[str, Any]], List[str]]`).
When `_transport=None` the client uses `httpx.AsyncClient` against the
ClickHouse HTTP interface (default port 8123, JSONEachRow format) — the
production path. The transport hook keeps the production code path *and* the
test path on a single `execute_query` orchestrator (§testing.mdc — no parallel
test-only branches inside production logic).
"""
import asyncio
import json
import threading
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import httpx

from semantic_search.config.analytics_models import ClickHouseClientConfig, ClickHouseMvFreshnessProbeConfig
from semantic_search.core.exceptions import RetrievalError
from semantic_search.core.logging_utils import get_logger
from semantic_search.resilience.health import BackendHealthRegistry

logger = get_logger(__name__)

Transport = Callable[[str, float], Awaitable[Tuple[List[Dict[str, Any]], List[str]]]]


def _probe_lag_seconds_from_rows(rows: List[Dict[str, Any]]) -> Optional[float]:
    """Extract a scalar lag in seconds from MV freshness probe rows."""
    if not rows:
        return None
    row0 = rows[0]
    if not isinstance(row0, dict):
        return None
    if 'lag_seconds' in row0 and row0['lag_seconds'] is not None:
        try:
            return float(row0['lag_seconds'])
        except (TypeError, ValueError):
            return None
    for _k, val in row0.items():
        if val is None:
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return None


class ClickHouseUnavailableError(RetrievalError):
    """Raised when ClickHouse is not reachable (network, auth, or bad config)."""


class ClickHouseQueryError(RetrievalError):
    """Raised when ClickHouse returns a query-level error (bad SQL, timeout, etc.)."""


class ClickHouseClient:
    """Async ClickHouse HTTP client.

    When a `BackendHealthRegistry` is injected, every successful
    or failed ``execute_query`` call records an outcome on the ``clickhouse``
    backend slot. The registry's degradation thresholds (failure rate, rolling
    window) are config-driven; this client only emits the per-call signal.

    :param config: ClickHouseClientConfig - Connection settings
    :param password: Optional[str] - Password from the secret provider chain
        (NOT stored in YAML — passed in by the registry); None when CH is
        configured without auth (dev/local)
    :param transport: Optional[Transport] - Test hook. When provided, replaces
        the httpx-based round-trip; the orchestrator (`execute_query`) is
        unchanged. Production callers MUST pass `transport=None`.
    :param health_registry: Optional[BackendHealthRegistry] - When provided,
        every call records success/failure under the ``clickhouse`` backend.
        None disables the recording (used by tests that exercise the client
        in isolation).
    """

    def __init__(self, config: ClickHouseClientConfig, password: Optional[str] = None, transport: Optional[Transport] = None, health_registry: Optional[BackendHealthRegistry] = None) -> None:
        if not isinstance(config, ClickHouseClientConfig):
            raise RetrievalError("ClickHouseClient requires a typed ClickHouseClientConfig")
        self._config = config
        self._password = password or ""
        self._transport: Optional[Transport] = transport
        self._health_registry = health_registry
        self._mv_probe: ClickHouseMvFreshnessProbeConfig = config.mv_freshness_probe
        self._mv_probe_lock = threading.Lock()
        self._mv_success_streak = 0
        self._available = False
        self._client = None
        if transport is not None:
            self._available = True
            logger.info("clickhouse_client_ready transport=stub")
            return
        self._init_http_client()

    @property
    def available(self) -> bool:
        """True iff the client is wired to a working transport (httpx or stub)."""
        return self._available

    @property
    def database(self) -> str:
        """Target database name."""
        return self._config.database

    def _init_http_client(self) -> None:
        """Initialize the httpx AsyncClient. Soft-fails so the registry can boot.

        The httpx package is part of the existing requirements (used by
        `LLMProvider`) so import is expected to succeed in the production
        env. We still gate so an environment without httpx degrades cleanly
        instead of crashing.

        When ``readiness_probe.enabled`` is True, a synchronous HTTP round-trip
        runs ``readiness_probe.sql`` before ``available`` is set True. Probe
        failure leaves ``available=False`` so callers see CH as down at boot.
        """
        scheme = "https" if self._config.secure else "http"
        base_url = f"{scheme}://{self._config.host}:{self._config.port}"
        timeout = httpx.Timeout(
            connect=float(self._config.connect_timeout_seconds),
            read=float(self._config.read_timeout_seconds),
            write=float(self._config.connect_timeout_seconds),
            pool=float(self._config.read_timeout_seconds),
        )
        try:
            self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)
        except Exception as e:
            logger.error(f"clickhouse_init_failed error_type={type(e).__name__} error={str(e)}")
            return
        probe = self._config.readiness_probe
        if not probe.enabled:
            self._available = True
            logger.info(
                f"clickhouse_client_ready host={self._config.host} port={self._config.port} "
                f"database={self._config.database} secure={self._config.secure} readiness_probe=skipped"
            )
            return
        if self._run_sync_readiness_probe(base_url=base_url):
            self._available = True
            logger.info(
                f"clickhouse_client_ready host={self._config.host} port={self._config.port} "
                f"database={self._config.database} secure={self._config.secure} readiness_probe=ok"
            )
        else:
            self._available = False
            logger.warning(
                f"clickhouse_readiness_probe_failed host={self._config.host} port={self._config.port} "
                f"database={self._config.database} available=false"
            )

    def _run_sync_readiness_probe(self, *, base_url: str) -> bool:
        """Issue readiness SQL over a one-shot sync HTTP client. Returns True on success."""
        probe = self._config.readiness_probe
        sql = str(probe.sql).strip()
        timeout_s = float(probe.timeout_seconds)
        params: Dict[str, str] = {'default_format': 'JSONEachRow'}
        if not sql.lstrip().upper().startswith('CREATE DATABASE'):
            params['database'] = self._config.database
        headers: Dict[str, str] = {}
        if self._config.user:
            headers['X-ClickHouse-User'] = self._config.user
        if self._password:
            headers['X-ClickHouse-Key'] = self._password
        try:
            with httpx.Client(base_url=base_url, timeout=timeout_s) as sync_client:
                response = sync_client.post('/', params=params, content=sql, headers=headers)
            if response.status_code != 200:
                body = response.text[:200] if response.text else ''
                logger.warning(
                    f"clickhouse_readiness_probe_http_error status={response.status_code} body={body!r}"
                )
                return False
            return True
        except Exception as e:
            logger.warning(
                f"clickhouse_readiness_probe_error error_type={type(e).__name__} error={str(e)}"
            )
            return False

    async def aclose(self) -> None:
        """Close the underlying httpx client (idempotent)."""
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception as e:
                logger.warning(f"clickhouse_close_failed error_type={type(e).__name__} error={e}")
            self._client = None

    async def execute_query(self, query: str, timeout_seconds: float, *, _skip_mv_freshness_probe: bool = False) -> Tuple[List[Dict[str, Any]], List[str], float]:
        """Execute `query` and return (rows, columns, latency_ms).

        :param query: str - Validated SQL (caller is responsible for security/AST checks)
        :param timeout_seconds: float - Wall-clock cap on the round-trip
        :param _skip_mv_freshness_probe: bool - Internal: when True skips the post-success MV lag side-query (probe recursion guard)
        :return: Tuple[List[Dict[str, Any]], List[str], float] - (rows, columns, latency_ms)
        :raises ClickHouseUnavailableError: When the client has no transport
        :raises ClickHouseQueryError: On a query-level CH error
        :raises RetrievalError: On timeout or other transport failure
        """
        if not isinstance(query, str) or not query.strip():
            raise RetrievalError("ClickHouseClient.execute_query requires non-empty SQL")
        if float(timeout_seconds) <= 0.0:
            raise RetrievalError("ClickHouseClient.execute_query requires timeout_seconds > 0")
        if not self._available:
            self._record_health(False)
            raise ClickHouseUnavailableError("ClickHouse client unavailable — host unreachable or httpx missing")
        t0 = time.monotonic()
        attempts = max(1, int(self._config.max_retries) + 1)
        last_exc: Optional[BaseException] = None
        for attempt in range(1, attempts + 1):
            try:
                if self._transport is not None:
                    rows, columns = await asyncio.wait_for(self._transport(query, float(timeout_seconds)), timeout=float(timeout_seconds))
                else:
                    rows, columns = await asyncio.wait_for(self._post_query(query), timeout=float(timeout_seconds))
                latency_ms = (time.monotonic() - t0) * 1000.0
                logger.info(f"clickhouse_query_succeeded rows={len(rows)} columns={len(columns)} latency_ms={latency_ms:.1f} attempt={attempt}")
                self._record_health(True)
                if not _skip_mv_freshness_probe:
                    await self._maybe_run_mv_freshness_probe()
                return rows, columns, latency_ms
            except asyncio.TimeoutError as e:
                last_exc = e
                logger.warning(f"clickhouse_query_timeout attempt={attempt} timeout_seconds={timeout_seconds:.1f}")
                if attempt >= attempts:
                    self._record_health(False)
                    raise RetrievalError(f"clickhouse query exceeded timeout={timeout_seconds:.1f}s") from e
            except ClickHouseQueryError:
                self._record_health(False)
                raise
            except Exception as e:
                last_exc = e
                logger.warning(f"clickhouse_query_transport_error attempt={attempt} error_type={type(e).__name__} error={e}")
                if attempt >= attempts:
                    self._record_health(False)
                    raise RetrievalError(f"clickhouse transport error: {e}") from e
        self._record_health(False)
        raise RetrievalError(f"clickhouse query failed after {attempts} attempts: {last_exc}") from last_exc

    async def _maybe_run_mv_freshness_probe(self) -> None:
        """Run configured lag SQL on a cadence; record clickhouse health failure on breach."""
        cfg = self._mv_probe
        if not cfg.enabled:
            return
        interval = int(cfg.interval_successful_queries)
        if interval < 1:
            return
        with self._mv_probe_lock:
            self._mv_success_streak += 1
            streak = self._mv_success_streak
            if streak % interval != 0:
                return
        try:
            rows, _cols, _ms = await self.execute_query(
                str(cfg.lag_sql),
                float(cfg.query_timeout_seconds),
                _skip_mv_freshness_probe=True,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"clickhouse_mv_freshness_probe_error error_type={type(e).__name__} error={str(e)}")
            self._record_health(False)
            return
        lag = _probe_lag_seconds_from_rows(rows)
        if lag is None:
            logger.warning("clickhouse_mv_freshness_probe_parse_failed lag_seconds_unavailable")
            self._record_health(False)
            return
        if float(lag) > float(cfg.max_lag_seconds):
            logger.warning(f"clickhouse_mv_lag_breach lag_seconds={lag:.3f} max_lag_seconds={float(cfg.max_lag_seconds):.3f}")
            self._record_health(False)

    def _record_health(self, success: bool) -> None:
        """Record a single ``clickhouse``-backend outcome on the injected registry.
        Best-effort: registry-side errors must never propagate into the call site
        (the request path stays alive even if health bookkeeping itself fails).
        :param success: bool - True for a successful round-trip, False for any failure
        """
        if self._health_registry is None:
            return
        try:
            self._health_registry.record('clickhouse', success)
        except Exception as e:
            logger.warning(f"clickhouse_health_record_failed success={success} error_type={type(e).__name__} error={e}")

    async def _post_query(self, query: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Production HTTP path. Returns (rows, columns).

        Uses the JSONEachRow response format so column metadata is preserved
        without an extra DESCRIBE round-trip.
        """
        if self._client is None:
            raise ClickHouseUnavailableError("ClickHouse httpx client not initialized")
        params: Dict[str, str] = {'default_format': 'JSONEachRow'}
        _first_kw = query.lstrip().split()[0].upper() if query.lstrip() else ''
        _is_write = _first_kw in ('INSERT', 'CREATE', 'DROP', 'ALTER', 'TRUNCATE')
        if not _is_write and float(self._config.query_max_execution_time_seconds) > 0.0:
            params['max_execution_time'] = str(int(self._config.query_max_execution_time_seconds))
        # CREATE DATABASE DDL must not target a database — the target db may not
        # exist yet, and ClickHouse rejects the request before executing the DDL.
        if not query.lstrip().upper().startswith('CREATE DATABASE'):
            params['database'] = self._config.database
        headers: Dict[str, str] = {}
        if self._config.user:
            headers['X-ClickHouse-User'] = self._config.user
        if self._password:
            headers['X-ClickHouse-Key'] = self._password
        try:
            response = await self._client.post('/', params=params, content=query, headers=headers)
        except Exception as e:
            raise RetrievalError(f"clickhouse http transport failed: {type(e).__name__}: {e}") from e
        if response.status_code != 200:
            body = response.text[:500] if response.text else ''
            raise ClickHouseQueryError(f"clickhouse http status={response.status_code} body={body!r}")
        rows: List[Dict[str, Any]] = []
        columns: List[str] = []
        text = response.text or ''
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as e:
                raise ClickHouseQueryError(f"clickhouse JSONEachRow parse error: {e}") from e
            if not isinstance(obj, dict):
                raise ClickHouseQueryError(f"clickhouse JSONEachRow row not an object: type={type(obj).__name__}")
            if not columns:
                columns = list(obj.keys())
            rows.append(obj)
        return rows, columns


__all__ = [
    'ClickHouseClient',
    'ClickHouseUnavailableError',
    'ClickHouseQueryError',
    'Transport',
]
