"""Per-session rate limit middleware.

Caps user-input endpoints at
``max_requests_per_window`` per ``window_seconds`` per session. Rejects
above the cap with HTTP 429 + a ``Retry-After`` header.

Bucket key resolution:
1. ``X-Session-Id`` request header (the canonical session id used by the
   search and history paths). Authenticated traffic always
   carries this.
2. ``request.client.host`` — anonymous fallback so untagged clients still
   share a meaningful bucket (per-IP, not per-process).

The store is a sliding-window deque per bucket with an LRU cap on tracked
buckets (memory bound). It is process-local; in a multi-replica deployment
the cap is per-pod (acceptable for the plan's "10/min/sess" target — a
distributed limiter is a follow-up).
"""
from collections import OrderedDict, deque
from threading import RLock
from time import monotonic
from typing import Deque, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from semantic_search.config.models import RateLimitConfig
from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)


class SlidingWindowRateLimiter:
    """Per-bucket sliding-window counter (LRU cap, single RLock for hot path)."""

    def __init__(self, max_requests: int, window_seconds: int, session_state_max: int, burst_requests: int = 0, burst_window_seconds: int = 10):
        if max_requests < 1:
            raise ValueError("max_requests must be >= 1")
        if window_seconds < 1:
            raise ValueError("window_seconds must be >= 1")
        if session_state_max < 1:
            raise ValueError("session_state_max must be >= 1")
        if burst_requests < 0:
            raise ValueError("burst_requests must be >= 0")
        if burst_window_seconds < 1:
            raise ValueError("burst_window_seconds must be >= 1")
        self._max = int(max_requests)
        self._window = float(window_seconds)
        self._cap = int(session_state_max)
        self._burst_max = int(burst_requests)
        self._burst_window = float(burst_window_seconds)
        self._lock = RLock()
        # OrderedDict gives us LRU eviction in O(1) via move_to_end + popitem.
        self._buckets: 'OrderedDict[str, Deque[float]]' = OrderedDict()
        self._burst_buckets: 'OrderedDict[str, Deque[float]]' = OrderedDict()

    def check_and_record(self, bucket_key: str) -> bool:
        """Atomic check + record: True = permitted, False = HTTP 429."""
        if not bucket_key:
            return True  # Empty key -> can't bucket; permit (defence-in-depth, never deny on null)
        now = monotonic()
        cutoff = now - self._window
        with self._lock:
            window = self._buckets.get(bucket_key)
            if window is None:
                # Cold bucket — check LRU cap before allocating.
                if len(self._buckets) >= self._cap:
                    self._buckets.popitem(last=False)
                window = deque()
                self._buckets[bucket_key] = window
            else:
                # Touch for LRU.
                self._buckets.move_to_end(bucket_key)
            # Drop expired timestamps.
            while window and window[0] < cutoff:
                window.popleft()
            if len(window) < self._max:
                window.append(now)
                return True
            # Main window full — try burst bucket when burst is configured.
            if self._burst_max <= 0:
                return False
            burst_window = self._burst_buckets.get(bucket_key)
            if burst_window is None:
                if len(self._burst_buckets) >= self._cap:
                    self._burst_buckets.popitem(last=False)
                burst_window = deque()
                self._burst_buckets[bucket_key] = burst_window
            else:
                self._burst_buckets.move_to_end(bucket_key)
            burst_cutoff = now - self._burst_window
            while burst_window and burst_window[0] < burst_cutoff:
                burst_window.popleft()
            if len(burst_window) < self._burst_max:
                burst_window.append(now)
                return True
            return False

    def retry_after_seconds(self, bucket_key: str) -> int:
        """Return the number of seconds until the bucket frees up by 1 slot.

        Used only on the deny path (the limiter has already returned False);
        callers populate the 429 response's ``Retry-After`` header from this.
        """
        if not bucket_key:
            return self._window_int()
        with self._lock:
            window = self._buckets.get(bucket_key)
            if not window:
                return 1
            oldest = window[0]
            elapsed = monotonic() - oldest
            remaining = max(1, int(self._window - elapsed) + 1)
            return remaining

    def _window_int(self) -> int:
        return max(1, int(self._window))

    def reset(self) -> None:
        """Drop every tracked bucket. Test-isolation hook only.

        Production code MUST NOT call this; it bypasses the per-session
        cap by clearing every active bucket. Tests use it via the autouse
        `_reset_rate_limiter` fixture so the limiter does not leak counts
        across test functions sharing the process-singleton middleware.
        """
        with self._lock:
            self._buckets.clear()
            self._burst_buckets.clear()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Starlette middleware applying the per-session rate limit.

    The limiter is constructed once at app boot (the middleware instance is
    long-lived) and passed in via the ``limiter`` constructor argument so
    tests can inject a fixed-time clock by stubbing ``monotonic``.

    :param app: ASGI app - The wrapped FastAPI application.
    :param config: RateLimitConfig - Rate-limit policy.
    :param limiter: SlidingWindowRateLimiter - The shared limiter instance.
    """

    def __init__(self, app, config: RateLimitConfig, limiter: SlidingWindowRateLimiter):
        super().__init__(app)
        self._config = config
        self._limiter = limiter
        self._path_prefixes = tuple(config.paths)

    def _is_limited_path(self, path: str) -> bool:
        return any(path.startswith(prefix) for prefix in self._path_prefixes)

    def _resolve_bucket_key(self, request: Request) -> str:
        # Header-based session id wins. Production gateways set this on
        # every authenticated request; clients that want isolation can also
        # set it explicitly.
        session_id = request.headers.get('X-Session-Id')
        if session_id and isinstance(session_id, str):
            return "sess:" + session_id.strip()
        # Anonymous fallback — bucket per client IP. This is intentionally
        # coarse: shared NATs collide, but the alternative (deny all
        # untagged traffic) is worse than a 10/min cap that spans a NAT.
        client_host = request.client.host if request.client else 'unknown'
        return "ip:" + client_host

    async def dispatch(self, request: Request, call_next) -> Response:
        async def _next() -> Response:
            # Starlette's BaseHTTPMiddleware raises RuntimeError("No response
            # returned.") when the inner request task is cancelled mid-flight
            # (work abandoned under load — e.g. ClickHouse stalls) and exits
            # without emitting a response. This is the OUTERMOST custom layer,
            # so converting it to a graceful 503 is what makes the
            # client-visible status a retryable 503 instead of a bare 500.
            try:
                return await call_next(request)
            except RuntimeError as exc:
                if "No response returned" not in str(exc):
                    raise
                logger.warning(f"request_cancelled_no_response path={request.url.path}")
                return JSONResponse(
                    status_code=503,
                    content={"detail": "request_cancelled: server busy under load, retry"},
                )

        if not self._config.enabled or not self._path_prefixes:
            return await _next()
        if not self._is_limited_path(request.url.path):
            return await _next()
        bucket_key = self._resolve_bucket_key(request)
        if self._limiter.check_and_record(bucket_key):
            return await _next()
        retry_after = self._limiter.retry_after_seconds(bucket_key)
        # Mask the bucket key in logs (it carries the session id) per
        # `responsible-ai.mdc`. The hash is enough for ops to correlate
        # without leaking the session token.
        bucket_hash = abs(hash(bucket_key)) % 10_000_000
        logger.warning(
            f"rate_limit_exceeded path={request.url.path} bucket_hash={bucket_hash} "
            f"max={self._config.max_requests_per_window} window={self._config.window_seconds} "
            f"retry_after={retry_after}"
        )
        return JSONResponse(
            status_code=429,
            content={
                "detail": (
                    f"rate_limit_exceeded: max {self._config.max_requests_per_window} requests "
                    f"per {self._config.window_seconds}s per session"
                ),
                "retry_after_seconds": retry_after,
            },
            headers={'Retry-After': str(retry_after)},
        )
