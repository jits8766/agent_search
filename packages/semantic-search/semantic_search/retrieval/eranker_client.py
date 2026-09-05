"""Layer-4 ranking via external eRanker service.

The search service fuses retrieval outputs (RRF) then delegates final ordering to
eRanker, which owns shopper personalization and learned ranking. This module
holds the client port; ``noop`` returns fused order unchanged; ``http`` posts a
typed JSON envelope and maps the response back onto ``RankedResults``.
"""
import os
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import httpx

from semantic_search.config.models import ERankerConfig
from semantic_search.contracts import QueryIntent, RankedItem, RankedResults, UserContext
from semantic_search.core.exceptions import RetrievalError, ValidationError
from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)


@runtime_checkable
class ERankerClient(Protocol):
    """Async port for Layer-4 re-ranking."""

    @property
    def name(self) -> str:
        """Stable client label for logs and ``ERankerOutcome.client``."""
        ...

    @property
    def last_http_status(self) -> Optional[int]:
        """HTTP status from the last ``backend=http`` call, else None."""
        ...

    async def rank(self, request_id: str, intent: QueryIntent, results: RankedResults, user_context: Optional[UserContext]) -> RankedResults:
        """Return ``results`` re-ordered by eRanker (or unchanged on noop / failure path)."""
        ...


def _wire_request_body(request_id: str, intent: QueryIntent, results: RankedResults, user_context: Optional[UserContext], send_user_id: bool) -> Dict[str, Any]:
    """Build JSON-serializable request body without raw query text (PII surface reduction)."""
    items: List[Dict[str, Any]] = [
        {'item_id': it.item_id, 'fused_score': float(it.fused_score), 'contributing_sources': list(it.contributing_sources)} for it in results.items
    ]
    body: Dict[str, Any] = {
        'request_id': request_id,
        'intent_record_id': intent.intent_record_id,
        'query_type': intent.query_type,
        'decision_tier': intent.decision_tier,
        'items': items,
    }
    if user_context is not None:
        u: Dict[str, Any] = {'is_authenticated': bool(user_context.is_authenticated)}
        if send_user_id and user_context.user_id:
            u['user_id'] = user_context.user_id
        body['user'] = u
    return body


def _reorder_by_item_ids(results: RankedResults, item_ids: List[str]) -> RankedResults:
    """Reorder ``results.items`` to match ``item_ids``; unknown ids dropped; missing tail-appended in fused order."""
    by_id = {it.item_id: it for it in results.items}
    seen: set[str] = set()
    out: List[RankedItem] = []
    for iid in item_ids:
        if iid in seen:
            continue
        it = by_id.get(iid)
        if it is None:
            continue
        seen.add(iid)
        out.append(it)
    for it in results.items:
        if it.item_id not in seen:
            out.append(it)
    return RankedResults(
        request_id=results.request_id,
        items=out,
        total_candidates=results.total_candidates,
        fusion_latency_ms=results.fusion_latency_ms,
        cache_hit=results.cache_hit,
        multi_intent_envelope=results.multi_intent_envelope,
        failure_mode=results.failure_mode,
        query_intent=results.query_intent,
        guidance_envelope=results.guidance_envelope,
    )


class NoOpERankerClient:
    """Placeholder client — passes through fused ``RankedResults``."""

    def __init__(self, config: ERankerConfig):
        if config is None:
            raise ValidationError("NoOpERankerClient requires a non-null ERankerConfig")
        self._config = config

    @property
    def name(self) -> str:
        return 'noop_eranker'

    @property
    def last_http_status(self) -> Optional[int]:
        return None

    async def rank(self, request_id: str, intent: QueryIntent, results: RankedResults, user_context: Optional[UserContext]) -> RankedResults:
        _ = user_context
        if not request_id or not isinstance(request_id, str):
            raise ValidationError("NoOpERankerClient.rank requires a non-empty request_id")
        if results is None:
            raise ValidationError("NoOpERankerClient.rank requires a non-null RankedResults")
        if not self._config.enabled:
            logger.info(f"eranker_disabled request_id={request_id} items={len(results.items)}")
            return results
        logger.info(f"eranker_noop request_id={request_id} items={len(results.items)}")
        return results


class HttpERankerClient:
    """HTTP client for Layer-4 — POST JSON, parse ordered ``item_ids``."""

    def __init__(self, config: ERankerConfig):
        if config is None or config.http is None:
            raise ValidationError("HttpERankerClient requires ERankerConfig with http block")
        self._config = config
        self._http = config.http
        self._url = f"{self._http.base_url}/{self._http.rank_path}"
        self._last_http_status: Optional[int] = None
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def name(self) -> str:
        return 'http_eranker'

    @property
    def last_http_status(self) -> Optional[int]:
        return self._last_http_status

    async def _client_aclose_safe(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception as e:
                logger.warning(f"eranker_client_close_failed error_type={type(e).__name__} error={str(e)}")
            self._client = None

    async def rank(self, request_id: str, intent: QueryIntent, results: RankedResults, user_context: Optional[UserContext]) -> RankedResults:
        if not request_id or not isinstance(request_id, str):
            raise ValidationError("HttpERankerClient.rank requires a non-empty request_id")
        if results is None:
            raise ValidationError("HttpERankerClient.rank requires a non-null RankedResults")
        if not self._config.enabled:
            logger.info(f"eranker_disabled request_id={request_id} items={len(results.items)}")
            return results
        self._last_http_status = None
        headers: Dict[str, str] = {'content-type': 'application/json'}
        if self._http.api_key_env_var:
            key = os.environ.get(self._http.api_key_env_var, '')
            if key:
                headers['authorization'] = f'Bearer {key}'
        payload = _wire_request_body(request_id, intent, results, user_context, self._http.send_user_id)
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self._http.timeout_seconds))
        try:
            resp = await self._client.post(self._url, json=payload, headers=headers)
        except httpx.RequestError as e:
            self._last_http_status = None
            raise RetrievalError(f"eranker_http_transport_failed request_id={request_id} error={str(e)}") from e
        self._last_http_status = int(resp.status_code)
        if resp.status_code < 200 or resp.status_code >= 300:
            raise RetrievalError(f"eranker_http_status request_id={request_id} status={resp.status_code}")
        try:
            body = resp.json()
        except ValueError as e:
            raise RetrievalError(f"eranker_http_json_decode request_id={request_id}") from e
        if not isinstance(body, dict):
            raise RetrievalError(f"eranker_http_body_shape request_id={request_id}")
        raw_ids = body.get('item_ids')
        if not isinstance(raw_ids, list):
            raise RetrievalError(f"eranker_http_missing_item_ids request_id={request_id}")
        item_ids = [str(x) for x in raw_ids if isinstance(x, (str, int, float)) and str(x).strip()]
        return _reorder_by_item_ids(results, item_ids)


def build_eranker_client(config: ERankerConfig) -> ERankerClient:
    """Dispatch factory for ``retrieval.eranker``."""
    if config.backend not in ('noop', 'http'):
        raise ValidationError(f"eranker backend {config.backend!r} is not wired")
    if config.backend == 'http':
        return HttpERankerClient(config)
    return NoOpERankerClient(config)


__all__ = ['ERankerClient', 'NoOpERankerClient', 'HttpERankerClient', 'build_eranker_client', '_reorder_by_item_ids']
