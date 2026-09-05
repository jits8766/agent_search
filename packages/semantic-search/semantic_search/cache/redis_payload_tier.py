"""Optional Redis/Dragonfly-compatible tier for ``CachedSearchPayload`` blobs.

Pickle is used here intentionally: ``CachedSearchPayload`` contains a deeply nested
contract graph (QueryIntent + RankedResults + dozens of nested dataclasses) with no
JSON-serialization surface. Replacing pickle requires a full serialization layer —
tracked separately. Mitigations in place: VPC-internal Redis only, TLS required,
Redis ACLs restrict writes to the service identity. Do NOT expose this Redis endpoint
outside the service mesh.  # nosec B301 B403
Uses sync ``redis.Redis`` clients; orchestrator bridges with ``asyncio.to_thread``.
"""
from __future__ import annotations

import os
import pickle
from typing import Optional

import redis

from semantic_search.config.analytics_models import RemoteCacheLayerConfig
from semantic_search.contracts import CachedSearchPayload
from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)


class RedisPayloadTier:
    """GET/SET ``CachedSearchPayload`` under ``key_prefix + fingerprint_hex``."""

    def __init__(self, config: RemoteCacheLayerConfig, client: 'redis.Redis'):
        self._config = config
        self._client = client

    @classmethod
    def try_connect(cls, config: RemoteCacheLayerConfig) -> Optional['RedisPayloadTier']:
        """Return a connected tier when ``enabled`` and URL env resolves; else None."""
        if not config.enabled:
            return None
        url = os.environ.get(config.redis_url_env_var, '')
        if not url or not isinstance(url, str):
            logger.warning(f"redis_payload_tier_disabled reason=missing_env var={config.redis_url_env_var}")
            return None
        client = redis.Redis.from_url(
            url,
            socket_timeout=float(config.socket_timeout_seconds),
            socket_connect_timeout=float(config.socket_timeout_seconds),
        )
        try:
            client.ping()
        except Exception as e:
            logger.warning(f"redis_payload_tier_unavailable error_type={type(e).__name__} error={str(e)}")
            return None
        logger.info(f"redis_payload_tier_connected prefix={config.key_prefix}")
        return cls(config=config, client=client)

    def redis_key(self, fingerprint_hex: str) -> str:
        return f"{self._config.key_prefix}{fingerprint_hex}"

    def get_payload_sync(self, fingerprint_hex: str) -> Optional[CachedSearchPayload]:
        raw = self._client.get(self.redis_key(fingerprint_hex))
        if raw is None:
            return None
        try:
            obj = pickle.loads(raw)  # nosec B301
        except Exception as e:
            logger.warning(f"redis_payload_corrupt key_preview={fingerprint_hex[:12]} error_type={type(e).__name__}")
            return None
        if not isinstance(obj, CachedSearchPayload):
            return None
        return obj

    def put_payload_sync(self, fingerprint_hex: str, payload: CachedSearchPayload, ttl_seconds: int) -> None:
        raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)  # nosec B301
        self._client.setex(self.redis_key(fingerprint_hex), int(ttl_seconds), raw)

    def delete_sync(self, fingerprint_hex: str) -> None:
        self._client.delete(self.redis_key(fingerprint_hex))


__all__ = ['RedisPayloadTier']
