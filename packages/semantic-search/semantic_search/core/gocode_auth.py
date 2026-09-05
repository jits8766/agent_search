"""GoCode S2S IAM JWT provider.

Mints an AWS IAM JWT via gd_auth.client.AwsIamAuthTokenClient and refreshes it
before expiry. Active only when GOCODE_ENV env var is set (dev / test / prod).

gd_auth is a GoDaddy-internal library — must be installed in the ECS image.
It reads the ECS task role credentials automatically via the AWS metadata endpoint.
"""
from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Optional

_SSO_HOSTS: dict[str, str] = {
    "dev": "sso.dev-godaddy.com",
    "test": "sso.test-godaddy.com",
    "prod": "sso.godaddy.com",
}

GOCODE_URLS: dict[str, str] = {
    "dev": "https://api.gocode.caas.dev-gdcorp.tools",
    "test": "https://api.gocode.caas.test-gdcorp.tools",
    "prod": "https://api.gocode.caas.gdcorp.tools",
}

# Refresh when fewer than 5 minutes remain before expiry.
_REFRESH_BEFORE_S: int = 300


def _jwt_exp(token: str) -> Optional[int]:
    """Decode exp claim from JWT payload without signature verification."""
    try:
        payload_b64 = token.split(".")[1]
        # Re-pad for base64 decode.
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        return int(payload["exp"])
    except Exception:  # noqa: BLE001
        return None


class GoCodeIAMAuth:
    """Mint and auto-refresh IAM JWTs for GoCode S2S Bearer auth.

    Thread/task-safe: get_token() serialises minting behind an asyncio.Lock.
    """

    def __init__(self, env: str) -> None:
        if env not in _SSO_HOSTS:
            raise ValueError(f"GOCODE_ENV must be dev/test/prod, got {env!r}")
        self._sso_host = _SSO_HOSTS[env]
        self._lock: asyncio.Lock = asyncio.Lock()
        self._token: Optional[str] = None
        self._exp: Optional[int] = None

    def _mint_sync(self) -> str:
        from gd_auth.client import AwsIamAuthTokenClient  # noqa: PLC0415
        return AwsIamAuthTokenClient(self._sso_host).token

    async def get_token(self) -> str:
        """Return current token, minting a fresh one if expired or close to expiry."""
        async with self._lock:
            now = int(time.time())
            if self._token and self._exp and self._exp - now > _REFRESH_BEFORE_S:
                return self._token
            token = await asyncio.get_running_loop().run_in_executor(None, self._mint_sync)
            self._token = token
            self._exp = _jwt_exp(token)
            return token

    @property
    def seconds_to_expiry(self) -> Optional[int]:
        if self._exp is None:
            return None
        return max(0, self._exp - int(time.time()))
