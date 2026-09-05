"""Security tests for URL validation and SSRF prevention.

Covers:
- _validate_endpoint_url: scheme allowlist, host allowlist
- _sync_post: never reaches urlopen for disallowed URLs
- call_search: propagates validation errors before network I/O
- Semgrep rules: godaddy.python.security.packages.ssrf
                 python.lang.security.audit.dynamic-urllib-use-detected
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from semantic_search.eval.query_eval_runner import (
    _validate_endpoint_url,
    _sync_post,
    call_search,
    validate_loopback_api_base,
)


# ──────────────────────────────────────────────────────────────────────────────
# _validate_endpoint_url — scheme allowlist
# ──────────────────────────────────────────────────────────────────────────────

class TestSchemeAllowlist:
    def test_http_localhost_allowed(self):
        _validate_endpoint_url("http://localhost:8085/search")

    def test_https_localhost_allowed(self):
        _validate_endpoint_url("https://localhost:8085/search")

    def test_http_127_allowed(self):
        _validate_endpoint_url("http://127.0.0.1:8085/search")

    def test_http_internal_allowed(self):
        _validate_endpoint_url("http://search-service.internal:8080/search")

    def test_http_local_allowed(self):
        _validate_endpoint_url("http://myservice.local/search")

    def test_file_scheme_rejected(self):
        with pytest.raises(ValueError, match="Disallowed URL scheme"):
            _validate_endpoint_url("file:///etc/passwd")

    def test_ftp_scheme_rejected(self):
        with pytest.raises(ValueError, match="Disallowed URL scheme"):
            _validate_endpoint_url("ftp://somehost/data")

    def test_javascript_scheme_rejected(self):
        with pytest.raises(ValueError, match="Disallowed URL scheme"):
            _validate_endpoint_url("javascript:alert(1)")

    def test_data_scheme_rejected(self):
        with pytest.raises(ValueError, match="Disallowed URL scheme"):
            _validate_endpoint_url("data:text/plain,hello")

    def test_empty_scheme_rejected(self):
        with pytest.raises(ValueError):
            _validate_endpoint_url("//localhost/search")

    def test_no_scheme_rejected(self):
        with pytest.raises(ValueError):
            _validate_endpoint_url("localhost:8085/search")


# ──────────────────────────────────────────────────────────────────────────────
# _validate_endpoint_url — host allowlist (SSRF)
# ──────────────────────────────────────────────────────────────────────────────

class TestHostAllowlist:
    def test_localhost_allowed(self):
        _validate_endpoint_url("http://localhost/search")

    def test_127_0_0_1_allowed(self):
        _validate_endpoint_url("http://127.0.0.1:9200/search")

    def test_dot_internal_allowed(self):
        _validate_endpoint_url("http://api.internal:8080/search")

    def test_dot_local_allowed(self):
        _validate_endpoint_url("http://svc.local/search")

    def test_external_host_rejected(self):
        with pytest.raises(ValueError, match="not in trusted allowlist"):
            _validate_endpoint_url("http://evil.com/exfil")

    def test_aws_metadata_endpoint_rejected(self):
        # AWS instance metadata service — classic SSRF target
        with pytest.raises(ValueError, match="not in trusted allowlist"):
            _validate_endpoint_url("http://169.254.169.254/latest/meta-data/")

    def test_gcp_metadata_endpoint_rejected(self):
        with pytest.raises(ValueError, match="not in trusted allowlist"):
            _validate_endpoint_url("http://metadata.google.internal/computeMetadata/v1/")

    def test_private_10_range_rejected(self):
        with pytest.raises(ValueError, match="not in trusted allowlist"):
            _validate_endpoint_url("http://10.0.0.1/internal-service")

    def test_private_192_168_range_rejected(self):
        with pytest.raises(ValueError, match="not in trusted allowlist"):
            _validate_endpoint_url("http://192.168.1.1/admin")

    def test_redirect_lookalike_rejected(self):
        # URL with @ to trick naive parsers: http://trusted@evil.com
        with pytest.raises(ValueError, match="not in trusted allowlist"):
            _validate_endpoint_url("http://localhost@evil.com/search")

    def test_numeric_ip_bypass_rejected(self):
        # Decimal-encoded IP: 2130706433 == 127.0.0.1 — urllib resolves hostname, not numeric
        with pytest.raises(ValueError, match="not in trusted allowlist"):
            _validate_endpoint_url("http://2130706433/search")


# ──────────────────────────────────────────────────────────────────────────────
# validate_loopback_api_base — strict harness SSRF guard
# ──────────────────────────────────────────────────────────────────────────────

class TestLoopbackApiBase:
    def test_loopback_ok(self):
        assert validate_loopback_api_base("http://127.0.0.1:8085") == "http://127.0.0.1:8085"
        assert validate_loopback_api_base("http://localhost:8085/") == "http://localhost:8085"

    def test_rejects_metadata(self):
        with pytest.raises(ValueError, match="loopback"):
            validate_loopback_api_base("http://169.254.169.254")

    def test_rejects_internal_host(self):
        with pytest.raises(ValueError, match="loopback"):
            validate_loopback_api_base("http://search.internal:8085")

    def test_rejects_url_credentials(self):
        with pytest.raises(ValueError, match="credentials"):
            validate_loopback_api_base("http://user:pass@127.0.0.1:8085")

    def test_rejects_file_scheme(self):
        with pytest.raises(ValueError, match="scheme"):
            validate_loopback_api_base("file:///etc/passwd")


# ──────────────────────────────────────────────────────────────────────────────
# _sync_post — validation fires BEFORE urlopen
# ──────────────────────────────────────────────────────────────────────────────

class TestSyncPostNeverCallsUrlopenOnBadUrl:
    def test_file_url_never_opens(self):
        with patch("httpx.post") as mock_post:
            with pytest.raises(ValueError, match="Disallowed URL scheme"):
                _sync_post("file:///etc/passwd", b"query=test")
            mock_post.assert_not_called()

    def test_external_host_never_opens(self):
        with patch("httpx.post") as mock_post:
            with pytest.raises(ValueError, match="not in trusted allowlist"):
                _sync_post("http://attacker.com/exfil", b"query=test")
            mock_post.assert_not_called()

    def test_aws_metadata_never_opens(self):
        with patch("httpx.post") as mock_post:
            with pytest.raises(ValueError, match="not in trusted allowlist"):
                _sync_post("http://169.254.169.254/latest/meta-data/", b"")
            mock_post.assert_not_called()

    def test_valid_url_calls_httpx_post(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"results": []}
        with patch("httpx.post", return_value=mock_response) as mock_post:
            result = _sync_post("http://localhost:8085/search", b"query=test")
            mock_post.assert_called_once()
        assert result == {"results": []}


# ──────────────────────────────────────────────────────────────────────────────
# call_search — async wrapper propagates validation errors
# ──────────────────────────────────────────────────────────────────────────────

class TestCallSearchValidation:
    def test_rejects_file_scheme(self):
        with pytest.raises(ValueError, match="Disallowed URL scheme"):
            asyncio.run(call_search("file:///etc/passwd", "test query", 10))

    def test_rejects_external_endpoint(self):
        with pytest.raises(ValueError, match="not in trusted allowlist"):
            asyncio.run(call_search("http://evil.com/steal", "test query", 10))

    def test_rejects_aws_metadata_ssrf(self):
        with pytest.raises(ValueError, match="not in trusted allowlist"):
            asyncio.run(
                call_search("http://169.254.169.254/latest/meta-data/", "test", 1)
            )

    def test_accepts_localhost_endpoint(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"ranked_results": [], "query_intelligence": {}}
        with patch("httpx.post", return_value=mock_response):
            result, latency = asyncio.run(
                call_search("http://localhost:8085/search", "cheap domains", 10)
            )
        assert isinstance(result, dict)
        assert latency >= 0
