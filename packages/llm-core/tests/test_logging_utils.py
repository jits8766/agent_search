"""Tests for `llm_core.logging_utils.mask_path` (PII-safe path masking)."""
import os
from pathlib import Path

import pytest

import llm_core.logging_utils as lu


@pytest.fixture(autouse=True)
def _reset_workspace_cache():
    """Force workspace-root re-resolution per test so env-var overrides are honored."""
    lu._WORKSPACE_ROOT_CACHE = None
    lu._WORKSPACE_ROOT_RESOLVED = False
    yield
    lu._WORKSPACE_ROOT_CACHE = None
    lu._WORKSPACE_ROOT_RESOLVED = False


def test_mask_path_workspace_relative_str() -> None:
    workspace = lu._resolve_workspace_root()
    assert workspace is not None, "workspace root should be discoverable in this repo"
    abs_path = str(workspace / "llm_core" / "pricing.yaml")
    assert lu.mask_path(abs_path) == "llm_core/pricing.yaml"


def test_mask_path_workspace_relative_pathlib() -> None:
    workspace = lu._resolve_workspace_root()
    abs_path = workspace / "agent_heal" / "config" / "base.yaml"
    assert lu.mask_path(abs_path) == "agent_heal/config/base.yaml"


def test_mask_path_workspace_root_itself_returns_dot() -> None:
    workspace = lu._resolve_workspace_root()
    assert lu.mask_path(workspace) == "."


def test_mask_path_under_home_outside_workspace_uses_tilde(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_home = tmp_path / "home" / "alice"
    fake_home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(fake_home))
    target = fake_home / "Documents" / "secret.txt"
    masked = lu.mask_path(str(target))
    assert masked == "~/Documents/secret.txt"


def test_mask_path_already_relative_passes_through() -> None:
    assert lu.mask_path("llm_core/pricing.yaml") == "llm_core/pricing.yaml"


def test_mask_path_none_returns_sentinel() -> None:
    assert lu.mask_path(None) == "<none>"


def test_mask_path_empty_returns_sentinel() -> None:
    assert lu.mask_path("") == "<none>"
    assert lu.mask_path("   ") == "<none>"


def test_mask_path_url_untouched() -> None:
    assert lu.mask_path("https://example.com/x") == "https://example.com/x"


def test_mask_path_s3_uri_untouched() -> None:
    assert lu.mask_path("s3://bucket/key.json") == "s3://bucket/key.json"


def test_mask_path_nonexistent_workspace_path_still_relativized() -> None:
    workspace = lu._resolve_workspace_root()
    target = str(workspace / "missing" / "dir" / "file.json")
    assert lu.mask_path(target) == "missing/dir/file.json"


def test_mask_path_env_var_override_pins_workspace_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    nested = tmp_path / "pkg" / "module.py"
    assert lu.mask_path(str(nested)) == "pkg/module.py"


def test_mask_path_invalid_env_var_falls_back_to_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKSPACE_ROOT", "/nonexistent/does/not/exist")
    workspace = lu._resolve_workspace_root()
    assert workspace is not None
    assert (workspace / ".cursor").exists() or (workspace / ".git").exists()


def test_mask_path_system_path_untouched_when_outside_home() -> None:
    masked = lu.mask_path("/etc/passwd")
    assert masked.startswith("/")
    assert "alice" not in masked.lower()


def test_mask_path_handles_path_with_tilde() -> None:
    workspace = lu._resolve_workspace_root()
    relative = Path(workspace).relative_to(Path(os.path.expanduser("~")))
    masked = lu.mask_path(f"~/{relative}/llm_core/pricing.yaml")
    assert masked == "llm_core/pricing.yaml"
