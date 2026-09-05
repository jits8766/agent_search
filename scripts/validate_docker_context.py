#!/usr/bin/env python3
"""Validate that every COPY/ADD source in all repo Dockerfiles exists locally.

Discovery (no hardcoded file list):
  1. Scan all docker-compose*.yaml|yml for services with a `build:` block.
     Extract build context and dockerfile path from each.
  2. Walk the repo tree for any Dockerfile* not already covered by step 1.
     Assign them repo-root as default build context (most common convention).
  3. For each discovered Dockerfile, parse every COPY/ADD instruction and
     verify the source path exists relative to its build context.

Skips:
  - COPY --from=<stage>   (multi-stage refs, not local paths)
  - Sources containing $VAR  (shell variables, unresolvable statically)
  - Paths inside _CI_ONLY_PREFIXES (e.g. pretrained/) unless --strict passed

Usage:
    python scripts/validate_docker_context.py
    python scripts/validate_docker_context.py --strict
    python scripts/validate_docker_context.py --repo-root /path/to/repo
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

try:
    import yaml as _yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

# Sources that are absent in a normal dev checkout (CI-synced, gitignored).
# Skipped by default; included with --strict.
_CI_ONLY_PREFIXES: tuple[str, ...] = ("pretrained",)

# Directories to skip when walking for Dockerfile* files.
_SKIP_DIRS: frozenset[str] = frozenset({".git", ".venv", "venv", "node_modules", "__pycache__", ".codegraph"})

_COPY_RE = re.compile(
    r'^\s*(?:COPY|ADD)\s+(?P<flags>(?:--[A-Za-z0-9_=:\-]+\s+)*)(?P<srcs>.+)',
    re.IGNORECASE,
)
_FLAG_FROM_RE = re.compile(r'--from\s*=', re.IGNORECASE)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _find_compose_files(repo_root: Path) -> list[Path]:
    results = []
    for pat in ("docker-compose*.yaml", "docker-compose*.yml"):
        results.extend(repo_root.glob(pat))
    return sorted(set(results))


def _parse_compose(compose_file: Path, repo_root: Path) -> dict[Path, Path]:
    """Return {dockerfile_abs: context_abs} from a compose file's build sections."""
    if not _YAML_AVAILABLE:
        print(
            f"  [warn] PyYAML not installed — cannot parse {compose_file.name}. "
            "Install pyyaml or run: pip install pyyaml",
            file=sys.stderr,
        )
        return {}

    try:
        with compose_file.open() as fh:
            data = _yaml.safe_load(fh)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] Failed to parse {compose_file}: {exc}", file=sys.stderr)
        return {}

    services = (data or {}).get("services") or {}
    mapping: dict[Path, Path] = {}

    for _svc, cfg in services.items():
        build = (cfg or {}).get("build")
        if not build:
            continue

        compose_dir = compose_file.parent

        if isinstance(build, str):
            context = (compose_dir / build).resolve()
            dockerfile = context / "Dockerfile"
        elif isinstance(build, dict):
            raw_ctx = build.get("context", ".")
            context = (compose_dir / raw_ctx).resolve()
            raw_df = build.get("dockerfile", "Dockerfile")
            # Dockerfile path in compose is relative to the project directory
            # (compose file location), not the context — confirmed by Docker docs.
            dockerfile = (compose_dir / raw_ctx / raw_df).resolve()
        else:
            continue

        if dockerfile.exists():
            mapping[dockerfile] = context

    return mapping


def _walk_dockerfiles(repo_root: Path, already_covered: frozenset[Path]) -> dict[Path, Path]:
    """Find Dockerfile* files not covered by compose, default context = repo_root."""
    mapping: dict[Path, Path] = {}
    for path in repo_root.rglob("Dockerfile*"):
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if not path.is_file():
            continue
        if path in already_covered:
            continue
        mapping[path] = repo_root  # default: repo root
    return mapping


def discover_dockerfiles(repo_root: Path) -> dict[Path, Path]:
    """Return {dockerfile_abs: build_context_abs} for every Dockerfile in the repo."""
    mapping: dict[Path, Path] = {}

    for cf in _find_compose_files(repo_root):
        mapping.update(_parse_compose(cf, repo_root))

    mapping.update(_walk_dockerfiles(repo_root, frozenset(mapping)))
    return mapping


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _parse_copy_sources(line: str) -> Optional[list[str]]:
    m = _COPY_RE.match(line)
    if not m:
        return None
    if _FLAG_FROM_RE.search(m.group("flags") or ""):
        return None
    tokens = m.group("srcs").split()
    return tokens[:-1] if len(tokens) >= 2 else None  # all but dest


def _is_ci_only(src: str) -> bool:
    top = src.lstrip("/").split("/")[0]
    return top in _CI_ONLY_PREFIXES


def check_dockerfile(dockerfile: Path, context: Path, strict: bool) -> list[str]:
    errors: list[str] = []
    try:
        lines = dockerfile.read_text().splitlines()
    except OSError as exc:
        return [f"  Cannot read {dockerfile}: {exc}"]

    for lineno, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        srcs = _parse_copy_sources(line)
        if srcs is None:
            continue

        for src in srcs:
            if "$" in src:  # shell variable — cannot resolve statically
                continue

            src_clean = src.rstrip("/")

            if _is_ci_only(src_clean):
                if strict and not (context / src_clean).exists():
                    errors.append(
                        f"  {dockerfile}:{lineno}  MISSING (CI-only)  {src_clean}"
                    )
                continue

            if "*" in src_clean or "?" in src_clean:
                parent = Path(src_clean).parent
                if not (context / parent).exists():
                    errors.append(
                        f"  {dockerfile}:{lineno}  MISSING parent dir  {parent}/"
                        f"  (context: {context})"
                    )
                continue

            if not (context / src_clean).exists():
                errors.append(
                    f"  {dockerfile}:{lineno}  MISSING  {src_clean}"
                    f"  (context: {context})"
                )

    return errors


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--strict", action="store_true",
                        help="Also check CI-only paths (e.g. pretrained/).")
    parser.add_argument("--repo-root", default=None,
                        help="Repo root path. Defaults to parent of this script.")
    args = parser.parse_args(argv)

    repo_root = (
        Path(args.repo_root).resolve()
        if args.repo_root
        else Path(__file__).resolve().parent.parent
    )

    dockerfiles = discover_dockerfiles(repo_root)
    if not dockerfiles:
        print("[docker-preflight] No Dockerfiles discovered — nothing to check.")
        return 0

    all_errors: list[str] = []
    for dockerfile, context in sorted(dockerfiles.items()):
        rel = dockerfile.relative_to(repo_root) if dockerfile.is_relative_to(repo_root) else dockerfile
        print(f"[docker-preflight] Checking {rel}  (context: {context.relative_to(repo_root) if context.is_relative_to(repo_root) else context})")
        errs = check_dockerfile(dockerfile, context, strict=args.strict)
        all_errors.extend(errs)

    if all_errors:
        print("\n[docker-preflight] FAILED — missing build context sources:", file=sys.stderr)
        for e in all_errors:
            print(e, file=sys.stderr)
        if not args.strict:
            print(
                "\n  Note: CI-only paths (e.g. pretrained/) excluded unless --strict passed.",
                file=sys.stderr,
            )
        return 1

    mode = "strict" if args.strict else "default"
    print(f"[docker-preflight] OK — all COPY sources present ({mode} mode).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
