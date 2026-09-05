#!/usr/bin/env python3
"""Read ``clickhouse.enabled`` from base.yaml (stdlib only — no PyYAML required).

Usage:
  python3 scripts/read_clickhouse_enabled.py              # prints enabled=true|false
  python3 scripts/read_clickhouse_enabled.py --value-only  # prints true|false
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_DEFAULT_YAML = (
    Path(__file__).resolve().parents[1]
    / 'packages'
    / 'semantic-search'
    / 'semantic_search'
    / 'config'
    / 'base.yaml'
)

# Top-level block: clickhouse:\n  enabled: true|false
_LEVER_RE = re.compile(
    r'(?m)^clickhouse:\s*\n(?:[ \t]+[^\n]*\n)*?[ \t]+enabled:\s*(true|false)\s*(?:#.*)?$',
)


def read_clickhouse_enabled(yaml_path: Path) -> bool:
    text = yaml_path.read_text(encoding='utf-8')
    match = _LEVER_RE.search(text)
    if match is None:
        raise SystemExit(f'clickhouse.enabled not found in {yaml_path}')
    return match.group(1) == 'true'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--yaml',
        type=Path,
        default=_DEFAULT_YAML,
        help='Path to base.yaml',
    )
    parser.add_argument(
        '--value-only',
        action='store_true',
        help='Print true|false only (for shell scripts)',
    )
    args = parser.parse_args()
    if not args.yaml.is_file():
        raise SystemExit(f'config file missing: {args.yaml}')
    enabled = read_clickhouse_enabled(args.yaml)
    value = 'true' if enabled else 'false'
    if args.value_only:
        print(value)
    else:
        # GitHub Actions GITHUB_OUTPUT shape
        print(f'enabled={value}')


if __name__ == '__main__':
    main()
    sys.exit(0)
