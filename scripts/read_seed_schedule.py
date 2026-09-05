#!/usr/bin/env python3
"""Read ``vectorization.seed.schedule`` and related knobs from YAML.

Resolves ``${VAR:-default}`` tokens against ``os.environ``. Stdlib only (no
PyYAML). Prints ``key=value`` lines for GitHub Actions ``GITHUB_OUTPUT``, or
JSON with ``--json``.

Usage:
  python3 scripts/read_seed_schedule.py
  python3 scripts/read_seed_schedule.py --environment Test
  python3 scripts/read_seed_schedule.py --json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Match, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_BASE_YAML = (
    _REPO_ROOT
    / 'packages'
    / 'semantic-search'
    / 'semantic_search'
    / 'config'
    / 'base.yaml'
)
_DEFAULT_KATANA_YAML = _REPO_ROOT / 'configs' / 'katana.yaml'

_ENV_VAR_RE = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}')

# ``schedule:`` block under vectorization.seed (direct children only).
_SCHEDULE_BLOCK_RE = re.compile(
    r'(?ms)^([ \t]+)schedule:\s*\n'
    r'((?:(?:\1[ \t]+|\s*#|\s*$).*\n)*)'
)
_KEY_RE = re.compile(
    r'^[ \t]+(enabled|in_process|run_on_deploy|interval_hours|run_at_hour_utc|'
    r'max_runtime_seconds|seed_mode):\s*(.+?)\s*(?:#.*)?$',
    re.MULTILINE,
)
_SEED_TABLE_LOOKBACK_RE = re.compile(
    r'(?ms)^[ \t]+tables:\s*\n(?:[ \t]+[^\n]*\n)*?[ \t]+lookback_days:\s*(\d+)',
)
_SEED_TABLE_STRATEGY_RE = re.compile(
    r'(?ms)^[ \t]+tables:\s*\n(?:[ \t]+[^\n]*\n)*?[ \t]+strategy:\s*(\S+)',
)
_ANALYTICS_LOOKBACK_RE = re.compile(
    r'(?ms)^[ \t]+analytics_backfill:\s*\n(?:[ \t]+[^\n]*\n)*?[ \t]+lookback_days:\s*(\d+)',
)
_KATANA_ENV_HOST_RE = re.compile(
    r'(?ms)^[ \t]{2}(dev-private|Test|Prod):\s*\n'
    r'(?:[ \t]*[^\n]*\n)*?'
    r'[ \t]+hosts:\s*\n'
    r'[ \t]+-\s*(\S+)',
)
_HEALTHCHECK_GRACE_RE = re.compile(
    r'(?m)^[ \t]+healthcheckGracePeriod:\s*(\d+)',
)


def _expand_env(value: str) -> str:
    def _replace(m: Match[str]) -> str:
        var = m.group(1)
        default = m.group(2)
        env_val = os.environ.get(var)
        if env_val is None:
            if default is not None:
                return default
            return m.group(0)
        return env_val

    return _ENV_VAR_RE.sub(_replace, value)


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _coerce_bool(value: str) -> bool:
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


def read_schedule(base_yaml: Path) -> Dict[str, Any]:
    text = base_yaml.read_text(encoding='utf-8')
    match = None
    for candidate in _SCHEDULE_BLOCK_RE.finditer(text):
        block = candidate.group(2)
        if 'run_on_deploy' in block or 'in_process' in block:
            match = candidate
            break
    if match is None:
        raise SystemExit(f'vectorization.seed.schedule block not found in {base_yaml}')
    block = match.group(2)
    raw: Dict[str, str] = {}
    for key_match in _KEY_RE.finditer(block):
        raw[key_match.group(1)] = _expand_env(_strip_quotes(key_match.group(2)))

    required = (
        'enabled',
        'in_process',
        'run_on_deploy',
        'interval_hours',
        'run_at_hour_utc',
        'max_runtime_seconds',
        'seed_mode',
    )
    missing = [k for k in required if k not in raw]
    if missing:
        raise SystemExit(
            f'schedule keys missing in {base_yaml}: {", ".join(missing)}'
        )

    rahu_raw = raw['run_at_hour_utc'].strip().lower()
    if rahu_raw in {'', 'null', 'none', '~'}:
        run_at_hour: Optional[int] = None
    else:
        run_at_hour = int(raw['run_at_hour_utc'])

    lookback_m = _SEED_TABLE_LOOKBACK_RE.search(text)
    strategy_m = _SEED_TABLE_STRATEGY_RE.search(text)
    analytics_m = _ANALYTICS_LOOKBACK_RE.search(text)
    if lookback_m is None:
        raise SystemExit(f'seed table lookback_days not found in {base_yaml}')
    if strategy_m is None:
        raise SystemExit(f'seed table strategy not found in {base_yaml}')
    if analytics_m is None:
        raise SystemExit(f'analytics_backfill.lookback_days not found in {base_yaml}')

    return {
        'enabled': _coerce_bool(raw['enabled']),
        'in_process': _coerce_bool(raw['in_process']),
        'run_on_deploy': _coerce_bool(raw['run_on_deploy']),
        'interval_hours': int(raw['interval_hours']),
        'run_at_hour_utc': run_at_hour,
        'max_runtime_seconds': int(raw['max_runtime_seconds']),
        'seed_mode': raw['seed_mode'].strip(),
        'seed_lookback_days': int(lookback_m.group(1)),
        'seed_strategy': strategy_m.group(1).strip('"\''),
        'analytics_lookback_days': int(analytics_m.group(1)),
    }


def read_katana_host(katana_yaml: Path, environment: str) -> Dict[str, Any]:
    text = katana_yaml.read_text(encoding='utf-8')
    hosts: Dict[str, str] = {}
    for m in _KATANA_ENV_HOST_RE.finditer(text):
        hosts[m.group(1)] = m.group(2)
    if environment not in hosts:
        raise SystemExit(
            f'ingress host for environment={environment!r} not found in {katana_yaml}; '
            f'known={sorted(hosts)}'
        )
    grace_m = _HEALTHCHECK_GRACE_RE.search(text)
    if grace_m is None:
        raise SystemExit(f'healthcheckGracePeriod not found in {katana_yaml}')
    return {
        'service_host': hosts[environment],
        'healthcheck_grace_period_seconds': int(grace_m.group(1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--yaml', type=Path, default=_DEFAULT_BASE_YAML)
    parser.add_argument('--katana-yaml', type=Path, default=_DEFAULT_KATANA_YAML)
    parser.add_argument(
        '--environment',
        default='dev-private',
        choices=['dev-private', 'Test', 'Prod'],
    )
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    if not args.yaml.is_file():
        raise SystemExit(f'config file missing: {args.yaml}')
    if not args.katana_yaml.is_file():
        raise SystemExit(f'katana config missing: {args.katana_yaml}')

    schedule = read_schedule(args.yaml)
    host_info = read_katana_host(args.katana_yaml, args.environment)
    out: Dict[str, Any] = {**schedule, **host_info, 'environment': args.environment}

    if args.json:
        print(json.dumps(out, indent=2, sort_keys=True))
        return

    # GITHUB_OUTPUT shape (bools as true/false strings)
    for key, value in out.items():
        if isinstance(value, bool):
            print(f'{key}={"true" if value else "false"}')
        elif value is None:
            print(f'{key}=')
        else:
            print(f'{key}={value}')


if __name__ == '__main__':
    main()
    sys.exit(0)
