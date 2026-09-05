#!/usr/bin/env bash
# Bring up local stack honoring clickhouse.enabled from base.yaml.
#
#   clickhouse.enabled: true  → Qdrant + ClickHouse + API (waits for CH healthy)
#   clickhouse.enabled: false → Qdrant + API only
#
# Usage:
#   scripts/compose-up.sh
#   scripts/compose-up.sh --build
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
ENABLED="$(python3 scripts/read_clickhouse_enabled.py --value-only)"
if [[ "${ENABLED}" == "true" ]]; then
  echo "compose_up clickhouse.enabled=true profile=analytics"
  exec docker compose \
    -f docker-compose.yaml \
    -f docker-compose.analytics.yaml \
    --profile analytics \
    up -d "$@"
else
  echo "compose_up clickhouse.enabled=false services=qdrant,agent-search"
  exec docker compose -f docker-compose.yaml up -d "$@"
fi
