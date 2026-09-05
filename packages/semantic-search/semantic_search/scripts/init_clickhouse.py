"""Initialize ClickHouse analytics schema: database, tables, materialized views.

Idempotent — safe to re-run. Connects via HTTP (JSONEachRow format).

Usage::

    source <venv>/bin/activate
    # Schema DDL only (idempotent, always safe):
    python -m semantic_search.scripts.init_clickhouse --host localhost --port 8123

    # DDL + async backfill mutations (run once on existing installs):
    python -m semantic_search.scripts.init_clickhouse --host localhost --port 8123 --migrate
"""
import argparse
import asyncio
import sys

import httpx

from semantic_search.analytics.ch_schema import DDL_STATEMENTS as _DDL_STATEMENTS
from semantic_search.analytics.ch_schema import MIGRATION_STATEMENTS as _MIGRATION_STATEMENTS


async def _execute_statements(
    client: httpx.AsyncClient, statements: list, label: str
) -> None:
    total = len(statements)
    for i, stmt in enumerate(statements):
        sql = stmt.strip()
        resp = await client.post("/", content=sql, headers={"Content-Type": "text/plain"})
        resp.raise_for_status()
        first_line = sql.split("\n")[0][:80]
        print(f"[{label} {i + 1}/{total}] OK: {first_line}")


async def run_ddl(host: str, port: int, timeout: float, migrate: bool) -> None:
    """Execute DDL (and optionally backfill mutations) against ClickHouse HTTP interface.
    :param host: str - ClickHouse host
    :param port: int - ClickHouse HTTP port
    :param timeout: float - HTTP request timeout in seconds
    :param migrate: bool - When True also run MIGRATION_STATEMENTS (async CH mutations)
    """
    base_url = f"http://{host}:{port}"
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
        await _execute_statements(client, _DDL_STATEMENTS, "DDL")
        print(f"Schema initialized at {base_url} ({len(_DDL_STATEMENTS)} statements)")
        if migrate:
            print(f"Running {len(_MIGRATION_STATEMENTS)} backfill mutations (async in CH)...")
            await _execute_statements(client, _MIGRATION_STATEMENTS, "MIG")
            print(
                f"Mutations submitted. ClickHouse runs them asynchronously — "
                f"check system.mutations for progress."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize ClickHouse analytics schema")
    parser.add_argument("--host", default="localhost", help="ClickHouse host")
    parser.add_argument("--port", type=int, default=8123, help="ClickHouse HTTP port")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP request timeout in seconds")
    parser.add_argument(
        "--migrate",
        action="store_true",
        default=False,
        help="Also run backfill mutations for expiry_status and auction_type_name",
    )
    args = parser.parse_args()
    try:
        asyncio.run(run_ddl(args.host, args.port, args.timeout, args.migrate))
    except httpx.HTTPStatusError as e:
        print(f"ClickHouse statement failed: {e}", file=sys.stderr)
        sys.exit(1)
    except httpx.ConnectError as e:
        print(f"Cannot connect to ClickHouse at {args.host}:{args.port}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
