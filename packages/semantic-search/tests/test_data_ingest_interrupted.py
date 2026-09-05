"""Data-ingest interrupt envelope: HTTP 503 + records completed + why (ALB/cancel)."""
from __future__ import annotations

import asyncio
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from semantic_search.app import (
    _data_ingest_interrupted_envelope,
    _data_ingest_interrupted_response,
    app_state,
)
from semantic_search.core.exceptions import DataIngestInterruptedError
from semantic_search.explore.ch_seed_writer import insert_seed_to_clickhouse


def test_envelope_from_typed_interrupt() -> None:
    exc = DataIngestInterruptedError(
        'ClickHouse seed interrupted after 1200 of 5000 rows.',
        stage='clickhouse_seed',
        records_completed=1200,
        records_attempted=5000,
        reason='cancelled',
        detail='CancelledError: batch_start=1200',
    )
    body = _data_ingest_interrupted_envelope(
        endpoint='/data-build/seed',
        exc=exc,
        started_at='2026-07-18T00:00:00Z',
    )
    assert body['status'] == 'interrupted'
    assert body['endpoint'] == '/data-build/seed'
    assert body['stage'] == 'clickhouse_seed'
    assert body['records_completed'] == 1200
    assert body['records_attempted'] == 5000
    assert body['reason'] == 'cancelled'
    assert '1200' in body['message']
    assert 'ALB' in body['hint'] or 'idle timeout' in body['hint'].lower()


def test_envelope_from_cancelled_uses_progress() -> None:
    app_state._ingest_progress = {
        'stage': 'qdrant_index',
        'records_completed': 350,
        'records_attempted': 1000,
    }
    try:
        body = _data_ingest_interrupted_envelope(
            endpoint='/data-build/seed',
            exc=asyncio.CancelledError(),
            started_at='2026-07-18T00:00:00Z',
        )
    finally:
        app_state._ingest_progress = None
    assert body['status'] == 'interrupted'
    assert body['stage'] == 'qdrant_index'
    assert body['records_completed'] == 350
    assert body['records_attempted'] == 1000
    assert body['reason'] == 'cancelled'


def test_response_is_http_503_and_records_history() -> None:
    before = len(app_state._build_history)
    exc = DataIngestInterruptedError(
        'Qdrant indexing interrupted after 42 of 100 points.',
        stage='qdrant_index',
        records_completed=42,
        records_attempted=100,
        reason='cancelled',
    )
    resp = _data_ingest_interrupted_response(
        endpoint='/data-build/seed',
        exc=exc,
        started_at='2026-07-18T00:00:00Z',
    )
    assert resp.status_code == 503
    assert len(app_state._build_history) == before + 1
    last = app_state._build_history[-1]
    assert last['status'] == 'interrupted'
    assert last['records_completed'] == 42
    assert last['stage'] == 'qdrant_index'


@pytest.mark.asyncio
async def test_ch_seed_raises_interrupt_on_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    docs: List[dict[str, Any]] = [
        {
            'domain_name': f'd{i}.com',
            'tld': 'com',
            'price': 10.0,
            'auction_type': 16,
            'ends_at': 1720000000.0,
        }
        for i in range(5)
    ]

    async def _ok_schema(
        _ch_executor: Any, source_table: str, *, timeout_seconds: float
    ) -> None:
        return None

    monkeypatch.setattr(
        'semantic_search.explore.ch_seed_writer._ensure_schema',
        _ok_schema,
    )

    async def _insert(sql: str, timeout_seconds: float = 0.0) -> None:
        if sql.lstrip().upper().startswith('INSERT'):
            raise asyncio.CancelledError()

    ch_exec = MagicMock()
    ch_exec.execute_insert = AsyncMock(side_effect=_insert)
    with pytest.raises(DataIngestInterruptedError) as caught:
        await insert_seed_to_clickhouse(
            docs,
            ch_exec,
            ensure_schema=True,
            batch_size=2,
            insert_timeout_seconds=60.0,
            schema_timeout_seconds=30.0,
            target_table='signals_platform_cln.auction_audit_cln',
            snapshot_table='signals_platform_cln.domain_snapshots',
        )
    err = caught.value
    assert err.stage == 'clickhouse_seed'
    assert err.reason == 'cancelled'
    assert err.records_completed == 0
    assert err.records_attempted == 5


@pytest.mark.asyncio
async def test_ch_seed_skips_schema_when_ensure_false(monkeypatch: pytest.MonkeyPatch) -> None:
    docs: List[dict[str, Any]] = [
        {
            'domain_name': 'a.com',
            'tld': 'com',
            'price': 10.0,
            'auction_type': 16,
            'ends_at': 1720000000.0,
        }
    ]
    schema_calls = {'n': 0}

    async def _schema(*_a: Any, **_k: Any) -> None:
        schema_calls['n'] += 1
        return None

    monkeypatch.setattr(
        'semantic_search.explore.ch_seed_writer._ensure_schema',
        _schema,
    )
    ch_exec = MagicMock()
    ch_exec.execute_insert = AsyncMock(return_value=None)
    summary = await insert_seed_to_clickhouse(
        docs,
        ch_exec,
        ensure_schema=False,
        batch_size=10,
        insert_timeout_seconds=60.0,
        schema_timeout_seconds=30.0,
        target_table='signals_platform_cln.auction_audit_cln',
        snapshot_table='signals_platform_cln.domain_snapshots',
    )
    assert schema_calls['n'] == 0
    assert summary.rows_written == 1
