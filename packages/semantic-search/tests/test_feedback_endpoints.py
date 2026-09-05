"""Tests for POST /feedback and GET /feedback endpoints.

POST: records UAT signal to in-memory store + fires S3 upload (unconditionally).
GET:  reads S3 CSVs for a date range (default last 7 days), returns merged CSV download.
"""
import asyncio
import csv
import io
from datetime import datetime, timedelta, timezone
from typing import Any, Coroutine
from unittest.mock import AsyncMock, MagicMock, patch  # AsyncMock: signal_store.record_async

import pytest
from fastapi import HTTPException

from semantic_search import app as app_module
from semantic_search.app import submit_feedback, get_feedback
from semantic_search.contracts import FeedbackSignal


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_subsystems(feedback_enabled: bool = True, max_comment_chars: int = 500) -> MagicMock:
    """Minimal Subsystems stub with a feedback-capable config and signal_store."""
    fb_cfg = MagicMock()
    fb_cfg.enabled = feedback_enabled
    fb_cfg.uat_max_comment_chars = max_comment_chars
    fb_cfg.uat_api_key_env_var = None

    id_cfg = MagicMock()
    id_cfg.request_id_prefix = "req"
    id_cfg.search_id_prefix = "search"
    id_cfg.id_hex_length = 12
    id_cfg.max_id_chars = 128
    id_cfg.id_value_pattern = r'^[A-Za-z0-9_.:-]+$'
    id_cfg.feedback_search_id_mode = "soft_generate"

    cfg = MagicMock()
    cfg.feedback = fb_cfg
    cfg.identity = id_cfg

    signal_store = MagicMock()
    signal_store.record_async = AsyncMock()

    sub = MagicMock()
    sub.config = cfg
    sub.signal_store = signal_store
    return sub


def _run(coro: Coroutine[Any, Any, Any]) -> Any:
    return asyncio.run(coro)


def _drain_create_task(coro: Any, *args: Any, **kwargs: Any) -> MagicMock:
    """Mock create_task: close fire-and-forget coro so pytest stays warning-clean."""
    if asyncio.iscoroutine(coro):
        coro.close()
    return MagicMock(name='Task')


# ---------------------------------------------------------------------------
# POST /feedback
# ---------------------------------------------------------------------------

class TestSubmitFeedback:

    def test_valid_comment_returns_recorded(self) -> None:
        sub = _make_subsystems()
        with patch.object(app_module, 'app_state') as state, \
             patch.object(app_module, 'upload_feedback_csv'), \
             patch.object(app_module.asyncio, 'create_task', side_effect=_drain_create_task):
            state.subsystems = sub
            resp = _run(submit_feedback(comment="great results", query="cheap .com"))

        assert resp['status'] == 'recorded'
        assert resp['signal_id']
        assert resp['request_id']
        assert resp['search_id']
        assert resp['search_id'].startswith('search_')
        assert resp['recorded_at']
        sub.signal_store.record_async.assert_awaited_once()

    def test_client_search_id_preferred(self) -> None:
        sub = _make_subsystems()
        recorded: list[FeedbackSignal] = []

        async def _capture(signal: FeedbackSignal) -> None:
            recorded.append(signal)

        sub.signal_store.record_async = _capture
        with patch.object(app_module, 'app_state') as state, \
             patch.object(app_module, 'upload_feedback_csv'), \
             patch.object(app_module.asyncio, 'create_task', side_effect=_drain_create_task):
            state.subsystems = sub
            resp = _run(submit_feedback(
                comment="filters look right",
                query="cheap .io",
                search_id="search_from_client_abc",
                request_id="req_from_search_abc",
            ))

        assert resp['search_id'] == 'search_from_client_abc'
        assert resp['request_id'] == 'req_from_search_abc'
        assert recorded[0].search_id == 'search_from_client_abc'
        assert recorded[0].request_id == 'req_from_search_abc'

    def test_missing_search_id_soft_generates(self) -> None:
        sub = _make_subsystems()
        recorded: list[FeedbackSignal] = []

        async def _capture(signal: FeedbackSignal) -> None:
            recorded.append(signal)

        sub.signal_store.record_async = _capture
        with patch.object(app_module, 'app_state') as state, \
             patch.object(app_module, 'upload_feedback_csv'), \
             patch.object(app_module.asyncio, 'create_task', side_effect=_drain_create_task):
            state.subsystems = sub
            resp = _run(submit_feedback(comment="no id from client", query=None, search_id=None, request_id=None))

        assert resp['search_id']
        assert resp['search_id'].startswith('search_')
        assert resp['request_id']
        assert resp['request_id'].startswith('req_')
        assert recorded[0].search_id == resp['search_id']
        assert recorded[0].request_id == resp['request_id']

    def test_required_search_id_mode_rejects_missing(self) -> None:
        sub = _make_subsystems()
        sub.config.identity.feedback_search_id_mode = "required"
        with patch.object(app_module, 'app_state') as state:
            state.subsystems = sub
            with pytest.raises(HTTPException) as ei:
                _run(submit_feedback(comment="need search_id", search_id=None))
        assert ei.value.status_code == 422

    def test_rejects_search_id_with_control_chars(self) -> None:
        sub = _make_subsystems()
        with patch.object(app_module, 'app_state') as state:
            state.subsystems = sub
            with pytest.raises(HTTPException) as ei:
                _run(submit_feedback(comment="bad id", search_id="search_ab\nc"))
        assert ei.value.status_code == 422

    def test_rejects_search_id_outside_pattern(self) -> None:
        sub = _make_subsystems()
        with patch.object(app_module, 'app_state') as state:
            state.subsystems = sub
            with pytest.raises(HTTPException) as ei:
                _run(submit_feedback(comment="bad id", search_id="search id with spaces"))
        assert ei.value.status_code == 422

    def test_upload_fired_unconditionally(self) -> None:
        """S3 upload create_task called regardless of environment."""
        sub = _make_subsystems()
        captured_tasks: list[Any] = []

        def _capture(coro: Any, *args: Any, **kwargs: Any) -> MagicMock:
            captured_tasks.append(coro)
            return _drain_create_task(coro)

        with patch.object(app_module, 'app_state') as state, \
             patch.object(app_module, 'upload_feedback_csv'), \
             patch.object(app_module.asyncio, 'create_task', side_effect=_capture):
            state.subsystems = sub
            _run(submit_feedback(comment="test upload trigger", query=None))

        assert len(captured_tasks) == 1

    def test_api_key_gate_rejects_missing_header(self) -> None:
        """When the secret env var is provisioned, a request with no X-Feedback-Key is 401."""
        sub = _make_subsystems()
        sub.config.feedback.uat_api_key_env_var = 'FEEDBACK_UAT_API_KEY'
        with patch.object(app_module, 'app_state') as state, \
             patch.dict(app_module.os.environ, {'FEEDBACK_UAT_API_KEY': 's3cret'}):
            state.subsystems = sub
            with pytest.raises(HTTPException) as exc_info:
                _run(submit_feedback(comment="hello", x_feedback_key=None))
        assert exc_info.value.status_code == 401

    def test_api_key_gate_rejects_wrong_key(self) -> None:
        sub = _make_subsystems()
        sub.config.feedback.uat_api_key_env_var = 'FEEDBACK_UAT_API_KEY'
        with patch.object(app_module, 'app_state') as state, \
             patch.dict(app_module.os.environ, {'FEEDBACK_UAT_API_KEY': 's3cret'}):
            state.subsystems = sub
            with pytest.raises(HTTPException) as exc_info:
                _run(submit_feedback(comment="hello", x_feedback_key="wrong"))
        assert exc_info.value.status_code == 401

    def test_api_key_gate_accepts_correct_key(self) -> None:
        sub = _make_subsystems()
        sub.config.feedback.uat_api_key_env_var = 'FEEDBACK_UAT_API_KEY'
        with patch.object(app_module, 'app_state') as state, \
             patch.object(app_module, 'upload_feedback_csv'), \
             patch.object(app_module.asyncio, 'create_task', side_effect=_drain_create_task), \
             patch.dict(app_module.os.environ, {'FEEDBACK_UAT_API_KEY': 's3cret'}):
            state.subsystems = sub
            resp = _run(submit_feedback(comment="hello", query=None, x_feedback_key="s3cret"))
        assert resp['status'] == 'recorded'

    def test_api_key_gate_rejects_when_env_unset(self) -> None:
        """Config names the env var but it is unset: fail closed with 403, not silently open."""
        sub = _make_subsystems()
        sub.config.feedback.uat_api_key_env_var = 'FEEDBACK_UAT_API_KEY'
        env_no_key = {k: v for k, v in app_module.os.environ.items() if k != 'FEEDBACK_UAT_API_KEY'}
        with patch.object(app_module, 'app_state') as state, \
             patch.dict(app_module.os.environ, env_no_key, clear=True):
            state.subsystems = sub
            with pytest.raises(HTTPException) as exc_info:
                _run(submit_feedback(comment="hello", query=None, x_feedback_key=None))
        assert exc_info.value.status_code == 403

    def test_feedback_disabled_returns_503(self) -> None:
        sub = _make_subsystems(feedback_enabled=False)
        with patch.object(app_module, 'app_state') as state:
            state.subsystems = sub
            with pytest.raises(HTTPException) as exc_info:
                _run(submit_feedback(comment="hello"))
        assert exc_info.value.status_code == 503

    def test_empty_comment_returns_422(self) -> None:
        sub = _make_subsystems()
        with patch.object(app_module, 'app_state') as state:
            state.subsystems = sub
            with pytest.raises(HTTPException) as exc_info:
                _run(submit_feedback(comment="   "))
        assert exc_info.value.status_code == 422

    def test_comment_too_long_returns_422(self) -> None:
        sub = _make_subsystems(max_comment_chars=10)
        with patch.object(app_module, 'app_state') as state:
            state.subsystems = sub
            with pytest.raises(HTTPException) as exc_info:
                _run(submit_feedback(comment="x" * 11))
        assert exc_info.value.status_code == 422

    def test_comment_stripped_before_length_check(self) -> None:
        """Leading/trailing whitespace stripped before length validation."""
        sub = _make_subsystems(max_comment_chars=5)
        with patch.object(app_module, 'app_state') as state, \
             patch.object(app_module, 'upload_feedback_csv'), \
             patch.object(app_module.asyncio, 'create_task', side_effect=_drain_create_task):
            state.subsystems = sub
            # "hello" stripped = 5 chars = exactly at limit (not over)
            resp = _run(submit_feedback(comment="  hello  ", query=None))
        assert resp['status'] == 'recorded'

    def test_signal_carries_query(self) -> None:
        sub = _make_subsystems()
        recorded: list[FeedbackSignal] = []

        async def _capture(signal: FeedbackSignal) -> None:
            recorded.append(signal)

        sub.signal_store.record_async = _capture
        with patch.object(app_module, 'app_state') as state, \
             patch.object(app_module, 'upload_feedback_csv'), \
             patch.object(app_module.asyncio, 'create_task', side_effect=_drain_create_task):
            state.subsystems = sub
            _run(submit_feedback(comment="good domain", query="cheap .io"))

        assert len(recorded) == 1
        sig = recorded[0]
        assert sig.payload['comment'] == 'good domain'
        assert sig.payload['query'] == 'cheap .io'
        assert sig.signal_type == 'uat_feedback'
        assert sig.signal_origin == 'frontend'

    def test_query_none_stored_as_none(self) -> None:
        sub = _make_subsystems()
        recorded: list[FeedbackSignal] = []

        async def _capture(signal: FeedbackSignal) -> None:
            recorded.append(signal)

        sub.signal_store.record_async = _capture
        with patch.object(app_module, 'app_state') as state, \
             patch.object(app_module, 'upload_feedback_csv'), \
             patch.object(app_module.asyncio, 'create_task', side_effect=_drain_create_task):
            state.subsystems = sub
            _run(submit_feedback(comment="feedback without query", query=None))

        assert recorded[0].payload['query'] is None

    def test_recorded_at_is_iso8601_utc(self) -> None:
        sub = _make_subsystems()
        with patch.object(app_module, 'app_state') as state, \
             patch.object(app_module, 'upload_feedback_csv'), \
             patch.object(app_module.asyncio, 'create_task', side_effect=_drain_create_task):
            state.subsystems = sub
            resp = _run(submit_feedback(comment="iso check", query=None))
        datetime.strptime(resp['recorded_at'], '%Y-%m-%dT%H:%M:%SZ')
        assert resp['recorded_at'].endswith('Z')


# ---------------------------------------------------------------------------
# GET /feedback
# ---------------------------------------------------------------------------

_SAMPLE_ROWS = [
    {
        'signal_id': 'sig_abc123',
        'search_id': 'search_001',
        'request_id': 'req_001',
        'signal_type': 'uat_feedback',
        'signal_origin': 'frontend',
        'created_at_iso': '2026-06-30T10:00:00Z',
        'comment': 'very helpful',
        'query': 'short .com under 500',
    },
    {
        'signal_id': 'sig_def456',
        'search_id': 'search_002',
        'request_id': 'req_002',
        'signal_type': 'uat_feedback',
        'signal_origin': 'frontend',
        'created_at_iso': '2026-06-29T08:30:00Z',
        'comment': 'results could be better',
        'query': None,
    },
]


class TestGetFeedback:

    def test_returns_csv_content_type(self) -> None:
        with patch.object(app_module, 'read_feedback_from_s3', return_value=_SAMPLE_ROWS):
            resp = _run(get_feedback())
        assert resp.media_type == 'text/csv'

    def test_csv_contains_all_expected_headers(self) -> None:
        with patch.object(app_module, 'read_feedback_from_s3', return_value=_SAMPLE_ROWS):
            resp = _run(get_feedback())
        content = resp.body.decode('utf-8')
        reader = csv.DictReader(io.StringIO(content))
        assert set(reader.fieldnames or []) == {
            'signal_id', 'search_id', 'request_id', 'signal_type',
            'signal_origin', 'created_at_iso', 'comment', 'query',
        }

    def test_csv_rows_match_s3_data(self) -> None:
        with patch.object(app_module, 'read_feedback_from_s3', return_value=_SAMPLE_ROWS):
            resp = _run(get_feedback())
        content = resp.body.decode('utf-8')
        rows = list(csv.DictReader(io.StringIO(content)))
        assert len(rows) == 2
        assert rows[0]['signal_id'] == 'sig_abc123'
        assert rows[0]['comment'] == 'very helpful'
        assert rows[1]['signal_id'] == 'sig_def456'

    def test_content_disposition_filename_contains_dates(self) -> None:
        date_from = '2026-06-25'
        date_to = '2026-07-02'
        with patch.object(app_module, 'read_feedback_from_s3', return_value=[]):
            resp = _run(get_feedback(date_from=date_from, date_to=date_to))
        disp = resp.headers['content-disposition']
        assert date_from in disp
        assert date_to in disp

    def test_default_date_range_is_7_days(self) -> None:
        """date_from defaults to today-7d, date_to to today."""
        captured: dict[str, str] = {}

        def _mock_s3(date_from: str, date_to: str) -> list[Any]:
            captured['date_from'] = date_from
            captured['date_to'] = date_to
            return []

        with patch.object(app_module, 'read_feedback_from_s3', side_effect=_mock_s3):
            _run(get_feedback())

        today = datetime.now(tz=timezone.utc).date()
        assert captured['date_to'] == today.isoformat()
        assert captured['date_from'] == (today - timedelta(days=7)).isoformat()

    def test_explicit_dates_passed_to_s3(self) -> None:
        captured: dict[str, str] = {}

        def _mock_s3(date_from: str, date_to: str) -> list[Any]:
            captured['date_from'] = date_from
            captured['date_to'] = date_to
            return []

        with patch.object(app_module, 'read_feedback_from_s3', side_effect=_mock_s3):
            _run(get_feedback(date_from='2026-06-01', date_to='2026-06-15'))

        assert captured['date_from'] == '2026-06-01'
        assert captured['date_to'] == '2026-06-15'

    def test_s3_unavailable_returns_503(self) -> None:
        with patch.object(
            app_module, 'read_feedback_from_s3',
            side_effect=RuntimeError('S3_PRETRAINED_DIR is not set or is not an s3:// URI'),
        ):
            with pytest.raises(HTTPException) as exc_info:
                _run(get_feedback())
        assert exc_info.value.status_code == 503

    def test_empty_s3_result_returns_csv_headers_only(self) -> None:
        with patch.object(app_module, 'read_feedback_from_s3', return_value=[]):
            resp = _run(get_feedback())
        content = resp.body.decode('utf-8')
        rows = list(csv.DictReader(io.StringIO(content)))
        assert rows == []
        assert 'signal_id' in content

    def test_only_csv_fields_present_in_output(self) -> None:
        """Extra keys in S3 rows are stripped (extrasaction='ignore')."""
        rows_with_extra = [dict(_SAMPLE_ROWS[0], unexpected_col='drop_me')]
        with patch.object(app_module, 'read_feedback_from_s3', return_value=rows_with_extra):
            resp = _run(get_feedback())
        content = resp.body.decode('utf-8')
        reader = csv.DictReader(io.StringIO(content))
        assert 'unexpected_col' not in (reader.fieldnames or [])
