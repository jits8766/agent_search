"""Tests for SignalAdapter — duck-typed FeedbackStore, no-op safety, aggregation."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from llm_core.signal_adapter import SignalAdapter, RuntimeStats


_LAT_WINDOW = 3600
_QUAL_WINDOW = 86400
_MAX_ENTRIES = 5000


def _make_adapter(store, latency_window_s: int = _LAT_WINDOW) -> SignalAdapter:
    return SignalAdapter(
        store=store,
        latency_window_s=latency_window_s,
        quality_window_s=_QUAL_WINDOW,
        max_entries_per_type=_MAX_ENTRIES,
    )


class _FakeStore:
    def __init__(self, entries_by_type):
        self._entries = entries_by_type

    def get_entries(self, signal_type=None, limit=None):
        return list(self._entries.get(signal_type, []))


def _entry(signal_type, value, model, ts=None):
    return SimpleNamespace(
        timestamp=(ts or datetime.now(timezone.utc)).isoformat(),
        source='test',
        signal_type=signal_type,
        value=value,
        metadata={'model': model},
        origin_system='test',
    )


def test_runtimestats_to_dict():
    s = RuntimeStats(p50_latency_ms=100.0, error_rate=0.1, quality_score=0.8, sample_count=10)
    d = s.to_dict()
    assert d == {'p50_latency_ms': 100.0, 'error_rate': 0.1, 'quality_score': 0.8, 'sample_count': 10}


def test_no_store_is_noop():
    a = _make_adapter(store=None)
    assert a.enabled is False
    assert a.collect() == {}


def test_basic_latency_aggregation():
    store = _FakeStore({
        'latency': [_entry('latency', 100.0, 'gpt-4o'), _entry('latency', 300.0, 'gpt-4o'), _entry('latency', 200.0, 'gpt-4o')],
        'error': [],
        'eval_quality_score': [],
    })
    out = _make_adapter(store=store).collect()
    assert 'gpt-4o' in out
    assert out['gpt-4o']['p50_latency_ms'] == 200.0
    assert out['gpt-4o']['error_rate'] == 0.0


def test_error_rate_computed():
    store = _FakeStore({
        'latency': [_entry('latency', 100.0, 'm1') for _ in range(8)],
        'error': [_entry('error', 1.0, 'm1') for _ in range(2)],
        'eval_quality_score': [],
    })
    out = _make_adapter(store=store).collect()
    assert out['m1']['error_rate'] == 0.2
    assert out['m1']['sample_count'] == 10


def test_window_filters_old_entries():
    old_ts = datetime.now(timezone.utc) - timedelta(seconds=99999)
    store = _FakeStore({
        'latency': [_entry('latency', 50.0, 'm1', ts=old_ts)],
        'error': [],
        'eval_quality_score': [],
    })
    out = _make_adapter(store=store, latency_window_s=60).collect()
    assert out == {}


def test_quality_score_aggregation():
    store = _FakeStore({
        'latency': [],
        'error': [],
        'eval_quality_score': [_entry('eval_quality_score', 0.8, 'm1'), _entry('eval_quality_score', 0.6, 'm1')],
    })
    out = _make_adapter(store=store).collect()
    assert abs(out['m1']['quality_score'] - 0.7) < 1e-9


def test_missing_model_metadata_skipped():
    bad = SimpleNamespace(timestamp=datetime.now(timezone.utc).isoformat(), source='test',
                          signal_type='latency', value=100.0, metadata={}, origin_system='test')
    store = _FakeStore({'latency': [bad], 'error': [], 'eval_quality_score': []})
    assert _make_adapter(store=store).collect() == {}


def test_store_exception_safe():
    class Boom:
        def get_entries(self, **kwargs):
            raise RuntimeError("nope")
    out = _make_adapter(store=Boom()).collect()
    assert out == {}
