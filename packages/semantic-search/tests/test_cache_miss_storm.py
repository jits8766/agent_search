"""Tests for the CacheMissStormDetector.

Coverage matrix (per ``testing.mdc`` §7):

``CacheMissStormConfig``:
- defaults_validate                        -> TestCacheMissStormConfig::test_defaults_validate
- window_seconds_non_positive_rejected     -> TestCacheMissStormConfig::test_window_seconds_rejected
- hit_rate_min_out_of_range_rejected       -> TestCacheMissStormConfig::test_hit_rate_min_rejected
- min_sample_size_zero_rejected            -> TestCacheMissStormConfig::test_min_sample_size_rejected
- empty_tiers_rejected                     -> TestCacheMissStormConfig::test_empty_tiers_rejected
- tiers_non_string_rejected                -> TestCacheMissStormConfig::test_tiers_non_string_rejected

``CacheMissStormDetector``:
- disabled_returns_unbreached              -> TestCacheMissStormDetector::test_disabled_unbreached
- first_poll_seeds_anchor_no_breach        -> TestCacheMissStormDetector::test_first_poll_seeds_anchor
- below_min_sample_returns_insufficient    -> TestCacheMissStormDetector::test_below_min_sample_insufficient
- above_floor_no_breach                    -> TestCacheMissStormDetector::test_above_floor_no_breach
- below_floor_breach_emits_signal          -> TestCacheMissStormDetector::test_breach_emits_signal
- breach_only_emits_once_per_transition    -> TestCacheMissStormDetector::test_breach_emits_once
- recovery_clears_breach_no_recovery_signal-> TestCacheMissStormDetector::test_recovery_clears_breach
- counter_reset_handled_safely             -> TestCacheMissStormDetector::test_counter_reset_handled
- old_samples_trimmed_outside_window       -> TestCacheMissStormDetector::test_old_samples_trimmed
- negative_counter_rejected                -> TestCacheMissStormDetector::test_negative_counter_rejected
- reset_clears_samples                     -> TestCacheMissStormDetector::test_reset_clears_samples
- status_does_not_perturb_window           -> TestCacheMissStormDetector::test_status_no_perturb
"""
import dataclasses
import os
import tempfile
import time

import pytest

from semantic_search.config.models import AgentSearchConfig, CacheMissStormConfig
from semantic_search.core.exceptions import ConfigurationError, ValidationError
from semantic_search.signal_store import SignalStore
from semantic_search.measurement.cache_miss_storm import CacheMissStormDetector, CacheMissStormStatus


def _mk_cfg(**overrides) -> CacheMissStormConfig:
    base = dict(
        enabled=True,
        window_seconds=300.0,
        hit_rate_min=0.30,
        min_sample_size=10,
        tiers=['exact', 'intent_plan'],
    )
    base.update(overrides)
    return CacheMissStormConfig(**base)


class TestCacheMissStormConfig:
    def test_defaults_validate(self):
        cfg = _mk_cfg()
        assert cfg.enabled is True
        assert cfg.window_seconds == 300.0
        assert cfg.hit_rate_min == 0.30

    def test_window_seconds_rejected(self):
        with pytest.raises(ConfigurationError, match="window_seconds"):
            _mk_cfg(window_seconds=0.0)
        with pytest.raises(ConfigurationError, match="window_seconds"):
            _mk_cfg(window_seconds=-1.0)

    def test_hit_rate_min_rejected(self):
        with pytest.raises(ConfigurationError, match="hit_rate_min"):
            _mk_cfg(hit_rate_min=0.0)
        with pytest.raises(ConfigurationError, match="hit_rate_min"):
            _mk_cfg(hit_rate_min=1.0)
        with pytest.raises(ConfigurationError, match="hit_rate_min"):
            _mk_cfg(hit_rate_min=1.5)

    def test_min_sample_size_rejected(self):
        with pytest.raises(ConfigurationError, match="min_sample_size"):
            _mk_cfg(min_sample_size=0)

    def test_empty_tiers_rejected(self):
        with pytest.raises(ConfigurationError, match="tiers"):
            _mk_cfg(tiers=[])

    def test_tiers_non_string_rejected(self):
        with pytest.raises(ConfigurationError, match="tiers"):
            _mk_cfg(tiers=['exact', ''])


@pytest.fixture()
def signal_store(config: AgentSearchConfig) -> SignalStore:
    """SignalStore with per-test JSONL log path."""
    tmpdir = tempfile.mkdtemp()
    fb = dataclasses.replace(config.feedback, signal_log_path=os.path.join(tmpdir, 'sigs.jsonl'))
    return SignalStore(fb)


class TestCacheMissStormDetector:
    def test_disabled_unbreached(self, signal_store: SignalStore):
        cfg = _mk_cfg(enabled=False)
        store = signal_store
        det = CacheMissStormDetector(cfg, store)
        # Below-floor traffic, but disabled → unbreached, no signals.
        det.poll(cumulative_hits=10, cumulative_misses=90)
        det.poll(cumulative_hits=20, cumulative_misses=180)
        status = det.poll(cumulative_hits=30, cumulative_misses=270)
        assert status.breached is False
        assert len(store.recent(100)) == 0

    def test_first_poll_seeds_anchor(self, signal_store: SignalStore):
        cfg = _mk_cfg(min_sample_size=1)
        det = CacheMissStormDetector(cfg, signal_store)
        # First poll has no prior anchor → delta is (0, 0) → no sample recorded.
        status = det.poll(cumulative_hits=100, cumulative_misses=10)
        assert status.sample_size == 0
        assert status.breached is False
        # Second poll: deltas are non-zero → sample recorded.
        status2 = det.poll(cumulative_hits=110, cumulative_misses=15)
        assert status2.sample_size == 15  # 10 hits + 5 misses

    def test_below_min_sample_insufficient(self, signal_store: SignalStore):
        cfg = _mk_cfg(min_sample_size=100)
        det = CacheMissStormDetector(cfg, signal_store)
        det.poll(cumulative_hits=0, cumulative_misses=0)
        # Add 30 misses < min_sample_size=100 → insufficient.
        status = det.poll(cumulative_hits=0, cumulative_misses=30)
        assert status.sample_size == 30
        assert status.hit_rate is None
        assert status.breached is False

    def test_above_floor_no_breach(self, signal_store: SignalStore):
        cfg = _mk_cfg(min_sample_size=10, hit_rate_min=0.30)
        det = CacheMissStormDetector(cfg, signal_store)
        det.poll(cumulative_hits=0, cumulative_misses=0)
        # Hit rate 80 / (80+20) = 0.80 — above floor.
        status = det.poll(cumulative_hits=80, cumulative_misses=20)
        assert status.hit_rate == pytest.approx(0.80)
        assert status.breached is False
        assert len(signal_store.recent(100)) == 0

    def test_breach_emits_signal(self, signal_store: SignalStore):
        cfg = _mk_cfg(min_sample_size=10, hit_rate_min=0.30)
        det = CacheMissStormDetector(cfg, signal_store)
        det.poll(cumulative_hits=0, cumulative_misses=0)
        # Hit rate 5 / (5+95) = 0.05 — well below 0.30 floor.
        status = det.poll(cumulative_hits=5, cumulative_misses=95)
        assert status.hit_rate == pytest.approx(0.05)
        assert status.breached is True
        signals = [s for s in signal_store.recent(100) if s.signal_type == 'cache_miss_storm']
        assert len(signals) == 1
        sig = signals[0]
        assert sig.signal_origin == 'cache'
        assert sig.payload['hit_rate'] == pytest.approx(0.05)
        assert sig.payload['hit_rate_min'] == 0.30
        assert sig.payload['window_seconds'] == 300.0
        assert sig.payload['sample_size'] == 100
        assert 'exact' in sig.payload['tiers']

    def test_breach_emits_once(self, signal_store: SignalStore):
        cfg = _mk_cfg(min_sample_size=10, hit_rate_min=0.30)
        det = CacheMissStormDetector(cfg, signal_store)
        det.poll(0, 0)
        det.poll(5, 95)  # breach — emits
        det.poll(7, 193)  # still breaching — must NOT re-emit
        det.poll(9, 291)  # still breaching — must NOT re-emit
        signals = [s for s in signal_store.recent(100) if s.signal_type == 'cache_miss_storm']
        assert len(signals) == 1

    def test_recovery_clears_breach(self, signal_store: SignalStore):
        cfg = _mk_cfg(min_sample_size=10, hit_rate_min=0.30)
        det = CacheMissStormDetector(cfg, signal_store)
        det.poll(0, 0)
        det.poll(5, 95)  # breach
        # Recovery: a later poll with hit-rate above floor in the window.
        # Cumulative hits jump well above misses to flip the rolling rate.
        det.poll(305, 195)  # +300 hits, +100 misses → 400 hits / 500 total = 0.80
        status = det.poll(605, 295)  # same trend
        assert status.breached is False
        # Only the breach edge emits — recovery is silent.
        signals = [s for s in signal_store.recent(100) if s.signal_type == 'cache_miss_storm']
        assert len(signals) == 1

    def test_counter_reset_handled(self, signal_store: SignalStore):
        cfg = _mk_cfg(min_sample_size=10)
        det = CacheMissStormDetector(cfg, signal_store)
        det.poll(100, 50)
        det.poll(200, 100)
        # Counter "resets" (e.g. process restart): cur < prev → delta clamped to 0.
        status = det.poll(10, 5)
        # No new sample added, anchor refreshed → next poll measures from here.
        assert status.sample_size <= 150  # samples from prior polls only

    def test_old_samples_trimmed(self, signal_store: SignalStore):
        cfg = _mk_cfg(min_sample_size=1, window_seconds=0.05)
        det = CacheMissStormDetector(cfg, signal_store)
        det.poll(0, 0)
        det.poll(50, 50)  # records sample
        time.sleep(0.10)  # exceed window
        # Force a poll with a tiny new delta — the old sample must trim.
        status = det.poll(51, 50)
        assert status.sample_size == 1  # only the latest 1-hit delta

    def test_negative_counter_rejected(self, signal_store: SignalStore):
        det = CacheMissStormDetector(_mk_cfg(), signal_store)
        with pytest.raises(ValidationError):
            det.poll(-1, 0)
        with pytest.raises(ValidationError):
            det.poll(0, -1)

    def test_reset_clears_samples(self, signal_store: SignalStore):
        cfg = _mk_cfg(min_sample_size=1)
        det = CacheMissStormDetector(cfg, signal_store)
        det.poll(0, 0)
        det.poll(50, 50)
        assert det.status().sample_size == 100
        det.reset()
        # After reset the anchor is None — first poll seeds, no sample.
        s = det.poll(60, 60)
        assert s.sample_size == 0

    def test_status_no_perturb(self, signal_store: SignalStore):
        cfg = _mk_cfg(min_sample_size=1)
        det = CacheMissStormDetector(cfg, signal_store)
        det.poll(0, 0)
        det.poll(50, 50)
        s1 = det.status()
        s2 = det.status()
        # Status calls do not record samples → values stay identical.
        assert s1.sample_size == s2.sample_size
        assert s1.hit_rate == s2.hit_rate
