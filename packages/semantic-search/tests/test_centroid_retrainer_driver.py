"""Tests for ``semantic_search.qi.centroid_retrainer_driver`` and ``SemanticRouter.swap_centroids``.

Coverage matrix:

``SemanticRouter.swap_centroids``:
- valid_full_swap                          -> TestSwapCentroids::test_valid_full_swap
- valid_partial_swap                       -> TestSwapCentroids::test_valid_partial_swap
- unknown_archetype_raises                 -> TestSwapCentroids::test_unknown_archetype_raises
- dim_mismatch_raises                      -> TestSwapCentroids::test_dim_mismatch_raises
- atomic_reference_replaced               -> TestSwapCentroids::test_atomic_reference_replaced

``CentroidRetrainerDriverConfig``:
- valid_construction                       -> TestDriverConfig::test_valid_construction
- interval_below_floor_raises             -> TestDriverConfig::test_interval_below_floor_raises
- shadow_limit_below_floor_raises         -> TestDriverConfig::test_shadow_limit_below_floor_raises
- max_failures_below_floor_raises         -> TestDriverConfig::test_max_failures_below_floor_raises
- empty_signal_types_raises               -> TestDriverConfig::test_empty_signal_types_raises
- from_dict_round_trip                    -> TestDriverConfig::test_from_dict_round_trip
- from_dict_missing_field_raises          -> TestDriverConfig::test_from_dict_missing_field_raises

``CentroidRetrainerCycleSummary``:
- valid_construction                       -> TestCycleSummary::test_valid_construction
- empty_cycle_id_raises                   -> TestCycleSummary::test_empty_cycle_id_raises
- invalid_verdict_raises                  -> TestCycleSummary::test_invalid_verdict_raises
- agreement_out_of_range_raises           -> TestCycleSummary::test_agreement_out_of_range_raises

``CentroidRetrainerDriver`` (construction):
- valid_construction                       -> TestDriverConstruction::test_valid_construction
- none_config_raises                       -> TestDriverConstruction::test_none_config_raises

``CentroidRetrainerDriver.run_cycle`` (cycle logic):
- skip_when_no_positives                  -> TestRunCycle::test_skip_when_no_positives
- skip_when_build_candidate_rejects       -> TestRunCycle::test_skip_when_build_candidate_rejects
- shadow_only_verdict                     -> TestRunCycle::test_shadow_only_verdict
- promote_calls_swap_centroids            -> TestRunCycle::test_promote_calls_swap_centroids
- emits_verdict_signal                    -> TestRunCycle::test_emits_verdict_signal
- emits_promoted_signal_on_promote        -> TestRunCycle::test_emits_promoted_signal_on_promote
- error_captured_in_summary               -> TestRunCycle::test_error_captured_in_summary

``CentroidRetrainerDriver`` (lifecycle):
- start_stop_idempotent                   -> TestDriverLifecycle::test_start_stop_idempotent
- is_running_reflects_state               -> TestDriverLifecycle::test_is_running_reflects_state
"""
import asyncio
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from semantic_search.config.models import AgentSearchConfig, CentroidRetrainerConfig, CentroidRetrainerDriverConfig
from semantic_search.contracts import (CENTROID_RETRAIN_VERDICTS, CentroidRetrainCandidate, CentroidRetrainVerdict, CentroidRetrainerCycleSummary, FeedbackSignal, RouterSeedDataset)
from semantic_search.core.exceptions import ConfigurationError, ValidationError
from semantic_search.qi.centroid_retrainer import CentroidRetrainer
from semantic_search.qi.centroid_retrainer_driver import CentroidRetrainerDriver
from semantic_search.qi.encoder import HashingEncoder
from semantic_search.qi.semantic_router import SemanticRouter
from semantic_search.signal_store import SignalStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def encoder(config: AgentSearchConfig) -> HashingEncoder:
    return HashingEncoder(dim=config.qi.semantic.embedding_dim, seed=config.qi.semantic.encoder_seed)


@pytest.fixture(scope='module')
def semantic_router(config: AgentSearchConfig, encoder: HashingEncoder, router_seeds: RouterSeedDataset) -> SemanticRouter:
    return SemanticRouter(config.qi.semantic, encoder, router_seeds)


@pytest.fixture(scope='module')
def retrainer_config() -> CentroidRetrainerConfig:
    return CentroidRetrainerConfig(enabled=True, min_samples_per_archetype=2, min_shadow_agreement=0.5, window_seconds=86400.0, max_signals_per_read=500)


@pytest.fixture(scope='module')
def retrainer(retrainer_config: CentroidRetrainerConfig, encoder: HashingEncoder, semantic_router: SemanticRouter) -> CentroidRetrainer:
    return CentroidRetrainer(config=retrainer_config, encoder=encoder, current_router=semantic_router)


@pytest.fixture
def driver_config() -> CentroidRetrainerDriverConfig:
    return CentroidRetrainerDriverConfig(enabled=True, interval_seconds=60.0, shadow_query_limit=50, max_consecutive_failures=2, positive_signal_types=['result_click', 'calibration_label'])


@pytest.fixture
def signal_store(config: AgentSearchConfig) -> SignalStore:
    return SignalStore(config.feedback)


@pytest.fixture
def driver(driver_config: CentroidRetrainerDriverConfig, retrainer: CentroidRetrainer, semantic_router: SemanticRouter, signal_store: SignalStore) -> CentroidRetrainerDriver:
    return CentroidRetrainerDriver(config=driver_config, retrainer=retrainer, router=semantic_router, signal_store=signal_store)


def _make_signal(signal_type: str, query_type: str, query_text: str, config: AgentSearchConfig) -> FeedbackSignal:
    return FeedbackSignal(signal_id=FeedbackSignal.new_signal_id(), request_id='req_test', signal_type=signal_type, payload={'query_type': query_type, 'query_text': query_text}, signal_origin='orchestrator')


# ---------------------------------------------------------------------------
# SemanticRouter.swap_centroids
# ---------------------------------------------------------------------------

class TestSwapCentroids:
    def test_valid_full_swap(self, semantic_router: SemanticRouter, config: AgentSearchConfig):
        dim = config.qi.semantic.embedding_dim
        new_np = {arch: np.random.randn(1, dim).astype(np.float32) for arch in semantic_router.archetypes}
        original_ids = {arch: id(v) for arch, v in semantic_router._sub_centroids.items()}
        semantic_router.swap_centroids(new_np)
        for arch in new_np:
            assert id(semantic_router._sub_centroids[arch]) != original_ids[arch]

    def test_valid_partial_swap(self, semantic_router: SemanticRouter, config: AgentSearchConfig):
        dim = config.qi.semantic.embedding_dim
        first_arch = semantic_router.archetypes[0]
        partial = {first_arch: np.random.randn(1, dim).astype(np.float32)}
        before_other = {a: id(v) for a, v in semantic_router._sub_centroids.items() if a != first_arch}
        semantic_router.swap_centroids(partial)
        for arch, oid in before_other.items():
            assert id(semantic_router._sub_centroids[arch]) == oid

    def test_unknown_archetype_raises(self, semantic_router: SemanticRouter, config: AgentSearchConfig):
        dim = config.qi.semantic.embedding_dim
        with pytest.raises(ConfigurationError, match="not in current archetypes"):
            semantic_router.swap_centroids({'nonexistent_archetype': np.zeros((1, dim), dtype=np.float32)})

    def test_dim_mismatch_raises(self, semantic_router: SemanticRouter, config: AgentSearchConfig):
        first_arch = semantic_router.archetypes[0]
        wrong_dim = config.qi.semantic.embedding_dim + 1
        with pytest.raises(ConfigurationError, match="dim mismatch"):
            semantic_router.swap_centroids({first_arch: np.zeros((1, wrong_dim), dtype=np.float32)})

    def test_atomic_reference_replaced(self, semantic_router: SemanticRouter, config: AgentSearchConfig):
        dim = config.qi.semantic.embedding_dim
        original_dict_id = id(semantic_router._sub_centroids)
        new_np = {arch: np.zeros((1, dim), dtype=np.float32) for arch in semantic_router.archetypes}
        semantic_router.swap_centroids(new_np)
        assert id(semantic_router._sub_centroids) != original_dict_id


# ---------------------------------------------------------------------------
# CentroidRetrainerDriverConfig
# ---------------------------------------------------------------------------

class TestDriverConfig:
    def test_valid_construction(self):
        cfg = CentroidRetrainerDriverConfig(enabled=True, interval_seconds=600.0, shadow_query_limit=100, max_consecutive_failures=3, positive_signal_types=['result_click'])
        assert cfg.enabled is True
        assert cfg.interval_seconds == pytest.approx(600.0)
        assert cfg.shadow_query_limit == 100
        assert cfg.max_consecutive_failures == 3

    def test_interval_below_floor_raises(self):
        with pytest.raises(ConfigurationError, match="interval_seconds must be >= 10.0"):
            CentroidRetrainerDriverConfig(enabled=True, interval_seconds=5.0, shadow_query_limit=10, max_consecutive_failures=1, positive_signal_types=['result_click'])

    def test_shadow_limit_below_floor_raises(self):
        with pytest.raises(ConfigurationError, match="shadow_query_limit must be >= 1"):
            CentroidRetrainerDriverConfig(enabled=True, interval_seconds=60.0, shadow_query_limit=0, max_consecutive_failures=1, positive_signal_types=['result_click'])

    def test_max_failures_below_floor_raises(self):
        with pytest.raises(ConfigurationError, match="max_consecutive_failures must be >= 1"):
            CentroidRetrainerDriverConfig(enabled=True, interval_seconds=60.0, shadow_query_limit=10, max_consecutive_failures=0, positive_signal_types=['result_click'])

    def test_empty_signal_types_raises(self):
        with pytest.raises(ConfigurationError, match="positive_signal_types must be non-empty"):
            CentroidRetrainerDriverConfig(enabled=True, interval_seconds=60.0, shadow_query_limit=10, max_consecutive_failures=1, positive_signal_types=[])

    def test_from_dict_round_trip(self):
        d = {'enabled': True, 'interval_seconds': 120.0, 'shadow_query_limit': 50, 'max_consecutive_failures': 2, 'positive_signal_types': ['result_click', 'calibration_label']}
        cfg = CentroidRetrainerDriverConfig.from_dict(d)
        assert cfg.interval_seconds == pytest.approx(120.0)
        assert cfg.positive_signal_types == ['result_click', 'calibration_label']

    def test_from_dict_missing_field_raises(self):
        with pytest.raises(ConfigurationError):
            CentroidRetrainerDriverConfig.from_dict({'enabled': True})


# ---------------------------------------------------------------------------
# CentroidRetrainerCycleSummary
# ---------------------------------------------------------------------------

class TestCycleSummary:
    def test_valid_construction(self):
        s = CentroidRetrainerCycleSummary(cycle_id='crd_abc', ran_at=0.0, skipped=False, skip_reason=None, candidate_id='cid_1', verdict='promote', shadow_agreement_rate=0.9, promoted=True, error=None)
        assert s.promoted is True
        assert s.verdict == 'promote'

    def test_empty_cycle_id_raises(self):
        with pytest.raises(ValidationError, match="cycle_id must be non-empty"):
            CentroidRetrainerCycleSummary(cycle_id='', ran_at=0.0, skipped=True, skip_reason=None, candidate_id=None, verdict=None, shadow_agreement_rate=None, promoted=False, error=None)

    def test_invalid_verdict_raises(self):
        with pytest.raises(ValidationError, match="verdict invalid"):
            CentroidRetrainerCycleSummary(cycle_id='crd_x', ran_at=0.0, skipped=False, skip_reason=None, candidate_id=None, verdict='unknown_verdict', shadow_agreement_rate=None, promoted=False, error=None)

    @pytest.mark.parametrize("rate", [-0.01, 1.01])
    def test_agreement_out_of_range_raises(self, rate: float):
        with pytest.raises(ValidationError, match="shadow_agreement_rate out of"):
            CentroidRetrainerCycleSummary(cycle_id='crd_x', ran_at=0.0, skipped=False, skip_reason=None, candidate_id=None, verdict='reject', shadow_agreement_rate=rate, promoted=False, error=None)


# ---------------------------------------------------------------------------
# CentroidRetrainerDriver construction
# ---------------------------------------------------------------------------

class TestDriverConstruction:
    def test_valid_construction(self, driver: CentroidRetrainerDriver):
        assert not driver.is_running
        assert driver.cycle_history == []

    def test_none_config_raises(self, retrainer: CentroidRetrainer, semantic_router: SemanticRouter, signal_store: SignalStore):
        with pytest.raises(ValidationError, match="requires config"):
            CentroidRetrainerDriver(config=None, retrainer=retrainer, router=semantic_router, signal_store=signal_store)


# ---------------------------------------------------------------------------
# CentroidRetrainerDriver.run_cycle
# ---------------------------------------------------------------------------

class TestRunCycle:
    async def test_skip_when_no_positives(self, driver: CentroidRetrainerDriver):
        summary = await driver.run_cycle()
        assert summary.skipped is True
        assert summary.skip_reason == 'no_labeled_positives'
        assert summary.promoted is False

    async def test_skip_when_build_candidate_rejects(self, driver: CentroidRetrainerDriver, signal_store: SignalStore, config: AgentSearchConfig):
        sig = FeedbackSignal(signal_id=FeedbackSignal.new_signal_id(), request_id='r1', signal_type='result_click', payload={'query_type': 'hybrid', 'query_text': 'only one'}, signal_origin='orchestrator')
        signal_store.record(sig)
        summary = await driver.run_cycle()
        assert summary.skipped is True
        assert 'build_candidate_rejected' in (summary.skip_reason or '')

    async def test_shadow_only_verdict(self, retrainer: CentroidRetrainer, semantic_router: SemanticRouter, signal_store: SignalStore, driver_config: CentroidRetrainerDriverConfig):
        stub_verdict = CentroidRetrainVerdict(candidate_id='cid_1', verdict='shadow_only', shadow_agreement_rate=0.3, min_sample_per_archetype=5, reasons=['below_threshold'])
        stub_candidate = CentroidRetrainCandidate(candidate_id='cid_1', new_centroids={'hybrid': [0.0] * retrainer._encoder.dim}, sample_counts={'hybrid': 5}, window_seconds=86400.0)
        mock_retrainer = MagicMock(spec=CentroidRetrainer)
        mock_retrainer.config = retrainer.config
        mock_retrainer.build_candidate.return_value = stub_candidate
        mock_retrainer.decide.return_value = stub_verdict
        fresh_store = SignalStore(signal_store._config)
        for _ in range(3):
            fresh_store.record(FeedbackSignal(signal_id=FeedbackSignal.new_signal_id(), request_id='r', signal_type='result_click', payload={'query_type': 'hybrid', 'query_text': f'domain {_}'}, signal_origin='orchestrator'))
        d = CentroidRetrainerDriver(config=driver_config, retrainer=mock_retrainer, router=semantic_router, signal_store=fresh_store)
        summary = await d.run_cycle()
        assert summary.skipped is False
        assert summary.verdict == 'shadow_only'
        assert summary.promoted is False

    async def test_promote_calls_swap_centroids(self, retrainer: CentroidRetrainer, semantic_router: SemanticRouter, signal_store: SignalStore, driver_config: CentroidRetrainerDriverConfig, config: AgentSearchConfig):
        dim = config.qi.semantic.embedding_dim
        stub_centroid = [float(i % 10) / 10.0 for i in range(dim)]
        stub_verdict = CentroidRetrainVerdict(candidate_id='cid_2', verdict='promote', shadow_agreement_rate=0.95, min_sample_per_archetype=5, reasons=['promoted'])
        stub_candidate = CentroidRetrainCandidate(candidate_id='cid_2', new_centroids={'hybrid': stub_centroid}, sample_counts={'hybrid': 5}, window_seconds=86400.0)
        mock_retrainer = MagicMock(spec=CentroidRetrainer)
        mock_retrainer.config = retrainer.config
        mock_retrainer.build_candidate.return_value = stub_candidate
        mock_retrainer.decide.return_value = stub_verdict
        fresh_store = SignalStore(signal_store._config)
        for _ in range(3):
            fresh_store.record(FeedbackSignal(signal_id=FeedbackSignal.new_signal_id(), request_id='r', signal_type='result_click', payload={'query_type': 'hybrid', 'query_text': f'domain {_}'}, signal_origin='orchestrator'))
        d = CentroidRetrainerDriver(config=driver_config, retrainer=mock_retrainer, router=semantic_router, signal_store=fresh_store)
        before = dict(semantic_router._sub_centroids)
        summary = await d.run_cycle()
        assert summary.promoted is True
        assert summary.verdict == 'promote'
        assert id(semantic_router._sub_centroids['hybrid']) != id(before['hybrid'])

    async def test_emits_verdict_signal(self, retrainer: CentroidRetrainer, semantic_router: SemanticRouter, signal_store: SignalStore, driver_config: CentroidRetrainerDriverConfig):
        stub_verdict = CentroidRetrainVerdict(candidate_id='cid_3', verdict='reject', shadow_agreement_rate=0.0, min_sample_per_archetype=0, reasons=['empty_shadow_set'])
        stub_candidate = CentroidRetrainCandidate(candidate_id='cid_3', new_centroids={'hybrid': [0.0] * retrainer._encoder.dim}, sample_counts={'hybrid': 5}, window_seconds=86400.0)
        mock_retrainer = MagicMock(spec=CentroidRetrainer)
        mock_retrainer.config = retrainer.config
        mock_retrainer.build_candidate.return_value = stub_candidate
        mock_retrainer.decide.return_value = stub_verdict
        fresh_store = SignalStore(signal_store._config)
        for _ in range(3):
            fresh_store.record(FeedbackSignal(signal_id=FeedbackSignal.new_signal_id(), request_id='r', signal_type='result_click', payload={'query_type': 'hybrid', 'query_text': f'domain {_}'}, signal_origin='orchestrator'))
        d = CentroidRetrainerDriver(config=driver_config, retrainer=mock_retrainer, router=semantic_router, signal_store=fresh_store)
        await d.run_cycle()
        await asyncio.sleep(0.1)
        types = {s.signal_type for s in fresh_store.recent(200)}
        assert 'centroid_retrain_verdict' in types

    async def test_emits_promoted_signal_on_promote(self, retrainer: CentroidRetrainer, semantic_router: SemanticRouter, signal_store: SignalStore, driver_config: CentroidRetrainerDriverConfig, config: AgentSearchConfig):
        dim = config.qi.semantic.embedding_dim
        stub_centroid = [0.0] * dim
        stub_verdict = CentroidRetrainVerdict(candidate_id='cid_4', verdict='promote', shadow_agreement_rate=0.98, min_sample_per_archetype=5, reasons=['promoted'])
        stub_candidate = CentroidRetrainCandidate(candidate_id='cid_4', new_centroids={'hybrid': stub_centroid}, sample_counts={'hybrid': 5}, window_seconds=86400.0)
        mock_retrainer = MagicMock(spec=CentroidRetrainer)
        mock_retrainer.config = retrainer.config
        mock_retrainer.build_candidate.return_value = stub_candidate
        mock_retrainer.decide.return_value = stub_verdict
        fresh_store = SignalStore(signal_store._config)
        for _ in range(3):
            fresh_store.record(FeedbackSignal(signal_id=FeedbackSignal.new_signal_id(), request_id='r', signal_type='result_click', payload={'query_type': 'hybrid', 'query_text': f'domain {_}'}, signal_origin='orchestrator'))
        d = CentroidRetrainerDriver(config=driver_config, retrainer=mock_retrainer, router=semantic_router, signal_store=fresh_store)
        await d.run_cycle()
        await asyncio.sleep(0.1)
        types = {s.signal_type for s in fresh_store.recent(200)}
        assert 'centroid_retrain_promoted' in types

    async def test_error_captured_in_summary(self, retrainer: CentroidRetrainer, semantic_router: SemanticRouter, signal_store: SignalStore, driver_config: CentroidRetrainerDriverConfig):
        mock_retrainer = MagicMock(spec=CentroidRetrainer)
        mock_retrainer.config = retrainer.config
        mock_retrainer.build_candidate.side_effect = RuntimeError("unexpected")
        fresh_store = SignalStore(signal_store._config)
        for _ in range(3):
            fresh_store.record(FeedbackSignal(signal_id=FeedbackSignal.new_signal_id(), request_id='r', signal_type='result_click', payload={'query_type': 'hybrid', 'query_text': f'domain {_}'}, signal_origin='orchestrator'))
        d = CentroidRetrainerDriver(config=driver_config, retrainer=mock_retrainer, router=semantic_router, signal_store=fresh_store)
        summary = await d.run_cycle()
        assert summary.error is not None
        assert 'RuntimeError' in summary.error


# ---------------------------------------------------------------------------
# CentroidRetrainerDriver lifecycle
# ---------------------------------------------------------------------------

class TestDriverLifecycle:
    async def test_start_stop_idempotent(self, driver: CentroidRetrainerDriver):
        await driver.start()
        assert driver.is_running
        await driver.start()
        assert driver.is_running
        await driver.stop()
        assert not driver.is_running
        await driver.stop()
        assert not driver.is_running

    async def test_is_running_reflects_state(self, driver_config: CentroidRetrainerDriverConfig, retrainer: CentroidRetrainer, semantic_router: SemanticRouter, signal_store: SignalStore):
        d = CentroidRetrainerDriver(config=driver_config, retrainer=retrainer, router=semantic_router, signal_store=signal_store)
        assert not d.is_running
        await d.start()
        assert d.is_running
        await d.stop()
        assert not d.is_running
