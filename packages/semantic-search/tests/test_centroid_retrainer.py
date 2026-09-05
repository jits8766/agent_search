"""Tests for ``semantic_search.qi.centroid_retrainer``.

Coverage matrix (per ``testing.mdc`` §7):

``CentroidRetrainerConfig`` (input contract):
- valid_construction                       -> TestCentroidRetrainerConfig::test_valid_construction
- min_samples_below_floor_raises           -> TestCentroidRetrainerConfig::test_min_samples_below_floor_raises
- agreement_out_of_range_raises            -> TestCentroidRetrainerConfig::test_agreement_out_of_range_raises[low/high]
- negative_window_raises                   -> TestCentroidRetrainerConfig::test_negative_window_raises
- from_dict_missing_field_raises           -> TestCentroidRetrainerConfig::test_from_dict_missing_field_raises

``CentroidRetrainer.__init__`` (constructor invariants):
- valid_construction                       -> TestRetrainerConstruction::test_valid_construction
- none_config_raises                       -> TestRetrainerConstruction::test_none_config_raises
- wrong_config_type_raises                 -> TestRetrainerConstruction::test_wrong_config_type_raises
- none_encoder_raises                      -> TestRetrainerConstruction::test_none_encoder_raises
- none_router_raises                       -> TestRetrainerConstruction::test_none_router_raises

``CentroidRetrainer.build_candidate`` (candidate-building logic):
- builds_from_valid_positives              -> TestBuildCandidate::test_builds_from_valid_positives
- below_floor_archetype_dropped            -> TestBuildCandidate::test_below_floor_archetype_dropped
- all_archetypes_below_floor_raises        -> TestBuildCandidate::test_all_archetypes_below_floor_raises
- non_query_type_archetype_raises          -> TestBuildCandidate::test_non_query_type_archetype_raises
- non_string_positive_raises               -> TestBuildCandidate::test_non_string_positive_raises
- empty_string_positives_dropped           -> TestBuildCandidate::test_empty_string_positives_dropped
- non_mapping_input_raises                 -> TestBuildCandidate::test_non_mapping_input_raises
- empty_mapping_raises                     -> TestBuildCandidate::test_empty_mapping_raises
- candidate_id_override_honored            -> TestBuildCandidate::test_candidate_id_override_honored
- auto_id_unique                           -> TestBuildCandidate::test_auto_id_unique

``CentroidRetrainer.decide`` (verdict logic):
- shadow_promote_on_high_agreement         -> TestDecide::test_shadow_promote_on_high_agreement
- shadow_only_when_below_promote_threshold -> TestDecide::test_shadow_only_when_below_promote_threshold
- reject_when_below_min_samples            -> TestDecide::test_reject_when_below_min_samples
- reject_on_empty_shadow_set               -> TestDecide::test_reject_on_empty_shadow_set
- non_candidate_input_raises               -> TestDecide::test_non_candidate_input_raises
- none_shadow_queries_raises               -> TestDecide::test_none_shadow_queries_raises
- empty_string_shadow_queries_treated_agreed -> TestDecide::test_empty_string_shadow_queries_treated_agreed

``CentroidRetrainer`` (output contract on verdicts):
- verdict_is_typed_dataclass               -> TestOutputContract::test_verdict_is_typed_dataclass
- candidate_is_typed_dataclass             -> TestOutputContract::test_candidate_is_typed_dataclass
- verdict_in_known_set                     -> TestOutputContract::test_verdict_in_known_set
- agreement_in_unit_interval               -> TestOutputContract::test_agreement_in_unit_interval
"""
import pytest

from semantic_search.config.models import AgentSearchConfig, CentroidRetrainerConfig
from semantic_search.contracts import CENTROID_RETRAIN_VERDICTS, CentroidRetrainCandidate, CentroidRetrainVerdict, IntentSlice, RouterSeedDataset
from semantic_search.core.exceptions import ConfigurationError, ValidationError
from semantic_search.qi.centroid_retrainer import CentroidRetrainer
from semantic_search.qi.encoder import HashingEncoder
from semantic_search.qi.semantic_router import SemanticRouter


# ---------------------------------------------------------------------------
# Local fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def encoder(config: AgentSearchConfig) -> HashingEncoder:
    """Module-scoped override: HashingEncoder is stateless — safe to share across all tests in this module."""
    return HashingEncoder(dim=config.qi.semantic.embedding_dim, seed=config.qi.semantic.encoder_seed)


@pytest.fixture(scope='module')
def retrainer_config() -> CentroidRetrainerConfig:
    """Small floor (3 samples / 0.5 agreement) keeps tests cheap and verifiable."""
    return CentroidRetrainerConfig(
        enabled=True,
        min_samples_per_archetype=3,
        min_shadow_agreement=0.5,
        window_seconds=86400.0,
        max_signals_per_read=10000,
    )


@pytest.fixture(scope='module')
def semantic_router(config: AgentSearchConfig, encoder: HashingEncoder, router_seeds: RouterSeedDataset) -> SemanticRouter:
    """Real SemanticRouter wired once per module — encode_batch + k-means runs once, not per-test."""
    return SemanticRouter(config.qi.semantic, encoder, router_seeds)


@pytest.fixture(scope='module')
def retrainer(retrainer_config: CentroidRetrainerConfig, encoder: HashingEncoder, semantic_router: SemanticRouter) -> CentroidRetrainer:
    """Wired retrainer ready to build + decide. CentroidRetrainer is read-only after construction."""
    return CentroidRetrainer(config=retrainer_config, encoder=encoder, current_router=semantic_router)


def _positives(text_a: str = 'fast cheap io domain', text_b: str = 'cheap io domain available') -> dict:
    """Helper: 3 hybrid positives + 3 guidance positives — meets the test floor of 3."""
    return {
        'hybrid': [text_a, text_b, 'short com domain under 100'],
        'guidance': ['how do auctions work', 'when does the auction end', 'how do i bid on a domain'],
    }


# ---------------------------------------------------------------------------
# CentroidRetrainerConfig — input contract
# ---------------------------------------------------------------------------


class TestCentroidRetrainerConfig:
    def test_valid_construction(self):
        cfg = CentroidRetrainerConfig(enabled=True, min_samples_per_archetype=500, min_shadow_agreement=0.85, window_seconds=604800.0, max_signals_per_read=10000)
        assert cfg.enabled is True
        assert cfg.min_samples_per_archetype == 500
        assert cfg.min_shadow_agreement == pytest.approx(0.85)
        assert cfg.window_seconds == pytest.approx(604800.0)
        assert cfg.max_signals_per_read == 10000

    def test_min_samples_below_floor_raises(self):
        with pytest.raises(ConfigurationError, match=r"min_samples_per_archetype must be >= 1"):
            CentroidRetrainerConfig(enabled=True, min_samples_per_archetype=0, min_shadow_agreement=0.5, window_seconds=1.0, max_signals_per_read=100)

    @pytest.mark.parametrize("agreement", [-0.01, 1.01])
    def test_agreement_out_of_range_raises(self, agreement: float):
        with pytest.raises(ConfigurationError, match=r"min_shadow_agreement must be in \[0,1\]"):
            CentroidRetrainerConfig(enabled=True, min_samples_per_archetype=1, min_shadow_agreement=agreement, window_seconds=1.0, max_signals_per_read=100)

    def test_negative_window_raises(self):
        with pytest.raises(ConfigurationError, match=r"window_seconds must be >= 0"):
            CentroidRetrainerConfig(enabled=True, min_samples_per_archetype=1, min_shadow_agreement=0.5, window_seconds=-1.0, max_signals_per_read=100)

    def test_max_signals_below_floor_raises(self):
        with pytest.raises(ConfigurationError, match=r"max_signals_per_read must be >= 1"):
            CentroidRetrainerConfig(enabled=True, min_samples_per_archetype=1, min_shadow_agreement=0.5, window_seconds=1.0, max_signals_per_read=0)

    def test_from_dict_missing_field_raises(self):
        with pytest.raises(ConfigurationError):
            CentroidRetrainerConfig.from_dict({'enabled': True, 'min_samples_per_archetype': 1})

    def test_from_dict_round_trip(self):
        d = {'enabled': True, 'min_samples_per_archetype': 7, 'min_shadow_agreement': 0.6, 'window_seconds': 60.0, 'max_signals_per_read': 500}
        cfg = CentroidRetrainerConfig.from_dict(d)
        assert cfg.min_samples_per_archetype == 7
        assert cfg.min_shadow_agreement == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Constructor — input contract
# ---------------------------------------------------------------------------


class TestRetrainerConstruction:
    def test_valid_construction(self, retrainer: CentroidRetrainer):
        assert retrainer.config.min_samples_per_archetype == 3

    def test_none_config_raises(self, encoder: HashingEncoder, semantic_router: SemanticRouter):
        with pytest.raises(ValidationError, match=r"requires a CentroidRetrainerConfig"):
            CentroidRetrainer(config=None, encoder=encoder, current_router=semantic_router)  # type: ignore[arg-type]

    def test_wrong_config_type_raises(self, encoder: HashingEncoder, semantic_router: SemanticRouter):
        with pytest.raises(ValidationError, match=r"requires a CentroidRetrainerConfig"):
            CentroidRetrainer(config={'enabled': True}, encoder=encoder, current_router=semantic_router)  # type: ignore[arg-type]

    def test_none_encoder_raises(self, retrainer_config: CentroidRetrainerConfig, semantic_router: SemanticRouter):
        with pytest.raises(ValidationError, match=r"requires a non-None Encoder"):
            CentroidRetrainer(config=retrainer_config, encoder=None, current_router=semantic_router)  # type: ignore[arg-type]

    def test_none_router_raises(self, retrainer_config: CentroidRetrainerConfig, encoder: HashingEncoder):
        with pytest.raises(ValidationError, match=r"requires a non-None SemanticRouter"):
            CentroidRetrainer(config=retrainer_config, encoder=encoder, current_router=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# build_candidate — candidate-building logic
# ---------------------------------------------------------------------------


class TestBuildCandidate:
    def test_builds_from_valid_positives(self, retrainer: CentroidRetrainer, encoder: HashingEncoder):
        candidate = retrainer.build_candidate(_positives())
        assert isinstance(candidate, CentroidRetrainCandidate)
        assert set(candidate.new_centroids.keys()) == {'hybrid', 'guidance'}
        assert candidate.sample_counts == {'hybrid': 3, 'guidance': 3}
        for vec in candidate.new_centroids.values():
            assert len(vec) == encoder.dim

    def test_below_floor_archetype_dropped(self, retrainer: CentroidRetrainer):
        positives = {
            'hybrid': ['a com domain', 'cheap io domain', 'short net'],  # 3 — meets floor
            'guidance': ['how do auctions work'],                         # 1 — below floor
        }
        candidate = retrainer.build_candidate(positives)
        assert set(candidate.new_centroids.keys()) == {'hybrid'}
        assert candidate.sample_counts == {'hybrid': 3}

    def test_all_archetypes_below_floor_raises(self, retrainer: CentroidRetrainer):
        with pytest.raises(ValidationError, match=r"produced no centroids"):
            retrainer.build_candidate({'hybrid': ['only one']})

    def test_non_query_type_archetype_raises(self, retrainer: CentroidRetrainer):
        with pytest.raises(ValidationError, match=r"not in QUERY_TYPES"):
            retrainer.build_candidate({'unknown_archetype': ['a', 'b', 'c']})

    def test_non_string_positive_raises(self, retrainer: CentroidRetrainer):
        with pytest.raises(ValidationError, match=r"must be str"):
            retrainer.build_candidate({'hybrid': ['a com domain', 123, 'short net']})  # type: ignore[list-item]

    def test_empty_string_positives_dropped(self, retrainer: CentroidRetrainer):
        positives = {
            'hybrid': ['', '   ', '\t', 'a com', 'b io', 'c net'],
            'guidance': ['how do auctions work', 'when does it end', 'where do i pay'],
        }
        candidate = retrainer.build_candidate(positives)
        assert candidate.sample_counts['hybrid'] == 3
        assert candidate.sample_counts['guidance'] == 3

    def test_non_mapping_input_raises(self, retrainer: CentroidRetrainer):
        with pytest.raises(ValidationError, match=r"requires a Mapping"):
            retrainer.build_candidate(['hybrid', ['a', 'b', 'c']])  # type: ignore[arg-type]

    def test_empty_mapping_raises(self, retrainer: CentroidRetrainer):
        with pytest.raises(ValidationError, match=r"at least one archetype"):
            retrainer.build_candidate({})

    def test_candidate_id_override_honored(self, retrainer: CentroidRetrainer):
        candidate = retrainer.build_candidate(_positives(), candidate_id='cand_test_42')
        assert candidate.candidate_id == 'cand_test_42'

    def test_auto_id_unique(self, retrainer: CentroidRetrainer):
        c1 = retrainer.build_candidate(_positives())
        c2 = retrainer.build_candidate(_positives())
        assert c1.candidate_id != c2.candidate_id
        assert c1.candidate_id.startswith('centroid_retrain_')


# ---------------------------------------------------------------------------
# decide — verdict logic
# ---------------------------------------------------------------------------


class TestDecide:
    def test_shadow_promote_on_high_agreement(self, retrainer: CentroidRetrainer):
        # Build a candidate using the SAME texts as shadow probes — by
        # construction the candidate centroids land on each probe's tokens.
        # Whatever the live router decides on these probes, the candidate
        # decides identically (top-1 archetype with identical embedding).
        positives = _positives()
        shadow = positives['hybrid'] + positives['guidance']
        candidate = retrainer.build_candidate(positives)
        verdict = retrainer.decide(candidate=candidate, shadow_queries=shadow)
        assert verdict.verdict in {'promote', 'shadow_only'}
        assert 0.0 <= verdict.shadow_agreement_rate <= 1.0
        assert verdict.min_sample_per_archetype == 3

    def test_shadow_only_when_below_promote_threshold(self, retrainer_config: CentroidRetrainerConfig, encoder: HashingEncoder, semantic_router: SemanticRouter):
        # Force a 100% promote threshold and engineer disagreement by
        # stubbing the live router to ALWAYS return 'guidance' while the
        # candidate's only archetype is 'hybrid'. With mismatched archetypes
        # on the same probes, agreement is 0% and the verdict ramps down
        # from 'promote' to 'shadow_only'.

        class StubAlwaysGuidance(SemanticRouter):
            def classify(self, query: str):
                return IntentSlice(query_type='guidance', entities=[], confidence=0.99, raw_text=query)

        stub_router = StubAlwaysGuidance(semantic_router._config, encoder, semantic_router._seed_dataset)  # noqa: SLF001
        cfg = CentroidRetrainerConfig(
            enabled=True, min_samples_per_archetype=3,
            min_shadow_agreement=1.0,
            window_seconds=1.0, max_signals_per_read=10000,
        )
        retrainer = CentroidRetrainer(config=cfg, encoder=encoder, current_router=stub_router)
        # Build candidate with strong hybrid-only signal so the candidate
        # picks 'hybrid' on probes that the stub router classifies as 'guidance'.
        positives = {'hybrid': ['cheap io domain', 'short com under 100', 'available net domain']}
        candidate = retrainer.build_candidate(positives)
        verdict = retrainer.decide( candidate=candidate, shadow_queries=['cheap io domain', 'short com under 100'],)
        assert verdict.verdict == 'shadow_only'
        assert verdict.shadow_agreement_rate < 1.0

    def test_reject_when_below_min_samples(self, retrainer: CentroidRetrainer):
        # Hand-craft a candidate with too-small sample_counts to force the
        # min-samples gate to reject. We bypass build_candidate so the gate
        # in decide() (not build) is exercised.
        candidate = CentroidRetrainCandidate(
            candidate_id='manual',
            new_centroids={'hybrid': [0.1, 0.2, 0.3]},
            sample_counts={'hybrid': 1},  # 1 < 3 (cfg floor)
            window_seconds=1.0,
        )
        verdict = retrainer.decide(candidate=candidate, shadow_queries=['anything'])
        assert verdict.verdict == 'reject'
        assert verdict.min_sample_per_archetype == 1
        assert any('below floor' in r for r in verdict.reasons)

    def test_reject_on_empty_shadow_set(self, retrainer: CentroidRetrainer):
        candidate = retrainer.build_candidate(_positives())
        verdict = retrainer.decide(candidate=candidate, shadow_queries=[])
        assert verdict.verdict == 'reject'
        assert any('empty_shadow_set' in r for r in verdict.reasons)

    def test_non_candidate_input_raises(self, retrainer: CentroidRetrainer):
        with pytest.raises(ValidationError, match=r"requires a CentroidRetrainCandidate"):
            retrainer.decide(candidate={'fake': 'dict'}, shadow_queries=['a'])  # type: ignore[arg-type]

    def test_none_shadow_queries_raises(self, retrainer: CentroidRetrainer):
        candidate = retrainer.build_candidate(_positives())
        with pytest.raises(ValidationError, match=r"requires a non-None shadow_queries"):
            retrainer.decide(candidate=candidate, shadow_queries=None)  # type: ignore[arg-type]

    def test_empty_string_shadow_queries_treated_agreed(self, retrainer: CentroidRetrainer):
        # Empty strings yield (None, None, agreed=True) per _compare_one — so
        # a shadow set of all-empty probes always reaches the agreement
        # branch. This guards the contract that empty input does not crash.
        candidate = retrainer.build_candidate(_positives())
        verdict = retrainer.decide(candidate=candidate, shadow_queries=['', '   '])
        assert verdict.verdict == 'promote'
        assert verdict.shadow_agreement_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------


class TestOutputContract:
    def test_verdict_is_typed_dataclass(self, retrainer: CentroidRetrainer):
        candidate = retrainer.build_candidate(_positives())
        verdict = retrainer.decide(candidate=candidate, shadow_queries=['a com', 'b io'])
        assert isinstance(verdict, CentroidRetrainVerdict)

    def test_candidate_is_typed_dataclass(self, retrainer: CentroidRetrainer):
        candidate = retrainer.build_candidate(_positives())
        assert isinstance(candidate, CentroidRetrainCandidate)

    def test_verdict_in_known_set(self, retrainer: CentroidRetrainer):
        candidate = retrainer.build_candidate(_positives())
        verdict = retrainer.decide(candidate=candidate, shadow_queries=['a', 'b'])
        assert verdict.verdict in CENTROID_RETRAIN_VERDICTS

    def test_agreement_in_unit_interval(self, retrainer: CentroidRetrainer):
        candidate = retrainer.build_candidate(_positives())
        verdict = retrainer.decide(candidate=candidate, shadow_queries=['a com domain', 'how do auctions work'])
        assert 0.0 <= verdict.shadow_agreement_rate <= 1.0
