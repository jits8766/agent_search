"""Tests for the QI hard-negative miner.

Coverage matrix (per ``testing.mdc`` §7):

``HardNegativeMinerConfig.__post_init__``:
- threshold_out_of_range_raises          -> TestHardNegativeMinerConfig::test_threshold_out_of_range_raises
- num_sub_centroids_below_one_raises     -> TestHardNegativeMinerConfig::test_num_sub_centroids_below_one_raises
- min_seeds_below_one_raises             -> TestHardNegativeMinerConfig::test_min_seeds_below_one_raises

``HardNegativeMinerConfig.from_dict``:
- missing_key_raises                     -> TestHardNegativeMinerConfig::test_missing_key_raises
- valid_dict_round_trips                 -> TestHardNegativeMinerConfig::test_valid_dict_round_trips

``HardNegativeRow.__post_init__``:
- empty_query_raises                     -> TestHardNegativeRow::test_empty_query_raises
- bad_archetype_raises                   -> TestHardNegativeRow::test_bad_archetype_raises
- score_out_of_range_raises              -> TestHardNegativeRow::test_score_out_of_range_raises
- self_in_hard_negative_for_raises       -> TestHardNegativeRow::test_self_in_hard_negative_for_raises
- to_dict_round_trips                    -> TestHardNegativeRow::test_to_dict_round_trips

``HardNegativeMiner.__init__``:
- non_encoder_raises                     -> TestHardNegativeMinerInit::test_non_encoder_raises
- below_floor_raises                     -> TestHardNegativeMinerInit::test_below_floor_raises

``HardNegativeMiner.mine``:
- one_row_per_seed                       -> TestHardNegativeMinerMine::test_one_row_per_seed
- scores_cover_every_archetype           -> TestHardNegativeMinerMine::test_scores_cover_every_archetype
- never_self_hard_negative               -> TestHardNegativeMinerMine::test_never_self_hard_negative
- threshold_one_emits_no_hard_negatives  -> TestHardNegativeMinerMine::test_threshold_one_emits_no_hard_negatives
- threshold_low_flags_confusables        -> TestHardNegativeMinerMine::test_threshold_low_flags_confusables
- deterministic_with_pinned_seed         -> TestHardNegativeMinerMine::test_deterministic_with_pinned_seed
- header_counts_match_dataset            -> TestHardNegativeMinerMine::test_header_counts_match_dataset

``write_artifact_jsonl`` / ``read_artifact_jsonl``:
- round_trip_preserves_rows              -> TestArtifactJsonl::test_round_trip_preserves_rows
- header_first_line                      -> TestArtifactJsonl::test_header_first_line
"""
import json
from pathlib import Path
from typing import Dict, List

import pytest

from semantic_search.contracts import RouterSeed, RouterSeedDataset
from semantic_search.core.exceptions import ValidationError
from semantic_search.qi.encoder import HashingEncoder
from semantic_search.qi.training.data_schema import HardNegativeArtifact, HardNegativeMinerConfig, HardNegativeRow
from semantic_search.qi.training.hard_negative_miner import HardNegativeMiner, read_artifact_jsonl, write_artifact_jsonl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_dataset(seed_count_per_arch: int = 4) -> RouterSeedDataset:
    """Build a 4-archetype dataset with deterministic, archetype-flavoured seeds.

    Phrasing is chosen so an L2-normalised hashing encoder gives each
    archetype a distinguishable centroid: all seeds within an archetype
    share a stem keyword, ensuring the K-means topology has signal.
    """
    seeds_by_archetype: Dict[str, List[RouterSeed]] = {}
    flavours = {
        'hybrid': 'expiring com domain under',
        'guidance': 'how do i pick a',
        'explore': 'trending top showcase',
        'analytics': 'how many auctions sold',
    }
    for archetype, stem in flavours.items():
        seeds_by_archetype[archetype] = [
            RouterSeed(query=f"{stem} variant {i}", archetype=archetype, origin='manual', source_id='unit')
            for i in range(seed_count_per_arch)
        ]
    return RouterSeedDataset(seeds_by_archetype=seeds_by_archetype, source_path='<test-fixture>')


def _miner_config(threshold: float = 0.65, k: int = 1, encoder_seed: int = 13, floor: int = 1) -> HardNegativeMinerConfig:
    return HardNegativeMinerConfig(
        threshold=threshold,
        num_sub_centroids=k,
        encoder_seed=encoder_seed,
        min_seeds_per_archetype=floor,
    )


# ---------------------------------------------------------------------------
# HardNegativeMinerConfig
# ---------------------------------------------------------------------------

class TestHardNegativeMinerConfig:

    def test_threshold_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            HardNegativeMinerConfig(threshold=0.0, num_sub_centroids=1, encoder_seed=0, min_seeds_per_archetype=1)
        with pytest.raises(ValidationError):
            HardNegativeMinerConfig(threshold=1.5, num_sub_centroids=1, encoder_seed=0, min_seeds_per_archetype=1)

    def test_num_sub_centroids_below_one_raises(self):
        with pytest.raises(ValidationError):
            HardNegativeMinerConfig(threshold=0.5, num_sub_centroids=0, encoder_seed=0, min_seeds_per_archetype=1)

    def test_min_seeds_below_one_raises(self):
        with pytest.raises(ValidationError):
            HardNegativeMinerConfig(threshold=0.5, num_sub_centroids=1, encoder_seed=0, min_seeds_per_archetype=0)

    def test_missing_key_raises(self):
        with pytest.raises(ValidationError):
            HardNegativeMinerConfig.from_dict({'threshold': 0.5, 'num_sub_centroids': 1})

    def test_valid_dict_round_trips(self):
        d = {'threshold': 0.65, 'num_sub_centroids': 2, 'encoder_seed': 13, 'min_seeds_per_archetype': 30}
        cfg = HardNegativeMinerConfig.from_dict(d)
        assert cfg.threshold == 0.65
        assert cfg.num_sub_centroids == 2
        assert cfg.encoder_seed == 13
        assert cfg.min_seeds_per_archetype == 30


# ---------------------------------------------------------------------------
# HardNegativeRow
# ---------------------------------------------------------------------------

class TestHardNegativeRow:

    def test_empty_query_raises(self):
        with pytest.raises(ValidationError):
            HardNegativeRow(
                query='   ', archetype='hybrid', label='hybrid',
                scores={'hybrid': 0.9, 'guidance': 0.1, 'explore': 0.1, 'analytics': 0.1},
                is_hard_negative_for=[],
            )

    def test_bad_archetype_raises(self):
        with pytest.raises(ValidationError):
            HardNegativeRow(
                query='x', archetype='filter', label='filter',
                scores={'hybrid': 0.5, 'guidance': 0.5, 'explore': 0.5, 'analytics': 0.5},
                is_hard_negative_for=[],
            )

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            HardNegativeRow(
                query='x', archetype='hybrid', label='hybrid',
                scores={'hybrid': 1.5, 'guidance': 0.1, 'explore': 0.1, 'analytics': 0.1},
                is_hard_negative_for=[],
            )

    def test_self_in_hard_negative_for_raises(self):
        with pytest.raises(ValidationError):
            HardNegativeRow(
                query='x', archetype='hybrid', label='hybrid',
                scores={'hybrid': 0.9, 'guidance': 0.1, 'explore': 0.1, 'analytics': 0.1},
                is_hard_negative_for=['hybrid'],
            )

    def test_to_dict_round_trips(self):
        row = HardNegativeRow(
            query='x', archetype='hybrid', label='hybrid',
            scores={'hybrid': 0.9, 'guidance': 0.7, 'explore': 0.1, 'analytics': 0.1},
            is_hard_negative_for=['guidance'],
        )
        d = row.to_dict()
        recovered = HardNegativeRow.from_dict(d)
        assert recovered == row


# ---------------------------------------------------------------------------
# HardNegativeMiner.__init__
# ---------------------------------------------------------------------------

class TestHardNegativeMinerInit:

    def test_non_encoder_raises(self):
        ds = _build_dataset()
        with pytest.raises(ValidationError):
            HardNegativeMiner(config=_miner_config(), encoder='not an encoder', dataset=ds)  # type: ignore[arg-type]

    def test_below_floor_raises(self):
        # Dataset has 4 seeds per archetype; floor of 5 must trip the check.
        ds = _build_dataset(seed_count_per_arch=4)
        encoder = HashingEncoder(dim=32, seed=0)
        cfg = _miner_config(floor=5)
        with pytest.raises(ValidationError, match="min required=5"):
            HardNegativeMiner(config=cfg, encoder=encoder, dataset=ds)


# ---------------------------------------------------------------------------
# HardNegativeMiner.mine
# ---------------------------------------------------------------------------

class TestHardNegativeMinerMine:

    def test_one_row_per_seed(self):
        ds = _build_dataset(seed_count_per_arch=4)
        encoder = HashingEncoder(dim=64, seed=7)
        miner = HardNegativeMiner(config=_miner_config(), encoder=encoder, dataset=ds)
        artifact = miner.mine()
        assert artifact.total_rows() == 4 * 4  # 4 archetypes * 4 seeds

    def test_scores_cover_every_archetype(self):
        ds = _build_dataset()
        encoder = HashingEncoder(dim=64, seed=7)
        miner = HardNegativeMiner(config=_miner_config(), encoder=encoder, dataset=ds)
        artifact = miner.mine()
        for row in artifact.rows:
            assert set(row.scores.keys()) == {'hybrid', 'guidance', 'explore', 'analytics'}

    def test_never_self_hard_negative(self):
        ds = _build_dataset()
        encoder = HashingEncoder(dim=64, seed=7)
        # Drive threshold to 0 so EVERY archetype would qualify as a
        # confusable; the miner must still filter out the row's own.
        miner = HardNegativeMiner(config=_miner_config(threshold=0.001), encoder=encoder, dataset=ds)
        artifact = miner.mine()
        for row in artifact.rows:
            assert row.archetype not in row.is_hard_negative_for

    def test_threshold_one_emits_no_hard_negatives(self):
        ds = _build_dataset()
        encoder = HashingEncoder(dim=64, seed=7)
        # threshold=1.0 means no wrong-archetype score can ever exceed it.
        miner = HardNegativeMiner(config=_miner_config(threshold=1.0), encoder=encoder, dataset=ds)
        artifact = miner.mine()
        assert artifact.total_hard_negatives() == 0

    def test_threshold_low_flags_confusables(self):
        ds = _build_dataset()
        encoder = HashingEncoder(dim=64, seed=7)
        # threshold=0.001 means almost any non-trivial cosine flags a hard neg.
        miner = HardNegativeMiner(config=_miner_config(threshold=0.001), encoder=encoder, dataset=ds)
        artifact = miner.mine()
        # With 16 rows and 4 archetypes, every row could in principle have
        # 3 hard negatives (one per other archetype). The dataset is
        # distinguishable so we expect at least *some* confusable flags.
        assert artifact.total_hard_negatives() > 0

    def test_deterministic_with_pinned_seed(self):
        ds = _build_dataset()
        encoder1 = HashingEncoder(dim=64, seed=11)
        encoder2 = HashingEncoder(dim=64, seed=11)
        cfg = _miner_config(encoder_seed=11)
        a1 = HardNegativeMiner(config=cfg, encoder=encoder1, dataset=ds).mine()
        a2 = HardNegativeMiner(config=cfg, encoder=encoder2, dataset=ds).mine()
        assert [r.to_dict() for r in a1.rows] == [r.to_dict() for r in a2.rows]
        assert a1.hard_negative_counts == a2.hard_negative_counts
        assert a1.counts_per_archetype == a2.counts_per_archetype

    def test_header_counts_match_dataset(self):
        ds = _build_dataset(seed_count_per_arch=3)
        encoder = HashingEncoder(dim=64, seed=7)
        miner = HardNegativeMiner(config=_miner_config(), encoder=encoder, dataset=ds)
        artifact = miner.mine()
        assert artifact.counts_per_archetype == {'hybrid': 3, 'guidance': 3, 'explore': 3, 'analytics': 3}
        assert sorted(artifact.archetypes) == ['analytics', 'explore', 'guidance', 'hybrid']
        assert artifact.encoder_dim == 64


# ---------------------------------------------------------------------------
# write_artifact_jsonl / read_artifact_jsonl
# ---------------------------------------------------------------------------

class TestArtifactJsonl:

    def test_round_trip_preserves_rows(self, tmp_path: Path):
        ds = _build_dataset()
        encoder = HashingEncoder(dim=32, seed=0)
        miner = HardNegativeMiner(config=_miner_config(), encoder=encoder, dataset=ds)
        artifact = miner.mine()
        out_path = tmp_path / 'sub' / 'hard_negatives.jsonl'
        write_artifact_jsonl(artifact, out_path)
        assert out_path.exists()
        header, rows = read_artifact_jsonl(out_path)
        assert len(rows) == artifact.total_rows()
        # Every row round-trips through to_dict / from_dict identically.
        for original, recovered in zip(artifact.rows, rows):
            assert original == recovered
        # Header preserves run-time provenance.
        assert header['_kind'] == 'hard_negative_artifact_header'
        assert header['threshold'] == artifact.threshold
        assert header['encoder_dim'] == artifact.encoder_dim
        assert header['archetypes'] == artifact.archetypes

    def test_header_first_line(self, tmp_path: Path):
        ds = _build_dataset()
        encoder = HashingEncoder(dim=32, seed=0)
        miner = HardNegativeMiner(config=_miner_config(), encoder=encoder, dataset=ds)
        artifact = miner.mine()
        out_path = tmp_path / 'hard_negatives.jsonl'
        write_artifact_jsonl(artifact, out_path)
        with open(out_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
        first = json.loads(first_line)
        assert first.get('_kind') == 'hard_negative_artifact_header'


# ---------------------------------------------------------------------------
# HardNegativeArtifact validation
# ---------------------------------------------------------------------------

class TestHardNegativeArtifactValidation:

    def test_bad_archetype_in_archetypes_raises(self):
        with pytest.raises(ValidationError):
            HardNegativeArtifact(
                rows=[],
                threshold=0.5,
                num_sub_centroids=1,
                encoder_seed=0,
                encoder_dim=32,
                archetypes=['filter'],  # 'filter' is not a valid archetype
                counts_per_archetype={'hybrid': 1},
                hard_negative_counts={},
                source_path='<x>',
            )

    def test_negative_count_raises(self):
        with pytest.raises(ValidationError):
            HardNegativeArtifact(
                rows=[],
                threshold=0.5,
                num_sub_centroids=1,
                encoder_seed=0,
                encoder_dim=32,
                archetypes=['hybrid'],
                counts_per_archetype={'hybrid': -1},
                hard_negative_counts={},
                source_path='<x>',
            )
