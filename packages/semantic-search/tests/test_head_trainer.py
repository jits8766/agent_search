"""Tests for the QI learned head trainer + SemanticRouter dispatch.

Coverage matrix (per ``testing.mdc`` §7):

``HeadTrainerConfig.__post_init__``:
- negative_l2_raises                 -> TestHeadTrainerConfig::test_negative_l2_raises
- non_positive_lr_raises             -> TestHeadTrainerConfig::test_non_positive_lr_raises
- max_epochs_below_one_raises        -> TestHeadTrainerConfig::test_max_epochs_below_one_raises
- holdout_out_of_range_raises        -> TestHeadTrainerConfig::test_holdout_out_of_range_raises

``LearnedHead.__init__``:
- bad_dims_raise                     -> TestLearnedHeadInit::test_bad_dims_raise
- bad_class_raises                   -> TestLearnedHeadInit::test_bad_class_raises

``LearnedHead.predict_proba`` / ``score_archetypes``:
- probs_sum_to_one                   -> TestLearnedHeadInference::test_probs_sum_to_one
- score_archetypes_descending        -> TestLearnedHeadInference::test_score_archetypes_descending
- vector_dim_mismatch_raises         -> TestLearnedHeadInference::test_vector_dim_mismatch_raises

``LearnedHead.save`` / ``LearnedHead.load``:
- save_load_round_trip               -> TestLearnedHeadPersistence::test_save_load_round_trip
- bad_kind_artefact_raises           -> TestLearnedHeadPersistence::test_bad_kind_artefact_raises

``HeadTrainer.fit``:
- fit_returns_valid_head             -> TestHeadTrainerFit::test_fit_returns_valid_head
- holdout_metadata_populated         -> TestHeadTrainerFit::test_holdout_metadata_populated
- deterministic_with_pinned_seed     -> TestHeadTrainerFit::test_deterministic_with_pinned_seed
- separable_seeds_high_accuracy      -> TestHeadTrainerFit::test_separable_seeds_high_accuracy

``SemanticRouter`` dispatch:
- legacy_path_when_learned_head_none -> TestSemanticRouterDispatch::test_legacy_path_when_learned_head_none
- learned_path_when_loaded           -> TestSemanticRouterDispatch::test_learned_path_when_loaded
- missing_artefact_falls_back        -> TestSemanticRouterDispatch::test_missing_artefact_falls_back
- dim_mismatch_falls_back            -> TestSemanticRouterDispatch::test_dim_mismatch_falls_back
- min_seed_count_below_floor_falls_back
                                     -> TestSemanticRouterDispatch::test_min_seed_count_below_floor_falls_back
- centroid_exclusions_respected      -> TestSemanticRouterDispatch::test_centroid_exclusions_respected
"""
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from semantic_search.config.models import LearnedHeadConfig, QISemanticConfig
from semantic_search.contracts import RouterSeed, RouterSeedDataset
from semantic_search.core.exceptions import ValidationError
from semantic_search.qi.encoder import HashingEncoder
from semantic_search.qi.semantic_router import SemanticRouter
from semantic_search.qi.training.head_trainer import HeadTrainer, HeadTrainerConfig, HeadTrainingMetadata, LearnedHead


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_dataset(seed_count_per_arch: int = 16) -> RouterSeedDataset:
    """Build a 4-archetype dataset with strong intra-class signal."""
    seeds_by_archetype: Dict[str, List[RouterSeed]] = {}
    flavours = {
        'hybrid': 'expiring com domain under 100 dollar',
        'guidance': 'how do i pick a great brandable',
        'explore': 'trending top showcase featured popular',
        'analytics': 'how many auctions sold last month',
    }
    for archetype, stem in flavours.items():
        seeds_by_archetype[archetype] = [
            RouterSeed(query=f"{stem} v{i}", archetype=archetype, origin='manual', source_id='unit')
            for i in range(seed_count_per_arch)
        ]
    return RouterSeedDataset(seeds_by_archetype=seeds_by_archetype, source_path='<test>')


def _semantic_config( encoder_dim: int = 64, learned_head: LearnedHeadConfig = None ) -> QISemanticConfig:
    return QISemanticConfig(
        enabled=True,
        confidence_threshold=0.5,
        embedding_dim=encoder_dim,
        encoder_seed=7,
        archetype_prototypes={},
        seeds_path='qi/router_seeds.yaml',
        min_seeds_per_archetype=1,
        centroid_exclusions=[],
        num_sub_centroids=1,
        learned_head=learned_head,
    )


def _train_head(tmp_path: Path, encoder_dim: int = 64) -> Path:
    """Helper: train a head from a synthetic dataset and return the output path."""
    ds = _build_dataset()
    encoder = HashingEncoder(dim=encoder_dim, seed=7)
    cfg = HeadTrainerConfig(holdout_fraction=0.25, max_epochs=200, random_seed=0)
    trainer = HeadTrainer(config=cfg, encoder=encoder, dataset=ds)
    head = trainer.fit()
    out = tmp_path / 'head.npz'
    head.save(out)
    return out


# ---------------------------------------------------------------------------
# HeadTrainerConfig validation
# ---------------------------------------------------------------------------

class TestHeadTrainerConfig:

    def test_negative_l2_raises(self):
        with pytest.raises(ValidationError):
            HeadTrainerConfig(l2_penalty=-1.0)

    def test_non_positive_lr_raises(self):
        with pytest.raises(ValidationError):
            HeadTrainerConfig(learning_rate=0.0)

    def test_max_epochs_below_one_raises(self):
        with pytest.raises(ValidationError):
            HeadTrainerConfig(max_epochs=0)

    def test_holdout_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            HeadTrainerConfig(holdout_fraction=0.0)
        with pytest.raises(ValidationError):
            HeadTrainerConfig(holdout_fraction=0.6)


# ---------------------------------------------------------------------------
# LearnedHead validation
# ---------------------------------------------------------------------------

class TestLearnedHeadInit:

    def _meta(self, encoder_dim: int = 64) -> HeadTrainingMetadata:
        return HeadTrainingMetadata(
            encoder_dim=encoder_dim,
            classes=['analytics', 'explore', 'guidance', 'hybrid'],
            total_samples=64,
            samples_per_class={'analytics': 16, 'explore': 16, 'guidance': 16, 'hybrid': 16},
            holdout_fraction=0.25,
            f1_macro_holdout=0.9,
            f1_per_class_holdout={'analytics': 0.9, 'explore': 0.9, 'guidance': 0.9, 'hybrid': 0.9},
            accuracy_holdout=0.9,
            epochs_run=100,
            final_loss=0.1,
            l2_penalty=1e-4,
            learning_rate=0.5,
            random_seed=0,
            source_seeds_path='<test>',
            source_hard_negatives_path='',
            created_at=0.0,
        )

    def test_bad_dims_raise(self):
        # weights.shape[0] != encoder_dim
        with pytest.raises(ValidationError):
            LearnedHead(
                weights=np.zeros((32, 4)), bias=np.zeros(4),
                classes=['analytics', 'explore', 'guidance', 'hybrid'],
                encoder_dim=64, metadata=self._meta(),
            )

    def test_bad_class_raises(self):
        with pytest.raises(ValidationError):
            LearnedHead(
                weights=np.zeros((64, 4)), bias=np.zeros(4),
                classes=['filter', 'explore', 'guidance', 'hybrid'],  # 'filter' removed in P1
                encoder_dim=64, metadata=self._meta(),
            )


# ---------------------------------------------------------------------------
# LearnedHead inference
# ---------------------------------------------------------------------------

class TestLearnedHeadInference:

    def test_probs_sum_to_one(self, tmp_path: Path):
        path = _train_head(tmp_path)
        head = LearnedHead.load(path)
        encoder = HashingEncoder(dim=head.encoder_dim, seed=7)
        vec = encoder.encode("expiring com domain")
        probs = head.predict_proba(np.asarray(vec))
        assert probs.shape == (1, 4)
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_score_archetypes_descending(self, tmp_path: Path):
        path = _train_head(tmp_path)
        head = LearnedHead.load(path)
        encoder = HashingEncoder(dim=head.encoder_dim, seed=7)
        vec = encoder.encode("expiring com domain")
        scored = head.score_archetypes(vec)
        assert len(scored) == 4
        scores = [s for (_, s) in scored]
        assert scores == sorted(scores, reverse=True)
        # All probabilities in [0,1].
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_vector_dim_mismatch_raises(self, tmp_path: Path):
        path = _train_head(tmp_path)
        head = LearnedHead.load(path)
        with pytest.raises(ValidationError):
            head.score_archetypes([0.0] * (head.encoder_dim + 1))


# ---------------------------------------------------------------------------
# LearnedHead persistence
# ---------------------------------------------------------------------------

class TestLearnedHeadPersistence:

    def test_save_load_round_trip(self, tmp_path: Path):
        path = _train_head(tmp_path)
        head1 = LearnedHead.load(path)
        # Re-save to a new path and reload — outputs must match byte-equivalent
        # weights and produce the same classification.
        path2 = tmp_path / 'roundtrip.npz'
        head1.save(path2)
        head2 = LearnedHead.load(path2)
        assert head1.classes == head2.classes
        assert head1.encoder_dim == head2.encoder_dim
        assert np.allclose(head1._weights, head2._weights)  # noqa: SLF001
        assert np.allclose(head1._bias, head2._bias)  # noqa: SLF001

    def test_bad_kind_artefact_raises(self, tmp_path: Path):
        bad_path = tmp_path / 'bad.npz'
        np.savez(bad_path, kind=np.array('not_a_head', dtype=str))
        with pytest.raises(ValidationError, match='artefact kind'):
            LearnedHead.load(bad_path)


# ---------------------------------------------------------------------------
# HeadTrainer.fit
# ---------------------------------------------------------------------------

class TestHeadTrainerFit:

    def test_fit_returns_valid_head(self):
        ds = _build_dataset()
        encoder = HashingEncoder(dim=64, seed=7)
        cfg = HeadTrainerConfig(holdout_fraction=0.25, max_epochs=100)
        trainer = HeadTrainer(config=cfg, encoder=encoder, dataset=ds)
        head = trainer.fit()
        assert isinstance(head, LearnedHead)
        assert sorted(head.classes) == ['analytics', 'explore', 'guidance', 'hybrid']
        assert head.encoder_dim == 64

    def test_holdout_metadata_populated(self):
        ds = _build_dataset()
        encoder = HashingEncoder(dim=64, seed=7)
        cfg = HeadTrainerConfig(holdout_fraction=0.25, max_epochs=100)
        head = HeadTrainer(config=cfg, encoder=encoder, dataset=ds).fit()
        meta = head.metadata
        assert 0.0 <= meta.f1_macro_holdout <= 1.0
        assert 0.0 <= meta.accuracy_holdout <= 1.0
        assert sorted(meta.f1_per_class_holdout.keys()) == ['analytics', 'explore', 'guidance', 'hybrid']
        assert meta.total_samples == 64

    def test_deterministic_with_pinned_seed(self):
        ds = _build_dataset()
        cfg = HeadTrainerConfig(holdout_fraction=0.25, max_epochs=80, random_seed=42)
        h1 = HeadTrainer(config=cfg, encoder=HashingEncoder(dim=64, seed=7), dataset=ds).fit()
        h2 = HeadTrainer(config=cfg, encoder=HashingEncoder(dim=64, seed=7), dataset=ds).fit()
        assert np.allclose(h1._weights, h2._weights)  # noqa: SLF001
        assert np.allclose(h1._bias, h2._bias)  # noqa: SLF001

    def test_separable_seeds_high_accuracy(self):
        # The synthetic dataset has very distinct flavours; we should hit
        # high accuracy on the held-out 25% slice.
        ds = _build_dataset(seed_count_per_arch=24)
        encoder = HashingEncoder(dim=128, seed=7)
        cfg = HeadTrainerConfig(holdout_fraction=0.25, max_epochs=200)
        head = HeadTrainer(config=cfg, encoder=encoder, dataset=ds).fit()
        assert head.metadata.accuracy_holdout >= 0.7  # generous floor for hashing encoder


# ---------------------------------------------------------------------------
# SemanticRouter dispatch
# ---------------------------------------------------------------------------

class TestSemanticRouterDispatch:

    def test_legacy_path_when_learned_head_none(self):
        ds = _build_dataset()
        encoder = HashingEncoder(dim=64, seed=7)
        cfg = _semantic_config(encoder_dim=64, learned_head=None)
        router = SemanticRouter(config=cfg, encoder=encoder, seed_dataset=ds)
        assert router._learned_head is None  # noqa: SLF001
        # _score should be the centroid path.
        scores = router._score(encoder.encode("expiring com domain"))  # noqa: SLF001
        assert all(0.0 <= s <= 1.0 for (_, s) in scores)
        assert len(scores) == 4

    def test_learned_path_when_loaded(self, tmp_path: Path):
        artefact_path = _train_head(tmp_path, encoder_dim=64)
        ds = _build_dataset()
        encoder = HashingEncoder(dim=64, seed=7)
        head_cfg = LearnedHeadConfig( kind='logistic', model_path=str(artefact_path), min_seed_count=1,)
        cfg = _semantic_config(encoder_dim=64, learned_head=head_cfg)
        router = SemanticRouter(config=cfg, encoder=encoder, seed_dataset=ds)
        assert router._learned_head is not None  # noqa: SLF001
        # Probabilities must sum to ~1.0 (legacy cosine never sums to 1.0).
        scores = router._score(encoder.encode("expiring com domain"))  # noqa: SLF001
        assert abs(sum(s for (_, s) in scores) - 1.0) < 1e-6

    def test_missing_artefact_falls_back(self, tmp_path: Path):
        ds = _build_dataset()
        encoder = HashingEncoder(dim=64, seed=7)
        head_cfg = LearnedHeadConfig( kind='logistic', model_path=str(tmp_path / 'does_not_exist.npz'), min_seed_count=1,)
        cfg = _semantic_config(encoder_dim=64, learned_head=head_cfg)
        router = SemanticRouter(config=cfg, encoder=encoder, seed_dataset=ds)
        assert router._learned_head is None  # noqa: SLF001 — fell back

    def test_dim_mismatch_falls_back(self, tmp_path: Path):
        # Train head at dim=64, then point a router with dim=32 encoder at it.
        artefact_path = _train_head(tmp_path, encoder_dim=64)
        ds = _build_dataset()
        encoder = HashingEncoder(dim=32, seed=7)
        head_cfg = LearnedHeadConfig( kind='logistic', model_path=str(artefact_path), min_seed_count=1,)
        cfg = _semantic_config(encoder_dim=32, learned_head=head_cfg)
        router = SemanticRouter(config=cfg, encoder=encoder, seed_dataset=ds)
        assert router._learned_head is None  # noqa: SLF001 — fell back

    def test_min_seed_count_below_floor_falls_back(self, tmp_path: Path):
        artefact_path = _train_head(tmp_path, encoder_dim=64)
        ds = _build_dataset()
        encoder = HashingEncoder(dim=64, seed=7)
        # Ridiculously high floor — must fall back.
        head_cfg = LearnedHeadConfig( kind='logistic', model_path=str(artefact_path), min_seed_count=10_000,)
        cfg = _semantic_config(encoder_dim=64, learned_head=head_cfg)
        router = SemanticRouter(config=cfg, encoder=encoder, seed_dataset=ds)
        assert router._learned_head is None  # noqa: SLF001

    def test_centroid_exclusions_respected(self, tmp_path: Path):
        """Excluded archetypes must not appear in the learned-path score list."""
        artefact_path = _train_head(tmp_path, encoder_dim=64)
        ds = _build_dataset()
        encoder = HashingEncoder(dim=64, seed=7)
        head_cfg = LearnedHeadConfig( kind='logistic', model_path=str(artefact_path), min_seed_count=1,)
        cfg = QISemanticConfig(
            enabled=True,
            confidence_threshold=0.5,
            embedding_dim=64,
            encoder_seed=7,
            archetype_prototypes={},
            seeds_path='qi/router_seeds.yaml',
            min_seeds_per_archetype=1,
            centroid_exclusions=['analytics'],
            num_sub_centroids=1,
            learned_head=head_cfg,
        )
        router = SemanticRouter(config=cfg, encoder=encoder, seed_dataset=ds)
        assert router._learned_head is not None  # noqa: SLF001
        scores = router._score(encoder.encode("expiring com domain"))  # noqa: SLF001
        archetypes_returned = {a for (a, _) in scores}
        assert 'analytics' not in archetypes_returned
        # Filtered output no longer sums to 1.0 — that's expected.
        assert len(scores) == 3
