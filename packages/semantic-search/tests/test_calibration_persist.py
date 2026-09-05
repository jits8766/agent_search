"""Tests for calibration fit persistence (save/load/fingerprint).

Persisting fitted temperatures lets warm boots skip the LLM-driven golden-seed
replay. These tests lock the round-trip and every cache-miss path (absent,
corrupt, fingerprint mismatch, invalid fit) that must fall back to a live fit.
"""
import json
from pathlib import Path

from semantic_search.calibration.calibrator import CalibrationFit
from semantic_search.calibration.persist import compute_fingerprint, load_fits, save_fits


def _fit(tier: str, temperature: float = 1.5, fitted: bool = True) -> CalibrationFit:
    return CalibrationFit(
        tier=tier, temperature=temperature, n_samples=120,
        pre_nll=0.6, post_nll=0.4, accuracy=0.8, ece_pre=0.1, ece_post=0.03,
        fitted=fitted, source='golden_seeds',
    )


def _sample_fits() -> dict:
    return {'L0_entity': _fit('L0_entity', 1.7), 'L1_semantic': CalibrationFit.identity('L1_semantic')}


class TestRoundTrip:

    def test_save_then_load_preserves_fits(self, tmp_path: Path) -> None:
        p = str(tmp_path / 'cal.json')
        fits = _sample_fits()
        save_fits(p, fits, 'fp-1')
        loaded = load_fits(p, 'fp-1')
        assert loaded is not None
        assert sorted(loaded) == ['L0_entity', 'L1_semantic']
        assert loaded['L0_entity'].temperature == 1.7
        assert loaded['L0_entity'].source == 'golden_seeds'
        assert loaded['L1_semantic'].fitted is False

    def test_load_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert load_fits(str(tmp_path / 'nope.json'), 'fp-1') is None

    def test_fingerprint_mismatch_returns_none(self, tmp_path: Path) -> None:
        p = str(tmp_path / 'cal.json')
        save_fits(p, _sample_fits(), 'fp-1')
        assert load_fits(p, 'fp-DIFFERENT') is None

    def test_corrupt_json_returns_none(self, tmp_path: Path) -> None:
        p = tmp_path / 'cal.json'
        p.write_text('{ not valid json')
        assert load_fits(str(p), 'fp-1') is None

    def test_invalid_fit_row_returns_none(self, tmp_path: Path) -> None:
        # temperature <= 0 fails CalibrationFit validation on reconstruction.
        p = tmp_path / 'cal.json'
        payload = {'schema': 1, 'fingerprint': 'fp-1', 'fits': [{
            'tier': 'L0_entity', 'temperature': 0.0, 'n_samples': 10,
            'pre_nll': 0.1, 'post_nll': 0.1, 'accuracy': 0.5,
            'ece_pre': 0.1, 'ece_post': 0.1, 'fitted': True, 'source': 'golden_seeds',
        }]}
        p.write_text(json.dumps(payload))
        assert load_fits(str(p), 'fp-1') is None

    def test_empty_fits_returns_none(self, tmp_path: Path) -> None:
        p = tmp_path / 'cal.json'
        p.write_text(json.dumps({'schema': 1, 'fingerprint': 'fp-1', 'fits': []}))
        assert load_fits(str(p), 'fp-1') is None


class TestFingerprint:

    def test_deterministic(self, tmp_path: Path) -> None:
        seeds = tmp_path / 'seeds.yaml'
        seeds.write_text('cases: []\n')
        params = {'min_temperature': 0.1, 'tier_keys': 'L0_entity,L1_semantic'}
        a = compute_fingerprint(str(seeds), 'qi.entity.v3', params, 1)
        b = compute_fingerprint(str(seeds), 'qi.entity.v3', params, 1)
        assert a == b

    def test_cache_version_busts(self, tmp_path: Path) -> None:
        seeds = tmp_path / 'seeds.yaml'
        seeds.write_text('cases: []\n')
        params = {'tier_keys': 'L0_entity'}
        assert compute_fingerprint(str(seeds), 'qi.entity.v3', params, 1) != compute_fingerprint(str(seeds), 'qi.entity.v3', params, 2)

    def test_seed_bytes_bust(self, tmp_path: Path) -> None:
        seeds = tmp_path / 'seeds.yaml'
        seeds.write_text('cases: []\n')
        params = {'tier_keys': 'L0_entity'}
        fp1 = compute_fingerprint(str(seeds), 'qi.entity.v3', params, 1)
        seeds.write_text('cases: [{input_query: x, expected_query_type: hybrid}]\n')
        fp2 = compute_fingerprint(str(seeds), 'qi.entity.v3', params, 1)
        assert fp1 != fp2

    def test_prompt_tag_busts(self, tmp_path: Path) -> None:
        seeds = tmp_path / 'seeds.yaml'
        seeds.write_text('cases: []\n')
        params = {'tier_keys': 'L0_entity'}
        assert compute_fingerprint(str(seeds), 'qi.entity.v3', params, 1) != compute_fingerprint(str(seeds), 'qi.entity.v4', params, 1)
