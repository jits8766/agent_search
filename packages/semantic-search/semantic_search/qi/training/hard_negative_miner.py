"""Hard-negative miner for the QI semantic router.

For every seed query in the curated ``RouterSeedDataset``, encode the seed
and score it against every archetype's K-means sub-centroids (the same
centroid topology the live ``SemanticRouter`` builds). When a seed scores
above ``threshold`` against a *wrong* archetype's centroids, that seed is
flagged as a hard negative for the wrong archetype.

The output JSONL is consumed by the learned-head trainer. The
miner is a **read-only, offline tool**: it never mutates the live router's
state and performs no network I/O.

CLI usage::

    python -m semantic_search.qi.training.hard_negative_miner \\
        --config-dir semantic_search/config \\
        --output    semantic_search/qi/training_data/hard_negatives.jsonl \\
        --threshold 0.65

When ``--config-dir`` is omitted the CLI uses the project default
(``semantic_search/config``). All numeric knobs default to the live router's
config values so the miner sees the same centroid topology by default.

Layer rules (per ``architecture.mdc``): stdlib + ``core`` + ``contracts`` +
``config`` + sibling QI primitives. No retrieval / orchestration imports.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from semantic_search.config.loader import load_config
from semantic_search.config.models import AgentSearchConfig
from semantic_search.contracts import QUERY_TYPES, RouterSeedDataset
from semantic_search.core.exceptions import ValidationError
from semantic_search.core.logging_utils import get_logger
from semantic_search.qi.encoder import Encoder, cosine_similarity, kmeans_spherical
from semantic_search.qi.seed_loader import RouterSeedLoader
from semantic_search.qi.training.data_schema import HardNegativeArtifact, HardNegativeMinerConfig, HardNegativeRow

logger = get_logger(__name__)


class HardNegativeMiner:
    """Mines cross-archetype hard negatives from a curated seed corpus.

    :param config: HardNegativeMinerConfig - Miner knobs (threshold, K,
        encoder seed, min-seeds floor).
    :param encoder: Encoder - Same encoder type the live router uses
        (production: FastEmbedEncoder; tests: HashingEncoder). Centroid
        math is dim-coupled, so a mismatched encoder dim is rejected.
    :param dataset: RouterSeedDataset - Curated seed dataset (loaded via
        ``RouterSeedLoader``). Floor is enforced upstream by the loader.
    :raises ValidationError: When inputs are wrong type / None or when the
        dataset's archetype counts violate ``min_seeds_per_archetype``.
    """

    def __init__(self, config: HardNegativeMinerConfig, encoder: Encoder, dataset: RouterSeedDataset):
        if config is None or not isinstance(config, HardNegativeMinerConfig):
            raise ValidationError("HardNegativeMiner requires a HardNegativeMinerConfig")
        if encoder is None or not isinstance(encoder, Encoder):
            raise ValidationError("HardNegativeMiner requires a non-None Encoder")
        if dataset is None or not isinstance(dataset, RouterSeedDataset):
            raise ValidationError("HardNegativeMiner requires a non-None RouterSeedDataset")
        # Defence-in-depth: the loader already enforces this, but the miner
        # is also entry-point for tests that build datasets via
        # ``dataset_from_inline``, which bypasses the floor by design.
        for archetype, seeds in dataset.seeds_by_archetype.items():
            if archetype not in QUERY_TYPES:
                raise ValidationError(
                    f"HardNegativeMiner dataset archetype '{archetype}' not in QUERY_TYPES"
                )
            if len(seeds) < config.min_seeds_per_archetype:
                raise ValidationError(
                    f"HardNegativeMiner archetype '{archetype}' has {len(seeds)} seeds; "
                    f"min required={config.min_seeds_per_archetype}"
                )
        self._config = config
        self._encoder = encoder
        self._dataset = dataset

    # ------------------------------------------------------------------
    # Centroid construction (mirrors SemanticRouter._build_centroids)
    # ------------------------------------------------------------------

    def _build_centroids(self) -> Dict[str, List[List[float]]]:
        """Compute per-archetype K-means sub-centroids.

        Mirrors ``SemanticRouter._build_centroids`` byte-for-byte (same K,
        same seed) so the miner observes the same per-archetype topology
        the live router would. The miner deliberately does NOT honour
        ``centroid_exclusions``: even excluded archetypes (e.g.
        ``analytics`` in some configs) must have their seeds scored — they
        are valid hard-negative *sources* even when L1 doesn't gate on
        them.
        """
        sub_centroids: Dict[str, List[List[float]]] = {}
        k = self._config.num_sub_centroids
        seed = self._config.encoder_seed
        for archetype in self._dataset.archetypes():
            texts = self._dataset.texts(archetype)
            vectors = self._encoder.encode_batch(texts)
            non_zero = [v for v in vectors if any(abs(x) > 0.0 for x in v)]
            if not non_zero:
                raise ValidationError(
                    f"HardNegativeMiner archetype '{archetype}' produced only zero vectors"
                )
            sub_centroids[archetype] = kmeans_spherical(non_zero, k=k, seed=seed)
        return sub_centroids

    @staticmethod
    def _max_score(query_vec: Sequence[float], centroids: Sequence[Sequence[float]]) -> float:
        """Max cosine similarity of ``query_vec`` vs a list of centroids."""
        return max(cosine_similarity(list(query_vec), list(c)) for c in centroids)

    # ------------------------------------------------------------------
    # Public mining entry point
    # ------------------------------------------------------------------

    def mine(self) -> HardNegativeArtifact:
        """Score every seed against every archetype; emit a typed artefact."""
        sub_centroids = self._build_centroids()
        archetypes_sorted = sorted(sub_centroids.keys())
        rows: List[HardNegativeRow] = []
        hn_counts: Dict[str, int] = {a: 0 for a in archetypes_sorted}
        counts_per_arch: Dict[str, int] = {}
        threshold = float(self._config.threshold)
        for archetype in archetypes_sorted:
            seed_texts = self._dataset.texts(archetype)
            counts_per_arch[archetype] = len(seed_texts)
            for query in seed_texts:
                vec = self._encoder.encode(query)
                if all(x == 0.0 for x in vec):
                    # Zero-vector queries cannot be reliably scored — skip
                    # rather than emit garbage. Should not happen in practice
                    # because _build_centroids would already have raised.
                    logger.warning(f"hard_negative_miner_zero_vector archetype={archetype} query_prefix={query[:32]!r}")
                    continue
                scores: Dict[str, float] = {}
                for other_arch, centroids in sub_centroids.items():
                    scores[other_arch] = self._max_score(vec, centroids)
                hard_negs = [
                    a for a in archetypes_sorted
                    if a != archetype and scores[a] > threshold
                ]
                for confused in hard_negs:
                    hn_counts[confused] = hn_counts.get(confused, 0) + 1
                rows.append(HardNegativeRow(query=query, archetype=archetype, label=archetype, scores=scores, is_hard_negative_for=hard_negs))
        encoder_dim = int(self._encoder.dim)
        artifact = HardNegativeArtifact(
            rows=rows,
            threshold=threshold,
            num_sub_centroids=int(self._config.num_sub_centroids),
            encoder_seed=int(self._config.encoder_seed),
            encoder_dim=encoder_dim,
            archetypes=archetypes_sorted,
            counts_per_archetype=counts_per_arch,
            hard_negative_counts=hn_counts,
            source_path=self._dataset.source_path,
        )
        logger.info(
            f"hard_negative_miner_complete rows={len(rows)} "
            f"hard_negatives_total={artifact.total_hard_negatives()} "
            f"counts_per_archetype={counts_per_arch} "
            f"hard_negative_counts={hn_counts} threshold={threshold}"
        )
        return artifact


# ---------------------------------------------------------------------------
# JSONL persistence helpers
# ---------------------------------------------------------------------------

def write_artifact_jsonl(artifact: HardNegativeArtifact, output_path: Path) -> None:
    """Write the artefact to a JSONL file (header line + one row per line).

    The output directory is created on-demand. Existing files are
    overwritten — runs are deterministic by construction (encoder seed
    + K-means seed pinned), so re-running with the same config produces an
    identical file (only the ``created_at`` differs).
    """
    if not isinstance(artifact, HardNegativeArtifact):
        raise ValidationError("write_artifact_jsonl requires a HardNegativeArtifact")
    if not isinstance(output_path, Path):
        raise ValidationError("write_artifact_jsonl requires a pathlib.Path output_path")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(artifact.header_dict(), sort_keys=True) + '\n')
        for row in artifact.rows:
            f.write(json.dumps(row.to_dict(), sort_keys=True) + '\n')
    logger.info(f"hard_negative_miner_artifact_written path={output_path} rows={artifact.total_rows()} hard_negatives={artifact.total_hard_negatives()}")


def read_artifact_jsonl(input_path: Path) -> Tuple[Dict[str, object], List[HardNegativeRow]]:
    """Inverse of ``write_artifact_jsonl`` — returns (header_dict, rows).

    The learned-head trainer consumes this. Returning the header as a plain dict
    (rather than reconstructing the full ``HardNegativeArtifact``) avoids
    re-validating archetype lists against the *current* QUERY_TYPES when
    the artefact was written before a taxonomy change — the trainer has its
    own compatibility check.
    """
    if not isinstance(input_path, Path):
        raise ValidationError("read_artifact_jsonl requires a pathlib.Path input_path")
    if not input_path.exists():
        raise ValidationError(f"read_artifact_jsonl: path does not exist: {input_path}")
    rows: List[HardNegativeRow] = []
    header: Dict[str, object] = {}
    with open(input_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if idx == 0 and obj.get('_kind') == 'hard_negative_artifact_header':
                header = obj
                continue
            rows.append(HardNegativeRow.from_dict(obj))
    return header, rows


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli_main() -> int:
    """CLI: load live config, build encoder + dataset, mine, write JSONL."""
    from semantic_search.registry import _build_encoder
    parser = argparse.ArgumentParser(description="Mine cross-archetype hard negatives from the QI seed corpus.")
    parser.add_argument('--config-path', type=str, default=None, help='Path to base.yaml (default: package-bundled base.yaml).')
    parser.add_argument('--output', type=str, default='semantic_search/qi/training_data/hard_negatives.jsonl', help='JSONL output path (created if missing).')
    parser.add_argument('--threshold', type=float, default=0.65, help='Cosine score above which a wrong-archetype seed is a hard negative.')
    args = parser.parse_args()

    raw = load_config(args.config_path)
    cfg = AgentSearchConfig.from_dict(raw)
    sem_cfg = cfg.qi.semantic
    if not sem_cfg.seeds_path:
        raise ValidationError("qi.semantic.seeds_path must be set; the inline-prototype path is not supported by the hard-negative miner CLI.")
    loader = RouterSeedLoader(seeds_path=sem_cfg.seeds_path, min_seeds_per_archetype=sem_cfg.min_seeds_per_archetype)
    dataset = loader.load()
    encoder = _build_encoder(cfg)
    miner_cfg = HardNegativeMinerConfig(
        threshold=float(args.threshold),
        num_sub_centroids=sem_cfg.num_sub_centroids,
        encoder_seed=sem_cfg.encoder_seed,
        min_seeds_per_archetype=sem_cfg.min_seeds_per_archetype,
    )
    miner = HardNegativeMiner(config=miner_cfg, encoder=encoder, dataset=dataset)
    artifact = miner.mine()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        # Resolve relative to the *project root*, never CWD, so the CLI is
        # safe to invoke from any directory.
        project_root = Path(__file__).resolve().parents[3]
        output_path = (project_root / output_path).resolve()
    write_artifact_jsonl(artifact, output_path)
    print(f"wrote rows={artifact.total_rows()} hard_negatives={artifact.total_hard_negatives()} path={output_path}")
    # Surface unexpected env hints so a misconfigured offline run is loud.
    if os.environ.get('HF_HUB_OFFLINE') != '1':
        print("warning: HF_HUB_OFFLINE != 1 — production runs should set HF_HUB_OFFLINE=1")
    return 0


if __name__ == '__main__':
    raise SystemExit(_cli_main())
