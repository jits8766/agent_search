"""Typed dataclasses for QI offline training tools.

Defines the contracts shared by ``hard_negative_miner`` and the
learned-head trainer. All dataclasses validate inputs in
``__post_init__`` and offer explicit ``from_dict`` / ``to_dict`` helpers so
JSONL artefacts round-trip deterministically.

Trust-boundary notes (per ``data-contracts.mdc``):
- Every numeric field is range-checked.
- String fields require non-empty content where semantically required.
- Archetype names are validated against ``QUERY_TYPES`` so a stale artefact
  cannot smuggle a removed archetype back into the system.
"""
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping

from semantic_search.contracts import QUERY_TYPES
from semantic_search.core.exceptions import ValidationError


@dataclass(frozen=True)
class HardNegativeMinerConfig:
    """Configuration for a single hard-negative mining run.

    :param threshold: float - Cosine score ``> threshold`` against a
        wrong-archetype centroid marks the seed as a hard negative for that
        archetype. Must be in (0.0, 1.0].
    :param num_sub_centroids: int - K for the per-archetype K-means
        clustering. Mirrors ``qi.semantic.num_sub_centroids`` in production
        so the miner sees the same centroid topology the live router sees.
        Must be >= 1.
    :param encoder_seed: int - Deterministic seed forwarded to
        ``kmeans_spherical`` so re-runs produce byte-identical artefacts.
    :param min_seeds_per_archetype: int - Loader floor; passed through so
        the CLI fails fast if the curated corpus shrinks below the safety
        net. Must be >= 1.
    """
    threshold: float
    num_sub_centroids: int
    encoder_seed: int
    min_seeds_per_archetype: int

    def __post_init__(self) -> None:
        if not 0.0 < float(self.threshold) <= 1.0:
            raise ValidationError("HardNegativeMinerConfig.threshold must be in (0.0, 1.0]")
        if int(self.num_sub_centroids) < 1:
            raise ValidationError("HardNegativeMinerConfig.num_sub_centroids must be >= 1")
        if int(self.min_seeds_per_archetype) < 1:
            raise ValidationError("HardNegativeMinerConfig.min_seeds_per_archetype must be >= 1")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> 'HardNegativeMinerConfig':
        for required in ('threshold', 'num_sub_centroids', 'encoder_seed', 'min_seeds_per_archetype'):
            if required not in d:
                raise ValidationError(f"HardNegativeMinerConfig.from_dict missing key '{required}'")
        return cls(
            threshold=float(d['threshold']),
            num_sub_centroids=int(d['num_sub_centroids']),
            encoder_seed=int(d['encoder_seed']),
            min_seeds_per_archetype=int(d['min_seeds_per_archetype']),
        )


@dataclass(frozen=True)
class HardNegativeRow:
    """One labelled training row produced by the hard-negative miner.

    The row is a hard *positive* for ``archetype`` (its true label) and a
    hard *negative* for every archetype listed in ``is_hard_negative_for``
    (those scored above ``threshold`` despite being wrong-class).

    :param query: str - The seed query text (non-empty).
    :param archetype: str - The seed's true archetype (one of QUERY_TYPES).
    :param label: str - Same as ``archetype`` — kept distinct so a future
        relabeller can override without losing provenance.
    :param scores: Dict[str, float] - Per-archetype max cosine vs the
        archetype's K-means sub-centroids. Keys are archetype names, all in
        QUERY_TYPES; values in [-1.0, 1.0].
    :param is_hard_negative_for: List[str] - Wrong archetypes whose score
        exceeded ``threshold`` for this seed. Empty when the seed is
        unambiguous.
    """
    query: str
    archetype: str
    label: str
    scores: Dict[str, float]
    is_hard_negative_for: List[str]

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValidationError("HardNegativeRow.query must be a non-empty string")
        if self.archetype not in QUERY_TYPES:
            raise ValidationError(f"HardNegativeRow.archetype must be one of {sorted(QUERY_TYPES)}")
        if self.label not in QUERY_TYPES:
            raise ValidationError(f"HardNegativeRow.label must be one of {sorted(QUERY_TYPES)}")
        if not isinstance(self.scores, dict) or not self.scores:
            raise ValidationError("HardNegativeRow.scores must be a non-empty dict")
        for arch, score in self.scores.items():
            if arch not in QUERY_TYPES:
                raise ValidationError(f"HardNegativeRow.scores key '{arch}' not in QUERY_TYPES")
            if not -1.0 <= float(score) <= 1.0:
                raise ValidationError(f"HardNegativeRow.scores['{arch}']={score} out of [-1.0, 1.0]")
        if not isinstance(self.is_hard_negative_for, list):
            raise ValidationError("HardNegativeRow.is_hard_negative_for must be a list")
        for arch in self.is_hard_negative_for:
            if arch not in QUERY_TYPES:
                raise ValidationError(f"HardNegativeRow.is_hard_negative_for entry '{arch}' not in QUERY_TYPES")
            if arch == self.archetype:
                raise ValidationError("HardNegativeRow.is_hard_negative_for must not contain the row's own archetype")

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for JSONL emission."""
        return {
            'query': self.query,
            'archetype': self.archetype,
            'label': self.label,
            'scores': dict(self.scores),
            'is_hard_negative_for': list(self.is_hard_negative_for),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> 'HardNegativeRow':
        for required in ('query', 'archetype', 'label', 'scores', 'is_hard_negative_for'):
            if required not in d:
                raise ValidationError(f"HardNegativeRow.from_dict missing key '{required}'")
        return cls(
            query=str(d['query']),
            archetype=str(d['archetype']),
            label=str(d['label']),
            scores={str(k): float(v) for k, v in d['scores'].items()},
            is_hard_negative_for=[str(x) for x in d['is_hard_negative_for']],
        )


@dataclass(frozen=True)
class HardNegativeArtifact:
    """One hard-negative mining run — header + rows.

    Persisted as JSONL where the first line is this header (without the
    ``rows`` field) and every subsequent line is a single ``HardNegativeRow``.
    The header carries enough provenance for the learned-head trainer to verify it is reading
    a compatible artefact (same encoder dim, same archetype set).

    :param rows: List[HardNegativeRow] - One row per seed (positives +
        cross-archetype hard-negative annotations).
    :param threshold: float - Threshold used during this run (same value
        as in the miner config).
    :param num_sub_centroids: int - K used for the K-means run.
    :param encoder_seed: int - Encoder/cluster seed (reproducibility).
    :param encoder_dim: int - Encoder dimensionality at run-time.
    :param archetypes: List[str] - Sorted archetype list scored against;
        every entry must be in QUERY_TYPES.
    :param counts_per_archetype: Dict[str, int] - Seed counts per archetype
        (sanity check vs the source YAML).
    :param hard_negative_counts: Dict[str, int] - For each archetype,
        the number of rows where that archetype appears in
        ``is_hard_negative_for``. Diagnoses confusable pairs at a glance.
    :param source_path: str - Filesystem path of the seed YAML the run
        consumed (audit only).
    :param created_at: float - Unix timestamp (auto-populated).
    """
    rows: List[HardNegativeRow]
    threshold: float
    num_sub_centroids: int
    encoder_seed: int
    encoder_dim: int
    archetypes: List[str]
    counts_per_archetype: Dict[str, int]
    hard_negative_counts: Dict[str, int]
    source_path: str
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not isinstance(self.rows, list):
            raise ValidationError("HardNegativeArtifact.rows must be a list")
        for r in self.rows:
            if not isinstance(r, HardNegativeRow):
                raise ValidationError("HardNegativeArtifact.rows entries must be HardNegativeRow")
        if not 0.0 < float(self.threshold) <= 1.0:
            raise ValidationError("HardNegativeArtifact.threshold must be in (0.0, 1.0]")
        if int(self.num_sub_centroids) < 1:
            raise ValidationError("HardNegativeArtifact.num_sub_centroids must be >= 1")
        if int(self.encoder_dim) < 4:
            raise ValidationError("HardNegativeArtifact.encoder_dim must be >= 4")
        if not isinstance(self.archetypes, list) or not self.archetypes:
            raise ValidationError("HardNegativeArtifact.archetypes must be a non-empty list")
        for a in self.archetypes:
            if a not in QUERY_TYPES:
                raise ValidationError(f"HardNegativeArtifact.archetypes entry '{a}' not in QUERY_TYPES")
        if not isinstance(self.counts_per_archetype, dict):
            raise ValidationError("HardNegativeArtifact.counts_per_archetype must be a dict")
        for k, v in self.counts_per_archetype.items():
            if k not in QUERY_TYPES:
                raise ValidationError(f"HardNegativeArtifact.counts_per_archetype key '{k}' not in QUERY_TYPES")
            if int(v) < 0:
                raise ValidationError(f"HardNegativeArtifact.counts_per_archetype['{k}']={v} must be >= 0")
        if not isinstance(self.hard_negative_counts, dict):
            raise ValidationError("HardNegativeArtifact.hard_negative_counts must be a dict")
        for k, v in self.hard_negative_counts.items():
            if k not in QUERY_TYPES:
                raise ValidationError(f"HardNegativeArtifact.hard_negative_counts key '{k}' not in QUERY_TYPES")
            if int(v) < 0:
                raise ValidationError(f"HardNegativeArtifact.hard_negative_counts['{k}']={v} must be >= 0")
        if not isinstance(self.source_path, str):
            raise ValidationError("HardNegativeArtifact.source_path must be a string")

    def header_dict(self) -> Dict[str, Any]:
        """Header-only dict for the first line of the JSONL artefact."""
        return {
            '_kind': 'hard_negative_artifact_header',
            'threshold': float(self.threshold),
            'num_sub_centroids': int(self.num_sub_centroids),
            'encoder_seed': int(self.encoder_seed),
            'encoder_dim': int(self.encoder_dim),
            'archetypes': list(self.archetypes),
            'counts_per_archetype': dict(self.counts_per_archetype),
            'hard_negative_counts': dict(self.hard_negative_counts),
            'source_path': self.source_path,
            'created_at': float(self.created_at),
        }

    def total_rows(self) -> int:
        """Total number of rows in the artefact."""
        return len(self.rows)

    def total_hard_negatives(self) -> int:
        """Sum of confusable wrong-class annotations across all rows."""
        return sum(len(r.is_hard_negative_for) for r in self.rows)
