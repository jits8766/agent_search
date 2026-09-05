"""Cold-start seed dataset loader for the Semantic Router.

Reads a curated YAML file of the shape::

    archetypes:
      hybrid:
        - "expiring .com under $100"
        - "short brandable .com"
        - ...
      explore:
        - "trending domain names"
        - ...
      ...

Each archetype must have at least `min_seeds_per_archetype` entries (the
floor — default 30). Loading is read-only; no writes touch the source file.
The loader emits one `RouterSeed` per query with `origin='manual'` and a stable
`source_id` derived from the file mtime so re-loads can be tracked in the audit.

Trust-boundary contract (data-contracts.mdc):
- Path: must resolve under the configured search-config directory (no traversal)
- Archetypes: validated against `QUERY_TYPES` frozenset
- Seeds: each query string is non-empty and stripped before construction
- Duplicate `(query, archetype)` pairs raise `ConfigurationError` at load time
"""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from llm_core.logging_utils import mask_path

from semantic_search.contracts import QUERY_TYPES, RouterSeed, RouterSeedDataset
from semantic_search.core.exceptions import ConfigurationError, ValidationError
from semantic_search.core.logging_utils import get_logger
from semantic_search.core.validation import sanitize_path_component

logger = get_logger(__name__)


class RouterSeedLoader:
    """Loads + revalidates the semantic-router seed dataset.

    :param seeds_path: str - YAML path (relative to package config dir or absolute)
    :param min_seeds_per_archetype: int - Per-archetype floor; loader fails if any archetype is below this count
    :param base_dir: Optional[Path] - Base dir for resolving relative paths (defaults to the semantic_search package root)
    """

    def __init__(self, seeds_path: str, min_seeds_per_archetype: int, base_dir: Optional[Path] = None):
        if not isinstance(seeds_path, str) or not seeds_path:
            raise ConfigurationError("RouterSeedLoader.seeds_path must be a non-empty string")
        if min_seeds_per_archetype < 1:
            raise ConfigurationError("RouterSeedLoader.min_seeds_per_archetype must be >= 1")
        self._seeds_path = seeds_path
        self._min = min_seeds_per_archetype
        # Default base dir is the `semantic_search` package root, so curated seed YAMLs
        # live next to their consumer (e.g. `qi/router_seeds.yaml`). Anchoring on
        # the package — never CWD — means a misconfigured working directory cannot
        # change which file is loaded.
        self._base_dir = base_dir if base_dir is not None else Path(__file__).resolve().parent.parent

    def _resolve_path(self) -> Path:
        """Resolve the seeds_path against the config dir + reject path traversal."""
        candidate = Path(self._seeds_path)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            # Validate each component to defeat `..`/null-byte traversal before joining.
            for part in candidate.parts:
                sanitize_path_component(part, "seeds_path")
            resolved = (self._base_dir / candidate).resolve()
        if not resolved.exists():
            raise ConfigurationError(f"RouterSeedLoader: seeds file not found path={mask_path(resolved)}")
        if not resolved.is_file():
            raise ConfigurationError(f"RouterSeedLoader: seeds path is not a regular file path={mask_path(resolved)}")
        return resolved

    def load(self) -> RouterSeedDataset:
        """Read and validate the seed YAML; return a typed `RouterSeedDataset`."""
        path = self._resolve_path()
        try:
            with open(path, 'r', encoding='utf-8') as f:
                raw = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"RouterSeedLoader: invalid YAML path={mask_path(path)} error={e}") from e
        if not isinstance(raw, dict):
            raise ConfigurationError(f"RouterSeedLoader: top-level YAML must be a mapping path={mask_path(path)}")
        archetypes_raw = raw.get('archetypes')
        if not isinstance(archetypes_raw, dict) or not archetypes_raw:
            raise ConfigurationError(f"RouterSeedLoader: missing or empty 'archetypes' section path={mask_path(path)}")
        # mtime gives us a stable per-file source_id so re-loads of the same file
        # produce identical RouterSeed.source_id values (audit trail stability).
        mtime = int(os.path.getmtime(path))
        source_id = f"manual_{path.stem}_{mtime}"
        seeds_by_archetype: Dict[str, List[RouterSeed]] = {}
        for archetype, queries in archetypes_raw.items():
            archetype_str = str(archetype)
            if archetype_str not in QUERY_TYPES:
                raise ConfigurationError(f"RouterSeedLoader: archetype '{archetype_str}' not in QUERY_TYPES path={mask_path(path)}")
            if not isinstance(queries, list):
                raise ConfigurationError(f"RouterSeedLoader: archetypes.{archetype_str} must be a list path={mask_path(path)}")
            built: List[RouterSeed] = []
            seen_in_arch: set = set()
            for raw_query in queries:
                if not isinstance(raw_query, str):
                    raise ConfigurationError(f"RouterSeedLoader: archetypes.{archetype_str} entries must be strings path={mask_path(path)}")
                normalized = raw_query.strip()
                if not normalized:
                    continue
                key = normalized.lower()
                if key in seen_in_arch:
                    # Within-archetype duplicates are a curator-side mistake; flag loudly.
                    raise ConfigurationError(f"RouterSeedLoader: duplicate seed within archetype='{archetype_str}' query='{normalized}'")
                seen_in_arch.add(key)
                try:
                    seed = RouterSeed(query=normalized, archetype=archetype_str, origin='manual', source_id=source_id)
                except ValidationError as e:
                    raise ConfigurationError(f"RouterSeedLoader: invalid seed query='{normalized}' archetype='{archetype_str}' error={e}") from e
                built.append(seed)
            if len(built) < self._min:
                raise ConfigurationError(f"RouterSeedLoader: archetype '{archetype_str}' has {len(built)} seeds; min required={self._min} (floor)")
            seeds_by_archetype[archetype_str] = built
        try:
            dataset = RouterSeedDataset(seeds_by_archetype=seeds_by_archetype, source_path=str(path))
        except ValidationError as e:
            raise ConfigurationError(f"RouterSeedLoader: dataset validation failed path={mask_path(path)} error={e}") from e
        logger.info(f"router_seeds_loaded path={mask_path(path)} archetypes={len(dataset.archetypes())} total={dataset.total()} min_required={self._min}")
        return dataset


def dataset_from_inline(archetype_prototypes: Dict[str, List[str]]) -> RouterSeedDataset:
    """Build a `RouterSeedDataset` from the inline YAML override (test path).

    Used when `qi.semantic.seeds_path` is empty and `archetype_prototypes` is the
    source. The inline path bypasses the minimum-seed floor on purpose:
    test fixtures need to keep prototype counts small for speed, and the production
    contract enforces the floor via the loader path (`RouterSeedLoader.load`).
    """
    if not isinstance(archetype_prototypes, dict) or not archetype_prototypes:
        raise ConfigurationError("dataset_from_inline: archetype_prototypes must be a non-empty dict")
    seeds_by_archetype: Dict[str, List[RouterSeed]] = {}
    for archetype, queries in archetype_prototypes.items():
        if archetype not in QUERY_TYPES:
            raise ConfigurationError(f"dataset_from_inline: archetype '{archetype}' not in QUERY_TYPES")
        if not isinstance(queries, list) or not queries:
            raise ConfigurationError(f"dataset_from_inline: archetype '{archetype}' must have a non-empty list of seeds")
        built: List[RouterSeed] = []
        for raw_query in queries:
            if not isinstance(raw_query, str) or not raw_query.strip():
                raise ConfigurationError(f"dataset_from_inline: archetype '{archetype}' has an empty seed string")
            built.append(RouterSeed(query=raw_query.strip(), archetype=archetype, origin='manual', source_id='inline_override'))
        seeds_by_archetype[archetype] = built
    return RouterSeedDataset(seeds_by_archetype=seeds_by_archetype, source_path='<inline>')
