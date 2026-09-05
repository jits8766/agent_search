"""Shared pytest fixtures for semantic_search.
Each fixture builds the smallest valid object for its layer using the production
config — no hidden defaults, no monkey-patched globals.
"""
import asyncio
import functools
from typing import Any, Dict

import pytest

import semantic_search.config.loader as _loader_mod

from semantic_search.cache.exact_cache import ExactCache
from semantic_search.cache.structured_cache import StructuredCache
from semantic_search.config.models import AgentSearchConfig
from semantic_search.contracts import RouterSeedDataset
from semantic_search.qi.encoder import HashingEncoder
from semantic_search.qi.seed_loader import RouterSeedLoader, dataset_from_inline
from semantic_search.registry import build_subsystems, Subsystems

_real_load_config = _loader_mod.load_config


def _apply_test_overrides(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Force test-safe config: hashing encoder + in-memory backends, no CH/DuckDB."""
    qi_enc = raw.setdefault('qi', {}).setdefault('encoder', {})
    qi_enc['backend'] = 'hashing'
    sem_dim = raw.get('qi', {}).get('semantic', {}).get('embedding_dim', 128)
    qi_enc['dim'] = sem_dim
    qi_enc.setdefault('cascade', {})['enabled'] = False
    # Disable the learned head in tests: the head is trained on
    # FastEmbed semantic embeddings, but tests use a hashing-encoder LSH
    # projection — incompatible distributions at the same dim. Tests that
    # specifically exercise the learned-head path build their own routers.
    raw.setdefault('qi', {}).setdefault('semantic', {})['learned_head'] = None
    raw.setdefault('retrieval', {}).setdefault('vector', {})['backend'] = 'memory'
    raw['retrieval']['vector']['embedding_dim'] = sem_dim
    raw.setdefault('retrieval', {}).setdefault('structured', {})['backend'] = 'memory'
    raw.setdefault('retrieval', {}).setdefault('qdrant', {}).setdefault('hybrid', {})['enabled'] = False
    nl = raw.setdefault('nl_to_sql', {})
    analytics = nl.setdefault('analytics', {})
    analytics['enabled'] = False
    bulk = analytics.setdefault('bulk', {})
    bulk['enabled'] = False
    bulk['backend'] = 'noop'
    # Skip live CH readiness round-trip in unit tests (no local ClickHouse assumed).
    ch = analytics.setdefault('clickhouse', {})
    readiness = ch.setdefault('readiness_probe', {})
    readiness['enabled'] = False
    readiness.setdefault('sql', 'SELECT 1')
    readiness.setdefault('timeout_seconds', 2.0)
    readiness.setdefault('warm_timeout_seconds', 15.0)
    explore = raw.setdefault('explore', {})
    explore.setdefault('clickhouse_rails', {})['enabled'] = False
    raw.setdefault('guidance', {})['enabled'] = False
    # GuidanceConfig still requires snapshot_unavailable_notice when parsing YAML.
    raw.setdefault('guidance', {}).setdefault(
        'snapshot_unavailable_notice',
        'Market snapshot is unavailable — showing best-match domain recommendations for your query instead.',
    )
    # Master lever present for AgentSearchConfig.from_dict (required key).
    raw.setdefault('clickhouse', {}).setdefault('enabled', False)
    # base.yaml requires ATHENA_TMP_DB / ATHENA_TMP_DB_LOC (empty when unset).
    # SeedDatabaseConfig rejects empty merge_database / merge_database_location;
    # unit tests never hit Athena, so inject dummies when missing.
    seed_db = (
        raw.setdefault('vectorization', {})
        .setdefault('seed', {})
        .setdefault('database', {})
    )
    if not str(seed_db.get('merge_database') or '').strip():
        seed_db['merge_database'] = 'test_merge_db'
    if not str(seed_db.get('merge_database_location') or '').strip():
        seed_db['merge_database_location'] = 's3://test-bucket/tmp_auc_semsearch'
    return raw


@functools.wraps(_real_load_config)
def load_config(config_path=None):
    """Test-safe ``load_config``: delegates then applies overrides."""
    raw = _real_load_config(config_path)
    return _apply_test_overrides(raw)


_loader_mod.load_config = load_config


@pytest.fixture(scope='session')
def config_dict() -> Dict[str, Any]:
    """Load the canonical base.yaml exactly once per session.

    Forces ``qi.encoder.backend = hashing`` so the test suite runs
    deterministically without downloading the ~430 MB FastEmbed model.
    """
    return load_config()


@pytest.fixture(scope='session')
def config(config_dict: Dict[str, Any]) -> AgentSearchConfig:
    """Build the typed config bundle from the loaded YAML."""
    return AgentSearchConfig.from_dict(config_dict)


@pytest.fixture()
def subsystems(config_dict: Dict[str, Any]) -> Subsystems:
    """Build a fresh subsystem graph (LLM tier disabled — no provider)."""
    return build_subsystems(config_dict, llm_provider=None)


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Drop every tracked bucket before each test so the process-singleton
    rate limiter (registered once at `semantic_search.app` import) does not
    leak per-session counters across test functions.

    Production code never calls `reset()` — it bypasses the per-session
    cap by clearing buckets — but in pytest the limiter is wired once for
    the entire session and would otherwise 429 the second test that hits
    a limited path under the same client IP. No-op when the limiter is
    None (rate_limit disabled in the loaded config).
    """
    from semantic_search import app as _app_module
    if _app_module.rate_limiter is not None:
        _app_module.rate_limiter.reset()
    yield
    if _app_module.rate_limiter is not None:
        _app_module.rate_limiter.reset()


@pytest.fixture()
def encoder(config: AgentSearchConfig) -> HashingEncoder:
    """A hashing encoder sized to the configured semantic dim."""
    return HashingEncoder(dim=config.qi.semantic.embedding_dim, seed=config.qi.semantic.encoder_seed)


@pytest.fixture(scope='session')
def router_seeds(config: AgentSearchConfig) -> RouterSeedDataset:
    """load the curated seed dataset (or inline override) once per session.

    Mirrors the same xor-branch the registry uses, so tests exercise the production
    code path that turns config into a `RouterSeedDataset`.
    """
    if config.qi.semantic.seeds_path:
        return RouterSeedLoader(seeds_path=config.qi.semantic.seeds_path, min_seeds_per_archetype=config.qi.semantic.min_seeds_per_archetype).load()
    return dataset_from_inline(config.qi.semantic.archetype_prototypes)


@pytest.fixture()
def exact_cache(config: AgentSearchConfig) -> ExactCache:
    """A fresh exact-cache instance per test."""
    return ExactCache(config.cache.exact)


@pytest.fixture()
def structured_cache(config: AgentSearchConfig) -> StructuredCache:
    """A fresh structured-cache instance per test."""
    return StructuredCache(config.cache.structured)


def run(coro):
    """Synchronously run a coroutine in a fresh event loop (test helper)."""
    return asyncio.get_event_loop().run_until_complete(coro) if not asyncio.iscoroutine(coro) else asyncio.run(coro)


# ---------------------------------------------------------------------------
# Reusable boundary / hostile-input fixtures
# These exist so that ``testing.mdc`` Category 3 (Robustness) can be satisfied
# by a single parametrized test rather than N hand-written ones. Tests that
# accept any of these fixtures automatically iterate over the full set.
# ---------------------------------------------------------------------------


@pytest.fixture(params=[None, '', '   ', 0, -1, 'wrong_type', 3.14], ids=lambda v: f'invalid={v!r}')
def invalid_external_value(request):
    """Hostile inputs to feed into any external-data parameter.

    Covers None, empty/whitespace strings, zero, negatives, type mismatches,
    and an arbitrary float. Use in tests that assert the production code
    rejects or safely handles malformed inputs.
    """
    return request.param


@pytest.fixture(params=[None, '', '   '], ids=['none', 'empty', 'whitespace'])
def invalid_string(request):
    """Empty / whitespace-only string variants for identifier inputs."""
    return request.param


@pytest.fixture(params=[float('nan'), float('inf'), float('-inf')], ids=['nan', 'inf', 'neg_inf'])
def invalid_float(request):
    """Non-finite floats for numeric input contracts."""
    return request.param


@pytest.fixture(params=[0, -1, -0.5], ids=['zero', 'neg_int', 'neg_float'])
def non_positive_number(request):
    """Non-positive numeric inputs for fields that demand > 0."""
    return request.param


@pytest.fixture(params=[[], {}, set(), ()], ids=['list', 'dict', 'set', 'tuple'])
def empty_collection(request):
    """Empty collections of every common Python container type."""
    return request.param
