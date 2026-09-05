"""Construction of the semantic_search subsystem graph from a config dict.
A single `build_subsystems` factory wires:
- Encoder (shared by QI semantic router, vector retriever, semantic cache)
- QI Engine (regex + semantic + optional LLM tier with `LLMCallRouter`)
- Retrievers (vector + structured + SQL — backed by in-memory indexes by default)
- Cache tiers (exact + semantic + structured)
- Signal store (`signal_store.py`) for typed signals consumed by measurement and optional offline consumers
- External eRanker client (Layer 4 ranking port)
- User search history store + Resume service
- A/B bucketer
- SearchOrchestrator + SearchSurface
"""
import contextlib
import csv
import os
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple, Union

from llm_core.logging_utils import mask_path

from semantic_search.analytics.clickhouse_client import ClickHouseClient
from semantic_search.analytics.clickhouse_executor import ClickHouseExecutor
from semantic_search.analytics.mv_router import MVRouter
from semantic_search.analytics.pipeline_router import AnalyticsRouter
from semantic_search.analytics.nl_sql_exact_cache import NLSqlExactCache
from semantic_search.cache.exact_cache import ExactCache
from semantic_search.cache.intent_plan_cache import IntentPlanCache
from semantic_search.cache.intent_result_cache import QIIntentResultCache
from semantic_search.cache.qi_semantic_intent_cache import QISemanticIntentCache
from semantic_search.cache.redis_payload_tier import RedisPayloadTier
from semantic_search.cache.structured_cache import StructuredCache
from semantic_search.calibration import CalibratorRegistry, ProbeRegistry, compute_normalized_entropy, fit_from_golden_seeds, fit_probes_from_golden_seeds
from semantic_search.calibration.persist import compute_fingerprint, load_fits, save_fits
from semantic_search.qi.l0_llm_filter_extractor import L0LLMFilterExtractor
from semantic_search.ingest import InMemoryBidConsumer, InMemoryListingConsumer, SnapshotVersionRegistry
from semantic_search.inventory import PercentileResolver
from semantic_search.config.models import (
    AgentSearchConfig,
    QIEnsembleConfig,
    QIEnsembleVoterConfig,
    QIEnsembleRoutingConfig,
)
from semantic_search.config.clickhouse_lever import apply_clickhouse_lever
from semantic_search.core.exceptions import ConfigurationError, ValidationError
from semantic_search.core.llm_client import LLMCallRouter
from semantic_search.cost.fleet_budget import FleetCostBudget, build_fleet_cost_store
from semantic_search.cost.query_budget import NoOpQueryCostBudget, QueryCostBudget
from semantic_search.core.structural_gate import ModelStructuralCapabilityRegistry
from semantic_search.core.llm_provider import LLMProvider
from semantic_search.core.logging_utils import get_logger
from semantic_search.calibration.seed_loader import load_calibration_boot_cases
from semantic_search.eval.llm_relevance_judge import LLMJudgedQueryBuilder, LLMRelevanceJudge
from semantic_search.eval.retrieval_quality import RetrievalQualityEvaluator
from semantic_search.explore.ch_clickhouse_sources import (
    ClickHouseEndingSoonExploreSource,
    ClickHouseFreshExploreSource,
    ClickHouseHighTrafficExploreSource,
    ClickHouseHighVolumeExploreSource,
    ClickHouseLastHourExploreSource,
    ClickHouseLastWeekExploreSource,
    ClickHouseLatestExploreSource,
    ClickHouseTrendingExploreSource,
    ClickHouseWatchDensityExploreSource,
)
from semantic_search.explore.ch_seed_writer import insert_seed_to_clickhouse
from semantic_search.explore.composer import ExploreComposer
from semantic_search.explore.sources import InMemoryEndingSoonSource, InMemoryTrendingSource
from semantic_search.explore.zero_result_guard import ZeroResultGuard
from semantic_search.guidance.guidance_service import GuidanceService
from semantic_search.contracts import FeedbackSignal, infer_breaker_transition_fallback_kind
from semantic_search.signal_store import SignalStore, schedule_feedback_signal_record
from semantic_search.safety.layer_zero_sanitizer import LayerZeroSanitizer
from semantic_search.history.compactor import HistoryCompactor, HistoryCompactorDriver, InMemoryUserFeatureVectorStore, UserFeatureVectorStore
from semantic_search.history.store import UserSearchHistoryStore
from semantic_search.measurement.cache_miss_storm import CacheMissStormDetector
from semantic_search.measurement.evaluator import ProxySignalEvaluator
from semantic_search.measurement.store import MeasurementStore
from semantic_search.nl_to_sql.athena_client import AthenaClient
from semantic_search.nl_to_sql.content_sanitizer import RetrievedContentSanitizer
from semantic_search.nl_to_sql.executor import SqlExecutor
from semantic_search.nl_to_sql.generator import SqlGenerator
from semantic_search.nl_to_sql.cost_class import CostClassifier
from semantic_search.nl_to_sql.logic_validator import ExplainProbe, LogicValidator, make_clickhouse_explain_probe
from semantic_search.config.nl_to_sql_models import SqlExecutionConfig
from semantic_search.nl_to_sql.pipeline import NLToSQLPipeline
from semantic_search.nl_to_sql.schema import JsonSchemaCatalog, SchemaDiscoverer
from semantic_search.nl_to_sql.security import AstSecurityValidator
from semantic_search.nl_to_sql.verifier import Verifier
from semantic_search.vectorization.db_seed_source import fetch_seed_pages_from_db
from semantic_search.orchestrator import SearchOrchestrator
from semantic_search.retrieval.eranker_client import ERankerClient, build_eranker_client
from semantic_search.qi.cascade_encoder import MatryoshkaCascadeEncoder
from semantic_search.qi.stage_encoder import StageEncoder
from semantic_search.qi.centroid_retrainer import CentroidRetrainer
from semantic_search.qi.centroid_retrainer_driver import CentroidRetrainerDriver
from semantic_search.qi.encoder import BatchingEncoder, Encoder, FastEmbedEncoder, HashingEncoder, PrefixedEncoder
from semantic_search.qi.aggregation_intent_gate import AggregationIntentGate
from semantic_search.qi.ngram_pre_gate import NgramPreGate
from semantic_search.qi.engine import QIEngine, _FILTER_SIGNAL_RE
from semantic_search.qi.ensemble_resolver import EnsembleResolver
from semantic_search.qi.entity_type_voter import EntityTypeVoter
from semantic_search.qi.vague_quantifier_resolver import VagueQuantifierResolver
from semantic_search.qi.term_disambiguator import TermDisambiguator
from semantic_search.qi.grounding import (
    ClickHouseInventoryContract,
    ClickHouseTLDRefreshDriver,
    EntityGrounder,
    InventoryContract,
    LiveInventoryContract,
    StaticInventoryContract,
)
from semantic_search.qi.llm_classifier import LLMClassifier
from semantic_search.qi.multi_intent_splitter import MultiIntentSplitter
from semantic_search.qi.regex_entity_extractor import RegexEntityExtractor
from semantic_search.qi.seed_loader import RouterSeedLoader, dataset_from_inline
from semantic_search.qi.semantic_router import SemanticRouter
from semantic_search.qi.spell_corrector import SymSpellCorrector
from semantic_search.qi.query_transformer import QueryTransformer
from semantic_search.resilience.circuit_breaker import CircuitBreaker
from semantic_search.resilience.degradation import DegradationPlanner
from semantic_search.resilience.health import BackendHealthRegistry
from semantic_search.retrieval.base import Retriever
from semantic_search.retrieval.diversifier_base import Diversifier
from semantic_search.retrieval.fusion import RRFFuser
from semantic_search.retrieval.lexical_jaccard_diversifier import LexicalJaccardDiversifier, NoOpDiversifier
from semantic_search.retrieval.qdrant_adapter import QdrantClientFactory, QdrantHybridRetriever, QdrantNoOpStructuredRetriever, QdrantStructuredIndex, QdrantVectorIndex
from semantic_search.retrieval.query_compound_expander import QueryCompoundExpander
from semantic_search.retrieval.clickhouse_price_band_store import ClickHousePriceBandStore
from semantic_search.retrieval.sql_retriever import InMemoryPriceBandStore, PriceBandStore, SqlRetriever
from semantic_search.retrieval.structured_retriever import InMemoryStructuredIndex, StructuredIndex, StructuredRetriever
from semantic_search.retrieval.vector_retriever import InMemoryVectorIndex, VectorIndex, VectorRetriever
from semantic_search.safety.egress_guard import EgressGuard, NoOpEgressGuard
from semantic_search.safety.lexical_blocklist_moderator import LexicalBlocklistModerator, Moderator, NoOpModerator
from semantic_search.vectorization import BM25DocEncoder, CompoundWordSplitter, DeltaRefreshDriver, DocVectorizationPipeline, DomainNameSegmenter, EnrichmentRefreshDriver, EventIngestDriver, OfflineIndexer, VectorRefreshDriver  # noqa: E501
from semantic_search.retrieval.dynamic_synonym_store import DynamicSynonymStore
from semantic_search.retrieval.synonym_expander import SynonymExpander
from semantic_search.vectorization.boot_loader import load_seed_into_indexes
from semantic_search.vectorization.stage_timing import StageTimingSession
from semantic_search.analytics.snapshot_port import HistoricalSnapshotAnalyticsPort
from semantic_search.analytics.domain_analytics_engine import DomainAnalyticsEngine
from semantic_search.analytics.engine_dispatcher import AnalyticsEngineDispatcher
from semantic_search.analytics.enriched_tables_builder import EnrichedTablesBuilder
from semantic_search.middleware import SlidingWindowRateLimiter
from semantic_search.retrieval.bm25_query_encoder import BM25QueryEncoder
from semantic_search.retrieval.bm42_sparse_encoder import BM42SparseEncoder
from semantic_search.retrieval.char_ngram_sparse_encoder import CharNgramSparseEncoder
import redis as _redis_mod
from llm_core.pricing import load_pricing

logger = get_logger(__name__)


def _load_compound_splitter_dictionary(path: str) -> Dict[str, float]:
    """Load unigram frequency dict from CSV (word,frequency format). Validated to workspace root; aggregates duplicates."""
    if not isinstance(path, str) or not path.strip():
        raise ConfigurationError("compound_splitter dictionary path must be a non-empty string")
    expanded = os.path.expanduser(path.strip())
    abs_path = os.path.abspath(expanded)
    _here = Path(__file__).resolve()
    workspace_root = str(_here)
    for _ancestor in _here.parents:
        if (_ancestor / ".git").exists() or (_ancestor / ".cursor").exists():
            workspace_root = str(_ancestor)
            break
    if not abs_path.startswith(workspace_root + os.sep) and abs_path != workspace_root:
        raise ConfigurationError(
            f"compound_splitter dictionary path must be inside the workspace; got {path}"
        )
    if not os.path.isfile(abs_path):
        raise ConfigurationError(f"compound_splitter dictionary not found at {path}")
    out: Dict[str, float] = {}
    with open(abs_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        for row_no, row in enumerate(reader, start=1):
            if not row:
                continue
            first = str(row[0]).strip()
            if not first or first.startswith("#"):
                continue
            if first.casefold() == "word":
                continue
            if len(row) < 2:
                raise ConfigurationError(
                    f"compound_splitter dictionary row {row_no} requires word,frequency"
                )
            try:
                freq = float(row[1])
            except (TypeError, ValueError) as e:
                raise ConfigurationError(
                    f"compound_splitter dictionary row {row_no} frequency must be numeric: {row[1]!r}"
                ) from e
            out[first.casefold()] = out.get(first.casefold(), 0.0) + freq
    if not out:
        raise ConfigurationError(f"compound_splitter dictionary at {path} is empty")
    return out


@dataclass
class Subsystems:
    """Container for all initialized subsystems. Backends selected per config; qdrant_factory is None when unavailable."""
    config: AgentSearchConfig
    encoder: Encoder
    qi_engine: QIEngine
    vector_index: VectorIndex
    structured_index: StructuredIndex
    price_band_store: PriceBandStore
    vector_retriever: Retriever
    structured_retriever: Retriever
    sql_retriever: SqlRetriever
    qdrant_factory: Optional[QdrantClientFactory]
    fuser: RRFFuser
    eranker_client: ERankerClient
    # Top-N latency-gated MMR diversifier (NoOpDiversifier when disabled).
    diversifier: Diversifier
    # Output-side egress guard (NoOpEgressGuard when disabled).
    egress_guard: Union[EgressGuard, NoOpEgressGuard]
    exact_cache: ExactCache
    structured_cache: StructuredCache
    intent_plan_cache: IntentPlanCache
    redis_payload_tier: Optional[RedisPayloadTier]
    snapshot_registry: SnapshotVersionRegistry
    listing_consumer: InMemoryListingConsumer
    bid_consumer: InMemoryBidConsumer
    inventory_resolver: PercentileResolver
    calibrator_registry: CalibratorRegistry
    signal_store: SignalStore
    history_store: UserSearchHistoryStore
    # 90-day raw -> aggregated compactor. Always built when history.enabled; driver is None when compactor.enabled=false.
    history_vector_store: Optional[UserFeatureVectorStore]
    history_compactor: Optional[HistoryCompactor]
    history_compactor_driver: Optional[HistoryCompactorDriver]
    sanitizer: LayerZeroSanitizer
    explore_trending_source: InMemoryTrendingSource
    explore_ending_soon_source: InMemoryEndingSoonSource
    explore_composer: ExploreComposer
    zero_result_guard: ZeroResultGuard
    # None when ``offline_eval.retrieval_eval.enabled=false``. The evaluator
    # measures NDCG@k / Recall@k / MRR@k of orchestrator output against a
    # labeled RelevanceJudgedQuery set (offline relevance eval). Library-only.
    retrieval_quality_evaluator: Optional[RetrievalQualityEvaluator]
    circuit_breaker: CircuitBreaker
    backend_health: BackendHealthRegistry
    degradation_planner: DegradationPlanner
    measurement_store: MeasurementStore
    proxy_signal_evaluator: ProxySignalEvaluator
    # Cache miss-storm alarm (None when omitted in YAML). Emits FeedbackSignal on breach.
    cache_miss_storm_detector: Optional['CacheMissStormDetector']
    orchestrator: SearchOrchestrator
    llm_provider: Optional[LLMProvider]
    nl_to_sql_pipeline: Optional[NLToSQLPipeline]
    analytics_router: Optional[AnalyticsRouter]
    enriched_tables_builder: Optional['EnrichedTablesBuilder']
    guidance_service: Optional[GuidanceService]
    # Structural gate (None when llm_structural_gate.enabled=false).
    structural_gate: Optional[ModelStructuralCapabilityRegistry]
    # Shared LLM call router (None when LLM provider init failed).
    call_router: Optional[LLMCallRouter]
    # Semantic router for MRL cascade tests.
    semantic_router: Optional['SemanticRouter']
    # Tier-0 spell-corrector (None when disabled; read-only at runtime).
    spell_corrector: Optional[SymSpellCorrector]
    # DistilBERT query transformer (None when disabled). Condenses or expands queries before QI.
    query_transformer: Optional[QueryTransformer]
    # LLM-as-judge for relevance grading (None when disabled or no call_router).
    llm_relevance_judge: Optional[LLMRelevanceJudge]
    llm_judged_query_builder: Optional[LLMJudgedQueryBuilder]
    # Doc-vectorization pipeline + indexer + refresh driver (None when disabled or vocab mismatch).
    doc_vectorization_pipeline: Optional[DocVectorizationPipeline]
    offline_indexer: Optional[OfflineIndexer]
    vector_refresh_driver: Optional[VectorRefreshDriver]
    # Real-time delta refresh driver (None when absent/disabled or Athena unavailable).
    delta_refresh_driver: Optional[DeltaRefreshDriver]
    # Real-time bid + watch event ingest driver (None when absent/disabled or Athena unavailable).
    event_ingest_driver: Optional[EventIngestDriver]
    # Periodic seed-time enrichment refresh driver, Qdrant-only (None when absent/disabled or Athena/Qdrant unavailable).
    enrichment_refresh_driver: Optional[EnrichmentRefreshDriver]
    # Matryoshka embedding-cascade wrapper (None when disabled).
    cascade_encoder: Optional[MatryoshkaCascadeEncoder]
    # Per-stage cascade encoders for MRL dim control (router/shortlist/rerank stages).
    router_encoder: Optional[StageEncoder]
    shortlist_encoder: Optional[StageEncoder]
    rerank_encoder: Optional[StageEncoder]
    # Tier-2 centroid-retrainer for L1 semantic router (None when disabled).
    centroid_retrainer: Optional[CentroidRetrainer]
    # Background driver for centroid retrain cycles (None when parent retrainer is None).
    centroid_retrainer_driver: Optional[CentroidRetrainerDriver]
    # Background driver for ClickHouse TLD refresh (None when CH refresh disabled or unavailable).
    ch_tld_refresh_driver: Optional['ClickHouseTLDRefreshDriver'] = None
    # Document-side encoder with query_prefix for task separation (None when vectorization disabled).
    indexing_encoder: Optional[Encoder] = None
    # Query-driven dynamic synonym store (populated when BM25 enabled).
    dynamic_synonym_store: Optional[DynamicSynonymStore] = None
    # Deferred calibration fit callable (runs in background thread on startup).
    calibration_boot_fit_fn: Optional[Callable[[], None]] = None


def build_subsystems(config_dict: Dict[str, Any], llm_provider: Optional[LLMProvider]) -> Subsystems:
    """Build full subsystem graph (encoder -> routers -> indexes -> orchestrator)."""
    config = apply_clickhouse_lever(AgentSearchConfig.from_dict(config_dict))
    encoder = _build_encoder(config)
    # Optional Matryoshka embedding-cascade wrapper. Constructed only when
    # ``qi.encoder.cascade`` is present in YAML AND ``enabled=true``. When
    # ``require_native_cascade=true`` (production default) the wrapper is
    # only built when the base encoder is a ``FastEmbedEncoder`` — a
    # ``HashingEncoder`` base in production would silently collapse the
    # cascade so the registry refuses the wiring loudly. In test mode
    # (``require_native_cascade=false``) the cascade is allowed to
    # collapse onto any encoder; consumers see one warning per requested
    # dim mismatch.
    cascade_encoder: Optional[MatryoshkaCascadeEncoder] = None
    cascade_cfg = config.qi.encoder.cascade
    if cascade_cfg is not None and cascade_cfg.enabled:
        if cascade_cfg.require_native_cascade and not isinstance(encoder, FastEmbedEncoder):
            logger.warning(
                f"matryoshka_cascade_disabled reason=base_encoder_not_native "
                f"base={type(encoder).__name__} require_native_cascade=true"
            )
        else:
            try:
                cascade_encoder = MatryoshkaCascadeEncoder(
                    base_encoder=encoder,
                    supported_dims=frozenset(cascade_cfg.supported_dims),
                )
                logger.info(
                    f"matryoshka_cascade_built base={type(encoder).__name__} "
                    f"supported_dims={sorted(cascade_cfg.supported_dims)} "
                    f"native_cascade={cascade_encoder.is_native_cascade}"
                )
            except (ValidationError, ConfigurationError) as e:
                logger.warning(
                    f"matryoshka_cascade_disabled reason=construction_failed "
                    f"error_type={type(e).__name__} error={str(e)}"
                )
                cascade_encoder = None
    # Per-stage cascade encoders. Validates cross-config dims early; rerank_encoder built for forward-compat.
    router_encoder: Optional[StageEncoder] = None
    shortlist_encoder: Optional[StageEncoder] = None
    rerank_encoder: Optional[StageEncoder] = None
    if cascade_encoder is not None and cascade_cfg is not None and cascade_cfg.stage_dims is not None:
        stage_dims = cascade_cfg.stage_dims
        if stage_dims.router is not None:
            if stage_dims.router != config.qi.semantic.embedding_dim:
                raise ConfigurationError(
                    f"qi.encoder.cascade.stage_dims.router={stage_dims.router} != "
                    f"qi.semantic.embedding_dim={config.qi.semantic.embedding_dim}; "
                    f"the SemanticRouter's encoder.dim check would reject the wired StageEncoder. "
                    f"Set the two dims to the same value (typically 256 for the router stage)."
                )
            router_encoder = StageEncoder(cascade_encoder, stage_dims.router, 'router', config.general.startup_log_detail)
        if stage_dims.shortlist is not None:
            if stage_dims.shortlist != config.retrieval.vector.embedding_dim:
                raise ConfigurationError(
                    f"qi.encoder.cascade.stage_dims.shortlist={stage_dims.shortlist} != "
                    f"retrieval.vector.embedding_dim={config.retrieval.vector.embedding_dim}; "
                    f"the VectorRetriever / QdrantHybridRetriever encoder.dim check would reject "
                    f"the wired StageEncoder. Set the two dims to the same value "
                    f"(typically 512 for the shortlist stage)."
                )
            shortlist_encoder = StageEncoder(cascade_encoder, stage_dims.shortlist, 'shortlist', config.general.startup_log_detail)
        if stage_dims.rerank is not None:
            # No live consumer for the rerank head today (the lexical reranker
            # does not consume embeddings). The seam is still wired so a future
            # cross-encoder reranker can pick it up via Subsystems.rerank_encoder
            # without revisiting registry construction. Validation against
            # supported_dims already ran in CascadeEncoderConfig.__post_init__.
            rerank_encoder = StageEncoder(cascade_encoder, stage_dims.rerank, 'rerank', config.general.startup_log_detail)
        logger.info(
            f"stage_encoders_wired router={'on' if router_encoder else 'off'}:"
            f"{stage_dims.router} shortlist={'on' if shortlist_encoder else 'off'}:"
            f"{stage_dims.shortlist} rerank={'on' if rerank_encoder else 'off'}:"
            f"{stage_dims.rerank}"
        )
    # Build order: encoder -> seeds -> router -> resilience -> call_router -> llm -> grounder -> backends -> caches -> engine.
    if config.qi.semantic.seeds_path:
        router_seeds = RouterSeedLoader(
            seeds_path=config.qi.semantic.seeds_path,
            min_seeds_per_archetype=config.qi.semantic.min_seeds_per_archetype,
        ).load()
    else:
        router_seeds = dataset_from_inline(config.qi.semantic.archetype_prototypes)
    # Cascade: hand router the stage encoder if wired; validation ensures dims match.
    semantic_router_encoder: Encoder = router_encoder if router_encoder is not None else encoder
    semantic_router = SemanticRouter(config.qi.semantic, semantic_router_encoder, router_seeds)
    # Tier-0 spell-corrector from same seed dataset. Soft-fails to None on construction error; search remains available.
    spell_corrector: Optional[SymSpellCorrector] = None
    if config.qi.spell_correct is not None and config.qi.spell_correct.enabled:
        try:
            spell_corrector = SymSpellCorrector(config=config.qi.spell_correct)
            logger.info(
                f"spell_corrector_built auto_apply={config.qi.spell_correct.auto_apply}"
            )
        except (ValidationError, ConfigurationError) as e:
            logger.warning(
                f"spell_corrector_build_failed reason={type(e).__name__} error={str(e)} "
                f"falling_back_to=disabled"
            )
            spell_corrector = None
    # SignalStore built before resilience primitives; passive recorder with no dependencies.
    signal_store = SignalStore(config.feedback)

    def _make_breaker_listener(breaker_name: str) -> Callable[[str, str, str, int], None]:
        """Build listener that emits breaker_transition FeedbackSignal."""
        def _emit(from_state: str, to_state: str, reason: str, observation_count: int) -> None:
            try:
                fk = infer_breaker_transition_fallback_kind(breaker_name, str(from_state), str(to_state), str(reason))
                payload = {
                    'breaker': breaker_name,
                    'from_state': str(from_state),
                    'to_state': str(to_state),
                    'reason': str(reason),
                    'observation_count': int(observation_count),
                    'fallback_kind': fk,
                }
                sig = FeedbackSignal(
                    signal_id=FeedbackSignal.new_signal_id(),
                    request_id=f"breaker_transition-{uuid.uuid4().hex[:12]}",
                    signal_type='breaker_transition',
                    payload=payload,
                    signal_origin='circuit_breaker',
                )
                schedule_feedback_signal_record(signal_store, sig)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"breaker_transition_signal_emit_failed breaker={breaker_name} error_type={type(e).__name__} error={str(e)}")
        return _emit

    def _backend_health_listener(backend: str, from_state: str, to_state: str, observation_count: int) -> None:
        """Listener for `BackendHealthRegistry` (per-backend signature)."""
        _make_breaker_listener(backend)(from_state, to_state, 'backend_health_transition', observation_count)

    # Build resilience before LLM router so it can consult health state.
    backend_health = BackendHealthRegistry(config.resilience.backend_health, transition_listener=_backend_health_listener)
    circuit_breaker = CircuitBreaker(config.resilience.circuit_breaker, transition_listener=_make_breaker_listener('llm'))
    degradation_planner = DegradationPlanner(config=config.resilience.degradation, health=backend_health)
    # Structural gate (None when disabled). Built before LLMCallRouter so router can consult it.
    structural_gate: Optional[ModelStructuralCapabilityRegistry] = None
    if config.llm_structural_gate.enabled:
        structural_gate = ModelStructuralCapabilityRegistry(
            ttl_seconds=config.llm_structural_gate.probe_ttl_seconds,
            probe_timeout_seconds=config.llm_structural_gate.probe_timeout_seconds,
            probe_max_tokens=config.llm_structural_gate.probe_max_tokens,
            probe_system_prompt=config.llm_structural_gate.probe_system_prompt,
            probe_user_prompt=config.llm_structural_gate.probe_user_prompt,
            treat_unknown_as_capable=config.llm_structural_gate.treat_unknown_as_capable,
        )
    # L0 sanitizer built before LLMCallRouter; reused for ingress, retrieved-content, and refinement paths.
    sanitizer = LayerZeroSanitizer(config.safety.ingress_sanitizer)
    call_router: Optional[LLMCallRouter] = None
    if llm_provider is not None and llm_provider.get_default_client() is not None:
        token_pricing_raw = config_dict.get('llm_token_pricing')
        token_pricing = dict(token_pricing_raw) if token_pricing_raw else dict(load_pricing())
        if 'default' not in token_pricing:
            raise ConfigurationError("llm_token_pricing must include a 'default' entry with input/output rates")
        default_pricing = token_pricing['default']
        call_router = LLMCallRouter(
            provider=llm_provider,
            max_tokens=int(config_dict['llm_models']['max_tokens']),
            temperature=float(config_dict['llm_models']['temperature']),
            token_warn_threshold=int(config_dict['llm_models']['token_warn_threshold']),
            token_pricing=token_pricing,
            default_pricing=default_pricing,
            circuit_breaker=circuit_breaker,
            health_registry=backend_health,
            structural_gate=structural_gate,
            sanitizer=sanitizer,
        )
    # Reuse signal_store built earlier; circuit-breaker and classifier emit to same instance.
    llm_classifier: Optional[LLMClassifier] = None
    if config.qi.llm.enabled and call_router is not None:
        llm_classifier = LLMClassifier(
            config=config.qi.llm,
            call_router=call_router,
            allowed_query_types=config.qi.query_types,
            alt_band_high=float(config.qi.llm.alternative_band_high),
            max_alternatives=int(config.qi.llm.max_alternative_interpretations),
            signal_store=signal_store,
        )
    else:
        logger.warning("qi_llm_tier_disabled reason=no_provider_or_disabled_in_config")
    # QueryTransformer: LLM-primary rewriter (via call_router) with a local seq2seq
    # fallback. Built after call_router so the LLM tier is wired; call_router may be
    # None (LLM tier disabled -> local fallback only). Soft-fails to None on build error.
    query_transformer: Optional[QueryTransformer] = None
    _qt_cfg = config.qi.query_transformer
    if _qt_cfg is not None and _qt_cfg.enabled and _qt_cfg.rewrite_enabled:
        try:
            query_transformer = QueryTransformer(config=_qt_cfg, call_router=call_router)
        except (ValidationError, ConfigurationError, OSError, RuntimeError, ImportError) as e:
            logger.warning(f"query_transformer_build_failed reason={type(e).__name__} error={str(e)} falling_back_to=disabled")
            query_transformer = None
    else:
        logger.info("query_transformer_not_configured — query rewrite feature off")
    multi_intent_splitter = MultiIntentSplitter(config=config.multi_intent)
    # Plain DomainNameSegmenter (no dict) for word-count filtering; compound splitter added in vectorization block.
    _word_segmenter = DomainNameSegmenter(compound_splitter=None)
    qdrant_factory, vector_index, structured_index, vector_retriever, structured_retriever, _bm25_dynamic_store = _build_retrieval_backends(
        config=config, encoder=encoder, shortlist_encoder=shortlist_encoder, rerank_encoder=rerank_encoder,
        word_segmenter=_word_segmenter,
    )
    # Entity grounding: static contract fallback; LiveInventory wired when ttl > 0. TLD-subs enable adapt-on-miss.
    static_inventory = StaticInventoryContract(
        tlds=config.qi.regex.known_tlds,
        auction_types=config.qi.regex.known_auction_types,
    )
    if config.qi.regex.live_inventory_ttl_seconds > 0:
        inventory: 'InventoryContract' = LiveInventoryContract(
            payload_source=structured_index.iter_payloads,
            fallback=static_inventory,
            ttl_seconds=config.qi.regex.live_inventory_ttl_seconds,
        )
    else:
        inventory = static_inventory
    _pending_ch_inventory: Optional[ClickHouseInventoryContract] = None
    if (
        config.qi.regex.ch_tld_refresh_enabled
        and config.nl_to_sql.analytics is not None
        and bool(getattr(config.clickhouse, 'enabled', False))
        and config.qi.regex.ch_tld_refresh_interval_seconds > 0
        and config.qi.regex.ch_tld_lookback_days > 0
        and config.qi.regex.ch_tld_query_sql
    ):
        _pending_ch_inventory = ClickHouseInventoryContract(fallback=inventory)
        inventory = _pending_ch_inventory
        logger.info(
            f"ch_tld_inventory_contract_wired lookback_days={config.qi.regex.ch_tld_lookback_days} "
            f"interval_seconds={config.qi.regex.ch_tld_refresh_interval_seconds}"
        )
    grounder = EntityGrounder(
        inventory=inventory,
        tld_substitutions=config.qi.regex.tld_substitutions,
    )
    price_band_store = _build_price_band_store(config=config, backend_health=backend_health)
    sql_retriever = SqlRetriever(config=config.retrieval.sql, store=price_band_store)
    fuser = RRFFuser(config=config.retrieval.fusion)
    eranker_client = build_eranker_client(config.retrieval.eranker)
    # Top-N latency-gated MMR diversifier (NoOpDiversifier on disabled, LexicalJaccardDiversifier default).
    diversity_cfg = config.retrieval.diversity
    if not diversity_cfg.enabled or diversity_cfg.backend == 'noop':
        diversifier = NoOpDiversifier()
    elif diversity_cfg.backend == 'lexical_jaccard_mmr':
        # `__post_init__` on DiversityConfig already guarantees `lexical` is
        # non-None when enabled+lexical_jaccard_mmr; assert for type narrowing.
        assert diversity_cfg.lexical is not None
        diversifier = LexicalJaccardDiversifier(diversity_cfg.lexical)
    else:
        raise ConfigurationError(
            f"retrieval.diversity.backend={diversity_cfg.backend!r} is not wired in registry; "
            "supported backends: lexical_jaccard_mmr, noop"
        )
    # Ingest subsystem: shared snapshot registry + in-memory consumers. Version-tagged cache entries bust on advance.
    snapshot_registry = SnapshotVersionRegistry()
    exact_cache = ExactCache(config.cache.exact, snapshot_version_fn=lambda: snapshot_registry.version)
    structured_cache = StructuredCache(config.cache.structured, snapshot_version_provider=lambda: snapshot_registry.version)
    intent_plan_cache = IntentPlanCache(config.cache.intent_plan, snapshot_version_fn=lambda: snapshot_registry.version)
    listing_consumer = InMemoryListingConsumer(config=config.ingest, registry=snapshot_registry)
    bid_consumer = InMemoryBidConsumer(config=config.ingest, registry=snapshot_registry)
    # Inventory percentile resolver scanned from structured index; invalidation hooked to snapshot events.
    inventory_resolver = PercentileResolver(config=config.inventory, index=structured_index)
    snapshot_registry.register(lambda _v: exact_cache.invalidate_all())
    snapshot_registry.register(lambda _v: structured_cache.invalidate_all())
    snapshot_registry.register(lambda _v: intent_plan_cache.invalidate_all())
    redis_payload_tier = RedisPayloadTier.try_connect(config.cache.remote)
    snapshot_registry.register(inventory_resolver.invalidate_on_bump)
    snapshot_registry.register(inventory_resolver.invalidate_market_on_bump)
    # Inject resolver into structured retriever for price_below_market market-baseline reads.
    if hasattr(structured_retriever, 'set_percentile_resolver'):
        structured_retriever.set_percentile_resolver(inventory_resolver)
    # Bust live-inventory TLD cache on ingest so newly-listed domains become groundable immediately.
    if isinstance(inventory, LiveInventoryContract):
        snapshot_registry.register(lambda _v: inventory.invalidate())
        live_inventory_hook_count = 1
    else:
        live_inventory_hook_count = 0
    # L0 entity extractor: same L0LLMFilterExtractor as qie_only + extract-script
    # grounding (single prompt/catalog). None in degraded (no-LLM) mode — QIEngine
    # skips the L0 task and routes via the L0_fallback path (ngram + L1 + aggregation).
    _l0_llm_cfg = config.qi.l0_llm_entity
    l0_entity_extractor: Optional[L0LLMFilterExtractor] = None
    if _l0_llm_cfg is None or not _l0_llm_cfg.enabled or call_router is None:
        logger.warning(
            "qi_l0_entity_extractor_disabled reason=no_call_router_or_disabled — "
            "L0 entity extraction off; QI uses the L0_fallback path."
        )
    else:
        _entity_slots = config.qi.entity_slots
        if _entity_slots is None:
            raise ConfigurationError(
                "qi.entity_slots is required when qi.l0_llm_entity.enabled=true"
            )
        l0_entity_extractor = L0LLMFilterExtractor(
            call_router,
            _l0_llm_cfg.task_type,
            entity_slots=_entity_slots,
            source_tag=_l0_llm_cfg.source_tag,
            max_entities=_l0_llm_cfg.max_entities,
            confidence=_l0_llm_cfg.confidence,
            enabled=_l0_llm_cfg.enabled,
            prompt_tag=_l0_llm_cfg.prompt_tag,
            keyword_min_probability=_l0_llm_cfg.keyword_min_probability,
            combined_prompt_tag=_l0_llm_cfg.combined_prompt_tag,
        )
    # Deterministic regex entity extractor: always available (no call_router dep).
    # Sole entity source in no-LLM mode; pre-empts the LLM extractor on hard slots otherwise.
    _l0_regex_cfg = config.qi.l0_regex_entity
    l0_regex_extractor: Optional[RegexEntityExtractor] = None
    if _l0_regex_cfg is not None and _l0_regex_cfg.enabled:
        if config.qi.entity_slots is None:
            raise ConfigurationError(
                "qi.entity_slots is required when qi.l0_regex_entity.enabled=true"
            )
        l0_regex_extractor = RegexEntityExtractor(
            _l0_regex_cfg,
            hard_entity_names=config.qi.entity_slots.hard_entity_set,
            soft_slot_names=config.qi.entity_slots.soft_slot_set,
            known_tlds=config.qi.regex.known_tlds,
        )
    if l0_regex_extractor is None:
        logger.warning("qi_l0_regex_extractor_disabled reason=absent_or_disabled — offline filter extraction off")
    # Per-tier calibrator registry boots with identity T=1.0; fit driver swaps in fitted Ts atomically.
    calibrator_registry = CalibratorRegistry(
        tier_keys=config.calibration.tier_keys,
        hot_swap_min_samples=config.calibration.hot_swap_min_samples,
        startup_log_detail=config.general.startup_log_detail,
    )
    qi_intent_result_cache: Optional[QIIntentResultCache] = None
    if config.qi.intent_result_cache is not None and config.qi.intent_result_cache.enabled:
        qi_intent_result_cache = QIIntentResultCache(config.qi.intent_result_cache)
    qi_semantic_intent_cache: Optional[QISemanticIntentCache] = None
    if config.qi.semantic_intent_cache is not None and config.qi.semantic_intent_cache.enabled:
        qi_semantic_intent_cache = QISemanticIntentCache(config.qi.semantic_intent_cache, semantic_router_encoder)
    aggregation_gate: Optional[AggregationIntentGate] = None
    _agg_cfg = config.qi.regex.aggregation_gate
    if _agg_cfg is not None and _agg_cfg.enabled:
        try:
            aggregation_gate = AggregationIntentGate(_agg_cfg)
            logger.info(f"aggregation_intent_gate_enabled nouns={len(_agg_cfg.marketplace_nouns)} strong_ops={len(_agg_cfg.strong_operators)} weak_ops={len(_agg_cfg.weak_operators)} companions={len(_agg_cfg.weak_operator_companions)}")  # noqa: E501
        except Exception as _agg_err:  # noqa: BLE001
            logger.warning(f"aggregation_intent_gate_disabled reason=construction_failed error_type={type(_agg_err).__name__} error={_agg_err}")
            aggregation_gate = None
    ngram_pre_gate: Optional[NgramPreGate] = None
    _ng_cfg = config.qi.regex.ngram_pre_gate
    if _ng_cfg is not None and _ng_cfg.enabled:
        try:
            ngram_pre_gate = NgramPreGate(_ng_cfg)
            logger.info(f"ngram_pre_gate_enabled vocab_size={len(ngram_pre_gate._weights)} threshold={_ng_cfg.confidence_threshold} order={_ng_cfg.max_ngram_order}")
        except Exception as _ng_err:  # noqa: BLE001
            logger.warning(f"ngram_pre_gate_disabled reason=construction_failed error_type={type(_ng_err).__name__} error={_ng_err}")
            ngram_pre_gate = None
    entity_type_voter: Optional[EntityTypeVoter] = None
    if config.qi.entity_voter is not None:
        try:
            entity_type_voter = EntityTypeVoter(config.qi.entity_voter, _FILTER_SIGNAL_RE)
            logger.info(f"entity_type_voter_enabled voter_id={config.qi.entity_voter.voter_id}")
        except Exception as _ev_err:  # noqa: BLE001
            logger.warning(f"entity_type_voter_disabled reason=construction_failed error_type={type(_ev_err).__name__} error={_ev_err}")
            entity_type_voter = None
    # Ensemble resolver — voter set derived from successfully-built components.
    _ens_voters: list = []
    if aggregation_gate is not None:
        _ens_voters.append(QIEnsembleVoterConfig(voter_id='aggregation_gate', weight=3.0, has_veto=True,  abstain_on_no_signal=True,  timeout_ms=0))
    if ngram_pre_gate is not None:
        _ens_voters.append(QIEnsembleVoterConfig(voter_id='ngram_gate',        weight=2.0, has_veto=False, abstain_on_no_signal=True,  timeout_ms=0))
    if entity_type_voter is not None:
        _ev_id = config.qi.entity_voter.voter_id if config.qi.entity_voter is not None else 'entity'
        _ens_voters.append(QIEnsembleVoterConfig(voter_id=_ev_id,              weight=2.0, has_veto=False, abstain_on_no_signal=True,  timeout_ms=0))
    _ens_voters.append(    QIEnsembleVoterConfig(voter_id='semantic',           weight=1.5, has_veto=False, abstain_on_no_signal=False, timeout_ms=200))
    if llm_classifier is not None:
        _ens_voters.append(QIEnsembleVoterConfig(voter_id='llm',               weight=1.0, has_veto=False, abstain_on_no_signal=True,  timeout_ms=3000))
    _ens_settings = config.qi.ensemble
    if _ens_settings is None:
        raise ConfigurationError("qi.ensemble is required")
    _ens_cfg = QIEnsembleConfig(
        voters=_ens_voters,
        routing=_ens_settings.routing,
        consensus_cancel_l2=bool(_ens_settings.consensus_cancel_l2),
        consensus_cancel_threshold=float(_ens_settings.consensus_cancel_threshold),
        extract_before_classify=bool(_ens_settings.extract_before_classify),
        fallback_archetype=config.qi.default_query_type,
    )
    ensemble_resolver: Optional[EnsembleResolver] = None
    try:
        ensemble_resolver = EnsembleResolver(_ens_cfg, frozenset(config.qi.query_types))
        logger.info(f"ensemble_resolver_enabled voters={[v.voter_id for v in _ens_cfg.voters]}")
    except Exception as _ens_err:  # noqa: BLE001
        logger.warning(f"ensemble_resolver_failed error_type={type(_ens_err).__name__} error={_ens_err}")
    _vague_resolver = (
        VagueQuantifierResolver(config.vague_quantifier)
        if config.vague_quantifier is not None and config.vague_quantifier.enabled
        else None
    )
    # Term disambiguator (soft-fail to None on construction error).
    _td_cfg = config.qi.term_disambiguator
    _term_disambiguator: Optional[TermDisambiguator] = None
    if _td_cfg is not None and _td_cfg.enabled:
        try:
            _term_disambiguator = TermDisambiguator(_td_cfg)
            logger.info(f"term_disambiguator_enabled rules={len(_td_cfg.rules)} context_window={_td_cfg.context_window}")
        except Exception as _td_err:  # noqa: BLE001
            logger.warning(f"term_disambiguator_disabled reason=construction_failed error_type={type(_td_err).__name__} error={_td_err}")
    qi_engine = QIEngine(
        config=config.qi,
        llm_classifier=llm_classifier,
        entity_grounder=grounder,
        max_query_length=config.general.max_query_length,
        circuit_breaker=circuit_breaker,
        multi_intent_config=config.multi_intent,
        multi_intent_splitter=multi_intent_splitter,
        calibrator_registry=calibrator_registry if config.calibration.enabled else None,
        intent_result_cache=qi_intent_result_cache,
        semantic_router=semantic_router,
        semantic_intent_cache=qi_semantic_intent_cache,
        entity_extractor=l0_entity_extractor,
        regex_entity_extractor=l0_regex_extractor,
        aggregation_gate=aggregation_gate,
        vague_quantifier_resolver=_vague_resolver,
        term_disambiguator=_term_disambiguator,
        ngram_pre_gate=ngram_pre_gate,
        ensemble_resolver=ensemble_resolver,
        entity_type_voter=entity_type_voter,
    )
    # Tier-2 centroid retrainer (optional, soft-fails on error). Stateless service; read-only vs. live router.
    centroid_retrainer: Optional[CentroidRetrainer] = None
    if config.qi.centroid_retrainer is not None and config.qi.centroid_retrainer.enabled:
        try:
            centroid_retrainer = CentroidRetrainer(
                config=config.qi.centroid_retrainer,
                encoder=encoder,
                current_router=semantic_router,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"centroid_retrainer_disabled reason=construction_failed "
                f"error_type={type(e).__name__} error={e}"
            )
            centroid_retrainer = None
    centroid_retrainer_driver: Optional[CentroidRetrainerDriver] = None
    if (centroid_retrainer is not None and config.qi.centroid_retrainer_driver is not None and config.qi.centroid_retrainer_driver.enabled):
        try:
            centroid_retrainer_driver = CentroidRetrainerDriver(config=config.qi.centroid_retrainer_driver, retrainer=centroid_retrainer, router=semantic_router, signal_store=signal_store)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"centroid_retrainer_driver_disabled reason=construction_failed error_type={type(e).__name__} error={e}")
            centroid_retrainer_driver = None
    # Reuse signal_store for Tier-3 timeout path. History store + 90-day compactor always built when enabled.
    history_store = UserSearchHistoryStore(config.history)
    history_vector_store: Optional[UserFeatureVectorStore] = None
    history_compactor: Optional[HistoryCompactor] = None
    history_compactor_driver: Optional[HistoryCompactorDriver] = None
    if config.history.enabled and config.history.compactor is not None:
        history_vector_store = InMemoryUserFeatureVectorStore()
        history_compactor = HistoryCompactor(
            history_store=history_store,
            vector_store=history_vector_store,
            config=config.history.compactor,
        )
        if config.history.compactor.enabled:
            history_compactor_driver = HistoryCompactorDriver(
                config=config.history.compactor,
                compactor=history_compactor,
            )
            logger.info(
                f"history_compactor_driver_enabled interval_seconds={config.history.compactor.interval_seconds} "
                f"max_consecutive_failures={config.history.compactor.max_consecutive_failures}"
            )
    # Sanitizer gates LLM ingress + retrieved-content + refinement (single source of truth for counters).
    # Explore subsystem (trending/ending-soon/zero-result guard). Composer shared; response carries source tag.
    explore_trending_source = InMemoryTrendingSource(config.explore.trending)
    explore_ending_soon_source = InMemoryEndingSoonSource(config.explore.ending_soon)
    explore_composer = ExploreComposer(config=config.explore, trending=explore_trending_source, ending_soon=explore_ending_soon_source)
    zero_result_guard = ZeroResultGuard(config=config.explore.zero_result_guard, composer=explore_composer)
    # Deferred calibration fit (runs in background thread to avoid startup block). Registry falls back to T=1.0 until fit completes.
    _calibration_boot_fit_fn: Optional[Callable[[], None]] = None
    if config.calibration.enabled and config.calibration.fit_on_load:
        boot_path = config.calibration.boot_seeds_path
        if boot_path is None:
            logger.warning("calibration_boot_fit skipped reason=boot_seeds_path_unset")
        else:
            cases_for_fit = load_calibration_boot_cases(boot_path)
            if len(cases_for_fit) == 0:
                logger.warning("calibration_boot_fit skipped reason=no_boot_cases")
            else:
                # Capture all locals needed by the fit in the closure so the
                # deferred callable is self-contained (no free variables that
                # could be mutated before the thread starts).
                _fit_cases = cases_for_fit
                _fit_registry = calibrator_registry
                _fit_cal_cfg = config.calibration
                _fit_l0 = l0_entity_extractor
                _fit_router = semantic_router
                _fit_startup_log_detail = config.general.startup_log_detail

                # Load persisted temperature fits (zero LLM) when the fingerprint
                # matches; the live L0 replay in the closure is skipped on a hit.
                _persist_enabled = config.calibration.persist_enabled
                _persist_path = config.calibration.persist_path
                _fit_fingerprint: Optional[str] = None
                _temp_from_cache = False
                if _persist_enabled and _persist_path:
                    _fit_params = {
                        'min_temperature': config.calibration.min_temperature,
                        'max_temperature': config.calibration.max_temperature,
                        'tolerance': config.calibration.tolerance,
                        'max_iterations': config.calibration.max_iterations,
                        'min_samples_per_tier': config.calibration.min_samples_per_tier,
                        'tier_keys': ','.join(config.calibration.tier_keys),
                    }
                    if config.qi.l0_llm_entity is None:
                        raise ConfigurationError(
                            "calibration.persist_enabled requires qi.l0_llm_entity.prompt_tag"
                        )
                    _fit_fingerprint = compute_fingerprint(
                        boot_path,
                        config.qi.l0_llm_entity.prompt_tag,
                        _fit_params,
                        config.calibration.cache_version,
                    )
                    _cached_fits = load_fits(_persist_path, _fit_fingerprint)
                    if _cached_fits:
                        for _cf in _cached_fits.values():
                            try:
                                calibrator_registry.register_fit(_cf)
                            except ValidationError as _cfe:
                                logger.warning(f"calibration_cached_fit_rejected tier={_cf.tier} error={_cfe}")
                        _temp_from_cache = True
                        logger.info(f"calibration_temperature_loaded_from_cache tiers={len(_cached_fits)} skip_live_l0_fit=True")
                _probe_active = config.calibration.probe is not None and config.calibration.probe.enabled

                def _calibration_boot_fit_fn(
                    _cases: List[Dict[str, Any]] = _fit_cases,
                    _registry: CalibratorRegistry = _fit_registry,
                    _cal_cfg: Any = _fit_cal_cfg,
                    _l0: Any = _fit_l0,
                    _router: Any = _fit_router,
                    _startup_log_detail: bool = _fit_startup_log_detail,
                    _temp_cached: bool = _temp_from_cache,
                    _persist_enabled_c: bool = _persist_enabled,
                    _persist_path_c: Optional[str] = _persist_path,
                    _fingerprint_c: Optional[str] = _fit_fingerprint,
                ) -> None:
                    # L0/L1 producers replay same classifiers QI engine uses. LLM tier has no producer (uses traffic labels).
                    def _produce_l0(query: str) -> Optional[Tuple[str, str, float]]:
                        slc = _l0.classify(query)
                        if slc is None:
                            return None
                        return ('L0_entity', slc.query_type, float(slc.confidence))

                    def _produce_l1(query: str) -> Optional[Tuple[str, str, float]]:
                        slc = _router.classify(query)
                        if slc is None:
                            return None
                        return ('L1_semantic', slc.query_type, float(slc.confidence))

                    # L0 producer only when the extractor exists (off in no-LLM mode).
                    producers = {'L1_semantic': _produce_l1}
                    if _l0 is not None:
                        producers['L0_entity'] = _produce_l0
                    fits = fit_from_golden_seeds(_registry, _cases, producers, _cal_cfg)
                    logger.info(
                        f"calibration_boot_fit cases={len(_cases)} tiers_fit={[t for t, f in fits.items() if f.fitted]} tiers_identity={[t for t, f in fits.items() if not f.fitted]}"
                    )
                    # Correctness probe (optional). L0 regex excluded (no distribution).
                    probe_cfg = _cal_cfg.probe
                    if probe_cfg is not None and probe_cfg.enabled:
                        probe_registry = ProbeRegistry(
                            tier_keys=_cal_cfg.tier_keys,
                            alpha=probe_cfg.alpha,
                            hot_swap_min_samples=_cal_cfg.hot_swap_min_samples,
                            startup_log_detail=_startup_log_detail,
                        )

                        def _produce_probe_l1(query: str) -> Optional[Tuple[str, str, float, float]]:
                            result = _router.classify_with_distribution(query)
                            if result is None:
                                return None
                            slc, scored = result
                            if not scored or len(scored) < 2:
                                return None
                            scores_only = [s for _a, s in scored]
                            h_norm = compute_normalized_entropy(scores_only)
                            return ('L1_semantic', slc.query_type, float(slc.confidence), float(h_norm))

                        probe_producers = {'L1_semantic': _produce_probe_l1}
                        probe_fits = fit_probes_from_golden_seeds(
                            probe_registry, _cases, probe_producers, probe_cfg,
                        )
                        _registry.attach_probe_registry(probe_registry, probe_cfg.weight_raw)
                        logger.info(
                            f"calibration_probe_boot_fit cases={len(_cases)} tiers_fit={[t for t, f in probe_fits.items() if f.fitted]} weight_raw={probe_cfg.weight_raw}"
                        )
    # Offline pipeline built when: vectorization enabled + BM25 vocab matches + qdrant available. All None if any precondition fails.
    doc_vectorization_pipeline: Optional[DocVectorizationPipeline] = None
    offline_indexer: Optional[OfflineIndexer] = None
    vector_refresh_driver: Optional[VectorRefreshDriver] = None
    delta_refresh_driver: Optional[DeltaRefreshDriver] = None
    event_ingest_driver: Optional[EventIngestDriver] = None
    enrichment_refresh_driver: Optional[EnrichmentRefreshDriver] = None
    indexing_encoder: Optional[Encoder] = None
    vec_cfg = config.vectorization
    if vec_cfg is not None and vec_cfg.enabled:
        if qdrant_factory is None or not qdrant_factory.available:
            logger.warning(
                "vectorization_pipeline_disabled reason=qdrant_unavailable "
                f"vectorization_enabled={vec_cfg.enabled}"
            )
        elif (
            config.retrieval.qdrant is None
            or config.retrieval.qdrant.hybrid is None
            or (
                config.retrieval.qdrant.hybrid.bm25_query_encoder is None
                and config.retrieval.qdrant.hybrid.sparse_encoder is None
            )
        ):
            logger.warning(
                "vectorization_pipeline_disabled reason=sparse_encoder_absent "
                "vectorization_enabled=true"
            )
        else:
            try:
                _hybrid_cfg = config.retrieval.qdrant.hybrid
                doc_encoder = None
                vocab_size: Optional[int] = None
                if _hybrid_cfg.sparse_encoder is not None:
                    try:
                        doc_encoder = BM42SparseEncoder(
                            model_name=_hybrid_cfg.sparse_encoder.model_name,
                            local_model_path=_hybrid_cfg.sparse_encoder.local_model_path,
                            log_local_path_at_info=config.general.startup_log_detail,
                            threads=_hybrid_cfg.sparse_encoder.threads if _hybrid_cfg.sparse_encoder.threads > 0 else None,
                            query_stop_list=_hybrid_cfg.sparse_encoder.query_stop_list,
                        )
                        logger.info(
                            f"bm42_doc_encoder_loaded model={_hybrid_cfg.sparse_encoder.model_name}"
                        )
                    except ConfigurationError as _bm42_err:
                        if not _hybrid_cfg.sparse_encoder.hash_bm25_fallback:
                            raise ConfigurationError(
                                f"bm42_doc_encoder_required hash_bm25_fallback=false "
                                f"error={_bm42_err}"
                            ) from _bm42_err
                        logger.warning(
                            f"bm42_doc_encoder_unavailable error={_bm42_err} "
                            "falling_back_to_hash_bm25 hash_bm25_fallback=true"
                        )
                if doc_encoder is None:
                    if (
                        _hybrid_cfg.sparse_encoder is not None
                        and not _hybrid_cfg.sparse_encoder.hash_bm25_fallback
                    ):
                        raise ConfigurationError(
                            "bm42_doc_encoder_required sparse_encoder present but "
                            "BM42 not loaded and hash_bm25_fallback=false"
                        )
                    if _hybrid_cfg.bm25_query_encoder is None:
                        logger.warning(
                            "vectorization_pipeline_disabled "
                            "reason=bm25_query_encoder_absent_and_bm42_failed"
                        )
                        raise ConfigurationError("no usable doc encoder")
                    vocab_size = int(_hybrid_cfg.bm25_query_encoder.vocab_size)  # type: ignore[assignment]
                    doc_encoder = BM25DocEncoder(
                        vocab_size=vocab_size,
                        k1=vec_cfg.bm25_doc_encoder.k1,
                        b=vec_cfg.bm25_doc_encoder.b,
                        avg_doc_length=vec_cfg.bm25_doc_encoder.avg_doc_length,
                    )
                # Optional compound splitter — only built when the sub-block is
                # present AND enabled AND the dictionary loads cleanly.
                compound_splitter: Optional[CompoundWordSplitter] = None
                if vec_cfg.compound_splitter is not None and vec_cfg.compound_splitter.enabled:
                    try:
                        compound_dict = _load_compound_splitter_dictionary(
                            vec_cfg.compound_splitter.dictionary_path
                        )
                        compound_splitter = CompoundWordSplitter(
                            dictionary=compound_dict,
                            min_segment_length=vec_cfg.compound_splitter.min_segment_length,
                            max_segments=vec_cfg.compound_splitter.max_segments,
                            oov_char_cost=vec_cfg.compound_splitter.oov_char_cost,
                            length_penalty=vec_cfg.compound_splitter.length_penalty,
                        )
                        logger.info(
                            f"compound_splitter_built dictionary_size={compound_splitter.dictionary_size} "
                            f"min_segment_length={vec_cfg.compound_splitter.min_segment_length}"
                        )
                    except (ValidationError, ConfigurationError) as e:
                        logger.warning(
                            f"compound_splitter_disabled error_type={type(e).__name__} error={str(e)} "
                            f"path={mask_path(vec_cfg.compound_splitter.dictionary_path)}"
                        )
                        compound_splitter = None
                doc_segmenter = DomainNameSegmenter(compound_splitter=compound_splitter)
                doc_synonym_expander: Optional[SynonymExpander] = None
                if vec_cfg.synonyms is not None and vec_cfg.synonyms.enabled:
                    doc_synonym_expander = SynonymExpander(vec_cfg.synonyms)
                doc_vectorization_pipeline = DocVectorizationPipeline(
                    segmenter=doc_segmenter,
                    doc_encoder=doc_encoder,
                    expander=doc_synonym_expander,
                )
                # Use the shortlist-stage encoder (256-dim Matryoshka slice) so
                # vectors match the Qdrant collection's dense_dim. The base
                # encoder (768-dim native) must NOT be used here — Qdrant silently
                # drops upserts that don't match the collection's vector size when
                # wait=False, producing points_count=0 with no reported failures.
                # Wrap with PrefixedEncoder so documents receive "search_document: "
                # instead of the query-side "search_query: " — task separation
                # requires distinct prefixes on each side.
                _enc_src = shortlist_encoder if shortlist_encoder is not None else encoder
                if vec_cfg.encoder_query_prefix:
                    _enc_src = PrefixedEncoder(_enc_src, vec_cfg.encoder_query_prefix)
                indexing_encoder = _enc_src
                _dense_enc = _enc_src.encode
                _batch_dense_enc = getattr(_enc_src, 'encode_batch', None)
                # Second-stage rerank vector (Design B): encode docs at the rerank
                # stage dim (768) under the configured rerank vector name so the
                # hybrid retriever can rescore the fused pool server-side. Active
                # only when the cascade built a rerank-stage encoder AND the hybrid
                # rerank config is enabled. Same "search_document: " doc prefix.
                _rr_cfg = config.retrieval.qdrant.hybrid.rerank if config.retrieval.qdrant is not None else None
                _rr_active = rerank_encoder is not None and _rr_cfg is not None and _rr_cfg.enabled
                _rerank_dense_enc = None
                _rerank_batch_dense_enc = None
                _rerank_dim = None
                _rerank_vec_name = None
                if _rr_active:
                    _rr_src = PrefixedEncoder(rerank_encoder, vec_cfg.encoder_query_prefix) if vec_cfg.encoder_query_prefix else rerank_encoder
                    _rerank_dense_enc = _rr_src.encode
                    _rerank_batch_dense_enc = getattr(_rr_src, 'encode_batch', None)
                    _rerank_dim = _rr_cfg.dim
                    _rerank_vec_name = _rr_cfg.vector_name
                # Character n-gram sparse channel (recall-side fuzzy/misspell). Doc
                # side: the indexer writes the ngram sparse vector. One encoder
                # config drives both doc + query sides (see retriever wiring below).
                _ngram_cfg = config.retrieval.qdrant.hybrid.ngram
                _ngram_doc_encoder = None
                _ngram_vec_name = None
                if _ngram_cfg is not None and _ngram_cfg.enabled:
                    _ngram_doc_encoder = CharNgramSparseEncoder(vocab_size=_ngram_cfg.vocab_size, min_n=_ngram_cfg.min_n, max_n=_ngram_cfg.max_n)
                    _ngram_vec_name = _ngram_cfg.vector_name
                    logger.info(f"char_ngram_doc_encoder_built vector_name={_ngram_vec_name} vocab_size={_ngram_cfg.vocab_size} min_n={_ngram_cfg.min_n} max_n={_ngram_cfg.max_n}")
                # Shared Matryoshka dense path: one cascade forward yields shortlist
                # + rerank dims. Unwrap PrefixedEncoder to reach StageEncoder.cascade;
                # apply the document task prefix here so both dims see the same text.
                _matryoshka_batch_enc = None
                _cascade_for_index: Optional[MatryoshkaCascadeEncoder] = None
                if isinstance(shortlist_encoder, StageEncoder):
                    _cascade_for_index = shortlist_encoder.cascade
                _doc_prefix = str(vec_cfg.encoder_query_prefix or "")
                if _cascade_for_index is not None and vec_cfg.indexer.shared_matryoshka_dense_encode:

                    def _matryoshka_batch_enc(
                        texts: List[str], dims: List[int]
                    ) -> Dict[int, List[List[float]]]:
                        if _doc_prefix:
                            prefixed: List[str] = []
                            for t in texts:
                                if t is None:
                                    prefixed.append(t)  # type: ignore[arg-type]
                                else:
                                    clean = str(t).strip()
                                    prefixed.append((_doc_prefix + clean) if clean else t)
                        else:
                            prefixed = list(texts)
                        return _cascade_for_index.encode_batch_at_dims(prefixed, dims)

                offline_indexer = OfflineIndexer(
                    config=vec_cfg.indexer,
                    pipeline=doc_vectorization_pipeline,
                    qdrant_factory=qdrant_factory,
                    dense_encoder=_dense_enc,
                    dense_dim=config.retrieval.vector.embedding_dim,
                    batch_dense_encoder=_batch_dense_enc,
                    rerank_dense_encoder=_rerank_dense_enc,
                    rerank_batch_dense_encoder=_rerank_batch_dense_enc,
                    rerank_dense_dim=_rerank_dim,
                    rerank_vector_name=_rerank_vec_name,
                    ngram_encoder=_ngram_doc_encoder,
                    ngram_vector_name=_ngram_vec_name,
                    matryoshka_batch_encoder=_matryoshka_batch_enc,
                )

                # The default document source factory streams payloads
                # from the in-process structured index. Production
                # deployments override this by passing a different
                # factory after `build_subsystems` returns (or by
                # subclassing `VectorRefreshDriver`); the default keeps
                # the registry self-contained and integration-test
                # friendly. The factory yields one document per payload
                # and tags each with the snapshot version it was
                # observed at — useful for downstream debugging.
                async def _default_doc_source_factory(snapshot_version: int) -> List[Any]:
                    docs: list = []
                    for payload in structured_index.iter_payloads():
                        if not isinstance(payload, dict):
                            continue
                        doc = dict(payload)
                        doc.setdefault('_snapshot_version', snapshot_version)
                        docs.append(doc)
                    return docs

                vector_refresh_driver = VectorRefreshDriver(
                    config=vec_cfg.refresh,
                    indexer=offline_indexer,
                    snapshot_registry=snapshot_registry,
                    document_source_factory=_default_doc_source_factory,
                )
                snapshot_registry.register_priority_hook(
                    lambda _v: vector_refresh_driver.notify_priority_refresh()
                )
                _doc_enc_desc = (
                    f"bm42={_hybrid_cfg.sparse_encoder.model_name}"
                    if type(doc_encoder).__name__ == "BM42SparseEncoder"
                    else f"bm25_vocab_size={vocab_size}"
                )
                logger.info(
                    f"vectorization_pipeline_built {_doc_enc_desc} "
                    f"batch_size={vec_cfg.indexer.batch_size} "
                    f"refresh_enabled={vec_cfg.refresh.enabled} "
                    f"compound_splitter={'on' if compound_splitter is not None else 'off'} "
                    f"synonyms={'on' if doc_synonym_expander is not None else 'off'}"
                )
            except (ValidationError, ConfigurationError) as e:
                logger.warning(
                    f"vectorization_pipeline_disabled reason=construction_failed "
                    f"error_type={type(e).__name__} error={str(e)}"
                )
                doc_vectorization_pipeline = None
                offline_indexer = None
                vector_refresh_driver = None
    if vec_cfg is not None and vec_cfg.delta_refresh is not None and vec_cfg.delta_refresh.enabled:
        _delta_cfg = vec_cfg.delta_refresh
        _athena_cfg = config.nl_to_sql.athena if config.nl_to_sql is not None else None
        if _athena_cfg is None:
            logger.warning("delta_refresh_driver_disabled reason=athena_config_absent")
        elif qdrant_factory is None or not qdrant_factory.available:
            logger.warning("delta_refresh_driver_disabled reason=qdrant_unavailable")
        else:
            try:
                _delta_athena = AthenaClient(_athena_cfg)
                if not _delta_athena.credentials_available:
                    logger.warning("delta_refresh_driver_credentials_unavailable will_retry_at_runtime")
                delta_refresh_driver = DeltaRefreshDriver(config=_delta_cfg, athena_client=_delta_athena, qdrant_factory=qdrant_factory)
                logger.info(f"delta_refresh_driver_built credentials_available={_delta_athena.credentials_available} interval_seconds={_delta_cfg.interval_seconds} source={_delta_cfg.source_database}.{_delta_cfg.source_table}")  # noqa: E501
            except (ValidationError, ConfigurationError) as e:
                logger.warning(f"delta_refresh_driver_disabled reason=construction_failed error_type={type(e).__name__} error={str(e)}")
    if vec_cfg is not None and vec_cfg.event_ingest is not None:
        _ei_cfg = vec_cfg.event_ingest
        _ei_athena_cfg = config.nl_to_sql.athena if config.nl_to_sql is not None else None
        if _ei_athena_cfg is None:
            logger.warning("event_ingest_driver_disabled reason=athena_config_absent")
        else:
            try:
                _ei_athena = AthenaClient(_ei_athena_cfg)
                if not _ei_athena.credentials_available:
                    logger.warning("event_ingest_driver_credentials_unavailable will_retry_at_runtime")
                _bid_dict = {
                    "enabled": _ei_cfg.bid_events.enabled if _ei_cfg.bid_events else False,
                    "interval_seconds": _ei_cfg.bid_events.interval_seconds if _ei_cfg.bid_events else 60.0,
                    "max_consecutive_failures": _ei_cfg.bid_events.max_consecutive_failures if _ei_cfg.bid_events else 3,
                    "source_database": _ei_cfg.bid_events.source_database if _ei_cfg.bid_events else "the_resale_place",
                    "source_table": _ei_cfg.bid_events.source_table if _ei_cfg.bid_events else "item_bids_cln",
                    "winning_bids_table": _ei_cfg.bid_events.winning_bids_table if _ei_cfg.bid_events else "item_winning_bids_cln",
                    "lookback_minutes": _ei_cfg.bid_events.lookback_minutes if _ei_cfg.bid_events else 60,
                    "chunk_minutes": _ei_cfg.bid_events.chunk_minutes if _ei_cfg.bid_events else 60,
                    "batch_size": _ei_cfg.bid_events.batch_size if _ei_cfg.bid_events else 10000,
                    "timeout_seconds": _ei_cfg.bid_events.timeout_seconds if _ei_cfg.bid_events else 30.0,
                } if _ei_cfg.bid_events else {"enabled": False}
                _watch_dict = {
                    "enabled": _ei_cfg.watch_events.enabled if _ei_cfg.watch_events else False,
                    "interval_seconds": _ei_cfg.watch_events.interval_seconds if _ei_cfg.watch_events else 120.0,
                    "max_consecutive_failures": _ei_cfg.watch_events.max_consecutive_failures if _ei_cfg.watch_events else 3,
                    "source_database": _ei_cfg.watch_events.source_database if _ei_cfg.watch_events else "the_resale_place",
                    "source_table": _ei_cfg.watch_events.source_table if _ei_cfg.watch_events else "member_items_watch_cln",
                    "watch_types_table": _ei_cfg.watch_events.watch_types_table if _ei_cfg.watch_events else "member_items_watch_types_cln",
                    "lookback_minutes": _ei_cfg.watch_events.lookback_minutes if _ei_cfg.watch_events else 60,
                    "chunk_minutes": _ei_cfg.watch_events.chunk_minutes if _ei_cfg.watch_events else 60,
                    "batch_size": _ei_cfg.watch_events.batch_size if _ei_cfg.watch_events else 10000,
                    "timeout_seconds": _ei_cfg.watch_events.timeout_seconds if _ei_cfg.watch_events else 30.0,
                } if _ei_cfg.watch_events else {"enabled": False}
                _enrich_dict = {
                    "enabled": _ei_cfg.qdrant_enrich.enabled if _ei_cfg.qdrant_enrich else False,
                    "enrich_limit": _ei_cfg.qdrant_enrich.enrich_limit if _ei_cfg.qdrant_enrich else 5000,
                } if _ei_cfg.qdrant_enrich else {"enabled": False}
                event_ingest_driver = EventIngestDriver(
                    bid_config=_bid_dict,
                    watch_config=_watch_dict,
                    qdrant_enrich_config=_enrich_dict,
                    athena_client=_ei_athena,
                    qdrant_factory=qdrant_factory,
                )
                logger.info(
                    f"event_ingest_driver_built"
                    f" bid_enabled={_bid_dict.get('enabled')}"
                    f" watch_enabled={_watch_dict.get('enabled')}"
                    f" credentials_available={_ei_athena.credentials_available}"
                )
            except (ValidationError, ConfigurationError) as e:
                logger.warning(f"event_ingest_driver_disabled reason=construction_failed error_type={type(e).__name__} error={str(e)}")
    if vec_cfg is not None and vec_cfg.enrichment_refresh is not None and vec_cfg.enrichment_refresh.enabled:
        _er_cfg = vec_cfg.enrichment_refresh
        _er_athena_cfg = config.nl_to_sql.athena if config.nl_to_sql is not None else None
        if _er_athena_cfg is None:
            logger.warning("enrichment_refresh_driver_disabled reason=athena_config_absent")
        elif vec_cfg.seed is None or vec_cfg.seed.database is None:
            logger.warning("enrichment_refresh_driver_disabled reason=seed_database_config_absent")
        elif qdrant_factory is None or not qdrant_factory.available:
            logger.warning("enrichment_refresh_driver_disabled reason=qdrant_unavailable")
        else:
            try:
                _er_athena = AthenaClient(_er_athena_cfg)
                if not _er_athena.credentials_available:
                    logger.warning("enrichment_refresh_driver_credentials_unavailable will_retry_at_runtime")
                enrichment_refresh_driver = EnrichmentRefreshDriver(
                    config=_er_cfg,
                    seed_database=vec_cfg.seed.database,
                    athena_client=_er_athena,
                    qdrant_factory=qdrant_factory,
                )
                logger.info(f"enrichment_refresh_driver_built credentials_available={_er_athena.credentials_available} interval_seconds={_er_cfg.interval_seconds}")
            except (ValidationError, ConfigurationError) as e:
                logger.warning(f"enrichment_refresh_driver_disabled reason=construction_failed error_type={type(e).__name__} error={str(e)}")
    measurement_store = MeasurementStore(config=config.measurement)
    # Build the NL-to-SQL pipeline + analytics router BEFORE the orchestrator
    # so the orchestrator can take an injected (Optional) analytics handle.
    # Both are None when their preconditions fail; the orchestrator's
    # `analytics_available` property reflects that state.
    shared_retrieved_content_sanitizer: Optional[RetrievedContentSanitizer] = None
    if config.nl_to_sql.enabled and call_router is not None:
        shared_retrieved_content_sanitizer = _build_retrieved_content_sanitizer(config, sanitizer, signal_store)
    nl_to_sql_pipeline: Optional[NLToSQLPipeline] = None
    _shared_ch_executor: Optional[ClickHouseExecutor] = None
    if config.nl_to_sql.enabled:
        if call_router is None:
            logger.warning("nl_to_sql_pipeline_disabled reason=no_llm_call_router")
        elif config.nl_to_sql.analytics is None or not config.nl_to_sql.analytics.enabled:
            logger.info("nl_to_sql_pipeline_disabled reason=analytics_not_configured")
        else:
            try:
                _, _shared_ch_executor = _nl_analytics_clickhouse_stack(config, backend_health)
                nl_to_sql_pipeline = _build_nl_to_sql_pipeline(
                    config, call_router, _shared_ch_executor,
                    sanitizer=sanitizer, signal_store=signal_store,
                    retrieved_content_sanitizer=shared_retrieved_content_sanitizer,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"nl_to_sql_pipeline_disabled reason=construction_failed error_type={type(e).__name__} error={e}")
                nl_to_sql_pipeline = None
    analytics_router: Optional[AnalyticsRouter] = None
    if (
        config.nl_to_sql.enabled
        and config.nl_to_sql.analytics is not None
        and config.nl_to_sql.analytics.enabled
        and nl_to_sql_pipeline is not None
        and call_router is not None
        and _shared_ch_executor is not None
    ):
        try:
            analytics_router = _build_analytics_router(
                config, nl_to_sql_pipeline, call_router, backend_health,
                _shared_ch_executor,
                sanitizer=sanitizer, signal_store=signal_store,
                retrieved_content_sanitizer=shared_retrieved_content_sanitizer,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"analytics_router_disabled reason=construction_failed "
                f"error_type={type(e).__name__} error={e}"
            )
            analytics_router = None
    enriched_tables_builder: Optional[EnrichedTablesBuilder] = None
    if (
        analytics_router is not None
        and _shared_ch_executor is not None
        and config.nl_to_sql.analytics is not None
        and config.nl_to_sql.analytics.capabilities is not None
        and config.nl_to_sql.analytics.capabilities.domain_analytics is not None
    ):
        try:
            _da_cfg = config.nl_to_sql.analytics.capabilities.domain_analytics
            enriched_tables_builder = EnrichedTablesBuilder(
                executor=_shared_ch_executor,
                watch_density_mv=getattr(_da_cfg, 'watch_density_mv', None),
                bid_velocity_item_mv=getattr(_da_cfg, 'bid_velocity_item_mv', None),
                transactions_table=getattr(_da_cfg, 'transactions_table', None),
            )
            logger.info("enriched_tables_builder_built")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"enriched_tables_builder_init_failed error_type={type(e).__name__} error={e}")
            enriched_tables_builder = None
    if (
        config.explore.clickhouse_rails.enabled
        and analytics_router is not None
        and analytics_router.clickhouse_executor.credentials_available
    ):
        try:
            _explore_ch = analytics_router.clickhouse_executor
            explore_trending_source = ClickHouseTrendingExploreSource(
                config.explore.trending, config.explore.clickhouse_rails, _explore_ch,
            )
            explore_ending_soon_source = ClickHouseEndingSoonExploreSource(
                config.explore.ending_soon, config.explore.clickhouse_rails, _explore_ch,
            )
            _last_hour_source = ClickHouseLastHourExploreSource(config.explore.clickhouse_rails, _explore_ch)
            _latest_source = ClickHouseLatestExploreSource(config.explore.clickhouse_rails, _explore_ch)
            _high_volume_source = ClickHouseHighVolumeExploreSource(config.explore.clickhouse_rails, _explore_ch)
            _fresh_source = ClickHouseFreshExploreSource(config.explore.clickhouse_rails, _explore_ch)
            _last_week_source = ClickHouseLastWeekExploreSource(config.explore.clickhouse_rails, _explore_ch)
            _watch_density_source = ClickHouseWatchDensityExploreSource(config.explore.clickhouse_rails, _explore_ch)
            _high_traffic_source = ClickHouseHighTrafficExploreSource(config.explore.clickhouse_rails, _explore_ch)
            explore_composer = ExploreComposer(config=config.explore, trending=explore_trending_source, ending_soon=explore_ending_soon_source, last_hour=_last_hour_source, latest=_latest_source, high_volume=_high_volume_source, fresh_listings=_fresh_source, last_week=_last_week_source, watch_density=_watch_density_source, high_traffic=_high_traffic_source)  # noqa: E501
            zero_result_guard = ZeroResultGuard(config=config.explore.zero_result_guard, composer=explore_composer)
            logger.info("explore_rails_wired_backend=clickhouse")
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"explore_ch_rewire_failed error_type={type(e).__name__} error={str(e)} keeping=in_memory_explore_sources"
            )
    elif (
        config.explore.clickhouse_rails.enabled
        and config.nl_to_sql.analytics is not None
        and config.nl_to_sql.analytics.enabled
    ):
        try:
            _, _explore_ch_only = _nl_analytics_clickhouse_stack(config, backend_health)
            if _explore_ch_only.credentials_available:
                explore_trending_source = ClickHouseTrendingExploreSource(
                    config.explore.trending, config.explore.clickhouse_rails, _explore_ch_only,
                )
                explore_ending_soon_source = ClickHouseEndingSoonExploreSource(
                    config.explore.ending_soon, config.explore.clickhouse_rails, _explore_ch_only,
                )
                _last_hour_source = ClickHouseLastHourExploreSource(config.explore.clickhouse_rails, _explore_ch_only)
                _latest_source = ClickHouseLatestExploreSource(config.explore.clickhouse_rails, _explore_ch_only)
                _high_volume_source = ClickHouseHighVolumeExploreSource(config.explore.clickhouse_rails, _explore_ch_only)
                _fresh_source = ClickHouseFreshExploreSource(config.explore.clickhouse_rails, _explore_ch_only)
                _last_week_source = ClickHouseLastWeekExploreSource(config.explore.clickhouse_rails, _explore_ch_only)
                _watch_density_source = ClickHouseWatchDensityExploreSource(config.explore.clickhouse_rails, _explore_ch_only)
                _high_traffic_source = ClickHouseHighTrafficExploreSource(config.explore.clickhouse_rails, _explore_ch_only)
                explore_composer = ExploreComposer(config=config.explore, trending=explore_trending_source, ending_soon=explore_ending_soon_source, last_hour=_last_hour_source, latest=_latest_source, high_volume=_high_volume_source, fresh_listings=_fresh_source, last_week=_last_week_source, watch_density=_watch_density_source, high_traffic=_high_traffic_source)  # noqa: E501
                zero_result_guard = ZeroResultGuard(config=config.explore.zero_result_guard, composer=explore_composer)
                logger.info("explore_rails_wired_backend=clickhouse_standalone_stack")
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"explore_ch_standalone_stack_failed error_type={type(e).__name__} error={str(e)} keeping=in_memory_explore_sources"
            )
    # Output-side egress guard. Backend dispatch is config-driven
    # (``NoOpEgressGuard`` / ``NoOpModerator`` on the disabled paths,
    # ``LexicalBlocklistModerator`` on the stdlib-only default). Reuses the
    # ``sanitizer`` instance already wired for the LLM-ingress and
    # retrieved-content paths so the PII regex set is the single source of
    # truth across input and output sides. Construction failures
    # (malformed sub-config, unknown moderator backend) raise
    # ``ConfigurationError`` at boot — the gate is
    # safety-critical and refuses to half-wire.
    egress_cfg = config.safety.egress_guard
    egress_guard: Union[EgressGuard, NoOpEgressGuard]
    if not egress_cfg.enabled:
        egress_guard = NoOpEgressGuard()
    else:
        moderator: Moderator
        moderator_cfg = egress_cfg.moderator
        if not moderator_cfg.enabled or moderator_cfg.backend == 'noop':
            moderator = NoOpModerator()
        elif moderator_cfg.backend == 'lexical_blocklist':
            # `__post_init__` on ModeratorConfig already guarantees `lexical`
            # is non-None when enabled+lexical_blocklist; assert for type
            # narrowing.
            assert moderator_cfg.lexical is not None
            moderator = LexicalBlocklistModerator(moderator_cfg.lexical)
        else:
            raise ConfigurationError(
                f"safety.egress_guard.moderator.backend={moderator_cfg.backend!r} is not wired in registry; "
                "supported backends: lexical_blocklist, noop"
            )
        egress_guard = EgressGuard(
            config=egress_cfg,
            sanitizer=sanitizer,
            moderator=moderator,
        )
    logger.info(
        f"egress_guard_built enabled={egress_cfg.enabled} moderator_backend={egress_cfg.moderator.backend} pii_scrub_enabled={egress_cfg.pii_scrub.enabled}"
    )
    # Per-query LLM cost-budget factory. Constructed ONLY when
    # ``cost_budget`` is present in YAML AND ``enabled=true``.
    # Default deployment has ``config.cost_budget=None`` so the
    # orchestrator wires no factory and skips contextvar binding (zero
    # overhead). When enabled, the factory closes over the typed config
    # so each ``SearchOrchestrator.search()`` / ``analytics()`` call gets
    # a fresh ``QueryCostBudget`` scoped to that request.
    cost_budget_factory: Optional[Callable[[Optional[str]], Union[QueryCostBudget, NoOpQueryCostBudget]]] = None
    fleet_cost_budget: Optional[FleetCostBudget] = None
    if config.cost_budget is not None and config.cost_budget.enabled:
        cb_cfg = config.cost_budget
        max_cost = float(cb_cfg.max_cost_usd_per_query)

        def _cost_budget_factory(request_id: Optional[str]) -> QueryCostBudget:
            return QueryCostBudget(max_cost_usd_per_query=max_cost, request_id=request_id)

        cost_budget_factory = _cost_budget_factory
        logger.info(
            f"cost_budget_factory_built max_cost_usd_per_query={max_cost:.6f} "
            f"fail_soft_on_breach={cb_cfg.fail_soft_on_breach} "
            f"degrade_on_breach=llm_error_regex_fallback"
        )
    else:
        logger.info("cost_budget_factory_built enabled=false")
    # Fleet budget is independent of per-query enabled: allow fleet-only when
    # cost_budget.fleet.enabled even if query factory is off (tracking NoOp).
    _cb = config.cost_budget
    _fleet_cfg = getattr(_cb, 'fleet', None) if _cb is not None else None
    if _fleet_cfg is not None and bool(_fleet_cfg.enabled):
        store = build_fleet_cost_store(
            backend=str(_fleet_cfg.backend),
            redis_url_env_var=str(_fleet_cfg.redis_url_env_var),
            key_prefix=str(_fleet_cfg.key_prefix),
            socket_timeout_seconds=float(_fleet_cfg.socket_timeout_seconds),
        )
        fleet_cost_budget = FleetCostBudget(
            store=store,
            max_cost_usd_per_hour=_fleet_cfg.max_cost_usd_per_hour,
            max_cost_usd_per_day=_fleet_cfg.max_cost_usd_per_day,
        )
        logger.info(
            f"fleet_cost_budget_built backend={_fleet_cfg.backend} "
            f"max_usd_per_hour={_fleet_cfg.max_cost_usd_per_hour} "
            f"max_usd_per_day={_fleet_cfg.max_cost_usd_per_day}"
        )
    else:
        logger.info("fleet_cost_budget_built enabled=false")
    guidance_service: Optional[GuidanceService] = None
    if config.guidance.enabled:
        guidance_ch: Optional[ClickHouseExecutor] = None
        if analytics_router is not None:
            guidance_ch = analytics_router.clickhouse_executor
        elif config.nl_to_sql.analytics is not None and config.nl_to_sql.analytics.enabled:
            try:
                _, guidance_ch = _nl_analytics_clickhouse_stack(config, backend_health)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    f"guidance_clickhouse_stack_failed error_type={type(e).__name__} error={str(e)}"
                )
                guidance_ch = None
        guidance_service = GuidanceService(config=config.guidance, ch_executor=guidance_ch)
        logger.info(f"guidance_service_built ch_executor_present={'yes' if guidance_ch is not None else 'no'}")
    # Gap 4: per-tenant analytics rate limit. Independent budget from the
    # ingress search rate limit (much more expensive per call). Built only
    # when the config block is enabled; otherwise the orchestrator skips
    # the bucket check entirely.
    analytics_rate_limit_cfg = getattr(config.general, 'analytics_rate_limit', None)
    analytics_rate_limiter = None
    if analytics_rate_limit_cfg is not None and bool(getattr(analytics_rate_limit_cfg, 'enabled', False)):
        analytics_rate_limiter = SlidingWindowRateLimiter(
            max_requests=int(analytics_rate_limit_cfg.max_requests_per_window),
            window_seconds=int(analytics_rate_limit_cfg.window_seconds),
            session_state_max=int(analytics_rate_limit_cfg.session_state_max),
        )
        logger.info(
            f"analytics_rate_limit_enabled max={analytics_rate_limit_cfg.max_requests_per_window} "
            f"window_s={analytics_rate_limit_cfg.window_seconds} "
            f"key={analytics_rate_limit_cfg.key_strategy}"
        )

    orchestrator = SearchOrchestrator(
        config=config,
        qi_engine=qi_engine,
        vector_retriever=vector_retriever,
        structured_retriever=structured_retriever,
        sql_retriever=sql_retriever,
        fuser=fuser,
        eranker_client=eranker_client,
        diversifier=diversifier,
        exact_cache=exact_cache,
        structured_cache=structured_cache,
        intent_plan_cache=intent_plan_cache,
        history_store=history_store,
        health_registry=backend_health,
        degradation_planner=degradation_planner,
        measurement_store=measurement_store,
        analytics_router=analytics_router,
        zero_result_guard=zero_result_guard,
        explore_composer=explore_composer,
        sanitizer=sanitizer,
        egress_guard=egress_guard,
        cost_budget_factory=cost_budget_factory,
        fleet_cost_budget=fleet_cost_budget,
        spell_corrector=spell_corrector,
        query_transformer=query_transformer,
        redis_payload_tier=redis_payload_tier,
        inventory_snapshot_version_fn=lambda: snapshot_registry.version,
        guidance_service=guidance_service,
        signal_store=signal_store,
        analytics_rate_limiter=analytics_rate_limiter,
        analytics_rate_limit_config=analytics_rate_limit_cfg,
        entity_extractor=l0_entity_extractor,
    )
    proxy_signal_evaluator = ProxySignalEvaluator(
        config=config.measurement,
        store=measurement_store,
        signal_store=signal_store,
        sanitizer=sanitizer,
        circuit_breaker=circuit_breaker,
        cache_stats_fn=orchestrator.cache_stats,
        # None when no LLM tier is wired (signal degrades to
        # `not_instrumented` automatically). When wired, the evaluator pulls
        # cached vs total input-token counters straight from the router.
        prompt_cache_stats_fn=(call_router.prompt_cache_stats if call_router is not None else None),
        # None when the analytics router is disabled (fast-path
        # verifier-skip is meaningless without the analytics path). When
        # wired, the evaluator pulls fast-path skip / invocation counters
        # straight from the router.
        verifier_skip_stats_fn=(analytics_router.verifier_skip_stats if analytics_router is not None else None),
    )
    # Cache miss-storm alarm. Built only when the YAML sub-block is
    # present; the detector's `enabled` flag inside the sub-block is the
    # second-level switch that lets ops silence the alarm without a config
    # restructure. Kept conditional so the feature can be rolled out per
    # environment without touching code.
    cache_miss_storm_detector: Optional[CacheMissStormDetector] = None
    if config.measurement.cache_miss_storm is not None:
        cache_miss_storm_detector = CacheMissStormDetector(
            config=config.measurement.cache_miss_storm,
            signal_store=signal_store,
        )
    # retrieval-quality evaluator. Constructed only when
    # enabled; consumed by offline scripts via the registry handle. The
    # evaluator binds to orchestrator.search (the production code path), so
    # any change to the orchestrator reaches the evaluator automatically.
    retrieval_quality_evaluator: Optional[RetrievalQualityEvaluator] = None
    if config.offline_eval.retrieval_eval.enabled:
        retrieval_quality_evaluator = RetrievalQualityEvaluator(
            config=config.offline_eval.retrieval_eval,
            search_fn=orchestrator.search,
        )
    # LLM-as-judge for relevance grading. Built only when
    # (a) the YAML sub-block exists and is enabled AND (b) the shared
    # LLMCallRouter is available. When the router is missing we log a
    # WARNING instead of raising — the rest of the system stays online
    # (graceful degradation, mirroring the nl_to_sql_pipeline gating
    # pattern used elsewhere in this composer).
    llm_relevance_judge: Optional[LLMRelevanceJudge] = None
    llm_judged_query_builder: Optional[LLMJudgedQueryBuilder] = None
    if config.offline_eval.llm_judge is not None and config.offline_eval.llm_judge.enabled:
        if call_router is None:
            logger.warning(
                "llm_relevance_judge_disabled reason=no_llm_call_router "
                f"task_type={config.offline_eval.llm_judge.task_type}"
            )
        else:
            llm_relevance_judge = LLMRelevanceJudge(
                config=config.offline_eval.llm_judge,
                call_router=call_router,
            )
            llm_judged_query_builder = LLMJudgedQueryBuilder(
                judge=llm_relevance_judge,
                config=config.offline_eval.llm_judge,
            )
    # Wire a ClickHouse executor into the background drivers. Prefer the
    # executor already built on the nl_to_sql analytics path; when that path
    # is off (or its stack construction failed) fall back to a standalone
    # analytics CH stack so the drivers stay functional independently of the
    # analytics router — mirroring the guidance / explore standalone fallbacks
    # above. Without this, an enabled event_ingest driver polls forever and
    # skips every cycle with reason=ch_executor_not_wired.
    if delta_refresh_driver is not None or event_ingest_driver is not None:
        _driver_ch_executor = _shared_ch_executor
        if (
            _driver_ch_executor is None
            and config.nl_to_sql.analytics is not None
            and config.nl_to_sql.analytics.enabled
        ):
            try:
                _, _driver_ch_executor = _nl_analytics_clickhouse_stack(config, backend_health)
                logger.info("driver_clickhouse_stack_built source=standalone_fallback")
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    f"driver_clickhouse_stack_failed error_type={type(e).__name__} error={str(e)}"
                )
                _driver_ch_executor = None
        if delta_refresh_driver is not None and _driver_ch_executor is not None:
            delta_refresh_driver.set_ch_executor(_driver_ch_executor)
            logger.info("delta_refresh_driver_ch_executor_wired")
        if event_ingest_driver is not None and _driver_ch_executor is not None:
            event_ingest_driver.set_ch_executor(_driver_ch_executor)
            logger.info("event_ingest_driver_ch_executor_wired")
        if event_ingest_driver is not None and _driver_ch_executor is None:
            logger.warning("event_ingest_driver_ch_executor_unavailable reason=no_analytics_clickhouse_stack")
    ch_tld_refresh_driver: Optional[ClickHouseTLDRefreshDriver] = None
    if _pending_ch_inventory is not None:
        _ch_for_tld = _shared_ch_executor
        if _ch_for_tld is None and config.nl_to_sql.analytics is not None and config.nl_to_sql.analytics.enabled:
            try:
                _, _ch_for_tld = _nl_analytics_clickhouse_stack(config, backend_health)
                logger.info("ch_tld_refresh_clickhouse_stack_built source=standalone")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"ch_tld_refresh_clickhouse_stack_failed error_type={type(e).__name__} error={str(e)}")
                _ch_for_tld = None
        if _ch_for_tld is not None:
            try:
                ch_tld_refresh_driver = ClickHouseTLDRefreshDriver(
                    contract=_pending_ch_inventory,
                    client=_ch_for_tld._client,  # noqa: SLF001
                    database=config.nl_to_sql.analytics.clickhouse.database,
                    lookback_days=config.qi.regex.ch_tld_lookback_days,
                    interval_seconds=config.qi.regex.ch_tld_refresh_interval_seconds,
                    tld_query_sql=config.qi.regex.ch_tld_query_sql,
                )
                logger.info(
                    f"ch_tld_refresh_driver_built lookback_days={config.qi.regex.ch_tld_lookback_days} "
                    f"interval_seconds={config.qi.regex.ch_tld_refresh_interval_seconds}"
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"ch_tld_refresh_driver_build_failed error_type={type(e).__name__} error={str(e)}")
        else:
            logger.warning("ch_tld_refresh_driver_skipped reason=clickhouse_unavailable")
    logger.info(
        f"semantic_search_subsystems_built llm_tier={'enabled' if llm_classifier is not None else 'disabled'} "
        f"eranker_backend={config.retrieval.eranker.backend} "
        f"history={'on' if config.history.enabled else 'off'} "
        f"retrieval_eval={'on' if retrieval_quality_evaluator is not None else 'off'} "
        f"llm_judge={'on' if llm_relevance_judge is not None else 'off'}"
    )
    return Subsystems(
        config=config,
        encoder=encoder,
        qi_engine=qi_engine,
        vector_index=vector_index,
        structured_index=structured_index,
        price_band_store=price_band_store,
        vector_retriever=vector_retriever,
        structured_retriever=structured_retriever,
        sql_retriever=sql_retriever,
        qdrant_factory=qdrant_factory,
        fuser=fuser,
        eranker_client=eranker_client,
        diversifier=diversifier,
        egress_guard=egress_guard,
        exact_cache=exact_cache,
        structured_cache=structured_cache,
        intent_plan_cache=intent_plan_cache,
        redis_payload_tier=redis_payload_tier,
        snapshot_registry=snapshot_registry,
        listing_consumer=listing_consumer,
        bid_consumer=bid_consumer,
        inventory_resolver=inventory_resolver,
        calibrator_registry=calibrator_registry,
        signal_store=signal_store,
        history_store=history_store,
        history_vector_store=history_vector_store,
        history_compactor=history_compactor,
        history_compactor_driver=history_compactor_driver,
        sanitizer=sanitizer,
        explore_trending_source=explore_trending_source,
        explore_ending_soon_source=explore_ending_soon_source,
        explore_composer=explore_composer,
        zero_result_guard=zero_result_guard,
        retrieval_quality_evaluator=retrieval_quality_evaluator,
        circuit_breaker=circuit_breaker,
        backend_health=backend_health,
        degradation_planner=degradation_planner,
        measurement_store=measurement_store,
        proxy_signal_evaluator=proxy_signal_evaluator,
        cache_miss_storm_detector=cache_miss_storm_detector,
        orchestrator=orchestrator,
        llm_provider=llm_provider,
        nl_to_sql_pipeline=nl_to_sql_pipeline,
        analytics_router=analytics_router,
        enriched_tables_builder=enriched_tables_builder,
        guidance_service=guidance_service,
        structural_gate=structural_gate,
        call_router=call_router,
        semantic_router=semantic_router,
        spell_corrector=spell_corrector,
        query_transformer=query_transformer,
        llm_relevance_judge=llm_relevance_judge,
        llm_judged_query_builder=llm_judged_query_builder,
        doc_vectorization_pipeline=doc_vectorization_pipeline,
        offline_indexer=offline_indexer,
        vector_refresh_driver=vector_refresh_driver,
        delta_refresh_driver=delta_refresh_driver,
        event_ingest_driver=event_ingest_driver,
        enrichment_refresh_driver=enrichment_refresh_driver,
        cascade_encoder=cascade_encoder,
        router_encoder=router_encoder,
        shortlist_encoder=shortlist_encoder,
        rerank_encoder=rerank_encoder,
        centroid_retrainer=centroid_retrainer,
        centroid_retrainer_driver=centroid_retrainer_driver,
        ch_tld_refresh_driver=ch_tld_refresh_driver,
        indexing_encoder=indexing_encoder,
        dynamic_synonym_store=_bm25_dynamic_store,
        calibration_boot_fit_fn=_calibration_boot_fit_fn,
    )


async def seed_boot_indexes(sub: Subsystems) -> Optional[Dict[str, Any]]:
    """Load domain records from seed DB into in-memory indexes (daily_snapshot or realtime source).

    Streams pages from _resolve_seed_pages and indexes/writes each page before
    fetching the next, bounding peak RAM to O(page_size) instead of O(max_records)
    - pages come from vectorization.seed_merge's Athena-side merged final table.
    """
    vec_cfg = sub.config.vectorization
    if vec_cfg is None or vec_cfg.seed is None or not vec_cfg.seed.enabled:
        return None
    # Qdrant-backed indexes have no .add(); only in-memory indexes are populated
    # by load_seed_into_indexes. On a Qdrant backend both are absent — do NOT
    # early-return here (that left the collection unseeded at boot, so the startup
    # prewarm task queried a missing collection until the first manual /data-build).
    # Fall through to offline_indexer.run(page_docs) below, which creates + seeds Qdrant.
    _mem_vec = sub.vector_index if isinstance(sub.vector_index, InMemoryVectorIndex) else None
    _mem_str = sub.structured_index if isinstance(sub.structured_index, InMemoryStructuredIndex) else None
    _qdrant_only = _mem_vec is None and _mem_str is None
    seed_cfg = vec_cfg.seed
    _ch_exec = (
        sub.analytics_router.clickhouse_executor
        if sub.analytics_router is not None
        and getattr(sub.analytics_router, 'clickhouse_executor', None) is not None
        and sub.analytics_router.clickhouse_executor.credentials_available
        else None
    )
    idempotency_key = vec_cfg.indexer.idempotency_key_field
    _encoder = sub.indexing_encoder if sub.indexing_encoder is not None else (
        sub.shortlist_encoder if sub.shortlist_encoder is not None else sub.encoder
    )

    total_docs = 0
    tables_seen: set = set()
    all_missing: List[str] = []
    resolved_meta: Dict[str, Any] = {}
    total_tables_skipped = 0

    load_totals = dict(
        documents_offered=0, vector_indexed=0, structured_indexed=0,
        bm25_encoded=0, bm25_skipped=0, skipped=0, elapsed_ms=0.0,
        explore_trending_added=0, explore_ending_soon_added=0,
    )
    indexer_totals = dict(points_seen=0, points_upserted=0, points_skipped=0, bm25_encoded=0, bm25_skipped=0, failures=0)
    indexer_qdrant_available = True
    indexer_ran = False
    ch_totals = dict(rows_written=0, errors=0, elapsed_ms=0.0)
    ch_ran = False
    _ch_schema_pending = True

    fetch_elapsed_ms = 0.0
    _prev_ts = time.monotonic()
    run_token = "boot"
    timing = StageTimingSession(seed_cfg.stage_timing)
    page_gate = bool(timing.config.log_page_stages)
    async with contextlib.aclosing(
        _resolve_seed_pages(seed_cfg, sub.config, run_token=run_token, timing=timing)
    ) as pages:
        async for page_docs, missing_cols, table_name, base_meta in pages:
            _now = time.monotonic()
            fetch_elapsed_ms += (_now - _prev_ts) * 1000.0
            resolved_meta = base_meta
            if table_name is None:
                # Sentinel: Athena unconfigured or credentials unavailable — no
                # tables were queryable at all.
                total_tables_skipped = base_meta.get("tables_skipped", 0)
                _prev_ts = time.monotonic()
                continue

            tables_seen.add(table_name)
            for c in missing_cols:
                if c not in all_missing:
                    all_missing.append(c)
            total_docs += len(page_docs)

            with timing.stage(
                "load_indexes",
                gate=page_gate,
                table=table_name,
                docs=len(page_docs),
            ):
                if _qdrant_only:
                    # In-memory indexes absent: skip the full dense/BM25 load (it would
                    # encode sparse vectors only to discard them — structured_index is
                    # None so they never reach the offline indexer, which re-encodes
                    # anyway). Populate only the in-memory explore sources; the offline
                    # indexer seeds Qdrant below.
                    page_load_summary = await load_seed_into_indexes(
                        documents=page_docs,
                        vector_index=None,
                        structured_index=None,
                        encoder=_encoder,
                        trending_source=sub.explore_trending_source,
                        ending_soon_source=sub.explore_ending_soon_source,
                        pipeline=None,
                        idempotency_key=idempotency_key,
                        batch_yield_size=seed_cfg.batch_yield_size,
                        encode_batch_size=seed_cfg.encode_batch_size,
                    )
                else:
                    page_load_summary = await load_seed_into_indexes(
                        documents=page_docs,
                        vector_index=_mem_vec,
                        structured_index=_mem_str,
                        encoder=_encoder,
                        trending_source=sub.explore_trending_source,
                        ending_soon_source=sub.explore_ending_soon_source,
                        pipeline=sub.doc_vectorization_pipeline,
                        idempotency_key=idempotency_key,
                        batch_yield_size=seed_cfg.batch_yield_size,
                        encode_batch_size=seed_cfg.encode_batch_size,
                    )
            for _k in ("documents_offered", "vector_indexed", "structured_indexed", "bm25_encoded", "bm25_skipped", "skipped", "explore_trending_added", "explore_ending_soon_added"):
                load_totals[_k] += getattr(page_load_summary, _k)
            load_totals["elapsed_ms"] += page_load_summary.elapsed_ms

            if sub.offline_indexer is not None:
                indexer_ran = True
                try:
                    with timing.stage(
                        "qdrant_index",
                        gate=page_gate,
                        table=table_name,
                        docs=len(page_docs),
                    ):
                        page_indexer_summary = await sub.offline_indexer.run(
                            page_docs, timing=timing,
                        )
                    for _k in ("points_seen", "points_upserted", "points_skipped", "bm25_encoded", "bm25_skipped", "failures"):
                        indexer_totals[_k] += getattr(page_indexer_summary, _k, 0)
                    indexer_qdrant_available = indexer_qdrant_available and page_indexer_summary.qdrant_available
                    if _qdrant_only:
                        # Qdrant-only: the load step above indexed nothing (in-memory
                        # absent); the offline indexer is what actually seeded the
                        # collection. Fold its stats into load_totals so the reported
                        # "loading" block reflects real work done for this page.
                        load_totals["documents_offered"] += page_indexer_summary.points_seen - page_load_summary.documents_offered
                        load_totals["vector_indexed"] += page_indexer_summary.points_upserted - page_load_summary.vector_indexed
                        load_totals["bm25_encoded"] += page_indexer_summary.bm25_encoded - page_load_summary.bm25_encoded
                        load_totals["bm25_skipped"] += page_indexer_summary.bm25_skipped - page_load_summary.bm25_skipped
                        load_totals["skipped"] += page_indexer_summary.points_skipped - page_load_summary.skipped
                    logger.info(
                        f"seed_boot_qdrant_indexed table={table_name} points_upserted={page_indexer_summary.points_upserted} "
                        f"qdrant_available={page_indexer_summary.qdrant_available}"
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        f"seed_boot_qdrant_index_failed table={table_name} error_type={type(e).__name__} error={str(e)}"
                    )

            if _ch_exec is not None:
                ch_ran = True
                try:
                    with timing.stage(
                        "clickhouse_seed",
                        gate=page_gate,
                        table=table_name,
                        docs=len(page_docs),
                    ):
                        _ensure = True
                        if seed_cfg.clickhouse_ensure_schema_once:
                            _ensure = _ch_schema_pending
                            _ch_schema_pending = False
                        _ch_write = await insert_seed_to_clickhouse(
                            page_docs,
                            _ch_exec,
                            ensure_schema=_ensure,
                            batch_size=seed_cfg.clickhouse_batch_size,
                            insert_timeout_seconds=seed_cfg.clickhouse_insert_timeout_seconds,
                            schema_timeout_seconds=seed_cfg.clickhouse_schema_timeout_seconds,
                            target_table=sub.config.nl_to_sql.analytics.seed_target_table,
                            snapshot_table=sub.config.nl_to_sql.analytics.seed_snapshot_table,
                        )
                    ch_totals["rows_written"] += _ch_write.rows_written
                    ch_totals["errors"] += _ch_write.errors
                    ch_totals["elapsed_ms"] += _ch_write.elapsed_ms
                    logger.info(
                        f"seed_boot_ch_write_complete table={table_name} rows_written={_ch_write.rows_written} "
                        f"errors={_ch_write.errors}"
                    )
                except Exception as _ce:  # noqa: BLE001
                    logger.warning(
                        f"seed_boot_ch_write_failed table={table_name} error_type={type(_ce).__name__} error={str(_ce)}"
                    )
            _prev_ts = time.monotonic()

    tables_queried = len(tables_seen)
    result: Dict[str, Any] = {
        "source": resolved_meta.get("source", seed_cfg.source),
        "fetch": {
            "database": resolved_meta.get("database", seed_cfg.database.name),
            "documents": total_docs,
            "tables_queried": tables_queried,
            "tables_skipped": total_tables_skipped if tables_queried == 0 else len(seed_cfg.database.tables) - tables_queried,
            "elapsed_ms": round(fetch_elapsed_ms, 1),
            "missing_columns": all_missing,
        },
        "loading": {**load_totals, "elapsed_ms": round(load_totals["elapsed_ms"], 1)},
    }
    if indexer_ran:
        result["qdrant_indexing"] = {
            "points_upserted": indexer_totals["points_upserted"],
            "points_skipped": indexer_totals["points_skipped"],
            "qdrant_available": indexer_qdrant_available,
            "failures": indexer_totals["failures"],
        }
    if ch_ran:
        result["ch_seed"] = {
            "rows_written": ch_totals["rows_written"],
            "errors": ch_totals["errors"],
            "elapsed_ms": round(ch_totals["elapsed_ms"], 1),
        }
    timing.log_job_complete(
        endpoint="seed_boot_indexes",
        documents=total_docs,
        source=seed_cfg.source,
        tables_queried=tables_queried,
    )
    if timing.config.include_in_response:
        result["stage_timing"] = timing.summary()
    logger.info(
        f"seed_boot_complete vector_indexed={load_totals['vector_indexed']} "
        f"structured_indexed={load_totals['structured_indexed']} "
        f"bm25_encoded={load_totals['bm25_encoded']} "
        f"documents={total_docs} source={seed_cfg.source} tables_queried={tables_queried}"
    )
    return result


async def _resolve_seed_pages(
    seed_cfg: Any,
    config: Any,
    run_token: str,
    timing: Any,
) -> AsyncIterator[Tuple[List[Dict[str, Any]], List[str], Optional[str], Dict[str, Any]]]:
    """Stream domain seed docs page-by-page from {db}.auction_audit_cln (daily_snapshot or realtime source).

    Mirrors _resolve_seed_docs's source-label resolution, Athena-client construction,
    and credential check, but delegates to fetch_seed_pages_from_db so a caller can
    index/write each page before fetching the next — bounding peak RAM to
    O(page_size) instead of O(max_records).

    Yields (page_docs, missing_cols, table_name, base_meta) per page. When Athena is
    unconfigured or credentials are unavailable, yields exactly one sentinel
    (page_docs=[], missing_cols=[], table_name=None, base_meta with tables_skipped
    set to the full table count) instead of raising, so a caller never has to
    special-case "the generator produced nothing".

    :param timing: StageTimingSession - Shared config-driven stage timer (required)
    """
    db_cfg = seed_cfg.database
    if seed_cfg.source == "realtime":
        if not db_cfg.realtime_name.strip():
            raise ConfigurationError(
                "vectorization.seed.source is 'realtime' but "
                "database.realtime_name is not configured"
            )
        resolved_db_name = db_cfg.realtime_name
        source_label = "realtime"
    else:
        resolved_db_name = db_cfg.name
        source_label = "daily_snapshot"

    base_meta = {"source": seed_cfg.source, "database": resolved_db_name}

    athena_cfg = config.nl_to_sql.athena
    if athena_cfg is None:
        logger.warning(f"_resolve_seed_pages database_seeding_skipped database={resolved_db_name} reason=athena_config_absent")
        yield [], [], None, {**base_meta, "tables_skipped": len(db_cfg.tables)}
        return
    athena_client = AthenaClient(athena_cfg)
    if not athena_client.credentials_available:
        logger.warning(
            f"_resolve_seed_pages database_seeding_skipped database={resolved_db_name} "
            f"reason=credentials_unavailable"
        )
        yield [], [], None, {**base_meta, "tables_skipped": len(db_cfg.tables)}
        return

    effective_db_cfg = replace(db_cfg, name=resolved_db_name) if resolved_db_name != db_cfg.name else db_cfg
    async with contextlib.aclosing(
        fetch_seed_pages_from_db(
            athena_client, effective_db_cfg, source_label,
            run_token=run_token,
            timing=timing,
        )
    ) as pages:
        async for page_docs, missing_cols, table_name in pages:
            yield page_docs, missing_cols, table_name, base_meta


def _build_encoder(config: AgentSearchConfig) -> Encoder:
    """Build shared text encoder (fastembed -> local model, fallback to hashing)."""
    enc_cfg = config.qi.encoder
    if enc_cfg.backend == 'fastembed':
        try:
            return FastEmbedEncoder(
                model_name=enc_cfg.model_name,
                dim=enc_cfg.dim,
                local_model_path=enc_cfg.local_model_path,
                threads=enc_cfg.threads if enc_cfg.threads > 0 else None,
                max_length=enc_cfg.max_length,
                batch_size=enc_cfg.batch_size,
                log_local_path_at_info=config.general.startup_log_detail,
                query_prefix=enc_cfg.query_prefix,
            )
        except ConfigurationError as exc:
            logger.warning(
                f"fastembed_encoder_unavailable error={exc} "
                f"— falling back to HashingEncoder; semantic recall quality is degraded"
            )
    return HashingEncoder(dim=enc_cfg.dim, seed=config.qi.semantic.encoder_seed)


def _build_retrieved_content_sanitizer(config: AgentSearchConfig, sanitizer: Optional[LayerZeroSanitizer], signal_store: Optional[SignalStore]) -> Optional[RetrievedContentSanitizer]:
    """Build per-fragment retrieved-content sanitizer (None when disabled or underlying sanitizer unavailable)."""
    rcs_cfg = config.nl_to_sql.retrieved_content_sanitizer
    if rcs_cfg is None or not rcs_cfg.enabled:
        logger.info("retrieved_content_sanitizer_disabled reason=config")
        return None
    if sanitizer is None:
        logger.warning(
            "retrieved_content_sanitizer_disabled reason=layer_zero_sanitizer_unavailable"
        )
        return None
    try:
        return RetrievedContentSanitizer(
            config=rcs_cfg, sanitizer=sanitizer, signal_store=signal_store,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            f"retrieved_content_sanitizer_disabled reason=construction_failed "
            f"error_type={type(e).__name__} error={e}"
        )
        return None


def _build_nl_to_sql_pipeline(
    config: AgentSearchConfig,
    call_router: LLMCallRouter,
    ch_executor: ClickHouseExecutor,
    sanitizer: Optional[LayerZeroSanitizer] = None,
    signal_store: Optional[SignalStore] = None,
    retrieved_content_sanitizer: Optional[RetrievedContentSanitizer] = None,
) -> NLToSQLPipeline:
    """Build NL-to-SQL pipeline (catalog -> generator -> verifier -> executor). Failures logged; caller gets usable graph."""
    nl_cfg = config.nl_to_sql
    catalog = JsonSchemaCatalog(nl_cfg.schema_discovery.catalog_path, canonical_table=nl_cfg.table)
    discoverer = SchemaDiscoverer(nl_cfg.schema_discovery, catalog)
    content_sanitizer = retrieved_content_sanitizer
    generator = SqlGenerator(
        config=nl_cfg.generation,
        call_router=call_router,
        dialect=nl_cfg.sql_dialect,
        max_rows=nl_cfg.execution.max_rows,
        content_sanitizer=content_sanitizer,
        signal_store=signal_store,
    )
    security_validator = AstSecurityValidator(nl_cfg.security, nl_cfg.sql_dialect, schema_catalog=catalog)
    cost_classifier = CostClassifier(nl_cfg.cost_class) if nl_cfg.cost_class is not None else None
    explain_probe = make_clickhouse_explain_probe(ch_executor) if nl_cfg.validation.enable_explain_probe else None
    logic_validator = LogicValidator(nl_cfg.validation, nl_cfg.sql_dialect, explain_probe=explain_probe, cost_classifier=cost_classifier)
    executor = SqlExecutor(nl_cfg.execution, ch_executor)
    verifier = Verifier(nl_cfg.verifier, call_router, content_sanitizer=content_sanitizer)
    pipeline = NLToSQLPipeline(
        config=nl_cfg,
        schema_discoverer=discoverer,
        sql_generator=generator,
        security_validator=security_validator,
        logic_validator=logic_validator,
        sql_executor=executor,
        verifier=verifier,
    )
    logger.info(
        f"nl_to_sql_pipeline_built table={nl_cfg.database}.{nl_cfg.table} "
        f"dialect={nl_cfg.sql_dialect} executor_credentials={executor.credentials_available} "
        f"retrieved_content_sanitizer_wired={content_sanitizer is not None}"
    )
    return pipeline


def _build_price_band_store(config: AgentSearchConfig, backend_health: BackendHealthRegistry) -> PriceBandStore:
    """Build PriceBandStore (InMemory by default, ClickHouse when adapter enabled). Each CH client pools own connections."""
    sql_cfg = config.retrieval.sql
    adapter_cfg = sql_cfg.clickhouse_adapter
    if adapter_cfg is None or not adapter_cfg.enabled:
        logger.info("price_band_store backend=in_memory adapter_enabled=false")
        return InMemoryPriceBandStore()
    nl_cfg = config.nl_to_sql
    if nl_cfg.analytics is None:
        # Adapter wants ClickHouse but no CH client config exists. Degrade to the
        # in-memory store and warn rather than crash boot — the service must come
        # up and serve hybrid search even when the analytics/CH side is absent.
        logger.warning(
            "price_band_store backend=in_memory reason=adapter_enabled_but_no_nl_to_sql_analytics_config "
            "action=degraded_to_in_memory"
        )
        return InMemoryPriceBandStore()
    password = os.environ.get('CLICKHOUSE_PASSWORD', '') or None
    ch_client = ClickHouseClient(
        config=nl_cfg.analytics.clickhouse,
        password=password,
        transport=None,
        health_registry=backend_health,
    )
    store = ClickHousePriceBandStore(config=adapter_cfg, client=ch_client)
    logger.info(f"price_band_store backend=clickhouse table={adapter_cfg.table} ch_available={ch_client.available}")
    return store


def _nl_analytics_clickhouse_stack(config: AgentSearchConfig, health_registry: BackendHealthRegistry) -> tuple[ClickHouseClient, ClickHouseExecutor]:
    """Build CH client + executor for analytics (shared by router + guidance)."""
    nl_cfg = config.nl_to_sql
    if nl_cfg.analytics is None:
        raise ConfigurationError("_nl_analytics_clickhouse_stack requires nl_to_sql.analytics")
    a_cfg = nl_cfg.analytics
    password = os.environ.get('CLICKHOUSE_PASSWORD', '') or None
    ch_client = ClickHouseClient(config=a_cfg.clickhouse, password=password, transport=None, health_registry=health_registry)
    ch_exec_config = SqlExecutionConfig(max_rows=nl_cfg.execution.max_rows, timeout_seconds=float(a_cfg.clickhouse.execution_timeout_seconds))
    ch_executor = ClickHouseExecutor(config=ch_exec_config, client=ch_client)
    return ch_client, ch_executor


def _build_analytics_router(
    config: AgentSearchConfig,
    pipeline: NLToSQLPipeline,
    call_router: LLMCallRouter,
    health_registry: BackendHealthRegistry,
    ch_executor: ClickHouseExecutor,
    sanitizer: Optional[LayerZeroSanitizer] = None,
    signal_store: Optional[SignalStore] = None,
    retrieved_content_sanitizer: Optional[RetrievedContentSanitizer] = None,
) -> AnalyticsRouter:
    """Build real-time analytics router (CH password from CLICKHOUSE_PASSWORD env; health_registry shared)."""
    nl_cfg = config.nl_to_sql
    if nl_cfg.analytics is None:
        raise ConfigurationError("_build_analytics_router called with nl_to_sql.analytics=None")
    a_cfg = nl_cfg.analytics
    ch_client = ch_executor._client  # noqa: SLF001
    mv_router = MVRouter(config=a_cfg.mv_router, dialect='clickhouse')
    # Wire the MV router into the canonical 6-stage pipeline so its
    # `run()` exercises Stage 5 (MV-aware rewrite) regardless of which
    # executor ultimately runs the SQL. A downstream caller that swaps
    # the executor (e.g. a ClickHouse-only deployment that disables
    # analytics_router but still wants MV rewrites) gets the rewrite for
    # free. The router is a no-op when `mv_router.enabled` is False
    # (i.e. when the materialized_views catalog is empty).
    pipeline._mv_router = mv_router  # noqa: SLF001
    nl_sql_redis = None
    if a_cfg.exact_cache.remote.enabled:
        ru = os.environ.get(a_cfg.exact_cache.remote.redis_url_env_var, '')
        if ru:
            try:
                nl_sql_redis = _redis_mod.Redis.from_url(
                    ru,
                    socket_timeout=float(a_cfg.exact_cache.remote.socket_timeout_seconds),
                    socket_connect_timeout=float(a_cfg.exact_cache.remote.socket_timeout_seconds),
                )
                nl_sql_redis.ping()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"nl_sql_redis_unavailable error_type={type(e).__name__} error={str(e)}")
                nl_sql_redis = None
        else:
            logger.warning(
                f"nl_sql_redis_disabled reason=missing_env var={a_cfg.exact_cache.remote.redis_url_env_var}"
            )
    exact_cache = NLSqlExactCache(config=a_cfg.exact_cache, redis_client=nl_sql_redis)
    security_validator = AstSecurityValidator(nl_cfg.security, 'clickhouse', schema_catalog=pipeline.schema_catalog)
    content_sanitizer = retrieved_content_sanitizer
    verifier = Verifier(nl_cfg.verifier, call_router, content_sanitizer=content_sanitizer)
    snapshot_port = None
    if (a_cfg.capabilities is not None and a_cfg.capabilities.historical_snapshots.enabled):
        snapshot_port = HistoricalSnapshotAnalyticsPort(config=a_cfg.capabilities.historical_snapshots, ch_executor=ch_executor)
    engine_dispatcher = None
    if (
        a_cfg.capabilities is not None
        and a_cfg.capabilities.domain_analytics is not None
        and a_cfg.engine_dispatch is not None
        and a_cfg.engine_dispatch.enabled
    ):
        try:
            domain_engine = DomainAnalyticsEngine(
                executor=ch_executor,
                config=a_cfg.capabilities.domain_analytics,
                capabilities=a_cfg.capabilities,
            )
            engine_dispatcher = AnalyticsEngineDispatcher(
                engine=domain_engine,
                config=a_cfg.engine_dispatch,
            )
            logger.info(f"analytics_engine_dispatcher_built routes={len(a_cfg.engine_dispatch.routes)}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"analytics_engine_dispatcher_build_failed route_count={len(a_cfg.engine_dispatch.routes)} error_type={type(e).__name__} error={e}")
            engine_dispatcher = None
    router = AnalyticsRouter(
        config=a_cfg,
        pipeline=pipeline,
        cache=exact_cache,
        mv_router=mv_router,
        security_validator=security_validator,
        ch_executor=ch_executor,
        verifier=verifier,
        verifier_config=nl_cfg.verifier,
        current_schema_version='',
        engine_dispatcher=engine_dispatcher,
        snapshot_port=snapshot_port,
    )
    logger.info(
        f"analytics_router_built ch_available={ch_client.available} "
        f"mv_catalog_size={mv_router.catalog_size} "
        f"exact_cache_enabled={a_cfg.exact_cache.enabled} "
        f"freshness_target_s={a_cfg.freshness_target_seconds:.1f} "
        f"snapshot_port_wired={snapshot_port is not None} "
        f"engine_dispatcher_wired={engine_dispatcher is not None}"
    )
    return router


def _build_retrieval_backends(
    config: AgentSearchConfig,
    encoder: Encoder,
    shortlist_encoder: Optional[StageEncoder] = None,
    rerank_encoder: Optional[StageEncoder] = None,
    word_segmenter: Optional[DomainNameSegmenter] = None,
) -> 'tuple[Optional[QdrantClientFactory], VectorIndex, StructuredIndex, Retriever, Retriever, Optional[DynamicSynonymStore]]':
    """Build vector + structured retrievers (memory or qdrant). Soft-fails to memory on qdrant unavailable; hybrid mode supported."""
    rcfg = config.retrieval
    wants_qdrant_vector = rcfg.vector.backend == 'qdrant'
    wants_qdrant_structured = rcfg.structured.backend == 'qdrant'
    wants_hybrid = wants_qdrant_vector and wants_qdrant_structured and (
        rcfg.qdrant is not None and rcfg.qdrant.hybrid.enabled
    )
    factory: Optional[QdrantClientFactory] = None
    if (wants_qdrant_vector or wants_qdrant_structured) and rcfg.qdrant is not None:
        factory = QdrantClientFactory(rcfg.qdrant)
        if not factory.available:
            logger.warning(
                "qdrant_factory_unavailable falling_back_to_memory "
                f"vector_backend={rcfg.vector.backend} structured_backend={rcfg.structured.backend}"
            )
            factory = None
    use_qdrant_vector = wants_qdrant_vector and factory is not None
    use_qdrant_structured = wants_qdrant_structured and factory is not None
    use_hybrid = wants_hybrid and factory is not None
    vector_index: VectorIndex
    if use_qdrant_vector:
        qd_vector = QdrantVectorIndex(factory)
        qd_vector.set_dim_hint(rcfg.vector.embedding_dim)
        vector_index = qd_vector
    else:
        vector_index = InMemoryVectorIndex(dim=rcfg.vector.embedding_dim)
    structured_index: StructuredIndex
    if use_qdrant_structured:
        structured_index = QdrantStructuredIndex(factory)
    else:
        structured_index = InMemoryStructuredIndex()
    vector_retriever: Retriever
    structured_retriever: Retriever
    # Shortlist stage: when the cascade is wired with a shortlist
    # stage encoder, hand it to the vector retriever in place of the base
    # encoder. The retriever's own ``encoder.dim != embedding_dim`` check
    # is satisfied by ``StageEncoder.dim`` because the registry validated
    # the cross-config equality before constructing the StageEncoder.
    vector_encoder: Encoder = shortlist_encoder if shortlist_encoder is not None else encoder
    batch_cfg = config.qi.encoder.batching
    if batch_cfg is not None and batch_cfg.enabled:
        vector_encoder = BatchingEncoder(base=vector_encoder, window_ms=batch_cfg.window_ms)
        logger.info(f"batching_encoder_enabled window_ms={batch_cfg.window_ms}")
    # Query-side compound splitter. Reuses the same dictionary + cost model the
    # ingest splitter applied to document labels so the query encode text splits
    # into the segments the index stored. Soft-fail: a missing / empty / malformed
    # dictionary disables the preprocessor (the boot path never crashes per
    # oversight.mdc); the encode text is then embedded verbatim.
    query_preprocessor: Optional[Callable[[str], str]] = None
    qcs_cfg = rcfg.query_compound_split
    if qcs_cfg is not None and qcs_cfg.enabled:
        try:
            _qsplit_dict = _load_compound_splitter_dictionary(qcs_cfg.dictionary_path)
            _qsplitter = CompoundWordSplitter(
                dictionary=_qsplit_dict,
                min_segment_length=qcs_cfg.min_segment_length,
                max_segments=qcs_cfg.max_segments,
                oov_char_cost=qcs_cfg.oov_char_cost,
                length_penalty=qcs_cfg.length_penalty,
            )
            query_preprocessor = QueryCompoundExpander(lambda label: _qsplitter.split(label).segments)
            logger.info(f"query_compound_split_built dictionary_size={_qsplitter.dictionary_size} min_segment_length={qcs_cfg.min_segment_length}")
        except (ValidationError, ConfigurationError) as e:
            logger.warning(f"query_compound_split_disabled error_type={type(e).__name__} error={str(e)} path={mask_path(qcs_cfg.dictionary_path)}")
            query_preprocessor = None
    _dynamic_synonym_store: Optional[DynamicSynonymStore] = None
    if use_hybrid:
        # Option A: hybrid retriever takes the vector slot (source='vector'),
        # structured slot becomes a no-op (source='structured', empty result).
        # When `bm25_enabled=true`, build the BM25 sparse query encoder from
        # config and hand it to the hybrid retriever as `bm25_query_fn` so
        # Qdrant fuses dense + sparse server-side in a single hybrid query.
        bm25_query_fn: Optional[Any] = None
        if rcfg.qdrant.hybrid.bm25_enabled:
            # Try BM42 attention-weighted sparse encoder first; fall back to
            # hash-BM25 if the local model directory is absent or unusable.
            _sparse_cfg = rcfg.qdrant.hybrid.sparse_encoder
            if _sparse_cfg is not None:
                try:
                    bm25_query_fn = BM42SparseEncoder(
                        model_name=_sparse_cfg.model_name,
                        local_model_path=_sparse_cfg.local_model_path,
                        log_local_path_at_info=config.general.startup_log_detail,
                        threads=_sparse_cfg.threads if _sparse_cfg.threads > 0 else None,
                        query_stop_list=_sparse_cfg.query_stop_list,
                    )
                    logger.info(
                        f"bm42_query_encoder_loaded model={_sparse_cfg.model_name}"
                    )
                except ConfigurationError as _bm42_err:
                    if not _sparse_cfg.hash_bm25_fallback:
                        raise ConfigurationError(
                            f"bm42_query_encoder_required hash_bm25_fallback=false "
                            f"error={_bm42_err}"
                        ) from _bm42_err
                    logger.warning(
                        f"bm42_query_encoder_unavailable error={_bm42_err} "
                        "falling_back_to_hash_bm25 hash_bm25_fallback=true"
                    )
            if bm25_query_fn is None:
                if _sparse_cfg is not None and not _sparse_cfg.hash_bm25_fallback:
                    raise ConfigurationError(
                        "bm42_query_encoder_required sparse_encoder present but "
                        "BM42 not loaded and hash_bm25_fallback=false"
                    )
                assert rcfg.qdrant.hybrid.bm25_query_encoder is not None  # config validates this
                bm25_query_fn = BM25QueryEncoder(rcfg.qdrant.hybrid.bm25_query_encoder)
        # Expose the dynamic store (only present on BM25QueryEncoder fallback).
        _dynamic_synonym_store = getattr(bm25_query_fn, 'dynamic_store', None)
        # Second-stage dense rescore: active only when the cascade built a
        # rerank-stage encoder AND the hybrid config enables rerank. The rerank
        # query is encoded at the rerank stage dim (768); the retriever issues a
        # nested-prefetch query that reranks the fused pool by the higher-dim
        # ``dense_rerank`` vector written at index time.
        _rerank_cfg = rcfg.qdrant.hybrid.rerank
        _rerank_active = rerank_encoder is not None and _rerank_cfg is not None and _rerank_cfg.enabled
        # Character n-gram fuzzy-recall channel (query side). Same encoder config
        # the indexer used on the doc side so doc + query share the bucket space.
        _ngram_cfg = rcfg.qdrant.hybrid.ngram
        _ngram_query_fn: Optional[Any] = None
        if _ngram_cfg is not None and _ngram_cfg.enabled:
            _ngram_query_fn = CharNgramSparseEncoder(vocab_size=_ngram_cfg.vocab_size, min_n=_ngram_cfg.min_n, max_n=_ngram_cfg.max_n)
            logger.info(f"char_ngram_query_encoder_built vector_name={_ngram_cfg.vector_name} vocab_size={_ngram_cfg.vocab_size} min_n={_ngram_cfg.min_n} max_n={_ngram_cfg.max_n}")
        vector_retriever = QdrantHybridRetriever(
            factory=factory,
            encoder=vector_encoder,
            embedding_dim=rcfg.vector.embedding_dim,
            min_similarity=rcfg.vector.min_similarity,
            bm25_query_fn=bm25_query_fn,
            rerank_encoder=rerank_encoder if _rerank_active else None,
            rerank_vector_name=_rerank_cfg.vector_name if _rerank_active else None,
            rerank_dim=_rerank_cfg.dim if _rerank_active else None,
            rerank_input_n=_rerank_cfg.input_n if _rerank_active else None,
            ngram_query_fn=_ngram_query_fn,
            query_preprocessor=query_preprocessor,
            keyword_match_mode=rcfg.structured.keyword_match_mode,
        )
        # Eager warm: BM42/ngram already loaded with lazy_load=False; one sync encode
        # forces ONNX session fully resident before first search request.
        _warm_text = "domain"
        if bm25_query_fn is not None:
            bm25_query_fn(_warm_text)
            logger.info("bm42_query_encoder_warmed")
        if _ngram_query_fn is not None:
            _ngram_query_fn(_warm_text)
            logger.info("char_ngram_query_encoder_warmed")
        structured_retriever = QdrantNoOpStructuredRetriever()
        logger.info(
            f"retrieval_backends_wired mode=hybrid bm25={rcfg.qdrant.hybrid.bm25_enabled} ngram={_ngram_cfg.enabled if _ngram_cfg is not None else False} fusion={rcfg.qdrant.hybrid.fusion_strategy} shortlist_stage_encoder={'on' if shortlist_encoder is not None else 'off'} rerank_stage={'on' if _rerank_active else 'off'}"  # noqa: E501
        )
    else:
        vector_retriever = VectorRetriever(
            config=rcfg.vector, encoder=vector_encoder, index=vector_index,
            query_preprocessor=query_preprocessor,
        )
        structured_retriever = StructuredRetriever(
            config=rcfg.structured, index=structured_index, word_segmenter=word_segmenter,
        )
        logger.info(
            f"retrieval_backends_wired mode=classic vector={'qdrant' if use_qdrant_vector else 'memory'} structured={'qdrant' if use_qdrant_structured else 'memory'}"
        )
    return factory, vector_index, structured_index, vector_retriever, structured_retriever, _dynamic_synonym_store
