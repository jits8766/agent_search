"""Typed contracts shared across QI, retrieval, cache, surface, and feedback layers.
Every cross-module hand-off uses these dataclasses — never raw `Dict[str, Any]`.
Each `__post_init__` enforces invariants so a malformed instance fails at construction.
"""
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, FrozenSet, List, Optional, Protocol, runtime_checkable

from semantic_search.core.exceptions import ValidationError

QUERY_TYPES = frozenset({'hybrid', 'guidance', 'explore', 'analytics'})

# User-facing cache tiers stamped on ``RankedResults.cache_hit`` / ``SearchObservation.cache_hit``.
CACHE_HIT_TIER_VALUES = frozenset({'exact', 'semantic', 'structured', 'intent_plan'})

DECISION_TIERS = frozenset({
    # QueryIntent.decision_tier — stamped by EnsembleResolver (by winning voter)
    # and QIEngine. L0_entity covers all non-LLM voter wins (entity extractor,
    # aggregation_gate, ngram_gate, entity_type_voter).
    'L0_entity', 'L0_multi_intent', 'L0_fallback',
    'L1_semantic',
    'L2_llm',
    'fallback',
    # EnsembleResolver: every voter abstained.
    'ensemble_all_abstain',
    # Entity.source tag written by L0LLMFilterExtractor (config qi.l0_llm_entity.source_tag).
    'L0_llm',
    # Entity.source tag written by RegexEntityExtractor (config qi.l0_regex_entity.source_tag).
    'L0_regex',
    # Internal synthetic tier for startup prewarm queries (never user-facing).
    'prewarm',
})

# Canonical auction-type labels -> numeric string IDs stored in the index payload.
# Single source of truth shared by qi.grounding (entity expansion) and
# retrieval.structured_retriever (filter expansion); numeric ID strings
# ('16', '38') pass through downstream unchanged.
#
# From analysis_auction.md — Types of Auctions We Care About:
#   16 = GoDaddy AutoExtend (Expireds)     20 = GoDaddy BuyNow (Closeouts)
#   38 = Partner AutoExtend (Expireds)     39 = Partner Closeout
#   25 = Drop Catch (Private Backorder)    37 = Firehose (Pre-registration)
AUCTION_TYPE_LABEL_TO_IDS: Dict[str, frozenset] = {
    'auction':          frozenset({'16', '38'}),
    'expiry':           frozenset({'16', '38'}),
    'closeout':         frozenset({'39'}),
    'buynow':           frozenset({'20'}),
    'buy_now':          frozenset({'20'}),
    'premium':          frozenset({'16', '38', '39'}),
    # Registrar-scoped — do not mix with the opposing registrar's IDs.
    'partner':          frozenset({'38', '39'}),
    'godaddy':          frozenset({'16', '20'}),
    # Private / specialty formats.
    'backorder':        frozenset({'25'}),
    'dropcatch':        frozenset({'25'}),
    'drop_catch':       frozenset({'25'}),
    'firehose':         frozenset({'37'}),
    'preregistration':  frozenset({'37'}),
    'pre_registration': frozenset({'37'}),
}

# When a registrar-scoped label is present, drop generic auction/expiry/premium
# and the opposing registrar's numeric IDs so "partner auction" never pulls 16.
_AUCTION_REGISTRAR_LABELS: FrozenSet[str] = frozenset({'partner', 'godaddy'})
_AUCTION_GENERIC_LABELS: FrozenSet[str] = frozenset({'auction', 'expiry', 'premium'})
_GODADDY_AUCTION_TYPE_IDS: FrozenSet[str] = frozenset({'16', '20'})
_PARTNER_AUCTION_TYPE_IDS: FrozenSet[str] = frozenset({'38', '39'})


def prefer_registrar_auction_values(values: List[str]) -> List[str]:
    """Prefer partner/godaddy labels over generic auction labels and opposing IDs.

    ``partner auction`` often also extracts generic ``auction``/``expiry`` (-> 16+38).
    When a registrar label is present, drop generics and the other registrar's IDs.
    """
    lowered = [str(v).lower() for v in values]
    has_partner = 'partner' in lowered
    has_godaddy = 'godaddy' in lowered
    if not has_partner and not has_godaddy:
        return lowered
    out: List[str] = []
    for v in lowered:
        if v in _AUCTION_GENERIC_LABELS:
            continue
        if has_partner and not has_godaddy and v in _GODADDY_AUCTION_TYPE_IDS:
            continue
        if has_godaddy and not has_partner and v in _PARTNER_AUCTION_TYPE_IDS:
            continue
        out.append(v)
    return out


SIGNAL_TYPES = frozenset({
    'filter_override',
    'query_rephrase',
    'result_click',
    'result_dismiss',
    'resume_clicked',
    'history_optout',
    'eranker_applied',
    'chip_promoted',
    'chip_demoted',
    # NL-to-SQL pipeline failure when ``AnalyticsResult.success=False``; payload carries ``failure_mode``.
    'analytics_failure',
    # Offline calibration label: ``tier``, ``raw_confidence``, ``is_correct`` on payload.
    'calibration_label',
    # Emitted by ``CacheMissStormDetector`` on breach transitions; payload carries hit_rate and window.
    'cache_miss_storm',
    # L2 LLM classify timeout; ``LLMClassifier`` emits with timing fields on payload.
    'llm_timeout',
    # SQL generator structural refusal; ``SqlGenerator`` emits capped reason fields on payload.
    'llm_refused',
    # Breaker state transition; payload uses ``BREAKER_TRANSITION_BREAKER_IDS`` and ``fallback_kind``.
    'breaker_transition',
    # Retrieved fragment masked before LLM ingress; payload ``kind`` + ``reasons`` (see nl_to_sql content sanitizer).
    'retrieved_content_sanitized',
    # UAT free-text comment + optional star rating submitted by end users via POST /feedback.
    'uat_feedback',
    # CentroidRetrainerDriver emits when decide() produces a verdict (promote/shadow_only/reject).
    'centroid_retrain_verdict',
    # CentroidRetrainerDriver emits when verdict='promote' and swap_centroids fires on the live router.
    'centroid_retrain_promoted',
})

# Allowed slice keys for ``ProxySignalEvaluator.evaluate_sliced(by=<key>)``.
# Slicing rate-bearing signals by intent
# bucket (`query_type`) lets operators see "did latency regress globally OR only
# for analytics queries?" Today only `query_type` is supported because it is the
# only categorical attribute on every SearchObservation; `decision_tier` is a
# natural future addition (extensible via this frozenset, no contract break).
MEASUREMENT_SLICE_KEYS = frozenset({'query_type'})

# Closed enum of `FeedbackSignal.signal_origin` values. Every feedback
# signal carries which subsystem produced it so dashboards and downstream
# consumers can slice by producer without inferring origin from `signal_type`
# heuristics. Origin is mandatory on every emit site; legacy emitters use
# 'unknown' (visible in dashboards as untyped, surfaces an actionable migration
# backlog). Distinct from `SIGNAL_ORIGINS` below which classifies *what user
# behaviour* emitted the signal — this enum classifies *which subsystem*
# emitted it.
FEEDBACK_SIGNAL_ORIGINS = frozenset({
    'orchestrator',          # /search, /analytics request handlers
    'qi_engine',             # QI cascade tiers (regex/router/llm/decompose)
    'retrieval',             # vector / structured / sql retrievers
    'analytics_router',      # NL-SQL fast-path + verifier
    'cache',                 # exact / semantic / structured tiers
    'eranker',                # external eRanker ranking layer
    'circuit_breaker',       # LLM / backend breakers
    'offline_eval',          # library-only retrieval eval / LLM judge hooks
    'measurement',           # proxy-signal evaluator
    'frontend',              # browser-emitted user signals (clicks, chips)
    'unknown',               # legacy callers without a known origin
    'centroid_retrainer_driver',  # background centroid retrain cycle driver
})

# Feedback-signal types that count as "positive engagement" for the
# search_assisted_conversion_rate proxy. A search converts when the same
# `session_id` issues at least one of these signal types within the configured
# post-search lookback window AFTER the search was recorded. Default set is intentionally
# conservative — only signal types that represent unambiguous high-intent
# engagement (click, chip promote, save-trigger). Operators tune via
# `measurement.assisted_conversion.positive_signal_types` in YAML.
ASSISTED_CONVERSION_POSITIVE_SIGNALS = frozenset({
    'result_click',
    'chip_promoted',
    'resume_clicked',
})

# Closed enum of stages that may emit a ``ReasoningStep`` onto a
# ``QueryIntent`` or ``RankedItem`` reasoning trace. Closed by design so trace
# consumers (debug UI, audit log) can switch on a finite set
# without any string typo silently masking an unrecognised stage. Each stage
# corresponds to a single named pipeline phase the orchestrator runs through.
# Stage taxonomy (ordering matches typical execution flow):
#   - spell_correct   — Tier-0 ``SymSpellCorrector`` outcome
#   - cache_lookup    — exact / semantic / structured / intent_plan cache hit/miss
#   - qi_classify     — Query Intelligence engine classification verdict
#   - retrieve        — per-backend candidate fetch (vector / structured / sql)
#   - fuse            — RRF fusion of multi-source candidates
#   - erank           — external eRanker call (soft-fail / latency-gated)
#   - diversify       — MMR diversifier decision
#   - truncate        — final top-K cut
#   - egress_guard    — output-side PII / moderation / grounding
# When a new pipeline phase ships, add it here AND extend the orchestrator
# emitter so the closed set never lags the producer surface.
REASONING_STAGES = frozenset({
    'spell_correct',
    'cache_lookup',
    'qi_classify',
    'retrieve',
    'fuse',
    'erank',
    'diversify',
    'truncate',
    'egress_guard',
})

CANDIDATE_SOURCES = frozenset({'vector', 'structured', 'sql'})

USER_MODES = frozenset({'conversational', 'advanced'})

GOLDEN_DIFFICULTIES = frozenset({'easy', 'hard', 'edge'})

GOLDEN_EDGE_TYPES = frozenset({'domain_boundary', 'complexity', 'ambiguity', 'adversarial', 'low_frequency', 'multi_intent'})

GOLDEN_REVIEW_STATUSES = frozenset({'pending', 'approved', 'rejected'})

SIGNAL_ORIGINS = frozenset({
    'explicit_negative',
    'filter_removed',
    'filter_modified',
    'rephrase',
    'clarification_turn_2',
    'zero_result',
    'high_confidence_override',
    'manual',
})

BACKEND_HEALTH_BACKENDS = frozenset({'vector', 'structured', 'sql', 'llm', 'clickhouse', 'eranker', 'bulk'})

# Emit sites for ``breaker_transition`` are the LLM ``CircuitBreaker`` and the
# per-slot ``BACKEND_HEALTH_BACKENDS`` health ids (``llm`` covers the breaker).
BREAKER_TRANSITION_BREAKER_IDS = frozenset(BACKEND_HEALTH_BACKENDS)

# Typed recovery hint for dashboards. Distinct from free-text ``reason`` — stable join key for "was traffic on fallback?".
BREAKER_TRANSITION_FALLBACK_KINDS = frozenset({'normal', 'fallback_active', 'degraded', 'probe'})

ANALYTICS_SUBSTRATE_MODES = frozenset({'disabled', 'clickhouse_primary'})

BACKEND_HEALTH_STATES = frozenset({'healthy', 'degraded', 'unhealthy'})

CIRCUIT_STATES = frozenset({'closed', 'open', 'half_open'})

BASELINE_NAMES = frozenset({'always_llm', 'always_rules', 'always_largest', 'cascade'})

PROXY_SIGNAL_STATUSES = frozenset({'ok', 'breach', 'insufficient_data', 'not_instrumented'})

PROXY_SIGNAL_DIRECTIONS = frozenset({'lower_is_better', 'higher_is_better', 'informational'})

# Origin tags for router seed queries. Track provenance so every seed in any
# centroid can be traced back to either a hand-curator, an LLM-synthesis
# batch, or a real-traffic retrain.
ROUTER_SEED_ORIGINS = frozenset({'manual', 'synthetic', 'retrain'})

# Verdict from a centroid retrain shadow run (promote / reject / shadow_only).
CENTROID_RETRAIN_VERDICTS = frozenset({'promote', 'reject', 'shadow_only'})

# Explore rails + landing rail. Rails come from real data sources (trending /
# ending_soon); the 'fallback' rail is reserved when every source is empty so the
# composer can still return a non-empty landing response (never dead-end).
EXPLORE_RAIL_KINDS = frozenset({'trending', 'ending_soon', 'fallback', 'latest', 'last_hour', 'high_volume', 'fresh', 'last_week', 'watch_density', 'high_traffic'})

# Call-path provenance for the explore composer. The same
# composer serves the public /landing-rail endpoint and the orchestrator's
# zero-result fallback; the dashboard separates the two so we can measure rail
# usefulness on each surface independently.
EXPLORE_RESPONSE_SOURCES = frozenset({'landing_rail_endpoint', 'zero_result_guard'})

# The ladder steps the Zero-Result Guard walks in order.
# 'none' means the guard did not fire (results were already non-empty).
ZERO_RESULT_LADDER_STEPS = frozenset({'none', 'widen_filters', 'relax_filters', 'semantic_only', 'explore_fallback'})

# Chip kind drives the promote/demote feedback channel.
# 'hard' = deterministic filter (TLD, price band, auction type, length cap) — promote re-applies the filter,
#          demote drops the filter slot from the IntentSlice.
# 'soft' = aspirational signal (brandability, quality, aesthetic) — promote re-injects the signal,
#          demote suppresses it from future cards.
# The Zero-Result Guard's relax-filters ladder consults this field to drop hard slots
# in priority order (price -> length -> auction_type -> tld) before retrying.
CHIP_KINDS = frozenset({'hard', 'soft'})

# Typed handshake between BackendHealth -> DegradationPlanner -> orchestrator.
# Modes the planner may emit:
#  'normal'     — every requested backend is healthy; no fallback applied.
#  'degraded'   — mode name in config: at least one retrieval backend inactive; partial coverage.
#  'cache_only' — every backend is unhealthy; the orchestrator must serve from cache or return empty
#                 with an explicit explanation (`allow_empty_when_all_unhealthy=True`).
DEGRADATION_MODES = frozenset({'normal', 'degraded', 'cache_only'})

# Retrieval-side backends the fallback planner (`DegradationPlanner`) may mark active.
# Sourced from the planner's internal ``_RETRIEVAL_BACKENDS`` constant
# (``{vector, structured, sql}``) and shared here so contract validation
# matches the planner's runtime gate without an import cycle.
DEGRADATION_RETRIEVAL_BACKENDS = frozenset({'vector', 'structured', 'sql'})


def _new_id(prefix: str) -> str:
    """Generate a short unique id with a stable prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class Entity:
    """A single entity extracted from a query.
    :param name: str - Entity slot name (e.g. 'tld', 'price_max', 'name_length_max')
    :param value: Any - Extracted value (typed by entity name)
    :param confidence: float - Extraction confidence in [0,1]
    :param source: str - Tier that produced the entity
    :param chip_kind: str - 'hard' = deterministic filter; 'soft' = aspirational signal
    """
    name: str
    value: Any
    confidence: float
    source: str
    chip_kind: str

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValidationError("Entity.name must be a non-empty string")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValidationError("Entity.confidence must be in [0,1]")
        if self.source not in DECISION_TIERS:
            raise ValidationError(f"Entity.source must be one of {sorted(DECISION_TIERS)}")
        if self.chip_kind not in CHIP_KINDS:
            raise ValidationError(f"Entity.chip_kind must be one of {sorted(CHIP_KINDS)}")


@dataclass
class IntentSlice:
    """One slice of a (possibly multi-intent) query.
    :param query_type: str - One of QUERY_TYPES
    :param entities: List[Entity] - Hard filter entities for this slice
    :param confidence: float - Overall confidence for this slice in [0,1]
    :param raw_text: str - Sub-query text this slice was derived from
    :param slice_id: str - Stable id used to attach badges in multi-intent fan-out.
        Empty string for single-intent flows; populated by the MultiIntentSplitter
        and the orchestrator's per-sub-intent retrieval loop.
    :param soft_entities: List[Entity] - Soft chips/topics/phrases (not in identified_filters)
    :param pre_ground_entities: Optional[List[Entity]] - Hard entities as identified
        before inventory ``EntityGrounder`` (response ``filters.identified``). ``None``
        when the slice never passed through grounding (legacy / cache). ``entities``
        remains the grounded list consumed by Qdrant / structured / CH retrieval.
    :param keywords: List[Dict[str, Any]] - ``{"term", "probability"}`` pairs from
        the L0 LLM keyword extraction. Independent of ``entities``/``soft_entities`` —
        never merged into ``identified_filters`` or the soft-signals response list.
    """
    query_type: str
    entities: List[Entity]
    confidence: float
    raw_text: str
    slice_id: str = ''
    soft_entities: List[Entity] = field(default_factory=list)
    pre_ground_entities: Optional[List[Entity]] = None
    keywords: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.query_type not in QUERY_TYPES:
            raise ValidationError(f"IntentSlice.query_type must be one of {sorted(QUERY_TYPES)}")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValidationError("IntentSlice.confidence must be in [0,1]")
        if not isinstance(self.entities, list):
            raise ValidationError("IntentSlice.entities must be a list")
        if not isinstance(self.raw_text, str):
            raise ValidationError("IntentSlice.raw_text must be a string")
        if not isinstance(self.slice_id, str):
            raise ValidationError("IntentSlice.slice_id must be a string")
        if not isinstance(self.soft_entities, list):
            raise ValidationError("IntentSlice.soft_entities must be a list")
        if self.pre_ground_entities is not None and not isinstance(self.pre_ground_entities, list):
            raise ValidationError("IntentSlice.pre_ground_entities must be a list or None")
        if not isinstance(self.keywords, list):
            raise ValidationError("IntentSlice.keywords must be a list")

    @staticmethod
    def new_slice_id() -> str:
        """Generate a stable slice id for multi-intent fan-out."""
        return _new_id('slc')


@dataclass
class SubIntentFilterSet:
    """Per-sub-intent entity snapshot captured before singleton deconfliction.

    Populated on ``QueryIntent.sub_intent_filters`` when
    ``multi_intent.preserve_sub_intent_filters`` is enabled in config.
    Each entry represents one split sub-query and the entities the entity
    extractor found in that clause before cross-intent merge logic ran.

    :param sub_query: str - Normalized sub-query text for this sub-intent
    :param entities: List[Entity] - Entities from sub_query before merge
    """
    sub_query: str
    entities: List[Entity]

    def __post_init__(self) -> None:
        if not isinstance(self.sub_query, str):
            raise ValidationError("SubIntentFilterSet.sub_query must be a string")
        if not isinstance(self.entities, list):
            raise ValidationError("SubIntentFilterSet.entities must be a list")


@dataclass(frozen=True)
class ConfidenceSignals:
    """Inputs to the calibrated-confidence transform for one classifier emission.

    Calibrated confidence combines (a) temperature-scaled raw probability with
    (b) a correctness probe trained on golden seeds. The probe needs a
    distribution-aware feature alongside the raw top-1 score: entropy of the
    score distribution captures "how peaked is the answer". A peaked
    distribution (low entropy) is more trustworthy than a flat one (high
    entropy) at the same top-1 confidence.

    L1 (semantic router) computes ``entropy_normalized`` from the softmax of
    cosine scores across all archetype centroids and ``score_margin`` from
    top1 - top2. L0 (regex) and L2 (LLM) tiers do not expose a distribution
    today; they pass ``entropy_normalized=1.0`` (max entropy = neutral signal)
    so the probe degrades to identity and combined ≡ temperature-scaled raw.

    :param raw_confidence: float - Classifier-emitted top-1 probability in [0,1]
    :param entropy_normalized: float - Shannon entropy of the score
        distribution divided by ln(N), N = number of classes. Range [0,1]:
        0 = perfectly peaked single-class answer, 1 = uniform across classes.
    :param score_margin: float - Top1 - Top2 score margin in [0,1]; 0 when
        only one class was scored. Used by the probe as a secondary feature
        when ``calibration.probe.use_margin`` is true.
    """
    raw_confidence: float
    entropy_normalized: float
    score_margin: float

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.raw_confidence) <= 1.0:
            raise ValidationError(f"ConfidenceSignals.raw_confidence must be in [0,1], got {self.raw_confidence}")
        if not 0.0 <= float(self.entropy_normalized) <= 1.0:
            raise ValidationError(f"ConfidenceSignals.entropy_normalized must be in [0,1], got {self.entropy_normalized}")
        if not 0.0 <= float(self.score_margin) <= 1.0:
            raise ValidationError(f"ConfidenceSignals.score_margin must be in [0,1], got {self.score_margin}")

    @staticmethod
    def from_raw(raw_confidence: float) -> 'ConfidenceSignals':
        """Build neutral-distribution signals for callers without a score distribution.

        Used by L0 (regex) and L2 (LLM, today) which emit a single top-1
        probability with no per-class scores. Sets ``entropy_normalized=1.0``
        (max entropy) and ``score_margin=0.0`` so the probe contributes
        neutrally and combined calibration collapses to pure temperature
        scaling — preserving today's behaviour exactly.

        :param raw_confidence: float - Classifier-emitted probability in [0,1]
        :return: ConfidenceSignals - Distribution-neutral signals
        """
        rc = float(raw_confidence)
        if rc < 0.0:
            rc = 0.0
        elif rc > 1.0:
            rc = 1.0
        return ConfidenceSignals(raw_confidence=rc, entropy_normalized=1.0, score_margin=0.0)


@dataclass(frozen=True)
class TokenCorrection:
    """One token-level correction applied by the Tier-0 spell corrector.

    Frozen so the corrector cannot accidentally mutate a correction after it
    is published on a ``SpellCorrection``. ``edit_distance`` carries the
    Damerau-Levenshtein distance the corrector used to pick ``corrected``
    over the runner-up; logged + asserted in tests so the cost bound is
    visible in the audit trail.

    :param original: str - Out-of-vocabulary token from the user's query
        (already lowercased + whitespace-stripped by ``normalize_query``).
    :param corrected: str - In-vocabulary replacement chosen from the
        frequency dictionary. Always non-empty and distinct from ``original``.
    :param edit_distance: int - Edit distance between ``original`` and
        ``corrected``. Bounded by the corrector's configured
        ``max_edit_distance``; recorded for explainability.
    """
    original: str
    corrected: str
    edit_distance: int

    def __post_init__(self) -> None:
        if not isinstance(self.original, str) or not self.original:
            raise ValidationError("TokenCorrection.original must be a non-empty string")
        if not isinstance(self.corrected, str) or not self.corrected:
            raise ValidationError("TokenCorrection.corrected must be a non-empty string")
        if self.original == self.corrected:
            raise ValidationError("TokenCorrection.corrected must differ from original")
        if not isinstance(self.edit_distance, int) or self.edit_distance < 1:
            raise ValidationError("TokenCorrection.edit_distance must be a positive int")


@dataclass(frozen=True)
class SpellCorrection:
    """Tier-0 spell-correction outcome attached to ``QueryIntent``.

    Emitted by the Tier-0 corrector BEFORE the QI cascade runs. Two surface
    modes the API can render off the same payload:

    1. ``applied=True`` (auto-apply): the corrector rewrote the query
       (``corrected_query`` differs from ``original_query``). Downstream QI +
       cache + retrieval all see the corrected text; the original is kept
       here so the response page can render "Showing results for X — search
       instead for Y".
    2. ``applied=False`` (suggest-only): the original query flowed through
       QI unchanged; ``corrected_query`` is the suggestion the UI can render
       as "Did you mean X?". When no corrections were made,
       ``corrected_query == original_query`` and ``corrections`` is empty —
       this state is dropped by the corrector before publication so a
       ``SpellCorrection`` instance always carries at least one correction.

    :param original_query: str - Normalized user query before correction.
        Length-capped + lowercased by ``QIEngine.normalize_query`` upstream.
    :param corrected_query: str - Normalized query after the corrector
        rewrote / proposed rewrites. Always non-empty and (by construction)
        distinct from ``original_query`` whenever ``corrections`` is non-empty.
    :param corrections: List[TokenCorrection] - Per-token corrections in
        the order they appear in the original query. Always non-empty for a
        published instance.
    :param applied: bool - True when the corrector rewrote the query in
        place (auto-apply mode); False when the original flowed through and
        the corrected text is only a suggestion.
    """
    original_query: str
    corrected_query: str
    corrections: List['TokenCorrection']
    applied: bool

    def __post_init__(self) -> None:
        if not isinstance(self.original_query, str) or not self.original_query:
            raise ValidationError("SpellCorrection.original_query must be a non-empty string")
        if not isinstance(self.corrected_query, str) or not self.corrected_query:
            raise ValidationError("SpellCorrection.corrected_query must be a non-empty string")
        if not isinstance(self.corrections, list) or not self.corrections:
            raise ValidationError("SpellCorrection.corrections must be a non-empty list")
        for c in self.corrections:
            if not isinstance(c, TokenCorrection):
                raise ValidationError("SpellCorrection.corrections entries must be TokenCorrection instances")
        if self.original_query == self.corrected_query:
            raise ValidationError("SpellCorrection.corrected_query must differ from original_query when any correction is recorded")
        if not isinstance(self.applied, bool):
            raise ValidationError("SpellCorrection.applied must be a bool")


# Sentinel ``stage`` value emitted when ``ReasoningTrace`` rolls
# off old steps after hitting ``max_steps``. Distinct from ``REASONING_STAGES``
# entries so consumers can detect "the trace was clipped" vs "this is a real
# pipeline step". The detail string carries the count of dropped steps.
REASONING_TRACE_TRUNCATED_STAGE = '_truncated'


@dataclass(frozen=True)
class ReasoningStep:
    """One immutable step in a ``ReasoningTrace``.

    Frozen so once a producer has emitted a step the caller cannot
    accidentally mutate it on the trace (the trace itself is mutable;
    individual steps are not). Carries only the structured trio
    ``(stage, detail, timestamp)`` — never raw query / item text — so a
    trace can be safely persisted or rendered without re-running the
    egress PII scrub on every consumer.

    :param stage: str - One of ``REASONING_STAGES`` or the sentinel
        ``REASONING_TRACE_TRUNCATED_STAGE`` emitted by the truncation guard.
    :param detail: str - Short, structured ``key=value``-style description
        of what the stage did (e.g. ``"hit=exact"``, ``"latency_ms=42.1"``).
        Bounded to 256 characters; the trace truncates oversize details
        with an ellipsis to keep the per-step payload small.
    :param timestamp: float - Unix timestamp at construction (defaults to
        ``time.time()``). Recorded so consumers can compute per-stage
        durations without a separate clock source.
    """
    stage: str
    detail: str
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not isinstance(self.stage, str) or not self.stage:
            raise ValidationError("ReasoningStep.stage must be a non-empty string")
        # The truncation sentinel is permitted alongside the closed enum so
        # the trace's own roll-off marker survives validation when it is
        # round-tripped through the dataclass constructor.
        if self.stage != REASONING_TRACE_TRUNCATED_STAGE and self.stage not in REASONING_STAGES:
            raise ValidationError(f"ReasoningStep.stage must be one of {sorted(REASONING_STAGES)} or the sentinel '{REASONING_TRACE_TRUNCATED_STAGE}', got {self.stage!r}")
        if not isinstance(self.detail, str):
            raise ValidationError("ReasoningStep.detail must be a string")
        if not isinstance(self.timestamp, (int, float)) or isinstance(self.timestamp, bool):
            raise ValidationError("ReasoningStep.timestamp must be a number")
        if float(self.timestamp) < 0.0:
            raise ValidationError("ReasoningStep.timestamp must be >= 0")


# Hard cap on a single ``ReasoningStep.detail`` length. Producers that emit
# longer strings get the tail truncated with an ellipsis marker. The cap is
# generous enough for ``key1=v1 key2=v2 ...`` lines but small enough to keep
# a 64-step trace under ~16 KiB even in the worst case.
_REASONING_DETAIL_MAX_CHARS = 256

# Default upper bound on the number of steps held by a single ``ReasoningTrace``.
# A pipeline with all phases enabled emits at most ~10 steps per search; a
# default of 64 leaves room for sub-intent fan-out and per-item ranking trails
# without unbounded growth. Configurable via ``ReasoningTraceConfig.max_steps``.
_REASONING_DEFAULT_MAX_STEPS = 64


@dataclass
class ReasoningTrace:
    """Append-only, bounded reasoning trail for explainability.

    Attached to ``QueryIntent`` (classification trace) and ``RankedItem``
    (per-item ranking trace). The trace is append-only: producers call
    :meth:`add` to record one step at a time; consumers read ``steps``
    as an ordered list. A ``max_steps`` ceiling enforces the bound so a
    runaway loop cannot grow the trace past the configured ceiling — when
    the cap is hit, the oldest step is dropped and a single sentinel step
    of stage ``REASONING_TRACE_TRUNCATED_STAGE`` is appended (or its detail
    is updated) so consumers always see "the trace was clipped" rather
    than silently losing entries.

    Two failure modes a trace deliberately swallows (it is a diagnostic,
    not a control surface):
      1. ``max_steps == 0`` — the trace is permanently empty and ``add`` is
         a no-op. Used to honour the global "trace disabled" config knob
         without forcing every producer site to gate on ``trace is None``.
      2. Detail oversize — the detail is truncated with an ellipsis marker
         instead of raising; producers should never have a debug log derail
         a search.

    :param max_steps: int - Hard ceiling on retained steps (>= 0). Default
        :data:`_REASONING_DEFAULT_MAX_STEPS`. Set to 0 for a permanently
        empty trace.
    :param steps: List[ReasoningStep] - Ordered list of recorded steps.
        Newest at the tail. When the cap is hit, the head is dropped first
        and the truncation sentinel is reconciled at the head.
    """
    max_steps: int = _REASONING_DEFAULT_MAX_STEPS
    steps: List[ReasoningStep] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.max_steps, int) or isinstance(self.max_steps, bool):
            raise ValidationError("ReasoningTrace.max_steps must be an int")
        if self.max_steps < 0:
            raise ValidationError("ReasoningTrace.max_steps must be >= 0")
        if not isinstance(self.steps, list):
            raise ValidationError("ReasoningTrace.steps must be a list")
        for s in self.steps:
            if not isinstance(s, ReasoningStep):
                raise ValidationError("ReasoningTrace.steps entries must be ReasoningStep instances")
        # Trim any oversize seed list down to the configured ceiling so a
        # caller cannot smuggle in more than ``max_steps`` total entries via
        # direct construction. The sentinel counts toward the cap (parity
        # with the runtime drop path in ``add``); when the cap is 0 the
        # trace is forced empty regardless of seed.
        if self.max_steps == 0:
            self.steps = []
        elif len(self.steps) > self.max_steps:
            # Reserve one slot for the sentinel; keep the trailing
            # ``max_steps - 1`` real steps as the most recent window.
            keep = self.max_steps - 1
            kept = self.steps[-keep:] if keep > 0 else []
            dropped = len(self.steps) - keep
            sentinel = ReasoningStep(stage=REASONING_TRACE_TRUNCATED_STAGE, detail=f"dropped={dropped}")
            self.steps = [sentinel] + kept

    def add(self, stage: str, detail: str) -> None:
        """Append one step. No-op when ``max_steps == 0``.

        :param stage: str - One of ``REASONING_STAGES`` (the sentinel stage
            is reserved for the trace itself).
        :param detail: str - Stage-specific detail; truncated to
            :data:`_REASONING_DETAIL_MAX_CHARS` if longer.
        """
        if self.max_steps == 0:
            return
        if stage == REASONING_TRACE_TRUNCATED_STAGE:
            raise ValidationError(f"ReasoningTrace.add: stage '{REASONING_TRACE_TRUNCATED_STAGE}' is reserved for the trace truncation guard")
        if not isinstance(detail, str):
            raise ValidationError("ReasoningTrace.add: detail must be a string")
        clipped = detail
        if len(clipped) > _REASONING_DETAIL_MAX_CHARS:
            # Keep the leading prefix that callers usually structure as
            # ``key=value`` and append an explicit ellipsis marker so the
            # truncation is visible to a debugger.
            clipped = clipped[: _REASONING_DETAIL_MAX_CHARS - 3] + '...'
        new_step = ReasoningStep(stage=stage, detail=clipped)
        # Bound: total step count (sentinel included) NEVER exceeds max_steps.
        # On overflow we (a) drop the oldest REAL step and (b) reconcile a
        # single head sentinel whose ``dropped=N`` counter reflects the
        # cumulative count of real steps that have been rolled off.
        # Pre-append, ensure post-append length will be exactly max_steps.
        # Append the new step first into a temporary working list, then
        # repeatedly drop the oldest real step (skipping the sentinel) and
        # bump the sentinel's counter until the list fits the cap.
        working = list(self.steps)
        working.append(new_step)
        while len(working) > self.max_steps:
            # Locate the oldest real step. The sentinel — when present — is
            # always at index 0, so the oldest real step is at index 1
            # (or 0 if no sentinel exists).
            has_sentinel = bool(working) and working[0].stage == REASONING_TRACE_TRUNCATED_STAGE
            real_idx = 1 if has_sentinel else 0
            if real_idx >= len(working):
                # Defensive: only the sentinel left and we still overflow,
                # which can only happen with max_steps == 0 — already
                # short-circuited above. Break to avoid infinite loop.
                break
            del working[real_idx]
            dropped_before = self._dropped_count_of(working)
            # Reconcile the sentinel: increment if present, prepend if not.
            if has_sentinel:
                working[0] = ReasoningStep(stage=REASONING_TRACE_TRUNCATED_STAGE, detail=f"dropped={dropped_before + 1}")
            else:
                working.insert(0, ReasoningStep(stage=REASONING_TRACE_TRUNCATED_STAGE, detail="dropped=1"))
        self.steps = working

    @staticmethod
    def _dropped_count_of(steps: List['ReasoningStep']) -> int:
        """Parse ``dropped=N`` from the head sentinel of an arbitrary list."""
        if not steps:
            return 0
        head = steps[0]
        if head.stage != REASONING_TRACE_TRUNCATED_STAGE:
            return 0
        prefix = 'dropped='
        if not head.detail.startswith(prefix):
            return 0
        try:
            return int(head.detail[len(prefix):])
        except ValueError:
            return 0

    def is_empty(self) -> bool:
        """True iff no real steps were ever recorded."""
        return not self.steps

    def __len__(self) -> int:
        return len(self.steps)


ROUTING_MODES = frozenset({'auto_execute', 'suggest', 'explore'})

# Classification of the entity-stripped semantic residual computed by the QI
# engine. Drives modality dispatch (skip vector when filter-dominant) and
# weighted-RRF profile selection. ``empty`` = nothing left after stripping;
# ``navigational`` = only navigational/corpus-noise tokens remain (e.g.
# "find domain"); ``semantic`` = ≥ ``min_content_tokens_for_semantic`` content
# tokens remain. ``None`` on multi-intent fan-out, cache-hit paths, and any
# call site that pre-dates the residual classifier or has the feature flag off.
RESIDUAL_KINDS = frozenset({'empty', 'navigational', 'semantic'})

# Kinds of filter conflict the detector can surface. ``range_inverted`` = a
# min/max pair where min > max (e.g. price_min > price_max). ``mutually_exclusive``
# = two constraints that cannot both hold (e.g. name_length_max < name_length_min,
# or a one-word constraint with an impossible character bound).
CONFLICT_KINDS = frozenset({'range_inverted', 'mutually_exclusive', 'qualitative_quantitative'})


@dataclass(frozen=True)
class FilterConflict:
    """One detected contradiction between extracted filter slots.

    :param kind: str - One of CONFLICT_KINDS
    :param slots: List[str] - The filter slot names that conflict
    :param message: str - Human-readable explanation for the UI / response
    """
    kind: str
    slots: List[str]
    message: str

    def __post_init__(self) -> None:
        if self.kind not in CONFLICT_KINDS:
            raise ValidationError(f"FilterConflict.kind must be one of {sorted(CONFLICT_KINDS)}")
        if not isinstance(self.slots, list) or len(self.slots) == 0 or not all(isinstance(s, str) and s for s in self.slots):
            raise ValidationError("FilterConflict.slots must be a non-empty list of non-empty strings")
        if not isinstance(self.message, str) or not self.message:
            raise ValidationError("FilterConflict.message must be a non-empty string")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FilterConflict':
        """Build a FilterConflict from a mapping, rejecting malformed input.
        :param d: Dict[str, Any] - Mapping with 'kind', 'slots', 'message'
        :return: FilterConflict - Validated instance
        """
        if not isinstance(d, dict):
            raise ValidationError("FilterConflict.from_dict requires a mapping")
        for key in ('kind', 'slots', 'message'):
            if key not in d:
                raise ValidationError(f"FilterConflict.from_dict missing required key '{key}'")
        return cls(kind=str(d['kind']), slots=[str(s) for s in d['slots']], message=str(d['message']))


@dataclass
class QueryIntent:
    """The full classification result for a single query.
    :param request_id: str - Per-call trace id (one new id per /search HTTP hop)
    :param search_id: str - Durable search-interaction id for feedback / analysis joins
        (distinct from ``request_id``; minted or client-supplied via identity config)
    :param raw_query: str - Original query text
    :param normalized_query: str - Lowercased / whitespace-collapsed query used for hashing
    :param query_type: str - Primary query type (one of QUERY_TYPES)
    :param confidence: float - Confidence of the primary classification
    :param decision_tier: str - Tier that "owned" the decision
    :param slices: List[IntentSlice] - Multi-intent decomposition (>=1 slice)
    :param decision_cost_usd: float - Cumulative LLM cost for this request's QI path
        (L0 extract + optional L2 classify; orchestrator may stamp the full
        request budget including rewrite / NL-SQL)
    :param intent_record_id: str - Stable IntentRecord id
        that can outlive a single ``request_id``. Resume + refine + measurement
        correlation all join on this field. When the caller does not supply one,
        a fresh id is minted (lifetime == request lifetime). When the caller
        supplies one (resume of a saved search, refine of a prior intent), the
        same id is preserved across every downstream artifact (history,
        measurement, feedback, batch exports).
    :param alternative_interpretations: List[IntentSlice] - Tier-3 LLM-emitted alternative readings of a single, low-confidence query.
        Each alternative is a distinct interpretation (different chip set, different
        query_type, or different entity-value combination) the gate can render as a
        Pattern-A MCQ card without depending on multi-intent decomposition. Empty
        for high-confidence single-intent queries and for L0/L1-resolved queries
        (the LLM is the only source of alternatives — see ``LLMClassifier``).
    :param did_you_mean: Optional[SpellCorrection] - Tier-0 spell-correction
        outcome. ``None`` when the corrector is disabled, the query had no
        corrections, or correction soft-failed. When present, the instance
        always carries at least one ``TokenCorrection`` and the ``applied``
        flag distinguishes auto-apply from suggest-only modes. Auto-apply
        means downstream QI / cache / retrieval all consumed the corrected
        query; the original is preserved here for the UI strip.
    :param reasoning_trace: Optional[ReasoningTrace] - Bounded, append-only
        audit trail of the pipeline stages that ran on this query
        (spell-correct, cache lookup, QI classification, retrieve, fuse,
        etc.). ``None`` when the global ``reasoning_trace`` config knob is
        disabled (the producers never construct a trace at all, zero
        overhead). Never PII-bearing — each step carries only a structured
        ``key=value`` detail string capped to 256 chars; raw query text
        MUST NOT be written into the trace.
    :param routing_mode: str - UX routing band derived from calibrated ``confidence``
        and config thresholds: ``auto_execute`` (high confidence), ``suggest``,
        or ``explore`` (low confidence). One of ``ROUTING_MODES``.
    :param created_at: float - Unix timestamp at construction
    :param prompt_tag: Optional[str] - Classify prompt version stamped from ``qi.llm.prompt_tag``
    :param schema_version: Optional[str] - Classify schema version stamped from ``qi.llm.schema_version``
    """
    request_id: str
    raw_query: str
    normalized_query: str
    query_type: str
    confidence: float
    decision_tier: str
    slices: List[IntentSlice]
    decision_cost_usd: float
    intent_record_id: str = ''
    search_id: str = ''
    prompt_tag: Optional[str] = None
    schema_version: Optional[str] = None
    # Entity-stripped semantic residual computed by the QI engine.
    # The QI engine strips price/TLD structural tokens before L1 encodes; this
    # field carries that residual so the vector retriever can encode it instead
    # of the raw normalized_query — removing structural token bias from cosine
    # similarity and focusing vector search on domain-name affinity.
    # None when the engine didn't compute a residual (multi-intent fan-out,
    # cache-hit paths, or legacy callers that pre-date this field).
    semantic_query: Optional[str] = None
    # Text that is SAFE TO EMBED — the single string every semantic + lexical
    # retrieval leg (dense vector, BM25 sparse, char-ngram, rerank) encodes.
    # Carries the residual concept when one was computed, otherwise the
    # normalized query with the TLD literals removed. ALWAYS TLD-free: the
    # ``tld`` / ``tldExcludeList`` tokens never appear here, so TLD is matched
    # only as an exact structured filter and never pollutes cosine / lexical
    # similarity. Built by the QI engine; ``None`` on cache-hit / multi-intent
    # wrapper / legacy construction sites, where retrievers fall back to the
    # ``semantic_encode_text_for`` helper which re-derives a TLD-stripped text.
    semantic_encode_text: Optional[str] = None
    # Classification of the post-stripping semantic residual; one of
    # ``RESIDUAL_KINDS`` or ``None``. Set by the QI engine when
    # ``qi.residual.enabled``; ``None`` on multi-intent fan-out, cache-hit
    # paths, and legacy callers. Drives Step 3 modality dispatch and Step 4
    # weighted-RRF profile selection.
    residual_kind: Optional[str] = None
    alternative_interpretations: List[IntentSlice] = field(default_factory=list)
    did_you_mean: Optional['SpellCorrection'] = None
    reasoning_trace: Optional[ReasoningTrace] = None
    routing_mode: str = 'explore'
    created_at: float = field(default_factory=time.time)
    # Pre-QI query-transform outcome (QueryTransformer rewrite step that
    # ran before classification). Carries whether the rewriter fired and the query
    # text it produced so HTTP surfaces can echo it without re-running the model.
    # Keys: mode ('llm_rewrite'|'local_fallback'|'passthrough'), engine (str: the
    # LLM model id, the local model path, or '' for passthrough), transformed (bool),
    # transformed_query (str). The original text is omitted — it is already the
    # top-level ``query`` on the response. ``None`` when the transformer is
    # unwired, soft-failed, or this intent came off a cache hit.
    # Stored as a plain dict (not the QueryTransformResult dataclass) to keep the
    # torch-backed query_transformer module out of the contracts import graph.
    query_transform: Optional[Dict[str, Any]] = None
    # Contradictions detected between extracted filter slots (e.g. price_min >
    # price_max, or an impossible character-length range). Empty when the filter
    # set is internally consistent. When non-empty the orchestrator short-circuits
    # to a 'filter_conflict' RankedResults carrying these explainers.
    conflicts: List['FilterConflict'] = field(default_factory=list)
    # Per-sub-intent entity snapshots captured before singleton deconfliction.
    # Populated only when multi_intent.preserve_sub_intent_filters is true in config
    # and the query was split into >1 sub-intents. Each entry holds the sub-query
    # text and the entities the L0 extractor found in that clause before cross-intent
    # merge logic ran (e.g. price_max=10 for ".net under $10", price_max=20 for ".io
    # under $20"). None for single-intent queries and cache-hit paths.
    sub_intent_filters: Optional[List['SubIntentFilterSet']] = None
    # Aggregated ``{"term", "probability"}`` pairs across all slices' L0 keyword
    # extraction, deduped by term (max probability kept), sorted descending.
    # Independent of entities/soft_entities — never merged into
    # identified_filters or the soft-signals response list.
    keywords: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValidationError("QueryIntent.request_id must be non-empty")
        if not isinstance(self.raw_query, str) or len(self.raw_query) == 0:
            raise ValidationError("QueryIntent.raw_query must be a non-empty string")
        if self.query_type not in QUERY_TYPES:
            raise ValidationError(f"QueryIntent.query_type must be one of {sorted(QUERY_TYPES)}")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValidationError("QueryIntent.confidence must be in [0,1]")
        if self.decision_tier not in DECISION_TIERS:
            raise ValidationError(f"QueryIntent.decision_tier must be one of {sorted(DECISION_TIERS)}")
        if not isinstance(self.slices, list) or len(self.slices) == 0:
            raise ValidationError("QueryIntent.slices must contain at least one slice")
        if float(self.decision_cost_usd) < 0.0:
            raise ValidationError("QueryIntent.decision_cost_usd must be >= 0")
        if not isinstance(self.intent_record_id, str):
            raise ValidationError("QueryIntent.intent_record_id must be a string")
        if not self.intent_record_id:
            # Default to a freshly-minted id so single-shot callers don't need to
            # construct one themselves. Callers that resume / refine pass an
            # existing id explicitly and that value is preserved.
            self.intent_record_id = QueryIntent.new_intent_record_id()
        if not isinstance(self.search_id, str):
            raise ValidationError("QueryIntent.search_id must be a string")
        if self.semantic_query is not None and not isinstance(self.semantic_query, str):
            raise ValidationError("QueryIntent.semantic_query must be a string or None")
        if self.semantic_encode_text is not None and not isinstance(self.semantic_encode_text, str):
            raise ValidationError("QueryIntent.semantic_encode_text must be a string or None")
        if self.residual_kind is not None and self.residual_kind not in RESIDUAL_KINDS:
            raise ValidationError(f"QueryIntent.residual_kind must be one of {sorted(RESIDUAL_KINDS)} or None")
        if not isinstance(self.alternative_interpretations, list):
            raise ValidationError("QueryIntent.alternative_interpretations must be a list")
        for alt in self.alternative_interpretations:
            if not isinstance(alt, IntentSlice):
                raise ValidationError("QueryIntent.alternative_interpretations entries must be IntentSlice instances")
        if self.did_you_mean is not None and not isinstance(self.did_you_mean, SpellCorrection):
            raise ValidationError("QueryIntent.did_you_mean must be a SpellCorrection instance or None")
        if self.reasoning_trace is not None and not isinstance(self.reasoning_trace, ReasoningTrace):
            raise ValidationError("QueryIntent.reasoning_trace must be a ReasoningTrace instance or None")
        if self.routing_mode not in ROUTING_MODES:
            raise ValidationError(f"QueryIntent.routing_mode must be one of {sorted(ROUTING_MODES)}")
        if not isinstance(self.conflicts, list):
            raise ValidationError("QueryIntent.conflicts must be a list")
        for c in self.conflicts:
            if not isinstance(c, FilterConflict):
                raise ValidationError("QueryIntent.conflicts entries must be FilterConflict instances")
        if self.query_transform is not None and not isinstance(self.query_transform, dict):
            raise ValidationError("QueryIntent.query_transform must be a dict or None")
        if not isinstance(self.keywords, list):
            raise ValidationError("QueryIntent.keywords must be a list")

    @staticmethod
    def derive_routing_mode(confidence: float, auto_execute_min: float, suggest_min: float) -> str:
        """Map calibrated confidence to plan UX bands (config-driven thresholds).
        :param confidence: float - Primary slice calibrated confidence in [0,1]
        :param auto_execute_min: float - Minimum confidence for ``auto_execute``
        :param suggest_min: float - Minimum confidence for ``suggest`` (exclusive upper band for explore below this)
        :return: str - One of ``ROUTING_MODES``
        """
        c = float(confidence)
        if c >= float(auto_execute_min):
            return 'auto_execute'
        if c >= float(suggest_min):
            return 'suggest'
        return 'explore'

    @staticmethod
    def new_request_id() -> str:
        """Generate a request_id for a new query."""
        return _new_id('req')

    @staticmethod
    def new_intent_record_id() -> str:
        """Generate a stable intent_record_id.

        Distinct prefix from ``request_id`` so dashboards / logs cannot confuse
        the two. Carried end-to-end through history, measurement, and feedback.
        """
        return _new_id('ir')


@dataclass
class Candidate:
    """A single retrieved candidate before fusion.
    :param item_id: str - Stable item identifier (e.g. domain id)
    :param score: float - Backend-native score in [0,1]
    :param source: str - Backend that produced this candidate (vector/structured/sql)
    :param payload: Dict[str, Any] - Backend-supplied attributes
    """
    item_id: str
    score: float
    source: str
    payload: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.item_id:
            raise ValidationError("Candidate.item_id must be non-empty")
        if self.source not in CANDIDATE_SOURCES:
            raise ValidationError(f"Candidate.source must be one of {sorted(CANDIDATE_SOURCES)}")
        if not 0.0 <= float(self.score) <= 1.0:
            raise ValidationError("Candidate.score must be in [0,1]")
        if not isinstance(self.payload, dict):
            raise ValidationError("Candidate.payload must be a dict")


@dataclass
class CandidateSet:
    """Unfused candidates from a single retrieval backend.
    :param source: str - Backend that produced these candidates
    :param candidates: List[Candidate] - Candidates in score-descending order
    :param latency_ms: float - Wall-clock time to produce the candidates
    """
    source: str
    candidates: List[Candidate]
    latency_ms: float

    def __post_init__(self) -> None:
        if self.source not in CANDIDATE_SOURCES:
            raise ValidationError(f"CandidateSet.source must be one of {sorted(CANDIDATE_SOURCES)}")
        if not isinstance(self.candidates, list):
            raise ValidationError("CandidateSet.candidates must be a list")
        if float(self.latency_ms) < 0.0:
            raise ValidationError("CandidateSet.latency_ms must be >= 0")
        for cand in self.candidates:
            if cand.source != self.source:
                raise ValidationError(f"CandidateSet.candidates[*].source must equal CandidateSet.source ({self.source})")


# Closed enum of `RankedItem.sub_intent_match_kinds` values. Sub-intent badges
# differentiate strict vs. loose matches so the UI renders them differently
# (filled chip = hard, outlined chip = soft) and the downstream ranker can
# weight hard matches higher. ``hard``: item satisfies the sub-intent's grounded
# filters (TLD/price/auction-type). ``soft``: item appears in the sub-intent's
# semantic neighborhood but does not strictly satisfy every filter.
SUB_INTENT_MATCH_KINDS = frozenset({'hard', 'soft'})


@dataclass
class RankedItem:
    """A single fused, ranked item.
    :param item_id: str - Stable item identifier
    :param fused_score: float - RRF-fused score (>= 0)
    :param contributing_sources: List[str] - Sources that surfaced this item
    :param payload: Dict[str, Any] - Merged backend payloads
    :param sub_intent_ids: List[str] - Sub-intent slice ids that
        matched this item. Empty for single-intent searches; multi-element when the
        item appears in more than one sub-intent's candidate set (powers the
        sub-intent badges rendered on each card).
    :param sub_intent_match_kinds: Dict[str, str] - Per-slice match strength
        keyed by ``slice_id``. Value is one of ``SUB_INTENT_MATCH_KINDS``
        (``'hard'`` = item satisfied the slice's grounded filters
        TLD/price/auction-type; ``'soft'`` = item only matched the slice's
        broader semantic neighborhood). The UI renders hard badges filled
        and soft badges outlined for multi-intent composition. Default empty dict for single-intent searches; every
        key MUST appear in ``sub_intent_ids`` when set.
    :param reasoning_trace: Optional[ReasoningTrace] - Bounded, per-item
        ranking trail (eRanker outcome, diversifier decision, etc.). ``None`` when the global ``reasoning_trace`` config
        knob is disabled. Same PII / structured-detail rules as
        ``QueryIntent.reasoning_trace``.
    """
    item_id: str
    fused_score: float
    contributing_sources: List[str]
    payload: Dict[str, Any] = field(default_factory=dict)
    sub_intent_ids: List[str] = field(default_factory=list)
    sub_intent_match_kinds: Dict[str, str] = field(default_factory=dict)
    reasoning_trace: Optional[ReasoningTrace] = None

    def __post_init__(self) -> None:
        if not self.item_id:
            raise ValidationError("RankedItem.item_id must be non-empty")
        if float(self.fused_score) < 0.0:
            raise ValidationError("RankedItem.fused_score must be >= 0")
        if not isinstance(self.contributing_sources, list) or len(self.contributing_sources) == 0:
            raise ValidationError("RankedItem.contributing_sources must be non-empty")
        for src in self.contributing_sources:
            if src not in CANDIDATE_SOURCES:
                raise ValidationError(f"RankedItem.contributing_sources contains invalid source '{src}'")
        if not isinstance(self.sub_intent_ids, list):
            raise ValidationError("RankedItem.sub_intent_ids must be a list")
        # Each id must be a non-empty string when present (no duplicate validation —
        # a sub-intent slice can match an item via multiple sources).
        for sid in self.sub_intent_ids:
            if not sid or not isinstance(sid, str):
                raise ValidationError("RankedItem.sub_intent_ids entries must be non-empty strings")
        if not isinstance(self.sub_intent_match_kinds, dict):
            raise ValidationError("RankedItem.sub_intent_match_kinds must be a dict[str, str]")
        sub_intent_id_set = set(self.sub_intent_ids)
        for k, v in self.sub_intent_match_kinds.items():
            if not isinstance(k, str) or not k:
                raise ValidationError("RankedItem.sub_intent_match_kinds keys must be non-empty strings")
            if v not in SUB_INTENT_MATCH_KINDS:
                raise ValidationError(f"RankedItem.sub_intent_match_kinds values must be one of {sorted(SUB_INTENT_MATCH_KINDS)}")
            if k not in sub_intent_id_set:
                raise ValidationError(f"RankedItem.sub_intent_match_kinds key '{k}' must appear in sub_intent_ids")
        if self.reasoning_trace is not None and not isinstance(self.reasoning_trace, ReasoningTrace):
            raise ValidationError("RankedItem.reasoning_trace must be a ReasoningTrace instance or None")


@dataclass
class IntentChipGroup:
    """One per-intent chip group in the multi-intent chip strip.

    Renders as a single row in the strip envelope:
      ``Intent N - <title>: [chip1] [chip2] [chip3]    <count> (ok)``

    :param slice_id: str - Stable id of the surviving sub-intent slice;
        joins to ``RankedItem.sub_intent_ids`` so the UI can highlight cards on chip click
    :param title: str - Plain-English label for the row (e.g. ``"Tech .com < $500"``)
    :param chips: List[str] - Locked filter chips in display order (e.g. ``[".com", "tech", "<$500"]``)
    :param count: int - Per-intent live result count (>= 0); the trailing badge after the chips
    :param raw_text: str - Original sub-query text the chips were derived from (audit / refine)
    """
    slice_id: str
    title: str
    chips: List[str]
    count: int
    raw_text: str

    def __post_init__(self) -> None:
        if not self.slice_id or not isinstance(self.slice_id, str):
            raise ValidationError("IntentChipGroup.slice_id must be a non-empty string")
        if not isinstance(self.title, str) or not self.title:
            raise ValidationError("IntentChipGroup.title must be a non-empty string")
        if not isinstance(self.chips, list):
            raise ValidationError("IntentChipGroup.chips must be a list")
        for ch in self.chips:
            if not isinstance(ch, str) or not ch:
                raise ValidationError("IntentChipGroup.chips entries must be non-empty strings")
        if int(self.count) < 0:
            raise ValidationError("IntentChipGroup.count must be >= 0")
        if not isinstance(self.raw_text, str):
            raise ValidationError("IntentChipGroup.raw_text must be a string")


@dataclass
class OverflowChip:
    """The collapsed overflow chip surfaced when the splitter found more sub-intents than the 5-cap kept.

    Rendered as ``[+ N more concepts — refine to see them]``. Clicking reopens
    the conversational input box pre-filled with the dropped sub-queries.

    :param label: str - The chip label shown to the user (e.g. ``"+ 2 more concepts — refine to see them"``)
    :param dropped_sub_queries: List[str] - Sub-queries the 5-cap dropped (display order = splitter order)
    :param count: int - Length of ``dropped_sub_queries`` (>= 1; an overflow chip only exists when something was dropped)
    """
    label: str
    dropped_sub_queries: List[str]
    count: int

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise ValidationError("OverflowChip.label must be a non-empty string")
        if not isinstance(self.dropped_sub_queries, list) or len(self.dropped_sub_queries) == 0:
            raise ValidationError("OverflowChip.dropped_sub_queries must be a non-empty list")
        for sq in self.dropped_sub_queries:
            if not isinstance(sq, str) or not sq:
                raise ValidationError("OverflowChip.dropped_sub_queries entries must be non-empty strings")
        if int(self.count) != len(self.dropped_sub_queries):
            raise ValidationError("OverflowChip.count must equal len(dropped_sub_queries)")


@dataclass
class MultiIntentChipStrip:
    """The server-side chip strip envelope rendered above the merged ranked list.

    Built once per multi-intent search by ``_retrieve_and_rank_multi`` and stamped
    on ``RankedResults.multi_intent_envelope``. The orchestrator emits this only
    when the splitter emitted >1 sub-queries; single-intent searches leave it None.

    :param per_intent_chips: List[IntentChipGroup] - One row per surviving sub-intent in 5-cap rank order
    :param overflow_chip: Optional[OverflowChip] - Set iff the 5-cap dropped >=1 sub-queries; None otherwise
    :param total_kept: int - len(per_intent_chips) — convenience for the UI
    :param total_dropped: int - Number of sub-queries dropped by the 5-cap (matches ``overflow_chip.count`` when set)
    """
    per_intent_chips: List['IntentChipGroup']
    overflow_chip: Optional['OverflowChip']
    total_kept: int
    total_dropped: int

    def __post_init__(self) -> None:
        if not isinstance(self.per_intent_chips, list):
            raise ValidationError("MultiIntentChipStrip.per_intent_chips must be a list")
        if int(self.total_kept) != len(self.per_intent_chips):
            raise ValidationError("MultiIntentChipStrip.total_kept must equal len(per_intent_chips)")
        if int(self.total_dropped) < 0:
            raise ValidationError("MultiIntentChipStrip.total_dropped must be >= 0")
        if self.overflow_chip is None and int(self.total_dropped) != 0:
            raise ValidationError("MultiIntentChipStrip.overflow_chip must be set when total_dropped > 0")
        if self.overflow_chip is not None:
            if int(self.total_dropped) != int(self.overflow_chip.count):
                raise ValidationError("MultiIntentChipStrip.total_dropped must equal overflow_chip.count when overflow_chip is set")
        slice_ids = [g.slice_id for g in self.per_intent_chips]
        if len(slice_ids) != len(set(slice_ids)):
            raise ValidationError("MultiIntentChipStrip.per_intent_chips slice_ids must be unique")


@dataclass
class GuidanceEnvelope:
    """Structured market snapshot attached to guidance-classified searches.
    :param headline: str - Short label for the envelope
    :param body: str - Bounded payload (typically JSON text from snapshot rows)
    :param substrate: str - Source tier label (for example ``clickhouse_hot``)
    :param as_of_hot: Optional[float] - Hot-tier freshness anchor when known
    """
    headline: str
    body: str
    substrate: str
    as_of_hot: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.headline, str) or not self.headline.strip():
            raise ValidationError("GuidanceEnvelope.headline must be a non-empty string")
        if not isinstance(self.body, str):
            raise ValidationError("GuidanceEnvelope.body must be a string")
        if not isinstance(self.substrate, str) or not self.substrate.strip():
            raise ValidationError("GuidanceEnvelope.substrate must be a non-empty string")
        if self.as_of_hot is not None:
            try:
                hot = float(self.as_of_hot)
            except (TypeError, ValueError) as e:
                raise ValidationError("GuidanceEnvelope.as_of_hot must be numeric when set") from e
            if hot <= 0.0:
                raise ValidationError("GuidanceEnvelope.as_of_hot must be > 0 when set")
            object.__setattr__(self, 'as_of_hot', hot)


@dataclass
class RankedResults:
    """The final ranked list returned by the search service.
    :param request_id: str - Per-call trace id
    :param search_id: str - Durable search-interaction id (empty when not stamped)
    :param items: List[RankedItem] - Items ordered by fused_score descending
    :param total_candidates: int - Total candidate count before fusion
    :param fusion_latency_ms: float - Time spent in fusion
    :param cache_hit: Optional[str] - 'exact' / 'semantic' / 'structured' / 'intent_plan' / None
    :param multi_intent_envelope: Optional[MultiIntentChipStrip] - Per-intent
        chip strip (with optional overflow chip) for multi-intent searches. None for single-intent
        searches and for cache-hit responses (the cache is user-agnostic and per-intent
        counts would be stale).
    :param failure_mode: Optional[str] - When set, the response used a
        constrained fallback path instead of the normal retrieval pipeline.
        ``'find_fallback'`` — every retrieval backend was unhealthy and
        the conversational box silently routed to FIND-only filter
        extraction (regex chips returned as a structured payload; items
        list is empty); the UI should render the chip strip as a
        FIND-equivalent advanced filter form so the user can complete the
        search via the existing FIND surface.
        ``'inventory_empty'`` — the zero-result guard exhausted the full
        ladder (relax -> semantic-only -> explore_fallback) and even the
        explore rails returned nothing because the corpus and ClickHouse
        rails are both unpopulated; items list is empty and the UI should
        prompt the user to widen or change their query.
        ``None`` (default) — normal retrieval response.
    :param query_intent: Optional[QueryIntent] - QI verdict for this response when
        the pipeline classified the query on this request (omitted on legacy bare-cache
        hits and budget-breach empty envelopes). Lets HTTP surfaces serialize intent
        without re-running ``QIEngine.classify``.
    :param guidance_envelope: Optional[GuidanceEnvelope] - Market snapshot when the
        query_type is ``guidance`` and the guidance service produced a payload
    """
    _VALID_FAILURE_MODES = frozenset({
        'find_fallback', 'inventory_empty', 'inventory_empty_under_filters', 'timeout',
        'filter_conflict', 'analytics_timeout', 'analytics_failed', 'analytics_failure_fallback',
        'analytics_exception', 'explore_fallback_rail', 'explore_fallback_timeout',
        'hybrid_explore_complement', 'force_semantic_nonempty',
        'qdrant_unavailable', 'clickhouse_unavailable',
    })

    request_id: str
    items: List[RankedItem]
    total_candidates: int
    fusion_latency_ms: float
    cache_hit: Optional[str] = None
    multi_intent_envelope: Optional['MultiIntentChipStrip'] = None
    failure_mode: Optional[str] = None
    query_intent: Optional['QueryIntent'] = None
    guidance_envelope: Optional[GuidanceEnvelope] = None
    search_id: str = ''

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValidationError("RankedResults.request_id must be non-empty")
        if not isinstance(self.search_id, str):
            raise ValidationError("RankedResults.search_id must be a string")
        if not isinstance(self.items, list):
            raise ValidationError("RankedResults.items must be a list")
        if int(self.total_candidates) < 0:
            raise ValidationError("RankedResults.total_candidates must be >= 0")
        if float(self.fusion_latency_ms) < 0.0:
            raise ValidationError("RankedResults.fusion_latency_ms must be >= 0")
        if self.cache_hit is not None and self.cache_hit not in CACHE_HIT_TIER_VALUES:
            raise ValidationError(f"RankedResults.cache_hit must be one of {sorted(CACHE_HIT_TIER_VALUES)} or None")
        if self.multi_intent_envelope is not None and not isinstance(self.multi_intent_envelope, MultiIntentChipStrip):
            raise ValidationError("RankedResults.multi_intent_envelope must be a MultiIntentChipStrip when set")
        if self.failure_mode is not None and self.failure_mode not in self._VALID_FAILURE_MODES:
            raise ValidationError(f"RankedResults.failure_mode must be one of {sorted(self._VALID_FAILURE_MODES)} or None")
        if self.query_intent is not None and not isinstance(self.query_intent, QueryIntent):
            raise ValidationError("RankedResults.query_intent must be a QueryIntent instance or None")
        if self.guidance_envelope is not None and not isinstance(self.guidance_envelope, GuidanceEnvelope):
            raise ValidationError("RankedResults.guidance_envelope must be a GuidanceEnvelope instance or None")


@dataclass
class CachedSearchPayload:
    """Cache-line payload pairing a classified intent with its pre-eRanker
    ranked results.

    The exact and semantic search caches store these pairs (instead of bare
    ``RankedResults``) so that on a cache hit the orchestrator can call eRanker
    and post-eRanker diversification without re-running QI.
    The cache itself stays user-agnostic — the intent here is the QI verdict
    the original (cache-priming) request produced and is independent of the
    requesting user.

    Per-entry snapshot version tagging (Wave 9 Gap 13). When the inventory
    snapshot advances (``SnapshotVersionRegistry.bump``), legacy invalidation
    drops the entire cache tier. Per-entry tagging lets the orchestrator
    silently SKIP a stale cached entry on read instead of preemptively
    blowing every entry on write — old entries simply expire on access
    while still-fresh entries (those tagged with the current version) keep
    serving traffic. Default ``snapshot_version=0`` matches the registry's
    initial value so legacy callers (and pre-Wave-9 priming sites) do not
    silently mark every entry as stale; the orchestrator opts into
    versioned reads via ``CachedSearchPayload.is_fresh_for(current)``.

    :param intent: QueryIntent - The QI verdict that drove the cached results
    :param results: RankedResults - Post-fusion candidate list (pre-eRanker)
    :param snapshot_version: int - Inventory snapshot version at the time the
        entry was written (>= 0). Compared against
        ``SnapshotVersionRegistry.version`` on read; consumers MAY treat
        any cached entry whose ``snapshot_version`` is below the live
        version as a miss.
    """
    intent: 'QueryIntent'
    results: RankedResults
    snapshot_version: int = 0

    def __post_init__(self) -> None:
        if self.intent is None or not isinstance(self.intent, QueryIntent):
            raise ValidationError("CachedSearchPayload.intent must be a QueryIntent")
        if self.results is None or not isinstance(self.results, RankedResults):
            raise ValidationError("CachedSearchPayload.results must be a RankedResults")
        if not isinstance(self.snapshot_version, int) or self.snapshot_version < 0:
            raise ValidationError("CachedSearchPayload.snapshot_version must be an int >= 0")

    def is_fresh_for(self, current_version: int) -> bool:
        """Return True iff the entry's snapshot version is at or above the
        live registry version. Used by the cache tiers' versioned-read path
        to silently skip entries primed against an older snapshot.

        :param current_version: int - Live snapshot version from the registry
        :return: bool
        """
        if not isinstance(current_version, int) or current_version < 0:
            raise ValidationError("current_version must be an int >= 0")
        return self.snapshot_version >= current_version


def infer_breaker_transition_fallback_kind(breaker: str, from_state: str, to_state: str, reason: str) -> str:
    """Map a breaker transition to a stable ``fallback_kind`` for dashboards.
    :param breaker: str - ``BREAKER_TRANSITION_BREAKER_IDS`` member
    :param from_state: str - Prior breaker or backend-health state label
    :param to_state: str - New breaker or backend-health state label
    :param reason: str - Emitter reason (``backend_health_transition`` for registry health listener)
    :return: str - One of ``BREAKER_TRANSITION_FALLBACK_KINDS``
    """
    _ = breaker
    if str(reason) == 'backend_health_transition':
        if str(to_state) == 'unhealthy':
            return 'fallback_active'
        if str(to_state) == 'degraded':
            return 'degraded'
        return 'normal'
    if str(to_state) == 'open':
        return 'fallback_active'
    if str(to_state) == 'half_open':
        return 'probe'
    return 'normal'


def _validate_breaker_transition_payload(payload: Dict[str, Any]) -> None:
    """Validate ``FeedbackSignal`` payload when ``signal_type=='breaker_transition'``."""
    required = ('breaker', 'from_state', 'to_state', 'reason', 'observation_count', 'fallback_kind')
    for key in required:
        if key not in payload:
            raise ValidationError(f"FeedbackSignal.breaker_transition payload missing required key {key!r}")
    if str(payload['breaker']) not in BREAKER_TRANSITION_BREAKER_IDS:
        raise ValidationError(f"FeedbackSignal.breaker_transition breaker must be one of {sorted(BREAKER_TRANSITION_BREAKER_IDS)}")
    if str(payload['fallback_kind']) not in BREAKER_TRANSITION_FALLBACK_KINDS:
        raise ValidationError(f"FeedbackSignal.breaker_transition fallback_kind must be one of {sorted(BREAKER_TRANSITION_FALLBACK_KINDS)}")
    oc = payload['observation_count']
    if not isinstance(oc, int) or int(oc) < 0:
        raise ValidationError("FeedbackSignal.breaker_transition observation_count must be an int >= 0")


@dataclass
class FeedbackSignal:
    """A single user-feedback signal captured for measurement and analytics.
    :param signal_id: str - Stable id
    :param request_id: str - Trace id for the emitting HTTP hop (or originating search hop)
    :param search_id: str - Durable search-interaction id for feedback / analysis joins.
        Empty when the emitter has no search-interaction context.
    :param signal_type: str - One of SIGNAL_TYPES
    :param payload: Dict[str, Any] - Signal-specific metadata (e.g. overridden filter values)
    :param intent_record_id: str - IntentRecord id this signal
        correlates with for measurement joins. Empty when the
        emitting site has no IntentRecord context (legacy / pre-A16 tests).
    :param signal_origin: str - One of ``FEEDBACK_SIGNAL_ORIGINS``. Identifies which
        subsystem produced the signal for measurement slices. Defaults to
        ``'unknown'`` for legacy emitters; new emit sites SHOULD set this.
    :param created_at: float - Unix timestamp at construction
    """
    signal_id: str
    request_id: str
    signal_type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    intent_record_id: str = ''
    signal_origin: str = 'unknown'
    search_id: str = ''
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.signal_id:
            raise ValidationError("FeedbackSignal.signal_id must be non-empty")
        if not self.request_id:
            raise ValidationError("FeedbackSignal.request_id must be non-empty")
        if not isinstance(self.search_id, str):
            raise ValidationError("FeedbackSignal.search_id must be a string")
        if self.signal_type not in SIGNAL_TYPES:
            raise ValidationError(f"FeedbackSignal.signal_type must be one of {sorted(SIGNAL_TYPES)}")
        if not isinstance(self.payload, dict):
            raise ValidationError("FeedbackSignal.payload must be a dict")
        if not isinstance(self.intent_record_id, str):
            raise ValidationError("FeedbackSignal.intent_record_id must be a string")
        if not isinstance(self.signal_origin, str) or self.signal_origin not in FEEDBACK_SIGNAL_ORIGINS:
            raise ValidationError(f"FeedbackSignal.signal_origin must be one of {sorted(FEEDBACK_SIGNAL_ORIGINS)}")
        if self.signal_type == 'breaker_transition':
            _validate_breaker_transition_payload(self.payload)

    @staticmethod
    def new_signal_id() -> str:
        """Generate a signal_id for a new feedback record."""
        return _new_id('sig')


@dataclass
class UserContext:
    """Per-request user context attached to /search and downstream subsystems.
    :param user_id: Optional[str] - Stable user id; None for unauthenticated requests
    :param is_authenticated: bool - True iff the caller is authenticated
    :param session_id: str - Per-session id used for save-trigger heuristics + saved-search storage
    :param explicit_mode: Optional[str] - Caller-asserted UI mode override
        ("conversational" | "advanced"). Validated against ``USER_MODES``.
    """
    user_id: Optional[str]
    is_authenticated: bool
    session_id: str
    explicit_mode: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.is_authenticated, bool):
            raise ValidationError("UserContext.is_authenticated must be bool")
        if self.is_authenticated and not self.user_id:
            raise ValidationError("UserContext.user_id is required when is_authenticated=True")
        if self.user_id is not None and not isinstance(self.user_id, str):
            raise ValidationError("UserContext.user_id must be a string when set")
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValidationError("UserContext.session_id must be a non-empty string")
        if self.explicit_mode is not None:
            if not isinstance(self.explicit_mode, str):
                raise ValidationError("UserContext.explicit_mode must be a string when set")
            if self.explicit_mode not in USER_MODES:
                raise ValidationError(f"UserContext.explicit_mode must be one of {sorted(USER_MODES)}")


@dataclass
class ERankerOutcome:
    """Audit envelope for the external eRanker (Layer 4) call.
    :param applied: bool - True when eRanker returned a re-ordered list (not pass-through skip)
    :param client: str - Client backend name (e.g. noop_eranker)
    :param skipped_reason: Optional[str] - When ``applied`` is False, why (e.g. ``disabled``, ``timeout``, ``cache_hit``)
    :param latency_ms: Optional[float] - Wall time spent inside the eRanker client call (excluding outer wait_for slack)
    :param http_status: Optional[int] - Last HTTP status from the eRanker transport when ``backend=http``
    """
    applied: bool
    client: str
    skipped_reason: Optional[str] = None
    latency_ms: Optional[float] = None
    http_status: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.client, str) or not self.client:
            raise ValidationError("ERankerOutcome.client must be a non-empty string")
        if not isinstance(self.applied, bool):
            raise ValidationError("ERankerOutcome.applied must be bool")
        if self.applied and self.skipped_reason is not None:
            raise ValidationError("ERankerOutcome.skipped_reason must be None when applied=True")
        if not self.applied and (self.skipped_reason is None or not str(self.skipped_reason).strip()):
            raise ValidationError("ERankerOutcome.skipped_reason is required when applied=False")
        if self.latency_ms is not None and float(self.latency_ms) < 0.0:
            raise ValidationError("ERankerOutcome.latency_ms must be >= 0 when set")
        if self.http_status is not None:
            if not isinstance(self.http_status, int) or int(self.http_status) < 100 or int(self.http_status) > 599:
                raise ValidationError("ERankerOutcome.http_status must be an int in [100, 599] when set")


@dataclass
class HistoryEntry:
    """A single user search history entry.
    :param entry_id: str - Stable id
    :param user_id: str - Authenticated user id
    :param normalized_query: str - Query stem used for repeat-query detection
    :param query_type: str - QI-derived query_type
    :param top_item_ids: List[str] - First N item ids returned (debug + delta detection)
    :param intent_record_id: str - ``IntentRecord`` id this history entry
        corresponds to. Resume reloads use this id to re-execute the same
        ``IntentRecord`` against current inventory. Empty for legacy entries.
    :param created_at: float - Unix timestamp
    """
    entry_id: str
    user_id: str
    normalized_query: str
    query_type: str
    top_item_ids: List[str]
    intent_record_id: str = ''
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.entry_id:
            raise ValidationError("HistoryEntry.entry_id must be non-empty")
        if not self.user_id:
            raise ValidationError("HistoryEntry.user_id must be non-empty")
        if not isinstance(self.normalized_query, str) or not self.normalized_query:
            raise ValidationError("HistoryEntry.normalized_query must be a non-empty string")
        if self.query_type not in QUERY_TYPES:
            raise ValidationError(f"HistoryEntry.query_type must be one of {sorted(QUERY_TYPES)}")
        if not isinstance(self.top_item_ids, list):
            raise ValidationError("HistoryEntry.top_item_ids must be a list")
        if not isinstance(self.intent_record_id, str):
            raise ValidationError("HistoryEntry.intent_record_id must be a string")

    @staticmethod
    def new_entry_id() -> str:
        """Generate an entry_id for a new history record."""
        return _new_id('hist')



HISTORY_AWARE_RULE_NAMES = frozenset({'shown_no_click', 'considered_then_expiring'})


@dataclass
class HistoryAdjustment:
    """Signed score delta from a history-aware sub-rule.
    :param rule_name: str - One of HISTORY_AWARE_RULE_NAMES
    :param item_id: str - Candidate item id
    :param delta: float - Finite signed adjustment
    :param reason: str - Audit string (no PII)
    """
    rule_name: str
    item_id: str
    delta: float
    reason: str

    def __post_init__(self) -> None:
        if self.rule_name not in HISTORY_AWARE_RULE_NAMES:
            raise ValidationError(f"HistoryAdjustment.rule_name must be one of {sorted(HISTORY_AWARE_RULE_NAMES)}")
        if not isinstance(self.item_id, str) or not self.item_id.strip():
            raise ValidationError("HistoryAdjustment.item_id must be non-empty")
        d = float(self.delta)
        if not (d == d) or d == float('inf') or d == float('-inf'):
            raise ValidationError("HistoryAdjustment.delta must be finite")
        object.__setattr__(self, 'delta', d)
        if not isinstance(self.reason, str):
            raise ValidationError("HistoryAdjustment.reason must be a string")

    def __post_init__(self) -> None:
        if self.mode not in USER_MODES:
            raise ValidationError(f"ModeRecommendation.mode must be one of {sorted(USER_MODES)}")
        if not 0.0 <= float(self.power_user_score) <= 1.0:
            raise ValidationError("ModeRecommendation.power_user_score must be in [0,1]")
        if not isinstance(self.reason, str):
            raise ValidationError("ModeRecommendation.reason must be a string")


@dataclass
class GoldenCase:
    """A single manually-reviewed golden-dataset eval case.
    :param case_id: str - Stable id (sha-prefixed for traceability)
    :param input_query: str - Raw user query (sanitized before storage)
    :param expected_query_type: str - Expected primary query_type
    :param expected_entities: List[Dict[str, Any]] - Expected entity slots ({name, value})
    :param difficulty: str - One of GOLDEN_DIFFICULTIES
    :param edge_type: str - One of GOLDEN_EDGE_TYPES
    :param source_session_id_hash: str - Hashed source session id (audit trail)
    :param signal_origin: str - One of SIGNAL_ORIGINS
    :param manual_review_status: str - One of GOLDEN_REVIEW_STATUSES
    :param reviewer_id: Optional[str] - Reviewer id when status != pending
    :param review_timestamp: Optional[float] - Unix timestamp when reviewed
    :param integrity_hash: str - SHA of (input_query, expected_query_type, expected_entities)
    :param sanitizer_passed: bool - Sanitizer verdict on input_query
    :param created_at: float - Unix timestamp at construction
    """
    case_id: str
    input_query: str
    expected_query_type: str
    expected_entities: List[Dict[str, Any]]
    difficulty: str
    edge_type: str
    source_session_id_hash: str
    signal_origin: str
    manual_review_status: str
    reviewer_id: Optional[str]
    review_timestamp: Optional[float]
    integrity_hash: str
    sanitizer_passed: bool
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValidationError("GoldenCase.case_id must be non-empty")
        if not isinstance(self.input_query, str) or not self.input_query:
            raise ValidationError("GoldenCase.input_query must be a non-empty string")
        if self.expected_query_type not in QUERY_TYPES:
            raise ValidationError(f"GoldenCase.expected_query_type must be one of {sorted(QUERY_TYPES)}")
        if not isinstance(self.expected_entities, list):
            raise ValidationError("GoldenCase.expected_entities must be a list")
        for ent in self.expected_entities:
            if not isinstance(ent, dict) or 'name' not in ent or 'value' not in ent:
                raise ValidationError("GoldenCase.expected_entities entries must be dicts with name + value")
        if self.difficulty not in GOLDEN_DIFFICULTIES:
            raise ValidationError(f"GoldenCase.difficulty must be one of {sorted(GOLDEN_DIFFICULTIES)}")
        if self.edge_type not in GOLDEN_EDGE_TYPES:
            raise ValidationError(f"GoldenCase.edge_type must be one of {sorted(GOLDEN_EDGE_TYPES)}")
        if self.signal_origin not in SIGNAL_ORIGINS:
            raise ValidationError(f"GoldenCase.signal_origin must be one of {sorted(SIGNAL_ORIGINS)}")
        if self.manual_review_status not in GOLDEN_REVIEW_STATUSES:
            raise ValidationError(f"GoldenCase.manual_review_status must be one of {sorted(GOLDEN_REVIEW_STATUSES)}")
        if self.manual_review_status != 'pending' and not self.reviewer_id:
            raise ValidationError("GoldenCase.reviewer_id required when status != pending")
        if self.manual_review_status != 'pending' and self.review_timestamp is None:
            raise ValidationError("GoldenCase.review_timestamp required when status != pending")
        if not isinstance(self.integrity_hash, str) or not self.integrity_hash:
            raise ValidationError("GoldenCase.integrity_hash must be a non-empty string")
        if not isinstance(self.sanitizer_passed, bool):
            raise ValidationError("GoldenCase.sanitizer_passed must be bool")
        if not isinstance(self.source_session_id_hash, str):
            raise ValidationError("GoldenCase.source_session_id_hash must be a string")



@dataclass
class RelevanceJudgment:
    """A single graded-relevance judgment for the retrieval-quality eval harness.

    Used by ``RelevanceJudgedQuery`` to drive the retrieval-quality
    measurement harness. Distinct from ``GoldenCase`` (which judges QI
    intent classification) because *retrieval* quality requires per-item
    relevance grades against the orchestrator's ranked output.

    :param item_id: str - Stable item id matching ``RankedItem.item_id``
    :param gain: int - Graded relevance in [0, 4] (0 = irrelevant, 4 = perfect).
        Binary judgments use {0, 1}; graded judgments use the full scale.
        NDCG uses ``2^gain - 1`` as the per-position gain (standard formulation).
        Recall@k / MRR@k treat any ``gain >= 1`` as relevant.
    """
    item_id: str
    gain: int

    def __post_init__(self) -> None:
        if not isinstance(self.item_id, str) or not self.item_id:
            raise ValidationError("RelevanceJudgment.item_id must be a non-empty string")
        if not isinstance(self.gain, int) or isinstance(self.gain, bool):
            raise ValidationError("RelevanceJudgment.gain must be an int")
        if self.gain < 0 or self.gain > 4:
            raise ValidationError("RelevanceJudgment.gain must be in [0, 4]")


@dataclass
class RelevanceJudgedQuery:
    """A query plus its complete set of relevance judgments.

    Consumed by ``semantic_search.eval.retrieval_quality.RetrievalQualityEvaluator``
    to compute NDCG@k / Recall@k / MRR@k against the orchestrator's ranked
    output. The full judgment set is the closed world for that query — items
    not listed are treated as ``gain=0`` (irrelevant) for metric purposes.

    :param query_id: str - Stable id (sha-prefixed for traceability)
    :param input_query: str - Raw user query (sanitized before storage)
    :param judgments: List[RelevanceJudgment] - Per-item graded relevance.
        MUST contain at least one judgment with ``gain >= 1`` so Recall@k /
        MRR@k are well-defined (a query with zero relevant items is excluded
        upstream by the dataset loader).
    :param difficulty: str - One of GOLDEN_DIFFICULTIES (re-uses the same
        difficulty taxonomy as ``GoldenCase`` for slice-level reporting)
    :param edge_type: str - One of GOLDEN_EDGE_TYPES (same rationale)
    :param reviewer_id: Optional[str] - Reviewer id (for audit)
    :param created_at: float - Unix timestamp at construction
    """
    query_id: str
    input_query: str
    judgments: List[RelevanceJudgment]
    difficulty: str
    edge_type: str
    reviewer_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, str) or not self.query_id:
            raise ValidationError("RelevanceJudgedQuery.query_id must be a non-empty string")
        if not isinstance(self.input_query, str) or not self.input_query:
            raise ValidationError("RelevanceJudgedQuery.input_query must be a non-empty string")
        if not isinstance(self.judgments, list) or not self.judgments:
            raise ValidationError("RelevanceJudgedQuery.judgments must be a non-empty list")
        for j in self.judgments:
            if not isinstance(j, RelevanceJudgment):
                raise ValidationError("RelevanceJudgedQuery.judgments entries must be RelevanceJudgment")
        seen: set = set()
        for j in self.judgments:
            if j.item_id in seen:
                raise ValidationError(f"RelevanceJudgedQuery.judgments duplicate item_id={j.item_id}")
            seen.add(j.item_id)
        if not any(j.gain >= 1 for j in self.judgments):
            raise ValidationError("RelevanceJudgedQuery.judgments must include at least one relevant item (gain >= 1)")
        if self.difficulty not in GOLDEN_DIFFICULTIES:
            raise ValidationError(f"RelevanceJudgedQuery.difficulty must be one of {sorted(GOLDEN_DIFFICULTIES)}")
        if self.edge_type not in GOLDEN_EDGE_TYPES:
            raise ValidationError(f"RelevanceJudgedQuery.edge_type must be one of {sorted(GOLDEN_EDGE_TYPES)}")

    @staticmethod
    def new_query_id() -> str:
        """Generate a query_id for a new judged query."""
        return _new_id('jq')

    @property
    def relevant_item_ids(self) -> List[str]:
        """Item ids with ``gain >= 1`` (relevant set for Recall / MRR)."""
        return [j.item_id for j in self.judgments if j.gain >= 1]

    @property
    def gain_by_item_id(self) -> Dict[str, int]:
        """Quick-lookup map ``item_id -> gain`` (zero for unjudged items)."""
        return {j.item_id: j.gain for j in self.judgments}


@dataclass
class BackendHealth:
    """Snapshot of one backend's health at a point in time.
    :param backend: str - One of BACKEND_HEALTH_BACKENDS
    :param state: str - One of BACKEND_HEALTH_STATES
    :param failure_rate: float - Observed failure rate in the rolling window (in [0,1])
    :param observation_count: int - Calls observed in the rolling window (>= 0)
    :param updated_at: float - Unix timestamp of last update
    """
    backend: str
    state: str
    failure_rate: float
    observation_count: int
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.backend not in BACKEND_HEALTH_BACKENDS:
            raise ValidationError(f"BackendHealth.backend must be one of {sorted(BACKEND_HEALTH_BACKENDS)}")
        if self.state not in BACKEND_HEALTH_STATES:
            raise ValidationError(f"BackendHealth.state must be one of {sorted(BACKEND_HEALTH_STATES)}")
        if not 0.0 <= float(self.failure_rate) <= 1.0:
            raise ValidationError("BackendHealth.failure_rate must be in [0,1]")
        if int(self.observation_count) < 0:
            raise ValidationError("BackendHealth.observation_count must be >= 0")


@dataclass
class CircuitBreakerState:
    """Snapshot of the LLM circuit breaker.
    :param state: str - One of CIRCUIT_STATES
    :param failure_count: int - Failures observed in the rolling window (>= 0)
    :param success_count: int - Successes observed in the rolling window (>= 0)
    :param consecutive_half_open_successes: int - Probe successes since last OPEN (>= 0)
    :param opened_at: Optional[float] - Unix timestamp the breaker last tripped OPEN
    :param total_open_transitions: int - Lifetime count of CLOSED/HALF_OPEN -> OPEN transitions
    :param updated_at: float - Unix timestamp of last update
    """
    state: str
    failure_count: int
    success_count: int
    consecutive_half_open_successes: int
    opened_at: Optional[float]
    total_open_transitions: int = 0
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.state not in CIRCUIT_STATES:
            raise ValidationError(f"CircuitBreakerState.state must be one of {sorted(CIRCUIT_STATES)}")
        if int(self.failure_count) < 0:
            raise ValidationError("CircuitBreakerState.failure_count must be >= 0")
        if int(self.success_count) < 0:
            raise ValidationError("CircuitBreakerState.success_count must be >= 0")
        if int(self.consecutive_half_open_successes) < 0:
            raise ValidationError("CircuitBreakerState.consecutive_half_open_successes must be >= 0")
        if int(self.total_open_transitions) < 0:
            raise ValidationError("CircuitBreakerState.total_open_transitions must be >= 0")


@dataclass
class AnalyticsSubstratePlan:
    """Typed analytics execution substrate given shared backend health (Layer 6 matrix).
    :param mode: str - One of ``ANALYTICS_SUBSTRATE_MODES``
    :param notes: List[str] - Cross-subsystem hints (e.g. vector+CH unhealthy)
    """
    mode: str
    notes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if str(self.mode) not in ANALYTICS_SUBSTRATE_MODES:
            raise ValidationError(f"AnalyticsSubstratePlan.mode must be one of {sorted(ANALYTICS_SUBSTRATE_MODES)}")
        if not isinstance(self.notes, list):
            raise ValidationError("AnalyticsSubstratePlan.notes must be a list")


@dataclass
class DegradationPlan:
    """Typed handshake between `DegradationPlanner` (retrieval fallback selection) and the orchestrator.

    The planner consumes a `BackendHealth` snapshot and returns this contract; the
    orchestrator reads it without re-deriving health logic. Validation in
    `__post_init__` rejects malformed plans at construction so a planner bug
    cannot silently produce a "no backends, mode=normal" plan that the
    orchestrator then trusts.

    :param active_backends: List[str] - Backends to actually invoke. Subset of
        `DEGRADATION_RETRIEVAL_BACKENDS`. Empty only when `mode='cache_only'`.
    :param dropped_backends: List[str] - Backends bypassed due to health. Subset
        of `DEGRADATION_RETRIEVAL_BACKENDS`. Disjoint from `active_backends`.
    :param mode: str - One of `DEGRADATION_MODES`. Must be `'normal'` iff
        `dropped_backends` is empty AND `active_backends` is non-empty.
    :param notes: List[str] - Per-drop human-readable reasons (one entry per
        dropped backend, plus optional fall-back markers).
    """
    active_backends: List[str]
    dropped_backends: List[str] = field(default_factory=list)
    mode: str = 'normal'
    notes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.active_backends, list):
            raise ValidationError("DegradationPlan.active_backends must be a list")
        if not isinstance(self.dropped_backends, list):
            raise ValidationError("DegradationPlan.dropped_backends must be a list")
        if not isinstance(self.notes, list):
            raise ValidationError("DegradationPlan.notes must be a list")
        if self.mode not in DEGRADATION_MODES:
            raise ValidationError(f"DegradationPlan.mode must be one of {sorted(DEGRADATION_MODES)}, got {self.mode!r}")
        for b in self.active_backends:
            if b not in DEGRADATION_RETRIEVAL_BACKENDS:
                raise ValidationError(f"DegradationPlan.active_backends contains invalid backend {b!r}; allowed={sorted(DEGRADATION_RETRIEVAL_BACKENDS)}")
        for b in self.dropped_backends:
            if b not in DEGRADATION_RETRIEVAL_BACKENDS:
                raise ValidationError(f"DegradationPlan.dropped_backends contains invalid backend {b!r}; allowed={sorted(DEGRADATION_RETRIEVAL_BACKENDS)}")
        if set(self.active_backends) & set(self.dropped_backends):
            raise ValidationError("DegradationPlan.active_backends and dropped_backends must be disjoint")
        if self.mode == 'normal' and self.dropped_backends:
            raise ValidationError("DegradationPlan.mode='normal' is incompatible with non-empty dropped_backends")
        if self.mode == 'cache_only' and self.active_backends:
            raise ValidationError("DegradationPlan.mode='cache_only' is incompatible with non-empty active_backends")


PROMPT_VERSION_AB_OUTCOMES = frozenset({'pending', 'inconclusive', 'won', 'lost', 'rolled_back'})


@dataclass
class PromptVersion:
    """Typed record for a registered prompt version and related audit fields.

    :param version_id: str - Stable id used as `prompt_tag` in LLM logs (non-empty)
    :param system_prompt: str - System role content (non-empty, fully rendered — no
        unbound `{placeholder}` tokens may leak; the registry renders defaults via
        `build_system_prompt` before storing)
    :param user_template: str - User role template (may include `{variable}`
        placeholders bound at call time by `build_user_prompt`)
    :param source_pattern_ids: List[str] - Optional correlation ids for audit (non-empty strings)
    :param created_at: float - Unix timestamp at construction
    :param baseline_scores: Dict[str, float] - Per-metric baseline scores (values in [0, 1])
    :param ab_outcome: str - One of ``PROMPT_VERSION_AB_OUTCOMES``
    :param rollback_target: str - Prior ``version_id`` when ``ab_outcome`` is ``rolled_back``; empty for genesis
    """
    version_id: str
    system_prompt: str
    user_template: str
    source_pattern_ids: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    baseline_scores: Dict[str, float] = field(default_factory=dict)
    ab_outcome: str = 'pending'
    rollback_target: str = ''

    def __post_init__(self) -> None:
        if not isinstance(self.version_id, str) or not self.version_id.strip():
            raise ValidationError("PromptVersion.version_id must be a non-empty string")
        if not isinstance(self.system_prompt, str) or not self.system_prompt.strip():
            raise ValidationError("PromptVersion.system_prompt must be a non-empty string")
        if not isinstance(self.user_template, str) or not self.user_template.strip():
            raise ValidationError("PromptVersion.user_template must be a non-empty string")
        if not isinstance(self.source_pattern_ids, list):
            raise ValidationError("PromptVersion.source_pattern_ids must be a list")
        for pid in self.source_pattern_ids:
            if not isinstance(pid, str) or not pid:
                raise ValidationError("PromptVersion.source_pattern_ids entries must be non-empty strings")
        if float(self.created_at) <= 0.0:
            raise ValidationError("PromptVersion.created_at must be > 0")
        if not isinstance(self.baseline_scores, dict):
            raise ValidationError("PromptVersion.baseline_scores must be a dict[str, float]")
        for k, v in self.baseline_scores.items():
            if not isinstance(k, str) or not k:
                raise ValidationError("PromptVersion.baseline_scores keys must be non-empty strings")
            if not isinstance(v, (int, float)):
                raise ValidationError(f"PromptVersion.baseline_scores[{k}] must be numeric")
            if not 0.0 <= float(v) <= 1.0:
                raise ValidationError(f"PromptVersion.baseline_scores[{k}] must be in [0, 1]")
        if not isinstance(self.ab_outcome, str) or self.ab_outcome not in PROMPT_VERSION_AB_OUTCOMES:
            raise ValidationError(f"PromptVersion.ab_outcome must be one of {sorted(PROMPT_VERSION_AB_OUTCOMES)}")
        if not isinstance(self.rollback_target, str):
            raise ValidationError("PromptVersion.rollback_target must be a string (empty allowed for genesis)")
        if self.rollback_target and self.rollback_target == self.version_id:
            raise ValidationError("PromptVersion.rollback_target must not equal version_id (no self-rollback)")


@dataclass
class SearchObservation:
    """A single end-to-end search observation captured by the orchestrator.

    Used by the measurement subsystem. Stores only the structured facts
    needed to compute proxy signals — never the raw query text — so the
    rolling window does not become a covert log of user input.

    :param request_id: str - Trace id of the search request
    :param search_id: str - Durable search-interaction id (empty when not stamped)
    :param session_id: str - Session id (raw — see registry wiring choice)
    :param query_type: str - QI-derived primary query_type (one of QUERY_TYPES)
    :param decision_tier: str - Tier that owned the QI decision
    :param confidence: float - Primary classification confidence in [0,1]
    :param decision_cost_usd: float - LLM cost incurred for the QI decision (>= 0)
    :param result_count: int - Number of items returned to the caller (>= 0)
    :param distinct_item_ratio: float - distinct(item_ids)/result_count, in [0,1]
    :param total_latency_ms: float - End-to-end search latency (>= 0)
    :param cache_hit: Optional[str] - 'exact' / 'semantic' / 'structured' / 'intent_plan' / None
    :param intent_record_id: str - ``IntentRecord`` id captured for the
        measurement correlation join. Empty when the orchestrator did not classify
        (cache hit) and the cached observation predates ``intent_record_id`` wiring.
    :param eranker_applied: Optional[bool] - True when Layer-4 re-ordered results (None when not recorded)
    :param eranker_latency_ms: Optional[float] - eRanker client latency in ms (None when not recorded)
    :param eranker_skipped_reason: Optional[str] - Layer-4 skip reason mirroring ``ERankerOutcome`` (None when absent)
    :param created_at: float - Unix timestamp at construction
    """
    request_id: str
    session_id: str
    query_type: str
    decision_tier: str
    confidence: float
    decision_cost_usd: float
    result_count: int
    distinct_item_ratio: float
    total_latency_ms: float
    cache_hit: Optional[str]
    intent_record_id: str = ''
    search_id: str = ''
    eranker_applied: Optional[bool] = None
    eranker_latency_ms: Optional[float] = None
    eranker_skipped_reason: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValidationError("SearchObservation.request_id must be non-empty")
        if not isinstance(self.session_id, str):
            raise ValidationError("SearchObservation.session_id must be a string")
        if not isinstance(self.intent_record_id, str):
            raise ValidationError("SearchObservation.intent_record_id must be a string")
        if not isinstance(self.search_id, str):
            raise ValidationError("SearchObservation.search_id must be a string")
        if self.query_type not in QUERY_TYPES:
            raise ValidationError(f"SearchObservation.query_type must be one of {sorted(QUERY_TYPES)}")
        if self.decision_tier not in DECISION_TIERS:
            raise ValidationError(f"SearchObservation.decision_tier must be one of {sorted(DECISION_TIERS)}")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValidationError("SearchObservation.confidence must be in [0,1]")
        if float(self.decision_cost_usd) < 0.0:
            raise ValidationError("SearchObservation.decision_cost_usd must be >= 0")
        if int(self.result_count) < 0:
            raise ValidationError("SearchObservation.result_count must be >= 0")
        if not 0.0 <= float(self.distinct_item_ratio) <= 1.0:
            raise ValidationError("SearchObservation.distinct_item_ratio must be in [0,1]")
        if float(self.total_latency_ms) < 0.0:
            raise ValidationError("SearchObservation.total_latency_ms must be >= 0")
        if self.cache_hit is not None and self.cache_hit not in CACHE_HIT_TIER_VALUES:
            raise ValidationError(f"SearchObservation.cache_hit must be one of {sorted(CACHE_HIT_TIER_VALUES)} or None")
        if self.eranker_applied is not None and not isinstance(self.eranker_applied, bool):
            raise ValidationError("SearchObservation.eranker_applied must be bool when set")
        if self.eranker_latency_ms is not None and float(self.eranker_latency_ms) < 0.0:
            raise ValidationError("SearchObservation.eranker_latency_ms must be >= 0 when set")
        if self.eranker_skipped_reason is not None:
            if not isinstance(self.eranker_skipped_reason, str) or not self.eranker_skipped_reason.strip():
                raise ValidationError("SearchObservation.eranker_skipped_reason must be a non-empty string when set")


@dataclass
class ProxySignal:
    """One row in the proxy-signal report.

    The evaluator emits a ``ProxySignal`` for every signal in the active
    measurement set. Signals the platform cannot yet observe (offline
    metrics, surface affordances not yet shipped) carry
    ``status='not_instrumented'`` so the dashboard tells the truth instead
    of fabricating a number.

    :param name: str - Signal name (matches the table row)
    :param value: Optional[float] - Computed value (None when not_instrumented)
    :param threshold: Optional[float] - Configured threshold (None when not applicable)
    :param direction: str - One of PROXY_SIGNAL_DIRECTIONS
    :param status: str - One of PROXY_SIGNAL_STATUSES
    :param sample_size: int - Observations contributing to the value (>= 0)
    :param details: Dict[str, Any] - Optional sub-aggregates (per-type breakdown, etc.)
    """
    name: str
    value: Optional[float]
    threshold: Optional[float]
    direction: str
    status: str
    sample_size: int
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValidationError("ProxySignal.name must be non-empty")
        if self.direction not in PROXY_SIGNAL_DIRECTIONS:
            raise ValidationError(f"ProxySignal.direction must be one of {sorted(PROXY_SIGNAL_DIRECTIONS)}")
        if self.status not in PROXY_SIGNAL_STATUSES:
            raise ValidationError(f"ProxySignal.status must be one of {sorted(PROXY_SIGNAL_STATUSES)}")
        if int(self.sample_size) < 0:
            raise ValidationError("ProxySignal.sample_size must be >= 0")
        if self.status == 'not_instrumented' and self.value is not None:
            raise ValidationError("ProxySignal.value must be None when status=not_instrumented")
        if self.status in ('ok', 'breach') and self.value is None:
            raise ValidationError("ProxySignal.value is required when status in {ok, breach}")
        if not isinstance(self.details, dict):
            raise ValidationError("ProxySignal.details must be a dict")


@dataclass
class RouterSeed:
    """A single seed query attached to one archetype.

    Provenance fields (`origin`, `source_id`, `created_at`) let the dashboard answer
    "where did this centroid come from?" — required by oversight.mdc and
    responsible-ai.mdc (explainability by design).

    :param query: str - Seed text (raw user-style query)
    :param archetype: str - One of QUERY_TYPES
    :param origin: str - One of ROUTER_SEED_ORIGINS (manual / synthetic / retrain)
    :param source_id: str - Optional batch / curator / request id (audit trail)
    :param created_at: float - Unix timestamp at construction
    """
    query: str
    archetype: str
    origin: str
    source_id: str = ''
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValidationError("RouterSeed.query must be a non-empty string")
        if self.archetype not in QUERY_TYPES:
            raise ValidationError(f"RouterSeed.archetype must be one of {sorted(QUERY_TYPES)}")
        if self.origin not in ROUTER_SEED_ORIGINS:
            raise ValidationError(f"RouterSeed.origin must be one of {sorted(ROUTER_SEED_ORIGINS)}")
        if not isinstance(self.source_id, str):
            raise ValidationError("RouterSeed.source_id must be a string")


@dataclass
class RouterSeedDataset:
    """Hand-curated + synthetic + retrained seed dataset for the Semantic Router.

    Holds every seed grouped by archetype. Construction guarantees:
    - Every archetype key is in QUERY_TYPES
    - Every value is a list of `RouterSeed` for that archetype
    - No duplicate `(query, archetype)` pairs (de-duplication is the loader's job
      so a malformed file fails fast at startup rather than silently merging)

    :param seeds_by_archetype: Dict[str, List[RouterSeed]] - Seeds grouped by archetype
    :param source_path: str - Filesystem path the dataset was loaded from (audit only)
    :param loaded_at: float - Unix timestamp at construction
    """
    seeds_by_archetype: Dict[str, List['RouterSeed']]
    source_path: str = ''
    loaded_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not isinstance(self.seeds_by_archetype, dict) or not self.seeds_by_archetype:
            raise ValidationError("RouterSeedDataset.seeds_by_archetype must be a non-empty dict")
        if not isinstance(self.source_path, str):
            raise ValidationError("RouterSeedDataset.source_path must be a string")
        seen_pairs = set()
        for archetype, seeds in self.seeds_by_archetype.items():
            if archetype not in QUERY_TYPES:
                raise ValidationError(f"RouterSeedDataset archetype '{archetype}' not in QUERY_TYPES")
            if not isinstance(seeds, list) or not seeds:
                raise ValidationError(f"RouterSeedDataset.seeds_by_archetype['{archetype}'] must be a non-empty list")
            for s in seeds:
                if not isinstance(s, RouterSeed):
                    raise ValidationError(f"RouterSeedDataset.seeds_by_archetype['{archetype}'] elements must be RouterSeed")
                if s.archetype != archetype:
                    raise ValidationError(f"RouterSeed.archetype='{s.archetype}' mismatched against parent key '{archetype}'")
                pair = (s.query.strip().lower(), archetype)
                if pair in seen_pairs:
                    raise ValidationError(f"RouterSeedDataset duplicate seed query='{s.query}' archetype='{archetype}'")
                seen_pairs.add(pair)

    def archetypes(self) -> List[str]:
        """Sorted list of archetype names present in the dataset."""
        return sorted(self.seeds_by_archetype.keys())

    def counts(self) -> Dict[str, int]:
        """Per-archetype seed counts (for coverage reporting)."""
        return {a: len(s) for a, s in self.seeds_by_archetype.items()}

    def total(self) -> int:
        """Total number of seeds across all archetypes."""
        return sum(len(v) for v in self.seeds_by_archetype.values())

    def texts(self, archetype: str) -> List[str]:
        """Return raw query texts for one archetype (used by `SemanticRouter`)."""
        if archetype not in self.seeds_by_archetype:
            raise ValidationError(f"RouterSeedDataset has no seeds for archetype '{archetype}'")
        return [s.query for s in self.seeds_by_archetype[archetype]]


@dataclass
class CentroidRetrainCandidate:
    """A retrain candidate built from real-traffic positives.

    Produced by `CentroidRetrainer.build_candidate()`. Holds the proposed new
    centroids plus the metadata needed for the shadow / promote decision.

    :param candidate_id: str - Stable id (audit trail)
    :param new_centroids: Dict[str, List[float]] - Proposed centroid per archetype
    :param sample_counts: Dict[str, int] - Trusted-positive count per archetype
    :param window_seconds: float - Time window of source positives (>= 0)
    :param created_at: float - Unix timestamp at construction
    """
    candidate_id: str
    new_centroids: Dict[str, List[float]]
    sample_counts: Dict[str, int]
    window_seconds: float
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValidationError("CentroidRetrainCandidate.candidate_id must be non-empty")
        if not isinstance(self.new_centroids, dict) or not self.new_centroids:
            raise ValidationError("CentroidRetrainCandidate.new_centroids must be a non-empty dict")
        if not isinstance(self.sample_counts, dict):
            raise ValidationError("CentroidRetrainCandidate.sample_counts must be a dict")
        if float(self.window_seconds) < 0.0:
            raise ValidationError("CentroidRetrainCandidate.window_seconds must be >= 0")
        for archetype, vec in self.new_centroids.items():
            if archetype not in QUERY_TYPES:
                raise ValidationError(f"CentroidRetrainCandidate.new_centroids archetype '{archetype}' not in QUERY_TYPES")
            if not isinstance(vec, list) or not vec:
                raise ValidationError(f"CentroidRetrainCandidate.new_centroids['{archetype}'] must be a non-empty list")
            if archetype not in self.sample_counts:
                raise ValidationError(f"CentroidRetrainCandidate.sample_counts missing archetype '{archetype}'")


@dataclass
class CentroidRetrainVerdict:
    """Verdict from running a CentroidRetrainCandidate through shadow + gates.

    :param candidate_id: str - The candidate this verdict applies to
    :param verdict: str - One of CENTROID_RETRAIN_VERDICTS
    :param shadow_agreement_rate: float - Fraction of shadow queries where new centroids
        agreed with the current router (in [0,1]).  Higher = safer to promote.
    :param min_sample_per_archetype: int - The smallest archetype sample size in the
        candidate.  If below the configured floor, verdict='reject' is forced.
    :param reasons: List[str] - Audit-log reasons (which gates passed / failed)
    :param decided_at: float - Unix timestamp at verdict
    """
    candidate_id: str
    verdict: str
    shadow_agreement_rate: float
    min_sample_per_archetype: int
    reasons: List[str]
    decided_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValidationError("CentroidRetrainVerdict.candidate_id must be non-empty")
        if self.verdict not in CENTROID_RETRAIN_VERDICTS:
            raise ValidationError(f"CentroidRetrainVerdict.verdict must be one of {sorted(CENTROID_RETRAIN_VERDICTS)}")
        if not 0.0 <= float(self.shadow_agreement_rate) <= 1.0:
            raise ValidationError("CentroidRetrainVerdict.shadow_agreement_rate must be in [0,1]")
        if int(self.min_sample_per_archetype) < 0:
            raise ValidationError("CentroidRetrainVerdict.min_sample_per_archetype must be >= 0")
        if not isinstance(self.reasons, list):
            raise ValidationError("CentroidRetrainVerdict.reasons must be a list")


@dataclass(frozen=True)
class CentroidRetrainerCycleSummary:
    """Summary of one CentroidRetrainerDriver cycle.

    :param cycle_id: str - Stable id for the cycle (audit trail)
    :param ran_at: float - asyncio loop time at cycle start
    :param skipped: bool - True when the cycle produced no candidate (insufficient positives or build_candidate rejected)
    :param skip_reason: Optional[str] - Human-readable reason when skipped=True
    :param candidate_id: Optional[str] - CentroidRetrainCandidate.candidate_id when a candidate was built
    :param verdict: Optional[str] - One of CENTROID_RETRAIN_VERDICTS when decide() ran
    :param shadow_agreement_rate: Optional[float] - Agreement rate from decide() when verdict is set
    :param promoted: bool - True when verdict='promote' and swap_centroids fired
    :param error: Optional[str] - Exception type + message when an unexpected error aborted the cycle
    """
    cycle_id: str
    ran_at: float
    skipped: bool
    skip_reason: Optional[str]
    candidate_id: Optional[str]
    verdict: Optional[str]
    shadow_agreement_rate: Optional[float]
    promoted: bool
    error: Optional[str]

    def __post_init__(self) -> None:
        if not self.cycle_id:
            raise ValidationError("CentroidRetrainerCycleSummary.cycle_id must be non-empty")
        if self.verdict is not None and self.verdict not in CENTROID_RETRAIN_VERDICTS:
            raise ValidationError(f"CentroidRetrainerCycleSummary.verdict invalid: {self.verdict!r}")
        if self.shadow_agreement_rate is not None and not (0.0 <= self.shadow_agreement_rate <= 1.0):
            raise ValidationError(f"CentroidRetrainerCycleSummary.shadow_agreement_rate out of [0,1]: {self.shadow_agreement_rate}")


@dataclass
class ExploreCard:
    """One card surfaced inside an `ExploreRail`.
    :param item_id: str - Stable item identifier (matches `RankedItem.item_id`)
    :param fused_score: float - Per-rail score (>= 0); rails compute this from their own native signal
    :param source_rail: str - One of EXPLORE_RAIL_KINDS — provenance for dedupe + dashboards
    :param payload: Dict[str, Any] - Backend payload (mirrors RankedItem.payload contract)
    """
    item_id: str
    fused_score: float
    source_rail: str
    payload: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.item_id:
            raise ValidationError("ExploreCard.item_id must be non-empty")
        if float(self.fused_score) < 0.0:
            raise ValidationError("ExploreCard.fused_score must be >= 0")
        if self.source_rail not in EXPLORE_RAIL_KINDS:
            raise ValidationError(f"ExploreCard.source_rail must be one of {sorted(EXPLORE_RAIL_KINDS)}")
        if not isinstance(self.payload, dict):
            raise ValidationError("ExploreCard.payload must be a dict")


@dataclass
class ExploreRail:
    """One rail in a landing/explore response.
    Holds the rail kind, a human-readable title (becomes the explanation chip
    on each rail), the cards in display order, and per-rail latency.
    :param rail_id: str - One of EXPLORE_RAIL_KINDS
    :param title: str - Rail title shown to the user (config-driven for trending /
        ending_soon / fallback)
    :param cards: List[ExploreCard] - Cards in rail-native order (no further sorting)
    :param explanation: str - Audit-log + UI explanation chip (no PII)
    :param latency_ms: float - Per-rail composition wall time (>= 0)
    """
    rail_id: str
    title: str
    cards: List['ExploreCard']
    explanation: str
    latency_ms: float

    def __post_init__(self) -> None:
        if self.rail_id not in EXPLORE_RAIL_KINDS:
            raise ValidationError(f"ExploreRail.rail_id must be one of {sorted(EXPLORE_RAIL_KINDS)}")
        if not isinstance(self.title, str) or not self.title:
            raise ValidationError("ExploreRail.title must be a non-empty string")
        if not isinstance(self.cards, list):
            raise ValidationError("ExploreRail.cards must be a list")
        for c in self.cards:
            if c.source_rail != self.rail_id:
                raise ValidationError(f"ExploreRail.cards[*].source_rail must equal rail_id ({self.rail_id})")
        if not isinstance(self.explanation, str):
            raise ValidationError("ExploreRail.explanation must be a string")
        if float(self.latency_ms) < 0.0:
            raise ValidationError("ExploreRail.latency_ms must be >= 0")


@dataclass
class LandingRailResponse:
    """The full landing-rail response — also the §3 zero-result fallback envelope.
    :param request_id: str - Correlation id
    :param user_id: Optional[str] - Authenticated user id (None for /search fallback path
        when caller is unauthenticated or has opted out of history)
    :param rails: List[ExploreRail] - Rails in display order (typically trending,
        ending_soon — fallback rail appears when primary rails are empty)
    :param generated_at: float - Unix timestamp at composition
    :param source: str - One of EXPLORE_RESPONSE_SOURCES — call-path provenance
    """
    request_id: str
    user_id: Optional[str]
    rails: List['ExploreRail']
    generated_at: float
    source: str

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValidationError("LandingRailResponse.request_id must be non-empty")
        if self.user_id is not None and not isinstance(self.user_id, str):
            raise ValidationError("LandingRailResponse.user_id must be a string when set")
        if not isinstance(self.rails, list) or len(self.rails) == 0:
            raise ValidationError("LandingRailResponse.rails must be a non-empty list (use the fallback rail when no real cards)")
        if float(self.generated_at) < 0.0:
            raise ValidationError("LandingRailResponse.generated_at must be >= 0")
        if self.source not in EXPLORE_RESPONSE_SOURCES:
            raise ValidationError(f"LandingRailResponse.source must be one of {sorted(EXPLORE_RESPONSE_SOURCES)}")

    @staticmethod
    def new_request_id() -> str:
        """Generate a request_id for a new landing-rail response."""
        return _new_id('rail')


@dataclass
class ZeroResultGuardOutcome:
    """Outcome stamped on the search response when the Zero-Result Guard fires.
    The ladder is ``relax filters -> semantic-only -> explore``. The outcome
    captures which step actually rescued the search so dashboards can
    compute the per-archetype zero-result rate without a separate signal.
    :param fired: bool - True iff the guard ran (results were initially empty)
    :param ladder_step: str - One of ZERO_RESULT_LADDER_STEPS — step that rescued
        the search; 'none' iff fired=False; 'explore_fallback' iff every retrieval
        attempt returned zero
    :param original_filter_count: int - Filter count before the guard ran (>= 0)
    :param relaxed_filter_count: int - Filter count after relaxation (>= 0).
        Equals original_filter_count when the guard did not need to relax.
    :param relaxation_reason: str - Audit-log reason ('zero_after_initial_retrieve',
        'still_zero_after_relax', 'still_zero_after_semantic_only', or '' when not fired)
    :param dropped_filter_names: List[str] - Ordered list of filter slot names the guard
        dropped during relax_filters (e.g. ['price_max', 'auction_type']). Empty when
        fired=False or when a non-relax step rescued the query.
    """
    fired: bool
    ladder_step: str
    original_filter_count: int
    relaxed_filter_count: int
    relaxation_reason: str
    dropped_filter_names: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.fired, bool):
            raise ValidationError("ZeroResultGuardOutcome.fired must be bool")
        if self.ladder_step not in ZERO_RESULT_LADDER_STEPS:
            raise ValidationError(f"ZeroResultGuardOutcome.ladder_step must be one of {sorted(ZERO_RESULT_LADDER_STEPS)}")
        if not self.fired and self.ladder_step != 'none':
            raise ValidationError("ZeroResultGuardOutcome.ladder_step must be 'none' when fired=False")
        if self.fired and self.ladder_step == 'none':
            raise ValidationError("ZeroResultGuardOutcome.ladder_step must not be 'none' when fired=True")
        if int(self.original_filter_count) < 0:
            raise ValidationError("ZeroResultGuardOutcome.original_filter_count must be >= 0")
        if int(self.relaxed_filter_count) < 0:
            raise ValidationError("ZeroResultGuardOutcome.relaxed_filter_count must be >= 0")
        if int(self.relaxed_filter_count) > int(self.original_filter_count):
            raise ValidationError("ZeroResultGuardOutcome.relaxed_filter_count must be <= original_filter_count")
        if not isinstance(self.relaxation_reason, str):
            raise ValidationError("ZeroResultGuardOutcome.relaxation_reason must be a string")
        if not self.fired and self.relaxation_reason:
            raise ValidationError("ZeroResultGuardOutcome.relaxation_reason must be '' when fired=False")
        if not isinstance(self.dropped_filter_names, list):
            raise ValidationError("ZeroResultGuardOutcome.dropped_filter_names must be a list")


# Event kinds accepted by the listing- and bid-stream consumers (FIND API +
# Auctions Kinesis listing stream + real-time bid-event stream).
LISTING_EVENT_KINDS = frozenset({'created', 'updated', 'ended', 'removed'})
BID_EVENT_KINDS = frozenset({'bid_placed', 'bid_won'})


@dataclass
class ListingEvent:
    """One Kinesis listing-stream event.

    The contract is the field set the production stream emits per upsert. The
    in-memory stub (``semantic_search.ingest.in_memory_consumer``) consumes the
    same dataclass so swapping in the live Kinesis consumer later does not
    change a single downstream call site.

    :param event_id: str - Stream-assigned event id (correlation across pipelines)
    :param item_id: str - Listing identifier (Qdrant payload key)
    :param event_kind: str - One of LISTING_EVENT_KINDS
    :param payload: Dict[str, Any] - Updated payload fields (tld, price, name_length, ...)
    :param event_time: float - Source-side wall-clock timestamp (Unix seconds)
    """
    event_id: str
    item_id: str
    event_kind: str
    payload: Dict[str, Any]
    event_time: float

    def __post_init__(self) -> None:
        if not self.event_id or not isinstance(self.event_id, str):
            raise ValidationError("ListingEvent.event_id must be a non-empty string")
        if not self.item_id or not isinstance(self.item_id, str):
            raise ValidationError("ListingEvent.item_id must be a non-empty string")
        if self.event_kind not in LISTING_EVENT_KINDS:
            raise ValidationError(f"ListingEvent.event_kind must be one of {sorted(LISTING_EVENT_KINDS)}")
        if not isinstance(self.payload, dict):
            raise ValidationError("ListingEvent.payload must be a dict")
        if float(self.event_time) <= 0.0:
            raise ValidationError("ListingEvent.event_time must be > 0")

    @staticmethod
    def new_event_id() -> str:
        """Generate an event_id for fixture / test use."""
        return _new_id('lst-evt')

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ListingEvent':
        """Build from a raw dict (used by ``seed_from_dict`` on the stub).

        Rejects missing required keys, wrong types, and unknown kinds with
        ``ValidationError`` so seed-data drift surfaces at ingest time.
        """
        if not isinstance(d, dict):
            raise ValidationError(f"ListingEvent.from_dict: expected dict, got {type(d).__name__}")
        for required in ('event_id', 'item_id', 'event_kind', 'payload', 'event_time'):
            if required not in d:
                raise ValidationError(f"ListingEvent.from_dict missing required key '{required}'")
        return cls(event_id=str(d['event_id']), item_id=str(d['item_id']), event_kind=str(d['event_kind']), payload=dict(d['payload']), event_time=float(d['event_time']))


@dataclass
class BidEvent:
    """One Kinesis bid-stream event.

    Drives trending-now / ending-soon ranker boosts and live ReTiRe triggers.
    The in-memory stub consumes the same dataclass.

    :param event_id: str - Stream-assigned event id
    :param item_id: str - Auction identifier (matches a Listing's item_id)
    :param event_kind: str - One of BID_EVENT_KINDS
    :param bid_amount_usd: float - Bid value in USD (>= 0)
    :param event_time: float - Source-side wall-clock timestamp (Unix seconds)
    """
    event_id: str
    item_id: str
    event_kind: str
    bid_amount_usd: float
    event_time: float

    def __post_init__(self) -> None:
        if not self.event_id or not isinstance(self.event_id, str):
            raise ValidationError("BidEvent.event_id must be a non-empty string")
        if not self.item_id or not isinstance(self.item_id, str):
            raise ValidationError("BidEvent.item_id must be a non-empty string")
        if self.event_kind not in BID_EVENT_KINDS:
            raise ValidationError(f"BidEvent.event_kind must be one of {sorted(BID_EVENT_KINDS)}")
        if float(self.bid_amount_usd) < 0.0:
            raise ValidationError("BidEvent.bid_amount_usd must be >= 0")
        if float(self.event_time) <= 0.0:
            raise ValidationError("BidEvent.event_time must be > 0")

    @staticmethod
    def new_event_id() -> str:
        """Generate an event_id for fixture / test use."""
        return _new_id('bid-evt')

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'BidEvent':
        if not isinstance(d, dict):
            raise ValidationError(f"BidEvent.from_dict: expected dict, got {type(d).__name__}")
        for required in ('event_id', 'item_id', 'event_kind', 'bid_amount_usd', 'event_time'):
            if required not in d:
                raise ValidationError(f"BidEvent.from_dict missing required key '{required}'")
        return cls(event_id=str(d['event_id']), item_id=str(d['item_id']), event_kind=str(d['event_kind']), bid_amount_usd=float(d['bid_amount_usd']), event_time=float(d['event_time']))


# consumer interfaces. The live (Kinesis) and the test (in-memory)
# implementations must satisfy these Protocols. Cache-invalidation hooks accept
# the new snapshot version produced by an event and return None.
SnapshotInvalidationHook = Callable[[int], None]


@runtime_checkable
class ListingEventConsumer(Protocol):
    """Protocol every listing-stream consumer (live or stubbed) must satisfy."""

    def register_invalidation_hook(self, hook: SnapshotInvalidationHook) -> None: ...
    async def consume(self, event: ListingEvent) -> int: ...
    @property
    def snapshot_version(self) -> int: ...


@runtime_checkable
class BidEventConsumer(Protocol):
    """Protocol every bid-stream consumer (live or stubbed) must satisfy."""

    def register_invalidation_hook(self, hook: SnapshotInvalidationHook) -> None: ...
    async def consume(self, event: BidEvent) -> int: ...
    @property
    def snapshot_version(self) -> int: ...
