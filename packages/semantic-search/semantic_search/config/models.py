"""Typed configuration dataclasses for semantic_search subsystems.
Each subsystem dataclass enforces invariants in `__post_init__` and rejects
malformed input via `from_dict`. No silent defaults — every field is required
unless explicitly stated as Optional in the schema.
"""
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from semantic_search.config.analytics_models import RemoteCacheLayerConfig, _coerce_bool
from semantic_search.config.nl_to_sql_models import NLToSQLConfig
from semantic_search.config.subsystem_bundle import (
    FeedbackConfig,
    IdentityConfig,
    LLMJudgeConfig,
    RetrievalEvalConfig,
)
from semantic_search.contracts import ASSISTED_CONVERSION_POSITIVE_SIGNALS, MEASUREMENT_SLICE_KEYS
from semantic_search.core.exceptions import ConfigurationError


# Default upper bound (seconds) for dynamic ending-soon horizon overrides. Matches base.yaml max_horizon_seconds.
_ENDING_SOON_DEFAULT_MAX_HORIZON_SECONDS: int = 604800


def _require(d: Dict[str, Any], keys: List[str], context: str) -> None:
    """Raise ConfigurationError if any key is missing from `d`."""
    if not isinstance(d, dict):
        raise ConfigurationError(f"{context}: expected dict, got {type(d).__name__}")
    for key in keys:
        if key not in d:
            raise ConfigurationError(f"{context}.{key} is required")


@dataclass
class RateLimitConfig:
    """Per-session rate-limit config.

    The middleware applies one bucket per ``X-Session-Id`` request header
    (anonymous traffic without the header falls back to client IP) and
    rejects requests above ``max_requests_per_window`` with HTTP 429.

    :param enabled: Master toggle.
    :param max_requests_per_window: Max requests permitted per (session, window).
    :param window_seconds: Sliding-window length.
    :param session_state_max: LRU cap on tracked buckets (memory bound).
    :param paths: List of URL prefixes the limiter applies to. An empty list
        disables the limiter even when ``enabled=True`` (defensive default).
    :param burst_requests: Extra requests allowed in the burst window on top of
        the main cap (0 = burst disabled). Lets bid-cascade traffic absorb short
        spikes without hitting 429 immediately.
    :param burst_window_seconds: Length of the burst window in seconds (>= 1).
    """
    enabled: bool = False
    max_requests_per_window: int = 10
    window_seconds: int = 60
    session_state_max: int = 10000
    paths: List[str] = field(default_factory=list)
    burst_requests: int = 0
    burst_window_seconds: int = 10

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("general.rate_limit.enabled must be bool")
        if self.max_requests_per_window < 1:
            raise ConfigurationError("general.rate_limit.max_requests_per_window must be >= 1")
        if self.window_seconds < 1:
            raise ConfigurationError("general.rate_limit.window_seconds must be >= 1")
        if self.session_state_max < 1:
            raise ConfigurationError("general.rate_limit.session_state_max must be >= 1")
        if not isinstance(self.paths, list) or not all(isinstance(p, str) and p.startswith('/') for p in self.paths):
            raise ConfigurationError("general.rate_limit.paths must be a list of URL prefixes starting with '/'")
        if self.burst_requests < 0:
            raise ConfigurationError("general.rate_limit.burst_requests must be >= 0")
        if self.burst_window_seconds < 1:
            raise ConfigurationError("general.rate_limit.burst_window_seconds must be >= 1")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'RateLimitConfig':
        _require(d, ['enabled', 'max_requests_per_window', 'window_seconds', 'session_state_max'], 'rate_limit')
        return cls(
            enabled=bool(d['enabled']),
            max_requests_per_window=int(d['max_requests_per_window']),
            window_seconds=int(d['window_seconds']),
            session_state_max=int(d['session_state_max']),
            paths=[str(p) for p in (d.get('paths') or [])],
            burst_requests=int(d.get('burst_requests') or 0),
            burst_window_seconds=int(d.get('burst_window_seconds') or 10),
        )


_ANALYTICS_RATE_LIMIT_KEY_STRATEGIES = frozenset({'user_id', 'session_id'})


@dataclass
class AnalyticsRateLimitConfig:
    """Per-tenant analytics-call rate limit (gap 4).

    Independent of the ingress ``RateLimitConfig`` because analytics calls
    are far more expensive (LLM gen + warehouse execute + verifier) than
    plain search calls and warrant their own budget. Backed by a separate
    :class:`SlidingWindowRateLimiter` instance built in the registry.

    :param enabled: bool - Master toggle. When False the orchestrator
        skips the limiter check entirely (legacy + first-rollout safety).
    :param max_requests_per_window: int - Cap per (key, window).
    :param window_seconds: int - Sliding-window length.
    :param session_state_max: int - LRU cap on tracked buckets.
    :param key_strategy: str - Either ``'user_id'`` or ``'session_id'``.
        ``user_id`` prefers the authenticated user when present, falls
        back to session id otherwise; ``session_id`` always buckets by
        session id (used in environments without user authentication).
    """
    enabled: bool = False
    max_requests_per_window: int = 30
    window_seconds: int = 60
    session_state_max: int = 10000
    key_strategy: str = 'user_id'

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("general.analytics_rate_limit.enabled must be bool")
        if self.max_requests_per_window < 1:
            raise ConfigurationError("general.analytics_rate_limit.max_requests_per_window must be >= 1")
        if self.window_seconds < 1:
            raise ConfigurationError("general.analytics_rate_limit.window_seconds must be >= 1")
        if self.session_state_max < 1:
            raise ConfigurationError("general.analytics_rate_limit.session_state_max must be >= 1")
        if self.key_strategy not in _ANALYTICS_RATE_LIMIT_KEY_STRATEGIES:
            raise ConfigurationError(f"general.analytics_rate_limit.key_strategy must be one of {sorted(_ANALYTICS_RATE_LIMIT_KEY_STRATEGIES)}; got {self.key_strategy!r}")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'AnalyticsRateLimitConfig':
        _require(d, ['enabled', 'max_requests_per_window', 'window_seconds', 'session_state_max', 'key_strategy'], 'general.analytics_rate_limit')
        return cls(
            enabled=bool(d['enabled']),
            max_requests_per_window=int(d['max_requests_per_window']),
            window_seconds=int(d['window_seconds']),
            session_state_max=int(d['session_state_max']),
            key_strategy=str(d['key_strategy']),
        )


_STARTUP_LOG_VERBOSITY = frozenset({'compact', 'detailed'})
_VALID_LOG_LEVELS = frozenset({'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'})


@dataclass
class AuctionTiebreakConfig:
    """Bounded, relevance-dominant auction-signal tie-break for ranked_results.

    Applied AFTER retrieval metrics are computed, so coherence_score and the
    NDCG/Coherence/Recall metrics are unaffected. The blended sort key is
    ``coherence + max_bonus * (weighted auction signals)``; max_bonus caps the
    reshuffle to near-equal-coherence items so a real relevance gap is never crossed.

    :param enabled: bool - Master switch; False => ranked_results order unchanged
    :param max_bonus: float - Upper bound added to coherence∈[0,1] before re-sort
    :param urgency_horizon_hours: float - Auctions ending within this window get
        nonzero urgency (linear; sooner = higher); beyond it urgency = 0
    :param weight_urgency: float - Weight for ending-soon signal
    :param weight_low_competition: float - Weight for low-bid-count signal
    :param weight_value: float - Weight for govalue_score (log-scaled) signal
    """
    enabled: bool = False
    max_bonus: float = 0.06
    urgency_horizon_hours: float = 72.0
    weight_urgency: float = 0.5
    weight_low_competition: float = 0.2
    weight_value: float = 0.3

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("general.search.auction_tiebreak.enabled must be a bool")
        if not 0.0 <= float(self.max_bonus) <= 1.0:
            raise ConfigurationError("general.search.auction_tiebreak.max_bonus must be in [0, 1]")
        if float(self.urgency_horizon_hours) <= 0.0:
            raise ConfigurationError("general.search.auction_tiebreak.urgency_horizon_hours must be > 0")
        ws = (float(self.weight_urgency), float(self.weight_low_competition), float(self.weight_value))
        if any(w < 0.0 for w in ws):
            raise ConfigurationError("general.search.auction_tiebreak weights must be >= 0")
        if abs(sum(ws) - 1.0) > 1e-6:
            raise ConfigurationError("general.search.auction_tiebreak weights must sum to 1.0")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'AuctionTiebreakConfig':
        """Build from the general.search.auction_tiebreak section (absent => disabled defaults)."""
        d = d or {}
        return cls(
            enabled=bool(d.get('enabled', False)),
            max_bonus=float(d.get('max_bonus', 0.06)),
            urgency_horizon_hours=float(d.get('urgency_horizon_hours', 72.0)),
            weight_urgency=float(d.get('weight_urgency', 0.5)),
            weight_low_competition=float(d.get('weight_low_competition', 0.2)),
            weight_value=float(d.get('weight_value', 0.3)),
        )


@dataclass
class ComplementRankLeaningsConfig:
    """Bounded reorder of ranked_results from analytics / guidance aggregates.

    :param enabled: bool - Master switch
    :param max_bonus: float - Cap added to coherence before re-sort; in [0, 1]
    :param weight_guidance: float - Weight for guidance TLD activity signal
    :param weight_analytics: float - Weight for analytics cohort signal; weights must sum to 1.0
    :param query_types: List[str] - QI types that may receive leanings (analytics, guidance)
    """
    enabled: bool
    max_bonus: float
    weight_guidance: float
    weight_analytics: float
    query_types: List[str]

    def __post_init__(self) -> None:
        _p = "general.search.ranked_results_complement.rank_leanings"
        if not isinstance(self.enabled, bool):
            raise ConfigurationError(f"{_p}.enabled must be a bool")
        if not 0.0 <= float(self.max_bonus) <= 1.0:
            raise ConfigurationError(f"{_p}.max_bonus must be in [0, 1]")
        wg, wa = float(self.weight_guidance), float(self.weight_analytics)
        if wg < 0.0 or wa < 0.0:
            raise ConfigurationError(f"{_p} weights must be >= 0")
        if abs((wg + wa) - 1.0) > 1e-6:
            raise ConfigurationError(f"{_p} weight_guidance + weight_analytics must sum to 1.0")
        if not isinstance(self.query_types, list) or not self.query_types:
            raise ConfigurationError(f"{_p}.query_types must be a non-empty list")
        _allowed = frozenset({'analytics', 'guidance'})
        for qt in self.query_types:
            if not isinstance(qt, str) or qt not in _allowed:
                raise ConfigurationError(
                    f"{_p}.query_types entries must be one of {sorted(_allowed)}; got {qt!r}"
                )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ComplementRankLeaningsConfig':
        """Build from ranked_results_complement.rank_leanings (all keys required)."""
        _p = "general.search.ranked_results_complement.rank_leanings"
        if not isinstance(d, dict) or not d:
            raise ConfigurationError(f"{_p} must be a non-empty mapping")
        _require(d, [
            'enabled',
            'max_bonus',
            'weight_guidance',
            'weight_analytics',
            'query_types',
        ], _p)
        return cls(
            enabled=bool(d['enabled']),
            max_bonus=float(d['max_bonus']),
            weight_guidance=float(d['weight_guidance']),
            weight_analytics=float(d['weight_analytics']),
            query_types=[str(x).strip() for x in d['query_types']],
        )


@dataclass
class RankedResultsComplementConfig:
    """Hybrid-first ranked_results; analytics / explore / guidance complement only.
    
    :param enabled: bool - When True, every QI type retrieves hybrid for ranked_results
    :param merge_explore_rails: bool - When True, RRF-merge explore rails into hybrid when primary is short
    :param merge_explore_rails_query_types: List[str] - QI types that receive rail merge
    :param merge_rrf_k: int - RRF k for hybrid+rails merge; must be >= 1
    :param merge_explore_rails_only_when_primary_short: bool - When True, start/merge
        explore rails only when primary item count is below the enough threshold
    :param merge_explore_rails_primary_enough_fraction: float - Fraction of request
        ``top_k`` that counts as enough primary evidence in ``(0, 1]`` (1.0 = full top_k)
    :param strip_temporal_when_clickhouse_available: bool - Strip catalog-temporal slots on retrieve even when CH up
    :param ensure_nonempty: bool - Run nonempty ladder when hybrid (+ merge) yields zero items
    :param force_semantic_when_empty: bool - Vector-only retrieve when ensure_nonempty and pool empty
    :param force_semantic_top_k: int - Top-k for force-semantic retrieve; must be >= 1
    :param rank_leanings: ComplementRankLeaningsConfig - Bounded reorder from CH aggregates
    """
    enabled: bool
    merge_explore_rails: bool
    merge_explore_rails_query_types: List[str]
    merge_rrf_k: int
    merge_explore_rails_only_when_primary_short: bool
    merge_explore_rails_primary_enough_fraction: float
    strip_temporal_when_clickhouse_available: bool
    ensure_nonempty: bool
    force_semantic_when_empty: bool
    force_semantic_top_k: int
    rank_leanings: ComplementRankLeaningsConfig

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("general.search.ranked_results_complement.enabled must be a bool")
        if not isinstance(self.merge_explore_rails, bool):
            raise ConfigurationError("general.search.ranked_results_complement.merge_explore_rails must be a bool")
        if not isinstance(self.merge_explore_rails_query_types, list) or not self.merge_explore_rails_query_types:
            raise ConfigurationError("general.search.ranked_results_complement.merge_explore_rails_query_types must be a non-empty list")
        _allowed = frozenset({'analytics', 'explore', 'guidance', 'hybrid'})
        for qt in self.merge_explore_rails_query_types:
            if not isinstance(qt, str) or qt not in _allowed:
                raise ConfigurationError(
                    "general.search.ranked_results_complement.merge_explore_rails_query_types "
                    f"entries must be one of {sorted(_allowed)}; got {qt!r}"
                )
        if int(self.merge_rrf_k) < 1:
            raise ConfigurationError("general.search.ranked_results_complement.merge_rrf_k must be >= 1")
        if not isinstance(self.merge_explore_rails_only_when_primary_short, bool):
            raise ConfigurationError(
                "general.search.ranked_results_complement.merge_explore_rails_only_when_primary_short must be a bool"
            )
        _frac = float(self.merge_explore_rails_primary_enough_fraction)
        if not 0.0 < _frac <= 1.0:
            raise ConfigurationError(
                "general.search.ranked_results_complement.merge_explore_rails_primary_enough_fraction "
                "must be in (0, 1]"
            )
        if not isinstance(self.strip_temporal_when_clickhouse_available, bool):
            raise ConfigurationError(
                "general.search.ranked_results_complement.strip_temporal_when_clickhouse_available must be a bool"
            )
        if not isinstance(self.ensure_nonempty, bool):
            raise ConfigurationError("general.search.ranked_results_complement.ensure_nonempty must be a bool")
        if not isinstance(self.force_semantic_when_empty, bool):
            raise ConfigurationError(
                "general.search.ranked_results_complement.force_semantic_when_empty must be a bool"
            )
        if int(self.force_semantic_top_k) < 1:
            raise ConfigurationError("general.search.ranked_results_complement.force_semantic_top_k must be >= 1")
        if not isinstance(self.rank_leanings, ComplementRankLeaningsConfig):
            raise ConfigurationError(
                "general.search.ranked_results_complement.rank_leanings must be a ComplementRankLeaningsConfig"
            )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'RankedResultsComplementConfig':
        """Build from general.search.ranked_results_complement (all keys required)."""
        if not isinstance(d, dict) or not d:
            raise ConfigurationError("general.search.ranked_results_complement must be a non-empty mapping")
        _require(d, [
            'enabled',
            'merge_explore_rails',
            'merge_explore_rails_query_types',
            'merge_rrf_k',
            'merge_explore_rails_only_when_primary_short',
            'merge_explore_rails_primary_enough_fraction',
            'strip_temporal_when_clickhouse_available',
            'ensure_nonempty',
            'force_semantic_when_empty',
            'force_semantic_top_k',
            'rank_leanings',
        ], 'general.search.ranked_results_complement')
        return cls(
            enabled=bool(d['enabled']),
            merge_explore_rails=bool(d['merge_explore_rails']),
            merge_explore_rails_query_types=[str(x).strip() for x in d['merge_explore_rails_query_types']],
            merge_rrf_k=int(d['merge_rrf_k']),
            merge_explore_rails_only_when_primary_short=bool(d['merge_explore_rails_only_when_primary_short']),
            merge_explore_rails_primary_enough_fraction=float(d['merge_explore_rails_primary_enough_fraction']),
            strip_temporal_when_clickhouse_available=bool(d['strip_temporal_when_clickhouse_available']),
            ensure_nonempty=bool(d['ensure_nonempty']),
            force_semantic_when_empty=bool(d['force_semantic_when_empty']),
            force_semantic_top_k=int(d['force_semantic_top_k']),
            rank_leanings=ComplementRankLeaningsConfig.from_dict(d['rank_leanings']),
        )


@dataclass
class TimeoutFallbackConfig:
    """SLA / timeout listing policy for ``get_timeout_fallback``.

    :param rail_first: bool - When True, never replace nonempty explore-rail
        (or Qdrant rail-ladder) cards with semantic-only output
    :param qdrant_rails_when_ch_empty: bool - When True and CH explore rails are
        empty/unavailable, fill from Qdrant ``filter_only_rails`` multi-scroll
    :param qdrant_rails_timeout_seconds: float - Wall timeout for the Qdrant
        rail ladder alone; must be > 0
    :param consume_search_explore_prewarm: bool - When True, prefer the in-flight
        explore-rail task registered by ``search()`` for this request_id before
        starting a second compose_fallback fan-out
    :param search_explore_prewarm_wait_seconds: float - Max seconds
        ``get_timeout_fallback`` waits for ``search()`` to register that task;
        must be > 0
    """
    rail_first: bool
    qdrant_rails_when_ch_empty: bool
    qdrant_rails_timeout_seconds: float
    consume_search_explore_prewarm: bool
    search_explore_prewarm_wait_seconds: float

    def __post_init__(self) -> None:
        _p = "general.search.timeout_fallback"
        if not isinstance(self.rail_first, bool):
            raise ConfigurationError(f"{_p}.rail_first must be a bool")
        if not isinstance(self.qdrant_rails_when_ch_empty, bool):
            raise ConfigurationError(f"{_p}.qdrant_rails_when_ch_empty must be a bool")
        if float(self.qdrant_rails_timeout_seconds) <= 0.0:
            raise ConfigurationError(f"{_p}.qdrant_rails_timeout_seconds must be > 0")
        if not isinstance(self.consume_search_explore_prewarm, bool):
            raise ConfigurationError(f"{_p}.consume_search_explore_prewarm must be a bool")
        if float(self.search_explore_prewarm_wait_seconds) <= 0.0:
            raise ConfigurationError(f"{_p}.search_explore_prewarm_wait_seconds must be > 0")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TimeoutFallbackConfig':
        _p = "general.search.timeout_fallback"
        _require(
            d,
            [
                'rail_first',
                'qdrant_rails_when_ch_empty',
                'qdrant_rails_timeout_seconds',
                'consume_search_explore_prewarm',
                'search_explore_prewarm_wait_seconds',
            ],
            _p,
        )
        return cls(
            rail_first=bool(d['rail_first']),
            qdrant_rails_when_ch_empty=bool(d['qdrant_rails_when_ch_empty']),
            qdrant_rails_timeout_seconds=float(d['qdrant_rails_timeout_seconds']),
            consume_search_explore_prewarm=bool(d['consume_search_explore_prewarm']),
            search_explore_prewarm_wait_seconds=float(d['search_explore_prewarm_wait_seconds']),
        )


@dataclass
class QieL0FilterCacheConfig:
    """In-process LRU+TTL cache for qie_only L0 filter extracts.

    All fields required in YAML — no in-code defaults.
    """
    max_entries: int
    ttl_seconds: int

    def __post_init__(self) -> None:
        if int(self.max_entries) < 1:
            raise ConfigurationError("general.search.qie_l0_filter_cache.max_entries must be >= 1")
        if int(self.ttl_seconds) < 1:
            raise ConfigurationError("general.search.qie_l0_filter_cache.ttl_seconds must be >= 1")
        self.max_entries = int(self.max_entries)
        self.ttl_seconds = int(self.ttl_seconds)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QieL0FilterCacheConfig':
        _require(d, ['max_entries', 'ttl_seconds'], 'general.search.qie_l0_filter_cache')
        return cls(max_entries=int(d['max_entries']), ttl_seconds=int(d['ttl_seconds']))


@dataclass
class FindWireConfig:
    """FIND auction/recommend wire policy for qie_only filters and keywords.

    All fields are required in YAML. Selects which L0 keyword terms become the
    FIND ``query`` string and whether FIND semantic search is enabled on that
    query. Keyword probability gate is NOT here — sole source is
    ``qi.l0_llm_entity.keyword_min_probability`` (passed by callers as a fraction).

    :param prefer_keywords_for_query: bool - When true, usable keyword terms
        become FIND ``query`` even when hard filters are present. When false
        and hard filters are present, FIND ``query`` is ``empty_query_fallback``.
    :param empty_query_fallback: str - FIND ``query`` when no usable keyword
        terms remain after probability and term-count gates.
    :param max_keyword_terms: int - Maximum terms kept after sorting by
        probability descending (>= 1).
    :param keyword_term_separator: str - Non-empty separator used to join
        selected keyword terms into FIND ``query``.
    :param set_use_semantic_search_when_keywords: bool - When selected keywords
        become FIND ``query``, also set the FIND semantic-search query param.
    :param use_semantic_search_param: str - FIND query-string key for semantic
        search (for example ``useSemanticSearch``).
    :param use_semantic_search_value: str - Value written when semantic search
        is enabled (for example ``true``).
    """

    prefer_keywords_for_query: bool
    empty_query_fallback: str
    max_keyword_terms: int
    keyword_term_separator: str
    set_use_semantic_search_when_keywords: bool
    use_semantic_search_param: str
    use_semantic_search_value: str

    def __post_init__(self) -> None:
        if not isinstance(self.prefer_keywords_for_query, bool):
            raise ConfigurationError(
                "general.search.find_wire.prefer_keywords_for_query must be a bool"
            )
        if not isinstance(self.empty_query_fallback, str) or not self.empty_query_fallback.strip():
            raise ConfigurationError(
                "general.search.find_wire.empty_query_fallback must be a non-empty string"
            )
        self.empty_query_fallback = self.empty_query_fallback.strip()
        if int(self.max_keyword_terms) < 1:
            raise ConfigurationError(
                "general.search.find_wire.max_keyword_terms must be >= 1"
            )
        self.max_keyword_terms = int(self.max_keyword_terms)
        if not isinstance(self.keyword_term_separator, str) or self.keyword_term_separator == "":
            raise ConfigurationError(
                "general.search.find_wire.keyword_term_separator must be a non-empty string"
            )
        if not isinstance(self.set_use_semantic_search_when_keywords, bool):
            raise ConfigurationError(
                "general.search.find_wire.set_use_semantic_search_when_keywords must be a bool"
            )
        if (
            not isinstance(self.use_semantic_search_param, str)
            or not self.use_semantic_search_param.strip()
        ):
            raise ConfigurationError(
                "general.search.find_wire.use_semantic_search_param must be a non-empty string"
            )
        self.use_semantic_search_param = self.use_semantic_search_param.strip()
        if (
            not isinstance(self.use_semantic_search_value, str)
            or not self.use_semantic_search_value.strip()
        ):
            raise ConfigurationError(
                "general.search.find_wire.use_semantic_search_value must be a non-empty string"
            )
        self.use_semantic_search_value = self.use_semantic_search_value.strip()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FindWireConfig':
        _require(
            d,
            [
                'prefer_keywords_for_query',
                'empty_query_fallback',
                'max_keyword_terms',
                'keyword_term_separator',
                'set_use_semantic_search_when_keywords',
                'use_semantic_search_param',
                'use_semantic_search_value',
            ],
            'general.search.find_wire',
        )
        return cls(
            prefer_keywords_for_query=bool(d['prefer_keywords_for_query']),
            empty_query_fallback=str(d['empty_query_fallback']),
            max_keyword_terms=int(d['max_keyword_terms']),
            keyword_term_separator=str(d['keyword_term_separator']),
            set_use_semantic_search_when_keywords=bool(
                d['set_use_semantic_search_when_keywords']
            ),
            use_semantic_search_param=str(d['use_semantic_search_param']),
            use_semantic_search_value=str(d['use_semantic_search_value']),
        )


@dataclass
class MoneyDisplayPairConfig:
    """Maps one money payload source key to FIND display target keys."""

    source: str
    targets: List[str]

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise ConfigurationError("general.search.find_listing.money_display_pairs[].source must be a non-empty string")
        if not isinstance(self.targets, list) or not self.targets:
            raise ConfigurationError("general.search.find_listing.money_display_pairs[].targets must be a non-empty list")
        for t in self.targets:
            if not isinstance(t, str) or not t.strip():
                raise ConfigurationError("general.search.find_listing.money_display_pairs[].targets entries must be non-empty strings")
        self.source = self.source.strip()
        self.targets = [str(t).strip() for t in self.targets]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'MoneyDisplayPairConfig':
        _require(d, ['source', 'targets'], 'general.search.find_listing.money_display_pairs[]')
        return cls(source=str(d['source']), targets=list(d['targets']))


@dataclass
class FindListingConfig:
    """FIND listing projection helpers for ranked_results (timestamps + money display)."""

    iso_timestamp_fields: List[str]
    iso_fallback_source: str
    iso_passthrough_string_fields: List[str]
    money_display_currency_prefix: str
    money_display_template: str
    money_display_value_placeholder: str
    money_display_number_format: str
    money_display_when_null: str
    money_display_pairs: List[MoneyDisplayPairConfig]
    domain_name_payload_keys: List[str]

    def __post_init__(self) -> None:
        if not isinstance(self.iso_timestamp_fields, list) or not self.iso_timestamp_fields:
            raise ConfigurationError("general.search.find_listing.iso_timestamp_fields must be a non-empty list")
        for f in self.iso_timestamp_fields:
            if not isinstance(f, str) or not f.strip():
                raise ConfigurationError("general.search.find_listing.iso_timestamp_fields entries must be non-empty strings")
        self.iso_timestamp_fields = [str(f).strip() for f in self.iso_timestamp_fields]
        if not isinstance(self.iso_fallback_source, str) or not self.iso_fallback_source.strip():
            raise ConfigurationError("general.search.find_listing.iso_fallback_source must be a non-empty string")
        self.iso_fallback_source = self.iso_fallback_source.strip()
        if not isinstance(self.iso_passthrough_string_fields, list):
            raise ConfigurationError("general.search.find_listing.iso_passthrough_string_fields must be a list")
        for f in self.iso_passthrough_string_fields:
            if not isinstance(f, str) or not f.strip():
                raise ConfigurationError("general.search.find_listing.iso_passthrough_string_fields entries must be non-empty strings")
        self.iso_passthrough_string_fields = [str(f).strip() for f in self.iso_passthrough_string_fields]
        if not isinstance(self.money_display_currency_prefix, str):
            raise ConfigurationError("general.search.find_listing.money_display_currency_prefix must be a string")
        if not isinstance(self.money_display_value_placeholder, str) or not self.money_display_value_placeholder:
            raise ConfigurationError("general.search.find_listing.money_display_value_placeholder must be a non-empty string")
        if not isinstance(self.money_display_template, str) or self.money_display_value_placeholder not in self.money_display_template:
            raise ConfigurationError(
                "general.search.find_listing.money_display_template must be a string containing money_display_value_placeholder"
            )
        if not isinstance(self.money_display_number_format, str) or not self.money_display_number_format.strip():
            raise ConfigurationError("general.search.find_listing.money_display_number_format must be a non-empty string")
        if not isinstance(self.money_display_when_null, str):
            raise ConfigurationError("general.search.find_listing.money_display_when_null must be a string")
        if not isinstance(self.money_display_pairs, list) or not self.money_display_pairs:
            raise ConfigurationError("general.search.find_listing.money_display_pairs must be a non-empty list")
        for p in self.money_display_pairs:
            if not isinstance(p, MoneyDisplayPairConfig):
                raise ConfigurationError("general.search.find_listing.money_display_pairs entries must be MoneyDisplayPairConfig")
        if not isinstance(self.domain_name_payload_keys, list) or not self.domain_name_payload_keys:
            raise ConfigurationError("general.search.find_listing.domain_name_payload_keys must be a non-empty list")
        for f in self.domain_name_payload_keys:
            if not isinstance(f, str) or not f.strip():
                raise ConfigurationError("general.search.find_listing.domain_name_payload_keys entries must be non-empty strings")
        self.domain_name_payload_keys = [str(f).strip() for f in self.domain_name_payload_keys]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FindListingConfig':
        _require(
            d,
            [
                'iso_timestamp_fields',
                'iso_fallback_source',
                'iso_passthrough_string_fields',
                'money_display_currency_prefix',
                'money_display_template',
                'money_display_value_placeholder',
                'money_display_number_format',
                'money_display_when_null',
                'money_display_pairs',
                'domain_name_payload_keys',
            ],
            'general.search.find_listing',
        )
        pairs = [MoneyDisplayPairConfig.from_dict(p) for p in d['money_display_pairs']]
        return cls(
            iso_timestamp_fields=list(d['iso_timestamp_fields']),
            iso_fallback_source=str(d['iso_fallback_source']),
            iso_passthrough_string_fields=list(d['iso_passthrough_string_fields']),
            money_display_currency_prefix=str(d['money_display_currency_prefix']),
            money_display_template=str(d['money_display_template']),
            money_display_value_placeholder=str(d['money_display_value_placeholder']),
            money_display_number_format=str(d['money_display_number_format']).strip(),
            money_display_when_null=str(d['money_display_when_null']),
            money_display_pairs=pairs,
            domain_name_payload_keys=list(d['domain_name_payload_keys']),
        )


@dataclass
class SearchParamsConfig:
    """Search endpoint execution parameters.
    :param top_k_cap: int - Hard ceiling on caller top_k; request values above this are clamped
    :param diversity_lambda: float - MMR diversity weight in [0,1]; higher = more diverse results
    :param search_timeout_seconds: float - Hard wall-clock SLA for orchestrator.search(); returns
        explore fallback with failure_mode='timeout' on breach instead of hanging indefinitely
    :param analytics_timeout_seconds: float - Hard cap for the inline analytics call; capped
        independently because analytics (LLM gen + CH exec) is more expensive than search
    :param analytics_total_budget_seconds: float - Hard ceiling on TOTAL analytics wall-clock
        measured from request start (classification + routing + analytics exec). The inline
        analytics call receives the remaining budget after the upstream search/classification
        phase, so the end-to-end analytics response never exceeds this value. Must be
        >= analytics_timeout_seconds.
    :param explore_fallback_timeout_seconds: float - Internal cap for the speculative explore
        fallback task; must be < search_timeout_seconds so results are ready before the SLA fires
    :param analytics_failure_explore_fallback: bool - When analytics fails and hybrid ranked_results
        is empty, surface explore rails/semantic as a last fill for ranked_results
    :param ranked_results_complement: RankedResultsComplementConfig - Hybrid-first ranks; CH
        substrates complement only
    :param timeout_fallback: TimeoutFallbackConfig - Rail-first + Qdrant ladder on SLA paths
    :param result_fields: List[str] - Ordered payload field names projected into each
        ranked_results entry, on top of the fixed envelope (rank, domain_name,
        coherence_score, matched_by).
    :param find_listing: FindListingConfig - ISO timestamp fields and money display formatting
    :param find_wire: FindWireConfig - qie_only FIND auction/recommend wire policy
        for keyword query selection and semantic-search flag
    :param explore_fallback_semantic_prefer_min_results: int - Minimum number of semantic
        results required to prefer semantic-only output over RRF fusion with explore rails
        in get_timeout_fallback. Set to 0 to disable semantic preference (always RRF-fuse).
    :param explore_fallback_semantic_prefer_min_score: float - Minimum average fused_score
        across the top explore_fallback_semantic_prefer_min_results semantic items to trigger
        semantic-only output. Must be in [0, 1].
    """
    top_k_cap: int
    diversity_lambda: float
    search_timeout_seconds: float
    analytics_timeout_seconds: float
    analytics_total_budget_seconds: float
    explore_fallback_timeout_seconds: float
    speculative_analytics_start: bool
    speculative_analytics_l1_confidence_threshold: float
    analytics_failure_explore_fallback: bool
    guidance_analytics_crosstype_enabled: bool
    guidance_analytics_crosstype_timeout_seconds: float
    result_fields: List[str]
    find_listing: FindListingConfig
    find_wire: FindWireConfig
    ranked_results_complement: RankedResultsComplementConfig
    timeout_fallback: TimeoutFallbackConfig
    explore_fallback_cache_ttl_seconds: float
    explore_fallback_semantic_prefer_min_results: int
    explore_fallback_semantic_prefer_min_score: float
    analytics_skip_uncorroborated_detour: bool
    analytics_budget_keywords: List[str]
    permanently_unavailable_columns: List[str]
    qie_only_mode: bool
    qie_l0_filter_cache: 'QieL0FilterCacheConfig'
    hybrid_prewarm_enabled: bool
    overlap_preprocess_with_classify: bool
    auction_tiebreak: 'AuctionTiebreakConfig'
    analytics_unavailable_hybrid_notice: str
    analytics_unavailable_explore_notice: str
    analytics_timeout_hybrid_notice: str
    analytics_timeout_explore_notice: str
    analytics_detour_skip_notice: str
    analytics_connection_unavailable_notice: str
    analytics_failure_with_explore_notice: str
    analytics_failure_empty_explore_notice: str

    def __post_init__(self) -> None:
        if self.top_k_cap < 1:
            raise ConfigurationError("general.search.top_k_cap must be >= 1")
        if not 0.0 <= self.diversity_lambda <= 1.0:
            raise ConfigurationError("general.search.diversity_lambda must be in [0, 1]")
        if float(self.search_timeout_seconds) <= 0.0:
            raise ConfigurationError("general.search.search_timeout_seconds must be > 0")
        if float(self.analytics_timeout_seconds) <= 0.0:
            raise ConfigurationError("general.search.analytics_timeout_seconds must be > 0")
        if float(self.analytics_total_budget_seconds) <= 0.0:
            raise ConfigurationError("general.search.analytics_total_budget_seconds must be > 0")
        if float(self.analytics_total_budget_seconds) < float(self.analytics_timeout_seconds):
            raise ConfigurationError("general.search.analytics_total_budget_seconds must be >= analytics_timeout_seconds")
        if float(self.explore_fallback_timeout_seconds) <= 0.0:
            raise ConfigurationError("general.search.explore_fallback_timeout_seconds must be > 0")
        if not 0.0 <= float(self.speculative_analytics_l1_confidence_threshold) <= 1.0:
            raise ConfigurationError("general.search.speculative_analytics_l1_confidence_threshold must be in [0, 1]")
        if not isinstance(self.analytics_failure_explore_fallback, bool):
            raise ConfigurationError("general.search.analytics_failure_explore_fallback must be a bool")
        if not isinstance(self.ranked_results_complement, RankedResultsComplementConfig):
            raise ConfigurationError("general.search.ranked_results_complement must be a RankedResultsComplementConfig")
        if not isinstance(self.timeout_fallback, TimeoutFallbackConfig):
            raise ConfigurationError("general.search.timeout_fallback must be a TimeoutFallbackConfig")
        if not isinstance(self.qie_only_mode, bool):
            raise ConfigurationError("general.search.qie_only_mode must be a bool")
        if not isinstance(self.qie_l0_filter_cache, QieL0FilterCacheConfig):
            raise ConfigurationError(
                "general.search.qie_l0_filter_cache must be a QieL0FilterCacheConfig"
            )
        if not isinstance(self.hybrid_prewarm_enabled, bool):
            raise ConfigurationError("general.search.hybrid_prewarm_enabled must be a bool")
        if not isinstance(self.overlap_preprocess_with_classify, bool):
            raise ConfigurationError("general.search.overlap_preprocess_with_classify must be a bool")
        if float(self.explore_fallback_cache_ttl_seconds) <= 0.0:
            raise ConfigurationError("general.search.explore_fallback_cache_ttl_seconds must be > 0")
        if int(self.explore_fallback_semantic_prefer_min_results) < 0:
            raise ConfigurationError("general.search.explore_fallback_semantic_prefer_min_results must be >= 0")
        if not 0.0 <= float(self.explore_fallback_semantic_prefer_min_score) <= 1.0:
            raise ConfigurationError("general.search.explore_fallback_semantic_prefer_min_score must be in [0, 1]")
        if not isinstance(self.guidance_analytics_crosstype_enabled, bool):
            raise ConfigurationError("general.search.guidance_analytics_crosstype_enabled must be a bool")
        if float(self.guidance_analytics_crosstype_timeout_seconds) <= 0.0:
            raise ConfigurationError("general.search.guidance_analytics_crosstype_timeout_seconds must be > 0")
        if not isinstance(self.result_fields, list) or len(self.result_fields) == 0:
            raise ConfigurationError("general.search.result_fields must be a non-empty list of payload field names")
        for f in self.result_fields:
            if not isinstance(f, str) or not f.strip():
                raise ConfigurationError("general.search.result_fields entries must be non-empty strings")
        if not isinstance(self.find_listing, FindListingConfig):
            raise ConfigurationError("general.search.find_listing must be a FindListingConfig")
        if not isinstance(self.find_wire, FindWireConfig):
            raise ConfigurationError("general.search.find_wire must be a FindWireConfig")
        if not isinstance(self.analytics_skip_uncorroborated_detour, bool):
            raise ConfigurationError("general.search.analytics_skip_uncorroborated_detour must be a bool")
        if not isinstance(self.analytics_budget_keywords, list):
            raise ConfigurationError("general.search.analytics_budget_keywords must be a list of strings")
        for kw in self.analytics_budget_keywords:
            if not isinstance(kw, str) or not kw.strip():
                raise ConfigurationError("general.search.analytics_budget_keywords entries must be non-empty strings")
        if not isinstance(self.permanently_unavailable_columns, list):
            raise ConfigurationError("general.search.permanently_unavailable_columns must be a list of strings")
        for col in self.permanently_unavailable_columns:
            if not isinstance(col, str) or not col.strip():
                raise ConfigurationError("general.search.permanently_unavailable_columns entries must be non-empty strings")
        for _notice_name in (
            'analytics_unavailable_hybrid_notice',
            'analytics_unavailable_explore_notice',
            'analytics_timeout_hybrid_notice',
            'analytics_timeout_explore_notice',
            'analytics_detour_skip_notice',
            'analytics_connection_unavailable_notice',
            'analytics_failure_with_explore_notice',
            'analytics_failure_empty_explore_notice',
        ):
            _notice_val = getattr(self, _notice_name)
            if not isinstance(_notice_val, str) or not _notice_val.strip():
                raise ConfigurationError(f"general.search.{_notice_name} must be a non-empty string")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SearchParamsConfig':
        """Build from config dict.
        :param d: Dict[str, Any] - general.search section
        :return: SearchParamsConfig
        """
        _require(d, ['top_k_cap', 'diversity_lambda', 'search_timeout_seconds', 'analytics_timeout_seconds',
                     'analytics_total_budget_seconds', 'explore_fallback_timeout_seconds',
                     'speculative_analytics_start',
                     'speculative_analytics_l1_confidence_threshold',
                     'analytics_failure_explore_fallback',
                     'guidance_analytics_crosstype_enabled',
                     'guidance_analytics_crosstype_timeout_seconds',
                     'result_fields',
                     'find_listing',
                     'find_wire',
                     'ranked_results_complement',
                     'timeout_fallback',
                     'explore_fallback_cache_ttl_seconds',
                     'explore_fallback_semantic_prefer_min_results',
                     'explore_fallback_semantic_prefer_min_score',
                     'analytics_skip_uncorroborated_detour',
                     'analytics_budget_keywords',
                     'permanently_unavailable_columns',
                     'qie_only_mode',
                     'qie_l0_filter_cache',
                     'hybrid_prewarm_enabled',
                     'overlap_preprocess_with_classify',
                     'auction_tiebreak',
                     'analytics_unavailable_hybrid_notice',
                     'analytics_unavailable_explore_notice',
                     'analytics_timeout_hybrid_notice',
                     'analytics_timeout_explore_notice',
                     'analytics_detour_skip_notice',
                     'analytics_connection_unavailable_notice',
                     'analytics_failure_with_explore_notice',
                     'analytics_failure_empty_explore_notice'], 'general.search')
        return cls(
            top_k_cap=int(d['top_k_cap']),
            diversity_lambda=float(d['diversity_lambda']),
            search_timeout_seconds=float(d['search_timeout_seconds']),
            analytics_timeout_seconds=float(d['analytics_timeout_seconds']),
            analytics_total_budget_seconds=float(d['analytics_total_budget_seconds']),
            explore_fallback_timeout_seconds=float(d['explore_fallback_timeout_seconds']),
            speculative_analytics_start=bool(d['speculative_analytics_start']),
            speculative_analytics_l1_confidence_threshold=float(d['speculative_analytics_l1_confidence_threshold']),
            analytics_failure_explore_fallback=bool(d['analytics_failure_explore_fallback']),
            guidance_analytics_crosstype_enabled=bool(d['guidance_analytics_crosstype_enabled']),
            guidance_analytics_crosstype_timeout_seconds=float(d['guidance_analytics_crosstype_timeout_seconds']),
            result_fields=[str(f).strip() for f in d['result_fields']],
            find_listing=FindListingConfig.from_dict(d['find_listing']),
            find_wire=FindWireConfig.from_dict(d['find_wire']),
            ranked_results_complement=RankedResultsComplementConfig.from_dict(d['ranked_results_complement']),
            timeout_fallback=TimeoutFallbackConfig.from_dict(d['timeout_fallback']),
            explore_fallback_cache_ttl_seconds=float(d['explore_fallback_cache_ttl_seconds']),
            explore_fallback_semantic_prefer_min_results=int(d['explore_fallback_semantic_prefer_min_results']),
            explore_fallback_semantic_prefer_min_score=float(d['explore_fallback_semantic_prefer_min_score']),
            analytics_skip_uncorroborated_detour=bool(d['analytics_skip_uncorroborated_detour']),
            analytics_budget_keywords=[str(kw) for kw in d['analytics_budget_keywords']],
            permanently_unavailable_columns=[str(c) for c in d['permanently_unavailable_columns']],
            qie_only_mode=bool(d['qie_only_mode']),
            qie_l0_filter_cache=QieL0FilterCacheConfig.from_dict(d['qie_l0_filter_cache']),
            hybrid_prewarm_enabled=bool(d['hybrid_prewarm_enabled']),
            overlap_preprocess_with_classify=bool(d['overlap_preprocess_with_classify']),
            auction_tiebreak=AuctionTiebreakConfig.from_dict(d['auction_tiebreak']),
            analytics_unavailable_hybrid_notice=str(d['analytics_unavailable_hybrid_notice']),
            analytics_unavailable_explore_notice=str(d['analytics_unavailable_explore_notice']),
            analytics_timeout_hybrid_notice=str(d['analytics_timeout_hybrid_notice']),
            analytics_timeout_explore_notice=str(d['analytics_timeout_explore_notice']),
            analytics_detour_skip_notice=str(d['analytics_detour_skip_notice']),
            analytics_connection_unavailable_notice=str(d['analytics_connection_unavailable_notice']),
            analytics_failure_with_explore_notice=str(d['analytics_failure_with_explore_notice']),
            analytics_failure_empty_explore_notice=str(d['analytics_failure_empty_explore_notice']),
        )


def _missing_search_config() -> 'SearchParamsConfig':
    raise ConfigurationError("general.search block is required in config but was not found")


@dataclass
class GeneralConfig:
    """Top-level service config.
    :param service_name: str - Service identifier (logs / signals)
    :param startup_log_verbosity: str - compact | detailed (boot-time log density)
    :param max_query_length: int - Maximum accepted query length
    :param max_results: int - Maximum results returned by /search
    :param rate_limit: RateLimitConfig - Per-session rate limit (Layer 0 ingress)
    :param search: SearchParamsConfig - Search endpoint execution parameters
    :param log_level: str - DEBUG | INFO | WARNING | ERROR | CRITICAL (runtime logger threshold; LOG_LEVEL env var overrides)
    """
    service_name: str
    startup_log_verbosity: str
    max_query_length: int
    max_results: int
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    analytics_rate_limit: AnalyticsRateLimitConfig = field(default_factory=AnalyticsRateLimitConfig)
    search: SearchParamsConfig = field(default_factory=_missing_search_config)
    log_level: str = 'INFO'

    def __post_init__(self) -> None:
        if not self.service_name:
            raise ConfigurationError("general.service_name must be non-empty")
        if self.startup_log_verbosity not in _STARTUP_LOG_VERBOSITY:
            raise ConfigurationError(f"general.startup_log_verbosity must be one of {sorted(_STARTUP_LOG_VERBOSITY)}; got {self.startup_log_verbosity!r}")
        if self.max_query_length < 1:
            raise ConfigurationError("general.max_query_length must be >= 1")
        if self.max_results < 1:
            raise ConfigurationError("general.max_results must be >= 1")
        if self.log_level not in _VALID_LOG_LEVELS:
            raise ConfigurationError(f"general.log_level must be one of {sorted(_VALID_LOG_LEVELS)}; got {self.log_level!r}")

    @property
    def startup_log_detail(self) -> bool:
        """True when boot logs should emit per-tier/per-stage INFO (not compact)."""
        return self.startup_log_verbosity == 'detailed'

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'GeneralConfig':
        """Build from config dict.
        :param d: Dict[str, Any] - general section
        :return: GeneralConfig
        """
        _require(d, ['service_name', 'startup_log_verbosity', 'max_query_length', 'max_results', 'search'], 'general')
        return cls(
            service_name=str(d['service_name']),
            startup_log_verbosity=str(d['startup_log_verbosity']).strip(),
            max_query_length=int(d['max_query_length']),
            max_results=int(d['max_results']),
            rate_limit=RateLimitConfig.from_dict(d.get('rate_limit') or {}),
            analytics_rate_limit=AnalyticsRateLimitConfig.from_dict(d.get('analytics_rate_limit') or {}),
            search=SearchParamsConfig.from_dict(d['search']),
            log_level=str(d.get('log_level', 'INFO')).strip().upper(),
        )


_ENCODER_BACKENDS = frozenset({'fastembed', 'hashing'})
_FASTEMBED_ALLOWED_DIMS = frozenset({64, 128, 256, 384, 768, 1024})
_CASCADE_STAGE_KEYS = frozenset({'router', 'shortlist', 'rerank'})


@dataclass
class CascadeStageDimsConfig:
    """Per-stage dim selections for the Matryoshka cascade.

    One vector served at three lengths picked per stage: ``Tier-2 router 128`` /
    ``Listing shortlist 256`` / ``Re-rank head 384``. This config block names
    the dim each consumer should ask for via ``MatryoshkaCascadeEncoder.encode_at_dim``.

    All three stages are optional individually so test deployments can
    pin only the stages they exercise; an unset stage means
    "consumer keeps using the base encoder at its single configured
    dim". Production sets all three.

    :param router: Optional[int] - Dim for the Tier-2 (L1 semantic
        router) consumer. When set, MUST equal ``qi.semantic.embedding_dim``
        (validated at registry boot) AND be a member of
        ``CascadeEncoderConfig.supported_dims``.
    :param shortlist: Optional[int] - Dim for the listing-shortlist
        consumer (``VectorRetriever`` / ``QdrantHybridRetriever``).
        When set, MUST equal ``retrieval.vector.embedding_dim``
        (validated at registry boot) AND be a member of
        ``CascadeEncoderConfig.supported_dims``.
    :param rerank: Optional[int] - Dim for a future model-backed
        cross-encoder rerank head. The registry exposes
        ``Subsystems.rerank_encoder`` only when this is set; today's
        lexical reranker does NOT consume it. Provided for forward
        wiring under the same cascade contract.
    """
    router: Optional[int] = None
    shortlist: Optional[int] = None
    rerank: Optional[int] = None

    def __post_init__(self) -> None:
        for name, value in (('router', self.router), ('shortlist', self.shortlist), ('rerank', self.rerank)):
            if value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ConfigurationError( f"qi.encoder.cascade.stage_dims.{name} must be int >= 1; got {value!r}" )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CascadeStageDimsConfig':
        if not isinstance(d, dict):
            raise ConfigurationError("qi.encoder.cascade.stage_dims must be a mapping when present")
        unknown = set(d.keys()) - _CASCADE_STAGE_KEYS
        if unknown:
            raise ConfigurationError( f"qi.encoder.cascade.stage_dims unknown keys={sorted(unknown)}; " f"allowed={sorted(_CASCADE_STAGE_KEYS)}" )
        return cls(
            router=int(d['router']) if 'router' in d and d['router'] is not None else None,
            shortlist=int(d['shortlist']) if 'shortlist' in d and d['shortlist'] is not None else None,
            rerank=int(d['rerank']) if 'rerank' in d and d['rerank'] is not None else None,
        )


@dataclass
class CascadeEncoderConfig:
    """Matryoshka embedding-cascade wrapper config.

    One underlying inference, three serving lengths (typically 128 router /
    256 shortlist / 384 rerank). The
    cascade wraps a base ``Encoder`` and exposes ``encode_at_dim``;
    consumers that opt in (semantic router, offline indexer's per-stage
    re-encode, future stage-aware retriever) inject the wrapper
    explicitly.

    :param enabled: bool - Master switch. ``False`` skips cascade
        construction entirely; the registry exposes ``None`` and
        downstream consumers fall back to the single-dim base encoder.
    :param supported_dims: List[int] - Allowed dims for
        ``encode_at_dim``. Each MUST be ``>= 1``; non-empty. Typical:
        ``[128, 256, 384]`` matching the Matryoshka cascade.
    :param require_native_cascade: bool - When ``True`` the registry
        refuses to construct the cascade unless the base encoder
        supports a native truncation source (i.e. ``FastEmbedEncoder``).
        Use ``True`` in production; ``False`` in test mode lets the
        cascade collapse onto a ``HashingEncoder`` for unit tests.
    :param stage_dims: Optional[CascadeStageDimsConfig] - Per-stage dim
        selections (``router`` / ``shortlist`` / ``rerank``). When
        absent or all three sub-fields are ``None``, the registry does
        NOT build per-stage encoders and every consumer keeps using the
        base encoder at its single configured dim. When present, every
        non-None stage dim MUST be a member of ``supported_dims`` and
        MUST equal the corresponding consumer's configured
        ``embedding_dim`` (registry validates and raises
        ``ConfigurationError`` on mismatch).
    """
    enabled: bool
    supported_dims: List[int]
    require_native_cascade: bool
    stage_dims: Optional[CascadeStageDimsConfig] = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("qi.encoder.cascade.enabled must be a bool")
        if not isinstance(self.supported_dims, list) or not self.supported_dims:
            raise ConfigurationError("qi.encoder.cascade.supported_dims must be a non-empty list")
        normalised: List[int] = []
        for d in self.supported_dims:
            if not isinstance(d, int) or d < 1:
                raise ConfigurationError( "qi.encoder.cascade.supported_dims entries must be int >= 1" )
            if d not in normalised:
                normalised.append(int(d))
        self.supported_dims = normalised
        if not isinstance(self.require_native_cascade, bool):
            raise ConfigurationError("qi.encoder.cascade.require_native_cascade must be a bool")
        if self.stage_dims is not None and not isinstance(self.stage_dims, CascadeStageDimsConfig):
            raise ConfigurationError( "qi.encoder.cascade.stage_dims must be a CascadeStageDimsConfig or None" )
        if self.stage_dims is not None:
            allowed = frozenset(self.supported_dims)
            for name, value in ( ('router', self.stage_dims.router), ('shortlist', self.stage_dims.shortlist), ('rerank', self.stage_dims.rerank), ):
                if value is not None and value not in allowed:
                    raise ConfigurationError( f"qi.encoder.cascade.stage_dims.{name}={value} not in supported_dims={sorted(allowed)}" )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CascadeEncoderConfig':
        _require(d, ['enabled', 'supported_dims', 'require_native_cascade'], 'qi.encoder.cascade')
        stage_dims_raw = d.get('stage_dims')
        stage_dims = ( CascadeStageDimsConfig.from_dict(stage_dims_raw) if isinstance(stage_dims_raw, dict) else None )
        return cls( enabled=bool(d['enabled']), supported_dims=list(d['supported_dims']), require_native_cascade=bool(d['require_native_cascade']), stage_dims=stage_dims, )


@dataclass
class BatchingEncoderConfig:
    """Config for the BatchingEncoder async coalescing wrapper.

    :param enabled: bool - When True the registry wraps the base encoder.
    :param window_ms: float - Batching window in milliseconds (must be > 0).
    """
    enabled: bool
    window_ms: float

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("qi.encoder.batching.enabled must be bool")
        if self.window_ms <= 0:
            raise ConfigurationError("qi.encoder.batching.window_ms must be > 0")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'BatchingEncoderConfig':
        _require(d, ['enabled', 'window_ms'], 'qi.encoder.batching')
        return cls( enabled=bool(d['enabled']), window_ms=float(d['window_ms']), )


@dataclass
class QIEncoderConfig:
    """Shared text-encoder config.

    The encoder is shared across the QI semantic router, the in-memory vector
    index, and the search-side + analytics-side semantic caches. Selecting the
    backend is a one-line composition-root swap; consumers always receive the
    same `qi.encoder.Encoder` protocol surface.

    :param backend: str - One of `_ENCODER_BACKENDS`. `'fastembed'` is the
        production encoder (FastEmbed/ONNX-Runtime). `'hashing'` is the
        deterministic test-only backend (`HashingEncoder`).
    :param model_name: str - HuggingFace model id when `backend == 'fastembed'`.
        Ignored for `'hashing'`.
    :param dim: int - Output vector dimension. For `'fastembed'` must be one
        of `_FASTEMBED_ALLOWED_DIMS` (Matryoshka steps). For `'hashing'` must
        be `>= 4`. Must equal `qi.semantic.embedding_dim` and
        `retrieval.vector.embedding_dim` (validated downstream by consumers).
    :param local_model_path: str - Absolute path to the pre-downloaded model
        directory. Must be set when ``backend='fastembed'``. No network
        download is attempted.
    :param cache_dir: str - FastEmbed internal scratch directory. Set to empty.
    :param threads: int - ONNX-Runtime intra-op thread count; `0` = library default.
    :param max_length: int - Token-truncation length for input text.
    :param batch_size: int - Internal FastEmbed batch size for `encode_batch`.
    :param query_prefix: str - Task-type prefix prepended to every input.
        Use ``"search_query: "`` at query/routing time and
        ``"search_document: "`` during vectorization runs. Empty = no prefix.
    :param cascade: Optional[CascadeEncoderConfig] - Matryoshka cascade wrapper.
    :param batching: Optional[BatchingEncoderConfig] - Async coalescing wrapper.
    """
    backend: str
    model_name: str
    dim: int
    local_model_path: str
    cache_dir: str
    threads: int
    max_length: int
    batch_size: int
    query_prefix: str = ""
    cascade: Optional[CascadeEncoderConfig] = None
    batching: Optional[BatchingEncoderConfig] = None

    def __post_init__(self) -> None:
        if self.backend not in _ENCODER_BACKENDS:
            raise ConfigurationError( f"qi.encoder.backend must be one of {sorted(_ENCODER_BACKENDS)}; got '{self.backend}'" )
        if not isinstance(self.model_name, str):
            raise ConfigurationError("qi.encoder.model_name must be a string")
        if self.backend == 'fastembed':
            if not self.model_name:
                raise ConfigurationError("qi.encoder.model_name must be non-empty when backend='fastembed'")
            if self.dim not in _FASTEMBED_ALLOWED_DIMS:
                raise ConfigurationError( f"qi.encoder.dim must be one of {sorted(_FASTEMBED_ALLOWED_DIMS)} when backend='fastembed'; got {self.dim}" )
        else:  # 'hashing'
            if self.dim < 4:
                raise ConfigurationError("qi.encoder.dim must be >= 4 when backend='hashing'")
        if not isinstance(self.local_model_path, str):
            raise ConfigurationError("qi.encoder.local_model_path must be a string")
        if not isinstance(self.cache_dir, str):
            raise ConfigurationError("qi.encoder.cache_dir must be a string")
        if self.threads < 0:
            raise ConfigurationError("qi.encoder.threads must be >= 0 (0 = library default)")
        if self.max_length < 1:
            raise ConfigurationError("qi.encoder.max_length must be >= 1")
        if self.batch_size < 1:
            raise ConfigurationError("qi.encoder.batch_size must be >= 1")
        if not isinstance(self.query_prefix, str):
            raise ConfigurationError("qi.encoder.query_prefix must be a string")
        if self.cascade is not None and not isinstance(self.cascade, CascadeEncoderConfig):
            raise ConfigurationError("qi.encoder.cascade must be a CascadeEncoderConfig or None")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QIEncoderConfig':
        _require(d, ['backend', 'model_name', 'dim', 'local_model_path', 'threads', 'max_length', 'batch_size'], 'qi.encoder')
        cascade_raw = d.get('cascade')
        cascade = ( CascadeEncoderConfig.from_dict(cascade_raw) if isinstance(cascade_raw, dict) else None )
        batching_raw = d.get('batching')
        batching = ( BatchingEncoderConfig.from_dict(batching_raw) if isinstance(batching_raw, dict) else None )
        return cls(
            backend=str(d['backend']),
            model_name=str(d['model_name']),
            dim=int(d['dim']),
            local_model_path=str(d['local_model_path']),
            cache_dir=str(d.get('cache_dir', '')),
            threads=int(d['threads']),
            max_length=int(d['max_length']),
            batch_size=int(d['batch_size']),
            query_prefix=str(d.get('query_prefix', '')),
            cascade=cascade,
            batching=batching,
        )


@dataclass
class QIAggregationConfig:
    """Config for the structural aggregate-question intent gate.

    Fires when the query names a marketplace object AND carries an aggregate
    operator. Two operator tiers separate unambiguous aggregate vocabulary from
    popularity vocabulary that the explore surface shares.

    :param enabled: bool - Toggle. When False the gate is a no-op.
    :param marketplace_nouns: List[str] - Marketplace objects an analytics
        question is computed over (e.g. 'listing', 'auction', 'tld'). Matched as
        substrings of the lowercased query so singular and plural forms both hit.
    :param strong_operators: List[str] - Unambiguous aggregate vocabulary that
        alone confirms analytics intent alongside a marketplace noun (e.g.
        'count', 'average', 'distribution', 'correlat', 'how many').
    :param weak_operators: List[str] - Popularity vocabulary shared with the
        explore surface (e.g. 'top', 'most', 'trend'). Fires only when paired
        with a companion term.
    :param weak_operator_companions: List[str] - Metric or grouping terms that
        promote a weak operator to analytics intent (e.g. 'by ', 'per ',
        'volume', 'current bid').
    :param word_boundary_max_length: int - Operator and companion terms whose
        length is at most this value are matched on a word boundary instead of a
        raw substring, so short tokens do not match inside unrelated words.
    """
    enabled: bool
    marketplace_nouns: List[str]
    strong_operators: List[str]
    weak_operators: List[str]
    weak_operator_companions: List[str]
    word_boundary_max_length: int

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("qi.regex.aggregation_gate.enabled must be bool")
        for _field in ('marketplace_nouns', 'strong_operators', 'weak_operators', 'weak_operator_companions'):
            _val = getattr(self, _field)
            if not isinstance(_val, list) or not _val:
                raise ConfigurationError(f"qi.regex.aggregation_gate.{_field} must be a non-empty list")
            if not all(isinstance(t, str) and t for t in _val):
                raise ConfigurationError(f"qi.regex.aggregation_gate.{_field} must contain non-empty strings")
        if not isinstance(self.word_boundary_max_length, int) or isinstance(self.word_boundary_max_length, bool) or self.word_boundary_max_length < 1:
            raise ConfigurationError("qi.regex.aggregation_gate.word_boundary_max_length must be an int >= 1")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QIAggregationConfig':
        _require(
            d,
            ['enabled', 'marketplace_nouns', 'strong_operators', 'weak_operators', 'weak_operator_companions', 'word_boundary_max_length'],
            'qi.regex.aggregation_gate',
        )
        return cls(
            enabled=bool(d['enabled']),
            marketplace_nouns=[str(t).lower() for t in d['marketplace_nouns']],
            strong_operators=[str(t).lower() for t in d['strong_operators']],
            weak_operators=[str(t).lower() for t in d['weak_operators']],
            weak_operator_companions=[str(t).lower() for t in d['weak_operator_companions']],
            word_boundary_max_length=int(d['word_boundary_max_length']),
        )



@dataclass
class QINgramPreGateConfig:
    """Config for the n-gram log-odds pre-gate that replaces IntentPreGate phrase lists.

    :param enabled: Master toggle.
    :param model_path: Path to the serialized weights JSON file produced by NgramTrainer.
        Relative paths are resolved from the project root.
    :param confidence_threshold: Minimum cumulative log-odds sum for a class to fire.
        Increase to raise precision; decrease for higher recall.
    :param max_ngram_order: Maximum n-gram order to extract at inference time (1 = unigrams
        only, 2 = unigrams + bigrams). Must match what was used during training.
    :param emit_confidence: Confidence value (0, 1] assigned to the Vote cast by the
        ngram gate in the ensemble resolver. Separate from confidence_threshold (which
        is a log-odds decision boundary, not 0-1 scaled).
    """
    enabled: bool
    model_path: str
    confidence_threshold: float
    max_ngram_order: int
    emit_confidence: float = 0.95
    margin_threshold: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("qi.regex.ngram_pre_gate.enabled must be bool")
        if not isinstance(self.model_path, str) or not self.model_path.strip():
            raise ConfigurationError("qi.regex.ngram_pre_gate.model_path must be a non-empty string")
        if not isinstance(self.confidence_threshold, (int, float)) or float(self.confidence_threshold) <= 0:
            raise ConfigurationError("qi.regex.ngram_pre_gate.confidence_threshold must be a positive number")
        if not isinstance(self.max_ngram_order, int) or self.max_ngram_order not in (1, 2):
            raise ConfigurationError("qi.regex.ngram_pre_gate.max_ngram_order must be 1 or 2")
        if not isinstance(self.emit_confidence, (int, float)) or not 0.0 < float(self.emit_confidence) <= 1.0:
            raise ConfigurationError("qi.regex.ngram_pre_gate.emit_confidence must be in (0, 1]")
        if not isinstance(self.margin_threshold, (int, float)) or float(self.margin_threshold) < 0:
            raise ConfigurationError("qi.regex.ngram_pre_gate.margin_threshold must be >= 0")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QINgramPreGateConfig':
        _require(d, ['enabled', 'model_path', 'confidence_threshold', 'max_ngram_order'], 'qi.regex.ngram_pre_gate')
        return cls(
            enabled=bool(d['enabled']),
            model_path=str(d['model_path']),
            confidence_threshold=float(d['confidence_threshold']),
            max_ngram_order=int(d['max_ngram_order']),
            emit_confidence=float(d.get('emit_confidence', 0.95)),
            margin_threshold=float(d.get('margin_threshold', 0.0)),
        )


@dataclass
class QIRegexConfig:
    """L0 entity extractor config.

    :param tld_substitutions: Adapt-on-miss map (``ai ->[io, tech]``). When the
        entity grounder drops a TLD value because it is not present in the live
        inventory, the substitution map provides alternatives the search surface
        can offer as chips. Optional — empty dict disables the feature.
    :param live_inventory_ttl_seconds: TTL for the ``LiveInventoryContract`` cache
        of distinct TLDs scanned from the structured index. Set ``> 0`` to opt into
        live grounding; set ``0`` to keep the static-only path.
    """
    enabled: bool
    confidence: float
    known_tlds: List[str] = field(default_factory=list)
    known_auction_types: List[str] = field(default_factory=list)
    tld_substitutions: Dict[str, List[str]] = field(default_factory=dict)
    live_inventory_ttl_seconds: float = 0.0
    ch_tld_refresh_enabled: bool = False
    ch_tld_refresh_interval_seconds: float = 0.0
    ch_tld_lookback_days: int = 0
    ch_tld_query_sql: str = ""
    aggregation_gate: Optional['QIAggregationConfig'] = None
    ngram_pre_gate: Optional['QINgramPreGateConfig'] = None
    paired_direction_slots: List[List[str]] = field(default_factory=list)
    tld_context_match_max_chars: int = 5
    tld_word_bare_exclusions: List[str] = field(default_factory=list)
    suppress_for_query_types: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ConfigurationError("qi.regex.confidence must be in [0,1]")
        if not isinstance(self.known_tlds, list):
            raise ConfigurationError("qi.regex.known_tlds must be a list")
        if not isinstance(self.known_auction_types, list) or not self.known_auction_types:
            raise ConfigurationError("qi.regex.known_auction_types must be a non-empty list")
        if not isinstance(self.tld_substitutions, dict):
            raise ConfigurationError("qi.regex.tld_substitutions must be a dict[str, list[str]]")
        for k, v in self.tld_substitutions.items():
            if not isinstance(k, str) or not k:
                raise ConfigurationError("qi.regex.tld_substitutions keys must be non-empty strings")
            if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                raise ConfigurationError("qi.regex.tld_substitutions values must be list[str]")
        if not isinstance(self.live_inventory_ttl_seconds, (int, float)) or self.live_inventory_ttl_seconds < 0:
            raise ConfigurationError("qi.regex.live_inventory_ttl_seconds must be a non-negative number")
        if not isinstance(self.ch_tld_refresh_interval_seconds, (int, float)) or self.ch_tld_refresh_interval_seconds < 0:
            raise ConfigurationError("qi.regex.ch_tld_refresh_interval_seconds must be a non-negative number")
        if not isinstance(self.ch_tld_lookback_days, int) or self.ch_tld_lookback_days < 0:
            raise ConfigurationError("qi.regex.ch_tld_lookback_days must be a non-negative integer")
        if not isinstance(self.ch_tld_query_sql, str):
            raise ConfigurationError("qi.regex.ch_tld_query_sql must be a string")
        if not isinstance(self.tld_context_match_max_chars, int) or not (2 <= self.tld_context_match_max_chars <= 12):
            raise ConfigurationError("qi.regex.tld_context_match_max_chars must be an integer in [2, 12]")
        if not isinstance(self.tld_word_bare_exclusions, list) or not all(isinstance(w, str) and w for w in self.tld_word_bare_exclusions):
            raise ConfigurationError("qi.regex.tld_word_bare_exclusions must be a list of non-empty strings")
        if self.aggregation_gate is not None and not isinstance(self.aggregation_gate, QIAggregationConfig):
            raise ConfigurationError("qi.regex.aggregation_gate must be a QIAggregationConfig instance")
        if self.ngram_pre_gate is not None and not isinstance(self.ngram_pre_gate, QINgramPreGateConfig):
            raise ConfigurationError("qi.regex.ngram_pre_gate must be a QINgramPreGateConfig instance")
        if not isinstance(self.paired_direction_slots, list):
            raise ConfigurationError("qi.regex.paired_direction_slots must be a list")
        for _pair in self.paired_direction_slots:
            if not isinstance(_pair, list) or len(_pair) != 2 or not all(isinstance(s, str) and s for s in _pair):
                raise ConfigurationError("qi.regex.paired_direction_slots entries must be [slot_a, slot_b] string pairs")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QIRegexConfig':
        _require(d, ['enabled', 'confidence', 'known_tlds', 'known_auction_types'], 'qi.regex')
        agg_raw = d.get('aggregation_gate')
        ng_raw = d.get('ngram_pre_gate')
        return cls(
            enabled=bool(d['enabled']),
            confidence=float(d['confidence']),
            known_tlds=[str(t).lower() for t in d['known_tlds']],
            known_auction_types=[str(a).lower() for a in d['known_auction_types']],
            tld_substitutions={str(k).lower(): [str(x).lower() for x in v] for k, v in (d.get('tld_substitutions') or {}).items()},
            live_inventory_ttl_seconds=float(d.get('live_inventory_ttl_seconds', 0.0)),
            ch_tld_refresh_enabled=bool(d.get('ch_tld_refresh_enabled', False)),
            ch_tld_refresh_interval_seconds=float(d.get('ch_tld_refresh_interval_seconds', 0.0)),
            ch_tld_lookback_days=int(d.get('ch_tld_lookback_days', 0)),
            ch_tld_query_sql=str(d.get('ch_tld_query_sql', '')),
            tld_context_match_max_chars=int(d.get('tld_context_match_max_chars', 5)),
            tld_word_bare_exclusions=[str(w).lower() for w in d.get('tld_word_bare_exclusions', [])],
            aggregation_gate=QIAggregationConfig.from_dict(agg_raw) if isinstance(agg_raw, dict) else None,
            ngram_pre_gate=QINgramPreGateConfig.from_dict(ng_raw) if isinstance(ng_raw, dict) else None,
            paired_direction_slots=[[str(s) for s in pair] for pair in (d.get('paired_direction_slots') or []) if isinstance(pair, list)],
            suppress_for_query_types=[str(t).lower() for t in (d.get('suppress_for_query_types') or []) if isinstance(t, str) and t],
        )


@dataclass
class LearnedHeadConfig:
    """Learned-head dispatch config.

    Optional sub-config of ``QISemanticConfig``. ``kind`` is binary in effect:
    ``'centroid'`` (default) keeps the legacy K-means centroid scorer; ANY other
    value enables the learned head. The concrete head type is NEVER taken from
    this field — the router peeks the ``kind`` tag embedded in the ``.npz``
    artefact and loads the matching class (logistic / svm / gradient_boosting /
    deep). So a single ``kind: auto`` works for every trained artefact and survives
    a retrain that swaps the algorithm. Reversible by flipping ``kind`` back to
    ``'centroid'``.

    :param kind: str - One of ``{'centroid', 'auto', 'logistic', 'svm', 'gradient_boosting', 'deep'}``.
        ``'centroid'`` (default) disables the learned head. ``'auto'`` (recommended)
        enables it and auto-detects the artefact type at startup. The algorithm-named
        values are accepted for back-compat and behave identically to ``'auto'`` (the
        loader still dispatches by the artefact tag, not by this field); when one is
        set and disagrees with the artefact on disk the router logs a warning.
    :param model_path: str - Filesystem path to the ``.npz`` artefact
        produced by ``head_trainer``. Required when ``kind != 'centroid'``;
        ignored otherwise. Resolved relative to the package root by
        the loader (no traversal).
    :param min_seed_count: int - Minimum total positives + hard negatives
        the trainer must see before a head is fit. Guards against shipping
        a head trained on a corpus that has shrunk unexpectedly.
    """
    kind: str
    model_path: str
    min_seed_count: int
    strict: bool = False

    _ALLOWED_KINDS = ('centroid', 'auto', 'logistic', 'svm', 'gradient_boosting', 'deep')

    def __post_init__(self) -> None:
        if self.kind not in self._ALLOWED_KINDS:
            raise ConfigurationError( f"qi.semantic.learned_head.kind must be one of {self._ALLOWED_KINDS}; got {self.kind!r}" )
        if not isinstance(self.model_path, str):
            raise ConfigurationError("qi.semantic.learned_head.model_path must be a string")
        if self.kind != 'centroid' and not self.model_path:
            raise ConfigurationError( "qi.semantic.learned_head.model_path is required when kind != 'centroid'" )
        if int(self.min_seed_count) < 1:
            raise ConfigurationError("qi.semantic.learned_head.min_seed_count must be >= 1")
        if not isinstance(self.strict, bool):
            raise ConfigurationError("qi.semantic.learned_head.strict must be a bool")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'LearnedHeadConfig':
        _require(d, ['kind', 'model_path', 'min_seed_count'], 'qi.semantic.learned_head')
        return cls( kind=str(d['kind']), model_path=str(d['model_path']), min_seed_count=int(d['min_seed_count']), strict=bool(d.get('strict', False)), )


@dataclass
class QISemanticConfig:
    """L1 semantic router config.

    Cold-start fields:
    - `seeds_path` points to the curated seed YAML loaded by `RouterSeedLoader`.
      When empty, the router falls back to `archetype_prototypes` (test-only
      override path; production wiring requires `seeds_path`).
    - `min_seeds_per_archetype` enforces the floor (default 30 in YAML).
    - `centroid_exclusions` lists query_type archetypes that skip Tier-2 centroids
      (plan: analytics uses Tier-1 cues + Tier-3, not L1 similarity routing).
    - `learned_head` optionally swaps the centroid scorer for a
      trained multinomial-logistic head. ``None`` (default) keeps legacy
      behaviour exactly.
    """
    enabled: bool
    confidence_threshold: float
    embedding_dim: int
    encoder_seed: int
    archetype_prototypes: Dict[str, List[str]]
    seeds_path: str
    min_seeds_per_archetype: int
    centroid_exclusions: List[str]
    num_sub_centroids: int = 1
    learned_head: Optional[LearnedHeadConfig] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ConfigurationError("qi.semantic.confidence_threshold must be in [0,1]")
        if self.embedding_dim < 4:
            raise ConfigurationError("qi.semantic.embedding_dim must be >= 4")
        if not isinstance(self.archetype_prototypes, dict):
            raise ConfigurationError("qi.semantic.archetype_prototypes must be a dict (may be empty when seeds_path is set)")
        for archetype, samples in self.archetype_prototypes.items():
            if not isinstance(samples, list):
                raise ConfigurationError(f"qi.semantic.archetype_prototypes.{archetype} must be a list")
        if not isinstance(self.seeds_path, str):
            raise ConfigurationError("qi.semantic.seeds_path must be a string (may be empty when archetype_prototypes is populated)")
        if self.min_seeds_per_archetype < 1:
            raise ConfigurationError("qi.semantic.min_seeds_per_archetype must be >= 1")
        # Exactly one source must be populated. Both is ambiguous; neither is unusable.
        has_inline = bool(self.archetype_prototypes)
        has_path = bool(self.seeds_path)
        if has_inline == has_path:
            raise ConfigurationError("qi.semantic must set exactly one of {seeds_path, archetype_prototypes}")
        if not isinstance(self.centroid_exclusions, list):
            raise ConfigurationError("qi.semantic.centroid_exclusions must be a list")
        for ex in self.centroid_exclusions:
            if not isinstance(ex, str) or not ex:
                raise ConfigurationError("qi.semantic.centroid_exclusions entries must be non-empty strings")
        if self.num_sub_centroids < 1:
            raise ConfigurationError("qi.semantic.num_sub_centroids must be >= 1")
        if self.learned_head is not None and not isinstance(self.learned_head, LearnedHeadConfig):
            raise ConfigurationError("qi.semantic.learned_head must be a LearnedHeadConfig or None")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QISemanticConfig':
        _require( d, [ 'enabled', 'confidence_threshold', 'embedding_dim', 'encoder_seed', 'archetype_prototypes', 'seeds_path', 'min_seeds_per_archetype', 'centroid_exclusions', ], 'qi.semantic', )
        learned_head_raw = d.get('learned_head')
        learned_head: Optional[LearnedHeadConfig]
        if learned_head_raw is None:
            learned_head = None
        else:
            learned_head = LearnedHeadConfig.from_dict(learned_head_raw)
        return cls(
            enabled=bool(d['enabled']),
            confidence_threshold=float(d['confidence_threshold']),
            embedding_dim=int(d['embedding_dim']),
            encoder_seed=int(d['encoder_seed']),
            archetype_prototypes={str(k): [str(s) for s in v] for k, v in d['archetype_prototypes'].items()},
            seeds_path=str(d['seeds_path']),
            min_seeds_per_archetype=int(d['min_seeds_per_archetype']),
            centroid_exclusions=[str(x) for x in d['centroid_exclusions']],
            num_sub_centroids=int(d.get('num_sub_centroids', 1)),
            learned_head=learned_head,
        )


@dataclass
class QILLMConfig:
    """L2 LLM classifier config (query_type only — entities come from L0).

    :param tier_3_timeout_seconds: float - Hard wall-clock cap on a single
        Tier-3 LLM classify() call; on timeout routes to Tier 1.
        Implemented via ``asyncio.wait_for`` around the
        ``LLMCallRouter.call_structured`` invocation in
        ``LLMClassifier.classify_with_prompt``. On timeout the classifier
        raises ``LLMError`` so the engine's existing fallback path activates,
        AND emits a ``llm_timeout`` ``FeedbackSignal`` (when a SignalStore
        is wired into the classifier) so dashboards can attribute timeouts. Must be > 0.
    :param alternative_band_high: float - Upper edge of the confidence band
        that triggers ``alternative_interpretations`` from the Tier-3 model.
        Must be in (0, 1].
    :param max_alternative_interpretations: int - Hard cap on alternatives
        emitted per call (pre-dedupe). Must be >= 0.
    :param keyword_expansion_max_terms: int - Cap injected into the L2 system prompt
        for verbatim OR-term lists. Must be >= 1.
    :param max_concurrent_l2: int - Global in-flight L2 classify semaphore size. Must be >= 1.
    :param l1_skip_l2_confidence_threshold: float - Skip L2 when L1 confidence >= this.
        Range [0.0, 1.01]; 1.01 disables skip.
    :param classify_timeout_seconds: float - Hard timeout for ``_classify_single`` /
        L0 extract stage. 0.0 disables. Must be >= 0.
    """
    enabled: bool
    task_type: str
    prompt_tag: str
    schema_version: str
    min_confidence: float
    tier_3_timeout_seconds: float
    alternative_band_high: float
    max_alternative_interpretations: int
    keyword_expansion_max_terms: int
    max_concurrent_l2: int
    l1_skip_l2_confidence_threshold: float
    classify_timeout_seconds: float

    def __post_init__(self) -> None:
        if not self.task_type:
            raise ConfigurationError("qi.llm.task_type must be non-empty")
        if not self.prompt_tag:
            raise ConfigurationError("qi.llm.prompt_tag must be non-empty")
        if not isinstance(self.schema_version, str) or not self.schema_version.strip():
            raise ConfigurationError("qi.llm.schema_version must be a non-empty string")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ConfigurationError("qi.llm.min_confidence must be in [0,1]")
        if float(self.tier_3_timeout_seconds) <= 0.0:
            raise ConfigurationError("qi.llm.tier_3_timeout_seconds must be > 0")
        if not 0.0 < float(self.alternative_band_high) <= 1.0:
            raise ConfigurationError("qi.llm.alternative_band_high must be in (0, 1]")
        if int(self.max_alternative_interpretations) < 0:
            raise ConfigurationError("qi.llm.max_alternative_interpretations must be >= 0")
        if int(self.keyword_expansion_max_terms) < 1:
            raise ConfigurationError("qi.llm.keyword_expansion_max_terms must be >= 1")
        if not 0.0 <= float(self.l1_skip_l2_confidence_threshold) <= 1.01:
            raise ConfigurationError("qi.llm.l1_skip_l2_confidence_threshold must be in [0.0, 1.01]")
        if float(self.classify_timeout_seconds) < 0.0:
            raise ConfigurationError("qi.llm.classify_timeout_seconds must be >= 0")
        if int(self.max_concurrent_l2) < 1:
            raise ConfigurationError("qi.llm.max_concurrent_l2 must be >= 1")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QILLMConfig':
        _require(
            d,
            [
                'enabled', 'task_type', 'prompt_tag', 'schema_version', 'min_confidence',
                'tier_3_timeout_seconds',
                'alternative_band_high', 'max_alternative_interpretations',
                'keyword_expansion_max_terms', 'max_concurrent_l2',
                'l1_skip_l2_confidence_threshold', 'classify_timeout_seconds',
            ],
            'qi.llm',
        )
        return cls(
            enabled=bool(d['enabled']),
            task_type=str(d['task_type']),
            prompt_tag=str(d['prompt_tag']),
            schema_version=str(d['schema_version']),
            min_confidence=float(d['min_confidence']),
            tier_3_timeout_seconds=float(d['tier_3_timeout_seconds']),
            alternative_band_high=float(d['alternative_band_high']),
            max_alternative_interpretations=int(d['max_alternative_interpretations']),
            keyword_expansion_max_terms=int(d['keyword_expansion_max_terms']),
            max_concurrent_l2=int(d['max_concurrent_l2']),
            l1_skip_l2_confidence_threshold=float(d['l1_skip_l2_confidence_threshold']),
            classify_timeout_seconds=float(d['classify_timeout_seconds']),
        )


@dataclass
class QIRoutingConfig:
    """Confidence-gate thresholds for cascade routing.

    ``margin_min`` adds a margin-aware escalation rule on top of the existing
    absolute-confidence gate. T1 only short-circuits when the top-2 archetype
    margin (top1 minus top2) ≥ ``margin_min``; otherwise T2 (LLM) is invoked
    even if the top1 confidence already cleared ``accept_confidence``. A small
    margin signals confusable archetypes (e.g. ``hybrid`` vs ``explore``)
    where the cosine difference cannot reliably arbitrate. ``0.0`` (default)
    preserves legacy behaviour.
    """
    accept_confidence: float
    fallback_confidence: float
    routing_auto_execute_min: float
    routing_suggest_min: float
    t0_hint_wait_ms: float
    margin_min: float = 0.0
    hard_chip_override_min: int = 0
    hard_chip_override_intents: List[str] = field(default_factory=list)
    # Entity-aware routing mode (R4): penalise routing confidence by the number of
    # hard-filter entities above entity_risk_penalty_above_count.  When any of
    # these three fields is None the entity-aware adjustment is skipped and the
    # existing confidence-only derivation runs unchanged (zero behavior change).
    entity_risk_penalty_per_filter: Optional[float] = None
    entity_risk_max_penalty: Optional[float] = None
    entity_risk_penalty_above_count: Optional[int] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.fallback_confidence <= self.accept_confidence <= 1.0:
            raise ConfigurationError("qi.routing requires 0 <= fallback_confidence <= accept_confidence <= 1")
        if not 0.0 <= self.routing_suggest_min <= self.routing_auto_execute_min <= 1.0:
            raise ConfigurationError("qi.routing requires 0 <= routing_suggest_min <= routing_auto_execute_min <= 1")
        if float(self.t0_hint_wait_ms) < 0.0:
            raise ConfigurationError("qi.routing.t0_hint_wait_ms must be >= 0")
        if not 0.0 <= float(self.margin_min) <= 1.0:
            raise ConfigurationError("qi.routing.margin_min must be in [0.0, 1.0]")
        if int(self.hard_chip_override_min) < 0:
            raise ConfigurationError("qi.routing.hard_chip_override_min must be >= 0")
        if self.entity_risk_penalty_per_filter is not None and float(self.entity_risk_penalty_per_filter) < 0.0:
            raise ConfigurationError("qi.routing.entity_risk_penalty_per_filter must be >= 0")
        if self.entity_risk_max_penalty is not None and not 0.0 <= float(self.entity_risk_max_penalty) <= 1.0:
            raise ConfigurationError("qi.routing.entity_risk_max_penalty must be in [0.0, 1.0]")
        if self.entity_risk_penalty_above_count is not None and int(self.entity_risk_penalty_above_count) < 0:
            raise ConfigurationError("qi.routing.entity_risk_penalty_above_count must be >= 0")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QIRoutingConfig':
        _require( d, [ 'accept_confidence', 'fallback_confidence', 'routing_auto_execute_min', 'routing_suggest_min', 't0_hint_wait_ms', ], 'qi.routing', )
        return cls(
            accept_confidence=float(d['accept_confidence']),
            fallback_confidence=float(d['fallback_confidence']),
            routing_auto_execute_min=float(d['routing_auto_execute_min']),
            routing_suggest_min=float(d['routing_suggest_min']),
            t0_hint_wait_ms=float(d['t0_hint_wait_ms']),
            margin_min=float(d.get('margin_min', 0.0)),
            hard_chip_override_min=int(d.get('hard_chip_override_min', 0)),
            hard_chip_override_intents=list(d.get('hard_chip_override_intents', [])),
            entity_risk_penalty_per_filter=float(d['entity_risk_penalty_per_filter']) if 'entity_risk_penalty_per_filter' in d and d['entity_risk_penalty_per_filter'] is not None else None,
            entity_risk_max_penalty=float(d['entity_risk_max_penalty']) if 'entity_risk_max_penalty' in d and d['entity_risk_max_penalty'] is not None else None,
            entity_risk_penalty_above_count=int(d['entity_risk_penalty_above_count']) if 'entity_risk_penalty_above_count' in d and d['entity_risk_penalty_above_count'] is not None else None,
        )


_SPELL_VERBOSITY_VALUES = frozenset({"TOP", "CLOSEST", "ALL"})


@dataclass
class SpellCorrectConfig:
    """Tier-0 spell-correction config (SymSpell frequency-dictionary-backed).

    The corrector runs BEFORE the QI cascade and BEFORE the cache lookup.

    :param enabled: bool - Master switch. When False the registry passes ``spell_corrector=None`` and zero work is done.
    :param frequency_dict_path: str - Path to the SymSpell dictionary file. Use "symspellpy:<filename>" to resolve a file bundled with the installed symspellpy package, or supply an absolute path.
    :param max_edit_distance: int - Edit distance cap for candidate lookup. Must be in [1, 3].
    :param prefix_length: int - SymSpell index prefix length. Must be >= 1.
    :param min_token_length: int - Tokens shorter than this are skipped. Must be >= 2.
    :param max_tokens_to_correct: int - Per-query cap on rewritten tokens. Must be >= 1.
    :param verbosity: str - SymSpell lookup verbosity: TOP | CLOSEST | ALL.
    :param auto_apply: bool - When True, the corrector rewrites the query in place; when False, the corrected text is a suggestion only.
    :param protected_tokens: List[str] - Tokens never rewritten regardless of context (metric abbreviations etc.).
    :param protected_phrase_patterns: List[str] - Regex patterns (one capture group each) whose matched tokens are protected.
    """
    enabled: bool
    frequency_dict_path: str
    max_edit_distance: int
    prefix_length: int
    min_token_length: int
    max_tokens_to_correct: int
    verbosity: str
    auto_apply: bool
    protected_tokens: List[str] = field(default_factory=list)
    protected_phrase_patterns: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("qi.spell_correct.enabled must be a bool")
        if not isinstance(self.frequency_dict_path, str) or not self.frequency_dict_path:
            raise ConfigurationError("qi.spell_correct.frequency_dict_path must be a non-empty string")
        if not 1 <= self.max_edit_distance <= 3:
            raise ConfigurationError("qi.spell_correct.max_edit_distance must be in [1, 3]")
        if self.prefix_length < 1:
            raise ConfigurationError("qi.spell_correct.prefix_length must be >= 1")
        if self.min_token_length < 2:
            raise ConfigurationError("qi.spell_correct.min_token_length must be >= 2")
        if self.max_tokens_to_correct < 1:
            raise ConfigurationError("qi.spell_correct.max_tokens_to_correct must be >= 1")
        if self.verbosity.upper() not in _SPELL_VERBOSITY_VALUES:
            raise ConfigurationError(f"qi.spell_correct.verbosity must be one of {sorted(_SPELL_VERBOSITY_VALUES)}")
        if not isinstance(self.auto_apply, bool):
            raise ConfigurationError("qi.spell_correct.auto_apply must be a bool")
        if not isinstance(self.protected_tokens, list):
            raise ConfigurationError("qi.spell_correct.protected_tokens must be a list")
        if not isinstance(self.protected_phrase_patterns, list):
            raise ConfigurationError("qi.spell_correct.protected_phrase_patterns must be a list")
        for _pat in self.protected_phrase_patterns:
            if not isinstance(_pat, str) or not _pat:
                raise ConfigurationError("qi.spell_correct.protected_phrase_patterns entries must be non-empty strings")
            try:
                re.compile(_pat)
            except re.error as _re_err:
                raise ConfigurationError(f"qi.spell_correct.protected_phrase_patterns: invalid regex {_pat!r}: {_re_err}") from _re_err

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SpellCorrectConfig':
        _require(d, ['enabled', 'frequency_dict_path', 'max_edit_distance', 'prefix_length', 'min_token_length', 'max_tokens_to_correct', 'verbosity', 'auto_apply', 'protected_tokens', 'protected_phrase_patterns'], 'qi.spell_correct')  # noqa: E501
        return cls(enabled=bool(d['enabled']), frequency_dict_path=str(d['frequency_dict_path']), max_edit_distance=int(d['max_edit_distance']), prefix_length=int(d['prefix_length']), min_token_length=int(d['min_token_length']), max_tokens_to_correct=int(d['max_tokens_to_correct']), verbosity=str(d['verbosity']), auto_apply=bool(d['auto_apply']), protected_tokens=[str(t).lower() for t in d['protected_tokens'] if isinstance(t, (str, int))], protected_phrase_patterns=[str(p) for p in d['protected_phrase_patterns'] if isinstance(p, str) and p])  # noqa: E501


@dataclass
class QueryTransformerConfig:
    """Two-tier query rewriter config: LLM primary (model via task_type), local seq2seq fallback on timeout or absence."""
    enabled: bool
    rewrite_enabled: bool
    llm_tier_enabled: bool
    rewrite_threshold: int
    rewrite_echo_max_start_offset: int
    task_type: str
    prompt_tag: str
    timeout_seconds: float
    llm_system_prompt_template: str
    llm_user_prompt_template: str
    model_path: str
    max_tokens: int
    rewrite_max_new_tokens: int
    rewrite_prompt_template: str
    signal_preservation_patterns: List[str]
    encode_from_rewrite: bool
    classify_on_transformed_query: bool
    combine_rewrite_with_l0_extract: bool
    on_rewrite_reject_reextract: bool

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("qi.query_transformer.enabled must be a bool")
        if not isinstance(self.rewrite_enabled, bool):
            raise ConfigurationError("qi.query_transformer.rewrite_enabled must be a bool")
        if not isinstance(self.llm_tier_enabled, bool):
            raise ConfigurationError("qi.query_transformer.llm_tier_enabled must be a bool")
        if not isinstance(self.encode_from_rewrite, bool):
            raise ConfigurationError("qi.query_transformer.encode_from_rewrite must be a bool")
        if not isinstance(self.classify_on_transformed_query, bool):
            raise ConfigurationError("qi.query_transformer.classify_on_transformed_query must be a bool")
        if not isinstance(self.combine_rewrite_with_l0_extract, bool):
            raise ConfigurationError("qi.query_transformer.combine_rewrite_with_l0_extract must be a bool")
        if not isinstance(self.on_rewrite_reject_reextract, bool):
            raise ConfigurationError("qi.query_transformer.on_rewrite_reject_reextract must be a bool")
        if self.rewrite_threshold < 1:
            raise ConfigurationError("qi.query_transformer.rewrite_threshold must be >= 1")
        if self.rewrite_echo_max_start_offset < 0:
            raise ConfigurationError("qi.query_transformer.rewrite_echo_max_start_offset must be >= 0")
        if not isinstance(self.task_type, str) or not self.task_type.strip():
            raise ConfigurationError("qi.query_transformer.task_type must be a non-empty string")
        if not isinstance(self.prompt_tag, str) or not self.prompt_tag.strip():
            raise ConfigurationError("qi.query_transformer.prompt_tag must be a non-empty string")
        if self.timeout_seconds <= 0.0:
            raise ConfigurationError("qi.query_transformer.timeout_seconds must be > 0")
        if not isinstance(self.llm_system_prompt_template, str) or not self.llm_system_prompt_template.strip():
            raise ConfigurationError("qi.query_transformer.llm_system_prompt_template must be a non-empty string")
        if not isinstance(self.llm_user_prompt_template, str) or not self.llm_user_prompt_template.strip():
            raise ConfigurationError("qi.query_transformer.llm_user_prompt_template must be a non-empty string")
        if "{query}" not in self.llm_user_prompt_template:
            raise ConfigurationError("qi.query_transformer.llm_user_prompt_template must contain the '{query}' placeholder")
        if not isinstance(self.model_path, str) or not self.model_path:
            raise ConfigurationError("qi.query_transformer.model_path must be a non-empty string")
        if self.max_tokens < 1:
            raise ConfigurationError("qi.query_transformer.max_tokens must be >= 1")
        if self.rewrite_max_new_tokens < 1:
            raise ConfigurationError("qi.query_transformer.rewrite_max_new_tokens must be >= 1")
        if not isinstance(self.rewrite_prompt_template, str) or not self.rewrite_prompt_template.strip():
            raise ConfigurationError("qi.query_transformer.rewrite_prompt_template must be a non-empty string")
        if "{query}" not in self.rewrite_prompt_template:
            raise ConfigurationError("qi.query_transformer.rewrite_prompt_template must contain the '{query}' placeholder")
        if not isinstance(self.signal_preservation_patterns, list):
            raise ConfigurationError("qi.query_transformer.signal_preservation_patterns must be a list of regex strings")
        for _p in self.signal_preservation_patterns:
            if not isinstance(_p, str) or not _p.strip():
                raise ConfigurationError("qi.query_transformer.signal_preservation_patterns entries must be non-empty strings")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QueryTransformerConfig':
        _require(
            d,
            [
                'enabled', 'rewrite_enabled', 'llm_tier_enabled', 'rewrite_threshold', 'rewrite_echo_max_start_offset',
                'task_type', 'prompt_tag', 'timeout_seconds', 'llm_system_prompt_template', 'llm_user_prompt_template',
                'model_path', 'max_tokens', 'rewrite_max_new_tokens', 'rewrite_prompt_template',
                'signal_preservation_patterns', 'encode_from_rewrite', 'classify_on_transformed_query',
                'combine_rewrite_with_l0_extract', 'on_rewrite_reject_reextract',
            ],
            'qi.query_transformer',
        )
        return cls(
            enabled=bool(d['enabled']),
            rewrite_enabled=bool(d['rewrite_enabled']),
            llm_tier_enabled=bool(d['llm_tier_enabled']),
            rewrite_threshold=int(d['rewrite_threshold']),
            rewrite_echo_max_start_offset=int(d['rewrite_echo_max_start_offset']),
            task_type=str(d['task_type']),
            prompt_tag=str(d['prompt_tag']),
            timeout_seconds=float(d['timeout_seconds']),
            llm_system_prompt_template=str(d['llm_system_prompt_template']),
            llm_user_prompt_template=str(d['llm_user_prompt_template']),
            model_path=str(d['model_path']),
            max_tokens=int(d['max_tokens']),
            rewrite_max_new_tokens=int(d['rewrite_max_new_tokens']),
            rewrite_prompt_template=str(d['rewrite_prompt_template']),
            signal_preservation_patterns=[str(p) for p in d['signal_preservation_patterns']],
            encode_from_rewrite=bool(d['encode_from_rewrite']),
            classify_on_transformed_query=bool(d['classify_on_transformed_query']),
            combine_rewrite_with_l0_extract=bool(d['combine_rewrite_with_l0_extract']),
            on_rewrite_reject_reextract=bool(d['on_rewrite_reject_reextract']),
        )


@dataclass
class CentroidRetrainerConfig:
    """L1 (semantic router) centroid-retrainer config.

    Recomputes centroids from real-traffic positives and shadow-deploys before promotion. The
    retrainer is a stateless service; its YAML block governs the
    sample floor + shadow-parity gate it applies.

    :param enabled: bool - When False, the retrainer is not constructed
        and ``Subsystems.centroid_retrainer`` is None.
    :param min_samples_per_archetype: int - Per-archetype trusted-positive
        floor (plan default 500). Below this floor, the archetype is
        dropped from the candidate; an archetype-empty candidate raises.
        Must be ``>= 1``.
    :param min_shadow_agreement: float - Minimum shadow agreement rate
        (in [0, 1]) for the candidate to receive verdict='promote'.
        Below this floor the verdict is 'shadow_only' (parked for
        manual review).
    :param window_seconds: float - Time window the source positives were
        drawn from. Stored on the candidate for auditing only — no
        retrainer logic depends on the value beyond ``>= 0``.
    :param max_signals_per_read: int - Ceiling on the number of signals
        fetched from SignalStore per collect call. Must be >= 1.
    """
    enabled: bool
    min_samples_per_archetype: int
    min_shadow_agreement: float
    window_seconds: float
    max_signals_per_read: int

    def __post_init__(self) -> None:
        if self.min_samples_per_archetype < 1:
            raise ConfigurationError("centroid_retrainer.min_samples_per_archetype must be >= 1")
        if not 0.0 <= self.min_shadow_agreement <= 1.0:
            raise ConfigurationError("centroid_retrainer.min_shadow_agreement must be in [0,1]")
        if float(self.window_seconds) < 0.0:
            raise ConfigurationError("centroid_retrainer.window_seconds must be >= 0")
        if self.max_signals_per_read < 1:
            raise ConfigurationError("centroid_retrainer.max_signals_per_read must be >= 1")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CentroidRetrainerConfig':
        _require(d, ['enabled', 'min_samples_per_archetype', 'min_shadow_agreement', 'window_seconds', 'max_signals_per_read'], 'qi.centroid_retrainer')
        return cls(
            enabled=bool(d['enabled']),
            min_samples_per_archetype=int(d['min_samples_per_archetype']),
            min_shadow_agreement=float(d['min_shadow_agreement']),
            window_seconds=float(d['window_seconds']),
            max_signals_per_read=int(d['max_signals_per_read']),
        )


@dataclass
class CentroidRetrainerDriverConfig:
    """Background driver that polls SignalStore and applies centroid retrain cycles.

    :param enabled: bool - When False, no driver is constructed and Subsystems.centroid_retrainer_driver is None.
    :param interval_seconds: float - Cadence between retrain cycles (>= 10.0).
    :param shadow_query_limit: int - Max shadow-probe queries per cycle (>= 1).
    :param max_consecutive_failures: int - Consecutive cycle errors that trigger exponential back-off (>= 1).
    :param positive_signal_types: List[str] - Signal types whose payloads carry query_text + query_type and are treated as archetype positives.
    """
    enabled: bool
    interval_seconds: float
    shadow_query_limit: int
    max_consecutive_failures: int
    positive_signal_types: List[str]

    def __post_init__(self) -> None:
        if self.interval_seconds < 10.0:
            raise ConfigurationError(f"qi.centroid_retrainer_driver.interval_seconds must be >= 10.0; got {self.interval_seconds}")
        if self.shadow_query_limit < 1:
            raise ConfigurationError(f"qi.centroid_retrainer_driver.shadow_query_limit must be >= 1; got {self.shadow_query_limit}")
        if self.max_consecutive_failures < 1:
            raise ConfigurationError(f"qi.centroid_retrainer_driver.max_consecutive_failures must be >= 1; got {self.max_consecutive_failures}")
        if not self.positive_signal_types:
            raise ConfigurationError("qi.centroid_retrainer_driver.positive_signal_types must be non-empty")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CentroidRetrainerDriverConfig':
        _require(d, ['enabled', 'interval_seconds', 'shadow_query_limit', 'max_consecutive_failures', 'positive_signal_types'], 'qi.centroid_retrainer_driver')
        return cls(enabled=bool(d['enabled']), interval_seconds=float(d['interval_seconds']), shadow_query_limit=int(d['shadow_query_limit']), max_consecutive_failures=int(d['max_consecutive_failures']), positive_signal_types=list(d['positive_signal_types']))  # noqa: E501


@dataclass(frozen=True)
class RouterRetrainingConfig:
    """Config for the ClickHouse ->traffic ring buffer ->router seeds ->head training pipeline.

    ClickHouse connection is NOT duplicated here — it is read from retrieval.clickhouse at
    runtime (single source of truth). This config governs only harvest + training behaviour.

    When enabled, app startup checks whether enough new positive signals have accumulated
    (new_signals_threshold) since the last retrain run. If so, a background task runs
    signal_harvester.py ->train_all_models.py to produce a new .npz artifact.

    :param enabled: bool - When False, no startup check or background task is launched.
    :param clickhouse_lookback_days: int - Days of ClickHouse history to pull per harvest run.
    :param positive_signal_types: List[str] - Signal types treated as archetype positives.
    :param max_signals_per_archetype: int - Cap on signals fetched per archetype per run.
    :param min_new_signals_per_archetype: int - Minimum new texts per archetype before merging into seeds.
    :param new_signals_threshold: int - Total new signals since last retrain before auto-trigger fires.
    :param traffic_signals_path: str - Path to JSONL ring buffer (${LOCAL_PRETRAINED_DIR}-relative).
    :param traffic_signals_max_rows: int - Ring buffer capacity (FIFO evict when exceeded).
    :param last_retrain_marker_path: str - Path to timestamp marker file.
    :param seeds_path: str - Path to router_seeds.yaml (empty ->resolved from qi.semantic.seeds_path).
    :param output_npz_path: str - Path to save the trained .npz head artifact.
    """
    enabled: bool
    clickhouse_lookback_days: int
    positive_signal_types: List[str]
    max_signals_per_archetype: int
    min_new_signals_per_archetype: int
    new_signals_threshold: int
    traffic_signals_path: str
    traffic_signals_max_rows: int
    last_retrain_marker_path: str
    seeds_path: str
    output_npz_path: str
    ngram_weights_output_path: str = ""
    ngram_top_k: int = 200
    ngram_min_log_odds: float = 0.3
    hard_negatives_path: str = ""
    hard_neg_oversample: int = 0
    hard_neg_threshold: float = 0.65

    def __post_init__(self) -> None:
        if self.clickhouse_lookback_days < 1:
            raise ConfigurationError(f"qi.router_retraining.clickhouse_lookback_days must be >= 1; got {self.clickhouse_lookback_days}")
        if not self.positive_signal_types:
            raise ConfigurationError("qi.router_retraining.positive_signal_types must be non-empty")
        if self.max_signals_per_archetype < 1:
            raise ConfigurationError(f"qi.router_retraining.max_signals_per_archetype must be >= 1; got {self.max_signals_per_archetype}")
        if self.min_new_signals_per_archetype < 1:
            raise ConfigurationError(f"qi.router_retraining.min_new_signals_per_archetype must be >= 1; got {self.min_new_signals_per_archetype}")
        if self.new_signals_threshold < 1:
            raise ConfigurationError(f"qi.router_retraining.new_signals_threshold must be >= 1; got {self.new_signals_threshold}")
        if self.traffic_signals_max_rows < 1:
            raise ConfigurationError(f"qi.router_retraining.traffic_signals_max_rows must be >= 1; got {self.traffic_signals_max_rows}")
        if not self.traffic_signals_path:
            raise ConfigurationError("qi.router_retraining.traffic_signals_path must be non-empty")
        if not self.last_retrain_marker_path:
            raise ConfigurationError("qi.router_retraining.last_retrain_marker_path must be non-empty")
        if not self.output_npz_path:
            raise ConfigurationError("qi.router_retraining.output_npz_path must be non-empty")
        if self.ngram_top_k < 1:
            raise ConfigurationError(f"qi.router_retraining.ngram_top_k must be >= 1; got {self.ngram_top_k}")
        if self.ngram_min_log_odds < 0:
            raise ConfigurationError(f"qi.router_retraining.ngram_min_log_odds must be >= 0; got {self.ngram_min_log_odds}")
        if self.hard_neg_oversample < 0:
            raise ConfigurationError(f"qi.router_retraining.hard_neg_oversample must be >= 0; got {self.hard_neg_oversample}")
        if not 0.0 < self.hard_neg_threshold < 1.0:
            raise ConfigurationError(f"qi.router_retraining.hard_neg_threshold must be in (0,1); got {self.hard_neg_threshold}")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'RouterRetrainingConfig':
        _require(d, [
            'enabled', 'clickhouse_lookback_days', 'positive_signal_types',
            'max_signals_per_archetype', 'min_new_signals_per_archetype', 'new_signals_threshold',
            'traffic_signals_path', 'traffic_signals_max_rows', 'last_retrain_marker_path',
            'seeds_path', 'output_npz_path',
        ], 'qi.router_retraining')
        return cls(
            enabled=bool(d['enabled']),
            clickhouse_lookback_days=int(d['clickhouse_lookback_days']),
            positive_signal_types=list(d['positive_signal_types']),
            max_signals_per_archetype=int(d['max_signals_per_archetype']),
            min_new_signals_per_archetype=int(d['min_new_signals_per_archetype']),
            new_signals_threshold=int(d['new_signals_threshold']),
            traffic_signals_path=str(d['traffic_signals_path']),
            traffic_signals_max_rows=int(d['traffic_signals_max_rows']),
            last_retrain_marker_path=str(d['last_retrain_marker_path']),
            seeds_path=str(d['seeds_path']),
            output_npz_path=str(d['output_npz_path']),
            ngram_weights_output_path=str(d.get('ngram_weights_output_path', '')),
            ngram_top_k=int(d.get('ngram_top_k', 200)),
            ngram_min_log_odds=float(d.get('ngram_min_log_odds', 0.3)),
            hard_negatives_path=str(d.get('hard_negatives_path', '')),
            hard_neg_oversample=int(d.get('hard_neg_oversample', 0)),
            hard_neg_threshold=float(d.get('hard_neg_threshold', 0.65)),
        )


# Must mirror ``semantic_search.contracts.RESIDUAL_KINDS`` exactly. Defined
# locally to keep the config layer free of contract imports (matches the
# existing ``_VECTOR_BACKENDS`` / ``_STRUCTURED_BACKENDS`` convention).
_RESIDUAL_KINDS_SET = frozenset({'empty', 'navigational', 'semantic'})


@dataclass
class ResidualQIConfig:
    """Residual-kind classifier config (qi.residual block).

    Drives :func:`semantic_search.qi.engine._classify_residual_kind`. When
    disabled, ``QueryIntent.residual_kind`` stays ``None`` for every query and
    all downstream consumers fall back to legacy behavior.

    :param enabled: bool - Master flag. Off = no classification, contract
        field stays None on every QueryIntent.
    :param navigational_tokens: List[str] - Lowercased query-side stop list.
        After T0 entity stripping, tokens in this set are removed; if zero
        content tokens remain the residual is labelled ``navigational``.
    :param min_content_tokens_for_semantic: int - Minimum non-navigational
        tokens that must remain after stripping for the residual to be
        labelled ``semantic``. Below this floor the residual is ``navigational``.
    """
    enabled: bool
    navigational_tokens: List[str]
    min_content_tokens_for_semantic: int

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("qi.residual.enabled must be a bool")
        if not isinstance(self.navigational_tokens, list):
            raise ConfigurationError("qi.residual.navigational_tokens must be a list")
        if self.enabled and len(self.navigational_tokens) == 0:
            raise ConfigurationError( "qi.residual.navigational_tokens must be non-empty when qi.residual.enabled=true" )
        for token in self.navigational_tokens:
            if not isinstance(token, str) or not token:
                raise ConfigurationError( "qi.residual.navigational_tokens entries must be non-empty strings" )
        if not isinstance(self.min_content_tokens_for_semantic, int):
            raise ConfigurationError( "qi.residual.min_content_tokens_for_semantic must be an integer" )
        if self.min_content_tokens_for_semantic < 0:
            raise ConfigurationError( "qi.residual.min_content_tokens_for_semantic must be >= 0" )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ResidualQIConfig':
        _require(d, ['enabled', 'navigational_tokens', 'min_content_tokens_for_semantic'], 'qi.residual')
        return cls( enabled=bool(d['enabled']), navigational_tokens=[str(t).lower() for t in d['navigational_tokens']], min_content_tokens_for_semantic=int(d['min_content_tokens_for_semantic']), )


@dataclass
class QINormalizeConfig:
    """Query text normalization for cache keys and QI encode/classify inputs.

    All fields required in YAML — no in-code defaults.
    """
    lowercase: bool
    collapse_whitespace: bool
    whitespace_pattern: str
    normalize_quotes: bool
    quote_translations: List[Dict[str, Any]]
    expand_gd_alias: bool
    strip_trailing_punctuation: bool
    trailing_punctuation_chars: str

    def __post_init__(self) -> None:
        if not isinstance(self.lowercase, bool):
            raise ConfigurationError("qi.normalize.lowercase must be bool")
        if not isinstance(self.collapse_whitespace, bool):
            raise ConfigurationError("qi.normalize.collapse_whitespace must be bool")
        if not isinstance(self.whitespace_pattern, str) or not self.whitespace_pattern:
            raise ConfigurationError("qi.normalize.whitespace_pattern must be a non-empty string")
        try:
            self._whitespace_re = re.compile(self.whitespace_pattern)
        except re.error as exc:
            raise ConfigurationError(
                f"qi.normalize.whitespace_pattern is not a valid regex: {exc}"
            ) from exc
        if not isinstance(self.normalize_quotes, bool):
            raise ConfigurationError("qi.normalize.normalize_quotes must be bool")
        if not isinstance(self.quote_translations, list):
            raise ConfigurationError("qi.normalize.quote_translations must be a list")
        table: Dict[int, str] = {}
        for i, entry in enumerate(self.quote_translations):
            if not isinstance(entry, dict):
                raise ConfigurationError(
                    f"qi.normalize.quote_translations[{i}] must be a mapping"
                )
            if 'from_codepoint' not in entry or 'to_char' not in entry:
                raise ConfigurationError(
                    f"qi.normalize.quote_translations[{i}] requires from_codepoint and to_char"
                )
            cp = int(entry['from_codepoint'])
            to_char = str(entry['to_char'])
            if len(to_char) != 1:
                raise ConfigurationError(
                    f"qi.normalize.quote_translations[{i}].to_char must be a single character"
                )
            table[cp] = to_char
        self._quote_table = str.maketrans(table) if table else None
        if not isinstance(self.expand_gd_alias, bool):
            raise ConfigurationError("qi.normalize.expand_gd_alias must be bool")
        if not isinstance(self.strip_trailing_punctuation, bool):
            raise ConfigurationError("qi.normalize.strip_trailing_punctuation must be bool")
        if not isinstance(self.trailing_punctuation_chars, str):
            raise ConfigurationError("qi.normalize.trailing_punctuation_chars must be a string")
        if self.strip_trailing_punctuation and not self.trailing_punctuation_chars:
            raise ConfigurationError(
                "qi.normalize.trailing_punctuation_chars must be non-empty when "
                "strip_trailing_punctuation is true"
            )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QINormalizeConfig':
        _require(
            d,
            [
                'lowercase',
                'collapse_whitespace',
                'whitespace_pattern',
                'normalize_quotes',
                'quote_translations',
                'expand_gd_alias',
                'strip_trailing_punctuation',
                'trailing_punctuation_chars',
            ],
            'qi.normalize',
        )
        return cls(
            lowercase=bool(d['lowercase']),
            collapse_whitespace=bool(d['collapse_whitespace']),
            whitespace_pattern=str(d['whitespace_pattern']),
            normalize_quotes=bool(d['normalize_quotes']),
            quote_translations=list(d['quote_translations']),
            expand_gd_alias=bool(d['expand_gd_alias']),
            strip_trailing_punctuation=bool(d['strip_trailing_punctuation']),
            trailing_punctuation_chars=str(d['trailing_punctuation_chars']),
        )


@dataclass
class QIIntentResultCacheConfig:
    """Tier-0 in-process intent-result cache config (normalized_query ->QueryIntent).

    Auto-scales from warm-up capacity to scaled capacity when cumulative hits
    reach `scale_at_hit_count`, deferring RAM commitment until proven effective.
    """
    enabled: bool
    ttl_seconds: int
    max_entries: int
    scale_at_hit_count: int
    max_entries_scaled: int
    ttl_seconds_scaled: int

    def __post_init__(self) -> None:
        if self.ttl_seconds < 1:
            raise ConfigurationError("qi.intent_result_cache.ttl_seconds must be >= 1")
        if self.max_entries < 1:
            raise ConfigurationError("qi.intent_result_cache.max_entries must be >= 1")
        if self.scale_at_hit_count < 1:
            raise ConfigurationError("qi.intent_result_cache.scale_at_hit_count must be >= 1")
        if self.max_entries_scaled < self.max_entries:
            raise ConfigurationError("qi.intent_result_cache.max_entries_scaled must be >= max_entries")
        if self.ttl_seconds_scaled < self.ttl_seconds:
            raise ConfigurationError("qi.intent_result_cache.ttl_seconds_scaled must be >= ttl_seconds")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QIIntentResultCacheConfig':
        _require(d, ['enabled', 'ttl_seconds', 'max_entries', 'scale_at_hit_count', 'max_entries_scaled', 'ttl_seconds_scaled'], 'qi.intent_result_cache')
        return cls(
            enabled=bool(d['enabled']),
            ttl_seconds=int(d['ttl_seconds']),
            max_entries=int(d['max_entries']),
            scale_at_hit_count=int(d['scale_at_hit_count']),
            max_entries_scaled=int(d['max_entries_scaled']),
            ttl_seconds_scaled=int(d['ttl_seconds_scaled']),
        )


@dataclass
class QISemanticIntentCacheConfig:
    """Tier-0.5 in-process semantic intent cache config (query embedding ->QueryIntent).

    On lookup the cache encodes the incoming query and returns a cached QueryIntent
    iff the top cosine similarity exceeds similarity_threshold. Catches rephrased
    queries that miss the exact-match cache without re-classifying.
    """
    enabled: bool
    similarity_threshold: float
    max_entries: int
    ttl_seconds: int
    embedding_dim: int
    type_specific_similarity_thresholds: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        if not 0.0 < self.similarity_threshold <= 1.0:
            raise ConfigurationError("qi.semantic_intent_cache.similarity_threshold must be in (0, 1]")
        if self.max_entries < 1:
            raise ConfigurationError("qi.semantic_intent_cache.max_entries must be >= 1")
        if self.ttl_seconds < 1:
            raise ConfigurationError("qi.semantic_intent_cache.ttl_seconds must be >= 1")
        if self.embedding_dim < 4:
            raise ConfigurationError("qi.semantic_intent_cache.embedding_dim must be >= 4")
        if self.type_specific_similarity_thresholds is not None:
            for qt, thr in self.type_specific_similarity_thresholds.items():
                if not 0.0 < thr <= 1.0:
                    raise ConfigurationError(f"qi.semantic_intent_cache.type_specific_similarity_thresholds[{qt!r}] must be in (0, 1]")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QISemanticIntentCacheConfig':
        _require(d, ['enabled', 'similarity_threshold', 'max_entries', 'ttl_seconds', 'embedding_dim'], 'qi.semantic_intent_cache')
        _type_thresholds: Optional[Dict[str, float]] = None
        if isinstance(d.get('type_specific_similarity_thresholds'), dict):
            _type_thresholds = {str(k): float(v) for k, v in d['type_specific_similarity_thresholds'].items()}
        return cls(
            enabled=bool(d['enabled']),
            similarity_threshold=float(d['similarity_threshold']),
            max_entries=int(d['max_entries']),
            ttl_seconds=int(d['ttl_seconds']),
            embedding_dim=int(d['embedding_dim']),
            type_specific_similarity_thresholds=_type_thresholds,
        )


@dataclass
class QIHeadTrainerConfig:
    """Config for the logistic-head trainer CLI (qi.training section in base.yaml).

    All values consumed by head_trainer._cli_main() as config-level defaults;
    CLI flags override per-run when supplied explicitly.

    :param max_epochs: Hard cap on gradient-descent iterations per trial.
    :param holdout_fraction: Stratified holdout split fraction (0, 0.5].
    :param random_search_trials: Number of (lr, l2) trials in random search.
    :param max_time_seconds: Wall-clock budget for the full training run.
    :param classifier: Classifier backend — 'logistic' | 'gradient_boosting' | 'random_forest'.
    :param lr_min: Lower bound for log-uniform LR sampling.
    :param lr_max: Upper bound for log-uniform LR sampling.
    :param l2_min: Lower bound for log-uniform L2 sampling.
    :param l2_max: Upper bound for log-uniform L2 sampling.
    :param gb_n_estimators: Candidate tree counts for gradient_boosting / random_forest search.
    :param gb_max_depths: Candidate depths for tree classifiers (0 = unlimited).
    :param gb_learning_rate: Fixed learning rate for GradientBoostingClassifier.
    :param log_interval_trials: Log progress every N trials per classifier.
    :param ensemble_classifiers: Classifiers to train in parallel for ensemble mode.
    :param ensemble_weight_logistic: Probability weight for logistic in ensemble average.
    :param ensemble_weight_rf: Probability weight for random forest in ensemble average.
    :param ensemble_weight_gb: Probability weight for gradient boosting in ensemble average.
    """
    max_epochs: int
    patience: int
    holdout_fraction: float
    random_search_trials: int
    max_time_seconds: int
    classifier: str
    lr_min: float
    lr_max: float
    l2_min: float
    l2_max: float
    gb_n_estimators: List[int]
    gb_max_depths: List[int]
    gb_learning_rate: float
    log_interval_trials: int
    ensemble_classifiers: List[str]
    ensemble_weight_logistic: float
    ensemble_weight_svm: float
    ensemble_weight_gb: float
    svm_c_min: float
    svm_c_max: float
    svm_kernels: List[str]
    svm_gamma_opts: List[str]

    def __post_init__(self) -> None:
        if int(self.max_epochs) < 1:
            raise ConfigurationError("qi.training.max_epochs must be >= 1")
        if int(self.patience) < 1:
            raise ConfigurationError("qi.training.patience must be >= 1")
        if not 0.0 < float(self.holdout_fraction) <= 0.5:
            raise ConfigurationError("qi.training.holdout_fraction must be in (0, 0.5]")
        if int(self.random_search_trials) < 1:
            raise ConfigurationError("qi.training.random_search_trials must be >= 1")
        if int(self.max_time_seconds) < 1:
            raise ConfigurationError("qi.training.max_time_seconds must be >= 1")
        _allowed_clf = frozenset(('logistic', 'gradient_boosting', 'svm', 'sklearn_logistic', 'ensemble'))
        if str(self.classifier) not in _allowed_clf:
            raise ConfigurationError(f"qi.training.classifier must be one of {sorted(_allowed_clf)}")
        if float(self.lr_min) <= 0.0 or float(self.lr_max) <= float(self.lr_min):
            raise ConfigurationError("qi.training.lr_min/lr_max must satisfy 0 < lr_min < lr_max")
        if float(self.l2_min) <= 0.0 or float(self.l2_max) <= float(self.l2_min):
            raise ConfigurationError("qi.training.l2_min/l2_max must satisfy 0 < l2_min < l2_max")
        if not self.gb_n_estimators:
            raise ConfigurationError("qi.training.gb_n_estimators must be a non-empty list")
        if not self.gb_max_depths:
            raise ConfigurationError("qi.training.gb_max_depths must be a non-empty list")
        if float(self.gb_learning_rate) <= 0.0:
            raise ConfigurationError("qi.training.gb_learning_rate must be > 0")
        if int(self.log_interval_trials) < 1:
            raise ConfigurationError("qi.training.log_interval_trials must be >= 1")
        if not self.ensemble_classifiers:
            raise ConfigurationError("qi.training.ensemble_classifiers must be a non-empty list")
        for w_name, w_val in (('ensemble_weight_logistic', self.ensemble_weight_logistic), ('ensemble_weight_svm', self.ensemble_weight_svm), ('ensemble_weight_gb', self.ensemble_weight_gb)):
            if float(w_val) <= 0.0:
                raise ConfigurationError(f"qi.training.{w_name} must be > 0")
        if float(self.svm_c_min) <= 0.0 or float(self.svm_c_max) <= float(self.svm_c_min):
            raise ConfigurationError("qi.training.svm_c_min/svm_c_max must satisfy 0 < svm_c_min < svm_c_max")
        if not self.svm_kernels:
            raise ConfigurationError("qi.training.svm_kernels must be a non-empty list")
        if not self.svm_gamma_opts:
            raise ConfigurationError("qi.training.svm_gamma_opts must be a non-empty list")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QIHeadTrainerConfig':
        _require(d, ['max_epochs', 'patience', 'holdout_fraction', 'random_search_trials',
                     'max_time_seconds', 'classifier', 'lr_min', 'lr_max',
                     'l2_min', 'l2_max', 'gb_n_estimators', 'gb_max_depths',
                     'gb_learning_rate', 'log_interval_trials',
                     'ensemble_classifiers', 'ensemble_weight_logistic',
                     'ensemble_weight_svm', 'ensemble_weight_gb',
                     'svm_c_min', 'svm_c_max', 'svm_kernels', 'svm_gamma_opts'], 'qi.training')
        return cls(
            max_epochs=int(d['max_epochs']),
            patience=int(d['patience']),
            holdout_fraction=float(d['holdout_fraction']),
            random_search_trials=int(d['random_search_trials']),
            max_time_seconds=int(d['max_time_seconds']),
            classifier=str(d['classifier']),
            lr_min=float(d['lr_min']),
            lr_max=float(d['lr_max']),
            l2_min=float(d['l2_min']),
            l2_max=float(d['l2_max']),
            gb_n_estimators=[int(x) for x in d['gb_n_estimators']],
            gb_max_depths=[int(x) for x in d['gb_max_depths']],
            gb_learning_rate=float(d['gb_learning_rate']),
            log_interval_trials=int(d['log_interval_trials']),
            ensemble_classifiers=[str(c) for c in d['ensemble_classifiers']],
            ensemble_weight_logistic=float(d['ensemble_weight_logistic']),
            ensemble_weight_svm=float(d['ensemble_weight_svm']),
            ensemble_weight_gb=float(d['ensemble_weight_gb']),
            svm_c_min=float(d['svm_c_min']),
            svm_c_max=float(d['svm_c_max']),
            svm_kernels=[str(k) for k in d['svm_kernels']],
            svm_gamma_opts=[str(g) for g in d['svm_gamma_opts']],
        )


@dataclass
class QIDeepHeadTrainerConfig:
    """Config for the deep-learning head trainer CLI (qi.deep_training section in base.yaml).

    :param hidden_dims: MLP block widths (list of ints). Input dim is encoder_dim; output is num_labels.
    :param dropout: Dropout probability in each residual block.
    :param lr: AdamW learning rate.
    :param weight_decay: AdamW weight decay (mapped to l2_penalty in HeadTrainingMetadata).
    :param label_smoothing: CE label-smoothing epsilon.
    :param batch_size: Mini-batch size for training.
    :param epochs: Hard cap on training iterations.
    :param patience: Early-stop patience (no val-score improvement).
    :param lr_patience: ReduceLROnPlateau patience epochs.
    :param lr_factor: ReduceLROnPlateau multiplicative factor.
    :param min_lr: ReduceLROnPlateau minimum learning rate floor.
    :param max_grad_norm: Gradient clipping norm.
    :param seed: Random seed for reproducibility.
    :param holdout_fraction: Val split fraction in (0, 0.5].
    """
    hidden_dims: List[int]
    dropout: float
    lr: float
    weight_decay: float
    label_smoothing: float
    batch_size: int
    epochs: int
    patience: int
    lr_patience: int
    lr_factor: float
    min_lr: float
    max_grad_norm: float
    seed: int
    holdout_fraction: float

    def __post_init__(self) -> None:
        if not self.hidden_dims:
            raise ConfigurationError("qi.deep_training.hidden_dims must be a non-empty list")
        if not 0.0 < float(self.dropout) < 1.0:
            raise ConfigurationError("qi.deep_training.dropout must be in (0, 1)")
        if float(self.lr) <= 0.0:
            raise ConfigurationError("qi.deep_training.lr must be > 0")
        if float(self.weight_decay) < 0.0:
            raise ConfigurationError("qi.deep_training.weight_decay must be >= 0")
        if int(self.batch_size) < 1:
            raise ConfigurationError("qi.deep_training.batch_size must be >= 1")
        if int(self.epochs) < 1:
            raise ConfigurationError("qi.deep_training.epochs must be >= 1")
        if int(self.patience) < 1:
            raise ConfigurationError("qi.deep_training.patience must be >= 1")
        if not 0.0 < float(self.holdout_fraction) <= 0.5:
            raise ConfigurationError("qi.deep_training.holdout_fraction must be in (0, 0.5]")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QIDeepHeadTrainerConfig':
        _require(d, ['hidden_dims', 'dropout', 'lr', 'weight_decay', 'label_smoothing', 'batch_size', 'epochs', 'patience', 'lr_patience', 'lr_factor', 'min_lr', 'max_grad_norm', 'seed', 'holdout_fraction'], 'qi.deep_training')  # noqa: E501
        return cls(hidden_dims=[int(x) for x in d['hidden_dims']], dropout=float(d['dropout']), lr=float(d['lr']), weight_decay=float(d['weight_decay']), label_smoothing=float(d['label_smoothing']), batch_size=int(d['batch_size']), epochs=int(d['epochs']), patience=int(d['patience']), lr_patience=int(d['lr_patience']), lr_factor=float(d['lr_factor']), min_lr=float(d['min_lr']), max_grad_norm=float(d['max_grad_norm']), seed=int(d['seed']), holdout_fraction=float(d['holdout_fraction']))  # noqa: E501


@dataclass(frozen=True)
class TermDisambiguationRule:
    """One rule entry for the term disambiguator (qi.term_disambiguator.rules list).

    :param term: str — lowercased token to watch for in the normalized query.
    :param tld_form: str — canonical TLD rewrite (e.g. '.ai') applied when TLD reading wins.
    :param tld_context_tokens: List[str] — tokens in the context window that signal TLD intent.
    :param topic_context_tokens: List[str] — tokens that signal the term is a topic/keyword.
        Topic signals take priority over TLD signals when both appear.
    :param default_to_tld: bool — when no context signal is found, apply TLD rewrite if True.
    """
    term: str
    tld_form: str
    tld_context_tokens: List[str]
    topic_context_tokens: List[str]
    default_to_tld: bool

    def __post_init__(self) -> None:
        if not isinstance(self.term, str) or not self.term:
            raise ConfigurationError("qi.term_disambiguator.rules[].term must be a non-empty string")
        if not isinstance(self.tld_form, str) or not self.tld_form.startswith('.'):
            raise ConfigurationError(f"qi.term_disambiguator.rules[].tld_form must start with '.' (got {self.tld_form!r})")
        if not isinstance(self.tld_context_tokens, list):
            raise ConfigurationError(f"qi.term_disambiguator.rules[{self.term}].tld_context_tokens must be a list")
        if not isinstance(self.topic_context_tokens, list):
            raise ConfigurationError(f"qi.term_disambiguator.rules[{self.term}].topic_context_tokens must be a list")
        if not isinstance(self.default_to_tld, bool):
            raise ConfigurationError(f"qi.term_disambiguator.rules[{self.term}].default_to_tld must be a bool")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TermDisambiguationRule':
        """Parse one rule entry from a YAML mapping."""
        _require(d, ['term', 'tld_form', 'tld_context_tokens', 'topic_context_tokens', 'default_to_tld'], 'qi.term_disambiguator.rules[]')
        return cls(
            term=str(d['term']).lower(),
            tld_form=str(d['tld_form']),
            tld_context_tokens=[str(t).lower() for t in d['tld_context_tokens'] if isinstance(t, (str, int))],
            topic_context_tokens=[str(t).lower() for t in d['topic_context_tokens'] if isinstance(t, (str, int))],
            default_to_tld=bool(d['default_to_tld']),
        )


@dataclass(frozen=True)
class TermDisambiguatorConfig:
    """Term-level polysemy disambiguator config (qi.term_disambiguator block).

    Runs before L0 (L0LLMFilterExtractor) and L1 (SemanticRouter) to rewrite ambiguous tokens — such as
    bare TLD stems (``ai``, ``io``, ``co``) — to their canonical dot-notation form when
    context signals support the TLD reading.  L2 (LLM) always receives the original text.

    All fields are required when the block is present. The block is optional on ``QIConfig``.

    :param enabled: bool — Master switch. When False, every call to disambiguate() is a no-op.
    :param context_window: int — Number of tokens on each side of the ambiguous token to
        examine for TLD or topic context signals. Must be >= 1.
    :param rules: List[TermDisambiguationRule] — Per-term disambiguation rules.
    """
    enabled: bool
    context_window: int
    rules: List[TermDisambiguationRule]

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("qi.term_disambiguator.enabled must be a bool")
        if not isinstance(self.context_window, int) or self.context_window < 1:
            raise ConfigurationError("qi.term_disambiguator.context_window must be int >= 1")
        if not isinstance(self.rules, list):
            raise ConfigurationError("qi.term_disambiguator.rules must be a list")
        terms_seen: set = set()
        for _r in self.rules:
            if not isinstance(_r, TermDisambiguationRule):
                raise ConfigurationError("qi.term_disambiguator.rules entries must be TermDisambiguationRule instances")
            if _r.term in terms_seen:
                raise ConfigurationError(f"qi.term_disambiguator.rules: duplicate term {_r.term!r}")
            terms_seen.add(_r.term)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TermDisambiguatorConfig':
        """Parse from YAML dict. Requires enabled, context_window, rules."""
        _require(d, ['enabled', 'context_window', 'rules'], 'qi.term_disambiguator')
        raw_rules = d['rules']
        if not isinstance(raw_rules, list):
            raise ConfigurationError("qi.term_disambiguator.rules must be a list")
        rules = [TermDisambiguationRule.from_dict(r) for r in raw_rules if isinstance(r, dict)]
        return cls(
            enabled=bool(d['enabled']),
            context_window=int(d['context_window']),
            rules=rules,
        )


@dataclass
class QIEnsembleVoterConfig:
    """Per-voter weight and veto settings for the ensemble resolver.

    :param voter_id: str - Unique identifier matching the voter implementation key.
    :param weight: float - Relative influence in the weighted vote tally. Must be > 0.
    :param has_veto: bool - When True, a vote differing from the tally winner overrides it.
    :param abstain_on_no_signal: bool - When True, the voter emits no ballot when it has
        no signal rather than defaulting to the configured fallback archetype.
    :param timeout_ms: float - Per-voter wall-clock budget in milliseconds. Zero disables
        the per-voter cap (outer classify_timeout_seconds still applies). Must be >= 0.
    """
    voter_id: str
    weight: float
    has_veto: bool
    abstain_on_no_signal: bool
    timeout_ms: float

    def __post_init__(self) -> None:
        if not isinstance(self.voter_id, str) or not self.voter_id.strip():
            raise ConfigurationError("qi.ensemble.voters[].voter_id must be a non-empty string")
        if not isinstance(self.weight, (int, float)) or float(self.weight) <= 0.0:
            raise ConfigurationError(f"qi.ensemble.voters[{self.voter_id!r}].weight must be > 0")
        if not isinstance(self.has_veto, bool):
            raise ConfigurationError(f"qi.ensemble.voters[{self.voter_id!r}].has_veto must be bool")
        if not isinstance(self.abstain_on_no_signal, bool):
            raise ConfigurationError(f"qi.ensemble.voters[{self.voter_id!r}].abstain_on_no_signal must be bool")
        if not isinstance(self.timeout_ms, (int, float)) or float(self.timeout_ms) < 0.0:
            raise ConfigurationError(f"qi.ensemble.voters[{self.voter_id!r}].timeout_ms must be >= 0")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QIEnsembleVoterConfig':
        _require(d, ['voter_id', 'weight', 'has_veto', 'abstain_on_no_signal', 'timeout_ms'], 'qi.ensemble.voters[]')
        return cls(
            voter_id=str(d['voter_id']),
            weight=float(d['weight']),
            has_veto=bool(d['has_veto']),
            abstain_on_no_signal=bool(d['abstain_on_no_signal']),
            timeout_ms=float(d['timeout_ms']),
        )


@dataclass
class QIEnsembleRoutingConfig:
    """Agreement-ratio and confidence-based routing mode thresholds for the ensemble resolver.

    Both ``agreement`` and ``confidence`` axes are checked independently; the mode is
    the most permissive band that either axis satisfies. This allows high agreement at
    moderate confidence (or vice-versa) to still reach ``auto_execute``.

    :param auto_execute_agreement_min: float - Minimum vote-weight agreement ratio for
        ``auto_execute``. In [0, 1]. Must be >= suggest_agreement_min.
    :param auto_execute_confidence_min: float - Minimum winner weighted-confidence for
        ``auto_execute``. In [0, 1]. Must be >= suggest_confidence_min.
    :param suggest_agreement_min: float - Minimum agreement ratio for ``suggest``. In [0, 1].
    :param suggest_confidence_min: float - Minimum winner weighted-confidence for
        ``suggest``. In [0, 1].
    """
    auto_execute_agreement_min: float
    auto_execute_confidence_min: float
    suggest_agreement_min: float
    suggest_confidence_min: float

    def __post_init__(self) -> None:
        for _name, _val in (
            ('auto_execute_agreement_min', self.auto_execute_agreement_min),
            ('auto_execute_confidence_min', self.auto_execute_confidence_min),
            ('suggest_agreement_min', self.suggest_agreement_min),
            ('suggest_confidence_min', self.suggest_confidence_min),
        ):
            if not isinstance(_val, (int, float)) or not 0.0 <= float(_val) <= 1.0:
                raise ConfigurationError(f"qi.ensemble.routing.{_name} must be in [0.0, 1.0]")
        if float(self.suggest_agreement_min) > float(self.auto_execute_agreement_min):
            raise ConfigurationError(
                "qi.ensemble.routing requires suggest_agreement_min <= auto_execute_agreement_min"
            )
        if float(self.suggest_confidence_min) > float(self.auto_execute_confidence_min):
            raise ConfigurationError(
                "qi.ensemble.routing requires suggest_confidence_min <= auto_execute_confidence_min"
            )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QIEnsembleRoutingConfig':
        _require(
            d,
            ['auto_execute_agreement_min', 'auto_execute_confidence_min',
             'suggest_agreement_min', 'suggest_confidence_min'],
            'qi.ensemble.routing',
        )
        return cls(
            auto_execute_agreement_min=float(d['auto_execute_agreement_min']),
            auto_execute_confidence_min=float(d['auto_execute_confidence_min']),
            suggest_agreement_min=float(d['suggest_agreement_min']),
            suggest_confidence_min=float(d['suggest_confidence_min']),
        )


@dataclass
class QIEnsembleConfig:
    """Top-level ensemble voter config for the QI engine.

    All configured voters fire in parallel; the winner is the archetype with the
    highest weighted vote sum, subject to veto-voter override.

    :param voters: List[QIEnsembleVoterConfig] - Ordered voter definitions.  voter_id
        values must be unique and must match the voter keys the engine recognises
        ('ngram', 'semantic', 'entity', 'aggregation', 'llm').
    :param routing: QIEnsembleRoutingConfig - Agreement-ratio routing thresholds.
    :param consensus_cancel_l2: bool - When True, unanimous fast-voter agreement
        cancels the in-flight L2 LLM task so its latency is not incurred.
    :param consensus_cancel_threshold: float - Minimum agreement ratio among fast
        voters (ngram, semantic, entity, aggregation) that triggers L2 cancellation
        when consensus_cancel_l2 is True. In (0, 1].
    :param extract_before_classify: bool - When True, await L0 extract before L1/L2
        so classification never races ahead of hard/soft entities + keywords.
    :param fallback_archetype: str - Archetype returned when all voters abstain.
        Must be in the parent QIConfig.query_types list.
    """
    voters: List[QIEnsembleVoterConfig]
    routing: QIEnsembleRoutingConfig
    consensus_cancel_l2: bool
    consensus_cancel_threshold: float
    extract_before_classify: bool
    fallback_archetype: str

    def __post_init__(self) -> None:
        if not isinstance(self.voters, list) or not self.voters:
            raise ConfigurationError("qi.ensemble.voters must be a non-empty list")
        ids = [v.voter_id for v in self.voters]
        if len(ids) != len(set(ids)):
            raise ConfigurationError(f"qi.ensemble.voters voter_id values must be unique; duplicates={[v for v in ids if ids.count(v) > 1]}")
        if not isinstance(self.consensus_cancel_l2, bool):
            raise ConfigurationError("qi.ensemble.consensus_cancel_l2 must be bool")
        if not isinstance(self.consensus_cancel_threshold, (int, float)) or not 0.0 < float(self.consensus_cancel_threshold) <= 1.0:
            raise ConfigurationError("qi.ensemble.consensus_cancel_threshold must be in (0.0, 1.0]")
        if not isinstance(self.extract_before_classify, bool):
            raise ConfigurationError("qi.ensemble.extract_before_classify must be bool")
        if not isinstance(self.fallback_archetype, str) or not self.fallback_archetype.strip():
            raise ConfigurationError("qi.ensemble.fallback_archetype must be a non-empty string")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QIEnsembleConfig':
        _require(
            d,
            [
                'voters', 'routing', 'consensus_cancel_l2', 'consensus_cancel_threshold',
                'extract_before_classify', 'fallback_archetype',
            ],
            'qi.ensemble',
        )
        voters_raw = d['voters']
        if not isinstance(voters_raw, list):
            raise ConfigurationError("qi.ensemble.voters must be a list")
        return cls(
            voters=[QIEnsembleVoterConfig.from_dict(v) for v in voters_raw if isinstance(v, dict)],
            routing=QIEnsembleRoutingConfig.from_dict(d['routing']),
            consensus_cancel_l2=bool(d['consensus_cancel_l2']),
            consensus_cancel_threshold=float(d['consensus_cancel_threshold']),
            extract_before_classify=bool(d['extract_before_classify']),
            fallback_archetype=str(d['fallback_archetype']),
        )


@dataclass
class QIEnsembleSettingsConfig:
    """YAML-facing ensemble controls (voters assembled at registry from live components)."""
    extract_before_classify: bool
    consensus_cancel_l2: bool
    consensus_cancel_threshold: float
    routing: QIEnsembleRoutingConfig

    def __post_init__(self) -> None:
        if not isinstance(self.extract_before_classify, bool):
            raise ConfigurationError("qi.ensemble.extract_before_classify must be bool")
        if not isinstance(self.consensus_cancel_l2, bool):
            raise ConfigurationError("qi.ensemble.consensus_cancel_l2 must be bool")
        if not isinstance(self.consensus_cancel_threshold, (int, float)) or not 0.0 < float(self.consensus_cancel_threshold) <= 1.0:
            raise ConfigurationError("qi.ensemble.consensus_cancel_threshold must be in (0.0, 1.0]")
        if not isinstance(self.routing, QIEnsembleRoutingConfig):
            raise ConfigurationError("qi.ensemble.routing must be a QIEnsembleRoutingConfig")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QIEnsembleSettingsConfig':
        _require(
            d,
            [
                'extract_before_classify',
                'consensus_cancel_l2',
                'consensus_cancel_threshold',
                'routing',
            ],
            'qi.ensemble',
        )
        return cls(
            extract_before_classify=bool(d['extract_before_classify']),
            consensus_cancel_l2=bool(d['consensus_cancel_l2']),
            consensus_cancel_threshold=float(d['consensus_cancel_threshold']),
            routing=QIEnsembleRoutingConfig.from_dict(d['routing']),
        )


@dataclass
class QIEntityVoterConfig:
    """Config for the entity-signal voter in the ensemble resolver.

    The entity voter infers archetype from L0-extracted entities and raw query
    text signals.  It has veto power by default because structured filter
    constraints (price, TLD, length) are deterministic signals that the semantic
    centroid classifier cannot reliably distinguish from advisory phrasing.

    :param voter_id: str - Must match the voter_id in QIEnsembleConfig.voters for
        weight and veto settings to be applied.
    :param confidence_emit: float - Confidence assigned to ballots cast from hard
        entity hits. In (0, 1].
    :param signal_only_scale: float - Multiplier applied to confidence_emit when
        the voter fires from raw query text signals only (no hard entities extracted).
        In (0, 1]. Lower values express lower certainty for signal-only inference.
    :param hard_filter_force_slots: List[str] - Entity slot names whose presence
        (with chip_kind='hard') forces a vote for 'hybrid'. Reads from config;
        no slot names are hardcoded in the voter implementation.
    :param veto_archetype: Optional[str] - Archetype the voter vetoes when its inferred
        archetype differs. Set to null to disable veto entirely.
    :param value_discovery_signals: List[str] - Lowercased substrings that signal
        value-discovery or investor intent and trigger a 'hybrid' vote even when
        no structured entities are present.
    """
    voter_id: str
    confidence_emit: float
    signal_only_scale: float
    hard_filter_force_slots: List[str]
    veto_archetype: Optional[str]
    value_discovery_signals: List[str]

    def __post_init__(self) -> None:
        if not isinstance(self.voter_id, str) or not self.voter_id.strip():
            raise ConfigurationError("qi.entity_voter.voter_id must be a non-empty string")
        if not isinstance(self.confidence_emit, (int, float)) or not 0.0 < float(self.confidence_emit) <= 1.0:
            raise ConfigurationError("qi.entity_voter.confidence_emit must be in (0.0, 1.0]")
        if not isinstance(self.signal_only_scale, (int, float)) or not 0.0 < float(self.signal_only_scale) <= 1.0:
            raise ConfigurationError("qi.entity_voter.signal_only_scale must be in (0.0, 1.0]")
        if not isinstance(self.hard_filter_force_slots, list) or not self.hard_filter_force_slots:
            raise ConfigurationError("qi.entity_voter.hard_filter_force_slots must be a non-empty list")
        if not all(isinstance(s, str) and s for s in self.hard_filter_force_slots):
            raise ConfigurationError("qi.entity_voter.hard_filter_force_slots must contain non-empty strings")
        if self.veto_archetype is not None and (not isinstance(self.veto_archetype, str) or not self.veto_archetype.strip()):
            raise ConfigurationError("qi.entity_voter.veto_archetype must be a non-empty string or null")
        if not isinstance(self.value_discovery_signals, list):
            raise ConfigurationError("qi.entity_voter.value_discovery_signals must be a list")
        if not all(isinstance(s, str) for s in self.value_discovery_signals):
            raise ConfigurationError("qi.entity_voter.value_discovery_signals must contain strings")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QIEntityVoterConfig':
        _require(
            d,
            ['voter_id', 'confidence_emit', 'signal_only_scale', 'hard_filter_force_slots'],
            'qi.entity_voter',
        )
        _raw_veto = d.get('veto_archetype')
        return cls(
            voter_id=str(d['voter_id']),
            confidence_emit=float(d['confidence_emit']),
            signal_only_scale=float(d['signal_only_scale']),
            hard_filter_force_slots=[str(s) for s in d['hard_filter_force_slots']],
            veto_archetype=str(_raw_veto) if _raw_veto is not None else None,
            value_discovery_signals=[str(s).lower() for s in d.get('value_discovery_signals', [])],
        )


_SOFT_APPLY_MODES = frozenset({'rank', 'off'})


@dataclass
class QIEntitySlotsConfig:
    """Hard vs soft entity-slot taxonomy for L0 extract + response shaping.

    All fields required in YAML — no in-code defaults.

    :param soft_slot_names: List[str] - Slots extracted by the batched SOFT LLM group;
        excluded from identified_filters; emitted under soft_response_key on full-search
        filter_summary only (not on qie_only).
    :param hard_entity_names: List[str] - Slots classified chip_kind=hard (FIND / local hard).
    :param soft_response_key: str - Response JSON key for soft signals (e.g. soft_signals).
    :param soft_group_tag: str - Prompt-tag suffix for the soft LLM group call.
    :param soft_apply_mode: str - ``rank`` (post-fusion SLD boost) or ``off`` (no boost).
    :param soft_rank_boost_weight: float - Additive fused_score boost weight when mode=rank.
    :param soft_rank_slot_names: List[str] - Soft slots eligible for SLD rank boost;
        must be a non-empty subset of soft_slot_names.
    :param soft_rank_miss_penalty_ratio: float - Fraction of boost weight subtracted when
        a positive soft constraint misses (0..1).
    :param soft_rank_partial_boost_ratio: float - Fraction of boost weight for partial
        matches (e.g. word_count) (0..1).
    """
    soft_slot_names: List[str]
    hard_entity_names: List[str]
    soft_response_key: str
    soft_group_tag: str
    soft_apply_mode: str
    soft_rank_boost_weight: float
    soft_rank_slot_names: List[str]
    soft_rank_miss_penalty_ratio: float
    soft_rank_partial_boost_ratio: float

    def __post_init__(self) -> None:
        if not isinstance(self.soft_slot_names, list) or not self.soft_slot_names:
            raise ConfigurationError("qi.entity_slots.soft_slot_names must be a non-empty list")
        if not isinstance(self.hard_entity_names, list) or not self.hard_entity_names:
            raise ConfigurationError("qi.entity_slots.hard_entity_names must be a non-empty list")
        if not isinstance(self.soft_rank_slot_names, list) or not self.soft_rank_slot_names:
            raise ConfigurationError("qi.entity_slots.soft_rank_slot_names must be a non-empty list")
        for field_name, values in (
            ('soft_slot_names', self.soft_slot_names),
            ('hard_entity_names', self.hard_entity_names),
            ('soft_rank_slot_names', self.soft_rank_slot_names),
        ):
            for v in values:
                if not isinstance(v, str) or not v.strip():
                    raise ConfigurationError(f"qi.entity_slots.{field_name} entries must be non-empty strings")
        if not isinstance(self.soft_response_key, str) or not self.soft_response_key.strip():
            raise ConfigurationError("qi.entity_slots.soft_response_key must be a non-empty string")
        if not isinstance(self.soft_group_tag, str) or not self.soft_group_tag.strip():
            raise ConfigurationError("qi.entity_slots.soft_group_tag must be a non-empty string")
        mode = str(self.soft_apply_mode).strip().lower()
        if mode not in _SOFT_APPLY_MODES:
            raise ConfigurationError(
                f"qi.entity_slots.soft_apply_mode must be one of {sorted(_SOFT_APPLY_MODES)}, got {self.soft_apply_mode!r}"
            )
        self.soft_apply_mode = mode
        if not isinstance(self.soft_rank_boost_weight, (int, float)) or float(self.soft_rank_boost_weight) <= 0:
            raise ConfigurationError("qi.entity_slots.soft_rank_boost_weight must be a positive number")
        self.soft_rank_boost_weight = float(self.soft_rank_boost_weight)
        for ratio_name in ('soft_rank_miss_penalty_ratio', 'soft_rank_partial_boost_ratio'):
            ratio = getattr(self, ratio_name)
            if not isinstance(ratio, (int, float)) or float(ratio) < 0 or float(ratio) > 1:
                raise ConfigurationError(f"qi.entity_slots.{ratio_name} must be a number in [0, 1]")
            setattr(self, ratio_name, float(ratio))
        soft = frozenset(str(s) for s in self.soft_slot_names)
        hard = frozenset(str(s) for s in self.hard_entity_names)
        rank_slots = frozenset(str(s) for s in self.soft_rank_slot_names)
        overlap = soft & hard
        if overlap:
            raise ConfigurationError(
                f"qi.entity_slots soft/hard overlap not allowed: {sorted(overlap)}"
            )
        not_soft = rank_slots - soft
        if not_soft:
            raise ConfigurationError(
                f"qi.entity_slots.soft_rank_slot_names must be subset of soft_slot_names; "
                f"unknown={sorted(not_soft)}"
            )
        self._soft_set = soft
        self._hard_set = hard
        self._soft_rank_set = rank_slots

    @property
    def soft_slot_set(self) -> frozenset:
        return self._soft_set

    @property
    def hard_entity_set(self) -> frozenset:
        return self._hard_set

    @property
    def soft_rank_slot_set(self) -> frozenset:
        return self._soft_rank_set

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QIEntitySlotsConfig':
        _require(
            d,
            [
                'soft_slot_names',
                'hard_entity_names',
                'soft_response_key',
                'soft_group_tag',
                'soft_apply_mode',
                'soft_rank_boost_weight',
                'soft_rank_slot_names',
                'soft_rank_miss_penalty_ratio',
                'soft_rank_partial_boost_ratio',
            ],
            'qi.entity_slots',
        )
        return cls(
            soft_slot_names=[str(s) for s in d['soft_slot_names']],
            hard_entity_names=[str(s) for s in d['hard_entity_names']],
            soft_response_key=str(d['soft_response_key']),
            soft_group_tag=str(d['soft_group_tag']),
            soft_apply_mode=str(d['soft_apply_mode']),
            soft_rank_boost_weight=float(d['soft_rank_boost_weight']),
            soft_rank_slot_names=[str(s) for s in d['soft_rank_slot_names']],
            soft_rank_miss_penalty_ratio=float(d['soft_rank_miss_penalty_ratio']),
            soft_rank_partial_boost_ratio=float(d['soft_rank_partial_boost_ratio']),
        )


@dataclass
class QIL0LLMEntityConfig:
    """LLM entity extractor config — step-agnostic; can be wired at any pipeline position.

    All fields required in YAML — no in-code defaults.

    :param enabled: bool - Master toggle. When False the extractor returns None immediately.
    :param task_type: str - LLM router task type; drives model selection via weights_by_task_type.
        Set cost: 1.0 in weights_by_task_type to always pick the cheapest available model.
    :param timeout_seconds: float - Per-group LLM call timeout. On timeout the group returns
        no entities (non-fatal); remaining groups are unaffected.
    :param max_entities: int - Hard cap on entities returned per classify call.
    :param confidence: float - Confidence written on every L0-LLM Entity / identified_filters entry.
    :param source_tag: str - Value written to Entity.source; must be in contracts.DECISION_TIERS.
    :param prompt_tag: str - Prompt version tag for LLM logs and L0 filter cache keys.
    :param schema_version: str - Extract schema version for L0 filter cache keys.
    :param combined_prompt_tag: str - Prompt tag when rewrite+extract share one LLM call.
    :param combined_schema_version: str - Schema version for combined rewrite+extract cache keys.
    :param keyword_min_probability: float - Minimum keyword probability, as a percentage
        (0-100; e.g. 70 = 70%). Keywords below this are dropped from the extracted
        keyword list. Compared against the LLM's per-keyword probability (0.0-1.0)
        after dividing by 100.
    """
    enabled: bool
    task_type: str
    timeout_seconds: float
    max_entities: int
    confidence: float
    source_tag: str
    prompt_tag: str
    schema_version: str
    combined_prompt_tag: str
    combined_schema_version: str
    keyword_min_probability: float

    def __post_init__(self) -> None:
        if not isinstance(self.task_type, str) or not self.task_type:
            raise ConfigurationError("qi.l0_llm_entity.task_type must be a non-empty string")
        if float(self.timeout_seconds) <= 0.0:
            raise ConfigurationError("qi.l0_llm_entity.timeout_seconds must be > 0")
        if int(self.max_entities) < 1:
            raise ConfigurationError("qi.l0_llm_entity.max_entities must be >= 1")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ConfigurationError("qi.l0_llm_entity.confidence must be in [0,1]")
        self.confidence = float(self.confidence)
        if not isinstance(self.source_tag, str) or not self.source_tag:
            raise ConfigurationError("qi.l0_llm_entity.source_tag must be a non-empty string")
        if not isinstance(self.prompt_tag, str) or not self.prompt_tag.strip():
            raise ConfigurationError("qi.l0_llm_entity.prompt_tag must be a non-empty string")
        if not isinstance(self.schema_version, str) or not self.schema_version.strip():
            raise ConfigurationError("qi.l0_llm_entity.schema_version must be a non-empty string")
        if not isinstance(self.combined_prompt_tag, str) or not self.combined_prompt_tag.strip():
            raise ConfigurationError("qi.l0_llm_entity.combined_prompt_tag must be a non-empty string")
        if not isinstance(self.combined_schema_version, str) or not self.combined_schema_version.strip():
            raise ConfigurationError("qi.l0_llm_entity.combined_schema_version must be a non-empty string")
        if not 0.0 <= float(self.keyword_min_probability) <= 100.0:
            raise ConfigurationError("qi.l0_llm_entity.keyword_min_probability must be in [0,100]")
        self.keyword_min_probability = float(self.keyword_min_probability)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QIL0LLMEntityConfig':
        _require(
            d,
            [
                'enabled',
                'task_type',
                'timeout_seconds',
                'max_entities',
                'confidence',
                'source_tag',
                'prompt_tag',
                'schema_version',
                'combined_prompt_tag',
                'combined_schema_version',
                'keyword_min_probability',
            ],
            'qi.l0_llm_entity',
        )
        return cls(
            enabled=bool(d['enabled']),
            task_type=str(d['task_type']),
            timeout_seconds=float(d['timeout_seconds']),
            max_entities=int(d['max_entities']),
            confidence=float(d['confidence']),
            source_tag=str(d['source_tag']),
            prompt_tag=str(d['prompt_tag']),
            schema_version=str(d['schema_version']),
            combined_prompt_tag=str(d['combined_prompt_tag']),
            combined_schema_version=str(d['combined_schema_version']),
            keyword_min_probability=float(d['keyword_min_probability']),
        )


@dataclass
class QIL0RegexEntityConfig:
    """Deterministic regex entity extractor config; runs offline with no call_router.

    :param enabled: bool - Master toggle for regex L0 extract.
    :param max_entities: int - Hard cap on entities returned per classify call.
    :param confidence: float - Confidence written on every regex-emitted Entity.
    :param source_tag: str - Value written to Entity.source / identified_filters.source.
    :param fallback_only_when_llm_unavailable: bool - When True, regex runs only if the
        L0 LLM extract did not complete (disabled / missing router / exception). When
        False, regex also runs when LLM completed with zero entities (legacy empty
        fallback). Successful LLM empty results never use regex when this is True.
    """
    enabled: bool
    max_entities: int
    confidence: float
    source_tag: str
    fallback_only_when_llm_unavailable: bool

    def __post_init__(self) -> None:
        if int(self.max_entities) < 1:
            raise ConfigurationError("qi.l0_regex_entity.max_entities must be >= 1")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ConfigurationError("qi.l0_regex_entity.confidence must be in [0,1]")
        if not isinstance(self.source_tag, str) or not self.source_tag:
            raise ConfigurationError("qi.l0_regex_entity.source_tag must be a non-empty string")
        if not isinstance(self.fallback_only_when_llm_unavailable, bool):
            raise ConfigurationError(
                "qi.l0_regex_entity.fallback_only_when_llm_unavailable must be bool"
            )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QIL0RegexEntityConfig':
        _require(
            d,
            [
                'enabled',
                'max_entities',
                'confidence',
                'source_tag',
                'fallback_only_when_llm_unavailable',
            ],
            'qi.l0_regex_entity',
        )
        return cls(
            enabled=bool(d['enabled']),
            max_entities=int(d['max_entities']),
            confidence=float(d['confidence']),
            source_tag=str(d['source_tag']),
            fallback_only_when_llm_unavailable=bool(d['fallback_only_when_llm_unavailable']),
        )


@dataclass
class QIConfig:
    """QI Engine config bundle."""
    enabled: bool
    query_types: List[str]
    default_query_type: str
    encoder: QIEncoderConfig
    regex: QIRegexConfig
    semantic: QISemanticConfig
    llm: QILLMConfig
    routing: QIRoutingConfig
    normalize: QINormalizeConfig
    # Tier-0 spell-correction. Optional/additive: pre-Rec-#12 YAML
    # continues to parse unchanged (the registry treats `None` and
    # `enabled=false` identically — no corrector is constructed).
    spell_correct: Optional[SpellCorrectConfig] = None
    # Tier-2 (L1 semantic router) centroid-retraining channel. Optional /
    # additive: when omitted or `enabled=false`,
    # `Subsystems.centroid_retrainer` is None and no retrain channel is
    # exposed. When enabled, the retrainer is wired against the live
    # encoder + SemanticRouter for shadow agreement scoring.
    centroid_retrainer: Optional[CentroidRetrainerConfig] = None
    # Residual-kind classifier. Optional/additive: when omitted or
    # ``enabled=false``, the QI engine sets ``QueryIntent.residual_kind=None``
    # on every classification.
    residual: Optional[ResidualQIConfig] = None
    # Tier-0 in-process intent-result cache. Optional/additive: when omitted or
    # `enabled=false`, every classify() call goes to the LLM. When enabled,
    # repeat queries are served from LRU+TTL memory and the cache auto-scales
    # once it proves effective (`scale_at_hit_count` hits observed).
    intent_result_cache: Optional[QIIntentResultCacheConfig] = None
    # Tier-0.5 semantic intent cache (query embedding ->QueryIntent). Optional/additive:
    # when omitted or enabled=false, every cache miss falls through to the L1/L2 cascade.
    # When enabled, semantically similar queries reuse a cached QueryIntent without
    # re-classifying. embedding_dim must equal qi.encoder.dim.
    semantic_intent_cache: Optional[QISemanticIntentCacheConfig] = None
    # Head-trainer config (qi.training). Optional/additive: when omitted, the
    # head_trainer CLI falls back to its own argparse defaults. When present,
    # config values become the source-of-truth defaults and CLI flags override.
    training: Optional[QIHeadTrainerConfig] = None
    # Deep-learning head trainer config (qi.deep_training). Optional/additive:
    # when omitted, deep_head_trainer CLI is unavailable. When present,
    # DeepHead (residual MLP) competes alongside sklearn heads; winner selected by score.
    deep_training: Optional[QIDeepHeadTrainerConfig] = None
    # Domain-term polysemy disambiguator. Optional/additive: when omitted or
    # ``enabled=false``, L0 and L1 receive the normalized query unchanged.
    # When enabled, ambiguous TLD stems (e.g. "ai", "io", "co") are rewritten
    # to dot-notation form (e.g. ".ai") for L0/L1 before entity extraction and
    # routing, reducing TLD vs. topic misclassification.
    term_disambiguator: Optional[TermDisambiguatorConfig] = None
    # Background driver that polls SignalStore and runs centroid retrain cycles.
    # Optional/additive: when omitted or enabled=false, no driver is constructed
    # and Subsystems.centroid_retrainer_driver is None. Requires centroid_retrainer
    # to also be enabled — the registry skips driver construction when the retrainer is None.
    centroid_retrainer_driver: Optional[CentroidRetrainerDriverConfig] = None
    # Offline router retraining pipeline (ClickHouse ->traffic ring buffer ->seeds ->.npz).
    # Optional/additive: when omitted or enabled=false, no startup check is performed
    # and no background task is launched. When enabled, app startup checks the
    # new_signals_threshold and fires the harvest + train pipeline if crossed.
    router_retraining: Optional[RouterRetrainingConfig] = None
    # Entity-signal voter for the ensemble resolver. Optional/additive: when omitted,
    # no entity-based voter participates in ensemble resolution. When present, entity
    # voter_id must appear in ensemble.voters for weight/veto settings to apply.
    entity_voter: Optional[QIEntityVoterConfig] = None
    # L0 LLM entity extractor. Required: omitting or disabling this causes a
    # ConfigurationError at startup — no fallback is available.
    l0_llm_entity: Optional[QIL0LLMEntityConfig] = None
    # Deterministic regex entity extractor. Always available (no call_router dep);
    # supplies hard filter slots offline and pre-empts the LLM extractor on hard slots.
    l0_regex_entity: Optional[QIL0RegexEntityConfig] = None
    # DistilBERT query rewriter (prunes verbose queries). Optional/additive:
    # when omitted or enabled=false, queries pass through unchanged.
    query_transformer: Optional[QueryTransformerConfig] = None
    # Hard/soft slot taxonomy — required when L0 entity extract is enabled.
    entity_slots: Optional[QIEntitySlotsConfig] = None
    # Ensemble pipeline controls (extract-before-classify, L2 cancel). Voters are
    # assembled at registry time from live components; this block is required in YAML.
    ensemble: Optional[QIEnsembleSettingsConfig] = None

    def __post_init__(self) -> None:
        if not isinstance(self.query_types, list) or len(self.query_types) == 0:
            raise ConfigurationError("qi.query_types must be a non-empty list")
        if self.default_query_type not in self.query_types:
            raise ConfigurationError("qi.default_query_type must be in qi.query_types")
        _l0_llm_on = self.l0_llm_entity is not None and self.l0_llm_entity.enabled
        _l0_regex_on = self.l0_regex_entity is not None and self.l0_regex_entity.enabled
        if (_l0_llm_on or _l0_regex_on) and self.entity_slots is None:
            raise ConfigurationError(
                "qi.entity_slots is required when qi.l0_llm_entity or qi.l0_regex_entity is enabled"
            )
        for archetype in self.semantic.archetype_prototypes.keys():
            if archetype not in self.query_types:
                raise ConfigurationError(f"qi.semantic.archetype_prototypes contains unknown query_type '{archetype}'")
        # encoder dim is the spine that ties semantic-router, vector
        # retriever, and semantic caches together. Reject mismatches at the
        # config boundary so a misaligned dim never makes it to a runtime
        # ValidationError.
        cascade_enabled = (
            self.encoder.cascade is not None
            and self.encoder.cascade.enabled
            and self.encoder.cascade.stage_dims is not None
        )
        if not cascade_enabled and self.encoder.dim != self.semantic.embedding_dim:
            raise ConfigurationError( f"qi.encoder.dim ({self.encoder.dim}) must equal qi.semantic.embedding_dim ({self.semantic.embedding_dim})" )
        qt_set = frozenset(self.query_types)
        for ex in self.semantic.centroid_exclusions:
            if ex not in qt_set:
                raise ConfigurationError(f"qi.semantic.centroid_exclusions contains unknown query_type '{ex}'")
        if self.entity_voter is not None and self.entity_voter.veto_archetype is not None:
            if self.entity_voter.veto_archetype not in qt_set:
                raise ConfigurationError(
                    f"qi.entity_voter.veto_archetype '{self.entity_voter.veto_archetype}' not in qi.query_types"
                )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QIConfig':
        _require(
            d,
            [
                'enabled',
                'query_types',
                'default_query_type',
                'encoder',
                'regex',
                'semantic',
                'llm',
                'routing',
                'normalize',
            ],
            'qi',
        )
        spell_raw = d.get('spell_correct')
        spell_correct = SpellCorrectConfig.from_dict(spell_raw) if isinstance(spell_raw, dict) else None
        retrainer_raw = d.get('centroid_retrainer')
        centroid_retrainer = CentroidRetrainerConfig.from_dict(retrainer_raw) if isinstance(retrainer_raw, dict) else None
        residual_raw = d.get('residual')
        residual = ResidualQIConfig.from_dict(residual_raw) if isinstance(residual_raw, dict) else None
        cache_raw = d.get('intent_result_cache')
        intent_result_cache = QIIntentResultCacheConfig.from_dict(cache_raw) if isinstance(cache_raw, dict) else None
        sem_cache_raw = d.get('semantic_intent_cache')
        semantic_intent_cache = QISemanticIntentCacheConfig.from_dict(sem_cache_raw) if isinstance(sem_cache_raw, dict) else None
        training_raw = d.get('training')
        training = QIHeadTrainerConfig.from_dict(training_raw) if isinstance(training_raw, dict) else None
        deep_training_raw = d.get('deep_training')
        deep_training = QIDeepHeadTrainerConfig.from_dict(deep_training_raw) if isinstance(deep_training_raw, dict) else None
        td_raw = d.get('term_disambiguator')
        term_disambiguator = TermDisambiguatorConfig.from_dict(td_raw) if isinstance(td_raw, dict) else None
        crd_raw = d.get('centroid_retrainer_driver')
        centroid_retrainer_driver = CentroidRetrainerDriverConfig.from_dict(crd_raw) if isinstance(crd_raw, dict) else None
        rr_raw = d.get('router_retraining')
        router_retraining = RouterRetrainingConfig.from_dict(rr_raw) if isinstance(rr_raw, dict) else None
        entity_voter_raw = d.get('entity_voter')
        entity_voter = QIEntityVoterConfig.from_dict(entity_voter_raw) if isinstance(entity_voter_raw, dict) else None
        l0_llm_entity_raw = d.get('l0_llm_entity')
        l0_llm_entity = QIL0LLMEntityConfig.from_dict(l0_llm_entity_raw) if isinstance(l0_llm_entity_raw, dict) else None
        l0_regex_entity_raw = d.get('l0_regex_entity')
        l0_regex_entity = QIL0RegexEntityConfig.from_dict(l0_regex_entity_raw) if isinstance(l0_regex_entity_raw, dict) else None
        qt_raw = d.get('query_transformer')
        query_transformer = QueryTransformerConfig.from_dict(qt_raw) if isinstance(qt_raw, dict) else None
        entity_slots_raw = d.get('entity_slots')
        entity_slots = QIEntitySlotsConfig.from_dict(entity_slots_raw) if isinstance(entity_slots_raw, dict) else None
        ensemble_raw = d.get('ensemble')
        if not isinstance(ensemble_raw, dict):
            raise ConfigurationError("qi.ensemble is required and must be a mapping")
        ensemble = QIEnsembleSettingsConfig.from_dict(ensemble_raw)
        return cls(
            enabled=bool(d['enabled']),
            query_types=[str(t) for t in d['query_types']],
            default_query_type=str(d['default_query_type']),
            encoder=QIEncoderConfig.from_dict(d['encoder']),
            regex=QIRegexConfig.from_dict(d['regex']),
            semantic=QISemanticConfig.from_dict(d['semantic']),
            llm=QILLMConfig.from_dict(d['llm']),
            routing=QIRoutingConfig.from_dict(d['routing']),
            normalize=QINormalizeConfig.from_dict(d['normalize']),
            spell_correct=spell_correct,
            centroid_retrainer=centroid_retrainer,
            residual=residual,
            intent_result_cache=intent_result_cache,
            semantic_intent_cache=semantic_intent_cache,
            training=training,
            deep_training=deep_training,
            term_disambiguator=term_disambiguator,
            centroid_retrainer_driver=centroid_retrainer_driver,
            router_retraining=router_retraining,
            entity_voter=entity_voter,
            l0_llm_entity=l0_llm_entity,
            l0_regex_entity=l0_regex_entity,
            query_transformer=query_transformer,
            entity_slots=entity_slots,
            ensemble=ensemble,
        )


_VECTOR_BACKENDS = frozenset({'memory', 'qdrant'})
_STRUCTURED_BACKENDS = frozenset({'memory', 'qdrant'})
# Combine modes for multi-term keyword slots — mirrors structured_retriever.KEYWORD_MATCH_MODES.
_KEYWORD_MATCH_MODES = frozenset({'any', 'all'})


@dataclass
class TopicNegationConfig:
    """Topic/category negation config for the vector retriever.

    :param enabled: bool - Master toggle.
    :param alpha: float - Subtraction strength in q' = normalize(q - alpha*sum(centroids)).
    :param category_seeds: Dict[str, List[str]] - Topic -> seed phrases used to build
        the per-topic centroid (encoded with the residual encoder).
    """
    enabled: bool
    alpha: float
    category_seeds: Dict[str, List[str]]

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("retrieval.vector.topic_negation.enabled must be a bool")
        if self.alpha < 0.0:
            raise ConfigurationError("retrieval.vector.topic_negation.alpha must be >= 0")
        if not isinstance(self.category_seeds, dict):
            raise ConfigurationError("retrieval.vector.topic_negation.category_seeds must be a mapping")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TopicNegationConfig':
        _require(d, ['enabled', 'alpha', 'category_seeds'], 'retrieval.vector.topic_negation')
        return cls(enabled=bool(d['enabled']), alpha=float(d['alpha']), category_seeds={str(k).lower(): [str(s) for s in v] for k, v in dict(d['category_seeds']).items()})


@dataclass
class VectorRetrievalConfig:
    """Vector retriever config.

    :param backend: str - Concrete VectorIndex implementation to use
        (`'memory'` = `InMemoryVectorIndex`, `'qdrant'` = `QdrantVectorIndex`).
        Selecting `'qdrant'` requires `retrieval.qdrant` to be present and the
        `qdrant-client` package installed; if construction fails the registry
        logs a warning and falls back to the in-memory backend (degradation
        contract — never crashes the boot path).
    """
    enabled: bool
    top_k: int
    min_similarity: float
    embedding_dim: int
    backend: str
    topic_negation: Optional[TopicNegationConfig] = None

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ConfigurationError("retrieval.vector.top_k must be >= 1")
        if not 0.0 <= self.min_similarity <= 1.0:
            raise ConfigurationError("retrieval.vector.min_similarity must be in [0,1]")
        if self.embedding_dim < 4:
            raise ConfigurationError("retrieval.vector.embedding_dim must be >= 4")
        if self.backend not in _VECTOR_BACKENDS:
            raise ConfigurationError( f"retrieval.vector.backend must be one of {sorted(_VECTOR_BACKENDS)}" )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'VectorRetrievalConfig':
        _require(d, ['enabled', 'top_k', 'min_similarity', 'embedding_dim', 'backend'], 'retrieval.vector')
        tn_raw = d.get('topic_negation')
        topic_negation = TopicNegationConfig.from_dict(tn_raw) if isinstance(tn_raw, dict) else None
        return cls( enabled=bool(d['enabled']), top_k=int(d['top_k']), min_similarity=float(d['min_similarity']), embedding_dim=int(d['embedding_dim']), backend=str(d['backend']), topic_negation=topic_negation, )  # noqa: E501


@dataclass
class StructuredRetrievalConfig:
    """Structured retriever config.

    :param backend: str - Concrete StructuredIndex implementation to use
        (`'memory'` = `InMemoryStructuredIndex`, `'qdrant'` = `QdrantStructuredIndex`).
        When the unified Qdrant index also serves vector queries (i.e.
        `retrieval.vector.backend='qdrant'` AND `retrieval.qdrant.hybrid.enabled=true`),
        the structured retriever degrades to a no-op because the hybrid path
        already applies payload filters in a single round-trip on the vector
        backend (Option A).
    :param word_count_filter_enabled: bool - When True, the retriever segments candidate
        SLDs via the injected DomainNameSegmenter and applies word_count_min / word_count_max
        filters from entity slots. Has no effect when the segmenter is not wired.
    :param keyword_match_mode: str - Default combine mode ('any' = OR, 'all' = AND) for
        multi-term keyword slots when the intent carries no explicit 'keyword_match_mode'.
    :param unknown_selectable_fields: Dict[str, str] - Maps a boolean "select unknown"
        filter slot (e.g. 'traffic_is_unknown') to the payload field whose absence/None
        the slot selects (e.g. 'monthly_traffic'). Lets a query ask for items whose
        enrichment value is unknown instead of excluding them.
    """
    enabled: bool
    top_k: int
    backend: str
    word_count_filter_enabled: bool
    keyword_match_mode: str
    unknown_selectable_fields: Dict[str, str]
    lifecycle_auction_type_map: Dict[str, List[str]]
    traffic_signal_fields: List[str]

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ConfigurationError("retrieval.structured.top_k must be >= 1")
        if self.backend not in _STRUCTURED_BACKENDS:
            raise ConfigurationError( f"retrieval.structured.backend must be one of {sorted(_STRUCTURED_BACKENDS)}" )
        if not isinstance(self.word_count_filter_enabled, bool):
            raise ConfigurationError("retrieval.structured.word_count_filter_enabled must be a bool")
        if self.keyword_match_mode not in _KEYWORD_MATCH_MODES:
            raise ConfigurationError( f"retrieval.structured.keyword_match_mode must be one of {sorted(_KEYWORD_MATCH_MODES)}" )
        if not isinstance(self.unknown_selectable_fields, dict):
            raise ConfigurationError("retrieval.structured.unknown_selectable_fields must be a mapping")
        if not isinstance(self.lifecycle_auction_type_map, dict):
            raise ConfigurationError("retrieval.structured.lifecycle_auction_type_map must be a mapping")
        if not isinstance(self.traffic_signal_fields, list):
            raise ConfigurationError("retrieval.structured.traffic_signal_fields must be a list")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'StructuredRetrievalConfig':
        _require(d, ['enabled', 'top_k', 'backend', 'word_count_filter_enabled', 'keyword_match_mode', 'unknown_selectable_fields', 'lifecycle_auction_type_map', 'traffic_signal_fields'], 'retrieval.structured')  # noqa: E501
        return cls( enabled=bool(d['enabled']), top_k=int(d['top_k']), backend=str(d['backend']), word_count_filter_enabled=bool(d['word_count_filter_enabled']), keyword_match_mode=str(d['keyword_match_mode']), unknown_selectable_fields={str(k): str(v) for k, v in dict(d['unknown_selectable_fields']).items()}, lifecycle_auction_type_map={str(k): [str(i) for i in list(vs)] for k, vs in dict(d['lifecycle_auction_type_map']).items()}, traffic_signal_fields=[str(f) for f in list(d['traffic_signal_fields'])], )  # noqa: E501


# Real-time price-fan-out adapter config. When a query needs a real-time
# price check, the pipeline fans out to SQL and merges.
# The substrates table specifies ClickHouse `events_raw` as the
# live backend. Defaults to `enabled=false` so test/local boots stay on the
# in-memory PriceBandStore; production deployments flip `enabled=true` and
# point `table` at the real `events_raw` (or its alias).
#
# Identifier guards (boot-time, fail-loud):
#   * `table` matches one or two dotted segments of `[A-Za-z_][A-Za-z0-9_]*`
#     so callers cannot smuggle a SQL fragment via the table name.
#   * `score_expr` rejects every SQL-comment / statement-terminator token
#     and every non-printable char so the operator-supplied scoring
#     expression cannot break out of its SELECT slot.
_TABLE_IDENT_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]{0,63}(\.[A-Za-z_][A-Za-z0-9_]{0,63})?$')
_SCORE_EXPR_FORBIDDEN = (';', '--', '/*', '*/')


@dataclass
class ClickHousePriceBandAdapterConfig:
    """ClickHouse-backed `PriceBandStore` adapter config.

    :param enabled: bool - When False, registry keeps the in-memory store
    :param table: str - Single-segment or `db.table` identifier (allowlisted by regex)
    :param score_expr: str - Per-row score expression injected verbatim into SELECT
        (operator-controlled; rejects comment / semicolon / non-printable tokens)
    :param timeout_seconds: float - Wall-clock cap on the lookup round-trip
    :param max_rows: int - Hard ceiling on rows returned per lookup (LIMIT clause)
    :param item_id_column: str - Column to use as `item_id` in SELECT (default: `item_id`).
        Use e.g. `domain_name` when the target table stores the domain in a differently-named column.
    :param price_column: str - Column to use for price range predicates (default: `price`).
        Use e.g. `current_price` when the target table has a different price column name.
    :param base_where: Optional[str] - Static predicate always appended to the WHERE clause.
        Use to restrict to active rows, e.g. `ends_at > now64(3)`. Same security rules as
        `score_expr` apply (no semicolons, no SQL comment tokens, printable chars only).
    :param extra_columns: List[str] - Additional SQL column expressions appended to the SELECT
        after `item_id` and `score`. Use to fetch payload fields (tld, price, auction_type, ends_at)
        that flow through to result payloads. Same security rules as `score_expr` apply.
        Example: ["tld", "current_price AS price", "toUnixTimestamp64Milli(ends_at) / 1000.0 AS ends_at"]
    :param filter_column_expressions: Dict[str, str] - Maps a logical filter name to the SQL
        expression that must appear in the WHERE clause. Required when the physical column name
        differs from the filter key — e.g. ``auction_type`` is only a SELECT alias
        (``CAST(auction_type_id AS VARCHAR) AS auction_type``), not a physical column, so
        ``WHERE auction_type IN (...)`` is silently ignored by ClickHouse. Mapping
        ``{"auction_type": "CAST(auction_type_id AS VARCHAR)"}`` produces the correct predicate.
        Same forbidden-token rules as `score_expr` apply to each expression value.
    """
    enabled: bool
    table: str
    score_expr: str
    timeout_seconds: float
    max_rows: int
    item_id_column: str = 'item_id'
    price_column: str = 'price'
    base_where: Optional[str] = None
    extra_columns: List[str] = field(default_factory=list)
    filter_column_expressions: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("retrieval.sql.clickhouse_adapter.enabled must be a bool")
        if not isinstance(self.table, str) or not self.table:
            raise ConfigurationError("retrieval.sql.clickhouse_adapter.table must be a non-empty string")
        if not _TABLE_IDENT_RE.match(self.table):
            raise ConfigurationError( "retrieval.sql.clickhouse_adapter.table must match " "<segment> or <segment>.<segment> with [A-Za-z_][A-Za-z0-9_]{0,63}" )
        if not isinstance(self.score_expr, str) or not self.score_expr.strip():
            raise ConfigurationError("retrieval.sql.clickhouse_adapter.score_expr must be a non-empty string")
        if any(tok in self.score_expr for tok in _SCORE_EXPR_FORBIDDEN):
            raise ConfigurationError( "retrieval.sql.clickhouse_adapter.score_expr contains a forbidden token " f"(one of {list(_SCORE_EXPR_FORBIDDEN)})" )
        if any((not ch.isprintable()) for ch in self.score_expr):
            raise ConfigurationError("retrieval.sql.clickhouse_adapter.score_expr must contain only printable characters")
        if float(self.timeout_seconds) <= 0.0:
            raise ConfigurationError("retrieval.sql.clickhouse_adapter.timeout_seconds must be > 0")
        if int(self.max_rows) < 1:
            raise ConfigurationError("retrieval.sql.clickhouse_adapter.max_rows must be >= 1")
        _col_re = re.compile(r'^[A-Za-z_][A-Za-z0-9_]{0,63}$')
        if not _col_re.match(self.item_id_column):
            raise ConfigurationError("retrieval.sql.clickhouse_adapter.item_id_column must be a valid column identifier")
        if not _col_re.match(self.price_column):
            raise ConfigurationError("retrieval.sql.clickhouse_adapter.price_column must be a valid column identifier")
        if self.base_where is not None:
            if any(tok in self.base_where for tok in _SCORE_EXPR_FORBIDDEN):
                raise ConfigurationError( "retrieval.sql.clickhouse_adapter.base_where contains a forbidden token " f"(one of {list(_SCORE_EXPR_FORBIDDEN)})" )
            if any((not ch.isprintable()) for ch in self.base_where):
                raise ConfigurationError("retrieval.sql.clickhouse_adapter.base_where must contain only printable characters")
        for i, col_expr in enumerate(self.extra_columns):
            if not isinstance(col_expr, str) or not col_expr.strip():
                raise ConfigurationError(f"retrieval.sql.clickhouse_adapter.extra_columns[{i}] must be a non-empty string")
            if any(tok in col_expr for tok in _SCORE_EXPR_FORBIDDEN):
                raise ConfigurationError( f"retrieval.sql.clickhouse_adapter.extra_columns[{i}] contains a forbidden token " f"(one of {list(_SCORE_EXPR_FORBIDDEN)})" )
            if any((not ch.isprintable()) for ch in col_expr):
                raise ConfigurationError(f"retrieval.sql.clickhouse_adapter.extra_columns[{i}] must contain only printable characters")
        if not isinstance(self.filter_column_expressions, dict):
            raise ConfigurationError("retrieval.sql.clickhouse_adapter.filter_column_expressions must be a dict")
        for k, v in self.filter_column_expressions.items():
            if not isinstance(k, str) or not k.strip():
                raise ConfigurationError("retrieval.sql.clickhouse_adapter.filter_column_expressions keys must be non-empty strings")
            if not isinstance(v, str) or not v.strip():
                raise ConfigurationError(f"retrieval.sql.clickhouse_adapter.filter_column_expressions[{k!r}] must be a non-empty string")
            if any(tok in v for tok in _SCORE_EXPR_FORBIDDEN):
                raise ConfigurationError( f"retrieval.sql.clickhouse_adapter.filter_column_expressions[{k!r}] contains a forbidden token " f"(one of {list(_SCORE_EXPR_FORBIDDEN)})" )
            if any((not ch.isprintable()) for ch in v):
                raise ConfigurationError(f"retrieval.sql.clickhouse_adapter.filter_column_expressions[{k!r}] must contain only printable characters")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ClickHousePriceBandAdapterConfig':
        _require(d, ['enabled', 'table', 'score_expr', 'timeout_seconds', 'max_rows'], 'retrieval.sql.clickhouse_adapter')
        return cls(
            enabled=bool(d['enabled']),
            table=str(d['table']),
            score_expr=str(d['score_expr']),
            timeout_seconds=float(d['timeout_seconds']),
            max_rows=int(d['max_rows']),
            item_id_column=str(d.get('item_id_column', 'item_id')),
            price_column=str(d.get('price_column', 'price')),
            base_where=str(d['base_where']) if d.get('base_where') else None,
            extra_columns=[str(c) for c in d['extra_columns']] if d.get('extra_columns') else [],
            filter_column_expressions={str(k): str(v) for k, v in d['filter_column_expressions'].items()} if d.get('filter_column_expressions') else {},
        )


@dataclass
class SqlRetrievalConfig:
    enabled: bool
    top_k: int
    allowed_filter_columns: List[str]
    clickhouse_adapter: Optional[ClickHousePriceBandAdapterConfig] = None

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ConfigurationError("retrieval.sql.top_k must be >= 1")
        if not isinstance(self.allowed_filter_columns, list) or len(self.allowed_filter_columns) == 0:
            raise ConfigurationError("retrieval.sql.allowed_filter_columns must be a non-empty list")
        if self.clickhouse_adapter is not None and not isinstance(self.clickhouse_adapter, ClickHousePriceBandAdapterConfig):
            raise ConfigurationError("retrieval.sql.clickhouse_adapter must be a ClickHousePriceBandAdapterConfig")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SqlRetrievalConfig':
        _require(d, ['enabled', 'top_k', 'allowed_filter_columns'], 'retrieval.sql')
        adapter_raw = d.get('clickhouse_adapter')
        adapter = ClickHousePriceBandAdapterConfig.from_dict(adapter_raw) if adapter_raw is not None else None
        return cls( enabled=bool(d['enabled']), top_k=int(d['top_k']), allowed_filter_columns=[str(c) for c in d['allowed_filter_columns']], clickhouse_adapter=adapter, )


# Fusion source names that the orchestrator emits as ``CandidateSet.source``.
# Must match the keys used by ``_gather_candidates`` in
# ``semantic_search.orchestrator`` and the registered retriever set.
_FUSION_SOURCE_NAMES = frozenset({'vector', 'sparse', 'structured'})


@dataclass
class FusionWeightsProfile:
    """Per-source RRF weights for one residual-kind profile.

    Each weight multiplies the source's ``1 / (k + rank)`` contribution
    inside :meth:`semantic_search.retrieval.fusion.RRFFuser.fuse`. A weight of
    ``0.0`` fully suppresses the source; ``1.0`` is the legacy symmetric
    behavior.

    :param vector: float - Weight applied to the vector retriever's
        contribution. In ``[0.0, 1.0]``.
    :param sparse: float - Weight applied to the sparse (BM42) leg. In ``[0.0, 1.0]``.
    :param structured: float - Weight applied to the structured/payload-filter
        retriever. In ``[0.0, 1.0]``.
    :param sql: float - Weight applied to the SQL/ClickHouse retriever.
        Defaults to ``1.0`` when omitted so existing profiles without the field
        keep symmetric behavior. In ``[0.0, 1.0]``.
    """
    vector: float
    sparse: float
    structured: float
    sql: float = 1.0

    def __post_init__(self) -> None:
        for field_name in ('vector', 'sparse', 'structured', 'sql'):
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)):
                raise ConfigurationError( f"retrieval.fusion.weights.profiles.<kind>.{field_name} must be a number" )
            if not 0.0 <= float(value) <= 1.0:
                raise ConfigurationError( f"retrieval.fusion.weights.profiles.<kind>.{field_name} must be in [0.0, 1.0]" )

    @classmethod
    def from_dict(cls, d: Dict[str, Any], context: str) -> 'FusionWeightsProfile':
        _require(d, ['vector', 'sparse', 'structured'], context)
        return cls( vector=float(d['vector']), sparse=float(d['sparse']), structured=float(d['structured']), sql=float(d.get('sql', 1.0)), )


@dataclass
class FusionWeightsConfig:
    """Weighted-RRF profile bundle.

    Profiles are keyed by ``QueryIntent.residual_kind`` value. The ``default``
    profile is required and is used when ``residual_kind`` is ``None`` or
    matches no other profile. When ``enabled=false`` the fuser ignores all
    profiles and falls back to symmetric RRF (legacy behavior).

    :param enabled: bool - Master flag.
    :param profiles: Dict[str, FusionWeightsProfile] - Profile keyed by residual
        kind. When ``enabled=true``, must contain one entry per
        ``RESIDUAL_KINDS`` value plus ``'default'``.
    """
    enabled: bool
    profiles: Dict[str, FusionWeightsProfile]

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("retrieval.fusion.weights.enabled must be a bool")
        if not isinstance(self.profiles, dict):
            raise ConfigurationError("retrieval.fusion.weights.profiles must be a dict")
        if 'default' not in self.profiles:
            raise ConfigurationError( "retrieval.fusion.weights.profiles.default is required" )
        if self.enabled:
            missing = [k for k in _RESIDUAL_KINDS_SET if k not in self.profiles]
            if missing:
                raise ConfigurationError( f"retrieval.fusion.weights.profiles is missing required keys when enabled=true: {sorted(missing)}" )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FusionWeightsConfig':
        _require(d, ['enabled', 'profiles'], 'retrieval.fusion.weights')
        profiles_raw = d['profiles']
        if not isinstance(profiles_raw, dict):
            raise ConfigurationError("retrieval.fusion.weights.profiles must be a dict")
        profiles = {
            str(k): FusionWeightsProfile.from_dict(v, f"retrieval.fusion.weights.profiles.{k}")
            for k, v in profiles_raw.items()
        }
        return cls(enabled=bool(d['enabled']), profiles=profiles)


@dataclass
class FusionConfig:
    rrf_k: int
    top_n: int
    # Per-residual-kind weighted RRF. Optional/additive: when omitted or
    # ``enabled=false``, the fuser uses symmetric RRF (legacy behavior).
    weights: Optional[FusionWeightsConfig] = None

    def __post_init__(self) -> None:
        if self.rrf_k < 1:
            raise ConfigurationError("retrieval.fusion.rrf_k must be >= 1")
        if self.top_n < 1:
            raise ConfigurationError("retrieval.fusion.top_n must be >= 1")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FusionConfig':
        _require(d, ['rrf_k', 'top_n'], 'retrieval.fusion')
        weights_raw = d.get('weights')
        weights = FusionWeightsConfig.from_dict(weights_raw) if isinstance(weights_raw, dict) else None
        return cls(rrf_k=int(d['rrf_k']), top_n=int(d['top_n']), weights=weights)


_ERANKER_BACKENDS = frozenset({'noop', 'http'})


@dataclass
class ERankerHttpConfig:
    """HTTP transport for live eRanker (optional until wired)."""
    base_url: str
    timeout_seconds: float
    rank_path: str
    send_user_id: bool
    api_key_env_var: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.base_url, str) or not self.base_url.strip():
            raise ConfigurationError("retrieval.eranker.http.base_url must be a non-empty string")
        if self.base_url.endswith('/'):
            raise ConfigurationError("retrieval.eranker.http.base_url must NOT end with '/'")
        if float(self.timeout_seconds) <= 0.0:
            raise ConfigurationError("retrieval.eranker.http.timeout_seconds must be > 0")
        if not isinstance(self.rank_path, str) or not self.rank_path.strip():
            raise ConfigurationError("retrieval.eranker.http.rank_path must be a non-empty string")
        if self.rank_path.startswith('/'):
            raise ConfigurationError("retrieval.eranker.http.rank_path must not start with '/'")
        if not isinstance(self.send_user_id, bool):
            raise ConfigurationError("retrieval.eranker.http.send_user_id must be a bool")
        if self.api_key_env_var is not None and (not isinstance(self.api_key_env_var, str) or not self.api_key_env_var.strip()):
            raise ConfigurationError("retrieval.eranker.http.api_key_env_var must be a non-empty string when provided")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ERankerHttpConfig':
        _require(d, ['base_url', 'timeout_seconds', 'rank_path', 'send_user_id'], 'retrieval.eranker.http')
        return cls(
            base_url=str(d['base_url']),
            timeout_seconds=float(d['timeout_seconds']),
            rank_path=str(d['rank_path']),
            send_user_id=bool(d['send_user_id']),
            api_key_env_var=str(d['api_key_env_var']) if d.get('api_key_env_var') is not None else None,
        )


@dataclass
class ERankerConfig:
    """Layer-4 external eRanker — final ordering before diversifier and truncate."""
    enabled: bool
    backend: str
    latency_budget_ms: float
    shadow_enabled: bool
    shadow_serve_fused: bool
    skip_when_backend_unhealthy: bool
    http: Optional[ERankerHttpConfig] = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("retrieval.eranker.enabled must be a bool")
        if self.backend not in _ERANKER_BACKENDS:
            raise ConfigurationError( f"retrieval.eranker.backend must be one of {sorted(_ERANKER_BACKENDS)}; got {self.backend!r}" )
        if float(self.latency_budget_ms) <= 0.0:
            raise ConfigurationError("retrieval.eranker.latency_budget_ms must be > 0")
        if not isinstance(self.shadow_enabled, bool):
            raise ConfigurationError("retrieval.eranker.shadow_enabled must be a bool")
        if not isinstance(self.shadow_serve_fused, bool):
            raise ConfigurationError("retrieval.eranker.shadow_serve_fused must be a bool")
        if not isinstance(self.skip_when_backend_unhealthy, bool):
            raise ConfigurationError("retrieval.eranker.skip_when_backend_unhealthy must be a bool")
        if self.backend == 'http' and self.http is None:
            raise ConfigurationError("retrieval.eranker.http block is required when backend='http'")
        if self.backend == 'noop' and self.http is not None:
            raise ConfigurationError("retrieval.eranker.http must NOT be set when backend='noop'")
        if self.http is not None and float(self.http.timeout_seconds) > float(self.latency_budget_ms) / 1000.0 + 1e-6:
            raise ConfigurationError( "retrieval.eranker.http.timeout_seconds must be <= retrieval.eranker.latency_budget_ms expressed in seconds" )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ERankerConfig':
        _require( d, [ 'enabled', 'backend', 'latency_budget_ms', 'shadow_enabled', 'shadow_serve_fused', 'skip_when_backend_unhealthy', ], 'retrieval.eranker', )
        http_block = d.get('http')
        http = ERankerHttpConfig.from_dict(http_block) if isinstance(http_block, dict) else None
        return cls(
            enabled=bool(d['enabled']),
            backend=str(d['backend']),
            latency_budget_ms=float(d['latency_budget_ms']),
            shadow_enabled=bool(d['shadow_enabled']),
            shadow_serve_fused=bool(d['shadow_serve_fused']),
            skip_when_backend_unhealthy=bool(d['skip_when_backend_unhealthy']),
            http=http,
        )


_DIVERSITY_BACKENDS = frozenset({'lexical_jaccard_mmr', 'noop'})


@dataclass
class LexicalDiversityConfig:
    """Lexical-Jaccard MMR diversifier config.

    Drives ``semantic_search.retrieval.lexical_jaccard_diversifier.LexicalJaccardDiversifier``,
    a stdlib-only deterministic diversifier that re-shuffles the head of the
    post-eRanker ranked list to improve novelty in the final output
    slice. Tokenization mirrors ``BM25QueryEncoderConfig`` and
    ``LexicalRerankerConfig`` so the diversifier scores on the same token
    universe as BM25 and the lexical reranker.

    :param lambda_relevance: float - The MMR relevance-vs-novelty knob in
        ``[0, 1]``. ``1.0`` = pure relevance (degenerates to input order on
        the head); ``0.0`` = pure novelty after step 1. Production defaults
        sit in ``[0.5, 0.8]``.
    :param payload_fields: List[str] - Ordered list of ``RankedItem.payload``
        keys to concatenate into the document representation. Missing keys
        are skipped silently. Non-string values are coerced via ``str()`` so
        numerics + lists also contribute tokens.
    :param min_term_length: int - Drop tokens shorter than this (>= 1).
    :param max_terms: int - Hard per-input cap (>= 1) — bounds worst-case CPU.
    :param stopwords: List[str] - Lowercased terms dropped after tokenization.
    """
    lambda_relevance: float
    payload_fields: List[str]
    min_term_length: int
    max_terms: int
    stopwords: List[str]

    def __post_init__(self) -> None:
        lam = float(self.lambda_relevance)
        if not (0.0 <= lam <= 1.0):
            raise ConfigurationError("retrieval.diversity.lexical.lambda_relevance must be in [0, 1]")
        if not isinstance(self.payload_fields, list) or not self.payload_fields:
            raise ConfigurationError("retrieval.diversity.lexical.payload_fields must be a non-empty list")
        for fld in self.payload_fields:
            if not isinstance(fld, str) or not fld:
                raise ConfigurationError("retrieval.diversity.lexical.payload_fields entries must be non-empty strings")
        if int(self.min_term_length) < 1:
            raise ConfigurationError("retrieval.diversity.lexical.min_term_length must be >= 1")
        if int(self.max_terms) < 1:
            raise ConfigurationError("retrieval.diversity.lexical.max_terms must be >= 1")
        if not isinstance(self.stopwords, list):
            raise ConfigurationError("retrieval.diversity.lexical.stopwords must be a list")
        for sw in self.stopwords:
            if not isinstance(sw, str):
                raise ConfigurationError("retrieval.diversity.lexical.stopwords entries must be strings")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'LexicalDiversityConfig':
        _require( d, ['lambda_relevance', 'payload_fields', 'min_term_length', 'max_terms', 'stopwords'], 'retrieval.diversity.lexical', )
        return cls(
            lambda_relevance=float(d['lambda_relevance']),
            payload_fields=list(d['payload_fields']),
            min_term_length=int(d['min_term_length']),
            max_terms=int(d['max_terms']),
            stopwords=list(d['stopwords']),
        )


@dataclass
class DiversityConfig:
    """Layer-4 diversifier config.

    Runs after eRanker and before the orchestrator's ``_truncate`` cut to
    ``general.max_results``. Re-shuffles the head ``top_n`` items returned
    post-eRanker, selects ``output_n`` of them via greedy MMR, then re-stitches the
    selection onto (a) the un-selected head remainder and (b) the
    un-touched tail beyond ``top_n``. The orchestrator wraps the
    diversifier call in ``asyncio.wait_for(..., timeout=latency_budget_ms /
    1000.0)`` and on timeout / exception falls back to the
    post-eRanker order — the user-facing path is never blocked
    by diversifier degradation.

    :param enabled: bool - Master switch. False = orchestrator skips the
        diversifier entirely (uses ``NoOpDiversifier`` internally so the
        instrumentation surface stays uniform).
    :param backend: str - One of ``{'lexical_jaccard_mmr', 'noop'}``. The
        ``noop`` backend is the typed-disabled path; ``lexical_jaccard_mmr``
        is the stdlib-only default that ships in this PR. Future
        embedding-backed backends register here behind the same
        ``Diversifier`` protocol.
    :param top_n: int - Head slice length the diversifier considers as MMR
        candidates (>= 1). Items beyond ``top_n`` keep their input order.
        Should be >= ``output_n``; typically ``top_n == output_n`` (the
        diversifier picks all of the final visible slice from the head).
    :param output_n: int - Number of items the diversifier actually selects
        (>= 1, <= top_n). Typically equal to ``general.max_results`` so the
        diversified slice IS the final visible slice; setting it lower
        leaves room for downstream layers (cache write, telemetry) to see
        more candidates than the user does.
    :param latency_budget_ms: float - Hard wall on the diversifier call
        (> 0). Exceeding this triggers fallback to the post-eRanker
        order.
    :param lexical: Optional[LexicalDiversityConfig] - Required iff
        ``backend='lexical_jaccard_mmr'``. Ignored for other backends.
    """
    enabled: bool
    backend: str
    top_n: int
    output_n: int
    latency_budget_ms: float
    lexical: Optional[LexicalDiversityConfig] = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("retrieval.diversity.enabled must be a bool")
        if self.backend not in _DIVERSITY_BACKENDS:
            raise ConfigurationError( f"retrieval.diversity.backend must be one of {sorted(_DIVERSITY_BACKENDS)}; got {self.backend!r}" )
        if int(self.top_n) < 1:
            raise ConfigurationError("retrieval.diversity.top_n must be >= 1")
        if int(self.output_n) < 1:
            raise ConfigurationError("retrieval.diversity.output_n must be >= 1")
        if int(self.output_n) > int(self.top_n):
            raise ConfigurationError("retrieval.diversity.output_n must be <= retrieval.diversity.top_n")
        if float(self.latency_budget_ms) <= 0.0:
            raise ConfigurationError("retrieval.diversity.latency_budget_ms must be > 0")
        if self.enabled and self.backend == 'lexical_jaccard_mmr' and self.lexical is None:
            raise ConfigurationError( "retrieval.diversity.lexical is required when enabled=true and backend='lexical_jaccard_mmr'" )
        if self.lexical is not None and not isinstance(self.lexical, LexicalDiversityConfig):
            raise ConfigurationError("retrieval.diversity.lexical must be a LexicalDiversityConfig")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DiversityConfig':
        _require(d, ['enabled', 'backend', 'top_n', 'output_n', 'latency_budget_ms'], 'retrieval.diversity')
        lex_dict = d.get('lexical')
        lex_cfg: Optional[LexicalDiversityConfig] = None
        if lex_dict is not None:
            lex_cfg = LexicalDiversityConfig.from_dict(lex_dict)
        return cls( enabled=bool(d['enabled']), backend=str(d['backend']), top_n=int(d['top_n']), output_n=int(d['output_n']), latency_budget_ms=float(d['latency_budget_ms']), lexical=lex_cfg, )


_QDRANT_DISTANCES = frozenset({'cosine', 'dot', 'euclid'})
_QDRANT_HYBRID_FUSION = frozenset({'rrf', 'dbsf'})


@dataclass
class SynonymExpansionConfig:
    """Synonym/abbreviation expansion config for the BM25 query leg.

    The expander runs INSIDE ``BM25QueryEncoder`` between tokenization and the
    bucket-weight aggregation stage. For each surviving query token it consults
    a domain-specific synonym map and emits up to ``max_synonyms_per_token``
    extra tokens, each carrying ``expansion_weight ∈ (0, 1]`` so the synonyms
    cannot dominate the original tokens at sparse-fusion time.

    Bidirectional by construction: at load time the expander adds the inverse
    edges so a single ``ai: [artificial, intelligence]`` line yields
    ``ai -> {artificial, intelligence}`` AND ``artificial -> {ai}`` AND
    ``intelligence -> {ai}``. This avoids the double-bookkeeping bug where a
    user authors only one direction and the retrieval-side never sees the rest.

    All fields are required when the block is present. The block as a whole is
    optional on ``BM25QueryEncoderConfig`` so existing YAML continues to load
    unchanged.

    :param enabled: bool - Master switch. When False, the encoder constructs
        the expander but skips it on every call (zero overhead). When False
        AND the YAML omits the block entirely, the encoder never builds an
        expander at all.
    :param expansion_weight: float - Multiplicative weight applied to each
        expanded synonym before BM25 aggregation. Must be in (0, 1].
        Typical values: 0.3 - 0.7. Lower = synonyms are background hints;
        higher = synonyms compete with originals.
    :param max_synonyms_per_token: int - Hard cap on synonyms emitted per
        original token. Must be >= 1; typical 3 - 10. Bounds the worst-case
        post-expansion token count to ``max_terms × max_synonyms_per_token``.
    :param max_tokens_to_expand: int - Per-query cap on the number of original
        tokens the expander will consult. Must be >= 1; bounds the dictionary
        lookup work to a constant per call.
    :param synonym_map: Dict[str, List[str]] - The forward synonym map. Keys
        are lowercased single tokens; values are lists of lowercased synonym
        tokens. Empty map is valid (expander degrades to a no-op even when
        ``enabled=true``). Inverse edges are added automatically at load time
        — DO NOT also list them yourself or you will introduce duplicate
        synonyms into the post-expansion sequence.
    """
    enabled: bool
    expansion_weight: float
    max_synonyms_per_token: int
    max_tokens_to_expand: int
    synonym_map: Dict[str, List[str]]

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError( "retrieval.qdrant.hybrid.bm25_query_encoder.synonyms.enabled must be a bool" )
        if not isinstance(self.expansion_weight, (int, float)):
            raise ConfigurationError( "retrieval.qdrant.hybrid.bm25_query_encoder.synonyms.expansion_weight must be a number" )
        if not 0.0 < float(self.expansion_weight) <= 1.0:
            raise ConfigurationError("retrieval.qdrant.hybrid.bm25_query_encoder.synonyms.expansion_weight must be in (0, 1]")
        if not isinstance(self.max_synonyms_per_token, int) or self.max_synonyms_per_token < 1:
            raise ConfigurationError( "retrieval.qdrant.hybrid.bm25_query_encoder.synonyms.max_synonyms_per_token must be int >= 1" )
        if not isinstance(self.max_tokens_to_expand, int) or self.max_tokens_to_expand < 1:
            raise ConfigurationError( "retrieval.qdrant.hybrid.bm25_query_encoder.synonyms.max_tokens_to_expand must be int >= 1" )
        if not isinstance(self.synonym_map, dict):
            raise ConfigurationError( "retrieval.qdrant.hybrid.bm25_query_encoder.synonyms.synonym_map must be a dict" )
        for key, vals in self.synonym_map.items():
            if not isinstance(key, str) or not key:
                raise ConfigurationError( "retrieval.qdrant.hybrid.bm25_query_encoder.synonyms.synonym_map keys must be non-empty strings" )
            if not isinstance(vals, list):
                raise ConfigurationError( f"retrieval.qdrant.hybrid.bm25_query_encoder.synonyms.synonym_map['{key}'] must be a list" )
            for v in vals:
                if not isinstance(v, str) or not v:
                    raise ConfigurationError( f"retrieval.qdrant.hybrid.bm25_query_encoder.synonyms.synonym_map['{key}'] entries must be non-empty strings" )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SynonymExpansionConfig':
        _require( d, ['enabled', 'expansion_weight', 'max_synonyms_per_token', 'max_tokens_to_expand', 'synonym_map'], 'retrieval.qdrant.hybrid.bm25_query_encoder.synonyms', )
        return cls(
            enabled=bool(d['enabled']),
            expansion_weight=float(d['expansion_weight']),
            max_synonyms_per_token=int(d['max_synonyms_per_token']),
            max_tokens_to_expand=int(d['max_tokens_to_expand']),
            synonym_map=dict(d['synonym_map']),
        )


@dataclass
class DynamicSynonymConfig:
    """Query-driven dynamic synonym store config for the BM25 sparse leg.

    Controls the SQLite-backed synonym store that learns new token->synonym
    mappings at query time. Every token that misses BOTH the static map
    AND the dynamic store is queued; once it accumulates ``min_freq_to_expand``
    misses the background expander fires an LLM call and writes the result
    back as a permanent bidirectional entry.

    :param enabled: bool - Master switch.
    :param db_path: str - SQLite file path. ``${DATA_DIR}`` env-var expansion
        is applied by the config loader so the path stays machine-portable.
    :param miss_queue_max: int - Deque capacity for pending LLM calls. Tokens
        beyond this cap are silently dropped until the queue drains.
    :param min_freq_to_expand: int - Miss-count threshold before a token is
        enqueued. Prevents one-off typos from burning LLM budget.
    :param expansion_concurrency: int - Parallel LLM calls in the background
        expander loop (>= 1).
    :param expansion_weight: float - BM25 multiplier for dynamic synonyms,
        same semantics as ``SynonymExpansionConfig.expansion_weight``. Must
        be in (0, 1].
    :param max_synonyms_per_token: int - Cap on dynamic synonyms emitted per
        original token per query (>= 1).
    :param max_db_entries: int - Hard cap on rows in the SQLite table. When
        this limit is reached, the oldest entries (by last_updated) are removed
        before writing new ones so disk usage stays bounded.
    :param min_confidence: float - Confidence gate applied to LLM-generated
        synonyms before they are persisted via ``DynamicSynonymStore.put``.
        Each candidate synonym carries an LLM-reported confidence in [0, 1];
        candidates below this floor are dropped so low-quality pairs never
        enter the permanent store. Must be in [0, 1]; ``0.0`` disables the gate.
    :param shutdown_grace_timeout_seconds: float - How long the API shutdown handler
        waits for the background expansion task to finish cleanly before cancelling it.
        Must be > 0.
    """
    enabled: bool
    db_path: str
    miss_queue_max: int
    min_freq_to_expand: int
    expansion_concurrency: int
    expansion_weight: float
    max_synonyms_per_token: int
    max_db_entries: int
    min_confidence: float
    shutdown_grace_timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("dynamic_synonyms.enabled must be a bool")
        if not isinstance(self.db_path, str) or not self.db_path:
            raise ConfigurationError("dynamic_synonyms.db_path must be a non-empty string")
        if not isinstance(self.miss_queue_max, int) or self.miss_queue_max < 1:
            raise ConfigurationError("dynamic_synonyms.miss_queue_max must be int >= 1")
        if not isinstance(self.min_freq_to_expand, int) or self.min_freq_to_expand < 1:
            raise ConfigurationError("dynamic_synonyms.min_freq_to_expand must be int >= 1")
        if not isinstance(self.expansion_concurrency, int) or self.expansion_concurrency < 1:
            raise ConfigurationError("dynamic_synonyms.expansion_concurrency must be int >= 1")
        if not isinstance(self.expansion_weight, (int, float)):
            raise ConfigurationError("dynamic_synonyms.expansion_weight must be a number")
        if not 0.0 < float(self.expansion_weight) <= 1.0:
            raise ConfigurationError("dynamic_synonyms.expansion_weight must be in (0, 1]")
        if not isinstance(self.max_synonyms_per_token, int) or self.max_synonyms_per_token < 1:
            raise ConfigurationError("dynamic_synonyms.max_synonyms_per_token must be int >= 1")
        if not isinstance(self.max_db_entries, int) or self.max_db_entries < 1:
            raise ConfigurationError("dynamic_synonyms.max_db_entries must be int >= 1")
        if not isinstance(self.min_confidence, (int, float)) or isinstance(self.min_confidence, bool):
            raise ConfigurationError("dynamic_synonyms.min_confidence must be a number")
        if not 0.0 <= float(self.min_confidence) <= 1.0:
            raise ConfigurationError("dynamic_synonyms.min_confidence must be in [0, 1]")
        if not isinstance(self.shutdown_grace_timeout_seconds, (int, float)) or float(self.shutdown_grace_timeout_seconds) <= 0:
            raise ConfigurationError("dynamic_synonyms.shutdown_grace_timeout_seconds must be a positive number")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DynamicSynonymConfig':
        _require(d, ['enabled', 'db_path', 'miss_queue_max', 'min_freq_to_expand', 'expansion_concurrency', 'expansion_weight', 'max_synonyms_per_token', 'max_db_entries', 'min_confidence'], 'dynamic_synonyms')  # noqa: E501
        return cls(
            enabled=bool(d['enabled']),
            db_path=str(d['db_path']),
            miss_queue_max=int(d['miss_queue_max']),
            min_freq_to_expand=int(d['min_freq_to_expand']),
            expansion_concurrency=int(d['expansion_concurrency']),
            expansion_weight=float(d['expansion_weight']),
            max_synonyms_per_token=int(d['max_synonyms_per_token']),
            max_db_entries=int(d['max_db_entries']),
            min_confidence=float(d['min_confidence']),
            shutdown_grace_timeout_seconds=float(d.get('shutdown_grace_timeout_seconds', 5.0)),
        )


@dataclass
class BM25QueryEncoderConfig:
    """BM25 sparse query-encoder config (registered as ``bm25_query_fn``).

    Consumed by ``semantic_search.retrieval.bm25_query_encoder.BM25QueryEncoder``,
    which the registry instantiates as the ``bm25_query_fn`` callable handed to
    ``QdrantHybridRetriever`` whenever ``retrieval.qdrant.hybrid.bm25_enabled``
    is True. The same hashing scheme (``vocab_size`` + sha1-mod) MUST be used at
    corpus ingest so query-side and index-side term ids align.

    :param vocab_size: int - Hashing vocabulary cardinality (>=1024); also caps
        the SparseVector index range. Mirror at ingest. 262144 is a sane prod
        starting point for the auctions corpus.
    :param min_term_length: int - Drop tokens shorter than this after lowercase
        ``[a-z0-9]+`` extraction (>=1, typically 2-3).
    :param max_terms: int - Hard cap on tokens passed to the hasher per query
        (>=1). Bounds worst-case sparse-vector size + per-query CPU.
    :param stopwords: List[str] - Tokens dropped before hashing (case-insensitive).
        Domain-specific; defaults to a minimal English set.
    :param synonyms: Optional[SynonymExpansionConfig] - Static synonym /
        abbreviation expansion for the BM25 leg. Optional; ``None`` (or
        ``enabled=False``) means the encoder skips static expansion entirely.
        When present and enabled, the encoder emits down-weighted synonym terms
        alongside each original token before BM25 aggregation.
    :param dynamic_synonyms: Optional[DynamicSynonymConfig] - Query-driven
        dynamic synonym store. When present and enabled, tokens missing the
        static map are looked up in the SQLite-backed bidirectional store and,
        if absent there too, queued for background LLM expansion. The store
        grows permanently — entries are never evicted.
    """
    vocab_size: int
    min_term_length: int
    max_terms: int
    stopwords: List[str]
    synonyms: Optional[SynonymExpansionConfig] = None
    dynamic_synonyms: Optional[DynamicSynonymConfig] = None

    def __post_init__(self) -> None:
        if not isinstance(self.vocab_size, int) or self.vocab_size < 1024:
            raise ConfigurationError( "retrieval.qdrant.hybrid.bm25_query_encoder.vocab_size must be int >= 1024" )
        if not isinstance(self.min_term_length, int) or self.min_term_length < 1:
            raise ConfigurationError( "retrieval.qdrant.hybrid.bm25_query_encoder.min_term_length must be int >= 1" )
        if not isinstance(self.max_terms, int) or self.max_terms < 1:
            raise ConfigurationError( "retrieval.qdrant.hybrid.bm25_query_encoder.max_terms must be int >= 1" )
        if not isinstance(self.stopwords, list):
            raise ConfigurationError( "retrieval.qdrant.hybrid.bm25_query_encoder.stopwords must be a list" )
        for s in self.stopwords:
            if not isinstance(s, str):
                raise ConfigurationError( "retrieval.qdrant.hybrid.bm25_query_encoder.stopwords entries must be strings" )
        if self.synonyms is not None and not isinstance(self.synonyms, SynonymExpansionConfig):
            raise ConfigurationError( "retrieval.qdrant.hybrid.bm25_query_encoder.synonyms must be a SynonymExpansionConfig or None" )
        if self.dynamic_synonyms is not None and not isinstance(self.dynamic_synonyms, DynamicSynonymConfig):
            raise ConfigurationError( "retrieval.qdrant.hybrid.bm25_query_encoder.dynamic_synonyms must be a DynamicSynonymConfig or None" )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'BM25QueryEncoderConfig':
        _require( d, ['vocab_size', 'min_term_length', 'max_terms', 'stopwords'], 'retrieval.qdrant.hybrid.bm25_query_encoder', )
        synonyms_raw = d.get('synonyms')
        synonyms = SynonymExpansionConfig.from_dict(synonyms_raw) if isinstance(synonyms_raw, dict) else None
        dyn_raw = d.get('dynamic_synonyms')
        dynamic_synonyms = DynamicSynonymConfig.from_dict(dyn_raw) if isinstance(dyn_raw, dict) else None
        return cls(
            vocab_size=int(d['vocab_size']),
            min_term_length=int(d['min_term_length']),
            max_terms=int(d['max_terms']),
            stopwords=list(d['stopwords']),
            synonyms=synonyms,
            dynamic_synonyms=dynamic_synonyms,
        )


@dataclass
class BM42QueryStopListConfig:
    """Query-side navigational stop list applied inside ``BM42SparseEncoder.__call__``.

    When ``enabled=true``, tokens in ``tokens`` (case-insensitive) are removed
    from the query string BEFORE it is passed to BM42's ``query_embed``.
    Doc-side encoding is NOT modified — corpus integrity is preserved.

    :param enabled: bool - Master flag.
    :param tokens: List[str] - Lowercased stop tokens. May be empty when
        ``enabled=false``; must be non-empty when ``enabled=true``.
    """
    enabled: bool
    tokens: List[str]

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("sparse_encoder.query_stop_list.enabled must be a bool")
        if not isinstance(self.tokens, list):
            raise ConfigurationError("sparse_encoder.query_stop_list.tokens must be a list")
        if self.enabled and len(self.tokens) == 0:
            raise ConfigurationError( "sparse_encoder.query_stop_list.tokens must be non-empty when enabled=true" )
        for token in self.tokens:
            if not isinstance(token, str) or not token:
                raise ConfigurationError( "sparse_encoder.query_stop_list.tokens entries must be non-empty strings" )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'BM42QueryStopListConfig':
        _require(d, ['enabled', 'tokens'], 'sparse_encoder.query_stop_list')
        return cls( enabled=bool(d['enabled']), tokens=[str(t).lower() for t in d['tokens']], )


@dataclass
class SparseEncoderConfig:
    """Config for the BM42 attention-weighted sparse encoder.

    When present under ``retrieval.qdrant.hybrid.sparse_encoder``, the registry
    attempts to load a local ``SparseTextEmbedding`` model and uses it for both
    the query and document sparse legs. Hash-BM25 fallback is gated by
    ``hash_bm25_fallback`` (required YAML bool — no silent default).

    :param model_name: str - FastEmbed registry name
        (e.g. ``Qdrant/bm42-all-minilm-l6-v2-attentions``).
    :param local_model_path: str - Absolute path to the pre-downloaded model directory.
        No network download is ever attempted.
    :param threads: int - ONNX-Runtime intra-op thread count; ``0`` means library
        default (auto-detected from CPU count).
    :param hash_bm25_fallback: bool - When true, registry falls back to
        ``bm25_query_encoder`` (hash-BM25) if BM42 load fails. When false, boot
        fails loud so a missing BM42 model cannot silently degrade sparse recall.
    :param query_stop_list: Optional[BM42QueryStopListConfig] - Query-side
        navigational stop list. Optional/additive: when omitted or
        ``enabled=false``, the encoder receives the raw stripped query unchanged.
    """
    model_name: str
    local_model_path: str
    threads: int
    hash_bm25_fallback: bool
    query_stop_list: Optional[BM42QueryStopListConfig] = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name:
            raise ConfigurationError( "sparse_encoder.model_name must be a non-empty string" )
        if not isinstance(self.local_model_path, str):
            raise ConfigurationError( "sparse_encoder.local_model_path must be a string" )
        if not isinstance(self.threads, int) or self.threads < 0:
            raise ConfigurationError( "sparse_encoder.threads must be an integer >= 0 (0 = library default)" )
        if not isinstance(self.hash_bm25_fallback, bool):
            raise ConfigurationError( "sparse_encoder.hash_bm25_fallback must be a bool" )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SparseEncoderConfig':
        _require(d, ['model_name', 'local_model_path', 'threads', 'hash_bm25_fallback'], 'sparse_encoder')
        stop_raw = d.get('query_stop_list')
        query_stop_list = BM42QueryStopListConfig.from_dict(stop_raw) if isinstance(stop_raw, dict) else None
        return cls(
            model_name=str(d['model_name']),
            local_model_path=str(d['local_model_path']),
            threads=int(d['threads']),
            hash_bm25_fallback=bool(d['hash_bm25_fallback']),
            query_stop_list=query_stop_list,
        )


@dataclass
class QdrantRerankConfig:
    """Second-stage dense rescore over a higher-dim dense vector field.
    The hybrid retriever fuses dense(shortlist)+sparse server-side, then reranks
    the fused pool by a separate, higher-fidelity dense vector (e.g. 768) stored
    on the same point. The offline indexer writes the rerank vector under
    ``vector_name``; the retriever issues a nested-prefetch query that reranks
    by ``vector_name`` at ``dim``.
    :param enabled: bool - Master switch for the rerank stage
    :param vector_name: str - Named dense vector field carrying the rerank embedding
    :param dim: int - Rerank embedding dim (e.g. 768); MUST equal cascade.stage_dims.rerank
    :param input_n: int - Fused-pool size handed from the fusion prefetch into the rerank stage
    """
    enabled: bool
    vector_name: str
    dim: int
    input_n: int

    def __post_init__(self) -> None:
        if not isinstance(self.vector_name, str):
            raise ConfigurationError("retrieval.qdrant.hybrid.rerank.vector_name must be a string")
        if self.enabled and not self.vector_name:
            raise ConfigurationError("retrieval.qdrant.hybrid.rerank.vector_name must be non-empty when enabled=true")
        if self.enabled and (not isinstance(self.dim, int) or self.dim < 4):
            raise ConfigurationError("retrieval.qdrant.hybrid.rerank.dim must be int >= 4 when enabled=true")
        if self.enabled and (not isinstance(self.input_n, int) or self.input_n < 1):
            raise ConfigurationError("retrieval.qdrant.hybrid.rerank.input_n must be int >= 1 when enabled=true")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QdrantRerankConfig':
        _require(d, ['enabled', 'vector_name', 'dim', 'input_n'], 'retrieval.qdrant.hybrid.rerank')
        return cls(enabled=bool(d['enabled']), vector_name=str(d['vector_name']), dim=int(d['dim']), input_n=int(d['input_n']))


@dataclass
class QdrantNgramConfig:
    """Character n-gram sparse channel for fuzzy / misspell recall.
    Adds a dedicated sparse vector leg to the hybrid RRF query so near-miss and
    misspelled queries retrieve lexically-close domains the dense + BM42 token
    legs miss (e.g. ``high rentals`` -> ``hi-rentals`` / ``rent.high``). Doc and
    query sides hash boundary-marked character n-grams into a shared bucket
    space. Requires a reindex that writes the named sparse vector.
    :param enabled: bool - Master switch for the n-gram recall channel
    :param vector_name: str - Named sparse vector field on the collection (e.g. 'ngram')
    :param vocab_size: int - Hashing bucket cardinality; doc + query share it
    :param min_n: int - Minimum character n-gram length (>= 1)
    :param max_n: int - Maximum character n-gram length (>= min_n)
    """
    enabled: bool
    vector_name: str
    vocab_size: int
    min_n: int
    max_n: int

    def __post_init__(self) -> None:
        if not isinstance(self.vector_name, str):
            raise ConfigurationError("retrieval.qdrant.hybrid.ngram.vector_name must be a string")
        if self.enabled and not self.vector_name:
            raise ConfigurationError("retrieval.qdrant.hybrid.ngram.vector_name must be non-empty when enabled=true")
        if self.enabled:
            if not isinstance(self.vocab_size, int) or self.vocab_size < 1:
                raise ConfigurationError("retrieval.qdrant.hybrid.ngram.vocab_size must be int >= 1 when enabled=true")
            if not isinstance(self.min_n, int) or self.min_n < 1:
                raise ConfigurationError("retrieval.qdrant.hybrid.ngram.min_n must be int >= 1 when enabled=true")
            if not isinstance(self.max_n, int) or self.max_n < self.min_n:
                raise ConfigurationError("retrieval.qdrant.hybrid.ngram.max_n must be int >= min_n when enabled=true")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QdrantNgramConfig':
        _require(d, ['enabled', 'vector_name', 'vocab_size', 'min_n', 'max_n'], 'retrieval.qdrant.hybrid.ngram')
        return cls(enabled=bool(d['enabled']), vector_name=str(d['vector_name']), vocab_size=int(d['vocab_size']), min_n=int(d['min_n']), max_n=int(d['max_n']))


@dataclass
class QdrantHybridConfig:
    """Qdrant single-round-trip hybrid retrieval (vector + payload filter [+ BM25]).

    When ``enabled=true``, the registry constructs a
    `QdrantHybridRetriever` and reports `source='vector'` to the orchestrator's
    fan-out logic. The structured retriever is replaced by a no-op for query
    types that the unified index already covers, so `SearchOrchestrator` keeps
    its existing contract (no changes to `_gather_candidates`).

    :param enabled: bool - Master switch for the hybrid path. When False the
        Qdrant vector + structured backends behave like classic single-modality
        backends and `SearchOrchestrator` fans out as before.
    :param fusion_strategy: str - Server-side fusion mode for dense + sparse
        prefetches (`'rrf'` = Reciprocal Rank Fusion, `'dbsf'` = Distribution-
        Based Score Fusion). Ignored when `bm25_enabled=false`.
    :param bm25_enabled: bool - Issue a sparse-vector prefetch alongside the
        dense vector prefetch and let Qdrant fuse them server-side. Requires
        Qdrant ≥1.10 and a sparse vector configured on the collection under
        `bm25_vector_name`.
    :param bm25_vector_name: str - Name of the sparse vector field on the
        collection (only consulted when `bm25_enabled=true`).
    :param dense_vector_name: str - Name of the dense vector field on the
        collection (always consulted; '' = unnamed default vector).
    :param prefetch_limit: int - Per-prefetch candidate cap (server-side
        prefetches feeding the fusion stage). Must be >= top_k of the parent
        retriever — enforced at registry-build time.
    :param bm25_query_encoder: Optional[BM25QueryEncoderConfig] - REQUIRED when
        ``bm25_enabled=true``; ignored otherwise. Drives the registry-level
        construction of the ``bm25_query_fn`` callable.
    :param sparse_encoder: Optional[SparseEncoderConfig] - When present the
        registry attempts to load the BM42 model and use it for both query and
        doc sparse encoding. Fallback to hash-BM25 is controlled by
        ``sparse_encoder.hash_bm25_fallback`` (required bool; no silent default).
    :param ngram: Optional[QdrantNgramConfig] - Character n-gram sparse channel
        for fuzzy / misspell recall. When present and ``enabled=true`` the
        registry wires a ``CharNgramSparseEncoder`` into both the offline indexer
        (doc side) and the hybrid retriever (query side); the retriever adds a
        third RRF prefetch leg on the named sparse vector.
    :param kw_post_oversample_factor: int - Candidate oversample multiplier applied
        to ``top_k`` and ``prefetch_limit`` when a ``keyword_contains`` or
        ``keyword_contains_exclude`` post-filter is active. Python-side substring
        post-filters run after Qdrant returns, so a wider pool is needed to
        deliver ``top_k`` matching results. Must be >= 1.
    :param kw_prefix_oversample_factor: int - Candidate oversample multiplier
        applied when a ``keyword_starts_with`` or ``keyword_ends_with`` post-filter
        is active. Prefix/suffix constraints are more selective than containment,
        so this should be >= ``kw_post_oversample_factor``. Must be >= 1.
    """
    enabled: bool
    fusion_strategy: str
    bm25_enabled: bool
    bm25_vector_name: str
    dense_vector_name: str
    prefetch_limit: int
    kw_post_oversample_factor: int
    kw_prefix_oversample_factor: int
    bm25_query_encoder: Optional[BM25QueryEncoderConfig] = None
    sparse_encoder: Optional[SparseEncoderConfig] = None
    rerank: Optional[QdrantRerankConfig] = None
    ngram: Optional[QdrantNgramConfig] = None

    def __post_init__(self) -> None:
        if self.fusion_strategy not in _QDRANT_HYBRID_FUSION:
            raise ConfigurationError( f"retrieval.qdrant.hybrid.fusion_strategy must be one of {sorted(_QDRANT_HYBRID_FUSION)}" )
        if not isinstance(self.bm25_vector_name, str):
            raise ConfigurationError("retrieval.qdrant.hybrid.bm25_vector_name must be a string")
        if self.bm25_enabled and not self.bm25_vector_name:
            raise ConfigurationError( "retrieval.qdrant.hybrid.bm25_vector_name must be non-empty when bm25_enabled=true" )
        if not isinstance(self.dense_vector_name, str):
            raise ConfigurationError("retrieval.qdrant.hybrid.dense_vector_name must be a string")
        if self.prefetch_limit < 1:
            raise ConfigurationError("retrieval.qdrant.hybrid.prefetch_limit must be >= 1")
        if int(self.kw_post_oversample_factor) < 1:
            raise ConfigurationError("retrieval.qdrant.hybrid.kw_post_oversample_factor must be >= 1")
        if int(self.kw_prefix_oversample_factor) < 1:
            raise ConfigurationError("retrieval.qdrant.hybrid.kw_prefix_oversample_factor must be >= 1")
        if self.bm25_enabled and self.bm25_query_encoder is None:
            raise ConfigurationError( "retrieval.qdrant.hybrid.bm25_query_encoder is required when bm25_enabled=true" )
        if self.bm25_query_encoder is not None and not isinstance(self.bm25_query_encoder, BM25QueryEncoderConfig):
            raise ConfigurationError( "retrieval.qdrant.hybrid.bm25_query_encoder must be a BM25QueryEncoderConfig" )
        if self.rerank is not None and not isinstance(self.rerank, QdrantRerankConfig):
            raise ConfigurationError( "retrieval.qdrant.hybrid.rerank must be a QdrantRerankConfig" )
        if self.ngram is not None and not isinstance(self.ngram, QdrantNgramConfig):
            raise ConfigurationError( "retrieval.qdrant.hybrid.ngram must be a QdrantNgramConfig" )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QdrantHybridConfig':
        _require( d, ['enabled', 'fusion_strategy', 'bm25_enabled', 'bm25_vector_name', 'dense_vector_name', 'prefetch_limit', 'kw_post_oversample_factor', 'kw_prefix_oversample_factor'], 'retrieval.qdrant.hybrid', )  # noqa: E501
        encoder_cfg: Optional[BM25QueryEncoderConfig] = None
        encoder_dict = d.get('bm25_query_encoder')
        if encoder_dict is not None:
            encoder_cfg = BM25QueryEncoderConfig.from_dict(encoder_dict)
        sparse_enc_cfg: Optional[SparseEncoderConfig] = None
        sparse_enc_dict = d.get('sparse_encoder')
        if sparse_enc_dict is not None:
            sparse_enc_cfg = SparseEncoderConfig.from_dict(sparse_enc_dict)
        rerank_cfg: Optional[QdrantRerankConfig] = None
        rerank_dict = d.get('rerank')
        if rerank_dict is not None:
            rerank_cfg = QdrantRerankConfig.from_dict(rerank_dict)
        ngram_cfg: Optional[QdrantNgramConfig] = None
        ngram_dict = d.get('ngram')
        if ngram_dict is not None:
            ngram_cfg = QdrantNgramConfig.from_dict(ngram_dict)
        return cls(
            enabled=bool(d['enabled']),
            fusion_strategy=str(d['fusion_strategy']),
            bm25_enabled=bool(d['bm25_enabled']),
            bm25_vector_name=str(d['bm25_vector_name']),
            dense_vector_name=str(d['dense_vector_name']),
            prefetch_limit=int(d['prefetch_limit']),
            kw_post_oversample_factor=int(d['kw_post_oversample_factor']),
            kw_prefix_oversample_factor=int(d['kw_prefix_oversample_factor']),
            bm25_query_encoder=encoder_cfg,
            sparse_encoder=sparse_enc_cfg,
            rerank=rerank_cfg,
            ngram=ngram_cfg,
        )


_FILTER_ONLY_ORDER_DIRECTIONS = frozenset({'asc', 'desc'})


@dataclass
class FilterOnlyRailLegConfig:
    """One ordered Qdrant scroll leg for filter-only RRF merge.

    :param rail_id: str - Stable leg id (logs / payload annotation)
    :param order_by_field: str - Indexed payload field for ``OrderBy.key``
    :param order_by_direction: str - ``asc`` or ``desc``
    :param limit_multiplier: int - Scroll limit = top_k * this (>= 1)
    """
    rail_id: str
    order_by_field: str
    order_by_direction: str
    limit_multiplier: int

    def __post_init__(self) -> None:
        _p = "retrieval.qdrant.filter_only_rails.legs[*]"
        if not isinstance(self.rail_id, str) or not self.rail_id.strip():
            raise ConfigurationError(f"{_p}.rail_id must be a non-empty string")
        if not isinstance(self.order_by_field, str) or not self.order_by_field.strip():
            raise ConfigurationError(f"{_p}.order_by_field must be a non-empty string")
        direction = str(self.order_by_direction).strip().lower()
        if direction not in _FILTER_ONLY_ORDER_DIRECTIONS:
            raise ConfigurationError(
                f"{_p}.order_by_direction must be one of {sorted(_FILTER_ONLY_ORDER_DIRECTIONS)}; "
                f"got {self.order_by_direction!r}"
            )
        self.order_by_direction = direction
        if int(self.limit_multiplier) < 1:
            raise ConfigurationError(f"{_p}.limit_multiplier must be >= 1")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FilterOnlyRailLegConfig':
        _p = "retrieval.qdrant.filter_only_rails.legs[*]"
        _require(d, ['rail_id', 'order_by_field', 'order_by_direction', 'limit_multiplier'], _p)
        return cls(
            rail_id=str(d['rail_id']).strip(),
            order_by_field=str(d['order_by_field']).strip(),
            order_by_direction=str(d['order_by_direction']).strip(),
            limit_multiplier=int(d['limit_multiplier']),
        )


@dataclass
class FilterOnlyRailsConfig:
    """Multi-scroll RRF for empty-encode / pure-filter hybrid retrieve.

    When enabled, each configured leg scrolls the same hard ``qfilter`` with a
    distinct ``order_by``, then legs are RRF-merged. Hard filters stay on every
    scroll — ranking never widens the match set.

    :param enabled: bool - Master toggle for multi-scroll RRF
    :param rrf_k: int - RRF constant k (>= 1)
    :param legs: List[FilterOnlyRailLegConfig] - Ordered scroll legs (non-empty)
    """
    enabled: bool
    rrf_k: int
    legs: List[FilterOnlyRailLegConfig]

    def __post_init__(self) -> None:
        _p = "retrieval.qdrant.filter_only_rails"
        if not isinstance(self.enabled, bool):
            raise ConfigurationError(f"{_p}.enabled must be a bool")
        if int(self.rrf_k) < 1:
            raise ConfigurationError(f"{_p}.rrf_k must be >= 1")
        if not isinstance(self.legs, list) or not self.legs:
            raise ConfigurationError(f"{_p}.legs must be a non-empty list")
        seen: set = set()
        for leg in self.legs:
            if not isinstance(leg, FilterOnlyRailLegConfig):
                raise ConfigurationError(f"{_p}.legs entries must be FilterOnlyRailLegConfig")
            if leg.rail_id in seen:
                raise ConfigurationError(f"{_p}.legs duplicate rail_id={leg.rail_id!r}")
            seen.add(leg.rail_id)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FilterOnlyRailsConfig':
        _p = "retrieval.qdrant.filter_only_rails"
        _require(d, ['enabled', 'rrf_k', 'legs'], _p)
        raw_legs = d['legs']
        if not isinstance(raw_legs, list):
            raise ConfigurationError(f"{_p}.legs must be a list")
        return cls(
            enabled=bool(d['enabled']),
            rrf_k=int(d['rrf_k']),
            legs=[FilterOnlyRailLegConfig.from_dict(leg) for leg in raw_legs],
        )


@dataclass
class QdrantConfig:
    """Qdrant client + collection settings for the unified listing index.

    Construction of the Qdrant client (in `QdrantClientFactory`) is
    soft-failure: a misconfigured or unreachable cluster does not crash the
    `build_subsystems` boot path; the registry logs a warning and falls back
    to the in-memory backends. Methods on the adapters raise
    `QdrantUnavailableError` when invoked against an unavailable client so the
    `BackendHealthRegistry` can drop them deterministically.

    :param host: str - Qdrant host (REST + gRPC dispatch via `qdrant-client`).
    :param port: int - REST port (1..65535). gRPC is enabled separately.
    :param grpc_port: int - gRPC port (1..65535).
    :param prefer_grpc: bool - Prefer gRPC for high-throughput points ops.
    :param https: bool - Use TLS for the REST channel.
    :param api_key_env_var: str - Name of the OS env var carrying the Qdrant
        API key. Empty = no auth (dev/local). The key itself is NEVER read
        from YAML (`responsible-ai.mdc` §secret-handling).
    :param collection_name: str - Target collection name.
    :param payload_id_field: str - Payload field used as the canonical
        `Candidate.item_id` when the point id is a numeric/UUID surrogate.
        When empty, the raw point id is stringified.
    :param payload_score_field: str - Payload field carrying the listing-side
        relevance score returned by `QdrantStructuredIndex.search`. Empty =
        derive a deterministic score from the rank position.
    :param connect_timeout_seconds: float - TCP connect timeout.
    :param read_timeout_seconds: float - Per-request read timeout.
    :param hnsw_ef_search: int - HNSW search-time candidate width
        (`SearchParams.hnsw_ef`). Lower = faster, less recall.
    :param hybrid: QdrantHybridConfig - Hybrid retrieval policy (Option A).
    :param filter_only_rails: FilterOnlyRailsConfig - Multi-scroll RRF for
        empty-encode filter-only retrieve (ending-soon / trending / value legs).
    :param check_compatibility: bool - Forwarded to `AsyncQdrantClient`. When
        False, the client skips its server-version proximity gate so the server
        image can drift from the pinned client version without a warning.
    """
    host: str
    port: int
    grpc_port: int
    prefer_grpc: bool
    https: bool
    api_key_env_var: str
    collection_name: str
    payload_id_field: str
    payload_score_field: str
    connect_timeout_seconds: float
    read_timeout_seconds: float
    hnsw_ef_search: int
    hybrid: QdrantHybridConfig
    filter_only_rails: FilterOnlyRailsConfig
    check_compatibility: bool
    active_only_baseline: bool = False
    price_gt_zero_baseline: bool = False
    starting_bid_gt_zero_baseline: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.host, str) or not self.host:
            raise ConfigurationError("retrieval.qdrant.host must be a non-empty string")
        if not 1 <= int(self.port) <= 65535:
            raise ConfigurationError("retrieval.qdrant.port must be in [1, 65535]")
        if not 1 <= int(self.grpc_port) <= 65535:
            raise ConfigurationError("retrieval.qdrant.grpc_port must be in [1, 65535]")
        if not isinstance(self.api_key_env_var, str):
            raise ConfigurationError("retrieval.qdrant.api_key_env_var must be a string (may be empty for no auth)")
        if not isinstance(self.collection_name, str) or not self.collection_name:
            raise ConfigurationError("retrieval.qdrant.collection_name must be a non-empty string")
        if not isinstance(self.payload_id_field, str):
            raise ConfigurationError("retrieval.qdrant.payload_id_field must be a string (may be empty)")
        if not isinstance(self.payload_score_field, str):
            raise ConfigurationError("retrieval.qdrant.payload_score_field must be a string (may be empty)")
        if float(self.connect_timeout_seconds) <= 0.0:
            raise ConfigurationError("retrieval.qdrant.connect_timeout_seconds must be > 0")
        if float(self.read_timeout_seconds) <= 0.0:
            raise ConfigurationError("retrieval.qdrant.read_timeout_seconds must be > 0")
        if int(self.hnsw_ef_search) < 1:
            raise ConfigurationError("retrieval.qdrant.hnsw_ef_search must be >= 1")
        if not isinstance(self.hybrid, QdrantHybridConfig):
            raise ConfigurationError("retrieval.qdrant.hybrid must be a QdrantHybridConfig")
        if not isinstance(self.filter_only_rails, FilterOnlyRailsConfig):
            raise ConfigurationError("retrieval.qdrant.filter_only_rails must be a FilterOnlyRailsConfig")
        if not isinstance(self.check_compatibility, bool):
            raise ConfigurationError("retrieval.qdrant.check_compatibility must be a bool")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QdrantConfig':
        _require(
            d,
            [
                'host', 'port', 'grpc_port', 'prefer_grpc', 'https', 'api_key_env_var',
                'collection_name', 'payload_id_field', 'payload_score_field',
                'connect_timeout_seconds', 'read_timeout_seconds', 'hnsw_ef_search', 'hybrid',
                'filter_only_rails',
                'check_compatibility',
            ],
            'retrieval.qdrant',
        )
        return cls(
            host=str(d['host']),
            port=int(d['port']),
            grpc_port=int(d['grpc_port']),
            prefer_grpc=_coerce_bool(d['prefer_grpc']),
            https=_coerce_bool(d['https']),
            api_key_env_var=str(d['api_key_env_var']),
            collection_name=str(d['collection_name']),
            payload_id_field=str(d['payload_id_field']),
            payload_score_field=str(d['payload_score_field']),
            connect_timeout_seconds=float(d['connect_timeout_seconds']),
            read_timeout_seconds=float(d['read_timeout_seconds']),
            hnsw_ef_search=int(d['hnsw_ef_search']),
            hybrid=QdrantHybridConfig.from_dict(d['hybrid']),
            filter_only_rails=FilterOnlyRailsConfig.from_dict(d['filter_only_rails']),
            check_compatibility=bool(d['check_compatibility']),
            active_only_baseline=bool(d.get('active_only_baseline', False)),
            price_gt_zero_baseline=bool(d.get('price_gt_zero_baseline', False)),
            starting_bid_gt_zero_baseline=bool(d.get('starting_bid_gt_zero_baseline', False)),
        )


@dataclass
class SkipVectorWhenConfig:
    """Predicate that selects which residual kinds and chip-coverage state
    cause the vector backend to be skipped from the retrieval fan-out.

    :param residual_kinds: List[str] - Subset of ``RESIDUAL_KINDS`` that
        triggers the skip. ``empty`` and ``navigational`` are typical values.
    :param require_hard_chip_coverage: bool - When True, the skip only fires
        when ``extract_filters_from_intent(intent)`` yields ≥1 hard chip. When
        False, the skip fires on residual_kind alone (more aggressive — risks
        empty result sets when no filter is present).
    :param hybrid_mode_residual_kinds: List[str] - Residual kinds that drop
        the unified-hybrid (Qdrant) vector backend even when hybrid mode is
        active. Subset of ``residual_kinds``. ``[]`` is required when hybrid
        pairs with ``QdrantNoOpStructuredRetriever`` — the hybrid vector leg
        is then the sole payload-filter path, and dropping it for
        residual=``empty`` makes pure-filter queries return zero then fire
        ZeroResultGuard. Non-empty lists only make sense when a real
        structured backend owns filter-only retrieval.
    """
    residual_kinds: List[str]
    require_hard_chip_coverage: bool
    hybrid_mode_residual_kinds: List[str]

    def __post_init__(self) -> None:
        if not isinstance(self.residual_kinds, list):
            raise ConfigurationError( "retrieval.residual_dispatch.skip_vector_when.residual_kinds must be a list" )
        if len(self.residual_kinds) == 0:
            raise ConfigurationError( "retrieval.residual_dispatch.skip_vector_when.residual_kinds must be non-empty" )
        for kind in self.residual_kinds:
            if kind not in _RESIDUAL_KINDS_SET:
                raise ConfigurationError( f"retrieval.residual_dispatch.skip_vector_when.residual_kinds contains " f"unknown kind '{kind}'; must be one of {sorted(_RESIDUAL_KINDS_SET)}" )
        if not isinstance(self.require_hard_chip_coverage, bool):
            raise ConfigurationError( "retrieval.residual_dispatch.skip_vector_when.require_hard_chip_coverage must be a bool" )
        if not isinstance(self.hybrid_mode_residual_kinds, list):
            raise ConfigurationError( "retrieval.residual_dispatch.skip_vector_when.hybrid_mode_residual_kinds must be a list" )
        for kind in self.hybrid_mode_residual_kinds:
            if kind not in _RESIDUAL_KINDS_SET:
                raise ConfigurationError( f"retrieval.residual_dispatch.skip_vector_when.hybrid_mode_residual_kinds contains " f"unknown kind '{kind}'; must be one of {sorted(_RESIDUAL_KINDS_SET)}" )
            if kind not in self.residual_kinds:
                raise ConfigurationError( f"retrieval.residual_dispatch.skip_vector_when.hybrid_mode_residual_kinds entry " f"'{kind}' must also appear in residual_kinds" )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SkipVectorWhenConfig':
        _require(d, ['residual_kinds', 'require_hard_chip_coverage', 'hybrid_mode_residual_kinds'], 'retrieval.residual_dispatch.skip_vector_when')
        return cls(
            residual_kinds=[str(k) for k in d['residual_kinds']],
            require_hard_chip_coverage=bool(d['require_hard_chip_coverage']),
            hybrid_mode_residual_kinds=[str(k) for k in d['hybrid_mode_residual_kinds']],
        )


@dataclass
class ResidualDispatchConfig:
    """Residual-aware modality dispatch policy.

    When ``enabled=true``, the orchestrator drops the vector backend from the
    retrieval fan-out for queries whose ``QueryIntent.residual_kind`` matches
    ``skip_vector_when.residual_kinds`` and (optionally) carry ≥1 hard chip.
    Under Qdrant unified-hybrid the drop is gated by
    ``skip_vector_when.hybrid_mode_residual_kinds`` (normally ``[]`` so the
    hybrid vector leg — the sole payload-filter path beside NoOp structured —
    stays active for pure-filter / empty-residual queries).

    :param enabled: bool - Master flag.
    :param skip_vector_when: SkipVectorWhenConfig - Skip predicate.
    """
    enabled: bool
    skip_vector_when: SkipVectorWhenConfig

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("retrieval.residual_dispatch.enabled must be a bool")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ResidualDispatchConfig':
        _require(d, ['enabled', 'skip_vector_when'], 'retrieval.residual_dispatch')
        return cls( enabled=bool(d['enabled']), skip_vector_when=SkipVectorWhenConfig.from_dict(d['skip_vector_when']), )


@dataclass
class RetrievalMetricsConfig:
    """Online retrieval-quality metrics surfaced under ``response.retrieval_metrics``.

    These metrics use the post-fusion **normalised score** (each item's score
    divided by the top item's score) as a label-free relevance proxy: any
    returned item whose normalised score is ``>= relevance_threshold`` is
    counted as relevant. This is a self-consistency proxy — it measures how
    sharply the ranking concentrates above the threshold, not absolute truth
    against a labelled ground set (offline NDCG/Recall against golden seeds
    live under ``offline_eval`` and consume real labels).

    :param relevance_threshold: float - Score in ``[0.0, 1.0]`` above which a
        returned item is considered "relevant" for Recall@K, Precision@K, and
        HitRate@K. Lower values are looser (more items count as relevant);
        higher values are stricter. No default — must be set in YAML.
    :param score_normalization: str - How post-fusion fused scores are mapped to
        the ``[0, 1]`` ``coherence_score`` / relevance proxy. ``'max'`` (default,
        legacy) divides by the top item's score, so the floor floats high and a
        flat ranking leaves most items above the threshold. ``'minmax'`` rescales
        ``(s - min) / (max - min)`` so the tail drops toward 0, sharpening the
        relevance proxy. Ranking order is identical under both modes; only the
        score *spread* (and thus threshold-based Recall/Precision/HitRate and the
        ``2**score`` NDCG gains) changes. Reversible via config.
    """
    relevance_threshold: float
    score_normalization: str = 'max'

    _SCORE_NORMALIZATION_MODES = frozenset({'max', 'minmax'})

    def __post_init__(self) -> None:
        if not isinstance(self.relevance_threshold, (int, float)):
            raise ConfigurationError( "retrieval.metrics.relevance_threshold must be a number" )
        thr = float(self.relevance_threshold)
        if thr < 0.0 or thr > 1.0:
            raise ConfigurationError( f"retrieval.metrics.relevance_threshold must be in [0.0, 1.0]; got {thr}" )
        self.relevance_threshold = thr
        if self.score_normalization not in self._SCORE_NORMALIZATION_MODES:
            raise ConfigurationError(
                f"retrieval.metrics.score_normalization must be one of "
                f"{sorted(self._SCORE_NORMALIZATION_MODES)}; got {self.score_normalization!r}"
            )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'RetrievalMetricsConfig':
        _require(d, ['relevance_threshold'], 'retrieval.metrics')
        # Defer numeric coercion to __post_init__ so non-numeric inputs surface
        # as ConfigurationError (consistent with sibling configs) instead of a
        # raw ValueError from float().
        # score_normalization is optional — absent ->'max' (legacy behavior).
        return cls(
            relevance_threshold=d['relevance_threshold'],
            score_normalization=str(d.get('score_normalization', 'max')),
        )


@dataclass
class FuzzyRerankConfig:
    """Fuzzy lexical reranker config (typo / near-match surfacing).

    Reorders the over-fetched fused pool so SLD near-matches (e.g. ``hi-rentals``
    for query ``high rentals``) are boosted above the final top_k truncation.
    Pure stdlib (Damerau-Levenshtein over candidate ``payload['sld']`` tokens);
    no re-index and no external model.

    :param enabled: bool - When False the orchestrator skips reranking entirely
    :param max_edit_distance: int - Inclusive edit-distance cap per token pair (>= 1)
    :param min_token_length: int - Drop query tokens shorter than this (>= 1)
    :param max_terms: int - Cap on query tokens scored (>= 1)
    :param max_candidates: int - Cap on pool items rescored per query (>= 1; latency bound)
    :param min_similarity: float - Per-token-pair similarity floor in [0,1] to count as a match
    :param boost_weight: float - Multiplier on the fuzzy coverage score added to fused_score (>= 0)
    :param stopwords: List[str] - Query tokens dropped before scoring
    """
    enabled: bool
    max_edit_distance: int
    min_token_length: int
    max_terms: int
    max_candidates: int
    min_similarity: float
    boost_weight: float
    stopwords: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.max_edit_distance < 1:
            raise ConfigurationError("retrieval.fuzzy_rerank.max_edit_distance must be >= 1")
        if self.min_token_length < 1:
            raise ConfigurationError("retrieval.fuzzy_rerank.min_token_length must be >= 1")
        if self.max_terms < 1:
            raise ConfigurationError("retrieval.fuzzy_rerank.max_terms must be >= 1")
        if self.max_candidates < 1:
            raise ConfigurationError("retrieval.fuzzy_rerank.max_candidates must be >= 1")
        if not 0.0 <= self.min_similarity <= 1.0:
            raise ConfigurationError("retrieval.fuzzy_rerank.min_similarity must be in [0.0, 1.0]")
        if self.boost_weight < 0.0:
            raise ConfigurationError("retrieval.fuzzy_rerank.boost_weight must be >= 0.0")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FuzzyRerankConfig':
        _require(d, ['enabled', 'max_edit_distance', 'min_token_length', 'max_terms', 'max_candidates', 'min_similarity', 'boost_weight'], 'retrieval.fuzzy_rerank')
        raw_stop = d.get('stopwords', [])
        stopwords = [str(s) for s in raw_stop] if isinstance(raw_stop, list) else []
        return cls(
            enabled=bool(d['enabled']),
            max_edit_distance=int(d['max_edit_distance']),
            min_token_length=int(d['min_token_length']),
            max_terms=int(d['max_terms']),
            max_candidates=int(d['max_candidates']),
            min_similarity=float(d['min_similarity']),
            boost_weight=float(d['boost_weight']),
            stopwords=stopwords,
        )


@dataclass
class ConflictConfig:
    """Filter-conflict detection policy.

    :param enabled: bool - When True, the orchestrator detects contradictory
        filter slots before retrieval and short-circuits to a 'filter_conflict'
        response carrying explainers instead of returning an empty result set.
    :param messages: Dict[str, str] - Template per conflict kind ('range_inverted',
        'qualitative_quantitative'). Templates may reference {field}, {min}, {max}
        and {qualitative_slot}, {price_max}, {price_max_cap} and are formatted with
        the conflicting slot values.
    :param qualitative_conflict_rules: List[Dict] - Config-driven rules for detecting
        qualitative-quantitative contradictions. Each rule has shape:
        {qualitative_slot: str, price_max_cap: int}. A conflict fires when the
        qualitative slot is present and price_max < price_max_cap.
    """
    enabled: bool
    messages: Dict[str, str]
    qualitative_conflict_rules: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("retrieval.conflict.enabled must be a bool")
        if not isinstance(self.messages, dict) or not self.messages:
            raise ConfigurationError("retrieval.conflict.messages must be a non-empty mapping")
        if not isinstance(self.qualitative_conflict_rules, list):
            raise ConfigurationError("retrieval.conflict.qualitative_conflict_rules must be a list")
        for _rule in self.qualitative_conflict_rules:
            if not isinstance(_rule, dict) or 'qualitative_slot' not in _rule or 'price_max_cap' not in _rule:
                raise ConfigurationError("retrieval.conflict.qualitative_conflict_rules entries must have qualitative_slot and price_max_cap")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ConflictConfig':
        _require(d, ['enabled', 'messages'], 'retrieval.conflict')
        raw_rules = d.get('qualitative_conflict_rules') or []
        rules = [
            {'qualitative_slot': str(r['qualitative_slot']), 'price_max_cap': int(r['price_max_cap'])}
            for r in raw_rules
            if isinstance(r, dict) and 'qualitative_slot' in r and 'price_max_cap' in r
        ]
        return cls(
            enabled=bool(d['enabled']),
            messages={str(k): str(v) for k, v in dict(d['messages']).items()},
            qualitative_conflict_rules=rules,
        )


@dataclass
class BrandabilityConfig:
    """Deterministic brandability scorer config (ranking signal).

    :param enabled: bool - Master toggle for the brandability rerank stage.
    :param weight_length / weight_vowel_balance / weight_pronounceability: float -
        Relative weights of the three sub-scores (normalised by their sum).
    :param ideal_vowel_ratio: float - Target vowel fraction (peak of the balance score).
    :param min_length / max_length: int - SLD-length band for the length sub-score.
    :param max_consonant_run: int - Consonant run length above which pronounceability drops.
    :param digit_penalty / hyphen_penalty: float - Subtracted when the SLD has a digit / hyphen.
    :param boost_weight: float - Multiplier in fused*(1+boost_weight*brandability).
    :param trigger_terms: List[str] - Lowercase cue substrings (e.g. 'brandable',
        'memorable') that activate the rerank for a query. Empty list never fires.
    :param pure_sort_residual_kinds: List[str] - residual_kind values for which the
        rerank sorts purely by brandability score, ignoring fused_score entirely.
    """
    enabled: bool
    weight_length: float
    weight_vowel_balance: float
    weight_pronounceability: float
    ideal_vowel_ratio: float
    min_length: int
    max_length: int
    max_consonant_run: int
    digit_penalty: float
    hyphen_penalty: float
    boost_weight: float
    trigger_terms: List[str]
    trigger_regex: Optional[str] = None
    pure_sort_residual_kinds: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("retrieval.brandability.enabled must be a bool")
        if not isinstance(self.trigger_terms, list):
            raise ConfigurationError("retrieval.brandability.trigger_terms must be a list")
        if self.trigger_regex is not None and not isinstance(self.trigger_regex, str):
            raise ConfigurationError("retrieval.brandability.trigger_regex must be a string or null")
        if not 0.0 < self.ideal_vowel_ratio <= 1.0:
            raise ConfigurationError("retrieval.brandability.ideal_vowel_ratio must be in (0,1]")
        if self.min_length < 1 or self.max_length <= self.min_length:
            raise ConfigurationError("retrieval.brandability.max_length must be > min_length >= 1")
        if self.max_consonant_run < 1:
            raise ConfigurationError("retrieval.brandability.max_consonant_run must be >= 1")
        if self.boost_weight < 0.0:
            raise ConfigurationError("retrieval.brandability.boost_weight must be >= 0")
        if not isinstance(self.pure_sort_residual_kinds, list):
            raise ConfigurationError("retrieval.brandability.pure_sort_residual_kinds must be a list")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'BrandabilityConfig':
        _require(d, ['enabled', 'weight_length', 'weight_vowel_balance', 'weight_pronounceability', 'ideal_vowel_ratio', 'min_length', 'max_length', 'max_consonant_run', 'digit_penalty', 'hyphen_penalty', 'boost_weight', 'trigger_terms', 'pure_sort_residual_kinds'], 'retrieval.brandability')  # noqa: E501
        _tr = d.get('trigger_regex')
        return cls(enabled=bool(d['enabled']), weight_length=float(d['weight_length']), weight_vowel_balance=float(d['weight_vowel_balance']), weight_pronounceability=float(d['weight_pronounceability']), ideal_vowel_ratio=float(d['ideal_vowel_ratio']), min_length=int(d['min_length']), max_length=int(d['max_length']), max_consonant_run=int(d['max_consonant_run']), digit_penalty=float(d['digit_penalty']), hyphen_penalty=float(d['hyphen_penalty']), boost_weight=float(d['boost_weight']), trigger_terms=[str(t).lower() for t in d['trigger_terms']], trigger_regex=str(_tr) if _tr else None, pure_sort_residual_kinds=[str(k).lower() for k in d['pure_sort_residual_kinds']])  # noqa: E501


@dataclass
class EngagementBoostConfig:
    """Real-time engagement signal boost applied post-RRF fusion.

    Reads live payload fields from Qdrant (populated by EventIngestDriver) and
    applies a multiplicative lift: ``fused_score *= (1 + max_boost * engagement)``,
    where ``engagement`` is a weighted combination of normalised signals in [0, 1].

    :param enabled: bool - Master toggle. When False the stage is a no-op.
    :param max_boost: float - Maximum multiplicative factor (e.g. 0.3 = up to 30% lift).
    :param bid_velocity_cap: float - Normalisation cap for bid_velocity_1h.
    :param watch_density_cap: float - Normalisation cap for watch_density_1d.
    :param bidder_watch_cap: float - Normalisation cap for bidder_watch_density_1d.
    :param unique_bidder_cap: float - Normalisation cap for unique_bidder_count_4h.
    :param bid_velocity_weight: float - Weight for bid velocity signal.
    :param watch_density_weight: float - Weight for passive watch density signal.
    :param bidder_watch_weight: float - Weight for bidder-intent watch signal.
    :param unique_bidder_weight: float - Weight for unique bidder count signal.
    """
    enabled: bool
    max_boost: float
    bid_velocity_cap: float
    watch_density_cap: float
    bidder_watch_cap: float
    unique_bidder_cap: float
    bid_velocity_weight: float
    watch_density_weight: float
    bidder_watch_weight: float
    unique_bidder_weight: float

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("retrieval.engagement_boost.enabled must be a bool")
        if self.max_boost < 0.0:
            raise ConfigurationError("retrieval.engagement_boost.max_boost must be >= 0")
        for cap_name in ('bid_velocity_cap', 'watch_density_cap', 'bidder_watch_cap', 'unique_bidder_cap'):
            if getattr(self, cap_name) <= 0.0:
                raise ConfigurationError(f"retrieval.engagement_boost.{cap_name} must be > 0")
        total_w = self.bid_velocity_weight + self.watch_density_weight + self.bidder_watch_weight + self.unique_bidder_weight
        if total_w <= 0.0:
            raise ConfigurationError("retrieval.engagement_boost signal weights must sum to > 0")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'EngagementBoostConfig':
        _require(d, ['enabled', 'max_boost', 'bid_velocity_cap', 'watch_density_cap',
                     'bidder_watch_cap', 'unique_bidder_cap', 'bid_velocity_weight',
                     'watch_density_weight', 'bidder_watch_weight', 'unique_bidder_weight'],
                 'retrieval.engagement_boost')
        return cls(
            enabled=bool(d['enabled']),
            max_boost=float(d['max_boost']),
            bid_velocity_cap=float(d['bid_velocity_cap']),
            watch_density_cap=float(d['watch_density_cap']),
            bidder_watch_cap=float(d['bidder_watch_cap']),
            unique_bidder_cap=float(d['unique_bidder_cap']),
            bid_velocity_weight=float(d['bid_velocity_weight']),
            watch_density_weight=float(d['watch_density_weight']),
            bidder_watch_weight=float(d['bidder_watch_weight']),
            unique_bidder_weight=float(d['unique_bidder_weight']),
        )


@dataclass
class HardFilterApplicationConfig:
    """Honest application of identified FIND hard filters.

    Slots listed in ``backend_unsupported_slots`` have no Qdrant / structured
    payload path; response ``not_applied`` uses reason ``backend_unsupported``.
    All fields required when this block is present — no silent defaults.

    :param backend_unsupported_slots: List[str] - Entity slot names that cannot
        be enforced as hard retrieval constraints (no indexable payload/column)
    """
    backend_unsupported_slots: List[str]

    def __post_init__(self) -> None:
        if not isinstance(self.backend_unsupported_slots, list):
            raise ConfigurationError(
                'retrieval.hard_filter_application.backend_unsupported_slots must be a list'
            )
        cleaned: List[str] = []
        seen: set = set()
        for raw in self.backend_unsupported_slots:
            if not isinstance(raw, str) or not raw.strip():
                raise ConfigurationError(
                    'retrieval.hard_filter_application.backend_unsupported_slots '
                    'entries must be non-empty strings'
                )
            name = raw.strip()
            if name in seen:
                continue
            seen.add(name)
            cleaned.append(name)
        self.backend_unsupported_slots = cleaned

    @property
    def backend_unsupported_set(self) -> frozenset:
        return frozenset(self.backend_unsupported_slots)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'HardFilterApplicationConfig':
        _require(d, ['backend_unsupported_slots'], 'retrieval.hard_filter_application')
        raw = d['backend_unsupported_slots']
        if not isinstance(raw, list):
            raise ConfigurationError(
                'retrieval.hard_filter_application.backend_unsupported_slots must be a list'
            )
        return cls(backend_unsupported_slots=list(raw))


@dataclass
class HardFilterPushdownSlotConfig:
    """One hard-filter slot pushed into explore ClickHouse SQL WHERE.

    :param column: str - Result-column alias used in the outer WHERE (must match
        SELECT aliases in rail SQL, e.g. ``tld``, ``price``, ``auction_type``)
    :param op: str - One of ``in``, ``gte``, ``lte``
    """
    column: str
    op: str

    def __post_init__(self) -> None:
        if not isinstance(self.column, str) or not self.column.strip():
            raise ConfigurationError(
                'explore.clickhouse_rails.hard_filter_pushdown.slots.*.column '
                'must be a non-empty string'
            )
        if self.op not in ('in', 'gte', 'lte'):
            raise ConfigurationError(
                'explore.clickhouse_rails.hard_filter_pushdown.slots.*.op '
                "must be one of: 'in', 'gte', 'lte'"
            )
        self.column = self.column.strip()

    @classmethod
    def from_dict(cls, d: Dict[str, Any], context: str) -> 'HardFilterPushdownSlotConfig':
        _require(d, ['column', 'op'], context)
        return cls(column=str(d['column']), op=str(d['op']))


@dataclass
class HardFilterPushdownConfig:
    """Push identified hard filters into explore CH rail SQL (outer WHERE).

    When ``enabled=true``, every key under ``slots`` is required to carry
    ``column`` + ``op`` — no defaults.

    :param enabled: bool - Master toggle for SQL pushdown
    :param slots: Dict[str, HardFilterPushdownSlotConfig] - Entity slot to column/op
    """
    enabled: bool
    slots: Dict[str, HardFilterPushdownSlotConfig]

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError(
                'explore.clickhouse_rails.hard_filter_pushdown.enabled must be a bool'
            )
        if not isinstance(self.slots, dict):
            raise ConfigurationError(
                'explore.clickhouse_rails.hard_filter_pushdown.slots must be a dict'
            )
        if self.enabled and not self.slots:
            raise ConfigurationError(
                'explore.clickhouse_rails.hard_filter_pushdown.slots must be non-empty '
                'when enabled=true'
            )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'HardFilterPushdownConfig':
        _require(d, ['enabled', 'slots'], 'explore.clickhouse_rails.hard_filter_pushdown')
        raw_slots = d['slots']
        if not isinstance(raw_slots, dict):
            raise ConfigurationError(
                'explore.clickhouse_rails.hard_filter_pushdown.slots must be a dict'
            )
        slots: Dict[str, HardFilterPushdownSlotConfig] = {}
        for slot_name, spec in raw_slots.items():
            if not isinstance(slot_name, str) or not slot_name.strip():
                raise ConfigurationError(
                    'explore.clickhouse_rails.hard_filter_pushdown.slots keys '
                    'must be non-empty strings'
                )
            if not isinstance(spec, dict):
                raise ConfigurationError(
                    f'explore.clickhouse_rails.hard_filter_pushdown.slots.{slot_name} '
                    'must be a dict'
                )
            slots[slot_name.strip()] = HardFilterPushdownSlotConfig.from_dict(
                spec,
                f'explore.clickhouse_rails.hard_filter_pushdown.slots.{slot_name.strip()}',
            )
        return cls(enabled=bool(d['enabled']), slots=slots)


@dataclass
class RetrievalConfig:
    """Retrieval bundle.

    :param qdrant: Optional[QdrantConfig] - Required iff
        `vector.backend='qdrant'` or `structured.backend='qdrant'`. When neither
        backend is set to qdrant, this slot may be omitted from YAML — the
        in-memory backends require no Qdrant config.
    :param residual_dispatch: Optional[ResidualDispatchConfig] -
        Residual-aware modality dispatch policy. Optional/additive: when
        omitted or ``enabled=false``, the orchestrator uses legacy type-only
        dispatch (every backend allowed by ``query_type`` fires).
    :param query_compound_split: Optional[CompoundSplitterConfig] -
        Query-side compound-word splitter. Optional/additive: when omitted or
        ``enabled=false``, the query encode text is embedded verbatim. When
        enabled, glued query labels (``techstartup``) split into segments
        (``tech startup``) BEFORE the dense / sparse / ngram legs encode them,
        so the query matches the document-side segments the ingest splitter
        wrote. Reuses :class:`CompoundSplitterConfig` so query + document share
        one dictionary + cost model; point ``dictionary_path`` at the same CSV
        the ingest ``vectorization.compound_splitter`` block uses.
    :param hard_filter_application: Optional[HardFilterApplicationConfig] -
        When set, lists FIND hard slots that cannot be enforced backend-side
        (surfaced as ``not_applied`` reason ``backend_unsupported``).
    """
    vector: VectorRetrievalConfig
    structured: StructuredRetrievalConfig
    sql: SqlRetrievalConfig
    fusion: FusionConfig
    eranker: ERankerConfig
    diversity: DiversityConfig
    metrics: RetrievalMetricsConfig
    qdrant: Optional[QdrantConfig] = None
    residual_dispatch: Optional[ResidualDispatchConfig] = None
    over_fetch_multiplier: float = 3.0
    enforce_categorical_gate_on_missing_payload: bool = True
    guard_widen_reapply_hard_gate: bool = True
    fuzzy_rerank: Optional[FuzzyRerankConfig] = None
    query_compound_split: Optional['CompoundSplitterConfig'] = None
    conflict: Optional[ConflictConfig] = None
    brandability: Optional[BrandabilityConfig] = None
    engagement_boost: Optional[EngagementBoostConfig] = None
    hard_filter_application: Optional[HardFilterApplicationConfig] = None

    def __post_init__(self) -> None:
        if self.over_fetch_multiplier < 1.0:
            raise ConfigurationError("retrieval.over_fetch_multiplier must be >= 1.0")
        needs_qdrant = self.vector.backend == 'qdrant' or self.structured.backend == 'qdrant'
        if needs_qdrant and self.qdrant is None:
            raise ConfigurationError( "retrieval.qdrant must be present when vector.backend='qdrant' or structured.backend='qdrant'" )
        # Hybrid mode is symmetric: both backends must point at qdrant when hybrid is on,
        # otherwise SearchOrchestrator's fan-out would issue two Qdrant calls (one wasted).
        if self.qdrant is not None and self.qdrant.hybrid.enabled:
            if self.vector.backend != 'qdrant' or self.structured.backend != 'qdrant':
                raise ConfigurationError(
                    "retrieval.qdrant.hybrid.enabled=true requires both retrieval.vector.backend='qdrant' "
                    "and retrieval.structured.backend='qdrant' (Option A unified-index contract)"
                )
            # Prefetch limit must cover the larger of the two backends' top_k so the
            # server-side fusion has enough candidates to rank from.
            max_top_k = max(self.vector.top_k, self.structured.top_k)
            if self.qdrant.hybrid.prefetch_limit < max_top_k:
                raise ConfigurationError(
                    f"retrieval.qdrant.hybrid.prefetch_limit ({self.qdrant.hybrid.prefetch_limit}) "
                    f"must be >= max(vector.top_k={self.vector.top_k}, structured.top_k={self.structured.top_k})"
                )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'RetrievalConfig':
        _require(d, ['vector', 'structured', 'sql', 'fusion', 'eranker', 'diversity', 'metrics'], 'retrieval')
        qdrant_block = d.get('qdrant') if isinstance(d, dict) else None
        qdrant_cfg = QdrantConfig.from_dict(qdrant_block) if isinstance(qdrant_block, dict) else None
        dispatch_raw = d.get('residual_dispatch')
        residual_dispatch = ResidualDispatchConfig.from_dict(dispatch_raw) if isinstance(dispatch_raw, dict) else None
        fuzzy_raw = d.get('fuzzy_rerank')
        fuzzy_rerank = FuzzyRerankConfig.from_dict(fuzzy_raw) if isinstance(fuzzy_raw, dict) else None
        qcs_raw = d.get('query_compound_split')
        query_compound_split = CompoundSplitterConfig.from_dict(qcs_raw) if isinstance(qcs_raw, dict) else None
        conflict_raw = d.get('conflict')
        conflict = ConflictConfig.from_dict(conflict_raw) if isinstance(conflict_raw, dict) else None
        brandability_raw = d.get('brandability')
        brandability = BrandabilityConfig.from_dict(brandability_raw) if isinstance(brandability_raw, dict) else None
        engagement_boost_raw = d.get('engagement_boost')
        engagement_boost = EngagementBoostConfig.from_dict(engagement_boost_raw) if isinstance(engagement_boost_raw, dict) else None
        hfa_raw = d.get('hard_filter_application')
        hard_filter_application = (
            HardFilterApplicationConfig.from_dict(hfa_raw) if isinstance(hfa_raw, dict) else None
        )
        return cls(
            vector=VectorRetrievalConfig.from_dict(d['vector']),
            structured=StructuredRetrievalConfig.from_dict(d['structured']),
            sql=SqlRetrievalConfig.from_dict(d['sql']),
            fusion=FusionConfig.from_dict(d['fusion']),
            eranker=ERankerConfig.from_dict(d['eranker']),
            diversity=DiversityConfig.from_dict(d['diversity']),
            metrics=RetrievalMetricsConfig.from_dict(d['metrics']),
            qdrant=qdrant_cfg,
            residual_dispatch=residual_dispatch,
            over_fetch_multiplier=float(d.get('over_fetch_multiplier', 3.0)),
            enforce_categorical_gate_on_missing_payload=bool(d.get('enforce_categorical_gate_on_missing_payload', True)),
            guard_widen_reapply_hard_gate=bool(d.get('guard_widen_reapply_hard_gate', True)),
            fuzzy_rerank=fuzzy_rerank,
            query_compound_split=query_compound_split,
            conflict=conflict,
            brandability=brandability,
            engagement_boost=engagement_boost,
            hard_filter_application=hard_filter_application,
        )


@dataclass
class ExactCacheConfig:
    enabled: bool
    ttl_seconds: int
    max_entries: int
    max_bytes: Optional[int] = None
    ttl_jitter_seconds: int = 0

    def __post_init__(self) -> None:
        if self.ttl_seconds < 1:
            raise ConfigurationError("cache.exact.ttl_seconds must be >= 1")
        if self.max_entries < 1:
            raise ConfigurationError("cache.exact.max_entries must be >= 1")
        if self.max_bytes is not None and self.max_bytes < 1:
            raise ConfigurationError("cache.exact.max_bytes must be >= 1 when set")
        if self.ttl_jitter_seconds < 0:
            raise ConfigurationError("cache.exact.ttl_jitter_seconds must be >= 0")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ExactCacheConfig':
        _require(d, ['enabled', 'ttl_seconds', 'max_entries'], 'cache.exact')
        raw_mb = d.get('max_bytes')
        return cls(
            enabled=bool(d['enabled']),
            ttl_seconds=int(d['ttl_seconds']),
            max_entries=int(d['max_entries']),
            max_bytes=int(raw_mb) if raw_mb is not None else None,
            ttl_jitter_seconds=int(d.get('ttl_jitter_seconds', 0)),
        )


@dataclass
class StructuredCacheConfig:
    enabled: bool
    ttl_seconds: int
    max_entries: int
    max_bytes: Optional[int] = None
    ttl_jitter_seconds: int = 0

    def __post_init__(self) -> None:
        if self.ttl_seconds < 1:
            raise ConfigurationError("cache.structured.ttl_seconds must be >= 1")
        if self.max_entries < 1:
            raise ConfigurationError("cache.structured.max_entries must be >= 1")
        if self.max_bytes is not None and self.max_bytes < 1:
            raise ConfigurationError("cache.structured.max_bytes must be >= 1 when set")
        if self.ttl_jitter_seconds < 0:
            raise ConfigurationError("cache.structured.ttl_jitter_seconds must be >= 0")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'StructuredCacheConfig':
        _require(d, ['enabled', 'ttl_seconds', 'max_entries'], 'cache.structured')
        raw_mb = d.get('max_bytes')
        return cls(
            enabled=bool(d['enabled']),
            ttl_seconds=int(d['ttl_seconds']),
            max_entries=int(d['max_entries']),
            max_bytes=int(raw_mb) if raw_mb is not None else None,
            ttl_jitter_seconds=int(d.get('ttl_jitter_seconds', 0)),
        )


@dataclass
class IntentPlanCacheConfig:
    """Plan Tier-3 intent-structure cache (post-QI, pre-retrieval)."""

    enabled: bool
    ttl_seconds: int
    max_entries: int
    max_bytes: Optional[int] = None
    ttl_jitter_seconds: int = 0

    def __post_init__(self) -> None:
        if self.ttl_seconds < 1:
            raise ConfigurationError("cache.intent_plan.ttl_seconds must be >= 1")
        if self.max_entries < 1:
            raise ConfigurationError("cache.intent_plan.max_entries must be >= 1")
        if self.max_bytes is not None and self.max_bytes < 1:
            raise ConfigurationError("cache.intent_plan.max_bytes must be >= 1 when set")
        if self.ttl_jitter_seconds < 0:
            raise ConfigurationError("cache.intent_plan.ttl_jitter_seconds must be >= 0")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'IntentPlanCacheConfig':
        _require(d, ['enabled', 'ttl_seconds', 'max_entries'], 'cache.intent_plan')
        raw_mb = d.get('max_bytes')
        return cls(
            enabled=bool(d['enabled']),
            ttl_seconds=int(d['ttl_seconds']),
            max_entries=int(d['max_entries']),
            max_bytes=int(raw_mb) if raw_mb is not None else None,
            ttl_jitter_seconds=int(d.get('ttl_jitter_seconds', 0)),
        )


@dataclass
class CacheConfig:
    enabled: bool
    exact: ExactCacheConfig
    structured: StructuredCacheConfig
    intent_plan: IntentPlanCacheConfig
    remote: RemoteCacheLayerConfig

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CacheConfig':
        _require(d, ['enabled', 'exact', 'structured', 'intent_plan', 'remote'], 'cache')
        return cls(
            enabled=bool(d['enabled']),
            exact=ExactCacheConfig.from_dict(d['exact']),
            structured=StructuredCacheConfig.from_dict(d['structured']),
            intent_plan=IntentPlanCacheConfig.from_dict(d['intent_plan']),
            remote=RemoteCacheLayerConfig.from_dict(d['remote'], context='cache.remote'),
        )


@dataclass
class SurfaceConfig:
    enabled: bool
    allowed_override_fields: List[str]
    max_saved_searches: int
    default_recommended_mode: str
    mode_recommendation_reason: str
    saved_search_layer0_max_depth: int
    saved_search_layer0_max_string_nodes: int

    def __post_init__(self) -> None:
        if not isinstance(self.allowed_override_fields, list) or len(self.allowed_override_fields) == 0:
            raise ConfigurationError("surface.allowed_override_fields must be a non-empty list")
        if self.max_saved_searches < 1:
            raise ConfigurationError("surface.max_saved_searches must be >= 1")
        if self.default_recommended_mode not in ('conversational', 'advanced'):
            raise ConfigurationError("surface.default_recommended_mode must be 'conversational' or 'advanced'")
        if not isinstance(self.mode_recommendation_reason, str) or not str(self.mode_recommendation_reason).strip():
            raise ConfigurationError("surface.mode_recommendation_reason must be a non-empty string")
        if int(self.saved_search_layer0_max_depth) < 1:
            raise ConfigurationError("surface.saved_search_layer0_max_depth must be >= 1")
        if int(self.saved_search_layer0_max_string_nodes) < 1:
            raise ConfigurationError("surface.saved_search_layer0_max_string_nodes must be >= 1")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SurfaceConfig':
        _require(
            d,
            [
                'enabled',
                'allowed_override_fields',
                'max_saved_searches',
                'default_recommended_mode',
                'mode_recommendation_reason',
                'saved_search_layer0_max_depth',
                'saved_search_layer0_max_string_nodes',
            ],
            'surface',
        )
        return cls(
            enabled=bool(d['enabled']),
            allowed_override_fields=[str(f) for f in d['allowed_override_fields']],
            max_saved_searches=int(d['max_saved_searches']),
            default_recommended_mode=str(d['default_recommended_mode']),
            mode_recommendation_reason=str(d['mode_recommendation_reason']),
            saved_search_layer0_max_depth=int(d['saved_search_layer0_max_depth']),
            saved_search_layer0_max_string_nodes=int(d['saved_search_layer0_max_string_nodes']),
        )


@dataclass
class HistoryCompactorConfig:
    """Compactor that aggregates 90-day raw history into a per-user feature vector.

    90-day raw retention then aggregation to a feature vector (raw text discarded).
    Consumer surface: a personalized landing rail composed offline daily from the user's history.
    Default ``enabled=false`` so the
    compactor is opt-in per-environment; the search/resume path continues to
    serve traffic from the raw store regardless of this flag.

    :param enabled: bool - When False the driver is constructed as None
    :param interval_seconds: float - Polling cadence (must be > 0)
    :param max_consecutive_failures: int - Cycles after which the driver pauses (>= 1)
    :param top_tld_count: int - Cap on top_tlds emitted in each vector (>= 1)
    :param top_theme_count: int - Cap on top_themes emitted in each vector (>= 1)
    :param min_theme_occurrences: int - Minimum repeats for a query stem to count as a theme (>= 1)
    :param max_users_per_cycle: int - Hard cap on users compacted per cycle (>= 1)
    """
    enabled: bool
    interval_seconds: float
    max_consecutive_failures: int
    top_tld_count: int
    top_theme_count: int
    min_theme_occurrences: int
    max_users_per_cycle: int

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("history.compactor.enabled must be a bool")
        if float(self.interval_seconds) <= 0.0:
            raise ConfigurationError("history.compactor.interval_seconds must be > 0")
        if int(self.max_consecutive_failures) < 1:
            raise ConfigurationError("history.compactor.max_consecutive_failures must be >= 1")
        if int(self.top_tld_count) < 1:
            raise ConfigurationError("history.compactor.top_tld_count must be >= 1")
        if int(self.top_theme_count) < 1:
            raise ConfigurationError("history.compactor.top_theme_count must be >= 1")
        if int(self.min_theme_occurrences) < 1:
            raise ConfigurationError("history.compactor.min_theme_occurrences must be >= 1")
        if int(self.max_users_per_cycle) < 1:
            raise ConfigurationError("history.compactor.max_users_per_cycle must be >= 1")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'HistoryCompactorConfig':
        _require( d, ['enabled', 'interval_seconds', 'max_consecutive_failures', 'top_tld_count', 'top_theme_count', 'min_theme_occurrences', 'max_users_per_cycle'], 'history.compactor', )
        return cls(
            enabled=bool(d['enabled']),
            interval_seconds=float(d['interval_seconds']),
            max_consecutive_failures=int(d['max_consecutive_failures']),
            top_tld_count=int(d['top_tld_count']),
            top_theme_count=int(d['top_theme_count']),
            min_theme_occurrences=int(d['min_theme_occurrences']),
            max_users_per_cycle=int(d['max_users_per_cycle']),
        )


@dataclass
class HistoryConfig:
    enabled: bool
    retention_days: int
    max_entries_per_user: int
    max_query_length: int
    resume_window_days: int
    max_resume_candidates: int
    compactor: Optional[HistoryCompactorConfig] = None

    def __post_init__(self) -> None:
        if self.retention_days < 1:
            raise ConfigurationError("history.retention_days must be >= 1")
        if self.max_entries_per_user < 1:
            raise ConfigurationError("history.max_entries_per_user must be >= 1")
        if self.max_query_length < 1:
            raise ConfigurationError("history.max_query_length must be >= 1")
        if self.resume_window_days < 1:
            raise ConfigurationError("history.resume_window_days must be >= 1")
        if self.resume_window_days > self.retention_days:
            raise ConfigurationError("history.resume_window_days must be <= retention_days")
        if self.max_resume_candidates < 1:
            raise ConfigurationError("history.max_resume_candidates must be >= 1")
        if self.compactor is not None and not isinstance(self.compactor, HistoryCompactorConfig):
            raise ConfigurationError("history.compactor must be a HistoryCompactorConfig")
        if self.compactor is not None and self.compactor.enabled and not self.enabled:
            raise ConfigurationError("history.compactor.enabled requires history.enabled=true")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'HistoryConfig':
        _require(d, ['enabled', 'retention_days', 'max_entries_per_user', 'max_query_length', 'resume_window_days', 'max_resume_candidates'], 'history')
        compactor_raw = d.get('compactor')
        compactor = HistoryCompactorConfig.from_dict(compactor_raw) if compactor_raw is not None else None
        return cls(
            enabled=bool(d['enabled']),
            retention_days=int(d['retention_days']),
            max_entries_per_user=int(d['max_entries_per_user']),
            max_query_length=int(d['max_query_length']),
            resume_window_days=int(d['resume_window_days']),
            max_resume_candidates=int(d['max_resume_candidates']),
            compactor=compactor,
        )


@dataclass
class ExploreRailSourceConfig:
    """Per-rail source toggle + display cap.
    :param enabled: bool - Whether this rail composes
    :param max_items: int - Hard cap on cards in this rail (>= 1)
    :param title: str - Rail title shown to the user (becomes ExploreRail.title)
    """
    enabled: bool
    max_items: int
    title: str

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("explore.<rail>.enabled must be bool")
        if self.max_items < 1:
            raise ConfigurationError("explore.<rail>.max_items must be >= 1")
        if not isinstance(self.title, str) or not self.title:
            raise ConfigurationError("explore.<rail>.title must be a non-empty string")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ExploreRailSourceConfig':
        _require(d, ['enabled', 'max_items', 'title'], 'explore.<rail>')
        return cls(enabled=bool(d['enabled']), max_items=int(d['max_items']), title=str(d['title']))


@dataclass
class TrendingRailConfig:
    """Trending rail config (bid-velocity windowed signal).
    :param source: ExploreRailSourceConfig - Common rail toggles
    :param window_seconds: int - Trending window (>= 60); the in-memory source treats
        items added inside the window as the candidate pool
    """
    source: ExploreRailSourceConfig
    window_seconds: int

    def __post_init__(self) -> None:
        if self.window_seconds < 60:
            raise ConfigurationError("explore.trending.window_seconds must be >= 60")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TrendingRailConfig':
        _require(d, ['enabled', 'max_items', 'title', 'window_seconds'], 'explore.trending')
        return cls(source=ExploreRailSourceConfig.from_dict(d), window_seconds=int(d['window_seconds']))


@dataclass
class EndingSoonRailConfig:
    """Ending-soon rail config (auctions ending in horizon window).
    :param source: ExploreRailSourceConfig - Common rail toggles
    :param horizon_seconds: int - Default items ending within `horizon_seconds` of `now` are rail candidates (>= 60)
    :param max_horizon_seconds: int - Upper bound for dynamic QI time_remaining_max overrides (>= horizon_seconds)
    """
    source: ExploreRailSourceConfig
    horizon_seconds: int
    max_horizon_seconds: int = _ENDING_SOON_DEFAULT_MAX_HORIZON_SECONDS

    def __post_init__(self) -> None:
        if self.horizon_seconds < 60:
            raise ConfigurationError("explore.ending_soon.horizon_seconds must be >= 60")
        if self.max_horizon_seconds < self.horizon_seconds:
            raise ConfigurationError("explore.ending_soon.max_horizon_seconds must be >= horizon_seconds")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'EndingSoonRailConfig':
        _require(d, ['enabled', 'max_items', 'title', 'horizon_seconds'], 'explore.ending_soon')
        return cls(source=ExploreRailSourceConfig.from_dict(d), horizon_seconds=int(d['horizon_seconds']), max_horizon_seconds=int(d.get('max_horizon_seconds') or _ENDING_SOON_DEFAULT_MAX_HORIZON_SECONDS))  # noqa: E501


@dataclass
class FallbackRailConfig:
    """Fallback rail config (used when every other rail is empty so the response
    is never empty — "a search never dead-ends on an empty page").
    :param source: ExploreRailSourceConfig - Common rail toggles
    """
    source: ExploreRailSourceConfig

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FallbackRailConfig':
        _require(d, ['enabled', 'max_items', 'title'], 'explore.fallback')
        return cls(source=ExploreRailSourceConfig.from_dict(d))


@dataclass
class ZeroResultGuardConfig:
    """Zero-Result Guard ladder config.
    :param enabled: bool - Master toggle
    :param relax_filters_drop_priority: List[str] - Filter slot names in the order
        the guard drops them (first dropped first). Names must be in the structured
        retriever's known filter slots.
    :param semantic_only_top_k: int - Top-k passed to the vector retriever in the
        semantic-only step (>= 1)
    :param explore_fallback_max_per_rail: int - Per-rail cap when the explore step
        is reached (>= 1) — distinct from the rail's own max_items so the fallback
        can be smaller than the public landing rail
    :param widen_filters_enabled: bool - When True, the guard widens numeric
        range filters (per ``widen_filters_multipliers``) before dropping any.
        This makes "find domains under $100" widen to $200/$500/$1000 before
        the bound is removed entirely.
    :param widen_filters_multipliers: List[float] - Multipliers applied in order
        to a "max"-bound (e.g. price_max) or inverse to a "min"-bound (price_min).
        Each must be > 1.0. Empty list disables widening even when
        ``widen_filters_enabled=True``.
    :param widen_filters_slots: List[str] - Slot names eligible for widening.
        Slots not in this list are still subject to the drop ladder. Limited
        to numeric bounds where widening is meaningful.
    :param rrf_k: int - Reciprocal Rank Fusion k constant (default 60). Higher = smoother
        blending; lower = stronger top-rank amplification. Must be >= 1.
    :param semantic_fallback_top_k: int - Number of vector search results to include
        when building the explore fallback (SLA breach or zero-result). Must be >= 1.
    :param min_results_before_relax: int - Guard fires only when primary retrieval
        returns fewer than this many results (default 10). Set to 1 to match the
        original behavior (fire on zero results only).
    :param apply_eranker_on_explore_fallback: bool - When True, the external eRanker is
        applied to explore-fallback results so investor/semantic queries receive
        coherence-ranked output instead of unranked rail order (default False).
    :param rail_timeout_seconds: float - Per-rail asyncio.wait_for timeout in _compose_rails.
        Slow CH sources that exceed this are skipped (logged as explore_rail_fetch_error).
        0.0 disables the per-rail timeout.
    """
    enabled: bool
    relax_filters_drop_priority: List[str]
    semantic_only_top_k: int
    explore_fallback_max_per_rail: int
    widen_filters_enabled: bool
    widen_filters_multipliers: List[float]
    widen_filters_slots: List[str]
    rrf_k: int
    semantic_fallback_top_k: int
    min_results_before_relax: int = 10
    relax_filters_protected_slots: List[str] = field(default_factory=list)
    apply_eranker_on_explore_fallback: bool = False
    explore_fallback_degraded_timeout_seconds: float = 0.5
    explore_fallback_timeout_seconds: float = 1.5
    rail_timeout_seconds: float = 3.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("explore.zero_result_guard.enabled must be bool")
        if not isinstance(self.relax_filters_drop_priority, list) or len(self.relax_filters_drop_priority) == 0:
            raise ConfigurationError("explore.zero_result_guard.relax_filters_drop_priority must be a non-empty list")
        for slot in self.relax_filters_drop_priority:
            if not isinstance(slot, str) or not slot:
                raise ConfigurationError("explore.zero_result_guard.relax_filters_drop_priority entries must be non-empty strings")
        if len(set(self.relax_filters_drop_priority)) != len(self.relax_filters_drop_priority):
            raise ConfigurationError("explore.zero_result_guard.relax_filters_drop_priority entries must be unique")
        if self.semantic_only_top_k < 1:
            raise ConfigurationError("explore.zero_result_guard.semantic_only_top_k must be >= 1")
        if self.explore_fallback_max_per_rail < 1:
            raise ConfigurationError("explore.zero_result_guard.explore_fallback_max_per_rail must be >= 1")
        if not isinstance(self.widen_filters_enabled, bool):
            raise ConfigurationError("explore.zero_result_guard.widen_filters_enabled must be bool")
        if not isinstance(self.widen_filters_multipliers, list):
            raise ConfigurationError("explore.zero_result_guard.widen_filters_multipliers must be a list")
        for m in self.widen_filters_multipliers:
            if not isinstance(m, (int, float)) or float(m) <= 1.0:
                raise ConfigurationError("explore.zero_result_guard.widen_filters_multipliers entries must be > 1.0")
        if not isinstance(self.widen_filters_slots, list):
            raise ConfigurationError("explore.zero_result_guard.widen_filters_slots must be a list")
        for slot in self.widen_filters_slots:
            if not isinstance(slot, str) or not slot:
                raise ConfigurationError("explore.zero_result_guard.widen_filters_slots entries must be non-empty strings")
        if len(set(self.widen_filters_slots)) != len(self.widen_filters_slots):
            raise ConfigurationError("explore.zero_result_guard.widen_filters_slots entries must be unique")
        if int(self.rrf_k) < 1:
            raise ConfigurationError("explore.zero_result_guard.rrf_k must be >= 1")
        if int(self.semantic_fallback_top_k) < 1:
            raise ConfigurationError("explore.zero_result_guard.semantic_fallback_top_k must be >= 1")
        if int(self.min_results_before_relax) < 0:
            raise ConfigurationError("explore.zero_result_guard.min_results_before_relax must be >= 0")
        if not isinstance(self.relax_filters_protected_slots, list) or not all(isinstance(s, str) and s for s in self.relax_filters_protected_slots):
            raise ConfigurationError("explore.zero_result_guard.relax_filters_protected_slots must be a list of non-empty strings")
        if not isinstance(self.apply_eranker_on_explore_fallback, bool):
            raise ConfigurationError("explore.zero_result_guard.apply_eranker_on_explore_fallback must be a bool")
        if float(self.explore_fallback_degraded_timeout_seconds) <= 0.0:
            raise ConfigurationError("explore.zero_result_guard.explore_fallback_degraded_timeout_seconds must be > 0")
        if float(self.explore_fallback_timeout_seconds) <= 0.0:
            raise ConfigurationError("explore.zero_result_guard.explore_fallback_timeout_seconds must be > 0")
        if float(self.rail_timeout_seconds) < 0.0:
            raise ConfigurationError("explore.zero_result_guard.rail_timeout_seconds must be >= 0")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ZeroResultGuardConfig':
        _require(
            d,
            [
                'enabled', 'relax_filters_drop_priority', 'semantic_only_top_k',
                'explore_fallback_max_per_rail', 'widen_filters_enabled',
                'widen_filters_multipliers', 'widen_filters_slots',
                'rrf_k', 'semantic_fallback_top_k',
            ],
            'explore.zero_result_guard',
        )
        return cls(
            enabled=bool(d['enabled']),
            relax_filters_drop_priority=[str(s) for s in d['relax_filters_drop_priority']],
            semantic_only_top_k=int(d['semantic_only_top_k']),
            explore_fallback_max_per_rail=int(d['explore_fallback_max_per_rail']),
            widen_filters_enabled=bool(d['widen_filters_enabled']),
            widen_filters_multipliers=[float(m) for m in d['widen_filters_multipliers']],
            widen_filters_slots=[str(s) for s in d['widen_filters_slots']],
            rrf_k=int(d['rrf_k']),
            semantic_fallback_top_k=int(d['semantic_fallback_top_k']),
            min_results_before_relax=int(d.get('min_results_before_relax', 10)),
            relax_filters_protected_slots=[str(s) for s in d.get('relax_filters_protected_slots', [])],
            apply_eranker_on_explore_fallback=bool(d.get('apply_eranker_on_explore_fallback', False)),
            explore_fallback_degraded_timeout_seconds=float(d.get('explore_fallback_degraded_timeout_seconds', 0.5)),
            explore_fallback_timeout_seconds=float(d.get('explore_fallback_timeout_seconds', 1.5)),
            rail_timeout_seconds=float(d.get('rail_timeout_seconds', 3.0)),
        )


@dataclass
class ExploreClickHouseRailsConfig:
    """Optional ClickHouse-backed explore rails (trending / ending-soon / last-hour / latest / high-volume / fresh / last-week / watch-density / high-traffic).
    :param enabled: bool - When True and credentials are available, registry wires CH sources instead of in-memory wells
    :param trending_sql: str - ClickHouse SQL returning ``item_id`` and ``trending_score`` columns
    :param ending_soon_sql: str - ClickHouse SQL returning ``item_id``, ``ends_at`` (unix), optional payload columns
    :param last_hour_sql: str - SQL for domains ending in the next 1 hour (ultra-urgent); empty disables the rail
    :param latest_sql: str - SQL for recently expiring domains (7-day window, newest-first); empty disables the rail
    :param high_volume_sql: str - SQL for high-bid-activity domains ordered by bid_count DESC; empty disables the rail
    :param fresh_sql: str - SQL for zero-bid fresh-arrival domains; empty disables the rail
    :param last_week_sql: str - SQL for long-duration auctions (> 7 days remaining); empty disables the rail
    :param watch_density_sql: str - SQL for top-watched domains (``watch_score`` column); empty disables the rail
    :param high_traffic_sql: str - SQL for domains ranked by weighted bid velocity + watch density (``traffic_score`` column); empty disables the rail
    :param max_rows_each: int - Row cap per rail query (>= 1)
    :param bid_recency_horizon_seconds: float - Decay window for bid recency in latest rail (seconds)
    :param high_volume_score_norm: float - Divisor for normalising bid_count to [0,1] in high_volume rail
    :param watch_density_score_norm: float - Divisor for normalising watch_score to [0,1]; required when watch_density_sql is non-empty
    :param high_traffic_score_norm: float - Divisor for normalising traffic_score to [0,1]; required when high_traffic_sql is non-empty
    :param clickhouse_db: str - ClickHouse database name substituted into all SQL strings at load time
    :param fresh_time_horizon_seconds: float - Window (seconds) for scoring fresh-arrival recency
    :param hard_filter_pushdown: Optional[HardFilterPushdownConfig] - When set and
        enabled, wrap rail SQL with outer WHERE for configured hard slots
    """
    enabled: bool
    trending_sql: str
    ending_soon_sql: str
    max_rows_each: int
    clickhouse_db: str = ''
    last_hour_sql: str = ''
    latest_sql: str = ''
    high_volume_sql: str = ''
    fresh_sql: str = ''
    last_week_sql: str = ''
    watch_density_sql: str = ''
    high_traffic_sql: str = ''
    bid_recency_horizon_seconds: float = 3600.0
    high_volume_score_norm: float = 100.0
    watch_density_score_norm: float = 0.0
    high_traffic_score_norm: float = 0.0
    fresh_time_horizon_seconds: float = 604800.0
    hard_filter_pushdown: Optional[HardFilterPushdownConfig] = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("explore.clickhouse_rails.enabled must be a bool")
        if not isinstance(self.trending_sql, str):
            raise ConfigurationError("explore.clickhouse_rails.trending_sql must be a string")
        if not isinstance(self.ending_soon_sql, str):
            raise ConfigurationError("explore.clickhouse_rails.ending_soon_sql must be a string")
        if self.enabled:
            if not self.trending_sql.strip():
                raise ConfigurationError("explore.clickhouse_rails.trending_sql must be non-empty when enabled=true")
            if not self.ending_soon_sql.strip():
                raise ConfigurationError("explore.clickhouse_rails.ending_soon_sql must be non-empty when enabled=true")
        if int(self.max_rows_each) < 1:
            raise ConfigurationError("explore.clickhouse_rails.max_rows_each must be >= 1")
        if not isinstance(self.bid_recency_horizon_seconds, (int, float)) or float(self.bid_recency_horizon_seconds) <= 0.0:
            raise ConfigurationError("explore.clickhouse_rails.bid_recency_horizon_seconds must be > 0")
        if not isinstance(self.high_volume_score_norm, (int, float)) or float(self.high_volume_score_norm) <= 0.0:
            raise ConfigurationError("explore.clickhouse_rails.high_volume_score_norm must be > 0")
        if not isinstance(self.fresh_time_horizon_seconds, (int, float)) or float(self.fresh_time_horizon_seconds) <= 0.0:
            raise ConfigurationError("explore.clickhouse_rails.fresh_time_horizon_seconds must be > 0")
        if not isinstance(self.watch_density_score_norm, (int, float)) or float(self.watch_density_score_norm) < 0:
            raise ConfigurationError("explore.clickhouse_rails.watch_density_score_norm must be >= 0")
        if self.watch_density_sql.strip() and float(self.watch_density_score_norm) <= 0.0:
            raise ConfigurationError("explore.clickhouse_rails.watch_density_score_norm must be > 0 when watch_density_sql is set")
        if not isinstance(self.high_traffic_score_norm, (int, float)) or float(self.high_traffic_score_norm) < 0:
            raise ConfigurationError("explore.clickhouse_rails.high_traffic_score_norm must be >= 0")
        if self.high_traffic_sql.strip() and float(self.high_traffic_score_norm) <= 0.0:
            raise ConfigurationError("explore.clickhouse_rails.high_traffic_score_norm must be > 0 when high_traffic_sql is set")
        self.bid_recency_horizon_seconds = float(self.bid_recency_horizon_seconds)
        self.high_volume_score_norm = float(self.high_volume_score_norm)
        self.watch_density_score_norm = float(self.watch_density_score_norm)
        self.high_traffic_score_norm = float(self.high_traffic_score_norm)
        self.fresh_time_horizon_seconds = float(self.fresh_time_horizon_seconds)
        if self.clickhouse_db:
            _sql_fields = [
                'trending_sql', 'ending_soon_sql', 'last_hour_sql', 'latest_sql',
                'high_volume_sql', 'fresh_sql', 'last_week_sql',
                'watch_density_sql', 'high_traffic_sql',
            ]
            for _f in _sql_fields:
                _v = getattr(self, _f, '')
                if _v and '{clickhouse_db}' in _v:
                    setattr(self, _f, _v.replace('{clickhouse_db}', self.clickhouse_db))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ExploreClickHouseRailsConfig':
        _require(d, ['enabled', 'trending_sql', 'ending_soon_sql', 'max_rows_each', 'clickhouse_db'], 'explore.clickhouse_rails')
        db = str(d['clickhouse_db'])
        _ht_bid_w = float(d.get('high_traffic_bid_weight') or 1.0)
        _ht_watch_w = float(d.get('high_traffic_watch_weight') or 1.0)
        def _sub(sql: str) -> str:
            return (sql
                    .replace('{clickhouse_db}', db)
                    .replace('{high_traffic_bid_weight}', str(_ht_bid_w))
                    .replace('{high_traffic_watch_weight}', str(_ht_watch_w))) if db else sql
        pushdown_raw = d.get('hard_filter_pushdown')
        hard_filter_pushdown = (
            HardFilterPushdownConfig.from_dict(pushdown_raw)
            if isinstance(pushdown_raw, dict) else None
        )
        return cls(
            enabled=bool(d['enabled']),
            clickhouse_db=db,
            trending_sql=_sub(str(d['trending_sql'])),
            ending_soon_sql=_sub(str(d['ending_soon_sql'])),
            max_rows_each=int(d['max_rows_each']),
            last_hour_sql=_sub(str(d.get('last_hour_sql', '') or '')),
            latest_sql=_sub(str(d.get('latest_sql', '') or '')),
            high_volume_sql=_sub(str(d.get('high_volume_sql', '') or '')),
            fresh_sql=_sub(str(d.get('fresh_sql', '') or '')),
            last_week_sql=_sub(str(d.get('last_week_sql', '') or '')),
            watch_density_sql=_sub(str(d.get('watch_density_sql', '') or '')),
            high_traffic_sql=_sub(str(d.get('high_traffic_sql', '') or '')),
            bid_recency_horizon_seconds=float(d.get('bid_recency_horizon_seconds') or 3600.0),
            high_volume_score_norm=float(d.get('high_volume_score_norm') or 100.0),
            watch_density_score_norm=float(d.get('watch_density_score_norm') or 0.0),
            high_traffic_score_norm=float(d.get('high_traffic_score_norm') or 0.0),
            fresh_time_horizon_seconds=float(d.get('fresh_time_horizon_seconds') or 604800.0),
            hard_filter_pushdown=hard_filter_pushdown,
        )


@dataclass
class ExploreConfig:
    """Top-level explore + zero-result guard config.
    :param enabled: bool - Master toggle for /landing-rail and the guard
    :param trending: TrendingRailConfig
    :param ending_soon: EndingSoonRailConfig
    :param fallback: FallbackRailConfig
    :param zero_result_guard: ZeroResultGuardConfig
    :param clickhouse_rails: ExploreClickHouseRailsConfig - Optional CH-backed rails
    """
    enabled: bool
    trending: TrendingRailConfig
    ending_soon: EndingSoonRailConfig
    fallback: FallbackRailConfig
    zero_result_guard: ZeroResultGuardConfig
    clickhouse_rails: ExploreClickHouseRailsConfig

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("explore.enabled must be bool")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ExploreConfig':
        _require(d, ['enabled', 'trending', 'ending_soon', 'fallback', 'zero_result_guard', 'clickhouse_rails'], 'explore')
        return cls(
            enabled=bool(d['enabled']),
            trending=TrendingRailConfig.from_dict(d['trending']),
            ending_soon=EndingSoonRailConfig.from_dict(d['ending_soon']),
            fallback=FallbackRailConfig.from_dict(d['fallback']),
            zero_result_guard=ZeroResultGuardConfig.from_dict(d['zero_result_guard']),
            clickhouse_rails=ExploreClickHouseRailsConfig.from_dict(d['clickhouse_rails']),
        )





@dataclass
class SanitizerConfig:
    """Layer-0 sanitizer config — reused on retrieved-content paths and at LLM ingress.

    When ``applies_to_llm_ingress=True`` (default), ``LLMCallRouter.call_structured``
    runs both ``system_prompt`` and ``user_prompt`` through this sanitizer BEFORE
    any client call fires; a block on either side raises ``LLMError`` so no
    provider call goes out — closes the indirect prompt-injection vector at the
    single chokepoint that the QI classifier and NL-SQL gen + verifier flow through.

    Two separate length caps:
    - ``max_chars``: applied to the user-prompt side (user-supplied query text).
    - ``system_max_chars``: applied to the system-prompt side (developer-authored
      prompt templates). Must be high enough to accommodate the full QI classify
      system prompt (~8 KB) and NL-SQL generation system prompts.
    """
    enabled: bool
    max_chars: int
    system_max_chars: int
    blocked_patterns: List[str]
    pii_patterns: List[str]
    applies_to_llm_ingress: bool
    encoding_normalize: bool = True

    def __post_init__(self) -> None:
        if self.max_chars < 16:
            raise ConfigurationError("safety.ingress_sanitizer.max_chars must be >= 16")
        if self.system_max_chars < self.max_chars:
            raise ConfigurationError("safety.ingress_sanitizer.system_max_chars must be >= max_chars")
        if not isinstance(self.blocked_patterns, list):
            raise ConfigurationError("safety.ingress_sanitizer.blocked_patterns must be a list")
        if not isinstance(self.pii_patterns, list):
            raise ConfigurationError("safety.ingress_sanitizer.pii_patterns must be a list")
        if not isinstance(self.applies_to_llm_ingress, bool):
            raise ConfigurationError("safety.ingress_sanitizer.applies_to_llm_ingress must be a bool")
        if not isinstance(self.encoding_normalize, bool):
            raise ConfigurationError("safety.ingress_sanitizer.encoding_normalize must be a bool")

    @classmethod
    def from_dict(cls, d: Dict[str, Any], *, section: str = 'safety.ingress_sanitizer') -> 'SanitizerConfig':
        _require(d, ['enabled', 'max_chars', 'blocked_patterns', 'pii_patterns', 'applies_to_llm_ingress', 'encoding_normalize'], section)
        return cls(
            enabled=bool(d['enabled']),
            max_chars=int(d['max_chars']),
            system_max_chars=int(d.get('system_max_chars', 50000)),
            blocked_patterns=[str(p) for p in d['blocked_patterns']],
            pii_patterns=[str(p) for p in d['pii_patterns']],
            applies_to_llm_ingress=bool(d['applies_to_llm_ingress']),
            encoding_normalize=bool(d['encoding_normalize']),
        )


@dataclass
class OfflineEvalConfig:
    """Offline retrieval-eval + LLM-judge hooks (library-only; no HTTP surface)."""
    retrieval_eval: RetrievalEvalConfig
    llm_judge: Optional[LLMJudgeConfig]

    def __post_init__(self) -> None:
        if not isinstance(self.retrieval_eval, RetrievalEvalConfig):
            raise ConfigurationError("offline_eval.retrieval_eval must be a RetrievalEvalConfig")
        if self.llm_judge is not None and not isinstance(self.llm_judge, LLMJudgeConfig):
            raise ConfigurationError("offline_eval.llm_judge must be an LLMJudgeConfig or null")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'OfflineEvalConfig':
        _require(d, ['retrieval_eval'], 'offline_eval')
        lj_raw = d.get('llm_judge')
        llm_judge = LLMJudgeConfig.from_dict(lj_raw) if isinstance(lj_raw, dict) else None
        return cls(retrieval_eval=RetrievalEvalConfig.from_dict(d['retrieval_eval']), llm_judge=llm_judge)


@dataclass
class CircuitBreakerConfig:
    """LLM circuit-breaker config."""
    enabled: bool
    failure_rate_threshold: float
    min_calls_before_trip: int
    rolling_window_seconds: float
    open_cooldown_seconds: float
    half_open_probe_ratio: float
    half_open_required_successes: int

    def __post_init__(self) -> None:
        if not 0.0 < self.failure_rate_threshold <= 1.0:
            raise ConfigurationError("resilience.circuit_breaker.failure_rate_threshold must be in (0,1]")
        if self.min_calls_before_trip < 1:
            raise ConfigurationError("resilience.circuit_breaker.min_calls_before_trip must be >= 1")
        if self.rolling_window_seconds <= 0.0:
            raise ConfigurationError("resilience.circuit_breaker.rolling_window_seconds must be > 0")
        if self.open_cooldown_seconds < 0.0:
            raise ConfigurationError("resilience.circuit_breaker.open_cooldown_seconds must be >= 0")
        if not 0.0 < self.half_open_probe_ratio <= 1.0:
            raise ConfigurationError("resilience.circuit_breaker.half_open_probe_ratio must be in (0,1]")
        if self.half_open_required_successes < 1:
            raise ConfigurationError("resilience.circuit_breaker.half_open_required_successes must be >= 1")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CircuitBreakerConfig':
        _require(d, ['enabled', 'failure_rate_threshold', 'min_calls_before_trip', 'rolling_window_seconds', 'open_cooldown_seconds', 'half_open_probe_ratio', 'half_open_required_successes'], 'resilience.circuit_breaker')  # noqa: E501
        return cls(
            enabled=bool(d['enabled']),
            failure_rate_threshold=float(d['failure_rate_threshold']),
            min_calls_before_trip=int(d['min_calls_before_trip']),
            rolling_window_seconds=float(d['rolling_window_seconds']),
            open_cooldown_seconds=float(d['open_cooldown_seconds']),
            half_open_probe_ratio=float(d['half_open_probe_ratio']),
            half_open_required_successes=int(d['half_open_required_successes']),
        )


@dataclass
class BackendHealthConfig:
    """Backend health-tracker config.

    :param force_unhealthy_backends: List[str] - Backend names forced to unhealthy
        regardless of probe window (empty = no force). Used for local CH-off rails checks.
    """
    enabled: bool
    failure_rate_threshold: float
    rolling_window_seconds: float
    min_observations: int
    recovery_probe_seconds: float
    force_unhealthy_backends: List[str]

    def __post_init__(self) -> None:
        if not 0.0 < self.failure_rate_threshold <= 1.0:
            raise ConfigurationError("resilience.backend_health.failure_rate_threshold must be in (0,1]")
        if self.rolling_window_seconds <= 0.0:
            raise ConfigurationError("resilience.backend_health.rolling_window_seconds must be > 0")
        if self.min_observations < 1:
            raise ConfigurationError("resilience.backend_health.min_observations must be >= 1")
        if self.recovery_probe_seconds < 0.0:
            raise ConfigurationError("resilience.backend_health.recovery_probe_seconds must be >= 0")
        if not isinstance(self.force_unhealthy_backends, list) or not all(
            isinstance(b, str) and b for b in self.force_unhealthy_backends
        ):
            raise ConfigurationError(
                "resilience.backend_health.force_unhealthy_backends must be a list of non-empty strings"
            )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'BackendHealthConfig':
        _require(
            d,
            [
                'enabled',
                'failure_rate_threshold',
                'rolling_window_seconds',
                'min_observations',
                'recovery_probe_seconds',
                'force_unhealthy_backends',
            ],
            'resilience.backend_health',
        )
        return cls(
            enabled=bool(d['enabled']),
            failure_rate_threshold=float(d['failure_rate_threshold']),
            rolling_window_seconds=float(d['rolling_window_seconds']),
            min_observations=int(d['min_observations']),
            recovery_probe_seconds=float(d['recovery_probe_seconds']),
            force_unhealthy_backends=[str(b).strip() for b in d['force_unhealthy_backends']],
        )


@dataclass
class DegradationConfig:
    """Degradation policy (per-backend mapping).

    `fallback_chains` maps an unhealthy retrieval-backend name to the ordered
    list of replacements the planner should try. Adding a new retrieval backend
    requires only a YAML edit (no code change), satisfying the open/closed
    principle for extensibility.

    Allowed retrieval-backend keys (from `semantic_search.contracts.CANDIDATE_SOURCES`):
        vector, structured, sql

    Allowed fallback values: any retrieval-backend key plus the markers
    `cache_only` and `none`.
    """
    enabled: bool
    fallback_chains: Dict[str, List[str]]
    allow_empty_when_all_unhealthy: bool

    _VALID_BACKEND_KEYS = frozenset({'vector', 'structured', 'sql'})
    _VALID_FALLBACK_VALUES = frozenset({'vector', 'structured', 'sql', 'cache_only', 'none'})

    def __post_init__(self) -> None:
        if not isinstance(self.fallback_chains, dict):
            raise ConfigurationError("resilience.degradation.fallback_chains must be a dict")
        if not self.fallback_chains:
            raise ConfigurationError("resilience.degradation.fallback_chains must be non-empty")
        for backend, chain in self.fallback_chains.items():
            if backend not in self._VALID_BACKEND_KEYS:
                raise ConfigurationError(f"resilience.degradation.fallback_chains contains invalid backend key '{backend}'; allowed={sorted(self._VALID_BACKEND_KEYS)}")
            if not isinstance(chain, list):
                raise ConfigurationError(f"resilience.degradation.fallback_chains.{backend} must be a list")
            for v in chain:
                if v not in self._VALID_FALLBACK_VALUES:
                    raise ConfigurationError(f"resilience.degradation.fallback_chains.{backend} contains invalid fallback '{v}'; allowed={sorted(self._VALID_FALLBACK_VALUES)}")
                if v == backend:
                    raise ConfigurationError(f"resilience.degradation.fallback_chains.{backend} must not list itself as a fallback")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DegradationConfig':
        _require(d, ['enabled', 'fallback_chains', 'allow_empty_when_all_unhealthy'], 'resilience.degradation')
        chains_in = d['fallback_chains']
        if not isinstance(chains_in, dict):
            raise ConfigurationError("resilience.degradation.fallback_chains must be a dict")
        chains = {str(k): [str(v) for v in vlist] for k, vlist in chains_in.items()}
        return cls( enabled=bool(d['enabled']), fallback_chains=chains, allow_empty_when_all_unhealthy=bool(d['allow_empty_when_all_unhealthy']), )


@dataclass
class RetryConfig:
    """Retry hierarchy."""
    enabled: bool
    llm_max_retries: int
    llm_retry_backoff_ms: int
    backend_max_retries: int

    def __post_init__(self) -> None:
        if self.llm_max_retries < 0:
            raise ConfigurationError("resilience.retry.llm_max_retries must be >= 0")
        if self.llm_retry_backoff_ms < 0:
            raise ConfigurationError("resilience.retry.llm_retry_backoff_ms must be >= 0")
        if self.backend_max_retries < 0:
            raise ConfigurationError("resilience.retry.backend_max_retries must be >= 0")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'RetryConfig':
        _require(d, ['enabled', 'llm_max_retries', 'llm_retry_backoff_ms', 'backend_max_retries'], 'resilience.retry')
        return cls( enabled=bool(d['enabled']), llm_max_retries=int(d['llm_max_retries']), llm_retry_backoff_ms=int(d['llm_retry_backoff_ms']), backend_max_retries=int(d['backend_max_retries']), )


@dataclass
class ResilienceConfig:
    """Resilience subsystem bundle (breakers, health registry, degradation)."""
    enabled: bool
    circuit_breaker: CircuitBreakerConfig
    backend_health: BackendHealthConfig
    degradation: DegradationConfig
    retry: RetryConfig

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ResilienceConfig':
        _require(d, ['enabled', 'circuit_breaker', 'backend_health', 'degradation', 'retry'], 'resilience')
        return cls(
            enabled=bool(d['enabled']),
            circuit_breaker=CircuitBreakerConfig.from_dict(d['circuit_breaker']),
            backend_health=BackendHealthConfig.from_dict(d['backend_health']),
            degradation=DegradationConfig.from_dict(d['degradation']),
            retry=RetryConfig.from_dict(d['retry']),
        )


@dataclass
class MeasurementThresholdsConfig:
    """Proxy-signal thresholds (one knob per signal we compute).

    Every signal documented in maps to a field here.
    Signals that the platform cannot yet observe (offline-only, surface-only,
    Pillar-2 features not yet built) are not represented — the evaluator emits
    them with status='not_instrumented' instead of inventing a number.
    """
    zero_result_rate_max: float
    filter_override_rate_max: float
    query_to_click_rate_min: float
    cache_hit_rate_min: float
    cache_hit_rate_tiers: List[str]
    high_confidence_rate_min: float
    p50_latency_ms_max: float
    p99_latency_ms_max: float
    feedback_signals_per_day_min: int
    circuit_breaker_activations_per_week_max: int
    regex_short_circuit_rate_min: float
    sanitizer_rejection_rate_max: float
    multi_intent_duplicate_rate_max: float
    high_confidence_band_min: float
    correct_intent_confidence_min: float
    cost_per_high_confidence_query_alert_multiplier: float
    # launch-blocking KPI: ≥ 0.50
    # within 4 weeks of launch. The proxy denominator is non-zero `prompt_tokens`
    # observed by the router (covers Tier-3 + NL-SQL gen + verifier +
    # refinement).
    prompt_cache_hit_rate_min: float
    # ≥ 0.85 once warm. Denominator is
    # analytics-router fast-path eligibility decisions (skip + invocation count).
    verifier_skip_rate_min: float

    def __post_init__(self) -> None:
        for name, value in (
            ('zero_result_rate_max', self.zero_result_rate_max),
            ('filter_override_rate_max', self.filter_override_rate_max),
            ('query_to_click_rate_min', self.query_to_click_rate_min),
            ('cache_hit_rate_min', self.cache_hit_rate_min),
            ('high_confidence_rate_min', self.high_confidence_rate_min),
            ('regex_short_circuit_rate_min', self.regex_short_circuit_rate_min),
            ('sanitizer_rejection_rate_max', self.sanitizer_rejection_rate_max),
            ('multi_intent_duplicate_rate_max', self.multi_intent_duplicate_rate_max),
            ('high_confidence_band_min', self.high_confidence_band_min),
            ('correct_intent_confidence_min', self.correct_intent_confidence_min),
            ('prompt_cache_hit_rate_min', self.prompt_cache_hit_rate_min),
            ('verifier_skip_rate_min', self.verifier_skip_rate_min),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ConfigurationError(f"measurement.thresholds.{name} must be in [0,1]")
        if float(self.p50_latency_ms_max) <= 0.0 or float(self.p99_latency_ms_max) <= 0.0:
            raise ConfigurationError("measurement.thresholds.p50/p99_latency_ms_max must be > 0")
        if float(self.p99_latency_ms_max) < float(self.p50_latency_ms_max):
            raise ConfigurationError("measurement.thresholds.p99_latency_ms_max must be >= p50_latency_ms_max")
        if int(self.feedback_signals_per_day_min) < 0:
            raise ConfigurationError("measurement.thresholds.feedback_signals_per_day_min must be >= 0")
        if int(self.circuit_breaker_activations_per_week_max) < 0:
            raise ConfigurationError("measurement.thresholds.circuit_breaker_activations_per_week_max must be >= 0")
        if float(self.cost_per_high_confidence_query_alert_multiplier) <= 1.0:
            raise ConfigurationError("measurement.thresholds.cost_per_high_confidence_query_alert_multiplier must be > 1.0")
        if not isinstance(self.cache_hit_rate_tiers, list) or not self.cache_hit_rate_tiers:
            raise ConfigurationError("measurement.thresholds.cache_hit_rate_tiers must be a non-empty list")
        for tier in self.cache_hit_rate_tiers:
            if not isinstance(tier, str) or not tier.strip():
                raise ConfigurationError(
                    "measurement.thresholds.cache_hit_rate_tiers entries must be non-empty strings"
                )
        self.cache_hit_rate_tiers = [str(t).strip() for t in self.cache_hit_rate_tiers]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'MeasurementThresholdsConfig':
        required = [
            'zero_result_rate_max', 'filter_override_rate_max', 'query_to_click_rate_min',
            'cache_hit_rate_min', 'cache_hit_rate_tiers', 'high_confidence_rate_min',
            'p50_latency_ms_max',
            'p99_latency_ms_max', 'feedback_signals_per_day_min',
            'circuit_breaker_activations_per_week_max', 'regex_short_circuit_rate_min',
            'sanitizer_rejection_rate_max', 'multi_intent_duplicate_rate_max',
            'high_confidence_band_min', 'correct_intent_confidence_min',
            'cost_per_high_confidence_query_alert_multiplier',
            'prompt_cache_hit_rate_min', 'verifier_skip_rate_min',
        ]
        _require(d, required, 'measurement.thresholds')
        tiers = d['cache_hit_rate_tiers']
        if not isinstance(tiers, list):
            raise ConfigurationError("measurement.thresholds.cache_hit_rate_tiers must be a list in YAML")
        return cls(
            zero_result_rate_max=float(d['zero_result_rate_max']),
            filter_override_rate_max=float(d['filter_override_rate_max']),
            query_to_click_rate_min=float(d['query_to_click_rate_min']),
            cache_hit_rate_min=float(d['cache_hit_rate_min']),
            cache_hit_rate_tiers=[str(t) for t in tiers],
            high_confidence_rate_min=float(d['high_confidence_rate_min']),
            p50_latency_ms_max=float(d['p50_latency_ms_max']),
            p99_latency_ms_max=float(d['p99_latency_ms_max']),
            feedback_signals_per_day_min=int(d['feedback_signals_per_day_min']),
            circuit_breaker_activations_per_week_max=int(d['circuit_breaker_activations_per_week_max']),
            regex_short_circuit_rate_min=float(d['regex_short_circuit_rate_min']),
            sanitizer_rejection_rate_max=float(d['sanitizer_rejection_rate_max']),
            multi_intent_duplicate_rate_max=float(d['multi_intent_duplicate_rate_max']),
            high_confidence_band_min=float(d['high_confidence_band_min']),
            correct_intent_confidence_min=float(d['correct_intent_confidence_min']),
            cost_per_high_confidence_query_alert_multiplier=float(d['cost_per_high_confidence_query_alert_multiplier']),
            prompt_cache_hit_rate_min=float(d['prompt_cache_hit_rate_min']),
            verifier_skip_rate_min=float(d['verifier_skip_rate_min']),
        )


@dataclass
class MeasurementSlicingConfig:
    """slicing configuration for ProxySignalEvaluator.evaluate_sliced.

    When enabled, the evaluator emits a per-bucket variant of every rate-bearing
    signal that is computed off the SearchObservation window. The bucket is
    chosen via `slice_by` (today only `query_type` is supported — see
    contracts.MEASUREMENT_SLICE_KEYS). `min_per_bucket_sample` is the
    per-bucket analogue of MeasurementConfig.min_sample_size: a bucket whose
    in-window count falls below this floor is emitted as `insufficient_data`
    rather than producing a noisy rate from a handful of observations.

    :param enabled: bool - Master toggle. Default False keeps the legacy single
        global report as the sole evaluator output.
    :param slice_by: str - Slice attribute (must be a member of
        contracts.MEASUREMENT_SLICE_KEYS). Today: 'query_type'.
    :param min_per_bucket_sample: int - Per-bucket sample floor (>= 1). Should
        be <= MeasurementConfig.min_sample_size; the evaluator does NOT take
        max(global_floor, bucket_floor) because a per-bucket analysis is the
        whole point of the slicing surface.
    """
    enabled: bool
    slice_by: str
    min_per_bucket_sample: int

    def __post_init__(self) -> None:
        if not isinstance(self.slice_by, str):
            raise ConfigurationError("measurement.slicing.slice_by must be a string")
        if self.slice_by not in MEASUREMENT_SLICE_KEYS:
            raise ConfigurationError( f"measurement.slicing.slice_by must be one of {sorted(MEASUREMENT_SLICE_KEYS)}" )
        if int(self.min_per_bucket_sample) < 1:
            raise ConfigurationError("measurement.slicing.min_per_bucket_sample must be >= 1")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'MeasurementSlicingConfig':
        _require(d, ['enabled', 'slice_by', 'min_per_bucket_sample'], 'measurement.slicing')
        return cls( enabled=bool(d['enabled']), slice_by=str(d['slice_by']), min_per_bucket_sample=int(d['min_per_bucket_sample']), )


@dataclass
class AssistedConversionConfig:
    """search_assisted_conversion_rate proxy-signal config.

    A search "converts" when the same `session_id` issues at least one
    feedback signal whose `signal_type` is in `positive_signal_types` within
    `attribution_window_seconds` AFTER the search was recorded by the
    measurement store. Numerator: searches with at least one matching
    follow-up; denominator: searches with a non-empty `session_id` (anonymous
    searches are excluded from the denominator). The min-sample floor uses the denominator
    so a narrow post-search window does not produce a rate that looks tracked.

    :param enabled: bool - Master toggle. Default False keeps the signal as
        `not_instrumented` (consistent with how the evaluator handles other
        not-yet-built capabilities).
    :param attribution_window_seconds: float - Per-session post-search window
        (> 0). Inside this window, any positive-signal arrival counts the
        originating search as converted.
    :param positive_signal_types: List[str] - Signal types that count as
        positive engagement. Each entry must be a member of
        contracts.ASSISTED_CONVERSION_POSITIVE_SIGNALS.
    :param threshold_min: float - Minimum acceptable conversion rate in [0, 1].
        Values below trigger status='breach'.
    """
    enabled: bool
    attribution_window_seconds: float
    positive_signal_types: List[str]
    threshold_min: float

    def __post_init__(self) -> None:
        if float(self.attribution_window_seconds) <= 0.0:
            raise ConfigurationError("measurement.assisted_conversion.attribution_window_seconds must be > 0")
        if not isinstance(self.positive_signal_types, list) or len(self.positive_signal_types) == 0:
            raise ConfigurationError("measurement.assisted_conversion.positive_signal_types must be a non-empty list")
        seen: set = set()
        for st in self.positive_signal_types:
            if not isinstance(st, str) or not st:
                raise ConfigurationError("measurement.assisted_conversion.positive_signal_types entries must be non-empty strings")
            if st not in ASSISTED_CONVERSION_POSITIVE_SIGNALS:
                raise ConfigurationError( f"measurement.assisted_conversion.positive_signal_types entry '{st}' must be one of " f"{sorted(ASSISTED_CONVERSION_POSITIVE_SIGNALS)}" )
            if st in seen:
                raise ConfigurationError( f"measurement.assisted_conversion.positive_signal_types entry '{st}' appears more than once" )
            seen.add(st)
        if not 0.0 <= float(self.threshold_min) <= 1.0:
            raise ConfigurationError("measurement.assisted_conversion.threshold_min must be in [0,1]")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'AssistedConversionConfig':
        _require(d, ['enabled', 'attribution_window_seconds', 'positive_signal_types', 'threshold_min'], 'measurement.assisted_conversion')
        raw_types = d['positive_signal_types']
        if not isinstance(raw_types, list):
            raise ConfigurationError("measurement.assisted_conversion.positive_signal_types must be a list")
        return cls(
            enabled=bool(d['enabled']),
            attribution_window_seconds=float(d['attribution_window_seconds']),
            positive_signal_types=[str(s) for s in raw_types],
            threshold_min=float(d['threshold_min']),
        )


@dataclass
class RankingStageAttributionConfig:
    """Per-request ranking stage counters for logs and pipeline_trace.

    All fields required in YAML — no in-code defaults.

    :param enabled: bool - Collect stage counters on the search path.
    :param include_in_response: bool - Attach ``pipeline_trace.stages`` when true.
    :param log_event: bool - Emit ``ranking_stage_attribution`` structured log line.
    """
    enabled: bool
    include_in_response: bool
    log_event: bool

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("measurement.ranking_stage_attribution.enabled must be a bool")
        if not isinstance(self.include_in_response, bool):
            raise ConfigurationError(
                "measurement.ranking_stage_attribution.include_in_response must be a bool"
            )
        if not isinstance(self.log_event, bool):
            raise ConfigurationError("measurement.ranking_stage_attribution.log_event must be a bool")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'RankingStageAttributionConfig':
        _require(
            d,
            ['enabled', 'include_in_response', 'log_event'],
            'measurement.ranking_stage_attribution',
        )
        return cls(
            enabled=bool(d['enabled']),
            include_in_response=bool(d['include_in_response']),
            log_event=bool(d['log_event']),
        )


@dataclass
class QieOnlyLaunchConfig:
    """Phase 1 qie_only launch-gate thresholds exposed by /measurement/qie_only."""
    min_sample_size: int
    fail_rate_max: float
    p95_latency_ms_max: float
    find_skipped_rate_max: float
    hard_params_empty_rate_max: float
    latency_window: int

    def __post_init__(self) -> None:
        if int(self.min_sample_size) < 1:
            raise ConfigurationError("measurement.qie_only_launch.min_sample_size must be >= 1")
        if int(self.latency_window) < 10:
            raise ConfigurationError("measurement.qie_only_launch.latency_window must be >= 10")
        for name, value in (
            ('fail_rate_max', self.fail_rate_max),
            ('find_skipped_rate_max', self.find_skipped_rate_max),
            ('hard_params_empty_rate_max', self.hard_params_empty_rate_max),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ConfigurationError(f"measurement.qie_only_launch.{name} must be in [0,1]")
        if float(self.p95_latency_ms_max) <= 0.0:
            raise ConfigurationError("measurement.qie_only_launch.p95_latency_ms_max must be > 0")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QieOnlyLaunchConfig':
        _require(
            d,
            [
                'min_sample_size',
                'fail_rate_max',
                'p95_latency_ms_max',
                'find_skipped_rate_max',
                'hard_params_empty_rate_max',
                'latency_window',
            ],
            'measurement.qie_only_launch',
        )
        return cls(
            min_sample_size=int(d['min_sample_size']),
            fail_rate_max=float(d['fail_rate_max']),
            p95_latency_ms_max=float(d['p95_latency_ms_max']),
            find_skipped_rate_max=float(d['find_skipped_rate_max']),
            hard_params_empty_rate_max=float(d['hard_params_empty_rate_max']),
            latency_window=int(d['latency_window']),
        )


@dataclass
class MeasurementConfig:
    """Measurement subsystem config.

    The MeasurementStore keeps the most-recent `window_size` SearchObservation
    instances in memory. The evaluator computes proxy signals over those
    observations plus snapshot state from sibling subsystems (cache, signal_store,
    sanitizer, circuit_breaker). `min_sample_size` guards against publishing
    metrics before the window has accumulated enough data to be meaningful.

    Two optional sub-blocks (both default None / disabled):
    - `slicing`: enables ProxySignalEvaluator.evaluate_sliced(by='query_type')
    - `assisted_conversion`: enables the search_assisted_conversion_rate signal
    Both are additive. When absent, the evaluator behaviour is unchanged from
    the pre-Rec-#11 baseline.

    ``ranking_stage_attribution`` is required: collects retrieve / fuse / hard-gate /
    soft-boost / eRanker / diversify / zero-result counters for attribution.
    """
    enabled: bool
    window_size: int
    min_sample_size: int
    explore_followup_window_seconds: float
    thresholds: MeasurementThresholdsConfig
    ranking_stage_attribution: RankingStageAttributionConfig
    qie_only_launch: QieOnlyLaunchConfig
    slicing: Optional[MeasurementSlicingConfig] = None
    assisted_conversion: Optional[AssistedConversionConfig] = None
    cache_miss_storm: Optional['CacheMissStormConfig'] = None

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ConfigurationError("measurement.window_size must be >= 1")
        if self.min_sample_size < 1:
            raise ConfigurationError("measurement.min_sample_size must be >= 1")
        if self.min_sample_size > self.window_size:
            raise ConfigurationError("measurement.min_sample_size must be <= window_size")
        if float(self.explore_followup_window_seconds) <= 0.0:
            raise ConfigurationError("measurement.explore_followup_window_seconds must be > 0")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'MeasurementConfig':
        _require(
            d,
            [
                'enabled',
                'window_size',
                'min_sample_size',
                'explore_followup_window_seconds',
                'thresholds',
                'ranking_stage_attribution',
                'qie_only_launch',
            ],
            'measurement',
        )
        slicing_raw = d.get('slicing') if isinstance(d, dict) else None
        slicing_cfg = MeasurementSlicingConfig.from_dict(slicing_raw) if isinstance(slicing_raw, dict) else None
        ac_raw = d.get('assisted_conversion') if isinstance(d, dict) else None
        ac_cfg = AssistedConversionConfig.from_dict(ac_raw) if isinstance(ac_raw, dict) else None
        cms_raw = d.get('cache_miss_storm') if isinstance(d, dict) else None
        cms_cfg = CacheMissStormConfig.from_dict(cms_raw) if isinstance(cms_raw, dict) else None
        return cls(
            enabled=bool(d['enabled']),
            window_size=int(d['window_size']),
            min_sample_size=int(d['min_sample_size']),
            explore_followup_window_seconds=float(d['explore_followup_window_seconds']),
            thresholds=MeasurementThresholdsConfig.from_dict(d['thresholds']),
            ranking_stage_attribution=RankingStageAttributionConfig.from_dict(
                d['ranking_stage_attribution']
            ),
            qie_only_launch=QieOnlyLaunchConfig.from_dict(d['qie_only_launch']),
            slicing=slicing_cfg,
            assisted_conversion=ac_cfg,
            cache_miss_storm=cms_cfg,
        )


@dataclass
class CacheMissStormConfig:
    """Cache miss-storm alarm config.

    Parameterises the rolling-window detector that emits a ``cache_miss_storm``
    FeedbackSignal when hit rate drops below the configured threshold.

    :param enabled: bool - Master toggle. When false, the detector is
        constructed but skips evaluation (poll() returns an unbreached
        status without recording samples). Allows ops to silence the
        alarm without rewiring the registry.
    :param window_seconds: float - Rolling window the hit rate is
        computed over. Plan default 300 (5 min).
    :param hit_rate_min: float - Floor below which the alarm fires.
        Plan default 0.30. Must be in (0, 1).
    :param min_sample_size: int - Minimum (hits + misses) within the
        window before the detector emits a verdict. Below this floor the
        detector reports ``insufficient_data`` instead of either state.
        Plan-implied: small samples should not produce a 0/100 rate.
    :param tiers: List[str] - Cache tiers the detector observes from
        ``cache_stats`` keys. Structured intermediate cache is typically
        omitted (different keying / traffic class). Align with
        ``measurement.thresholds.cache_hit_rate_tiers`` when both are used.
    """
    enabled: bool
    window_seconds: float
    hit_rate_min: float
    min_sample_size: int
    tiers: List[str]

    def __post_init__(self) -> None:
        if float(self.window_seconds) <= 0.0:
            raise ConfigurationError("measurement.cache_miss_storm.window_seconds must be > 0")
        if not (0.0 < float(self.hit_rate_min) < 1.0):
            raise ConfigurationError("measurement.cache_miss_storm.hit_rate_min must be in (0, 1)")
        if int(self.min_sample_size) < 1:
            raise ConfigurationError("measurement.cache_miss_storm.min_sample_size must be >= 1")
        if not isinstance(self.tiers, list) or not self.tiers:
            raise ConfigurationError("measurement.cache_miss_storm.tiers must be a non-empty list")
        for t in self.tiers:
            if not isinstance(t, str) or not t:
                raise ConfigurationError("measurement.cache_miss_storm.tiers entries must be non-empty strings")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CacheMissStormConfig':
        _require(d, ['enabled', 'window_seconds', 'hit_rate_min', 'min_sample_size', 'tiers'], 'measurement.cache_miss_storm')
        tiers = d['tiers']
        if not isinstance(tiers, list):
            raise ConfigurationError("measurement.cache_miss_storm.tiers must be a list in YAML")
        return cls(
            enabled=bool(d['enabled']),
            window_seconds=float(d['window_seconds']),
            hit_rate_min=float(d['hit_rate_min']),
            min_sample_size=int(d['min_sample_size']),
            tiers=[str(t) for t in tiers],
        )


@dataclass
class MultiIntentConfig:
    """MultiIntentSplitter config.

    L0 deterministic splitter that fires before classification. Splits on
    AND/OR/comma/semicolon connectives, fans classify() across sub-queries in
    parallel, drops sub-intents with zero expected results, applies a 5-cap by
    weighted ranking (0.5 * confidence + 0.3 * expected_results_norm + 0.2 *
    specificity), and surfaces a sub-intent badge on each card.

    :param enabled: bool - Master toggle. When false, the splitter is bypassed
        and classification follows the single-intent path.
    :param max_sub_intents: int - Hard cap on the number of sub-intents that
        survive the weighted ranking pass when raw count is below
        ``collapse_threshold``. Default 5 is the chip-strip's clean upper bound.
    :param collapse_threshold: int - Raw-candidate count at which the strip
        switches from "keep top max_sub_intents" to "keep top
        ``top_k_after_collapse`` + emit an overflow chip for the rest".
        Default: when the splitter returns 6 or more candidates keep the top 3.
        Must be > ``top_k_after_collapse`` and >
        ``max_sub_intents``-friendly (caller invariants enforced in
        ``__post_init__``).
    :param top_k_after_collapse: int - When raw count >= ``collapse_threshold``
        this many sub-intents survive (the rest go to the overflow chip).
        Plan default 3. Must be in [1, max_sub_intents].
    :param max_split_candidates: int - Maximum raw sub-queries the splitter is
        allowed to emit before ranking; protects against pathological inputs
        ("a, b, c, d, e, f, g, h, ...") consuming unbounded LLM cost.
    :param min_sub_query_chars: int - Lower bound on a sub-query's character
        length; shorter fragments are dropped as splitter noise.
    :param weight_confidence: float - Ranking weight on calibrated classification
        confidence in the 5-cap formula.
    :param weight_expected_results: float - Weight on log-normalized expected
        result count from the structured pre-screen.
    :param weight_specificity: float - Weight on entity-richness specificity
        signal (number of grounded entities / max observed across sub-intents).
    :param expected_results_norm_cap: int - Upper bound used to normalize
        expected_results before weighting (counts above this all map to 1.0;
        prevents one giant slice from dominating).
    :param cross_intent_bonus: float - Multiplier applied to RRF-fused score
        when an item appears in two or more sub-intent candidate lists
        ("appears in multiple sub-intent results").
    :param drop_zero_expected: bool - When true, sub-intents whose pre-screen
        count == 0 are dropped before retrieval fan-out.
    :param drop_zero_expected_skip_unreliable_prescreen: bool - When true, skip
        zero-expected drops when the structured retriever reports
        ``provides_inventory_estimate=False`` (hybrid NoOp structured always
        returns empty counts that are not real inventory).
    :param drop_zero_expected_exempt_query_types: List[str] - Slice query_types
        that never drop on zero structured pre-screen (ANN/hybrid inventory is
        not measured by structured count).
    :param all_slices_dropped_fallback_to_single: bool - When every slice is
        dropped, run single-intent gather+fuse on the parent intent instead of
        returning an empty pool (which would only feed explore-rail fill).
    :param merge_strategy: str - Cross-slice merge rule. ``"rrf"`` accumulates
        ``sum_over_slices(1 / (rrf_k + rank_in_slice))``. ``"max_score"`` takes
        the maximum per-slice fused_score and applies ``cross_intent_bonus``
        once per multi-slice item.
    :param merge_rrf_k: int - RRF k constant when ``merge_strategy="rrf"``.
        Must be >= 1.
    """
    enabled: bool
    max_sub_intents: int
    max_split_candidates: int
    min_sub_query_chars: int
    weight_confidence: float
    weight_expected_results: float
    weight_specificity: float
    expected_results_norm_cap: int
    cross_intent_bonus: float
    drop_zero_expected: bool
    sub_intent_failure_policy: str
    drop_zero_expected_requires_hard_filters: bool = True
    drop_zero_expected_skip_unreliable_prescreen: bool = True
    drop_zero_expected_exempt_query_types: List[str] = field(default_factory=list)
    all_slices_dropped_fallback_to_single: bool = True
    merge_strategy: str = "rrf"
    merge_rrf_k: int = 60
    collapse_threshold: int = 6
    top_k_after_collapse: int = 3
    # Max sub-intent slices whose retrieval fan-out (vector+structured+sql each)
    # runs concurrently in the step-3 gather. Each slice is a full backend
    # fan-out; without a bound, a 3-comparator query ("like slack stripe zoom")
    # fires 3x the backend load at once, all contending on the same Qdrant HNSW
    # and ClickHouse connection pool, inflating wall-clock past the search SLA.
    # A semaphore caps in-flight slices; surviving slices beyond the cap queue.
    max_concurrent_slices: int = 2
    # Entity names for which only the first occurrence across sub-intents is kept.
    # Multi-intent splitting can produce the same entity from both a correct sub-query
    # and a noise fragment (e.g. similar_to=['openai'] and similar_to='open').
    # For singleton slots, first-wins prevents fragment noise from leaking in.
    singleton_merge_slots: List[str] = field(default_factory=list)
    # Post-merge entity deconfliction rules: [[flag_slot, presence_a, presence_b], ...].
    # After multi-intent slices are merged, removes flag_slot from any slice where
    # flag_slot is present AND at least one of presence_a/presence_b is also present.
    # Use empty string ("") for presence_b to check only presence_a.
    post_merge_deconflict_rules: List[List[str]] = field(default_factory=list)
    # Regex patterns that suppress splitting entirely when any matches the query.
    # Useful for "no X or Y" (combined constraint, not two intents) and other
    # constructions that are unambiguously single-intent despite containing a connective.
    no_split_patterns: List[str] = field(default_factory=list)
    # Slot names whose values are union-merged (list append + dedup) across all
    # sub-intent slices after splitting and classification, instead of first-wins.
    # Use for list-valued slots like tld, similar_to, auction_type where each
    # sub-intent may contribute distinct values that together form the full set.
    list_merge_slots: List[str] = field(default_factory=list)
    # Value-level deconfliction for keyword_contains. Each rule is
    # [keyword_value, guard_slot_a, guard_slot_b]: removes keyword_value from
    # keyword_contains when guard_slot_a or guard_slot_b is present. Prevents
    # metric brand names (e.g. 'majestic') from bleeding into keyword filter.
    keyword_contains_value_deconflict: List[List[str]] = field(default_factory=list)
    # Drop spurious low-confidence, entity-less sub-intent slices from false splits
    # (e.g. a conversational preamble "i am from delhi ..."). Safety net behind
    # no_split_patterns. Never empties the slice list — keeps the best slice if all
    # would be pruned.
    prune_noise_slices: bool = True
    prune_noise_max_confidence: float = 0.5
    # When true, QueryIntent.sub_intent_filters is populated with each sub-query's
    # entities captured before singleton deconfliction, so callers can inspect
    # per-sub-intent price/filter constraints that were merged away.
    preserve_sub_intent_filters: bool = False
    # How to resolve conflicting scalar slots (e.g. price_max) across sub-intents.
    # "first_seen": keep the first sub-intent's value (legacy behavior).
    # "max_permissive": for *_max slots take max(), for *_min slots take min(),
    # so the merged constraint covers all sub-intents (e.g. price_max=$20 covers
    # both ".net under $10" and ".io under $20" sub-intents).
    numeric_singleton_merge_strategy: str = "first_seen"
    # Slots copied from any slice onto every peer-bearing slice after merge.
    # Empty list disables broadcast. peer_slots empty means broadcast onto all slices.
    cross_slice_broadcast_slots: List[str] = field(default_factory=list)
    cross_slice_broadcast_peer_slots: List[str] = field(default_factory=list)
    # At per-slice retrieve: copy these hard-entity slots from sibling slices when
    # the current slice lacks them (e.g. price on "under $50" lands in one chunk).
    cross_slice_retrieve_propagate_slots: List[str] = field(default_factory=list)
    # When true, blend parent listing-concept encode (or parent normalized residual)
    # into each slice encode so fragment sub-queries keep topical ANN signal.
    slice_encode_blend_parent_concept: bool = True
    # Expand single-fragment queries into one retrieval leg per L0 keyword term.
    split_on_l0_keywords: bool = False
    split_on_l0_keywords_min_terms: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("multi_intent.enabled must be bool")
        if self.max_sub_intents < 1:
            raise ConfigurationError("multi_intent.max_sub_intents must be >= 1")
        if self.max_split_candidates < self.max_sub_intents:
            raise ConfigurationError("multi_intent.max_split_candidates must be >= max_sub_intents")
        if self.min_sub_query_chars < 1:
            raise ConfigurationError("multi_intent.min_sub_query_chars must be >= 1")
        if self.top_k_after_collapse < 1:
            raise ConfigurationError("multi_intent.top_k_after_collapse must be >= 1")
        if self.top_k_after_collapse > self.max_sub_intents:
            raise ConfigurationError("multi_intent.top_k_after_collapse must be <= max_sub_intents")
        if self.collapse_threshold <= self.max_sub_intents:
            raise ConfigurationError("multi_intent.collapse_threshold must be > max_sub_intents (collapse only fires above the normal cap)")
        for name, val in (('weight_confidence', self.weight_confidence), ('weight_expected_results', self.weight_expected_results), ('weight_specificity', self.weight_specificity)):
            if not 0.0 <= float(val) <= 1.0:
                raise ConfigurationError(f"multi_intent.{name} must be in [0,1]")
        weight_sum = float(self.weight_confidence) + float(self.weight_expected_results) + float(self.weight_specificity)
        if abs(weight_sum - 1.0) > 1e-6:
            raise ConfigurationError(f"multi_intent weights must sum to 1.0 (got {weight_sum:.6f})")
        if self.expected_results_norm_cap < 1:
            raise ConfigurationError("multi_intent.expected_results_norm_cap must be >= 1")
        if not 1.0 <= float(self.cross_intent_bonus) <= 5.0:
            raise ConfigurationError("multi_intent.cross_intent_bonus must be in [1.0, 5.0]")
        if not isinstance(self.drop_zero_expected, bool):
            raise ConfigurationError("multi_intent.drop_zero_expected must be bool")
        if not isinstance(self.drop_zero_expected_requires_hard_filters, bool):
            raise ConfigurationError("multi_intent.drop_zero_expected_requires_hard_filters must be bool")
        if not isinstance(self.drop_zero_expected_skip_unreliable_prescreen, bool):
            raise ConfigurationError("multi_intent.drop_zero_expected_skip_unreliable_prescreen must be bool")
        if not isinstance(self.drop_zero_expected_exempt_query_types, list) or not all(
            isinstance(qt, str) and qt for qt in self.drop_zero_expected_exempt_query_types
        ):
            raise ConfigurationError(
                "multi_intent.drop_zero_expected_exempt_query_types must be a list of non-empty strings"
            )
        _qt_allowed = frozenset({'hybrid', 'guidance', 'explore', 'analytics'})
        for qt in self.drop_zero_expected_exempt_query_types:
            if qt not in _qt_allowed:
                raise ConfigurationError(
                    "multi_intent.drop_zero_expected_exempt_query_types entries must be one of "
                    f"{sorted(_qt_allowed)}; got {qt!r}"
                )
        if not isinstance(self.all_slices_dropped_fallback_to_single, bool):
            raise ConfigurationError("multi_intent.all_slices_dropped_fallback_to_single must be bool")
        if self.merge_strategy not in ("rrf", "max_score"):
            raise ConfigurationError( f"multi_intent.merge_strategy must be one of " f"['rrf', 'max_score'] (got {self.merge_strategy!r})" )
        if int(self.merge_rrf_k) < 1:
            raise ConfigurationError("multi_intent.merge_rrf_k must be >= 1")
        if int(self.max_concurrent_slices) < 1:
            raise ConfigurationError("multi_intent.max_concurrent_slices must be >= 1")
        if self.sub_intent_failure_policy not in ('fail_soft', 'fail_closed'):
            raise ConfigurationError("multi_intent.sub_intent_failure_policy must be 'fail_soft' or 'fail_closed'")
        if not isinstance(self.post_merge_deconflict_rules, list):
            raise ConfigurationError("multi_intent.post_merge_deconflict_rules must be a list")
        for _rule in self.post_merge_deconflict_rules:
            if not isinstance(_rule, list) or len(_rule) != 3 or not all(isinstance(s, str) for s in _rule):
                raise ConfigurationError("multi_intent.post_merge_deconflict_rules entries must be [flag, presence_a, presence_b] string triples")
        if not isinstance(self.no_split_patterns, list) or not all(isinstance(p, str) and p for p in self.no_split_patterns):
            raise ConfigurationError("multi_intent.no_split_patterns must be a list of non-empty strings")
        if not isinstance(self.list_merge_slots, list) or not all(isinstance(s, str) and s for s in self.list_merge_slots):
            raise ConfigurationError("multi_intent.list_merge_slots must be a list of non-empty strings")
        if not isinstance(self.keyword_contains_value_deconflict, list):
            raise ConfigurationError("multi_intent.keyword_contains_value_deconflict must be a list")
        for _rule in self.keyword_contains_value_deconflict:
            if not isinstance(_rule, list) or len(_rule) != 3 or not all(isinstance(s, str) for s in _rule):
                raise ConfigurationError("multi_intent.keyword_contains_value_deconflict entries must be [value, guard_a, guard_b] string triples")
        if not isinstance(self.prune_noise_slices, bool):
            raise ConfigurationError("multi_intent.prune_noise_slices must be bool")
        if not 0.0 <= float(self.prune_noise_max_confidence) <= 1.0:
            raise ConfigurationError("multi_intent.prune_noise_max_confidence must be in [0,1]")
        if not isinstance(self.preserve_sub_intent_filters, bool):
            raise ConfigurationError("multi_intent.preserve_sub_intent_filters must be bool")
        if self.numeric_singleton_merge_strategy not in ('first_seen', 'max_permissive'):
            raise ConfigurationError(
                "multi_intent.numeric_singleton_merge_strategy must be 'first_seen' or 'max_permissive'"
            )
        if not isinstance(self.cross_slice_broadcast_slots, list) or not all(
            isinstance(s, str) and s for s in self.cross_slice_broadcast_slots
        ):
            raise ConfigurationError("multi_intent.cross_slice_broadcast_slots must be a list of non-empty strings")
        if not isinstance(self.cross_slice_broadcast_peer_slots, list) or not all(
            isinstance(s, str) and s for s in self.cross_slice_broadcast_peer_slots
        ):
            raise ConfigurationError(
                "multi_intent.cross_slice_broadcast_peer_slots must be a list of non-empty strings"
            )
        if not isinstance(self.cross_slice_retrieve_propagate_slots, list) or not all(
            isinstance(s, str) and s for s in self.cross_slice_retrieve_propagate_slots
        ):
            raise ConfigurationError(
                "multi_intent.cross_slice_retrieve_propagate_slots must be a list of non-empty strings"
            )
        if not isinstance(self.slice_encode_blend_parent_concept, bool):
            raise ConfigurationError("multi_intent.slice_encode_blend_parent_concept must be bool")
        if not isinstance(self.split_on_l0_keywords, bool):
            raise ConfigurationError("multi_intent.split_on_l0_keywords must be bool")
        if int(self.split_on_l0_keywords_min_terms) < 2:
            raise ConfigurationError("multi_intent.split_on_l0_keywords_min_terms must be >= 2")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'MultiIntentConfig':
        _require(
            d,
            [
                'enabled', 'max_sub_intents', 'max_split_candidates', 'min_sub_query_chars', 'weight_confidence',
                'weight_expected_results', 'weight_specificity', 'expected_results_norm_cap', 'cross_intent_bonus',
                'drop_zero_expected', 'sub_intent_failure_policy',
                'drop_zero_expected_skip_unreliable_prescreen',
                'drop_zero_expected_exempt_query_types',
                'all_slices_dropped_fallback_to_single',
                'cross_slice_retrieve_propagate_slots',
                'slice_encode_blend_parent_concept',
                'split_on_l0_keywords',
                'split_on_l0_keywords_min_terms',
            ],
            'multi_intent',
        )
        return cls(
            enabled=bool(d['enabled']),
            max_sub_intents=int(d['max_sub_intents']),
            max_split_candidates=int(d['max_split_candidates']),
            min_sub_query_chars=int(d['min_sub_query_chars']),
            weight_confidence=float(d['weight_confidence']),
            weight_expected_results=float(d['weight_expected_results']),
            weight_specificity=float(d['weight_specificity']),
            expected_results_norm_cap=int(d['expected_results_norm_cap']),
            cross_intent_bonus=float(d['cross_intent_bonus']),
            drop_zero_expected=bool(d['drop_zero_expected']),
            sub_intent_failure_policy=str(d['sub_intent_failure_policy']),
            drop_zero_expected_requires_hard_filters=bool(d.get('drop_zero_expected_requires_hard_filters', True)),
            drop_zero_expected_skip_unreliable_prescreen=bool(d['drop_zero_expected_skip_unreliable_prescreen']),
            drop_zero_expected_exempt_query_types=[
                str(qt).strip() for qt in d['drop_zero_expected_exempt_query_types']
            ],
            all_slices_dropped_fallback_to_single=bool(d['all_slices_dropped_fallback_to_single']),
            merge_strategy=str(d.get('merge_strategy', 'rrf')),
            merge_rrf_k=int(d.get('merge_rrf_k', 60)),
            collapse_threshold=int(d.get('collapse_threshold', 6)),
            top_k_after_collapse=int(d.get('top_k_after_collapse', 3)),
            max_concurrent_slices=int(d.get('max_concurrent_slices', 2)),
            singleton_merge_slots=[str(s) for s in d.get('singleton_merge_slots', [])],
            post_merge_deconflict_rules=[[str(s) for s in r] for r in d.get('post_merge_deconflict_rules', []) if isinstance(r, list)],
            no_split_patterns=[str(p) for p in d.get('no_split_patterns', []) if isinstance(p, str) and p],
            list_merge_slots=[str(s) for s in d.get('list_merge_slots', []) if isinstance(s, str) and s],
            keyword_contains_value_deconflict=[[str(s) for s in r] for r in d.get('keyword_contains_value_deconflict', []) if isinstance(r, list)],
            prune_noise_slices=bool(d.get('prune_noise_slices', True)),
            prune_noise_max_confidence=float(d.get('prune_noise_max_confidence', 0.5)),
            preserve_sub_intent_filters=bool(d.get('preserve_sub_intent_filters', False)),
            numeric_singleton_merge_strategy=str(d.get('numeric_singleton_merge_strategy', 'first_seen')),
            cross_slice_broadcast_slots=[
                str(s) for s in d.get('cross_slice_broadcast_slots', []) if isinstance(s, str) and s
            ],
            cross_slice_broadcast_peer_slots=[
                str(s) for s in d.get('cross_slice_broadcast_peer_slots', []) if isinstance(s, str) and s
            ],
            cross_slice_retrieve_propagate_slots=[
                str(s) for s in d['cross_slice_retrieve_propagate_slots'] if isinstance(s, str) and s
            ],
            slice_encode_blend_parent_concept=bool(d['slice_encode_blend_parent_concept']),
            split_on_l0_keywords=bool(d['split_on_l0_keywords']),
            split_on_l0_keywords_min_terms=int(d['split_on_l0_keywords_min_terms']),
        )


@dataclass
class IngestConfig:
    """listing + bid event ingestion config.

    The in-memory stub (``semantic_search.ingest.in_memory_consumer``) implements
    the same Protocols a future Kinesis consumer will satisfy. Until the live
    consumer is wired, this block governs the stub's behavior.

    :param enabled: bool - Master toggle for the ingest subsystem
    :param invalidate_caches_on_event: bool - When true, every consumed event bumps
        the snapshot version and any registered cache invalidation hook is called.
        This is the cache-invalidation cascade trigger.
    :param max_events_in_memory: int - Hard cap on retained events per stream
        (replay buffer for tests; protects against unbounded growth in dev)
    """
    enabled: bool
    invalidate_caches_on_event: bool
    max_events_in_memory: int

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("ingest.enabled must be bool")
        if not isinstance(self.invalidate_caches_on_event, bool):
            raise ConfigurationError("ingest.invalidate_caches_on_event must be bool")
        if self.max_events_in_memory < 1:
            raise ConfigurationError("ingest.max_events_in_memory must be >= 1")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'IngestConfig':
        _require(d, ['enabled', 'invalidate_caches_on_event', 'max_events_in_memory'], 'ingest')
        return cls( enabled=bool(d['enabled']), invalidate_caches_on_event=bool(d['invalidate_caches_on_event']), max_events_in_memory=int(d['max_events_in_memory']), )


@dataclass
class InventoryConfig:
    """Live inventory-grounded value resolution.

    Replaces hardcoded "cheap" / "expiring" thresholds with live percentiles
    computed off the structured-index price + remaining-time columns. Config
    priors (``cheap_price_max_prior``, ``expiring_seconds_max_prior``) are the
    fallback path when the index is empty or the resolver hasn't refreshed yet.

    :param enabled: bool - Master toggle. When false, callers fall back to priors.
    :param cheap_percentile: float - The percentile of live prices defining "cheap"
        ("cheap ->p25 of live prices"). In [0.01, 0.50].
    :param expiring_percentile: float - The percentile of remaining-time values
        defining "expiring" ("expiring ->p20 of remaining time").
        In [0.01, 0.50].
    :param refresh_interval_seconds: int - Minimum seconds between consecutive
        full re-scans of the structured index (cache for the resolver).
    :param min_sample_size: int - Minimum rows required before percentile is trusted;
        below this, the resolver returns the configured prior.
    :param cheap_price_max_prior: float - Fallback "cheap" threshold (USD) when the
        live percentile cannot be computed.
    :param expiring_seconds_max_prior: int - Fallback "expiring" remaining-time
        threshold (seconds) when the live percentile cannot be computed.
    """
    enabled: bool
    cheap_percentile: float
    expiring_percentile: float
    refresh_interval_seconds: int
    min_sample_size: int
    cheap_price_max_prior: float
    expiring_seconds_max_prior: int
    # Percentile of live prices defining the per-TLD "market" baseline used by the
    # price_below_market / cheaper_than_avg predicates. 0.5 = median ("average").
    market_percentile: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("inventory.enabled must be bool")
        for name, val in (('cheap_percentile', self.cheap_percentile), ('expiring_percentile', self.expiring_percentile)):
            if not 0.01 <= float(val) <= 0.50:
                raise ConfigurationError(f"inventory.{name} must be in [0.01, 0.50] (got {val})")
        if not 0.01 <= float(self.market_percentile) <= 0.99:
            raise ConfigurationError(f"inventory.market_percentile must be in [0.01, 0.99] (got {self.market_percentile})")
        if self.refresh_interval_seconds < 1:
            raise ConfigurationError("inventory.refresh_interval_seconds must be >= 1")
        if self.min_sample_size < 1:
            raise ConfigurationError("inventory.min_sample_size must be >= 1")
        if float(self.cheap_price_max_prior) <= 0.0:
            raise ConfigurationError("inventory.cheap_price_max_prior must be > 0")
        if int(self.expiring_seconds_max_prior) <= 0:
            raise ConfigurationError("inventory.expiring_seconds_max_prior must be > 0")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'InventoryConfig':
        _require(d, ['enabled', 'cheap_percentile', 'expiring_percentile', 'refresh_interval_seconds', 'min_sample_size', 'cheap_price_max_prior', 'expiring_seconds_max_prior'], 'inventory')
        return cls(
            enabled=bool(d['enabled']),
            cheap_percentile=float(d['cheap_percentile']),
            expiring_percentile=float(d['expiring_percentile']),
            refresh_interval_seconds=int(d['refresh_interval_seconds']),
            min_sample_size=int(d['min_sample_size']),
            cheap_price_max_prior=float(d['cheap_price_max_prior']),
            expiring_seconds_max_prior=int(d['expiring_seconds_max_prior']),
            market_percentile=float(d.get('market_percentile', 0.5)),
        )


@dataclass
class CalibrationProbeConfig:
    """Correctness-probe sub-config for calibrated confidence.

    The probe is the second leg of calibrated confidence: it predicts
    P(correct | distribution_shape) from the normalized entropy of the
    classifier's score distribution. Combined with the temperature-scaled
    raw probability via geometric mean so raw scores are never used directly
    for routing.

    Disabled by default: enabling requires per-tier producers that emit
    score distributions (today only L1 semantic router does). When disabled
    the registry installs identity probes and the combined-calibration
    operator collapses to pure temperature scaling — preserving today's
    behaviour exactly.

    :param enabled: bool - Master switch (False = identity probe per tier)
    :param alpha: float - Fixed slope of the logistic probe; > 0. Higher
        alpha makes the probe more responsive to entropy. Default 4.0
        gives a useful range across [0, 1] entropy with a single fit
        intercept.
    :param weight_raw: float - Weight on the temperature-scaled leg of the
        combined calibration in [0, 1]. 1.0 disables the probe contribution
        (combined ≡ T-scaled), 0.0 disables the raw contribution. Default
        0.6 anchors the decision on the calibrated raw score while letting
        the entropy probe pull calibrated confidence down on flat
        distributions.
    :param min_samples_per_tier: int - Minimum (entropy_normalized, correct)
        pairs needed before fitting; tiers below threshold use identity probe.
        Independent of CalibrationConfig.min_samples_per_tier (probe needs
        a tier-emitted distribution which not every tier has).
    :param min_beta: float - Lower bound for the intercept search.
    :param max_beta: float - Upper bound for the intercept search.
    :param tolerance: float - Convergence tolerance on beta.
    :param max_iterations: int - Hard cap on the bracket-shrink iterations.
    """
    enabled: bool
    alpha: float
    weight_raw: float
    min_samples_per_tier: int
    min_beta: float
    max_beta: float
    tolerance: float
    max_iterations: int

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("calibration.probe.enabled must be bool")
        if self.alpha <= 0.0:
            raise ConfigurationError("calibration.probe.alpha must be > 0")
        if not 0.0 <= float(self.weight_raw) <= 1.0:
            raise ConfigurationError("calibration.probe.weight_raw must be in [0,1]")
        if self.min_samples_per_tier < 5:
            raise ConfigurationError("calibration.probe.min_samples_per_tier must be >= 5 (1-param fit needs at least a handful of pairs)")
        if self.max_beta <= self.min_beta:
            raise ConfigurationError("calibration.probe.max_beta must be > min_beta")
        if self.tolerance <= 0.0:
            raise ConfigurationError("calibration.probe.tolerance must be > 0")
        if self.max_iterations < 5:
            raise ConfigurationError("calibration.probe.max_iterations must be >= 5")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CalibrationProbeConfig':
        _require(d, ['enabled', 'alpha', 'weight_raw', 'min_samples_per_tier', 'min_beta', 'max_beta', 'tolerance', 'max_iterations'], 'calibration.probe')
        return cls(
            enabled=bool(d['enabled']),
            alpha=float(d['alpha']),
            weight_raw=float(d['weight_raw']),
            min_samples_per_tier=int(d['min_samples_per_tier']),
            min_beta=float(d['min_beta']),
            max_beta=float(d['max_beta']),
            tolerance=float(d['tolerance']),
            max_iterations=int(d['max_iterations']),
        )


@dataclass
class CalibrationConfig:
    """Temperature-scaling calibration config.

    Confidence values emitted by L0 / L1 / L2 are raw classifier scores. The
    calibrator is a 1-parameter Platt scaling fit per tier on the golden seed
    dataset; calibrated confidence is what the L0 short-circuit gate reads.
    ``enabled=False`` makes the calibrator a no-op pass-through so disabling per-tier or all tiers degrades gracefully to the pre-calibration behaviour.

    :param enabled: bool - Master switch (False = identity calibrator)
    :param fit_on_load: bool - Fit calibrators against golden seeds at registry boot.
        When False the registry constructs identity calibrators (no fit cost) — useful
        for tests and for environments that load a pre-fit T from disk later.
    :param min_samples_per_tier: int - Minimum (raw_conf, correct) pairs needed
        per tier before a tier-specific T is fit. Tiers below the threshold use
        the identity calibrator (T=1.0).
    :param min_temperature: float - Lower bound for the search range (T must be > 0).
    :param max_temperature: float - Upper bound for the search range.
    :param tolerance: float - Convergence tolerance on T for the bounded
        scalar minimizer (golden-section search on log-T).
    :param max_iterations: int - Hard cap on minimizer iterations.
    :param tier_keys: List[str] - Stable tier identifiers stored alongside T;
        used by `CalibratorRegistry.get(tier)`. Defaults to the cascade tiers.
    :param hot_swap_min_samples: int - Minimum sample count before
        ``register_fit(source='traffic_hot_swap')`` is accepted by
        ``CalibratorRegistry`` (batch or offline callers only).
    :param boot_seeds_path: Optional[str] - Path to YAML with ``cases:`` list
        (``input_query``, ``expected_query_type`` per row). Required material for
        ``fit_on_load``; when ``None`` and ``fit_on_load`` is True the registry
        boots with zero seed cases (identity calibrators).
    """
    enabled: bool
    fit_on_load: bool
    min_samples_per_tier: int
    min_temperature: float
    max_temperature: float
    tolerance: float
    max_iterations: int
    tier_keys: List[str]
    hot_swap_min_samples: int
    probe: Optional[CalibrationProbeConfig] = None
    boot_seeds_path: Optional[str] = None
    persist_enabled: bool = False
    persist_path: Optional[str] = None
    cache_version: int = 1

    def __post_init__(self) -> None:
        if self.min_samples_per_tier < 5:
            raise ConfigurationError("calibration.min_samples_per_tier must be >= 5 (1-param Platt needs at least a handful of pairs)")
        if self.min_temperature <= 0:
            raise ConfigurationError("calibration.min_temperature must be > 0 (T parameterises sigmoid scaling)")
        if self.max_temperature <= self.min_temperature:
            raise ConfigurationError("calibration.max_temperature must be > min_temperature")
        if self.tolerance <= 0:
            raise ConfigurationError("calibration.tolerance must be > 0")
        if self.max_iterations < 5:
            raise ConfigurationError("calibration.max_iterations must be >= 5")
        if not isinstance(self.tier_keys, list) or not self.tier_keys:
            raise ConfigurationError("calibration.tier_keys must be a non-empty list")
        for tk in self.tier_keys:
            if not isinstance(tk, str) or not tk:
                raise ConfigurationError(f"calibration.tier_keys entry must be non-empty string, got {tk!r}")
        if self.hot_swap_min_samples < self.min_samples_per_tier:
            raise ConfigurationError("calibration.hot_swap_min_samples must be >= min_samples_per_tier")
        if self.probe is not None and not isinstance(self.probe, CalibrationProbeConfig):
            raise ConfigurationError("calibration.probe must be CalibrationProbeConfig or None")
        if self.boot_seeds_path is not None:
            if not isinstance(self.boot_seeds_path, str) or not str(self.boot_seeds_path).strip():
                raise ConfigurationError("calibration.boot_seeds_path must be a non-empty string when set")
        if self.persist_enabled and (not isinstance(self.persist_path, str) or not str(self.persist_path).strip()):
            raise ConfigurationError("calibration.persist_path must be a non-empty string when persist_enabled is true")
        if int(self.cache_version) < 1:
            raise ConfigurationError("calibration.cache_version must be >= 1")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CalibrationConfig':
        _require(d, ['enabled', 'fit_on_load', 'min_samples_per_tier', 'min_temperature', 'max_temperature', 'tolerance', 'max_iterations', 'tier_keys', 'hot_swap_min_samples'], 'calibration')
        probe_raw = d.get('probe')
        probe_cfg: Optional[CalibrationProbeConfig]
        if probe_raw is None:
            probe_cfg = None
        elif isinstance(probe_raw, dict):
            probe_cfg = CalibrationProbeConfig.from_dict(probe_raw)
        else:
            raise ConfigurationError(f"calibration.probe must be a dict or omitted; got {type(probe_raw).__name__}")
        boot_raw = d.get('boot_seeds_path')
        if boot_raw is None:
            boot_seeds_path: Optional[str] = None
        elif isinstance(boot_raw, str):
            boot_seeds_path = str(boot_raw).strip() or None
        else:
            raise ConfigurationError(f"calibration.boot_seeds_path must be a string or omitted; got {type(boot_raw).__name__}")
        return cls(
            enabled=bool(d['enabled']),
            fit_on_load=bool(d['fit_on_load']),
            min_samples_per_tier=int(d['min_samples_per_tier']),
            min_temperature=float(d['min_temperature']),
            max_temperature=float(d['max_temperature']),
            tolerance=float(d['tolerance']),
            max_iterations=int(d['max_iterations']),
            tier_keys=[str(t) for t in d['tier_keys']],
            hot_swap_min_samples=int(d['hot_swap_min_samples']),
            probe=probe_cfg,
            boot_seeds_path=boot_seeds_path,
            persist_enabled=bool(d.get('persist_enabled', False)),
            persist_path=(str(d['persist_path']).strip() or None) if d.get('persist_path') is not None else None,
            cache_version=int(d.get('cache_version', 1)),
        )


@dataclass
class LexicalModeratorConfig:
    """Lexical blocklist moderator config (egress guard)."""
    payload_fields: List[str]
    banned_terms: List[str]
    min_term_length: int
    max_terms: int
    stopwords: List[str]

    def __post_init__(self) -> None:
        if not isinstance(self.payload_fields, list) or not self.payload_fields:
            raise ConfigurationError("payload_fields must be a non-empty list")
        for fld in self.payload_fields:
            if not isinstance(fld, str) or not fld:
                raise ConfigurationError("payload_fields entries must be non-empty strings")
        if not isinstance(self.banned_terms, list) or not self.banned_terms:
            raise ConfigurationError("banned_terms must be a non-empty list")
        for term in self.banned_terms:
            if not isinstance(term, str) or not term:
                raise ConfigurationError("banned_terms entries must be non-empty strings")
            if term != term.lower():
                raise ConfigurationError("banned_terms entries must be lowercase")
        if int(self.min_term_length) < 1:
            raise ConfigurationError("min_term_length must be >= 1")
        if int(self.max_terms) < 1:
            raise ConfigurationError("max_terms must be >= 1")
        sw = {str(s).lower() for s in self.stopwords if isinstance(s, str) and s}
        for term in self.banned_terms:
            if term.lower() in sw:
                raise ConfigurationError( f"safety.egress_guard.moderator.lexical.banned_terms entry {term!r} is also in stopwords; remove it from one of the lists" )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'LexicalModeratorConfig':
        _require(d, ['payload_fields', 'banned_terms', 'min_term_length', 'max_terms', 'stopwords'], 'safety.egress_guard.moderator.lexical')
        return cls(
            payload_fields=list(d['payload_fields']),
            banned_terms=list(d['banned_terms']),
            min_term_length=int(d['min_term_length']),
            max_terms=int(d['max_terms']),
            stopwords=list(d['stopwords']),
        )


_MODERATOR_BACKENDS = frozenset({'noop', 'lexical_blocklist'})
_MODERATOR_POLICIES = frozenset({'drop', 'mask'})


@dataclass
class ModeratorConfig:
    """Top-level moderator config.

    :param enabled: bool - Master switch. False = the EgressGuard skips the
        moderation check entirely (NoOpModerator wired internally so the
        per-item decision shape stays uniform).
    :param backend: str - One of ``{'lexical_blocklist', 'noop'}``. The
        ``noop`` backend is the typed-disabled path; ``lexical_blocklist`` is
        the stdlib-only default that ships in this PR. Future model-backed
        backends register here.
    :param policy: str - One of ``{'drop', 'mask'}``. ``drop`` removes the
        flagged item from the result set entirely (the safer default).
        ``mask`` keeps the item but redacts the configured payload fields
        with ``[REDACTED]``; intended for low-confidence categories where
        dropping the listing would cause user friction.
    :param lexical: Optional[LexicalModeratorConfig] - Required iff
        ``backend='lexical_blocklist'``. Ignored for other backends.
    """
    enabled: bool
    backend: str
    policy: str
    lexical: Optional[LexicalModeratorConfig] = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("safety.egress_guard.moderator.enabled must be a bool")
        if self.backend not in _MODERATOR_BACKENDS:
            raise ConfigurationError( f"safety.egress_guard.moderator.backend must be one of {sorted(_MODERATOR_BACKENDS)}; got {self.backend!r}" )
        if self.policy not in _MODERATOR_POLICIES:
            raise ConfigurationError( f"safety.egress_guard.moderator.policy must be one of {sorted(_MODERATOR_POLICIES)}; got {self.policy!r}" )
        if self.enabled and self.backend == 'lexical_blocklist' and self.lexical is None:
            raise ConfigurationError( "safety.egress_guard.moderator.lexical is required when enabled=true and backend='lexical_blocklist'" )
        if self.lexical is not None and not isinstance(self.lexical, LexicalModeratorConfig):
            raise ConfigurationError("safety.egress_guard.moderator.lexical must be a LexicalModeratorConfig")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ModeratorConfig':
        _require(d, ['enabled', 'backend', 'policy'], 'safety.egress_guard.moderator')
        lex_dict = d.get('lexical')
        lex_cfg: Optional[LexicalModeratorConfig] = None
        if lex_dict is not None:
            lex_cfg = LexicalModeratorConfig.from_dict(lex_dict)
        return cls( enabled=bool(d['enabled']), backend=str(d['backend']), policy=str(d['policy']), lexical=lex_cfg, )


@dataclass
class PIIScrubConfig:
    """PII-scrub config.

    The PII *patterns* themselves live on ``safety.ingress_sanitizer.pii_patterns``
    (single source of truth — the LayerZeroSanitizer's regex set is reused so
    a policy change applies to both ingress and egress). This config selects
    which payload fields the gate walks on egress.

    :param enabled: bool - Master switch. False = the EgressGuard skips PII
        scrubbing entirely (no payload mutation).
    :param payload_fields: List[str] - Ordered list of ``RankedItem.payload``
        keys whose string values are scrubbed. Each field is walked
        recursively for ``str``/``list[str]``/``tuple[str]``/``dict`` values.
        Non-string scalars are left untouched. An empty list with
        enabled=true is a configuration error (defense against silent
        no-op on a misconfigured deployment).
    :param max_field_chars: int - Hard cap on per-field length (>= 16) — a
        defense against pathological payloads that would slow the regex pass.
        Fields longer than this are still scrubbed but truncated to this
        length before regex apply (oversize fields are flagged in the
        per-item ``reasons`` list as ``'oversize_field_truncated'``).
    """
    enabled: bool
    payload_fields: List[str]
    max_field_chars: int

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("safety.egress_guard.pii_scrub.enabled must be a bool")
        if not isinstance(self.payload_fields, list):
            raise ConfigurationError("safety.egress_guard.pii_scrub.payload_fields must be a list")
        if self.enabled and not self.payload_fields:
            raise ConfigurationError( "safety.egress_guard.pii_scrub.payload_fields must be non-empty when enabled=true" )
        for fld in self.payload_fields:
            if not isinstance(fld, str) or not fld:
                raise ConfigurationError( "safety.egress_guard.pii_scrub.payload_fields entries must be non-empty strings" )
        if int(self.max_field_chars) < 16:
            raise ConfigurationError("safety.egress_guard.pii_scrub.max_field_chars must be >= 16")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'PIIScrubConfig':
        _require(d, ['enabled', 'payload_fields', 'max_field_chars'], 'safety.egress_guard.pii_scrub')
        return cls( enabled=bool(d['enabled']), payload_fields=list(d['payload_fields']), max_field_chars=int(d['max_field_chars']), )


@dataclass
class GroundingCheckConfig:
    """Grounding-check config.

    Verifies that LLM-generated explanation strings on ``RankedItem.payload``
    only cite ``item_id``s actually present in the surrounding
    ``RankedResults.items`` set. Mismatched citations are masked at the span
    (the listing is preserved — only the unsafe explanation is scrubbed).

    :param enabled: bool - Master switch. False = the EgressGuard skips the
        grounding check.
    :param explanation_field: str - Payload key carrying the LLM-generated
        explanation text (e.g. ``'explanation'``, ``'why_this_match'``).
        Items lacking this field are passed through (the check only applies
        when an explanation is present — there is nothing to ground otherwise).
    :param item_id_pattern: str - Regex fragment that captures item id citations
        inside the explanation. Default style: ``\\bitem[_-]?([A-Za-z0-9_-]+)\\b``
        — ``item_42``, ``item-42``, ``item42`` all match. The first capture
        group is treated as the cited id; ids whose normalized form is not
        in the result set are span-masked.
    :param mask_token: str - Replacement string written in place of an
        ungrounded citation span (default: ``'[UNVERIFIED]'``).
    """
    enabled: bool
    explanation_field: str
    item_id_pattern: str
    mask_token: str

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("safety.egress_guard.grounding_check.enabled must be a bool")
        if not isinstance(self.explanation_field, str) or not self.explanation_field:
            raise ConfigurationError("safety.egress_guard.grounding_check.explanation_field must be a non-empty string")
        if not isinstance(self.item_id_pattern, str) or not self.item_id_pattern:
            raise ConfigurationError("safety.egress_guard.grounding_check.item_id_pattern must be a non-empty string")
        # Compile-check the regex at construction so a malformed pattern fails loudly at boot.
        try:
            compiled = re.compile(self.item_id_pattern)
        except re.error as e:
            raise ConfigurationError( f"safety.egress_guard.grounding_check.item_id_pattern is not a valid regex: {e}" ) from e
        if compiled.groups < 1:
            raise ConfigurationError( "safety.egress_guard.grounding_check.item_id_pattern must contain at least one capture group " "(the captured group is the cited item id)" )
        if not isinstance(self.mask_token, str) or not self.mask_token:
            raise ConfigurationError("safety.egress_guard.grounding_check.mask_token must be a non-empty string")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'GroundingCheckConfig':
        _require(d, ['enabled', 'explanation_field', 'item_id_pattern', 'mask_token'], 'safety.egress_guard.grounding_check')
        return cls( enabled=bool(d['enabled']), explanation_field=str(d['explanation_field']), item_id_pattern=str(d['item_id_pattern']), mask_token=str(d['mask_token']), )


@dataclass
class EgressGuardConfig:
    """Top-level output-side guardrails config.

    Composes the three independent checks into a single envelope so the
    orchestrator wires one handle. The gate is fail-closed on construction
    (a malformed sub-config refuses to boot) and fail-soft per item at
    runtime (an exception scrubbing one item drops that item rather than
    leaking; an exception in the moderator is handled the same way).

    :param enabled: bool - Master switch for the entire gate. False =
        ``NoOpEgressGuard`` is wired and the orchestrator's egress is a
        zero-cost passthrough.
    :param pii_scrub: PIIScrubConfig - PII mask sub-config.
    :param moderator: ModeratorConfig - Moderation sub-config.
    :param grounding_check: GroundingCheckConfig - Grounding-check sub-config.
    """
    enabled: bool
    pii_scrub: PIIScrubConfig
    moderator: ModeratorConfig
    grounding_check: GroundingCheckConfig

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("safety.egress_guard.enabled must be a bool")
        if not isinstance(self.pii_scrub, PIIScrubConfig):
            raise ConfigurationError("safety.egress_guard.pii_scrub must be a PIIScrubConfig")
        if not isinstance(self.moderator, ModeratorConfig):
            raise ConfigurationError("safety.egress_guard.moderator must be a ModeratorConfig")
        if not isinstance(self.grounding_check, GroundingCheckConfig):
            raise ConfigurationError("safety.egress_guard.grounding_check must be a GroundingCheckConfig")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'EgressGuardConfig':
        _require(d, ['enabled', 'pii_scrub', 'moderator', 'grounding_check'], 'safety.egress_guard')
        return cls(
            enabled=bool(d['enabled']),
            pii_scrub=PIIScrubConfig.from_dict(d['pii_scrub']),
            moderator=ModeratorConfig.from_dict(d['moderator']),
            grounding_check=GroundingCheckConfig.from_dict(d['grounding_check']),
        )


@dataclass
class FleetCostBudgetConfig:
    """Cross-request (fleet) LLM USD caps — hour and/or day UTC windows.

    Complements per-query ``CostBudgetConfig.max_cost_usd_per_query``.
    When ``enabled=true``, at least one of ``max_cost_usd_per_hour`` /
    ``max_cost_usd_per_day`` must be > 0.

    :param enabled: bool - Master switch for fleet accumulation + admit.
    :param max_cost_usd_per_hour: Optional[float] - Hour window cap (UTC);
        ``None`` / ``<=0`` disables the hour window.
    :param max_cost_usd_per_day: Optional[float] - Day window cap (UTC);
        ``None`` / ``<=0`` disables the day window.
    :param backend: str - ``memory`` (per-pod) or ``redis`` (shared).
    :param redis_url_env_var: str - Env var holding Redis URL when backend=redis.
    :param key_prefix: str - Redis key prefix for hour/day counters.
    :param socket_timeout_seconds: float - Redis socket timeout.
    """
    enabled: bool
    max_cost_usd_per_hour: Optional[float] = None
    max_cost_usd_per_day: Optional[float] = None
    backend: str = 'memory'
    redis_url_env_var: str = 'SEMANTIC_SEARCH_REDIS_URL'
    key_prefix: str = 'ss:fleet_llm_cost:'
    socket_timeout_seconds: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("cost_budget.fleet.enabled must be a bool")
        if not isinstance(self.backend, str) or self.backend not in ('memory', 'redis'):
            raise ConfigurationError("cost_budget.fleet.backend must be 'memory' or 'redis'")
        if not isinstance(self.redis_url_env_var, str) or not self.redis_url_env_var.strip():
            raise ConfigurationError("cost_budget.fleet.redis_url_env_var must be a non-empty str")
        if not isinstance(self.key_prefix, str) or not self.key_prefix:
            raise ConfigurationError("cost_budget.fleet.key_prefix must be a non-empty str")
        if not isinstance(self.socket_timeout_seconds, (int, float)) or float(self.socket_timeout_seconds) <= 0.0:
            raise ConfigurationError("cost_budget.fleet.socket_timeout_seconds must be > 0")
        hour = None if self.max_cost_usd_per_hour is None else float(self.max_cost_usd_per_hour)
        day = None if self.max_cost_usd_per_day is None else float(self.max_cost_usd_per_day)
        if hour is not None and hour <= 0.0:
            hour = None
        if day is not None and day <= 0.0:
            day = None
        object.__setattr__(self, 'max_cost_usd_per_hour', hour)
        object.__setattr__(self, 'max_cost_usd_per_day', day)
        if self.enabled and hour is None and day is None:
            raise ConfigurationError(
                "cost_budget.fleet.enabled=true requires max_cost_usd_per_hour > 0 "
                "and/or max_cost_usd_per_day > 0"
            )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FleetCostBudgetConfig':
        _require(d, ['enabled'], 'cost_budget.fleet')
        hour_raw = d.get('max_cost_usd_per_hour')
        day_raw = d.get('max_cost_usd_per_day')
        return cls(
            enabled=bool(d['enabled']),
            max_cost_usd_per_hour=(float(hour_raw) if hour_raw is not None else None),
            max_cost_usd_per_day=(float(day_raw) if day_raw is not None else None),
            backend=str(d.get('backend', 'memory')),
            redis_url_env_var=str(d.get('redis_url_env_var', 'SEMANTIC_SEARCH_REDIS_URL')),
            key_prefix=str(d.get('key_prefix', 'ss:fleet_llm_cost:')),
            socket_timeout_seconds=float(d.get('socket_timeout_seconds', 1.0)),
        )


@dataclass
class CostBudgetConfig:
    """Per-query + optional fleet LLM cost-budget config.

    Wires the ``QueryCostBudget`` factory and optional ``FleetCostBudget``
    into ``SearchOrchestrator``. Additive — the root
    ``AgentSearchConfig.cost_budget`` field is ``Optional``.

    :param enabled: bool - When False the orchestrator wires no enforcing
        query factory (tracking-only NoOp still binds for observability).
    :param max_cost_usd_per_query: float - Hard cap on cumulative LLM
        USD spent during one search()/analytics() call. Must be > 0.0
        when ``enabled=True``.
    :param fail_soft_on_breach: bool - Legacy flag. Search/QI breaches are
        mapped to ``LLMError`` so the cascade degrades to regex L0 + L1
        (queries are never empty-rejected for spend). Analytics may still
        return ``failure_mode='cost_budget_exceeded'``.
    :param fleet: Optional[FleetCostBudgetConfig] - Cross-request hour/day
        caps. Absent / ``enabled=false`` means no fleet enforcement.
    """
    enabled: bool
    max_cost_usd_per_query: float
    fail_soft_on_breach: bool = True
    fleet: Optional[FleetCostBudgetConfig] = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("cost_budget.enabled must be a bool")
        if not isinstance(self.fail_soft_on_breach, bool):
            raise ConfigurationError("cost_budget.fail_soft_on_breach must be a bool")
        if not isinstance(self.max_cost_usd_per_query, (int, float)):
            raise ConfigurationError("cost_budget.max_cost_usd_per_query must be numeric")
        if self.enabled and float(self.max_cost_usd_per_query) <= 0.0:
            raise ConfigurationError(
                "cost_budget.max_cost_usd_per_query must be > 0.0 when enabled=true; "
                "set enabled=false to disable enforcement"
            )
        if self.fleet is not None and not isinstance(self.fleet, FleetCostBudgetConfig):
            raise ConfigurationError("cost_budget.fleet must be a FleetCostBudgetConfig when set")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CostBudgetConfig':
        _require(d, ['enabled', 'max_cost_usd_per_query'], 'cost_budget')
        fleet_block = d.get('fleet')
        fleet = FleetCostBudgetConfig.from_dict(fleet_block) if isinstance(fleet_block, dict) else None
        return cls(
            enabled=bool(d['enabled']),
            max_cost_usd_per_query=float(d['max_cost_usd_per_query']),
            fail_soft_on_breach=bool(d.get('fail_soft_on_breach', True)),
            fleet=fleet,
        )


@dataclass
class SafetyConfig:
    """Top-level safety config.

    Carries LLM ingress sanitizer policy plus the egress guard. Both live under
    ``safety`` so input-side and output-side policy stay in one YAML subtree.

    :param ingress_sanitizer: SanitizerConfig - Layer-0 text gate (LLM ingress + retrieved content paths).
    :param egress_guard: EgressGuardConfig - Output-side guardrails sub-config.
    """
    ingress_sanitizer: SanitizerConfig
    egress_guard: EgressGuardConfig

    def __post_init__(self) -> None:
        if not isinstance(self.ingress_sanitizer, SanitizerConfig):
            raise ConfigurationError("safety.ingress_sanitizer must be a SanitizerConfig")
        if not isinstance(self.egress_guard, EgressGuardConfig):
            raise ConfigurationError("safety.egress_guard must be an EgressGuardConfig")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SafetyConfig':
        _require(d, ['ingress_sanitizer', 'egress_guard'], 'safety')
        return cls( ingress_sanitizer=SanitizerConfig.from_dict(d['ingress_sanitizer']), egress_guard=EgressGuardConfig.from_dict(d['egress_guard']), )


@dataclass
class ReasoningTraceConfig:
    """global toggle + bound for the reasoning-trace explainability surface.

    Drives the ``reasoning_trace`` field on ``QueryIntent`` and ``RankedItem``
    (see ``contracts.ReasoningTrace``). Default-disabled so existing
    deployments pay zero overhead until they opt in.

    When ``enabled=False`` (the default), the orchestrator never constructs
    a ``ReasoningTrace`` and the contract fields stay ``None`` end-to-end.
    When ``enabled=True``, the orchestrator constructs one trace per
    ``QueryIntent`` and (in the multi-intent fan-out path) one trace per
    ``RankedItem`` it produces; both are bounded by ``max_steps_per_trace``.

    :param enabled: bool - Master switch. Default-disabled so existing
        deployments pay no cost.
    :param max_steps_per_trace: int - Hard ceiling on the number of steps
        a single trace may retain (>=1; the underlying ``ReasoningTrace``
        rolls off the oldest real step and stamps a head sentinel beyond
        this point). Modest defaults (e.g. 64) cover an all-stages-enabled
        pipeline with room for sub-intent fan-out.
    """
    enabled: bool
    max_steps_per_trace: int

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("reasoning_trace.enabled must be a bool")
        if not isinstance(self.max_steps_per_trace, int) or isinstance(self.max_steps_per_trace, bool):
            raise ConfigurationError("reasoning_trace.max_steps_per_trace must be an int")
        if self.max_steps_per_trace < 1:
            raise ConfigurationError("reasoning_trace.max_steps_per_trace must be >= 1")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ReasoningTraceConfig':
        _require(d, ['enabled', 'max_steps_per_trace'], 'reasoning_trace')
        return cls( enabled=bool(d['enabled']), max_steps_per_trace=int(d['max_steps_per_trace']), )


@dataclass
class BM25DocEncoderConfig:
    """Corpus-side BM25 sparse encoder config.

    Mirrors the query-side encoder's hashing scheme so query and document
    sparse vectors line up bucket-for-bucket inside Qdrant. ``vocab_size``
    MUST equal ``retrieval.qdrant.hybrid.bm25_query_encoder.vocab_size``;
    a mismatch is rejected at the top-level cross-field check.

    :param k1: float - BM25 TF saturation parameter. Must be > 0; typical
        ``1.2 - 2.0``.
    :param b: float - Length-normalisation strength in ``[0.0, 1.0]``.
        ``0.0`` disables length normalisation (recommended for short-text
        corpora like domain names where length variance is low).
    :param avg_doc_length: float - Average tokens per document across the
        corpus. Must be > 0. Used only when ``b > 0``; pass the observed
        average so the indexer does not have to recompute per document.
    """
    k1: float
    b: float
    avg_doc_length: float

    def __post_init__(self) -> None:
        if not isinstance(self.k1, (int, float)) or float(self.k1) <= 0.0:
            raise ConfigurationError("vectorization.bm25_doc_encoder.k1 must be a number > 0")
        if not isinstance(self.b, (int, float)) or not 0.0 <= float(self.b) <= 1.0:
            raise ConfigurationError("vectorization.bm25_doc_encoder.b must be a number in [0.0, 1.0]")
        if not isinstance(self.avg_doc_length, (int, float)) or float(self.avg_doc_length) <= 0.0:
            raise ConfigurationError("vectorization.bm25_doc_encoder.avg_doc_length must be a number > 0")
        self.k1 = float(self.k1)
        self.b = float(self.b)
        self.avg_doc_length = float(self.avg_doc_length)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'BM25DocEncoderConfig':
        _require(d, ['k1', 'b', 'avg_doc_length'], 'vectorization.bm25_doc_encoder')
        return cls( k1=float(d['k1']), b=float(d['b']), avg_doc_length=float(d['avg_doc_length']), )


@dataclass
class IndexerConfig:
    """Offline corpus indexer config.

    Controls the batch-size + idempotency knobs the offline indexer uses
    when upserting to Qdrant. Pure orchestration knobs — the encoders,
    Qdrant client, and segmenter come from elsewhere in the registry.

    :param batch_size: int - Documents per Qdrant ``upsert`` call. Must be
        ``>= 1``; typical ``128 - 1024`` depending on payload size and
        Qdrant ingest throughput. Smaller batches use less memory and
        recover faster from transient errors; larger batches amortise
        per-call overhead.
    :param wait_for_index: bool - When ``True``, every ``upsert`` waits
        for Qdrant to confirm the points are searchable before returning.
        Trades latency for write-after-read consistency; required when the
        caller (e.g. an integration test or a snapshot-bump cascade)
        immediately reads what it just wrote.
    :param idempotency_key_field: str - Payload field used to derive the
        Qdrant point id. The indexer hashes this field to an int64 so
        re-running the indexer over the same input produces the same id and
        Qdrant treats the upsert as an in-place replace, not a duplicate
        insert. Typical: ``'domain'`` or ``'listing_id'``. Must be a
        non-empty string; the field MUST be present on every payload at
        indexing time or the indexer raises ``ValidationError`` for that
        document and continues with the rest of the batch.
    :param upsert_timeout_seconds: int - Per-call gRPC timeout for Qdrant upsert
    :param shared_matryoshka_dense_encode: bool - When True and a Matryoshka
        cascade encoder is wired, shortlist + rerank dense vectors are sliced
        from one ``encode_batch_at_dims`` call instead of two separate encodes
    :param sparse_encode_concurrency: int - Max concurrent sparse-encode chunks
        (BM42/BM25) submitted to the thread pool during materialise
    :param sparse_embed_batch_size: int - Texts per BM42 ``embed`` call (and
        chunk width for the concurrent sparse encode path)
    """
    batch_size: int
    wait_for_index: bool
    idempotency_key_field: str
    hnsw_m: int
    hnsw_ef_construct: int
    quantization_quantile: float
    optimizer_indexing_threshold: int
    optimizer_memmap_threshold: int
    upsert_timeout_seconds: int
    shared_matryoshka_dense_encode: bool
    sparse_encode_concurrency: int
    sparse_embed_batch_size: int

    def __post_init__(self) -> None:
        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ConfigurationError("vectorization.indexer.batch_size must be int >= 1")
        if not isinstance(self.wait_for_index, bool):
            raise ConfigurationError("vectorization.indexer.wait_for_index must be a bool")
        if not isinstance(self.idempotency_key_field, str) or not self.idempotency_key_field.strip():
            raise ConfigurationError("vectorization.indexer.idempotency_key_field must be a non-empty string")
        if not isinstance(self.hnsw_m, int) or self.hnsw_m < 1:
            raise ConfigurationError("vectorization.indexer.hnsw_m must be int >= 1")
        if not isinstance(self.hnsw_ef_construct, int) or self.hnsw_ef_construct < 1:
            raise ConfigurationError("vectorization.indexer.hnsw_ef_construct must be int >= 1")
        if not 0.0 < float(self.quantization_quantile) <= 1.0:
            raise ConfigurationError("vectorization.indexer.quantization_quantile must be in (0, 1]")
        if not isinstance(self.optimizer_indexing_threshold, int) or self.optimizer_indexing_threshold < 0:
            raise ConfigurationError("vectorization.indexer.optimizer_indexing_threshold must be int >= 0")
        if not isinstance(self.optimizer_memmap_threshold, int) or self.optimizer_memmap_threshold < 0:
            raise ConfigurationError("vectorization.indexer.optimizer_memmap_threshold must be int >= 0")
        if not isinstance(self.upsert_timeout_seconds, int) or isinstance(self.upsert_timeout_seconds, bool) or self.upsert_timeout_seconds < 1:
            raise ConfigurationError("vectorization.indexer.upsert_timeout_seconds must be int >= 1")
        if not isinstance(self.shared_matryoshka_dense_encode, bool):
            raise ConfigurationError("vectorization.indexer.shared_matryoshka_dense_encode must be a bool")
        if (
            not isinstance(self.sparse_encode_concurrency, int)
            or isinstance(self.sparse_encode_concurrency, bool)
            or self.sparse_encode_concurrency < 1
        ):
            raise ConfigurationError("vectorization.indexer.sparse_encode_concurrency must be int >= 1")
        if (
            not isinstance(self.sparse_embed_batch_size, int)
            or isinstance(self.sparse_embed_batch_size, bool)
            or self.sparse_embed_batch_size < 1
        ):
            raise ConfigurationError("vectorization.indexer.sparse_embed_batch_size must be int >= 1")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'IndexerConfig':
        _require(
            d,
            [
                'batch_size',
                'wait_for_index',
                'idempotency_key_field',
                'hnsw_m',
                'hnsw_ef_construct',
                'quantization_quantile',
                'optimizer_indexing_threshold',
                'optimizer_memmap_threshold',
                'upsert_timeout_seconds',
                'shared_matryoshka_dense_encode',
                'sparse_encode_concurrency',
                'sparse_embed_batch_size',
            ],
            'vectorization.indexer',
        )
        return cls(
            batch_size=int(d['batch_size']),
            wait_for_index=bool(d['wait_for_index']),
            idempotency_key_field=str(d['idempotency_key_field']).strip(),
            hnsw_m=int(d['hnsw_m']),
            hnsw_ef_construct=int(d['hnsw_ef_construct']),
            quantization_quantile=float(d['quantization_quantile']),
            optimizer_indexing_threshold=int(d['optimizer_indexing_threshold']),
            optimizer_memmap_threshold=int(d['optimizer_memmap_threshold']),
            upsert_timeout_seconds=int(d['upsert_timeout_seconds']),
            shared_matryoshka_dense_encode=bool(d['shared_matryoshka_dense_encode']),
            sparse_encode_concurrency=int(d['sparse_encode_concurrency']),
            sparse_embed_batch_size=int(d['sparse_embed_batch_size']),
        )


@dataclass
class VectorRefreshConfig:
    """Snapshot-version-aware vector refresh driver config.

    The driver runs in the background and triggers a re-index whenever the
    process-wide ``SnapshotVersionRegistry`` advances. The cadence + size
    knobs are config-driven; the driver itself owns no business logic
    beyond version-bump detection and dispatch to the indexer.

    :param enabled: bool - Master switch. When ``False`` the driver is
        constructed but never starts a background task; ``run_once()``
        still works for manual triggers and tests.
    :param interval_seconds: float - How often the driver wakes up to
        compare the current snapshot version against the last-indexed
        version. Must be > 0; typical ``30 - 300``. Tighter intervals
        reduce stale-window length but cost CPU + Qdrant write throughput.
    :param max_consecutive_failures: int - After this many back-to-back
        cycle errors the loop interval doubles (back-off against thundering
        a broken downstream). Must be >= 1.
    :param priority_interval_seconds: float - Reduced sleep used after an
        auction-extension (priority) bump so the vector index catches up
        faster than the normal cadence. Must be > 0 and <= interval_seconds.
    """
    enabled: bool
    interval_seconds: float
    max_consecutive_failures: int
    priority_interval_seconds: float = 5.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("vectorization.refresh.enabled must be a bool")
        if not isinstance(self.interval_seconds, (int, float)) or float(self.interval_seconds) <= 0.0:
            raise ConfigurationError("vectorization.refresh.interval_seconds must be a number > 0")
        if not isinstance(self.max_consecutive_failures, int) or self.max_consecutive_failures < 1:
            raise ConfigurationError("vectorization.refresh.max_consecutive_failures must be int >= 1")
        self.interval_seconds = float(self.interval_seconds)
        if not isinstance(self.priority_interval_seconds, (int, float)) or float(self.priority_interval_seconds) <= 0.0:
            raise ConfigurationError("vectorization.refresh.priority_interval_seconds must be a number > 0")
        if float(self.priority_interval_seconds) > self.interval_seconds:
            raise ConfigurationError( "vectorization.refresh.priority_interval_seconds must be <= interval_seconds" )
        self.priority_interval_seconds = float(self.priority_interval_seconds)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'VectorRefreshConfig':
        _require(d, ['enabled', 'interval_seconds', 'max_consecutive_failures'], 'vectorization.refresh')
        return cls(
            enabled=bool(d['enabled']),
            interval_seconds=float(d['interval_seconds']),
            max_consecutive_failures=int(d['max_consecutive_failures']),
            priority_interval_seconds=float(d.get('priority_interval_seconds') or 5.0),
        )


@dataclass
class DeltaRefreshConfig:
    """Mutable-field delta refresh driver config — polls auction_audit_cln.

    Queries the real-time event stream table at each interval, deduplicates to
    one row per domain, and issues Qdrant set_payload calls for the mutable
    fields only. No re-encoding occurs.

    :param enabled: bool - Master switch. When False the driver is constructed
        but never starts.
    :param interval_seconds: float - Poll cadence in seconds. Must be > 0.
    :param max_consecutive_failures: int - Back-off ceiling; loop doubles its
        sleep after this many consecutive cycle errors. Must be >= 1.
    :param source_table: str - Athena table name (no database prefix).
    :param source_database: str - Athena database that owns source_table.
    :param lookback_minutes: int - How far back the first cycle queries on cold
        start. Must be >= 1.
    :param batch_size: int - LIMIT applied to the dedup CTE. Must be >= 1.
    :param timeout_seconds: float - Per-cycle Athena query wall-clock cap.
        Must be > 0.
    :param mutable_fields: List[str] - Payload keys written by set_payload.
        Only these keys are updated; all other payload fields are untouched.
    :param chunk_minutes: int - Athena time-window slice size. Must be >= 1.
    :param find_payload_aliases: Dict[str, List[str]] - Mutable key to FIND listing aliases
    :param find_bool_aliases: Dict[str, str] - Mutable 0/1 key to FIND bool listing key
    """
    enabled: bool
    interval_seconds: float
    max_consecutive_failures: int
    source_table: str
    source_database: str
    lookback_minutes: int
    batch_size: int
    timeout_seconds: float
    mutable_fields: List[str]
    chunk_minutes: int
    find_payload_aliases: Dict[str, List[str]]
    find_bool_aliases: Dict[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("vectorization.delta_refresh.enabled must be a bool")
        if not isinstance(self.interval_seconds, (int, float)) or float(self.interval_seconds) <= 0.0:
            raise ConfigurationError("vectorization.delta_refresh.interval_seconds must be a number > 0")
        self.interval_seconds = float(self.interval_seconds)
        if not isinstance(self.max_consecutive_failures, int) or self.max_consecutive_failures < 1:
            raise ConfigurationError("vectorization.delta_refresh.max_consecutive_failures must be int >= 1")
        if not isinstance(self.source_table, str) or not self.source_table.strip():
            raise ConfigurationError("vectorization.delta_refresh.source_table must be a non-empty string")
        if not isinstance(self.source_database, str) or not self.source_database.strip():
            raise ConfigurationError("vectorization.delta_refresh.source_database must be a non-empty string")
        if not isinstance(self.lookback_minutes, int) or self.lookback_minutes < 1:
            raise ConfigurationError("vectorization.delta_refresh.lookback_minutes must be int >= 1")
        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ConfigurationError("vectorization.delta_refresh.batch_size must be int >= 1")
        if not isinstance(self.timeout_seconds, (int, float)) or float(self.timeout_seconds) <= 0.0:
            raise ConfigurationError("vectorization.delta_refresh.timeout_seconds must be a number > 0")
        self.timeout_seconds = float(self.timeout_seconds)
        if not isinstance(self.mutable_fields, list) or not self.mutable_fields:
            raise ConfigurationError("vectorization.delta_refresh.mutable_fields must be a non-empty list")
        if not all(isinstance(f, str) and f.strip() for f in self.mutable_fields):
            raise ConfigurationError("vectorization.delta_refresh.mutable_fields entries must be non-empty strings")
        if not isinstance(self.chunk_minutes, int) or self.chunk_minutes < 1:
            raise ConfigurationError("vectorization.delta_refresh.chunk_minutes must be int >= 1")
        if not isinstance(self.find_payload_aliases, dict) or not self.find_payload_aliases:
            raise ConfigurationError("vectorization.delta_refresh.find_payload_aliases must be a non-empty dict")
        for src, targets in self.find_payload_aliases.items():
            if not isinstance(src, str) or not src.strip():
                raise ConfigurationError("vectorization.delta_refresh.find_payload_aliases keys must be non-empty strings")
            if not isinstance(targets, list) or not targets or not all(isinstance(t, str) and t.strip() for t in targets):
                raise ConfigurationError("vectorization.delta_refresh.find_payload_aliases values must be non-empty string lists")
        if not isinstance(self.find_bool_aliases, dict):
            raise ConfigurationError("vectorization.delta_refresh.find_bool_aliases must be a dict")
        for src, target in self.find_bool_aliases.items():
            if not isinstance(src, str) or not src.strip() or not isinstance(target, str) or not target.strip():
                raise ConfigurationError("vectorization.delta_refresh.find_bool_aliases entries must be non-empty strings")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DeltaRefreshConfig':
        _require(d, [
            'enabled', 'interval_seconds', 'max_consecutive_failures', 'source_table',
            'source_database', 'lookback_minutes', 'batch_size', 'timeout_seconds',
            'mutable_fields', 'chunk_minutes', 'find_payload_aliases', 'find_bool_aliases',
        ], 'vectorization.delta_refresh')
        aliases = {
            str(k).strip(): [str(t).strip() for t in v]
            for k, v in dict(d['find_payload_aliases']).items()
        }
        bool_aliases = {str(k).strip(): str(v).strip() for k, v in dict(d['find_bool_aliases']).items()}
        return cls(
            enabled=_coerce_bool(d['enabled']),
            interval_seconds=float(d['interval_seconds']),
            max_consecutive_failures=int(d['max_consecutive_failures']),
            source_table=str(d['source_table']).strip(),
            source_database=str(d['source_database']).strip(),
            lookback_minutes=int(d['lookback_minutes']),
            batch_size=int(d['batch_size']),
            timeout_seconds=float(d['timeout_seconds']),
            mutable_fields=[str(f).strip() for f in d['mutable_fields']],
            chunk_minutes=int(d['chunk_minutes']),
            find_payload_aliases=aliases,
            find_bool_aliases=bool_aliases,
        )


@dataclass
class EnrichmentRefreshConfig:
    """Seed-time enrichment refresh driver config — polls majestic/semrush/
    estibot/search_rollup Athena sources on their own cadence (independent of
    the real-time DeltaRefreshConfig) and patches the corresponding Qdrant
    payload fields via a domain_name filter. No ClickHouse write path —
    these fields have no live ClickHouse consumer today.

    :param enabled: bool - Master switch. When False the driver is
        constructed but never starts.
    :param interval_seconds: float - Poll cadence in seconds. Must be > 0.
    :param max_consecutive_failures: int - Back-off ceiling; loop doubles its
        sleep after this many consecutive cycle errors. Must be >= 1.
    :param lookback_minutes: int - How far back the first cycle queries on
        cold start. Must be >= 1.
    :param batch_size: int - LIMIT applied to the dedup CTE. Must be >= 1.
    :param timeout_seconds: float - Per-cycle Athena query wall-clock cap.
        Must be > 0.
    :param chunk_minutes: int - Athena time-window slice size. Must be >= 1.
    """
    enabled: bool
    interval_seconds: float
    max_consecutive_failures: int
    lookback_minutes: int
    batch_size: int
    timeout_seconds: float
    chunk_minutes: int

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("vectorization.enrichment_refresh.enabled must be a bool")
        if not isinstance(self.interval_seconds, (int, float)) or float(self.interval_seconds) <= 0.0:
            raise ConfigurationError("vectorization.enrichment_refresh.interval_seconds must be a number > 0")
        self.interval_seconds = float(self.interval_seconds)
        if not isinstance(self.max_consecutive_failures, int) or self.max_consecutive_failures < 1:
            raise ConfigurationError("vectorization.enrichment_refresh.max_consecutive_failures must be int >= 1")
        if not isinstance(self.lookback_minutes, int) or self.lookback_minutes < 1:
            raise ConfigurationError("vectorization.enrichment_refresh.lookback_minutes must be int >= 1")
        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ConfigurationError("vectorization.enrichment_refresh.batch_size must be int >= 1")
        if not isinstance(self.timeout_seconds, (int, float)) or float(self.timeout_seconds) <= 0.0:
            raise ConfigurationError("vectorization.enrichment_refresh.timeout_seconds must be a number > 0")
        self.timeout_seconds = float(self.timeout_seconds)
        if not isinstance(self.chunk_minutes, int) or self.chunk_minutes < 1:
            raise ConfigurationError("vectorization.enrichment_refresh.chunk_minutes must be int >= 1")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'EnrichmentRefreshConfig':
        _require(d, [
            'enabled', 'interval_seconds', 'max_consecutive_failures',
            'lookback_minutes', 'batch_size', 'timeout_seconds', 'chunk_minutes',
        ], 'vectorization.enrichment_refresh')
        return cls(
            enabled=_coerce_bool(d['enabled']),
            interval_seconds=float(d['interval_seconds']),
            max_consecutive_failures=int(d['max_consecutive_failures']),
            lookback_minutes=int(d['lookback_minutes']),
            batch_size=int(d['batch_size']),
            timeout_seconds=float(d['timeout_seconds']),
            chunk_minutes=int(d['chunk_minutes']),
        )


@dataclass
class AnalyticsBackfillConfig:
    """Config for the POST /data-build/analytics-backfill endpoint.

    Controls the Athena query that populates ClickHouse signals_platform_cln.auction_audit_cln
    with a longer historical window than the Qdrant seed (which uses active_only=true
    and a short lookback). Decoupled from the Qdrant seed so the two stores can be
    sized independently.

    :param lookback_days: int - Default rolling window in days for the Athena query.
        Range 1–730. 180 days covers two quarters; fits 22 GB ClickHouse budget
        (~10–16 GB compressed for types 16/20/38/39).
    :param max_records: int - Row cap per Athena query (hard ceiling).
        25 000 000 gives 25% headroom over ~20M records in the 180-day window.
    :param batch_size: int - ClickHouse INSERT batch size (rows per round-trip).
    :param ensure_schema_once: bool - When True, DDL ensure runs on the first
        ClickHouse write of a backfill job only; when False, every page
    :param insert_timeout_seconds: float - Per-batch INSERT timeout
    :param schema_timeout_seconds: float - Per-DDL statement timeout
    """

    lookback_days: int
    max_records: int
    batch_size: int
    ensure_schema_once: bool
    insert_timeout_seconds: float
    schema_timeout_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.lookback_days, int) or not (1 <= self.lookback_days <= 730):
            raise ConfigurationError("vectorization.analytics_backfill.lookback_days must be int in [1, 730]")
        if not isinstance(self.max_records, int) or self.max_records < 1:
            raise ConfigurationError("vectorization.analytics_backfill.max_records must be int >= 1")
        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool) or self.batch_size < 1:
            raise ConfigurationError("vectorization.analytics_backfill.batch_size must be int >= 1")
        if not isinstance(self.ensure_schema_once, bool):
            raise ConfigurationError("vectorization.analytics_backfill.ensure_schema_once must be a bool")
        if not isinstance(self.insert_timeout_seconds, (int, float)) or isinstance(self.insert_timeout_seconds, bool) or float(self.insert_timeout_seconds) <= 0.0:
            raise ConfigurationError("vectorization.analytics_backfill.insert_timeout_seconds must be a number > 0")
        if not isinstance(self.schema_timeout_seconds, (int, float)) or isinstance(self.schema_timeout_seconds, bool) or float(self.schema_timeout_seconds) <= 0.0:
            raise ConfigurationError("vectorization.analytics_backfill.schema_timeout_seconds must be a number > 0")
        self.insert_timeout_seconds = float(self.insert_timeout_seconds)
        self.schema_timeout_seconds = float(self.schema_timeout_seconds)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'AnalyticsBackfillConfig':
        _require(
            d,
            [
                'lookback_days',
                'max_records',
                'batch_size',
                'ensure_schema_once',
                'insert_timeout_seconds',
                'schema_timeout_seconds',
            ],
            'vectorization.analytics_backfill',
        )
        return cls(
            lookback_days=int(d['lookback_days']),
            max_records=int(d['max_records']),
            batch_size=int(d['batch_size']),
            ensure_schema_once=bool(d['ensure_schema_once']),
            insert_timeout_seconds=float(d['insert_timeout_seconds']),
            schema_timeout_seconds=float(d['schema_timeout_seconds']),
        )


@dataclass
class BidEventIngestConfig:
    """Config for real-time bid event ingest from the_resale_place.item_bids_cln.

    :param enabled: bool - Master switch.
    :param interval_seconds: float - Poll cadence in seconds.
    :param max_consecutive_failures: int - Pause threshold.
    :param source_database: str - Athena database (``the_resale_place``).
    :param source_table: str - Bid events table (``item_bids_cln``).
    :param winning_bids_table: str - Join table for auction_id (``item_winning_bids_cln``).
    :param lookback_minutes: int - Cold-start lookback window.
    :param chunk_minutes: int - Time-window slice per Athena query.
    :param batch_size: int - LIMIT applied to the dedup CTE.
    :param timeout_seconds: float - Athena query wall-clock cap.
    """

    enabled: bool
    interval_seconds: float
    max_consecutive_failures: int
    source_database: str
    source_table: str
    winning_bids_table: str
    lookback_minutes: int
    chunk_minutes: int
    batch_size: int
    timeout_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("vectorization.event_ingest.bid_events.enabled must be bool")
        if not isinstance(self.interval_seconds, (int, float)) or float(self.interval_seconds) <= 0.0:
            raise ConfigurationError("vectorization.event_ingest.bid_events.interval_seconds must be > 0")
        self.interval_seconds = float(self.interval_seconds)
        if not isinstance(self.max_consecutive_failures, int) or self.max_consecutive_failures < 1:
            raise ConfigurationError("vectorization.event_ingest.bid_events.max_consecutive_failures must be int >= 1")
        if not isinstance(self.source_database, str) or not self.source_database.strip():
            raise ConfigurationError("vectorization.event_ingest.bid_events.source_database must be non-empty")
        if not isinstance(self.source_table, str) or not self.source_table.strip():
            raise ConfigurationError("vectorization.event_ingest.bid_events.source_table must be non-empty")
        if not isinstance(self.winning_bids_table, str) or not self.winning_bids_table.strip():
            raise ConfigurationError("vectorization.event_ingest.bid_events.winning_bids_table must be non-empty")
        if not isinstance(self.lookback_minutes, int) or self.lookback_minutes < 1:
            raise ConfigurationError("vectorization.event_ingest.bid_events.lookback_minutes must be int >= 1")
        if not isinstance(self.chunk_minutes, int) or self.chunk_minutes < 1:
            raise ConfigurationError("vectorization.event_ingest.bid_events.chunk_minutes must be int >= 1")
        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ConfigurationError("vectorization.event_ingest.bid_events.batch_size must be int >= 1")
        if not isinstance(self.timeout_seconds, (int, float)) or float(self.timeout_seconds) <= 0.0:
            raise ConfigurationError("vectorization.event_ingest.bid_events.timeout_seconds must be > 0")
        self.timeout_seconds = float(self.timeout_seconds)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'BidEventIngestConfig':
        _require(d, ['enabled', 'interval_seconds', 'max_consecutive_failures', 'source_database',
                     'source_table', 'winning_bids_table', 'lookback_minutes', 'chunk_minutes',
                     'batch_size', 'timeout_seconds'], 'vectorization.event_ingest.bid_events')
        return cls(
            enabled=_coerce_bool(d['enabled']),
            interval_seconds=float(d['interval_seconds']),
            max_consecutive_failures=int(d['max_consecutive_failures']),
            source_database=str(d['source_database']).strip(),
            source_table=str(d['source_table']).strip(),
            winning_bids_table=str(d['winning_bids_table']).strip(),
            lookback_minutes=int(d['lookback_minutes']),
            chunk_minutes=int(d['chunk_minutes']),
            batch_size=int(d['batch_size']),
            timeout_seconds=float(d['timeout_seconds']),
        )


@dataclass
class WatchEventIngestConfig:
    """Config for real-time watch event ingest from the_resale_place.member_items_watch_cln.

    :param enabled: bool - Master switch.
    :param interval_seconds: float - Poll cadence in seconds.
    :param max_consecutive_failures: int - Pause threshold.
    :param source_database: str - Athena database (``the_resale_place``).
    :param source_table: str - Watch events table (``member_items_watch_cln``).
    :param watch_types_table: str - Dimension table for labels (``member_items_watch_types_cln``).
    :param lookback_minutes: int - Cold-start lookback window.
    :param chunk_minutes: int - Time-window slice per Athena query.
    :param batch_size: int - LIMIT applied to the dedup CTE.
    :param timeout_seconds: float - Athena query wall-clock cap.
    """

    enabled: bool
    interval_seconds: float
    max_consecutive_failures: int
    source_database: str
    source_table: str
    watch_types_table: str
    lookback_minutes: int
    chunk_minutes: int
    batch_size: int
    timeout_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("vectorization.event_ingest.watch_events.enabled must be bool")
        if not isinstance(self.interval_seconds, (int, float)) or float(self.interval_seconds) <= 0.0:
            raise ConfigurationError("vectorization.event_ingest.watch_events.interval_seconds must be > 0")
        self.interval_seconds = float(self.interval_seconds)
        if not isinstance(self.max_consecutive_failures, int) or self.max_consecutive_failures < 1:
            raise ConfigurationError("vectorization.event_ingest.watch_events.max_consecutive_failures must be int >= 1")
        if not isinstance(self.source_database, str) or not self.source_database.strip():
            raise ConfigurationError("vectorization.event_ingest.watch_events.source_database must be non-empty")
        if not isinstance(self.source_table, str) or not self.source_table.strip():
            raise ConfigurationError("vectorization.event_ingest.watch_events.source_table must be non-empty")
        if not isinstance(self.watch_types_table, str) or not self.watch_types_table.strip():
            raise ConfigurationError("vectorization.event_ingest.watch_events.watch_types_table must be non-empty")
        if not isinstance(self.lookback_minutes, int) or self.lookback_minutes < 1:
            raise ConfigurationError("vectorization.event_ingest.watch_events.lookback_minutes must be int >= 1")
        if not isinstance(self.chunk_minutes, int) or self.chunk_minutes < 1:
            raise ConfigurationError("vectorization.event_ingest.watch_events.chunk_minutes must be int >= 1")
        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ConfigurationError("vectorization.event_ingest.watch_events.batch_size must be int >= 1")
        if not isinstance(self.timeout_seconds, (int, float)) or float(self.timeout_seconds) <= 0.0:
            raise ConfigurationError("vectorization.event_ingest.watch_events.timeout_seconds must be > 0")
        self.timeout_seconds = float(self.timeout_seconds)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'WatchEventIngestConfig':
        _require(d, ['enabled', 'interval_seconds', 'max_consecutive_failures', 'source_database',
                     'source_table', 'watch_types_table', 'lookback_minutes', 'chunk_minutes',
                     'batch_size', 'timeout_seconds'], 'vectorization.event_ingest.watch_events')
        return cls(
            enabled=_coerce_bool(d['enabled']),
            interval_seconds=float(d['interval_seconds']),
            max_consecutive_failures=int(d['max_consecutive_failures']),
            source_database=str(d['source_database']).strip(),
            source_table=str(d['source_table']).strip(),
            watch_types_table=str(d['watch_types_table']).strip(),
            lookback_minutes=int(d['lookback_minutes']),
            chunk_minutes=int(d['chunk_minutes']),
            batch_size=int(d['batch_size']),
            timeout_seconds=float(d['timeout_seconds']),
        )


@dataclass
class QdrantEnrichConfig:
    """Qdrant payload enrichment from ClickHouse bid/watch MVs.

    :param enabled: bool - Master switch for Qdrant enrichment after each ingest cycle.
    :param enrich_limit: int - Max auction_ids enriched per cycle.
    """

    enabled: bool
    enrich_limit: int

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("vectorization.event_ingest.qdrant_enrich.enabled must be bool")
        if not isinstance(self.enrich_limit, int) or self.enrich_limit < 1:
            raise ConfigurationError("vectorization.event_ingest.qdrant_enrich.enrich_limit must be int >= 1")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QdrantEnrichConfig':
        _require(d, ['enabled', 'enrich_limit'], 'vectorization.event_ingest.qdrant_enrich')
        return cls(
            enabled=_coerce_bool(d['enabled']),
            enrich_limit=int(d['enrich_limit']),
        )


@dataclass
class EventIngestionConfig:
    """Container for bid + watch event ingest configs + Qdrant enrichment.

    All sub-blocks are optional; omitting a block disables that source.

    :param bid_events: Optional[BidEventIngestConfig] - Bid events ingest.
    :param watch_events: Optional[WatchEventIngestConfig] - Watch events ingest.
    :param qdrant_enrich: Optional[QdrantEnrichConfig] - Qdrant enrichment settings.
    """

    bid_events: Optional['BidEventIngestConfig'] = None
    watch_events: Optional['WatchEventIngestConfig'] = None
    qdrant_enrich: Optional['QdrantEnrichConfig'] = None

    def __post_init__(self) -> None:
        if self.bid_events is not None and not isinstance(self.bid_events, BidEventIngestConfig):
            raise ConfigurationError("vectorization.event_ingest.bid_events must be BidEventIngestConfig or None")
        if self.watch_events is not None and not isinstance(self.watch_events, WatchEventIngestConfig):
            raise ConfigurationError("vectorization.event_ingest.watch_events must be WatchEventIngestConfig or None")
        if self.qdrant_enrich is not None and not isinstance(self.qdrant_enrich, QdrantEnrichConfig):
            raise ConfigurationError("vectorization.event_ingest.qdrant_enrich must be QdrantEnrichConfig or None")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'EventIngestionConfig':
        bid_raw = d.get('bid_events')
        watch_raw = d.get('watch_events')
        enrich_raw = d.get('qdrant_enrich')
        return cls(
            bid_events=BidEventIngestConfig.from_dict(bid_raw) if isinstance(bid_raw, dict) else None,
            watch_events=WatchEventIngestConfig.from_dict(watch_raw) if isinstance(watch_raw, dict) else None,
            qdrant_enrich=QdrantEnrichConfig.from_dict(enrich_raw) if isinstance(enrich_raw, dict) else None,
        )


@dataclass
class CompoundSplitterConfig:
    """Compound-word splitter for unspaced domain registrable labels.

    The splitter takes a label like ``techstartup`` and emits
    ``[tech, startup]`` against an injected unigram dictionary. The
    dictionary itself is loaded from a CSV file whose every row is
    ``word,frequency`` — there is no built-in vocabulary; an empty or
    missing dictionary means the splitter is disabled even when
    ``enabled=True``. See
    :class:`semantic_search.vectorization.compound_splitter.CompoundWordSplitter`
    for the algorithm.

    :param enabled: bool - Master switch. When ``False`` the segmenter
        is constructed without a splitter and emits the legacy
        single-token-per-alpha-run output (preserves all pre-existing
        behaviour).
    :param dictionary_path: str - Filesystem path to the unigram
        frequency CSV. Each row must be ``word,frequency`` with the
        header line ``word,frequency``. Path traversal (``..``,
        absolute paths outside the workspace) is rejected at load
        time. Required when ``enabled=True``.
    :param min_segment_length: int - Smallest dictionary segment the
        splitter will emit. Must be ``>= 1``; ``2`` is typical for
        English (filters single-letter noise).
    :param max_segments: int - Hard cap on emitted segments per label.
        Labels that would split into more segments than this fall back
        to the single-token baseline. Must be ``>= 1``.
    :param oov_char_cost: float - Per-character cost charged when a
        run has no dictionary coverage. Must be ``> 0``; higher values
        discourage out-of-dictionary segmentation.
    :param length_penalty: float - Per-character bonus subtracted from
        each in-dictionary segment's cost. ``0.0`` disables length
        preference; ``0.5`` gently biases toward longer matches.
    """
    enabled: bool
    dictionary_path: str
    min_segment_length: int
    max_segments: int
    oov_char_cost: float
    length_penalty: float

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("vectorization.compound_splitter.enabled must be a bool")
        if not isinstance(self.dictionary_path, str) or not self.dictionary_path.strip():
            raise ConfigurationError( "vectorization.compound_splitter.dictionary_path must be a non-empty string" )
        if not isinstance(self.min_segment_length, int) or self.min_segment_length < 1:
            raise ConfigurationError( "vectorization.compound_splitter.min_segment_length must be int >= 1" )
        if not isinstance(self.max_segments, int) or self.max_segments < 1:
            raise ConfigurationError( "vectorization.compound_splitter.max_segments must be int >= 1" )
        if not isinstance(self.oov_char_cost, (int, float)) or float(self.oov_char_cost) <= 0.0:
            raise ConfigurationError( "vectorization.compound_splitter.oov_char_cost must be a number > 0" )
        if not isinstance(self.length_penalty, (int, float)) or float(self.length_penalty) < 0.0:
            raise ConfigurationError( "vectorization.compound_splitter.length_penalty must be a number >= 0" )
        self.oov_char_cost = float(self.oov_char_cost)
        self.length_penalty = float(self.length_penalty)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CompoundSplitterConfig':
        _require( d, ['enabled', 'dictionary_path', 'min_segment_length', 'max_segments', 'oov_char_cost', 'length_penalty'], 'vectorization.compound_splitter', )
        return cls(
            enabled=bool(d['enabled']),
            dictionary_path=str(d['dictionary_path']).strip(),
            min_segment_length=int(d['min_segment_length']),
            max_segments=int(d['max_segments']),
            oov_char_cost=float(d['oov_char_cost']),
            length_penalty=float(d['length_penalty']),
        )


@dataclass
class SeedTableConfig:
    """Per-table configuration for the database seed fetch.

    Both strategies apply ``LIMIT max_records`` so the result set is always
    bounded regardless of which strategy is used.

    :param table_name: str - Table name within the configured database
        (e.g. ``auction_audit_cln``)
    :param strategy: str - ``"datewise"`` pulls records within the last
        ``lookback_days`` days (by ``auctionstarttime``); ``"count"`` pulls
        the most recent ``max_records`` rows ordered by ``auctionstarttime DESC``
    :param lookback_days: int - Rolling window in days (``datewise`` only)
    :param max_records: int - Hard row cap applied to both strategies
    :param active_only: bool - When True, additionally filters
        ``auctionendtime > CURRENT_TIMESTAMP`` so only live auctions enter the
        index.  Default False (include recently-ended auctions in lookback window).
        Recommended True for Qdrant semantic-search indexes; False for analytics
        back-fills.
    """
    table_name: str
    strategy: str
    lookback_days: int
    max_records: int
    active_only: bool = False

    _VALID_STRATEGIES = ("datewise", "count")

    def __post_init__(self) -> None:
        if not isinstance(self.table_name, str) or not self.table_name.strip():
            raise ConfigurationError("vectorization.seed.database.tables[].table_name must be a non-empty string")
        if self.strategy not in self._VALID_STRATEGIES:
            raise ConfigurationError( f"vectorization.seed.database.tables[].strategy must be one of {self._VALID_STRATEGIES}" )
        if not isinstance(self.lookback_days, int) or self.lookback_days < 1:
            raise ConfigurationError("vectorization.seed.database.tables[].lookback_days must be int >= 1")
        if not isinstance(self.max_records, int) or self.max_records < 1:
            raise ConfigurationError("vectorization.seed.database.tables[].max_records must be int >= 1")

    @classmethod
    def from_dict(cls, d: Dict[str, Any], context: str = "vectorization.seed.database.tables[]") -> 'SeedTableConfig':
        _require(d, ['table_name', 'strategy', 'lookback_days', 'max_records'], context)
        _active = bool(d.get('active_only', False))
        return cls( table_name=str(d['table_name']), strategy=str(d['strategy']), lookback_days=int(d['lookback_days']), max_records=int(d['max_records']), active_only=_active, )


@dataclass
class SeedMajesticConfig:
    """Majestic feature-mart snapshot table used to enrich seed documents.

    Joined to the seed source by ``domain_name`` (case-insensitive) to
    populate ``majestic_ext_back_links``, ``majestic_ref_domains_fm``,
    ``majestic_citation_flow_score``, ``majestic_trust_flow_score``, and
    ``majestic_metric_exists`` on every indexed document.

    :param database: str - Athena database owning the Majestic snapshot table
        (e.g. ``domain_feature_mart``)
    :param table_name: str - Table name within ``database``
        (e.g. ``domain_majestic_metric_snap``)
    :param timeout_seconds: float - Per-query Athena timeout for the majestic join
    """
    database: str
    table_name: str
    timeout_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.database, str) or not self.database.strip():
            raise ConfigurationError("vectorization.seed.database.majestic.database must be a non-empty string")
        if not isinstance(self.table_name, str) or not self.table_name.strip():
            raise ConfigurationError("vectorization.seed.database.majestic.table_name must be a non-empty string")
        if not isinstance(self.timeout_seconds, (int, float)) or self.timeout_seconds <= 0:
            raise ConfigurationError("vectorization.seed.database.majestic.timeout_seconds must be > 0")
        self.timeout_seconds = float(self.timeout_seconds)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SeedMajesticConfig':
        _require(d, ['database', 'table_name', 'timeout_seconds'], 'vectorization.seed.database.majestic')
        return cls(
            database=str(d['database']),
            table_name=str(d['table_name']),
            timeout_seconds=float(d['timeout_seconds']),
        )


@dataclass
class SeedSearchRollupConfig:
    """Search-log rollup table used to enrich seed documents with
    ``unique_search_count``.

    Joined to the seed source by ``domain_name`` (case-insensitive) to
    populate ``unique_search_count`` (``COUNT(DISTINCT customer_id)`` over
    ``lookback_days``) on every indexed document.

    :param database: str - Athena database owning the search rollup table
    :param table_name: str - Table name within ``database``
    :param lookback_days: int - Rolling window in days for the distinct-customer count
    :param timeout_seconds: float - Per-query Athena timeout for the rollup join
    """
    database: str
    table_name: str
    lookback_days: int
    timeout_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.database, str) or not self.database.strip():
            raise ConfigurationError("vectorization.seed.database.search_rollup.database must be a non-empty string")
        if not isinstance(self.table_name, str) or not self.table_name.strip():
            raise ConfigurationError("vectorization.seed.database.search_rollup.table_name must be a non-empty string")
        if not isinstance(self.lookback_days, int) or self.lookback_days < 1:
            raise ConfigurationError("vectorization.seed.database.search_rollup.lookback_days must be int >= 1")
        if not isinstance(self.timeout_seconds, (int, float)) or self.timeout_seconds <= 0:
            raise ConfigurationError("vectorization.seed.database.search_rollup.timeout_seconds must be > 0")
        self.timeout_seconds = float(self.timeout_seconds)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SeedSearchRollupConfig':
        _require(d, ['database', 'table_name', 'lookback_days', 'timeout_seconds'], 'vectorization.seed.database.search_rollup')
        return cls(
            database=str(d['database']),
            table_name=str(d['table_name']),
            lookback_days=int(d['lookback_days']),
            timeout_seconds=float(d['timeout_seconds']),
        )


@dataclass
class SeedSemrushConfig:
    """SEMrush domain-enrichment table used to enrich seed documents.

    Joined to the seed source by ``domain_name`` (case-insensitive) to
    populate ``semrush_ascore``, ``semrush_total``, ``semrush_domains_num``,
    ``semrush_urls_num``, ``semrush_keyword``, ``semrush_search_volume``,
    ``semrush_cpc``, and ``semrush_refdomains`` on every indexed document.

    :param database: str - Athena database owning the SEMrush enrichment table
        (e.g. ``domain_auction_mart``)
    :param table_name: str - Table name within ``database``
        (e.g. ``semrush_domain_enrichments``)
    :param timeout_seconds: float - Per-query Athena timeout for the semrush join
    """
    database: str
    table_name: str
    timeout_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.database, str) or not self.database.strip():
            raise ConfigurationError("vectorization.seed.database.semrush.database must be a non-empty string")
        if not isinstance(self.table_name, str) or not self.table_name.strip():
            raise ConfigurationError("vectorization.seed.database.semrush.table_name must be a non-empty string")
        if not isinstance(self.timeout_seconds, (int, float)) or self.timeout_seconds <= 0:
            raise ConfigurationError("vectorization.seed.database.semrush.timeout_seconds must be > 0")
        self.timeout_seconds = float(self.timeout_seconds)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SeedSemrushConfig':
        _require(d, ['database', 'table_name', 'timeout_seconds'], 'vectorization.seed.database.semrush')
        return cls(
            database=str(d['database']),
            table_name=str(d['table_name']),
            timeout_seconds=float(d['timeout_seconds']),
        )


@dataclass
class SeedEstibotConfig:
    """Estibot domain-enrichment table used to enrich seed documents.

    Joined to the seed source by ``domain_name`` (case-insensitive) to
    populate ``estibot_domain_count``, ``estibot_domain_count_dev``,
    ``estibot_ext_count``, and ``estibot_ext_count_dev`` on every indexed
    document.

    :param database: str - Athena database owning the Estibot enrichment table
        (e.g. ``domain_auction_mart``)
    :param table_name: str - Table name within ``database``
        (e.g. ``estibot_domain_enrichments``)
    :param timeout_seconds: float - Per-query Athena timeout for the estibot join
    """
    database: str
    table_name: str
    timeout_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.database, str) or not self.database.strip():
            raise ConfigurationError("vectorization.seed.database.estibot.database must be a non-empty string")
        if not isinstance(self.table_name, str) or not self.table_name.strip():
            raise ConfigurationError("vectorization.seed.database.estibot.table_name must be a non-empty string")
        if not isinstance(self.timeout_seconds, (int, float)) or self.timeout_seconds <= 0:
            raise ConfigurationError("vectorization.seed.database.estibot.timeout_seconds must be > 0")
        self.timeout_seconds = float(self.timeout_seconds)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SeedEstibotConfig':
        _require(d, ['database', 'table_name', 'timeout_seconds'], 'vectorization.seed.database.estibot')
        return cls(
            database=str(d['database']),
            table_name=str(d['table_name']),
            timeout_seconds=float(d['timeout_seconds']),
        )


@dataclass
class SeedAftermarketBoostConfig:
    """Aftermarket boost-tier flag, LEFT JOINed onto seed rows by domain_name
    at seed time via the Athena seed_merge CTAS pipeline so
    ``is_boosted_aftermarket`` is baked into the Qdrant payload rather than
    looked up per ``/search`` request.

    :param database: str - Athena database owning the boost-tier table
        (e.g. ``domain_aftermarket_mart``)
    :param table_name: str - Table name within ``database`` (e.g. ``ims_listing``)
    :param domain_name_column: str - Column holding the domain name
    :param tier_column: str - Column holding the boost tier label
    :param boosted_tier_values: List[str] - Tier values that mark a domain as boosted
    """
    database: str
    table_name: str
    domain_name_column: str
    tier_column: str
    boosted_tier_values: List[str]

    def __post_init__(self) -> None:
        if not isinstance(self.database, str) or not self.database.strip():
            raise ConfigurationError("vectorization.seed.database.aftermarket_boost.database must be a non-empty string")
        if not isinstance(self.table_name, str) or not self.table_name.strip():
            raise ConfigurationError("vectorization.seed.database.aftermarket_boost.table_name must be a non-empty string")
        if not isinstance(self.domain_name_column, str) or not self.domain_name_column.strip():
            raise ConfigurationError("vectorization.seed.database.aftermarket_boost.domain_name_column must be a non-empty string")
        if not isinstance(self.tier_column, str) or not self.tier_column.strip():
            raise ConfigurationError("vectorization.seed.database.aftermarket_boost.tier_column must be a non-empty string")
        if (
            not isinstance(self.boosted_tier_values, list)
            or not self.boosted_tier_values
            or not all(isinstance(v, str) and v.strip() for v in self.boosted_tier_values)
        ):
            raise ConfigurationError( "vectorization.seed.database.aftermarket_boost.boosted_tier_values must be a non-empty list of non-empty strings" )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SeedAftermarketBoostConfig':
        _require(
            d,
            ['database', 'table_name', 'domain_name_column', 'tier_column', 'boosted_tier_values'],
            'vectorization.seed.database.aftermarket_boost',
        )
        return cls(
            database=str(d['database']),
            table_name=str(d['table_name']),
            domain_name_column=str(d['domain_name_column']),
            tier_column=str(d['tier_column']),
            boosted_tier_values=[str(v).strip() for v in d['boosted_tier_values']],
        )


@dataclass
class SeedDatabaseConfig:
    """Database configuration for the data-build seed source.

    Always fetches real records from ``{name}.auction_audit_cln``
    (daily snapshot).  Set ``realtime_name`` to a live-replica database
    identifier in YAML (or via env-var override) to enable the
    ``realtime`` source.

    :param name: str - Daily-snapshot database name (from ``vectorization.seed.database.name``)
    :param realtime_name: str - Realtime-replica database name.
        Empty string means realtime source is unavailable;
        selecting ``source: realtime`` raises an error when this is empty.
    :param tables: List[SeedTableConfig] - Ordered list of tables to query
    :param max_records: int - Row cap per table; override per-request via the
        ``max_records`` form field up to ``max_records_cap``
    :param max_records_cap: int - YAML-configurable ceiling for ``max_records``.
        Absolute system hard-cap is 30 000 000.
    :param timeout_seconds: float - Per-query Athena timeout
    :param find_payload_aliases: Dict[str, List[str]] - Internal payload key to FIND listing aliases
    :param find_bool_aliases: Dict[str, str] - Internal 0/1 flag key to FIND bool listing key
    :param majestic: SeedMajesticConfig - Majestic feature-mart snapshot join config
    :param merge_database: str - Athena scratch database name for the seed-merge
        CTAS staging/final tables (``seed_merge.build_seed_merge_plan``). Required —
        no hardcoded fallback database is used.
    :param merge_database_location: str - S3 URI root for the scratch database's
        own storage location (``CREATE SCHEMA ... LOCATION``), distinct from the
        Athena query-results output prefix. Required — no hardcoded fallback.
    :param bid_source_database: str - Athena database for the bid-offer join
        (``seed_merge``'s ``stg_bid_offer_*`` CTAS). Required — no hardcoded fallback.
    :param bid_source_table: str - Bid events table within ``bid_source_database``.
        Required — no hardcoded fallback.
    :param bid_winning_table: str - Winning-bids join table within
        ``bid_source_database``. Required — no hardcoded fallback.
    :param merge_page_size: int - Rows per page baked into the seed-merge final
        CTAS's ``_page_num`` column. Required — no hardcoded fallback.
    :param aftermarket_boost: Optional[SeedAftermarketBoostConfig] - Aftermarket
        boost-tier join config. ``None`` means the feature is disabled — no
        hardcoded fallback table/tier is used.
    :param search_rollup: Optional[SeedSearchRollupConfig] - Search-log rollup
        join config for ``unique_search_count``. ``None`` means the feature is
        disabled — no hardcoded fallback table/window is used.
    :param semrush: Optional[SeedSemrushConfig] - SEMrush domain-enrichment
        join config. ``None`` means the feature is disabled — no hardcoded
        fallback table is used.
    :param estibot: Optional[SeedEstibotConfig] - Estibot domain-enrichment
        join config. ``None`` means the feature is disabled — no hardcoded
        fallback table is used.
    """
    name: str
    realtime_name: str
    tables: List[SeedTableConfig]
    max_records: int
    max_records_cap: int
    timeout_seconds: float
    find_payload_aliases: Dict[str, List[str]]
    find_bool_aliases: Dict[str, str]
    majestic: SeedMajesticConfig
    merge_database: str
    merge_database_location: str
    bid_source_database: str
    bid_source_table: str
    bid_winning_table: str
    merge_page_size: int
    aftermarket_boost: Optional['SeedAftermarketBoostConfig'] = None
    search_rollup: Optional['SeedSearchRollupConfig'] = None
    semrush: Optional['SeedSemrushConfig'] = None
    estibot: Optional['SeedEstibotConfig'] = None

    _MAX_RECORDS_HARD_CAP = 30_000_000

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ConfigurationError("vectorization.seed.database.name must be a non-empty string")
        if not isinstance(self.realtime_name, str):
            raise ConfigurationError("vectorization.seed.database.realtime_name must be a string")
        if not isinstance(self.tables, list) or len(self.tables) == 0:
            raise ConfigurationError("vectorization.seed.database.tables must be a non-empty list")
        for t in self.tables:
            if not isinstance(t, SeedTableConfig):
                raise ConfigurationError("vectorization.seed.database.tables entries must be SeedTableConfig")
        if not isinstance(self.max_records, int) or self.max_records < 1:
            raise ConfigurationError("vectorization.seed.database.max_records must be int >= 1")
        if not isinstance(self.max_records_cap, int) or self.max_records_cap < self.max_records:
            raise ConfigurationError( "vectorization.seed.database.max_records_cap must be int >= max_records" )
        if self.max_records_cap > self._MAX_RECORDS_HARD_CAP:
            raise ConfigurationError( f"vectorization.seed.database.max_records_cap must be <= {self._MAX_RECORDS_HARD_CAP:,}" )
        if not isinstance(self.timeout_seconds, (int, float)) or self.timeout_seconds <= 0:
            raise ConfigurationError("vectorization.seed.database.timeout_seconds must be > 0")
        self.timeout_seconds = float(self.timeout_seconds)
        if not isinstance(self.find_payload_aliases, dict) or not self.find_payload_aliases:
            raise ConfigurationError("vectorization.seed.database.find_payload_aliases must be a non-empty dict")
        for src, targets in self.find_payload_aliases.items():
            if not isinstance(src, str) or not src.strip():
                raise ConfigurationError("vectorization.seed.database.find_payload_aliases keys must be non-empty strings")
            if not isinstance(targets, list) or not targets or not all(isinstance(t, str) and t.strip() for t in targets):
                raise ConfigurationError("vectorization.seed.database.find_payload_aliases values must be non-empty string lists")
        if not isinstance(self.find_bool_aliases, dict):
            raise ConfigurationError("vectorization.seed.database.find_bool_aliases must be a dict")
        for src, target in self.find_bool_aliases.items():
            if not isinstance(src, str) or not src.strip() or not isinstance(target, str) or not target.strip():
                raise ConfigurationError("vectorization.seed.database.find_bool_aliases entries must be non-empty strings")
        if not isinstance(self.majestic, SeedMajesticConfig):
            raise ConfigurationError("vectorization.seed.database.majestic must be a SeedMajesticConfig")
        if not isinstance(self.merge_database, str) or not self.merge_database.strip():
            raise ConfigurationError("vectorization.seed.database.merge_database must be a non-empty string")
        if not isinstance(self.merge_database_location, str) or not self.merge_database_location.strip():
            raise ConfigurationError("vectorization.seed.database.merge_database_location must be a non-empty string")
        if not isinstance(self.bid_source_database, str) or not self.bid_source_database.strip():
            raise ConfigurationError("vectorization.seed.database.bid_source_database must be a non-empty string")
        if not isinstance(self.bid_source_table, str) or not self.bid_source_table.strip():
            raise ConfigurationError("vectorization.seed.database.bid_source_table must be a non-empty string")
        if not isinstance(self.bid_winning_table, str) or not self.bid_winning_table.strip():
            raise ConfigurationError("vectorization.seed.database.bid_winning_table must be a non-empty string")
        if not isinstance(self.merge_page_size, int) or isinstance(self.merge_page_size, bool) or self.merge_page_size < 1:
            raise ConfigurationError("vectorization.seed.database.merge_page_size must be int >= 1")
        if self.aftermarket_boost is not None and not isinstance(self.aftermarket_boost, SeedAftermarketBoostConfig):
            raise ConfigurationError("vectorization.seed.database.aftermarket_boost must be a SeedAftermarketBoostConfig")
        if self.search_rollup is not None and not isinstance(self.search_rollup, SeedSearchRollupConfig):
            raise ConfigurationError("vectorization.seed.database.search_rollup must be a SeedSearchRollupConfig")
        if self.semrush is not None and not isinstance(self.semrush, SeedSemrushConfig):
            raise ConfigurationError("vectorization.seed.database.semrush must be a SeedSemrushConfig")
        if self.estibot is not None and not isinstance(self.estibot, SeedEstibotConfig):
            raise ConfigurationError("vectorization.seed.database.estibot must be a SeedEstibotConfig")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SeedDatabaseConfig':
        _require(
            d,
            [
                'name', 'realtime_name', 'tables', 'max_records', 'max_records_cap', 'timeout_seconds',
                'find_payload_aliases', 'find_bool_aliases', 'majestic', 'merge_database',
                'merge_database_location',
                'bid_source_database', 'bid_source_table', 'bid_winning_table', 'merge_page_size',
            ],
            'vectorization.seed.database',
        )
        tables = [SeedTableConfig.from_dict(t) for t in d['tables']]
        aliases = {
            str(k).strip(): [str(t).strip() for t in v]
            for k, v in dict(d['find_payload_aliases']).items()
        }
        bool_aliases = {str(k).strip(): str(v).strip() for k, v in dict(d['find_bool_aliases']).items()}
        _boost_raw = d.get('aftermarket_boost')
        _rollup_raw = d.get('search_rollup')
        _semrush_raw = d.get('semrush')
        _estibot_raw = d.get('estibot')
        return cls(
            name=str(d['name']),
            realtime_name=str(d['realtime_name']),
            tables=tables,
            max_records=int(d['max_records']),
            max_records_cap=int(d['max_records_cap']),
            timeout_seconds=float(d['timeout_seconds']),
            find_payload_aliases=aliases,
            find_bool_aliases=bool_aliases,
            majestic=SeedMajesticConfig.from_dict(dict(d['majestic'])),
            merge_database=str(d['merge_database']),
            merge_database_location=str(d['merge_database_location']),
            bid_source_database=str(d['bid_source_database']),
            bid_source_table=str(d['bid_source_table']),
            bid_winning_table=str(d['bid_winning_table']),
            merge_page_size=int(d['merge_page_size']),
            aftermarket_boost=SeedAftermarketBoostConfig.from_dict(dict(_boost_raw)) if _boost_raw else None,
            search_rollup=SeedSearchRollupConfig.from_dict(dict(_rollup_raw)) if _rollup_raw else None,
            semrush=SeedSemrushConfig.from_dict(dict(_semrush_raw)) if _semrush_raw else None,
            estibot=SeedEstibotConfig.from_dict(dict(_estibot_raw)) if _estibot_raw else None,
        )


@dataclass
class SeedScheduleConfig:
    """Schedule for the seed data-build pipeline (``POST /data-build/full``).

    When ``enabled`` and ``in_process`` are both true, the service starts an
    asyncio loop that runs the build on the configured cadence. When
    ``in_process`` is false, schedule fields are still loaded and returned from
    ``GET /data-build/status`` for external callers (GitHub Actions).

    YAML may use ``${SEED_SCHEDULE_RUN_ON_DEPLOY}``,
    ``${SEED_SCHEDULE_INTERVAL_HOURS}``, and ``${SEED_SCHEDULE_RUN_AT_HOUR_UTC}``;
    the config loader expands those from the process environment at load time.

    :param enabled: bool - Master switch for scheduled ingest
    :param in_process: bool - Start the in-container background loop when true
    :param run_on_deploy: bool - Allow post-deploy ingest when true
    :param interval_hours: int - Hours between runs; also the first-fire delay
        when ``run_at_hour_utc`` is ``None``
    :param run_at_hour_utc: Optional[int] - UTC hour (0-23) for the first fire;
        later runs use ``interval_hours``. ``None`` means wait
        ``interval_hours`` after startup (in-process loop only)
    :param max_runtime_seconds: int - Wall-clock cap per ingest step (seed or
        analytics-backfill). When ClickHouse is enabled, a full sequence may use
        up to ``2 * max_runtime_seconds`` across the two sequential jobs.
    :param seed_mode: str - ``seed_mode`` for ``POST /data-build/full``
        (``auto`` | ``rebuild`` | ``append``)
    """
    enabled: bool
    in_process: bool
    run_on_deploy: bool
    interval_hours: int
    run_at_hour_utc: Optional[int]
    max_runtime_seconds: int
    seed_mode: str

    _VALID_SEED_MODES = ("auto", "rebuild", "append")

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("vectorization.seed.schedule.enabled must be bool")
        if not isinstance(self.in_process, bool):
            raise ConfigurationError("vectorization.seed.schedule.in_process must be bool")
        if not isinstance(self.run_on_deploy, bool):
            raise ConfigurationError("vectorization.seed.schedule.run_on_deploy must be bool")
        if not isinstance(self.interval_hours, int) or isinstance(self.interval_hours, bool) or self.interval_hours < 1:
            raise ConfigurationError("vectorization.seed.schedule.interval_hours must be int >= 1")
        if self.run_at_hour_utc is not None:
            if (
                not isinstance(self.run_at_hour_utc, int)
                or isinstance(self.run_at_hour_utc, bool)
                or not 0 <= self.run_at_hour_utc <= 23
            ):
                raise ConfigurationError(
                    "vectorization.seed.schedule.run_at_hour_utc must be 0–23 or null"
                )
        if (
            not isinstance(self.max_runtime_seconds, int)
            or isinstance(self.max_runtime_seconds, bool)
            or self.max_runtime_seconds < 60
        ):
            raise ConfigurationError(
                "vectorization.seed.schedule.max_runtime_seconds must be int >= 60"
            )
        if self.seed_mode not in self._VALID_SEED_MODES:
            raise ConfigurationError(
                f"vectorization.seed.schedule.seed_mode must be one of {self._VALID_SEED_MODES}"
            )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SeedScheduleConfig':
        _require(
            d,
            [
                'enabled',
                'in_process',
                'run_on_deploy',
                'interval_hours',
                'run_at_hour_utc',
                'max_runtime_seconds',
                'seed_mode',
            ],
            'vectorization.seed.schedule',
        )
        rahu = d['run_at_hour_utc']
        if rahu is None or (isinstance(rahu, str) and rahu.strip().lower() in {'', 'null', 'none'}):
            run_at_hour: Optional[int] = None
        else:
            run_at_hour = int(rahu)
        return cls(
            enabled=_coerce_bool(d['enabled']),
            in_process=_coerce_bool(d['in_process']),
            run_on_deploy=_coerce_bool(d['run_on_deploy']),
            interval_hours=int(d['interval_hours']),
            run_at_hour_utc=run_at_hour,
            max_runtime_seconds=int(d['max_runtime_seconds']),
            seed_mode=str(d['seed_mode']).strip(),
        )


@dataclass
class SeedStageTimingConfig:
    """Stage-wise wall-clock timing for ``/data-build/seed`` and boot seed.

    All fields are required in YAML. When ``enabled`` is False, timers are
    no-ops (no start/end log lines).

    :param enabled: bool - Master switch for stage timing logs
    :param log_page_stages: bool - Log per-page fetch / load / qdrant / CH spans
    :param log_merge_phases: bool - Log Athena merge create_database and CTAS spans
    :param log_indexer_substages: bool - Log OfflineIndexer encode vs upsert totals
    :param include_in_response: bool - Attach ``stage_timing`` summary to API JSON
    :param elapsed_ms_decimals: int - Decimal places for ``elapsed_ms`` values (>= 0)
    """
    enabled: bool
    log_page_stages: bool
    log_merge_phases: bool
    log_indexer_substages: bool
    include_in_response: bool
    elapsed_ms_decimals: int

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("vectorization.seed.stage_timing.enabled must be a bool")
        if not isinstance(self.log_page_stages, bool):
            raise ConfigurationError("vectorization.seed.stage_timing.log_page_stages must be a bool")
        if not isinstance(self.log_merge_phases, bool):
            raise ConfigurationError("vectorization.seed.stage_timing.log_merge_phases must be a bool")
        if not isinstance(self.log_indexer_substages, bool):
            raise ConfigurationError("vectorization.seed.stage_timing.log_indexer_substages must be a bool")
        if not isinstance(self.include_in_response, bool):
            raise ConfigurationError("vectorization.seed.stage_timing.include_in_response must be a bool")
        if not isinstance(self.elapsed_ms_decimals, int) or isinstance(self.elapsed_ms_decimals, bool) or self.elapsed_ms_decimals < 0:
            raise ConfigurationError("vectorization.seed.stage_timing.elapsed_ms_decimals must be int >= 0")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SeedStageTimingConfig':
        _require(
            d,
            [
                'enabled',
                'log_page_stages',
                'log_merge_phases',
                'log_indexer_substages',
                'include_in_response',
                'elapsed_ms_decimals',
            ],
            'vectorization.seed.stage_timing',
        )
        return cls(
            enabled=bool(d['enabled']),
            log_page_stages=bool(d['log_page_stages']),
            log_merge_phases=bool(d['log_merge_phases']),
            log_indexer_substages=bool(d['log_indexer_substages']),
            include_in_response=bool(d['include_in_response']),
            elapsed_ms_decimals=int(d['elapsed_ms_decimals']),
        )


@dataclass
class SeedConfig:
    """Boot-time seed corpus configuration.

    Always fetches real domain records from ``{database.name}.auction_audit_cln``
    via the ``database`` block.  No synthetic fallback — the endpoint
    requires Athena credentials.

    ``source`` selects which database instance to query:

    ``daily_snapshot`` (default)
        Queries ``database.name`` — the T+1 daily
        snapshot refreshed every 24 hours via bulk load (analysis_auction.md §5).

    ``realtime``
        Queries ``database.realtime_name`` — a live-replica identifier
        configured in YAML when a realtime feed becomes available.
        Raises ``ConfigurationError`` at boot if ``realtime_name`` is empty.

    :param enabled: bool - When True, seed data is loaded into in-memory
        indexes at startup
    :param source: str - ``"daily_snapshot"`` or ``"realtime"``
    :param batch_yield_size: int - Yield to the event loop every N documents
        during ``load_seed_into_indexes``
    :param encode_batch_size: int - Documents per dense ``encode_batch`` call
        during in-memory seed load
    :param clickhouse_batch_size: int - Rows per ClickHouse INSERT round-trip
        for seed and boot seed writes
    :param clickhouse_ensure_schema_once: bool - When True, DDL ensure runs on
        the first ClickHouse write of a job only; when False, every page
    :param clickhouse_insert_timeout_seconds: float - Per-batch INSERT timeout
    :param clickhouse_schema_timeout_seconds: float - Per-DDL statement timeout
        during schema ensure
    :param database: SeedDatabaseConfig - Database connection and table config
    :param stage_timing: SeedStageTimingConfig - Stage wall-clock timing knobs
    :param schedule: Optional[SeedScheduleConfig] - Seed data-build schedule;
        ``None`` omits schedule fields from status; see ``SeedScheduleConfig``
    """
    enabled: bool
    source: str
    batch_yield_size: int
    encode_batch_size: int
    clickhouse_batch_size: int
    clickhouse_ensure_schema_once: bool
    clickhouse_insert_timeout_seconds: float
    clickhouse_schema_timeout_seconds: float
    database: 'SeedDatabaseConfig'
    stage_timing: SeedStageTimingConfig
    schedule: Optional['SeedScheduleConfig'] = None

    _VALID_SOURCES = ("daily_snapshot", "realtime")

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("vectorization.seed.enabled must be a bool")
        if self.source not in self._VALID_SOURCES:
            raise ConfigurationError( f"vectorization.seed.source must be one of {self._VALID_SOURCES}" )
        if not isinstance(self.batch_yield_size, int) or self.batch_yield_size < 1:
            raise ConfigurationError("vectorization.seed.batch_yield_size must be int >= 1")
        if not isinstance(self.encode_batch_size, int) or self.encode_batch_size < 1:
            raise ConfigurationError("vectorization.seed.encode_batch_size must be int >= 1")
        if not isinstance(self.clickhouse_batch_size, int) or isinstance(self.clickhouse_batch_size, bool) or self.clickhouse_batch_size < 1:
            raise ConfigurationError("vectorization.seed.clickhouse_batch_size must be int >= 1")
        if not isinstance(self.clickhouse_ensure_schema_once, bool):
            raise ConfigurationError("vectorization.seed.clickhouse_ensure_schema_once must be a bool")
        if not isinstance(self.clickhouse_insert_timeout_seconds, (int, float)) or isinstance(self.clickhouse_insert_timeout_seconds, bool) or float(self.clickhouse_insert_timeout_seconds) <= 0.0:
            raise ConfigurationError("vectorization.seed.clickhouse_insert_timeout_seconds must be a number > 0")
        if not isinstance(self.clickhouse_schema_timeout_seconds, (int, float)) or isinstance(self.clickhouse_schema_timeout_seconds, bool) or float(self.clickhouse_schema_timeout_seconds) <= 0.0:
            raise ConfigurationError("vectorization.seed.clickhouse_schema_timeout_seconds must be a number > 0")
        self.clickhouse_insert_timeout_seconds = float(self.clickhouse_insert_timeout_seconds)
        self.clickhouse_schema_timeout_seconds = float(self.clickhouse_schema_timeout_seconds)
        if not isinstance(self.database, SeedDatabaseConfig):
            raise ConfigurationError("vectorization.seed.database must be a SeedDatabaseConfig")
        if not isinstance(self.stage_timing, SeedStageTimingConfig):
            raise ConfigurationError("vectorization.seed.stage_timing must be a SeedStageTimingConfig")
        if self.source == "realtime" and not self.database.realtime_name.strip():
            raise ConfigurationError( "vectorization.seed.database.realtime_name must be set when source == 'realtime'" )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SeedConfig':
        _require(
            d,
            [
                'enabled',
                'batch_yield_size',
                'encode_batch_size',
                'clickhouse_batch_size',
                'clickhouse_ensure_schema_once',
                'clickhouse_insert_timeout_seconds',
                'clickhouse_schema_timeout_seconds',
                'database',
                'stage_timing',
            ],
            'vectorization.seed',
        )
        db_raw = d.get('database')
        if db_raw is None:
            raise ConfigurationError("vectorization.seed.database block is required")
        database = SeedDatabaseConfig.from_dict(db_raw)
        stage_timing = SeedStageTimingConfig.from_dict(dict(d['stage_timing']))
        schedule_raw = d.get('schedule')
        schedule = SeedScheduleConfig.from_dict(schedule_raw) if schedule_raw else None
        return cls(
            enabled=bool(d['enabled']),
            source=str(d.get('source', 'daily_snapshot')),
            batch_yield_size=int(d['batch_yield_size']),
            encode_batch_size=int(d['encode_batch_size']),
            clickhouse_batch_size=int(d['clickhouse_batch_size']),
            clickhouse_ensure_schema_once=bool(d['clickhouse_ensure_schema_once']),
            clickhouse_insert_timeout_seconds=float(d['clickhouse_insert_timeout_seconds']),
            clickhouse_schema_timeout_seconds=float(d['clickhouse_schema_timeout_seconds']),
            database=database,
            stage_timing=stage_timing,
            schedule=schedule,
        )


@dataclass
class VectorizationConfig:
    """Offline domain-name vectorization pipeline config.

    Bundle for the offline indexer + refresh driver. The pipeline composes:
    segmentation (``DomainNameSegmenter``) ->optional compound-word
    splitting (``CompoundWordSplitter``) ->optional synonym expansion
    (reuses the query-side ``SynonymExpansionConfig`` so the same map
    applies on both sides) ->BM25 sparse encoding (``BM25DocEncoder``).

    The whole block is OPTIONAL on ``AgentSearchConfig`` so existing YAML
    continues to load. When present, the registry constructs the
    indexer + refresh driver; when absent the offline pipeline is simply
    not available.

    :param enabled: bool - Master switch. When ``False`` the indexer +
        refresh driver are NOT constructed even when the block is present.
    :param synonyms: Optional[SynonymExpansionConfig] - Synonym map applied
        on the doc side. ``None`` (or ``enabled=False``) means no
        expansion. When present and enabled, the same map SHOULD also be
        wired into ``BM25QueryEncoderConfig.synonyms`` so query and
        document terms align — a top-level cross-field check enforces the
        symmetry when both sides carry a non-empty map.
    :param compound_splitter: Optional[CompoundSplitterConfig] - When
        present and enabled, the segmenter applies a Viterbi compound
        splitter to ASCII-alpha runs so unspaced compounds like
        ``techstartup`` emit as ``[tech, startup]``. ``None`` (or
        ``enabled=False``) preserves the legacy single-token output.
    :param bm25_doc_encoder: BM25DocEncoderConfig - Corpus-side BM25
        encoder parameters.
    :param indexer: IndexerConfig - Batch-size + idempotency knobs for
        the offline upsert path.
    :param refresh: VectorRefreshConfig - Snapshot-version-aware refresh
        driver parameters.
    :param encoder_query_prefix: str - Task-type prefix prepended to
        every document before dense encoding. Set via
        ``vectorization.encoder_query_prefix`` in YAML. Task separation
        requires different prefixes for query vs document sides — omitting
        this degrades retrieval recall. Empty string disables prefixing.
    """
    enabled: bool
    bm25_doc_encoder: BM25DocEncoderConfig
    indexer: IndexerConfig
    refresh: VectorRefreshConfig
    synonyms: Optional[SynonymExpansionConfig] = None
    compound_splitter: Optional[CompoundSplitterConfig] = None
    seed: Optional[SeedConfig] = None
    encoder_query_prefix: str = "search_document: "
    delta_refresh: Optional[DeltaRefreshConfig] = None
    analytics_backfill: Optional[AnalyticsBackfillConfig] = None
    event_ingest: Optional[EventIngestionConfig] = None
    enrichment_refresh: Optional[EnrichmentRefreshConfig] = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("vectorization.enabled must be a bool")
        if not isinstance(self.bm25_doc_encoder, BM25DocEncoderConfig):
            raise ConfigurationError("vectorization.bm25_doc_encoder must be a BM25DocEncoderConfig")
        if not isinstance(self.indexer, IndexerConfig):
            raise ConfigurationError("vectorization.indexer must be an IndexerConfig")
        if not isinstance(self.refresh, VectorRefreshConfig):
            raise ConfigurationError("vectorization.refresh must be a VectorRefreshConfig")
        if self.synonyms is not None and not isinstance(self.synonyms, SynonymExpansionConfig):
            raise ConfigurationError("vectorization.synonyms must be a SynonymExpansionConfig or None")
        if self.compound_splitter is not None and not isinstance(self.compound_splitter, CompoundSplitterConfig):
            raise ConfigurationError( "vectorization.compound_splitter must be a CompoundSplitterConfig or None" )
        if self.seed is not None and not isinstance(self.seed, SeedConfig):
            raise ConfigurationError("vectorization.seed must be a SeedConfig or None")
        if not isinstance(self.encoder_query_prefix, str):
            raise ConfigurationError("vectorization.encoder_query_prefix must be a string")
        if self.delta_refresh is not None and not isinstance(self.delta_refresh, DeltaRefreshConfig):
            raise ConfigurationError("vectorization.delta_refresh must be a DeltaRefreshConfig or None")
        if self.analytics_backfill is not None and not isinstance(self.analytics_backfill, AnalyticsBackfillConfig):
            raise ConfigurationError("vectorization.analytics_backfill must be an AnalyticsBackfillConfig or None")
        if self.event_ingest is not None and not isinstance(self.event_ingest, EventIngestionConfig):
            raise ConfigurationError("vectorization.event_ingest must be an EventIngestionConfig or None")
        if self.enrichment_refresh is not None and not isinstance(self.enrichment_refresh, EnrichmentRefreshConfig):
            raise ConfigurationError("vectorization.enrichment_refresh must be an EnrichmentRefreshConfig or None")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'VectorizationConfig':
        _require(d, ['enabled', 'bm25_doc_encoder', 'indexer', 'refresh'], 'vectorization')
        synonyms_raw = d.get('synonyms')
        synonyms = SynonymExpansionConfig.from_dict(synonyms_raw) if isinstance(synonyms_raw, dict) else None
        compound_raw = d.get('compound_splitter')
        compound_splitter = (
            CompoundSplitterConfig.from_dict(compound_raw)
            if isinstance(compound_raw, dict)
            else None
        )
        seed_raw = d.get('seed')
        seed = SeedConfig.from_dict(seed_raw) if isinstance(seed_raw, dict) else None
        delta_raw = d.get('delta_refresh')
        delta_refresh = DeltaRefreshConfig.from_dict(delta_raw) if isinstance(delta_raw, dict) else None
        abf_raw = d.get('analytics_backfill')
        analytics_backfill = AnalyticsBackfillConfig.from_dict(abf_raw) if isinstance(abf_raw, dict) else None
        ei_raw = d.get('event_ingest')
        event_ingest = EventIngestionConfig.from_dict(ei_raw) if isinstance(ei_raw, dict) else None
        er_raw = d.get('enrichment_refresh')
        enrichment_refresh = EnrichmentRefreshConfig.from_dict(er_raw) if isinstance(er_raw, dict) else None
        return cls(
            enabled=bool(d['enabled']),
            bm25_doc_encoder=BM25DocEncoderConfig.from_dict(d['bm25_doc_encoder']),
            indexer=IndexerConfig.from_dict(d['indexer']),
            refresh=VectorRefreshConfig.from_dict(d['refresh']),
            synonyms=synonyms,
            compound_splitter=compound_splitter,
            seed=seed,
            encoder_query_prefix=str(d.get('encoder_query_prefix', 'search_document: ')),
            delta_refresh=delta_refresh,
            analytics_backfill=analytics_backfill,
            event_ingest=event_ingest,
            enrichment_refresh=enrichment_refresh,
        )


@dataclass
class LLMStructuralGateConfig:
    """OneOf-discriminator structural gate config."""
    enabled: bool
    probe_ttl_seconds: int
    probe_timeout_seconds: float
    probe_max_tokens: int
    probe_system_prompt: str
    probe_user_prompt: str
    treat_unknown_as_capable: bool

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("llm_structural_gate.enabled must be bool")
        if int(self.probe_ttl_seconds) < 1:
            raise ConfigurationError("llm_structural_gate.probe_ttl_seconds must be >= 1")
        if float(self.probe_timeout_seconds) <= 0.0:
            raise ConfigurationError("llm_structural_gate.probe_timeout_seconds must be > 0")
        if int(self.probe_max_tokens) < 1:
            raise ConfigurationError("llm_structural_gate.probe_max_tokens must be >= 1")
        if not isinstance(self.probe_system_prompt, str) or not str(self.probe_system_prompt).strip():
            raise ConfigurationError("llm_structural_gate.probe_system_prompt must be non-empty string")
        if not isinstance(self.probe_user_prompt, str) or not str(self.probe_user_prompt).strip():
            raise ConfigurationError("llm_structural_gate.probe_user_prompt must be non-empty string")
        if not isinstance(self.treat_unknown_as_capable, bool):
            raise ConfigurationError("llm_structural_gate.treat_unknown_as_capable must be bool")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'LLMStructuralGateConfig':
        _require( d, [ 'enabled', 'probe_ttl_seconds', 'probe_timeout_seconds', 'probe_max_tokens', 'probe_system_prompt', 'probe_user_prompt', 'treat_unknown_as_capable', ], 'llm_structural_gate', )
        return cls(
            enabled=bool(d['enabled']),
            probe_ttl_seconds=int(d['probe_ttl_seconds']),
            probe_timeout_seconds=float(d['probe_timeout_seconds']),
            probe_max_tokens=int(d['probe_max_tokens']),
            probe_system_prompt=str(d['probe_system_prompt']),
            probe_user_prompt=str(d['probe_user_prompt']),
            treat_unknown_as_capable=bool(d['treat_unknown_as_capable']),
        )


@dataclass
class GuidanceConfig:
    """Market-guidance snapshot path (ClickHouse-backed when analytics CH is wired).

    Ranked listings for guidance intents use hybrid-first retrieve
    (``general.search.ranked_results_complement``); this block only configures
    the optional ``guidance`` envelope snapshot.

    :param enabled: bool - When True the orchestrator may attach a ``GuidanceEnvelope``
    :param market_snapshot_sql: str - Read-only SQL executed on the analytics ClickHouse
        executor (must return rows the service JSON-serializes into the envelope body)
    :param max_payload_chars: int - Hard cap on ``GuidanceEnvelope.body`` length (>= 64)
    :param clickhouse_db: str - ClickHouse database name substituted into market_snapshot_sql at load time
    :param snapshot_timeout_seconds: float - Cap for awaiting the pre-fired snapshot task
    :param snapshot_unavailable_notice: str - User-facing notice when the market snapshot
        cannot be built (CH down / disabled / empty SQL) while ranked_results still return
    """
    enabled: bool
    market_snapshot_sql: str
    max_payload_chars: int
    clickhouse_db: str
    snapshot_timeout_seconds: float
    snapshot_unavailable_notice: str

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("guidance.enabled must be a bool")
        if not isinstance(self.market_snapshot_sql, str):
            raise ConfigurationError("guidance.market_snapshot_sql must be a string")
        if self.enabled and not self.market_snapshot_sql.strip():
            raise ConfigurationError("guidance.market_snapshot_sql must be non-empty when guidance.enabled=true")
        if int(self.max_payload_chars) < 64:
            raise ConfigurationError("guidance.max_payload_chars must be >= 64")
        if float(self.snapshot_timeout_seconds) <= 0.0:
            raise ConfigurationError("guidance.snapshot_timeout_seconds must be > 0")
        if not isinstance(self.snapshot_unavailable_notice, str) or not self.snapshot_unavailable_notice.strip():
            raise ConfigurationError("guidance.snapshot_unavailable_notice must be a non-empty string")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'GuidanceConfig':
        _require(
            d,
            [
                'enabled',
                'market_snapshot_sql',
                'max_payload_chars',
                'clickhouse_db',
                'snapshot_timeout_seconds',
                'snapshot_unavailable_notice',
            ],
            'guidance',
        )
        db = str(d['clickhouse_db'])
        def _sub(sql: str) -> str:
            return sql.replace('{clickhouse_db}', db) if db else sql
        return cls(
            enabled=bool(d['enabled']),
            clickhouse_db=db,
            market_snapshot_sql=_sub(str(d['market_snapshot_sql'])),
            max_payload_chars=int(d['max_payload_chars']),
            snapshot_timeout_seconds=float(d['snapshot_timeout_seconds']),
            snapshot_unavailable_notice=str(d['snapshot_unavailable_notice']),
        )


@dataclass
class VagueQuantifierConfig:
    """Config for the vague-quantifier resolver (vague_quantifier block).

    :param high_authority_min: int - Semrush authority score threshold for phrases like
        "high authority", "strong authority", "good SEO".
    :param strong_backlinks_tf_min: int - Majestic Trust Flow threshold for phrases like
        "strong backlinks", "good backlinks", "backlink juice".
    """
    enabled: bool
    decent_traffic_floor: int
    strong_traffic_floor: int
    affordable_price_cap: int
    expensive_price_floor: int
    premium_govalue_floor: int
    expiring_soon_seconds: int
    domain_age_mature_years: int
    short_name_max_chars: int
    long_name_min_chars: int
    high_authority_min: int
    strong_backlinks_tf_min: int

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("vague_quantifier.enabled must be a bool")
        for _fname, _val in [
            ('decent_traffic_floor', self.decent_traffic_floor),
            ('strong_traffic_floor', self.strong_traffic_floor),
            ('affordable_price_cap', self.affordable_price_cap),
            ('expensive_price_floor', self.expensive_price_floor),
            ('premium_govalue_floor', self.premium_govalue_floor),
            ('expiring_soon_seconds', self.expiring_soon_seconds),
            ('domain_age_mature_years', self.domain_age_mature_years),
            ('short_name_max_chars', self.short_name_max_chars),
            ('long_name_min_chars', self.long_name_min_chars),
            ('high_authority_min', self.high_authority_min),
            ('strong_backlinks_tf_min', self.strong_backlinks_tf_min),
        ]:
            if int(_val) < 0:
                raise ConfigurationError(f"vague_quantifier.{_fname} must be >= 0")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'VagueQuantifierConfig':
        _require(
            d,
            [
                'enabled', 'decent_traffic_floor', 'strong_traffic_floor', 'affordable_price_cap',
                'expensive_price_floor', 'premium_govalue_floor', 'expiring_soon_seconds',
                'domain_age_mature_years', 'short_name_max_chars', 'long_name_min_chars',
                'high_authority_min', 'strong_backlinks_tf_min',
            ],
            'vague_quantifier',
        )
        return cls(
            enabled=bool(d['enabled']),
            decent_traffic_floor=int(d['decent_traffic_floor']),
            strong_traffic_floor=int(d['strong_traffic_floor']),
            affordable_price_cap=int(d['affordable_price_cap']),
            expensive_price_floor=int(d['expensive_price_floor']),
            premium_govalue_floor=int(d['premium_govalue_floor']),
            expiring_soon_seconds=int(d['expiring_soon_seconds']),
            domain_age_mature_years=int(d['domain_age_mature_years']),
            short_name_max_chars=int(d['short_name_max_chars']),
            long_name_min_chars=int(d['long_name_min_chars']),
            high_authority_min=int(d['high_authority_min']),
            strong_backlinks_tf_min=int(d['strong_backlinks_tf_min']),
        )


@dataclass
class ClickHouseLeverConfig:
    """Master ClickHouse lever (``clickhouse.enabled`` in YAML).

    When ``enabled`` is False, boot forces analytics / explore CH rails /
    price-band CH adapter off. CI and local compose skip the ClickHouse service.
    """
    enabled: bool

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("clickhouse.enabled must be a bool")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ClickHouseLeverConfig':
        _require(d, ['enabled'], 'clickhouse')
        return cls(enabled=bool(d['enabled']))


@dataclass
class AgentSearchConfig:
    """Top-level config bundle for the semantic_search service."""
    general: GeneralConfig
    qi: QIConfig
    retrieval: RetrievalConfig
    cache: CacheConfig
    surface: SurfaceConfig
    history: HistoryConfig
    explore: ExploreConfig
    guidance: GuidanceConfig
    multi_intent: MultiIntentConfig
    resilience: ResilienceConfig
    measurement: MeasurementConfig
    nl_to_sql: NLToSQLConfig
    ingest: IngestConfig
    inventory: InventoryConfig
    calibration: CalibrationConfig
    feedback: FeedbackConfig
    identity: IdentityConfig
    offline_eval: OfflineEvalConfig
    llm_structural_gate: LLMStructuralGateConfig
    safety: SafetyConfig
    clickhouse: ClickHouseLeverConfig
    cost_budget: Optional[CostBudgetConfig] = None
    reasoning_trace: Optional[ReasoningTraceConfig] = None
    vectorization: Optional[VectorizationConfig] = None
    vague_quantifier: Optional[VagueQuantifierConfig] = None

    def __post_init__(self) -> None:
        cascade_enabled = (
            self.qi.encoder.cascade is not None
            and self.qi.encoder.cascade.enabled
            and self.qi.encoder.cascade.stage_dims is not None
        )
        if not cascade_enabled:
            if self.qi.semantic.embedding_dim != self.retrieval.vector.embedding_dim:
                raise ConfigurationError("qi.semantic.embedding_dim must equal retrieval.vector.embedding_dim")
        # Vectorization symmetry: when the offline pipeline is wired AND the
        # query side has a non-empty synonym map, the doc side MUST carry a
        # non-empty map too (otherwise query terms expand but corpus terms
        # don't and the BM25 fusion silently misses matches). The reverse is
        # also enforced. Both empty is fine; one empty + one non-empty is not.
        if self.vectorization is not None:
            qside = self._query_synonym_map_or_none()
            dside = self.vectorization.synonyms.synonym_map if self.vectorization.synonyms is not None else None
            qside_active = bool(qside) and self._query_synonyms_enabled()
            dside_active = bool(dside) and self.vectorization.synonyms is not None and self.vectorization.synonyms.enabled
            if qside_active != dside_active:
                raise ConfigurationError(
                    "vectorization.synonyms and retrieval.qdrant.hybrid.bm25_query_encoder.synonyms must "
                    "be either both enabled with non-empty maps or both disabled/empty so query-side and "
                    "doc-side expansion stay symmetric"
                )

    def _query_synonym_map_or_none(self) -> Optional[Dict[str, List[str]]]:
        """Return the query-side synonym map if wired, else None.

        Walks ``retrieval.qdrant.hybrid.bm25_query_encoder.synonyms``
        defensively: every node is Optional in the config schema so any
        missing layer collapses to None without raising.
        """
        try:
            qcfg = self.retrieval.qdrant
            if qcfg is None or qcfg.hybrid is None or qcfg.hybrid.bm25_query_encoder is None:
                return None
            syn = qcfg.hybrid.bm25_query_encoder.synonyms
            return syn.synonym_map if syn is not None else None
        except AttributeError:
            return None

    def _query_synonyms_enabled(self) -> bool:
        """Return True iff the query-side synonym block is present + enabled."""
        try:
            qcfg = self.retrieval.qdrant
            if qcfg is None or qcfg.hybrid is None or qcfg.hybrid.bm25_query_encoder is None:
                return False
            syn = qcfg.hybrid.bm25_query_encoder.synonyms
            return syn is not None and bool(syn.enabled)
        except AttributeError:
            return False

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'AgentSearchConfig':
        _require(
            d,
            [
                'general', 'qi', 'retrieval', 'cache', 'surface', 'history', 'explore', 'guidance', 'multi_intent',
                'resilience', 'measurement', 'nl_to_sql', 'ingest', 'inventory', 'calibration', 'feedback', 'identity',
                'offline_eval',
                'llm_structural_gate',
                'safety',
                'clickhouse',
            ],
            'semantic_search',
        )
        # cost_budget is OPTIONAL for additivity. Existing
        # deployments without the block keep the legacy zero-overhead
        # path (cost_budget_factory=None in the orchestrator).
        cost_budget_block = d.get('cost_budget')
        cost_budget = CostBudgetConfig.from_dict(cost_budget_block) if cost_budget_block is not None else None
        # reasoning_trace is OPTIONAL for additivity. Existing
        # deployments without the block keep the legacy zero-overhead path
        # (the orchestrator never constructs a ``ReasoningTrace``).
        reasoning_trace_block = d.get('reasoning_trace')
        reasoning_trace = (
            ReasoningTraceConfig.from_dict(reasoning_trace_block)
            if reasoning_trace_block is not None
            else None
        )
        # vectorization is OPTIONAL for additivity. Existing deployments
        # without the block keep the legacy zero-overhead path (no offline
        # indexer + no refresh driver constructed).
        vectorization_block = d.get('vectorization')
        vectorization = (
            VectorizationConfig.from_dict(vectorization_block)
            if vectorization_block is not None
            else None
        )
        # vague_quantifier is OPTIONAL for additivity. Existing deployments
        # without the block skip vague-phrase resolution (no behaviour change).
        vq_block = d.get('vague_quantifier')
        vague_quantifier = VagueQuantifierConfig.from_dict(vq_block) if vq_block is not None else None
        return cls(
            general=GeneralConfig.from_dict(d['general']),
            qi=QIConfig.from_dict(d['qi']),
            retrieval=RetrievalConfig.from_dict(d['retrieval']),
            cache=CacheConfig.from_dict(d['cache']),
            surface=SurfaceConfig.from_dict(d['surface']),
            history=HistoryConfig.from_dict(d['history']),
            explore=ExploreConfig.from_dict(d['explore']),
            guidance=GuidanceConfig.from_dict(d['guidance']),
            multi_intent=MultiIntentConfig.from_dict(d['multi_intent']),
            resilience=ResilienceConfig.from_dict(d['resilience']),
            measurement=MeasurementConfig.from_dict(d['measurement']),
            nl_to_sql=NLToSQLConfig.from_dict(d['nl_to_sql']),
            ingest=IngestConfig.from_dict(d['ingest']),
            inventory=InventoryConfig.from_dict(d['inventory']),
            calibration=CalibrationConfig.from_dict(d['calibration']),
            feedback=FeedbackConfig.from_dict(d['feedback']),
            identity=IdentityConfig.from_dict(d['identity']),
            offline_eval=OfflineEvalConfig.from_dict(d['offline_eval']),
            llm_structural_gate=LLMStructuralGateConfig.from_dict(d['llm_structural_gate']),
            safety=SafetyConfig.from_dict(d['safety']),
            clickhouse=ClickHouseLeverConfig.from_dict(d['clickhouse']),
            cost_budget=cost_budget,
            reasoning_trace=reasoning_trace,
            vectorization=vectorization,
            vague_quantifier=vague_quantifier,
        )


