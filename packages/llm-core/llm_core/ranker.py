"""Ranker — score and order models by weighted capability/cost/latency, optionally adjusted by runtime stats."""
from dataclasses import dataclass
from functools import cmp_to_key
from typing import Any, Dict, List, Optional, Tuple

from llm_core.logging_utils import get_logger
from llm_core.model_registry import _detect_provider
from llm_core.pricing import infer_model_family, infer_model_recency

logger = get_logger(__name__)

# Entry: (model, score, cost_tiebreak, latency_tiebreak, family, recency, provider_pref)
# latency_tiebreak is -p50_ms when measured (higher = faster), else None (skip in cmp).
_RankEntry = Tuple[str, float, float, Optional[float], str, Tuple[float, int, int, int], int]


@dataclass(frozen=True)
class RankWeights:
    """Composite scoring weights summing to ~1.0 after normalization.
    :param capability: float - Weight on capability tier (higher = better)
    :param cost: float - Weight on cost (higher = cheaper preferred)
    :param latency: float - Weight on latency (higher = faster preferred)
    """
    capability: float
    cost: float
    latency: float

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "RankWeights":
        """Build from explicit config dict. All three keys are required and non-negative.
        :param raw: Dict[str, Any] - Must contain capability, cost, latency keys
        :return: RankWeights - Normalized weights
        :raises ValueError: If any key missing or sum <= 0
        """
        if not isinstance(raw, dict):
            raise ValueError("RankWeights.from_dict requires a dict with capability/cost/latency keys")
        for key in ('capability', 'cost', 'latency'):
            if key not in raw:
                raise ValueError(f"RankWeights.from_dict missing required key: {key}")
        cap = float(raw['capability'])
        cost = float(raw['cost'])
        lat = float(raw['latency'])
        total = cap + cost + lat
        if total <= 0:
            raise ValueError(f"RankWeights weights must sum to > 0; got capability={cap}, cost={cost}, latency={lat}")
        return cls(capability=cap / total, cost=cost / total, latency=lat / total)


class Ranker:
    """Composite scorer with optional runtime adjustment from observed signals.

    Scoring formula:
        score = w_cap * capability_tier
              + w_cost * (6 - cost_tier)
              + w_lat * (6 - effective_latency_tier)

    Where effective_latency_tier blends static config tier with observed p50_latency
    when runtime adaptation is enabled and sample_count >= min_samples.
    Models with error_rate above the demotion threshold are pushed to the end of the chain.

    Tiebreaks (after equal composite score):
      1. Cheaper `total_cost` wins.
      2. Faster measured inference (`-p50_latency_ms`) when both models have
         runtime/startup probe samples — same-cost pairs prefer lowest latency.
      3. Same product family — higher `infer_model_recency` wins (latest grok
         beats older grok; never lets grok jump Claude on version alone).
      4. Different families — higher config `provider_tiebreak_priority` wins
         (e.g. anthropic before openai before google).
    """

    def __init__(
        self,
        latency_buckets_ms: List[int],
        error_demotion_threshold: float,
        runtime_min_samples: int,
        provider_tiebreak_priority: List[str],
    ):
        """Initialize ranker. All parameters are required (no silent defaults).
        :param latency_buckets_ms: List[int] - Exactly 4 thresholds (ms) splitting tiers 1..5
        :param error_demotion_threshold: float - Error rate above which a model is demoted to the end
        :param runtime_min_samples: int - Min samples before runtime stats override static tiers
        :param provider_tiebreak_priority: List[str] - Provider ids, first = highest cross-family preference
        :raises ValueError: If latency_buckets_ms does not have exactly 4 thresholds or priority invalid
        """
        if not isinstance(latency_buckets_ms, list) or len(latency_buckets_ms) != 4:
            raise ValueError("latency_buckets_ms must be a list of exactly 4 thresholds")
        if not isinstance(provider_tiebreak_priority, list) or not provider_tiebreak_priority:
            raise ValueError("provider_tiebreak_priority must be a non-empty list")
        self._buckets = [int(b) for b in latency_buckets_ms]
        self._error_threshold = float(error_demotion_threshold)
        self._min_samples = int(runtime_min_samples)
        n = len(provider_tiebreak_priority)
        # Higher int = preferred. Providers omitted from the list get 0.
        self._provider_pref: Dict[str, int] = {
            str(pid).strip().lower(): n - i for i, pid in enumerate(provider_tiebreak_priority)
        }
        # Per-instance dedup so the unregistered-model warning fires once per
        # model per process (rank() is called dozens of times at startup, once
        # per dimension/component, which would otherwise spam the log).
        self._unregistered_warned: set = set()

    def _provider_preference(self, model: str) -> int:
        """Return config preference rank for model's provider (higher = preferred)."""
        return self._provider_pref.get(_detect_provider(model), 0)

    def _latency_to_tier(self, p50_ms: float) -> int:
        """Bucket observed latency into tier 1..5 (1=fastest)."""
        if p50_ms <= self._buckets[0]:
            return 1
        if p50_ms <= self._buckets[1]:
            return 2
        if p50_ms <= self._buckets[2]:
            return 3
        if p50_ms <= self._buckets[3]:
            return 4
        return 5

    def _effective_latency_tier(self, model: str, static_tier: int, runtime_stats: Optional[Dict[str, Any]]) -> int:
        """Return latency tier blended with runtime observation when enough samples."""
        if not runtime_stats:
            return static_tier
        stats = runtime_stats.get(model)
        if not stats:
            return static_tier
        sample_count = int(stats.get('sample_count', 0))
        p50 = stats.get('p50_latency_ms')
        if sample_count < self._min_samples or p50 is None:
            return static_tier
        return self._latency_to_tier(float(p50))

    def _is_demoted(self, model: str, runtime_stats: Optional[Dict[str, Any]]) -> bool:
        """True when error rate above demotion threshold with enough samples."""
        if not runtime_stats:
            return False
        stats = runtime_stats.get(model)
        if not stats:
            return False
        sample_count = int(stats.get('sample_count', 0))
        if sample_count < self._min_samples:
            return False
        return float(stats.get('error_rate', 0.0)) > self._error_threshold

    def _measured_latency_tiebreak(
        self, model: str, runtime_stats: Optional[Dict[str, Any]],
    ) -> Optional[float]:
        """Return ``-p50_ms`` when enough samples exist (higher = faster); else None."""
        if not runtime_stats:
            return None
        stats = runtime_stats.get(model)
        if not stats:
            return None
        sample_count = int(stats.get('sample_count', 0))
        p50 = stats.get('p50_latency_ms')
        if sample_count < self._min_samples or p50 is None:
            return None
        return -float(p50)

    _NEUTRAL_TIER = 3
    _SCORE_BASE = 6
    _NEUTRAL_META = {'capability_tier': _NEUTRAL_TIER, 'cost_tier': _NEUTRAL_TIER, 'latency_tier': _NEUTRAL_TIER, 'total_cost': 0.0}

    def score_model(self, model: str, meta: Dict[str, Any], weights: RankWeights, runtime_stats: Optional[Dict[str, Any]] = None) -> float:
        """Compute composite score for one model.
        :param model: str - Model name
        :param meta: Dict[str, Any] - Registry metadata; must contain capability_tier/cost_tier/latency_tier
        :param weights: RankWeights - Composite weights
        :param runtime_stats: Optional[Dict[str, Any]] - Per-model runtime stats keyed by model name
        :return: float - Composite score
        :raises KeyError: If meta is missing any required tier key
        """
        cap_tier = int(meta['capability_tier'])
        cost_tier = int(meta['cost_tier'])
        lat_tier = int(meta['latency_tier'])
        eff_lat_tier = self._effective_latency_tier(model, lat_tier, runtime_stats)
        return weights.capability * cap_tier + weights.cost * (self._SCORE_BASE - cost_tier) + weights.latency * (self._SCORE_BASE - eff_lat_tier)

    @staticmethod
    def _cmp_rank_entries(a: _RankEntry, b: _RankEntry) -> int:
        """Compare rank entries: score, then cost, then measured latency, then same-family recency, else provider.

        Cross-family pairs ignore version recency so grok cannot displace Claude on
        version alone; provider_tiebreak_priority decides those ties instead.
        Measured-latency tiebreak applies only when both sides have probe/runtime p50.
        """
        if a[1] != b[1]:
            return 1 if a[1] > b[1] else -1
        if a[2] != b[2]:
            return 1 if a[2] > b[2] else -1
        lat_a, lat_b = a[3], b[3]
        if lat_a is not None and lat_b is not None and lat_a != lat_b:
            return 1 if lat_a > lat_b else -1
        if a[4] == b[4] and a[4]:
            if a[5] != b[5]:
                return 1 if a[5] > b[5] else -1
            return 0
        if a[6] != b[6]:
            return 1 if a[6] > b[6] else -1
        return 0

    def rank(self, models: List[str], registry: Dict[str, Dict[str, Any]], weights: RankWeights, runtime_stats: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float]]:
        """Rank models by composite score. Demoted models pushed to the end.
        Models missing from `registry` are scored with neutral tiers (logged) instead of dropped,
        because callers (e.g. ensemble selectors) may pass models that have not been registered.
        :param models: List[str] - Candidate models
        :param registry: Dict[str, Dict[str, Any]] - Model metadata registry
        :param weights: RankWeights - Composite weights
        :param runtime_stats: Optional[Dict[str, Any]] - Per-model runtime stats
        :return: List[Tuple[str, float]] - Ordered (model, score) pairs
        """
        if not models:
            return []
        primary: List[_RankEntry] = []
        demoted: List[_RankEntry] = []
        for model in models:
            meta = registry.get(model)
            if meta is None:
                if model not in self._unregistered_warned:
                    logger.warning(f"ranker_unregistered_model model={model} using_neutral_tiers=true")
                    self._unregistered_warned.add(model)
                meta = self._NEUTRAL_META
            score = self.score_model(model, meta, weights, runtime_stats)
            cost_tie = -float(meta.get('total_cost', 0.0))
            latency_tie = self._measured_latency_tiebreak(model, runtime_stats)
            family = infer_model_family(model)
            recency = infer_model_recency(model)
            provider_pref = self._provider_preference(model)
            entry: _RankEntry = (model, score, cost_tie, latency_tie, family, recency, provider_pref)
            if self._is_demoted(model, runtime_stats):
                demoted.append(entry)
            else:
                primary.append(entry)
        key_fn = cmp_to_key(Ranker._cmp_rank_entries)
        primary.sort(key=key_fn, reverse=True)
        demoted.sort(key=key_fn, reverse=True)
        return [(m, s) for m, s, _, _, _, _, _ in primary + demoted]
