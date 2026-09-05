"""Retrieval-quality evaluator.

Runs a labeled ``RelevanceJudgedQuery`` set through ``SearchOrchestrator.search``
and computes per-query + aggregated NDCG@k / Recall@k / MRR@k / Precision@k.
Library-only — no HTTP surface. Intended consumers:

- Manual offline runs (``RetrievalQualityEvaluator(...).run()`` from a script)
- CI or notebooks that batch-score labeled ``RelevanceJudgedQuery`` sets

Difficulty / edge-type slicing is done at aggregation time so the report
includes both overall and per-slice metrics.

The evaluator is a pure consumer of ``SearchOrchestrator.search`` — it does
not mutate orchestrator state, does not write to caches or signal stores, and
emits only ``logger.info`` lines (no metric/event sinks; per the user's scope
directive, observability is provided by Katana, not by this package).

Math: see ``semantic_search.eval.retrieval_metrics``.
"""
import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from semantic_search.config.models import RetrievalEvalConfig
from semantic_search.contracts import RankedResults, RelevanceJudgedQuery
from semantic_search.core.exceptions import AgentSearchError, ConfigurationError, ValidationError
from semantic_search.core.logging_utils import get_logger
from semantic_search.eval.retrieval_metrics import mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k

logger = get_logger(__name__)


@dataclass(frozen=True)
class PerQueryMetric:
    """Per-query metrics + the slice tags used for aggregation.

    :param query_id: str - From the ``RelevanceJudgedQuery``
    :param difficulty: str - Slice key (mirrors GoldenCase.difficulty)
    :param edge_type: str - Slice key (mirrors GoldenCase.edge_type)
    :param ndcg: float - NDCG@k in [0,1]
    :param recall: float - Recall@k in [0,1]
    :param mrr: float - Reciprocal rank in [0,1]
    :param precision: float - Precision@k in [0,1]
    :param items_returned: int - Count of items the orchestrator actually returned
    :param latency_ms: float - Wall-clock time for the orchestrator call
    :param error: Optional[str] - Categorical error name (None on success)
    """
    query_id: str
    difficulty: str
    edge_type: str
    ndcg: float
    recall: float
    mrr: float
    precision: float
    items_returned: int
    latency_ms: float
    error: Optional[str]


@dataclass(frozen=True)
class RetrievalEvalReport:
    """Aggregated report from one evaluation pass.

    :param k: int - Cutoff rank used (top_k from config)
    :param query_count: int - Number of queries scored (excludes hard errors)
    :param error_count: int - Queries that hit a hard error (excluded from means)
    :param mean_ndcg: float - Mean NDCG@k across non-error queries
    :param mean_recall: float - Mean Recall@k across non-error queries
    :param mean_mrr: float - Mean reciprocal rank across non-error queries
    :param mean_precision: float - Mean Precision@k across non-error queries
    :param mean_latency_ms: float - Mean orchestrator latency across non-error queries
    :param verdict: str - 'passes_release_gate' / 'fails_release_gate'
    :param failure_reasons: List[str] - Audit-log reasons (e.g. 'mean_recall_below_threshold')
    :param slice_metrics: Dict[str, Dict[str, float]] - {slice_key: {metric_name: value}}
        where slice_key is e.g. 'difficulty=easy' or 'edge_type=adversarial'.
        Each value is the slice mean across that slice's non-error queries.
    :param per_query: List[PerQueryMetric] - Full per-query record for downstream use
    """
    k: int
    query_count: int
    error_count: int
    mean_ndcg: float
    mean_recall: float
    mean_mrr: float
    mean_precision: float
    mean_latency_ms: float
    verdict: str
    failure_reasons: List[str] = field(default_factory=list)
    slice_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    per_query: List[PerQueryMetric] = field(default_factory=list)


class RetrievalQualityEvaluator:
    """Score ``SearchOrchestrator.search`` output against a labeled judged set.

    Concurrency: queries are run via ``asyncio.gather`` with a semaphore bound
    by ``config.max_concurrent_queries`` so a large judged set cannot stampede
    the orchestrator (which itself fans out to retrievers + LLM).

    :param config: RetrievalEvalConfig - Top-k, thresholds, concurrency cap
    :param search_fn: Async callable matching ``SearchOrchestrator.search``
        signature, narrowed to ``(raw_query, request_id) -> Tuple[RankedResults, ...]``.
        Accepting the callable rather than the orchestrator object decouples
        the evaluator from the full orchestrator surface and keeps tests light.
    """

    def __init__(self, config: RetrievalEvalConfig, search_fn: Any):
        if not isinstance(config, RetrievalEvalConfig):
            raise ConfigurationError("RetrievalQualityEvaluator requires a RetrievalEvalConfig")
        if not callable(search_fn):
            raise ConfigurationError("RetrievalQualityEvaluator search_fn must be a callable")
        self._config = config
        self._search_fn = search_fn
        self._semaphore = asyncio.Semaphore(int(config.max_concurrent_queries))

    async def _score_one(self, query: RelevanceJudgedQuery) -> PerQueryMetric:
        """Score a single judged query against the orchestrator output."""
        if not isinstance(query, RelevanceJudgedQuery):
            raise ValidationError("RetrievalQualityEvaluator._score_one requires a RelevanceJudgedQuery")
        request_id = f"{self._config.request_id_prefix}_{query.query_id}"
        k = int(self._config.top_k)
        async with self._semaphore:
            t0 = time.monotonic()
            try:
                result = await self._search_fn(raw_query=query.input_query, request_id=request_id)
            except (AgentSearchError, asyncio.TimeoutError) as e:
                latency_ms = (time.monotonic() - t0) * 1000.0
                logger.warning(
                    f"retrieval_eval_query_failed query_id={query.query_id} "
                    f"error_type={type(e).__name__} error={str(e)} latency_ms={latency_ms:.1f}"
                )
                return PerQueryMetric(
                    query_id=query.query_id,
                    difficulty=query.difficulty,
                    edge_type=query.edge_type,
                    ndcg=0.0, recall=0.0, mrr=0.0, precision=0.0,
                    items_returned=0,
                    latency_ms=latency_ms,
                    error=type(e).__name__,
                )
            latency_ms = (time.monotonic() - t0) * 1000.0
        ranked_results = self._extract_ranked_results(result)
        ranked_ids = [item.item_id for item in ranked_results.items]
        gain_map = query.gain_by_item_id
        relevant_ids = query.relevant_item_ids
        ndcg = ndcg_at_k(ranked_ids, gain_map, k)
        recall = recall_at_k(ranked_ids, relevant_ids, k)
        mrr = mrr_at_k(ranked_ids, relevant_ids, k)
        precision = precision_at_k(ranked_ids, relevant_ids, k)
        return PerQueryMetric(
            query_id=query.query_id,
            difficulty=query.difficulty,
            edge_type=query.edge_type,
            ndcg=ndcg, recall=recall, mrr=mrr, precision=precision,
            items_returned=len(ranked_ids),
            latency_ms=latency_ms,
            error=None,
        )

    @staticmethod
    def _extract_ranked_results(result: Any) -> RankedResults:
        """Pull RankedResults out of the orchestrator's tuple return.

        ``SearchOrchestrator.search`` returns a 3-tuple
        ``(RankedResults, ERankerOutcome, ZeroResultGuardOutcome)``.
        We only need the first element.
        """
        if isinstance(result, RankedResults):
            return result
        if isinstance(result, tuple) and result and isinstance(result[0], RankedResults):
            return result[0]
        raise ValidationError(
            f"RetrievalQualityEvaluator search_fn returned unexpected shape: {type(result).__name__}"
        )

    async def run(self, queries: Sequence[RelevanceJudgedQuery]) -> RetrievalEvalReport:
        """Score every query and produce an aggregated report.

        :param queries: Sequence[RelevanceJudgedQuery] - Closed-world judged set
        :return: RetrievalEvalReport - Aggregated + per-query + per-slice metrics
        """
        if not queries:
            logger.warning("retrieval_eval_no_queries verdict=fails_release_gate")
            return RetrievalEvalReport(
                k=int(self._config.top_k),
                query_count=0, error_count=0,
                mean_ndcg=0.0, mean_recall=0.0, mean_mrr=0.0, mean_precision=0.0,
                mean_latency_ms=0.0,
                verdict='fails_release_gate',
                failure_reasons=['no_queries'],
                slice_metrics={},
                per_query=[],
            )
        per_query: List[PerQueryMetric] = await asyncio.gather(*[self._score_one(q) for q in queries])
        successes = [m for m in per_query if m.error is None]
        error_count = len(per_query) - len(successes)
        if not successes:
            logger.warning(
                f"retrieval_eval_all_queries_errored total={len(queries)} verdict=fails_release_gate"
            )
            return RetrievalEvalReport(
                k=int(self._config.top_k),
                query_count=0, error_count=error_count,
                mean_ndcg=0.0, mean_recall=0.0, mean_mrr=0.0, mean_precision=0.0,
                mean_latency_ms=0.0,
                verdict='fails_release_gate',
                failure_reasons=['all_queries_errored'],
                slice_metrics={},
                per_query=per_query,
            )
        mean_ndcg = sum(m.ndcg for m in successes) / len(successes)
        mean_recall = sum(m.recall for m in successes) / len(successes)
        mean_mrr = sum(m.mrr for m in successes) / len(successes)
        mean_precision = sum(m.precision for m in successes) / len(successes)
        mean_latency_ms = sum(m.latency_ms for m in successes) / len(successes)
        slice_metrics = self._aggregate_slices(successes)
        failure_reasons: List[str] = []
        verdict = self._evaluate_release_gate(
            mean_ndcg, mean_recall, mean_mrr, error_count, len(queries), failure_reasons
        )
        logger.info(
            f"retrieval_eval_completed queries={len(queries)} errors={error_count} "
            f"k={self._config.top_k} mean_ndcg={mean_ndcg:.4f} mean_recall={mean_recall:.4f} "
            f"mean_mrr={mean_mrr:.4f} mean_precision={mean_precision:.4f} "
            f"mean_latency_ms={mean_latency_ms:.1f} verdict={verdict} reasons={failure_reasons}"
        )
        return RetrievalEvalReport(
            k=int(self._config.top_k),
            query_count=len(successes),
            error_count=error_count,
            mean_ndcg=mean_ndcg,
            mean_recall=mean_recall,
            mean_mrr=mean_mrr,
            mean_precision=mean_precision,
            mean_latency_ms=mean_latency_ms,
            verdict=verdict,
            failure_reasons=failure_reasons,
            slice_metrics=slice_metrics,
            per_query=per_query,
        )

    @staticmethod
    def _aggregate_slices(successes: Sequence[PerQueryMetric]) -> Dict[str, Dict[str, float]]:
        """Group successes by difficulty and edge_type; compute per-slice means."""
        grouped: Dict[str, List[PerQueryMetric]] = {}
        for m in successes:
            grouped.setdefault(f"difficulty={m.difficulty}", []).append(m)
            grouped.setdefault(f"edge_type={m.edge_type}", []).append(m)
        out: Dict[str, Dict[str, float]] = {}
        for key, group in grouped.items():
            n = len(group)
            out[key] = {
                'count': float(n),
                'mean_ndcg': sum(m.ndcg for m in group) / n,
                'mean_recall': sum(m.recall for m in group) / n,
                'mean_mrr': sum(m.mrr for m in group) / n,
                'mean_precision': sum(m.precision for m in group) / n,
            }
        return out

    def _evaluate_release_gate(self, mean_ndcg: float, mean_recall: float, mean_mrr: float, error_count: int, total: int, failure_reasons: List[str]) -> str:
        """All thresholds must pass; error rate must stay under config cap."""
        passes = True
        if mean_ndcg < self._config.min_mean_ndcg:
            failure_reasons.append(f"mean_ndcg_below_threshold actual={mean_ndcg:.4f} min={self._config.min_mean_ndcg:.4f}")
            passes = False
        if mean_recall < self._config.min_mean_recall:
            failure_reasons.append(f"mean_recall_below_threshold actual={mean_recall:.4f} min={self._config.min_mean_recall:.4f}")
            passes = False
        if mean_mrr < self._config.min_mean_mrr:
            failure_reasons.append(f"mean_mrr_below_threshold actual={mean_mrr:.4f} min={self._config.min_mean_mrr:.4f}")
            passes = False
        error_rate = error_count / total if total > 0 else 0.0
        if error_rate > self._config.max_error_rate:
            failure_reasons.append(
                f"error_rate_above_threshold actual={error_rate:.4f} max={self._config.max_error_rate:.4f}"
            )
            passes = False
        return 'passes_release_gate' if passes else 'fails_release_gate'
