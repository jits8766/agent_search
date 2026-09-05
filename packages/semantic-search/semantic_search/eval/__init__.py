"""Offline retrieval-quality evaluation framework.

Provides standard ranking-quality metrics (NDCG@k / Recall@k / MRR@k /
Precision@k) and a runner that scores ``SearchOrchestrator.search()`` output
against a labeled ``RelevanceJudgedQuery`` set. Library-only — no HTTP surface
(retrieval-quality eval is an internal control-loop signal, not external
observability).
"""
from semantic_search.eval.retrieval_metrics import dcg_at_k, mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k
from semantic_search.eval.retrieval_quality import PerQueryMetric, RetrievalEvalReport, RetrievalQualityEvaluator
from semantic_search.eval.llm_relevance_judge import LLMJudgedQueryBuilder, LLMRelevanceJudge, RelevanceGradeOutput

__all__ = [
    'dcg_at_k',
    'mrr_at_k',
    'ndcg_at_k',
    'precision_at_k',
    'recall_at_k',
    'PerQueryMetric',
    'RetrievalEvalReport',
    'RetrievalQualityEvaluator',
    'LLMJudgedQueryBuilder',
    'LLMRelevanceJudge',
    'RelevanceGradeOutput',
]
