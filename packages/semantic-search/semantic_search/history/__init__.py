"""User search history layer — repeat-query save-trigger heuristic and the
offline 90-day → feature-vector compactor that feeds the personalized landing
rail. Per-user opt-out is honored at the write boundary; right-to-delete clears
both raw rows and the aggregated vector.
"""
from semantic_search.history.compactor import CompactionCycleSummary, CompactorDriverSummary, HistoryCompactor, HistoryCompactorDriver, InMemoryUserFeatureVectorStore
from semantic_search.history.compactor import UserFeatureVector, UserFeatureVectorStore
from semantic_search.history.store import UserSearchHistoryStore

__all__ = [
    'CompactionCycleSummary',
    'CompactorDriverSummary',
    'HistoryCompactor',
    'HistoryCompactorDriver',
    'InMemoryUserFeatureVectorStore',
    'UserFeatureVector',
    'UserFeatureVectorStore',
    'UserSearchHistoryStore',
]
