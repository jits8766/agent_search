"""Explore subsystem — trending / ending-soon rails (+ fallback).

Two consumers:
  1. /landing-rail/{user_id} — the public landing surface
  2. SearchOrchestrator      — the zero-result fallback rail

Both reach the same `ExploreComposer`; the response carries a `source` field so
dashboards keep the two surfaces separate.
"""
from semantic_search.explore.composer import ExploreComposer
from semantic_search.explore.sources import EndingSoonSource, ExploreSource, InMemoryEndingSoonSource, InMemoryTrendingSource, TrendingSource
from semantic_search.explore.zero_result_guard import ZeroResultGuard, ZeroResultGuardResult

__all__ = [
    'ExploreComposer',
    'ExploreSource',
    'TrendingSource',
    'EndingSoonSource',
    'InMemoryTrendingSource',
    'InMemoryEndingSoonSource',
    'ZeroResultGuard',
    'ZeroResultGuardResult',
]
