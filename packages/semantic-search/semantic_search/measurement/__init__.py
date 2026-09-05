"""Measurement subsystem — proxy-signal capture without a search funnel.

Collaborators:
- ``MeasurementStore``: thread-safe in-memory rolling window of
  ``SearchObservation``.
- ``ProxySignalEvaluator``: computes the proxy-signal table on demand.
- ``QieOnlyLaunchStats``: Phase 1 ``qie_only`` launch gate rates
  (``GET /measurement/qie_only``).

Storage is in-memory only by design. Restarts drop the window; the
dashboard recovers as new observations accumulate. No PII is recorded —
only the structured facts the evaluator needs (``query_type``,
``decision_tier``, ``confidence``, ``result_count``, ``cache_hit``,
``latency``, ``decision_cost``). The raw query text is never stored.
"""
from semantic_search.measurement.evaluator import ProxySignalEvaluator
from semantic_search.measurement.qie_only_launch import (
    QieOnlyLaunchStats,
    get_qie_only_launch_stats,
)
from semantic_search.measurement.store import MeasurementStore

__all__ = [
    'MeasurementStore',
    'ProxySignalEvaluator',
    'QieOnlyLaunchStats',
    'get_qie_only_launch_stats',
]
