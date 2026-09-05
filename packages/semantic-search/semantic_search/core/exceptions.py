"""Exception hierarchy for semantic_search package.
All modules import exceptions from here for consistency.
"""


class AgentSearchError(Exception):
    """Base exception for all semantic_search errors."""
    pass


class ConfigurationError(AgentSearchError):
    """Raised when required configuration is missing or invalid."""
    pass


class ValidationError(AgentSearchError):
    """Raised when input validation fails."""
    pass


class QueryIntelligenceError(AgentSearchError):
    """Raised when query intent classification, decomposition, or grounding fails."""
    pass


class RetrievalError(AgentSearchError):
    """Raised when a retrieval backend (vector / structured / SQL) fails."""
    pass


class DataIngestInterruptedError(AgentSearchError):
    """Raised when data-build / seed ingest stops before completion.

    Typical causes: ALB idle timeout, client disconnect, task cancel, backend
    failure mid-batch. Carries progress so the API can report how far ingest
    got and why it stopped (not an opaque Internal Server Error).
    """

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        records_completed: int,
        records_attempted: int = 0,
        reason: str = 'interrupted',
        detail: str = '',
    ) -> None:
        super().__init__(message)
        self.stage = str(stage or 'unknown')
        self.records_completed = int(records_completed)
        self.records_attempted = int(records_attempted)
        self.reason = str(reason or 'interrupted')
        self.detail = str(detail or '')


class QdrantUnavailableError(RetrievalError):
    """Raised when the Qdrant client is not reachable (network, auth, or import).

    Construction of `QdrantClientFactory` never raises; methods on the Qdrant
    adapters raise this when invoked against an unavailable client so the
    `BackendHealthRegistry` + `DegradationPlanner` can fall back deterministically.
    """
    pass


class QdrantQueryError(RetrievalError):
    """Raised when Qdrant returns a query-level error (bad filter, collection missing, timeout)."""
    pass


class CacheError(AgentSearchError):
    """Raised when cache read/write operations fail."""
    pass


class LLMError(AgentSearchError):
    """Raised when an LLM call fails (call, parse, schema validation)."""
    pass


class LLMRefusalError(LLMError):
    """Raised when the LLM structurally refuses to answer.

    Distinct from a generic ``LLMError`` because the model committed to a
    ``kind='refuse'`` branch — there is no fallback chain that will produce
    a different answer, and the right product surface is to ask the user to
    rephrase / clarify rather than retry. Carries the model's structured
    ``reason`` and ``suggested_clarification`` for the UI.
    """

    def __init__(self, message: str, reason: str = '', suggested_clarification: str = ''):
        super().__init__(message)
        self.reason = reason
        self.suggested_clarification = suggested_clarification


class HistoryError(AgentSearchError):
    """Raised when the user search-history store rejects an operation."""
    pass


class RerankerError(AgentSearchError):
    """Raised when a reranker backend (lexical / model-backed / no-op) is misconfigured.

    The reranker is latency-gated by the orchestrator (``asyncio.wait_for`` +
    ``try/except Exception``), so a runtime ``RerankerError`` from a backend
    will be caught and the deterministic-ranker order preserved. This class
    exists so misconfiguration vs. runtime-fallback can be distinguished in
    logs and alarmed on separately.
    """
    pass


class DiversityError(AgentSearchError):
    """Raised when a diversifier backend (lexical-jaccard MMR / no-op / future model-backed) is misconfigured.

    Mirrors ``RerankerError`` semantics: the diversifier is latency-gated by
    the orchestrator, so a runtime ``DiversityError`` from a backend is
    caught and the post-eRanker order is preserved. This class
    distinguishes misconfiguration (backend cannot serve requests at all)
    from runtime-fallback (backend served but returned an unusable result)
    in logs + alarm thresholds.
    """
    pass


class CostBudgetExceeded(AgentSearchError):
    """Base for LLM spend-policy breaches (per-query or fleet).

    Distinct from ``LLMError`` — transport/parse succeeded or the call was
    blocked before provider I/O by admit control. Orchestrator fail-soft
    paths catch this base class so both query and fleet caps degrade the same way.
    """
    pass


class QueryCostBudgetExceeded(CostBudgetExceeded):
    """Raised when the per-query LLM cost budget is exhausted mid-request.

    The ``QueryCostBudget`` accumulates ``cost_usd`` across every LLM call
    issued during a single ``SearchOrchestrator.search()`` invocation. When
    a recorded cost would push the running total above the configured
    ``max_cost_usd_per_query``, this exception is raised IMMEDIATELY (before
    the costly call's result is consumed) so the caller can decide to:

    - drop to a degraded path (return cached / partial results), OR
    - surface a typed 503 to the user, OR
    - log + alert that a single query saturated the per-request budget.

    Distinct from ``LLMError`` because the LLM call itself succeeded —
    the failure is policy, not transport. Distinct from ``ConfigurationError``
    because the cap is per-request, not boot-time.
    """
    pass


class FleetCostBudgetExceeded(CostBudgetExceeded):
    """Raised when the fleet (hour/day) LLM cost budget is exhausted.

    Fired by ``FleetCostBudget.check_admit`` (before provider I/O) or by
    ``record_cost`` after a call tips the shared window. Cross-request;
    complements ``QueryCostBudgetExceeded``.
    """
    pass


class EgressGuardError(AgentSearchError):
    """Raised when the output-side egress guard is misconfigured or a backend fails.

    Unlike ``RerankerError`` / ``DiversityError`` this class is NOT latency-gated
    away by the orchestrator: the egress guard is a SAFETY gate, so a
    construction-time ``EgressGuardError`` is fatal (the orchestrator will
    refuse to start if the guard cannot be built) and a per-request
    runtime ``EgressGuardError`` is caught and converted to a
    "fail-closed-on-this-item" decision (drop the item rather than leak
    an unscrubbed payload). Callers MUST distinguish this class from
    ``ValidationError`` / ``RetrievalError`` so SRE alarms on guard failures
    fire on a separate channel from upstream-pipeline failures.
    """
    pass
