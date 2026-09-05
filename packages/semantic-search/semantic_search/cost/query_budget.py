"""Per-query LLM cost budget.

A ``QueryCostBudget`` accumulates ``cost_usd`` across every LLM call
issued during the lifetime of a single ``SearchOrchestrator.search()``
invocation. When a recorded cost would push the running total above
``max_cost_usd_per_query``, the accumulator raises
``QueryCostBudgetExceeded`` IMMEDIATELY so the caller can degrade.

Design contract:

- One instance per ``search()`` call. The orchestrator constructs it on
  entry and drops the reference on return — there is intentionally no
  shared cross-request state. (Pipeline / per-tenant budgets live in a
  different layer, out of scope.)
- Pure compute. No I/O. Thread-safe via an internal lock so a budget
  shared with concurrent LLM calls (e.g. ``asyncio.gather`` of
  parallel LLM sub-calls) sees a consistent view.
- Direction: cost is monotonically non-decreasing. Negative records are
  rejected with ``ValidationError`` so a buggy callback can never
  silently cancel an over-budget call.
- Fail-closed at the boundary: when ``max_cost_usd_per_query <= 0`` the
  budget is treated as DISABLED and ``record_cost`` is a no-op
  (``running_total`` still accumulates for observability — the
  enforcement just never trips). Pass a ``NoOpQueryCostBudget`` to
  make the disabled state structurally explicit at the call site.
"""
import threading
from typing import Optional

from semantic_search.core.exceptions import QueryCostBudgetExceeded, ValidationError
from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)


class QueryCostBudget:
    """Track LLM USD/query; raise on exceed (must be > 0; use NoOpQueryCostBudget to disable)."""

    def __init__(self, max_cost_usd_per_query: float, request_id: Optional[str] = None):
        if not isinstance(max_cost_usd_per_query, (int, float)):
            raise ValidationError(
                f"QueryCostBudget.max_cost_usd_per_query must be numeric; got {type(max_cost_usd_per_query).__name__}"
            )
        if float(max_cost_usd_per_query) <= 0.0:
            raise ValidationError(
                "QueryCostBudget.max_cost_usd_per_query must be > 0.0; use NoOpQueryCostBudget for disabled budgets"
            )
        self._max = float(max_cost_usd_per_query)
        self._request_id = str(request_id) if request_id is not None else None
        self._lock = threading.Lock()
        self._total_usd = 0.0
        self._call_count = 0

    @property
    def max_cost_usd(self) -> float:
        """Configured per-query ceiling (USD)."""
        return self._max

    @property
    def request_id(self) -> Optional[str]:
        """Correlation id for log breadcrumbs."""
        return self._request_id

    @property
    def running_total_usd(self) -> float:
        """Cumulative USD recorded so far during the lifetime of this budget."""
        with self._lock:
            return self._total_usd

    @property
    def call_count(self) -> int:
        """Number of cost records observed so far."""
        with self._lock:
            return self._call_count

    @property
    def is_enabled(self) -> bool:
        """True iff this budget enforces a cap (always True for the live class)."""
        return True

    def check_admit(self) -> None:
        """Raise when this request already exhausted its per-query cap (block next LLM)."""
        with self._lock:
            total = float(self._total_usd)
            calls = int(self._call_count)
        if total >= self._max:
            request_breadcrumb = f"request_id={self._request_id} " if self._request_id else ""
            logger.warning(
                f"query_cost_budget_admit_denied {request_breadcrumb}"
                f"running_total_usd={total:.6f} max_cost_usd={self._max:.6f} call_count={calls}"
            )
            raise QueryCostBudgetExceeded(
                f"per-query LLM cost budget exhausted: running_total_usd={total:.6f} "
                f"max_cost_usd={self._max:.6f} after {calls} call(s)"
            )

    def record_cost(self, cost_usd: float) -> None:
        """Add cost; raise QueryCostBudgetExceeded if exceeds cap (rejects negative)."""
        if not isinstance(cost_usd, (int, float)):
            raise ValidationError(
                f"QueryCostBudget.record_cost requires numeric cost_usd; got {type(cost_usd).__name__}"
            )
        if float(cost_usd) < 0.0:
            raise ValidationError(
                f"QueryCostBudget.record_cost rejects negative cost_usd; got {cost_usd}"
            )
        with self._lock:
            cumulative_total = self._total_usd + float(cost_usd)
            self._call_count += 1
            self._total_usd = cumulative_total
        if cumulative_total > self._max:
            request_breadcrumb = f"request_id={self._request_id} " if self._request_id else ""
            logger.warning(
                f"query_cost_budget_exceeded {request_breadcrumb}"
                f"running_total_usd={cumulative_total:.6f} max_cost_usd={self._max:.6f} "
                f"call_count={self._call_count} last_call_usd={float(cost_usd):.6f}"
            )
            raise QueryCostBudgetExceeded(f"per-query LLM cost budget exceeded: running_total_usd={cumulative_total:.6f} max_cost_usd={self._max:.6f} after {self._call_count} call(s)")

    def snapshot(self) -> dict:
        """Audit snapshot (thread-safe): max, running_total, call_count, request_id."""
        with self._lock:
            return {
                'max_cost_usd_per_query': float(self._max),
                'running_total_usd': float(self._total_usd),
                'call_count': int(self._call_count),
                'request_id': self._request_id,
            }


class NoOpQueryCostBudget:
    """No-op budget (no enforcement, tracking only; is_enabled=False)."""

    def __init__(self, request_id: Optional[str] = None):
        self._request_id = str(request_id) if request_id is not None else None
        self._lock = threading.Lock()
        self._total_usd = 0.0
        self._call_count = 0

    @property
    def max_cost_usd(self) -> float:
        """Sentinel value (0.0) — never enforced by this class."""
        return 0.0

    @property
    def request_id(self) -> Optional[str]:
        return self._request_id

    @property
    def running_total_usd(self) -> float:
        with self._lock:
            return self._total_usd

    @property
    def call_count(self) -> int:
        with self._lock:
            return self._call_count

    @property
    def is_enabled(self) -> bool:
        return False

    def record_cost(self, cost_usd: float) -> None:
        if not isinstance(cost_usd, (int, float)):
            raise ValidationError(
                f"NoOpQueryCostBudget.record_cost requires numeric cost_usd; got {type(cost_usd).__name__}"
            )
        if float(cost_usd) < 0.0:
            raise ValidationError(
                f"NoOpQueryCostBudget.record_cost rejects negative cost_usd; got {cost_usd}"
            )
        with self._lock:
            self._total_usd += float(cost_usd)
            self._call_count += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                'max_cost_usd_per_query': 0.0,
                'running_total_usd': float(self._total_usd),
                'call_count': int(self._call_count),
                'request_id': self._request_id,
                'enforcement_disabled': True,
            }


__all__ = ['QueryCostBudget', 'NoOpQueryCostBudget']
