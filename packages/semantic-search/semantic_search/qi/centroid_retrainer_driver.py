"""Background asyncio driver for L1 centroid retrain cycles.

Polls SignalStore for labeled positives, builds a CentroidRetrainCandidate,
runs shadow-agreement verdict, and applies 'promote' verdicts via
SemanticRouter.swap_centroids. Lifecycle managed by the FastAPI lifespan.
"""
import asyncio, uuid
from typing import Dict, List, Optional
import numpy as np
from semantic_search.config.models import CentroidRetrainerDriverConfig
from semantic_search.contracts import CentroidRetrainerCycleSummary, CENTROID_RETRAIN_VERDICTS, FeedbackSignal
from semantic_search.core.exceptions import AgentSearchError, ValidationError
from semantic_search.core.logging_utils import get_logger
from semantic_search.qi.centroid_retrainer import CentroidRetrainer
from semantic_search.qi.semantic_router import SemanticRouter
from semantic_search.signal_store import SignalStore, schedule_feedback_signal_record

logger = get_logger(__name__)
_MAX_CYCLE_HISTORY = 200


class CentroidRetrainerDriver:
    """Asyncio background driver that closes the feedback loop between user signals and L1 centroids.

    Each cycle: reads SignalStore → extracts labeled positives → build_candidate → decide → swap_centroids on promote.
    start() / stop() are called by the FastAPI lifespan. run_cycle() is the single-cycle entry point for tests and ops.

    :param config: CentroidRetrainerDriverConfig
    :param retrainer: CentroidRetrainer - stateless candidate + verdict service
    :param router: SemanticRouter - live router; swap_centroids called on promote verdict
    :param signal_store: SignalStore - source of labeled positives and shadow probes
    """

    def __init__(self, config: CentroidRetrainerDriverConfig, retrainer: CentroidRetrainer, router: SemanticRouter, signal_store: SignalStore) -> None:
        if config is None:
            raise ValidationError("CentroidRetrainerDriver requires config")
        if retrainer is None:
            raise ValidationError("CentroidRetrainerDriver requires retrainer")
        if router is None:
            raise ValidationError("CentroidRetrainerDriver requires router")
        if signal_store is None:
            raise ValidationError("CentroidRetrainerDriver requires signal_store")
        self._config = config
        self._retrainer = retrainer
        self._router = router
        self._signal_store = signal_store
        self._task: Optional[asyncio.Task] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._cycle_counter: int = 0
        self._consecutive_failures: int = 0
        self._history: List[CentroidRetrainerCycleSummary] = []
        self._positive_types: frozenset = frozenset(config.positive_signal_types)
        logger.info(f"centroid_retrainer_driver_initialized interval_seconds={config.interval_seconds} shadow_query_limit={config.shadow_query_limit} positive_signal_types={sorted(self._positive_types)}")

    @property
    def is_running(self) -> bool:
        """True iff the background loop task is alive."""
        return self._task is not None and not self._task.done()

    @property
    def cycle_history(self) -> List[CentroidRetrainerCycleSummary]:
        """Snapshot of cycle summaries (newest at tail, capped at _MAX_CYCLE_HISTORY)."""
        return list(self._history)

    async def start(self) -> None:
        """Start the background loop. Idempotent — second call while running is a no-op."""
        if not self._config.enabled:
            logger.info("centroid_retrainer_driver_start_skipped reason=disabled")
            return
        if self.is_running:
            logger.info("centroid_retrainer_driver_start_skipped reason=already_running")
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.get_event_loop().create_task(self._loop_body())
        logger.info(f"centroid_retrainer_driver_started interval_seconds={self._config.interval_seconds}")

    async def stop(self) -> None:
        """Signal the background loop to stop and await its completion (10s timeout)."""
        if not self.is_running or self._stop_event is None:
            return
        self._stop_event.set()
        try:
            await asyncio.wait_for(self._task, timeout=10.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            self._task.cancel()
        self._task = None
        self._stop_event = None
        logger.info("centroid_retrainer_driver_stopped")

    async def run_cycle(self) -> CentroidRetrainerCycleSummary:
        """Run one retrain cycle synchronously. Used by tests and manual ops triggers."""
        self._cycle_counter += 1
        cycle_id = f"crd_{uuid.uuid4().hex[:12]}"
        ran_at = asyncio.get_event_loop().time()
        try:
            summary = await self._run_cycle_inner(cycle_id, ran_at)
        except Exception as exc:
            summary = CentroidRetrainerCycleSummary(cycle_id=cycle_id, ran_at=ran_at, skipped=False, skip_reason=None, candidate_id=None, verdict=None, shadow_agreement_rate=None, promoted=False, error=f"{type(exc).__name__}: {exc}")
            logger.warning(f"centroid_retrainer_driver_cycle_error cycle_id={cycle_id} error_type={type(exc).__name__} error={exc}")
        if len(self._history) >= _MAX_CYCLE_HISTORY:
            self._history.pop(0)
        self._history.append(summary)
        return summary

    async def _run_cycle_inner(self, cycle_id: str, ran_at: float) -> CentroidRetrainerCycleSummary:
        signals = await asyncio.to_thread(self._signal_store.recent, self._retrainer.config.max_signals_per_read)
        positives_by_archetype: Dict[str, List[str]] = {}
        shadow_queries: List[str] = []
        for sig in signals:
            payload = sig.payload if isinstance(sig.payload, dict) else {}
            qt = payload.get('query_type')
            qt_text = str(payload.get('query_text') or '').strip()
            if qt_text and len(shadow_queries) < self._config.shadow_query_limit:
                shadow_queries.append(qt_text)
            if sig.signal_type in self._positive_types and qt and qt_text:
                positives_by_archetype.setdefault(qt, []).append(qt_text)
        if not positives_by_archetype:
            return CentroidRetrainerCycleSummary(cycle_id=cycle_id, ran_at=ran_at, skipped=True, skip_reason='no_labeled_positives', candidate_id=None, verdict=None, shadow_agreement_rate=None, promoted=False, error=None)
        try:
            candidate = self._retrainer.build_candidate(positives_by_archetype, candidate_id=cycle_id)
        except ValidationError as exc:
            return CentroidRetrainerCycleSummary(cycle_id=cycle_id, ran_at=ran_at, skipped=True, skip_reason=f"build_candidate_rejected: {exc}", candidate_id=None, verdict=None, shadow_agreement_rate=None, promoted=False, error=None)
        verdict = self._retrainer.decide(candidate, shadow_queries)
        schedule_feedback_signal_record(self._signal_store, FeedbackSignal(signal_id=FeedbackSignal.new_signal_id(), request_id=cycle_id, signal_type='centroid_retrain_verdict', payload={'verdict': verdict.verdict, 'candidate_id': candidate.candidate_id, 'shadow_agreement_rate': verdict.shadow_agreement_rate}, signal_origin='centroid_retrainer_driver'))
        promoted = False
        if verdict.verdict == 'promote':
            # CentroidRetrainCandidate.new_centroids is Dict[str, List[float]] — a single L2-normalised centroid per archetype.
            # Wrap as (1, dim) ndarray so swap_centroids receives the (k, dim) shape SemanticRouter._score_centroid expects.
            new_np = {arch: np.array(c, dtype=np.float32)[np.newaxis, :] for arch, c in candidate.new_centroids.items()}
            self._router.swap_centroids(new_np)
            schedule_feedback_signal_record(self._signal_store, FeedbackSignal(signal_id=FeedbackSignal.new_signal_id(), request_id=cycle_id, signal_type='centroid_retrain_promoted', payload={'candidate_id': candidate.candidate_id, 'archetypes': sorted(new_np.keys())}, signal_origin='centroid_retrainer_driver'))
            promoted = True
        return CentroidRetrainerCycleSummary(cycle_id=cycle_id, ran_at=ran_at, skipped=False, skip_reason=None, candidate_id=candidate.candidate_id, verdict=verdict.verdict, shadow_agreement_rate=verdict.shadow_agreement_rate, promoted=promoted, error=None)

    async def _loop_body(self) -> None:
        while not self._stop_event.is_set():
            summary = await self.run_cycle()
            if summary.error is not None:
                self._consecutive_failures += 1
            else:
                self._consecutive_failures = 0
            backoff = min(self._consecutive_failures, self._config.max_consecutive_failures)
            sleep_seconds = self._config.interval_seconds * (2 ** backoff if backoff > 0 else 1)
            try:
                await asyncio.wait_for(asyncio.shield(self._stop_event.wait()), timeout=sleep_seconds)
                break
            except asyncio.TimeoutError:
                pass
