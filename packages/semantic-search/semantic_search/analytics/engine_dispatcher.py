"""Keyword-driven dispatch from analytics questions to DomainAnalyticsEngine methods.

Routes a natural-language analytics question to the appropriate
DomainAnalyticsEngine method using config-driven keyword matching.
All routing rules and method parameters live in YAML — no values are
hardcoded in this module.

On any failure (no match, engine error, timeout) the dispatcher returns
None so the caller falls through to the cache + NL-to-SQL pipeline.
"""
import asyncio
import inspect
import time
from typing import Any, Dict, List, Optional, Tuple

from semantic_search.analytics.domain_analytics_engine import DomainAnalyticsEngine
from semantic_search.analytics.filter_bridge import sql_hint_to_filters
from semantic_search.analytics.query_templates import TimeGrain
from semantic_search.config.analytics_models import EngineDispatchConfig
from semantic_search.core.exceptions import RetrievalError
from semantic_search.core.logging_utils import get_logger
from semantic_search.nl_to_sql.contracts import AnalyticsResult, SqlExecutionResult

logger = get_logger(__name__)

_GRAIN_MAP: Dict[str, TimeGrain] = {
    'hour': TimeGrain.HOUR,
    'day': TimeGrain.DAY,
    'week': TimeGrain.WEEK,
    'month': TimeGrain.MONTH,
}


def _coerce_params(raw: Dict[str, Any]) -> Dict[str, Any]:
    """YAML params → typed values (grain string → TimeGrain enum, else passthrough)."""
    out: Dict[str, Any] = {}
    for k, v in raw.items():
        if k == 'grain' and isinstance(v, str):
            grain = _GRAIN_MAP.get(v.lower())
            if grain is None:
                raise ValueError(f"engine_dispatch: unknown grain value {v!r}; valid: {sorted(_GRAIN_MAP)}")
            out[k] = grain
        else:
            out[k] = v
    return out


class AnalyticsEngineDispatcher:
    """Route analytics questions → DomainAnalyticsEngine methods (config-driven keyword match)."""

    def __init__(self, engine: DomainAnalyticsEngine, config: EngineDispatchConfig) -> None:
        if engine is None:
            raise ValueError("AnalyticsEngineDispatcher requires a DomainAnalyticsEngine")
        if config is None:
            raise ValueError("AnalyticsEngineDispatcher requires an EngineDispatchConfig")
        self._engine = engine
        self._config = config
        # Pre-coerce params/keywords at construction time (dispatch = dict lookup + scan)
        self._routes: List[Tuple[str, List[str], Dict[str, Any]]] = []
        for route in config.routes:
            try:
                coerced = _coerce_params(route.params)
                keywords = [k.lower() for k in route.keywords]
                self._routes.append((route.method, keywords, coerced))
            except Exception as e:
                raise ValueError(
                    f"AnalyticsEngineDispatcher: bad route config for method={route.method}: {e}"
                ) from e

    @property
    def enabled(self) -> bool:
        """True iff the dispatcher is toggled on in config."""
        return bool(self._config.enabled)

    def _match(self, question: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Return (method_name, coerced_params) for the first matching route, or None."""
        q = question.lower()
        for method_name, keywords, params in self._routes:
            if any(kw in q for kw in keywords):
                return method_name, params
        return None

    async def try_dispatch(
        self,
        question: str,
        sql_hint: str,
        request_id: str,
        t0: float,
    ) -> Optional[AnalyticsResult]:
        """Attempt keyword-match dispatch to the appropriate engine method.

        Returns a typed AnalyticsResult on success (analytics_substrate='domain_engine').
        Returns None on: no keyword match, engine disabled, CH failure, or timeout.
        The caller must fall through to the cache + NL-to-SQL pipeline on None.
        Never raises.

        :param question: str - Natural-language analytics question
        :param sql_hint: str - Structured hint from QI (passed through to result)
        :param request_id: str - Correlation ID
        :param t0: float - Wall-clock start from time.monotonic() (for latency accounting)
        :return: Optional[AnalyticsResult]
        """
        if not self.enabled:
            return None

        match = self._match(question)
        if match is None:
            return None

        method_name, params = match
        method = getattr(self._engine, method_name, None)
        if method is None or not callable(method):
            logger.warning(
                f"analytics_engine_dispatch_unknown_method request_id={request_id} method={method_name}"
            )
            return None

        logger.info(
            f"analytics_engine_dispatch_attempt request_id={request_id} method={method_name}"
        )

        call_params = dict(params)
        if (
            self._config.filter_bridge is not None
            and sql_hint
            and 'filters' in inspect.signature(method).parameters
        ):
            bridge_filters = sql_hint_to_filters(
                sql_hint,
                self._config.filter_bridge,
                self._engine._cfg,
            )
            if bridge_filters:
                existing = call_params.get('filters') or []
                call_params['filters'] = list(existing) + bridge_filters
                logger.debug(
                    f"analytics_engine_dispatch_bridge request_id={request_id} "
                    f"method={method_name} injected_filters={len(bridge_filters)}"
                )

        try:
            rows = await asyncio.wait_for(
                method(**call_params),
                timeout=self._config.timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"analytics_engine_dispatch_timeout request_id={request_id} "
                f"method={method_name} timeout_s={self._config.timeout_seconds}"
            )
            return None
        except RetrievalError as exc:
            logger.warning(
                f"analytics_engine_dispatch_retrieval_error request_id={request_id} "
                f"method={method_name} error={exc}"
            )
            return None
        except Exception as exc:
            logger.warning(
                f"analytics_engine_dispatch_error request_id={request_id} "
                f"method={method_name} error_type={type(exc).__name__} error={exc}"
            )
            return None

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        row_list = list(rows) if rows is not None else []
        column_names = list(row_list[0].keys()) if row_list else []

        execution = SqlExecutionResult(
            sql=f'-- domain_engine:{method_name}',
            rows=row_list,
            column_names=column_names,
            row_count=len(row_list),
            latency_ms=elapsed_ms,
            truncated=False,
        )

        logger.info(
            f"analytics_engine_dispatch_success request_id={request_id} "
            f"method={method_name} rows={len(row_list)} latency_ms={elapsed_ms:.1f}"
        )

        return AnalyticsResult(
            request_id=request_id,
            question=question,
            sql_hint=sql_hint,
            success=True,
            failure_mode=None,
            failure_reason='',
            pruned_schema=None,
            generation=None,
            validation=None,
            execution=execution,
            verifier=None,
            total_latency_ms=elapsed_ms,
            analytics_substrate='domain_engine',
            as_of=time.time(),
        )


__all__ = ['AnalyticsEngineDispatcher']
