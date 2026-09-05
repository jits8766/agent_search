"""Parallel logic validator (stage 4).

Runs lightweight pre-execution checks in parallel via `asyncio.gather`:

- syntax_check: re-parse with sqlglot (defensive — security parsed already).
- join_correctness: every JOIN must have an ON clause (no implicit cross-join).
- cardinality_estimate: parse LIMIT and report it (or fall back to the configured
  max_estimated_rows ceiling); if it exceeds max_estimated_rows, fail.
- explain_probe: optional EXPLAIN-based probe wired through a caller-supplied
  callback (kept pluggable so unit tests don't require Athena).

The validator NEVER mutates SQL — clamping is the security validator's job.
"""
import asyncio
import time
from typing import Awaitable, Callable, List, Optional, Tuple

from semantic_search.config.nl_to_sql_models import SqlLogicValidationConfig
from semantic_search.core.exceptions import ValidationError as AgentValidationError
from semantic_search.core.logging_utils import get_logger
from semantic_search.nl_to_sql.contracts import SqlValidationResult
from semantic_search.nl_to_sql.cost_class import CostClassifier

from sqlglot import exp as _sg_exp, parse_one as _sg_parse_one
from sqlglot.errors import ParseError as _SgParseError

logger = get_logger(__name__)

ExplainProbe = Callable[[str], Awaitable[Tuple[bool, Optional[int], str]]]
"""Optional EXPLAIN probe: takes SQL → returns (ok, estimated_rows, reason)."""


class LogicValidator:
    """Stage-4 logic validator. Composes parallel checks via asyncio.gather.

    :param config: SqlLogicValidationConfig - Toggles + cardinality cap
    :param dialect: str - sqlglot dialect
    :param explain_probe: Optional[ExplainProbe] - Optional EXPLAIN backend; None disables probe
    :param cost_classifier: Optional[CostClassifier] - When supplied, the
        validator stamps a ``cost_class`` onto the resulting
        ``SqlValidationResult`` based on the cardinality estimate. Bytes
        signals are not yet plumbed end-to-end, so the row estimate alone
        drives the verdict for now.
    """

    def __init__(self, config: SqlLogicValidationConfig, dialect: str, explain_probe: Optional[ExplainProbe] = None, cost_classifier: Optional[CostClassifier] = None):
        if not isinstance(config, SqlLogicValidationConfig):
            raise AgentValidationError("LogicValidator requires a SqlLogicValidationConfig")
        if not isinstance(dialect, str) or not dialect:
            raise AgentValidationError("LogicValidator requires a non-empty dialect")
        if cost_classifier is not None and not isinstance(cost_classifier, CostClassifier):
            raise AgentValidationError("LogicValidator.cost_classifier must be a CostClassifier")
        self._config = config
        self._dialect = dialect
        self._probe = explain_probe
        self._cost_classifier = cost_classifier

    async def validate(self, sql: str) -> SqlValidationResult:
        """Run all enabled checks in parallel and aggregate the verdict.

        :param sql: str - SQL after security clamp
        :return: SqlValidationResult - is_valid + failure_mode (when enabled=False, returns is_valid=True trivially)
        """
        t0 = time.monotonic()
        if not self._config.enabled:
            return SqlValidationResult(
                is_valid=True,
                sql=sql,
                failure_mode=None,
                failure_reasons=[],
                mutations=[],
                estimated_rows=None,
                latency_ms=(time.monotonic() - t0) * 1000.0,
            )
        coros: List[Awaitable] = [
            self._syntax_check(sql),
            self._join_correctness(sql),
            self._cardinality_check(sql),
        ]
        if self._config.enable_explain_probe and self._probe is not None:
            coros.append(self._probe_check(sql))
        results = await asyncio.gather(*coros, return_exceptions=True)
        failures: List[str] = []
        failure_mode: Optional[str] = None
        estimated_rows: Optional[int] = None
        for label, result in zip(
            ('syntax', 'join_correctness', 'cardinality', 'explain_probe'),
            results,
        ):
            if isinstance(result, BaseException):
                logger.warning(f"logic_validator_check_errored check={label} error_type={type(result).__name__}")
                failures.append(f"{label}: {type(result).__name__}: {result}")
                if failure_mode is None:
                    failure_mode = label if label in {'syntax', 'join_correctness', 'cardinality'} else 'unknown'
                continue
            ok, est_rows, reason = result
            if est_rows is not None and (estimated_rows is None or est_rows > estimated_rows):
                estimated_rows = est_rows
            if not ok:
                failures.append(f"{label}: {reason}")
                if failure_mode is None:
                    failure_mode = label if label in {'syntax', 'join_correctness', 'cardinality'} else 'unknown'
        latency_ms = (time.monotonic() - t0) * 1000.0
        cost_class: Optional[str] = None
        if self._cost_classifier is not None:
            cost_class = self._cost_classifier.classify(
                estimated_bytes=None,
                estimated_rows=estimated_rows,
            )
        if failures:
            logger.warning(f"logic_validation_failed reasons={failures} latency_ms={latency_ms:.1f}")
            return SqlValidationResult(
                is_valid=False,
                sql=sql,
                failure_mode=failure_mode or 'unknown',
                failure_reasons=failures,
                mutations=[],
                estimated_rows=estimated_rows,
                latency_ms=latency_ms,
                cost_class=cost_class,
            )
        logger.info(
            f"logic_validation_ok estimated_rows={estimated_rows} "
            f"cost_class={cost_class} latency_ms={latency_ms:.1f}"
        )
        return SqlValidationResult(
            is_valid=True,
            sql=sql,
            failure_mode=None,
            failure_reasons=[],
            mutations=[],
            estimated_rows=estimated_rows,
            latency_ms=latency_ms,
            cost_class=cost_class,
        )

    async def _syntax_check(self, sql: str) -> Tuple[bool, Optional[int], str]:
        def _run() -> Tuple[bool, Optional[int], str]:
            try:
                _sg_parse_one(sql, dialect=self._dialect)
                return True, None, "ok"
            except _SgParseError as e:
                return False, None, f"sqlglot parse error: {e}"
        return await asyncio.to_thread(_run)

    async def _join_correctness(self, sql: str) -> Tuple[bool, Optional[int], str]:
        def _run() -> Tuple[bool, Optional[int], str]:
            try:
                ast = _sg_parse_one(sql, dialect=self._dialect)
            except Exception as e:
                return False, None, f"join check parse failed: {e}"
            for join in ast.find_all(_sg_exp.Join):
                if join.args.get('on') is None and join.args.get('using') is None:
                    kind = (join.args.get('kind') or "").upper()
                    if kind == 'CROSS':
                        continue
                    return False, None, "JOIN missing ON / USING clause (implicit cross-join)"
            return True, None, "ok"
        return await asyncio.to_thread(_run)

    async def _cardinality_check(self, sql: str) -> Tuple[bool, Optional[int], str]:
        ceiling = int(self._config.max_estimated_rows)
        def _run() -> Tuple[bool, Optional[int], str]:
            try:
                ast = _sg_parse_one(sql, dialect=self._dialect)
            except Exception as e:
                return False, None, f"cardinality parse failed: {e}"
            limit_value: Optional[int] = None
            for limit in ast.find_all(_sg_exp.Limit):
                expression = limit.expression
                if isinstance(expression, _sg_exp.Literal) and expression.is_int:
                    try:
                        candidate = int(expression.this)
                    except (TypeError, ValueError):
                        candidate = None
                    if candidate is not None and (limit_value is None or candidate > limit_value):
                        limit_value = candidate
            estimate = limit_value if limit_value is not None else ceiling
            if estimate > ceiling:
                return False, estimate, f"estimated_rows={estimate} > max_estimated_rows={ceiling}"
            return True, estimate, "ok"
        return await asyncio.to_thread(_run)

    async def _probe_check(self, sql: str) -> Tuple[bool, Optional[int], str]:
        try:
            return await self._probe(sql)
        except Exception as e:
            return False, None, f"probe failed: {type(e).__name__}: {e}"


def make_clickhouse_explain_probe(ch_executor: 'ClickHouseExecutor') -> ExplainProbe:
    """Return an ExplainProbe that validates SQL against ClickHouse via EXPLAIN.

    Probe runs EXPLAIN on the candidate SQL and returns (ok, None, reason).
    Estimated rows are not extracted from CH EXPLAIN text output — only
    syntax/dialect validity is checked. Failures soft-return False so a
    CH outage cannot block all SQL traffic (consistent with _probe_check
    exception handling in LogicValidator).

    :param ch_executor: ClickHouseExecutor - Live executor with explain() method
    :return: ExplainProbe - Callable compatible with LogicValidator.explain_probe
    """
    async def _probe(sql: str) -> Tuple[bool, Optional[int], str]:
        try:
            await ch_executor.explain(sql)
            return True, None, "ok"
        except Exception as e:
            return False, None, f"ch_explain: {type(e).__name__}: {e}"
    return _probe


__all__ = ['LogicValidator', 'ExplainProbe', 'make_clickhouse_explain_probe']
