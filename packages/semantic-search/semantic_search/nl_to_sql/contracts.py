"""Typed contracts for the NL-to-SQL pipeline.

Every cross-stage hand-off uses these dataclasses (never raw dicts). Each
`__post_init__` enforces invariants so a malformed instance fails at
construction — caller can never accidentally route a half-built result
into the next stage.
"""
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from semantic_search.core.exceptions import ValidationError


VALIDATION_FAILURE_MODES = frozenset({
    'security',
    'syntax',
    'join_correctness',
    'cardinality',
    'execution',
    'verifier',
    'schema_unavailable',
    'generation',
    'cost_budget_exceeded',
    # LLM committed to kind='refuse'. Distinct from 'generation'
    # (which covers schema-validation / under-confidence / provider
    # failures) so dashboards can attribute hard refusals separately
    # from soft generator failures.
    'llm_refused',
    # Per-tenant analytics rate limit exceeded. Distinct from
    # 'cost_budget_exceeded' (which is post-validation cost gate) so
    # dashboards can attribute capacity-shedding separately.
    'rate_limited',
    # All execution substrates (ClickHouse) were tried and
    # exhausted without producing a result. Distinct from 'execution' (which
    # means a substrate ran but returned an error) so dashboards can attribute
    # full-pipeline exhaustion separately from single-substrate failures.
    'no_substrate_available',
    # Query time range exceeds the configured data-window cap (max_days in
    # bulk_time_windows). Distinct from 'security' (which covers structural
    # SQL threats) so the API layer can surface a user-readable notice and
    # dashboards can track window-exceeded queries separately from attacks.
    'beyond_data_window',
    'unknown',
    # SQL generator referenced columns absent from the pruned schema; distinct
    # from 'generation' (which covers LLM/provider errors) so dashboards can
    # attribute schema-mismatch failures separately.
    'invalid_columns',
    # Orchestrator.analytics() wall-clock wait_for exceeded
    # general.search.analytics_timeout_seconds. Distinct from 'execution'
    # (substrate ran and errored) so dashboards can attribute SLA breaches.
    'timeout',
})

ANALYTICS_SUBSTRATE_LABELS = frozenset({
    '',
    'exact_cache',
    'clickhouse_hot',
    'clickhouse_snapshot',
    'domain_engine',
})

VERIFIER_VERDICTS = frozenset({'ok', 'empty_result', 'wrong_dimension', 'degenerate_aggregate', 'insufficient_data', 'unknown'})

COST_CLASS_LABELS = frozenset({'cheap', 'medium', 'expensive', 'oversized'})


def _new_id(prefix: str) -> str:
    """Build a short id with a typed prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class SchemaColumn:
    """A single column descriptor surfaced by the schema-discovery layer.

    The discovery context makes BIRD-style sample-value + distribution context
    available to the SQL-generating LLM, which prevents type-mismatch and
    out-of-domain literal generation (e.g. comparing an INT column against a
    string literal).

    :param name: str - Column name as it appears in the source table
    :param data_type: str - SQL data type (e.g. 'INT', 'VARCHAR', 'TIMESTAMP')
    :param sample_values: List[Any] - Top sample values present in production
    :param distribution: Dict[str, float] - {sample_value_str: row_fraction}
        — fraction in [0,1]; only values clearing the configured floor are kept
    :param description: str - Optional human-curated note (audit / prompt only)
    :param is_pii: bool - True if the source catalog flags this column as PII
    :param relevance_score: float - In [0,1]; weighted score from the pruner
    """
    name: str
    data_type: str
    sample_values: List[Any]
    distribution: Dict[str, float]
    description: str
    is_pii: bool
    relevance_score: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValidationError("SchemaColumn.name must be a non-empty string")
        if not isinstance(self.data_type, str) or not self.data_type:
            raise ValidationError("SchemaColumn.data_type must be a non-empty string")
        if not isinstance(self.sample_values, list):
            raise ValidationError("SchemaColumn.sample_values must be a list")
        if not isinstance(self.distribution, dict):
            raise ValidationError("SchemaColumn.distribution must be a dict")
        for key, value in self.distribution.items():
            if not isinstance(key, str):
                raise ValidationError("SchemaColumn.distribution keys must be strings")
            if not 0.0 <= float(value) <= 1.0:
                raise ValidationError("SchemaColumn.distribution values must be in [0,1]")
        if not isinstance(self.description, str):
            raise ValidationError("SchemaColumn.description must be a string")
        if not isinstance(self.is_pii, bool):
            raise ValidationError("SchemaColumn.is_pii must be a bool")
        if not 0.0 <= float(self.relevance_score) <= 1.0:
            raise ValidationError("SchemaColumn.relevance_score must be in [0,1]")


@dataclass
class PrunedSchema:
    """Result of the schema-discovery stage.

    :param table: str - Fully-qualified table the SQL must target
    :param database: str - Database the table lives in
    :param columns: List[SchemaColumn] - Pruned columns ordered by relevance desc
    :param total_columns_considered: int - Columns in the source catalog before pruning
    :param latency_ms: float - Wall-clock time spent in discovery
    """
    table: str
    database: str
    columns: List[SchemaColumn]
    total_columns_considered: int
    latency_ms: float

    def __post_init__(self) -> None:
        if not isinstance(self.table, str) or not self.table:
            raise ValidationError("PrunedSchema.table must be a non-empty string")
        if not isinstance(self.database, str) or not self.database:
            raise ValidationError("PrunedSchema.database must be a non-empty string")
        if not isinstance(self.columns, list):
            raise ValidationError("PrunedSchema.columns must be a list")
        for col in self.columns:
            if not isinstance(col, SchemaColumn):
                raise ValidationError("PrunedSchema.columns entries must be SchemaColumn")
        if int(self.total_columns_considered) < 0:
            raise ValidationError("PrunedSchema.total_columns_considered must be >= 0")
        if float(self.latency_ms) < 0.0:
            raise ValidationError("PrunedSchema.latency_ms must be >= 0")


@dataclass
class SqlGenerationResult:
    """Result of the LLM SQL-generation stage.

    :param sql: str - Generated SQL (may still be invalid; validation comes next)
    :param confidence: float - Self-reported confidence in [0,1]
    :param model: str - Model that produced the SQL
    :param attempt_count: int - How many generation attempts were used (1 + retries)
    :param prompt_tokens: int - Prompt tokens spent (>= 0)
    :param completion_tokens: int - Completion tokens spent (>= 0)
    :param cost_usd: float - Estimated USD cost for the generation pass
    :param latency_ms: float - Wall-clock time spent in generation
    """
    sql: str
    confidence: float
    model: str
    attempt_count: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float

    def __post_init__(self) -> None:
        if not isinstance(self.sql, str) or not self.sql.strip():
            raise ValidationError("SqlGenerationResult.sql must be a non-empty string")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValidationError("SqlGenerationResult.confidence must be in [0,1]")
        if not isinstance(self.model, str):
            raise ValidationError("SqlGenerationResult.model must be a string")
        if int(self.attempt_count) < 1:
            raise ValidationError("SqlGenerationResult.attempt_count must be >= 1")
        if int(self.prompt_tokens) < 0:
            raise ValidationError("SqlGenerationResult.prompt_tokens must be >= 0")
        if int(self.completion_tokens) < 0:
            raise ValidationError("SqlGenerationResult.completion_tokens must be >= 0")
        if float(self.cost_usd) < 0.0:
            raise ValidationError("SqlGenerationResult.cost_usd must be >= 0")
        if float(self.latency_ms) < 0.0:
            raise ValidationError("SqlGenerationResult.latency_ms must be >= 0")


@dataclass
class SqlValidationResult:
    """Outcome of the AST-security and parallel logic-validation stages.

    :param is_valid: bool - True iff every gate passed
    :param sql: str - SQL after any auto-mutation (e.g. LIMIT clamp). Equal to
        input SQL when no mutation was applied.
    :param failure_mode: Optional[str] - One of VALIDATION_FAILURE_MODES (set
        when is_valid=False). MUST be None when is_valid=True.
    :param failure_reasons: List[str] - Human-readable reasons (for logs / UI)
    :param mutations: List[str] - Audit trail of auto-mutations applied
    :param estimated_rows: Optional[int] - Cardinality estimate (None when not probed)
    :param latency_ms: float - Wall-clock time spent across security + logic validation
    :param cost_class: Optional[str] - Cost-class verdict from the cost classifier
        ('cheap' | 'medium' | 'expensive' | 'oversized'). None when no classifier
        was wired in (legacy callers).
    """
    is_valid: bool
    sql: str
    failure_mode: Optional[str]
    failure_reasons: List[str]
    mutations: List[str]
    estimated_rows: Optional[int]
    latency_ms: float
    cost_class: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.is_valid, bool):
            raise ValidationError("SqlValidationResult.is_valid must be a bool")
        if not isinstance(self.sql, str) or not self.sql.strip():
            raise ValidationError("SqlValidationResult.sql must be a non-empty string")
        if self.is_valid and self.failure_mode is not None:
            raise ValidationError("SqlValidationResult.failure_mode must be None when is_valid=True")
        if not self.is_valid and self.failure_mode is None:
            raise ValidationError("SqlValidationResult.failure_mode is required when is_valid=False")
        if self.failure_mode is not None and self.failure_mode not in VALIDATION_FAILURE_MODES:
            raise ValidationError(
                f"SqlValidationResult.failure_mode must be one of {sorted(VALIDATION_FAILURE_MODES)}"
            )
        if not isinstance(self.failure_reasons, list):
            raise ValidationError("SqlValidationResult.failure_reasons must be a list")
        if not isinstance(self.mutations, list):
            raise ValidationError("SqlValidationResult.mutations must be a list")
        if self.estimated_rows is not None and int(self.estimated_rows) < 0:
            raise ValidationError("SqlValidationResult.estimated_rows must be >= 0")
        if float(self.latency_ms) < 0.0:
            raise ValidationError("SqlValidationResult.latency_ms must be >= 0")
        if self.cost_class is not None and self.cost_class not in COST_CLASS_LABELS:
            raise ValidationError(
                f"SqlValidationResult.cost_class must be one of {sorted(COST_CLASS_LABELS)}"
            )


@dataclass
class SqlExecutionResult:
    """Outcome of executing the validated SQL against Athena.

    Holds *only* structured rows + metadata — no Pandas DataFrame leaks
    through the pipeline boundary so callers (caches, UI) get a typed,
    JSON-serializable payload.

    :param sql: str - SQL that was executed (after validator mutations)
    :param rows: List[Dict[str, Any]] - Result rows, each row a column->value dict
    :param column_names: List[str] - Result column order (preserved from Athena)
    :param row_count: int - len(rows); duplicated for cheap consumer access
    :param latency_ms: float - Wall-clock execution time (Athena + S3 fetch)
    :param truncated: bool - True iff the row limit was hit
    """
    sql: str
    rows: List[Dict[str, Any]]
    column_names: List[str]
    row_count: int
    latency_ms: float
    truncated: bool

    def __post_init__(self) -> None:
        if not isinstance(self.sql, str) or not self.sql.strip():
            raise ValidationError("SqlExecutionResult.sql must be a non-empty string")
        if not isinstance(self.rows, list):
            raise ValidationError("SqlExecutionResult.rows must be a list")
        for row in self.rows:
            if not isinstance(row, dict):
                raise ValidationError("SqlExecutionResult.rows entries must be dicts")
        if not isinstance(self.column_names, list):
            raise ValidationError("SqlExecutionResult.column_names must be a list")
        for c in self.column_names:
            if not isinstance(c, str) or not c:
                raise ValidationError("SqlExecutionResult.column_names entries must be non-empty strings")
        if int(self.row_count) != len(self.rows):
            raise ValidationError("SqlExecutionResult.row_count must equal len(rows)")
        if float(self.latency_ms) < 0.0:
            raise ValidationError("SqlExecutionResult.latency_ms must be >= 0")
        if not isinstance(self.truncated, bool):
            raise ValidationError("SqlExecutionResult.truncated must be a bool")


@dataclass
class VerifierVerdict:
    """Output of the L3 post-execution verifier gate (stage 6).

    ``sufficient=True`` means the verifier judges the result actually
    answers the question. ``sufficient=False`` carries a ``failure_mode``
    so callers can route through the Zero-Result Guard rather than surface
    a bad answer.

    :param sufficient: bool - Verdict
    :param failure_mode: str - One of VERIFIER_VERDICTS ('ok' when sufficient=True)
    :param confidence: float - Verifier-reported confidence in [0,1]
    :param model: str - Model that produced the verdict
    :param latency_ms: float - Wall-clock verifier latency
    :param notes: str - Human-readable reason (audit log only — no PII)
    """
    sufficient: bool
    failure_mode: str
    confidence: float
    model: str
    latency_ms: float
    notes: str

    def __post_init__(self) -> None:
        if not isinstance(self.sufficient, bool):
            raise ValidationError("VerifierVerdict.sufficient must be a bool")
        if self.failure_mode not in VERIFIER_VERDICTS:
            raise ValidationError(
                f"VerifierVerdict.failure_mode must be one of {sorted(VERIFIER_VERDICTS)}"
            )
        if self.sufficient and self.failure_mode != 'ok':
            raise ValidationError("VerifierVerdict.failure_mode must be 'ok' when sufficient=True")
        if not self.sufficient and self.failure_mode == 'ok':
            raise ValidationError("VerifierVerdict.failure_mode must NOT be 'ok' when sufficient=False")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValidationError("VerifierVerdict.confidence must be in [0,1]")
        if not isinstance(self.model, str):
            raise ValidationError("VerifierVerdict.model must be a string")
        if float(self.latency_ms) < 0.0:
            raise ValidationError("VerifierVerdict.latency_ms must be >= 0")
        if not isinstance(self.notes, str):
            raise ValidationError("VerifierVerdict.notes must be a string")


@dataclass
class AnalyticsResult:
    """Final, end-to-end result of the NL-to-SQL pipeline.

    Carries enough metadata for the orchestrator to decide whether to
    surface the result to the user
    (``success=True and verifier.sufficient=True``) or route through the
    Zero-Result Guard (``success=False``).

    :param request_id: str - Correlation id (auto-generated when caller omits)
    :param question: str - Original natural-language analytics question
    :param sql_hint: str - Optional structured hint piped from the QI engine
    :param success: bool - True iff every stage passed AND verifier said sufficient
    :param failure_mode: Optional[str] - One of VALIDATION_FAILURE_MODES; None on success
    :param failure_reason: str - Human-readable reason (logs / UI)
    :param pruned_schema: Optional[PrunedSchema] - Schema discovery output (None if skipped)
    :param generation: Optional[SqlGenerationResult] - LLM-gen output (None if skipped)
    :param validation: Optional[SqlValidationResult] - Validator output (None if skipped)
    :param execution: Optional[SqlExecutionResult] - Athena execution output (None if skipped)
    :param verifier: Optional[VerifierVerdict] - Post-execution verifier verdict (None if skipped)
    :param total_latency_ms: float - End-to-end wall-clock latency
    :param created_at: float - Unix timestamp at construction
    :param as_of: Optional[float] - Data freshness anchor (Unix timestamp). Set to the
        most recent ingested-event time on ClickHouse, the snapshot time on Athena, or
        the cached value's verified moment when served from the semantic cache. None
        when the executor cannot report a freshness signal (legacy callers, stub paths).
        The UI renders ``as_of`` as "as of <ts> — <Δ> ago" so users always see how fresh
        the analytics answer is.
    :param freshness_lag_seconds: Optional[float] - Lag between ``as_of`` and
        ``created_at``. Pre-computed for caller convenience; None when ``as_of`` is None.
        Always >= 0 when set.
    :param mv_used: str - Name of the Materialized View used to answer the
        query (stage 5 — MV-aware rewrite). Empty when no rewrite occurred
        (router disabled, no matching MV, rewrite security-rejected, parse
        failure, or executor was Athena where MVs do not exist). Set by
        either :class:`AnalyticsRouter` (CH path) or
        :class:`NLToSQLPipeline` (canonical 6-stage path) when an MVRouter
        is wired in. Surfaced in ops dashboards as the ``mv_used`` label.
    :param intent_record_id: str - IntentRecord id for this analytics call.
        Threaded from the QI-emitted intent so an `analytics_failure`
        FeedbackSignal carries the same join key as the originating
        search; empty when the caller has no IntentRecord context (legacy
        / legacy callers without a known origin).
    :param analytics_substrate: str - Execution ladder label: ``exact_cache``,
        ``clickhouse_hot``, ``clickhouse_snapshot``, or empty when unknown
        (failure stubs, legacy callers).
    :param as_of_hot: Optional[float] - Freshness anchor for the hot ClickHouse tier
        (Unix timestamp). None when this path did not serve the answer.
    :param as_of_analytics: Optional[float] - Freshness anchor for the analytics
        warehouse tier (semantic-cache verification time, bulk/Athena snapshot, etc.).
        None when not applicable.
    :param llm_cost_usd: float - LLM spend (USD) observed during this analytics
        call only (NL-SQL gen, etc.). Search/QI spend is separate; the /search
        envelope sums both into ``query_intelligence.decision_cost_usd``.
    """
    request_id: str
    question: str
    sql_hint: str
    success: bool
    failure_mode: Optional[str]
    failure_reason: str
    pruned_schema: Optional[PrunedSchema]
    generation: Optional[SqlGenerationResult]
    validation: Optional[SqlValidationResult]
    execution: Optional[SqlExecutionResult]
    verifier: Optional[VerifierVerdict]
    total_latency_ms: float
    created_at: float = field(default_factory=time.time)
    as_of: Optional[float] = None
    freshness_lag_seconds: Optional[float] = None
    mv_used: str = ''
    intent_record_id: str = ''
    analytics_substrate: str = ''
    as_of_hot: Optional[float] = None
    as_of_analytics: Optional[float] = None
    llm_cost_usd: float = 0.0
    # Schema-drift detection (gap 7). ``schema_version_at_gen`` is the
    # ``SchemaCatalog.version`` snapshot taken when the prompt was built.
    # ``schema_drift_detected`` is set by the analytics router when it
    # observes that the gen-time version no longer matches the executor's
    # current authoritative version, which forces the verifier to run
    # regardless of cache warmth.
    schema_version_at_gen: str = ''
    schema_drift_detected: bool = False
    # Multi-period aggregation output. Populated by MultiPeriodHandler instead
    # of the NL-to-SQL pipeline. Structure: {dimension, applied_filters,
    # last_1h, last_24h, last_7d, last_30d} each period carrying rows/columns/
    # row_count/latency_ms/source. None on all other paths.
    periods: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValidationError("AnalyticsResult.request_id must be a non-empty string")
        if not isinstance(self.question, str) or not self.question:
            raise ValidationError("AnalyticsResult.question must be a non-empty string")
        if not isinstance(self.sql_hint, str):
            raise ValidationError("AnalyticsResult.sql_hint must be a string (empty allowed)")
        if not isinstance(self.success, bool):
            raise ValidationError("AnalyticsResult.success must be a bool")
        if self.success and self.failure_mode is not None:
            raise ValidationError("AnalyticsResult.failure_mode must be None on success")
        if not self.success and self.failure_mode is None:
            raise ValidationError("AnalyticsResult.failure_mode is required on failure")
        if self.failure_mode is not None and self.failure_mode not in VALIDATION_FAILURE_MODES:
            raise ValidationError(
                f"AnalyticsResult.failure_mode must be one of {sorted(VALIDATION_FAILURE_MODES)}"
            )
        if not isinstance(self.failure_reason, str):
            raise ValidationError("AnalyticsResult.failure_reason must be a string")
        if float(self.total_latency_ms) < 0.0:
            raise ValidationError("AnalyticsResult.total_latency_ms must be >= 0")
        # Freshness contract. as_of is optional (legacy callers, fast-path without
        # server timestamp), but when present it must be a positive Unix timestamp
        # and freshness_lag_seconds must be derivable (>=0). Auto-compute the lag
        # so callers only need to set as_of; they may override the lag if they
        # have a more accurate lag (e.g. when as_of comes from a future-dated cache write).
        if self.as_of is not None:
            try:
                as_of_val = float(self.as_of)
            except (TypeError, ValueError) as e:
                raise ValidationError("AnalyticsResult.as_of must be numeric when set") from e
            if as_of_val <= 0.0:
                raise ValidationError("AnalyticsResult.as_of must be > 0 when set")
            object.__setattr__(self, 'as_of', as_of_val)
            if self.freshness_lag_seconds is None:
                computed_lag = max(0.0, float(self.created_at) - as_of_val)
                object.__setattr__(self, 'freshness_lag_seconds', computed_lag)
            else:
                lag_val = float(self.freshness_lag_seconds)
                if lag_val < 0.0:
                    raise ValidationError("AnalyticsResult.freshness_lag_seconds must be >= 0 when set")
                object.__setattr__(self, 'freshness_lag_seconds', lag_val)
        else:
            if self.freshness_lag_seconds is not None:
                raise ValidationError("AnalyticsResult.freshness_lag_seconds requires as_of to be set")
        if not isinstance(self.mv_used, str):
            raise ValidationError("AnalyticsResult.mv_used must be a string (empty allowed)")
        if not isinstance(self.intent_record_id, str):
            raise ValidationError("AnalyticsResult.intent_record_id must be a string (empty allowed)")
        if not isinstance(self.analytics_substrate, str):
            raise ValidationError("AnalyticsResult.analytics_substrate must be a string")
        if self.analytics_substrate not in ANALYTICS_SUBSTRATE_LABELS:
            raise ValidationError(
                f"AnalyticsResult.analytics_substrate must be one of {sorted(ANALYTICS_SUBSTRATE_LABELS)}"
            )
        for label, val in (('as_of_hot', self.as_of_hot), ('as_of_analytics', self.as_of_analytics)):
            if val is None:
                continue
            try:
                fv = float(val)
            except (TypeError, ValueError) as e:
                raise ValidationError(f"AnalyticsResult.{label} must be numeric when set") from e
            if fv <= 0.0:
                raise ValidationError(f"AnalyticsResult.{label} must be > 0 when set")
            object.__setattr__(self, label, fv)
        if not isinstance(self.schema_version_at_gen, str):
            raise ValidationError("AnalyticsResult.schema_version_at_gen must be a string (empty allowed)")
        if not isinstance(self.schema_drift_detected, bool):
            raise ValidationError("AnalyticsResult.schema_drift_detected must be a bool")

    @staticmethod
    def new_request_id() -> str:
        """Generate a request_id for a new analytics request."""
        return _new_id('ana')


__all__ = [
    'SchemaColumn',
    'PrunedSchema',
    'SqlGenerationResult',
    'SqlValidationResult',
    'SqlExecutionResult',
    'VerifierVerdict',
    'AnalyticsResult',
    'ANALYTICS_SUBSTRATE_LABELS',
    'VALIDATION_FAILURE_MODES',
    'VERIFIER_VERDICTS',
    'COST_CLASS_LABELS',
]
