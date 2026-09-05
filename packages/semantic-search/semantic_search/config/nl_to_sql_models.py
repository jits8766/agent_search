"""Typed configuration dataclasses for the NL-to-SQL Analytics pipeline.

Each subsystem-specific dataclass enforces invariants in `__post_init__` and
rejects malformed input via `from_dict`. Table / database / dialect names are
config-driven so a future rename never requires a code change.

Layout mirrors `config/models.py`:
- One dataclass per pipeline stage (athena, schema_discovery, generation,
  security, validation, execution, verifier).
- Aggregate `NLToSQLConfig` composes them with cross-cutting invariants.
- Optional `AnalyticsConfig` (real-time ClickHouse + MV router + semantic
  NL-to-SQL cache) — additive; absent in YAML keeps the legacy Athena path.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from semantic_search.config.analytics_models import AnalyticsConfig
from semantic_search.core.exceptions import ConfigurationError

_ALLOWED_SQL_DIALECTS = frozenset({'trino', 'presto', 'clickhouse'})


def _require(d: Dict[str, Any], keys: List[str], context: str) -> None:
    """Raise ConfigurationError if any required key is missing from `d`."""
    if not isinstance(d, dict):
        raise ConfigurationError(f"{context}: expected dict, got {type(d).__name__}")
    for key in keys:
        if key not in d:
            raise ConfigurationError(f"{context}.{key} is required")


@dataclass
class SchemaDiscoveryWeightsConfig:
    """Per-component weights for schema-column relevance scoring.

    Weights MUST sum to 1.0 (within float tolerance) so the resulting score
    stays in [0,1] and downstream gates can reason about it without hidden
    re-normalization.
    """
    keyword: float
    semantic: float
    fk: float

    _SUM_TOLERANCE = 1e-6

    def __post_init__(self) -> None:
        for name, value in (('keyword', self.keyword), ('semantic', self.semantic), ('fk', self.fk)):
            if not 0.0 <= float(value) <= 1.0:
                raise ConfigurationError(f"nl_to_sql.schema_discovery.weights.{name} must be in [0,1]")
        total = float(self.keyword) + float(self.semantic) + float(self.fk)
        if abs(total - 1.0) > self._SUM_TOLERANCE:
            raise ConfigurationError(f"nl_to_sql.schema_discovery.weights must sum to 1.0 (got {total:.6f})")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SchemaDiscoveryWeightsConfig':
        _require(d, ['keyword', 'semantic', 'fk'], 'nl_to_sql.schema_discovery.weights')
        return cls(keyword=float(d['keyword']), semantic=float(d['semantic']), fk=float(d['fk']))


@dataclass
class SchemaDiscoveryConfig:
    """Long-context schema discovery / column-pruning config (stage 1)."""
    enabled: bool
    weights: SchemaDiscoveryWeightsConfig
    max_columns: int
    min_score: float
    sample_values_per_column: int
    min_distribution_fraction: float
    catalog_path: str
    min_columns_fallback: int
    cache_ttl_seconds: int = 300
    pinned_columns: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if int(self.max_columns) < 1:
            raise ConfigurationError("nl_to_sql.schema_discovery.max_columns must be >= 1")
        if not 0.0 <= float(self.min_score) <= 1.0:
            raise ConfigurationError("nl_to_sql.schema_discovery.min_score must be in [0,1]")
        if int(self.sample_values_per_column) < 0:
            raise ConfigurationError("nl_to_sql.schema_discovery.sample_values_per_column must be >= 0")
        if not 0.0 <= float(self.min_distribution_fraction) <= 1.0:
            raise ConfigurationError("nl_to_sql.schema_discovery.min_distribution_fraction must be in [0,1]")
        if not isinstance(self.catalog_path, str):
            raise ConfigurationError("nl_to_sql.schema_discovery.catalog_path must be a string")
        if int(self.min_columns_fallback) < 0:
            raise ConfigurationError("nl_to_sql.schema_discovery.min_columns_fallback must be >= 0")
        if int(self.min_columns_fallback) > int(self.max_columns):
            raise ConfigurationError("nl_to_sql.schema_discovery.min_columns_fallback must be <= max_columns")
        if int(self.cache_ttl_seconds) < 0:
            raise ConfigurationError("nl_to_sql.schema_discovery.cache_ttl_seconds must be >= 0 (0 disables caching)")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SchemaDiscoveryConfig':
        _require(d, ['enabled', 'weights', 'max_columns', 'min_score', 'sample_values_per_column', 'min_distribution_fraction', 'catalog_path', 'min_columns_fallback'], 'nl_to_sql.schema_discovery')
        return cls(
            enabled=bool(d['enabled']),
            weights=SchemaDiscoveryWeightsConfig.from_dict(d['weights']),
            max_columns=int(d['max_columns']),
            min_score=float(d['min_score']),
            sample_values_per_column=int(d['sample_values_per_column']),
            min_distribution_fraction=float(d['min_distribution_fraction']),
            catalog_path=str(d['catalog_path']),
            min_columns_fallback=int(d['min_columns_fallback']),
            cache_ttl_seconds=int(d.get('cache_ttl_seconds', 300)),
            pinned_columns=list(d.get('pinned_columns', [])),
        )


@dataclass
class SqlGenerationConfig:
    """LLM-driven SQL generation config (stage 2)."""
    task_type: str
    prompt_tag: str
    min_confidence: float
    max_retries: int
    max_few_shot_examples: int
    model_override: str
    gen_cache_ttl_seconds: int = 60
    generation_timeout_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not self.task_type:
            raise ConfigurationError("nl_to_sql.generation.task_type must be non-empty")
        if not self.prompt_tag:
            raise ConfigurationError("nl_to_sql.generation.prompt_tag must be non-empty")
        if not 0.0 <= float(self.min_confidence) <= 1.0:
            raise ConfigurationError("nl_to_sql.generation.min_confidence must be in [0,1]")
        if int(self.max_retries) < 0:
            raise ConfigurationError("nl_to_sql.generation.max_retries must be >= 0")
        if int(self.max_few_shot_examples) < 0:
            raise ConfigurationError("nl_to_sql.generation.max_few_shot_examples must be >= 0")
        if not isinstance(self.model_override, str):
            raise ConfigurationError("nl_to_sql.generation.model_override must be a string")
        if int(self.gen_cache_ttl_seconds) < 0:
            raise ConfigurationError("nl_to_sql.generation.gen_cache_ttl_seconds must be >= 0 (0 disables caching)")
        if float(self.generation_timeout_seconds) < 0:
            raise ConfigurationError("nl_to_sql.generation.generation_timeout_seconds must be >= 0 (0 disables)")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SqlGenerationConfig':
        _require(d, ['task_type', 'prompt_tag', 'min_confidence', 'max_retries', 'max_few_shot_examples', 'model_override'], 'nl_to_sql.generation')
        return cls(
            task_type=str(d['task_type']),
            prompt_tag=str(d['prompt_tag']),
            min_confidence=float(d['min_confidence']),
            max_retries=int(d['max_retries']),
            max_few_shot_examples=int(d['max_few_shot_examples']),
            model_override=str(d['model_override']),
            gen_cache_ttl_seconds=int(d.get('gen_cache_ttl_seconds', 60)),
            generation_timeout_seconds=float(d.get('generation_timeout_seconds', 0.0)),
        )


@dataclass
class BulkTimeWindowConfig:
    """Per-table time-window contract for the security validator (gap 3).

    The AST validator uses this to inject a default time-window predicate when
    a query touches a known bulk table without one — and to reject queries that
    explicitly request a window beyond ``max_days``. Operator tunes per table.

    :param table_name: str - Bare table name to match against the FROM clause.
    :param column: str - Column the time-window predicate applies to (e.g. ``event_ts``).
    :param max_days: int - Hard ceiling. Existing predicates referencing more days are rejected.
    :param default_days: int - Window injected when no predicate exists. Must be ``<= max_days``.
    """
    table_name: str
    column: str
    max_days: int
    default_days: int

    def __post_init__(self) -> None:
        if not isinstance(self.table_name, str) or not self.table_name:
            raise ConfigurationError("nl_to_sql.security.bulk_time_windows[].table_name must be non-empty string")
        if not isinstance(self.column, str) or not self.column:
            raise ConfigurationError("nl_to_sql.security.bulk_time_windows[].column must be non-empty string")
        if int(self.max_days) <= 0:
            raise ConfigurationError("nl_to_sql.security.bulk_time_windows[].max_days must be > 0")
        if int(self.default_days) <= 0:
            raise ConfigurationError("nl_to_sql.security.bulk_time_windows[].default_days must be > 0")
        if int(self.default_days) > int(self.max_days):
            raise ConfigurationError("nl_to_sql.security.bulk_time_windows[].default_days must be <= max_days")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'BulkTimeWindowConfig':
        _require(d, ['table_name', 'column', 'max_days', 'default_days'], 'nl_to_sql.security.bulk_time_windows[]')
        return cls(table_name=str(d['table_name']), column=str(d['column']), max_days=int(d['max_days']), default_days=int(d['default_days']))


@dataclass
class SqlSecurityConfig:
    """AST-based security validation config (stage 3)."""
    allowed_tables: List[str]
    pii_columns: List[str]
    require_where_or_limit: bool
    max_limit: int
    bulk_time_windows: List[BulkTimeWindowConfig] = field(default_factory=list)
    # When True and a SchemaCatalog is wired into AstSecurityValidator,
    # every column reference in the generated SQL is validated against the
    # catalog's known columns for the target table. Queries referencing
    # columns absent from the catalog are rejected with failure_mode='security'.
    # Requires schema_catalog to be wired at validator construction time;
    # has no effect when schema_catalog=None.
    validate_columns_against_catalog: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.allowed_tables, list) or not self.allowed_tables:
            raise ConfigurationError("nl_to_sql.security.allowed_tables must be a non-empty list")
        for t in self.allowed_tables:
            if not isinstance(t, str) or not t:
                raise ConfigurationError("nl_to_sql.security.allowed_tables entries must be non-empty strings")
        if not isinstance(self.pii_columns, list):
            raise ConfigurationError("nl_to_sql.security.pii_columns must be a list (empty allowed)")
        for c in self.pii_columns:
            if not isinstance(c, str) or not c:
                raise ConfigurationError("nl_to_sql.security.pii_columns entries must be non-empty strings")
        if not isinstance(self.require_where_or_limit, bool):
            raise ConfigurationError("nl_to_sql.security.require_where_or_limit must be a bool")
        if int(self.max_limit) < 1:
            raise ConfigurationError("nl_to_sql.security.max_limit must be >= 1")
        if not isinstance(self.bulk_time_windows, list):
            raise ConfigurationError("nl_to_sql.security.bulk_time_windows must be a list (empty allowed)")
        for tw in self.bulk_time_windows:
            if not isinstance(tw, BulkTimeWindowConfig):
                raise ConfigurationError("nl_to_sql.security.bulk_time_windows entries must be BulkTimeWindowConfig")
        if not isinstance(self.validate_columns_against_catalog, bool):
            raise ConfigurationError("nl_to_sql.security.validate_columns_against_catalog must be a bool")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SqlSecurityConfig':
        _require(d, ['allowed_tables', 'pii_columns', 'require_where_or_limit', 'max_limit'], 'nl_to_sql.security')
        raw_windows = d.get('bulk_time_windows') or []
        if not isinstance(raw_windows, list):
            raise ConfigurationError("nl_to_sql.security.bulk_time_windows must be a list when present")
        windows = [BulkTimeWindowConfig.from_dict(w) for w in raw_windows]
        return cls(
            allowed_tables=[str(t) for t in d['allowed_tables']],
            pii_columns=[str(c) for c in d['pii_columns']],
            require_where_or_limit=bool(d['require_where_or_limit']),
            max_limit=int(d['max_limit']),
            bulk_time_windows=windows,
            validate_columns_against_catalog=bool(d.get('validate_columns_against_catalog', False)),
        )


@dataclass
class SqlLogicValidationConfig:
    """Parallel logic-validation config (stage 4).

    Optional ``max_bytes_scanned`` controls the EXPLAIN-probe byte-scan cap.
    When ``enable_explain_probe=true`` AND ``max_bytes_scanned`` is set,
    the registry constructs an ``AthenaExplainProbe`` that runs
    ``EXPLAIN (TYPE IO, FORMAT JSON)`` against candidate SQL and rejects
    queries whose estimated byte-scan exceeds the cap. ``None`` leaves
    probe construction to the registry's explicit defaults.

    :param enabled: bool - Master toggle for logic validation.
    :param max_estimated_rows: int - Row-cardinality cap (existing).
    :param enable_explain_probe: bool - When True, an ``ExplainProbe``
        callable is invoked per SQL. Probe failures (timeout, transport,
        parse) soft-fail; only a confirmed cap breach blocks the SQL.
    :param max_bytes_scanned: Optional[int] - Hard cap on Athena bytes
        scanned. ``None`` disables byte-scan enforcement (the probe may
        still be wired for row estimation only). When set, must be > 0.
    """
    enabled: bool
    max_estimated_rows: int
    enable_explain_probe: bool
    max_bytes_scanned: Optional[int] = None

    def __post_init__(self) -> None:
        if int(self.max_estimated_rows) < 1:
            raise ConfigurationError("nl_to_sql.validation.max_estimated_rows must be >= 1")
        if not isinstance(self.enable_explain_probe, bool):
            raise ConfigurationError("nl_to_sql.validation.enable_explain_probe must be a bool")
        if self.max_bytes_scanned is not None:
            if not isinstance(self.max_bytes_scanned, int) or isinstance(self.max_bytes_scanned, bool):
                raise ConfigurationError("nl_to_sql.validation.max_bytes_scanned must be int when set")
            if self.max_bytes_scanned <= 0:
                raise ConfigurationError("nl_to_sql.validation.max_bytes_scanned must be > 0 when set; omit or set null to disable")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SqlLogicValidationConfig':
        _require(d, ['enabled', 'max_estimated_rows', 'enable_explain_probe'], 'nl_to_sql.validation')
        raw_bytes = d.get('max_bytes_scanned')
        max_bytes_scanned: Optional[int] = int(raw_bytes) if raw_bytes is not None else None
        return cls(enabled=bool(d['enabled']), max_estimated_rows=int(d['max_estimated_rows']), enable_explain_probe=bool(d['enable_explain_probe']), max_bytes_scanned=max_bytes_scanned)


@dataclass
class SqlExecutionConfig:
    """Execution gate config (stage 5)."""
    max_rows: int
    timeout_seconds: float

    def __post_init__(self) -> None:
        if int(self.max_rows) < 1:
            raise ConfigurationError("nl_to_sql.execution.max_rows must be >= 1")
        if float(self.timeout_seconds) <= 0.0:
            raise ConfigurationError("nl_to_sql.execution.timeout_seconds must be > 0")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SqlExecutionConfig':
        _require(d, ['max_rows', 'timeout_seconds'], 'nl_to_sql.execution')
        return cls(max_rows=int(d['max_rows']), timeout_seconds=float(d['timeout_seconds']))


@dataclass
class SqlVerifierConfig:
    """Post-execution verifier-gate config.

    Implements the conditional verifier-skip on warm cached SQL templates.
    When ``skip_warmth_threshold`` is reached AND the cached SQL template's
    ``schema_version`` matches the current schema, the analytics fast path is
    allowed to short-circuit the verifier with an auto-pass verdict
    (``model='skipped_warm_cache'``). The skip is tracked via
    ``AnalyticsRouter.verifier_skip_stats`` so the verifier-skip-rate KPI
    dashboard can plot the ratio without inferring it from logs.
    """
    enabled: bool
    task_type: str
    prompt_tag: str
    sample_rows: int
    min_confidence: float
    model_override: str
    skip_enabled: bool
    skip_warmth_threshold: int
    # Cost classes that ALWAYS run the verifier regardless of cache warmth
    # or schema-version match. Typical setting: ['expensive', 'oversized']
    # so a stale-but-warm cache cannot ship an unverified expensive answer.
    mandatory_for_classes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.task_type:
            raise ConfigurationError("nl_to_sql.verifier.task_type must be non-empty")
        if not self.prompt_tag:
            raise ConfigurationError("nl_to_sql.verifier.prompt_tag must be non-empty")
        if int(self.sample_rows) < 0:
            raise ConfigurationError("nl_to_sql.verifier.sample_rows must be >= 0")
        if not 0.0 <= float(self.min_confidence) <= 1.0:
            raise ConfigurationError("nl_to_sql.verifier.min_confidence must be in [0,1]")
        if not isinstance(self.model_override, str):
            raise ConfigurationError("nl_to_sql.verifier.model_override must be a string")
        if not isinstance(self.skip_enabled, bool):
            raise ConfigurationError("nl_to_sql.verifier.skip_enabled must be a bool")
        if int(self.skip_warmth_threshold) < 1:
            raise ConfigurationError("nl_to_sql.verifier.skip_warmth_threshold must be >= 1 (need at least one verifier-pass before trusting a cached SQL)")
        if not isinstance(self.mandatory_for_classes, list):
            raise ConfigurationError("nl_to_sql.verifier.mandatory_for_classes must be a list of strings")
        allowed = {'cheap', 'medium', 'expensive', 'oversized'}
        for cls_name in self.mandatory_for_classes:
            if not isinstance(cls_name, str) or cls_name not in allowed:
                raise ConfigurationError(f"nl_to_sql.verifier.mandatory_for_classes entries must be one of {sorted(allowed)}")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SqlVerifierConfig':
        _require(d, ['enabled', 'task_type', 'prompt_tag', 'sample_rows', 'min_confidence', 'model_override', 'skip_enabled', 'skip_warmth_threshold'], 'nl_to_sql.verifier')
        return cls(
            enabled=bool(d['enabled']),
            task_type=str(d['task_type']),
            prompt_tag=str(d['prompt_tag']),
            sample_rows=int(d['sample_rows']),
            min_confidence=float(d['min_confidence']),
            model_override=str(d['model_override']),
            skip_enabled=bool(d['skip_enabled']),
            skip_warmth_threshold=int(d['skip_warmth_threshold']),
            mandatory_for_classes=list(d.get('mandatory_for_classes', []) or []),
        )


@dataclass
class RetrievedContentSanitizerConfig:
    """Per-fragment Layer-0 sanitizer policy for retrieved content.

    Decoupled from `safety.ingress_sanitizer.max_chars` (the conversational
    500-char cap on user inputs) because schema dumps + verifier sample
    rows + verifier-notes-as-retry-context routinely exceed 500 chars
    and would otherwise fail-close the entire analytics request at the
    LLM ingress gate.

    Policy is **mask-on-block**, not fail-closed: a single poisoned cell
    or a single PII hit replaces the offending fragment with
    `mask_replacement` and emits one `retrieved_content_sanitized`
    audit signal (when the wired `SignalStore` is non-None and
    `emit_audit_signal=True`). The analytics request continues with the
    redacted prompt.

    :param enabled: bool - Master toggle (False = pass-through,
        no truncation, no masking, no signal)
    :param max_chars_per_fragment: int - Hard length cap per fragment
        (e.g. one column description, one sample value, one verifier
        sample row). Anything above is hard-truncated with a
        `[TRUNCATED]` suffix BEFORE the underlying sanitizer's
        blocklist + PII checks run.
    :param mask_replacement: str - Token written in place of any
        fragment that fails any check (length, blocklist, PII)
    :param emit_audit_signal: bool - When True, every block emits one
        `retrieved_content_sanitized` FeedbackSignal via the wired
        SignalStore. False keeps the warning log but skips the signal
        store write.
    """
    enabled: bool
    max_chars_per_fragment: int
    mask_replacement: str
    emit_audit_signal: bool

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("nl_to_sql.retrieved_content_sanitizer.enabled must be a bool")
        if not isinstance(self.max_chars_per_fragment, int) or self.max_chars_per_fragment < 16:
            raise ConfigurationError("nl_to_sql.retrieved_content_sanitizer.max_chars_per_fragment must be int >= 16")
        if not isinstance(self.mask_replacement, str) or not self.mask_replacement:
            raise ConfigurationError("nl_to_sql.retrieved_content_sanitizer.mask_replacement must be a non-empty string")
        if not isinstance(self.emit_audit_signal, bool):
            raise ConfigurationError("nl_to_sql.retrieved_content_sanitizer.emit_audit_signal must be a bool")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'RetrievedContentSanitizerConfig':
        _require(d, ['enabled', 'max_chars_per_fragment', 'mask_replacement', 'emit_audit_signal'], 'nl_to_sql.retrieved_content_sanitizer')
        return cls(enabled=bool(d['enabled']), max_chars_per_fragment=int(d['max_chars_per_fragment']), mask_replacement=str(d['mask_replacement']), emit_audit_signal=bool(d['emit_audit_signal']))


@dataclass
class AthenaClientConfig:
    """Athena client config for the seed pipeline and seed_clickhouse script.

    All fields are required — values flow from nl_to_sql.athena in base.yaml.
    s3_output_location and region may be empty strings when the service relies
    on S3_ATHENA_DIR / AWS_REGION environment variables.

    :param s3_output_location: str - S3 URI prefix for Athena query results
    :param region: str - AWS region; empty falls back to AWS_REGION env var
    :param poll_sleep_seconds: float - Polling interval for query status
    :param connect_timeout: int - TCP connect timeout in seconds
    :param read_timeout: int - S3 read timeout in seconds
    :param max_retries: int - Boto3 retry attempts
    :param s3_list_max_keys: int - Max keys for S3 list operations
    """
    s3_output_location: str
    region: str
    poll_sleep_seconds: float
    connect_timeout: int
    read_timeout: int
    max_retries: int
    s3_list_max_keys: int
    role_arn: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.s3_output_location, str):
            raise ConfigurationError("nl_to_sql.athena.s3_output_location must be a string")
        if not isinstance(self.region, str):
            raise ConfigurationError("nl_to_sql.athena.region must be a string")
        if float(self.poll_sleep_seconds) <= 0.0:
            raise ConfigurationError("nl_to_sql.athena.poll_sleep_seconds must be > 0")
        if int(self.connect_timeout) <= 0:
            raise ConfigurationError("nl_to_sql.athena.connect_timeout must be > 0")
        if int(self.read_timeout) <= 0:
            raise ConfigurationError("nl_to_sql.athena.read_timeout must be > 0")
        if int(self.max_retries) < 0:
            raise ConfigurationError("nl_to_sql.athena.max_retries must be >= 0")
        if int(self.s3_list_max_keys) < 1:
            raise ConfigurationError("nl_to_sql.athena.s3_list_max_keys must be >= 1")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'AthenaClientConfig':
        _require(d, ['s3_output_location', 'region', 'poll_sleep_seconds', 'connect_timeout', 'read_timeout', 'max_retries', 's3_list_max_keys'], 'nl_to_sql.athena')
        return cls(s3_output_location=str(d['s3_output_location']), region=str(d['region']), poll_sleep_seconds=float(d['poll_sleep_seconds']), connect_timeout=int(d['connect_timeout']), read_timeout=int(d['read_timeout']), max_retries=int(d['max_retries']), s3_list_max_keys=int(d['s3_list_max_keys']), role_arn=str(d.get('role_arn') or ""))


@dataclass
class NLToSQLConfig:
    """Top-level NL-to-SQL pipeline config.

    Cross-cutting invariants enforced here (not on the children):
    - `table` MUST appear in `security.allowed_tables` so the AST validator
      can never reject the very table the discovery layer prunes against.
    """
    enabled: bool
    table: str
    database: str
    sql_dialect: str
    schema_discovery: SchemaDiscoveryConfig
    generation: SqlGenerationConfig
    security: SqlSecurityConfig
    validation: SqlLogicValidationConfig
    execution: SqlExecutionConfig
    verifier: SqlVerifierConfig
    analytics: Optional[AnalyticsConfig] = None
    retrieved_content_sanitizer: Optional[RetrievedContentSanitizerConfig] = None
    # Cost-class bands for the analytics-side cost classifier (gap 10).
    # Optional so existing configs without the block remain valid; when
    # absent the classifier is wired off and SqlValidationResult.cost_class
    # stays None.
    cost_class: Optional['CostClassConfig'] = None
    # Athena client settings for the seed pipeline. Optional for backward
    # compatibility; when absent seeding skips with a warning.
    athena: Optional[AthenaClientConfig] = None

    def __post_init__(self) -> None:
        if not isinstance(self.table, str) or not self.table:
            raise ConfigurationError("nl_to_sql.table must be a non-empty string")
        if not isinstance(self.database, str) or not self.database:
            raise ConfigurationError("nl_to_sql.database must be a non-empty string")
        if self.sql_dialect not in _ALLOWED_SQL_DIALECTS:
            raise ConfigurationError(f"nl_to_sql.sql_dialect must be one of {sorted(_ALLOWED_SQL_DIALECTS)}")
        if self.table not in self.security.allowed_tables:
            raise ConfigurationError(f"nl_to_sql.table='{self.table}' must appear in nl_to_sql.security.allowed_tables")
        if self.security.max_limit < self.execution.max_rows:
            raise ConfigurationError("nl_to_sql.security.max_limit must be >= nl_to_sql.execution.max_rows")
        if self.analytics is not None and not isinstance(self.analytics, AnalyticsConfig):
            raise ConfigurationError("nl_to_sql.analytics must be a typed AnalyticsConfig or omitted")
        if self.retrieved_content_sanitizer is not None and not isinstance(self.retrieved_content_sanitizer, RetrievedContentSanitizerConfig):
            raise ConfigurationError("nl_to_sql.retrieved_content_sanitizer must be a typed RetrievedContentSanitizerConfig or omitted")
        if self.athena is not None and not isinstance(self.athena, AthenaClientConfig):
            raise ConfigurationError("nl_to_sql.athena must be a typed AthenaClientConfig or omitted")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'NLToSQLConfig':
        from semantic_search.nl_to_sql.cost_class import CostClassConfig as _CostClassConfig
        _require(d, ['enabled', 'table', 'database', 'sql_dialect', 'schema_discovery', 'generation', 'security', 'validation', 'execution', 'verifier'], 'nl_to_sql')
        analytics = AnalyticsConfig.from_dict(d['analytics']) if d.get('analytics') is not None else None
        rcs_block = d.get('retrieved_content_sanitizer')
        retrieved_content_sanitizer = (RetrievedContentSanitizerConfig.from_dict(rcs_block) if rcs_block is not None else None)
        cost_class_block = d.get('cost_class')
        cost_class_cfg = (_CostClassConfig.from_dict(cost_class_block) if cost_class_block is not None else None)
        athena_block = d.get('athena')
        athena_cfg = AthenaClientConfig.from_dict(athena_block) if isinstance(athena_block, dict) else None
        return cls(
            enabled=bool(d['enabled']),
            table=str(d['table']),
            database=str(d['database']),
            sql_dialect=str(d['sql_dialect']),
            schema_discovery=SchemaDiscoveryConfig.from_dict(d['schema_discovery']),
            generation=SqlGenerationConfig.from_dict(d['generation']),
            security=SqlSecurityConfig.from_dict(d['security']),
            validation=SqlLogicValidationConfig.from_dict(d['validation']),
            execution=SqlExecutionConfig.from_dict(d['execution']),
            verifier=SqlVerifierConfig.from_dict(d['verifier']),
            analytics=analytics,
            retrieved_content_sanitizer=retrieved_content_sanitizer,
            cost_class=cost_class_cfg,
            athena=athena_cfg,
        )


__all__ = [
    'AthenaClientConfig',
    'NLToSQLConfig',
    'SchemaDiscoveryConfig',
    'SchemaDiscoveryWeightsConfig',
    'SqlGenerationConfig',
    'SqlSecurityConfig',
    'SqlLogicValidationConfig',
    'SqlExecutionConfig',
    'SqlVerifierConfig',
    'RetrievedContentSanitizerConfig',
]
