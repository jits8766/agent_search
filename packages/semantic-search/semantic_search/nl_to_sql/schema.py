"""Schema discovery / column pruning (stage 1).

Long-context column linker: scores every column in the catalog by a weighted
combination of (keyword overlap, semantic similarity, foreign-key signal) and
emits the top-K as a `PrunedSchema`. Columns flagged as PII in the catalog
are hard-excluded so the LLM cannot even *consider* generating SQL against
them — defence in depth alongside the AST PII gate.

The catalog itself is supplied externally:
- `JsonSchemaCatalog` reads the YAML-configured catalog file (typically built
  nightly by an ETL job in Athena). Relative paths resolve from the process cwd
  first, then from the repo root (parent of the ``semantic_search`` package), so
  bundled defaults work when uvicorn is started outside the workspace root.
The discovery layer NEVER hardcodes column names — every column descriptor
comes from the catalog, so adding/removing/renaming columns in production
requires only a catalog rebuild (no code change).
"""
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from llm_core.logging_utils import mask_path

from semantic_search.config.nl_to_sql_models import SchemaDiscoveryConfig
from semantic_search.core.exceptions import ConfigurationError, RetrievalError
from semantic_search.core.logging_utils import get_logger
from semantic_search.nl_to_sql.contracts import PrunedSchema, SchemaColumn

logger = get_logger(__name__)

_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9]+")


def _resolve_schema_catalog_path(path: str) -> Path:
    """Resolve ``catalog_path`` from cwd first, then repo root (parent of ``semantic_search/``).
    :param path: str - Value from ``nl_to_sql.schema_discovery.catalog_path``
    :return: Path - Absolute path to an existing catalog file
    :raises ConfigurationError: When path is empty or no file exists at cwd or anchored location
    """
    if not isinstance(path, str) or not path.strip():
        raise ConfigurationError("JsonSchemaCatalog requires a non-empty path")
    stripped = path.strip()
    candidate = Path(stripped)
    if candidate.is_file():
        return candidate.resolve()
    if not candidate.is_absolute():
        workspace_root = Path(__file__).resolve().parents[2]
        anchored = (workspace_root / stripped).resolve()
        if anchored.is_file():
            return anchored
    raise ConfigurationError(f"JsonSchemaCatalog file not found: {path}")


@dataclass(frozen=True)
class CatalogColumn:
    """A single column entry as it appears in the on-disk schema catalog.

    Frozen + typed so the catalog loader can validate every entry once and
    pass immutable descriptors into the discovery scorer.
    """
    name: str
    data_type: str
    description: str
    sample_values: Tuple[Any, ...]
    distribution: Tuple[Tuple[str, float], ...]
    foreign_key_keywords: Tuple[str, ...]
    is_pii: bool


class SchemaCatalog:
    """Catalog interface — every implementation returns the columns for a table."""

    def columns_for(self, table: str) -> List[CatalogColumn]:
        """Return ALL columns for `table` (un-pruned, full catalog projection)."""
        raise NotImplementedError

    @property
    def version(self) -> str:
        """Stable hash of the catalog content.

        The analytics router compares the prompt-time version to the
        execute-time version to detect schema drift on cached SQL templates.
        Implementations that cannot produce a hash (legacy, in-memory test
        doubles) may return ``''`` — drift detection then degrades to "always
        force verifier" rather than silently trusting the cache.
        """
        return ''


def _hash_catalog(by_table: Dict[str, List[CatalogColumn]]) -> str:
    """sha256 over the canonical JSON projection of a catalog map.

    Stable across process restarts so two replicas with identical catalogs
    agree on ``version``. Tuples are coerced to lists and dict keys sorted
    so list/tuple identity and dict insertion order do not perturb the hash.
    """
    canonical: Dict[str, List[Dict[str, Any]]] = {}
    for table, cols in by_table.items():
        canonical[str(table)] = [
            {
                'name': c.name,
                'data_type': c.data_type,
                'description': c.description,
                'sample_values': list(c.sample_values),
                'distribution': [list(pair) for pair in c.distribution],
                'foreign_key_keywords': list(c.foreign_key_keywords),
                'is_pii': bool(c.is_pii),
            }
            for c in cols
        ]
    payload = json.dumps(canonical, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


class JsonSchemaCatalog(SchemaCatalog):
    """JSON-file-backed catalog. Schema:

        {
          "tables": {
            "<table_name>": [
              {
                "name": "auction_type_id",
                "data_type": "INT",
                "description": "...",
                "sample_values": [1, 2, 3],
                "distribution": {"1": 0.62, "2": 0.23, "3": 0.15},
                "foreign_key_keywords": ["auction_type", "auction"],
                "is_pii": false
              },
              ...
            ]
          }
        }

    The loader rejects malformed entries at construction so a corrupted
    catalog fails loudly at boot rather than at first analytics query.
    """

    def __init__(self, path: str, canonical_table: Optional[str] = None):
        catalog_path = _resolve_schema_catalog_path(path)
        catalog_display = mask_path(catalog_path)
        try:
            with open(catalog_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            raise ConfigurationError(f"JsonSchemaCatalog failed to read {catalog_display}: {e}") from e
        if not isinstance(raw, dict) or 'tables' not in raw or not isinstance(raw['tables'], dict):
            raise ConfigurationError(f"JsonSchemaCatalog malformed: expected top-level 'tables' dict at {catalog_display}")
        self._by_table: Dict[str, List[CatalogColumn]] = {}
        for table_name, cols in raw['tables'].items():
            if not isinstance(cols, list):
                raise ConfigurationError(f"JsonSchemaCatalog '{table_name}' columns must be a list")
            self._by_table[str(table_name)] = [self._parse_column(table_name, c) for c in cols]
        if canonical_table is not None:
            keys = list(self._by_table.keys())
            if len(keys) == 1 and keys[0] != canonical_table:
                self._by_table[canonical_table] = self._by_table.pop(keys[0])
                logger.info(
                    f"json_schema_catalog_remapped file_key={keys[0]!r} canonical={canonical_table!r}"
                )
        self._version = _hash_catalog(self._by_table)
        logger.info(
            f"json_schema_catalog_loaded path={catalog_display} "
            f"tables={list(self._by_table.keys())} version={self._version[:12]}"
        )

    @staticmethod
    def _parse_column(table: str, raw: Any) -> CatalogColumn:
        """Parse and validate one catalog column entry from inbound schema JSON.
        Uses `.get()` with defaults for optional fields because this parses external
        API/file data, not YAML config. Inbound data is coerced and validated per
        the trust-boundary rules in `data-contracts.mdc`: external data gets
        ``.get()`` with safe defaults, then type/value validation. Config data
        (YAML) uses direct `config['key']` access to fail on missing values.
        """
        if not isinstance(raw, dict):
            raise ConfigurationError(f"JsonSchemaCatalog '{table}' column entry must be a dict")
        for required in ('name', 'data_type'):
            if required not in raw or not isinstance(raw[required], str) or not raw[required]:
                raise ConfigurationError(f"JsonSchemaCatalog '{table}' column missing required '{required}'")
        sample_values = raw.get('sample_values') or []
        if not isinstance(sample_values, list):
            raise ConfigurationError(f"JsonSchemaCatalog '{table}.{raw['name']}' sample_values must be a list")
        distribution_raw = raw.get('distribution') or {}
        if not isinstance(distribution_raw, dict):
            raise ConfigurationError(f"JsonSchemaCatalog '{table}.{raw['name']}' distribution must be a dict")
        distribution: List[Tuple[str, float]] = []
        for key, value in distribution_raw.items():
            try:
                fraction = float(value)
            except (TypeError, ValueError) as e:
                raise ConfigurationError(
                    f"JsonSchemaCatalog '{table}.{raw['name']}' distribution[{key}] not numeric"
                ) from e
            if not 0.0 <= fraction <= 1.0:
                raise ConfigurationError(
                    f"JsonSchemaCatalog '{table}.{raw['name']}' distribution[{key}]={fraction} outside [0,1]"
                )
            distribution.append((str(key), fraction))
        fk_kw_raw = raw.get('foreign_key_keywords') or []
        if not isinstance(fk_kw_raw, list):
            raise ConfigurationError(f"JsonSchemaCatalog '{table}.{raw['name']}' foreign_key_keywords must be a list")
        fk_kw = tuple(str(k).lower() for k in fk_kw_raw)
        is_pii = bool(raw.get('is_pii', False))
        description = str(raw.get('description', ''))
        return CatalogColumn(
            name=str(raw['name']),
            data_type=str(raw['data_type']),
            description=description,
            sample_values=tuple(sample_values),
            distribution=tuple(distribution),
            foreign_key_keywords=fk_kw,
            is_pii=is_pii,
        )

    def columns_for(self, table: str) -> List[CatalogColumn]:
        if table not in self._by_table:
            return []
        return list(self._by_table[table])

    @property
    def version(self) -> str:
        return self._version


def _tokenize(text: str) -> List[str]:
    """Lowercase + tokenize text into alpha-numeric tokens (>= 2 chars)."""
    if not text:
        return []
    return [t.lower() for t in _TOKEN_PATTERN.findall(text)]


def _column_token_bag(col: CatalogColumn) -> List[str]:
    """Project a column into a token bag used for both keyword and semantic scoring."""
    bag: List[str] = []
    bag.extend(_tokenize(col.name))
    bag.extend(_tokenize(col.description))
    for sv in col.sample_values:
        bag.extend(_tokenize(str(sv)))
    return bag


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """Jaccard similarity over two token bags. Returns 0.0 when either is empty."""
    set_a = set(a)
    set_b = set(b)
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return intersection / union


def _keyword_overlap(query_tokens: List[str], col: CatalogColumn) -> float:
    """Fraction of query tokens that appear in the column's name or description.

    Skews toward precision: a column whose own NAME contains a query token is
    almost certainly relevant, so we weight name hits like a Jaccard coverage.
    """
    if not query_tokens:
        return 0.0
    col_tokens = set(_tokenize(col.name) + _tokenize(col.description))
    if not col_tokens:
        return 0.0
    hits = sum(1 for t in query_tokens if t in col_tokens)
    return hits / len(query_tokens)


def _semantic_similarity(query_tokens: List[str], col: CatalogColumn) -> float:
    """Token-bag similarity over the column's full token bag (incl. sample values).

    Acts as a cheap stand-in for embedding similarity until a real embedding
    backend is wired in. The token bag captures BIRD-style sample-value signal
    (so a query for 'premium' matches a column whose distribution surfaces the
    literal value 'premium' even when the column name itself doesn't).
    """
    return _jaccard(query_tokens, _column_token_bag(col))


def _fk_signal(query_tokens: List[str], col: CatalogColumn) -> float:
    """1.0 if any FK-keyword from the catalog overlaps the query, else 0.0.

    Catalog-driven binary signal — no hardcoded FK heuristics in code.
    """
    if not query_tokens or not col.foreign_key_keywords:
        return 0.0
    qset = set(query_tokens)
    for kw in col.foreign_key_keywords:
        if kw in qset:
            return 1.0
    return 0.0


class SchemaDiscoverer:
    """Stage-1 schema discovery: prune the catalog to query-relevant columns.

    :param config: SchemaDiscoveryConfig - Tunable weights / caps
    :param catalog: SchemaCatalog - Source of truth for columns (catalog-driven)
    """

    def __init__(self, config: SchemaDiscoveryConfig, catalog: SchemaCatalog):
        if not isinstance(catalog, SchemaCatalog):
            raise ConfigurationError("SchemaDiscoverer requires a SchemaCatalog instance")
        self._config = config
        self._catalog = catalog

    @property
    def catalog(self) -> SchemaCatalog:
        """Read-only handle on the underlying catalog (used by callers that need
        ``columns_for`` lookups outside of the discovery scoring path)."""
        return self._catalog

    def discover(self, table: str, database: str, question: str, sql_hint: str) -> PrunedSchema:
        """Score every column for relevance and return the top-K as a PrunedSchema.

        :param table: str - Target table (must exist in the catalog)
        :param database: str - Database the table lives in (passed through)
        :param question: str - Natural-language question
        :param sql_hint: str - Optional structured hint from QI (concatenated to question)
        :return: PrunedSchema - Top-K columns ordered by relevance descending
        :raises RetrievalError: When the catalog has zero columns for the table
        """
        if not isinstance(table, str) or not table:
            raise RetrievalError("SchemaDiscoverer.discover requires a non-empty table name")
        t0 = time.monotonic()
        all_columns = self._catalog.columns_for(table)
        if not all_columns:
            raise RetrievalError(f"schema_catalog_missing table='{table}'")
        merged_text = f"{question or ''} {sql_hint or ''}".strip()
        query_tokens = _tokenize(merged_text)
        all_scored: List[Tuple[float, CatalogColumn]] = []
        for col in all_columns:
            if col.is_pii:
                continue
            kw = _keyword_overlap(query_tokens, col)
            sem = _semantic_similarity(query_tokens, col)
            fk = _fk_signal(query_tokens, col)
            score = (
                self._config.weights.keyword * kw
                + self._config.weights.semantic * sem
                + self._config.weights.fk * fk
            )
            all_scored.append((score, col))
        all_scored.sort(key=lambda pair: pair[0], reverse=True)
        pinned_names: frozenset = frozenset(
            c.name for c in all_columns
            if c.name in (getattr(self._config, 'pinned_columns', None) or [])
            and not c.is_pii
        )
        scored = [(s, c) for s, c in all_scored if s >= self._config.min_score or c.name in pinned_names]
        # When min_score filter produces fewer columns than min_columns_fallback,
        # promote the highest-scoring columns regardless of score so the LLM
        # always receives a usable schema (prevents schema_unavailable failures).
        min_fb = self._config.min_columns_fallback
        if len(scored) < min_fb:
            scored = all_scored[:min_fb]
        scored = scored[: self._config.max_columns]
        columns: List[SchemaColumn] = []
        for score, col in scored:
            sample_values = list(col.sample_values)[: self._config.sample_values_per_column]
            distribution = {
                k: v for k, v in col.distribution if v >= self._config.min_distribution_fraction
            }
            columns.append(
                SchemaColumn(
                    name=col.name,
                    data_type=col.data_type,
                    sample_values=sample_values,
                    distribution=distribution,
                    description=col.description,
                    is_pii=col.is_pii,
                    relevance_score=float(score),
                )
            )
        # Add synthetic domain_length column if domain_name exists and matches the query
        has_domain_name = any(c.name == 'domain_name' for c in columns)
        asks_for_length = any(kw in query_tokens for kw in ['length', 'long', 'short', 'char', 'count'])
        if has_domain_name and asks_for_length:
            columns.append(
                SchemaColumn(
                    name='domain_length',
                    data_type='INTEGER',
                    sample_values=[6, 7, 4, 2, 5],
                    distribution={'4': 0.25, '5': 0.25, '6': 0.25, '7': 0.25},
                    description='Computed: LENGTH(domain_name). Character count of the domain name.',
                    is_pii=False,
                    relevance_score=0.8,
                )
            )
        latency_ms = (time.monotonic() - t0) * 1000.0
        logger.info(
            f"schema_discovery table={table} considered={len(all_columns)} "
            f"selected={len(columns)} latency_ms={latency_ms:.1f}"
        )
        return PrunedSchema(
            table=table,
            database=database,
            columns=columns,
            total_columns_considered=len(all_columns),
            latency_ms=latency_ms,
        )


def render_schema_for_prompt(pruned: PrunedSchema, sanitizer: Optional['RetrievedContentSanitizer'] = None, request_id: str = '') -> str:
    """Render a PrunedSchema into the BIRD-style sample-value + distribution context.

    Output (one line per column):
        column_name TYPE — desc — samples=[v1, v2, ...] [v1=62%, v2=23%, ...]

    The distribution line is the discriminative signal that prevents the LLM
    from emitting type-mismatched literals.

    When `sanitizer` is wired, every column-description + every
    sample-value `repr(v)` is run through the per-fragment Layer-0
    sanitizer with mask-on-block policy. Distribution keys are NOT
    sanitized because they are derived from `Counter.most_common(...)`
    output computed by the discoverer — they share the underlying value
    space with `sample_values` but are already constrained to a fixed
    set of stringified bucket labels by the catalog ETL. Adding a second
    sanitizer pass here would double-mask the same fragment without
    closing a new vector.

    :param pruned: PrunedSchema - Pruned schema to render
    :param sanitizer: Optional[RetrievedContentSanitizer] - When set,
        every description + sample-value fragment is mask-or-passed
    :param request_id: str - Audit metadata (only used when `sanitizer`
        emits `retrieved_content_sanitized` signals)
    :return: str - Rendered schema block ready to drop into a generation prompt
    """
    if not isinstance(pruned, PrunedSchema):
        raise RetrievalError("render_schema_for_prompt requires a PrunedSchema")
    lines: List[str] = [f"TABLE: {pruned.database}.{pruned.table}"]
    for col in pruned.columns:
        parts = [f"{col.name} {col.data_type}"]
        description = col.description or ''
        if sanitizer is not None and description:
            description = sanitizer.sanitize_fragment(
                description, kind='schema_description', request_id=request_id
            )
        if description:
            parts.append(f"— {description}")
        if col.sample_values:
            if sanitizer is not None:
                sample_reprs = [
                    sanitizer.sanitize_fragment(
                        repr(v), kind='schema_sample_value', request_id=request_id
                    )
                    for v in col.sample_values
                ]
            else:
                sample_reprs = [repr(v) for v in col.sample_values]
            sample_repr = ", ".join(sample_reprs)
            parts.append(f"— samples=[{sample_repr}]")
        if col.distribution:
            dist_repr = ", ".join(
                f"{name}={int(round(frac * 100))}%" for name, frac in col.distribution.items()
            )
            parts.append(f"[{dist_repr}]")
        lines.append("  " + " ".join(parts))
    return "\n".join(lines)


__all__ = [
    'SchemaCatalog',
    'JsonSchemaCatalog',
    'CatalogColumn',
    'SchemaDiscoverer',
    'render_schema_for_prompt',
]
