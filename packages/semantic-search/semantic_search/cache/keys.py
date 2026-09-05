"""Cache key derivation helpers.
Centralized so every tier hashes the same fields the same way.

Result caching is exact-match only — there is no embedding/paraphrase result
tier. The sole semantic (embedding) cache is the QI intent cache
(``QISemanticIntentCache``), which keys on its own query embedding and is not
a result cache.

Tier-to-function mapping:
  Exact result tier: exact_query_key(normalized_query)  # case-insensitive
  Structured tier: structured_intermediate_key(query_type, filters)
  Intent-plan tier: intent_plan_key(intent)
  Intent / L0 extract: versioned_query_key(normalized, prompt_tag, schema_version)
"""
import hashlib
from typing import Any, Dict

from semantic_search.cache.intent_fingerprint import _stable_dumps, intent_plan_fingerprint
from semantic_search.contracts import QueryIntent
from semantic_search.core.exceptions import CacheError


def versioned_query_key(normalized_query: str, *, prompt_tag: str, schema_version: str) -> str:
    """Compose a cache key that includes prompt and schema version.

    All three components are required and must be non-empty. Used by the
    intent-result cache and the qie_only L0 filter cache so a prompt or
    schema bump isolates entries from prior versions.

    :param normalized_query: str - Normalized query text
    :param prompt_tag: str - Prompt version tag from config
    :param schema_version: str - Response/schema version from config
    :return: str - Composite cache key
    :raises CacheError: When any component is empty or not a string
    """
    if not isinstance(normalized_query, str) or not normalized_query:
        raise CacheError("versioned_query_key requires non-empty normalized_query")
    if not isinstance(prompt_tag, str) or not prompt_tag.strip():
        raise CacheError("versioned_query_key requires non-empty prompt_tag")
    if not isinstance(schema_version, str) or not schema_version.strip():
        raise CacheError("versioned_query_key requires non-empty schema_version")
    return f"{prompt_tag.strip()}\x1f{schema_version.strip()}\x1f{normalized_query}"


def exact_query_key(normalized_query: str) -> str:
    """Hash the normalized query for the exact-hash result cache.
    Case-folds and strips at the key boundary so the exact result cache is
    case-insensitive regardless of caller normalization (idempotent for the
    already-lowercased output of ``normalize_query``).
    :param normalized_query: str - Normalized query (lowercased upstream)
    :return: str - Hex digest
    """
    return hashlib.sha256(normalized_query.strip().lower().encode('utf-8')).hexdigest()


def structured_intermediate_key(query_type: str, filters: Dict[str, Any]) -> str:
    """Hash query_type + filter dict for the structured intermediate cache (Tier 3).
    :param query_type: str - Primary query_type from QueryIntent
    :param filters: Dict[str, Any] - Filter projection from `extract_filters_from_intent`
    :return: str - Hex digest
    """
    canonical_filters: Dict[str, Any] = {}
    for k in sorted(filters.keys()):
        v = filters[k]
        if isinstance(v, list):
            canonical_filters[k] = sorted([str(x) for x in v])
        else:
            canonical_filters[k] = v
    payload = {'query_type': query_type, 'filters': canonical_filters}
    return hashlib.sha256(_stable_dumps(payload).encode('utf-8')).hexdigest()


def intent_plan_key(intent: QueryIntent) -> str:
    """Derive the cache key for the intent-plan tier.

    Delegates to ``intent_plan_fingerprint`` — the structural SHA-256 of
    the ``QueryIntent`` (excludes raw query text so same-intent /
    different-wording queries collide by design).

    :param intent: QueryIntent - Post-classification intent
    :return: str - 64-character hex digest
    """
    return intent_plan_fingerprint(intent)
