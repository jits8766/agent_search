"""SQL price-band fallback retriever.
Ships a `PriceBandStore` interface plus an in-memory implementation. The public
boundary uses *parameterized* filter dicts (never string concatenation) and
rejects any column not in the configured allowlist before issuing the lookup.

The ``PriceBandStore.lookup`` contract is ``async`` — every production
implementation talks to a network backend (ClickHouse, Athena), so the seam
is async at the interface to keep the orchestrator free of ``to_thread``
wrappers. The in-memory store satisfies the same async contract for
tests and local bootstrapping.
"""
import time
from typing import Any, Dict, List

from semantic_search.config.models import SqlRetrievalConfig
from semantic_search.core.exceptions import RetrievalError
from semantic_search.core.logging_utils import get_logger
from semantic_search.retrieval.base import Retriever, slice_candidates
from semantic_search.retrieval.structured_retriever import extract_filters_from_intent, item_matches_filters
from semantic_search.contracts import Candidate, CandidateSet, QueryIntent

logger = get_logger(__name__)


class PriceBandStore:
    """SQL-style price-band lookup interface (async).

    Production implementations talk to a network backend (e.g. ClickHouse
    ``events_raw``). The interface is async at the boundary so the
    orchestrator can ``await`` directly — no ``to_thread`` shim required.
    """

    async def lookup(self, filters: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
        """Return up to top_k items matching the filter dict (each row contains 'item_id' + 'score')."""
        raise NotImplementedError


class InMemoryPriceBandStore(PriceBandStore):
    """In-memory price-band store for tests and local bootstrapping."""

    def __init__(self):
        self._items: List[Dict[str, Any]] = []

    def add(self, item: Dict[str, Any]) -> None:
        """Add an item with at minimum 'item_id', 'price', and 'score' fields."""
        for key in ('item_id', 'price', 'score'):
            if key not in item:
                raise RetrievalError(f"InMemoryPriceBandStore item missing '{key}'")
        try:
            score = float(item['score'])
        except (TypeError, ValueError) as e:
            raise RetrievalError(f"InMemoryPriceBandStore item score not numeric: {e}") from e
        if not 0.0 <= score <= 1.0:
            raise RetrievalError("InMemoryPriceBandStore item score must be in [0,1]")
        self._items.append(dict(item))

    async def lookup(self, filters: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
        if top_k < 1:
            return []
        out = [item for item in self._items if item_matches_filters(item, filters)]
        out.sort(key=lambda it: float(it.get('score', 0.0)), reverse=True)
        return out[:top_k]


class SqlRetriever(Retriever):
    """Price-band SQL fallback retriever.
    Only runs when the intent has at least one of {price_min, price_max} extracted.
    :param config: SqlRetrievalConfig - Tier-specific config (allowlist + top_k)
    :param store: PriceBandStore - Backing price-band store
    """

    def __init__(self, config: SqlRetrievalConfig, store: PriceBandStore):
        self._config = config
        self._store = store
        self._allowed = frozenset(config.allowed_filter_columns)

    @property
    def source(self) -> str:
        return 'sql'

    def _validated_filters(self, intent: QueryIntent) -> Dict[str, Any]:
        """Project intent entities to filters and reject any column outside the allowlist."""
        raw = extract_filters_from_intent(intent)
        validated: Dict[str, Any] = {}
        for key, value in raw.items():
            if key not in self._allowed:
                logger.warning(f"sql_retrieval_rejected_column request_id={intent.request_id} column={key}")
                continue
            validated[key] = value
        return validated

    async def retrieve(self, intent: QueryIntent, top_k: int) -> CandidateSet:
        if not self._config.enabled:
            return CandidateSet(source='sql', candidates=[], latency_ms=0.0)
        if top_k < 1:
            return CandidateSet(source='sql', candidates=[], latency_ms=0.0)
        filters = self._validated_filters(intent)
        if not filters:
            return CandidateSet(source='sql', candidates=[], latency_ms=0.0)
        # Price-band fan-out only — tld/auction_type alone already hit Qdrant filters.
        # Running CH SQL without a price bound burns ~1-1.5s and pollutes RRF.
        if 'price_min' not in filters and 'price_max' not in filters:
            logger.debug(
                f"sql_retrieval_skipped_no_price request_id={intent.request_id} "
                f"filters={sorted(filters.keys())}"
            )
            return CandidateSet(source='sql', candidates=[], latency_ms=0.0)
        t0 = time.monotonic()
        rows: List[Dict[str, Any]] = await self._store.lookup(filters, top_k)
        candidates: List[Candidate] = []
        for row in rows:
            payload = {k: v for k, v in row.items() if k not in {'item_id', 'score'}}
            candidates.append(Candidate(item_id=str(row['item_id']), score=float(row['score']), source='sql', payload=payload))
        candidates = slice_candidates(candidates, top_k)
        latency_ms = (time.monotonic() - t0) * 1000.0
        logger.info(f"sql_retrieval request_id={intent.request_id} candidates={len(candidates)} filters={sorted(filters.keys())} latency_ms={latency_ms:.1f}")
        return CandidateSet(source='sql', candidates=candidates, latency_ms=latency_ms)
