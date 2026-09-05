"""Offline indexer — idempotent batched upsert into Qdrant.

Drives the doc-vectorization pipeline over a stream of input documents and
upserts the resulting points into the Qdrant collection backing the unified
vector + sparse + filterable index.

Design choices
--------------
- **Idempotent point IDs.** The indexer derives each point's Qdrant ID
  deterministically from a configured payload field (default ``domain``).
  Re-running the indexer over the same input produces the same point IDs
  and replaces existing points in place — re-runs are safe and cheap.
- **Stable hashing for non-numeric IDs.** Qdrant requires point IDs to be
  either non-negative integers or UUIDs. We map an arbitrary string ID
  to the deterministic UUID5(URL_NAMESPACE, id) so two runs of the
  indexer over the same input collide on the same point.
- **Batched upsert with bounded backpressure.** ``batch_size`` controls
  the number of points per Qdrant call. We process the input iterable
  lazily so the indexer scales to corpora that don't fit in memory.
- **Soft-fail on Qdrant unavailability.** When the injected
  ``QdrantClientFactory`` reports ``available=False`` (qdrant-client not
  installed, or cluster unreachable at boot), the indexer logs + skips
  upserts and reports zero ``points_upserted``. The caller decides
  whether to retry or escalate. This mirrors the registry's policy of
  soft-failing optional Qdrant subsystems so the rest of the platform
  stays operational.
- **No dense-encoder coupling.** The indexer accepts a callable
  ``dense_encoder(domain) -> List[float]`` to keep the dense backend
  swappable (FastEmbed, hashing, future ONNX). Passing ``None`` skips
  the dense leg — useful for incremental sparse-only re-encodes.
- **Layer rules.** Imports stdlib + ``core`` + sibling vectorization
  primitives + ``retrieval.qdrant_adapter`` (factory only). Never
  imports the orchestrator or registry.

Typical wiring (composition root):

    pipeline = DocVectorizationPipeline(segmenter, doc_encoder, expander)
    indexer = OfflineIndexer(
        config=cfg.vectorization.indexer,
        pipeline=pipeline,
        qdrant_factory=qdrant_factory,
        dense_encoder=fastembed_encoder.encode,
    )
    summary = await indexer.run(documents)
"""
import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterable, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from semantic_search.config.models import IndexerConfig
from semantic_search.core.exceptions import (
    ConfigurationError,
    DataIngestInterruptedError,
    RetrievalError,
    ValidationError,
)
from semantic_search.core.logging_utils import get_logger
from semantic_search.retrieval.qdrant_adapter import QdrantClientFactory
from semantic_search.vectorization.pipeline import DocVectorizationPipeline

try:
    from qdrant_client import models as _qm
except ImportError:
    _qm = None

logger = get_logger(__name__)

# A document is any payload mapping that contains the idempotency-key field
# (default ``domain``). The mapping is stored verbatim in Qdrant (modulo the
# sparse + dense vectors the indexer attaches), so callers shape it to match
# the structured-filter schema that the retriever expects.
Document = Dict[str, Any]
DocumentSource = Union[Iterable[Document], AsyncIterable[Document]]
DenseEncoder = Callable[[str], Union[List[float], Awaitable[List[float]]]]
BatchDenseEncoder = Callable[[List[str]], Union[List[List[float]], Awaitable[List[List[float]]]]]
# texts, dims -> dim -> list of vectors (one Matryoshka forward for multiple dims)
MatryoshkaBatchEncoder = Callable[[Sequence[str], Sequence[int]], Dict[int, List[List[float]]]]
# Prepared point: (id, dense, rerank_dense, sparse_idx, sparse_val, ngram_idx, ngram_val, payload)
_PreparedPoint = Tuple[
    str, Optional[List[float]], Optional[List[float]], List[int], List[float], List[int], List[float], Document,
]


@dataclass(frozen=True)
class IndexerRunSummary:
    """Outcome of one ``OfflineIndexer.run`` invocation.

    :param points_seen: int - Documents drawn from the source
    :param points_skipped: int - Documents skipped (missing key, validation)
    :param points_upserted: int - Points successfully upserted into Qdrant
    :param batches: int - Number of upsert calls issued
    :param qdrant_available: bool - Whether the Qdrant client was reachable
    :param failures: int - Number of upsert calls that raised
    :param bm25_encoded: int - Documents that produced a non-empty sparse vector
    :param bm25_skipped: int - Documents where sparse encoding yielded no tokens
    """

    points_seen: int
    points_skipped: int
    points_upserted: int
    batches: int
    qdrant_available: bool
    failures: int
    bm25_encoded: int = 0
    bm25_skipped: int = 0


class OfflineIndexer:
    """Batched, idempotent upserter for the unified Qdrant index.

    :param config: IndexerConfig - Validated batching + idempotency config
    :param pipeline: DocVectorizationPipeline - Doc-side encode pipeline
    :param qdrant_factory: QdrantClientFactory - Holds the AsyncQdrantClient
        and target collection name. May be unavailable; the indexer
        soft-fails and reports zero upserts in that case.
    :param dense_encoder: Optional[DenseEncoder] - ``None`` skips the dense
        leg. May be sync or async (the indexer awaits if a coroutine
        comes back).
    :param ngram_encoder: Optional[Any] - Character n-gram sparse encoder
        exposing ``encode(text) -> (indices, values)`` (e.g.
        ``CharNgramSparseEncoder``). When wired together with
        ``ngram_vector_name`` the indexer attaches a second sparse vector for
        the fuzzy-recall channel. ``None`` skips the ngram leg.
    :param ngram_vector_name: Optional[str] - Named sparse vector field the
        ngram vector is written under (must match the retriever's
        ``retrieval.qdrant.hybrid.ngram.vector_name``).
    :raises ValidationError: When required dependencies are missing/wrong-typed.
    """

    def __init__(  # noqa: E501
        self,
        config: IndexerConfig,
        pipeline: DocVectorizationPipeline,
        qdrant_factory: QdrantClientFactory,
        dense_encoder: Optional[DenseEncoder] = None,
        dense_dim: Optional[int] = None,
        batch_dense_encoder: Optional[BatchDenseEncoder] = None,
        rerank_dense_encoder: Optional[DenseEncoder] = None,
        rerank_batch_dense_encoder: Optional[BatchDenseEncoder] = None,
        rerank_dense_dim: Optional[int] = None,
        rerank_vector_name: Optional[str] = None,
        ngram_encoder: Optional[Any] = None,
        ngram_vector_name: Optional[str] = None,
        matryoshka_batch_encoder: Optional[MatryoshkaBatchEncoder] = None,
    ):
        if config is None or not isinstance(config, IndexerConfig):
            raise ValidationError("OfflineIndexer requires a typed IndexerConfig")
        if pipeline is None or not isinstance(pipeline, DocVectorizationPipeline):
            raise ValidationError("OfflineIndexer requires a DocVectorizationPipeline")
        if qdrant_factory is None or not isinstance(qdrant_factory, QdrantClientFactory):
            raise ValidationError("OfflineIndexer requires a QdrantClientFactory")
        if dense_encoder is not None and not callable(dense_encoder):
            raise ValidationError("OfflineIndexer.dense_encoder must be callable or None")
        if batch_dense_encoder is not None and not callable(batch_dense_encoder):
            raise ValidationError("OfflineIndexer.batch_dense_encoder must be callable or None")
        if rerank_dense_encoder is not None and not callable(rerank_dense_encoder):
            raise ValidationError("OfflineIndexer.rerank_dense_encoder must be callable or None")
        if rerank_batch_dense_encoder is not None and not callable(rerank_batch_dense_encoder):
            raise ValidationError("OfflineIndexer.rerank_batch_dense_encoder must be callable or None")
        if matryoshka_batch_encoder is not None and not callable(matryoshka_batch_encoder):
            raise ValidationError("OfflineIndexer.matryoshka_batch_encoder must be callable or None")
        if rerank_vector_name is not None and (not isinstance(rerank_vector_name, str) or not rerank_vector_name):
            raise ValidationError("OfflineIndexer.rerank_vector_name must be a non-empty string or None")
        if ngram_encoder is not None and not callable(getattr(ngram_encoder, 'encode', None)):
            raise ValidationError("OfflineIndexer.ngram_encoder must expose a callable encode(text) or be None")
        if ngram_vector_name is not None and (not isinstance(ngram_vector_name, str) or not ngram_vector_name):
            raise ValidationError("OfflineIndexer.ngram_vector_name must be a non-empty string or None")
        self._config = config
        self._pipeline = pipeline
        self._factory = qdrant_factory
        self._dense_encoder = dense_encoder
        self._batch_dense_encoder = batch_dense_encoder
        self._dense_dim = dense_dim
        self._rerank_dense_encoder = rerank_dense_encoder
        self._rerank_batch_dense_encoder = rerank_batch_dense_encoder
        self._rerank_dense_dim = rerank_dense_dim
        self._rerank_vector_name = rerank_vector_name
        self._matryoshka_batch_encoder = matryoshka_batch_encoder
        # Rerank leg is active only when a vector name, dim, and at least one
        # encoder are all wired. Otherwise the indexer writes the single dense
        # field exactly as before.
        self._rerank_enabled = (rerank_vector_name is not None and rerank_dense_dim is not None and (rerank_dense_encoder is not None or rerank_batch_dense_encoder is not None))
        self._ngram_encoder = ngram_encoder
        self._ngram_vector_name = ngram_vector_name
        # N-gram leg is active only when both the encoder and the vector name are
        # wired. Otherwise the indexer writes no ngram sparse field.
        self._ngram_enabled = (ngram_encoder is not None and ngram_vector_name is not None)
        self._shared_matryoshka_active = (
            bool(config.shared_matryoshka_dense_encode)
            and matryoshka_batch_encoder is not None
            and self._rerank_enabled
            and dense_dim is not None
            and rerank_dense_dim is not None
        )
        if config.shared_matryoshka_dense_encode and not self._shared_matryoshka_active and self._rerank_enabled:
            logger.warning(
                "indexer_shared_matryoshka_unavailable "
                "falling_back_to_dual_dense_encode "
                f"has_matryoshka_batch_encoder={matryoshka_batch_encoder is not None} "
                f"dense_dim={dense_dim} rerank_dense_dim={rerank_dense_dim}"
            )
        # Dense / BM25 sparse field names MUST match retrieval.qdrant.hybrid.* —
        # never invent names; a YAML rename otherwise yields a silent empty sparse leg.
        # Empty dense_vector_name is valid (Qdrant unnamed default vector).
        hybrid = qdrant_factory.config.hybrid
        self._dense_vector_name = str(hybrid.dense_vector_name)
        if hybrid.bm25_enabled and not str(hybrid.bm25_vector_name).strip():
            raise ConfigurationError(
                "OfflineIndexer: retrieval.qdrant.hybrid.bm25_vector_name must be "
                "non-empty when bm25_enabled=true"
            )
        self._sparse_vector_name = str(hybrid.bm25_vector_name)

    @property
    def collection_name(self) -> str:
        return self._factory.collection_name

    @staticmethod
    def _stable_point_id(raw: str) -> str:
        """Map an arbitrary string key to a deterministic UUID5.

        Qdrant accepts UUIDs as point IDs. Using URL_NAMESPACE makes the
        derivation collision-resistant in practice for domain-name inputs
        and makes re-runs idempotent.
        """
        return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))

    def _dense_texts_for_stage(self, domains: List[str]) -> List[str]:
        """Map each domain to its segmented dense-encoder input text."""
        return [self._pipeline.dense_text_for(d) for d in domains]

    async def _encode_dense(self, text: str) -> Optional[List[float]]:
        if self._dense_encoder is None:
            return None
        result = self._dense_encoder(text)
        if asyncio.iscoroutine(result):
            result = await result
        if result is None:
            return None
        return [float(x) for x in result]

    async def _encode_rerank_dense(self, text: str) -> Optional[List[float]]:
        if self._rerank_dense_encoder is None:
            return None
        result = self._rerank_dense_encoder(text)
        if asyncio.iscoroutine(result):
            result = await result
        if result is None:
            return None
        return [float(x) for x in result]

    async def _build_point(self, document: Document) -> Optional[Tuple[str, Optional[List[float]], List[int], List[float], List[int], List[float], Document]]:
        """Run pipeline over one document; return None when unindexable. Sparse + ngram encoding run in a thread pool."""
        key_field = self._config.idempotency_key_field
        raw_id = document.get(key_field)
        if raw_id is None or not isinstance(raw_id, str) or not raw_id.strip():
            return None
        point_id = self._stable_point_id(raw_id.strip().lower())
        try:
            sparse_indices, sparse_values = await asyncio.to_thread(self._pipeline.encode, raw_id)
        except ValidationError as e:
            logger.warning(f"indexer_encode_failed domain={raw_id} error_type={type(e).__name__} error={str(e)}")
            return None
        ngram_indices: List[int] = []
        ngram_values: List[float] = []
        if self._ngram_enabled:
            try:
                ngram_indices, ngram_values = await asyncio.to_thread(self._ngram_encoder.encode, raw_id)
            except ValidationError as e:
                logger.warning(f"indexer_ngram_encode_failed domain={raw_id} error_type={type(e).__name__} error={str(e)}")
                ngram_indices, ngram_values = [], []
        return point_id, None, sparse_indices, sparse_values, ngram_indices, ngram_values, document

    def _stage_docs(
        self, raw_batch: List[Document]
    ) -> List[Tuple[str, str, Document]]:
        """Filter batch to indexable docs: ``(point_id, domain_key_value, payload)``."""
        key_field = self._config.idempotency_key_field
        staged: List[Tuple[str, str, Document]] = []
        for document in raw_batch:
            raw_id = document.get(key_field)
            if raw_id is None or not isinstance(raw_id, str) or not raw_id.strip():
                continue
            domain = raw_id.strip().lower()
            staged.append((self._stable_point_id(domain), domain, document))
        return staged

    async def _encode_sparse_batch(
        self, domains: List[str]
    ) -> List[Tuple[List[int], List[float]]]:
        """Batch BM42/BM25 sparse encode with config-driven chunk concurrency."""
        if not domains:
            return []
        chunk_size = int(self._config.sparse_embed_batch_size)
        concurrency = int(self._config.sparse_encode_concurrency)
        chunks = [domains[i:i + chunk_size] for i in range(0, len(domains), chunk_size)]
        sem = asyncio.Semaphore(concurrency)

        async def _one(chunk: List[str]) -> List[Tuple[List[int], List[float]]]:
            async with sem:
                try:
                    return await asyncio.to_thread(
                        self._pipeline.encode_batch,
                        chunk,
                        sparse_embed_batch_size=chunk_size,
                    )
                except ValidationError as e:
                    logger.warning(
                        f"indexer_sparse_batch_failed chunk_size={len(chunk)} "
                        f"error_type={type(e).__name__} error={str(e)}"
                    )
                    return [([], []) for _ in chunk]

        parts = await asyncio.gather(*[_one(c) for c in chunks])
        flat: List[Tuple[List[int], List[float]]] = []
        for part in parts:
            flat.extend(part)
        return flat

    async def _encode_ngram_batch(
        self, domains: List[str]
    ) -> List[Tuple[List[int], List[float]]]:
        """Encode ngram sparse vectors for ``domains`` in one thread hop."""
        if not self._ngram_enabled or not domains:
            return [([], []) for _ in domains]

        def _run() -> List[Tuple[List[int], List[float]]]:
            out: List[Tuple[List[int], List[float]]] = []
            for domain in domains:
                try:
                    out.append(self._ngram_encoder.encode(domain))
                except ValidationError as e:
                    logger.warning(
                        f"indexer_ngram_encode_failed domain={domain} "
                        f"error_type={type(e).__name__} error={str(e)}"
                    )
                    out.append(([], []))
            return out

        return await asyncio.to_thread(_run)

    async def _encode_dense_pair(
        self, dense_texts: List[str]
    ) -> Tuple[List[Optional[List[float]]], List[Optional[List[float]]]]:
        """Encode shortlist (+ optional rerank) dense vectors for ``dense_texts``."""
        if self._shared_matryoshka_active:
            assert self._matryoshka_batch_encoder is not None
            assert self._dense_dim is not None
            assert self._rerank_dense_dim is not None
            dims = [int(self._dense_dim), int(self._rerank_dense_dim)]
            per_dim = await asyncio.to_thread(
                self._matryoshka_batch_encoder, dense_texts, dims
            )
            dense_results: List[Optional[List[float]]] = [
                [float(x) for x in v] if v is not None else None
                for v in per_dim[int(self._dense_dim)]
            ]
            rerank_results: List[Optional[List[float]]] = [
                [float(x) for x in v] if v is not None else None
                for v in per_dim[int(self._rerank_dense_dim)]
            ]
            return dense_results, rerank_results

        if self._batch_dense_encoder is not None:
            raw = self._batch_dense_encoder(dense_texts)
            if asyncio.iscoroutine(raw):
                raw = await raw
            dense_results = [
                [float(x) for x in v] if v is not None else None for v in raw
            ]
        else:
            dense_results = list(
                await asyncio.gather(*[self._encode_dense(t) for t in dense_texts])
            )
        rerank_results = await self._encode_rerank_stage(dense_texts)
        return dense_results, rerank_results

    async def _materialise_batch(self, raw_batch: List[Document]) -> Tuple[List[_PreparedPoint], int, int]:
        """Encode batch: batched sparse/ngram, then shared or dual dense encode.

        Returns ``(prepared, bm25_encoded, bm25_skipped)``.
        """
        staged = self._stage_docs(raw_batch)
        if not staged:
            return [], 0, 0

        domains = [domain for (_pid, domain, _doc) in staged]
        sparse_pairs, ngram_pairs = await asyncio.gather(
            self._encode_sparse_batch(domains),
            self._encode_ngram_batch(domains),
        )

        stage: List[Tuple[str, List[int], List[float], List[int], List[float], Document]] = []
        bm25_ok = 0
        bm25_skip = 0
        for (point_id, _domain, payload), (si, sv), (ni, nv) in zip(
            staged, sparse_pairs, ngram_pairs
        ):
            if si:
                bm25_ok += 1
            else:
                bm25_skip += 1
            stage.append((point_id, si, sv, ni, nv, payload))

        domain_key = self._config.idempotency_key_field
        # Embed the segmented natural-language form ("tech startup com") rather
        # than the raw concatenated label ("techstartup.com") so dense vectors
        # align with natural-language queries. Segmentation is pure compute;
        # to_thread keeps the event loop free for the batch.
        dense_texts: List[str] = await asyncio.to_thread(
            self._dense_texts_for_stage, [p[domain_key] for (_pid, _si, _sv, _ni, _nv, p) in stage]
        )
        dense_results, rerank_results = await self._encode_dense_pair(dense_texts)
        prepared = [
            (pid, dense, rerank_dense, si, sv, ni, nv, payload)
            for (pid, si, sv, ni, nv, payload), dense, rerank_dense in zip(stage, dense_results, rerank_results)
        ]
        return prepared, bm25_ok, bm25_skip

    async def _encode_rerank_stage(self, dense_texts: List[str]) -> List[Optional[List[float]]]:
        """Encode the rerank-dim dense vector for each text, or all-None when disabled."""
        if not self._rerank_enabled:
            return [None] * len(dense_texts)
        if self._rerank_batch_dense_encoder is not None:
            raw = self._rerank_batch_dense_encoder(dense_texts)
            if asyncio.iscoroutine(raw):
                raw = await raw
            return [[float(x) for x in v] if v is not None else None for v in raw]
        return list(await asyncio.gather(*[self._encode_rerank_dense(t) for t in dense_texts]))

    def _to_qdrant_points(self, prepared: List[_PreparedPoint]) -> List[Any]:
        """Convert prepared tuples into ``qdrant_client.models.PointStruct``."""
        if _qm is None:
            raise RetrievalError("qdrant_client is not installed; OfflineIndexer cannot build PointStruct")
        points: List[Any] = []
        dense_name = self._dense_vector_name
        sparse_name = self._sparse_vector_name
        for point_id, dense, rerank_dense, sparse_indices, sparse_values, ngram_indices, ngram_values, payload in prepared:
            vector_dict: Dict[str, Any] = {}
            if dense is not None:
                vector_dict[dense_name] = dense
            if self._rerank_enabled and rerank_dense is not None and self._rerank_vector_name:
                vector_dict[self._rerank_vector_name] = rerank_dense
            if sparse_indices:
                vector_dict[sparse_name] = _qm.SparseVector(
                    indices=sparse_indices, values=sparse_values
                )
            if self._ngram_enabled and ngram_indices and self._ngram_vector_name:
                vector_dict[self._ngram_vector_name] = _qm.SparseVector(
                    indices=ngram_indices, values=ngram_values
                )
            if not vector_dict:
                # Nothing to index — skip rather than upsert an empty point
                continue
            points.append(
                _qm.PointStruct(id=point_id, vector=vector_dict, payload=dict(payload))
            )
        return points


    async def _ensure_payload_indexes(self, client: Any) -> None:
        """Idempotent payload indexes for filter fields used by hybrid search.

        Safe on existing collections — creates any missing indexes (e.g. starting_bid
        required by starting_bid_gt_zero_baseline on every query).
        """
        if _qm is None:
            return
        _payload_schema = [
            ("domain_name",             _qm.PayloadSchemaType.KEYWORD),
            ("sld",                     _qm.PayloadSchemaType.KEYWORD),
            ("tld",                     _qm.PayloadSchemaType.KEYWORD),
            ("auction_type",            _qm.PayloadSchemaType.KEYWORD),
            ("price",                   _qm.PayloadSchemaType.FLOAT),
            ("auction_price",           _qm.PayloadSchemaType.FLOAT),
            ("current_bid_price",       _qm.PayloadSchemaType.FLOAT),
            ("starting_bid",            _qm.PayloadSchemaType.FLOAT),
            ("buy_it_now_price",        _qm.PayloadSchemaType.FLOAT),
            ("is_gem",                  _qm.PayloadSchemaType.INTEGER),
            ("unique_search_count",     _qm.PayloadSchemaType.INTEGER),
            ("govalue_score",           _qm.PayloadSchemaType.FLOAT),
            ("name_length",             _qm.PayloadSchemaType.INTEGER),
            ("quality",                 _qm.PayloadSchemaType.FLOAT),
            ("ends_at",                 _qm.PayloadSchemaType.FLOAT),
            ("bid_count",               _qm.PayloadSchemaType.INTEGER),
            ("has_hyphen",              _qm.PayloadSchemaType.INTEGER),
            ("has_number",              _qm.PayloadSchemaType.INTEGER),
            ("is_idn",                  _qm.PayloadSchemaType.INTEGER),
            ("domain_age_years",        _qm.PayloadSchemaType.INTEGER),
            ("monthly_traffic",         _qm.PayloadSchemaType.INTEGER),
            ("majestic_tf",             _qm.PayloadSchemaType.INTEGER),
            ("majestic_cf",             _qm.PayloadSchemaType.INTEGER),
            ("majestic_backlinks",      _qm.PayloadSchemaType.INTEGER),
            ("majestic_ref_domains",    _qm.PayloadSchemaType.INTEGER),
            ("tlf_exact_match",         _qm.PayloadSchemaType.INTEGER),
            ("tlf_keyword_regs",        _qm.PayloadSchemaType.INTEGER),
            ("tlf_developed",           _qm.PayloadSchemaType.INTEGER),
            ("semrush_backlinks",       _qm.PayloadSchemaType.INTEGER),
            ("semrush_indexed_pages",   _qm.PayloadSchemaType.INTEGER),
            ("semrush_ref_domains",     _qm.PayloadSchemaType.INTEGER),
            ("semrush_authority_score", _qm.PayloadSchemaType.FLOAT),
            ("semrush_search_volume",   _qm.PayloadSchemaType.INTEGER),
            ("semrush_cpc",             _qm.PayloadSchemaType.FLOAT),
            ("traffic_proxy_score",     _qm.PayloadSchemaType.FLOAT),
            ("has_web_traffic_signal",  _qm.PayloadSchemaType.INTEGER),
            ("estimated_traffic_tier",  _qm.PayloadSchemaType.INTEGER),
        ]
        await asyncio.gather(*[
            client.create_payload_index(
                collection_name=self.collection_name,
                field_name=f,
                field_schema=t,
                wait=True,
                timeout=120,
            )
            for f, t in _payload_schema
        ])
        logger.info(
            f"indexer_payload_indexes_ensured collection={self.collection_name} "
            f"fields={len(_payload_schema)}"
        )

    async def _ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not yet exist.

        Called once at the start of ``run()``. Idempotent — safe to call on
        every build. Requires ``dense_dim`` to be set; skips silently when not.
        """
        if self._dense_dim is None:
            return
        client = self._factory.client
        if client is None:
            return
        try:
            if _qm is None:
                return
            exists = await client.collection_exists(collection_name=self.collection_name)
            if exists:
                await self._ensure_payload_indexes(client)
                return
            vectors_config = {
                self._dense_vector_name: _qm.VectorParams(
                    size=self._dense_dim,
                    distance=_qm.Distance.COSINE,
                    # Store raw float32 vectors on disk; only INT8-quantized
                    # copies (always_ram=True below) live in RAM.
                    # At 20M × 256-dim: on_disk=False would require ~20 GB RAM.
                    on_disk=True,
                ),
            }
            if self._rerank_enabled:
                # Higher-fidelity rerank vector (e.g. 768). Used only by the
                # second-stage rescore; quantized + on_disk like the dense field.
                vectors_config[self._rerank_vector_name] = _qm.VectorParams(
                    size=self._rerank_dense_dim,
                    distance=_qm.Distance.COSINE,
                    on_disk=True,
                )
            sparse_vectors_config = {
                # modifier=IDF required for BM42: the model encodes only attention
                # weights; Qdrant must apply corpus-level IDF at query time.
                # Name must match retrieval.qdrant.hybrid.bm25_vector_name.
                self._sparse_vector_name: _qm.SparseVectorParams(modifier=_qm.Modifier.IDF),
            }
            if self._ngram_enabled:
                # No IDF modifier on the ngram leg: the encoder writes its own
                # sub-linear TF weights at index time; corpus IDF would skew the
                # boundary-marked character grams (which are not document terms).
                sparse_vectors_config[self._ngram_vector_name] = _qm.SparseVectorParams()
            for _attempt in range(2):
                try:
                    await client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=vectors_config,
                        sparse_vectors_config=sparse_vectors_config,
                        hnsw_config=_qm.HnswConfigDiff(
                            m=int(self._config.hnsw_m),
                            ef_construct=int(self._config.hnsw_ef_construct),
                            on_disk=True,
                        ),
                        quantization_config=_qm.ScalarQuantization(
                            scalar=_qm.ScalarQuantizationConfig(
                                type=_qm.ScalarType.INT8,
                                quantile=float(self._config.quantization_quantile),
                                always_ram=True,
                            ),
                        ),
                        optimizers_config=_qm.OptimizersConfigDiff(
                            indexing_threshold=int(self._config.optimizer_indexing_threshold),
                            memmap_threshold=int(self._config.optimizer_memmap_threshold),
                        ),
                        on_disk_payload=True,
                        timeout=120,
                    )
                    break
                except Exception as create_e:  # noqa: BLE001 — Qdrant SDK surface is unbounded
                    # Qdrant registry removed the collection (e.g. from a timed-out prior
                    # delete) but disk data still exists -> INVALID_ARGUMENT "data already
                    # exists". Force-delete with a generous timeout and retry once.
                    if _attempt == 0 and "data already exists" in str(create_e):
                        logger.warning(
                            f"indexer_collection_stale_disk collection={self.collection_name} "
                            f"forcing_delete error={create_e}"
                        )
                        await client.delete_collection(self.collection_name, timeout=120)
                        continue
                    raise
            await self._ensure_payload_indexes(client)
            logger.info(
                f"indexer_collection_created collection={self.collection_name} dense_dim={self._dense_dim}"
            )
        except Exception as e:  # noqa: BLE001 — Qdrant SDK surface is unbounded
            logger.warning(
                f"indexer_ensure_collection_failed collection={self.collection_name} "
                f"error_type={type(e).__name__} error={e}"
            )
            raise

    async def _upsert_batch(self, points: List[Any]) -> int:
        """Issue one Qdrant upsert call with retry on transient gRPC errors."""
        if not points:
            return 0
        client = self._factory.client
        if client is None:
            return 0
        last_exc: Optional[Exception] = None
        for attempt in range(3):
            try:
                await client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=bool(self._config.wait_for_index),
                    timeout=int(self._config.upsert_timeout_seconds),
                )
                return len(points)
            except Exception as exc:  # noqa: BLE001 — retry only DEADLINE/UNAVAILABLE
                last_exc = exc
                err = str(exc)
                if "DEADLINE_EXCEEDED" not in err and "UNAVAILABLE" not in err:
                    raise
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
        raise last_exc  # type: ignore[misc]

    async def _iterate(self, source: DocumentSource) -> AsyncIterable[Document]:
        """Normalise sync + async iterables into a single async stream."""
        if hasattr(source, "__aiter__"):
            async for item in source:  # type: ignore[union-attr]
                yield item
            return
        for item in source:  # type: ignore[assignment]
            yield item

    async def run(
        self,
        source: DocumentSource,
        *,
        timing: Optional[Any] = None,
    ) -> IndexerRunSummary:
        """Drain ``source`` into the Qdrant collection in ``batch_size`` chunks.

        :param source: DocumentSource - Sync or async iterable of payload dicts
        :param timing: Optional StageTimingSession - When set and
            ``log_indexer_substages`` is true, records encode vs upsert totals
        :return: IndexerRunSummary - Counts + Qdrant availability flag
        :raises ValidationError: When ``source`` is None
        """
        if source is None:
            raise ValidationError("OfflineIndexer.run requires a non-null source")

        await self._ensure_collection()

        if not self._factory.available:
            logger.warning(
                "indexer_skipped_qdrant_unavailable collection={c}".format(
                    c=self.collection_name
                )
            )
            seen = 0
            async for _ in self._iterate(source):
                seen += 1
            return IndexerRunSummary(
                points_seen=seen,
                points_skipped=seen,
                points_upserted=0,
                batches=0,
                qdrant_available=False,
                failures=0,
                bm25_encoded=0,
                bm25_skipped=0,
            )

        try:
            seen = 0
            skipped = 0
            upserted = 0
            batches = 0
            failures = 0
            bm25_encoded = 0
            bm25_skipped_count = 0
            batch_size = int(self._config.batch_size)
            encode_elapsed_s = 0.0
            upsert_elapsed_s = 0.0

            raw_batch: List[Document] = []
            async for document in self._iterate(source):
                seen += 1
                raw_batch.append(document)
                if len(raw_batch) < batch_size:
                    continue
                try:
                    _enc_t0 = time.monotonic()
                    prepared, bm25_ok, bm25_skip = await self._materialise_batch(raw_batch)
                    encode_elapsed_s += time.monotonic() - _enc_t0
                except (ValidationError, RetrievalError, OSError, RuntimeError, ValueError, TypeError) as e:
                    failures += 1
                    logger.error(
                        f"indexer_materialise_failed collection={self.collection_name} "
                        f"batch_size={len(raw_batch)} error_type={type(e).__name__} error={str(e)}"
                    )
                    raw_batch = []
                    continue
                bm25_encoded += bm25_ok
                bm25_skipped_count += bm25_skip
                skipped += len(raw_batch) - len(prepared)
                try:
                    _ups_t0 = time.monotonic()
                    points = self._to_qdrant_points(prepared)
                    upserted += await self._upsert_batch(points)
                    upsert_elapsed_s += time.monotonic() - _ups_t0
                    batches += 1
                    if batches % 10 == 0:
                        logger.info(
                            f"indexer_progress collection={self.collection_name} "
                            f"seen={seen} upserted={upserted} batches={batches} failures={failures}"
                        )
                except (RetrievalError, OSError, RuntimeError, ValueError, TypeError, TimeoutError) as e:
                    failures += 1
                    logger.error(
                        f"indexer_upsert_failed collection={self.collection_name} "
                        f"batch_size={len(prepared)} error_type={type(e).__name__} error={str(e)}"
                    )
                raw_batch = []

            if raw_batch:
                try:
                    _enc_t0 = time.monotonic()
                    prepared, bm25_ok, bm25_skip = await self._materialise_batch(raw_batch)
                    encode_elapsed_s += time.monotonic() - _enc_t0
                except (ValidationError, RetrievalError, OSError, RuntimeError, ValueError, TypeError) as e:
                    failures += 1
                    logger.error(
                        f"indexer_materialise_failed collection={self.collection_name} "
                        f"batch_size={len(raw_batch)} error_type={type(e).__name__} error={str(e)}"
                    )
                    prepared = []
                    bm25_ok = bm25_skip = 0
                bm25_encoded += bm25_ok
                bm25_skipped_count += bm25_skip
                skipped += len(raw_batch) - len(prepared)
                if prepared:
                    try:
                        _ups_t0 = time.monotonic()
                        points = self._to_qdrant_points(prepared)
                        upserted += await self._upsert_batch(points)
                        upsert_elapsed_s += time.monotonic() - _ups_t0
                        batches += 1
                    except (RetrievalError, OSError, RuntimeError, ValueError, TypeError, TimeoutError) as e:
                        failures += 1
                        logger.error(
                            f"indexer_upsert_failed collection={self.collection_name} "
                            f"batch_size={len(prepared)} error_type={type(e).__name__} error={str(e)}"
                        )

            if timing is not None:
                _sub_gate = bool(timing.config.log_indexer_substages)
                timing.record_elapsed(
                    "indexer_encode",
                    gate=_sub_gate,
                    elapsed_ms=encode_elapsed_s * 1000.0,
                    collection=self.collection_name,
                    batches=batches,
                    points_seen=seen,
                )
                timing.record_elapsed(
                    "indexer_upsert",
                    gate=_sub_gate,
                    elapsed_ms=upsert_elapsed_s * 1000.0,
                    collection=self.collection_name,
                    batches=batches,
                    points_upserted=upserted,
                )

            logger.info(
                f"indexer_run_complete collection={self.collection_name} seen={seen} "
                f"skipped={skipped} upserted={upserted} batches={batches} failures={failures} "
                f"bm25_encoded={bm25_encoded} bm25_skipped={bm25_skipped_count}"
            )
            return IndexerRunSummary(
                points_seen=seen,
                points_skipped=skipped,
                points_upserted=upserted,
                batches=batches,
                qdrant_available=True,
                failures=failures,
                bm25_encoded=bm25_encoded,
                bm25_skipped=bm25_skipped_count,
            )
        except asyncio.CancelledError as cancel_exc:
            raise DataIngestInterruptedError(
                (
                    f"Qdrant indexing interrupted after {upserted} of {seen} points "
                    f"(batches={batches}). Often ALB idle timeout or client disconnect."
                ),
                stage='qdrant_index',
                records_completed=upserted,
                records_attempted=seen,
                reason='cancelled',
                detail=f"{type(cancel_exc).__name__}: collection={self.collection_name}",
            ) from cancel_exc
