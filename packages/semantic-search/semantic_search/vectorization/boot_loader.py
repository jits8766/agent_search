"""Boot-time seed loader for in-memory retrieval indexes.

Populates ``InMemoryVectorIndex`` and ``InMemoryStructuredIndex`` from a list
of domain payload dicts (fetched from ``{vectorization.seed.database.name}.auction_audit_cln``).

Per-document encoding runs in two sequential stages:

1. **Dense** — ``encoder.encode_batch(domains)`` encodes ``encode_batch_size``
   domains per call (one ONNX session.run() per chunk) instead of one
   ``encoder.encode(domain)`` per document.  Falls back to per-doc
   ``encode()`` when the encoder does not expose ``encode_batch``.
   Results are stored in ``InMemoryVectorIndex``.

2. **BM25 sparse + synonym expansion** — ``pipeline.encode(domain)`` runs
   ``DomainNameSegmenter`` → ``SynonymExpander`` → ``BM25DocEncoder``,
   producing ``(sparse_indices, sparse_values)``.  ``InMemoryVectorIndex``
   is dense-only so sparse vectors are not stored there; instead they are
   attached to the structured payload as ``_sparse_indices`` /
   ``_sparse_values`` so the ``OfflineIndexer`` (Qdrant upsert path) can
   read them directly from the same payload dict without re-encoding.
   When ``pipeline`` is ``None`` this stage is skipped entirely.

The loader is async (yields to the event loop on ``batch_yield_size``
boundaries and once per encode chunk), idempotent (re-runs replace
existing points), and produces a ``BootLoadSummary`` for ops logging and
endpoint responses.

Layer rules: imports stdlib + ``core`` + sibling vectorization + retrieval
index interfaces. Never imports registry or orchestrator.
"""
import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from semantic_search.core.exceptions import ValidationError
from semantic_search.core.logging_utils import get_logger
from semantic_search.vectorization.pipeline import DocVectorizationPipeline

logger = get_logger(__name__)


@dataclass(frozen=True)
class BootLoadSummary:
    """Outcome of one ``load_seed_into_indexes`` invocation.

    :param documents_offered: int - Documents presented to the loader
    :param vector_indexed: int - Documents with dense vectors written to the
        vector index
    :param structured_indexed: int - Documents written to the structured index
    :param bm25_encoded: int - Documents whose BM25 sparse vector was produced
        (pipeline stage; 0 when ``pipeline`` is None)
    :param bm25_skipped: int - Documents where BM25 encoding raised (domain
        not encodeable by the segmenter/BM25 encoder)
    :param skipped: int - Documents skipped before any encoding (missing
        idempotency-key field or dense encode failure)
    :param elapsed_ms: float - Wall-clock load time
    :param explore_trending_added: int - Documents added to the trending source
    :param explore_ending_soon_added: int - Documents added to the ending-soon source
    """
    documents_offered: int
    vector_indexed: int
    structured_indexed: int
    bm25_encoded: int
    bm25_skipped: int
    skipped: int
    elapsed_ms: float
    explore_trending_added: int = 0
    explore_ending_soon_added: int = 0


async def load_seed_into_indexes(documents: List[Dict[str, Any]], vector_index: Any, structured_index: Any, encoder: Any, *, trending_source: Any = None, ending_soon_source: Any = None, pipeline: Optional[DocVectorizationPipeline] = None, idempotency_key: str, batch_yield_size: int, encode_batch_size: int) -> BootLoadSummary:  # noqa: E501
    """Load seed documents into in-memory vector and structured indexes.

    For each document the loader runs two encoding stages sequentially:

    * **Dense** (always) — ``encoder.encode_batch(domains)`` encodes
      ``encode_batch_size`` domains per call (one ONNX session.run() per
      chunk) instead of one ``encode()`` per document.  Falls back to
      per-doc ``encode()`` when the encoder does not expose
      ``encode_batch``.  Results are stored in ``vector_index``.
    * **BM25 sparse + synonyms** (when ``pipeline`` is not None) —
      ``pipeline.encode(domain)`` runs segment → synonym-expand → BM25.
      Sparse vectors are attached to the structured payload as
      ``_sparse_indices`` / ``_sparse_values`` so the ``OfflineIndexer``
      (Qdrant path, triggered separately by the endpoint) can upsert them
      without re-encoding.

    :param documents: List[Dict] - Domain payload dicts from ``db_seed_source``
    :param vector_index: VectorIndex - In-memory dense vector index
        (must support ``.add(item_id, vector, payload)``)
    :param structured_index: StructuredIndex - In-memory structured index
        (must support ``.add(payload)``)
    :param encoder: Encoder - Dense encoder; exposes ``encode_batch(domains)``
        or falls back to ``encode(domain)``
    :param pipeline: Optional[DocVectorizationPipeline] - Full doc-side
        pipeline (segment → synonym-expand → BM25).  ``None`` skips the
        sparse stage.
    :param idempotency_key: str - Payload field containing the domain string
        (``"domain"`` by default, from ``vectorization.indexer.idempotency_key_field``)
    :param batch_yield_size: int - Yield to the event loop every N documents
    :param encode_batch_size: int - Documents encoded per ``encode_batch()``
        call. Larger values improve ONNX throughput.
    :return: BootLoadSummary
    :raises ValidationError: When required parameters are invalid
    """
    if not isinstance(documents, list):
        raise ValidationError("boot_loader.documents must be a list")
    if not isinstance(idempotency_key, str) or len(idempotency_key) == 0:
        raise ValidationError("boot_loader.idempotency_key must be a non-empty string")
    if batch_yield_size < 1:
        raise ValidationError("boot_loader.batch_yield_size must be >= 1")
    if encode_batch_size < 1:
        raise ValidationError("boot_loader.encode_batch_size must be >= 1")

    start = time.monotonic()
    vector_count = 0
    structured_count = 0
    bm25_count = 0
    bm25_skip = 0
    skipped = 0
    explore_trending_count = 0
    explore_ending_soon_count = 0

    # One encode_batch() per chunk instead of one encode() per doc.
    # Reduces ONNX inference overhead ~10-50× on CPU. Falls back to per-doc
    # when the encoder does not expose encode_batch (e.g. a bare Encoder stub).
    _has_batch_encode = (
        vector_index is not None
        and hasattr(encoder, 'encode_batch')
        and callable(encoder.encode_batch)
    )

    for chunk_start in range(0, len(documents), encode_batch_size):
        chunk = documents[chunk_start:chunk_start + encode_batch_size]

        # ── Batch-encode dense vectors for the whole chunk ────────────────────
        # Collect valid (local_j, domain) pairs, call encode_batch once, store
        # results in chunk_vecs[j]. Failed docs receive None and are skipped.
        chunk_vecs: Dict[int, Optional[List[float]]] = {}
        if _has_batch_encode:
            valid_pairs: List[Tuple[int, str]] = [
                (j, doc[idempotency_key])
                for j, doc in enumerate(chunk)
                if doc.get(idempotency_key)
                and isinstance(doc.get(idempotency_key), str)
                and len(str(doc.get(idempotency_key, ""))) > 0
            ]
            if valid_pairs:
                idxs = [x[0] for x in valid_pairs]
                # Embed the segmented natural-language form instead of the raw
                # concatenated label so dense vectors align with NL queries.
                doms = [_dense_input(x[1], pipeline) for x in valid_pairs]
                try:
                    vecs = encoder.encode_batch(doms)
                    for j, v in zip(idxs, vecs):
                        chunk_vecs[j] = v if (v is not None and len(v) > 0) else None
                except Exception as _be:
                    logger.warning(
                        f"boot_load_batch_encode_failed chunk_start={chunk_start} "
                        f"error_type={type(_be).__name__} error={_be} — per-doc fallback"
                    )
                    for j, dom in valid_pairs:
                        try:
                            v = encoder.encode(_dense_input(dom, pipeline))
                            chunk_vecs[j] = v if (v is not None and len(v) > 0) else None
                        except Exception as _enc_e:
                            logger.warning(f"boot_load_per_doc_encode_failed domain={dom[:50]} error_type={type(_enc_e).__name__}")
                            chunk_vecs[j] = None

        # ── Pre-compute sparse encodings for entire chunk in parallel ─────────
        # CPU-bound (tokenize → synonym-expand → BM25); to_thread releases GIL
        # so N encoders can run concurrently across the thread pool.
        _chunk_sparse: Dict[int, Tuple[List[int], List[float]]] = {}
        if pipeline is not None:
            _sparse_jobs = [
                (j, doc.get(idempotency_key))
                for j, doc in enumerate(chunk)
                if doc.get(idempotency_key) and isinstance(doc.get(idempotency_key), str)
            ]
            if _sparse_jobs:
                _sparse_results = await asyncio.gather(*[
                    asyncio.to_thread(_encode_sparse, dom, pipeline) for _, dom in _sparse_jobs
                ])
                for (j, _), (si, sv) in zip(_sparse_jobs, _sparse_results):
                    _chunk_sparse[j] = (si, sv)

        # ── Process each document in the chunk ───────────────────────────────
        for j, doc in enumerate(chunk):
            i = chunk_start + j
            domain = doc.get(idempotency_key)
            if domain is None or not isinstance(domain, str) or len(domain) == 0:
                skipped += 1
                continue
            item_id = doc.get("item_id", domain)

            # ── Stage 1: dense vector ────────────────────────────────────────
            if vector_index is not None:
                if _has_batch_encode:
                    dense_vec = chunk_vecs.get(j)
                    if dense_vec is None:
                        logger.warning(
                            f"boot_load_dense_skip domain={domain[:50]} error_type=BatchEncodeFailed"
                        )
                        skipped += 1
                        continue
                else:
                    try:
                        dense_vec = encoder.encode(_dense_input(domain, pipeline))
                        if dense_vec is None or len(dense_vec) == 0:
                            skipped += 1
                            continue
                    except Exception as e:
                        logger.warning(
                            f"boot_load_dense_skip domain={domain[:50]} error_type={type(e).__name__}"
                        )
                        skipped += 1
                        continue
                try:
                    vector_index.add(item_id=str(item_id), vector=dense_vec, payload=dict(doc))
                    vector_count += 1
                except Exception as e:
                    logger.warning(
                        f"boot_load_dense_skip domain={domain[:50]} error_type={type(e).__name__}"
                    )
                    skipped += 1
                    continue

            # ── Explore sources (trending + ending-soon) ─────────────────────
            # Populate in-memory explore sources so the zero-result guard
            # fallback rail has real corpus data to return. Duck-typed:
            # ClickHouse sources have no .add(), so the hasattr guards skip them.
            if trending_source is not None and hasattr(trending_source, 'add'):
                try:
                    _tscore = float(doc.get('score') or doc.get('govalue_score') or 0.0)
                    trending_source.add(item_id=str(item_id), trending_score=_tscore, payload=dict(doc))
                    explore_trending_count += 1
                except Exception as _te:
                    logger.warning(f"boot_load_trending_skip domain={domain[:50]} error_type={type(_te).__name__}")

            if ending_soon_source is not None and hasattr(ending_soon_source, 'add'):
                _ends_at = _extract_ends_at(doc)
                if _ends_at is not None and _ends_at > time.time():
                    try:
                        ending_soon_source.add(item_id=str(item_id), ends_at=_ends_at, payload=dict(doc))
                        explore_ending_soon_count += 1
                    except Exception as _ese:
                        logger.warning(f"boot_load_ending_soon_skip domain={domain[:50]} error_type={type(_ese).__name__}")

            # ── Stage 2: BM25 sparse + synonym expansion ─────────────────────
            # InMemoryVectorIndex is dense-only; sparse vectors are attached to
            # the structured payload so the OfflineIndexer (Qdrant path) can
            # read them from the same dict without re-running the pipeline.
            structured_payload = dict(doc)
            structured_payload.setdefault("item_id", str(item_id))

            if pipeline is not None:
                _sp = _chunk_sparse.get(j, ([], []))
                sparse_indices, sparse_values = _sp[0], _sp[1]
                if sparse_indices:
                    structured_payload["_sparse_indices"] = sparse_indices
                    structured_payload["_sparse_values"] = sparse_values
                    bm25_count += 1
                else:
                    bm25_skip += 1

            if structured_index is not None:
                try:
                    structured_index.add(structured_payload)
                    structured_count += 1
                except Exception as e:
                    logger.warning(
                        f"boot_load_structured_skip domain={domain[:50]} error_type={type(e).__name__}"
                    )

            if (i + 1) % batch_yield_size == 0:
                await asyncio.sleep(0)

        await asyncio.sleep(0)  # yield to event loop after each encode chunk

    elapsed = (time.monotonic() - start) * 1000
    summary = BootLoadSummary(
        documents_offered=len(documents),
        vector_indexed=vector_count,
        structured_indexed=structured_count,
        bm25_encoded=bm25_count,
        bm25_skipped=bm25_skip,
        skipped=skipped,
        elapsed_ms=round(elapsed, 1),
        explore_trending_added=explore_trending_count,
        explore_ending_soon_added=explore_ending_soon_count,
    )
    logger.info(
        f"boot_seed_loaded vector_indexed={vector_count} "
        f"structured_indexed={structured_count} "
        f"bm25_encoded={bm25_count} bm25_skipped={bm25_skip} "
        f"skipped={skipped} total={len(documents)} elapsed_ms={summary.elapsed_ms} "
        f"explore_trending={explore_trending_count} explore_ending_soon={explore_ending_soon_count}"
    )
    return summary


def _extract_ends_at(doc: Dict[str, Any]) -> Optional[float]:
    """Extract auction end time as a Unix timestamp from a doc payload.

    Checks ``ends_at`` (numeric) first, then ``end_time`` (VARCHAR datetime
    from ``CAST(auctionendtime AS VARCHAR)``).  Returns None when neither
    field is present or parseable.
    """
    v = doc.get('ends_at')
    if v is not None:
        try:
            return float(v)
        except (ValueError, TypeError):
            pass
    v = doc.get('end_time')
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%SZ'):
            try:
                dt = datetime.strptime(v, fmt)
                return dt.replace(tzinfo=timezone.utc).timestamp()
            except ValueError:
                continue
    return None


def _dense_input(domain: str, pipeline: Optional[DocVectorizationPipeline]) -> str:
    """Dense-encoder input text for ``domain``.
    Returns the segmented natural-language form ("tech startup com") when a
    pipeline is wired so dense vectors align with natural-language queries;
    falls back to the raw domain when no pipeline is available or when the
    segmenter rejects the input (e.g. a dotless label).
    :param domain: str - Raw domain string (caller guarantees non-empty)
    :param pipeline: Optional[DocVectorizationPipeline] - Doc-side pipeline
    :return: str - Segmented dense text, or the raw domain when pipeline is None
    """
    if pipeline is None:
        return domain
    try:
        return pipeline.dense_text_for(domain)
    except ValidationError as e:
        logger.warning(f"boot_load_dense_text_fallback domain={domain[:50]} error_type={type(e).__name__}")
        return domain


def _encode_sparse(domain: str, pipeline: DocVectorizationPipeline) -> Tuple[List[int], List[float]]:
    """Run the BM25 sparse + synonym pipeline for one domain string.

    Returns ``([], [])`` on any encode failure so callers can treat an
    empty pair as "no sparse leg" without raising.

    :param domain: str - Raw domain string (e.g. ``"cars.com"``)
    :param pipeline: DocVectorizationPipeline - Wired pipeline instance
    :return: Tuple[List[int], List[float]] - (indices, values) or ([], [])
    """
    try:
        return pipeline.encode(domain)
    except Exception as e:
        logger.warning(
            f"boot_load_bm25_skip domain={domain[:50]} error_type={type(e).__name__}"
        )
        return [], []
