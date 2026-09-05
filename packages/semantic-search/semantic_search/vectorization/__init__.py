"""Domain-name vectorization primitives + offline pipeline composition.

Five symbols live here:

- ``DomainNameSegmenter`` / ``SegmentedDomain`` — split a registered
  domain into the registrable label, subdomain labels (rare in the
  auctions corpus but honoured), and a TLD-aware token stream that
  surfaces ASCII alphabetic segments, numeric segments, and the TLD as a
  first-class token. Handles ASCII-only labels, mixed alphanumeric
  labels (``cloud9``), hyphenated labels (``foo-bar``), and Unicode/IDN
  TLDs (``.中国``, punycoded ``.xn--fiqs8s``) via stdlib
  ``unicodedata`` + ``codecs``.

- ``BM25DocEncoder`` — corpus-side counterpart to
  ``retrieval.bm25_query_encoder.BM25QueryEncoder``. Produces sparse
  vectors keyed on the same ``vocab_size`` + ``sha1`` hashing scheme so
  query and document vectors line up at retrieval time.

- ``DocVectorizationPipeline`` — composes the segmenter, optional
  ``SynonymExpander`` (kept symmetric with the query side via the
  config-layer cross-field check), and ``BM25DocEncoder`` into a single
  per-document encode call.

- ``OfflineIndexer`` / ``IndexerRunSummary`` — drives the pipeline over
  a stream of input documents and idempotently upserts them into the
  Qdrant collection backing the unified vector + sparse + filterable
  index. Soft-fails when Qdrant is unavailable.

- ``VectorRefreshDriver`` — periodically polls the shared
  ``SnapshotVersionRegistry`` and triggers an indexer run whenever the
  inventory snapshot version advances. Lifecycle (``start`` / ``stop``)
  mirrors ``LLMProvider.start_background_refresh`` so the registry can
  manage it from the FastAPI lifespan.

These primitives are stateless and dependency-light so they reuse cleanly
from offline indexer jobs, ad-hoc backfills, and integration tests
without dragging in the full search runtime.
"""

from semantic_search.vectorization.bm25_doc_encoder import BM25DocEncoder
from semantic_search.vectorization.compound_splitter import CompoundWordSplitter, SplitResult
from semantic_search.vectorization.delta_driver import DeltaRefreshDriver
from semantic_search.vectorization.enrichment_refresh_driver import EnrichmentRefreshDriver
from semantic_search.vectorization.event_ingest_driver import EventIngestDriver
from semantic_search.vectorization.indexer import IndexerRunSummary, OfflineIndexer
from semantic_search.vectorization.pipeline import DocVectorizationPipeline
from semantic_search.vectorization.refresh_driver import VectorRefreshDriver
from semantic_search.vectorization.segmenter import DomainNameSegmenter, SegmentedDomain

__all__ = [
    "BM25DocEncoder",
    "CompoundWordSplitter",
    "DeltaRefreshDriver",
    "DocVectorizationPipeline",
    "DomainNameSegmenter",
    "EnrichmentRefreshDriver",
    "EventIngestDriver",
    "IndexerRunSummary",
    "OfflineIndexer",
    "SegmentedDomain",
    "SplitResult",
    "VectorRefreshDriver",
]
