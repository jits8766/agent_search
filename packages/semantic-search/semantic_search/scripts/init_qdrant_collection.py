"""Create the Qdrant collection for semantic_search with the correct vector schema.

Run once before the first indexing pass. Idempotent: if the collection
already exists the script exits with a warning (does not drop/recreate).

Usage (from repo root, rag venv active)::

    python -m semantic_search.scripts.init_qdrant_collection
    python -m semantic_search.scripts.init_qdrant_collection --host qdrant --port 6333

The collection is created with:
  - Named dense vector ``"dense"`` (cosine distance) sized to the cascade
    shortlist stage (stage_dims.shortlist = 512). The dimension is
    read from base.yaml so the index can never drift from the encoder.
  - Named sparse vector ``"sparse"`` with IDF modifier for BM42 term-weight
    vectors (modifier=IDF required: model encodes attention weights only,
    Qdrant applies corpus-level IDF at query time).
  - HNSW m=16, ef_construct=200 for build quality.
  - INT8 scalar quantization (always_ram=True) — ~4x memory reduction with
    <1% recall loss; quantile=0.99 clips outliers.
  - on_disk_payload=True — payload stored on SSD, vectors in RAM.
  - Payload indexes on every field used by the filter path:
      tld, auction_type        (KEYWORD — MatchAny)
      price                    (FLOAT   — Range, used by price_min/price_max)
      name_length              (INTEGER — Range, used by name_length_max)
      quality                  (FLOAT   — Range, used by quality_min)
      ends_at                  (FLOAT   — Range, used by time_remaining_max)
      domain_name              (KEYWORD — item-id lookup / dedup)
      buy_it_now, has_reserve_price, gd_transfer (INTEGER — MatchValue booleans)
"""

import argparse
import asyncio
from typing import Optional

from qdrant_client import AsyncQdrantClient, models as qm

from semantic_search.config.loader import load_config


DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

# Payload indexes: must match build_qdrant_filter keys (tld, price, name_length, ends_at, etc)
PAYLOAD_INDEXES = [
    # Listing fields
    ("domain_name", qm.PayloadSchemaType.KEYWORD),
    ("sld", qm.PayloadSchemaType.KEYWORD),
    ("tld", qm.PayloadSchemaType.KEYWORD),
    ("auction_type", qm.PayloadSchemaType.KEYWORD),
    ("price", qm.PayloadSchemaType.FLOAT),
    # Baseline on every hybrid query when starting_bid_gt_zero_baseline=true.
    ("starting_bid", qm.PayloadSchemaType.FLOAT),
    ("buy_it_now_price", qm.PayloadSchemaType.FLOAT),
    ("is_gem", qm.PayloadSchemaType.INTEGER),
    ("unique_search_count", qm.PayloadSchemaType.INTEGER),
    ("govalue_score", qm.PayloadSchemaType.FLOAT),
    ("name_length", qm.PayloadSchemaType.INTEGER),
    ("quality", qm.PayloadSchemaType.FLOAT),
    ("ends_at", qm.PayloadSchemaType.FLOAT),
    ("bid_count", qm.PayloadSchemaType.INTEGER),
    # Character
    ("has_hyphen", qm.PayloadSchemaType.INTEGER),
    ("has_number", qm.PayloadSchemaType.INTEGER),
    ("is_idn", qm.PayloadSchemaType.INTEGER),
    # Boolean auction fields (must match _AUCTION_BOOLEAN_PAYLOAD_FIELDS in qdrant_adapter.py).
    ("buy_it_now", qm.PayloadSchemaType.INTEGER),
    ("has_reserve_price", qm.PayloadSchemaType.INTEGER),
    ("gd_transfer", qm.PayloadSchemaType.INTEGER),
    # Provenance
    ("domain_age_years", qm.PayloadSchemaType.INTEGER),
    ("monthly_traffic", qm.PayloadSchemaType.INTEGER),
    # Majestic
    ("majestic_tf", qm.PayloadSchemaType.INTEGER),
    ("majestic_cf", qm.PayloadSchemaType.INTEGER),
    ("majestic_backlinks", qm.PayloadSchemaType.INTEGER),
    ("majestic_ref_domains", qm.PayloadSchemaType.INTEGER),
    # TLF
    ("tlf_exact_match", qm.PayloadSchemaType.INTEGER),
    ("tlf_keyword_regs", qm.PayloadSchemaType.INTEGER),
    ("tlf_developed", qm.PayloadSchemaType.INTEGER),
    # SEMrush
    ("semrush_backlinks", qm.PayloadSchemaType.INTEGER),
    ("semrush_indexed_pages", qm.PayloadSchemaType.INTEGER),
    ("semrush_ref_domains", qm.PayloadSchemaType.INTEGER),
    ("semrush_authority_score", qm.PayloadSchemaType.FLOAT),
    ("semrush_search_volume", qm.PayloadSchemaType.INTEGER),
    ("semrush_cpc", qm.PayloadSchemaType.FLOAT),
]


async def create_collection(
    host: str,
    port: int,
    collection_name: str,
    dense_dim: int,
    ngram_vector_name: Optional[str] = None,
) -> None:
    """Create the Qdrant collection and payload indexes.
    :param host: str - Qdrant host
    :param port: int - Qdrant REST port
    :param collection_name: str - Collection name
    :param dense_dim: int - Dense vector dimension (must match cascade.stage_dims.shortlist)
    :param ngram_vector_name: Optional[str] - When set, add a second sparse vector
        (no IDF modifier) for the character n-gram fuzzy-recall channel
    """
    client = AsyncQdrantClient(host=host, port=port)

    existing = await client.collection_exists(collection_name=collection_name)
    if existing:
        info = await client.get_collection(collection_name=collection_name)
        print(
            f"Collection '{collection_name}' already exists (points={info.points_count}). Skipping creation."
        )
        await client.close()
        return

    sparse_vectors_config = {
        # modifier=IDF required for BM42: attention-weight model; Qdrant
        # applies corpus IDF at query time (not doc time).
        SPARSE_VECTOR_NAME: qm.SparseVectorParams(
            modifier=qm.Modifier.IDF,
        ),
    }
    if ngram_vector_name:
        # No IDF modifier: the CharNgramSparseEncoder writes its own sub-linear
        # TF weights at index time; corpus IDF would skew the character grams.
        sparse_vectors_config[ngram_vector_name] = qm.SparseVectorParams()

    await client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: qm.VectorParams(
                size=dense_dim,
                distance=qm.Distance.COSINE,
                # Raw float32 vectors on disk; INT8-quantized copy (always_ram=True)
                # stays in RAM for ANN. At 20M × 512-dim on_disk=False = ~40 GB RAM.
                on_disk=True,
            ),
        },
        sparse_vectors_config=sparse_vectors_config,
        hnsw_config=qm.HnswConfigDiff(
            m=16,
            ef_construct=200,
            # HNSW graph on disk; pages in on demand. The graph for 20M nodes
            # at m=16 occupies several GB if held entirely in RAM.
            on_disk=True,
        ),
        quantization_config=qm.ScalarQuantization(
            scalar=qm.ScalarQuantizationConfig(
                type=qm.ScalarType.INT8,
                quantile=0.99,
                always_ram=True,  # quantized vectors always in RAM - fast ANN
            ),
        ),
        # Raise indexing threshold to avoid re-building HNSW after every small
        # segment during a 20M-record bulk load (default 20 000 would trigger
        # ~1 000 reindex cycles). Tune back down after the initial load if
        # lower write latency matters more than throughput.
        optimizers_config=qm.OptimizersConfigDiff(
            indexing_threshold=100_000,
            memmap_threshold=100_000,
        ),
        on_disk_payload=True,
    )
    _ngram_note = (
        f", sparse='{ngram_vector_name}' (ngram, no IDF)" if ngram_vector_name else ""
    )
    print(
        f"Collection '{collection_name}' created "
        f"(dense={dense_dim}-dim cosine on_disk, sparse='{SPARSE_VECTOR_NAME}' IDF{_ngram_note}, "
        f"INT8-quantized always_ram, HNSW on_disk, optimizers indexing_threshold=100k)."
    )

    for field_name, schema_type in PAYLOAD_INDEXES:
        await client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=schema_type,
            wait=True,
        )
        print(f"  payload index: {field_name} ({schema_type.name})")

    await client.close()
    print("Done.")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Create the Qdrant collection for semantic_search"
    )
    parser.add_argument("--host", type=str, default="localhost", help="Qdrant host")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant REST port")
    parser.add_argument(
        "--collection-name",
        type=str,
        default="auctions_listings",
        help="Qdrant collection name",
    )
    parser.add_argument(
        "--dense-dim",
        type=int,
        default=None,
        help="Dense vector dimension (default: cascade.stage_dims.shortlist from base.yaml)",
    )
    args = parser.parse_args()
    cfg = load_config()
    dense_dim = (
        args.dense_dim
        if args.dense_dim is not None
        else int(cfg["qi"]["encoder"]["cascade"]["stage_dims"]["shortlist"])
    )
    _ngram = (cfg.get("retrieval", {}).get("qdrant", {}).get("hybrid", {}) or {}).get(
        "ngram"
    ) or {}
    ngram_vector_name = (
        str(_ngram.get("vector_name")) if _ngram.get("enabled") else None
    )
    asyncio.run(
        create_collection(
            host=args.host,
            port=args.port,
            collection_name=args.collection_name,
            dense_dim=dense_dim,
            ngram_vector_name=ngram_vector_name,
        )
    )


if __name__ == "__main__":
    main()
