"""listing + bid event ingestion.

Two collaborators (both implement the Protocols in ``semantic_search.contracts``):
- ``InMemoryListingConsumer``: stub for the live FIND-Kinesis listing stream
- ``InMemoryBidConsumer``: stub for the live Auctions bid stream

Both stubs are replayable (``seed_from_dict``) so integration tests can stamp a
deterministic event sequence and assert the cache-invalidation cascade fires.
The contracts are the same ones the live consumers will satisfy — wiring the
production substrates is a registry-only change.
"""
from semantic_search.ingest.in_memory_consumer import InMemoryBidConsumer, InMemoryListingConsumer, SnapshotVersionRegistry

__all__ = ['InMemoryListingConsumer', 'InMemoryBidConsumer', 'SnapshotVersionRegistry']
