"""Offline training utilities for the QI semantic router.

This package hosts offline-only artefacts:

- ``data_schema``: typed dataclasses for training rows + run artefacts.
- ``hard_negative_miner``: scans the curated seed corpus for cross-archetype
  confusables and emits a labelled JSONL file consumed by the learned-head trainer.

Layer rules (per ``architecture.mdc``): stdlib + ``core`` + ``contracts`` +
``config`` + sibling QI primitives (``encoder``, ``seed_loader``). No
retrieval, orchestration, registry, or LLM imports — these tools must run
in any offline / Docker context the live router runs in.
"""
