"""Live inventory-grounded value resolution.

The resolver computes percentiles over the structured-index payload columns so
fuzzy terms like "cheap" and "expiring" map to runtime-current values rather
than stale config priors. Cold path / empty index falls back to the priors in
``InventoryConfig``.
"""
from semantic_search.inventory.resolver import PercentileResolver

__all__ = ['PercentileResolver']
