"""Shared budget / locate price cues (``in $50``, ``for 100``, ``around $75``).

Single parse used by qie_only ``filters_to_identified``, full-search
``apply_post_merge_reconcile``, and regex L0 parity — paths must not drift.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

# Exclusive under/below owns ceiling — budget-prefix must not also bind.
_UNDER_PRICE_CUE_RE = re.compile(
    r"\b(?:under|below|less\s+than)\s+\$?(?P<n>\d[\d,]*)\s*(?P<k>k|thousand|million)?\b"
    r"|\b(?P<n2>\d[\d,]*)\s*(?P<k2>k|thousand)?\s+or\s+(?:under|below)\b",
    re.IGNORECASE,
)

# "in $50", "for 100", "at ~$75", "costing 2k", "budget of $200", "around 500".
_BUDGET_PREFIX_PRICE_RE = re.compile(
    r"(?:"
    r"\b(?P<kind>in|for|at|about|near|around|approx(?:imately)?|"
    r"up\s+to|within|"
    r"priced?\s+at|"
    r"cost(?:s|ing)?|"
    r"(?:with\s+(?:a\s+)?)?budget(?:\s+of)?)\s+"
    r"|"
    r"(?P<tilde>~)\s*"
    r")"
    r"(?P<dollar>\$)?\s*(?P<n>\d[\d,]*)\s*(?P<k>k|thousand|million)?\b"
    r"(?!\s*(?:chars?|characters?|letters?|words?|years?|yrs?|days?|hours?|"
    r"bids?|backlinks?|percent|%|auctions?))",
    re.IGNORECASE,
)
_BUDGET_BAND_KINDS = frozenset({
    'at', 'about', 'near', 'around', 'approx', 'approximately',
})

# "around N maybe less" / "around N or below" -- one-sided ceiling, no band.
_AROUND_ONE_SIDED_RE = re.compile(
    r"\b(?:(?:budget\s+)?around|~)\s*\$?\d[\d,]*\s*(?:k|thousand)?\b"
    r".{0,32}\b(?:maybe\s+|perhaps\s+)?(?:less|under|below|or\s+(?:less|under|below))\b",
    re.IGNORECASE,
)


def _budget_prefix_price_ok(kind: str, dollar: Optional[str], n: int, scale: str) -> bool:
    """Guard FPs: 'in 16' auction-type, 'for 5' length, bare 'in 50' without $."""
    if scale or dollar:
        return n > 0
    if kind == 'in':
        # Require $ or k/m for bare "in N" (avoids "ending in 50", "in 16").
        return False
    return n >= 50


def parse_budget_prefixed_price(query: str) -> Optional[Tuple[int, bool]]:
    """Parse ``in $50`` / ``for 100`` / ``around $75`` budget cues.

    :return: ``(amount, is_band)`` or ``None``. ``is_band`` means min and max both bind.
    """
    if not query:
        return None
    if _UNDER_PRICE_CUE_RE.search(query):
        return None
    m = _BUDGET_PREFIX_PRICE_RE.search(query)
    if m is None:
        return None
    kind = (m.group('kind') or '').lower()
    if m.group('tilde'):
        kind = 'around'
    kind = re.sub(r'\s+', ' ', kind).strip()
    if kind.startswith('price') or kind.startswith('priced'):
        kind = 'at'
    if kind.startswith('cost'):
        kind = 'for'
    if 'budget' in kind or kind in ('up to', 'within'):
        kind = 'for'
    scale = (m.group('k') or '').lower()
    try:
        n = int(str(m.group('n')).replace(',', ''))
    except (TypeError, ValueError):
        return None
    if not _budget_prefix_price_ok(kind, m.group('dollar'), n, scale):
        return None
    if scale in ('k', 'thousand'):
        n *= 1000
    elif scale == 'million':
        n *= 1_000_000
    if n <= 0:
        return None
    band = kind in _BUDGET_BAND_KINDS or bool(m.group('tilde'))
    if band and _AROUND_ONE_SIDED_RE.search(query):
        band = False
    return n, band


__all__ = ['parse_budget_prefixed_price']
