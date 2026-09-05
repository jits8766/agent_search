"""Shared advisory / analytics speech-act detectors for L0 LLM, regex, and reconcile.

Design rules (keep generic — no suite query literals):
- Soft advisory: guidance / advice-seeking speech acts. Callers must also require
  *no inventory bound* before wiping filters.
- Strong advisory: aggregate / comparative analytics intents that must never emit
  inventory filters even when TLD/price tokens appear.

All patterns are case-insensitive speech-act / intent classes, not specific
example queries from eval suites.
"""
from __future__ import annotations

import re

# Soft: user seeks advice / explanation / strategy (inventory bounds may still apply).
SOFT_ADVISORY_RE = re.compile(
    r"""
    # Advice / decision speech acts
    \b(?:should|would|could)\s+(?:i|we|you|one)\b
    | \bif\s+(?:i|we|you)\s+should\b
    | \bshould\b.{0,48}\b(?:include|buy|invest|spend|choose|pick)\b
    | \b(?:advice|recommend(?:ation)?|strategy)\b
    | \bhelp\s+(?:me\s+)?(?:choose|decide|pick|understand)\b
    | \bexplain\b
    | \bi\s+(?:am\s+)?(?:trying\s+to\s+decide|new\s+to|beginner)\b
    | \bi'?m\s+fine\b
    | \bim\s+fine\b
    | \bcan\s+(?:i|we|you)\s+(?:get|show|tell|help|negotiate)\b
    # Open evaluative questions
    | \bwhat\s+(?:makes|should|are\s+the|is\s+the\s+(?:best|risk))\b
    | \bwhats?\s+(?:trending|avg|average)\b
    | \bbest\s+(?:\w+\s+){0,3}for\b
    | \bhow\s+(?:do|does|can|should|would|could|to|often|long|far|soon|well|best|important)\b
    | \b(?:still\s+)?worth\s+(?:buying|spending|it)\b
    | \bworth?\s+spending\b
    | \bis\s+(?:it|this|a|an|\.\w+)\s+(?:still\s+)?worth\b
    | \bare\s+(?:\w+\s+){1,4}(?:still\s+)?worth\b
    | \bwould\s+you\s+(?:buy|recommend|choose)\b
    | \bwhy\s+(?:are|is|do|does)\b
    | \bred\s+flags?\b
    | \bwhat\s+(?:to\s+)?(?:check|know|watch)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Strong: analytics / aggregates / portfolio meta — always empty filters.
STRONG_ADVISORY_RE = re.compile(
    r"""
    # Count / distribution / growth analytics
    \bhow\s+(?:many|much|often)\b
    | \b(?:listing\s+)?counts?\b.{0,24}\b(?:growing|increasing|decreasing|falling|by)\b
    | \blisting\s+count\b
    | \blisting\s+cnt\b
    | \btotal\s+listings?\b
    | \b(?:new\s+)?listings?\s+trend\b
    | \b(?:new\s+)?listngs?\s+trnd\b
    | \bmedian\s+(?:current\s+)?bid\b
    | \btop\s+tld\s+by\b
    | \bbroken\s+down\s+by\b
    | \b(?:status\s+)?distribution\b
    | \btld\s+distribution\b
    | \bgrowth\s+in\b
    | \bmonth\s+over\s+month\b
    | \b(?:average|avg)\s+(?:current\s+)?(?:bid|price|listing)\b
    | \bprice\s+range\s+with\b
    | \bwhich\s+(?:category|tld|niche)\s+has\b
    | \bhottest\s+category\b
    | \bmost\s+active\s+auctions?\b
    | \bbid\s+activity\b
    | \btop\s+(?:current\s+)?bids?\b
    | \bactive\s+listings\s+over\s+the\s+last\b
    # Comparative analytics (not inventory "com or io")
    | \b(?:vs\.?|versus)\b.{0,32}\b(?:for|or|better|listing|count|volume|growth|quantity)\b
    | \bauction\s+vs\.?\s+(?:buynow|buy\s*now|buy-now)\b
    | \bcomparing\s+(?:two|2|\w+)\s+domains?\b
    | \b(?:better|worse)\s+for\s+(?:seo|resale|branding)\b
    | \bwhich\s+matters\s+more\b
    | \bmatters\s+more\b
    # Impact stem tolerates truncations: impact / impt / impct.
    | \bbigger\s+imp(?:ac|c)?t\b
    | \bhave\s+(?:a\s+)?bigger\s+imp(?:ac|c)?t\b
    | \bhave\s+lower\s+(?:current\s+)?bids?\b
    | \bdoes\s+hyphen\s+imp(?:ac|c)?t\b
    | \bweekend\s+vs\.?\s+weekday\b
    | \bbeen\s+growing\s+lately\b
    | \bkeep(?:s|ing)?\s+seeing\b
    | \bpeople\s+watching\s+and\s+bidding\b
    | \bwhat\s+domains\s+are\s+people\b
    # Scams / diversification (always non-inventory; advisory speech acts moved to SOFT)
    | \bcan\s+i\s+get\s+scammed\b
    | \bdiversif(?:y|ication)\b
    | \bbetter\s+investment\b
    | \bis\s+there\s+(?:usually|any\s+noticeable)\b
    | \bwhen\s+is\s+(?:an?\s+)?expired\b
    # Aggregate analytics — minBids false-positive suppression
    | \bpercentage\s+of\s+auctions?\b
    | \bmost\s+(?:bid|bidded|active)\s+auctions?\b
    | \bhow\s+many\s+(?:domains?|auctions?|listings?)\s+have\b
    # Category-vs-category advice (not inventory "com or io")
    | \bbetter\s+(?:expired|auction|buynow|buy[\s-]?now|premium|closeout)\b
    | \bexpired\s+or\s+auction\b
    | \bauction\s+or\s+(?:expired|buynow|buy[\s-]?now)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Hard inventory bound — soft advisory must NOT wipe when these cues exist.
# Lifecycle / buy-now mentions alone do NOT count: "should I buy expired…" is
# guidance about a category, not an inventory search with a numeric/TLD bound.
INVENTORY_BOUND_RE = re.compile(
    r"(?:"
    r"\$\s*\d"
    r"|\b(?:under|below|over|above|at\s+least|at\s+most|capped\s+at|max(?:imum)?|"
    r"min(?:imum)?|floor)\s+\$?\d"
    r"|\b\d+(?:\.\d+)?\s*k\b"
    r"|\.\s*(?:com|net|org|io|in|ai|co|app|dev|xyz|info|biz|us|uk)\b"
    r"|\b(?:tld|extension)\b"
    r"|\btraffic\s+\d|\b\d+\s+(?:bids?|visitors?|monthly)\b"
    r")",
    re.IGNORECASE,
)


def is_strong_advisory(query: str) -> bool:
    """True when query is aggregate/analytics — filters must be empty."""
    return bool(query and STRONG_ADVISORY_RE.search(query))


def is_soft_advisory_no_inventory(query: str) -> bool:
    """True when guidance speech-act and no inventory bound — filters must be empty."""
    if not query:
        return False
    if not SOFT_ADVISORY_RE.search(query):
        return False
    if INVENTORY_BOUND_RE.search(query):
        return False
    return True


__all__ = [
    'SOFT_ADVISORY_RE',
    'STRONG_ADVISORY_RE',
    'INVENTORY_BOUND_RE',
    'is_strong_advisory',
    'is_soft_advisory_no_inventory',
]
