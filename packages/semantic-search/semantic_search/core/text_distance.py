"""Shared edit-distance primitives (stdlib-only).

Single source of truth for Damerau-Levenshtein so the Tier-0 spell corrector
(``qi.spell_corrector``) and the fuzzy lexical reranker
(``retrieval.fuzzy_lexical_reranker``) score typos on identical math. Lives in
``core/`` because both ``qi/`` and ``retrieval/`` may import core without a layer
violation.
"""
from typing import List


def damerau_levenshtein(a: str, b: str, max_distance: int) -> int:
    """Edit distance with transposition (returns true dist or max_distance+1 sentinel)."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > max_distance:
        return max_distance + 1
    prev_prev: List[int] = []
    prev: List[int] = list(range(lb + 1))
    for i in range(1, la + 1):
        curr: List[int] = [i] + [0] * lb
        row_min = curr[0]
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                curr[j] = min(curr[j], prev_prev[j - 2] + 1)
            if curr[j] < row_min:
                row_min = curr[j]
        if row_min > max_distance:
            return max_distance + 1
        prev_prev = prev
        prev = curr
    final = prev[lb]
    if final > max_distance:
        return max_distance + 1
    return final


def similarity_ratio(a: str, b: str, max_distance: int) -> float:
    """Normalised edit-distance: 1 - (dist / max(len(a), len(b))) in [0, 1]."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    dist = damerau_levenshtein(a, b, max_distance)
    if dist > max_distance:
        return 0.0
    longest = max(len(a), len(b))
    if longest == 0:
        return 0.0
    return 1.0 - (float(dist) / float(longest))
