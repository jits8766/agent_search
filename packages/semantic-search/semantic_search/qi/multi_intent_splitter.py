"""MultiIntentSplitter — deterministic L0 split for compound queries.

"L0 deterministic splitter on AND/OR/comma/semicolon fires
BEFORE classification."

Splitter contract:
  1. Split a compound query into raw sub-queries on connectives (AND/OR/comma/
     semicolon) and conflict signals.
  2. Filter empty / too-short fragments (config.min_sub_query_chars).
  3. Hard cap raw output at config.max_split_candidates to bound LLM cost.
  4. Single-intent queries pass through as a one-element list with no overhead.

The splitter does NOT classify, ground, pre-screen, or rank — those are the
responsibilities of QIEngine + the orchestrator. Splitter output is a pure
list of normalized sub-query strings, deterministic over the same input.

Operator-style connectives (and / or / vs / versus) are matched only when
whitespace-bounded on both sides, so substrings like "Anders" or "organic"
(which contain "and"/"or" without surrounding whitespace) are never split.
There is no token-length constraint on the split point; semantic false-split
pruning is the QIEngine's job, not the splitter's. Punctuation connectives
(`,` `;`) are matched anywhere, except `,` is preserved between digits to
keep thousands separators intact. A numeric range "X and Y" (e.g.
"$100 and $2000") is masked before the word-delim pass so it is not split.
"""
import re
from typing import Any, Dict, List, Sequence, Tuple

from semantic_search.config.models import MultiIntentConfig
from semantic_search.core.exceptions import ValidationError
from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)

# ; always splits. , splits except between digits (preserve thousands separator)
_PUNCT_DELIMS = re.compile(r";+|(?<!\d),|,(?!\d)")

# Word delimiters: whitespace-bounded AND/OR/vs/versus (order: long first)
_WORD_DELIMS = re.compile(r"\s+(?:and|or|vs|versus)\s+", flags=re.IGNORECASE)

# Whitespace normalizer for each emitted sub-query.
_WHITESPACE = re.compile(r"\s+")

# Guard: "$100 and $2000" (range) must not split; use placeholder, restore per fragment
_RANGE_AND_RE = re.compile(r"(\$?[\d][\d,]*)\s+(and)\s+(\$?[\d])", re.IGNORECASE)
_RANGE_AND_PLACEHOLDER = "\x00RANGE_AND\x00"


class MultiIntentSplitter:
    """Deterministic split on AND/OR/comma/semicolon (L0, pre-classification)."""

    def __init__(self, config: MultiIntentConfig) -> None:
        if config is None:
            raise ValidationError("MultiIntentSplitter requires MultiIntentConfig")
        self._config = config
        self._no_split_res = [re.compile(p, re.IGNORECASE) for p in (config.no_split_patterns or [])]

    def split(self, normalized_query: str) -> List[str]:
        """Query -> 1..N sub-queries (single-intent passes through, bounded by max_split_candidates)."""
        if normalized_query is None or not isinstance(normalized_query, str):
            raise ValidationError("MultiIntentSplitter.split requires a string input")
        if not normalized_query.strip():
            raise ValidationError("MultiIntentSplitter.split: query is empty")

        if not self._config.enabled:
            return [normalized_query]

        # Short-circuit: if any no_split_pattern matches, treat as single intent.
        for _nsp_re in self._no_split_res:
            if _nsp_re.search(normalized_query):
                return [normalized_query]

        # Protect numeric range "X and Y" from being split as an intent boundary.
        # "domains between $100 and $2000" must not split into ["domains between $100",
        # "$2000"]. Mask the "and" between two numeric bounds before word-delim split.
        masked_query = _RANGE_AND_RE.sub(
            lambda m: m.group(1) + " " + _RANGE_AND_PLACEHOLDER + " " + m.group(3),
            normalized_query,
        )

        # Two-pass split: punctuation first (simpler regex), then word delims on
        # each surviving fragment. This preserves order and avoids regex-engine
        # corner cases when both classes appear in the same query.
        punct_split: List[str] = _PUNCT_DELIMS.split(masked_query)
        sub_queries: List[str] = []
        for fragment in punct_split:
            stripped = fragment.strip()
            if not stripped:
                continue
            sub_queries.extend(_WORD_DELIMS.split(stripped))
        # Restore placeholder back to "and" in each fragment.
        sub_queries = [sq.replace(_RANGE_AND_PLACEHOLDER, "and") for sq in sub_queries]

        # Normalize whitespace + drop fragments below the minimum length.
        cleaned: List[str] = []
        for sq in sub_queries:
            collapsed = _WHITESPACE.sub(' ', sq.strip())
            if len(collapsed) < self._config.min_sub_query_chars:
                continue
            cleaned.append(collapsed)

        if not cleaned:
            # Splitter ate the entire query — fall back to the original to
            # preserve the single-intent guarantee.
            return [normalized_query]

        # Hard cap on raw split count (cost control). We do NOT rank here — the
        # 5-cap weighted ranking happens after pre-screen in the orchestrator.
        if len(cleaned) > self._config.max_split_candidates:
            logger.warning(f"multi_intent_split_cap_hit raw={len(cleaned)} cap={self._config.max_split_candidates} truncating_to_first_n")
            cleaned = cleaned[: self._config.max_split_candidates]

        # Single-intent: return the (cleaned) original even if it differs from
        # input in whitespace. This keeps downstream identical to the no-split
        # path for the common case.
        if len(cleaned) == 1:
            return [normalized_query] if cleaned[0] == normalized_query else cleaned

        # De-duplicate while preserving order — "domains, domains" should split
        # to one sub-query, not two.
        seen: set = set()
        deduped: List[str] = []
        for sq in cleaned:
            if sq in seen:
                continue
            seen.add(sq)
            deduped.append(sq)

        if len(deduped) == 1:
            # All splits collapsed to the same sub-query after dedup — single
            # intent after all.
            return [normalized_query] if deduped[0] == normalized_query else deduped

        logger.info(f"multi_intent_split count={len(deduped)} from_query_len={len(normalized_query)}")
        return deduped

    def expand_with_keywords(
        self,
        normalized_query: str,
        keywords: Sequence[Dict[str, Any]],
    ) -> List[str]:
        """Expand a single fragment into one leg per L0 keyword + shared constraints.

        Used when delimiter split yields one query but L0 extracted multiple topical
        keywords (e.g. coffee + pizza). Disabled when ``split_on_l0_keywords`` is false.
        """
        if not bool(self._config.split_on_l0_keywords):
            return [normalized_query]
        if not isinstance(normalized_query, str) or not normalized_query.strip():
            raise ValidationError("expand_with_keywords requires a non-empty string query")
        min_terms = int(self._config.split_on_l0_keywords_min_terms)
        if min_terms < 2:
            raise ValidationError("split_on_l0_keywords_min_terms must be >= 2")
        terms: List[str] = []
        seen: set = set()
        for kw in keywords or []:
            if not isinstance(kw, dict):
                continue
            term = str(kw.get("term") or "").strip()
            if not term:
                continue
            key = term.casefold()
            if key in seen:
                continue
            seen.add(key)
            terms.append(term)
        if len(terms) < min_terms:
            return [normalized_query]
        residual = normalized_query
        for term in terms:
            residual = re.sub(
                rf"\b{re.escape(term)}\b",
                " ",
                residual,
                flags=re.IGNORECASE,
            )
        residual = _WHITESPACE.sub(" ", residual).strip()
        legs: List[str] = []
        for term in terms:
            leg = f"{term} {residual}".strip() if residual else term
            leg = _WHITESPACE.sub(" ", leg).strip()
            if len(leg) < self._config.min_sub_query_chars:
                continue
            legs.append(leg)
        if len(legs) < min_terms:
            return [normalized_query]
        if len(legs) > self._config.max_split_candidates:
            logger.warning(
                f"multi_intent_keyword_split_cap_hit raw={len(legs)} "
                f"cap={self._config.max_split_candidates}"
            )
            legs = legs[: self._config.max_split_candidates]
        logger.info(
            f"multi_intent_keyword_split count={len(legs)} "
            f"from_query_len={len(normalized_query)} terms={len(terms)}"
        )
        return legs


def rank_sub_intents(candidates: List[Tuple[str, float, int, int]], config: MultiIntentConfig) -> List[Tuple[str, float, int, int, float]]:
    """Score sub-intent candidates and return them sorted with the 5-cap applied.

    Weighted ranking formula:
        score = w_conf * confidence
              + w_exp  * min(expected_results, norm_cap) / norm_cap
              + w_spec * specificity_count / max_specificity_observed

    :param candidates: List[Tuple[sub_query, confidence, expected_results,
        specificity_count]] - One tuple per sub-intent candidate. Order in the
        input has no semantic meaning; this function reorders.
    :param config: MultiIntentConfig - Provides weights, norm cap,
        ``max_sub_intents``, ``collapse_threshold``, ``top_k_after_collapse``.
    :return: List[Tuple[sub_query, confidence, expected_results, specificity,
        rank_score]] - Sorted by rank_score descending. Truncation policy
        (multi-intent truncation policy):
          - len(candidates) <  collapse_threshold -> keep top
            ``max_sub_intents``
          - len(candidates) >= collapse_threshold -> keep top
            ``top_k_after_collapse``; the remaining candidates collapse
            into the orchestrator-emitted overflow chip.
        Empty input -> empty output.

    Hand-traceable examples (norm_cap=1000, weights 0.5/0.3/0.2,
    max_sub_intents=5, collapse_threshold=6, top_k_after_collapse=3):
      A) Input: [("a", 0.8, 100, 2), ("b", 0.4, 1000, 1)]  -> 2 cands < 6
         max_specificity = 2
         score("a") = 0.5*0.8 + 0.3*(100/1000) + 0.2*(2/2) = 0.63
         score("b") = 0.5*0.4 + 0.3*(1000/1000) + 0.2*(1/2) = 0.60
         Output (cap=5, both fit): [("a", ..., 0.63), ("b", ..., 0.60)]
      B) Input: 6 candidates -> collapse fires -> only top 3 returned.
    """
    if not isinstance(candidates, list):
        raise ValidationError("rank_sub_intents requires a list of candidate tuples")
    if not candidates:
        return []

    max_specificity = max((spec for _, _, _, spec in candidates), default=0)
    norm_cap = float(config.expected_results_norm_cap)
    scored: List[Tuple[str, float, int, int, float]] = []
    for sub_query, confidence, expected, specificity in candidates:
        if not isinstance(sub_query, str) or not sub_query:
            raise ValidationError("rank_sub_intents: sub_query must be non-empty string")
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValidationError("rank_sub_intents: confidence must be in [0,1]")
        if int(expected) < 0:
            raise ValidationError("rank_sub_intents: expected_results must be >= 0")
        if int(specificity) < 0:
            raise ValidationError("rank_sub_intents: specificity must be >= 0")
        exp_norm = min(float(expected), norm_cap) / norm_cap if norm_cap > 0 else 0.0
        spec_norm = (float(specificity) / float(max_specificity)) if max_specificity > 0 else 0.0
        score = (
            config.weight_confidence * float(confidence)
            + config.weight_expected_results * exp_norm
            + config.weight_specificity * spec_norm
        )
        scored.append((sub_query, float(confidence), int(expected), int(specificity), float(score)))

    scored.sort(key=lambda t: t[4], reverse=True)
    if len(scored) >= config.collapse_threshold:
        return scored[: config.top_k_after_collapse]
    return scored[: config.max_sub_intents]
