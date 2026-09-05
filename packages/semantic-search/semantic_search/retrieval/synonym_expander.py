"""Synonym/abbreviation expansion for the BM25 sparse query leg.

Sits between ``BM25QueryEncoder._tokenize`` and ``_aggregate_term_weights``.
For each surviving query token the expander consults a domain synonym map and
emits up to ``max_synonyms_per_token`` extra tokens, each carrying
``expansion_weight ∈ (0, 1]`` so the synonyms add background lift to the
sparse vector without being able to drown out the original tokens.

Bidirectionality:
The expander mirrors every forward edge into an inverse edge at construction.
A user-authored line ``ai: [artificial, intelligence]`` produces
``ai → {artificial, intelligence}``, ``artificial → {ai}``, and
``intelligence → {ai}``. This guarantees retrieval-time symmetry without
requiring authors to maintain both directions.

Bounded cost:
- Per query: at most ``max_tokens_to_expand`` original tokens are inspected.
- Per token: at most ``max_synonyms_per_token`` synonyms are emitted (after
  de-duplication against original-token presence and against synonyms
  already emitted earlier in the same query).
- Worst-case post-expansion sequence length is therefore
  ``len(original_tokens) + max_tokens_to_expand × max_synonyms_per_token``.

Stdlib-only. No optional deps. Module-private to ``retrieval/`` per the
package layer rules.
"""
from typing import Dict, List, Sequence, Tuple

from semantic_search.config.models import SynonymExpansionConfig
from semantic_search.core.exceptions import RetrievalError, ValidationError
from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)


def build_bidirectional_map(forward_map: Dict[str, List[str]]) -> Dict[str, Tuple[str, ...]]:
    """Mirror every forward edge into the inverse direction.

    Output value lists are de-duplicated and sorted lexicographically so the
    expansion order is deterministic across runs (a different process on a
    different host produces the same SparseVector for the same query).

    Self-edges (a token mapped to itself) are dropped — they would emit the
    original token twice and inflate its BM25 weight in a way the user
    almost certainly did not intend.

    Empty value lists in the input are silently skipped (the user wrote the
    key but provided no synonyms).

    :param forward_map: Dict[str, List[str]] - User-authored synonym map
    :return: Dict[str, Tuple[str, ...]] - Bidirectional, deduped, sorted map
    :raises ValidationError: When inputs are malformed
    """
    if not isinstance(forward_map, dict):
        raise ValidationError("build_bidirectional_map requires a dict")
    bidir: Dict[str, set] = {}
    for raw_key, raw_vals in forward_map.items():
        if not isinstance(raw_key, str) or not raw_key:
            raise ValidationError(f"synonym map key must be non-empty string, got {raw_key!r}")
        if not isinstance(raw_vals, list):
            raise ValidationError(f"synonym map value for '{raw_key}' must be a list")
        key = raw_key.lower().strip()
        if not key:
            continue
        for raw_v in raw_vals:
            if not isinstance(raw_v, str) or not raw_v:
                raise ValidationError(f"synonym map value entries for '{raw_key}' must be non-empty strings")
            v = raw_v.lower().strip()
            if not v or v == key:
                continue
            bidir.setdefault(key, set()).add(v)
            bidir.setdefault(v, set()).add(key)
    return {k: tuple(sorted(vs)) for k, vs in bidir.items()}


class SynonymExpander:
    """Bidirectional, weight-capped synonym expansion for BM25.

    :param config: SynonymExpansionConfig - Validated config bundle
    :raises ValidationError: When ``config`` is None / wrong-typed
    :raises RetrievalError: When the synonym map cannot be built (malformed
        entries that escape the config-layer validation)
    """

    def __init__(self, config: SynonymExpansionConfig):
        if config is None or not isinstance(config, SynonymExpansionConfig):
            raise ValidationError("SynonymExpander requires a SynonymExpansionConfig")
        self._config = config
        try:
            self._bidir_map = build_bidirectional_map(config.synonym_map)
        except ValidationError as e:
            raise RetrievalError(f"synonym_expander_build_failed: {e}") from e

    @property
    def enabled(self) -> bool:
        """Master switch — when False, ``expand`` returns the input unchanged."""
        return bool(self._config.enabled)

    @property
    def map_size(self) -> int:
        """Number of keys in the bidirectional map (diagnostics)."""
        return len(self._bidir_map)

    def expand(self, tokens: Sequence[str]) -> List[Tuple[str, float]]:
        """Expand ``tokens`` into ``[(token, weight), ...]``.

        Original tokens always receive weight 1.0. Expanded synonyms receive
        ``expansion_weight``. The output preserves the first-seen ordering of
        the original tokens, with each token's synonyms appended immediately
        after it (so consumers that care about token locality — e.g. lexical
        rerankers — see related terms grouped).

        Deduplication rules:
        - A token (original) appearing twice in the input emits weight-1.0
          only once (BM25's TF stage re-aggregates the count from the
          original sequence anyway; we deduplicate here so the expander does
          not double-count).
        - A synonym that is also an original token anywhere in the same
          query (not just earlier in the sequence) is dropped — the
          original carries weight 1.0 and adding the same surface form at
          ``expansion_weight`` would only confuse the BM25 sum.
        - A synonym that has already been emitted by a previous original
          token is dropped (avoids the cross-token duplicate where two
          originals share a synonym).

        Two-pass implementation: pass 1 builds ``seen_original`` from the
        full input so the dedup-against-originals check works regardless of
        whether the duplicate appears before or after the synonym-emitting
        original in the input order.

        :param tokens: Sequence[str] - Original tokenized query terms
        :return: List[Tuple[str, float]] - Original + synonym pairs
        :raises ValidationError: When ``tokens`` is None
        """
        if tokens is None:
            raise ValidationError("SynonymExpander.expand requires a non-None token sequence")
        if not self._config.enabled or not self._bidir_map:
            # Fast path: pass-through with weight 1.0 for every original.
            return [(t, 1.0) for t in tokens]
        # Pass 1 — collect lowercased original tokens (all of them, not just
        # those processed so far in pass 2) so we can drop synonyms that
        # already appear as originals anywhere in the query.
        normalized_originals: List[str] = []
        seen_original: set = set()
        for raw_tok in tokens:
            if not isinstance(raw_tok, str):
                continue
            tok = raw_tok.lower()
            if not tok:
                continue
            normalized_originals.append(tok)
            seen_original.add(tok)
        # Pass 2 — emit originals in first-seen order with their synonyms
        # interleaved, applying the dedup rules above.
        out: List[Tuple[str, float]] = []
        emitted_originals: set = set()
        seen_synonym: set = set()
        consulted = 0
        weight = float(self._config.expansion_weight)
        cap_per_token = int(self._config.max_synonyms_per_token)
        cap_tokens = int(self._config.max_tokens_to_expand)
        for tok in normalized_originals:
            if tok not in emitted_originals:
                out.append((tok, 1.0))
                emitted_originals.add(tok)
            if consulted >= cap_tokens:
                continue
            consulted += 1
            synonyms = self._bidir_map.get(tok)
            if not synonyms:
                continue
            emitted_for_token = 0
            for syn in synonyms:
                if emitted_for_token >= cap_per_token:
                    break
                if syn in seen_original:
                    continue
                if syn in seen_synonym:
                    continue
                out.append((syn, weight))
                seen_synonym.add(syn)
                emitted_for_token += 1
        return out
