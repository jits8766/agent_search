"""Query-side compound-word expansion for the semantic + lexical encode text.

The ingest pipeline runs a Viterbi compound-word splitter over each document's
registrable label (``techstartup`` -> ``tech startup``) so the index stores
segmented tokens. The query side historically left labels glued, so a query for
``techstartup`` could not match the indexed ``tech startup`` segments on the
sparse / ngram legs and scored weakly on the dense leg.

This module applies the SAME splitter to the query encode text before it reaches
any encoder, restoring query/document symmetry. It does not import the splitter
type directly — it takes an injected ``segment_fn`` (the registry binds it to
``CompoundWordSplitter.split(...).segments``) so this layer stays free of the
vectorization package and the unit tests can drive it with a stub.

Stateless once constructed; safe to share across threads / asyncio tasks. No
numeric parameters live here — every cost / length bound is owned by the
injected splitter, which is built from ``retrieval.query_compound_split`` config.
"""
from typing import Callable, List, Sequence

from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)

# Segment function contract: a glued label -> its ordered segments. The registry
# binds this to a configured CompoundWordSplitter; a label that does not split
# returns a single-element sequence (the label itself).
SegmentFn = Callable[[str], Sequence[str]]


class QueryCompoundExpander:
    """Split each whitespace token of the encode text via an injected splitter.

    :param segment_fn: SegmentFn - Maps one glued label to its ordered
        segments. Must return a non-empty sequence (the splitter's
        single-token baseline guarantees this).
    """

    def __init__(self, segment_fn: SegmentFn) -> None:
        self._segment_fn = segment_fn

    def __call__(self, text: str) -> str:
        """Return ``text`` with every whitespace token replaced by its segments.

        Order-preserving and deterministic. Empty / whitespace-only input
        returns unchanged. A token whose segmentation raises is kept verbatim
        (the expander never blocks retrieval on a splitter glitch).

        :param text: str - Encode text (already TLD-safe via the QI engine)
        :return: str - Space-joined segmented tokens
        """
        if not text or not text.strip():
            return text
        out: List[str] = []
        for token in text.split():
            try:
                segments = self._segment_fn(token)
            except Exception as exc:  # splitter glitch must not block retrieval
                logger.warning(f"query_compound_split_token_failed token_len={len(token)} error_type={type(exc).__name__} error={str(exc)}")
                out.append(token)
                continue
            if not segments:
                out.append(token)
                continue
            out.extend(str(s) for s in segments)
        return " ".join(out)


__all__ = ['QueryCompoundExpander', 'SegmentFn']
