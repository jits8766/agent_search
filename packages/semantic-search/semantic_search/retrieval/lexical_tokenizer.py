"""Shared lexical tokenizer used by the BM25 query encoder and the lexical reranker.

Single source of truth for "lowercase + ``[a-z0-9]+`` runs + min-length filter +
stopword filter + max-terms cap" so the reranker scores documents on the SAME
token universe the BM25 sparse leg of the hybrid retriever uses. Drift between
the two would silently bias the reranker against terms that survive BM25 and
vice versa.

Stdlib-only. Module-private; consumers re-export the function via their own
public surfaces.
"""
import re
from typing import List, Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize_lexical(text: str, min_term_length: int, max_terms: int, stopwords: Sequence[str]) -> List[str]:
    """Lowercase, extract ``[a-z0-9]+`` runs, drop short tokens + stopwords, cap to ``max_terms``.

    Order-preserving (first-seen). ``None`` / empty / whitespace-only input
    returns ``[]`` — never raises.

    :param text: str - Raw input (already-normalized OK; the function lower-cases internally)
    :param min_term_length: int - Drop tokens shorter than this (>= 1)
    :param max_terms: int - Stop after this many surviving tokens (>= 1; soft cap to bound work)
    :param stopwords: Sequence[str] - Tokens to drop verbatim (post-lowercase comparison;
        a frozenset/set/list/tuple all work — converted internally)
    :return: List[str] - Surviving tokens preserving first-seen order, length <= ``max_terms``
    """
    if not text:
        return []
    if int(min_term_length) < 1:
        raise ValueError("min_term_length must be >= 1")
    if int(max_terms) < 1:
        raise ValueError("max_terms must be >= 1")
    stop = frozenset(s.lower() for s in stopwords)
    tokens: List[str] = []
    seen = 0
    for m in _TOKEN_RE.finditer(text.lower()):
        tok = m.group(0)
        if len(tok) < min_term_length:
            continue
        if tok in stop:
            continue
        tokens.append(tok)
        seen += 1
        if seen >= max_terms:
            break
    return tokens
