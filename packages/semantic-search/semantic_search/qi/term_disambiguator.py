"""Domain-term polysemy disambiguator — resolves ambiguous tokens before L0/L1 processing.

Certain tokens are valid both as TLD stems and as topic/intent modifiers in domain search
queries (e.g. "ai" → Anguilla `.ai` TLD vs. Artificial Intelligence topic).  This module
inspects the surrounding context window to choose the correct interpretation and rewrites
the bare token to its canonical TLD form (e.g. ``ai`` → ``.ai``) when the TLD reading
applies.

Rewrite scope: L0 (L0LLMFilterExtractor) and L1 (SemanticRouter) both receive the
disambiguated text.  L2 (LLM) always receives the original normalized text — the model
is expressive enough to resolve ambiguity independently, and preserving the original
prevents the rewrite from appearing in reasoning traces.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, List, NamedTuple, Optional

from semantic_search.config.models import TermDisambiguatorConfig, TermDisambiguationRule
from semantic_search.core.logging_utils import get_logger

logger = get_logger(__name__)


class _CompiledRule(NamedTuple):
    tld_form: str
    tld_ctx: FrozenSet[str]
    topic_ctx: FrozenSet[str]
    default_to_tld: bool


class TermDisambiguator:
    """Disambiguate polysemous tokens (e.g. "ai" → .ai TLD vs AI topic) via context window."""

    def __init__(self, config: TermDisambiguatorConfig) -> None:
        self._config = config
        # Pre-compile context sets as frozensets for O(1) set-intersection at query time.
        self._rules: Dict[str, _CompiledRule] = {
            r.term.lower(): _CompiledRule(
                tld_form=r.tld_form,
                tld_ctx=frozenset(r.tld_context_tokens),
                topic_ctx=frozenset(r.topic_context_tokens),
                default_to_tld=r.default_to_tld,
            )
            for r in config.rules
        }

    def disambiguate(self, normalized_query: str) -> str:
        """Rewrite polysemous tokens: topic context wins, else TLD if explicit/default."""
        if not self._config.enabled or not self._rules:
            return normalized_query
        tokens: List[str] = normalized_query.split()
        if not tokens:
            return normalized_query
        rewritten = list(tokens)
        changed = False
        win = self._config.context_window
        for i, token in enumerate(tokens):
            if token.startswith('.'):
                continue
            rule = self._rules.get(token)
            if rule is None:
                continue
            lo = max(0, i - win)
            hi = min(len(tokens), i + win + 1)
            ctx: FrozenSet[str] = frozenset(tokens[lo:i]) | frozenset(tokens[i + 1:hi])
            has_topic = bool(ctx & rule.topic_ctx)
            has_tld = bool(ctx & rule.tld_ctx)
            if has_topic:
                # Explicit topic context; keep as-is
                continue
            if has_tld or rule.default_to_tld:
                rewritten[i] = rule.tld_form
                changed = True
                logger.debug(
                    f"term_disambiguate token={token!r} → tld={rule.tld_form!r} "
                    f"has_tld_ctx={has_tld} has_topic_ctx={has_topic} "
                    f"default_to_tld={rule.default_to_tld}"
                )

        return " ".join(rewritten) if changed else normalized_query
