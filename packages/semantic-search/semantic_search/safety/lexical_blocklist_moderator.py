"""Lexical-blocklist moderator + NoOpModerator.

Stdlib-only deterministic moderator that flags items whose payload
(concatenated configured fields, tokenised through the same
``tokenize_lexical`` shared utility BM25 + reranker + diversifier all use)
contains any token in the configured banned-term set.

The ``Moderator`` Protocol defines the surface every backend must satisfy.
``LexicalBlocklistModerator`` is the stdlib default included in this module;
``NoOpModerator`` is the typed-disabled path. Future model-backed
implementations (e.g. a Perspective-API-style toxicity classifier or a
domain-specific safety LLM) register behind the same Protocol — the
``EgressGuard`` is unaware of which backend serves the call.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Protocol, runtime_checkable

from semantic_search.config.models import LexicalModeratorConfig
from semantic_search.core.exceptions import EgressGuardError, ValidationError
from semantic_search.core.logging_utils import get_logger
from semantic_search.retrieval.lexical_tokenizer import tokenize_lexical

logger = get_logger(__name__)


@dataclass(frozen=True)
class ModeratorVerdict:
    """Per-item verdict returned by every moderator backend.

    :param flagged: bool - True iff the moderator considers the item unsafe
    :param matched_terms: List[str] - Banned tokens that triggered the flag
        (lowercased, in first-seen order; empty when ``flagged=False``).
        Capped to the moderator's internal limit (matches beyond the cap are
        truncated and `truncated_matches=True` is set so callers know).
    :param truncated_matches: bool - True iff ``matched_terms`` was truncated
        to the moderator's internal cap. The flag is preserved separately so
        downstream alerting can distinguish a partial-list match from a
        full-list match without introspecting list length.
    """
    flagged: bool
    matched_terms: List[str] = field(default_factory=list)
    truncated_matches: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.flagged, bool):
            raise ValidationError("ModeratorVerdict.flagged must be a bool")
        if not isinstance(self.matched_terms, list):
            raise ValidationError("ModeratorVerdict.matched_terms must be a list")
        for t in self.matched_terms:
            if not isinstance(t, str) or not t:
                raise ValidationError("ModeratorVerdict.matched_terms entries must be non-empty strings")
        if not isinstance(self.truncated_matches, bool):
            raise ValidationError("ModeratorVerdict.truncated_matches must be a bool")
        # Self-consistency: flagged=False implies matched_terms is empty
        # (so callers can rely on either field as the safety signal).
        if not self.flagged and len(self.matched_terms) > 0:
            raise ValidationError("ModeratorVerdict: matched_terms must be empty when flagged=False")
        # The truncation flag only makes sense in the flagged-true direction.
        if not self.flagged and self.truncated_matches:
            raise ValidationError("ModeratorVerdict: truncated_matches cannot be True when flagged=False")


@runtime_checkable
class Moderator(Protocol):
    """Output-side moderator contract.

    Every backend (stdlib lexical, future model-backed) implements this so
    the ``EgressGuard`` is decoupled from the moderator implementation.

    :method name: Stable string identifier (e.g. ``'lexical_blocklist'``,
        ``'noop'``) used in logs and per-item ``reasons`` records.
    :method moderate: Inspect a single payload and return a ``ModeratorVerdict``.
        MUST be deterministic for a given (config, payload) pair so test
        assertions are stable. MUST be safe to call concurrently — backends
        that need state (e.g. a model client) take that as a constructor
        dependency, not as a per-call mutable handle.
    """

    @property
    def name(self) -> str:
        ...

    def moderate(self, payload: Mapping[str, Any]) -> ModeratorVerdict:
        ...


class NoOpModerator:
    """Identity moderator.

    Always returns ``flagged=False``. Wired by the registry when
    ``safety.egress_guard.moderator.enabled=false`` or
    ``backend='noop'`` so the ``EgressGuard`` can call ``moderate``
    unconditionally without a None check.
    """

    _NAME = 'noop'

    @property
    def name(self) -> str:
        return self._NAME

    def moderate(self, payload: Mapping[str, Any]) -> ModeratorVerdict:
        # Argument is ignored on purpose — the Protocol contract requires the
        # signature but the no-op backend returns the same verdict for every input.
        del payload
        return ModeratorVerdict(flagged=False, matched_terms=[], truncated_matches=False)


# Hard internal cap on the per-verdict ``matched_terms`` list. A pathological
# payload could in principle hit hundreds of banned tokens; we cap the audit
# record at this size and set ``truncated_matches=True`` so dashboards can
# still detect "many matches" without unbounded list growth.
_MAX_REPORTED_MATCHES = 16


class LexicalBlocklistModerator:
    """Stdlib-only deterministic moderator.

    Concatenates the configured payload fields into a single document
    string, tokenises through the shared ``tokenize_lexical`` utility, and
    flags the item iff any surviving token is in the banned-term set.

    Construction is fail-fast: a malformed config (already enforced by
    ``LexicalModeratorConfig.__post_init__``) prevents the registry from
    booting. Per-call behaviour is fail-soft: any exception during
    payload extraction or tokenisation is caught and re-raised as
    :class:`EgressGuardError` so the orchestrator can downgrade the
    affected item to "drop" rather than letting an unscrubbed payload
    leak past the gate.

    :param config: LexicalModeratorConfig - Already-validated config
    """

    _NAME = 'lexical_blocklist'

    def __init__(self, config: LexicalModeratorConfig):
        if config is None or not isinstance(config, LexicalModeratorConfig):
            raise EgressGuardError("LexicalBlocklistModerator requires a LexicalModeratorConfig instance")
        self._config = config
        # Frozenset for O(1) membership; entries are already validated as
        # lowercase non-empty strings by LexicalModeratorConfig.__post_init__.
        self._banned: frozenset = frozenset(self._config.banned_terms)

    @property
    def name(self) -> str:
        return self._NAME

    def _extract_doc_text(self, payload: Mapping[str, Any]) -> str:
        """Concatenate configured payload fields into a single string.

        Missing keys are skipped silently (they don't contribute tokens).
        Non-string scalars are coerced via ``str()`` so numerics + lists also
        contribute tokens — same coercion rule as the lexical reranker /
        diversifier so the moderator scores on the same surface.
        """
        if payload is None:
            return ''
        if not isinstance(payload, Mapping):
            # Defensive: the gate may hand us a payload from a malformed
            # upstream — refuse rather than silently dropping the check.
            raise EgressGuardError(f"LexicalBlocklistModerator: payload must be a Mapping, got {type(payload).__name__}")
        parts: List[str] = []
        for fld in self._config.payload_fields:
            if fld in payload and payload[fld] is not None:
                parts.append(str(payload[fld]))
        return ' '.join(parts)

    def moderate(self, payload: Mapping[str, Any]) -> ModeratorVerdict:
        """Tokenise the payload and flag the item iff any banned token is present.

        :param payload: Mapping[str, Any] - The ``RankedItem.payload`` dict
        :return: ModeratorVerdict - With ``flagged=True`` iff a banned token matched
        :raises EgressGuardError: When payload extraction or tokenisation fails
            in a way the caller must surface (rather than silently passing the item)
        """
        try:
            doc_text = self._extract_doc_text(payload)
            tokens = tokenize_lexical(text=doc_text, min_term_length=self._config.min_term_length, max_terms=self._config.max_terms, stopwords=self._config.stopwords,)
        except EgressGuardError:
            raise
        except (ValueError, TypeError) as e:
            # tokenize_lexical raises ValueError on bad bounds; re-package.
            raise EgressGuardError(f"LexicalBlocklistModerator: tokenisation failed: {e}") from e
        if not tokens:
            return ModeratorVerdict(flagged=False, matched_terms=[], truncated_matches=False)
        seen: Dict[str, None] = {}
        truncated = False
        for tok in tokens:
            if tok in self._banned and tok not in seen:
                if len(seen) >= _MAX_REPORTED_MATCHES:
                    truncated = True
                    break
                seen[tok] = None
        if not seen:
            return ModeratorVerdict(flagged=False, matched_terms=[], truncated_matches=False)
        return ModeratorVerdict(flagged=True, matched_terms=list(seen.keys()), truncated_matches=truncated)
