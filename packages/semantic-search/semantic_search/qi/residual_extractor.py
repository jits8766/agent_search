"""Residual-kind classifier for the QI engine.

Two-path strategy — no brittle word lists required for common cases:

  Path A (keyword-entity-first): When L0 (L0LLMFilterExtractor) already extracted a
    keyword entity (keyword_contains / keyword_starts_with / keyword_ends_with)
    or a soft concept entity, those values ARE the semantic concept. Return them
    directly as semantic_query without any heuristic token stripping. Covers
    "coffee shop domains", "find brandable tech names", "show me saas domains".

  Path B (token-stripping fallback): When no keyword entity was extracted, strip
    the filter entity values (TLD strings, auction type labels, etc.) from the
    query tokens, then remove a small universal stop list (function words +
    platform-specific terms). The residual is what's left.

    The navigational_tokens config list is intentionally kept short (~15 tokens).
    It should only contain:
      - Platform-specific high-DF terms that appear in nearly every query but carry
        zero discriminative signal ("domain", "domains", "auction", "listing").
      - Universal grammatical function words (determiners, prepositions, pronouns)
        that entity extraction can never capture because they are not filter values.
    Do NOT add content words (adjectives, nouns) — those are semantic signal.

Location context stripping (build_semantic_encode_text):
  Uses spaCy en_core_web_sm NER to detect GPE/LOC entities in user location framing
  and strip them before the text reaches the dense encoder. Lazy-loads on first call;
  soft-fails to passthrough when unavailable. Install with: uv pip install ".[rag]"

Called from QIEngine._make_intent when QIConfig.residual.enabled=True.
"""
from __future__ import annotations

import re
from typing import Any, FrozenSet, List, Optional, Set, Tuple

from semantic_search.config.models import ResidualQIConfig
from semantic_search.contracts import Entity, QueryIntent
from semantic_search.core.logging_utils import get_logger
from semantic_search.qi.regex_entity_extractor import numeric_filter_surfaces

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# ── spaCy location context stripping ────────────────────────────────────────
# Lazy-loaded en_core_web_sm NER instance. None until first call or if unavailable.
_spacy_nlp = None
_spacy_load_attempted = False

# Prepositions that, when appearing within the window before a GPE/LOC entity,
# indicate user location framing rather than domain search intent.
# "from" only — "in" is too generic and appears in legitimate category searches.
_LOCATION_PREPS: FrozenSet[str] = frozenset({"from"})

# Token look-back window: check this many tokens before GPE start for a prep.
# 2 covers a direct prep + GPE pair as well as a one-token gap (e.g. auxiliary).
_LOCATION_GPE_WINDOW: int = 2


def _get_spacy_nlp() -> Optional[Any]:
    """Lazy-load spaCy en_core_web_sm (NER only). Returns None if unavailable."""
    global _spacy_nlp, _spacy_load_attempted
    if _spacy_load_attempted:
        return _spacy_nlp
    _spacy_load_attempted = True
    try:
        import spacy  # noqa: PLC0415
        _spacy_nlp = spacy.load(
            "en_core_web_sm",
            disable=["tagger", "parser", "attribute_ruler", "lemmatizer"],
        )
        logger.info("spacy_ner_loaded model=en_core_web_sm")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"spacy_ner_unavailable error_type={type(exc).__name__} error={exc} - location context stripping disabled")
    return _spacy_nlp


def _strip_location_context(text: str) -> str:
    """Strip GPE/LOC entities preceded by a location-framing preposition using spaCy NER.

    Condition: a _LOCATION_PREPS token within _LOCATION_GPE_WINDOW tokens before
    the entity start. Soft-fails to original text on any error or when unavailable.

    :param text: str - Query text (normalized or condensed)
    :return: str - Text with location-context entity tokens removed
    """
    nlp = _get_spacy_nlp()
    if nlp is None:
        return text
    try:
        doc = nlp(text)
        drop: Set[int] = set()
        for ent in doc.ents:
            if ent.label_ not in ("GPE", "LOC"):
                continue
            window_start = max(0, ent.start - _LOCATION_GPE_WINDOW)
            context_tokens = {doc[i].lower_ for i in range(window_start, ent.start)}
            if context_tokens & _LOCATION_PREPS:
                for i in range(ent.start, ent.end):
                    drop.add(i)
        if not drop:
            return text
        kept = [tok.text for i, tok in enumerate(doc) if i not in drop]
        result = " ".join(kept).strip()
        logger.debug(f"spacy_location_strip_applied original_len={len(text)} stripped_len={len(result)} dropped_spans={len(drop)}")
        return result if result else text
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"spacy_location_strip_error error_type={type(exc).__name__} error={exc}")
        return text


# Slots whose extracted values ARE the semantic concept.
# Path A: present -> skip all heuristic stripping, return value directly.
# similar_to carries seed SLD strings that encode the brand concept for ANN retrieval.
_KEYWORD_SLOTS: FrozenSet[str] = frozenset({
    "keyword_contains", "keyword_starts_with", "keyword_ends_with", "similar_to",
})

# Soft-chip entity slots that represent brand/theme concepts (emitted by LLM).
# Also Path A candidates when the LLM extracted a concept soft-entity.
_SOFT_CONCEPT_CHIP = "soft"

# Soft/meta slots that must NOT become ANN encode text (negatives, modes, counts).
_ENCODE_SKIP_SLOTS: FrozenSet[str] = frozenset({
    "topic_exclude",
    "keyword_contains_exclude",
    "keyword_match_mode",
    "word_count_min",
    "word_count_max",
    "lifecycle_state",
    "lifecycle_disjunction",
})

# TLD-only filter slots. The semantic-encode-text builder strips just these from
# the normalized fallback so the TLD literal can never reach any encode leg, even
# when no full residual was computed (navigational / empty / disabled-residual).
_TLD_SLOTS: FrozenSet[str] = frozenset({"tld", "tldExcludeList"})

# Hard-chip slots whose surface value IS the semantic concept and must survive into the
# encode text (keyword / brand / topic signal). Every other hard chip is a structural
# filter literal (tld, price number, auction word, length, char, time) and is stripped so
# the vector leg ranks on the concept, never on a filter number like "50".
_CONCEPT_KEEP_SLOTS: FrozenSet[str] = frozenset({
    "keyword_contains", "keyword_starts_with", "keyword_ends_with", "keyword_phrase", "similar_to", "topic_include",
})


def _token_set(text: Any) -> Set[str]:
    """Return lowercase alphanumeric tokens from a scalar or list value."""
    if text is None:
        return set()
    combined = " ".join(str(v) for v in text) if isinstance(text, list) else str(text)
    return set(_TOKEN_RE.findall(combined.lower()))


def strip_filter_literals(normalized: str, entities: List[Entity], nav_tokens: FrozenSet[str] = frozenset()) -> List[str]:
    """Return ``normalized`` tokens with every entity-value token + nav stop-token removed.

    Pure helper shared by Path B residual extraction and
    :func:`build_semantic_encode_text`. Booleans carry no surface token to strip
    and are skipped.

    :param normalized: str - Lowercased, whitespace-collapsed query text
    :param entities: List[Entity] - Entities whose surface values to remove
    :param nav_tokens: FrozenSet[str] - Additional stop-tokens to remove
    :return: List[str] - Surviving query tokens, in original order
    """
    strip_tokens: Set[str] = set()
    for ent in entities:
        if ent.value is None or isinstance(ent.value, bool):
            continue
        strip_tokens |= _token_set(ent.value)
    query_tokens = _TOKEN_RE.findall(normalized)
    return [t for t in query_tokens if t not in strip_tokens and t not in nav_tokens]


def build_semantic_encode_text(normalized: str, entities: List[Entity], semantic_query: Optional[str], nav_tokens: FrozenSet[str] = frozenset()) -> str:
    """Return the text every retrieval leg embeds.

    Residual concept when present, else the normalized query with hard-filter
    literals (tld, price, auction, length, char, time) stripped and concept
    keyword/topic tokens kept.
    """
    if semantic_query:
        return semantic_query
    if not normalized:
        return normalized
    strip_entities = [e for e in entities if (e.name in _TLD_SLOTS or getattr(e, 'chip_kind', 'soft') == 'hard') and e.name not in _CONCEPT_KEEP_SLOTS]
    # A numeric entity's value carries only the parsed number ("5"), never the
    # comparator/coordination/unit surface ("under", "or", "chars", the dropped "3").
    # Subtract the full matched numeric-filter surface so no operator/filter word
    # reaches the dense leg. Single source of truth is the extractor's own matchers.
    surface_tokens = numeric_filter_surfaces(normalized)
    stripped = strip_filter_literals(normalized, strip_entities, surface_tokens | nav_tokens)
    if stripped:
        candidate = " ".join(stripped)
    elif surface_tokens:
        # Emptied by a numeric-filter surface -> pure numeric-filter query. Return ""
        # rather than resurrecting the full normalized query, which would re-embed the
        # operator/coordination/unit tokens ("under", "or", "chars") into the dense leg.
        candidate = ""
    else:
        # Degenerate non-numeric case (e.g. TLD-only): preserve verbatim fallback.
        candidate = normalized
    return _strip_location_context(candidate) if candidate else ""


def semantic_encode_text_for(intent: QueryIntent) -> str:
    """Retriever-facing accessor for the TLD-safe encode text of ``intent``.

    Prefers the contract field ``semantic_encode_text`` (set by the QI engine),
    then ``semantic_query``, then re-derives a TLD-stripped text from the slices.
    The final fallback means wrapper / cache-hit / legacy intents that never set
    the field still cannot leak a TLD token into any encode leg.

    LLM-extracted keyword terms (``IntentSlice.keywords``) are appended when they
    introduce tokens not already present in the base encode text — deduplicated
    across slices and against the base so no term reaches the encoder twice.

    :param intent: QueryIntent - Classified intent (or a per-slice wrapper)
    :return: str - Text to embed
    """
    if intent.semantic_encode_text is not None:
        base = intent.semantic_encode_text
    elif intent.semantic_query:
        base = intent.semantic_query
    else:
        all_entities = [e for s in intent.slices for e in s.entities]
        base = build_semantic_encode_text(intent.normalized_query, all_entities, None)

    kw_seen: set = set()
    extra: list = []
    base_tokens: set = set(_TOKEN_RE.findall((base or "").lower()))
    for s in intent.slices:
        for kw in s.keywords:
            term = str(kw.get("term") or "").strip()
            if not term:
                continue
            key = term.lower()
            if key in kw_seen:
                continue
            kw_seen.add(key)
            term_tokens = set(_TOKEN_RE.findall(key))
            if term_tokens - base_tokens:
                extra.append(term)
                base_tokens |= term_tokens

    if extra:
        return f"{base} {' '.join(extra)}".strip() if base else " ".join(extra)
    return base


def _classify(token_count: int, config: ResidualQIConfig) -> str:
    if token_count == 0:
        return "empty"
    if token_count < config.min_content_tokens_for_semantic:
        return "navigational"
    return "semantic"


def _entity_concept_value(ent: Entity) -> Optional[str]:
    """Return keyword or soft-concept text for one entity, else None."""
    if ent.value is None:
        return None
    if ent.name in _ENCODE_SKIP_SLOTS:
        return None
    if not (
        (ent.name in _KEYWORD_SLOTS and ent.chip_kind == "hard")
        or ent.chip_kind == _SOFT_CONCEPT_CHIP
    ):
        return None
    val = ent.value if isinstance(ent.value, str) else (
        " ".join(str(v) for v in ent.value) if hasattr(ent.value, "__iter__") else str(ent.value)
    )
    # topic_include=climate_tech becomes "climate tech" for denser embeddings.
    cleaned = val.strip().replace("_", " ")
    return cleaned or None


def listing_concept_encode_text(entities: List[Entity]) -> str:
    """Return dense/BM25 encode text from keyword and soft-concept entities only.

    Does not derive encode text from full-query token stripping.
    Returns an empty string when no concept entities are present.
    Callers should pass hard ``entities`` plus ``soft_entities`` when both exist.
    """
    parts: List[str] = []
    seen: Set[str] = set()
    for ent in entities:
        concept = _entity_concept_value(ent)
        if not concept:
            continue
        key = concept.lower()
        if key in seen:
            continue
        seen.add(key)
        parts.append(concept)
    return " ".join(parts).strip()


def extract_residual(normalized: str, entities: List[Entity], config: ResidualQIConfig) -> Tuple[Optional[str], Optional[str]]:
    """Return (semantic_query, residual_kind) for the given normalized query + entities.

    Path A — keyword-entity shortcut:
      If any keyword_contains / keyword_starts_with / keyword_ends_with entity was
      extracted (chip_kind='hard') OR any soft-concept entity was emitted (chip_kind='soft'),
      treat the union of those values as the semantic_query. No token stripping needed.

    Path B — heuristic token stripping:
      Strip entity filter values + navigational stop-tokens from the query, classify
      the remainder.

    :param normalized: str - Lowercased, whitespace-collapsed query text
    :param entities: List[Entity] - All entities from the primary intent slice
    :param config: ResidualQIConfig - Stop-list and classification threshold
    :return: (semantic_query, residual_kind) where semantic_query is None when kind
      is 'empty' or 'navigational'
    """
    if not normalized:
        return None, "empty"

    # ── Path A: keyword / soft-concept entities carry the semantic concept ───
    semantic_from_concepts = listing_concept_encode_text(entities)
    if semantic_from_concepts:
        kind = _classify(len(_TOKEN_RE.findall(semantic_from_concepts)), config)
        return (semantic_from_concepts if kind == "semantic" else None), kind

    # ── Path B: strip filter entity values + nav stop-tokens ────────────────
    # Also strip the full matched numeric-filter surface (comparator + coordinated
    # numbers + unit): an entity value carries only the collapsed bound ("5"), so
    # sibling numbers ("3"), the comparator ("under") and the unit ("chars") would
    # otherwise survive into the concept residual.
    nav_tokens: FrozenSet[str] = frozenset(t.lower() for t in config.navigational_tokens) | numeric_filter_surfaces(normalized)
    residual = strip_filter_literals(normalized, entities, nav_tokens)

    kind = _classify(len(residual), config)
    semantic_query = " ".join(residual) if kind == "semantic" else None
    return semantic_query, kind
