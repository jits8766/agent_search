"""Vague-quantifier resolver — converts qualitative phrases to concrete Entity constraints.

Runs after the L1+L2 classification pass in QIEngine._make_intent().
Only injects an Entity for a given slot when that slot is NOT already occupied by an
L0-extracted entity. Prevents accidental override of explicit user constraints.

All thresholds come from VagueQuantifierConfig so the resolver is fully config-driven.
Pattern tuples: (compiled_regex, slot_name, config_attr_name, exclusion_guard_re|None).
"""
from __future__ import annotations

import re
from typing import Any, FrozenSet, List, Optional, Tuple

from semantic_search.config.models import VagueQuantifierConfig
from semantic_search.contracts import Entity

_PatternRule = Tuple[re.Pattern, str, str, Optional[re.Pattern]]

# Phrases that negate "expensive" — when present the expensive→price_min rule must not fire
# because a separate price_max rule already handles "not too expensive" / "not expensive".
# Also: "look/sound expensive" is appearance (not a price floor).
_NEGATED_EXPENSIVE_RE = re.compile(
    r"\b(?:not\s+(?:too|to|very|overly)\s+expensive|not\s+expensive|reasonably\s+priced)\b"
    r"|\b(?:look|looks|sound|sounds)\s+expensive\b"
    r"|\bexpensive[- ]looking\b",
    re.IGNORECASE,
)
# Dual qualitative "high X and high Y" with no digits — omit soft metric invents.
_DUAL_HIGH_QUAL_RE = re.compile(
    r"\bhigh\s+(?:go\s*value|govalue|domain\s+authority|da|authority|"
    r"traffic|backlinks?|valuation)\b"
    r".{0,48}\bhigh\s+(?:go\s*value|govalue|domain\s+authority|da|"
    r"authority|traffic|backlinks?|valuation)\b",
    re.IGNORECASE,
)
_SOFT_OR_LINK_TRAFFIC_RE = re.compile(
    r"\b(?:some\s+)?(?:traffic|backlinks?|authority)\s+or\s+"
    r"(?:(?:some\s+)?(?:traffic|backlinks?|authority)|"
    r"(?:at\s+least\s+)?(?:be\s+)?(?:really\s+|very\s+)?brandable)",
    re.IGNORECASE,
)
_VALUATION_RELATED_SLOTS = frozenset({
    'govalue_min', 'govalue_max', 'minValuationPrice', 'maxValuationPrice',
})


def _build_rules() -> List[_PatternRule]:
    return [
        # Traffic ─────────────────────────────────────────────────────────────
        (re.compile(r"\b(?:decent|reasonable|moderate|some)\s+traffic\b", re.I), "traffic_min", "decent_traffic_floor", None),
        (re.compile(r"\b(?:significant|strong|high|good|heavy|solid)\s+traffic\b", re.I), "traffic_min", "strong_traffic_floor", None),
        # Price ───────────────────────────────────────────────────────────────
        (re.compile(r"\b(?:not\s+(?:too|to|very|overly)\s+expensive|not\s+expensive|reasonably\s+priced)\b", re.I), "price_max", "affordable_price_cap", None),
        (re.compile(r"\b(?:affordable|cheap(?:er)?|budget|inexpensive|low[- ]?cost|low[- ]?price|cheapest|most\s+affordable|lowest\s+price|less\s+expensive|lower\s+(?:price|cost))\b", re.I), "price_max", "affordable_price_cap", None),
        # Exclusion guard: do not set price_min when the query negates "expensive"
        # (e.g. "not too expensive") — the price_max rule above already handles that phrase.
        (re.compile(r"(?<!too )\b(?:expensive|high[- ]?end|pricey|high[- ]?priced)\b", re.I), "price_min", "expensive_price_floor", _NEGATED_EXPENSIVE_RE),
        # Domain age ──────────────────────────────────────────────────────────
        (
            re.compile(r"\b(?:mature|aged|established|well[- ]?aged)\s+(?:domain|domains|name|names)\b", re.I),
            "domain_age_min",
            "domain_age_mature_years",
            # Contrastive aged|fresh preference → no age invent (L0 grounding empty).
            re.compile(
                r"\baged\s+domains?\s+or\s+fresh(?:\s+domains?)?\b"
                r"|\bfresh\s+domains?\s+or\s+aged(?:\s+domains?)?\b",
                re.I,
            ),
        ),
        # Expiry ──────────────────────────────────────────────────────────────
        (re.compile(r"\b(?:expiring\s+soon|ending\s+soon|closing\s+soon|expire\s+soon)\b", re.I), "time_remaining_max", "expiring_soon_seconds", None),
        # Name length ─────────────────────────────────────────────────────────
        # Allow comma-separated intervening adjectives: "short, memorable domains".
        (re.compile(r"\bshort(?:,\s*\w+(?:\s+\w+)*)?\s+(?:domain|domains|name|names|sld)\b", re.I), "name_length_max", "short_name_max_chars", None),
        (re.compile(r"\blong(?:,\s*\w+(?:\s+\w+)*)?\s+(?:domain|domains|name|names|sld)\b", re.I), "name_length_min", "long_name_min_chars", None),
        # GoValue / premium ───────────────────────────────────────────────────
        # Allow comma-separated intervening adjectives: "premium, trust-evoking domains".
        (re.compile(r"\b(?:premium|top[- ]?quality|high[- ]?value)(?:,\s*[\w.-]+(?:\s+[\w.-]+)*|\s+[\w.-]+)?\s+(?:domain|domains|name|names)\b", re.I), "govalue_min", "premium_govalue_floor", None),
        # SEO authority ───────────────────────────────────────────────────────
        (
            re.compile(
                r"\b(?:high|strong|good|solid|great)\s+(?:domain\s+)?authority\b"
                r"|\bauthority\s+score\b"
                r"|\bseo\s+authority\b"
                r"|\bda\s+score\b"
                r"|\bpage\s+authority\b",
                re.I,
            ),
            "semrush_authority_min",
            "high_authority_min",
            None,
        ),
        # Backlinks / link equity ─────────────────────────────────────────────
        (
            re.compile(
                r"\b(?:strong|good|high|solid|decent)\s+backlinks?\b"
                r"|\bbacklink\s+juice\b"
                # "good seo" / "seo value" → topic_include or valuation — not TF invent.
                r"|\blink\s+(?:juice|equity|profile|power)\b"
                r"|\blink\s+building\s+value\b"
                r"|\bbacklinks?\s+profile\b",
                re.I,
            ),
            "majestic_tf_min",
            "strong_backlinks_tf_min",
            None,
        ),
    ]


# When L0 already extracted any of these, skip vague TF inject (parity with qie_only).
_BACKLINK_RELATED_SLOTS = frozenset({
    'majestic_tf_min', 'majestic_tf_max',
    'majestic_cf_min', 'majestic_cf_max',
    'majestic_backlinks_min', 'majestic_backlinks_max',
    'majestic_ref_domains_min', 'majestic_ref_domains_max',
    'semrush_backlinks_min', 'semrush_backlinks_max',
    'semrush_ref_domains_min', 'semrush_ref_domains_max',
})

# L0 traffic signal / bound already expresses intent — do not stack vague traffic_min.
_TRAFFIC_RELATED_SLOTS = frozenset({
    'traffic_min', 'traffic_max', 'has_web_traffic_signal', 'traffic_is_unknown',
    'traffic_proxy_min', 'traffic_proxy_max',
})

# Underpriced / below-market is not "cheap → price_max".
_PRICE_BELOW_MARKET_SLOTS = frozenset({'price_below_market'})

# End-urgency already expressed as FIND endTimeBefore — do not stack time_remaining_max soft.
_END_URGENCY_SLOTS = frozenset({
    'time_remaining_max', 'endTimeBefore', 'endTimeAfter',
})

# L0 rule 4 parity: bare cheap/affordable/mid-budget without a numeric price cue
# must NOT invent hard/soft price_max (full-path soft_signals would break all4).
_QUALITATIVE_PRICE_CUE_RE = re.compile(
    r"\b(?:not\s+(?:too|to|very|overly)\s+expensive|not\s+expensive|reasonably\s+priced|"
    r"affordable|cheap(?:er)?|budget|inexpensive|low[- ]?cost|low[- ]?price|"
    r"cheapest|most\s+affordable|lowest\s+price|less\s+expensive|"
    r"lower\s+(?:price|cost)|mid\s+budget|around\s+budget)\b",
    re.IGNORECASE,
)
_EXPLICIT_NUMERIC_PRICE_RE = re.compile(
    r"(?:"
    r"\$\s*\d"
    r"|\b(?:under|below|over|above|at\s+least|at\s+most|less\s+than|more\s+than|"
    r"up\s+to|capped\s+at|max(?:imum)?|min(?:imum)?|floor|between|from)\s+\$?\d"
    r"(?!\s*(?:chars?|characters?|letters?|words?|years?|yrs?|bids?|"
    r"backlinks?|referring|ref\s*domains?|visitors?))"
    r"|\b\d+(?:\.\d+)?\s*k\b"
    r"|\b\d{3,}\b"
    r")",
    re.IGNORECASE,
)
# "mid budget … around N" / bare "around N" is approximate band — not cheap→1000 invent.
_AROUND_OR_MID_BUDGET_RE = re.compile(
    r"\b(?:mid\s+budget|around\s+budget|around\s+\$?\d|approx(?:imately)?\s+\$?\d)\b",
    re.IGNORECASE,
)
# Bare "premium … domain" is quality/topic signal — not govalue_min invent.
# Require an explicit valuation-family cue (price $ alone is not enough).
_EXPLICIT_VALUATION_CUE_RE = re.compile(
    r"\b(?:govalue|go\s*value|appraised|appraisal|valuation|estibot\s+value)\b",
    re.IGNORECASE,
)


class VagueQuantifierResolver:
    """Resolves vague/qualitative phrases to concrete Entity filter constraints.

    :param config: VagueQuantifierConfig - Resolution thresholds
    """

    def __init__(self, config: VagueQuantifierConfig) -> None:
        self._config = config
        self._rules: List[_PatternRule] = _build_rules()

    def resolve(self, entities: List[Entity], normalized_query: str) -> List[Entity]:
        """Return new Entity objects for vague patterns not already covered by LLM extraction.

        Slots already present in ``entities`` are skipped — explicit user constraints are
        never overridden by vague-phrase inference. Callers should pass hard + soft L0
        entities so soft chips (e.g. ``has_web_traffic_signal``) block stacking.

        :param entities: List[Entity] - Already-extracted entities from L0 (L0LLMFilterExtractor)
        :param normalized_query: str - Lowercased, whitespace-collapsed query text
        :return: List[Entity] - New synthetic entities to merge into the primary slice
        """
        if not self._config.enabled:
            return []
        # Advisory speech-acts must stay empty — do not re-invent after reconcile wipe.
        from semantic_search.qi.advisory_patterns import (
            is_soft_advisory_no_inventory,
            is_strong_advisory,
        )
        if is_strong_advisory(normalized_query) or is_soft_advisory_no_inventory(normalized_query):
            return []
        # Dual "high X and high Y" with no numbers → keep empty (L0 rule-4 parity).
        if _DUAL_HIGH_QUAL_RE.search(normalized_query) and not re.search(r"\d", normalized_query):
            return []
        occupied: FrozenSet[str] = frozenset(e.name for e in entities)
        new_entities: List[Entity] = []
        from semantic_search.qi.entity_reconcile import qualitative_floor_scrub_blocks_slot
        for pattern, slot, cfg_attr, exclusion_guard in self._rules:
            if slot in occupied:
                continue
            # L0 already set a backlink/ref-domain/TF chip — do not stack vague TF.
            if slot == 'majestic_tf_min' and occupied & _BACKLINK_RELATED_SLOTS:
                continue
            # Valuation chip already owns "seo value" / govalue — no TF invent.
            if slot == 'majestic_tf_min' and occupied & _VALUATION_RELATED_SLOTS:
                continue
            # topic_include already carries niche (e.g. seo) — do not stack TF invent.
            if slot == 'majestic_tf_min' and 'topic_include' in occupied:
                continue
            # Soft OR traffic|backlinks|authority|brandable — no invented metric floors.
            if (
                slot in ('majestic_tf_min', 'semrush_authority_min', 'traffic_min')
                and _SOFT_OR_LINK_TRAFFIC_RE.search(normalized_query)
            ):
                continue
            # "some traffic" with L0 has_web_traffic_signal / traffic_* — no floor inject.
            if slot == 'traffic_min' and occupied & _TRAFFIC_RELATED_SLOTS:
                continue
            # Underpriced / below market ≠ affordable price_max:1000.
            if slot == 'price_max' and occupied & _PRICE_BELOW_MARKET_SLOTS:
                continue
            # Qualitative cheap/budget without numeric bound → omit (L0 scrub parity).
            if (
                slot == 'price_max'
                and _QUALITATIVE_PRICE_CUE_RE.search(normalized_query)
                and not _EXPLICIT_NUMERIC_PRICE_RE.search(normalized_query)
            ):
                continue
            # Mid/around budget approximate band → omit soft affordable_price_cap invent.
            if slot == 'price_max' and _AROUND_OR_MID_BUDGET_RE.search(normalized_query):
                continue
            # Bare premium without valuation cue → omit govalue_min invent.
            if slot == 'govalue_min' and not _EXPLICIT_VALUATION_CUE_RE.search(normalized_query):
                continue
            # L0 already set end urgency — skip stacking ending-soon soft seconds.
            if slot == 'time_remaining_max' and occupied & _END_URGENCY_SLOTS:
                continue
            if exclusion_guard is not None and exclusion_guard.search(normalized_query):
                continue
            if pattern.search(normalized_query):
                # Shared qualitative_floor_scrub (entity_reconcile_rules.json).
                if qualitative_floor_scrub_blocks_slot(normalized_query, slot):
                    continue
                value: Any = getattr(self._config, cfg_attr)
                new_entities.append(Entity(name=slot, value=value, confidence=0.75, source="fallback", chip_kind="soft"))
                occupied = occupied | {slot}
        return new_entities
