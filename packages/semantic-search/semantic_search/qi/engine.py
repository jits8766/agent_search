"""QI Engine - ensemble voter resolver for query routing and classification.

Architecture:
  - Classifiers run as a sequential cascade: L1 (semantic router) fires first;
    L2 (LLM) fires only afterward, and only when L1 confidence is below
    qi.llm.l1_skip_l2_confidence_threshold. L2 never runs in parallel with L1.
  - The L0 entity extractor (entities only, not a type vote) runs concurrently
    with L1; the aggregation and ngram gates are sub-millisecond sync checks.
  - Winning archetype is determined by weighted vote tally with optional veto.
  - High-confidence L1 short-circuits L2 to save latency and LLM cost.
  - A semantic intent cache (QISemanticIntentCache) sits above the ensemble:
    a hit returns a cached QueryIntent for a semantically similar past query without
    re-classifying.
  - An exact normalized-query cache (QIIntentResultCache) is checked first.
  - Multi-intent: the deterministic splitter fires first; each sub-query runs the
    full ensemble dispatch independently in parallel.
"""
import asyncio
import re
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

from semantic_search.calibration.calibrator import CalibratorRegistry
from semantic_search.config.models import (
    MultiIntentConfig,
    QIConfig,
    QIEnsembleConfig,
    QINormalizeConfig,
)
from semantic_search.core.exceptions import ConfigurationError, LLMError, QueryIntelligenceError, ValidationError
from semantic_search.core.logging_utils import get_logger
from semantic_search.qi.l0_llm_filter_extractor import L0LLMFilterExtractor, expand_gd_to_godaddy
from semantic_search.qi.llm_entity_extractor import _reconcile_keyword_meta_blocklist
from semantic_search.qi.entity_reconcile import apply_post_merge_reconcile
from semantic_search.qi.regex_entity_extractor import (
    RegexEntityExtractor,
    _SLOT_TO_FAMILY,
    numeric_authority as _regex_numeric_authority,
)
from semantic_search.qi.grounding import EntityGrounder, ground_hard_entities
from semantic_search.qi.llm_classifier import LLMClassifier
from semantic_search.qi.multi_intent_splitter import MultiIntentSplitter
from semantic_search.resilience.circuit_breaker import CircuitBreaker, CircuitOpenError
from semantic_search.contracts import ConfidenceSignals, Entity, IntentSlice, QueryIntent, SubIntentFilterSet
from llm_core.pricing import compute_call_cost_usd
from semantic_search.cache.intent_result_cache import QIIntentResultCache
from semantic_search.cache.keys import versioned_query_key
from semantic_search.cache.qi_semantic_intent_cache import QISemanticIntentCache
from semantic_search.qi.aggregation_intent_gate import AggregationIntentGate
from semantic_search.qi.residual_extractor import build_semantic_encode_text, extract_residual
from semantic_search.qi.semantic_router import SemanticRouter
from semantic_search.qi.vague_quantifier_resolver import VagueQuantifierResolver
from semantic_search.qi.term_disambiguator import TermDisambiguator
from semantic_search.qi.ngram_pre_gate import NgramPreGate
from semantic_search.qi.ensemble_resolver import EnsembleResolver, EnsembleResult, Vote, abstain
from semantic_search.qi.entity_type_voter import EntityTypeVoter

logger = get_logger(__name__)

# Detects 'containing both X and Y' / 'with both X and Y' - signals AND semantics in a single
# keyword_contains context. Used to inject keyword_match_mode='all' when multi-intent decomposition
# loses the 'both' constraint.
_KEYWORD_BOTH_AND_RE = re.compile(r'\bcontain(?:s|ing)?\s+both\b|\bwith\s+both\b', re.IGNORECASE)

# Signals that indicate the query contains structured filter constraints.
# When L1 wins the type-classification gate AND any of these patterns match,
# we still await the in-flight LLM so its archetype vote participates in ensemble scoring.
# Winner archetype comes from the ensemble; entities always come from L0 exclusively.
#
# Design: three structural (generic) patterns cover most numeric filter categories,
# plus named-concept triggers for each entity slot the extractor handles.
#
#   1. Numeric constraint  - comparator + number fires for price, bids, age,
#      traffic, TF/CF, authority, SV, CPC, backlinks, govalue, ref-domains.
#   2. Number + metric unit - fires for length, bids, age, traffic, backlinks.
#   3. Named domain metrics - single-term triggers for SEO/TLF/GoValue slots.
_FILTER_SIGNAL_RE = re.compile(
    r"""
    # ── TLD ────────────────────────────────────────────────────────────────────
      \.[a-z]{2,8}\b
        # dotted extension (.io, .com, .co.uk …)
    | \b(?:in|ending\s+in|extension)\s+\.?[a-z]{2,8}\b
        # "ending in io", "extension net", "in .com"
    | \b(?:com|net|org|io|co|ai|xyz|app|dev|us|uk|eu|de|fr|ca|au|nz|
           me|tv|cc|biz|info|tech|shop|store|site|online|pro|live|club|
           ly|gg|vc|ag|so|fm|am|pm|id|to|it|es|nl|be|pl|se|no|dk|fi|
           cn|jp|in|br|mx|za|ru|ae|sa|sg|hk|tw|kr)\s+(?:only\s+)?domains?\b
        # known TLD word + "domains" - "io domains", "net only domains"

    # ── Generic numeric constraint ──────────────────────────────────────────────
    # Fires for: price, bids, age, traffic, TF, CF, authority, SV, CPC,
    # backlinks, ref-domains, indexed pages, govalue - any "comparator + number".
    | \b(?:under|below|above|over|
          at\s+least|at\s+most|
          less\s+than|more\s+than|fewer\s+than|greater\s+than|
          older\s+than|younger\s+than|newer\s+than|
          between|from|min(?:imum)?|max(?:imum)?)\s+[$€£¥₹]?\s*[\d,]+
        # comparator word + optional currency + number
    | [<>]=?\s*[$€£¥₹]?\s*[\d,]+
        # operator form: >50, <=500, <$200

    # ── Currency / price (when comparator is absent) ────────────────────────────
    | [$€£¥₹]\s*[\d,]+
        # bare currency + number: $500, €100
    | \b[\d,]+\s*(?:usd|eur|gbp|jpy|cad|aud|inr|dollars?|euros?|pounds?)\b
        # number + currency word: "500 usd", "200 dollars"
    | \b(?:cheap(?:er|est)?|affordable?|inexpensive|budget|pricey|expensive)\b
        # fuzzy price adjective

    # ── Number + domain metric unit ─────────────────────────────────────────────
    # Fires for: name length, bids, age, traffic, backlinks, ref-domains.
    | \b[\d,]+\+?\s*
        (?:letter|char(?:acter)?|word|
           bid|
           visitor|visit|pageview|
           backlink|link|
           ref(?:erring)?\s*domain|
           year|month)s?\b

    # ── Name length ─────────────────────────────────────────────────────────────
    | \b(?:short|long|brief|tiny|concise)\b
    | \b(?:one|two|three|four|five|six|seven|eight|nine|ten)[\s-]word\b
    | \bname\s*length\b

    # ── Keyword slot (starts-with / ends-with / contains) ───────────────────────
    | \b(?:start(?:s|ing)?|begin(?:s|ning)?)\s+with\b
    | \bend(?:s|ing)?\s+with\b
    | \b(?:contain(?:s|ing)?|includ(?:es?|ing)?)\b
    | \bwith\s+the\s+(?:word|keyword|phrase)\b

    # ── Auction type ─────────────────────────────────────────────────────────────
    | \b(?:expir\w+|closeout|buy\s*now|buyout|premium\s+auction)\b

    # ── Time remaining ───────────────────────────────────────────────────────────
    | \b(?:ends?|closes?|expires?|expiring|closing)\s+(?:in|soon|today|tomorrow)\b
    | \bwithin\s+\d+\s*(?:min(?:ute)?|hour|day|week)s?\b
    | \b(?:last\s+chance|closing\s+soon|ending\s+soon|expiring\s+soon)\b

    # ── Named domain SEO / metric concepts ──────────────────────────────────────
    # Each term maps 1-to-1 onto a filter entity slot; seeing the term means
    # structured extraction will find something useful.
    | \b(?:majestic|semrush|ahrefs|moz)\b
    | \b(?:trust\s+flow|citation\s+flow)\b
    | \b(?:authority\s+score|domain\s+authority|page\s+authority)\b
    | \b(?:search\s+volume|monthly\s+searches?)\b
    | \bcpc\b
    | \b(?:backlinks?|inbound\s+links?)\b
    | \b(?:referring\s+domains?|ref\s+domains?)\b
    | \b(?:indexed\s+pages?|crawled\s+pages?)\b
    | \b(?:traffic|visitors?|pageviews?)\b
    | \b(?:govalue|valuation|appraised|estimated\s+(?:value|price))\b
    | \b(?:tlf|exact\s+match\s+tld|keyword\s+registrations?)\b
    | \b(?:has\s+(?:website|content)|active\s+website|developed\s+(?:tld|domain))\b

    # ── Domain age ───────────────────────────────────────────────────────────────
    | \b(?:aged|established|vintage|mature)\s+domains?\b
    | \b(?:registered|created|dropped)\s+(?:before|after|since)\s+\d{4}\b

    # ── Character / structural constraints ──────────────────────────────────────
    | \b(?:no|without|exclude)\s+(?:hyphen|number|digit|special|idn)\b
    | \bhyphens?\b | \bnumbers?\b | \bdigits?\b
    | \b(?:ascii|latin|english)[\s-]only\b
    | \b(?:idn|unicode|international(?:ized)?)\b
    | \b(?:all[\s-])?(?:vowels?|consonants?|alpha(?:numeric)?|numeric[\s-]only)\b
    | \bpalindrome\b

    # ── Quality ──────────────────────────────────────────────────────────────────
    | \b(?:high[\s-]quality|quality\s+(?:score|min|above|over|above))\b
    | \bpremium\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


# Advisory/guidance query detector. Fires when the query is seeking help, advice, or
# explanation rather than searching for domains. Used to veto guidance rerouting when
# the query is genuinely advisory (prevents structured-filter chips from triggering
# a domain-search reroute on "how do I buy .com domains").
#
# Matches:   "how do/does/can/should/often/long/far/soon/well/best ..."
#            "should I/we/you ...", "explain ...", "best approach ..."
#            "best X for Y", "is X worth ...", "what makes ...", "are X worth ..."
# No-match:  "how many ...", "how much ..." (aggregate analytics, not advisory)
_GUIDANCE_VETO_RE = re.compile(
    r"""
    \bhow\s+(?:do|does|can|should|would|could|to|often|long|far|soon|well|best)\b
    | \bshould\s+(?:i|we|you|one)\b
    | \bexplain\b
    | \bbest\s+approach\b
    | \bbest\s+\w+\s+for\b
    | \bis\s+(?:it|this|a|an)\s+worth\b
    | \bare\s+\w+\s+(?:still\s+)?worth\b
    | \bwhat\s+makes\b
    | \badvice\b
    | \badvice\s+for\b
    | \brecommend(?:ation)?\b
    | \bhelp\s+me\s+(?:choose|decide|pick|understand)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Browse-intent veto: prevents hard-filter entities from overriding explore routing
# when the query clearly expresses a "browse by category" intent, not a structured filter.
# "browse ai domains", "explore healthcare names", "look at gaming domains" etc.
_EXPLORE_BROWSE_VETO_RE = re.compile(
    r"""
    \b(?:browse|explore|look\s+(?:at|through)|show(?:\s+me)?\s+|discover\s+)\w
    | \b(?:ending|closing|expiring)\s+(?:today|tonight|soon|this\s+week|tomorrow|next)\b
    | \b(?:last\s+chance|final\s+hours?|closing\s+tonight|auctions?\s+(?:ending|closing|wrapping))\b
    | \b(?:just\s+listed|recently\s+added|new\s+arrivals?|fresh\s+listings?)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _has_filter_signals(query: str) -> bool:
    """Return True when the query contains patterns that suggest structured filter entities."""
    return bool(_FILTER_SIGNAL_RE.search(query))


# Maps hard-filter entity names to category strings for multi-filter counting (Path B).
# Two entities of different categories indicate search intent (tld + price, tld + auction_type, etc.).
_ENTITY_NAME_TO_FILTER_CAT: 'Dict[str, str]' = {
    'tld': 'tld', 'price_max': 'price', 'price_min': 'price',
    'auction_type': 'auction_type', 'name_length_max': 'length', 'name_length_min': 'length',
    'keyword_contains': 'keyword', 'keyword_starts_with': 'keyword', 'keyword_ends_with': 'keyword',
    'time_remaining_max': 'time', 'majestic_tf_min': 'seo', 'majestic_tf_max': 'seo',
    'domain_age_min': 'age', 'domain_age_max': 'age', 'traffic_min': 'traffic', 'traffic_max': 'traffic',
}


# Patterns that indicate an UPPER bound on name length.
# Exclusive (< N): under/less/fewer ONLY with an explicit length unit (chars/letters).
# Bare "under 500" is PRICE - never invent maxSldLen from it.
# "shorter than N" implies length even without unit.
_NAME_LEN_UPPER_EXCLUSIVE_RE = re.compile(
    r'\bshort(?:er)?\s+than\s+(\d+)\s*(?:char(?:acter)?s?|letters?|chars?)?'
    r'|\b(?:under|less\s+than|fewer\s+than)\s+(\d+)\s+(?:char(?:acter)?s?|letters?|chars?)\b',
    re.IGNORECASE,
)
_NAME_LEN_UPPER_INCLUSIVE_RE = re.compile(
    r'\b(?:at\s+most|no\s+more\s+than|max(?:imum)?)\s+(\d+)\s+(?:char(?:acter)?s?|letters?|chars?)\b'
    r'|\b(\d+)\s*(?:char(?:acter)?s?|letters?|chars?)\s+or\s+less\b'
    r'|\bnothing\s+over\s+(\d+)\s*(?:char(?:acter)?s?|letters?|chars?)\b',
    re.IGNORECASE,
)
# Legacy alias used by callers that still import the old name.
_NAME_LEN_UPPER_RE = _NAME_LEN_UPPER_EXCLUSIVE_RE
# Patterns that indicate a LOWER bound on name length.
# "longer than N", "over N chars", "more than N chars", "at least N chars".
_NAME_LEN_LOWER_RE = re.compile(
    r'\b(?:long(?:er)?\s+than|over|more\s+than|greater\s+than|at\s+least)\s+(\d+)\s*(?:char(?:acter)?s?|letters?|chars?)',
    re.IGNORECASE,
)


def _correct_name_length_direction(slices: 'List[IntentSlice]', query: str) -> 'List[IntentSlice]':
    """Correct name_length_min/max when LLM swaps direction or uses inclusive N for exclusive under.

    Exclusive: 'under 5 chars' / 'shorter than 8' -> name_length_max = N-1 (drop min).
    Inclusive: 'at most 7 chars' / '4 chars or less' / 'nothing over 7 chars' -> max = N (drop min).
    Lower: 'longer than 5' -> name_length_min = N+1 when max was wrongly emitted.
    Bare 'under 500' (no length unit) is price - never injects name_length_max.
    """
    excl = _NAME_LEN_UPPER_EXCLUSIVE_RE.search(query)
    incl = _NAME_LEN_UPPER_INCLUSIVE_RE.search(query)
    lower_match = _NAME_LEN_LOWER_RE.search(query)
    if not excl and not incl and not lower_match:
        return slices

    def _first_int(m: 're.Match') -> int:
        for g in m.groups():
            if g is not None:
                return int(g)
        raise ValueError('length match missing number')

    result = []
    for s in slices:
        corrected_entities = []
        saw_max = False
        target_max: 'Optional[int]' = None
        if excl is not None:
            target_max = max(1, _first_int(excl) - 1)
        elif incl is not None:
            target_max = _first_int(incl)

        for e in s.entities:
            if target_max is not None and e.name in ('name_length_min', 'name_length_max', 'minLetters'):
                if e.name == 'name_length_max':
                    saw_max = True
                    if e.value != target_max:
                        corrected = Entity(
                            name='name_length_max', value=target_max,
                            confidence=e.confidence, source=e.source, chip_kind=e.chip_kind,
                        )
                        corrected_entities.append(corrected)
                        logger.info(
                            f'qi_name_length_ceiling_forced query={query[:80]!r} '
                            f'was={e.value!r} now={target_max} exclusive={excl is not None}'
                        )
                    else:
                        corrected_entities.append(e)
                elif e.name == 'name_length_min':
                    # Ceiling cue - drop exact/floor twin.
                    logger.info(
                        f'qi_name_length_min_dropped_ceiling query={query[:80]!r} value={e.value!r}'
                    )
                # minLetters on a length-ceiling phrase -> drop
                continue
            if e.name in ('name_length_min', 'name_length_max') and lower_match and target_max is None:
                n = int(lower_match.group(1))
                if e.name == 'name_length_min':
                    corrected_entities.append(e)
                elif e.name == 'name_length_max':
                    corrected = Entity(
                        name='name_length_min', value=n + 1,
                        confidence=e.confidence, source=e.source, chip_kind=e.chip_kind,
                    )
                    corrected_entities.append(corrected)
                    logger.info(
                        f'qi_name_length_direction_corrected max_to_min query={query[:80]!r} '
                        f'n={n} corrected_min={n + 1}'
                    )
                continue
            corrected_entities.append(e)
        if target_max is not None and not saw_max:
            corrected_entities.append(Entity(
                name='name_length_max', value=target_max,
                confidence=0.9, source='L0_llm', chip_kind='hard',
            ))
            logger.info(
                f'qi_name_length_max_injected query={query[:80]!r} value={target_max} '
                f'exclusive={excl is not None}'
            )
        result.append(replace(s, entities=corrected_entities))
    return result


# Numeric range slots where multiple extractions of the same slot should resolve
# to the most restrictive value. For upper-bound slots the minimum value is most
# restrictive; for lower-bound slots the maximum value is most restrictive.
# Controlled by slot name convention rather than a hardcoded list so new slots
# added to the extractor inherit the correct resolution automatically.
_RANGE_MAX_SLOT_NAMES: 'frozenset[str]' = frozenset({
    'price_max', 'name_length_max', 'time_remaining_max', 'word_count_max',
    'domain_age_max', 'traffic_max', 'bids_max', 'majestic_tf_max', 'majestic_cf_max',
    'semrush_authority_max', 'semrush_search_volume_max', 'semrush_cpc_max',
    'semrush_backlinks_max', 'semrush_indexed_pages_max', 'semrush_ref_domains_max',
    'govalue_max', 'majestic_backlinks_max', 'majestic_ref_domains_max',
})
_RANGE_MIN_SLOT_NAMES: 'frozenset[str]' = frozenset({
    'price_min', 'name_length_min', 'domain_age_min', 'traffic_min', 'bids_min',
    'majestic_tf_min', 'majestic_cf_min', 'semrush_authority_min',
    'semrush_search_volume_min', 'semrush_cpc_min', 'semrush_backlinks_min',
    'semrush_indexed_pages_min', 'semrush_ref_domains_min', 'govalue_min',
    'majestic_backlinks_min', 'majestic_ref_domains_min',
})

# Value-discovery vocabulary: queries containing these substrings are hybrid search
# intent regardless of centroid similarity. Checked in _resolve_l0_fallback_type
# as a low-cost override before calling best_guess().
_VALUE_DISCOVERY_SIGNALS: 'frozenset[str]' = frozenset({
    # Value / hidden-gem vocab
    'hidden gem', 'sleeper pick', 'sleeper domain', 'sleeper domains', 'sleeper',
    'undervalued', 'underpriced', 'under the radar', 'flying under', 'overlooked',
    'gems nobody', 'nobody noticed', 'good value', 'best value',
    # Resale / investor vocab
    'flip potential', 'resale potential', 'domain flip', 'investor flip', 'domain investor',
    'upside potential', 'below market', 'investment grade domain', 'long-term value',
    'bargain domain', 'bargain domains', 'bargain names', 'bargain brandable', 'bargain brandables',
    'below its value', 'liquidity potential',
    # Expired / dropped / backorder vocab
    'expired domains', 'expired .com', 'expired ai', 'expired one word',
    'expired brandable', 'expired with', 'expiring domains', 'old expired',
    'recently expired', 'buy now expired', 'backorder worthy', 'backorder',
    'dropping .com', 'auction domains', 'auction ending',
    # Fragment / partial-query vocab
    'ending soon traffic',
    # Category / opportunity vocab
    'future category', 'growing sectors', 'growing sector',
})

def _entity_norm_key(entity: 'Entity') -> 'Tuple[str, object]':
    """Return a dedup key of (name, normalized_value) for an Entity.

    Normalization:
      list  -> sorted tuple of lowercased str items
      other -> lowercased str of the value
    """
    val = entity.value
    if isinstance(val, list):
        norm_val: object = tuple(sorted(str(v).lower() for v in val))
    else:
        norm_val = str(val).lower()
    return (entity.name, norm_val)


def _resolve_range_entity(existing: 'Entity', candidate: 'Entity') -> 'Entity':
    """Return the more restrictive of two entities for the same numeric range slot.

    For upper-bound slots (_RANGE_MAX_SLOT_NAMES) the entity with the smaller
    numeric value is more restrictive. For lower-bound slots (_RANGE_MIN_SLOT_NAMES)
    the entity with the larger numeric value is more restrictive. Falls back to
    keeping the existing entity when values cannot be compared numerically.
    """
    try:
        ex_v = float(existing.value)  # type: ignore[arg-type]
        cd_v = float(candidate.value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return existing
    if existing.name in _RANGE_MAX_SLOT_NAMES:
        return candidate if cd_v < ex_v else existing
    if existing.name in _RANGE_MIN_SLOT_NAMES:
        return candidate if cd_v > ex_v else existing
    return existing


def _dedup_entities(entities: 'List[Entity]') -> 'List[Entity]':
    """Return entities with duplicates removed.

    For each slot name, at most one entity survives. Resolution policy:
      - Numeric upper-bound slots: keep the entity with the smallest value (tightest ceiling).
      - Numeric lower-bound slots: keep the entity with the largest value (tightest floor).
      - All other slots: keep the first occurrence (first-wins, order preserved).

    List-valued categorical slots (e.g. tld, auction_type) are deduplicated on
    exact (name, normalized_value) pairs as before - two tld entities with distinct
    value sets both survive so the caller can union-merge them downstream.
    """
    seen_exact: 'set[Tuple[str, object]]' = set()
    range_best: 'Dict[str, Entity]' = {}
    first_order: 'List[Entity]' = []
    for e in entities:
        if e.name in _RANGE_MAX_SLOT_NAMES or e.name in _RANGE_MIN_SLOT_NAMES:
            existing = range_best.get(e.name)
            if existing is None:
                range_best[e.name] = e
                first_order.append(e)
            else:
                resolved = _resolve_range_entity(existing, e)
                if resolved is not existing:
                    range_best[e.name] = resolved
                    idx = next((i for i, x in enumerate(first_order) if x is existing), None)
                    if idx is not None:
                        first_order[idx] = resolved
        else:
            key = _entity_norm_key(e)
            if key not in seen_exact:
                seen_exact.add(key)
                first_order.append(e)
    return first_order


def _dedup_merged_slice_entities(slices: 'List[IntentSlice]') -> 'List[IntentSlice]':
    """Dedup entities across all slices in the multi-intent merged set.

    Walks slices in order (highest-priority first). For numeric range slots, the
    most restrictive value wins regardless of slice order. For all other slots, the
    first occurrence wins and its confidence is preserved. Slices with no surviving
    entities are kept (empty entities list is valid).
    """
    seen_exact: 'set[Tuple[str, object]]' = set()
    range_best: 'Dict[str, Entity]' = {}
    range_slice_idx: 'Dict[str, int]' = {}
    deduped: 'List[IntentSlice]' = []
    for s_idx, s in enumerate(slices):
        surviving: 'List[Entity]' = []
        for e in s.entities:
            if e.name in _RANGE_MAX_SLOT_NAMES or e.name in _RANGE_MIN_SLOT_NAMES:
                existing = range_best.get(e.name)
                if existing is None:
                    range_best[e.name] = e
                    range_slice_idx[e.name] = s_idx
                    surviving.append(e)
                else:
                    resolved = _resolve_range_entity(existing, e)
                    if resolved is not existing:
                        prior_sidx = range_slice_idx[e.name]
                        if prior_sidx == s_idx:
                            idx = next((i for i, x in enumerate(surviving) if x is existing), None)
                            if idx is not None:
                                surviving[idx] = resolved
                        else:
                            old_slice = deduped[prior_sidx]
                            old_ents = [resolved if x is existing else x for x in old_slice.entities]
                            deduped[prior_sidx] = replace(old_slice, entities=old_ents)
                        range_best[e.name] = resolved
            else:
                key = _entity_norm_key(e)
                if key not in seen_exact:
                    seen_exact.add(key)
                    surviving.append(e)
        deduped.append(replace(s, entities=surviving))
    return deduped


def _merge_multi_keyword_contains(slices: 'List[IntentSlice]') -> 'List[IntentSlice]':
    """Combine per-sub-intent keyword_contains strings into one list-valued entity.

    When multi-intent splitting processes "domains containing X or Y", each sub-query
    emits a single-string keyword_contains entity.  After dedup these appear as separate
    entities across slices; the search API expects a single entity with value=[X, Y] and
    mode='any'.  This pass collects all string-valued keyword_contains entities across all
    slices, promotes them to a single list entity on the first slice that carries one, and
    removes the extras from the remaining slices.  List-valued entities (already merged by
    the extractor) are kept intact.
    """
    # Gather all string-valued keyword_contains values across slices in order.
    kw_strings: 'List[str]' = []
    for s in slices:
        for e in s.entities:
            if e.name == 'keyword_contains' and isinstance(e.value, str):
                kw_strings.append(e.value)
    if len(kw_strings) <= 1:
        # Nothing to merge - either zero or one string entity (no OR semantics needed).
        return slices

    seen_kw: 'set[str]' = set()
    unique_kw: 'List[str]' = []
    for v in kw_strings:
        vl = v.lower()
        if vl not in seen_kw:
            seen_kw.add(vl)
            unique_kw.append(v)

    merged_inserted = False
    result: 'List[IntentSlice]' = []
    for s in slices:
        new_entities: 'List[Entity]' = []
        for e in s.entities:
            if e.name == 'keyword_contains' and isinstance(e.value, str):
                if not merged_inserted:
                    # Replace first occurrence with the merged list entity.
                    merged_entity = Entity(
                        name='keyword_contains',
                        value=unique_kw,
                        confidence=e.confidence,
                        source=e.source,
                        chip_kind=e.chip_kind,
                    )
                    new_entities.append(merged_entity)
                    merged_inserted = True
                # Drop all subsequent string occurrences.
            else:
                new_entities.append(e)
        result.append(replace(s, entities=new_entities))
    return result


def _resolve_singleton_conflicts(
    slices: 'List[IntentSlice]',
    singleton_names: 'frozenset[str]',
) -> 'List[IntentSlice]':
    """Enforce first-entity-wins for slots that should have at most one value.

    Multi-intent splitting may produce one correct entity from the main sub-query
    (e.g. similar_to=['openai'] from "similar to openai.com") and a noisy entity
    from a fragment sub-query (e.g. similar_to='open' from a bare "open" token).
    _dedup_merged_slice_entities deduplicates by (name, value), so both survive.
    This function keeps only the FIRST entity encountered per singleton slot name,
    regardless of value, and drops subsequent duplicates.

    :param slices: List[IntentSlice] - Merged, deduped slices from sub-intent fan-out
    :param singleton_names: frozenset[str] - Entity names that must not repeat
    :return: List[IntentSlice] - Slices with at most one entity per singleton slot
    """
    if not singleton_names:
        return slices
    seen_singletons: 'set[str]' = set()
    result: 'List[IntentSlice]' = []
    for s in slices:
        new_entities: 'List[Entity]' = []
        for e in s.entities:
            if e.name in singleton_names:
                if e.name not in seen_singletons:
                    seen_singletons.add(e.name)
                    new_entities.append(e)
            else:
                new_entities.append(e)
        result.append(replace(s, entities=new_entities))
    return result


def _resolve_singletons_max_permissive(
    slices: 'List[IntentSlice]',
    singleton_names: 'frozenset[str]',
) -> 'List[IntentSlice]':
    """Resolve singleton slot conflicts using the most permissive numeric value.

    For slots whose name ends in ``_max``: take the maximum across all sub-intent
    values so the merged filter covers every sub-intent's upper bound.
    For slots whose name ends in ``_min``: take the minimum.
    Non-numeric values and unrecognized name patterns fall back to first-seen.

    :param slices: List[IntentSlice] - Merged, deduped slices from sub-intent fan-out
    :param singleton_names: frozenset[str] - Entity names that must not repeat
    :return: List[IntentSlice] - Slices with at most one entity per singleton slot
    """
    if not singleton_names:
        return slices
    all_values: 'Dict[str, List[Any]]' = {}
    all_entities_by_name: 'Dict[str, Entity]' = {}
    for s in slices:
        for e in s.entities:
            if e.name in singleton_names:
                all_values.setdefault(e.name, []).append(e.value)
                all_entities_by_name.setdefault(e.name, e)
    winners: 'Dict[str, Any]' = {}
    for name, vals in all_values.items():
        numeric_vals = []
        for v in vals:
            try:
                numeric_vals.append(float(v))
            except (TypeError, ValueError):
                pass
        if not numeric_vals:
            winners[name] = vals[0]
        elif name.endswith('_max'):
            winners[name] = max(numeric_vals)
        elif name.endswith('_min'):
            winners[name] = min(numeric_vals)
        else:
            winners[name] = vals[0]
    seen_singletons: 'set[str]' = set()
    result: 'List[IntentSlice]' = []
    for s in slices:
        new_entities: 'List[Entity]' = []
        for e in s.entities:
            if e.name in singleton_names:
                if e.name not in seen_singletons:
                    seen_singletons.add(e.name)
                    winner_val = winners.get(e.name, e.value)
                    ref = all_entities_by_name[e.name]
                    try:
                        typed_val = type(e.value)(winner_val) if not isinstance(e.value, list) else winner_val
                    except (TypeError, ValueError):
                        typed_val = winner_val
                    new_entities.append(Entity(
                        name=e.name, value=typed_val,
                        confidence=ref.confidence, source=ref.source, chip_kind=ref.chip_kind,
                    ))
            else:
                new_entities.append(e)
        result.append(replace(s, entities=new_entities))
    return result


def _merge_multi_keyword_contains_exclude(slices: 'List[IntentSlice]') -> 'List[IntentSlice]':
    """Combine per-sub-intent keyword_contains_exclude strings into one list-valued entity.

    Mirrors _merge_multi_keyword_contains but for exclusion keywords.  When multi-intent
    splitting breaks "excluding names containing coin, crypto, or block" into sub-queries,
    each extracts only a partial exclusion list.  This pass collects all string-valued
    keyword_contains_exclude entities across slices, promotes them to a single deduplicated
    list on the first slice that carries one, and removes the extras.
    """
    kw_strings: 'List[str]' = []
    for s in slices:
        for e in s.entities:
            if e.name == 'keyword_contains_exclude' and isinstance(e.value, str):
                kw_strings.append(e.value)
            elif e.name == 'keyword_contains_exclude' and isinstance(e.value, list):
                kw_strings.extend(str(v) for v in e.value)
    if not kw_strings:
        return slices
    seen_kw: 'set[str]' = set()
    unique_kw: 'List[str]' = []
    for v in kw_strings:
        vl = v.lower()
        if vl not in seen_kw:
            seen_kw.add(vl)
            unique_kw.append(v)
    if len(unique_kw) <= 1 and all(
        (isinstance(e.value, list) and len(e.value) >= 1)
        for s in slices for e in s.entities if e.name == 'keyword_contains_exclude'
    ):
        return slices
    merged_inserted = False
    result: 'List[IntentSlice]' = []
    for s in slices:
        new_entities: 'List[Entity]' = []
        for e in s.entities:
            if e.name == 'keyword_contains_exclude':
                if not merged_inserted:
                    merged_entity = Entity(
                        name='keyword_contains_exclude',
                        value=unique_kw,
                        confidence=e.confidence,
                        source=e.source,
                        chip_kind=e.chip_kind,
                    )
                    new_entities.append(merged_entity)
                    merged_inserted = True
            else:
                new_entities.append(e)
        result.append(replace(s, entities=new_entities))
    return result


def _merge_list_valued_entities(
    slices: 'List[IntentSlice]',
    list_merge_slot_names: 'frozenset',
) -> 'List[IntentSlice]':
    """Union-merge list-valued entities across all slices for slots in list_merge_slot_names.

    Multi-intent splits (e.g. "similar to google or apple") produce separate slices each
    with a single-item value for the same slot. This pass collects all values for each
    configured slot across all slices, deduplicates them, and rebuilds slices so the first
    occurrence of the slot carries the union list while all later occurrences are dropped.

    Soft slots (``topic_include``, …) live on ``soft_entities`` - merge those too so
    OR-niches survive as one list (parity with grounding ``finance|legal``).

    :param slices: List[IntentSlice] - Merged multi-intent slices (any order).
    :param list_merge_slot_names: frozenset[str] - Slot names to union-merge.
    :return: List[IntentSlice] - Slices with list-valued slots union-merged.
    """
    if not list_merge_slot_names or not slices:
        return slices
    slot_vals: 'Dict[str, List]' = {}
    slot_meta: 'Dict[str, tuple]' = {}
    soft_slot_names: 'set[str]' = set()

    def _collect(e: 'Entity', *, soft: bool) -> None:
        if e.name not in list_merge_slot_names:
            return
        vals = e.value if isinstance(e.value, list) else [e.value]
        if e.name not in slot_vals:
            slot_vals[e.name] = []
            slot_meta[e.name] = (e.confidence, e.source, e.chip_kind)
        if soft:
            soft_slot_names.add(e.name)
        for v in vals:
            if v not in slot_vals[e.name]:
                slot_vals[e.name].append(v)

    for s in slices:
        for e in s.entities:
            _collect(e, soft=False)
        for e in list(getattr(s, 'soft_entities', None) or []):
            _collect(e, soft=True)
    if not slot_vals:
        return slices
    merged_inserted: 'Dict[str, bool]' = {}
    result: 'List[IntentSlice]' = []
    for s in slices:
        new_entities: 'List[Entity]' = []
        for e in s.entities:
            if e.name not in slot_vals or e.name in soft_slot_names:
                if e.name not in slot_vals:
                    new_entities.append(e)
                continue
            if e.name not in merged_inserted:
                conf, source, chip_kind = slot_meta[e.name]
                new_entities.append(Entity(
                    name=e.name,
                    value=slot_vals[e.name],
                    confidence=conf,
                    source=source,
                    chip_kind=chip_kind,
                ))
                merged_inserted[e.name] = True
        new_soft: 'List[Entity]' = []
        for e in list(getattr(s, 'soft_entities', None) or []):
            if e.name not in slot_vals:
                new_soft.append(e)
                continue
            if e.name not in merged_inserted:
                conf, source, chip_kind = slot_meta[e.name]
                new_soft.append(Entity(
                    name=e.name,
                    value=slot_vals[e.name],
                    confidence=conf,
                    source=source,
                    chip_kind=chip_kind or 'soft',
                ))
                merged_inserted[e.name] = True
        result.append(replace(s, entities=new_entities, soft_entities=new_soft))
    return result




def _deconflict_merged_entities(
    slices: 'List[IntentSlice]',
    rules: 'List[List[str]]',
) -> 'List[IntentSlice]':
    """Apply post-merge entity deconfliction rules to a list of IntentSlices.

    Each rule is [flag_slot, presence_a, presence_b]: removes flag_slot from any slice
    where flag_slot is present AND at least one presence slot (presence_a or presence_b)
    is also present.  An empty string for presence_b disables that second check.
    Handles false-positive flag entities (e.g. traffic_is_unknown) that survive the
    per-sub-query deconfliction but conflict with explicit bound entities after merge.

    :param slices: Merged IntentSlices from the multi-intent or single-intent path.
    :param rules: Config-driven [[flag_slot, presence_a, presence_b], ...] rules.
    :return: Slices with conflicting flag entities removed.
    """
    if not rules:
        return slices
    all_names = {e.name for s in slices for e in s.entities}
    flags_to_remove: 'set[str]' = set()
    for rule in rules:
        if len(rule) != 3:
            continue
        flag, pa, pb = rule[0], rule[1], rule[2]
        if flag in all_names and (pa in all_names or (pb and pb in all_names)):
            flags_to_remove.add(flag)
    if not flags_to_remove:
        return slices
    out: 'List[IntentSlice]' = []
    for s in slices:
        filtered = [e for e in s.entities if e.name not in flags_to_remove]
        out.append(replace(s, entities=filtered))
    return out


def _deconflict_keyword_contains_values(
    slices: 'List[IntentSlice]',
    rules: 'List[List[str]]',
) -> 'List[IntentSlice]':
    """Remove specific values from keyword_contains(_exclude) when guard slots are present.

    Each rule is [keyword_value, guard_slot_a, guard_slot_b]: removes keyword_value
    from any keyword_contains or keyword_contains_exclude entity when guard_slot_a
    or guard_slot_b is present across any slice.  Prevents metric brand names
    (e.g. 'majestic', 'semrush') and auction-type labels (e.g. 'backorder') from
    bleeding into keyword filters when the corresponding typed slots are set.

    Also walks ``soft_entities`` (keyword chips are soft).

    :param slices: IntentSlices to filter.
    :param rules: Config-driven [[kw_value, guard_a, guard_b], ...] rules.
    :return: Slices with matching keyword values pruned.
    """
    if not rules:
        return slices
    all_names = {
        e.name
        for s in slices
        for e in list(s.entities) + list(getattr(s, 'soft_entities', None) or [])
    }
    values_to_remove: 'set[str]' = set()
    for rule in rules:
        if len(rule) != 3:
            continue
        kw_val, ga, gb = rule[0], rule[1], rule[2]
        if ga in all_names or (gb and gb in all_names):
            values_to_remove.add(kw_val.lower())
    if not values_to_remove:
        return slices
    _KW_SLOTS = frozenset({'keyword_contains', 'keyword_contains_exclude'})

    def _prune_list(ents: 'List[Entity]') -> 'List[Entity]':
        new_entities: 'List[Entity]' = []
        for e in ents:
            if e.name in _KW_SLOTS:
                if isinstance(e.value, list):
                    pruned = [v for v in e.value if str(v).lower() not in values_to_remove]
                    if pruned:
                        new_entities.append(Entity(
                            name=e.name,
                            value=pruned if len(pruned) > 1 else pruned[0],
                            confidence=e.confidence,
                            source=e.source,
                            chip_kind=e.chip_kind,
                        ))
                elif isinstance(e.value, str) and e.value.lower() in values_to_remove:
                    continue
                else:
                    new_entities.append(e)
            else:
                new_entities.append(e)
        return new_entities

    out: 'List[IntentSlice]' = []
    for s in slices:
        out.append(replace(
            s,
            entities=_prune_list(list(s.entities)),
            soft_entities=_prune_list(list(getattr(s, 'soft_entities', None) or [])),
        ))
    return out


def _drop_keyword_overlapping_topics(
    slices: 'List[IntentSlice]',
) -> 'List[IntentSlice]':
    """Drop keyword_contains values already covered by topic_include (niche ≠ contain cue).

    Multi-intent fragments often emit ``keyword_contains:legal`` alongside
    ``topic_include:legal`` for \"finance or legal niche\". Grounding/qie_only keep
    topics only - strip the keyword bleed here.
    """
    if not slices:
        return slices
    topics: 'set[str]' = set()
    for s in slices:
        for e in list(s.entities) + list(getattr(s, 'soft_entities', None) or []):
            if e.name != 'topic_include':
                continue
            vals = e.value if isinstance(e.value, list) else [e.value]
            for v in vals:
                if v is not None and str(v).strip():
                    topics.add(str(v).strip().lower())
    if not topics:
        return slices

    def _prune(ents: 'List[Entity]') -> 'List[Entity]':
        out_ents: 'List[Entity]' = []
        for e in ents:
            if e.name != 'keyword_contains':
                out_ents.append(e)
                continue
            if isinstance(e.value, list):
                kept = [v for v in e.value if str(v).strip().lower() not in topics]
                if not kept:
                    continue
                out_ents.append(e if kept == e.value else replace(
                    e, value=kept if len(kept) > 1 else kept[0],
                ))
            elif isinstance(e.value, str) and e.value.strip().lower() in topics:
                continue
            else:
                out_ents.append(e)
        return out_ents

    return [
        replace(
            s,
            entities=_prune(list(s.entities)),
            soft_entities=_prune(list(getattr(s, 'soft_entities', None) or [])),
        )
        for s in slices
    ]


def _apply_cue_reconcile_to_slices(
    slices: 'List[IntentSlice]',
    query: str,
    soft_slot_names: frozenset,
    hard_entity_names: frozenset,
) -> 'List[IntentSlice]':
    """Re-run cue reconciles on each slice against the full query text.

    Per-sub-query merge only sees split fragments, so OR/exact/stale/majestic cues on
    the original query are re-applied here after single- and multi-intent assembly.
    Soft entities are re-partitioned from the reconciled combined list.
    """
    if not slices or not query:
        return slices
    out: List[IntentSlice] = []
    for s in slices:
        soft_prev = list(getattr(s, 'soft_entities', None) or [])
        combined = list(s.entities) + soft_prev
        reconciled = apply_post_merge_reconcile(query, combined, hard_entity_names)
        hard = [e for e in reconciled if e.name not in soft_slot_names]
        soft = [e for e in reconciled if e.name in soft_slot_names]
        _pre = getattr(s, 'pre_ground_entities', None)
        _pre_list = list(_pre) if _pre is not None else list(s.entities or [])
        _pre_seen = {(e.name, repr(e.value)) for e in _pre_list if getattr(e, 'name', None)}
        for ent in hard:
            key = (ent.name, repr(ent.value))
            if key in _pre_seen:
                continue
            _pre_seen.add(key)
            _pre_list.append(ent)
        out.append(IntentSlice(
            query_type=s.query_type,
            entities=hard,
            confidence=s.confidence,
            raw_text=s.raw_text,
            slice_id=getattr(s, 'slice_id', None) or '',
            soft_entities=soft,
            pre_ground_entities=_pre_list,
            keywords=list(getattr(s, 'keywords', None) or []),
        ))
    return out


def _keep_hard_entities_on_suppress(slices: 'List[IntentSlice]', suppress_types: frozenset) -> 'List[IntentSlice]':
    """For advisory query types, keep hard filters; preserve soft on soft_entities.

    guidance/explore/analytics previously wiped ALL entities, which zeroed real
    filters when L1/L2 mis-routed a filter query. Hard chips stay on ``entities``.
    Soft chips stay on ``soft_entities`` (and soft-kind rows on ``entities`` are
    migrated there) so full-search soft_signals matches qie_only / grounding.
    """
    if not suppress_types or not slices:
        return slices
    out: List[IntentSlice] = []
    for s in slices:
        if s.query_type not in suppress_types:
            out.append(s)
            continue
        hard: List[Entity] = []
        soft = list(getattr(s, 'soft_entities', None) or [])
        seen_soft = {e.name for e in soft}
        for e in s.entities or []:
            if getattr(e, 'chip_kind', None) == 'hard':
                hard.append(e)
                continue
            # Soft-kind chips -> soft_entities, not hard FIND path.
            if e.name not in seen_soft:
                soft.append(replace(e, chip_kind='soft'))
                seen_soft.add(e.name)
        _pre = getattr(s, 'pre_ground_entities', None)
        out.append(IntentSlice(
            query_type=s.query_type,
            entities=hard,
            confidence=s.confidence,
            raw_text=s.raw_text,
            slice_id=getattr(s, 'slice_id', None) or '',
            soft_entities=soft,
            pre_ground_entities=list(_pre) if _pre is not None else None,
            keywords=list(getattr(s, 'keywords', None) or []),
        ))
    return out


def _l0_keywords_from_slice(slice_: Optional[IntentSlice]) -> List[Dict[str, Any]]:
    """Copy L0 topical keywords (``term`` / ``probability``) from an IntentSlice.

    Shared by ``extract_l0_filters`` (qie_only) and ``_classify_ensemble`` (full
    search) so both legs stamp the same keyword list shape. Thresholding
    (``qi.l0_llm_entity.keyword_min_probability``) already ran inside
    ``L0LLMFilterExtractor`` — this helper does not re-filter.
    """
    if slice_ is None:
        return []
    return list(getattr(slice_, 'keywords', None) or [])


def _merge_extractor_entities(
    regex_result: Optional[IntentSlice],
    llm_result: Optional[IntentSlice],
    query: str,
    soft_slot_names: frozenset,
    hard_entity_names: frozenset,
    *,
    llm_completed: bool,
) -> 'Tuple[List[Entity], List[Entity]]':
    """LLM extract wins when the LLM call completed; regex only when LLM unavailable.

    Full-search policy (hard + soft / keyword / topic patterns):
      - ``llm_completed=True`` -> keep LLM only (including empty). Discard regex
        entirely - regex L0 is not activated when the LLM call completed.
      - ``llm_completed=False`` -> regex whole-result fallback (hard + soft).
        Regex owns the same soft-slot patterns; they fire only in this branch
        (LLM absence), not as a merge/gap-fill against LLM soft.
    No per-slot gap-fill. Cue reconcile runs after the chosen extract.

    :return: Tuple[hard_entities, soft_entities] after scrub + cue reconcile
    """
    llm_hard = list(llm_result.entities) if llm_result is not None else []
    llm_soft = list(getattr(llm_result, 'soft_entities', None) or []) if llm_result is not None else []
    regex_all = list(regex_result.entities) if regex_result is not None else []
    regex_soft_extra = list(getattr(regex_result, 'soft_entities', None) or []) if regex_result is not None else []
    if llm_completed:
        chosen = llm_hard + llm_soft
        source = 'llm' if chosen else 'llm_empty'
    else:
        # Regex soft patterns (keyword/topic/…) activate only on LLM absence.
        chosen = regex_all + regex_soft_extra
        source = 'regex_fallback' if chosen else 'empty'
    logger.info(
        f"extractor_merge_policy source={source} llm_completed={llm_completed} "
        f"llm_hard={len(llm_hard)} llm_soft={len(llm_soft)} "
        f"regex_n={len(regex_all) + len(regex_soft_extra)} kept_n={len(chosen)}"
    )
    scrubbed = _reconcile_keyword_meta_blocklist(chosen)
    reconciled = apply_post_merge_reconcile(query or '', scrubbed, hard_entity_names)
    hard = [e for e in reconciled if e.name not in soft_slot_names]
    soft = [e for e in reconciled if e.name in soft_slot_names]
    return hard, soft


# A word/keyword count is meaningful only when the query names that unit.
_WORD_COUNT_SIGNAL_RE = re.compile(r"\b(?:keywords?|words?)\b", re.IGNORECASE)


def _drop_numeric_misattributions(
    entities: 'List[Entity]',
    query: str,
) -> 'List[Entity]':
    """Remove LLM numeric entities that contradict the deterministic query reading.

    The regex parser knows, from unit/currency anchoring, which metric family every
    number in the query belongs to (``numeric_authority``). An LLM-sourced numeric
    entity is dropped when either:

    - value cross-family - its value is a number the query anchors to a *different*
      family (e.g. a char-count number surfacing as a price), or
    - family precedence - the query anchors that family to specific numbers and the
      LLM value is not among them (the deterministic reading owns the family).

    Regex is never mixed back in here. When LLM returned ≥1 entity, merge already
    discarded regex entirely; scrubbed LLM slots stay empty rather than reintroducing
    regex false positives. Regex-only results (LLM empty/unavailable) pass through
    merge as the whole extract and are not removed here.

    Regex-sourced entities are never removed here. A word/keyword count with no unit
    token anywhere in the query is unfounded and is dropped regardless of source
    (regex counts always carry the unit token, so they are unaffected).
    """
    val_to_families, families = _regex_numeric_authority(query)
    word_signal = bool(_WORD_COUNT_SIGNAL_RE.search(query or ''))

    def _family_accepts_value(fam: str, value: float) -> bool:
        """True when ``value`` matches an anchored number for ``fam``.

        Exclusive bounds emit N±1 (under 100 -> traffic_max=99; above 15 -> min=16).
        Accept the exclusive neighbor so correct L0 chips are not wiped.
        """
        if fam in val_to_families.get(value, set()):
            return True
        # under/below N -> max slot N-1
        if fam in val_to_families.get(value + 1.0, set()):
            return True
        # above/over N -> min slot N+1
        if fam in val_to_families.get(value - 1.0, set()):
            return True
        return False

    kept: 'List[Entity]' = []
    for e in entities:
        fam = _SLOT_TO_FAMILY.get(e.name)
        if fam is None:
            kept.append(e)
            continue
        if not str(getattr(e, 'source', '')).startswith('L0_regex'):
            try:
                value = float(e.value)
            except (TypeError, ValueError):
                value = None
            # value cross-family: this number is anchored to a different family in the query
            if value is not None and value in val_to_families and fam not in val_to_families[value]:
                # Exclusive neighbor of an in-family anchor is not a cross-family hit
                # (under 100 traffic -> authority 100/traffic, L0 emits 99).
                if not _family_accepts_value(fam, value):
                    continue
            # family precedence: drop only when LLM value is not an anchored number
            # for this family (docstring contract). Matching values (under $50 -> 50)
            # must survive - blind `fam in families` previously wiped correct prices.
            if fam in families and (value is None or not _family_accepts_value(fam, value)):
                continue
        if fam == 'word_count' and not word_signal:
            continue
        kept.append(e)
    return kept


def _reconcile_numeric_misattribution_slices(slices: 'List[IntentSlice]', query: str) -> 'List[IntentSlice]':
    """Apply numeric-misattribution reconciliation across every slice of an intent.

    Multi-intent fan-out classifies each sub-query independently, so a deterministic
    reading and an LLM misread of the same constraint can land in different slices.
    The authority is computed over the whole query, so cross-slice contradictions are
    resolved consistently. Preserves ``soft_entities`` (replace, not IntentSlice()).
    """
    if not slices:
        return slices
    return [
        replace(s, entities=_drop_numeric_misattributions(list(s.entities), query))
        for s in slices
    ]


def _broadcast_slots_across_peer_slices(
    slices: 'List[IntentSlice]',
    broadcast_slots: 'frozenset[str]',
    peer_slots: 'frozenset[str]',
) -> 'List[IntentSlice]':
    """Copy broadcast-slot entities onto every peer-bearing slice when missing.

    Config-driven: ``cross_slice_broadcast_slots`` + ``cross_slice_broadcast_peer_slots``.
    When peer_slots is empty, broadcast onto all slices. First-seen value per
    broadcast slot wins. No-op when broadcast_slots is empty.
    """
    if not slices or not broadcast_slots:
        return slices
    donors: 'Dict[str, Entity]' = {}
    for s in slices:
        for e in s.entities:
            if e.name in broadcast_slots and e.name not in donors:
                donors[e.name] = e
    if not donors:
        return slices
    out: 'List[IntentSlice]' = []
    for s in slices:
        present = {e.name for e in s.entities}
        has_peer = (not peer_slots) or any(e.name in peer_slots for e in s.entities)
        if not has_peer:
            out.append(s)
            continue
        extras = [donors[name] for name in donors if name not in present]
        if not extras:
            out.append(s)
            continue
        out.append(replace(s, entities=list(s.entities) + extras))
    return out


def _deconflict_similar_to_vs_tld(entities: 'List[Entity]') -> 'List[Entity]':
    """Remove TLD tokens from similar_to when the same value is already a TLD filter.

    LLM entity extraction sometimes emits a 'similar domain' entity for TLD extension
    tokens ('.ai', '.io') that are really TLD hard-chip filters.  A similar_to value
    that duplicates a tld filter value is not a brand seed and would pollute the similarity retrieval pass.

    :param entities: Entity list from a single IntentSlice.
    :return: Entity list with TLD-duplicate values pruned from similar_to.
    """
    tld_vals: 'set[str]' = set()
    for e in entities:
        if e.name == 'tld' and isinstance(e.value, list):
            tld_vals.update(str(v).lower() for v in e.value)
    if not tld_vals:
        return entities
    out: 'List[Entity]' = []
    for e in entities:
        if e.name != 'similar_to':
            out.append(e)
            continue
        if isinstance(e.value, list):
            filtered_vals = [v for v in e.value if str(v).lower() not in tld_vals]
            if filtered_vals:
                out.append(Entity(name=e.name, value=filtered_vals, confidence=e.confidence,
                                  source=e.source, chip_kind=e.chip_kind))
        elif str(e.value).lower() not in tld_vals:
            out.append(e)
    return out


def _detect_slot_contradictions(
    slices: 'List[IntentSlice]',
    paired_slots: 'List[List[str]]',
) -> 'List[Tuple[str, str, float, float]]':
    """Return (min_slot, max_slot, min_val, max_val) for impossible filter combos.

    For each [min_slot, max_slot] pair in paired_slots: when both are present
    and min_val > max_val the combination produces an empty result set. Examples:
    name_length_min=15 + name_length_max=4, price_min=1000 + price_max=100.
    Reuses the existing paired_direction_slots config so no new config keys are needed.

    :param slices: List[IntentSlice] - Extracted intent slices to validate
    :param paired_slots: List[List[str]] - Config-driven [[min_slot, max_slot], ...] pairs
    :return: List of (min_slot, max_slot, min_val, max_val); empty when no contradiction
    """
    if not paired_slots or not slices:
        return []
    all_entities: 'Dict[str, object]' = {}
    for s in slices:
        for e in s.entities:
            if e.name not in all_entities:
                all_entities[e.name] = e.value
    contradictions: 'List[Tuple[str, str, float, float]]' = []
    for pair in paired_slots:
        if len(pair) != 2:
            continue
        min_slot, max_slot = pair[0], pair[1]
        if min_slot not in all_entities or max_slot not in all_entities:
            continue
        try:
            min_val = float(all_entities[min_slot])  # type: ignore[arg-type]
            max_val = float(all_entities[max_slot])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if min_val > max_val:
            contradictions.append((min_slot, max_slot, min_val, max_val))
    return contradictions


def _deconflict_numeric_auction_type_price(slices: 'List[IntentSlice]', numeric_type_ids: 'frozenset[str]') -> 'List[IntentSlice]':
    """Remove price_max/price_min whose integer value equals a numeric auction_type ID.

    L2 sometimes reads the numeric type identifier (e.g. 16 in 'auction type 16') as a price
    constraint, producing a spurious price_max=16. When auction_type holds a numeric ID and
    price_max or price_min equals that integer, drop the price slot.

    :param slices: List[IntentSlice] - Slices after L0+L2 merge
    :param numeric_type_ids: frozenset[str] - Digit-only auction type strings from config (e.g. {'16','20','38','39'})
    :return: List[IntentSlice] - Slices with spurious price slots removed
    """
    if not numeric_type_ids:
        return slices
    numeric_int_vals: 'set[int]' = set()
    for s in slices:
        for e in s.entities:
            if e.name == 'auction_type':
                vals = e.value if isinstance(e.value, list) else [e.value]
                for v in vals:
                    if str(v) in numeric_type_ids:
                        try:
                            numeric_int_vals.add(int(str(v)))
                        except (ValueError, TypeError):
                            pass
    if not numeric_int_vals:
        return slices
    result: 'List[IntentSlice]' = []
    for s in slices:
        cleaned = [e for e in s.entities if not (e.name in ('price_max', 'price_min') and isinstance(e.value, (int, float)) and int(e.value) in numeric_int_vals)]
        result.append(replace(s, entities=cleaned))
    return result


def _inject_keyword_match_mode_all(slices: 'List[IntentSlice]', query: str) -> 'List[IntentSlice]':
    """Inject keyword_match_mode='all' when query has 'containing both X and Y' but no mode entity.

    Multi-intent decomposition of 'containing both X and Y' loses the 'both' (AND) constraint:
    each sub-query gets a single keyword_contains without a mode entity. When the merged slices
    carry a multi-value keyword_contains but no keyword_match_mode, and the query has the
    'containing/with both ... and' pattern, inject mode='all' on the keyword-bearing slice.

    :param slices: List[IntentSlice] - Merged slices after multi-intent or single-intent path
    :param query: str - Normalized query text
    :return: List[IntentSlice] - Slices with keyword_match_mode='all' injected when applicable
    """
    if not _KEYWORD_BOTH_AND_RE.search(query):
        return slices
    has_multi_kw = any(e.name == 'keyword_contains' and isinstance(e.value, list) and len(e.value) > 1 for s in slices for e in s.entities)
    if not has_multi_kw:
        return slices
    if any(e.name == 'keyword_match_mode' for s in slices for e in s.entities):
        return slices
    result: 'List[IntentSlice]' = []
    injected = False
    for s in slices:
        if not injected and any(e.name == 'keyword_contains' and isinstance(e.value, list) for e in s.entities):
            mode_e = Entity(name='keyword_match_mode', value='all', confidence=0.95, source='fallback', chip_kind='hard')
            result.append(replace(s, entities=list(s.entities) + [mode_e]))
            injected = True
        else:
            result.append(s)
    return result


def _downgrade_excess_hard_chips(slices: 'List[IntentSlice]', max_hard: int, request_id: str) -> 'List[IntentSlice]':
    """Downgrade excess hard-chip entities to soft on the primary slice.

    Prevents candidate pool collapse when L2 extracts too many hard filters on
    low-confidence queries. The first max_hard hard entities (by position) are kept;
    any beyond that are set to chip_kind='soft' so they rank rather than filter.
    """
    if not slices:
        return slices
    primary = slices[0]
    hard_indices = [i for i, e in enumerate(primary.entities) if e.chip_kind == 'hard']
    if len(hard_indices) <= max_hard:
        return slices
    downgrade_set = set(hard_indices[max_hard:])
    new_entities = [
        Entity(name=e.name, value=e.value, confidence=e.confidence, source=e.source, chip_kind='soft')
        if i in downgrade_set else e
        for i, e in enumerate(primary.entities)
    ]
    logger.info(
        f"qi_hard_chip_downgrade request_id={request_id} "
        f"original_hard={len(hard_indices)} max_hard={max_hard} "
        f"downgraded={len(hard_indices) - max_hard}"
    )
    new_primary = replace(primary, entities=new_entities)
    return [new_primary] + list(slices[1:])


def normalize_query(query: str, max_length: int, *, normalize: QINormalizeConfig) -> str:
    """Normalize a query for hashing / encoding using ``qi.normalize`` settings.

    :param query: str - Raw query text
    :param max_length: int - Hard cap on returned length
    :param normalize: QINormalizeConfig - From ``qi.normalize`` (required)
    :return: str - Normalized query
    :raises ValidationError: When `query` is None / not a string / empty after stripping
    :raises ConfigurationError: When ``normalize`` is missing
    """
    if not isinstance(normalize, QINormalizeConfig):
        raise ConfigurationError("normalize_query requires qi.normalize config")
    if query is None or not isinstance(query, str):
        raise ValidationError("normalize_query requires a string input")
    if int(max_length) < 1:
        raise ConfigurationError("normalize_query max_length must be >= 1")
    cleaned = query.strip()
    if normalize.lowercase:
        cleaned = cleaned.lower()
    if normalize.normalize_quotes and normalize._quote_table is not None:
        cleaned = cleaned.translate(normalize._quote_table)
    if normalize.collapse_whitespace:
        cleaned = normalize._whitespace_re.sub(" ", cleaned).strip()
    if normalize.expand_gd_alias:
        cleaned = expand_gd_to_godaddy(cleaned)
    if normalize.strip_trailing_punctuation:
        cleaned = cleaned.rstrip(normalize.trailing_punctuation_chars).strip()
    if not cleaned:
        raise ValidationError("normalize_query: query is empty after normalization")
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length]
    return cleaned


class QIEngine:
    """Ensemble voter resolver for query routing and classification.

    :param config: QIConfig - Engine config bundle
    :param llm_classifier: Optional[LLMClassifier] - LLM voter
    :param entity_grounder: EntityGrounder - Inventory grounding for entity values
    :param max_query_length: int - Length cap applied during normalization
    :param circuit_breaker: CircuitBreaker - Guards the LLM call
    :param multi_intent_config: Optional[MultiIntentConfig] - Multi-intent config
    :param multi_intent_splitter: Optional[MultiIntentSplitter] - Splitter instance
    :param calibrator_registry: Optional[CalibratorRegistry] - Confidence calibrators
    :param intent_result_cache: Optional[QIIntentResultCache] - Exact-match result cache
    :param semantic_router: Optional[SemanticRouter] - Semantic voter
    :param semantic_intent_cache: Optional[QISemanticIntentCache] - Fuzzy intent cache
    :param term_disambiguator: Optional[TermDisambiguator] - Resolves polysemous TLD stems before encoding
    """

    def __init__(
        self,
        config: QIConfig,
        entity_grounder: EntityGrounder,
        max_query_length: int,
        circuit_breaker: CircuitBreaker,
        llm_classifier: Optional[LLMClassifier] = None,
        multi_intent_config: Optional[MultiIntentConfig] = None,
        multi_intent_splitter: Optional[MultiIntentSplitter] = None,
        calibrator_registry: Optional[CalibratorRegistry] = None,
        intent_result_cache: Optional[QIIntentResultCache] = None,
        semantic_router: Optional[SemanticRouter] = None,
        semantic_intent_cache: Optional[QISemanticIntentCache] = None,
        entity_extractor: Optional[L0LLMFilterExtractor] = None,
        regex_entity_extractor: Optional['RegexEntityExtractor'] = None,
        aggregation_gate: Optional[AggregationIntentGate] = None,
        vague_quantifier_resolver: Optional[VagueQuantifierResolver] = None,
        term_disambiguator: Optional[TermDisambiguator] = None,
        ngram_pre_gate: Optional['NgramPreGate'] = None,
        ensemble_resolver: Optional['EnsembleResolver'] = None,
        entity_type_voter: Optional['EntityTypeVoter'] = None,
    ):
        if circuit_breaker is None:
            raise QueryIntelligenceError("QIEngine requires a CircuitBreaker instance")
        self._config = config
        self._llm = llm_classifier
        self._grounder = entity_grounder
        self._max_query_length = max_query_length
        self._circuit_breaker = circuit_breaker
        if (multi_intent_config is None) != (multi_intent_splitter is None):
            raise QueryIntelligenceError("QIEngine: multi_intent_config and multi_intent_splitter must be passed together")
        self._multi_intent_config = multi_intent_config
        self._splitter = multi_intent_splitter
        self._calibrators = calibrator_registry
        self._intent_result_cache = intent_result_cache
        self._semantic_router = semantic_router
        self._semantic_intent_cache = semantic_intent_cache
        self._entity_extractor = entity_extractor
        self._regex_entity_extractor = regex_entity_extractor
        self._aggregation_gate = aggregation_gate
        self._vague_resolver = vague_quantifier_resolver
        self._term_disambiguator = term_disambiguator
        self._ngram_pre_gate = ngram_pre_gate
        self._ensemble_resolver = ensemble_resolver
        self._entity_type_voter = entity_type_voter
        self._numeric_auction_type_ids: 'frozenset[str]' = frozenset(t for t in config.regex.known_auction_types if t.isdigit())
        # Build a config-driven variant of _FILTER_SIGNAL_RE:
        # 1. The bare-word TLD context group (`\b(?:in|ending in|extension)\s+\.?[a-z]{2,N}\b`)
        #    uses N=tld_context_match_max_chars from config (default 5) instead of the
        #    hardcoded 8 - avoids FPs on generic nouns like "in auctions" (8 chars).
        # 2. Words in tld_word_bare_exclusions are stripped from the TLD word list pattern
        #    (`\b(?:com|...|in|...)\s+(?:only\s+)?domains?\b`) - removes FPs from ccTLDs
        #    that double as common English words (e.g. "in" = India ccTLD / preposition).
        _max_tld_ctx = config.regex.tld_context_match_max_chars
        _raw_pattern = _FILTER_SIGNAL_RE.pattern.replace(
            r'\.?[a-z]{2,8}\b',
            rf'\.?[a-z]{{2,{_max_tld_ctx}}}\b',
            1,  # only the context-word group; dotted-extension group (\.[a-z]{2,8}) untouched
        )
        for _excl_word in config.regex.tld_word_bare_exclusions:
            # Remove word from middle of alternation (most common case): com|net|...|in|br
            _raw_pattern = _raw_pattern.replace(f'|{_excl_word}|', '|')
            # Remove word at start of alternation (after `(?:`): (?:in|com|...
            _raw_pattern = _raw_pattern.replace(f'(?:{_excl_word}|', '(?:')
            # Remove word at end of alternation (before `)`): ...|in)
            _raw_pattern = _raw_pattern.replace(f'|{_excl_word})', ')')
        self._filter_signal_re = re.compile(_raw_pattern, re.IGNORECASE | re.VERBOSE)
        # ARCH-1: initialize here to avoid concurrent-creation race on the first request.
        self._l2_semaphore: Optional[asyncio.Semaphore] = (
            asyncio.Semaphore(int(config.llm.max_concurrent_l2))
            if llm_classifier is not None else None
        )

    def _calibrate(self, slices: List[IntentSlice], decision_tier: str) -> List[IntentSlice]:
        """Apply per-tier calibration to every slice's confidence.

        Identity passthrough when the registry is absent or the slice list is empty.
        """
        if self._calibrators is None or not slices:
            return slices
        calibrated: List[IntentSlice] = []
        for s in slices:
            signals = ConfidenceSignals(raw_confidence=float(s.confidence), entropy_normalized=1.0, score_margin=0.0)
            cal_conf = self._calibrators.calibrate_combined(decision_tier, signals)
            calibrated.append(replace(s, confidence=cal_conf))
        return calibrated

    async def _run_llm(self, query: str) -> Optional[tuple]:
        """Run T2 LLM classification. Returns None on any failure."""
        try:
            return await self._llm.classify(query, always_return=True)
        except (LLMError, CircuitOpenError) as e:
            logger.warning(f"qi_llm_failed query_len={len(query)} error_type={type(e).__name__} error={str(e)}")
            return None

    async def _run_llm_guarded(self, query: str) -> Optional[tuple]:
        async with self._l2_semaphore:
            return await self._run_llm(query)

    async def _l2_classify(self, query: str, request_id: str, timeout_seconds: float) -> Optional[tuple]:
        """Run L2 LLM with concurrency guard and timeout. Returns None on any failure."""
        async with self._l2_semaphore:
            try:
                if timeout_seconds > 0.0:
                    return await asyncio.wait_for(self._run_llm(query), timeout=timeout_seconds)
                return await self._run_llm(query)
            except asyncio.TimeoutError:
                logger.info(f"qi_ensemble_l2_timeout request_id={request_id} timeout_s={timeout_seconds:.1f}")
                return None
            except Exception as _e:  # noqa: BLE001 - resilience boundary: L2 failure degrades to None
                logger.warning(f"qi_ensemble_l2_error request_id={request_id} error_type={type(_e).__name__} error={_e}")
                return None

    def clear_cache(self) -> int:
        """Drop exact + semantic QI intent caches. Returns total entries dropped.

        Semantic intent cache must clear with exact - stale intents without
        soft_entities otherwise shadow fresh L0 extracts after code/deploy changes.
        """
        dropped = 0
        if self._intent_result_cache is not None:
            dropped += int(self._intent_result_cache.invalidate_all())
        if self._semantic_intent_cache is not None:
            dropped += int(self._semantic_intent_cache.invalidate_all())
        return dropped

    def cache_stats(self) -> Dict[str, Dict[str, int]]:
        """Per-tier hit/miss counters for QI intent caches."""
        intent_hits = 0
        intent_misses = 0
        if self._intent_result_cache is not None:
            intent_hits = int(self._intent_result_cache.hits)
            intent_misses = int(self._intent_result_cache.misses)
        sem_hits = 0
        sem_misses = 0
        if self._semantic_intent_cache is not None:
            sem_hits = int(self._semantic_intent_cache.hits)
            sem_misses = int(self._semantic_intent_cache.misses)
        return {
            'intent_result': {'hits': intent_hits, 'misses': intent_misses},
            'semantic_intent': {'hits': sem_hits, 'misses': sem_misses},
        }

    def quick_classify(self, raw_query: str) -> Optional[Tuple[str, float]]:
        """Run L1 SemanticRouter only. Returns (query_type, confidence) or None when router unavailable.

        Thread-safe: synchronous, suitable for executor dispatch from async callers.
        Does not consult caches or L2 LLM. Used for pre-screening before speculative task dispatch.
        :param raw_query: str - Query text; normalized internally
        :return: Optional[Tuple[str, float]] - (query_type, confidence) from L1, or None
        """
        if self._semantic_router is None:
            return None
        try:
            normalized = normalize_query(
                raw_query, self._max_query_length, normalize=self._config.normalize,
            )
            result = self._semantic_router.classify(normalized)
            if result is not None:
                return (result.query_type, float(result.confidence))
        except Exception as _e:  # noqa: BLE001 - resilience boundary: pre-screen failure degrades to None
            logger.debug(f"qi_quick_classify_error error_type={type(_e).__name__} error={str(_e)}")
        return None

    def _resolve_l0_fallback_type(
        self,
        query: str,
        l0_entities: Optional[List[Entity]] = None,
        ngram_intent: Optional[str] = None,
    ) -> 'Tuple[str, float]':
        """Determine the query type and confidence to use when the L0_fallback path fires.

        Priority order:
          1. AggregationIntentGate (deterministic structural check, fastest).
          2. Hard structured-filter entities from L0 -> force hybrid.
          3. Value-discovery vocabulary -> force hybrid.
          4. NgramPreGate result (when ensemble is disabled, ngram votes are otherwise lost).
          5. SemanticRouter.best_guess (sub-threshold L1 signal, better than default).
          6. default_query_type from config (last resort).

        :param query: str - Normalized sub-query text.
        :param l0_entities: Optional[List[Entity]] - L0-extracted entities.
        :param ngram_intent: Optional[str] - NgramPreGate archetype when ensemble is disabled.
        :return: Tuple[str, float] - (query_type, confidence).
        """
        if self._aggregation_gate is not None and self._aggregation_gate.is_analytics(query):
            logger.debug(f"qi_l0_fallback_analytics_gate query_len={len(query)}")
            return 'analytics', 1.0
        if l0_entities:
            _hard_filter_slots = getattr(self._config.routing, 'l0_fallback_force_hybrid_slots', None)
            if _hard_filter_slots is None:
                _hard_filter_slots = frozenset(_ENTITY_NAME_TO_FILTER_CAT)
            elif not isinstance(_hard_filter_slots, frozenset):
                _hard_filter_slots = frozenset(_hard_filter_slots)
            _advisory = _GUIDANCE_VETO_RE.search(query)
            _browse = _EXPLORE_BROWSE_VETO_RE.search(query)
            for _e in l0_entities:
                if _e.chip_kind == 'hard' and _e.name in _hard_filter_slots:
                    if _advisory or _browse:
                        logger.debug(
                            f"qi_l0_fallback_entity_veto entity={_e.name} "
                            f"advisory={bool(_advisory)} browse={bool(_browse)} query_len={len(query)}"
                        )
                        continue
                    logger.debug(f"qi_l0_fallback_force_hybrid entity={_e.name} query_len={len(query)}")
                    return 'hybrid', 1.0
        _q_lower = query.lower()
        if any(sig in _q_lower for sig in _VALUE_DISCOVERY_SIGNALS):
            logger.debug(f"qi_l0_fallback_vd_hybrid query_len={len(query)}")
            return 'hybrid', 1.0
        # Ngram gate: only use when it fires for a non-default archetype (explore/guidance/analytics).
        # 'hybrid' from ngram means "no strong browse/advisory signal" - fall through to L1 instead.
        if ngram_intent is not None and ngram_intent != self._config.default_query_type:
            _ngram_cfg = getattr(self._config.regex, 'ngram_pre_gate', None)
            _ngram_conf = float(_ngram_cfg.emit_confidence) if _ngram_cfg is not None else 0.75
            logger.debug(f"qi_l0_fallback_ngram type={ngram_intent} confidence={_ngram_conf:.3f} query_len={len(query)}")
            return ngram_intent, _ngram_conf
        if self._semantic_router is not None:
            _best = self._semantic_router.best_guess(query)
            if _best is not None and _best[1] > self._config.routing.fallback_confidence:
                logger.debug(f"qi_l0_fallback_l1_best_guess type={_best[0]} score={_best[1]:.3f} query_len={len(query)}")
                return _best[0], float(_best[1])
        return self._config.default_query_type, self._config.routing.fallback_confidence

    def _ground_slices(self, slices: List[IntentSlice]) -> List[IntentSlice]:
        """Ground hard entities only. Soft stays on soft_entities (rank / response).

        Soft never reattached onto entities - FIND-hard alone drives Qdrant
        payload filters. SoftKeywordApplier boosts from soft_entities.

        Snapshots pre-inventory-ground hard entities onto
        ``IntentSlice.pre_ground_entities`` so response
        ``query_intelligence.filters.identified`` can report as-identified while
        ``entities`` (grounded) feed retrieval + ``pipeline_trace.applied_filters``.
        """
        out: List[IntentSlice] = []
        for s in slices:
            pre_ground = list(s.entities)
            grounded_hard, _drop_count = ground_hard_entities(s.entities, self._grounder)
            soft = list(getattr(s, 'soft_entities', None) or [])
            out.append(IntentSlice(
                query_type=s.query_type,
                entities=list(grounded_hard),
                confidence=s.confidence,
                raw_text=s.raw_text,
                slice_id=getattr(s, 'slice_id', '') or '',
                soft_entities=soft,
                pre_ground_entities=pre_ground,
                keywords=list(getattr(s, 'keywords', None) or []),
            ))
        return out

    def _make_intent(
        self,
        request_id: str,
        intent_record_id: str,
        raw_query: str,
        normalized: str,
        slices: List[IntentSlice],
        decision_tier: str,
        decision_cost_usd: float,
        alternative_interpretations: Optional[List[IntentSlice]] = None,
        sub_intent_snapshot: Optional[List[Tuple[str, List[Entity]]]] = None,
    ) -> QueryIntent:
        """Assemble the final QueryIntent. The primary slice is the highest-confidence one."""
        if not slices:
            slices = [IntentSlice(query_type=self._config.default_query_type, entities=[], confidence=self._config.routing.fallback_confidence, raw_text=normalized)]
            decision_tier = 'fallback'
        sorted_slices = sorted(slices, key=lambda s: s.confidence, reverse=True)
        primary = sorted_slices[0]
        mode = QueryIntent.derive_routing_mode(primary.confidence, self._config.routing.routing_auto_execute_min, self._config.routing.routing_suggest_min)
        sem_q: Optional[str] = None
        res_kind: Optional[str] = None
        # Hard entities only after _ground_slices (soft lives on soft_entities).
        # Vague occupied-set includes soft so has_web_traffic_signal blocks traffic_min.
        all_entities = [e for s in sorted_slices for e in s.entities] if sorted_slices else []
        if self._vague_resolver is not None and sorted_slices:
            _occupied_for_vague = all_entities + [
                e for s in sorted_slices for e in list(getattr(s, 'soft_entities', None) or [])
            ]
            _new_ents = self._vague_resolver.resolve(_occupied_for_vague, normalized)
            if _new_ents:
                _primary = sorted_slices[0]
                # Vague invents are soft rank signals (chip_kind=soft) — never merge
                # into hard ``entities`` or they become FIND hard filters and inflate
                # qie_only filter sets vs LLMJ/Full applied_filters.
                _augmented = IntentSlice(
                    query_type=_primary.query_type,
                    entities=list(_primary.entities),
                    confidence=_primary.confidence,
                    raw_text=_primary.raw_text,
                    slice_id=getattr(_primary, 'slice_id', None) or '',
                    soft_entities=list(getattr(_primary, 'soft_entities', None) or [])
                    + list(_new_ents),
                    pre_ground_entities=(
                        list(_primary.pre_ground_entities)
                        if _primary.pre_ground_entities is not None
                        else None
                    ),
                    keywords=list(getattr(_primary, 'keywords', None) or []),
                )
                sorted_slices = [_augmented] + list(sorted_slices[1:])
                all_entities = [e for s in sorted_slices for e in s.entities]
        if self._config.residual is not None and self._config.residual.enabled and sorted_slices:
            sem_q, res_kind = extract_residual(normalized, all_entities, self._config.residual)
        _nav = frozenset(self._config.residual.navigational_tokens) if self._config.residual is not None else frozenset()
        encode_text = build_semantic_encode_text(normalized, all_entities, sem_q, _nav)
        _sub_filters = None
        if (
            sub_intent_snapshot
            and self._multi_intent_config is not None
            and self._multi_intent_config.preserve_sub_intent_filters
        ):
            _sub_filters = [
                SubIntentFilterSet(sub_query=sq, entities=list(entities))
                for sq, entities in sub_intent_snapshot
                if entities
            ] or None
        _kw_by_term: Dict[str, float] = {}
        for s in sorted_slices:
            for kw in getattr(s, 'keywords', None) or []:
                term = str(kw.get('term') or '').strip()
                if not term:
                    continue
                prob = float(kw.get('probability') or 0.0)
                if term not in _kw_by_term or prob > _kw_by_term[term]:
                    _kw_by_term[term] = prob
        keywords = sorted(
            [{'term': t, 'probability': p} for t, p in _kw_by_term.items()],
            key=lambda k: k['probability'],
            reverse=True,
        )
        return QueryIntent(
            request_id=request_id,
            raw_query=raw_query,
            normalized_query=normalized,
            query_type=primary.query_type,
            confidence=primary.confidence,
            decision_tier=decision_tier,
            slices=sorted_slices,
            decision_cost_usd=decision_cost_usd,
            intent_record_id=intent_record_id,
            semantic_query=sem_q,
            semantic_encode_text=encode_text,
            alternative_interpretations=list(alternative_interpretations) if alternative_interpretations else [],
            routing_mode=mode,
            residual_kind=res_kind,
            sub_intent_filters=_sub_filters,
            prompt_tag=self._config.llm.prompt_tag,
            schema_version=self._config.llm.schema_version,
            keywords=keywords,
        )

    def _intent_cache_key(self, normalized: str) -> str:
        """Build intent-result / semantic-intent cache key with classify versions."""
        return versioned_query_key(
            normalized,
            prompt_tag=self._config.llm.prompt_tag,
            schema_version=self._config.llm.schema_version,
        )

    async def classify(
        self,
        raw_query: str,
        request_id: str,
        intent_record_id: Optional[str] = None,
        pre_normalized: Optional[str] = None,
        pre_l0_slice: Optional[IntentSlice] = None,
        pre_l0_cost_usd: float = 0.0,
        pre_l0_llm_completed: bool = False,
    ) -> QueryIntent:
        """Classify a raw query through the LLM.
        :param raw_query: str - Original query text from the API boundary
        :param request_id: str - Correlation id propagated through retrieval / cache / feedback
        :param intent_record_id: Optional[str] - Carry-forward id for refine turns
        :param pre_normalized: Optional[str] - Pre-normalized query (skip re-normalization)
        :param pre_l0_slice: Optional[IntentSlice] - Precomputed L0 from combined rewrite+extract
        :param pre_l0_cost_usd: float - Cost already spent on precomputed L0
        :param pre_l0_llm_completed: bool - Whether precomputed L0 LLM completed
        :return: QueryIntent - Final classification with grounded entities
        :raises ValidationError: When the query is empty / non-string
        :raises QueryIntelligenceError: When the engine is disabled in config
        """
        if not self._config.enabled:
            raise QueryIntelligenceError("qi engine disabled in config")
        if pre_normalized is not None and isinstance(pre_normalized, str) and pre_normalized:
            normalized = pre_normalized
        else:
            normalized = normalize_query(
                raw_query, self._max_query_length, normalize=self._config.normalize,
            )
        _cache_key = self._intent_cache_key(normalized)
        if self._intent_result_cache is not None:
            cached = self._intent_result_cache.get(_cache_key)
            if cached is not None:
                return cached
        # Tier-0.5 semantic intent cache - fuzzy match for rephrased queries.
        # Sits above the L1/L2 cascade: a hit returns a stored QueryIntent for a
        # semantically similar past query without re-classifying (documented in
        # the module docstring). Checked after the exact cache so identical queries
        # still hit the cheaper tier first.
        if self._semantic_intent_cache is not None:
            sem_cached = self._semantic_intent_cache.get(
                normalized,
                prompt_tag=self._config.llm.prompt_tag,
                schema_version=self._config.llm.schema_version,
            )
            if sem_cached is not None:
                logger.debug(f"qi_semantic_cache_hit query_len={len(normalized)} query_type={sem_cached.query_type}")
                return sem_cached
        t0 = time.monotonic()
        rid = intent_record_id if (intent_record_id and isinstance(intent_record_id, str)) else QueryIntent.new_intent_record_id()

        if self._splitter is not None:
            sub_queries = self._splitter.split(normalized)
        else:
            sub_queries = [normalized]

        # Keyword-leg expansion on transformed/effective text: when delimiter
        # split yields one fragment but L0 already extracted multiple topical
        # keywords, expand into one SERP leg per keyword + shared constraints.
        _keyword_legs = False
        if (
            len(sub_queries) == 1
            and self._splitter is not None
            and self._multi_intent_config is not None
            and bool(self._multi_intent_config.split_on_l0_keywords)
            and pre_l0_slice is not None
        ):
            _pre_kws = list(getattr(pre_l0_slice, 'keywords', None) or [])
            if _pre_kws:
                _expanded = self._splitter.expand_with_keywords(normalized, _pre_kws)
                if len(_expanded) > 1:
                    sub_queries = _expanded
                    _keyword_legs = True

        # Signal-based split screen (general, no lexical rules): when the
        # deterministic splitter emits multiple candidates, score each with the
        # L1 semantic router (sync, no LLM) and drop fragments below the noise
        # floor. Conversational preamble ("i am from delhi") matches no intent
        # centroid strongly and is dropped here, collapsing a spurious split back
        # to a single-intent classify - one LLM call instead of N. A genuine
        # multi-intent ("expiring .com and short .io under $500") keeps >=2 strong
        # fragments and still fans out. Never empties: the strongest survives.
        _prune_requested = (
            len(sub_queries) > 1
            and self._multi_intent_config is not None
            and self._multi_intent_config.prune_noise_slices
        )
        if _prune_requested and self._semantic_router is not None:
            _floor = float(self._multi_intent_config.prune_noise_max_confidence)
            _scored = [(sq, self.quick_classify(sq)) for sq in sub_queries]
            _kept = [sq for sq, sc in _scored if sc is None or sc[1] >= _floor]
            if not _kept:
                _best = max(_scored, key=lambda x: (x[1][1] if x[1] is not None else -1.0))
                _kept = [_best[0]]
            if len(_kept) < len(sub_queries):
                logger.info(
                    f"qi_multi_intent_prescreen request_id={request_id} "
                    f"candidates={len(sub_queries)} kept={len(_kept)} floor={_floor:.2f}"
                )
                sub_queries = _kept
        elif _prune_requested and self._semantic_router is None:
            # Fail-safe: a regex-only multi-split cannot be semantically vetted
            # without the router, so do not trust it. Collapse to single-intent
            # rather than pay for an N-way LLM fan-out on a possibly-spurious
            # split (e.g. a vague query the regex over-split). The strongest
            # signal - the full query - is preserved as one intent.
            logger.warning(
                f"qi_multi_intent_prescreen_skipped_no_router request_id={request_id} "
                f"candidates={len(sub_queries)} action=collapse_single_intent"
            )
            sub_queries = [normalized]

        _entity_suppress_types = frozenset(self._config.regex.suppress_for_query_types)
        if len(sub_queries) == 1:
            slices, decision_tier, cost, alternatives = await self._classify_single(
                sub_queries[0], request_id, t_start=t0,
                pre_l0_slice=pre_l0_slice,
                pre_l0_cost_usd=pre_l0_cost_usd,
                pre_l0_llm_completed=pre_l0_llm_completed,
            )
            if _entity_suppress_types:
                slices = _keep_hard_entities_on_suppress(slices, _entity_suppress_types)
            soft_names, hard_names = self._slot_sets()
            slices = _apply_cue_reconcile_to_slices(slices, normalized, soft_names, hard_names)
            _paired = self._config.regex.paired_direction_slots
            for _cmin, _cmax, _cmin_v, _cmax_v in _detect_slot_contradictions(slices, _paired):
                logger.warning(
                    f"qi_filter_contradiction request_id={request_id} "
                    f"min_slot={_cmin} min_val={_cmin_v} max_slot={_cmax} max_val={_cmax_v} "
                    f"query_len={len(sub_queries[0])}"
                )
            if self._multi_intent_config is not None and self._multi_intent_config.post_merge_deconflict_rules:
                slices = _deconflict_merged_entities(slices, self._multi_intent_config.post_merge_deconflict_rules)
            if self._multi_intent_config is not None and self._multi_intent_config.keyword_contains_value_deconflict:
                slices = _deconflict_keyword_contains_values(slices, self._multi_intent_config.keyword_contains_value_deconflict)
            slices = _reconcile_numeric_misattribution_slices(slices, normalized)
            # Full-query cue reconcile after numeric deconflict (stale polarity may rewrite days_listed_*).
            slices = _apply_cue_reconcile_to_slices(slices, normalized, soft_names, hard_names)
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            self._log_decision(request_id, decision_tier, slices, elapsed_ms, cost, sub_intents=1)
            intent = self._make_intent(request_id, rid, raw_query, normalized, slices, decision_tier, cost, alternative_interpretations=alternatives)
            # ROB-2: don't cache timeout-derived fallback - the decision_tier 'fallback'
            # means _classify_single timed out; caching it would serve stale fallback for 30 min.
            if self._intent_result_cache is not None and intent.decision_tier != 'fallback':
                self._intent_result_cache.put(_cache_key, intent)
            if self._semantic_intent_cache is not None and intent.decision_tier != 'fallback':
                self._semantic_intent_cache.put(
                    normalized,
                    intent,
                    prompt_tag=self._config.llm.prompt_tag,
                    schema_version=self._config.llm.schema_version,
                )
            return intent

        logger.info(
            f"qi_multi_intent_fanout request_id={request_id} "
            f"sub_intents={len(sub_queries)} keyword_legs={_keyword_legs}"
        )
        # Keyword legs reuse pre_l0 hard/soft filters (one LLM already spent);
        # each leg keeps a single topical keyword for encode/rank boost.
        async def _classify_leg(sq: str):
            if (
                _keyword_legs
                and pre_l0_llm_completed
                and pre_l0_slice is not None
            ):
                _sq_cf = sq.casefold()
                _leg_kws = [
                    kw for kw in (getattr(pre_l0_slice, 'keywords', None) or [])
                    if str(kw.get('term') or '').strip()
                    and str(kw.get('term') or '').strip().casefold() in _sq_cf
                ]
                _leg_slice = replace(
                    pre_l0_slice,
                    raw_text=sq,
                    keywords=_leg_kws or list(getattr(pre_l0_slice, 'keywords', None) or []),
                )
                return await self._classify_single(
                    sq, request_id, t_start=t0,
                    pre_l0_slice=_leg_slice,
                    pre_l0_cost_usd=0.0,
                    pre_l0_llm_completed=True,
                )
            return await self._classify_single(sq, request_id, t_start=t0)

        results = await asyncio.gather(
            *[_classify_leg(sq) for sq in sub_queries],
            return_exceptions=True,
        )
        policy = self._multi_intent_config.sub_intent_failure_policy if self._multi_intent_config is not None else 'fail_soft'
        merged_slices: List[IntentSlice] = []
        merged_cost: float = 0.0
        tiers_used: List[str] = []
        _per_sub_intent_snapshot: List[Tuple[str, List[Entity]]] = []
        for sq, result in zip(sub_queries, results):
            if isinstance(result, BaseException):
                if policy == 'fail_closed':
                    raise QueryIntelligenceError(f"qi_subquery_failed sub_query_prefix={sq[:80]}") from result
                logger.warning(f"qi_subquery_failed request_id={request_id} sub_query={sq[:80]} error_type={type(result).__name__} error={str(result)}")
                continue
            sub_slices, sub_tier, sub_cost, _sub_alts = result
            if _entity_suppress_types:
                sub_slices = _keep_hard_entities_on_suppress(sub_slices, _entity_suppress_types)
            _per_sub_intent_snapshot.append((sq, [e for s in sub_slices for e in s.entities]))
            merged_cost += sub_cost
            tiers_used.append(sub_tier)
            for s in sub_slices:
                merged_slices.append(replace(
                    s,
                    entities=list(s.entities),
                    soft_entities=list(getattr(s, 'soft_entities', None) or []),
                    slice_id=IntentSlice.new_slice_id(),
                ))
        merged_slices = _dedup_merged_slice_entities(merged_slices)
        # Drop spurious low-confidence, entity-less slices from false splits
        # (e.g. conversational preamble "i am from delhi"). Only fires with >1
        # slice and never empties the list.
        if (
            self._multi_intent_config is not None
            and self._multi_intent_config.prune_noise_slices
            and len(merged_slices) > 1
        ):
            _prune_max = float(self._multi_intent_config.prune_noise_max_confidence)
            _kept = [
                s for s in merged_slices
                if s.entities or s.keywords or float(s.confidence) >= _prune_max
            ]
            if not _kept:
                _kept = [max(merged_slices, key=lambda s: float(s.confidence))]
            if len(_kept) < len(merged_slices):
                logger.info(
                    f"qi_multi_intent_noise_prune request_id={request_id} "
                    f"kept={len(_kept)} dropped={len(merged_slices) - len(_kept)}"
                )
                merged_slices = _kept
        if self._multi_intent_config is not None and self._multi_intent_config.post_merge_deconflict_rules:
            merged_slices = _deconflict_merged_entities(merged_slices, self._multi_intent_config.post_merge_deconflict_rules)
        if self._multi_intent_config is not None and self._multi_intent_config.keyword_contains_value_deconflict:
            merged_slices = _deconflict_keyword_contains_values(merged_slices, self._multi_intent_config.keyword_contains_value_deconflict)
        _list_merge_names = frozenset(self._multi_intent_config.list_merge_slots) if self._multi_intent_config else frozenset()
        if _list_merge_names:
            merged_slices = _merge_list_valued_entities(merged_slices, _list_merge_names)
        merged_slices = _merge_multi_keyword_contains(merged_slices)
        merged_slices = _merge_multi_keyword_contains_exclude(merged_slices)
        merged_slices = _drop_keyword_overlapping_topics(merged_slices)
        _singleton_names = frozenset(self._multi_intent_config.singleton_merge_slots) if self._multi_intent_config else frozenset()
        _mi_strategy = self._multi_intent_config.numeric_singleton_merge_strategy if self._multi_intent_config else 'first_seen'
        if _mi_strategy == 'max_permissive':
            merged_slices = _resolve_singletons_max_permissive(merged_slices, _singleton_names)
        else:
            merged_slices = _resolve_singleton_conflicts(merged_slices, _singleton_names)
        _bcast = frozenset(self._multi_intent_config.cross_slice_broadcast_slots) if self._multi_intent_config else frozenset()
        _bcast_peers = frozenset(self._multi_intent_config.cross_slice_broadcast_peer_slots) if self._multi_intent_config else frozenset()
        if _bcast:
            merged_slices = _broadcast_slots_across_peer_slices(merged_slices, _bcast, _bcast_peers)
        _paired_multi = self._config.regex.paired_direction_slots
        for _cmin, _cmax, _cmin_v, _cmax_v in _detect_slot_contradictions(merged_slices, _paired_multi):
            logger.warning(
                f"qi_filter_contradiction request_id={request_id} "
                f"min_slot={_cmin} min_val={_cmin_v} max_slot={_cmax} max_val={_cmax_v} "
                f"query_len={len(normalized)}"
            )
        merged_slices = _reconcile_numeric_misattribution_slices(merged_slices, normalized)
        # Full original query cues (OR lifecycle / exact / stale / majestic) after fan-out merge.
        soft_names, hard_names = self._slot_sets()
        merged_slices = _apply_cue_reconcile_to_slices(merged_slices, normalized, soft_names, hard_names)
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        primary_tier = tiers_used[0] if tiers_used else 'fallback'
        decision_tier = 'L0_multi_intent' if len(merged_slices) > 1 else primary_tier
        self._log_decision(request_id, decision_tier, merged_slices, elapsed_ms, merged_cost, sub_intents=len(sub_queries))
        intent = self._make_intent(request_id, rid, raw_query, normalized, merged_slices, decision_tier, merged_cost, alternative_interpretations=[], sub_intent_snapshot=_per_sub_intent_snapshot)
        if self._intent_result_cache is not None and intent.decision_tier != 'fallback':
            self._intent_result_cache.put(_cache_key, intent)
        return intent

    def _slot_sets(self) -> Tuple[frozenset, frozenset]:
        """Return (soft_slot_names, hard_entity_names) from qi.entity_slots (required)."""
        slots = self._config.entity_slots
        if slots is None:
            raise ConfigurationError("qi.entity_slots is required for L0 entity extract")
        return slots.soft_slot_set, slots.hard_entity_set

    def _should_run_regex_l0_fallback(
        self, *, llm_completed: bool, llm_hard: List[Entity], llm_soft: List[Entity],
    ) -> bool:
        """True when sequential regex L0 fallback should run after LLM attempt.

        Regex (hard + soft keyword/topic patterns) activates only on LLM absence
        when ``qi.l0_regex_entity.fallback_only_when_llm_unavailable`` is True:
        - True -> regex only when LLM did not complete (missing / exception).
        - False -> regex also when LLM completed with zero entities (legacy).
        Not gated on whether LLM soft/hard was nonempty - when LLM completed,
        regex stays off entirely.
        """
        rex = self._regex_entity_extractor
        if rex is None:
            return False
        cfg = rex._config
        if not cfg.enabled:
            return False
        if cfg.fallback_only_when_llm_unavailable:
            return not llm_completed
        return not llm_hard and not llm_soft

    async def _call_l0_llm_priced(
        self, l0_text: str,
    ) -> Tuple[Optional[IntentSlice], float]:
        """Invoke L0 LLM extract; return (slice_or_None, cost_usd).

        Prefers ``classify_async_priced`` (looked up on the **class**, not the
        instance - MagicMock auto-attrs would otherwise shadow ``classify_async``
        in tests). Falls back to ``classify_async`` with ``cost_usd=0.0`` for
        duck-typed extractors.
        """
        extractor = self._entity_extractor
        if extractor is None:
            return None, 0.0
        # Class lookup avoids unittest.mock.MagicMock inventing the attr.
        priced_on_cls = getattr(type(extractor), 'classify_async_priced', None)
        if callable(priced_on_cls):
            return await extractor.classify_async_priced(l0_text)
        return await extractor.classify_async(l0_text), 0.0

    async def _run_l0_extractors(
        self, l0_text: str, request_id: str,
    ) -> Tuple[List[Entity], List[Entity], float, List[Dict[str, Any]]]:
        """LLM entity extract first; regex sequential fallback per l0_regex_entity config.

        Regex keeps hard + soft patterns; they run only when LLM is unavailable
        (``fallback_only_when_llm_unavailable``). Not a merge against LLM soft.

        Keywords (term/probability) come from the L0 LLM slice — same source as
        ``_classify_ensemble`` — already thresholded by the extractor.

        :return: (hard_entities, soft_entities, l0_llm_cost_usd, keywords)
        """
        soft_names, hard_names = self._slot_sets()
        l0_result = None
        llm_completed = False
        l0_cost_usd = 0.0
        if self._entity_extractor is not None:
            try:
                l0_result, l0_cost_usd = await self._call_l0_llm_priced(l0_text)
                llm_completed = True
            except Exception as _e:  # noqa: BLE001 - resilience boundary: L0 failure degrades to None
                logger.warning(
                    f"qi_ensemble_l0_error request_id={request_id} "
                    f"error_type={type(_e).__name__} error={_e}"
                )
                l0_result = None
                llm_completed = False
                l0_cost_usd = 0.0

        llm_hard = list(l0_result.entities) if l0_result is not None else []
        llm_soft = list(getattr(l0_result, 'soft_entities', None) or []) if l0_result is not None else []
        l0_keywords = _l0_keywords_from_slice(l0_result)
        regex_result = None
        if self._should_run_regex_l0_fallback(
            llm_completed=llm_completed, llm_hard=llm_hard, llm_soft=llm_soft,
        ):
            try:
                regex_result = await self._regex_entity_extractor.classify_async(l0_text)
            except asyncio.CancelledError:
                raise
            except Exception as _e:  # noqa: BLE001 - resilience boundary: regex failure degrades to None
                logger.warning(
                    f"qi_ensemble_regex_error request_id={request_id} "
                    f"error_type={type(_e).__name__} error={_e}"
                )
                regex_result = None

        hard, soft = _merge_extractor_entities(
            regex_result, l0_result, l0_text, soft_names, hard_names,
            llm_completed=llm_completed,
        )
        combined = _deconflict_similar_to_vs_tld(
            _drop_numeric_misattributions(hard + soft, l0_text)
        )
        return (
            [e for e in combined if e.name not in soft_names],
            [e for e in combined if e.name in soft_names],
            float(l0_cost_usd),
            l0_keywords,
        )

    async def _extract_l0_entities(
        self, l0_text: str, request_id: str,
    ) -> Tuple[List[Entity], List[Entity], float, List[Dict[str, Any]]]:
        """Run sequential L0 LLM then regex-fallback; merge + deconflict.

        No L1 semantic router and no L2 intent LLM. Used by ``extract_l0_filters``
        (qie_only_mode). Full-search ``_classify_ensemble`` keeps a concurrent
        L0 task but stamps keywords via the same ``_l0_keywords_from_slice``.

        :return: (hard_entities, soft_entities, l0_llm_cost_usd, keywords)
        """
        return await self._run_l0_extractors(l0_text, request_id)

    async def extract_l0_filters(
        self,
        raw_query: str,
        request_id: str,
        intent_record_id: Optional[str] = None,
        pre_normalized: Optional[str] = None,
        pre_l0_slice: Optional[IntentSlice] = None,
        pre_l0_cost_usd: float = 0.0,
        pre_l0_llm_completed: bool = False,
    ) -> QueryIntent:
        """Extract + ground filters via L0 only. No intent classification (no L1/L2).

        Used by ``qie_only_mode`` for a latency-cheap filters-only verdict. Does not
        consult or write intent result caches (those store classified query types).
        Keywords mirror full-search: pre_l0 slice when combined rewrite already ran,
        else from the fresh L0 LLM extract (threshold already applied in extractor).
        """
        if not self._config.enabled:
            raise QueryIntelligenceError("qi engine disabled in config")
        if pre_normalized is not None and isinstance(pre_normalized, str) and pre_normalized:
            normalized = pre_normalized
        else:
            normalized = normalize_query(
                raw_query, self._max_query_length, normalize=self._config.normalize,
            )
        t0 = time.monotonic()
        rid = intent_record_id if (intent_record_id and isinstance(intent_record_id, str)) else QueryIntent.new_intent_record_id()
        l0_text = (
            self._term_disambiguator.disambiguate(normalized)
            if self._term_disambiguator is not None
            else normalized
        )
        soft_names, hard_names = self._slot_sets()
        extract_timeout = float(self._config.llm.classify_timeout_seconds)
        l0_cost_usd = 0.0
        l0_keywords: List[Dict[str, Any]] = []
        if pre_l0_llm_completed and pre_l0_slice is not None:
            hard_ents = list(pre_l0_slice.entities or [])
            soft_ents = list(getattr(pre_l0_slice, 'soft_entities', None) or [])
            l0_cost_usd = float(pre_l0_cost_usd)
            l0_keywords = _l0_keywords_from_slice(pre_l0_slice)
            # Still run regex fallback when LLM completed empty and config allows.
            if self._should_run_regex_l0_fallback(
                llm_completed=True, llm_hard=hard_ents, llm_soft=soft_ents,
            ):
                try:
                    regex_result = await self._regex_entity_extractor.classify_async(l0_text)
                except Exception as _e:  # noqa: BLE001
                    logger.warning(
                        f"qi_l0_extract_regex_error request_id={request_id} "
                        f"error_type={type(_e).__name__} error={_e}"
                    )
                    regex_result = None
                hard_ents, soft_ents = _merge_extractor_entities(
                    regex_result, pre_l0_slice, l0_text, soft_names, hard_names,
                    llm_completed=True,
                )
                combined = _deconflict_similar_to_vs_tld(
                    _drop_numeric_misattributions(hard_ents + soft_ents, l0_text)
                )
                hard_ents = [e for e in combined if e.name not in soft_names]
                soft_ents = [e for e in combined if e.name in soft_names]
        else:
            try:
                if extract_timeout > 0.0:
                    hard_ents, soft_ents, l0_cost_usd, l0_keywords = await asyncio.wait_for(
                        self._extract_l0_entities(l0_text, request_id),
                        timeout=extract_timeout,
                    )
                else:
                    hard_ents, soft_ents, l0_cost_usd, l0_keywords = await self._extract_l0_entities(
                        l0_text, request_id,
                    )
            except asyncio.TimeoutError:
                logger.warning(
                    f"qi_l0_extract_timeout request_id={request_id} timeout_sec={extract_timeout}"
                )
                hard_ents, soft_ents, l0_cost_usd, l0_keywords = [], [], 0.0, []

        conf = 1.0
        _all_conf = hard_ents + soft_ents
        if _all_conf:
            conf = max(float(e.confidence) for e in _all_conf)
        slices: List[IntentSlice] = [
            IntentSlice(
                query_type=self._config.default_query_type,
                entities=list(hard_ents),
                confidence=conf,
                raw_text=normalized,
                soft_entities=list(soft_ents),
                keywords=list(l0_keywords),
            )
        ]
        # Cue patterns must see the user-facing query text. When preprocess rewrote
        # the extract target, ``normalized``/``l0_text`` may have dropped OR-topic
        # cues; fall back to raw_query for reconcile so "b2b saas or maybe fintech"
        # still merges.
        _cue_text = (
            raw_query.strip()
            if isinstance(raw_query, str) and raw_query.strip()
            else normalized
        )
        slices = _apply_cue_reconcile_to_slices(slices, _cue_text, soft_names, hard_names)
        if self._multi_intent_config is not None and self._multi_intent_config.post_merge_deconflict_rules:
            slices = _deconflict_merged_entities(slices, self._multi_intent_config.post_merge_deconflict_rules)
        if self._multi_intent_config is not None and self._multi_intent_config.keyword_contains_value_deconflict:
            slices = _deconflict_keyword_contains_values(slices, self._multi_intent_config.keyword_contains_value_deconflict)
        slices = _reconcile_numeric_misattribution_slices(slices, _cue_text)
        slices = _apply_cue_reconcile_to_slices(slices, _cue_text, soft_names, hard_names)
        slices = self._ground_slices(slices)
        # After hard grounding: soft stays on soft_entities (rank boost / response).
        decision_tier = 'L0_entity'
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        self._log_decision(request_id, decision_tier, slices, elapsed_ms, l0_cost_usd, sub_intents=1)
        return self._make_intent(request_id, rid, raw_query, normalized, slices, decision_tier, float(l0_cost_usd))

    def _ensemble_consensus_cancel_l2(self) -> 'Tuple[bool, float]':
        """Return (enabled, threshold) from EnsembleResolver config; False when unwired/mocked."""
        resolver = self._ensemble_resolver
        if resolver is None:
            return False, 1.0
        ens_cfg = getattr(resolver, '_config', None)
        if not isinstance(ens_cfg, QIEnsembleConfig):
            return False, 1.0
        return bool(ens_cfg.consensus_cancel_l2), float(ens_cfg.consensus_cancel_threshold)

    def _ensemble_extract_before_classify(self) -> bool:
        """Return extract_before_classify from ensemble config; False when unwired/mocked."""
        resolver = self._ensemble_resolver
        if resolver is None:
            return False
        ens_cfg = getattr(resolver, '_config', None)
        if not isinstance(ens_cfg, QIEnsembleConfig):
            return False
        return bool(ens_cfg.extract_before_classify)

    @staticmethod
    def _l0_ngram_agree_skip_l2(
        ngram_intent: Optional[str],
        entity_vote: Optional[Vote],
        cancel_threshold: float,
    ) -> bool:
        """True when ngram + entity voter share an archetype (both non-abstain).

        With two agreeing fast voters, agreement_ratio is 1.0; require it >= threshold.
        """
        if ngram_intent is None or entity_vote is None or entity_vote.abstained:
            return False
        if entity_vote.archetype != ngram_intent:
            return False
        return 1.0 >= float(cancel_threshold)

    async def _classify_ensemble(
        self,
        l0_text: str,
        normalized_sub_query: str,
        request_id: str,
        t_start: float = 0.0,
        pre_l0_slice: Optional[IntentSlice] = None,
        pre_l0_cost_usd: float = 0.0,
        pre_l0_llm_completed: bool = False,
    ) -> 'Tuple[List[IntentSlice], str, float, List[IntentSlice]]':
        """Classify one sub-query via sequential L1->L2 cascade + ensemble voting.

        When ``extract_before_classify`` is on, L0 completes before L1/L2. Otherwise
        L0 may run concurrently with L1 (legacy). L2 never runs in parallel with L1.
        When ``consensus_cancel_l2`` is on, L0 is awaited before the L2 decision so
        ngram+entity agreement can skip L2. Precomputed L0 from combined
        rewrite+extract skips a second L0 LLM call on the single-intent path.

        Returns (grounded_slices, decision_tier, cost_usd, alternative_interpretations).
        """
        # Step 1 - L1 text mirrors L0 text (short-query expansion removed).
        l1_text = l0_text

        circuit_allows = self._circuit_breaker.allow_request()
        _extract_before = self._ensemble_extract_before_classify()
        _use_pre_l0 = bool(pre_l0_llm_completed and pre_l0_slice is not None)

        l0_task = (
            None
            if _use_pre_l0
            else (
                asyncio.create_task(self._call_l0_llm_priced(l0_text))
                if self._entity_extractor is not None
                else None
            )
        )

        # Sync gates - sub-millisecond; computed while L0 task (if any) is pending.
        _agg_intent: Optional[str] = None
        if self._aggregation_gate is not None and self._aggregation_gate.is_analytics(l0_text):
            _agg_intent = 'analytics'
        _ngram_intent: Optional[str] = None
        if self._ngram_pre_gate is not None:
            _ngram_intent = self._ngram_pre_gate.classify(l0_text)

        soft_names, hard_names = self._slot_sets()
        l0_result = pre_l0_slice if _use_pre_l0 else None
        llm_completed = bool(pre_l0_llm_completed) if _use_pre_l0 else False
        regex_result = None
        hard_ents: List[Entity] = []
        soft_ents: List[Entity] = []
        l0_resolved = False
        l0_cost_usd = float(pre_l0_cost_usd) if _use_pre_l0 else 0.0
        l0_keywords: List[Dict[str, Any]] = (
            _l0_keywords_from_slice(pre_l0_slice) if _use_pre_l0 else []
        )

        async def _resolve_l0_entities() -> None:
            nonlocal l0_result, llm_completed, regex_result, hard_ents, soft_ents, l0_resolved, l0_cost_usd, l0_keywords
            if l0_resolved:
                return
            if l0_task is not None:
                try:
                    l0_result, l0_cost_usd = await l0_task
                    llm_completed = True
                except Exception as _e:  # noqa: BLE001 - resilience boundary: L0 failure degrades to None
                    logger.warning(
                        f"qi_ensemble_l0_error request_id={request_id} "
                        f"error_type={type(_e).__name__} error={_e}"
                    )
                    l0_result = None
                    llm_completed = False
                    l0_cost_usd = 0.0
            llm_hard = list(l0_result.entities) if l0_result is not None else []
            llm_soft = list(getattr(l0_result, 'soft_entities', None) or []) if l0_result is not None else []
            if not _use_pre_l0:
                # Same keyword copy as extract_l0_filters / _run_l0_extractors.
                l0_keywords = _l0_keywords_from_slice(l0_result)
            if self._should_run_regex_l0_fallback(
                llm_completed=llm_completed, llm_hard=llm_hard, llm_soft=llm_soft,
            ):
                try:
                    regex_result = await self._regex_entity_extractor.classify_async(l0_text)
                except asyncio.CancelledError:
                    raise
                except Exception as _e:  # noqa: BLE001 - resilience boundary: regex failure degrades to None
                    logger.warning(
                        f"qi_ensemble_regex_error request_id={request_id} "
                        f"error_type={type(_e).__name__} error={_e}"
                    )
                    regex_result = None
            merged_hard, merged_soft = _merge_extractor_entities(
                regex_result, l0_result, l0_text, soft_names, hard_names,
                llm_completed=llm_completed,
            )
            combined = _deconflict_similar_to_vs_tld(
                _drop_numeric_misattributions(merged_hard + merged_soft, l0_text)
            )
            hard_ents = [e for e in combined if e.name not in soft_names]
            soft_ents = [e for e in combined if e.name in soft_names]
            l0_resolved = True

        if _extract_before or _use_pre_l0:
            await _resolve_l0_entities()

        # L1 SemanticRouter - after L0 when extract_before_classify; else may overlap L0.
        l1_result = None
        if self._semantic_router is not None:
            try:
                l1_result = await asyncio.to_thread(self._semantic_router.classify, l1_text)
            except Exception as _e:  # noqa: BLE001 - resilience boundary: L1 failure degrades to None
                logger.warning(
                    f"qi_ensemble_l1_error request_id={request_id} "
                    f"error_type={type(_e).__name__} error={_e}"
                )
                l1_result = None

        # L2 LLM - after L1; optionally after L0 when consensus_cancel_l2 so ngram+entity
        # agreement can skip L2 before the LLM call starts.
        _l1_conf = float(l1_result.confidence) if l1_result is not None else 0.0
        _skip_threshold = float(self._config.llm.l1_skip_l2_confidence_threshold)
        l2_result = None
        _l2_available = self._llm is not None and circuit_allows and self._ensemble_resolver is not None
        _consensus_cancel, _cancel_threshold = self._ensemble_consensus_cancel_l2()
        _skip_l2_l0_ngram = False
        _entity_vote_early: Optional[Vote] = None
        _entity_vote_computed = False

        if _l2_available and _l1_conf < _skip_threshold and _consensus_cancel:
            await _resolve_l0_entities()
            l0_entities_early = hard_ents + soft_ents
            if self._entity_type_voter is not None:
                _entity_vote_early = self._entity_type_voter.classify(l0_text, l0_entities_early)
                _entity_vote_computed = True
            _skip_l2_l0_ngram = self._l0_ngram_agree_skip_l2(
                _ngram_intent, _entity_vote_early, _cancel_threshold,
            )
            if _skip_l2_l0_ngram:
                logger.info(
                    f"qi_ensemble_l2_skipped_l0_ngram_agree request_id={request_id} "
                    f"archetype={_ngram_intent!r} threshold={_cancel_threshold:.3f}"
                )

        if _l2_available and _l1_conf < _skip_threshold and not _skip_l2_l0_ngram:
            _t3_timeout = float(self._config.llm.tier_3_timeout_seconds)
            l2_result = await self._l2_classify(normalized_sub_query, request_id, _t3_timeout)
        elif _l2_available and _l1_conf >= _skip_threshold:
            logger.debug(
                f"qi_ensemble_l2_skipped request_id={request_id} "
                f"l1_confidence={_l1_conf:.3f} threshold={_skip_threshold:.3f}"
            )

        if not l0_resolved:
            await _resolve_l0_entities()

        l0_entities = hard_ents + soft_ents  # voter sees all slots

        # Step 3 - Build votes: L0 entity voter, L1, L2, aggregation_gate, ngram_gate
        all_votes: 'List[Vote]' = []

        if _agg_intent:
            all_votes.append(Vote(voter_id='aggregation_gate', archetype=_agg_intent, confidence=1.0))
        elif self._aggregation_gate is not None:
            all_votes.append(abstain('aggregation_gate'))

        if _ngram_intent:
            all_votes.append(Vote(voter_id='ngram_gate', archetype=_ngram_intent, confidence=0.95))
        elif self._ngram_pre_gate is not None:
            all_votes.append(abstain('ngram_gate'))

        if self._entity_type_voter is not None:
            # Reuse early ballot when already computed for the L2 skip gate
            # (including abstain=None).
            _ev = (
                _entity_vote_early
                if _entity_vote_computed
                else self._entity_type_voter.classify(l0_text, l0_entities)
            )
            all_votes.append(_ev if _ev is not None else abstain(self._entity_type_voter.voter_id))

        if l1_result is not None:
            all_votes.append(Vote(
                voter_id='semantic',
                archetype=l1_result.query_type,
                confidence=float(l1_result.confidence),
            ))
        elif self._semantic_router is not None:
            all_votes.append(abstain('semantic'))

        l2_cost_usd = 0.0
        _l2_slices: 'List[IntentSlice]' = []
        _l2_alts: 'List[IntentSlice]' = []
        if l2_result is not None:
            _l2_slices, _top_conf, _model_used, _usage, _l2_alts = l2_result
            l2_cost_usd = float(compute_call_cost_usd(_model_used, _usage))
            if _l2_slices:
                all_votes.append(Vote(
                    voter_id='llm',
                    archetype=_l2_slices[0].query_type,
                    confidence=float(_l2_slices[0].confidence),
                ))
            else:
                all_votes.append(abstain('llm'))
        elif self._llm is not None:
            all_votes.append(abstain('llm'))

        # L0 (mandatory extract) + L2 (threshold escalation) - both LLM spend.
        cost = float(l0_cost_usd) + float(l2_cost_usd)

        # Resolve ensemble - always active; all available votes tallied.
        try:
            ensemble_result: EnsembleResult = self._ensemble_resolver.resolve(all_votes)
        except Exception as _ens_err:  # noqa: BLE001 - resilience boundary: ensemble resolve failure handled below
            logger.warning(
                f"qi_ensemble_resolve_error request_id={request_id} "
                f"error_type={type(_ens_err).__name__} error={str(_ens_err)}"
            )
            _fb_type = _agg_intent or self._config.default_query_type
            _fb_conf = 1.0 if _agg_intent else self._config.routing.fallback_confidence
            _fb_slice = IntentSlice(
                query_type=_fb_type,
                entities=list(hard_ents),
                confidence=_fb_conf,
                raw_text=normalized_sub_query,
                soft_entities=list(soft_ents),
                keywords=list(l0_keywords),
            )
            return self._ground_slices([_fb_slice]), 'L0_fallback', cost, []

        # L2 provides classification decision only (entities=[]).
        # Winner entities come exclusively from L0 entity extractor.
        winner_archetype = ensemble_result.archetype
        winner_slice = IntentSlice(
            query_type=winner_archetype,
            entities=list(hard_ents),
            confidence=ensemble_result.winner_confidence,
            raw_text=normalized_sub_query,
            soft_entities=list(soft_ents),
            keywords=list(l0_keywords),
        )
        grounded = self._ground_slices([winner_slice])
        return grounded, ensemble_result.decision_tier, cost, _l2_alts

    async def _classify_timeout_regex_fallback(
        self,
        normalized_sub_query: str,
        request_id: str,
    ) -> 'Tuple[List[IntentSlice], str, float, List[IntentSlice]]':
        """Keep inventory filters via regex L0 when the classify budget is exhausted.

        ``asyncio.wait_for`` cancels L1/L2/L0-LLM mid-flight, so the normal
        ``llm_completed=False -> regex`` merge path never runs. Re-run regex here
        (deterministic, sub-ms) and apply the same merge + cue reconcile so a
        timeout does not wipe structured filters that regex already knows.
        """
        l0_text = (
            self._term_disambiguator.disambiguate(normalized_sub_query)
            if self._term_disambiguator is not None
            else normalized_sub_query
        )
        if not self._should_run_regex_l0_fallback(
            llm_completed=False, llm_hard=[], llm_soft=[],
        ):
            return [], 'fallback', 0.0, []
        regex_result = None
        try:
            regex_result = await self._regex_entity_extractor.classify_async(l0_text)
        except asyncio.CancelledError:
            raise
        except Exception as _e:  # noqa: BLE001 - resilience boundary: regex failure stays empty fallback
            logger.warning(
                f"qi_classify_timeout_regex_error request_id={request_id} "
                f"error_type={type(_e).__name__} error={_e}"
            )
            return [], 'fallback', 0.0, []
        soft_names, hard_names = self._slot_sets()
        hard_ents, soft_ents = _merge_extractor_entities(
            regex_result, None, l0_text, soft_names, hard_names,
            llm_completed=False,
        )
        combined = _deconflict_similar_to_vs_tld(
            _drop_numeric_misattributions(hard_ents + soft_ents, l0_text)
        )
        hard_ents = [e for e in combined if e.name not in soft_names]
        soft_ents = [e for e in combined if e.name in soft_names]
        if not hard_ents and not soft_ents:
            return [], 'fallback', 0.0, []
        qtype, conf = self._resolve_l0_fallback_type(l0_text, hard_ents + soft_ents)
        slice_ = IntentSlice(
            query_type=qtype,
            entities=list(hard_ents),
            confidence=conf,
            raw_text=normalized_sub_query,
            soft_entities=list(soft_ents),
        )
        logger.info(
            f"qi_classify_timeout_regex_fallback request_id={request_id} "
            f"hard={len(hard_ents)} soft={len(soft_ents)} query_type={qtype}"
        )
        return self._ground_slices([slice_]), 'L0_fallback', 0.0, []

    async def _classify_single(
        self,
        normalized_sub_query: str,
        request_id: str,
        t_start: float = 0.0,
        pre_l0_slice: Optional[IntentSlice] = None,
        pre_l0_cost_usd: float = 0.0,
        pre_l0_llm_completed: bool = False,
    ) -> 'Tuple[List[IntentSlice], str, float, List[IntentSlice]]':
        """Classify one sub-query. Wraps _classify_single_inner with a hard timeout.

        Returns (grounded_slices, decision_tier, cost_usd, alternative_interpretations).
        """
        # Hard timeout for entire classification stage - prevents runaway classifier from consuming SLA
        classify_timeout = float(self._config.llm.classify_timeout_seconds)
        try:
            if classify_timeout > 0.0:
                return await asyncio.wait_for(
                    self._classify_single_inner(
                        normalized_sub_query, request_id, t_start,
                        pre_l0_slice=pre_l0_slice,
                        pre_l0_cost_usd=pre_l0_cost_usd,
                        pre_l0_llm_completed=pre_l0_llm_completed,
                    ),
                    timeout=classify_timeout,
                )
            else:
                return await self._classify_single_inner(
                    normalized_sub_query, request_id, t_start,
                    pre_l0_slice=pre_l0_slice,
                    pre_l0_cost_usd=pre_l0_cost_usd,
                    pre_l0_llm_completed=pre_l0_llm_completed,
                )
        except asyncio.TimeoutError:
            logger.warning(f"qi_classify_timeout request_id={request_id} timeout_sec={classify_timeout}")
            return await self._classify_timeout_regex_fallback(normalized_sub_query, request_id)

    async def _classify_single_inner(
        self,
        normalized_sub_query: str,
        request_id: str,
        t_start: float = 0.0,
        pre_l0_slice: Optional[IntentSlice] = None,
        pre_l0_cost_usd: float = 0.0,
        pre_l0_llm_completed: bool = False,
    ) -> 'Tuple[List[IntentSlice], str, float, List[IntentSlice]]':
        """Internal classification logic (wrapped by timeout in _classify_single)."""
        # TLD disambiguation before any gate - all downstream paths see ".ai" not "ai"
        l0_text = (
            self._term_disambiguator.disambiguate(normalized_sub_query)
            if self._term_disambiguator is not None
            else normalized_sub_query
        )

        return await self._classify_ensemble(
            l0_text, normalized_sub_query, request_id, t_start,
            pre_l0_slice=pre_l0_slice,
            pre_l0_cost_usd=pre_l0_cost_usd,
            pre_l0_llm_completed=pre_l0_llm_completed,
        )

    @staticmethod
    def _log_decision(request_id: str, tier: str, slices: List[IntentSlice], elapsed_ms: float, cost: float, sub_intents: int) -> None:
        """Emit a structured qi_decision log line."""
        if slices:
            primary = max(slices, key=lambda s: s.confidence)
            logger.info(
                f"qi_decision request_id={request_id} tier={tier} query_type={primary.query_type} "
                f"confidence={primary.confidence:.3f} sub_intents={sub_intents} slices={len(slices)} "
                f"latency_ms={elapsed_ms:.1f} cost_usd={cost:.6f}"
            )
        else:
            logger.info(f"qi_decision request_id={request_id} tier={tier} sub_intents={sub_intents} slices=0 latency_ms={elapsed_ms:.1f} cost_usd={cost:.6f}")
