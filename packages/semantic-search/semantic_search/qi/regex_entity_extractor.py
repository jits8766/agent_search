"""L0 deterministic entity extractor — offline regex slot filler for hard filter params.

Runs with no call_router, in sub-millisecond time, and emits the hard filter slots a
structured search needs: tld, price, auction type, name length, word count, bids, age,
traffic, value, character constraints, relative time, Majestic (trust/citation flow,
backlinks, referring domains) and SEMrush (authority, backlinks, referring domains,
indexed pages, search volume, cpc) metrics, digit count, lifecycle state, character
pattern, gem flag, and similar_to seeds. Numeric constraints are anchored to a comparator
polarity plus currency, unit, or metric noun. Bare comparator+number ("under 500") defaults
to the price family; a bare number with no currency, unit, or comparator is never slotted.
Topic classification stays with the LLM extractor (semantic, not deterministic).
Output is a hybrid IntentSlice of typed Entity objects tagged with the config source tag;
grounding validates tld/auction_type values.
"""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import datetime, timezone
from typing import Dict, FrozenSet, List, Optional, Protocol, Set, Tuple

from semantic_search.config.models import QIL0RegexEntityConfig
from semantic_search.contracts import Entity, IntentSlice
from semantic_search.core.exceptions import ConfigurationError
from semantic_search.core.logging_utils import get_logger
from semantic_search.qi.advisory_patterns import (
    INVENTORY_BOUND_RE as _INVENTORY_BOUND_RE,
    SOFT_ADVISORY_RE as _ADVISORY_NO_INVENTORY_RE,
    STRONG_ADVISORY_RE as _STRONG_ADVISORY_RE,
    is_soft_advisory_no_inventory,
    is_strong_advisory,
)
from semantic_search.qi.budget_price_cues import parse_budget_prefixed_price
from semantic_search.qi.llm_classifier import infer_chip_kind

logger = get_logger(__name__)

__all__ = ["EntityExtractor", "RegexEntityExtractor", "_STRONG_ADVISORY_RE"]

# Comparator phrase -> bound direction. 'older'/'newer' resolve on the same axis as
# min/max (older domain => higher age => min bound; newer/younger => max bound).
_COMPARATOR_DIRECTION: Dict[str, str] = {
    "under": "max",
    "below": "max",
    "less than": "max",
    "fewer than": "max",
    "at most": "max",
    "no more than": "max",
    "up to": "max",
    "maximum": "max",
    "max": "max",
    "cheaper than": "max",
    "younger than": "max",
    "newer than": "max",
    "or fewer": "max",
    "or less": "max",
    "and below": "max",
    "and under": "max",
    # Negated floor verbs = ceiling ("nothing over 7 chars" ≠ floor).
    "nothing over": "max",
    "not over": "max",
    "nothing above": "max",
    "not above": "max",
    # LLM: ".io domains capped at 500" -> maxPrice
    "capped at": "max",
    "cap at": "max",
    "capped to": "max",
    "over": "min",
    "above": "min",
    "more than": "min",
    "greater than": "min",
    # Truncated comparator stems (surface typos) — class, not query literals.
    "abov": "min",
    "belo": "max",
    "at least": "min",
    "minimum": "min",
    "min": "min",
    "older than": "min",
    "or more": "min",
    "and above": "min",
    "and over": "min",
    "plus": "min",
}

# Metric family -> (min_slot, max_slot, is_float, requires_comparator). A family whose
# requires_comparator is False treats a bare "N unit" as an exact bound (min == max).
_FAMILY_SPEC: Dict[str, Tuple[str, str, bool, bool]] = {
    "price": ("price_min", "price_max", True, True),
    "govalue": ("govalue_min", "govalue_max", True, True),
    "name_length": ("name_length_min", "name_length_max", False, False),
    "word_count": ("word_count_min", "word_count_max", False, False),
    "bids": ("bids_min", "bids_max", False, True),
    "domain_age": ("domain_age_min", "domain_age_max", False, True),
    "traffic": ("traffic_min", "traffic_max", False, True),
    "semrush_backlinks": (
        "semrush_backlinks_min",
        "semrush_backlinks_max",
        False,
        True,
    ),
    "semrush_authority": (
        "semrush_authority_min",
        "semrush_authority_max",
        False,
        True,
    ),
    "semrush_ref_domains": (
        "semrush_ref_domains_min",
        "semrush_ref_domains_max",
        False,
        True,
    ),
    "semrush_indexed_pages": (
        "semrush_indexed_pages_min",
        "semrush_indexed_pages_max",
        False,
        True,
    ),
    "semrush_search_volume": (
        "semrush_search_volume_min",
        "semrush_search_volume_max",
        False,
        True,
    ),
    "semrush_cpc": ("semrush_cpc_min", "semrush_cpc_max", True, True),
    "majestic_tf": ("majestic_tf_min", "majestic_tf_max", False, True),
    "majestic_cf": ("majestic_cf_min", "majestic_cf_max", False, True),
    "majestic_backlinks": (
        "majestic_backlinks_min",
        "majestic_backlinks_max",
        False,
        True,
    ),
    "majestic_ref_domains": (
        "majestic_ref_domains_min",
        "majestic_ref_domains_max",
        False,
        True,
    ),
    "digits": ("minDigits", "maxDigits", False, False),
    "buy_it_now_price": ("buy_it_now_min", "buy_it_now_max", True, True),
    # FIND63 families mirrored from LLM prompt (minUniqueSearches / Estibot*)
    "unique_searches": ("minUniqueSearches", "maxUniqueSearches", False, True),
    "estibot_domain_count": (
        "minEstibotDomainCount",
        "maxEstibotDomainCount",
        False,
        True,
    ),
    "estibot_domain_count_dev": (
        "minEstibotDomainCountDev",
        "maxEstibotDomainCountDev",
        False,
        True,
    ),
    "estibot_ext_count": ("minEstibotExtCount", "maxEstibotExtCount", False, True),
    "estibot_ext_count_dev": (
        "minEstibotExtCountDev",
        "maxEstibotExtCountDev",
        False,
        True,
    ),
}

# Trailing unit noun -> family (noun follows the number: "5 letters", "100 backlinks").
_UNIT_FAMILY: Dict[str, str] = {
    "letter": "name_length",
    "letters": "name_length",
    "character": "name_length",
    "characters": "name_length",
    "char": "name_length",
    "chars": "name_length",
    "word": "word_count",
    "words": "word_count",
    "keyword": "word_count",
    "keywords": "word_count",
    "length": "name_length",
    "lengths": "name_length",
    "bid": "bids",
    "bids": "bids",
    "year": "domain_age",
    "years": "domain_age",
    "visitor": "traffic",
    "visitors": "traffic",
    "visit": "traffic",
    "visits": "traffic",
    "hit": "traffic",
    "hits": "traffic",
    "pageview": "traffic",
    "pageviews": "traffic",
    "traffic": "traffic",
    "backlink": "semrush_backlinks",
    "backlinks": "semrush_backlinks",
    "link": "semrush_backlinks",
    "links": "semrush_backlinks",
    "digit": "digits",
    "digits": "digits",
    "number": "digits",
    "numbers": "digits",
    "numeral": "digits",
    "numerals": "digits",
    # bidding / bid count (LLM: "lots of bidding 20 plus")
    "bidding": "bids",
    # Trailing multi-word SEO units (Form A/C).
    "indexed page": "semrush_indexed_pages",
    "indexed pages": "semrush_indexed_pages",
    "unique searcher": "unique_searches",
    "unique searchers": "unique_searches",
    "unique search": "unique_searches",
    "unique searches": "unique_searches",
    # Bare "referring domains" -> Semrush (LLM grounding); Majestic when "majestic" cued.
    "referring domain": "semrush_ref_domains",
    "referring domains": "semrush_ref_domains",
    "ref domain": "semrush_ref_domains",
    "ref domains": "semrush_ref_domains",
    "majestic referring domain": "majestic_ref_domains",
    "majestic referring domains": "majestic_ref_domains",
}

# Leading metric noun -> family (noun precedes the number: "price under 50", "value over 5000").
# Multi-word keys are matched by _FORM_B_RE and normalized to single-spaced lowercase before lookup.
_METRIC_FAMILY: Dict[str, str] = {
    "price": "price",
    "cost": "price",
    "budget": "price",
    # Opening/current bid amount = price floor (not bid-count family).
    "starting bid": "price",
    "opening bid": "price",
    "current bid": "price",
    "bid floor": "price",
    "floor": "price",
    "floor price": "price",
    "value": "govalue",
    "valuation": "govalue",
    "govalue": "govalue",
    "appraisal": "govalue",
    "appraised": "govalue",
    "appraised value": "govalue",
    "traffic": "traffic",
    "visitors": "traffic",
    "hits": "traffic",
    "monthly hits": "unique_searches",
    "monthly visitors": "unique_searches",
    "monthly traffic": "traffic",
    "visitors per month": "unique_searches",
    "semrush links": "semrush_backlinks",
    "semrush link": "semrush_backlinks",
    "monthly unique searchers": "unique_searches",
    "unique searchers": "unique_searches",
    "age": "domain_age",
    "bid": "bids",
    "bids": "bids",
    "bidding": "bids",
    "backlink": "semrush_backlinks",
    "backlinks": "semrush_backlinks",
    "majestic backlinks": "majestic_backlinks",
    "majestic referring domains": "majestic_ref_domains",
    "domain authority": "semrush_authority",
    "authority score": "semrush_authority",
    "authority": "semrush_authority",
    "ascore": "semrush_authority",
    "a score": "semrush_authority",
    "trust flow": "majestic_tf",
    "majestic trust flow": "majestic_tf",
    "tf": "majestic_tf",
    "citation flow": "majestic_cf",
    "majestic citation flow": "majestic_cf",
    "cf": "majestic_cf",
    "referring domains": "semrush_ref_domains",
    "ref domains": "semrush_ref_domains",
    "indexed pages": "semrush_indexed_pages",
    "crawled pages": "semrush_indexed_pages",
    "search volume": "semrush_search_volume",
    "monthly searches": "semrush_search_volume",
    "cpc": "semrush_cpc",
    "cost per click": "semrush_cpc",
    # BIN *price* only — bare "buy it now" is the buy_it_now bool flag, not a price metric.
    "buy it now price": "buy_it_now_price",
    "bin price": "buy_it_now_price",
    # FIND63 — unique searches / Estibot (LLM Parameter hints + Group 3)
    "unique searches": "unique_searches",
    "unique search": "unique_searches",
    "estibot domain count": "estibot_domain_count",
    "estibot count": "estibot_domain_count",
    "estibot domain count developed": "estibot_domain_count_dev",
    "estibot developed domain count": "estibot_domain_count_dev",
    "estibot ext count": "estibot_ext_count",
    "estibot extension count": "estibot_ext_count",
    # "(extension|ext) saturation" is a quality/authority cue in grounding
    # (row evidence: "extension saturation above 30" -> minSemrushAScore:31),
    # not an Estibot ext-count bound.
    "extension saturation": "semrush_authority",
    "ext saturation": "semrush_authority",
    "estibot ext count developed": "estibot_ext_count_dev",
    "estibot developed ext count": "estibot_ext_count_dev",
    # "dev ext count" -> base Estibot ext count (grounding uses minEstibotExtCount).
    # Bare "dev extension" without count/saturation stays TLD via _BARE_TLD_EXT_RE.
    "dev ext count": "estibot_ext_count",
    "dev extension count": "estibot_ext_count",
    # "dev ext saturation" -> developed-domain-count family, distinct from bare
    # "dev ext count" (row evidence: "dev ext saturation above 15" ->
    # minEstibotDomainCountDev:15, not minEstibotExtCount/Dev).
    "dev ext saturation": "estibot_domain_count_dev",
    "estibot dev ext": "estibot_ext_count_dev",
    "estibot dev extension": "estibot_ext_count_dev",
    "developed ext count": "estibot_ext_count_dev",
    "developed extension count": "estibot_ext_count_dev",
    "dev namespace count": "estibot_domain_count",
}

# Relative-time unit -> seconds (pure unit conversion, not a tunable threshold).
_TIME_UNIT_SECONDS: Dict[str, int] = {
    "minute": 60,
    "min": 60,
    "hour": 3600,
    "hr": 3600,
    "day": 86400,
    "week": 604800,
}

# Slots whose accumulated value is a list; every other emitted slot is scalar first-seen.
_LIST_SLOTS: frozenset = frozenset(
    {
        "tld",
        "tldExcludeList",
        "auction_type",
        "typeExcludeList",
        "keyword_contains",
        "keyword_starts_with",
        "keyword_ends_with",
        "keyword_contains_exclude",
        "similar_to",
        "topic_include",
        "topic_exclude",
    }
)

# Longer negated ceilings before bare over/above ("nothing over 1k" ≠ minPrice).
_CMP = (
    r"nothing\s+over|not\s+over|nothing\s+above|not\s+above|"
    r"under|below|less\s+than|fewer\s+than|at\s+most|no\s+more\s+than|up\s+to|maximum|max|cheaper\s+than|"
    r"capped\s+at|cap\s+at|capped\s+to|"
    r"younger\s+than|newer\s+than|over|above|more\s+than|greater\s+than|at\s+least|minimum|min|older\s+than"
)
_CUR = r"[$€£¥₹]"
_CURWORD = r"usd|eur|gbp|inr|cad|aud|jpy|dollars?|euros?|pounds?"
# ISO / word currency tokens -> API filterPriceCurrency (uppercase).
_CURRENCY_CODE_RE = re.compile(
    r"\b(?P<code>usd|eur|gbp|inr|cad|aud|jpy)\b", re.IGNORECASE
)
_CURRENCY_WORD_RE = re.compile(r"\b(?P<word>dollars?|euros?|pounds?)\b", re.IGNORECASE)
# Nationality / adjective + dollars -> ISO (before bare "dollars"->USD).
_CURRENCY_LOCALE_RE = re.compile(
    r"\b(?P<locale>canadian|canadien|aussie|australian|american|us|u\.s\.)\s+dollars?\b",
    re.IGNORECASE,
)
_CURRENCY_LOCALE_TO_CODE: Dict[str, str] = {
    "canadian": "CAD",
    "canadien": "CAD",
    "aussie": "AUD",
    "australian": "AUD",
    "american": "USD",
    "us": "USD",
    "u.s.": "USD",
}
_CURRENCY_WORD_TO_CODE: Dict[str, str] = {
    "dollar": "USD",
    "dollars": "USD",
    "euro": "EUR",
    "euros": "EUR",
    "pound": "GBP",
    "pounds": "GBP",
}
# ISO codes that must never be treated as TLDs (LLM often emits eur->tld).
_CURRENCY_CODES_LOWER: FrozenSet[str] = frozenset(
    {"usd", "eur", "gbp", "inr", "cad", "aud", "jpy"}
)
# Metric nouns whose bare "metric N" (no comparator) means a ceiling, not a floor.
_CEILING_METRICS: FrozenSet[str] = frozenset({"budget", "cost"})
# Trailing polarity after the number: "budget 1500 max", "price 200 minimum".
# Trailing polarity after the number: "budget 1500 max", "dev ext 25 plus".
_TRAIL_CMP = r"max(?:imum)?|min(?:imum)?|plus|or\s+more"
_UNIT = (
    r"letters?|characters?|chars?|keywords?|words?|lengths?|bids?|bidding|years?|"
    r"(?:monthly\s+)?(?:visitors?|visits?|traffic|pageviews?|hits?|unique\s+searchers?|unique\s+searches?)|"
    r"indexed\s+pages?|crawled\s+pages?|referring\s+domains?|ref\s+domains?|"
    r"semrush\s+links?|backlinks?|links?|digits?|numbers?|numerals?"
)
# One number, or several coordinated by or / comma / slash / range word or dash.
# 'and' is intentionally excluded: it links separate clauses ("price under 500 and
# 3 chars") and the between-form already owns "between X and Y".
# Optional k/m suffix: "under 3k" -> 3000, "under 2m" -> 2000000 (via _to_number).
# Digit token with optional k/m suffix OR word scale ("2 thousand" / "2 million").
_NUM = r"\d[\d,]*(?:[kKmM]|(?:\s*(?:thousand|million|billion)))?"
_NUMLIST = _NUM + r"(?:\s*(?:or|,|/|to|through|[-–])\s*" + _NUM + r")*"
_NUM_RE = re.compile(_NUM)

_FORM_A_RE = re.compile(
    r"(?:(?P<cmp>" + _CMP + r")\s+)?(?P<cur>" + _CUR + r")?\s*(?P<num>" + _NUM + r")"
    r"\s*(?P<curword>" + _CURWORD + r")?\s*(?P<unit>" + _UNIT + r")?",
    re.IGNORECASE,
)
_FORM_B_RE = re.compile(
    r"\b(?P<metric>majestic\s+trust\s+flow|majestic\s+citation\s+flow|majestic\s+referring\s+domains|majestic\s+backlinks|"
    r"estibot\s+domain\s+count\s+developed|estibot\s+developed\s+domain\s+count|"
    r"estibot\s+ext\s+count\s+developed|estibot\s+developed\s+ext\s+count|"
    r"estibot\s+extension\s+count|estibot\s+ext\s+count|estibot\s+domain\s+count|"
    r"estibot\s+count|dev\s+ext\s+saturation|dev\s+ext(?:ension)?\s+count|dev\s+ext(?:ension)?|"
    r"dev\s+namespace\s+count|"
    r"extension\s+saturation|ext\s+saturation|"
    r"unique\s+searches|unique\s+search|monthly\s+unique\s+searchers|unique\s+searchers|"
    r"trust\s+flow|citation\s+flow|referring\s+domains|ref\s+domains|indexed\s+pages|crawled\s+pages|search\s+volume|"
    r"monthly\s+searches|cost\s+per\s+click|domain\s+authority|authority\s+score|a\s+score|ascore|"
    r"buy\s+it\s+now\s+price|bin\s+price|"
    r"starting\s+bid|opening\s+bid|current\s+bid|bid\s+floor|floor\s+price|floor|"
    r"price|cost|budget|valuation|value|govalue|"
    r"appraised\s+value|appraised|appraisal|semrush\s+links?|"
    r"traffic|visitors|monthly\s+hits|monthly\s+visitors|hits|authority|age|cpc|tf|cf|bids?|bidding|backlinks?)\b\s+"
    # Optional soft polarity between metric and bound ("ext count high under 1500").
    r"(?:(?:high|low|strong|barely\s+any)\s+)?"
    r"(?:(?P<cmp>" + _CMP + r")\s+)?(?P<cur>" + _CUR + r")?\s*(?P<num>" + _NUM + r")"
    r"(?:\s*(?P<trail_cmp>" + _TRAIL_CMP + r"))?",
    re.IGNORECASE,
)
_BETWEEN_RE = re.compile(
    r"\bbetween\s+(?P<cur1>"
    + _CUR
    + r")?\s*(?P<lo>"
    + _NUM
    + r")\s+(?:and|to|[-–])\s+(?P<cur2>"
    + _CUR
    + r")?\s*"
    r"(?P<hi>" + _NUM + r")\s*(?P<unit>" + _UNIT + r")?",
    re.IGNORECASE,
)
# "euros 500 to 1500 range" / "500 to 1500" price span without the word between.
_RANGE_TO_RE = re.compile(
    r"\b(?P<lo>" + _NUM + r")\s*(?:to|[-–])\s*(?P<hi>" + _NUM + r")"
    r"(?:\s*(?P<unit>" + _UNIT + r"))?(?:\s+range)?\b",
    re.IGNORECASE,
)
# Post-fix measure list: comparator + coordinated numbers + trailing unit
# ("under 3 or 5 chars", "5 to 7 letters"). Unit is required, so currency is untouched.
_FORM_LIST_RE = re.compile(
    r"(?:(?P<cmp>"
    + _CMP
    + r")\s+)?(?P<nums>"
    + _NUMLIST
    + r")\s*(?P<unit>"
    + _UNIT
    + r")",
    re.IGNORECASE,
)
# Unit-first measure: optional scope qualifier + measure noun + comparator + numbers
# ("length 7", "total length under 7", "sld length 5 to 7", "keywords 3"). scope
# 'tld'/'top-level' on a length measure carries no numeric filter slot and is dropped.
_MEASURE_PREFIX_RE = re.compile(
    r"\b(?P<scope>total|overall|domain|name|sld|tld|second[\s-]?level|top[\s-]?level)?\s*"
    r"(?P<measure>lengths?|char(?:acter)?s?|letters?|keywords?|words?)\s+"
    r"(?:of\s+|in\s+(?:the\s+)?(?:domain|name)\s+)?"
    r"(?:(?P<cmp>" + _CMP + r")\s+)?(?P<nums>" + _NUMLIST + r")",
    re.IGNORECASE,
)
# Measure noun -> metric family. SLD character length and word/keyword count only.
_MEASURE_FAMILY: Dict[str, str] = {
    "length": "name_length",
    "char": "name_length",
    "character": "name_length",
    "letter": "name_length",
    "keyword": "word_count",
    "word": "word_count",
}
# Spelled small quantities that precede a measure unit ("one word", "three letter",
# "single word"). Kept small and generic — the token must sit next to a measure unit.
_SPELLED_NUM: Dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "single": 1,
}
_SPELLED_MEASURE_RE = re.compile(
    r"\b(?:(?P<cmp>" + _CMP + r")\s+)?(?P<qty>" + "|".join(_SPELLED_NUM) + r")[\s-]+"
    r"(?P<unit>letters?|characters?|chars?|keywords?|words?|lengths?)\b",
    re.IGNORECASE,
)
# Full-unit spelled measure: covers all _UNIT families (bids, domain_age, traffic, etc.).
# Unlike _SPELLED_MEASURE_RE (name_length/word_count only), this handles every trailing-unit family.
_SPELLED_NUMERIC_RE = re.compile(
    r"\b(?:(?P<cmp>"
    + _CMP
    + r")\s+)?(?P<qty>"
    + "|".join(re.escape(k) for k in sorted(_SPELLED_NUM, key=len, reverse=True))
    + r")[\s-]+"
    r"(?P<unit>" + _UNIT + r")\b",
    re.IGNORECASE,
)

# Exclude cue + coordinated TLD list: "exclude .xyz and .club", "just not .xyz .info or .club".
# Allows space-separated dotted TLDs (no and/or required between them).
_TLD_EXCLUDE_RE = re.compile(
    r"\b(?:not|exclude|excluding|without|except|avoid|no|skip|just\s+not)\s+"
    r"(?P<body>\.?(?P<tld>[a-z]{2,24})"
    r"(?:\s*(?:(?:and|or|,|/)\s*)?\.?[a-z]{2,24})*)\b",
    re.IGNORECASE,
)
# LLM: tldExcludeList is extension tokens only. Block char/auction meta words ("no hyphens").
_TLD_EXCLUDE_BLOCKLIST: FrozenSet[str] = frozenset(
    {
        "hyphen",
        "hyphens",
        "dash",
        "dashes",
        "number",
        "numbers",
        "digit",
        "digits",
        "numeral",
        "numerals",
        "letter",
        "letters",
        "character",
        "characters",
        "char",
        "chars",
        "word",
        "words",
        "vibe",
        "vibes",
        "backorder",
        "backorders",
        "closeout",
        "partner",
        "partners",
        "godaddy",
        "premium",
        "expiry",
        "expiring",
        "firehose",
        "dropcatch",
        "preregistration",
        "prereg",
        # Non-TLD industry tokens often after "but not …"
        "crypto",
        "fintech",
        "saas",
        "startup",
        "brand",
        "brands",
    }
)

# Static set of well-known TLDs accepted as bare exclude tokens (no leading dot required)
# when preceded by an exclude speech-act ("exclude xyz", "no com").
_KNOWN_TLDS_FOR_BARE_EXCLUDE: FrozenSet[str] = frozenset(
    {
        "com",
        "net",
        "org",
        "io",
        "ai",
        "co",
        "app",
        "dev",
        "xyz",
        "info",
        "biz",
        "us",
        "uk",
        "in",
        "de",
        "fr",
        "ca",
        "au",
    }
)
# Noun suffixes and connectors that appear in TLD-exclude body text but are not TLDs.
# Stripped from toks before gate evaluation so "exclude xyz domains" -> ['xyz'] only.
_TLD_BODY_NOISE_WORDS: FrozenSet[str] = frozenset(
    {
        "domain",
        "domains",
        "extension",
        "extensions",
        "name",
        "names",
        "tld",
        "tlds",
        "site",
        "sites",
    }
)
_TLD_DOTTED_RE = re.compile(r"\.(?P<tld>[a-z]{2,24})\b", re.IGNORECASE)
_TLD_CONTEXT_RE = re.compile(
    r"\b(?:ending\s+in|extension|tld)\s+\.?(?P<tld>[a-z]{2,24})\b|\bin\s+\.(?P<tld2>[a-z]{2,24})\b",
    re.IGNORECASE,
)

# Keys hold the normalized surface form (lowercase, no spaces or hyphens); see _auction_label.
# buy-now / buyout / bin intentionally excluded — these map to the buy_it_now bool slot,
# not to the auction_type structural param, avoiding a systematic false-positive overlap.
_AUCTION_KEYWORD_TO_LABEL: Dict[str, str] = {
    # Bare "expired" is lifecycle_state (LLM), not auction type — keep expiring/expiry/expires.
    "expiring": "expiry",
    "expiry": "expiry",
    "expire": "expiry",
    "expires": "expiry",
    "closeout": "closeout",
    "premiumauction": "premium",
    "bidpremium": "premium",
    # LLM grounding: "standard auction" -> typeIncludeList=listed
    # "standard auction" -> listed; "standard or partner" remapped in _parse_auction.
    "standard": "listed",
    "standardauction": "listed",
    "standardauctions": "listed",
    # 25 = Drop Catch (Private Backorder)
    "backorder": "backorder",
    "backorders": "backorder",
    "dropping": "backorder",  # "dropping .com" -> typeIncludeList=backorder
    "dropcatch": "dropcatch",
    "dropcatchauction": "dropcatch",
    "dropcatchauctions": "dropcatch",
    # 37 = Firehose (Pre-registration)
    "firehose": "firehose",
    "firehoseauction": "firehose",
    "firehoseauctions": "firehose",
    "preregistration": "preregistration",
    "preregistrations": "preregistration",
    "prereg": "preregistration",
    # 38|39 = Partner; 16|20 = GoDaddy
    "partner": "partner",
    "partnerauction": "partner",
    "partnerauctions": "partner",
    "partners": "partner",
    "godaddy": "godaddy",
    "godaddyauction": "godaddy",
    "godaddyauctions": "godaddy",
    # Bare ``gd`` = GoDaddy (same as godaddy); ``gd transfer`` handled by gd_transfer bool.
    "gd": "godaddy",
    "gdauction": "godaddy",
    "gdauctions": "godaddy",
}
_AUCTION_RE = re.compile(
    r"\b("
    r"expiring|expiry|expires?|close\s?-?out|premium\s+auction|bid\s+premium|backorders?"
    # "dropping .com" -> typeIncludeList=backorder (LLM parity; not lifecycle_state).
    r"|dropping"
    r"|standard(?:\s+auctions?)?"
    r"|drop\s*-?\s*catch(?:\s+auctions?)?"
    r"|firehose(?:\s+auctions?)?"
    r"|pre-?registrations?"
    r"|partner(?:\s+auctions?)?"
    r"|(?:go\s*daddy|godaddy)(?:\s+auctions?)?"
    # Bare gd = GoDaddy auctions; do not steal "gd transfer" (gd_transfer bool).
    r"|gd(?:\s+auctions?)?(?!\s+transfer)"
    r")\b",
    re.IGNORECASE,
)
# Explicit numeric auction-type IDs: "auction type 16", "type id 25" -> value kept as ID.
_AUCTION_TYPE_ID_RE = re.compile(
    r"\b(?:auction\s+)?types?\s+(?:id\s+|number\s+|code\s+|#\s*)?(?P<id>\d{1,3})\b",
    re.IGNORECASE,
)
_AUCTION_EXCLUDE_RE = re.compile(
    r"\b(?:not|exclude|excluding|no)\s+("
    r"expiring|expiry|close\s?-?out|premium\s+auction|backorders?"
    r"|drop\s*-?\s*catch(?:\s+auctions?)?"
    r"|firehose(?:\s+auctions?)?"
    r"|pre-?registrations?"
    r"|partner(?:\s+auctions?)?"
    r"|(?:go\s*daddy|godaddy)(?:\s+auctions?)?"
    r"|gd(?:\s+auctions?)?(?!\s+transfer)"
    r")\b",
    re.IGNORECASE,
)

_KEYWORD_STARTS_RE = re.compile(
    r"\b(?:start(?:s|ing)?\s+with|begin(?:s|ning)?\s+with|prefix(?:ed\s+with)?)\s+"
    r"(?:the\s+(?:word|letters?)\s+)?['\"]?(?P<kw>[a-z0-9]{1,32})",
    re.IGNORECASE,
)
# "starts with get or go" — multi-prefix (LLM pipe-joins).
_KEYWORD_STARTS_MULTI_RE = re.compile(
    r"\b(?:start(?:s|ing)?\s+with|begin(?:s|ning)?\s+with)\s+"
    r"(?P<body>[a-z0-9]{1,32}(?:\s+(?:or|and)\s+[a-z0-9]{1,32})+)",
    re.IGNORECASE,
)
# "contains cloud or hub" — multi-value contains (LLM pipe-joins, e.g.
# keyword_contains:cloud|hub). Row evidence: "contains cloud or hub io under 500"
# -> keyword_contains:cloud|hub.
_KEYWORD_CONTAINS_MULTI_RE = re.compile(
    r"\b(?:contain(?:s|ing)?|includ(?:e|es|ing))\s+"
    r"(?:the\s+(?:word|keyword|phrase)\s+)?"
    r"(?P<body>[a-z0-9]{2,32}(?:\s+(?:or|and)\s+[a-z0-9]{2,32})+)",
    re.IGNORECASE,
)
_KEYWORD_ENDS_RE = re.compile(
    r"\b(?:end(?:s|ing)?\s+with|suffix(?:ed\s+with)?)\s+(?:the\s+(?:word|letters?)\s+)?['\"]?(?P<kw>[a-z0-9]{1,32})"
    # "ends in hq" (suffix cue). "ending in .com" stays TLD via _TLD_CONTEXT_RE.
    r"|\bends?\s+in\s+['\"]?(?P<kw2>[a-z0-9]{2,32})\b",
    re.IGNORECASE,
)

# Filter meta-words that describe numeric/structural properties of a domain name.
# These must NOT be extracted as keyword_contains values — they are filter operators,
# not domain name substrings. Extend this set to add new meta-terms.
_KEYWORD_CONTAINS_BLOCKLIST: FrozenSet[str] = frozenset(
    {
        "digit",
        "digits",
        "number",
        "numbers",
        "numeral",
        "numerals",
        "letter",
        "letters",
        "character",
        "characters",
        "char",
        "chars",
        "word",
        "words",
        "keyword",
        "keywords",
        "minimum",
        "maximum",
        "min",
        "max",
        "length",
        "lengths",
        "pattern",
        "name",
        "domain",
        "sld",
        # Auction-type / registrar labels — belong in auction_type / typeExcludeList, not keyword slots.
        "expiry",
        "expiring",
        "expired",
        "closeout",
        "premium",
        "backorder",
        "backorders",
        "partner",
        "partners",
        "godaddy",
        "gd",
        "firehose",
        "dropcatch",
        "preregistration",
        # Hedge / filler tokens from "not sure" / "not too" / "getting burned".
        # Quality adjectives after "doesn't sound X" / "non X" are allowed via speech-act bypass.
        "sure",
        "too",
        "getting",
        "direction",
        # Common function/quantifier/comparative words that can land in the
        # captured slot of "<word> keyword(s)" without being an actual target
        # token (e.g. "not just keyword", "popular keyword", "new keyword").
        "not",
        "just",
        "and",
        "or",
        "but",
        "also",
        "longer",
        "shorter",
        "new",
        "old",
        "popular",
        "proven",
        "some",
        "any",
        "many",
        "few",
        "good",
        "bad",
        "best",
        "worst",
        "top",
        "real",
        "specific",
        "particular",
        "certain",
        "single",
        "another",
        "other",
    }
)
# Inventory / quality / closed-class tokens before "X .tld" — never SLD substrings.
# Category stop-set for the dotted-TLD grammar (not per-query literals).
_KEYWORD_BEFORE_TLD_STOP: FrozenSet[str] = frozenset(
    {
        # size / price / age inventory modifiers
        "short",
        "long",
        "longer",
        "cheap",
        "budget",
        "aged",
        "fresh",
        "new",
        "old",
        "numeric",
        "eligible",
        "premium",
        "extended",
        # quality / brandability adjectives
        "clean",
        "catchy",
        "friendly",
        "brandable",
        "brandible",
        "popular",
        "proven",
        "trustworthy",
        "typeable",
        "pronounceable",
        # lifecycle / auction surface (not contain-cues)
        "dropping",
        "expired",
        "expiring",
        "auction",
        "auctions",
        "accepted",
        # closed-class / polarity / filler
        "is",
        "no",
        "or",
        "not",
        "now",
        "and",
        "the",
        "a",
        "an",
        "for",
        "with",
        "under",
        "below",
        "above",
        "over",
        "either",
        "just",
        "skip",
        "exclude",
        "added",
        "explore",
        "maybe",
        "perhaps",
        "please",
        "pls",
        "probs",
        "prob",
        # Inventory / navigational verbs — "find .com" is not keyword_contains=find.
        "find",
        "finding",
        "search",
        "searching",
        "show",
        "showing",
        "get",
        "getting",
        "list",
        "listing",
        "browse",
        "browsing",
        "look",
        "looking",
        "discover",
        # char meta (also in keyword_meta_blocklist.json)
        "hyphen",
        "hyphens",
        "hypen",
        "hypens",
        "hyphn",
        "dash",
        "dashes",
    }
)

# Tuple of patterns for keyword_contains; each has a named group `kw`.
# Extend by appending a new compiled pattern — no logic changes required.
_KEYWORD_CONTAINS_PATTERNS: Tuple[re.Pattern, ...] = (
    # "contains [the word/keyword/phrase] X" — optional article+noun before keyword
    re.compile(
        r"\b(?:contain(?:s|ing)?|includ(?:e|es|ing))\s+(?:the\s+(?:word|keyword|phrase)\s+)?['\"]?(?P<kw>[a-z0-9]{2,32})\b",
        re.IGNORECASE,
    ),
    # "with the word/keyword/phrase X"
    re.compile(
        r"\bwith\s+the\s+(?:word|keyword|phrase)\s+['\"]?(?P<kw>[a-z0-9]{2,32})\b",
        re.IGNORECASE,
    ),
    # "X keyword" / "X keywords" (cloud keyword .io)
    re.compile(r"\b(?P<kw>[a-z][a-z0-9]{1,24})\s+keywords?\b", re.IGNORECASE),
    # "X somewhere in [the] name/domain/sld"
    # NOTE: "dictionary word(s)" is word_count quality lexicon, not keyword_contains.
    re.compile(
        r"\b(?P<kw>[a-z0-9]{2,32})\s+(?:somewhere\s+)?in\s+(?:the\s+)?(?:name|domain|sld)\b",
        re.IGNORECASE,
    ),
    # "name has health in it" / "has X somewhere in it" / "has X in the name"
    re.compile(
        r"\b(?:name\s+)?has\s+(?P<kw>[a-z0-9]{2,32})\s+(?:somewhere\s+)?in\s+"
        r"(?:it|(?:the\s+)?(?:name|domain|sld))\b",
        re.IGNORECASE,
    ),
)

_CHAR_RULES: Tuple[Tuple[str, str, bool], ...] = (
    ("has_hyphen", r"\b(?:no|without|exclude|not)\s+hyphens?\b", False),
    ("has_hyphen", r"\bno\s+hypen(?:s)?\b", False),  # common typo
    ("has_hyphen", r"\bno\s+hyphn(?:s)?\b", False),  # common typo
    ("has_hyphen", r"\bhypen(?:s)?\b", False),  # common typo
    # Clean / pronounceable -> no hyphens (inventory or quality-noun scopes).
    (
        "has_hyphen",
        r"\b(?:easy\s+to\s+(?:spell|say|pronounce|type)|(?:typeable|pronounceable))\b",
        False,
    ),
    (
        "has_hyphen",
        r"\bclean(?:er)?\s+(?:brand(?:ables?)?|domains?|names?|letters?|slds?|com|start|minimal)\b",
        False,
    ),
    ("has_hyphen", r"\bclean\s+(?:backlinks?|links?|profiles?|history)\b", False),
    (
        "has_hyphen",
        r"\b(?:want|keep|make|prefer|need)\s+(?:it|them|names?|domains?)\s+clean\b",
        False,
    ),
    ("has_hyphen", r"\b(?:clean\s+and\s+brandable|brandable\s+and\s+clean)\b", False),
    (
        "has_hyphen",
        r"\b(?:with|has|contains?|include[sd]?)\s+(?:a\s+)?hyphens?\b",
        True,
    ),
    (
        "has_number",
        r"\b(?:no|without|exclude|not)\s+(?:numbers?|digits?|numerals?)\b",
        False,
    ),
    ("has_number", r"\bno\s+numbrs?\b", False),  # common typo
    (
        "has_number",
        r"\b(?:with|has|contains?|include[sd]?)\s+(?:numbers?|digits?|numerals?)\b",
        True,
    ),
    # Typeability cues -> excludeDigits. Bare "short" also implies no digits (LLM parity).
    (
        "has_number",
        r"\b(?:can\s+actually\s+type|easy\s+to\s+(?:say|type|spell|pronounce)|(?:typeable|pronounceable)|people\s+can\s+(?:actually\s+)?type)\b",
        False,
    ),
    ("has_number", r"\bleters?\s+only\b|\bletters?\s+only\b", False),
    # LLM: excludeLetters — 'no letters'/'digits only'/'numeric only'/'numbers only'/'pure digit|numeric'
    (
        "excludeLetters",
        r"\b(?:no\s+letters|digits?\s+only|numbers?\s+only|numeric\s+only|pure\s+(?:digits?|numeric)|numeric\s+domains?)\b",
        True,
    ),
    # short + brandable / explicit no-digits -> excludeDigits.
    # "short and clean" alone is hyphen-clean preference (not digit filter).
    (
        "has_number",
        r"\bshort\b.{0,32}\b(?:no\s+(?:numbers?|digits?|numbrs?)|letters?\s+only|typeable|easy\s+to\s+type|brandable|brandible)\b",
        False,
    ),
    (
        "has_number",
        r"\b(?:no\s+(?:numbers?|digits?|numbrs?)|letters?\s+only|typeable|easy\s+to\s+type|brandable|brandible)\b.{0,32}\bshort\b",
        False,
    ),
    ("is_idn", r"\b(?:idn|unicode|international(?:ized)?)\b", True),
    ("is_idn", r"\b(?:ascii|latin|english)[\s-]only\b|\bno\s+idn\b", False),
)
_CHAR_RULES_COMPILED: Tuple[Tuple[str, re.Pattern, bool], ...] = tuple(
    (slot, re.compile(pat, re.IGNORECASE), val) for slot, pat, val in _CHAR_RULES
)

# LLM: time_remaining_max — 'ending in 2 hours', 'closing in the next 6 hours', 'closing today'
_TIME_REMAINING_RE = re.compile(
    r"\b(?:ending|closing|expiring|ends?|closes?|expires?|within)\s+"
    r"(?:(?:in\s+(?:the\s+)?)?next\s+|in\s+|within\s+)?"
    r"(?P<num>\d+)\s*(?P<unit>minutes?|mins?|hours?|hrs?|days?|weeks?)\b",
    re.IGNORECASE,
)
# Calendar-word time aliases: extend by adding entries to this dict (no code changes needed).
_TIME_ALIAS_SECONDS: Dict[str, int] = {
    "today": 86_400,
    "tonight": 43_200,
    # Calendar "tomorrow" -> same -1d bucket as soon/today (L0 grounding).
    "tomorrow": 86_400,
    "this weekend": 259_200,
    "next week": 604_800,
    # LLM: ending/closing/expiring soon -> endTimeBefore=-1d (via ≤86400 mapping).
    "soon": 86_400,
    "last chance": 86_400,
    # Sub-day urgency still maps to -1d for L0 LLM parity (prompt rule: tonight/soon -> -1d).
    "next few hours": 86_400,
    "few hours": 86_400,
    "final hours": 86_400,
    "auctions ending": 86_400,
}
# Allow common typo stem "endng" (missing i) for calendar aliases.
_TIME_ALIAS_RE = re.compile(
    r"\b(?:end(?:ing|ng)|closing|expiring|ends?|closes?|expires?|wrapping\s+up|to\s+close)\s+"
    r"(?P<alias>"
    + "|".join(re.escape(k) for k in sorted(_TIME_ALIAS_SECONDS, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)
# Bare urgency without an ending verb: "last chance domains", "final hours domains".
_BARE_END_URGENCY_RE = re.compile(
    r"\b(?:last\s+chance|final\s+hours|soon\s+to\s+close|wrapping\s+up\s+soon|"
    r"ending\s+in\s+the\s+next\s+few\s+hours|next\s+few\s+hours)\b",
    re.IGNORECASE,
)

# Maps natural-language phrase (single-space-normalized lowercase) -> canonical lifecycle_state.
# _LIFECYCLE_RE is built from these keys (longest first). Add a phrase here to support new surface forms.
_LIFECYCLE_PHRASE_MAP: Dict[str, str] = {
    "pending delete": "pending_delete",
    "pending deletion": "pending_delete",
    # LLM grounding collapses recently_expired -> expired for filter parity.
    "recently expired": "expired",
    "expired domain": "expired",
    "expired domains": "expired",
    "expired auction": "expired",
    "expired auctions": "expired",
    # Bare "expired" / "expired with traffic" / typo "expird" -> lifecycle (LLM parity; not typeIncludeList).
    "expired": "expired",
    "expird": "expired",
    "live domains": "active",
    "live domain": "active",
    # "available now" -> active only when other inventory bound (see _parse_lifecycle).
    "available now": "active",
    # Drop calendar: "dropped today" -> deleted inventory (not pending_delete).
    "dropped today": "deleted",
    "what dropped today": "deleted",
    "newly dropped": "pending_delete",
    "active websites": "active",
    "expiring now": "expired",
    # "dropping" alone -> auction type backorder (see _AUCTION); soon variants -> pending_delete.
    "dropping soon": "pending_delete",
    "drop soon": "pending_delete",
    "lifecycle active": "active",
    "active auctions": "active",
    "active auction": "active",
    "active listings": "active",
    "active listing": "active",
    "active domains": "active",
    "active domain": "active",
    "status expiring": "expiring_soon",
    "expiring status": "expiring_soon",
    # "expiring soon" -> endTimeBefore via time alias (not lifecycle / auction type).
    "still open": "active",
    "domains live": "active",
    "auctions live": "active",
    "live auctions": "active",
}
_LIFECYCLE_RE = re.compile(
    r"\b(?P<phrase>"
    + "|".join(
        re.escape(k) for k in sorted(_LIFECYCLE_PHRASE_MAP, key=len, reverse=True)
    )
    + r")\b",
    re.IGNORECASE,
)
# keyword_match_mode: "match all/any [of these] keywords/terms/words" -> 'all' or 'any'.
_KEYWORD_MATCH_MODE_RE = re.compile(
    r"\b(?:match(?:ing)?|must\s+(?:have|include|contain))\s+(?:all|every)\b"
    r"|\ball\s+(?:keywords?|terms?|words?)\s+(?:must\s+)?(?:match|appear)\b"
    r"|\bmatch\s+any\s+(?:of\s+(?:the[se]?\s+)?)?(?:keywords?|terms?|words?)\b"
    r"|\bany\s+(?:of\s+(?:these?\s+)?)?(?:keywords?|terms?|words?)\b",
    re.IGNORECASE,
)
# keyword_contains_exclude: "not containing X", "without the word X", "exclude word X",
# "no X in name", "no crypto vibe", "non commercial", "doesn't sound sketchy".
_KEYWORD_CONTAINS_EXCLUDE_RE = re.compile(
    r"\b(?:not\s+containing|without\s+(?:the\s+)?(?:word|keyword)\s+|excluding\s+(?:the\s+)?(?:word\s+)?"
    r"|no\s+(?:word|keyword)\s+)\s*['\"]?(?P<kw>[a-z0-9]{2,32})\b"
    r"|\bnon[- ](?P<kw2>[a-z]{3,20})\b"
    r"|\bdoesn'?t\s+sound\s+(?:like\s+)?(?P<kw3>[a-z]{3,20})\b"
    r"|\bnot\s+sound\s+(?:like\s+)?(?P<kw4>[a-z]{3,20})\b",
    re.IGNORECASE,
)
_KEYWORD_NO_VIBE_OR_WORDS_RE = re.compile(
    r"\bno\s+(?P<multi>[a-z0-9]{2,20}(?:\s+or\s+[a-z0-9]{2,20})+)\s+words?\b"
    r"|\bno\s+(?P<single>[a-z0-9]{2,20})\s+(?:words?|vibe)\b",
    re.IGNORECASE,
)
# keyword_phrase: "exact phrase X" / "phrase match X Y" — quoted or 1–3 tokens.
_KEYWORD_PHRASE_RE = re.compile(
    r"\b(?:exact\s+phrase|the\s+phrase|phrase\s+match|phrase)\s+"
    r"(?:['\"](?P<qphrase>[^'\"]{1,48})['\"]|"
    r"(?P<phrase>[a-z0-9][a-z0-9\-]{1,32}(?:\s+[a-z0-9][a-z0-9\-]{1,32}){0,2}))\b",
    re.IGNORECASE,
)
# "with bids" / "has bids" / "most bids" / "heating up" -> minBids=1.
_WITH_BIDS_RE = re.compile(
    r"\b(?:with|has|have)\s+bids?\b"
    r"|\bmost\s+(?:bid\s+on\s+)?bids?\b|\bmost\s+bid\s+on\b"
    r"|\bbidding\s+on\b|\bheating\s+up\s+in\s+bids\b"
    r"|\bmost\s+competitive\s+auctions?\b"
    r"|\btop\s+current\s+bids?\b"
    r"|\bmost\s+bidded\b",
    re.IGNORECASE,
)
# Backoff guard for the bare _WITH_BIDS_RE fallback: a number or an explicit
# count-qualifier ("no", "zero", "one", "less than", "at least", ...) sitting
# close to the word bid(s) means a more specific numeric bid-count pattern
# owns this query (e.g. "under 2k with bids" is price + topic, not a bid
# filter; "one bid or less" / "3 bids" should bind via the specific pattern,
# not this bare fallback). Mirrors the bounded-window proximity idiom used
# elsewhere in this module (see _BID_COUNT_CUE_RE usage) instead of variable
# width lookaround, which `re` cannot express.
_BARE_BIDS_QUALIFIER_NEAR_RE = re.compile(
    r"\d.{0,12}\bbids?\b"
    r"|\bbids?\b.{0,12}\d"
    r"|\b(?:no|zero|one|less\s+than|fewer\s+than|at\s+least|more\s+than|greater\s+than)\b"
    r".{0,12}\bbids?\b"
    r"|\bbids?\b.{0,12}\b(?:no|zero|one|less\s+than|fewer\s+than|at\s+least|"
    r"more\s+than|greater\s+than)\b",
    re.IGNORECASE,
)
# "vintage domain 20 years old" / "aged domain 10 years" / "20 years old".
_VINTAGE_AGE_RE = re.compile(
    r"\b(?:vintage\s+domain\s+)?(?P<n>\d+)\s+years?\s+old\b"
    r"|\bvintage\s+(?:domain\s+)?(?P<n2>\d+)\s+years?\b"
    r"|\baged\s+(?:domain\s+)?(?P<n3>\d+)\s+years?\b"
    r"|\b(?:old|aged)\s+domains?\b"
    r"|\ban?\s+aged\s+domain\b",
    re.IGNORECASE,
)
# "cloud platform domain" -> topic_include=cloud.
_PLATFORM_TOPIC_RE = re.compile(
    r"\b(?P<topic>cloud|saas|fintech|ai)\s+platform\s+(?:domains?|names?)\b",
    re.IGNORECASE,
)
# Qualitative authority -> minSemrushAScore=1 (LLM soft floor).
_DECENT_AUTHORITY_RE = re.compile(
    r"\b(?:decent|strong|good|real|some)\s+authority\b"
    r"|\bdecent\s+da\b"
    r"|\bhigh\s+(?:da|domain\s+authority)\b"
    r"|\bauthority\s+(?:above|over)\b",
    re.IGNORECASE,
)
# "registered before 2010" -> minAge = current_year - year.
_REGISTERED_BEFORE_RE = re.compile(
    r"\bregistered\s+before\s+(?P<year>19\d{2}|20\d{2})\b",
    re.IGNORECASE,
)
# Qualitative SEO floors matching LLM soft stand-ins (rule-4 exceptions LLM already emits).
_STRONG_LINK_PROFILE_RE = re.compile(
    r"\bstrong\s+(?:link|backlink)\s+profiles?\b|\bstrong\s+backlinks?\b"
    r"|\b(?:with|has|have|and|or|decent|good|real|some|clean)\s+backlinks?\b"
    r"|\bbacklinks?\s+(?:present|available|juice)\b"
    r"|\bbacklink\s+juice\b"
    r"|\bbarely\s+any\s+backlinks?\b",
    re.IGNORECASE,
)
# Soft OR "traffic or backlinks or authority" — do NOT invent minMajesticBackLinks.
_SOFT_OR_LINK_TRAFFIC_RE = re.compile(
    r"\b(?:traffic|backlinks?|authority)\s+or\s+(?:traffic|backlinks?|authority)"
    r"(?:\s+or\s+(?:traffic|backlinks?|authority|some\s+real\s+value))?",
    re.IGNORECASE,
)
# Contrastive/conditional inventory clauses ("but if it's expired…") — do not invent
# lifecycle/backlink chips from the hypothetical branch.
_CONDITIONAL_INVENTORY_RE = re.compile(
    r"\bbut\s+if\b|\bif\s+it'?s\b|\bif\s+(?:they|those|it)\s+(?:are|is|were)\b"
    r"|\bif\s+(?:expired|pending|dropped)\b",
    re.IGNORECASE,
)
# "dont want to spend over N" / "spend over N" budget ceiling -> maxPrice.
_SPEND_OVER_MAX_RE = re.compile(
    r"\b(?:don'?t\s+want\s+to\s+)?(?:spend|spending|pay(?:ing)?)\s+"
    r"(?:over|above|more\s+than)\s+\$?(?P<n>\d[\d,]*)\s*(?P<k>k)?\b"
    r"|\bunder\s+a\s+thousand\b",
    re.IGNORECASE,
)
# Bare "spending 1k" / "spend $500" budget mention -> inclusive price ceiling.
_SPEND_AMOUNT_RE = re.compile(
    r"\b(?:spend(?:ing)?|pay(?:ing)?)\s+\$?(?P<n>\d[\d,]*)\s*(?P<k>k)?\b",
    re.IGNORECASE,
)
# Trailing incomplete "under" with no number -> maxPrice=999 (LLM invent for "ai domain under").
_BARE_UNDER_INCOMPLETE_RE = re.compile(
    r"\bunder\s*$",
    re.IGNORECASE,
)
# Current bid count (not starting-bid price): "bid is already above 500".
_BID_COUNT_CUE_RE = re.compile(
    r"\b(?:the\s+)?bids?\s+(?:(?:is|are)\s+)?(?:already\s+)?"
    r"(?:above|over|under|below|at\s+least|more\s+than|greater\s+than)",
    re.IGNORECASE,
)
_HIGH_CPC_RE = re.compile(
    r"\bhigh\s+cpc\b|\bhigh\s+cost\s+per\s+click\b", re.IGNORECASE
)
# "finance or legal niche" / "fintech niche" -> topic_include (LLM soft chip).
_NICHE_TOPIC_RE = re.compile(
    r"\b(?P<body>[a-z][a-z0-9]*(?:\s+or\s+[a-z][a-z0-9]*)+)\s+niche\b"
    r"|\b(?P<single>[a-z][a-z0-9]{2,20})\s+niche\b",
    re.IGNORECASE,
)
_NICHE_TOPIC_STOP = frozenset(
    {
        "the",
        "a",
        "an",
        "this",
        "that",
        "any",
        "some",
        "good",
        "best",
        "hot",
        "new",
        "low",
        "high",
        "top",
        # Metric / inventory nouns — not industry niches ("unique searches niche").
        "searches",
        "searchers",
        "search",
        "backlinks",
        "backlink",
        "links",
        "link",
        "traffic",
        "authority",
        "score",
        "ascore",
        "bids",
        "bid",
        "price",
        "prices",
        "domains",
        "domain",
        "names",
        "name",
        "words",
        "word",
        "letters",
        "letter",
        "count",
        "volume",
        "cpc",
        "da",
        "tf",
        "cf",
    }
)
# "new keyword" / "new keywords" -> startTimeAfter=-1d via days_listed_max=1.
_NEW_KEYWORD_RE = re.compile(r"\bnew\s+keywords?\b", re.IGNORECASE)
# Bare TLD token after keyword/prefix cues: "get or go com under 500".
# Bare TLD noun after inventory adjectives: "aged com", "expired com", "short com".
# Do NOT treat "short/premium ai …" as tld=ai (niche/topic owns ai/io/app/dev).
_BARE_TLD_NOUN_RE = re.compile(
    r"\b(?:aged|expired|expird|short|premium|single\s+word|one\s+word|word|"
    r"eligible|numeric|brandable|clean)\s+"
    r"(?P<tld>com|net|org|co|xyz)\b",
    re.IGNORECASE,
)
# "ai ending" inventory TLD noun. Not "dev ext count" (Estibot metric).
_BARE_TLD_EXT_RE = re.compile(
    r"\b(?P<tld>com|net|org|io|ai|co|app|xyz)\s+ending\b"
    r"|\b(?P<tld2>com|net|org|io|ai|co|app|dev|xyz)\s+ext(?:ension)?s?\b"
    r"(?!\s+count)",
    re.IGNORECASE,
)
_BARE_TLD_AFTER_RE = re.compile(
    r"\b(?:get|go|app|dev|net|org|io|ai|co|xyz|info)\s+"
    r"(?P<tld>com|net|org|io|ai|co|xyz|app|dev)\b"
    r"|\b(?P<tld2>com|net|org|io|ai|co|xyz|app|dev)\s+under\b",
    re.IGNORECASE,
)
# charPattern: v/c/n mnemonic ("cvcv pattern", "cvcc 5 letter", bare "vcvc under").
_CHAR_PATTERN_RE = re.compile(
    r"\bpattern\s+(?P<pat>[vcn]{2,12})\b"
    r"|\b(?P<pat2>[vcn]{2,12})\s+pattern\b"
    r"|\b(?P<pat3>[vcn]{2,8})\s+(?:\d+\s+)?(?:letter|letters|names?|under|below|brandable)\b"
    # "cvcv or vcvc" coordinated bare mnemonics
    r"|\b(?P<pat4>[vcn]{2,8})(?:\s+or\s+(?P<pat5>[vcn]{2,8}))+\b",
    re.IGNORECASE,
)
# Bare "gem domains" / "gems only" -> price_below_market (LLM); not isGemDomain.
_GEM_RE = re.compile(
    r"(?<!hidden )\bgem\s+(?:domains?|names?)\b|\bgems?\s+only\b"
    r"|(?<!hidden )\b(?:ai|io|com)\s+gem\b|(?<!hidden )\bgem\b(?=\s+(?:ending|under|below))",
    re.IGNORECASE,
)
_FRACTION_OF_WORTH_RE = re.compile(
    r"\bfraction\s+of\s+what\s+(?:it'?s|its|they'?re)\s+worth\b"
    r"|\bselling\s+for\s+a\s+fraction\b",
    re.IGNORECASE,
)
# similar_to: brand/seed. Includes "like X" / "sounds like X" / "X vibe" / "X style".
_SIMILAR_TO_RE = re.compile(
    r"\bsounds?\s+like\s+(?:a\s+|an\s+|real\s+)(?P<seed_brand>brand)\b"
    r"|\bsounds?\s+like\s+it\s+could\s+be\s+(?P<seed_startup>startup)\b"
    r"|\b(?:similar\s+to|resembling|sounds?\s+like|alternative\s+to|"
    r"spirit\s+of|in\s+the\s+spirit\s+of)\s+['\"]?(?P<seed>[a-z0-9][a-z0-9.\-]{1,40})"
    r"|\b(?:domains?|names?)\s+like\s+['\"]?(?P<seed2>[a-z0-9][a-z0-9.\-]{1,40})"
    # "somethin(g) like a fintech brand" / "brandable like stripe"
    r"|\b(?:something|somethin|name|domain|want|looking|brandable)\s+like\s+"
    r"(?:an?\s+|the\s+)?['\"]?(?P<seed4>[a-z0-9][a-z0-9.\-]{1,40})\b"
    r"|\blike\s+(?:an?\s+|the\s+)?(?P<seed5>[a-z0-9][a-z0-9.\-]{1,40})\s+brand\b"
    r"|\b(?P<seed3>[a-z0-9][a-z0-9.\-]{1,40})\s+(?:kinda\s+)?(?:vibe|style(?:\s+name)?|naming)\b",
    re.IGNORECASE,
)
# Multi-token brand compounds before style/naming ("linear app style naming").
_SIMILAR_STYLE_COMPOUND_RE = re.compile(
    r"\b(?:similar\s+to|like|sounds?\s+like)\s+"
    r"(?P<stylebody>[a-z][a-z0-9.\-]{1,40}(?:\s+[a-z][a-z0-9.\-]{1,40}){0,2})"
    r"\s+(?:kinda\s+)?(?:vibe|style(?:\s+name)?|naming)\b",
    re.IGNORECASE,
)
_SIMILAR_TO_MULTI_RE = re.compile(
    r"\b(?:domains?|names?)?\s*(?:like|similar\s+to|sounds?\s+like|alternative\s+to)\s+"
    r"(?P<body>[a-z0-9][a-z0-9.\-]*"
    r"(?:\s+(?:or|and)\s+(?:maybe\s+)?[a-z0-9][a-z0-9.\-]*)+)"
    # Space-separated brand list: "like slack stripe zoom" (no "but"/clause noise).
    r"|\b(?:domains?|names?)?\s*like\s+"
    r"(?P<body_sp>[a-z]{2,20}(?:\s+[a-z]{2,20}){1,3})"
    r"(?=\s*$|\s+(?:under|below|with|and|that|available))"
    # "product hunt or yc" multiword brands
    r"|\b(?:on\s+)?(?P<body2>product\s+hunt(?:\s+(?:or|and)\s+[a-z0-9][a-z0-9.\-]*)+|"
    r"[a-z0-9][a-z0-9.\-]*(?:\s+(?:or|and)\s+product\s+hunt)+)\b"
    r"|\b(?P<vibebody>[a-z0-9][a-z0-9.\-]*"
    r"(?:\s+(?:or|and)\s+(?:maybe\s+)?[a-z0-9][a-z0-9.\-]*)+)\s+(?:kinda\s+)?(?:vibe|style|naming)\b",
    re.IGNORECASE,
)
# Reject generic pronouns / fillers after "like" (keep brand seeds only).
# Closed-class / non-brand fillers after "like" (shape rules reject price seeds separately).
_SIMILAR_TO_STOP = frozenset(
    {
        "these",
        "those",
        "this",
        "that",
        "them",
        "it",
        "a",
        "an",
        "the",
        "my",
        "our",
        "real",
        "brands",
        "name",
        "names",
        "domain",
        "domains",
        "something",
        "anything",
        "ones",
        "one",
        "maybe",
        "specifically",
        "but",
        "for",
        "not",
        "too",
        "with",
        "and",
        "or",
        "to",
        "on",
        "in",
        "be",
        "could",
        "would",
        "they",
        "you",
        "see",
        "just",
        "under",
        "below",
        "cheaper",
        "related",
        "design",
        "services",
        "company",
        "keyword",
        "spam",
        "meets",
        "sure",
        "stays",
        "aged",
        "premium",
        "vibe",
        "available",
        "know",
        "buy",
        "spend",
        "spending",
        "money",
        "market",
        "marketing",
        "style",
        "naming",
        "vibe",
        "feel",
        "type",
        "kinda",
        # Keep 'app' — product compounds ("linear app style") are brand seeds.
    }
)
# Price-ish / numeric seeds after "like" ("like 2k", "like 500") — never brands.
_SIMILAR_TO_PRICEISH_RE = re.compile(r"^\d+[km]?$", re.IGNORECASE)
# "brand"/"startup" only via seed_brand / seed_startup named groups.
# Bare short TLD + "domains" (LLM: "ai domain" / "ai domains" -> tldIncludeList).
# Requires word boundary so "devtools domains" does not become tld=dev.
_TLD_BARE_DOMAIN_RE = re.compile(
    r"\b(?P<tld>ai|io|com|net|org|co|app|dev|xyz)\s+domains?\b",
    re.IGNORECASE,
)
# Bare multi-TLD disjunction: "com or io", "ai or io", "io or com under 1k".
_TLD_BARE_OR_RE = re.compile(
    r"\b(?P<body>(?:com|net|org|io|ai|co|app|dev|xyz)"
    r"(?:\s+or\s+(?:com|net|org|io|ai|co|app|dev|xyz))+)\b",
    re.IGNORECASE,
)
# Qualitative short names -> maxSldLen=5 (LLM: "short com or io").
_SHORT_NAME_RE = re.compile(r"\bshort\b", re.IGNORECASE)
# "fintech topic domains" / "saas topic names" -> topic_include (not keyword_contains).
_TOPIC_DOMAINS_RE = re.compile(
    r"\b(?P<topic>[a-z][a-z0-9]{1,24})\s+topic\s+(?:domains?|names?)\b",
    re.IGNORECASE,
)
# Niche token before price/auction that must NOT become tldIncludeList ("extended ai auction").
_TOPICISH_BEFORE_SHORT_TLD_RE = re.compile(
    r"\b(?:premium|extended|brandable|startup|fintech|saas|crypto|cloud|devtools|"
    r"edtech|healthcare|climate|gaming)\s+(?P<tok>ai|io|app|dev)\b"
    r"|\b(?P<tok2>ai)\s+(?:agent|names?|gems?|inventory|compan(?:y|ies)|products?|"
    r"startups?|brands?|tools?|niches?|meets)\b",
    re.IGNORECASE,
)
# "has fintech but not crypto" -> topic + keyword exclude (no contain-in-name cue).
_HAS_TOPIC_BUT_NOT_RE = re.compile(
    r"\bhas\s+(?P<topic>[a-z][a-z0-9]{1,24})\s+but\s+not\s+(?P<excl>[a-z][a-z0-9]{1,24})\b",
    re.IGNORECASE,
)
# Opening/starting bid / floor price -> price family (not bid count).
_STARTING_BID_PRICE_RE = re.compile(
    r"\b(?:starting|opening|current)\s+bid\b|\bbid\s+floor\b|\bfloor\s+(?:price|bid)\b"
    r"|\bfloor\s+\d",
    re.IGNORECASE,
)
# Data-driven niche lexicon (not TLDs) — extend set to add topics; grammar below.
_TOPIC_LEXICON: FrozenSet[str] = frozenset(
    {
        "saas",
        "fintech",
        "edtech",
        "devtools",
        "healthcare",
        "cybersecurity",
        "ecommerce",
        "climate",
        "climate_tech",
        "gaming",
        "legal",
        "logistics",
        "travel",
        "cloud",
        "b2b",
        "b2b_saas",
        "startup",
        "crypto",
        "ai",
        "ml",
        "iot",
        "web3",
        "health",
        "finance",
        "brandable",
        "tech",
        "seo",
        "developer",
    }
)
# Known TLD tokens — never emit as topic_include (prompt rule 1 -> tldIncludeList).
_TOPIC_TLD_TOKENS: FrozenSet[str] = frozenset(
    {
        "com",
        "net",
        "org",
        "io",
        "ai",
        "co",
        "app",
        "dev",
        "xyz",
        "info",
        "biz",
        "us",
        "uk",
    }
)
# "fintech domains" / "saas name" / "climate tech domains" (multiword via _).
_BARE_TOPIC_DOMAIN_RE = re.compile(
    r"\b(?P<topic>b2b\s+saas|climate\s+tech|cyber\s*security|"
    r"saas|fintech|edtech|devtools|healthcare|cybersecurity|"
    r"ecommerce|gaming|legal|logistics|travel|cloud|crypto|startup|"
    r"health|finance|developer|luxury)\s+(?:domains?|names?|apps?|startups?|listings?|"
    r"brand\s+domains?|audience|vertical|niche|segment|market)\b"
    # Bare "tech domains" only when not "climate/legal tech domains".
    r"|(?<!climate\s)(?<!legal\s)\b(?P<topic2>tech)\s+(?:domains?|names?)\b"
    # "seo domains" / "good seo domains" — not "good for seo" (quality cue elsewhere).
    r"|\b(?P<topic3>seo)\s+(?:topic|category|domains?)\b"
    # "luxury brand domains" (brand between niche and domains).
    r"|\b(?P<topic4>luxury)\s+brand\s+(?:domains?|names?)\b",
    re.IGNORECASE,
)
# "fintech or saas under 1500" / "saas or cloud but not gaming"
_TOPIC_OR_RE = re.compile(
    r"\b(?:in\s+)?(?P<body>(?:saas|fintech|edtech|devtools|healthcare|cybersecurity|"
    r"ecommerce|gaming|legal|logistics|travel|cloud|crypto|startup|tech|"
    r"health|finance|seo|developer|ai)"
    r"(?:\s+or\s+(?:saas|fintech|edtech|devtools|healthcare|cybersecurity|"
    r"ecommerce|gaming|legal|logistics|travel|cloud|crypto|startup|tech|"
    r"health|finance|seo|developer|ai))+)"
    r"(?:\s+(?:categories?|topics?|niches?))?\b",
    re.IGNORECASE,
)
_TOPIC_EXCLUDE_RE = re.compile(
    r"\b(?:not|no|without|exclude)\s+(?P<topic>saas|fintech|edtech|devtools|"
    r"healthcare|cybersecurity|ecommerce|gaming|legal|logistics|travel|cloud|"
    r"crypto|startup|tech|health|finance|seo|developer)\b"
    r"(?!\s+(?:vibe|keyword)s?\b)",  # vibe/keyword -> keyword_contains_exclude elsewhere
    re.IGNORECASE,
)
# "exclude X keyword" / "skip anything gaming" -> keyword_contains_exclude.
# Do not match "skip the …" determiner noise.
_KEYWORD_EXCLUDE_EXPLICIT_RE = re.compile(
    r"\b(?:exclude|excluding)\s+(?P<kw>[a-z][a-z0-9]{1,24})\s+keywords?\b"
    r"|\bskip\s+anything\s+(?P<kw2>[a-z][a-z0-9]{1,24})\b"
    r"|\bwithout\s+(?P<kw3>[a-z][a-z0-9]{1,24})\b"
    r"|\bbut\s+not\s+(?P<kw4>[a-z][a-z0-9]{1,24})\b",
    re.IGNORECASE,
)
# "want tech" / "ai gem" / "fintech com" / "saas topic" / "healthcare category"
_TOPICISH_WANT_RE = re.compile(
    r"\bwant\s+(?P<topic>tech|ai|seo|developer|saas|fintech|cloud|startup|health)\b"
    r"|\b(?P<ai_compound>ai)\s+(?:gem|inventory|agent|meets|startup|and|company|"
    r"product|startp|names?)\b"
    r"|\b(?P<topic2>saas|fintech|edtech|devtools|healthcare|cybersecurity|ecommerce|"
    r"gaming|legal|logistics|travel|cloud|crypto|health|finance|"
    r"developer|healthtech|real\s+estate|b2b\s+saas|climate\s+tech)"
    r"\s+(?:com|io|app|dev|topic|category|brand|space|names?|domains?|"
    r"listings?|cheap|under)\b"
    # Bare "tech domain" only when not "legal/climate tech".
    r"|(?<!legal\s)(?<!climate\s)\b(?P<topic3>tech)\s+(?:domains?|names?)\b"
    # "in tech space" / "tech brand" / "for tech startup"
    r"|\bin\s+(?P<topic4>tech)\s+space\b"
    r"|\b(?P<topic5>tech)\s+(?:brand|startup)\b"
    r"|\bfor\s+(?P<topic6>tech)\s+startup\b",
    re.IGNORECASE,
)
_BARE_TOPIC_BEFORE_TLD_RE = re.compile(
    r"\b(?P<topic>saas|fintech|edtech|devtools|healthcare|cybersecurity|"
    r"ecommerce|climate|gaming|legal|logistics|travel|cloud|startup|crypto|ai|health)\s+\.",
    re.IGNORECASE,
)
_STARTUP_TOPIC_RE = re.compile(
    r"\b(?:yc\s+)?startup\s+(?:type\s+)?(?:domain|domains|name|names|sounding|brands?)\b"
    r"|\bstartup\s+type\b"
    r"|\bstartup\s+sounding\b"
    r"|\bpremium\s+startup\b",
    re.IGNORECASE,
)
# Bare niche token with inventory/browse cue (typos normalized upstream).
# Avoid "good for seo", "my startup", "X startup domain" (startup=modifier).
_BARE_TOPIC_LOOSE_RE = re.compile(
    r"\b(?P<topic>b2b\s+saas|climate\s+tech|saas|fintech|edtech|devtools|healthcare|"
    r"cybersecurity|ecommerce|gaming|legal|logistics|travel|cloud|crypto|"
    r"health|finance|developer|healthtech|brandables)"
    r"(?:\s+(?:need|listings?|cheap|category|names?|domains?|under|com|io|app|"
    r"no\s+more|at\s+most)|$)",
    re.IGNORECASE,
)
# "logistics saas domain" -> both niche tokens. Skip soft "legal tech trustworthy".
_COMPOUND_TOPIC_RE = re.compile(
    r"\b(?P<a>logistics|travel|gaming|health|cloud|b2b)\s+"
    r"(?P<b>saas|tech|fintech|startup)\s+(?:domains?|names?)\b",
    re.IGNORECASE,
)
# Common surface typos -> canonical tokens (applied before slot parse).
_QUERY_TYPO_SUBS: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r"\bfintec\b", re.IGNORECASE), "fintech"),
    (re.compile(r"\bdevtols\b", re.IGNORECASE), "devtools"),
    (re.compile(r"\bcybersecuirty\b", re.IGNORECASE), "cybersecurity"),
    (re.compile(r"\bhealthcar\b", re.IGNORECASE), "healthcare"),
    (re.compile(r"\bstartp\b", re.IGNORECASE), "startup"),
    (re.compile(r"\bdomians\b", re.IGNORECASE), "domains"),
    (re.compile(r"\bdoamins\b", re.IGNORECASE), "domains"),
    (re.compile(r"\bdoamin\b", re.IGNORECASE), "domain"),
    (re.compile(r"\bdomian\b", re.IGNORECASE), "domain"),
    (re.compile(r"\bexpird\b", re.IGNORECASE), "expired"),
    (re.compile(r"\bgemm?\b", re.IGNORECASE), "gem"),
    (re.compile(r"\bbrandible\b", re.IGNORECASE), "brandable"),
    (re.compile(r"\bbrandng\b", re.IGNORECASE), "branding"),
    (re.compile(r"\bleters\b", re.IGNORECASE), "letters"),
    (re.compile(r"\bnumbrs\b", re.IGNORECASE), "numbers"),
    (re.compile(r"\bhyphn\b", re.IGNORECASE), "hyphen"),
    (re.compile(r"\btrafic\b", re.IGNORECASE), "traffic"),
    (re.compile(r"\btraffik\b", re.IGNORECASE), "traffic"),
    (re.compile(r"\bcheep\b", re.IGNORECASE), "cheap"),
    (re.compile(r"\blistngs\b", re.IGNORECASE), "listings"),
    (re.compile(r"\btrendin\b", re.IGNORECASE), "trending"),
    # Comparator / filler truncations (class stems).
    (re.compile(r"\babov\b", re.IGNORECASE), "above"),
    (re.compile(r"\bbelo\b", re.IGNORECASE), "below"),
    (re.compile(r"\bwort\b", re.IGNORECASE), "worth"),
    (re.compile(r"\bgoo+d\b", re.IGNORECASE), "good"),
    (re.compile(r"\bwit\b", re.IGNORECASE), "with"),
    (re.compile(r"\byrs\b", re.IGNORECASE), "years"),
)
# Listing recency -> days_listed_max / startTimeAfter.
# Browse fluff ("right now"/"lately"/"available now") is NOT listing age.
_FRESH_LISTING_RE = re.compile(
    r"\bfresh\s+last\s+(?P<hours2>\d+)\s+hours?\b"
    r"|\blast\s+(?P<hours>\d+)\s+hours?\b"
    r"|\bfresh\s+listings?\b"
    r"|\bfresh\s+(?:this\s+)?(?:morning|afternoon|evening)\b"
    r"|\b(?:just\s+listed|new\s+today|listed\s+today|listings?\s+today|"
    r"closeout\s+listings?\s+today|available\s+today|domains?\s+today|"
    r"what\s+dropped\s+today|since\s+yesterday|this\s+morning|this\s+afternoon)\b"
    r"|\brecently\s+(?:added|listed)\b|\b(?:added|listed)\s+recently\b"
    r"|\bnew\s+arrivals\b",
    re.IGNORECASE,
)
_FRESH_7D_RE = re.compile(
    r"\b(?:fresh\s+listings?|recently\s+(?:added|listed)|(?:added|listed)\s+recently|"
    r"new\s+arrivals)\b",
    re.IGNORECASE,
)
# Traffic typos as inventory cues (not advisory "does trafic help…").
_TRAFFIC_TYPO_RE = re.compile(
    r"\b(?:with|has|have|and)\s+trafic\b|\btrafic\s+(?:data|signal|history)\b"
    r"|\b(?:with|has|have|and)\s+traffik\b",
    re.IGNORECASE,
)
# Soft/strong advisory + inventory-bound detectors live in advisory_patterns
# (shared with L0 LLM scrub + post-merge reconcile). Aliases keep call sites stable.


# Boolean slot -> canonical signal phrases (data-driven: extend table to add new boolean slots).
_BOOLEAN_SLOT_SIGNALS: Dict[str, List[str]] = {
    "isBidAccepted": [
        "bid accepted",
        "accepted bid",
        "accepted offer",
        "bid interest",
        "active bid interest",
        "active bidding",
    ],
    "isExtended": [
        "extended auction",
        "extended bidding",
        "auction extended",
        "extended ai auction",
        "extended .ai auction",
        "auto extended",
        "auto-extended",
        "automatically extended",
        "last minute extension",
        "last min extension",
        "extension triggered",
        "time extended",
        "premium extension",
        "short premium extension",
    ],
    "has_reserve_price": [
        "has reserve price",
        "with reserve",
        "reserve price set",
        "reserve set",
        "reserve auction",
        "reserve auctions",
        "has reserve",
    ],
    "gd_transfer": ["gd transfer", "godaddy transfer"],
    "has_web_traffic_signal": [
        "real traffic",
        "existing traffic",
        "already has traffic",
        "has traffic",
        "real visitors",
        "real visitor",
        "existing visitors",
        "gets visitors",
        "has visitors",
        "already has visitors",
        "has existing audience",
        "receives traffic",
        "with traffic",
        "with web traffic",
        "web traffic",
        "organic traffic",
        "with traffic data",
        "proven traffic",
        "actual traffic",
        "some traffic",
        "established traffic",
        "traffic history",
        "real demand",
        "shows real demand",
        "shows demand",
        "decent traffic",
        "traffic rich",
        "getting visitors",
        "still getting visitors",
        "active websites",
    ],
    "traffic_is_unknown": [
        "traffic unknown",
        "unknown traffic",
        "no prior traffic",
        "no traffic history",
        "traffic not available",
        "traffic is unknown",
        "no traffic data",
        "no traffic data needed",
        "without traffic data",
        "traffic doesnt matter",
        "traffic doesn't matter",
        "traffic does not matter",
    ],
    "domain_age_is_unknown": [
        "age unknown",
        "unknown age",
        "domain age unknown",
        "age not available",
        "age is unknown",
        "fresh slate",
        "young or unknown age",
        "young or unknown",
    ],
    "price_below_market": [
        "below market",
        "below market price",
        "below market value",
        "underpriced",
        "priced below market",
        "value pick",
        "below market rate",
        "hidden gem",
        "hidden gems",
        "sleeper",
        "sleeper picks",
        "undervalued",
        "flip potential",
        "gems only",
        "gem only",
        "gem domains",
        "gem domain",
        "fraction of what",
        "selling for a fraction",
        "cheaper than worth",
        "but cheaper",
        "cheaper alternative",
        "priced way below",
        "way below what",
        "flip later",
        "resale value",
        "strong resale",
        "under budget",
        "good deal",
        "worth more in future",
        # Value-vs-market lexicon (generic speech class, not suite literals).
        "good value compared",
        "compared to market",
        "overlooked",
        "under radar",
        "under the radar",
        "bargain",
    ],
    "buy_it_now": [
        "buy it now",
        "buy now option",
        "has bin",
        "bin available",
        "bin price",
        "buy-it-now",
        "buy now",
        "buy-now",
        "buyout available",
        "buynow",
        "buy now listings",
        "buynow listings",
        # Synonyms without the literal "buy now" bigram (instant/immediate buy).
        "instant buy",
        "immediate buy",
        "instant purchase",
        "immediate purchase",
        "buy instantly",
        "buy immediately",
        "instant bin",
        "immediate bin",
        "instant buyout",
        "immediate buyout",
        # "want to just buy it not bid" / "dont want to bid just buy"
        "just buy it not bid",
        "buy it not bid",
        "not bid just buy",
        "dont want to bid",
        "don't want to bid",
        "do not want to bid",
        "want to just buy",
        "buy not bid",
        "no bidding just buy",
        # Soft BIN inventory intent.
        "available to buy right now",
        "want to buy something",
        "nobody bought yet",
        "no one bought yet",
    ],
}

# Explicit FALSE polarity for slots where absence is an active filter (not inactive).
_BOOLEAN_FALSE_SLOT_SIGNALS: Dict[str, List[str]] = {
    "has_reserve_price": [
        "no reserve",
        "without reserve",
        "no reserve price",
        "without a reserve",
        "without reserve price",
        "reserve free",
        "no-reserve",
        "non reserve",
        "no reserve auction",
        "no reserve auctions",
    ],
}

# Maps recency alias (single-space-normalized lowercase) -> days_listed_max value.
# _RECENCY_ALIAS_RE is built from these keys (longest first). Add an alias here for new surface forms.
_RECENCY_ALIAS_DAYS: Dict[str, int] = {
    "this month": 30,
    "last month": 30,
    "this week": 7,
    "last week": 7,
    "recently": 7,
    "today": 1,
}
_RECENCY_ALIAS_RE = re.compile(
    r"\b(?:listed|added|posted|new(?:ly)?\s+(?:listed|added)|just\s+(?:listed|added)|recently\s+(?:listed|added|posted))\s+"
    r"(?P<alias>"
    + "|".join(re.escape(k) for k in sorted(_RECENCY_ALIAS_DAYS, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)
_RECENCY_NUMERIC_RE = re.compile(
    r"\b(?:listed|added|posted|new(?:ly)?\s+(?:listed|added)|recently\s+(?:listed|added|posted)|fresh\s+listings?)"
    r"\s+(?:in\s+(?:the\s+)?)?last\s+(?P<num>\d+)\s+days?\b",
    re.IGNORECASE,
)
# Implicit recency: "fresh listings", "new listings", "recently listed" with no explicit duration -> 7 days.
_RECENCY_IMPLICIT_RE = re.compile(
    r"\b(?:fresh|recent)\s+listings?\b|\brecently\s+(?:listed|added|posted)\b|\bnewly\s+added\b",
    re.IGNORECASE,
)
# LLM: days_listed_min — 'stale over 2 weeks' / 'listed more than 10 days ago' (NOT minAge/minPrice)
_STALE_LISTED_RE = re.compile(
    r"\bstale\s+(?:over|more\s+than)\s+(?P<num>\d+)\s+(?P<unit>days?|weeks?)(?:\s+old)?\b"
    r"|\b(?:listed|added)\s+more\s+than\s+(?P<num2>\d+)\s+(?P<unit2>days?|weeks?)\s+ago\b",
    re.IGNORECASE,
)
# LLM: ownerMemberIncludeList / ownerMemberExcludeList
_OWNER_INCLUDE_RE = re.compile(
    r"\b(?:from\s+(?:seller|member|user|owner)|listed\s+by(?:\s+(?:member|seller|user))?|"
    r"by\s+(?:owner|seller|member|user))\s+['\"]?(?P<id>[a-z0-9][a-z0-9_\-.]{1,63})\b",
    re.IGNORECASE,
)
_OWNER_EXCLUDE_RE = re.compile(
    r"\b(?:not\s+from|exclude\s+(?:seller|member|user|owner)|excluding\s+(?:seller|member|user))\s+"
    r"['\"]?(?P<id>[a-z0-9][a-z0-9_\-.]{1,63})\b",
    re.IGNORECASE,
)
# LLM exclusive polarity (bids + maxSldLen + price under/below — parity with L0 rule 2).
_EXCLUSIVE_MAX_CMP: FrozenSet[str] = frozenset(
    {
        "under",
        "below",
        "less than",
        "fewer than",
        "shorter than",
    }
)
_EXCLUSIVE_MIN_CMP: FrozenSet[str] = frozenset(
    {
        "more than",
        "greater than",
        "over",
        "above",
    }
)
_UNKNOWN_AGE_CUE_RE = re.compile(r"\b(?:unknown\s+age|age\s+unknown)\b", re.IGNORECASE)

# Postfix-only comparator phrases (appear AFTER the number, not before).
_POSTFIX_CMP = (
    r"or\s+more|or\s+fewer|or\s+less|and\s+above|and\s+below|and\s+over|and\s+under|"
    r"plus|minimum|min|maximum|max"
)
_POSTFIX_CMP_FOLLOWS_RE = re.compile(
    r"^\s+(?:" + _POSTFIX_CMP + r")\b",
    re.IGNORECASE,
)
# Postfix form: "10 bids or more" (num unit cmp) or "10 or more bids" (num cmp unit).
_FORM_C_RE = re.compile(
    r"\b(?P<num>\d[\d,]*)\s+"
    r"(?:(?P<unit1>" + _UNIT + r")\s+(?P<cmp1>" + _POSTFIX_CMP + r")"
    r"|(?P<cmp2>" + _POSTFIX_CMP + r")\s+(?P<unit2>" + _UNIT + r"))\b",
    re.IGNORECASE,
)


class EntityExtractor(Protocol):
    """Async L0 slot filler: normalized query text in, hybrid IntentSlice or None out."""

    async def classify_async(self, query: str) -> Optional[IntentSlice]: ...


def _parse_num_token(raw: str) -> Tuple[float, bool]:
    """Parse digit token -> (value, had_k_or_m_suffix).

    '2k'/'2K' -> (2000, True); '2,000' -> (2000, False);
    '2 thousand' -> (2000, True); '2 million' -> (2_000_000, True).
    """
    s = raw.replace(",", "").strip().lower()
    mult = 1.0
    scaled = False
    for word, factor in (
        ("thousand", 1_000.0),
        ("million", 1_000_000.0),
        ("billion", 1_000_000_000.0),
    ):
        if s.endswith(word):
            scaled = True
            mult = factor
            s = s[: -len(word)].strip()
            break
    if not scaled and s and s[-1] in "km":
        scaled = True
        mult = 1_000.0 if s[-1] == "k" else 1_000_000.0
        s = s[:-1]
    return float(s) * mult, scaled


def _to_number(raw: str) -> float:
    """Parse digit token with optional thousands commas and k/m suffix to float."""
    return _parse_num_token(raw)[0]


def _time_unit_key(unit: str) -> str:
    """Map a plural/abbreviated relative-time unit token to its _TIME_UNIT_SECONDS key."""
    u = unit.lower().rstrip("s")
    return "min" if u in ("min", "minute") else ("hr" if u in ("hr", "hour") else u)


# Age-flavored comparators: bare "younger/older than N" -> domain_age, not price.
_AGE_COMPARATORS: FrozenSet[str] = frozenset(
    {"younger than", "newer than", "older than"}
)


_BUDGET_STEAL_FAMILIES: FrozenSet[str] = frozenset(
    {
        "semrush_backlinks",
        "majestic_backlinks",
        "semrush_ref_domains",
        "majestic_ref_domains",
        "traffic",
        "semrush_authority",
        "majestic_tf",
        "majestic_cf",
        "semrush_indexed_pages",
        "semrush_search_volume",
        # "decent cpc under 2k" -> price ceiling (CPC quality cue, not CPC=2000).
        "semrush_cpc",
        # "min 9 chars under 300" -> under 300 is price, not maxSldLen=300.
        "name_length",
        # "dev ext count high under 1500" -> price ceiling (qualitative ext-count
        # cue, not maxEstibotExtCount=1500). Small under-N (<100) still binds the
        # metric via the final number>=100 gate in _price_if_budget_under.
        "estibot_ext_count",
        # Bare govalue/appraisal word + solitary "under N" -> price ceiling +
        # existence floor (minValuationPrice:1), not maxValuationPrice=N-1.
        # Two-number rows ("appraisal over 5000 asking under 1k") bind via the
        # min direction ("over") and never hit this max-side steal.
        "govalue",
    }
)


def _price_if_budget_under(
    family: Optional[str],
    number: float,
    cmp: Optional[str],
    *,
    scaled: bool = False,
    unit: Optional[str] = None,
    prefix: Optional[str] = None,
) -> Optional[str]:
    """Rewrite SEO/traffic 'under Nk' / 'under N>=100' as price (budget ceiling).

    "strong backlinks under 2k" is a price ceiling. Explicit traffic units
    ("visitors"/"hits"/"traffic") keep the traffic family at any scale — except
    soft presence phrasing ("getting visitors under N") which is a price budget.
    """
    if family is None or family not in _BUDGET_STEAL_FAMILIES:
        return family
    if not cmp:
        return family
    cmp_norm = re.sub(r"\s+", " ", cmp.lower())
    if cmp_norm not in (
        "under",
        "below",
        "less than",
        "at most",
        "no more than",
        "up to",
    ):
        return family
    unit_stem = (unit or "").lower().rstrip("s")
    unit_raw = (unit or "").lower()
    pref = (prefix or "").lower()
    # Soft presence + visitors/traffic + under-N -> price (not maxTraffic).
    if (
        family == "traffic"
        and (scaled or number >= 100)
        and re.search(
            r"\b(?:getting|still\s+getting|has|have|with)\s+"
            r"(?:visitors?|traffic|hits?)\s*$"
            r"|\b(?:getting|still\s+getting)\s*$",
            pref,
        )
    ):
        return "price"
    # Qualified traffic ceilings ("steady/real/monthly traffic under N") stay traffic.
    if family == "traffic" and re.search(
        r"\b(?:steady|real|existing|monthly|organic|good|decent|some)\s+"
        r"(?:web\s+)?traffic\b",
        f"{pref} {unit_raw}",
        re.I,
    ):
        return family
    # Explicit visitor/hit ceilings stay traffic. Bare leading "traffic under Nk"
    # is a price budget (auction convention) — do not protect on unit=traffic alone.
    if family == "traffic" and (
        unit_stem in ("visitor", "visit", "hit", "pageview")
        or re.search(r"\b(?:visitors?|hits?|pageviews?)\b", unit_raw, re.I)
    ):
        return family
    if family == "semrush_search_volume" and re.search(
        r"\b(?:search\s*volumes?|monthly\s+searches?|searches?)\b",
        unit_raw,
        re.I,
    ):
        return family
    if family == "semrush_indexed_pages" and re.search(
        r"\b(?:indexed\s+pages?|pages?\s+indexed|index(?:ed)?)\b",
        unit_raw,
        re.I,
    ):
        return family
    if scaled or number >= 100:
        return "price"
    return family


def _resolve_family(
    cur: Optional[str],
    curword: Optional[str],
    unit: Optional[str],
    cmp: Optional[str] = None,
) -> Optional[str]:
    """Return the metric family for a numeric match.

    Priority: currency -> price; trailing unit noun -> that family; bare comparator+number
    defaults to price (domain-auction convention: "under 500" = price_max) unless the
    comparator is age-flavored ("younger than 5" -> domain_age).
    """
    if cur or curword:
        return "price"
    if unit:
        unit_l = unit.lower()
        # "monthly hits" / "monthly visitors" -> traffic (strip period adjective).
        unit_l = re.sub(r"^(?:monthly|daily|weekly)\s+", "", unit_l)
        return _UNIT_FAMILY.get(unit_l) or _UNIT_FAMILY.get(unit.lower())
    if cmp:
        cmp_norm = re.sub(r"\s+", " ", cmp.lower())
        if cmp_norm in _AGE_COMPARATORS:
            return "domain_age"
        return "price"
    return None


def _add_slot(slots: Dict[str, object], name: str, value: object) -> None:
    """Accumulate a slot value: append-unique for list slots, first-seen-wins for scalars."""
    if name in _LIST_SLOTS:
        bucket = slots.setdefault(name, [])
        if isinstance(bucket, list) and value not in bucket:
            bucket.append(value)
    elif name not in slots:
        slots[name] = value


# Families where "under N" is exclusive (N-1) even without a k/m suffix.
# domain_age is intentionally inclusive (under 3 years -> maxAge=3) — L0 grounding.
_EXCLUSIVE_MAX_FAMILIES: FrozenSet[str] = frozenset(
    {
        "bids",
        "name_length",
        "traffic",
        "govalue",
        "estibot_domain_count",
        "estibot_domain_count_dev",
        "estibot_ext_count",
        "estibot_ext_count_dev",
    }
)

# Families where "above/over N" stays inclusive (N, not N+1) on the min side.
# Evidence: "over 2000 monthly hits" -> minUniqueSearches:2000 (not 2001);
# "dev ext saturation above 15" -> minEstibotDomainCountDev:15 (not 16).
_INCLUSIVE_MIN_FAMILIES: FrozenSet[str] = frozenset(
    {
        "unique_searches",
        "estibot_domain_count_dev",
    }
)


def _adjust_exclusive(
    family: str,
    direction: str,
    number: float,
    cmp_norm: Optional[str],
    *,
    scaled: bool = False,
) -> float:
    """Apply LLM exclusive polarity: under N -> N-1; above/more than N -> N+1.

    Price ceilings: under/below/less than -> N-1 for all scales (incl. under 1k->999,
    under 500->499) — matches L0 prompt rule 2 / grounding. Inclusive ceilings use
    at most / no more than / capped at (not in ``_EXCLUSIVE_MAX_CMP``).
    """
    del scaled  # kept for call-site compat; price exclusivity no longer scale-gated
    if not cmp_norm:
        return number
    if direction == "max" and cmp_norm in _EXCLUSIVE_MAX_CMP:
        if family in _EXCLUSIVE_MAX_FAMILIES or family == "price":
            return max(0.0, number - 1.0)
    # "above/over/more than N" -> N+1. "N plus" / "at least N" stay inclusive.
    # "nothing over N" is a max comparator — never treat bare "over" inside it as min.
    if (
        direction == "min"
        and family not in _INCLUSIVE_MIN_FAMILIES
        and cmp_norm
        in (
            "above",
            "over",
            "more than",
            "greater than",
        )
    ):
        return number + 1.0
    return number


def _emit_bound(
    slots: Dict[str, object],
    family: str,
    direction: str,
    number: float,
    cmp: Optional[str] = None,
    unit: Optional[str] = None,
    *,
    scaled: bool = False,
    num_raw: Optional[str] = None,
) -> None:
    """Write one numeric bound (or an exact pair for direction 'exact') for a metric family.

    LLM prompt rules mirrored here:
    - 'minimum N letters' / floor + letters -> minLetters (not name_length)
    - exclusive bids / maxSldLen via comparator polarity
    """
    unit_stem = (unit or "").lower().rstrip("s")
    # LLM: minLetters for alphabetic letter COUNT floor cues — not minSldLen.
    if family == "name_length" and direction == "min" and unit_stem == "letter":
        _add_slot(slots, "minLetters", int(number))
        return
    cmp_norm = re.sub(r"\s+", " ", cmp.lower()) if cmp else None
    had_scale = scaled
    if num_raw is not None:
        had_scale = _parse_num_token(num_raw)[1]
    adjusted = _adjust_exclusive(
        family,
        direction,
        number,
        cmp_norm,
        scaled=had_scale,
    )
    min_slot, max_slot, is_float, _requires = _FAMILY_SPEC[family]
    value: object = float(adjusted) if is_float else int(adjusted)
    if direction == "min":
        _add_slot(slots, min_slot, value)
    elif direction == "max":
        _add_slot(slots, max_slot, value)
    else:
        _add_slot(slots, min_slot, value)
        _add_slot(slots, max_slot, value)


def _emit_numeric_list(
    slots: Dict[str, object],
    family: str,
    direction: str,
    nums: List[float],
    cmp: Optional[str] = None,
    unit: Optional[str] = None,
) -> None:
    """Collapse coordinated numbers into one bound pair for a family.

    Alternatives resolve to the loosest bound: a 'max' comparator keeps max(nums),
    'min' keeps min(nums); a bare or exact list spans [min(nums), max(nums)].
    """
    if not nums:
        return
    lo, hi = min(nums), max(nums)
    if direction == "min":
        _emit_bound(slots, family, "min", lo, cmp=cmp, unit=unit)
    elif direction == "max":
        _emit_bound(slots, family, "max", hi, cmp=cmp, unit=unit)
    else:
        _emit_bound(slots, family, "min", lo, cmp=cmp, unit=unit)
        _emit_bound(slots, family, "max", hi, cmp=cmp, unit=unit)


def _parse_numeric(text: str, slots: Dict[str, object]) -> None:
    """Extract every comparator/currency/unit-anchored numeric constraint into slots."""
    consumed: List[Tuple[int, int]] = []
    # LLM: stale listing-age must not become minPrice/minAge from 'over N weeks'.
    for m in _STALE_LISTED_RE.finditer(text):
        consumed.append((m.start(), m.end()))
    # LLM: 'min N unique searches' must not also become price_min from bare min N.
    for m in _UNIQUE_SEARCHES_BOUND_RE.finditer(text):
        consumed.append((m.start(), m.end()))
    # Monthly-visitor cues owned by unique_searches; block generic Form-A traffic parse.
    # Exception: postfix-cmp variant ("5000 monthly visitors minimum") is handled by FORM_C.
    for m in _MONTHLY_VISITORS_RE.finditer(text):
        if not _POSTFIX_CMP_FOLLOWS_RE.match(text[m.end() :]):
            consumed.append((m.start(), m.end()))
    unknown_age = bool(_UNKNOWN_AGE_CUE_RE.search(text))
    starting_bid_price = bool(_STARTING_BID_PRICE_RE.search(text))
    bid_count_cue = bool(_BID_COUNT_CUE_RE.search(text)) and not starting_bid_price

    def _overlaps(start: int, end: int) -> bool:
        return any(cs <= start < ce or cs < end <= ce for cs, ce in consumed)

    def _price_if_starting_bid(family: Optional[str]) -> Optional[str]:
        if family == "bids" and starting_bid_price:
            return "price"
        return family

    def _bids_if_count_cue(family: Optional[str]) -> Optional[str]:
        if bid_count_cue and family in (None, "price"):
            return "bids"
        return family

    for m in _BETWEEN_RE.finditer(text):
        if _overlaps(m.start(), m.end()):
            continue
        # "tf between 20 and 50" is metric range, not price.
        prefix = text[max(0, m.start() - 48) : m.start()].lower()
        if re.search(
            r"\b(?:citation\s+flow|trust\s+flow|authority|backlinks?|bids?|"
            r"years?|age|traffic|visitors?|govalue|valuation|tf|cf|"
            r"search\s+volume|ref\s+domains?|letters?|chars?|characters?)\s*$",
            prefix,
        ):
            continue
        family = _resolve_family(
            m.group("cur1") or m.group("cur2"), None, m.group("unit")
        )
        # Bare "between 100 and 500" (no currency/unit) -> price in domain-auction queries.
        if family is None and not m.group("unit"):
            family = "price"
        if family is None:
            continue
        unit = m.group("unit")
        _emit_bound(slots, family, "min", _to_number(m.group("lo")), unit=unit)
        _emit_bound(slots, family, "max", _to_number(m.group("hi")), unit=unit)
        consumed.append((m.start(), m.end()))
    for m in _RANGE_TO_RE.finditer(text):
        if _overlaps(m.start(), m.end()):
            continue
        # Skip metric ranges already owned by a leading metric noun
        # ("citation flow 20 to 40" ≠ price).
        prefix = text[max(0, m.start() - 48) : m.start()].lower()
        if re.search(
            r"\b(?:citation\s+flow|trust\s+flow|authority|backlinks?|bids?|"
            r"years?|age|traffic|visitors?|govalue|valuation|tf|cf|"
            r"search\s+volume|ref\s+domains?|letters?|chars?|characters?)\s*$",
            prefix,
        ):
            continue
        family = _resolve_family(None, None, m.group("unit"))
        if family is None and not m.group("unit"):
            # Require price/currency/range cue — bare "20 to 40" alone is too ambiguous.
            window = text[max(0, m.start() - 24) : m.end() + 12].lower()
            if not re.search(
                r"\b(?:price|budget|cost|range|usd|eur|gbp|cad|aud|"
                r"dollars?|euros?|pounds?)\b|" + _CUR,
                window,
            ):
                continue
            family = "price"
        if family is None:
            continue
        lo, hi = _to_number(m.group("lo")), _to_number(m.group("hi"))
        if hi < lo:
            continue
        unit = m.group("unit")
        _emit_bound(slots, family, "min", lo, unit=unit)
        _emit_bound(slots, family, "max", hi, unit=unit)
        consumed.append((m.start(), m.end()))
    for m in _MEASURE_PREFIX_RE.finditer(text):
        if _overlaps(m.start(), m.end()):
            continue
        family = _MEASURE_FAMILY.get(m.group("measure").lower().rstrip("s"))
        if family is None:
            continue
        scope = re.sub(r"[\s-]+", "", (m.group("scope") or "").lower())
        if family == "name_length" and scope in ("tld", "toplevel"):
            continue
        cmp_raw = m.group("cmp")
        nums = [_to_number(x) for x in _NUM_RE.findall(m.group("nums"))]
        # "chars under 300" is a price ceiling, not maxSldLen=299.
        if nums:
            family = (
                _price_if_budget_under(
                    family,
                    max(nums),
                    cmp_raw,
                    scaled=bool(re.search(r"[kKmM]", m.group("nums") or "")),
                    unit=m.group("measure"),
                    prefix=text[max(0, m.start() - 40) : m.start()],
                )
                or family
            )
        direction = (
            _COMPARATOR_DIRECTION.get(re.sub(r"\s+", " ", cmp_raw.lower()))
            if cmp_raw
            else "exact"
        )
        if direction is None:
            continue
        unit = m.group("measure") if family != "price" else None
        _emit_numeric_list(
            slots,
            family,
            direction,
            nums,
            cmp=cmp_raw,
            unit=unit,
        )
        consumed.append((m.start(), m.end()))
    for m in _FORM_LIST_RE.finditer(text):
        if _overlaps(m.start(), m.end()):
            continue
        family = _resolve_family(None, None, m.group("unit"))
        if family is None:
            continue
        nums = [_to_number(x) for x in _NUM_RE.findall(m.group("nums"))]
        if len(nums) < 2:
            continue
        cmp_raw = m.group("cmp")
        if cmp_raw:
            direction = _COMPARATOR_DIRECTION.get(re.sub(r"\s+", " ", cmp_raw.lower()))
        elif not _FAMILY_SPEC[family][3]:
            direction = "exact"
        else:
            continue
        if direction is None:
            continue
        _emit_numeric_list(
            slots, family, direction, nums, cmp=cmp_raw, unit=m.group("unit")
        )
        consumed.append((m.start(), m.end()))
    for m in _FORM_B_RE.finditer(text):
        if _overlaps(m.start(), m.end()):
            continue
        metric = re.sub(r"\s+", " ", m.group("metric").lower())
        family_orig = _price_if_starting_bid(_METRIC_FAMILY.get(metric))
        if family_orig is None:
            continue
        # LLM: 'unknown age under 500' -> maxPrice only — never bind age metric here.
        if family_orig == "domain_age" and unknown_age:
            continue
        # Prefer leading cmp ("budget under 1500"); else trailing ("budget 1500 max").
        cmp_raw = m.group("cmp") or m.group("trail_cmp")
        num_val, scaled = _parse_num_token(m.group("num"))
        # Soft "barely any <metric> under N" (N small) -> soft floor later; skip tiny max.
        if (
            family_orig in ("estibot_ext_count", "estibot_ext_count_dev")
            and cmp_raw
            and re.sub(r"\s+", " ", cmp_raw.lower()) in ("under", "below", "less than")
            and not scaled
            and num_val < 50
            and re.search(
                r"\bbarely\s+any\b", text[max(0, m.start() - 24) : m.start()], re.I
            )
        ):
            continue
        family = _price_if_budget_under(
            family_orig,
            num_val,
            cmp_raw,
            scaled=scaled,
            unit=metric,
            prefix=text[max(0, m.start() - 40) : m.start()],
        )
        if cmp_raw:
            direction = _COMPARATOR_DIRECTION.get(re.sub(r"\s+", " ", cmp_raw.lower()))
        elif metric in _CEILING_METRICS:
            # Bare "budget 1500" / "cost 200" = ceiling (include $0 listings).
            direction = "max"
        elif not _FAMILY_SPEC[family][3]:
            direction = "exact"
        else:
            direction = "min"
        if direction is None:
            continue
        _emit_bound(
            slots,
            family,
            direction,
            num_val,
            cmp=cmp_raw,
            num_raw=m.group("num"),
        )
        # Budget-steal ("indexed pages under 2k" -> price): consume only the under-N
        # tail so "100 plus indexed pages" Form C can still bind the metric floor.
        if family == "price" and family_orig != "price" and cmp_raw:
            tail = re.search(
                r"\b(?:under|below|less\s+than|at\s+most|no\s+more\s+than|up\s+to)\s+"
                r"\$?\d[\d,]*(?:[kKmM])?\b",
                m.group(0),
                re.IGNORECASE,
            )
            if tail:
                consumed.append((m.start() + tail.start(), m.start() + tail.end()))
            else:
                consumed.append((m.start(), m.end()))
        else:
            consumed.append((m.start(), m.end()))
    for m in _FORM_A_RE.finditer(text):
        if not any(m.group(g) for g in ("cmp", "cur", "curword", "unit")):
            continue
        if any(cs <= m.start() < ce or cs < m.end() <= ce for cs, ce in consumed):
            continue
        family = _bids_if_count_cue(
            _price_if_starting_bid(
                _resolve_family(
                    m.group("cur"), m.group("curword"), m.group("unit"), m.group("cmp")
                )
            )
        )
        if family is None:
            continue
        # LLM: with unknown-age cue, bare under N is price — never domain_age.
        if family == "domain_age" and unknown_age:
            continue
        cmp_raw = m.group("cmp")
        num_val, scaled = _parse_num_token(m.group("num"))
        # Bare comparator+N (<100, no unit/$/k) after estibot/dev-ext / soft-metric
        # lexicon -> bind that metric family (not price). Tiny under after "barely any"
        # is skipped (soft floor owns; later price-scale under owns budget).
        if (
            family == "price"
            and not m.group("unit")
            and not m.group("cur")
            and not scaled
            and num_val < 100
            and cmp_raw
        ):
            cmp_norm = re.sub(r"\s+", " ", cmp_raw.lower())
            prefix = text[max(0, m.start() - 48) : m.start()].lower()
            if cmp_norm in ("under", "below", "less than") and re.search(
                r"\bbarely\s+any\b",
                prefix,
            ):
                continue
            # Ext-count reclassification requires the cue phrase to sit
            # immediately against the number (short window, anchored to the
            # end of the prefix) — a loose 48-char unanchored scan wrongly
            # grabbed cues separated from the number by unrelated words
            # (e.g. "dev ext saturation is above 15" is a saturation metric,
            # not an ext-count bound; the bare substring "dev ext" inside it
            # should not hijack the classification).
            ext_cue_prefix = text[max(0, m.start() - 28) : m.start()].lower()
            if re.search(
                r"\b(?:estibot\s+dev\s+ext(?:ension)?|developed\s+ext(?:ension)?|"
                r"dev\s+ext(?:ension)?)(?:\s+count)?\s*$",
                ext_cue_prefix,
            ):
                family = "estibot_ext_count_dev"
            elif re.search(
                r"\b(?:estibot\s+)?ext(?:ension)?\s+count\s*$",
                ext_cue_prefix,
            ):
                family = "estibot_ext_count"
            elif (
                cmp_norm in ("under", "below", "less than")
                and num_val < 50
                and re.search(
                    r"\b(?:tf|cf|trust\s+flow|citation\s+flow|estibot)\b", prefix
                )
            ):
                continue
        family = _price_if_budget_under(
            family,
            num_val,
            cmp_raw,
            scaled=scaled,
            unit=m.group("unit"),
            prefix=text[max(0, m.start() - 40) : m.start()],
        )
        # "4 chars or less" — Form A may match before Form C; honor trailing postfix cmp.
        end_idx = m.end()
        if not cmp_raw:
            post = re.match(
                r"\s+(?P<pcmp>or\s+more|or\s+fewer|or\s+less|and\s+above|and\s+below|"
                r"and\s+over|and\s+under|plus|minimum|min|maximum|max)\b",
                text[m.end() :],
                re.IGNORECASE,
            )
            if post:
                cmp_raw = post.group("pcmp")
                end_idx = m.end() + post.end()
        if cmp_raw:
            direction = _COMPARATOR_DIRECTION.get(re.sub(r"\s+", " ", cmp_raw.lower()))
        elif not _FAMILY_SPEC[family][3]:
            direction = "exact"
        elif m.group("unit"):
            # Bare "200 referring domains" / "100 indexed pages" -> floor (LLM).
            direction = "min"
        else:
            continue
        if direction is None:
            continue
        _emit_bound(
            slots,
            family,
            direction,
            num_val,
            cmp=cmp_raw,
            unit=m.group("unit"),
            num_raw=m.group("num"),
        )
        consumed.append((m.start(), end_idx))
    # Second-pass: bare "under N" price when earlier metric spans ate the only under-cue.
    # Pass raw N + cmp=under — _adjust_exclusive applies N-1 (do NOT pre-subtract).
    _BARE_PRICE_SECOND_RE = re.compile(
        r"\b(?:under|below|less\s+than)\s+\$?(?P<num>\d[\d,]*(?:[kKmM])?)\b",
        re.IGNORECASE,
    )
    if "price_max" not in slots:
        for _m2 in _BARE_PRICE_SECOND_RE.finditer(text):
            if _overlaps(_m2.start(), _m2.end()):
                continue
            raw_n = _m2.group("num")
            n2 = _to_number(raw_n)
            # Prefer price-scale numbers (>=50 or k/m) so "under 5 bids" isn't re-bound.
            if n2 >= 50 or re.search(r"[kKmM]$", raw_n or ""):
                _emit_bound(slots, "price", "max", n2, cmp="under", num_raw=raw_n)
                break
    for m in _FORM_C_RE.finditer(text):
        if _overlaps(m.start(), m.end()):
            continue
        unit = m.group("unit1") or m.group("unit2")
        cmp_raw = m.group("cmp1") or m.group("cmp2")
        if not unit or not cmp_raw:
            continue
        unit_key = re.sub(r"^monthly\s+", "", unit.lower())
        family = _UNIT_FAMILY.get(unit_key) or _UNIT_FAMILY.get(unit.lower())
        if family is None:
            continue
        direction = _COMPARATOR_DIRECTION.get(re.sub(r"\s+", " ", cmp_raw.lower()))
        if direction is None:
            continue
        _emit_bound(
            slots,
            family,
            direction,
            _to_number(m.group("num")),
            cmp=cmp_raw,
            unit=unit_key,
            num_raw=m.group("num"),
        )
        consumed.append((m.start(), m.end()))


def _parse_spelled_measure(text: str, slots: Dict[str, object]) -> None:
    """Extract a spelled small quantity bound to a measure unit ("one word" -> word_count=1)."""
    for m in _SPELLED_MEASURE_RE.finditer(text):
        family = _UNIT_FAMILY.get(m.group("unit").lower())
        qty = _SPELLED_NUM.get(m.group("qty").lower())
        if family is None or qty is None:
            continue
        cmp_raw = m.group("cmp")
        direction = (
            _COMPARATOR_DIRECTION.get(re.sub(r"\s+", " ", cmp_raw.lower()))
            if cmp_raw
            else "exact"
        )
        if direction is None:
            continue
        _emit_bound(
            slots, family, direction, float(qty), cmp=cmp_raw, unit=m.group("unit")
        )


# Numeric-family authority: entity slot name -> metric family, inverted from _FAMILY_SPEC.
_SLOT_TO_FAMILY: Dict[str, str] = {}
for _fam, (_lo, _hi, _isf, _rq) in _FAMILY_SPEC.items():
    _SLOT_TO_FAMILY[_lo] = _fam
    _SLOT_TO_FAMILY[_hi] = _fam


def numeric_authority(query: str) -> Tuple[Dict[float, set], set]:
    """Map every unit/currency-anchored number in the query to the metric family it belongs to.

    Records ALL numeric mentions (not the collapsed bounds), so a coordinated list
    ("3 or 5 chars") contributes both 3 and 5 to name_length. Returns
    (value -> set(families), set(families present)). Used to detect misattribution:
    a number the query anchors to one family must not surface under a different one.
    """
    val_to_families: Dict[float, set] = {}
    families: set = set()

    def _add(family: Optional[str], nums: List[float]) -> None:
        if not family:
            return
        families.add(family)
        for n in nums:
            val_to_families.setdefault(float(n), set()).add(family)

    for m in _BETWEEN_RE.finditer(query):
        _add(
            _resolve_family(m.group("cur1") or m.group("cur2"), None, m.group("unit")),
            [_to_number(m.group("lo")), _to_number(m.group("hi"))],
        )
    for m in _MEASURE_PREFIX_RE.finditer(query):
        fam = _MEASURE_FAMILY.get(m.group("measure").lower().rstrip("s"))
        if fam == "name_length" and re.sub(
            r"[\s-]+", "", (m.group("scope") or "").lower()
        ) in ("tld", "toplevel"):
            continue
        _add(fam, [_to_number(x) for x in _NUM_RE.findall(m.group("nums"))])
    for m in _FORM_LIST_RE.finditer(query):
        _add(
            _resolve_family(None, None, m.group("unit")),
            [_to_number(x) for x in _NUM_RE.findall(m.group("nums"))],
        )
    # LLM: 'unknown age under N' -> price, not domain_age (same gate as _parse_numeric).
    unknown_age = bool(_UNKNOWN_AGE_CUE_RE.search(query))
    for m in _FORM_B_RE.finditer(query):
        family = _METRIC_FAMILY.get(re.sub(r"\s+", " ", m.group("metric").lower()))
        if family == "domain_age" and unknown_age:
            continue
        _add(family, [_to_number(m.group("num"))])
    for m in _FORM_A_RE.finditer(query):
        if not any(m.group(g) for g in ("cur", "curword", "unit")):
            continue
        family = _resolve_family(
            m.group("cur"), m.group("curword"), m.group("unit"), m.group("cmp")
        )
        if family == "domain_age" and unknown_age:
            continue
        _add(family, [_to_number(m.group("num"))])
    for m in _SPELLED_MEASURE_RE.finditer(query):
        fam = _UNIT_FAMILY.get(m.group("unit").lower())
        qty = _SPELLED_NUM.get(m.group("qty").lower())
        if fam is not None and qty is not None:
            _add(fam, [float(qty)])
    for m in _FORM_C_RE.finditer(query):
        unit = m.group("unit1") or m.group("unit2")
        if unit:
            _add(_UNIT_FAMILY.get(unit.lower()), [_to_number(m.group("num"))])
    for m in _SPELLED_NUMERIC_RE.finditer(query):
        fam = _UNIT_FAMILY.get(m.group("unit").lower())
        qty = _SPELLED_NUM.get(m.group("qty").lower())
        if fam is not None and qty is not None:
            _add(fam, [float(qty)])
    return val_to_families, families


# Alnum tokenizer for surface stripping; mirrors residual_extractor._TOKEN_RE so the
# tokens returned here subtract cleanly from the encode-text token stream.
_SURFACE_TOKEN_RE = re.compile(r"[a-z0-9]+")


def numeric_filter_surfaces(query: str) -> Set[str]:
    """Return the lowercase alnum tokens of every numeric/measure/comparator match.

    Reuses the same matchers (and the same family/unit guards) that emit numeric
    filter slots, so the returned set is exactly the surface a slotted numeric
    constraint consumed — comparator + coordinated numbers + unit
    ("under 3 or 5 chars" -> {"under","3","or","5","chars"}). The residual /
    encode-text builder subtracts these so no operator, filter number, or unit
    word reaches the dense leg. Keyword / topic / tld / auction surfaces are
    intentionally excluded: they carry concept signal (or are stripped elsewhere).

    :param query: str - Normalized query text
    :return: Set[str] - Lowercase alnum surface tokens to strip
    """
    tokens: Set[str] = set()

    def _collect(surface: str) -> None:
        tokens.update(_SURFACE_TOKEN_RE.findall(surface.lower()))

    for m in _BETWEEN_RE.finditer(query):
        if _resolve_family(m.group("cur1") or m.group("cur2"), None, m.group("unit")):
            _collect(m.group(0))
    for m in _MEASURE_PREFIX_RE.finditer(query):
        fam = _MEASURE_FAMILY.get(m.group("measure").lower().rstrip("s"))
        if fam == "name_length" and re.sub(
            r"[\s-]+", "", (m.group("scope") or "").lower()
        ) in ("tld", "toplevel"):
            continue
        if fam:
            _collect(m.group(0))
    for m in _FORM_LIST_RE.finditer(query):
        if _resolve_family(None, None, m.group("unit")):
            _collect(m.group(0))
    for m in _FORM_B_RE.finditer(query):
        if _METRIC_FAMILY.get(re.sub(r"\s+", " ", m.group("metric").lower())):
            _collect(m.group(0))
    for m in _FORM_A_RE.finditer(query):
        if not any(m.group(g) for g in ("cur", "curword", "unit")):
            continue
        if _resolve_family(m.group("cur"), m.group("curword"), m.group("unit")):
            _collect(m.group(0))
    for m in _SPELLED_MEASURE_RE.finditer(query):
        if (
            _UNIT_FAMILY.get(m.group("unit").lower()) is not None
            and _SPELLED_NUM.get(m.group("qty").lower()) is not None
        ):
            _collect(m.group(0))
    for m in _FORM_C_RE.finditer(query):
        unit = m.group("unit1") or m.group("unit2")
        if unit and _UNIT_FAMILY.get(unit.lower()):
            _collect(m.group(0))
    for m in _SPELLED_NUMERIC_RE.finditer(query):
        if (
            _UNIT_FAMILY.get(m.group("unit").lower()) is not None
            and _SPELLED_NUM.get(m.group("qty").lower()) is not None
        ):
            _collect(m.group(0))
    return tokens


_CURRENCY_SYMBOL_RE = re.compile(r"(?P<sym>[$€£¥₹])")
_CURRENCY_SYMBOL_TO_CODE: Dict[str, str] = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "₹": "INR",
}


def _parse_currency(text: str, slots: Dict[str, object]) -> None:
    """Extract filterPriceCurrency from ISO codes, words, or symbols ($/€/£/…).

    Always emit the slot when a currency cue is present — including USD for
    ``$`` / ``usd`` / ``dollars`` (qie_only + full-search regex paths).
    """
    m = _CURRENCY_LOCALE_RE.search(text)
    if m is not None:
        code = _CURRENCY_LOCALE_TO_CODE.get(m.group("locale").lower())
        if code:
            _add_slot(slots, "filterPriceCurrency", code)
            return
    m = _CURRENCY_CODE_RE.search(text)
    if m is not None:
        _add_slot(slots, "filterPriceCurrency", m.group("code").upper())
        return
    m = _CURRENCY_WORD_RE.search(text)
    if m is not None:
        code = _CURRENCY_WORD_TO_CODE.get(m.group("word").lower())
        if code:
            _add_slot(slots, "filterPriceCurrency", code)
            return
    m = _CURRENCY_SYMBOL_RE.search(text)
    if m is not None:
        code = _CURRENCY_SYMBOL_TO_CODE.get(m.group("sym"))
        if code:
            _add_slot(slots, "filterPriceCurrency", code)


def _parse_tld(
    text: str, slots: Dict[str, object], known_tlds: FrozenSet[str] = frozenset()
) -> None:
    """Extract include (dotted / contextual) and exclude TLD tokens into slots.

    A contextual candidate written without a leading dot ("tld length") is
    accepted only when it appears in known_tlds; a candidate carrying an explicit
    dot ("extension .store") is trusted as-is. When known_tlds is empty no
    contextual validation runs.

    Exclude tokens mirror LLM tldExcludeList: extension labels only — never
    char-constraint or auction-type words ("no hyphens", "exclude backorder").
    """
    excluded: Set[str] = set()
    excl_spans: List[Tuple[int, int]] = []
    for m in _TLD_EXCLUDE_RE.finditer(text):
        # "but not crypto …" -> keyword/topic exclude. "but not .net" stays TLD exclude.
        prefix = text[max(0, m.start() - 6) : m.start()]
        body = m.group("body") or m.group("tld") or ""
        if re.search(r"\bbut\s*$", prefix, re.IGNORECASE):
            first = (m.group("tld") or "").lower()
            dotted = "." in (m.group(0) or "")
            if not dotted and (
                not first or first not in known_tlds or first in _TLD_EXCLUDE_BLOCKLIST
            ):
                continue
        # "no hyphen .com" / "no numbers .io" -> char constraint + TLD include, not exclude.
        if re.search(
            r"^(?:hyphen|hyphens|dash|dashes|number|numbers|digit|digits|"
            r"letter|letters)\b",
            body,
            re.IGNORECASE,
        ):
            continue
        # "no ai or gpt words" / "no crypto vibe" -> keyword exclude, not tldExcludeList.
        if re.search(r"\b(?:words?|vibe)\b", body, re.IGNORECASE):
            continue
        rest = text[m.end() : m.end() + 48]
        if re.search(
            r"^\s*(?:or\s+\w[\w.-]*\s+)?words?\b|^\s*vibe\b", rest, re.IGNORECASE
        ):
            continue
        toks = re.findall(r"[a-z]{2,24}", body.lower())
        toks = [t for t in toks if t not in _TLD_BODY_NOISE_WORDS]
        if not toks:
            continue
        # Undotted multi-token bodies must be pure known TLDs (reject "not crypto com").
        has_dot = "." in (m.group(0) or "")
        if not has_dot and len(toks) > 1:
            all_static = all(t in _KNOWN_TLDS_FOR_BARE_EXCLUDE for t in toks)
            if not all_static and (
                not known_tlds or any(t not in known_tlds for t in toks)
            ):
                continue
        matched_any = False
        for tld in toks:
            if tld in _TLD_EXCLUDE_BLOCKLIST or tld in _CURRENCY_CODES_LOWER:
                continue
            explicit_dot = bool(
                re.search(r"\." + re.escape(tld) + r"\b", m.group(0), re.IGNORECASE)
            )
            if (
                not explicit_dot
                and tld not in _KNOWN_TLDS_FOR_BARE_EXCLUDE
                and known_tlds
                and tld not in known_tlds
            ):
                continue
            excluded.add(tld)
            matched_any = True
            _add_slot(slots, "tldExcludeList", tld)
        if matched_any:
            excl_spans.append((m.start(), m.end()))
    for m in _TLD_DOTTED_RE.finditer(text):
        if any(cs <= m.start() < ce for cs, ce in excl_spans):
            continue
        tld = m.group("tld").lower()
        if tld in _CURRENCY_CODES_LOWER:
            continue
        if tld not in excluded:
            _add_slot(slots, "tld", tld)
    for m in _TLD_CONTEXT_RE.finditer(text):
        tld = (m.group("tld") or m.group("tld2") or "").lower()
        if not tld or tld in excluded or tld in _CURRENCY_CODES_LOWER:
            continue
        explicit_dot = "." in m.group(0)
        if explicit_dot or not known_tlds or tld in known_tlds:
            _add_slot(slots, "tld", tld)
    # Bare "ai domains" / "io domain" (LLM maps short TLD noun -> tldIncludeList).
    # Skip when niche/premium modifier owns the token as topic_include.
    for m in _TLD_BARE_DOMAIN_RE.finditer(text):
        tld = m.group("tld").lower()
        if tld in excluded or tld in _CURRENCY_CODES_LOWER:
            continue
        if known_tlds and tld not in known_tlds:
            continue
        prefix = text[max(0, m.start() - 28) : m.start()].lower()
        if re.search(
            r"\b(?:premium|extended|brandable|startup|fintech|saas|crypto|cloud|"
            r"short|gem)\s*$",
            prefix,
        ):
            continue
        _add_slot(slots, "tld", tld)
    # "com or io" / "ai or io" without a leading dot.
    for m in _TLD_BARE_OR_RE.finditer(text):
        for tok in re.findall(
            r"\b(?:com|net|org|io|ai|co|app|dev|xyz)\b",
            m.group("body"),
            flags=re.IGNORECASE,
        ):
            tld = tok.lower()
            if tld in excluded or tld in _CURRENCY_CODES_LOWER:
                continue
            if known_tlds and tld not in known_tlds:
                continue
            _add_slot(slots, "tld", tld)


def _auction_label(raw: str) -> Optional[str]:
    """Map a raw auction-type surface token to its canonical label via normalized lookup."""
    return _AUCTION_KEYWORD_TO_LABEL.get(re.sub(r"[\s-]+", "", raw.lower()))


def _parse_auction(text: str, slots: Dict[str, object]) -> None:
    """Extract auction-type include/exclude canonical labels into slots."""
    excl_spans: List[Tuple[int, int]] = []
    for m in _AUCTION_EXCLUDE_RE.finditer(text):
        label = _auction_label(m.group(1))
        if label:
            _add_slot(slots, "typeExcludeList", label)
        excl_spans.append((m.start(), m.end()))
    for m in _AUCTION_RE.finditer(text):
        if any(cs <= m.start() < ce for cs, ce in excl_spans):
            continue
        # "expiring soon/this week" is end-urgency, not typeIncludeList=expiry.
        rest = text[m.end() : m.end() + 24]
        if re.match(
            r"\s*(?:soon|this\s+week|today|tonight|tomorrow)\b",
            rest,
            re.IGNORECASE,
        ):
            continue
        label = _auction_label(m.group(1))
        # Lifecycle "pending delete" owns the inventory; skip redundant expiry type.
        if label == "expiry" and re.search(
            r"\bpending\s+delete\b", text, re.IGNORECASE
        ):
            continue
        # Analytics "expiry status distribution" -> no auction type filter.
        if label == "expiry" and re.search(
            r"\b(?:distribution|volume|growing)\b", text, re.IGNORECASE
        ):
            continue
        # "standard or partner" -> keep canonical "standard" (not listed).
        if label == "listed" and re.search(
            r"\bstandard\s+or\s+partner\b|\bpartner\s+or\s+standard\b",
            text,
            re.IGNORECASE,
        ):
            label = "standard"
        if label:
            _add_slot(slots, "auction_type", label)
    # Cue owns both labels even when one surface token was missed.
    if re.search(
        r"\bstandard\s+or\s+partner\b|\bpartner\s+or\s+standard\b",
        text,
        re.IGNORECASE,
    ):
        _add_slot(slots, "auction_type", "standard")
        _add_slot(slots, "auction_type", "partner")
    # Numeric IDs ("auction type 16") kept as storage IDs on auction_type.
    for m in _AUCTION_TYPE_ID_RE.finditer(text):
        aid = m.group("id")
        if aid:
            _add_slot(slots, "auction_type", aid)


def _parse_keywords(text: str, slots: Dict[str, object]) -> None:
    """Extract prefix / suffix / contains keyword tokens into slots."""
    m_multi = _KEYWORD_STARTS_MULTI_RE.search(text)
    if m_multi is not None:
        for part in re.split(
            r"\s+(?:or|and)\s+", m_multi.group("body"), flags=re.IGNORECASE
        ):
            kw = (part or "").strip().lower()
            if kw and kw not in _KEYWORD_CONTAINS_BLOCKLIST:
                _add_slot(slots, "keyword_starts_with", kw)
    else:
        for m in _KEYWORD_STARTS_RE.finditer(text):
            kw = m.group("kw").lower()
            if kw not in _KEYWORD_CONTAINS_BLOCKLIST:
                _add_slot(slots, "keyword_starts_with", kw)
    for m in _KEYWORD_ENDS_RE.finditer(text):
        kw = (m.group("kw") or m.group("kw2") or "").lower()
        if not kw or kw in _KEYWORD_CONTAINS_BLOCKLIST:
            continue
        # "ends in com" -> TLD parser; keep non-TLD suffixes ("ends in hq").
        if kw in _TOPIC_TLD_TOKENS:
            continue
        # Time/auction lexicon — not suffix keywords ("ends in auction soon").
        if kw in (
            "auction",
            "auctions",
            "soon",
            "today",
            "tonight",
            "tomorrow",
            "no",
            "bids",
            "bid",
            "hours",
            "days",
            "week",
            "weekend",
        ):
            continue
        _add_slot(slots, "keyword_ends_with", kw)
    # Skip contains when exact-phrase cue owns the keyword (avoid double-emit).
    skip_contains = bool(re.search(r"\b(?:exact\s+)?phrase\b", text, re.IGNORECASE))
    if not skip_contains:
        m_contains_multi = _KEYWORD_CONTAINS_MULTI_RE.search(text)
        if m_contains_multi is not None:
            for part in re.split(
                r"\s+(?:or|and)\s+", m_contains_multi.group("body"), flags=re.IGNORECASE
            ):
                kw = (part or "").strip().lower()
                if kw and kw not in _KEYWORD_CONTAINS_BLOCKLIST:
                    _add_slot(slots, "keyword_contains", kw)
    for pat in _KEYWORD_CONTAINS_PATTERNS:
        for m in pat.finditer(text):
            if skip_contains:
                continue
            kw = m.group("kw").lower()
            if kw not in _KEYWORD_CONTAINS_BLOCKLIST:
                _add_slot(slots, "keyword_contains", kw)


def _parse_char_constraints(text: str, slots: Dict[str, object]) -> None:
    """Extract has_hyphen / has_number / excludeLetters / is_idn constraints into slots."""
    for slot, pattern, value in _CHAR_RULES_COMPILED:
        if pattern.search(text):
            _add_slot(slots, slot, value)


def _parse_time_remaining(text: str, slots: Dict[str, object]) -> None:
    """Extract time_remaining_max from numeric ('ending in N hours') or calendar-alias ('ending today') phrases."""
    m = _TIME_REMAINING_RE.search(text)
    if m is not None:
        seconds = (
            int(m.group("num")) * _TIME_UNIT_SECONDS[_time_unit_key(m.group("unit"))]
        )
        # "next few hours" / small hour counts -> L0 -1d parity (not -Nh).
        if seconds < 86_400 and re.search(
            r"\b(?:few\s+hours|next\s+few|final\s+hours)\b", text, re.IGNORECASE
        ):
            seconds = 86_400
        _add_slot(slots, "time_remaining_max", seconds)
        return
    m = _TIME_ALIAS_RE.search(text)
    if m is not None:
        alias = re.sub(r"\s+", " ", m.group("alias").lower())
        secs = _TIME_ALIAS_SECONDS.get(alias)
        if secs is not None:
            _add_slot(slots, "time_remaining_max", secs)
            return
    if _BARE_END_URGENCY_RE.search(text) and "time_remaining_max" not in slots:
        _add_slot(slots, "time_remaining_max", 86_400)
    # "ends in auction soon" / "ending soon under" (incomplete budget) -> -1d.
    if "time_remaining_max" not in slots and re.search(
        r"\bends?\s+in\s+auction\s+soon\b"
        r"|\bending\s+soon\s+under\b"
        r"|\bexpiring\s+now\b",
        text,
        re.IGNORECASE,
    ):
        _add_slot(slots, "time_remaining_max", 86_400)


def _parse_lifecycle(text: str, slots: Dict[str, object]) -> None:
    """Extract a marketplace lifecycle_state canonical label into slots."""
    # Advisory/guidance about expired domains -> no inventory lifecycle chip.
    if _ADVISORY_NO_INVENTORY_RE.search(text) and not _INVENTORY_BOUND_RE.search(text):
        return
    m = _LIFECYCLE_RE.search(text)
    if m is None:
        return
    phrase = re.sub(r"\s+", " ", m.group("phrase").lower())
    label = _LIFECYCLE_PHRASE_MAP.get(phrase)
    if not label:
        return
    # "available now" is browsing emphasis; only fire on price/auction/inventory bound.
    # TLD or topic alone does not indicate lifecycle intent.
    if phrase == "available now" and not (
        slots.get("price_min")
        or slots.get("price_max")
        or slots.get("auction_type")
        or _INVENTORY_BOUND_RE.search(text)
    ):
        return
    # Bare "expired" in soft preference lists ("expired or brandable or…") -> skip.
    if (
        phrase == "expired"
        and re.search(
            r"\bexpired\s+or\b|\bor\s+expired\b|\bexpired\s+or\s+\w+\s+or\b",
            text,
            re.IGNORECASE,
        )
        and not re.search(
            r"\bexpired\s+(?:domains?|with|under|com|io)\b", text, re.IGNORECASE
        )
    ):
        return
    # "bid accepted still open" -> isBidAccepted owns intent; skip active lifecycle chip.
    if label == "active" and slots.get("isBidAccepted") is True:
        return
    # Contrastive "but if … expired" — skip when conditional precedes the lifecycle cue.
    if label in ("expired", "pending_delete", "deleted"):
        m_cond = _CONDITIONAL_INVENTORY_RE.search(text)
        if m_cond is not None:
            m_life = re.search(
                r"\b(?:expired|pending\s+delete|dropped\s+today|newly\s+dropped)\b",
                text,
                re.IGNORECASE,
            )
            if m_life is not None and m_cond.start() < m_life.start():
                return
    _add_slot(slots, "lifecycle_state", label)


def _parse_boolean_flags(text: str, slots: Dict[str, object]) -> None:
    """Extract boolean flag slots via canonical signal-phrase lookup (data-driven).

    False-polarity phrases (_BOOLEAN_FALSE_SLOT_SIGNALS) win over true phrases when
    both could match (e.g. 'no reserve' must not become has_reserve_price=True).

    LLM: 'unknown age under 500' -> maxPrice only — omit domain_age_is_unknown when a
    bare under-N price cue sits next to the unknown-age phrase (no years unit).
    """
    text_lower = text.lower()
    skip_unknown_age = bool(
        _UNKNOWN_AGE_CUE_RE.search(text)
        and re.search(r"\bunder\s+\d", text_lower)
        and not re.search(r"\byears?\b", text_lower)
    )
    false_claimed: set = set()
    for slot, signals in _BOOLEAN_FALSE_SLOT_SIGNALS.items():
        if any(sig in text_lower for sig in signals):
            _add_slot(slots, slot, False)
            false_claimed.add(slot)
    for slot, signals in _BOOLEAN_SLOT_SIGNALS.items():
        if slot in false_claimed:
            continue
        if slot == "domain_age_is_unknown" and skip_unknown_age:
            continue
        if any(sig in text_lower for sig in signals):
            _add_slot(slots, slot, True)


_SPELLED_POSTFIX_RE = re.compile(
    r"\b(?P<qty>"
    + "|".join(re.escape(k) for k in sorted(_SPELLED_NUM, key=len, reverse=True))
    + r")\s+"
    r"(?P<unit>" + _UNIT + r")\s+(?P<cmp>" + _POSTFIX_CMP + r")\b",
    re.IGNORECASE,
)


def _parse_spelled_numeric(text: str, slots: Dict[str, object]) -> None:
    """Extract spelled numeric quantities for all unit families ('zero bids', 'at least five backlinks')."""
    for m in _SPELLED_POSTFIX_RE.finditer(text):
        family = _UNIT_FAMILY.get(m.group("unit").lower())
        qty = _SPELLED_NUM.get(m.group("qty").lower())
        if family is None or qty is None:
            continue
        cmp_raw = m.group("cmp")
        direction = _COMPARATOR_DIRECTION.get(re.sub(r"\s+", " ", cmp_raw.lower()))
        if direction is None:
            continue
        _emit_bound(
            slots, family, direction, float(qty), cmp=cmp_raw, unit=m.group("unit")
        )
    for m in _SPELLED_NUMERIC_RE.finditer(text):
        family = _UNIT_FAMILY.get(m.group("unit").lower())
        qty = _SPELLED_NUM.get(m.group("qty").lower())
        if family is None or qty is None:
            continue
        cmp_raw = m.group("cmp")
        if cmp_raw:
            direction = _COMPARATOR_DIRECTION.get(re.sub(r"\s+", " ", cmp_raw.lower()))
        elif family == "bids" and qty == 0:
            # "zero bids yet" -> maxBids=0 (ceiling), not exact min=max.
            direction = "max"
        elif not _FAMILY_SPEC[family][3]:
            direction = "exact"
        else:
            continue
        if direction is None:
            continue
        _emit_bound(
            slots, family, direction, float(qty), cmp=cmp_raw, unit=m.group("unit")
        )


def _parse_char_pattern_and_gem(text: str, slots: Dict[str, object]) -> None:
    """Extract charPattern; gem / fraction cues -> price_below_market (LLM parity)."""
    pats: List[str] = []
    for pat_match in _CHAR_PATTERN_RE.finditer(text):
        gd = pat_match.groupdict()
        for key in ("pat", "pat2", "pat3", "pat4", "pat5"):
            raw = (gd.get(key) or "").lower()
            if raw and raw not in pats:
                pats.append(raw)
        # Coordinated body may have more than pat4/pat5 via full span split.
        span = pat_match.group(0)
        if re.search(r"\bor\b", span, re.IGNORECASE):
            for part in re.split(r"\s+or\s+", span, flags=re.IGNORECASE):
                tok = part.strip().lower()
                if re.fullmatch(r"[vcn]{2,8}", tok) and tok not in pats:
                    pats.append(tok)
    if pats:
        _add_slot(slots, "charPattern", pats if len(pats) > 1 else pats[0])
    # Gem / fraction cues -> price_below_market (keep with letters-only typeability).
    if _FRACTION_OF_WORTH_RE.search(text) or _GEM_RE.search(text):
        _add_slot(slots, "price_below_market", True)


def _parse_keyword_advanced(text: str, slots: Dict[str, object]) -> None:
    """Extract keyword_match_mode, keyword_contains_exclude, keyword_phrase."""
    m = _KEYWORD_MATCH_MODE_RE.search(text)
    if m is not None:
        mode = (
            "all" if re.search(r"\ball\b|every", m.group(0), re.IGNORECASE) else "any"
        )
        _add_slot(slots, "keyword_match_mode", mode)
    m = _KEYWORD_CONTAINS_EXCLUDE_RE.search(text)
    if m is not None:
        kw = (
            m.group("kw") or m.group("kw2") or m.group("kw3") or m.group("kw4") or ""
        ).lower()
        # Speech-act excludes (non-X / doesn't-sound-X) bypass structural blocklist.
        speech_act = bool(m.group("kw2") or m.group("kw3") or m.group("kw4"))
        if kw and (speech_act or kw not in _KEYWORD_CONTAINS_BLOCKLIST):
            _add_slot(slots, "keyword_contains_exclude", kw)
    # "no crypto vibe" / "no ai or gpt words" (multi-token exclude).
    m = _KEYWORD_NO_VIBE_OR_WORDS_RE.search(text)
    if m is not None:
        multi = m.group("multi")
        single = m.group("single")
        parts = re.split(r"\s+or\s+", multi, flags=re.IGNORECASE) if multi else [single]
        for part in parts:
            kw = (part or "").strip().lower()
            if kw and kw not in _KEYWORD_CONTAINS_BLOCKLIST:
                _add_slot(slots, "keyword_contains_exclude", kw)
    m = _KEYWORD_PHRASE_RE.search(text)
    if m is not None:
        phrase = (m.group("qphrase") or m.group("phrase") or "").strip().lower()
        # "exact phrase fintech in the name" -> keep the seed, drop inventory filler.
        if m.group("phrase"):
            phrase = re.split(
                r"\s+\b(?:in|on|of|the|a|an|name|names|domain|domains)\b",
                phrase,
                maxsplit=1,
            )[0].strip()
        if phrase:
            _add_slot(slots, "keyword_phrase", phrase)
    # Companion-field default: match style mirrors whichever keyword slot fired.
    # Leaves the explicit all/any boolean-logic match above untouched (rare, wins first).
    if "keyword_match_mode" not in slots:
        if "keyword_starts_with" in slots:
            _add_slot(slots, "keyword_match_mode", "prefix")
        elif "keyword_ends_with" in slots:
            _add_slot(slots, "keyword_match_mode", "suffix")
        elif "keyword_phrase" in slots and "keyword_contains" not in slots:
            _add_slot(slots, "keyword_match_mode", "exact")
        elif "keyword_contains" in slots:
            _add_slot(slots, "keyword_match_mode", "contains")


def _reconcile_keyword_contains_exclusion(slots: Dict[str, object]) -> None:
    """A token can't be both an include and exclude target from one query
    (zero counterexamples across the grounded dataset) — drop it from the
    positive side once an exclude cue has claimed it, and drop the
    match-mode default along with it if nothing else backs it."""
    excluded = slots.get("keyword_contains_exclude")
    contains = slots.get("keyword_contains")
    if not excluded or not isinstance(contains, list):
        return
    remaining = [kw for kw in contains if kw not in excluded]
    if remaining:
        slots["keyword_contains"] = remaining
        return
    del slots["keyword_contains"]
    if (
        slots.get("keyword_match_mode") == "contains"
        and "keyword_starts_with" not in slots
        and "keyword_ends_with" not in slots
        and "keyword_phrase" not in slots
    ):
        del slots["keyword_match_mode"]


def _parse_extra_llm_parity_cues(text: str, slots: Dict[str, object]) -> None:
    """Additive cues that mirror LLM filters missing from core numeric/keyword forms."""
    if (
        _WITH_BIDS_RE.search(text)
        and "bids_min" not in slots
        and "bids_max" not in slots
        # A number/qualifier near "bid(s)" means a specific bid-count pattern
        # should own this query — back off the bare fallback so it doesn't
        # silently override (e.g. "...under 2k with bids", "one bid or less").
        and not _BARE_BIDS_QUALIFIER_NEAR_RE.search(text)
    ):
        _add_slot(slots, "bids_min", 1)
    m = _SPEND_OVER_MAX_RE.search(text)
    if m is not None and "price_max" not in slots:
        if m.groupdict().get("n"):
            n, scaled = _parse_num_token(m.group("n") + ("k" if m.group("k") else ""))
            # Inclusive ceiling for "spend over N" budget language (LLM grounding).
            _add_slot(slots, "price_max", int(n) if not scaled else int(n))
            # Drop mistaken minPrice from "over N" polarity if present.
            slots.pop("price_min", None)
        else:
            _add_slot(slots, "price_max", 999)
    # Bare "spending N" / "spend 1k" (no over/under) -> inclusive maxPrice.
    # Skip when advisory speech-act already emptied slots, or over-form handled above.
    if "price_max" not in slots:
        m = _SPEND_AMOUNT_RE.search(text)
        if m is not None and not _SPEND_OVER_MAX_RE.search(text):
            n, _scaled = _parse_num_token(m.group("n") + ("k" if m.group("k") else ""))
            if n >= 50 or m.group("k"):
                _add_slot(slots, "price_max", int(n))
    # "in $50" / "for 100" / "around $75" — shared parse with qie_only + reconcile.
    if "price_max" not in slots or "price_min" not in slots:
        parsed = parse_budget_prefixed_price(text)
        if parsed is not None:
            n, is_band = parsed
            if is_band and "price_min" not in slots:
                _add_slot(slots, "price_min", n)
            if "price_max" not in slots:
                _add_slot(slots, "price_max", n)
    # Incomplete trailing "under" with no number -> omit price (L0 rule 4: no numeric cue).
    # Do NOT invent maxPrice=999 — that overfits dangling fragments.
    # Bare govalue/appraisal cue + "under N" already steals the number to price_max
    # (see "govalue"/_BUDGET_STEAL_FAMILIES above); when it does, grounding still
    # expects an existence floor on the valuation metric itself (row evidence:
    # "govalue under 500" -> maxPrice:499, minValuationPrice:1; "low appraisal
    # under 1k" -> maxPrice:999, minValuationPrice:1). Deliberately excludes bare
    # "value" (too generic) — only govalue/go value/appraisal/appraised/
    # appraised value/valuation cue the floor. Requires price_max already present
    # so a govalue mention with no numeric bound at all ("underpriced compared to
    # its appraisal") does not get an invented floor.
    if (
        re.search(
            r"\bgo\s*value\b|\bappraisal\b|\bappraised(?:\s+value)?\b|\bvaluation\b",
            text,
            re.IGNORECASE,
        )
        and "price_max" in slots
        and "govalue_min" not in slots
        and "govalue_max" not in slots
    ):
        _add_slot(slots, "govalue_min", 1)
    m = _VINTAGE_AGE_RE.search(text)
    if (
        m is not None
        and "domain_age_min" not in slots
        and "domain_age_max" not in slots
    ):
        raw = m.group("n") or m.group("n2") or m.group("n3")
        if raw:
            _add_slot(slots, "domain_age_min", int(raw))
        elif re.search(
            r"\b(?:old|aged)\s+domains?\b|\ban?\s+aged\s+domain\b", text, re.IGNORECASE
        ):
            _add_slot(slots, "domain_age_min", 1)
    m = _PLATFORM_TOPIC_RE.search(text)
    if m is not None:
        tok = (m.group("topic") or "").strip().lower()
        if tok and tok not in _TOPIC_TLD_TOKENS:
            _add_slot(slots, "topic_include", tok)
    m = _REGISTERED_BEFORE_RE.search(text)
    if (
        m is not None
        and "domain_age_min" not in slots
        and "domain_age_max" not in slots
    ):
        year = int(m.group("year"))
        # Domains registered before YEAR are at least (now.year - YEAR) years old.
        age = max(1, datetime.now(timezone.utc).year - year)
        _add_slot(slots, "domain_age_min", age)
    # "backlinks already built/established/there/in place" — generic existing-inventory
    # phrasing that grounding treats as a soft web-traffic signal, not a hard Majestic
    # floor (row evidence: "domain with backlinks already built" ->
    # has_web_traffic_signal:true, not minMajesticBackLinks:1).
    if (
        re.search(
            r"\bbacklinks?\s+already\s+(?:built|established|there|in\s+place)\b",
            text,
            re.IGNORECASE,
        )
        and "majestic_backlinks_min" not in slots
        and "has_web_traffic_signal" not in slots
    ):
        _add_slot(slots, "has_web_traffic_signal", True)
    elif (
        _STRONG_LINK_PROFILE_RE.search(text)
        and "majestic_backlinks_min" not in slots
        and not _SOFT_OR_LINK_TRAFFIC_RE.search(text)
        and not _ADVISORY_NO_INVENTORY_RE.search(text)
        and not _STRONG_ADVISORY_RE.search(text)
        # Contrastive "but if … with backlinks" — hypothetical, not hard invent.
        and not _CONDITIONAL_INVENTORY_RE.search(text)
        # "clean backlinks" -> hyphen-clean + age; not soft backlink floor (LLM).
        and not re.search(r"\bclean\s+backlinks?\b", text, re.IGNORECASE)
    ):
        # "barely any backlinks" -> soft floor 0 (L0); other soft link cues -> 1.
        floor = (
            0 if re.search(r"\bbarely\s+any\s+backlinks?\b", text, re.IGNORECASE) else 1
        )
        _add_slot(slots, "majestic_backlinks_min", floor)
    # "google already likes" / "liked by google" -> soft Trust Flow floor.
    if (
        re.search(
            r"\bgoogle\s+(?:already\s+)?likes?\b"
            r"|\bliked\s+by\s+google\b"
            r"|\bgoogle[- ]friendly\b",
            text,
            re.IGNORECASE,
        )
        and "majestic_tf_min" not in slots
        and "majestic_tf_max" not in slots
    ):
        _add_slot(slots, "majestic_tf_min", 1)
    # Leading keyword before dotted TLD: "shop .store domains" -> keyword_contains.
    # Skip niche lexicon (topic owns), inventory/quality modifiers, and meta blocklist.
    m_kw_tld = re.search(
        r"\b(?P<kw>[a-z][a-z0-9]{1,24})\s+\.(?P<tld>[a-z]{2,24})\b",
        text,
        re.IGNORECASE,
    )
    if m_kw_tld is not None:
        kw = m_kw_tld.group("kw").lower()
        if (
            kw not in _KEYWORD_CONTAINS_BLOCKLIST
            and kw not in _KEYWORD_BEFORE_TLD_STOP
            and kw not in _NICHE_TOPIC_STOP
            and kw not in _TOPIC_TLD_TOKENS
            and kw not in _TOPIC_LEXICON
            and "keyword_contains" not in slots
        ):
            _add_slot(slots, "keyword_contains", kw)
    # "500 backlinks majestic" -> minMajesticBackLinks (prefer majestic when named).
    m = re.search(
        r"\b(?P<n>\d[\d,]*)\s+backlinks?\s+majestic\b"
        r"|\bmajestic\s+(?P<n2>\d[\d,]*)\s+backlinks?\b",
        text,
        re.IGNORECASE,
    )
    if m is not None and "majestic_backlinks_min" not in slots:
        raw = m.group("n") or m.group("n2")
        _add_slot(slots, "majestic_backlinks_min", int(raw.replace(",", "")))
    if (
        _HIGH_CPC_RE.search(text)
        and "semrush_cpc_min" not in slots
        and "semrush_cpc_max" not in slots
    ):
        _add_slot(slots, "semrush_cpc_min", 1.0)
    # "cpc under a dollar" / "cpc below one dollar" -> exclusive max < $1.
    if (
        re.search(
            r"\b(?:cpc|cost\s+per\s+click)\s+(?:under|below|less\s+than)\s+"
            r"(?:a\s+|one\s+)?dollars?\b",
            text,
            re.IGNORECASE,
        )
        and "semrush_cpc_max" not in slots
    ):
        _add_slot(slots, "semrush_cpc_max", 0.99)
    # Soft quantifier + referring-domain lexicon -> soft floor (no number).
    if (
        re.search(
            r"\b(?:plenty|lots|many|some|decent)\s+(?:of\s+)?(?:referring\s+domains?|ref\s+domains?)\b"
            r"|\b(?:referring\s+domains?|ref\s+domains?)\s+(?:plenty|lots)\b",
            text,
            re.IGNORECASE,
        )
        and "majestic_ref_domains_min" not in slots
        and "semrush_ref_domains_min" not in slots
        and not _SOFT_OR_LINK_TRAFFIC_RE.search(text)
    ):
        # Vendor-neutral soft floor defaults to Semrush; Majestic when cued.
        if re.search(r"\bmajestic\b", text, re.IGNORECASE):
            _add_slot(slots, "majestic_ref_domains_min", 1)
        else:
            _add_slot(slots, "semrush_ref_domains_min", 1)
    # Bare "domain count" / "estibot count" without a metric numeric bound -> soft floor 1.
    # "domain count under 2k/100" with price cue -> still invent floor (under-N is budget).
    if (
        re.search(
            r"\b(?:estibot\s+)?(?:domain\s+count|namespace\s+count)\b",
            text,
            re.IGNORECASE,
        )
        and "minEstibotDomainCount" not in slots
        and "maxEstibotDomainCount" not in slots
    ):
        # Skip only when estibot owns a small exclusive/inclusive numeric bound.
        estibot_owns_num = bool(
            re.search(
                r"\bestibot\b.{0,24}\b(?:under|below|over|above|at\s+least)\s+\$?\d"
                r"|\b(?:under|below|over|above|at\s+least)\s+\$?\d[\d,]*(?:\s*[km])?.{0,16}\bestibot\b",
                text,
                re.IGNORECASE,
            )
        )
        if not estibot_owns_num:
            _add_slot(slots, "minEstibotDomainCount", 1)
    # "dictionary word(s)" / "single dictionary word" -> word_count=1 (quality lexicon).
    if (
        re.search(
            r"\b(?:single\s+|one\s+|1[- ])?dictionary\s+words?\b", text, re.IGNORECASE
        )
        and "word_count_min" not in slots
        and "word_count_max" not in slots
    ):
        _add_slot(slots, "word_count_min", 1)
        _add_slot(slots, "word_count_max", 1)
    # "N syllable brand" -> word_count (LLM maps syllable count onto word slots).
    m_syl = re.search(
        r"\b(?P<w>one|two|three|four|five|six|1|2|3|4|5|6)\s*[- ]?syllables?\b",
        text,
        re.IGNORECASE,
    )
    # N-syllable brand -> word_count=N (L0 maps syllable cue onto word slots).
    if (
        m_syl is not None
        and "word_count_min" not in slots
        and "word_count_max" not in slots
    ):
        raw = m_syl.group("w").lower()
        n = _SPELLED_NUM.get(raw) if raw.isalpha() else int(raw)
        if n:
            _add_slot(slots, "word_count_min", n)
            _add_slot(slots, "word_count_max", n)
    # "single bid" / "one bid" auctions -> maxBids=1.
    if (
        re.search(r"\b(?:single|one|1)\s+bids?\b", text, re.IGNORECASE)
        and "bids_max" not in slots
        and "bids_min" not in slots
    ):
        _add_slot(slots, "bids_max", 1)
    # Soft "barely any / low / high" + estibot ext lexicon -> floor 1.
    # Allow when tiny under-N is present (budget/price owns that under; soft floor stays).
    if (
        re.search(
            r"\b(?:barely\s+any|low|high|strong)\s+(?:dev\s+)?ext(?:ension)?(?:\s+count)?\b"
            r"|\b(?:dev\s+)?ext(?:ension)?\s+count\s+(?:high|low|strong)\b",
            text,
            re.IGNORECASE,
        )
        and "minEstibotExtCount" not in slots
        and "minEstibotExtCountDev" not in slots
        and "maxEstibotExtCount" not in slots
        and "maxEstibotExtCountDev" not in slots
    ):
        # All soft-polarity ext count cues map to base ExtCount (grounding parity).
        _add_slot(slots, "minEstibotExtCount", 1)
    # "dev ext N [plus]" (no "count") -> also keep .dev TLD inventory chip.
    # Soft polarity ("barely any/high/low dev ext") and "dev ext count" stay metric-only.
    if (
        ("minEstibotExtCountDev" in slots or "maxEstibotExtCountDev" in slots)
        and re.search(
            r"\bdev\s+ext(?:ension)?\s+\d",
            text,
            re.IGNORECASE,
        )
        and not re.search(r"\bdev\s+ext(?:ension)?\s+count\b", text, re.IGNORECASE)
        and "tld" not in slots
    ):
        _add_slot(slots, "tld", "dev")
    if (
        _DECENT_AUTHORITY_RE.search(text)
        and "semrush_authority_min" not in slots
        and "semrush_authority_max" not in slots
        # Soft OR "authority or brandable" — no invented authority floor.
        and not re.search(
            r"\bauthority\s+or\s+(?:at\s+least\s+)?(?:be\s+)?brandable\b", text, re.I
        )
        and not re.search(r"\bor\s+authority\b", text, re.I)
    ):
        _add_slot(slots, "semrush_authority_min", 1)
    m = _NICHE_TOPIC_RE.search(text)
    if m is not None and "topic_include" not in slots:
        body = m.group("body")
        single = m.group("single")
        parts = re.split(r"\s+or\s+", body, flags=re.IGNORECASE) if body else [single]
        for part in parts:
            topic = (part or "").strip().lower()
            if topic and topic not in _NICHE_TOPIC_STOP and len(topic) >= 2:
                _add_slot(slots, "topic_include", topic)
    if (
        _NEW_KEYWORD_RE.search(text)
        and "days_listed_max" not in slots
        and "days_listed_min" not in slots
    ):
        _add_slot(slots, "days_listed_max", 1)
    if (
        _SHORT_NAME_RE.search(text)
        and "name_length_min" not in slots
        and "name_length_max" not in slots
    ):
        # Bare "short" -> maxSldLen=5 unless an explicit char/letter bound exists,
        # word-count owns the length cue, or short is a soft preference.
        has_len_unit = bool(
            re.search(
                r"\b(?:chars?|characters?|letters?|sld\s*len)\b",
                text,
                re.IGNORECASE,
            )
        )
        # Soft hedge only when it scopes the short cue itself.
        soft_pref = bool(
            re.search(
                r"\b(?:preferably|ideally)\s+short\b"
                r"|\bshort\b.{0,24}\b(?:preferably|ideally|if\s+possible)\b"
                r"|\bshould\s+sound\b",
                text,
                re.IGNORECASE,
            )
        )
        word_count_owns = "word_count_min" in slots or "word_count_max" in slots
        # Soft "short domain" beside traffic+price — skip invented length UNLESS
        # short is bound to an explicit TLD inventory cue ("short com or io").
        soft_beside_inventory = bool(
            re.search(r"\b(?:some|existing|real)\s+traffic\b", text, re.IGNORECASE)
            and re.search(r"\b(?:under|below|over|above)\s+\$?\d", text, re.IGNORECASE)
            and not re.search(
                r"\bshort\b.{0,16}\b(?:com|io|ai|net|org)\b"
                r"|\bshort\s+(?:names?|domains?)\b",
                text,
                re.IGNORECASE,
            )
        )
        if (
            not has_len_unit
            and not soft_pref
            and not word_count_owns
            and not soft_beside_inventory
        ):
            _add_slot(slots, "name_length_max", 5)
    # "max N words" / word_count_max alone -> also word_count_min=1 (LLM inventory).
    if "word_count_max" in slots and "word_count_min" not in slots:
        _add_slot(slots, "word_count_min", 1)
    m = _TOPIC_DOMAINS_RE.search(text)
    if m is not None and "topic_include" not in slots:
        topic = (m.group("topic") or "").strip().lower()
        if topic and topic not in _NICHE_TOPIC_STOP and topic not in _TOPIC_TLD_TOKENS:
            _add_slot(slots, "topic_include", topic)
    for m in _BARE_TOPIC_DOMAIN_RE.finditer(text):
        topic = re.sub(
            r"\s+",
            "_",
            (
                m.group("topic")
                or m.group("topic2")
                or m.group("topic3")
                or m.group("topic4")
                or ""
            )
            .strip()
            .lower(),
        )
        topic = topic.replace("cyber_security", "cybersecurity")
        if not topic or topic in _NICHE_TOPIC_STOP or topic in _TOPIC_TLD_TOKENS:
            continue
        # Bare cybersecurity/security names without inventory -> no filter (LLM).
        if topic == "cybersecurity" and not _INVENTORY_BOUND_RE.search(text):
            continue
        # "ai startup domain" -> topic=ai only (startup is a modifier).
        if topic == "startup":
            prefix = text[max(0, m.start() - 24) : m.start()].lower()
            if re.search(
                r"\b(?:saas|fintech|edtech|devtools|healthcare|cybersecurity|"
                r"ecommerce|gaming|legal|logistics|travel|cloud|crypto|ai|"
                r"health|tech|seo|b2b)\s*$",
                prefix,
            ):
                continue
            # Soft help/browse — no topic filter.
            if re.search(
                r"\b(?:help|pls|please|advice|guidance)\b", text, re.IGNORECASE
            ):
                continue
            if re.search(r"\b(?:names?|domains?)\s+like\b", text, re.IGNORECASE):
                continue
        _add_slot(slots, "topic_include", topic)
    for m in _COMPOUND_TOPIC_RE.finditer(text):
        # Require inventory bound — "legal tech domain trustworthy" alone is soft fluff.
        if not _INVENTORY_BOUND_RE.search(text) and not re.search(
            r"\b(?:under|below|cheap|ending|with)\b",
            text,
            re.IGNORECASE,
        ):
            continue
        for raw in (m.group("a"), m.group("b")):
            topic = (raw or "").strip().lower()
            if topic == "startup":
                continue  # modifier
            if (
                topic
                and topic not in _NICHE_TOPIC_STOP
                and topic not in _TOPIC_TLD_TOKENS
            ):
                _add_slot(slots, "topic_include", topic)
    # "cloud .com under…" -> keyword_contains; "fintech .io names" / "ai .com domains" -> topic.
    for m in _BARE_TOPIC_BEFORE_TLD_RE.finditer(text):
        tok = (m.group("topic") or "").strip().lower()
        if not tok or tok in _NICHE_TOPIC_STOP:
            continue
        rest = text[m.end() : m.end() + 48]
        if re.search(r"\b(?:domains?|names?)\b", rest, re.IGNORECASE):
            if tok not in _TOPIC_TLD_TOKENS or tok == "ai":
                _add_slot(slots, "topic_include", tok)
        elif tok not in _KEYWORD_CONTAINS_BLOCKLIST:
            _add_slot(slots, "keyword_contains", tok)
    # "startup domain/name" alone -> topic. Niche+"startup domain" -> niche only
    # ("fintech startup domain" -> fintech; startup is a modifier). Soft help omit.
    _niche_already = False
    _ti = slots.get("topic_include")
    if isinstance(_ti, list):
        _niche_already = any(t != "startup" for t in _ti)
    elif isinstance(_ti, str) and _ti != "startup":
        _niche_already = True
    _niche_startup_mod = bool(
        re.search(
            r"\b(?:saas|fintech|edtech|devtools|healthcare|cybersecurity|ecommerce|"
            r"gaming|legal|logistics|travel|cloud|crypto|ai|health|tech|seo|b2b|"
            r"climate)\s+startup\b",
            text,
            re.IGNORECASE,
        )
    )
    if (
        _STARTUP_TOPIC_RE.search(text)
        and not _niche_already
        and not _niche_startup_mod
        and not re.search(
            r"\b(?:help|pls|please|advice|guidance)\b", text, re.IGNORECASE
        )
        and not _ADVISORY_NO_INVENTORY_RE.search(text)
        and "similar_to" not in slots
        and not re.search(r"\b(?:names?|domains?)\s+like\b", text, re.IGNORECASE)
    ):
        _add_slot(slots, "topic_include", "startup")
    # Drop startup co-topic when another niche owns the intent (L0 grounding).
    if isinstance(slots.get("topic_include"), list):
        topics = slots["topic_include"]
        if "startup" in topics and any(t != "startup" for t in topics):
            slots["topic_include"] = [t for t in topics if t != "startup"]
            if not slots["topic_include"]:
                del slots["topic_include"]
    # Drop startup topic when brand seeds own the intent ("startup names like …").
    if "similar_to" in slots and isinstance(slots.get("topic_include"), list):
        slots["topic_include"] = [t for t in slots["topic_include"] if t != "startup"]
        if not slots["topic_include"]:
            del slots["topic_include"]
    elif "similar_to" in slots and slots.get("topic_include") == "startup":
        del slots["topic_include"]
    # "ai startup or fintech" — startup is a modifier between niches.
    if re.search(
        r"\b(?:ai|fintech|saas|tech)\s+startup\s+or\s+(?:ai|fintech|saas|tech)\b"
        r"|\b(?:ai|fintech|saas|tech)\s+or\s+(?:ai|fintech|saas|tech)\s+startup\b",
        text,
        re.IGNORECASE,
    ) and isinstance(slots.get("topic_include"), list):
        slots["topic_include"] = [t for t in slots["topic_include"] if t != "startup"]
        if not slots["topic_include"]:
            del slots["topic_include"]
    # "want tech" / "ai gem" / "fintech com" / "saas topic" -> topic_include
    # Note: bare TLD token "ai" is allowed ONLY via ai_compound ("ai gem"), not as TLD.
    for m in _TOPICISH_WANT_RE.finditer(text):
        gd = m.groupdict()
        tok = (
            m.group("topic")
            or m.group("ai_compound")
            or m.group("topic2")
            or gd.get("topic3")
            or gd.get("topic4")
            or gd.get("topic5")
            or gd.get("topic6")
            or ""
        )
        tok = re.sub(r"\s+", "_", tok.strip().lower())
        if not tok or tok in _NICHE_TOPIC_STOP:
            continue
        if tok in _TOPIC_TLD_TOKENS and not m.group("ai_compound"):
            continue
        # "exclude crypto topic" must not also emit topic_include=crypto.
        prefix = text[max(0, m.start() - 16) : m.start()].lower()
        if re.search(r"\b(?:exclude|excluding|not|no|without)\s*$", prefix):
            continue
        # climate_tech / legal tech — do not also emit bare tech.
        if tok == "tech" and re.search(
            r"\b(?:climate|legal)\s+tech\b",
            text,
            re.IGNORECASE,
        ):
            continue
        _add_slot(slots, "topic_include", tok)
    # Loose bare niche: "launching a saas need…", "show me fintech listings".
    for m in _BARE_TOPIC_LOOSE_RE.finditer(text):
        tok = re.sub(r"\s+", "_", (m.group("topic") or "").strip().lower())
        if not tok or tok in _NICHE_TOPIC_STOP or tok in _TOPIC_TLD_TOKENS:
            continue
        if tok == "tech" and re.search(r"\bclimate\s+tech\b", text, re.IGNORECASE):
            continue
        if tok == "brandables":
            tok = "brandable"
        # "bargain/premium/browse brandables" -> PBM/browse, not topic.
        # Bare "brandables under/no more than" -> topic_include (LLM).
        if tok == "brandable":
            prefix_b = text[max(0, m.start() - 24) : m.start()].lower()
            if re.search(
                r"\b(?:bargain|premium|browse|overlooked|cheap)\s*$",
                prefix_b,
            ):
                continue
        prefix = text[max(0, m.start() - 24) : m.start()].lower()
        if re.search(
            r"\b(?:exclude|excluding|not|no|without|for|my|skip|anything|but)\s*$",
            prefix,
        ):
            continue
        _add_slot(slots, "topic_include", tok)
    # "has fintech but not crypto"
    m = _HAS_TOPIC_BUT_NOT_RE.search(text)
    if m is not None:
        topic = (m.group("topic") or "").strip().lower()
        excl = (m.group("excl") or "").strip().lower()
        if topic and topic not in _TOPIC_TLD_TOKENS and topic not in _NICHE_TOPIC_STOP:
            _add_slot(slots, "topic_include", topic)
        if excl and excl not in _KEYWORD_CONTAINS_BLOCKLIST:
            _add_slot(slots, "keyword_contains_exclude", excl)
    # "fintech or saas …"
    m = _TOPIC_OR_RE.search(text)
    if m is not None:
        for part in re.split(r"\s+or\s+", m.group("body"), flags=re.IGNORECASE):
            topic = (part or "").strip().lower()
            # Allow "ai" as topic in or-lists (not bare TLD inventory).
            if (
                topic
                and topic not in _NICHE_TOPIC_STOP
                and (topic not in _TOPIC_TLD_TOKENS or topic == "ai")
            ):
                _add_slot(slots, "topic_include", topic)
    for m in _TOPIC_EXCLUDE_RE.finditer(text):
        topic = (m.group("topic") or "").strip().lower()
        if not topic or topic in _TOPIC_TLD_TOKENS:
            continue
        # Default to topic_exclude ("not gaming", "exclude crypto topic", "not
        # fintech specifically") — vibe/keyword-suffixed phrasing never reaches
        # this loop (blocked by the regex's own negative lookahead), so an
        # explicit "keyword" cue is the only case that should route elsewhere.
        if re.search(r"\bkeywords?\b", text, re.IGNORECASE):
            _add_slot(slots, "keyword_contains_exclude", topic)
        else:
            _add_slot(slots, "topic_exclude", topic)
        cur = slots.get("topic_include")
        if isinstance(cur, list):
            slots["topic_include"] = [t for t in cur if t != topic]
            if not slots["topic_include"]:
                del slots["topic_include"]
        elif cur == topic:
            del slots["topic_include"]
    for m in _KEYWORD_EXCLUDE_EXPLICIT_RE.finditer(text):
        kw = (
            (m.group("kw") or m.group("kw2") or m.group("kw3") or m.group("kw4") or "")
            .strip()
            .lower()
        )
        topic_excl = slots.get("topic_exclude")
        already_topic_excluded = isinstance(topic_excl, list) and kw in topic_excl
        if (
            kw
            and kw not in _KEYWORD_CONTAINS_BLOCKLIST
            and not already_topic_excluded
            and kw
            not in (
                "the",
                "a",
                "an",
                "this",
                "that",
                "any",
                "some",
                "my",
                "our",
            )
        ):
            _add_slot(slots, "keyword_contains_exclude", kw)
    # Niche modifier + short token -> topic, not tld ("extended ai auction", "ai agent").
    for m in _TOPICISH_BEFORE_SHORT_TLD_RE.finditer(text):
        tok = (m.group("tok") or m.group("tok2") or "").lower()
        if tok:
            _add_slot(slots, "topic_include", tok)
    # Typo traffic cue — skip on advisory/guidance ("does trafic help…").
    if (
        _TRAFFIC_TYPO_RE.search(text)
        and "has_web_traffic_signal" not in slots
        and not _ADVISORY_NO_INVENTORY_RE.search(text)
        and "traffic_min" not in slots
        and "traffic_max" not in slots
    ):
        _add_slot(slots, "has_web_traffic_signal", True)
    # Bare inventory traffic cue (after numeric parse so traffic_min/max gate works).
    if (
        "has_web_traffic_signal" not in slots
        and "traffic_min" not in slots
        and "traffic_max" not in slots
        and re.search(
            r"(?:^|\b(?:ending\s+soon|websites?|with|have|has|some|real|and)\s+)traffic\b"
            r"|\btraffic\s*$",
            text,
            re.IGNORECASE,
        )
        and not re.search(
            r"\b(?:no|zero|without|unknown)\s+traffic\b"
            r"|\btraffic\s+(?:unknown|doesn'?t|does\s+not|not)\b"
            r"|\btraffic\s+doesn'?t\s+matter\b",
            text,
            re.IGNORECASE,
        )
        and not _ADVISORY_NO_INVENTORY_RE.search(text)
        and not _STRONG_ADVISORY_RE.search(text)
    ):
        _add_slot(slots, "has_web_traffic_signal", True)
    # Soft OR "traffic or backlinks [or authority]" -> traffic signal only.
    # Do NOT invent backlink floors (L0 keeps OR as soft traffic, not dual invent).
    if (
        _SOFT_OR_LINK_TRAFFIC_RE.search(text)
        and not _ADVISORY_NO_INVENTORY_RE.search(text)
        and not _STRONG_ADVISORY_RE.search(text)
        and (
            _INVENTORY_BOUND_RE.search(text)
            or re.search(
                r"\b(?:today|added|listed|browse|show\s+me|ending|soon)\b",
                text,
                re.IGNORECASE,
            )
        )
    ):
        if "has_web_traffic_signal" not in slots and "traffic_min" not in slots:
            _add_slot(slots, "has_web_traffic_signal", True)
    for m in _BARE_TLD_AFTER_RE.finditer(text):
        tld = (m.group("tld") or m.group("tld2") or "").lower()
        if not tld:
            continue
        # Skip only when niche modifier owns THIS short token as topic
        # ("extended ai under…" -> topic; "crypto com under…" still includes com).
        if tld in ("ai", "io", "app", "dev"):
            prefix = text[max(0, m.start() - 24) : m.start()].lower()
            if re.search(
                r"\b(?:premium|extended|brandable|startup|fintech|saas|crypto|cloud)\s*$",
                prefix,
            ):
                continue
        _add_slot(slots, "tld", tld)
    for m in _BARE_TLD_NOUN_RE.finditer(text):
        tld = (m.group("tld") or "").lower()
        if tld:
            _add_slot(slots, "tld", tld)
    for m in _BARE_TLD_EXT_RE.finditer(text):
        tld = (m.group("tld") or m.group("tld2") or "").lower()
        if not tld:
            continue
        # Metric-owned "dev/ai ext [N|count|saturation|cmp|plus]" ≠ TLD noun.
        span = text[max(0, m.start() - 16) : min(len(text), m.end() + 28)].lower()
        if re.search(
            r"\b(?:count|saturation|barely|low|high|strong|under|below|above|over|"
            r"plus|or\s+more|\d+)\b",
            span,
        ):
            continue
        _add_slot(slots, "tld", tld)
    # "zero traffic" -> traffic_is_unknown by default; "fresh slate" -> exact 0/0.
    # "traffic unknown or zero under N" -> unknown + maxTraffic=0; under N stays price.
    if re.search(
        r"\b(?:zero\s+traffic|traffic\s+unknown|unknown\s+traffic|traffic\s+unknown\s+or\s+zero)\b",
        text,
        re.IGNORECASE,
    ) or re.search(r"\btraffic\s+unknown\s+or\s+zero\b", text, re.IGNORECASE):
        if re.search(r"\bfresh\s+slate\b", text, re.IGNORECASE):
            slots.pop("traffic_is_unknown", None)
            if "traffic_min" not in slots:
                _add_slot(slots, "traffic_min", 0)
            if "traffic_max" not in slots:
                _add_slot(slots, "traffic_max", 0)
        else:
            _add_slot(slots, "traffic_is_unknown", True)
            # Exact-empty traffic band when "zero" is stated alongside unknown.
            # Keep maxTraffic=0 only (no traffic_min twin — L0 grounding omits min).
            if re.search(r"\bzero\b", text, re.IGNORECASE):
                slots.pop("traffic_min", None)
                if "traffic_max" not in slots:
                    _add_slot(slots, "traffic_max", 0)
            else:
                slots.pop("traffic_min", None)
                # Drop mistaken traffic ceilings from bare under N (price owns that).
                if slots.get("traffic_max") not in (0, None):
                    # Keep only exact-zero band; large under-N ceilings are price.
                    if (
                        isinstance(slots.get("traffic_max"), (int, float))
                        and slots["traffic_max"] >= 100
                    ):
                        del slots["traffic_max"]
    # Vendor cue "majestic" owns backlinks -> drop Semrush twin of same floor.
    if (
        re.search(r"\bmajestic\b", text, re.IGNORECASE)
        and "majestic_backlinks_min" in slots
        and "semrush_backlinks_min" in slots
        and slots.get("majestic_backlinks_min") == slots.get("semrush_backlinks_min")
    ):
        slots.pop("semrush_backlinks_min", None)
    # "low estibot (domain) count" + numeric max -> soft floor 0 (presence band).
    if (
        re.search(r"\blow\s+estibot(?:\s+(?:domain\s+)?count)?\b", text, re.IGNORECASE)
        and "maxEstibotDomainCount" in slots
        and "minEstibotDomainCount" not in slots
    ):
        _add_slot(slots, "minEstibotDomainCount", 0)
    # Leading inventory keyword before "domains" (blog/forum/…) -> keyword_contains.
    m_kw_dom = re.search(
        r"\b(?P<kw>blog|forum|wiki|news|portal)\s+domains?\b",
        text,
        re.IGNORECASE,
    )
    if m_kw_dom is not None and "keyword_contains" not in slots:
        _add_slot(slots, "keyword_contains", m_kw_dom.group("kw").lower())
    # Qualified traffic ceiling also implies traffic presence signal.
    # Same under-N must not also become maxPrice (traffic owns the bound).
    if ("traffic_max" in slots or "traffic_min" in slots) and re.search(
        r"\b(?:steady|real|existing|monthly|organic|good|decent|some)\s+"
        r"(?:web\s+)?traffic\b",
        text,
        re.IGNORECASE,
    ):
        if "has_web_traffic_signal" not in slots and "traffic_is_unknown" not in slots:
            _add_slot(slots, "has_web_traffic_signal", True)
        if "price_max" in slots and slots.get("price_max") == slots.get("traffic_max"):
            slots.pop("price_max", None)
    # "new brand" / "brand new" soft age floor.
    if (
        re.search(r"\b(?:new\s+brand|brand\s+new)\b", text, re.IGNORECASE)
        and "domain_age_min" not in slots
        and "domain_age_max" not in slots
    ):
        _add_slot(slots, "domain_age_min", 0)
    # Profession cues -> legal topic.
    if (
        re.search(
            r"\b(?:lawyer|attorney|accountant|advisor)\b",
            text,
            re.IGNORECASE,
        )
        and "topic_include" not in slots
    ):
        _add_slot(slots, "topic_include", "legal")
    # "seo value domains" -> topic + soft valuation floor. Not "exact match or brandable for seo".
    if re.search(
        r"\bseo\s+value\s+domains?\b|\bseo\s+(?:topic|category)\s+domains?\b",
        text,
        re.IGNORECASE,
    ):
        _add_slot(slots, "topic_include", "seo")
        if "govalue_min" not in slots and "minValuationPrice" not in slots:
            _add_slot(slots, "govalue_min", 1)
        # "expiring now" owns endTime/lifecycle — drop mistaken auction-type=expiry.
        at = slots.get("auction_type")
        if isinstance(at, list):
            slots["auction_type"] = [v for v in at if str(v).lower() != "expiry"]
            if not slots["auction_type"]:
                del slots["auction_type"]
        elif str(at or "").lower() == "expiry":
            del slots["auction_type"]
    # Contrastive aged-or-fresh / better-expired preference -> wipe invented age/lifecycle.
    # topic_include is independent of the aged-vs-fresh dichotomy, so it is not wiped here.
    if re.search(
        r"\baged\s+domains?\s+or\s+fresh(?:\s+domains?)?\b"
        r"|\bfresh\s+domains?\s+or\s+aged(?:\s+domains?)?\b"
        r"|\bbetter\s+expired\s+or\s+auction\b",
        text,
        re.IGNORECASE,
    ):
        for k in (
            "domain_age_min",
            "domain_age_max",
            "has_web_traffic_signal",
            "lifecycle_state",
            "lifecycle_disjunction",
        ):
            slots.pop(k, None)
        if re.search(r"\bbetter\s+expired\s+or\s+auction\b", text, re.IGNORECASE):
            slots.pop("auction_type", None)
            _add_slot(slots, "auction_type", ["expiry", "listed"])
    # Bare lexicon topic mention (LLM parity). Advisory/no-inventory queries already
    # short-circuit before this parser runs, so this cannot fire on pure Q&A about SEO.
    if re.search(r"\bseo\b", text, re.IGNORECASE):
        _add_slot(slots, "topic_include", "seo")
    # "not keyword spam" -> drop junk keyword_contains=not.
    if re.search(r"\bnot\s+keyword\b", text, re.IGNORECASE):
        kw = slots.get("keyword_contains")
        if kw == "not":
            slots.pop("keyword_contains", None)
        elif isinstance(kw, list):
            kept = [v for v in kw if str(v).lower() != "not"]
            if kept:
                slots["keyword_contains"] = kept
            else:
                slots.pop("keyword_contains", None)
    # "ai .com or .io" -> topic ai (+ tech if present) + tld com|io (not keyword=ai).
    if re.search(r"\bai\s+\.com\b|\bai\s+com\b", text, re.IGNORECASE) and re.search(
        r"\b\.?io\b",
        text,
        re.IGNORECASE,
    ):
        kw = slots.get("keyword_contains")
        if kw == "ai" or (isinstance(kw, list) and kw == ["ai"]):
            slots.pop("keyword_contains", None)
        tld = slots.get("tld")
        parts = list(tld) if isinstance(tld, list) else ([tld] if tld else [])
        parts = [str(p).lower().lstrip(".") for p in parts if p]
        for need in ("com", "io"):
            if need not in parts:
                parts.append(need)
        parts = [p for p in parts if p != "ai"]
        if parts:
            slots["tld"] = parts
        cur = slots.get("topic_include")
        if isinstance(cur, list):
            if "ai" not in cur:
                cur.append("ai")
            slots["topic_include"] = cur
        elif cur is None:
            _add_slot(slots, "topic_include", "ai")
        elif cur != "ai":
            slots["topic_include"] = [cur, "ai"]
    # "ai startup or fintech" -> drop startup from co-topic merge.
    if re.search(r"\bai\s+startup\s+or\s+fintech\b", text, re.IGNORECASE):
        cur = slots.get("topic_include")
        if isinstance(cur, list):
            slots["topic_include"] = [t for t in cur if t != "startup"]
            for tok in ("ai", "fintech"):
                if tok not in slots["topic_include"]:
                    slots["topic_include"].append(tok)
            if not slots["topic_include"]:
                del slots["topic_include"]
        else:
            slots["topic_include"] = ["ai", "fintech"]
    # "keyword frequency in high bid listings" -> soft bid floor.
    if (
        re.search(
            r"\bkeyword\s+frequency\b|\bhigh\s+bid\s+listings?\b",
            text,
            re.IGNORECASE,
        )
        and "bids_min" not in slots
        and "minBids" not in slots
    ):
        _add_slot(slots, "bids_min", 1)
    # "top domains by traffic" -> traffic presence signal.
    if re.search(r"\btop\s+domains?\s+by\s+traffic\b", text, re.IGNORECASE):
        if "has_web_traffic_signal" not in slots:
            _add_slot(slots, "has_web_traffic_signal", True)
    # "high value pending delete … by domain authority" -> soft DA floor.
    if (
        re.search(r"\b(?:domain\s+authority|high\s+da|\bda\b)\b", text, re.IGNORECASE)
        and re.search(r"\bpending\s+delete\b", text, re.IGNORECASE)
        and "semrush_authority_min" not in slots
    ):
        _add_slot(slots, "semrush_authority_min", 1)
    # "names people keep watching" -> traffic/watch presence signal.
    if re.search(
        r"\bkeep\s+watching\b|\bpeople\s+(?:keep\s+)?watching\b", text, re.IGNORECASE
    ):
        if "has_web_traffic_signal" not in slots:
            _add_slot(slots, "has_web_traffic_signal", True)
    # Soft advisory with startup inventory noun -> topic_include=startup.
    if (
        re.search(r"\bworth\s+spending\b", text, re.IGNORECASE)
        and re.search(r"\bstartup\b", text, re.IGNORECASE)
        and "topic_include" not in slots
    ):
        _add_slot(slots, "topic_include", "startup")
    # premium extension -> auction type premium (isExtended from boolean table).
    if re.search(r"\bpremium\s+extension\b", text, re.IGNORECASE):
        at = slots.get("auction_type")
        if isinstance(at, list):
            if "premium" not in {str(v).lower() for v in at}:
                at.append("premium")
                slots["auction_type"] = at
        elif at is None:
            _add_slot(slots, "auction_type", "premium")
    # Drop junk keyword_contains=short when short owns maxSldLen.
    kw = slots.get("keyword_contains")
    if kw == "short" or (isinstance(kw, list) and kw == ["short"]):
        if (
            "name_length_max" in slots
            or "maxSldLen" in slots
            or re.search(r"\bshort\b", text, re.IGNORECASE)
        ):
            slots.pop("keyword_contains", None)
    # "in healthcare and legal categories" / "healthcare category".
    if re.search(r"\bhealthcare\b.{0,40}\bcategor", text, re.IGNORECASE):
        _add_slot(slots, "topic_include", "healthcare")
    if re.search(r"\blegal\s+categor", text, re.IGNORECASE):
        _add_slot(slots, "topic_include", "legal")
    if re.search(r"\bhealthcare\s+and\s+legal\b", text, re.IGNORECASE):
        _add_slot(slots, "topic_include", "healthcare")
        _add_slot(slots, "topic_include", "legal")
    # "b2b startup or tech" -> expand co-topics (not collapse to b2b_saas).
    if re.search(r"\bb2b\s+startup\b.{0,40}\bor\b.{0,40}\btech\b", text, re.IGNORECASE):
        cur = slots.get("topic_include")
        parts = list(cur) if isinstance(cur, list) else ([cur] if cur else [])
        parts = [t for t in parts if t not in ("b2b_saas", "startup")]
        for tok in ("b2b", "saas", "tech"):
            if tok not in parts:
                parts.append(tok)
        slots["topic_include"] = parts
    # "for b2b saas" / compound owns topic — collapse b2b|saas into b2b_saas.
    elif re.search(r"\bb2b\s+saas\b|\bb2b\s+software\b", text, re.IGNORECASE):
        cur = slots.get("topic_include")
        if isinstance(cur, list):
            collapsed = [t for t in cur if t not in ("b2b", "saas", "startup", "tech")]
            if "b2b_saas" not in collapsed:
                collapsed.append("b2b_saas")
            slots["topic_include"] = collapsed
        else:
            _add_slot(slots, "topic_include", "b2b_saas")
    # ".law" TLD owns legal extension — drop redundant topic_include=legal.
    tlds = slots.get("tld")
    tld_list = tlds if isinstance(tlds, list) else ([tlds] if tlds else [])
    if "law" in {str(t).lower() for t in tld_list}:
        cur = slots.get("topic_include")
        if isinstance(cur, list):
            slots["topic_include"] = [t for t in cur if t != "legal"]
            if not slots["topic_include"]:
                del slots["topic_include"]
        elif cur == "legal":
            del slots["topic_include"]
    # keyword_phrase owns tokens — drop topic/keyword duplicates from phrase words.
    phrase = slots.get("keyword_phrase")
    if isinstance(phrase, str) and phrase:
        for tok in phrase.split():
            cur = slots.get("topic_include")
            if isinstance(cur, list) and tok in cur:
                slots["topic_include"] = [t for t in cur if t != tok]
                if not slots["topic_include"]:
                    del slots["topic_include"]
            elif cur == tok:
                del slots["topic_include"]


def _parse_similar_to(text: str, slots: Dict[str, object]) -> None:
    """Extract a similar_to brand/seed reference into slots."""
    # Guard: "no X vibe" -> keyword_contains_exclude, not similar_to.
    no_vibe_spans: List[Tuple[int, int]] = []
    for m0 in re.finditer(r"\bno\s+[a-z]{2,20}\s+vibe\b", text, re.IGNORECASE):
        no_vibe_spans.append((m0.start(), m0.end()))

    # Compound brands before style/naming ("linear app style") — keep all content tokens.
    m_style = _SIMILAR_STYLE_COMPOUND_RE.search(text)
    if m_style is not None and not any(
        cs <= m_style.start() < ce for cs, ce in no_vibe_spans
    ):
        style_seeds: List[str] = []
        for part in re.split(
            r"\s+", (m_style.group("stylebody") or "").strip().lower()
        ):
            seed = re.sub(r"[^\w.-]+", "", part).rstrip(".")
            if (
                seed
                and len(seed) >= 2
                and seed not in _SIMILAR_TO_STOP
                and not _SIMILAR_TO_PRICEISH_RE.fullmatch(seed)
            ):
                style_seeds.append(seed)
        # Allow short TLD-ish tokens (app/io) when another brand seed is present.
        if len(style_seeds) >= 2 or (
            style_seeds and style_seeds[0] not in _TOPIC_TLD_TOKENS
        ):
            for seed in style_seeds:
                if seed in _TOPIC_TLD_TOKENS and len(style_seeds) < 2:
                    continue
                _add_slot(slots, "similar_to", seed)
            if style_seeds:
                return

    m_multi = _SIMILAR_TO_MULTI_RE.search(text)
    if m_multi is not None:
        # Skip if match falls inside a "no X vibe" span.
        if not any(cs <= m_multi.start() < ce for cs, ce in no_vibe_spans):
            gd = m_multi.groupdict()
            body = (
                gd.get("body")
                or gd.get("body2")
                or gd.get("vibebody")
                or gd.get("body_sp")
                or ""
            )
            seeds = []
            # Truncate at contrastive "but …" clause noise.
            body = re.split(r"\s+but\b", body, maxsplit=1, flags=re.IGNORECASE)[0]
            # Space-separated brand lists (body_sp) split on whitespace; else or/and/maybe.
            splitter = r"\s+" if gd.get("body_sp") else r"\s*(?:or|and|maybe)\s*"
            for part in re.split(splitter, body, flags=re.IGNORECASE):
                seed = re.sub(r"[^\w.-]+", "", (part or "").strip().lower()).rstrip(".")
                seed = seed.replace("producthunt", "product_hunt")
                if (
                    seed == "producthunt"
                    or (part or "").strip().lower() == "product hunt"
                ):
                    seed = "product_hunt"
                if (
                    " " in (part or "").strip().lower()
                    and "product" in (part or "").lower()
                ):
                    seed = "product_hunt"
                if (
                    seed
                    and len(seed) >= 2
                    and seed not in _SIMILAR_TO_STOP
                    and not _SIMILAR_TO_PRICEISH_RE.fullmatch(seed)
                ):
                    seeds.append(seed)
            for seed in seeds:
                _add_slot(slots, "similar_to", seed)
            if seeds:
                return
    m = _SIMILAR_TO_RE.search(text)
    if m is not None:
        if not any(cs <= m.start() < ce for cs, ce in no_vibe_spans):
            gd = m.groupdict()
            seed = (
                (
                    gd.get("seed_brand")
                    or gd.get("seed_startup")
                    or gd.get("seed")
                    or gd.get("seed2")
                    or gd.get("seed3")
                    or gd.get("seed4")
                    or gd.get("seed5")
                    or ""
                )
                .lower()
                .rstrip(".")
            )
            # Reject price-ish / filler seeds ("like 2k", "like to").
            if seed and _SIMILAR_TO_PRICEISH_RE.fullmatch(seed):
                seed = ""
            if seed and seed not in _SIMILAR_TO_STOP and seed not in _TOPIC_TLD_TOKENS:
                _add_slot(slots, "similar_to", seed)


_ADDED_RECENTLY_1D_RE = re.compile(
    r"\b(?:added|listed)\s+recently\b|\brecently\s+(?:added|listed)\b",
    re.IGNORECASE,
)


def _parse_recency(text: str, slots: Dict[str, object]) -> None:
    """Extract days_listed_max / days_listed_min from recency and stale cues (LLM Group 4)."""
    # Hour-bound first: "fresh last 48 hours" -> startTimeAfter="-48h" (prompt hint).
    m_fresh = _FRESH_LISTING_RE.search(text)
    if m_fresh is not None:
        hours = m_fresh.group("hours") or m_fresh.group("hours2")
        if hours:
            _add_slot(slots, "startTimeAfter", f"-{int(hours)}h")
            return
        # "fresh listings" ≈ recently listed -> -7d; today/just listed -> -1d.
        span = m_fresh.group(0).lower()
        if (
            _FRESH_7D_RE.search(span)
            or "fresh listings" in span
            or "fresh listing" in span
            or "latest" in span
            or "newest" in span
            or "new arrivals" in span
        ):
            _add_slot(slots, "days_listed_max", 7)
        elif (
            "today" in span
            or "just listed" in span
            or "morning" in span
            or "afternoon" in span
            or "evening" in span
            or "dropped today" in span
            or "since yesterday" in span
            or "available today" in span
        ):
            _add_slot(slots, "days_listed_max", 1)
        else:
            _add_slot(slots, "days_listed_max", 1)
        return
    # Few-shot: "added recently" -> startTimeAfter="-7d".
    if _ADDED_RECENTLY_1D_RE.search(text):
        _add_slot(slots, "days_listed_max", 7)
        return
    m = _STALE_LISTED_RE.search(text)
    if m is not None:
        num = int(m.group("num") or m.group("num2"))
        unit = (m.group("unit") or m.group("unit2") or "days").lower()
        days = num * (7 if unit.startswith("week") else 1)
        _add_slot(slots, "days_listed_min", days)
        return
    m = _RECENCY_ALIAS_RE.search(text)
    if m is not None:
        alias = re.sub(r"\s+", " ", m.group("alias").lower())
        days = _RECENCY_ALIAS_DAYS.get(alias)
        if days is not None:
            _add_slot(slots, "days_listed_max", days)
        return
    m = _RECENCY_NUMERIC_RE.search(text)
    if m is not None:
        _add_slot(slots, "days_listed_max", int(m.group("num")))
        return
    if _RECENCY_IMPLICIT_RE.search(text):
        _add_slot(slots, "days_listed_max", 7)


def _parse_owner_members(text: str, slots: Dict[str, object]) -> None:
    """Extract ownerMemberIncludeList / ownerMemberExcludeList (LLM Parameter hints)."""
    for m in _OWNER_EXCLUDE_RE.finditer(text):
        _add_slot(slots, "ownerMemberExcludeList", m.group("id"))
    for m in _OWNER_INCLUDE_RE.finditer(text):
        _add_slot(slots, "ownerMemberIncludeList", m.group("id"))


_UNIQUE_SEARCHES_BOUND_RE = re.compile(
    r"\b(?:(?P<cmp>" + _CMP + r")\s+)?(?P<num>" + _NUM + r")\s+unique\s+searches?\b"
    r"|\bmin(?:imum)?\s+(?P<num2>" + _NUM + r")\s+unique\s+searches?\b",
    re.IGNORECASE,
)
# "N monthly visitors" / "N monthly hits" / "N per month visitors" -> minUniqueSearches.
# Consumed before generic numeric parse so leading monthly-visitor cues emit
# unique_searches, not traffic.
# NOTE: the trailing order ("N visitors monthly" / "N hits per month") is
# deliberately NOT matched here — grounding evidence ("at least 500 visitors
# monthly" -> minTraffic:500; "under 200 visitors monthly" -> maxTraffic:199)
# shows that order routes to the traffic family via the generic Form-A/
# _UNIT_FAMILY visitor/hit/pageview mapping instead. Only the leading
# "monthly X" compound is a unique_searches cue.
_MONTHLY_VISITORS_RE = re.compile(
    r"\b(?:(?P<cmp2>" + _CMP + r")\s+)?(?P<num2>" + _NUM + r")\s+"
    r"(?:monthly|per\s+month)\s+(?:visitors?|hits?|pageviews?)\b",
    re.IGNORECASE,
)


def _parse_unique_searches(text: str, slots: Dict[str, object]) -> None:
    """Extract minUniqueSearches / maxUniqueSearches (LLM FIND63 / Parameter hints)."""
    m = _UNIQUE_SEARCHES_BOUND_RE.search(text)
    if m is None:
        # Also handle "N monthly visitors" (leading monthly-cued unique-searches form).
        mv = _MONTHLY_VISITORS_RE.search(text)
        if mv is not None:
            if _POSTFIX_CMP_FOLLOWS_RE.match(text[mv.end() :]):
                return  # FORM_C owns postfix-cmp variants (e.g. "5000 monthly visitors minimum")
            num_raw = mv.group("num2")
            cmp_raw = mv.group("cmp2")
            direction = (
                _COMPARATOR_DIRECTION.get(re.sub(r"\s+", " ", cmp_raw.lower()))
                if cmp_raw
                else "min"
            )
            if direction is not None and num_raw is not None:
                _emit_bound(
                    slots,
                    "unique_searches",
                    direction,
                    _to_number(num_raw),
                    cmp=cmp_raw,
                )
        return
    if m.group("num2") is not None:
        _emit_bound(slots, "unique_searches", "min", _to_number(m.group("num2")))
        return
    cmp_raw = m.group("cmp")
    direction = (
        _COMPARATOR_DIRECTION.get(re.sub(r"\s+", " ", cmp_raw.lower()))
        if cmp_raw
        else "min"
    )
    if direction is None:
        return
    _emit_bound(
        slots, "unique_searches", direction, _to_number(m.group("num")), cmp=cmp_raw
    )


class RegexEntityExtractor:
    """Deterministic offline entity extractor conforming to the EntityExtractor protocol."""

    def __init__(
        self,
        config: QIL0RegexEntityConfig,
        hard_entity_names: FrozenSet[str],
        soft_slot_names: FrozenSet[str],
        known_tlds: Optional[FrozenSet[str]] = None,
    ) -> None:
        """Construct extractor.

        :param config: QIL0RegexEntityConfig - Required regex L0 config (incl. source_tag)
        :param hard_entity_names: FrozenSet[str] - From qi.entity_slots.hard_entity_names
        :param soft_slot_names: FrozenSet[str] - From qi.entity_slots.soft_slot_names
        :param known_tlds: Optional[FrozenSet[str]] - TLD allowlist; empty set when omitted
        """
        if not isinstance(hard_entity_names, frozenset):
            raise ConfigurationError(
                "RegexEntityExtractor requires hard_entity_names frozenset from qi.entity_slots"
            )
        if not isinstance(soft_slot_names, frozenset):
            raise ConfigurationError(
                "RegexEntityExtractor requires soft_slot_names frozenset from qi.entity_slots"
            )
        if not hard_entity_names:
            raise ConfigurationError(
                "RegexEntityExtractor hard_entity_names must be non-empty"
            )
        if not soft_slot_names:
            raise ConfigurationError(
                "RegexEntityExtractor soft_slot_names must be non-empty"
            )
        self._config = config
        self._known_tlds: FrozenSet[str] = frozenset(
            t.lower() for t in (known_tlds or ())
        )
        self._hard_names: FrozenSet[str] = hard_entity_names
        self._soft_slots: FrozenSet[str] = soft_slot_names

    def _extract_slots(self, query: str) -> Dict[str, object]:
        """Run every slot-family parser over the query and return the accumulated slot dict."""
        # Normalize common surface typos before parsers (fintec->fintech, domians->domains).
        for rx, repl in _QUERY_TYPO_SUBS:
            query = rx.sub(repl, query)
        # Strong advisory/analytics -> [] even with TLD/price tokens (LLM rule 16).
        if is_strong_advisory(query):
            return {}
        # Soft advisory/guidance with no inventory bound -> [] (L0 / grounding parity).
        slots: Dict[str, object] = {}
        if is_soft_advisory_no_inventory(query):
            return {}
        # Dual qualitative "high X and high Y" with no numbers -> [] (rule-4 omit).
        if re.search(
            r"\bhigh\s+(?:go\s*value|govalue|domain\s+authority|da|authority|"
            r"traffic|backlinks?|valuation)\b"
            r".{0,48}\bhigh\s+(?:go\s*value|govalue|domain\s+authority|da|"
            r"authority|traffic|backlinks?|valuation)\b",
            query,
            re.IGNORECASE,
        ) and not re.search(r"\d", query):
            return {}
        _parse_tld(query, slots, self._known_tlds)
        _parse_auction(query, slots)
        _parse_keywords(query, slots)
        _parse_keyword_advanced(query, slots)
        _parse_char_constraints(query, slots)
        _parse_boolean_flags(query, slots)
        _parse_time_remaining(query, slots)
        _parse_lifecycle(query, slots)
        _parse_char_pattern_and_gem(query, slots)
        _parse_similar_to(query, slots)
        _parse_recency(query, slots)
        _parse_owner_members(query, slots)
        _parse_unique_searches(query, slots)
        _parse_numeric(query, slots)
        _parse_currency(query, slots)
        _parse_spelled_measure(query, slots)
        _parse_spelled_numeric(query, slots)
        _parse_extra_llm_parity_cues(query, slots)
        _reconcile_keyword_contains_exclusion(slots)
        return slots

    async def classify_async(self, query: str) -> Optional[IntentSlice]:
        """Extract hard + soft entities; partition via qi.entity_slots.

        Soft keyword/topic patterns live here. Engine activates this extractor
        only when LLM L0 is unavailable (fallback_only_when_llm_unavailable).
        """
        if not self._config.enabled or not query:
            return None
        slots = self._extract_slots(query)
        hard: List[Entity] = []
        soft: List[Entity] = []
        for name, value in slots.items():
            ent = Entity(
                name=name,
                value=value,
                confidence=self._config.confidence,
                source=self._config.source_tag,
                chip_kind=infer_chip_kind(name, self._hard_names),
            )
            if name in self._soft_slots:
                soft.append(replace(ent, chip_kind="soft"))
            else:
                hard.append(ent)
        # Cap total hard+soft.
        cap = int(self._config.max_entities)
        if len(hard) + len(soft) > cap:
            soft_budget = max(0, cap - len(hard))
            soft = soft[:soft_budget]
            if len(hard) > cap:
                hard = hard[:cap]
                soft = []
        if not hard and not soft:
            return None
        logger.info(
            f"regex_entity_extract_done query_len={len(query)} hard={len(hard)} "
            f"soft={len(soft)} slots={sorted(slots.keys())}"
        )
        return IntentSlice(
            query_type="hybrid",
            entities=hard,
            confidence=1.0,
            raw_text=query,
            soft_entities=soft,
        )
