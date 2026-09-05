"""L0 LLM filter extractor — single path for qie_only, full-search, and grounding.

Single-call extract: FIND-63 API filters **and** soft/local chips (keyword_*,
topic_*, lifecycle_*, buy_it_now, word_count_*, …) — same catalog/prompt the
offline grounding path in ``extract_test_filters_queries.py`` uses.

- ``extract()`` -> identified_filters dicts (qie_only / script)
- ``classify_async()`` -> IntentSlice hard/soft entities (full-search QI engine)

FIND names come from central ``FIND_FILTERABLE_API_PARAMS_ORDERED``; soft/local
slots are the shared allowlist below (not a separate prompt fork). Full-search
must not use a second parallel-group LLM entity extractor.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from llm_core.pricing import compute_call_cost_usd

from semantic_search.config.models import QIEntitySlotsConfig
from semantic_search.contracts import Entity, IntentSlice
from semantic_search.core.exceptions import ConfigurationError, LLMError
from semantic_search.core.logging_utils import get_logger
from semantic_search.qi.advisory_patterns import (
    is_soft_advisory_no_inventory,
    is_strong_advisory,
)
from semantic_search.qi.budget_price_cues import parse_budget_prefixed_price
from semantic_search.qi.llm_classifier import infer_chip_kind
from semantic_search.qi.llm_entity_extractor import (
    _API_INVERT_BOOL,
    _API_TO_INTERNAL,
    VALID_ENTITY_NAMES,
)
from semantic_search.qi.slot_to_api_param import (
    FIND_FILTERABLE_API_PARAMS_ORDERED,
    normalize_type_include_list_for_public,
    transform_slot_value,
)

logger = get_logger(__name__)

# Bare ``gd`` = GoDaddy (registrar / auction / seller). Shared by grounding,
# qie_only, and full-search so all three paths see the same surface form.
_GD_TOKEN_RE = re.compile(r"\bgd\b", re.IGNORECASE)


def expand_gd_to_godaddy(text: str) -> str:
    """Rewrite bare ``gd`` / ``GD`` -> ``godaddy`` (word-boundary; keeps ``godaddy`` intact).

    ``gd transfer`` becomes ``godaddy transfer`` which still maps to ``gd_transfer``.
    """
    if not isinstance(text, str) or not text:
        return text
    return _GD_TOKEN_RE.sub("godaddy", text)


# Slots that expect list values after API->internal mapping (pipe/CSV from LLM).
_MULTI_VALUE_SLOTS: frozenset = frozenset(
    {
        "tld",
        "auction_type",
        "tldExcludeList",
        "typeExcludeList",
        "keyword_contains",
        "keyword_starts_with",
        "keyword_ends_with",
        "keyword_contains_exclude",
        "topic_include",
        "topic_exclude",
    }
)

# Local (non-FIND) slots taught by the parameter hints. Allowlisted so the model
# may emit them — chip_kind hard/soft still comes from qi.entity_slots
# (e.g. keyword_contains_exclude is hard). Matches extract_test_filters_queries
# grounding + qie_only identified_filters.
_LOCAL_SOFT_PARAMS: Tuple[str, ...] = (
    "buy_it_now",
    "has_reserve_price",
    "lifecycle_state",
    "lifecycle_disjunction",
    "keyword_contains",
    "keyword_starts_with",
    "keyword_ends_with",
    "keyword_phrase",
    "keyword_match_mode",
    "keyword_contains_exclude",
    "word_count_min",
    "word_count_max",
    "price_below_market",
    "has_web_traffic_signal",
    "domain_age_is_unknown",
    "traffic_is_unknown",
    "gd_transfer",
    "similar_to",
    "topic_include",
    "topic_exclude",
)


def _chip_kind_for_param(name: str, soft_slot_names: frozenset) -> str:
    """Return ``hard`` / ``soft`` from ``qi.entity_slots.soft_slot_names`` (internal slots)."""
    internal = _API_TO_INTERNAL.get(name, name)
    return "soft" if internal in soft_slot_names else "hard"


_seen: set = set()
FILTERABLE_PARAMS: List[str] = []
for _name in list(FIND_FILTERABLE_API_PARAMS_ORDERED) + list(_LOCAL_SOFT_PARAMS):
    if _name not in _seen:
        _seen.add(_name)
        FILTERABLE_PARAMS.append(_name)

# Allowlist for post-parse keep (FIND + soft/local).
_FILTERABLE_PARAM_SET = frozenset(FILTERABLE_PARAMS)

L0_FILTER_SYSTEM = (
    "You are a domain marketplace search filter extraction expert. "
    "Given natural language search queries, identify which FIND API filter parameters apply "
    "and what values they should have. "
    "Return ONLY valid JSON — no explanation, no markdown fences, no extra text."
)

# Full user prompt: FIND-63 + soft/local catalog + hints + queries + JSON contract.
# Keep in sync with extract_test_filters_queries grounding (imports this prompt).
L0_FILTER_BATCH_PROMPT = """\
For each query below, identify which FIND API filter parameters apply and their values.

L0 extract contract (always satisfy):
- G1 FIND-63 hard filters: emit every cued hard param from the known list; never invent off-catalog params.
- G2 Soft/local chips: emit topics, patterns, phrases, categories (topic_*, keyword_*, lifecycle, …) when cued.
	- G3 Keywords: ONLY industry/niche nouns (coffee, fintech, …) with probability in [0.0,1.0]. Default keywords=[]. Abstain on browse/quality/filter-only/meta. Never TLD/inventory/quality adjectives.
- G4 Price+currency: when a numeric price bound and currency cue both appear, emit BOTH maxPrice/minPrice
  AND filterPriceCurrency (e.g. "under EUR 80" -> maxPrice=79 + filterPriceCurrency=EUR;
  "under GBP 95" -> maxPrice=94 + filterPriceCurrency=GBP). Never emit currency alone when a number is present.
- G5 Analytics / advice: aggregate counts, trends, "how many", "should I…", "red flags", category-vs-category
  advice with NO numeric/TLD inventory bound -> emit ZERO filters (empty filters array).

Known filterable parameters:
{params}

Parameter hints (use these to recognize intent in natural language):
- charPattern: consonant/vowel/digit structure of SLD e.g. "cvcv" or "vcc". Triggers on: "consonant vowel pattern", "cvvc domains", "numeric pattern", "4-char pattern like LLLL or CCVC".
- typeIncludeList: auction type filter. Values: closeout, expiry, backorder, listed, dutch, godaddy, partner, premium, firehose, dropcatch.
  Triggers on: "closeout only", "expiring auctions", "backorder domains", "godaddy auctions", "gd auctions" (gd=godaddy), "partner auctions".
  NEVER put godaddy/gd in keyword_contains.
- typeExcludeList: exclude auction types. Same values as typeIncludeList.
- startTimeAfter: recently listed. Emit relative offset ONLY as "-<N>h" or "-<N>d"
  (e.g. "last 48 hours"->"-48h", "last 3 days"->"-3d", "this week"->"-7d", "just listed"/"new today"->"-1d").
  NEVER prose like "last 48 hours" / "recent" / true.
- startTimeBefore: auctions that started before a date. Triggers on: "old listings", "listed long ago". Absolute ISO when date given.
- ownerMemberIncludeList: filter by specific seller/owner member ID or handle. Triggers on: "from seller X", "listed by member X", "by owner X", "from specific seller", "from user X".
- ownerMemberExcludeList: exclude specific sellers. Triggers on: "not from X", "exclude seller X".
- minUniqueSearches: GoDaddy unique-searcher demand ONLY. Triggers on: "unique searches", "unique searchers", "N unique searches", "N searches per month", "search volume of N".
  Disambiguation: "N unique searches" / "N searches per month" / "search volume of N" -> minUniqueSearches.
  "N visitors" / "N hits" / "N traffic" per month -> minTraffic, NOT minUniqueSearches.
  minTraffic = raw session/daily/visitor/hit traffic count. Never use minTraffic for "unique searches" or "searches per month".
  Do NOT use for "semrush search volume" (that is minSemrushSearchVolume). Qualitative "good search demand" without a number -> omit.
- minLetters: minimum alphabetic letter COUNT in SLD (not total length). Triggers on: "minimum N letters", "at least N letters", "N letters or more". Do NOT also emit minSldLen for the same cue.
- minSldLen: minimum total SLD character length. Triggers on: "longer names N chars", "N chars plus", "N characters or more", "at least N chars".
  Floor only -- do NOT also emit maxSldLen unless an exact length is stated.
  Bare "short" / "short names" with NO numeric floor cue -> NEVER emit minSldLen (ceiling-only via maxSldLen).
- maxSldLen: maximum SLD character length. Length unit REQUIRED.
  Bare "short" / "short names" (no "N chars") -> maxSldLen=5 ONLY.
  EXCLUSIVE (<N): "under N chars" / "less than N characters" / "shorter than N" -> maxSldLen=N-1 (e.g. "under 5 chars"->4).
  INCLUSIVE (<=N): "at most N chars" / "N chars or less" / "max N chars" / "max N letters" / "nothing over N chars" -> maxSldLen=N ONLY.
  "N letter names" / "N-letter" exact -> minSldLen=N AND maxSldLen=N.
  CRITICAL: bare "under 500" / ".io names under 500" is maxPrice -- NEVER emit maxSldLen.
  Do NOT invent maxLetters (not a valid param -- use maxSldLen). Do NOT invent maxSldLen from word-count cues.
- typeIncludeList: auction-type filter. Bare "premium" (quality signal) without "auction"/"listing" -> omit. Do NOT map bare premium -> 16|38|39.
- maxPrice: ONLY with a numeric price bound. Bare "cheap"/"affordable" -> omit price.
  NEVER emit maxPrice=0 for qualitative "low price" / "price is low" without a numeric cue.
  Qualitative cheap/low price without a number -> price_below_market=true instead.
  Exclusive ceilings: "under/below/less than N" -> maxPrice=N-1 for ALL scales ("under 500"->499, "under 1k"->999, "under 3k"->2999).
  Inclusive: "at most"/"no more than"/"capped at N" -> maxPrice=N.
  "young or unknown age under 500"->maxPrice=499 ONLY (not maxAge, not maxSldLen).
- minPrice: price floor. "over N"/"above N"/"more than N"/"greater than N" -> minPrice=N+1. "at least N"/"N or more" -> minPrice=N.
  NEVER use maxPrice for over/above floor cues.
- minAge/maxAge: registration age in YEARS. Requires years unit ("older than 10 years", "aged 15 years min").
  NEVER bind bare "under N" (price) to maxAge. Qualitative "young" alone -> omit.
  "unknown age" alone -> domain_age_is_unknown (not maxAge).
- domain_age_is_unknown: domain age not available. Triggers on: "unknown age", "age unknown".
- traffic_is_unknown: traffic data not available. Triggers on: "unknown traffic", "traffic unknown", "no traffic data".
- minBids/maxBids: bid COUNT. EXCLUSIVE: "more than 15 bids"->minBids=16;
  "fewer than 5 bids"/"less than 5 bids"/"under 5 bids"->maxBids=4.
  INCLUSIVE: "at least 10"/"10 or more"->minBids=10; "at most 5"->maxBids=5.
- keyword_contains_exclude: ONLY on explicit exclude cues ("no X", "without X"). NEVER invent antonyms from "short"->"long".
- word_count_min/word_count_max: "single word"/"two word" only. Do NOT also emit maxSldLen.
- minSemrushDomainsNum: SEMrush referring-domains floor. Triggers on: "semrush ref domains above N", "ref domains N plus semrush".
  Do NOT invent minSemrushRefDomains -- use minSemrushDomainsNum (Majestic ref domains use minMajesticRefDomains).
- excludeLetters: domains with no letters (pure numeric). Triggers on: "no letters", "numbers only", "numeric domains", "pure digit", "digits only", "digit-only", "with digits only", "numeric only".
- excludeDigits: domains with no digits. Triggers on: "no numbers", "letters only", "no digits", "pure alpha".
- excludeHyphens: domains without hyphens. Triggers on: "no hyphens", "no dashes", "clean domains".
- tldExcludeList: TLD(s) to exclude. Triggers on: "exclude .xyz", "no .com", "skip .xyz", "exclude xyz domains" (bare known TLD after exclude speech-act: exclude/no/without/avoid/skip).
  Multiple excludes pipe-join: "no .info no .biz" -> tldExcludeList="info|biz", "exclude .xyz and .club" -> tldExcludeList="xyz|club".
  Pipe-join multiple: tldExcludeList="com|net".
- lifecycle_state: lifecycle status string(s). Values: active, pending_delete, expired, deleted, ...
  Pipe-join when OR'd: "active and pending delete" -> lifecycle_state="active|pending_delete" AND lifecycle_disjunction=true.
- lifecycle_disjunction: bool ONLY (true/false). Set true when query ORs lifecycle statuses. NEVER put status names in lifecycle_disjunction.
- has_reserve_price: auction has a reserve price set. Triggers on: "with reserve", "reserve price", "has reserve", "reserve set".
- buy_it_now: buy-it-now / BIN / fixed price / "instant buy" / "dont want to bid just buy". Do NOT emit for bare "ending soon i want to buy" (urgency ≠ BIN) unless buy-now language is present.
- price_below_market: "below market"/"underpriced"/"good deal"/"value pick"/"feels underpriced". "like X but cheaper" -> price_below_market=true (not a price ceiling unless a number is stated).
- has_web_traffic_signal: domain has web traffic data. Triggers on: "has traffic", "with web traffic", "traffic signal".
- keyword_contains: ONLY on explicit contain cues ("contains X", "with the word X", "including X in the name").
  NEVER for bare "<tld> domains", industry topics ("fintech app"), or brand-style ("figma style") --
  those are tldIncludeList / topic_include, not keyword_contains.
- keyword_starts_with: SLD starts with prefix. Triggers on: "starts with X", "beginning with X", "prefix X".
- keyword_ends_with: SLD ends with suffix. Triggers on: "ends with X", "suffix X", "ending in X".
- keyword_phrase: SLD matches a phrase. Triggers on: "phrase X", "exact phrase".
- keyword_match_mode: how keyword matching works (exact, prefix, suffix, contains).
  ALWAYS emit keyword_match_mode alongside any keyword_contains/keyword_starts_with/keyword_ends_with. If ambiguous, default to keyword_match_mode="contains".
- topic_include: industry/theme without a contain cue: "fintech app", "figma style naming", "health tech".
- word_count_min/word_count_max: min/max word count in domain. Triggers on: "one-word domains", "two-word domains", "multi-word".
- gd_transfer: GoDaddy transfer-ready domains. Triggers on: "gd transfer", "godaddy transfer", "transfer ready".
  Bare "gd"/"GD" ALWAYS means GoDaddy (registrar) -- never a keyword;
  "gd auctions"->typeIncludeList=godaddy; "from gd"/"gd seller"->ownerMemberIncludeList=GoDaddy; "gd transfer"->gd_transfer=true.
- isBidAccepted: auction where a bid has been accepted. Triggers on: "bid accepted", "accepted bid", "offer accepted".
- isExtended: auction that has been extended. Triggers on: "extended auction", "last minute extension", "auction extended".
- minMajesticTrustFlowScore / maxMajesticTrustFlowScore: TF / trust flow ONLY. "tf 30 plus" -> minMajesticTrustFlowScore (NOT minTraffic / minSldLen).
- minMajesticCitationFlowScore / maxMajesticCitationFlowScore: CF / citation flow ONLY. "cf 30 plus" -> minMajesticCitationFlowScore (NOT minSldLen).
- minMajesticRefDomains: Majestic referring-domains floor ONLY. Triggers on: "ref domains", "referring domains", "majestic ref domains", "ref domains above N", "N referring domains".
  Bare "ref domains"/"referring domains" (no "semrush" qualifier) -> minMajesticRefDomains, NOT minSemrushDomainsNum.
- minValuationPrice / maxValuationPrice: govalue / appraised / worth / estimated value ONLY. "govalue above 2000" -> minValuationPrice (NOT Estibot / SEMrush / unique searches).
  Qualitative govalue/valuation presence (no number) -> minValuationPrice=1 (never 0). "govalue gap", "price vs valuation" -> minValuationPrice=1.
- minSemrushSearchVolume / maxSemrushSearchVolume: SEMrush "search volume" ONLY. Do NOT map to minUniqueSearches.
- minSemrushAScore: "semrush authority" / "authority score" / "AS above N".
  "high DA/authority/ascore" -> minSemrushAScore=60; "decent/moderate DA" -> minSemrushAScore=30; bare "domain authority" presence / "any DA" -> minSemrushAScore=0.
- minSemrushUrlsNum: "indexed pages".
- minSemrushLinksTotal: "semrush links" / "semrush backlinks".
- minSemrushDomainsNum: "semrush ref domains" / "semrush referring domains".
- minEstibotDomainCount: Estibot domain-count ONLY (must say estibot). Never map govalue here.
- minEstibotDomainCountDev: Estibot DEV-extension domain-count ONLY. Triggers on: "dev ext count", "dev extension count", "dev namespace count", "dev tld count", "dev ext saturation".
  Distinct from minEstibotDomainCount (plain, non-dev) -- the "dev ext(ension)"/"dev namespace"/"dev tld" cue is required.
- filterPriceCurrency: ALWAYS emit when any currency cue is present — never omit.
  "$50"/"under 50 dollars"/"USD"/"US dollars"->USD;
  "€"/"euro"/"euros"/"EUR"->EUR;
  "£"/"pound"/"pounds"/"GBP"->GBP;
  "¥"/"yen"/"JPY"->JPY;
  "₹"/"rupee"/"rupees"/"INR"->INR;
  "CAD"/"Canadian dollars"->CAD; "AUD"/"Australian dollars"->AUD.
  Prefer the non-USD code when both a symbol and a word appear.
- similar_to: brand/name similarity. Triggers on: "like stripe", "similar to linear", "sounds like vercel", "names like slack or zoom". Pipe-join concrete brand tokens ONLY.
  NEVER emit abstract placeholders ("existing_tech_companies", "modern_brands"). If no concrete brand named, omit similar_to.
  NEVER use keyword_contains for bare "like X".
- has_web_traffic_signal (visitor soft): ALSO "real visitors", "existing traffic", "with traffic", "has visitors"
  when NO numeric traffic bound. Numeric "traffic 1000+" / "N+ traffic" -> minTraffic (not this soft alone).
- endTimeBefore / ending soon: "ending soon", "ending tonight", "closing tonight", "ending this week" -> relative end bound.
  Prefer same offset style as startTimeAfter when emitting endTimeBefore ("tonight"/"soon"->"-1d", "this week"->"-7d").
  "closing in the next N hours"/"ending in the next N hours" -> endTimeBefore="-<N>h"; "closing/ending in the next N days" -> endTimeBefore="-<N>d".
  Urgency != buy_it_now.
- minPrice+maxPrice range: "between A and B", "from A to B", "A to B" (price) -> BOTH minPrice=A and maxPrice=B. "capped at N" -> maxPrice=N. "exactly under N" -> exclusive maxPrice=N-1.
- around / mid budget: "around 1500", "mid budget ~1k" with NO hard range -> OMIT hard price (rule 4). Do not invent minPrice=maxPrice.
- letter-exact length: "three/four/five letter", "N-letter", "N letter .com" -> minSldLen=N AND maxSldLen=N (+ tld when stated). Not minLetters unless letter-count cue.
- word_count synonyms: "one word", "1-word", "single word" -> word_count_min=1 and word_count_max=1; "two word"/"2-word" -> 2/2. Do NOT also emit maxSldLen.
- price_below_market extras: ALSO "hidden gem(s)", "gems only", "gem domains", "sleeper",
  "sleeper picks", "undervalued", "below market value", "flip potential",
  "fraction of what it's worth" alone (no number) -> price_below_market=true.
  Never invent isGemDomain / valuation floors from these.
- minSemrushAScore DA alias: "DA", "domain authority", "high DA" WITH a number -> minSemrushAScore. Qualitative "high da" alone -> omit (rule 4).
- topic_exclude / keyword_contains_exclude vibe: "no crypto vibe", "no X vibe", "without X words",
  "exclude X topic", "avoid X" -> keyword_contains_exclude (or topic_exclude for industry). Explicit exclude cue required.
  "exclude"/"avoid" ALWAYS wins over topic_include even when followed by a niche/inventory word
  ("exclude crypto topic" -> topic_exclude=crypto, NOT topic_include=crypto).
- pending_delete / dropping / backorder: "pending delete" -> lifecycle_state=pending_delete; "dropping" / "backorder" / "buy now expired" -> typeIncludeList / buy_it_now as cued (existing value sets).
- topic_include niches: "devtools", "climate tech", "edtech", "fintech", "b2b saas" without contain cue -> topic_include (not keyword_contains).
- most viewed / watchlist / trending: popularity/browse language with NO numeric bid/price/time bound -> filters []. Do NOT invent minBids=1.
- advisory / guidance: "should I…", "what makes…", "advice", "recommend a strategy" without inventory bounds -> filters [].

	Keyword extraction (separate from filters — false positives are worse than empty):
	DEFAULT: keywords=[]. Emit a keyword ONLY when it is a concrete industry / product /
	niche / brand noun a buyer would search for (coffee, pizza, fintech, climate, edtech,
	ecommerce, startup, cloud, logistics, gaming, stripe, figma, …).
	If that niche also maps to topic_include / similar_to, STILL emit the niche/brand in
	keywords (dual-OK). KEEP the niche even when price/TLD/traffic/ending-soon filters
	also apply ("ecommerce domains with traffic" -> keywords=[ecommerce]).
	Assign probability in [0.0,1.0]; order descending. Prefer >= 0.85 for kept niches.
	
	ABSTAIN (keywords MUST be []) only when NO niche/brand noun exists — including:
	- Browse / quality-only: short, catchy, premium, brandable, luxury, punchy, clean,
	  minimal, strong brand, easy to say/spell, memorable, one-word, two-syllable
	  with no industry/brand noun.
	- Filter/inventory-only: price/TLD/bids/traffic/backlinks/govalue/age/ending soon /
	  pending delete / auction type with no industry/brand noun.
	- Meta / advice: "should I…", "what makes…", strategy / evaluation questions.
	
	NEVER put in keywords (route to filters or drop):
	- TLD tokens alone: ai, io, com, net, org, app, dev, xyz, … (bare "ai domain" /
	  "premium ai domain" with no other niche -> keywords=[]; "ai startup" -> [startup]).
	- Domain fillers: domain, domains, website, url, name, names, brand (as "brandable").
	- Auction/inventory metrics: bid, bids, ending, soon, now, today, week, traffic,
	  backlinks, govalue, worth, average, count, authority, visitors, professional, category.
	- Quality/marketing adjectives listed under ABSTAIN.
	- Numerics, stop words, buy/sell/find/search/register.
	
	Empty keywords is CORRECT for browse/filter-only/meta. Do not invent a keyword to
	avoid []. Never force a single "primary" pick.
	Example KEEP: "I want to make coffee pizza and sell it to everyone. The domain must be
	below $100." -> keywords: [{{"term": "coffee", "probability": 0.95}}, {{"term": "pizza", "probability": 0.93}}]
	Example KEEP: "ecommerce domains with existing traffic" -> keywords: [{{"term": "ecommerce", "probability": 0.9}}]
	Example KEEP: "ai startup domain under 1500 with some traffic" -> keywords: [{{"term": "startup", "probability": 0.9}}]
	Example ABSTAIN: "short catchy .io names below 500" -> keywords: []
	Example ABSTAIN: "premium ai domain with real visitors" -> keywords: [] (ai=TLD; premium/visitors≠niche)
	Example ABSTAIN: "domains with more than 15 bids ending soon" -> keywords: []

Return JSON exactly in this shape:
{{
  "results": [
    {{"idx": 1, "filters": [{{"param": "paramName", "value": "extractedValue"}}], "keywords": [{{"term": "coffee", "probability": 0.95}}]}},
    {{"idx": 2, "filters": [], "keywords": []}}
  ]
}}

Rules:
- One entry per query, idx matches the query number above.
- Use exact param names from the known list. Do not invent params (no maxLetters).
- If no filter applies set filters to [].
- Do not add params not in the list.
- Use the parameter hints above to recognize non-obvious phrasings.
- keywords is independent of filters — apply the keyword-extraction rules above, not the filter param hints.

Critical disambiguation (apply before emitting):
1) TLD vs keyword/topic: bare "<tld> domains" / "<tld> domains only" / "<tld> under N" /
   "app or dev extension" -> tldIncludeList (ai, io, com, app, dev, ...). Do NOT emit topic_include
   for those bare TLD inventory cues. Niche/premium modifiers keep topic
   ("premium ai domain", "extended ai auction" -> topic_include=ai).
   keyword_contains ONLY with an explicit contain cue.
2) Numeric floors -- EXCLUSIVE (>N): "above N" / "over N" / "more than N" / "greater than N" -> min=N+1.
   INCLUSIVE (>=N): "at least N" / "N or more" / "N plus" / "N+" -> min=N.
   Same for ceilings: "under N"/"below N"/"less than N" -> max=N-1 (all scales, incl. under 1k->999);
   "at most N"/"no more than N"/"capped at N" -> max=N.
3) Metric families never cross-map: TF->MajesticTrustFlow; CF->MajesticCitationFlow;
   govalue/worth/appraised->ValuationPrice; SEMrush search volume->minSemrushSearchVolume;
   unique searchers->minUniqueSearches; traffic/visitors->minTraffic;
   bid count->minBids/maxBids; current bid/price->minPrice/maxPrice.
4) Qualitative with no number ("high", "low", "good", "strong", "lots", "proven", "cheap" alone) -> OMIT that numeric filter (do not emit 1/0/true/high/low/none as a stand-in).
5) Relative listing age -> startTimeAfter as "-<N>h"/"-<N>d" only.
   Registration age in years -> minAge/maxAge.
   "stale over 2 weeks" -> startTimeBefore or listing-age via startTimeAfter="-14d" floor semantics:
   prefer startTimeAfter omitted + do NOT use minAge for weeks.
6) "letters only" / "short letters only" with buy-now -> buy_it_now + excludeDigits
   (and maxSldLen only if a number is stated).
   "N letter" exact length -> minSldLen=N + maxSldLen=N (not minLetters alone unless "letters" count is explicit).
7) Multi-value lists: pipe-join sorted unique tokens (tldIncludeList="app|dev"). Booleans must be true/false.
8) When price + valuation both stated ("govalue above 5k … under 2k") emit BOTH minValuationPrice and maxPrice (not maxBids).
9) Registrar alias: bare "gd"/"GD" = GoDaddy (same as "godaddy" / "go daddy").
   Map auction/seller cues to typeIncludeList=godaddy or ownerMemberIncludeList=GoDaddy.
   "gd transfer"/"godaddy transfer" -> gd_transfer=true (not keyword_contains).
10) Visitor language: "real visitors" / "existing traffic" (no number) -> has_web_traffic_signal=true. Numeric traffic/visitors -> minTraffic. Never map visitors -> bids or valuation.
11) Letter-count exact: "N letter .com" / "four letter" -> minSldLen=N + maxSldLen=N (+ tldIncludeList when TLD stated). Use minLetters ONLY for letter-count floors ("at least N letters").
12) Price ranges: "between A and B" / "from A to B" -> minPrice=A and maxPrice=B. "around N" / "mid budget" alone -> omit hard price. "capped at N" -> maxPrice=N.
13) Auction urgency vs BIN: "ending tonight/soon" -> end time bound only. buy_it_now ONLY with buy-now / BIN / instant-buy language.
14) Similarity vs keyword: "like stripe" / "similar to notion" / "sounds like vercel" -> similar_to (pipe-join). Not keyword_contains unless an explicit contain cue appears.
15) Investor fluff: "hidden gem" / "sleeper" / "flip potential" alone -> price_below_market=true OR []. Never invent isGemDomain or numeric valuation/SEO floors without numbers.
16) Advisory queries: guidance / "should I…" / "what makes…" / strategy advice -> filters=[] unless an explicit inventory bound appears (price, TLD, expired, traffic N, …).
17) DA alias: "DA" / "domain authority" + number -> minSemrushAScore. Qualitative "high da" -> omit (rule 4).
18) Conditional price: "below 1500 or 2000 if backlinks…" -> emit the lower ceiling only (maxPrice for 1500 path). Do not invent dual maxPrice values.

Few-shot examples (shape only — follow rules above; do not copy blindly when cues differ):
- "climate tech domains added recently" -> topic_include=climate_tech, startTimeAfter="-7d"
- "four letter .com no numbers" -> minSldLen=4, maxSldLen=4, tldIncludeList=com, excludeDigits=true
- "domains between 500 and 1000" -> minPrice=500, maxPrice=1000
- "domains like stripe or plaid" -> similar_to=plaid|stripe
- "openai style name no ai or gpt words" -> similar_to=openai, keyword_contains_exclude=ai|gpt
- "auction domains ending tonight under 1k" -> endTimeBefore="-1d", maxPrice=999
- "closing in the next 6 hours" -> endTimeBefore="-6h"
- "dev ext count above 30" -> minEstibotDomainCountDev=31
- "domains below market value" -> price_below_market=true
- "gems only" -> price_below_market=true
- "pending delete domains" -> lifecycle_state=pending_delete
- "ai domains only" -> tldIncludeList=ai
- "premium ai domain with real visitors" -> has_web_traffic_signal=true, topic_include=ai
- "cloud platform domain traffic 1000+" -> minTraffic=1000, topic_include=cloud
- ".io domains capped at 500" -> tldIncludeList=io, maxPrice=500
- "fintech startup domain under 2k no crypto vibe" -> topic_include=fintech, maxPrice=1999, keyword_contains_exclude=crypto
- "exclude crypto topic" -> topic_exclude=crypto
- "avoid flagged accounts" -> filters=[] (no inventory bound; "flagged accounts" is not a domain param)
- "govalue above 5k current bid under 2k" -> minValuationPrice=5001, maxPrice=1999 (the "above" cue belongs to govalue, NOT price -- "under 2k" is still a price ceiling, never minPrice)
- "tf above 20 short com under 1500" -> minMajesticTrustFlowScore=21, tldIncludeList=com, maxSldLen=5, maxPrice=1499
- "brandable coffee domains under EUR 80 .com" -> maxPrice=79, filterPriceCurrency=EUR, tldIncludeList=com, keywords=[coffee]
- "pizza restaurant domains under GBP 95 .io" -> maxPrice=94, filterPriceCurrency=GBP, tldIncludeList=io, keywords=[pizza, restaurant]

Queries:
{numbered_queries}
"""


class _L0Filter(BaseModel):
    param: str = Field(min_length=1)
    value: Any = None


class _L0Keyword(BaseModel):
    term: str = Field(min_length=1)
    probability: float = Field(ge=0.0, le=1.0)


class _L0ResultEntry(BaseModel):
    idx: int
    filters: List[_L0Filter] = Field(default_factory=list)
    keywords: List[_L0Keyword] = Field(default_factory=list)


class _L0BatchResponse(BaseModel):
    results: List[_L0ResultEntry] = Field(default_factory=list)


class _L0CombinedResultEntry(BaseModel):
    idx: int
    rewritten_query: str = Field(min_length=1, max_length=512)
    transformed: bool = False
    filters: List[_L0Filter] = Field(default_factory=list)
    keywords: List[_L0Keyword] = Field(default_factory=list)


class _L0CombinedBatchResponse(BaseModel):
    results: List[_L0CombinedResultEntry] = Field(default_factory=list)


# Combined rewrite+extract: same catalog/hints as extract-only; JSON includes rewrite.
L0_COMBINED_BATCH_PROMPT = """\
For each query below (JSON only — no prose outside JSON; do not omit fields to stay brief).
Satisfy G1-G4 (FIND-63 hard, soft chips, keywords, price+currency pairing) on the rewritten query.
1) Rewrite verbose natural-language domain search into a concise search query.
   Drop filler and polite words. Keep every topical niche word, domain extension,
   price bound, and currency code/symbol (USD/EUR/GBP/INR/CAD/AUD/JPY/$/€/£/₹/¥).
   Set transformed=true only when the rewrite differs from the input.
	2) Identify FIND API filter parameters AND topical keywords for the REWRITTEN query ONLY.
	   Keywords: emit ONLY concrete industry/niche nouns (coffee, pizza, fintech, …).
	   Default keywords=[]. Empty is CORRECT when the rewrite is browse/quality/filter-only.
	   Never extract filters/keywords from the raw verbose text when a rewrite is produced.
   If the query is already concise, set rewritten_query to the same text and transformed=false,
   then extract filters/keywords from that text.
   G4: always emit filterPriceCurrency with maxPrice/minPrice when a numeric price bound
   appears with a currency cue (under EUR 80 -> maxPrice=79 + filterPriceCurrency=EUR;
   under GBP 95 -> maxPrice=94 + filterPriceCurrency=GBP).

Known filterable parameters:
{params}

Parameter hints (use these to recognize intent in natural language):
""" + L0_FILTER_BATCH_PROMPT.split("Parameter hints (use these to recognize intent in natural language):\n", 1)[1].replace(
    """Return JSON exactly in this shape:
{{
  "results": [
    {{"idx": 1, "filters": [{{"param": "paramName", "value": "extractedValue"}}], "keywords": [{{"term": "coffee", "probability": 0.95}}]}},
    {{"idx": 2, "filters": [], "keywords": []}}
  ]
}}""",
    """Return JSON exactly in this shape:
	{{
	  "results": [
	    {{"idx": 1, "rewritten_query": "coffee pizza under 100", "transformed": true, "keywords": [{{"term": "coffee", "probability": 0.95}}, {{"term": "pizza", "probability": 0.93}}], "filters": [{{"param": "maxPrice", "value": 99}}, {{"param": "filterPriceCurrency", "value": "USD"}}]}},
	    {{"idx": 2, "rewritten_query": "short query", "transformed": false, "keywords": [], "filters": []}}
	  ]
	}}""",
)


def build_l0_filter_user_prompt(
    queries: List[Tuple[int, str]], params: Optional[List[str]] = None
) -> str:
    """Build the L0 filter user prompt for a numbered query batch (1-based idx, text)."""
    if not queries:
        raise ConfigurationError(
            "build_l0_filter_user_prompt requires at least one query"
        )
    param_list = params if params is not None else FILTERABLE_PARAMS
    numbered = "\n".join(f"{idx}. {expand_gd_to_godaddy(q)}" for idx, q in queries)
    return L0_FILTER_BATCH_PROMPT.format(
        params=", ".join(param_list),
        numbered_queries=numbered,
    )


def build_l0_combined_user_prompt(
    queries: List[Tuple[int, str]], params: Optional[List[str]] = None
) -> str:
    """Build the combined rewrite+extract L0 user prompt for a numbered query batch."""
    if not queries:
        raise ConfigurationError(
            "build_l0_combined_user_prompt requires at least one query"
        )
    param_list = params if params is not None else FILTERABLE_PARAMS
    numbered = "\n".join(f"{idx}. {expand_gd_to_godaddy(q)}" for idx, q in queries)
    return L0_COMBINED_BATCH_PROMPT.format(
        params=", ".join(param_list),
        numbered_queries=numbered,
    )


def parse_l0_filters_payload(
    raw: str, expected_idxs: List[int]
) -> Dict[int, List[Dict[str, Any]]]:
    """Parse L0 filter JSON (regex + loads). Keep FIND + soft/local allowlist only."""
    out: Dict[int, List[Dict[str, Any]]] = {i: [] for i in expected_idxs}
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        raise LLMError("l0_filter_parse_error: no JSON object in response")
    parsed = json.loads(json_match.group())
    for entry in parsed.get("results", []) or []:
        if not isinstance(entry, dict):
            continue
        idx = entry.get("idx")
        if idx is None:
            continue
        try:
            idx_i = int(idx)
        except (TypeError, ValueError):
            continue
        filters_raw = entry.get("filters") or []
        filters: List[Dict[str, Any]] = []
        if isinstance(filters_raw, list):
            for f in filters_raw:
                if not isinstance(f, dict):
                    continue
                param = str(f.get("param") or "").strip()
                if not param or param not in _FILTERABLE_PARAM_SET:
                    continue
                filters.append({"param": param, "value": f.get("value")})
        out[idx_i] = filters
    return out


# Qualitative stand-ins banned by L0 rule 4 — drop so qie_only / full / grounding agree.
# Do NOT include true/false — those coerce to bools for buy_it_now / flags.
# ---------------------------------------------------------------------------
# Scrub constants and helpers (added for L0 4-way parity)
# ---------------------------------------------------------------------------

_INDUSTRY_TOPIC_LEXICON: frozenset = frozenset(
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
        "tech",
        "seo",
        "developer",
        "brandable",
        "real_estate",
        "luxury",
    }
)

_STRONG_PBM_CUE_RE = re.compile(
    r"\b(?:below\s+market|underpriced|undervalued|hidden\s+gems?|gems?\s+only|"
    r"gem\s+domains?|sleeper(?:\s+picks?)?|flip\s+potential|but\s+cheaper|"
    r"cheaper\s+alternative|priced\s+way\s+below|fraction\s+of\s+what|"
    r"selling\s+for\s+a\s+fraction|bargain|overlooked|under\s+radar|"
    r"quick\s+flip|good\s+value\s+compared|resale\s+value|flip\s+later|"
    r"under\s+budget|worth\s+more\s+in\s+future)\b",
    re.IGNORECASE,
)

_LISTING_AGE_CUE_RE = re.compile(
    r"\b(?:just\s+listed|new\s+today|listed\s+today|listings?\s+today|"
    r"added\s+recently|listed\s+recently|recently\s+(?:added|listed)|"
    r"fresh\s+listings?|last\s+\d+\s+hours?|"
    r"(?:added|listed)\s+(?:in\s+)?(?:the\s+)?last\s+hour|last\s+hour|"
    r"this\s+week\s+(?:added|listed|new)|"
    r"(?:added|listed|new)\s+this\s+week|"
    r"fresh\s+listings?\s+this\s+week|"
    r"available\s+today|domains?\s+today|"
    r"what\s+dropped\s+today|since\s+yesterday|this\s+morning|"
    r"this\s+afternoon|new\s+arrivals)\b",
    re.IGNORECASE,
)

_ACTIVE_PENDING_CUE_RE = re.compile(
    r"\bactive\b.{0,40}\bpending\s+delete\b|\bpending\s+delete\b.{0,40}\bactive\b",
    re.IGNORECASE,
)
_EXPIRED_DROPPING_CUE_RE = re.compile(
    r"\bexpired(?:\s+domains?)?\b.{0,40}\b(?:or|and)\b.{0,40}\b(?:dropping|pending\s+delete)"
    r"|\b(?:dropping|pending\s+delete)\b.{0,40}\b(?:or|and)\b.{0,40}\bexpired(?:\s+domains?)?"
    r"|\bexpired\s+domains?\b.{0,40}\b(?:or|and)\b.{0,40}\bauction\b"
    r"|\bauction\b.{0,40}\b(?:or|and)\b.{0,40}\bexpired\s+domains?\b",
    re.IGNORECASE,
)

_PRICE_LIKE_UNDER_RE = re.compile(
    r"\b(?:under|below|less\s+than)\s+\$?\d[\d,]*(?:\.\d+)?\s*[kKmM]?\b",
    re.IGNORECASE,
)

# Leading industry lexicon token before inventory/browse cue -> topic_include.
# Niche set is _INDUSTRY_TOPIC_LEXICON; follower class is generic inventory grammar.
_LEADING_NICHE_TOPIC_RE = re.compile(
    r"\b(?P<topic>b2b\s+saas|climate\s+tech|real\s+estate|"
    r"saas|fintech|edtech|devtools|healthcare|cybersecurity|ecommerce|"
    r"gaming|legal|logistics|travel|cloud|crypto|startup|tech|seo|"
    r"developer|health|finance|luxury)\s+"
    r"(?:\.?[a-z]{2,10}|under|below|over|above|cheap|affordable|budget|"
    r"domain|name|brand|listing|space|category|topic|auction|extension|"
    r"audience|vertical|niche|segment|market)\b",
    re.IGNORECASE,
)
# Trailing audience/vertical class: "<industry> audience|vertical|niche".
_AUDIENCE_TOPIC_RE = re.compile(
    r"\b(?P<topic>b2b\s+saas|climate\s+tech|real\s+estate|"
    r"saas|fintech|edtech|devtools|healthcare|cybersecurity|ecommerce|"
    r"gaming|legal|logistics|travel|cloud|crypto|startup|tech|seo|"
    r"developer|health|finance|luxury)\s+"
    r"(?:audience|vertical|niche|segment|market)\b",
    re.IGNORECASE,
)
# Pronounceability / typeability class -> excludeDigits (not query literals).
# short+brandable co-occurrence matches char_constraint_rules.json has_number templates.
_EASY_SAY_DIGITS_RE = re.compile(
    r"\b(?:easy\s+to\s+(?:say|type|spell|pronounce)|(?:typeable|pronounceable)|"
    r"letters?\s+only|can\s+actually\s+type)\b"
    r"|\bshort\b.{0,32}\b(?:brandable|brandible|letters?\s+only|typeable|"
    r"easy\s+to\s+type|no\s+(?:numbers?|digits?))\b"
    r"|\b(?:brandable|brandible|letters?\s+only|typeable|"
    r"easy\s+to\s+type|no\s+(?:numbers?|digits?))\b.{0,32}\bshort\b",
    re.IGNORECASE,
)


_EXCLUDE_CUE_BEFORE_TOPIC_RE = re.compile(
    r"\b(?:exclude|excluding|no|without|non-?|not|avoid|avoiding)\s+$",
    re.IGNORECASE,
)


def _inject_leading_niche_topic(
    query: str,
    filters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Inject topic_include for leading niche / audience cues when LLM omitted it."""
    if not query:
        return filters
    if any(
        str(f.get("param") or "").strip() in ("topic_include", "topic_exclude")
        for f in filters
    ):
        return filters
    m = _LEADING_NICHE_TOPIC_RE.search(query) or _AUDIENCE_TOPIC_RE.search(query)
    if not m:
        return filters
    if _EXCLUDE_CUE_BEFORE_TOPIC_RE.search(query[: m.start()]):
        return filters
    tok = re.sub(r"\s+", "_", (m.group("topic") or "").strip().lower())
    if not tok or tok not in _INDUSTRY_TOPIC_LEXICON:
        return filters
    return list(filters) + [{"param": "topic_include", "value": tok}]


_EXPLICIT_NO_DIGITS_RE = re.compile(
    r"\b(?:no\s+(?:numbers?|digits?|numerals?|numbrs?)|without\s+(?:numbers?|digits?)|"
    r"letters?\s+only|pure\s+alpha)\b",
    re.IGNORECASE,
)


_EXPLICIT_NO_HYPHENS_RE = re.compile(
    r"\b(?:no\s+(?:hyphens?|hypens?|hyphns?|dashes?)|without\s+(?:hyphens?|dashes?)|"
    r"hyphen[-\s]?free)\b",
    re.IGNORECASE,
)


def _inject_easy_say_exclude_digits(
    query: str,
    filters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Force excludeDigits when typeability / easy-to-say cue present."""
    if not query or not (
        _EASY_SAY_DIGITS_RE.search(query) or _EXPLICIT_NO_DIGITS_RE.search(query)
    ):
        return filters
    if any(str(f.get("param") or "").strip() == "excludeDigits" for f in filters):
        return filters
    return list(filters) + [{"param": "excludeDigits", "value": True}]


def _inject_easy_say_exclude_hyphens(
    query: str,
    filters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Force excludeHyphens for pronounceable / no-hyphen cues."""
    if not query or not (
        _EASY_SAY_DIGITS_RE.search(query) or _EXPLICIT_NO_HYPHENS_RE.search(query)
    ):
        return filters
    if any(str(f.get("param") or "").strip() == "excludeHyphens" for f in filters):
        return filters
    return list(filters) + [{"param": "excludeHyphens", "value": True}]


def _scrub_ungrounded_exclude_digits(
    query: str,
    filters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Drop excludeDigits=true invented without typeability / no-digits cue."""
    if not query or not filters:
        return filters
    if _EASY_SAY_DIGITS_RE.search(query) or _EXPLICIT_NO_DIGITS_RE.search(query):
        return filters
    return [
        f
        for f in filters
        if not (
            str(f.get("param") or "").strip() == "excludeDigits"
            and f.get("value") is True
        )
    ]


# Explicit exclude speech-acts — required before keyword_contains_exclude sticks.
_EXCLUDE_SPEECH_ACT_RE = re.compile(
    r"\b(?:non[- ]|doesn'?t\s+sound|not\s+sound|not\s+containing|"
    r"without\s+(?:the\s+)?(?:word|keyword)|excluding|exclude\s+(?:the\s+)?"
    r"(?:word|keyword)|no\s+(?:word|keyword)|no\s+\w+\s+vibe|"
    r"without\s+\w+\s+words?|avoid(?:ing)?)\b",
    re.IGNORECASE,
)


def _scrub_ungrounded_keyword_exclude(
    query: str,
    filters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Drop keyword_contains_exclude invented without an exclude speech-act cue."""
    if not query or not filters:
        return filters
    if _EXCLUDE_SPEECH_ACT_RE.search(query):
        return filters
    return [
        f
        for f in filters
        if str(f.get("param") or "").strip() != "keyword_contains_exclude"
    ]


_KEYWORD_VALUE_PARAMS = frozenset(
    {
        "keyword_contains",
        "keyword_starts_with",
        "keyword_ends_with",
        "keyword_phrase",
        "keyword_contains_exclude",
    }
)


def _scrub_orphaned_keyword_match_mode(
    query: str,
    filters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Drop keyword_match_mode left behind once its companion value was scrubbed."""
    if not filters:
        return filters
    if any(str(f.get("param") or "").strip() in _KEYWORD_VALUE_PARAMS for f in filters):
        return filters
    return [
        f for f in filters if str(f.get("param") or "").strip() != "keyword_match_mode"
    ]


# Quality adjectives after "sounds like/sounds" are not brand seeds.
_QUALITY_SIMILAR_SEEDS = frozenset(
    {
        "legit",
        "brandable",
        "clean",
        "modern",
        "premium",
        "catchy",
        "punchy",
        "cheap",
        "short",
        "nice",
        "cool",
        "fresh",
        "sleek",
        "snappy",
        "trustworthy",
        "trusty",
        "reliable",
        "professional",
        "legitmate",
        "brand",
        "brands",
        "real",
    }
)


_MEETS_COTOPIC_RE = re.compile(
    r"\b(?:like\s+)?(?P<a>ai|fintech|saas|devtools|crypto|startup)\s+meets\s+"
    r"(?P<b>ai|fintech|saas|devtools|crypto|startup)\b",
    re.IGNORECASE,
)
_SOUNDS_LIKE_REAL_BRAND_RE = re.compile(
    r"\b(?:sounds?\s+like|similar\s+to)\s+(?:a\s+)?(?:real\s+)?brands?\b",
    re.IGNORECASE,
)


def _scrub_quality_adjective_similar_to(
    query: str,
    filters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Drop similar_to when every seed is a quality adjective (not a brand)."""
    if not filters:
        return filters
    # "X meets Y" is co-topic, not brand similarity. "sounds like real brand" -> empty seed.
    drop_all = bool(
        query
        and (
            _MEETS_COTOPIC_RE.search(query) or _SOUNDS_LIKE_REAL_BRAND_RE.search(query)
        )
    )
    out: List[Dict[str, Any]] = []
    for f in filters:
        if str(f.get("param") or "").strip() != "similar_to":
            out.append(f)
            continue
        if drop_all:
            continue
        raw = f.get("value")
        parts: List[str] = []
        if isinstance(raw, str):
            parts = [
                p.strip().lower() for p in raw.replace(",", "|").split("|") if p.strip()
            ]
        elif isinstance(raw, (list, tuple)):
            parts = [str(p).strip().lower() for p in raw if str(p).strip()]
        else:
            out.append(f)
            continue
        if parts and all(p in _QUALITY_SIMILAR_SEEDS for p in parts):
            continue
        out.append(f)
    return out


def _inject_meets_cotopic(
    query: str,
    filters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Ensure topic_include carries both sides of 'X meets Y' co-topic cue."""
    if not query:
        return filters
    m = _MEETS_COTOPIC_RE.search(query)
    if m is None:
        return filters
    need = sorted(
        {
            (m.group("a") or "").strip().lower(),
            (m.group("b") or "").strip().lower(),
        }
        - {""}
    )
    if not need:
        return filters
    out: List[Dict[str, Any]] = []
    saw = False
    for f in filters:
        if str(f.get("param") or "").strip() != "topic_include":
            out.append(f)
            continue
        saw = True
        raw = f.get("value")
        parts: List[str] = []
        if isinstance(raw, str):
            parts = [
                p.strip().lower() for p in raw.replace(",", "|").split("|") if p.strip()
            ]
        elif isinstance(raw, (list, tuple)):
            parts = [str(p).strip().lower() for p in raw if str(p).strip()]
        merged = sorted(set(parts) | set(need))
        out.append(
            {
                "param": "topic_include",
                "value": "|".join(merged) if len(merged) > 1 else merged[0],
            }
        )
    if not saw:
        out.append(
            {
                "param": "topic_include",
                "value": "|".join(need) if len(need) > 1 else need[0],
            }
        )
    return out


def _scrub_non_industry_topics(
    query: str,
    filters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Drop topic_include tokens not in _INDUSTRY_TOPIC_LEXICON (or convert to similar_to)."""
    if not filters:
        return filters
    out: List[Dict[str, Any]] = []
    for f in filters:
        if str(f.get("param") or "").strip() != "topic_include":
            out.append(f)
            continue
        raw = f.get("value")
        parts: List[str] = []
        if isinstance(raw, str):
            parts = [p.strip().lower() for p in raw.split("|") if p.strip()]
        elif isinstance(raw, (list, tuple)):
            parts = [str(p).strip().lower() for p in raw if str(p).strip()]
        else:
            out.append(f)
            continue
        keep: List[str] = []
        similar: List[str] = []
        for tok in parts:
            if tok in _INDUSTRY_TOPIC_LEXICON:
                keep.append(tok)
            elif re.search(
                rf"\b{re.escape(tok)}\s+(?:style|naming|vibe)\b", query, re.IGNORECASE
            ) or re.search(rf"\blike\s+{re.escape(tok)}\b", query, re.IGNORECASE):
                similar.append(tok)
            # else DROP
        if keep:
            out.append(
                {
                    "param": "topic_include",
                    "value": "|".join(sorted(set(keep))) if len(keep) > 1 else keep[0],
                }
            )
        if similar:
            # Merge into existing similar_to or add new
            existing_sim = next(
                (
                    i
                    for i, x in enumerate(out)
                    if str(x.get("param") or "").strip() == "similar_to"
                ),
                None,
            )
            if existing_sim is not None:
                prev = out[existing_sim].get("value") or ""
                prev_parts = [p.strip() for p in str(prev).split("|") if p.strip()]
                merged = sorted(set(prev_parts + similar))
                out[existing_sim] = {"param": "similar_to", "value": "|".join(merged)}
            else:
                out.append(
                    {"param": "similar_to", "value": "|".join(sorted(set(similar)))}
                )
    return out


def _scrub_weak_price_below_market(
    query: str,
    filters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Drop price_below_market when invented without a strong cue (prompt rules 4/15)."""
    has_pbm = any(
        str(f.get("param") or "").strip() == "price_below_market" for f in filters
    )
    if not has_pbm:
        return filters
    if _STRONG_PBM_CUE_RE.search(query):
        return filters
    return [
        f for f in filters if str(f.get("param") or "").strip() != "price_below_market"
    ]


def _inject_strong_price_below_market(
    query: str,
    filters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Inject price_below_market when strong gem/underpriced cue present and LLM omitted."""
    if not query or not _STRONG_PBM_CUE_RE.search(query):
        return filters
    if any(str(f.get("param") or "").strip() == "price_below_market" for f in filters):
        return filters
    return list(filters) + [{"param": "price_below_market", "value": True}]


def _scrub_weak_start_time(
    query: str,
    filters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Drop startTimeAfter when no listing-age cue is present."""
    has_sta = any(
        str(f.get("param") or "").strip() == "startTimeAfter" for f in filters
    )
    if not has_sta:
        return filters
    if _LISTING_AGE_CUE_RE.search(query):
        return filters
    return [f for f in filters if str(f.get("param") or "").strip() != "startTimeAfter"]


def _normalize_active_pending(
    query: str,
    filters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Force lifecycle_state=active|pending_delete + disjunction=true when both cued."""
    if not _ACTIVE_PENDING_CUE_RE.search(query):
        return filters
    out = [
        f
        for f in filters
        if str(f.get("param") or "").strip()
        not in ("lifecycle_state", "lifecycle_disjunction")
    ]
    out.append({"param": "lifecycle_state", "value": "active|pending_delete"})
    out.append({"param": "lifecycle_disjunction", "value": True})
    return out


def _normalize_expired_dropping(
    query: str,
    filters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Force lifecycle_state=expired|pending_delete + disjunction when OR-cued."""
    if not query or not _EXPIRED_DROPPING_CUE_RE.search(query):
        return filters
    cleaned: List[Dict[str, Any]] = []
    for f in filters:
        param = str(f.get("param") or "").strip()
        if param in ("lifecycle_state", "lifecycle_disjunction"):
            continue
        if param in ("typeIncludeList", "auction_type"):
            raw = f.get("value")
            if isinstance(raw, str):
                parts = [
                    p.strip().lower()
                    for p in raw.replace(",", "|").split("|")
                    if p.strip()
                ]
            elif isinstance(raw, (list, tuple)):
                parts = [str(p).strip().lower() for p in raw if str(p).strip()]
            else:
                cleaned.append(f)
                continue
            kept = [p for p in parts if p not in ("backorder", "expiry")]
            if kept:
                cleaned.append(
                    {
                        "param": param,
                        "value": "|".join(kept) if len(kept) > 1 else kept[0],
                    }
                )
            continue
        cleaned.append(f)
    cleaned.append({"param": "lifecycle_state", "value": "expired|pending_delete"})
    cleaned.append({"param": "lifecycle_disjunction", "value": True})
    return cleaned


_ZERO_BIDS_BAND_RE = re.compile(r"\b(?:no|zero)\s+bids?\b", re.IGNORECASE)
_NOBODY_BIDDING_RE = re.compile(
    r"\b(?:nobody|no\s*one|no\s*body)(?:'s)?\s+bidd(?:ing|in')\b",
    re.IGNORECASE,
)


def _normalize_zero_bids_band(
    query: str,
    filters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Parity with reconcile_bid_count_bounds: zero/no bids -> minBids=0 + maxBids=0."""
    if not query or not _ZERO_BIDS_BAND_RE.search(query):
        return filters
    if _NOBODY_BIDDING_RE.search(query):
        return filters
    out = [
        f
        for f in filters
        if str(f.get("param") or "").strip()
        not in ("minBids", "maxBids", "bids_min", "bids_max")
    ]
    out.append({"param": "minBids", "value": 0})
    out.append({"param": "maxBids", "value": 0})
    return out


def _scrub_is_gem_domain(filters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop isGemDomain (invalid param; intent maps to price_below_market)."""
    return [f for f in filters if str(f.get("param") or "").strip() != "isGemDomain"]


def _scrub_invented_max_sld_len_from_price(
    query: str,
    filters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Drop maxSldLen stolen from price 'under 1k' (999/1000) or large invents."""
    has_price_under = _PRICE_LIKE_UNDER_RE.search(query)
    if not has_price_under:
        return filters
    out: List[Dict[str, Any]] = []
    for f in filters:
        if str(f.get("param") or "").strip() != "maxSldLen":
            out.append(f)
            continue
        try:
            v = float(f.get("value") or 0)
        except (TypeError, ValueError):
            out.append(f)
            continue
        # under 1k -> 999/1000 as length; or any >=100 price steal
        if v >= 100 or v in (999.0, 1000.0):
            continue
        out.append(f)
    return out


_SHORT_CUE_RE = re.compile(r"\bshort\b", re.IGNORECASE)


def _normalize_short_max_sld_len(
    query: str,
    filters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Bare 'short' -> maxSldLen=5 (regex + LLM parity). Collapse any invent; inject if absent."""
    if not query or not _SHORT_CUE_RE.search(query):
        return filters
    # Explicit numeric length cue owns the value.
    if re.search(r"\b\d+\s*(?:chars?|characters?|letters?)\b", query, re.IGNORECASE):
        return filters
    out: List[Dict[str, Any]] = []
    saw = False
    for f in filters:
        if str(f.get("param") or "").strip() != "maxSldLen":
            out.append(f)
            continue
        try:
            v = float(f.get("value") or 0)
        except (TypeError, ValueError):
            out.append(f)
            continue
        # Bare short owns a length ceiling of 5 — collapse any invented maxSldLen
        # (including 10 / "short vs descriptive" overshoots), not only the 3–6 band.
        if not saw:
            out.append({"param": "maxSldLen", "value": 5})
            saw = True
        continue
    if not saw:
        out.append({"param": "maxSldLen", "value": 5})
    return out


_EXPLICIT_MIN_SLD_LEN_RE = re.compile(
    r"\b(?:at\s+least|min(?:imum)?|no\s+shorter\s+than|longer\s+than)\s+\d+"
    r"|\b\d+\s*(?:chars?|characters?|letters?)\s*(?:min|minimum|or\s+more|plus)\b"
    r"|\bmin(?:imum)?\s+(?:sld\s+)?(?:len(?:gth)?|chars?|characters?|letters?)\b",
    re.IGNORECASE,
)


def _scrub_bare_short_min_sld_len(
    query: str,
    filters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Drop invented minSldLen on bare 'short' cues (ceiling-only; no floor)."""
    if not query or not filters:
        return filters
    if not _SHORT_CUE_RE.search(query):
        return filters
    if _EXPLICIT_MIN_SLD_LEN_RE.search(query):
        return filters
    return [
        f for f in filters if str(f.get("param") or "").strip() != "minSldLen"
    ]


def _scrub_ending_soon_expired_lifecycle(
    query: str,
    filters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Drop lifecycle_state=expired invented beside ending-soon endTimeBefore.

    ``expiring soon`` owns ``endTimeBefore``; bare ``expired`` inventory browse keeps
    lifecycle. ``\\bexpired\\b`` does not match ``expiring``.
    """
    if not query or not filters:
        return filters
    if not _ENDING_SOON_CUE_RE.search(query):
        return filters
    if re.search(r"\bexpired\b", query, re.IGNORECASE):
        return filters
    out = [
        f
        for f in filters
        if not (
            str(f.get("param") or "").strip() == "lifecycle_state"
            and str(f.get("value") or "").strip().lower() == "expired"
        )
    ]
    if not any(str(f.get("param") or "").strip() == "lifecycle_state" for f in out):
        out = [
            f
            for f in out
            if str(f.get("param") or "").strip() != "lifecycle_disjunction"
        ]
    return out


def _scrub_redundant_active_with_extended(
    query: str,
    filters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Drop default lifecycle_state=active when isExtended already constrains."""
    if not query or not filters:
        return filters
    if re.search(r"\bactive\b", query, re.IGNORECASE):
        return filters
    has_extended = any(
        str(f.get("param") or "").strip() == "isExtended" and f.get("value") is True
        for f in filters
    )
    if not has_extended:
        return filters
    return [
        f
        for f in filters
        if not (
            str(f.get("param") or "").strip() == "lifecycle_state"
            and str(f.get("value") or "").strip().lower() == "active"
        )
    ]


_EXCLUSIVE_FLOOR_CUE_RE = re.compile(
    r"\b(?:above|over|more\s+than|greater\s+than)\s+\$?(?P<n>\d[\d,]*(?:\.\d+)?)\s*(?P<k>k)?\b",
    re.IGNORECASE,
)
_EXCLUSIVE_FLOOR_PARAMS = frozenset(
    {
        "minAge",
        "minValuationPrice",
        "minPrice",
        "minTraffic",
        "minMajesticTrustFlowScore",
        "minMajesticCitationFlowScore",
        "minMajesticBackLinks",
        "minMajesticRefDomains",
        "minSemrushAScore",
        "minSemrushBackLinks",
        "minSemrushRefDomains",
        "minSemrushCostPerClick",
        "minEstibotAppraisedValue",
        "minEstibotExtCount",
        "minEstibotExtCountDev",
        "minEstibotDomainCount",
        "minBids",
        "minSldLen",
    }
)


def _normalize_exclusive_min_floors(
    query: str,
    filters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Force above/over/more-than mins -> N+1 when LLM emitted inclusive N."""
    if not query or not filters:
        return filters
    nums: List[float] = []
    for m in _EXCLUSIVE_FLOOR_CUE_RE.finditer(query):
        raw = (m.group("n") or "").replace(",", "")
        try:
            v = float(raw)
        except ValueError:
            continue
        if m.group("k"):
            v *= 1000.0
        nums.append(v)
    if not nums:
        return filters
    targets = set(nums)
    out: List[Dict[str, Any]] = []
    for f in filters:
        param = str(f.get("param") or "").strip()
        if param not in _EXCLUSIVE_FLOOR_PARAMS:
            out.append(f)
            continue
        try:
            v = float(f.get("value"))
        except (TypeError, ValueError):
            out.append(f)
            continue
        if v in targets:
            out.append({"param": param, "value": int(v + 1) if v == int(v) else v + 1})
        else:
            out.append(f)
    return out


_QUALITATIVE_SENTINELS = frozenset(
    {
        "none",
        "null",
        "n/a",
        "na",
        "high",
        "low",
        "good",
        "strong",
        "lots",
        "proven",
        "cheap",
    }
)


def _is_qualitative_sentinel(value: Any) -> bool:
    """True when value is a non-numeric qualitative stand-in (``none``/``high``/…)."""
    if value is None:
        return True
    if isinstance(value, (bool, int, float)):
        return False
    if isinstance(value, str):
        return value.strip().lower() in _QUALITATIVE_SENTINELS
    return False


# Bare qualitative price words (L0 rule 4) — drop invented maxPrice/minPrice when no numeric cue.
_BARE_QUALITATIVE_PRICE_RE = re.compile(
    r"\b(?:not\s+(?:too|to|very|overly)\s+expensive|not\s+expensive|reasonably\s+priced|"
    r"affordable|cheap(?:er)?|inexpensive|low[- ]?cost|low[- ]?price|"
    r"cheapest|most\s+affordable|lowest\s+price|less\s+expensive|"
    r"lower\s+(?:price|cost)|pricey|expensive|high[- ]?end|high[- ]?priced|"
    r"mid\s+budget|around\s+budget|budget)\b",
    re.IGNORECASE,
)
# Length/count units after under/below N are NOT price cues ("under 5 chars cheap").
_NUMERIC_PRICE_CUE_RE = re.compile(
    r"(?:"
    r"\$\s*\d"
    r"|\b(?:under|below|over|above|at\s+least|at\s+most|less\s+than|more\s+than|"
    r"up\s+to|capped\s+at|max(?:imum)?|min(?:imum)?|floor|between|from)\s+\$?\d"
    r"(?!\s*(?:chars?|characters?|letters?|words?|years?|yrs?|bids?|"
    r"backlinks?|referring|ref\s*domains?|visitors?))"
    r"|\b\d+(?:\.\d+)?\s*k\b"
    r"|\b\d{3,}\b"  # 100+ bare ints as price-ish; skip 1–2 digit noise
    r")",
    re.IGNORECASE,
)


def _scrub_ungrounded_price_bounds(
    query: str,
    filters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Drop maxPrice/minPrice invented from bare cheap/expensive (no numeric cue)."""
    if not query or not filters:
        return filters
    if not _BARE_QUALITATIVE_PRICE_RE.search(query):
        return filters
    if _NUMERIC_PRICE_CUE_RE.search(query):
        return filters
    return [
        f
        for f in filters
        if str(f.get("param") or "").strip()
        not in ("maxPrice", "minPrice", "price_max", "price_min")
    ]


def _scrub_dangling_under_price(
    query: str,
    filters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Drop maxPrice invented from incomplete 'under'/'below' with no amount."""
    if not query or not filters:
        return filters
    if not re.search(r"\b(?:under|below)\b", query, re.IGNORECASE):
        return filters
    if re.search(
        r"\b(?:under|below|less\s+than|at\s+most|no\s+more\s+than|capped\s+at)\s+"
        r"(?:(?:usd|eur|gbp|inr|cad|aud|jpy)\s+)?\$?\d",
        query,
        re.IGNORECASE,
    ):
        return filters
    return [
        f
        for f in filters
        if str(f.get("param") or "").strip() not in ("maxPrice", "price_max")
    ]


# Optional ISO currency between cue and amount: "under EUR 80", "below GBP 95".
_UNDER_PRICE_INJECT_RE = re.compile(
    r"\b(?:under|below|less\s+than)\s+"
    r"(?:(?:usd|eur|gbp|inr|cad|aud|jpy)\s+)?"
    r"\$?(?P<n>\d[\d,]*)\s*(?P<k>k|thousand|million)?\b"
    r"|\b(?P<n2>\d[\d,]*)\s*(?P<k2>k|thousand)?\s+or\s+(?:under|below)\b",
    re.IGNORECASE,
)
_LOOK_EXPENSIVE_RE = re.compile(
    r"\b(?:look|looks|sound|sounds)\s+expensive\b|\bexpensive[- ]looking\b",
    re.IGNORECASE,
)
_DUAL_HIGH_QUAL_RE = re.compile(
    r"\bhigh\s+(?:go\s*value|govalue|domain\s+authority|da|authority|traffic|backlinks?|"
    r"valuation|trust\s+flow|citation\s+flow)\b"
    r".{0,48}\bhigh\s+(?:go\s*value|govalue|domain\s+authority|da|authority|traffic|"
    r"backlinks?|valuation|trust\s+flow|citation\s+flow)\b",
    re.IGNORECASE,
)


def _inject_under_price(
    query: str,
    filters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Inject maxPrice from under/below N (or 'N or under') when LLM omitted it."""
    if not query:
        return filters
    if any(
        str(f.get("param") or "").strip() in ("maxPrice", "price_max") for f in filters
    ):
        return filters
    m = _UNDER_PRICE_INJECT_RE.search(query)
    if m is None:
        return filters
    raw = m.group("n") or m.group("n2")
    scale = (m.group("k") or m.group("k2") or "").lower()
    try:
        n = int(str(raw).replace(",", ""))
    except (TypeError, ValueError):
        return filters
    if scale in ("k", "thousand"):
        n *= 1000
    elif scale == "million":
        n *= 1_000_000
    # Exclusive under/below -> N-1; "N or under" inclusive -> N.
    if m.group("n2"):
        hi = n
    else:
        hi = max(0, n - 1)
    return list(filters) + [{"param": "maxPrice", "value": hi}]


def _inject_budget_prefixed_price(
    query: str,
    filters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Inject inclusive maxPrice (and band min for around/at/about) from prefix+$N."""
    if not query:
        return filters
    has_max = any(
        str(f.get("param") or "").strip() in ("maxPrice", "price_max") for f in filters
    )
    has_min = any(
        str(f.get("param") or "").strip() in ("minPrice", "price_min") for f in filters
    )
    if has_max and has_min:
        return filters
    parsed = parse_budget_prefixed_price(query)
    if parsed is None:
        return filters
    n, band = parsed
    out = list(filters)
    if band:
        if not has_min:
            out.append({"param": "minPrice", "value": n})
        if not has_max:
            out.append({"param": "maxPrice", "value": n})
    elif not has_max:
        out.append({"param": "maxPrice", "value": n})
    return out


# Currency cues for qie_only LLM scrub (parity with reconcile_currency_vs_tld / regex).
_L0_CURRENCY_CUE_RE = re.compile(
    r"\b(?P<code>usd|eur|gbp|inr|cad|aud|jpy)\b"
    r"|\b(?P<word>dollars?|euros?|pounds?|rupees?|yen|"
    r"canadian\s+dollars?|australian\s+dollars?|us\s+dollars?)\b"
    r"|(?P<sym>[$€£¥₹])",
    re.IGNORECASE,
)
_L0_CURRENCY_WORD_TO_CODE = {
    "dollar": "USD",
    "dollars": "USD",
    "us dollar": "USD",
    "us dollars": "USD",
    "euro": "EUR",
    "euros": "EUR",
    "pound": "GBP",
    "pounds": "GBP",
    "rupee": "INR",
    "rupees": "INR",
    "yen": "JPY",
    "canadian dollar": "CAD",
    "canadian dollars": "CAD",
    "australian dollar": "AUD",
    "australian dollars": "AUD",
}
_L0_CURRENCY_SYMBOL_TO_CODE = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "₹": "INR",
}


def _inject_filter_price_currency(
    query: str,
    filters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Ensure filterPriceCurrency when query has $/usd/dollars/euros/… (LLM omit backstop)."""
    if not query:
        return filters
    if any(str(f.get("param") or "").strip() == "filterPriceCurrency" for f in filters):
        return filters
    m = _L0_CURRENCY_CUE_RE.search(query)
    if m is None:
        return filters
    codes: List[str] = []
    if m.group("code"):
        codes.append(m.group("code").upper())
    elif m.group("word"):
        mapped = _L0_CURRENCY_WORD_TO_CODE.get(m.group("word").lower())
        if mapped:
            codes.append(mapped)
    elif m.group("sym"):
        mapped = _L0_CURRENCY_SYMBOL_TO_CODE.get(m.group("sym"))
        if mapped:
            codes.append(mapped)
    if not codes:
        return filters
    non_usd = [c for c in codes if c != "USD"]
    code = non_usd[0] if non_usd else codes[0]
    logger.info(f"l0_filter_currency_injected value={code!r}")
    return list(filters) + [{"param": "filterPriceCurrency", "value": code}]


_TRAFFIC_UNKNOWN_OR_ZERO_RE = re.compile(
    r"\btraffic\s+unknown\s+or\s+zero\b|\bunknown\s+or\s+zero\s+traffic\b"
    r"|\b(?:traffic\s+unknown|unknown\s+traffic).{0,24}\bzero\b"
    r"|\bzero\b.{0,24}\b(?:traffic\s+unknown|unknown\s+traffic)\b",
    re.IGNORECASE,
)
_NEW_BRAND_AGE_RE = re.compile(r"\b(?:new\s+brand|brand\s+new)\b", re.IGNORECASE)
_PREMIUM_EXTENSION_RE = re.compile(
    r"\bpremium\s+extension\b|\bextended\s+(?:premium\s+)?(?:auction|listing)s?\b"
    r"|\bshort\s+premium\s+extension\b",
    re.IGNORECASE,
)
_ADDED_LAST_HOUR_L0_RE = re.compile(
    r"\b(?:added|listed)\s+(?:in\s+)?(?:the\s+)?last\s+hour\b|\blast\s+hour\b",
    re.IGNORECASE,
)


def _inject_traffic_unknown_zero_band(
    query: str,
    filters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """traffic unknown or zero -> traffic_is_unknown + maxTraffic=0; new brand -> minAge=0."""
    if not query or not _TRAFFIC_UNKNOWN_OR_ZERO_RE.search(query):
        return filters
    out = list(filters)
    params = {str(f.get("param") or "").strip() for f in out}
    if "traffic_is_unknown" not in params:
        out.append({"param": "traffic_is_unknown", "value": True})
    if "maxTraffic" not in params and "traffic_max" not in params:
        out.append({"param": "maxTraffic", "value": 0})
    if (
        _NEW_BRAND_AGE_RE.search(query)
        and "minAge" not in params
        and "domain_age_min" not in params
    ):
        out.append({"param": "minAge", "value": 0})
    return out


def _inject_premium_extension(
    query: str,
    filters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """premium extension -> isExtended + typeIncludeList:premium when omitted."""
    if not query or not _PREMIUM_EXTENSION_RE.search(query):
        return filters
    out = list(filters)
    params = {str(f.get("param") or "").strip() for f in out}
    if "isExtended" not in params:
        out.append({"param": "isExtended", "value": True})
    if "typeIncludeList" not in params and "auction_type" not in params:
        out.append({"param": "typeIncludeList", "value": "premium"})
    return out


def _inject_added_last_hour(
    query: str,
    filters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Inject startTimeAfter=-1h for last-hour listing cues when omitted."""
    if not query or not _ADDED_LAST_HOUR_L0_RE.search(query):
        return filters
    if any(
        str(f.get("param") or "").strip()
        in (
            "startTimeAfter",
            "days_listed_max",
            "days_listed_min",
        )
        for f in filters
    ):
        return filters
    return list(filters) + [{"param": "startTimeAfter", "value": "-1h"}]


_AROUND_MAYBE_LESS_RE = re.compile(
    r"\b(?:budget\s+)?around\s+\$?(?P<n>\d[\d,]*)\s*(?P<k>k|thousand)?\b"
    r".{0,24}\b(?:maybe\s+)?(?:less|under|below)\b",
    re.IGNORECASE,
)
_AI_OR_FINTECH_RE = re.compile(
    r"\bai\s+or\s+fintech\b|\bfintech\s+or\s+ai\b",
    re.IGNORECASE,
)


def _inject_around_maybe_less_price(
    query: str,
    filters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """budget around Nk maybe less -> exclusive maxPrice when omitted."""
    if not query:
        return filters
    if any(
        str(f.get("param") or "").strip() in ("maxPrice", "price_max") for f in filters
    ):
        return filters
    m = _AROUND_MAYBE_LESS_RE.search(query)
    if m is None:
        return filters
    try:
        n = int(str(m.group("n")).replace(",", ""))
    except (TypeError, ValueError):
        return filters
    if (m.group("k") or "").lower() in ("k", "thousand") or n >= 1000:
        n = n * 1000 if (m.group("k") or "").lower() in ("k", "thousand") else n
        hi = n - 1
    else:
        hi = n
    return list(filters) + [{"param": "maxPrice", "value": hi}]


def _inject_ai_or_fintech_topic(
    query: str,
    filters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Force topic_include=ai|fintech for 'ai or fintech' co-topic cue."""
    if not query or not _AI_OR_FINTECH_RE.search(query):
        return filters
    out: List[Dict[str, Any]] = []
    saw = False
    for f in filters:
        if str(f.get("param") or "").strip() != "topic_include":
            out.append(f)
            continue
        saw = True
        raw = f.get("value")
        parts: List[str] = []
        if isinstance(raw, str):
            parts = [
                p.strip().lower() for p in raw.replace(",", "|").split("|") if p.strip()
            ]
        elif isinstance(raw, (list, tuple)):
            parts = [str(p).strip().lower() for p in raw if str(p).strip()]
        merged = sorted(set(parts) | {"ai", "fintech"})
        out.append(
            {
                "param": "topic_include",
                "value": "|".join(merged) if len(merged) > 1 else merged[0],
            }
        )
    if not saw:
        out.append({"param": "topic_include", "value": "ai|fintech"})
    # Drop tldIncludeList=ai when co-topic owns ai.
    final: List[Dict[str, Any]] = []
    for f in out:
        if str(f.get("param") or "").strip() not in ("tldIncludeList", "tld"):
            final.append(f)
            continue
        raw = f.get("value")
        parts = []
        if isinstance(raw, str):
            parts = [
                p.strip().lower().lstrip(".")
                for p in raw.replace(",", "|").split("|")
                if p.strip()
            ]
        elif isinstance(raw, (list, tuple)):
            parts = [str(p).strip().lower().lstrip(".") for p in raw if str(p).strip()]
        kept = [p for p in parts if p != "ai"]
        if kept:
            final.append(
                {"param": "tldIncludeList", "value": "|".join(sorted(set(kept)))}
            )
    return final


def _scrub_look_expensive_min_price(
    query: str,
    filters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Drop invented minPrice from 'look expensive' without numeric floor cue."""
    if not query or not filters or not _LOOK_EXPENSIVE_RE.search(query):
        return filters
    if re.search(
        r"\b(?:over|above|at\s+least|more\s+than|from|min(?:imum)?)\s+\$?\d",
        query,
        re.IGNORECASE,
    ):
        return filters
    return [
        f
        for f in filters
        if str(f.get("param") or "").strip() not in ("minPrice", "price_min")
    ]


def _scrub_dual_high_qualitative(
    query: str,
    filters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Wipe soft metric floors when query is dual 'high X and high Y' with no numbers."""
    if not query or not filters or not _DUAL_HIGH_QUAL_RE.search(query):
        return filters
    if re.search(r"\d", query):
        return filters
    drop = {
        "minValuationPrice",
        "maxValuationPrice",
        "govalue_min",
        "govalue_max",
        "minSemrushAScore",
        "maxSemrushAScore",
        "semrush_authority_min",
        "minMajesticTrustFlowScore",
        "minMajesticCitationFlowScore",
        "minMajesticBackLinks",
        "minTraffic",
        "maxTraffic",
    }
    return [f for f in filters if str(f.get("param") or "").strip() not in drop]


_SOFT_OR_LINK_TRAFFIC_L0_RE = re.compile(
    r"\b(?:traffic|visitors?|backlinks?|authority|brandable)\b"
    r".{0,32}\bor\b.{0,32}\b(?:traffic|visitors?|backlinks?|authority|brandable)\b",
    re.IGNORECASE,
)


def _scrub_soft_or_link_floors(
    query: str,
    filters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Drop soft backlink/traffic floors when query is traffic|backlinks soft OR."""
    if not query or not filters or not _SOFT_OR_LINK_TRAFFIC_L0_RE.search(query):
        return filters
    drop = {
        "minMajesticBackLinks",
        "majestic_backlinks_min",
        "minSemrushLinksTotal",
        "semrush_backlinks_min",
        "minMajesticTrustFlowScore",
        "majestic_tf_min",
        "minTraffic",
        "traffic_min",
        "minSemrushAScore",
        "semrush_authority_min",
    }
    out: List[Dict[str, Any]] = []
    for f in filters:
        param = str(f.get("param") or "").strip()
        if param in drop:
            try:
                if int(f.get("value")) <= 1:
                    continue
            except (TypeError, ValueError):
                pass
        out.append(f)
    return out


def _scrub_false_exclude_letters(
    filters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Drop excludeLetters=false noise (default; only true is meaningful)."""
    out: List[Dict[str, Any]] = []
    for f in filters:
        if str(f.get("param") or "").strip() != "excludeLetters":
            out.append(f)
            continue
        if f.get("value") in (False, "false", "False", 0, "0"):
            continue
        out.append(f)
    return out


_REGISTERED_BEFORE_YEAR_RE = re.compile(
    r"\bregistered\s+before\s+(?P<year>19\d{2}|20\d{2})\b",
    re.IGNORECASE,
)
_UNKNOWN_AGE_WITH_PRICE_RE = re.compile(
    r"\b(?:young|unknown)\b.{0,40}\bage\b|\bage\b.{0,40}\b(?:young|unknown)\b|"
    r"\bunknown\s+age\b",
    re.IGNORECASE,
)


def _normalize_registered_before(
    query: str,
    filters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Force minAge = now.year - YEAR for 'registered before YEAR' (not maxAge)."""
    if not query:
        return filters
    m = _REGISTERED_BEFORE_YEAR_RE.search(query)
    if not m:
        return filters
    year = int(m.group("year"))
    age = max(1, datetime.now(timezone.utc).year - year)
    out = [
        f
        for f in filters
        if str(f.get("param") or "").strip()
        not in ("minAge", "maxAge", "domain_age_min", "domain_age_max")
    ]
    out.append({"param": "minAge", "value": age})
    return out


def _scrub_unknown_age_with_price(
    query: str,
    filters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Prompt: 'young or unknown age under N' -> maxPrice only (drop unknown-age chip)."""
    if not query or not filters:
        return filters
    if not _UNKNOWN_AGE_WITH_PRICE_RE.search(query):
        return filters
    if not re.search(r"\b(?:under|below|less\s+than)\s+\$?\d", query, re.IGNORECASE):
        return filters
    if re.search(r"\byears?\b", query, re.IGNORECASE):
        return filters
    drop = {
        "domain_age_is_unknown",
        "minAge",
        "maxAge",
        "domain_age_min",
        "domain_age_max",
    }
    return [f for f in filters if str(f.get("param") or "").strip() not in drop]


_ENDING_WEEKEND_CUE_RE = re.compile(
    r"\b(?:end(?:ing|ng)|closing|expiring|ends?|closes?)\s+this\s+weekend\b",
    re.IGNORECASE,
)
_ENDING_SOON_CUE_RE = re.compile(
    r"\b(?:end(?:ing|ng)|closing|expiring|ends?|closes?|expires?)"
    r"(?:\s+\w+){0,3}\s+soon\b",
    re.IGNORECASE,
)

# Prompt rule 1: bare "<tld> domains" / "<tld> under N" -> tldIncludeList, not topic_include.
_KNOWN_TLD_TOPIC_VALUES = frozenset(
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
# Niche/quality modifier + short token, OR short industry token + role noun.
_TOPICISH_BEFORE_TLD_RE = re.compile(
    r"\b(?:premium|extended|brandable|startup|fintech|saas|crypto|cloud|devtools|"
    r"edtech|healthcare|climate|gaming)\s+(?:ai|io|app|dev)\b"
    r"|\b(?:ai)\s+(?:agent|names?|gems?|inventory|compan(?:y|ies)|products?|"
    r"startups?|brands?|tools?|niches?|meets)\b"
    r"|\b(?:ai)\s+\.(?:com|net|org|io)\b"
    r"|\b(?:ai)\s+or\s+(?:fintech|saas|startup|devtools)\b",
    re.IGNORECASE,
)


def _normalize_ending_weekend(
    query: str,
    filters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Force endTimeBefore=-3d when query cues this weekend (typo ``endng`` included)."""
    if not query or not _ENDING_WEEKEND_CUE_RE.search(query):
        return filters
    out: List[Dict[str, Any]] = []
    saw = False
    for f in filters:
        param = str(f.get("param") or "").strip()
        if param in ("endTimeBefore", "endTimeAfter", "time_remaining_max"):
            if not saw:
                out.append({"param": "endTimeBefore", "value": "-3d"})
                saw = True
            continue
        out.append(f)
    if not saw:
        out.append({"param": "endTimeBefore", "value": "-3d"})
    return out


def _normalize_ending_soon(
    query: str,
    filters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Force endTimeBefore=-1d for ending/closing/expiring soon (prompt rule 13)."""
    if not query or not _ENDING_SOON_CUE_RE.search(query):
        return filters
    if _ENDING_WEEKEND_CUE_RE.search(query):
        return filters
    out: List[Dict[str, Any]] = []
    saw = False
    for f in filters:
        param = str(f.get("param") or "").strip()
        if param in ("endTimeBefore", "endTimeAfter", "time_remaining_max"):
            if not saw:
                out.append({"param": "endTimeBefore", "value": "-1d"})
                saw = True
            continue
        out.append(f)
    if not saw:
        out.append({"param": "endTimeBefore", "value": "-1d"})
    return out


_EXCLUSIVE_PRICE_CEILING_RE = re.compile(
    r"\b(?:under|below|less\s+than)\s+\$?\d",
    re.IGNORECASE,
)
_INCLUSIVE_PRICE_CEILING_RE = re.compile(
    r"\b(?:at\s+most|no\s+more\s+than|capped\s+at|nothing\s+over|not\s+over)\s+\$?\d"
    r"|\b\d[\d,]*(?:\s*[kKmM])?\s+or\s+(?:under|below|less)\b",
    re.IGNORECASE,
)
_OR_UNDER_PRICE_RE = re.compile(
    r"\b(?P<n>\d[\d,]*)\s*(?P<k>k|thousand|million)?\s+or\s+(?:under|below|less)\b",
    re.IGNORECASE,
)


def _normalize_or_under_inclusive(
    query: str,
    filters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Force 'N or under' -> inclusive maxPrice=N (not exclusive N-1)."""
    if not query or not filters:
        return filters
    m = _OR_UNDER_PRICE_RE.search(query)
    if m is None:
        return filters
    try:
        n = int(str(m.group("n")).replace(",", ""))
    except (TypeError, ValueError):
        return filters
    scale = (m.group("k") or "").lower()
    if scale in ("k", "thousand"):
        n *= 1000
    elif scale == "million":
        n *= 1_000_000
    out: List[Dict[str, Any]] = []
    saw = False
    for f in filters:
        param = str(f.get("param") or "").strip()
        if param not in ("maxPrice", "price_max"):
            out.append(f)
            continue
        saw = True
        out.append({"param": param, "value": n})
    if not saw:
        out.append({"param": "maxPrice", "value": n})
    return out


def _normalize_exclusive_price_ceilings(
    query: str,
    filters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Force under/below/less-than maxPrice -> N-1 when LLM emitted inclusive N."""
    if not query or not filters:
        return filters
    if _INCLUSIVE_PRICE_CEILING_RE.search(query):
        return filters
    if not _EXCLUSIVE_PRICE_CEILING_RE.search(query):
        return filters
    out: List[Dict[str, Any]] = []
    for f in filters:
        param = str(f.get("param") or "").strip()
        if param not in ("maxPrice", "price_max"):
            out.append(f)
            continue
        raw = f.get("value")
        try:
            n = float(raw)
        except (TypeError, ValueError):
            out.append(f)
            continue
        if n != int(n) or n <= 0:
            out.append(f)
            continue
        # Already exclusive (999 for under 1k) — leave alone when query scale matches.
        out.append(
            {
                "param": param,
                "value": int(n) - 1
                if _looks_inclusive_ceiling(query, int(n))
                else int(n),
            }
        )
    return out


def _looks_inclusive_ceiling(query: str, n: int) -> bool:
    """True when query states under/below X and filter value still equals X."""
    m = re.search(
        r"\b(?:under|below|less\s+than)\s+\$?(?P<num>\d[\d,]*(?:[kKmM])?)\b",
        query,
        re.IGNORECASE,
    )
    if not m:
        return False
    raw = m.group("num").lower().replace(",", "")
    if raw.endswith("k"):
        stated = int(float(raw[:-1]) * 1000)
    elif raw.endswith("m"):
        stated = int(float(raw[:-1]) * 1_000_000)
    else:
        stated = int(float(raw))
    return n == stated


def _normalize_tld_vs_topic(
    query: str,
    filters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Rewrite topic_include=<known TLD> -> tldIncludeList when cue is bare TLD inventory.

    Keeps topic_include when a niche/premium modifier owns the short token
    (``premium ai domain``, ``extended ai auction``, ``ai agent domains``).
    Topicish cues also rewrite lone tldIncludeList=ai -> topic_include=ai.
    """
    if not query or not filters:
        return filters
    topicish_toks = set()
    for m in _TOPICISH_BEFORE_TLD_RE.finditer(query):
        # Entire match implies the short industry token is topic-owned (usually ai).
        blob = (m.group(0) or "").lower()
        for tok in ("ai", "io", "app", "dev"):
            if re.search(rf"\b{tok}\b", blob):
                # Only promote to topic when the cue is topicish for that token
                # (ai .com / ai or fintech / premium ai) — not bare .io inventory.
                if tok == "ai" or re.search(
                    rf"\b(?:premium|extended|brandable|startup|fintech|saas)\s+{tok}\b",
                    blob,
                ):
                    topicish_toks.add(tok)
    if topicish_toks:
        # Drop only topic-owned short tokens from tld; keep real TLD chips (.com/.io).
        out: List[Dict[str, Any]] = []
        has_topic = False
        dropped_tld: List[str] = []
        for f in filters:
            param = str(f.get("param") or "").strip()
            if param == "topic_include":
                has_topic = True
                out.append(f)
                continue
            if param not in ("tldIncludeList", "tld"):
                out.append(f)
                continue
            raw = f.get("value")
            parts: List[str] = []
            if isinstance(raw, str):
                parts = [
                    p.strip().lower().lstrip(".")
                    for p in raw.replace(",", "|").split("|")
                    if p.strip()
                ]
            elif isinstance(raw, (list, tuple)):
                parts = [
                    str(p).strip().lower().lstrip(".") for p in raw if str(p).strip()
                ]
            else:
                out.append(f)
                continue
            kept = [p for p in parts if p not in topicish_toks]
            dropped_tld.extend(p for p in parts if p in topicish_toks)
            if kept:
                out.append(
                    {"param": "tldIncludeList", "value": "|".join(sorted(set(kept)))}
                )
        if dropped_tld and not has_topic:
            tok = "ai" if "ai" in dropped_tld else sorted(set(dropped_tld))[0]
            out.append({"param": "topic_include", "value": tok})
        elif dropped_tld and has_topic:
            # Ensure topic chip carries the topicish token alongside other niches.
            final: List[Dict[str, Any]] = []
            for f in out:
                if str(f.get("param") or "").strip() != "topic_include":
                    final.append(f)
                    continue
                raw = f.get("value")
                parts = []
                if isinstance(raw, str):
                    parts = [
                        p.strip().lower()
                        for p in raw.replace(",", "|").split("|")
                        if p.strip()
                    ]
                elif isinstance(raw, (list, tuple)):
                    parts = [str(p).strip().lower() for p in raw if str(p).strip()]
                merged = sorted(set(parts) | set(dropped_tld))
                final.append(
                    {
                        "param": "topic_include",
                        "value": "|".join(merged) if len(merged) > 1 else merged[0],
                    }
                )
            return final
        return out
    out: List[Dict[str, Any]] = []
    tlds_from_topic: List[str] = []
    for f in filters:
        param = str(f.get("param") or "").strip()
        if param != "topic_include":
            out.append(f)
            continue
        raw = f.get("value")
        parts: List[str] = []
        if isinstance(raw, str):
            parts = [p.strip().lower() for p in raw.split("|") if p.strip()]
        elif isinstance(raw, (list, tuple)):
            parts = [str(p).strip().lower() for p in raw if str(p).strip()]
        else:
            out.append(f)
            continue
        keep_topics: List[str] = []
        for tok in parts:
            bare_domains = bool(
                re.search(
                    rf"\b{re.escape(tok)}\s+domains?(?:\s+only)?\b",
                    query,
                    re.IGNORECASE,
                )
            )
            bare_under = bool(
                re.search(
                    rf"\b{re.escape(tok)}\s+(?:under|below|over|above)\b",
                    query,
                    re.IGNORECASE,
                )
            )
            if tok in _KNOWN_TLD_TOPIC_VALUES and (bare_domains or bare_under):
                tlds_from_topic.append(tok)
            else:
                keep_topics.append(tok)
        if keep_topics:
            out.append(
                {
                    "param": "topic_include",
                    "value": "|".join(sorted(set(keep_topics)))
                    if len(keep_topics) > 1
                    else keep_topics[0],
                }
            )
    if not tlds_from_topic:
        return out if out else filters
    # Merge into existing tldIncludeList if present.
    merged = sorted(set(tlds_from_topic))
    final: List[Dict[str, Any]] = []
    saw_tld = False
    for f in out:
        param = str(f.get("param") or "").strip()
        if param in ("tldIncludeList", "tld"):
            if saw_tld:
                continue
            raw = f.get("value")
            existing: List[str] = []
            if isinstance(raw, str):
                existing = [
                    p.strip().lower()
                    for p in raw.replace(",", "|").split("|")
                    if p.strip()
                ]
            elif isinstance(raw, (list, tuple)):
                existing = [str(p).strip().lower() for p in raw if str(p).strip()]
            merged = sorted(set(existing) | set(merged))
            final.append({"param": "tldIncludeList", "value": "|".join(merged)})
            saw_tld = True
        else:
            final.append(f)
    if not saw_tld:
        final.append({"param": "tldIncludeList", "value": "|".join(merged)})
    return final


_QUERY_TYPO_SUBS_L0: tuple = (
    (re.compile(r"\bfintec\b", re.IGNORECASE), "fintech"),
    (re.compile(r"\bdevtols\b", re.IGNORECASE), "devtools"),
    (re.compile(r"\bcybersecuirty\b", re.IGNORECASE), "cybersecurity"),
    (re.compile(r"\bhealthcar\b", re.IGNORECASE), "healthcare"),
    (re.compile(r"\bstartp\b", re.IGNORECASE), "startup"),
    (re.compile(r"\bdomians\b", re.IGNORECASE), "domains"),
    (re.compile(r"\bdoamins\b", re.IGNORECASE), "domains"),
    (re.compile(r"\bdoamin\b", re.IGNORECASE), "domain"),
    (re.compile(r"\bdomian\b", re.IGNORECASE), "domain"),
    (re.compile(r"\bnumbrs?\b", re.IGNORECASE), "numbers"),
    (re.compile(r"\bhyphns?\b", re.IGNORECASE), "hyphens"),
    (re.compile(r"\bbrandible\b", re.IGNORECASE), "brandable"),
    (re.compile(r"\bcompny\b", re.IGNORECASE), "company"),
    (re.compile(r"\btrafic\b", re.IGNORECASE), "traffic"),
    (re.compile(r"\bgoood\b", re.IGNORECASE), "good"),
    (re.compile(r"\bbrandng\b", re.IGNORECASE), "branding"),
    (re.compile(r"\bwit\b", re.IGNORECASE), "with"),
    (re.compile(r"\bexpird\b", re.IGNORECASE), "expired"),
    (re.compile(r"\blistngs\b", re.IGNORECASE), "listings"),
    (re.compile(r"\btrafic\b", re.IGNORECASE), "traffic"),
    (re.compile(r"\btraffik\b", re.IGNORECASE), "traffic"),
    (re.compile(r"\bcheep\b", re.IGNORECASE), "cheap"),
    (re.compile(r"\babov\b", re.IGNORECASE), "above"),
    (re.compile(r"\bbelo\b", re.IGNORECASE), "below"),
    (re.compile(r"\bwort\b", re.IGNORECASE), "worth"),
    (re.compile(r"\byrs\b", re.IGNORECASE), "years"),
)


def _normalize_query_typos(query: str) -> str:
    """Apply shared surface-typo fixes so scrub regexes see canonical tokens."""
    out = query or ""
    for rx, repl in _QUERY_TYPO_SUBS_L0:
        out = rx.sub(repl, out)
    return out


_NUMERIC_ONLY_CUE_RE = re.compile(
    r"\bdigits?\s+only\b|\bonly\s+digits?\b"
    r"|\bnumeric\s+only\b|\bonly\s+numeric\b"
    r"|\bnumbers?\s+only\b|\bwith\s+digits?\s+only\b|\bdigit[\s-]only\b",
    re.IGNORECASE,
)


def _inject_exclude_letters(
    query: str, filters: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if not _NUMERIC_ONLY_CUE_RE.search(query):
        return filters
    if any(str(f.get("param") or "").strip() == "excludeLetters" for f in filters):
        return filters
    return filters + [{"param": "excludeLetters", "value": True}]


_OVER_PRICE_FLOOR_CUE_RE = re.compile(
    r"\b(?:over|above|more\s+than|greater\s+than)\s*[\$£€]?\d[\d,]*[\$]?\s*(?:k|thousand)?\b",
    re.IGNORECASE,
)


def _fix_over_price_direction(
    query: str, filters: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if not _OVER_PRICE_FLOOR_CUE_RE.search(query):
        return filters
    has_min = any(str(f.get("param") or "").strip() == "minPrice" for f in filters)
    has_max = any(str(f.get("param") or "").strip() == "maxPrice" for f in filters)
    if has_min or not has_max:
        return filters
    # "over/above N" already claimed by a different metric's own min* floor
    # (govalue, tf, cf, estibot, semrush, traffic, bids, age, ...) -- the maxPrice
    # here comes from an unrelated "under M" price clause, not a mislabeled price
    # floor. Compound "X above N ... under M" queries hit this every time; renaming
    # would double-corrupt an already-correct maxPrice (rule 8).
    other_min_claimed = any(
        str(f.get("param") or "").strip().startswith("min")
        and str(f.get("param") or "").strip() not in ("minPrice", "price_min")
        for f in filters
    )
    if other_min_claimed:
        return filters
    out = []
    for f in filters:
        name = str(f.get("param") or "").strip()
        if name == "maxPrice":
            val = f.get("value")
            try:
                val = int(val) + 1
            except (TypeError, ValueError):
                pass
            out.append({"param": "minPrice", "value": val})
        else:
            out.append(f)
    return out


_KNOWN_TLDS_FOR_BARE_EXCLUDE = frozenset(
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

_BARE_TLD_EXCLUDE_CUE_RE = re.compile(
    r"\b(?:exclude|no|without|avoid|except)\s+(?P<tld>[a-z]{2,6})(?:\s+domains?)?\b",
    re.IGNORECASE,
)


def _inject_bare_tld_exclude(
    query: str, filters: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    m = _BARE_TLD_EXCLUDE_CUE_RE.search(query)
    if not m:
        return filters
    tld = m.group("tld").lower()
    if tld not in _KNOWN_TLDS_FOR_BARE_EXCLUDE:
        return filters
    if any(str(f.get("param") or "").strip() == "tldExcludeList" for f in filters):
        return filters
    return filters + [{"param": "tldExcludeList", "value": tld}]


_BUDGET_CEILING_ONLY_RE = re.compile(
    r"\b(?:budget\s+)?(?:around|~)\s*\$?\d[\d,]*\s*(?:k|thousand)?\b"
    r".{0,32}\b(?:maybe\s+|perhaps\s+)?(?:less|under|below|or\s+(?:less|under|below))\b",
    re.IGNORECASE,
)


def _scrub_price_band_ceiling_only(
    query: str, filters: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if not _BUDGET_CEILING_ONLY_RE.search(query):
        return filters
    has_max = any(str(f.get("param") or "").strip() == "maxPrice" for f in filters)
    has_min = any(str(f.get("param") or "").strip() == "minPrice" for f in filters)
    if not (has_max and has_min):
        return filters
    max_val = next(
        (
            f.get("value")
            for f in filters
            if str(f.get("param") or "").strip() == "maxPrice"
        ),
        None,
    )
    min_val = next(
        (
            f.get("value")
            for f in filters
            if str(f.get("param") or "").strip() == "minPrice"
        ),
        None,
    )
    try:
        if int(max_val) - int(min_val) <= 1:
            return [
                f for f in filters if str(f.get("param") or "").strip() != "minPrice"
            ]
    except (TypeError, ValueError):
        pass
    return filters


_QUALITATIVE_LOW_PRICE_RE = re.compile(
    r"\b(?:low|cheap|affordable|inexpensive|budget(?:friendly)?)\s+"
    r"(?:price|cost|pricing|valued?)\b"
    r"|\bprice\s+is\s+(?:low|cheap)\b"
    r"|\bcurrent\s+price\s+is\s+(?:low|cheap)\b",
    re.IGNORECASE,
)


def _scrub_maxprice_zero_qualitative(
    query: str, filters: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if not _QUALITATIVE_LOW_PRICE_RE.search(query):
        return filters
    if re.search(r"\$?\d[\d,]*\s*(?:k|thousand)?", query, re.IGNORECASE):
        return filters
    out = []
    for f in filters:
        name = str(f.get("param") or "").strip()
        if name == "maxPrice" and f.get("value") == 0:
            out.append({"param": "price_below_market", "value": True})
        else:
            out.append(f)
    return out


_MONTHLY_VISITORS_CUE_RE = re.compile(
    r"\b(?:monthly\s+(?:visitors?|hits?|searches?|views?)"
    r"|unique\s+searches?\b"
    r"|visitors?\s+(?:per\s+)?month\b)",
    re.IGNORECASE,
)


def _fix_monthly_visitors_param(
    query: str, filters: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if not _MONTHLY_VISITORS_CUE_RE.search(query):
        return filters
    has_unique = any(
        str(f.get("param") or "").strip() == "minUniqueSearches" for f in filters
    )
    if has_unique:
        return filters
    out = []
    for f in filters:
        name = str(f.get("param") or "").strip()
        if name == "minTraffic":
            out.append({"param": "minUniqueSearches", "value": f.get("value")})
        else:
            out.append(f)
    return out


def filters_to_identified(
    filters: List[Dict[str, Any]],
    *,
    source: str,
    soft_slot_names: frozenset,
    confidence: float,
    query: str = "",
) -> List[Dict[str, Any]]:
    """Map ``{param,value}`` list to ``identified_filters`` (FIND-63 + soft/local)."""
    if not isinstance(source, str) or not source.strip():
        raise ConfigurationError("filters_to_identified requires non-empty source")
    if not isinstance(soft_slot_names, frozenset):
        raise ConfigurationError(
            "filters_to_identified requires soft_slot_names frozenset from qi.entity_slots"
        )
    if not soft_slot_names:
        raise ConfigurationError(
            "filters_to_identified soft_slot_names must be non-empty"
        )
    if not 0.0 <= float(confidence) <= 1.0:
        raise ConfigurationError("filters_to_identified confidence must be in [0,1]")
    confidence = float(confidence)
    source = source.strip()
    scrubbed = filters
    scrubbed = _scrub_is_gem_domain(scrubbed)
    if query:
        query = _normalize_query_typos(query)
        # Analytics -> empty; guidance with no inventory bound -> empty.
        # Shared speech-act detectors (advisory_patterns) — not suite literals.
        if is_strong_advisory(query) or is_soft_advisory_no_inventory(query):
            return []
        if _DUAL_HIGH_QUAL_RE.search(query) and not re.search(r"\d", query):
            return []
        scrubbed = _scrub_ungrounded_price_bounds(query, scrubbed)
        scrubbed = _scrub_dangling_under_price(query, scrubbed)
        scrubbed = _scrub_look_expensive_min_price(query, scrubbed)
        scrubbed = _scrub_dual_high_qualitative(query, scrubbed)
        scrubbed = _scrub_soft_or_link_floors(query, scrubbed)
        scrubbed = _inject_under_price(query, scrubbed)
        scrubbed = _inject_around_maybe_less_price(query, scrubbed)
        scrubbed = _inject_budget_prefixed_price(query, scrubbed)
        scrubbed = _inject_filter_price_currency(query, scrubbed)
        scrubbed = _normalize_or_under_inclusive(query, scrubbed)
        scrubbed = _scrub_false_exclude_letters(scrubbed)
        scrubbed = _normalize_registered_before(query, scrubbed)
        scrubbed = _scrub_unknown_age_with_price(query, scrubbed)
        scrubbed = _normalize_ending_weekend(query, scrubbed)
        scrubbed = _normalize_ending_soon(query, scrubbed)
        scrubbed = _scrub_ending_soon_expired_lifecycle(query, scrubbed)
        scrubbed = _normalize_tld_vs_topic(query, scrubbed)
        scrubbed = _normalize_exclusive_price_ceilings(query, scrubbed)
        scrubbed = _normalize_or_under_inclusive(query, scrubbed)
        scrubbed = _inject_leading_niche_topic(query, scrubbed)
        scrubbed = _inject_meets_cotopic(query, scrubbed)
        scrubbed = _inject_ai_or_fintech_topic(query, scrubbed)
        scrubbed = _scrub_non_industry_topics(query, scrubbed)
        scrubbed = _scrub_weak_price_below_market(query, scrubbed)
        scrubbed = _inject_strong_price_below_market(query, scrubbed)
        scrubbed = _inject_added_last_hour(query, scrubbed)
        scrubbed = _scrub_weak_start_time(query, scrubbed)
        scrubbed = _normalize_active_pending(query, scrubbed)
        scrubbed = _normalize_expired_dropping(query, scrubbed)
        scrubbed = _scrub_redundant_active_with_extended(query, scrubbed)
        scrubbed = _inject_traffic_unknown_zero_band(query, scrubbed)
        scrubbed = _inject_premium_extension(query, scrubbed)
        scrubbed = _scrub_invented_max_sld_len_from_price(query, scrubbed)
        scrubbed = _normalize_short_max_sld_len(query, scrubbed)
        scrubbed = _scrub_bare_short_min_sld_len(query, scrubbed)
        scrubbed = _inject_easy_say_exclude_digits(query, scrubbed)
        scrubbed = _inject_easy_say_exclude_hyphens(query, scrubbed)
        scrubbed = _scrub_ungrounded_exclude_digits(query, scrubbed)
        scrubbed = _scrub_ungrounded_keyword_exclude(query, scrubbed)
        scrubbed = _scrub_orphaned_keyword_match_mode(query, scrubbed)
        scrubbed = _scrub_quality_adjective_similar_to(query, scrubbed)
        scrubbed = _normalize_exclusive_min_floors(query, scrubbed)
        scrubbed = _normalize_zero_bids_band(query, scrubbed)
        scrubbed = _inject_exclude_letters(query, scrubbed)
        scrubbed = _fix_over_price_direction(query, scrubbed)
        scrubbed = _inject_bare_tld_exclude(query, scrubbed)
        scrubbed = _scrub_price_band_ceiling_only(query, scrubbed)
        scrubbed = _scrub_maxprice_zero_qualitative(query, scrubbed)
        scrubbed = _fix_monthly_visitors_param(query, scrubbed)
    identified: List[Dict[str, Any]] = []
    seen: set = set()
    for f in scrubbed:
        name = str(f.get("param") or "").strip()
        if not name or name in seen or name not in _FILTERABLE_PARAM_SET:
            continue
        value = f.get("value")
        if _is_qualitative_sentinel(value):
            continue
        seen.add(name)
        # Pipe-join list values for multi-value slots (similar_to, topic_include, etc.)
        if isinstance(value, list) and name in _MULTI_VALUE_SLOTS:
            value = "|".join(str(v) for v in value if str(v).strip())
        # Public identified_filters use FIND labels, not storage IDs (16 -> premium).
        if name == "typeIncludeList":
            value = normalize_type_include_list_for_public(value)
        identified.append(
            {
                "name": name,
                "value": value,
                "source": source,
                "chip_kind": _chip_kind_for_param(name, soft_slot_names),
                "confidence": confidence,
            }
        )
    return identified


# Reverse of _API_TO_INTERNAL (identity for names not renamed, e.g. keyword_contains,
# keyword_match_mode, lifecycle_state, tldExcludeList — API name == internal name there).
_INTERNAL_TO_API: Dict[str, str] = {v: k for k, v in _API_TO_INTERNAL.items()}


def _time_remaining_max_to_relative_offset(secs: int) -> Any:
    """Mirror l0_regex_filter_extractor's seconds->relative-string table (endTimeBefore parity)."""
    if secs == 86_400:
        return "-1d"
    if secs == 259_200:
        return "-3d"
    if secs == 604_800:
        return "-7d"
    if 0 < secs < 86_400:
        return f"-{max(1, secs // 3600)}h"
    if secs <= 259_200:
        return "-3d"
    if secs <= 604_800:
        return "-7d"
    return transform_slot_value("time_remaining_max", secs)


def entities_to_identified(
    entities: List[Entity],
    soft_slot_names: frozenset,
) -> List[Dict[str, Any]]:
    """Inverse of ``L0LLMFilterExtractor._identified_to_intent_slice``.

    Maps an internal-name ``Entity`` list (e.g. post ``apply_post_merge_reconcile``)
    back to API-shaped ``identified_filters`` dicts (``tldIncludeList``, ``typeIncludeList``,
    ``excludeHyphens``/``excludeDigits`` re-inverted, etc.) so ``sanitize_identified_filters`` /
    ``ground_identified_filters`` — which expect API param names — can consume it directly.
    """
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for ent in entities:
        internal = str(getattr(ent, "name", "") or "").strip()
        if not internal or internal in seen:
            continue
        seen.add(internal)
        value = ent.value
        if internal == "time_remaining_max":
            api_name = "endTimeBefore"
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                value = _time_remaining_max_to_relative_offset(int(value))
        else:
            api_name = _INTERNAL_TO_API.get(internal, internal)
        if api_name in _API_INVERT_BOOL and isinstance(value, bool):
            value = not value
        if api_name in ("typeIncludeList", "typeExcludeList"):
            # Reverse EntityGrounder.ground()'s label->numeric-ID expansion (needed by
            # retrieval) back to canonical labels for the public identified_filters shape.
            value = normalize_type_include_list_for_public(value)
        out.append(
            {
                "name": api_name,
                "value": value,
                "source": str(getattr(ent, "source", "") or ""),
                "chip_kind": "soft" if (internal in soft_slot_names or getattr(ent, "chip_kind", None) == "soft") else "hard",
                "confidence": float(getattr(ent, "confidence", 0.0) or 0.0),
            }
        )
    return out


def identified_to_ground_entities(identified: List[Dict[str, Any]]) -> List[Entity]:
    """Inverse of ``entities_to_identified`` — map API-shaped ``identified_filters``
    dicts back to internal-name ``Entity`` objects for a pure re-grounding call
    (``ground_hard_entities``).

    Deliberately NOT a reuse of ``L0LLMFilterExtractor._identified_to_intent_slice``:
    that method also applies allowed-name filtering, entity-count capping, and
    soft/hard splitting — L0-extraction-reconcile concerns, not grounding ones.
    Every item passed through ``ground_identified_filters`` is already hard, so
    this preserves each item's own ``confidence``/``source``/``chip_kind`` as-is.
    """
    out: List[Entity] = []
    seen: set = set()
    for item in identified:
        api_name = str(item.get("name") or "").strip()
        if not api_name:
            continue
        internal = _API_TO_INTERNAL.get(api_name, api_name)
        if internal in seen:
            continue
        value = item.get("value")
        if _is_qualitative_sentinel(value):
            continue
        if api_name in _API_INVERT_BOOL and isinstance(value, bool):
            value = not value
        value = _coerce_l0_filter_value(internal, value)
        if _is_qualitative_sentinel(value):
            continue
        seen.add(internal)
        out.append(
            Entity(
                name=internal,
                value=value,
                confidence=float(item.get("confidence", 0.9) or 0.9),
                source=str(item.get("source") or "L0_llm"),
                chip_kind=str(item.get("chip_kind") or "hard"),
            )
        )
    return out


def _coerce_l0_filter_value(slot: str, value: Any) -> Any:
    """Normalize LLM filter values to engine entity shapes (lists / bools).

    Multi-value slots (tld, auction_type, keyword_*, topic_*, exclude lists) MUST
    always be lists. L0 prompt emits pipe-joined strings (``\"app|dev\"``) or a
    bare single token (``\"com\"``, ``\"16\"``). A bare token without ``|`` used
    to pass through as a scalar str — downstream ``for t in filters['tld']`` then
    iterated characters (``'c','o','m'``), zeroing Qdrant MatchAny and hard-chip
    gate matches. Always wrap single tokens for ``_MULTI_VALUE_SLOTS``.
    """
    if isinstance(value, str):
        text = value.strip()
        low = text.lower()
        if low in ("true", "false"):
            return low == "true"
        if "|" in text:
            parts = [p.strip() for p in text.split("|") if p.strip()]
            if parts:
                return (
                    parts if len(parts) > 1 or slot in _MULTI_VALUE_SLOTS else parts[0]
                )
        if "," in text and slot in _MULTI_VALUE_SLOTS:
            parts = [p.strip() for p in text.split(",") if p.strip()]
            if parts:
                return parts
        if slot in _MULTI_VALUE_SLOTS:
            return [text] if text else []
        return text
    if (
        slot in _MULTI_VALUE_SLOTS
        and value is not None
        and not isinstance(value, (list, tuple, set, frozenset))
    ):
        return [value]
    return value


@dataclass
class L0CombinedExtractOutcome:
    """Result of a combined rewrite+extract LLM call (or extract-only follow-up)."""
    identified: List[Dict[str, Any]]
    keywords: List[Dict[str, Any]]
    cost_usd: float
    rewritten_query: str
    model_transformed: bool
    llm_completed: bool
    intent_slice: Optional[IntentSlice]


class L0LLMFilterExtractor:
    """Single-call L0 LLM extractor for qie_only, full-search, and grounding.

    Same prompt + FILTERABLE_PARAMS catalog for every path. Full-search uses
    ``classify_async`` (API names -> internal slots -> hard/soft IntentSlice).
    """

    def __init__(
        self,
        call_router: Any,
        task_type: str,
        *,
        entity_slots: QIEntitySlotsConfig,
        source_tag: str,
        max_entities: int,
        confidence: float,
        enabled: bool,
        prompt_tag: str,
        keyword_min_probability: float,
        combined_prompt_tag: str,
    ) -> None:
        if call_router is None:
            raise ConfigurationError("L0LLMFilterExtractor requires call_router")
        if not isinstance(task_type, str) or not task_type.strip():
            raise ConfigurationError(
                "L0LLMFilterExtractor requires non-empty task_type"
            )
        if entity_slots is None:
            raise ConfigurationError(
                "L0LLMFilterExtractor requires entity_slots (qi.entity_slots)"
            )
        if not isinstance(source_tag, str) or not source_tag.strip():
            raise ConfigurationError(
                "L0LLMFilterExtractor requires non-empty source_tag"
            )
        if int(max_entities) < 1:
            raise ConfigurationError("L0LLMFilterExtractor max_entities must be >= 1")
        if not 0.0 <= float(confidence) <= 1.0:
            raise ConfigurationError("L0LLMFilterExtractor confidence must be in [0,1]")
        if not isinstance(prompt_tag, str) or not prompt_tag.strip():
            raise ConfigurationError(
                "L0LLMFilterExtractor requires non-empty prompt_tag"
            )
        if not isinstance(combined_prompt_tag, str) or not combined_prompt_tag.strip():
            raise ConfigurationError(
                "L0LLMFilterExtractor requires non-empty combined_prompt_tag"
            )
        if not 0.0 <= float(keyword_min_probability) <= 100.0:
            raise ConfigurationError(
                "L0LLMFilterExtractor keyword_min_probability must be in [0,100]"
            )
        self._call_router = call_router
        self._task_type = task_type.strip()
        self._entity_slots = entity_slots
        self._soft_slots = entity_slots.soft_slot_set
        self._hard_names = entity_slots.hard_entity_set
        self._source_tag = source_tag.strip()
        self._max_entities = int(max_entities)
        self._confidence = float(confidence)
        self._enabled = bool(enabled)
        self._prompt_tag = prompt_tag.strip()
        self._combined_prompt_tag = combined_prompt_tag.strip()
        self._keyword_min_probability = float(keyword_min_probability) / 100.0
        if not FILTERABLE_PARAMS:
            raise ConfigurationError(
                "L0LLMFilterExtractor: FILTERABLE_PARAMS empty - check "
                "entity_slot_to_api_param.json / find_api_params.json"
            )

    @property
    def prompt_tag(self) -> str:
        """Config-driven prompt version tag used for LLM logs and cache isolation."""
        return self._prompt_tag

    @property
    def combined_prompt_tag(self) -> str:
        """Prompt tag for the combined rewrite+extract LLM call."""
        return self._combined_prompt_tag

    async def extract(self, query: str) -> List[Dict[str, Any]]:
        """Extract FIND + soft/local filters; raise ``LLMError`` when LLM unavailable."""
        identified, _cost, _keywords = await self.extract_priced(query)
        return identified

    async def extract_priced(
        self, query: str,
    ) -> Tuple[List[Dict[str, Any]], float, List[Dict[str, Any]]]:
        """Like ``extract`` but also returns USD cost and extracted keywords.

        Cost is ``0.0`` when disabled / empty query (no provider call). Uses the
        same ``compute_call_cost_usd`` rates as the call-router observer path.
        Keywords are ``{"term", "probability"}`` pairs from the same shared L0
        prompt/schema (no second LLM call) — reported independently of
        ``filters``/``identified`` and never routed through ``filters_to_identified``.
        Keywords below ``qi.l0_llm_entity.keyword_min_probability`` are dropped.
        """
        if not self._enabled:
            return [], 0.0, []
        if not isinstance(query, str) or not query.strip():
            return [], 0.0, []
        query = expand_gd_to_godaddy(query.strip())
        user_prompt = build_l0_filter_user_prompt([(1, query)])
        try:
            response, metadata = await self._call_router.call_structured(
                task_type=self._task_type,
                prompt_tag=self._prompt_tag,
                system_prompt=L0_FILTER_SYSTEM,
                user_prompt=user_prompt,
                response_schema=_L0BatchResponse,
                model_override=None,
            )
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface as LLM unavailable for sequential fallback
            logger.warning(
                f"l0_filter_extract_failed prompt_tag={self._prompt_tag} query_len={len(query)} "
                f"error_type={type(exc).__name__} error={exc}"
            )
            raise LLMError(
                f"l0_filter_extract_unavailable: {type(exc).__name__}: {exc}"
            ) from exc

        filters: List[Dict[str, Any]] = []
        keywords: List[Dict[str, Any]] = []
        for entry in response.results:
            if int(entry.idx) != 1:
                continue
            for f in entry.filters:
                param = str(f.param or "").strip()
                if param and param in _FILTERABLE_PARAM_SET:
                    filters.append({"param": param, "value": f.value})
            for kw in entry.keywords:
                term = str(kw.term or "").strip()
                if term:
                    keywords.append({"term": term, "probability": float(kw.probability)})
        keywords = [k for k in keywords if k["probability"] >= self._keyword_min_probability]
        keywords.sort(key=lambda k: k["probability"], reverse=True)
        meta = metadata or {}
        model = str(meta.get("model") or "")
        cost_usd = float(compute_call_cost_usd(model, meta.get("usage")))
        identified = filters_to_identified(
            filters,
            source=self._source_tag,
            soft_slot_names=self._soft_slots,
            confidence=self._confidence,
            query=query,
        )
        keywords = self._fill_keywords_if_empty(query, identified, keywords)
        logger.info(
            f"l0_filter_extract_done prompt_tag={self._prompt_tag} query_len={len(query)} "
            f"filters={len(filters)} keywords={len(keywords)} model={model} cost_usd={cost_usd:.6f}"
        )
        return identified, cost_usd, keywords

    async def extract_priced_combined(self, query: str) -> L0CombinedExtractOutcome:
        """One LLM call: rewrite + filters/keywords grounded on the rewritten query.

        Raises ``LLMError`` when the LLM call fails so callers can fall back to
        regex on raw/normalized (no transform).
        """
        if not self._enabled:
            q = query.strip() if isinstance(query, str) else ''
            return L0CombinedExtractOutcome(
                identified=[], keywords=[], cost_usd=0.0,
                rewritten_query=q, model_transformed=False,
                llm_completed=False, intent_slice=None,
            )
        if not isinstance(query, str) or not query.strip():
            return L0CombinedExtractOutcome(
                identified=[], keywords=[], cost_usd=0.0,
                rewritten_query='', model_transformed=False,
                llm_completed=False, intent_slice=None,
            )
        query = expand_gd_to_godaddy(query.strip())
        user_prompt = build_l0_combined_user_prompt([(1, query)])
        try:
            response, metadata = await self._call_router.call_structured(
                task_type=self._task_type,
                prompt_tag=self._combined_prompt_tag,
                system_prompt=L0_FILTER_SYSTEM,
                user_prompt=user_prompt,
                response_schema=_L0CombinedBatchResponse,
                model_override=None,
            )
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface as LLM unavailable for sequential fallback
            logger.warning(
                f"l0_filter_combined_failed prompt_tag={self._combined_prompt_tag} "
                f"query_len={len(query)} error_type={type(exc).__name__} error={exc}"
            )
            raise LLMError(
                f"l0_filter_combined_unavailable: {type(exc).__name__}: {exc}"
            ) from exc

        rewritten = query
        model_transformed = False
        filters: List[Dict[str, Any]] = []
        keywords: List[Dict[str, Any]] = []
        for entry in response.results:
            if int(entry.idx) != 1:
                continue
            candidate = str(entry.rewritten_query or "").strip()
            if candidate:
                rewritten = expand_gd_to_godaddy(candidate)
            model_transformed = bool(entry.transformed) or (
                rewritten.lower() != query.lower()
            )
            for f in entry.filters:
                param = str(f.param or "").strip()
                if param and param in _FILTERABLE_PARAM_SET:
                    filters.append({"param": param, "value": f.value})
            for kw in entry.keywords:
                term = str(kw.term or "").strip()
                if term:
                    keywords.append({"term": term, "probability": float(kw.probability)})
        keywords = [k for k in keywords if k["probability"] >= self._keyword_min_probability]
        keywords.sort(key=lambda k: k["probability"], reverse=True)
        meta = metadata or {}
        model = str(meta.get("model") or "")
        cost_usd = float(compute_call_cost_usd(model, meta.get("usage")))
        # Ground filters against the rewritten text (effective extract target).
        identified = filters_to_identified(
            filters,
            source=self._source_tag,
            soft_slot_names=self._soft_slots,
            confidence=self._confidence,
            query=rewritten,
        )
        keywords = self._fill_keywords_if_empty(rewritten, identified, keywords)
        intent_slice = self._identified_to_intent_slice(identified, rewritten, keywords)
        logger.info(
            f"l0_filter_combined_done prompt_tag={self._combined_prompt_tag} "
            f"query_len={len(query)} rewritten_len={len(rewritten)} "
            f"model_transformed={model_transformed} filters={len(filters)} "
            f"keywords={len(keywords)} model={model} cost_usd={cost_usd:.6f}"
        )
        return L0CombinedExtractOutcome(
            identified=identified,
            keywords=keywords,
            cost_usd=cost_usd,
            rewritten_query=rewritten,
            model_transformed=model_transformed,
            llm_completed=True,
            intent_slice=intent_slice,
        )

    def _fill_keywords_if_empty(
        self,
        query: str,
        identified: List[Dict[str, Any]],
        keywords: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Pass through LLM keywords; do not residual-fill when the model abstains.

        Empty ``keywords`` after a successful L0 call is a valid abstain (browse /
        quality / filter-only). Harvesting tokens from the query re-introduces the
        false positives the keyword contract forbids. Regex offline extract still
        uses ``_extract_topical_keywords`` on its own path.
        """
        del query, identified  # kept for call-site signature stability
        return list(keywords or [])

    def _identified_to_intent_slice(
        self,
        identified: List[Dict[str, Any]],
        query: str,
        keywords: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[IntentSlice]:
        """Map identified_filters (API / soft names) -> hard/soft IntentSlice."""

        allowed = self._soft_slots | self._hard_names | VALID_ENTITY_NAMES
        hard: List[Entity] = []
        soft: List[Entity] = []
        seen: set = set()
        for item in identified:
            api_name = str(item.get("name") or "").strip()
            if not api_name:
                continue
            internal = _API_TO_INTERNAL.get(api_name, api_name)
            if internal not in allowed or internal in seen:
                continue
            value = item.get("value")
            if _is_qualitative_sentinel(value):
                continue
            if api_name in _API_INVERT_BOOL and isinstance(value, bool):
                value = not value
            value = _coerce_l0_filter_value(internal, value)
            if _is_qualitative_sentinel(value):
                continue
            seen.add(internal)
            ent = Entity(
                name=internal,
                value=value,
                confidence=self._confidence,
                source=self._source_tag,
                chip_kind=infer_chip_kind(internal, self._hard_names),
            )
            if internal in self._soft_slots:
                soft.append(replace(ent, chip_kind="soft"))
            else:
                hard.append(ent)

        cap = self._max_entities
        if len(hard) + len(soft) > cap:
            soft_budget = max(0, cap - len(hard))
            soft = soft[:soft_budget]
            if len(hard) > cap:
                hard = hard[:cap]
                soft = []
        kw = list(keywords or [])
        if not hard and not soft:
            return IntentSlice(
                query_type="hybrid",
                entities=[],
                confidence=1.0,
                raw_text=query,
                soft_entities=[],
                keywords=kw,
            )
        logger.info(
            f"l0_filter_classify_done prompt_tag={self._prompt_tag} query_len={len(query)} "
            f"hard={len(hard)} soft={len(soft)} keywords={len(kw)} source={self._source_tag}"
        )
        return IntentSlice(
            query_type="hybrid",
            entities=hard,
            confidence=1.0,
            raw_text=query,
            soft_entities=soft,
            keywords=kw,
        )

    async def classify_async(self, query: str) -> Optional[IntentSlice]:
        """Full-search L0 entrypoint — same LLM extract as qie_only/grounding.

        Raises ``LLMError`` when the LLM call fails so the engine can run regex
        fallback (unavailable-only). Empty extract -> empty IntentSlice (LLM completed).
        """
        slice_, _cost = await self.classify_async_priced(query)
        return slice_

    async def classify_async_priced(
        self,
        query: str,
    ) -> Tuple[Optional[IntentSlice], float]:
        """Like ``classify_async`` but also returns USD cost of the L0 LLM call.

        Concurrent multi-intent fan-out must use this (not a shared last-cost
        field) so each sub-query's L0 spend is accounted without races.
        """
        if not self._enabled:
            return None, 0.0
        if not isinstance(query, str) or not query.strip():
            return None, 0.0
        query = expand_gd_to_godaddy(query.strip())
        identified, cost_usd, keywords = await self.extract_priced(query)
        return self._identified_to_intent_slice(identified, query, keywords), float(cost_usd)

    def classify(self, query: str) -> Optional[IntentSlice]:
        """Sync calibration producer — entity-only L0 has no type vote.

        Registry ``calibration_boot_fit`` calls ``.classify(query)`` and reads
        only ``(query_type, confidence)``. Filter L0 always emits ``hybrid``
        (entities come from ``classify_async`` / ``extract``). Return a cheap
        hybrid slice — do **not** call the LLM here or boot fit N-calls the
        provider for every golden seed.
        """
        if not self._enabled:
            return None
        if not isinstance(query, str) or not query.strip():
            return None
        return IntentSlice(
            query_type="hybrid",
            entities=[],
            confidence=1.0,
            raw_text=query.strip(),
            soft_entities=[],
        )
