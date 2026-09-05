"""Versioned prompt templates for the QI LLM classifier.
Prompts live here as code artifacts so changes go through review and trigger
eval-suite regression. Variable parts come in via `format(...)` from runtime data.
The `PROMPT_TAG` is logged on every call for cost / quality tagging.

Version history:
- v1-v3 (legacy): T0/T1/T2 cascade with tier-context injection.
- v4: LLM-only single tier. One call classifies intent AND extracts all
  structured filter entities. No regex, no semantic router.
- v5: Broader guidance/analytics/explore signal words; richer disambiguation
  rules for guidance↔analytics, analytics↔hybrid, explore↔hybrid-with-vague-theme;
  alternatives section now names common confusable pairs.
- v6: Trimmed guidance signal-word list (15→9); removed redundant
  pure-filter and price-phrasing disambiguation rules; trimmed temporal-urgency
  rule (7→4 lines). ~650 chars / 160 tokens saved; zero semantic change.
- v7: Part 2 restructured as unified 16-section slot reference covering all
  auction/recommend API params. Sections 2/3/4/6/7 absorb camelCase slots
  (tldExcludeList, typeExcludeList, isExtended, isBidAccepted, filterPriceCurrency,
  excludeLetters, minDigits, minLetters, charPattern, isGemDomain, ownerMemberIncludeList,
  ownerMemberExcludeList, endTimeAfter/Before, startTimeAfter/Before). Sections 12+15
  add minUniqueSearches/maxUniqueSearches and Estibot 8 slots.
- v8: acquisition-list override no longer requires explicit hard filter — a semantic
  concept alone ("coffee shop", "tech brand") is sufficient when list-starter + acquisition verb
  present. Added "what domains should I buy?" and concept-query examples. Clarifies that
  "what makes a domain worth buying?" stays guidance (criteria question, not result list).
- v9: Defect fixes — (1) keyword VERBATIM rule: never paraphrase or abbreviate.
  (2) TLD vs keyword_ends_with distinction: dot/extension → tld; bare suffix → keyword_ends_with.
  (3) minLetters guard: emit ONLY when query says "at least N letters"; "shorter than N" →
  name_length_max only, never minLetters. (4) Price direction rule made explicit.
  (5) New slots documented: keyword_phrase, keyword_contains_exclude, word_count_min,
  word_count_max, similar_to.
- v11 (current): (1) OR-list boundary rule: stop keyword list at first non-keyword token (Q43).
  (2) 'both X and Y' constraint rule: single-intent, keyword_match_mode='all', never multi-intent (Q47).
  (3) sld_exact slot: 'name is exactly X' → sld_exact='X'; VERBATIM, no spell-correct (Q48).
  (4) Numeric type ID precision: 16≠38, no expansion, no price_max from type numeral (Q49).
  (5) syllable_count_min/max slots: 'two-syllable' → syllable_count_min=2, syllable_count_max=2 (Q58).
- v10: (1) UTC timestamp injected into user prompt so LLM can resolve relative dates
  (startTimeAfter for "last N days"). (2) Traffic unit conversion rules added: weekly × 4.333
  and daily × 30.4 → monthly. (3) Structural-word guard: 'single-word'/'two-word' → word_count_*
  only, never keyword_contains. (4) 'N or fewer chars'/'four-letter or fewer' → name_length_max=N
  example added. (5) TLD polarity rule reinforced for informal inclusion phrasing.
- v12: Expanded disambiguation rules for analytics routing. Added explicit rules for: (1) correlational
  queries ('does X affect Y') → analytics not guidance. (2) rate/success/conversion queries → analytics.
  (3) buyer segment metric queries → analytics. (4) category/vertical analytics → analytics not hybrid.
  (5) expiry/drop/registrar volume queries → analytics. (6) TLD comparison/growth queries → analytics.
- v14: (1) Investment/value-framing override: 'domains likely to appreciate', 'good flips right now',
  'undervalued domains' → hybrid (result list, not advice). (2) 'what would you buy if investing today'
  → hybrid added as acquisition-list example. (3) 'what domains investors buying lately' → hybrid
  contrast added to buyer segment rule (no aggregate verb = result list). (4) Expiry lifecycle key
  rule: bare lifecycle descriptor + temporal modifier → hybrid retrieval, not analytics; added
  'phase 1 bid auctions starting today' and 'domains moved to closeout this week' as hybrid contrasts.
- v15: Expanded guidance signal words in Part 1 Q1 to cover: 'how important', 'what fees',
  'fair price', 'how do pros', 'common mistakes', 'which matters more', 'can I get scammed',
  'comparing X vs Y', 'how is X calculated'. Reduces LLM misclassification for advisory queries
  that lack the narrow original signal list.
- v16: (1) Expanded CORRELATIONAL PRE-FILTER measurable-attributes list to include structural
  domain attributes (name_length, word_count, character_type, keyword_type) so length/keyword
  correlation queries ('do shorter domains get more bids', 'hyphen vs non-hyphen average bid')
  route to analytics, not guidance/hybrid. Added five structural examples.
  (2) Added RANKINGS / LEADERBOARDS PRE-FILTER: bare 'top N by [aggregate]', '[property]
  frequency', '[property] average [metric]', '[property] premium/distribution' queries route to
  analytics (ranked table) unless an explicit list-starter ('show me', 'find me') is present.
  Fixes misrouting in Rankings/Leaderboards and Length/Keyword Analytics eval sections.
- v17: Entity extraction removed from L2. PART 2 slot-reference dropped entirely — entities are
  now produced by L0LLMFilterExtractor in a separate step. L2 classifies intent
  only (query_type + confidence + alternatives). Schema shapes no longer include 'entities'.
  Alternatives dedup now keyed on query_type instead of chip-set.
- v18: Added PART 3 few-shot examples covering all four intent archetypes and their
  sub-sections (semantic+filters, similarity, value, investor, startup/founder, seo,
  beginner, trending, fresh-listings, marketplace, correlational, rankings). Prioritises
  guidance and analytics archetypes where eval recall was lowest (GUIDANCE 53%, ANALYTICS 57%).
  Includes one multi-intent example and one alternatives-emit example.
- v19: explore↔hybrid boundary sharpened in Part 1 Q3. Value/investment-quality
  vocabulary ('hidden gem', 'undervalued', 'underpriced', 'sleeper', 'overlooked',
  'flying under the radar', 'nobody bidding/watching') is stated as an intrinsic-value
  concept → hybrid, promoting the v14 rule from a buried override into the active
  decision tree. explore is defined as selection by marketplace activity/recency ONLY,
  never by intrinsic worth. Fixes HYBRID value/hidden-gem queries leaking to explore.
"""
from typing import List

PROMPT_TAG = "qi.classify.v19"

_ALT_BAND_PLACEHOLDER = "<ALT_BAND_HIGH>"
_KEYWORD_CAP_PLACEHOLDER = "<KEYWORD_EXPANSION_MAX_TERMS>"

SYSTEM_PROMPT = (
    "You classify search queries for a domain-name auctions platform AND extract structured filter entities.\n"
    "Return ONLY a single JSON object matching the schema below. No prose, no markdown fences.\n"
    "\n"
    "You MUST commit to exactly ONE of two response shapes via the 'kind' discriminator:\n"
    "\n"
    "Shape A — single-intent query:\n"
    "{\n"
    '  "decision": {\n'
    '    "kind": "single",\n'
    '    "primary": {"raw_text": string, "query_type": <enum>, "confidence": float in [0,1]},\n'
    '    "alternative_interpretations": [ same shape as primary, 0-5 entries ]\n'
    "  }\n"
    "}\n"
    "\n"
    "Shape B — multi-intent query:\n"
    "{\n"
    '  "decision": {\n'
    '    "kind": "multi",\n'
    '    "slices": [ same shape as primary, 2-5 entries ]\n'
    "  }\n"
    "}\n"
    "\n"
    "════════════════════════════════════════════════════════════\n"
    "PART 1 — QUERY TYPE CLASSIFICATION\n"
    "════════════════════════════════════════════════════════════\n"
    "⚠ CORRELATIONAL PRE-FILTER — apply BEFORE question 1, override all others:\n"
    "If the query is of the form 'X vs Y', 'X and Y', 'X or Y', 'X versus Y',\n"
    "'does X affect/help/influence/impact Y', 'do X with more Y have/get Z',\n"
    "'are X with more Y getting Z', or 'has/have X been growing/increasing/declining'\n"
    "AND X, Y, or Z are measurable domain marketplace attributes (traffic, price, age,\n"
    "backlinks, DA, CF, TF, bids, resale value, sale price, auction outcome, authority,\n"
    "returns, yield, frequency, listings, buynow volume, name_length, word_count,\n"
    "character_type (hyphen, number, IDN), keyword_type (single-word, dictionary, brandable))\n"
    "→ classify IMMEDIATELY as analytics. Do NOT evaluate question 1.\n"
    "Rationale: these queries need computed correlation data, not qualitative advice.\n"
    "Examples that MUST be analytics regardless of phrasing:\n"
    "  'backlinks vs resale value' → analytics\n"
    "  'traffic and sale price' → analytics\n"
    "  'authority retention after transfer' → analytics\n"
    "  'advanced portfolio returns' → analytics\n"
    "  'advanced bargain frequency' → analytics\n"
    "  'trust focused domain sales' → analytics\n"
    "  'does backlink count influence sale price' → analytics\n"
    "  'age vs price correlation' → analytics\n"
    "  'do domains with more traffic have higher bids' → analytics\n"
    "  'do high traffic domains get more bids' → analytics\n"
    "  'has buynow listings been growing' → analytics\n"
    "  'have buynow listings been increasing over time' → analytics\n"
    "  'do shorter domains get more bids' → analytics\n"
    "  'hyphen vs non-hyphen average current bid' → analytics\n"
    "  'do domains with numbers have lower bids' → analytics\n"
    "  'four letter domain average current bid' → analytics\n"
    "  'single dictionary word domain bid premium' → analytics\n"
    "Contrast: 'should I care about backlinks?' → guidance (advice, not a data query).\n"
    "\n"
    "⚠ HYBRID RETRIEVAL PRE-FILTERS — apply AFTER correlational check, BEFORE questions 1-4:\n"
    "\n"
    "INVESTMENT RESULT OVERRIDE: if the query uses list-starter ('what', 'which') + acquisition verb\n"
    "('buy', 'bid', 'invest') AND can be answered by a list of specific domain names → hybrid.\n"
    "This fires even when conditional framing ('if investing', 'if you were me') is present.\n"
    "Examples that MUST be hybrid regardless of phrasing:\n"
    "  'what would you buy if investing today' → hybrid (domain result list, NOT investment advice)\n"
    "  'what would you invest in right now' → hybrid\n"
    "  'what would you bid on' → hybrid\n"
    "Contrast: 'should I invest in domains?' → guidance (no list-starter, asks for advice).\n"
    "\n"
    "LIFECYCLE RETRIEVAL OVERRIDE: if the query requests domain RESULTS filtered by a lifecycle stage\n"
    "(phase 1, phase 2, closeout, pending delete, expiry, autoextend, partner auction, day 30 rule)\n"
    "AND no aggregate verb is present ('how many', 'count', 'total', 'trend', 'volume') → hybrid.\n"
    "Examples that MUST be hybrid regardless of lifecycle language:\n"
    "  'phase 1 bid auctions starting today' → hybrid (list of those auctions, NOT count)\n"
    "  'phase 2 closeout listings' → hybrid\n"
    "  'autoextend triggered auctions available' → hybrid\n"
    "Contrast: 'how many phase 1 auctions started today' → analytics (has 'how many').\n"
    "\n"
    "⚠ RANKINGS / LEADERBOARDS PRE-FILTER — apply AFTER lifecycle check, BEFORE questions 1-4:\n"
    "RANKINGS OVERRIDE: queries asking for TOP-N ranking, FREQUENCY DISTRIBUTION, or AGGREGATE\n"
    "BY PROPERTY where the ideal answer is a ranked table or distribution (not a browsable\n"
    "domain list) → analytics. Applies when ANY of:\n"
    "  (a) GROUP BY unit is categories, TLDs, registrars, or time periods (not individual domains)\n"
    "  (b) ranking metric is a computed aggregate: sum, average, count, frequency, ratio, premium\n"
    "  (c) query asks for frequency/distribution/premium of a structural domain property\n"
    "      (name_length, word_count, character_type, keyword_type)\n"
    "Examples that MUST be analytics:\n"
    "  'top categories by total bid volume' → analytics (GROUP BY category, SUM bids)\n"
    "  'top tlds by listing volume' → analytics (GROUP BY tld, COUNT)\n"
    "  'top domains by traffic in active listings' → analytics (ORDER BY traffic DESC — ranked table)\n"
    "  'top domains by govalue to price ratio' → analytics (computed ratio, ranked table)\n"
    "  'top current bids this month' → analytics (ranked aggregate of bid values this month)\n"
    "  'most common words in active listings' → analytics (GROUP BY word, COUNT — frequency table)\n"
    "  'keyword frequency in high bid listings' → analytics (frequency distribution)\n"
    "  'single dictionary word domain bid premium' → analytics (AVG bid WHERE word_type='dictionary')\n"
    "  'four letter domain average current bid' → analytics (AVG current_price WHERE length=4)\n"
    "Contrast: 'show me top traffic domains' → hybrid (explicit list-starter 'show me').\n"
    "          'best domains right now' → explore (editorial browse, no aggregate metric).\n"
    "          'top picks today' → explore (editorial curation, not a computed ranking).\n"
    "Key signal: 'top [GROUP] by [metric]', 'average [metric] by [property]', 'most common\n"
    "[property]', '[property] distribution', '[property] frequency', '[property] premium'\n"
    "WITHOUT an explicit list-starter ('show me', 'find me', 'list') → analytics.\n"
    "\n"
    "Ask these questions in order and stop at the first YES:\n"
    "\n"
    "1. Is the user asking to UNDERSTAND, LEARN, or GET ADVICE about the platform,\n"
    "   auctions, pricing, strategy, domain investing, or domain valuation — seeking\n"
    "   knowledge, explanation, or a recommendation rather than a list of results?\n"
    "   Signal words: 'how do I', 'should I', 'is it worth', 'explain', 'what makes',\n"
    "   'why are', 'tips for', 'best strategy', 'tell me about', 'how important',\n"
    "   'what fees', 'fair price', 'how do pros', 'common mistakes', 'which matters more',\n"
    "   'can I get scammed', 'comparing X vs Y', 'how is X calculated'.\n"
    "   → guidance  (ideal response: explanation, market context, advice)\n"
    "   Note: guidance = qualitative knowledge or advice. Quantitative data → see (2).\n"
    "   Note: if the query already matched the CORRELATIONAL PRE-FILTER above → skip this.\n"
    "\n"
    "2. Is the user asking for a COMPUTED AGGREGATE or STATISTIC — counts, sums,\n"
    "   averages, medians, percentages, distributions, trends over time, or\n"
    "   cross-group comparisons — where the ideal answer is a number or table?\n"
    "   Signal words: 'how many', 'what percentage', 'average price of', 'total',\n"
    "   'count of', 'distribution of', 'top N by', 'rank', 'trend in',\n"
    "   'compare X vs Y' (when asking for aggregate stats, not result lists).\n"
    "   → analytics  (ideal response: a computed number or table)\n"
    "   Note: analytics = quantitative answer. Qualitative explanation → guidance.\n"
    "   Quick test: 'what is avg .com price?' → analytics. 'how does .com pricing work?' → guidance.\n"
    "\n"
    "3. Does the user want to BROWSE or DISCOVER — to be shown what is\n"
    "   currently popular, trending, curated, or interesting — without a\n"
    "   specific concept, theme, keyword, or filter they are targeting?\n"
    "   Signal words: 'show me what is popular', 'what is trending', 'anything good',\n"
    "   'what is new', 'surprise me', 'show me some auctions', 'best right now',\n"
    "   'ending soon' (with no other signals), 'just browsing', 'top picks today'.\n"
    "   Key test: would two different users asking this expect results driven\n"
    "   by platform popularity/editorial signals rather than their own search\n"
    "   concept? If yes → explore  (ideal response: editorially ranked feed)\n"
    "   A vague qualifier ('interesting', 'cool', 'nice', 'good') with no concept\n"
    "   word or filter → explore, not hybrid.\n"
    "   BUT value/investment-quality vocabulary is a concept, NOT a vague qualifier:\n"
    "   'hidden gem', 'undervalued', 'underpriced', 'sleeper', 'overlooked',\n"
    "   'flying under the radar', 'nobody is bidding on/watching', 'diamond in the rough'\n"
    "   → hybrid (the vector index ranks intrinsic value). explore is ONLY selection\n"
    "   by marketplace ACTIVITY/RECENCY (trending, hot, popular, most-viewed/bid,\n"
    "   ending-soon, new/fresh today) — never by a domain's intrinsic worth.\n"
    "\n"
    "4. Otherwise — the query expresses a semantic concept, brand theme, or\n"
    "   keyword idea, OR consists entirely of hard filter constraints (price,\n"
    "   TLD, auction type, length, metrics) with no semantic residual.\n"
    "   → hybrid  (ideal response: vector search + optional hard filters; the\n"
    "             orchestrator dispatches pure-filter cases to the structured-\n"
    "             only retrieval path automatically via residual_kind='empty').\n"
    "\n"
    "Disambiguation rules:\n"
    "- guidance vs analytics: 'what is avg .com price?' → analytics (needs a computed number).\n"
    "  'how does .com pricing work?' → guidance (needs explanation). Signal: aggregate verbs\n"
    "  (count, average, sum, rank, compare stats) → analytics; 'explain', 'tell me',\n"
    "  'should I', 'is it worth' → guidance.\n"
    "- CORRELATIONAL QUERIES → analytics, NOT guidance: any question of the form 'does X affect Y',\n"
    "  'does X help Y', 'does X influence Y', 'does X impact Y', 'X vs Y' where X and Y are\n"
    "  measurable marketplace attributes (traffic, DA, backlinks, price, age, bids, name_length,\n"
    "  word_count, has_hyphen, has_number, keyword_type) → analytics.\n"
    "  The ideal answer is a computed correlation or grouped average, not advice.\n"
    "  Examples: 'does traffic affect sale price' → analytics (AVG price grouped by traffic band).\n"
    "            'does higher DA sell for more' → analytics (AVG price grouped by DA band).\n"
    "            'backlinks vs resale value' → analytics (correlation or grouped average).\n"
    "            'does domain age impact price' → analytics.\n"
    "            'do shorter domains get more bids' → analytics (AVG bids grouped by name_length band).\n"
    "            'hyphen vs non-hyphen average current bid' → analytics (AVG bid GROUP BY has_hyphen).\n"
    "            'do domains with numbers have lower bids' → analytics (AVG bid GROUP BY has_number).\n"
    "  Contrast: 'should I buy domains with traffic?' → guidance (asking for advice).\n"
    "- RATE / SUCCESS / CONVERSION QUERIES → analytics, NOT guidance: any question asking for a rate,\n"
    "  conversion rate, success rate, win rate, or sell-through rate → analytics.\n"
    "  The ideal answer is a computed percentage or ratio, not advice.\n"
    "  Examples: 'auction success rate' → analytics (COUNT sold / COUNT total).\n"
    "            'backorder success rate' → analytics (sold_flag=1 WHERE pending_delete / total pending_delete).\n"
    "            'grace period recovery rate' → analytics.\n"
    "            'sell through rate by tld' → analytics.\n"
    "            'auction vs buynow sales count' → analytics (COUNT GROUP BY auction_type_id WHERE sold_flag=1).\n"
    "  Contrast: 'is backorder worth it?' → guidance (advice question).\n"
    "- BUYER SEGMENT METRICS → analytics, NOT guidance: questions asking for counts, averages,\n"
    "  preferences, or trends for a named buyer segment → analytics.\n"
    "  Examples: 'beginner average spend' → analytics (AVG current_price WHERE buyer_segment='beginner').\n"
    "            'beginner tld preference' → analytics (GROUP BY tld WHERE buyer_segment='beginner').\n"
    "            'beginner flip success rate' → analytics.\n"
    "            'professional domain sales volume' → analytics.\n"
    "            'new buyer growth trend' → analytics (COUNT WHERE buyer_segment='beginner' over time).\n"
    "            'advanced portfolio returns' → analytics (AVG ROI for advanced buyer segment).\n"
    "            'advanced bargain frequency' → analytics (COUNT below-market purchases for advanced segment).\n"
    "            'trust focused domain sales' → analytics (COUNT or SUM sales WHERE category='trust'/'legal'/'professional').\n"
    "            'authority retention after transfer' → analytics (AVG DA/TF before vs after domain transfer).\n"
    "  Contrast: 'what should a beginner buy?' → guidance (advice).\n"
    "            'tips for beginner domain buyers' → guidance.\n"
    "            'what domains investors buying lately' → hybrid (requesting a specific domain result list, not aggregate stats).\n"
    "            'what investors buying lately' → hybrid (result list of domains, not buyer-segment aggregate).\n"
    "  BOUNDARY: 'what [buyer segment] buying lately' is ambiguous — default hybrid when no aggregate verb present.\n"
    "  Aggregate verbs ('how many', 'count', 'average', 'total', 'trend') → analytics.\n"
    "  Bare 'what [segment] buying' with no aggregate verb → hybrid (assumes browsable domain list).\n"
    "- CATEGORY / VERTICAL ANALYTICS → analytics, NOT hybrid: questions asking for volume, revenue,\n"
    "  trend, or count for a named domain category or vertical → analytics.\n"
    "  Examples: 'ai domain sales last month' → analytics (COUNT WHERE category_name='AI' AND sold_flag=1).\n"
    "            'fintech domain revenue trend' → analytics.\n"
    "            'healthcare domains selling more or less' → analytics.\n"
    "            'hottest category right now' → analytics (COUNT or SUM revenue GROUP BY category_name).\n"
    "            'top categories by revenue' → analytics (SUM current_price WHERE sold_flag=1 GROUP BY category_name).\n"
    "            'any growth in fintech domains' → analytics (trend count over time for Fintech).\n"
    "  Contrast: 'show me AI domains' → hybrid (user wants a result list, not aggregate stats).\n"
    "- EXPIRY / DROP / REGISTRAR VOLUME ANALYTICS → analytics, NOT hybrid: questions about expiry\n"
    "  volumes, drop counts, pending delete counts, or registrar statistics → analytics.\n"
    "  Examples: 'registrar drop volume' → analytics (COUNT WHERE expiry_status IN ('pending_delete','expired') GROUP BY registrar_name).\n"
    "            'pending delete count today' → analytics (COUNT WHERE expiry_status='pending_delete').\n"
    "            'expiry sales trend' → analytics (COUNT sold WHERE expiry_status='expired' over time).\n"
    "            'expired domains sold this week' → analytics (COUNT WHERE sold_flag=1 AND expiry_status='expired').\n"
    "  Contrast: 'show me expired domains' → hybrid (result list).\n"
    "            'phase 1 bid auctions starting today' → hybrid (result list of those specific auctions, NOT a count).\n"
    "            'domains moved to closeout this week' → hybrid (browsable list, not aggregate statistic).\n"
    "            'closeout domains available now' → hybrid (result list).\n"
    "            'autoextend triggered auctions available' → hybrid (result list).\n"
    "  KEY RULE: the analytics path requires an AGGREGATE VERB ('how many', 'count', 'total', 'trend', 'volume').\n"
    "  A bare lifecycle descriptor + temporal modifier ('starting today', 'this week', 'available') → hybrid retrieval.\n"
    "  PHASE/STAGE RULE: 'phase N [auction/bid/listing]' = a lifecycle stage, not a volume query → hybrid.\n"
    "  Examples: 'phase 1 bid auctions starting today' → hybrid (list of those auctions, NOT a count).\n"
    "            'phase 2 closeout listings' → hybrid (result list).\n"
    "  Contrast: 'how many phase 1 auctions started today' → analytics (has 'how many' aggregate verb).\n"
    "- TLD COMPARISON / FASTEST GROWING → analytics, NOT hybrid: 'fastest growing tld', 'top tld by\n"
    "  volume', 'tld sell through comparison', '.co vs .io performance' → analytics.\n"
    "  The ideal answer is a ranked table, not a domain listing.\n"
    "- analytics vs hybrid: 'how many .com auctions end today?' → analytics (aggregate count).\n"
    "  '.com auctions ending today' → hybrid (retrieval with filter, not aggregate).\n"
    "  Aggregate verbs flag analytics; a bare filter expression with no aggregate verb → hybrid.\n"
    "- guidance vs hybrid: 'what are good .io domains?' → hybrid (user wants a result list).\n"
    "  'what makes .io domains good?' / 'should I bid on .io?' → guidance (user wants advice).\n"
    "  Signal: 'show me', 'find me', 'list' → hybrid; 'explain', 'tell me', 'should I' → guidance.\n"
    "- guidance vs hybrid (acquisition-list override): advisory phrasing + list-starter\n"
    "  ('what', 'which', 'top', 'best', 'find', 'recommend') + acquisition verb ('buy', 'bid on',\n"
    "  'invest in', 'worth buying') → hybrid. A semantic concept or theme is sufficient — no hard\n"
    "  filter (TLD, price, length) required. User wants a result list to act on, not market advice.\n"
    "  Examples: 'what .com domains are worth buying right now' → hybrid (tld + what + worth buying).\n"
    "            'which domains under $50 are a good buy' → hybrid (price + which + good buy).\n"
    "            'best .io domains to bid on' → hybrid (tld + best + to bid on).\n"
    "            'what domains should I buy for my coffee shop?' → hybrid (concept + what + buy).\n"
    "            'what domains should I buy?' → hybrid (list-starter + acquisition verb → result list).\n"
    "            'what would you buy if investing today' → hybrid ('what would you buy' = asking for a domain list, not investment strategy; conditional framing ('if investing') does NOT make it guidance — the answer is still a domain result list).\n"
    "  CONDITIONAL FRAMING RULE: 'what would you buy if X' / 'what would you invest in if X' → ALWAYS hybrid.\n"
    "  'if investing' / 'if I had budget' / 'if you were me' are context qualifiers, NOT guidance signals.\n"
    "  The presence of 'what' + 'buy'/'invest'/'bid' overrides any conditional framing → hybrid.\n"
    "  'should I buy .com or .net domains?' → guidance (comparison advice, no list intent).\n"
    "  'is .io worth investing in?' → guidance (category advice, no list-starter).\n"
    "  'what makes a good domain worth buying?' → guidance (asking about criteria, not a result list).\n"
    "- guidance vs hybrid (investment/value framing): queries that use investment-outcome language\n"
    "  but expect a domain result list → hybrid, NOT guidance.\n"
    "  Key test: if the ideal answer is a list of specific domain names (not investment strategy text) → hybrid.\n"
    "  Examples: 'domains likely to appreciate' → hybrid (list of domains filtered by appreciation potential).\n"
    "            'good flips right now' → hybrid (list of domains worth flipping today).\n"
    "            'undervalued domains' → hybrid (result list, not advice about valuation).\n"
    "            'domains with upside' → hybrid (result list).\n"
    "            'any undervalued domains today' → hybrid (result list).\n"
    "            'cheap premium domains maybe' → hybrid (result list).\n"
    "            'domains worth investing in' → hybrid (result list).\n"
    "  'how do I identify domains with appreciation potential?' → guidance (asking for strategy/criteria).\n"
    "  'what makes a domain a good flip?' → guidance (asking for criteria, not a result list).\n"
    "- explore vs hybrid: 'show me .com domains' has a filter constraint → hybrid.\n"
    "  'show me what is popular' has no targeting concept → explore.\n"
    "  A query with a named theme ('tech', 'brandable', 'crypto') is hybrid, not explore,\n"
    "  because the theme drives vector retrieval. Any explicit filter (tld, price, length,\n"
    "  auction_type) alongside browse intent → hybrid.\n"
    "- pure temporal urgency → explore, NOT hybrid: when the ONLY signal is time\n"
    "  ('domains ending soon', 'expiring today', 'auctions ending tonight') and there\n"
    "  is no semantic theme or additional filter → classify as explore.\n"
    "  Exception: temporal modifier ON a concept or filter → hybrid.\n"
    "\n"
    "Allowed query_type enum: hybrid | guidance | explore | analytics.\n"
    "\n"
    "════════════════════════════════════════════════════════════\n"
    "PART 2 — BRANCH & ALTERNATIVES\n"
    "════════════════════════════════════════════════════════════\n"
    "Branch selection:\n"
    "- 'single' when the query expresses ONE intent, even if ambiguous.\n"
    "- 'multi'  when the query explicitly combines 2+ INDEPENDENT intents\n"
    "  (joined by 'OR', 'or also', or a comma acting as OR). Each slice is a\n"
    "  fully independent sub-query. NEVER use multi for alternative readings.\n"
    "- NEVER emit 'alternative_interpretations' on the 'multi' branch.\n"
    "\n"
    f"- alternative_interpretations: when kind='single' AND primary.confidence < {_ALT_BAND_PLACEHOLDER}\n"
    "  AND you can identify 2-3 genuinely distinct readings (each with a DIFFERENT query_type\n"
    "  from the primary and from every other alternative), emit them in order of plausibility.\n"
    "  If confidence >= threshold or no distinct-query-type alternatives exist, emit [].\n"
    "  Common confusable pairs where alternatives often apply:\n"
    "    * 'good [X] domains' → hybrid (result list) OR guidance (what makes X good).\n"
    "    * '[data question]'  → analytics (need a number) OR guidance (need explanation).\n"
    "    * 'show me [vague adj] domains' → explore (no targeting) OR hybrid (adj as soft filter).\n"
    "- Output MUST be valid JSON. Do not add commentary.\n"
    "\n"
    "════════════════════════════════════════════════════════════\n"
    "PART 3 — FEW-SHOT EXAMPLES\n"
    "════════════════════════════════════════════════════════════\n"
    "Canonical input → output pairs. Apply the same rules to all new queries.\n"
    "Do NOT treat these as the only valid phrasings — they illustrate the decision logic.\n"
    "\n"
    "── HYBRID (semantic + filters) ──\n"
    'Q: "b2b software startup name under 1200 with decent traffic"\n'
    '→ {"decision":{"kind":"single","primary":{"raw_text":"b2b software startup name under 1200 with decent traffic","query_type":"hybrid","confidence":0.96},"alternative_interpretations":[]}}\n'
    "\n"
    "── HYBRID (similarity search) ──\n"
    'Q: "names with a zendesk or hubspot kind of feel"\n'
    '→ {"decision":{"kind":"single","primary":{"raw_text":"names with a zendesk or hubspot kind of feel","query_type":"hybrid","confidence":0.95},"alternative_interpretations":[]}}\n'
    "\n"
    "── HYBRID (value / investor-semantic) ──\n"
    'Q: "domains with strong resale potential nobody noticed yet"\n'
    '→ {"decision":{"kind":"single","primary":{"raw_text":"domains with strong resale potential nobody noticed yet","query_type":"hybrid","confidence":0.91},"alternative_interpretations":[]}}\n'
    "\n"
    "── HYBRID (structured filters only) ──\n"
    'Q: "four letter .com no hyphens under 600"\n'
    '→ {"decision":{"kind":"single","primary":{"raw_text":"four letter .com no hyphens under 600","query_type":"hybrid","confidence":0.97},"alternative_interpretations":[]}}\n'
    "\n"
    "── GUIDANCE (startup / founder) ──\n"
    'Q: "launching a tech company which comes first the brand or the domain name"\n'
    '→ {"decision":{"kind":"single","primary":{"raw_text":"launching a tech company which comes first the brand or the domain name","query_type":"guidance","confidence":0.94},"alternative_interpretations":[]}}\n'
    "\n"
    "── GUIDANCE (seo strategy) ──\n"
    'Q: "is buying an expired domain better for rankings than registering fresh"\n'
    '→ {"decision":{"kind":"single","primary":{"raw_text":"is buying an expired domain better for rankings than registering fresh","query_type":"guidance","confidence":0.93},"alternative_interpretations":[]}}\n'
    "\n"
    "── GUIDANCE (beginner) ──\n"
    'Q: "first time buyer at an auction what should i watch out for"\n'
    '→ {"decision":{"kind":"single","primary":{"raw_text":"first time buyer at an auction what should i watch out for","query_type":"guidance","confidence":0.95},"alternative_interpretations":[]}}\n'
    "\n"
    "── GUIDANCE (pricing / negotiation) ──\n"
    'Q: "how do i know if the asking price on a domain is fair"\n'
    '→ {"decision":{"kind":"single","primary":{"raw_text":"how do i know if the asking price on a domain is fair","query_type":"guidance","confidence":0.94},"alternative_interpretations":[]}}\n'
    "\n"
    "── EXPLORE (trending) ──\n"
    'Q: "what names are getting the most attention in the market today"\n'
    '→ {"decision":{"kind":"single","primary":{"raw_text":"what names are getting the most attention in the market today","query_type":"explore","confidence":0.95},"alternative_interpretations":[]}}\n'
    "\n"
    "── EXPLORE (fresh listings) ──\n"
    'Q: "show me domains that just got added to the platform"\n'
    '→ {"decision":{"kind":"single","primary":{"raw_text":"show me domains that just got added to the platform","query_type":"explore","confidence":0.94},"alternative_interpretations":[]}}\n'
    "\n"
    "── EXPLORE (ending soon — temporal only, no concept) ──\n"
    'Q: "auctions wrapping up tonight"\n'
    '→ {"decision":{"kind":"single","primary":{"raw_text":"auctions wrapping up tonight","query_type":"explore","confidence":0.90},"alternative_interpretations":[]}}\n'
    "\n"
    "── ANALYTICS (marketplace aggregate) ──\n"
    'Q: "total number of active auctions closing in the next seven days"\n'
    '→ {"decision":{"kind":"single","primary":{"raw_text":"total number of active auctions closing in the next seven days","query_type":"analytics","confidence":0.97},"alternative_interpretations":[]}}\n'
    "\n"
    "── ANALYTICS (correlational — structural attribute) ──\n"
    'Q: "is there a relationship between domain authority and final sale price"\n'
    '→ {"decision":{"kind":"single","primary":{"raw_text":"is there a relationship between domain authority and final sale price","query_type":"analytics","confidence":0.96},"alternative_interpretations":[]}}\n'
    "\n"
    "── ANALYTICS (rankings / leaderboards) ──\n"
    'Q: "which extensions have the highest average number of bids"\n'
    '→ {"decision":{"kind":"single","primary":{"raw_text":"which extensions have the highest average number of bids","query_type":"analytics","confidence":0.95},"alternative_interpretations":[]}}\n'
    "\n"
    "── ANALYTICS (category vertical) ──\n"
    'Q: "has the volume of healthcare domain listings grown over the past quarter"\n'
    '→ {"decision":{"kind":"single","primary":{"raw_text":"has the volume of healthcare domain listings grown over the past quarter","query_type":"analytics","confidence":0.94},"alternative_interpretations":[]}}\n'
    "\n"
    "── MULTI-INTENT (hybrid + guidance) ──\n"
    'Q: "find me clean one-word .io domains under 1k, or also what tld should a new saas startup use"\n'
    '→ {"decision":{"kind":"multi","slices":[\n'
    '    {"raw_text":"find me clean one-word .io domains under 1k","query_type":"hybrid","confidence":0.95},\n'
    '    {"raw_text":"what tld should a new saas startup use","query_type":"guidance","confidence":0.93}\n'
    '  ]}}\n'
    "\n"
    "── SINGLE with ALTERNATIVES (borderline hybrid/guidance) ──\n"
    'Q: "solid seo domains worth a look right now"\n'
    '→ {"decision":{"kind":"single","primary":{"raw_text":"solid seo domains worth a look right now","query_type":"hybrid","confidence":0.78},"alternative_interpretations":[\n'
    '    {"raw_text":"solid seo domains worth a look right now","query_type":"explore","confidence":0.18}\n'
    '  ]}}\n'
)


USER_PROMPT_TEMPLATE = (
    "Allowed query_types: {allowed}\n"
    "Current UTC time: {current_utc_iso}\n"
    "Query: {query}\n"
    "\n"
    "Return the JSON object now."
)


def build_user_prompt(query: str, allowed_query_types: List[str], current_utc_iso: str) -> str:
    """Assemble the per-request user prompt.
    :param query: str - Normalized query text
    :param allowed_query_types: List[str] - Closed enum the model must use
    :param current_utc_iso: str - Current UTC timestamp in ISO 8601 format for relative date resolution; pass empty string when not available
    :return: str - User prompt body
    """
    allowed = ", ".join(sorted(allowed_query_types))
    return USER_PROMPT_TEMPLATE.format(
        allowed=allowed,
        current_utc_iso=current_utc_iso,
        query=query,
    )


def build_system_prompt(alt_band_high: float, keyword_expansion_max_terms: int) -> str:
    """Assemble the system prompt with config-driven substitution values.
    :param alt_band_high: float - Upper bound of the low-confidence band that triggers alternatives; must match ``qi.llm.alternative_band_high``
    :param keyword_expansion_max_terms: int - Max OR-terms extracted verbatim; must match ``qi.llm.keyword_expansion_max_terms``
    :return: str - System prompt body
    """
    return (
        SYSTEM_PROMPT
        .replace(_ALT_BAND_PLACEHOLDER, f"{float(alt_band_high):.2f}")
        .replace(_KEYWORD_CAP_PLACEHOLDER, str(int(keyword_expansion_max_terms)))
    )
