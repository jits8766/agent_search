# Search Output Contract — Full vs. `qie_only`

Contract for **UX** (what a card can render) and **Backend** (what data survives into response). Informative — tracks coverage, not implementation.

---

## 1. Two response shapes, one endpoint

`POST /search` returns one of two shapes depending on `qie_only_mode`:


| Mode                  | Retrieval?       | Output unit                  | Use case                                             |
| --------------------- | ---------------- | ---------------------------- | ---------------------------------------------------- |
| Full search (default) | Yes — hybrid RRF | Ranked item list             | Listing UI, cards                                    |
| `qie_only_mode=true`  | No               | Single filter-extract object | Query-understanding preview, chip debug, no listings |


---



## 2. SEMANTIC-SEARCH (Full search) — `ranked_results[i]` shape

Fixed envelope + configured payload projection.


| Group              | Fields                                                 | Source                                                |
| ------------------ | ------------------------------------------------------ | ----------------------------------------------------- |
| Envelope           | `rank`, `domain_name`, `coherence_score`, `matched_by` | Computed by ranking pipeline — no upstream equivalent |
| Payload projection | 68 fields, config-ordered                              | `general.search.result_fields` (`config/base.yaml`)   |



| Category                        | Count | Examples                                                                                                                                                                                   |
| ------------------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Upstream auction fields         | 53    | `fqdn`, `auction_price`, `auction_end_time`, `bids`, `reserved_price_amount`, `buy_it_now_flag`, `owner_member_id`, `data_source`, `is_gem`, `domain_create_date`                          |
| Service-derived                 | 12    | `tld`, `sld`, `time_to_end_hours/minutes/days`, `name_length`, `domain_age_years`, `traffic_proxy_score`, `has_web_traffic_signal`, `estimated_traffic_tier`, `isidn`, `highest_bidder_id` |
| Bid-derived (join at seed time) | 1     | `last_bid_offer_dtm`                                                                                                                                                                       |
| Majestic-derived (join at seed time) | 5 | `majestic_ext_back_links`, `majestic_ref_domains_fm`, `majestic_citation_flow_score`, `majestic_trust_flow_score`, `majestic_metric_exists` |
| Search-log-derived (join at seed time) | 1 | `unique_search_count` — `COUNT(DISTINCT customer_id)` over a 30-day window, `domain_search.domain_search_rollup` |


---



## 3. Baseline auction-record params vs. `ranked_results`

Baseline = every param the upstream auction record carries (73). `In ranked_results` = survives into response today.

**Core / identity**


| FIND-API-PARAM        | SEMANTIC-SEARCH | Criticality (UX) |
| --------------------- | --------------- | ---------------- |
| `active`              | Yes             | —                |
| `fqdn`                | Yes             | —                |
| `fqdn_from_feed`      | Yes             | —                |
| `auction_id`          | Yes             | —                |
| `auction_type`        | Yes             | —                |
| `auction_status`      | Yes             | —                |
| `auction_adult`       | Yes             | —                |
| `domain_extension_id` | Yes             | —                |
| `vendor_id`           | Yes             | —                |
| `owner_member_id`     | Yes             | —                |
| `item_description`    | Yes             | —                |
| `data_source`         | Yes             | —                |
| `domain_create_date`  | Yes             | —                |
| `is_reseller_owned`   | No              | Non-critical     |


**Timing**


| FIND-API-PARAM       | SEMANTIC-SEARCH | Criticality (UX) |
| -------------------- | --------------- | ---------------- |
| `auction_end_time`   | Yes             | —                |
| `end_time`           | Yes             | —                |
| `data_update_time`   | Yes             | —                |
| `last_bid_offer_dtm` | Yes             | —                |




**Pricing**


| FIND-API-PARAM                      | SEMANTIC-SEARCH | Criticality (UX) |
| ----------------------------------- | --------------- | ---------------- |
| `auction_price`                     | Yes             | —                |
| `auction_price_usd`                 | Yes             | —                |
| `auction_price_display`             | Yes             | —                |
| `auction_price_display_usd`         | Yes             | —                |
| `current_bid_price`                 | Yes             | —                |
| `current_bid_price_usd`             | Yes             | —                |
| `current_bid_price_display`         | Yes             | —                |
| `current_bid_price_display_usd`     | Yes             | —                |
| `start_bid_amount`                  | Yes             | —                |
| `start_bid_amount_usd`              | Yes             | —                |
| `start_bid_amount_display`          | Yes             | —                |
| `start_bid_amount_display_usd`      | Yes             | —                |
| `valuation_price`                   | Yes             | —                |
| `valuation_price_usd`               | Yes             | —                |
| `valuation_price_display`           | Yes             | —                |
| `valuation_price_display_usd`       | Yes             | —                |
| `reserved_price_flag`               | Yes             | —                |
| `reserved_price_amount`             | Yes             | —                |
| `reserved_price_amount_usd`         | Yes             | —                |
| `reserved_price_amount_display`     | Yes             | —                |
| `reserved_price_amount_display_usd` | Yes             | —                |
| `buy_it_now_flag`                   | Yes             | —                |
| `buy_it_now_amount`                 | Yes             | —                |
| `buy_it_now_amount_usd`             | Yes             | —                |
| `buy_it_now_amount_display`         | Yes             | —                |
| `buy_it_now_amount_display_usd`     | Yes             | —                |
| `appraised_value`                   | Yes             | —                |


**Bidding / listing flags**


| FIND-API-PARAM                         | SEMANTIC-SEARCH | Criticality (UX) |
| -------------------------------------- | --------------- | ---------------- |
| `bids`                                 | Yes             | —                |
| `bid_accepted_flag`                    | Yes             | —                |
| `monthly_traffic`                      | Yes             | —                |
| `is_website_included`                  | Yes             | —                |
| `feature_listing_flag`                 | Yes             | —                |
| `on_sale_percent`                      | Yes             | —                |
| `include_in_search_flag`               | Yes             | —                |
| `display_result_in_category_list_flag` | Yes             | —                |
| `sub_category_feature_listing_flag`    | Yes             | —                |
| `add_i_category_listing_flag`          | Yes             | —                |
| `is_gem`                               | Yes             | —                |
| `unique_search_count`                  | Yes             | —                |


**Majestic authority**


| FIND-API-PARAM                 | SEMANTIC-SEARCH                                        | Criticality (UX) |
| ------------------------------ | ------------------------------------------------------- | ---------------- |
| `majestic_ext_back_links`      | Yes                                                      | —                |
| `majestic_ref_domains`         | Partial — surfaced as `majestic_ref_domains_fm` (separate join-at-seed-time column, `vectorization/enrichment_source.py`; raw `majestic_ref_domains` is stored at `db_seed_source.py:616` but never projected). Value parity with the raw field is unverified. | Naming mismatch |
| `majestic_citation_flow_score` | Yes                                                      | —                |
| `majestic_trust_flow_score`    | Yes                                                      | —                |
| `majestic_metric_exists`       | Yes                                                      | —                |


**Enrichment block (**`enrichments.`***)**

Flattened at ingest — the nested FIND API `enrichments.*` block is not carried through as a sub-object; each field lands as a top-level key sourced directly from the `semrush_domain_enrichments` / `estibot_domain_enrichments` join tables (`vectorization/db_seed_source.py`). All 12 are present in `config/base.yaml` `result_fields` (lines 255-266).

| FIND-API-PARAM             | SEMANTIC-SEARCH | Criticality (UX) |
| -------------------------- | --------------- | ---------------- |
| `estibot_domain_count`     | Yes             | —                |
| `estibot_domain_count_dev` | Yes             | —                |
| `estibot_ext_count`        | Yes             | —                |
| `estibot_ext_count_dev`    | Yes             | —                |
| `semrush_ascore`           | Yes             | —                |
| `semrush_total`            | Yes             | —                |
| `semrush_domains_num`      | Yes             | —                |
| `semrush_urls_num`         | Yes             | —                |
| `semrush_keyword`          | Yes             | —                |
| `semrush_search_volume`    | Yes             | —                |
| `semrush_cpc`              | Yes             | —                |
| `semrush_refdomains`       | Yes             | —                |


**Counts**


|                                     | Count |
| ----------------------------------- | ----- |
| Baseline params (FIND API)          | 73    |
| Present in `ranked_results`         | 71    |
| **Gap (missing), total**            | **1** |
| — Admin / provenance                | 1     |
| **Naming mismatch (present under a different key)** | **1** |
| — `majestic_ref_domains` → `majestic_ref_domains_fm` | 1 |


---



## 4. `qie_only_mode=true` — response shape

No `ranked_results`, no item list. Same preprocess as full search
(`_preprocess_query` → `extract_filters_only`). Full response:

```
query, request_id, search_id, answer_mode="qie_only", latency_ms, decision_tier="L0_entity",
decision_cost_usd, identified_filters, keywords, grounded_drop_count, prompt_tag, schema_version,
find_query_params, find_query_string, soft_chips, find_skipped
[+ query_transform when rewrite accepted]
```

`request_id` = ephemeral trace id for this HTTP hop. `search_id` = durable search-interaction
id for feedback / analysis joins. Knobs: `identity.*` in `base.yaml`; ops: PLAYBOOK §13.

`identified_filters` = chip UX vocabulary (FIND API param names; may use labels /
relative times). Not card fields.

`find_query_params` / `find_query_string` = FIND `auction/recommend` wire view
(ISO times, numeric `typeIncludeList` IDs, stringified values, hard FIND-63 only).
FoS appends `find_query_string` to live FIND; FIND binary unchanged. See
[FIND_FOS_PHASE1.md](FIND_FOS_PHASE1.md).

`soft_chips` = soft/local chips excluded from FIND wire. `find_skipped` = hard
chips that failed normalize (with reason).

`keywords` = `[{term, probability}, …]` — topical terms from the same L0 call,
independent of `identified_filters`/soft signals (see `PLAYBOOK.md` §3). Kept when
`probability >= qi.l0_llm_entity.keyword_min_probability` (percent in YAML —
sole threshold for JSON, FIND wire `query` terms, and full-search soft rank-boost).
Regex path does not emit keyword probabilities (N/A). On `qie_only`, terms that
pass that threshold (plus `find_wire` term-count / separator policy) become FIND
`query` (and may set FIND `useSemanticSearch`). Full search carries the same data at
`query_intelligence.filters.keywords` and uses the same gate to rank-boost matching listings.

`query_transform` (optional) = `{mode, engine, transformed, transformed_query}`
when the shared preprocess accepts a rewrite. Absent on passthrough / extract-only
short queries. Filters/keywords for that response are grounded on
`transformed_query`.

UX: chips from `identified_filters` / `soft_chips`; listings from FIND using
`find_query_string` - not a results grid from this service.

---



## 5. Coverage summary


|                                                             | Full search                                                      | `qie_only`           |
| ----------------------------------------------------------- | ---------------------------------------------------------------- | -------------------- |
| Item-level card data                                        | Yes (87 projected fields, `config/base.yaml` `result_fields`)    | No                   |
| Authority / enrichment signals (Majestic: Yes, w/ 1 naming mismatch; Estibot/Semrush: Yes) | Yes                              | No                   |
| Filter-understanding metadata                               | `pipeline_trace.applied_filters` (name/value/api_param) + `pipeline_trace.applied_keywords` (term/roles[/probability]; encode + soft_boost; omits terms already in `filters.soft_signals`). Top-level `filters.identified` = cross-slice union; per-leg public identified under `sub_intent_filters[].identified` | Yes — primary output |
| Latency / cost accounting                                   | Yes                                                              | Yes                  |


