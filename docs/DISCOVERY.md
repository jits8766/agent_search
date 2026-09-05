# AUC Semantic Search Discovery Document

## 1. Executive Summary

Auction discovery is the core problem. Users search with intent, but the current experience is filter-first and depends on snapshots, known filters, and existing recommendation paths.

Discovery conclusion:

- Keep the existing frontend/filter contract.
- Add intent-aware backend search.
- Combine filters, semantic meaning, analytics, and live engagement.
- Use real-time auction/bid/watch sources for freshness.
- When ClickHouse down: skip complements and `rank_leanings`; hybrid-first `ranked_results` + nonempty ladder still fill listings.



## 2. Problem


| Area       | Finding                                                                           |
| ---------- | --------------------------------------------------------------------------------- |
| User       | Natural-language intent is richer than filter-only search.                        |
| Inventory  | Price, bid count, watch activity, and end time change during auctions.            |
| Business   | Search/SERP discovery is a major lever for exposure, bids, sell-through, and GCR. |
| Operations | Daily snapshots are not valid online search truth.                                |


Target problem: help users find relevant expired-domain inventory without knowing exact filters, TLDs, auction types, or keywords upfront.

## 3. Current System Learnings

- Expired domains move through renewal, auction, closeout, and deletion over a roughly 72-day lifecycle.
- Priority auction types for delivery: `16`, `20`, `38`, `39`.
- Buyer signals include TLD, length, GoValue, SEO authority, traffic, bid count, bidder count, watches, and ending-time pressure.
- Existing systems include GoValue, recommendations, E-Ranker, Hidden Gems, Gen AI experiments, and fraud prevention.
- This service does not replace those systems; it creates a unified search backend that can consume or bypass their signals.



## 4. SQL Analysis Findings



### Real-Time Auction Feed

Source: `signals_platform_cln.auction_audit_cln`


| Finding                   | Value                                                       |
| ------------------------- | ----------------------------------------------------------- |
| p99 create-to-receive lag | ~`992 ms`                                                   |
| 24h volume                | ~`2.6M` rows                                                |
| throughput                | ~`30 rows/sec`                                              |
| caveat                    | `audit_utc_ts` is a source audit clock, not replication lag |


Conclusion: suitable for live auction state and delta updates: price, bid count, auction type, end time.

### Daily Snapshot Limit

Source: `theresaleplace.dam_auction_snap`


| Finding       | Value                           |
| ------------- | ------------------------------- |
| staleness     | ~`17-21 hours`                  |
| measured rows | `UPDATE` only in sampled window |
| online use    | not suitable                    |


Conclusion: keep for backfill/reference. Do not use as online search truth.

### Auction Coverage

Hot-feed coverage for target auction types:


| Type | Meaning                   | Coverage |
| ---- | ------------------------- | -------- |
| `16` | GoDaddy AutoExtend        | `99.5%`  |
| `20` | GoDaddy BuyNow / Closeout | `100.0%` |
| `38` | Partner AutoExtend        | `99.1%`  |
| `39` | Partner Closeout          | `99.95%` |


Conclusion: target inventory is hot-path eligible for later semantic-search phases.

### Bid and Watch Signals

Source: `the_resale_place.item_bids_cln`

- Bid stream showed sub-second p99 freshness in sample.
- Daily bid snapshots remain useful for history/training, not hot-path ranking.
- Watch events support watch-density and opportunity rails.

Conclusion: live bid/watch signals can support trending, ending-soon, competitiveness, and ranking boosts.

### Reference Tables


| Table                            | Use                         |
| -------------------------------- | --------------------------- |
| `trp_auction_type_roles_cln`     | auction-role vocabulary     |
| `trp_cancel_reason_cln`          | cancel-reason dictionary    |
| `trp_partner_expiry_shopper_cln` | registrar partner directory |


Conclusion: load as boot-time/daily lookup data, not real-time streams.

### Shopper / Saved State

- `domains.shopperdomainlist_snap` supports shopper-owned domain portfolio data.
- Daily freshness is acceptable for partner-expiry triggers measured in days.
- No verified auction watchlist table.
- No verified saved-search table in analyzed catalogs.

Conclusion: partner-expiry triggers are feasible; watchlist/saved-search re-execution is not data-backed from analyzed sources.

## 5. Discovery Problem Statement

Users ask for mixed intent:

- `cheap .io domains ending soon`
- `brandable cloud names with traffic`
- `show partner expiry auctions`
- `how many .com closeouts sold last week`

These combine filters, semantic concepts, urgency, analytics, and browse intent. The backend must classify intent first, then route to `hybrid`, `explore`, `guidance`, or `analytics`.

## 6. Solution Direction

Delivery shape:

- One backend search service.
- Same frontend/filter contract.
- Text input becomes grounded filters + residual semantic concept.

Runtime paths (hybrid-first `ranked_results` for every QI type):


| Path        | `ranked_results`                                       | Complement when ClickHouse up                                   |
| ----------- | ------------------------------------------------------ | --------------------------------------------------------------- |
| `hybrid`    | Qdrant vector/sparse + structured filters              | optional SQL price-band                                         |
| `explore`   | same hybrid retrieve; rails RRF-merged when CH up      | trending / ending-soon / latest rails                           |
| `guidance`  | same hybrid retrieve; optional `rank_leanings` reorder | market snapshot in `guidance` envelope (block unchanged)        |
| `analytics` | same hybrid retrieve; optional `rank_leanings` reorder | multi-period / NL-to-SQL in `analytics` block (block unchanged) |


Config: `general.search.ranked_results_complement` (incl. `rank_leanings`). CH down: skip complements / leanings; hybrid + nonempty ladder still fills listings.

Data paths:

- Athena/daily snapshots: seed and backfill.
- Auction/bid/watch streams: ClickHouse events and Qdrant payload updates.
- Mutable fields: payload patches, no re-embedding.
- Caches: absorb repeated exact and paraphrased queries.



## 7. Why This Solves It

- Hard constraints remain filters.
- Semantic text improves discovery beyond exact keywords.
- Live engagement makes ranking current.
- Explore rails reduce dead ends.
- Analytics answers questions without blocking listing discovery.
- Daily snapshots stay out of online search truth.



## 8. Delivery Outcome

Implemented capability:

- FastAPI search service with QI, orchestration, hybrid retrieval, analytics, feedback, health, measurement, and data-build APIs.
- Qdrant-backed hybrid search index.
- ClickHouse complements analytics/explore/guidance when enabled+reachable; `ranked_results` always hybrid-first on Qdrant; `rank_leanings` may reorder listings from analytics/guidance aggregates without changing those sibling blocks (CH off/down → no rail merge / no analytics SQL / no snapshot / no leanings; listings still return).
- Data-build and refresh flows.
- Guardrails, rate limits, security headers, circuit breakers, degradation, fallback.
- Katana ECS and local Docker Compose runtime.

Phase 1:

- Filters-only discovery via `POST /search` with `qie_only_mode=true`.
- Runs L0 LLM entity extract + regex fallback + inventory grounding only (no L1/L2 intent classification, no retrieval).
- Slim response: `answer_mode='qie_only'`, `identified_filters`, `decision_tier='L0_entity'`, `latency_ms`.
- No `applied_filters` / `grounding_applied` / `query_intelligence` / `pipeline_trace` / ranked results.
- No retrieval, ranking, semantic results, or current FIND API impact.
- Output is for UX/product/MLE friction learning.

Phase 2:

- Consume Phase 1 learnings for semantic/hybrid search and infrastructure onboarding.

Phase 3:

- Use search telemetry for saved search, memory, personalization, fine-tuning, and continuity.



## 9. Expected Outcomes

User outcome:

- Search by intent instead of manually assembling filters.
- Phase 1 shows understood filters only.
- Phase 2 search can return listings, explore alternatives, analytics answers, and faster cached repeats.

Business outcome:

- Better discovery is expected to improve relevant inventory exposure.
- More exposure is expected to improve clicks, bids, sell-through, and GCR.
- Live engagement can surface domains where buyer interest is forming.

Planning estimate from `plan_agentic_search.md`:

- Revenue lift: `$750K-$800K` incremental gross commerce.
- Capacity anchor: ~`2M` queries/month.
- Local cold-path run confirmed all four query types; analytics was the latency tail.



## 10. Boundaries

In scope:

- Phase 1 `qie_only_mode` L0 filters-only review.
- Backend search service for Phase 2/3.
- Existing frontend contract.
- Query intent, hybrid retrieval, explore/guidance, analytics, feedback, measurement, resilience.
- Auction types `16`, `20`, `38`, `39`.
- Verified real-time auction/bid/watch signals.

Out of scope:

- Phase 1 replacement of current FIND API.
- Phase 1 semantic search, Qdrant retrieval, ClickHouse analytics, or ranking impact.
- Phase 1 GCR/search-conversion impact claim.
- Frontend redesign.
- Kubernetes/Helm/Terraform deployment.
- Watchlist and saved-search re-execution from analyzed catalogs.
- Daily snapshots as online search truth.
- Production latency/cost targets before live validation.

