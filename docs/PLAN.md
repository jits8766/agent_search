# AUC Semantic Search Delivery Plan

## 1. Delivery Intent

Discovery confirmed the long-term direction: intent-aware auction search with hybrid retrieval, analytics, live signals, and graceful degradation.

Delivery sequencing changed after MLE/platform support discussion:

- ClickHouse onboarding is new for the team and not ready for Phase 1.
- Qdrant onboarding also carries setup and maintenance effort.
- Phase 1 therefore shifts to filters-only UX learning.
- Semantic retrieval and full search execution move to Phase 2.
- The prior Phase 2 scope moves to Phase 3.

## 2. Updated Phase Overview


| Phase   | Name                          | Outcome                                                                                                                                            |
| ------- | ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Phase 1 | Filters-Only / QI UX Learning | Return identified filters and intent from Query Intelligence only; no semantic retrieval, Qdrant dependency, or ClickHouse dependency.             |
| Phase 2 | Semantic Search Service       | Launch actual search service with semantic/hybrid retrieval. ClickHouse is optional; service can run degraded if ClickHouse onboarding is delayed. |
| Phase 3 | Continuity + Personalization  | Add saved search, memory, fine-tuning, live listing ingest, external ranking, and production model optimization.                                   |




## 3. Phase 1 Scope

Phase 1 is a low-dependency learning release.

Goal:

- Learn user friction in the existing UX.
- Identify which filters users expect from natural-language input.
- Validate grounded L0 filter extraction before onboarding heavier search infrastructure.
- Produce evidence for Phase 2/3 consumption; do not change the current FIND API outcome.

Delivery mechanism:

- Use `POST /search` with `qie_only_mode=true`.
- Path runs L0 only: LLM entity extractor + regex fallback.
- Skips L1 semantic router, L2 intent LLM, intent classification, and retrieval.
- Slim response fields: `request_id`, `identified_filters` (`[{name,value,source}]` FIND/soft names), `decision_tier` (`L0_entity`), `latency_ms`, `decision_cost_usd`. Success log: `qie_only_complete` (`PLAYBOOK.md` §3, `DESIGN.md` §8/§12).

Phase 1 output:

- Identified filters 
- Inventory grounding still runs server-side before `identified_filters` are emitted.
- No intent / query-type classification.
- No semantic results.
- No listing results.
- No change to current FIND API ranking/search behavior.
- No Qdrant requirement.
- No ClickHouse requirement.

Phase 1 out of scope:

- Replacing the current FIND API.
- Changing current backend search results.
- Serving semantic or hybrid search results.
- Launching Qdrant-backed retrieval.
- Launching ClickHouse-backed analytics, explore rails, guidance, or MVs.
- Claiming GCR/search-conversion impact from Phase 1 alone.

Phase 1 impact boundary:

- Phase 1 output is consumed for UX/product/MLE review.
- Phase 1 output is not consumed by the current FIND API runtime.
- Phase 1 learnings feed Phase 2 semantic search and Phase 3 continuity/personalization work.

Example API usage:

```bash
curl -X POST "$SEARCH_URL/search" \
  -F "query=cheap .io domains ending soon" \
  -F "qie_only_mode=true"
```



## 4. Phase 1 Timeline

Q2 delivery window: first week of June through `30 Jun` (`3 weeks`).  
Capacity: `1` L5 engineer.

Focus:

- Expose L0 filter extract + ground output through the existing `/search` API (`qie_only_mode=true`).
- Wire or demo the slim filters-only response path to UX.
- Review `identified_filters` against expected FIND slots.
- Capture user friction: missing filters, wrong filters, unclear wording.
- Build the evidence needed for Phase 2 search investment.

Exit by `30 Jun`:

- Filters-only API path is demoable.
- UX can consume identified filters or review the response contract.
- Phase 2 infrastructure asks are grounded in observed user friction and QI quality.
- Current FIND API behavior remains unchanged.



## 5. Phase 1 Dependencies


| Dependency                       | Needed For                                                                    | Status                                           |
| -------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------ |
| Existing filter API mapping      | Map QI output to UX/FIND slots                                                | Required                                         |
| `qie_only_mode=true` search call | L0 extract + ground only; slim filter JSON; no retrieval                      | Implemented                                      |
| Dev/runtime LLM credentials      | L0 LLM entity extraction where configured (regex fallback when LLM empty/off) | Required unless running degraded regex-only path |
| Frontend/UX support              | Validate filters in the existing flow                                         | Needed for user-facing learning                  |
| Qdrant                           | Semantic retrieval                                                            | Not needed in Phase 1                            |
| ClickHouse                       | Analytics/explore/guidance                                                    | Not needed in Phase 1                            |




## 6. Phase 1 Quality Gates


| Gate                    | Acceptance Criteria                                                                     |
| ----------------------- | --------------------------------------------------------------------------------------- |
| Filter extraction       | Expected filters appear in `identified_filters` for the reviewed query set.             |
| Grounding               | Invalid / unsupported inventory values are dropped by `EntityGrounder` before response. |
| Extract source          | Each identified entry carries `source` (`L0_llm` or `L0_regex`).                        |
| No intent leg           | Path does not run L1 or L2 intent classification (`decision_tier=L0_entity`).           |
| No retrieval dependency | Slim filter JSON returns with retrieval skipped.                                        |
| No FIND API impact      | Existing FIND API ranking/search behavior is unchanged.                                 |
| UX learning             | Feedback captures where users struggle with query wording and filters.                  |




## 7. Phase 1 Rollout Plan

1. Validate `qie_only_mode=true` API response in dev.
2. Build a curated query set from common buyer/search intents.
3. Review query output with UX/product/MLE.
4. Map applied filters to existing FIND slots.
5. Capture friction and gaps.
6. Decide Phase 2 search scope from observed misses and dependency readiness.



## 8. Phase 2 Scope

Phase 2 delivers actual semantic search.

Core scope:

- Qdrant-backed semantic/hybrid retrieval.
- Data-build/indexing path for priority auction inventory.
- Ranking, zero-result guard, cache, and retrieval metrics.
- Live mutable-field updates where event ingest is available.
- ClickHouse complements (analytics block, explore rail merge, guidance snapshot, optional `rank_leanings` reorder) when ClickHouse onboarding is ready — hybrid-first `ranked_results` always; sibling JSON blocks unchanged by leanings.

Degraded mode (toggle off, no deploy, or runtime unavailable):

- Search service launches and serves hybrid/semantic search without ClickHouse (`clickhouse.enabled=false` or CH down).
- Hybrid/semantic search remains available through Qdrant and structured payloads.
- Analytics SQL, explore rail merge, guidance snapshots, `rank_leanings`, and ClickHouse MVs skip when CH down — hybrid-first `ranked_results` still serve; not a hard outage for `/search`.
- Phase 2 still progresses from Phase 1 learning without waiting on ClickHouse infra.

Qdrant note:

- Qdrant onboarding is now Phase 2 because it carries setup and maintenance ownership.
- Phase 2 launch depends on accepting that operational ownership or agreeing on a managed/owned runtime path.



## 9. Phase 2 Timeline

Indicative post-Phase-1 sequence: July 2026.

## 10. Phase 2 Dependencies


| Dependency                         | Needed For                                                                                            | Delivery Handling                                                                                                                                                                                   |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Qdrant onboarding                  | Semantic/hybrid retrieval                                                                             | Required for actual semantic search launch.                                                                                                                                                         |
| Priority inventory source          | Seed/build search index                                                                               | **Defined:** live (`active_only`) + FIND types **16/20/38/39**; Athena `auction_audit_cln` → `/data-build/seed`; default lookback 14d / cap 1M (`base.yaml`). Bids/watches = freshness, not membership. See `DESIGN.md` §7.2 / `DISCOVERY.md`. |
| Event ingest source                | Mutable fields and engagement signals                                                                 | Used when available; static search can still run without live boosts.                                                                                                                               |
| Kinesis real-time stream ingestion | NRT updates into Qdrant payloads and ClickHouse event/analytics tables (bid/watch/enrich/delta paths) | **Reuse Auctions/FIND streams** (FIND already consumes events) — do not create parallel feeds. Phase 2: partner Auctions for access + contracts, then **Lambda consumer** → Qdrant/CH. Drivers config-gated / default off until wired; soft-fail when lagging; seed/index still serves. |
| ClickHouse onboarding              | Analytics, explore rails, guidance, MVs                                                               | Optional for Phase 2 launch; degraded mode available.                                                                                                                                               |
| MLE/platform support               | Store ownership, deployment, maintenance                                                              | Needed before productionizing Qdrant/ClickHouse paths.                                                                                                                                              |
| Phase 1 UX findings                | Search prioritization                                                                                 | Drives which filters/intents matter most.                                                                                                                                                           |




## 11. Phase 2 Quality Gates


| Gate                     | Acceptance Criteria                                                                                                               |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| Retrieval quality        | Results match intended filters and semantic concept for reviewed query set.                                                       |
| Qdrant readiness         | Index build, query path, health, and maintenance ownership accepted.                                                              |
| Degraded ClickHouse mode | Service starts and serves semantic search without ClickHouse.                                                                     |
| Fallback behavior        | When ClickHouse unavailable: skip complements and `rank_leanings`; hybrid `ranked_results` + nonempty ladder still fill listings. |
| Freshness                | Mutable fields update when event ingest is enabled.                                                                               |
| Observability            | Search, cache, data-build, and resilience status are visible.                                                                     |




## 12. Phase 3 Scope


| Capability                  | Delivery                                                                               |
| --------------------------- | -------------------------------------------------------------------------------------- |
| Real-Time Listing Ingest    | Stream new listings, embed them, and index them live.                                  |
| Saved Searches + Alerts     | Persist serialized intent and filters; support alerts based on repeat/adjust behavior. |
| Session Memory              | Maintain per-user search history with retention and delete controls.                   |
| External Ranker             | Add personalization/ranking through typed client and fallback path.                    |
| Small-Model Fine-Tuning     | Tune smaller model path when live volume and cost justify it.                          |
| Production LLM Accounts     | Add provider rotation, budget controls, and production credential management.          |
| Partner-Expiry Shopper Rail | Use portfolio expiration data and auction roles for opt-in partner-expiry discovery.   |




## 13. Phase 3 Dependencies


| Dependency                     | Needed For                            |
| ------------------------------ | ------------------------------------- |
| New-listing stream access      | Live listing ingest                   |
| Saved-search persistence owner | Saved searches and alerts             |
| User/session identity contract | Session memory and resume             |
| Production LLM accounts        | Provider rotation and spend control   |
| Live telemetry from Phase 2    | Fine-tuning and optimization triggers |




## 14. Cross-Phase Milestones


| Milestone                        | Phase     | Completion Signal                                                                            |
| -------------------------------- | --------- | -------------------------------------------------------------------------------------------- |
| Discovery complete               | Pre-phase | SQL-backed source map and discovery document approved.                                       |
| Filters-only UX learning         | Phase 1   | `qie_only_mode=true` output reviewed by UX/product/MLE; current FIND API behavior unchanged. |
| Qdrant search ready              | Phase 2   | Priority inventory indexed and searchable.                                                   |
| ClickHouse-degraded launch ready | Phase 2   | Semantic search works with ClickHouse unavailable.                                           |
| ClickHouse-enhanced search ready | Phase 2   | Analytics/explore/guidance enabled after onboarding.                                         |
| Continuity ready                 | Phase 3   | Saved searches and session resume work against current inventory.                            |
| Personalization ready            | Phase 3   | External ranker or memory-based ranking path is A/B-ready.                                   |




## 15. Metrics To Track

Phase 1:

- Filter extraction correctness (`identified_filters`).
- Unsupported / grounded-drop rate.
- L0 source mix (`L0_llm` / `L0_llm_cache` / `L0_regex` fallback).
- UX acceptance/rejection of suggested filters.
- Query wording friction themes.
- LLM spend: primary from `task_model_allowlists` ∩ GoCaas discovery (`DESIGN.md` §11). Watch `decision_cost_usd` / `model_id`.

Phase 2:

- Retrieval relevance.
- Zero-result rate.
- Query-to-click rate.
- Cache hit rate.
- Latency by route.
- Qdrant availability/query timeout rate.
- ClickHouse fallback rate.
- Event ingest lag when enabled.
- LLM spend: same allowlist chain for classify / extract / rewrite; refine from live `decision_cost_usd` (`DESIGN.md` §11).

Phase 3:

- Saved-search reuse.
- Alert usefulness.
- Session resume conversion.
- Ranker lift.
- LLM cost per correct query.



## 16. Risks And Controls


| Risk                                         | Control                                                                                                                                       |
| -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Phase 1 overbuilds infrastructure too early  | Keep Phase 1 filters-only through `qie_only_mode=true`; no Qdrant/ClickHouse dependency.                                                      |
| Phase 1 confused with search launch          | State explicitly that Phase 1 does not replace FIND API or change current search outcomes.                                                    |
| QI output does not map cleanly to UX filters | Use Phase 1 to identify mapping gaps before building full search.                                                                             |
| Qdrant onboarding takes longer               | Move actual semantic search to Phase 2 and make ownership explicit.                                                                           |
| ClickHouse onboarding slips                  | Launch Phase 2 without ClickHouse; hybrid-first ranks serve; analytics/rail-merge/guidance/`rank_leanings` complements stay off.              |
| MLE/platform support not ready               | Keep Phase 1 focused on UX learning and defer operational stores.                                                                             |
| Live ingest lag or failure                   | Event ingest updates Qdrant payloads and ClickHouse event tables when enabled; service still runs with static/indexed payloads when disabled. |
| Dependency outage blocks user                | Circuit breakers and per-route fallbacks preserve response path.                                                                              |




## 17. Delivery Dependencies Summary

Phase 1 required:

- Existing filter API mapping.
- `qie_only_mode=true` API usage.
- Dev/runtime LLM credentials or accepted rules-only degradation.
- UX/product review loop.
- Agreement that current FIND API behavior is unchanged in Phase 1.

Phase 2 required:

- Qdrant onboarding and ownership.
- Priority inventory indexing path.
- Event ingest access for live mutable fields where available.
- ClickHouse onboarding only for enhanced analytics/explore/guidance.
- Degraded-mode validation when ClickHouse is unavailable.

Phase 3 required:

- New-listing stream access.
- Saved-search/session persistence.
- User/session identity decisions.
- External ranker ownership decision.
- Production LLM provider accounts and cost controls.
- Live telemetry from Phase 2.

