# AUC Semantic Search Process Playbook

## 1. End-To-End Query


| What                                  | How                                                   | Outcome                                                                                                         |
| ------------------------------------- | ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `premium .io under $100 with traffic` | guardrails -> QI -> hybrid-first retrieve -> response | `ranked_results` always; analytics/explore/guidance complement when CH up; `rank_leanings` may reorder listings |


```text
Input: premium .io under $100 with traffic
Compute (Phase 1 qie_only): L0 extract + ground → identified_filters
Compute (Phase 2 full search): intent + filters + residual → hybrid-first ranked_results (+ CH complements / optional rank_leanings)
```



## 2. Query Intelligence


| What               | How                   | Outcome                                                        |
| ------------------ | --------------------- | -------------------------------------------------------------- |
| `.io under $100`   | extract hard filters  | `tldIncludeList=io`, `maxPrice` (FIND names; exclusive ceiling) |
| `cloud hosting`    | keep as residual text | semantic meaning for retrieval                                 |
| `how many .com...` | classify as analytics | hybrid ranks + analytics block                                 |


```text
Input: premium .io domains under $100
Output: query_type=hybrid, filters=[tldIncludeList, maxPrice], residual=premium domains
```



## 3. QI-Only Mode


| What                 | How                                                                          | Outcome                                              |
| -------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------- |
| `qie_only_mode=true` | L0 LLM extract + regex fallback when LLM unavailable; skip L1/L2 + retrieve | slim JSON + success log `qie_only_complete`          |


```text
Input: cheap .io under 100
Response:
  request_id=req_…
  answer_mode=qie_only
  decision_tier=L0_entity
  latency_ms=…
  decision_cost_usd=…
  prompt_tag=qi.entity.l0_filter.v2
  schema_version=1
  identified_filters=[
    {name: "tldIncludeList", value: "io", source: "L0_llm"},
    {name: "maxPrice", value: 99, source: "L0_llm"}
  ]
  keywords=[
    {term: "coffee", probability: 0.95},
    {term: "pizza", probability: 0.93}
  ]

Log (INFO, semantic_search.app):
  qie_only_complete request_id=… query_len=… latency_ms=… decision_tier=L0_entity
  filter_count=… hard_filter_count=… source=L0_llm|L0_llm_cache|L0_regex model_id=… token_count=…
  decision_cost_usd=… grounded_drop_count=0 prompt_tag=… schema_version=…
```

**`identified_filters` schema** (array of objects):

| Field | Type | Notes |
| ----- | ---- | ----- |
| `name` | string | FIND API param (e.g. `tldIncludeList`, `maxPrice`) or soft/local chip (e.g. `keyword_contains`, `topic_include`) |
| `value` | string \| number \| bool \| list | Slot transform applied (e.g. exclusive `under $100` → `maxPrice=99`). Multi-value soft slots may be pipe-joined strings. |
| `source` | string | `L0_llm` \| `L0_llm_cache` \| `L0_regex` |

- Catalog: FIND-63 from `find_api_params.json` ∩ slot map (`slot_to_api_param.py`); plus soft/local in `FILTERABLE_PARAMS`. Internal slots (`tld`, `price_max`) are not public names.
- Body + log: `request_id`, `prompt_tag`, `schema_version`. L0 cache key = normalize + those versions (LLM only; regex not cached).
- Filters-only; no FIND ranking/result impact.

**`keywords`** — `{term, probability}` pairs extracted by the same L0 call, independent of `identified_filters` (excludes numerics, domain/TLD terms, stop words, generic intent verbs). Never merged into `identified_filters` or soft signals. Same shape appears on full search at `query_intelligence.filters.keywords`, where the terms also feed `SoftKeywordApplier`'s rank boost (`entity_slots.soft_rank_boost_weight`, no probability floor).



## 4. Domain Vectorization And Indexing


| What             | How                                     | Outcome            |
| ---------------- | --------------------------------------- | ------------------ |
| listing rows     | build dense text, sparse terms, payload | Qdrant points      |
| analytics rows   | mirror when enabled                     | ClickHouse data    |
| bid/watch events | ingest live activity                    | engagement signals |


```text
Listing: cloudhost.io, $75, traffic=1200
Compute: dense text + sparse tokens + payload
Output: searchable Qdrant point
```

Delta refresh patches payload only. Full upsert/re-embedding stays in indexing.

## 5. Hybrid Retrieval


| What                | How                               | Outcome            |
| ------------------- | --------------------------------- | ------------------ |
| `QueryIntent`       | encode residual and apply filters | candidate domains  |
| dense + sparse legs | fuse ranks                        | stable result pool |


```text
Intent: .io, price<=100, residual=premium traffic
Search: Qdrant filter + semantic match
Output: candidates for ranking
```



## 6. Fallback Paths


| What                       | How                                | Outcome           |
| -------------------------- | ---------------------------------- | ----------------- |
| strict filters return zero | widen/relax lower-priority filters | broader matches   |
| still zero                 | semantic-only then explore ladder  | nonempty listings |


```text
Query: premium .io traffic>50000 under $50
First: 0 results
Fallback: relax price → semantic-only → explore ladder (config: ranked_results_complement)
```



## 7. Mutable Field Refresh


| What                       | How                  | Outcome                 |
| -------------------------- | -------------------- | ----------------------- |
| price/bids/end time change | patch Qdrant payload | current filters/ranking |


```text
Before: cloudhost.io price=$50
After: price=$75
DeltaRefreshDriver: set_payload(price=$75)
```

No vector rebuild. No point recreation.

## 8. Event Ingest And Live Signals


| Signal             | How                                 | Outcome        |
| ------------------ | ----------------------------------- | -------------- |
| `bid_velocity_1h`  | recent bid count                    | trending boost |
| `watch_density_1d` | recent watch count                  | interest boost |
| `engagement_ratio` | bidder interest vs watcher interest | demand signal  |


```text
Domain A: better semantic match, no activity
Domain B: slightly weaker match, many bids/watches
Output: Domain B can move up
```



## 9. Ranking And Reranking


| What                               | How                                                             | Outcome                                                                        |
| ---------------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| fused candidates                   | keep retrieval baseline                                         | relevance order                                                                |
| soft keywords / chips              | SLD boost; 2+ keywords → all-terms first, then fair mix by prob | multi-topic queries keep both themes in `top_k` (not one-term monopoly)      |
| engagement fields                  | boost active listings                                           | fresher ordering                                                               |
| brandable query                    | score short/clean names                                         | brandable lift                                                                 |
| CH analytics / guidance aggregates | bounded `rank_leanings` on coherence (before auction tie-break) | near-tied TLD order reflects market/analytics signal; sibling blocks unchanged |
| auction urgency / bids / value     | auction tie-break last                                          | listing urgency lift                                                           |


```text
Before: cloud.com, hosting.io, cloudhost.net
After: cloud.com, cloudhost.net, hosting.io
```

Config: `general.search.ranked_results_complement.rank_leanings`. Missing/malformed CH payload or disabled config → identity order. If optional reranker is unavailable, current order stays.

## 10. Analytics Query


| What                               | How                                                   | Outcome                              |
| ---------------------------------- | ----------------------------------------------------- | ------------------------------------ |
| count/aggregate question           | cache -> shortcut -> SQL                              | numeric answer                       |
| repeated question                  | exact cache hit                                       | faster answer                        |
| ClickHouse off / down / undeployed | skip analytics SQL / rail merge / snapshot / leanings | hybrid `ranked_results` keep running |


```text
Ask: how many .com domains have traffic > 1000?
Compute: validate/run SQL, cache result
Output: count answer
```



## 11. Response Composition


| Part                     | Meaning                                                                                                                          |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| `query`                  | original text                                                                                                                    |
| `query_intelligence`     | full-search only: intent, filters, confidence                                                                                    |
| `ranked_results`         | listings (full search); optional `rank_leanings` reorder when CH analytics/guidance present; absent on `qie_only_mode` slim path |
| `analytics` / `guidance` | sibling CH blocks when present; not mutated by `rank_leanings`                                                                   |
| `pipeline_trace`         | `applied_filters` + `applied_keywords` (encode / soft_boost) + `ranked_results_role`; optional `stages` when attribution on   |
| `query_intelligence.filters` | full search: top-level `identified` = union across multi-intent legs; per-leg under `sub_intent_filters[].identified`      |
| `identified_filters`     | `qie_only_mode` slim path: grounded extracted entities                                                                           |


```text
Full search: query_intelligence + ranked_results + pipeline_trace(+stages)
qie_only:    identified_filters + decision_tier=L0_entity + prompt_tag + schema_version
```



## 12. Measurement And Resilience


| What                   | How                                                           |
| ---------------------- | ------------------------------------------------------------- |
| latency/cache/fallback | recorded per request                                          |
| dependency health      | controls degraded routing                                     |
| LLM spend              | Caps `$0.05`/q, `$25/h`, `$100/day` → regex/L1 on breach. Primary = `task_model_allowlists` ∩ discovery (`DESIGN.md` §11). Watch `model_id` / `decision_cost_usd`. |


```text
Vector store down
Record: degraded dependency + fallback path
Output: degraded response instead of total failure

LLM budget exhausted (query or fleet hour/day)
Record: llm_cost_budget_* + decision_cost_usd
Output: regex L0 + L1 search continues (analytics: cost_budget_exceeded)
```

### Alerts

[`OBSERVABILITY.md`](OBSERVABILITY.md) **§ Boundaries** — code emits; this doc = names/filters only; platform wires CW. **No in-repo CW alarms.** Caps/thresholds: `base.yaml`.

| Alarm | Fire |
| ----- | ---- |
| liveness | `/healthz` unhealthy ≥ 2m |
| latency p99 / p50 | > 2500 ms / 5m · > 300 ms / 15m |
| http-5xx | > 1% / 5m |
| store-health | unhealthy > 5m |
| llm query / fleet budget | `$0.05`/q · `$25/h` · `$100/day` |
| zero-result | rate > 3% |



## 13. Feedback Signals And Optional Retraining


| What | How | Outcome |
| ---- | --- | ------- |
| UAT write (Auction / UX) | `POST /feedback` Form: `comment`, optional `query`, preferred `search_id`, optional `request_id` | JSONL + ring + best-effort S3 CSV |
| Auth | Header `X-Feedback-Key` when `feedback.uat_api_key_env_var` is set in env | 401 if missing/wrong |
| Correlate search | Pass durable `search_id` from `POST /search` | Business join for feedback/analysis; `request_id` is trace-only |
| Identity config | `identity.*` in `base.yaml` (prefixes, headers, `feedback_search_id_mode`) | All mint/accept behavior config-driven |
| UAT export (operator) | `GET /feedback?date_from=&date_to=` (`YYYY-MM-DD`; default last 7 days) | CSV download from S3; **503** if S3 unset |


```text
POST /search → request_id=req_… search_id=search_…
POST /feedback -F comment=… -F search_id=search_… [-F request_id=req_…] [-H X-Feedback-Key:…]
→ {signal_id, search_id, request_id, status, recorded_at}

GET /feedback?date_from=YYYY-MM-DD&date_to=YYYY-MM-DD
→ attachment feedback_{from}_{to}.csv
  columns: signal_id, search_id, request_id, signal_type, signal_origin, created_at_iso, comment, query
```

**Audience:** POST = Auction/UX (or UAT clients). GET = operator offline export — not the Auction runtime path.

**Identity:** `request_id` = ephemeral per-HTTP-hop trace id. `search_id` = durable search-interaction id for feedback / analysis joins. Missing `search_id` on feedback follows `identity.feedback_search_id_mode` (`soft_generate` or `required`). Client-supplied ids must match `identity.id_value_pattern`.

**Learning loop:** UX posts free-text feedback with `search_id` from `/search`. Offline review via GET CSV / logs as backup.

**Retention (`feedback.*` in `base.yaml`):**

| Layer | Policy | Knob / path |
| ----- | ------ | ----------- |
| In-memory ring | Hot buffer; FIFO at **5000** (`deque`); no TTL | `max_in_memory_signals` |
| Local JSONL | Append-only; no app rotation/TTL (ops/volume manage disk) | `signal_log_path` |
| S3 CSV (UAT) | Day keys `feedback/YYYY/MM/DD/{signal_id}.csv`; durable for `uat_feedback` | `S3_PRETRAINED_DIR` parent bucket |
| Allowlist | Only listed types recorded | `allowed_signal_types` |
| UAT limits | Comment cap; optional `X-Feedback-Key` | `uat_max_comment_chars`, `uat_api_key_env_var` |

Ring / JSONL = process-local hot path. GET reads **S3**, not the ring. Retrain should prefer durable S3 (UAT today).

Internal click/resume signals stay config-driven measurement; optional retrain does not change calibration.

## 14. Background Jobs And Shutdown


| What         | How                             |
| ------------ | ------------------------------- |
| startup      | start prewarm tasks and drivers |
| steady state | poll deltas/events/health       |
| shutdown     | stop tasks and drivers (no signal flush; see `PROCESS.md` §15) |


```text
Startup: drivers begin polling
Shutdown: drivers/LLM stop; ring discarded; JSONL already on disk if written; UAT S3 may still be in flight
```



## 15. Storage Summary


| Store                | Role                                |
| -------------------- | ----------------------------------- |
| Qdrant               | vectors + payload filters           |
| ClickHouse           | analytics/events/views when enabled |
| in-process caches    | repeated QI/analytics speedup       |
| JSONL / ring / S3    | signals; UAT durable on S3 (`PLAYBOOK.md` §13) |
| Athena/source reads  | seed/backfill source                |


```text
ranked_results: always hybrid-first on Qdrant (works without ClickHouse)
Analytics block / explore rail merge / guidance envelope: ClickHouse when enabled+reachable; else skip
rank_leanings: reorder ranked_results from analytics/guidance aggregates when those substrates succeed; blocks unchanged
qie_only_mode: L0 extract + ground only — no Qdrant / ClickHouse required
```

