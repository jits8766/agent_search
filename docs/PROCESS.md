# AUC Semantic Search Process Flow

## 1. End-To-End Flow

Both legs share one pre-QI pipeline (`SearchOrchestrator._preprocess_query`):
normalize → optional spell → token-gated combined rewrite+L0 extract (or extract-only).
No separate transform/extract stack for `qie_only` vs full search.

```mermaid
flowchart LR
    Query[Raw user query] --> Guard[Guardrails<br/>sanitize rate limit validate]
    Guard --> Pre[Shared preprocess<br/>spell + token-gated<br/>rewrite+L0 or extract-only]
    Pre --> Mode{qie_only_mode?}

    Mode -->|true| L0Only[extract_filters_only<br/>reuse pre_l0 / L0 ground<br/>no L1/L2 / no retrieve]
    L0Only --> Slim[Slim qie_only JSON<br/>identified_filters + keywords<br/>+ query_transform when rewritten]

    Mode -->|false| QI[Full QI on effective text<br/>split → L1 → L2 / ensemble]
    QI --> Encode[Encode/retrieve on effective<br/>transformed text only when rewritten]
    Encode --> HybridFirst[Hybrid-first retrieve<br/>Qdrant + filters + keyword boost]
    HybridFirst --> Complement{ClickHouse up?}
    Complement -->|yes| Merge[Optional rails RRF merge<br/>analytics block / guidance envelope]
    Complement -->|no| Strip[Temporal strip + hybrid only<br/>nonempty ladder]
    Merge --> Ranking[Rank + rerank + diversify]
    Strip --> Ranking
    Ranking --> Response[Response envelope<br/>ranked_results always hybrid-based]
    Slim --> Measure[Measurement + feedback + resilience]
    Response --> Measure
```




| Path                                                      | Purpose                                                               | Result                                                                                                                             |
| --------------------------------------------------------- | --------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `qie_only_mode=true` (Phase 1)                            | Same preprocess as full search; L0 ground only (no intent / retrieve) | Slim `identified_filters` + `keywords` + optional `query_transform`                                                                |
| `hybrid` / `explore` / `guidance` / `analytics` (Phase 2) | Same preprocess → full QI → hybrid-first retrieve on effective text   | Listings always; plus `analytics` / `guidance` / rail merge when CH up; `rank_leanings` may reorder listings from those aggregates |
| saved searches (Phase 3)                                  | persist and replay intent                                             | alerts / resumed search                                                                                                            |
| SLM path (Phase 3)                                        | lower-cost tuned model path                                           | optimized QI / LLM calls                                                                                                           |




## 2. Query Intelligence (Intent Engine)

```mermaid
flowchart TD
    A[Guarded query] --> B[Normalize + optional spell]
    B --> Gate{token_count > rewrite_threshold?<br/>and combine_rewrite_with_l0_extract}
    Gate -->|yes| Comb[1x LLM: rewrite + L0 filters/keywords<br/>grounded on rewritten only]
    Gate -->|no| Ext[1x LLM: L0 extract-only on normalized]
    Comb -->|LLM ok + rewrite accepted| Eff[effective = rewritten]
    Comb -->|LLM ok rewrite rejected| Eff2[effective = normalized<br/>optional reextract]
    Comb -->|LLM fail| Raw[effective = normalized<br/>regex on raw/normalized]
    Ext -->|LLM ok| Eff3[effective = normalized]
    Ext -->|LLM fail| Raw
    Eff --> Split[multi-intent split on effective]
    Eff2 --> Split
    Eff3 --> Split
    Raw --> Split
    Split --> Cls[L0 entities ready then L1<br/>then conditional L2 / ensemble]
    Cls --> Q[Final QueryIntent]
```



**Happy path = one structured LLM call for rewrite/extract** (`qi.query_transformer.combine_rewrite_with_l0_extract`). Token gate picks prompt/schema:


| Token count           | LLM call                 | Filters / keywords grounded on  |
| --------------------- | ------------------------ | ------------------------------- |
| ≤ `rewrite_threshold` | Extract-only L0          | Normalized / passthrough        |
| > `rewrite_threshold` | Combined rewrite+extract | Rewritten query (same response) |


Shared contract for **both** `qie_only` and full search:

- Entry: `SearchOrchestrator._preprocess_query` (spell + token gate). Combined rewrite skips standalone `QueryTransformer.transform`.
- When rewrite accepted: `effective` = rewritten; `query_transform` attached (`mode`, `engine`, `transformed=true`, `transformed_query`). Downstream QI / multi-intent split / L0 reuse / ANN encode / retrieve use **effective only** — never raw when transformed (`classify_on_transformed_query`, `encode_from_rewrite`).
- L0 extract quality gates (prompt contract, same path for both legs):


| Gate   | Requirement                                                                                                                                                                          |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **G1** | FIND-63 hard filters — emit every cued hard param from the catalog; never invent off-catalog params                                                                                  |
| **G2** | Soft/local chips — topics, patterns, phrases, categories (`topic_`*, `keyword_*`, lifecycle, …) when cued; keep soft out of FIND hard wire                                           |
| **G3** | Keywords — topical niche terms with `probability` in [0,1]; keep when `>= qi.l0_llm_entity.keyword_min_probability` (percent YAML; sole source for JSON, FIND wire, soft rank-boost) |
| **G4** | Price+currency pairing — numeric price bound **and** `filterPriceCurrency` together (e.g. `under EUR 80` → `maxPrice=79` + `filterPriceCurrency=EUR`)                                |


- Happy path = **one** structured LLM call (token gate picks extract-only vs combined). Multi-intent keyword legs fan out with `asyncio.gather`; no second extract LLM per leg when `pre_l0` is present.
- Regex path has **no** keyword probabilities (N/A — not a silent 50%).
- Full search: keyword terms (same G3 gate) feed soft rank-boost and FIND wire `query`. Multi-intent split runs on effective (transformed) text for multi-leg SERP.
- Multi-intent filters in the response: top-level `filters.identified` = **union** of hard params across legs (dedupe by name+value). Per-leg identified lives only under `sub_intent_filters[].identified` — not duplicated at the top.
- `extract_before_classify`: L0 finishes before L1/L2. LLM unavailable → regex on raw/normalized; no rewrite from that failed path. Combined tags: `qi.l0_llm_entity.combined_prompt_tag` / `combined_schema_version`.

Note: semantic cache miss ≠ "semantic search = no." It only means no cached paraphrase intent; QI continues.

## 3. QI-Only Mode (Phase 1: Filters Identification)

Same preprocess as full search. API must not run a second transform/extract stack —
call `SearchOrchestrator.extract_filters_only()` (which calls `_preprocess_query` then
`QIEngine.extract_l0_filters`, reusing `pre_l0` when the combined call already ran).

```mermaid
sequenceDiagram
    participant Client
    participant API as POST /search
    participant Orch as SearchOrchestrator
    participant QI as QIEngine
    participant Resp as Response

    Client->>API: query + qie_only_mode=true
    API->>API: sanitize input
    API->>Orch: extract_filters_only()
    Note over Orch: SAME _preprocess_query as full search<br/>spell + token-gated combined or extract-only
    Orch->>QI: extract_l0_filters (reuse pre_l0 when combined already ran)
    Note over QI: L0 LLM or precomputed combined result<br/>+ regex fallback when LLM unavailable<br/>merge/deconflict + EntityGrounder<br/>NO L1 semantic router, NO L2 intent LLM
    QI-->>Orch: QueryIntent decision_tier=L0_entity<br/>+ query_transform when rewritten
    Orch-->>API: grounded intent
    API->>API: shape slim JSON + FIND wire + optional L0 cache
    API->>API: log qie_only_complete
    API->>Resp: answer_mode=qie_only
    Resp-->>Client: request_id + identified_filters + keywords<br/>optional query_transform<br/>decision_tier + latency_ms + decision_cost_usd<br/>prompt_tag + schema_version<br/>find_query_params / find_query_string
```



L0 only (regex when LLM unavailable). Skips L1/L2 and retrieval. Token gate +
`combine_rewrite_with_l0_extract` identical to §2. Cache keys use
`combined_prompt_tag` / `combined_schema_version` when rewrite accepted; else
extract-only tags. LLM cache key = `normalize_query` + L0 versions (regex not
cached). Success → `qie_only_complete` (`PLAYBOOK.md` §3, `DESIGN.md` §12).
FIND binary ranking unchanged; FoS consumes FIND wire fields from this response.

## 4. Domain Vectorization And Indexing (Data ingestion)

```mermaid
flowchart TD
    Manual[data-build API<br/>seed then analytics-backfill] --> Seed[Seed listing records]
    GHA[GitHub Actions data-ingest.yml<br/>seed → optional CH backfill] --> Seed

    Seed --> Segment[Domain segmentation]
    Segment --> Syn[Static synonym expansion]
    Segment --> DenseText[Dense text construction]
    Syn --> Sparse[Sparse vector encode]
    DenseText --> Dense[Dense batch encode]
    Sparse --> Point[Qdrant point]
    Dense --> Point
    Point --> Upsert[Qdrant upsert]
    Seed --> CHCheck{ClickHouse executor?}
    CHCheck -->|yes| CHMirror[Mirror seed rows to ClickHouse]
    CHCheck -->|no| CHSkip[Skip analytics mirror]

    Lambda[Lambda real-time ingest<br/>auction/bid/watch event trigger] --> Delta[DeltaRefresh-style update<br/>auction mutable fields]
    Lambda --> EventIngest[EventIngest-style update<br/>bid/watch engagement events]
    Delta --> Patch[Patch existing Qdrant payloads<br/>set_payload only, no point upsert, no re-embedding]
    EventIngest --> CHEvents[Write ClickHouse bid/watch event rows<br/>when ClickHouse enabled]
    EventIngest --> Enrich[Enrich Qdrant payloads<br/>from ClickHouse live-signal views]
    CHEvents --> Views[ClickHouse materialized views<br/>auto-refresh derived rails/signals]
```



Schedule fields live in `vectorization.seed.schedule` (`base.yaml`). The ingest
workflow reads effective values from `GET /data-build/status` after `/healthz`.


| Env var                         | Effect                                |
| ------------------------------- | ------------------------------------- |
| `SEED_SCHEDULE_RUN_ON_DEPLOY`   | allow `workflow_call` trigger=deploy  |
| `SEED_SCHEDULE_INTERVAL_HOURS`  | minimum hours between scheduled fires |
| `SEED_SCHEDULE_RUN_AT_HOUR_UTC` | UTC hour for scheduled fire           |


Set with `gdx katana envvar set` + `env sync`, or local `.env`, then restart the
task. With `in_process: false`, only the GitHub Actions workflow runs scheduled
builds. See `.github/workflows/data-ingest.yml`, `scripts/read_seed_schedule.py`.

**CI vs ingest (decoupled):** `ci-deploy.yml` ends after API / Qdrant / optional
ClickHouse **promote**. Seed is **not** a CI job — deploy stays green when
packaging succeeds; index freshness comes from the daily cron or manual dispatch.

**Ingest triggers** (`.github/workflows/data-ingest.yml`):


| Trigger         | Cron / event                 | Behavior                                                     |
| --------------- | ---------------------------- | ------------------------------------------------------------ |
| Daily freshness | `0 2 * * *` (`trigger=cron`) | hour/interval gates vs `last_build`; phase=all               |
| Manual          | `workflow_dispatch`          | optional `force_run`                                         |


Order inside ingest: Qdrant `POST /data-build/seed` first; ClickHouse `POST /data-build/analytics-backfill` only if lever on and Qdrant OK.
`seed_mode: rebuild` deletes collection `auctions_listings` before reload (`_clear_for_rebuild`). Prefer EFS at `/qdrant/storage` on Katana Qdrant (see `configs/katana-qdrant.yaml`) so promote does not wipe the index.


| Ingestion path          | Purpose                                                                                                              |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------- |
| data-build API          | `POST /data-build/seed` (Qdrant); `POST /data-build/analytics-backfill` (CH)                                         |
| GitHub Actions ingest   | separate from CI: daily cron + manual; seed then CH if enabled. Ingest failure → **ingest** workflow RED (not CI)    |
| Lambda real-time ingest | delta payload patches and bid/watch event ingest                                                                     |




## 5. Hybrid Retrieval

```mermaid
sequenceDiagram
    participant Orch as SearchOrchestrator
    participant Cache
    participant Ret as QdrantHybridRetriever
    participant Qdrant
    participant Fuse as RRF fusion

    Orch->>Cache: exact/structured lookup
    alt cache hit
        Cache-->>Orch: cached payload
    else cache miss
        Orch->>Ret: retrieve(QueryIntent)
        Ret->>Ret: encode dense/sparse/ngram legs
        Ret->>Ret: build payload filter
        Ret->>Qdrant: query_points()
        Qdrant-->>Ret: candidate points
        Ret-->>Orch: CandidateSet
        Orch->>Fuse: fuse candidate sets
    end
```





## 6. Fallback Paths

```mermaid
flowchart TD
    A[Search path] --> B{results found?}
    B -->|yes| C[Rank results]
    B -->|no| D[ZeroResultGuard]
    D --> E[Widen numeric filters]
    E --> F{results?}
    F -->|yes| C
    F -->|no| G[Relax lower-priority filters]
    G --> H{results?}
    H -->|yes| C
    H -->|no| I[Semantic-only retry]
    I --> J{results?}
    J -->|yes| C
    J -->|no| K[Explore fallback rails]
```





## 7. Mutable Field Refresh

```mermaid
flowchart TD
    A[DeltaRefreshDriver tick] --> B{lock available?}
    B -->|no| C[skip overlap]
    B -->|yes| D[Fetch changed auction rows]
    D --> E[Chunked window walk]
    E --> F[Patch existing Qdrant payloads<br/>set_payload by item_id filter]
    F --> H[Advance cursor]
    H --> I[Sleep until next interval]
```



Refreshed mutable fields: `price`, `bid_count`, `auction_type`, `ends_at`, plus configured mutable payload fields. Vectors are not re-embedded. This driver patches existing Qdrant points; full point creation/upsert stays in the data-build/indexing path.

## 8. Event Ingest And Live Signals

```mermaid
flowchart LR
    Bid[Bid event source] --> Ingest[EventIngestDriver]
    Watch[Watch event source] --> Ingest
    Ingest --> CHEvents[(ClickHouse bid/watch events)]
    CHEvents --> MVs[Materialized views<br/>velocity density competitiveness]
    MVs --> Payload[Qdrant payload enrichment]
    Payload --> Ranking[Engagement-aware ranking]
```




| Signal                    | Meaning                               | Used By                          |
| ------------------------- | ------------------------------------- | -------------------------------- |
| `bid_velocity_1h`         | bids in recent 1h window              | trending, ranking boost          |
| `watch_density_1d`        | active watches today                  | opportunity rails, ranking boost |
| `bidder_watch_density_1d` | bidder-intent watches today           | ranking boost                    |
| `engagement_ratio`        | bidder-watch density vs watch density | ranking boost                    |




## 9. Ranking And Reranking

```mermaid
flowchart TD
    A[Fused candidates] --> B[Keep fusion order baseline]
    B --> KW{soft keywords / chips?}
    KW -->|yes| KW2[Keyword + soft-chip boost<br/>2+ keywords: combo first<br/>then fair mix by keyword prob]
    KW -->|no| C
    KW2 --> C{engagement fields present?}
    C -->|yes| D["Apply engagement boost (bid & watch densities)"]
    C -->|no| E[Skip engagement boost]
    D --> F{brandable query?}
    E --> F
    F -->|yes| G[Apply brandability score]
    F -->|no| H[Skip brandability]
    G --> Lean{analytics/guidance substrate + rank_leanings?}
    H --> Lean
    Lean -->|yes| L1[Bounded complement leanings<br/>coherence + max_bonus * signals]
    Lean -->|no| L2[Skip leanings]
    L1 --> I{external eRanker enabled?}
    L2 --> I
    I -->|yes + healthy| J[External rerank]
    I -->|no/slow| K[Keep current order]
    J --> Tie[Auction tie-break]
    K --> Tie
    Tie --> L[RankedResults]
```



**Soft keywords (full search, G3):** after retrieve/truncate, matching SLD names get a soft boost (not hard filters). With **two or more** gated keywords, order prefers domains that contain **all** terms, then **fairly mixes** single-term hits so the highest-probability keyword cannot fill the whole `top_k` (e.g. coffee + pizza both surface). Soft chips (`topic_`*, `keyword_*`, …) still boost the same stage. Trace: `pipeline_trace.applied_keywords` (roles `encode` / `soft_boost`; omits terms already listed as soft signals).

Post-slim order: soft keyword/chip boost → complement `rank_leanings` (when CH analytics/guidance payload present) → auction tie-break. 

## 10. Analytics Query

```mermaid
flowchart TD
    A[Analytics intent] --> B{exact NL-to-SQL cache hit?}
    B -->|yes| R[Return cached answer]
    B -->|no| C{known aggregate shortcut?}
    C -->|yes| D[Run ClickHouse template periods]
    C -->|no| E[Generate SQL]
    E --> F[Validate SQL]
    F --> G[Execute ClickHouse query]
    D --> H{result available?}
    G --> H
    H -->|yes| I[Cache eligible result]
    H -->|no/timeout| J[Typed analytics failure<br/>ranked_results stay hybrid]
    I --> R
```



If ClickHouse is toggled off, not deployed, or unavailable at runtime: skip analytics SQL / explore rail merge / guidance snapshot / `rank_leanings`. `ranked_results` still come from hybrid-first Qdrant retrieve (`ranked_results_complement`, temporal strip + nonempty ladder). 

## 11. Response Composition

```mermaid
flowchart TD
    A[Route output] --> B[Safety checks]
    B --> C[Compose response envelope]
    C --> D[query]
    C --> E[query_intelligence]
    C --> F[ranked_results<br/>hybrid + optional leanings]
    C --> G[analytics<br/>unchanged by leanings]
    C --> H[guidance<br/>unchanged by leanings]
    C --> I[retrieval_metrics]
    C --> J[pipeline_trace<br/>applied_filters + applied_keywords<br/>+ optional stages]
    C --> K[guard_notice]
```



`pipeline_trace`: always `applied_filters` + `ranked_results_role`; `applied_keywords` when L0 keywords drove encode and/or soft boost; optional `stages` when `measurement.ranking_stage_attribution.include_in_response` (log token `ranking_stage_attribution`). Multi-intent: see §2 (union identified + per-leg `sub_intent_filters`).

## 12. Measurement And Resilience

```mermaid
flowchart TD
    A[Response emitted] --> B[Record SearchObservation]
    A --> C[Record feedback/proxy signals]
    A --> D[Update cache stats]
    A --> E[Backend health registry]
    E --> F{dependency degraded?}
    F -->|yes| G[Degradation planner]
    G --> H[vector -> structured -> cache]
    G --> I[structured -> sql -> cache]
    G --> J[sql -> structured -> cache]
    F -->|no| K[Normal routing continues]
    A --> L[LLM cost gate<br/>per-query + fleet]
    L -->|over budget| M[LLMError → regex L0 + L1]
    L -->|under budget| K
```



Cost: `cost_budget` (per-query USD + fleet hour/day). Search never empty-rejects on spend; analytics may return `cost_budget_exceeded`.

## 13. Feedback Signals And Optional Retraining

```mermaid
flowchart TD
    A[User action<br/>click resume promote label] --> B[FeedbackSignal]
    B --> C{positive signal?}
    C -->|no| D[Observability only]
    C -->|yes| E{retraining enabled?}
    E -->|no| D
    E -->|yes| F[Group by archetype]
    F --> G[Centroid retrainer candidate]
    G --> H{shadow agreement passes?}
    H -->|yes| I[Promote centroids]
    H -->|no| J[Reject candidate]
    F --> K[Offline router retraining<br/>when threshold clears]
```



Boundaries: retraining is optional/config-driven. Positive signals do not update confidence calibration; golden calibration seeds remain separate from traffic-derived seeds.

## 14. Error And Retry Flow

**LLM model chain:** `LLMProvider.get_fallback_chain(task)` =
`task_model_allowlists.<task>` ∩ GoCaas discovery (preference order preserved).
Primary = chain[0] (`get_primary_model`). Empty chain → fail loud / degrade.
Keys from `llm_api_keys` (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`).
Offline harness LLMJ uses the same extraction primary unless `L0_GROUNDING_MODEL`
overrides — see `[EVALUATION.md](EVALUATION.md)`.

```mermaid
flowchart TD
    A[Failure] --> B{failure type}
    B -->|validation| V[422]
    B -->|rate limit| R[429]
    B -->|missing subsystems| S[503]
    B -->|LLM error| L{fallback model in chain?}
    L -->|yes| L2[try next allowlisted model]
    L -->|no| LE[LLMError / degraded route]
    B -->|Qdrant transient| Q{attempt < 3?}
    Q -->|yes| QB[backoff retry]
    Q -->|no| QE[raise/degrade]
    B -->|timeout| T[explore fallback or typed timeout]
    B -->|driver repeat failure| P[pause driver]
```





## 15. Background Jobs And Shutdown

```mermaid
sequenceDiagram
    participant ASGI
    participant App as lifespan
    participant Jobs as background jobs
    participant Drivers
    participant LLM

    ASGI->>App: startup
    App->>Jobs: start calibration/prewarm/synonym/enriched tasks
    App->>Drivers: start vector/delta/event/history drivers
    App-->>ASGI: ready

    ASGI->>App: shutdown
    App->>Jobs: stop/cancel tasks
    App->>Drivers: stop drivers
    App->>LLM: stop refresh
    App-->>ASGI: shutdown complete
```



Shutdown stops background jobs, drivers, and LLM refresh only. Signals are **not** flushed on exit: JSONL is append-on-write per `record`; the in-memory ring is discarded. In-flight UAT S3 uploads are best-effort and may not finish on hard stop. Sudden task death (OOM / SIGKILL) loses the ring and any ephemeral local JSONL; completed UAT S3 objects remain.

## 16. Storage And Runtime Summary

```mermaid
flowchart LR
    API[FastAPI app] --> QI[QI runtime models]
    API --> Q[(Qdrant<br/>vectors + payloads)]
    API --> CH[(ClickHouse<br/>analytics events MVs)]
    API --> Cache[(in-process caches)]
    API --> Signals[(JSONL + memory ring<br/>UAT S3 CSV)]
    API --> LLM[LLM APIs<br/>when configured]
    Athena[Athena/source reads] --> Seed[data-build seed/backfill]
    Seed --> Q
    Seed --> CH
```



Signals: ring = hot buffer; local JSONL = process/volume path; UAT day-partitioned S3 CSV = durable for `uat_feedback` when upload completes. Retrain / offline learning should prefer durable S3 over local JSONL. Retention knobs: `PLAYBOOK.md` §13.