# Agentic Search Platform for Auctions

**Projected revenue lift:** $750K – $800K incremental gross commerce
**Experience goal:** fast, intent-aware search — near-instant on repeat queries, sub-second on the warm path.

---

## Scope — Backend Only, Same Advanced UX

No frontend redesign. Same Advanced page, search field, and FIND slots. One unified search service replaces today's legacy search wiring with hybrid (semantic + lexical) retrieval.

| Topic | Intent |
| ----- | ------ |
| **Frontend** | Unchanged — same page, same filter slots, same listing cards |
| **Backend** | Single search service; filter slots stay 1:1 with the existing API contract |
| **Typed text** | Backend infers intent and auto-applies only the filters clearly grounded in the sentence; unset slots keep their defaults |
| **Each search** | Combines the active filter set with semantic ranking, and analytics when the query asks a question |
| **Filter change after search** | Backend narrows then re-ranks; widening triggers a bounded refetch |

---

## System Architecture

User input flows through one backend pipeline. Each stage adds intent or narrows results; failures degrade gracefully rather than dead-end.

```
USER INPUT
   │
   ▼
GUARDRAILS        Sanitize input, normalize encoding, defend against injection, rate-limit.
   │
   ▼
QUERY INTELLIGENCE   Decide what the user wants: query type + grounded filters + leftover concept.
   │                 Multiple lightweight classifiers vote; a resolver settles the intent.
   ▼
CACHING            Reuse prior work — exact repeats, near-paraphrases, and resolved plans.
   │
   ▼
ROUTING + RETRIEVAL   Send the query to the right engine:
   │                  search → hybrid retrieval; explore → trending rails;
   │                  guidance → grounded advice; analytics → question-answering.
   ▼
RANKING            Order results by relevance plus live engagement signals.
   │               Optional personalization layer when available.
   ▼
RESPONSE           Compose results, applied filters, and any analytics or guidance.
   │
   ▼
MEASUREMENT & RESILIENCE   Track health and cost; circuit-break failing dependencies. Never blocks the user.
```

### Infrastructure, Compute & Environment (Katana-managed)

The pipeline above is a stateless service deployed on **Katana**, which provisions and operates the AWS platform (compute, networking, secrets, observability, security) inside a private VPC. The search team owns application logic; Katana owns everything under the dashed line.

```
                                Client
                                  │  HTTPS
                                  ▼
                          ┌───────────────┐
                          │  ALB (Katana) │   SSL · health checks · routing
                          └───────┬───────┘
                                  │
                                  ▼
              ┌─────────────────────────────────────────┐
              │   SEARCH SERVICE  (ECS Fargate, Katana)  │   stateless · auto-scaled
              │   intent → retrieve → rank → compose     │   embedder + in-proc cache co-located
              └───┬───────────────┬───────────────┬──────┘
                  │               │               │
        ┌─────────┘               │               └─────────┐
        ▼                         ▼                         ▼
 ┌──────────────┐         ┌──────────────┐          ┌──────────────┐
 │ Vector store │         │  ClickHouse  │          │ LLM providers│
 │ hybrid index │         │  analytics   │          │ (ranked pool │
 │ (self-mgd)   │         │  (self-mgd)  │          │  via proxy)  │
 └──────────────┘         └──────▲───────┘          └──────────────┘
                                 │ live signals
                          ┌──────┴───────────────────┐
                          │ Kinesis + consumers       │  real-time inventory + engagement
                          └───────────────────────────┘

 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
 Cross-cutting (Katana):  Secrets Mgr (keys)  ·  CloudWatch (logs/metrics/alarms)
                          S3 (config + embedding artifacts)  ·  IAM least-privilege
                          VPC: private subnets · NAT · VPC endpoints · security groups
```

**Request flow:** `Client → ALB → Search Service → {vector store | ClickHouse | LLM} as the route requires → response`. Because tasks are stateless, horizontal auto-scaling absorbs the ~5× peak burst with no per-node state.

**Who owns what:** Katana provides CI/CD, ECS compute + auto-scaling, the load balancer, secrets, networking, observability, and IAM. The search team owns the intent/retrieval/ranking logic, the analytics queries, and the vector + ClickHouse stores.

**Environments:** the same container is promoted dev → prod through the Katana pipeline; environment differences (scale, keys, endpoints) come from config and secrets, not code — so the self-host-vs-managed split and the LLM self-host trigger are config flags, not rewrites.

---

## Analytics Engine Strategy

**What:** question-style queries ("how many .com expired this week", "average price of aged domains") resolve through three tiers, cheapest first, stopping at the first tier that can answer.

**Why:** most analytics traffic is either an exact repeat or one of a handful of common shapes (counts, averages, trends, distributions). Cheap tiers absorb that majority, so only novel questions pay the full cost — and even those have a hard time bound, so a question-style query never hangs.

**How — three tiers:**

| Tier | Serves | Speed | Covers |
| ---- | ------ | ----- | ------ |
| 1 — Cache | Exact repeat of a prior question | Instant (sub-ms) | Repeated questions |
| 2 — Hot aggregates | Pre-computed rollups kept current off the live stream | Tens of ms | Trending, price fan-out, ending-soon, common counts/averages |
| 3 — Recent-history scan | Bounded scan of the last ~30 days, run at low priority | Sub-second | Month-window counts, averages, and history not pre-rolled |

Tier 3 runs at low priority so it never starves the hot path. If the full chain can't answer inside the time bound, the response degrades to trending, query-scoped listings rather than an error — the user always gets results. One engine runs all three tiers, so there is a single SQL dialect and a single failure domain. Hot aggregates and the history scan share the same live ingest, so analytics reflects activity within a few seconds.

**Phase 1 focus:** retrieval and ranking scoped first to the two priority auction families (the auto-extend expiry auctions). Widening to more families is a later product step.

---

## Traffic, Latency & Cost

**Projected traffic.** Sized for ~2M queries/month — ~67K/day, ~0.8 QPS average, ~4 QPS at a 5× peak burst (~33% headroom over measured search traffic of ~1.5M/month). All capacity projections use the 2M anchor.

**Validated against a representative mix.** A local single-instance run over ~700 queries spanning the four query types confirmed routing and surfaced real latency and cost. Mix and measured per-type response time:

| Query type | Share of mix | Response p50 | Response p95 | Cost / query |
| ---------- | -----------: | -----------: | -----------: | -----------: |
| Hybrid search | ~34% | ~2.1 s | ~3.7 s | ~$0.0035 |
| Explore / browse | ~21% | ~1.9 s | ~3.2 s | ~$0.0022 |
| Guidance | ~22% | ~1.9 s | ~5.2 s | ~$0.0032 |
| Analytics | ~23% | ~4.7 s | ~8.6 s | ~$0.0039 |

Single local instance sustained ~1.8 QPS with a small memory footprint; routing accuracy held 86–93% across types.

**Latency — how to read these.** These are full-pipeline, effectively cold numbers: every query ran the LLM path with no warm result-cache. They are the upper bound, not the warm-path target. Two levers pull p50 down in production: the result cache absorbing repeats and paraphrases, and the fast classifier short-circuiting before the LLM. **Analytics is the latency tail** (p50 ~4.7 s, p95 ~8.6 s) and currently exceeds the ≤ 2 s aspiration — end-to-end analytics time is dominated by LLM SQL generation, not the engine scan. Closing it depends on the cache and the pre-built query templates carrying the common shapes without LLM generation; until then ≤ 2 s is a target, not a measured guarantee.

**Cost — how to read these.** Measured blended cost is **~$0.003/query** in this configuration (capable model, cold cache) — far above the optimization target. The target is order **$0.0002–0.0005/query**, and it depends entirely on two assumptions holding at scale: heavy prompt-cache reuse, and routing the majority of traffic through a small model rather than the capable one. Budget honestly — until those are validated live, plan against the measured number (~$0.003/q ⇒ **low-thousands $/month at 2M**), and treat the sub-$1K/month figure as the goal the caching and model-tier work must earn, not a given. Fixed infrastructure (vector store + analytics engine + service nodes + ingest) is separate and lands in the low-to-mid four figures per month self-hosted.

**Total monthly cost — both traffic levels, both LLM scenarios.** Fixed infrastructure is roughly flat across this range; LLM spend scales with traffic and swings an order of magnitude on whether the caching + small-model optimization lands.

| Traffic | Fixed infra (self-host) | LLM — measured (~$0.003/q) | LLM — optimized target ($0.0002–0.0005/q) | **Total now** | **Total at target** |
| ------- | ----------------------: | -------------------------: | ----------------------------------------: | ------------: | ------------------: |
| 1.5M/mo (current) | ~$1,650 | ~$4,900 | ~$300–750 | **~$6,500** | **~$2,000–2,400** |
| 2M/mo (design anchor) | ~$1,730 | ~$6,500 | ~$400–1,000 | **~$8,200** | **~$2,150–2,750** |

The gap between "now" and "at target" is entirely the LLM line. Reaching target means cache hits and the small-model fast path carrying the bulk of traffic; until that is proven live, budget against the "now" column.

**Self-host LLM trigger.** Move the LLM in-house only if sustained spend and warm-path latency both cross defined thresholds for several weeks. Migration is a provider swap, not a rewrite.

---

## Capabilities

### 1. Query Intelligence

Goal: turn raw user text into an intent — query type + grounded filters + a leftover semantic concept — using an ensemble of lightweight classifiers that vote in parallel.

- **Normalize first:** spell-correct, expand short queries, disambiguate terms, resolve vague quantifiers. Raw text is preserved for grounding filters.
- **Ensemble:** the components below fire in parallel and a resolver settles the route from their votes.

| Component | Role | Output |
| --------- | ---- | ------ |
| Entity extractor | Pull filter candidates from the text | Grounded filter slots |
| Semantic router | Fast similarity match on query type | Vote (cheap, always on) |
| LLM classifier | Higher-accuracy query-type call | Vote (skipped when degraded) |
| Rule gates | Catch obvious analytics / strong signals | Add or veto a vote |
| Resolver | Tally votes, pick route, set confidence | Final intent |

- **Grounding:** extracted filters are validated against live inventory and adapt when they don't match — vague terms ("cheap", "popular") resolve to live market percentiles rather than fixed defaults. Search never silently returns empty.
- **Hard vs soft filters:** explicit constraints narrow the result pool; implied preferences only nudge ranking, so recall isn't reduced.
- **Compound queries:** each sub-intent runs the full pipeline independently in parallel.
- **Injection defense:** any retrieved content that re-enters an LLM prompt is re-sanitized.

#### What the extractor recognizes

High-level filter categories grounded from the query: domain extension, price, auction type, bid activity, domain age, traffic, valuation, keyword and character constraints, authority/backlink and SEO metrics, and time/urgency. Domain-extension literals are matched as exact filters only — never fed to the semantic encoder.

### 2. Domain-Name Vectorization

Domain names are run-together strings with no word boundaries, so off-the-shelf embeddings need normalization. Offline pipeline: **segment** the name into words, **expand** with related terms, then **embed** into one vector per listing stored alongside structured attributes. The same vector serves every stage. The query side mirrors the index so query and listing meet in the same space.

### 3. Hybrid Retrieval & Caching

- **One retrieval round-trip:** semantic, lexical, and structured-filter matching run together against a single index and fuse server-side. Live-price checks are the one case that merges in a separate signal.
- **Rank fusion:** when more than one ranked list comes back, they merge by rank position, so lists from different engines combine without score calibration.
- **Zero-result guard:** relax filters → semantic-only → trending fallback. A search never dead-ends.

**Tiered result cache** — in-process, so no extra network hop or operational surface. Entries are tagged to the inventory version and ignored once inventory advances.

| Tier | Matches | Speed |
| ---- | ------- | ----- |
| Exact | Same query text | sub-ms |
| Semantic | Near-paraphrase of a prior query | ~ms |
| Plan | Resolved intent + retrieval plan | ~ms |

### 4. Ranking & Personalization Boundary

Phase 1 orders results by fusion rank plus live engagement signals (bid velocity, watch density, competitiveness) and a brandability boost when relevant — no learned ranking in-service. Phase 2 adds an external personalization/ranking service with a latency budget; if it's disabled or slow, ordering falls back to fusion rank.

### 5. Measurement — Proxy Signals

Phase 1 targets the discovery problem: the large majority of priority-auction inventory receives zero bids, so surfacing it is the core opportunity. Success is tracked through proxy signals rather than a single metric:

| Signal | Intent |
| ------ | ------ |
| Zero-result rate | Never dead-end the user |
| Filter-override rate | Intent understood correctly |
| Query-to-click rate | Results are useful |
| Explore-to-search conversion | Empty input converts |
| Cache hit rate | Cost discipline holds |
| Retrieval recall (human-rated) | Retrieval quality |
| Intent classification accuracy | Query-intelligence quality floor |
| Cost per correct query | Catch cost/quality regressions |

### 6. Resilience & Graceful Degradation

Every external dependency sits behind its own circuit breaker — independent failure modes need independent thresholds. When a dependency trips, the system falls back rather than failing.

| Dependency down | Fallback | User impact |
| --------------- | -------- | ----------- |
| LLM | Rules-based intent path | Slightly coarser routing; spend stops |
| Vector store | In-memory + structured search | Lower recall, still answers |
| Analytics engine | Explore / trending stays live | No question answers; listings still flow |
| Everything | Existing filter-based behavior | Same as today — no contract change |

Breakers compose: multiple trips stack their fallbacks rather than conflicting.

**LLM cost controls** (`cost_budget` in `base.yaml`; rates in `llm_core/pricing.yaml`):

| Control | Scope | On breach |
| ------- | ----- | --------- |
| Per-query USD cap | One search / classify / analytics call | Further LLM calls → `LLMError` |
| Fleet USD cap | Hour and/or day (UTC); memory or Redis | Same — admit denied before provider I/O |
| Search / QI | — | **Degrade to regex L0 + L1**; query not rejected |
| Analytics | — | Typed `failure_mode=cost_budget_exceeded` (no regex path) |

Both caps can be on together. Estimates use token counts × `pricing.yaml` (not vendor invoices).

---

## Tool Selection Rationale

Each choice favors fewer moving parts, open licensing, and fit to this workload over raw peak capability. Decisions below are summarized; each carries a re-evaluation trigger so the choice is revisited when assumptions change.

| Area | Choice | Why (high level) |
| ---- | ------ | ---------------- |
| **Vector / hybrid retrieval** | Self-hosted open-source vector store | Single round-trip hybrid search, open license, already deployed. Hosted/closed options cost more and lock in; heavier stores don't pay back at this scale |
| **Analytics backend** | Columnar SQL engine (self-host) | Workload is SQL aggregations, not search — see comparison below |
| **Embedding model** | Open-license, dimension-flexible model | Commercial-friendly license, one model serves all stages, runs CPU-first. Closed embedding APIs add network latency per query |
| **Embedding runtime** | CPU-first ONNX runtime | No GPU dependency, fast cold-start, light footprint |
| **LLM selection** | Ranked provider pool | Pick by per-task need (capability vs cost); auto-failover across providers; no single-vendor lock-in |
| **LLM hosting** | Hosted API in Phase 1 | Zero infra lift; self-host only when spend and latency triggers both fire |
| **SQL safety** | AST-level validator | Rejects destructive statements deterministically before execution, rather than relying on permissions that can drift |
| **Cache** | In-process | No network hop, no extra runtime or backup story at this scale |
| **Rank fusion** | Rank-position fusion | Merges lists from different engines without score calibration |

### Why ClickHouse over OpenSearch / AWS alternatives

The analytics workload is natural-language questions turned into aggregations over a live auction event stream. ClickHouse (a columnar SQL engine) fits that better than a search engine such as Amazon OpenSearch. Head to head at 2M queries/month:

| Dimension | **ClickHouse (chosen)** | Amazon OpenSearch | Druid / Pinot / Materialize |
| --------- | ----------------------- | ----------------- | --------------------------- |
| **Infra** | 2 small nodes + ~300 GB disk; data footprint under 1 GB at current retention | Managed cluster: dedicated masters + data nodes + shards | Specialist cluster + coordination/storage tier |
| **Compute** | Columnar, vectorized scans — built for aggregations | JVM heap; aggregations bolted onto a search core | Sub-second capable, but heavier moving parts |
| **Latency** | Hot rollups in tens of ms; bounded history scan sub-second | Several times slower on the same aggregations | Sub-second, but overkill at this load |
| **Cost / mo** | **~$140–180 self-host** | **~$2,400–2,800 managed** (~15× more) | $$$$ |
| **Maintenance debt** | Low — minimal JVM surface, no shard/heap tuning | High — JVM heap, shard, and lifecycle tuning ongoing | High — more services to operate |

Four decisive reasons:

1. **Native SQL, no translation.** The pipeline already generates SQL, which ClickHouse runs directly. OpenSearch would force the queries into its own DSL — a second query language to generate, validate, and keep safe, doubling the failure surface.
2. **Right engine for the job.** Aggregations and time-window scans are exactly what a columnar engine is built for. Search engines run aggregations on top of an inverted-index search core and are several times slower at it. Full-text — a search engine's real strength — is already covered by the vector store, so OpenSearch would add cost without adding capability.
3. **Cost and operations.** ~15× cheaper to self-host than managed OpenSearch, on two small nodes, with no shard/heap/lifecycle tuning to carry. Sub-second specialist stores (Druid/Pinot/Materialize) cost and operate well above this scale's needs — deferred unless load outgrows ClickHouse.
4. **Freshness.** Pre-computed rollups stay current straight off the live stream, giving few-second freshness without trading query latency against indexing throughput the way a search index does.

**Re-evaluate if:** sustained analytics load climbs well past current peaks, hot-query latency degrades for weeks, or the history scan repeatedly misses its bound after tuning.

**Net:** ClickHouse fits the workload, reuses the existing SQL pipeline, and stays an order of magnitude inside the budget of a managed search engine — which would re-solve a problem the vector store already handles, at higher cost and operational load.

---

## Response Envelope Contract

Every search returns the same envelope, and ranked results are always populated — the user always gets listings. The envelope carries: the query echo, the answer mode (search / analytics / guidance / explore-fallback), latency, the intent verdict, ranked results, analytics and guidance blocks (empty when not applicable), retrieval metrics, a pipeline trace, and a guard notice when any fallback or relaxation occurred.

---

## Multi-Intent Response Composition

Compound queries fan out per sub-intent in parallel and return one merged ranked list with per-intent badges and filter chips. A cap keeps the response focused; lower-priority sub-intents collapse into a refinement option. One merged list is preferred over stacked rails because it surfaces the single best result first, regardless of which intent found it; a side-by-side view stays available as an opt-in.

---

## Explore Rails

Empty or browsing queries get trending rails, each driven by a live signal over a time window: ending within the hour, currently trending, ending soon, latest expiring, high activity today, fresh arrivals, and longer-running auctions. Trending and ending-soon are always live; the rest activate when their source is configured. All rails plus a semantic leg run together and merge into one list, then post-filtered by intent. Signals and windows are configuration-driven.

---

## Fixed-Infra Cost Envelope @ 2M queries/month

Rough monthly self-hosted infrastructure (LLM excluded — see the total-cost projection above): vector store is the largest line, analytics engine and search-service nodes moderate, streaming ingest small. Fixed infra lands at ~$1,650–1,730/month across the 1.5M–2M range; storage and memory footprints sit comfortably inside the sized hardware. The LLM line is variable and sits on top of this — order ~$5–6.5K/month at measured rates, or sub-$1K once the caching and model-tier optimization lands.

---

## Phased Approach

### Phase 1 — Query Intelligence, Retrieval, Live Data

Core service shipped against the two priority auction families, with inventory and behavioral signals sourced from live event streams in real time.

| Capability | Delivery |
| ---------- | -------- |
| Query Intelligence | Ensemble routing across four query types; multi-intent fan-out; entity grounding; calibrated auto-execute / suggest / explore thresholds |
| Vectorization & Retrieval | Segment → expand → embed pipeline; single hybrid index; real-time inventory ingest for the priority families; tiered in-process cache; zero-result guard |
| Analytics | Cache → hot aggregates → recent-history scan; natural-language-to-SQL with safety validation and few-second freshness |
| Ranking | Fusion order plus live engagement boosts; no external ranker yet |
| Live Data | Mutable fields (price, bids, end time) refresh on a short delta cadence — no re-embedding; bid and watch streams feed engagement aggregates |
| Resilience | Per-dependency circuit breakers with layered fallbacks |
| Continuity | Mid-session filter changes re-rank retained candidates; resume last search; saved/shareable searches at the API layer |
| Measurement | Proxy-signal tracking; end-to-end intent trace; a pre-launch quality gate vs an LLM baseline before ramp |

### Phase 2 — Saved Search, Memory, Fine-Tuning, Live Listing Ingest

| Capability | Delivery |
| ---------- | -------- |
| **Real-time listing ingest** | New-listing events streamed in, embedded, and indexed live. The one outstanding data-side dependency; cutover is a config flag (blocked on stream access) |
| **Saved searches & alerts** | Serialized intent + filters as a shareable record; prompts to save a search or alert based on repeat/adjust behavior |
| **Session memory** | Per-user intent history with retention and right-to-delete; cross-session resume against current inventory |
| **External ranker** | Production personalization/ranking via an external service over a typed client and latency budget |
| **Small-model fine-tuning** | Trigger-based: fine-tune a small in-house model on curated intent labels when accuracy plateaus and volume justifies; provider swap is one config change |
| **Production LLM credentials** | Multiple provider accounts with rotation; same ranked-provider engine |
| **Partner-expiry shopper rail** | Per-shopper domains entering the partner-expiry window; opt-in and dismissable |

---

## Data Sources

All sources are real-time streams or static reference dimensions — analytics and listings both run on live data, not daily snapshots. Live auction records, bid streams, and watch streams power hot-path listings, trending and competitiveness signals, and watch-density signals. New-listing ingest is the one outstanding access dependency. If a real-time source goes stale beyond its threshold, the service returns an explicit error rather than serving stale data.

---

## External Dependencies

| Dependency | What's needed | If unavailable |
| ---------- | ------------- | -------------- |
| **Existing filter API** | Stable 1:1 map from each filter slot to its parameter | Same degradation as today — no regression |
| **New-listing stream** | Consumer access for listing ingest — the one outstanding data-side item | Listing index lags new arrivals; existing signals unaffected |
| **Auction event streams** | Auction, bid, and watch streams (available) | Power live records and engagement signals |
| **External ranker** (Phase 2) | Typed client + latency budget | Falls back to fusion order — no regression |
| **LLM credentials** | Dev key for Phase 1; multiple service accounts for Phase 2 | Ranked providers fall back to the rules path |

---

## Measured Baselines

Anchors that size the system: monthly auction inventory and listing-creation rates set storage and ingest capacity (a single CPU absorbs ingest with wide headroom); the diurnal shape concentrates heavy offline jobs in the low-traffic window; bid throughput keeps trending-aggregate write load trivial; the high zero-bid rate on priority auctions defines the discovery problem; and measured search volume anchors the 2M/month design target.

---

## Open Questions

| # | Item | Phase |
| - | ---- | ----- |
| 1 | Frontend engineering commitment to wire the existing search UI to the unified backend. Without it, the headless API is the Phase 1 deliverable | Phase 1 |
| 2 | Confirm a dev/personal LLM key for Phase 1 with budget headroom | Phase 1 |
| 3 | New-listing stream access, schema, and retention — the outstanding data-side dependency for listing indexing | Phase 1 |
| 4 | Analytics retention window and the listing-stream ingest contract | Phase 2 |
| 5 | Locate the valuation/odds dataset and confirm refresh cadence and owner; guidance runs on market aggregates only until it lands | Phase 1-soft |
| 6 | Formal SLA sign-off from data engineering on auction-record replication lag before promoting it to a streaming primary | Phase 1-soft |
