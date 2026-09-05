# MLE Service Review

**Owner role:** MLE (Machine Learning Engineer / service & platform delivery)  
**Parent:** [AUC Semantic Search - Review Sign-off](REVIEW_PACKET.md)  
**Scope:** Service shape, API contracts, runtime degrade, deploy/ops, infra dependencies.  
**Out of scope:** Filter eval methodology and MLS quality gates (see [MLS Technical Review](REVIEW_MLS.md)).

---

## 1. Reading

Review only what is needed to answer the questions below.


| Doc                 | Focus                                                 |
| ------------------- | ----------------------------------------------------- |
| `DESIGN.md`         | Phase goals, API shape, readiness gaps                |
| `ARCHITECTURE.md`   | Components, hybrid-first path, ClickHouse complements |
| `INFRASTRUCTURE.md` | Katana/ECS, local Compose, deployment assumptions     |
| `PLAN.md`           | Phase sequencing and dependencies                     |
| `PROCESS.md`        | Request paths, `qie_only_mode`, degrade behavior      |
| `PLAYBOOK.md`       | Health, cache, measurement, resilience                |
| `OBSERVABILITY.md`  | Alert catalog + boundaries (code emit / docs names / platform CW wire) |


---



## 2. Predefined questions

Answer each question with **Yes / No / Follow-up needed** and a short note.


| #   | Question                                                                                                    | Answer           | Notes                                                                                                                                                |
| --- | ----------------------------------------------------------------------------------------------------------- | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Can Phase 1 run without Qdrant or ClickHouse using `qie_only_mode=true`?                                    | Yes              | L0 extract only; no retrieve. [DESIGN.md:23](DESIGN.md), [PLAN.md:53-54](PLAN.md), `app.py` qie_only path.                                           |
| 2   | Are Phase 2 dependencies clear: Qdrant/index first, Kinesis when freshness is enabled, ClickHouse optional? | Yes              | [DESIGN.md:24](DESIGN.md), [PLAN.md §2](PLAN.md). CH via `clickhouse.enabled`.                                                                       |
| 3   | Is ClickHouse-down behavior acceptable: hybrid search still serves and complements are stripped?            | Yes              | Hybrid-first; CH complements only. [DESIGN.md:13](DESIGN.md). Phase 1 independent of CH.                                                             |
| 4   | Is the `POST /search` API surface clear for Phase 1 and later full search?                                  | Follow-up needed | Form/OpenAPI exists. Designed for **eRanker** as consumer; fuller contracts planned after personalization / proven quality (not yet). See MLE-Q4/Q5. |
| 5   | Is FIND replacement explicitly out of scope for Phase 1?                                                    | Yes              | [PLAN.md:56-58](PLAN.md): filters for UX learning only; no FIND impact.                                                                              |
| 6   | Are deploy/runtime assumptions acceptable: Katana ECS, Qdrant, optional ClickHouse, Compose local?          | Yes              | [INFRASTRUCTURE.md](INFRASTRUCTURE.md). Phase 1: ECS + LLM; Qdrant/CH for Phase 2.                                                                   |
| 7   | Are readiness gaps visible enough to decide blockers vs follow-up debt?                                     | Follow-up needed | [DESIGN.md §14](DESIGN.md) lists ops gaps; GoCode LLM path remains open (MLE-Q1).                                                                    |


---



## 3. Phase 1 follow-up questions and author resolutions

Use this table for reviewer questions. Author fills the resolution so MLS can trace what changed or why no change was made.


| ID      | Reviewer question                                            | Author resolution                                                                                                                                                                                                                                                                                                                            | Doc / code / ticket link                                                                                | Status   |
| ------- | ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | -------- |
| MLE-Q1  | **BLOCKER:** Route LLM via GoCode/GoCaaS before deploy?      | **Open — raised early, not new.** Service-account / GoCode dependency (plus data, LLM, real-time ingest, contracts) was flagged **first week of June**. Followed up ~2–3 weeks; still open amid other MLE priorities. Happy to partner on GoCode S2S + `gd_auth` / `#go-caas`. Deploy gated on path or waiver. Regex L0 remains for degrade. | [DESIGN.md:42](DESIGN.md), [ARCHITECTURE.md:60-61](ARCHITECTURE.md), `llm_core/provider.py`; TDL GoCode | Open     |
| MLE-Q2  | Add app-owned L0 LLM cache for `qie_only`?                   | **Fixed.** In-process TTL+LRU (~1h / 10K). Key = `exact_query_key(versioned_query_key(normalize_query(…), prompt_tag, schema_version))` → `identified_filters` (LLM successes only; regex not cached). Hit → `source=L0_llm_cache`, `decision_cost_usd=0`. Per-task memory only (lost on ECS restart — see Q3). | `app.py` `_QIE_L0_FILTER_CACHE`; `cache/keys.py` `versioned_query_key`; `cache/lru_ttl.py` | Answered |
| MLE-Q3  | Make learning signals durable across ECS restarts?           | **Partial.** UAT feedback → JSONL + ring + best-effort S3 CSV, keyed by durable `search_id`. Full durable SoT for all signal types still open (Q13/Q14).                                                                                                                          | `submit_feedback`; `s3_feedback_uploader.py`; `identity.*`                                               | Open     |
| MLE-Q4  | Publish `/search` + `/feedback` consumer contracts?          | **Planned later (by design).** Primary consumer is **eRanker** (ranked results). Auction/feedback contract pack scheduled for later personalization once quality is proven; experiment not there yet, so not prioritized now. Form/OpenAPI is interim. Will publish schemas + samples when that stage lands.                                 | `app.py` `search` / `submit_feedback`; ties §2 Q4                                                       | Open     |
| MLE-Q5  | `request_id` on search? Feedback shape? `X-Feedback-Key`?    | **Current.** Search returns `request_id` (trace) and `search_id` (durable). Feedback joins on `search_id` (`identity.feedback_search_id_mode`: soft_generate \| required). Client Form/header supply via `identity.*` + `id_value_pattern`. `request_id` optional on feedback (trace). `X-Feedback-Key` env-gated. | `identity.py`; `IdentityConfig`; `submit_feedback`; PLAYBOOK §13 | Answered |
| MLE-Q6  | Define Phase 1 logging schema?                               | **Fixed in code.** `qie_only_complete` emits `request_id`, `query_len`, `latency_ms`, `decision_tier`, `filter_count`, `hard_filter_count`, `source`, `model_id`, `token_count`, `decision_cost_usd`, `grounded_drop_count`, `prompt_tag`, `schema_version`. Response echoes `request_id`, `grounded_drop_count`, `prompt_tag`, `schema_version`. Grounder missing → `sanitize_identified_filters` (include∩exclude strip + inverted price swap). See Q15. | `app.py` qie_only; `qi/grounding.py`; [DESIGN.md §12](DESIGN.md); [OBSERVABILITY.md](OBSERVABILITY.md) | Answered |
| MLE-Q7  | Document model tier + monthly cost target?                   | **Fixed.** Primary = first discovered id in `task_model_allowlists` (shared classify/extract/rewrite preference in `base.yaml`). Caps `$0.05`/q, `$25/h`, `$100/day`. Keys: `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY`. Owner: MLE. Refine fleet $ from live `decision_cost_usd`. | [DESIGN.md §11](DESIGN.md); `base.yaml` `task_model_allowlists` | Answered |
| MLE-Q8  | FIND API param names + JSON schema for `identified_filters`? | **Fixed (docs).** Schema `{name,value,source}`; FIND names (`tldIncludeList`/`maxPrice`); PLAYBOOK §2 drift cleared. Catalog: `find_api_params.json` + `slot_to_api_param.py`.                                                                                                                                                               | [PLAYBOOK.md §3](PLAYBOOK.md); [DESIGN.md §8](DESIGN.md)                                                | Answered |
| MLE-Q9  | Auth on `/search` before prod?                               | **Accepted for prod.** Feedback key exists; `/search` is rate-limited only today. Shared secret or IAM before prod; confirm Katana edge for early demo.                                                                                                                                                                                      | [DESIGN.md §12](DESIGN.md)                                                                              | Open     |
| MLE-Q10 | Adopt MLE FastAPI template as scaffold?                      | **Deferred.** Keep current service; reuse template patterns for GoCode, signals, logging where useful.                                                                                                                                                                                                                                       | `packages/semantic-search/`                                                                             | Deferred |
| MLE-Q11 | ECS 2 vCPU / 4 GB; no GPU — agree?                           | **Agree.** Phase 1 is LLM API + regex; embeddings are Phase 2. No change.                                                                                                                                                                                                                                                                    | [INFRASTRUCTURE.md §5](INFRASTRUCTURE.md)                                                               | Answered |
| MLE-Q12 | LLM dominates cost; no Phase 1 GCR claim — agree?            | **Agree.** Learning phase; levers are model tier (Q7) + cache (Q2) after GoCode (Q1).                                                                                                                                                                                                                                                        | [PLAN.md:63](PLAN.md)                                                                                   | Answered |
| MLE-Q13 | Signal retention policy (ring size, rotation, TTL)?          | **Fixed (docs).** Ring FIFO **5000** (no TTL); JSONL append (no app rotation); UAT S3 day CSV = durable SoT for `uat_feedback`. Knobs in `feedback.`*. Full durable SoT for all types still Q3.                                                                                                                                              | [PLAYBOOK.md §13](PLAYBOOK.md); [DESIGN.md §9](DESIGN.md)                                               | Answered |
| MLE-Q14 | Shutdown flush / recovery when task dies?                    | **Partial (docs).** PROCESS/PLAYBOOK state current behavior: no signal flush on shutdown; JSONL append-on-write; UAT S3 best-effort; hard kill loses ring/ephemeral JSONL. Full Accepted (await durable write + retrain → durable SoT) blocked on **Q3**.                                                                                    | [PROCESS.md §15–16](PROCESS.md); PLAYBOOK §14                                                           | Open     |
| MLE-Q15 | Log levels, CW retention, alerting, token/cost in logs?      | **Fixed (docs).** Levels + cost fields with Q6/Q7. CW retention = platform. Alert **catalog** in [`OBSERVABILITY.md`](OBSERVABILITY.md) § Boundaries: code = emit; doc = names/filters only; **no in-repo CW alarms** (platform wires). OTel deferred.                                                                                    | [OBSERVABILITY.md](OBSERVABILITY.md) § Boundaries; [PLAYBOOK.md §12](PLAYBOOK.md); [DESIGN.md §12](DESIGN.md) | Answered |
| MLE-Q16 | What does `GET /feedback` return?                            | **Fixed (docs).** Operator S3 CSV export: `date_from`/`date_to` (default 7d); columns signal_id, search_id, request_id, signal_type, signal_origin, created_at_iso, comment, query; 503 if S3 unset. Not Auction runtime.                                                                                                                      | [PLAYBOOK.md §13](PLAYBOOK.md); [DESIGN.md §8](DESIGN.md)                                               | Answered |
| MLE-Q17 | Automated UX feedback vs offline review?                     | **Partial.** UX `POST /feedback` with durable `search_id`; offline via GET CSV/logs. Free-text only today. Auction/product confirm still open (ties Q4).                                                                                                                                                                                     | [PLAYBOOK.md §13](PLAYBOOK.md); Q4/Q5                                                                   | Open     |




**Status values:** Open · Answered · Deferred · Closed

---

## 3b. Phase 2 follow-up questions and author resolutions

Scope: Phase 2 semantic search (`qie_only_mode=false`). Thank you for the clear sizing and TDL notes — responses below sync with current code/config.


| ID       | Reviewer question                                              | Author resolution                                                                                                                                                                                                                                                                 | Doc / code / ticket link                         | Status   |
| -------- | -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ | -------- |
| MLE-P2-Q1 | Justify Qdrant vs in-process / TDL OpenSearch preference?     | **Keep Qdrant for Phase 2.** It already powers the current live semantic search in prod, so we stay on the same store to limit risk and reuse ops. Also fits hybrid dense + sparse + payload filter and delta `set_payload` without re-embed. Happy to discuss OpenSearch later if platform asks. | [DESIGN.md §10](DESIGN.md); `delta_refresh` in `base.yaml` | Answered |
| MLE-P2-Q2 | Ship Phase 2 without ClickHouse (TDL StarRocks)?               | **Agree for core search.** Hybrid-first already works with CH off (`clickhouse.enabled`). Phase 2 launch = Qdrant hybrid; analytics/explore/`rank_leanings` optional later. If OLAP returns, will evaluate StarRocks per TDL.                                                   | [DESIGN.md:13](DESIGN.md); [PLAN.md §8](PLAN.md) | Answered |
| MLE-P2-Q3 | Production LLM cost target for multi-stage QI?                 | **Fixed (docs).** Same allowlist-driven primary per stage (classify / L0 extract / rewrite). Caps `$0.05`/q, `$25/h`, `$100/day`. Owner: MLE. Refine from live `decision_cost_usd` / `model_id`. | [DESIGN.md §11](DESIGN.md); `base.yaml` `task_model_allowlists` | Answered |
| MLE-P2-Q4 | Do Kinesis / event streams exist, or must we create them?      | **Planned Phase 2 — not creating new streams.** Expect to reuse Auctions/FIND event feeds (FIND already consumes). Follow with Auctions team for access + contracts; then wire a **Lambda consumer** into our Qdrant/CH paths. Drivers stay config-gated / default **off** until contracts learned; seed/index still serves without NRT. | [PLAN.md §10](PLAN.md); [DESIGN.md §7.2](DESIGN.md) | Answered |
| MLE-P2-Q5 | Define “priority inventory” for indexing?                      | **Fixed + verified in code.** Live (`active_only`) + FIND types **16/20/38/39** hard-filtered in seed SQL (`auction_type_id IN (…)`) + YAML 14d/1M. Query `active_only_baseline`. Bid/watch NRT ≠ membership. | [DESIGN.md §7.2](DESIGN.md); `db_seed_source.py` `_SELECT`; [PLAN.md §10](PLAN.md) | Answered |
| MLE-P2-Q6 | **BLOCKER:** Route LLM via GoCode before Phase 2 prod?         | **Open — same as Phase 1 MLE-Q1.** Multi-stage QI raises urgency. Partnering on GoCode S2S / `#go-caas`. Regex L0 + degrade remain.                                                                                                                                               | MLE-Q1; `llm_core/provider.py`                   | Open     |
| MLE-P2-Q7 | Fill Test/Prod Katana account IDs + store network access?      | **Accepted.** Will complete before higher-env promote.                                                                                                                                                                                                                            | [DESIGN.md §14](DESIGN.md); `configs/katana*`    | Open     |
| MLE-P2-Q8 | Document eRanker integration path (or remove from Phase 2)?    | **Deferred for Phase 2.** Base is `eranker.backend: noop` with `latency_budget_ms: 200`; unhealthy → skip. Will document model source/train path when enabling; not required for hybrid launch.                                                                                   | [DESIGN.md §10](DESIGN.md); `base.yaml` `eranker` | Deferred |
| MLE-P2-Q9 | Observability artifacts (latency/error/store/LLM alerts)?      | **Fixed (docs catalog).** [`OBSERVABILITY.md`](OBSERVABILITY.md) § Boundaries: **code** emits logs/probes/`measurement.thresholds`; **doc** names `auc-ss-{env}-*` + CW filter patterns + wire checklist; **platform** owns CW wire/SNS — naming ≠ create. No in-repo CW/Prom/OTel. Closes with MLE-Q15. | [OBSERVABILITY.md](OBSERVABILITY.md) § Boundaries; [PLAYBOOK.md §12](PLAYBOOK.md); `base.yaml` | Answered |
| MLE-P2-Q10 | API/Qdrant/CH sizing + no GPU — agree?                        | **Agree.** Steady-state 2–3 API replicas; Qdrant ≥8 GB task if used; no GPU (FastEmbed CPU). CH optional (see P2-Q2).                                                                                                                                                             | [INFRASTRUCTURE.md](INFRASTRUCTURE.md); review sizing table | Answered |


**Status values:** Open · Answered · Deferred · Closed

### Feedback / review coverage

**Phase 1:** [arch-review-2026-07-16.md](https://github.com/gdcorp-domains/MLOps-hq/blob/auc-semantic-search-arch-review/projects/agentic-search/arch-review-2026-07-16.md) → MLE-Q1…Q17.

| Source item                               | Mapped to                                          |
| ----------------------------------------- | -------------------------------------------------- |
| Concerns 1–6                              | Q1, Q3, Q4, Q6, Q2, Q7                             |
| Rec Critical 1 / Important 2–6 / Nice 7–8 | Q1; Q2–Q4, Q6–Q7; Q10, Q9                          |
| Design-review §§1–6                       | Q8; Q1; Q11/Q10; Q3/Q13/Q14; Q6/Q15; Q4/Q5/Q16/Q17 |
| Sizing + cost observations                | Q11, Q12                                           |
| Summary table                             | Q8, Q1, Q10, Q3, Q6, Q4                            |

**Phase 2:** [arch-review-phase2-2026-07-18.md](https://github.com/gdcorp-domains/MLOps-hq/blob/auc-semantic-search-arch-review/projects/agentic-search/arch-review-phase2-2026-07-18.md) → MLE-P2-Q1…Q10.

| Source item                                      | Mapped to                |
| ------------------------------------------------ | ------------------------ |
| Concerns 1–5                                     | P2-Q1, Q2, Q3, Q4, Q5    |
| Rec Critical 1–2                                 | P2-Q6, P2-Q5             |
| Rec Important 3–6                                | P2-Q2, P2-Q1, P2-Q3, P2-Q4 |
| Rec Nice 7–9                                     | P2-Q7, P2-Q8, P2-Q9      |
| Sizing + cost / CH question / Qdrant assessment  | P2-Q10, P2-Q2, P2-Q1     |
| Design gaps (inventory, streams, eRanker, ops)   | P2-Q5, Q4, Q8, Q7, Q9    |


---



## 4. Sign-off


| Reviewer | Date | Status | Blocking? | Final notes |
| -------- | ---- | ------ | --------- | ----------- |
| MLE      |      |        | Yes / No  |             |


**Status values:** Approved · Approved with follow-ups · Changes requested · Not reviewed

After final status: update the roll-up table on [Review Sign-off](REVIEW_PACKET.md).