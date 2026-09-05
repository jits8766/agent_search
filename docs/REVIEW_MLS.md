# MLS Technical Review

**Owner role:** MLS (Machine Learning Scientist / relevance & QI)  
**Parent:** [AUC Semantic Search - Review Sign-off](REVIEW_PACKET.md)  
**Scope:** QI quality, filter grounding, evaluation, hybrid-first ML intent, cache/filter policy, auction ranking.  
**Out of scope:** Katana/ECS, secrets, CI (see [MLE Service Review](REVIEW_MLE.md)).  
**Source:** [Confluence MLS review](https://godaddy-corp.atlassian.net/wiki/spaces/~ftahmasebian/pages/4516184369/MLS+Review+AUC+Semantic+Search+Query+Intelligence+Retrieval+and+Auction+Decision+Support) (v10).

---

## 1. Reading

Review only what is needed to answer the questions below.


| Doc                | Focus                                |
| ------------------ | ------------------------------------ |
| `DESIGN.md`        | Phases, QI/degrade, hybrid-first     |
| `PLAN.md`          | Sequencing, FIND boundary            |
| `PROCESS.md`       | L0 / `qie_only_mode`, pipeline_trace |
| `PLAYBOOK.md`      | Cache, measurement, resilience       |
| `OBSERVABILITY.md` | QI emit / cost / latency fields      |


---



## 2. Predefined questions


| #   | Question                                                                                          | Answer           | Notes                                                              |
| --- | ------------------------------------------------------------------------------------------------- | ---------------- | ------------------------------------------------------------------ |
| 1   | Is Phase 1 scope clear: grounded `identified_filters` only, no FIND ranking change?               | Yes              | Learning only; `qie_only_mode=true`.                               |
| 2   | Is the L0 extraction path acceptable: LLM extraction plus fallback when unavailable or empty?     | Yes              | LLM-first; regex only when LLM unavailable. See MLS-Q1 / Q5.       |
| 3   | Is inventory grounding required before filters are emitted?                                       | Yes, with caveat | Phase 1: value ground TLD/type. Phase 2+: cardinality. See MLS-Q3. |
| 4   | Is the proposed filter eval approach enough for Phase 1 learning?                                 | Yes              | Offline suites + Auctions ML review + Hivemind. See MLS-Q2.        |
| 5   | Are known MLS failure modes captured well enough to proceed?                                      | Yes, with caveat | Phase 1–2 experiment OK; Phase 3 dashboards. See MLS-Q18.          |
| 6   | Is the hybrid-first rule acceptable: listing ranks come from retrieval, not ClickHouse analytics? | Yes              | CH complements only.                                               |
| 7   | Are Phase 2/3 ML items correctly deferred and non-blocking for Phase 1?                           | Yes, with caveat | Phase 1 instruments go/no-go. See MLS-Q14 / Q15 / Q17.             |


---



## 3. Follow-up questions and author resolutions


| ID      | Reviewer question                                                   | Author resolution                                                                                                                                                                                                                                                                                                                                                                                   | Doc / code / ticket link                                                                                                                                                                                                               | Status   |
| ------- | ------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| MLS-Q1  | Justify LLM-first L0 vs regex-first / hybrid via offline benchmark? | **Phase 1:** LLM-first (accuracy > latency); regex only if LLM unavailable. **Phase 2/3:** optional A/B if needed — not blocking.                                                                                                                                                                                                                                                                   | `l0_regex_entity`; `_should_run_regex_l0_fallback`; [DESIGN.md §11](DESIGN.md)                                                                                                                                                         | Answered |
| MLS-Q2  | Phase 1 offline eval plan for `identified_filters`?                 | **Phase 1:** Offline done — 696 search + 306 filter queries; Auctions ML reviewing filter suite; Hivemind/proxy for live UX friction. **Phase 2/3:** go/no-go → MLS-Q14 / Q15.                                                                                                                                                                                                                      | `[test_search_queries.md](../../test_search_queries.md)`; `[test_filter_queries.md](../test_filter_queries.md)`; `extract_test_filters_queries.py`; `eval_test_search_queries.py`; Hivemind                                            | Answered |
| MLS-Q3  | Inventory grounding + cardinality before hard filters?              | **Phase 1:** TLD/type include+exclude value-grounded in-process (allowlist; no Qdrant); inverted min/max price swapped. **Phase 2:** Qdrant cardinality count on retrieve path; must fit ~2–5s SLO. CH optional. **Phase 3:** N/A.                                                                                                                                                                  | `qi/grounding.py` `ground_identified_filters`; `app.py` qie_only; [DESIGN.md](DESIGN.md) Search SLA; `base.yaml` `search_timeout_seconds` / `p99_latency_ms_max`                                                                       | Partial  |
| MLS-Q4  | Per-entity confidence + hard/soft quality?                          | **Phase 1:** each filter has `chip_kind` + config-driven `confidence`; soft from entity-slot soft set; FIND/hard locals stay hard; not used to drop hard apply. **Phase 2:** hard-chip gate + soft rank applier for chip↔list parity. **Phase 3:** confidence→ranking priors deferred (`eranker` noop / flat confidence).                                                                           | `filters_to_identified`; `qi.entity_slots`; `qi.l0_llm_entity.confidence`; `qi.l0_regex_entity.confidence`; `extract_hard_filters_from_intent`; SoftKeywordApplier; MLS-Q10                                                            | Partial  |
| MLS-Q5  | Regex gap-fill when LLM returns partial results?                    | **Phase 1:** won't do — tried; diluted quality; regex only when LLM unavailable. **Phase 2/3:** same unless new evidence.                                                                                                                                                                                                                                                                           | `_should_run_regex_l0_fallback`; `l0_regex_entity`                                                                                                                                                                                     | Answered |
| MLS-Q6  | Negative constraints represented and tested?                        | **Phase 1:** exclude slots already emitted; include∩exclude contradictions stripped both sides after grounding. **Phase 2:** TLD/type excludes → Qdrant `must_not` + structured exclude; keyword exclude = post-filter. **Phase 3:** N/A.                                                                                                                                                           | `ground_identified_filters`; `_strip_include_exclude_conflicts`; `qdrant_adapter.py` `must_not`; `structured_retriever.py`                                                                                                             | Answered |
| MLS-Q7  | Cache correctness when entry from regex fallback?                   | **Phase 1:** `qie_only` caches LLM successes only; regex not cached. **Full-search:** intent-result cache may store regex-sourced entities when tier ≠ fallback. **Phase 2:** A/B cost/latency on recovery repeats. **Phase 3:** N/A.                                                                                                                                                               | `app.py` `_QIE_L0_FILTER_CACHE`; `engine.py` intent cache put; `base.yaml` `intent_result_cache`; [DESIGN.md](DESIGN.md) Search SLA / cost; [OBSERVABILITY.md](OBSERVABILITY.md)                                                       | Answered |
| MLS-Q8  | Semantic cache disabled or structurally gated?                      | **Phase 1:** fuzzy semantic intent cache off; unused by `qie_only`. Counters already on `/cache/stats` (MLS-Q9). **Phase 2:** structural gate (TLD/negation/price) before enable — ~2–3 weeks. **Phase 3:** N/A.                                                                                                                                                                                    | `base.yaml` `semantic_intent_cache`; `qi_semantic_intent_cache.py`; MLS-Q9; [OBSERVABILITY.md](OBSERVABILITY.md)                                                                                                                       | Partial  |
| MLS-Q9  | Measured cache hit rate / safe `normalize_query` improvements?      | **Phase 1:** Yes — hit/miss measurable per cache tier; hit-rate / miss-storm signals use config tier lists (user-facing search tiers; not `qie_l0`). Safer normalize is config-driven (case, whitespace, quotes, trailing punctuation). **Phase 2 (done with MLS-Q16):** `qie_only` L0 cache keys `normalize_query` + L0 `prompt_tag`/`schema_version` (trailing `?` shares key). **Phase 3:** N/A. | `qi.normalize`; `versioned_query_key`; `QIEngine.cache_stats` / `app.cache_stats`; `general.search.qie_l0_filter_cache`; `measurement.thresholds.cache_hit_rate_tiers`; `cache_miss_storm.tiers`; [OBSERVABILITY.md](OBSERVABILITY.md) | Partial  |
| MLS-Q10 | Prevent hard-filter over-constraining?                              | **Phase 1:** report all identified filters; no hard-chip cap; log hard count; inverted price swapped in grounding. **Phase 2:** hard chips stay applied for chip↔list parity; zero-result ladder + cardinality (MLS-Q3) still open. **Phase 3:** N/A.                                                                                                                                               | `qie_only_complete` `hard_filter_count`; `extract_hard_filters_from_intent`; `zero_result_guard`; MLS-Q3 / Q4                                                                                                                          | Partial  |
| MLS-Q11 | Auction signals in ranking (traffic, bids, time, price, win prob)?  | **Phase 1:** out of scope (`qie_only`, no ranking). **Phase 2:** fusion ranks; eRanker stays noop at hybrid launch. **Phase 3:** auction signal layer when eRanker on.                                                                                                                                                                                                                              | `base.yaml` `eranker`; [DESIGN.md §10](DESIGN.md); MLE-P2-Q8                                                                                                                                                                           | Deferred |
| MLS-Q12 | Personalization vs QI price for logged-in bidders?                  | **Phase 1:** explicit price = hard filter; no implicit hard price. **Phase 2:** same. **Phase 3:** personalization via eRanker / ranking prior.                                                                                                                                                                                                                                                     | [PLAN.md](PLAN.md) Phase 3                                                                                                                                                                                                             | Deferred |
| MLS-Q13 | Qdrant: enhance existing vs new AUC vs consolidate?                 | **Phase 1:** N/A (no Qdrant). **Phase 2:** keep prod auction Qdrant. **Phase 3 / later:** consolidation ADR if platform asks.                                                                                                                                                                                                                                                                       | [DESIGN.md §10](DESIGN.md); MLE-P2-Q1                                                                                                                                                                                                  | Open     |
| MLS-Q14 | Retrieval benchmark vs baselines?                                   | **Phase 1:** filter-suite eval only (MLS-Q2) — not retrieval NDCG. **Phase 2:** NDCG@10 / Recall@K / MRR / zero-result vs baselines. **Phase 3:** N/A.                                                                                                                                                                                                                                              | `eval_test_search_queries.py`                                                                                                                                                                                                          | Open     |
| MLS-Q15 | Stop condition if QI ≤ simpler baselines?                           | **Phase 1:** learning success via Hivemind + Auctions ML review (MLS-Q2). Poor retrieval would be stop condition here. Phase **2:** publish go/no-go before quality claim. **Phase 3:** N/A.                                                                                                                                                                                                        | MLS-Q2 / Q14                                                                                                                                                                                                                           | Answered |
| MLS-Q16 | Prompt / model / schema versioning?                                 | **Phase 1:** L0 complete logs model/tokens/cost/source/`prompt_tag`; response echoes tag. **Phase 2:** config `prompt_tag` + `schema_version` stamped on `QueryIntent` and included in intent-result / semantic-intent / `qie_only` L0 cache keys (bump isolates prior entries). **Phase 3:** N/A.                                                                                                  | `qi.llm.prompt_tag` / `schema_version`; `qi.l0_llm_entity.prompt_tag` / `schema_version`; `cache/keys.py` `versioned_query_key`; `QIEngine._intent_cache_key`; `app.py` qie_only                                                       | Answered |
| MLS-Q17 | Phase 2 ranking attribution / ablations?                            | **Phase 1:** N/A. **Shipped:** log token + optional `stages` (retrieve / fuse / hard-gate / soft / eRanker / diversify / ZRG); ~0 prod latency/$ beyond one log line. **Next:** offline ablation ladder ~1–2w (frozen QI); eRanker shadow ~1w (+≤200ms if sync). **Phase 3:** deeper attribution with live eRanker.                                                                                 | `measurement.ranking_stage_attribution`; `orchestrator.last_ranking_stages`; `_build_pipeline_trace`; [OBSERVABILITY.md](OBSERVABILITY.md)                                                                                             | Partial  |
| MLS-Q18 | Failure modes enumerated, tested, monitored?                        | **Phase 1:** availability degrade + basic logs. **Phase 2:** keep experiment MVP scope. **Phase 3:** failure taxonomy + dashboards after positive signals.                                                                                                                                                                                                                                          | [PLAN.md](PLAN.md); [OBSERVABILITY.md](OBSERVABILITY.md)                                                                                                                                                                               | Deferred |




**Status values:** Open · Answered · Partial · Deferred · Closed

---



### Feedback / review coverage

[Confluence v10](https://godaddy-corp.atlassian.net/wiki/spaces/~ftahmasebian/pages/4516184369/MLS+Review+AUC+Semantic+Search+Query+Intelligence+Retrieval+and+Auction+Decision+Support) → MLS-Q1…Q18.


| Source                                          | Mapped to                     |
| ----------------------------------------------- | ----------------------------- |
| §2 Q1–Q7                                        | §2 above                      |
| LLM-first / no gap-fill                         | Q1, Q5 (answered)             |
| Eval (696 + filter suite) / Hivemind / go-no-go | Q2, Q14–Q15                   |
| Grounding / cardinality / hard                  | Q3, Q10                       |
| Confidence / hard-soft                          | Q4                            |
| Negation (exclude slots)                        | Q6 (answered)                 |
| Cache                                           | Q7–Q9                         |
| Ranking / personalization                       | Q11, Q12                      |
| Qdrant                                          | Q13 / MLE-P2-Q1               |
| Versioning / attribution                        | Q16 (answered), Q17 (partial) |
| Failure modes                                   | Q18                           |


---



## 4. Sign-off


| Reviewer | Date | Status | Blocking? | Final notes |
| -------- | ---- | ------ | --------- | ----------- |
| MLS      |      |        | Yes / No  |             |


**Status values:** Approved · Approved with follow-ups · Changes requested · Not reviewed

After final status: update [Review Sign-off](REVIEW_PACKET.md).