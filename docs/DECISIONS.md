# Decisions And Open Questions

Living log for the [Review Packet](REVIEW_PACKET.md). Capture ADRs and parking-lot items from MLS/MLE (and later roles). Do not bury decisions only in Confluence comments.

---

## 1. Decision log (ADR-lite)


| ID | Date | Decision | Context | Owner | Status |
| -- | ---- | -------- | ------- | ----- | ------ |
| D1 |      | Phase 1 = filters-only (`qie_only`); no Qdrant/CH | PLAN / DESIGN | MLE+MLS | Proposed (from docs) |
| D2 |      | Hybrid-first listing ranks; CH complements only | DESIGN / ARCH | MLS+MLE | Proposed (from docs) |
| D3 |      | L0: LLM first; regex only if LLM empty/unavailable | QI impl | MLS | Proposed |
| D4 | 2026-08-02 | Draft: `qie_only` emits FIND wire fields (`find_query_params` / `find_query_string`); chips stay in `identified_filters`; FIND unchanged; FoS calls FIND | Phase 1 FoS experiment / USIDOM-4100 | MLE | Reference - [FIND/FoS Phase 1](FIND_FOS_PHASE1.md) |
| D5 | 2026-08-02 | `qie_only` FIND wire uses L0 keywords for FIND `query` (with hard filters) and sets `useSemanticSearch` when keywords drive query; term-count/separator policy in `general.search.find_wire`; probability gate is `qi.l0_llm_entity.keyword_min_probability` only | FIND topical relevance on FoS treatment path | MLE | Accepted - [FIND/FoS Phase 1](FIND_FOS_PHASE1.md) §4 |
| D6 | 2026-08-02 | Combined rewrite+L0 extract: token_count > `rewrite_threshold` → one LLM call (rewrite + filters from rewritten); ≤ threshold → extract-only; LLM fail → regex on raw/normalized (no transform). `extract_before_classify` awaits L0 before L1/L2. Flags: `qi.query_transformer.combine_rewrite_with_l0_extract`, `on_rewrite_reject_reextract`, `qi.ensemble.extract_before_classify`, `qi.l0_llm_entity.combined_*` | QI latency (drop separate transform LLM) | MLE | Accepted - [PROCESS](PROCESS.md) §2 |
| D7 | 2026-08-02 | `qie_only` and full search share one preprocess (`_preprocess_query` / `extract_filters_only`). No duplicate app-level transform+extract. When rewritten: JSON includes `query_transform`; classify/split/encode/retrieve use effective text only. Keywords gated by `qi.l0_llm_entity.keyword_min_probability` (sole source for JSON + FIND wire + soft boost; regex path N/A). Soft chips + gated keywords boost full-search rank. L0 extract contract G1–G4 in PROCESS §2 / DESIGN §7.1. | Single QI contract across Phase 1/2 surfaces | MLE | Accepted - [PROCESS](PROCESS.md) §1–§3, [DESIGN](DESIGN.md) §7.1 |
| D8 | 2026-08-02 | Multi-intent: top-level `filters.identified` = union (dedupe name+value); per-leg only in `sub_intent_filters[].identified`. Soft keywords: `pipeline_trace.applied_keywords`; with 2+ keywords, prefer all-terms-in-SLD then fair interleave by keyword probability so one term cannot monopolize `top_k`. | Multi-topic SERP fairness + filter UX clarity | MLE | Accepted - [PROCESS](PROCESS.md) §2 / §9 / §11, [API_OUTPUT_CONTRACT](API_OUTPUT_CONTRACT.md) §5 |
| D9 | 2026-08-04 | Runtime + harness LLM primary = first *discovered* id in `model_selection_strategy.task_model_allowlists` for `query_intent_classification` / `l0_entity_extraction` / `query_rewrite` (shared preference order in `base.yaml`). No hardcoded harness model id. Keys: `llm_api_keys` → `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GOOGLE_API_KEY` (Gemini OpenAI-compat dual-path). Harness LLMJ: env `L0_GROUNDING_MODEL` or extraction primary; HTTP injects primary when form omitted. | Config-driven model selection; remove `DEFAULT_HARNESS_LLM_MODEL` hardcode | MLE | Accepted - [DESIGN](DESIGN.md) §11, [EVALUATION](EVALUATION.md), `llm_core/provider.py` `get_primary_model`, `config/offline_harness_defaults.py` |


**Status:** Proposed · Accepted · Superseded · Rejected

---

## 2. Open questions


| ID | Raised | Question | Raised by | Blocking? | Disposition |
| -- | ------ | -------- | --------- | --------- | ----------- |
| Q1 |        |           |           | Y/N       | Open |


Move closed items to §3.

---

## 3. Closed / resolved


| ID | Resolution | Date | Link |
| -- | ---------- | ---- | ---- |
|    |            |      |      |


---

## 4. Change requests from review


| ID | Source page | Request | Owner | Due | Status |
| -- | ----------- | ------- | ----- | --- | ------ |
|    | MLS / MLE   |         |       |     | Open |


When a stakeholder marks **Changes requested**, add a row here before re-asking for sign-off.
