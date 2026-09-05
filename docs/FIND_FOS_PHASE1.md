# FIND / FoS Phase 1 — `qie_only` → live FIND

**Audience:** FIND eng, FoS eng, ML, BA (USIDOM-4100)  
**Purpose:** Integration contract for FoS consuming `qie_only` in a Phase 1 experiment with live FIND.  
**Scope:** response shape, FIND wire compatibility, integration flow, readiness checks, failure handling.  
**Code paths (source of truth):** `qi/qie_only.py`, `qi/find_query_params.py`, `measurement/qie_only_launch.py`, `app.py` `qie_only_mode` branch.

## Quick read

Phase 1 tests whether `qie_only` can turn natural-language auction searches into FIND-safe filters. It does **not** replace FIND listings. FoS owns the experiment split and UI/funnel events; ML owns QI output, FIND wire fields, and QI health gates; FIND stays unchanged except for existing request logging / Kinesis attribution.

Decision makers should use:

- `find_query_string` to call FIND in treatment.
- `GET /measurement/qie_only` for ML-owned pass/fail gates.
- FIND Kinesis + FoS/BA telemetry for treatment/control business outcomes.

**Success = two layers (do not mix):**

| Layer | Question | Pass means | See |
| ----- | -------- | ---------- | --- |
| Launch safety | Safe to scale the pilot? | QI up, fast enough, FIND wire usable | §8 |
| Experiment outcome | Did treatment help users / bids? | Override + zero-result + search-to-bid vs control | §8.1 + §9 |

---



## 1. What Phase 1 is


| Is                                         | Is not                              |
| ------------------------------------------ | ----------------------------------- |
| NL -> FIND structured filters (`qie_only`) | Qdrant / semantic listing retrieval |
| Listings still from FIND OpenSearch        | Replacement of FIND                 |
| Experiment on FoS search submit            | Saved-search schema change          |


Phase 2 (later): swap listing source to hybrid ranks; same chips / filters contract.

---



## 2. Treatment vs control

Hivemind assigns each search session/user to **control** or **treatment**. Same NL query box; only backend path differs. Stamp `experimentInfo=<id>:<split>` on the listing call so Kinesis/BA can attribute.

### Phase 1: test QI filters, keep FIND listings


| Arm       | Flow                                                                                                                                | Listing source  |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------- | --------------- |
| Control   | FoS calls FIND as today. No `qie_only`.                                                                                             | FIND OpenSearch |
| Treatment | FoS calls `POST /search` with `qie_only_mode=true`, then calls FIND with `find_query_string` plus pagination/sort/`experimentInfo`. | FIND OpenSearch |
| Degrade   | QI 422/503/timeout falls back to control. Never show a blank SERP because QI failed.                                                | FIND OpenSearch |


Phase 1 A/B tests **filter quality, chip UX, and search-to-bid** under the existing FIND listing engine. It does not test semantic ranking.

### Phase 2: test listing engine, keep filter contract comparable


| Arm       | Flow                                                                                                       | Listing source         |
| --------- | ---------------------------------------------------------------------------------------------------------- | ---------------------- |
| Control   | Recommended: Phase 1 treatment path (QI filters + FIND listings). This isolates the listing-source change. | FIND OpenSearch        |
| Treatment | FoS calls full/hybrid search with the same chip/filter contract and consumes `ranked_results`.             | Semantic-search hybrid |
| Degrade   | Hybrid/QI failure falls back to the Phase 1-safe FIND path.                                                | FIND OpenSearch        |


Phase 2 A/B tests **listing relevance and SERP quality** after Phase 1 passes. Gate on section 9 metrics: override rate, zero-result rate, and search-to-bid.

---



## 3. qie_only response fields

`POST /search` form: `query`, `qie_only_mode=true`  
Verified response keys from `app.py` `qie_only_mode` branch + `_attach_find_wire_fields`:


| Field                | Use                                                                           |
| -------------------- | ----------------------------------------------------------------------------- |
| `request_id`         | Trace id for this HTTP hop (CloudWatch / retries)                             |
| `search_id`          | Durable search-interaction id — FoS/BA feedback and analysis joins            |
| `answer_mode`        | Always `qie_only` on this path                                                |
| `identified_filters` | Chip UI / override tracking (`name`, `value`, `source`, `chip_kind`, `confidence`; labels / relative times OK) |
| `keywords`           | L0 topical terms (`term`, `probability`); source for FIND `query` when usable |
| `find_query_params`  | Dict of FIND query keys → string values                                       |
| `find_query_string`  | Append after `/auction/recommend?`                                            |
| `soft_chips`         | UX only — do not send to FIND                                                 |
| `find_skipped`       | Hard chips that could not wire (log `name` + `reason`)                        |
| `query_transform`    | Optional; present when shared preprocess accepted a rewrite                   |
| `decision_tier`      | `L0_entity`                                                                   |
| `latency_ms` / `decision_cost_usd` / `grounded_drop_count` / `prompt_tag` / `schema_version` | Ops / measurement |


Example (illustrative values; ISO times and keywords vary by clock / L0 extract):

```http
POST /search
Content-Type: application/x-www-form-urlencoded

query=aged+.com+under+$1000+ending+soon&qie_only_mode=true
```

```json
{
  "answer_mode": "qie_only",
  "request_id": "…",
  "identified_filters": [
    {"name": "tldIncludeList", "value": "com", "source": "L0_llm", "chip_kind": "hard", "confidence": 0.9},
    {"name": "maxPrice", "value": 1000, "source": "L0_llm", "chip_kind": "hard", "confidence": 0.9},
    {"name": "endTimeBefore", "value": "-1d", "source": "L0_llm", "chip_kind": "hard", "confidence": 0.9}
  ],
  "keywords": [
    {"term": "aged", "probability": 0.86}
  ],
  "find_query_params": {
    "query": "aged",
    "endTimeBefore": "2026-08-03T08:30:00Z",
    "maxPrice": "1000",
    "tldIncludeList": "com",
    "useSemanticSearch": "true"
  },
  "find_query_string": "query=aged&endTimeBefore=2026-08-03T08%3A30%3A00Z&maxPrice=1000&tldIncludeList=com&useSemanticSearch=true",
  "soft_chips": [],
  "find_skipped": [],
  "decision_tier": "L0_entity"
}
```

FIND call (FoS owns URL, auth, pagination, sort, `experimentInfo` — not produced by this service):

```http
GET /v4/aftermarket/find/auction/recommend?query=aged&endTimeBefore=2026-08-03T08%3A30%3A00Z&maxPrice=1000&tldIncludeList=com&useSemanticSearch=true&experimentInfo=USIDOM-4100:treatment
X-GDFindAM-APIKey: <FoS FIND API key>
```

Config split (`semantic_search/config/base.yaml`):

| Setting | Where |
| ------- | ----- |
| Keyword probability gate | `qi.l0_llm_entity.keyword_min_probability` (percent → fraction; sole gate) |
| Max terms, separator, empty-query fallback, `prefer_keywords_for_query`, `useSemanticSearch` param/value | `general.search.find_wire` |
| Launch gate thresholds | `measurement.qie_only_launch` |

---



## 4. How to treat `keywords`

`keywords` are topical terms from the same L0 extraction call:

```json
[
  {"term": "brandable", "probability": 0.91},
  {"term": "fintech", "probability": 0.77}
]
```

Wire and consumer rules:

- Terms that pass `qi.l0_llm_entity.keyword_min_probability` (percent → fraction;
  sole probability gate) are sorted by probability and capped by
  `general.search.find_wire.max_keyword_terms`, then become FIND `query`, including
  when hard filters are present (`prefer_keywords_for_query: true`).
- When keywords become `query` and `set_use_semantic_search_when_keywords` is true,
  FIND also receives `useSemanticSearch=<use_semantic_search_value>`.
- When no usable keyword terms remain, FIND `query` is `empty_query_fallback`
  (typically `*`).
- Do not pass `probability` to FIND.
- Do not treat keywords as hard filters / chips.
- Log `keywords` for analysis and chip explanation.
- L0 filter cache envelope is `{identified, keywords, query_transform?}`
  (`_qie_cache_envelope`). Cache hits return the **stored** keywords (not emptied)
  and re-attach FIND wire from those keywords + re-grounded identified filters.
  Cache hit log `source=L0_llm_cache`; cost on hit is `0.0`.

---



## 5. Compatibility rules


| qie_only chip value       | FIND-safe value   |
| ------------------------- | ----------------- |
| `endTimeBefore=-1d`       | Absolute ISO UTC  |
| `typeIncludeList=premium` | `16,38,39`        |
| `excludeHyphens=true`     | `"true"`          |
| `tldIncludeList=com|io`   | `"com,io"`        |
| `topic_include=...`       | `soft_chips` only |


This prevents FIND from silently ignoring chip-friendly values that are not valid live FIND query params.

---



## 6. Launch readiness

`qie_only` is ready to produce Phase 1 FIND input. FoS can use `find_query_string` or `find_query_params` for the FIND call, and use `identified_filters`, `soft_chips`, `find_skipped`, and `keywords` for chips, logging, and review.

What `qie_only` does **not** produce:

- FIND listings, pagination totals, or SERP body. FIND still owns those.
- FIND auth, pagination, sort, and `experimentInfo`. FoS appends those.
- Hivemind split, chip override UI, and funnel reporting. FoS / BA own those.

Known caveats:

- Cache hits restore stored `keywords` (see §4); do not assume empty keywords on hit.
- Top-level JSON has no `source` field; extract source is on `identified_filters[].source` and on the `qie_only_complete` log field (`L0_llm` / `L0_regex` / `L0_llm_cache`).
- When no usable keywords remain, FIND `query` is `empty_query_fallback` (YAML default `*`); log that rate via `hard_params_empty` / wire params.
- Regex keyword probabilities are N/A; regex recovery on this path is inventory-bound empty-LLM only (`L0_regex`).

Reviewer references:

- `packages/semantic-search/semantic_search/qi/qie_only.py` — cache/ground/FIND wire helpers (`_attach_find_wire_fields`)
- `packages/semantic-search/semantic_search/qi/find_query_params.py` — FIND-safe conversion and `find_query_string`
- `packages/semantic-search/tests/test_find_query_params.py` — wire-format coverage
- `docs/API_OUTPUT_CONTRACT.md` — public response field contract

---



## 7. Needed pieces before experiment



### FoS

- [ ] Call `POST /search` with `qie_only_mode=true`.
- [ ] Call FIND with `find_query_string`.
- [ ] Append pagination / sort params as needed.
- [ ] Fallback to current FIND path if qie_only fails or times out.
- [ ] Stamp `experimentInfo=<id>:<split>` on FIND for Kinesis.
- [ ] Log raw query, `identified_filters`, `keywords`, `find_query_params`, `find_skipped`, FIND `Pagination.Total`, chip overrides, and bid funnel.
  Note: semantic-search emits QI fields only; FoS / BA must record chip overrides and search-to-bid.



### FIND

- [ ] No API / ES query changes required for Phase 1.
- [ ] Confirm existing FoS FIND API key + JWT path is valid for experiment volume.



### ML / MLE

- [ ] Keep `qie_only` available for FoS.
- [ ] Monitor QI health: `qie_only_complete` / `qie_only_failed` logs + `GET /measurement/qie_only`.
- [ ] Review Phase 1 pass/fail gates (section 8) before experiment scale-up.
- [ ] Confirm FoS / BA Phase 2 metrics are recording before Phase 2 go/no-go.
- [ ] After L0 prompt / regex changes: spot-check regex vs LLM on a fixed sample (`reground_filters_four_way` or grounded regex + `qie_only`); expect structured hard filters to stay close (keywords N/A on regex).

QI health metrics (Phase 1 ops - structured logs to CloudWatch):


| Metric             | Log field                                              | Why                                  |
| ------------------ | ------------------------------------------------------ | ------------------------------------ |
| Latency p50/p95    | `latency_ms` on `qie_only_complete`                    | QI SLA without over-weighting one gateway outlier |
| Extract source mix | `source` (`L0_llm` / `L0_regex` / `L0_llm_cache`)      | Quality + cost mix                   |
| Grounding drops    | `grounded_drop_count`                                  | Over-extraction / inventory mismatch |
| Chip / hard counts | `filter_count`, `hard_filter_count`                    | Empty vs useful extract              |
| Cost / tokens      | `decision_cost_usd`, `token_count`                     | Cost guard                           |
| FIND wire loss     | `find_skipped_count`, `find_skipped_reasons`           | Silent FIND drops                    |
| Soft-only bleed    | `soft_chip_count`                                      | UX chips not sent to FIND            |
| Broad FIND risk    | `hard_params_empty` (`1` = no hard FIND params)        | `query=*` / keyword-only             |
| Availability       | `qie_only_failed status=422/503` + FoS timeout degrade | Fail-open rate                       |


Measurement ownership (do not invent a second QI path):


| Signal class                | Owner / sink                                           | Notes                                                                                                                       |
| --------------------------- | ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| QI ops above                | ML logs to CloudWatch                                  | Already on `qie_only_complete` / `qie_only_failed`                                                                          |
| FIND SERP total             | FIND Kinesis via `experimentInfo` + `Pagination.Total` | FoS stamps experiment                                                                                                       |
| Chip override / click / bid | FoS + BA analytics                                     | `POST /feedback` is UAT comments only. FoS must emit these events through FoS/BA telemetry or a future typed-signal ingest. |




### BA / experiment

- [ ] Configure control vs treatment split.
- [ ] Track search-to-bid conversion (treatment vs control).
- [ ] Track chip shown vs removed/edited (join to `request_id`).
- [ ] Track FIND `Pagination.Total` after QI params (zero-result rate).
- [ ] Gate Phase 2 on low override rate and non-negative search-to-bid lift.

---



## 8. Phase 1 launch - pass / fail (decision makers)

**Audience:** ML lead, FoS eng, BA, FIND on-call.  
**Question:** Is treatment safe to scale beyond a small pilot?  
**Not this section:** business win/loss of the A/B (§8.1) or Phase 2 listing cutover (§9).

### Decision route (what decides pass vs fail)


| #   | Gate                        | Pass threshold                                                                                   | Have it?        | Where                                                                                      | Owner           |
| --- | --------------------------- | ------------------------------------------------------------------------------------------------ | --------------- | ------------------------------------------------------------------------------------------ | --------------- |
| 1   | QI availability             | fail rate **≤ 5%** (`measurement.qie_only_launch.fail_rate_max`)                                 | **Yes (wired)** | `qie_only_failed` logs + `GET /measurement/qie_only` → `gates.qi_availability`             | ML              |
| 2   | QI latency p95              | **≤ 3000 ms** (`p95_latency_ms_max`)                                                             | **Yes (wired)** | `latency_ms` on `qie_only_complete` + `/measurement/qie_only` → `gates.qi_latency_p95_ms`  | ML              |
| 3   | FIND wire health            | `find_skipped_count > 0` share **≤ 15%** (`find_skipped_rate_max`)                               | **Yes (wired)** | log fields + `/measurement/qie_only` → `gates.find_skipped_rate`                           | ML              |
| 4   | Empty hard FIND params      | `hard_params_empty=1` share **≤ 40%** (`hard_params_empty_rate_max`)                             | **Yes (wired)** | log fields + `/measurement/qie_only` → `gates.hard_params_empty_rate`                      | ML              |
| 5   | FoS wiring                  | treatment: `qie_only` -> FIND + `experimentInfo`; control unchanged; degrade on QI error/timeout | **External**    | FoS deploy checklist / traffic review                                                      | FoS             |
| 6   | FoS timeout degrade         | counted in FoS client metrics; blank SERP forbidden                                              | **External**    | FoS APM / logs                                                                             | FoS             |
| 7   | FIND experiment attribution | Kinesis shows `experimentInfo` splits + `Pagination.Total`                                       | **External**    | FIND Kinesis                                                                               | FoS + FIND      |
| 8   | Safety / auth               | no PII in QI logs; FIND auth unchanged                                                           | **External**    | security review                                                                            | FoS + FIND + ML |


**Wired now (ML):** gates 1-4 are recorded on every `qie_only` success/failure and aggregated in-process.

```http
GET /measurement/qie_only
```

Response fields for decision makers:

- `ml_owned_overall` (gates 1–4 only; needs `min_sample_size`, YAML default **50**):

  ```text
  ml_owned_overall
    = qi_availability
    ∧ qi_latency_p95_ms
    ∧ find_skipped_rate
    ∧ hard_params_empty_rate
  ```

  | Result | When |
  | ------ | ---- |
  | `pass` | all four gate statuses = `pass` |
  | `fail` | any one of the four = `fail` |
  | `insufficient_sample` | any one still under min sample (or no value yet) |

  Talk shorthand: **ml_owned_overall = availability + latency_p95 + find_skipped + hard_params_empty** (AND, not sum).

- `phase1_launch_overall`: always `pending_external_gates` in this service until FoS/FIND confirm gates 5–8 (not auto-flipped in-process).
- `gates.<name>.status`: `pass` / `fail` / `insufficient_sample` / `external`.
- `rates` / `thresholds` / `source_mix` / `counts`: supporting detail.

In-process counters reset on process restart. Durable CloudWatch source: parse `qie_only_complete` and `qie_only_failed` log lines.

### How to call the decision

1. Pilot traffic reaches min sample (default 50 attempts; prefer at least 3 pilot days).
2. Open `GET /measurement/qie_only` (or CloudWatch log math).
3. If `ml_owned_overall=fail`, **hold** scale-up; fix QI / wire issues.
4. If `ml_owned_overall=insufficient_sample`, keep pilot; do not scale.
5. If `ml_owned_overall=pass`, FoS/FIND sign gates 5-8.
6. All eight green means **Phase 1 pass** (safe to scale filter experiment).

Phase 1 **launch** pass is not experiment win and not Phase 2 listing cutover.

**Fail / hold** = any ML gate `fail`, or any external gate unsigned / broken, without owner + mitigation date.

### 8.1 Experiment outcome KPIs (record in Phase 1; decide treatment value)

**Question:** Did QI filters improve (or not hurt) the FIND funnel vs control?  
**Owners:** FoS + BA for business rows; ML for QI proxy logs.  
**Join:** Hivemind split + `experimentInfo` on FIND; QI `request_id` on chip events.

**Decision formula** (BA/FoS call — **not** a field on `GET /measurement/qie_only`):

```text
experiment_outcome_overall
  = ml_owned_overall                          # §8 prerequisite (safe to trust QI)
  ∧ search_to_bid_treatment ≥ control         # primary business KPI
  ∧ zero_result_rate_treatment ≤ control      # over-filter proxy
  ∧ chip_override_rate_treatment ≤ control    # noisy-filter proxy (and under ~25% plan)
  ∧ fos_degrade_on_qi_fail_ok                 # no blank SERP
```

Talk shorthand: **experiment_outcome = ml_owned_overall + bid + zero_result + override + degrade** (AND; each “+” arm vs control).  
Supporting (not in the AND): search-to-click, source_mix, find_skipped — diagnose, don’t alone ship/hold.

| Result | When |
| ------ | ---- |
| `pass` | all AND terms hold with agreed sample |
| `fail` / hold | any business arm worse than control, or blank-SERP degrade broken |
| `insufficient_sample` / blocked | FoS/BA events missing, or `ml_owned_overall` ≠ `pass` |

#### Wired now vs still to build


| KPI | In semantic-search today? | Store / record | Who builds / owns |
| --- | ------------------------- | -------------- | ----------------- |
| Extract source mix | **Yes** | `qie_only_complete.source` → CloudWatch; rollup `GET /measurement/qie_only` → `source_mix` | ML (done) |
| FIND wire loss | **Yes** | `find_skipped_*` on complete logs; `/measurement/qie_only` → `gates.find_skipped_rate` | ML (done) |
| QI fail rate / latency | **Yes** (§8 launch) | same logs + `/measurement/qie_only` | ML (done) |
| QI fail → FoS degrade | **Partial** — fail logs yes; degrade path is FoS client | FoS APM / client metrics + `qie_only_failed` | FoS must emit degrade counts |
| Chip override rate | **Ingest path only** — typed `filter_override` via `POST /feedback` → `SignalStore` JSONL; `GET /measurement/signals` | Empty until FoS posts events (or BA warehouse) | **FoS must emit** UI remove/edit |
| Search-to-click | **Ingest path only** — typed click → `/measurement/signals` | Same; or FoS/BA analytics | **FoS + BA** |
| Search-to-bid | **No** in this service | FoS/BA + bid stream / warehouse | **FoS + BA** (primary outcome) |
| FIND zero-result | **No** in this service | FIND Kinesis `Pagination.Total` + `experimentInfo` | **FoS stamps experimentInfo; FIND/BA aggregates** |
| Treatment vs control split | **No** | Hivemind + `experimentInfo=<id>:<split>` on FIND | **FoS** |


`POST /feedback` without typed funnel signals = UAT comments only — does **not** fill override/click KPIs. `/measurement/signals` ≈0 means **not wired**, not “quality perfect”.

#### How to validate in pilot / prod


| Stage | What to check | Command / sink |
| ----- | ------------- | -------------- |
| Pre-launch | ML launch gates | `GET /measurement/qie_only` → `ml_owned_overall` + gates 1–4 |
| Experiment live? | Treatment traffic + attribution | FoS: Hivemind on; FIND Kinesis shows both `experimentInfo` arms; QI logs show `qie_only` volume |
| Outcome (weekly) | Override / zero-result / click / bid by arm | BA tables (split ↔ `request_id` ↔ FIND total ↔ bid). Optional ML mirror: `/measurement/signals` if FoS posts typed events |
| Hold / ship | Business + safety | §8 green **and** search-to-bid + zero-result + override not worse than control |


**Capture checklist (FoS/BA still own most of this):**

1. FoS stamps `experimentInfo` on every FIND call (control + treatment).
2. FoS logs: `request_id`, chips shown, chip edits/removes, FIND total, click, bid (BA warehouse and/or typed `POST /feedback`).
3. ML: leave `qie_only_complete` / `qie_only_failed` on; read rollup via `GET /measurement/qie_only`.
4. BA joins Hivemind split ↔ FIND Kinesis ↔ FoS events for treatment vs control tables.

**Outcome call:** `experiment_outcome_overall=pass` (formula above). Missing FoS/BA events = cannot claim outcome success, even if `ml_owned_overall=pass`.

---



## 9. Phase 2 readiness metrics (same signals, later decision)

Phase 2 = hybrid / semantic listings replace FIND SERP. Reuse §8.1 capture; decide listing cutover, not filter pilot alone.


| Metric                               | Needed for Phase 2                     | Who records                                                      | Join key                       |
| ------------------------------------ | -------------------------------------- | ---------------------------------------------------------------- | ------------------------------ |
| Chip shown vs removed/edited         | Intent quality / override rate         | FoS UI events (or future typed signal ingest into `SignalStore`) | QI `request_id`                |
| FIND total after QI params           | Zero-result / breadth                  | FIND Kinesis `Pagination.Total`                                  | `experimentInfo` + FIND req id |
| Treatment vs control search-to-click | Engagement proxy                       | FoS / BA                                                         | Hivemind split                 |
| Treatment vs control search-to-bid   | Primary business gate                  | FoS / BA + bid stream                                            | Hivemind split                 |
| Soft-chip interactions               | Soft-to-hard promotion candidates      | FoS                                                              | QI `request_id`                |
| Regex vs LLM mix under load          | Cost/quality for Phase 2 traffic shape | `qie_only_complete.source`                                       | n/a                            |


Phase 2 go/no-go (defaults; BA may tighten):

- Chip override / removal rate **low** vs agreed threshold (plan proxy: filter-override < ~25%).
- Search-to-bid lift is at least neutral (treatment not worse than control) with sufficient sample.
- Zero-result rate not worse than control after QI params.
- QI availability + latency still inside Phase 1 gates at Phase 2 volume.

---



## 10. Failure handling


| Case                         | Expected behavior                             |
| ---------------------------- | --------------------------------------------- |
| qie_only 422 / 503 / timeout | FoS calls current FIND path                   |
| `find_skipped` not empty     | Log it; still call FIND with remaining params |
| `soft_chips` present         | Show/log only; do not send to FIND            |
| FIND returns zero results    | Record FIND total and follow-up user action   |


---



## 11. Sources

- Ticket: [USIDOM-4100](https://godaddy-corp.atlassian.net/browse/USIDOM-4100)
- Code: `qi/find_query_params.py`, `qi/qie_only.py`, `app.py` (`qie_only_mode`), `measurement/qie_only_launch.py`
- Config: `semantic_search/config/base.yaml` → `general.search.find_wire`, `qi.l0_llm_entity.keyword_min_probability`, `measurement.qie_only_launch`
- Decision endpoint: `GET /measurement/qie_only`
- API response notes: [API_OUTPUT_CONTRACT.md](API_OUTPUT_CONTRACT.md)
- Decision log: [DECISIONS.md](DECISIONS.md) D4 / D5
- Plan proxies: [plan_agentic_search.md](../plan_agentic_search.md) measurement table

**Out of this service (FoS / FIND / BA — not enforced in semantic-search code):** Hivemind split, `experimentInfo` stamping, FIND auth/key, FIND `Pagination.Total`, chip-override UI, search-to-bid telemetry.

