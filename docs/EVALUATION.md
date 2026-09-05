# Offline Evaluation Harnesses

Offline QA scripts, run against a live `:8085` (`make run`). Location:
`packages/semantic-search/semantic_search/offline_harness/`, run via
`python -m semantic_search.offline_harness.<script>` or `POST /internal/harness/*` (§4).
Output: `auc-semantic-search/output/` (gitignored).

**LLMJ model:** env `L0_GROUNDING_MODEL` if set; else first model in
`model_selection_strategy.task_model_allowlists.l0_entity_extraction` that
GoCaas discovery lists (same rule as runtime `LLMProvider.get_primary_model`).
HTTP `POST /internal/harness/reground-four-way` injects that primary when the
`grounding_model` form field is omitted. Gemini auth: `GOOGLE_API_KEY` preferred,
else `OPENAI_API_KEY` (OpenAI-compat / `LLM_BASE_URL`).

## 1. `reground_filters_four_way.py`

4-arm L0 filter-extraction compare, all grounded through the same live pipeline:
`LLMJ` (offline LLM extract, grounded via `/internal/l0_ground`), `QIE_Only_LLM` (live
`/search?qie_only_mode=true`), `Full_Search_LLM` (live `/search`), `Regex` (offline
`L0RegexFilterExtractor`, grounded via `/internal/l0_ground`). Row status
`OK`/`DIFF`/`ERROR`, pairwise exact-match matrix (`C(4,2)=6` pairs), per-arm gap +
grounding-drop breakdown.

```bash
python -m semantic_search.offline_harness.reground_filters_four_way --seed \
    --md test_search_queries.md --md test_filter_queries.md
curl -sS -X POST http://localhost:8085/internal/harness/reground-four-way -F seed=true -F limit=50

# Rebuild xlsx + sheet JSON (incl. holdout) from existing results — no HTTP/LLM:
python -m semantic_search.offline_harness.reground_filters_four_way --analysis-only \
    --results output/reground_four_way/results_YYYYMMDDTHHMMSSZ.json \
    --holdout-frac 0.2 --holdout-seed 42
```

| Arg / Form | Default | Purpose |
| --- | --- | --- |
| `--seed` / `seed` | `true` | Build fresh from `--md` |
| `--md` (repeatable) / `md` | both suites | Query suite(s) |
| `--fail-only` / `fail_only` | `false` | Re-run non-`OK` rows |
| `--analysis-only` / `analysis_only` | `false` | Rebuild report only, no HTTP/LLM |
| `--holdout-frac` / `holdout_frac` | `0.2` | Formal train/test holdout for pairwise metrics (`0` disables); stratified by suite |
| `--holdout-seed` / `holdout_seed` | `42` | Deterministic holdout split seed |
| env `L0_GROUNDING_MODEL` | unset → allowlist primary | Override LLMJ model |
| env `GOOGLE_API_KEY` / `OPENAI_API_KEY` (required*) | from `.env` | LLM auth (Gemini prefers Google) |
| env `LLM_BASE_URL` | from config / `.env` | OpenAI-compat gateway |

\* not required for `--analysis-only` / `--seed`-only.

### Output (`output/reground_four_way/`)

| Artifact | Role |
| --- | --- |
| `results_*.json` | `per_query` source for `--analysis-only` |
| `pairwise_summary_*.json` | Pairwise matrix + nested `"holdout"` |
| `missing_extra_vs_llmj_*.json` | Sheet-wise JSON |
| `keyword_overlap_*.json` | Sheet-wise JSON |
| `grounding_drops_*.json` | Sheet-wise JSON |
| `reground_*.xlsx` | Workbook; HOLDOUT block on sheet `pairwise_summary` |

No separate `holdout_*.json` / `holdout_pairwise` sheet. `per_query` columns unchanged.

### Holdout (reground)

- **Where:** xlsx sheet `pairwise_summary` (HOLDOUT section at bottom); JSON key
  `pairwise_summary_*.json` → `"holdout"`.
- **Focus pairs (prod capability only):** `QIE_Only_LLM` vs `Regex`,
  `Full_Search_LLM` vs `Regex`. `LLMJ` vs `Regex` is **not** a holdout focus
  (LLMJ = offline harness reference arm).
- **Metric:** arm filter-set agreement (not gold-label accuracy).
- **Full_Search headline:** Exact+gnd+pipe+soft (Prod exact alone undercounts —
  grounding / pipeline `not_applied` / soft-downgrade drops). Columns also include
  Exact+gnd% and Exact+gnd+pipe%.
- **QIE headline:** Prod exact (no pipeline/soft drops on that path).

## 2. `eval_test_search_queries.py`

Eval over `HYBRID`/`EXPLORE`/`GUIDANCE`/`ANALYTICS` suites — routing accuracy, SLA,
`NDCG@k`/precision/recall/F1. Full-search `/search` path only (not arm pairwise).

```bash
python -m semantic_search.offline_harness.eval_test_search_queries http://localhost:8085 --suite HYBRID
# multi-minute (240 q/suite @ ~0.8 qps) — long client timeout
curl -sS -X POST http://localhost:8085/internal/harness/eval-search -F api=http://127.0.0.1:8085 -F suite=HYBRID

# Rebuild sheet JSON + HOLDOUT on suite_summary xlsx from results — no HTTP:
python -m semantic_search.offline_harness.eval_test_search_queries --analysis-only \
    --results output/eval_search_queries/results_XXXX.json \
    --holdout-frac 0.2 --holdout-seed 42
```

| Arg / Form | Default | Purpose |
| --- | --- | --- |
| positional / `--api` / `api` | required at CLI; `http://127.0.0.1:8085` over HTTP | API base URL |
| `--suite` / `suite` | all four | Comma-separated suite filter |
| `--analysis-only` | `false` | Rebuild sheet JSON + HOLDOUT on existing xlsx; no HTTP |
| `--results` | latest under out dir | `results_*.json` path (with `--analysis-only`) |
| `--holdout-frac` | `0.2` | Train/test holdout fraction (`0` disables); stratified by suite |
| `--holdout-seed` | `42` | Deterministic holdout split seed |

### Output (`output/eval_search_queries/`)

| Artifact | Role |
| --- | --- |
| `results_*.json` | `queries` sheet; analysis-only source |
| `suite_summary_*.json` | Suite rollup + nested `"holdout"` |
| `section_summary_*.json` | Sheet-wise JSON |
| `fail_only_*.json` | Sheet-wise JSON |
| `misroutes_*.json` | Sheet-wise JSON |
| `eval_report_*.{md,xlsx}` | Report; HOLDOUT block on sheet `suite_summary` |

No separate `holdout_summary_*.json` / holdout sheet. `queries` columns unchanged.

### Holdout (eval)

- **Where:** xlsx sheet `suite_summary` (HOLDOUT section below suite rows); JSON key
  `suite_summary_*.json` → `"holdout"`.
- **What:** train/test suite metrics (routing / NDCG@5 / SLA / latency) for the
  full-search eval — not LLM vs regex arm agreement.

## 3. HTTP routes

Each spawns the script as a subprocess (never in-process `asyncio.run` — would raise
inside uvicorn's loop); returns artifact paths + inlined summary. All multi-minute.

**Auth / SSRF:**
- Local: open unless `HARNESS_API_KEY` is set (then send `X-Harness-Key`).
- Katana: `HARNESS_ENABLED` defaults on; set `HARNESS_API_KEY` and send
  `X-Harness-Key` on `/internal/harness/*` and `/internal/l0_ground`.
  Missing `HARNESS_API_KEY` on Katana → 403.
- `HARNESS_ENABLED=false` disables routes everywhere.
- `eval-search` `api`: loopback only (`127.0.0.1` / `localhost` / `::1`).
- Artifact `path`: under `output/` only (not config/source).

| Route | Key response fields |
| --- | --- |
| `POST /internal/harness/reground-four-way` | `results_json_path`, `analysis_md_path`, `analysis_markdown`, `xlsx_path`, `log_path`, `grounding_model` (+ `holdout_frac` / `holdout_seed` / optional `grounding_model` form) |
| `POST /internal/harness/eval-search` | `report_md_path`, `report_markdown`, `xlsx_path`, `log_path` |
| `GET /internal/harness/artifact?path=...` | Streams file under `output/` only |

Full logs: `output/harness_runs/{name}_{unix_ts}.log`.

## References

`PLAN.md` §2 · `PROCESS.md` §1 · `../CLAUDE.md` (`/internal/l0_ground`) ·
`packages/semantic-search/semantic_search/offline_harness/` ·
`config/base.yaml` (`task_model_allowlists`, `llm_api_keys`) ·
`config/offline_harness_defaults.py` (resolver; no hardcoded model id)
