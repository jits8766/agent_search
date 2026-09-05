# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup
run `/aieml-onboard` skill from aieml-harness plugin.  
It installed gdx and sets up aws cli with profiles

Once the skill is run the env should have this profile. Use it for this project.
```sh
aws sts get-caller-identity  --profile=auctions-dev-private-ops
```

## What this is

A conversational discovery platform for domains and auctions. It turns natural-language
queries (e.g. "expiring tech .com under $500") into intent-classified, hybrid-ranked
results. A `uv` workspace with two packages:

- `packages/semantic-search` (`semantic_search`) — the FastAPI service + all search logic. Port **8085**.
- `packages/llm-core` (`llm_core`) — shared LLM provider, model registry, ranker, and clients.

Python is pinned to **>=3.12,<3.13** (torch compatibility).

## Commands

```bash
make install              # uv sync
make test                 # uv run pytest packages/   (works)
make run                  # uvicorn semantic_search.app:app on :8085 --reload
uv run ruff format packages/

# Single package / single test (use real paths — see Makefile caveat below):
uv run pytest packages/semantic-search/
uv run pytest packages/llm-core/
uv run pytest packages/semantic-search/tests/test_cache.py
uv run pytest packages/semantic-search/tests/test_cache.py::TestCacheClass::test_name -x
```

**Makefile caveat:** `make test-llm-core`, `make test-agent-search`, and `make lint`
reference stale paths (`packages/agent_search/`, `packages/llm_core/`) that do not exist
(actual dirs use hyphens: `semantic-search`, `llm-core`). Use the explicit `uv run`
commands above, or fix the Makefile, rather than trusting those targets.

`pytest` is configured with `asyncio_mode = auto` (no `@pytest.mark.asyncio` needed) and
`--import-mode=importlib`. The test suite runs offline/deterministic by default
(`qi.encoder.backend = hashing`, not fastembed).

`HF_HUB_OFFLINE=1` is required at runtime to block HuggingFace network downloads.

## Architecture

`registry.build_subsystems(config, llm_provider)` is the **sole composition root** — there
are no peer cross-imports between subsystems outside it. Layer dependency order:
`core` → `config` → subsystems → `signal_store` → `orchestrator` → `registry` → `app`.

Request flow (`SearchOrchestrator.search`), all under a single `POST /search` entrypoint:

1. **Intent classification** (`qi.QIEngine`) — cascade L0 regex → L1 semantic routing →
   L2 LLM (gated). Multi-intent queries optionally decomposed.
2. **Hybrid retrieval** (`retrieval.HybridRetriever`) — Qdrant vector + payload filters in
   one RPC; ClickHouse analytic fallback (`analytics.*`); 3-tier in-memory cache
   (exact / semantic / structured).
3. **Fusion & ranking** — `retrieval.RRFFuser` (Reciprocal Rank Fusion, k=60) →
   `retrieval.DeterministicRanker` (Layer-4) → optional `retrieval.eranker_client` (HTTP
   personalization re-rank).
4. **Output** — `surface.SearchSurface` typed envelope; `signal_store.*` async feedback +
   measurement; opt-in `history.*`.
5. **Resilience** (`resilience.*`) — per-backend circuit breakers, health registry, routing
   planner; always returns bounded results on backend failure.

`nl_to_sql/` and `analytics/` form the structured/analytic path: NL→SQL generation,
validation, security checks, and a ClickHouse materialized-view router with NL-SQL fallback.

`llm_core.LLMProvider` discovers the model universe at runtime via `client.models.list()`
(service config does not hardcode model IDs) and ranks models per task using explicit
`{capability, cost, latency}` weights. Pricing is in `llm_core/pricing.yaml`.

The single `/search` route fans out internally to hybrid / analytics / guidance — there are
no separate `/analytics`, `/qi/*`, or `/history/*` routes. Other endpoints: `/feedback`,
`/data-build/*`, `/healthz`, `/health`, `/capabilities`, `/cache/*`, `/resilience/health`,
`/measurement/*`. OpenAPI at `/openapi.json`, Swagger at `/docs`.

`POST /internal/l0_ground` is a test-harness-only utility route (used by
`reground_filters_four_way.py`): reconciles + live-inventory-grounds a client-supplied
`(query, identified)` pair via the same pipeline the qie_only `/search` branch uses. No DB
writes, read-only.

## Config

`semantic_search/config/base.yaml` is the base config, loaded via `config.loader.load_config`.
Pydantic models live in `config/models.py`, `config/analytics_models.py`,
`config/nl_to_sql_models.py`. Env vars override (e.g. `QDRANT_HOST`, `CLICKHOUSE_HOST`,
`CLICKHOUSE_SECURE`, `CLICKHOUSE_PORT`, `QDRANT_HTTPS`, `PRETRAINED_DIR`).

## Offline QA harness

`python -m semantic_search.offline_harness.reground_filters_four_way` drives L0
filter-extraction QA outside the test suite. Requires `:8085` (`make run`) —
grounding always goes through the live service.

4-arm comparison: `LLMJ` (offline LLM extract → `/internal/l0_ground`),
`QIE_Only_LLM` (`/search?qie_only_mode=true`), `Full_Search_LLM` (`/search`),
`Regex` (`L0RegexFilterExtractor` → `/internal/l0_ground`). Artifacts under
`output/reground_four_way/` (see [`docs/EVALUATION.md`](docs/EVALUATION.md)).

LLMJ model: `L0_GROUNDING_MODEL` or first discovered
`task_model_allowlists.l0_entity_extraction` primary (same as runtime).

```bash
# From auc-semantic-search/, with :8085 already running:
# Model = L0_GROUNDING_MODEL or first discovered allowlist primary (l0_entity_extraction).
python3 -m semantic_search.offline_harness.reground_filters_four_way --seed \
    --md test_search_queries.md --md test_filter_queries.md
python3 -m semantic_search.offline_harness.reground_filters_four_way --fail-only
python3 -m semantic_search.offline_harness.reground_filters_four_way --analysis-only
```

Env vars:

| Var | Required | Default | Purpose |
| --- | --- | --- | --- |
| `L0_GROUNDING_MODEL` | no | allowlist primary (`l0_entity_extraction`) | Offline LLM model for the `LLMJ` arm |
| `GOOGLE_API_KEY` | preferred for Gemini | from `.env` | Gemini via OpenAI-compat gateway |
| `OPENAI_API_KEY` | yes* (unless `--analysis-only`) | from `.env` | LLM auth / Gemini dual-path |
| `LLM_BASE_URL` | no | provider default | Override OpenAI-compatible base URL |
| `SEARCH_ENDPOINT` / `L0_COMPARE_ENDPOINT` | no | `http://localhost:8085/search` | Live `/search` target |
| `L0_GROUND_ENDPOINT` | no | derived from the above (`/internal/l0_ground`) | Live grounding target |
| `COMPARE_CONCURRENCY` | no | `2` (hard-capped at 2) | Queries in flight |
| `COMPARE_HTTP_INFLIGHT` | no | `1` | Concurrent HTTP calls to `:8085` |
| `COMPARE_PROGRESS_EVERY` | no | `25` | Progress log / checkpoint-write interval |

`POST /internal/l0_ground` (see Architecture above) is what makes uniform grounding
possible across all 4 arms without duplicating subsystem wiring.

Full phase-by-phase breakdown (Phase 1 filter accuracy, Phase 2 hybrid/explore/guidance/
analytics eval via `eval_test_search_queries.py`, Phase 3 status): [`docs/EVALUATION.md`](docs/EVALUATION.md).

## DevOps

See [DevOpsHandbook.md](./DevOpsHandbook.md) for deployment, logs, secrets, and ClickHouse testing.

## Local stack

`docker compose up -d` starts Qdrant + ClickHouse + the API (compose DNS resolves
`QDRANT_HOST`/`CLICKHOUSE_HOST`). Local ClickHouse stays on plain HTTP 8123; data persists in
named volumes (`docker compose down -v` to wipe).