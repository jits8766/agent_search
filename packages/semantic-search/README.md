# Semantic Search

FastAPI service for conversational auction-domain discovery. It classifies natural-language queries, extracts structured filters, routes requests to search, explore, guidance, or analytics paths, and returns ranked domain results or analytics output.

**Port:** 8085
**Package:** `semantic-search`
**Service name:** `semantic_search`

## Problem & Use Case

Users can describe domain-auction intent in natural language, such as `expiring tech .com under $500`. The service converts the query into typed intent and filters, retrieves candidates from vector and structured backends, applies ranking and fallback logic, and returns a consistent API envelope.

## Process Flow

1. **Query Intelligence** - Applies sanitization, entity extraction, semantic routing, optional LLM classification, multi-intent handling, spell correction, and grounding.
2. **Hybrid-first ranked_results** - Every QI type (`hybrid` / `analytics` / `explore` / `guidance`) retrieves via hybrid (vector, sparse, structured, optional SQL). Config: `general.search.ranked_results_complement`.
3. **Complement substrates** - ClickHouse up: analytics fills the `analytics` block; explore rails RRF-merge into hybrid ranks; guidance attaches a snapshot envelope. When those substrates succeed, `rank_leanings` may reorder `ranked_results` from per-TLD aggregates; sibling `analytics` / `guidance` JSON blocks stay unchanged. ClickHouse down: skip CH substrates; strip temporal filters; hybrid (+ force-semantic nonempty ladder) still fills `ranked_results`.
4. **Fusion, Ranking, and Guards** - RRF fuse, diversity, complement rank leanings (analytics/guidance), auction tie-break, optional eRanker, zero-result guard, `ensure_nonempty` / `force_semantic_when_empty`.
5. **Signals and Observability** - Records feedback, measurement observations, proxy KPI signals, cache stats, backend health, and data-build status. In-API Athena pollers (`delta_refresh` / `event_ingest` / `qdrant_enrich`) are **off by default**; enable in config and restart when Athena path is ready.



## Architecture

`semantic_search.app` defines the FastAPI application and startup lifecycle. Startup loads `.env`, loads `config/base.yaml`, initializes the optional LLM provider, validates pretrained model availability, builds subsystems through `registry.build_subsystems(config_dict, llm_provider)`, prewarms selected components, starts configured background drivers, and exposes API routes.

`registry.build_subsystems` is the composition root. It validates `AgentSearchConfig`, builds encoders, QI components, retrieval backends, caches, signal stores, resilience primitives, analytics services, history components, and the `SearchOrchestrator`.

`SearchOrchestrator.search(...)` is the library entrypoint for the main search pipeline. It returns ranked results, eRanker outcome, and zero-result guard outcome.

## How to Use



### Service

```bash
uv pip install -r packages/semantic-search/requirements.txt
uvicorn semantic_search.app:app --host 0.0.0.0 --port 8085 --reload
```

```bash
curl -X POST http://localhost:8085/search \
  -H 'X-Session-Id: s1' \
  -F 'query=expiring tech .com under $500' \
  -F 'top_k=10'
```



### Local Docker Stack

Prefer the lever-aware script (reads `clickhouse.enabled` from `base.yaml`):

```bash
# from repo root (auc-semantic-search/)
scripts/compose-up.sh
```

| `clickhouse.enabled` | What starts |
| -------------------- | ----------- |
| `true` (default)     | Qdrant + ClickHouse + API (waits for CH healthy) |
| `false`              | Qdrant + API only |

Manual Compose (bypasses the lever):

```bash
docker compose up -d                              # Qdrant + API
docker compose --profile analytics up -d          # + ClickHouse
docker compose -f docker-compose.yaml \
  -f docker-compose.analytics.yaml --profile analytics up -d
```

Ports: API `8085`; Qdrant `6333/6334`; ClickHouse `8123/9000` (when started).

### Library

```python
import asyncio

from semantic_search.config.loader import load_config
from semantic_search.registry import build_subsystems


async def main() -> None:
    config_dict = load_config()
    subsystems = build_subsystems(config_dict, llm_provider=None)
    ranked, eranker, guard = await subsystems.orchestrator.search(
        raw_query="expiring .com",
        request_id=None,
        user_context=None,
    )


asyncio.run(main())
```



## API Endpoints

OpenAPI: `GET /openapi.json`
Swagger UI: `GET /docs`
ReDoc: `GET /redoc`


| Method | Endpoint                         | Purpose                                                                     |
| ------ | -------------------------------- | --------------------------------------------------------------------------- |
| GET    | `/`                              | Redirects to `/docs`                                                        |
| GET    | `/healthz`                       | Liveness response                                                           |
| GET    | `/health`                        | Same liveness response as `/healthz`                                        |
| GET    | `/capabilities`                  | Reports configured subsystem capabilities                                   |
| POST   | `/search`                        | Unified query endpoint. Form `qie_only_mode=true` runs L0 filter extract + grounding only (no L1/L2, no retrieval). Full mode routes hybrid / explore / guidance / analytics |
| POST   | `/feedback`                      | Records UAT feedback comments                                               |
| GET    | `/feedback`                      | Downloads feedback CSV for a date range                                     |
| GET    | `/cache/stats`                   | Returns per-tier cache counters                                             |
| POST   | `/cache/clear`                   | Clears orchestrator cache tiers                                             |
| GET    | `/resilience/health`             | Returns backend health and analytics substrate status                       |
| GET    | `/measurement/observations`      | Returns recent measurement observations                                     |
| GET    | `/measurement/signals`           | Returns proxy KPI report                                                    |
| POST   | `/data-build/seed`               | Builds and indexes the Qdrant/search corpus from source data                |
| POST   | `/data-build/analytics-backfill` | Backfills ClickHouse analytics data                                         |
| POST   | `/data-build/full`               | Runs Qdrant seed and ClickHouse backfill together                           |
| GET    | `/data-build/status`             | Reports Qdrant, ClickHouse, vectorization, and column availability status   |




## Configuration

Primary config file: `packages/semantic-search/semantic_search/config/base.yaml`.

Typed root: `AgentSearchConfig.from_dict(...)`.

Key config areas:

- `clickhouse` - **master lever** (`clickhouse.enabled`). When `false`, boot forces analytics / explore CH rails / price-band adapter off; CI skips ClickHouse deploy; `scripts/compose-up.sh` omits the ClickHouse service. Nested feature flags may stay `true` in YAML — the lever wins at boot. Same posture when CH is undeployed or unreachable: hybrid-first `ranked_results` keep serving; analytics / rail merge / guidance snapshots / `rank_leanings` skip (temporal strip + nonempty ladder via `ranked_results_complement`).
- `general` - service identity, query limits, rate limits, result fields, timeout budgets, search behavior (`qie_only_mode` default for Phase-1 filter-only).
- `qi` - query types, regex/entity extraction, semantic router, spell correction, query transformer, optional LLM classifier.
- `retrieval` - vector, structured, SQL, fusion, diversity, eRanker, and metric settings.
- `cache` - cache enablement and tier behavior.
- `nl_to_sql` - ClickHouse, Athena, SQL generation, execution, and verification settings (gated by `clickhouse.enabled`).
- `explore.clickhouse_rails` - CH-backed explore rails (gated by `clickhouse.enabled`).
- `resilience` - circuit breaker, backend health, and degradation planning.
- `measurement` - observation windows and proxy KPI evaluation.
- `feedback` - feedback signal storage and UAT feedback constraints.
- `history` - optional user history and compaction behavior.
- `vectorization` - seed, indexing, analytics backfill, pretrained model paths, and refresh drivers.
- `llm_*` and `model_*` - LLM provider, routing, capability, pricing, and model-selection settings.

`.env` is loaded before config use by walking parent directories from the module path, then checking the current working directory, then `/app/.env`.

Pretrained models are expected under `${LOCAL_PRETRAINED_DIR:-/app/pretrained}` in containerized runtime. The Docker image copies `pretrained/` into `/app/pretrained` and sets `HF_HUB_OFFLINE=1`.

## Implementation Status


| Capability               | Status                                                                                                             |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| Query intelligence       | Implemented: sanitization, entity extraction, semantic routing, optional LLM tier, grounding, multi-intent support |
| Hybrid retrieval         | Implemented: vector, sparse, structured, SQL-assisted paths, cache tiers, RRF fusion                               |
| Analytics / explore / guidance | Complement hybrid ranks (`ranked_results_complement`). CH up: analytics block + rail merge + guidance envelope; `rank_leanings` may reorder `ranked_results` from those substrates while sibling `analytics` / `guidance` blocks stay unchanged. CH down: hybrid retrieve with temporal strip + nonempty ladder |
| `qie_only_mode`          | Implemented: L0 extract + ground only; no Qdrant/ClickHouse required |
| Ranking                  | Implemented: diversity, complement `rank_leanings`, auction tie-break, zero-result guard, optional eRanker         |
| Feedback and measurement | Implemented: feedback endpoints, signal store, measurement store, proxy KPI reports                                |
| Resilience               | Implemented: health registry, circuit breaker, degradation planner; LLM per-query + fleet USD caps → regex/L1 degrade |
| History                  | Implemented in configuration and subsystem wiring when enabled                                                     |
| Deployment               | Implemented through Dockerfile, Docker Compose (`scripts/compose-up.sh` + `clickhouse.enabled`), and Katana manifests; CI ClickHouse job gated by the same lever |




## Testing

```bash
pytest packages/semantic-search/tests -v
```

The test suite includes coverage for QI, entity extraction, grounding, retrieval, Qdrant adapter behavior, analytics, cache behavior, feedback endpoints, vectorization, URL/security guards, reranking, diversification, history, and related contracts.

## License & Ownership

Internal auctions discovery platform.