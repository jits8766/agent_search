# Infrastructure and Deployment

## 1. Runtime Topology

```mermaid
flowchart LR
    Client[Client / caller] --> TG[Katana HTTPS target group<br/>public API entry]
    TG --> API[FastAPI service<br/>search + QI orchestration<br/>HTTPS :8085]
    API --> Q[Qdrant service<br/>vectors + payload filters<br/>internal HTTPS]
    API --> CH[ClickHouse service<br/>analytics + event views<br/>optional]
    API --> LLM[LLM provider<br/>classifier/entity/SQL support<br/>env configured]
```



Evidence: `configs/katana*.yaml`, `docker-compose*.yaml`, `base.yaml`.

## 2. Deployment Shape

```mermaid
flowchart LR
    GH[GitHub Actions<br/>push/manual trigger] --> Build[build + test + scan<br/>API/store images]
    Build --> Pub[Katana publish<br/>artifact upload]
    Pub --> Promote[Katana promote<br/>target environment]
    Promote --> ECS[Katana ECS runtime<br/>service deployed]
```



Evidence: `.github/workflows/*.yml`, `.github/workflows/config.json`, `configs/katana*.yaml`.

## 3. Service Runtime

```mermaid
flowchart LR
    IMG[API image<br/>Chainguard Python nonroot] --> Model["/app/pretrained<br/>baked model artifacts"]
    IMG --> TLS["/app/tls<br/>runtime cert path"]
    TLS --> UV[uvicorn<br/>HTTPS :8085]
    ENV[Katana env/secrets<br/>hosts + keys] --> App[FastAPI app<br/>runtime config]
```



Evidence: `packages/semantic-search/Dockerfile`, `tls_entrypoint.py`, `configs/katana.yaml`.

## 4. Local Runtime

```mermaid
flowchart LR
    Compose[docker compose<br/>local launcher] --> API[API container<br/>localhost :8085]
    API --> Q[Qdrant container<br/>vector store]
    API -. analytics profile .-> CH[ClickHouse container<br/>analytics enabled]
```



Evidence: `scripts/compose-up.sh`, `scripts/read_clickhouse_enabled.py`, `docker-compose.yaml`, `docker-compose.analytics.yaml`.

## 5. Scalability

```mermaid
flowchart LR
    LB[Katana load balancer<br/>HTTPS target group<br/>health-routed traffic]
    LB --> Scale[API autoscale reference<br/>CPU or memory at 70%]
    Scale --> Tasks[API task pool<br/>up to 10 replicas<br/>2 nCPU + 4GB RAM each]
    Tasks --> Q[Qdrant store<br/>semantic vectors + filters]
    Tasks --> CH[ClickHouse optional<br/>analytics + event views]
```



Evidence: `configs/katana.yaml`, `configs/katana-qdrant.yaml`, `configs/katana-clickhouse.yaml`, `ARCHITECTURE.md`, `plan_agentic_search.md`.

## 6. Networking and Discovery

```mermaid
flowchart LR
    Internet[External caller] --> APIHost[API ingress enabled<br/>Katana public host]
    APIHost --> API[API HTTPS :8085<br/>public service]
    API --> QH[QDRANT_HOST<br/>env-provided endpoint]
    API --> CHH[CLICKHOUSE_HOST<br/>env-provided endpoint]
    QH --> Q[Qdrant<br/>ingress disabled]
    CHH --> CH[ClickHouse<br/>ingress disabled]
```



Evidence: `configs/katana*.yaml`, `docker-compose.yaml`, `base.yaml`.

## 7. Configuration and Secrets

```mermaid
flowchart LR
    Base[base.yaml<br/>runtime behavior] --> Runtime[timeouts, retries<br/>rate limits, store flags]
    Katana[Katana env/secrets<br/>runtime injection] --> Hosts[Qdrant/ClickHouse/LLM<br/>hosts, passwords, keys<br/>OPENAI/ANTHROPIC/GOOGLE]
    GH[config.json<br/>CI account map] --> Roles[deploy/task roles<br/>publish/promote access]
```



Evidence: `base.yaml`, `config/clickhouse_lever.py`, `configs/katana.yaml`, `.github/workflows/config.json`.

Batch ingest schedule: `vectorization.seed.schedule` and
`.github/workflows/data-ingest.yml` (daily cron + manual dispatch).
Environment: `SEED_SCHEDULE_RUN_ON_DEPLOY`, `SEED_SCHEDULE_INTERVAL_HOURS`,
`SEED_SCHEDULE_RUN_AT_HOUR_UTC` (Katana or local `.env`; restart the task to
apply). Effective values: `GET /data-build/status` → `seed.schedule`.

## 8. Security

```mermaid
flowchart LR
    IAM[CI IAM roles<br/>deploy permissions] --> Deploy[deploy + task role<br/>Katana artifact options]
    TLS[Runtime TLS<br/>self-signed generated] --> API[API cert<br/>created at startup]
    TLS --> Q[Qdrant cert<br/>container entrypoint]
    TLS --> CH[ClickHouse cert<br/>container entrypoint]
    App[Application controls<br/>request protection] --> Guard[rate limits<br/>security headers<br/>sanitizers]
```



Evidence: `tls_entrypoint.py`, `docker/qdrant/entrypoint-tls.sh`, `docker/clickhouse/entrypoint-tls.sh`, `clickhouse-config/tls.xml`, `base.yaml`.

## 9. Observability Ownership

```mermaid
flowchart LR
    subgraph App[Application-owned signals]
      Logs[structured logs<br/>intent, errors, latency]
      Metrics[response metrics<br/>retrieval_metrics + thresholds]
      Trace[pipeline trace<br/>route + fallback path]
      Resilience[resilience health<br/>/resilience/health]
    end

    subgraph Platform[Platform-owned checks]
      Health[health checks<br/>/healthz and /ping]
      LB[target routing<br/>healthy target only]
      RuntimeLogs[container logs<br/>runtime stdout/stderr]
    end

    subgraph Stores[Store-owned signals]
      Q[Qdrant<br/>health + service logs]
      CH[ClickHouse<br/>ping + logs + views]
    end

    App --> Platform --> Stores
```



Evidence: `configs/katana*.yaml`, `base.yaml`, `docker-compose.yaml`.

## 10. Resilience

```mermaid
flowchart LR
    Timeout[Timeout budgets<br/>request guardrails] --> Search[search 10s<br/>online path]
    Timeout --> Analytics[analytics 20s<br/>question path]
    Retry[Retry policy<br/>bounded attempts] --> LLM[LLM/backend retry<br/>limited retries]
    Health[Backend health<br/>dependency status] --> Degrade[degraded routing<br/>vector -> structured -> cache]
    Cost[LLM cost budgets<br/>per-query + fleet] --> Regex[QI → regex L0 + L1<br/>no query reject]
```



**Cost controls:** per call + hour/day USD, memory|redis. Breach → LLM skipped, search continues via regex/L1. Analytics → `cost_budget_exceeded`. Rates: `packages/llm-core/llm_core/pricing.yaml`.

Evidence: `base.yaml`, `cost/query_budget.py`, `cost/fleet_budget.py`. ClickHouse optional: toggle off, skip deploy, or runtime unavailability. 

## 11. Storage

```mermaid
flowchart LR
    QV[(qdrant_data<br/>local volume)] --> Q["/qdrant/storage<br/>vector persistence"]
    CV[(clickhouse_data<br/>local volume)] --> CH["/var/lib/clickhouse<br/>analytics persistence"]
    APIW["/app/output + /app/tls<br/>writable runtime paths"] --> API[API container<br/>outputs + certs]
    PRE["/app/pretrained<br/>read-only model path"] --> IMG[image-baked files<br/>offline startup]
```



Evidence: `docker-compose.yaml`, `configs/katana-qdrant.yaml`, `configs/katana-clickhouse.yaml`, `packages/semantic-search/Dockerfile`.