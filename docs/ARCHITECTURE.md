# AUC Semantic Search Architecture

## What This Answers

- Services: API + Qdrant; ClickHouse optional via `clickhouse.enabled` (default `true`). `llm-core` ships inside the API image.
- Connectivity: clients → API → Qdrant / optional ClickHouse-Athena / LLM APIs / baked local models. If ClickHouse is off, CH-backed analytics/explore/price-band paths are disabled; Qdrant/in-memory paths remain.
- Technology: FastAPI/Python, Qdrant, ClickHouse, FastEmbed/hashing/sklearn/spaCy, OpenAI/Anthropic/Google (GoCaas), Docker, Katana ECS.
- Data: Qdrant collections, optional ClickHouse tables, Athena/S3 build paths, baked models, in-process caches, JSONL feedback.
- Security: Katana HTTPS/IAM/secrets plus app headers, rate limits, sanitizers, moderator, egress guard.
- Scalability: Katana ECS replicas/autoscaling, separated stores, external LLMs, timeouts, degradation.



## 1. Context Diagram

```mermaid
flowchart TB
    %% =============================================
    %% Styling
    %% =============================================
    classDef actors fill:#fefce8,stroke:#854d0e,stroke-width:3px,rx:12,ry:12;
    classDef katana fill:#f0f9ff,stroke:#1e40af,stroke-width:3px,rx:12,ry:12;
    classDef stores fill:#ecfdf5,stroke:#0f766e,stroke-width:3px,rx:12,ry:12;
    classDef ai fill:#f3e8ff,stroke:#6b21a8,stroke-width:3px,rx:12,ry:12;
    classDef build fill:#fef2f2,stroke:#b91c1c,stroke-width:2.5px,rx:10,ry:10;

    %% =============================================
    %% External Actors
    %% =============================================
    subgraph Actors["External Actors"]
        direction TB
        User[User / Client]
        Operator[Operator / Data Build User]
    end

    %% =============================================
    %% Katana Hosted System
    %% =============================================
    subgraph Katana["Katana-Hosted System"]
        direction TB
        Edge["Katana Edge<br/>LB + HTTPS Redirect"]
        API["AUC Semantic Search API<br/>FastAPI on ECS"]
        Ops["Katana Platform Services<br/>IAM • Secrets • Networking • Logging • Monitoring • Autoscaling"]
    end

    %% =============================================
    %% Search & Analytics Stores
    %% =============================================
    subgraph Stores["Search and Analytics Stores"]
        direction TB
        Qdrant[(Qdrant ECS Service)]
        ClickHouse[(ClickHouse ECS Service)]
        Athena[(Athena)]
    end

    %% =============================================
    %% AI & Model Dependencies
    %% =============================================
    subgraph AI["AI and Model Dependencies"]
        direction TB
        LLM[OpenAI / Anthropic / Google via GoCaas]
        Models[(Baked Pretrained Models)]
    end

    %% =============================================
    %% Build-Time Artifacts
    %% =============================================
    subgraph Build["Build-Time Artifact Source"]
        S3[S3 Pretrained Model Source]
    end

    %% =============================================
    %% Connections
    %% =============================================
    User -->|search • analytics • feedback • history| Edge
    Operator -->|data-build • health • measurement • resilience| Edge
    
    Edge -->|HTTPS App Traffic| API
    
    API -->|hybrid vector + payload search| Qdrant
    API -->|analytics queries + snapshots| ClickHouse
    API -->|classification / extraction / NL-SQL| LLM
    API -->|local model inference| Models
    
    S3 -.->|CI sync before Docker build| Models
    
    Ops -.->|platform controls| Edge
    Ops -.->|task role • secrets • logs • autoscale| API

    %% Apply styles
    class Actors actors
    class Katana katana
    class Stores stores
    class AI ai
    class Build build
```





## 2. Container Diagram

```mermaid
flowchart TB
    %% =============================================
    %% Styling
    %% =============================================
    classDef local fill:#fefce8,stroke:#854d0e,stroke-width:3px,rx:12,ry:12;
    classDef katana fill:#f0f9ff,stroke:#1e40af,stroke-width:3px,rx:12,ry:12;
    classDef external fill:#f3e8ff,stroke:#6b21a8,stroke-width:2.5px,rx:10,ry:10;

    %% =============================================
    %% Local Development
    %% =============================================
    subgraph DevLocal["Local Development\n(docker-compose via scripts/compose-up.sh)"]
        direction TB
        LocalAPI[agent-search API :8085]
        LocalQdrant[(Qdrant :6333 / :6334)]
        LocalClickHouse[(ClickHouse :8123 / :9000)]
        
        LocalAPI --> LocalQdrant
        LocalAPI -.->|clickhouse.enabled=true| LocalClickHouse
    end

    %% =============================================
    %% Katana Dev Environment
    %% =============================================
    subgraph KatanaDev["Katana dev-private Environment"]
        direction TB
        App[auc-semantic-search ECS App :8085]
        Qdrant[auc-semsearch-qdrant ECS Service :8443]
        ClickHouse[auc-semsearch-clickhouse ECS Service :8443]
        TaskRole[IAM Task Role]
        Secrets[Secrets / Env Vars]
        Logs[CloudWatch Logging]
        Autoscale["CPU/Memory Autoscale (70%)"]
        
        App --> Qdrant
        App -.->|clickhouse.enabled=true in CI| ClickHouse
        App --> TaskRole
        App --> Secrets
        App --> Logs
        Autoscale --> App
    end

    %% =============================================
    %% External Dependencies
    %% =============================================
    LLM[OpenAI / Anthropic / Google]
    Pretrained[(Pretrained Models)]

    %% =============================================
    %% Connections
    %% =============================================
    LocalAPI --> LLM
    App --> LLM
    LocalAPI --> Pretrained
    App --> Pretrained

    %% Apply styles
    class DevLocal local
    class KatanaDev katana
    class LLM,Pretrained external
```



Katana owns load balancers, subnets, HTTP-to-HTTPS redirect, logging, monitoring, security, networking, and IAM policy attachments. Repo files define/publish the ECS artifacts and app-level configuration.

## 3. Layered Architecture

```mermaid
flowchart TB
    %% Larger, cleaner subgraphs
    classDef layer fill:#f0f4f8,stroke:#1e40af,stroke-width:3px,rx:12,ry:12;
    classDef security fill:#fee2e2,stroke:#b91c1c,stroke-width:3px,rx:12,ry:12;
    classDef observability fill:#ecfdf5,stroke:#0f766e,stroke-width:3px,rx:12,ry:12;

    subgraph Business["Business Layer"]
        direction TB
        Personas[Users & Operators]
        Capabilities[Search • Guidance • Explore • Analytics • Feedback]
        ExternalBiz[LLM APIs • Athena • S3]
    end

    subgraph Application["Application Layer"]
        direction TB
        FastAPI[FastAPI Routes]
        Orchestrator[SearchOrchestrator]
        DataBuild[Data-build Endpoints]
        Registry[build_subsystems]
    end

    subgraph AI["AI Layer"]
        direction TB
        QI[QIEngine]
        Extract[Regex + LLM Entity Extraction]
        Router[Semantic Router + LLM Classifier]
        NLSQL[NL-to-SQL + Verifier]
        Eval[Measurement / Eval Hooks]
        AICache[QI + NL-SQL Caches]
    end

    subgraph Data["Data Layer"]
        direction TB
        QdrantData[(Qdrant Vectors + Payloads)]
        ClickHouseData[(ClickHouse Analytics)]
        AthenaData[Athena Read Paths]
        Signals[JSONL + In-memory Signals]
        Models[(Baked Models)]
    end

    subgraph Security["Security Layer"]
        direction TB
        HTTPS[HTTPS Ingress]
        Headers[Security Headers]
        RateLimit[Rate Limiting]
        Guardrails[Sanitizer • Moderator • Egress Guard]
        Scans[Semgrep • Container Scan • DAST]
    end

    subgraph Observability["Observability Layer"]
        direction TB
        Health["/healthz"]
        Measurement["/measurement/*"]
        Resilience["/resilience/health"]
        Logs[Structured Logs + CloudWatch]
    end

    subgraph Infra["Infrastructure Layer"]
        direction TB
        Katana[Katana ECS]
        Docker[Docker Images]
        TaskRole[IAM Task Role]
        Secrets[Secrets / Env Vars]
    end

    %% Flow connections
    Business --> Application
    Application --> AI
    AI --> Data

    Application --> Security
    Application --> Observability

    Infra --> Application
    Infra --> Data

    class Business,Application,AI,Data,Infra layer
    class Security security
    class Observability observability
```





## 4. Component Diagram

```mermaid
flowchart LR
    Routes[FastAPI route handlers]
    State[AppState]
    Registry[Subsystem registry]
    Orchestrator[SearchOrchestrator]

    subgraph QI[Query Intelligence]
        Splitter[MultiIntentSplitter]
        Regex[RegexEntityExtractor]
        LLMExtract[LLMEntityExtractor]
        Semantic[SemanticRouter]
        LLMClass[LLMClassifier]
        Residual[Residual extractor]
    end

    subgraph Search[Search + Ranking]
        Retriever[QdrantHybridRetriever]
        Fusion[RRF fusion]
        Diversifier[Diversifier / leanings / tie-break]
        ZeroGuard[ZeroResultGuard]
    end

    subgraph Analytics[Analytics]
        Router[AnalyticsRouter]
        SQL[NL-to-SQL pipeline]
        Verifier[Verifier]
        MV[MV router]
    end

    subgraph CrossCutting[Cross-cutting]
        Cache[In-process caches]
        Safety[Sanitizer moderator egress guard]
        Resilience[Circuit breaker health degradation]
        Signals[Feedback measurement signals]
        Workers[Vector delta event history retrain drivers]
        LLM[LLMProvider / LLMCallRouter]
    end

    Routes --> State --> Registry
    Routes --> Orchestrator
    Registry --> QI
    Registry --> Search
    Registry --> Analytics
    Registry --> CrossCutting

    Orchestrator --> QI
    Orchestrator --> Search
    Orchestrator --> Analytics
    Orchestrator --> CrossCutting

    LLMExtract --> LLM
    LLMClass --> LLM
    SQL --> LLM
```





## 5. Deployment Diagram

```mermaid
flowchart TB
    %% =============================================
    %% Styling
    %% =============================================
    classDef platform fill:#f0f9ff,stroke:#1e40af,stroke-width:3px,rx:12,ry:12;
    classDef service fill:#ecfdf5,stroke:#0f766e,stroke-width:3px,rx:12,ry:12;
    classDef aws fill:#fefce8,stroke:#854d0e,stroke-width:3px,rx:12,ry:12;
    classDef external fill:#f3e8ff,stroke:#6b21a8,stroke-width:2.5px,rx:10,ry:10;

    %% =============================================
    %% Katana Platform
    %% =============================================
    subgraph Katana["Katana dev-private Platform"]
        direction TB
        LB["Katana Load Balancer<br/>Subnets + HTTPS Redirect"]
        TG["HTTPS Target Group<br/>/healthz"]
        Auto["Autoscale<br/>CPU/Memory 70%"]
        CW[CloudWatch Logging]
        Net["Networking • Security • Monitoring"]
    end

    %% =============================================
    %% API Service
    %% =============================================
    subgraph APIService["auc-semantic-search ECS Service"]
        direction TB
        APIContainer["API Container<br/>2 vCPU / 4 GB RAM"]
        AppCode[semantic_search + llm_core]
        AppModels["/app/pretrained"]
    end

    %% =============================================
    %% Search Store
    %% =============================================
    subgraph SearchStore["auc-semsearch-qdrant ECS"]
        Qdrant["Qdrant HTTPS :8443"]
    end

    %% =============================================
    %% Analytics Store
    %% =============================================
    subgraph AnalyticsStore["auc-semsearch-clickhouse ECS"]
        ClickHouse["ClickHouse HTTPS :8443"]
    end

    %% =============================================
    %% AWS Access
    %% =============================================
    subgraph AWS["AWS Access via Task Role"]
        direction TB
        IAM["IAM Policies<br/>S3 • Athena • Glue"]
        S3[S3 Pretrained Artifacts]
        Athena[Athena]
    end

    %% =============================================
    %% External / Shared
    %% =============================================
    Secrets["Katana Secrets<br/>LLM • Qdrant • ClickHouse"]
    LLM[OpenAI / Anthropic / Google]

    %% =============================================
    %% Connections
    %% =============================================
    LB --> TG
    TG --> APIContainer
    Auto --> APIContainer
    APIContainer --> CW
    APIContainer --> Net
    APIContainer --> Secrets
    APIContainer --> Qdrant
    APIContainer --> ClickHouse
    APIContainer --> IAM
    APIContainer --> LLM
    APIContainer --> AppCode
    APIContainer --> AppModels

    IAM --> S3
    IAM --> Athena

    %% Apply styles
    class Katana platform
    class APIService,SearchStore,AnalyticsStore service
    class AWS aws
    class Secrets,LLM external
```





## 6. Data Flow

```mermaid
flowchart TB
    classDef request fill:#fefce8,stroke:#854d0e,stroke-width:2.5px,rx:10,ry:10;
    classDef app fill:#f0f9ff,stroke:#1e40af,stroke-width:2.5px,rx:10,ry:10;
    classDef path fill:#ecfdf5,stroke:#0f766e,stroke-width:2.5px,rx:10,ry:10;
    classDef store fill:#f3e8ff,stroke:#6b21a8,stroke-width:2.5px,rx:10,ry:10;
    classDef refresh fill:#fff7ed,stroke:#c2410c,stroke-width:2.5px,rx:10,ry:10;

    subgraph RequestFlow["Online request flow"]
        Query["Raw Query"] --> Guard["Ingress Security<br/>Sanitize • Rate Limit • Validate"]
        Guard --> QI["QI Engine<br/>Classify + Extract Entities"]
        QI --> Intent{Intent}

        Intent --> HybridFirst["Hybrid-first retrieve<br/>Qdrant + filters"]
        HybridFirst --> Complement{"ClickHouse up?"}
        Complement -->|yes| Merge["Rails RRF merge<br/>analytics / guidance envelope"]
        Complement -->|no| Strip["Temporal strip<br/>nonempty ladder"]
        Merge --> Rank["Rank + Diversify + rank_leanings + Tie-break"]
        Strip --> Rank
        Rank --> Response["Response Envelope<br/>ranked_results hybrid-based; sibling blocks unchanged"]
        Response --> Observe["Logs • Measurement • Feedback"]
    end

    subgraph Stores["Runtime stores"]
        Qdrant[(Qdrant<br/>vectors + payloads)]
        ClickHouse[(ClickHouse<br/>snapshots, events, MVs)]
        Cache[(In-process caches)]
        LLM[LLM APIs<br/>when configured]
    end

    subgraph RefreshFlow["Build / refresh flow"]
        Athena[Athena<br/>source reads]
        Seed["Data-build seed / backfill"]
        Delta["Delta refresh driver"]
        Event["Event ingest driver"]
    end

    QI --> LLM
    QI --> Cache
    HybridFirst --> Qdrant
    HybridFirst --> Cache
    Merge --> ClickHouse

    Athena --> Seed
    Athena --> Delta
    Athena --> Event
    Seed --> Qdrant
    Seed --> ClickHouse
    Delta --> Qdrant
    Delta --> ClickHouse
    Event --> ClickHouse
    Event --> Qdrant

    class Query,Guard request
    class QI,Intent,Rank,Response,Observe app
    class HybridFirst,Complement,Merge,Strip path
    class Qdrant,ClickHouse,Cache,LLM store
    class Athena,Seed,Delta,Event refresh
```





## 7. Dependency Diagram

```mermaid
flowchart TB
    classDef entry fill:#fefce8,stroke:#854d0e,stroke-width:2.5px,rx:10,ry:10;
    classDef core fill:#f0f9ff,stroke:#1e40af,stroke-width:2.5px,rx:10,ry:10;
    classDef feature fill:#ecfdf5,stroke:#0f766e,stroke-width:2.5px,rx:10,ry:10;
    classDef shared fill:#f3e8ff,stroke:#6b21a8,stroke-width:2.5px,rx:10,ry:10;
    classDef external fill:#fff7ed,stroke:#c2410c,stroke-width:2.5px,rx:10,ry:10;

    App["semantic_search.app<br/>FastAPI routes + lifespan"]
    Registry["semantic_search.registry<br/>build_subsystems runtime wiring"]
    Contracts["semantic_search.contracts<br/>shared DTOs + enums"]
    Orchestrator["semantic_search.orchestrator<br/>request workflow"]

    subgraph RequestDeps["Request-time feature packages"]
        QI["semantic_search.qi<br/>intent + entities"]
        Retrieval["semantic_search.retrieval<br/>Qdrant retrieval"]
        Analytics["semantic_search.analytics<br/>ClickHouse analytics"]
        NLSQL["semantic_search.nl_to_sql<br/>SQL generation + verification"]
        Cache["semantic_search.cache<br/>exact / structured / intent-plan"]
        Safety["semantic_search.safety<br/>sanitize + egress guard"]
        Resilience["semantic_search.resilience<br/>health + degradation"]
    end

    subgraph BuildDeps["Build / refresh packages"]
        Vectorization["semantic_search.vectorization<br/>indexing + refresh drivers"]
        Explore["semantic_search.explore<br/>seed + ClickHouse mirror"]
    end

    subgraph IntegrationDeps["Integration clients"]
        LLMCore["llm_core<br/>LLM provider + router support"]
        Stores["Qdrant / ClickHouse / Athena / S3"]
    end

    App --> Registry
    App --> Orchestrator
    App --> Contracts

    Registry --> QI
    Registry --> Retrieval
    Registry --> Analytics
    Registry --> NLSQL
    Registry --> Cache
    Registry --> Safety
    Registry --> Resilience
    Registry --> Vectorization
    Registry --> Explore
    Registry --> LLMCore

    Orchestrator --> QI
    Orchestrator --> Retrieval
    Orchestrator --> Analytics
    Orchestrator --> Cache
    Orchestrator --> Safety
    Orchestrator --> Resilience
    Orchestrator --> Contracts

    Analytics --> NLSQL
    QI --> LLMCore
    NLSQL --> LLMCore
    Retrieval --> Stores
    Analytics --> Stores
    Vectorization --> Stores
    Explore --> Stores

    class App entry
    class Registry,Orchestrator,Contracts core
    class QI,Retrieval,Analytics,NLSQL,Vectorization,Explore feature
    class Cache,Safety,Resilience shared
    class LLMCore,Stores external
```





## Component Responsibility Map

```mermaid
flowchart TB
    API[FastAPI API<br/>HTTP boundary]
    Registry[Subsystem registry<br/>runtime graph]
    Orchestrator[SearchOrchestrator<br/>request workflow]
    QI[QIEngine<br/>intent + entities]
    Retrieval[Retrieval<br/>Qdrant + fusion]
    Analytics[Analytics<br/>ClickHouse + NL-to-SQL]
    Workers[Workers<br/>seed refresh delta event history retrain]
    Security[Security<br/>headers rate limit guardrails egress]
    Observability[Observability<br/>health measurement logs signals]
    LLM[LLM boundary<br/>provider router clients]

    API --> Orchestrator
    API --> Workers
    Registry --> Orchestrator
    Registry --> QI
    Registry --> Retrieval
    Registry --> Analytics
    Registry --> Workers
    Registry --> Security
    Registry --> Observability
    Registry --> LLM
    Orchestrator --> QI
    Orchestrator --> Retrieval
    Orchestrator --> Analytics
```





## Notes

- Redis is not an active runtime service in the base config. The code has disabled remote cache mirrors for search payloads and NL-to-SQL exact cache. If enabled later, the API would need a reachable Redis/Dragonfly endpoint via env var; PostgreSQL is not involved.
- Background work is implemented as in-process async drivers. No separate queue service is part of the current architecture.
- Katana owns platform concerns: load balancer, subnets, HTTPS redirect, logging, monitoring, security, networking, autoscaling, and IAM policy attachments.

