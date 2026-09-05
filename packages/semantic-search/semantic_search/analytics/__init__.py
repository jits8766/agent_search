"""Real-time analytics path: ClickHouse client + MV-aware router + exact NL-to-SQL cache.

Sits alongside `semantic_search/nl_to_sql/` as the high-throughput, low-latency
analytics backend. Every component is enabled-but-empty by default:

- `ClickHouseClient` is construction-safe when CH is unreachable (mirrors
  `AthenaClient` — boots offline, fails IO methods explicitly).
- `MVRouter` becomes a pass-through when its catalog is empty (no MV match).
- `NLSqlExactCache` returns no hits until production traffic populates it.

Wiring contract (see `pipeline_router.AnalyticsRouter`):

  question
     ↓
  exact NL-to-SQL cache lookup              ── hit ──► validate → execute → verify
     ↓ miss
  existing 6-stage NL-to-SQL pipeline (LLM gen → validate → execute → verify)
     ↓ verifier-pass
  cache update (template extracted from the verified SQL)

The router NEVER bypasses the security validator or the post-execution verifier
gate, so a poisoned cache entry cannot exfiltrate data or surface a wrong
answer (defence-in-depth per `responsible-ai.mdc` §grounding).
"""

__all__: list[str] = []
