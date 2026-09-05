# Observability

## Boundaries

| | What | Where |
| --- | ---- | ----- |
| **In place (code)** | Structured logs, probes, cost caps, `/measurement/signals` vs YAML thresholds | `app.py`, `cost/*`, `resilience/health.py`, `base.yaml` |
| **Docs only (this file)** | Alarm names, CW metric-filter patterns, CLI sketches, IaC YAML handoff, wire checklist | `docs/OBSERVABILITY.md` |
| **Platform (not in repo)** | Log group, retention, metric filters, CW alarms, SNS/Slack pages | Katana / AWS account |

**Not in this repo:** CloudWatch alarms, metric filters, SNS topics, Terraform/Katana alarm config, Prometheus, OTel. Naming something `auc-ss-{env}-*` here does **not** create it — paging stays off until platform wires.

**Why this split:** app owns emit; platform owns wire (log group + IAM). Catalog = handoff contract, not a live alarm system.

Threshold knobs: `base.yaml` → `measurement.thresholds`, cost caps. Ops summary: `PLAYBOOK.md` §12.

---

## Probes

| Path | Role |
| ---- | ---- |
| `GET /healthz` | Liveness / ALB |
| `GET /resilience/health` | Store / LLM degrade |
| `GET /capabilities` | Subsystem snapshot |
| `GET /measurement/signals` | Proxy rates vs YAML |

---

## Log tokens

| Token | Meaning |
| ----- | ------- |
| `qie_only_complete` | qie_only done (`latency_ms`, `decision_cost_usd`, `prompt_tag`, `schema_version`, `hard_filter_count`, …) |
| `search_timeout_sla_breach` | Search SLA miss |
| `retrieval_degraded` | Backends dropped |
| `dependency_health_degraded` | Boot dep fail |
| `backend_health_transition` | Qdrant/CH/LLM state |
| `query_cost_budget_exceeded` / `query_cost_budget_admit_denied` | `$0.05`/q |
| `fleet_cost_budget_exceeded` / `fleet_cost_budget_admit_denied` | `$25/h` · `$100/day` |
| `zero_result_guard_`* | Zero-result ladder |
| `ranking_stage_attribution` | Ranking stage counters; optional `pipeline_trace.stages` when `include_in_response=true` |

---

## Log level

`general.log_level` in `base.yaml` (default `INFO`). `LOG_LEVEL` env var overrides when set (default unset → `INFO`). Applied once at startup via `logging_utils.apply_log_level_from_config`, called from `lifespan()` after config load.

## Search-log S3 persistence

Every `/search` call (qie_only + full) fire-and-forget dumps its JSON result to S3 for offline log analysis/dashboarding — `asyncio.create_task(asyncio.to_thread(upload_search_result_json, ...))`, never blocks the response.

| | |
| --- | --- |
| **Bucket** | Parsed from `S3_PRETRAINED_DIR` env var (must be `s3://...`); skipped (warning logged) if unset/non-S3 |
| **Key** | `search_logs/{YYYY}/{MM}/{DD}/{request_id}.json` |
| **Retention** | S3 Lifecycle Expiration, 1095 days (3 years), rule ID `search-logs-3yr-expiration`, `Filter.Prefix = search_logs/`. Objects auto-delete after 3 years — no custom scan/delete job |
| **Applied** | Once at startup (`ensure_search_logs_retention_policy`, fire-and-forget, best-effort — missing IAM perm degrades to a logged warning, service still boots) |
| **Code** | `s3_search_log_uploader.py` |

---

## Alarms

Name: `auc-ss-{env}-<signal>` (`dev` \| `test` \| `prod`).


| Alarm | Sev | Signal | Fire when |
| ----- | --- | ------ | --------- |
| `…-liveness` | 1 | ALB `/healthz` | unhealthy ≥ 2m |
| `…-latency-p99` | 2 | ALB `TargetResponseTime` (prefer) or log `latency_ms` | p99 > 2500 ms / 5m |
| `…-latency-p50` | 3 | same | p50 > 300 ms / 15m |
| `…-http-5xx` | 1 | ALB 5xx rate | > 1% / 5m (min 50 req) |
| `…-store-health` | 2 | `backend_health_transition` or `/resilience/health` | unhealthy > 5m |
| `…-retrieval-degraded` | 2 | `retrieval_degraded` \| `search_timeout_sla_breach` | ≥ 10 / 5m |
| `…-llm-query-budget` | 2 | `query_cost_budget_*` | ≥ 5 / 15m |
| `…-llm-fleet-budget` | 1 | `fleet_cost_budget_*` | ≥ 1 / 5m |
| `…-zero-result` | 3 | `zero_result_rate` or `zero_result_guard_`* | > 3% after min sample |

---

## Metric filters

Log group = `$LOG_GROUP` (platform). Namespace example: `AucSemanticSearch`.

```text
# auc-ss-query-cost-budget → QueryCostBudgetEvents (Sum)
?query_cost_budget_exceeded ?query_cost_budget_admit_denied

# auc-ss-fleet-cost-budget → FleetCostBudgetEvents
?fleet_cost_budget_exceeded ?fleet_cost_budget_admit_denied

# auc-ss-retrieval-degraded → RetrievalDegradedEvents
?retrieval_degraded ?search_timeout_sla_breach

# auc-ss-backend-unhealthy → BackendHealthUnhealthy
%"backend_health_transition" %"to=unhealthy"
```

Latency / 5xx / liveness: use ALB metrics, not log parse.

```bash
aws logs put-metric-filter \
  --log-group-name "$LOG_GROUP" \
  --filter-name auc-ss-fleet-cost-budget \
  --filter-pattern '?fleet_cost_budget_exceeded ?fleet_cost_budget_admit_denied' \
  --metric-transformations \
    metricName=FleetCostBudgetEvents,metricNamespace=AucSemanticSearch,metricValue=1,defaultValue=0

aws cloudwatch put-metric-alarm \
  --alarm-name "auc-ss-prod-llm-fleet-budget" \
  --metric-name FleetCostBudgetEvents --namespace AucSemanticSearch \
  --statistic Sum --period 300 --evaluation-periods 1 --threshold 1 \
  --comparison-operator GreaterThanOrEqualToThreshold \
  --alarm-actions "$SNS_ONCALL_ARN"
```

---

## Catalog (IaC handoff)

```yaml
service: auc-semantic-search
cw_owner: platform
alarms:
  - {name: auc-ss-{env}-liveness, sev: 1, src: alb_/healthz, when: unhealthy_gte_2m}
  - {name: auc-ss-{env}-latency-p99, sev: 2, src: alb_TargetResponseTime, when: p99_gt_2500ms_5m}
  - {name: auc-ss-{env}-latency-p50, sev: 3, src: alb_TargetResponseTime, when: p50_gt_300ms_15m}
  - {name: auc-ss-{env}-http-5xx, sev: 1, src: alb_5xx, when: gt_1pct_5m}
  - {name: auc-ss-{env}-store-health, sev: 2, src: backend_health_transition, when: unhealthy_gt_5m}
  - {name: auc-ss-{env}-retrieval-degraded, sev: 2, src: retrieval_degraded|search_timeout_sla_breach, when: gte_10_5m}
  - {name: auc-ss-{env}-llm-query-budget, sev: 2, src: query_cost_budget_*, when: gte_5_15m}
  - {name: auc-ss-{env}-llm-fleet-budget, sev: 1, src: fleet_cost_budget_*, when: gte_1_5m}
  - {name: auc-ss-{env}-zero-result, sev: 3, src: zero_result_rate, when: gt_0.03}
```

---

## Wire

1. Confirm `$LOG_GROUP` + retention.
2. ALB health → `/healthz`.
3. Add filters above; add alarms → SNS/Slack.
4. ALB: 5xx, TargetResponseTime p99, unhealthy hosts.
5. Smoke non-prod; set deploy silence windows.

---

## Oncall

| Alarm | Check | Action |
| ----- | ----- | ------ |
| Liveness | startup log, deploy | Rollback |
| p99 | `search_timeout_sla_breach`, `retrieval_degraded` | Scale API; soft-fail CH |
| 5xx | `/resilience/health`, deploy | Fix hard fail |
| Store | Qdrant/CH secrets, network | Repair store; CH optional |
| Fleet budget | `model_id` / allowlist primary, cache | Cap spend; prefer allowlist head (see `DESIGN.md` §11) |
| Zero-result | `/data-build/status`, seed | Rebuild index / loosen filters |

---


## Cache stats

`GET /cache/stats` returns per-tier `{hits, misses}`:

| Tier | Source |
| ---- | ------ |
| `exact` | result exact cache |
| `structured` | structured intermediate |
| `intent_plan` | intent-plan cache |
| `intent_result` | QI exact intent-result cache |
| `semantic_intent` | QI fuzzy intent cache (off when disabled) |
| `qie_l0` | qie_only L0 LLM filter cache (key = normalize + `prompt_tag`/`schema_version`; LLM only) |

Hit-rate proxy signal / miss-storm detector use YAML lists:

- `measurement.thresholds.cache_hit_rate_tiers`
- `measurement.cache_miss_storm.tiers`

`qie_l0` is exposed for ops but omitted from those rollups (different traffic class).

Query-text normalization for QI / cache keys: `qi.normalize` in `base.yaml`.

---

## Out of scope

Prometheus, Grafana, OTel SDK, CI-applied Terraform for these alarms.

## Code

`app.py` · `orchestrator.py` · `resilience/health.py` · `cost/query_budget.py` · `cost/fleet_budget.py` · `base.yaml` `measurement.thresholds`
