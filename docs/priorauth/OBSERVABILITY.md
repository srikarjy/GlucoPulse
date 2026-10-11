# Observability

| Signal | Where | Notes |
|---|---|---|
| Metrics | `/metrics` on the API (`:8080`) and worker (`:9102`) | `pa_requests_total`, `pa_drafts_total{recommendation,tier}`, `pa_flags_total`, `pa_agent_seconds`, `pa_tool_calls_total{tool,outcome}`, `pa_draft_confidence`, `pa_pipeline_failures_total`, `pa_queue_jobs{state}`, `pa_review_tasks_open{tier}`, `pa_review_overrides_total` |
| Dashboard | `monitoring/priorauth/grafana/dashboards/priorauth.json` (auto-provisioned) | 11 panels: throughput, recommendation mix, review backlog by tier, queue states, agent p95, confidence, flags, fail-closed rate, override rate, tool outcomes |
| Alerts | `monitoring/priorauth/alerts.yml` (7 rules, all load healthy) | dead-lettered jobs (page), fail-closed rate (page), queue stuck (page), physician backlog, override rate >30%/day, tool-auth denials, injection spike |
| Traces | OTLP/HTTP → Jaeger (compose) | one `pipeline` span with five `agent.*` children per request; Strands adds `gen_ai.*` model/tool spans when the LLM backend runs |
| Audit | `audit_log` table, `/v1/audit` | tamper-evident; separate from telemetry |

## Langfuse

Langfuse ingests OTLP. Point the exporter at it instead of Jaeger (not tested here):

```
OTEL_EXPORTER_OTLP_ENDPOINT=https://<langfuse-host>/api/public/otel
OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic <base64(public_key:secret_key)>"
```

Prompts/completions contain chart text. They are **not** recorded unless `PRIORAUTH_TRACE_CONTENT=1`; only enable that against a self-hosted or BAA-covered Langfuse.

## Suggested SLOs (set after real traffic)

p95 request→review-task < 60 s (standard) / 15 s (expedited); dead-letter count = 0; fail-closed rate < 1%; physician backlog age < 4 h.
