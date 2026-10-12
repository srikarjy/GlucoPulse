# Prior-Authorization Agent Platform

A **review assistant**, not a decision maker. Five agents read a prior-authorization request and the patient's FHIR chart,
find the matching coverage policy, check each criterion against evidence, and **draft** an advisory recommendation. Every
request, regardless of confidence, is then escalated to a human reviewer. Agents cannot approve or deny; the type system,
API and tests all enforce that ([SAFETY.md](SAFETY.md)).

> **Synthetic data only.** All patients, notes, policies and scenarios are generated (`priorauth/synth/`). The policies are
> invented and reflect no real payer. Nothing here has been used with PHI, and the platform has not been validated for
> clinical use. See the wording rule in [SECURITY.md](SECURITY.md): it *implements technical safeguards*; it is not "HIPAA compliant".

## Architecture at a glance

```
 EHR / intake ──POST /v1/requests (Idempotency-Key)──▶  FastAPI  ──▶ Postgres (requests, jobs, audit chain, review tasks)
                                                          │                      ▲
                                                          ▼ job queue (leases, retry, dead-letter)
                                                    async worker pool ───────────┘
                                                          │  Orchestrator (fail-closed)
        ┌────────────┬───────────────┬────────────────────┬─────────────────┬───────────────┐
        ▼            ▼               ▼                    ▼                 ▼
     Intake  →  Policy retrieval →  Evidence validation → Decision drafting → Escalation       (5 Strands agents,
        └────────────┴───────────────┴──── MCP (streamable HTTP, JWT per role) ────┴────────   least-privilege tokens)
                                       │
                         17 tools: FHIR chart reads · policy RAG (pgvector | OpenSearch) · rule evaluation · draft · escalate · audit
                                       │
                 Reviewer queue ◀── ReviewTask (tier + priority chosen server-side) ──▶ human decision (only writer of FinalDecision)
```

Telemetry: Prometheus metrics + alerts, Grafana dashboard, OpenTelemetry traces (Jaeger / Langfuse via OTLP). See [OBSERVABILITY.md](OBSERVABILITY.md).

## Quickstart (OrbStack / Docker, no cloud account needed)

```bash
python -m venv .venv-pa && source .venv-pa/bin/activate           # all Python runs in a venv
pip install -r priorauth/requirements.txt -r priorauth/requirements-dev.txt
export PRIORAUTH_JWT_SECRET=$(openssl rand -hex 32)

pytest tests/priorauth -q                                          # ~90 tests; 10 need Postgres/OpenSearch/HAPI and skip unless their env var is set
python -m priorauth.eval.run_eval --min-accuracy 0.95             # 200-scenario eval + hard safety gates

docker compose -f docker-compose.priorauth.yml up -d --build      # api :8080  grafana :3001  prometheus :9091  jaeger :16687
python -m priorauth.scripts.smoke                                  # end-to-end through the real stack
```

## What is verified, and what is not

| Area | Status | Evidence |
|---|---|---|
| 5 agents, 16 MCP tools, scopes, idempotency, audit chain | **Verified** | `tests/priorauth/` (unit + real MCP-over-HTTP with Strands' `MCPClient`) |
| Strands agent loop → MCP → grounded draft → escalation | **Verified with a scripted model, and live on Bedrock (small sample)** | `test_strands_backend.py`; Claude Haiku 4.5, 16-scenario focus sample + re-runs, ~$1.8 total. Not a full 200-scenario LLM run |
| Postgres queue (`SKIP LOCKED`), audit chain under concurrency, pgvector HNSW | **Verified** on real Postgres 16 | `test_postgres.py`, with negative controls proving the tests can fail |
| OpenSearch k-NN retrieval | **Verified** on real OpenSearch 2.17 | `test_opensearch.py` (ranking parity with in-memory) |
| FHIR read integration | **Verified** against real HAPI 7.4.0 | `test_hapi.py`. Read-only; no SMART/OAuth handshake, no Epic/Cerner sandbox |
| Docker stack, metrics, alerts, dashboard, traces | **Verified** on OrbStack | `priorauth.scripts.smoke`; Jaeger shows `pipeline` + 5 `agent.*` spans |
| Containers as non-root, read-only rootfs, all caps dropped | **Verified** | ran API/worker/MCP + pipeline under those flags |
| 200-scenario eval | **Verified** (deterministic backend) | [EVALUATION.md](EVALUATION.md); read its limits section |
| Terraform (ECS/RDS/ALB/KMS…), K8s manifests (EKS) | **Statically validated only** | `terraform validate`, `kubeconform`; KEDA query tested on real Postgres. **Never applied to AWS.** |
| GitHub Actions workflow | **Linted only** (`actionlint`) | has not run on GitHub; the `deploy` job is untested |
| QLoRA training | **Not run, and currently not recommended** | needs a CUDA GPU. Dataset, leakage guard, baseline and dry-run are tested. See FINETUNING.md decision |
| Langfuse | **Not run** | OTLP wiring documented; verified only against Jaeger |
| Clinical validity / real payer policies / PHI handling | **Out of scope** | synthetic only |

## Documents

[ARCHITECTURE](ARCHITECTURE.md) · [SAFETY](SAFETY.md) · [SECURITY](SECURITY.md) · [MCP_TOOLS](MCP_TOOLS.md) · [EVALUATION](EVALUATION.md) ·
[FINETUNING](FINETUNING.md) · [OBSERVABILITY](OBSERVABILITY.md) · [DEPLOYMENT](DEPLOYMENT.md) · [RUNBOOK](RUNBOOK.md)
