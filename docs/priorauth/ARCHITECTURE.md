# Architecture and design decisions

## Components

| Component | Module | Notes |
|---|---|---|
| API | `priorauth/api.py` | Submit (idempotent), status, reviewer queue/claim/decide, audit. Three token audiences: submitter, reviewer-rn/md, auditor |
| Queue + worker | `store.py` (`jobs`), `worker.py` | Postgres-backed: leases, expiry-based crash recovery, retry with backoff, dead-letter. `asyncio` workers run the sync pipeline in threads |
| Orchestrator | `agents/orchestrator.py` | Runs the five agents in order; **fails closed** to physician review |
| Agents | `agents/deterministic.py`, `agents/strands_backend.py` | Same five roles, two backends (rules / Strands+Bedrock) |
| MCP server | `mcp_server/` | One `ToolBox` serves both the HTTP MCP server and an in-process gateway, so tests exercise the same auth/idempotency/audit path |
| Rules engine | `criteria.py` | Tri-state criterion evaluation (met / contradicted / undeterminable), injection quarantine |
| RAG | `rag.py` | Code-first lookup, semantic fallback; in-memory, pgvector (HNSW) or OpenSearch (k-NN) |
| FHIR | `fhir.py` | R4 mapper + read-only `HttpFhirSource` |
| Store | `store.py` | SQLite (tests) / Postgres (prod); advisory locks around DDL and audit appends |

## The five agents

| Agent | Role / token | Does | Cannot |
|---|---|---|---|
| Intake | `intake-agent` | Reads the request, validates code syntax and diagnosis↔policy fit | Read the chart |
| Policy retrieval | `policy-agent` | Maps service → policy (exact code first, then semantic) | Read the chart or write anything |
| Evidence validation | `evidence-agent` | Evaluates each criterion against the FHIR chart | Write drafts |
| Decision drafting | `drafting-agent` | Stages an advisory draft | Read raw notes, escalate |
| Escalation | `escalation-agent` | Creates the review task, records info requests and audit notes | Change the draft |

## Decisions worth challenging

* **Rules decide *facts*; the model decides *language*.** Criteria like "A1c ≥ 7.0 within 90 days" are evaluated by code, not an LLM, because
  they are exact and auditable. The model's marginal value is free-text documentation judgement and rationale, which is what
  the eval shows the rules miss (paraphrased evidence) and what the fine-tune targets. If the rules were enough, an LLM would add risk, not value.
* **The model's output is *grounded*, not trusted** (`ToolBox._ground`): rule-derived flags can't be dropped, the recommendation can't be
  more favourable than the rules, and a model may only turn "no evidence" into "met" for a *documentation* criterion by citing a real, un-quarantined note.
  A lying-model test runs this through the real MCP stack.
* **Postgres as the queue.** One fewer system, transactional with the request/audit writes, `SKIP LOCKED` scales to this workload. SQS/Kafka would
  win at much higher throughput or for cross-service fan-out; the `Store` queue methods are the seam. Dead-lettered requests are marked `failed` and alert.
* **Code-first retrieval.** A coverage lookup must not be fooled by a nearby-sounding policy, so an exact service-code match outranks any semantic score.
  Semantic search handles unknown codes and is measured separately (see eval).
* **JWT (HS256) per agent role.** Simple and sufficient for one service; swap `auth.verify` for an OIDC/Cognito verifier in a multi-service deployment.
* **Alembic is absent on purpose-for-now.** Schema is created idempotently under an advisory lock (a real startup race was found and fixed this way). Production should use versioned migrations.
