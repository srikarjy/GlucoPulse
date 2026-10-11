# Runbook

| Symptom / alert | Meaning | Action |
|---|---|---|
| `PADeadLetterJobs` | Retries exhausted, even the fail-closed fallback failed; the request is `failed` and has **no review task** | Find cause (`jobs.last_error`, worker logs). Fix. `python -m priorauth.scripts.replay_dead`. Confirm tasks appear. Until then triage those requests manually: a patient is waiting |
| `PAPipelineFailingClosed` | Agents are erroring; every affected request is forced to physician review | Check Bedrock/throttling, MCP health, recent deploy. Switch `PRIORAUTH_BACKEND=deterministic` as a degraded mode (drafts continue, no LLM) |
| `PAQueueStuck` | Jobs queued, no drafts produced | Workers down/wedged; check task count and DB connectivity. Expired leases are reclaimed automatically once workers return |
| `PAPhysicianBacklog` | Review capacity | Expedited requests sort first (priority 1). Add reviewers; do **not** lower thresholds to shrink the queue |
| `PAReviewerOverrideRateHigh` | Humans disagree with >30% of drafts | Possible drift or a policy change. Re-run the eval, inspect recent overrides, consider disabling the LLM backend |
| `PAToolAuthDenials` | An agent role is calling a tool outside its scope | Misconfigured role/version skew, or probing. Audit entries `tool.denied` show who and which tool |
| `PAPromptInjectionSpike` | Chart/letter text with instruction-like content | Expected to be rare; review the source facility. These already route to physicians |
| `/v1/audit/verify` → `intact: false` | Audit rows were altered or deleted | Treat as a security incident. `first_broken_seq` locates it; restore from backup and compare. Do not "repair" the chain |

Never "fix" a backlog by auto-approving. There is no code path that can.
