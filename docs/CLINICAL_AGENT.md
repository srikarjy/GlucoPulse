# Clinical Reasoning Agent

A **clinical decision support** tool, never a decision maker. One agent reads a patient's data through GlucoPulse's
existing FHIR R4 facade (`serving/healthcare/`), drafts a cited finding, and that draft is always routed to a human
reviewer. The agent cannot approve, reject, or edit its own output; only a `reviewer`-scoped principal can call
`submit_review_decision`, and the scope boundary is enforced server-side (`agents/mcp_server/auth.py`), not by
agent discipline. This is clinical decision **support**, never decision **making** — nothing here has been used
with PHI, and the platform has not been validated for clinical use.

> **Synthetic data only.** All patients and notes in the eval harness are synthetic. See the wording rule already
> established for this repo's healthcare layer: it *implements technical safeguards*, it is not "HIPAA compliant".

This unifies two things already in this repo: GlucoPulse's FHIR facade, hash-chained audit log, and SMART auth, and
`priorauth/`'s MCP tool-server pattern, idempotency, prompt-injection quarantine, and human-in-the-loop design. No
new auth system and no second FHIR client were built — see the reuse map below.

## Architecture at a glance

```
 Strands agent ──MCP (streamable HTTP, JWT)──▶ ToolBox ──▶ ehr.sync.EhrClient ──▶ existing /fhir facade
  (clinical-agent role,     agents/mcp_server/        agents/mcp_server/      (Patient, Observation,
   fhir:read + audit:write)      auth.py                   tools.py           DocumentReference, AuditEvent)
        │
        ▼ raw JSON draft
 agents.schemas.ClinicalFinding ──▶ agents.citation.ground_finding() ──▶ grounded finding
        │  (deterministic re-grounding: citation rule + missing-data rule -- the agent's
        │   own claim of "I cited this" or "I don't need to abstain" is never trusted)
        ▼
 submit_review_decision (reviewer role only, review:decide scope) ──▶ same hash-chained audit log
```

Tools: `fhir_read`, `fhir_search` (Patient/Observation, read-only), `fhir_search_notes` (DocumentReference,
quarantines injection attempts *inside the tool* before the agent ever sees note text), `audit_write`, and
`submit_review_decision` — the last is the only tool a `reviewer` principal can call and the only one a
`clinical-agent` principal cannot.

## Reuse map

| Component | Mirrors |
|---|---|
| MCP `ToolBox`/`@tool(scope, mutating)` registry, dual MCP-server/`LocalGateway` path | `priorauth/mcp_server/tools.py` |
| MCP auth (HS256 JWT, `Principal`, `BearerASGI`) | `priorauth/mcp_server/auth.py` (own namespace/secret) |
| Agent loop (Strands + MCPClient over streamable HTTP, untrusted-data system prompt) | `priorauth/agents/strands_backend.py` |
| Prompt-injection quarantine | `priorauth/criteria.py::find_injections` (ported, same regex) |
| `fhir_read`/`fhir_search` | `ehr.sync.EhrClient` (extended, not replaced) against the existing `/fhir` facade |
| `audit_write`/`submit_review_decision` | `serving.healthcare.audit.AuditLogger` — the SAME hash-chained log the facade itself writes to |

## Measured eval numbers (20 synthetic patients, `agents/eval/run_eval.py`)

Run: `.venv-pa/bin/python -m pytest tests/agents/test_eval_harness.py -q -s`

| Metric | Value | What it measures |
|---|---|---|
| `citation_precision_textual` | **1.0** | % of surviving sentences with a textually-matching evidence entry — a guarantee of `ground_finding()`'s own mechanism, not a live finding. |
| `citation_precision_grounded` | **1.0** | Stricter: of those, % whose cited `fhir_resource_id` was *actually retrieved* this session. Two scenarios script a model citing a fabricated resource id. `ground_finding()` checks every FHIR citation against the ids the tools actually returned (`retrieved_resource_ids()`, read from the agent's tool-result messages, not from the model's output). It rejects the fabricated citation, drops the sentence, and forces an abstention with a `rejected_…_unretrieved_resource` red flag. This was 0.875 before that check existed. |
| `abstention_accuracy` | **1.0** | Across all 20 scenarios: abstains when it should (4 no-data patients + 2 fabricated-citation patients) and doesn't when it shouldn't. |
| `quarantine_accuracy` | **1.0** | Across 2 injection-note patients (4 notes): the clean note is never quarantined, the injected one always is — checked against the tool's real output, independent of what the scripted model said. |

**What this does and doesn't prove**, same honesty limit as `priorauth`'s own `test_strands_backend.py`: a
`ScriptedModel` test double stands in for the LLM, so these numbers measure the deterministic re-grounding and
quarantine *mechanism's* correctness given controlled model outputs — not live LLM judgment quality. A live-model
eval against the same 20 scenarios is a natural next step, not yet done.

## What is verified, and what is not

| | Status |
|---|---|
| Tool layer against the real FHIR facade (in-process) | Verified — `tests/agents/test_phase_a.py` |
| Agent loop over real MCP/streamable-HTTP, citation + missing-data rules | Verified — `tests/agents/test_phase_b.py` |
| DocumentReference facade route + quarantine | Verified — ad-hoc smoke test at build time, and continuously via the eval harness's `quarantine_accuracy` metric (no standalone `test_phase_c.py` file; folded into `test_eval_harness.py`) |
| 20-scenario eval harness | Verified, real numbers above — `tests/agents/test_eval_harness.py` |
| Full audit chain (tool call → draft → review decision) | Verified — `tests/agents/test_phase_d.py` |
| Reviewer-only scope boundary on `submit_review_decision` | Verified — `tests/agents/test_phase_d.py` |
| Live Bedrock/Claude model | **Not run.** `agents/reasoning_agent.py::default_model()` needs AWS credentials; only the `ScriptedModel` path has been exercised, same caveat `priorauth/agents/strands_backend.py` states about itself. |
| A real human-review UI page | **Not built.** `submit_review_decision` is a backend MCP tool only; `priorauth-ui/app/(app)/review` has the UI pattern this would plug into if built out. |

## Demo script

1. Pick a synthetic patient with data (e.g. one of `agents/eval/run_eval.py`'s `gp-eval-grounded-*` ids).
2. Ask: *"Summarize glycemic control over the last 30 days."*
3. Show the agent's tool call (`fhir_search`) and its raw drafted JSON on screen.
4. Show `ground_finding()` running: a sentence without a matching citation gets visibly dropped, flagged in
   `red_flags`.
5. Switch to a patient with no data: show the agent abstain via `missing_data`, not guess.
6. Switch to a patient with an injected note: show `fhir_search_notes` return `quarantined: true` and empty text —
   the agent never sees the raw instruction-like content.
7. Call `submit_review_decision` as a `reviewer` principal, approving the finding.
8. Open the audit log and show the continuous hash chain: the tool call, the draft, and the review decision, in
   order, each linking to the previous entry's hash (`verify_chain()` returns `(True, None)`).

No `PORTFOLIO_VIDEO_GUIDE.md` or similar storyboard file exists elsewhere in this repo to extend — this script is
the demo's only written form today.
