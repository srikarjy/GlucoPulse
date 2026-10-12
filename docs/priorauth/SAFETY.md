# Safety model: draft, escalate, never decide

## Invariants (each enforced by a test)

1. **No agent-reachable type can express a final decision.** `DraftRecommendation.recommendation ∈ {supports_approval, does_not_support, insufficient_info}`;
   `approve`/`deny` raise. `FinalDecision` is referenced only by `api.py`, `store.py`, `schemas.py` (AST test over the whole package).
2. **Every request ends in a human review task.** Confidence changes *who* reviews and *how fast*, never *whether*. (`test_every_scenario_ends_in_an_open_review_task`; eval gate `human_review_rate = 1.0`.)
3. **Denial-leaning drafts always go to a physician**, and the API refuses a non-physician `denied` decision or physician-tier claim.
4. **Confidence is computed server-side** from evidence; an agent's number is only an upper bound.
5. **Fail closed.** Any agent error produces an `insufficient_info` draft flagged `agent_failure`, confidence 0.05, physician tier. If even that fails, the job dead-letters, the request is marked `failed`, and a page fires.
6. **Chart text is untrusted data.** Instruction-like notes are quarantined (excluded as evidence, text blanked in tool output), raise `prompt_injection`, cap confidence at 0.5, and force physician review. The model can't remove the flag.

## Routing (`priorauth/safety.py`)

| Condition | Tier | Priority |
|---|---|---|
| `does_not_support`, or confidence < 0.60, or flag `prompt_injection`/`conflicting_data` | physician | 2 |
| confidence 0.60–0.85, or any other flag (`code_mismatch`, `stale_evidence`, `no_policy`, `evidence_discrepancy`…) | priority | 3 |
| otherwise (`confidence ≥ 0.85`, no flags) | standard | 4 |
| `urgency = expedited` | (unchanged) | 1 |

## Honest limits

* The confidence formula is a **heuristic, not calibrated against real outcomes**. On the synthetic eval ECE is 0.18 (under-confident overall). Routing does not rely on it for safety:
  denials, flags and failures go to physicians regardless. Do not present it to reviewers as a probability.
* The injection filter is a regex plus structural isolation. It stops the demonstrated attacks, not a determined adversary; the real defences are that agents cannot decide and a human sees every draft.
* `agreed_with_draft` is recorded, but automation bias (reviewers rubber-stamping drafts) is a human-factors risk this repo cannot test. The override-rate alert is a weak proxy.
