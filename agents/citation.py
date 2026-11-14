"""
Server-side re-grounding of the agent's draft -- same philosophy as
priorauth's create_draft_determination: the agent's own say-so that a
sentence is cited, or that it has abstained correctly, is never trusted.
This module is the thing that actually enforces it, deterministically.

Citation rule: the agent is instructed (reasoning_agent.SYSTEM_PROMPT) to
set evidence[].claim to the EXACT sentence it supports. A sentence in
`finding` with no evidence entry whose claim matches it exactly is
dropped before anything reaches a human reviewer.

Missing-data rule: if nothing survives citation enforcement (everything
the agent asserted was uncited) and it also didn't declare a missing_data
gap, that is worse than an honest gap -- it means the agent asserted
things it couldn't back up AND didn't admit not knowing. That combination
is itself flagged and forced into an abstention.
"""
from __future__ import annotations

import re

from agents.schemas import ClinicalFinding, Evidence

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


def _verified(e: Evidence, retrieved_ids: set[str] | None) -> bool:
    """A FHIR citation only counts if that resource was actually returned
    by a tool call this session. Without this, a model can cite an id it
    invented and the sentence survives looking grounded. kb_source-only
    evidence isn't a FHIR claim, so it isn't checked here."""
    if retrieved_ids is None or not e.fhir_resource_id:
        return True
    return e.fhir_resource_id.strip() in retrieved_ids


def ground_finding(draft: ClinicalFinding,
                   retrieved_ids: set[str] | None = None) -> tuple[ClinicalFinding, list[str]]:
    """Returns (grounded_finding, dropped_sentences) -- callers keep
    `dropped` for the transcript/audit trail, never for the human-facing
    finding.

    retrieved_ids: "ResourceType/id" strings the agent's tools actually
    returned this session. When given, evidence citing any other FHIR id
    is discarded, so its sentence is dropped like an uncited one. None
    skips the check (textual grounding only)."""
    sentences = split_sentences(draft.finding)
    valid_evidence = [e for e in draft.evidence if e.claim and _verified(e, retrieved_ids)]
    unverified = sum(1 for e in draft.evidence if e.claim and not _verified(e, retrieved_ids))
    cited = {e.claim.strip() for e in valid_evidence}
    kept = [s for s in sentences if s in cited]
    dropped = [s for s in sentences if s not in cited]
    kept_evidence = [e for e in valid_evidence if e.claim.strip() in set(kept)]

    red_flags = list(draft.red_flags)
    missing_data = list(draft.missing_data)
    if unverified:
        red_flags.append(f"rejected_{unverified}_citation(s)_to_unretrieved_resource")
    if dropped:
        red_flags.append(f"dropped_{len(dropped)}_uncited_sentence(s)")
    if sentences and not kept and not missing_data:
        red_flags.append("no_grounded_finding_and_no_missing_data_declared")
        missing_data.append("insufficient cited evidence to support any finding")

    grounded = draft.model_copy(update={
        "finding": " ".join(kept),
        "evidence": kept_evidence,
        "red_flags": red_flags,
        "missing_data": missing_data,
    })
    return grounded, dropped
