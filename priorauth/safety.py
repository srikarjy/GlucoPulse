"""
Human-in-the-loop safety policy.

Invariants (each has a test in tests/priorauth/test_safety.py):
  1. Every request that completes the agent pipeline yields a ReviewTask. There is
     no "auto-approve" or "auto-deny" branch -- confidence only changes WHO reviews
     and HOW URGENTLY, never WHETHER a human reviews.
  2. Denial-leaning drafts (does_not_support) always route to a physician reviewer.
  3. Agent-reachable code cannot construct a FinalDecision.
  4. Confidence is computed server-side from evidence; an agent's self-reported
     number is only an upper bound, never trusted upward.
"""
from __future__ import annotations

from dataclasses import dataclass
from priorauth.schemas import (CriterionResult, DraftRecommendation, Recommendation, ReviewTier)

HIGH_CONFIDENCE = 0.85
LOW_CONFIDENCE = 0.60
PHYSICIAN_FLAGS = {"prompt_injection", "conflicting_data"}
FLAG_PENALTY = {"prompt_injection": 0.30, "conflicting_data": 0.15, "code_mismatch": 0.15,
                "stale_evidence": 0.05, "no_policy": 0.30}


def calibrate_confidence(rec: Recommendation, results: list[CriterionResult], flags: list[str],
                         agent_claim: float | None = None) -> float:
    n = max(len(results), 1)
    met = sum(r.met is True for r in results)
    unmet = sum(r.met is False for r in results)
    unk = sum(r.met is None for r in results)
    grounded = sum(1 for r in results if r.met is not None and r.evidence) / n
    if rec == Recommendation.SUPPORTS_APPROVAL:
        base = 0.70 + 0.27 * grounded * (met / n)
    elif rec == Recommendation.DOES_NOT_SUPPORT:
        base = 0.62 + 0.25 * min(1.0, unmet / 2) + (0.08 if unk == 0 else 0.0)
    else:
        base = 0.82 if unk == 1 else 0.72 if unk == 2 else 0.62
    base -= sum(FLAG_PENALTY.get(f, 0.05) for f in flags)
    if "no_policy" in flags:
        base = min(base, 0.45)
    if "prompt_injection" in flags:
        base = min(base, 0.50)
    if agent_claim is not None:
        base = min(base, agent_claim)        # agent may lower, never raise
    return round(max(0.05, min(0.97, base)), 3)


@dataclass
class Routing:
    tier: ReviewTier
    priority: int
    reasons: list[str]


def route(draft: DraftRecommendation, urgency: str = "standard") -> Routing:
    reasons = ["agent_draft_always_requires_human"]
    tier = ReviewTier.STANDARD
    if draft.recommendation == Recommendation.DOES_NOT_SUPPORT:
        tier = ReviewTier.PHYSICIAN
        reasons.append("denial_leaning_requires_physician")
    if draft.confidence < LOW_CONFIDENCE:
        tier = ReviewTier.PHYSICIAN
        reasons.append("low_confidence")
    elif draft.confidence < HIGH_CONFIDENCE and tier == ReviewTier.STANDARD:
        tier = ReviewTier.PRIORITY
        reasons.append("moderate_confidence")
    for f in draft.flags:
        reasons.append(f"flag:{f}")
        if f in PHYSICIAN_FLAGS:
            tier = ReviewTier.PHYSICIAN
        elif tier == ReviewTier.STANDARD:
            tier = ReviewTier.PRIORITY
    if draft.recommendation == Recommendation.INSUFFICIENT_INFO:
        reasons.append("missing_information")
    prio = {ReviewTier.PHYSICIAN: 2, ReviewTier.PRIORITY: 3, ReviewTier.STANDARD: 4}[tier]
    if urgency == "expedited":
        prio = 1
        reasons.append("expedited_request")
    return Routing(tier, prio, reasons)
