"""
Core domain types for the prior-authorization platform.

Safety invariant (enforced by types, not convention): no agent-reachable type can
express a *final* determination. `DraftRecommendation` is advisory; the only
final outcomes live in `ReviewDecision`, which is created exclusively by the
human-reviewer API path.
"""
from __future__ import annotations

from datetime import datetime, UTC
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(UTC)


class Recommendation(str, Enum):
    """Advisory only. Deliberately NOT named approve/deny."""
    SUPPORTS_APPROVAL = "supports_approval"
    DOES_NOT_SUPPORT = "does_not_support"      # never a denial; a human must decide
    INSUFFICIENT_INFO = "insufficient_info"


class FinalDecision(str, Enum):
    """Human-only. Not importable from agent code paths (see test_safety)."""
    APPROVED = "approved"
    DENIED = "denied"
    PENDED_FOR_INFO = "pended_for_info"


class RequestStatus(str, Enum):
    RECEIVED = "received"
    PROCESSING = "processing"
    AWAITING_REVIEW = "awaiting_review"   # terminal for the agent pipeline
    FAILED = "failed"
    DECIDED = "decided"                   # set only by a human decision


class ReviewTier(str, Enum):
    STANDARD = "standard"            # RN/UM reviewer
    PRIORITY = "priority"            # fuller evidence check
    PHYSICIAN = "physician"          # MD required (denial-leaning or low confidence)


class PARRequest(BaseModel):
    """Inbound prior-authorization request (synthetic; no real PHI)."""
    request_id: str
    patient_ref: str                 # pseudonymous id, e.g. Patient/syn-0001
    service_code: str                # CPT/HCPCS or NDC-like drug code
    service_description: str = ""
    diagnosis_codes: list[str] = Field(default_factory=list)   # ICD-10
    provider_npi: str = ""
    urgency: str = "standard"        # standard | expedited
    free_text: str = ""              # provider's letter of medical necessity
    submitted_at: datetime = Field(default_factory=utcnow)


class Criterion(BaseModel):
    id: str
    text: str
    kind: str                        # diagnosis | lab_threshold | prior_therapy | documentation | duration | age
    params: dict[str, Any] = Field(default_factory=dict)
    required: bool = True


class Policy(BaseModel):
    policy_id: str
    title: str
    service_codes: list[str]
    summary: str
    criteria: list[Criterion]


class CriterionResult(BaseModel):
    criterion_id: str
    met: Optional[bool]              # None = could not be determined
    evidence: list[str] = Field(default_factory=list)   # source refs, never raw notes
    note: str = ""


class IntakeResult(BaseModel):
    request_id: str
    normalized_codes: list[str]
    service_code: str
    issues: list[str] = Field(default_factory=list)
    urgency: str = "standard"


class EvidenceReport(BaseModel):
    policy_id: str
    results: list[CriterionResult]
    flags: list[str] = Field(default_factory=list)   # e.g. prompt_injection, conflicting_data, code_mismatch


class DraftRecommendation(BaseModel):
    request_id: str
    policy_id: str
    recommendation: Recommendation
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    criteria_summary: list[CriterionResult]
    flags: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    status: str = "DRAFT"            # constant; there is no code path that sets anything else


class ReviewTask(BaseModel):
    task_id: str
    request_id: str
    tier: ReviewTier
    priority: int                    # 1 = most urgent
    reasons: list[str]
    draft: DraftRecommendation
    state: str = "open"              # open | claimed | closed
    claimed_by: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)


class ReviewDecision(BaseModel):
    task_id: str
    reviewer: str
    decision: FinalDecision
    comment: str = ""
    agreed_with_draft: bool = True
