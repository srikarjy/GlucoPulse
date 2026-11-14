"""
Structured output for the clinical-reasoning agent -- the exact shape the
locked scope specifies, enforced by pydantic the same way priorauth's
schemas.py enforces DraftRecommendation's shape.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class Confidence(str, Enum):
    LOW = "low"
    MED = "med"
    HIGH = "high"


class Evidence(BaseModel):
    claim: str
    fhir_resource_id: Optional[str] = None
    kb_source: Optional[str] = None

    @model_validator(mode="after")
    def _one_source(self):
        if not self.fhir_resource_id and not self.kb_source:
            raise ValueError("evidence needs fhir_resource_id or kb_source")
        return self


class ClinicalFinding(BaseModel):
    """The agent's draft. Never trusted as-is -- agents/citation.py
    re-grounds it server-side before a human ever sees it."""
    finding: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: Confidence
    missing_data: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
