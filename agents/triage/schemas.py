"""Shapes for the triage panel. Clinical decision SUPPORT, never decision making:
every report requires human review, whatever level it carries."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Level = Literal["routine", "monitor", "urgent"]
ORDER = {"routine": 0, "monitor": 1, "urgent": 2}


def higher(a: Level, b: Level) -> Level:
    return a if ORDER[a] >= ORDER[b] else b


class Claim(BaseModel):
    text: str
    evidence: list[str] = Field(default_factory=list, description='refs like "ScenarioForecast/<id>"')


class Assessment(BaseModel):
    """One panelist's view. Parsed from the model's JSON, then re-grounded server-side."""
    role: str
    level: Level
    claims: list[Claim] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    confidence: Literal["low", "med", "high"] = "low"


class ScenarioRow(BaseModel):
    """Copied deterministically from the forecast tool output; never written by a model."""
    scenario: str
    ref: str
    t30_p50: float
    t60_p50: float
    t60_p10: float
    t60_p90: float
    minutes_until_below_70: int | None
    minutes_until_above_180: int | None


class TriageReport(BaseModel):
    patient: str
    as_of: str
    level: Level                      # final = max(rule floor, panel) -- the panel can raise, never lower
    rule_level: Level
    panel_level: Level
    claims: list[Claim]               # grounded claims only
    dropped_claims: list[dict]        # claims removed by grounding, with the reason (audit trail)
    dissent: list[str]
    scenarios: list[ScenarioRow]
    panel_incomplete: bool = False    # a panelist or the moderator failed; the clinician must see this
    errors: list[str] = Field(default_factory=list)
    requires_human_review: bool = True
    disclaimer: str = ("Clinical decision support from model-based forecasts on research data; "
                       "not a diagnosis or treatment advice. A clinician makes every decision.")
