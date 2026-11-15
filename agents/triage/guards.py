"""Leak guards for the triage panel, on the way IN to the agents and on the way OUT.

IN   untrusted text inside tool results (meal notes, descriptions, device strings) is
     scanned with the same injection detector the clinical agent uses; a match is
     replaced before any model sees it, and recorded.
OUT  every claim, concern and dissent line is checked for direct identifiers (emails,
     phone numbers, MRN-like ids: serving.healthcare.deid.scan_text) and for references
     to any patient other than the case's own. Offending text is dropped, never redacted
     in place, so nothing half-sanitized reaches a clinician.
TOOLS each agent gets a bounded number of tool calls, so a looping or probing model
     cannot keep querying.
"""
from __future__ import annotations

import re
from typing import Any

from agents.quarantine import find_injections
from serving.healthcare.deid import scan_text

QUARANTINED = "[quarantined: untrusted text]"
TOOL_BUDGET = 8
# Things that look like a patient/participant id; anything other than the case's own is a leak.
PATIENT_LIKE = re.compile(r"\b(?:cgmacros|patient|participant|subject|pt)[-_ /#:]*[a-z]*\d{2,}\b", re.IGNORECASE)


def sanitize(obj: Any, path: str = "") -> tuple[Any, list[str]]:
    """Copy of a tool result with injection-like strings replaced; returns (clean, flagged paths)."""
    flagged: list[str] = []
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[k], f = sanitize(v, f"{path}.{k}" if path else k)
            flagged += f
        return out, flagged
    if isinstance(obj, list):
        out_list = []
        for i, v in enumerate(obj):
            c, f = sanitize(v, f"{path}[{i}]")
            out_list.append(c)
            flagged += f
        return out_list, flagged
    if isinstance(obj, str) and find_injections([(path, obj)]):
        return QUARANTINED, [path]
    return obj, flagged


def leak_issues(text: str, patient: str) -> list[str]:
    issues = [f"identifier_in_text:{kind}" for kind in scan_text(text)]
    for m in PATIENT_LIKE.finditer(text):
        if m.group(0).lower() != patient.lower():
            issues.append(f"other_patient_reference:{m.group(0)}")
    if QUARANTINED in text:
        issues.append("repeats_quarantined_text")
    if find_injections([("out", text)]):
        issues.append("injection_text_in_output")
    return issues


class ToolBudget:
    def __init__(self, limit: int = TOOL_BUDGET):
        self.limit, self.used = limit, 0

    def take(self) -> bool:
        self.used += 1
        return self.used <= self.limit
