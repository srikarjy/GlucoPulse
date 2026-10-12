"""
Deterministic criterion evaluation against a FHIR-derived chart.

Tri-state per criterion:  met=True | met=False (contradicted) | met=None (no usable evidence).
Evidence strings are *references* (e.g. "Observation/4548-4@2025-12-01"), never note text.

Prompt-injection defence: any note matching INJECTION_RE is quarantined -- it is
excluded from evidence matching and raises the `prompt_injection` flag. Notes are
treated as untrusted data, never as instructions.
"""
from __future__ import annotations

import operator
import re
from datetime import date

from priorauth.fhir import Chart
from priorauth.schemas import Criterion, CriterionResult, Policy

INJECTION_RE = re.compile(
    r"(ignore (all |any )?(previous|prior|policy|the)\b|disregard (all |any )?(previous|prior)|system override|"
    r"note to (the )?(ai|assistant|model)|you must (output|approve|mark)|###\s*instructions|mark (this|the) request as approved)",
    re.I)
OPS = {">=": operator.ge, ">": operator.gt, "<=": operator.le, "<": operator.lt}


def find_injections(texts: list[tuple[str, str]]) -> list[str]:
    """texts: [(ref, text)] -> refs of quarantined sources."""
    return [ref for ref, t in texts if INJECTION_RE.search(t or "")]


def _days(a: date, b: date) -> int:
    return (a - b).days


def evaluate_criterion(c: Criterion, chart: Chart, quarantined: set[str]) -> tuple[CriterionResult, list[str]]:
    p, flags = c.params, []
    mk = lambda met, ev=(), note="": CriterionResult(criterion_id=c.id, met=met, evidence=list(ev), note=note)

    if c.kind == "age":
        if chart.age is None:
            return mk(None, note="age unavailable"), flags
        return mk(chart.age >= p["min"], [f"Patient/{chart.patient_ref}#age"]), flags

    if c.kind == "diagnosis":
        if not chart.conditions:
            return mk(None, note="no conditions on chart"), flags
        hit = [x for x in chart.conditions if any(x.code.startswith(pr) for pr in p["prefixes"])]
        return (mk(True, [f"Condition/{x.code}" for x in hit]) if hit
                else mk(False, note="no qualifying diagnosis on chart")), flags

    if c.kind == "duration":
        hit = [x for x in chart.conditions if any(x.code.startswith(pr) for pr in p["prefixes"])]
        if not hit or all(h.onset is None for h in hit):
            return mk(None, note="no dated qualifying condition"), flags
        best = max(_days(chart.as_of, h.onset) for h in hit if h.onset)
        return mk(best >= p["min_days"], [f"Condition/{h.code}#onset" for h in hit if h.onset],
                  f"{best} days observed"), flags

    if c.kind == "lab_threshold":
        obs = [o for o in chart.obs if o.loinc == p["loinc"]]
        if not obs:
            return mk(None, note="no result on chart"), flags
        recent = [o for o in obs if 0 <= _days(chart.as_of, o.day) <= p["within_days"]]
        if not recent:
            flags.append("stale_evidence")
            return mk(None, note=f"no result within {p['within_days']} days"), flags
        latest_day = max(o.day for o in recent)
        same_day = [o for o in recent if o.day == latest_day]
        verdicts = {OPS[p["op"]](o.value, p["value"]) for o in same_day}
        if len(verdicts) > 1:
            flags.append("conflicting_data")
            return mk(None, [f"Observation/{p['loinc']}@{latest_day}"], "conflicting results on same date"), flags
        return mk(verdicts.pop(), [f"Observation/{p['loinc']}@{latest_day}"]), flags

    if c.kind == "prior_therapy":
        meds = [m for m in chart.meds if m.drug_class in p["classes"] and m.start]
        if not meds:
            return mk(None, note="no matching therapy on chart"), flags
        best = max(_days(m.end or chart.as_of, m.start) for m in meds)
        return mk(best >= p["min_days"], [f"MedicationStatement/{m.name}" for m in meds], f"{best} days of therapy"), flags

    if c.kind == "documentation":
        for n in chart.notes:
            if n.note_id in quarantined:
                continue
            low = n.text.lower()
            if any(k in low for k in p["keywords"]):
                return mk(True, [n.note_id]), flags
        return mk(None, note="no supporting documentation found"), flags

    return mk(None, note=f"unsupported criterion kind {c.kind}"), flags


def evaluate_policy(policy: Policy, chart: Chart, request_text: str = "") -> tuple[list[CriterionResult], list[str]]:
    sources = [(n.note_id, n.text) for n in chart.notes] + [("request.free_text", request_text)]
    bad = find_injections(sources)
    flags: list[str] = ["prompt_injection"] if bad else []
    quarantined = set(bad)
    results = []
    for c in policy.criteria:
        r, f = evaluate_criterion(c, chart, quarantined)
        results.append(r)
        flags += [x for x in f if x not in flags]
    return results, flags


def overall(policy: Policy, results: list[CriterionResult]) -> str:
    req = {c.id for c in policy.criteria if c.required}
    rr = [r for r in results if r.criterion_id in req]
    if any(r.met is False for r in rr):
        return "does_not_support"
    if any(r.met is None for r in rr):
        return "insufficient_info"
    return "supports_approval"
