"""Server-side grounding for panel text. A claim survives only if
  1. it cites at least one resource, and every cited ref was actually returned by a tool
     in this session (same rule as agents/citation.py);
  2. every measurement number in it is a NUMERIC FIELD VALUE of a cited resource (never a
     key, id, timestamp or description), or one of the published thresholds 54/70/180/250;
  3. a percentage is the forecast interval level (80%), and a duration is a protocol
     horizon (5/15/30/60/120 min, 2 h) or a time value present in a cited resource;
  4. it contains no treatment, dosing, medication or diagnostic language.
Free text without citations (dissent, concerns) must pass rules 2-4 against everything
the panel retrieved, or it is dropped too. Rule 2 is what stops a model from citing a real
forecast while quoting a value that forecast never produced.
"""
from __future__ import annotations

import json
import re

from agents.triage.schemas import Claim

NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")
NUMBER_TOL = 0.5
THRESHOLDS = {54.0, 70.0, 180.0, 250.0}
INTERVAL_LEVELS = {80.0}
HORIZON_MINUTES = {5.0, 15.0, 30.0, 60.0, 120.0}
PERCENT = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*(?:%|percent\b)", re.IGNORECASE)
DURATION = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*(min(?:ute)?s?\b|h\b|hrs?\b|hours?\b)", re.IGNORECASE)
TIME_FIELDS = ("minutes", "min_", "_min")      # numeric fields that carry a time value
ADVICE = re.compile(
    r"\b(?:give|take|administer|inject|bolus|prescrib\w*|recommend\w*|should (?:eat|drink|take|have)|"
    r"(?:eat|drink|consume) (?:some|a|an|\d+)|glucose tablets?|juice|metformin|insulin|sulfonylureas?|"
    r"medicat\w*|dose|doses|dosing|dosage|diagnos\w*|treatment|"
    r"treat (?:with|it|this|the (?:patient|low|high|hypo\w*))|call 911|emergency room)\b",
    re.IGNORECASE)   # whole words only: "gives", "treat the median as a guide" are not advice


def resources_in(messages: list) -> dict[str, dict]:
    """ref -> resource for every resource a tool returned (read from tool-result
    messages, never from model text)."""
    found: dict[str, dict] = {}

    def walk(obj):
        if isinstance(obj, dict):
            if isinstance(obj.get("resourceType"), str) and isinstance(obj.get("id"), str):
                found[f"{obj['resourceType']}/{obj['id']}"] = obj
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    for msg in messages or []:
        for block in msg.get("content", []) or []:
            tr = block.get("toolResult") if isinstance(block, dict) else None
            for item in (tr or {}).get("content", []) or []:
                if "json" in item:
                    walk(item["json"])
                elif "text" in item:
                    try:
                        walk(json.loads(item["text"]))
                    except (ValueError, TypeError):
                        pass
    return found


def numeric_fields(obj, key: str = "") -> list[tuple[str, float]]:
    """(field path, value) for numeric leaves only. Strings (ids, timestamps,
    descriptions) and dict keys never contribute numbers."""
    out: list[tuple[str, float]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += numeric_fields(v, f"{key}.{k}" if key else str(k))
    elif isinstance(obj, list):
        for v in obj:
            out += numeric_fields(v, key)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.append((key, float(obj)))
    return out


def _match(x: float, pool: set[float]) -> bool:
    return any(abs(abs(x) - abs(n)) <= NUMBER_TOL for n in pool)


def problems(text: str, resources: list[dict]) -> list[str]:
    """Why this text is not grounded in `resources` (empty list = grounded)."""
    issues = []
    if ADVICE.search(text):
        issues.append("treatment_or_diagnosis_language")
    fields = [f for r in resources for f in numeric_fields(r)]
    values = {v for _, v in fields}
    times = {v for k, v in fields if any(t in k for t in TIME_FIELDS)} | HORIZON_MINUTES
    rest = text
    for m in PERCENT.finditer(text):
        if float(m.group(1)) not in INTERVAL_LEVELS and not _match(float(m.group(1)), values):
            issues.append(f"percentage_not_supported:{m.group(1)}")
    rest = PERCENT.sub(" ", rest)
    for m in DURATION.finditer(rest):
        minutes = float(m.group(1)) * (60 if m.group(2).lower().startswith("h") else 1)
        if not _match(minutes, times):
            issues.append(f"duration_not_supported:{m.group(1)}")
    rest = DURATION.sub(" ", rest)
    stray = [x for x in (float(n) for n in NUMBER.findall(rest)) if not _match(x, values | THRESHOLDS)]
    if stray:
        issues.append("number_not_in_cited_resource:" + ",".join(f"{x:g}" for x in stray))
    return issues


def ground_claims(claims: list[Claim], retrieved: dict[str, dict]) -> tuple[list[Claim], list[dict]]:
    kept, dropped = [], []
    for c in claims:
        refs = [r.strip() for r in c.evidence]
        if not refs:
            dropped.append({"text": c.text, "reason": "no_citation"})
            continue
        missing = [r for r in refs if r not in retrieved]
        if missing:
            dropped.append({"text": c.text, "reason": "cites_unretrieved_resource", "refs": missing})
            continue
        issues = problems(c.text, [retrieved[r] for r in refs])
        if issues:
            dropped.append({"text": c.text, "reason": issues[0].split(":")[0], "detail": issues})
            continue
        kept.append(Claim(text=c.text, evidence=refs))
    return kept, dropped


def ground_free_text(items: list[str], retrieved: dict[str, dict]) -> tuple[list[str], list[dict]]:
    """Dissent / concerns carry no citations: they must still contain no advice and no
    number that isn't somewhere in what the panel retrieved."""
    pool = list(retrieved.values())
    kept, dropped = [], []
    for text in items:
        issues = problems(text, pool)
        if issues:
            dropped.append({"text": text, "reason": issues[0].split(":")[0], "detail": issues})
        else:
            kept.append(text)
    return kept, dropped
