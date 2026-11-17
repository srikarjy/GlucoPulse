"""The tools ChatGPT and Claude may call. Read-only, stateless: the caller supplies glucose readings; nothing is stored
and no stored patient data can be read. Only numbers and timestamps are accepted, so no free text from the input is
ever echoed back to the model (no prompt-injection path through tool output).

Safety rules built into the contract:
  * units are explicit (mg/dL or mmol/L, converted server-side); implausible values fail closed with a hint;
  * "current" alerts refuse stale data (latest reading > 15 min old); old data must be asked for as
    "retrospective" and is labelled as not current;
  * every result carries the validated population and a disclaimer: screening only, not for insulin dosing.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from serving.alerts import evaluate

MGDL_PER_MMOL = 18.016
STALE_AFTER = timedelta(minutes=15)
FUTURE_SLACK = timedelta(minutes=5)
RATE_PER_MIN = int(os.getenv("GLUCOPULSE_MCP_RATE_PER_MIN", "30"))
AUDIT = Path(os.getenv("GLUCOPULSE_MCP_AUDIT_LOG", "logs/public_mcp_audit.jsonl"))
# keyed hash of the user id in the audit log (a plain hash of a guessable id, e.g. an email, can be reversed). Without
# GLUCOPULSE_MCP_AUDIT_KEY a random per-process key is used, so hashes are not linkable across restarts.
AUDIT_KEY = (os.getenv("GLUCOPULSE_MCP_AUDIT_KEY") or secrets.token_hex(32)).encode()

VALIDATED = ("Tested on adults NOT using insulin (CGMacros; BIG IDEAs; Hall 2018; ShanghaiT2DM non-insulin). Not "
             "validated for type 1 diabetes, insulin users, children or pregnancy.")
DISCLAIMER = ("Screening information from CGM trends, not a diagnosis or medical advice. Not for insulin dosing. "
              "Confirm with a meter before acting and follow your care plan; for symptoms of a severe low or high, "
              "seek urgent care.")


class Reading(BaseModel):
    timestamp: datetime = Field(..., description="ISO 8601 time WITH timezone offset, e.g. 2026-01-15T10:00:00Z")
    glucose: float = Field(..., gt=0, le=1000, description="CGM glucose value in the unit given by `unit`")


class ToolError(Exception):
    """Shown to the calling model as a tool error; messages tell it how to fix the call."""


class RateLimiter:
    def __init__(self, per_min: int):
        self.per_min, self.hits, self.lock = per_min, {}, threading.Lock()

    def check(self, subject: str) -> None:
        now = time.monotonic()
        with self.lock:
            recent = [t for t in self.hits.get(subject, []) if now - t < 60]
            if len(recent) >= self.per_min:
                raise ToolError(f"rate limit: at most {self.per_min} calls per minute; try again shortly")
            recent.append(now)
            self.hits[subject] = recent


LIMITER = RateLimiter(RATE_PER_MIN)


def _audit(tool: str, subject: str, outcome: str, **meta) -> None:
    """Metadata only: no glucose values, no times of readings, no identifiers beyond a keyed hash of the subject.
    GLUCOPULSE_MCP_AUDIT_LOG=stdout writes to the platform log stream (serverless hosts have no writable disk)."""
    rec = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tool": tool, "outcome": outcome,
           "subject_hmac": hmac.new(AUDIT_KEY, subject.encode(), hashlib.sha256).hexdigest()[:16], **meta}
    if str(AUDIT) == "stdout":
        print(json.dumps({"glucopulse_audit": rec}), flush=True)
        return
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")


def _subject() -> str:
    from mcp.server.auth.middleware.auth_context import get_access_token
    tok = get_access_token()
    return tok.subject if tok and tok.subject else "anonymous"


def _validate(readings: list[Reading], unit: str) -> list[tuple[datetime, float]]:
    if not 4 <= len(readings) <= 288:
        raise ToolError("send between 4 and 288 readings (20 min to 24 h at 5-min spacing), oldest first")
    out = []
    for r in readings:
        if r.timestamp.tzinfo is None:
            raise ToolError("every timestamp needs a timezone offset, e.g. 2026-01-15T10:00:00Z")
        g = r.glucose * MGDL_PER_MMOL if unit == "mmol/L" else r.glucose
        out.append((r.timestamp.astimezone(timezone.utc), g))
    if any(b[0] <= a[0] for a, b in zip(out, out[1:])):
        raise ToolError("readings must be in time order, oldest first, with no duplicate times")
    values = [g for _, g in out]
    if unit == "mg/dL" and max(values) < 35:
        raise ToolError("these values look like mmol/L; call again with unit='mmol/L'")
    if min(values) < 20 or max(values) > 600:
        raise ToolError("values outside 20-600 mg/dL (1.1-33 mmol/L) are not plausible CGM readings; check the unit")
    return [(t, min(max(g, 40.0), 400.0)) for t, g in out]          # CGM sensors report 40-400 mg/dL


def get_glucose_alert(readings: list[Reading], unit: Literal["mg/dL", "mmol/L"] = "mg/dL",
                      mode: Literal["current", "retrospective"] = "current") -> dict:
    """Check recent CGM readings for a coming low (< 70 mg/dL) or high (> 250 mg/dL) using GlucoPulse's tested trend
    alert. Send the person's readings, oldest first, ideally 5 min apart and covering the last 1-2 hours.

    Returns the alert at the latest reading: none, "heads_up" (glucose heading out of range within about 20 minutes)
    or "loud" (already near the limit and still heading out), with the direction, a plain-language reason and a
    20-minute projection. mode="current" (default) refuses data whose latest reading is more than 15 minutes old;
    use mode="retrospective" to review older data (the result is then labelled as not a current alert).

    Read-only and stateless: nothing is stored. Screening information only, not a diagnosis.
    Not for insulin dosing. Tell the person to confirm with a meter and follow their care plan.
    Validated only in adults not using insulin."""
    subject = _subject()
    try:
        LIMITER.check(subject)
        series = _validate(readings, unit)
        now = datetime.now(timezone.utc)
        if series[-1][0] > now + FUTURE_SLACK:
            raise ToolError("the latest reading is in the future; check the timestamps and timezone")
        if mode == "current" and now - series[-1][0] > STALE_AFTER:
            raise ToolError("the latest reading is more than 15 minutes old, so this cannot be a current alert; "
                            "call again with mode='retrospective' to review past data")
        res = evaluate(series)
        if res["continuous_readings"] < 4:
            raise ToolError("cannot check: the latest readings are not a run of at least 4 readings about 5 minutes "
                            "apart (20 minutes). 15-minute data (e.g. Libre history) or gaps cannot be used; this is "
                            "NOT an all-clear")
    except ToolError as e:
        _audit("get_glucose_alert", subject, "rejected", n_readings=len(readings), reason=str(e)[:60])
        raise
    _audit("get_glucose_alert", subject, "ok", n_readings=len(readings), mode=mode)
    k = MGDL_PER_MMOL if unit == "mmol/L" else 1.0
    fmt = (lambda v: f"{v / k:.1f} mmol/L") if unit == "mmol/L" else (lambda v: f"{v:.0f} mg/dL")
    reason = None
    if res["alert"]:
        reason = (f"Glucose {fmt(res['current_glucose'])} and "
                  f"{'rising' if res['direction'] == 'high' else 'falling'}; on this trend it would be about "
                  f"{fmt(res['projected_20min'])} in 20 minutes.")
    out = {"mode": mode, "alert": res["alert"], "direction": res["direction"], "reason": reason, "unit": unit,
           "current_glucose": round(res["current_glucose"] / k, 1), "projected_20min": round(res["projected_20min"] / k, 1),
           "current_glucose_mg_dl": res["current_glucose"], "projected_20min_mg_dl": res["projected_20min"],
           "continuous_readings_used": res["continuous_readings"],
           "earlier_alerts_in_window": [{"time": a["timestamp"], "tier": a["tier"], "direction": a["direction"]}
                                        for a in res["alerts_in_window"]][-10:],
           "validated_population": VALIDATED, "disclaimer": DISCLAIMER}
    if mode == "retrospective":
        out["note"] = "Retrospective review of past data: this is NOT a current alert."
    return out


def about_glucopulse_alerts() -> dict:
    """What GlucoPulse's glucose alerts are, how well they were tested, and their limits. Call this before
    describing the alerts' accuracy to a person, so the numbers you quote are the tested ones."""
    _audit("about_glucopulse_alerts", _subject(), "ok")
    return {
        "what": "A trend alert on CGM data: the 15-minute slope projected 20 minutes ahead; 'heads_up' if it points "
                "below 70 or above 250 mg/dL, 'loud' if glucose is already <= 100 or >= 200 and still heading out "
                "on two readings in a row. Each tier is quiet for 30 minutes after it fires.",
        "tested_results": {
            "CGMacros (45 adults, 5-min CGM)": "68% of lows/highs warned >= 15 min ahead; ~2.4 alerts/day not "
                                               "followed by an event",
            "BIG IDEAs + Hall 2018 (73 new adults, 5-min CGM, mostly lows)": "72% of lows warned >= 15 min ahead "
                                                                            "(median 25 min); ~3.2 alerts/day "
                                                                            "not followed by an event, 79% of them "
                                                                            "near-lows (glucose reached 70-85)",
            "ShanghaiT2DM (42 adults with type 2 diabetes, no insulin, 15-min CGM)": "97% of highs and 85% of lows "
                                                                                    "warned, usually 15 min ahead, "
                                                                                    "by a 15-minute version of the "
                                                                                    "alert; THIS tool needs 5-minute "
                                                                                    "readings",
            "loud tier on the 73": "53% of lows warned >= 15 min ahead; ~2.1 alerts/day not followed by an event "
                                   "(84% near-lows); missed its pre-set target of <= 1.5/day",
        },
        "limits": [VALIDATED,
                   "An event means >= 15 minutes beyond 70 or 250 mg/dL; shorter dips are not counted.",
                   "No information about meals, insulin or activity is used.",
                   "Forecasts and triage levels from the research are not exposed by this server."],
        "evidence": "docs/TRIAGE.md sections 9 and 15-19 of the GlucoPulse repository (pre-registered tests).",
        "disclaimer": DISCLAIMER,
    }
