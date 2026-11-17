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

from serving.alerts import HI, LO, LOCKOUT, STEP, evaluate, snap

MGDL_PER_MMOL = 18.016
STALE_AFTER = timedelta(minutes=15)
FUTURE_SLACK = timedelta(minutes=5)
WATCH_MARGIN = 15.0        # mg/dL: projection within 70-85 or 235-250 -> "watch" (information, not an alarm)
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
    glucose: float = Field(..., description="CGM glucose value in the unit given by `unit`")


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
    for i, r in enumerate(readings, 1):
        if r.timestamp.tzinfo is None:
            raise ToolError(f"reading {i} has no timezone offset; every timestamp needs one, e.g. 2026-01-15T10:00:00Z")
        g = r.glucose * MGDL_PER_MMOL if unit == "mmol/L" else r.glucose
        out.append((r.timestamp.astimezone(timezone.utc), g))
    dedup = [out[0]] + [b for a, b in zip(out, out[1:]) if b != a]      # exact repeated rows: keep one
    if len(dedup) < len(out):
        out = dedup
        if len(out) < 4:
            raise ToolError("send between 4 and 288 different readings (20 min to 24 h at 5-min spacing), oldest first")
    for i, (a, b) in enumerate(zip(out, out[1:]), 1):
        if b[0] == a[0]:
            raise ToolError(f"readings {i} and {i + 1} have the same time but different values; send each time once")
        if b[0] < a[0]:
            raise ToolError(f"reading {i + 1} is earlier than reading {i}; send readings in time order, oldest first")
    values = [g for _, g in out]
    if unit == "mg/dL" and max(values) < 35:
        raise ToolError("these values look like mmol/L; call again with unit='mmol/L'")
    for i, g in enumerate(values, 1):
        if not 20 <= g <= 600:                    # numbers only are echoed (the input is a validated float)
            raise ToolError(f"reading {i} ({g:.0f} mg/dL) is outside 20-600 mg/dL (1.1-33 mmol/L), which is not a "
                            "plausible CGM reading; check that reading and the unit")
    return out


def _sentence(status: str, now: str, proj: str, direction: str | None, very_low: bool, since: int | None,
              proj_very_low: bool = False) -> str:
    check = "Check with a meter and follow your care plan."
    if status == "low_now":
        return (f"Your CGM reads {now}, which is {'very low (below 54 mg/dL, 3.0 mmol/L)' if very_low else 'low'}. "
                "Check with a meter now and treat the low as your care plan says; if you feel very unwell or cannot "
                "treat it yourself, get urgent help.")
    if status == "high_now":
        return (f"Your CGM reads {now}, which is high (above 250 mg/dL, 13.9 mmol/L). {check} Get urgent care if you "
                "feel very unwell, for example vomiting, confusion or trouble breathing.")
    if status == "ok":
        return f"Your CGM reads {now}; on this trend no low or high is expected in the next 20 minutes."
    way, edge = ("falling", "low") if direction == "low" else ("rising", "high")
    if proj_very_low:
        edge = "very low"
    if status == "watch":
        return (f"Your CGM reads {now} and is {way} (about {proj} in 20 minutes on this trend). Not out of range "
                "yet, but keep an eye on it and check again in about 10 minutes.")
    if status == "ongoing":
        return (f"Warning: glucose is {now} and still {way}; it could be {edge} (about {proj}) within about 20 "
                f"minutes. This trend first crossed the warning line about {since} minutes ago. {check}")
    lead = "Warning: g" if status == "loud" else "Heads-up: g"
    return f"{lead}lucose is {now} and {way}; it could be {edge} (about {proj}) within about 20 minutes. {check}"


def get_glucose_alert(readings: list[Reading], unit: Literal["mg/dL", "mmol/L"] = "mg/dL",
                      mode: Literal["current", "retrospective"] = "current") -> dict:
    """Check recent CGM readings for a coming low (< 70 mg/dL) or high (> 250 mg/dL) using GlucoPulse's tested trend
    alert. Send the person's readings, oldest first, ideally 5 min apart and covering the last 1-2 hours.

    Read `status` (never empty) and pass on `say_to_person`:
      "low_now" / "high_now"  the latest reading is already below 70 or above 250 mg/dL;
      "loud"                  near the limit and still heading out within about 20 minutes;
      "heads_up"              heading out of range within about 20 minutes;
      "ongoing"               a warning was given in the last 30 minutes and glucose is still heading out;
      "watch"                 in range, projection close to the line (70-85 or 235-250 mg/dL); not an alarm;
      "ok"                    in range and no low or high projected in the next 20 minutes.
    Also returns the 20-minute projection (capped to the 40-400 mg/dL sensor range), the trend per minute and
    `notes` on anything adjusted. `new_alarm_at_latest_reading` is for notification apps (null during the 30-minute
    quiet period); do not read it as an all-clear. mode="current" (default) refuses data whose latest reading is more than 15 minutes old;
    use mode="retrospective" to review older data (the result is then labelled as not a current alert).

    Read-only and stateless: nothing is stored. Screening information only, not a diagnosis.
    Not for insulin dosing. Tell the person to confirm with a meter and follow their care plan.
    Validated only in adults not using insulin."""
    subject = _subject()
    try:
        LIMITER.check(subject)
        series = _validate(readings, unit)
        n_dropped = len(readings) - len(series)
        now = datetime.now(timezone.utc)
        if series[-1][0] > now + FUTURE_SLACK:
            raise ToolError("the latest reading is in the future; check the timestamps and timezone")
        if mode == "current" and now - series[-1][0] > STALE_AFTER:
            raise ToolError("the latest reading is more than 15 minutes old, so this cannot be a current alert; "
                            "call again with mode='retrospective' to review past data")
        res = evaluate([(t, min(max(g, 40.0), 400.0)) for t, g in series])   # sensor range, as tested
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
    raw_now = series[-1][1]
    proj_raw = res["projected_20min"]
    trend = (proj_raw - res["current_glucose"]) / 20.0                    # mg/dL per minute (15-min LS slope)
    proj = min(max(proj_raw, 40.0), 400.0)                                # CGM sensors report 40-400 mg/dL
    proj_txt = ("below " + fmt(40.0)) if proj_raw < 40 else ("above " + fmt(400.0)) if proj_raw > 400 else fmt(proj)
    last_t = snap(series)[-1][0]
    fired_now = res["alert"]
    events = res["alerts_in_window"]
    earlier = events[:-1] if fired_now else events
    heading = "low" if proj_raw < LO else "high" if proj_raw > HI else None
    moving = "low" if trend < 0 else "high" if trend > 0 else None
    recent = [a for a in earlier if a["direction"] == (heading or moving)
              and last_t - datetime.fromisoformat(a["timestamp"]) < STEP * LOCKOUT]
    near = proj_raw < LO + WATCH_MARGIN or proj_raw > HI - WATCH_MARGIN
    since = None
    if raw_now < LO:
        status, direction = "low_now", "low"
    elif raw_now > HI:
        status, direction = "high_now", "high"
    elif fired_now:
        status, direction = fired_now, res["direction"]
    elif recent and (heading or near):                      # warned < 30 min ago and still heading toward the line
        status, direction = "ongoing", heading or moving
        since = int((last_t - datetime.fromisoformat(recent[0]["timestamp"])).total_seconds() // 60)
    elif heading:                                           # never silent while the projection is out of range
        status, direction = "heads_up", heading
    elif proj_raw < LO + WATCH_MARGIN or proj_raw > HI - WATCH_MARGIN:   # not an alarm: close to the line
        status, direction = "watch", "low" if proj_raw < LO + WATCH_MARGIN else "high"
    else:
        status, direction = "ok", None
    notes = []
    n_clamped = sum(1 for _, g in series if g < 40 or g > 400)
    if n_clamped:
        notes.append(f"{n_clamped} reading(s) were outside the 40-400 mg/dL range CGM sensors report and were treated "
                     "as 40 or 400 (sensor LO/HI) for the trend; current_glucose is the value you sent.")
    if (proj_raw < 40 or proj_raw > 400) and not (raw_now < LO or raw_now > HI):
        notes.append("The straight-line projection went past the sensor range and is shown capped at 40-400 mg/dL; "
                     "real glucose does not keep falling or rising in a straight line.")
    out_now = status in ("low_now", "high_now")
    if raw_now <= 40 or raw_now >= 400:
        notes.append("The latest reading is at the sensor's limit (40 or 400 mg/dL); real glucose may be further out.")
    if n_dropped:
        notes.append(f"{n_dropped} exact duplicate row(s) were removed.")
    say = _sentence(status, fmt(raw_now), proj_txt, direction, raw_now < 54, since, proj_raw < 54)
    age = int((datetime.now(timezone.utc) - series[-1][0]).total_seconds() // 60)
    if mode == "retrospective":
        say = (f"Past data, NOT current: as of the latest reading ({series[-1][0]:%Y-%m-%d %H:%M} UTC, about "
               f"{age // 60} h {age % 60} min ago), status was '{status}'. "
               + say.replace("Your CGM reads", "The CGM read").replace(" is expected", " was expected")
               + " This says nothing about glucose now; check your CGM or meter for the current value.")
    elif age >= 8:
        est = min(max(raw_now + trend * age, 40.0), 400.0)
        say = (f"(Latest reading was {age} minutes ago; on its trend glucose may already be about {fmt(est)} now, "
               "and the 20 minutes below count from that reading.) " + say)
    if out_now:
        notes.append("Glucose is already out of range, so no projection is given.")
    out = {"status": status, "direction": direction, "say_to_person": say,
           "mode": mode, "unit": unit,
           "current_glucose": round(raw_now / k, 1), "projected_20min": None if out_now else round(proj / k, 1),
           "trend_per_min": round(trend / k, 2 if unit == "mg/dL" else 3),
           "current_glucose_mg_dl": round(raw_now, 1), "projected_20min_mg_dl": None if out_now else round(proj, 1), "latest_reading_minutes_ago": age,
           "trend_mg_dl_per_min": round(trend, 2),
           "new_alarm_at_latest_reading": fired_now,
           "continuous_readings_used": res["continuous_readings"],
           "earlier_alerts_in_window": [{"time": a["timestamp"], "tier": a["tier"], "direction": a["direction"]}
                                        for a in earlier][-10:],
           "notes": notes, "validated_population": VALIDATED, "disclaimer": DISCLAIMER}
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
                "on two readings in a row. Each tier is quiet for 30 minutes after it fires; the tool's `status` stays 'ongoing' "
                "during that time while glucose is still heading out, and is 'low_now'/'high_now' when the latest "
                "reading is already out of range. Results are about the next ~20 minutes only.",
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
