"""Two-tier CGM trend alerts for the API (numpy only; the serving image ships nothing from model/).

Same rules as the research code, tested in docs/TRIAGE.md sections 9 and 15-18 (parity with
model/cgmacros/replay.py and model/cgmacros/alarms.py is checked in tests/model_cgmacros/test_serving_alerts.py):
  heads_up  15-min least-squares slope projected 20 min ahead is outside 70-250 (alert20). On new people it warned of
            72% of lows (ShanghaiT2DM: 97% of highs) >= 15 min ahead; ~3 alerts/day that did not end in an event,
            most of them near-misses (glucose came within 70-85 or 220-250).
  loud      the same projection, only if glucose is already >= 200 (heading high) or <= 100 (heading low), on two
            readings in a row. Catches about half as many events in time with about a third fewer non-event alerts.
An alert opens a 30-min quiet period for its tier; a loud alert may still escalate during a heads-up quiet period.
Readings are snapped to the 5-minute grid (floored, first reading per slot kept), exactly as the research data were
binned, so the few seconds of drift in real CGM exports do not break the history; any gap after snapping restarts it.
At least 4 consecutive 5-minute readings are needed for a trend (15-minute data, e.g. Libre history, cannot be used).
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

LO, HI = 70.0, 250.0
STEP = timedelta(minutes=5)
LOCKOUT = 6                       # readings (30 min)
GATE_HI, GATE_LO = 200.0, 100.0
DISCLAIMER = ("Screening alert from CGM trend data, not a diagnosis or treatment advice. Confirm with a meter before "
              "acting; follow your care plan.")


def slope15(x: np.ndarray) -> np.ndarray:
    """Least-squares slope (mg/dL per min) over the last 4 readings (15 min); 0 for the first 3."""
    out = np.zeros(len(x))
    t = np.array([-7.5, -2.5, 2.5, 7.5])
    for k in range(3, len(x)):
        w = x[k - 3:k + 1]
        out[k] = float((t * (w - w.mean())).sum() / (t ** 2).sum())
    return out


def flags(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per reading: heads_up, loud, projected value 20 min ahead."""
    proj = x + slope15(x) * 20.0
    up, down = proj > HI, proj < LO
    heads = up | down
    hi = up & (x >= GATE_HI)
    lo = down & (x <= GATE_LO)
    loud = (hi & np.r_[False, hi[:-1]]) | (lo & np.r_[False, lo[:-1]])
    return heads, loud, proj


def snap(readings: list[tuple[datetime, float]]) -> list[tuple[datetime, float]]:
    """Floor timestamps to the 5-minute grid and keep the first reading in each slot (as dataset.grid_from_measurements
    does for the research data)."""
    out, seen = [], set()
    for t, g in sorted(readings, key=lambda r: r[0]):
        slot = t - timedelta(minutes=t.minute % 5, seconds=t.second, microseconds=t.microsecond)
        if slot not in seen:
            seen.add(slot)
            out.append((slot, g))
    return out


def segment_tail(readings: list[tuple[datetime, float]]) -> list[tuple[datetime, float]]:
    """The last run of readings spaced exactly 5 min apart (alerts need a continuous history)."""
    tail = readings[-1:]
    for prev in reversed(readings[:-1]):
        if tail[0][0] - prev[0] != STEP:
            break
        tail.insert(0, prev)
    return tail


def evaluate(readings: list[tuple[datetime, float]]) -> dict:
    """Run the alert rules over a window of readings (oldest first) and report the state at the last reading."""
    seg = segment_tail(snap(readings))
    x = np.array([g for _, g in seg], dtype=float)
    heads, loud, proj = flags(x)
    last = {"heads_up": -10 ** 9, "loud": -10 ** 9}
    events = []
    for k in range(len(x)):
        if not LO <= x[k] <= HI:
            continue                                   # already out of range: that is a reading, not a warning
        tier = "loud" if loud[k] and k - last["loud"] >= LOCKOUT else (
            "heads_up" if heads[k] and k - last["heads_up"] >= LOCKOUT else None)
        if tier:
            last[tier] = k
            if tier == "loud":
                last["heads_up"] = k
            events.append({"timestamp": seg[k][0].isoformat(), "tier": tier,
                           "direction": "high" if proj[k] > HI else "low",
                           "glucose": round(float(x[k]), 1), "projected_20min": round(float(proj[k]), 1)})
    now = events[-1] if events and events[-1]["timestamp"] == seg[-1][0].isoformat() else None
    reason = None
    if now:
        reason = (f"Glucose {now['glucose']:.0f} mg/dL and {'rising' if now['direction'] == 'high' else 'falling'}; "
                  f"on this trend it would be about {now['projected_20min']:.0f} mg/dL in 20 min.")
    return {"alert": now["tier"] if now else None, "direction": now["direction"] if now else None,
            "reason": reason, "current_glucose": round(float(x[-1]), 1),
            "projected_20min": round(float(proj[-1]), 1) if len(x) >= 4 else None,
            "continuous_readings": len(x), "alerts_in_window": events, "disclaimer": DISCLAIMER}
