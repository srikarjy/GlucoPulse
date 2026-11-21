"""Two-tier CGM trend alerts for the API (numpy only; the serving image ships nothing from model/).

Same rules as the research code, tested in docs/TRIAGE.md sections 9 and 15-18 (parity with
model/cgmacros/replay.py and model/cgmacros/alarms.py is checked in tests/model_cgmacros/test_serving_alerts.py):
  heads_up  15-min least-squares slope projected 20 min ahead is outside 70-250 (alert20). On new people it warned of
            72% of lows (ShanghaiT2DM: 97% of highs) >= 15 min ahead; ~3 alerts/day that did not end in an event,
            most of them near-misses (glucose came within 70-85 or 220-250).
  loud      the same projection, only if glucose is already >= 200 (heading high) or <= 100 (heading low), on two
            readings in a row; OR, from any level, the projection is out of range and glucose is moving >= 1.5 mg/dL/min,
            on two readings in a row (fast-fall escalation, docs/TRIAGE.md / alarms.py FAST-FALL protocol: on the 73
            external people timely .526 -> .542, 95% CI of the gain [.003, .033], +0.13 false alarms/day).
An alert opens a 30-min quiet period for its tier; a loud alert may still escalate during a heads-up quiet period.
Readings are snapped to the 5-minute grid (floored, first reading per slot kept), exactly as the research data were
binned, so the few seconds of drift in real CGM exports do not break the history; any gap after snapping restarts it.
At least 4 consecutive 5-minute readings are needed for a trend.

15-minute data (e.g. Libre history; evaluate() picks it when there is no 5-minute run): the version tested on
ShanghaiT2DM (docs/TRIAGE.md, model/cgmacros/shanghai.py): readings 13-17 min apart, not snapped; slope = last
step / 15 min, projected 20 min ahead; heads_up only (97% of highs, 85% of lows warned, 1.13 alerts/day not followed by
an event); 30-min quiet period = 2 readings. LOUD at 15 min (alarms_r3.py protocol B, chosen on CGMacros Libre, tested
on ShanghaiT2DM 2026-11-18): the heads-up condition AND (glucose <= 100 heading low, >= 200 heading high, or moving
>= 1.5 mg/dL/min) on two readings in a row: 36% of events warned >= 15 min ahead (random 6%; CI of the gain .245-.365)
at 0.37 alerts/day not followed by an event. A late confirmation, weaker than the 5-min loud tier. At least 2
consecutive 15-minute readings are needed (one step gives the slope, as tested).
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

LO, HI = 70.0, 250.0
STEP = timedelta(minutes=5)
LOCKOUT = 6                       # readings (30 min)
STEP15 = timedelta(minutes=15)
LOCKOUT15 = 2                     # readings (30 min) at 15-min spacing
GAP15 = (timedelta(minutes=13), timedelta(minutes=17))   # a 15-min run, as segmented in model/cgmacros/shanghai.py
GATE_HI, GATE_LO = 200.0, 100.0
FAST = 1.5                        # mg/dL per min: fast-move escalation to loud
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
    sl = slope15(x)
    proj = x + sl * 20.0
    up, down = proj > HI, proj < LO
    heads = up | down
    hi = up & (x >= GATE_HI)
    lo = down & (x <= GATE_LO)
    fast = heads & (np.abs(sl) >= FAST)
    two = lambda a: a & np.r_[False, a[:-1]]                       # noqa: E731  (2 readings in a row)
    loud = two(hi) | two(lo) | two(fast)
    return heads, loud, proj


def flags15(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """15-min spacing, as tested on ShanghaiT2DM: heads_up = shanghai.alert20_15; loud = alarms_r3.B_ARMS['B2_two']."""
    sl = np.r_[0.0, np.diff(x)] / 15.0
    proj = x + sl * 20.0
    lo, hi = proj < LO, proj > HI
    fast = np.abs(sl) >= FAST
    cond = (lo & ((x <= GATE_LO) | fast)) | (hi & ((x >= GATE_HI) | fast))
    return lo | hi, cond & np.r_[False, cond[:-1]], proj


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


def segment_tail15(readings: list[tuple[datetime, float]]) -> list[tuple[datetime, float]]:
    """The last run of readings 13-17 min apart (no snapping, as in the 15-min research segments)."""
    rs = sorted(readings, key=lambda r: r[0])
    tail = rs[-1:]
    for prev in reversed(rs[:-1]):
        if not GAP15[0] <= tail[0][0] - prev[0] <= GAP15[1]:
            break
        tail.insert(0, prev)
    return tail


def evaluate(readings: list[tuple[datetime, float]]) -> dict:
    """Run the alert rules over a window of readings (oldest first) and report the state at the last reading.
    `enough` is False when there is no usable run (4 x 5-min or 2 x 15-min): that is 'could not check', never an
    all-clear."""
    seg = segment_tail(snap(readings))
    spacing, need, lockout = 5, 4, LOCKOUT
    if len(seg) < 4:
        seg15 = segment_tail15(readings)
        if len(seg15) >= 2:
            seg, spacing, need, lockout = seg15, 15, 2, LOCKOUT15
    x = np.array([g for _, g in seg], dtype=float)
    heads, loud, proj = (flags15 if spacing == 15 else flags)(x)
    last = {"heads_up": -10 ** 9, "loud": -10 ** 9}
    events = []
    for k in range(len(x)):
        if not LO <= x[k] <= HI:
            continue                                   # already out of range: that is a reading, not a warning
        tier = "loud" if loud[k] and k - last["loud"] >= lockout else (
            "heads_up" if heads[k] and k - last["heads_up"] >= lockout else None)
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
            "projected_20min": round(float(proj[-1]), 1) if len(x) >= need else None,
            "continuous_readings": len(x), "spacing_min": spacing, "enough": len(x) >= need,
            "loud_tier_validated": True, "alerts_in_window": events, "disclaimer": DISCLAIMER}
