"""Streaming version of the tested CGM alert (serving/alerts.py), one reading at a time. No Kafka here.

Why this is not just `evaluate(last 2 hours)`: the 30-minute quiet period is a chain. Whether reading k alerts depends
on whether an alert fired up to 30 min earlier, which depended on the one before that, back to the start of the
continuous run. evaluate() recomputes that chain from the start of whatever window it is given, so a truncated window
can restart the chain in the wrong place. Here each person's last alert time per tier is kept explicitly and updated
reading by reading, which reproduces evaluate() on the full history exactly (tests/streaming/test_alert_state.py).
That small state is also what must survive a restart (Phase 3).

Measured (2026-11-20): on two CGMacros people (5,718 whole-number mg/dL readings, 117 alerts) a plain evaluate() over
the last 2 hours gave the same alerts as the full history. It diverged only on a constructed 5-hour creep toward 70
with fractional values (as mmol/L converted to mg/dL can give): 3 of 6 alerts at 1 decimal, none with whole numbers.
This class matched the full history in every case, at about 2.5 ms per reading.

Arrival rules (decided in arrival order, never retroactively):
  * exact duplicate (same time, same value)          -> ignored
  * same time, different value                       -> ignored, first value kept ("conflict")
  * older than the person's newest reading ("late")  -> kept for future trends, but cannot raise an alert
  * older than the retention window                  -> dropped ("too_old")
  * in the same 5-min slot as the newest reading     -> evaluate() keeps the first per slot, so no new decision
Only the person's newest reading can alert, so a late reading never produces an alert about the past.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from serving.alerts import (HI, LO, LOCKOUT, LOCKOUT15, evaluate, flags, flags15, segment_tail, segment_tail15,
                            snap)

RETENTION = timedelta(hours=24)      # long enough to recompute the chain after a gap or a 5/15-min switch


@dataclass
class _Person:
    readings: list[tuple[datetime, float]] = field(default_factory=list)   # sorted, unique times
    mode: int | None = None              # 5 or 15: spacing of the run the last decision used
    newest_seg_time: datetime | None = None
    last: dict[str, datetime | None] = field(default_factory=lambda: {"heads_up": None, "loud": None})


def _segment(readings: list[tuple[datetime, float]]) -> tuple[int, list[tuple[datetime, float]]]:
    """The run evaluate() would use, and its spacing (same selection as serving.alerts.evaluate)."""
    seg = segment_tail(snap(readings))
    if len(seg) < 4:
        seg15 = segment_tail15(readings)
        if len(seg15) >= 2:
            return 15, seg15
    return 5, seg


class AlertStream:
    """Feed readings with `process`; it returns (status, alert-or-None)."""

    def __init__(self, retention: timedelta = RETENTION):
        self.retention = retention
        self.people: dict[str, _Person] = {}

    def process(self, patient_id: str, ts: datetime, glucose: float) -> tuple[str, dict | None]:
        p = self.people.setdefault(patient_id, _Person())
        times = [t for t, _ in p.readings]
        i = bisect.bisect_left(times, ts)
        if i < len(times) and times[i] == ts:
            return ("duplicate" if p.readings[i][1] == glucose else "conflict"), None
        newest = times[-1] if times else None
        if newest is not None and ts < newest - self.retention:
            return "too_old", None
        p.readings.insert(i, (ts, glucose))
        if newest is not None and ts < newest:
            return "late", None
        cutoff = ts - self.retention
        while p.readings and p.readings[0][0] < cutoff:
            p.readings.pop(0)
        return self._decide(patient_id, p)

    def _decide(self, patient_id: str, p: _Person) -> tuple[str, dict | None]:
        mode, seg = _segment(p.readings)
        if mode == 5 and p.newest_seg_time == seg[-1][0] and p.mode == 5:
            return "same_slot", None                      # snap kept the earlier reading in this slot
        need, lockout, flag_fn = (4, LOCKOUT, flags) if mode == 5 else (2, LOCKOUT15, flags15)
        continues = (p.mode == mode and len(seg) >= 2 and seg[-2][0] == p.newest_seg_time)
        if not continues:
            return self._recompute(patient_id, p, mode, seg, need)
        p.newest_seg_time = seg[-1][0]
        if len(seg) < need:
            return "ok", None
        x = np.array([g for _, g in seg], dtype=float)
        heads, loud, proj = flag_fn(x)
        k = len(x) - 1
        if not LO <= x[k] <= HI:
            return "ok", None                             # already out of range: a reading, not a warning
        seg_times = [t for t, _ in seg]

        def since(t: datetime | None) -> float:
            if t is None or t < seg_times[0]:
                return float("inf")                       # not in this continuous run: no quiet period
            return len(seg_times) - 1 - bisect.bisect_left(seg_times, t)

        tier = ("loud" if loud[k] and since(p.last["loud"]) >= lockout else
                "heads_up" if heads[k] and since(p.last["heads_up"]) >= lockout else None)
        if tier is None:
            return "ok", None
        p.last[tier] = seg_times[k]
        if tier == "loud":
            p.last["heads_up"] = seg_times[k]
        return "alert", self._alert(patient_id, seg_times[k], tier, float(x[k]), float(proj[k]), mode)

    def _recompute(self, patient_id: str, p: _Person, mode: int, seg, need: int) -> tuple[str, dict | None]:
        """After a gap or a 5/15-min switch: rebuild the chain over the whole run with evaluate() itself."""
        res = evaluate(p.readings)
        p.mode, p.newest_seg_time = mode, seg[-1][0]
        p.last = {"heads_up": None, "loud": None}
        for e in res["alerts_in_window"]:
            t = datetime.fromisoformat(e["timestamp"])
            p.last[e["tier"]] = t
            if e["tier"] == "loud":
                p.last["heads_up"] = t
        if not res["alert"]:
            return "ok", None
        now = res["alerts_in_window"][-1]
        return "alert", self._alert(patient_id, datetime.fromisoformat(now["timestamp"]), now["tier"], now["glucose"],
                                    now["projected_20min"], mode)

    @staticmethod
    def _alert(patient_id: str, t: datetime, tier: str, glucose: float, projected: float, mode: int) -> dict:
        return {"patient_id": patient_id, "time": t, "tier": tier,
                "direction": "high" if projected > HI else "low",
                "glucose": round(glucose, 1), "projected_20min": round(projected, 1), "spacing_min": mode}
