"""5-minute monitoring loop: feed one CGM reading at a time, get an alarm when a low or high is coming.

Default policy: alert20. It warned of 68% of urgent events >= 15 min ahead on all 45 CGMacros people and of 72% of
lows on 73 people from two other studies, at about 2-3 false alarms/day. The event model is optional: on the external
lows it warned of only 19% (docs/TRIAGE.md section 15), so it is not trusted for lows.

A thin wrapper over pieces tested elsewhere (docs/TRIAGE.md sections 8, 9, 14, 15):
  event_model  P(urgent within 60 min) from the 2-h history (model/cgmacros/event.py), alarm if P >= its threshold
  alert20      CGM-style predictive alert: 15-min least-squares slope projected 20 min ahead, outside 70-250
               (model/cgmacros/replay.py)
An alarm opens an episode; another alarm is not raised for 30 min (the lockout used in every replay test). A gap in
the readings (anything but a 5-min step) clears the history, as the replay's gap-free segments do.

    m = Monitor()                                     # alert20; Monitor.from_confirm_model() adds the event model
    for row in readings:                              # dicts with ts, glucose, and optionally carbs, hr, mets
        alarm = m.step(row)
        if alarm: notify(alarm)
"""
from __future__ import annotations

from collections import deque
from datetime import timedelta

import numpy as np
import pandas as pd

from model.cgmacros.replay import DT, HI, LO, LOCKOUT, alert20, slope15
from model.cgmacros.train import ENCODER

STEP = timedelta(minutes=DT)


class Monitor:
    def __init__(self, model=None, threshold: float | None = None, policy: tuple[str, ...] = ("alert20",)):
        unknown = set(policy) - {"event_model", "alert20"}
        if unknown:
            raise ValueError(f"unknown policy {sorted(unknown)}")
        if "event_model" in policy and (model is None or threshold is None):
            raise ValueError("policy event_model needs a fitted model and its threshold")
        self.model, self.threshold, self.policy = model, threshold, policy
        self.buf: deque[dict] = deque(maxlen=ENCODER)
        self.since_alarm = LOCKOUT                     # readings since the last alarm episode began

    @classmethod
    def from_confirm_model(cls, policy: tuple[str, ...] = ("alert20", "event_model")) -> "Monitor":
        """The event model as pre-registered: fit on the 31 confirm-training people, threshold from their CV."""
        from model.cgmacros.event import TRAIN_CACHE, fit_with_threshold
        from model.cgmacros.splits import SCHEME
        if SCHEME != "confirm":
            raise RuntimeError("load the confirm-split event model with CGMACROS_SPLIT=confirm")
        model, t, _ = fit_with_threshold(pd.read_csv(TRAIN_CACHE))
        return cls(model, t, policy)

    def step(self, row: dict) -> dict | None:
        ts = pd.Timestamp(row["ts"])
        if self.buf and ts - self.buf[-1]["ts"] != STEP:
            self.buf.clear()                          # gap: start a new segment
        self.buf.append({"ts": ts, "glucose": float(row["glucose"]), "carbs": float(row.get("carbs", 0.0) or 0.0),
                         "hr": float(row.get("hr", 70.0)), "hr_observed": bool(row.get("hr_observed", False)),
                         "mets": float(row.get("mets", 1.0)), "hour": ts.hour + ts.minute / 60.0})
        self.since_alarm += 1
        g = np.array([r["glucose"] for r in self.buf])
        if not LO <= g[-1] <= HI:
            return None                               # already low/high: that is a current, not a coming, event
        reasons, info = [], {}
        if "alert20" in self.policy and len(g) >= 4 and bool(alert20(g)[-1]):
            proj = float(g[-1] + slope15(g)[-1] * 20)
            reasons.append(f"trend: projected {proj:.0f} mg/dL in 20 min")
            info["projected_20min"] = round(proj, 1)
        if "event_model" in self.policy and len(g) == ENCODER:
            from model.cgmacros.event import FEATURES, history_features
            f = history_features(pd.DataFrame(list(self.buf)))
            p = float(self.model.predict_proba(np.array([[f[c] for c in FEATURES]]))[0, 1])
            info["p_urgent_60min"] = round(p, 3)
            if p >= self.threshold:
                reasons.append(f"event model: P(low or high within 60 min) = {p:.2f}")
        if not reasons or self.since_alarm < LOCKOUT:
            return None
        self.since_alarm = 0
        return {"ts": ts.isoformat(), "glucose": float(g[-1]), "reasons": reasons, **info,
                "note": "screening alert from CGM data, not a diagnosis or treatment advice"}
