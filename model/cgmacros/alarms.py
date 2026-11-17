"""Fewer false alarms from the CGM trend alarm, without missing more events.

    python -m model.cgmacros.alarms --trial confirm2        # dev trial on CGMacros, appended to model/results/alarm_log.jsonl
    python -m model.cgmacros.alarms --list

PROTOCOL (written 2026-11-16, before the first trial):
  Problem     alert20 (15-min least-squares slope projected 20 min, alarm outside 70-250) warns of 68% of urgent events
              >= 15 min ahead on CGMacros but raises ~2.4 false alarms/day (3.2 on the external 73); people tune out.
  Dev         all 45 CGMacros people (alert20 and every variant here are fixed rules with no fitted parameters, so no
              CGMacros person is held out for them). Scoring is replay.py's, unchanged: events, timely >= 15 min,
              30-min lockout, false alarms per day, rate-matched random baseline. Every trial, including failures,
              appends a line to model/results/alarm_log.jsonl. At most 12 trials.
  Variants    built only from the stream itself (no models): confirmation over consecutive readings, wider projection
              margins, a level gate (high alarms only above a level, low alarms only below one), and combinations.
  Target      false alarms <= 1.0/day with timely share >= 0.60 (alert20: 2.37 and .677 on all 45).
  Test        the chosen variant is then pre-registered against alert20 on the 73 external people at 5-min spacing
              (BIG IDEAs + Hall; their third use, mostly lows) and, translated to 15-min spacing, on ShanghaiT2DM
              (second use, highs). Nothing about the test sets is looked at before that pre-registration.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np

from model.cgmacros import replay as R
from model.cgmacros.dataset import ROOT, build_grid

LOG = ROOT / "model" / "results" / "alarm_log.jsonl"


def _proj(x: np.ndarray, minutes: float = 20.0) -> np.ndarray:
    return x + R.slope15(x) * minutes


def _consecutive(a: np.ndarray, n: int) -> np.ndarray:
    """True where a has been true for the last n readings."""
    out = a.copy()
    for k in range(1, n):
        out &= np.r_[np.zeros(k, bool), a[:-k]]
    return out


def make(hi_margin=0.0, lo_margin=0.0, confirm=1, hi_gate=None, lo_gate=None, minutes=20.0):
    """alert20 with optional: projection margins (alarm if proj > 250 + hi_margin or < 70 - lo_margin), confirmation
    over `confirm` consecutive readings, and level gates (high alarm only if now >= hi_gate, low only if now <= lo_gate)."""
    def alarm(x: np.ndarray) -> np.ndarray:
        p = _proj(x, minutes)
        hi = p > R.HI + hi_margin
        lo = p < R.LO - lo_margin
        if hi_gate is not None:
            hi &= x >= hi_gate
        if lo_gate is not None:
            lo &= x <= lo_gate
        return _consecutive(hi, confirm) | _consecutive(lo, confirm)
    return alarm


VARIANTS = {
    "alert20": make(),
    "confirm2": make(confirm=2),
    "margin10": make(hi_margin=10, lo_margin=5),
    "gate": make(hi_gate=200, lo_gate=100),
    "confirm2_gate": make(confirm=2, hi_gate=200, lo_gate=100),
    "confirm2_margin10": make(confirm=2, hi_margin=10, lo_margin=5),
    "confirm3": make(confirm=3),
    "confirm2_gate_margin10": make(confirm=2, hi_margin=10, lo_margin=5, hi_gate=200, lo_gate=100),
}


def run_trial(name: str, note: str = "") -> dict:
    grid = build_grid()
    people = sorted(grid.participant.unique())
    frames = list(R.segments(grid, people))
    fn = VARIANTS[name]
    per = R.run(frames, {name: lambda p, f: fn(f["glucose"].to_numpy(float))})[name]
    s = R.summarize(per, people)
    res = {"trial": name, "when": datetime.now(timezone.utc).isoformat(timespec="seconds"), "note": note,
           "timely": s["timely_share"], "random_timely": s["random_timely_share"],
           "false_alarms_per_day": s["false_alarms_per_day"], "warned_at_all": s["warned_at_all"],
           "median_lead": s["median_lead_min_if_warned"], "high": s["high"], "low": s["low"], "events": s["events"]}
    with open(LOG, "a") as fh:
        fh.write(json.dumps(res) + "\n")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", nargs="*", default=[])
    ap.add_argument("--note", default="")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        print(", ".join(VARIANTS))
    for t in a.trial:
        r = run_trial(t, a.note)
        print(f"{t:24s} timely {r['timely']} (random {r['random_timely']}) FA/day {r['false_alarms_per_day']} "
              f"lead {r['median_lead']} high {r['high']} low {r['low']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
