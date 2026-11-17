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

DEV RESULT (11 trials, model/results/alarm_log.jsonl): no variant met the target; every stream-only rule traded timely
share for false alarms at about the same rate. The best trade was "gate" (high alarm only if glucose >= 200, low only
if <= 100): .637 timely at 1.88 false alarms/day vs alert20's .677 at 2.37.

PRE-REGISTERED TEST (written 2026-11-16 after the dev trials, before "gate" was run on any external person; run with
--test):
  Arms        alert20 vs gate, unchanged from dev. 5-min data: the 73 external people (BIG IDEAs + Hall; third use).
              15-min data: ShanghaiT2DM non-insulin (second use), both arms translated as in shanghai.py (two-reading
              slope), the gate on the current reading.
  Primary     on the 73, both of: (1) fewer false alarms: 95% person-bootstrap interval of (alert20 - gate) false
              alarms/day > 0; (2) timely share non-inferior: lower bound of (gate - alert20) > -0.10.
  Secondary   lows and highs separately; ShanghaiT2DM with the same two criteria (highs reported separately).
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
        shifted = np.zeros_like(a)
        if len(a) > k:
            shifted[k:] = a[:-k]
        out &= shifted                                # segments shorter than n can never confirm
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
    # round 2: confirmation costs 5 min of lead per reading, so look further ahead to win it back
    "confirm2_p30": make(confirm=2, minutes=30.0),
    "confirm2_gate_p30": make(confirm=2, hi_gate=200, lo_gate=100, minutes=30.0),
}


def run_trial(name: str, note: str = "") -> dict:
    grid = build_grid()
    people = sorted(grid.participant.unique())
    frames = list(R.segments(grid, people))
    fn = VARIANTS[name]
    res = {"trial": name, "when": datetime.now(timezone.utc).isoformat(timespec="seconds"), "note": note}
    try:
        per = R.run(frames, {name: lambda p, f: fn(f["glucose"].to_numpy(float))})[name]
        s = R.summarize(per, people)
        res.update({"timely": s["timely_share"], "random_timely": s["random_timely_share"],
                    "false_alarms_per_day": s["false_alarms_per_day"], "warned_at_all": s["warned_at_all"],
                    "median_lead": s["median_lead_min_if_warned"], "high": s["high"], "low": s["low"],
                    "events": s["events"]})
    except Exception as e:                            # failures are logged too
        res["error"] = f"{type(e).__name__}: {e}"
    with open(LOG, "a") as fh:
        fh.write(json.dumps(res) + "\n")
    return res


def _compare(frames, a_fn, b_fn, people=None) -> dict:
    per = R.run(frames, {"alert20": lambda p, f: a_fn(f["glucose"].to_numpy(float)),
                         "gate": lambda p, f: b_fn(f["glucose"].to_numpy(float))})
    ps = people or sorted(per["alert20"])
    out = {n: R.summarize(per[n], ps) for n in per}
    out["gate_minus_alert20_timely_ci95"] = R.boot(per["gate"], per["alert20"], ps)
    rng, d = np.random.default_rng(0), []
    for _ in range(2000):
        pick = rng.choice(ps, len(ps))
        days = sum(per["alert20"][p]["readings"] for p in pick) * R.DT / 1440
        d.append((sum(per["alert20"][p]["fa"] for p in pick) - sum(per["gate"][p]["fa"] for p in pick)) / days)
    out["alert20_minus_gate_fa_per_day_ci95"] = [round(float(np.percentile(d, 2.5)), 2),
                                                 round(float(np.percentile(d, 97.5)), 2)]
    out["criterion_1_fewer_false_alarms"] = bool(out["alert20_minus_gate_fa_per_day_ci95"][0] > 0)
    out["criterion_2_timely_noninferior"] = bool(out["gate_minus_alert20_timely_ci95"][0] > -0.10)
    return out


def test() -> dict:
    import pandas as pd

    from model.cgmacros import shanghai as S
    ext = pd.read_pickle(ROOT / "data" / "bigideas" / "external_grid.pkl")
    res = {"preregistration": __doc__.split("PRE-REGISTERED TEST")[1].strip(),
           "external_73": _compare(list(R.segments(ext, sorted(ext.participant.unique()))), VARIANTS["alert20"],
                                   VARIANTS["gate"])}
    S.use_15min_constants()

    def gate15(x):
        a = S.alert20_15(x)
        slope = np.r_[0.0, np.diff(x)] / 15.0
        p = x + slope * 20.0
        return a & (((p > R.HI) & (x >= 200)) | ((p < R.LO) & (x <= 100)))
    res["shanghai_t2dm_15min"] = _compare(S.frames(), S.alert20_15, gate15)
    e = res["external_73"]
    res["confirmed"] = e["criterion_1_fewer_false_alarms"] and e["criterion_2_timely_noninferior"]
    (ROOT / "model" / "results" / "cgmacros_alarm_gate_test.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", nargs="*", default=[])
    ap.add_argument("--note", default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    if a.test:
        r = test()
        print(json.dumps({k: v for k, v in r.items() if k != "preregistration"}, indent=1, default=str))
    if a.list:
        print(", ".join(VARIANTS))
    for t in a.trial:
        r = run_trial(t, a.note)
        if "error" in r:
            print(t, "ERROR", r["error"])
            continue
        print(f"{t:24s} timely {r['timely']} (random {r['random_timely']}) FA/day {r['false_alarms_per_day']} "
              f"lead {r['median_lead']} high {r['high']} low {r['low']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
