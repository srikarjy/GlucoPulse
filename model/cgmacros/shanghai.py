"""External test of the trend alarm on HIGHS: ShanghaiT2DM, people with type 2 diabetes not on insulin, CGM every 15 min.

    python -m model.cgmacros.shanghai

Data: Zhao et al., Sci Data 2023 (doi:10.1038/s41597-023-01940-7), figshare doi:10.6084/m9.figshare.21600933, CC BY 4.0
(zip MD5 4bfb61cfa506b48155fd9e841ee48e21), read through sources/external_cgm.parse_shanghai_t2dm. Never used for any
model choice.

PRE-REGISTERED (written 2026-11-16; before any alarm was run on these people; only event counts and the sensor-floor
counts per person were looked at):
  People      the recordings with no insulin anywhere (summary agents and every insulin column): 44 recordings,
              42 people, 548 days; 81 high and 162 low events by the count below. Readings outside 40-400 are clipped
              to the sensor limits (as for Dexcom Low/High in the other external sets).
  Spacing     the real 15-min readings; nothing is interpolated. Segments break at any step other than 15 min (+-2).
              Same clock durations as replay.py, in 15-min readings: event = >= 2 consecutive readings > 250 or < 70
              (>= 15 min, like 3 readings at 5 min), events of a kind < 30 min apart merge, warning window 60 min
              (4 readings), alarm lockout 30 min (2 readings), timely = first warning >= 15 min before onset.
  Alarm       alert20 at 15-min spacing: slope from the last two readings (the 15-min least-squares slope with two
              points), projected 20 min ahead; alarm if > 250 or < 70. Fixed, not tuned. Threshold (reactive) shown for
              reference. Scoring, false alarms and the rate-matched random baseline (20 seeds) are replay.py's.
  Primary     HIGH events: alert20 timely share minus its random baseline's, 95% person-bootstrap interval > 0.
  Secondary   lows (same test); false alarms per day; median lead; warned at all; excluding the two people with long
              runs at the sensor floor (shanghai-076, shanghai-078: 120 and 144 floor readings, possibly sensor
              failure); CGMacros context: alert20 timely on highs .636 at 5-min spacing (section 9).
  Not tested  forecasters and the triage rule (trained on 5-min data; running them here would need interpolation or a
              15-min retrain).
"""
from __future__ import annotations

import json
import random

import numpy as np
import pandas as pd

from model.cgmacros import replay as R
from model.cgmacros.dataset import ROOT

RAW = ROOT / "data" / "shanghai" / "raw"
RESULTS = ROOT / "model" / "results" / "cgmacros_shanghai_alarms.json"
FLOOR_PEOPLE = {"shanghai-076", "shanghai-078"}


def use_15min_constants() -> None:
    """replay.py reads these module constants at call time; set them to the same clock durations at 15-min spacing."""
    R.DT, R.SUSTAIN, R.LOCKOUT, R.WINDOW = 15.0, 2, 2, 4


def alert20_15(x: np.ndarray) -> np.ndarray:
    slope = np.r_[0.0, np.diff(x)] / 15.0              # first reading of a segment has no slope: no projection
    proj = x + slope * 20.0
    return (proj > R.HI) | (proj < R.LO)


def frames() -> list[tuple[str, pd.DataFrame]]:
    from sources.external_cgm import parse_shanghai_t2dm
    from sources.normalize import normalize
    out = []
    for rec, res in parse_shanghai_t2dm(RAW).items():
        person = rec.rsplit("-", 1)[0]
        ms = sorted(normalize(res.raws, tz="UTC").measurements, key=lambda m: m.ts)
        df = pd.DataFrame({"ts": [m.ts for m in ms], "glucose": [float(m.value) for m in ms]})
        step = df.ts.diff().dt.total_seconds().div(60)
        seg = (~step.between(13, 17)).cumsum()
        for _, f in df.groupby(seg):
            if len(f) >= 2:
                out.append((person, f.reset_index(drop=True)))
    return out


def kind_stats(per: dict, people: list[str], kind: str) -> dict:
    ev = {p: [e for e in per[p]["events"] if e["kind"] == kind] for p in people}
    rnd = {p: [[e for e in per[p]["rand"][i][0] if e["kind"] == kind] for i in range(R.SEEDS)] for p in people}

    def timely(ps):
        return R._timely([x for p in ps for x in ev[p]])

    def rand_timely(ps):
        return float(np.nanmean([R._timely([x for p in ps for x in rnd[p][i]]) for i in range(R.SEEDS)]))

    rng, d = random.Random(7), []
    for _ in range(2000):
        ps = [rng.choice(people) for _ in people]
        if any(ev[p] for p in ps):
            d.append(timely(ps) - rand_timely(ps))
    d.sort()
    all_ev = [x for p in people for x in ev[p]]
    leads = [x["lead"] for x in all_ev if x["lead"] > 0]
    return {"events": len(all_ev), "people_with_events": sum(bool(ev[p]) for p in people),
            "timely": round(timely(people), 3), "random_timely": round(rand_timely(people), 3),
            "warned_at_all": round(float(np.mean([x["lead"] > 0 for x in all_ev])), 3) if all_ev else None,
            "median_lead_min": float(np.median(leads)) if leads else None,
            "timely_minus_random_ci95": [round(d[int(0.025 * len(d))], 3), round(d[int(0.975 * len(d)) - 1], 3)]}


def evaluate(fr) -> dict:
    per = R.run(fr, {"alert20_15": lambda p, f: alert20_15(f["glucose"].to_numpy(float)),
                     "threshold": lambda p, f: R.threshold(f["glucose"].to_numpy(float))})
    people = sorted(per["alert20_15"])
    out = {}
    for name in per:
        s = R.summarize(per[name], people)
        out[name] = {"days": s["days"], "false_alarms_per_day": s["false_alarms_per_day"],
                     "random_false_alarms_per_day": s["random_false_alarms_per_day"],
                     "high": kind_stats(per[name], people, "high"), "low": kind_stats(per[name], people, "low")}
    return out


def main() -> int:
    use_15min_constants()
    fr = frames()
    res = {"preregistration": __doc__.split("PRE-REGISTERED")[1].strip(), "segments": len(fr),
           "people": len({p for p, _ in fr}), "all": evaluate(fr),
           "excluding_floor_people": evaluate([(p, f) for p, f in fr if p not in FLOOR_PEOPLE])}
    hi = res["all"]["alert20_15"]["high"]
    res["primary_high_timely_beats_random"] = bool(hi["timely_minus_random_ci95"][0] > 0)
    RESULTS.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "preregistration"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
