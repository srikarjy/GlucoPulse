"""Alarm round 3: fewer loud false alarms (5-min) and a loud tier for 15-minute (Libre) data.

    python -m model.cgmacros.alarms_r3 --fa-dev      # A: dev trials on 45 CGMacros (5-min Dexcom)
    python -m model.cgmacros.alarms_r3 --fa-test     # A: chosen arm vs shipped loud rule on CGMND adults (first use)
    python -m model.cgmacros.alarms_r3 --l15-dev     # B: dev trials on 45 CGMacros (15-min Libre Pro)
    python -m model.cgmacros.alarms_r3 --l15-test    # B: chosen arm on ShanghaiT2DM non-insulin (15-min)

PRE-REGISTERED (written 2026-11-18, before any trial of these arms on any data). Looked at beforehand: CGMND adult
counts only (107 people, 869 days, 6168 gap-free segments after 5-min snapping, 433 low events from 79 people with
>= 1 h of history, 114 of them 00:00-06:00, 0 high events); CGMacros Libre coverage (45 people, median 10.3 days, 99.9%
of steps 13-17 min). Scoring is replay.py unchanged (events, timely = warned >= 15 min ahead within 60 min, false
alarm episodes, rate-matched random baseline); every trial is appended to model/results/alarm_log.jsonl (tag r3_).

A. FEWER LOUD FALSE ALARMS (5-min). The tier being reduced is the SHIPPED loud rule, BASE = alarms.ESC["esc_fast"]
   (gate <= 100 / >= 200 or |slope| >= 1.5, projection out of range, 2 readings in a row). Arms must work in the
   stateless tool (only the request window, no per-person history):
     A1 depth   the same rule, but the 20-min projection must be past the line by 10 mg/dL (< 60 or > 260)
     A2 three   the same rule, 3 readings in a row instead of 2
   Dev (45 CGMacros): eligible if timely >= BASE timely - 0.03 AND false alarms/day <= BASE - 0.2. Choose the
   eligible arm with the lowest false alarms/day; within 0.05/day -> higher timely; still tied -> A1. None eligible
   -> stop: the rule does not change and CGMND is not opened for A.
   Test (CGMND adults, first and only use for alarms; healthy people, lows only): both
     1. 95% person-bootstrap lower bound of FA/day(BASE) - FA/day(arm) > 0;
     2. 95% person-bootstrap lower bound of timely(arm) - timely(BASE) > -0.05 (lows; 433 events / 79 people).
   Secondary: excluding events 00:00-06:00 (sensor-compression lows); median lead. CGMND's absolute rate is NOT
   comparable to the 73's 2.3/day and is not judged against the 1.5/day target. Pass -> serving/alerts.py adopts it.

B. LOUD TIER AT 15 MIN. Comparator: the shipped 15-min heads-up (shanghai.alert20_15). 15-min scoring constants as in
   model/cgmacros/shanghai.py (event = 2 readings, lockout 2, window 4, timely >= 15 min).
     B1 one     heads-up AND (now <= 100 heading low, or >= 200 heading high, or |slope| >= 1.5) on 1 reading
     B2 two     the same on 2 readings in a row (adds 15 min of delay)
   Dev (45 CGMacros, Libre Pro readings recovered by sources/cgmacros.py, runs of 13-17-min steps): eligible if
   false alarms/day <= 0.7 x heads-up's AND the 95% person-bootstrap interval of timely minus its random baseline is
   > 0. Choose the higher timely; within 0.02 -> the one with fewer false alarms. None -> stop.
   Test (ShanghaiT2DM non-insulin, 42 people; their second alarm use): primary, the 95% person-bootstrap lower bound
   of timely(arm) - its random baseline > 0 (highs and lows together). Secondary: per kind; false alarms/day vs the
   heads-up. Pass -> the 15-min loud tier is enabled in serving/alerts.py.
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from model.cgmacros import replay as R
from model.cgmacros.alarms import ESC, LOG, _consecutive, _proj
from model.cgmacros.dataset import ROOT, build_grid

OUT = ROOT / "model" / "results"
BASE = ESC["esc_fast"]

# DEV RESULTS and choices are written here after the dev runs, before any test run.
# A (2026-11-18, 45 CGMacros): BASE timely .511 at 1.67 FA/day (reproduces round 2); A1_depth .345 at 0.92;
# A2_three .269 at 0.95. Both cut ~0.7 FA/day but lose far more than 0.03 timely: NONE eligible -> stop. The loud rule
# is unchanged and CGMND was not opened for A.
A_CHOSEN: str | None = None
# B (2026-11-18, 45 CGMacros Libre Pro, 493 low / 81 high events): HEADS15 .725 timely at 1.94 FA/day; B1_one
# IDENTICAL to HEADS15 (design flaw seen only now: at 15-min spacing a projection out of range from >= 100 already
# implies |slope| >= 1.5, so B1 filters nothing); B2_two .209 timely (random .054, CI of the gain .117-.205) at 0.59
# FA/day <= 0.7 x 1.94. Only B2 eligible -> chosen, before --l15-test. It is a weak loud tier (late confirmation).
B_CHOSEN: str | None = "B2_two"


# ------------------------------------------------------------------ A arms (5-min)
def make_loud(margin: float = 0.0, confirm: int = 2):
    def alarm(x: np.ndarray) -> np.ndarray:
        p, sl = _proj(x), np.abs(R.slope15(x))
        lo, hi = p < R.LO - margin, p > R.HI + margin
        fast = (lo | hi) & (sl >= 1.5)
        return (_consecutive(hi & (x >= 200), confirm) | _consecutive(lo & (x <= 100), confirm)
                | _consecutive(fast, confirm))
    return alarm                                     # make_loud(0, 2) is BASE exactly (checked in fa_dev)


A_ARMS = {"A1_depth": make_loud(margin=10.0), "A2_three": make_loud(confirm=3)}


# ------------------------------------------------------------------ B arms (15-min)
def loud15(x: np.ndarray, confirm: int) -> np.ndarray:
    sl = np.r_[0.0, np.diff(x)] / 15.0
    p = x + sl * 20.0
    lo, hi = p < R.LO, p > R.HI
    cond = (lo & ((x <= 100) | (np.abs(sl) >= 1.5))) | (hi & ((x >= 200) | (np.abs(sl) >= 1.5)))
    return _consecutive(cond, confirm)


B_ARMS = {"B1_one": lambda x: loud15(x, 1), "B2_two": lambda x: loud15(x, 2)}


def heads15(x: np.ndarray) -> np.ndarray:
    from model.cgmacros.shanghai import alert20_15
    return alert20_15(x)


# ------------------------------------------------------------------ data
def cgmacros_frames():
    g = build_grid()
    people = sorted(g.participant.unique())
    return people, list(R.segments(g, people))


def cgmnd_frames():
    from model.cgmacros.dataset import grid_from_measurements
    from sources.external_cgm import parse_cgmnd
    from sources.normalize import normalize
    cache = ROOT / "data" / "cgmnd" / "grid.pkl"
    if cache.exists():
        g = pd.read_pickle(cache)
    else:
        g = pd.concat([grid_from_measurements(p, normalize(r.raws, tz="UTC").measurements, "dexcom-g6")
                       for p, r in parse_cgmnd(ROOT / "data" / "cgmnd" / "raw").items()], ignore_index=True)
        g.to_pickle(cache)
    people = sorted(g.participant.unique())
    return people, list(R.segments(g, people))


def libre_frames():
    """CGMacros Libre Pro native readings, split into runs of 13-17-min steps."""
    from sources.cgmacros import parse_cgmacros
    from sources.model import Kind
    from sources.normalize import normalize
    out = []
    for f in sorted((ROOT / "data" / "cgmacros" / "csv").glob("CGMacros-0*.csv")):
        ms = [m for m in normalize(parse_cgmacros(f).raws, tz="UTC").measurements
              if m.kind is Kind.GLUCOSE_CGM and m.device == "libre-pro"]
        df = pd.DataFrame({"ts": [m.ts for m in ms], "glucose": [min(max(float(m.value), 40.0), 400.0) for m in ms]})
        df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
        step = df.ts.diff().dt.total_seconds().div(60)
        seg = (~step.between(13, 17)).cumsum()
        for _, s in df.groupby(seg):
            if len(s) >= 2:
                s = s.reset_index(drop=True)
                s["hour"] = pd.to_datetime(s.ts).dt.hour
                out.append((f.stem.lower(), s))
    return sorted({p for p, _ in out}), out


def use_15(on: bool) -> None:
    R.DT, R.SUSTAIN, R.LOCKOUT, R.WINDOW = (15.0, 2, 2, 4) if on else (5.0, 3, 6, 12)


# ------------------------------------------------------------------ scoring helpers
def stream(fn):
    return lambda p, f: fn(f["glucose"].to_numpy(float))


def fa_diff_ci(per_a: dict, per_b: dict, people: list[str], n: int = 2000) -> list[float]:
    """95% person-bootstrap interval of FA/day(a) - FA/day(b)."""
    rng, d = random.Random(7), []
    for _ in range(n):
        ps = [rng.choice(people) for _ in people]
        days = sum(per_a[p]["readings"] for p in ps) * R.DT / 1440
        d.append((sum(per_a[p]["fa"] for p in ps) - sum(per_b[p]["fa"] for p in ps)) / days)
    d.sort()
    return [round(d[int(0.025 * n)], 3), round(d[int(0.975 * n) - 1], 3)]


def tag_night(frames, per: dict) -> None:
    """Mark each scored event with whether its onset is 00:00-06:00 (score_segment appends events in this order)."""
    flags: dict[str, list[bool]] = {p: [] for p in per}
    for p, f in frames:
        x = f["glucose"].to_numpy(float)
        for e in R.events(x):
            if e["onset"] >= R.WINDOW:
                flags[p].append(int(f["hour"].iloc[e["onset"]]) < 6)
    for p, d in per.items():
        assert len(flags[p]) == len(d["events"]), "event order mismatch"
        for e, fl in zip(d["events"], flags[p]):
            e["night"] = fl


def night_free(per: dict) -> dict:
    return {p: {**d, "events": [e for e in d["events"] if not e.get("night")]} for p, d in per.items()}


def log(entry: dict) -> None:
    entry = {"when": datetime.now(timezone.utc).isoformat(timespec="seconds"), **entry}
    with open(LOG, "a") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")


# ------------------------------------------------------------------ runs
def fa_dev() -> dict:
    use_15(False)
    people, frames = cgmacros_frames()
    same = make_loud()
    assert all(np.array_equal(same(f["glucose"].to_numpy(float)), BASE(f["glucose"].to_numpy(float)))
               for _, f in frames), "arms must differ from BASE only by their parameter"
    per = R.run(frames, {"BASE": stream(BASE), **{k: stream(v) for k, v in A_ARMS.items()}})
    res = {k: R.summarize(per[k], people) for k in per}
    b = res["BASE"]
    elig = [k for k in A_ARMS if res[k]["timely_share"] >= b["timely_share"] - 0.03
            and res[k]["false_alarms_per_day"] <= b["false_alarms_per_day"] - 0.2]
    elig.sort(key=lambda k: res[k]["false_alarms_per_day"])
    if len(elig) == 2 and res[elig[1]]["false_alarms_per_day"] - res[elig[0]]["false_alarms_per_day"] <= 0.05:
        elig.sort(key=lambda k: (-res[k]["timely_share"], k))
    out = {"dev": res, "eligible": elig, "would_choose": elig[0] if elig else None}
    for k in per:
        log({"trial": f"r3_A_dev_{k}", "timely": res[k]["timely_share"], "fa_per_day": res[k]["false_alarms_per_day"]})
    (OUT / "alarm_r3_A_dev.json").write_text(json.dumps(out, indent=1))
    return out


def fa_test() -> dict:
    assert A_CHOSEN in A_ARMS, "record the dev choice in A_CHOSEN first"
    use_15(False)
    people, frames = cgmnd_frames()
    per = R.run(frames, {"BASE": stream(BASE), A_CHOSEN: stream(A_ARMS[A_CHOSEN])})
    for k in per:
        tag_night(frames, per[k])
    res = {k: R.summarize(per[k], people) for k in per}
    fa_ci = fa_diff_ci(per["BASE"], per[A_CHOSEN], people)
    t_ci = R.boot(per[A_CHOSEN], per["BASE"], people)
    nf = {k: night_free(per[k]) for k in per}
    out = {"preregistration": "model/cgmacros/alarms_r3.py, A", "arm": A_CHOSEN, "people": len(people), **res,
           "fa_reduction_ci95": fa_ci, "timely_diff_ci95": t_ci,
           "criterion_1_fewer_false_alarms": bool(fa_ci[0] > 0),
           "criterion_2_timely_noninferior": bool(t_ci[0] > -0.05),
           "secondary_excluding_night": {k: R.summarize(nf[k], people) for k in nf}}
    out["confirmed"] = out["criterion_1_fewer_false_alarms"] and out["criterion_2_timely_noninferior"]
    log({"trial": f"r3_A_test_{A_CHOSEN}", "confirmed": out["confirmed"], "fa_ci": fa_ci, "timely_ci": t_ci})
    (OUT / "alarm_r3_A_test.json").write_text(json.dumps(out, indent=1, default=str))
    return out


def l15_dev() -> dict:
    use_15(True)
    try:
        people, frames = libre_frames()
        per = R.run(frames, {"HEADS15": stream(heads15), **{k: stream(v) for k, v in B_ARMS.items()}})
        res = {k: R.summarize(per[k], people) for k in per}
        for k in B_ARMS:
            res[k]["timely_minus_random_ci95"] = R.boot(per[k], None, people)
        h = res["HEADS15"]["false_alarms_per_day"]
        elig = [k for k in B_ARMS if res[k]["false_alarms_per_day"] <= 0.7 * h
                and res[k]["timely_minus_random_ci95"][0] > 0]
        elig.sort(key=lambda k: -res[k]["timely_share"])
        if len(elig) == 2 and res[elig[0]]["timely_share"] - res[elig[1]]["timely_share"] <= 0.02:
            elig.sort(key=lambda k: res[k]["false_alarms_per_day"])
        out = {"dev": res, "eligible": elig, "would_choose": elig[0] if elig else None}
        for k in per:
            log({"trial": f"r3_B_dev_{k}", "timely": res[k]["timely_share"],
                 "fa_per_day": res[k]["false_alarms_per_day"]})
        (OUT / "alarm_r3_B_dev.json").write_text(json.dumps(out, indent=1))
        return out
    finally:
        use_15(False)


def l15_test() -> dict:
    assert B_CHOSEN in B_ARMS, "record the dev choice in B_CHOSEN first"
    from model.cgmacros.shanghai import frames as sh_frames
    use_15(True)
    try:
        fr = sh_frames()
        people = sorted({p for p, _ in fr})
        per = R.run(fr, {"HEADS15": stream(heads15), B_CHOSEN: stream(B_ARMS[B_CHOSEN])})
        res = {k: R.summarize(per[k], people) for k in per}
        ci = R.boot(per[B_CHOSEN], None, people)
        out = {"preregistration": "model/cgmacros/alarms_r3.py, B", "arm": B_CHOSEN, "people": len(people), **res,
               "timely_minus_random_ci95": ci, "confirmed": bool(ci[0] > 0)}
        log({"trial": f"r3_B_test_{B_CHOSEN}", "confirmed": out["confirmed"], "ci": ci})
        (OUT / "alarm_r3_B_test.json").write_text(json.dumps(out, indent=1, default=str))
        return out
    finally:
        use_15(False)


def main() -> int:
    ap = argparse.ArgumentParser()
    for f in ("--fa-dev", "--fa-test", "--l15-dev", "--l15-test"):
        ap.add_argument(f, action="store_true")
    a = ap.parse_args()
    for flag, fn in (("fa_dev", fa_dev), ("fa_test", fa_test), ("l15_dev", l15_dev), ("l15_test", l15_test)):
        if getattr(a, flag):
            r = fn()
            print(json.dumps({k: v for k, v in r.items() if k != "preregistration"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
