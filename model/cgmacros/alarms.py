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


# ================================================================ PERSONAL + TIERS PROTOCOL (2026-11-16, before any trial)
# 1. Per-person thresholds: each person's first 2 days are a calibration period (no alarms scored, for every arm);
#    from them the person's own |15-min slope| distribution is taken, and afterwards a trend alarm only fires if the
#    current |slope| is also at or above the person's q-th percentile (jittery sensors / people need a steeper trend).
#    Arms are scored on the SAME post-calibration frames, including alert20 itself (so the comparison is fair).
# 2. Two tiers: "heads-up" = alert20 (or its personal version) and "alarm" = a strict rule (gate + 2 readings in a row);
#    reported per tier: the strict tier's false alarms/day and timely share, and the heads-up tier's.
# Dev = all 45 CGMacros people; trials tagged pers_/tier_ in alarm_log.jsonl; at most 10. Target unchanged (<= 1.0
# false alarms/day at >= .60 timely), judged on the single alarm for (1) and on the strict tier for (2).
CAL_DAYS = 2.0


def personal_frames(grid, people, cal_days=CAL_DAYS):
    """Post-calibration frames per person and each person's calibration |slope15| sample (readings before day 2)."""
    import pandas as pd
    cal, out = {}, []
    for p in people:
        g = grid[grid.participant == p]
        cut = g["ts"].min() + pd.Timedelta(days=cal_days)
        slopes = []
        for _, f in R.segments(g[g.ts < cut], [p]):
            x = f["glucose"].to_numpy(float)
            if len(x) >= 4:
                slopes.append(np.abs(R.slope15(x)[3:]))
        cal[p] = np.concatenate(slopes) if slopes else np.array([])
        out += list(R.segments(g[g.ts >= cut], [p]))
    return out, cal


def make_personal(q, cal, base=None):
    base = base or make()

    def stream(p, f):
        x = f["glucose"].to_numpy(float)
        a = base(x)
        sample = cal.get(p, np.array([]))
        if len(sample) < 50:                       # too little calibration data: fall back to the plain alarm
            return a
        return a & (np.abs(R.slope15(x)) >= np.percentile(sample, q))
    return stream


def run_personal(names: list[str], note: str = "") -> list[dict]:
    grid = build_grid()
    people = sorted(grid.participant.unique())
    frames, cal = personal_frames(grid, people)
    strict = make(confirm=2, hi_gate=200, lo_gate=100)
    streams = {"pers_alert20_postcal": lambda p, f: make()(f["glucose"].to_numpy(float)),
               "pers_q50": make_personal(50, cal), "pers_q75": make_personal(75, cal),
               "pers_q90": make_personal(90, cal),
               "pers_q75_gate": make_personal(75, cal, make(hi_gate=200, lo_gate=100)),
               "tier_strict_postcal": lambda p, f: strict(f["glucose"].to_numpy(float)),
               "tier_strict_personal_q75": make_personal(75, cal, strict)}
    rows = []
    for n in names:
        res = {"trial": n, "when": datetime.now(timezone.utc).isoformat(timespec="seconds"), "note": note}
        try:
            per = R.run(frames, {n: streams[n]})[n]
            s = R.summarize(per, sorted(per))
            res.update({"timely": s["timely_share"], "random_timely": s["random_timely_share"],
                        "false_alarms_per_day": s["false_alarms_per_day"], "median_lead": s["median_lead_min_if_warned"],
                        "high": s["high"], "low": s["low"], "events": s["events"], "days": s["days"]})
        except Exception as e:
            res["error"] = f"{type(e).__name__}: {e}"
        with open(LOG, "a") as fh:
            fh.write(json.dumps(res) + "\n")
        rows.append(res)
    return rows


# DEV RESULT (personal + tiers, 7 trials): per-person slope thresholds did not beat the trade-off (q75: .599 timely at
# 1.98/day vs gate .637 at 1.88) and are dropped. The strict tier (gate + 2 readings in a row) reached .471 timely at
# 1.19/day. Descriptive: of alert20's "false" alarms on CGMacros, 25% were followed by a near-high (220-250), 30% by a
# near-low (70-85) and 14% by a large move; 31% by nothing much (strict tier: 14% nothing much).
#
# PRE-REGISTERED TIERS TEST (written 2026-11-16, before either tier was run on any external person for this question;
# run with --tiers-test). Data: the 73 external people at 5-min spacing (BIG IDEAs + Hall; fourth use), all days.
#   heads-up  = alert20; loud = gate (high >= 200, low <= 100) + 2 readings in a row. Fixed from dev.
#   Primary   loud tier: false alarms <= 1.5/day AND timely share >= 0.40.
#   Secondary heads-up tier timely and false alarms; what follows each tier's false alarms (near-high, near-low,
#             large move, nothing much); ShanghaiT2DM (15 min) with loud = gate only, because a 2-reading confirmation
#             at 15-min spacing costs 15 min of a lead that is usually 15 min.
def false_alarm_kinds(frames, fn) -> dict:
    kinds = dict.fromkeys(["near_high_220_250", "near_low_70_85", "large_move", "nothing_much"], 0)
    for _, f in frames:
        x = f["glucose"].to_numpy(float)
        ev, a = R.events(x), fn(x)
        blocked = np.zeros(len(x), bool)
        for e in ev:
            blocked[e["onset"]:min(len(x), e["end"] + 1 + R.WINDOW)] = True
        onsets, last = np.array([e["onset"] for e in ev], int), -10 ** 9
        for k in np.flatnonzero(a):
            if k - last < R.LOCKOUT:
                continue
            last = k
            if not (R.LO <= x[k] <= R.HI) or blocked[k] or k + R.WINDOW >= len(x):
                continue
            if ((onsets > k) & (onsets <= k + R.WINDOW)).any():
                continue
            nxt = x[k + 1:k + 1 + R.WINDOW]
            key = ("near_high_220_250" if nxt.max() >= 220 else "near_low_70_85" if nxt.min() <= 85 else
                   "large_move" if (nxt.max() - x[k] >= 40 or x[k] - nxt.min() >= 30) else "nothing_much")
            kinds[key] += 1
    n = sum(kinds.values())
    return {"false_alarms": n, **{k: round(v / n, 3) if n else None for k, v in kinds.items()}}


def _on_glucose(fn):
    return lambda p, f: fn(f["glucose"].to_numpy(float))


def tiers_test() -> dict:
    import pandas as pd

    from model.cgmacros import shanghai as S
    ext = pd.read_pickle(ROOT / "data" / "bigideas" / "external_grid.pkl")
    frames = list(R.segments(ext, sorted(ext.participant.unique())))
    tiers = {"heads_up": make(), "loud": make(confirm=2, hi_gate=200, lo_gate=100)}
    per = R.run(frames, {n: _on_glucose(fn) for n, fn in tiers.items()})
    out = {"preregistration": "see model/cgmacros/alarms.py, PRE-REGISTERED TIERS TEST", "external_73": {}}
    for n in tiers:
        out["external_73"][n] = {**R.summarize(per[n], sorted(per[n])), "after_false_alarm": false_alarm_kinds(
            frames, tiers[n])}
    loud = out["external_73"]["loud"]
    out["criterion_loud_fa_at_most_1.5"] = bool(loud["false_alarms_per_day"] <= 1.5)
    out["criterion_loud_timely_at_least_0.40"] = bool(loud["timely_share"] >= 0.40)
    out["confirmed"] = out["criterion_loud_fa_at_most_1.5"] and out["criterion_loud_timely_at_least_0.40"]
    S.use_15min_constants()
    sh = S.frames()

    def loud15(x):
        slope = np.r_[0.0, np.diff(x)] / 15.0
        p = x + slope * 20.0
        return ((p > R.HI) & (x >= 200)) | ((p < R.LO) & (x <= 100))
    per15 = R.run(sh, {"heads_up": lambda p, f: S.alert20_15(f["glucose"].to_numpy(float)),
                       "loud_gate_only": lambda p, f: loud15(f["glucose"].to_numpy(float))})
    out["shanghai_15min"] = {n: R.summarize(per15[n], sorted(per15[n])) for n in per15}
    (ROOT / "model" / "results" / "cgmacros_alarm_tiers_test.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


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


# ================================================================ FAST-FALL ESCALATION PROTOCOL (2026-11-17, before any trial)
# Problem   the loud tier needs glucose already <= 100 (lows) / >= 200 (highs) on two readings in a row, so a FAST move
#           from further away only gets a heads-up even when its projection is deeper: a real-Claude review of the MCP
#           tool found 160 -> 100 at 2 mg/dL/min returned heads_up (projection 60) while a slow 105 -> 81 got loud.
# Arms      loud tier = strict (gate + 2 in a row, unchanged) OR an escalation confirmed on 2 readings, with no gate:
#             esc_sev   projection < 55 (lows; 54 = level-2 hypoglycaemia) or > 300 (highs)
#             esc_fast  projection outside 70-250 and |15-min slope| >= 1.5 mg/dL/min
#             esc_both  either
#           Comparator: strict (confirm2_gate) on the same frames.
# Dev       all 45 CGMacros people, scored by replay.py unchanged; at most 4 trials, each appended to alarm_log.jsonl.
#           Choose the arm with the highest timely share whose false alarms/day are <= strict + 0.5; none -> stop.
# Test      (run with --esc-test) the chosen arm vs strict on the 73 external people at 5-min spacing (BIG IDEAs + Hall;
#           fifth use). Primary, both: (1) 95% person-bootstrap lower bound of timely(arm) - timely(strict) > 0;
#           (2) false alarms/day increase (point estimate) <= 0.5. Secondary: lows alone, median lead, heads-up tier
#           unchanged. If not confirmed, the alarm rule is not changed.
def make_esc(sev: bool, fast: bool):
    strict = make(confirm=2, hi_gate=200, lo_gate=100)

    def alarm(x: np.ndarray) -> np.ndarray:
        p, sl = _proj(x), np.abs(R.slope15(x))
        e = np.zeros(len(x), bool)
        if sev:
            e |= _consecutive(p < 55, 2) | _consecutive(p > 300, 2)
        if fast:
            e |= _consecutive(((p < R.LO) | (p > R.HI)) & (sl >= 1.5), 2)
        return strict(x) | e
    return alarm


ESC = {"esc_sev": make_esc(True, False), "esc_fast": make_esc(False, True), "esc_both": make_esc(True, True)}
VARIANTS.update(ESC)
# DEV RESULT (4 trials, 45 CGMacros people, loud tier): strict .475 timely at 1.30 FA/day; esc_sev .480 at 1.35;
# esc_fast .511 at 1.67; esc_both .511 at 1.67 (identical). Rule picks the highest timely within 1.80 FA/day: esc_fast
# and esc_both tie; the protocol had no tie-break, so the simpler arm (esc_fast) is taken. Chosen before --esc-test.
ESC_CHOSEN = "esc_fast"


def esc_test() -> dict:
    import pandas as pd
    assert ESC_CHOSEN in ESC, "choose the arm from the dev trials first"
    ext = pd.read_pickle(ROOT / "data" / "bigideas" / "external_grid.pkl")
    people = sorted(ext.participant.unique())
    frames = list(R.segments(ext, people))
    arms = {"strict": make(confirm=2, hi_gate=200, lo_gate=100), ESC_CHOSEN: ESC[ESC_CHOSEN]}
    per = R.run(frames, {n: _on_glucose(fn) for n, fn in arms.items()})
    out = {"preregistration": "model/cgmacros/alarms.py, FAST-FALL ESCALATION PROTOCOL", "arm": ESC_CHOSEN,
           **{n: R.summarize(per[n], people) for n in arms}}
    out["timely_diff_ci95"] = R.boot(per[ESC_CHOSEN], per["strict"], people)
    out["fa_increase_per_day"] = round(out[ESC_CHOSEN]["false_alarms_per_day"] - out["strict"]["false_alarms_per_day"], 2)
    out["criterion_1_more_timely"] = bool(out["timely_diff_ci95"][0] > 0)
    out["criterion_2_fa_increase_at_most_0.5"] = bool(out["fa_increase_per_day"] <= 0.5)
    out["confirmed"] = out["criterion_1_more_timely"] and out["criterion_2_fa_increase_at_most_0.5"]
    (ROOT / "model" / "results" / "cgmacros_alarm_esc_test.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", nargs="*", default=[])
    ap.add_argument("--note", default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--personal", nargs="*", default=[])
    ap.add_argument("--tiers-test", action="store_true")
    ap.add_argument("--esc-test", action="store_true")
    a = ap.parse_args()
    if a.esc_test:
        r = esc_test()
        print(json.dumps({k: v for k, v in r.items() if k != "preregistration"}, indent=1, default=str))
    if a.tiers_test:
        r = tiers_test()
        print(json.dumps({k: v for k, v in r.items() if k != "preregistration"}, indent=1, default=str))
    for r in run_personal(a.personal, a.note) if a.personal else []:
        print(r["trial"], "ERROR " + r["error"] if "error" in r else
              f"timely {r['timely']} (random {r['random_timely']}) FA/day {r['false_alarms_per_day']} "
              f"lead {r['median_lead']} high {r['high']} low {r['low']} days {r['days']}")
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
