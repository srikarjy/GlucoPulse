"""5-minute replay: how many minutes of warning does each algorithm give before an urgent glucose event?

    python -m model.cgmacros.replay                                       # published algorithms, all 45 people
    CGMACROS_SPLIT=confirm python -m model.cgmacros.replay --build-rule   # rule every 5 min (14 people, slow)
    CGMACROS_SPLIT=confirm python -m model.cgmacros.replay --vs-rule      # rule vs rule + alert (14 people)

The triage evals scored one moment per 30 min. A CGM reads every 5 min, and most urgent misses were spikes from a
flat start that become visible once they begin. This replays every person's data reading by reading, as a live
system would, and measures warning lead time and false alarms per day.

PRE-REGISTERED (written 2026-11-16, before any algorithm below was run on CGMacros):

Events      within each gap-free segment, a run of >= 3 consecutive readings > 250 (high) or < 70 (low) mg/dL
            (the outcome_label definition). Onset = first reading of the run. Runs of the same kind less than 30 min
            apart are one event. An event is evaluable if its segment has >= 60 min of readings before onset.
Warning     an alarm at reading k warns event E if onset - 60 min <= k < onset, the reading at k is within 70-250,
            and k is not inside another event. Lead time = onset - first warning alarm. Primary: share of events
            warned >= 15 min ahead ("timely"). Alarms are not typed: any alarm can warn either kind of event.
False alarm alarm episodes start at an alarm reading >= 30 min after the previous episode start (one lockout for all
            algorithms). An episode start is a false alarm if no event onset follows within 60 min; starts outside
            70-250, inside an event or within 60 min after one, or with < 60 min of segment left, are not counted.
            Reported per day of monitored readings (gaps excluded).
Random      for each algorithm, alarms at random eligible readings at the same per-person rate (20 seeds, averaged):
            a method that alarms often "warns" more by chance, and this baseline measures how much.
Algorithms  (published parameters, nothing fitted to CGMacros; filters restart at every gap)
  threshold     alarm if the reading is > 250 or < 70 (reactive reference: its lead time is 0 by construction).
  alert20       CGM-style predictive alert: least-squares slope over the last 15 min (4 readings), projected 20 min
                ahead; alarm if the projection is > 250 or < 70. (Modelled on CGM predictive alerts; Dexcom's own
                method is not public and its urgent-low threshold is 55.)
  rate2         CGM-style rate alert: |slope over last 15 min| >= 2 mg/dL/min.
  grid          GRID, Harvey et al. 2014 (doi:10.1177/1932296814523881), its published tuning: noise-spike limit
                3 mg/dL/min (15 per 5-min sample), low-pass tau_F = 6 min, 3-point Lagrangian derivative
                (G[k-2] - 4 G[k-1] + 3 G[k]) / (2 dt), GRID+ if G_F > 130 and (last 3 rates > 1.5 or last 2 > 1.6).
  grid_healthy  GRID with the mean per-person tuning reported for healthy adults by Hochsmann et al. 2026
                (doi:10.1038/s41598-026-50699-5): tau_F 4.5, G_min 103.1, G'_3 0.8, G'_2 1.1. Secondary.
Primary 1   (all 45 people; no algorithm here is trained on CGMacros) alert20's timely share is higher than its
            random baseline, with the 95% person-bootstrap interval of the difference excluding 0.
Primary 2   (the 14 people the confirm-split forecaster never trained on: 9 new + 5 early-stopping) policy
            "rule + alert20" (alarm if the rule says urgent, evaluated every 5 min, or alert20 fires) versus the rule
            alone: (a) timely share higher, interval excluding 0 (the policy only adds alarms, so the gain is >= 0 by
            construction; the test is whether it is reliably above 0 across people); (b) false alarms at most 1.0 per
            day more than the rule.
Meal check  GRID is a MEAL detector, so it is also scored as one, against logged meals (carbs > 0; logs within 30
            min merged), with Hochsmann's matching: a detection (GRID+ rising edge) within 120 min after a meal start
            is a true positive, a 120-min lockout follows each, unmatched detections are false positives. No night
            or afternoon exclusions (CGMacros timestamps are shifted). Published reference on healthy adults: 70.4%
            sensitivity, 0.26 FP/day, 37 min. This is an implementation and population check, not a criterion.
Secondary   alert20 vs threshold/rate2/grid/grid_healthy; the event model (model/cgmacros/event.py) every 5 min on
            the 14; highs and lows separately; median lead time.
"""
from __future__ import annotations

import argparse
import json
import random
import zlib

import numpy as np
import pandas as pd

from model.cgmacros.cases import OUT
from model.cgmacros.dataset import ROOT, build_grid
from model.cgmacros.splits import SCHEME, split_participants
from model.cgmacros.train import ENCODER

DT = 5.0
HI, LO, SUSTAIN = 250.0, 70.0, 3
WINDOW, LOCKOUT, TIMELY_MIN = 12, 6, 15.0       # 60-min credit window and 30-min lockout (readings); 15-min timely
RULE_CACHE = OUT / f"replay_rule_{SCHEME}.csv"
RESULTS = ROOT / "model" / "results" / f"cgmacros_replay_{SCHEME}.json"
MAX_FA_INCREASE_PER_DAY = 1.0
SEEDS = 20


# --------------------------------------------------------------------------------------------- algorithms (causal)
def slope15(x: np.ndarray) -> np.ndarray:
    """Least-squares slope (mg/dL/min) over the last 4 readings; NaN until 4 readings exist."""
    out = np.full(len(x), np.nan)
    t = np.arange(4) * DT
    t = t - t.mean()
    for k in range(3, len(x)):
        w = x[k - 3:k + 1]
        out[k] = float(np.dot(t, w - w.mean()) / np.dot(t, t))
    return out


def threshold(x):
    return (x > HI) | (x < LO)


def alert20(x):
    proj = x + slope15(x) * 20.0
    return (proj > HI) | (proj < LO)            # NaN compares False: no alarm before 4 readings


def rate2(x):
    return np.abs(slope15(x)) >= 2.0


def grid(x, tau_f=6.0, g_min=130.0, r3=1.5, r2=1.6):
    n = len(x)
    ns, gf, d = np.empty(n), np.empty(n), np.full(n, np.nan)
    dg, a = 3.0 * DT, DT / (tau_f + DT)
    for k in range(n):
        if k == 0:
            ns[k] = gf[k] = x[k]
            continue
        step = x[k] - ns[k - 1]
        ns[k] = x[k] if abs(step) <= dg else ns[k - 1] + np.sign(step) * dg
        gf[k] = a * ns[k] + (1 - a) * gf[k - 1]
        if k >= 2:
            d[k] = (gf[k - 2] - 4 * gf[k - 1] + 3 * gf[k]) / (2 * DT)
    out = np.zeros(n, bool)
    for k in range(4, n):
        out[k] = gf[k] > g_min and (bool((d[k - 2:k + 1] > r3).all()) or bool((d[k - 1:k + 1] > r2).all()))
    return out


def grid_healthy(x):
    return grid(x, tau_f=4.5, g_min=103.1, r3=0.8, r2=1.1)


ALGORITHMS = {"threshold": threshold, "alert20": alert20, "rate2": rate2, "grid": grid, "grid_healthy": grid_healthy}


# ------------------------------------------------------------------------------------------------------ accounting
def segments(grid_df: pd.DataFrame, people: list[str]):
    """(participant, frame) per gap-free run of 5-min readings."""
    for p in people:
        g = grid_df[grid_df.participant == p].reset_index(drop=True)
        cut = np.flatnonzero(np.diff(g["time_idx"].to_numpy()) != 1) + 1
        for s in np.split(np.arange(len(g)), cut):
            if len(s):
                yield p, g.iloc[s].reset_index(drop=True)


def events(x: np.ndarray) -> list[dict]:
    found = []
    for kind, mask in (("high", x > HI), ("low", x < LO)):
        mine: list[dict] = []
        k = 0
        while k < len(x):
            if not mask[k]:
                k += 1
                continue
            j = k
            while j + 1 < len(x) and mask[j + 1]:
                j += 1
            if j - k + 1 >= SUSTAIN:
                if mine and k - mine[-1]["end"] < LOCKOUT:
                    mine[-1]["end"] = j
                else:
                    mine.append({"kind": kind, "onset": k, "end": j})
            k = j + 1
        found += mine
    return sorted(found, key=lambda e: e["onset"])


def score_segment(x: np.ndarray, alarm: np.ndarray, ev: list[dict]) -> dict:
    in_range = (x >= LO) & (x <= HI)
    inside, blocked = np.zeros(len(x), bool), np.zeros(len(x), bool)
    for e in ev:
        inside[e["onset"]:e["end"] + 1] = True
        blocked[e["onset"]:min(len(x), e["end"] + 1 + WINDOW)] = True
    warned = []
    for e in ev:
        if e["onset"] < WINDOW:
            continue
        ks = [k for k in range(e["onset"] - WINDOW, e["onset"]) if alarm[k] and in_range[k] and not inside[k]]
        warned.append({"kind": e["kind"], "lead": (e["onset"] - ks[0]) * DT if ks else 0.0})
    onsets = np.array([e["onset"] for e in ev], dtype=int)
    fa, last = 0, -10 ** 9
    for k in np.flatnonzero(alarm):
        if k - last < LOCKOUT:
            continue
        last = k
        if not in_range[k] or blocked[k] or k + WINDOW >= len(x):
            continue
        if not ((onsets > k) & (onsets <= k + WINDOW)).any():
            fa += 1
    eligible = in_range & ~blocked
    return {"events": warned, "fa": fa, "readings": len(x), "eligible": eligible,
            "alarm_on_eligible": int((alarm & eligible).sum())}


def run(frames: list[tuple[str, pd.DataFrame]], streams: dict) -> dict:
    """Per stream, per person: event leads, false alarms, readings, and SEEDS rate-matched random draws."""
    out = {}
    for name, fn in streams.items():
        per: dict[str, dict] = {}
        for p, f in frames:
            x = f["glucose"].to_numpy(dtype=float)
            ev = events(x)
            s = score_segment(x, np.asarray(fn(p, f), bool), ev)
            d = per.setdefault(p, {"events": [], "fa": 0, "readings": 0, "on": 0, "el": 0, "segs": []})
            d["events"] += s["events"]
            d["fa"] += s["fa"]
            d["readings"] += s["readings"]
            d["on"] += s["alarm_on_eligible"]
            d["el"] += int(s["eligible"].sum())
            d["segs"].append((x, ev, s["eligible"]))
        for p, d in per.items():
            rate = d["on"] / d["el"] if d["el"] else 0.0
            rng = np.random.default_rng(zlib.crc32(f"{name}|{p}".encode()))
            d["rand"] = []
            for _ in range(SEEDS):
                ev_all, fa = [], 0
                for x, ev, el in d["segs"]:
                    s = score_segment(x, el & (rng.random(len(x)) < rate), ev)
                    ev_all += s["events"]
                    fa += s["fa"]
                d["rand"].append((ev_all, fa))
            del d["segs"]
        out[name] = per
    return out


def _timely(ev: list[dict]) -> float:
    return float(np.mean([e["lead"] >= TIMELY_MIN for e in ev])) if ev else float("nan")


def summarize(per: dict, people: list[str] | None = None) -> dict:
    people = people or sorted(per)
    ev = [e for p in people for e in per[p]["events"]]
    days = sum(per[p]["readings"] for p in people) * DT / 1440
    rand_t = [_timely([e for p in people for e in per[p]["rand"][i][0]]) for i in range(SEEDS)]
    rand_fa = [sum(per[p]["rand"][i][1] for p in people) / days for i in range(SEEDS)]
    leads = [e["lead"] for e in ev if e["lead"] > 0]
    by = {k: [e for e in ev if e["kind"] == k] for k in ("high", "low")}
    return {"events": len(ev), "people_with_events": sum(bool(per[p]["events"]) for p in people),
            "days": round(days, 1), "timely_share": round(_timely(ev), 3),
            "random_timely_share": round(float(np.nanmean(rand_t)), 3),
            "warned_at_all": round(float(np.mean([e["lead"] > 0 for e in ev])), 3) if ev else None,
            "median_lead_min_if_warned": float(np.median(leads)) if leads else None,
            "false_alarms_per_day": round(sum(per[p]["fa"] for p in people) / days, 2),
            "random_false_alarms_per_day": round(float(np.mean(rand_fa)), 2),
            "high": {"n": len(by["high"]), "timely": round(_timely(by["high"]), 3)},
            "low": {"n": len(by["low"]), "timely": round(_timely(by["low"]), 3)}}


def boot(per_a: dict, per_b: dict | None, people: list[str], n: int = 2000) -> list[float]:
    """95% person-bootstrap interval of timely(a) - timely(b); with per_b None, b is a's own random baseline."""
    rng, draws = random.Random(7), []
    for _ in range(n):
        ps = [rng.choice(people) for _ in people]
        ev = [e for p in ps for e in per_a[p]["events"]]
        if not ev:
            continue
        if per_b is None:
            b = float(np.nanmean([_timely([e for p in ps for e in per_a[p]["rand"][i][0]]) for i in range(SEEDS)]))
        else:
            b = _timely([e for p in ps for e in per_b[p]["events"]])
        draws.append(_timely(ev) - b)
    draws.sort()
    return [round(draws[int(0.025 * len(draws))], 3), round(draws[int(0.975 * len(draws)) - 1], 3)]


# ------------------------------------------------------------------------------------------------------ meal check
def meal_check(frames, fn) -> dict:
    tp = missed = fp = readings = 0
    delays = []
    for _, f in frames:
        x = f["glucose"].to_numpy(dtype=float)
        det = np.asarray(fn(x), bool)
        edges = np.flatnonzero(det & ~np.r_[False, det[:-1]])
        rows = np.flatnonzero(f["carbs"].to_numpy() > 0)
        meals = [m for i, m in enumerate(rows) if i == 0 or m - rows[i - 1] > LOCKOUT]
        used: set[int] = set()
        for m in meals:
            hit = [e for e in edges if m <= e <= m + 24 and e not in used]
            if hit:
                tp += 1
                delays.append((hit[0] - m) * DT)
                used.update(e for e in edges if hit[0] <= e <= hit[0] + 24)      # 120-min lockout after a TP
            else:
                missed += 1
        fp += sum(1 for e in edges if e not in used)
        readings += len(x)
    days = readings * DT / 1440
    return {"meals": tp + missed, "sensitivity": round(tp / max(1, tp + missed), 3), "fp_per_day": round(fp / days, 2),
            "median_delay_min": float(np.median(delays)) if delays else None,
            "published_healthy_reference": {"sensitivity": 0.704, "fp_per_day": 0.26, "delay_min": 37.3}}


# ------------------------------------------------------------------------------------------------- rule, 5-minute
def build_rule(grid_df: pd.DataFrame, people: list[str]) -> pd.DataFrame:
    from agents.triage.rules import rule_level
    from model.cgmacros.scenarios import ScenarioForecaster, scenario_resource
    meal_model, hist_model = ScenarioForecaster("meals_activity"), ScenarioForecaster("glucose")
    rows = []
    for p, f in segments(grid_df, people):
        for k in range(ENCODER - 1, len(f)):
            h = f.iloc[k - ENCODER + 1:k + 1]
            as_of = h["ts"].iloc[-1].isoformat()
            q = {"no_new_meal": meal_model.forecast(h, ["no_new_meal"])["no_new_meal"],
                 "history_only": hist_model.forecast(h, ["no_new_meal"])["no_new_meal"]}
            sc = {n: scenario_resource(p, as_of, n, q[n], "glucose" if n == "history_only" else "meals_activity")
                  for n in q}
            rows.append({"participant": p, "time_idx": int(f["time_idx"].iloc[k]),
                         "rule": rule_level({"current_mg_dl": float(h["glucose"].iloc[-1])},
                                            sc["no_new_meal"], sc["history_only"])})
        print(f"{p}: {sum(r['participant'] == p for r in rows)} readings so far", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RULE_CACHE, index=False)
    return out


def event_model_stream():
    """The history-only event model (model/cgmacros/event.py), trained on the confirm-split training people, every
    5 min, with the same features and threshold as its pre-registered test."""
    from model.cgmacros.event import FEATURES, TRAIN_CACHE, fit_with_threshold, history_features
    model, t, _ = fit_with_threshold(pd.read_csv(TRAIN_CACHE))

    def stream(p, f):
        out = np.zeros(len(f), bool)
        if len(f) >= ENCODER:
            X = np.array([[history_features(f.iloc[k - ENCODER + 1:k + 1])[c] for c in FEATURES]
                          for k in range(ENCODER - 1, len(f))])
            out[ENCODER - 1:] = model.predict_proba(X)[:, 1] >= t
        return out
    return stream


def held_out_people(grid_df) -> tuple[list[str], list[str]]:
    _, val, test = split_participants(sorted(grid_df.participant.unique()))
    return sorted(test), sorted(val + test)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-rule", action="store_true")
    ap.add_argument("--vs-rule", action="store_true")
    args = ap.parse_args()
    grid_df = build_grid()
    if args.build_rule:
        if SCHEME != "confirm":
            raise SystemExit("--build-rule needs CGMACROS_SPLIT=confirm (a forecaster that never saw these people)")
        build_rule(grid_df, held_out_people(grid_df)[1])
        return 0
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    results["preregistration"] = __doc__.split("PRE-REGISTERED")[1].strip()
    if not args.vs_rule:
        people = sorted(grid_df.participant.unique())
        frames = list(segments(grid_df, people))
        per = run(frames, {n: (lambda p, f, fn=fn: fn(f["glucose"].to_numpy(dtype=float)))
                           for n, fn in ALGORITHMS.items()})
        res = {n: summarize(per[n]) for n in per}
        res["alert20"]["timely_minus_random_ci95"] = boot(per["alert20"], None, people)
        results["all_45_people"] = res
        results["primary_1_alert20_beats_random"] = bool(res["alert20"]["timely_minus_random_ci95"][0] > 0)
        results["meal_check"] = {n: meal_check(frames, ALGORITHMS[n]) for n in ("grid", "grid_healthy")}
    else:
        if SCHEME != "confirm" or not RULE_CACHE.exists():
            raise SystemExit("run --build-rule with CGMACROS_SPLIT=confirm first")
        new9, all14 = held_out_people(grid_df)
        frames = list(segments(grid_df, all14))
        r = pd.read_csv(RULE_CACHE)
        urgent = {(p, t) for p, t, lv in zip(r.participant, r.time_idx, r.rule) if lv == "urgent"}

        def rule_stream(p, f):
            return np.array([(p, t) in urgent for t in f["time_idx"]])

        streams = {"rule": rule_stream,
                   "rule+alert20": lambda p, f: rule_stream(p, f) | alert20(f["glucose"].to_numpy(dtype=float)),
                   "alert20": lambda p, f: alert20(f["glucose"].to_numpy(dtype=float)),
                   "event_model": event_model_stream()}
        per = run(frames, streams)
        res = {"rule_reading_coverage": round(len(r) / sum(len(f) for _, f in frames), 3)}
        for label, ps in (("new_9_people", new9), ("all_14_people", all14)):
            res[label] = {n: summarize(per[n], ps) for n in per}
            res[label]["rule+alert20_gain_ci95"] = boot(per["rule+alert20"], per["rule"], ps)
        a, b = res["all_14_people"]["rule+alert20"], res["all_14_people"]["rule"]
        res["primary_2"] = {"a_timely_gain_ci_excludes_0": bool(res["all_14_people"]["rule+alert20_gain_ci95"][0] > 0),
                            "b_false_alarm_increase_at_most_1_per_day": bool(
                                a["false_alarms_per_day"] - b["false_alarms_per_day"] <= MAX_FA_INCREASE_PER_DAY)}
        results["vs_rule"] = res
    RESULTS.write_text(json.dumps(results, indent=1, default=float) + "\n")
    print(json.dumps({k: v for k, v in results.items() if k != "preregistration"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
