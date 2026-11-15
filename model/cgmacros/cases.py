"""Build triage evaluation cases from held-out CGMacros participants.

Each case is a moment `as_of` in someone's data. Two files, kept apart on purpose:
  packets.json  what an agent may see: summaries of the 2 h BEFORE as_of and
                scenario forecasts for the next hour. Built from history only.
  labels.json   what actually happened in the next 60 min (+ pool sizes and the
                participant per case, for prevalence and per-person reporting).

Outcome label (adults with prediabetes/T2D/no diabetes, no insulin). Events must be
SUSTAINED for >= 15 min (3 consecutive 5-min readings), following CGM consensus event
definitions, so one sensor artifact cannot make a case urgent:
  urgent   sustained < 70 or sustained > 250 mg/dL in the next 60 min
  monitor  sustained > 180, or any single reading < 70 (and not urgent)
  routine  otherwise
These thresholds are a research choice that needs clinical sign-off.

Sampling: candidates every 30 min; at most --cap cases per participant per label, so one
person cannot dominate a class; case ids are shuffled so --limit is not stratified by label.

    python -m model.cgmacros.cases --per-label 40 --cap 10
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from model.cgmacros.dataset import ROOT, build_grid
from model.cgmacros.scenarios import MENU, ScenarioForecaster, scenario_resource
from model.cgmacros.splits import split_participants
from model.cgmacros.train import ENCODER, HORIZON

OUT = ROOT / "data" / "cgmacros" / "triage"
SUSTAIN = 3   # consecutive 5-min readings = 15 min


def _sustained(mask: np.ndarray) -> bool:
    run = 0
    for v in mask:
        run = run + 1 if v else 0
        if run >= SUSTAIN:
            return True
    return False


def outcome_label(future: np.ndarray) -> str:
    if _sustained(future < 70) or _sustained(future > 250):
        return "urgent"
    if _sustained(future > 180) or (future < 70).any():
        return "monitor"
    return "routine"


def _rid(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def summaries(patient: str, as_of: str, hist: pd.DataFrame) -> dict:
    g = hist["glucose"].to_numpy()
    last30 = hist.tail(6)
    meals = hist[hist.carbs > 0]
    last_meal = None
    if len(meals):
        m = meals.iloc[-1]
        last_meal = {"minutes_ago": int((hist["ts"].iloc[-1] - m["ts"]).total_seconds() // 60),
                     "carbs_g": round(float(m["carbs"]), 1), "protein_g": round(float(m["protein"]), 1),
                     "fat_g": round(float(m["fat"]), 1), "fiber_g": round(float(m["fiber"]), 1)}
    hr_obs = float(last30["hr_observed"].mean()) if "hr_observed" in last30 else 0.0
    return {
        "glucose_summary": {
            "resourceType": "GlucoseSummary", "id": "gs" + _rid(patient, as_of, "glucose")[:14], "as_of": as_of,
            "window": "previous two hours of Dexcom G6 readings", "current_mg_dl": round(float(g[-1]), 1),
            "change_last_15min_mg_dl": round(float(g[-1] - g[-4]), 1), "min_2h_mg_dl": round(float(g.min()), 1),
            "max_2h_mg_dl": round(float(g.max()), 1), "readings_2h": int(len(g))},
        "lifestyle_summary": {
            "resourceType": "LifestyleSummary", "id": "ls" + _rid(patient, as_of, "lifestyle")[:14], "as_of": as_of,
            "last_meal_in_2h": last_meal, "carbs_logged_2h_g": round(float(hist["carbs"].sum()), 1),
            "mean_heart_rate_30min_bpm": round(float(last30["hr"].mean()), 1) if hr_obs > 0 else None,
            "heart_rate_observed_fraction_30min": round(hr_obs, 2),
            "mean_mets_30min": round(float(last30["mets"].mean()), 2)},
        "data_quality": {
            "resourceType": "DataQuality", "id": "dq" + _rid(patient, as_of, "quality")[:14], "as_of": as_of,
            "expected_readings_2h": ENCODER, "received_readings_2h": int(len(g)),
            "heart_rate_observed_fraction_30min": round(hr_obs, 2),
            "meal_log": "self-reported; unlogged meals are invisible to every forecast"},
    }


def candidate_moments(grid: pd.DataFrame, ids: list[str], stride: int = 6):
    for p in ids:
        g = grid[grid.participant == p].reset_index(drop=True)
        t = g["time_idx"].to_numpy()
        for i in range(ENCODER - 1, len(g) - HORIZON, stride):
            if t[i] - t[i - ENCODER + 1] == ENCODER - 1 and t[i + HORIZON] - t[i] == HORIZON:
                yield p, g, i


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-label", type=int, default=40)
    ap.add_argument("--cap", type=int, default=10, help="max cases per participant per label")
    ap.add_argument("--variant", default="meals_activity")   # most accurate in the ablation
    args = ap.parse_args()
    grid = build_grid()
    _, _, test_ids = split_participants(sorted(grid.participant.unique()))
    pools: dict[str, list] = {"urgent": [], "monitor": [], "routine": []}
    for p, g, i in candidate_moments(grid, test_ids):
        pools[outcome_label(g["glucose"].to_numpy()[i + 1:i + 1 + HORIZON])].append((p, g, i))
    rng = random.Random(11)
    chosen = []
    for label, pool in pools.items():
        rng.shuffle(pool)
        per_person: Counter = Counter()
        taken = 0
        for p, g, i in pool:
            if taken >= args.per_label:
                break
            if per_person[p] < args.cap:
                per_person[p] += 1
                taken += 1
                chosen.append((label, p, g, i))
    rng.shuffle(chosen)                          # ids no longer sorted by label
    meal_model, history_model = ScenarioForecaster(args.variant), ScenarioForecaster("glucose")
    packets, labels = {}, {"_meta": {"pool_sizes": {k: len(v) for k, v in pools.items()},
                                     "cap_per_participant_per_label": args.cap, "sustain_minutes": SUSTAIN * 5}}
    flags = 0
    for k, (label, p, g, i) in enumerate(chosen):
        hist = g.iloc[i - ENCODER + 1:i + 1]
        as_of = hist["ts"].iloc[-1].isoformat()
        q = meal_model.forecast(hist, MENU)
        q["history_only"] = history_model.forecast(hist, ["no_new_meal"])["no_new_meal"]
        scen = {n: scenario_resource(p, as_of, n, q[n], args.variant if n != "history_only" else "glucose")
                for n in q}
        base = q["no_new_meal"]["p50"]
        for n in ("meal_30g_now", "meal_60g_now"):
            if any(m < b - 1.0 for m, b in zip(q[n]["p50"], base)):
                scen[n]["consistency_warning"] = "forecast with this meal dips below the no-meal forecast"
                flags += 1
        packets[f"case-{k:03d}"] = {"patient": p, "as_of": as_of, **summaries(p, as_of, hist), "scenarios": scen}
        future = g["glucose"].to_numpy()[i + 1:i + 1 + HORIZON]
        labels[f"case-{k:03d}"] = {"label": label, "participant": p, "actual_min_60": round(float(future.min()), 1),
                                   "actual_max_60": round(float(future.max()), 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "packets.json").write_text(json.dumps(packets, indent=1))
    (OUT / "labels.json").write_text(json.dumps(labels, indent=1))
    counts = Counter(v["label"] for c, v in labels.items() if c != "_meta")
    people = defaultdict(Counter)
    for c, v in labels.items():
        if c != "_meta":
            people[v["label"]][v["participant"]] += 1
    print(f"{len(packets)} cases {dict(counts)} pools {labels['_meta']['pool_sizes']} "
          f"meal-scenario consistency warnings {flags}")
    print({lab: dict(c) for lab, c in people.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
