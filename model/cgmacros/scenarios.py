"""What-if forecasts from the CGMacros forecaster.

A scenario changes only the KNOWN-FUTURE inputs for the next hour (planned meal,
planned activity) and re-runs the same model on the same 2 h of history. The
output is a model-based association, not a causal estimate or advice.

MENU is fixed on purpose: agents choose among these, so every number they can
cite was produced here, and forecasts can be cached for evaluation.
"""
from __future__ import annotations

import hashlib
import pickle
from typing import Iterable

import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet

from model.cgmacros.train import ART, ENCODER, HORIZON, MemSafeTFT

STEP_MIN = 5
MEALS = {"meal_30g_now": {"carbs": 30.0, "protein": 10.0, "fat": 10.0, "fiber": 3.0},
         "meal_60g_now": {"carbs": 60.0, "protein": 15.0, "fat": 15.0, "fiber": 5.0}}
# Activity what-ifs were evaluated and are deliberately NOT offered: the variant with planned
# METs as a known-future input moved the forecast by about -0.5 mg/dL for a 30-minute walk
# (model/results/cgmacros_ablation.json), i.e. it learned no usable activity effect.
MENU = {
    "no_new_meal": "no meal in the next hour (meals are known-future inputs, so this assumes none is eaten)",
    "meal_30g_now": "30 g carbohydrate meal (10 g protein, 10 g fat, 3 g fiber) starting now",
    "meal_60g_now": "60 g carbohydrate meal (15 g protein, 15 g fat, 5 g fiber) starting now",
}


# Unconditional forecast from the glucose-only model (no meal inputs at all): the rule uses it
# alongside no_new_meal so an unlogged or upcoming meal is not simply assumed away.
DESCRIPTIONS = {**MENU, "history_only": "glucose-history-only model: no assumption about meals"}


def future_frame(history: pd.DataFrame, scenario: str, group: str) -> pd.DataFrame:
    """history: the last ENCODER contiguous grid rows of one person. Returns
    history + HORIZON decoder rows carrying the scenario's known-future inputs."""
    if scenario not in DESCRIPTIONS:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {list(MENU)}")
    h = history.tail(ENCODER).copy()
    last = h.iloc[-1]
    steps = np.arange(1, HORIZON + 1)
    ts = last["ts"] + pd.to_timedelta(steps * STEP_MIN, unit="min")
    fut = pd.DataFrame({
        "participant": group, "time_idx": last["time_idx"] + steps, "ts": ts,
        "glucose": last["glucose"], "delta": 0.0, "hr": last["hr"], "hr_observed": False, "mets": 1.0,
        "cal_act": 0.0,
        "carbs": 0.0, "protein": 0.0, "fat": 0.0, "fiber": 0.0,
        "hour": ts.hour + ts.minute / 60.0, "dow": ts.dayofweek.astype(float),
    })
    for col, v in MEALS.get(scenario, {}).items():
        fut.loc[fut.index[0], col] = v
    h["participant"] = group
    return pd.concat([h, fut[h.columns]], ignore_index=True)


class ScenarioForecaster:
    def __init__(self, variant: str = "meals_activity"):
        with open(ART / f"{variant}.dataset.pkl", "rb") as fh:
            self.training = pickle.load(fh)
        self.model = MemSafeTFT.load_from_checkpoint(ART / f"{variant}.ckpt", map_location="cpu")
        self.model.eval()
        self.variant = variant

    def forecast(self, history: pd.DataFrame, scenarios: Iterable[str]) -> dict[str, dict]:
        names = list(scenarios)
        frames = [future_frame(history, n, f"s{i}") for i, n in enumerate(names)]
        ds = TimeSeriesDataSet.from_dataset(self.training, pd.concat(frames, ignore_index=True), predict=True,
                                            stop_randomization=True)
        out = self.model.predict(ds.to_dataloader(train=False, batch_size=64, num_workers=0), mode="raw", trainer_kwargs={"accelerator": "cpu", "logger": False},
                                 return_index=True)
        q = self.model.loss.to_quantiles(out.output.prediction).numpy()
        levels = self.model.loss.quantiles
        lo, mid, hi = levels.index(0.1), levels.index(0.5), levels.index(0.9)
        by_group = {g: i for i, g in enumerate(out.index["participant"])}
        def r(row):
            return [round(float(v), 1) for v in row]
        return {n: {"p10": r(q[by_group[f"s{i}"], :, lo]), "p50": r(q[by_group[f"s{i}"], :, mid]),
                    "p90": r(q[by_group[f"s{i}"], :, hi])} for i, n in enumerate(names)}


def _first_minutes(values: list[float], test) -> int | None:
    for i, v in enumerate(values):
        if test(v):
            return (i + 1) * STEP_MIN
    return None


def scenario_resource(patient: str, as_of: str, name: str, q: dict, variant: str) -> dict:
    """A citable record of one scenario forecast (resourceType/id like FHIR, so the
    citation check can verify an agent only cites forecasts it actually retrieved)."""
    rid = "sf" + hashlib.sha256(f"{patient}|{as_of}|{name}|{variant}".encode()).hexdigest()[:14]
    at = {m: {k: q[k][m // STEP_MIN - 1] for k in ("p10", "p50", "p90")} for m in (30, 60)}
    return {
        "resourceType": "ScenarioForecast", "id": rid, "scenario": name, "description": DESCRIPTIONS[name],
        "as_of": as_of, "model": f"cgmacros-tft-{variant}",
        "added_meal_carbs_g": MEALS.get(name, {}).get("carbs", 0.0),
        "t+30": at[30], "t+60": at[60],
        "lowest_p10_next_60min": round(min(q["p10"]), 1), "highest_p90_next_60min": round(max(q["p90"]), 1),
        "lowest_p50_next_60min": round(min(q["p50"]), 1), "highest_p50_next_60min": round(max(q["p50"]), 1),
        "minutes_until_p50_below_70": _first_minutes(q["p50"], lambda v: v < 70),
        "minutes_until_p50_above_180": _first_minutes(q["p50"], lambda v: v > 180),
        "label": "model-based scenario, not a causal estimate or medical advice",
    }
