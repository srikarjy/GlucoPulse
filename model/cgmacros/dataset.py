"""5-minute modelling grid for CGMacros, built from the sources/ pipeline.

Rows are recovered Dexcom G6 readings (native 5-minute sensor; Libre's 15-minute
readings are not the target). Covariates per row at bin time t:
  past-observed  hr, mets, cal_act: Fitbit minutes in [t-5min, t) only, so a row
                 never sees its own future; no Fitbit logged -> resting values
  known-future   carbs, protein, fat, fiber: a logged meal lands on the first row
                 at or after it (a *planned* meal is what a what-if supplies)
                 hour, dow: calendar
Gaps in the CGM stay gaps (TimeSeriesDataSet allow_missing_timesteps).

    python -m model.cgmacros.dataset      # builds and caches data/cgmacros/grid.pkl
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from sources.cgmacros import parse_cgmacros
from sources.model import Kind, Measurement
from sources.normalize import normalize

ROOT = Path(__file__).resolve().parents[2]
CSV_DIR = ROOT / "data" / "cgmacros" / "csv"
CACHE = ROOT / "data" / "cgmacros" / "grid.pkl"
BIN = pd.Timedelta(minutes=5)
MEALS = {Kind.MEAL_CARBS: "carbs", Kind.MEAL_PROTEIN: "protein", Kind.MEAL_FAT: "fat", Kind.MEAL_FIBER: "fiber"}
FITBIT = {Kind.HEART_RATE: "hr", Kind.METS: "mets", Kind.ACTIVE_CALORIES: "cal_act"}
RESTING = {"mets": 1.0, "cal_act": 0.0, "hr": 70.0}
COLUMNS = ["participant", "time_idx", "ts", "glucose", "delta", "hr", "hr_observed", "mets", "cal_act",
           "carbs", "protein", "fat", "fiber", "hour", "dow"]


def grid_from_measurements(participant: str, ms: Iterable[Measurement]) -> pd.DataFrame:
    ms = list(ms)
    glucose = [(m.ts, float(m.value)) for m in ms if m.kind is Kind.GLUCOSE_CGM and m.device == "dexcom-g6"]
    if not glucose:
        return pd.DataFrame(columns=COLUMNS)
    g = pd.DataFrame(glucose, columns=["ts", "glucose"])
    g["ts"] = pd.to_datetime(g["ts"], utc=True).dt.floor("5min")
    g = g.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    g["time_idx"] = ((g["ts"] - g["ts"].iloc[0]) / BIN).astype(int)
    g["delta"] = g["glucose"].diff().fillna(0.0)

    for kind, col in FITBIT.items():
        f = pd.DataFrame([(m.ts, float(m.value)) for m in ms if m.kind is kind and m.device == "fitbit"],
                         columns=["ts", col])
        if f.empty:
            g[col] = RESTING[col]
            if col == "hr":
                g["hr_observed"] = False
            continue
        f["ts"] = pd.to_datetime(f["ts"], utc=True).dt.floor("5min") + BIN   # minutes [t-5, t) feed row t
        g = g.merge(f.groupby("ts", as_index=False)[col].mean(), on="ts", how="left")
        if col == "hr":
            g["hr_observed"] = g[col].notna()          # imputed values are flagged, never shown as measured
            g[col] = g[col].ffill(limit=3)
            # past-only fallback: the running median up to each row (a whole-series median would use the future)
            g[col] = g[col].fillna(g[col].expanding().median()).fillna(RESTING[col])
        else:
            g[col] = g[col].fillna(RESTING[col])

    meals = pd.DataFrame([(m.ts, MEALS[m.kind], float(m.value)) for m in ms if m.kind in MEALS],
                         columns=["meal_ts", "col", "value"])
    for col in MEALS.values():
        g[col] = 0.0
    if not meals.empty:
        meals["meal_ts"] = pd.to_datetime(meals["meal_ts"], utc=True)
        meals = meals.sort_values("meal_ts")
        rows = g[["ts"]].rename(columns={"ts": "row_ts"}).sort_values("row_ts")
        placed = pd.merge_asof(meals, rows, left_on="meal_ts", right_on="row_ts", direction="forward",
                               tolerance=pd.Timedelta(minutes=30)).dropna(subset=["row_ts"])
        sums = placed.groupby(["row_ts", "col"])["value"].sum().unstack(fill_value=0.0)
        for col in sums.columns:
            g = g.merge(sums[[col]].rename(columns={col: f"_{col}"}), left_on="ts", right_index=True, how="left")
            g[col] = g[f"_{col}"].fillna(0.0)
            g = g.drop(columns=[f"_{col}"])

    g["participant"] = participant
    g["hour"] = g["ts"].dt.hour.astype(float) + g["ts"].dt.minute / 60.0
    g["dow"] = g["ts"].dt.dayofweek.astype(float)
    return g[COLUMNS]


def build_grid(paths: Optional[list[Path]] = None, use_cache: bool = True) -> pd.DataFrame:
    if paths is None and use_cache and CACHE.exists():
        return pd.read_pickle(CACHE)
    files = paths or sorted(CSV_DIR.glob("CGMacros-0*.csv"))
    if not files:
        raise FileNotFoundError(f"no CGMacros CSVs in {CSV_DIR} (see docs/SOURCES.md for download)")
    frames = []
    for path in files:
        result = normalize(parse_cgmacros(path).raws, tz="UTC")
        frames.append(grid_from_measurements(path.stem.lower(), result.measurements))
    grid = pd.concat(frames, ignore_index=True)
    if paths is None:
        grid.to_pickle(CACHE)
    return grid


if __name__ == "__main__":
    grid = build_grid(use_cache=False)
    print(f"{len(grid)} rows, {grid.participant.nunique()} participants, "
          f"{int((grid.carbs > 0).sum())} meal rows -> {CACHE}")
    sys.exit(0)
