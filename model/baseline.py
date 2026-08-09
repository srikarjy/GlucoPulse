"""
Persistence baseline: predict glucose at T+30/T+60 as "same as now".

Replaces the old OhioT1DM-era 15-25 mg/dL T+30 figure, which docs/STATUS.md
already flags as not carrying over to AZT1D -- this measures it fresh.

Exact-timestamp matching (not a fixed row-offset): cgm_features is a 5-min
grid but rows with a null glucose_value were dropped at write time (see
spark/feature_job.py), so a patient's rows aren't guaranteed evenly spaced.
Joining on time + horizon and keeping only exact matches means every error
here is a real "value now" vs "real value that many minutes later" pair,
never a stale/interpolated stand-in.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from db import load_features
from splits import TEST_PATIENTS, TRAIN_PATIENTS, VAL_PATIENTS

HORIZONS_MIN = (30, 60)
RESULTS_PATH = Path(__file__).parent / "results" / "baseline_metrics.json"


def persistence_errors(df: pd.DataFrame, patients: list[str], horizon_min: int) -> pd.DataFrame:
    sub = df[df["patient_id"].isin(patients)]
    frames = []
    for pid, g in sub.groupby("patient_id"):
        g = g.sort_values("time")
        cur = g[["time", "glucose_value"]].rename(columns={"glucose_value": "pred"})
        fut = g[["time", "glucose_value"]].copy()
        fut["time"] = fut["time"] - pd.Timedelta(minutes=horizon_min)
        fut = fut.rename(columns={"glucose_value": "actual"})
        merged = cur.merge(fut, on="time", how="inner")
        merged["patient_id"] = pid
        frames.append(merged)
    if not frames:
        return pd.DataFrame(columns=["time", "pred", "actual", "patient_id"])
    return pd.concat(frames, ignore_index=True)


def rmse(err: pd.Series) -> float:
    return float(np.sqrt(np.mean(err**2)))


def mae(err: pd.Series) -> float:
    return float(np.mean(np.abs(err)))


def summarize(errors: pd.DataFrame) -> dict:
    err = errors["actual"] - errors["pred"]
    out = {"n_pairs": int(len(errors)), "rmse": rmse(err), "mae": mae(err), "per_patient": {}}
    for pid, g in errors.groupby("patient_id"):
        e = g["actual"] - g["pred"]
        out["per_patient"][pid] = {"n_pairs": int(len(g)), "rmse": rmse(e), "mae": mae(e)}
    return out


def main():
    df = load_features()
    results = {"splits": {"train": TRAIN_PATIENTS, "val": VAL_PATIENTS, "test": TEST_PATIENTS}, "horizons": {}}

    for horizon in HORIZONS_MIN:
        horizon_results = {}
        for split_name, patients in (("train", TRAIN_PATIENTS), ("val", VAL_PATIENTS), ("test", TEST_PATIENTS)):
            errors = persistence_errors(df, patients, horizon)
            summary = summarize(errors)
            horizon_results[split_name] = summary
            print(
                f"[T+{horizon}min] {split_name:5s} (n_patients={len(patients)}): "
                f"RMSE={summary['rmse']:.2f} mg/dL, MAE={summary['mae']:.2f} mg/dL, "
                f"pairs={summary['n_pairs']}"
            )
        results["horizons"][str(horizon)] = horizon_results
        print()

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
