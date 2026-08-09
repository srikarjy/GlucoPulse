"""
Evaluate the trained TFT against the persistence baseline on TEST_PATIENTS --
patients the model has never seen in training or validation (docs/STATUS.md).

Reports, for T+30 and T+60 (steps 6 and 12 of the 12-step/60-min forecast):
  - pooled RMSE/MAE across all test patients
  - per-patient RMSE/MAE (small-N generalization risk, docs/PROBLEMS.md --
    report honestly rather than hiding behind one aggregate number)
  - Clarke error grid zone counts (clinical-accuracy check, not just RMSE)
  - 80% prediction-interval coverage (quantile calibration check,
    docs/PROBLEMS.md's "multi-horizon probabilistic evaluation" ask)
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer

from splits import TEST_PATIENTS
from train import CHECKPOINT_PATH, MODEL_DIR, build_dataframe

RESULTS_PATH = Path(__file__).parent / "results" / "tft_eval.json"
HORIZON_STEPS = {30: 5, 60: 11}  # 0-indexed step into the 12-step prediction


def clarke_zone(actual: float, pred: float) -> str:
    """Classic Clarke Error Grid zone classification (mg/dL). Standard reference
    piecewise-linear boundaries (Clarke et al. 1987); order matters (A/E/C/D
    checked before falling back to B)."""
    if (actual <= 70 and pred <= 70) or (pred <= 1.2 * actual and pred >= 0.8 * actual):
        return "A"
    if (actual >= 180 and pred <= 70) or (actual <= 70 and pred >= 180):
        return "E"
    if ((actual >= 70 and actual <= 290) and (pred >= actual + 110)) or (
        (actual >= 130 and actual <= 180) and (pred <= (7 / 5) * actual - 182)
    ):
        return "C"
    if (
        (actual >= 240 and (pred >= 70 and pred <= 180))
        or (actual <= 175 / 3 and pred <= 180 and pred >= 70)
        or ((actual >= 175 / 3 and actual <= 70) and pred >= (6 / 5) * actual)
    ):
        return "D"
    return "B"


def load_model_and_dataset():
    with open(MODEL_DIR / "training_dataset.pkl", "rb") as f:
        training = pickle.load(f)
    tft = TemporalFusionTransformer.load_from_checkpoint(CHECKPOINT_PATH, map_location="cpu")
    tft.eval()
    return tft, training


def main():
    tft, training = load_model_and_dataset()

    test_df = build_dataframe(patients=TEST_PATIENTS)
    test_dataset = training.__class__.from_dataset(training, test_df, predict=False, stop_randomization=True)
    test_loader = test_dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

    raw = tft.predict(test_loader, mode="raw", return_x=True, return_index=True)
    raw_pred = raw.output.prediction
    median_pred = tft.loss.to_prediction(raw_pred).numpy()  # (n_samples, 12)
    quantiles = tft.loss.to_quantiles(raw_pred).numpy()  # (n_samples, 12, n_quantiles)
    q_levels = tft.loss.quantiles
    lo_idx = q_levels.index(0.1)
    hi_idx = q_levels.index(0.9)

    actual = raw.x["decoder_target"].numpy()  # (n_samples, 12)
    group_ids = raw.index["patient_id"].to_numpy()

    results = {"test_patients": TEST_PATIENTS, "horizons": {}}
    all_zone_counts_by_horizon = {}

    for horizon_min, step in HORIZON_STEPS.items():
        pred_h = median_pred[:, step]
        actual_h = actual[:, step]
        lo_h = quantiles[:, step, lo_idx]
        hi_h = quantiles[:, step, hi_idx]

        err = actual_h - pred_h
        rmse = float(np.sqrt(np.mean(err**2)))
        mae = float(np.mean(np.abs(err)))
        coverage = float(np.mean((actual_h >= lo_h) & (actual_h <= hi_h)))

        zones = [clarke_zone(a, p) for a, p in zip(actual_h.tolist(), pred_h.tolist())]
        zone_counts = {z: zones.count(z) for z in "ABCDE"}
        all_zone_counts_by_horizon[horizon_min] = zone_counts

        per_patient = {}
        for pid in TEST_PATIENTS:
            mask = group_ids == pid
            if mask.sum() == 0:
                continue
            e = actual_h[mask] - pred_h[mask]
            per_patient[pid] = {
                "n": int(mask.sum()),
                "rmse": float(np.sqrt(np.mean(e**2))),
                "mae": float(np.mean(np.abs(e))),
            }

        results["horizons"][str(horizon_min)] = {
            "n_samples": int(len(pred_h)),
            "rmse": rmse,
            "mae": mae,
            "prediction_interval_80pct_coverage": coverage,
            "clarke_zones": zone_counts,
            "per_patient": per_patient,
        }

        print(f"[T+{horizon_min}min] TFT test RMSE={rmse:.2f} mg/dL, MAE={mae:.2f} mg/dL, n={len(pred_h)}")
        print(f"  80% PI coverage: {coverage:.1%} (target ~80%)")
        print(f"  Clarke zones: {zone_counts}")
        for pid, m in per_patient.items():
            print(f"  patient {pid}: RMSE={m['rmse']:.2f} MAE={m['mae']:.2f} n={m['n']}")
        print()

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
