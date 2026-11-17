"""Reliability audit of the confirmed forecaster (TRIAGE.md section 11): where is it wrong, and is it safe?

    python -m model.cgmacros.reliability                         # dev: primary held-out 12 (fixes are tuned here)
    CGMACROS_SPLIT=confirm python -m model.cgmacros.reliability  # locked 9: describe only, nothing is tuned on it

Checks, for the blend and for TFT A:
  safety       Clarke error grid zones at T+60 (A+B = clinically acceptable; C/D/E = could lead to a wrong action)
  ranges       RMSE and 80% coverage by the glucose level that actually occurred (< 70, 70-180, 180-250, > 250)
  extremes     share of actual lows (< 70) / highs (> 250) that the band reaches (p10 < 70 / p90 > 250)
  cold start   by days of the person's data before as_of (personal-history features are thin early)
  people       RMSE and coverage per person
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from model.cgmacros import lab
from model.cgmacros.cases import OUT
from model.cgmacros.dataset import ROOT
from model.cgmacros.splits import SCHEME

RESULTS = ROOT / "model" / "results" / f"cgmacros_reliability_{SCHEME}.json"


def clarke(ref: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Clarke error grid zone per point (Clarke et al., Diabetes Care 1987), mg/dL."""
    z = np.full(len(ref), "B", dtype="<U1")
    a = (np.abs(pred - ref) <= 0.2 * ref) | ((ref < 70) & (pred < 70))
    e = ((ref <= 70) & (pred >= 180)) | ((ref >= 180) & (pred <= 70))
    c = ((ref >= 70) & (ref <= 290) & (pred >= ref + 110)) | ((ref >= 130) & (ref <= 180) & (pred <= 1.4 * ref - 182))
    d = (((ref >= 240) & (pred >= 70) & (pred <= 180)) | ((ref <= 175 / 3) & (pred <= 180) & (pred >= 70))
         | ((ref >= 175 / 3) & (ref <= 70) & (pred >= 1.2 * ref)))
    z[d] = "D"
    z[c] = "C"
    z[e] = "E"
    z[a] = "A"
    return z


def predictions():
    """Blend + TFT A forecasts on the audit set (dev for primary, locked 9 for confirm), refit deterministically."""
    if SCHEME == "confirm":
        c = pd.read_pickle(OUT / "lab_locked_confirm.pkl")
        train, test, q = c["train"], c["test"], np.load(OUT / "lab_locked_pred_chronos2.npy")
    else:
        c = pd.read_pickle(lab.CACHE)
        train, test, q = c["train"], c["dev"], np.load(OUT / "lab_pred_chronos2.npy")
    med_h, _, _ = lab.TRIALS["hgb_long_person"](train, test)
    blend = 0.7 * med_h + 0.3 * q[:, :, 1]
    lo, hi = blend + q[:, :, 0] - q[:, :, 1], blend + q[:, :, 2] - q[:, :, 1]
    test = test.copy()
    test[["p_days"]] = lab.extra_features(test)[["p_days"]].to_numpy()
    return test, blend, lo, hi, lab.stack(test.glucose_med), lab.stack(test.glucose_lo), lab.stack(test.glucose_hi)


def audit(test, med, lo, hi, s=lab.S60) -> dict:
    y = lab.stack(test.y)[:, s]
    p, l, h = med[:, s], lo[:, s], hi[:, s]
    cov = (y >= l) & (y <= h)
    zones = clarke(y, p)
    out = {"rmse": round(lab.rmse(p, y), 2), "coverage80": round(float(cov.mean()), 3),
           "clarke": {k: round(float((zones == k).mean()), 4) for k in "ABCDE"},
           "clarke_A_plus_B": round(float(np.isin(zones, ["A", "B"]).mean()), 4)}
    out["by_actual_range"] = {}
    for name, m in (("<70", y < 70), ("70-180", (y >= 70) & (y <= 180)), ("180-250", (y > 180) & (y <= 250)),
                    (">250", y > 250)):
        out["by_actual_range"][name] = None if not m.any() else {
            "n": int(m.sum()), "rmse": round(lab.rmse(p[m], y[m]), 1), "bias": round(float((p[m] - y[m]).mean()), 1),
            "coverage80": round(float(cov[m].mean()), 3)}
    low, high = y < 70, y > 250
    out["extremes"] = {"lows_reached_by_p10": None if not low.any() else round(float((l[low] < 70).mean()), 3),
                       "highs_reached_by_p90": None if not high.any() else round(float((h[high] > 250).mean()), 3),
                       "lows_reached_by_median": None if not low.any() else round(float((p[low] < 70).mean()), 3),
                       "highs_reached_by_median": None if not high.any() else round(float((p[high] > 250).mean()), 3)}
    days = test["p_days"].to_numpy()
    out["cold_start"] = {}
    for name, m in (("day_1", days < 1), ("day_2", (days >= 1) & (days < 2)), ("day_3_plus", days >= 2)):
        out["cold_start"][name] = None if not m.any() else {
            "n": int(m.sum()), "rmse": round(lab.rmse(p[m], y[m]), 2), "coverage80": round(float(cov[m].mean()), 3)}
    out["people"] = {pp: {"n": len(ix), "rmse": round(lab.rmse(p[ix], y[ix]), 1),
                          "coverage80": round(float(cov[ix].mean()), 3),
                          "sd_actual": round(float(y[ix].std()), 1)}
                     for pp, ix in test.groupby("participant").indices.items()}
    return out


def main() -> int:
    test, b, bl, bh, a, al, ah = predictions()
    res = {"scheme": SCHEME, "set": "locked 9 (describe only)" if SCHEME == "confirm" else "dev 12",
           "windows": len(test), "blend": audit(test, b, bl, bh), "tft_A": audit(test, a, al, ah)}
    RESULTS.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
