"""Locked test of the forecast-lab winner on the 9 confirm-test people.

    CGMACROS_SPLIT=confirm python -m model.cgmacros.lab_confirm --prepare        # windows, TFT A, Chronos context
    .venv-chronos/bin/python model/cgmacros/chronos_run.py chronos2 512 locked   # Chronos-2 on the locked windows
    CGMACROS_SPLIT=confirm python -m model.cgmacros.lab_confirm --score          # fit on the 31, score once

PRE-REGISTERED (written 2026-11-16, after 14 dev trials in model/results/lab_log.jsonl on the primary held-out 12, and
before any window, feature or forecast of the 9 locked people was computed):
  People      fit on the 31 confirm-training people only; test on the 9 confirm-test people (003, 008, 009, 014, 016,
              017, 038, 043, 045). This is these 9 people's fifth confirmatory use (TRIAGE.md sections 7-10 before it).
              The 5 confirm early-stopping people were dev people in the lab, so no all-14 number is reported.
  Windows     lab.person_windows: every as_of with 2 h contiguous history and a contiguous next hour (stride 1).
  Primary     "blend": 0.7 x hgb_long_person + 0.3 x Chronos-2 median, band = Chronos-2's p10/p90 offsets around the
              blended median (fixed weights, not tuned).
                hgb_long_person  HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                 l2_regularization=1.0, random_state=0), one per step (T+30, T+60), target = change
                                 from now, features lab.FEATURES + LONG + PERSON (no future-meal feature).
                Chronos-2        chronos-forecasting==2.0.1, amazon/chronos-2 at revision
                                 29ec3766d36d6f73f0696f85560a422f50e8498c, zero-shot, 512-reading context, CPU.
  Baseline    TFT A (glucose history only), the confirm-split checkpoint (never trained on these 9).
  Criterion   T+60 RMSE gain of the blend over A, 95% person-bootstrap interval lower bound > 0.
  Secondary   T+30; slices (meal in the next hour, rising >= 20 mg/dL per 15 min); 80% band coverage at T+60, judged
              acceptable in 0.75-0.85; hgb_long_person alone and Chronos-2 alone; per-person wins vs A.
  Not tested  triage levels. Swapping the forecaster would change the rule's levels; that needs its own test.
  Expectation dev numbers are the best of 14 trials, so the locked gain should come in below dev's +4.5.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from model.cgmacros import lab
from model.cgmacros.cases import OUT
from model.cgmacros.dataset import ROOT, build_grid
from model.cgmacros.splits import SCHEME, split_participants

CACHE = OUT / "lab_locked_confirm.pkl"
CTX = OUT / "lab_locked_context.npy"
PRED = OUT / "lab_locked_pred_chronos2.npy"
RESULTS = ROOT / "model" / "results" / "cgmacros_lab_locked.json"


def prepare() -> None:
    grid = build_grid()
    train_ids, _, test_ids = split_participants(sorted(grid.participant.unique()))
    train, test = lab.all_windows(grid, train_ids), lab.all_windows(grid, test_ids)
    assert not set(train.participant) & set(test.participant)
    p = lab.tft_preds("glucose", grid, test_ids)                     # confirm checkpoint: SCHEME=confirm
    got = [p.get((a, t + 1)) for a, t in zip(test.participant, test.time_idx)]
    for j, part in enumerate(("med", "lo", "hi")):
        test[f"glucose_{part}"] = [x[j] if x is not None else None for x in got]
    test = test[test.glucose_med.notna()].reset_index(drop=True)
    test["section1_test"] = True
    pd.to_pickle({"train": train, "test": test, "train_ids": train_ids, "test_ids": test_ids}, CACHE)
    ctx = np.full((len(test), lab.CTX_LEN), np.nan, dtype=np.float32)
    for person, w in test.groupby("participant"):
        g = grid[grid.participant == person].set_index("time_idx")["glucose"]
        lo = int(g.index.min())
        s = np.r_[np.full(lab.CTX_LEN, np.nan, np.float32),
                  g.reindex(range(lo, int(g.index.max()) + 1)).to_numpy(np.float32)]
        for row, t in zip(w.index, w.time_idx):
            end = int(t) - lo + lab.CTX_LEN + 1
            ctx[row] = s[end - lab.CTX_LEN:end]
    assert np.allclose(ctx[:, -1], test["now"].to_numpy())
    np.save(CTX, ctx)
    print(f"prepared: train {len(train)} windows / {train.participant.nunique()} people; "
          f"locked {len(test)} windows / {test.participant.nunique()} people; context {ctx.shape}")


def score() -> None:
    c = pd.read_pickle(CACHE)
    train, test = c["train"], c["test"]
    q = np.load(PRED)
    assert len(q) == len(test)
    med_h, lo_h, hi_h = lab.TRIALS["hgb_long_person"](train, test)
    blend = 0.7 * med_h + 0.3 * q[:, :, 1]
    arms = {"blend": (blend, blend + q[:, :, 0] - q[:, :, 1], blend + q[:, :, 2] - q[:, :, 1]),
            "hgb_long_person": (med_h, lo_h, hi_h), "chronos2": (q[:, :, 1], q[:, :, 0], q[:, :, 2])}
    y, A = lab.stack(test.y), lab.stack(test.glucose_med)
    out = {"preregistration": __doc__.split("PRE-REGISTERED")[1].strip(), "people": c["test_ids"],
           "windows": len(test), "A_t60": round(lab.rmse(A[:, lab.S60], y[:, lab.S60]), 2),
           "A_t30": round(lab.rmse(A[:, lab.S30], y[:, lab.S30]), 2)}
    for name, (m, lo, hi) in arms.items():
        r = lab.score(test, m, lo, hi)
        r.pop("section1_test7", None)
        per = {p: [round(lab.rmse(m[ix, lab.S60], y[ix, lab.S60]), 1), round(lab.rmse(A[ix, lab.S60], y[ix, lab.S60]), 1)]
               for p, ix in test.groupby("participant").indices.items()}
        r["per_person_t60_vs_A"] = per
        r["people_better_than_A"] = sum(a < b for a, b in per.values())
        out[name] = r
    b = out["blend"]
    out["criterion_t60_gain_ci_lower_above_0"] = bool(b["t60_gain_vs_A_ci95"][0] > 0)
    out["coverage_in_0.75_0.85"] = bool(0.75 <= b["coverage80_t60"] <= 0.85)
    RESULTS.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "preregistration"}, indent=1))


def main() -> int:
    if SCHEME != "confirm":
        raise SystemExit("run with CGMACROS_SPLIT=confirm")
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    if a.prepare:
        prepare()
    if a.score:
        score()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
