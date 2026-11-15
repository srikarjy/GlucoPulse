"""Ablation + what-if sanity checks for the CGMacros forecaster variants.

    python -m model.cgmacros.evaluate        # writes model/results/cgmacros_ablation.json

On the held-out participants, for every trained variant and a persistence baseline:
  * RMSE / MAE / 80% interval coverage at T+30 and T+60
  * the same, restricted to windows with a logged meal inside the forecast hour
    (where a known-future meal input should matter most)
What-if probes (meal variants only, on quiet windows with no meal for 2 h before and
during the forecast): inject a 60 g carb meal at forecast start, and for the activity
variant a 30-minute walk (4 METs). Report how the median forecast moves. These check
the *direction* the model learned; they are model-based associations, not causal
estimates, and say nothing about any individual.
"""
from __future__ import annotations

import json
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet

from model.cgmacros.dataset import build_grid
from model.cgmacros.splits import split_participants
from model.cgmacros.train import ART, ENCODER, HORIZON, VARIANTS, MemSafeTFT

RESULTS = Path(__file__).resolve().parents[1] / "results" / "cgmacros_ablation.json"
STEPS = {30: 5, 60: 11}
MEAL_SCENARIO = {"carbs": 60.0, "protein": 15.0, "fat": 15.0, "fiber": 5.0}
WALK_METS, WALK_STEPS = 4.0, 6


def load(variant: str):
    with open(ART / f"{variant}.dataset.pkl", "rb") as fh:
        training = pickle.load(fh)
    model = MemSafeTFT.load_from_checkpoint(ART / f"{variant}.ckpt", map_location="cpu")
    model.eval()
    return model, training


def _metrics(actual, pred, lo, hi):
    err = actual - pred
    return {"rmse": round(float(np.sqrt(np.mean(err ** 2))), 2), "mae": round(float(np.mean(np.abs(err))), 2),
            "coverage80": None if lo is None else round(float(np.mean((actual >= lo) & (actual <= hi))), 3),
            "n": int(len(actual))}


def predict_test(model, training, test_df):
    ds = TimeSeriesDataSet.from_dataset(training, test_df, predict=False, stop_randomization=True)
    raw = model.predict(ds.to_dataloader(train=False, batch_size=512, num_workers=0), mode="raw", trainer_kwargs={"accelerator": "cpu", "logger": False},
                        return_x=True, return_index=True)
    pred = raw.output.prediction
    q = model.loss.to_quantiles(pred).numpy()
    levels = model.loss.quantiles
    median = model.loss.to_prediction(pred).numpy()
    enc_t, enc_len = raw.x["encoder_target"].numpy(), raw.x["encoder_lengths"].numpy()
    last = enc_t[np.arange(len(enc_t)), enc_len - 1]
    return {"median": median, "lo": q[:, :, levels.index(0.1)], "hi": q[:, :, levels.index(0.9)],
            "actual": raw.x["decoder_target"].numpy(), "persistence": last, "index": raw.index}


def meal_mask(index: pd.DataFrame, grid: pd.DataFrame) -> np.ndarray:
    meals = grid[grid.carbs > 0][["participant", "time_idx"]]
    by_p = {p: set(g.time_idx) for p, g in meals.groupby("participant")}
    return np.array([any(t in by_p.get(p, ()) for t in range(t0, t0 + HORIZON))
                     for p, t0 in zip(index["participant"], index["time_idx"])])


def quiet_windows(grid: pd.DataFrame, ids, n: int = 300, seed: int = 7):
    """(participant, start time_idx) with a contiguous 2 h + 1 h span and no meal in it."""
    out = []
    for p in ids:
        g = grid[grid.participant == p].set_index("time_idx")
        idx = g.index.to_numpy()
        for t0 in idx[ENCODER::6]:
            span = range(t0 - ENCODER, t0 + HORIZON)
            if all(t in g.index for t in span) and g.loc[list(span), "carbs"].sum() == 0:
                out.append((p, int(t0)))
    random.Random(seed).shuffle(out)
    return out[:n]


def probe(model, training, grid, windows, edit) -> dict:
    base, scen = [], []
    for k, (p, t0) in enumerate(windows):
        g = grid[(grid.participant == p) & grid.time_idx.between(t0 - ENCODER, t0 + HORIZON - 1)].copy()
        g["participant"] = f"{p}#w{k}"
        base.append(g)
        s = g.copy()
        edit(s, t0)
        scen.append(s)
    preds = {}
    for name, frames in (("base", base), ("scenario", scen)):
        ds = TimeSeriesDataSet.from_dataset(training, pd.concat(frames, ignore_index=True), predict=True,
                                            stop_randomization=True)
        out = model.predict(ds.to_dataloader(train=False, batch_size=512, num_workers=0), mode="raw", trainer_kwargs={"accelerator": "cpu", "logger": False},
                            return_index=True)
        order = out.index["participant"].to_numpy()
        med = model.loss.to_prediction(out.output.prediction).numpy()
        preds[name] = dict(zip(order, med))
    keys = sorted(preds["base"])
    delta = np.array([preds["scenario"][k] - preds["base"][k] for k in keys])
    return {f"t+{m}": {"mean_delta": round(float(delta[:, s].mean()), 2),
                       "median_delta": round(float(np.median(delta[:, s])), 2),
                       "share_rising": round(float((delta[:, s] > 0).mean()), 3)} for m, s in STEPS.items()} | {
        "windows": len(keys)}


def main() -> int:
    torch.set_grad_enabled(False)
    grid = build_grid()
    _, _, test_ids = split_participants(sorted(grid.participant.unique()))
    test_df = grid[grid.participant.isin(test_ids)]
    results = {"test_participants": test_ids, "horizons_min": list(STEPS), "variants": {}, "probes": {},
               "notes": "Held-out participants (CGMacros: healthy, prediabetes and T2D adults, no insulin). "
                        "Meal and activity inputs are known-future: they assume the plan for the next hour is "
                        "known. Probe deltas are model-based associations, not causal effects."}
    windows = quiet_windows(grid, test_ids)
    for variant in VARIANTS:
        if not (ART / f"{variant}.ckpt").exists():
            continue
        model, training = load(variant)
        r = predict_test(model, training, test_df)
        # Only windows whose decoder hour has no CGM gap: pytorch-forecasting fills missing
        # timesteps, and scoring against filled values would flatter persistence-like models.
        present = {p: set(g.time_idx) for p, g in test_df.groupby("participant")}
        keep = np.array([all(t in present[p] for t in range(t0, t0 + HORIZON))
                         for p, t0 in zip(r["index"]["participant"], r["index"]["time_idx"])])
        r = {k: (v[keep] if isinstance(v, np.ndarray) else v[keep].reset_index(drop=True)) for k, v in r.items()}
        mm = meal_mask(r["index"], grid)
        entry = {}
        for minutes, s in STEPS.items():
            a = r["actual"][:, s]
            entry[f"t+{minutes}"] = {
                "all": _metrics(a, r["median"][:, s], r["lo"][:, s], r["hi"][:, s]),
                "meal_windows": _metrics(a[mm], r["median"][mm, s], r["lo"][mm, s], r["hi"][mm, s]),
            }
            results.setdefault("persistence", {})[f"t+{minutes}"] = {
                "all": _metrics(a, r["persistence"], None, None),
                "meal_windows": _metrics(a[mm], r["persistence"][mm], None, None)}
        results["variants"][variant] = entry
        if "carbs" in VARIANTS[variant]["known"]:
            def add_meal(s, t0):
                for col, v in MEAL_SCENARIO.items():
                    s.loc[s.time_idx == t0, col] = v
            results["probes"][f"{variant}: +60 g carb meal"] = probe(model, training, grid, windows, add_meal)
        if "mets" in VARIANTS[variant]["known"]:
            def add_walk(s, t0):
                s.loc[s.time_idx.between(t0, t0 + WALK_STEPS - 1), "mets"] = WALK_METS
            results["probes"][f"{variant}: 30 min walk (4 METs)"] = probe(model, training, grid, windows, add_walk)
        print(f"[{variant}] " + "  ".join(f"T+{m}: RMSE {entry[f't+{m}']['all']['rmse']} "
                                          f"(meal windows {entry[f't+{m}']['meal_windows']['rmse']})"
                                          for m in STEPS), flush=True)
    RESULTS.write_text(json.dumps(results, indent=2) + "\n")
    print("persistence: " + "  ".join(f"T+{m}: RMSE {results['persistence'][f't+{m}']['all']['rmse']}"
                                       for m in STEPS))
    print(json.dumps(results["probes"], indent=1))
    print(f"-> {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
