"""Hybrid meal what-if: the forecaster's no-meal forecast plus a physiological meal-response curve.

    CGMACROS_SPLIT=confirm python -m model.cgmacros.hybrid      # pre-registered test (needs the confirm checkpoints)

Why: the forecaster (TFT, variant D) learned meals only from data, and its meal what-ifs are not physiologically
consistent: in 160 of 196 triage scenarios the meal forecast dips below the no-meal forecast, and in case-021 a 30 g
meal forecasts higher than a 60 g meal (docs/TRIAGE.md section 1). Idea from Miller, Foti & Fox (Apple, MLHC 2020,
arXiv:2008.02852): keep a physiological model's structure and let a learned model set its parameters. Without
insulin data (CGMacros adults use none) only the meal part applies.

Model:  forecast(t) = D_no_meal(t) + sum over meals m of  S(x) * carbs_m * e(t - t_m)
  e(.)  glucose effect of 1 g carbohydrate: Hovorka two-compartment gut absorption, tau_D = 40 min (Hovorka et al.
        2004, doi:10.1088/0967-3334/25/4/010; equations and values as reproduced by Pompa et al. 2021,
        doi:10.1371/journal.pone.0257789): U(s) = s / tau^2 * exp(-s / tau), convolved with first-order glucose
        clearance exp(-k t). FIXED shape; k is chosen from a pre-set grid (below) on training people only.
  S(x)  gain, mg/dL per g: exp(b + w . x), x = history features at as_of (current glucose, 2-h mean, 2-h sd,
        30-min change, hour sin/cos, carbs logged in the past 2 h), standardized on training people.
        S >= 0 and S does not depend on the planned meal, so more carbs can never forecast lower glucose and a meal
        can never forecast below no meal (tested in tests/model_cgmacros/test_hybrid.py; a property, not a criterion).

PRE-REGISTERED (written 2026-11-16, before any hybrid output for held-out people was computed):
  Windows     every logged meal (carbs > 0) of a person: as_of = the reading 5 min before the meal; 2 h of contiguous
              history and a fully observed next hour required. Both arms get every meal logged in that hour, at its
              step, with the deployment decoder (scenarios.future_frame: METs 1.0, heart rate held, no oracle activity).
  Arms        D      the confirm-split TFT (variant meals_activity) given the meals' carbs, protein, fat and fiber.
              hybrid D's no-meal forecast + S(x) * carbs * e(t), carbs only.
  Fitting     on the 31 confirm training people only: residual = actual - D_no_meal, least squares on all 12 steps,
              ridge 1.0 on w; k in {0, 0.005, 0.01, 0.02, 0.04} per min chosen by 5-fold GroupKFold (by person) loss.
              (D trained on these people, so their residuals are a little smaller than on new people.)
  Primary     the global-S hybrid, on the 9 NEW people: T+60 RMSE(hybrid) - RMSE(D), 95% person-bootstrap interval;
              non-inferior if the upper bound is < +2.0 mg/dL (D's meal-hour T+60 RMSE was 28.5 on the primary split).
              With 9 people the interval may be wide; an interval spanning the margin is inconclusive, not a failure
              of the property the hybrid guarantees.
  Secondary   RMSE averaged over the hour; the same on all 14 held-out people (9 new + 5 early-stopping); calibration:
              mean predicted vs observed rise over no-meal at +60 for meals of 20-40 g and 50-70 g; D's own violations
              on these windows (meal forecast below no-meal at any step); a personalized gain (S scaled by the
              person's earlier meals, only windows that ended before as_of, shrunk to 1 with 2 pseudo-meals).
  Not changed triage levels: the rule reads no_new_meal and history_only, not the meal scenarios. The hybrid would
              change only the meal what-if numbers agents and clinicians see, and only after this test.
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold

from model.cgmacros.cases import OUT
from model.cgmacros.dataset import ROOT, build_grid
from model.cgmacros.splits import SCHEME, split_participants
from model.cgmacros.train import ENCODER, HORIZON

STEP = 5.0
TAU_D = 40.0
K_GRID = (0.0, 0.005, 0.01, 0.02, 0.04)
RIDGE = 1.0
MARGIN = 2.0
X_COLS = ["now", "mean2h", "sd2h", "d30", "hour_sin", "hour_cos", "carbs2h"]
MACROS = ["carbs", "protein", "fat", "fiber"]
CACHE = OUT / f"hybrid_windows_{SCHEME}.pkl"
RESULTS = ROOT / "model" / "results" / f"cgmacros_hybrid_{SCHEME}.json"


def effect_curve(k: float, minutes: int = int(STEP * HORIZON)) -> np.ndarray:
    """Glucose effect of 1 unit absorbed carbohydrate at 0..minutes (1-min resolution, mg/dL per unit gain)."""
    t = np.arange(minutes + 1, dtype=float)
    u = t / TAU_D ** 2 * np.exp(-t / TAU_D)               # absorption rate, integrates to 1
    e = np.zeros_like(t)
    for i in range(1, len(t)):                            # de/dt = u - k e  (1-min Euler steps)
        e[i] = e[i - 1] + (u[i - 1] - k * e[i - 1])
    return e


def meal_basis(carbs: np.ndarray, k: float) -> np.ndarray:
    """carbs: (n, HORIZON) grams logged at decoder steps 1..HORIZON. Returns (n, HORIZON) sum_m carbs_m e(t - t_m),
    where step j is (j - m) * 5 min after a meal at step m (a meal at step j has no effect yet at j)."""
    e = effect_curve(k)
    lag = np.array([[e[int((j - m) * STEP)] if j >= m else 0.0 for j in range(HORIZON)] for m in range(HORIZON)])
    return carbs @ lag


def gain(X: np.ndarray, params: np.ndarray) -> np.ndarray:
    return np.exp(params[0] + X @ params[1:])


def hybrid_forecast(no_meal: np.ndarray, X: np.ndarray, carbs: np.ndarray, params: np.ndarray, k: float,
                    scale: np.ndarray | float = 1.0) -> np.ndarray:
    """no_meal: (n, HORIZON); scale >= 0 (personalization). Non-decreasing in every carbs entry by construction."""
    s = np.broadcast_to(np.asarray(scale, float), (len(X),))
    return no_meal + (s * gain(X, params))[:, None] * meal_basis(carbs, k)


def fit_gain(X: np.ndarray, resid: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, float]:
    def loss(p):
        r = resid - gain(X, p)[:, None] * basis
        return float((r ** 2).mean()) + RIDGE * float((p[1:] ** 2).sum()) / len(X)
    best = min((minimize(loss, np.r_[b0, np.zeros(X.shape[1])], method="L-BFGS-B") for b0 in (-3.0, -1.0, 0.0)),
               key=lambda r: r.fun)
    return best.x, best.fun


def windows(grid: pd.DataFrame, ids: list[str]) -> list[tuple[str, pd.DataFrame, int]]:
    """(person, their grid, as_of row) for every logged meal row with contiguous 2-h history and next hour."""
    out = []
    for p in ids:
        g = grid[grid.participant == p].reset_index(drop=True)
        ti, carbs = g["time_idx"].to_numpy(), g["carbs"].to_numpy()
        for j in np.flatnonzero(carbs > 0):
            i = j - 1
            if i < ENCODER - 1 or i + HORIZON >= len(g):
                continue
            if ti[i + HORIZON] - ti[i - ENCODER + 1] != ENCODER + HORIZON - 1:
                continue
            out.append((p, g, i))
    return out


def build_windows(grid: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    """D with the logged meals and D with no meal, through the deployment decoder; cached (no hybrid output)."""
    from pytorch_forecasting import TimeSeriesDataSet

    from model.cgmacros.event import history_features
    from model.cgmacros.scenarios import ScenarioForecaster, future_frame
    f = ScenarioForecaster("meals_activity")
    rows, frames = [], []
    for n, (p, g, i) in enumerate(windows(grid, ids)):
        hist = g.iloc[i - ENCODER + 1:i + 1]
        fut = g.iloc[i + 1:i + 1 + HORIZON]
        base = future_frame(hist, "no_new_meal", f"n{n}")
        meal = future_frame(hist, "no_new_meal", f"m{n}")
        for c in MACROS:
            meal.loc[meal.index[ENCODER:], c] = fut[c].to_numpy()
        frames += [base, meal]
        hf = history_features(hist)
        rows.append({"participant": p, "as_of": hist["ts"].iloc[-1], "n": n,
                     "actual": fut["glucose"].to_numpy(float), "carbs": fut["carbs"].to_numpy(float),
                     **{c: hf[c] for c in X_COLS}})
    preds = {}
    for s in range(0, len(frames), 400):
        chunk = pd.concat(frames[s:s + 400], ignore_index=True)
        ds = TimeSeriesDataSet.from_dataset(f.training, chunk, predict=True, stop_randomization=True)
        out = f.model.predict(ds.to_dataloader(train=False, batch_size=256, num_workers=0), mode="raw",
                              return_index=True, trainer_kwargs={"accelerator": "cpu", "logger": False})
        med = f.model.loss.to_prediction(out.output.prediction).numpy()
        preds.update(dict(zip(out.index["participant"], med)))
        print(f"  predicted {min(s + 400, len(frames))}/{len(frames)}", flush=True)
    df = pd.DataFrame(rows)
    df["d_no_meal"] = [preds[f"n{n}"] for n in df.n]
    df["d_meal"] = [preds[f"m{n}"] for n in df.n]
    return df


def stack(df: pd.DataFrame, col: str) -> np.ndarray:
    return np.vstack(df[col].to_numpy())


def fit(train: pd.DataFrame) -> dict:
    mu, sd = train[X_COLS].mean(), train[X_COLS].std().replace(0, 1)
    X = ((train[X_COLS] - mu) / sd).fillna(0).to_numpy()
    resid = stack(train, "actual") - stack(train, "d_no_meal")
    carbs, groups = stack(train, "carbs"), train["participant"].to_numpy()
    cv = {}
    for k in K_GRID:
        basis, err = meal_basis(carbs, k), []
        for tr, te in GroupKFold(n_splits=5).split(X, groups=groups):
            p, _ = fit_gain(X[tr], resid[tr], basis[tr])
            err.append(((resid[te] - gain(X[te], p)[:, None] * basis[te]) ** 2).sum())
        cv[k] = float(np.sum(err) / resid.size)
    k = min(cv, key=cv.get)
    params, _ = fit_gain(X, resid, meal_basis(carbs, k))
    return {"k": k, "cv_mse": cv, "params": params, "mu": mu, "sd": sd}


def design(df: pd.DataFrame, m: dict) -> np.ndarray:
    return ((df[X_COLS] - m["mu"]) / m["sd"]).fillna(0).to_numpy()


def personal_scale(df: pd.DataFrame, m: dict) -> np.ndarray:
    """Per window: the person's response to earlier meals (windows whose hour ended by as_of) relative to the global
    model, least-squares ratio shrunk toward 1 with 2 pseudo-meals, clipped to [0, 3]. Uses no future data."""
    pred = gain(design(df, m), m["params"])[:, None] * meal_basis(stack(df, "carbs"), m["k"])
    resid = stack(df, "actual") - stack(df, "d_no_meal")
    prior = float((pred ** 2).sum(1).mean()) * 2.0
    scale = np.ones(len(df))
    hour = pd.Timedelta(minutes=STEP * HORIZON)
    for _, idx in df.groupby("participant").groups.items():
        idx = list(idx)
        for r in idx:
            ended = [q for q in idx if df.at[q, "as_of"] + hour <= df.at[r, "as_of"]]
            num = sum(float((resid[q] * pred[q]).sum()) for q in ended) + prior
            den = sum(float((pred[q] ** 2).sum()) for q in ended) + prior
            scale[r] = min(max(num / den, 0.0), 3.0)
    return scale


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).mean()))


def boot_rmse_diff(df: pd.DataFrame, a: np.ndarray, b: np.ndarray, step: int | None, n: int = 2000) -> list[float]:
    people = df["participant"].unique()
    rows = {p: np.flatnonzero(df["participant"].to_numpy() == p) for p in people}
    y = stack(df, "actual")
    cols = slice(None) if step is None else [step]
    rng, diffs = np.random.default_rng(0), []
    for _ in range(n):
        ix = np.concatenate([rows[p] for p in rng.choice(people, len(people))])
        diffs.append(rmse(a[ix][:, cols], y[ix][:, cols]) - rmse(b[ix][:, cols], y[ix][:, cols]))
    return [round(float(np.percentile(diffs, 2.5)), 2), round(float(np.percentile(diffs, 97.5)), 2)]


def evaluate(df: pd.DataFrame, m: dict) -> dict:
    y, d, d0, carbs = stack(df, "actual"), stack(df, "d_meal"), stack(df, "d_no_meal"), stack(df, "carbs")
    X = design(df, m)
    h = hybrid_forecast(d0, X, carbs, m["params"], m["k"])
    hp = hybrid_forecast(d0, X, carbs, m["params"], m["k"], personal_scale(df, m))
    total = carbs.sum(1)
    calib = {}
    for lo, hi in ((20, 40), (50, 70)):
        s = (total >= lo) & (total <= hi)
        calib[f"{lo}-{hi}g"] = None if not s.any() else {
            "n": int(s.sum()), "observed_rise_over_no_meal_t60": round(float((y - d0)[s, -1].mean()), 1),
            "hybrid": round(float((h - d0)[s, -1].mean()), 1), "D": round(float((d - d0)[s, -1].mean()), 1)}
    return {"windows": len(df), "people": int(df.participant.nunique()),
            "t60_rmse": {"D": round(rmse(d[:, -1], y[:, -1]), 2), "hybrid": round(rmse(h[:, -1], y[:, -1]), 2),
                         "hybrid_personal": round(rmse(hp[:, -1], y[:, -1]), 2),
                         "D_no_meal": round(rmse(d0[:, -1], y[:, -1]), 2)},
            "t60_diff_hybrid_minus_D_ci95": boot_rmse_diff(df, h, d, HORIZON - 1),
            "t60_diff_personal_minus_D_ci95": boot_rmse_diff(df, hp, d, HORIZON - 1),
            "hour_rmse": {"D": round(rmse(d, y), 2), "hybrid": round(rmse(h, y), 2),
                          "hybrid_personal": round(rmse(hp, y), 2), "D_no_meal": round(rmse(d0, y), 2)},
            "hour_diff_hybrid_minus_D_ci95": boot_rmse_diff(df, h, d, None),
            "calibration": calib,
            "D_meal_below_no_meal_any_step": round(float((d < d0 - 1.0).any(1).mean()), 3),
            "hybrid_meal_below_no_meal_any_step": round(float((h < d0 - 1e-9).any(1).mean()), 3)}


def main() -> int:
    if SCHEME != "confirm":
        raise SystemExit("run with CGMACROS_SPLIT=confirm (the forecaster must never have seen the test people)")
    grid = build_grid()
    train_ids, val_ids, test_ids = split_participants(sorted(grid.participant.unique()))
    if CACHE.exists():
        df = pd.read_pickle(CACHE)
    else:
        df = build_windows(grid, sorted(train_ids + val_ids + test_ids))
        df.to_pickle(CACHE)
    train = df[df.participant.isin(train_ids)].reset_index(drop=True)
    m = fit(train)
    print(f"train: {len(train)} meal windows, {train.participant.nunique()} people; k={m['k']} cv={m['cv_mse']}")
    new9 = df[df.participant.isin(test_ids)].reset_index(drop=True)
    all14 = df[df.participant.isin(val_ids + test_ids)].reset_index(drop=True)
    assert not set(new9.participant) & set(train_ids), "test people in training"
    out = {"scheme": SCHEME, "preregistration": __doc__.split("PRE-REGISTERED")[1].strip(),
           "fit": {"k_per_min": m["k"], "cv_mse_by_k": m["cv_mse"], "params": [round(float(v), 4) for v in m["params"]],
                   "features": X_COLS, "train_windows": len(train)},
           "train_in_sample": evaluate(train, m), "new_9_people": evaluate(new9, m), "all_14_people": evaluate(all14, m)}
    ci = out["new_9_people"]["t60_diff_hybrid_minus_D_ci95"]
    out["primary_non_inferior"] = bool(ci[1] < MARGIN)
    RESULTS.write_text(json.dumps(out, indent=1, default=float) + "\n")
    with open(OUT / f"hybrid_model_{SCHEME}.pkl", "wb") as fh:
        pickle.dump(m, fh)
    print(json.dumps({k: out[k] for k in ("fit", "new_9_people", "all_14_people", "primary_non_inferior")}, indent=1,
                     default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
