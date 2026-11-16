"""Forecast lab: trial-and-error search for better 30/60-min forecasts, on DEV people only.

    python -m model.cgmacros.lab --cache            # once: windows + the TFT baselines on the dev people
    python -m model.cgmacros.lab --trial ridge      # run one candidate; appends a line to model/results/lab_log.jsonl
    python -m model.cgmacros.lab --list

Protocol (fixed before the first trial, 2026-11-16):
  Dev       primary split only: fit on the 33 primary training people, score on the 12 primary held-out people
            (7 test + 5 early-stopping), the set already used for exploration. The 9 confirm-test people are LOCKED:
            nothing here reads their outcomes. They are among the 33 primary training people, so a winner must be
            refit on the 31 confirm-training people before its one pre-registered run on them.
  Target    forecast made at as_of with no future-meal information (what triage uses): T+60 RMSE (primary dev metric)
            and T+30. Past meals, heart rate and METs up to as_of are allowed.
  Windows   every as_of with 2 h of contiguous history and a contiguous next hour (stride 1, like section 1).
  Bar       model A (TFT, glucose history only; 25.48 at T+60 on the 7 test people in section 1), re-scored on exactly
            these windows. Model D with ORACLE meals is shown for context only (it is told the future meals).
  Slices    windows with a logged meal in the next hour; rising windows (>= 20 mg/dL over the last 15 min).
  Bands     a candidate must also give an 80% band (p10/p90); coverage is reported.
  Log       every trial, including failures, appends one line to model/results/lab_log.jsonl.
  Bar to carry forward: T+60 RMSE gain vs A whose 95% person-bootstrap interval excludes 0 and is >= 1.0 mg/dL.
  Stop      after at most 25 trials or the five families in docs/TRIAGE_LITERATURE.md section 8, whichever first;
            then pre-register the top one or two for the locked people.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from model.cgmacros.cases import OUT
from model.cgmacros.dataset import ROOT, build_grid
from model.cgmacros.splits import SCHEME, split_participants
from model.cgmacros.train import ENCODER, HORIZON

CACHE = OUT / "lab_windows_primary.pkl"
LOG = ROOT / "model" / "results" / "lab_log.jsonl"
S30, S60 = 5, 11
_win = np.lib.stride_tricks.sliding_window_view


# ---------------------------------------------------------------- windows and features (vectorized, past only)
def person_windows(g: pd.DataFrame) -> pd.DataFrame:
    """One row per valid as_of in one person's grid: features from the 2 h ending at as_of, targets for the next hour."""
    g = g.reset_index(drop=True)
    n = len(g)
    ti = g["time_idx"].to_numpy()
    glu = g["glucose"].to_numpy(float)
    rows = np.arange(ENCODER - 1, n - HORIZON)
    rows = rows[(ti[rows + HORIZON] - ti[rows - ENCODER + 1]) == ENCODER + HORIZON - 1]
    if not len(rows):
        return pd.DataFrame()
    W = _win(glu, ENCODER)[rows - ENCODER + 1]               # (m, 24), last column is as_of
    Y = _win(glu, HORIZON)[rows + 1]                          # (m, 12), the next hour
    now = W[:, -1]
    t = np.arange(7) * 5.0
    tc = t - t.mean()
    slope30 = ((W[:, -7:] - W[:, -7:].mean(1, keepdims=True)) * tc).sum(1) / (tc ** 2).sum()
    carbs = g["carbs"].to_numpy(float)
    C = _win(carbs, ENCODER)[rows - ENCODER + 1]
    last_meal = np.where(C > 0, np.arange(ENCODER), -1).max(1)
    H6 = _win(g["hr"].to_numpy(float), 6)[rows - 5]
    O6 = _win(g["hr_observed"].to_numpy(bool), 6)[rows - 5]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        hr30 = np.nanmean(np.where(O6, H6, np.nan), 1)
    hour = g["hour"].to_numpy(float)[rows]
    df = pd.DataFrame({
        "participant": g["participant"].iloc[0], "time_idx": ti[rows], "now": now,
        "d5": now - W[:, -2], "d10": now - W[:, -3], "d15": now - W[:, -4], "d30": now - W[:, -7],
        "d60": now - W[:, -13], "d120": now - W[:, 0], "slope30": slope30,
        "accel": (now - W[:, -4]) - (W[:, -4] - W[:, -7]), "min2h": W.min(1), "max2h": W.max(1),
        "mean2h": W.mean(1), "sd2h": W.std(1), "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24), "carbs2h": C.sum(1),
        "meal_min_ago": np.where(last_meal >= 0, (ENCODER - 1 - last_meal) * 5.0, 240.0),
        "meal_carbs": np.where(last_meal >= 0, C[np.arange(len(C)), np.maximum(last_meal, 0)], 0.0),
        "hr30": hr30, "mets30": _win(g["mets"].to_numpy(float), 6)[rows - 5].mean(1),
        "meal_next_hour": _win(carbs, HORIZON)[rows + 1].sum(1) > 0})
    lags = pd.DataFrame(W - now[:, None], columns=[f"lag{k}" for k in range(ENCODER)])
    df = pd.concat([df, lags], axis=1)
    df["y"] = list(Y)
    return df


FEATURES = (["now", "d5", "d10", "d15", "d30", "d60", "d120", "slope30", "accel", "min2h", "max2h", "mean2h", "sd2h",
             "hour_sin", "hour_cos", "carbs2h", "meal_min_ago", "meal_carbs", "hr30", "mets30"]
            + [f"lag{k}" for k in range(ENCODER)])


def all_windows(grid: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    parts = [person_windows(grid[grid.participant == p]) for p in ids]
    return pd.concat([p for p in parts if len(p)], ignore_index=True)


# ---------------------------------------------------------------- TFT baselines on the dev windows
def tft_preds(variant: str, grid: pd.DataFrame, ids: list[str]) -> dict[tuple, tuple]:
    from model.cgmacros.evaluate import load, predict_test
    model, training = load(variant)
    r = predict_test(model, training, grid[grid.participant.isin(ids)])
    keys = zip(r["index"]["participant"], r["index"]["time_idx"])
    return {k: (r["median"][i], r["lo"][i], r["hi"][i]) for i, k in enumerate(keys)}


def build_cache() -> None:
    if SCHEME != "primary":
        raise SystemExit("the lab runs on the primary split only (CGMACROS_SPLIT unset)")
    grid = build_grid()
    train_ids, val_ids, test_ids = split_participants(sorted(grid.participant.unique()))
    dev_ids = sorted(val_ids + test_ids)
    train = all_windows(grid, train_ids)
    dev = all_windows(grid, dev_ids)
    dev["section1_test"] = dev.participant.isin(test_ids)
    for v in ("glucose", "meals_activity"):
        p = tft_preds(v, grid, dev_ids)
        got = [p.get((a, t + 1)) for a, t in zip(dev.participant, dev.time_idx)]   # TFT keys = first decoder step
        for j, part in enumerate(("med", "lo", "hi")):
            dev[f"{v}_{part}"] = [x[j] if x is not None else None for x in got]
    before = len(dev)
    dev = dev[dev["glucose_med"].notna() & dev["meals_activity_med"].notna()].reset_index(drop=True)
    y, a = np.vstack(dev.y.to_numpy()), np.vstack(dev.glucose_med.to_numpy())
    pd.to_pickle({"train": train, "dev": dev, "train_ids": train_ids, "dev_ids": dev_ids}, CACHE)
    print(f"cached: train {len(train)} windows / {train.participant.nunique()} people; dev {len(dev)} of {before} "
          f"windows / {dev.participant.nunique()} people; A T+60 RMSE on dev {rmse(a[:, S60], y[:, S60]):.2f}")


# ---------------------------------------------------------------- scoring
def stack(s: pd.Series) -> np.ndarray:
    return np.vstack(s.to_numpy()).astype(float)


def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def boot_gain(dev: pd.DataFrame, pred: np.ndarray, base: np.ndarray, s: int, n: int = 1000) -> list[float]:
    """95% person-bootstrap interval of RMSE(base) - RMSE(pred) at step s (positive = candidate better)."""
    y = stack(dev.y)[:, s]
    idx = dev.groupby("participant").indices
    people = list(idx)
    rng, g = np.random.default_rng(0), []
    for _ in range(n):
        ix = np.concatenate([idx[p] for p in rng.choice(people, len(people))])
        g.append(rmse(base[ix, s], y[ix]) - rmse(pred[ix, s], y[ix]))
    return [round(float(np.percentile(g, 2.5)), 2), round(float(np.percentile(g, 97.5)), 2)]


def score(dev: pd.DataFrame, med: np.ndarray, lo, hi) -> dict:
    y, A = stack(dev.y), stack(dev.glucose_med)
    out = {}
    for name, m in (("all", np.ones(len(dev), bool)), ("meal_next_hour", dev.meal_next_hour.to_numpy()),
                    ("rising", (dev.d15 >= 20).to_numpy()), ("section1_test7", dev.section1_test.to_numpy())):
        out[name] = {"n": int(m.sum()), "t30": round(rmse(med[m, S30], y[m, S30]), 2),
                     "t60": round(rmse(med[m, S60], y[m, S60]), 2), "A_t60": round(rmse(A[m, S60], y[m, S60]), 2)}
    out["t60_gain_vs_A"] = round(out["all"]["A_t60"] - out["all"]["t60"], 2)
    out["t60_gain_vs_A_ci95"] = boot_gain(dev, med, A, S60)
    if lo is not None:
        out["coverage80_t60"] = round(float(np.mean((y[:, S60] >= lo[:, S60]) & (y[:, S60] <= hi[:, S60]))), 3)
    return out


# ---------------------------------------------------------------- candidates
def _per_step(fit_predict, train, dev, steps):
    Xt, Xd = train[FEATURES].to_numpy(float), dev[FEATURES].to_numpy(float)
    Yt = stack(train.y) - train["now"].to_numpy()[:, None]          # predict the change from now
    out = np.full((len(dev), HORIZON), np.nan)
    for s in steps:
        out[:, s] = fit_predict(Xt, Yt[:, s], Xd)
    return out + dev["now"].to_numpy()[:, None]


def _band(train, med, fit_predict, steps):
    """80% band: residual quantiles on the last fifth of the training people, from a model fit on the rest."""
    people = sorted(train.participant.unique())
    cal_p = set(people[-max(1, len(people) // 5):])
    fit, cal = train[~train.participant.isin(cal_p)], train[train.participant.isin(cal_p)]
    r = stack(cal.y) - _per_step(fit_predict, fit, cal, steps)
    lo, hi = med.copy(), med.copy()
    for s in steps:
        lo[:, s] += np.percentile(r[:, s], 10)
        hi[:, s] += np.percentile(r[:, s], 90)
    return lo, hi


def t_persistence(train, dev):
    return np.repeat(dev["now"].to_numpy()[:, None], HORIZON, 1), None, None


def t_tft_A(train, dev):
    return stack(dev.glucose_med), stack(dev.glucose_lo), stack(dev.glucose_hi)


def t_tft_D_oracle_meals(train, dev):
    return stack(dev.meals_activity_med), stack(dev.meals_activity_lo), stack(dev.meals_activity_hi)


def _ridge(X, y, Xd):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(np.nan_to_num(X), y).predict(np.nan_to_num(Xd))


def _hgb(X, y, Xd):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
                                         random_state=0).fit(X, y).predict(Xd)


def t_ridge(train, dev):
    med = _per_step(_ridge, train, dev, (S30, S60))
    return (med, *_band(train, med, _ridge, (S30, S60)))


def t_hgb(train, dev):
    med = _per_step(_hgb, train, dev, (S30, S60))
    return (med, *_band(train, med, _hgb, (S30, S60)))


PERSON = ["p_mean", "p_sd", "p_days", "p_meal_rise", "p_n_meals", "p_resid60"]
LONG = ["m6h", "sd6h", "m24h", "sd24h", "now_minus_m24h"]


def extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """Past-only extras per (participant, time_idx):
    LONG    mean/sd of glucose over the previous 6 h and 24 h (readings present only);
    PERSON  the person's own history before as_of: mean/sd of all earlier glucose, days observed, mean 60-min rise
            after earlier logged meals (only meals whose hour ended by as_of), count of those meals, and the mean
            60-min change after earlier moments (a personal drift term; only hours that ended by as_of)."""
    grid = build_grid()
    out = []
    for p, w in df.groupby("participant", sort=False):
        g = grid[grid.participant == p].set_index("time_idx").sort_index()
        full = g.reindex(range(int(g.index.min()), int(g.index.max()) + 1))
        s = full["glucose"]
        e = pd.DataFrame(index=full.index)
        e["m6h"] = s.rolling(72, min_periods=12).mean()
        e["sd6h"] = s.rolling(72, min_periods=12).std()
        e["m24h"] = s.rolling(288, min_periods=36).mean()
        e["sd24h"] = s.rolling(288, min_periods=36).std()
        e["now_minus_m24h"] = s - e["m24h"]
        e["p_mean"] = s.expanding(min_periods=1).mean()
        e["p_sd"] = s.expanding(min_periods=2).std()
        e["p_days"] = s.notna().cumsum() / 288.0
        fut60 = s.shift(-HORIZON)                                   # glucose 60 min later, known 60 min later
        change = (fut60 - s).shift(HORIZON)                         # aligned to when it becomes known
        e["p_resid60"] = change.expanding(min_periods=12).mean()
        maxnext = s[::-1].rolling(HORIZON, min_periods=6).max()[::-1].shift(-1)
        meal = full["carbs"].fillna(0) > 0
        rise = (maxnext - s).where(meal).shift(HORIZON)             # meal rise, known once its hour has passed
        e["p_meal_rise"] = rise.expanding(min_periods=1).mean()
        e["p_n_meals"] = rise.notna().cumsum()
        out.append(e.loc[w["time_idx"].to_numpy()].set_index(w.index))
    return pd.concat(out).loc[df.index]


def _with(train, dev, cols):
    tr, dv = train.copy(), dev.copy()
    for d in (tr, dv):
        d[cols] = extra_features(d)[cols].to_numpy()
    return tr, dv


def _hgb_cols(cols):
    def trial(train, dev):
        global FEATURES
        base = FEATURES
        tr, dv = _with(train, dev, cols)
        FEATURES = base + cols
        try:
            med = _per_step(_hgb, tr, dv, (S30, S60))
            return (med, *_band(tr, med, _hgb, (S30, S60)))
        finally:
            FEATURES = base
    return trial


TRIALS = {"persistence": t_persistence, "tft_A": t_tft_A, "tft_D_oracle_meals": t_tft_D_oracle_meals,
          "ridge": t_ridge, "hgb": t_hgb, "hgb_long": _hgb_cols(LONG), "hgb_person": _hgb_cols(PERSON),
          "hgb_long_person": _hgb_cols(LONG + PERSON)}


def _hgb_big(X, y, Xd):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=800, learning_rate=0.03, max_leaf_nodes=63, min_samples_leaf=50,
                                         l2_regularization=1.0, random_state=0).fit(X, y).predict(Xd)


def t_hgb_big_long_person(train, dev):
    global FEATURES
    base = FEATURES
    tr, dv = _with(train, dev, LONG + PERSON)
    FEATURES = base + LONG + PERSON
    try:
        med = _per_step(_hgb_big, tr, dv, (S30, S60))
        return (med, *_band(tr, med, _hgb_big, (S30, S60)))
    finally:
        FEATURES = base


def t_blend_hgb_lp_tftA(train, dev):
    med, lo, hi = TRIALS["hgb_long_person"](train, dev)
    a = stack(dev.glucose_med)
    return 0.7 * med + 0.3 * a, lo, hi                 # fixed 70/30 weights, not tuned on dev


TRIALS.update({"hgb_big_long_person": t_hgb_big_long_person, "blend_hgb_lp_tftA": t_blend_hgb_lp_tftA})

# ---------------------------------------------------------------- retrieval: the person's own similar past moments
KNN = ["knn_d30", "knn_d60", "knn_dist", "knn_n"]


def knn_features(df: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    """For each window, the k most similar EARLIER windows of the same person whose next hour had fully passed by
    as_of (time_idx_j + 12 <= time_idx_i). Similarity: Euclidean on the 2-h shape (lags) plus the level (now / 2).
    Returns their mean change at +30 and +60 min, mean distance, and how many candidates existed."""
    lag_cols = [f"lag{j}" for j in range(ENCODER)]
    res = pd.DataFrame(np.nan, index=df.index, columns=KNN)
    for _, w in df.groupby("participant", sort=False):
        w = w.sort_values("time_idx")
        Z = np.c_[w[lag_cols].to_numpy(float), w["now"].to_numpy(float)[:, None] / 2.0]
        Y = stack(w.y) - w["now"].to_numpy()[:, None]
        t = w["time_idx"].to_numpy()
        avail = np.searchsorted(t, t - HORIZON, side="right")      # windows [0, avail) have finished hours
        for s in range(0, len(w), 512):
            e = min(s + 512, len(w))
            m = int(avail[e - 1])
            if m == 0:
                continue
            D = np.sqrt(((Z[s:e, None, :] - Z[None, :m, :]) ** 2).sum(-1))
            D[np.arange(m)[None, :] >= avail[s:e, None]] = np.inf
            nn = np.argsort(D, 1)[:, :k]
            dn = np.take_along_axis(D, nn, 1)
            ok = np.isfinite(dn)
            cnt = ok.sum(1)
            w30 = np.where(ok, Y[nn, S30], 0).sum(1)
            w60 = np.where(ok, Y[nn, S60], 0).sum(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                res.loc[w.index[s:e], "knn_d30"] = np.where(cnt > 0, w30 / cnt, np.nan)
                res.loc[w.index[s:e], "knn_d60"] = np.where(cnt > 0, w60 / cnt, np.nan)
                res.loc[w.index[s:e], "knn_dist"] = np.where(cnt > 0, np.where(ok, dn, 0).sum(1) / cnt, np.nan)
            res.loc[w.index[s:e], "knn_n"] = np.minimum(avail[s:e], k)
    return res


def t_hgb_long_person_knn(train, dev):
    global FEATURES
    base = FEATURES
    tr, dv = _with(train, dev, LONG + PERSON)
    for d in (tr, dv):
        d[KNN] = knn_features(d)[KNN].to_numpy()
    FEATURES = base + LONG + PERSON + KNN
    try:
        med = _per_step(_hgb, tr, dv, (S30, S60))
        return (med, *_band(tr, med, _hgb, (S30, S60)))
    finally:
        FEATURES = base


TRIALS["hgb_long_person_knn"] = t_hgb_long_person_knn

# ---------------------------------------------------------------- Chronos (runs in .venv-chronos; see chronos_run.py)
CTX_LEN = 512                                         # ~42 h of 5-min readings; gaps are NaN
CTX_FILE = OUT / "lab_dev_context.npy"


def export_context() -> None:
    """Glucose up to and including as_of for every dev window (same row order as the cache), NaN in gaps."""
    c = pd.read_pickle(CACHE)
    grid = build_grid()
    out = np.full((len(c["dev"]), CTX_LEN), np.nan, dtype=np.float32)
    for p, w in c["dev"].groupby("participant"):
        g = grid[grid.participant == p].set_index("time_idx")["glucose"]
        lo = int(g.index.min())
        s = g.reindex(range(lo, int(g.index.max()) + 1)).to_numpy(np.float32)
        pad = np.r_[np.full(CTX_LEN, np.nan, np.float32), s]
        for row, t in zip(w.index, w.time_idx):
            end = int(t) - lo + CTX_LEN + 1                  # one past as_of in the padded series
            out[row] = pad[end - CTX_LEN:end]
    assert np.allclose(out[:, -1], c["dev"]["now"].to_numpy()), "context does not end at as_of"
    np.save(CTX_FILE, out)
    print(f"context {out.shape} -> {CTX_FILE}")


def _chronos(name):
    def trial(train, dev):
        q = np.load(OUT / f"lab_pred_{name}.npy")             # (n, 12, 3): p10, p50, p90
        assert len(q) == len(dev)
        return q[:, :, 1], q[:, :, 0], q[:, :, 2]
    return trial


TRIALS.update({"chronos_bolt_small": _chronos("chronos_bolt_small"), "chronos2": _chronos("chronos2")})


def t_blend_hgb_lp_chronos2(train, dev):
    med, _, _ = TRIALS["hgb_long_person"](train, dev)
    c_med, c_lo, c_hi = _chronos("chronos2")(train, dev)
    m = 0.7 * med + 0.3 * c_med                       # fixed 70/30, as for the TFT blend; not tuned on dev
    return m, m + (c_lo - c_med), m + (c_hi - c_med)  # Chronos-2's band width around the blended median


TRIALS["blend_hgb_lp_chronos2"] = t_blend_hgb_lp_chronos2


def run_trial(name: str, note: str = "") -> dict:
    c = pd.read_pickle(CACHE)
    t0 = time.time()
    try:
        med, lo, hi = TRIALS[name](c["train"], c["dev"])
        res = score(c["dev"], med, lo, hi)
    except Exception as e:                                # failures are logged too
        res = {"error": f"{type(e).__name__}: {e}"}
    res = {"trial": name, "when": datetime.now(timezone.utc).isoformat(timespec="seconds"), "note": note,
           "seconds": round(time.time() - t0, 1), **res}
    with open(LOG, "a") as fh:
        fh.write(json.dumps(res) + "\n")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--trial", nargs="*", default=[])
    ap.add_argument("--note", default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--export-context", action="store_true")
    a = ap.parse_args()
    if a.cache:
        build_cache()
    if a.export_context:
        export_context()
    if a.list:
        print(", ".join(TRIALS))
    for name in a.trial:
        r = run_trial(name, a.note)
        print(name, json.dumps({k: r.get(k) for k in ("all", "rising", "meal_next_hour", "t60_gain_vs_A",
                                                       "t60_gain_vs_A_ci95", "coverage80_t60", "error")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
