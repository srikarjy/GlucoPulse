"""Event model: predict the urgent event directly from glucose history, instead of thresholding a median forecast.

    CGMACROS_SPLIT=confirm python -m model.cgmacros.event     # pre-registered test (9 new people)
    python -m model.cgmacros.event                            # exploratory: primary train 33 -> primary held-out 12

Why: the rule calls 'urgent' only when the forecaster's MEDIAN crosses 250 (or 70), but the label asks whether the
path crosses at ANY time in the next hour, and the forecaster's median barely continues a rise (of 20 eval cases
rising >= 20 mg/dL per 15 min, 14 had a 60-min median at or below the current reading and the other 6 rose by at
most 16 mg/dL, while their 15-min trend implied +80 or more). This model is trained on the
event itself, from history only, on the forecaster's TRAINING people (many more people than the 12-person ceiling).
Forecaster outputs are not features: they are in-sample on the training people.

PRE-REGISTERED (written 2026-11-16, before any event-model output for the test people was computed):
  Training     the 31 confirm-split training people, every 30-min moment (same candidate_moments and outcome_label as
               the eval cases). Target: urgent (sustained > 250 or sustained < 70 within 60 min).
  Features     history only, ending at as_of: current glucose; changes over 5/10/15/30/60 min; 30-min least-squares
               slope; acceleration (last 15-min change minus the previous 15-min change); 2-h min/max/mean/sd;
               hour of day (sin, cos); carbs logged in 2 h; minutes since and carbs of the last logged meal; 30-min
               mean heart rate (missing if unobserved) and METs.
  Model        HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
               l2_regularization=1.0, class_weight="balanced", random_state=0). No tuning.
  Threshold t  from 5-fold GroupKFold (by person) out-of-fold probabilities on the training people only: the smallest
               t whose specificity on non-urgent moments is >= 0.985 (the rule's urgent specificity on the primary
               12-person pool, model/results/cgmacros_ceiling.json). Test labels are never used to set t.
  Policy       "event": urgent if the rule says urgent OR P(urgent) >= t; otherwise the rule's level.
  Comparator   "trend" (fixed now, not tuned): urgent if the rule says urgent OR current + 4 x (15-min change) > 250
               or < 70; otherwise the rule's level.
  Criteria     on the 9 NEW people, versus the rule, BOTH of:
                 1. urgent recall higher, with the 95% participant-bootstrap interval of the gain excluding 0;
                 2. urgent false-alarm rate (non-urgent moments called urgent) at most 2 points higher.
               Power (simulated from the rule's misses): a gain of about +15 points is needed for the interval to
               exclude 0 reliably; smaller real gains will likely read as inconclusive.
  Secondary    event vs trend (interval of the recall difference); highs and lows separately; AUROC; under/over-triage
               and urgent hours called routine; everything on all 14 people the new forecaster never trained on
               (9 new + 5 early-stopping people; the rule's forecasts are mildly optimistic on the 5).
  Exploratory  the primary split (train 33 -> held-out 12). Those 12 people's eval cases were used to explore
               thresholds before this was written, so that run is development, not evidence.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from model.cgmacros.cases import OUT, candidate_moments, outcome_label
from model.cgmacros.ceiling import POOL, boot_diff, threshold_at_specificity
from model.cgmacros.dataset import ROOT, build_grid
from model.cgmacros.splits import SCHEME, split_participants
from model.cgmacros.train import ENCODER, HORIZON

RESULTS = ROOT / "model" / "results" / f"cgmacros_event_{SCHEME}.json"
TRAIN_CACHE = OUT / f"event_train_{SCHEME}.csv"
SPEC_TARGET = 0.985
MAX_FALSE_ALARM_INCREASE = 0.02
O = {"routine": 0, "monitor": 1, "urgent": 2}
FEATURES = ["now", "d5", "d10", "d15", "d30", "d60", "slope30", "accel", "min2h", "max2h", "mean2h", "sd2h",
            "hour_sin", "hour_cos", "carbs2h", "meal_min_ago", "meal_carbs", "hr30", "mets30"]


def history_features(h: pd.DataFrame) -> dict[str, float]:
    """Features from the 2-h window ending at as_of (24 five-minute rows). Nothing after as_of."""
    g = h["glucose"].to_numpy(dtype=float)
    last30 = h.tail(6)
    meals = h[h.carbs > 0]
    hr_obs = float(last30["hr_observed"].mean())
    hour = float(h["hour"].iloc[-1])
    return {
        "now": g[-1], "d5": g[-1] - g[-2], "d10": g[-1] - g[-3], "d15": g[-1] - g[-4], "d30": g[-1] - g[-7],
        "d60": g[-1] - g[-13], "slope30": float(np.polyfit(np.arange(7) * 5.0, g[-7:], 1)[0]),
        "accel": (g[-1] - g[-4]) - (g[-4] - g[-7]), "min2h": g.min(), "max2h": g.max(), "mean2h": g.mean(),
        "sd2h": g.std(), "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
        "carbs2h": float(h["carbs"].sum()),
        "meal_min_ago": float((h["ts"].iloc[-1] - meals["ts"].iloc[-1]).total_seconds() / 60) if len(meals) else np.nan,
        "meal_carbs": float(meals["carbs"].iloc[-1]) if len(meals) else 0.0,
        "hr30": float(last30["hr"].mean()) if hr_obs > 0 else np.nan, "mets30": float(last30["mets"].mean())}


def moments(grid: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    rows = []
    for p, g, i in candidate_moments(grid, ids):
        h = g.iloc[i - ENCODER + 1:i + 1]
        future = g["glucose"].to_numpy()[i + 1:i + 1 + HORIZON]
        assert h["ts"].iloc[-1] == g["ts"].iloc[i] and len(future) == HORIZON   # features end at as_of
        rows.append({"participant": p, "as_of": g["ts"].iloc[i].isoformat(), "label": outcome_label(future),
                     "actual_max": float(future.max()), "actual_min": float(future.min()), **history_features(h)})
    return pd.DataFrame(rows)


def make():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                                          class_weight="balanced", random_state=0)


def fit_with_threshold(train: pd.DataFrame):
    X, y, groups = train[FEATURES].to_numpy(), (train["label"] == "urgent").to_numpy(), train["participant"]
    oof = np.zeros(len(train))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        oof[te] = make().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    t = threshold_at_specificity(oof, ~y, SPEC_TARGET)
    return make().fit(X, y), t, {"oof_auroc": round(roc_auc_score(y, oof), 3),
                                 "oof_specificity": round(float((oof[~y] < t).mean()), 4),
                                 "oof_recall": round(float((oof[y] >= t).mean()), 3)}


def score(df: pd.DataFrame, level: np.ndarray, rule: np.ndarray, prob: np.ndarray, trend_level: np.ndarray) -> dict:
    y = df["label"].map(O).to_numpy()
    pos, hit, rule_hit, trend_hit = y == 2, level == 2, rule == 2, trend_level == 2
    high = pos & (df["actual_max"].to_numpy() > 250)
    low = pos & ~high
    every = np.ones(len(df), bool)

    def r(a, m):
        return round(float(a[m].mean()), 3) if m.any() else None

    return {"moments": len(df), "urgent": int(pos.sum()),
            "urgent_people": int(df.loc[pos, "participant"].nunique()),
            "rule_recall": r(rule_hit, pos), "recall": r(hit, pos), "trend_recall": r(trend_hit, pos),
            "recall_gain_vs_rule_ci95": boot_diff(df, hit, rule_hit, pos),
            "recall_diff_vs_trend_ci95": boot_diff(df, hit, trend_hit, pos),
            "rule_false_alarm": r(rule_hit, ~pos), "false_alarm": r(hit, ~pos), "trend_false_alarm": r(trend_hit, ~pos),
            "high": {"n": int(high.sum()), "rule": r(rule_hit, high), "event": r(hit, high), "trend": r(trend_hit, high)},
            "low": {"n": int(low.sum()), "rule": r(rule_hit, low), "event": r(hit, low), "trend": r(trend_hit, low)},
            "auroc": round(roc_auc_score(pos, prob), 3) if 0 < pos.sum() < len(pos) else None,
            "under_triage": {"rule": r(rule < y, every), "event": r(level < y, every), "trend": r(trend_level < y, every)},
            "over_triage": {"rule": r(rule > y, every), "event": r(level > y, every), "trend": r(trend_level > y, every)},
            "urgent_called_routine": {"rule": int((pos & (rule == 0)).sum()), "event": int((pos & (level == 0)).sum()),
                                      "trend": int((pos & (trend_level == 0)).sum())}}


def main() -> int:
    grid = build_grid()
    train_ids, _, _ = split_participants(sorted(grid.participant.unique()))
    if TRAIN_CACHE.exists():
        train = pd.read_csv(TRAIN_CACHE)
    else:
        train = moments(grid, train_ids)
        train.to_csv(TRAIN_CACHE, index=False)
    model, t, cv = fit_with_threshold(train)
    print(f"train: {len(train)} moments, {train.participant.nunique()} people, {int((train.label == 'urgent').sum())} "
          f"urgent; threshold {t:.4f}; out-of-fold {cv}", flush=True)

    pool = pd.read_csv(POOL)                     # rule levels (+ split) for the held-out people, built earlier
    test = moments(grid, sorted(pool["participant"].unique()))
    df = test.merge(pool[["participant", "as_of", "split", "rule", "label"]], on=["participant", "as_of"],
                    suffixes=("", "_pool"))
    assert len(df) == len(pool) and (df["label"] == df["label_pool"]).all(), "pool and history moments disagree"
    assert not set(df["participant"]) & set(train["participant"]), "test people in training"
    prob = model.predict_proba(df[FEATURES].to_numpy())[:, 1]
    rule = df["rule"].map(O).to_numpy()
    level = np.where((rule == 2) | (prob >= t), 2, rule)
    proj = df["now"].to_numpy() + 4 * df["d15"].to_numpy()
    trend_level = np.where((rule == 2) | (proj > 250) | (proj < 70), 2, rule)

    headline = "new_9_people" if SCHEME == "confirm" else "test_7_people"
    out = {"scheme": SCHEME, "status": "pre-registered test" if SCHEME == "confirm" else "EXPLORATORY",
           "preregistration": __doc__.split("PRE-REGISTERED")[1].strip(), "features": FEATURES,
           "threshold": t, "training": {"people": int(train.participant.nunique()), "moments": len(train), **cv}}
    for name, mask in ((headline, (df["split"] == "test").to_numpy()), ("all_held_out_people", np.ones(len(df), bool))):
        out[name] = score(df[mask].reset_index(drop=True), level[mask], rule[mask], prob[mask], trend_level[mask])
    h = out[headline]
    out["criteria"] = {
        "1_recall_gain_ci_excludes_0": bool(h["recall_gain_vs_rule_ci95"][0] > 0),
        "2_false_alarm_increase_at_most_2_points": bool(h["false_alarm"] - h["rule_false_alarm"]
                                                        <= MAX_FALSE_ALARM_INCREASE)}
    out["confirmed"] = SCHEME == "confirm" and all(out["criteria"].values())
    RESULTS.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps({k: out[k] for k in ("status", "criteria", "confirmed")}), f"(headline: {headline})")
    print(json.dumps(h, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
