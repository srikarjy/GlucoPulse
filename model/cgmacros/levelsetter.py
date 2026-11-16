"""Level-setter: the rule keeps 'urgent'; a classifier decides routine vs monitor below it. Evaluate, then export.

    python -m model.cgmacros.levelsetter     # needs data/cgmacros/triage/ceiling_pool.csv (python -m model.cgmacros.ceiling)

PRE-REGISTERED (written before the first run, 2026-11-16):
  Primary policy  "rule_urgent+clf": level = urgent if the rule says urgent; otherwise monitor if the classifier's
                  P(at least monitor) >= t_m, else routine. It may turn a rule 'monitor' into 'routine'; it never
                  lowers a rule 'urgent'.
  Primary model   multinomial logistic regression (exports to plain coefficients, so the agents need no sklearn;
                  it had the strongest routine/monitor signal on the test people in model/cgmacros/ceiling.py).
  Protocol        nested leave-one-participant-out over the 12 people the forecaster never trained on: for each
                  held-out person, the model AND t_m come only from the other 11 (t_m matches the rule's
                  'routine' specificity on those 11 people's own out-of-fold predictions).
  Criterion       on the 7 TEST people, versus the rule: under-triage is lower, with the 95% participant-bootstrap
                  interval of (rule - policy) under-triage excluding 0, AND over-triage is at most 5 points higher.
  Everything else (boosted trees, classifier-only, max(rule, classifier)) is secondary.
  Caveat          the routine/monitor signal was found on these same 12 people, so this shows whether a deployable
                  version keeps it; it cannot confirm it. Confirmation needs people neither model has seen.
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter

import numpy as np
import pandas as pd

from agents.triage.features import features
from model.cgmacros.ceiling import OUT, POOL, models, threshold_at_specificity
from model.cgmacros.dataset import ROOT

RESULTS = ROOT / "model" / "results" / "cgmacros_levelsetter.json"
EXPORT = ROOT / "agents" / "triage" / "levelsetter_model.json"
O = {"routine": 0, "monitor": 1, "urgent": 2}
MAX_OVER_TRIAGE_INCREASE = 0.05


def _fit_predict(make, X, y, train, test):
    clf = make()
    clf.fit(X[train], y[train])
    prob = np.zeros((int(test.sum()), 3))
    prob[:, clf.classes_] = clf.predict_proba(X[test])
    return prob


def nested(df, cols, make):
    """Per held-out person: P(at least monitor), P(urgent), and thresholds chosen without that person's labels."""
    X, y = df[cols].to_numpy(), df["label"].map(O).to_numpy()
    rule = df["rule"].map(O).to_numpy()
    people = df["participant"].to_numpy()
    p_m, p_u, t_m, t_u = (np.zeros(len(df)) for _ in range(4))
    for p in np.unique(people):
        test, dev = people == p, people != p
        dev_idx = np.flatnonzero(dev)
        dev_people = people[dev]
        inner = np.zeros((len(dev_idx), 3))
        for q in np.unique(dev_people):          # out-of-fold on the 11 dev people
            tr, te = dev.copy(), np.zeros(len(df), bool)
            tr[dev_idx[dev_people == q]] = False
            te[dev_idx[dev_people == q]] = True
            inner[dev_people == q] = _fit_predict(make, X, y, tr, te)
        y_dev, r_dev = y[dev], rule[dev]
        neg_m = y_dev == 0
        tm = threshold_at_specificity(inner[:, 1] + inner[:, 2], neg_m, float((r_dev[neg_m] == 0).mean()))
        neg_u = y_dev < 2
        tu = threshold_at_specificity(inner[:, 2], neg_u, float((r_dev[neg_u] < 2).mean()))
        prob = _fit_predict(make, X, y, dev, test)
        p_m[test], p_u[test], t_m[test], t_u[test] = prob[:, 1] + prob[:, 2], prob[:, 2], tm, tu
    return p_m, p_u, t_m, t_u


def policies(rule, p_m, p_u, t_m, t_u) -> dict[str, np.ndarray]:
    below = np.where(p_m >= t_m, 1, 0)
    clf = np.where(p_u >= t_u, 2, below)
    return {"rule_urgent+clf": np.where(rule == 2, 2, below), "classifier_only": clf,
            "max(rule,classifier)": np.maximum(rule, clf)}


def boot(df, f, n=2000):
    people = df["participant"].unique().tolist()
    idx = {p: np.flatnonzero((df["participant"] == p).to_numpy()) for p in people}
    rng, d = random.Random(7), []
    for _ in range(n):
        d.append(f(np.concatenate([idx[rng.choice(people)] for _ in people])))
    d.sort()
    return [round(d[int(0.025 * n)], 4), round(d[int(0.975 * n) - 1], 4)]


def report(df, pred, rule) -> dict:
    y = df["label"].map(O).to_numpy()
    under, over = (pred < y), (pred > y)
    r_under, r_over = (rule < y), (rule > y)
    down = (rule == 1) & (pred == 0)
    by_person = {}
    for p in sorted(df["participant"].unique()):
        m = down & (df["participant"] == p).to_numpy()
        if m.any():
            by_person[p] = {"downgrades": int(m.sum()), "needed_attention": int((y[m] >= 1).sum())}
    return {"n": len(df), "accuracy": round(float((pred == y).mean()), 3),
            "under_triage": round(float(under.mean()), 4), "over_triage": round(float(over.mean()), 4),
            "rule_under_triage": round(float(r_under.mean()), 4), "rule_over_triage": round(float(r_over.mean()), 4),
            "under_reduction_ci95_by_person": boot(df, lambda s: r_under[s].mean() - under[s].mean()),
            "over_increase_ci95_by_person": boot(df, lambda s: over[s].mean() - r_over[s].mean()),
            "severe_under_triage_urgent_called_routine": int(((y == 2) & (pred == 0)).sum()),
            "rule_severe_under_triage_urgent_called_routine": int(((y == 2) & (rule == 0)).sum()),
            "urgent_recall": round(float((pred[y == 2] == 2).mean()), 3),
            "routine_recall": round(float((pred[y == 0] == 0).mean()), 3),
            "monitor_recall": round(float((pred[y == 1] >= 1).mean()), 3),
            "downgrades_rule_monitor_to_routine": int(down.sum()),
            "downgrades_that_needed_attention": int((y[down] >= 1).sum()),
            "downgrades_that_were_urgent": int((y[down] == 2).sum()),
            "downgrades_by_person": by_person,
            "upgrades": int((pred > rule).sum())}


def export(df, cols):
    """Final logistic on all 12 people; t_m from their own LOPO out-of-fold predictions. Written as plain numbers."""
    make = models()["logistic"]
    X, y, rule = df[cols].to_numpy(), df["label"].map(O).to_numpy(), df["rule"].map(O).to_numpy()
    people = df["participant"].to_numpy()
    oof = np.zeros((len(df), 3))
    for p in np.unique(people):
        oof[people == p] = _fit_predict(make, X, y, people != p, people == p)
    neg = y == 0
    t_m = threshold_at_specificity(oof[:, 1] + oof[:, 2], neg, float((rule[neg] == 0).mean()))
    pipe = make()
    pipe.fit(X, y)
    imp = pipe.named_steps["simpleimputer"]
    sc = pipe.named_steps["standardscaler"]
    lr = pipe.named_steps["logisticregression"]
    assert list(lr.classes_) == [0, 1, 2]
    # Parity references are SYNTHETIC (medians plus noise, some missing): no participant data enters the repo.
    rng = np.random.default_rng(0)
    ref_X = imp.statistics_ + rng.normal(0.0, 1.0, (5, len(cols))) * sc.scale_[:len(cols)]
    for r, i in enumerate(imp.indicator_.features_[:5]):
        ref_X[r, i] = np.nan
    model = {"description": "rule_urgent+clf level-setter: logistic P(at least monitor) on agents/triage/features.py",
             "features": cols, "classes": ["routine", "monitor", "urgent"],
             "impute_median": [float(v) for v in imp.statistics_],
             "missing_indicator_features": [int(i) for i in imp.indicator_.features_],
             "scale_mean": [float(v) for v in sc.mean_], "scale_std": [float(v) for v in sc.scale_],
             "coef": [[float(v) for v in row] for row in lr.coef_], "intercept": [float(v) for v in lr.intercept_],
             "threshold_at_least_monitor": float(t_m),
             "trained_on": "12 CGMacros participants the forecaster never trained on (7 test + 5 validation)",
             "reference": [{"features": [None if math.isnan(v) else float(v) for v in row],
                            "p_at_least_monitor": float(pp[1] + pp[2])}
                           for row, pp in zip(ref_X, pipe.predict_proba(ref_X))]}
    EXPORT.write_text(json.dumps(model, indent=1) + "\n")
    return t_m


def main() -> int:
    df = pd.read_csv(POOL)
    cols = [c for c in df.columns if c not in ("participant", "split", "as_of", "label", "rule")]
    packets = json.loads((OUT / "packets.json").read_text())
    assert cols == list(features(packets["case-000"]))
    rule = df["rule"].map(O).to_numpy()
    test = (df["split"] == "test").to_numpy()
    key = {(packets[c]["patient"], packets[c]["as_of"]) for c in packets}
    rows98 = np.array([i for i, (p, a) in enumerate(zip(df.participant, df.as_of)) if (p, a) in key])
    out = {"preregistration": __doc__.split("PRE-REGISTERED")[1].strip(), "labels_test": dict(Counter(df.label[test]))}
    for mname, make in models().items():
        p_m, p_u, t_m, t_u = nested(df, cols, make)
        for pname, pred in policies(rule, p_m, p_u, t_m, t_u).items():
            name = f"{mname}/{pname}"
            out[name] = {"test_7_people": report(df[test].reset_index(drop=True), pred[test], rule[test]),
                         "all_12_people": report(df, pred, rule),
                         "the_98_eval_cases": report(df.iloc[rows98].reset_index(drop=True), pred[rows98],
                                                     rule[rows98])}
            t = out[name]["test_7_people"]
            print(f"{name:34s} under {t['rule_under_triage']}->{t['under_triage']} "
                  f"CI{t['under_reduction_ci95_by_person']}  over {t['rule_over_triage']}->{t['over_triage']}  "
                  f"downgrades {t['downgrades_rule_monitor_to_routine']} "
                  f"(needed attention {t['downgrades_that_needed_attention']})", flush=True)
    prim = out["logistic/rule_urgent+clf"]["test_7_people"]
    out["primary_criterion_met"] = bool(prim["under_reduction_ci95_by_person"][0] > 0 and
                                        prim["over_triage"] - prim["rule_over_triage"] <= MAX_OVER_TRIAGE_INCREASE)
    out["exported_threshold"] = export(df, cols)
    RESULTS.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print("primary criterion met:", out["primary_criterion_met"], f"-> {RESULTS}, {EXPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
