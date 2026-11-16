"""Is there signal in the triage packet beyond the rule? A learned ceiling, no LLM.

    python -m model.cgmacros.ceiling            # builds the pool once (~8 min), then scores
    python -m model.cgmacros.ceiling --reuse    # reuse data/cgmacros/triage/ceiling_pool.csv

Every candidate moment (30-min stride) of the 12 participants the forecaster never trained on
(7 test + 5 validation) is turned into the same packet the agents see: glucose/lifestyle/data-quality
summaries plus the four scenario forecasts. Features are the packet's numeric fields only; the
outcome (next 60 min) is used for the label and nothing else. The rule is the real agents/triage/rules.py.

Classifiers: multinomial logistic regression and gradient-boosted trees, leave-one-participant-out,
so every prediction is for a person the classifier never saw.

Comparison is at a MATCHED operating point, not accuracy (the pool is ~85% routine): the classifier's
urgent threshold is set so its urgent specificity equals the rule's, then urgent recall is compared.
Same for "at least monitor" vs routine.

Pre-registered criterion (fixed before any result was seen): the classifier "clearly beats the rule"
only if, on the 7 TEST participants, the 95% participant-bootstrap interval of
(classifier urgent recall - rule urgent recall) at matched specificity excludes 0.
The 12-person pool is supporting evidence only: validation people were used for the forecaster's
early stopping, so their forecast features are mildly optimistic. A run without cgmacros-035
(146 of 256 urgent moments) checks the result is not one person.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from agents.triage.features import features
from agents.triage.rules import rule_level
from model.cgmacros.cases import OUT, candidate_moments, outcome_label, summaries
from model.cgmacros.dataset import ROOT, build_grid
from model.cgmacros.scenarios import MENU, ScenarioForecaster, scenario_resource
from model.cgmacros.splits import split_participants
from model.cgmacros.train import ENCODER, HORIZON

POOL = OUT / "ceiling_pool.csv"
RESULTS = ROOT / "model" / "results" / "cgmacros_ceiling.json"
O = {"routine": 0, "monitor": 1, "urgent": 2}
def build_pool() -> pd.DataFrame:
    grid = build_grid()
    _, val_ids, test_ids = split_participants(sorted(grid.participant.unique()))
    meal_model, history_model = ScenarioForecaster("meals_activity"), ScenarioForecaster("glucose")
    rows = []
    for split, ids in (("test", test_ids), ("val", val_ids)):
        for p, g, i in candidate_moments(grid, ids):
            hist = g.iloc[i - ENCODER + 1:i + 1]
            as_of = hist["ts"].iloc[-1].isoformat()
            q = meal_model.forecast(hist, MENU)
            q["history_only"] = history_model.forecast(hist, ["no_new_meal"])["no_new_meal"]
            scen = {n: scenario_resource(p, as_of, n, q[n], "glucose" if n == "history_only" else "meals_activity")
                    for n in q}
            for n in ("meal_30g_now", "meal_60g_now"):
                if any(m < b - 1.0 for m, b in zip(q[n]["p50"], q["no_new_meal"]["p50"])):
                    scen[n]["consistency_warning"] = "dips below no-meal"
            packet = {**summaries(p, as_of, hist), "scenarios": scen}
            future = g["glucose"].to_numpy()[i + 1:i + 1 + HORIZON]
            assert g["ts"].iloc[i].isoformat() == as_of and len(future) == HORIZON   # features end at as_of
            rows.append({"participant": p, "split": split, "as_of": as_of, "label": outcome_label(future),
                         "rule": rule_level(packet["glucose_summary"], scen["no_new_meal"], scen["history_only"]),
                         **features(packet)})
        print(f"{split}: {sum(r['split'] == split for r in rows)} moments", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(POOL, index=False)
    return df


def models():
    return {
        "logistic": lambda: make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                                          LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)),
        "boosted_trees": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                                l2_regularization=1.0, class_weight="balanced",
                                                                random_state=0),
    }


def lopo(df: pd.DataFrame, cols: list[str], make) -> np.ndarray:
    """Out-of-fold class probabilities [routine, monitor, urgent], one held-out participant at a time."""
    prob = np.zeros((len(df), 3))
    y = df["label"].map(O).to_numpy()
    for p in df["participant"].unique():
        test = (df["participant"] == p).to_numpy()
        clf = make()
        clf.fit(df.loc[~test, cols].to_numpy(), y[~test])
        prob[np.ix_(test, clf.classes_)] = clf.predict_proba(df.loc[test, cols].to_numpy())
    return prob


def threshold_at_specificity(score: np.ndarray, negative: np.ndarray, spec: float) -> float:
    """Smallest threshold whose specificity on the negatives is >= spec (reading the ROC at the rule's point)."""
    neg = np.sort(score[negative])
    k = int(np.ceil(spec * len(neg)))
    if k >= len(neg):
        return np.inf
    return float(np.nextafter(neg[k - 1], np.inf)) if k > 0 else -np.inf


def boot_diff(df: pd.DataFrame, a: np.ndarray, b: np.ndarray, pos: np.ndarray, n: int = 2000) -> list[float]:
    """95% interval of recall(a) - recall(b) on positives, resampling PARTICIPANTS (moments within a person correlate)."""
    people = df["participant"].unique().tolist()
    idx = {p: np.flatnonzero((df["participant"] == p).to_numpy()) for p in people}
    rng, draws = random.Random(7), []
    for _ in range(n):
        sel = np.concatenate([idx[rng.choice(people)] for _ in people])
        m = pos[sel]
        if m.any():
            draws.append(a[sel][m].mean() - b[sel][m].mean())
    draws.sort()
    return [round(draws[int(0.025 * len(draws))], 3), round(draws[int(0.975 * len(draws)) - 1], 3)]


def evaluate(df: pd.DataFrame, prob: np.ndarray) -> dict:
    y = df["label"].map(O).to_numpy()
    rule = df["rule"].map(O).to_numpy()
    out = {"moments": len(df), "labels": dict(Counter(df["label"])),
           "urgent_people": sorted(df.loc[df.label == "urgent", "participant"].unique().tolist())}
    for name, pos, score, rule_hit in (
            ("urgent", y == 2, prob[:, 2], rule == 2),
            ("at_least_monitor", y >= 1, prob[:, 1] + prob[:, 2], rule >= 1)):
        neg = ~pos
        rule_spec = float((~rule_hit[neg]).mean())
        t = threshold_at_specificity(score, neg, rule_spec)
        clf_hit = score >= t
        out[name] = {"positives": int(pos.sum()),
                     "auroc": round(roc_auc_score(pos, score), 3) if 0 < pos.sum() < len(pos) else None,
                     "rule_specificity": round(rule_spec, 3),
                     "classifier_specificity": round(float((~clf_hit[neg]).mean()), 3),
                     "rule_recall": round(float(rule_hit[pos].mean()), 3),
                     "classifier_recall": round(float(clf_hit[pos].mean()), 3),
                     "recall_diff_ci95_by_participant": boot_diff(df, clf_hit, rule_hit, pos),
                     "recall_by_person": {
                         p: {"n": int(m.sum()), "rule": round(float(rule_hit[m].mean()), 3),
                             "classifier": round(float(clf_hit[m].mean()), 3)}
                         for p in sorted(df.loc[pos, "participant"].unique())
                         for m in [pos & (df["participant"] == p).to_numpy()]},
                     "_threshold": t}
    return out


def levels_at(prob: np.ndarray, t_urgent: float, t_monitor: float) -> np.ndarray:
    lv = np.where(prob[:, 1] + prob[:, 2] >= t_monitor, 1, 0)
    return np.where(prob[:, 2] >= t_urgent, 2, lv)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    df = pd.read_csv(POOL) if args.reuse and POOL.exists() else build_pool()
    cols = [c for c in df.columns if c not in ("participant", "split", "as_of", "label", "rule")]
    assert not any(c.startswith("actual") for c in cols)
    results = {"features": cols,
               "criterion": __doc__.split("Pre-registered criterion")[1].split("\n\n")[0].strip()}
    eval_labels = json.loads((OUT / "labels.json").read_text())
    packets = json.loads((OUT / "packets.json").read_text())
    key = {(packets[c]["patient"], packets[c]["as_of"]): c for c in packets}
    for mname, make in models().items():
        for scope, frame in (("all_12", df), ("without_035", df[df.participant != "cgmacros-035"])):
            frame = frame.reset_index(drop=True)
            prob = lopo(frame, cols, make)
            test = (frame["split"] == "test").to_numpy()
            res = {"headline_test_7_people": evaluate(frame[test].reset_index(drop=True), prob[test]),
                   "pool_all_12_people": evaluate(frame, prob)}
            if scope == "all_12":
                # The 98 LLM-eval cases: out-of-fold levels at the thresholds matched on the test pool.
                h = res["headline_test_7_people"]
                lv = levels_at(prob, h["urgent"]["_threshold"], h["at_least_monitor"]["_threshold"])
                rows = [(key[(p, a)], lv[i]) for i, (p, a) in enumerate(zip(frame.participant, frame.as_of))
                        if (p, a) in key]
                y = np.array([O[eval_labels[c]["label"]] for c, _ in rows])
                pr = np.array([v for _, v in rows])
                res["on_the_98_eval_cases"] = {
                    "matched": len(rows), "accuracy": round(float((pr == y).mean()), 3),
                    "urgent_recall": round(float((pr[y == 2] == 2).mean()), 3),
                    "routine_recall": round(float((pr[y == 0] == 0).mean()), 3),
                    "under_triage": round(float((pr < y).mean()), 3),
                    "over_triage": round(float((pr > y).mean()), 3)}
            for part in ("pool_all_12_people", "headline_test_7_people"):
                for k in ("urgent", "at_least_monitor"):
                    res[part][k].pop("_threshold")
            results[f"{mname}/{scope}"] = res
            u = res["headline_test_7_people"]["urgent"]
            print(f"== {mname}/{scope} test urgent: auroc={u['auroc']} rule={u['rule_recall']} "
                  f"clf={u['classifier_recall']} diff_ci={u['recall_diff_ci95_by_participant']}", flush=True)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=1, default=float) + "\n")
    print(f"-> {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
