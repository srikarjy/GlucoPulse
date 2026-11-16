"""Confirmation on people nothing was designed on: does the guarded level-setter hold up?

    CGMACROS_SPLIT=confirm python -m model.cgmacros.train meals_activity glucose   # forecaster without them
    CGMACROS_SPLIT=confirm python -m model.cgmacros.confirm                         # pool + evaluation

The confirmation split (model/cgmacros/splits.py) holds out 9 people drawn at random (seed 2026) from the
forecaster's original TRAINING people, so neither the forecaster nor anything designed on the primary held-out 12
(the ceiling, the level-setter, the guard) has seen them. 5 of the primary held-out people are used only for the
forecaster's early stopping; the other 31 train it.

PRE-REGISTERED (written 2026-11-16, before any forecast or label of the 9 new people was computed):
  Primary policy  "guarded": urgent if the rule says urgent; else if the rule says monitor and the lowest reading of
                  the past 2 h is below 80 mg/dL, monitor (no downgrade); else monitor if the logistic model's
                  P(at least monitor) >= t_m, otherwise routine. The 80 mg/dL guard was chosen on the primary 12
                  people (it is post hoc there), which is why it is tested here.
  Model/protocol  multinomial logistic regression, nested leave-one-participant-out over the 14 people the new
                  forecaster never trained on (9 new + 5 early-stopping people): model and t_m come only from
                  the other 13.
  Criteria        on the 9 NEW people, versus the rule, ALL of:
                    1. under-triage lower, with the 95% participant-bootstrap interval of the reduction excluding 0;
                    2. over-triage at most 5 points higher;
                    3. no more urgent hours called routine than the rule.
  Secondary       the unguarded policy (replication of model/cgmacros/levelsetter.py); everything on all 14 people.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from model.cgmacros.ceiling import POOL, build_pool, models
from model.cgmacros.dataset import ROOT
from model.cgmacros.levelsetter import MAX_OVER_TRIAGE_INCREASE, O, nested, report
from model.cgmacros.splits import SCHEME

RESULTS = ROOT / "model" / "results" / f"cgmacros_{SCHEME}.json"
GUARD_MIN_2H = 80.0


def main() -> int:
    if SCHEME != "confirm":
        raise SystemExit("run with CGMACROS_SPLIT=confirm (and a forecaster trained on that split)")
    df = pd.read_csv(POOL) if POOL.exists() else build_pool()
    cols = [c for c in df.columns if c not in ("participant", "split", "as_of", "label", "rule")]
    rule = df["rule"].map(O).to_numpy()
    p_m, _, t_m, _ = nested(df, cols, models()["logistic"])
    below = np.where(p_m >= t_m, 1, 0)
    guarded = np.where((rule == 1) & (df["gs_min2h"].to_numpy() < GUARD_MIN_2H), 1, below)
    preds = {"guarded": np.where(rule == 2, 2, guarded), "unguarded": np.where(rule == 2, 2, below)}
    new = (df["split"] == "test").to_numpy()
    out = {"preregistration": __doc__.split("PRE-REGISTERED")[1].strip(),
           "new_people": sorted(df.loc[new, "participant"].unique().tolist()),
           "labels_new": df.loc[new, "label"].value_counts().to_dict()}
    for name, pred in preds.items():
        out[name] = {"new_9_people": report(df[new].reset_index(drop=True), pred[new], rule[new]),
                     "all_14_people": report(df, pred, rule)}
        t = out[name]["new_9_people"]
        print(f"{name:10s} under {t['rule_under_triage']}->{t['under_triage']} CI{t['under_reduction_ci95_by_person']} "
              f"over {t['rule_over_triage']}->{t['over_triage']}  urgent->routine "
              f"{t['rule_severe_under_triage_urgent_called_routine']}->{t['severe_under_triage_urgent_called_routine']}",
              flush=True)
    g = out["guarded"]["new_9_people"]
    out["criteria"] = {
        "1_under_triage_reduction_excludes_0": bool(g["under_reduction_ci95_by_person"][0] > 0),
        "2_over_triage_increase_at_most_5_points": bool(g["over_triage"] - g["rule_over_triage"]
                                                        <= MAX_OVER_TRIAGE_INCREASE),
        "3_no_more_urgent_called_routine": bool(g["severe_under_triage_urgent_called_routine"]
                                                <= g["rule_severe_under_triage_urgent_called_routine"])}
    out["confirmed"] = all(out["criteria"].values())
    RESULTS.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print("criteria:", out["criteria"], "confirmed:", out["confirmed"], f"-> {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
