"""Score the deterministic keyword matcher on the fine-tuning TEST split: the number a tuned model must beat."""
from __future__ import annotations

import json

from priorauth.finetune.dataset import build
from priorauth.synth.policies import BY_ID


def keyword_answer(ex: dict) -> bool | None:
    crit = next(c for c in BY_ID[ex["meta"]["policy"]].criteria if c.id == ex["meta"]["criterion"])
    user = ex["messages"][1]["content"]
    notes = [l for l in user.splitlines() if l.startswith("[note_id=") and "quarantined=false" in l]
    return True if any(k in l.lower() for l in notes for k in crit.params["keywords"]) else None


def score(rows: list[dict], predict) -> dict:
    n = len(rows); ok = 0; tp = fp = fn = 0; by = {"paraphrased": [0, 0], "verbatim": [0, 0], "injected": [0, 0]}
    for r in rows:
        gold = json.loads(r["messages"][2]["content"])["met"]; pred = predict(r)
        ok += gold == pred
        tp += gold is True and pred is True; fp += gold is None and pred is True; fn += gold is True and pred is None
        key = "injected" if r["meta"]["injected"] else "paraphrased" if r["meta"]["paraphrased"] else "verbatim"
        by[key][0] += gold == pred; by[key][1] += 1
    return {"n": n, "accuracy": round(ok / n, 4), "recall_on_met": round(tp / (tp + fn), 4) if tp + fn else None,
            "false_positive_met": fp, "by_slice": {k: {"acc": round(a / t, 4) if t else None, "n": t} for k, (a, t) in by.items()}}


if __name__ == "__main__":  # pragma: no cover
    d = build()
    print(json.dumps({f"keyword_baseline_on_{k}": score(d[k], keyword_answer) for k in ("test", "test_large")}, indent=2))
