"""Evaluate rule vs single agent vs panel on held-out CGMacros triage cases.

    python -m agents.triage.eval --model openai --limit 60     # needs OPENAI_API_KEY
    python -m agents.triage.eval --model scripted              # plumbing check, no API calls

Cases come from model/cgmacros/cases.py (packets.json, history only) and are scored
against labels.json (what actually happened in the next 60 min). Reported per method:
accuracy, recall per level, under-triage (said lower than reality: the dangerous
error), over-triage (said higher: alarm fatigue), and for the LLM methods the call
count, latency, parse failures and how many claims grounding removed.

"scripted" plays back each case's rule level, so its numbers only show the harness
works; they say nothing about LLM quality. Results are written next to the cases.
"""
from __future__ import annotations

import argparse
import json
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from agents.triage.panel import run_panel, run_rule, run_single
from agents.triage.provider import CaseProvider
from agents.triage.schemas import ORDER
from agents.triage.testing import ScriptedModel
from serving.healthcare.audit import AuditLogger

CASES = Path(__file__).resolve().parents[2] / "data" / "cgmacros" / "triage"
LEVELS = ("routine", "monitor", "urgent")


def _urgent_recall(ids, labels, pred):
    urgent = [c for c in ids if labels[c]["label"] == "urgent"]
    return sum(pred[c] == "urgent" for c in urgent) / len(urgent) if urgent else None


def score(pred: dict[str, str], labels: dict[str, dict]) -> dict:
    ids = [c for c in pred if c in labels and c != "_meta"]
    y = [labels[c]["label"] for c in ids]
    p = [pred[c] for c in ids]
    n = len(ids)
    out = {"n": n, "accuracy": round(sum(a == b for a, b in zip(y, p)) / n, 3) if n else None,
           "under_triage": round(sum(ORDER[b] < ORDER[a] for a, b in zip(y, p)) / n, 3) if n else None,
           "over_triage": round(sum(ORDER[b] > ORDER[a] for a, b in zip(y, p)) / n, 3) if n else None,
           "recall": {}, "confusion": {}}
    for lv in LEVELS:
        truth = [i for i, a in enumerate(y) if a == lv]
        out["recall"][lv] = round(sum(p[i] == lv for i in truth) / len(truth), 3) if truth else None
        out["confusion"][lv] = dict(Counter(p[i] for i in truth))
    non_urgent = [i for i, a in enumerate(y) if a != "urgent"]
    out["urgent_specificity"] = (round(sum(p[i] != "urgent" for i in non_urgent) / len(non_urgent), 3)
                                 if non_urgent else None)
    # Cases are sampled ~balanced, so accuracy/PPV at that mix are not real-world numbers.
    # Re-weight to the true prevalence of the held-out pool.
    pools = labels.get("_meta", {}).get("pool_sizes")
    sens, spec = out["recall"]["urgent"], out["urgent_specificity"]
    if pools and sens is not None and spec is not None:
        prev = pools["urgent"] / sum(pools.values())
        tp, fp = sens * prev, (1 - spec) * (1 - prev)
        fn, tn = (1 - sens) * prev, spec * (1 - prev)
        out["urgent_at_true_prevalence"] = {"prevalence": round(prev, 4),
                                            "ppv": round(tp / (tp + fp), 3) if tp + fp else None,
                                            "npv": round(tn / (tn + fn), 3) if tn + fn else None}
    by_person = defaultdict(list)
    for c in ids:
        by_person[labels[c].get("participant", "?")].append(c)
    out["urgent_recall_by_participant"] = {
        p: {"urgent_cases": sum(labels[c]["label"] == "urgent" for c in cs),
            "caught": sum(labels[c]["label"] == "urgent" and pred[c] == "urgent" for c in cs)}
        for p, cs in sorted(by_person.items())}
    # 95% interval for urgent recall, resampling PARTICIPANTS (cases from one person are correlated)
    people = sorted(by_person)
    rng, draws = random.Random(0), []
    for _ in range(2000):
        sample = [c for p in (rng.choice(people) for _ in people) for c in by_person[p]]
        r = _urgent_recall(sample, labels, pred)
        if r is not None:
            draws.append(r)
    if draws:
        draws.sort()
        out["urgent_recall_ci95_by_participant"] = [round(draws[int(0.025 * len(draws))], 3),
                                                    round(draws[int(0.975 * len(draws)) - 1], 3)]
    return out


def scripted_factory(provider: CaseProvider):
    """Plays back the rule's level with one grounded claim: exercises the full loop."""
    level = run_rule(provider)
    plan = provider.packet["scenarios"]["no_new_meal"]
    claim = {"text": f"As planned, the median at 60 min is {plan['t+60']['p50']} mg/dL.",
             "evidence": [f"ScenarioForecast/{plan['id']}"]}
    tool = ("tool", "forecast_scenarios", {"patient": provider.patient, "scenarios": ["no_new_meal"]})
    answer = ("text", json.dumps({"role": "x", "level": level, "claims": [claim], "confidence": "med"}))
    moderator = ("text", json.dumps({"level": level, "claims": [claim], "dissent": []}))
    return lambda role, rnd: ScriptedModel([moderator] if role == "moderator" else [tool, answer])


def openai_factory(model_id: str):
    from agents.reasoning_agent import openai_model
    import os
    os.environ.setdefault("CLINICAL_AGENT_MODEL_ID", model_id)
    return lambda role, rnd: openai_model()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["scripted", "openai"], default="scripted")
    ap.add_argument("--model-id", default="gpt-4o-mini")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cases", type=Path, default=CASES)
    args = ap.parse_args()
    packets = json.loads((args.cases / "packets.json").read_text())
    labels = json.loads((args.cases / "labels.json").read_text())
    ids = sorted(packets)[: args.limit or None]       # ids are shuffled at build time: not label-ordered
    preds = {"rule": {}, "single": {}, "panel": {}}
    llm = {m: Counter() for m in ("single", "panel")}
    with tempfile.TemporaryDirectory() as tmp:
        audit = AuditLogger(path=str(Path(tmp) / "audit.jsonl"), pg_dsn=None)
        for k, cid in enumerate(ids, 1):
            provider = CaseProvider(packets[cid])
            factory = scripted_factory(provider) if args.model == "scripted" else openai_factory(args.model_id)
            preds["rule"][cid] = run_rule(provider)
            for mode, runner in (("single", run_single), ("panel", run_panel)):
                report, trace = runner(provider, factory, audit)
                preds[mode][cid] = report.level
                llm[mode].update(calls=trace.calls, seconds=round(trace.seconds, 2),
                                 parse_errors=len(trace.parse_errors), claims_kept=len(report.claims),
                                 claims_dropped=len(report.dropped_claims),
                                 raised_above_rule=int(ORDER[report.level] > ORDER[report.rule_level]))
            print(f"[{k}/{len(ids)}] {cid} label={labels[cid]['label']} rule={preds['rule'][cid]} "
                  f"single={preds['single'][cid]} panel={preds['panel'][cid]}", flush=True)
    results = {"model": args.model if args.model == "scripted" else args.model_id, "cases": len(ids),
               "label_counts": dict(Counter(labels[c]["label"] for c in ids)),
               "pool_sizes": labels.get("_meta", {}).get("pool_sizes"),
               "methods": {m: score(p, labels) for m, p in preds.items()},
               "llm_totals": {m: dict(c) for m, c in llm.items()},
               "note": ("scripted mode replays the rule; only the openai run measures LLM triage quality"
                        if args.model == "scripted" else "live model run")}
    out = args.cases / f"eval_{results['model']}.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results["methods"], indent=1))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
