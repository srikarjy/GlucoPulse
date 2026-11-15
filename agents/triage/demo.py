"""Watch the triage panel discuss one held-out case.

    python -m agents.triage.demo                         # first urgent case, scripted role-play
    python -m agents.triage.demo --case case-012 --model openai   # live (needs OPENAI_API_KEY)

Scripted mode is a deterministic role-play built from the case's own tool outputs
(it is not an LLM): it shows the protocol, the grounding and the report. One panelist
deliberately quotes a number its evidence doesn't contain, to show it being removed.
Only --model openai shows real model reasoning.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from agents.triage.eval import CASES, openai_factory
from agents.triage.panel import run_panel
from agents.triage.provider import CaseProvider
from agents.triage.rules import rule_level
from agents.triage.testing import ScriptedModel
from serving.healthcare.audit import AuditLogger


def roleplay_factory(provider: CaseProvider):
    pk, p = provider.packet, provider.patient
    plan, small, meal = (pk["scenarios"][n] for n in ("no_new_meal", "meal_30g_now", "meal_60g_now"))
    g, life, dq = pk["glucose_summary"], pk["lifestyle_summary"], pk["data_quality"]
    level = rule_level(g, plan, pk["scenarios"].get("history_only"))
    ref = lambda r: f"{r['resourceType']}/{r['id']}"  # noqa: E731

    def say(lv, claims, conf="med", concerns=()):
        return ("text", json.dumps({"role": "x", "level": lv, "claims": claims, "concerns": list(concerns),
                                    "confidence": conf}))

    forecaster = [("tool", "forecast_scenarios", {"patient": p, "scenarios": ["no_new_meal"]}), say(level, [
        {"text": f"As planned, the median forecast is {plan['t+30']['p50']} mg/dL at 30 min and "
                 f"{plan['t+60']['p50']} mg/dL at 60 min.", "evidence": [ref(plan)]},
        {"text": f"The 80% interval at 60 min spans {plan['t+60']['p10']} to {plan['t+60']['p90']} mg/dL.",
         "evidence": [ref(plan)]}])]
    lifestyle = [("tool", "lifestyle_summary", {"patient": p}),
                 ("tool", "forecast_scenarios", {"patient": p, "scenarios": ["meal_30g_now", "meal_60g_now"]}),
                 say(level, [
                     {"text": f"{life['carbs_logged_2h_g']} g of carbs were logged in the last 2 hours.",
                      "evidence": [ref(life)]},
                     {"text": f"A 60 g meal now would put the 60-min median at {meal['t+60']['p50']} mg/dL; "
                              f"a 30 g meal gives {small['t+60']['p50']} mg/dL.",
                      "evidence": [ref(meal), ref(small)]},
                     {"text": "Glucose will definitely return to 100 mg/dL by dinner.", "evidence": [ref(life)]}])]
    quality = [("tool", "data_quality", {"patient": p}), say("routine", [
        {"text": f"{dq['received_readings_2h']} of {dq['expected_readings_2h']} expected CGM readings arrived "
                 f"in the last 2 hours.", "evidence": [ref(dq)]}], conf="high")]
    skeptic = [("tool", "glucose_summary", {"patient": p}),
               ("tool", "forecast_scenarios", {"patient": p, "scenarios": ["no_new_meal"]}),
               say(level, [{"text": f"Current glucose is {g['current_mg_dl']} mg/dL, changing "
                                    f"{g['change_last_15min_mg_dl']} mg/dL over 15 min.", "evidence": [ref(g)]}],
                   conf="low", concerns=["The interval is wide; treat the median as a guide, not a fact."])]
    moderator = [("text", json.dumps({"level": level, "claims": [
        {"text": f"As planned, the median forecast is {plan['t+60']['p50']} mg/dL at 60 min.",
         "evidence": [ref(plan)]}], "dissent": ["Data quality saw no reason for concern; others did."]}))]
    scripts = {"forecaster": forecaster, "lifestyle": lifestyle, "data_quality": quality, "skeptic": skeptic,
               "moderator": moderator}
    return lambda role, rnd: ScriptedModel(scripts[role])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case")
    ap.add_argument("--model", choices=["roleplay", "openai"], default="roleplay")
    ap.add_argument("--model-id", default="gpt-4o-mini")
    args = ap.parse_args()
    packets = json.loads((CASES / "packets.json").read_text())
    labels = json.loads((CASES / "labels.json").read_text())
    cid = args.case or next(c for c in sorted(packets) if labels[c]["label"] == "urgent")  # noqa: E501
    provider = CaseProvider(packets[cid])
    factory = roleplay_factory(provider) if args.model == "roleplay" else openai_factory(args.model_id)
    with tempfile.TemporaryDirectory() as tmp:
        report, trace = run_panel(provider, factory, AuditLogger(path=str(Path(tmp) / "a.jsonl"), pg_dsn=None))
    bar = "=" * 78
    print(f"{bar}\nTRIAGE PANEL  {cid}  patient {report.patient}  as of {report.as_of}"
          f"\nmode: {'scripted role-play (not an LLM)' if args.model == 'roleplay' else 'live ' + args.model_id}\n{bar}")
    for t in trace.transcript:
        head = "MODERATOR" if t["role"] == "moderator" else f"ROUND {t['round']}  {t['role'].upper()}"
        print(f"\n{head}  ->  {t['level'].upper()}" + (f"  (confidence {t.get('confidence')})" if t.get("confidence") else ""))
        for c in t["claims"]:
            print(f"   ✓ {c['text']}\n       cites {', '.join(c['evidence'])}")
        for d in t["dropped"]:
            print(f"   ✕ REMOVED ({d['reason']}): {d['text']}")
        for x in t.get("concerns", []) + t.get("dissent", []):
            print(f"   ! {x}")
    print(f"\n{bar}\nREPORT  level {report.level.upper()}  (rule {report.rule_level}, panel {report.panel_level}; "
          f"the panel can raise, never lower)\n{bar}")
    print(f"{'scenario':<22}{'+30 min':>9}{'+60 min':>9}{'60-min 80% range':>20}")
    for r in report.scenarios:
        print(f"{r.scenario:<22}{r.t30_p50:>9}{r.t60_p50:>9}{f'{r.t60_p10} - {r.t60_p90}':>20}")
    print(f"\nWhat actually happened (hidden from the panel): {labels[cid]['label'].upper()}, "
          f"range {labels[cid]['actual_min_60']}-{labels[cid]['actual_max_60']} mg/dL")
    print(f"Requires human review: {report.requires_human_review}.  {report.disclaimer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
