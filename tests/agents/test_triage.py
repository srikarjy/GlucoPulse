"""Triage panel: rule floor, numeric grounding, panel/single orchestration, tool scoping.
Scripted models only (no API key, deterministic)."""
from __future__ import annotations

import json
from collections import Counter
import os
import secrets

import pytest

os.environ.setdefault("CLINICAL_AGENT_JWT_SECRET", secrets.token_hex(32))

from agents.mcp_server.auth import ForbiddenError, Principal  # noqa: E402
from agents.mcp_server.tools import LocalGateway, ToolBox  # noqa: E402
from agents.triage.grounding import ground_claims  # noqa: E402
from agents.triage.panel import ROLES, run_panel, run_single  # noqa: E402
from agents.triage.provider import CaseProvider  # noqa: E402
from agents.triage.rules import rule_level  # noqa: E402
from agents.triage.schemas import Claim  # noqa: E402
from agents.triage.testing import ScriptedModel  # noqa: E402
from serving.healthcare.audit import AuditLogger  # noqa: E402

P = "cgmacros-007"


def scen(rid, p30, p60, p10=None, p90=None, below=None, above=None, name="no_new_meal"):
    return {"resourceType": "ScenarioForecast", "id": rid, "scenario": name, "description": name,
            "t+30": {"p10": p30 - 10, "p50": p30, "p90": p30 + 10},
            "t+60": {"p10": p10 if p10 is not None else p60 - 15, "p50": p60, "p90": p90 if p90 is not None else p60 + 15},
            "lowest_p10_next_60min": p10 if p10 is not None else min(p30, p60) - 15,
            "highest_p90_next_60min": p90 if p90 is not None else max(p30, p60) + 15,
            "minutes_until_p50_below_70": below, "minutes_until_p50_above_180": above}


def packet(no_new_meal=None, current=110.0):
    return {"patient": P, "as_of": "2020-05-03T12:00:00+00:00",
            "glucose_summary": {"resourceType": "GlucoseSummary", "id": "g1", "current_mg_dl": current,
                                "change_last_15min_mg_dl": 4.0, "min_2h_mg_dl": 98.0, "max_2h_mg_dl": 131.0},
            "lifestyle_summary": {"resourceType": "LifestyleSummary", "id": "l1", "carbs_logged_2h_g": 45.0,
                                  "last_meal_in_2h": {"minutes_ago": 40, "carbs_g": 45.0}},
            "data_quality": {"resourceType": "DataQuality", "id": "q1", "expected_readings_2h": 24,
                             "received_readings_2h": 24},
            "scenarios": {"no_new_meal": no_new_meal or scen("s1", 150.0, 190.0, above=45),
                          "meal_60g_now": scen("s2", 175.0, 230.0, above=25, name="meal_60g_now")}}


def reply(level, claims, role="x"):
    return json.dumps({"role": role, "level": level, "claims": claims, "concerns": [], "confidence": "med"})


# ---------------------------------------------------------------- rule floor
@pytest.mark.parametrize("current, plan, expected", [
    (65.0, scen("a", 80, 90), "urgent"),                       # low right now
    (100.0, scen("a", 75, 60, below=30), "urgent"),            # median below 70 within 30 min
    (200.0, scen("a", 255, 270), "urgent"),                    # median above 250
    (110.0, scen("a", 150, 190, above=45), "monitor"),         # median above 180 within the hour
    (110.0, scen("a", 95, 90, p10=66), "monitor"),             # interval reaches below 70
    (110.0, scen("a", 112, 118), "routine"),
])
def test_rule_level(current, plan, expected):
    assert rule_level({"current_mg_dl": current}, plan) == expected


# ---------------------------------------------------------------- grounding
def test_numbers_must_come_from_the_cited_resource():
    retrieved = {"ScenarioForecast/s1": scen("s1", 150.0, 190.0, above=45)}
    kept, dropped = ground_claims([
        Claim(text="Median reaches 190 mg/dL at 60 min.", evidence=["ScenarioForecast/s1"]),
        Claim(text="Median reaches 120 mg/dL at 60 min.", evidence=["ScenarioForecast/s1"]),   # number not in it
        Claim(text="It will rise.", evidence=["ScenarioForecast/zzz"]),                          # never retrieved
        Claim(text="It will rise.", evidence=[]),
    ], retrieved)
    assert [c.text for c in kept] == ["Median reaches 190 mg/dL at 60 min."]
    assert [d["reason"] for d in dropped] == ["number_not_in_cited_resource", "cites_unretrieved_resource",
                                              "no_citation"]


# ---------------------------------------------------------------- tool scoping
def test_triage_tools_are_scoped_to_the_case_patient_and_role(tmp_path):
    box = ToolBox(fhir=None, audit=AuditLogger(path=str(tmp_path / "a.jsonl"), pg_dsn=None),
                  triage=CaseProvider(packet()))
    agent = LocalGateway(box, Principal(subject="a", role="clinical-agent",
                                        scopes=frozenset({"triage:read", "audit:write"})))
    assert agent.invoke("glucose_summary", patient=P)["current_mg_dl"] == 110.0
    with pytest.raises(Exception, match="scoped to patient"):
        agent.invoke("glucose_summary", patient="cgmacros-001")
    with pytest.raises(Exception, match="unknown scenarios"):
        agent.invoke("forecast_scenarios", patient=P, scenarios=["eat_cake"])
    reviewer = LocalGateway(box, Principal(subject="r", role="reviewer", scopes=frozenset({"review:decide"})))
    with pytest.raises(ForbiddenError):
        reviewer.invoke("glucose_summary", patient=P)


# ---------------------------------------------------------------- orchestration
def panel_factory(moderator_level="routine"):
    good = [("tool", "forecast_scenarios", {"patient": P, "scenarios": ["no_new_meal"]}),
            ("text", reply("monitor", [{"text": "As planned, the median reaches 190 mg/dL by 60 min.",
                                        "evidence": ["ScenarioForecast/s1"]}]))]
    fabricated = [("tool", "lifestyle_summary", {"patient": P}),
                  ("text", reply("routine", [{"text": "Glucose will settle near 120 mg/dL.",
                                              "evidence": ["LifestyleSummary/l1"]}]))]
    quality = [("tool", "data_quality", {"patient": P}),
               ("text", reply("routine", [{"text": "24 of 24 expected readings arrived.",
                                           "evidence": ["DataQuality/q1"]}]))]
    moderator = [("text", json.dumps({"level": moderator_level, "claims": [
        {"text": "As planned, the median reaches 190 mg/dL by 60 min.", "evidence": ["ScenarioForecast/s1"]},
        {"text": "Invented 150 reading.", "evidence": ["ScenarioForecast/never"]}],
        "dissent": ["lifestyle saw no risk"]}))]
    scripts = {"forecaster": good, "lifestyle": fabricated, "data_quality": quality, "skeptic": good,
               "moderator": moderator, "single": good}
    return lambda role, rnd: ScriptedModel(scripts[role])


def test_panel_end_to_end_with_rule_floor_and_grounding(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    report, trace = run_panel(CaseProvider(packet()), panel_factory("routine"),
                              AuditLogger(path=str(audit_path), pg_dsn=None))
    assert report.rule_level == "monitor"
    assert report.panel_level == "monitor"     # moderator said routine; clamped to the most cautious panelist
    assert report.level == "monitor"           # and the panel can never lower the rule floor
    assert [c.text for c in report.claims] == ["As planned, the median reaches 190 mg/dL by 60 min."]
    reasons = {(d["by"].split("/")[0], d["reason"]) for d in report.dropped_claims}
    assert ("lifestyle", "number_not_in_cited_resource") in reasons        # 120 is not a field of LifestyleSummary/l1
    assert ("moderator", "cites_unretrieved_resource") in reasons
    assert report.requires_human_review and [r.scenario for r in report.scenarios] == ["no_new_meal", "meal_60g_now"]
    assert trace.calls == len(ROLES) * 2 + 1 and set(trace.rounds) == {1, 2}
    audited = [json.loads(line) for line in audit_path.read_text().splitlines()]
    assert audited[-1]["action"] == "draft_triage_panel" and audited[-1]["detail"]["level"] == "monitor"


def test_panel_can_raise_above_the_rule(tmp_path):
    report, _ = run_panel(CaseProvider(packet()), panel_factory("urgent"),
                          AuditLogger(path=str(tmp_path / "a.jsonl"), pg_dsn=None))
    assert report.level == "urgent"


def test_single_agent_baseline(tmp_path):
    report, trace = run_single(CaseProvider(packet()), panel_factory(),
                               AuditLogger(path=str(tmp_path / "a.jsonl"), pg_dsn=None))
    assert report.level == "monitor" and trace.calls == 1 and len(report.claims) == 1


def test_unparseable_model_output_is_contained(tmp_path):
    report, trace = run_single(CaseProvider(packet(no_new_meal=scen("s1", 112, 118))),
                               lambda role, rnd: ScriptedModel([("text", "I think it's fine.")]),
                               AuditLogger(path=str(tmp_path / "a.jsonl"), pg_dsn=None))
    assert report.level == "monitor" and trace.parse_errors                 # fails safe, not silent


def test_durations_and_percentages_are_not_treated_as_measurements():
    retrieved = {"ScenarioForecast/s1": scen("s1", 150.0, 190.0, above=45),
                 "DataQuality/q1": {"resourceType": "DataQuality", "id": "q1", "expected_readings_2h": 24,
                                    "received_readings_2h": 24}}
    kept, dropped = ground_claims([
        Claim(text="The 80% interval at 60 min spans 175.0 to 205.0 mg/dL.", evidence=["ScenarioForecast/s1"]),
        Claim(text="24 of 24 readings arrived in the last 2 hours.", evidence=["DataQuality/q1"]),
        Claim(text="Over the next 60 minutes glucose reaches 100 mg/dL.", evidence=["ScenarioForecast/s1"]),
    ], retrieved)
    assert len(kept) == 2 and dropped[0]["detail"] == ["number_not_in_cited_resource:100"]   # glucose still checked


# ---------------------------------------------------------------- healthcare-review regressions
REAL_SUMMARY = {"resourceType": "GlucoseSummary", "id": "gs2fbbed70357a9d", "as_of": "2022-04-26T20:35:00+00:00",
                "window": "previous two hours of Dexcom G6 readings", "current_mg_dl": 191.0,
                "change_last_15min_mg_dl": 23.0, "min_2h_mg_dl": 160.0, "max_2h_mg_dl": 191.0}


@pytest.mark.parametrize("text", [
    "Glucose is 35 mg/dL.",                    # 35 only appears inside the as_of timestamp
    "Glucose may fall to 2022 mg/dL.",         # a year
    "Readings show 2 mg/dL swings.",           # '2' only appears inside the id
])
def test_numbers_only_come_from_numeric_fields_not_ids_or_timestamps(text):
    kept, dropped = ground_claims([Claim(text=text, evidence=["GlucoseSummary/gs2fbbed70357a9d"])],
                                  {"GlucoseSummary/gs2fbbed70357a9d": REAL_SUMMARY})
    assert not kept and dropped[0]["reason"] == "number_not_in_cited_resource"


def test_thresholds_and_real_values_are_allowed():
    kept, _ = ground_claims([Claim(text="Current glucose is 191.0 mg/dL, above the 180 threshold, rising 23 over "
                                        "15 minutes.", evidence=["GlucoseSummary/gs2fbbed70357a9d"])],
                            {"GlucoseSummary/gs2fbbed70357a9d": REAL_SUMMARY})
    assert len(kept) == 1


@pytest.mark.parametrize("text, reason", [
    ("Give 15 g of fast-acting carbs now.", "treatment_or_diagnosis_language"),
    ("This looks like diagnosed diabetes; start metformin.", "treatment_or_diagnosis_language"),
    ("There is a 90% chance of a low.", "percentage_not_supported"),
    ("Glucose drops below 70 within 10 min.", "duration_not_supported"),
])
def test_advice_and_unsupported_percentages_or_durations_are_dropped(text, reason):
    _, dropped = ground_claims([Claim(text=text, evidence=["GlucoseSummary/gs2fbbed70357a9d"])],
                               {"GlucoseSummary/gs2fbbed70357a9d": REAL_SUMMARY})
    assert dropped and dropped[0]["reason"] == reason


@pytest.mark.parametrize("current, plan, history, expected", [
    (100.0, scen("a", 90, 72, below=45), None, "urgent"),                   # hypo at 45 min is urgent too
    (180.0, {**scen("a", 240, 245), "highest_p50_next_60min": 262.0}, None, "urgent"),  # peak between checkpoints
    (260.0, scen("a", 240, 245), None, "urgent"),                           # already above 250
    (190.0, scen("a", 170, 160), None, "monitor"),                          # already above 180
    (120.0, scen("a", 125, 130), scen("h", 160, 200, above=40), "monitor"), # history-only model sees the rise
])
def test_rule_uses_the_full_hour_current_value_and_both_forecasts(current, plan, history, expected):
    assert rule_level({"current_mg_dl": current}, plan, history) == expected


def test_moderator_cannot_go_below_the_most_cautious_panelist(tmp_path):
    report, _ = run_panel(CaseProvider(packet()), panel_factory("routine"),
                          AuditLogger(path=str(tmp_path / "a.jsonl"), pg_dsn=None))
    assert report.panel_level == "monitor" and any("raised to monitor" in d for d in report.dissent)


def test_failed_panelist_keeps_previous_view_and_failures_are_reported(tmp_path):
    base = panel_factory("routine")
    calls = Counter()

    class Boom(ScriptedModel):
        async def stream(self, *a, **k):
            raise RuntimeError("rate limited")
            yield  # pragma: no cover

    def factory(role, rnd):
        calls[(role, rnd)] += 1
        return Boom([]) if (role, rnd) == ("forecaster", 2) else base(role, rnd)

    report, trace = run_panel(CaseProvider(packet()), factory, AuditLogger(path=str(tmp_path / "a.jsonl"), pg_dsn=None))
    assert report.panel_incomplete and any("model call failed" in e for e in report.errors)
    assert report.panel_level == "monitor"               # forecaster's round-1 'monitor' was carried forward


@pytest.mark.parametrize("text", ["The meal scenario gives 160.0 mg/dL at 60 min.",
                                  "The interval is wide; treat the median as a guide, not a fact."])
def test_descriptive_wording_is_not_mistaken_for_advice(text):
    kept, dropped = ground_claims([Claim(text=text, evidence=["ScenarioForecast/s1"])],
                                  {"ScenarioForecast/s1": scen("s1", 150.0, 160.0)})
    assert kept, dropped
