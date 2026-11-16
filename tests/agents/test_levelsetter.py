"""Level-setter: pure-Python scorer parity with the exported sklearn model, the safety rule (a rule 'urgent' is never
lowered), and the opt-in panel mode (the LLM cannot raise the level; disagreement is shown; failures fall back)."""
from __future__ import annotations

import math
import os
import secrets

import pytest

os.environ.setdefault("CLINICAL_AGENT_JWT_SECRET", secrets.token_hex(32))

from agents.triage.levelsetter import load, p_at_least_monitor_from_vector, set_level  # noqa: E402
from agents.triage.panel import run_panel, run_single  # noqa: E402
from agents.triage.provider import CaseProvider  # noqa: E402
from serving.healthcare.audit import AuditLogger  # noqa: E402
from tests.agents.test_triage import packet, panel_factory, scen  # noqa: E402


def full_packet(current=110.0, p30=112.0, p60=118.0):
    """All four scenarios, as model/cgmacros/cases.py builds them (the level-setter needs every feature)."""
    p = packet(no_new_meal=scen("s1", p30, p60), current=current)
    p["lifestyle_summary"].update({"mean_heart_rate_30min_bpm": None, "heart_rate_observed_fraction_30min": 0.0,
                                   "mean_mets_30min": 1.0})
    p["scenarios"]["meal_30g_now"] = scen("s3", p30 + 5, p60 + 20, name="meal_30g_now")
    p["scenarios"]["history_only"] = scen("s4", p30, p60, name="history_only")
    for s in p["scenarios"].values():
        s.setdefault("lowest_p50_next_60min", min(s["t+30"]["p50"], s["t+60"]["p50"]))
        s.setdefault("highest_p50_next_60min", max(s["t+30"]["p50"], s["t+60"]["p50"]))
    return p


def test_pure_python_matches_the_exported_sklearn_probabilities():
    m = load()
    assert m["reference"], "export must carry reference probabilities"
    for ref in m["reference"]:
        x = [math.nan if v is None else v for v in ref["features"]]
        assert abs(p_at_least_monitor_from_vector(x, m) - ref["p_at_least_monitor"]) < 1e-9


def test_rule_urgent_is_never_lowered():
    level, _ = set_level(full_packet(), "urgent")
    assert level == "urgent"


@pytest.mark.parametrize("rule", ["routine", "monitor"])
def test_below_urgent_the_model_decides(rule):
    level, p = set_level(full_packet(), rule)        # 2-h minimum is 98: the guard does not apply
    assert level == ("monitor" if p >= load()["threshold_at_least_monitor"] else "routine")


def test_guard_blocks_downgrade_after_a_recent_near_low():
    pk = full_packet()
    pk["glucose_summary"]["min_2h_mg_dl"] = 79.0
    assert set_level(pk, "monitor")[0] == "monitor"
    assert set_level(pk, "routine")[0] in ("routine", "monitor")   # the guard only stops downgrades


def test_opt_in_mode_llm_cannot_raise_and_disagreement_is_shown(tmp_path):
    pk = full_packet()
    expected, _ = set_level(pk, "routine")
    report, _ = run_panel(CaseProvider(pk), panel_factory("urgent"),
                          AuditLogger(path=str(tmp_path / "a.jsonl"), pg_dsn=None), level_source="levelsetter")
    assert report.level_source == "levelsetter" and report.level == expected != "urgent"
    assert report.panel_level == "urgent"
    assert any("language model proposed urgent" in d for d in report.dissent)


def test_default_mode_is_unchanged(tmp_path):
    report, _ = run_panel(CaseProvider(full_packet()), panel_factory("urgent"),
                          AuditLogger(path=str(tmp_path / "a.jsonl"), pg_dsn=None))
    assert report.level_source == "rule_floor" and report.level == "urgent"


def test_levelsetter_failure_falls_back_to_rule_floor(tmp_path):
    report, _ = run_single(CaseProvider(packet()), panel_factory(),          # packet lacks two scenarios
                           AuditLogger(path=str(tmp_path / "a.jsonl"), pg_dsn=None), level_source="levelsetter")
    assert report.level_source == "rule_floor" and report.level == "monitor"
    assert any(e.startswith("levelsetter:") for e in report.errors)


def test_unknown_level_source_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        run_single(CaseProvider(full_packet()), panel_factory(),
                   AuditLogger(path=str(tmp_path / "a.jsonl"), pg_dsn=None), level_source="vibes")
