"""Deterministic triage rule. It is both the baseline the panel is evaluated against and a
safety floor: the panel may raise the level, never lower it below what this rule says.

It reads two forecasts and takes the more severe result:
  no_new_meal   meal model assuming nothing is eaten in the next hour
  history_only  glucose-only model with no meal assumption (so an unlogged meal is not
                simply assumed away)
Horizons match the label and the panel prompt: anything within the next 60 minutes."""
from __future__ import annotations

from agents.triage.schemas import Level, higher


def _one(current: float, f: dict) -> Level:
    if current < 70 or f.get("minutes_until_p50_below_70") is not None:
        return "urgent"
    if current > 250 or f.get("highest_p50_next_60min", max(f["t+30"]["p50"], f["t+60"]["p50"])) > 250:
        return "urgent"
    if (current > 180 or f["lowest_p10_next_60min"] < 70 or f.get("minutes_until_p50_above_180") is not None
            or f["highest_p90_next_60min"] > 250):
        return "monitor"
    return "routine"


def rule_level(glucose_summary: dict, no_new_meal: dict, history_only: dict | None = None) -> Level:
    current = glucose_summary["current_mg_dl"]
    level = _one(current, no_new_meal)
    return higher(level, _one(current, history_only)) if history_only else level
