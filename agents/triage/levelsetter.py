"""Level-setter for the triage report: the rule keeps 'urgent'; a logistic model decides routine vs monitor below it.

The model is plain numbers in levelsetter_model.json, exported by model/cgmacros/levelsetter.py (which also holds the
pre-registered evaluation). Pure Python: the agents' environment needs no sklearn. Opt-in only: the panel uses it
when called with level_source="levelsetter"; the default stays max(rule, LLM).
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

from agents.triage.features import features
from agents.triage.schemas import Level

MODEL = Path(__file__).with_name("levelsetter_model.json")


@lru_cache(maxsize=1)
def load(path: str = str(MODEL)) -> dict:
    return json.loads(Path(path).read_text())


def p_at_least_monitor_from_vector(x: list[float], m: dict) -> float:
    z = [m["impute_median"][i] if math.isnan(v) else v for i, v in enumerate(x)]
    z += [1.0 if math.isnan(x[i]) else 0.0 for i in m["missing_indicator_features"]]
    z = [(v - mu) / sd for v, mu, sd in zip(z, m["scale_mean"], m["scale_std"])]
    logits = [b + sum(w * v for w, v in zip(row, z)) for row, b in zip(m["coef"], m["intercept"])]
    top = max(logits)
    e = [math.exp(v - top) for v in logits]
    return (e[1] + e[2]) / sum(e)


def p_at_least_monitor(packet: dict, m: dict | None = None) -> float:
    m = m or load()
    f = features(packet)
    return p_at_least_monitor_from_vector([f[k] for k in m["features"]], m)


def set_level(packet: dict, rule: Level, m: dict | None = None) -> tuple[Level, float]:
    """Final level and the model probability behind it. A rule 'urgent' is never lowered."""
    m = m or load()
    p = p_at_least_monitor(packet, m)
    if rule == "urgent":
        return "urgent", p
    return ("monitor" if p >= m["threshold_at_least_monitor"] else "routine"), p
