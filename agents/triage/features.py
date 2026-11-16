"""Numeric features of a triage packet: exactly the fields the agents see, nothing from the future.

Pure Python (no numpy/torch) so the same function serves training (model/cgmacros) and the agents (.venv-pa).
Missing values are float('nan').
"""
from __future__ import annotations

import math

SCEN = {"no_new_meal": "nnm", "meal_30g_now": "m30", "meal_60g_now": "m60", "history_only": "hist"}
_SUMMARY_KEYS = ("lowest_p10_next_60min", "highest_p90_next_60min", "lowest_p50_next_60min",
                 "highest_p50_next_60min", "minutes_until_p50_below_70", "minutes_until_p50_above_180")


def _num(v) -> float:
    return math.nan if v is None else float(v)


def features(packet: dict) -> dict[str, float]:
    gs, ls = packet["glucose_summary"], packet["lifestyle_summary"]
    meal = ls["last_meal_in_2h"]
    f = {"gs_current": _num(gs["current_mg_dl"]), "gs_change15": _num(gs["change_last_15min_mg_dl"]),
         "gs_min2h": _num(gs["min_2h_mg_dl"]), "gs_max2h": _num(gs["max_2h_mg_dl"]),
         "ls_meal_minutes_ago": _num(meal and meal["minutes_ago"]), "ls_meal_carbs": _num(meal and meal["carbs_g"]),
         "ls_carbs_2h": _num(ls["carbs_logged_2h_g"]), "ls_hr": _num(ls["mean_heart_rate_30min_bpm"]),
         "ls_hr_frac": _num(ls["heart_rate_observed_fraction_30min"]), "ls_mets": _num(ls["mean_mets_30min"])}
    for name, pre in SCEN.items():
        s = packet["scenarios"][name]
        for h in ("t+30", "t+60"):
            for q in ("p10", "p50", "p90"):
                f[f"{pre}_{h[2:]}_{q}"] = _num(s[h][q])
        for k in _SUMMARY_KEYS:
            f[f"{pre}_{k}"] = _num(s.get(k))
        if name.startswith("meal"):
            f[f"{pre}_inconsistent"] = float("consistency_warning" in s)
    return f
