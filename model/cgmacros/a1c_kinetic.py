"""Red-cell kinetic model of HbA1c: why a few days of CGM cannot recover a lab A1c (explains docs/TRIAGE.md §22).

    python -m model.cgmacros.a1c_kinetic

Model (Xu, Dunn, Ajjan; J Diabetes Sci Technol 2021, doi:10.1177/1932296819897613; parameters from Xu et al., Diabetes
Obes Metab 2022, doi:10.1111/dom.14763, and eLife 2021, doi:10.7554/eLife.69456). The source equations are not in the
PubMed Central text we could read, so this is a RECONSTRUCTION, checked against the papers' published numbers
(tests/model_cgmacros/test_a1c_kinetic.py):
    G_eff = G * K_M / (K_M + G)                     glucose inside the red cell (GLUT1 saturation), K_M = 464 mg/dL
    dA/dt = k_gly * G_eff * (1 - A) - A / L         A = glycated fraction, L = red-cell lifespan (days)
    steady state: A = x / (1 + x),  x = AGR * G_eff,  AGR = k_gly * L  (glucose in g/mL)
    reference AGR 65.1 mL/g = 0.62 mL/g/day x 105 days (people without diabetes); T1D cohort mean 72.1 +- 7.0.
    check: AGR 65.1 at 154 mg/dL -> 7.0 %; AGR 80 -> 8.5 %; AGR 70 -> 7.5 % (dom.14763 text and Table 2).

This is NOT a new A1c estimator and was not pre-registered: it is a physiological reference run after the §22 result.
It does three things:
  1. step simulation: months at a high glucose, then treatment lowers it; how far is lab A1c from GMI computed on a
     10-day CGM window, by days since the drop;
  2. steady-state check: population-AGR kinetic estimate vs GMI on each dataset (descriptive, post hoc);
  3. Shanghai: the steady glucose each lab A1c implies, vs the CGM mean actually recorded.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from model.cgmacros.dataset import ROOT

K_M = 464.0                       # mg/dL
AGR_REF = 65.1                    # mL/g, people without diabetes
LIFESPAN = 105.0                  # days
OUT = ROOT / "model" / "results" / "a1c_kinetic.json"


def g_eff(g_mgdl):
    g = np.asarray(g_mgdl, float)
    return (g / 1e5) * K_M / (K_M + g)          # mg/dL -> g/mL, then GLUT1 saturation


def steady_a1c(g_mgdl, agr: float = AGR_REF):
    x = agr * g_eff(g_mgdl)
    return 100.0 * x / (1.0 + x)                 # NGSP %


def glucose_for_a1c(a1c_pct, agr: float = AGR_REF):
    """Inverse of steady_a1c: the constant glucose (mg/dL) that would give this A1c."""
    a = np.asarray(a1c_pct, float) / 100.0
    gm = a / (1.0 - a) / agr * 1e5               # required G_eff, on the mg/dL scale: gm = g*K/(K+g)
    return gm * K_M / (K_M - gm)


def simulate(glucose_by_day, a0: float, agr: float = AGR_REF, lifespan: float = LIFESPAN, dt: float = 0.05):
    """Integrate dA/dt day by day (glucose constant within a day). Returns A1c % at the end of each day."""
    k_gly, k_out, a, out = agr / lifespan, 1.0 / lifespan, a0 / 100.0, []
    steps = int(round(1 / dt))
    for g in glucose_by_day:
        ge = float(g_eff(g))
        for _ in range(steps):
            a += dt * (k_gly * ge * (1 - a) - k_out * a)
        out.append(100.0 * a)
    return np.array(out)


def gmi(mean_mgdl):
    return 3.31 + 0.02392 * np.asarray(mean_mgdl, float)


def step_table(before=(180, 220, 260, 300), after: float = 140.0, days=(10, 30, 60, 90, 180)) -> list[dict]:
    rows = []
    for g0 in before:
        path = simulate([after] * max(days), float(steady_a1c(g0)))
        for d in days:
            lab = float(path[d - 1])
            rows.append({"glucose_before": g0, "glucose_after": after, "days_since_drop": d,
                         "lab_a1c": round(lab, 2), "gmi_on_cgm": round(float(gmi(after)), 2),
                         "lab_minus_gmi": round(lab - float(gmi(after)), 2)})
    return rows


def steady_check() -> dict:
    from model.cgmacros import a1c as A
    out = {}
    for name, t in (("cgmacros", A.dev_table()), ("bigideas", A.bigideas_table()), ("shanghai", A.shanghai_table())):
        y, m = t.a1c.to_numpy(float), t["mean"].to_numpy(float)
        k, g = steady_a1c(m), gmi(m)
        implied = glucose_for_a1c(y)
        out[name] = {"n": len(t),
                     "kinetic_ref_agr": {"mae": float(np.abs(k - y).mean()), "bias": float((k - y).mean())},
                     "gmi": {"mae": float(np.abs(g - y).mean()), "bias": float((g - y).mean())},
                     "median_implied_minus_cgm_mean_mgdl": float(np.median(implied - m))}
        if name == "shanghai":
            q = np.quantile(y, [1 / 3, 2 / 3])
            bands = [(y < q[0]), (y >= q[0]) & (y < q[1]), (y >= q[1])]
            out[name]["by_a1c_tertile"] = [
                {"a1c_median": round(float(np.median(y[b])), 1), "n": int(b.sum()),
                 "median_cgm_mean": round(float(np.median(m[b]))),
                 "median_glucose_implied_by_lab_a1c": round(float(np.median(implied[b])))} for b in bands]
    return out


def main() -> int:
    res = {"model": {"K_M": K_M, "AGR_ref": AGR_REF, "lifespan_days": LIFESPAN,
                     "anchor_154mgdl": round(float(steady_a1c(154)), 2)},
           "step_simulation": step_table(), "steady_state_check": steady_check()}
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
