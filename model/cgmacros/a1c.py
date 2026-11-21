"""Person-level glycemic state from days of data: does averaging over days estimate HbA1c?

    python -m model.cgmacros.a1c --dev         # CGMacros only (45 people); freezes the choices
    python -m model.cgmacros.a1c --external    # needs the frozen dev file; BIG IDEAs + ShanghaiT2DM

Idea borrowed from Google's wearable hypertension work (Nature 2026; blog 2026-11-05): judge a person's persistent
state from data aggregated over days, not one reading. Their method (PPG/ACC waveform encoder) is NOT replicated: our
development data has no raw waveforms. Only the "aggregate over days, then classify the person" idea is tested.

PRE-REGISTERED (written 2026-11-18, before any feature was computed against any HbA1c label; only label counts per
group and wearable coverage were looked at: CGMacros 15 <5.7 / 16 5.7-6.4 / 14 >=6.5, HR coverage median .93;
BIG IDEAs 8 <5.7 / 8 5.7-6.4 / 0 >=6.5, E4 HR only; Shanghai T2DM 109 recordings, HbA1c in mmol/mol)

Data rules (both arms)
  Dev        CGMacros 45 people, lab A1c (bio.csv "A1c PDL (Lab)"), Dexcom G6 5-min grid (model/cgmacros/dataset.py).
  External   BIG IDEAs 16 (Demographics.csv HbA1c, Dexcom G6; 4th use of these people, first with an A1c label) and
             ShanghaiT2DM (all recordings incl. insulin users: the A1c / mean-glucose relation does not depend on the
             drug; first recording per person; 15-min CGM, readings clipped to 40-400). HbA1c mmol/mol -> % by
             % = mmol/mol / 10.929 + 2.15 (NGSP/IFCC master equation). A unit guard refuses values outside 4-15 %.
             DEVIATION (2026-11-18, after dev was frozen, before any external prediction was scored): the guard
             stopped on a Shanghai value of 144.8 mmol/mol = 15.4 %. The raw column is all mmol/mol (23.5-144.8, none
             < 20), so 15.4 % is a real high A1c, not a unit error; the guard was widened to 4-20 %. Nothing else
             changed; only the label column's range was looked at.
             Hall 2018 has no A1c and is not used.
  Minimum    >= 5 days of CGM readings (count x spacing) per person; otherwise excluded, and the count is printed.
  Known      HbA1c reflects ~3 months; we have 8-14 days. Short windows put a ceiling on any CGM estimate.
  Privacy    results files hold aggregates only; CGMacros stays local.

ARM A (primary): CGM over days -> HbA1c (%)
  Features   on real readings only (no filled grid rows): mean, CV (SD/mean), share > 140, share > 180 mg/dL.
  Models     GMI   = 3.31 + 0.02392 x mean (Bergenstal et al., Diabetes Care 2018, doi:10.2337/dc18-1581). No fit.
             LIN   = a + b x mean, OLS (GMI recalibrated to this population). 2 parameters.
             RIDGE = ridge on the 4 standardized features, alpha = 1.0, fixed (no tuning on 45 people). 5 parameters.
  Dev        leave-one-person-out MAE in A1c points; paired person-bootstrap (2000, seed 0) of MAE(GMI) - MAE(model).
  Choice     a fitted model is eligible only if its LOO MAE <= MAE(GMI) - 0.05 AND the bootstrap lower bound of
             MAE(GMI) - MAE(model) > 0. Among eligible: lowest MAE; if two are within 0.02 of each other, the one with
             fewer parameters (tie-break written in advance). None eligible -> CHOSEN = GMI. Chosen model is refit on
             all 45 and frozen to the dev results file.
  External   only if CHOSEN != GMI. Primary: on all eligible external people pooled, 95% person-bootstrap (resampled
             within dataset, 2000, seed 0) lower bound of MAE(GMI) - MAE(CHOSEN) > 0.
             Secondary, and descriptive when CHOSEN == GMI: MAE, mean bias and share within +-0.5 points for every
             model, per dataset. Expect range restriction (BIG has no diabetes, Shanghai is all T2D) and a sampling
             shift (15-min in Shanghai).

ARM B (secondary, FEASIBILITY ONLY): wearable heart rate alone -> A1c >= 5.7 (prediabetes or diabetes)
  Features   from observed heart rate only, in 5-min bins (CGMacros: Fitbit grid rows with hr_observed; BIG IDEAs:
             E4 HR averaged into 5-min bins). A valid day has >= 12 h of observed bins. Per valid day: rest = 5th
             percentile, night = mean 00:00-06:00 (needs >= 2 h), sd = SD, day_night = mean 08:00-20:00 minus night.
             Person feature = median over valid days; >= 3 valid days required.
  Models     HR     = L2 logistic (C = 1.0) on standardized [rest, night, sd, day_night].
             DEMO   = same logistic on [age, BMI, sex]; HR+DEMO = both. (DEMO exists in dev only: BIG IDEAs has sex,
             no age or BMI; externally the comparator is sex only.)
  Dev        LOO AUC with 95% person-bootstrap CI for each; secondary: AUC gain of HR+DEMO over DEMO (is HR more than a
             proxy for fitness or body size?).
  Gate       external run only if the HR model's dev LOO AUC lower bound > 0.5.
  External   HR model fit on all 45, applied to BIG IDEAs (8 vs 8). Pass if AUC 95% bootstrap lower bound > 0.5.
             Expected CI width about +-0.25 at n = 16: a point estimate near 0.7 is NOT a finding on its own.
             Devices differ (Fitbit vs Empatica E4; both optical).

Every run is logged to model/results/a1c_log.jsonl. Nothing from this module goes into the MCP tool unless Arm A passes
externally, and any output stays an estimate (like GMI), never a diagnosis.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from model.cgmacros.dataset import ROOT, build_grid
from sources.model import Kind

RESULTS = ROOT / "model" / "results"
DEV_OUT = RESULTS / "a1c_dev.json"
EXT_OUT = RESULTS / "a1c_external.json"
LOG = RESULTS / "a1c_log.jsonl"
SEED, NBOOT, MIN_DAYS = 0, 2000, 5.0
A_FEATS = ["mean", "cv", "gt140", "gt180"]
B_FEATS = ["rest", "night", "sd", "day_night"]


# ---------------------------------------------------------------- features
def cgm_features(g: np.ndarray, step_min: float) -> dict | None:
    g = g[np.isfinite(g)]
    if len(g) * step_min / 1440.0 < MIN_DAYS:
        return None
    m = float(g.mean())
    return {"mean": m, "cv": float(g.std() / m), "gt140": float((g > 140).mean()), "gt180": float((g > 180).mean()),
            "days": len(g) * step_min / 1440.0}


def hr_features(ts: pd.Series, hr: pd.Series) -> dict | None:
    """ts/hr: observed 5-min bins only (local clock time)."""
    d = pd.DataFrame({"ts": pd.to_datetime(ts).to_numpy(), "hr": np.asarray(hr, float)}).dropna()
    d["ts"] = pd.to_datetime(d.ts)
    d["day"], d["h"] = d.ts.dt.date, d.ts.dt.hour
    rows = []
    for _, x in d.groupby("day"):
        if len(x) < 12 * 12:
            continue
        night = x[x.h < 6].hr
        day = x[(x.h >= 8) & (x.h < 20)].hr
        rows.append({"rest": x.hr.quantile(0.05), "sd": x.hr.std(),
                     "night": night.mean() if len(night) >= 24 else np.nan,
                     "day_night": (day.mean() - night.mean()) if len(night) >= 24 and len(day) else np.nan})
    if len(rows) < 3:
        return None
    f = pd.DataFrame(rows).median()
    return None if f.isna().any() else {k: float(f[k]) for k in B_FEATS} | {"hr_days": len(rows)}


# ---------------------------------------------------------------- data
def dev_table() -> pd.DataFrame:
    bio = pd.read_csv(ROOT / "data" / "cgmacros" / "csv" / "bio.csv")
    bio.columns = [c.strip() for c in bio.columns]
    grid = build_grid()
    rows, skipped = [], 0
    for _, b in bio.iterrows():
        pid = f"cgmacros-{int(b['subject']):03d}"
        g = grid[grid.participant == pid]
        a = cgm_features(g.glucose.to_numpy(float), 5.0) if len(g) else None
        if a is None:
            skipped += 1
            continue
        obs = g[g.hr_observed.astype(bool)]
        h = hr_features(obs.ts.dt.tz_localize(None) if obs.ts.dt.tz is not None else obs.ts, obs.hr)
        rows.append({"pid": pid, "dataset": "cgmacros", "a1c": float(b["A1c PDL (Lab)"]), **a,
                     "age": float(b["Age"]), "bmi": float(b["BMI"]),
                     "male": float(str(b["Gender"]).strip().upper().startswith("M")), **(h or {})})
    print(f"dev: {len(rows)} people, {skipped} excluded (< {MIN_DAYS} days CGM)", file=sys.stderr)
    return pd.DataFrame(rows)


def mmol_mol_to_pct(v: float) -> float:
    return v / 10.929 + 2.15


def unit_guard(p: float, who: str) -> float:
    if not 4.0 <= p <= 20.0:
        raise ValueError(f"{who}: A1c {p:.2f}% outside 4-20 %, unit problem")
    return p


def bigideas_table() -> pd.DataFrame:
    from sources.external_cgm import parse_bigideas
    from sources.normalize import normalize
    demo = pd.read_csv(ROOT / "data" / "bigideas" / "Demographics.csv", encoding="utf-8-sig")
    rows = []
    for _, d in demo.iterrows():
        num = f"{int(d['ID']):03d}"
        r = parse_bigideas(ROOT / "data" / "bigideas" / num, f"bigideas-{num}")
        g = np.array([float(m.value) for m in normalize(r.raws, tz="UTC").measurements
                      if m.kind is Kind.GLUCOSE_CGM], float)          # the parser also emits meal macros
        a = cgm_features(np.clip(g, 40, 400), 5.0)
        if a is None:
            continue
        hr = pd.read_csv(ROOT / "data" / "bigideas" / num / f"HR_{num}.csv", skipinitialspace=True)
        hr.columns = ["ts", "hr"]
        hr["ts"] = pd.to_datetime(hr.ts, format="mixed")
        b5 = hr.set_index("ts").hr.resample("5min").mean().dropna()
        h = hr_features(pd.Series(b5.index), pd.Series(b5.values))
        rows.append({"pid": f"bigideas-{num}", "dataset": "bigideas",
                     "a1c": unit_guard(float(d["HbA1c"]), num), **a,
                     "male": float(str(d["Gender"]).strip().upper().startswith("M")), **(h or {})})
    return pd.DataFrame(rows)


def shanghai_table() -> pd.DataFrame:
    raw = ROOT / "data" / "shanghai" / "raw"
    s = pd.read_excel(raw / "Shanghai_T2DM_Summary.xlsx")
    rows, seen = [], set()
    for _, r in s.sort_values("Patient Number").iterrows():
        num = str(r["Patient Number"])
        person = num.split("_")[0]
        h = pd.to_numeric(r["HbA1c (mmol/mol)"], errors="coerce")      # missing is "/" in the source
        if person in seen or pd.isna(h):
            continue
        files = sorted((raw / "Shanghai_T2DM").glob(f"{num}.*"))
        if not files:
            continue
        d = pd.read_excel(files[0])
        col = next(c for c in d.columns if str(c).strip().startswith("CGM"))   # 2 files name it just "CGM "
        g = pd.to_numeric(d[col], errors="coerce").to_numpy(float)
        if np.nanmedian(g) < 30:                                               # mmol/L, not mg/dL
            g = g * 18.016
        a = cgm_features(np.clip(g, 40, 400), 15.0)
        if a is None:
            continue
        seen.add(person)
        rows.append({"pid": f"shanghai-{len(seen):03d}", "dataset": "shanghai",
                     "a1c": unit_guard(mmol_mol_to_pct(float(h)), f"shanghai-{len(seen):03d}"),
                     **a})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- models
def gmi(df: pd.DataFrame) -> np.ndarray:
    return 3.31 + 0.02392 * df["mean"].to_numpy(float)


class Lin:
    k = 2

    def fit(self, df):
        self.b, self.a = np.polyfit(df["mean"].to_numpy(float), df.a1c.to_numpy(float), 1)
        return self

    def predict(self, df):
        return self.a + self.b * df["mean"].to_numpy(float)


class Ridge:
    k = 5

    def fit(self, df, alpha: float = 1.0):
        X = df[A_FEATS].to_numpy(float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        Z = (X - self.mu) / self.sd
        y = df.a1c.to_numpy(float)
        self.c = y.mean()
        self.w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - self.c))
        return self

    def predict(self, df):
        return self.c + ((df[A_FEATS].to_numpy(float) - self.mu) / self.sd) @ self.w


def loo(df: pd.DataFrame, make) -> np.ndarray:
    out = np.empty(len(df))
    for i in range(len(df)):
        out[i] = make().fit(df.drop(df.index[i])).predict(df.iloc[[i]])[0]
    return out


def boot_idx(groups: np.ndarray, rng) -> np.ndarray:
    """Person bootstrap, resampled within each group (dataset)."""
    return np.concatenate([rng.choice(np.where(groups == g)[0], (groups == g).sum()) for g in np.unique(groups)])


def mae_diff_ci(y, p_ref, p_new, groups) -> list[float]:
    rng = np.random.default_rng(SEED)
    e_ref, e_new = np.abs(y - p_ref), np.abs(y - p_new)
    d = [(e_ref[i].mean() - e_new[i].mean()) for i in (boot_idx(groups, rng) for _ in range(NBOOT))]
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def describe(y, p) -> dict:
    e = p - y
    return {"mae": float(np.abs(e).mean()), "bias": float(e.mean()), "within_0.5": float((np.abs(e) <= 0.5).mean())}


def auc(y, s) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def auc_ci(y, s) -> list[float]:
    rng = np.random.default_rng(SEED)
    v = []
    for _ in range(NBOOT):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(y):
            v.append(auc(y[i], s[i]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def logit(cols):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    class M:
        def fit(self, df):
            self.m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0)).fit(df[cols], df.a1c >= 5.7)
            return self

        def predict(self, df):
            return self.m.predict_proba(df[cols])[:, 1]
    return M


def log(entry: dict) -> None:
    entry = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **entry}
    with LOG.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------- runs
def run_dev() -> dict:
    df = dev_table()
    y, grp = df.a1c.to_numpy(float), np.zeros(len(df))
    p = {"GMI": gmi(df), "LIN": loo(df, Lin), "RIDGE": loo(df, Ridge)}
    a = {m: describe(y, v) for m, v in p.items()}
    for m in ("LIN", "RIDGE"):
        a[m]["gmi_minus_model_ci"] = mae_diff_ci(y, p["GMI"], p[m], grp)
    elig = [m for m in ("LIN", "RIDGE")
            if a[m]["mae"] <= a["GMI"]["mae"] - 0.05 and a[m]["gmi_minus_model_ci"][0] > 0]
    elig.sort(key=lambda m: a[m]["mae"])
    if len(elig) == 2 and a[elig[1]]["mae"] - a[elig[0]]["mae"] <= 0.02:
        elig.sort(key=lambda m: {"LIN": Lin.k, "RIDGE": Ridge.k}[m])
    chosen = elig[0] if elig else "GMI"
    frozen = None
    if chosen == "LIN":
        f = Lin().fit(df)
        frozen = {"a": float(f.a), "b": float(f.b)}
    elif chosen == "RIDGE":
        f = Ridge().fit(df)
        frozen = {"mu": f.mu.tolist(), "sd": f.sd.tolist(), "c": float(f.c), "w": f.w.tolist()}

    b = {"people_with_hr": int(df[B_FEATS].notna().all(1).sum()) if set(B_FEATS) <= set(df) else 0}
    hb = df.dropna(subset=B_FEATS) if b["people_with_hr"] else df.iloc[:0]
    if len(hb) >= 10:
        yb = (hb.a1c >= 5.7).to_numpy()
        b["positives"] = int(yb.sum())
        scores = {}
        for name, cols in {"HR": B_FEATS, "DEMO": ["age", "bmi", "male"],
                           "HR+DEMO": B_FEATS + ["age", "bmi", "male"]}.items():
            scores[name] = loo(hb, logit(cols))
            b[name] = {"auc": auc(yb, scores[name]), "ci": auc_ci(yb, scores[name])}
        rng, gain = np.random.default_rng(SEED), []
        for _ in range(NBOOT):
            i = rng.integers(0, len(yb), len(yb))
            if 0 < yb[i].sum() < len(yb):
                gain.append(auc(yb[i], scores["HR+DEMO"][i]) - auc(yb[i], scores["DEMO"][i]))
        b["gain_hr_over_demo_ci"] = [float(np.percentile(gain, 2.5)), float(np.percentile(gain, 97.5))]
        b["gate_external"] = bool(b["HR"]["ci"][0] > 0.5)
    out = {"arm_a": {"n": len(df), "models": a, "eligible": elig, "chosen": chosen, "frozen": frozen},
           "arm_b": b}
    DEV_OUT.write_text(json.dumps(out, indent=1))
    log({"run": "dev", "chosen": chosen, "arm_b_gate": b.get("gate_external")})
    return out


def run_external() -> dict:
    if not DEV_OUT.exists():
        raise SystemExit("run --dev first: the external test needs the frozen dev choice")
    dev = json.loads(DEV_OUT.read_text())
    A = dev["arm_a"]
    ext = pd.concat([bigideas_table(), shanghai_table()], ignore_index=True)
    y, grp = ext.a1c.to_numpy(float), ext.dataset.to_numpy()
    preds = {"GMI": gmi(ext)}
    fr = A["frozen"]
    if A["chosen"] == "LIN":
        preds["LIN"] = fr["a"] + fr["b"] * ext["mean"].to_numpy(float)
    elif A["chosen"] == "RIDGE":
        Z = (ext[A_FEATS].to_numpy(float) - np.array(fr["mu"])) / np.array(fr["sd"])
        preds["RIDGE"] = fr["c"] + Z @ np.array(fr["w"])
    res = {"n": {k: int(v) for k, v in ext.dataset.value_counts().items()}, "chosen": A["chosen"],
           "pooled": {m: describe(y, p) for m, p in preds.items()},
           "per_dataset": {ds: {m: describe(y[grp == ds], p[grp == ds]) for m, p in preds.items()}
                           for ds in np.unique(grp)}}
    if A["chosen"] != "GMI":
        ci = mae_diff_ci(y, preds["GMI"], preds[A["chosen"]], grp)        # GMI - chosen; pass if lower bound > 0
        res["primary"] = {"gmi_minus_chosen_ci": ci, "pass": bool(ci[0] > 0)}
    else:
        res["primary"] = "not run: no fitted model beat GMI on dev (pre-registered); external is descriptive"
    b = dev["arm_b"]
    if b.get("gate_external"):
        dv = dev_table().dropna(subset=B_FEATS)
        bi = ext[ext.dataset == "bigideas"].dropna(subset=B_FEATS)
        s = logit(B_FEATS)().fit(dv).predict(bi)
        yb = (bi.a1c >= 5.7).to_numpy()
        ci = auc_ci(yb, s)
        res["arm_b"] = {"n": len(bi), "positives": int(yb.sum()), "auc": auc(yb, s), "ci": ci, "pass": bool(ci[0] > 0.5)}
    else:
        res["arm_b"] = "not run: HR model failed the dev gate (pre-registered)"
    EXT_OUT.write_text(json.dumps(res, indent=1))
    log({"run": "external", "chosen": A["chosen"], "primary": res["primary"], "arm_b": res["arm_b"]})
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dev", action="store_true")
    g.add_argument("--external", action="store_true")
    args = ap.parse_args()
    print(json.dumps(run_dev() if args.dev else run_external(), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
