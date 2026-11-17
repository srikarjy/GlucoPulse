"""External test: does the new forecaster change triage safely on people from other studies?

    CGMACROS_SPLIT=confirm python -m model.cgmacros.external --prepare      # grids, moments, TFT A/D, HGB, context
    .venv-chronos/bin/python model/cgmacros/chronos_run.py chronos2 512 external
    CGMACROS_SPLIT=confirm python -m model.cgmacros.external --score

Data (never used for any model choice): BIG IDEAs (16 adults, Dexcom G6, food logs; doi:10.13026/aw6y-fc44, ODC-By)
and Hall et al. 2018 (57 adults incl. type 2 diabetes, Dexcom G4, no meals; doi:10.1371/journal.pbio.2005143,
CC BY), read through sources/external_cgm.py into the same 5-min grid as CGMacros.

PRE-REGISTERED (written 2026-11-16, before any forecast for these 73 people was computed; event counts by person and
hour were looked at to choose the night secondary):
  Models      frozen, nothing refit or tuned on these people: TFT A (glucose only) and TFT D (no_new_meal) from the
              confirm-split checkpoints; the new forecaster = 0.7 x hgb_long_person (fit on the 31 confirm-training
              people) + 0.3 x Chronos-2 (chronos-forecasting 2.0.1, amazon/chronos-2 rev 29ec3766..., 512 readings),
              band = Chronos-2 offsets. Change from sections 11-12: HGB is fit for all 12 steps (same recipe per step),
              because the rule reads the whole hour.
  Missing     heart rate and METs are resting defaults (not observed) in both datasets; Hall has no meals. Identical
              for every arm. The personal-history features start from zero for each person, as for a new user.
  Moments     every 6th valid window per person (30 min apart): 2 h contiguous history and a contiguous next hour.
              Label = outcome_label (urgent: sustained < 70 or > 250 within 60 min).
  Arms        current = rule_level(current, TFT D no_new_meal, TFT A history_only)
              new     = rule_level(current, NEW forecaster,      TFT A history_only)
              (A stays in the history_only slot, which is the safety floor of section 12 inside the rule)
  Primary     on all 73 people, both of:
                1. urgent recall non-inferior: 95% person-bootstrap lower bound of (new - current) > -0.05;
                2. urgent false alarms (non-urgent moments called urgent) at most 2.0 points higher.
  Secondary   per dataset; recall on urgent moments not already out of range at as_of; excluding as_of 00:00-06:00
              (night lows include sensor-compression artifacts); monitor calls; under/over-triage; T+60 RMSE of NEW vs
              A vs D (per dataset, person-bootstrap); highs are NOT testable (3 events in both datasets).
  Power       urgent moments and the people they come from are printed by --prepare before scoring.

PRE-REGISTERED, 5-MIN ALARMS (written 2026-11-16 after the triage result above and before any alarm stream was run on
these people; their second use; run with --alarms):
  Question    which alarm, checked at every 5-min reading, warns of lows early with the fewest false alarms?
  Streams     event_model (model/cgmacros/event.py, fit on the 31 confirm-training people, its pre-registered
              threshold; replay.event_model_stream), alert20 (15-min slope projected 20 min, replay.alert20), and
              threshold (reactive reference). Nothing is refit or tuned on these people.
  Scoring     unchanged from model/cgmacros/replay.py: events = >= 3 readings < 70 or > 250 in a gap-free segment,
              timely = an in-range alarm >= 15 min before onset within 60 min, false alarm = an alarm episode (30-min
              lockout) with no onset in the next 60 min, per day of readings; each stream against its own
              rate-matched random baseline (20 seeds). Lows only are claimed (3 highs in both datasets).
  Primary     event_model vs alert20 on all 73 people, both of:
                1. fewer false alarms: 95% person-bootstrap interval of (alert20 - event_model) false alarms/day
                   excludes 0 (above 0);
                2. timely share non-inferior: lower bound of (event_model - alert20) timely share > -0.10.
  Secondary   each stream vs its random baseline; per dataset; daytime only (readings 06:00-24:00 local clock,
              re-segmented; night lows include sensor-compression artifacts); median lead.
  Disclosed   heart rate / METs resting defaults; Hall has no meal logs (meal features 0); hour-of-day here is real
              clock time while CGMacros timestamps were shifted, so the event model's hour features are out of
              distribution; the rule-every-5-min arm is not run (TFT at ~142k readings).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from model.cgmacros import lab
from model.cgmacros.cases import OUT, outcome_label
from model.cgmacros.dataset import ROOT, build_grid, grid_from_measurements
from model.cgmacros.splits import SCHEME, split_participants
from model.cgmacros.train import ENCODER, HORIZON

EXT = ROOT / "data" / "bigideas" / "external_grid.pkl"
CACHE = OUT / "external_confirm.pkl"
CTX = OUT / "lab_external_context.npy"
PRED = OUT / "lab_external_pred_chronos2.npy"
RESULTS = ROOT / "model" / "results" / "cgmacros_external.json"
O = {"routine": 0, "monitor": 1, "urgent": 2}


def external_grid() -> pd.DataFrame:
    from sources.external_cgm import parse_bigideas, parse_hall2018
    from sources.normalize import normalize
    frames = []
    for i in range(1, 17):
        pid = f"bigideas-{i:03d}"
        r = parse_bigideas(ROOT / "data" / "bigideas" / f"{i:03d}", pid)
        frames.append(grid_from_measurements(pid, normalize(r.raws, tz="UTC").measurements, "dexcom-g6"))
    for pid, r in parse_hall2018(ROOT / "data" / "hall2018" / "S1_Data.txt").items():
        frames.append(grid_from_measurements(pid, normalize(r.raws, tz="UTC").measurements, "dexcom-g4"))
    g = pd.concat(frames, ignore_index=True)
    g.to_pickle(EXT)
    return g


def tft_scenarios(variant: str, grid: pd.DataFrame, moments: pd.DataFrame) -> np.ndarray:
    """(n, 12, 3) p10/p50/p90 from the deployment decoder (scenarios.future_frame, no new meal), batched."""
    from pytorch_forecasting import TimeSeriesDataSet

    from model.cgmacros.scenarios import ScenarioForecaster, future_frame
    f = ScenarioForecaster(variant)
    by_p = {p: g.set_index("time_idx") for p, g in grid.groupby("participant")}
    frames = []
    for n, (p, t) in enumerate(zip(moments.participant, moments.time_idx)):
        hist = by_p[p].loc[int(t) - ENCODER + 1:int(t)].reset_index()
        frames.append(future_frame(hist, "no_new_meal", f"w{n}"))
    out = np.zeros((len(moments), HORIZON, 3), np.float32)
    lv = f.model.loss.quantiles
    for s in range(0, len(frames), 500):
        ds = TimeSeriesDataSet.from_dataset(f.training, pd.concat(frames[s:s + 500], ignore_index=True),
                                            predict=True, stop_randomization=True)
        r = f.model.predict(ds.to_dataloader(train=False, batch_size=500, num_workers=0), mode="raw",
                            return_index=True, trainer_kwargs={"accelerator": "cpu", "logger": False})
        q = f.model.loss.to_quantiles(r.output.prediction).numpy()
        for j, gid in enumerate(r.index["participant"]):
            out[int(gid[1:])] = q[j][:, [lv.index(0.1), lv.index(0.5), lv.index(0.9)]]
        print(f"  {variant}: {min(s + 500, len(frames))}/{len(frames)}", flush=True)
    return out


def prepare() -> None:
    ext = external_grid()
    people = sorted(ext.participant.unique())
    w = lab.all_windows(ext, people)
    moments = pd.concat([d.iloc[::6] for _, d in w.groupby("participant")], ignore_index=True)
    moments["label"] = [outcome_label(y) for y in moments.y]
    moments["dataset"] = moments.participant.str.split("-").str[0]
    ts = ext.set_index(["participant", "time_idx"])["ts"]
    moments["hour"] = ts.loc[list(zip(moments.participant, moments.time_idx))].dt.hour.to_numpy()
    moments[lab.LONG + lab.PERSON] = lab.extra_features(moments, grid=ext)[lab.LONG + lab.PERSON].to_numpy()

    cg = build_grid()
    train_ids, _, _ = split_participants(sorted(cg.participant.unique()))
    train = lab.all_windows(cg, train_ids)
    train[lab.LONG + lab.PERSON] = lab.extra_features(train, grid=cg)[lab.LONG + lab.PERSON].to_numpy()
    cols = lab.FEATURES + lab.LONG + lab.PERSON
    Xt, Xd = train[cols].to_numpy(float), moments[cols].to_numpy(float)
    Yt = lab.stack(train.y) - train["now"].to_numpy()[:, None]
    hgb = np.zeros((len(moments), HORIZON))
    for s in range(HORIZON):
        hgb[:, s] = moments["now"].to_numpy() + lab._hgb(Xt, Yt[:, s], Xd)
        print(f"  hgb step {s + 1}/12", flush=True)

    A = tft_scenarios("glucose", ext, moments)
    D = tft_scenarios("meals_activity", ext, moments)
    ctx = np.full((len(moments), lab.CTX_LEN), np.nan, dtype=np.float32)
    for p, d in moments.groupby("participant"):
        g = ext[ext.participant == p].set_index("time_idx")["glucose"]
        lo = int(g.index.min())
        s = np.r_[np.full(lab.CTX_LEN, np.nan, np.float32),
                  g.reindex(range(lo, int(g.index.max()) + 1)).to_numpy(np.float32)]
        for row, t in zip(d.index, d.time_idx):
            end = int(t) - lo + lab.CTX_LEN + 1
            ctx[row] = s[end - lab.CTX_LEN:end]
    assert np.allclose(ctx[:, -1], moments["now"].to_numpy())
    np.save(CTX, ctx)
    pd.to_pickle({"moments": moments, "hgb": hgb, "A": A, "D": D}, CACHE)
    urg = moments[moments.label == "urgent"]
    print(f"prepared {len(moments)} moments / {moments.participant.nunique()} people; urgent {len(urg)} from "
          f"{urg.participant.nunique()} people; by dataset {urg.dataset.value_counts().to_dict()}; urgent not "
          f"already < 70 at as_of: {int((urg.now >= 70).sum())}")


def _res(q: np.ndarray, name: str) -> dict:
    from model.cgmacros.scenarios import scenario_resource
    d = {"p10": [float(v) for v in q[:, 0]], "p50": [float(v) for v in q[:, 1]], "p90": [float(v) for v in q[:, 2]]}
    return scenario_resource("x", "t", name, d, "external")


def score() -> None:
    from agents.triage.rules import rule_level
    c = pd.read_pickle(CACHE)
    m, hgb, A, D = c["moments"], c["hgb"], c["A"], c["D"]
    ch = np.load(PRED)
    assert len(ch) == len(m)
    med = 0.7 * hgb + 0.3 * ch[:, :, 1]
    NEW = np.stack([med + ch[:, :, 0] - ch[:, :, 1], med, med + ch[:, :, 2] - ch[:, :, 1]], axis=2)
    cur, new = [], []
    for i in range(len(m)):
        gs = {"current_mg_dl": float(m.now.iat[i])}
        a = _res(A[i], "history_only")
        cur.append(O[rule_level(gs, _res(D[i], "no_new_meal"), a)])
        new.append(O[rule_level(gs, _res(NEW[i], "no_new_meal"), a)])
    m = m.assign(cur=cur, new=new, y_lvl=m.label.map(O)).reset_index(drop=True)
    y = lab.stack(m.y)

    def summary(mask):
        d = m[mask].reset_index(drop=True)
        pos = (d.y_lvl == 2).to_numpy()
        out = {"moments": int(len(d)), "urgent": int(pos.sum()), "urgent_people": int(d.participant[pos].nunique())}
        for arm in ("cur", "new"):
            lvl = d[arm].to_numpy()
            out[arm] = {"urgent_recall": round(float((lvl[pos] == 2).mean()), 3) if pos.any() else None,
                        "urgent_false_alarm": round(float((lvl[~pos] == 2).mean()), 4),
                        "monitor_or_higher_when_routine": round(float((lvl[d.y_lvl == 0] >= 1).mean()), 4),
                        "under_triage": round(float((lvl < d.y_lvl).mean()), 4),
                        "over_triage": round(float((lvl > d.y_lvl).mean()), 4)}
        if pos.any():
            dp = d[pos].reset_index(drop=True)
            idx = dp.groupby("participant").indices
            ppl, rng, diffs = list(idx), np.random.default_rng(0), []
            dn, dc = (dp.new == 2).to_numpy(), (dp.cur == 2).to_numpy()
            for _ in range(2000):
                ix = np.concatenate([idx[p] for p in rng.choice(ppl, len(ppl))])
                diffs.append(dn[ix].mean() - dc[ix].mean())
            out["recall_diff_ci95"] = [round(float(np.percentile(diffs, 2.5)), 3),
                                       round(float(np.percentile(diffs, 97.5)), 3)]
        return out

    def rmse_block(mask):
        yy = y[mask, lab.S60]
        nm, am, dm = med[mask, lab.S60], A[mask][:, lab.S60, 1], D[mask][:, lab.S60, 1]
        r = {"new": round(lab.rmse(nm, yy), 2), "A": round(lab.rmse(am, yy), 2), "D_no_meal": round(lab.rmse(dm, yy), 2)}
        idx = m[mask].reset_index(drop=True).groupby("participant").indices
        ppl, rng, g = list(idx), np.random.default_rng(0), []
        for _ in range(1000):
            ix = np.concatenate([idx[p] for p in rng.choice(ppl, len(ppl))])
            g.append(lab.rmse(am[ix], yy[ix]) - lab.rmse(nm[ix], yy[ix]))
        r["gain_new_vs_A_ci95"] = [round(float(np.percentile(g, 2.5)), 2), round(float(np.percentile(g, 97.5)), 2)]
        return r

    allm = np.ones(len(m), bool)
    out = {"preregistration": __doc__.split("PRE-REGISTERED")[1].strip(), "all_73": summary(allm)}
    for ds in ("bigideas", "hall"):
        out[ds] = summary((m.dataset == ds).to_numpy())
    out["not_out_of_range_at_as_of"] = summary(((m.now >= 70) & (m.now <= 250)).to_numpy())
    out["excluding_night_00_06"] = summary((m.hour >= 6).to_numpy())
    out["t60_rmse"] = {"all_73": rmse_block(allm), "bigideas": rmse_block((m.dataset == "bigideas").to_numpy()),
                       "hall": rmse_block((m.dataset == "hall").to_numpy())}
    a = out["all_73"]
    out["criterion_1_recall_noninferior"] = bool(a["recall_diff_ci95"][0] > -0.05)
    out["criterion_2_false_alarm_at_most_2pt"] = bool(
        a["new"]["urgent_false_alarm"] - a["cur"]["urgent_false_alarm"] <= 0.02)
    out["confirmed"] = out["criterion_1_recall_noninferior"] and out["criterion_2_false_alarm_at_most_2pt"]
    RESULTS.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "preregistration"}, indent=1))


def alarms() -> None:
    from model.cgmacros import replay as R
    ext = pd.read_pickle(EXT)
    people = sorted(ext.participant.unique())
    em = R.event_model_stream()
    streams = {"event_model": em, "alert20": lambda p, f: R.alert20(f["glucose"].to_numpy(dtype=float)),
               "threshold": lambda p, f: R.threshold(f["glucose"].to_numpy(dtype=float))}

    def evaluate(grid_df):
        frames = list(R.segments(grid_df, people))
        per = R.run(frames, streams)
        ps = [p for p in people if p in per["alert20"]]
        res = {n: R.summarize(per[n], ps) for n in per}
        for n in per:
            res[n]["timely_minus_random_ci95"] = R.boot(per[n], None, ps)
        res["event_minus_alert20_timely_ci95"] = R.boot(per["event_model"], per["alert20"], ps)
        rng, d = np.random.default_rng(0), []
        for _ in range(2000):
            pick = rng.choice(ps, len(ps))
            days = sum(per["alert20"][p]["readings"] for p in pick) * R.DT / 1440
            d.append((sum(per["alert20"][p]["fa"] for p in pick) - sum(per["event_model"][p]["fa"] for p in pick))
                     / days)
        res["alert20_minus_event_fa_per_day_ci95"] = [round(float(np.percentile(d, 2.5)), 2),
                                                      round(float(np.percentile(d, 97.5)), 2)]
        for ds in ("bigideas", "hall"):
            sub = [p for p in ps if p.startswith(ds)]
            res[ds] = {n: R.summarize(per[n], sub) for n in per}
        return res

    out = {"preregistration": __doc__.split("PRE-REGISTERED, 5-MIN ALARMS")[1].strip(), "all": evaluate(ext)}
    day = ext[ext.ts.dt.hour >= 6].copy()
    day["time_idx"] = day.groupby("participant").cumcount() + day.groupby("participant")["time_idx"].transform(
        lambda t: (t.diff().fillna(1) != 1).cumsum() * 10 ** 6)          # new segment wherever night was removed
    out["daytime_only"] = evaluate(day)
    a = out["all"]
    out["criterion_1_fewer_false_alarms"] = bool(a["alert20_minus_event_fa_per_day_ci95"][0] > 0)
    out["criterion_2_timely_noninferior"] = bool(a["event_minus_alert20_timely_ci95"][0] > -0.10)
    out["confirmed"] = out["criterion_1_fewer_false_alarms"] and out["criterion_2_timely_noninferior"]
    path = ROOT / "model" / "results" / "cgmacros_external_alarms.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "preregistration"}, indent=1, default=str))


def main() -> int:
    if SCHEME != "confirm":
        raise SystemExit("run with CGMACROS_SPLIT=confirm (frozen confirm-split models)")
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--alarms", action="store_true")
    a = ap.parse_args()
    if a.prepare:
        prepare()
    if a.score:
        score()
    if a.alarms:
        alarms()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
