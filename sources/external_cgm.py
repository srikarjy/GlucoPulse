"""Adapters for two public CGM datasets used as an external test (people CGMacros never saw).

BIG IDEAs Lab Glycemic Variability and Wearable Device Data, v1.1.3 (Cho, Kim, Bent, Dunn; PhysioNet,
doi:10.13026/aw6y-fc44; ODC-By 1.0): 16 adults with high-normal glucose / prediabetes, Dexcom G6 every 5 min,
food logs, Empatica E4. Folder layout: <root>/0NN/Dexcom_0NN.csv, Food_Log_0NN.csv.

Hall et al., "Glucotypes reveal new patterns of glucose dysregulation", PLoS Biol 2018,
doi:10.1371/journal.pbio.2005143, S1 Data (CC BY 4.0): 57 adults (normal, prediabetes, type 2 diabetes),
Dexcom G4 every 5 min, no meals or heart rate. Tab-separated: DisplayTime, GlucoseValue, subjectId, InternalTime.

Choices (same for every arm of the test):
  * Dexcom "Low"/"High" are the sensor's reporting limits and become 40 / 400 mg/dL (dropping them would delete
    real lows).
  * Hall timestamps: DisplayTime (the receiver's local clock, so hour of day means something); InternalTime is ignored.
  * Empatica heart rate is not read: the grid only takes Fitbit heart rate, so heart rate and METs are resting
    defaults for both datasets, flagged as not observed.
  * Patient ids are pseudonyms by position (bigideas-001.., hall-001..), never the source's subject codes.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from sources._common import AdapterResult, checked_patient
from sources.model import Kind, RawMeasurement

LIMITS = {"low": "40", "high": "400"}
FOOD = {"total_carb": Kind.MEAL_CARBS, "protein": Kind.MEAL_PROTEIN, "total_fat": Kind.MEAL_FAT,
        "dietary_fiber": Kind.MEAL_FIBER}


def _glucose_cell(cell: str) -> str:
    c = cell.strip()
    return LIMITS.get(c.lower(), c)


def parse_bigideas(folder: str | Path, patient: str) -> AdapterResult:
    folder = Path(folder)
    patient = checked_patient(patient)
    out = AdapterResult()
    num = folder.name
    dex = folder / f"Dexcom_{num}.csv"
    with dex.open(newline="", encoding="utf-8-sig") as fh:
        for k, row in enumerate(csv.DictReader(fh), start=2):
            if (row.get("Event Type") or "").strip() != "EGV":
                out.stats["skipped_non_egv"] += 1
                continue
            ts_text = (row.get("Timestamp (YYYY-MM-DDThh:mm:ss)") or "").strip()
            cell = _glucose_cell(row.get("Glucose Value (mg/dL)") or "")
            if not ts_text or not cell:
                out.stats["skipped_empty_cell"] += 1
                continue
            out.raws.append(RawMeasurement(patient=patient, kind=Kind.GLUCOSE_CGM, ts=datetime.fromisoformat(ts_text),
                                           value=cell, unit="mg/dL", source="bigideas", device="dexcom-g6",
                                           raw_ref=f"{dex.name}:{k}"))
    food = folder / f"Food_Log_{num}.csv"
    if food.exists():
        # one row per food ITEM; items with the same start time are one meal, so their macros are summed (otherwise
        # normalize() sees same-time values of one kind as conflicting duplicates and quarantines them)
        meals: dict[datetime, dict[str, float]] = {}
        refs: dict[datetime, str] = {}
        with food.open(newline="", encoding="utf-8-sig") as fh:
            for k, row in enumerate(csv.DictReader(fh), start=2):
                row = {(key or "").strip(): v for key, v in row.items()}
                ts_text = (row.get("time_begin") or "").strip()
                if not ts_text:
                    out.stats["skipped_food_no_time"] += 1
                    continue
                ts = datetime.fromisoformat(ts_text)
                m = meals.setdefault(ts, {})
                refs.setdefault(ts, f"{food.name}:{k}")
                out.stats["food_items"] += 1
                for col in FOOD:
                    v = (row.get(col) or "").strip()
                    try:
                        m[col] = m.get(col, 0.0) + float(v) if v else m.get(col, 0.0)
                    except ValueError:
                        out.stats["skipped_food_bad_number"] += 1
        for ts, m in sorted(meals.items()):
            for col, kind in FOOD.items():
                out.raws.append(RawMeasurement(patient=patient, kind=kind, ts=ts, value=round(m.get(col, 0.0), 2),
                                               unit="g", source="bigideas", device="food-log", raw_ref=refs[ts]))
        out.stats["meals"] = len(meals)
    return out


INSULIN_WORDS = ("insulin", "glargine", "aspart", "lispro", "detemir", "degludec", "glulisine", "humulin", "novolin",
                 "protamine")


def parse_shanghai_t2dm(root: str | Path) -> dict[str, AdapterResult]:
    """ShanghaiT2DM (Zhao et al., Sci Data 2023, doi:10.1038/s41597-023-01940-7; figshare 10.6084/m9.figshare.21600933,
    CC BY 4.0): 100 people with type 2 diabetes, CGM every 15 min. Only recordings with NO insulin are returned: none in
    the summary's hypoglycemic agents and no value in any insulin column of the recording. One AdapterResult per
    recording, keyed shanghai-PPP-R (PPP = person pseudonym by sorted person number, R = recording index); the source's
    patient numbers and dates are not kept. Needs pandas with openpyxl/xlrd."""
    import pandas as pd
    root = Path(root)
    summary = pd.read_excel(root / "Shanghai_T2DM_Summary.xlsx")
    persons = sorted({str(n).split("_")[0] for n in summary["Patient Number"]})
    pid = {p: f"{i:03d}" for i, p in enumerate(persons, start=1)}
    seen: dict[str, int] = {}
    out = {}
    for num, agents in zip(summary["Patient Number"], summary["Hypoglycemic Agents"].astype(str)):
        if any(w in agents.lower() for w in INSULIN_WORDS):
            continue
        files = sorted((root / "Shanghai_T2DM").glob(f"{num}.*"))
        if not files:
            continue
        d = pd.read_excel(files[0])
        icols = [c for c in d.columns if "insulin" in c.lower() and "non-insulin" not in c.lower()]
        if icols and d[icols].notna().any().any():
            continue
        person = pid[str(num).split("_")[0]]
        seen[person] = seen.get(person, 0) + 1
        patient = checked_patient(f"shanghai-{person}-{seen[person]}")
        res = AdapterResult()
        for k, (ts, v) in enumerate(zip(d["Date"], d["CGM (mg / dl)"]), start=2):
            if pd.isna(ts) or pd.isna(v):
                res.stats["skipped_empty_cell"] += 1
                continue
            g = float(v)
            if not 40.0 <= g <= 400.0:                # the sensor's reporting limits (2.2 mmol/L floor = 39.6)
                res.stats["clipped_to_sensor_limit"] += 1
                g = min(max(g, 40.0), 400.0)
            res.raws.append(RawMeasurement(patient=patient, kind=Kind.GLUCOSE_CGM,
                                           ts=pd.Timestamp(ts).to_pydatetime(), value=g, unit="mg/dL",
                                           source="shanghai_t2dm", device="cgm-15min", raw_ref=f"{files[0].name}:{k}"))
        out[patient] = res
    return out


def parse_hall2018(path: str | Path) -> dict[str, AdapterResult]:
    """One AdapterResult per subject, keyed by pseudonym hall-NNN (subjects in sorted order)."""
    path = Path(path)
    rows: dict[str, list[tuple[int, dict]]] = {}
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for k, row in enumerate(csv.DictReader(fh, delimiter="\t"), start=2):
            rows.setdefault(row["subjectId"].strip(), []).append((k, row))
    out = {}
    for n, subject in enumerate(sorted(rows), start=1):
        patient = checked_patient(f"hall-{n:03d}")
        res = AdapterResult()
        for k, row in rows[subject]:
            cell = _glucose_cell(row.get("GlucoseValue") or "")
            ts_text = (row.get("DisplayTime") or "").strip()
            if not cell or not ts_text:
                res.stats["skipped_empty_cell"] += 1
                continue
            res.raws.append(RawMeasurement(patient=patient, kind=Kind.GLUCOSE_CGM,
                                           ts=datetime.fromisoformat(ts_text), value=cell, unit="mg/dL",
                                           source="hall2018", device="dexcom-g4", raw_ref=f"{path.name}:{k}"))
        out[patient] = res
    return out


def parse_cgmnd(root: str | Path, min_age: int = 18) -> dict[str, AdapterResult]:
    """Jaeb Center "Glucose Sensor Profiles in Healthy Non-Diabetic Subjects" (CGMND, 2017; Shah et al., J Clin
    Endocrinol Metab 2019, doi:10.1210/jc.2018-02763; public.jaeb.org/datasets/diabetes, CGMND.zip). Blinded 5-min CGM.
    Only CGM rows (calibration rows skipped) of participants aged >= min_age at enrolment, from NonDiabPtRoster.csv.
    Times are relative (days from enrolment + clock time), so each person's day 0 is anchored to 2017-01-01: hour of day
    is real, the calendar date is not. Values 39 and 401 are the sensor's LOW/HIGH codes and become 40 / 400.
    Keyed cgmnd-NNN by sorted PtID (the source ids are not kept)."""
    from datetime import timedelta
    root = Path(root)
    with (root / "NonDiabPtRoster.csv").open(newline="", encoding="utf-8-sig") as fh:
        adults = {r["PtID"].strip() for r in csv.DictReader(fh)
                  if (r.get("AgeAsOfEnrollDt") or "").strip().isdigit() and int(r["AgeAsOfEnrollDt"]) >= min_age}
    rows: dict[str, list[tuple[int, dict]]] = {}
    with (root / "NonDiabDeviceCGM.csv").open(newline="", encoding="utf-8-sig") as fh:
        for k, row in enumerate(csv.DictReader(fh), start=2):
            pt = row["PtID"].strip()
            if pt in adults and row["RecordType"].strip() == "CGM":
                rows.setdefault(pt, []).append((k, row))
    base = datetime(2017, 1, 1)
    out = {}
    for n, pt in enumerate(sorted(rows, key=int), start=1):
        patient = checked_patient(f"cgmnd-{n:03d}")
        res = AdapterResult()
        for k, row in rows[pt]:
            v = float(row["Value"])
            v = 40.0 if v <= 39 else 400.0 if v >= 401 else v
            hh, mm, ss = (int(p) for p in row["DeviceTm"].split(":"))
            ts = base + timedelta(days=int(row["DeviceDtDaysFromEnroll"]), hours=hh, minutes=mm, seconds=ss)
            res.raws.append(RawMeasurement(patient=patient, kind=Kind.GLUCOSE_CGM, ts=ts, value=str(v), unit="mg/dL",
                                           source="cgmnd", device="dexcom-g6", raw_ref=f"NonDiabDeviceCGM.csv:{k}"))
        out[patient] = res
    return out
