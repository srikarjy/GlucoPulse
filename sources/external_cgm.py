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
