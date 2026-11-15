"""CGMacros per-participant CSV (PhysioNet, CC BY-NC-SA 4.0).

One row per minute: two CGMs (Libre Pro, native every 15 min; Dexcom G6, native
every 5 min), Fitbit Sense heart rate / activity calories / METs (stored x10) /
steps / intensity (0-3), and meal rows (type, kcal, carbs, protein, fat, fiber,
sugar, % eaten). Header layouts vary between participants (9 variants in v1.0.0,
e.g. "Steps" or "Intensity" columns, a "Sugar" column, a trailing space in
"Amount Consumed "), so columns are matched by normalized name, not position.

Glucose: the published glucose columns are LINEARLY INTERPOLATED to one value
per minute (verified on all 45 participants: second differences are zero between
readings, and the remaining "knots" are spaced at exactly 15 min for Libre and
5 min for Dexcom). Interpolated values are not measurements, so
`recover_native_readings` keeps only the original sensor readings (see below).

Timestamps are randomly shifted by the dataset authors and carry no time zone,
so normalize() needs an explicit tz. Meal macros are recorded exactly as given;
whether they are already scaled by "Amount Consumed" is not stated in the data
dictionary, so "% eaten" is kept as its own measurement and nothing is rescaled.

The patient id defaults to the file stem ONLY for dataset-style names
("CGMacros-001.csv"); any other file name must pass an explicit pseudonymous id,
so a file named after a person never becomes a Patient reference.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from sources._common import AdapterResult, checked_patient, clip
from sources.model import Kind, Quarantined, RawMeasurement

# normalized header -> (kind, unit, device)
COLUMNS = {
    "libre gl": (Kind.GLUCOSE_CGM, "mg/dL", "libre-pro"),
    "dexcom gl": (Kind.GLUCOSE_CGM, "mg/dL", "dexcom-g6"),
    "hr": (Kind.HEART_RATE, "/min", "fitbit"),
    "calories (activity)": (Kind.ACTIVE_CALORIES, "kcal", "fitbit"),
    "mets": (Kind.METS, "MET*10", "fitbit"),
    "steps": (Kind.STEPS, "{steps}", "fitbit"),
    "intensity": (Kind.ACTIVITY_INTENSITY, "", "fitbit"),
    "meal type": (Kind.MEAL_TYPE, "", "meal-log"),
    "calories": (Kind.MEAL_CALORIES, "kcal", "meal-log"),
    "carbs": (Kind.MEAL_CARBS, "g", "meal-log"),
    "protein": (Kind.MEAL_PROTEIN, "g", "meal-log"),
    "fat": (Kind.MEAL_FAT, "g", "meal-log"),
    "fiber": (Kind.MEAL_FIBER, "g", "meal-log"),
    "sugar": (Kind.MEAL_SUGAR, "g", "meal-log"),
    "amount consumed": (Kind.MEAL_PORTION_EATEN, "%", "meal-log"),
}
IGNORED = {"", "unnamed: 0", "timestamp", "image path", "recordindex"}
NATIVE_MINUTES = {"libre-pro": 15, "dexcom-g6": 5}
TS_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%m/%d/%Y %H:%M", "%m/%d/%Y %H:%M:%S")
DATASET_STEM = re.compile(r"cgmacros-\d{3}", re.IGNORECASE)
KNOT_EPS = 1e-6

Point = tuple[datetime, str, str]   # (timestamp, raw cell text, raw_ref)


def _parse_ts(text: str) -> Optional[datetime]:
    for fmt in TS_FORMATS:
        try:
            return datetime.strptime(text.strip(), fmt)
        except ValueError:
            continue
    return None


@dataclass
class RecoveryStats:
    knots: int = 0        # slope changes + run endpoints: certainly original readings
    refilled: int = 0     # collinear readings restored at exact native multiples between knots


def _num(text: str) -> Optional[float]:
    from sources.normalize import _to_number   # same strict rule as normalize(): plain decimals only
    value = _to_number(text)
    return value if value is not None and value == value and abs(value) != float("inf") else None


def recover_native_readings(points: list[Point], interval_min: int,
                            stats: Optional[RecoveryStats] = None) -> list[Point]:
    """Undo per-minute linear interpolation and return only the original readings.

    Within a run of points exactly one minute apart, an interpolated point lies on
    the straight line through its neighbours (second difference ~0). The original
    readings are therefore the run's endpoints plus every "knot" where the slope
    changes. A reading that happens to sit on a straight line (a flat or constant-
    slope stretch) is not a knot; such readings are restored by taking the series
    value at every native interval between two knots whose gap is an exact multiple
    of the interval. Points off the one-minute grid, or non-numeric, are passed
    through untouched (normalize() decides what to do with them).

    Limit: a collinear value is indistinguishable from a real reading. If the
    sensor dropped a reading inside a straight interpolated stretch, the refill
    step emits the interpolated value at that time. `stats.refilled` counts
    these separately from `stats.knots` so the recovered cadence can be audited.
    """
    kept: list[Point] = []
    run: list[tuple[Point, float]] = []

    def flush() -> None:
        if len(run) < 3:
            kept.extend(p for p, _ in run)
            return
        vals = [v for _, v in run]
        knots = [0] + [i for i in range(1, len(run) - 1)
                       if abs(vals[i - 1] - 2 * vals[i] + vals[i + 1]) > KNOT_EPS] + [len(run) - 1]
        keep = set(knots)
        for a, b in zip(knots, knots[1:]):
            gap = b - a                               # minutes: the run is on a 1-minute grid
            if gap > interval_min and gap % interval_min == 0:
                keep.update(range(a + interval_min, b, interval_min))
        if stats is not None:
            stats.knots += len(set(knots))
            stats.refilled += len(keep) - len(set(knots))
        kept.extend(run[i][0] for i in sorted(keep))

    for p in points:
        value = _num(p[1])
        if value is None:
            flush()
            run = []
            kept.append(p)
            continue
        if run and (p[0] - run[-1][0][0]).total_seconds() != 60:
            flush()
            run = []
        run.append((p, value))
    flush()
    return kept


def parse_cgmacros(path: str | Path, patient: Optional[str] = None) -> AdapterResult:
    path = Path(path)
    if patient is None:
        if not DATASET_STEM.fullmatch(path.stem):
            raise ValueError(f"{path.name}: not a CGMacros-NNN file name; pass an explicit pseudonymous --patient")
        patient = path.stem.lower()
    patient = checked_patient(patient)
    out = AdapterResult()
    glucose: dict[str, list[Point]] = {device: [] for device in NATIVE_MINUTES}
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        header = [h.strip().lower() for h in next(reader, [])]
        if "timestamp" not in header:
            raise ValueError(f"{path.name}: no Timestamp column")
        ts_i = header.index("timestamp")
        cols = [(i, COLUMNS[h]) for i, h in enumerate(header) if h in COLUMNS]
        if not cols:
            raise ValueError(f"{path.name}: no recognized value columns (expected e.g. 'Dexcom GL', 'HR')")
        unrecognized = sorted(h for h in header if h not in COLUMNS and h not in IGNORED)
        out.stats["unrecognized_columns"] = len(unrecognized)
        while True:
            try:
                row = next(reader)
            except StopIteration:
                break
            except csv.Error as exc:                 # e.g. a field larger than csv.field_size_limit()
                out.quarantined.append(Quarantined("malformed_record", f"{path.name}:{reader.line_num}", "cgmacros",
                                                   {"why": clip(str(exc))}))
                continue
            rref = f"{path.name}:{reader.line_num}"
            ts = _parse_ts(row[ts_i]) if ts_i < len(row) else None
            if ts is None:
                out.quarantined.append(Quarantined("malformed_record", rref, "cgmacros",
                                                   {"timestamp": clip(row[ts_i]) if ts_i < len(row) else None}))
                continue
            for i, (kind, unit, device) in cols:
                cell = row[i].strip() if i < len(row) else ""
                if not cell:
                    out.stats["skipped_empty_cell"] += 1
                    continue
                if device in glucose:
                    glucose[device].append((ts, cell, rref))
                    continue
                out.raws.append(RawMeasurement(patient=patient, kind=kind, ts=ts, value=cell, unit=unit,
                                               source="cgmacros", device=device, raw_ref=rref))
    for device, points in glucose.items():
        rs = RecoveryStats()
        readings = recover_native_readings(points, NATIVE_MINUTES[device], rs)
        out.stats["interpolated_dropped"] += len(points) - len(readings)
        out.stats["recovered_knots"] += rs.knots
        out.stats["recovered_refilled"] += rs.refilled
        out.raws.extend(RawMeasurement(patient=patient, kind=Kind.GLUCOSE_CGM, ts=ts, value=cell, unit="mg/dL",
                                       source="cgmacros", device=device, raw_ref=rref) for ts, cell, rref in readings)
    return out
