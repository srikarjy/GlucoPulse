"""Normalization: units/labels -> canonical, time -> UTC, plausibility, dedup, conflicts.

Rules (pinned by tests/sources/test_normalize.py and test_types.py):
  * Quantities must be plain ASCII decimals ("72", "-0.5", ".5") or real ints/floats.
    Booleans, exponents, underscores, non-ASCII digits and huge values are
    quarantined ("non_numeric"), never coerced. Unknown units -> "unsupported_unit".
  * Codes are mapped through the kind's synonym table after normalizing case,
    whitespace, a trailing ordinal ("snack 1") and float-looking labels ("2.0").
    Anything unmapped -> "unknown_code", never guessed.
  * A naive timestamp is localized with the caller's `tz`; with no `tz` it is
    quarantined ("missing_timezone"). Never assume UTC: that silently shifts meals.
    Local times that do not exist (spring-forward gap) or occur twice (fall-back
    overlap) are quarantined ("nonexistent_local_time" / "ambiguous_local_time"), and so are
    times outside 1971-2100 or that overflow on conversion ("invalid_timestamp").
  * Period kinds (e.g. sleep stages) need an end after the start and no longer
    than the kind's maximum ("missing_end" / "invalid_period").
  * Duplicates/conflicts are judged per stream (patient, kind, device). A reading
    is "the same sample" as the first reading of the current cluster if it is
    within HALF the device's native sampling interval (fitbit 30 s, dexcom 150 s,
    libre 450 s); devices with no known interval only match on an identical time.
    This is phase-independent: fixed clock-aligned slots would split one sample
    interval in two, or merge two genuinely consecutive readings.
      - all values equal (codes) / within DUP_TOLERANCE (quantities) -> keep the first
      - otherwise -> quarantine the WHOLE cluster ("conflicting_values"); never pick one.
    Two different devices (e.g. Dexcom vs Libre) are separate streams, so their
    disagreement is not a conflict here; it is a data-quality signal for later.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from serving.healthcare.fhir_mapping import MGDL_PER_MMOLL
from sources._common import clip
from sources.model import DEVICE_SLOT_SECONDS, KINDS, Kind, KindSpec, Measurement, Quarantined, RawMeasurement

DUP_TOLERANCE = 0.1
# Plausible time window for any health measurement. Rejects overflow-prone years (0001, 9999),
# the 1900 dates a time-only strptime format produces, and epoch 0/-1 placeholders.
EARLIEST = datetime(1971, 1, 1, tzinfo=timezone.utc)
LATEST = datetime(2100, 1, 1, tzinfo=timezone.utc)
_DECIMAL = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", re.ASCII)

LB_TO_KG = 0.45359237
_GLUCOSE = {"mg/dl": 1.0, "mmol/l": MGDL_PER_MMOLL, "mmol<180.1558800000541>/l": MGDL_PER_MMOLL}
_PER_MIN = {"/min": 1.0, "count/min": 1.0, "bpm": 1.0, "beats/min": 1.0, "breaths/min": 1.0}
_KCAL = {"kcal": 1.0, "Cal": 1.0}     # "Cal" (capital C) is a kilocalorie; lowercase "cal" is not accepted
_MMHG = {"mm[hg]": 1.0, "mmhg": 1.0}
_IU = {"[iu]": 1.0, "iu": 1.0, "u": 1.0}
_GRAM = {"g": 1.0}
UNIT_FACTORS: dict[Kind, dict[str, float]] = {
    Kind.GLUCOSE_CGM: _GLUCOSE, Kind.GLUCOSE_BLOOD: _GLUCOSE,
    Kind.HEART_RATE: _PER_MIN, Kind.RESPIRATORY_RATE: _PER_MIN,
    Kind.HRV_SDNN: {"ms": 1.0},
    Kind.SPO2: {"%": 1.0, "fraction": 100.0},
    Kind.BP_SYSTOLIC: _MMHG, Kind.BP_DIASTOLIC: _MMHG,
    Kind.BODY_WEIGHT: {"kg": 1.0, "g": 0.001, "lb": LB_TO_KG, "[lb_av]": LB_TO_KG},
    Kind.STEPS: {"{steps}": 1.0, "count": 1.0, "steps": 1.0},
    Kind.ACTIVE_CALORIES: _KCAL, Kind.MEAL_CALORIES: _KCAL,
    Kind.METS: {"{met}": 1.0, "met": 1.0, "met*10": 0.1},
    Kind.INSULIN_BOLUS: _IU, Kind.INSULIN_BASAL: _IU,
    Kind.MEAL_CARBS: _GRAM, Kind.MEAL_PROTEIN: _GRAM, Kind.MEAL_FAT: _GRAM, Kind.MEAL_FIBER: _GRAM,
    Kind.MEAL_SUGAR: _GRAM,
    Kind.MEAL_PORTION_EATEN: {"%": 1.0},
}


@dataclass
class NormalizeResult:
    measurements: list[Measurement] = field(default_factory=list)
    quarantined: list[Quarantined] = field(default_factory=list)
    stats: Counter = field(default_factory=Counter)


def _to_number(v) -> Optional[float]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        try:
            return float(v)
        except OverflowError:
            return None
    if isinstance(v, str):
        s = v.strip()
        return float(s) if _DECIMAL.fullmatch(s) else None
    return None


def _to_code(spec: KindSpec, v) -> Optional[str]:
    if isinstance(v, bool) or not isinstance(v, (str, int, float)):
        return None
    s = re.sub(r"\s+", " ", str(v).strip().lower())
    if re.fullmatch(r"\d+\.0+", s):          # "2.0" -> "2"
        s = s.split(".")[0]
    candidates = [s]
    ordinal = re.fullmatch(r"(.*\D)\s+\d+", s)      # "snack 1" -> "snack", but never "1 2" -> "1"
    if ordinal:
        candidates.append(ordinal.group(1).strip())
    for candidate in candidates:
        if candidate in spec.synonyms:
            return spec.synonyms[candidate]
    return None


def _unit_factor(kind: Kind, unit: str) -> Optional[float]:
    table = UNIT_FACTORS[kind]
    if unit in table:
        return table[unit]
    return table.get(unit.strip().lower())


def _to_utc(ts: datetime, zone: Optional[ZoneInfo]) -> tuple[Optional[datetime], Optional[str]]:
    try:
        if ts.tzinfo is None:
            if zone is None:
                return None, "missing_timezone"
            first, second = ts.replace(tzinfo=zone, fold=0), ts.replace(tzinfo=zone, fold=1)
            if first.utcoffset() != second.utcoffset():
                round_trip = first.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None)
                return None, "nonexistent_local_time" if round_trip != ts else "ambiguous_local_time"
            ts = first
        utc = ts.astimezone(timezone.utc)
    except OverflowError:
        return None, "invalid_timestamp"
    if not (EARLIEST <= utc < LATEST):
        return None, "invalid_timestamp"
    return utc, None


def _q(r: RawMeasurement, reason: str, **detail) -> Quarantined:
    return Quarantined(reason=reason, raw_ref=r.raw_ref, source=r.source,
                       detail={"kind": r.kind.value, "device": r.device, "value": clip(r.value),
                               "unit": clip(r.unit), **detail})


def _normalize_one(r: RawMeasurement, zone: Optional[ZoneInfo]) -> Measurement | Quarantined:
    spec = KINDS[r.kind]
    if spec.value_type == "code":
        value = _to_code(spec, r.value)
        if value is None:
            return _q(r, "unknown_code")
    else:
        number = _to_number(r.value)
        if number is None or not math.isfinite(number):
            return _q(r, "non_numeric")
        factor = _unit_factor(r.kind, r.unit)
        if factor is None:
            return _q(r, "unsupported_unit")
        value = number * factor
    ts, why = _to_utc(r.ts, zone)
    if ts is None:
        return _q(r, why)
    end = None
    if spec.period:
        if r.end is None:
            return _q(r, "missing_end")
        end, why = _to_utc(r.end, zone)
        if end is None:
            return _q(r, why)
        if not (ts < end and end - ts <= timedelta(hours=spec.max_period_hours)):
            return _q(r, "invalid_period", end=end.isoformat())
    if spec.value_type == "quantity" and not (spec.low <= value <= spec.high):
        return _q(r, "implausible_value", normalized=round(value, 3))
    return Measurement(patient=r.patient, kind=r.kind, ts=ts, value=value, source=r.source,
                       device=r.device, raw_ref=r.raw_ref, end=end)


def normalize(raws: Iterable[RawMeasurement], tz: Optional[str] = None) -> NormalizeResult:
    try:
        zone = ZoneInfo(tz) if tz else None
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"unknown time zone: {tz!r}") from exc

    out = NormalizeResult()
    streams: dict[tuple, list[Measurement]] = defaultdict(list)
    for r in raws:
        out.stats["input"] += 1
        result = _normalize_one(r, zone)
        if isinstance(result, Quarantined):
            out.quarantined.append(result)
        else:
            streams[(result.patient, result.kind, result.device)].append(result)

    for (_patient, kind, device), stream in streams.items():
        # The sampling window is for point-in-time readings. Period records (sleep stages) are
        # contiguous segments that can start seconds apart; they only match on an identical start.
        window = 0 if KINDS[kind].period else DEVICE_SLOT_SECONDS.get(device, 0) / 2
        stream.sort(key=lambda m: m.ts)          # stable: ties keep input order
        cluster: list[Measurement] = []
        for m in stream + [None]:                # sentinel flushes the last cluster
            if m is not None and cluster and (m.ts - cluster[0].ts).total_seconds() <= window:
                cluster.append(m)
                continue
            if cluster:
                _flush(cluster, out)
            cluster = [m] if m is not None else []

    out.measurements.sort(key=lambda m: (m.patient, m.kind.value, m.device, m.ts))
    out.stats["accepted"] = len(out.measurements)
    out.stats["quarantined"] = len(out.quarantined)
    return out


def _same(a: Measurement, b: Measurement) -> bool:
    if isinstance(a.value, str) or isinstance(b.value, str):
        return a.value == b.value and a.end == b.end
    return abs(a.value - b.value) <= DUP_TOLERANCE and a.end == b.end


def _flush(cluster: list[Measurement], out: NormalizeResult) -> None:
    first = cluster[0]
    if all(_same(first, m) for m in cluster):
        out.measurements.append(first)
        out.stats["duplicates_dropped"] += len(cluster) - 1
        return
    values = [m.value if isinstance(m.value, str) else round(m.value, 3) for m in cluster]
    for m in cluster:
        out.quarantined.append(Quarantined(
            reason="conflicting_values", raw_ref=m.raw_ref, source=m.source,
            detail={"kind": m.kind.value, "device": m.device, "cluster_values": values[:20],
                    "ts": m.ts.isoformat()}))
