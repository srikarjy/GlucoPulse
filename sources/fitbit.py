"""Fitbit Web API JSON: intraday heart rate and steps, sleep logs (v1.2 stages), SpO2 intraday.

Fitbit intraday times are "HH:MM:SS" in the user's profile time zone with no
offset, so timestamps come out naive and normalize() needs an explicit tz.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from sources._common import AdapterResult, checked_patient, clip
from sources.model import Kind, Quarantined, RawMeasurement

SECTIONS = {
    "activities-heart-intraday": ("activities-heart", Kind.HEART_RATE, "/min"),
    "activities-steps-intraday": ("activities-steps", Kind.STEPS, "{steps}"),
}


def _date(doc: dict, summary_key: str):
    try:
        return datetime.strptime(doc[summary_key][0]["dateTime"], "%Y-%m-%d").date()
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def parse_fitbit(doc: Any, patient: str, ref: str) -> AdapterResult:
    patient = checked_patient(patient)
    out = AdapterResult()
    if not isinstance(doc, dict):
        out.quarantined.append(Quarantined("malformed_record", ref, "fitbit", {"why": "top level is not an object"}))
        return out
    for intraday_key, (summary_key, kind, unit) in SECTIONS.items():
        if intraday_key not in doc:
            continue
        day = _date(doc, summary_key)
        section = doc[intraday_key]
        dataset = section.get("dataset") if isinstance(section, dict) else None
        if day is None or not isinstance(dataset, list):
            out.quarantined.append(Quarantined("malformed_record", f"{ref}#{intraday_key}", "fitbit",
                                               {"why": f"missing {summary_key}[0].dateTime or dataset"}))
            continue
        for i, point in enumerate(dataset):
            rref = f"{ref}#{intraday_key}[{i}]"
            try:
                t = datetime.strptime(point["time"], "%H:%M:%S").time()
                value = point["value"]
            except (KeyError, TypeError, ValueError):
                shown = {k: clip(point[k]) for k in ("time", "value") if isinstance(point, dict) and k in point}
                out.quarantined.append(Quarantined("malformed_record", rref, "fitbit", {"point": shown}))
                continue
            out.raws.append(RawMeasurement(patient=patient, kind=kind, ts=datetime.combine(day, t), value=value,
                                           unit=unit, source="fitbit", device="fitbit", raw_ref=rref))
    if "sleep" in doc:
        _sleep(doc["sleep"], patient, ref, out)
    if isinstance(doc.get("minutes"), list):
        _spo2(doc["minutes"], patient, ref, out)
    return out


def _sleep(logs: Any, patient: str, ref: str, out: AdapterResult) -> None:
    """Sleep log v1.2: levels.data is the stage timeline. levels.shortData holds short
    wakes that OVERLAP data segments; they are counted, not ingested, so stages never overlap."""
    if not isinstance(logs, list):
        out.quarantined.append(Quarantined("malformed_record", f"{ref}#sleep", "fitbit", {"why": "sleep is not a list"}))
        return
    for li, log in enumerate(logs):
        levels = log.get("levels") if isinstance(log, dict) else None
        data = levels.get("data") if isinstance(levels, dict) else None
        if not isinstance(data, list):
            out.quarantined.append(Quarantined("malformed_record", f"{ref}#sleep[{li}]", "fitbit",
                                               {"why": "missing levels.data"}))
            continue
        out.stats["skipped_short_data"] += len(levels.get("shortData") or [])
        for i, seg in enumerate(data):
            rref = f"{ref}#sleep[{li}].levels.data[{i}]"
            try:
                start = datetime.fromisoformat(seg["dateTime"])
                end = start + timedelta(seconds=int(seg["seconds"]))
                level = seg["level"]
            except (KeyError, TypeError, ValueError, OverflowError):
                shown = {k: clip(seg[k]) for k in ("dateTime", "level", "seconds") if isinstance(seg, dict) and k in seg}
                out.quarantined.append(Quarantined("malformed_record", rref, "fitbit", {"segment": shown}))
                continue
            out.raws.append(RawMeasurement(patient=patient, kind=Kind.SLEEP_STAGE, ts=start, value=level, unit="",
                                           source="fitbit", device="fitbit", raw_ref=rref, end=end))


def _spo2(minutes: list, patient: str, ref: str, out: AdapterResult) -> None:
    for i, point in enumerate(minutes):
        rref = f"{ref}#minutes[{i}]"
        try:
            ts = datetime.fromisoformat(point["minute"])
            value = point["value"]
        except (KeyError, TypeError, ValueError):
            shown = {k: clip(point[k]) for k in ("minute", "value") if isinstance(point, dict) and k in point}
            out.quarantined.append(Quarantined("malformed_record", rref, "fitbit", {"point": shown}))
            continue
        out.raws.append(RawMeasurement(patient=patient, kind=Kind.SPO2, ts=ts, value=value, unit="%",
                                       source="fitbit", device="fitbit", raw_ref=rref))
