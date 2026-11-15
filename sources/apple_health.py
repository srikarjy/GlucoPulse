"""Apple Health export.xml (streamed; these files are often gigabytes).

Parsed with defusedxml: the export is untrusted input, so entity declarations
and external references (XXE, billion laughs) are refused. An unsafe file yields
no data at all (one "unsafe_xml" quarantine record), never a partial parse.
A truncated or otherwise malformed file keeps what parsed before the error.

Memory stays bounded: the root is cleared after every top-level child, so
non-Record elements (Workout, Correlation, ActivitySummary, ...) don't accumulate.

Privacy: Apple's default sourceName is "<Owner Name>'s iPhone". The free-text
name never leaves this module; it is mapped to a closed set of device types, or
to an opaque "other-<hash>" stream id.
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from xml.etree.ElementTree import ParseError

from defusedxml import DefusedXmlException
from defusedxml.ElementTree import iterparse

from sources._common import AdapterResult, checked_patient, clip
from sources.model import Kind, Quarantined, RawMeasurement

TYPES = {
    "HKQuantityTypeIdentifierBloodGlucose": Kind.GLUCOSE_BLOOD,
    "HKQuantityTypeIdentifierHeartRate": Kind.HEART_RATE,
    "HKQuantityTypeIdentifierStepCount": Kind.STEPS,
    "HKQuantityTypeIdentifierOxygenSaturation": Kind.SPO2,
    "HKQuantityTypeIdentifierHeartRateVariabilitySDNN": Kind.HRV_SDNN,
    "HKQuantityTypeIdentifierRespiratoryRate": Kind.RESPIRATORY_RATE,
    "HKQuantityTypeIdentifierBodyMass": Kind.BODY_WEIGHT,
    "HKQuantityTypeIdentifierBloodPressureSystolic": Kind.BP_SYSTOLIC,
    "HKQuantityTypeIdentifierBloodPressureDiastolic": Kind.BP_DIASTOLIC,
    "HKQuantityTypeIdentifierInsulinDelivery": Kind.INSULIN_BOLUS,   # refined by HKInsulinDeliveryReason
    "HKCategoryTypeIdentifierSleepAnalysis": Kind.SLEEP_STAGE,
}
INSULIN_REASON = {"1": Kind.INSULIN_BASAL, "2": Kind.INSULIN_BOLUS}   # HKInsulinDeliveryReason
# token in the (lower-cased) source name -> device type. Order matters: first match wins.
DEVICE_TOKENS = (
    ("dexcom", "dexcom"), ("libre", "libre"), ("guardian", "guardian"), ("eversense", "eversense"),
    ("contour", "contour"), ("accu-chek", "accu-chek"), ("accuchek", "accu-chek"), ("onetouch", "onetouch"),
    ("watch", "apple-watch"), ("iphone", "iphone"), ("ipad", "ipad"),
)
CGM_DEVICES = {"dexcom", "libre", "guardian", "eversense"}
LAST_ROOT_CHILDREN = 0   # exposed for the memory-bound test


def device_for(source_name: str) -> str:
    low = source_name.lower()
    for token, device in DEVICE_TOKENS:
        if token in low:
            return device
    return "other-" + hashlib.sha256(source_name.encode("utf-8", "replace")).hexdigest()[:10]


def parse_apple_health(path: str | Path, patient: str) -> AdapterResult:
    global LAST_ROOT_CHILDREN
    patient = checked_patient(patient)
    path = Path(path)
    out = AdapterResult()
    n, depth, root = 0, 0, None
    try:
        for event, elem in iterparse(str(path), events=("start", "end")):
            if event == "start":
                if root is None:
                    root = elem
                depth += 1
                continue
            depth -= 1
            if elem.tag == "Record":
                n += 1
                _record(elem, f"{path.name}#record{n}", patient, out)
            if depth == 1 and root is not None:       # a top-level child just ended
                root.clear()
        LAST_ROOT_CHILDREN = len(root) if root is not None else 0
    except DefusedXmlException as exc:
        return AdapterResult(quarantined=[Quarantined("unsafe_xml", path.name, "apple_health",
                                                      {"why": type(exc).__name__})])
    except ParseError as exc:
        out.quarantined.append(Quarantined("malformed_xml", path.name, "apple_health", {"why": clip(str(exc))}))
    return out


def _record(elem, rref: str, patient: str, out: AdapterResult) -> None:
    kind = TYPES.get(elem.get("type", ""))
    if kind is None:
        out.stats["skipped_unsupported_type"] += 1
        return
    device = device_for(elem.get("sourceName", ""))
    if kind is Kind.GLUCOSE_BLOOD and device in CGM_DEVICES:
        kind = Kind.GLUCOSE_CGM
    if kind is Kind.INSULIN_BOLUS:
        reason = next((m.get("value") for m in elem.iter("MetadataEntry")
                       if m.get("key") == "HKInsulinDeliveryReason"), None)
        if reason not in INSULIN_REASON:
            out.quarantined.append(Quarantined("unknown_code", rref, "apple_health",
                                               {"type": "insulin delivery", "HKInsulinDeliveryReason": clip(reason)}))
            return
        kind = INSULIN_REASON[reason]
    fmt = "%Y-%m-%d %H:%M:%S %z"
    try:
        ts = datetime.strptime(elem.get("startDate", ""), fmt)
        end = datetime.strptime(elem.get("endDate", ""), fmt) if kind is Kind.SLEEP_STAGE else None
        value = elem.attrib["value"]
        unit = "" if kind is Kind.SLEEP_STAGE else elem.attrib["unit"]
    except (KeyError, ValueError):
        out.quarantined.append(Quarantined("malformed_record", rref, "apple_health",
                                           {"type": clip(elem.get("type")), "startDate": clip(elem.get("startDate"))}))
        return
    if kind is Kind.SPO2 and unit == "%":
        # Apple writes oxygen saturation as a 0-1 fraction under the unit "%" (e.g. value="0.97").
        try:
            if float(value) <= 1.0:
                unit = "fraction"
        except ValueError:
            pass                       # normalize() quarantines it as non_numeric
    out.raws.append(RawMeasurement(patient=patient, kind=kind, ts=ts, value=value, unit=unit,
                                   source="apple_health", device=device, raw_ref=rref, end=end))
