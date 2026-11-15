"""Format adapters: each turns one source format into RawMeasurements + Quarantined.

Adapters only handle *structure* (find the date, the time, the value, the unit).
Numeric validity, units, time zones and duplicates are normalize()'s job, so a
garbage value like "abc" passes through here and is quarantined there.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from sources.apple_health import parse_apple_health
from sources.cgmacros import parse_cgmacros
from sources.fitbit import parse_fitbit
from sources.model import Kind
from sources.normalize import normalize

FX = Path(__file__).parent / "fixtures"


def _load(name):
    return json.loads((FX / name).read_text())


# ---------------------------------------------------------------- fitbit
def test_fitbit_heart_intraday():
    res = parse_fitbit(_load("fitbit_heart.json"), patient="p-07", ref="fitbit_heart.json")
    assert [r.kind for r in res.raws] == [Kind.HEART_RATE] * 3
    assert res.raws[0].ts == datetime(2024, 1, 15, 7, 0)          # naive: Fitbit times carry no offset
    assert res.raws[0].unit == "/min" and res.raws[0].device == "fitbit"
    assert [q.reason for q in res.quarantined] == ["malformed_record"]  # the row with no "time"


def test_fitbit_steps_intraday():
    res = parse_fitbit(_load("fitbit_steps.json"), patient="p-07", ref="s.json")
    assert [(r.kind, r.value) for r in res.raws] == [(Kind.STEPS, 0), (Kind.STEPS, 42)]


def test_fitbit_without_tz_is_quarantined_by_normalize():
    res = parse_fitbit(_load("fitbit_heart.json"), patient="p-07", ref="h")
    n = normalize(res.raws)
    assert not n.measurements
    assert {q.reason for q in n.quarantined} == {"missing_timezone", "non_numeric"}  # "abc" fails first


def test_fitbit_with_tz_end_to_end():
    res = parse_fitbit(_load("fitbit_heart.json"), patient="p-07", ref="h")
    n = normalize(res.raws, tz="America/Chicago")
    assert [m.value for m in n.measurements] == [64, 66]
    assert n.measurements[0].ts == datetime(2024, 1, 15, 13, 0, tzinfo=timezone.utc)
    assert [q.reason for q in n.quarantined] == ["non_numeric"]   # the "abc" row


def test_fitbit_document_with_no_date_is_rejected_whole():
    res = parse_fitbit({"activities-heart-intraday": {"dataset": [{"time": "07:00:00", "value": 60}]}},
                       patient="p-07", ref="x")
    assert not res.raws and [q.reason for q in res.quarantined] == ["malformed_record"]


def test_patient_id_is_validated():
    with pytest.raises(ValueError):
        parse_fitbit(_load("fitbit_heart.json"), patient="john.doe@example.com", ref="h")


# ---------------------------------------------------------------- apple health
def test_apple_health_records():
    res = parse_apple_health(FX / "apple_export.xml", patient="p-07")
    kinds = [(r.kind, r.device) for r in res.raws]
    assert kinds[:4] == [(Kind.GLUCOSE_CGM, "dexcom"), (Kind.GLUCOSE_BLOOD, "contour"),
                         (Kind.HEART_RATE, "apple-watch"), (Kind.STEPS, "iphone")]
    assert kinds[4][0] is Kind.BODY_WEIGHT and kinds[4][1].startswith("other-")   # "Scale" -> opaque id
    assert res.raws[0].ts.utcoffset() == timedelta(hours=-5)        # offset preserved from the export
    assert res.stats["skipped_unsupported_type"] == 0               # BodyMass is now ingested (body weight)
    assert [q.reason for q in res.quarantined] == ["malformed_record"]


def test_apple_health_mmol_normalized():
    n = normalize(parse_apple_health(FX / "apple_export.xml", patient="p-07").raws)
    blood = next(m for m in n.measurements if m.kind is Kind.GLUCOSE_BLOOD)
    assert blood.value == pytest.approx(6.2 * 18.016)
    assert blood.ts == datetime(2024, 1, 15, 17, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize("name", ["apple_xxe.xml", "apple_billion_laughs.xml"])
def test_apple_health_entity_attacks_are_refused(name):
    res = parse_apple_health(FX / name, patient="p-07")
    assert not res.raws
    assert [q.reason for q in res.quarantined] == ["unsafe_xml"]


def test_apple_health_truncated_file_keeps_what_parsed(tmp_path):
    text = (FX / "apple_export.xml").read_text()
    cut = text[: text.index("<Record type=\"HKQuantityTypeIdentifierHeartRate\"")]
    (tmp_path / "cut.xml").write_text(cut)
    res = parse_apple_health(tmp_path / "cut.xml", patient="p-07")
    assert len(res.raws) == 2
    assert res.quarantined[-1].reason == "malformed_xml"


# ---------------------------------------------------------------- cgmacros
def test_cgmacros_rows():
    res = parse_cgmacros(FX / "CGMacros-001.csv")
    assert res.raws[0].patient == "cgmacros-001"                    # from the file name
    by_kind = {}
    for r in res.raws:
        by_kind.setdefault(r.kind, []).append(r)
    assert {r.device for r in by_kind[Kind.GLUCOSE_CGM]} == {"libre-pro", "dexcom-g6"}
    assert len(by_kind[Kind.GLUCOSE_CGM]) == 4                       # 98, 101, 104, "abc"
    assert [r.value for r in by_kind[Kind.MEAL_CARBS]] == ["45"]          # raw cell; normalize() converts
    assert by_kind[Kind.METS][0].unit == "MET*10"
    assert len(res.raws) == 23            # + meal type, meal kcal and % eaten on the breakfast row
    assert [q.reason for q in res.quarantined] == ["malformed_record"]   # the "garbage" timestamp


def test_cgmacros_two_sensors_are_separate_streams_end_to_end():
    n = normalize(parse_cgmacros(FX / "CGMacros-001.csv").raws, tz="UTC")
    glucose = [(m.device, m.value) for m in n.measurements if m.kind is Kind.GLUCOSE_CGM]
    assert ("libre-pro", 98.0) in glucose and ("dexcom-g6", 101.0) in glucose
    mets = [m.value for m in n.measurements if m.kind is Kind.METS]
    assert mets[0] == pytest.approx(1.0)
    assert {q.reason for q in n.quarantined} == {"non_numeric"}


def test_cgmacros_missing_timestamp_column_raises(tmp_path):
    (tmp_path / "CGMacros-002.csv").write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        parse_cgmacros(tmp_path / "CGMacros-002.csv")


# ---------------------------------------------------------------- review fixes
def test_fitbit_non_object_document_is_malformed():
    res = parse_fitbit([{"time": "07:00:00"}], patient="p-07", ref="x")
    assert not res.raws and [q.reason for q in res.quarantined] == ["malformed_record"]


def test_apple_source_names_never_reach_output_as_free_text():
    # Apple's default source name is "<Owner Name>'s iPhone": map to a closed set of device types.
    from sources.apple_health import device_for
    assert device_for("John Smith's iPhone 555-123-4567") == "iphone"
    assert device_for("Jane's Apple Watch") == "apple-watch"
    assert device_for("Dexcom G7") == "dexcom" and device_for("FreeStyle LibreLink") == "libre"
    other = device_for("John Smith Health Sync")
    assert other.startswith("other-") and "john" not in other and "smith" not in other
    assert device_for("Sync A") != device_for("Sync B")


def test_cgmacros_bom_header_is_read(tmp_path):
    p = tmp_path / "CGMacros-003.csv"
    p.write_bytes("\ufeffTimestamp,Dexcom GL\n5/1/2023 7:45,101\n".encode("utf-8"))
    assert len(parse_cgmacros(p).raws) == 1


def test_cgmacros_counts_empty_cells_and_rejects_files_with_no_known_columns(tmp_path):
    res = parse_cgmacros(FX / "CGMacros-001.csv")
    assert res.stats["skipped_empty_cell"] > 0
    p = tmp_path / "CGMacros-004.csv"
    p.write_text("Timestamp,Libre_GL\n5/1/2023 7:45,101\n")      # renamed header -> nothing recognized
    with pytest.raises(ValueError):
        parse_cgmacros(p)


# ---------------------------------------------------------------- security review fixes
def test_cgmacros_default_patient_only_from_dataset_file_names(tmp_path):
    p = tmp_path / "Smith_John.csv"
    p.write_text("Timestamp,Dexcom GL\n5/1/2023 7:45,101\n")
    with pytest.raises(ValueError):
        parse_cgmacros(p)                                   # name-like stem: must pass --patient
    assert parse_cgmacros(p, patient="p-09").raws[0].patient == "p-09"


def test_cgmacros_oversized_cell_quarantines_row_not_file(tmp_path):
    p = tmp_path / "CGMacros-005.csv"
    p.write_text("Timestamp,Dexcom GL\n5/1/2023 7:45,101\n5/1/2023 7:50," + "9" * 200_000 + "\n5/1/2023 7:55,103\n")
    res = parse_cgmacros(p)
    assert [r.value for r in res.raws] == ["101", "103"]
    assert [q.reason for q in res.quarantined] == ["malformed_record"]


def test_fitbit_section_of_wrong_shape_is_malformed_not_a_crash():
    doc = {"activities-heart": [{"dateTime": "2024-01-15"}], "activities-heart-intraday": ["not", "a", "dict"]}
    res = parse_fitbit(doc, patient="p-07", ref="x")
    assert not res.raws and [q.reason for q in res.quarantined] == ["malformed_record"]


def test_quarantine_detail_is_clipped_and_whitelisted():
    doc = {"activities-heart": [{"dateTime": "2024-01-15"}],
           "activities-heart-intraday": {"dataset": [{"value": "x" * 10_000, "name": "John Smith"}]}}
    q = parse_fitbit(doc, patient="p-07", ref="x").quarantined[0]
    assert "name" not in q.detail.get("point", {}) and len(str(q.detail)) < 300


def test_apple_memory_bounded_root_is_cleared(tmp_path):
    body = "".join("<Correlation><Foo/></Correlation>" for _ in range(2000))
    p = tmp_path / "big.xml"
    p.write_text(f'<HealthData locale="en_US">{body}<Record type="HKQuantityTypeIdentifierHeartRate" '
                 f'sourceName="Apple Watch" unit="count/min" startDate="2024-01-15 10:00:00 -0500" value="72"/></HealthData>')
    from sources import apple_health
    res = apple_health.parse_apple_health(p, patient="p-07")
    assert len(res.raws) == 1 and apple_health.LAST_ROOT_CHILDREN <= 1


def test_patient_id_with_trailing_newline_rejected():
    with pytest.raises(ValueError):
        parse_fitbit(_load("fitbit_heart.json"), patient="p-07\n", ref="h")


# ---------------------------------------------------------------- more measurement types
def test_apple_more_types():
    res = parse_apple_health(FX / "apple_more.xml", patient="p-07")
    n = normalize(res.raws)
    got = {m.kind: m for m in n.measurements}
    assert got[Kind.SPO2].value == pytest.approx(97.0)            # Apple stores SpO2 as a 0-1 fraction with unit "%"
    assert got[Kind.HRV_SDNN].value == pytest.approx(48.2)
    assert got[Kind.RESPIRATORY_RATE].value == 14
    assert (got[Kind.BP_SYSTOLIC].value, got[Kind.BP_DIASTOLIC].value) == (121, 79)   # nested in a Correlation
    assert got[Kind.INSULIN_BOLUS].value == 4.5 and got[Kind.INSULIN_BASAL].value == 0.8
    assert got[Kind.SLEEP_STAGE].value == "deep"
    assert (got[Kind.SLEEP_STAGE].end - got[Kind.SLEEP_STAGE].ts).total_seconds() == 2400
    assert [q.reason for q in res.quarantined] == ["unknown_code"]   # insulin with no delivery reason
    assert not n.quarantined


def test_fitbit_sleep_stages():
    res = parse_fitbit(_load("fitbit_sleep.json"), patient="p-07", ref="sleep.json")
    n = normalize(res.raws, tz="UTC")
    assert [(m.value, (m.end - m.ts).total_seconds()) for m in n.measurements] == [("awake", 30), ("light", 1200)]
    assert sorted(q.reason for q in n.quarantined + res.quarantined) == ["invalid_period", "malformed_record"]
    assert res.stats["skipped_short_data"] == 1


def test_fitbit_spo2_intraday():
    n = normalize(parse_fitbit(_load("fitbit_spo2.json"), patient="p-07", ref="spo2.json").raws, tz="UTC")
    assert [m.value for m in n.measurements] == [95.3, 96.1] and n.measurements[0].kind is Kind.SPO2
