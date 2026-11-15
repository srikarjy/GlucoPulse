"""Config-driven mapping for any app's CSV/JSON export, and the typed record contract."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from sources.mapped import MappingSpec, load_mapping, parse_mapped
from sources.model import Kind, Measurement
from sources.normalize import normalize
from sources.schema import MeasurementRecord, json_schemas

FX = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).resolve().parents[2]
MAPS = ROOT / "sources" / "mappings"


# ---------------------------------------------------------------- mapped CSV
def test_logbook_csv_mapping():
    spec = load_mapping(MAPS / "example_logbook_csv.toml")
    res = parse_mapped(FX / "app_export.csv", spec, patient="p-07")
    n = normalize(res.raws, tz="Europe/Berlin")
    by = {}
    for m in n.measurements:
        by.setdefault(m.kind, []).append(m)
    assert [round(m.value, 1) for m in by[Kind.GLUCOSE_BLOOD]] == [109.9, 140.5]     # mmol/L -> mg/dL
    assert by[Kind.GLUCOSE_BLOOD][0].ts == datetime(2024, 1, 15, 6, 45, tzinfo=timezone.utc)
    assert [m.value for m in by[Kind.MEAL_TYPE]] == ["breakfast", "lunch", "dinner"]   # per-field label map
    assert [m.value for m in by[Kind.INSULIN_BOLUS]] == [4.5, 6.0, 3.0]
    assert res.raws[0].source == "mapped:example-logbook-csv"
    assert [q.reason for q in res.quarantined] == ["malformed_record"]     # 32.01.2024
    assert res.stats["skipped_empty_cell"] > 0 and res.stats["unmapped_columns"] == 1   # "Note"


def test_wearable_json_mapping_with_dotted_paths_and_epoch():
    spec = load_mapping(MAPS / "example_wearable_json.toml")
    res = parse_mapped(FX / "app_export.json", spec, patient="p-07")
    n = normalize(res.raws)                                   # epoch seconds are UTC: no tz needed
    hr = [m for m in n.measurements if m.kind is Kind.HEART_RATE]
    assert [m.value for m in hr] == [64, 66] and hr[0].ts == datetime(2024, 1, 15, 12, 45, tzinfo=timezone.utc)
    assert [m.value for m in n.measurements if m.kind is Kind.SPO2] == [pytest.approx(97.0)]
    assert [q.reason for q in res.quarantined] == ["malformed_record"]     # ts "bad"


# ---------------------------------------------------------------- mapping validation fails fast
BASE = {"name": "x", "format": "csv", "time": {"columns": ["t"], "format": "%Y-%m-%d"},
        "fields": [{"column": "g", "kind": "glucose_cgm", "unit": "mg/dL"}]}


def _with(**changes):
    spec = json.loads(json.dumps(BASE))
    for path, value in changes.items():
        target = spec
        *parents, leaf = path.split("__")
        for p in parents:
            target = target[p] if not p.isdigit() else target[int(p)]
        target[leaf] = value
    return spec


@pytest.mark.parametrize("bad", [
    _with(format="xml"),
    _with(fields__0__kind="glucose_cgmm"),             # typo in kind
    _with(fields__0__unit="mg/L"),                       # unit not convertible for the kind
    _with(fields__0__device="My Phone!"),                # free text device ids are not allowed
    _with(name="Bad Name"),
    _with(time={"columns": ["t"]}),                      # neither format nor epoch
    _with(time={"columns": ["t"], "format": "%Y", "epoch": "seconds"}),
    _with(colour="red"),                                 # unknown key (typo protection)
])
def test_invalid_mappings_rejected(bad):
    with pytest.raises(ValidationError):
        MappingSpec.model_validate(bad)


def test_code_field_must_not_have_unit_and_label_targets_must_be_valid_codes():
    with pytest.raises(ValidationError):
        MappingSpec.model_validate(_with(fields=[{"column": "m", "kind": "meal_type", "values": {"x": "brunch"}}]))
    with pytest.raises(ValidationError):
        MappingSpec.model_validate(_with(fields=[{"column": "m", "kind": "meal_type", "unit": "g"}]))


def test_json_mapping_requires_records_path():
    with pytest.raises(ValidationError):
        MappingSpec.model_validate(_with(format="json"))


def test_cli_mapped(tmp_path):
    p = subprocess.run([sys.executable, "-m", "sources.ingest", "--format", "mapped", "--mapping",
                        str(MAPS / "example_logbook_csv.toml"), str(FX / "app_export.csv"), "--patient", "p-07",
                        "--tz", "Europe/Berlin", "--out", str(tmp_path / "b.json"),
                        "--records-out", str(tmp_path / "r.ndjson")],
                       cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    records = [json.loads(line) for line in (tmp_path / "r.ndjson").read_text().splitlines()]
    assert records and all(MeasurementRecord.model_validate(r) for r in records)


# ---------------------------------------------------------------- typed contract
TS = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)


def test_record_round_trip_from_measurement():
    m = Measurement(patient="p-07", kind=Kind.GLUCOSE_CGM, ts=TS, value=120.0, source="s", device="dexcom-g6",
                    raw_ref="f:1")
    rec = MeasurementRecord.from_measurement(m)
    assert rec.unit == "mg/dL" and rec.value_type == "quantity"
    assert MeasurementRecord.model_validate_json(rec.model_dump_json()) == rec


GOOD = {"patient": "p-07", "kind": "glucose_cgm", "ts": "2024-01-15T10:00:00Z", "value": 120.0,
        "unit": "mg/dL", "source": "s", "device": "d", "raw_ref": "f:1"}


def test_contract_accepts_a_good_record():
    assert MeasurementRecord.model_validate(GOOD).value_type == "quantity"
    sleep = dict(GOOD, kind="sleep_stage", value="deep", unit=None, end="2024-01-15T10:30:00Z")
    assert MeasurementRecord.model_validate(sleep).value_type == "code"


@pytest.mark.parametrize("bad", [
    {"kind": "meal_type", "value": 12.0, "unit": None},           # quantity given for a code kind
    {"kind": "meal_type", "value": "brunch", "unit": None},        # not an allowed code
    {"value": "high"},                                            # code given for a quantity kind
    {"value": 900.0},                                             # out of the plausible range
    {"ts": "2024-01-15T10:00:00"},                                # naive time
    {"kind": "sleep_stage", "value": "deep", "unit": None},       # period kind without end
    {"end": "2024-01-15T11:00:00Z"},                              # end on a point kind
    {"unit": "mmol/L"},                                           # records carry the canonical unit only
    {"patient": "jane@example.com"},                              # not a pseudonymous id
    {"value_type": "code"},                                       # declared type contradicts the kind
])
def test_contract_rejects_inconsistent_records(bad):
    with pytest.raises(ValidationError):
        MeasurementRecord.model_validate(dict(GOOD, **bad))


def test_committed_json_schemas_match_the_code():
    # The published contract can't silently drift from the models.
    for name, schema in json_schemas().items():
        committed = json.loads((ROOT / "sources" / "schema" / f"{name}.schema.json").read_text())
        assert committed == schema, f"regenerate: python -m sources.schema ({name})"


def test_unhashable_kind_is_a_validation_error():
    for bad in (["x"], {"a": 1}):
        with pytest.raises(ValidationError):
            MeasurementRecord.model_validate(dict(GOOD, kind=bad))


def test_boolean_value_rejected_by_contract():
    with pytest.raises(ValidationError):
        MeasurementRecord.model_validate(dict(GOOD, kind="steps", unit="{steps}", value=True))


def test_mapping_rejects_two_columns_for_the_same_stream_and_case_colliding_labels():
    two = _with(fields=[{"column": "a", "kind": "glucose_cgm", "unit": "mg/dL"},
                        {"column": "b", "kind": "glucose_cgm", "unit": "mg/dL"}])
    with pytest.raises(ValidationError):
        MappingSpec.model_validate(two)
    clash = _with(fields=[{"column": "m", "kind": "meal_type", "values": {"A": "lunch", "a": "dinner"}}])
    with pytest.raises(ValidationError):
        MappingSpec.model_validate(clash)


def test_epoch_strings_must_be_plain_decimals(tmp_path):
    spec = MappingSpec.model_validate({"name": "e", "format": "csv", "time": {"columns": ["t"], "epoch": "seconds"},
                                       "fields": [{"column": "g", "kind": "glucose_cgm", "unit": "mg/dL"}]})
    f = tmp_path / "e.csv"
    f.write_text("t,g\n1e9,100\n1_705_322_700,100\n1705322700,100\n")
    res = parse_mapped(f, spec, patient="p-07")
    assert len(res.raws) == 1 and len(res.quarantined) == 2


def test_end_time_only_column_combines_with_start_date_and_crosses_midnight(tmp_path):
    spec = MappingSpec.model_validate({
        "name": "sleep", "format": "csv", "time": {"columns": ["Date", "Start"], "format": "%Y-%m-%d %H:%M"},
        "fields": [{"column": "Stage", "kind": "sleep_stage", "end_column": "End", "end_format": "%H:%M"}]})
    f = tmp_path / "s.csv"
    f.write_text("Date,Start,End,Stage\n2024-01-14,23:30,00:10,deep\n2024-01-15,01:00,01:20,rem\n")
    n = normalize(parse_mapped(f, spec, patient="p-07").raws, tz="UTC")
    assert [(m.value, m.ts.hour, m.end.day, m.end.hour) for m in n.measurements] == [("deep", 23, 15, 0), ("rem", 1, 15, 1)]


def test_json_numeric_labels_use_the_values_map(tmp_path):
    spec = MappingSpec.model_validate({"name": "j", "format": "json", "records": "r",
                                       "time": {"columns": ["t"], "epoch": "seconds"},
                                       "fields": [{"column": "lvl", "kind": "activity_intensity",
                                                   "values": {"4": "vigorous"}}]})
    f = tmp_path / "j.json"
    f.write_text('{"r": [{"t": 1705322700, "lvl": 4}]}')
    n = normalize(parse_mapped(f, spec, patient="p-07").raws)
    assert n.measurements[0].value == "vigorous"
