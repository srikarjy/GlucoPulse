"""FHIR mapping of normalized measurements + the ingest CLI end to end."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from fhir.resources.R4B.bundle import Bundle
from fhir.resources.R4B.observation import Observation

from serving.healthcare import fhir_mapping as fm
from sources.model import KINDS, Kind, Measurement

FX = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).resolve().parents[2]
TS = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)


def m(kind=Kind.GLUCOSE_CGM, value=120.0, device="dexcom-g6", source="cgmacros"):
    return Measurement(patient="p-07", kind=kind, ts=TS, value=value, source=source, device=device, raw_ref="f:1")


def test_every_quantity_kind_maps_to_a_valid_r4_observation():
    # code/period kinds are covered in test_types.py
    for kind in Kind:
        spec = KINDS[kind]
        if spec.value_type != "quantity" or spec.period:
            continue
        obs = fm.observation_from_measurement(m(kind=kind, value=spec.low + 1))
        Observation.model_validate(obs)                               # raises if not valid FHIR R4
        assert obs["code"]["coding"][0] == {"system": spec.system, "code": spec.code, "display": spec.display}
        assert obs["valueQuantity"]["unit"] == spec.unit
        assert obs["category"][0]["coding"][0]["code"] == spec.category
        assert obs["subject"] == {"reference": "Patient/p-07"}


def test_provenance_of_source_and_device_is_kept():
    obs = fm.observation_from_measurement(m())
    assert obs["meta"]["source"] == "urn:glucopulse:source:cgmacros"
    assert obs["device"] == {"display": "dexcom-g6"}


def test_two_cgms_at_same_instant_get_distinct_ids():
    a = fm.observation_from_measurement(m(device="dexcom-g6"))
    b = fm.observation_from_measurement(m(device="libre-pro"))
    assert a["id"] != b["id"]
    assert fm.observation_from_measurement(m(device="dexcom-g6"))["id"] == a["id"]   # deterministic


def test_existing_single_source_ids_are_unchanged():
    # observation_from_reading is what the live facade serves; its ids must not move.
    assert fm.observation_from_reading("p-07", TS, 120.0)["id"] == fm.det_id("obs", "p-07", fm.iso(TS))


def _run(*args):
    return subprocess.run([sys.executable, "-m", "sources.ingest", *args], cwd=ROOT,
                           capture_output=True, text=True, timeout=60)


def test_cli_cgmacros_end_to_end(tmp_path):
    out, qout = tmp_path / "bundle.json", tmp_path / "quarantine.json"
    p = _run("--format", "cgmacros", str(FX / "CGMacros-001.csv"), "--tz", "UTC", "--out", str(out),
             "--quarantine-out", str(qout))
    assert p.returncode == 0, p.stderr
    bundle = json.loads(out.read_text())
    Bundle.model_validate(bundle)
    assert len(bundle["entry"]) == 22                # 23 raw values (incl. meal type/kcal/% eaten) - "abc"
    q = json.loads(qout.read_text())
    assert sorted(r["reason"] for r in q) == ["malformed_record", "non_numeric"]
    summary = json.loads(p.stdout)
    assert summary["accepted"] == 22 and summary["quarantined"] == 2


def test_cli_fitbit_requires_patient_and_reports_missing_tz(tmp_path):
    out = tmp_path / "b.json"
    p = _run("--format", "fitbit", str(FX / "fitbit_heart.json"), "--patient", "p-07", "--out", str(out))
    assert p.returncode == 3, p.stderr                            # nothing usable was produced
    summary = json.loads(p.stdout)
    assert summary["accepted"] == 0 and summary["reasons"]["missing_timezone"] == 2


def test_cli_rejects_unsafe_xml_without_crashing(tmp_path):
    p = _run("--format", "apple", str(FX / "apple_xxe.xml"), "--patient", "p-07", "--out", str(tmp_path / "b.json"))
    assert p.returncode == 3, p.stderr
    assert json.loads(p.stdout)["reasons"] == {"unsafe_xml": 1}


def test_cli_rejects_direct_identifier_as_patient(tmp_path):
    p = _run("--format", "fitbit", str(FX / "fitbit_heart.json"), "--patient", "jane@example.com",
             "--out", str(tmp_path / "b.json"))
    assert p.returncode == 2


def test_cli_missing_input_file_is_a_clean_usage_error(tmp_path):
    p = _run("--format", "cgmacros", str(tmp_path / "nope.csv"), "--tz", "UTC", "--out", str(tmp_path / "b.json"))
    assert p.returncode == 2 and "Traceback" not in p.stderr


def test_cli_reports_adapter_stats(tmp_path):
    p = _run("--format", "apple", str(FX / "apple_export.xml"), "--patient", "p-07", "--out", str(tmp_path / "b.json"))
    assert p.returncode == 0, p.stderr
    summary = json.loads(p.stdout)
    assert "adapter_stats" in summary and summary["accepted"] == 5     # glucose x2, HR, steps, weight


def test_cli_refuses_to_overwrite_its_input(tmp_path):
    src = tmp_path / "CGMacros-001.csv"
    src.write_text((FX / "CGMacros-001.csv").read_text())
    p = _run("--format", "cgmacros", str(src), "--tz", "UTC", "--out", str(src))
    assert p.returncode == 2 and src.read_text().startswith(",Timestamp")


def test_cli_deeply_nested_json_is_a_clean_error(tmp_path):
    f = tmp_path / "deep.json"
    f.write_text("[" * 100_000 + "]" * 100_000)
    p = _run("--format", "fitbit", str(f), "--patient", "p-07", "--out", str(tmp_path / "b.json"))
    assert p.returncode == 2 and "Traceback" not in p.stderr
