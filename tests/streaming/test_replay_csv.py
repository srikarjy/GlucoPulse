"""Replay of CGMacros exports (scripts/export_stream_csv.py -> producer/replay_sensor.py --source csv)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytest.importorskip("confluent_kafka")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


replay = load("replay_sensor", ROOT / "producer" / "replay_sensor.py")
export = load("export_stream_csv", ROOT / "scripts" / "export_stream_csv.py")


def test_stream_csv_round_trip_gives_ingest_shaped_messages(tmp_path):
    export.write_stream_csv([("2020-05-01 11:39:00", 141.0), ("2020-05-01 11:44:00", 159.5)],
                            tmp_path / "cgmacros-001.csv")
    groups = replay.load_stream_csv(tmp_path, "cgmacros-001")
    assert list(groups) == ["2020-05-01 11:39:00", "2020-05-01 11:44:00"]          # file order kept
    msg = replay.cast_stream_row(groups["2020-05-01 11:44:00"][0], "cgmacros-001")
    azt = replay.cast_row({"EventDateTime": "x", "CGM": "1", "DeviceMode": "", "BolusType": "", "Basal": "",
                           "CorrectionDelivered": "", "TotalBolusInsulinDelivered": "", "FoodDelivered": "",
                           "CarbSize": ""}, "CGM", "1")
    assert set(msg) == set(azt)                                   # same message shape: ingest.py needs no change
    assert (msg["patient_id"], msg["event_datetime"], msg["cgm"]) == ("cgmacros-001", "2020-05-01 11:44:00", 159.5)
    assert msg["carb_size"] is None and msg["basal"] is None


def test_blank_glucose_becomes_null_for_ingest_to_route(tmp_path):
    (tmp_path / "p.csv").write_text("timestamp,glucose\n2020-05-01 11:39:00,\n")
    msg = replay.cast_stream_row(replay.load_stream_csv(tmp_path, "p")["2020-05-01 11:39:00"][0], "p")
    assert msg["cgm"] is None                                     # ingest sends missing cgm to cgm-parse-errors


def test_missing_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        replay.load_stream_csv(tmp_path, "cgmacros-099")


def test_export_keeps_only_dexcom_sorted_unique():
    src = ROOT / "data" / "cgmacros" / "csv" / "CGMacros-001.csv"
    if not src.exists():
        pytest.skip("CGMacros data not downloaded")
    patient, readings, _ = export.dexcom_readings(src)
    times = [t for t, _ in readings]
    assert patient == "cgmacros-001" and times == sorted(times) and len(times) == len(set(times))
    assert all(40 <= g <= 400 for _, g in readings)
