"""The fault generators used by scripts/chaos_streaming.py (the run itself needs the Docker stack)."""
from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("chaos_streaming", ROOT / "scripts" / "chaos_streaming.py")
chaos = importlib.util.module_from_spec(spec)
sys.modules["chaos_streaming"] = chaos
spec.loader.exec_module(chaos)

ROWS = [(f"2020-05-01 10:{m:02d}:00", 100.0 + m) for m in range(0, 60, 5)]


def test_duplicates_send_each_reading_twice_in_place():
    out = chaos.with_duplicates(ROWS)
    assert len(out) == 2 * len(ROWS) and out[0] == out[1] and out[2] == ROWS[1]


def test_reorder_only_swaps_neighbours_and_keeps_every_reading():
    out = chaos.reordered(ROWS, random.Random(1), p=0.5)
    assert sorted(out) == sorted(ROWS) and out != ROWS
    for i, r in enumerate(out):                        # each reading is at most one place away
        assert abs(ROWS.index(r) - i) <= 1


def test_gap_removes_exactly_twenty_minutes():
    out = chaos.with_gap(ROWS, 2, minutes=20)
    assert [r[0][-5:-3] for r in out] == ["00", "05", "30", "35", "40", "45", "50", "55"]


def test_expected_alerts_follow_arrival_order_and_ignore_duplicates():
    rows = [(f"2020-05-01 10:{m:02d}:00", g) for m, g in zip(range(0, 50, 5), [130, 120, 110, 100, 90, 82, 76, 74,
                                                                               74, 75])]
    clean = chaos.expected_alerts("p", rows)
    assert clean and chaos.expected_alerts("p", chaos.with_duplicates(rows)) == clean


def test_garbage_scenario_mixes_in_six_bad_messages(monkeypatch):
    monkeypatch.setattr(chaos, "load", lambda pid: ROWS)
    rows = chaos.scenario_inputs("garbage", "", random.Random(0))["cgmacros-012"]
    assert sum(isinstance(r, bytes) for r in rows) == 6 and sum(isinstance(r, tuple) for r in rows) == len(ROWS)
