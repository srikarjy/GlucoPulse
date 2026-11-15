"""CGMacros 5-minute modelling grid (model/cgmacros/dataset.py).

The grid is built from sources/ output (recovered Dexcom readings, Fitbit, meals),
never from the raw interpolated CSV columns. Pinned here: binning, meal placement,
and that past-only covariates never see the future.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

pd = pytest.importorskip("pandas")

from model.cgmacros.dataset import grid_from_measurements  # noqa: E402
from model.cgmacros.splits import split_participants  # noqa: E402
from sources.model import Kind, Measurement  # noqa: E402

T0 = datetime(2020, 5, 1, 10, 0, tzinfo=timezone.utc)


def m(kind, minutes, value, device):
    return Measurement(patient="cgmacros-001", kind=kind, ts=T0 + timedelta(minutes=minutes), value=value,
                       source="cgmacros", device=device, raw_ref="x")


def test_grid_rows_are_dexcom_readings_on_5_min_bins():
    ms = [m(Kind.GLUCOSE_CGM, 0, 100, "dexcom-g6"), m(Kind.GLUCOSE_CGM, 5, 104, "dexcom-g6"),
          m(Kind.GLUCOSE_CGM, 15, 110, "dexcom-g6"), m(Kind.GLUCOSE_CGM, 7, 999, "libre-pro")]
    g = grid_from_measurements("cgmacros-001", ms)
    assert list(g["time_idx"]) == [0, 1, 3]                  # gap at 10 min stays a gap (allow_missing_timesteps)
    assert list(g["glucose"]) == [100, 104, 110]             # Libre is not the target
    assert list(g["delta"]) == [0.0, 4.0, 6.0]


def test_meal_lands_on_the_first_row_at_or_after_it():
    ms = [m(Kind.GLUCOSE_CGM, t, 100 + t, "dexcom-g6") for t in (0, 5, 10, 15)]
    ms += [m(Kind.MEAL_CARBS, 6, 45, "meal-log"), m(Kind.MEAL_FAT, 6, 12, "meal-log")]
    g = grid_from_measurements("cgmacros-001", ms)
    assert list(g["carbs"]) == [0, 0, 45, 0] and list(g["fat"]) == [0, 0, 12, 0]


def test_fitbit_aggregates_only_the_minutes_before_each_row():
    ms = [m(Kind.GLUCOSE_CGM, t, 100, "dexcom-g6") for t in (0, 5, 10)]
    ms += [m(Kind.HEART_RATE, t, 60 + t, "fitbit") for t in range(0, 10)]
    ms += [m(Kind.METS, t, 2.0, "fitbit") for t in range(5, 10)]
    g = grid_from_measurements("cgmacros-001", ms)
    # row at 10 min sees minutes 5..9 only (no peeking at minute 10 or later)
    assert g.loc[g.time_idx == 2, "hr"].item() == pytest.approx(sum(60 + t for t in range(5, 10)) / 5)
    assert g.loc[g.time_idx == 2, "mets"].item() == pytest.approx(2.0)
    assert g.loc[g.time_idx == 1, "mets"].item() == pytest.approx(1.0)      # no activity logged -> resting MET


def test_split_is_deterministic_disjoint_and_by_participant():
    ids = [f"cgmacros-{i:03d}" for i in range(1, 46)]
    a, b = split_participants(ids), split_participants(list(reversed(ids)))
    assert a == b
    train, val, test = a
    assert len(train) + len(val) + len(test) == 45 and len(test) >= 6
    assert not (set(train) & set(val)) and not (set(train) & set(test)) and not (set(val) & set(test))
