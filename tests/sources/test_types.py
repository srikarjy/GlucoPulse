"""Value types (quantity / code / period), the extended kind registry, and
CGMacros reading recovery from its per-minute linear interpolation."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fhir.resources.R4B.observation import Observation

from serving.healthcare import fhir_mapping as fm
from sources.cgmacros import parse_cgmacros, recover_native_readings
from sources.model import KINDS, Kind, RawMeasurement
from sources.normalize import normalize

UTC = timezone.utc
T0 = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)


def raw(kind, value, unit="", ts=T0, end=None, device="dev"):
    return RawMeasurement(patient="p-07", kind=kind, ts=ts, value=value, unit=unit, source="t",
                          device=device, raw_ref="f:1", end=end)


# ---------------------------------------------------------------- registry
@pytest.mark.parametrize("kind, code, unit", [
    (Kind.SPO2, "59408-5", "%"), (Kind.HRV_SDNN, "80404-7", "ms"), (Kind.RESPIRATORY_RATE, "9279-1", "/min"),
    (Kind.BODY_WEIGHT, "29463-7", "kg"), (Kind.BP_SYSTOLIC, "8480-6", "mm[Hg]"), (Kind.BP_DIASTOLIC, "8462-4", "mm[Hg]"),
])
def test_new_quantity_kinds_have_verified_loinc(kind, code, unit):
    assert KINDS[kind].system == "http://loinc.org" and KINDS[kind].code == code and KINDS[kind].unit == unit


def test_value_types_are_declared():
    assert KINDS[Kind.MEAL_TYPE].value_type == "code"
    assert KINDS[Kind.SLEEP_STAGE].value_type == "code" and KINDS[Kind.SLEEP_STAGE].period
    assert KINDS[Kind.GLUCOSE_CGM].value_type == "quantity" and not KINDS[Kind.GLUCOSE_CGM].period


# ---------------------------------------------------------------- quantity conversions
@pytest.mark.parametrize("kind, value, unit, expected", [
    (Kind.BODY_WEIGHT, 150, "lb", 150 * 0.45359237),
    (Kind.SPO2, 0.97, "fraction", 97.0),
    (Kind.BP_SYSTOLIC, 120, "mmHg", 120.0),
    (Kind.INSULIN_BOLUS, 4.5, "IU", 4.5),
    (Kind.MEAL_PORTION_EATEN, 75, "%", 75.0),
])
def test_quantity_units(kind, value, unit, expected):
    r = normalize([raw(kind, value, unit)])
    assert r.measurements[0].value == pytest.approx(expected)


def test_portion_over_100_percent_is_implausible():
    assert normalize([raw(Kind.MEAL_PORTION_EATEN, 130, "%")]).quarantined[0].reason == "implausible_value"


# ---------------------------------------------------------------- code kinds
@pytest.mark.parametrize("text, code", [("Breakfast", "breakfast"), ("breakfast", "breakfast"), ("Snacks", "snack"),
                                        ("snack 1", "snack"), (" Dinner ", "dinner")])
def test_meal_type_synonyms(text, code):
    assert normalize([raw(Kind.MEAL_TYPE, text)]).measurements[0].value == code


@pytest.mark.parametrize("text, code", [("0", "sedentary"), ("0.0", "sedentary"), ("1", "light"),
                                        ("2.0", "moderate"), ("3", "vigorous")])
def test_fitbit_intensity_codes(text, code):
    assert normalize([raw(Kind.ACTIVITY_INTENSITY, text)]).measurements[0].value == code


def test_unknown_code_is_quarantined():
    assert normalize([raw(Kind.MEAL_TYPE, "brunch-ish")]).quarantined[0].reason == "unknown_code"


def test_same_instant_different_codes_conflict():
    r = normalize([raw(Kind.MEAL_TYPE, "lunch"), raw(Kind.MEAL_TYPE, "dinner")])
    assert {q.reason for q in r.quarantined} == {"conflicting_values"}


# ---------------------------------------------------------------- periods
def test_sleep_stage_period():
    r = normalize([raw(Kind.SLEEP_STAGE, "HKCategoryValueSleepAnalysisAsleepDeep", end=T0 + timedelta(minutes=30))])
    m = r.measurements[0]
    assert m.value == "deep" and m.end == T0 + timedelta(minutes=30)


@pytest.mark.parametrize("end, reason", [(None, "missing_end"), (T0, "invalid_period"),
                                         (T0 + timedelta(hours=30), "invalid_period")])
def test_bad_periods_quarantined(end, reason):
    assert normalize([raw(Kind.SLEEP_STAGE, "deep", end=end)]).quarantined[0].reason == reason


def test_naive_period_end_localised_like_start():
    r = normalize([raw(Kind.SLEEP_STAGE, "rem", ts=datetime(2024, 1, 15, 2, 0), end=datetime(2024, 1, 15, 2, 20))],
                  tz="America/New_York")
    assert r.measurements[0].end == datetime(2024, 1, 15, 7, 20, tzinfo=UTC)


# ---------------------------------------------------------------- FHIR for new value types
def test_code_and_period_map_to_valid_fhir():
    meal = normalize([raw(Kind.MEAL_TYPE, "Lunch")]).measurements[0]
    obs = fm.observation_from_measurement(meal)
    Observation.model_validate(obs)
    assert obs["valueCodeableConcept"]["coding"][0]["code"] == "lunch" and "valueQuantity" not in obs
    sleep = normalize([raw(Kind.SLEEP_STAGE, "rem", end=T0 + timedelta(minutes=20))]).measurements[0]
    obs = fm.observation_from_measurement(sleep)
    Observation.model_validate(obs)
    assert obs["effectivePeriod"] == {"start": fm.iso(T0), "end": fm.iso(T0 + timedelta(minutes=20))}
    assert "effectiveDateTime" not in obs


def test_every_kind_maps_to_valid_fhir():
    for kind, spec in KINDS.items():
        if spec.value_type == "code":
            value, unit = spec.codes[0][0], ""
        else:
            value, unit = (spec.low + spec.high) / 2, spec.unit
        end = T0 + timedelta(minutes=10) if spec.period else None
        n = normalize([raw(kind, value, unit, end=end)])
        assert n.measurements, (kind, n.quarantined)
        Observation.model_validate(fm.observation_from_measurement(n.measurements[0]))


# ---------------------------------------------------------------- CGMacros: recover real readings
def _interpolated(readings: list[tuple[int, float]], start=datetime(2020, 5, 1, 10, 0)):
    """Linear per-minute interpolation between (minute, value) readings, like CGMacros."""
    out = []
    for (m0, v0), (m1, v1) in zip(readings, readings[1:]):
        for m in range(m0, m1):
            out.append((start + timedelta(minutes=m), repr(v0 + (v1 - v0) * (m - m0) / (m1 - m0)), f"r{m}"))
    m, v = readings[-1]
    out.append((start + timedelta(minutes=m), repr(float(v)), f"r{m}"))
    return out


def test_recovers_libre_readings_including_flat_runs():
    pts = _interpolated([(0, 84), (15, 86), (30, 86), (45, 86), (60, 90)])
    kept = recover_native_readings(pts, 15)
    assert [(p[0].minute, float(p[1])) for p in kept] == [(0, 84), (15, 86), (30, 86), (45, 86), (0, 90)]


def test_recovers_dexcom_readings_on_a_straight_run():
    pts = _interpolated([(0, 100), (5, 105), (10, 110), (15, 115), (20, 112)])
    assert [float(p[1]) for p in recover_native_readings(pts, 5)] == [100, 105, 110, 115, 112]


def test_non_grid_series_is_left_untouched():
    t = datetime(2020, 5, 1, 10, 0)
    pts = [(t, "100", "a"), (t + timedelta(minutes=7), "101", "b"), (t + timedelta(minutes=9), "103", "c")]
    assert recover_native_readings(pts, 5) == pts


def test_real_format_cgmacros_file(tmp_path):
    rows = ["Unnamed: 0,Timestamp,Libre GL,Dexcom GL,HR,Steps,Intensity,Meal Type,Calories,Carbs,Protein,Fat,Fiber,"
            "Sugar,Amount Consumed ,Image path"]
    libre = dict((p[0], p[1]) for p in _interpolated([(0, 84), (15, 86), (30, 92)]))
    dex = dict((p[0], p[1]) for p in _interpolated([(0, 100), (5, 104), (10, 108), (15, 110), (20, 112), (25, 111),
                                                    (30, 109)]))
    for i, t in enumerate(sorted(libre)):
        meal = "Lunch,640,80,25,20,6,30,100,photos/x.jpg" if i == 3 else ",,,,,,,,"
        rows.append(f"{i},{t:%Y-%m-%d %H:%M:%S},{libre[t]},{dex[t]},70,12,1.0,{meal}")
    p = tmp_path / "CGMacros-099.csv"
    p.write_text("\n".join(rows) + "\n")
    res = parse_cgmacros(p)
    n = normalize(res.raws, tz="UTC")
    glucose = [(m.device, m.ts.minute, m.value) for m in n.measurements if m.kind is Kind.GLUCOSE_CGM]
    assert [g for g in glucose if g[0] == "libre-pro"] == [("libre-pro", 0, 84), ("libre-pro", 15, 86), ("libre-pro", 30, 92)]
    assert len([g for g in glucose if g[0] == "dexcom-g6"]) == 7
    assert res.stats["interpolated_dropped"] == 31 * 2 - 3 - 7
    kinds = {m.kind for m in n.measurements}
    assert {Kind.STEPS, Kind.ACTIVITY_INTENSITY, Kind.MEAL_TYPE, Kind.MEAL_CALORIES, Kind.MEAL_SUGAR,
            Kind.MEAL_PORTION_EATEN} <= kinds
    assert not n.quarantined


def test_period_at_far_future_does_not_overflow():
    far = datetime(9999, 12, 31, 22, 0, tzinfo=UTC)
    r = normalize([raw(Kind.SLEEP_STAGE, "deep", ts=far, end=far + timedelta(minutes=30))])
    assert r.quarantined[0].reason == "invalid_timestamp"


@pytest.mark.parametrize("bad", ["nan", "inf", "1_0", "1e400"])
def test_recovery_treats_non_decimal_cells_as_breaks_not_numbers(bad):
    pts = _interpolated([(0, 100), (5, 105), (10, 112)])
    pts[4] = (pts[4][0], bad, "bad")
    kept = recover_native_readings(pts, 5)
    assert any(p[1] == bad for p in kept)                  # passed through to be quarantined by normalize()


def test_recovery_counts_refilled_collinear_readings():
    from sources.cgmacros import RecoveryStats
    stats = RecoveryStats()
    recover_native_readings(_interpolated([(0, 100), (5, 105), (10, 110), (15, 115), (20, 112)]), 5, stats)
    assert stats.knots == 3 and stats.refilled == 2
