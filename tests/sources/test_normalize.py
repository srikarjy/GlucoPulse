"""Canonical model + normalization rules (sources/model.py, sources/normalize.py).

The three algorithm decisions approved at Gate 1 are pinned here:
  1. same device + same time slot + different values -> quarantine BOTH (never pick one);
     two *different* devices disagreeing is not a conflict.
  2. a naive timestamp with no --tz is quarantined, never assumed UTC.
  3. duplicates are matched per device at its native sampling interval
     (fitbit 60 s, dexcom 300 s, libre 900 s); same value within 0.1 -> dropped.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from sources.model import KINDS, Kind, RawMeasurement
from sources.normalize import normalize

UTC = timezone.utc


def raw(kind=Kind.GLUCOSE_CGM, value=120.0, unit="mg/dL", ts=datetime(2024, 1, 15, 10, 0, tzinfo=UTC),
        device="dexcom-g6", patient="p1", ref="f:1", source="test"):
    return RawMeasurement(patient=patient, kind=kind, ts=ts, value=value, unit=unit,
                          source=source, device=device, raw_ref=ref)


# ---------------------------------------------------------------- kind registry
def test_every_kind_has_a_complete_spec():
    for kind in Kind:
        spec = KINDS[kind]
        assert spec.code and spec.system and spec.display and spec.category
        if spec.value_type == "quantity":
            assert spec.unit and spec.low < spec.high
        else:
            assert spec.value_type == "code" and spec.codes and spec.synonyms


def test_verified_loinc_codes():
    assert KINDS[Kind.GLUCOSE_CGM].code == "99504-3"   # already used by the facade
    assert KINDS[Kind.GLUCOSE_BLOOD].code == "2339-0"
    assert KINDS[Kind.HEART_RATE].code == "8867-4"
    assert KINDS[Kind.STEPS].code == "55423-8"
    assert KINDS[Kind.ACTIVE_CALORIES].code == "41981-2"


def test_per_meal_nutrients_use_local_codes_not_24h_loinc_totals():
    # Every LOINC macronutrient intake code is a 24-hour total (g/(24.h)); a single meal is not that.
    for k in (Kind.MEAL_CARBS, Kind.MEAL_PROTEIN, Kind.MEAL_FAT, Kind.MEAL_FIBER):
        assert KINDS[k].system == "urn:glucopulse:code"


# ---------------------------------------------------------------- units
def test_mmol_glucose_converted_to_mgdl():
    r = normalize([raw(value=5.5, unit="mmol/L")])
    assert r.measurements[0].value == pytest.approx(5.5 * 18.016)


def test_apple_mmol_unit_string_is_recognised():
    r = normalize([raw(value=6.0, unit="mmol<180.1558800000541>/L")])
    assert r.measurements[0].value == pytest.approx(6.0 * 18.016)


def test_cgmacros_mets_times_ten_is_divided():
    r = normalize([raw(kind=Kind.METS, value=15, unit="MET*10", device="fitbit")])
    assert r.measurements[0].value == pytest.approx(1.5)


def test_unknown_unit_is_quarantined_not_guessed():
    r = normalize([raw(unit="mg/L")])
    assert not r.measurements
    assert r.quarantined[0].reason == "unsupported_unit"


# ---------------------------------------------------------------- time zones
def test_naive_timestamp_without_tz_is_quarantined():
    r = normalize([raw(ts=datetime(2024, 1, 15, 10, 0))])
    assert not r.measurements
    assert r.quarantined[0].reason == "missing_timezone"


def test_naive_timestamp_localised_with_given_tz():
    r = normalize([raw(ts=datetime(2024, 1, 15, 10, 0))], tz="America/New_York")
    assert r.measurements[0].ts == datetime(2024, 1, 15, 15, 0, tzinfo=UTC)


def test_aware_timestamp_converted_to_utc_and_tz_arg_ignored():
    from zoneinfo import ZoneInfo
    ts = datetime(2024, 7, 1, 9, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
    r = normalize([raw(ts=ts)], tz="Asia/Tokyo")
    assert r.measurements[0].ts == datetime(2024, 7, 1, 16, 0, tzinfo=UTC)


def test_invalid_tz_name_raises():
    with pytest.raises(ValueError):
        normalize([raw()], tz="Mars/Olympus")


# ---------------------------------------------------------------- values
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_values_quarantined(bad):
    r = normalize([raw(value=bad)])
    assert r.quarantined[0].reason == "non_numeric"


def test_implausible_values_quarantined():
    r = normalize([raw(value=900.0), raw(kind=Kind.HEART_RATE, value=400, unit="/min", device="fitbit")])
    assert {q.reason for q in r.quarantined} == {"implausible_value"}
    assert not r.measurements


# ---------------------------------------------------------------- dedup / conflict
def test_exact_duplicate_within_half_interval_is_dropped():
    a = raw(ts=datetime(2024, 1, 15, 10, 0, 10, tzinfo=UTC), ref="f:1")
    b = raw(ts=datetime(2024, 1, 15, 10, 2, 0, tzinfo=UTC), value=120.05, ref="f:2")  # 110 s < 150 s (half of dexcom's 300)
    r = normalize([a, b])
    assert len(r.measurements) == 1 and r.measurements[0].raw_ref == "f:1"
    assert r.stats["duplicates_dropped"] == 1 and not r.quarantined


def test_conflicting_values_same_device_same_slot_quarantines_both():
    r = normalize([raw(value=138, ref="f:1"), raw(value=211, ref="f:2")])
    assert not r.measurements
    assert sorted(q.raw_ref for q in r.quarantined) == ["f:1", "f:2"]
    assert {q.reason for q in r.quarantined} == {"conflicting_values"}


def test_two_devices_disagreeing_is_not_a_conflict():
    r = normalize([raw(device="dexcom-g6", value=120), raw(device="libre-pro", value=145)])
    assert len(r.measurements) == 2 and not r.quarantined


def test_slot_width_follows_device_native_interval():
    # 4 min apart: within half of Libre's 15 min (same sample), beyond half of Fitbit's 1 min (both kept).
    t1, t2 = datetime(2024, 1, 15, 10, 0, tzinfo=UTC), datetime(2024, 1, 15, 10, 4, tzinfo=UTC)
    libre = normalize([raw(device="libre-pro", ts=t1), raw(device="libre-pro", ts=t2)])
    fitbit = normalize([raw(kind=Kind.HEART_RATE, unit="/min", device="fitbit", value=70, ts=t1),
                        raw(kind=Kind.HEART_RATE, unit="/min", device="fitbit", value=70, ts=t2)])
    assert len(libre.measurements) == 1
    assert len(fitbit.measurements) == 2


def test_different_patients_never_collide():
    r = normalize([raw(patient="p1", value=100), raw(patient="p2", value=200)])
    assert len(r.measurements) == 2


def test_output_sorted_and_stats_counted():
    r = normalize([raw(ts=datetime(2024, 1, 15, 11, 0, tzinfo=UTC)), raw(ts=datetime(2024, 1, 15, 10, 0, tzinfo=UTC)),
                   raw(unit="bogus")])
    assert [m.ts.hour for m in r.measurements] == [10, 11]
    assert r.stats["accepted"] == 2 and r.stats["quarantined"] == 1 and r.stats["input"] == 3


# ---------------------------------------------------------------- review fixes
def test_consecutive_dexcom_readings_in_one_clock_slot_are_not_a_conflict():
    # 290 s apart: fixed epoch buckets put these in one 300 s slot; they are two real readings.
    a = raw(ts=datetime(2024, 1, 15, 10, 0, 5, tzinfo=UTC), value=100)
    b = raw(ts=datetime(2024, 1, 15, 10, 4, 55, tzinfo=UTC), value=110)
    r = normalize([a, b])
    assert len(r.measurements) == 2 and not r.quarantined


def test_unknown_device_high_rate_samples_are_all_kept():
    # Apple writes several heart-rate samples per minute; no known native interval -> exact-time matching only.
    t = datetime(2024, 1, 15, 8, 0, tzinfo=UTC)
    hr = [raw(kind=Kind.HEART_RATE, unit="count/min", device="apple-watch", value=v,
              ts=t.replace(second=s_), ref=f"x:{s_}") for v, s_ in ((70, 0), (72, 10), (75, 20))]
    steps = [raw(kind=Kind.STEPS, unit="count", device="iphone", value=v, ts=t.replace(second=s_))
             for v, s_ in ((20, 0), (15, 30))]
    r = normalize(hr + steps)
    assert len(r.measurements) == 5 and not r.quarantined


def test_unknown_device_same_instant_different_values_still_conflict():
    t = datetime(2024, 1, 15, 8, 0, tzinfo=UTC)
    r = normalize([raw(kind=Kind.HEART_RATE, unit="/min", device="apple-watch", value=70, ts=t),
                   raw(kind=Kind.HEART_RATE, unit="/min", device="apple-watch", value=90, ts=t)])
    assert {q.reason for q in r.quarantined} == {"conflicting_values"}


def test_spring_forward_nonexistent_local_time_is_quarantined():
    r = normalize([raw(ts=datetime(2024, 3, 10, 2, 30))], tz="America/New_York")
    assert not r.measurements and r.quarantined[0].reason == "nonexistent_local_time"


def test_fall_back_ambiguous_local_time_is_quarantined():
    r = normalize([raw(ts=datetime(2024, 11, 3, 1, 30))], tz="America/New_York")
    assert not r.measurements and r.quarantined[0].reason == "ambiguous_local_time"


@pytest.mark.parametrize("ts, tz", [
    (datetime(1, 1, 1, 0, 0), "Asia/Tokyo"),                 # overflows when converted to UTC
    (datetime(9999, 12, 31, 23, 0), "America/New_York"),
    (datetime(1900, 1, 1, 7, 45), "UTC"),                    # what a time-only strptime format yields
    (datetime(1970, 1, 1, 0, 0, tzinfo=UTC), None),          # epoch 0
    (datetime(2150, 1, 1, tzinfo=UTC), None),
])
def test_out_of_range_timestamps_quarantined_not_crashing(ts, tz):
    r = normalize([raw(ts=ts)], tz=tz)
    assert not r.measurements and r.quarantined[0].reason == "invalid_timestamp"


@pytest.mark.parametrize("bad", ["7_2", "\u0667\u0662", "1e2", True, " ", "72abc", "0x48"])
def test_strict_numeric_parsing(bad):
    r = normalize([raw(kind=Kind.STEPS, unit="count", device="fitbit", value=bad)])
    assert not r.measurements and r.quarantined[0].reason == "non_numeric"


@pytest.mark.parametrize("ok, expected", [("72", 72.0), (" 72.5 ", 72.5), ("-0", 0.0), (".5", 0.5), (12, 12.0)])
def test_plain_decimals_accepted(ok, expected):
    r = normalize([raw(kind=Kind.STEPS, unit="count", device="fitbit", value=ok)])
    assert r.measurements[0].value == expected


def test_huge_integer_is_non_numeric_not_a_crash():
    r = normalize([raw(kind=Kind.STEPS, unit="count", device="fitbit", value=10**400)])
    assert r.quarantined[0].reason == "non_numeric"


def test_quarantine_detail_value_is_clipped():
    r = normalize([raw(value="9" * 50_000 + "x")])
    assert len(str(r.quarantined[0].detail)) < 400


def test_ordinal_strip_only_for_word_labels():
    from sources.model import Kind as K
    assert normalize([raw(kind=K.ACTIVITY_INTENSITY, value="1 2", unit="")]).quarantined[0].reason == "unknown_code"
    assert normalize([raw(kind=K.MEAL_TYPE, value="snack 2", unit="")]).measurements[0].value == "snack"
