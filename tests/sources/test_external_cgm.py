"""External CGM adapters (sources/external_cgm.py): Dexcom limits, one meal per start time, pseudonymous ids."""
from __future__ import annotations

from sources.external_cgm import parse_bigideas, parse_hall2018
from sources.model import Kind
from sources.normalize import normalize

DEX_HEADER = ("Index,Timestamp (YYYY-MM-DDThh:mm:ss),Event Type,Event Subtype,Patient Info,Device Info,Source Device ID,"
              "Glucose Value (mg/dL),Insulin Value (u),Carb Value (grams),Duration (hh:mm:ss),"
              "Glucose Rate of Change (mg/dL/min),Transmitter Time (Long Integer)\n")


def test_bigideas_limits_and_meal_aggregation(tmp_path):
    d = tmp_path / "007"
    d.mkdir()
    (d / "Dexcom_007.csv").write_text(
        DEX_HEADER
        + "1,,FirstName,,x,,,,,,,,\n"
        + "2,2020-02-13T10:00:00,EGV,,,,,95,,,,,1\n"
        + "3,2020-02-13T10:05:00,EGV,,,,,Low,,,,,2\n")
    (d / "Food_Log_007.csv").write_text(
        "date,time,time_begin,time_end,logged_food,amount,unit,searched_food,calorie,total_carb,dietary_fiber,sugar,"
        "protein,total_fat\n"
        "2020-02-13,12:00:00,2020-02-13 12:00:00,,Rice,1,cup,rice,200,45,1,0,4,0.5\n"
        "2020-02-13,12:00:00,2020-02-13 12:00:00,,Beans,1,cup,beans,220,40,15,1,15,1\n")
    r = parse_bigideas(d, "bigideas-007")
    glucose = sorted(float(m.value) for m in r.raws if m.kind is Kind.GLUCOSE_CGM)
    assert glucose == [40.0, 95.0]                                   # "Low" is the sensor limit, not dropped
    carbs = [m for m in r.raws if m.kind is Kind.MEAL_CARBS]
    assert len(carbs) == 1 and carbs[0].value == 85.0                # two items at one time = one meal
    n = normalize(r.raws, tz="UTC")
    assert not n.quarantined
    assert all(m.patient == "bigideas-007" for m in n.measurements)


def test_hall_pseudonyms_and_display_time(tmp_path):
    f = tmp_path / "S1.txt"
    f.write_text("DisplayTime\tGlucoseValue\tsubjectId\tInternalTime\n"
                 "2014-02-03 03:42:12\t93\t2133-018\t2016-02-03 00:05:23\n"
                 "2014-02-03 03:47:12\tHigh\t2133-018\t2016-02-03 00:10:23\n"
                 "2014-02-03 03:42:12\t88\t1636-69-001\t2016-02-03 00:05:23\n")
    out = parse_hall2018(f)
    assert sorted(out) == ["hall-001", "hall-002"]                   # by sorted subject code, codes never kept
    h2 = out["hall-002"].raws
    assert [float(m.value) for m in h2] == [93.0, 400.0]
    assert h2[0].ts.hour == 3 and h2[0].device == "dexcom-g4"        # DisplayTime, not InternalTime
