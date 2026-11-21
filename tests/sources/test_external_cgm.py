"""External CGM adapters (sources/external_cgm.py): Dexcom limits, one meal per start time, pseudonymous ids."""
from __future__ import annotations

from datetime import datetime

import pytest

from sources.external_cgm import parse_bigideas, parse_cgmnd, parse_hall2018, parse_shanghai_t2dm
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


def test_bigideas_skips_bad_rows_and_counts_them(tmp_path):
    d = tmp_path / "003"
    d.mkdir()
    (d / "Dexcom_003.csv").write_text(
        DEX_HEADER
        + "1,2020-02-13T10:00:00,Calibration,,,,,110,,,,,1\n"         # not an EGV row
        + "2,,EGV,,,,,95,,,,,2\n"                                     # no timestamp
        + "3,2020-02-13T10:10:00,EGV,,,,,,,,,,3\n"                    # no glucose
        + "4,2020-02-13T10:15:00,EGV,,,,,High,,,,,4\n")
    (d / "Food_Log_003.csv").write_text(
        "date,time,time_begin,time_end,logged_food,amount,unit,searched_food,calorie,total_carb,dietary_fiber,sugar,"
        "protein,total_fat\n"
        "2020-02-13,,,,Apple,1,each,apple,90,25,4,19,0,0\n"                            # no start time
        "2020-02-13,08:00:00,2020-02-13 08:00:00,,Toast,1,slice,toast,80,abc,1,1,3,\n")  # bad carb, blank fat
    r = parse_bigideas(d, "bigideas-003")
    assert [float(m.value) for m in r.raws if m.kind is Kind.GLUCOSE_CGM] == [400.0]
    assert r.stats["skipped_non_egv"] == 1 and r.stats["skipped_empty_cell"] == 2
    assert r.stats["skipped_food_no_time"] == 1 and r.stats["skipped_food_bad_number"] == 1
    by_kind = {m.kind: m.value for m in r.raws if m.device == "food-log"}
    assert by_kind[Kind.MEAL_CARBS] == 0.0 and by_kind[Kind.MEAL_PROTEIN] == 3.0 and by_kind[Kind.MEAL_FAT] == 0.0
    assert r.stats["meals"] == 1


def test_bigideas_without_food_log_returns_glucose_only(tmp_path):
    d = tmp_path / "011"
    d.mkdir()
    (d / "Dexcom_011.csv").write_text(DEX_HEADER + "1,2020-02-13T10:00:00,EGV,,,,,101,,,,,1\n")
    r = parse_bigideas(d, "bigideas-011")
    assert [m.kind for m in r.raws] == [Kind.GLUCOSE_CGM]
    assert "meals" not in r.stats


def test_hall_empty_cells_skipped(tmp_path):
    f = tmp_path / "S1.txt"
    f.write_text("DisplayTime\tGlucoseValue\tsubjectId\tInternalTime\n"
                 "2014-02-03 03:42:12\t\t2133-018\t\n"
                 "\t90\t2133-018\t\n"
                 "2014-02-03 03:52:12\t91\t2133-018\t\n")
    res = parse_hall2018(f)["hall-001"]
    assert [float(m.value) for m in res.raws] == [91.0]
    assert res.stats["skipped_empty_cell"] == 2


def test_cgmnd_adults_only_cgm_rows_codes_and_time_anchor(tmp_path):
    (tmp_path / "NonDiabPtRoster.csv").write_text(
        "PtID,AgeAsOfEnrollDt\n10,34\n9,17\n2,61\n5,\n")              # 9 is a child; 5 has no age
    (tmp_path / "NonDiabDeviceCGM.csv").write_text(
        "PtID,RecordType,Value,DeviceDtDaysFromEnroll,DeviceTm\n"
        "10,CGM,39,1,06:30:00\n"
        "10,Calibration,120,1,06:35:00\n"
        "10,CGM,401,2,23:55:10\n"
        "9,CGM,100,0,00:00:00\n"
        "5,CGM,100,0,00:00:00\n"
        "2,CGM,88,0,12:00:00\n")
    out = parse_cgmnd(tmp_path)
    assert sorted(out) == ["cgmnd-001", "cgmnd-002"]                 # PtID 2 then 10, sorted numerically
    assert [m.value for m in out["cgmnd-001"].raws] == ["88.0"]
    p10 = out["cgmnd-002"].raws
    assert [m.value for m in p10] == ["40.0", "400.0"]                # LOW/HIGH codes; calibration row skipped
    assert p10[0].ts == datetime(2017, 1, 2, 6, 30) and p10[1].ts == datetime(2017, 1, 3, 23, 55, 10)
    assert all(m.source == "cgmnd" for m in p10)


def test_cgmnd_min_age_is_configurable(tmp_path):
    (tmp_path / "NonDiabPtRoster.csv").write_text("PtID,AgeAsOfEnrollDt\n9,17\n")
    (tmp_path / "NonDiabDeviceCGM.csv").write_text(
        "PtID,RecordType,Value,DeviceDtDaysFromEnroll,DeviceTm\n9,CGM,100,0,00:00:00\n")
    assert parse_cgmnd(tmp_path) == {}
    assert list(parse_cgmnd(tmp_path, min_age=12)) == ["cgmnd-001"]


def test_shanghai_drops_insulin_users_clips_and_pseudonymises(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")
    pd.DataFrame({"Patient Number": ["2001_0_20201230", "2001_1_20210110", "2002_0_20210301", "2003_0_20210401",
                                     "2004_0_20210501"],
                  "Hypoglycemic Agents": ["metformin", "acarbose", "insulin glargine", "metformin", "metformin"]}
                 ).to_excel(tmp_path / "Shanghai_T2DM_Summary.xlsx", index=False)
    data = tmp_path / "Shanghai_T2DM"
    data.mkdir()
    t = pd.date_range("2021-01-01 08:00", periods=4, freq="15min")
    pd.DataFrame({"Date": t, "CGM (mg / dl)": [30.0, 120.0, None, 450.0], "Insulin dose - s.c.": [None] * 4,
                  "Non-insulin hypoglycemic agents": ["metformin"] * 4}).to_excel(data / "2001_0_20201230.xlsx",
                                                                                  index=False)
    pd.DataFrame({"Date": t[:2], "CGM (mg / dl)": [100.0, 110.0]}).to_excel(data / "2001_1_20210110.xlsx",
                                                                             index=False)
    pd.DataFrame({"Date": t[:2], "CGM (mg / dl)": [100.0, 110.0]}).to_excel(data / "2002_0_20210301.xlsx",
                                                                             index=False)
    pd.DataFrame({"Date": t[:2], "CGM (mg / dl)": [100.0, 110.0], "Insulin dose - i.v.": [None, 4.0]}
                 ).to_excel(data / "2003_0_20210401.xlsx", index=False)      # insulin given in the recording
    # 2004 has no recording file at all
    out = parse_shanghai_t2dm(tmp_path)
    assert sorted(out) == ["shanghai-001-1", "shanghai-001-2"]       # 2002 (summary insulin), 2003 (column) dropped
    first = out["shanghai-001-1"]
    assert [m.value for m in first.raws] == [40.0, 120.0, 400.0]
    assert first.stats["clipped_to_sensor_limit"] == 2 and first.stats["skipped_empty_cell"] == 1
    assert first.raws[0].raw_ref.startswith("2001_0_20201230.xlsx:") and first.raws[0].device == "cgm-15min"
