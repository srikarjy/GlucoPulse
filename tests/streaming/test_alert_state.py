"""Streaming alerts (consumer/alert_state.py) must match the tested batch rule (serving/alerts.evaluate) exactly."""
from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

import pytest

pytest.importorskip("numpy")
from consumer.alert_state import AlertStream  # noqa: E402
from serving.alerts import evaluate  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
T0 = datetime(2026, 1, 1, 8, 0)


def batch_alerts(readings):
    """Reference: evaluate() on the full history up to each reading, as the API would if called every 5 min."""
    out, seen = [], set()
    for n in range(1, len(readings) + 1):
        r = evaluate(readings[:n])
        if r["alert"]:
            e = r["alerts_in_window"][-1]
            key = (datetime.fromisoformat(e["timestamp"]), e["tier"])
            if key not in seen:                     # evaluate repeats the same alert for a 2nd reading in one slot
                seen.add(key)
                out.append(key)
    return out


def stream_run(readings, patient="p"):
    s = AlertStream()
    return [a for _, a in (s.process(patient, t, g) for t, g in readings) if a]


def stream_alerts(readings, patient="p"):
    return [(a["time"], a["tier"]) for a in stream_run(readings, patient)]


def series(values, start=T0, step=5):
    return [(start + timedelta(minutes=step * k), float(v)) for k, v in enumerate(values)]


def test_matches_batch_on_a_real_person():
    path = ROOT / "data" / "stream" / "cgmacros-001.csv"
    if not path.exists():
        pytest.skip("run scripts/export_stream_csv.py --patients 1 first")
    rows = [(datetime.fromisoformat(r["timestamp"]), float(r["glucose"])) for r in csv.DictReader(path.open())]
    rows = rows[:1200]                              # ~4 days; the batch reference is quadratic
    expected = batch_alerts(rows)
    assert len(expected) > 10                       # the test is about something
    assert stream_alerts(rows) == expected


def test_long_warning_run_keeps_the_30_min_rhythm():
    """Glucose drifts down slowly enough to project low for hours: alerts every 30 min, exactly as batch."""
    rows = series([150, 148, 146, 144] + [round(110 - 0.6 * k, 1) for k in range(60)])
    got = stream_alerts(rows)
    assert got == batch_alerts(rows)
    assert len(got) >= 2


def test_matches_batch_on_synthetic_runs_with_gaps_and_fast_moves():
    vals = [120, 115, 108, 100, 92, 86, 80, 76, 74, 73, 75, 80, 90, 105, 125, 150, 180, 210, 235, 245, 248, 249]
    rows = series(vals) + series([200, 220, 238, 246, 249, 249], start=T0 + timedelta(hours=3))   # after a gap
    got = stream_alerts(rows)
    assert got == batch_alerts(rows)
    assert got                                      # something fired


def test_duplicate_and_conflict_ignored():
    s = AlertStream()
    assert s.process("p", T0, 100)[0] == "ok"
    assert s.process("p", T0, 100) == ("duplicate", None)
    assert s.process("p", T0, 140) == ("conflict", None)
    assert s.people["p"].readings == [(T0, 100.0)]


def test_late_reading_never_alerts_about_the_past():
    s = AlertStream()
    rows = series([130, 120, 110, 100, 90, 82])
    for t, g in rows[:2] + rows[3:]:
        s.process("p", t, g)
    status, alert = s.process("p", rows[2][0], rows[2][1])    # arrives last, belongs in the middle
    assert (status, alert) == ("late", None)
    assert [t for t, _ in s.people["p"].readings] == [t for t, _ in rows]


def test_too_old_dropped():
    s = AlertStream(retention=timedelta(hours=2))
    s.process("p", T0 + timedelta(hours=5), 100)
    assert s.process("p", T0, 100) == ("too_old", None)


def test_second_reading_in_the_same_5_min_slot_makes_no_new_decision():
    s = AlertStream()
    for t, g in series([130, 120, 110, 100]):
        s.process("p", t, g)
    assert s.process("p", T0 + timedelta(minutes=16), 95) == ("same_slot", None)


def test_people_are_independent():
    rows = series([130, 120, 110, 100, 90, 82])
    s = AlertStream()
    a_alerts, b_alerts = [], []
    for t, g in rows:
        a_alerts.append(s.process("a", t, g)[1])
        b_alerts.append(s.process("b", t, 150)[1])
    assert any(a_alerts) and not any(b_alerts)


def test_fifteen_minute_data_matches_batch():
    rows = series([180, 170, 155, 135, 112, 95, 82, 76, 74, 80, 95], step=15)
    alerts = stream_run(rows)
    assert [(a["time"], a["tier"]) for a in alerts] == batch_alerts(rows)
    assert alerts and all(a["spacing_min"] == 15 for a in alerts)


def test_slow_creep_toward_a_limit_with_decimal_values():
    """Glucose closing in on 70 for ~5 h with one decimal (as mmol/L converted to mg/dL gives): alerts stay every
    30 min of the continuous run. A plain evaluate() over only the last 2 hours restarts the chain at the window edge
    and gets 3 of these 6 alerts wrong; the streaming state must not."""
    import math
    rows = series([round(70 + 60 * math.exp(-5 * k / 18), 1) for k in range(60)])
    got = stream_alerts(rows)
    assert got == batch_alerts(rows)
    assert len(got) == 6
