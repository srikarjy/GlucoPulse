"""5-minute monitoring loop (model/cgmacros/monitor.py) on synthetic streams."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pandas")

from model.cgmacros.monitor import Monitor  # noqa: E402

T0 = datetime(2020, 5, 1, 8, 0)


def stream(values, start=T0, step=5):
    return [{"ts": start + timedelta(minutes=step * i), "glucose": v} for i, v in enumerate(values)]


def alarms(m, rows):
    return [a for a in (m.step(r) for r in rows) if a]


def test_flat_stream_never_alarms():
    assert alarms(Monitor(policy=("alert20",)), stream([100.0] * 300)) == []


def test_projected_low_alarms_before_the_low():
    vals = [120.0] * 10 + [120.0 - 3 * k for k in range(1, 20)]       # falls 0.6 mg/dL/min
    out = alarms(Monitor(policy=("alert20",)), stream(vals))
    assert out, "a steady fall towards 70 should alarm"
    first = out[0]
    assert first["glucose"] >= 70 and first["projected_20min"] < 70
    assert "trend" in first["reasons"][0]


def test_lockout_one_alarm_per_30_minutes():
    vals = [200.0 + 4 * k for k in range(12)]                          # rising 0.8/min, projections stay > 250
    out = alarms(Monitor(policy=("alert20",)), stream(vals))
    times = [datetime.fromisoformat(a["ts"]) for a in out]
    assert len(out) >= 1 and all((b - a) >= timedelta(minutes=30) for a, b in zip(times, times[1:]))


def test_no_alarm_once_already_out_of_range():
    assert alarms(Monitor(policy=("alert20",)), stream([60.0, 55.0, 50.0, 45.0, 40.0])) == []


def test_gap_clears_history():
    m = Monitor(policy=("alert20",))
    for r in stream([150.0, 140.0, 130.0]) + stream([100.0], start=T0 + timedelta(hours=2)):
        m.step(r)
    assert len(m.buf) == 1


class _Stub:
    def predict_proba(self, X):
        return np.array([[0.1, 0.9]])


def test_event_model_policy_uses_threshold_and_needs_two_hours():
    out = alarms(Monitor(_Stub(), threshold=0.5, policy=("event_model",)), stream([110.0] * 30))
    assert out and datetime.fromisoformat(out[0]["ts"]) == T0 + timedelta(minutes=5 * 23)   # first full 2-h window
    assert out[0]["p_urgent_60min"] == 0.9


def test_bad_policy_is_rejected():
    with pytest.raises(ValueError):
        Monitor(policy=("vibes",))
    with pytest.raises(ValueError):
        Monitor(policy=("event_model",))
