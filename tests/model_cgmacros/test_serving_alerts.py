"""serving/alerts.py: the API's copy of the trend alerts must fire exactly where the tested research rules fire."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

np = pytest.importorskip("numpy")

from serving.alerts import evaluate, flags  # noqa: E402

T0 = datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc)


def series(values, start=T0):
    return [(start + timedelta(minutes=5 * i), float(v)) for i, v in enumerate(values)]


def test_parity_with_research_rules_on_cgmacros():
    pytest.importorskip("pandas")
    from model.cgmacros import replay as R
    from model.cgmacros.alarms import make
    from model.cgmacros.dataset import CACHE, build_grid
    if not CACHE.exists():
        pytest.skip("CGMacros grid not built here")
    g = build_grid()
    loud_ref = make(confirm=2, hi_gate=200, lo_gate=100)
    n = 0
    for _, f in R.segments(g, sorted(g.participant.unique())[:15]):
        x = f["glucose"].to_numpy(float)
        heads, loud, _ = flags(x)
        ok = (x >= R.LO) & (x <= R.HI)
        assert np.array_equal(heads[ok], R.alert20(x)[ok])
        assert np.array_equal(loud[ok], loud_ref(x)[ok])
        n += int(ok.sum())
    assert n > 10000


def test_flat_stream_no_alert():
    r = evaluate(series([110] * 30))
    assert r["alert"] is None and r["alerts_in_window"] == []


def test_fall_gives_heads_up_then_loud():
    vals = [130 - 4 * k for k in range(14)]           # 130 -> 78, falling 0.8 mg/dL/min
    out = evaluate(series(vals))["alerts_in_window"]
    tiers = [a["tier"] for a in out]
    assert tiers[0] == "heads_up" and "loud" in tiers
    first_loud = next(a for a in out if a["tier"] == "loud")
    assert first_loud["glucose"] <= 100 and first_loud["direction"] == "low"


def test_earlier_alert_stays_in_history_but_not_current():
    vals = [130 - 4 * k for k in range(14)]            # falls to 78 (alerts), then holds steady for 40 min
    r = evaluate(series(vals + [vals[-1]] * 8))
    assert r["alert"] is None and r["alerts_in_window"]


def test_gap_restarts_history():
    early = series([200, 220, 240, 245])
    late = series([120, 121, 122, 121], start=T0 + timedelta(hours=3))
    r = evaluate(early + late)
    assert r["continuous_readings"] == 4 and r["alerts_in_window"] == []


def test_api_endpoint(monkeypatch):
    pytest.importorskip("fastapi")
    monkeypatch.delenv("GLUCOPULSE_AUTH_MODE", raising=False)
    from fastapi.testclient import TestClient
    try:
        from serving.api import app
    except Exception as exc:                            # model artifacts or heavy deps missing in this env
        pytest.skip(f"serving.api not importable here: {exc}")
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    vals = [130 - 4 * k for k in range(14)]
    readings = [{"timestamp": (now - timedelta(minutes=5 * (13 - i))).isoformat(), "glucose_value": v}
                for i, v in enumerate(vals)]
    c = TestClient(app)
    r = c.post("/alerts", json={"patient_id": "demo-1", "readings": readings})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["alerts_in_window"] and "not a diagnosis" in body["disclaimer"]
    bad = c.post("/alerts", json={"patient_id": "demo-1", "readings": list(reversed(readings))})
    assert bad.status_code == 422
