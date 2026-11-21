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
    from model.cgmacros.alarms import ESC_CHOSEN, ESC
    from model.cgmacros.dataset import CACHE, build_grid
    if not CACHE.exists():
        pytest.skip("CGMacros grid not built here")
    g = build_grid()
    loud_ref = ESC[ESC_CHOSEN]                          # strict + fast-fall escalation (tested)
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


def test_export_drift_snapped():
    drift = [(T0 + timedelta(minutes=5 * i, seconds=(31 if i % 2 else 32)), v)
             for i, v in enumerate([130 - 4 * k for k in range(14)])]
    r = evaluate(drift)
    assert r["continuous_readings"] == 14 and r["spacing_min"] == 5 and r["enough"]


def fifteen(values, start=T0, drift_s=0):
    return [(start + timedelta(minutes=15 * i, seconds=drift_s * i), float(v)) for i, v in enumerate(values)]


def test_fifteen_minute_data_heads_up_then_loud_on_second_reading():
    r = evaluate(fifteen([150, 130, 110, 90, 80], drift_s=40))          # Libre timestamps drift; 13-17 min is a run
    assert r["spacing_min"] == 15 and r["enough"] and r["continuous_readings"] == 5
    tiers = [(a["tier"], a["glucose"]) for a in r["alerts_in_window"]]
    assert tiers == [("heads_up", 90.0), ("loud", 80.0)]                # loud needs the condition on 2 readings
    one = evaluate(fifteen([150, 130, 110, 90]))
    assert one["alert"] == "heads_up"                                   # one reading: heads-up only


def test_fifteen_minute_gap_breaks_the_run():
    pts = fifteen([150, 130]) + [(T0 + timedelta(minutes=70), 110.0)]   # 40-min gap before the last reading
    r = evaluate(pts)
    assert not r["enough"] and r["projected_20min"] is None


def test_five_minute_run_preferred_over_fifteen():
    r = evaluate(series([130 - 4 * k for k in range(14)]))
    assert r["spacing_min"] == 5 and r["loud_tier_validated"]


def test_parity_15min_with_research_rule_on_shanghai():
    pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")
    from model.cgmacros import shanghai as S
    from model.cgmacros.alarms_r3 import B_ARMS
    from serving.alerts import flags15, segment_tail15
    if not S.RAW.exists():
        pytest.skip("ShanghaiT2DM not downloaded here")
    n = 0
    for _, f in S.frames():
        x = f["glucose"].to_numpy(float)
        pts = list(zip(f["ts"].dt.to_pydatetime(), x))
        assert len(segment_tail15(pts)) == len(pts)                    # the research segment is one 15-min run
        heads, loud, _ = flags15(x)
        assert np.array_equal(heads, S.alert20_15(x))
        assert np.array_equal(loud, B_ARMS["B2_two"](x))
        n += len(x)
    assert n > 40000


def test_fast_fall_from_above_100_gets_loud():
    """Review finding: a fast fall only got heads_up until glucose was <= 100 twice. At 3 mg/dL/min the escalation
    makes it loud at 100 (one reading earlier than the gate allows: 115 was outside it)."""
    vals = [190 - 15 * k for k in range(7)]                    # 190 -> 100
    out = evaluate(series(vals))
    assert out["alert"] == "loud" and out["direction"] == "low"
    from serving import alerts as A
    x = np.array(vals, float)
    sl = A.slope15(x)
    proj = x + sl * 20
    strict_lo = (proj < A.LO) & (x <= A.GATE_LO)
    assert not (strict_lo[-1] and strict_lo[-2])               # the old rule would NOT have been loud here
