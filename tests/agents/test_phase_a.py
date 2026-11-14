"""
Phase A: the clinical-reasoning agent's tool layer, wrapping the EXISTING
GlucoPulse /fhir facade -- no new FHIR client, no new auth system for the
facade hop (constraint #1 of the locked scope).

Runs the facade as a real in-process FastAPI app (fhir_routes.install,
same as tests/healthcare/test_healthcare.py) wired to ehr.sync.EhrClient
via an ASGI transport -- no network socket, no torch/onnxruntime needed
(this venv doesn't have them; the facade's Patient/Observation routes
don't touch the model). 3 hardcoded questions against synthetic patients,
resolved via direct tool calls (the LLM loop is Phase B), with every call
and its result captured as a transcript.

Run from repo root inside the venv:
    .venv/bin/pytest tests/agents/test_phase_a.py -q -s
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agents.mcp_server.auth import Principal
from agents.mcp_server.tools import LocalGateway, ToolBox
from ehr.sync import EhrClient
from serving.healthcare import fhir_routes
from serving.healthcare.audit import AuditLogger

T0 = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
PATIENT_A = "gp-synth-glycemic-01"  # has data
PATIENT_B = "gp-synth-no-data-02"   # deliberately empty -- tests the abstention precursor


def series(n=24, start=120.0, step=1.0, t0=T0):
    return [(t0 + timedelta(minutes=5 * i), start + step * i) for i in range(n)]


def fake_forecast(readings):
    last = readings[-1][1]
    return {30: (last + 10, last - 5, last + 25), 60: (last + 20, last - 15, last + 55)}


@pytest.fixture
def transcript():
    log: list[dict] = []
    yield log
    print("\n\n=== Phase A transcript ===")
    print(json.dumps(log, indent=2, default=str))


@pytest.fixture
def gateway(tmp_path):
    """Builds the real FHIR facade in-process, points a real EhrClient at
    it over an ASGI transport (no sockets), and wraps that in a ToolBox +
    LocalGateway -- the same ToolBox the real MCP server would use."""
    audit = AuditLogger(path=str(tmp_path / "audit.jsonl"), pg_dsn=None)
    store = fhir_routes.InMemoryStore({PATIENT_A: series(), PATIENT_B: []})
    app = FastAPI()
    fhir_routes.install(app, audit=audit, forecast_fn=fake_forecast, store=store)

    # The facade is fail-closed (serving/healthcare/security.py); the tool
    # layer needs its own real facade bearer token, same as ehr/sync.py's
    # --token flag would supply against a live deployment.
    import os
    import time

    import jwt
    os.environ["GLUCOPULSE_AUTH_MODE"] = "hs256"
    os.environ["GLUCOPULSE_JWT_SECRET"] = "x" * 40
    os.environ["GLUCOPULSE_AUDIENCE"] = "glucopulse-fhir"
    os.environ["GLUCOPULSE_ISSUER"] = "https://auth.test"
    now = int(time.time())
    facade_token = jwt.encode(
        {"sub": "clinical-agent", "azp": "clinical-agent-client", "scope": "system/*.read",
         "iss": "https://auth.test", "aud": "glucopulse-fhir", "iat": now, "exp": now + 600},
        os.environ["GLUCOPULSE_JWT_SECRET"], algorithm="HS256")

    test_client = TestClient(app, headers={"Authorization": f"Bearer {facade_token}"})
    fhir_client = EhrClient("http://facade/fhir", http_client=test_client)
    box = ToolBox(fhir=fhir_client, audit=audit)
    gw = LocalGateway(box, Principal(subject="clinical-agent-01", role="clinical-agent",
                                     scopes=frozenset({"fhir:read", "audit:write"})))
    yield gw, audit


def test_q1_most_recent_reading(gateway, transcript):
    """Q1: What is PATIENT_A's most recent glucose reading, and when?
    The facade returns oldest-first (fhir_routes.InMemoryStore.readings),
    so the question is answered by requesting the full set and taking the
    last entry -- there's no "_sort=-date" on this facade yet, unlike
    ehr.sync.EhrClient.cgm_observations' EHR-side search."""
    gw, _ = gateway
    result = gw.invoke("fhir_search", patient=PATIENT_A, count=100)
    transcript.append({"question": "Q1: most recent reading", "tool_call": "fhir_search",
                       "args": {"patient": PATIENT_A, "count": 100}, "result": result})

    assert result["resourceType"] == "Bundle"
    assert result["total"] >= 1
    latest = result["entry"][-1]["resource"]
    assert latest["resourceType"] == "Observation"
    assert latest["valueQuantity"]["value"] == pytest.approx(143.0)  # series(): start=120, step=1, n=24 -> last=143
    transcript[-1]["answer"] = f"{latest['valueQuantity']['value']} mg/dL at {latest['effectiveDateTime']}"


def test_q2_readings_in_window(gateway, transcript):
    """Q2: List PATIENT_A's readings in the first 30 minutes of the series."""
    gw, _ = gateway
    window_end = (T0 + timedelta(minutes=30)).isoformat().replace("+00:00", "Z")
    result = gw.invoke("fhir_search", patient=PATIENT_A, date=[f"le{window_end}"], count=100)
    transcript.append({"question": "Q2: readings in a 30-min window", "tool_call": "fhir_search",
                       "args": {"patient": PATIENT_A, "date": [f"le{window_end}"]}, "result": result})

    assert result["total"] == 7  # T0, T0+5, ..., T0+30 at 5-min steps = 7 points
    transcript[-1]["answer"] = f"{result['total']} readings between {T0.isoformat()} and {window_end}"


def test_q3_no_data_abstains(gateway, transcript):
    """Q3: Is there any CGM data on file for PATIENT_B? Must abstain, not guess."""
    gw, _ = gateway
    result = gw.invoke("fhir_search", patient=PATIENT_B, count=100)
    transcript.append({"question": "Q3: data availability for a patient with none", "tool_call": "fhir_search",
                       "args": {"patient": PATIENT_B}, "result": result})

    assert result["total"] == 0
    transcript[-1]["answer"] = "No CGM Observations on file -- insufficient data, do not infer."


def test_audit_write_logs_every_qa(gateway, transcript):
    """Every answer gets written to the SAME hash-chained audit log the
    facade itself uses -- one continuous chain, not a second log."""
    gw, audit = gateway
    gw.invoke("fhir_search", patient=PATIENT_A, count=1)
    entry = gw.invoke("audit_write", action="answer_question", resource_type="Observation",
                      patient=PATIENT_A, outcome="success",
                      detail={"question": "most recent reading"}, idempotency_key="q1-001")
    transcript.append({"tool_call": "audit_write", "result": entry})

    assert entry["action"] == "answer_question" and entry["outcome"] == "success"
    rows = audit.read(10, PATIENT_A)
    assert any(r["action"] == "answer_question" for r in rows)
