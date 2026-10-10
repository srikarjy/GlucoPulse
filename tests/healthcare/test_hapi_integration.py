"""
Validates every resource type we emit against a real HAPI FHIR R4 server's
validator ($validate), which catches R4-vs-R4B differences that fhir.resources
(R4B) can't. Skipped when HAPI isn't reachable:
    docker compose up -d hapi-fhir
    HAPI_URL=http://localhost:8090/fhir .venv/bin/pytest tests/healthcare/test_hapi_integration.py
Best-practice warnings (narrative, performer) are tolerated; errors are not.
"""
import os
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from serving.healthcare import fhir_mapping as fm

HAPI = os.getenv("HAPI_URL", "http://localhost:8090/fhir")
T0 = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
PID = "gp-aaaaaaaaaaaaaaaaaaaaaaaa"

try:
    httpx.get(f"{HAPI}/metadata", timeout=3).raise_for_status()
except Exception:
    pytest.skip("HAPI FHIR not reachable", allow_module_level=True)


def errors(resource: dict) -> list[str]:
    r = httpx.post(f"{HAPI}/{resource['resourceType']}/$validate", json=resource, timeout=60,
                   headers={"Content-Type": fm.FHIR_JSON})
    issues = r.json().get("issue", [])
    return [i.get("diagnostics", "") for i in issues if i["severity"] in ("error", "fatal")]


def _forecast_set():
    dev = fm.device_resource("v1")
    obs = [fm.forecast_observation(PID, T0, T0 + timedelta(minutes=h), h, 140, 120, 170, dev["id"])
           for h in (30, 60)]
    return obs, dev, fm.provenance_resource(obs, dev["id"], T0)


def test_patient_valid():
    assert errors(fm.patient_resource(PID)) == []


def test_reading_observation_valid():
    assert errors(fm.observation_from_reading(PID, T0, 123.4)) == []


def test_forecast_resources_valid():
    obs, dev, prov = _forecast_set()
    for r in [*obs, dev, prov]:
        assert errors(r) == [], r["resourceType"]


def test_capability_statement_valid():
    assert errors(fm.capability_statement("http://localhost:8000/fhir", True)) == []


def test_ehr_write_back_transaction_accepted():
    from ehr.sync import EhrClient
    obs, dev, prov = _forecast_set()
    pat = {"resourceType": "Patient", "id": PID}
    out = EhrClient(HAPI).put_all([pat, *obs, dev, prov])
    assert out["type"] == "transaction-response"
