"""
Healthcare layer tests. Run from repo root inside the venv:
    .venv/bin/pytest tests/healthcare -q
No torch/onnx/DB needed: the model and store are injected.
"""

import json
import time
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fhir.resources.R4B.bundle import Bundle
from fhir.resources.R4B.capabilitystatement import CapabilityStatement
from fhir.resources.R4B.observation import Observation
from fhir.resources.R4B.patient import Patient

from serving.healthcare import fhir_mapping as fm
from serving.healthcare import fhir_routes
from serving.healthcare.audit import AuditLogger, verify_chain
from serving.healthcare.deid import (IdentifierRejected, pseudonymize,
                                     reject_direct_identifiers, validate_patient_id)

SECRET = "x" * 40
T0 = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
PID = "gp-aaaaaaaaaaaaaaaaaaaaaaaa"
OTHER = "gp-bbbbbbbbbbbbbbbbbbbbbbbb"


def series(n=24, start=120.0, step=1.0, t0=T0):
    return [(t0 + timedelta(minutes=5 * i), start + step * i) for i in range(n)]


def fake_forecast(readings):
    last = readings[-1][1]
    return {30: (last + 10, last - 5, last + 25), 60: (last + 20, last - 15, last + 55)}


@pytest.fixture(autouse=True)
def auth_env(monkeypatch, tmp_path):
    monkeypatch.setenv("GLUCOPULSE_AUTH_MODE", "hs256")
    monkeypatch.setenv("GLUCOPULSE_JWT_SECRET", SECRET)
    monkeypatch.setenv("GLUCOPULSE_AUDIENCE", "glucopulse-fhir")
    monkeypatch.setenv("GLUCOPULSE_ISSUER", "https://auth.test")
    monkeypatch.setenv("GLUCOPULSE_PSEUDONYM_KEY", "k" * 40)
    monkeypatch.delenv("PG_HOST", raising=False)


@pytest.fixture
def ctx(tmp_path):
    audit = AuditLogger(path=str(tmp_path / "audit.jsonl"), pg_dsn=None)
    store = fhir_routes.InMemoryStore({PID: series(), OTHER: series()})
    app = FastAPI()
    fhir_routes.install(app, audit=audit, forecast_fn=fake_forecast, store=store)
    return TestClient(app), audit, tmp_path / "audit.jsonl"


def token(scope, patient=None, ttl=600, **over):
    now = int(time.time())
    claims = {"sub": "dr-test", "azp": "client-1", "scope": scope, "iss": "https://auth.test",
              "aud": "glucopulse-fhir", "iat": now, "exp": now + ttl}
    if patient:
        claims["patient"] = patient
    claims.update(over)
    return {"Authorization": "Bearer " + jwt.encode(claims, SECRET, algorithm="HS256")}


# ---------------- FHIR structure ----------------

def test_observation_validates_as_fhir_r4():
    Observation.model_validate(fm.observation_from_reading(PID, T0, 123.4))


def test_patient_validates_and_has_no_direct_identifiers():
    p = fm.patient_resource(PID)
    Patient.model_validate(p)
    assert not ({"name", "birthDate", "address", "telecom"} & p.keys())


def test_forecast_bundle_validates():
    dev = fm.device_resource("v1")
    obs = [fm.forecast_observation(PID, T0, T0 + timedelta(minutes=30), 30, 140, 120, 170, dev["id"])]
    prov = fm.provenance_resource(obs, dev["id"], T0)
    Bundle.model_validate(fm.bundle(obs + [dev, prov]))


def test_forecast_ids_stable_across_runs():
    dev = fm.device_resource("v1")
    mk = lambda issued: fm.forecast_observation(  # noqa: E731
        PID, issued, T0 + timedelta(minutes=30), 30, 140, 120, 170, dev["id"])
    a, b = mk(T0), mk(T0 + timedelta(minutes=2))
    assert a["id"] == b["id"]
    assert fm.provenance_resource([a], dev["id"], T0)["id"] == fm.provenance_resource([b], dev["id"], T0)["id"]


def test_capability_statement_validates():
    CapabilityStatement.model_validate(fm.capability_statement("http://x/fhir", True))


def test_mmol_converted_and_forecasts_not_fed_back():
    b = {"entry": [
        {"resource": {"resourceType": "Observation",
                      "code": {"coding": [{"system": fm.LOINC, "code": fm.CGM_LOINC}]},
                      "effectiveDateTime": "2024-01-15T10:00:00Z",
                      "valueQuantity": {"value": 6.0, "code": "mmol/L"}}},
        {"resource": {"resourceType": "Observation",
                      "meta": {"tag": [{"code": "AIAST"}]},
                      "code": {"coding": [{"system": fm.LOINC, "code": fm.CGM_LOINC}]},
                      "effectiveDateTime": "2024-01-15T10:05:00Z",
                      "valueQuantity": {"value": 999, "code": "mg/dL"}}},
        {"resource": {"resourceType": "Observation",
                      "code": {"coding": [{"system": fm.LOINC, "code": "1234-5"}]},
                      "effectiveDateTime": "2024-01-15T10:10:00Z",
                      "valueQuantity": {"value": 100, "code": "mg/dL"}}},
    ]}
    out = fm.readings_from_bundle(b)
    assert len(out) == 1 and out[0][1] == pytest.approx(108.096)


# ---------------- authn / authz ----------------

def test_fails_closed_without_auth_config(ctx, monkeypatch):
    client, *_ = ctx
    monkeypatch.delenv("GLUCOPULSE_AUTH_MODE")
    r = client.get(f"/fhir/Patient/{PID}", headers=token("system/*.read"))
    assert r.status_code == 401
    assert r.json()["resourceType"] == "OperationOutcome"


def test_no_token_401_and_metadata_public(ctx):
    client, *_ = ctx
    assert client.get(f"/fhir/Patient/{PID}").status_code == 401
    assert client.get("/fhir/metadata").status_code == 200
    assert client.get("/fhir/.well-known/smart-configuration").status_code == 200


@pytest.mark.parametrize("bad", [
    {"aud": "someone-else"}, {"iss": "https://evil"}, {"exp": int(time.time()) - 10},
])
def test_bad_claims_rejected(ctx, bad):
    client, *_ = ctx
    r = client.get(f"/fhir/Patient/{PID}", headers=token("system/*.read", **bad))
    assert r.status_code == 401


def test_overlong_token_lifetime_rejected(ctx):
    client, *_ = ctx
    r = client.get(f"/fhir/Patient/{PID}", headers=token("system/*.read", ttl=7200))
    assert r.status_code == 401


def test_wrong_signature_and_alg_none_rejected(ctx):
    client, *_ = ctx
    forged = jwt.encode({"sub": "x", "scope": "system/*.read", "aud": "glucopulse-fhir",
                         "iss": "https://auth.test", "iat": int(time.time()),
                         "exp": int(time.time()) + 60}, "y" * 40, algorithm="HS256")
    assert client.get(f"/fhir/Patient/{PID}", headers={"Authorization": f"Bearer {forged}"}).status_code == 401
    unsigned = jwt.encode({"sub": "x", "scope": "system/*.read"}, None, algorithm="none")
    assert client.get(f"/fhir/Patient/{PID}", headers={"Authorization": f"Bearer {unsigned}"}).status_code == 401


def test_scope_required_per_resource(ctx):
    client, *_ = ctx
    h = token("system/Patient.read")
    assert client.get(f"/fhir/Patient/{PID}", headers=h).status_code == 200
    assert client.get(f"/fhir/Observation?patient={PID}", headers=h).status_code == 403


def test_patient_scope_confined_to_launch_patient(ctx):
    client, *_ = ctx
    h = token("patient/Patient.read patient/Observation.read", patient=PID)
    assert client.get(f"/fhir/Patient/{PID}", headers=h).status_code == 200
    assert client.get(f"/fhir/Patient/{OTHER}", headers=h).status_code == 403
    assert client.get(f"/fhir/Observation?patient={OTHER}", headers=h).status_code == 403
    # unknown id gives the same 403 -> no patient enumeration
    assert client.get("/fhir/Patient/gp-doesnotexist", headers=h).status_code == 403


def test_user_scope_unknown_patient_404(ctx):
    client, *_ = ctx
    assert client.get("/fhir/Patient/gp-nope", headers=token("user/Patient.read")).status_code == 404


def test_write_scope_does_not_grant_read(ctx):
    client, *_ = ctx
    assert client.get(f"/fhir/Patient/{PID}", headers=token("system/Patient.write")).status_code == 403


# ---------------- Observation search ----------------

def test_search_paging_and_dates(ctx):
    client, *_ = ctx
    h = token("system/Observation.read")
    r = client.get(f"/fhir/Observation?patient={PID}&_count=10", headers=h)
    body = r.json()
    Bundle.model_validate(body)
    assert body["total"] == 24 and len(body["entry"]) == 10
    assert any(l["relation"] == "next" for l in body["link"])
    ge = (T0 + timedelta(minutes=100)).isoformat().replace("+00:00", "Z")
    r = client.get(f"/fhir/Observation?patient={PID}&date=ge{ge}", headers=h)
    assert r.json()["total"] == 4
    assert client.get(f"/fhir/Observation?patient={PID}&date=gt{ge}", headers=h).status_code == 400
    assert client.get("/fhir/Observation", headers=h).status_code == 422  # patient required


def test_response_content_type_is_fhir_json(ctx):
    client, *_ = ctx
    r = client.get(f"/fhir/Patient/{PID}", headers=token("system/Patient.read"))
    assert r.headers["content-type"].startswith("application/fhir+json")
    assert r.headers["cache-control"] == "no-store"


# ---------------- $forecast ----------------

def _bundle(readings, pid=PID):
    return fm.bundle([fm.observation_from_reading(pid, t, v) for t, v in readings])


def test_forecast_from_bundle(ctx):
    client, *_ = ctx
    h = token("system/Observation.read")
    r = client.post(f"/fhir/Patient/{PID}/$forecast", json=_bundle(series(12)), headers=h)
    assert r.status_code == 200
    out = r.json()
    Bundle.model_validate(out)
    kinds = [e["resource"]["resourceType"] for e in out["entry"]]
    assert kinds == ["Observation", "Observation", "Device", "Provenance"]
    f = out["entry"][0]["resource"]
    assert f["status"] == "preliminary" and f["meta"]["tag"][0]["code"] == "model-forecast"
    assert f["effectiveDateTime"] == fm.iso(series(12)[-1][0] + timedelta(minutes=30))


def test_forecast_from_store_when_no_body(ctx):
    client, *_ = ctx
    r = client.post(f"/fhir/Patient/{PID}/$forecast", headers=token("system/Observation.read"))
    assert r.status_code == 200


def test_forecast_refuses_gap_short_and_implausible(ctx):
    client, *_ = ctx
    h = token("system/Observation.read")
    gapped = series(10) + series(3, t0=T0 + timedelta(hours=3))
    assert client.post(f"/fhir/Patient/{PID}/$forecast", json=_bundle(gapped), headers=h).status_code == 422
    assert client.post(f"/fhir/Patient/{PID}/$forecast", json=_bundle(series(4)), headers=h).status_code == 422
    bad = series(10)
    bad[3] = (bad[3][0], 900.0)
    assert client.post(f"/fhir/Patient/{PID}/$forecast", json=_bundle(bad), headers=h).status_code == 422


def test_forecast_rejects_other_patients_data(ctx):
    client, *_ = ctx
    r = client.post(f"/fhir/Patient/{PID}/$forecast", json=_bundle(series(12), pid=OTHER),
                    headers=token("system/Observation.read"))
    assert r.status_code == 403


def test_contiguous_run_takes_only_trailing_segment():
    gapped = series(10) + series(7, t0=T0 + timedelta(hours=3))
    assert len(fhir_routes.trailing_contiguous(gapped)) == 7


# ---------------- audit ----------------

def test_every_outcome_is_audited_without_phi(ctx):
    client, audit, path = ctx
    h = token("patient/Patient.read", patient=PID)
    client.get(f"/fhir/Patient/{PID}", headers=h)            # success
    client.get(f"/fhir/Patient/{OTHER}", headers=h)          # denied
    client.get(f"/fhir/Patient/{PID}")                       # unauthenticated
    client.get("/fhir/Patient/gp-x", headers=token("user/Patient.read"))  # 404
    rows = audit.read(10)
    assert [r["outcome"] for r in rows] == ["success", "denied", "denied", "error"]
    assert rows[2]["actor"] == "anonymous"
    blob = path.read_text()
    assert "120.0" not in blob and "glucose" not in blob.lower()


def test_purpose_of_use_recorded(ctx):
    client, audit, _ = ctx
    client.get(f"/fhir/Patient/{PID}", headers={**token("system/Patient.read"), "X-Purpose-Of-Use": "TREAT"})
    assert audit.read(1)[0]["purpose_of_use"] == "TREAT"


def test_audit_chain_detects_tampering(ctx):
    client, audit, path = ctx
    h = token("system/Patient.read")
    for _ in range(4):
        client.get(f"/fhir/Patient/{PID}", headers=h)
    assert verify_chain(str(path)) == (True, None)
    lines = path.read_text().splitlines()
    row = json.loads(lines[1])
    row["actor"] = "someone-else"
    lines[1] = json.dumps(row, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")
    assert verify_chain(str(path)) == (False, 2)


def test_audit_chain_survives_restart(tmp_path):
    p = str(tmp_path / "a.jsonl")
    a = AuditLogger(path=p, pg_dsn=None)
    for i in range(3):
        a.record(actor="u", client_id=None, action="read", resource_type="Patient",
                 patient=PID, outcome="success")
    b = AuditLogger(path=p, pg_dsn=None)
    b.record(actor="u", client_id=None, action="read", resource_type="Patient",
             patient=PID, outcome="success")
    assert verify_chain(p) == (True, None)


def test_audit_event_endpoint_patient_scoped(ctx):
    client, *_ = ctx
    client.get(f"/fhir/Patient/{PID}", headers=token("system/Patient.read"))
    own = token("patient/AuditEvent.read", patient=PID)
    assert client.get(f"/fhir/AuditEvent?patient={PID}", headers=own).status_code == 200
    assert client.get(f"/fhir/AuditEvent?patient={OTHER}", headers=own).status_code == 403
    assert client.get("/fhir/AuditEvent", headers=own).status_code == 403
    r = client.get("/fhir/AuditEvent", headers=token("system/AuditEvent.read"))
    assert r.status_code == 200
    Bundle.model_validate(r.json())


# ---------------- de-identification ----------------

def test_pseudonym_is_stable_keyed_and_opaque(monkeypatch):
    a = pseudonymize("Subject 1")
    assert a == pseudonymize("Subject 1") and a != pseudonymize("Subject 2")
    assert "Subject" not in a
    monkeypatch.setenv("GLUCOPULSE_PSEUDONYM_KEY", "z" * 40)
    assert pseudonymize("Subject 1") != a


def test_pseudonymize_requires_key(monkeypatch):
    monkeypatch.delenv("GLUCOPULSE_PSEUDONYM_KEY")
    with pytest.raises(RuntimeError):
        pseudonymize("1")


@pytest.mark.parametrize("val", [
    "jane@example.com", "617-555-0100", "123-45-6789", "192.168.1.1",
    "https://x.org/p", "01/02/1980", "1980-02-01", "12345678",
])
def test_identifier_screen_flags(val):
    with pytest.raises(IdentifierRejected):
        reject_direct_identifiers(val)


@pytest.mark.parametrize("val", ["gp-abc123", "Subject_7", "12"])
def test_identifier_screen_passes(val):
    assert validate_patient_id(val) == val


@pytest.mark.parametrize("val", ["a b", "../etc", "x" * 65, "a@b.com"])
def test_patient_id_validation_rejects(val):
    with pytest.raises(IdentifierRejected):
        validate_patient_id(val)
