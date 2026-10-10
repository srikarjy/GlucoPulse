"""
FHIR R4 mapping for GlucoPulse.

Plain-dict builders (no pydantic model layer at runtime) so the standalone
serving image keeps its small dependency set. Output is validated against the
FHIR R4B structure definitions in tests/healthcare via `fhir.resources`.

Mapping decisions:
- CGM reading            -> Observation, LOINC 99504-3 (Glucose [Mass/volume]
                            in Interstitial fluid), UCUM mg/dL.
- Subject                -> Patient, pseudonymous id only. No name, no
                            birthDate, no address -- the dataset has none.
- TFT forecast (T+30/60) -> Observation, status "preliminary", same LOINC code
                            (it's a predicted value of the same quantity),
                            effective time in the *future*, tagged
                            `model-forecast` with the
                            80% prediction interval carried as components,
                            plus Device (the model) and Provenance.
- Capability advertisement -> CapabilityStatement.

This validates against base R4 resource structure. It is NOT claimed to
conform to the HL7 CGM IG profiles -- that conformance check is not done.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

LOINC = "http://loinc.org"
UCUM = "http://unitsofmeasure.org"
V3_OBS_VALUE = "http://terminology.hl7.org/CodeSystem/v3-ObservationValue"
OBS_CATEGORY = "http://terminology.hl7.org/CodeSystem/observation-category"
PSEUDONYM_SYSTEM = "urn:glucopulse:pseudonym"
MODEL_SYSTEM = "urn:glucopulse:model"
TAG_SYSTEM = "urn:glucopulse:tag"
# HL7's v3 "AIAST" (AI asserted) code would be the better tag, but HAPI 7.4's
# bundled terminology rejects it as unknown (measured), so forecasts carry a
# local code. Inbound AIAST-tagged data is still treated as a forecast.
FORECAST_TAGS = {"model-forecast", "AIAST"}

CGM_LOINC = "99504-3"
CGM_DISPLAY = "Glucose [Mass/volume] in Interstitial fluid"
MGDL_PER_MMOLL = 18.016  # molar mass of glucose / 10

FHIR_JSON = "application/fhir+json"

_NS = uuid.UUID("6f1c1d8e-5f0e-4a5e-9a39-3c2f0b7a9d11")


def det_id(*parts: str) -> str:
    """Deterministic resource id: re-mapping the same reading yields the same
    Observation id, so FHIR consumers can upsert instead of duplicating."""
    return str(uuid.uuid5(_NS, "|".join(parts)))


def iso(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_dt(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _cgm_code() -> dict:
    return {"coding": [{"system": LOINC, "code": CGM_LOINC, "display": CGM_DISPLAY}]}


def _mgdl(value: float) -> dict:
    return {"value": round(float(value), 2), "unit": "mg/dL", "system": UCUM, "code": "mg/dL"}


def patient_resource(pseudonym: str) -> dict:
    return {
        "resourceType": "Patient",
        "id": pseudonym,
        "meta": {
            "security": [{
                "system": V3_OBS_VALUE, "code": "PSEUDED", "display": "pseudonymized",
            }]
        },
        "identifier": [{"system": PSEUDONYM_SYSTEM, "value": pseudonym}],
        "active": True,
    }


def observation_from_reading(pseudonym: str, ts: datetime, glucose_mgdl: float) -> dict:
    return {
        "resourceType": "Observation",
        "id": det_id("obs", pseudonym, iso(ts)),
        "status": "final",
        "category": [{
            "coding": [{"system": OBS_CATEGORY, "code": "laboratory", "display": "Laboratory"}]
        }],
        "code": _cgm_code(),
        "subject": {"reference": f"Patient/{pseudonym}"},
        "effectiveDateTime": iso(ts),
        "valueQuantity": _mgdl(glucose_mgdl),
    }


def device_resource(model_version: str) -> dict:
    return {
        "resourceType": "Device",
        "id": det_id("device", model_version),
        "identifier": [{"system": MODEL_SYSTEM, "value": f"glucopulse-tft-{model_version}"}],
        "deviceName": [{"name": "GlucoPulse Temporal Fusion Transformer", "type": "model-name"}],
        "version": [{"value": model_version}],
        "note": [{"text": (
            "Research forecasting model. Not a medical device and not "
            "clinically validated; not for diagnosis or treatment decisions."
        )}],
    }


def forecast_observation(
    pseudonym: str,
    issued: datetime,
    target_time: datetime,
    horizon_min: int,
    predicted: float,
    pi_lower: float,
    pi_upper: float,
    device_id: str,
) -> dict:
    return {
        "resourceType": "Observation",
        # Keyed on the target time, not wall-clock `issued`: re-forecasting from the
        # same readings must overwrite, not accumulate duplicates, in an EHR.
        "id": det_id("forecast", pseudonym, iso(target_time), str(horizon_min)),
        "meta": {"tag": [{"system": TAG_SYSTEM, "code": "model-forecast",
                          "display": "Model-generated forecast, not a measurement"}]},
        "status": "preliminary",
        "category": [{
            "coding": [{"system": OBS_CATEGORY, "code": "laboratory", "display": "Laboratory"}]
        }],
        "code": _cgm_code(),
        "subject": {"reference": f"Patient/{pseudonym}"},
        "effectiveDateTime": iso(target_time),
        "issued": iso(issued),
        "valueQuantity": _mgdl(predicted),
        "device": {"reference": f"Device/{device_id}"},
        "method": {"text": f"TFT forecast, T+{horizon_min} min"},
        "note": [{"text": (
            "Model forecast, not a measurement. Research model; not a medical "
            "device and not clinically validated. Insulin and carbohydrate "
            "covariates were unavailable and treated as absent.")}],
        "component": [
            {"code": {"text": "80% prediction interval lower bound (10th percentile)"},
             "valueQuantity": _mgdl(pi_lower)},
            {"code": {"text": "80% prediction interval upper bound (90th percentile)"},
             "valueQuantity": _mgdl(pi_upper)},
        ],
    }


def provenance_resource(targets: Iterable[dict], device_id: str, recorded: datetime) -> dict:
    return {
        "resourceType": "Provenance",
        "id": det_id("prov", *[t["id"] for t in targets]),
        "target": [{"reference": f"Observation/{t['id']}"} for t in targets],
        "recorded": iso(recorded),
        "agent": [{
            "type": {"coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/provenance-participant-type",
                "code": "assembler", "display": "Assembler"}]},
            "who": {"reference": f"Device/{device_id}"},
        }],
    }


def bundle(entries: list[dict], bundle_type: str = "collection", *,
           total: Optional[int] = None, links: Optional[list[dict]] = None) -> dict:
    out: dict[str, Any] = {
        "resourceType": "Bundle",
        "id": str(uuid.uuid4()),
        "type": bundle_type,
        "timestamp": iso(datetime.now(timezone.utc)),
        "entry": [{"fullUrl": f"urn:uuid:{e['id']}", "resource": e} for e in entries],
    }
    if total is not None:
        out["total"] = total
    if links:
        out["link"] = links
    return out


def operation_outcome(severity: str, code: str, diagnostics: str) -> dict:
    return {
        "resourceType": "OperationOutcome",
        "issue": [{"severity": severity, "code": code, "diagnostics": diagnostics}],
    }


def capability_statement(base_url: str, auth_enabled: bool) -> dict:
    def res(rtype: str, interactions: list[str], extra: Optional[dict] = None) -> dict:
        r = {"type": rtype, "interaction": [{"code": c} for c in interactions]}
        if extra:
            r.update(extra)
        return r

    return {
        "resourceType": "CapabilityStatement",
        "status": "active",
        "date": iso(datetime.now(timezone.utc)),
        "kind": "instance",
        "software": {"name": "GlucoPulse", "version": "1.1.0"},
        "implementation": {"description": "GlucoPulse CGM forecasting FHIR facade", "url": base_url},
        "fhirVersion": "4.0.1",
        "format": ["json"],
        "rest": [{
            "mode": "server",
            "security": {
                "service": [{"coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/restful-security-service",
                    "code": "SMART-on-FHIR"}]}] if auth_enabled else [],
                "description": "SMART Backend Services / bearer JWT. Scopes enforced per resource.",
            },
            "resource": [
                res("Patient", ["read"]),
                res("Observation", ["search-type"], {
                    "searchParam": [
                        {"name": "patient", "type": "reference"},
                        {"name": "date", "type": "date"},
                        {"name": "_count", "type": "number"},
                    ]}),
                res("AuditEvent", ["search-type"]),
            ],
            "operation": [{
                "name": "forecast",
                "definition": "urn:glucopulse:OperationDefinition/forecast",
            }],
        }],
    }


def smart_configuration(issuer: str, jwks_uri: Optional[str],
                        auth_endpoint: Optional[str], token_endpoint: Optional[str]) -> dict:
    conf: dict[str, Any] = {
        "issuer": issuer,
        "grant_types_supported": ["client_credentials", "authorization_code"],
        "token_endpoint_auth_methods_supported": ["private_key_jwt"],
        "scopes_supported": [
            "launch", "launch/patient", "openid", "fhirUser",
            "patient/Patient.read", "patient/Observation.read",
            "user/Patient.read", "user/Observation.read",
            "system/Patient.read", "system/Observation.read",
            "system/Observation.write", "system/AuditEvent.read",
        ],
        "response_types_supported": ["code"],
        "capabilities": [
            "launch-standalone", "client-confidential-asymmetric",
            "context-standalone-patient", "permission-patient", "permission-user",
            "permission-v1",
        ],
        "code_challenge_methods_supported": ["S256"],
    }
    if jwks_uri:
        conf["jwks_uri"] = jwks_uri
    if auth_endpoint:
        conf["authorization_endpoint"] = auth_endpoint
    if token_endpoint:
        conf["token_endpoint"] = token_endpoint
    return conf


def readings_from_bundle(bundle_json: dict) -> list[tuple[datetime, float]]:
    """Extract (time, mg/dL) from a Bundle of CGM Observations, oldest first.

    Accepts mg/dL and mmol/L (UCUM) -- EHRs outside the US report mmol/L.
    Anything else (wrong code, no value, unknown unit) is skipped, not coerced.
    Forecast Observations (AIAST-tagged) are skipped so a round-tripped bundle
    can't feed the model its own predictions.
    """
    out: dict[datetime, float] = {}
    for entry in bundle_json.get("entry", []):
        r = entry.get("resource", {})
        if r.get("resourceType") != "Observation":
            continue
        if any(t.get("code") in FORECAST_TAGS for t in r.get("meta", {}).get("tag", [])):
            continue
        codes = {c.get("code") for c in r.get("code", {}).get("coding", [])
                 if c.get("system") == LOINC}
        if CGM_LOINC not in codes:
            continue
        q, when = r.get("valueQuantity"), r.get("effectiveDateTime")
        if not q or not when or "value" not in q:
            continue
        unit = q.get("code") or q.get("unit")
        value = float(q["value"])
        if unit == "mmol/L":
            value *= MGDL_PER_MMOLL
        elif unit != "mg/dL":
            continue
        try:
            out[parse_dt(when)] = value
        except ValueError:
            continue
    return sorted(out.items())
