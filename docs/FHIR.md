# FHIR R4 and EHR integration

## Resource mapping

| GlucoPulse | FHIR R4 | Notes |
|---|---|---|
| CGM reading | `Observation` | LOINC `99504-3` (Glucose [Mass/volume] in Interstitial fluid), UCUM `mg/dL`, category `laboratory`. Id is deterministic (patient + timestamp) so re-export upserts |
| Patient | `Patient` | HMAC pseudonym id only, `meta.security` = `PSEUDED`. No name/DOB/address (the dataset has none) |
| T+30 / T+60 forecast | `Observation` | `status: preliminary`, same LOINC code, `effectiveDateTime` in the future, tag `urn:glucopulse:tag#model-forecast`, 80% interval as two `component`s, `note` disclaimer. Id keyed on target time so re-forecasting overwrites |
| Model | `Device` | version-keyed |
| Forecast lineage | `Provenance` | targets the forecast Observations, agent = the Device |
| Access trail | `AuditEvent` | mapped from the hash-chained audit log |
| Server description | `CapabilityStatement`, SMART `smart-configuration` | |

Inbound mmol/L is converted (×18.016); other units, other codes and anything tagged as a forecast are skipped, never coerced. Readings outside 40–400 mg/dL (the G6 reporting range, same bound as the DLQ policy) cause a 422, not a silent clamp. Gaps in the 5-minute grid break the history: only the trailing contiguous run (≥ 6 points) is used.

> **Tag note.** HL7's `AIAST` code is the better tag, but HAPI 7.4's bundled terminology rejects it as unknown (found when validating), so a local code is used. Inbound `AIAST` is still treated as a forecast.

## Endpoints (`/fhir`)

| | |
|---|---|
| `GET /metadata` | CapabilityStatement (public) |
| `GET /.well-known/smart-configuration` | SMART discovery (public) |
| `GET /Patient/{id}` | needs `*/Patient.read` |
| `GET /Observation?patient=&date=ge..&date=le..&_count=&_page=` | needs `*/Observation.read`; `patient` required |
| `POST /Patient/{id}/$forecast` | body: Bundle of CGM Observations (or empty to use the store). Returns forecast Observations + Device + Provenance |
| `GET /AuditEvent?patient=&_count=` | needs `*/AuditEvent.read`; patients limited to their own |

Errors are `OperationOutcome`. Send `X-Purpose-Of-Use: TREAT` (etc.) to have it recorded in the audit trail.

## EHR round trip (`ehr/sync.py`)

```
EHR FHIR server → Observation search (LOINC 99504-3, paged)
   → pseudonymize patient id (HMAC) → POST /fhir/Patient/{pseudo}/$forecast
   → map pseudonym back → transaction Bundle of PUTs into the EHR
```

GlucoPulse never sees the EHR's patient id. Write-back is idempotent: 3 syncs of the same readings leave 2 forecast Observations.

```bash
python -m venv .venv && .venv/bin/pip install -r serving/requirements.txt httpx pytest "fhir.resources>=8"
docker compose up -d hapi-fhir           # local stand-in EHR, synthetic data only
.venv/bin/python -m ehr.sync --ehr http://localhost:8090/fhir --patient <id> \
    --glucopulse https://localhost:8443 --token <bearer>
```

Verified against HAPI FHIR 7.4.0 (request validation on). **Not verified** against Epic/Cerner sandboxes; those need a registered client and SMART backend-services JWT auth, which `ehr/sync.py` does not implement (it takes a pre-issued bearer token).

## Tests

```bash
.venv/bin/pytest tests/healthcare -q     # 51 tests; HAPI ones skip if HAPI is down
```
