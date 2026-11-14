"""
EHR <-> GlucoPulse round trip over FHIR R4.

    EHR (FHIR server)  --Observation search-->  this script
          ^                                        |  pseudonymize patient id
          |                                        v
          |                          GlucoPulse  POST /fhir/Patient/{pseudo}/$forecast
          |                                        |
          +----- conditional PUT (idempotent) -----+  forecast Observation + Device + Provenance

The patient id is replaced with a keyed pseudonym before anything leaves for
GlucoPulse and mapped back on the way in, so the forecasting service's logs
and audit trail never hold the EHR's real patient id.

Idempotent: forecast resources have deterministic ids, so re-running for the
same readings overwrites rather than duplicates.

Usage (venv):
    GLUCOPULSE_PSEUDONYM_KEY=... python -m ehr.sync \
        --ehr http://localhost:8090/fhir --patient <id> \
        --glucopulse http://localhost:8000 --token <glucopulse bearer> \
        [--ehr-token <ehr bearer>] [--dry-run]

Only test it against sandboxes with synthetic patients (HAPI, SMART). Pointing
it at a production EHR needs a BAA, a registered backend-services client, and
a review that is outside what this script is.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

import httpx

from serving.healthcare import fhir_mapping as fm
from serving.healthcare.deid import pseudonymize

CGM_TOKEN = f"{fm.LOINC}|{fm.CGM_LOINC}"
MAX_PAGES = 20
INVALID_STATUSES = {"entered-in-error", "cancelled"}


def _is_usable_reading(resource: dict) -> bool:
    """Exclude EHR-side corrections and GlucoPulse's own prior forecasts
    written back on an earlier sync -- without this filter, $forecast's
    output becomes $forecast's next input on the following run, and the
    FORECAST_TAGS skip in fhir_mapping.readings_from_bundle never gets a
    chance to fire because this script's outgoing bundle (below) doesn't
    carry status/meta.tag through anyway."""
    if resource.get("status") in INVALID_STATUSES:
        return False
    tags = {t.get("code") for t in resource.get("meta", {}).get("tag", [])}
    return not (tags & fm.FORECAST_TAGS)


class EhrClient:
    def __init__(self, base_url: str, token: Optional[str] = None, timeout: float = 30.0,
                 http_client: Optional[httpx.Client] = None):
        """`http_client` lets a caller inject an already-configured sync
        httpx.Client (e.g. fastapi.testclient.TestClient against an
        in-process ASGI app) instead of a real socket -- used by the
        clinical-reasoning agent's tool-layer tests so it has one FHIR HTTP
        client, not a second one, whether it's hitting a live deployment or
        an in-process test facade. When supplied, its own headers/auth are
        used as-is; `token` is ignored."""
        self.base = base_url.rstrip("/")
        if http_client is not None:
            self.http = http_client
            return
        headers = {"Accept": fm.FHIR_JSON}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.http = httpx.Client(headers=headers, timeout=timeout)

    def get_resource(self, resource_type: str, resource_id: str) -> dict:
        r = self.http.get(f"{self.base}/{resource_type}/{resource_id}")
        r.raise_for_status()
        return r.json()

    def search(self, resource_type: str, **params) -> dict:
        r = self.http.get(f"{self.base}/{resource_type}", params=params)
        r.raise_for_status()
        return r.json()

    def cgm_observations(self, patient_id: str, count: int = 100) -> list[dict]:
        """Newest-first search, following `next` links; returned oldest-first."""
        url: Optional[str] = f"{self.base}/Observation"
        params: Optional[dict] = {"patient": patient_id, "code": CGM_TOKEN,
                                  "_sort": "-date", "_count": count}
        out: list[dict] = []
        for _ in range(MAX_PAGES):
            r = self.http.get(url, params=params)
            r.raise_for_status()
            page = r.json()
            out += [e for e in page.get("entry", []) if "resource" in e]
            nxt = next((l["url"] for l in page.get("link", []) if l.get("relation") == "next"), None)
            if not nxt or len(out) >= count:
                break
            url, params = nxt, None
        return list(reversed(out[:count]))

    def put_all(self, resources: list[dict]) -> dict:
        """One transaction Bundle of PUTs (create-or-update by id)."""
        tx = {"resourceType": "Bundle", "type": "transaction", "entry": [
            {"fullUrl": f"{self.base}/{r['resourceType']}/{r['id']}", "resource": r,
             "request": {"method": "PUT", "url": f"{r['resourceType']}/{r['id']}"}}
            for r in resources]}
        r = self.http.post(self.base, json=tx, headers={"Content-Type": fm.FHIR_JSON})
        r.raise_for_status()
        return r.json()


def sync_patient(ehr: EhrClient, patient_id: str, glucopulse: str, gp_token: str,
                 dry_run: bool = False) -> dict:
    entries = ehr.cgm_observations(patient_id)
    if not entries:
        raise SystemExit(f"No CGM Observations (LOINC {fm.CGM_LOINC}) for Patient/{patient_id}")

    usable = [e for e in entries if _is_usable_reading(e["resource"])]
    if not usable:
        raise SystemExit(
            f"All {len(entries)} Observation(s) for Patient/{patient_id} are forecasts or "
            "invalid-status; nothing real to forecast from.")

    pseudo = pseudonymize(patient_id)
    # Outgoing: strip to the fields the model needs, re-point subject at the pseudonym.
    outgoing = {"resourceType": "Bundle", "type": "collection", "entry": [
        {"resource": {
            "resourceType": "Observation",
            "code": e["resource"].get("code"),
            "subject": {"reference": f"Patient/{pseudo}"},
            "effectiveDateTime": e["resource"].get("effectiveDateTime"),
            "valueQuantity": e["resource"].get("valueQuantity"),
        }} for e in usable]}

    # Not TREAT: this call asks GlucoPulse to *simulate* a forecast, not to
    # treat the patient -- device_resource's own note says "not for
    # diagnosis or treatment decisions", so claiming TREAT here would
    # contradict it in the audit trail.
    r = httpx.post(f"{glucopulse.rstrip('/')}/fhir/Patient/{pseudo}/$forecast", json=outgoing,
                   headers={"Authorization": f"Bearer {gp_token}",
                            "Content-Type": fm.FHIR_JSON}, timeout=60)
    if r.status_code != 200:
        raise SystemExit(f"GlucoPulse refused: {r.status_code} "
                         f"{r.json().get('issue', [{}])[0].get('diagnostics', r.text)}")

    # Incoming: map the pseudonym back to the EHR's patient id.
    back = json.loads(json.dumps(r.json()).replace(f"Patient/{pseudo}", f"Patient/{patient_id}"))
    resources = [e["resource"] for e in back["entry"]]
    if dry_run:
        return {"dry_run": True, "resources": resources}
    return {"written": [f"{x['resourceType']}/{x['id']}" for x in resources],
            "ehr_response_type": ehr.put_all(resources).get("type")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ehr", required=True, help="EHR FHIR base URL")
    ap.add_argument("--patient", required=True, help="Patient id in the EHR")
    ap.add_argument("--glucopulse", default="http://localhost:8000")
    ap.add_argument("--token", required=True, help="GlucoPulse bearer token")
    ap.add_argument("--ehr-token")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    result = sync_patient(EhrClient(a.ehr, a.ehr_token), a.patient, a.glucopulse, a.token, a.dry_run)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
