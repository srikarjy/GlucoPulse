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


class EhrClient:
    def __init__(self, base_url: str, token: Optional[str] = None, timeout: float = 30.0):
        headers = {"Accept": fm.FHIR_JSON}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.base = base_url.rstrip("/")
        self.http = httpx.Client(headers=headers, timeout=timeout)

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

    pseudo = pseudonymize(patient_id)
    # Outgoing: strip to the fields the model needs, re-point subject at the pseudonym.
    outgoing = {"resourceType": "Bundle", "type": "collection", "entry": [
        {"resource": {
            "resourceType": "Observation",
            "code": e["resource"].get("code"),
            "subject": {"reference": f"Patient/{pseudo}"},
            "effectiveDateTime": e["resource"].get("effectiveDateTime"),
            "valueQuantity": e["resource"].get("valueQuantity"),
        }} for e in entries]}

    r = httpx.post(f"{glucopulse.rstrip('/')}/fhir/Patient/{pseudo}/$forecast", json=outgoing,
                   headers={"Authorization": f"Bearer {gp_token}", "X-Purpose-Of-Use": "TREAT",
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
