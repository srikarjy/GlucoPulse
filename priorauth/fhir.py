"""
FHIR R4 integration: synthetic-resource builder, chart sources, and mapper.

`FhirSource` is the seam to an EHR. `InMemoryFhirSource` serves synthetic
resources for tests/eval; `HttpFhirSource` speaks FHIR REST (HAPI, or any
SMART-on-FHIR-style server given a bearer token). Both feed the same mapper, so
the mapper is exercised on real FHIR JSON either way.
"""
from __future__ import annotations

import base64
from datetime import date, timedelta
from typing import Protocol

import httpx
from pydantic import BaseModel, Field

AS_OF_DEFAULT = date(2026, 1, 15)

DRUG_CLASS = {
    "ibuprofen": "nsaid", "naproxen": "nsaid", "metformin": "metformin", "insulin glargine": "insulin",
    "insulin lispro": "insulin", "methotrexate": "methotrexate", "triamcinolone injection": "corticosteroid-injection",
}


# ---------- internal chart model ----------
class ChartCondition(BaseModel):
    code: str
    onset: date | None = None


class ChartMed(BaseModel):
    name: str
    drug_class: str
    start: date | None = None
    end: date | None = None


class ChartObs(BaseModel):
    loinc: str
    value: float
    day: date


class ChartNote(BaseModel):
    note_id: str
    day: date | None
    text: str


class Chart(BaseModel):
    patient_ref: str
    age: int | None = None
    conditions: list[ChartCondition] = Field(default_factory=list)
    meds: list[ChartMed] = Field(default_factory=list)
    obs: list[ChartObs] = Field(default_factory=list)
    notes: list[ChartNote] = Field(default_factory=list)
    coverage_active: bool | None = None
    as_of: date = AS_OF_DEFAULT


# ---------- synthetic facts -> FHIR ----------
def _iso(as_of: date, days_ago: int) -> str:
    return (as_of - timedelta(days=days_ago)).isoformat()


def facts_to_fhir(pid: str, facts: dict, as_of: date = AS_OF_DEFAULT) -> list[dict]:
    res: list[dict] = [{
        "resourceType": "Patient", "id": pid, "gender": facts["gender"],
        "birthDate": date(as_of.year - facts["age"] - 1, 6, 1).isoformat(),
    }]
    for i, c in enumerate(facts["conditions"]):
        res.append({"resourceType": "Condition", "id": f"{pid}-c{i}", "subject": {"reference": f"Patient/{pid}"},
                    "code": {"coding": [{"system": "http://hl7.org/fhir/sid/icd-10-cm", "code": c["code"]}]},
                    "onsetDateTime": _iso(as_of, c["onset_days_ago"])})
    for i, m in enumerate(facts["meds"]):
        period = {"start": _iso(as_of, m["start_days_ago"])}
        if m.get("end_days_ago") is not None:
            period["end"] = _iso(as_of, m["end_days_ago"])
        res.append({"resourceType": "MedicationStatement", "id": f"{pid}-m{i}", "status": "active",
                    "subject": {"reference": f"Patient/{pid}"},
                    "medicationCodeableConcept": {"text": m["name"]}, "effectivePeriod": period})
    for i, o in enumerate(facts["obs"]):
        res.append({"resourceType": "Observation", "id": f"{pid}-o{i}", "status": "final",
                    "subject": {"reference": f"Patient/{pid}"},
                    "code": {"coding": [{"system": "http://loinc.org", "code": o["loinc"]}]},
                    "valueQuantity": {"value": o["value"]}, "effectiveDateTime": _iso(as_of, o["days_ago"])})
    for i, n in enumerate(facts["notes"]):
        res.append({"resourceType": "DocumentReference", "id": f"{pid}-n{i}", "status": "current",
                    "subject": {"reference": f"Patient/{pid}"}, "date": _iso(as_of, n["days_ago"]) + "T00:00:00Z",
                    "content": [{"attachment": {"contentType": "text/plain",
                                                "data": base64.b64encode(n["text"].encode()).decode()}}]})
    res.append({"resourceType": "Coverage", "id": f"{pid}-cov", "status": "active" if facts["coverage_active"] else "cancelled",
                "beneficiary": {"reference": f"Patient/{pid}"}})
    return res


# ---------- FHIR -> chart ----------
def _d(s: str | None) -> date | None:
    return date.fromisoformat(s[:10]) if s else None


def parse_chart(pid: str, resources: list[dict], as_of: date = AS_OF_DEFAULT) -> Chart:
    chart = Chart(patient_ref=pid, as_of=as_of)
    for r in resources:
        t = r.get("resourceType")
        if t == "Patient" and r.get("birthDate"):
            b = _d(r["birthDate"])
            chart.age = as_of.year - b.year - ((as_of.month, as_of.day) < (b.month, b.day))
        elif t == "Condition":
            code = (r.get("code", {}).get("coding") or [{}])[0].get("code")
            if code:
                chart.conditions.append(ChartCondition(code=code, onset=_d(r.get("onsetDateTime"))))
        elif t in ("MedicationStatement", "MedicationRequest"):
            name = (r.get("medicationCodeableConcept", {}).get("text") or "").lower()
            per = r.get("effectivePeriod") or {}
            chart.meds.append(ChartMed(name=name, drug_class=DRUG_CLASS.get(name, "other"),
                                       start=_d(per.get("start") or r.get("authoredOn")), end=_d(per.get("end"))))
        elif t == "Observation":
            code = (r.get("code", {}).get("coding") or [{}])[0].get("code")
            val = r.get("valueQuantity", {}).get("value")
            if code and val is not None and r.get("effectiveDateTime"):
                chart.obs.append(ChartObs(loinc=code, value=float(val), day=_d(r["effectiveDateTime"])))
        elif t == "DocumentReference":
            for c in r.get("content", []):
                data = c.get("attachment", {}).get("data")
                if data:
                    chart.notes.append(ChartNote(note_id=f"DocumentReference/{r.get('id')}", day=_d(r.get("date")),
                                                 text=base64.b64decode(data).decode("utf-8", "replace")))
        elif t == "Coverage":
            chart.coverage_active = r.get("status") == "active"
    return chart


# ---------- sources ----------
class FhirSource(Protocol):
    def fetch(self, patient_ref: str) -> list[dict]: ...


class InMemoryFhirSource:
    def __init__(self):
        self._store: dict[str, list[dict]] = {}

    def put(self, patient_ref: str, resources: list[dict]) -> None:
        self._store[patient_ref] = resources

    def fetch(self, patient_ref: str) -> list[dict]:
        return self._store.get(patient_ref, [])


class HttpFhirSource:
    """Read-only FHIR REST client. Never writes to the EHR."""
    TYPES = ["Condition", "MedicationStatement", "MedicationRequest", "Observation", "DocumentReference", "Coverage"]

    def __init__(self, base_url: str, token: str | None = None, timeout: float = 15.0):
        h = {"Accept": "application/fhir+json"}
        if token:
            h["Authorization"] = f"Bearer {token}"
        self._c = httpx.Client(base_url=base_url.rstrip("/"), headers=h, timeout=timeout)

    def fetch(self, patient_ref: str) -> list[dict]:
        out: list[dict] = []
        r = self._c.get(f"/Patient/{patient_ref}")
        if r.status_code == 404:
            return []
        r.raise_for_status()
        out.append(r.json())
        for t in self.TYPES:
            url = f"/{t}"
            params = {"patient": patient_ref, "_count": "200"}
            while url:
                rr = self._c.get(url, params=params)
                rr.raise_for_status()
                b = rr.json()
                out += [e["resource"] for e in b.get("entry", [])]
                url = next((l["url"] for l in b.get("link", []) if l["relation"] == "next"), None)
                params = None
        return out
