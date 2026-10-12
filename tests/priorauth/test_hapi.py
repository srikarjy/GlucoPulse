"""Live FHIR test: synthetic resources are loaded into a real HAPI R4 server, then read back through HttpFhirSource
and the full agent pipeline. Skipped unless PRIORAUTH_TEST_FHIR_URL is set, e.g.
  docker run -d -p 18090:8080 hapiproject/hapi:v7.4.0 ; PRIORAUTH_TEST_FHIR_URL=http://localhost:18090/fhir pytest tests/priorauth/test_hapi.py
Writes ONLY synthetic data, to a throwaway local server."""
import os

import httpx
import pytest

from priorauth.agents.orchestrator import Orchestrator
from priorauth.fhir import HttpFhirSource, facts_to_fhir, parse_chart
from priorauth.mcp_server.tools import ToolBox
from priorauth.rag import PolicyRetriever
from priorauth.schemas import PARRequest
from priorauth.store import Store

URL = os.getenv("PRIORAUTH_TEST_FHIR_URL")
pytestmark = pytest.mark.skipif(not URL, reason="PRIORAUTH_TEST_FHIR_URL not set")


def load_into_hapi(pid: str, resources: list[dict]) -> None:
    bundle = {"resourceType": "Bundle", "type": "transaction", "entry": [
        {"resource": r, "request": {"method": "PUT", "url": f"{r['resourceType']}/{r['id']}"}} for r in resources]}
    r = httpx.post(URL, json=bundle, headers={"Content-Type": "application/fhir+json"}, timeout=60)
    assert r.status_code == 200, r.text[:300]


def test_http_source_round_trips_every_scenario_type_through_real_hapi(scenarios):
    picks = {}
    for s in scenarios:
        picks.setdefault(s["profile"], s)
    src = HttpFhirSource(URL)
    for prof, s in picks.items():
        pid = "hapi-" + s["request"]["patient_ref"]
        res = [{**r, "id": r["id"].replace(s["request"]["patient_ref"], pid)} for r in facts_to_fhir(s["request"]["patient_ref"], s["facts"])]
        for r in res:                                                    # re-point references at the renamed patient
            if "subject" in r:
                r["subject"] = {"reference": f"Patient/{pid}"}
            if "beneficiary" in r:
                r["beneficiary"] = {"reference": f"Patient/{pid}"}
        load_into_hapi(pid, res)
        got = parse_chart(pid, src.fetch(pid))
        want = parse_chart(pid, res)
        assert got.model_dump() == want.model_dump(), prof
    assert src.fetch("does-not-exist") == []


def test_full_pipeline_reads_chart_from_real_fhir_server(scenarios):
    s = next(x for x in scenarios if x["profile"] == "meets" and not x["paraphrased_evidence"])
    pid = "hapi-pipe-" + s["request"]["patient_ref"]
    res = facts_to_fhir(pid, s["facts"])
    load_into_hapi(pid, res)
    box = ToolBox(Store(), HttpFhirSource(URL), PolicyRetriever())
    req = {**s["request"], "patient_ref": pid, "request_id": "REQ-HAPI-1"}
    box.store.create_request(PARRequest.model_validate(req))
    Orchestrator(box).run("REQ-HAPI-1")
    t = box.store.get_task("TASK-REQ-HAPI-1")
    assert t.draft.recommendation.value == "supports_approval" and t.state == "open"
