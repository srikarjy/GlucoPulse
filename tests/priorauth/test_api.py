import pytest
from fastapi.testclient import TestClient

from priorauth.agents.orchestrator import Orchestrator
from priorauth.api import create_app
from priorauth.mcp_server.auth import issue_token
from priorauth.worker import process_one
from tests.priorauth.conftest import by_profile, clean_meets


def H(role, sub=None, **extra):
    return {"Authorization": f"Bearer {issue_token(role, sub)}", **extra}


@pytest.fixture()
def client(box):
    return TestClient(create_app(box.store))


def body(s):
    r = s["request"]
    return {k: r[k] for k in ("patient_ref", "service_code", "service_description", "diagnosis_codes", "urgency", "free_text")}


def test_submit_requires_auth_and_idempotency_key(client, scenarios):
    b = body(scenarios[0])
    assert client.post("/v1/requests", json=b).status_code == 401
    assert client.post("/v1/requests", json=b, headers=H("submitter")).status_code == 400
    assert client.post("/v1/requests", json=b, headers=H("reviewer-rn", Idempotency_Key="k")).status_code in (400, 403)


def test_submit_is_idempotent(client, scenarios):
    h = H("submitter", **{"Idempotency-Key": "abc"})
    a = client.post("/v1/requests", json=body(scenarios[0]), headers=h)
    b = client.post("/v1/requests", json=body(scenarios[0]), headers=h)
    assert a.status_code == 202 and b.status_code == 200
    assert a.json()["request_id"] == b.json()["request_id"] and b.json()["replayed"]


def test_agent_token_rejected_by_api(client):
    assert client.get("/v1/reviews", headers={"Authorization": f"Bearer {issue_token('drafting-agent')}"}).status_code == 401


def _flow(client, box, scn, reviewer="reviewer-rn"):
    r = client.post("/v1/requests", json=body(scn), headers=H("submitter", **{"Idempotency-Key": scn["scenario_id"]}))
    rid = r.json()["request_id"]
    # patient refs in the synthetic FHIR source are keyed by scenario; rebind for the run
    assert process_one(box.store, Orchestrator(box)) == "done"
    return rid


def _patch_ref(box, scn):
    # API-created requests get new ids but keep patient_ref -> chart lookup still works
    return scn


def test_end_to_end_submit_process_review_decide(client, box, scenarios):
    scn = clean_meets()
    rid = _flow(client, box, scn)
    assert client.get(f"/v1/requests/{rid}", headers=H("submitter")).json() == {"request_id": rid, "status": "awaiting_review"}   # no draft for submitter
    q = client.get("/v1/reviews", headers=H("reviewer-rn")).json()
    assert len(q) == 1 and q[0]["draft"]["status"] == "DRAFT"
    tid = q[0]["task_id"]
    # deciding without claiming is refused
    assert client.post(f"/v1/reviews/{tid}/decision", json={"decision": "approved"}, headers=H("reviewer-rn", "alice")).status_code == 409
    assert client.post(f"/v1/reviews/{tid}/claim", headers=H("reviewer-rn", "alice")).status_code == 200
    assert client.post(f"/v1/reviews/{tid}/claim", headers=H("reviewer-rn", "bob")).status_code == 409
    # bob cannot decide alice's task
    assert client.post(f"/v1/reviews/{tid}/decision", json={"decision": "approved"}, headers=H("reviewer-rn", "bob")).status_code == 409
    d = client.post(f"/v1/reviews/{tid}/decision", json={"decision": "approved", "comment": "ok"}, headers=H("reviewer-rn", "alice"))
    assert d.status_code == 200 and d.json()["agreed_with_draft"] is True
    assert client.get(f"/v1/requests/{rid}", headers=H("submitter")).json()["status"] == "decided"
    assert client.post(f"/v1/reviews/{tid}/decision", json={"decision": "denied"}, headers=H("reviewer-md", "carol")).status_code == 409   # closed


def test_denial_and_physician_tier_require_physician(client, box):
    _flow(client, box, by_profile("fail")[0])
    tid = client.get("/v1/reviews", headers=H("reviewer-rn")).json()[0]["task_id"]
    assert client.post(f"/v1/reviews/{tid}/claim", headers=H("reviewer-rn", "alice")).status_code == 403      # MD-tier
    assert client.post(f"/v1/reviews/{tid}/claim", headers=H("reviewer-md", "carol")).status_code == 200
    assert client.post(f"/v1/reviews/{tid}/decision", json={"decision": "denied"}, headers=H("reviewer-rn", "carol")).status_code == 403
    assert client.post(f"/v1/reviews/{tid}/decision", json={"decision": "denied"}, headers=H("reviewer-md", "carol")).status_code == 200


def test_override_is_recorded(client, box):
    _flow(client, box, by_profile("fail")[0])
    tid = client.get("/v1/reviews", headers=H("reviewer-rn")).json()[0]["task_id"]
    client.post(f"/v1/reviews/{tid}/claim", headers=H("reviewer-md", "carol"))
    r = client.post(f"/v1/reviews/{tid}/decision", json={"decision": "approved", "comment": "peer-to-peer"}, headers=H("reviewer-md", "carol"))
    assert r.json()["agreed_with_draft"] is False


def test_audit_requires_role_and_chain_verifies_and_detects_tampering(client, box):
    _flow(client, box, by_profile("meets")[0])
    assert client.get("/v1/audit", headers=H("reviewer-md")).status_code == 403
    assert client.get("/v1/audit/verify", headers=H("auditor")).json()["intact"] is True
    from priorauth.store import audit_t
    with box.store.engine.begin() as c:
        c.execute(audit_t.update().where(audit_t.c.seq == 2).values(entry={"tampered": True}))
    v = client.get("/v1/audit/verify", headers=H("auditor")).json()
    assert v == {"intact": False, "first_broken_seq": 2}


def test_metrics_and_health(client):
    assert client.get("/healthz").json() == {"ok": True}
    assert "pa_requests_total" in client.get("/metrics").text


def test_me_returns_role_and_scopes_and_rejects_bad_tokens(client):
    r = client.get("/v1/me", headers=H("reviewer-rn", "alice")).json()
    assert r["role"] == "reviewer-rn" and r["subject"] == "alice" and "review:decide" in r["scopes"]
    assert client.get("/v1/me").status_code == 401
    assert client.get("/v1/me", headers={"Authorization": f"Bearer {issue_token('drafting-agent')}"}).status_code == 401


def test_timeline_shows_agents_in_order_without_clinical_content(client, box):
    s = clean_meets()
    rid = _flow(client, box, s)
    t = client.get(f"/v1/requests/{rid}/timeline", headers=H("submitter")).json()
    actors = [e["actor"] for e in t if e["action"] == "tool.call"]
    assert actors[0] == "intake-agent" and actors[-1] == "escalation-agent"
    assert [e["action"] for e in t][0] == "request.received" and "draft.created" in [e["action"] for e in t]
    blob = str(t)
    assert all(n["text"] not in blob for n in s["facts"]["notes"])
    assert client.get("/v1/requests/NOPE/timeline", headers=H("submitter")).status_code == 404
    assert client.get(f"/v1/requests/{rid}/timeline").status_code == 401


def test_timeline_is_complete_on_a_busy_log(client, box):
    """Regression: audit filtering used to happen after fetching only the latest 200 rows."""
    rid = _flow(client, box, clean_meets())
    for i in range(400):
        box.store.audit("noise", "x", f"other-{i}")
    t = client.get(f"/v1/requests/{rid}/timeline", headers=H("submitter")).json()
    assert t[0]["action"] == "request.received" and any(e["action"] == "draft.created" for e in t)


def test_playground_role_can_submit_and_read_drafts_but_never_decide(client, box):
    h = H("playground", "demo", **{"Idempotency-Key": "pg1"})
    s = clean_meets()
    rid = client.post("/v1/requests", json=body(s), headers=h).json()["request_id"]
    process_one(box.store, Orchestrator(box))
    assert client.get(f"/v1/requests/{rid}", headers=H("playground")).json()["draft"]["status"] == "DRAFT"
    tid = client.get("/v1/reviews", headers=H("playground")).json()[0]["task_id"]
    assert client.post(f"/v1/reviews/{tid}/claim", headers=H("playground")).status_code == 403
    assert client.post(f"/v1/reviews/{tid}/decision", json={"decision": "approved"}, headers=H("playground")).status_code == 403


def test_demo_scenarios_endpoint_is_off_by_default_and_gated(box):
    off = TestClient(create_app(box.store, demo=False))
    assert off.get("/v1/demo/scenarios", headers=H("playground")).status_code == 404
    on = TestClient(create_app(box.store, demo=True))
    assert on.get("/v1/demo/scenarios").status_code == 401
    assert on.get("/v1/demo/scenarios", headers=H("reviewer-rn")).status_code == 403
    d = on.get("/v1/demo/scenarios", headers=H("playground")).json()
    assert len(d) == 200 and {"expected", "request", "profile"} <= set(d[0])


def test_get_single_review_includes_final_decision_once_made(client, box):
    _flow(client, box, clean_meets())
    tid = client.get("/v1/reviews", headers=H("reviewer-rn")).json()[0]["task_id"]
    t = client.get(f"/v1/reviews/{tid}", headers=H("reviewer-rn")).json()
    assert t["state"] == "open" and t["decision"] is None and t["draft"]["status"] == "DRAFT"
    client.post(f"/v1/reviews/{tid}/claim", headers=H("reviewer-rn", "alice"))
    client.post(f"/v1/reviews/{tid}/decision", json={"decision": "approved", "comment": "ok"}, headers=H("reviewer-rn", "alice"))
    t = client.get(f"/v1/reviews/{tid}", headers=H("auditor")).json()
    assert t["state"] == "closed" and t["decision"]["decision"] == "approved" and t["decision"]["reviewer"] == "alice"
    assert client.get("/v1/reviews/NOPE", headers=H("reviewer-rn")).status_code == 404
    assert client.get(f"/v1/reviews/{tid}", headers=H("submitter")).status_code == 403
