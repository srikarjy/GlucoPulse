"""
Phase D: full audit wiring and the human-review scope boundary.

Confirms, with a test (not an assertion in prose): every tool call, every
retrieved resource id, every draft finding, and every final review
decision lands in the SAME hash-chained audit log -- and that the
clinical-agent role cannot call submit_review_decision itself (scope
boundary enforced server-side, not by agent discipline).

Run from repo root (needs strands/mcp):
    .venv-pa/bin/python -m pytest tests/agents/test_phase_d.py -q -s
"""
from __future__ import annotations

import os
import secrets
import time
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

os.environ.setdefault("CLINICAL_AGENT_JWT_SECRET", secrets.token_hex(32))

from agents.mcp_server.auth import ForbiddenError, Principal
from agents.mcp_server.tools import LocalGateway, ToolBox
from ehr.sync import EhrClient
from serving.healthcare import fhir_routes
from serving.healthcare.audit import AuditLogger, verify_chain

T0 = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
PATIENT = "gp-eval-d-01"


def series(n=24, start=120.0, step=1.0, t0=T0):
    return [(t0 + timedelta(minutes=5 * i), start + step * i) for i in range(n)]


@pytest.fixture()
def box_and_audit_path(tmp_path):
    audit_path = str(tmp_path / "audit.jsonl")
    audit = AuditLogger(path=audit_path, pg_dsn=None)
    store = fhir_routes.InMemoryStore({PATIENT: series()})
    app = FastAPI()

    os.environ["GLUCOPULSE_AUTH_MODE"] = "hs256"
    os.environ["GLUCOPULSE_JWT_SECRET"] = "x" * 40
    os.environ["GLUCOPULSE_AUDIENCE"] = "glucopulse-fhir"
    os.environ["GLUCOPULSE_ISSUER"] = "https://auth.test"
    fhir_routes.install(app, audit=audit, forecast_fn=lambda r: {30: (1, 1, 1), 60: (1, 1, 1)}, store=store)

    now = int(time.time())
    facade_token = jwt.encode(
        {"sub": "clinical-agent", "azp": "c", "scope": "system/*.read", "iss": "https://auth.test",
         "aud": "glucopulse-fhir", "iat": now, "exp": now + 600},
        os.environ["GLUCOPULSE_JWT_SECRET"], algorithm="HS256")
    test_client = TestClient(app, headers={"Authorization": f"Bearer {facade_token}"})
    fhir_client = EhrClient("http://facade/fhir", http_client=test_client)
    box = ToolBox(fhir=fhir_client, audit=audit)
    return box, audit, audit_path


def test_full_chain_tool_call_draft_and_review_decision(box_and_audit_path):
    """One continuous hash chain across a tool call, a drafted finding
    (recorded via audit_write), and a human's review decision."""
    box, audit, audit_path = box_and_audit_path
    agent_gw = LocalGateway(box, Principal(subject="clinical-agent-01", role="clinical-agent",
                                          scopes=frozenset({"fhir:read", "audit:write"})))
    reviewer_gw = LocalGateway(box, Principal(subject="dr-smith", role="reviewer",
                                             scopes=frozenset({"audit:read", "review:decide"})))

    # 1. Agent retrieves data (a tool call).
    search_result = agent_gw.invoke("fhir_search", patient=PATIENT, count=100)
    assert search_result["total"] == 24

    # 2. Agent's draft finding gets logged (stands in for reasoning_agent.ask()'s
    # grounded output -- Phase B/C already prove the grounding logic itself).
    draft = {"finding": "Most recent reading 143 mg/dL.", "evidence": [
        {"claim": "Most recent reading 143 mg/dL.",
         "fhir_resource_id": f"Observation/{search_result['entry'][-1]['resource']['id']}"}],
        "confidence": "high", "missing_data": [], "red_flags": []}
    agent_gw.invoke("audit_write", action="draft_finding", resource_type="ClinicalFinding",
                                  patient=PATIENT, outcome="success", detail=draft,
                                  idempotency_key="draft-001")

    # 3. A human reviewer approves it.
    reviewer_gw.invoke("submit_review_decision", patient=PATIENT, reviewer="dr-smith",
                                        decision="approve", finding=draft, idempotency_key="decision-001")

    # All three actions are in the SAME chain, in order, each linking to the previous hash.
    rows = audit.read(10, PATIENT)
    actions = [r["action"] for r in rows]
    assert "draft_finding" in actions and "review_decision" in actions
    ok, bad_line = verify_chain(audit_path)
    assert ok is True and bad_line is None

    draft_row = next(r for r in rows if r["action"] == "draft_finding")
    decision_row = next(r for r in rows if r["action"] == "review_decision")
    assert decision_row["prev_hash"] != draft_row["prev_hash"]  # chain actually advanced
    assert decision_row["outcome"] == "approve"
    assert decision_row["detail"]["reviewer"] == "dr-smith"


def test_agent_role_cannot_submit_review_decisions(box_and_audit_path):
    """Scope boundary enforced server-side: the clinical-agent role has no
    review:decide scope, so it cannot call this tool even if it tried --
    this is not agent discipline, it's a hard denial."""
    box, audit, _ = box_and_audit_path
    agent_gw = LocalGateway(box, Principal(subject="clinical-agent-01", role="clinical-agent",
                                          scopes=frozenset({"fhir:read", "audit:write"})))
    with pytest.raises(ForbiddenError):
        agent_gw.invoke("submit_review_decision", patient=PATIENT, reviewer="clinical-agent-01",
                        decision="approve", finding={})


def test_edit_without_edited_finding_rejected(box_and_audit_path):
    box, audit, _ = box_and_audit_path
    reviewer_gw = LocalGateway(box, Principal(subject="dr-smith", role="reviewer",
                                             scopes=frozenset({"audit:read", "review:decide"})))
    from agents.mcp_server.tools import ToolError
    with pytest.raises(ToolError):
        reviewer_gw.invoke("submit_review_decision", patient=PATIENT, reviewer="dr-smith",
                           decision="edit", finding={})
