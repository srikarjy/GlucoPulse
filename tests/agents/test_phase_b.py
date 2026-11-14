"""
Phase B: the Strands agent loop, wired to the REAL clinical-agent MCP
server (agents.mcp_server.server) over real streamable HTTP, with a
ScriptedModel standing in for Bedrock -- same technique and same honesty
as priorauth/tests/priorauth/test_strands_backend.py: this proves the
wiring and the server-side re-grounding (citation + missing-data rules),
NOT LLM quality. A live-model eval is a separate, future step.

Requires strands/mcp, which are only installed in .venv-pa, not the main
.venv. Run from repo root:
    .venv-pa/bin/python -m pytest tests/agents/test_phase_b.py -q -s
"""
from __future__ import annotations

import json
import os
import secrets
import socket
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

os.environ.setdefault("CLINICAL_AGENT_JWT_SECRET", secrets.token_hex(32))

import uvicorn
from strands.models import Model

from agents.mcp_server.server import build_app
from agents.mcp_server.tools import ToolBox
from agents.reasoning_agent import ask
from ehr.sync import EhrClient
from serving.healthcare import fhir_routes
from serving.healthcare.audit import AuditLogger

T0 = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
PATIENT_A = "gp-synth-glycemic-01"
PATIENT_B = "gp-synth-no-data-02"


def series(n=24, start=120.0, step=1.0, t0=T0):
    return [(t0 + timedelta(minutes=5 * i), start + step * i) for i in range(n)]


def fake_forecast(readings):
    last = readings[-1][1]
    return {30: (last + 10, last - 5, last + 25), 60: (last + 20, last - 15, last + 55)}


class ScriptedModel(Model):
    """Replays: for each call to stream(), the next scripted turn; a turn
    is ('tool', name, input) or ('text', str). Identical shape to
    priorauth's test double -- this is generic Strands test plumbing, not
    project-specific."""

    def __init__(self, turns):
        self.turns, self.i = list(turns), 0

    def update_config(self, **kw): ...
    def get_config(self): return {}

    async def structured_output(self, output_model, prompt, system_prompt=None, **kw):  # pragma: no cover
        raise NotImplementedError

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kw):
        kind, *rest = self.turns[self.i]
        self.i += 1
        yield {"messageStart": {"role": "assistant"}}
        if kind == "text":
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": rest[0]}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
        else:
            name, inp = rest
            yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": uuid.uuid4().hex[:8], "name": name}}}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(inp)}}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        yield {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}, "metrics": {"latencyMs": 1}}}


@pytest.fixture()
def box(tmp_path):
    audit = AuditLogger(path=str(tmp_path / "audit.jsonl"), pg_dsn=None)
    store = fhir_routes.InMemoryStore({PATIENT_A: series(), PATIENT_B: []})
    app = FastAPI()

    os.environ["GLUCOPULSE_AUTH_MODE"] = "hs256"
    os.environ["GLUCOPULSE_JWT_SECRET"] = "x" * 40
    os.environ["GLUCOPULSE_AUDIENCE"] = "glucopulse-fhir"
    os.environ["GLUCOPULSE_ISSUER"] = "https://auth.test"
    fhir_routes.install(app, audit=audit, forecast_fn=fake_forecast, store=store)

    import jwt
    now = int(time.time())
    facade_token = jwt.encode(
        {"sub": "clinical-agent", "azp": "clinical-agent-client", "scope": "system/*.read",
         "iss": "https://auth.test", "aud": "glucopulse-fhir", "iat": now, "exp": now + 600},
        os.environ["GLUCOPULSE_JWT_SECRET"], algorithm="HS256")

    test_client = TestClient(app, headers={"Authorization": f"Bearer {facade_token}"})
    fhir_client = EhrClient("http://facade/fhir", http_client=test_client)
    return ToolBox(fhir=fhir_client, audit=audit)


@pytest.fixture()
def mcp_url(box):
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    cfg = uvicorn.Config(build_app(box), host="127.0.0.1", port=port, log_level="error")
    srv = uvicorn.Server(cfg)
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/mcp"
    srv.should_exit = True
    t.join(5)


def _last_obs_id(box) -> str:
    """The real id the facade assigns -- citations must point at a resource
    the tools actually returned (agents.citation rejects anything else)."""
    from agents.mcp_server.auth import Principal
    from agents.mcp_server.tools import LocalGateway
    gw = LocalGateway(box, Principal(subject="test-reader", role="clinical-agent",
                                     scopes=frozenset({"fhir:read"})))
    bundle = gw.invoke("fhir_search", patient=PATIENT_A, count=100)
    return f"Observation/{bundle['entry'][-1]['resource']['id']}"


def test_grounded_finding_survives_regrounding(mcp_url, box, capsys):
    """A properly cited, single-sentence finding passes through unchanged."""
    obs_id = _last_obs_id(box)
    turns = [
        ("tool", "fhir_search", {"patient": PATIENT_A, "count": 100}),
        ("text", json.dumps({
            "finding": "The patient's most recent glucose reading was 143 mg/dL.",
            "evidence": [{"claim": "The patient's most recent glucose reading was 143 mg/dL.",
                         "fhir_resource_id": obs_id, "kb_source": None}],
            "confidence": "high", "missing_data": [], "red_flags": [],
        })),
    ]
    finding, dropped = ask(mcp_url, "What was the patient's most recent reading?",
                           model_factory=lambda: ScriptedModel(turns))
    print("\n=== grounded case ===", finding.model_dump(), dropped)
    assert dropped == []
    assert finding.finding == "The patient's most recent glucose reading was 143 mg/dL."
    assert len(finding.evidence) == 1
    assert "dropped" not in " ".join(finding.red_flags)


def test_uncited_sentence_is_dropped(mcp_url, box, capsys):
    """The citation rule: a second, uncited sentence must not survive."""
    obs_id = _last_obs_id(box)
    turns = [
        ("tool", "fhir_search", {"patient": PATIENT_A, "count": 100}),
        ("text", json.dumps({
            "finding": "The patient's most recent glucose reading was 143 mg/dL. "
                      "The patient is at high risk of complications.",
            "evidence": [{"claim": "The patient's most recent glucose reading was 143 mg/dL.",
                         "fhir_resource_id": obs_id, "kb_source": None}],
            "confidence": "med", "missing_data": [], "red_flags": [],
        })),
    ]
    finding, dropped = ask(mcp_url, "Summarize the patient's glycemic status.",
                           model_factory=lambda: ScriptedModel(turns))
    print("\n=== uncited-sentence case ===", finding.model_dump(), dropped)
    assert dropped == ["The patient is at high risk of complications."]
    assert finding.finding == "The patient's most recent glucose reading was 143 mg/dL."
    assert any("dropped_1_uncited_sentence" in f for f in finding.red_flags)


def test_no_data_patient_abstains_correctly(mcp_url, capsys):
    """Missing-data rule: a patient with zero CGM data must abstain, not
    guess -- and a correct abstention (missing_data already declared)
    must NOT trigger the additional 'no grounded finding' red flag."""
    turns = [
        ("tool", "fhir_search", {"patient": PATIENT_B, "count": 100}),
        ("text", json.dumps({
            "finding": "", "evidence": [], "confidence": "low",
            "missing_data": ["No CGM Observations on file for this patient."],
            "red_flags": [],
        })),
    ]
    finding, dropped = ask(mcp_url, "Summarize this patient's glycemic control.",
                           model_factory=lambda: ScriptedModel(turns))
    print("\n=== no-data abstention case ===", finding.model_dump(), dropped)
    assert dropped == []
    assert finding.finding == ""
    assert finding.missing_data == ["No CGM Observations on file for this patient."]
    assert "no_grounded_finding_and_no_missing_data_declared" not in finding.red_flags


def test_asserts_without_citing_or_abstaining_is_caught(mcp_url, capsys):
    """The failure mode the missing-data rule exists to catch: the agent
    asserts something, cites nothing, and doesn't admit a gap either."""
    turns = [
        ("tool", "fhir_search", {"patient": PATIENT_A, "count": 100}),
        ("text", json.dumps({
            "finding": "The patient is well controlled.",
            "evidence": [], "confidence": "high", "missing_data": [], "red_flags": [],
        })),
    ]
    finding, dropped = ask(mcp_url, "Is this patient well controlled?",
                           model_factory=lambda: ScriptedModel(turns))
    print("\n=== uncited-and-no-abstention case ===", finding.model_dump(), dropped)
    assert finding.finding == ""
    assert "no_grounded_finding_and_no_missing_data_declared" in finding.red_flags
    assert finding.missing_data  # forced non-empty
