"""Real MCP over streamable HTTP, driven by Strands' MCPClient, with per-role bearer tokens."""
import socket
import threading
import time

import httpx
import pytest
import uvicorn
from mcp.client.streamable_http import streamablehttp_client
from strands.tools.mcp import MCPClient

from priorauth.mcp_server.auth import issue_token
from priorauth.mcp_server.server import build_app
from priorauth.mcp_server.tools import TOOLS
from tests.priorauth.conftest import clean_meets


@pytest.fixture()
def server(box):
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    cfg = uvicorn.Config(build_app(box), host="127.0.0.1", port=port, log_level="error")
    srv = uvicorn.Server(cfg)
    t = threading.Thread(target=srv.run, daemon=True); t.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/mcp", box
    srv.should_exit = True; t.join(5)


def client(url, role):
    tok = issue_token(role)
    return MCPClient(lambda: streamablehttp_client(url, headers={"Authorization": f"Bearer {tok}"}))


def test_unauthenticated_and_bad_tokens_get_401(server):
    url, _ = server
    for h in ({}, {"Authorization": "Bearer nope"}, {"Authorization": f"Bearer {issue_token('submitter')}"}):   # API token on MCP
        r = httpx.post(url, headers={**h, "Accept": "application/json, text/event-stream"}, json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
        assert r.status_code == 401


def test_all_tools_are_discoverable_over_mcp(server):
    url, _ = server
    with client(url, "evidence-agent") as c:
        names = {t.tool_name for t in c.list_tools_sync()}
    assert names == set(TOOLS)


def test_tool_call_round_trip_and_scope_denial_over_mcp(server):
    url, box = server
    s = clean_meets()
    from priorauth.schemas import PARRequest
    box.store.create_request(PARRequest.model_validate(s["request"]))
    rid, pid = s["request"]["request_id"], s["request"]["patient_ref"]
    with client(url, "evidence-agent") as c:
        ok = c.call_tool_sync("t1", "get_patient_summary", {"patient_ref": pid})
        assert ok["status"] == "success" and "age" in ok["content"][0]["text"]
    with client(url, "intake-agent") as c:                        # no chart:read
        bad = c.call_tool_sync("t2", "get_patient_summary", {"patient_ref": pid})
        assert bad["status"] == "error" and "lacks scope" in bad["content"][0]["text"]
    with client(url, "drafting-agent") as c:
        a = c.call_tool_sync("t3", "create_draft_determination", dict(
            request_id=rid, policy_id="SYN-CGM", recommendation="insufficient_info", confidence=0.7, rationale="r",
            criteria_summary=[], flags=[], missing_information=[], idempotency_key="mcp-1"))
        b = c.call_tool_sync("t4", "create_draft_determination", dict(
            request_id=rid, policy_id="SYN-CGM", recommendation="insufficient_info", confidence=0.7, rationale="r",
            criteria_summary=[], flags=[], missing_information=[], idempotency_key="mcp-1"))
        assert a["status"] == b["status"] == "success" and a["content"] == b["content"]
        forged = c.call_tool_sync("t5", "create_draft_determination", dict(
            request_id=rid, policy_id="SYN-CGM", recommendation="approve", confidence=0.7, rationale="r",
            criteria_summary=[], flags=[], missing_information=[], idempotency_key="mcp-2"))
        assert forged["status"] == "error"
    actions = [e["action"] for e in box.store.audit_entries(limit=100)]
    assert "tool.denied" in actions and "tool.replayed" in actions
