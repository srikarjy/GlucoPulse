"""Vercel entry point (api/index.py): settings come from the environment at import, ASGI lifespan is answered
without starting anything, and every request runs a fresh stateless MCP server behind the body limit."""
from __future__ import annotations

import asyncio
import importlib
import secrets
import sys

import pytest

pytest.importorskip("mcp")
from starlette.testclient import TestClient  # noqa: E402

from agents.public_mcp import tools  # noqa: E402
from agents.public_mcp.auth import issue_dev_token  # noqa: E402

ACCEPT = "application/json, text/event-stream"
ENV = ("GLUCOPULSE_MCP_AUTH_MODE", "GLUCOPULSE_MCP_RESOURCE", "GLUCOPULSE_MCP_ISSUER", "GLUCOPULSE_MCP_JWKS_URL",
       "GLUCOPULSE_MCP_DEV_SECRET")


def load_index(monkeypatch, **env):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delitem(sys.modules, "api.index", raising=False)
    return importlib.import_module("api.index")


@pytest.fixture
def audit(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "AUDIT", tmp_path / "audit.jsonl")
    tools.LIMITER.hits.clear()


def test_import_fails_without_configuration(monkeypatch):
    with pytest.raises(RuntimeError, match="DEV_SECRET"):
        load_index(monkeypatch)


def test_host_taken_from_resource(monkeypatch):
    idx = load_index(monkeypatch, GLUCOPULSE_MCP_AUTH_MODE="jwks", GLUCOPULSE_MCP_RESOURCE="https://mcp.example.org/mcp",
                     GLUCOPULSE_MCP_ISSUER="https://issuer.example.org/",
                     GLUCOPULSE_MCP_JWKS_URL="https://issuer.example.org/.well-known/jwks.json")
    assert idx._HOST == "mcp.example.org" and idx.SETTINGS.mode == "jwks"


def test_lifespan_startup_and_shutdown_acknowledged(monkeypatch):
    idx = load_index(monkeypatch, GLUCOPULSE_MCP_DEV_SECRET=secrets.token_hex(32))
    inbox = [{"type": "lifespan.startup"}, {"type": "lifespan.shutdown"}]
    sent = []

    async def receive():
        return inbox.pop(0)

    async def send(msg):
        sent.append(msg["type"])

    asyncio.run(idx._per_request({"type": "lifespan"}, receive, send))
    assert sent == ["lifespan.startup.complete", "lifespan.shutdown.complete"]


def test_jwks_deployment_rejects_missing_token(monkeypatch, audit):
    idx = load_index(monkeypatch, GLUCOPULSE_MCP_AUTH_MODE="jwks", GLUCOPULSE_MCP_RESOURCE="https://mcp.example.org/mcp",
                     GLUCOPULSE_MCP_ISSUER="https://issuer.example.org/",
                     GLUCOPULSE_MCP_JWKS_URL="https://issuer.example.org/.well-known/jwks.json")
    c = TestClient(idx.app, base_url="https://mcp.example.org")
    r = c.post("/mcp", headers={"accept": ACCEPT, "content-type": "application/json"},
               json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert r.status_code == 401
    assert "https://mcp.example.org/.well-known/oauth-protected-resource/mcp" in r.headers["www-authenticate"]


def test_each_request_serves_tools_with_a_valid_token(monkeypatch, audit):
    idx = load_index(monkeypatch, GLUCOPULSE_MCP_DEV_SECRET=secrets.token_hex(32))
    token = issue_dev_token(idx.SETTINGS)
    c = TestClient(idx.app, base_url="http://127.0.0.1:8770")
    h = {"accept": ACCEPT, "content-type": "application/json", "authorization": f"Bearer {token}"}
    for rid in (1, 2):                                       # a second request gets its own fresh server
        r = c.post("/mcp", headers=h, json={"jsonrpc": "2.0", "id": rid, "method": "tools/list"})
        assert r.status_code == 200
        assert {t["name"] for t in r.json()["result"]["tools"]} == {"get_glucose_alert", "about_glucopulse_alerts"}


def test_oversized_body_rejected_before_the_server(monkeypatch, audit):
    idx = load_index(monkeypatch, GLUCOPULSE_MCP_DEV_SECRET=secrets.token_hex(32))
    c = TestClient(idx.app, base_url="http://127.0.0.1:8770")
    r = c.post("/mcp", headers={"accept": ACCEPT, "content-type": "application/json"}, content=b"x" * (2 * 1024 * 1024))
    assert r.status_code == 413
