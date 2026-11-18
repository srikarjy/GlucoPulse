"""Vercel entry point for the public GlucoPulse MCP server (agents/public_mcp). See docs/AI_CONNECTORS.md.

Serverless functions may not run ASGI lifespan events, and the MCP SDK's session manager must be started inside one
(`session_manager.run()`, once per server object). So each request builds a fresh, stateless server and runs it inside
its own session manager: correct regardless of how the platform reuses instances, and cheap (two small tools).

Required environment (Vercel project settings, never in the repo):
  GLUCOPULSE_MCP_AUTH_MODE=jwks, GLUCOPULSE_MCP_RESOURCE=https://<deployment>/mcp, GLUCOPULSE_MCP_ISSUER,
  GLUCOPULSE_MCP_JWKS_URL, GLUCOPULSE_MCP_AUDIT_KEY (>= 32 random chars), GLUCOPULSE_MCP_AUDIT_LOG=stdout.
Rate limits are per function instance on a serverless host (see docs/AI_CONNECTORS.md, "Not done yet").
"""
from __future__ import annotations

from urllib.parse import urlparse

from agents.public_mcp.auth import Settings
from agents.public_mcp.server import BodyLimit, build_mcp

SETTINGS = Settings.from_env()
_HOST = urlparse(SETTINGS.resource).hostname or "127.0.0.1"


async def _per_request(scope, receive, send):
    if scope["type"] == "lifespan":                     # nothing to start globally
        while True:
            msg = await receive()
            if msg["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif msg["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return
    mcp = build_mcp(SETTINGS, host="0.0.0.0" if SETTINGS.mode == "jwks" else "127.0.0.1")
    inner = mcp.streamable_http_app()
    async with mcp.session_manager.run():
        await inner(scope, receive, send)


app = BodyLimit(_per_request)
