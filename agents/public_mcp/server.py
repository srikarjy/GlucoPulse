"""Public GlucoPulse MCP server for ChatGPT and Claude (streamable HTTP, OAuth resource server).

    GLUCOPULSE_MCP_DEV_SECRET=... python -m agents.public_mcp.server          # local, dev tokens, 127.0.0.1:8770

Tools: get_glucose_alert, about_glucopulse_alerts (agents/public_mcp/tools.py); both read-only.
Security: bearer token on every request (agents/public_mcp/auth.py), 401 + WWW-Authenticate pointing at
/.well-known/oauth-protected-resource/mcp, DNS-rebinding protection (Host/Origin allowlists), request bodies capped at
256 KB, per-subject rate limit, metadata-only audit log. Production: put it behind HTTPS and set
GLUCOPULSE_MCP_AUTH_MODE=jwks with your IdP (see docs/AI_CONNECTORS.md).
"""
from __future__ import annotations

import os
from urllib.parse import urlparse

from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations

from agents.public_mcp import tools
from agents.public_mcp.auth import SCOPE, Settings, Verifier

MAX_BODY = 256 * 1024
BAD_ARGS = ("invalid arguments: send readings as a list of {timestamp: ISO 8601 with timezone offset, glucose: "
            "number}, unit 'mg/dL' or 'mmol/L', mode 'current' or 'retrospective'")


class SafeFastMCP(FastMCP):
    """Error messages never echo caller input back to the model. The SDK's validation errors quote the offending
    value (e.g. text smuggled into a number field), which would hand injected text back to ChatGPT/Claude; only our
    own fixed tool messages, or a generic fix-it message, leave the server."""

    async def call_tool(self, name, arguments):
        try:
            return await super().call_tool(name, arguments)
        except Exception as e:
            cause = e
            while cause is not None and not isinstance(cause, tools.ToolError):
                cause = cause.__cause__ or cause.__context__
            raise tools.ToolError(str(cause) if cause is not None else BAD_ARGS) from None
READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)


def build_mcp(settings: Settings, host: str = "127.0.0.1", port: int = 8770) -> FastMCP:
    if settings.mode == "dev" and host not in ("127.0.0.1", "localhost", "::1"):
        raise RuntimeError("dev tokens are for local use only: bind to 127.0.0.1 or use jwks mode")
    if settings.mode == "jwks":
        from pydantic import AnyHttpUrl
        if str(AnyHttpUrl(settings.issuer)) != settings.issuer:
            # the protected-resource metadata must list the IdP's issuer EXACTLY (ChatGPT checks); pydantic would
            # silently add a trailing slash to a bare-host issuer
            raise RuntimeError(f"GLUCOPULSE_MCP_ISSUER must be written exactly as the IdP's issuer, e.g. "
                               f"{AnyHttpUrl(settings.issuer)}")
    public = urlparse(settings.resource)
    allowed_hosts = ["127.0.0.1:*", "localhost:*", "[::1]:*"] + ([public.netloc] if public.netloc else [])
    allowed_origins = ["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*",
                       "https://chatgpt.com", "https://claude.ai"] + (
        [f"{public.scheme}://{public.netloc}"] if public.netloc else [])
    mcp = SafeFastMCP(
        "glucopulse", stateless_http=True, json_response=True, host=host, port=port,
        token_verifier=Verifier(settings),
        auth=AuthSettings(issuer_url=settings.issuer if settings.mode == "jwks" else "https://glucopulse-dev.invalid",
                          resource_server_url=settings.resource, required_scopes=[SCOPE],
                          validate_token_resource=True),
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=True,
                                                     allowed_hosts=allowed_hosts, allowed_origins=allowed_origins),
        instructions="GlucoPulse checks CGM glucose readings for a coming low or high. Results are screening "
                     "information only: never use them for insulin dosing; tell the person to confirm with a meter "
                     "and follow their care plan. Quote accuracy only from about_glucopulse_alerts.")
    mcp.add_tool(tools.get_glucose_alert, title="Check glucose readings for a coming low or high",
                 annotations=READ_ONLY)
    mcp.add_tool(tools.about_glucopulse_alerts, title="How GlucoPulse alerts work and were tested",
                 annotations=READ_ONLY)
    return mcp


class BodyLimit:
    """Reject requests whose body exceeds MAX_BODY bytes, declared or streamed (chunked). The body is read and checked
    BEFORE the MCP app sees the request (MCP messages are small JSON), so an oversized upload never reaches the
    handler and the 413 is the only response sent."""

    def __init__(self, app, limit: int = MAX_BODY):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") in ("GET", "HEAD", "OPTIONS"):
            return await self.app(scope, receive, send)
        declared = dict(scope.get("headers") or []).get(b"content-length")
        if declared and declared.isdigit() and int(declared) > self.limit:
            return await self._too_large(send)
        chunks, size = [], 0
        while True:
            msg = await receive()
            if msg["type"] == "http.disconnect":
                return
            body = msg.get("body", b"")
            size += len(body)
            if size > self.limit:
                return await self._too_large(send)
            chunks.append(body)
            if not msg.get("more_body"):
                break
        buffered = [{"type": "http.request", "body": b"".join(chunks), "more_body": False}]

        async def replay():
            return buffered.pop() if buffered else await receive()
        await self.app(scope, replay, send)

    @staticmethod
    async def _too_large(send):
        body = b'{"error":"request body too large"}'
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})


def build_app(settings: Settings | None = None, host: str = "127.0.0.1", port: int = 8770):
    mcp = build_mcp(settings or Settings.from_env(), host, port)
    return BodyLimit(mcp.streamable_http_app()), mcp


def main() -> None:  # pragma: no cover
    import uvicorn
    host = os.getenv("GLUCOPULSE_MCP_HOST", "127.0.0.1")
    port = int(os.getenv("GLUCOPULSE_MCP_PORT", "8770"))
    app, mcp = build_app(host=host, port=port)
    uvicorn.run(app, host=host, port=port, lifespan="on")


if __name__ == "__main__":  # pragma: no cover
    main()
