"""
MCP authentication & authorization for the clinical-reasoning agent's own
tool server -- same mechanism as priorauth/mcp_server/auth.py (HS256 JWT,
role -> scopes, audience-bound, scopes never read from the token), under
its own namespace/secret so the two tool servers can't replay tokens
against each other.

This is a SEPARATE credential from the GlucoPulse FHIR facade's bearer
token (serving/healthcare/security.py): this one authenticates "is this
caller allowed to invoke this MCP tool", the facade one authenticates
"is this caller allowed to read this FHIR resource". The tool
implementations in tools.py hold a facade token themselves and pass it
through ehr.sync.EhrClient -- there is still only one FHIR-facing auth
scheme, just used at two hops.
"""
from __future__ import annotations

import contextvars
import os
import time
from dataclasses import dataclass

import jwt

ALG = "HS256"
AUDIENCE = "clinical-agent-mcp"

ROLE_SCOPES: dict[str, set[str]] = {
    # Deliberately excludes "review:decide": the agent drafts, it never decides.
    "clinical-agent": {"fhir:read", "audit:write"},
    # Deliberately excludes "fhir:read": a reviewer judges a finding that's
    # already been drafted and handed to them -- they don't need, and
    # shouldn't be routed around the agent to, direct FHIR access via this
    # tool surface.
    "reviewer": {"audit:read", "review:decide"},
}


class AuthError(Exception):
    pass


class ForbiddenError(Exception):
    pass


@dataclass(frozen=True)
class Principal:
    subject: str
    role: str
    scopes: frozenset[str]


_current: contextvars.ContextVar[Principal | None] = contextvars.ContextVar("principal", default=None)


def current_principal() -> Principal:
    p = _current.get()
    if p is None:
        raise AuthError("no authenticated principal")
    return p


def set_principal(p: Principal | None):
    return _current.set(p)


def reset_principal(token) -> None:
    _current.reset(token)


def _secret() -> str:
    s = os.getenv("CLINICAL_AGENT_JWT_SECRET")
    if not s or len(s) < 32:
        raise AuthError("CLINICAL_AGENT_JWT_SECRET must be set (>=32 chars)")
    return s


def issue_token(role: str, subject: str | None = None, ttl: int = 900, secret: str | None = None) -> str:
    if role not in ROLE_SCOPES:
        raise AuthError(f"unknown role {role}")
    now = int(time.time())
    return jwt.encode({"sub": subject or role, "role": role, "iat": now, "exp": now + ttl, "aud": AUDIENCE},
                      secret or _secret(), algorithm=ALG)


def verify(token: str, secret: str | None = None) -> Principal:
    try:
        c = jwt.decode(token, secret or _secret(), algorithms=[ALG], audience=AUDIENCE,
                       options={"require": ["exp", "sub", "aud"]})
    except jwt.PyJWTError as e:
        raise AuthError(f"invalid token: {e}") from e
    role = c.get("role")
    if role not in ROLE_SCOPES:
        raise AuthError("unknown role")
    # scopes are derived from the role server-side; any `scope` claim in the token is ignored
    return Principal(subject=c["sub"], role=role, scopes=frozenset(ROLE_SCOPES[role]))


class BearerASGI:
    """ASGI middleware: 401 without a valid bearer token; binds the principal for the request."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        hdr = dict(scope["headers"]).get(b"authorization", b"").decode()
        try:
            if not hdr.lower().startswith("bearer "):
                raise AuthError("missing bearer token")
            p = verify(hdr[7:])
        except AuthError:
            body = b'{"error":"unauthorized"}'
            await send({"type": "http.response.start", "status": 401,
                        "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
            await send({"type": "http.response.body", "body": body})
            return
        tok = set_principal(p)
        try:
            await self.app(scope, receive, send)
        finally:
            reset_principal(tok)
