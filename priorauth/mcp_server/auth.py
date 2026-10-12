"""
MCP authentication & authorization: HS256 JWT bearer tokens carrying a role and
scopes. Each agent gets the narrowest role that works (least privilege).

HS256 + a shared secret is a deliberate simplification for a single-service
deployment; production should use an asymmetric issuer (Cognito/OIDC) -- the
`verify` seam is the only thing to swap. See docs/priorauth/SECURITY.md.
"""
from __future__ import annotations

import contextvars
import os
import time
from dataclasses import dataclass

import jwt

ALG = "HS256"

ROLE_SCOPES: dict[str, set[str]] = {
    "intake-agent": {"request:read", "validate:read"},
    "policy-agent": {"policy:read", "request:read"},
    "evidence-agent": {"chart:read", "policy:read", "validate:read", "request:read", "evidence:write"},
    "drafting-agent": {"policy:read", "draft:write", "request:read"},
    "escalation-agent": {"escalate:write", "audit:write", "request:read"},
}


API_ROLE_SCOPES: dict[str, set[str]] = {
    "submitter": {"request:submit", "request:read"},
    "reviewer-rn": {"review:read", "review:decide", "request:read"},
    "reviewer-md": {"review:read", "review:decide", "review:decide:denial", "request:read"},
    "auditor": {"audit:read", "request:read", "review:read"},
    # DEMO/EVAL ONLY: can submit and read drafts and timelines, can never claim or decide. Powers the UI playground.
    "playground": {"request:submit", "request:read", "review:read"},
}
ALL_ROLES = {**ROLE_SCOPES, **API_ROLE_SCOPES}


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
    s = os.getenv("PRIORAUTH_JWT_SECRET")
    if not s or len(s) < 32:
        raise AuthError("PRIORAUTH_JWT_SECRET must be set (>=32 chars)")
    return s


def audience_for(role: str) -> str:
    return "priorauth-mcp" if role in ROLE_SCOPES else "priorauth-api"


def issue_token(role: str, subject: str | None = None, ttl: int = 900, secret: str | None = None) -> str:
    if role not in ALL_ROLES:
        raise AuthError(f"unknown role {role}")
    now = int(time.time())
    return jwt.encode({"sub": subject or role, "role": role, "iat": now, "exp": now + ttl, "aud": audience_for(role)},
                      secret or _secret(), algorithm=ALG)


def verify(token: str, aud: str = "priorauth-mcp", secret: str | None = None) -> Principal:
    """Audience-bound: an API token cannot be replayed against the MCP server or vice versa."""
    try:
        c = jwt.decode(token, secret or _secret(), algorithms=[ALG], audience=aud,
                       options={"require": ["exp", "sub", "aud"]})
    except jwt.PyJWTError as e:
        raise AuthError(f"invalid token: {e}") from e
    role = c.get("role")
    if role not in ALL_ROLES or audience_for(role) != aud:
        raise AuthError("unknown role for audience")
    # scopes are derived from the role server-side; any `scope` claim in the token is ignored
    return Principal(subject=c["sub"], role=role, scopes=frozenset(ALL_ROLES[role]))


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
