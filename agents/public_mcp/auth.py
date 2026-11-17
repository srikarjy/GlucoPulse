"""Who may call the public GlucoPulse MCP server (the one ChatGPT and Claude connect to).

This server is an OAuth 2.1 RESOURCE SERVER only (MCP authorization spec): it never issues tokens to real users. It
validates bearer tokens from an external identity provider and publishes /.well-known/oauth-protected-resource (done
by the MCP SDK from AuthSettings) so clients can find that provider. Separate from the internal clinical-agent MCP
server (agents/mcp_server): different audience, different keys, different tools; an internal token is rejected here.

Modes (GLUCOPULSE_MCP_AUTH_MODE):
  jwks  production. Tokens from your IdP (e.g. Auth0), checked against its JWKS: signature (RS256/ES256), issuer
        (GLUCOPULSE_MCP_ISSUER), audience = this server's resource URL (GLUCOPULSE_MCP_RESOURCE), expiry, and the
        scope glucose:alerts.
  dev   local testing only (e.g. Claude Code with --header). HS256 tokens minted by `python -m
        agents.public_mcp.auth --dev-token`, secret GLUCOPULSE_MCP_DEV_SECRET (>= 32 chars), lifetime <= 12 h.
        Refused unless the server binds to localhost.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import jwt
from mcp.server.auth.provider import AccessToken

SCOPE = "glucose:alerts"
DEV_ISSUER = "glucopulse-dev"
MAX_DEV_TTL = 12 * 3600


@dataclass(frozen=True)
class Settings:
    mode: str
    resource: str               # canonical URL of this MCP endpoint, e.g. https://mcp.example.org/mcp
    issuer: str
    jwks_url: str | None = None
    dev_secret: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        mode = os.getenv("GLUCOPULSE_MCP_AUTH_MODE", "dev")
        resource = os.getenv("GLUCOPULSE_MCP_RESOURCE", "http://127.0.0.1:8770/mcp")
        if mode == "jwks":
            issuer, jwks = os.getenv("GLUCOPULSE_MCP_ISSUER"), os.getenv("GLUCOPULSE_MCP_JWKS_URL")
            if not issuer or not jwks:
                raise RuntimeError("jwks mode needs GLUCOPULSE_MCP_ISSUER and GLUCOPULSE_MCP_JWKS_URL")
            if not resource.startswith("https://"):
                raise RuntimeError("jwks mode needs an https:// GLUCOPULSE_MCP_RESOURCE")
            return cls("jwks", resource, issuer, jwks_url=jwks)
        if mode == "dev":
            secret = os.getenv("GLUCOPULSE_MCP_DEV_SECRET", "")
            if len(secret) < 32:
                raise RuntimeError("dev mode needs GLUCOPULSE_MCP_DEV_SECRET (>= 32 chars)")
            return cls("dev", resource, DEV_ISSUER, dev_secret=secret)
        raise RuntimeError(f"unknown GLUCOPULSE_MCP_AUTH_MODE {mode!r} (dev or jwks)")


def _scopes(claims: dict) -> list[str]:
    s = claims.get("scope") or claims.get("scp") or []
    if isinstance(s, str):
        s = s.split()
    perms = claims.get("permissions") or []           # Auth0 RBAC puts API permissions here
    return sorted(set(s) | set(perms))


class Verifier:
    """MCP SDK TokenVerifier: returns an AccessToken for a valid token, None otherwise (the SDK then answers 401 with
    a WWW-Authenticate header pointing at the protected-resource metadata)."""

    def __init__(self, settings: Settings):
        self.s = settings
        self._jwks = jwt.PyJWKClient(settings.jwks_url, cache_keys=True) if settings.mode == "jwks" else None

    def decode(self, token: str) -> dict:
        common = {"audience": self.s.resource, "issuer": self.s.issuer,
                  "options": {"require": ["exp", "iat", "sub", "aud", "iss"]}, "leeway": 30}
        if self.s.mode == "jwks":
            key = self._jwks.get_signing_key_from_jwt(token).key
            return jwt.decode(token, key, algorithms=["RS256", "ES256"], **common)
        claims = jwt.decode(token, self.s.dev_secret, algorithms=["HS256"], **common)
        if claims["exp"] - claims["iat"] > MAX_DEV_TTL:
            raise jwt.InvalidTokenError("dev token lifetime too long")
        return claims

    async def verify_token(self, token: str) -> AccessToken | None:
        try:
            claims = self.decode(token)
        except (jwt.PyJWTError, jwt.PyJWKClientError):
            return None
        scopes = _scopes(claims)
        if SCOPE not in scopes:
            return None
        return AccessToken(token=token, client_id=str(claims.get("azp") or claims.get("client_id") or "unknown"),
                           scopes=scopes, expires_at=int(claims["exp"]), resource=self.s.resource,
                           subject=str(claims["sub"]), claims={"iss": claims["iss"]})


def issue_dev_token(settings: Settings, subject: str = "local-dev", ttl: int = 3600) -> str:
    if settings.mode != "dev":
        raise RuntimeError("dev tokens only exist in dev mode")
    now = int(time.time())
    return jwt.encode({"sub": subject, "iss": DEV_ISSUER, "aud": settings.resource, "iat": now,
                       "exp": now + min(ttl, MAX_DEV_TTL), "scope": SCOPE}, settings.dev_secret, algorithm="HS256")


if __name__ == "__main__":  # pragma: no cover
    import sys
    if "--dev-token" not in sys.argv:
        sys.exit("usage: python -m agents.public_mcp.auth --dev-token")
    print(issue_dev_token(Settings.from_env()))
