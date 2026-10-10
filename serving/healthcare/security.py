"""
Authentication and authorization for the FHIR facade (SMART-on-FHIR resource
server side). HIPAA Security Rule 164.312(a) access control, (d) person or
entity authentication.

Fail-closed: with no auth configured, every protected endpoint returns 401.
There is deliberately no "auth disabled" switch.

Modes (GLUCOPULSE_AUTH_MODE):
  jwks   -- RS256/ES256 tokens verified against GLUCOPULSE_JWKS_URL. The mode
            to use with a real authorization server (Keycloak, Okta, Epic...).
  hs256  -- shared-secret tokens (GLUCOPULSE_JWT_SECRET, >= 32 bytes). Local
            dev and tests only; the facade says so in /fhir/metadata.

Scopes are SMART v1 style: <context>/<Resource>.<read|write|*>
  patient/*  -- only the patient named in the token's `patient` claim
  user/*, system/* -- any patient
Per-endpoint scope checks happen in `Principal.can`.

Automatic logoff (164.312(a)(2)(iii)): tokens whose lifetime (exp - iat)
exceeds GLUCOPULSE_MAX_TOKEN_TTL (default 900s) are rejected.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import jwt
from fastapi import Header, HTTPException

JWKS_ALGS = ["RS256", "ES256"]


@dataclass(frozen=True)
class Principal:
    sub: str
    client_id: Optional[str]
    scopes: frozenset[str]
    patient: Optional[str] = None  # SMART patient-context launch claim
    claims: dict = field(default_factory=dict, compare=False, hash=False)

    def can(self, resource_type: str, action: str, patient_id: Optional[str] = None) -> bool:
        """action: 'read' | 'write'. patient_id None = not patient-scoped."""
        for scope in self.scopes:
            try:
                ctx, rest = scope.split("/", 1)
                res, perm = rest.split(".", 1)
            except ValueError:
                continue
            if res not in (resource_type, "*"):
                continue
            if not _perm_allows(perm, action):
                continue
            if ctx in ("user", "system"):
                return True
            if ctx == "patient":
                # patient/* scopes never reach beyond the launch patient
                if patient_id is not None and self.patient == patient_id:
                    return True
        return False


def _has_type_scope(self, resource_type: str, action: str) -> bool:
    """Any scope (any context) granting `action` on `resource_type`. Coarse
    gate used before the per-patient `can` check."""
    for scope in self.scopes:
        try:
            _, rest = scope.split("/", 1)
            res, perm = rest.split(".", 1)
        except ValueError:
            continue
        if res in (resource_type, "*") and _perm_allows(perm, action):
            return True
    return False


Principal.has_type_scope = _has_type_scope


def _perm_allows(perm: str, action: str) -> bool:
    if perm == "*":
        return True
    if perm in ("read", "write"):
        return perm == action
    # SMART v2 letters: c r u d s
    letters = set(perm)
    if action == "read":
        return bool(letters & {"r", "s"})
    return bool(letters & {"c", "u", "d"})


def _config() -> dict:
    return {
        "mode": os.getenv("GLUCOPULSE_AUTH_MODE", "").lower(),
        "secret": os.getenv("GLUCOPULSE_JWT_SECRET", ""),
        "jwks_url": os.getenv("GLUCOPULSE_JWKS_URL", ""),
        "issuer": os.getenv("GLUCOPULSE_ISSUER", ""),
        "audience": os.getenv("GLUCOPULSE_AUDIENCE", "glucopulse-fhir"),
        "max_ttl": int(os.getenv("GLUCOPULSE_MAX_TOKEN_TTL", "900")),
    }


def auth_enabled() -> bool:
    c = _config()
    return (c["mode"] == "jwks" and bool(c["jwks_url"] and c["issuer"])) or \
           (c["mode"] == "hs256" and len(c["secret"]) >= 32)


_jwks_client: Optional[jwt.PyJWKClient] = None


def _decode(token: str) -> dict:
    global _jwks_client
    c = _config()
    common = dict(audience=c["audience"], options={"require": ["exp", "iat", "sub"]})
    if c["issuer"]:
        common["issuer"] = c["issuer"]
    if c["mode"] == "jwks" and c["jwks_url"] and c["issuer"]:
        if _jwks_client is None:
            _jwks_client = jwt.PyJWKClient(c["jwks_url"], cache_keys=True)
        key = _jwks_client.get_signing_key_from_jwt(token).key
        return jwt.decode(token, key, algorithms=JWKS_ALGS, **common)
    if c["mode"] == "hs256" and len(c["secret"]) >= 32:
        return jwt.decode(token, c["secret"], algorithms=["HS256"], **common)
    raise HTTPException(status_code=401, detail="Authentication is not configured on this server.",
                        headers={"WWW-Authenticate": "Bearer"})


def authenticate(authorization: Optional[str] = Header(default=None)) -> Principal:
    """FastAPI dependency. Raises 401 for anything wrong with the token."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Bearer token required.",
                            headers={"WWW-Authenticate": "Bearer"})
    token = authorization[7:].strip()
    try:
        claims = _decode(token)
    except HTTPException:
        raise
    except jwt.PyJWTError:
        # Same message for every failure: don't tell a caller *why* a token failed.
        raise HTTPException(status_code=401, detail="Invalid or expired token.",
                            headers={"WWW-Authenticate": "Bearer"})
    ttl = int(claims["exp"]) - int(claims["iat"])
    if ttl > _config()["max_ttl"]:
        raise HTTPException(status_code=401, detail="Token lifetime exceeds server maximum.",
                            headers={"WWW-Authenticate": "Bearer"})
    scope = claims.get("scope", "")
    scopes = frozenset(scope.split() if isinstance(scope, str) else scope)
    return Principal(
        sub=str(claims["sub"]),
        client_id=claims.get("azp") or claims.get("client_id"),
        scopes=scopes,
        patient=claims.get("patient"),
        claims=claims,
    )


def forbid(detail: str = "Insufficient scope.") -> HTTPException:
    return HTTPException(status_code=403, detail=detail)
