"""
FHIR R4 REST facade (read-only data + a `$forecast` operation).

    GET  /fhir/metadata                              CapabilityStatement (public)
    GET  /fhir/.well-known/smart-configuration       SMART discovery (public)
    GET  /fhir/Patient/{id}
    GET  /fhir/Observation?patient=&date=&_count=&_page=
    POST /fhir/Patient/{id}/$forecast                Bundle in -> Bundle out
    GET  /fhir/AuditEvent?patient=&_count=

Every protected call is authenticated (security.py), scope-checked, and
audited (audit.py) -- including denials and errors. Errors are returned as
FHIR OperationOutcome.

Data access and the model are injected (`ReadingStore`, `forecast_fn`) so this
module has no dependency on torch/onnx or on a database, and the standalone
serving image (no DB) still works: it just can't answer Observation searches.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional, Protocol

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from . import fhir_mapping as fm
from .audit import AuditLogger, to_fhir_audit_event
from .deid import IdentifierRejected, validate_patient_id
from .security import Principal, auth_enabled, authenticate

GRID = timedelta(minutes=5)
GRID_TOLERANCE = timedelta(seconds=90)
MIN_HISTORY, MAX_HISTORY = 6, 24
MAX_BUNDLE_ENTRIES = 288  # 24h of 5-min readings
CGM_MIN, CGM_MAX = 40.0, 400.0  # Dexcom G6 reporting range; same bound as the DLQ policy
MODEL_VERSION = "v1"

Readings = list[tuple[datetime, float]]
# Returns {30: (pred, lo, hi), 60: (pred, lo, hi)}
ForecastFn = Callable[[Readings], dict[int, tuple[float, float, float]]]


class ReadingStore(Protocol):
    def has_patient(self, pseudonym: str) -> bool: ...
    def readings(self, pseudonym: str, start: Optional[datetime], end: Optional[datetime],
                 limit: int, offset: int) -> tuple[Readings, int]: ...
    def latest(self, pseudonym: str, n: int) -> Readings: ...


class InMemoryStore:
    """Test/demo store keyed by pseudonym."""

    def __init__(self, data: dict[str, Readings]):
        self._d = {k: sorted(v) for k, v in data.items()}

    def has_patient(self, pseudonym: str) -> bool:
        return pseudonym in self._d

    def readings(self, pseudonym, start, end, limit, offset):
        rows = [r for r in self._d.get(pseudonym, [])
                if (start is None or r[0] >= start) and (end is None or r[0] <= end)]
        return rows[offset:offset + limit], len(rows)

    def latest(self, pseudonym, n):
        return self._d.get(pseudonym, [])[-n:]


class FhirError(Exception):
    def __init__(self, status: int, code: str, message: str, headers: Optional[dict] = None,
                 severity: str = "error"):
        self.status, self.code, self.message = status, code, message
        self.headers, self.severity = headers or {}, severity


def _resp(content: dict, status: int = 200) -> JSONResponse:
    return JSONResponse(content, status_code=status, media_type=fm.FHIR_JSON)


def _public_url() -> str:
    return os.getenv("GLUCOPULSE_PUBLIC_URL", "http://localhost:8000").rstrip("/")


def install(app: FastAPI, *, audit: AuditLogger, forecast_fn: ForecastFn,
            store: Optional[ReadingStore] = None) -> None:
    """Mount the FHIR facade on `app`."""

    @app.exception_handler(FhirError)
    async def _fhir_error(_: Request, exc: FhirError):
        return JSONResponse(
            fm.operation_outcome(exc.severity, exc.code, exc.message),
            status_code=exc.status, media_type=fm.FHIR_JSON, headers=exc.headers)

    @app.middleware("http")
    async def _no_store(request: Request, call_next):
        # PHI-bearing responses must not sit in shared/proxy/browser caches.
        resp = await call_next(request)
        if request.url.path.startswith("/fhir"):
            resp.headers["Cache-Control"] = "no-store"
            resp.headers["X-Content-Type-Options"] = "nosniff"
        return resp

    router = APIRouter(prefix="/fhir", tags=["FHIR R4"])

    def _log(request: Request, who: Optional[Principal], action: str, rtype: str,
             patient: Optional[str], outcome: str, status: int,
             purpose: Optional[str]) -> None:
        audit.record(
            actor=who.sub if who else "anonymous",
            client_id=who.client_id if who else None,
            action=action, resource_type=rtype, patient=patient, outcome=outcome,
            purpose_of_use=purpose or "unspecified",
            method=request.method, path=request.url.path, status=status,
            request_id=request.headers.get("x-request-id") or str(uuid.uuid4()),
            source_ip=request.client.host if request.client else "",
        )

    def guard(rtype: str, action: str, audit_action: str):
        """Authenticate + require some scope for (rtype, action). Audits denials."""
        def dep(request: Request, authorization: Optional[str] = Header(default=None),
                x_purpose_of_use: Optional[str] = Header(default=None)) -> Principal:
            try:
                who = authenticate(authorization)
            except HTTPException as exc:
                _log(request, None, audit_action, rtype, None, "denied", exc.status_code, x_purpose_of_use)
                raise FhirError(exc.status_code, "login", exc.detail, headers=exc.headers)
            if not who.has_type_scope(rtype, action):
                _log(request, who, audit_action, rtype, None, "denied", 403, x_purpose_of_use)
                raise FhirError(403, "forbidden", "Insufficient scope.")
            return who
        return dep

    def _patient_or_404(request, who, rtype, action, pid, purpose, require_known=True) -> str:
        try:
            validate_patient_id(pid)
        except IdentifierRejected as exc:
            _log(request, who, action, rtype, None, "error", 400, purpose)
            raise FhirError(400, "invalid", str(exc))
        if not who.can(rtype, "read", pid):
            _log(request, who, action, rtype, pid, "denied", 403, purpose)
            raise FhirError(403, "forbidden", "Not authorized for this patient.")
        if require_known and store is not None and not store.has_patient(pid):
            # Deliberately after the authorization check: unauthorized callers
            # get 403 for every id, so they can't enumerate which patients exist.
            _log(request, who, action, rtype, pid, "error", 404, purpose)
            raise FhirError(404, "not-found", "Patient not found.")
        return pid

    # ---- public discovery ----------------------------------------------

    @router.get("/metadata")
    def metadata():
        return _resp(fm.capability_statement(_public_url() + "/fhir", auth_enabled()))

    @router.get("/.well-known/smart-configuration")
    def smart_config():
        issuer = os.getenv("GLUCOPULSE_ISSUER", _public_url())
        return _resp(fm.smart_configuration(
            issuer, os.getenv("GLUCOPULSE_JWKS_URL") or None,
            os.getenv("GLUCOPULSE_AUTH_ENDPOINT") or None,
            os.getenv("GLUCOPULSE_TOKEN_ENDPOINT") or None))

    # ---- Patient ---------------------------------------------------------

    @router.get("/Patient/{pid}")
    def read_patient(pid: str, request: Request,
                     who: Principal = Depends(guard("Patient", "read", "read")),
                     x_purpose_of_use: Optional[str] = Header(default=None)):
        _patient_or_404(request, who, "Patient", "read", pid, x_purpose_of_use)
        _log(request, who, "read", "Patient", pid, "success", 200, x_purpose_of_use)
        return _resp(fm.patient_resource(pid))

    # ---- Observation search ---------------------------------------------

    @router.get("/Observation")
    def search_observation(
        request: Request,
        patient: str = Query(..., description="Patient id (required; bulk access is not offered)"),
        date: list[str] = Query(default=[], description="ge/le prefixed instants, e.g. ge2024-01-01T00:00:00Z"),
        count: int = Query(100, ge=1, le=1000, alias="_count"),
        page: int = Query(0, ge=0, alias="_page"),
        who: Principal = Depends(guard("Observation", "read", "search")),
        x_purpose_of_use: Optional[str] = Header(default=None),
    ):
        pid = _patient_or_404(request, who, "Observation", "search", patient, x_purpose_of_use)
        if store is None:
            _log(request, who, "search", "Observation", pid, "error", 501, x_purpose_of_use)
            raise FhirError(501, "not-supported", "No data store configured on this deployment.")
        start = end = None
        try:
            for d in date:
                prefix, val = d[:2], d[2:]
                if prefix == "ge":
                    start = fm.parse_dt(val)
                elif prefix == "le":
                    end = fm.parse_dt(val)
                else:
                    raise ValueError(d)
        except ValueError:
            _log(request, who, "search", "Observation", pid, "error", 400, x_purpose_of_use)
            raise FhirError(400, "invalid", "date must be ge<instant> and/or le<instant>.")
        rows, total = store.readings(pid, start, end, count, page * count)
        links = [{"relation": "self", "url": str(request.url)}]
        if (page + 1) * count < total:
            links.append({"relation": "next",
                          "url": str(request.url.include_query_params(_page=page + 1))})
        _log(request, who, "search", "Observation", pid, "success", 200, x_purpose_of_use)
        return _resp(fm.bundle([fm.observation_from_reading(pid, t, v) for t, v in rows],
                               "searchset", total=total, links=links))

    # ---- $forecast -------------------------------------------------------

    @router.post("/Patient/{pid}/$forecast")
    def forecast(pid: str, request: Request, body: Optional[dict] = None,
                 who: Principal = Depends(guard("Observation", "read", "execute")),
                 x_purpose_of_use: Optional[str] = Header(default=None)):
        # A caller-supplied Bundle needn't match a patient in our store (an EHR
        # patient is never in it); the store lookup only matters when we pull data.
        supplied = bool(body and body.get("entry"))
        pid = _patient_or_404(request, who, "Observation", "execute", pid,
                              x_purpose_of_use, require_known=not supplied)
        if supplied:
            if body.get("resourceType") != "Bundle":
                _log(request, who, "execute", "Observation", pid, "error", 400, x_purpose_of_use)
                raise FhirError(400, "invalid", "Body must be a FHIR Bundle of Observations.")
            if len(body["entry"]) > MAX_BUNDLE_ENTRIES:
                _log(request, who, "execute", "Observation", pid, "error", 413, x_purpose_of_use)
                raise FhirError(413, "too-costly", f"At most {MAX_BUNDLE_ENTRIES} entries.")
            for e in body["entry"]:
                ref = (e.get("resource", {}).get("subject") or {}).get("reference")
                if ref and ref != f"Patient/{pid}":
                    _log(request, who, "execute", "Observation", pid, "denied", 403, x_purpose_of_use)
                    raise FhirError(403, "forbidden", "Bundle contains another patient's data.")
            readings = fm.readings_from_bundle(body)
        elif store is not None:
            readings = store.latest(pid, MAX_HISTORY * 2)
        else:
            _log(request, who, "execute", "Observation", pid, "error", 400, x_purpose_of_use)
            raise FhirError(400, "invalid", "Provide a Bundle of CGM Observations in the body.")

        bad = [v for _, v in readings if not CGM_MIN <= v <= CGM_MAX]
        if bad:
            _log(request, who, "execute", "Observation", pid, "error", 422, x_purpose_of_use)
            raise FhirError(422, "invalid", (
                f"{len(bad)} reading(s) outside {CGM_MIN:.0f}-{CGM_MAX:.0f} mg/dL "
                "(sensor reporting range); refusing to forecast from implausible data."))
        run = trailing_contiguous(readings)[-MAX_HISTORY:]
        if len(run) < MIN_HISTORY:
            _log(request, who, "execute", "Observation", pid, "error", 422, x_purpose_of_use)
            raise FhirError(422, "business-rule", (
                f"Need >= {MIN_HISTORY} consecutive 5-minute readings ending at the latest "
                f"reading; found {len(run)}. Gaps break the contiguous run."))

        preds = forecast_fn(run)
        issued = datetime.now(timezone.utc)
        last_t = run[-1][0]
        device = fm.device_resource(MODEL_VERSION)
        obs = [fm.forecast_observation(pid, issued, last_t + timedelta(minutes=h), h,
                                       p, lo, hi, device["id"])
               for h, (p, lo, hi) in sorted(preds.items())]
        prov = fm.provenance_resource(obs, device["id"], issued)
        _log(request, who, "execute", "Observation", pid, "success", 200, x_purpose_of_use)
        return _resp(fm.bundle(obs + [device, prov]))

    # ---- AuditEvent (accounting of disclosures, 45 CFR 164.528 support) ---

    @router.get("/AuditEvent")
    def search_audit(request: Request,
                     patient: Optional[str] = Query(default=None),
                     count: int = Query(100, ge=1, le=1000, alias="_count"),
                     who: Principal = Depends(guard("AuditEvent", "read", "search")),
                     x_purpose_of_use: Optional[str] = Header(default=None)):
        if patient is not None:
            if not who.can("AuditEvent", "read", patient):
                _log(request, who, "search", "AuditEvent", patient, "denied", 403, x_purpose_of_use)
                raise FhirError(403, "forbidden", "Not authorized for this patient.")
        elif not any(s.startswith(("user/", "system/")) for s in who.scopes):
            _log(request, who, "search", "AuditEvent", None, "denied", 403, x_purpose_of_use)
            raise FhirError(403, "forbidden", "patient parameter required.")
        rows = audit.read(count, patient)
        _log(request, who, "search", "AuditEvent", patient, "success", 200, x_purpose_of_use)
        return _resp(fm.bundle([to_fhir_audit_event(r) for r in rows], "searchset", total=len(rows)))

    app.include_router(router)


def trailing_contiguous(readings: Readings) -> Readings:
    """Longest run of readings ending at the latest one with ~5-minute spacing.

    The model was trained on a 5-minute grid; stitching across a sensor gap
    would silently present stale history as recent (the failure mode the
    ingestion pipeline's gap thresholds exist to catch)."""
    if not readings:
        return []
    readings = sorted(readings)
    run = [readings[-1]]
    for prev in reversed(readings[:-1]):
        if abs((run[0][0] - prev[0]) - GRID) <= GRID_TOLERANCE:
            run.insert(0, prev)
        else:
            break
    return run
