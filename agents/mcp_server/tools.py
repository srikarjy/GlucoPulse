"""
The clinical-reasoning agent's tool surface. One `ToolBox` backs both the
real MCP server (server.py) and the in-process `LocalGateway` used by
tests, mirroring priorauth/mcp_server/tools.py's dual-path design so auth,
idempotency and auditing are exercised identically either way.

Every call: authenticate -> scope check -> (mutating) idempotency
replay/conflict -> execute -> audit. Read tools call the EXISTING
GlucoPulse FHIR facade through ehr.sync.EhrClient (constraint: no parallel
FHIR client); audit_write appends to the EXISTING hash-chained audit log
(serving.healthcare.audit.AuditLogger), the same log the facade itself
writes to, so a reviewer sees one continuous chain, not two.

This tool layer is read-only against the EHR by design: there is no
fhir_write tool, and nothing in this file issues a PUT/POST to `/fhir/*`
except audit_write, which never touches a FHIR resource.
"""
from __future__ import annotations

import functools
import inspect
import typing
from typing import Any, Optional

from agents.mcp_server.auth import ForbiddenError, Principal, current_principal
from agents.quarantine import find_injections
from ehr.sync import EhrClient
from serving.healthcare import fhir_mapping as fm
from serving.healthcare.audit import AuditLogger

TOOLS: dict[str, dict] = {}


def tool(scope: str, mutating: bool = False):
    def deco(fn):
        TOOLS[fn.__name__] = {"scope": scope, "mutating": mutating, "fn": fn, "doc": inspect.getdoc(fn) or ""}
        return fn
    return deco


class ToolError(Exception):
    pass


class ToolBox:
    """`fhir` is an ehr.sync.EhrClient already pointed at (and authenticated
    against) the GlucoPulse facade -- this class never constructs its own
    FHIR auth or transport. `audit` is the same AuditLogger instance the
    facade itself uses, if the caller wants one continuous chain; a fresh
    one is fine too (it's append-only either way)."""

    def __init__(self, fhir: Optional[EhrClient], audit: AuditLogger, triage: Any = None):
        self.fhir, self.audit = fhir, audit
        self.triage = triage   # agents.triage.provider.CaseProvider for triage sessions, else None
        self._idempotency: dict[tuple, Any] = {}

    def invoke(self, name: str, kwargs: dict) -> Any:
        spec = TOOLS.get(name)
        if not spec:
            raise ToolError(f"unknown tool {name}")
        p: Principal = current_principal()
        if spec["scope"] not in p.scopes:
            raise ForbiddenError(f"{p.role} lacks scope {spec['scope']}")
        kwargs = dict(kwargs)
        idem = kwargs.pop("idempotency_key", None)
        if spec["mutating"]:
            if not idem:
                raise ToolError(f"{name} requires idempotency_key")
            key = (p.subject, name, idem)
            if key in self._idempotency:
                return self._idempotency[key]
        try:
            out = spec["fn"](self, **kwargs)
        except ToolError:
            raise
        if spec["mutating"]:
            self._idempotency[(p.subject, name, idem)] = out
        return out

    # ----- read tools (FHIR, via the existing facade only) ---------------

    @tool("fhir:read")
    def fhir_read(self, resource_type: str, resource_id: str) -> dict:
        """Fetch one FHIR resource (Patient or Observation) by id from the
        existing GlucoPulse /fhir facade. Read-only."""
        return self.fhir.get_resource(resource_type, resource_id)

    @tool("fhir:read")
    def fhir_search(self, patient: str, date: Optional[list[str]] = None, count: int = 100) -> dict:
        """Search Observations for one patient via the existing facade's
        /fhir/Observation?patient=&date=&_count= -- the facade's only
        searchable resource type today. `date` entries are ge/le-prefixed
        instants, e.g. ['ge2024-01-01T00:00:00Z']."""
        params: dict = {"patient": patient, "_count": count}
        if date:
            params["date"] = date
        return self.fhir.search("Observation", **params)

    @tool("fhir:read")
    def fhir_search_notes(self, patient: str, count: int = 100) -> dict:
        """Search DocumentReference (clinical notes) for one patient via
        the facade's /fhir/DocumentReference?patient=&_count=, then
        QUARANTINE before returning -- any note matching an injection
        pattern comes back with quarantined=true and empty text, never
        the raw text, regardless of what the caller asks for. Mirrors
        priorauth's get_clinical_notes tool: free text is UNTRUSTED DATA,
        and that is enforced here, not left to agent discipline."""
        bundle = self.fhir.search("DocumentReference", patient=patient, **{"_count": count})
        rows = fm.notes_from_bundle(bundle)
        quarantined = set(find_injections([(note_id, text) for note_id, _, text in rows]))
        return {"notes": [
            {"note_id": note_id, "date": date, "quarantined": note_id in quarantined,
             "text": "" if note_id in quarantined else text}
            for note_id, date, text in rows
        ]}

    # ----- triage read tools (one case packet, history only) -------------

    def _case(self, patient: str):
        if self.triage is None:
            raise ToolError("no triage case is loaded in this session")
        try:
            self.triage.check(patient)
        except PermissionError as exc:
            raise ToolError(str(exc)) from exc
        return self.triage

    @tool("triage:read")
    def glucose_summary(self, patient: str) -> dict:
        """CGM summary for the 2 hours before now: current value, 15-minute change,
        2-hour min/max, reading count. Read-only."""
        return self._case(patient).section("glucose_summary")

    @tool("triage:read")
    def lifestyle_summary(self, patient: str) -> dict:
        """Meals and activity in the 2 hours before now: last meal and its macros,
        carbs logged, mean heart rate and METs over the last 30 minutes. Read-only."""
        return self._case(patient).section("lifestyle_summary")

    @tool("triage:read")
    def data_quality(self, patient: str) -> dict:
        """How complete and trustworthy the recent data is: expected vs received CGM
        readings, sensors and logs in use. Read-only."""
        return self._case(patient).section("data_quality")

    @tool("triage:read")
    def list_scenarios(self, patient: str) -> dict:
        """The what-if scenarios that can be forecast for this patient, by name."""
        return {"scenarios": self._case(patient).menu()}

    @tool("triage:read")
    def forecast_scenarios(self, patient: str, scenarios: list[str]) -> dict:
        """Model-based glucose forecasts (median and 80% interval at +30 and +60 min)
        for named what-if scenarios, e.g. ["as_planned", "meal_60g_now"]. Each result
        has a resourceType/id to cite. Associations, not causal estimates or advice."""
        try:
            return {"scenarios": self._case(patient).scenarios(list(scenarios))}
        except KeyError as exc:
            raise ToolError(str(exc)) from exc

    # ----- write tool (audit log only -- never the EHR) -------------------

    @tool("audit:write", mutating=True)
    def audit_write(self, action: str, resource_type: str, patient: Optional[str],
                    outcome: str, detail: Optional[dict] = None) -> dict:
        """Append one entry to the hash-chained audit log. This is one of
        only two mutating tools in this ToolBox (the other is
        submit_review_decision, below); neither ever touches the EHR."""
        p = current_principal()
        return self.audit.record(
            actor=p.subject, client_id=p.role, action=action, resource_type=resource_type,
            patient=patient, outcome=outcome, purpose_of_use="clinical-reasoning-agent",
            method="TOOL", path=f"tool:{action}", status=200 if outcome == "success" else 0,
            request_id="", source_ip="", detail=detail,
        )

    @tool("review:decide", mutating=True)
    def submit_review_decision(self, patient: str, reviewer: str, decision: str,
                               finding: dict, edited_finding: Optional[dict] = None,
                               comment: str = "") -> dict:
        """The locked scope's "approve/edit/reject + export to audit log",
        scoped as a backend tool rather than a new UI page (a minimal
        deliverable the locked scope's own process rules explicitly allow;
        priorauth-ui/app/(app)/review has the UI pattern this would plug
        into if/when built out). `decision` in {"approve","edit","reject"};
        "edit" requires `edited_finding`. This is the ONLY place a human
        reviewer's identity and decision enter the system -- the agent
        itself never calls this tool (no scope grants it to the
        clinical-agent role; only a "reviewer"-role principal can)."""
        if decision not in {"approve", "edit", "reject"}:
            raise ToolError(f"unknown decision {decision!r}")
        if decision == "edit" and edited_finding is None:
            raise ToolError("decision='edit' requires edited_finding")
        p = current_principal()
        return self.audit.record(
            actor=p.subject, client_id=p.role, action="review_decision", resource_type="ClinicalFinding",
            patient=patient, outcome=decision, purpose_of_use="human-review",
            method="TOOL", path="tool:submit_review_decision", status=200,
            request_id="", source_ip="",
            detail={"reviewer": reviewer, "decision": decision, "comment": comment,
                   "original_finding": finding, "edited_finding": edited_finding},
        )


class LocalGateway:
    """In-process stand-in for a bearer-authenticated MCP client call --
    binds `principal` for the duration of one invoke(), same as BearerASGI
    does per-request on the real server."""

    def __init__(self, box: ToolBox, principal: Principal):
        self.box, self.principal = box, principal

    def invoke(self, name: str, **kwargs) -> Any:
        from agents.mcp_server.auth import reset_principal, set_principal
        tok = set_principal(self.principal)
        try:
            return self.box.invoke(name, kwargs)
        finally:
            reset_principal(tok)


def _bind(box: ToolBox, name: str):
    """Mirrors priorauth/mcp_server/tools.py::_bind exactly -- a plain
    functools.wraps here (what this function used to do) leaves `self` in
    the signature FastMCP introspects to build the tool's JSON schema, so
    a real model sees `self` as a required string argument, guesses a
    value for it, and every call fails server-side with "multiple values
    for argument 'self'" (ScriptedModel-based tests never catch this --
    they specify exact args and never read the advertised schema)."""
    fn = TOOLS[name]["fn"]

    @functools.wraps(fn)
    def wrapper(**kwargs):
        return box.invoke(name, kwargs)
    sig, hints = inspect.signature(fn), typing.get_type_hints(fn)
    params = [p.replace(annotation=hints.get(p.name, p.annotation)) for p in list(sig.parameters.values())[1:]]
    wrapper.__signature__ = sig.replace(parameters=params, return_annotation=hints.get("return", sig.return_annotation))
    wrapper.__annotations__ = {p.name: p.annotation for p in params} | {"return": wrapper.__signature__.return_annotation}
    return wrapper
