"""
FastAPI service. Three audiences, three token roles:
  submitter  -> POST /v1/requests (Idempotency-Key required)      EHR / intake system
  reviewer-* -> review queue, claim, decide                        humans
  auditor    -> audit trail + chain verification                   compliance

The ONLY route that writes a final decision is POST /v1/reviews/{task}/decision, and it
requires a human reviewer principal. Denials additionally require `review:decide:denial`
(physician). Agents hold MCP-audience tokens, which this API rejects.
"""
from __future__ import annotations

import os
import uuid
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from priorauth.mcp_server.auth import AuthError, Principal, verify
from priorauth.observability import QUEUE_DEPTH, REGISTRY, REQUESTS, REVIEW_OPEN, REVIEW_OVERRIDE
from priorauth.schemas import FinalDecision, PARRequest, Recommendation, ReviewDecision
from priorauth.store import Store

AGREE = {Recommendation.SUPPORTS_APPROVAL: FinalDecision.APPROVED, Recommendation.DOES_NOT_SUPPORT: FinalDecision.DENIED,
         Recommendation.INSUFFICIENT_INFO: FinalDecision.PENDED_FOR_INFO}


class PARIn(BaseModel):
    patient_ref: str
    service_code: str
    service_description: str = ""
    diagnosis_codes: list[str] = Field(default_factory=list)
    provider_npi: str = ""
    urgency: str = Field("standard", pattern="^(standard|expedited)$")
    free_text: str = Field("", max_length=20000)


class DecisionIn(BaseModel):
    decision: FinalDecision
    comment: str = Field("", max_length=4000)


TIMELINE_ACTIONS = {"request.received", "tool.call", "tool.denied", "tool.replayed", "tool.error", "draft.created",
                    "review.claimed", "review.decided", "request.failed", "agent.note"}


def create_app(store: Store, demo: bool | None = None) -> FastAPI:
    demo = os.getenv("PRIORAUTH_DEMO") == "1" if demo is None else demo
    app = FastAPI(title="Prior Authorization Agent Platform", version="1.0")
    app.state.store = store

    def principal(need: str):
        def dep(authorization: str = Header(default="")) -> Principal:
            try:
                if not authorization.lower().startswith("bearer "):
                    raise AuthError("missing bearer token")
                p = verify(authorization[7:], aud="priorauth-api")
            except AuthError as e:
                raise HTTPException(401, "unauthorized") from e
            if need not in p.scopes:
                store.audit(p.subject, "api.denied", need)
                raise HTTPException(403, f"requires scope {need}")
            return p
        return dep

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.get("/readyz")
    def readyz():
        store.job_counts()
        return {"ready": True}

    @app.get("/metrics")
    def metrics():
        for s, n in store.job_counts().items():
            QUEUE_DEPTH.labels(s).set(n)
        for tier in ("standard", "priority", "physician"):
            REVIEW_OPEN.labels(tier).set(len(store.list_tasks("open", tier, limit=10000)))
        return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/me")
    def me(authorization: str = Header(default="")):
        try:
            if not authorization.lower().startswith("bearer "):
                raise AuthError("missing bearer token")
            p = verify(authorization[7:], aud="priorauth-api")
        except AuthError as e:
            raise HTTPException(401, "unauthorized") from e
        return {"subject": p.subject, "role": p.role, "scopes": sorted(p.scopes)}

    @app.get("/v1/requests/{request_id}/timeline")
    def timeline(request_id: str, p: Principal = Depends(principal("request:read"))):
        """Sanitised audit view: who (agent role / human) did what, when. No arguments, no clinical content."""
        if not store.get_request(request_id):
            raise HTTPException(404)
        rows = store.audit_entries(request_id, limit=500, ascending=True)
        return [{"seq": r["seq"], "ts": r["ts"], "actor": r["actor"], "action": r["action"], "tool": r["detail"].get("tool"),
                 "detail": {k: v for k, v in r["detail"].items() if k in ("recommendation", "confidence", "flags", "tier", "decision", "agreed_with_draft")}}
                for r in rows if r["action"] in TIMELINE_ACTIONS]

    if demo:
        @app.get("/v1/demo/scenarios")
        def demo_scenarios(p: Principal = Depends(principal("request:submit"))):
            """Synthetic scenarios incl. ground truth, for the UI playground. Only mounted when PRIORAUTH_DEMO=1."""
            from priorauth.synth.scenarios import load
            return [{"scenario_id": s["scenario_id"], "profile": s["profile"], "paraphrased_evidence": s["paraphrased_evidence"],
                     "request": {k: s["request"][k] for k in ("patient_ref", "service_code", "service_description", "diagnosis_codes", "urgency", "free_text")},
                     "expected": s["expected"]} for s in load()]

    @app.post("/v1/requests", status_code=202)
    def submit(body: PARIn, response: Response, idempotency_key: str = Header(default=""),
               p: Principal = Depends(principal("request:submit"))):
        if not idempotency_key:
            raise HTTPException(400, "Idempotency-Key header required")
        req = PARRequest(request_id="REQ-" + uuid.uuid4().hex[:12].upper(), **body.model_dump())
        stored, created = store.create_request(req, client_key=f"{p.subject}:{idempotency_key}")
        if created:
            REQUESTS.labels(stored.urgency).inc()
        else:
            response.status_code = 200
        return {"request_id": stored.request_id, "status": "received", "replayed": not created}

    @app.get("/v1/requests/{request_id}")
    def get_request(request_id: str, p: Principal = Depends(principal("request:read"))):
        r = store.get_request(request_id)
        if not r:
            raise HTTPException(404)
        out = {"request_id": request_id, "status": r[1]}
        if "review:read" in p.scopes:                   # drafts are internal; submitters see status only
            d = store.get_draft(request_id)
            out["draft"] = d.model_dump(mode="json") if d else None
        return out

    @app.get("/v1/reviews")
    def queue(tier: Optional[str] = None, state: str = Query("open", pattern="^(open|claimed|closed)$"),
              limit: int = Query(50, le=200), p: Principal = Depends(principal("review:read"))):
        return [t.model_dump(mode="json") for t in store.list_tasks(state, tier, limit)]

    @app.get("/v1/reviews/{task_id}")
    def get_review(task_id: str, p: Principal = Depends(principal("review:read"))):
        t = store.get_task(task_id)
        if not t:
            raise HTTPException(404)
        out = t.model_dump(mode="json")
        d = store.get_decision(task_id)
        out["decision"] = ({k: d[k] for k in ("reviewer", "decision", "comment", "agreed", "created_at")} if d else None)
        return out

    @app.post("/v1/reviews/{task_id}/claim")
    def claim(task_id: str, p: Principal = Depends(principal("review:decide"))):
        t = store.get_task(task_id)
        if not t:
            raise HTTPException(404)
        if t.tier.value == "physician" and "review:decide:denial" not in p.scopes:
            raise HTTPException(403, "physician-tier task requires a physician reviewer")
        if not store.claim_task(task_id, p.subject):
            raise HTTPException(409, "task already claimed or closed")
        return {"task_id": task_id, "claimed_by": p.subject}

    @app.post("/v1/reviews/{task_id}/decision")
    def decide(task_id: str, body: DecisionIn, p: Principal = Depends(principal("review:decide"))):
        t = store.get_task(task_id)
        if not t:
            raise HTTPException(404)
        if body.decision == FinalDecision.DENIED and "review:decide:denial" not in p.scopes:
            raise HTTPException(403, "denial requires a physician reviewer")
        agreed = AGREE[t.draft.recommendation] == body.decision
        ok = store.record_decision(ReviewDecision(task_id=task_id, reviewer=p.subject, decision=body.decision,
                                                  comment=body.comment, agreed_with_draft=agreed))
        if not ok:
            raise HTTPException(409, "task must be claimed by you before deciding")
        if not agreed:
            REVIEW_OVERRIDE.inc()
        return {"task_id": task_id, "decision": body.decision.value, "agreed_with_draft": agreed}

    @app.get("/v1/audit")
    def audit(request_id: Optional[str] = None, limit: int = Query(200, le=1000), p: Principal = Depends(principal("audit:read"))):
        return store.audit_entries(request_id, limit)

    @app.get("/v1/audit/verify")
    def verify_chain(p: Principal = Depends(principal("audit:read"))):
        ok, bad = store.verify_audit_chain()
        return {"intact": ok, "first_broken_seq": bad}

    return app


def app_factory():  # pragma: no cover - uvicorn --factory priorauth.api:app_factory
    from priorauth.bootstrap import make_store
    from priorauth.telemetry_setup import setup_tracing
    setup_tracing("priorauth-api")
    return create_app(make_store())
