"""
Pipeline: intake -> policy retrieval -> evidence validation -> drafting -> escalation.

Fail-closed: if any stage raises, the orchestrator does NOT drop the request and does
NOT guess -- it stages an `insufficient_info` draft flagged `agent_failure` and routes it
to a physician reviewer. A failure can only ever make a human look *harder*.
"""
from __future__ import annotations

import logging

from priorauth.agents.base import PipelineState, gateway_for
from priorauth.agents.deterministic import (DecisionDraftingAgent, EscalationAgent, EvidenceValidationAgent,
                                            IntakeAgent, PolicyRetrievalAgent)
from priorauth.mcp_server.tools import ToolBox
from priorauth.observability import CONFIDENCE, DRAFTS, FLAGS, PIPELINE_FAIL, tracer

log = logging.getLogger("priorauth.pipeline")


class Orchestrator:
    def __init__(self, box: ToolBox, agents: list | None = None):
        self.box = box
        self.last_error = None
        self.agents = agents or [IntakeAgent(box), PolicyRetrievalAgent(box), EvidenceValidationAgent(box),
                                 DecisionDraftingAgent(box), EscalationAgent(box)]

    def run(self, request_id: str) -> PipelineState:
        state = PipelineState(request_id=request_id)
        with tracer.start_as_current_span("pipeline") as span:
            span.set_attribute("pa.request_id", request_id)
            for a in self.agents:
                try:
                    state = a.run(state)
                except Exception as e:   # noqa: BLE001 - deliberate catch-all, fail closed
                    log.exception("agent %s failed for %s", a.name, request_id)
                    self.last_error = f"{a.name}: {type(e).__name__}: {str(e)[:300]}"
                    PIPELINE_FAIL.labels(a.name).inc()
                    return self._fail_closed(state, a.name, e)
        d = state.staged
        DRAFTS.labels(d["recommendation"], state.task["tier"]).inc()
        CONFIDENCE.observe(d["confidence"])
        for f in state.flags:
            FLAGS.labels(f).inc()
        return state

    def _fail_closed(self, state: PipelineState, agent: str, err: Exception) -> PipelineState:
        flags = sorted(set(state.flags) | {"agent_failure", f"failed_stage:{agent}"})
        drafter = gateway_for(self.box, "drafting-agent")
        esc = gateway_for(self.box, "escalation-agent")
        drafter.call("create_draft_determination", request_id=state.request_id, policy_id=state.policy_id or "",
                     recommendation="insufficient_info", confidence=0.05,
                     rationale=f"Automated analysis failed at stage '{agent}' ({type(err).__name__}). No recommendation can be offered; full manual review required.",
                     criteria_summary=[], flags=flags, missing_information=["Automated review incomplete"],
                     idempotency_key=f"{state.request_id}:failclosed:draft")
        state.task = esc.call("escalate_to_reviewer", request_id=state.request_id, idempotency_key=f"{state.request_id}:failclosed:esc")
        state.flags = flags
        return state
