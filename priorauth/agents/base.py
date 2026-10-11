from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from priorauth.mcp_server.auth import Principal, ROLE_SCOPES
from priorauth.mcp_server.tools import LocalGateway, ToolBox
from priorauth.observability import AGENT_LAT, tracer
from priorauth.schemas import EvidenceReport, IntakeResult


@dataclass
class PipelineState:
    request_id: str
    intake: Optional[IntakeResult] = None
    policy_id: Optional[str] = None
    policy_candidates: list[dict] = field(default_factory=list)
    evidence: Optional[EvidenceReport] = None
    flags: list[str] = field(default_factory=list)
    staged: Optional[dict] = None
    task: Optional[dict] = None
    trace: list[dict] = field(default_factory=list)


def gateway_for(box: ToolBox, role: str) -> LocalGateway:
    return LocalGateway(box, Principal(subject=role, role=role, scopes=frozenset(ROLE_SCOPES[role])))


class Agent:
    """One agent = one role = one least-privilege principal. Subclasses implement `_run`."""
    name = "agent"
    role = ""

    def __init__(self, box: ToolBox):
        self.box = box
        self.gw = gateway_for(box, self.role)

    def run(self, state: PipelineState) -> PipelineState:
        t0 = time.perf_counter()
        with tracer.start_as_current_span(f"agent.{self.name}") as span:
            span.set_attribute("pa.request_id", state.request_id)
            state = self._run(state)
        dt = time.perf_counter() - t0
        AGENT_LAT.labels(self.name).observe(dt)
        state.trace.append({"agent": self.name, "seconds": round(dt, 4)})
        return state

    def _run(self, state: PipelineState) -> PipelineState:  # pragma: no cover
        raise NotImplementedError

    def key(self, state: PipelineState, tool: str) -> str:
        return f"{state.request_id}:{self.name}:{tool}"
