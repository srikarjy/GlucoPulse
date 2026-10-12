"""
Drives REAL Strands agents -> REAL MCP server (HTTP + JWT) with a scripted model standing in for Bedrock.
Proves wiring, per-role least privilege, grounding against a lying model, and fail-closed behaviour.
It does NOT prove LLM quality; that needs a live-model eval run (see docs/priorauth/EVALUATION.md).
"""
import json
import uuid

from strands.models import Model

from priorauth.agents.orchestrator import Orchestrator
from priorauth.agents.strands_backend import build_strands_agents, tools_for_role
from priorauth.mcp_server.tools import TOOLS
from priorauth.schemas import PARRequest, ReviewTier
from tests.priorauth.conftest import by_profile, clean_meets


class ScriptedModel(Model):
    """Replays: for each agent invocation, a list of turns; a turn is ('tool', name, input) or ('text', str)."""
    def __init__(self, turns): self.turns, self.i = list(turns), 0
    def update_config(self, **kw): ...
    def get_config(self): return {}
    async def structured_output(self, output_model, prompt, system_prompt=None, **kw):  # pragma: no cover
        raise NotImplementedError
    async def stream(self, messages, tool_specs=None, system_prompt=None, **kw):
        kind, *rest = self.turns[self.i]; self.i += 1
        yield {"messageStart": {"role": "assistant"}}
        if kind == "text":
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": rest[0]}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
        else:
            name, inp = rest
            yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": uuid.uuid4().hex[:8], "name": name}}}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(inp)}}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        yield {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}, "metrics": {"latencyMs": 1}}}


def scripts(rid, pid, policy, draft_args, escalate=True):
    key = lambda a, t: f"{rid}:{a}:{t}"
    return {
        "intake": [("tool", "get_request", {"request_id": rid}), ("text", '{"issues": []}')],
        "policy_retrieval": [("tool", "search_policies", {"query": "x", "service_code": "A9276"}), ("text", json.dumps({"policy_id": policy}))],
        "evidence_validation": [("tool", "evaluate_criteria", {"policy_id": policy, "patient_ref": pid, "request_id": rid}),
                                ("text", '{"results": [], "flags": []}')],
        "decision_drafting": [("tool", "create_draft_determination", {"request_id": rid, "policy_id": policy,
                              "idempotency_key": key("decision_drafting", "draft"), **draft_args}), ("text", "drafted")],
        "escalation": ([("tool", "escalate_to_reviewer", {"request_id": rid, "idempotency_key": key("escalation", "escalate")})] if escalate else [])
                      + [("text", "escalated")],
    }


def run(server, scn, draft_args, policy=None, escalate=True):
    url, box = server
    box.store.create_request(PARRequest.model_validate(scn["request"]))
    rid, pid = scn["request"]["request_id"], scn["request"]["patient_ref"]
    sc = scripts(rid, pid, policy or scn["expected"]["policy_id"], draft_args, escalate)
    stages = build_strands_agents(url, box.store, model_factory=None)
    for st in stages:
        turns = sc[st.name]
        st.model_factory = lambda t=turns: ScriptedModel(t)
    Orchestrator(box, agents=stages).run(rid)
    return box, rid


DRAFT = dict(recommendation="supports_approval", confidence=0.99, rationale="meets criteria", criteria_summary=[], flags=[], missing_information=[])


def test_role_tool_lists_are_least_privilege():
    assert "create_draft_determination" not in tools_for_role("evidence-agent")
    assert "get_clinical_notes" not in tools_for_role("drafting-agent")
    assert tools_for_role("escalation-agent") == {"request_additional_info", "escalate_to_reviewer", "record_audit_note", "get_request"}
    assert set().union(*(tools_for_role(r) for r in ("intake-agent", "policy-agent", "evidence-agent", "drafting-agent", "escalation-agent"))) == set(TOOLS)


def test_full_strands_pipeline_over_real_mcp_ends_in_human_review(server):
    box, rid = run(server, clean_meets(), DRAFT)
    t = box.store.get_task(f"TASK-{rid}")
    assert t and t.state == "open" and t.draft.status == "DRAFT"
    assert box.store.get_request(rid)[1] == "awaiting_review"
    assert box.store.verify_audit_chain()[0]


def test_lying_model_is_grounded_over_the_real_stack(server):
    scn = by_profile("fail")[0]
    box, rid = run(server, scn, DRAFT)           # model claims supports_approval@0.99 on a failing case
    t = box.store.get_task(f"TASK-{rid}")
    assert t.draft.recommendation.value == "does_not_support"
    assert t.tier == ReviewTier.PHYSICIAN and "flag:recommendation_clamped" in t.reasons


def test_model_that_skips_escalation_fails_closed_to_physician(server):
    box, rid = run(server, clean_meets(), DRAFT, escalate=False)
    t = box.store.get_task(f"TASK-{rid}")
    assert t.tier == ReviewTier.PHYSICIAN and "flag:agent_failure" in t.reasons


def test_eval_harness_can_drive_the_llm_backend_end_to_end():
    """Wiring test for `run_eval --backend bedrock` with scripted models (no AWS, no cost). Not an LLM-quality measurement."""
    from priorauth.eval.llm_backend import make_factory
    from priorauth.eval.run_eval import run
    s = clean_meets()
    rid, pid, pol = s["request"]["request_id"], s["request"]["patient_ref"], s["expected"]["policy_id"]
    sc = scripts(rid, pid, pol, DRAFT)
    r = run([s], orchestrator_factory=make_factory(per_stage=lambda name: (lambda t=sc[name]: ScriptedModel(t))))
    assert r["n"] == 1 and r["safety"]["human_review_rate"] == 1.0 and r["safety"]["no_final_decisions_by_agents"] == 1.0
    assert r["recommendation_accuracy"] == 1.0
