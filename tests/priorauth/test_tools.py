import pytest

from priorauth.agents.base import gateway_for
from priorauth.mcp_server.auth import AuthError, ForbiddenError, issue_token, verify
from priorauth.mcp_server.tools import TOOLS, ToolError
from tests.priorauth.conftest import by_profile


def test_at_least_twelve_tools_and_all_mutating_ones_demand_idempotency_key():
    assert len(TOOLS) >= 12
    import inspect
    for n, t in TOOLS.items():
        has = "idempotency_key" in inspect.signature(t["fn"]).parameters
        assert has == t["mutating"], n


def test_scope_enforcement_and_denial_is_audited(box, submit):
    rid = submit(by_profile("meets")[0])
    intake = gateway_for(box, "intake-agent")
    with pytest.raises(ForbiddenError):
        intake.call("get_conditions", patient_ref="syn-0001")        # intake has no chart:read
    assert any(e["action"] == "tool.denied" for e in box.store.audit_entries())
    with pytest.raises(ForbiddenError):
        gateway_for(box, "evidence-agent").call("create_draft_determination", request_id=rid, policy_id="x",
            recommendation="insufficient_info", confidence=.5, rationale="", criteria_summary=[], flags=[],
            missing_information=[], idempotency_key="k")


def test_mutating_tool_requires_key(box, submit):
    rid = submit(by_profile("meets")[0])
    with pytest.raises(ToolError, match="idempotency_key"):
        gateway_for(box, "escalation-agent").call("record_audit_note", request_id=rid, note="x")


def _draft(gw, rid, rec="insufficient_info", conf=0.9, key="k1", **kw):
    return gw.call("create_draft_determination", request_id=rid, policy_id="SYN-CGM", recommendation=rec, confidence=conf,
                   rationale="r", criteria_summary=[], flags=kw.get("flags", []), missing_information=[], idempotency_key=key)


def test_idempotent_replay_returns_same_result_without_side_effects(box, submit):
    rid = submit(by_profile("meets")[0])
    gw = gateway_for(box, "drafting-agent")
    a = _draft(gw, rid); b = _draft(gw, rid)
    assert a == b
    assert any(e["action"] == "tool.replayed" for e in box.store.audit_entries())


def test_same_key_different_args_is_rejected(box, submit):
    rid = submit(by_profile("meets")[0])
    gw = gateway_for(box, "drafting-agent")
    _draft(gw, rid, conf=0.9)
    with pytest.raises(ToolError, match="different arguments"):
        _draft(gw, rid, conf=0.8)


@pytest.mark.parametrize("word", ["approve", "approved", "deny", "denied", "APPROVE", ""])
def test_agents_cannot_express_a_final_decision(box, submit, word):
    rid = submit(by_profile("meets")[0])
    with pytest.raises(ValueError):
        _draft(gateway_for(box, "drafting-agent"), rid, rec=word)


def test_agent_cannot_raise_confidence_above_server_calculation(box, submit):
    rid = submit(by_profile("meets")[0])
    out = _draft(gateway_for(box, "drafting-agent"), rid, rec="supports_approval", conf=1.0, key="hi")
    assert out["confidence"] <= 0.97                 # with no criteria evidence it is far lower than the claim
    out2 = _draft(gateway_for(box, "drafting-agent"), rid, rec="insufficient_info", conf=0.1, key="lo")
    assert out2["confidence"] == 0.1                  # agent may lower it


def test_tokens_audience_bound_and_scope_claims_ignored():
    mcp_tok = issue_token("evidence-agent")
    with pytest.raises(AuthError):
        verify(mcp_tok, aud="priorauth-api")
    import os
    import time

    import jwt
    forged = jwt.encode({"sub": "x", "role": "intake-agent", "scope": "chart:read draft:write", "aud": "priorauth-mcp",
                         "exp": int(time.time()) + 60}, os.environ["PRIORAUTH_JWT_SECRET"], algorithm="HS256")
    assert "chart:read" not in verify(forged).scopes
    with pytest.raises(AuthError):
        verify(forged + "x")
    expired = jwt.encode({"sub": "x", "role": "intake-agent", "aud": "priorauth-mcp", "exp": int(time.time()) - 5},
                         os.environ["PRIORAUTH_JWT_SECRET"], algorithm="HS256")
    with pytest.raises(AuthError):
        verify(expired)


def test_audit_never_contains_clinical_text(box, submit):
    from priorauth.agents.orchestrator import Orchestrator
    s = by_profile("injection")[0]
    Orchestrator(box).run(submit(s))
    blob = str(box.store.audit_entries(limit=1000))
    for n in s["facts"]["notes"]:
        assert n["text"] not in blob
    assert "Assessment" not in blob


def test_draft_is_grounded_model_cannot_fabricate_evidence_or_be_more_favourable(box, submit):
    s = by_profile("missing")[0]
    rid = submit(s)
    gw = gateway_for(box, "drafting-agent")
    pol = s["expected"]["policy_id"]
    lied = [{"criterion_id": s["expected"]["broken_criterion"], "met": True, "evidence": ["Observation/fake"]}]
    out = gw.call("create_draft_determination", request_id=rid, policy_id=pol, recommendation="supports_approval", confidence=0.99,
                  rationale="all good", criteria_summary=lied, flags=[], missing_information=[], idempotency_key="g1")
    assert out["recommendation"] == "insufficient_info"                    # clamped to the rules
    assert {"evidence_discrepancy", "recommendation_clamped"} <= set(out["flags"])
    d = box.store.get_staged(rid)
    assert {c.criterion_id: c for c in d.criteria_summary}[s["expected"]["broken_criterion"]].met is None


def test_model_cannot_drop_rule_derived_flags(box, submit):
    s = by_profile("injection")[0]
    rid = submit(s)
    out = gateway_for(box, "drafting-agent").call("create_draft_determination", request_id=rid, policy_id=s["expected"]["policy_id"],
        recommendation="supports_approval", confidence=0.99, rationale="", criteria_summary=[], flags=[], missing_information=[], idempotency_key="g2")
    assert "prompt_injection" in out["flags"] and out["recommendation"] != "supports_approval" and out["confidence"] <= 0.5


def test_model_may_upgrade_paraphrased_documentation_only_with_a_real_note(box, submit):
    s = next(x for x in __import__("tests.priorauth.conftest", fromlist=["x"])._SC if x["paraphrased_evidence"] and x["profile"] == "meets")
    rid = submit(s); pid = s["request"]["patient_ref"]
    from priorauth.synth.policies import BY_ID
    doc = [c.id for c in BY_ID[s["expected"]["policy_id"]].criteria if c.kind == "documentation"][0]
    gw = gateway_for(box, "drafting-agent")
    fake = gw.call("create_draft_determination", request_id=rid, policy_id=s["expected"]["policy_id"], recommendation="supports_approval",
                   confidence=0.9, rationale="", criteria_summary=[{"criterion_id": doc, "met": True, "evidence": ["DocumentReference/made-up"]}],
                   flags=[], missing_information=[], idempotency_key="p1")
    assert fake["recommendation"] == "insufficient_info"
    real = gw.call("create_draft_determination", request_id=rid, policy_id=s["expected"]["policy_id"], recommendation="supports_approval",
                   confidence=0.9, rationale="", criteria_summary=[{"criterion_id": doc, "met": True, "evidence": [f"DocumentReference/{pid}-n0"]}],
                   flags=[], missing_information=[], idempotency_key="p2")
    assert real["recommendation"] == "supports_approval"
