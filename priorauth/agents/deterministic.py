"""
Deterministic backend for the five agents. No LLM: structured logic over the same MCP
tools. Used for CI, regression tests, and as the eval baseline that an LLM backend
must beat on free-text judgement (see docs/priorauth/EVALUATION.md).
"""
from __future__ import annotations

from priorauth.agents.base import Agent, PipelineState
from priorauth.criteria import overall
from priorauth.schemas import CriterionResult, EvidenceReport, IntakeResult
from priorauth.synth.policies import BY_ID

SEMANTIC_MIN = 0.30


class IntakeAgent(Agent):
    name, role = "intake", "intake-agent"

    def _run(self, s: PipelineState) -> PipelineState:
        req = self.gw.call("get_request", request_id=s.request_id)
        v = self.gw.call("validate_codes", service_code=req["service_code"], diagnosis_codes=req["diagnosis_codes"])
        issues = list(v["issues"])
        s.intake = IntakeResult(request_id=s.request_id, normalized_codes=sorted({c.upper() for c in req["diagnosis_codes"]}),
                                service_code=req["service_code"], issues=issues, urgency=req["urgency"])
        s.flags += [i for i in issues if i in ("code_mismatch", "no_policy")]
        return s


class PolicyRetrievalAgent(Agent):
    name, role = "policy_retrieval", "policy-agent"

    def _run(self, s: PipelineState) -> PipelineState:
        req = self.gw.call("get_request", request_id=s.request_id)
        q = f"{req['service_description']} {' '.join(s.intake.normalized_codes)}"
        m = self.gw.call("search_policies", query=q, service_code=req["service_code"], k=3)["matches"]
        s.policy_candidates = m
        top = m[0] if m else None
        if top and (top["via"] == "code" or top["score"] >= SEMANTIC_MIN):
            s.policy_id = top["policy_id"]
            if top["via"] != "code":
                s.flags.append("policy_matched_semantically")
        else:
            s.policy_id = None
            if "no_policy" not in s.flags:
                s.flags.append("no_policy")
        return s


class EvidenceValidationAgent(Agent):
    name, role = "evidence_validation", "evidence-agent"

    def _run(self, s: PipelineState) -> PipelineState:
        if not s.policy_id:
            s.evidence = EvidenceReport(policy_id="", results=[], flags=["no_policy"])
            return s
        req = self.gw.call("get_request", request_id=s.request_id)
        out = self.gw.call("evaluate_criteria", policy_id=s.policy_id, patient_ref=req["patient_ref"], request_id=s.request_id)
        elig = self.gw.call("check_coverage_eligibility", patient_ref=req["patient_ref"])
        flags = list(out["flags"])
        if elig["active"] is False:
            flags.append("coverage_inactive")
        s.evidence = EvidenceReport(policy_id=s.policy_id, results=[CriterionResult.model_validate(r) for r in out["results"]], flags=flags)
        s.flags += [f for f in flags if f not in s.flags]
        return s


class DecisionDraftingAgent(Agent):
    name, role = "decision_drafting", "drafting-agent"

    def _run(self, s: PipelineState) -> PipelineState:
        if not s.policy_id:
            rec, results, missing = "insufficient_info", [], ["No applicable coverage policy located for the requested service."]
            rationale = "No coverage policy could be matched to the requested service code; a human reviewer must determine coverage rules."
        else:
            pol = BY_ID[s.policy_id]
            results = s.evidence.results
            rec = overall(pol, results)
            if rec != "does_not_support" and any(f in s.flags for f in ("conflicting_data", "code_mismatch")):
                rec = "insufficient_info"     # never recommend approval on conflicted or mis-coded input
            ctext = {c.id: c.text for c in pol.criteria}
            missing = [ctext[r.criterion_id] for r in results if r.met is None]
            unmet = [ctext[r.criterion_id] for r in results if r.met is False]
            rationale = (f"Policy {pol.policy_id}: {sum(r.met is True for r in results)}/{len(results)} criteria met. "
                         + (f"Not met: {'; '.join(unmet)}. " if unmet else "")
                         + (f"Unable to determine: {'; '.join(missing)}. " if missing else "")
                         + (f"Flags: {', '.join(s.flags)}. " if s.flags else "")
                         + "ADVISORY DRAFT ONLY - requires human review.")
        s.staged = self.gw.call("create_draft_determination", request_id=s.request_id, policy_id=s.policy_id or "",
                                recommendation=rec, confidence=0.97, rationale=rationale,
                                criteria_summary=[r.model_dump(mode="json") for r in results], flags=s.flags,
                                missing_information=missing, idempotency_key=self.key(s, "draft"))
        return s


class EscalationAgent(Agent):
    name, role = "escalation", "escalation-agent"

    def _run(self, s: PipelineState) -> PipelineState:
        self.gw.call("record_audit_note", request_id=s.request_id, idempotency_key=self.key(s, "note"),
                     note=f"policy={s.policy_id} recommendation={s.staged['recommendation']} confidence={s.staged['confidence']} flags={s.flags}")
        s.task = self.gw.call("escalate_to_reviewer", request_id=s.request_id, idempotency_key=self.key(s, "escalate"))
        return s
