export type Recommendation = "supports_approval" | "does_not_support" | "insufficient_info";
export type Tier = "standard" | "priority" | "physician";
export type FinalDecision = "approved" | "denied" | "pended_for_info";

export interface CriterionResult { criterion_id: string; met: boolean | null; evidence: string[]; note: string }
export interface Draft {
  request_id: string; policy_id: string; recommendation: Recommendation; confidence: number; rationale: string;
  criteria_summary: CriterionResult[]; flags: string[]; missing_information: string[]; status: "DRAFT";
}
export interface ReviewTask {
  task_id: string; request_id: string; tier: Tier; priority: number; reasons: string[]; draft: Draft;
  state: "open" | "claimed" | "closed"; claimed_by: string | null; created_at: string;
  decision?: { reviewer: string; decision: FinalDecision; comment: string; agreed: boolean; created_at: string } | null;
}
export interface Me { subject: string; role: string; scopes: string[] }
export interface RequestStatus { request_id: string; status: "received" | "processing" | "awaiting_review" | "failed" | "decided"; draft?: Draft | null }
export interface TimelineEvent { seq: number; ts: string; actor: string; action: string; tool: string | null; detail: Record<string, unknown> }
export interface AuditEntry { seq: number; ts: string; actor: string; action: string; subject: string; detail: Record<string, unknown>; entry_hash: string }
export interface Scenario {
  scenario_id: string; profile: string; paraphrased_evidence: boolean;
  request: { patient_ref: string; service_code: string; service_description: string; diagnosis_codes: string[]; urgency: "standard" | "expedited"; free_text: string };
  expected: { policy_id: string | null; recommendation: Recommendation; flags: string[]; broken_criterion: string | null };
}
