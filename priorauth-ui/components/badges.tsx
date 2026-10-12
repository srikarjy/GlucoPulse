import type { Recommendation, Tier } from "@/lib/types";

const REC: Record<Recommendation, { label: string; cls: string; icon: string }> = {
  supports_approval: { label: "Draft: supports approval", cls: "bg-emerald-50 text-emerald-900 ring-emerald-300", icon: "✓" },
  does_not_support: { label: "Draft: does not support", cls: "bg-rose-50 text-rose-900 ring-rose-300", icon: "✕" },
  insufficient_info: { label: "Draft: insufficient information", cls: "bg-amber-50 text-amber-900 ring-amber-300", icon: "?" },
};
export function RecBadge({ rec, kind = "draft" }: { rec: Recommendation; kind?: "draft" | "expected" }) {
  const r = REC[rec];
  const label = kind === "expected" ? r.label.replace("Draft:", "Expected label:") : r.label;   // ground truth must never read as a draft
  return <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ${r.cls}`}><span aria-hidden>{r.icon}</span>{label}</span>;
}

const TIER: Record<Tier, { label: string; cls: string }> = {
  standard: { label: "Standard review", cls: "bg-slate-100 text-slate-800 ring-slate-300" },
  priority: { label: "Priority review", cls: "bg-sky-50 text-sky-900 ring-sky-300" },
  physician: { label: "Physician required", cls: "bg-violet-50 text-violet-900 ring-violet-300" },
};
export function TierBadge({ tier }: { tier: Tier }) {
  const t = TIER[tier];
  return <span className={`inline-flex rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ${t.cls}`}>{t.label}</span>;
}

const FLAG_HELP: Record<string, string> = {
  prompt_injection: "Instruction-like text was found in chart or letter and was ignored as evidence.",
  conflicting_data: "Contradictory results on the same date.",
  code_mismatch: "Diagnosis codes do not fit the requested service.",
  stale_evidence: "A required result is older than the policy window.",
  no_policy: "No coverage policy matched this service.",
  evidence_discrepancy: "The drafting model's claim did not match verified evidence; rules prevailed.",
  recommendation_clamped: "The model was more favourable than the rules allow; lowered.",
  agent_failure: "Automated analysis failed; full manual review needed.",
  coverage_inactive: "Coverage is not active on the chart.",
  policy_matched_semantically: "Policy matched by description, not by exact code.",
};
export function FlagChip({ flag }: { flag: string }) {
  const serious = ["prompt_injection", "conflicting_data", "agent_failure"].includes(flag);
  return (
    <span title={FLAG_HELP[flag] ?? flag}
      className={`inline-flex items-center gap-1 rounded px-2 py-0.5 text-xs font-mono ring-1 ${serious ? "bg-rose-50 text-rose-900 ring-rose-300" : "bg-slate-100 text-slate-800 ring-slate-300"}`}>
      <span aria-hidden>{serious ? "⚠" : "•"}</span>{flag}
    </span>
  );
}
export const flagHelp = (f: string) => FLAG_HELP[f];

export function Confidence({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  return (
    <div className="flex items-center gap-2" title="Heuristic score, not a calibrated probability">
      <div className="h-1.5 w-24 rounded bg-slate-200" role="meter" aria-label="draft confidence" aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct}>
        <div className="h-1.5 rounded bg-slate-600" style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs tabular-nums text-slate-700">{pct}%</span>
    </div>
  );
}
