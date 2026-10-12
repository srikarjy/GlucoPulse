import type { Draft } from "@/lib/types";
import { Confidence, FlagChip, flagHelp, RecBadge } from "./badges";

const MET = (m: boolean | null) =>
  m === true ? { t: "Met", cls: "text-emerald-800", i: "✓" } : m === false ? { t: "Not met", cls: "text-rose-800", i: "✕" } : { t: "Cannot determine", cls: "text-amber-800", i: "?" };

export function DraftCard({ draft }: { draft: Draft }) {
  return (
    <section aria-labelledby="draft-h" className="rounded-lg border border-slate-300 bg-white">
      <div role="note" className="rounded-t-lg border-b border-amber-300 bg-amber-50 px-4 py-2 text-sm text-amber-950">
        <strong>AI-generated draft, advisory only.</strong> Nothing here is a decision. A human reviewer decides every request.
      </div>
      <div className="space-y-4 p-4">
        <div className="flex flex-wrap items-center gap-3">
          <h2 id="draft-h" className="text-base font-semibold">Draft recommendation</h2>
          <RecBadge rec={draft.recommendation} />
          <Confidence value={draft.confidence} />
          <span className="text-xs text-slate-600">policy <code>{draft.policy_id || "none matched"}</code></span>
        </div>

        {draft.flags.length > 0 && (
          <div>
            <h3 className="mb-1 text-sm font-medium">Flags</h3>
            <ul className="space-y-1">
              {draft.flags.map((f) => (
                <li key={f} className="flex items-start gap-2 text-sm"><FlagChip flag={f} /><span className="text-slate-700">{flagHelp(f)}</span></li>
              ))}
            </ul>
          </div>
        )}

        <div>
          <h3 className="mb-1 text-sm font-medium">Rationale</h3>
          <p className="whitespace-pre-wrap text-sm text-slate-800">{draft.rationale}</p>
        </div>

        {draft.criteria_summary.length > 0 && (
          <div className="overflow-x-auto">
            <h3 className="mb-1 text-sm font-medium">Criteria and cited evidence</h3>
            <table className="w-full text-left text-sm">
              <caption className="sr-only">Policy criteria with verification result and evidence references</caption>
              <thead className="text-xs uppercase text-slate-600"><tr><th scope="col" className="py-1 pr-3">Criterion</th><th scope="col" className="pr-3">Result</th><th scope="col">Evidence (references only)</th></tr></thead>
              <tbody>
                {draft.criteria_summary.map((c) => {
                  const m = MET(c.met);
                  return (
                    <tr key={c.criterion_id} className="border-t border-slate-200 align-top">
                      <th scope="row" className="py-1.5 pr-3 font-mono text-xs font-normal">{c.criterion_id}</th>
                      <td className={`pr-3 ${m.cls}`}><span aria-hidden>{m.i} </span>{m.t}{c.note ? <span className="block text-xs text-slate-600">{c.note}</span> : null}</td>
                      <td className="font-mono text-xs text-slate-700">{c.evidence.length ? c.evidence.join(", ") : "-"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {draft.missing_information.length > 0 && (
          <div>
            <h3 className="mb-1 text-sm font-medium">Missing information</h3>
            <ul className="list-disc pl-5 text-sm text-slate-800">{draft.missing_information.map((m) => <li key={m}>{m}</li>)}</ul>
          </div>
        )}
      </div>
    </section>
  );
}
