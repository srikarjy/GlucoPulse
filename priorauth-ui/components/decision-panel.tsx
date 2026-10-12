"use client";
import { useState } from "react";
import { api } from "@/lib/api";
import { can, useMe } from "@/lib/hooks";
import type { FinalDecision, Recommendation, ReviewTask } from "@/lib/types";

const OPTIONS: { v: FinalDecision; label: string; help: string }[] = [
  { v: "approved", label: "Approve", help: "Authorize the requested service." },
  { v: "pended_for_info", label: "Pend for information", help: "Request the missing documentation before deciding." },
  { v: "denied", label: "Deny", help: "Physician reviewers only. Requires a written reason." },
];
const IMPLIED: Record<Recommendation, FinalDecision> = { supports_approval: "approved", does_not_support: "denied", insufficient_info: "pended_for_info" };

export function DecisionPanel({ task, onChange }: { task: ReviewTask; onChange: () => void }) {
  const me = useMe();
  const [choice, setChoice] = useState<FinalDecision | null>(null);   // deliberately no default: the draft must not anchor the reviewer
  const [comment, setComment] = useState("");
  const [confirm, setConfirm] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const canDecide = can(me, "review:decide");
  const isMD = can(me, "review:decide:denial");
  const mine = task.claimed_by === me?.subject;
  const disagrees = choice !== null && choice !== IMPLIED[task.draft.recommendation];
  const commentRequired = choice === "denied" || disagrees;
  const commentOk = !commentRequired || comment.trim().length >= 10;

  const act = async (fn: () => Promise<unknown>) => { setBusy(true); setErr(null); try { await fn(); setConfirm(false); onChange(); } catch (e) { setErr((e as Error).message); } finally { setBusy(false); } };

  if (task.state === "closed") {
    const d = task.decision;
    return (
      <section aria-labelledby="dec-h" className="rounded-lg border border-slate-300 bg-white p-4">
        <h2 id="dec-h" className="text-base font-semibold">Final decision (human)</h2>
        {d ? <dl className="mt-2 grid grid-cols-[8rem_1fr] gap-y-1 text-sm">
          <dt className="text-slate-600">Decision</dt><dd className="font-semibold">{d.decision.replace(/_/g, " ")}</dd>
          <dt className="text-slate-600">Decided by</dt><dd>{d.reviewer}</dd>
          <dt className="text-slate-600">Draft</dt><dd>{d.agreed ? "Reviewer agreed with the AI draft" : "Reviewer overrode the AI draft"}</dd>
          {d.comment && <><dt className="text-slate-600">Reason</dt><dd className="whitespace-pre-wrap">{d.comment}</dd></>}
        </dl> : <p className="text-sm">Closed.</p>}
      </section>
    );
  }

  if (!canDecide) return <section className="rounded-lg border border-slate-300 bg-white p-4 text-sm" aria-labelledby="dec-h"><h2 id="dec-h" className="font-semibold">Decision</h2><p className="mt-1 text-slate-700">Your role is read-only here; only reviewers can claim or decide.</p></section>;

  if (task.state === "open") {
    const blocked = task.tier === "physician" && !isMD;
    return (
      <section aria-labelledby="dec-h" className="rounded-lg border border-slate-300 bg-white p-4">
        <h2 id="dec-h" className="text-base font-semibold">Take this review</h2>
        {blocked ? <p className="mt-1 text-sm text-violet-900">This task requires a physician reviewer. Your role cannot claim it.</p>
          : <p className="mt-1 text-sm text-slate-700">Claim the task to record a decision under your identity ({me?.subject}).</p>}
        {err && <p role="alert" className="mt-2 text-sm text-rose-800">{err}</p>}
        <button disabled={blocked || busy} onClick={() => act(() => api.post(`reviews/${task.task_id}/claim`))}
          className="mt-3 rounded bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-50">{busy ? "Claiming…" : "Claim review"}</button>
      </section>
    );
  }

  if (!mine) return <section className="rounded-lg border border-slate-300 bg-white p-4 text-sm" aria-labelledby="dec-h"><h2 id="dec-h" className="font-semibold">Decision</h2><p className="mt-1">Claimed by <strong>{task.claimed_by}</strong>. Only the claimant can decide.</p></section>;

  return (
    <section aria-labelledby="dec-h" className="rounded-lg border border-slate-300 bg-white p-4">
      <h2 id="dec-h" className="text-base font-semibold">Your decision</h2>
      <p className="mt-1 text-sm text-slate-700">Decide from the evidence, not the draft. No option is preselected. Your decision is final and recorded under <strong>{me?.subject}</strong>.</p>
      <fieldset className="mt-3 space-y-2" disabled={busy}>
        <legend className="sr-only">Final decision</legend>
        {OPTIONS.map((o) => {
          const off = o.v === "denied" && !isMD;
          return (
            <label key={o.v} className={`flex items-start gap-2 rounded border p-2 text-sm ${choice === o.v ? "border-slate-900 bg-slate-50" : "border-slate-300"} ${off ? "opacity-50" : ""}`}>
              <input type="radio" name="decision" className="mt-1" disabled={off} checked={choice === o.v} onChange={() => { setChoice(o.v); setConfirm(false); }} />
              <span><span className="font-medium">{o.label}</span><span className="block text-xs text-slate-600">{off ? "Only a physician reviewer can deny." : o.help}</span></span>
            </label>
          );
        })}
      </fieldset>
      <div className="mt-3">
        <label htmlFor="cm" className="text-sm font-medium">Reason {commentRequired ? <span className="text-rose-800">(required, at least 10 characters{disagrees ? "; you are overriding the draft" : ""})</span> : "(optional)"}</label>
        <textarea id="cm" rows={3} value={comment} maxLength={4000} onChange={(e) => setComment(e.target.value)} className="mt-1 w-full rounded border border-slate-400 p-2 text-sm" />
      </div>
      {err && <p role="alert" className="mt-2 text-sm text-rose-800">{err}</p>}
      {!confirm ? (
        <button disabled={!choice || !commentOk || busy} onClick={() => setConfirm(true)} className="mt-3 rounded bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-50">Review decision</button>
      ) : (
        <div role="alertdialog" aria-label="Confirm final decision" className="mt-3 rounded border border-amber-400 bg-amber-50 p-3 text-sm">
          <p>Record <strong>{choice?.replace(/_/g, " ")}</strong> as the final decision for {task.request_id}? This cannot be undone.</p>
          <div className="mt-2 flex gap-2">
            <button disabled={busy} onClick={() => act(() => api.post(`reviews/${task.task_id}/decision`, { decision: choice, comment: comment.trim() }))} className="rounded bg-emerald-800 px-3 py-1.5 font-medium text-white disabled:opacity-50">{busy ? "Recording…" : "Confirm final decision"}</button>
            <button disabled={busy} onClick={() => setConfirm(false)} className="rounded border border-slate-500 px-3 py-1.5">Back</button>
          </div>
        </div>
      )}
    </section>
  );
}
