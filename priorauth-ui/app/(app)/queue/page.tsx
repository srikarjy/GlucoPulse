"use client";
import Link from "next/link";
import { useState } from "react";
import { FlagChip, RecBadge, TierBadge } from "@/components/badges";
import { api } from "@/lib/api";
import { can, useMe, usePoll } from "@/lib/hooks";
import type { ReviewTask } from "@/lib/types";

const STATES = ["open", "claimed", "closed"] as const;
const age = (iso: string) => { const m = Math.max(0, Math.round((Date.now() - new Date(iso).getTime()) / 60000)); return m < 60 ? `${m}m` : `${Math.round(m / 60)}h`; };

export default function Queue() {
  const me = useMe();
  const [state, setState] = useState<(typeof STATES)[number]>("open");
  const [tier, setTier] = useState("");
  const { data, error, loading } = usePoll(() => api.get<ReviewTask[]>(`reviews?state=${state}${tier ? `&tier=${tier}` : ""}`), 4000, { enabled: can(me, "review:read") });
  if (!can(me, "review:read")) return <p role="alert">Your role cannot view the review queue.</p>;
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <h1 className="text-xl font-semibold">Review queue</h1>
        <div className="flex gap-3 text-sm">
          <div role="tablist" aria-label="Task state" className="flex overflow-hidden rounded border border-slate-400">
            {STATES.map((s) => <button key={s} role="tab" aria-selected={state === s} onClick={() => setState(s)} className={`px-3 py-1 capitalize ${state === s ? "bg-slate-900 text-white" : "bg-white hover:bg-slate-100"}`}>{s}</button>)}
          </div>
          <div className="flex items-center gap-1">
            <label htmlFor="tier-f">Tier</label>
            <select id="tier-f" value={tier} onChange={(e) => setTier(e.target.value)} className="rounded border border-slate-400 bg-white px-1 py-1">
              <option value="">all</option><option value="standard">standard</option><option value="priority">priority</option><option value="physician">physician</option>
            </select>
          </div>
        </div>
      </div>
      {error && <p role="alert" className="text-rose-800">Could not load the queue: {error.message}</p>}
      {loading ? <p>Loading…</p> : !data?.length ? <p className="rounded border border-dashed border-slate-400 p-6 text-center text-slate-700">No {state} tasks{tier ? ` in the ${tier} tier` : ""}.</p> : (
        <div className="overflow-x-auto rounded-lg border border-slate-300 bg-white">
          <table className="w-full text-left text-sm">
            <caption className="sr-only">Review tasks, most urgent first</caption>
            <thead className="bg-slate-100 text-xs uppercase text-slate-700"><tr>
              <th scope="col" className="px-3 py-2">Pri</th><th scope="col">Tier</th><th scope="col">Request</th><th scope="col">AI draft (advisory)</th><th scope="col">Flags</th><th scope="col">Age</th><th scope="col">Owner</th></tr></thead>
            <tbody>
              {data.map((t) => (
                <tr key={t.task_id} className="border-t border-slate-200 align-top hover:bg-slate-50">
                  <td className="px-3 py-2 tabular-nums">{t.priority === 1 ? <span className="font-semibold text-rose-800">1 · expedited</span> : t.priority}</td>
                  <td><TierBadge tier={t.tier} /></td>
                  <td><Link className="font-mono text-xs text-blue-800 underline" href={`/review/${t.task_id}`}>{t.request_id}</Link></td>
                  <td><RecBadge rec={t.draft.recommendation} /></td>
                  <td><div className="flex flex-wrap gap-1">{t.draft.flags.map((f) => <FlagChip key={f} flag={f} />)}</div></td>
                  <td className="tabular-nums">{age(t.created_at)}</td>
                  <td className="text-xs">{t.claimed_by ?? "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
