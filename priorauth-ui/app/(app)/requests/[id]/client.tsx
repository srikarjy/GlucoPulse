"use client";
import Link from "next/link";
import { useParams } from "next/navigation";
import { DraftCard } from "@/components/draft-card";
import { AgentTimeline } from "@/components/timeline";
import { api } from "@/lib/api";
import { can, useMe, usePoll } from "@/lib/hooks";
import type { RequestStatus, TimelineEvent } from "@/lib/types";

const TERMINAL = ["awaiting_review", "failed", "decided"];

export function RequestClient() {
  const { id } = useParams<{ id: string }>();
  const me = useMe();
  const st = usePoll(() => api.get<RequestStatus>(`requests/${id}`), 1500, { stop: (s) => TERMINAL.includes(s.status) });
  const tl = usePoll(() => api.get<TimelineEvent[]>(`requests/${id}/timeline`), 1500, { stop: () => !!st.data && TERMINAL.includes(st.data.status) });
  if (st.error && !st.data) return <p role="alert" className="text-rose-800">{st.error.message}</p>;
  const done = !!st.data && TERMINAL.includes(st.data.status);
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Request <span className="font-mono">{id}</span> <span className="ml-2 rounded bg-slate-200 px-2 py-0.5 text-sm font-normal">{st.data?.status.replace(/_/g, " ") ?? "…"}</span></h1>
      {st.data?.status === "failed" && <p role="alert" className="rounded bg-rose-50 p-3 text-rose-900 ring-1 ring-rose-300">Automated processing failed after retries. This request needs manual handling by an operator.</p>}
      <section aria-labelledby="tl-h" className="rounded-lg border border-slate-300 bg-white p-4"><h2 id="tl-h" className="mb-3 text-base font-semibold">Agent pipeline</h2>
        <AgentTimeline events={tl.data ?? []} done={done && st.data?.status !== "failed"} failed={st.data?.status === "failed"} /></section>
      {st.data?.draft ? <DraftCard draft={st.data.draft} /> : done && !can(me, "review:read") ? <p className="text-sm text-slate-700">Drafts are visible to reviewers only. Status: {st.data?.status.replace(/_/g, " ")}.</p> : null}
      {can(me, "review:read") && st.data?.draft && <Link className="text-blue-800 underline" href={`/review/TASK-${id}`}>Open the review task →</Link>}
    </div>
  );
}
