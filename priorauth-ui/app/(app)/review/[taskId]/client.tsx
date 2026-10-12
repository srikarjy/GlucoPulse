"use client";
import Link from "next/link";
import { useParams } from "next/navigation";
import { DraftCard } from "@/components/draft-card";
import { TierBadge } from "@/components/badges";
import { DecisionPanel } from "@/components/decision-panel";
import { api } from "@/lib/api";
import { can, useMe, usePoll } from "@/lib/hooks";
import type { ReviewTask } from "@/lib/types";

export function ReviewClient() {
  const { taskId } = useParams<{ taskId: string }>();
  const me = useMe();
  const { data: task, error, reload } = usePoll(() => api.get<ReviewTask>(`reviews/${taskId}`), 5000, { enabled: can(me, "review:read"), stop: (t) => t.state === "closed" });
  if (!can(me, "review:read")) return <p role="alert">Your role cannot view review tasks.</p>;
  if (error && !task) return <p role="alert" className="text-rose-800">{error.message}. <Link className="underline" href="/queue">Back to queue</Link></p>;
  if (!task) return <p>Loading…</p>;
  return (
    <div className="space-y-4">
      <div><Link href="/queue" className="text-sm text-blue-800 underline">← Review queue</Link></div>
      <header className="flex flex-wrap items-center gap-3">
        <h1 className="text-xl font-semibold">Request <span className="font-mono">{task.request_id}</span></h1>
        <TierBadge tier={task.tier} />
        <span className="text-sm text-slate-700">priority {task.priority}{task.priority === 1 ? " (expedited)" : ""}</span>
      </header>
      <section aria-label="Why this was routed to you" className="rounded border border-slate-300 bg-white p-3 text-sm">
        <span className="font-medium">Routing reasons: </span>
        {task.reasons.map((r) => <code key={r} className="mr-2 rounded bg-slate-100 px-1">{r}</code>)}
        <Link className="ml-2 text-blue-800 underline" href={`/requests/${task.request_id}`}>agent timeline</Link>
      </section>
      <div className="grid gap-4 lg:grid-cols-[3fr_2fr]">
        <DraftCard draft={task.draft} />
        <DecisionPanel task={task} onChange={reload} />
      </div>
    </div>
  );
}
