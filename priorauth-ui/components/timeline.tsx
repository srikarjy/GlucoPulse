import type { TimelineEvent } from "@/lib/types";

const STAGES = [
  { actor: "intake-agent", label: "Intake", does: "validates codes" },
  { actor: "policy-agent", label: "Policy retrieval", does: "finds the coverage policy" },
  { actor: "evidence-agent", label: "Evidence validation", does: "checks each criterion against the chart" },
  { actor: "drafting-agent", label: "Decision drafting", does: "stages an advisory draft" },
  { actor: "escalation-agent", label: "Escalation", does: "routes to a human reviewer" },
];

export function AgentTimeline({ events, done, failed }: { events: TimelineEvent[]; done: boolean; failed?: boolean }) {
  const calls = events.filter((e) => e.action === "tool.call" || e.action === "tool.denied" || e.action === "tool.error");
  const reached = STAGES.map((s) => calls.filter((c) => c.actor === s.actor));
  const lastActive = reached.map((r) => r.length > 0).lastIndexOf(true);
  return (
    <ol className="space-y-2" aria-label="Agent pipeline progress" aria-live="polite">
      {STAGES.map((s, i) => {
        const n = reached[i].length;
        const complete = n > 0 && (i < lastActive || done);
        const active = !done && !failed && i === lastActive + 1 && lastActive < STAGES.length - 1 || (!done && !failed && i === lastActive && !complete);
        const tools = [...new Set(reached[i].map((c) => c.tool).filter(Boolean))] as string[];
        return (
          <li key={s.actor} className="flex gap-3">
            <span aria-hidden className={`mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs font-bold ${complete ? "bg-emerald-600 text-white" : active ? "animate-pulse bg-sky-600 text-white" : "bg-slate-200 text-slate-600"}`}>{complete ? "✓" : i + 1}</span>
            <div>
              <p className="text-sm font-medium">{s.label} <span className="font-normal text-slate-600">: {s.does}</span><span className="sr-only">{complete ? " (done)" : active ? " (running)" : " (waiting)"}</span></p>
              {tools.length > 0 && <p className="font-mono text-xs text-slate-600">{tools.join(" · ")}</p>}
            </div>
          </li>
        );
      })}
    </ol>
  );
}
