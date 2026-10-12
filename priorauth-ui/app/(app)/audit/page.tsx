"use client";
import { useState } from "react";
import { api } from "@/lib/api";
import { can, useMe, usePoll } from "@/lib/hooks";
import type { AuditEntry } from "@/lib/types";

export default function Audit() {
  const me = useMe();
  const [rid, setRid] = useState(""); const [q, setQ] = useState("");
  const [verify, setVerify] = useState<{ intact: boolean; first_broken_seq: number | null } | null>(null);
  const { data, error, loading } = usePoll(() => api.get<AuditEntry[]>(`audit?limit=200${q ? `&request_id=${encodeURIComponent(q)}` : ""}`), 8000, { enabled: can(me, "audit:read") });
  if (!can(me, "audit:read")) return <p role="alert">Only auditors can view the audit trail.</p>;
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Audit trail</h1>
      <p className="text-sm text-slate-700">Append-only and hash-chained. Entries hold ids, tool names and outcomes, never clinical text.</p>
      <div className="flex flex-wrap items-end gap-3">
        <form className="flex items-end gap-2" onSubmit={(e) => { e.preventDefault(); setQ(rid.trim()); }}>
          <label className="text-sm">Request id<input value={rid} onChange={(e) => setRid(e.target.value)} placeholder="REQ-…" className="ml-2 rounded border border-slate-400 px-2 py-1 font-mono text-xs" /></label>
          <button className="rounded border border-slate-500 px-3 py-1 text-sm">Filter</button>
        </form>
        <button className="rounded bg-slate-900 px-3 py-1 text-sm text-white" onClick={async () => setVerify(await api.get("audit/verify"))}>Verify hash chain</button>
        {verify && <span role="status" className={`text-sm font-medium ${verify.intact ? "text-emerald-800" : "text-rose-800"}`}>{verify.intact ? "✓ Chain intact" : `✕ Chain broken at entry ${verify.first_broken_seq}`}</span>}
      </div>
      {error && <p role="alert" className="text-rose-800">{error.message}</p>}
      {loading ? <p>Loading…</p> : (
        <div className="overflow-x-auto rounded-lg border border-slate-300 bg-white">
          <table className="w-full text-left text-xs">
            <caption className="sr-only">Audit entries, newest first</caption>
            <thead className="bg-slate-100 uppercase text-slate-700"><tr><th scope="col" className="px-3 py-2">#</th><th scope="col">Time</th><th scope="col">Actor</th><th scope="col">Action</th><th scope="col">Subject</th><th scope="col">Hash</th></tr></thead>
            <tbody>{data?.map((e) => (
              <tr key={e.seq} className="border-t border-slate-200 font-mono">
                <td className="px-3 py-1">{e.seq}</td><td>{e.ts.replace("T", " ").slice(0, 19)}</td><td>{e.actor}</td>
                <td>{e.action}{e.detail?.tool ? ` · ${String(e.detail.tool)}` : ""}</td><td>{e.subject}</td><td title={e.entry_hash}>{e.entry_hash.slice(0, 10)}…</td>
              </tr>))}</tbody>
          </table>
        </div>
      )}
    </div>
  );
}
