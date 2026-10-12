"use client";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { signIn } from "@/lib/api";

export default function Login() {
  const [token, setToken] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const router = useRouter();
  return (
    <main className="mx-auto max-w-xl px-4 py-16">
      <h1 className="text-2xl font-semibold">Prior Auth Review Workbench</h1>
      <p className="mt-2 text-sm text-slate-700">Synthetic data only. Paste an API bearer token for your role. It is verified by the API, then stored in an httpOnly cookie and never exposed to page scripts.</p>
      <form className="mt-6 space-y-3" onSubmit={async (e) => {
        e.preventDefault(); setBusy(true); setErr(null);
        try { await signIn(token); router.replace("/"); } catch (x) { setErr((x as Error).message); setBusy(false); }
      }}>
        <label htmlFor="tok" className="block text-sm font-medium">API token</label>
        <textarea id="tok" required rows={5} value={token} onChange={(e) => setToken(e.target.value)} spellCheck={false} autoComplete="off"
          className="w-full rounded border border-slate-400 bg-white p-2 font-mono text-xs" aria-describedby="tok-help" />
        <p id="tok-help" className="text-xs text-slate-600">Local demo: run <code>python -m priorauth.scripts.demo_server</code>; it prints a token per role (playground, submitter, reviewer-rn, reviewer-md, auditor).</p>
        {err && <p role="alert" className="rounded bg-rose-50 p-2 text-sm text-rose-900 ring-1 ring-rose-300">{err}</p>}
        <button disabled={busy || !token.trim()} className="rounded bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-50">{busy ? "Verifying…" : "Sign in"}</button>
      </form>
    </main>
  );
}
