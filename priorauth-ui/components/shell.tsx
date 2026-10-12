"use client";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { can, MeProvider, useSignedInGuard } from "@/lib/hooks";
import { signOut } from "@/lib/api";

export function Shell({ children }: { children: React.ReactNode }) {
  const { me, err } = useSignedInGuard();
  const path = usePathname();
  const router = useRouter();
  const nav = [
    { href: "/queue", label: "Review queue", show: can(me, "review:read") },
    { href: "/playground", label: "Playground", show: can(me, "request:submit") },
    { href: "/audit", label: "Audit", show: can(me, "audit:read") },
  ].filter((n) => n.show);
  return (
    <div className="mx-auto flex min-h-screen max-w-6xl flex-col px-4">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:bg-white focus:p-2">Skip to content</a>
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-300 py-3">
        <div className="flex items-center gap-6">
          <Link href="/" className="font-semibold">Prior Auth Workbench</Link>
          <nav aria-label="Main" className="flex gap-4 text-sm">
            {nav.map((n) => (
              <Link key={n.href} href={n.href} aria-current={path.startsWith(n.href) ? "page" : undefined}
                className={path.startsWith(n.href) ? "font-semibold underline underline-offset-4" : "text-slate-700 hover:underline"}>{n.label}</Link>
            ))}
          </nav>
        </div>
        <div className="flex items-center gap-3 text-sm">
          {me && <span className="rounded bg-slate-200 px-2 py-0.5 font-mono text-xs" title={me.scopes.join(", ")}>{me.role} · {me.subject}</span>}
          <button className="rounded border border-slate-400 px-2 py-1 hover:bg-slate-100"
            onClick={async () => { await signOut(); router.replace("/login"); }}>Sign out</button>
        </div>
      </header>
      <div role="status" className="mt-2 rounded bg-slate-200 px-3 py-1 text-xs text-slate-800">
        Synthetic data only. Not for clinical use. Drafts are advisory; a human makes every decision.
      </div>
      <main id="main" className="flex-1 py-6">
        {err ? <p role="alert" className="text-rose-800">Could not reach the API: {err}</p> : me ? <MeProvider value={me}>{children}</MeProvider> : <p>Loading…</p>}
      </main>
    </div>
  );
}
