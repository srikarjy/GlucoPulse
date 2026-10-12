import { Suspense } from "react";
import { Shell } from "@/components/shell";

export default function AppLayout({ children }: { children: React.ReactNode }) {
  // Shell reads the pathname and session at request time, so it streams in behind a boundary (Cache Components).
  return <Suspense fallback={<p className="p-6">Loading…</p>}><Shell>{children}</Shell></Suspense>;
}
