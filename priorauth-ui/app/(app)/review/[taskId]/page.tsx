import { Suspense } from "react";
import { ReviewClient } from "./client";

export default function Page() {
  return <Suspense fallback={<p>Loading…</p>}><ReviewClient /></Suspense>;
}
