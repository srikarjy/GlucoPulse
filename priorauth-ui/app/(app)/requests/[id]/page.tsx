import { Suspense } from "react";
import { RequestClient } from "./client";

export default function Page() {
  return <Suspense fallback={<p>Loading…</p>}><RequestClient /></Suspense>;
}
