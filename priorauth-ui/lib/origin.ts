import type { NextRequest } from "next/server";

/** State-changing requests must come from this app's own origin (defence in depth on top of SameSite=Strict). */
export function sameOrigin(req: NextRequest): boolean {
  const origin = req.headers.get("origin");
  if (!origin) return false;
  const host = req.headers.get("x-forwarded-host") ?? req.headers.get("host");
  try {
    return new URL(origin).host === host;
  } catch {
    return false;
  }
}
