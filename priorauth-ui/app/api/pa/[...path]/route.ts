import { cookies } from "next/headers";
import { NextRequest, NextResponse } from "next/server";
import { ALLOWED, API_URL, COOKIE } from "@/lib/server-config";
import { sameOrigin } from "@/lib/origin";

// Authenticated pass-through to the prior-auth API. The bearer token is read from the httpOnly cookie here and attached
// server-side. Only an allow-list of API areas is reachable, and only a few request headers are forwarded.
async function handle(req: NextRequest, ctx: RouteContext<"/api/pa/[...path]">) {
  const { path } = await ctx.params;
  const rel = path.join("/");
  if (!ALLOWED.test(rel) || path.some((s) => s === ".." || s === ".")) {
    return NextResponse.json({ error: "not found" }, { status: 404 });
  }
  const mutating = req.method !== "GET" && req.method !== "HEAD";
  if (mutating && !sameOrigin(req)) return NextResponse.json({ error: "cross-origin request refused" }, { status: 403 });

  const token = (await cookies()).get(COOKIE)?.value;
  if (!token) return NextResponse.json({ error: "not signed in" }, { status: 401 });

  const headers: Record<string, string> = { Authorization: `Bearer ${token}`, Accept: "application/json" };
  const idem = req.headers.get("idempotency-key");
  if (idem) headers["Idempotency-Key"] = idem.slice(0, 200);
  let body: string | undefined;
  if (mutating) {
    headers["Content-Type"] = "application/json";
    body = await req.text();
    if (body.length > 100_000) return NextResponse.json({ error: "payload too large" }, { status: 413 });
  }
  const url = `${API_URL}/v1/${rel}${req.nextUrl.search}`;
  const upstream = await fetch(url, { method: req.method, headers, body, cache: "no-store", signal: AbortSignal.timeout(20_000) }).catch(() => null);
  if (!upstream) return NextResponse.json({ error: "API unreachable" }, { status: 502 });
  const text = await upstream.text();
  return new NextResponse(text, { status: upstream.status, headers: { "Content-Type": upstream.headers.get("content-type") ?? "application/json" } });
}

export { handle as GET, handle as POST };
