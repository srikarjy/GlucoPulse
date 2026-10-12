import { cookies } from "next/headers";
import { NextRequest, NextResponse } from "next/server";
import { API_URL, COOKIE, cookieOptions } from "@/lib/server-config";
import { sameOrigin } from "@/lib/origin";

// POST {token}: verify the token against the API, then store it in an httpOnly cookie. The browser never sees it again.
export async function POST(req: NextRequest) {
  if (!sameOrigin(req)) return NextResponse.json({ error: "cross-origin request refused" }, { status: 403 });
  const body = await req.json().catch(() => null);
  const token = typeof body?.token === "string" ? body.token.trim() : "";
  if (!token || token.length > 4096) return NextResponse.json({ error: "paste a token" }, { status: 400 });
  const r = await fetch(`${API_URL}/v1/me`, { headers: { Authorization: `Bearer ${token}` }, cache: "no-store" }).catch(() => null);
  if (!r) return NextResponse.json({ error: "API unreachable" }, { status: 502 });
  if (!r.ok) return NextResponse.json({ error: "token rejected by the API (wrong audience, expired, or invalid)" }, { status: 401 });
  const me = await r.json();
  (await cookies()).set(COOKIE, token, cookieOptions(12 * 3600));
  return NextResponse.json(me);
}

export async function DELETE(req: NextRequest) {
  if (!sameOrigin(req)) return NextResponse.json({ error: "cross-origin request refused" }, { status: 403 });
  (await cookies()).delete(COOKIE);
  return NextResponse.json({ ok: true });
}
