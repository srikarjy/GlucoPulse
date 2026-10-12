import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

// Cheap gate: no session cookie -> /login. The API is the real authority (it re-verifies the token on every call).
export function proxy(req: NextRequest) {
  if (!req.cookies.get("pa_token")) {
    const url = req.nextUrl.clone();
    url.pathname = "/login";
    url.search = "";
    return NextResponse.redirect(url);
  }
  return NextResponse.next();
}

export const config = { matcher: ["/((?!login|api|_next/static|_next/image|favicon.ico).*)"] };
