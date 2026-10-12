import "server-only";

export const API_URL = (process.env.PRIORAUTH_API_URL ?? "http://127.0.0.1:8080").replace(/\/$/, "");
export const COOKIE = "pa_token";
// Only these API areas may be reached through the browser-facing proxy.
export const ALLOWED = /^(me|requests|reviews|audit|demo)(\/|$)/;
export const cookieOptions = (maxAgeSeconds: number) => ({
  httpOnly: true,
  sameSite: "strict" as const,
  secure: process.env.NODE_ENV === "production",
  path: "/",
  maxAge: maxAgeSeconds,
});
