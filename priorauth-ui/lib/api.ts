// Browser-side API client. Every call goes to this app's own /api/pa proxy; the bearer token is never visible here.
export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message); }
}

async function call<T>(method: string, path: string, body?: unknown, headers: Record<string, string> = {}): Promise<T> {
  const res = await fetch(`/api/pa/${path}`, {
    method, cache: "no-store", credentials: "same-origin",
    headers: { ...(body !== undefined ? { "Content-Type": "application/json" } : {}), ...headers },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  let data: unknown = null;
  try { data = text ? JSON.parse(text) : null; } catch { /* non-JSON error body */ }
  if (!res.ok) {
    const d = data as { detail?: unknown; error?: string } | null;
    const msg = typeof d?.detail === "string" ? d.detail : d?.error ?? `request failed (${res.status})`;
    throw new ApiError(res.status, msg);
  }
  return data as T;
}

export const api = {
  get: <T,>(path: string) => call<T>("GET", path),
  post: <T,>(path: string, body?: unknown, headers?: Record<string, string>) => call<T>("POST", path, body ?? {}, headers),
};

export async function signIn(token: string) {
  const res = await fetch("/api/session", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ token }) });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new ApiError(res.status, data.error ?? "sign-in failed");
  return data as import("./types").Me;
}
export async function signOut() { await fetch("/api/session", { method: "DELETE" }); }
