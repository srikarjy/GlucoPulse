"use client";
import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { api, ApiError } from "./api";
import type { Me } from "./types";

const MeCtx = createContext<Me | null>(null);
export const MeProvider = MeCtx.Provider;
export const useMe = () => useContext(MeCtx);
export const can = (me: Me | null, scope: string) => !!me?.scopes.includes(scope);

/** Poll an async loader. Stops on unmount; pauses while the tab is hidden; `stop(data)` ends polling (e.g. terminal status). */
export function usePoll<T>(load: () => Promise<T>, intervalMs: number, opts: { enabled?: boolean; stop?: (d: T) => boolean } = {}) {
  const { enabled = true, stop } = opts;
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<ApiError | Error | null>(null);
  const [loading, setLoading] = useState(true);
  const loadRef = useRef(load);
  const stopRef = useRef(stop);
  useEffect(() => { loadRef.current = load; stopRef.current = stop; });   // latest closures, read at call time by the poll loop
  const [tick, setTick] = useState(0);
  const reload = useCallback(() => setTick((t) => t + 1), []);

  useEffect(() => {
    if (!enabled) return;
    let alive = true; let timer: ReturnType<typeof setTimeout> | undefined;
    const run = async () => {
      if (typeof document !== "undefined" && document.hidden) { timer = setTimeout(run, intervalMs); return; }
      try {
        const d = await loadRef.current();
        if (!alive) return;
        setData(d); setError(null); setLoading(false);
        if (stopRef.current?.(d)) return;
      } catch (e) {
        if (!alive) return;
        setError(e as Error); setLoading(false);
        if (e instanceof ApiError && (e.status === 401 || e.status === 403 || e.status === 404)) return;   // don't hammer on auth/not-found
      }
      timer = setTimeout(run, intervalMs);
    };
    run();
    return () => { alive = false; if (timer) clearTimeout(timer); };
  }, [intervalMs, enabled, tick]);

  return { data, error, loading, reload };
}

export function useSignedInGuard() {
  const [me, setMe] = useState<Me | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const router = useRouter();
  useEffect(() => {
    api.get<Me>("me").then(setMe).catch((e: ApiError) => {
      if (e.status === 401) router.replace("/login"); else setErr(e.message);
    });
  }, [router]);
  return { me, err };
}
