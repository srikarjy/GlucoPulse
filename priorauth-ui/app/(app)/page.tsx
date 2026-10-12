"use client";
import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { can, useMe } from "@/lib/hooks";

export default function Home() {
  const me = useMe(); const router = useRouter();
  useEffect(() => {
    if (!me) return;
    router.replace(can(me, "review:read") ? "/queue" : can(me, "request:submit") ? "/playground" : "/audit");
  }, [me, router]);
  return <p>Redirecting…</p>;
}
