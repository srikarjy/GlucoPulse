"""End-to-end smoke test against a RUNNING stack (docker-compose.priorauth.yml). Needs PRIORAUTH_JWT_SECRET.
   python -m priorauth.scripts.smoke [--api http://localhost:8080] [--prom http://localhost:9091]"""
from __future__ import annotations

import argparse
import sys
import time

import httpx

from priorauth.mcp_server.auth import issue_token
from priorauth.synth.scenarios import load


def H(role, sub=None, **k):
    return {"Authorization": f"Bearer {issue_token(role, sub)}", **k}


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--api", default="http://localhost:8080"); ap.add_argument("--prom", default="http://localhost:9091")
    ap.add_argument("--grafana", default="http://localhost:3001")
    a = ap.parse_args(); c = httpx.Client(timeout=30); sc = load()
    picks = {"meets": next(s for s in sc if s["profile"] == "meets" and not s["paraphrased_evidence"]),
             "fail": next(s for s in sc if s["profile"] == "fail"), "injection": next(s for s in sc if s["profile"] == "injection")}
    ids = {}
    for name, s in picks.items():
        r = s["request"]
        resp = c.post(a.api + "/v1/requests", headers=H("submitter", **{"Idempotency-Key": f"smoke-{time.time_ns()}-{name}"}),
                      json={k: r[k] for k in ("patient_ref", "service_code", "service_description", "diagnosis_codes", "urgency", "free_text")})
        check(resp.status_code == 202, f"submit {name}: {resp.status_code} {resp.text[:200]}")
        ids[name] = resp.json()["request_id"]
    tasks = {}
    for _ in range(40):
        time.sleep(1)
        q = c.get(a.api + "/v1/reviews", headers=H("reviewer-rn")).json()
        tasks = {t["request_id"]: t for t in q}
        if all(i in tasks for i in ids.values()):
            break
    check(all(i in tasks for i in ids.values()), "all 3 requests reached the human review queue")
    for name, rid in ids.items():
        t = tasks[rid]
        print(f"     {name:9} -> {t['draft']['recommendation']:18} conf={t['draft']['confidence']:<5} tier={t['tier']:9} flags={t['draft']['flags']}")
    check(tasks[ids["fail"]]["tier"] == "physician" and tasks[ids["injection"]]["tier"] == "physician", "denial-leaning and injected requests route to a physician")
    check(all(t["draft"]["status"] == "DRAFT" for t in tasks.values()), "every artifact is a DRAFT; nothing was decided by agents")
    status = c.get(f"{a.api}/v1/requests/{ids['fail']}", headers=H("submitter")).json()
    check(status["status"] == "awaiting_review" and "draft" not in status, "submitter sees status only, not the draft")
    tid = tasks[ids["fail"]]["task_id"]
    check(c.post(f"{a.api}/v1/reviews/{tid}/claim", headers=H("reviewer-rn", "rn-alice")).status_code == 403, "RN cannot claim a physician-tier task")
    check(c.post(f"{a.api}/v1/reviews/{tid}/claim", headers=H("reviewer-md", "dr-carol")).status_code == 200, "physician claims it")
    d = c.post(f"{a.api}/v1/reviews/{tid}/decision", headers=H("reviewer-md", "dr-carol"), json={"decision": "denied", "comment": "criterion not met"})
    check(d.status_code == 200 and d.json()["agreed_with_draft"], "physician records the FINAL decision")
    check(c.get(f"{a.api}/v1/requests/{ids['fail']}", headers=H("submitter")).json()["status"] == "decided", "request is decided only after the human decision")
    v = c.get(a.api + "/v1/audit/verify", headers=H("auditor")).json()
    check(v["intact"], f"audit hash chain intact: {v}")
    time.sleep(12)
    def prom(q):
        r = c.get(a.prom + "/api/v1/query", params={"query": q}).json()["data"]["result"]
        return float(r[0]["value"][1]) if r else 0.0
    check(prom("sum(pa_drafts_total)") >= 3, "Prometheus scraped worker pipeline metrics (pa_drafts_total >= 3)")
    check(prom("sum(pa_review_overrides_total)") == 0.0, "no overrides recorded")
    g = c.get(a.grafana + "/api/search?query=Prior", auth=("admin", "admin-dev-only")).json()
    check(any(x["uid"] == "priorauth-overview" for x in g), "Grafana dashboard provisioned")
    print("\nSMOKE OK")


if __name__ == "__main__":
    main()
