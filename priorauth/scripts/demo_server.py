"""
One-process demo: API + background worker on SQLite, synthetic data, deterministic agents. No Docker needed.

    python -m priorauth.scripts.demo_server [--port 8080] [--db demo.db]

Prints a paste-ready bearer token per role (valid 12 h). If PRIORAUTH_JWT_SECRET is unset an ephemeral one is
generated, so tokens stop working when the process exits. DEMO ONLY: the real deployment uses compose/ECS/EKS.
"""
from __future__ import annotations

import argparse
import os
import secrets
import threading
import time

import uvicorn

ROLES = ["playground", "submitter", "reviewer-rn", "reviewer-md", "auditor"]


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8080); ap.add_argument("--db", default="demo.db")
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    os.environ.setdefault("PRIORAUTH_JWT_SECRET", secrets.token_hex(32))
    os.environ["PRIORAUTH_DEMO"] = "1"
    os.environ["PRIORAUTH_DATABASE_URL"] = f"sqlite:///{a.db}"

    from priorauth.agents.orchestrator import Orchestrator
    from priorauth.api import create_app
    from priorauth.bootstrap import make_box
    from priorauth.mcp_server.auth import issue_token
    from priorauth.worker import process_one

    box = make_box()
    orch = Orchestrator(box)
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            if process_one(box.store, orch) is None:
                time.sleep(0.3)
    threading.Thread(target=loop, daemon=True).start()

    print(f"\nPrior-auth demo API (synthetic data only)  ->  http://{a.host}:{a.port}")
    print("Paste one of these tokens into the UI login (valid 12 h):\n")
    for r in ROLES:
        print(f"  {r:12} {issue_token(r, subject=f'demo-{r}', ttl=12 * 3600)}")
    print()
    try:
        uvicorn.run(create_app(box.store, demo=True), host=a.host, port=a.port, log_level="warning")
    finally:
        stop.set()


if __name__ == "__main__":
    main()
