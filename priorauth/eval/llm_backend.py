"""Run the 200-scenario eval through the Strands/Bedrock agents instead of the deterministic ones.

The eval's ToolBox is exposed over a real MCP HTTP server on a loopback port (same auth + tools as production), and
the five Strands stages talk to it with per-role tokens. Cost: ~5 model calls (+ tool turns) per scenario; use --limit first.
"""
from __future__ import annotations

import socket
import threading
import time
from collections.abc import Callable

import uvicorn

from priorauth.agents.orchestrator import Orchestrator
from priorauth.agents.strands_backend import build_strands_agents, default_model
from priorauth.mcp_server.server import build_app


def make_factory(model_factory: Callable = default_model, per_stage: Callable | None = None):
    """Returns factory(box) -> Orchestrator. `per_stage(stage_name) -> model_factory` overrides the model per stage (tests)."""
    def factory(box):
        s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
        srv = uvicorn.Server(uvicorn.Config(build_app(box), host="127.0.0.1", port=port, log_level="error"))
        threading.Thread(target=srv.run, daemon=True).start()
        for _ in range(100):
            if srv.started:
                break
            time.sleep(0.05)
        stages = build_strands_agents(f"http://127.0.0.1:{port}/mcp", box.store, model_factory)
        if per_stage:
            for st in stages:
                st.model_factory = per_stage(st.name)
        orch = Orchestrator(box, agents=stages)
        orch._server = srv        # keep a handle so callers can stop it
        return orch
    return factory
