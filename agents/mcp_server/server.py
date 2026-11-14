"""
MCP server (streamable HTTP) exposing the clinical-reasoning agent's
ToolBox -- mirrors priorauth/mcp_server/server.py. Every request must
carry a valid bearer JWT (BearerASGI); each tool then enforces its own
scope.

Run:  python -m agents.mcp_server.server
      (CLINICAL_AGENT_MCP_PORT, default 8766 -- one above priorauth's 8765
      so both tool servers can run side by side)
"""
from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

from agents.mcp_server.auth import BearerASGI
from agents.mcp_server.tools import TOOLS, ToolBox, _bind


def build_mcp(box: ToolBox) -> FastMCP:
    mcp = FastMCP("clinical-reasoning-tools", stateless_http=True, json_response=True)
    for name, spec in TOOLS.items():
        mcp.add_tool(_bind(box, name), name=name, description=spec["doc"])
    return mcp


def build_app(box: ToolBox):
    return BearerASGI(build_mcp(box).streamable_http_app())


def main() -> None:  # pragma: no cover
    import uvicorn
    from ehr.sync import EhrClient
    from serving.healthcare.audit import AuditLogger

    glucopulse_url = os.environ["GLUCOPULSE_URL"]
    glucopulse_token = os.environ["GLUCOPULSE_TOKEN"]
    box = ToolBox(fhir=EhrClient(glucopulse_url, glucopulse_token), audit=AuditLogger())
    uvicorn.run(build_app(box), host=os.getenv("CLINICAL_AGENT_MCP_HOST", "0.0.0.0"),
                port=int(os.getenv("CLINICAL_AGENT_MCP_PORT", "8766")))


if __name__ == "__main__":  # pragma: no cover
    main()
