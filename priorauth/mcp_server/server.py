"""
MCP server (streamable HTTP) exposing the ToolBox. Every request must carry a valid
bearer JWT (BearerASGI); each tool then enforces its own scope.

Run:  python -m priorauth.mcp_server.server   (PRIORAUTH_MCP_PORT, default 8765)
"""
from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

from priorauth.mcp_server.auth import BearerASGI
from priorauth.mcp_server.tools import TOOLS, ToolBox, _bind


def build_mcp(box: ToolBox) -> FastMCP:
    mcp = FastMCP("priorauth-tools", stateless_http=True, json_response=True)
    for name, spec in TOOLS.items():
        mcp.add_tool(_bind(box, name), name=name, description=spec["doc"])
    return mcp


def build_app(box: ToolBox):
    return BearerASGI(build_mcp(box).streamable_http_app())


def main() -> None:  # pragma: no cover
    import uvicorn
    from priorauth.bootstrap import make_box
    uvicorn.run(build_app(make_box()), host=os.getenv("PRIORAUTH_MCP_HOST", "0.0.0.0"),
                port=int(os.getenv("PRIORAUTH_MCP_PORT", "8765")))


if __name__ == "__main__":  # pragma: no cover
    main()
