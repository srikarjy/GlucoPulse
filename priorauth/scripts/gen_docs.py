"""Regenerate docs/priorauth/MCP_TOOLS.md from the live tool registry:  python -m priorauth.scripts.gen_docs"""
from __future__ import annotations

import inspect
from pathlib import Path

from priorauth.mcp_server.auth import ROLE_SCOPES
from priorauth.mcp_server.tools import TOOLS

OUT = Path(__file__).resolve().parents[2] / "docs" / "priorauth" / "MCP_TOOLS.md"


def render() -> str:
    L = ["# MCP tool catalog", "", "_Generated from `priorauth/mcp_server/tools.py` by `python -m priorauth.scripts.gen_docs`; a test fails if this file is stale._", "",
         f"{len(TOOLS)} tools. Every call requires a bearer JWT; the tool's **scope** must be in the caller's role. Mutating tools require an `idempotency_key` "
         "(same key + same args replays the stored result; same key + different args is rejected). Every call, denial, replay and error is written to the audit chain.", "",
         "| Tool | Scope | Mutating | Parameters | Purpose |", "|---|---|---|---|---|"]
    for n, t in TOOLS.items():
        ps = [p for p in inspect.signature(t["fn"]).parameters if p != "self"]
        L.append(f"| `{n}` | `{t['scope']}` | {'yes' if t['mutating'] else 'no'} | {', '.join(f'`{p}`' for p in ps)} | {t['doc'].splitlines()[0] if t['doc'] else ''} |")
    L += ["", "## Roles (least privilege)", "", "| Agent role | Scopes | Tools it can call |", "|---|---|---|"]
    for role, sc in ROLE_SCOPES.items():
        L.append(f"| `{role}` | {', '.join(f'`{s}`' for s in sorted(sc))} | {', '.join(f'`{n}`' for n, t in TOOLS.items() if t['scope'] in sc)} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render())
    print(f"wrote {OUT}")
