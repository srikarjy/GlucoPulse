"""
Live-model demo runner: asks the clinical-reasoning agent an ARBITRARY
question (not one of the fixed eval scenarios) using a real model
(OpenAI or Anthropic, whichever key is set -- see
agents.reasoning_agent.default_model), against the real facade/MCP
server.

Requires OPENAI_API_KEY or ANTHROPIC_API_KEY. Run (needs .venv-pa):
    OPENAI_API_KEY=sk-... .venv-pa/bin/python -m agents.demo_live "your question" [patient_id]
"""
from __future__ import annotations

import os
import sys

from agents.eval.run_eval import build_environment
from agents.reasoning_agent import ask, live_model_available


def main() -> None:
    if not live_model_available():
        print("No OPENAI_API_KEY or ANTHROPIC_API_KEY set -- cannot run live. "
              "Export one and re-run: OPENAI_API_KEY=sk-... .venv-pa/bin/python -m agents.demo_live \"...\"")
        sys.exit(1)

    question = sys.argv[1] if len(sys.argv) > 1 else "What was this patient's most recent glucose reading?"
    gw, mcp_url, ids, cleanup = build_environment()
    try:
        patient_id = sys.argv[2] if len(sys.argv) > 2 else ids["grounded"][0]
        provider = "OpenAI" if os.getenv("OPENAI_API_KEY") else "Anthropic"
        print(f"Patient: {patient_id}")
        print(f"Provider: {provider}")
        print(f"Question (arbitrary, not scripted): {question!r}\n")
        finding, dropped = ask(mcp_url, f"The patient id is {patient_id}. {question}")
        print("Grounded finding (after citation + missing-data re-grounding):")
        print(f"  finding      = {finding.finding!r}")
        print(f"  evidence     = {[e.model_dump() for e in finding.evidence]}")
        print(f"  confidence   = {finding.confidence}")
        print(f"  missing_data = {finding.missing_data}")
        print(f"  red_flags    = {finding.red_flags}")
        print(f"  dropped      = {dropped}")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
