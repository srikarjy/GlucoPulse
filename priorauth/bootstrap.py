"""Wiring: build Store / FHIR source / retriever / ToolBox from environment."""
from __future__ import annotations

import os

from priorauth.fhir import FhirSource, HttpFhirSource, InMemoryFhirSource, facts_to_fhir
from priorauth.mcp_server.tools import ToolBox
from priorauth.rag import PolicyRetriever
from priorauth.store import Store


def make_store() -> Store:
    return Store(os.getenv("PRIORAUTH_DATABASE_URL", "sqlite:///priorauth.db"))


def make_fhir() -> FhirSource:
    if os.getenv("PRIORAUTH_FHIR_BASE_URL"):
        return HttpFhirSource(os.environ["PRIORAUTH_FHIR_BASE_URL"], os.getenv("PRIORAUTH_FHIR_TOKEN"))
    src = InMemoryFhirSource()          # synthetic demo data only
    from priorauth.synth.scenarios import load
    for s in load():
        pid = s["request"]["patient_ref"]
        src.put(pid, facts_to_fhir(pid, s["facts"]))
    return src


def make_retriever() -> PolicyRetriever:
    dsn = os.getenv("PRIORAUTH_PGVECTOR_URL")
    if dsn:
        from priorauth.rag import PgVectorStore
        return PolicyRetriever(PgVectorStore(dsn))
    return PolicyRetriever()


def make_box(store: Store | None = None) -> ToolBox:
    return ToolBox(store or make_store(), make_fhir(), make_retriever())


def make_orchestrator(box: ToolBox):
    """PRIORAUTH_BACKEND=deterministic (default; no LLM) | bedrock (Strands agents over MCP at PRIORAUTH_MCP_URL)."""
    from priorauth.agents.orchestrator import Orchestrator
    if os.getenv("PRIORAUTH_BACKEND", "deterministic") == "bedrock":
        from priorauth.agents.strands_backend import build_strands_agents
        return Orchestrator(box, agents=build_strands_agents(os.environ["PRIORAUTH_MCP_URL"], box.store))
    return Orchestrator(box)
