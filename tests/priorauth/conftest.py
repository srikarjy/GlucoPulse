import os
import secrets
import socket
import threading
import time

import pytest
import uvicorn

os.environ.setdefault("PRIORAUTH_JWT_SECRET", secrets.token_hex(32))

from priorauth.fhir import InMemoryFhirSource, facts_to_fhir
from priorauth.mcp_server.server import build_app
from priorauth.mcp_server.tools import ToolBox
from priorauth.rag import PolicyRetriever
from priorauth.schemas import PARRequest
from priorauth.store import Store
from priorauth.synth.scenarios import load

_SC = load()


@pytest.fixture(scope="session")
def scenarios():
    return _SC


def by_profile(profile):
    return [s for s in _SC if s["profile"] == profile]


def clean_meets():
    """A 'meets' scenario whose evidence uses policy vocabulary (the deterministic backend should approve-draft it)."""
    return next(s for s in _SC if s["profile"] == "meets" and not s["paraphrased_evidence"])


@pytest.fixture()
def box():
    src = InMemoryFhirSource()
    for s in _SC:
        src.put(s["request"]["patient_ref"], facts_to_fhir(s["request"]["patient_ref"], s["facts"]))
    return ToolBox(Store(), src, PolicyRetriever())


@pytest.fixture()
def submit(box):
    def _s(scn):
        req = PARRequest.model_validate(scn["request"])
        box.store.create_request(req)
        return req.request_id
    return _s


@pytest.fixture()
def server(box):
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    cfg = uvicorn.Config(build_app(box), host="127.0.0.1", port=port, log_level="error")
    srv = uvicorn.Server(cfg)
    t = threading.Thread(target=srv.run, daemon=True); t.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/mcp", box
    srv.should_exit = True; t.join(5)
