import os
import secrets

import pytest

os.environ.setdefault("PRIORAUTH_JWT_SECRET", secrets.token_hex(32))

from priorauth.fhir import InMemoryFhirSource, facts_to_fhir          # noqa: E402
from priorauth.mcp_server.tools import ToolBox                        # noqa: E402
from priorauth.rag import PolicyRetriever                             # noqa: E402
from priorauth.schemas import PARRequest                              # noqa: E402
from priorauth.store import Store                                     # noqa: E402
from priorauth.synth.scenarios import load                            # noqa: E402

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
