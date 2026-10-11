"""Integration test against real OpenSearch. Skipped unless PRIORAUTH_TEST_OPENSEARCH_URL is set, e.g.
  docker run -d -p 19200:9200 -e discovery.type=single-node -e DISABLE_SECURITY_PLUGIN=true -e DISABLE_INSTALL_DEMO_CONFIG=true opensearchproject/opensearch:2.17.1
  PRIORAUTH_TEST_OPENSEARCH_URL=http://localhost:19200 pytest tests/priorauth/test_opensearch.py"""
import os
import uuid

import pytest

from priorauth.agents.orchestrator import Orchestrator
from priorauth.rag import InMemoryVectorStore, OpenSearchVectorStore, PolicyRetriever, chunk_policies
from priorauth.schemas import PARRequest
from priorauth.synth.scenarios import load

URL = os.getenv("PRIORAUTH_TEST_OPENSEARCH_URL")
pytestmark = pytest.mark.skipif(not URL, reason="PRIORAUTH_TEST_OPENSEARCH_URL not set")


@pytest.fixture()
def store():
    s = OpenSearchVectorStore(URL, index=f"pa-test-{uuid.uuid4().hex[:8]}")
    s.upsert(chunk_policies())
    yield s
    s.client.indices.delete(index=s.index, ignore_unavailable=True)


def test_ranking_matches_in_memory_backend_on_all_eval_descriptions(store):
    mem = PolicyRetriever(InMemoryVectorStore()); os_ = PolicyRetriever(store)
    scn = [s for s in load() if s["expected"]["policy_id"]]
    agree = 0
    for s in scn:
        q = s["request"]["service_description"]
        a, b = mem.retrieve(q, None, 3), os_.retrieve(q, None, 3)
        agree += a[0]["policy_id"] == b[0]["policy_id"]
        assert abs(a[0]["score"] - b[0]["score"]) < 0.02          # cosine recovered from the Lucene score
    assert agree / len(scn) >= 0.98


def test_pipeline_runs_end_to_end_on_opensearch_retrieval(store, box, scenarios):
    box.retriever = PolicyRetriever(store)
    s = scenarios[4]; box.store.create_request(PARRequest.model_validate(s["request"]))
    Orchestrator(box).run(s["request"]["request_id"])
    assert box.store.get_task(f"TASK-{s['request']['request_id']}") is not None
