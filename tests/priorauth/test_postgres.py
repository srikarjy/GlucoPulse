"""
Integration tests against real Postgres + pgvector. Skipped unless PRIORAUTH_TEST_PG_URL is set, e.g.
  docker run -d -p 55432:5432 -e POSTGRES_PASSWORD=pa -e POSTGRES_DB=priorauth pgvector/pgvector:pg16
  PRIORAUTH_TEST_PG_URL=postgresql+psycopg://postgres:pa@localhost:55432/priorauth_test pytest tests/priorauth/test_postgres.py
These tests DROP SCHEMA public: the database name must end in `_test` or the module refuses to run.
"""
import os
import threading

import pytest
import sqlalchemy as sa

from priorauth.agents.orchestrator import Orchestrator
from priorauth.fhir import InMemoryFhirSource, facts_to_fhir
from priorauth.mcp_server.tools import ToolBox
from priorauth.rag import PgVectorStore, PolicyRetriever
from priorauth.schemas import PARRequest
from priorauth.store import Store
from priorauth.worker import process_one

URL = os.getenv("PRIORAUTH_TEST_PG_URL")
pytestmark = pytest.mark.skipif(not URL, reason="PRIORAUTH_TEST_PG_URL not set")
if URL and not URL.rsplit("/", 1)[-1].split("?")[0].endswith("_test"):
    raise RuntimeError("refusing to run destructive tests: PRIORAUTH_TEST_PG_URL database name must end with '_test'")


@pytest.fixture()
def pg(scenarios):
    eng = sa.create_engine(URL)
    with eng.begin() as c:
        c.execute(sa.text("DROP SCHEMA public CASCADE; CREATE SCHEMA public;"))
    store = Store(URL)
    src = InMemoryFhirSource()
    for s in scenarios[:40]:
        src.put(s["request"]["patient_ref"], facts_to_fhir(s["request"]["patient_ref"], s["facts"]))
    return store, src


def test_pipeline_on_postgres_with_pgvector_retrieval(pg, scenarios):
    store, src = pg
    box = ToolBox(store, src, PolicyRetriever(PgVectorStore(URL)))
    for s in scenarios[:10]:
        store.create_request(PARRequest.model_validate(s["request"]))
    orch = Orchestrator(box)
    assert [process_one(store, orch) for _ in range(10)] == ["done"] * 10
    assert len(store.list_tasks("open")) == 10
    assert store.verify_audit_chain() == (True, None)


def test_pgvector_semantic_retrieval_uses_hnsw_index_and_ranks_correctly():
    store = PgVectorStore(URL)
    from priorauth.rag import chunk_policies
    store.upsert(chunk_policies())
    r = PolicyRetriever(store).retrieve("magnetic resonance imaging of the lumbar spine for persistent low back pain", None, 3)
    assert r[0]["policy_id"] == "SYN-MRI-LSPINE"
    with sa.create_engine(URL).connect() as c:
        idx = [row[0] for row in c.execute(sa.text("SELECT indexdef FROM pg_indexes WHERE tablename='policy_chunks'"))]
    assert any("hnsw" in i and "vector_cosine_ops" in i for i in idx)


def test_concurrent_workers_never_double_claim_with_skip_locked(pg):
    store, _ = pg
    for i in range(60):
        store.enqueue("noop", {"i": i})
    claimed, lock = [], threading.Lock()

    def grab():
        mine = Store(URL)          # own engine + own in-process lock: only Postgres can arbitrate between these
        while (j := mine.lease_job(300)):
            with lock:
                claimed.append(j["job_id"])
    ts = [threading.Thread(target=grab) for _ in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert len(claimed) == 60 and len(set(claimed)) == 60


def test_concurrent_audit_appends_keep_one_unbroken_chain(pg):
    store, _ = pg
    def w(n):
        mine = Store(URL)          # separate engine/lock per writer, like separate worker processes
        [mine.audit(f"w{n}", "x", f"s{k}") for k in range(25)]
    ts = [threading.Thread(target=w, args=(n,)) for n in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert store.verify_audit_chain() == (True, None)
    assert len(store.audit_entries(limit=1000)) == 200


def test_simultaneous_startup_does_not_race_on_schema_creation():
    """Regression: api/worker/mcp containers boot together; unguarded create_all crashed two of three."""
    with sa.create_engine(URL).begin() as c:
        c.execute(sa.text("DROP SCHEMA public CASCADE; CREATE SCHEMA public;"))
    errs = []
    def boot():
        try:
            Store(URL)
        except Exception as e:    # noqa: BLE001
            errs.append(e)
    ts = [threading.Thread(target=boot) for _ in range(6)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert not errs, errs


def test_simultaneous_pgvector_init_does_not_race_on_extension_and_index():
    """Regression: worker + mcp containers both ran CREATE EXTENSION concurrently; one crashed on pg_extension_name_index."""
    with sa.create_engine(URL).begin() as c:
        c.execute(sa.text("DROP SCHEMA public CASCADE; CREATE SCHEMA public; DROP EXTENSION IF EXISTS vector CASCADE"))
    errs = []
    def boot():
        try:
            PolicyRetriever(PgVectorStore(URL))
        except Exception as e:    # noqa: BLE001
            errs.append(e)
    ts = [threading.Thread(target=boot) for _ in range(6)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert not errs, errs
