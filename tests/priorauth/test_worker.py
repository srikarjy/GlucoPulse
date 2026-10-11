import asyncio

from priorauth.agents.orchestrator import Orchestrator
from priorauth.schemas import PARRequest
from priorauth.store import jobs_t
from priorauth.worker import Worker, process_one
from tests.priorauth.conftest import by_profile


def test_retry_then_dead_letter(box):
    box.store.enqueue("bogus", {"request_id": "x"}, max_attempts=2)
    orch = Orchestrator(box)
    assert process_one(box.store, orch) == "queued"
    with box.store.engine.begin() as c:                      # skip the backoff delay
        c.execute(jobs_t.update().values(visible_at="2000-01-01T00:00:00+00:00"))
    assert process_one(box.store, orch) == "dead"
    assert box.store.job_counts() == {"dead": 1}


def test_expired_lease_is_reclaimed_after_worker_crash(box):
    box.store.enqueue("process_request", {"request_id": "r"})
    j = box.store.lease_job(lease_seconds=-1)               # leased then 'crashed': lease already expired
    assert j and box.store.lease_job() is not None          # another worker reclaims it
    assert box.store.job_counts() == {"leased": 1}


def test_leased_job_is_not_double_claimed(box):
    box.store.enqueue("process_request", {"request_id": "r"})
    assert box.store.lease_job(lease_seconds=300)
    assert box.store.lease_job() is None


def test_async_workers_drain_queue_concurrently(box, scenarios):
    for s in scenarios[:25]:
        box.store.create_request(PARRequest.model_validate(s["request"]))
    async def go():
        stop = asyncio.Event()
        w = Worker(box.store, Orchestrator(box), concurrency=4, poll=0.05)
        t = asyncio.create_task(w.run(stop))
        for _ in range(200):
            await asyncio.sleep(0.05)
            if box.store.job_counts().get("done") == 25:
                break
        stop.set(); await t
    asyncio.run(go())
    assert box.store.job_counts() == {"done": 25}
    assert len(box.store.list_tasks("open", limit=100)) == 25
    assert box.store.verify_audit_chain()[0]


def test_crash_between_draft_and_ack_does_not_duplicate_tasks(box, scenarios):
    s = by_profile("meets")[0]
    box.store.create_request(PARRequest.model_validate(s["request"]))
    orch = Orchestrator(box)
    orch.run(s["request"]["request_id"])                     # pipeline finished, job never acked
    assert process_one(box.store, orch) == "done"             # redelivery runs again
    assert len(box.store.list_tasks("open")) == 1


def test_file_backed_store_non_shared_connection_path(tmp_path, scenarios):
    """Regression: the non-StaticPool read path (what Postgres uses) must work end to end."""
    from priorauth.fhir import InMemoryFhirSource, facts_to_fhir
    from priorauth.mcp_server.tools import ToolBox
    from priorauth.rag import PolicyRetriever
    from priorauth.store import Store
    src = InMemoryFhirSource()
    for s in scenarios[:8]:
        src.put(s["request"]["patient_ref"], facts_to_fhir(s["request"]["patient_ref"], s["facts"]))
    store = Store(f"sqlite:///{tmp_path/'pa.db'}")
    assert store._shared_conn is False
    box = ToolBox(store, src, PolicyRetriever())
    for s in scenarios[:8]:
        store.create_request(PARRequest.model_validate(s["request"]))
    orch = Orchestrator(box)
    assert [process_one(store, orch) for _ in range(8)] == ["done"] * 8
    assert len(store.list_tasks("open")) == 8 and store.verify_audit_chain()[0]


def test_dead_lettered_request_is_marked_failed_and_replayable(box, scenarios):
    s = scenarios[0]
    box.store.create_request(PARRequest.model_validate(s["request"]))
    class Broken:                                    # even fail-closed cannot save this one: orchestrator itself explodes
        def run(self, rid): raise RuntimeError("db down")
    with box.store.engine.begin() as c:
        c.execute(jobs_t.update().values(max_attempts=1))
    assert process_one(box.store, Broken()) == "dead"
    rid = s["request"]["request_id"]
    assert box.store.get_request(rid)[1] == "failed"
    assert any(e["action"] == "request.failed" for e in box.store.audit_entries())
    assert box.store.requeue_dead() == 1
    assert box.store.get_request(rid)[1] == "received"
    assert process_one(box.store, Orchestrator(box)) == "done"
    assert box.store.get_task(f"TASK-{rid}") is not None
