"""
Async worker: leases jobs from the DB queue, runs the agent pipeline in a thread pool
(agents/tools are sync), acks on success, retries with backoff, dead-letters after
max attempts. Lease expiry makes worker crashes safe: another worker reclaims the job,
and tool/task idempotency makes the re-run a no-op for completed steps.
"""
from __future__ import annotations

import asyncio
import logging
import signal

from priorauth.agents.orchestrator import Orchestrator
from priorauth.schemas import RequestStatus
from priorauth.store import Store

log = logging.getLogger("priorauth.worker")


def process_one(store: Store, orch: Orchestrator) -> str | None:
    """Lease and run a single job synchronously. Returns outcome or None if idle."""
    job = store.lease_job()
    if not job:
        return None
    try:
        if job["kind"] != "process_request":
            raise ValueError(f"unknown job kind {job['kind']}")
        rid = job["payload"]["request_id"]
        store.set_status(rid, RequestStatus.PROCESSING)
        orch.run(rid)
        store.complete_job(job["job_id"])
        return "done"
    except Exception as e:   # noqa: BLE001
        log.exception("job %s failed", job["job_id"])
        return store.fail_job(job["job_id"], f"{type(e).__name__}: {e}")


class Worker:
    def __init__(self, store: Store, orch: Orchestrator, concurrency: int = 4, poll: float = 0.5):
        self.store, self.orch, self.concurrency, self.poll = store, orch, concurrency, poll

    async def _loop(self, stop: asyncio.Event, n: int) -> None:
        while not stop.is_set():
            outcome = await asyncio.to_thread(process_one, self.store, self.orch)
            if outcome is None:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=self.poll)
                except TimeoutError:
                    pass

    async def run(self, stop: asyncio.Event | None = None) -> None:
        stop = stop or asyncio.Event()
        try:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):  # pragma: no cover
            pass
        await asyncio.gather(*[self._loop(stop, i) for i in range(self.concurrency)])


def main() -> None:  # pragma: no cover
    import os
    from prometheus_client import start_http_server
    from priorauth.bootstrap import make_box, make_orchestrator
    from priorauth.observability import REGISTRY
    from priorauth.telemetry_setup import setup_tracing
    setup_tracing("priorauth-worker")
    start_http_server(int(os.getenv("PRIORAUTH_WORKER_METRICS_PORT", "9102")), registry=REGISTRY)
    box = make_box()
    asyncio.run(Worker(box.store, make_orchestrator(box), concurrency=int(os.getenv("PRIORAUTH_WORKER_CONCURRENCY", "4"))).run())


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    main()
