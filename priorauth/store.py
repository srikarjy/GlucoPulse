"""
Persistence: requests, drafts, review tasks, human decisions, tool idempotency,
a leased job queue, and a hash-chained audit trail.

Runs on SQLite (tests, local) and Postgres (compose/AWS). Postgres gets
`FOR UPDATE SKIP LOCKED` for queue claims and an advisory lock around audit
appends so concurrent writers cannot fork the hash chain.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import threading
import uuid
from datetime import datetime, timedelta, UTC
from typing import Any, Optional

import sqlalchemy as sa

from priorauth.schemas import (DraftRecommendation, PARRequest, RequestStatus,
                               ReviewDecision, ReviewTask)

GENESIS = "0" * 64
md = sa.MetaData()

requests_t = sa.Table("par_requests", md,
    sa.Column("request_id", sa.String, primary_key=True), sa.Column("payload", sa.JSON),
    sa.Column("status", sa.String, index=True), sa.Column("client_key", sa.String, unique=True, nullable=True),
    sa.Column("created_at", sa.String), sa.Column("updated_at", sa.String))
drafts_t = sa.Table("par_drafts", md,
    sa.Column("request_id", sa.String, primary_key=True), sa.Column("payload", sa.JSON), sa.Column("created_at", sa.String))
staged_t = sa.Table("staged_drafts", md,
    sa.Column("request_id", sa.String, primary_key=True), sa.Column("payload", sa.JSON), sa.Column("created_at", sa.String))
evidence_t = sa.Table("evidence_reports", md,
    sa.Column("request_id", sa.String, primary_key=True), sa.Column("payload", sa.JSON), sa.Column("created_at", sa.String))
tasks_t = sa.Table("review_tasks", md,
    sa.Column("task_id", sa.String, primary_key=True), sa.Column("request_id", sa.String, unique=True),
    sa.Column("tier", sa.String), sa.Column("priority", sa.Integer), sa.Column("state", sa.String, index=True),
    sa.Column("claimed_by", sa.String, nullable=True), sa.Column("payload", sa.JSON), sa.Column("created_at", sa.String))
decisions_t = sa.Table("review_decisions", md,
    sa.Column("task_id", sa.String, primary_key=True), sa.Column("reviewer", sa.String), sa.Column("decision", sa.String),
    sa.Column("comment", sa.String), sa.Column("agreed", sa.Boolean), sa.Column("created_at", sa.String))
idem_t = sa.Table("tool_idempotency", md,
    sa.Column("principal", sa.String, primary_key=True), sa.Column("tool", sa.String, primary_key=True),
    sa.Column("key", sa.String, primary_key=True), sa.Column("args_hash", sa.String),
    sa.Column("result", sa.JSON), sa.Column("created_at", sa.String))
jobs_t = sa.Table("jobs", md,
    sa.Column("job_id", sa.String, primary_key=True), sa.Column("kind", sa.String), sa.Column("payload", sa.JSON),
    sa.Column("state", sa.String, index=True), sa.Column("attempts", sa.Integer, default=0),
    sa.Column("max_attempts", sa.Integer, default=3), sa.Column("visible_at", sa.String),
    sa.Column("leased_until", sa.String, nullable=True), sa.Column("last_error", sa.String, nullable=True))
audit_t = sa.Table("audit_log", md,
    sa.Column("seq", sa.Integer, primary_key=True, autoincrement=True), sa.Column("entry", sa.JSON),
    sa.Column("subject", sa.String, index=True, nullable=True),
    sa.Column("prev_hash", sa.String), sa.Column("entry_hash", sa.String))


def now() -> datetime:
    return datetime.now(UTC)


def iso(d: datetime) -> str:
    return d.isoformat()


def _canon(o: Any) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def args_hash(args: dict) -> str:
    return hashlib.sha256(_canon(args).encode()).hexdigest()


class IdempotencyConflict(Exception):
    """Same idempotency key reused with different arguments."""


class Store:
    def __init__(self, url: str = "sqlite:///:memory:"):
        kw: dict[str, Any] = {}
        if url.startswith("sqlite"):
            kw = {"connect_args": {"check_same_thread": False}}
            if ":memory:" in url:
                kw["poolclass"] = sa.pool.StaticPool
        self.engine = sa.create_engine(url, future=True, **kw)
        self.is_pg = self.engine.dialect.name == "postgresql"
        self._lock = threading.RLock()   # serialises SQLite writers / audit appends in-process
        # In-memory SQLite shares ONE DBAPI connection across threads, so reads must be serialised too
        # or they interleave with another thread's transaction. Real pools (Postgres, file SQLite) don't need it.
        self._shared_conn = url.startswith("sqlite") and ":memory:" in url
        self._init_schema()

    def _init_schema(self) -> None:
        """Idempotent and safe when api/worker/mcp boot simultaneously (advisory lock serialises DDL).
        Production should run versioned migrations (Alembic) as a separate deploy step instead."""
        with self.engine.begin() as c:
            if self.is_pg:
                c.execute(sa.text("SELECT pg_advisory_xact_lock(727002)"))
            md.create_all(c)

    @contextmanager
    def _read(self):
        if self._shared_conn:
            with self._lock, self.engine.connect() as c:
                yield c
        else:
            with self.engine.connect() as c:
                yield c

    # ---------------- audit ----------------
    def audit(self, actor: str, action: str, subject: str, detail: Optional[dict] = None, conn=None) -> dict:
        """Append to the hash chain. `detail` must never contain clinical text."""
        def _do(c):
            if self.is_pg:
                c.execute(sa.text("SELECT pg_advisory_xact_lock(727001)"))
            row = c.execute(sa.select(audit_t.c.entry_hash).order_by(audit_t.c.seq.desc()).limit(1)).first()
            prev = row[0] if row else GENESIS
            entry = {"ts": iso(now()), "actor": actor, "action": action, "subject": subject, "detail": detail or {}}
            h = hashlib.sha256((prev + _canon(entry)).encode()).hexdigest()
            c.execute(audit_t.insert().values(entry=entry, subject=subject, prev_hash=prev, entry_hash=h))
            return {**entry, "entry_hash": h}
        if conn is not None:
            return _do(conn)
        with self._lock, self.engine.begin() as c:
            return _do(c)

    def verify_audit_chain(self) -> tuple[bool, Optional[int]]:
        prev = GENESIS
        with self._read() as c:
            for seq, entry, ph, eh in c.execute(sa.select(audit_t.c.seq, audit_t.c.entry, audit_t.c.prev_hash,
                                                          audit_t.c.entry_hash).order_by(audit_t.c.seq)):
                if ph != prev or hashlib.sha256((prev + _canon(entry)).encode()).hexdigest() != eh:
                    return False, seq
                prev = eh
        return True, None

    def audit_entries(self, subject: Optional[str] = None, limit: int = 200, ascending: bool = False) -> list[dict]:
        q = sa.select(audit_t.c.seq, audit_t.c.entry, audit_t.c.entry_hash)
        if subject is not None:
            q = q.where(audit_t.c.subject == subject)       # filtered in SQL: a busy log must not hide a request's history
        q = q.order_by(audit_t.c.seq if ascending else audit_t.c.seq.desc()).limit(limit)
        with self._read() as c:
            return [{"seq": s, **e, "entry_hash": h} for s, e, h in c.execute(q)]

    # ---------------- requests ----------------
    def create_request(self, req: PARRequest, client_key: Optional[str] = None) -> tuple[PARRequest, bool]:
        """Idempotent on client_key. Returns (request, created)."""
        with self._lock, self.engine.begin() as c:
            if client_key:
                ex = c.execute(sa.select(requests_t.c.payload).where(requests_t.c.client_key == client_key)).first()
                if ex:
                    return PARRequest.model_validate(ex[0]), False
            t = iso(now())
            c.execute(requests_t.insert().values(request_id=req.request_id, payload=json.loads(req.model_dump_json()),
                                                 status=RequestStatus.RECEIVED.value, client_key=client_key,
                                                 created_at=t, updated_at=t))
            self.audit("system", "request.received", req.request_id, {"service_code": req.service_code}, conn=c)
            self._enqueue(c, "process_request", {"request_id": req.request_id})
        return req, True

    def get_request(self, request_id: str) -> Optional[tuple[PARRequest, str]]:
        with self._read() as c:
            r = c.execute(sa.select(requests_t.c.payload, requests_t.c.status).where(requests_t.c.request_id == request_id)).first()
        return (PARRequest.model_validate(r[0]), r[1]) if r else None

    def set_status(self, request_id: str, status: RequestStatus, conn=None) -> None:
        stmt = requests_t.update().where(requests_t.c.request_id == request_id).values(status=status.value, updated_at=iso(now()))
        if conn is not None:
            conn.execute(stmt)
        else:
            with self._lock, self.engine.begin() as c:
                c.execute(stmt)

    # ---------------- drafts + review tasks (atomic) ----------------
    def save_draft_and_task(self, draft: DraftRecommendation, task: ReviewTask) -> ReviewTask:
        """Draft, review task, status flip and audit commit together or not at all.
        Re-running for the same request returns the existing task (worker retries are safe)."""
        with self._lock, self.engine.begin() as c:
            ex = c.execute(sa.select(tasks_t.c.payload).where(tasks_t.c.request_id == draft.request_id)).first()
            if ex:
                return ReviewTask.model_validate(ex[0])
            t = iso(now())
            c.execute(drafts_t.insert().values(request_id=draft.request_id, payload=json.loads(draft.model_dump_json()), created_at=t))
            c.execute(tasks_t.insert().values(task_id=task.task_id, request_id=task.request_id, tier=task.tier.value,
                                              priority=task.priority, state="open", claimed_by=None,
                                              payload=json.loads(task.model_dump_json()), created_at=t))
            self.set_status(draft.request_id, RequestStatus.AWAITING_REVIEW, conn=c)
            self.audit("agent-pipeline", "draft.created", draft.request_id,
                       {"recommendation": draft.recommendation.value, "confidence": draft.confidence,
                        "flags": draft.flags, "tier": task.tier.value}, conn=c)
        return task

    def stage_draft(self, draft: DraftRecommendation) -> None:
        """Drafting agent output, not yet visible to reviewers. Overwrites (draft revisions)."""
        with self._lock, self.engine.begin() as c:
            c.execute(staged_t.delete().where(staged_t.c.request_id == draft.request_id))
            c.execute(staged_t.insert().values(request_id=draft.request_id,
                                               payload=json.loads(draft.model_dump_json()), created_at=iso(now())))

    def save_evidence(self, request_id: str, results: list[dict]) -> None:
        """Evidence-stage findings recorded server-side, so the drafting stage never has to re-type evidence."""
        with self._lock, self.engine.begin() as c:
            c.execute(evidence_t.delete().where(evidence_t.c.request_id == request_id))
            c.execute(evidence_t.insert().values(request_id=request_id, payload=results, created_at=iso(now())))

    def get_evidence(self, request_id: str) -> list[dict]:
        with self._read() as c:
            r = c.execute(sa.select(evidence_t.c.payload).where(evidence_t.c.request_id == request_id)).first()
        return r[0] if r else []

    def get_staged(self, request_id: str) -> Optional[DraftRecommendation]:
        with self._read() as c:
            r = c.execute(sa.select(staged_t.c.payload).where(staged_t.c.request_id == request_id)).first()
        return DraftRecommendation.model_validate(r[0]) if r else None

    def get_draft(self, request_id: str) -> Optional[DraftRecommendation]:
        with self._read() as c:
            r = c.execute(sa.select(drafts_t.c.payload).where(drafts_t.c.request_id == request_id)).first()
        return DraftRecommendation.model_validate(r[0]) if r else None

    def list_tasks(self, state: str = "open", tier: Optional[str] = None, limit: int = 50) -> list[ReviewTask]:
        q = sa.select(tasks_t.c.payload, tasks_t.c.state, tasks_t.c.claimed_by).where(tasks_t.c.state == state)
        if tier:
            q = q.where(tasks_t.c.tier == tier)
        q = q.order_by(tasks_t.c.priority, tasks_t.c.created_at).limit(limit)
        with self._read() as c:
            out = []
            for p, s, cb in c.execute(q):
                t = ReviewTask.model_validate(p); t.state, t.claimed_by = s, cb; out.append(t)
        return out

    def get_task(self, task_id: str) -> Optional[ReviewTask]:
        with self._read() as c:
            r = c.execute(sa.select(tasks_t.c.payload, tasks_t.c.state, tasks_t.c.claimed_by).where(tasks_t.c.task_id == task_id)).first()
        if not r:
            return None
        t = ReviewTask.model_validate(r[0]); t.state, t.claimed_by = r[1], r[2]
        return t

    def claim_task(self, task_id: str, reviewer: str) -> bool:
        with self._lock, self.engine.begin() as c:
            res = c.execute(tasks_t.update().where(sa.and_(tasks_t.c.task_id == task_id, tasks_t.c.state == "open"))
                            .values(state="claimed", claimed_by=reviewer))
            if res.rowcount:
                self.audit(reviewer, "review.claimed", task_id, conn=c)
            return bool(res.rowcount)

    def record_decision(self, d: ReviewDecision) -> bool:
        """The ONLY place a final decision is written. Caller (API) has verified a human reviewer principal."""
        with self._lock, self.engine.begin() as c:
            row = c.execute(sa.select(tasks_t.c.request_id, tasks_t.c.state, tasks_t.c.claimed_by)
                            .where(tasks_t.c.task_id == d.task_id)).first()
            if not row or row[1] != "claimed" or row[2] != d.reviewer:
                return False
            c.execute(decisions_t.insert().values(task_id=d.task_id, reviewer=d.reviewer, decision=d.decision.value,
                                                  comment=d.comment, agreed=d.agreed_with_draft, created_at=iso(now())))
            c.execute(tasks_t.update().where(tasks_t.c.task_id == d.task_id).values(state="closed"))
            self.set_status(row[0], RequestStatus.DECIDED, conn=c)
            self.audit(d.reviewer, "review.decided", row[0],
                       {"decision": d.decision.value, "agreed_with_draft": d.agreed_with_draft}, conn=c)
        return True

    def get_decision(self, task_id: str) -> Optional[dict]:
        with self._read() as c:
            r = c.execute(sa.select(decisions_t).where(decisions_t.c.task_id == task_id)).mappings().first()
        return dict(r) if r else None

    # ---------------- tool idempotency ----------------
    def idem_lookup(self, principal: str, tool: str, key: str, ahash: str) -> Optional[dict]:
        with self._read() as c:
            r = c.execute(sa.select(idem_t.c.args_hash, idem_t.c.result).where(
                sa.and_(idem_t.c.principal == principal, idem_t.c.tool == tool, idem_t.c.key == key))).first()
        if not r:
            return None
        if r[0] != ahash:
            raise IdempotencyConflict(f"idempotency key {key!r} reused with different arguments")
        return r[1]

    def idem_store(self, principal: str, tool: str, key: str, ahash: str, result: dict) -> None:
        with self._lock, self.engine.begin() as c:
            c.execute(idem_t.insert().values(principal=principal, tool=tool, key=key, args_hash=ahash,
                                             result=result, created_at=iso(now())))

    # ---------------- queue ----------------
    def _enqueue(self, c, kind: str, payload: dict, max_attempts: int = 3) -> str:
        jid = uuid.uuid4().hex
        c.execute(jobs_t.insert().values(job_id=jid, kind=kind, payload=payload, state="queued", attempts=0,
                                         max_attempts=max_attempts, visible_at=iso(now()), leased_until=None))
        return jid

    def enqueue(self, kind: str, payload: dict, max_attempts: int = 3) -> str:
        with self._lock, self.engine.begin() as c:
            return self._enqueue(c, kind, payload, max_attempts)

    def lease_job(self, lease_seconds: int = 120) -> Optional[dict]:
        """Claim one runnable job. Expired leases are reclaimable (worker crash recovery)."""
        t = iso(now())
        with self._lock, self.engine.begin() as c:
            q = sa.select(jobs_t).where(sa.or_(
                sa.and_(jobs_t.c.state == "queued", jobs_t.c.visible_at <= t),
                sa.and_(jobs_t.c.state == "leased", jobs_t.c.leased_until < t))).order_by(jobs_t.c.visible_at).limit(1)
            if self.is_pg:
                q = q.with_for_update(skip_locked=True)
            row = c.execute(q).mappings().first()
            if not row:
                return None
            attempts = row["attempts"] + 1
            if attempts > row["max_attempts"]:
                c.execute(jobs_t.update().where(jobs_t.c.job_id == row["job_id"]).values(state="dead", attempts=attempts))
                self.audit("queue", "job.dead", row["job_id"], {"kind": row["kind"]}, conn=c)
                return None
            c.execute(jobs_t.update().where(jobs_t.c.job_id == row["job_id"]).values(
                state="leased", attempts=attempts, leased_until=iso(now() + timedelta(seconds=lease_seconds))))
            return {**dict(row), "attempts": attempts}

    def complete_job(self, job_id: str) -> None:
        with self._lock, self.engine.begin() as c:
            c.execute(jobs_t.update().where(jobs_t.c.job_id == job_id).values(state="done", leased_until=None))

    def fail_job(self, job_id: str, error: str, backoff_seconds: int = 5) -> str:
        """Requeue with backoff, or dead-letter once attempts are exhausted. Returns new state."""
        with self._lock, self.engine.begin() as c:
            row = c.execute(sa.select(jobs_t.c.attempts, jobs_t.c.max_attempts, jobs_t.c.kind).where(jobs_t.c.job_id == job_id)).first()
            if row[0] >= row[1]:
                c.execute(jobs_t.update().where(jobs_t.c.job_id == job_id).values(state="dead", last_error=error[:500], leased_until=None))
                self.audit("queue", "job.dead", job_id, {"kind": row[2]}, conn=c)
                return "dead"
            c.execute(jobs_t.update().where(jobs_t.c.job_id == job_id).values(
                state="queued", last_error=error[:500], leased_until=None,
                visible_at=iso(now() + timedelta(seconds=backoff_seconds * row[0]))))
            return "queued"

    def requeue_dead(self, kind: Optional[str] = None) -> int:
        """Operator action: give dead-lettered jobs a fresh attempt budget. Requests marked FAILED return to RECEIVED."""
        with self._lock, self.engine.begin() as c:
            q = sa.select(jobs_t.c.job_id, jobs_t.c.payload).where(jobs_t.c.state == "dead")
            if kind:
                q = q.where(jobs_t.c.kind == kind)
            rows = c.execute(q).all()
            for jid, payload in rows:
                c.execute(jobs_t.update().where(jobs_t.c.job_id == jid).values(state="queued", attempts=0, visible_at=iso(now()), leased_until=None))
                if payload.get("request_id"):
                    self.set_status(payload["request_id"], RequestStatus.RECEIVED, conn=c)
                self.audit("operator", "job.requeued", jid, conn=c)
            return len(rows)

    def job_counts(self) -> dict[str, int]:
        with self._read() as c:
            return {s: n for s, n in c.execute(sa.select(jobs_t.c.state, sa.func.count()).group_by(jobs_t.c.state))}
