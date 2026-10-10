"""
Tamper-evident audit log (HIPAA Security Rule 164.312(b) audit controls,
164.312(c)(1) integrity).

Every access to a FHIR resource -- allowed, denied, or errored -- is recorded.
Entries are hash-chained: entry_hash = SHA256(prev_hash || canonical_json(entry)).
Editing or deleting any past line breaks every hash after it, which
`verify_chain` detects. A hash chain makes tampering *detectable*; it does not
make it impossible -- an attacker with write access to the whole file can
recompute the chain. For that, ship the head hash to an external append-only
store (see docs/HIPAA_MAPPING.md).

Never logged: request/response bodies, glucose values, query string values.
Only who / what resource type / which (pseudonymous) patient / outcome.

Sinks: a JSONL file (always) and, if PG_HOST is set, the append-only
`audit_log` table (timescaledb/init/007_healthcare.sql).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

GENESIS = "0" * 64


def _canonical(entry: dict) -> str:
    return json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash(prev: str, entry: dict) -> str:
    return hashlib.sha256((prev + _canonical(entry)).encode()).hexdigest()


class AuditLogger:
    def __init__(self, path: Optional[str] = None, pg_dsn: Optional[dict] = None):
        self.path = Path(path or os.getenv("GLUCOPULSE_AUDIT_PATH", "audit/audit.jsonl"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._pg_dsn = pg_dsn if pg_dsn is not None else self._pg_dsn_from_env()
        self._prev = self._load_head()

    @staticmethod
    def _pg_dsn_from_env() -> Optional[dict]:
        if not os.getenv("PG_HOST"):
            return None
        return dict(
            host=os.environ["PG_HOST"],
            port=int(os.getenv("PG_PORT", "5432")),
            dbname=os.getenv("POSTGRES_DB", "glucopulse"),
            user=os.getenv("GLUCOPULSE_FHIR_DB_USER") or os.getenv("POSTGRES_USER", "glucopulse"),
            password=os.getenv("GLUCOPULSE_FHIR_DB_PASSWORD") or os.getenv("POSTGRES_PASSWORD", ""),
        )

    def _load_head(self) -> str:
        if not self.path.exists():
            return GENESIS
        last = None
        with self.path.open("rb") as f:
            for line in f:
                if line.strip():
                    last = line
        return json.loads(last)["entry_hash"] if last else GENESIS

    def record(
        self, *, actor: str, client_id: Optional[str], action: str,
        resource_type: str, patient: Optional[str], outcome: str,
        purpose_of_use: str = "unspecified", method: str = "", path: str = "",
        status: int = 0, request_id: str = "", source_ip: str = "",
    ) -> dict:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "actor": actor, "client_id": client_id, "action": action,
            "resource_type": resource_type, "patient": patient, "outcome": outcome,
            "purpose_of_use": purpose_of_use, "method": method, "path": path,
            "status": status, "request_id": request_id, "source_ip": source_ip,
        }
        with self._lock:
            entry["prev_hash"] = self._prev
            entry["entry_hash"] = _hash(self._prev, {k: v for k, v in entry.items()
                                                      if k not in ("prev_hash", "entry_hash")})
            # fsync: an audit record that's only in the page cache isn't a record.
            with self.path.open("a", encoding="utf-8") as f:
                f.write(_canonical(entry) + "\n")
                f.flush()
                os.fsync(f.fileno())
            self._prev = entry["entry_hash"]
        self._write_pg(entry)
        return entry

    def _write_pg(self, entry: dict) -> None:
        if not self._pg_dsn:
            return
        try:
            import psycopg2
            with psycopg2.connect(**self._pg_dsn) as conn, conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO audit_log (ts, actor, client_id, action, resource_type,
                           patient, outcome, purpose_of_use, method, path, status,
                           request_id, source_ip, prev_hash, entry_hash)
                       VALUES (%(ts)s, %(actor)s, %(client_id)s, %(action)s, %(resource_type)s,
                           %(patient)s, %(outcome)s, %(purpose_of_use)s, %(method)s, %(path)s,
                           %(status)s, %(request_id)s, %(source_ip)s, %(prev_hash)s, %(entry_hash)s)""",
                    entry,
                )
        except Exception as exc:  # file sink already has the record; don't fail the request
            print(f"audit: postgres sink failed: {exc}", flush=True)

    def read(self, limit: int = 100, patient: Optional[str] = None) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    if patient is None or row.get("patient") == patient:
                        rows.append(row)
        return rows[-limit:]


def verify_chain(path: str) -> tuple[bool, Optional[int]]:
    """Returns (ok, first_bad_line_number). Line numbers are 1-based."""
    prev = GENESIS
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            body = {k: v for k, v in row.items() if k not in ("prev_hash", "entry_hash")}
            if row.get("prev_hash") != prev or row.get("entry_hash") != _hash(prev, body):
                return False, n
            prev = row["entry_hash"]
    return True, None


def to_fhir_audit_event(row: dict) -> dict:
    """Map an audit row to a FHIR R4 AuditEvent."""
    action_code = {"read": "R", "search": "E", "create": "C", "update": "U",
                   "delete": "D", "execute": "E"}.get(row["action"], "E")
    outcome_code = {"success": "0", "denied": "4", "error": "8"}.get(row["outcome"], "8")
    event = {
        "resourceType": "AuditEvent",
        "id": row["entry_hash"][:32],
        "type": {"system": "http://terminology.hl7.org/CodeSystem/audit-event-type",
                 "code": "rest", "display": "Restful Operation"},
        "action": action_code,
        "recorded": row["ts"].replace("+00:00", "Z"),
        "outcome": outcome_code,
        "agent": [{"who": {"identifier": {"value": row["actor"]}},
                   "requestor": True,
                   "network": {"address": row["source_ip"], "type": "2"}}],
        "source": {"observer": {"display": "GlucoPulse FHIR facade"}},
        "entity": [],
    }
    if row.get("patient"):
        event["entity"].append({"what": {"reference": f"Patient/{row['patient']}"},
                                "type": {"system": "http://terminology.hl7.org/CodeSystem/audit-entity-type",
                                         "code": "1", "display": "Person"}})
    if row.get("resource_type"):
        event["entity"].append({"what": {"identifier": {"value": row["resource_type"]}},
                                "type": {"system": "http://terminology.hl7.org/CodeSystem/audit-entity-type",
                                         "code": "2", "display": "System Object"}})
    return event


if __name__ == "__main__":
    import sys
    ok, bad = verify_chain(sys.argv[1] if len(sys.argv) > 1 else "audit/audit.jsonl")
    print("audit chain OK" if ok else f"audit chain BROKEN at line {bad}")
    sys.exit(0 if ok else 1)
