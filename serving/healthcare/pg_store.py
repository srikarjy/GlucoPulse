"""
TimescaleDB-backed ReadingStore for the FHIR facade.

Source ids in cgm_readings are the dataset's ('1', '2', ...). They never leave
this class: callers see only HMAC pseudonyms (deid.pseudonymize). The
pseudonym -> source id map is rebuilt from `SELECT DISTINCT patient_id`, so
nothing identifying is stored beside the data.

Connects as the read-only role created in 007_healthcare.sql
(GLUCOPULSE_FHIR_DB_USER) when set -- the facade has no reason to hold write
credentials on clinical data (minimum necessary, 164.502(b)).
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import psycopg2

from .deid import pseudonymize

Readings = list[tuple[datetime, float]]


class PostgresReadingStore:
    def __init__(self, dsn: dict):
        self._dsn = dsn
        self._map: dict[str, str] = {}
        self._refresh()

    @classmethod
    def from_env(cls) -> "PostgresReadingStore":
        return cls(dict(
            host=os.environ["PG_HOST"],
            port=int(os.getenv("PG_PORT", "5432")),
            dbname=os.getenv("POSTGRES_DB", "glucopulse"),
            user=os.getenv("GLUCOPULSE_FHIR_DB_USER") or os.getenv("POSTGRES_USER", "glucopulse"),
            password=os.getenv("GLUCOPULSE_FHIR_DB_PASSWORD") or os.getenv("POSTGRES_PASSWORD", ""),
        ))

    def _conn(self):
        return psycopg2.connect(**self._dsn)

    def _refresh(self) -> None:
        with self._conn() as c, c.cursor() as cur:
            cur.execute("SELECT DISTINCT patient_id FROM cgm_readings")
            self._map = {pseudonymize(r[0]): r[0] for r in cur.fetchall()}

    def has_patient(self, pseudonym: str) -> bool:
        return pseudonym in self._map

    def readings(self, pseudonym: str, start: Optional[datetime], end: Optional[datetime],
                 limit: int, offset: int) -> tuple[Readings, int]:
        raw = self._map[pseudonym]
        where, args = ["patient_id = %s"], [raw]
        if start:
            where.append("time >= %s"); args.append(start)
        if end:
            where.append("time <= %s"); args.append(end)
        w = " AND ".join(where)
        with self._conn() as c, c.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM cgm_readings WHERE {w}", args)
            total = cur.fetchone()[0]
            cur.execute(f"SELECT time, glucose_value FROM cgm_readings WHERE {w} "
                        "ORDER BY time LIMIT %s OFFSET %s", [*args, limit, offset])
            return [(t, float(v)) for t, v in cur.fetchall()], total

    def latest(self, pseudonym: str, n: int) -> Readings:
        with self._conn() as c, c.cursor() as cur:
            cur.execute("SELECT time, glucose_value FROM cgm_readings WHERE patient_id = %s "
                        "ORDER BY time DESC LIMIT %s", [self._map[pseudonym], n])
            return [(t, float(v)) for t, v in reversed(cur.fetchall())]
