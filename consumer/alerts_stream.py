"""Streaming glucose alerts: consumes cgm-raw, decides alerts with AlertStream, stores them and publishes cgm-alerts.

Run: python -m consumer.alerts_stream (from the repo root, or the `alerts` compose service).

Guarantees (docs/STREAMING.md):
  * No duplicate alerts. glucose_alerts has primary key (patient_id, time, tier); a reading processed twice (restart,
    redelivery) gives the same key and the insert is a no-op.
  * No lost alerts. Outbox: the row is saved first with published = false, sent to cgm-alerts, and marked published
    only in Kafka's delivery callback. A row that is still unpublished (crash after the save, before delivery) is sent
    again when the same reading is replayed, and by a sweep at startup. Delivery is at-least-once, so each message
    carries a fixed alert_id (patient|time|tier) for downstream de-duplication.
  * Memory survives restarts. Every --checkpoint-every messages / --checkpoint-seconds, the AlertStream memory of the
    people in each partition and the next offset to read are written to alert_checkpoints in ONE transaction (after
    flushing pending deliveries). On partition assignment the consumer restores that memory and seeks to that offset,
    so only messages after the last checkpoint are processed again. Kafka's own committed offsets are also updated,
    for lag monitoring only; the checkpoint table is the source of truth.

Safety: alerts are validated only in adults NOT using insulin. Only patient ids with an allowed prefix (default
"cgmacros-") are evaluated; others (e.g. AZT1D, type 1 on insulin, in the same topic) are counted and skipped.
Invalid or implausible readings are skipped here; ingest.py routes them to their error topics.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Callable, Protocol

from consumer.alert_state import AlertStream
from serving.alerts import DISCLAIMER

PLAUSIBLE = (40.0, 400.0)


def classify_reading(raw: bytes) -> tuple[tuple[str, datetime, float] | None, str]:
    """((patient_id, naive UTC time, mg/dL), "ok") or (None, reason): malformed / bad_timestamp / implausible.
    Naive timestamps are taken as UTC (the replay sends CGMacros' date-shifted clock times unchanged; every stage
    treats them the same way, so the alert rhythm is right even though the absolute zone is nominal)."""
    try:
        m = json.loads(raw)
        pid, ts, g = m["patient_id"], m["event_datetime"], m["cgm"]
    except (ValueError, KeyError, TypeError):
        return None, "malformed"
    if not isinstance(pid, str) or not pid or isinstance(g, bool) or not isinstance(g, (int, float)):
        return None, "malformed"
    try:
        t = datetime.fromisoformat(ts)
        if t.tzinfo is not None:
            t = t.astimezone(timezone.utc).replace(tzinfo=None)
    except (ValueError, TypeError, OverflowError):
        return None, "bad_timestamp"
    if not PLAUSIBLE[0] <= float(g) <= PLAUSIBLE[1]:
        return None, "implausible"
    return (pid, t, float(g)), "ok"


def parse_reading(raw: bytes) -> tuple[str, datetime, float] | None:
    """(patient_id, naive UTC time, mg/dL), or None when the message is not a usable reading."""
    return classify_reading(raw)[0]


def utc(t: datetime) -> datetime:
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t.astimezone(timezone.utc)


def alert_id(a: dict) -> str:
    return f"{a['patient_id']}|{utc(a['time']).isoformat()}|{a['tier']}"


def alert_message(a: dict) -> dict:
    return {"alert_id": alert_id(a), "patient_id": a["patient_id"], "time": utc(a["time"]).isoformat(),
            "tier": a["tier"], "direction": a["direction"], "glucose": a["glucose"],
            "projected_20min": a["projected_20min"], "spacing_min": a["spacing_min"], "disclaimer": DISCLAIMER}


class Store(Protocol):
    def save_alert(self, a: dict) -> bool: ...                         # returns True if already published
    def mark_published(self, alert_id: str) -> bool: ...                # False if no row matched
    def unpublished(self) -> list[dict]: ...
    def load_checkpoint(self, topic: str, partition: int) -> tuple[int, dict] | None: ...
    def save_checkpoints(self, topic: str, rows: list[tuple[int, int, dict]]) -> None: ...


class Publisher(Protocol):
    def publish(self, key: str, value: dict) -> None: ...             # delivery reported via Processor.delivered
    def flush(self) -> int: ...                                       # messages still pending


class Processor:
    """Everything between 'a message arrived' and 'checkpoint written', with no Kafka or Postgres in it."""

    def __init__(self, store: Store, publisher: Publisher, topic: str, prefixes: tuple[str, ...] = ("cgmacros-",),
                 max_age: timedelta | None = None, now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                 future_skew: timedelta = timedelta(hours=1), log: Callable[[str], None] | None = None):
        self.store, self.publisher, self.topic = store, publisher, topic
        self.prefixes, self.max_age, self.now, self.future_skew = prefixes, max_age, now, future_skew
        self.log = log or (lambda s: print(s, file=sys.stderr, flush=True))
        self.stream = AlertStream()
        self.next_offset: dict[int, int] = {}
        self.people_in: dict[int, set[str]] = {}
        self.stats: Counter = Counter()
        self.dirty = False                  # a message failed mid-way: memory may be ahead of what was saved
        self._failed_deliveries = 0

    # --- partitions -------------------------------------------------------------------------------------------
    def assign(self, partition: int) -> int | None:
        """Restore a partition's memory; returns the offset to seek to (None = no checkpoint, use the group default)."""
        cp = self.store.load_checkpoint(self.topic, partition)
        self.people_in[partition] = set()
        if cp is None:
            return None
        offset, state = cp
        self.stream.restore(state)
        self.people_in[partition] = set(state)
        self.next_offset[partition] = offset
        self.stats["restored_people"] += len(state)
        return offset

    def revoke(self, partition: int) -> None:
        self.checkpoint([partition])
        for pid in self.people_in.pop(partition, set()):
            self.stream.people.pop(pid, None)
        self.next_offset.pop(partition, None)

    # --- messages ---------------------------------------------------------------------------------------------
    def handle(self, partition: int, offset: int, raw: bytes) -> None:
        """Process one message. The offset advances only after the message is fully handled; if anything raises, the
        processor is marked dirty and refuses to checkpoint, so a restart replays from the last consistent checkpoint
        (safe: alerts are idempotent) instead of saving memory that is ahead of the stored alerts."""
        try:
            self._handle(partition, raw)
        except BaseException:
            self.dirty = True
            raise
        self.next_offset[partition] = offset + 1

    def _warn(self, reason: str, detail: str) -> None:
        """Count, and log the first few of each reason (then every 1000th) so a systematic problem is visible."""
        self.stats[reason] += 1
        n = self.stats[reason]
        if n <= 5 or n % 1000 == 0:
            self.log(f"WARN {reason} (#{n}): {detail}")

    def _handle(self, partition: int, raw: bytes) -> None:
        r, reason = classify_reading(raw)
        if r is None:
            self._warn(f"skipped_{reason}", f"partition {partition}")
            return
        pid, t, g = r
        if not pid.startswith(self.prefixes):
            self.stats["skipped_not_validated_population"] += 1
            return
        if utc(t) > self.now() + self.future_skew:
            # a wrong sensor clock would otherwise make every real reading look 'too old' and silence this person
            self._warn("skipped_future_timestamp", f"{pid} {t.isoformat()}")
            return
        self.people_in.setdefault(partition, set()).add(pid)
        status, alert = self.stream.process(pid, t, g)
        self.stats[status] += 1
        if status == "too_old":
            self._warn("reading_too_old", f"{pid} {t.isoformat()}")
        if alert is None:
            return
        if self.max_age is not None and utc(alert["time"]) < self.now() - self.max_age:
            self._warn("alert_too_old_to_send", f"{alert_id(alert)}")  # live mode: never page about old data
            return
        if self.store.save_alert(alert):
            self.stats["alert_already_published"] += 1
            return
        self.publisher.publish(alert_id(alert), alert_message(alert))
        self.stats["alert_sent"] += 1

    def delivered(self, key: str, ok: bool, error: str | None = None) -> None:
        """Kafka delivery callback. Never raises: a failure here leaves the row unpublished, and the next checkpoint
        resends it (a raise would escape from producer.poll/flush and stop the consumer)."""
        if not ok:
            self._failed_deliveries += 1
            self._warn("delivery_failed", f"{key}: {error}")
            return
        try:
            if self.store.mark_published(key):
                self.stats["delivered"] += 1
            else:
                self._warn("mark_published_no_row", key)
        except Exception as exc:                                       # noqa: BLE001 - must not escape the callback
            self._failed_deliveries += 1
            self._warn("mark_published_failed", f"{key}: {exc!r}")

    def resend_unpublished(self) -> int:
        n = self._resend()
        self.stats["resent_at_startup"] += n
        return n

    def _resend(self) -> int:
        n = 0
        for a in self.store.unpublished():
            if self.max_age is not None and utc(a["time"]) < self.now() - self.max_age:
                self._warn("unpublished_too_old_to_send", alert_id(a))    # same live-mode rule as new alerts
                continue
            self.publisher.publish(alert_id(a), alert_message(a))
            n += 1
        return n

    # --- checkpoints ------------------------------------------------------------------------------------------
    def checkpoint(self, partitions: list[int] | None = None) -> list[tuple[int, int]]:
        """Flush deliveries, retry failed ones, then save memory + next offset per partition in one transaction.
        Skipped (returns []) when a message failed mid-way or deliveries are still pending: the previous checkpoint
        stays the resume point, and replaying from it is safe because alerts are idempotent."""
        if self.dirty:
            self.stats["checkpoint_refused_dirty"] += 1
            return []
        pending = self.publisher.flush()
        if self._failed_deliveries:
            self._failed_deliveries = 0
            self.stats["resent_after_failure"] += self._resend()
            pending = self.publisher.flush()
        if pending:
            self.stats["checkpoint_skipped_pending"] += 1
            return []
        parts = [p for p in (partitions if partitions is not None else list(self.next_offset)) if p in self.next_offset]
        if not parts:
            return []
        rows = [(p, self.next_offset[p], self.stream.snapshot(self.people_in.get(p, set()))) for p in parts]
        self.store.save_checkpoints(self.topic, rows)
        self.stats["checkpoints"] += 1
        return [(p, o) for p, o, _ in rows]


# --- Postgres and Kafka wiring (exercised on the real stack, docs/STREAMING.md) ------------------------------------
class PgStore:
    UPSERT = """
        INSERT INTO glucose_alerts (patient_id, time, tier, direction, glucose, projected_20min, spacing_min)
        VALUES (%(patient_id)s, %(time)s, %(tier)s, %(direction)s, %(glucose)s, %(projected_20min)s, %(spacing_min)s)
        ON CONFLICT (patient_id, time, tier) DO UPDATE SET published = glucose_alerts.published
        RETURNING published"""

    def __init__(self, conn):
        self.conn = conn

    def save_alert(self, a: dict) -> bool:
        with self.conn, self.conn.cursor() as cur:
            cur.execute(self.UPSERT, {**a, "time": utc(a["time"])})
            return cur.fetchone()[0]

    def mark_published(self, key: str) -> bool:
        pid, t, tier = key.rsplit("|", 2)
        with self.conn, self.conn.cursor() as cur:
            cur.execute("UPDATE glucose_alerts SET published = TRUE WHERE patient_id = %s AND time = %s AND tier = %s",
                        (pid, datetime.fromisoformat(t), tier))
            return cur.rowcount == 1

    def unpublished(self) -> list[dict]:
        with self.conn, self.conn.cursor() as cur:
            cur.execute("SELECT patient_id, time, tier, direction, glucose, projected_20min, spacing_min "
                        "FROM glucose_alerts WHERE NOT published ORDER BY time")
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    def load_checkpoint(self, topic: str, partition: int):
        with self.conn, self.conn.cursor() as cur:
            cur.execute("SELECT next_offset, state FROM alert_checkpoints WHERE topic = %s AND kafka_partition = %s",
                        (topic, partition))
            row = cur.fetchone()
        return (row[0], row[1]) if row else None

    def save_checkpoints(self, topic: str, rows) -> None:
        with self.conn, self.conn.cursor() as cur:                   # one transaction for all partitions
            for p, offset, state in rows:
                cur.execute("""
                    INSERT INTO alert_checkpoints (topic, kafka_partition, next_offset, state, updated_at)
                    VALUES (%s, %s, %s, %s::jsonb, now())
                    ON CONFLICT (topic, kafka_partition)
                    DO UPDATE SET next_offset = EXCLUDED.next_offset, state = EXCLUDED.state, updated_at = now()
                    -- fencing: a consumer that lost the partition must not move a newer checkpoint backwards
                    WHERE alert_checkpoints.next_offset <= EXCLUDED.next_offset""",
                            (topic, p, offset, json.dumps(state)))


class KafkaPublisher:
    def __init__(self, producer, topic: str, on_delivery):
        self.producer, self.topic, self.on_delivery = producer, topic, on_delivery

    def publish(self, key: str, value: dict) -> None:
        self.producer.produce(self.topic, key=key.encode(), value=json.dumps(value).encode(),
                              on_delivery=lambda err, msg: self.on_delivery(key, err is None,
                                                                            None if err is None else str(err)))
        self.producer.poll(0)

    def flush(self) -> int:
        """Messages still undelivered after 30 s (0 = all confirmed or failed)."""
        return self.producer.flush(30)


def main() -> None:
    import psycopg2
    from confluent_kafka import OFFSET_BEGINNING, Consumer, KafkaError, Producer, TopicPartition

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kafka-bootstrap", default="kafka:9092")
    ap.add_argument("--raw-topic", default="cgm-raw")
    ap.add_argument("--alerts-topic", default="cgm-alerts")
    ap.add_argument("--group-id", default="cgm-alert-consumer")
    ap.add_argument("--allowed-prefix", action="append", default=None,
                    help="patient id prefix allowed to get alerts (repeatable; default cgmacros-)")
    ap.add_argument("--max-age-min", type=float, default=None,
                    help="live use: do not send alerts about readings older than this (off for replays)")
    ap.add_argument("--checkpoint-every", type=int, default=500)
    ap.add_argument("--checkpoint-seconds", type=float, default=5.0)
    ap.add_argument("--pg-host", default="timescaledb")
    ap.add_argument("--pg-port", default="5432")
    ap.add_argument("--pg-dbname", default=os.environ.get("POSTGRES_DB"))
    ap.add_argument("--pg-user", default=os.environ.get("POSTGRES_USER"))
    ap.add_argument("--pg-password", default=os.environ.get("POSTGRES_PASSWORD"))
    args = ap.parse_args()

    stop = {"now": False, "failed": False}

    def kafka_error(err) -> None:
        print(f"Kafka client error: {err}", file=sys.stderr, flush=True)
        if err.fatal():
            stop.update(now=True, failed=True)

    conn = psycopg2.connect(host=args.pg_host, port=args.pg_port, dbname=args.pg_dbname, user=args.pg_user,
                            password=args.pg_password)
    producer = Producer({"bootstrap.servers": args.kafka_bootstrap, "enable.idempotence": True, "acks": "all",
                         "error_cb": kafka_error})
    publisher = KafkaPublisher(producer, args.alerts_topic, lambda k, ok, e: proc.delivered(k, ok, e))
    proc = Processor(PgStore(conn), publisher, args.raw_topic, tuple(args.allowed_prefix or ["cgmacros-"]),
                     timedelta(minutes=args.max_age_min) if args.max_age_min is not None else None)
    consumer = Consumer({"bootstrap.servers": args.kafka_bootstrap, "group.id": args.group_id,
                         "enable.auto.commit": False, "auto.offset.reset": "earliest", "error_cb": kafka_error})

    def on_assign(c, parts):
        for tp in parts:
            off = proc.assign(tp.partition)
            # no checkpoint = no memory: start from the beginning, never from a Kafka offset without matching memory
            tp.offset = off if off is not None else OFFSET_BEGINNING
        c.assign(parts)
        print(f"assigned {[(tp.partition, tp.offset) for tp in parts]}", flush=True)

    def on_revoke(c, parts):
        for tp in parts:
            proc.revoke(tp.partition)
        print(f"revoked {[tp.partition for tp in parts]}", flush=True)

    def checkpoint() -> None:
        saved = proc.checkpoint()
        if saved:
            consumer.commit(offsets=[TopicPartition(args.raw_topic, p, o) for p, o in saved], asynchronous=True)

    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))

    resent = proc.resend_unpublished()
    print(f"startup: resent {resent} unpublished alerts", flush=True)
    consumer.subscribe([args.raw_topic], on_assign=on_assign, on_revoke=on_revoke)
    since, last_cp, last_stats, errors = 0, time.monotonic(), time.monotonic(), 0
    clean = False
    try:
        while not stop["now"]:
            msg = consumer.poll(1.0)
            producer.poll(0)
            if msg is not None and msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                errors += 1
                print(f"Kafka error ({errors} in a row): {msg.error()}", file=sys.stderr, flush=True)
                if msg.error().fatal() or errors >= 30:
                    raise RuntimeError(f"giving up on Kafka: {msg.error()}")   # restart policy brings us back
            elif msg is not None:
                errors = 0
                proc.handle(msg.partition(), msg.offset(), msg.value())
                since += 1
            if since >= args.checkpoint_every or (since and time.monotonic() - last_cp >= args.checkpoint_seconds):
                checkpoint()
                since, last_cp = 0, time.monotonic()
            if time.monotonic() - last_stats >= 60:
                print(f"stats {dict(proc.stats)}", flush=True)
                last_stats = time.monotonic()
        clean = not stop.get("failed")
    finally:
        # Checkpoint only after a clean stop. After a failure the last checkpoint is the only state known to match
        # the stored alerts; replaying from it is safe (alerts are idempotent).
        try:
            if clean:
                checkpoint()
        finally:
            try:
                consumer.close()
            finally:
                conn.close()
                print(f"stopped ({'clean' if clean else 'after an error'}). {dict(proc.stats)}", flush=True)
    if not clean:
        sys.exit(1)


if __name__ == "__main__":
    main()
