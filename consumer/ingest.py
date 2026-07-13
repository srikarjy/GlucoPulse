"""
Consumes cgm-raw and writes raw glucose + covariates to TimescaleDB.

Deliberately stateless / raw-only: no delta or rolling-stat computation here
-- that's Phase 3's PySpark batch job over the full history, not this
consumer's (see docs/QUESTIONS.md, 2026-07-13 entry).

Failure classes route to separate topics, each with a different downstream
intent:
  - malformed JSON / missing or wrong-typed required field -> cgm-parse-errors
  - well-formed but CGM < 40 or > 400 mg/dL (outside the Dexcom G6's own
    reporting range)                                        -> cgm-implausible
  - otherwise                                                -> cgm_readings

Offsets are committed only after the message's outcome (DB write, or a
confirmed produce to an error topic) actually completes -- at-least-once,
made safe by ON CONFLICT (patient_id, time) DO NOTHING. Commit granularity is
per-message: this consumer isn't the throughput bottleneck, so batching
commits would only widen the crash-replay window for no real benefit.
"""

import argparse
import json
import os
import sys

import psycopg2
from confluent_kafka import Consumer, Producer

REQUIRED_FIELDS = ("patient_id", "event_datetime", "cgm")
OPTIONAL_FIELDS = (
    "device_mode",
    "bolus_type",
    "basal",
    "correction_delivered",
    "total_bolus_insulin_delivered",
    "food_delivered",
    "carb_size",
)

IMPLAUSIBLE_LOW = 40
IMPLAUSIBLE_HIGH = 400

INSERT_SQL = """
INSERT INTO cgm_readings (
    patient_id, time, glucose_value, device_mode, bolus_type, basal,
    correction_delivered, total_bolus_insulin_delivered, food_delivered, carb_size
) VALUES (
    %(patient_id)s, %(event_datetime)s, %(cgm)s, %(device_mode)s, %(bolus_type)s,
    %(basal)s, %(correction_delivered)s, %(total_bolus_insulin_delivered)s,
    %(food_delivered)s, %(carb_size)s
)
ON CONFLICT (patient_id, time) DO NOTHING;
"""


def parse_message(raw: bytes) -> dict:
    message = json.loads(raw)
    for field in REQUIRED_FIELDS:
        if message.get(field) is None:
            raise ValueError(f"missing required field: {field}")
    if not isinstance(message["cgm"], (int, float)):
        raise ValueError(f"cgm is not numeric: {message['cgm']!r}")
    return message


def is_implausible(message: dict) -> bool:
    return not (IMPLAUSIBLE_LOW <= message["cgm"] <= IMPLAUSIBLE_HIGH)


def normalize(message: dict) -> dict:
    row = {field: message[field] for field in REQUIRED_FIELDS}
    for field in OPTIONAL_FIELDS:
        row[field] = message.get(field)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kafka-bootstrap", default="kafka:9092")
    parser.add_argument("--raw-topic", default="cgm-raw")
    parser.add_argument("--parse-error-topic", default="cgm-parse-errors")
    parser.add_argument("--implausible-topic", default="cgm-implausible")
    parser.add_argument("--group-id", default="cgm-ingest-consumer")
    parser.add_argument("--pg-host", default="timescaledb")
    parser.add_argument("--pg-port", default="5432")
    parser.add_argument("--pg-dbname", default=os.environ.get("POSTGRES_DB"))
    parser.add_argument("--pg-user", default=os.environ.get("POSTGRES_USER"))
    parser.add_argument("--pg-password", default=os.environ.get("POSTGRES_PASSWORD"))
    args = parser.parse_args()

    conn = psycopg2.connect(
        host=args.pg_host,
        port=args.pg_port,
        dbname=args.pg_dbname,
        user=args.pg_user,
        password=args.pg_password,
    )
    conn.autocommit = True

    consumer = Consumer({
        "bootstrap.servers": args.kafka_bootstrap,
        "group.id": args.group_id,
        "enable.auto.commit": False,
        "auto.offset.reset": "earliest",
    })
    consumer.subscribe([args.raw_topic])

    producer = Producer({"bootstrap.servers": args.kafka_bootstrap})

    written = parse_errors = implausible = 0
    print(f"Consuming {args.raw_topic} -> {args.pg_dbname}@{args.pg_host}:{args.pg_port} ...")

    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                print(f"Kafka error: {msg.error()}", file=sys.stderr)
                continue

            try:
                message = parse_message(msg.value())
            except (json.JSONDecodeError, ValueError) as e:
                producer.produce(
                    args.parse_error_topic,
                    key=msg.key(),
                    value=json.dumps({
                        "reason": str(e),
                        "raw": msg.value().decode(errors="replace"),
                    }).encode(),
                )
                producer.flush()
                consumer.commit(msg)
                parse_errors += 1
                continue

            if is_implausible(message):
                producer.produce(
                    args.implausible_topic,
                    key=msg.key(),
                    value=json.dumps(message).encode(),
                )
                producer.flush()
                consumer.commit(msg)
                implausible += 1
                continue

            with conn.cursor() as cur:
                cur.execute(INSERT_SQL, normalize(message))
            consumer.commit(msg)
            written += 1

            total = written + parse_errors + implausible
            if total % 500 == 0:
                print(f"... written={written} parse_errors={parse_errors} implausible={implausible}")
    except KeyboardInterrupt:
        pass
    finally:
        print(f"Done. written={written} parse_errors={parse_errors} implausible={implausible}")
        consumer.close()


if __name__ == "__main__":
    main()
