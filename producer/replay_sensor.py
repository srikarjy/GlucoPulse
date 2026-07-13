"""
Replays one AZT1D patient's real CSV as a live Kafka stream.

Two modes:
  bulk  - no delay between readings; loads a patient's full history quickly
          (for populating TimescaleDB so later phases have data to work with)
  live  - fixed 5s delay between readings, limited to --hours of real time
          (for watching Grafana update as a demo)

Timestamp-groups with more than one row (see docs/QUESTIONS.md) are handled as:
  - identical rows            -> collapsed to one message on the raw topic
  - genuinely conflicting rows -> sent to the DLQ topic instead of the raw topic
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from confluent_kafka import Producer

# Subject 14 alone uses this column name instead of "CGM" (see docs/QUESTIONS.md)
GLUCOSE_COLUMN_ALIASES = ("CGM", "Readings (CGM / BGM)")

NUMERIC_FIELDS = (
    "Basal",
    "CorrectionDelivered",
    "TotalBolusInsulinDelivered",
    "FoodDelivered",
    "CarbSize",
)


def find_glucose_column(fieldnames):
    for name in GLUCOSE_COLUMN_ALIASES:
        if name in fieldnames:
            return name
    raise ValueError(f"No known glucose column in header: {fieldnames}")


def load_patient_rows(data_dir: Path, patient: str):
    csv_path = data_dir / f"Subject {patient}" / f"Subject {patient}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        glucose_col = find_glucose_column(reader.fieldnames)
        rows = list(reader)

    # Group by EventDateTime, preserving first-seen (chronological) order.
    groups = {}
    for row in rows:
        groups.setdefault(row["EventDateTime"], []).append(row)

    return groups, glucose_col


def cast_row(raw_row: dict, glucose_col: str, patient: str) -> dict:
    def to_float(v):
        return float(v) if v not in ("", None) else None

    return {
        "patient_id": patient,
        "event_datetime": raw_row["EventDateTime"],
        "cgm": to_float(raw_row[glucose_col]),
        "device_mode": raw_row["DeviceMode"] or None,
        "bolus_type": raw_row["BolusType"] or None,
        "basal": to_float(raw_row["Basal"]),
        "correction_delivered": to_float(raw_row["CorrectionDelivered"]),
        "total_bolus_insulin_delivered": to_float(raw_row["TotalBolusInsulinDelivered"]),
        "food_delivered": to_float(raw_row["FoodDelivered"]),
        "carb_size": to_float(raw_row["CarbSize"]),
    }


def rows_are_identical(rows: list) -> bool:
    return len({tuple(sorted(r.items())) for r in rows}) == 1


def delivery_report(err, msg):
    if err is not None:
        print(f"DELIVERY FAILED: {err}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patient", required=True, help="Subject number, e.g. 1")
    parser.add_argument("--mode", required=True, choices=["bulk", "live"])
    parser.add_argument("--hours", type=float, default=6.0, help="live mode only: replay this many hours from the start of the file")
    parser.add_argument("--data-dir", default="/data", help="Path to the CGM Records directory")
    parser.add_argument("--kafka-bootstrap", default="kafka:9092")
    parser.add_argument("--raw-topic", default="cgm-raw")
    parser.add_argument("--dlq-topic", default="cgm-dlq")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    groups, glucose_col = load_patient_rows(data_dir, args.patient)
    print(f"Loaded {sum(len(v) for v in groups.values())} rows, {len(groups)} distinct timestamps "
          f"for patient {args.patient} (glucose column: {glucose_col})")

    timestamps = list(groups.keys())
    if args.mode == "live":
        from datetime import datetime, timedelta
        start = datetime.strptime(timestamps[0], "%Y-%m-%d %H:%M:%S")
        cutoff = start + timedelta(hours=args.hours)
        timestamps = [
            ts for ts in timestamps
            if datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") <= cutoff
        ]
        print(f"live mode: replaying {len(timestamps)} timestamps within {args.hours}h of {start}")

    producer = Producer({"bootstrap.servers": args.kafka_bootstrap})

    sent_raw = 0
    sent_dlq = 0
    for i, ts in enumerate(timestamps):
        rows = groups[ts]
        key = args.patient.encode()

        if len(rows) == 1 or rows_are_identical(rows):
            message = cast_row(rows[0], glucose_col, args.patient)
            producer.produce(args.raw_topic, key=key, value=json.dumps(message).encode(), callback=delivery_report)
            sent_raw += 1
        else:
            dlq_message = {
                "patient_id": args.patient,
                "event_datetime": ts,
                "reason": "conflicting_timestamp_group",
                "rows": rows,
            }
            producer.produce(args.dlq_topic, key=key, value=json.dumps(dlq_message).encode(), callback=delivery_report)
            sent_dlq += 1

        producer.poll(0)

        if args.mode == "live":
            print(f"[{ts}] sent (patient {args.patient})")
            time.sleep(5)
        elif i % 1000 == 0:
            print(f"... {i}/{len(timestamps)} timestamps processed")

    producer.flush()
    print(f"Done. Sent {sent_raw} messages to {args.raw_topic}, {sent_dlq} messages to {args.dlq_topic}.")


if __name__ == "__main__":
    main()
