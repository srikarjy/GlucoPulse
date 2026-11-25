"""Break the streaming alert pipeline on purpose and check it (docs/STREAMING.md, "Chaos run").

Each scenario streams one or two real CGMacros people into cgm-raw with a fault, then compares what the alert service
stored and published with the expected answer: AlertStream run offline on exactly the same messages in exactly the
same arrival order (for clean data that equals the tested batch rule, tests/streaming/test_alert_state.py).

  duplicates  every reading sent twice                                          (cgmacros-007)
  reorder     about 10% of readings swapped with their neighbour, i.e. late    (cgmacros-008)
  gap         20 minutes of readings missing                                    (cgmacros-009)
  kill        docker kill of the alert service mid-stream, then restart         (cgmacros-010, -011)
  garbage     broken JSON, impossible values and a year-2099 reading mixed in   (cgmacros-012)

Needs the stack from docs/STREAMING.md running (kafka, timescaledb, consumer, alerts) and data/stream/ exported:
    .venv/bin/python scripts/export_stream_csv.py --patients 7 8 9 10 11 12
    .venv/bin/python scripts/chaos_streaming.py                 # all scenarios -> model/results/streaming_chaos.json
Each person must be new to the stack (the comparison is per person); use --suffix to rerun with fresh ids.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from consumer.alert_state import AlertStream  # noqa: E402

Row = tuple[str, float]            # (timestamp text "YYYY-MM-DD HH:MM:SS", mg/dL)
TS = "%Y-%m-%d %H:%M:%S"


# --- faults (pure, unit-tested in tests/streaming/test_chaos.py) ---------------------------------------------------
def with_duplicates(rows: list[Row]) -> list[Row]:
    return [r for r in rows for _ in (0, 1)]


def reordered(rows: list[Row], rng: random.Random, p: float = 0.1) -> list[Row]:
    """Swap non-overlapping neighbour pairs with probability p: the second reading of a pair arrives late."""
    out, i = list(rows), 0
    while i < len(out) - 1:
        if rng.random() < p:
            out[i], out[i + 1] = out[i + 1], out[i]
            i += 2
        else:
            i += 1
    return out


def with_gap(rows: list[Row], start: int, minutes: int = 20) -> list[Row]:
    t0 = datetime.strptime(rows[start][0], TS)
    t1 = t0 + timedelta(minutes=minutes)
    return [r for r in rows if not t0 <= datetime.strptime(r[0], TS) < t1]


def expected_alerts(pid: str, arrivals: list[Row]) -> set[tuple[str, str]]:
    """What the service must store for this person: AlertStream over the same arrival order."""
    s = AlertStream()
    out = set()
    for ts, g in arrivals:
        _, a = s.process(pid, datetime.strptime(ts, TS), g)
        if a:
            out.add((a["time"].replace(tzinfo=timezone.utc).isoformat(), a["tier"]))
    return out


def message(pid: str, ts: str, g) -> bytes:
    return json.dumps({"patient_id": pid, "event_datetime": ts, "cgm": g}).encode()


# --- talking to the stack -------------------------------------------------------------------------------------------
def load(pid: str) -> list[Row]:
    with (ROOT / "data" / "stream" / f"{pid}.csv").open() as f:
        return [(r["timestamp"], float(r["glucose"])) for r in csv.DictReader(f)]


def env() -> dict:
    vals = {}
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            vals[k.strip()] = v.strip().strip('"').strip("'")
    return vals


class Stack:
    def __init__(self, bootstrap: str, pg_port: int):
        import psycopg2
        from confluent_kafka import Producer
        e = env()
        self.conn = psycopg2.connect(host="localhost", port=pg_port, dbname=e["POSTGRES_DB"], user=e["POSTGRES_USER"],
                                     password=e["POSTGRES_PASSWORD"])
        self.conn.autocommit = True
        self.producer = Producer({"bootstrap.servers": bootstrap, "enable.idempotence": True})

    def send(self, msgs: list[tuple[str, bytes]]) -> None:
        for key, value in msgs:
            while True:
                try:
                    self.producer.produce("cgm-raw", key=key.encode(), value=value)
                    break
                except BufferError:
                    self.producer.poll(0.5)
            self.producer.poll(0)
        if self.producer.flush(60):
            raise RuntimeError("readings not delivered to Kafka")

    def stored(self, pids: list[str]) -> dict[str, dict]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT patient_id, time, tier, published FROM glucose_alerts WHERE patient_id = ANY(%s)",
                        (pids,))
            out: dict[str, dict] = {p: {"alerts": set(), "unpublished": 0} for p in pids}
            for pid, t, tier, pub in cur.fetchall():
                out[pid]["alerts"].add((t.astimezone(timezone.utc).isoformat(), tier))
                out[pid]["unpublished"] += 0 if pub else 1
            return out

    def lag(self) -> int | None:
        r = subprocess.run(["docker", "exec", "glucopulse-kafka", "/opt/kafka/bin/kafka-consumer-groups.sh",
                            "--bootstrap-server", "kafka:9092", "--describe", "--group", "cgm-alert-consumer"],
                           capture_output=True, text=True)
        total, rows = 0, [ln.split() for ln in r.stdout.splitlines() if " cgm-raw " in f" {ln} "]
        if not rows:
            return None                                # group not (yet) visible, e.g. during a rebalance
        for cols in rows:                              # GROUP TOPIC PARTITION CURRENT-OFFSET LOG-END-OFFSET LAG ...
            end, lag = cols[4], cols[5]
            if lag == "-":
                if end != "0":
                    return None                        # a partition with data but no committed offset yet
                continue                               # an empty partition that never had a message: nothing to do
            total += int(lag)
        return total

    def wait_until_done(self, pids: list[str], timeout: float = 240) -> float:
        """Until the alert consumer's committed lag is 0 and the stored alerts stop changing."""
        t0, last, stable = time.monotonic(), None, 0
        while time.monotonic() - t0 < timeout:
            snap = {p: (len(v["alerts"]), v["unpublished"]) for p, v in self.stored(pids).items()}
            stable = stable + 1 if (snap == last and self.lag() == 0) else 0
            if stable >= 3:
                return time.monotonic() - t0
            last = snap
            time.sleep(3)
        raise TimeoutError(f"alert service did not finish within {timeout}s")

    def topic_ids(self) -> list[str]:
        r = subprocess.run(["docker", "exec", "glucopulse-kafka", "/opt/kafka/bin/kafka-console-consumer.sh",
                            "--bootstrap-server", "kafka:9092", "--topic", "cgm-alerts", "--from-beginning",
                            "--timeout-ms", "8000", "--property", "print.key=true", "--property", "print.value=false"],
                           capture_output=True, text=True)
        return [ln for ln in r.stdout.splitlines() if ln.strip()]


def docker(*args: str) -> None:
    subprocess.run(["docker", *args], check=True, capture_output=True)


# --- scenarios ------------------------------------------------------------------------------------------------------
def scenario_inputs(name: str, suffix: str, rng: random.Random) -> dict[str, list]:
    """{patient id: arrival-ordered list of rows (or raw garbage bytes)}."""
    def pid(n: int) -> str:
        return f"cgmacros-{n:03d}{suffix}"
    if name == "duplicates":
        return {pid(7): with_duplicates(load("cgmacros-007"))}
    if name == "reorder":
        return {pid(8): reordered(load("cgmacros-008"), rng)}
    if name == "gap":
        rows = load("cgmacros-009")
        return {pid(9): with_gap(rows, len(rows) // 2)}
    if name == "kill":
        return {pid(10): load("cgmacros-010"), pid(11): load("cgmacros-011")}
    if name == "garbage":
        rows: list = list(load("cgmacros-012"))
        junk = [b"{not json", json.dumps({"patient_id": pid(12), "cgm": 120}).encode(),
                message(pid(12), "2020-05-03 10:00:00", 12), message(pid(12), "2020-05-03 10:00:00", 900),
                message(pid(12), "2099-01-01 00:00:00", 120), message(pid(12), "not a time", 120)]
        for k, j in enumerate(junk):
            rows.insert((k + 1) * len(rows) // (len(junk) + 1), j)
        return {pid(12): rows}
    raise ValueError(name)


def run(name: str, stack: Stack, suffix: str, rng: random.Random) -> dict:
    inputs = scenario_inputs(name, suffix, rng)
    expected = {p: expected_alerts(p, [r for r in rows if isinstance(r, tuple)]) for p, rows in inputs.items()}
    msgs = [(p, r if isinstance(r, bytes) else message(p, *r)) for p, rows in inputs.items() for r in rows]
    killed_with = None
    if name == "kill":
        sender = threading.Thread(target=stack.send, args=(msgs,))
        sender.start()
        time.sleep(4)
        docker("kill", "glucopulse-alerts")
        killed_with = sum(len(v["alerts"]) for v in stack.stored(list(inputs)).values())
        sender.join()
        time.sleep(2)
        docker("start", "glucopulse-alerts")
    else:
        stack.send(msgs)
    took = stack.wait_until_done(list(inputs))
    got = stack.stored(list(inputs))
    people = {}
    for p in inputs:
        missing, extra = expected[p] - got[p]["alerts"], got[p]["alerts"] - expected[p]
        people[p] = {"messages": len(inputs[p]), "expected": len(expected[p]), "stored": len(got[p]["alerts"]),
                     "missing": len(missing), "extra": len(extra), "unpublished": got[p]["unpublished"]}
    ok = all(v["missing"] == v["extra"] == v["unpublished"] == 0 for v in people.values())
    out = {"scenario": name, "ok": ok, "people": people, "seconds_to_settle": round(took, 1)}
    if killed_with is not None:
        out["alerts_stored_when_killed"] = killed_with
    if name == "reorder":                            # how much late data changes the answer vs. sorted data
        p = next(iter(inputs))
        out["late_readings"] = sum(1 for a, b in zip(inputs[p], inputs[p][1:]) if b[0] < a[0])
        out["alerts_differing_from_sorted_order"] = len(expected_alerts(p, sorted(inputs[p])) ^ expected[p])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="*", default=["duplicates", "reorder", "gap", "garbage", "kill"])
    ap.add_argument("--suffix", default="", help="appended to patient ids, for a fresh rerun (e.g. -r2)")
    ap.add_argument("--bootstrap", default="localhost:29092")
    ap.add_argument("--pg-port", type=int, default=5544)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=str(ROOT / "model" / "results" / "streaming_chaos.json"))
    args = ap.parse_args()
    stack, rng = Stack(args.bootstrap, args.pg_port), random.Random(args.seed)
    results = []
    for name in args.scenarios:
        r = run(name, stack, args.suffix, rng)
        results.append(r)
        print(json.dumps(r), flush=True)
    pids = tuple(f"{p}|" for r in results for p in r["people"])
    ids = [i for i in stack.topic_ids() if i.startswith(pids)]          # this run's people only
    summary = {"when": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seed": args.seed,
               "all_ok": all(r["ok"] for r in results), "scenarios": results,
               "topic_messages_added": len(ids), "topic_unique_alert_ids_added": len(set(ids))}
    Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")
    print(f"all_ok={summary['all_ok']}; topic +{len(ids)} messages, {len(set(ids))} unique -> {args.out}")


if __name__ == "__main__":
    main()
