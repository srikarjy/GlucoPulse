# Streaming glucose alerts

The tested CGM trend alert (`serving/alerts.py`, docs/TRIAGE.md §9, §15-23) runs on every reading as it arrives, not
only when someone asks:

```
export_stream_csv.py ──► replay_sensor.py --source csv ──► Kafka cgm-raw ─┬─► ingest.py ──► cgm_readings
 (CGMacros Dexcom,          (bulk or live)                 (3 partitions,  │
  adults not on insulin)                                    keyed by person)└─► alerts_stream.py ──► glucose_alerts
                                                                                   │                  (+ checkpoints)
                                                                                   └──► Kafka cgm-alerts
```

## Run it

```bash
docker compose up -d kafka kafka-init timescaledb consumer alerts
.venv/bin/python scripts/export_stream_csv.py --patients 1 21          # -> data/stream/cgmacros-001.csv, ...
docker compose --profile tools run --rm producer --source csv --patient cgmacros-001 --mode bulk --data-dir /stream
```

On an existing database volume, apply the new tables once:
`docker exec -i glucopulse-timescaledb psql -U $POSTGRES_USER -d $POSTGRES_DB < timescaledb/init/009_glucose_alerts.sql`
(init scripts only run on a fresh volume), and create the topic `cgm-alerts` (3 partitions).

## Who gets alerts

Only patient ids starting with `cgmacros-` (repeatable `--allowed-prefix`). The alert is validated only in adults not
using insulin; AZT1D (type 1, on insulin) readings share the topic and are counted and skipped. Invalid or implausible
readings are skipped here; `ingest.py` routes them to the error topics.

## Guarantees and how they are made

| Failure | What happens | Mechanism |
|---|---|---|
| A reading is delivered twice, or replayed after a restart | One alert row; it is not sent again once marked published | `glucose_alerts` primary key `(patient_id, time, tier)`; insert is a no-op on conflict |
| Crash after an alert is saved, before Kafka confirms it | It is sent after the restart | Outbox: `published` is set only in the delivery callback; unpublished rows are re-sent when the reading is replayed, and by a startup sweep |
| Crash at any point | Memory and position restored together | Every 500 messages / 5 s: delivery flush, then the `AlertStream` memory of each partition's people and the next offset in one transaction (`alert_checkpoints`). On assignment the consumer restores it and seeks there. Kafka committed offsets are only for lag monitoring. |
| The 30-min quiet period across a restart | Same decisions as an uninterrupted run | The quiet period is a chain of earlier alerts, so last-alert times per tier are part of the saved memory (`consumer/alert_state.py`) |
| A late or out-of-order reading | Used for later trends, never an alert about the past | Only a person's newest reading can alert |

Each message's `projected_20min` is capped to the 40-400 mg/dL sensor range with `projection_capped: true` when the
straight line ran past it, as the MCP tool reports it (a live demo on 2 Feb of cgmacros-008 showed -1, 29 and 35 mg/dL
before this cap; `glucose_alerts` keeps the raw value).

Delivery to `cgm-alerts` is at-least-once; each message has a fixed `alert_id` (`patient|time|tier`) so a consumer can
drop a repeat.

Failure handling added after an independent review (ECC python-reviewer and silent-failure-hunter agents):

| Found | Fix | Test |
|---|---|---|
| A database error while saving an alert let the shutdown checkpoint record the reading as done: the alert was lost for good | The offset advances only after a message is fully handled; any error marks the processor dirty and it refuses to checkpoint (also on rebalance), so the restart replays from the last consistent checkpoint | `test_failed_save_is_never_checkpointed_past_and_the_alert_comes_back` (fails on the old code) |
| A failed Kafka delivery waited for the next restart | Each checkpoint re-sends failed deliveries; a checkpoint is skipped while deliveries are still pending | `test_failed_delivery_is_resent_...`, `test_checkpoint_waits_...` |
| A reading with a wrong sensor clock (e.g. 2099) would make every later reading "too old" and silence that person | Readings more than 1 h in the future are skipped and logged | `test_a_future_timestamp_cannot_silence_a_person` |
| Fatal Kafka errors were printed and ignored | Fatal or 30 consecutive errors stop the process with exit code 1 (compose restarts it); `error_cb` logs client errors | (real stack) |
| Errors in the delivery callback could stop the loop; a 0-row "mark published" counted as delivered | The callback never raises; the row stays unpublished for the next re-send; 0-row updates are logged | `test_mark_published_*` |
| Startup re-send ignored the live age limit | Same `--max-age-min` rule as new alerts | `test_startup_resend_respects_the_live_age_limit` |
| A consumer that lost its partition could move a newer checkpoint backwards | The checkpoint upsert only moves forward | (SQL `WHERE next_offset <= EXCLUDED.next_offset`) |
| No checkpoint but a Kafka committed offset would resume with empty memory | No checkpoint means start from the beginning | (real stack: offset -2 on restart) |

Skipped readings are counted per reason (`skipped_malformed`, `skipped_bad_timestamp`, `skipped_implausible`,
`skipped_future_timestamp`, `reading_too_old`); the first five of each and every thousandth are logged, and all
counters print every 60 s.

## Measured (2026-11-24, OrbStack, local Kafka 3.7 + TimescaleDB 2.16)

- **Same alerts as the tested rule.** Streaming decisions equal `serving.alerts.evaluate()` run on the full history up
  to every reading: cgmacros-001 72 alerts, cgmacros-021 45 (unit test on 1,200 readings; full histories checked
  offline at ~2.5 ms per reading).
- **On the real stack:** cgmacros-001 72/72 and cgmacros-021 45/45 alerts stored and published.
- **Hard crash:** cgmacros-003 and -005 replayed together; `docker kill` of the alert service 6 s in (21 alerts saved),
  restart. It resumed from checkpoints (partition offsets 4,857 and 13,728) and ended with exactly the expected 32 and
  34 alerts, all published.
- **Topic:** 183 messages on `cgm-alerts`, 183 unique `alert_id`s, 183 rows.
- **Second run, after the review fixes** (tables emptied, topic recreated, the whole `cgm-raw` topic reprocessed):
  `docker kill` 4 s in, before the first checkpoint, with 27 alerts saved and 1 not yet confirmed. On restart the
  startup sweep re-sent the unconfirmed one, processing restarted from the beginning (no checkpoint), the 27 saved
  alerts were recognized and not re-sent, and the end state was again 72 / 32 / 34 / 45 alerts, all published.
  The topic held **184 messages for 183 alerts**: one alert had reached Kafka just before the kill but was not yet
  marked published, so it was sent again. This is the at-least-once case the `alert_id` exists for; nothing was lost.
- **Safety filter:** an AZT1D patient (10,866 readings) produced no alerts; the readings were counted as skipped.
- After `docker kill`, Kafka hands the partitions back only when the dead member's session times out (~45 s); alerts
  for that period arrive late, not lost.

## Chaos run (scripted, `scripts/chaos_streaming.py`)

Each scenario streams real CGMacros people into the running stack with a fault. The expected answer is `AlertStream`
run offline on exactly the same messages in the same arrival order. Run 2026-11-24 (seed 2026, ids suffixed `-r1`),
results in `model/results/streaming_chaos.json`:

| Scenario | Person | Messages | Expected = stored | Missing / extra / unpublished | Settled in |
|---|---|---|---|---|---|
| Every reading sent twice | 007 | 3,650 | 6 = 6 | 0 / 0 / 0 | 16 s |
| ~10% of readings one step late (258) | 008 | 2,861 | 73 = 73 | 0 / 0 / 0 | 20 s |
| 20 minutes missing | 009 | 2,854 | 24 = 24 | 0 / 0 / 0 | 20 s |
| Broken JSON, 12 and 900 mg/dL, a 2099 timestamp, a bad time mixed in | 012 | 2,865 | 45 = 45 | 0 / 0 / 0 | 19 s |
| `docker kill` 4 s in (46 alerts stored), restart | 010, 011 | 5,712 | 63 + 24 = 87 | 0 / 0 / 0 | 63 s (Kafka session timeout) |

The topic gained 235 messages for 235 unique `alert_id`s in this run (no re-send was needed).

**Finding: late readings cost alerts.** With 258 readings arriving one step late, cgmacros-008 got 73 alerts instead of
the 79 the same data gives in time order: 6 alerts (8%) never fired, none were added. By design a late reading cannot
raise an alert (so nothing is ever said about the past), but the reading that arrived before it was already judged
with a hole in its history, which breaks the trend, and that decision is not revisited. The service does exactly what
it was designed to do here; the design itself loses alerts when data is only minutes late. A possible fix (re-decide
the newest reading when a late reading fills a hole right before it) changes the alert semantics and is not built.

A plain `evaluate(last 2 hours)` per reading also matched on this whole-number data; it diverged only on a constructed
5-hour creep toward 70 with decimal values (3 of 6 alerts). The explicit per-tier state avoids that case.

## Limits

- Replays are of 2020 data, so the "too old to page" check is off by default; set `--max-age-min` for live use.
- Changing the number of `cgm-raw` partitions moves people between partitions and invalidates the checkpoints;
  delete `alert_checkpoints` rows and replay from the start if that is ever done.
- Late readings are never re-decided, so minutes-late data loses alerts (6 of 79 in the chaos run, above).
- Each person must stay on one partition; the producer keys messages by patient id, which guarantees it.
- Naive timestamps are treated as UTC. CGMacros clock times are date-shifted and sent unchanged, so alert timing is
  right but the stored time zone is nominal.
- No metric is exported yet for unpublished alerts or a stuck consumer; the counters are only in the logs.
- One consumer instance was tested. Several instances share partitions through the consumer group, but rebalancing
  with more than one instance has not been exercised.
- Not a medical device: screening information only, validated only in adults not using insulin.
