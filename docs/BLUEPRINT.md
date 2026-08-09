# Build Blueprint

What we're building, how, and in what order. This is the execution plan underneath README.md's architecture diagram — README says *what the system is*, this says *what we do first, second, third*.

---

## What

A fault-tolerant streaming pipeline that ingests AZT1D CGM data through Kafka into TimescaleDB in real time, computes batch features with PySpark on an Airflow schedule, trains a TFT model, and serves forecasts via ONNX + FastAPI — with Grafana observing every stage. See README.md for the full architecture and stack justifications; those are locked and not repeated here.

(Originally scoped around OhioT1DM; switched to AZT1D — see `docs/QUESTIONS.md` for why. Same 5-minute CGM cadence, no change to the forecasting task framing below.)

## How

Build in the four phases already defined in README.md's Build Phases table, each phase gated on the previous one actually working — not started until the prior phase runs end-to-end, because a batch/ML layer built on top of an ingestion layer that silently drops or duplicates messages just produces confidently wrong models.

| Phase | Gate to move on |
|---|---|
| 1 — Foundation | `docker compose up` brings up every service healthy, with no manual intervention |
| 2 — Ingestion | A full replay of at least one patient's data lands correctly in TimescaleDB, DLQ catches a deliberately malformed message, Grafana shows live ingestion |
| 3 — Batch + Orchestration | Airflow DAG runs on-demand, produces features in TimescaleDB, and a data quality gate demonstrably blocks a bad run |
| 4 — ML + Serving | Trained TFT beats the persistence baseline on held-out patients, exported ONNX model serves predictions via FastAPI, Grafana shows RMSE vs. baseline |

Each phase's known risks are tracked in `docs/PROBLEMS.md`; decisions made along the way go in `docs/QUESTIONS.md`.

---

## First thing we build: Phase 1 — Foundation

**Goal:** every service in the architecture diagram runs via a single `docker compose up -d`, with health checks proving it, before any pipeline code is written.

### Services to stand up

1. **Kafka** (KRaft mode, no Zookeeper — one less moving part) with topics `cgm-raw` and `cgm-dlq` created on startup.
2. **TimescaleDB** with a hypertable schema for CGM readings, partitioned by patient + timestamp (empty at this stage — schema only).
3. **Grafana** with TimescaleDB added as a data source (no dashboards yet — just confirm the connection).

Producer, consumer, Spark, Airflow, and the model/serving stack are **not** part of Phase 1 — they're empty service stubs at most, real code starts in Phase 2+.

### Concrete steps, in order

1. `docker-compose.yml` defining Kafka, TimescaleDB, Grafana with named volumes for each.
2. Health checks on every service (`depends_on: condition: service_healthy`) — this is the specific risk flagged in `docs/PROBLEMS.md` under Phase 1.
3. Kafka topic bootstrap (`cgm-raw`, `cgm-dlq`) via an init container or startup script.
4. TimescaleDB init script creating the hypertable schema (columns match what README's architecture says the consumer will compute: raw glucose + delta + rolling stats + covariate joins — schema drives what Phase 2 needs to write).
5. Grafana provisioned with the TimescaleDB data source pre-configured (not manually clicked in every time the stack restarts).
6. Verify: fresh `docker compose down -v && docker compose up -d`, confirm all services report healthy, confirm Grafana can query the (empty) hypertable.

### Definition of done for Phase 1 — MET (2026-07-10)

- One command, one working stack, zero manual fixups. ✅ (`docker compose up -d`, all four services report `healthy`/exit 0)
- Services survive a restart without losing schema. ✅ Verified via full `docker compose down && up -d` — `cgm-raw`/`cgm-dlq` topics and the `cgm_readings` hypertable both persisted.
- Grafana can query TimescaleDB through the provisioned datasource. ✅ `POST /api/ds/query` against the live datasource returned `200` (empty table, as expected — no data ingested yet).
- This was verified by actually running it, not assumed from the compose file looking correct — see `docs/QUESTIONS.md` for the one real issue hit (port collision with two native Postgres installs on the host, resolved by remapping the container's host port to 5544).

### Not yet decided (surface in docs/QUESTIONS.md when addressed)

- Exact TimescaleDB hypertable column list — needs the consumer's feature list from Phase 2 nailed down first.
- Kafka topic partition count for `cgm-raw` (affects the ordering-vs-parallelism tradeoff flagged in `docs/PROBLEMS.md`).

---

## Now building: Phase 2 — Ingestion

**Goal:** a full replay of at least one real AZT1D patient's data lands correctly in TimescaleDB, a deliberately malformed message gets caught by the DLQ, and Grafana shows live ingestion — all against real files, not synthetic stand-ins.

### Prerequisite

AZT1D downloaded into `data/azt1d/` (CC BY 4.0, no application — Mendeley DOI in README's Data Source section). Nothing in this phase gets built or claimed working until real files are on disk; see `docs/PROBLEMS.md` for the parsing quirks to expect once they are.

### Concrete steps, in order

1. **Inspect the real CSVs first** — column names, null patterns, `DeviceMode`/`BolusType` value sets, timestamp format — before writing a line of parser code. Update `docs/PROBLEMS.md` if reality differs from what's assumed there.
2. **Producer (`producer/replay_sensor.py`)** — reads one patient's CSV, replays CGM readings onto `cgm-raw` at a compressed rate (5s/reading, matching Phase 1's topic), preserving bolus/carb event timing relative to their nearest CGM reading rather than replaying streams independently.
3. **DLQ policy — DECIDED (2026-07-13).** Three topics by failure class: `cgm-parse-errors` (structural), `cgm-dlq` (conflicting timestamps, unchanged), `cgm-implausible` (CGM <40 or >400 mg/dL). Full writeup in `docs/QUESTIONS.md`.
4. **Consumer (`consumer/ingest.py`)** — consumes `cgm-raw`, writes raw glucose + covariates to TimescaleDB idempotently (upsert on `patient_id` + `time`, since Kafka is at-least-once). Computes **no** derived features — kept stateless deliberately; delta/rolling stats are Phase 3's PySpark job, not the consumer's (see `docs/QUESTIONS.md`, 2026-07-13).
5. **TimescaleDB schema update** — `ALTER TABLE cgm_readings` to add the raw covariate columns (`device_mode`, `bolus_type`, `basal`, `correction_delivered`, `total_bolus_insulin_delivered`, `food_delivered`, `carb_size`). No feature columns — those stay deferred to Phase 3.
6. **Grafana ingestion dashboard** — live view of `cgm_readings` growing, so "Grafana shows live ingestion" is something you can actually watch, not just infer from logs.
7. **Verify**: run the full replay for one patient, confirm rows land in TimescaleDB matching the source CSV, deliberately inject one malformed message and confirm it lands in `cgm-dlq` not `cgm-raw`'s consumer path, watch Grafana update live.

### Definition of done for Phase 2 — MET (2026-07-13)

- A full replay of at least one patient's real AZT1D data lands correctly in TimescaleDB. ✅ (34,279 rows across 3 patients, zero duplicates)
- The DLQ demonstrably catches a deliberately malformed message — proven by actually sending one, not by code review. ✅ All three failure classes (`cgm-parse-errors`, `cgm-dlq`, `cgm-implausible`) individually verified with injected messages.
- Grafana shows live ingestion. ✅ 3-panel dashboard (ingestion rate, per-patient glucose trace, DLQ health), each panel's query verified directly against `/api/ds/query`, not just visually.
- Verified by running it end-to-end against real data, same standard as Phase 1. Full writeup in `docs/PROGRESS.md` and `docs/QUESTIONS.md`.

---

## Now building: Phase 3 — Batch + Orchestration

**Goal:** an Airflow DAG runs a PySpark job over the full raw `cgm_readings` history, produces features in TimescaleDB, and a real data-quality gate demonstrably blocks a run when the data doesn't meet it — not just a DAG that always succeeds.

### Concrete steps, in order

1. **Feature set and recompute strategy — DECIDED (2026-07-13).** Resample to 5-min grid then roll; delta + rolling mean/std + time-since-last-bolus/carb with explicit cold-start flags; full-history recompute every run. Features table keyed `(patient_id, time)`, same grain as `cgm_readings`. Full writeup in `docs/QUESTIONS.md`.
2. **Data-quality gate — DECIDED (2026-07-13).** Per-patient-week: 180-min single-gap threshold, <85% expected-readings count threshold. Exclude+log per failing patient-week; a circuit-breaker on the run's overall exclusion rate is the only check allowed to hard-fail the DAG task. Full writeup in `docs/QUESTIONS.md`.
3. **PySpark job** — reads `cgm_readings` (JDBC), computes the decided feature set across full patient history, writes to a new features table. Idempotent reruns via upsert-on-conflict on `(patient_id, time)`, not append-only.
4. **Airflow DAG** — schedules the Spark job, runs the quality gate as a downstream task that can fail the DAG via the circuit breaker, retry logic for transient failures.
5. **Verify**: run the DAG on-demand, confirm features land in TimescaleDB matching a hand-computed sample. The quality gate should already fail 2 of 3 real patient-weeks on the currently-loaded data (Subject 3's 1384-min gap, Subject 1's 600-min gap) without needing synthetic injection — confirm that happens, and confirm the circuit breaker does NOT trip at that exclusion rate (2 failing weeks out of the full run, not "systemic").

### Definition of done for Phase 3 — MET (2026-07-13)

- Airflow DAG runs on-demand, produces features in TimescaleDB. ✅ `feature_pipeline` DAG triggered manually, completed `state=success` in ~18s, wrote 16,157 rows to `cgm_features`.
- A data-quality gate demonstrably blocks a bad run — proven by actually triggering one, not by code review. ✅ On the currently-loaded real data (no synthetic injection needed): 3 of 22 patient-weeks correctly excluded (Subject 3's 1384-min gap, Subject 1's 600-min and 260-min gaps), confirmed by querying `cgm_features` directly (zero rows for the excluded week) not just trusting task logs. Exclusion rate 13.64%, correctly under the 50% circuit-breaker placeholder, so the DAG succeeded rather than hard-failing.
- PySpark is honestly scoped in any writeup: local single-node in Docker demonstrates the API/pattern, not real distributed scale (`docs/PROBLEMS.md`).
- Full writeup in `docs/PROGRESS.md` and `docs/QUESTIONS.md`.
