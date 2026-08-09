# GlucoPulse — Project Status

_Consolidated snapshot as of 2026-07-15. For authoritative detail see `BLUEPRINT.md` (plan/gates), `QUESTIONS.md` (decision log), `PROBLEMS.md` (known risks), and `PROGRESS.md` (chronological build record)._

GlucoPulse is a data-engineering portfolio project. The **pipeline is the hero**; the ML model exists to validate the pipeline. Every decision is justified by a requirement, not by resume appeal. Domain: continuous glucose monitoring (CGM) data for Type 1 diabetes, forecasting glucose at **T+30 and T+60 minutes**.

---

## Dataset

**AZT1D** (arXiv:2506.14789, Mendeley DOI `10.17632/gk9m674wcx.1`, CC BY 4.0 — no application required). 25 Type 1 diabetes patients on automated insulin delivery, 5-minute Dexcom G6 Pro CGM cadence. Collected at Mayo Clinic Scottsdale, Dec 2023–Apr 2024.

Measured directly from the downloaded files (not trusted from the paper): **306,712 CGM rows**, 28–49 days/patient (avg ~6 weeks). Covariates: glucose, bolus insulin (dose/type/correction), basal rate, carb intake, device mode (sleep/exercise).

**Real data-quality finding:** `EventDateTime` is a per-second event log, not a clean 5-min grid. 823 (patient, timestamp) pairs have >1 row — 429 exact duplicates (silently deduped), 394 genuine conflicts (routed to `cgm-dlq` for human review rather than silently picking one). Real sensor gaps up to ~23 hours observed.

> **Switched from OhioT1DM → AZT1D (2026-07-10).** OhioT1DM needs a gated institutional request (~1 week); the only openly available copy was re-hosted on Kaggle outside that process, which we won't build on. AZT1D is openly licensed, matches the 5-minute cadence exactly (no change to the forecasting framing), and has more patients (25 vs 12). HUPA-UCM was rejected — its 15-minute interval would break the T+30/T+60 task.

---

## Architecture Decisions (locked, all justified)

| Component | Choice | Why this and not the obvious alternative |
|---|---|---|
| **Ingestion transport** | Kafka (KRaft mode, no Zookeeper) | Decouples producer from consumer, buffers on consumer crash, demonstrates offset management. KRaft = one less moving part. |
| **Real-time consumer** | Plain Python, **not Spark** | 1 msg / 5 min — Spark's JVM overhead buys nothing on the real-time path. |
| **Storage** | TimescaleDB | `time_bucket()`, continuous aggregates, native Grafana integration. |
| **Batch features** | PySpark | Feature engineering across full patient history is a legitimately distributed workload (even if demo-scale is single-node). |
| **Orchestration** | Airflow | Retry logic, SLA monitoring, dependency gating — not replaceable by cron. |
| **Model** | Temporal Fusion Transformer (PyTorch) | AZT1D's covariates map cleanly onto TFT's known-future and past-observed input channels. |
| **Serving** | ONNX + FastAPI | Portable artifact served via REST; decouples inference from the training runtime. |
| **Observability** | Grafana | Pipeline health + model drift on one surface. |

### Key implementation decisions made during the build

- **Consumer is stateless / raw-only.** No delta or rolling stats computed in the consumer — deferred entirely to Phase 3's PySpark job, so the consumer never holds fragile per-patient in-process state that's lost on restart.
- **At-least-once made safe.** Offsets commit only after a successful DB write (or confirmed produce to an error topic), per-message. Duplicates are absorbed by `PRIMARY KEY (patient_id, time)` + `ON CONFLICT DO NOTHING`.
- **DLQ split into 3 topics by failure class:** `cgm-parse-errors` (structural/malformed → alerting), `cgm-dlq` (conflicting same-timestamp groups → human review), `cgm-implausible` (CGM <40 or >400 mg/dL, matching the Dexcom G6's own documented reporting range → archival/monitoring).
- **Two time columns on `cgm_readings`:** `time` (historical sensor timestamp from the CSVs) and `ingested_at` (wall-clock write time) — the live Grafana panel needs wall-clock, not 2023–2024 sensor time.
- **DLQ observability reuses the existing TimescaleDB datasource** (via a `dlq_events` log table) rather than adding Prometheus/JMX — stays inside the locked stack.
- **Feature strategy:** resample to 5-min grid then roll; delta + rolling mean/std (15/60 min) + time-since-last-bolus/carb with explicit `has_prior_*` cold-start flags (NaN, not a sentinel). Full-history recompute every run — a documented scaling tradeoff, justified at N=3 patients rather than building incremental/lookback logic prematurely.
- **Quality gate:** per-patient-week, 180-min single-gap threshold + <85% expected-readings threshold (prorated for partial boundary weeks), both derived from the real measured gap distribution. Exclude-and-log locally; a circuit breaker on the run's overall exclusion rate (50% placeholder) is the only thing allowed to hard-fail the DAG.
- **Airflow + Spark share one container** (custom image on `apache/airflow:2.9.3-python3.11` + JDK + PySpark + Postgres JDBC). Avoided docker-in-docker / DockerOperator for a demo-scale single-node job. Airflow's metadata DB is a second database on the existing TimescaleDB instance, not a new service.
- **Host port remap:** TimescaleDB exposed on host `5544` (container-internal `5432` unchanged) to avoid collision with two native Postgres installs (v17 on 5432, v18 on 5433).

---

## What's been done

### ✅ Phase 1 — Foundation (verified 2026-07-10)
Full stack runs via one `docker compose up -d`, health-check gated so nothing races Kafka's readiness.
- Kafka (`apache/kafka:3.7.0`), TimescaleDB (`timescale/timescaledb:2.16.1-pg16`), Grafana (`grafana/grafana:11.2.0`), one-shot `kafka-init` for topic creation.
- Thin `cgm_readings` hypertable (schema only — feature columns deferred by design).
- Grafana TimescaleDB datasource provisioned as code.
- **Verified:** all services healthy on one command; topics + hypertable survive a full `down`/`up` cycle; Grafana queries the datasource live (`/api/ds/query` → 200).

### ✅ Phase 2 — Ingestion (verified 2026-07-13, done-gate MET)
- `producer/replay_sensor.py` — reads a patient CSV, collapses exact duplicates, routes conflicts to `cgm-dlq`, streams to `cgm-raw`. `bulk` and `live` modes.
- `consumer/ingest.py` — stateless raw-only writer with the commit-after-write / 3-topic-DLQ design above.
- Schema migrations for covariate columns, `dlq_events` log table, and `ingested_at`.
- 3-panel Grafana ingestion dashboard (ingestion rate, per-patient glucose trace, DLQ health by topic).
- **Verified:** drained a real 34,279-message backlog across 3 patients with zero duplicate rows despite an earlier crash-loop; 5 synthetic bad messages all routed to the correct DLQ topic; boundary CGM=40 correctly accepted; each dashboard panel's SQL checked directly against `/api/ds/query`.
- **Bug found + fixed:** consumer logs were empty because Python buffers stdout when piped — fixed with `PYTHONUNBUFFERED=1` in both Dockerfiles.

### ✅ Phase 3 — Batch + Orchestration (verified 2026-07-13, done-gate MET)
- `spark/feature_job.py` — PySpark JDBC read, per-patient resample + rolling features, per-patient-week quality evaluation, upsert to `cgm_features` (`ON CONFLICT DO UPDATE`), writes a `feature_run_summary` row each run.
- `airflow/Dockerfile` + `airflow/dags/feature_pipeline.py` — 2-task DAG: `run_feature_job` (never fails on one bad patient-week) → `quality_gate_check` (the circuit breaker, the only task allowed to fail the DAG).
- `cgm_features` hypertable + `feature_run_summary` table; Airflow metadata DB bootstrap script.
- **Verified:** DAG loads with zero import errors; manual run `success` in ~18s writing 16,157 rows; quality gate excluded 3 of 22 real patient-weeks (Subject 3's 1384-min gap, Subject 1's 600-min and 260-min gaps) confirmed by direct query, not logs; exclusion rate 13.64% correctly under the 50% breaker.

---

## Future tasks

### ⬜ Phase 4 — ML + Serving (not started)
Gated on Phases 1–3, all now met. Done-gate (`BLUEPRINT.md`): trained TFT beats the persistence baseline on held-out patients, exported ONNX model serves predictions via FastAPI, Grafana shows RMSE vs. baseline.

1. **Re-measure the persistence baseline on AZT1D.** The previously cited T+30 RMSE (15–25 mg/dL) was OhioT1DM-specific and does **not** carry over — measure it fresh once AZT1D runs through the pipeline.
2. **Train the TFT** (PyTorch) on `cgm_features`, held out by patient (not by time within a patient), predicting T+30 and T+60.
3. **Evaluate** with RMSE/MAE vs. the persistence baseline + a Clarke error grid.
4. **Export to ONNX** and serve via **FastAPI** (REST inference decoupled from the training runtime).
5. **Grafana RMSE panel** — model performance vs. baseline + drift observability.
6. Add a `model_version` concern at serving time (deliberately kept out of the features table).

### ⬜ Housekeeping / open items
- Confirm the pinned Python version across producer/consumer/Spark.
- `cgm-dlq` has no dedup — re-running the producer for the same patient appends real duplicate conflict messages. Fine today; matters if `cgm-dlq` volume ever drives a count-based alert.
- Circuit-breaker threshold (50%) is an acknowledged placeholder at N=3 patients — revisit when patient count grows.
- Not yet exercised: Subject 14's `Readings (CGM / BGM)` column-alias path, and a patient with populated `device_mode`/bolus/carb fields.

---

## Repository map

```
docker-compose.yml                          # full stack, health-check gated
producer/replay_sensor.py                   # CSV → cgm-raw (+ conflicts → cgm-dlq)
consumer/ingest.py                          # cgm-raw → TimescaleDB (stateless, 3-topic DLQ)
spark/feature_job.py                        # batch feature engineering + quality gate
airflow/dags/feature_pipeline.py            # 2-task orchestration DAG
airflow/Dockerfile                          # Airflow + JDK + PySpark + JDBC in one image
timescaledb/init/                           # schema migrations (000–005)
monitoring/grafana/                         # provisioned datasource + ingestion dashboard
docs/                                       # BLUEPRINT, QUESTIONS, PROBLEMS, PROGRESS, STATUS
data/azt1d/                                 # dataset (gitignored)
```
