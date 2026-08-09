# Progress Log

Chronological record of what's actually been built and verified, as of 2026-07-10. This is a summary for orientation — the authoritative detail lives in `docs/BLUEPRINT.md` (plan/gates), `docs/QUESTIONS.md` (decisions), and `docs/PROBLEMS.md` (known risks).

---

## Planning (before any code)

- Reviewed `README.md` (GlucoPulse architecture) and `CLAUDE.md` (cardinal rules — note: `CLAUDE.md` is actually written for a different project, FlowCast; kept as-is per explicit instruction, not reconciled with GlucoPulse).
- Created `docs/PROBLEMS.md` — anticipated engineering risks per build phase (health-check ordering, OhioT1DM XML quirks, at-least-once delivery/idempotency, ONNX export fragility, small-N generalization, etc.).
- Created `docs/QUESTIONS.md` — running log template for decisions made during the build.
- Created `docs/BLUEPRINT.md` — phase-by-phase execution plan with a concrete done-gate per phase, and Phase 1 broken into specific steps.

## Phase 1 — Foundation (DONE, verified 2026-07-10)

**Goal:** every service in the architecture runs via one `docker compose up -d`, proven healthy, before any pipeline code exists.

**Built:**
- `docker-compose.yml` — Kafka (KRaft mode, `apache/kafka:3.7.0`, no Zookeeper), TimescaleDB (`timescale/timescaledb:2.16.1-pg16`), Grafana (`grafana/grafana:11.2.0`), plus a one-shot `kafka-init` service for topic creation. All service dependencies gated on `condition: service_healthy`, not just container start.
- `timescaledb/init/001_cgm_readings.sql` — deliberately thin schema (`patient_id`, `time`, `glucose_value` only) as a hypertable. Feature columns (delta, rolling stats, covariates) intentionally deferred until Phase 2 defines what the consumer actually computes.
- `monitoring/grafana/provisioning/datasources/timescaledb.yml` — Grafana's TimescaleDB connection provisioned as code (survives `down`/`up`, not clicked in through the UI).
- `.env.example` / `.env` / `.gitignore` — credentials kept out of git.

**Problem hit and resolved:** `docker compose up` failed on port 5432, then 5433 — the dev machine has two native PostgreSQL installs (v17, v18) already bound to both. Fixed by remapping TimescaleDB's host-side port to `5544:5432` (container-internal port unchanged; other containers still reach it at `timescaledb:5432`). Full writeup in `docs/QUESTIONS.md`.

**Verified (not just assumed from config):**
- All four services report healthy / exit 0 on a single `docker compose up -d`.
- Kafka topics `cgm-raw` and `cgm-dlq` exist (`kafka-topics.sh --list`).
- `cgm_readings` confirmed as a real hypertable via `timescaledb_information.hypertables`.
- Grafana's provisioned datasource successfully queries TimescaleDB (`POST /api/ds/query` → `200`, count `0` as expected — no data yet).
- Full `docker compose down && docker compose up -d` cycle — topics and hypertable both survived, confirming the restart-durability gate.

**Docs updated to reflect this:** `README.md` (Phase 1 status → Complete, setup instructions note the 5544 port), `docs/BLUEPRINT.md` (Phase 1 gate marked met with evidence).

## Dataset switch: OhioT1DM → AZT1D (2026-07-10)

OhioT1DM requires a gated institutional request (~1 week). A single OhioT1DM file was found re-hosted on Kaggle, outside that access process — using it would mean building the ingestion story on data of provenance we couldn't honestly stand behind. Switched to **AZT1D** (arXiv:2506.14789, Mendeley DOI `10.17632/gk9m674wcx.1`, CC BY 4.0, no application needed): 25 patients, same 5-minute Dexcom cadence as OhioT1DM (no change to the T+30/T+60 forecasting framing), 320,488 total CGM readings (more than OhioT1DM's ~193K despite a shorter ~26-day/patient window). HUPA-UCM was considered and rejected — its 15-minute interval would have broken the existing forecasting task design. Full decision trail in `docs/QUESTIONS.md`. Updated: `README.md`, `docs/PROBLEMS.md`, `docs/BLUEPRINT.md`, `.gitignore`, memory.

---

## Phase 2 — Producer verified (2026-07-10)

**Built:** `producer/replay_sensor.py` — reads a patient's AZT1D CSV, groups rows by `EventDateTime`, collapses exact duplicates, routes genuinely conflicting groups to `cgm-dlq` (per the decision in `docs/QUESTIONS.md`), and streams the rest to `cgm-raw` via `confluent_kafka.Producer`. Two modes: `bulk` (no delay, full history) and `live` (5s delay, capped by `--hours`). Runs as a one-off via the `tools` compose profile (`docker compose --profile tools run --rm producer ...`), not part of `up -d`.

**Verified (not just assumed from reading the code):**
- `bulk` mode, Subject 1: producer logged "Sent 10866 messages to cgm-raw, 15 messages to cgm-dlq" — reading both topics back from the broker (`kafka-console-consumer.sh --from-beginning`, counted independently) confirmed the exact same counts. No silent drops between producer and broker.
- Sampled message content off the real topic (not the producer's own stdout): `cgm-raw` messages are correctly shaped; `cgm-dlq` samples are genuine conflicts (e.g. two different CGM readings, `122` vs `115`, at the identical second) — confirms the DLQ routing decision actually fires on real data, not just in theory.
- `live` mode, Subject 1, `--hours 0.5`: correctly filtered to 7 timestamps (5-min cadence within a 30-min window), sent with the real 5s per-message delay, and the broker's `cgm-raw` count rose from 10,866 → 10,873 — exactly 7 more.

**Not yet tested:** Subject 14's `Readings (CGM / BGM)` column-alias path, a patient with populated `device_mode`/bolus/carb fields (Subject 1's early records are mostly null on those), and the consumer side doesn't exist yet — messages are sitting in Kafka unread.

## Phase 2 — Consumer built and verified end-to-end (2026-07-13)

**Decisions made (full writeup in `docs/QUESTIONS.md`, 2026-07-13 entry):**
- Consumer scope is raw-only/stateless — no delta/rolling stats. Deferred entirely to Phase 3's PySpark batch job, so the consumer never needs in-process per-patient state that's fragile on restart.
- Offset commits happen only after a successful outcome (DB write, or confirmed produce to an error topic), at per-message granularity — at-least-once, made safe by the existing `PRIMARY KEY (patient_id, time)` via `ON CONFLICT DO NOTHING`.
- DLQ policy fully resolved into three topics by failure class: `cgm-parse-errors` (structural, alerting), `cgm-dlq` (conflicting timestamps, unchanged, human review), `cgm-implausible` (CGM <40 or >400 mg/dL — matches the Dexcom G6's own reporting range, archival/monitoring not urgent alerting).

**Built:** `consumer/ingest.py` (+ `consumer/Dockerfile`, `consumer/requirements.txt`, `timescaledb/init/002_add_covariates.sql` adding the raw covariate columns, new `consumer` service in `docker-compose.yml` running as part of `up -d`, not the `tools` profile).

**Verified (not just assumed from reading the code):**
- Drained a real backlog of 34,279 messages across 3 patients (subjects 1, 3, 14) sitting in `cgm-raw` from earlier producer runs — consumer group lag is 0 on all 3 partitions, and `count(*) == count(DISTINCT time)` per patient in `cgm_readings`, confirming no duplicate rows despite the consumer crash-looping earlier in the session (before the schema migration was applied) — those failed attempts never committed offsets, so nothing double-wrote once fixed.
- Injected 5 synthetic messages directly onto `cgm-raw`: invalid JSON and a message missing `cgm` both landed in `cgm-parse-errors` with the correct reason string; CGM values of 900 and 5 both landed in `cgm-implausible` and were correctly excluded from `cgm_readings`; a boundary value of exactly 40 (inclusive) was correctly accepted and written.
- Confirmed via `docker compose ps` that all 4 services (kafka, timescaledb, grafana, consumer) are healthy/up together, consumer under `restart: unless-stopped`.

## Phase 2 — Grafana ingestion dashboard built and verified (2026-07-13), Phase 2 done-gate MET

**Design decision:** DLQ health observability reuses the existing TimescaleDB/Postgres Grafana datasource rather than adding new infrastructure (Prometheus, Kafka JMX exporter) just for topic-level metrics. The consumer now also subscribes to `cgm-dlq` (in addition to `cgm-raw`) purely to log an observability row into a new `dlq_events` table for each of the three failure classes -- it does not touch conflicting-timestamp resolution, which stays human-review-only.

**Built:**
- `timescaledb/init/003_dlq_events.sql` -- plain (non-hypertable) `dlq_events` log table.
- `timescaledb/init/004_add_ingested_at.sql` -- added `ingested_at` (wall-clock write time) to `cgm_readings`, since the existing `time` column is the sensor's own historical event timestamp (Dec 2023-Apr 2024) and is the wrong column for a live/real-time panel against Grafana's default now-relative time picker.
- `consumer/ingest.py` extended to log every parse-error, implausible-value, and cgm-dlq event into `dlq_events`.
- `monitoring/grafana/dashboards/ingestion.json` + `monitoring/grafana/provisioning/dashboards/dashboards.yml` -- 3-panel dashboard: (1) ingestion rate rows/min by `ingested_at`, (2) per-patient glucose trace templated on `$patient`, deliberately ignoring the dashboard time-range picker (LIMIT-based latest-500 query) since patient data is historical, (3) DLQ health events/min split by topic.
- Datasource provisioning given an explicit `uid: timescaledb` so dashboard JSON can reference it reliably.

**Bug found and fixed:** the consumer's `docker logs` showed nothing even after processing thousands of messages -- Python buffers stdout when piped (not a TTY), so a long-running process's `print()` never flushed. Added `PYTHONUNBUFFERED=1` to both `consumer/Dockerfile` and `producer/Dockerfile`.

**Verified (not just assumed from reading the JSON):**
- `GET /api/search` and `/api/dashboards/uid/glucopulse-ingestion` confirm the dashboard is actually provisioned with all 3 panels.
- Each panel's exact SQL was run through `/api/ds/query` directly: panel 1 (ingestion rate) returned real time-bucketed counts; panel 2 (patient trace) returned exactly 500 rows for patient 1 with real glucose values; panel 3 (DLQ health) correctly split into 3 series (`cgm-dlq`, `cgm-implausible`, `cgm-parse-errors`), confirmed by inspecting the raw Grafana dataframe schema, not just a visual check.
- Injected fresh synthetic messages across all 3 failure classes (including directly onto `cgm-dlq`, not just `cgm-raw`) after the rebuild; each produced exactly one new `dlq_events` row with the correct topic/patient_id/reason.
- **Observation (not a bug from today's work):** 15 `cgm-dlq` entries for patient 1 are genuine duplicates in the underlying Kafka topic, from the producer having been run for patient 1 in two separate sessions -- `cgm-dlq` has no dedup the way `cgm_readings` does via its primary key, so re-running the producer for the same patient does append real duplicate conflict messages. Not addressed now; worth knowing if `cgm-dlq` volume ever gets used for a real count-based alert.

**Phase 2 done-gate (docs/BLUEPRINT.md) is now fully met**: real patient replay lands in TimescaleDB, DLQ demonstrably catches deliberately bad messages (all 3 classes, not just one), and Grafana shows live ingestion.

## Phase 3 — PySpark feature job + Airflow orchestration built and verified (2026-07-13), Phase 3 done-gate MET

**Decisions made (full writeup in `docs/QUESTIONS.md`, 2026-07-13 Phase 3 entry):** resample-to-5-min-grid then roll; delta + rolling mean/std (15min, 60min) + time-since-last-bolus/carb with explicit `has_prior_*` cold-start flags (NaN, not a sentinel); full-history recompute every run (scaling tradeoff documented in `docs/PROBLEMS.md` rather than building unneeded incremental/lookback logic); features table keyed `(patient_id, time)`, no `model_version` (that's Phase 4's concern, not a feature-engineering one); per-patient-week quality gate (180-min single-gap threshold, <85% expected-readings count threshold, prorated for partial boundary weeks); exclude-and-log locally, with a circuit breaker on the run's overall exclusion rate as the only thing allowed to hard-fail the DAG (placeholder threshold at 50%, documented as arbitrary at N=3 patients).

**Built:**
- `timescaledb/init/005_cgm_features.sql` — `cgm_features` hypertable + `feature_run_summary` log table.
- `spark/feature_job.py` — PySpark batch job: JDBC read of `cgm_readings`, per-patient resample + rolling features, per-patient-week quality evaluation, upsert to `cgm_features` (`ON CONFLICT DO UPDATE`, not `DO NOTHING`, since a recompute can legitimately change values), writes a `feature_run_summary` row every run.
- `airflow/Dockerfile` — custom image extending `apache/airflow:2.9.3-python3.11` with a JDK, PySpark, and the Postgres JDBC driver, running Airflow and Spark in the same container (avoided docker-in-docker/DockerOperator, since a demo-scale single-node Spark job doesn't need it).
- `airflow/dags/feature_pipeline.py` — 2-task DAG: `run_feature_job` (never raises on a single bad patient-week) → `quality_gate_check` (the circuit breaker, reads `feature_run_summary` and is the only task allowed to fail the DAG).
- `timescaledb/init/000_create_airflow_db.sh` — Airflow's metadata store lives in its own database on the same TimescaleDB Postgres instance, not a separate Postgres service (not in the locked stack).
- New `airflow` service in `docker-compose.yml` (`airflow standalone`, LocalExecutor, webserver on :8080).

**Verified (not just assumed from reading the code):**
- Custom image builds and the DAG loads with zero import errors (`airflow dags list-import-errors` → "No data found") — confirms Java + PySpark + JDBC driver + psycopg2 are all correctly wired in one image.
- Triggered `feature_pipeline` manually: `state=success` in ~18s, wrote 16,157 rows to `cgm_features`.
- **Quality gate fired on real data already in TimescaleDB, with no synthetic injection needed**: 3 of 22 patient-weeks excluded — Subject 3's known 1384-min gap, Subject 1's 600-min gap, and a 260-min gap for Subject 1 not previously called out but correctly above the 180-min threshold. Confirmed by querying `cgm_features` directly for the excluded week (zero rows), not by trusting the task log.
- Exclusion rate 13.64%, correctly under the 50% circuit-breaker placeholder — DAG succeeded rather than hard-failing, confirming exclude-locally/escalate-globally behaves as designed at a real, current exclusion rate.
- Spot-checked feature rows for physiological sanity (delta, rolling means, `time_since_last_bolus_min` all populated and consistent with the raw series).

**Phase 3 done-gate (`docs/BLUEPRINT.md`) is now fully met.**

## Not yet started

- **Phase 4 — ML + Serving:** not started, depends on Phase 3 completing first per the phase-gating rule in `docs/BLUEPRINT.md`.
