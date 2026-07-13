# Anticipated Problems

Real engineering problems we expect to hit, organized by build phase. This list gets updated as we actually hit things — if a problem here turns out not to be real, cross it out with a note; if we hit something not listed here, add it.

---

## Phase 1 — Foundation (Docker Compose)

- **Service startup ordering.** Kafka/TimescaleDB/Grafana won't all be ready at the same moment `docker compose up` returns. Consumer will crash-loop on startup if it connects before Kafka's controller is ready. Needs health checks + `depends_on: condition: service_healthy`, not just `depends_on`.
- **Resource load.** Kafka + TimescaleDB + Grafana + (later) Spark + Airflow running simultaneously on a laptop is heavy. Watch memory before adding Spark/Airflow in Phase 3 — may need to stop Phase 1/2 services or tune JVM heap sizes.
- **Volume persistence.** Named volumes for Kafka logs and TimescaleDB data need to survive `docker compose down` (but not `down -v`) or every restart loses ingested data — annoying during dev, silently corrupting during a demo.
- **Port collisions.** Kafka (9092), TimescaleDB (5432), Grafana (3000) — 5432 in particular collides with any local Postgres already running.

## Phase 2 — Ingestion

- **`EventDateTime` is a per-second event log, not a clean 5-minute grid — CONFIRMED, measured.** Inspecting all 25 real CSVs found 823 (patient, timestamp) pairs where more than one row shares the exact same second: 429 are pure exact duplicates (silently dedupe — 1,705 redundant rows), but 394 have genuinely conflicting values (e.g. Subject 15 at `2024-01-21 14:06:09` has CGM `226` in one row, `239` in another). Decision made: conflicting groups route to `cgm-dlq`, not silently resolved. See `docs/QUESTIONS.md`.
- **Real sensor gaps up to 23 hours — CONFIRMED, measured.** Subject 3 has a gap of 1,384 minutes; several other subjects have gaps over 3+ hours. This is the exact "undetected sensor gap" scenario from README's opening problem statement — the consumer/rolling-stats logic needs to detect and not silently smooth over a gap this large.
- **`DeviceMode` is sparse and asymmetric.** Most rows have it blank (implicitly "regular" — there's no explicit "regular" value, just empty string); only `sleep` and `exercise` appear explicitly, and some subjects (e.g. Subject 1, Subject 11) never log a mode at all. Don't assume every subject's file has mode data.
- **`BolusType` values are messier than a clean enum.** Real values include e.g. `Extended 50.00%/23.75`, `Extended/Correction 70.00%/23.75` — percentage/duration figures embedded directly in the string. Treat as opaque categorical for now; parsing the embedded numbers out is a separate, not-yet-justified feature.
- **Replay timing vs. event correlation.** Replaying at 5s/reading (speedup of 60x) is easy for glucose alone, but bolus/carb events don't occur on the 5-minute CGM grid — need to preserve their real relative offset from the nearest glucose reading, not just replay every stream independently.
- **At-least-once delivery → duplicate writes.** Kafka's at-least-once semantics mean the consumer can process the same message twice after a rebalance or restart. TimescaleDB writes need to be idempotent (upsert on patient_id + timestamp), or duplicates silently double-count in rolling stats.
- **Dead letter queue policy — DECIDED (2026-07-13).** Three separate topics by failure class: `cgm-parse-errors` (structural/malformed), `cgm-dlq` (conflicting same-timestamp groups, unchanged), `cgm-implausible` (CGM <40 or >400 mg/dL, matching the Dexcom G6's own reporting range). See `docs/QUESTIONS.md`.
- **Stateful rolling features without a stream processing framework — AVOIDED, not solved.** Resolved by scoping this risk out of the consumer entirely (2026-07-13, see `docs/QUESTIONS.md`): `ingest.py` writes raw values only and holds no per-patient state, so there's nothing to lose on restart. Delta/rolling stats are deferred to Phase 3's PySpark batch job instead.
- **Partitioning vs. ordering.** Kafka partitioning by patient_id gives parallelism but only guarantees ordering within a partition — need to confirm a single patient's readings never span partitions, or rolling features break.

## Phase 3 — Batch + Orchestration

- **Backfill / idempotent reruns.** Airflow triggers weekly, but a rerun (manual backfill, retry after failure) must not double-append features into TimescaleDB or corrupt the training set.
- **Data quality gate definitions.** "Gate the DAG on data quality" is a nice sentence in the README — the actual checks (max allowed gap length, sensor error codes, min readings per patient per week) need to be decided and justified, not defaulted to arbitrary thresholds (see CLAUDE.md-style rule: no invented thresholds without a reason).
- **PySpark in Docker isn't a real cluster.** Local single-node PySpark in a container gives none of the distributed benefits the README claims Spark is "for" — worth being honest in interview talking points that this demonstrates the API/pattern, not real distributed scale.
- **Cross-patient normalization leakage.** Normalizing across all 25 patients' full history risks leaking test-period statistics into training if not split by time first, patient second.

## Phase 4 — ML + Serving

- **Sparse covariate alignment.** Meal/insulin events are sparse relative to the 5-minute glucose grid; TFT's known-future/past-observed inputs need explicit alignment/imputation logic, not just a join.
- **Multi-horizon probabilistic evaluation.** Quantile loss and calibration checks for T+30/T+60 are more involved than a single RMSE number — eval code needs to actually compute and justify these, not just report point-forecast RMSE.
- **ONNX export of TFT.** Attention layers and any custom PyTorch ops in TFT implementations are historically fragile to export cleanly — expect this to be the single most likely place Phase 4 stalls.
- **INT8 quantization accuracy loss.** Needs a measured before/after RMSE comparison, not an assumption that quantization is "free" — if it degrades accuracy meaningfully, that's a real tradeoff to document, not skip.
- **CLARK error grid correctness.** No standard library implements this — the zone boundaries need to be implemented and verified against a known reference before being trusted in any dashboard or claim.

## Cross-cutting

- **Small-N generalization.** 25 patients is a small cohort — real risk of overfitting to patient-specific glucose dynamics, compounded by AZT1D's shorter per-patient window (~26 days avg vs. what a gated academic dataset like OhioT1DM offers at ~8 weeks). Needs an honest per-patient vs. pooled evaluation, not just an aggregate RMSE.
- **venv discipline inside containers.** All Python must run in a venv per project rule — inside Docker this means the image itself needs its own isolated environment setup, not global pip installs in the container either.
