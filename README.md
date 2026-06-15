# GlucoPulse

Real-time streaming pipeline for continuous glucose monitor (CGM) data. Built to answer one question a production data engineer faces daily: **how do you move sensor data reliably from source to storage to model, and know when something breaks before your users do?**

---

## The Engineering Problem

Continuous glucose monitors emit one reading every 5 minutes per patient. The interesting problem is not throughput — it is reliability. A silently corrupted reading, a dropped Kafka message, or an undetected sensor gap can cause a downstream forecasting model to produce wrong predictions without any visible error.

GlucoPulse is designed around that constraint: fault-tolerant ingestion, orchestrated batch processing, and operational observability at every stage.

---

## Architecture

```
OhioT1DM XML (dataset replay)
         |
 Python Replay Producer
 (simulates sensor at 5s/reading)
         |
       Kafka
 (topic: cgm-raw | dead letter: cgm-dlq)
         |
 Python Kafka Consumer ──────────────→ TimescaleDB
 (real-time feature compute:            (hypertable partitioned
  glucose delta, rolling stats,          by patient + timestamp)
  covariate joins)                              |
                                           Grafana
                                    (live ingestion + pipeline health)

TimescaleDB
         | (Airflow triggers weekly)
   PySpark Batch Job
   (lag features, window aggregates,
    cross-patient normalization)
         |
   TFT Model Training (PyTorch)
   (known-future: meal/insulin covariates
    past-observed: glucose history)
         |
   ONNX Export + INT8 Quantization
         |
   FastAPI Inference Endpoint
         |
      Grafana
   (RMSE vs persistence baseline,
    CLARK error grid distribution)
```

---

## Stack Decisions

Every component is justified by a specific requirement.

| Component | Requirement it satisfies |
|---|---|
| **Kafka** | Decouples sensor replay from processing; buffers data if consumer crashes; enables at-least-once delivery and offset management |
| **Python Consumer** (not Spark Streaming) | Real-time path handles 1 msg/5min — Spark adds JVM overhead with zero throughput benefit at this scale |
| **TimescaleDB** | `time_bucket()` and continuous aggregates for rolling glucose windows; hypertable partitioning for time-range queries; native Grafana data source |
| **PySpark** | Batch feature engineering across all patients' full history is a legitimate distributed workload — runs on training path, not hot path |
| **Airflow** | Retry logic, SLA alerting, and data-quality-gated job dependencies — none of which exist in cron |
| **TFT (PyTorch)** | OhioT1DM includes meal/insulin covariates that feed TFT's known-future and past-observed input channels; designed for multi-horizon probabilistic forecasting |
| **ONNX + FastAPI** | Portable model artifact served via REST; decouples inference runtime from PyTorch training environment |
| **Grafana** | A pipeline without monitoring does not exist in production |

---

## Data Source

**OhioT1DM Dataset** — 12 Type 1 diabetes patients, ~8 weeks each, 5-minute CGM intervals.

Covariates: CGM glucose (mg/dL), meal events + carbohydrate amounts, insulin bolus doses, basal insulin rates, exercise events, fingerstick readings.

Access requires a brief application: http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html

---

## Forecasting Task

Predict glucose value at **T+30 minutes** and **T+60 minutes**.

**Baseline:** Persistence model — predict the last known value. At T+30, this achieves ~15–25 mg/dL RMSE on OhioT1DM. Every model is evaluated against this baseline first.

**Evaluation:** RMSE, MAE, and CLARK error grid (Zone A = clinically accurate, Zone E = dangerous).

---

## Project Structure

```
glucopulse/
├── docker-compose.yml
├── producer/
│   ├── Dockerfile
│   └── replay_sensor.py
├── consumer/
│   ├── Dockerfile
│   └── ingest.py
├── spark/
│   └── feature_engineering.py
├── dags/
│   └── glucopulse_dag.py
├── model/
│   ├── train.py
│   ├── evaluate.py
│   └── export_onnx.py
├── serving/
│   ├── Dockerfile
│   └── api.py
├── monitoring/
│   └── grafana/
│       └── dashboards/
├── data/
│   └── ohiot1dm/          # gitignored — apply for access above
├── notebooks/
│   └── eda.ipynb
└── requirements.txt
```

---

## Prerequisites

- Docker Desktop
- Python 3.10+
- OhioT1DM dataset (apply for access above)

---

## Setup

```bash
git clone <repo-url>
cd glucopulse

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

docker compose up -d
```

---

## Build Phases

| Phase | Scope | Status |
|---|---|---|
| 1 — Foundation | Docker Compose: all services running with one command | In progress |
| 2 — Ingestion | Producer → Kafka → Consumer → TimescaleDB + dead letter queue + Grafana | Planned |
| 3 — Batch + Orchestration | PySpark feature job + Airflow DAGs + data quality gates | Planned |
| 4 — ML + Serving | TFT training → ONNX export → FastAPI + Grafana RMSE panel | Planned |

---

## Interview Talking Points

**On Kafka over a direct database write:**
Kafka decouples the sensor simulation from processing. If the consumer restarts, Kafka buffers messages and replay resumes from the last committed offset — no data loss.

**On Python consumer over Spark Streaming:**
CGM data is 1 message per 5 minutes per patient. Spark Streaming adds a JVM cluster with zero throughput benefit at this scale. The batch training path uses PySpark where the distributed workload is actually justified.

**On Airflow over cron:**
Cron has no retry logic, no dependency management, and no SLA alerting. A cron job that silently fails at 3am corrupts your training data without any signal. Airflow makes failures loud.

**On the model as a pipeline validator:**
If prediction RMSE degrades, it signals either a model problem or a data quality problem upstream. Grafana surfaces which one it is.
