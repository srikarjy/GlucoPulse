# GlucoPulse

**Streaming CGM pipeline → glucose forecasting → FHIR-native clinical decision support.**

GlucoPulse moves continuous glucose monitor data from sensor to forecast reliably, serves the forecasts through a standards-based healthcare API, and puts a citation-grounded AI agent with a human reviewer on top of it.

**Live demo:** [srikarjy025-glucopulse.hf.space](https://srikarjy025-glucopulse.hf.space) · [Vercel mirror](https://glucopulse-inference.vercel.app). This is the forecasting API: `POST /predict`, Swagger at `/docs`.

[![glucopulse-ci](https://github.com/srikarjy/GlucoPulse/actions/workflows/glucopulse-ci.yml/badge.svg)](https://github.com/srikarjy/GlucoPulse/actions/workflows/glucopulse-ci.yml) [![priorauth-ci](https://github.com/srikarjy/GlucoPulse/actions/workflows/priorauth-ci.yml/badge.svg)](https://github.com/srikarjy/GlucoPulse/actions/workflows/priorauth-ci.yml) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Kafka](https://img.shields.io/badge/Kafka-streaming-black) ![PyTorch](https://img.shields.io/badge/PyTorch-TFT-orange) ![ONNX](https://img.shields.io/badge/ONNX-INT8-lightgrey) ![FHIR](https://img.shields.io/badge/FHIR-R4-red) ![License](https://img.shields.io/badge/license-MIT-green)

---

## Highlights

| | |
|---|---|
| **Forecasting** | Temporal Fusion Transformer beats the persistence baseline on **fully held-out patients**: T+60 RMSE **37.65 vs 42.10 mg/dL** (−11%). At T+30, **85.5%** of forecasts land in Clarke Zone A and **0%** in the dangerous zones C/E. |
| **Data reliability** | Kafka ingestion with a dead-letter queue. Exact duplicates are deduplicated, and **394 conflicting sensor readings** are quarantined instead of being resolved silently. |
| **Serving** | ONNX + INT8 model with a PyTorch fallback for short history. It runs standalone from a single model file, with no database or Kafka, on a free-tier container. |
| **Interoperability** | FHIR R4 facade (`Observation`, `Patient`, `$forecast`, `Provenance`, `AuditEvent`), SMART-scoped JWT auth that fails closed, and a hash-chained audit log. Round-trip tested against HAPI FHIR 7.4.0. |
| **Clinical AI agent** | Drafts cited findings from FHIR data. Any sentence without a matching citation is dropped, and every draft goes to a human reviewer. The agent has no permission to approve its own output. |

---

## Architecture

```
 AZT1D CSV ──► Replay producer ──► Kafka (cgm-raw / cgm-dlq) ──► Consumer ──► TimescaleDB ──► Grafana
                                                                                  │
                                                       Airflow (weekly) ──► PySpark features
                                                                                  │
                                                                     TFT training (PyTorch)
                                                                                  │
                                                                     ONNX export + INT8
                                                                                  │
                                         ┌────────────── FastAPI: /predict + FHIR R4 facade ──────────────┐
                                         │                                                               │
                                   EHR round trip                                       Clinical Reasoning Agent
                                  (ehr/sync.py)                                (MCP tools → cited draft → human review)
```

---

## Results

Patient-level split: 18 train / 3 validation / 4 test patients, seed 42. The test patients are never seen during training.

| Horizon | Model | RMSE (mg/dL) | MAE (mg/dL) |
|---|---|---|---|
| T+30 | Persistence baseline | 25.80 | 18.88 |
| T+30 | **TFT** | **25.47** | **17.61** |
| T+60 | Persistence baseline | 42.10 | 31.69 |
| T+60 | **TFT** | **37.65** | **27.48** |

- **Clarke error grid (T+30):** 85.5% A · 13.2% B · 1.3% D · 0% C/E
- **Calibration:** 80% prediction intervals cover 76.6% (T+30) and 69.4% (T+60). They are under-confident at longer horizons, and this is the next thing to improve.
- Source: `model/results/baseline_metrics.json`, `model/results/tft_eval.json`

**Clinical agent eval** (20 synthetic patients, scripted model): citation precision 1.0 (textual and resource-grounded), abstention accuracy 1.0, quarantine accuracy 1.0. Every FHIR citation is checked against the resources the tools actually returned, so a model that invents a reference has its sentence dropped and the finding forced into abstention. These numbers test the guardrails, not live LLM judgment. Details are in [`docs/CLINICAL_AGENT.md`](docs/CLINICAL_AGENT.md).

---

## Key Design Decisions

| Decision | Why |
|---|---|
| **Kafka** between replay and storage | The consumer can crash without losing data. Replay resumes from the committed offset. |
| **Python consumer**, not Spark Streaming | One message per 5 minutes per patient is far too little traffic to justify a JVM cluster. |
| **PySpark** only on the batch path | Feature engineering over every patient's full history is a real distributed workload. The hot path isn't. |
| **Airflow** over cron | Retries, SLAs and data-quality-gated dependencies. A silent 3 a.m. failure corrupts training data. |
| **TimescaleDB** | `time_bucket()` and continuous aggregates for rolling glucose windows, and Grafana reads it natively. |
| **TFT** | AZT1D has insulin and carb covariates, which map directly onto TFT's known-future and past-observed inputs. |
| **ONNX + INT8** | Separates the inference runtime from the training stack, so the API ships as one small artifact. |
| **Patient-level split** | Tests generalization to *new patients*, the realistic deployment case, not just to unseen weeks. |

---

## Dataset

**[AZT1D](https://data.mendeley.com/datasets/gk9m674wcx/1)** (CC BY 4.0, arXiv:2506.14789): 25 Type 1 diabetes patients on automated insulin delivery, about 6 weeks each, Dexcom G6 at 5-minute intervals. The downloaded files contain **306,712 CGM readings**, measured from the files themselves (the paper cites 320,488). Covariates include bolus and basal insulin, carbs, and device mode.

Real data-quality problems handled in the pipeline: duplicate timestamps, conflicting readings at the same second, and sensor gaps of up to 23 hours. See [`docs/QUESTIONS.md`](docs/QUESTIONS.md).

---

## Tech Stack & Dependencies

Each service pins its own dependencies in its own `requirements.txt`. Docker Compose installs them inside each service's container, so you don't need a single global install.

### Prerequisites

| Tool | Version | Used for |
|---|---|---|
| Docker + Docker Compose | 24+ (Docker Desktop, OrbStack, or Colima) | Running the full pipeline |
| Python | 3.12 (3.11 inside the Airflow image) | Local dev, tests, model training |
| Node.js | 22 | `priorauth-ui/` only |
| AZT1D dataset | [Mendeley DOI 10.17632/gk9m674wcx.1](https://data.mendeley.com/datasets/gk9m674wcx/1) | Training and replay data |

### Infrastructure (`docker-compose.yml`)

| Service | Image | Role |
|---|---|---|
| Kafka | `apache/kafka:3.7.0` (KRaft, no ZooKeeper) | `cgm-raw` stream + `cgm-dlq` dead-letter topic |
| TimescaleDB | `timescale/timescaledb:2.16.1-pg16` | Time-series storage (host port **5544**) |
| Airflow | `apache/airflow:2.9.3-python3.11` | Weekly PySpark feature job |
| Grafana | `grafana/grafana:11.2.0` | Ingestion health + model RMSE dashboards |
| HAPI FHIR | `hapiproject/hapi:v7.4.0` | Local EHR for the FHIR round-trip test |
| Caddy | `caddy:2.8.4` | TLS termination in front of the serving API |

### Python packages by component

| Component | Manifest | Key packages (pinned) |
|---|---|---|
| Producer | `producer/requirements.txt` | `confluent-kafka 2.15.0` |
| Consumer | `consumer/requirements.txt` | `confluent-kafka 2.15.0`, `psycopg2-binary 2.9.10` |
| Batch features | `airflow/requirements.txt` | `pyspark 3.5.1`, `psycopg2-binary 2.9.10` |
| Model training | `model/requirements.txt` | `torch 2.4.1`, `pytorch-lightning 2.4.0`, `pytorch-forecasting 1.1.1`, `scikit-learn 1.5.2`, `pandas 2.2.3`, `numpy 1.26.4`, `onnx 1.17.0`, `onnxruntime 1.19.2` |
| Serving + FHIR | `serving/requirements.txt` | `fastapi 0.115.4`, `uvicorn 0.32.0`, `onnxruntime 1.19.2`, `torch 2.4.1`, `pytorch-forecasting 1.1.1`, `pyjwt[crypto] 2.9.0`, `psycopg2-binary 2.9.9` |
| Clinical agent | `agents/requirements.txt` + `priorauth/requirements.txt` | `strands-agents 1.59.0`, `mcp ≥1.23`, `anthropic`, `openai` (only for the optional live demo) |
| Prior-auth platform | `priorauth/requirements.txt` | `fastapi 0.143.0`, `sqlalchemy 2.1.4`, `psycopg 3.3.6`, `pgvector 0.5.1`, `pydantic 2.14.1`, `httpx 0.28.1`, `opentelemetry-sdk 1.45.1`, `prometheus-client 0.26.0`, `boto3`, `opensearch-py` |
| Prior-auth fine-tuning | `priorauth/finetune/requirements-train.txt` | `transformers`, `peft`, `trl`, `bitsandbytes`, `accelerate`, `datasets` (GPU, optional) |
| Hosted demos | `hf_space/requirements.txt`, `hf_space_agent/requirements.txt` | Smaller serving-only sets |
| Dev / test | `requirements-ci.txt`, `priorauth/requirements-dev.txt` | `pytest ≥8`, `ruff ≥0.6`, `httpx`, `fhir.resources 8.3.0` (validates FHIR output in `tests/healthcare`) |

### Frontend (`priorauth-ui/package.json`)

`next ^16.4.6`, `react 19.3.0`, `tailwindcss 4`, `typescript 5`, plus `@playwright/test` and `@axe-core/playwright` for end-to-end and accessibility tests.

---

## Quickstart

**1. Run the full pipeline (Docker)**

```bash
git clone https://github.com/srikarjy/GlucoPulse.git && cd GlucoPulse
cp .env.example .env                     # set credentials
# put the AZT1D CSVs in data/azt1d/ (the folder is gitignored)
docker compose up -d                     # Kafka, TimescaleDB, Airflow, Grafana, producer, consumer, serving
```

**2. Local dev + healthcare tests**

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-ci.txt          # minimal pinned test deps (no torch)
pytest tests/healthcare -q               # FHIR facade, SMART auth, audit log (50 tests)
```

**3. Clinical agent tests** (separate venv, because it needs a newer FastAPI and pydantic stack)

```bash
python3.12 -m venv .venv-pa
.venv-pa/bin/pip install -r priorauth/requirements.txt -r priorauth/requirements-dev.txt
.venv-pa/bin/python -m pytest tests/agents -q
```

**4. Prior-auth UI**

```bash
cd priorauth-ui && npm ci && npm run dev
```

---

## Repository Layout

```
producer/  consumer/       Kafka replay + ingestion (with DLQ)
spark/  dags/              PySpark features, Airflow orchestration
model/                     TFT training, evaluation, baseline, ONNX export, results
serving/                   FastAPI /predict + serving/healthcare (FHIR R4, SMART auth, audit log)
ehr/                       EHR <-> GlucoPulse FHIR round trip
agents/                    Clinical Reasoning Agent (MCP tools, citation grounding, quarantine)
priorauth/                 Prior-authorization agent platform (see docs/priorauth/)
hf_space/  hf_space_agent/ Hosted demos
monitoring/grafana/        Dashboards
docs/                      FHIR.md, HIPAA_MAPPING.md, CLINICAL_AGENT.md, QUESTIONS.md
```

---

## Scope & Limitations

- **Not a medical device** and not clinically validated. The forecasts are an ML benchmark on a research dataset.
- **No PHI.** The data is public and de-identified, and the FHIR/EHR tests use synthetic data only.
- **Technical HIPAA safeguards are implemented, but the project is not "HIPAA compliant"**, which is an organizational status. The gaps (encryption at rest, emergency access, authorization server) are listed in [`docs/HIPAA_MAPPING.md`](docs/HIPAA_MAPPING.md) and [`SECURITY.md`](SECURITY.md).
- The live demo hosts only the serving layer. The full pipeline runs locally with Docker Compose.

## License

MIT, see [LICENSE](LICENSE). The AZT1D dataset is licensed separately under CC BY 4.0.
