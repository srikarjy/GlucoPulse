-- Model monitoring tables for Phase 4

-- Patient splits (train/val/test) - populated once after dataset is known
CREATE TABLE IF NOT EXISTS patient_splits (
    patient_id TEXT PRIMARY KEY,
    split TEXT NOT NULL CHECK (split IN ('train', 'val', 'test')),
    assigned_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('patient_splits', 'assigned_at', if_not_exists => TRUE);

-- TFT model predictions for monitoring
CREATE TABLE IF NOT EXISTS model_predictions (
    time TIMESTAMPTZ NOT NULL,
    patient_id TEXT NOT NULL,
    horizon_min INT NOT NULL CHECK (horizon_min IN (30, 60)),
    actual_glucose DOUBLE PRECISION,
    tft_prediction DOUBLE PRECISION,
    tft_pi_lower DOUBLE PRECISION,
    tft_pi_upper DOUBLE PRECISION,
    persistence_prediction DOUBLE PRECISION,
    model_version TEXT DEFAULT 'v1',
    received_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('model_predictions', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_model_predictions_patient_horizon ON model_predictions (patient_id, horizon_min, time DESC);

-- Aggregated model metrics (hourly rollups for dashboard performance)
CREATE TABLE IF NOT EXISTS model_metrics (
    time TIMESTAMPTZ NOT NULL,
    patient_id TEXT NOT NULL,
    horizon_min INT NOT NULL CHECK (horizon_min IN (30, 60)),
    rmse DOUBLE PRECISION,
    mae DOUBLE PRECISION,
    n_predictions INT,
    model_version TEXT DEFAULT 'v1'
);

SELECT create_hypertable('model_metrics', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_model_metrics_patient_horizon ON model_metrics (patient_id, horizon_min, time DESC);

-- Persistence baseline metrics for comparison
CREATE TABLE IF NOT EXISTS persistence_metrics (
    time TIMESTAMPTZ NOT NULL,
    patient_id TEXT NOT NULL,
    horizon_min INT NOT NULL CHECK (horizon_min IN (30, 60)),
    rmse DOUBLE PRECISION,
    mae DOUBLE PRECISION,
    n_predictions INT
);

SELECT create_hypertable('persistence_metrics', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_persistence_metrics_patient_horizon ON persistence_metrics (patient_id, horizon_min, time DESC);

-- Clarke error grid zone counts
CREATE TABLE IF NOT EXISTS clarke_zones (
    time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    horizon_min INT NOT NULL CHECK (horizon_min IN (30, 60)),
    zone TEXT NOT NULL CHECK (zone IN ('A', 'B', 'C', 'D', 'E')),
    count INT NOT NULL,
    model_version TEXT DEFAULT 'v1'
);

-- Continuous aggregate for hourly model metrics
CREATE MATERIALIZED VIEW IF NOT EXISTS model_metrics_1h
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    patient_id,
    horizon_min,
    AVG(rmse) AS avg_rmse,
    AVG(mae) AS avg_mae,
    SUM(n_predictions) AS total_predictions
FROM model_metrics
GROUP BY bucket, patient_id, horizon_min;

-- Continuous aggregate for hourly persistence metrics
CREATE MATERIALIZED VIEW IF NOT EXISTS persistence_metrics_1h
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    patient_id,
    horizon_min,
    AVG(rmse) AS avg_rmse,
    AVG(mae) AS avg_mae,
    SUM(n_predictions) AS total_predictions
FROM persistence_metrics
GROUP BY bucket, patient_id, horizon_min;

-- Add refresh policies (run every hour)
SELECT add_continuous_aggregate_policy('model_metrics_1h',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

SELECT add_continuous_aggregate_policy('persistence_metrics_1h',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');