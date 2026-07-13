-- Phase 1: bare raw-ingestion schema only.
-- Delta/rolling-stat/covariate columns are deferred to Phase 2 once the
-- consumer's actual computed feature list is real (see docs/BLUEPRINT.md).

CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS cgm_readings (
    patient_id   TEXT        NOT NULL,
    time         TIMESTAMPTZ NOT NULL,
    glucose_value DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (patient_id, time)
);

SELECT create_hypertable('cgm_readings', by_range('time'), if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_cgm_readings_patient_time
    ON cgm_readings (patient_id, time DESC);
