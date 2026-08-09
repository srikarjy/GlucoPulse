-- Phase 3 batch features (see docs/QUESTIONS.md, 2026-07-13 entry).
-- Same grain as cgm_readings: one row per (patient_id, time) that has a
-- real reading. Written by the PySpark batch job via upsert -- full-history
-- recompute every run, so ON CONFLICT DO UPDATE (not DO NOTHING): a
-- recompute with changed feature logic should overwrite, not be ignored.

CREATE TABLE IF NOT EXISTS cgm_features (
    patient_id               TEXT        NOT NULL,
    time                     TIMESTAMPTZ NOT NULL,
    glucose_value            DOUBLE PRECISION,
    delta                    DOUBLE PRECISION,
    rolling_mean_15          DOUBLE PRECISION,
    rolling_std_15           DOUBLE PRECISION,
    rolling_mean_60          DOUBLE PRECISION,
    rolling_std_60           DOUBLE PRECISION,
    has_prior_bolus          BOOLEAN,
    time_since_last_bolus_min DOUBLE PRECISION,
    has_prior_carb           BOOLEAN,
    time_since_last_carb_min  DOUBLE PRECISION,
    computed_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (patient_id, time)
);

SELECT create_hypertable('cgm_features', by_range('time'), if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_cgm_features_patient_time
    ON cgm_features (patient_id, time DESC);

-- One row per Airflow DAG run of the feature job -- read by the downstream
-- quality-gate task to decide whether to hard-fail the DAG (circuit breaker
-- on overall exclusion rate, not on any single patient-week).
CREATE TABLE IF NOT EXISTS feature_run_summary (
    id                   BIGSERIAL PRIMARY KEY,
    run_at               TIMESTAMPTZ NOT NULL DEFAULT now(),
    total_patient_weeks  INTEGER NOT NULL,
    excluded_count       INTEGER NOT NULL,
    excluded_fraction    DOUBLE PRECISION NOT NULL,
    excluded_detail      JSONB
);
