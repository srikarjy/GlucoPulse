-- Observability log for all three DLQ failure classes (see docs/QUESTIONS.md,
-- 2026-07-13 entry). Plain table, not a hypertable -- DLQ volume is expected
-- to be low, partitioning isn't justified at this scale.
--
-- This table is a record of *that* a failure happened, for the Grafana DLQ
-- health panel. It does not replace the underlying topics (cgm-parse-errors,
-- cgm-dlq, cgm-implausible) as the source of truth for the actual message
-- content -- conflicting-timestamp resolution in particular stays
-- human-review-only, this table doesn't touch that.

CREATE TABLE IF NOT EXISTS dlq_events (
    id             BIGSERIAL PRIMARY KEY,
    topic          TEXT        NOT NULL,
    patient_id     TEXT,
    event_datetime TEXT,
    reason         TEXT,
    received_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_dlq_events_topic_received
    ON dlq_events (topic, received_at DESC);
