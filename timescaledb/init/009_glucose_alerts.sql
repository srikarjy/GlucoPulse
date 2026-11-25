-- Streaming glucose alerts (consumer/alerts_stream.py, docs/STREAMING.md).
--
-- glucose_alerts: one row per alert decision. The primary key makes re-processing after a restart harmless: the same
-- reading replayed gives the same (patient, time, tier), and the upsert keeps a single row (ON CONFLICT ... DO UPDATE
-- SET published = published RETURNING published, so the consumer learns whether it was already sent). `published`
-- is the outbox flag: set only after Kafka confirms delivery to cgm-alerts, so an alert saved just before a crash is
-- re-sent on restart instead of being lost. Plain table (low volume: ~4 alerts per person per day).
CREATE TABLE IF NOT EXISTS glucose_alerts (
    patient_id      TEXT             NOT NULL,
    time            TIMESTAMPTZ      NOT NULL,      -- the reading the alert was raised on (5-min slot for 5-min data)
    tier            TEXT             NOT NULL CHECK (tier IN ('heads_up', 'loud')),
    direction       TEXT             NOT NULL CHECK (direction IN ('low', 'high')),
    glucose         DOUBLE PRECISION NOT NULL,
    projected_20min DOUBLE PRECISION NOT NULL,
    spacing_min     SMALLINT         NOT NULL CHECK (spacing_min IN (5, 15)),
    published       BOOLEAN          NOT NULL DEFAULT FALSE,
    detected_at     TIMESTAMPTZ      NOT NULL DEFAULT now(),
    PRIMARY KEY (patient_id, time, tier)
);

CREATE INDEX IF NOT EXISTS idx_glucose_alerts_unpublished
    ON glucose_alerts (detected_at) WHERE NOT published;

-- alert_checkpoints: the alert consumer's memory (AlertStream.snapshot() for the people in one partition) and the
-- Kafka offset to resume from, written in one transaction so they always agree. On restart the consumer loads the
-- checkpoint and seeks to next_offset; only messages after the checkpoint are processed again.
CREATE TABLE IF NOT EXISTS alert_checkpoints (
    topic            TEXT        NOT NULL,
    kafka_partition  INTEGER     NOT NULL,
    next_offset      BIGINT      NOT NULL,
    state            JSONB       NOT NULL,
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (topic, kafka_partition)
);
