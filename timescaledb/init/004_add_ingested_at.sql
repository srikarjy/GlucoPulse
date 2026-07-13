-- cgm_readings.time is the sensor's own event timestamp (historical replay
-- data, Dec 2023-Apr 2024 for AZT1D) -- it's the wrong column for a "live
-- ingestion rate" Grafana panel, since Grafana's time picker defaults to
-- real wall-clock "now". ingested_at captures when this consumer actually
-- wrote the row, so the ingestion-rate panel reflects real-time activity.
ALTER TABLE cgm_readings ADD COLUMN IF NOT EXISTS ingested_at TIMESTAMPTZ NOT NULL DEFAULT now();

CREATE INDEX IF NOT EXISTS idx_cgm_readings_ingested_at ON cgm_readings (ingested_at DESC);
