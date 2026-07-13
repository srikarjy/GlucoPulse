-- Phase 2: raw covariate columns for the ingestion consumer (ingest.py).
-- Consumer is deliberately stateless/raw-only -- no delta/rolling-stat
-- columns here, those are Phase 3's PySpark batch job (see docs/QUESTIONS.md,
-- 2026-07-13 entry).

ALTER TABLE cgm_readings ADD COLUMN IF NOT EXISTS device_mode TEXT;
ALTER TABLE cgm_readings ADD COLUMN IF NOT EXISTS bolus_type TEXT;
ALTER TABLE cgm_readings ADD COLUMN IF NOT EXISTS basal DOUBLE PRECISION;
ALTER TABLE cgm_readings ADD COLUMN IF NOT EXISTS correction_delivered DOUBLE PRECISION;
ALTER TABLE cgm_readings ADD COLUMN IF NOT EXISTS total_bolus_insulin_delivered DOUBLE PRECISION;
ALTER TABLE cgm_readings ADD COLUMN IF NOT EXISTS food_delivered DOUBLE PRECISION;
ALTER TABLE cgm_readings ADD COLUMN IF NOT EXISTS carb_size DOUBLE PRECISION;
