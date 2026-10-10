-- Healthcare compliance tables (see docs/HIPAA_MAPPING.md).

-- Append-only, hash-chained audit log (164.312(b), 164.312(c)(1)).
-- Same record the FHIR facade writes to its JSONL sink; the DB copy is what
-- survives container loss and is what the AuditEvent endpoint can be pointed at.
CREATE TABLE IF NOT EXISTS audit_log (
    id             BIGSERIAL PRIMARY KEY,
    ts             TIMESTAMPTZ NOT NULL,
    actor          TEXT NOT NULL,
    client_id      TEXT,
    action         TEXT NOT NULL,
    resource_type  TEXT NOT NULL,
    patient        TEXT,
    outcome        TEXT NOT NULL CHECK (outcome IN ('success', 'denied', 'error')),
    purpose_of_use TEXT NOT NULL DEFAULT 'unspecified',
    method         TEXT,
    path           TEXT,
    status         INT,
    request_id     TEXT,
    source_ip      TEXT,
    prev_hash      TEXT NOT NULL,
    entry_hash     TEXT NOT NULL UNIQUE
);

CREATE INDEX IF NOT EXISTS idx_audit_log_patient_ts ON audit_log (patient, ts DESC);

-- Audit rows are never edited or deleted, by anyone, including the table owner
-- (a superuser can still drop the trigger -- that's why the chain exists).
CREATE OR REPLACE FUNCTION audit_log_immutable() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'audit_log is append-only';
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS audit_log_no_update ON audit_log;
CREATE TRIGGER audit_log_no_update
    BEFORE UPDATE OR DELETE ON audit_log
    FOR EACH ROW EXECUTE FUNCTION audit_log_immutable();

DROP TRIGGER IF EXISTS audit_log_no_truncate ON audit_log;
CREATE TRIGGER audit_log_no_truncate
    BEFORE TRUNCATE ON audit_log
    FOR EACH STATEMENT EXECUTE FUNCTION audit_log_immutable();

-- Minimum-necessary DB access for the FHIR facade: read-only on readings,
-- insert-only on the audit log. No password here -- set it out of band:
--   ALTER ROLE glucopulse_fhir PASSWORD '...';
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'glucopulse_fhir') THEN
        CREATE ROLE glucopulse_fhir LOGIN;
    END IF;
END
$$;

GRANT SELECT ON cgm_readings TO glucopulse_fhir;
GRANT INSERT ON audit_log TO glucopulse_fhir;
GRANT USAGE ON SEQUENCE audit_log_id_seq TO glucopulse_fhir;
