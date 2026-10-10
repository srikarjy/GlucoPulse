#!/bin/bash
# Sets the password for the read-only FHIR role created in 007_healthcare.sql.
# Runs only on a fresh data volume. On an existing one:
#   docker exec -it glucopulse-timescaledb psql -U $POSTGRES_USER -d $POSTGRES_DB \
#     -c "ALTER ROLE glucopulse_fhir PASSWORD '<value of GLUCOPULSE_FHIR_DB_PASSWORD>'"
set -e
if [ -z "$GLUCOPULSE_FHIR_DB_PASSWORD" ]; then
  echo "GLUCOPULSE_FHIR_DB_PASSWORD not set; glucopulse_fhir has no password and cannot log in."
  exit 0
fi
psql -v ON_ERROR_STOP=1 -v pw="$GLUCOPULSE_FHIR_DB_PASSWORD" --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
  <<'SQL'
ALTER ROLE glucopulse_fhir PASSWORD :'pw';
SQL
