#!/bin/bash
# Airflow's metadata store lives on this same Postgres instance, in its own
# database (not mixed into the glucopulse app DB) -- avoids standing up a
# separate Postgres service just for Airflow's bookkeeping, which isn't in
# the locked stack. Numbered to run before 001+ so the DB exists before any
# app-schema scripts, though order doesn't actually matter between them.
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    SELECT 'CREATE DATABASE airflow' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'airflow')\gexec
EOSQL
