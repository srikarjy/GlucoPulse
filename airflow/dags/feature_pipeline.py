"""
Phase 3 batch feature pipeline (see docs/QUESTIONS.md, 2026-07-13 entry).

Two tasks, deliberately kept separate:
  1. run_feature_job -- computes features, excludes+logs failing patient-weeks
     locally, never raises for a single bad week (mirrors the DLQ
     separation-of-concerns principle: one patient's bad sensor week is a
     localized, expected-to-happen data issue, not a pipeline failure).
  2. quality_gate_check -- the circuit breaker. Reads the run's exclusion
     rate and is the *only* thing allowed to fail this DAG, and only when
     the exclusion rate crosses a threshold that would indicate a systemic
     problem (schema drift, broken ingestion) rather than 1-2 patients
     having a bad week. The threshold is an explicit placeholder tied to
     fleet size (arbitrary at N=3 patients) -- revisit once patient count is
     large enough for a percentage to mean something.
"""

from datetime import datetime

from airflow.decorators import dag, task
from airflow.exceptions import AirflowException

# Circuit-breaker placeholder -- see module docstring. Arbitrary at N=3
# patients; revisit at real fleet scale.
EXCLUSION_RATE_CIRCUIT_BREAKER = 0.5


@dag(
    dag_id="feature_pipeline",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["glucopulse", "phase3"],
)
def feature_pipeline():
    @task
    def run_feature_job():
        from spark.feature_job import run_feature_job as _run

        _run()

    @task
    def quality_gate_check():
        import os

        import psycopg2

        conn = psycopg2.connect(
            host=os.environ.get("PG_HOST", "timescaledb"),
            port=os.environ.get("PG_PORT", "5432"),
            dbname=os.environ.get("POSTGRES_DB"),
            user=os.environ.get("POSTGRES_USER"),
            password=os.environ.get("POSTGRES_PASSWORD"),
        )
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT total_patient_weeks, excluded_count, excluded_fraction, excluded_detail
                       FROM feature_run_summary ORDER BY run_at DESC LIMIT 1"""
                )
                row = cur.fetchone()
        finally:
            conn.close()

        if row is None:
            raise AirflowException("No feature_run_summary row found -- feature job did not report a run.")

        total, excluded_count, excluded_fraction, detail = row
        print(f"Latest run: {excluded_count}/{total} patient-weeks excluded ({excluded_fraction:.2%})")
        print(f"Excluded detail: {detail}")

        if excluded_fraction > EXCLUSION_RATE_CIRCUIT_BREAKER:
            raise AirflowException(
                f"Circuit breaker tripped: {excluded_fraction:.2%} of patient-weeks excluded "
                f"(threshold {EXCLUSION_RATE_CIRCUIT_BREAKER:.0%}) -- treating as a systemic data "
                f"issue, not independent per-patient noise."
            )

    run_feature_job() >> quality_gate_check()


feature_pipeline()
