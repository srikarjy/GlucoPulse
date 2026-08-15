"""Shared TimescaleDB read helper for Phase 4 scripts."""

import os

import pandas as pd
import psycopg2


def pg_conn_params():
    return dict(
        host=os.environ.get("PG_HOST", "timescaledb"),
        port=os.environ.get("PG_PORT", "5432"),
        dbname=os.environ.get("POSTGRES_DB"),
        user=os.environ.get("POSTGRES_USER"),
        password=os.environ.get("POSTGRES_PASSWORD"),
    )


def load_features() -> pd.DataFrame:
    """All of cgm_features, one row per (patient_id, time)."""
    conn = psycopg2.connect(**pg_conn_params())
    try:
        df = pd.read_sql(
            """
            SELECT patient_id, time, glucose_value, delta,
                   rolling_mean_15, rolling_std_15,
                   rolling_mean_60, rolling_std_60,
                   has_prior_bolus, time_since_last_bolus_min,
                   has_prior_carb, time_since_last_carb_min
            FROM cgm_features
            ORDER BY patient_id, time
            """,
            conn,
        )
    finally:
        conn.close()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df
