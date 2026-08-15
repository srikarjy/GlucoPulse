"""
Phase 3 batch feature job (see docs/QUESTIONS.md, 2026-07-13 entry).

Full-history recompute every run -- deliberate choice at the current scale
(N=3 patients, ~40 days each); see docs/PROBLEMS.md for the production-scale
incremental/lookback-buffer tradeoff this punts on. Resamples each patient
to the 5-min grid, then computes rolling stats on the resampled series, so
a rolling window spanning a real sensor gap reflects that gap (fewer real
data points) rather than silently averaging over stale values.

Quality gate is per patient-week, not per-patient-full-history, so one bad
week doesn't permanently disqualify a patient's other weeks under full
recompute:
  - a single gap > GAP_THRESHOLD_MIN in that week  -> excluded
  - reading count < COUNT_RATIO_THRESHOLD of expected (prorated for partial
    boundary weeks, which are calendar artifacts, not real dropout) -> excluded
Excluded weeks are logged and skipped here, never raised -- the circuit
breaker on the run's overall exclusion rate lives one layer up, in the
Airflow DAG's quality-gate task.

Note on the write path: Spark's JDBC writer has no upsert/ON CONFLICT
support, so passing rows are collected to the driver and upserted via
psycopg2. Acceptable at this data scale (a few hundred thousand rows); a
real-scale version would write to a staging table and run a SQL MERGE
instead of collecting to one process.
"""

import json
import os

import psycopg2
from psycopg2.extras import execute_values
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

GAP_THRESHOLD_MIN = 180
COUNT_RATIO_THRESHOLD = 0.85
EXPECTED_READINGS_PER_DAY = 288  # 24h * 60 / 5min

JDBC_JAR = "/opt/spark_jars/postgresql-42.7.3.jar"

FEATURE_COLUMNS = (
    "patient_id",
    "time",
    "glucose_value",
    "delta",
    "rolling_mean_15",
    "rolling_std_15",
    "rolling_mean_60",
    "rolling_std_60",
    "has_prior_bolus",
    "time_since_last_bolus_min",
    "has_prior_carb",
    "time_since_last_carb_min",
)


def pg_conn_params():
    return dict(
        host=os.environ.get("PG_HOST", "timescaledb"),
        port=os.environ.get("PG_PORT", "5432"),
        dbname=os.environ.get("POSTGRES_DB"),
        user=os.environ.get("POSTGRES_USER"),
        password=os.environ.get("POSTGRES_PASSWORD"),
    )


def jdbc_url():
    p = pg_conn_params()
    return f"jdbc:postgresql://{p['host']}:{p['port']}/{p['dbname']}"


def get_spark():
    return (
        SparkSession.builder.appName("glucopulse-feature-job")
        .master("local[*]")
        .config("spark.jars", JDBC_JAR)
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def read_raw(spark):
    p = pg_conn_params()
    return (
        spark.read.format("jdbc")
        .option("url", jdbc_url())
        .option("dbtable", "cgm_readings")
        .option("user", p["user"])
        .option("password", p["password"])
        .option("driver", "org.postgresql.Driver")
        .load()
    )


def resample_and_engineer(df):
    """Resample each patient to the 5-min grid, then compute rolling features.

    Real Dexcom timestamps drift by a second or two over time (not perfectly
    periodic) -- joining an exact arithmetic sequence against raw timestamps
    on equality only matches for a few hours before drift permanently misses
    every subsequent reading (see docs/QUESTIONS.md). Bucketing each reading
    to the nearest 5-min slot (aligned to the epoch, not per-patient) before
    joining makes the match robust to that drift.
    """
    bucket_seconds = 300
    bucketed = df.withColumn(
        "bucket_time",
        F.from_unixtime((F.round(F.unix_timestamp(F.col("time")) / bucket_seconds) * bucket_seconds).cast("long")).cast(
            "timestamp"
        ),
    )
    # Two raw readings can round into the same bucket; keep the one closest to
    # the bucket's exact center rather than an arbitrary one.
    w_pick = Window.partitionBy("patient_id", "bucket_time").orderBy(
        F.abs(F.unix_timestamp("time") - F.unix_timestamp("bucket_time"))
    )
    bucketed = (
        bucketed.withColumn("_rn", F.row_number().over(w_pick))
        .filter(F.col("_rn") == 1)
        .drop("_rn", "time")
        .withColumnRenamed("bucket_time", "time")
    )

    bounds = bucketed.groupBy("patient_id").agg(
        F.min("time").alias("min_time"), F.max("time").alias("max_time")
    )
    spine = bounds.withColumn(
        "time",
        F.explode(F.expr("sequence(min_time, max_time, interval 5 minutes)")),
    ).select("patient_id", "time")

    grid = spine.join(bucketed, on=["patient_id", "time"], how="left")

    w = Window.partitionBy("patient_id").orderBy("time")
    w15 = w.rowsBetween(-2, 0)  # 3 rows = 15 min on the 5-min grid
    w60 = w.rowsBetween(-11, 0)  # 12 rows = 60 min
    w_full = w.rowsBetween(Window.unboundedPreceding, 0)

    is_bolus_event = (F.coalesce(F.col("total_bolus_insulin_delivered"), F.lit(0.0)) > 0) | (
        F.coalesce(F.col("correction_delivered"), F.lit(0.0)) > 0
    )
    is_carb_event = F.coalesce(F.col("carb_size"), F.lit(0.0)) > 0

    last_bolus_time = F.last(F.when(is_bolus_event, F.col("time")), ignorenulls=True).over(w_full)
    last_carb_time = F.last(F.when(is_carb_event, F.col("time")), ignorenulls=True).over(w_full)

    return (
        grid.withColumn("delta", F.col("glucose_value") - F.lag("glucose_value").over(w))
        .withColumn("rolling_mean_15", F.avg("glucose_value").over(w15))
        .withColumn("rolling_std_15", F.stddev("glucose_value").over(w15))
        .withColumn("rolling_mean_60", F.avg("glucose_value").over(w60))
        .withColumn("rolling_std_60", F.stddev("glucose_value").over(w60))
        .withColumn("last_bolus_time", last_bolus_time)
        .withColumn("last_carb_time", last_carb_time)
        .withColumn("has_prior_bolus", F.col("last_bolus_time").isNotNull())
        .withColumn("has_prior_carb", F.col("last_carb_time").isNotNull())
        .withColumn(
            "time_since_last_bolus_min",
            F.when(
                F.col("last_bolus_time").isNotNull(),
                (F.col("time").cast("long") - F.col("last_bolus_time").cast("long")) / 60.0,
            ),
        )
        .withColumn(
            "time_since_last_carb_min",
            F.when(
                F.col("last_carb_time").isNotNull(),
                (F.col("time").cast("long") - F.col("last_carb_time").cast("long")) / 60.0,
            ),
        )
    )


def compute_weekly_quality(df):
    """Per (patient_id, week_start): max single gap, reading count vs. expected."""
    w = Window.partitionBy("patient_id").orderBy("time")
    with_gap = df.withColumn(
        "gap_min", (F.col("time").cast("long") - F.lag("time").over(w).cast("long")) / 60.0
    ).withColumn("week_start", F.date_trunc("week", F.col("time")))

    weekly = with_gap.groupBy("patient_id", "week_start").agg(
        F.max("gap_min").alias("max_gap_min"),
        F.count("*").alias("reading_count"),
        F.min("time").alias("week_data_min"),
        F.max("time").alias("week_data_max"),
    )

    return (
        weekly.withColumn(
            "days_observed",
            F.greatest(
                F.lit(1.0),
                (F.col("week_data_max").cast("long") - F.col("week_data_min").cast("long")) / 86400.0,
            ),
        )
        .withColumn("expected_readings", F.col("days_observed") * EXPECTED_READINGS_PER_DAY)
        .withColumn("count_ratio", F.col("reading_count") / F.col("expected_readings"))
        .withColumn(
            "failed",
            (F.col("max_gap_min") > GAP_THRESHOLD_MIN) | (F.col("count_ratio") < COUNT_RATIO_THRESHOLD),
        )
    )


def upsert_features(rows):
    if not rows:
        return
    conn = psycopg2.connect(**pg_conn_params())
    conn.autocommit = True
    update_cols = [c for c in FEATURE_COLUMNS if c not in ("patient_id", "time")]
    sql = (
        f"INSERT INTO cgm_features ({', '.join(FEATURE_COLUMNS)}) VALUES %s "
        f"ON CONFLICT (patient_id, time) DO UPDATE SET "
        + ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
    )
    try:
        with conn.cursor() as cur:
            execute_values(cur, sql, [tuple(r[c] for c in FEATURE_COLUMNS) for r in rows])
    finally:
        conn.close()


def write_run_summary(total_weeks, excluded_weeks):
    conn = psycopg2.connect(**pg_conn_params())
    conn.autocommit = True
    excluded_fraction = (len(excluded_weeks) / total_weeks) if total_weeks else 0.0
    detail = [
        {
            "patient_id": w["patient_id"],
            "week_start": str(w["week_start"]),
            "max_gap_min": w["max_gap_min"],
            "count_ratio": w["count_ratio"],
        }
        for w in excluded_weeks
    ]
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO feature_run_summary
                   (total_patient_weeks, excluded_count, excluded_fraction, excluded_detail)
                   VALUES (%s, %s, %s, %s)""",
                (total_weeks, len(excluded_weeks), excluded_fraction, json.dumps(detail)),
            )
    finally:
        conn.close()


def run_feature_job():
    spark = get_spark()
    try:
        raw = read_raw(spark)
        raw.cache()

        features = resample_and_engineer(raw).withColumn(
            "week_start", F.date_trunc("week", F.col("time"))
        )
        weekly = compute_weekly_quality(raw)

        weekly_rows = [r.asDict() for r in weekly.collect()]
        total_weeks = len(weekly_rows)
        excluded = [r for r in weekly_rows if r["failed"]]
        excluded_keys = {(r["patient_id"], r["week_start"]) for r in excluded}

        print(f"Quality gate: {len(excluded)}/{total_weeks} patient-weeks excluded")
        for r in excluded:
            print(
                f"  EXCLUDED patient={r['patient_id']} week={r['week_start']} "
                f"max_gap_min={r['max_gap_min']:.1f} count_ratio={r['count_ratio']:.2f}"
            )

        feature_rows = [r.asDict() for r in features.collect()]
        passing_rows = [
            r
            for r in feature_rows
            if (r["patient_id"], r["week_start"]) not in excluded_keys and r["glucose_value"] is not None
        ]

        upsert_features(passing_rows)
        write_run_summary(total_weeks, excluded)

        print(
            f"Done. wrote {len(passing_rows)} feature rows "
            f"({total_weeks - len(excluded)}/{total_weeks} patient-weeks passing)"
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    run_feature_job()
