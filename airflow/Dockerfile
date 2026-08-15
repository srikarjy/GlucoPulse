# PySpark needs a JVM; Airflow's own image doesn't include one, and the
# "batch feature job" is honestly a local single-node Spark process here
# (see docs/PROBLEMS.md) -- this image runs Airflow *and* Spark in the same
# container rather than launching a separate job container, to avoid taking
# on docker-in-docker (mounting the host docker socket) for a demo-scale
# pipeline that doesn't need it.
FROM apache/airflow:2.9.3-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        default-jdk-headless \
        curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="${JAVA_HOME}/bin:${PATH}"

RUN mkdir -p /opt/spark_jars && \
    curl -Lo /opt/spark_jars/postgresql-42.7.3.jar \
        https://repo1.maven.org/maven2/org/postgresql/postgresql/42.7.3/postgresql-42.7.3.jar && \
    chown -R airflow:root /opt/spark_jars

USER airflow
COPY airflow/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

ENV PYTHONPATH=/opt/airflow:${PYTHONPATH}
COPY --chown=airflow:root spark/ /opt/airflow/spark/
