# Airflow for the cluster: the official image plus the retraining pipeline's packages, installed
# once at build time (installing them at every container start took minutes per container).
# Same pins as the pipeline's tests (mlops/requirements.txt; apache-airflow==3.3.2 is the base).
# Build from the repo root:  docker build -f infra/docker/airflow.Dockerfile -t airflow-mlops .
FROM apache/airflow:3.3.2
COPY mlops/requirements.txt /tmp/mlops-requirements.txt
RUN pip install --no-cache-dir -r /tmp/mlops-requirements.txt
