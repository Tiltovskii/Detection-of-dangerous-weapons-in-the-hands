ARG PYTHON_VERSION=3.10.4
FROM python:${PYTHON_VERSION}-slim as base

WORKDIR /app

RUN pip install --upgrade pip && \
    pip install --no-cache-dir mlflow

EXPOSE 5000

CMD mlflow server \
    --backend-store-uri sqlite:////app/mlflow.db \
    --default-artifact-root /Users/bende/.volumes/mlflow/artifacts \
#    --default-artifact-root $ARTIFACT_ROOT \
    --host 0.0.0.0 \
    --port 5000