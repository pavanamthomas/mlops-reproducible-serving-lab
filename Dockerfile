FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir .

ENV MLSERV_ARTIFACT_PATH=/app/models/bundle.joblib
ENV MPLBACKEND=Agg

RUN python scripts/train.py --n-samples 400 --skip-mlflow --artifact-dir /app/models

EXPOSE 8000

CMD ["uvicorn", "mlserv.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
