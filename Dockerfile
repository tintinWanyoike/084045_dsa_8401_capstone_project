# Dormancy early-warning API
# Build:  docker build -t dormancy-api .
# Run:    docker run --rm -p 8000:8000 dormancy-api      (docs at http://localhost:8000/docs)
FROM python:3.12-slim

# LightGBM needs the OpenMP runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models

WORKDIR /app

# Dependencies first, so code changes do not reinstall them
COPY service/requirements.txt service/requirements.txt
RUN pip install --no-cache-dir -r service/requirements.txt

# Only what serving needs: the API code and the model artefacts
COPY service/__init__.py service/__init__.py
COPY service/app service/app
COPY models/lgbm_hazard.txt models/metadata.json models/monitoring_reference.json models/
COPY feature_dictionary.csv ./

# Run as a non-root user
RUN useradd --create-home --uid 1000 api && chown -R api /app
USER api

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"

CMD ["uvicorn", "service.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
