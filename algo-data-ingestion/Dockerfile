# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies for pyarrow/snappy and builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    libsnappy-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip tooling
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Core Python dependencies
# NOTE: transformers + torch are large; needed for sentiment endpoints.
# If you don't need them immediately, you can remove them to slim the image.
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pydantic>=2.6 \
    pydantic-settings>=2.2 \
    pandas \
    pyarrow \
    python-snappy \
    httpx \
    tweepy \
    tenacity \
    ratelimit \
    prometheus-client \
    ccxt \
    redis \
    numpy \
    numba \
    feedparser \
    sentence-transformers \
    transformers \
    torch

# Copy application code
COPY . .

EXPOSE 8000

# Simple container healthcheck against the FastAPI health endpoint
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fsS http://127.0.0.1:8000/health | grep -q '"status":"ok"' || exit 1

# Run the ingestion API
CMD ["uvicorn", "app.ingestion_service.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]