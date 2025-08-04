import pytest
from fastapi.testclient import TestClient
from app.ingestion_service.main import app
from pydantic_settings import BaseSettings
from pydantic import Field

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"service": "raw-data-ingestion", "version": "1.0.0"}

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    # Should include Prometheus metric for writes (even if zero)
    assert "parquet_writes_total" in response.text
    assert "parquet_write_errors_total" in response.text
    assert "parquet_write_latency_seconds" in response.text