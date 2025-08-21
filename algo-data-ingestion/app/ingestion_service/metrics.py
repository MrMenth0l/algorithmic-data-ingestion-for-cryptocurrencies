# app/ingestion_service/metrics.py
from __future__ import annotations
import time
from contextlib import contextmanager
from prometheus_client import Counter, Histogram

# --- Request-level metrics (per ingest route) ---
INGEST_REQUESTS = Counter(
    "ingest_requests_total",
    "Total ingest requests by domain and final status.",
    labelnames=("domain", "status"),
)

INGEST_DURATION = Histogram(
    "ingest_duration_seconds",
    "Ingest request duration seconds by domain.",
    labelnames=("domain",),
    # Reasonable buckets for API work
    buckets=(0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)

# --- Feature write metrics (per domain) ---
FEATURE_ROWS_WRITTEN = Counter(
    "feature_rows_written_total",
    "Total feature rows written to the feature store (Redis) by domain.",
    labelnames=("domain",),
)

class _IngestSpan:
    __slots__ = ("domain", "_start", "_status")

    def __init__(self, domain: str):
        self.domain = domain
        self._start = time.perf_counter()
        # Default to "error" unless explicitly marked otherwise.
        self._status = "error"

    def set_status(self, status: str) -> None:
        # expected: "ok" | "no_data" | "error"
        self._status = status

    def finish(self) -> None:
        duration = time.perf_counter() - self._start
        INGEST_REQUESTS.labels(domain=self.domain, status=self._status).inc()
        INGEST_DURATION.labels(domain=self.domain).observe(duration)

@contextmanager
def ingest_span(domain: str):
    """
    Usage:
        with ingest_span("market") as span:
            ... do work ...
            span.set_status("ok")  # or "no_data"
    If an exception bubbles out, status remains "error".
    """
    span = _IngestSpan(domain)
    try:
        yield span
    finally:
        span.finish()

def record_rows_written(domain: str, n: int) -> None:
    """
    Increment the rows-written counter if n > 0.
    Call this in your _write_*_features_to_store helpers.
    """
    try:
        if n and n > 0:
            FEATURE_ROWS_WRITTEN.labels(domain=domain).inc(n)
    except Exception:
        # Never let metrics throw; swallow to avoid affecting the ingest flow.
        pass