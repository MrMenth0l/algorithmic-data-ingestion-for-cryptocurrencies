

# async_infra.py
from __future__ import annotations
"""Shared async infrastructure for HTTP and retries.

This module centralizes how we create async HTTP clients and how we retry
network calls. Use `make_async_client` to build an `httpx.AsyncClient`, and
decorate I/O-bound coroutines with `@retry_httpx()`.
"""

from typing import Optional
import httpx
from tenacity import (
    retry,
    wait_random_exponential,
    stop_after_attempt,
    retry_if_exception,
)

__all__ = ["make_async_client", "retry_httpx", "get_http", "close_http"]

# -------------------------------------------------------------
# Retry policy
# -------------------------------------------------------------

def _should_retry(exc: BaseException) -> bool:
    """Return True if the exception is transient and should be retried.

    We retry on typical transient httpx errors and on HTTP 429 / 5xx when
    callers use `response.raise_for_status()`.
    """
    if isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.WriteError,
            httpx.TimeoutException,
            httpx.RemoteProtocolError,
            httpx.PoolTimeout,
        ),
    ):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or 500 <= code <= 599
    return False


def retry_httpx(max_attempts: int = 5):
    """Decorator factory: retry transient HTTP errors with jittered backoff.

    Example:
        @retry_httpx()
        async def fetch(...):
            resp = await client.get("/endpoint")
            resp.raise_for_status()
            return resp.json()
    """
    return retry(
        retry=retry_if_exception(_should_retry),
        wait=wait_random_exponential(multiplier=0.5, max=30.0),
        stop=stop_after_attempt(max_attempts),
        reraise=True,
    )


# -------------------------------------------------------------
# Async HTTPX client factory
# -------------------------------------------------------------

def make_async_client(
    base_url: Optional[str] = None,
    headers: Optional[dict] = None,
    timeout_sec: float = 10.0,
    max_keepalive: int = 100,
    max_connections: int = 200,
) -> httpx.AsyncClient:
    """Create a tuned `httpx.AsyncClient` with sane defaults.

    Args:
        base_url: Optional base URL for the client.
        headers: Optional default headers.
        timeout_sec: Per-operation timeout (connect/read/write and total).
        max_keepalive: Max number of keep-alive connections to pool.
        max_connections: Max total concurrent connections.
    """
    limits = httpx.Limits(
        max_keepalive_connections=max_keepalive,
        max_connections=max_connections,
    )
    timeout = httpx.Timeout(
        timeout_sec, connect=timeout_sec, read=timeout_sec, write=timeout_sec
    )
    return httpx.AsyncClient(
        base_url=base_url or "",
        headers=headers or {},
        limits=limits,
        timeout=timeout,
    )


# -------------------------------------------------------------
# Shared singleton client & lifecycle helpers
# -------------------------------------------------------------

_CLIENT: Optional[httpx.AsyncClient] = None

def get_http() -> httpx.AsyncClient:
    """Return a process-wide shared AsyncClient (created lazily).

    Uses `make_async_client` with sane pool sizes for our service. Centralizing
    this ensures connection reuse and simplifies DI via FastAPI lifespan.
    """
    global _CLIENT
    if _CLIENT is None:
        # Defaults aligned with Phase-2 Batch 1.1 acceptance
        _CLIENT = make_async_client(timeout_sec=10.0, max_keepalive=20, max_connections=100)
    return _CLIENT

async def close_http() -> None:
    """Close and clear the shared AsyncClient (idempotent)."""
    global _CLIENT
    if _CLIENT is not None:
        client, _CLIENT = _CLIENT, None
        await client.aclose()