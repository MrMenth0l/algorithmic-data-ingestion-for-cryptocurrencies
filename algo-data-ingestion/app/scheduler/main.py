# app/scheduler/main.py
from __future__ import annotations

import os
import json
import time
import logging
import asyncio
from typing import Any, Dict, List, Optional

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from prometheus_client import start_http_server, Counter, Histogram, Gauge
from zoneinfo import ZoneInfo

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [scheduler] %(message)s")
log = logging.getLogger("scheduler")

# ------------------------------------------------------------------------------
# Env
# ------------------------------------------------------------------------------
API_BASE_URL: str = os.getenv("API_BASE_URL", "http://ingestion-api:8000").rstrip("/")
ADMIN_TOKEN: str = os.getenv("ADMIN_TOKEN", "changeme")  # keep out of logs!
RUN_ON_START: bool = os.getenv("RUN_ON_START", "1") not in ("0", "false", "False", "")
SCHED_TZ: str = os.getenv("SCHED_TZ", "UTC")
SCHED_METRICS_PORT: int = int(os.getenv("SCHED_METRICS_PORT", "9002"))

# Market jobs JSON: list of {"exchange","symbol","timeframe","lookback_minutes","cron"}
def _get_market_jobs() -> List[Dict[str, Any]]:
    raw = os.getenv("MARKET_JOBS", "[]")
    try:
        jobs = json.loads(raw) if raw.strip() else []
        if not isinstance(jobs, list):
            raise ValueError("MARKET_JOBS must be a JSON list")
        return jobs
    except Exception as e:
        log.warning("Failed to parse MARKET_JOBS (%s). Using empty list.", e)
        return []

# TTL sweep config
TTL_SWEEP_CRON: str = os.getenv("TTL_SWEEP_CRON", "*/15 * * * *")
TTL_SWEEP_PATTERN: str = os.getenv("TTL_SWEEP_PATTERN", "features:market:*")
TTL_SWEEP_TTL: int = int(os.getenv("TTL_SWEEP_TTL", "3600"))
TTL_SWEEP_MAX_KEYS: Optional[int] = int(os.getenv("TTL_SWEEP_MAX_KEYS", "0")) or None  # optional

# ------------------------------------------------------------------------------
# Prometheus metrics
# ------------------------------------------------------------------------------
# How often jobs get invoked
JOB_RUNS = Counter(
    "scheduler_job_runs_total",
    "Total number of scheduler job invocations",
    ["job_id"],
)

# Success/failure counts
JOB_SUCCESS = Counter(
    "scheduler_job_success_total",
    "Total successful job completions",
    ["job_id"],
)
JOB_FAILURE = Counter(
    "scheduler_job_failure_total",
    "Total failed job completions",
    ["job_id", "reason"],
)

# How long each job takes
JOB_DURATION = Histogram(
    "scheduler_job_duration_seconds",
    "Duration of scheduler jobs in seconds",
    ["job_id"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120, 300),
)

# Last run timestamps (unix epoch)
JOB_LAST_RUN_TS = Gauge(
    "scheduler_job_last_run_timestamp",
    "Unix timestamp of the last time the job started",
    ["job_id"],
)
JOB_LAST_SUCCESS_TS = Gauge(
    "scheduler_job_last_success_timestamp",
    "Unix timestamp of the last time the job succeeded",
    ["job_id"],
)

# API health (1 = up, 0 = down)
API_HEALTH = Gauge(
    "scheduler_api_up",
    "Whether the ingestion API health endpoint responded OK (1) or not (0)",
)

# ------------------------------------------------------------------------------
# HTTP helpers
# ------------------------------------------------------------------------------
def _auth_headers() -> Dict[str, str]:
    # Never log the token value.
    return {"X-Admin-Token": ADMIN_TOKEN}

async def _get_health() -> bool:
    url = f"{API_BASE_URL}/health"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(url)
            r.raise_for_status()
        API_HEALTH.set(1)
        log.info('API /health OK at %s', API_BASE_URL)
        return True
    except Exception as e:
        API_HEALTH.set(0)
        log.warning("API not ready yet (%s); retrying...", e)
        return False

async def wait_for_api_ready(max_tries: int = 20, delay_s: float = 2.0) -> None:
    for _ in range(max_tries):
        if await _get_health():
            return
        await asyncio.sleep(delay_s)
    # If still not ready, continue — jobs will 404/ConnectError and we’ll record failures.

async def _post_admin(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    POST to an admin endpoint using query params.
    NOTE: our API expects parameters in the query string (not JSON body).
    """
    url = f"{API_BASE_URL}{path}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, headers=_auth_headers(), params=params)
        # 2xx -> OK, else raises
        resp.raise_for_status()
        # content may be json
        try:
            return resp.json()
        except Exception:
            return {"raw": resp.text}

# ------------------------------------------------------------------------------
# Job wrappers (with metrics)
# ------------------------------------------------------------------------------
async def _run_with_metrics(job_id: str, coro_fn, *args, **kwargs) -> Optional[Dict[str, Any]]:
    """
    Wrap a coroutine job with standardized metrics and logging.
    """
    JOB_RUNS.labels(job_id=job_id).inc()
    JOB_LAST_RUN_TS.labels(job_id=job_id).set(time.time())

    start = time.perf_counter()
    try:
        result = await coro_fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        JOB_DURATION.labels(job_id=job_id).observe(elapsed)
        JOB_SUCCESS.labels(job_id=job_id).inc()
        JOB_LAST_SUCCESS_TS.labels(job_id=job_id).set(time.time())
        return result
    except httpx.HTTPStatusError as e:
        elapsed = time.perf_counter() - start
        JOB_DURATION.labels(job_id=job_id).observe(elapsed)
        reason = f"http_{e.response.status_code}"
        JOB_FAILURE.labels(job_id=job_id, reason=reason).inc()
        log.error("%s failed: %s\n%s", job_id, e, "For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/%s" % e.response.status_code if hasattr(e, "response") else "")
        return None
    except Exception as e:
        elapsed = time.perf_counter() - start
        JOB_DURATION.labels(job_id=job_id).observe(elapsed)
        JOB_FAILURE.labels(job_id=job_id, reason=type(e).__name__).inc()
        log.exception("%s failed: %s", job_id, e)
        return None

# ------------------------------------------------------------------------------
# Concrete jobs
# ------------------------------------------------------------------------------
async def _market_backfill(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Endpoint expects: exchange, symbol, timeframe, lookback_minutes
    return await _post_admin("/ingest/admin/backfill/market", payload)

async def _ttl_sweep(params: Dict[str, Any]) -> Dict[str, Any]:
    # Endpoint expects: pattern, ttl_default (and optional max_keys)
    return await _post_admin("/ingest/admin/features/ttl-sweep", params)

async def run_market_backfill_job(exchange: str, symbol: str, timeframe: str, lookback_minutes: int) -> None:
    job_id = f"backfill:{exchange}:{symbol}:{timeframe}"
    payload = {
        "exchange": exchange,
        "symbol": symbol,
        "timeframe": timeframe,
        "lookback_minutes": lookback_minutes,
    }
    out = await _run_with_metrics(job_id, _market_backfill, payload)
    if out is not None:
        log.info("Market backfill ok %s -> %s", job_id, out)

async def run_ttl_sweep_job(pattern: str, ttl_default: int, max_keys: Optional[int]) -> None:
    job_id = "ttl_sweep"
    params = {"pattern": pattern, "ttl_default": ttl_default}
    if max_keys:
        params["max_keys"] = max_keys
    out = await _run_with_metrics(job_id, _ttl_sweep, params)
    if out is not None:
        log.info("TTL sweep ok -> %s", out)

# ------------------------------------------------------------------------------
# Bootstrapping + scheduling
# ------------------------------------------------------------------------------
def _tz() -> ZoneInfo:
    try:
        return ZoneInfo(SCHED_TZ)
    except Exception:
        log.warning("Unknown timezone %s, falling back to UTC", SCHED_TZ)
        return ZoneInfo("UTC")

def _add_market_jobs(sched: AsyncIOScheduler, jobs: List[Dict[str, Any]]) -> None:
    for j in jobs:
        exchange = j["exchange"]
        symbol = j["symbol"]
        timeframe = j["timeframe"]
        lookback_minutes = int(j.get("lookback_minutes", 15))
        cron = j["cron"]
        job_id = f"backfill:{exchange}:{symbol}:{timeframe}"

        # Cron job
        sched.add_job(
            run_market_backfill_job,
            trigger=CronTrigger.from_crontab(cron, timezone=_tz()),
            id=job_id,
            kwargs={
                "exchange": exchange,
                "symbol": symbol,
                "timeframe": timeframe,
                "lookback_minutes": lookback_minutes,
            },
            max_instances=1,  # keep it simple; dedupe/locks could be added later
            replace_existing=True,
        )

        # One-shot on boot (optional)
        if RUN_ON_START:
            sched.add_job(
                run_market_backfill_job,
                trigger="date",
                run_date=None,  # now
                id=f"boot:{exchange}:{symbol}:{timeframe}",
                kwargs={
                    "exchange": exchange,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "lookback_minutes": lookback_minutes,
                },
                replace_existing=True,
            )

def _add_ttl_sweep_job(sched: AsyncIOScheduler) -> None:
    # Cron job
    sched.add_job(
        run_ttl_sweep_job,
        trigger=CronTrigger.from_crontab(TTL_SWEEP_CRON, timezone=_tz()),
        id="ttl_sweep",
        kwargs={
            "pattern": TTL_SWEEP_PATTERN,
            "ttl_default": TTL_SWEEP_TTL,
            "max_keys": TTL_SWEEP_MAX_KEYS,
        },
        max_instances=1,
        replace_existing=True,
    )
    # One-shot on boot (optional)
    if RUN_ON_START:
        sched.add_job(
            run_ttl_sweep_job,
            trigger="date",
            run_date=None,  # now
            id="boot:ttl_sweep",
            kwargs={
                "pattern": TTL_SWEEP_PATTERN,
                "ttl_default": TTL_SWEEP_TTL,
                "max_keys": TTL_SWEEP_MAX_KEYS,
            },
            replace_existing=True,
        )

async def _amain() -> None:
    # Start metrics HTTP server
    start_http_server(SCHED_METRICS_PORT)
    log.info("Scheduler metrics on :%d", SCHED_METRICS_PORT)
    log.info("Using API_BASE_URL=%s", API_BASE_URL)

    # Optionally wait a bit for API to come up
    await wait_for_api_ready()

    # Build scheduler
    sched = AsyncIOScheduler(timezone=_tz())
    sched.start()
    log.info("Scheduler started.")

    # Add jobs
    _add_market_jobs(sched, _get_market_jobs())
    _add_ttl_sweep_job(sched)

    # Park forever
    while True:
        await asyncio.sleep(3600)

def main() -> None:
    try:
        asyncio.run(_amain())
    except (KeyboardInterrupt, SystemExit):
        log.info("Scheduler exiting...")

if __name__ == "__main__":
    main()