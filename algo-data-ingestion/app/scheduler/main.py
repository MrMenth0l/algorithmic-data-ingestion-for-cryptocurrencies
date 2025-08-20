# app/scheduler/main.py
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.triggers.cron import CronTrigger
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# ---------- Config ----------

def _getenv(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v not in (None, "", "None") else default

API_BASE_URL = _getenv("API_BASE_URL", "http://ingestion-api:8000")
ADMIN_TOKEN  = _getenv("ADMIN_TOKEN")
TIMEZONE     = _getenv("SCHED_TZ", "UTC")

# JSON string: list of jobs
# e.g. [{"exchange":"binance","symbol":"BTC/USDT","timeframe":"1m","lookback_minutes":15,"cron":"*/5 * * * *"}]
MARKET_JOBS_JSON = _getenv("MARKET_JOBS", "[]")

# TTL Sweep config
TTL_SWEEP_CRON     = _getenv("TTL_SWEEP_CRON", "*/15 * * * *")  # every 15m
TTL_SWEEP_PATTERN  = _getenv("TTL_SWEEP_PATTERN", "features:*")
TTL_SWEEP_TTL_DEF  = _getenv("TTL_SWEEP_TTL", None)  # seconds or None

# Metrics endpoint for scheduler itself
METRICS_PORT = int(_getenv("SCHED_METRICS_PORT", "9002"))

RUN_ON_START = _getenv("RUN_ON_START", "1") == "1"

# ---------- Logging ----------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [scheduler] %(message)s",
)
log = logging.getLogger("scheduler")

# ---------- Prometheus ----------
start_http_server(METRICS_PORT)
SCHED_UP = Gauge("scheduler_up", "Scheduler liveness")
SCHED_UP.set(1)

JOB_RUNS = Counter("scheduler_job_runs_total", "Number of scheduler job runs", ["job"])
JOB_FAILS = Counter("scheduler_job_failures_total", "Number of scheduler job failures", ["job"])
JOB_LAT = Histogram("scheduler_job_latency_seconds", "Latency of scheduler jobs", ["job"])


# ---------- HTTP helpers ----------

def _auth_headers() -> Dict[str, str]:
    # Admin router accepts X-Admin-Token or Bearer
    return {
        "Authorization": f"Bearer {ADMIN_TOKEN}" if ADMIN_TOKEN else "",
        "X-Admin-Token": ADMIN_TOKEN or "",
        "Content-Type": "application/json",
    }

async def _post_admin(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{API_BASE_URL}{path}"
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Admin endpoints take query params (FastAPI Query)
        resp = await client.post(url, headers=_auth_headers(), params=params)
        resp.raise_for_status()
        return resp.json()


# ---------- Jobs ----------

async def run_market_backfill_job(job: Dict[str, Any]) -> None:
    """
    Job fields: exchange, symbol, timeframe, lookback_minutes
    """
    label = f"backfill:{job.get('exchange')}:{job.get('symbol')}:{job.get('timeframe')}"
    with JOB_LAT.labels(job=label).time():
        try:
            payload = {
                "exchange": job["exchange"],
                "symbol": job["symbol"],
                "timeframe": job.get("timeframe", "1m"),
                "lookback_minutes": int(job.get("lookback_minutes", 60)),
            }
            out = await _post_admin("/admin/backfill/market", payload)
            JOB_RUNS.labels(job=label).inc()
            log.info("Market backfill ok %s -> %s", label, out)
        except Exception as e:
            JOB_FAILS.labels(job=label).inc()
            log.exception("Market backfill failed %s: %s", label, e)

async def run_ttl_sweep_job() -> None:
    label = "ttl_sweep"
    with JOB_LAT.labels(job=label).time():
        try:
            params = {
                "pattern": TTL_SWEEP_PATTERN,
            }
            if TTL_SWEEP_TTL_DEF is not None:
                params["ttl_default"] = int(TTL_SWEEP_TTL_DEF)
            out = await _post_admin("/admin/features/ttl-sweep", params)
            JOB_RUNS.labels(job=label).inc()
            log.info("TTL sweep ok -> %s", out)
        except Exception as e:
            JOB_FAILS.labels(job=label).inc()
            log.exception("TTL sweep failed: %s", e)


# ---------- Bootstrap ----------

def _parse_market_jobs(raw: str) -> List[Dict[str, Any]]:
    try:
        data = json.loads(raw or "[]")
        if not isinstance(data, list):
            raise ValueError("MARKET_JOBS must be a JSON list")
        return data
    except Exception as e:
        log.warning("Failed to parse MARKET_JOBS JSON; defaulting to []: %s", e)
        return []


async def _startup_check() -> None:
    if not ADMIN_TOKEN:
        log.error("ADMIN_TOKEN is not set; scheduler cannot authenticate to admin endpoints.")
        # Don't exit hard; keep metrics server alive to expose failure
    # Basic ping of api root/health (best-effort)
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{API_BASE_URL}/health")
            log.info("API /health -> %s", r.status_code)
    except Exception as e:
        log.warning("API /health check failed: %s", e)


def _add_jobs(sched: AsyncIOScheduler) -> None:
    jobs = _parse_market_jobs(MARKET_JOBS_JSON)

    # Market jobs
    for j in jobs:
        cron = j.get("cron", "*/5 * * * *")  # default every 5 minutes
        sched.add_job(
            run_market_backfill_job,
            CronTrigger.from_crontab(cron, timezone=TIMEZONE),
            kwargs={"job": j},
            name=f"backfill:{j.get('exchange')}:{j.get('symbol')}:{j.get('timeframe')}",
            misfire_grace_time=120,
            coalesce=True,
            max_instances=1,
        )
        if RUN_ON_START:
            # fire once shortly after boot to verify
            sched.add_job(
                run_market_backfill_job,
                "date",
                kwargs={"job": j},
                name=f"boot:{j.get('exchange')}:{j.get('symbol')}:{j.get('timeframe')}",
                run_date=None,  # ASAP
            )

    # TTL sweep
    if TTL_SWEEP_CRON:
        sched.add_job(
            run_ttl_sweep_job,
            CronTrigger.from_crontab(TTL_SWEEP_CRON, timezone=TIMEZONE),
            name="ttl_sweep",
            misfire_grace_time=120,
            coalesce=True,
            max_instances=1,
        )
        if RUN_ON_START:
            sched.add_job(run_ttl_sweep_job, "date", name="boot:ttl_sweep")

async def main() -> None:
    await _startup_check()
    jobstores = {"default": MemoryJobStore()}
    sched = AsyncIOScheduler(jobstores=jobstores, timezone=TIMEZONE)
    _add_jobs(sched)
    sched.start()
    log.info("Scheduler started. Metrics on :%d", METRICS_PORT)
    try:
        # Keep the loop alive
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        sched.shutdown(wait=False)
        log.info("Scheduler stopped.")

if __name__ == "__main__":
    asyncio.run(main())