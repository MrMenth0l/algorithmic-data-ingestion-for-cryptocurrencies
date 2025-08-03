from fastapi import FastAPI
from .routes import router
from prometheus_client import make_asgi_app
from fastapi.middleware.cors import CORSMiddleware
import logging
from app.ingestion_service.utils import _METRICS_REGISTRY

app = FastAPI(title="Raw Data Ingestion Service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router, prefix="/ingest")
# Expose Prometheus metrics at /metrics
app.mount("/metrics", make_asgi_app(registry=_METRICS_REGISTRY))

@app.get("/")
async def root():
    return {"service": "raw-data-ingestion", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.on_event("startup")
async def on_startup():
    logging.info("Starting Raw Data Ingestion Service")

@app.on_event("shutdown")
async def on_shutdown():
    logging.info("Shutting down Raw Data Ingestion Service")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
