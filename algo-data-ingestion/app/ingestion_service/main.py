from fastapi import FastAPI, Request
from .routes import router
from prometheus_client import make_asgi_app
from fastapi.middleware.cors import CORSMiddleware
import logging
from app.ingestion_service.utils import _METRICS_REGISTRY
from app.ingestion_service.config import settings
from contextlib import asynccontextmanager
from app.features.ingestion.news_client import NewsClient
from app.features.ingestion.ccxt_client import CCXTClient
from app.features.ingestion.social_client import SocialClient
from app.features.ingestion.onchain_client import OnchainClient

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create shared NewsClient instance
    app.state.news_client = NewsClient()
    # Create shared CCXT client (async)
    app.state.ccxt_client = CCXTClient(
        exchange_name=getattr(settings, "exchange_name", "binance"),
        api_key=getattr(settings, "exchange_api_key", None),
        secret=getattr(settings, "exchange_api_secret", None),
    )
    app.state.onchain_client = OnchainClient()
    app.state.social_client = SocialClient()
    try:
        yield
    finally:
        # Graceful shutdown
        await app.state.social_client.aclose()
        await app.state.onchain_client.aclose()
        await app.state.ccxt_client.aclose()
        await app.state.news_client.aclose()

app = FastAPI(lifespan=lifespan)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials = True,
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

def get_news(request: Request) -> NewsClient:
    return request.app.state.news_client

def get_ccxt(request: Request) -> CCXTClient:
    return request.app.state.ccxt_client

def get_onchain(request: Request) -> OnchainClient:
    return request.app.state.onchain_client

def get_social(request: Request) -> SocialClient:
    return request.app.state.social_client

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.ingest_host,
        port=settings.ingest_port,
        reload=True
    )
