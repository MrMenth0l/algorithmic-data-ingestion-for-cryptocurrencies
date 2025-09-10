from __future__ import annotations
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.ingestion_service.config import settings
from app.ingestion_service import ml_utils
from prometheus_client import Counter, Histogram
from app.ingestion_service.utils import _METRICS_REGISTRY
import time

router = APIRouter()

INFER_REQUESTS = Counter(
    "ml_infer_requests_total",
    "Total ML inference requests",
    labelnames=("model",),
    registry=_METRICS_REGISTRY,
)
INFER_ERRORS = Counter(
    "ml_infer_errors_total",
    "Total ML inference errors",
    labelnames=("model", "type"),
    registry=_METRICS_REGISTRY,
)
INFER_DURATION = Histogram(
    "ml_infer_duration_seconds",
    "ML inference latency seconds",
    labelnames=("model",),
    registry=_METRICS_REGISTRY,
    buckets=(0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)


class PredictRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=256)


@router.post("/ml/sentiment/predict")
async def predict_sentiment(req: PredictRequest) -> Dict[str, Any]:
    if not settings.ML_SENTIMENT_ENABLED:
        raise HTTPException(status_code=503, detail="ML sentiment disabled")
    texts = [t if isinstance(t, str) else str(t) for t in req.texts]
    if any(len(t) > 2000 for t in texts):
        raise HTTPException(status_code=422, detail="Text too long (max 2000 chars)")

    model_id = settings.SENTIMENT_MODEL_ID
    INFER_REQUESTS.labels(model=model_id).inc()
    t0 = time.perf_counter()
    try:
        items = await ml_utils.predict(texts)
        INFER_DURATION.labels(model=model_id).observe(time.perf_counter() - t0)
        return {"items": items}
    except Exception as e:
        INFER_ERRORS.labels(model=model_id, type=type(e).__name__).inc()
        raise HTTPException(status_code=500, detail=str(e))

