from __future__ import annotations
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional

from app.ingestion_service.config import settings

_PIPELINE = None
_EXECUTOR: Optional[ThreadPoolExecutor] = None

def _ensure_hf_cache() -> None:
    if settings.HF_HOME:
        os.environ.setdefault("HF_HOME", settings.HF_HOME)

def _load_pipeline():
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE
    _ensure_hf_cache()
    from transformers import pipeline
    model_id = settings.SENTIMENT_MODEL_ID
    # return_all_scores=True to get both pos/neg
    _PIPELINE = pipeline("sentiment-analysis", model=model_id, return_all_scores=True)
    return _PIPELINE

def _ensure_executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        max_workers = max(1, int(settings.ML_MAX_WORKERS or 4))
        _EXECUTOR = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ml-infer")
    return _EXECUTOR

def _normalize_result(scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    # For DistilBERT SST-2: labels are 'NEGATIVE' and 'POSITIVE'
    p_pos = 0.0
    p_neg = 0.0
    for x in scores:
        label = str(x.get("label", "")).lower()
        if "pos" in label:
            p_pos = float(x.get("score", 0.0))
        elif "neg" in label:
            p_neg = float(x.get("score", 0.0))
    # signed score in [-1,1]
    score_signed = max(-1.0, min(1.0, (p_pos - p_neg)))  # since p_pos+p_neg~=1
    # simple 3-class mapping with margins
    if p_pos >= 0.6:
        label = "positive"
    elif p_pos <= 0.4:
        label = "negative"
    else:
        label = "neutral"
    score = max(p_pos, p_neg)
    return {"label": label, "score": score, "score_signed": score_signed}

def predict_sync(texts: List[str]) -> List[Dict[str, Any]]:
    pipe = _load_pipeline()
    # transformers supports list batching; enable truncation to avoid overlong inputs
    raw = pipe(texts, truncation=True)
    # raw is List[List[{label,score}]] because return_all_scores=True
    return [_normalize_result(r) for r in raw]

async def predict(texts: List[str]) -> List[Dict[str, Any]]:
    if not texts:
        return []
    loop = asyncio.get_running_loop()
    ex = _ensure_executor()
    return await loop.run_in_executor(ex, predict_sync, texts)

def shutdown() -> None:
    global _EXECUTOR
    ex = _EXECUTOR
    _EXECUTOR = None
    if ex is not None:
        ex.shutdown(wait=False, cancel_futures=True)

