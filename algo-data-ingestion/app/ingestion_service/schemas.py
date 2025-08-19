# app/ingestion_service/schemas.py
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class MarketIngestRequest(BaseModel):
    symbol: str
    granularity: str = Field(..., description="e.g. 1m, 5m, 1h")
    limit: int = Field(100, ge=1, le=1000)

class OnchainIngestRequest(BaseModel):
    # Glassnode path
    symbol: Optional[str] = None
    metric: Optional[str] = None
    days: Optional[int] = 1

    # Covalent path
    chain_id: Optional[int] = None
    address: Optional[str] = None

class SocialIngestRequest(BaseModel):
    query: str
    since: Optional[datetime] = None   # <- OPTIONAL
    until: Optional[datetime] = None   # <- OPTIONAL
    max_results: int = Field(100, ge=1, le=100)

class NewsIngestRequest(BaseModel):
    source_type: str  # "api" or "rss"
    category: Optional[str] = None
    feed_url: Optional[str] = None

class FeatureVector(BaseModel):
    symbol: str
    timestamp: int
    payload: dict = Field(default_factory=dict)
