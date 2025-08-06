from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict

class MarketIngestRequest(BaseModel):
    symbol: str
    granularity: str
    limit: int = 100

class OnchainIngestRequest(BaseModel):
    source: str
    chain_id: int
    symbol: Optional[str] = None
    address: Optional[str] = None
    metric: Optional[str] = None
    days: int = Field(default=1, ge=1)

class SocialIngestRequest(BaseModel):
    query: str
    since: datetime
    until: datetime
    max_results: int = Field(default=10, ge=1, le=100)

class NewsIngestRequest(BaseModel):
    source_type: str  # "api" or "rss"
    feed_url: Optional[str] = None
    category: Optional[str] = None

class FeatureVector(BaseModel):
    symbol: str
    timestamp: int
    features: Dict[str, float]
