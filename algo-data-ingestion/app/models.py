#Legacy file, not in actual implementation.

from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Literal
from datetime import datetime

class TickerData(BaseModel):
    symbol: str
    bid: float
    ask: float
    last: float
    timestamp: datetime

class Envelope(BaseModel):
    data: TickerData
    metadata: Dict[str, Any] = {}
    errors: Dict[str, Any] = {}

class SentimentItem(BaseModel):
    source: Literal["twitter", "reddit"]
    id: str
    text: str
    user: str
    created_at: datetime
    sentiment_score: float  # placeholder, set to 0.0 for now

class SentimentEnvelope(BaseModel):
    data: List[SentimentItem]
    metadata: Dict[str, Any] = {}
    errors: Dict[str, Any] = {}
class NewsItem(BaseModel):
    id: str
    title: str
    url: str
    source: str
    author: Optional[str] = None
    description: Optional[str] = None
    published_at: datetime

class NewsEnvelope(BaseModel):
    data: List[NewsItem]
    metadata: Dict[str, Any] = {}
    errors: Dict[str, Any] = {}

class OnchainItem(BaseModel):
    source: Literal["glassnode", "covalent"]
    symbol: str             # e.g. "BTC"
    metric: str             # e.g. "exchange_netflow"
    value: float
    timestamp: datetime          # datetime of the metric

class OnchainEnvelope(BaseModel):
    data: List[OnchainItem]
    metadata: Dict[str, Any] = {}
    errors: Dict[str, Any] = {}