from pydantic import BaseModel
from typing import Any, Dict

class TickerData(BaseModel):
    symbol: str
    bid: float
    ask: float
    last: float
    timestamp: int

class Envelope(BaseModel):
    data: TickerData
    metadata: Dict[str, Any] = {}
    errors: Dict[str, Any] = {}