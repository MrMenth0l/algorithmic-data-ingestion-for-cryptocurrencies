from fastapi import FastAPI

app = FastAPI(
    title="Algo Data Ingestion",
    version="0.1.0",
    description="Unified API for market, on-chain, sentiment & news data"
)

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

# placeholder for a CCXT endpoint
@app.get("/v1/market/ticker")
async def get_ticker(symbol: str):
    return {"data": {"symbol": symbol, "price": None}, "metadata": {}}