import pytest
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace

import httpx

# Async context manager helper for httpx.AsyncClient in tests
class FakeClient:
    def __init__(self, get_fn):
        self.get = get_fn

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

# Adjust this import to your project’s layout:
from app.adapters.onchain_adapter import (
    fetch_glassnode,
    fetch_covalent,
    GLASSNODE_KEY,
    COVALENT_KEY,
)

# A helper dummy response
class DummyResponse:
    def __init__(self, status_code: int, json_data=None):
        self.status_code = status_code
        self._json = json_data or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=None)

    def json(self):
        return self._json

# Fixture to ensure API keys are set and import picks them up
@pytest.fixture(autouse=True)
def fix_env(monkeypatch):
    monkeypatch.setenv("GLASSNODE_API_KEY", "glass-token")
    monkeypatch.setenv("COVALENT_API_KEY", "covalent-token")
    import importlib
    import app.adapters.onchain_adapter as onchain_mod
    importlib.reload(onchain_mod)
    yield

@pytest.mark.asyncio
async def test_fetch_glassnode(monkeypatch):
    # Prepare fake data: two points [timestamp_ms, value]
    now_ms = int(datetime.utcnow().timestamp() * 1000)
    one_day_ago_ms = int((datetime.utcnow() - timedelta(days=1)).timestamp() * 1000)
    fake_payload = [
        [one_day_ago_ms, 42.0],
        [now_ms, 84.0],
    ]

    async def fake_get(url, params):
        # verify URL and params
        assert url.startswith("https://api.glassnode.com/v1/metrics")
        assert params["api_key"] == "glass-token"
        assert "a" in params and params["a"] == "BTC"
        return DummyResponse(200, json_data=fake_payload)

    # Patch AsyncClient
    monkeypatch.setattr(
        "app.adapters.onchain_adapter.httpx.AsyncClient",
        lambda *args, **kwargs: FakeClient(fake_get)
    )

    df = await fetch_glassnode("BTC", "address_exits", days=1)
    # Expect a DataFrame with two rows, correct columns and dtypes
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"source", "symbol", "metric", "timestamp", "value"}
    assert df.shape[0] == 2
    assert df["source"].unique().tolist() == ["glassnode"]
    assert df["symbol"].unique().tolist() == ["BTC"]
    assert df["metric"].unique().tolist() == ["address_exits"]
    # timestamp is datetime64[ns, UTC]
    assert pd.api.types.is_datetime64_ns_dtype(df["timestamp"])
    # values match
    assert df["value"].tolist() == [42.0, 84.0]

@pytest.mark.asyncio
async def test_fetch_covalent(monkeypatch):
    fake_items = [
        {
            "contract_ticker_symbol": "ETH",
            "balance": "1000000000000000000",  # 1 ETH in wei
            "contract_decimals": 18,
        },
        {
            "contract_ticker_symbol": "USDC",
            "balance": "5000000",  # 5 USDC with 6 decimals
            "contract_decimals": 6,
        },
    ]

    async def fake_get(url, params):
        # verify URL contains the chain_id and address
        assert "covalenthq.com/v1/1/address/0xabc" in url
        assert params["key"] == "covalent-token"
        return DummyResponse(200, json_data={"data": {"items": fake_items}})

    monkeypatch.setattr(
        "app.adapters.onchain_adapter.httpx.AsyncClient",
        lambda *args, **kwargs: FakeClient(fake_get)
    )

    df = await fetch_covalent(chain_id=1, address="0xabc")
    # Validate DataFrame
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"source", "symbol", "metric", "timestamp", "value"}
    assert df.shape[0] == 2
    # Check conversions
    eth_row = df[df["symbol"] == "ETH"].iloc[0]
    assert eth_row["source"] == "covalent"
    assert eth_row["metric"] == "balance"
    # value should be 1.0
    assert pytest.approx(eth_row["value"], rel=1e-6) == 1.0

    usdc_row = df[df["symbol"] == "USDC"].iloc[0]
    assert pytest.approx(usdc_row["value"], rel=1e-6) == 5.0