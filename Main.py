# pip install ccxt ccxtpro
import asyncio
import ccxt.pro as ccxt

async def main():
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',  # USDT‐margined perpetual futures
        },
    })
    symbol = 'BTC/USDT'
    timeframe = '1m'
    print(f"Streaming {symbol} @ {timeframe}")
    try:
        while True:
            ohlcv = await exchange.watch_ohlcv(symbol, timeframe)
            # ohlcv: [[ts, open, high, low, close, volume], …]
            ts, o, h, l, c, v = ohlcv[-1]
            print(f"{exchange.iso8601(ts)}  O:{o:.2f}  H:{h:.2f}  L:{l:.2f}  C:{c:.2f}  V:{v:.6f}")
            # → Hook this data into your feature‐gen pipeline…

    except Exception as e:
        print(f"Stream error: {e}")
    finally:
        await exchange.close()

if __name__ == '__main__':
    asyncio.run(main())