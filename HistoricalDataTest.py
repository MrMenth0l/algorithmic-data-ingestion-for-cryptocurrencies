import ccxt
import pandas as pd
from datetime import datetime
import talib as ta
import vectorbt as vbt
import itertools
import optuna



# --- Live Order Book Imbalance Streaming ---
import ccxt.pro as ccxt_pro
import asyncio



# 1. Instantiate the REST client
exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'future'},
})

# 2. Fetch historical OHLCV

symbol    = 'BTC/USDT'
timeframe = '5m'
# 2. Fetch historical OHLCV for the last 30 days in batches
now = exchange.milliseconds()
since = now - 30 * 24 * 60 * 60 * 1000  # 7 days ago
all_ohlcv = []

while True:
    batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1500)
    if not batch:
        break
    all_ohlcv.extend(batch)
    # move since to last timestamp + 1 ms
    since = batch[-1][0] + 1
    # if we received fewer than the limit, we're done
    if len(batch) < 1500:
        break

ohlcv = all_ohlcv
print(f"Fetched {len(ohlcv)} candles spanning {len(ohlcv)/1440:.1f} days")

# 3. Build DataFrame
df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

print(df.tail())

# 3b. Compute 1-hour SMA50 trend filter
df_hour = df['close'].resample('1H').last()
sma1h50 = ta.SMA(df_hour.values, timeperiod=50)
# Align back to minute index
df['SMA1H50'] = pd.Series(sma1h50, index=df_hour.index).reindex(df.index, method='ffill')
df['trend_up'] = df['SMA1H50'].diff() > 0

# assuming df is your DataFrame from HistoricalDataTest.py
df['SMA20'] = ta.SMA(df['close'], timeperiod=20)
df['RSA40'] = ta.RSI(df['close'], timeperiod=14)

# Add after df['RSA40'] = ...
df['Vol20'] = df['volume'].rolling(window=20).mean()

# 20-period rolling VWAP
df['typical'] = (df['high'] + df['low'] + df['close']) / 3
df['pv'] = df['typical'] * df['volume']
df['pv_cum'] = df['pv'].rolling(window=20).sum()
df['vol_cum'] = df['volume'].rolling(window=20).sum()
df['VWAP20'] = df['pv_cum'] / df['vol_cum']

# ATR-based stop-loss calculations
df['ATR14'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
df['sl_rel'] = df['ATR14'] / df['close']



print(df[['close','SMA20','RSA40']].tail(10))

# Entry: oversold momentum above trend OR SMA breakout, only on volume spike and in 1h uptrend and above VWAP20
long_entry = (
    (
        (df['close'] > df['SMA20']) &
        (df['RSA40'] < 40) &
        (df['volume'] > df['Vol20'])
    ) | (
        (df['close'].shift(1) <= df['SMA20'].shift(1)) &
        (df['close'] > df['SMA20']) &
        (df['volume'] > df['Vol20'])
    )
) & df['trend_up'] & (df['close'] > df['VWAP20'])
long_exit  = (df['close'] < df['SMA20']) | (df['RSA40'] > 70)

# convert to integer signal: 1 for long, 0 for flat
df['signal'] = 0
df.loc[long_entry, 'signal'] = 1
df.loc[long_exit,  'signal'] = 0

# forward-fill positions
df['position'] = df['signal'].ffill().fillna(0)
print(df[['close','SMA20','RSA40','position']].tail(10))

# 1. Prepare your series
price = df['close']

# 2. Define entries/exits from your signals
entries = df['signal'] == 1
exits   = df['signal'] == 0

print(vbt.__version__)
# 3. Run the backtest
pf = vbt.Portfolio.from_signals(
    price,
    entries,
    exits,
    init_cash=10_000,
    fees=0.0005,
    slippage=0.0005,
    tp_stop=0.01,            # 1% profit target
    sl_stop=0.3 * df['sl_rel'],  # 0.3 ATR static stop-loss
    freq='5T'
)

# 4. Print performance metrics
print(pf.stats())

# 5. Plot equity curve & drawdown
fig = pf.plot(title='SMA20+RSA40 Strategy on BTC/USDT Perps')
fig.show()

# --- Optuna Hyperparameter Optimization ---
def objective(trial):
    sma_period = trial.suggest_int('sma_period', 20, 50, step=5)
    rsi_period = trial.suggest_int('rsi_period', 30, 50, step=5)
    tp = trial.suggest_float('tp', 0.005, 0.02, step=0.005)
    sl_mult = trial.suggest_float('sl_mult', 0.2, 1.0, step=0.1)

    # Recompute indicators for this trial
    df['SMAtmp'] = ta.SMA(df['close'], timeperiod=sma_period)
    df['RSItmp'] = ta.RSI(df['close'], timeperiod=rsi_period)
    entries = (
        (
            (df['close'] > df['SMAtmp']) &
            (df['RSItmp'] < 40) &
            (df['volume'] > df['Vol20'])
        ) | (
            (df['close'].shift(1) <= df['SMAtmp'].shift(1)) &
            (df['close'] > df['SMAtmp']) &
            (df['volume'] > df['Vol20'])
        )
    ) & df['trend_up'] & (df['close'] > df['VWAP20'])
    exits = (df['close'] < df['SMAtmp']) | (df['RSItmp'] > 70)

    pf = vbt.Portfolio.from_signals(
        df['close'],
        entries,
        exits,
        init_cash=10_000,
        fees=0.0005,
        slippage=0.0005,
        tp_stop=tp,
        sl_stop=sl_mult * df['sl_rel'],
        freq='5T'
    )
    return pf.stats()['Sharpe Ratio']

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("=== Optuna Best Params ===")
print(study.best_params)
print("=== Best Sharpe ===")
print(study.best_value)


# --- Live Order-Book Imbalance Confirmation ---
async def live_with_orderbook_imbalance():
    exchange_live = ccxt_pro.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'},
    })
    symbol = 'BTC/USDT'
    timeframe = '5m'
    imbalance_threshold = 0.2  # require at least 20% more bid volume than ask volume

    print(f"Streaming {symbol}@{timeframe} for order-book imbalance > {imbalance_threshold}")
    try:
        while True:
            ohlcv = await exchange_live.watch_ohlcv(symbol, timeframe)
            ts, o, h, l, c, v = ohlcv[-1]
            # Fetch top 10 levels of order book
            ob = await exchange_live.watch_order_book(symbol, limit=10)
            bid_vol = sum([level[1] for level in ob['bids']])
            ask_vol = sum([level[1] for level in ob['asks']])
            imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) else 0
            tag = '🟢' if imbalance > imbalance_threshold else '🔴'
            print(f"{exchange_live.iso8601(ts)} {tag} Imbalance: {imbalance:.2f}")
    except Exception as e:
        print(f"Live stream error: {e}")
    finally:
        await exchange_live.close()

if __name__ == '__main__':
    # Uncomment the next line to run the live imbalance streamer
    asyncio.run(live_with_orderbook_imbalance())