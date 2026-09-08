import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px

st.set_page_config(page_title="Crypto Radar AI", page_icon="ðŸ”¥", layout="wide")

st.title("ðŸ”¥ Crypto Radar AI")
st.caption("Research dashboard â€” signals are experimental and do not guarantee future price moves.")

# Coinbase Exchange public API is used instead of Binance because Binance can return
# HTTP 451 from some cloud hosting locations.
COINBASE_BASE = "https://api.exchange.coinbase.com"

PRODUCTS = {
    "BTCUSDT": "BTC-USD",
    "ETHUSDT": "ETH-USD",
    "SOLUSDT": "SOL-USD",
    "DOGEUSDT": "DOGE-USD",
}

@st.cache_data(ttl=60, show_spinner=False)
def get_ticker(product):
    url = f"{COINBASE_BASE}/products/{product}/ticker"
    r = requests.get(url, timeout=15, headers={"User-Agent": "CryptoRadarAI/1.0"})
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=60, show_spinner=False)
def get_candles(product, granularity=3600):
    url = f"{COINBASE_BASE}/products/{product}/candles"
    params = {"granularity": granularity}
    r = requests.get(url, params=params, timeout=15,
                     headers={"User-Agent": "CryptoRadarAI/1.0"})
    r.raise_for_status()

    raw = r.json()
    if not raw:
        return pd.DataFrame()

    # Coinbase returns: time, low, high, open, close, volume
    df = pd.DataFrame(
        raw, columns=["time", "low", "high", "open", "close", "volume"]
    )
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.sort_values("time").reset_index(drop=True)

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def radar_score(df):
    if len(df) < 60:
        return 0, []

    close = df["close"]
    volume = df["volume"]

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    current_rsi = float(rsi(close).iloc[-1])

    vol_ma = volume.rolling(20).mean().iloc[-1]
    vol_ratio = float(volume.iloc[-1] / vol_ma) if vol_ma and not pd.isna(vol_ma) else 1.0

    previous_high = close.rolling(48).max().shift(1).iloc[-1]
    breakout = bool(close.iloc[-1] > previous_high) if not pd.isna(previous_high) else False

    score = 0
    signals = []

    if ema12.iloc[-1] > ema26.iloc[-1]:
        score += 30
        signals.append("EMA bullish")
    else:
        signals.append("EMA bearish")

    if 50 <= current_rsi <= 70:
        score += 20
        signals.append("RSI healthy")
    elif current_rsi > 70:
        score += 8
        signals.append("RSI hot")
    else:
        signals.append("RSI weak")

    if vol_ratio >= 1.5:
        score += 25
        signals.append("Volume surge")
    elif vol_ratio >= 1.15:
        score += 12
        signals.append("Volume rising")

    if breakout:
        score += 25
        signals.append("48h breakout")

    return min(score, 100), signals

# ---------------- Dashboard ----------------

rows = []
data_by_symbol = {}

for symbol, product in PRODUCTS.items():
    try:
        ticker = get_ticker(product)
        df = get_candles(product)

        if df.empty:
            raise ValueError("No candle data returned")

        score, signals = radar_score(df)

        price = float(ticker["price"])
        volume_24h = float(ticker.get("volume", 0))

        # Approximate 24h percentage from the candle data when enough history exists.
        pct_24h = np.nan
        if len(df) >= 25:
            old = float(df["close"].iloc[-25])
            if old:
                pct_24h = (price / old - 1) * 100

        rows.append({
            "Symbol": symbol,
            "Price": price,
            "24h Volume": volume_24h,
            "24h %": pct_24h,
            "Radar Score": score,
            "Signals": ", ".join(signals),
        })
        data_by_symbol[symbol] = df

    except Exception as e:
        rows.append({
            "Symbol": symbol,
            "Price": np.nan,
            "24h Volume": np.nan,
            "24h %": np.nan,
            "Radar Score": 0,
            "Signals": f"Data error: {type(e).__name__}",
        })

out = pd.DataFrame(rows)
st.subheader("Opportunity Radar")
st.dataframe(
    out,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Price": st.column_config.NumberColumn(format="$%.6f"),
        "24h Volume": st.column_config.NumberColumn(format="$%.0f"),
        "24h %": st.column_config.NumberColumn(format="%.2f%%"),
        "Radar Score": st.column_config.ProgressColumn(min_value=0, max_value=100),
    },
)

st.divider()

symbol = st.selectbox("Choose a coin", list(PRODUCTS.keys()))

if symbol in data_by_symbol:
    df = data_by_symbol[symbol].copy()
    df["EMA12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["RSI"] = rsi(df["close"])

    st.subheader(f"ðŸ“ˆ {symbol} chart")

    chart_df = df.tail(120).copy()
    fig = px.line(
        chart_df,
        x="time",
        y=["close", "EMA12", "EMA26"],
        labels={"value": "Price (USD)", "time": "Time", "variable": ""},
    )
    fig.update_layout(legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

    latest_score, latest_signals = radar_score(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Radar Score", f"{latest_score}/100")
    c2.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
    vol_ma = df["volume"].rolling(20).mean().iloc[-1]
    vol_ratio = df["volume"].iloc[-1] / vol_ma if vol_ma else np.nan
    c3.metric("Volume Ratio", f"{vol_ratio:.2f}x")

    st.write("**Signals:** " + (" â€¢ ".join(latest_signals) if latest_signals else "Not enough data"))
else:
    st.error("Live market data could not be loaded. Refresh the page and check the app logs if this persists.")

st.divider()
st.info(
    "âš ï¸ This is a research/paper-trading tool. A high Radar Score is not a promise that a coin will pump. "
    "Before using real money, we should add historical backtesting, scam/rug checks, liquidity, news, "
    "on-chain activity, and proper risk management."
)

if st.button("ðŸ”„ Refresh data"):
    st.cache_data.clear()
    st.rerun()
