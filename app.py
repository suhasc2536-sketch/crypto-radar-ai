import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone
from pathlib import Path

st.set_page_config(page_title="Crypto Radar AI", page_icon="⚡", layout="wide")

BINANCE = "https://api.binance.com"
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

@st.cache_data(ttl=30)
def get_ticker(symbol):
    r = requests.get(f"{BINANCE}/api/v3/ticker/24hr", params={"symbol": symbol}, timeout=10)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=60)
def get_klines(symbol, interval="1h", limit=500):
    r = requests.get(f"{BINANCE}/api/v3/klines",
                     params={"symbol": symbol, "interval": interval, "limit": limit},
                     timeout=15)
    r.raise_for_status()
    raw = r.json()
    cols = ["open_time","open","high","low","close","volume","close_time",
            "quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["open","high","low","close","volume","quote_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df[["time","open","high","low","close","volume","quote_volume","trades"]]

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def score_market(df):
    x = df.copy()
    x["ema_fast"] = x["close"].ewm(span=12, adjust=False).mean()
    x["ema_slow"] = x["close"].ewm(span=26, adjust=False).mean()
    x["rsi"] = rsi(x["close"])
    x["ret_24h"] = x["close"].pct_change(24)
    x["vol_ratio"] = x["volume"] / x["volume"].rolling(24).mean()
    x["high_48"] = x["high"].rolling(48).max().shift(1)
    last = x.iloc[-1]

    score = 50
    reasons = []
    if last["ema_fast"] > last["ema_slow"]:
        score += 12; reasons.append("EMA trend bullish")
    else:
        score -= 12
    if last["vol_ratio"] > 1.5:
        score += 15; reasons.append(f"Volume spike {last['vol_ratio']:.1f}x")
    elif last["vol_ratio"] > 1.1:
        score += 7; reasons.append("Volume above average")
    if 55 <= last["rsi"] <= 72:
        score += 10; reasons.append("Healthy momentum RSI")
    elif last["rsi"] > 78:
        score -= 8; reasons.append("Overbought risk")
    if pd.notna(last["high_48"]) and last["close"] > last["high_48"]:
        score += 13; reasons.append("48-bar breakout")
    if pd.notna(last["ret_24h"]):
        if last["ret_24h"] > 0.05:
            score += 5; reasons.append("Strong 24h momentum")
        elif last["ret_24h"] < -0.05:
            score -= 8; reasons.append("Sharp 24h decline")
    return int(np.clip(score, 0, 100)), reasons, x

st.title("⚡ Crypto Radar AI")
st.caption("Research / paper-trading dashboard — not a guarantee of future returns.")

symbols = st.multiselect(
    "Markets",
    ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","DOGEUSDT","ADAUSDT","AVAXUSDT","LINKUSDT","SUIUSDT"],
    default=["BTCUSDT","ETHUSDT","SOLUSDT","DOGEUSDT"]
)

if st.button("🔄 Scan now", type="primary") or symbols:
    rows = []
    for s in symbols:
        try:
            ticker = get_ticker(s)
            df = get_klines(s)
            score, reasons, x = score_market(df)
            rows.append({
                "Symbol": s,
                "Price": float(ticker["lastPrice"]),
                "24h %": float(ticker["priceChangePercent"]),
                "24h Volume": float(ticker["quoteVolume"]),
                "Radar Score": score,
                "Signals": " • ".join(reasons) if reasons else "No strong signal"
            })
        except Exception as e:
            rows.append({"Symbol": s, "Price": np.nan, "24h %": np.nan,
                         "24h Volume": np.nan, "Radar Score": 0, "Signals": str(e)})

    out = pd.DataFrame(rows).sort_values("Radar Score", ascending=False)
    st.subheader("🔥 Opportunity Radar")
    st.dataframe(out, use_container_width=True, hide_index=True)

    if not out.empty:
        top = out.iloc[0]["Symbol"]
        st.subheader(f"📈 {top} chart")
        chart_df = get_klines(top, "1h", 300).set_index("time")
        st.line_chart(chart_df["close"])

st.divider()
st.subheader("📰 News integration")
st.info(
    "The next module should ingest timestamped crypto news and score each headline "
    "for catalyst strength and sentiment. Keep historical news timestamps separate "
    "from future price data so the backtest cannot look into the future."
)

st.subheader("🧪 Backtesting")
st.write("Run `python backtest.py` after downloading historical data with `python download_data.py`.")
st.write("The backtester uses chronological train/test splits and reports return, drawdown, "
         "Sharpe-like ratio, trade count, win rate and profit factor.")
