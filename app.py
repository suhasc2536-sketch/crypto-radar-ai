import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go

COINBASE_BASE = "https://api.exchange.coinbase.com"

PRODUCTS = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "SOL-USD": "Solana",
    "DOGE-USD": "Dogecoin",
}

st.set_page_config(
    page_title="Crypto Radar AI",
    layout="wide",
)

st.title("Crypto Radar AI")
st.caption(
    "Research dashboard - experimental signals only. "
    "A high score is not a guarantee of a future price increase."
)


@st.cache_data(ttl=60)
def get_ticker(product_id):
    url = f"{COINBASE_BASE}/products/{product_id}/ticker"
    response = requests.get(
        url,
        timeout=15,
        headers={"User-Agent": "CryptoRadarAI/1.0"},
    )
    response.raise_for_status()
    data = response.json()

    price = float(data["price"])
    base_volume_24h = float(data["volume"])

    return {
        "price": price,
        "base_volume_24h": base_volume_24h,
        "usd_volume_24h": price * base_volume_24h,
    }


@st.cache_data(ttl=60)
def get_candles(product_id, granularity=3600):
    url = f"{COINBASE_BASE}/products/{product_id}/candles"
    params = {
        "granularity": granularity,
    }

    response = requests.get(
        url,
        params=params,
        timeout=15,
        headers={"User-Agent": "CryptoRadarAI/1.0"},
    )
    response.raise_for_status()

    rows = response.json()

    if not rows:
        return pd.DataFrame()

    # Coinbase returns:
    # [timestamp, low, high, open, close, volume]
    df = pd.DataFrame(
        rows,
        columns=[
            "timestamp",
            "low",
            "high",
            "open",
            "close",
            "volume",
        ],
    )

    for column in ["low", "high", "open", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="s",
        utc=True,
    )

    df = df.dropna().sort_values("timestamp").reset_index(drop=True)

    return df


def calculate_rsi(series, period=14):
    delta = series.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = losses.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50)


def analyze(df):
    if len(df) < 60:
        return {
            "score": 0,
            "signals": ["Not enough historical candles"],
            "rsi": np.nan,
            "volume_ratio": np.nan,
            "breakout": False,
            "ema_bullish": False,
        }

    data = df.copy()

    data["ema12"] = data["close"].ewm(
        span=12,
        adjust=False,
    ).mean()

    data["ema26"] = data["close"].ewm(
        span=26,
        adjust=False,
    ).mean()

    data["rsi"] = calculate_rsi(data["close"])

    # Compare the latest hourly volume with the previous 24-hour
    # average volume.
    data["volume_avg_24"] = (
        data["volume"]
        .rolling(24)
        .mean()
        .shift(1)
    )

    data["volume_ratio"] = (
        data["volume"] / data["volume_avg_24"]
    )

    # Breakout means the latest close is above the highest close
    # seen during the previous 48 hourly candles.
    data["previous_48_high"] = (
        data["close"]
        .rolling(48)
        .max()
        .shift(1)
    )

    latest = data.iloc[-1]

    score = 0
    signals = []

    ema_bullish = latest["ema12"] > latest["ema26"]
    rsi_value = float(latest["rsi"])
    volume_ratio = float(latest["volume_ratio"])

    if ema_bullish:
        score += 30
        signals.append("Bullish EMA trend")
    else:
        signals.append("EMA trend is not bullish")

    if 50 <= rsi_value < 70:
        score += 20
        signals.append("RSI supports momentum")
    elif 70 <= rsi_value < 80:
        score += 10
        signals.append("RSI is strong but becoming extended")
    elif rsi_value < 30:
        score += 5
        signals.append("RSI is oversold")
    else:
        signals.append("RSI is not in the preferred zone")

    if volume_ratio >= 1.50:
        score += 25
        signals.append("Strong volume expansion")
    elif volume_ratio >= 1.15:
        score += 12
        signals.append("Moderate volume expansion")
    else:
        signals.append("No meaningful volume expansion")

    breakout = (
        pd.notna(latest["previous_48_high"])
        and latest["close"] > latest["previous_48_high"]
    )

    if breakout:
        score += 25
        signals.append("48-hour breakout")
    else:
        signals.append("No 48-hour breakout")

    return {
        "score": int(score),
        "signals": signals,
        "rsi": rsi_value,
        "volume_ratio": volume_ratio,
        "breakout": bool(breakout),
        "ema_bullish": bool(ema_bullish),
    }


def format_price(price):
    if price >= 1000:
        return f"${price:,.0f}"
    if price >= 1:
        return f"${price:,.2f}"
    return f"${price:,.5f}"


def format_usd(value):
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"${value / 1_000:.2f}K"
    return f"${value:,.0f}"


# -------------------------------------------------------------------
# Market overview
# -------------------------------------------------------------------

st.subheader("Market overview")

overview_rows = []

for product_id, name in PRODUCTS.items():
    try:
        ticker = get_ticker(product_id)
        candles = get_candles(product_id)
        analysis = analyze(candles)

        price = ticker["price"]

        if len(candles) >= 25:
            old_price = float(candles.iloc[-25]["close"])
            change_24h = ((price / old_price) - 1) * 100
        else:
            change_24h = np.nan

        overview_rows.append(
            {
                "Asset": name,
                "Market": product_id,
                "Price": format_price(price),
                "24h Change": (
                    f"{change_24h:+.2f}%"
                    if pd.notna(change_24h)
                    else "N/A"
                ),
                "24h USD Volume": format_usd(
                    ticker["usd_volume_24h"]
                ),
                "Radar Score": analysis["score"],
            }
        )

    except Exception as exc:
        overview_rows.append(
            {
                "Asset": name,
                "Market": product_id,
                "Price": "Error",
                "24h Change": "Error",
                "24h USD Volume": "Error",
                "Radar Score": 0,
            }
        )

overview_df = pd.DataFrame(overview_rows)

st.dataframe(
    overview_df,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Data source: Coinbase public market-data API. "
    "Volume is converted to approximate USD notional using the latest price."
)


# -------------------------------------------------------------------
# Detailed analysis
# -------------------------------------------------------------------

st.subheader("Detailed analysis")

selected_product = st.selectbox(
    "Select an asset",
    list(PRODUCTS.keys()),
    format_func=lambda x: f"{PRODUCTS[x]} ({x})",
)

try:
    ticker = get_ticker(selected_product)
    candles = get_candles(selected_product)
    analysis = analyze(candles)

    if candles.empty:
        st.error("No candle data was returned.")
        st.stop()

    current_price = ticker["price"]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Current price",
            format_price(current_price),
        )

    with col2:
        rsi_display = (
            f"{analysis['rsi']:.1f}"
            if pd.notna(analysis["rsi"])
            else "N/A"
        )
        st.metric("RSI", rsi_display)

    with col3:
        volume_display = (
            f"{analysis['volume_ratio']:.2f}x"
            if pd.notna(analysis["volume_ratio"])
            else "N/A"
        )
        st.metric("Volume vs 24h avg", volume_display)

    with col4:
        st.metric(
            "Radar Score",
            f"{analysis['score']}/100",
        )

    # Candlestick chart
    chart_data = candles.tail(120)

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=chart_data["timestamp"],
                open=chart_data["open"],
                high=chart_data["high"],
                low=chart_data["low"],
                close=chart_data["close"],
                name=selected_product,
            )
        ]
    )

    fig.update_layout(
        height=520,
        xaxis_title="Time (UTC)",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=20, b=20),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.subheader("Why the score is what it is")

    for signal in analysis["signals"]:
        st.write(f"- {signal}")

    st.info(
        "Score construction: EMA trend 30 points, RSI 20 points, "
        "volume expansion 25 points, and 48-hour breakout 25 points. "
        "This is a transparent rule-based research model, not a "
        "guarantee or financial advice."
    )

except requests.RequestException as exc:
    st.error(
        "The market-data provider could not be reached right now."
    )
    st.code(str(exc))

except Exception as exc:
    st.error("The dashboard encountered an unexpected error.")
    st.code(str(exc))


# -------------------------------------------------------------------
# Controls
# -------------------------------------------------------------------

if st.button("Refresh market data"):
    st.cache_data.clear()
    st.rerun()

st.divider()

st.caption(
    "Research and paper-trading tool. Do not connect a wallet or "
    "place real trades based only on this score."
)
