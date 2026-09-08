import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ============================================================
# CRYPTO AI TRADING DESK - FREE MVP
# SCAN -> ANALYZE -> RISK -> PLAN
#
# NO API KEYS REQUIRED
# NO LIVE ORDERS
# PUBLIC MARKET DATA ONLY
# ============================================================

st.set_page_config(
    page_title="Crypto AI Trading Desk",
    page_icon="₿",
    layout="wide"
)

# -----------------------------
# EXCHANGE
# -----------------------------

@st.cache_resource
def get_exchange():
    return ccxt.binance({
        "enableRateLimit": True
    })

exchange = get_exchange()


# -----------------------------
# SETTINGS
# -----------------------------

DEFAULT_PAIRS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "DOGE/USDT",
    "ADA/USDT",
    "AVAX/USDT",
    "LINK/USDT",
    "SUI/USDT"
]


# -----------------------------
# DATA
# -----------------------------

@st.cache_data(ttl=60)
def get_market_data(symbol, timeframe="1h", limit=250):

    try:
        data = exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            limit=limit
        )

        df = pd.DataFrame(
            data,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume"
            ]
        )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            unit="ms",
            utc=True
        )

        return df

    except Exception as e:
        return None


# -----------------------------
# INDICATORS
# -----------------------------

def calculate_indicators(df):

    df = df.copy()

    # EMA
    df["ema20"] = df["close"].ewm(span=20).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["ema200"] = df["close"].ewm(span=200).mean()

    # RSI
    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["rsi"] = 100 - (100 / (1 + rs))

    # ATR
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())

    true_range = pd.concat(
        [high_low, high_close, low_close],
        axis=1
    ).max(axis=1)

    df["atr"] = true_range.rolling(14).mean()

    # Average volume
    df["avg_volume"] = df["volume"].rolling(20).mean()

    # Volume ratio
    df["volume_ratio"] = (
        df["volume"] /
        df["avg_volume"].replace(0, np.nan)
    )

    # Recent high / low
    df["recent_high"] = df["high"].rolling(20).max()
    df["recent_low"] = df["low"].rolling(20).min()

    return df


# -----------------------------
# ANALYSIS
# -----------------------------

def analyze_symbol(symbol):

    df = get_market_data(symbol)

    if df is None or len(df) < 100:
        return None

    df = calculate_indicators(df)

    last = df.iloc[-1]

    price = float(last["close"])
    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])
    ema200 = float(last["ema200"])
    rsi = float(last["rsi"])
    atr = float(last["atr"])
    volume_ratio = float(last["volume_ratio"])

    recent_high = float(
        df["high"].iloc[-21:-1].max()
    )

    recent_low = float(
        df["low"].iloc[-21:-1].min()
    )

    # -------------------------
    # TREND
    # -------------------------

    trend_score = 0

    if price > ema20:
        trend_score += 1

    if ema20 > ema50:
        trend_score += 1

    if ema50 > ema200:
        trend_score += 1

    if price < ema20:
        trend_score -= 1

    if ema20 < ema50:
        trend_score -= 1

    if ema50 < ema200:
        trend_score -= 1

    if trend_score >= 2:
        trend = "BULLISH"

    elif trend_score <= -2:
        trend = "BEARISH"

    else:
        trend = "NEUTRAL"


    # -------------------------
    # MOMENTUM
    # -------------------------

    if rsi >= 70:
        momentum = "OVERBOUGHT"

    elif rsi >= 55:
        momentum = "POSITIVE"

    elif rsi <= 30:
        momentum = "OVERSOLD"

    elif rsi <= 45:
        momentum = "NEGATIVE"

    else:
        momentum = "NEUTRAL"


    # -------------------------
    # VOLUME
    # -------------------------

    if volume_ratio >= 2:
        volume_signal = "VERY HIGH"

    elif volume_ratio >= 1.3:
        volume_signal = "HIGH"

    elif volume_ratio >= 0.8:
        volume_signal = "NORMAL"

    else:
        volume_signal = "LOW"


    # -------------------------
    # BREAKOUT
    # -------------------------

    breakout = price > recent_high

    breakdown = price < recent_low


    # -------------------------
    # SCORE
    # -------------------------

    score = 50

    if trend == "BULLISH":
        score += 20

    elif trend == "BEARISH":
        score -= 20

    if momentum == "POSITIVE":
        score += 10

    elif momentum == "NEGATIVE":
        score -= 10

    if volume_ratio >= 1.5:
        score += 10

    if breakout:
        score += 10

    if breakdown:
        score -= 10

    score = max(0, min(100, score))


    # -------------------------
    # SETUP
    # -------------------------

    if breakout and volume_ratio >= 1.3:
        setup = "BREAKOUT"

    elif breakdown and volume_ratio >= 1.3:
        setup = "BREAKDOWN"

    elif trend == "BULLISH":
        setup = "UPTREND"

    elif trend == "BEARISH":
        setup = "DOWNTREND"

    else:
        setup = "RANGE / WAIT"


    # -------------------------
    # LEVELS
    # -------------------------

    support = recent_low
    resistance = recent_high

    # ATR-based stop
    if trend == "BULLISH":

        stop = price - (1.5 * atr)

        target1 = price + (2 * (price - stop))
        target2 = price + (3 * (price - stop))

        direction = "LONG BIAS"

    elif trend == "BEARISH":

        stop = price + (1.5 * atr)

        target1 = price - (2 * (stop - price))
        target2 = price - (3 * (stop - price))

        direction = "SHORT BIAS"

    else:

        stop = None
        target1 = None
        target2 = None

        direction = "NO TRADE"


    return {
        "symbol": symbol,
        "price": price,
        "trend": trend,
        "momentum": momentum,
        "rsi": rsi,
        "volume_ratio": volume_ratio,
        "volume_signal": volume_signal,
        "setup": setup,
        "score": score,
        "support": support,
        "resistance": resistance,
        "atr": atr,
        "stop": stop,
        "target1": target1,
        "target2": target2,
        "direction": direction,
        "breakout": breakout,
        "breakdown": breakdown,
        "updated": datetime.now(timezone.utc)
    }


# -----------------------------
# SCANNER
# -----------------------------

def scan_market(pairs):

    results = []

    progress = st.progress(0)

    for i, symbol in enumerate(pairs):

        try:

            result = analyze_symbol(symbol)

            if result:
                results.append(result)

        except Exception:
            pass

        progress.progress((i + 1) / len(pairs))

    progress.empty()

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results


# ============================================================
# UI
# ============================================================

st.title("₿ Crypto AI Trading Desk")

st.caption(
    "SCAN → ANALYZE → RISK → PLAN | Research only | No live trading"
)

st.info(
    "This MVP uses public market data. It does NOT place orders "
    "and does NOT require exchange API keys."
)


# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.header("⚙️ Settings")

account_size = st.sidebar.number_input(
    "Account Size (USDT)",
    min_value=10.0,
    value=1000.0,
    step=50.0
)

risk_percent = st.sidebar.number_input(
    "Max Risk Per Trade (%)",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
    step=0.1
)

timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["15m", "1h", "4h", "1d"],
    index=1
)

selected_pairs = st.sidebar.multiselect(
    "Crypto Pairs",
    DEFAULT_PAIRS,
    default=DEFAULT_PAIRS[:6]
)

scan_button = st.sidebar.button(
    "🔎 SCAN MARKET",
    use_container_width=True
)


# ============================================================
# MAIN
# ============================================================

if scan_button:

    if not selected_pairs:

        st.warning("Select at least one crypto pair.")

    else:

        with st.spinner("Scanning crypto market..."):

            results = scan_market(selected_pairs)

        if not results:

            st.error(
                "No market data returned. Try again in a moment."
            )

        else:

            st.success(
                f"Scan complete — {len(results)} markets analyzed."
            )

            # -----------------------------
            # TOP SETUPS
            # -----------------------------

            st.header("🔥 Top Crypto Setups")

            table = []

            for r in results:

                table.append({
                    "Pair": r["symbol"],
                    "Price": round(r["price"], 6),
                    "Score": r["score"],
                    "Trend": r["trend"],
                    "Momentum": r["momentum"],
                    "RSI": round(r["rsi"], 2),
                    "Volume": f"{r['volume_ratio']:.2f}x",
                    "Setup": r["setup"],
                    "Bias": r["direction"]
                })

            st.dataframe(
                pd.DataFrame(table),
                use_container_width=True,
                hide_index=True
            )


            # -----------------------------
            # BEST SETUP
            # -----------------------------

            best = results[0]

            st.header(
                f"🎯 Best Setup: {best['symbol']}"
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Setup Score",
                f"{best['score']}/100"
            )

            c2.metric(
                "Trend",
                best["trend"]
            )

            c3.metric(
                "RSI",
                f"{best['rsi']:.1f}"
            )

            c4.metric(
                "Volume",
                f"{best['volume_ratio']:.2f}x"
            )


            # -----------------------------
            # ANALYZE
            # -----------------------------

            st.subheader("📊 ANALYZE")

            col1, col2 = st.columns(2)

            with col1:

                st.write(
                    f"**Current Price:** "
                    f"{best['price']:.6f}"
                )

                st.write(
                    f"**Trend:** {best['trend']}"
                )

                st.write(
                    f"**Momentum:** {best['momentum']}"
                )

                st.write(
                    f"**RSI:** {best['rsi']:.2f}"
                )

                st.write(
                    f"**Volume:** "
                    f"{best['volume_ratio']:.2f}x average"
                )

            with col2:

                st.write(
                    f"**Support:** "
                    f"{best['support']:.6f}"
                )

                st.write(
                    f"**Resistance:** "
                    f"{best['resistance']:.6f}"
                )

                st.write(
                    f"**ATR:** "
                    f"{best['atr']:.6f}"
                )

                st.write(
                    f"**Setup:** {best['setup']}"
                )

                st.write(
                    f"**Breakout:** "
                    f"{'YES' if best['breakout'] else 'NO'}"
                )


            # -----------------------------
            # RISK CALCULATOR
            # -----------------------------

            st.header("🛡️ RISK")

            if best["stop"] is not None:

                max_risk = (
                    account_size *
                    risk_percent /
                    100
                )

                entry = best["price"]
                stop = best["stop"]

                risk_per_coin = abs(
                    entry - stop
                )

                if risk_per_coin > 0:

                    position_size = (
                        max_risk /
                        risk_per_coin
                    )

                    notional = (
                        position_size *
                        entry
                    )

                    risk_reward = abs(
                        best["target1"] - entry
                    ) / risk_per_coin

                else:

                    position_size = 0
                    notional = 0
                    risk_reward = 0


                r1, r2, r3, r4 = st.columns(4)

                r1.metric(
                    "Max Risk",
                    f"${max_risk:.2f}"
                )

                r2.metric(
                    "Position Size",
                    f"{position_size:.6f}"
                )

                r3.metric(
                    "Notional",
                    f"${notional:.2f}"
                )

                r4.metric(
                    "R:R",
                    f"{risk_reward:.2f}:1"
                )


                st.write(
                    f"**Entry:** `{entry:.6f}`"
                )

                st.write(
                    f"**Stop:** `{stop:.6f}`"
                )

                st.write(
                    f"**Target 1:** "
                    f"`{best['target1']:.6f}`"
                )

                st.write(
                    f"**Target 2:** "
                    f"`{best['target2']:.6f}`"
                )

                if risk_reward >= 2:

                    st.success(
                        "Risk/reward passes the 2:1 research threshold."
                    )

                else:

                    st.warning(
                        "Risk/reward is below 2:1. "
                        "Treat this setup as weak."
                    )

            else:

                st.warning(
                    "No clean directional setup. "
                    "Risk calculation skipped."
                )


            # -----------------------------
            # PLAN
            # -----------------------------

            st.header("📋 TRADE PLAN")

            st.code(
                f"""
CRYPTO TRADE PLAN
=================

PAIR:
{best['symbol']}

TIMEFRAME:
{timeframe}

DIRECTION:
{best['direction']}

SETUP:
{best['setup']}

CURRENT PRICE:
{best['price']:.6f}

TREND:
{best['trend']}

MOMENTUM:
{best['momentum']}

RSI:
{best['rsi']:.2f}

VOLUME:
{best['volume_ratio']:.2f}x average

SUPPORT:
{best['support']:.6f}

RESISTANCE:
{best['resistance']:.6f}

ENTRY:
{best['price']:.6f}

STOP:
{best['stop'] if best['stop'] else 'N/A'}

TARGET 1:
{best['target1'] if best['target1'] else 'N/A'}

TARGET 2:
{best['target2'] if best['target2'] else 'N/A'}

ACCOUNT:
${account_size:.2f}

MAX RISK:
{risk_percent:.2f}%

STATUS:
AWAITING HUMAN REVIEW

IMPORTANT:
This is a research signal, not an automatic trade.
Verify market conditions before acting.
"""
            )


            st.warning(
                "⚠️ NOTHING HAS BEEN EXECUTED. "
                "This application is research-only."
            )


else:

    st.markdown(
        """
## How it works

**1. SCAN**
        
Scans selected crypto pairs for trend, momentum,
volume and breakout conditions.

**2. ANALYZE**

Evaluates EMA structure, RSI, ATR, support,
resistance and volume.

**3. RISK**

Calculates maximum account risk, position size,
stop distance and risk/reward.

**4. PLAN**

Creates one structured crypto trade plan.

### Start

Choose your account size and risk in the sidebar,
select your crypto pairs and press:

**🔎 SCAN MARKET**
"""
    )

st.divider()

st.caption(
    "Educational/research software. Crypto markets are highly volatile. "
    "No profit is guaranteed. Never risk money you cannot afford to lose."
)
