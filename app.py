
import sqlite3
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

# Crypto Radar AI V4
# Public-data research engine. It intentionally does NOT claim guaranteed predictions.
# Sources currently wired: CoinGecko, Coinbase Exchange, Binance Futures, DEX Screener,
# DefiLlama. Paid/keyed sources can be added later as optional adapters.

COINGECKO = "https://api.coingecko.com/api/v3"
COINBASE = "https://api.exchange.coinbase.com"
BINANCE_FAPI = "https://fapi.binance.com"
DEX = "https://api.dexscreener.com"
LLAMA = "https://api.llama.fi"
DB = "radar_v4.sqlite"

st.set_page_config(page_title="Crypto Radar AI V4", layout="wide")

# -----------------------------
# Generic helpers
# -----------------------------

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def request_json(url, params=None, timeout=20):
    r = requests.get(
        url,
        params=params,
        timeout=timeout,
        headers={"User-Agent": "CryptoRadarAI/4.0"},
    )
    r.raise_for_status()
    return r.json()


def init_db():
    with sqlite3.connect(DB) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                asset TEXT NOT NULL,
                timeframe TEXT,
                payload TEXT NOT NULL
            )
        """)


def save_observation(asset, timeframe, payload):
    try:
        with sqlite3.connect(DB) as con:
            con.execute(
                "INSERT INTO observations(created_at, asset, timeframe, payload) VALUES (?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), asset, timeframe, str(payload)),
            )
    except Exception:
        pass


# -----------------------------
# CoinGecko broad universe
# -----------------------------

@st.cache_data(ttl=180)
def get_universe(n=100):
    rows = []
    remaining = n
    page = 1
    while remaining > 0:
        take = min(250, remaining)
        data = request_json(
            f"{COINGECKO}/coins/markets",
            {
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": take,
                "page": page,
                "sparkline": "false",
                "price_change_percentage": "1h,24h,7d",
            },
        )
        if not data:
            break
        rows.extend(data)
        remaining -= len(data)
        page += 1

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    return df.rename(columns={
        "current_price": "price",
        "total_volume": "volume_24h",
        "price_change_percentage_1h_in_currency": "change_1h",
        "price_change_percentage_24h_in_currency": "change_24h",
        "price_change_percentage_7d_in_currency": "change_7d",
    })


# -----------------------------
# Coinbase OHLCV
# -----------------------------

CB_GRAN = {
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 21600,
    "1D": 86400,
}

@st.cache_data(ttl=60)
def coinbase_candles(product, timeframe="1h", bars=300):
    gran = CB_GRAN[timeframe]
    bars = min(int(bars), 300)
    end = int(time.time())
    start = end - gran * bars

    raw = request_json(
        f"{COINBASE}/products/{product}/candles",
        {"granularity": gran, "start": start, "end": end},
    )

    if not raw:
        return pd.DataFrame()

    # Coinbase schema: [time, low, high, open, close, volume]
    df = pd.DataFrame(
        raw,
        columns=["timestamp", "low", "high", "open", "close", "volume"],
    )
    for c in ["low", "high", "open", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


# -----------------------------
# Technical feature engine
# -----------------------------

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1/period, adjust=False).mean()
    avg_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df, period=14):
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def adx(df, period=14):
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = atr(df, period)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / tr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / tr
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1/period, adjust=False).mean()


def enrich_technicals(df):
    x = df.copy()
    x["ema20"] = x["close"].ewm(span=20, adjust=False).mean()
    x["ema50"] = x["close"].ewm(span=50, adjust=False).mean()
    x["ema100"] = x["close"].ewm(span=100, adjust=False).mean()
    x["rsi14"] = rsi(x["close"], 14)
    x["atr14"] = atr(x, 14)
    x["atr_pct"] = 100 * x["atr14"] / x["close"]
    x["adx14"] = adx(x, 14)
    x["ret1"] = x["close"].pct_change()
    x["ret3"] = x["close"].pct_change(3)
    x["ret6"] = x["close"].pct_change(6)
    x["ret12"] = x["close"].pct_change(12)
    x["vol_ma20"] = x["volume"].rolling(20).mean()
    x["volume_ratio"] = x["volume"] / x["vol_ma20"]
    x["volume_z"] = (
        (x["volume"] - x["volume"].rolling(50).mean())
        / x["volume"].rolling(50).std()
    )
    x["high20"] = x["high"].rolling(20).max().shift(1)
    x["low20"] = x["low"].rolling(20).min().shift(1)
    x["breakout20"] = x["close"] / x["high20"] - 1
    x["drawdown20"] = x["close"] / x["high20"] - 1
    x["ema_spread"] = x["ema20"] / x["ema50"] - 1
    x["ema_long_spread"] = x["ema50"] / x["ema100"] - 1
    x["bb_mid"] = x["close"].rolling(20).mean()
    x["bb_std"] = x["close"].rolling(20).std()
    x["bb_z"] = (x["close"] - x["bb_mid"]) / x["bb_std"].replace(0, np.nan)
    return x


# -----------------------------
# Order book / spot microstructure
# -----------------------------

@st.cache_data(ttl=20)
def coinbase_book(product, level=2):
    return request_json(
        f"{COINBASE}/products/{product}/book",
        {"level": level},
        timeout=15,
    )


def book_metrics(book, depth=20):
    bids = book.get("bids", [])[:depth]
    asks = book.get("asks", [])[:depth]
    if not bids or not asks:
        return {}

    bid_qty = sum(safe_float(x[1], 0) for x in bids)
    ask_qty = sum(safe_float(x[1], 0) for x in asks)
    best_bid = safe_float(bids[0][0])
    best_ask = safe_float(asks[0][0])
    mid = (best_bid + best_ask) / 2 if best_bid and best_ask else np.nan
    spread_bps = 10000 * (best_ask - best_bid) / mid if mid else np.nan

    return {
        "bid_qty_top20": bid_qty,
        "ask_qty_top20": ask_qty,
        "book_imbalance": (bid_qty - ask_qty) / (bid_qty + ask_qty)
        if bid_qty + ask_qty else np.nan,
        "spread_bps": spread_bps,
    }


@st.cache_data(ttl=20)
def coinbase_trades(product):
    return request_json(
        f"{COINBASE}/products/{product}/trades",
        {"limit": 1000},
        timeout=15,
    )


def trade_flow_metrics(trades):
    if not trades:
        return {}
    buy = 0.0
    sell = 0.0
    for t in trades:
        qty = safe_float(t.get("size"), 0)
        if str(t.get("side", "")).lower() == "buy":
            buy += qty
        else:
            sell += qty
    total = buy + sell
    return {
        "trade_buy_volume": buy,
        "trade_sell_volume": sell,
        "trade_flow_imbalance": (buy - sell) / total if total else np.nan,
        "trade_count": len(trades),
    }


# -----------------------------
# Binance Futures derivatives
# -----------------------------

BINANCE_PAIR_MAP = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT",
    "BNB": "BNBUSDT", "XRP": "XRPUSDT", "DOGE": "DOGEUSDT",
    "ADA": "ADAUSDT", "AVAX": "AVAXUSDT", "LINK": "LINKUSDT",
    "DOT": "DOTUSDT", "TRX": "TRXUSDT", "LTC": "LTCUSDT",
    "BCH": "BCHUSDT", "SUI": "SUIUSDT", "TON": "TONUSDT",
    "NEAR": "NEARUSDT", "APT": "APTUSDT", "UNI": "UNIUSDT",
    "AAVE": "AAVEUSDT", "PEPE": "PEPEUSDT", "SHIB": "SHIBUSDT",
}


@st.cache_data(ttl=30)
def binance_derivatives(symbol):
    pair = BINANCE_PAIR_MAP.get(symbol.upper())
    if not pair:
        return {"available": False, "reason": "No Binance mapping for this asset"}

    out = {"available": True, "symbol": pair}

    try:
        prem = request_json(
            f"{BINANCE_FAPI}/fapi/v1/premiumIndex",
            {"symbol": pair},
            timeout=10,
        )
        out["funding_rate"] = safe_float(prem.get("lastFundingRate"))
        out["mark_price"] = safe_float(prem.get("markPrice"))
        out["index_price"] = safe_float(prem.get("indexPrice"))
    except Exception as e:
        out["funding_error"] = str(e)

    try:
        oi = request_json(
            f"{BINANCE_FAPI}/fapi/v1/openInterest",
            {"symbol": pair},
            timeout=10,
        )
        out["open_interest"] = safe_float(oi.get("openInterest"))
    except Exception as e:
        out["oi_error"] = str(e)

    def hist(endpoint, params):
        try:
            data = request_json(f"{BINANCE_FAPI}{endpoint}", params, timeout=10)
            return data[-1] if data else {}
        except Exception:
            return {}

    ls = hist(
        "/futures/data/globalLongShortAccountRatio",
        {"pair": symbol.upper(), "period": "1h", "contractType": "PERPETUAL", "limit": 1},
    )
    if ls:
        out["global_long_pct"] = safe_float(ls.get("longAccount"))
        out["global_short_pct"] = safe_float(ls.get("shortAccount"))
        out["global_long_short"] = safe_float(ls.get("longShortRatio"))

    top = hist(
        "/futures/data/topLongShortAccountRatio",
        {"pair": symbol.upper(), "period": "1h", "contractType": "PERPETUAL", "limit": 1},
    )
    if top:
        out["top_long_pct"] = safe_float(top.get("longAccount"))
        out["top_short_pct"] = safe_float(top.get("shortAccount"))
        out["top_long_short"] = safe_float(top.get("longShortRatio"))

    return out


# -----------------------------
# DEX Screener and DefiLlama
# -----------------------------

@st.cache_data(ttl=120)
def dex_search(symbol, name):
    q = symbol or name
    try:
        data = request_json(
            f"{DEX}/latest/dex/search",
            {"q": q},
            timeout=15,
        )
        pairs = data.get("pairs") or []
        rows = []
        for p in pairs[:20]:
            rows.append({
                "chain": p.get("chainId"),
                "dex": p.get("dexId"),
                "pair": p.get("pairAddress"),
                "price_usd": safe_float(p.get("priceUsd")),
                "liquidity_usd": safe_float((p.get("liquidity") or {}).get("usd")),
                "fdv": safe_float(p.get("fdv")),
                "market_cap": safe_float(p.get("marketCap")),
                "volume_24h": safe_float((p.get("volume") or {}).get("h24")),
                "change_24h": safe_float((p.get("priceChange") or {}).get("h24")),
                "buys_24h": safe_float((p.get("txns") or {}).get("h24", {}).get("buys")),
                "sells_24h": safe_float((p.get("txns") or {}).get("h24", {}).get("sells")),
                "pair_created_at": p.get("pairCreatedAt"),
                "url": p.get("url"),
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def llama_protocol(name):
    try:
        data = request_json(f"{LLAMA}/protocols", timeout=20)
        target = name.lower()
        matches = [
            p for p in data
            if target in str(p.get("name", "")).lower()
        ]
        return matches[:10]
    except Exception:
        return []


# -----------------------------
# Evidence and probability engine
# -----------------------------

def latest_features(tech):
    if tech.empty:
        return {}
    r = tech.iloc[-1]
    return {k: safe_float(r.get(k)) for k in [
        "close", "rsi14", "atr_pct", "adx14", "ret1", "ret3", "ret6",
        "ret12", "volume_ratio", "volume_z", "breakout20",
        "ema_spread", "ema_long_spread", "bb_z"
    ]}


def technical_evidence(f):
    score = 0.0
    reasons = []
    total = 0

    def add(condition, weight, positive, negative):
        nonlocal score, total
        total += abs(weight)
        if condition:
            score += weight
            reasons.append(positive if weight > 0 else negative)

    add(f.get("ema_spread", 0) > 0, 2, "EMA20 is above EMA50", "EMA20 is below EMA50")
    add(f.get("ema_long_spread", 0) > 0, 1.5, "EMA50 is above EMA100", "EMA50 is below EMA100")
    add(f.get("rsi14", 50) > 55, 1, "RSI has bullish momentum", "RSI is not in bullish momentum")
    add(f.get("rsi14", 50) < 75, 0.5, "RSI is below an extreme-overbought zone", "RSI is stretched")
    add(f.get("adx14", 0) > 20, 1, "Trend strength is meaningful", "Trend strength is weak")
    add(f.get("volume_ratio", 1) > 1.5, 1.5, "Volume is unusually elevated", "Volume is not unusually elevated")
    add(f.get("breakout20", 0) > 0, 2, "Price is above its prior 20-bar high", "No 20-bar breakout")
    add(f.get("ret6", 0) > 0, 1, "Recent momentum is positive", "Recent momentum is negative")

    normalized = 50 + 50 * score / max(total, 1)
    return float(np.clip(normalized, 0, 100)), reasons


def derivative_evidence(d):
    if not d.get("available"):
        return 50, ["Derivatives unavailable for this asset"]

    score = 0.0
    reasons = []
    count = 0

    funding = d.get("funding_rate")
    if np.isfinite(funding):
        count += 1
        if funding > 0.0008:
            score -= 1.5
            reasons.append("Funding is elevated; crowded longs can increase reversal risk")
        elif funding < -0.0003:
            score += 1
            reasons.append("Negative funding can indicate crowded shorts / squeeze potential")
        else:
            score += 0.5
            reasons.append("Funding is not strongly crowded")

    ls = d.get("global_long_short")
    if np.isfinite(ls):
        count += 1
        if ls > 1.5:
            score -= 0.8
            reasons.append("Global long/short positioning is long-heavy")
        elif ls < 0.8:
            score += 0.8
            reasons.append("Global positioning is relatively short-heavy")

    top = d.get("top_long_short")
    if np.isfinite(top):
        count += 1
        if top > 1.5:
            score -= 0.8
            reasons.append("Top-trader accounts are long-heavy")
        elif top < 0.8:
            score += 0.8
            reasons.append("Top-trader accounts are relatively short-heavy")

    return float(np.clip(50 + score * 15, 0, 100)), reasons or ["Limited derivatives evidence"]


def microstructure_evidence(book_m, flow_m):
    score = 0.0
    reasons = []

    imb = book_m.get("book_imbalance", np.nan)
    if np.isfinite(imb):
        if imb > 0.15:
            score += 1
            reasons.append("Top-of-book depth is bid-heavy")
        elif imb < -0.15:
            score -= 1
            reasons.append("Top-of-book depth is ask-heavy")

    flow = flow_m.get("trade_flow_imbalance", np.nan)
    if np.isfinite(flow):
        if flow > 0.12:
            score += 1
            reasons.append("Recent trade flow is buy-heavy")
        elif flow < -0.12:
            score -= 1
            reasons.append("Recent trade flow is sell-heavy")

    spread = book_m.get("spread_bps", np.nan)
    if np.isfinite(spread) and spread > 30:
        score -= 0.7
        reasons.append("Wide spread increases execution/liquidity risk")

    return float(np.clip(50 + score * 18, 0, 100)), reasons or ["Limited spot microstructure evidence"]


def historical_analog_probability(tech, horizon_bars=6, k=40):
    """Nearest-neighbor historical analogs; no look-ahead because outcomes are future returns."""
    cols = ["rsi14", "atr_pct", "adx14", "ret3", "ret6", "volume_ratio", "volume_z", "ema_spread", "bb_z"]
    x = tech.copy()
    x["future_return"] = x["close"].shift(-horizon_bars) / x["close"] - 1
    x = x.dropna(subset=cols + ["future_return"]).copy()

    if len(x) < 80:
        return {
            "p_up5": np.nan, "p_down5": np.nan, "sample": len(x),
            "mean_future": np.nan, "note": "Not enough historical observations"
        }

    current = x.iloc[-1][cols].astype(float)
    hist = x.iloc[:-horizon_bars].copy()

    # Robust scaling by historical median/MAD.
    med = hist[cols].median()
    mad = (hist[cols] - med).abs().median().replace(0, np.nan)
    z = (hist[cols] - med) / (1.4826 * mad)
    cz = (current - med) / (1.4826 * mad)
    dist = ((z - cz).pow(2).mean(axis=1)).pow(0.5)
    hist = hist.assign(distance=dist).replace([np.inf, -np.inf], np.nan).dropna(subset=["distance"])

    nearest = hist.nsmallest(min(k, len(hist)), "distance")
    if nearest.empty:
        return {
            "p_up5": np.nan, "p_down5": np.nan, "sample": 0,
            "mean_future": np.nan, "note": "No comparable historical states"
        }

    fr = nearest["future_return"]
    return {
        "p_up5": float((fr >= 0.05).mean()),
        "p_down5": float((fr <= -0.05).mean()),
        "mean_future": float(fr.mean()),
        "sample": int(len(nearest)),
        "note": "Nearest historical states; not a guaranteed forecast",
    }


def fuse_probability(tech_score, deriv_score, micro_score, analog):
    components = [tech_score, deriv_score, micro_score]
    weights = [0.45, 0.25, 0.30]
    base = sum(a*w for a, w in zip(components, weights))

    p5 = analog.get("p_up5", np.nan)
    pdn = analog.get("p_down5", np.nan)

    if np.isfinite(p5):
        # Shrink the analog estimate toward 50% to avoid overconfidence from small samples.
        shrink = min(1.0, analog["sample"] / 80)
        p5_shrunk = 0.5 + (p5 - 0.5) * shrink
        p5_final = 0.65 * (base / 100) + 0.35 * p5_shrunk
    else:
        p5_final = base / 100

    if np.isfinite(pdn):
        shrink = min(1.0, analog["sample"] / 80)
        pdn_shrunk = 0.5 + (pdn - 0.5) * shrink
        pdn_final = 0.65 * ((100 - base) / 100) + 0.35 * pdn_shrunk
    else:
        pdn_final = (100 - base) / 100

    return float(np.clip(p5_final, 0.02, 0.98)), float(np.clip(pdn_final, 0.02, 0.98))


# -----------------------------
# App
# -----------------------------

init_db()

st.title("Crypto Radar AI V4")
st.caption(
    "Multi-source evidence engine: market structure + volume + liquidity + "
    "derivatives + DEX/DeFi + historical analogs."
)

with st.sidebar:
    st.header("Scanner settings")
    universe_n = st.selectbox("Market universe", [50, 100, 250], index=1)
    min_mcap = st.number_input("Minimum market cap (USD)", min_value=0, value=10_000_000, step=1_000_000)
    min_vol = st.number_input("Minimum 24h volume (USD)", min_value=0, value=1_000_000, step=100_000)
    st.caption("Public APIs are rate-limited. Deep analysis is performed for one selected asset.")
    if st.button("Refresh all"):
        st.cache_data.clear()
        st.rerun()

st.subheader("1. Market universe")

try:
    universe = get_universe(universe_n)
    universe = universe[
        (universe["market_cap"].fillna(0) >= min_mcap)
        & (universe["volume_24h"].fillna(0) >= min_vol)
    ].copy()
except Exception as e:
    st.error(f"Market data failed: {e}")
    universe = pd.DataFrame()

if not universe.empty:
    # Lightweight market ranking
