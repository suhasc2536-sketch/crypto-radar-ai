import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
DEXSCREENER_BASE = "https://api.dexscreener.com"
DEFILLAMA_BASE = "https://api.llama.fi"
DB = Path("radar.sqlite")

st.set_page_config(page_title="Crypto Radar AI V3", layout="wide")


def init_db():
    with sqlite3.connect(DB) as conn:
        conn.execute(""" CREATE TABLE IF NOT EXISTS scans ( id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT DEFAULT CURRENT_TIMESTAMP, payload TEXT NOT NULL ) """)


def save_scan(df):
    if df is None or df.empty:
        return
    payload = df.head(100).to_json(orient="records")
    with sqlite3.connect(DB) as conn:
        conn.execute("INSERT INTO scans(payload) VALUES (?)", (payload,))


@st.cache_data(ttl=120)
def load_market_universe(per_page=100):
    rows = []
    pages = (per_page + 249) // 250
    for page in range(1, pages + 1):
        size = min(250, per_page - len(rows))
        if size <= 0:
            break
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": size,
            "page": page,
            "sparkline": "false",
            "price_change_percentage": "1h,24h,7d",
        }
        r = requests.get(
            f"{COINGECKO_BASE}/coins/markets",
            params=params,
            timeout=20,
            headers={"User-Agent": "CryptoRadarAI/3.0"},
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    rename = {
        "current_price": "price",
        "total_volume": "volume_24h",
        "price_change_percentage_1h_in_currency": "change_1h",
        "price_change_percentage_24h_in_currency": "change_24h",
        "price_change_percentage_7d_in_currency": "change_7d",
    }
    df = df.rename(columns=rename)
    wanted = [
        "id", "symbol", "name", "market_cap_rank", "price",
        "market_cap", "fully_diluted_valuation", "volume_24h",
        "change_1h", "change_24h", "change_7d",
        "circulating_supply", "total_supply", "max_supply",
        "ath_change_percentage", "atl_change_percentage",
        "last_updated",
    ]
    return df[[c for c in wanted if c in df.columns]]


def clip(x, lo=0, hi=100):
    return max(lo, min(hi, float(x)))


def score_single_asset(asset):
    score = 0.0
    reasons = []

    c1 = float(asset.get("change_1h") or 0)
    c24 = float(asset.get("change_24h") or 0)
    c7 = float(asset.get("change_7d") or 0)
    vol = float(asset.get("volume_24h") or 0)
    mcap = float(asset.get("market_cap") or 0)

    if c1 > 0:
        score += 8
        reasons.append("Positive 1h momentum")
    if c24 > 2:
        score += 12
        reasons.append("Positive 24h momentum")
    elif c24 < -5:
        score -= 8
        reasons.append("Strong 24h weakness")

    if c7 > 5:
        score += 10
        reasons.append("Positive 7d trend")
    elif c7 < -10:
        score -= 6
        reasons.append("Weak 7d trend")

    turnover = vol / mcap if mcap > 0 else 0
    if turnover >= 0.20:
        score += 12
        reasons.append("High daily turnover relative to market cap")
    elif turnover >= 0.08:
        score += 6
        reasons.append("Healthy daily turnover")

    if mcap >= 10_000_000_000:
        score += 8
        reasons.append("Very large market-cap/liquidity universe")
    elif mcap >= 1_000_000_000:
        score += 5
        reasons.append("Large market-cap universe")
    elif mcap < 50_000_000:
        score -= 8
        reasons.append("Small-cap risk")

    if c24 > 30:
        score -= 8
        reasons.append("Extreme 24h move increases reversal risk")

    available = sum(
        pd.notna(asset.get(k))
        for k in [
            "price", "market_cap", "volume_24h",
            "change_1h", "change_24h", "change_7d"
        ]
    )
    confidence = min(clip(40 + available * 10), 65)

    if mcap < 100_000_000 or vol < 5_000_000:
        risk = "High"
    elif mcap < 1_000_000_000 or turnover < 0.03:
        risk = "Medium"
    else:
        risk = "Lower"

    return {
        "opportunity_score": round(clip(score + 50)),
        "confidence": round(confidence),
        "risk_level": risk,
        "reasons": reasons[:8] or ["Insufficient evidence"],
    }


def score_market_dataframe(df, min_market_cap, min_volume):
    if df.empty:
        return df

    x = df[
        (df["market_cap"].fillna(0) >= min_market_cap)
        & (df["volume_24h"].fillna(0) >= min_volume)
    ].copy()

    results = x.apply(lambda row: score_single_asset(row.to_dict()), axis=1)
    x["opportunity_score"] = results.map(lambda r: r["opportunity_score"])
    x["risk_level"] = results.map(lambda r: r["risk_level"])
    x["confidence"] = results.map(lambda r: r["confidence"])
    x["score_reasons"] = results.map(lambda r: " | ".join(r["reasons"]))

    x = x.sort_values(
        ["opportunity_score", "confidence"],
        ascending=False
    ).reset_index(drop=True)
    x["rank"] = range(1, len(x) + 1)
    return x


def get_dex_snapshot(symbol="", name=""):
    query = symbol or name
    r = requests.get(
        f"{DEXSCREENER_BASE}/latest/dex/search",
        params={"q": query},
        timeout=15,
        headers={"User-Agent": "CryptoRadarAI/3.0"},
    )
    r.raise_for_status()
    pairs = (r.json().get("pairs") or [])[:20]

    clean = []
    for p in pairs:
        clean.append({
            "chain": p.get("chainId"),
            "dex": p.get("dexId"),
            "pair": p.get("pairAddress"),
            "price_usd": p.get("priceUsd"),
            "liquidity_usd": (p.get("liquidity") or {}).get("usd"),
            "fdv": p.get("fdv"),
            "market_cap": p.get("marketCap"),
            "volume": p.get("volume"),
            "price_change": p.get("priceChange"),
            "transactions": p.get("txns"),
            "pair_created_at": p.get("pairCreatedAt"),
            "url": p.get("url"),
            "boosts": p.get("boosts"),
        })
    return {"query": query, "pairs": clean}


def get_protocol_snapshot(name):
    r = requests.get(
        f"{DEFILLAMA_BASE}/protocols",
        timeout=20,
        headers={"User-Agent": "CryptoRadarAI/3.0"},
    )
    r.raise_for_status()
    protocols = r.json()
    target = name.lower()
    matches = [
        p for p in protocols
        if target in str(p.get("name", "")).lower()
    ][:10]

    return {
        "query": name,
        "matches": [
            {
                "name": p.get("name"),
                "symbol": p.get("symbol"),
                "category": p.get("category"),
                "tvl": p.get("tvl"),
                "chainTvls": p.get("chainTvls"),
                "url": p.get("url"),
            }
            for p in matches
        ],
    }


init_db()

st.title("Crypto Radar AI V3")
st.caption(
    "Multi-source crypto market intelligence. Research signals only; "
    "not guaranteed predictions or financial advice."
)

with st.sidebar:
    st.header("Scanner")
    universe_size = st.selectbox(
        "Universe size",
        [100, 250, 500],
        index=0,
        help="Larger scans use more API requests and may be slower.",
    )
    min_market_cap = st.number_input(
        "Minimum market cap (USD)",
        min_value=0,
        value=10_000_000,
        step=1_000_000,
    )
    min_volume = st.number_input(
        "Minimum 24h volume (USD)",
        min_value=0,
        value=1_000_000,
        step=100_000,
    )
    if st.button("Refresh market scan"):
        st.cache_data.clear()
        st.rerun()

st.subheader("1. Full-market scanner")

ranked = pd.DataFrame()

try:
    market = load_market_universe(per_page=universe_size)
    ranked = score_market_dataframe(
        market,
        min_market_cap=min_market_cap,
        min_volume=min_volume,
    )

    st.metric("Assets scanned", f"{len(ranked):,}")

    display_cols = [
        "rank", "name", "symbol", "price", "market_cap",
        "volume_24h", "change_1h", "change_24h", "change_7d",
        "opportunity_score", "risk_level", "confidence",
    ]
    table = ranked[[c for c in display_cols if c in ranked.columns]].copy()

    for col in ["price", "market_cap", "volume_24h"]:
        if col in table.columns:
            table[col] = table[col].map(
                lambda x: (
                    f"${x:,.2f}" if pd.notna(x) and x < 1000
                    else f"${x:,.0f}" if pd.notna(x)
                    else "N/A"
                )
            )

    for col in ["change_1h", "change_24h", "change_7d"]:
        if col in table.columns:
            table[col] = table[col].map(
                lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A"
            )

    st.dataframe(table, use_container_width=True, hide_index=True)

    st.subheader("Top opportunities")
    st.dataframe(
        ranked.head(20)[
            [
                "name", "symbol", "opportunity_score",
                "risk_level", "confidence", "score_reasons"
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )
    save_scan(ranked)

except Exception as exc:
    st.error("The market scan failed.")
    st.code(str(exc))
    st.info(
        "The scanner is designed to degrade gracefully when a public provider "
        "rate-limits or becomes temporarily unavailable."
    )

st.divider()
st.subheader("2. Deep asset analysis")

if not ranked.empty:
    choices = ranked["id"].tolist()
    selected_id = st.selectbox(
        "Choose an asset",
        choices,
        format_func=lambda x: (
            f"{ranked.loc[ranked['id'] == x, 'name'].iloc[0]} "
            f"({ranked.loc[ranked['id'] == x, 'symbol'].iloc[0].upper()})"
        ),
    )

    asset = ranked[ranked["id"] == selected_id].iloc[0].to_dict()
    detail = score_single_asset(asset)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Opportunity", f"{detail['opportunity_score']}/100")
    c2.metric("Confidence", f"{detail['confidence']}/100")
    c3.metric("Risk", detail["risk_level"])
    c4.metric("24h change", f"{asset.get('change_24h', 0):+.2f}%")

    st.write("### Evidence")
    for item in detail["reasons"]:
        st.write(f"- {item}")

    st.write("### Data coverage")
    coverage = pd.DataFrame(
        [
            ["Market data", "CoinGecko", "Available"],
            ["DEX liquidity / pairs", "DEX Screener", "On-demand"],
            ["DeFi fundamentals", "DefiLlama", "On-demand"],
            ["Derivatives", "Exchange adapters", "Next module"],
            ["On-chain wallets", "Chain adapters", "Next module"],
            ["News / catalysts", "News adapters", "Next module"],
        ],
        columns=["Category", "Primary source", "Status"],
    )
    st.dataframe(coverage, use_container_width=True, hide_index=True)

    if st.button("Load DEX snapshot for selected asset"):
        try:
            st.json(get_dex_snapshot(asset["symbol"], asset["name"]))
        except Exception as exc:
            st.error(f"DEX lookup failed: {exc}")

    if st.button("Search DeFi fundamentals"):
        try:
            st.json(get_protocol_snapshot(asset["name"]))
        except Exception as exc:
            st.error(f"DeFi lookup failed: {exc}")

st.divider()
st.caption(
    "V3 is intentionally probability-first: collect evidence, normalize it, "
    "backtest it, and only then allow machine learning to influence probabilities."
            )
