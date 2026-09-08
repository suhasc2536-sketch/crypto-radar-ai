import os
import io
import json
import math
import sqlite3
import time
import hashlib
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    import plotly.graph_objects as go
except Exception:
    go = None
try:
    import networkx as nx
except Exception:
    nx = None
try:
    import feedparser
except Exception:
    feedparser = None
try:
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, brier_score_loss
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# ============================================================
# CRYPTO RADAR AI V100 — COMPLETE LOW-BUDGET RESEARCH BUILD
# ============================================================
# Public-data first. Optional keyed adapters are included where
# practical. This is research/paper-trading software, not a
# guaranteed prediction system and not a live trading terminal.
# ============================================================

APP_VERSION = "V100 COMPLETE"
DB = "radar_v100.sqlite"
TIMEFRAMES = ["15m", "1h", "4h", "1D"]

COINGECKO = "https://api.coingecko.com/api/v3"
COINBASE = "https://api.exchange.coinbase.com"
BINANCE = "https://api.binance.com"
BINANCE_FAPI = "https://fapi.binance.com"
BYBIT = "https://api.bybit.com"
OKX = "https://www.okx.com"
DEX = "https://api.dexscreener.com"
LLAMA = "https://api.llama.fi"
FNG = "https://api.alternative.me/fng/"
GOPLUS = "https://api.gopluslabs.io/api/v1"
ETHERSCAN = "https://api.etherscan.io/v2/api"

st.set_page_config(page_title="Crypto Radar AI V100", layout="wide", initial_sidebar_state="expanded")

# --------------------------- Utilities -----------------------

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def now_utc():
    return datetime.now(timezone.utc)


def request_json(url, params=None, timeout=15, headers=None):
    h = {"User-Agent": "CryptoRadarAI/100"}
    if headers:
        h.update(headers)
    r = requests.get(url, params=params, timeout=timeout, headers=h)
    r.raise_for_status()
    return r.json()


def key(name):
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, "")


def finite(x):
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


def clamp(x, lo, hi):
    return float(np.clip(x, lo, hi))


def pct(x):
    return "N/A" if not finite(x) else f"{float(x)*100:.2f}%"


def money(x):
    if not finite(x):
        return "N/A"
    x = float(x)
    if abs(x) >= 1e9:
        return f"${x/1e9:.2f}B"
    if abs(x) >= 1e6:
        return f"${x/1e6:.2f}M"
    if abs(x) >= 1e3:
        return f"${x/1e3:.1f}K"
    return f"${x:,.2f}"

# --------------------------- Database ------------------------

def init_db():
    with sqlite3.connect(DB) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS observations( id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, asset TEXT, source TEXT, field TEXT, value REAL, quality REAL, note TEXT)""")
        con.execute("""CREATE TABLE IF NOT EXISTS predictions( id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, asset TEXT, direction TEXT, horizon TEXT, entry REAL, invalidation REAL, target REAL, p_up REAL, p_down REAL, confidence REAL, risk REAL, regime TEXT, status TEXT DEFAULT 'OPEN', outcome REAL, thesis TEXT)""")
        con.execute("""CREATE TABLE IF NOT EXISTS events( id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, asset TEXT, source TEXT, title TEXT, url TEXT, published TEXT, sentiment REAL, category TEXT, raw TEXT)""")


def log_observation(asset, source, field, value, quality=1.0, note=""):
    if not finite(value):
        return
    try:
        with sqlite3.connect(DB) as con:
            con.execute("INSERT INTO observations(created_at,asset,source,field,value,quality,note) VALUES(?,?,?,?,?,?,?)",
                        (now_utc().isoformat(), asset, source, field, float(value), float(quality), note[:500]))
    except Exception:
        pass


def save_prediction(asset, direction, entry, invalidation, target, p_up, p_down, confidence, risk, regime, thesis):
    try:
        with sqlite3.connect(DB) as con:
            con.execute("INSERT INTO predictions(created_at,asset,direction,horizon,entry,invalidation,target,p_up,p_down,confidence,risk,regime,thesis) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (now_utc().isoformat(), asset, direction, "24h", entry, invalidation, target,
                         p_up, p_down, confidence, risk, regime, thesis[:4000]))
    except Exception:
        pass

init_db()

# ---------------------- Market universe ----------------------

@st.cache_data(ttl=180)
def get_universe(n=100):
    rows = []
    page = 1
    while len(rows) < n:
        take = min(250, n-len(rows))
        data = request_json(f"{COINGECKO}/coins/markets", {
            "vs_currency":"usd", "order":"market_cap_desc", "per_page":take,
            "page":page, "sparkline":"false", "price_change_percentage":"1h,24h,7d"
        })
        if not data:
            break
        rows.extend(data)
        page += 1
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.rename(columns={
        "current_price":"price", "total_volume":"volume_24h",
        "price_change_percentage_1h_in_currency":"change_1h",
        "price_change_percentage_24h_in_currency":"change_24h",
        "price_change_percentage_7d_in_currency":"change_7d"})

# ---------------------- Exchange adapters --------------------

CB_GRAN = {"15m":900, "1h":3600, "4h":21600, "1D":86400}
BIN_GRAN = {"15m":"15m", "1h":"1h", "4h":"4h", "1D":"1d"}
OKX_BAR = {"15m":"15m", "1h":"1H", "4h":"4H", "1D":"1D"}
BYBIT_INTERVAL = {"15m":"15", "1h":"60", "4h":"240", "1D":"D"}


def normalize_ohlcv(rows, exchange):
    if not rows:
        return pd.DataFrame()
    try:
        if exchange == "coinbase":
            x = pd.DataFrame(rows, columns=["timestamp","low","high","open","close","volume"])
        elif exchange == "binance":
            x = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume","close_time","qav","trades","tbv","tqv","ignore"])
            x = x[["timestamp","low","high","open","close","volume"]]
        elif exchange == "bybit":
            x = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume","turnover"])
            x = x[["timestamp","low","high","open","close","volume"]]
        elif exchange == "okx":
            x = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume","vol_ccy","vol_ccy_quote","confirm"])
            x = x[["timestamp","low","high","open","close","volume"]]
        else:
            return pd.DataFrame()
        for c in ["low","high","open","close","volume"]:
            x[c] = pd.to_numeric(x[c], errors="coerce")
        x["timestamp"] = pd.to_datetime(x["timestamp"], unit="ms" if exchange != "coinbase" else "s", utc=True)
        return x.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def coinbase_candles(product, timeframe="1h", bars=300):
    end = int(time.time()); gran = CB_GRAN[timeframe]; start = end-gran*min(bars,300)
    return normalize_ohlcv(request_json(f"{COINBASE}/products/{product}/candles", {"granularity":gran,"start":start,"end":end}), "coinbase")


@st.cache_data(ttl=60)
def binance_candles(symbol, timeframe="1h", bars=300):
    return normalize_ohlcv(request_json(f"{BINANCE}/api/v3/klines", {"symbol":symbol+"USDT","interval":BIN_GRAN[timeframe],"limit":min(bars,1000)}), "binance")


@st.cache_data(ttl=60)
def bybit_candles(symbol, timeframe="1h", bars=300):
    j = request_json(f"{BYBIT}/v5/market/kline", {"category":"spot","symbol":symbol+"USDT","interval":BYBIT_INTERVAL[timeframe],"limit":min(bars,1000)})
    return normalize_ohlcv(list(reversed(j.get("result",{}).get("list",[]))), "bybit")


@st.cache_data(ttl=60)
def okx_candles(symbol, timeframe="1h", bars=300):
    j = request_json(f"{OKX}/api/v5/market/candles", {"instId":symbol+"-USDT","bar":OKX_BAR[timeframe],"limit":min(bars,300)})
    return normalize_ohlcv(list(reversed(j.get("data",[]))), "okx")


@st.cache_data(ttl=30)
def coinbase_book(product):
    return request_json(f"{COINBASE}/products/{product}/book", {"level":2})


@st.cache_data(ttl=30)
def coinbase_trades(product):
    return request_json(f"{COINBASE}/products/{product}/trades")


@st.cache_data(ttl=30)
def binance_ticker(symbol):
    return request_json(f"{BINANCE}/api/v3/ticker/price", {"symbol":symbol+"USDT"})


@st.cache_data(ttl=30)
def bybit_ticker(symbol):
    j = request_json(f"{BYBIT}/v5/market/tickers", {"category":"spot","symbol":symbol+"USDT"})
    a = (j.get("result",{}).get("list") or [{}])[0]
    return {"price":safe_float(a.get("lastPrice")),"bid":safe_float(a.get("bid1Price")),"ask":safe_float(a.get("ask1Price"))}


@st.cache_data(ttl=30)
def okx_ticker(symbol):
    j = request_json(f"{OKX}/api/v5/market/ticker", {"instId":symbol+"-USDT"})
    a = (j.get("data") or [{}])[0]
    return {"price":safe_float(a.get("last")),"bid":safe_float(a.get("bidPx")),"ask":safe_float(a.get("askPx"))}

# ---------------------- Technical engine ---------------------

def rsi(s, period=14):
    d=s.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    au=up.ewm(alpha=1/period,adjust=False).mean(); ad=dn.ewm(alpha=1/period,adjust=False).mean()
    rs=au/ad.replace(0,np.nan)
    return 100-100/(1+rs)


def atr(df, period=14):
    p=df.close.shift(1)
    tr=pd.concat([df.high-df.low,(df.high-p).abs(),(df.low-p).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/period,adjust=False).mean()


def adx(df, period=14):
    up=df.high.diff(); dn=-df.low.diff(); tr=atr(df,period)
    plus=np.where((up>dn)&(up>0),up,0.0); minus=np.where((dn>up)&(dn>0),dn,0.0)
    pdi=100*pd.Series(plus,index=df.index).ewm(alpha=1/period,adjust=False).mean()/tr
    mdi=100*pd.Series(minus,index=df.index).ewm(alpha=1/period,adjust=False).mean()/tr
    dx=(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)*100
    return dx.ewm(alpha=1/period,adjust=False).mean()


def enrich(df):
    x=df.copy()
    if x.empty: return x
    x["ema20"]=x.close.ewm(span=20,adjust=False).mean(); x["ema50"]=x.close.ewm(span=50,adjust=False).mean(); x["ema100"]=x.close.ewm(span=100,adjust=False).mean()
    x["rsi14"]=rsi(x.close); x["atr14"]=atr(x); x["atr_pct"]=x.atr14/x.close
    x["adx14"]=adx(x); x["ret1"]=x.close.pct_change(); x["ret3"]=x.close.pct_change(3); x["ret6"]=x.close.pct_change(6); x["ret12"]=x.close.pct_change(12)
    x["vol_ma20"]=x.volume.rolling(20).mean(); x["volume_ratio"]=x.volume/x.vol_ma20
    x["volume_z"]=(x.volume-x.volume.rolling(50).mean())/x.volume.rolling(50).std()
    x["high20"]=x.high.rolling(20).max().shift(1); x["low20"]=x.low.rolling(20).min().shift(1)
    x["breakout20"]=x.close/x.high20-1; x["breakdown20"]=x.close/x.low20-1
    x["ema_spread"]=x.ema20/x.ema50-1; x["ema_long_spread"]=x.ema50/x.ema100-1
    x["bb_mid"]=x.close.rolling(20).mean(); x["bb_std"]=x.close.rolling(20).std(); x["bb_z"]=(x.close-x.bb_mid)/x.bb_std.replace(0,np.nan)
    x["range_pct"]=(x.high-x.low)/x.close
    return x


def last_features(x):
    if x.empty: return {}
    r=x.iloc[-1]
    return {c:safe_float(r.get(c)) for c in ["close","rsi14","atr_pct","adx14","ret1","ret3","ret6","ret12","volume_ratio","volume_z","breakout20","breakdown20","ema_spread","ema_long_spread","bb_z","range_pct"]}

# ---------------------- Microstructure -----------------------

def book_metrics(book):
    try:
        bids=[(safe_float(a[0]),safe_float(a[1])) for a in book.get("bids",[])[:50]]; asks=[(safe_float(a[0]),safe_float(a[1])) for a in book.get("asks",[])[:50]]
        if not bids or not asks: return {}
        bid=sum(p*q for p,q in bids); ask=sum(p*q for p,q in asks); bp=bids[0][0]; ap=asks[0][0]; mid=(bp+ap)/2
        return {"bid_depth":bid,"ask_depth":ask,"book_imbalance":(bid-ask)/(bid+ask) if bid+ask else np.nan,"spread_bps":(ap-bp)/mid*10000 if mid else np.nan}
    except Exception: return {}


def trade_flow_metrics(trades):
    try:
        # Coinbase: side is taker side in returned trade objects.
        buy=sell=0.0
        for t in trades[:500]:
            q=safe_float(t.get("size")); side=str(t.get("side","")).lower()
            if side=="buy": buy+=q
            elif side=="sell": sell+=q
        total=buy+sell
        return {"buy_volume":buy,"sell_volume":sell,"trade_flow_imbalance":(buy-sell)/total if total else np.nan}
    except Exception: return {}

# ---------------------- Derivatives ---------------------------

@st.cache_data(ttl=60)
def binance_derivatives(symbol):
    out={"available":False}
    try:
        s=symbol+"USDT"
        oi=request_json(f"{BINANCE_FAPI}/fapi/v1/openInterest",{"symbol":s})
        tk=request_json(f"{BINANCE_FAPI}/fapi/v1/premiumIndex",{"symbol":s})
        fr=request_json(f"{BINANCE_FAPI}/fapi/v1/fundingRate",{"symbol":s,"limit":1})
        out.update({"available":True,"open_interest":safe_float(oi.get("openInterest")),"mark":safe_float(tk.get("markPrice")),"index":safe_float(tk.get("indexPrice")),"funding":safe_float(tk.get("lastFundingRate")),"funding_hist":safe_float((fr or [{}])[0].get("fundingRate"))})
        return out
    except Exception:
        return out

@st.cache_data(ttl=60)
def bybit_derivatives(symbol):
    try:
        s=symbol+"USDT"; j=request_json(f"{BYBIT}/v5/market/tickers",{"category":"linear","symbol":s}); a=(j.get("result",{}).get("list") or [{}])[0]
        return {"available":bool(a),"open_interest":safe_float(a.get("openInterest")),"mark":safe_float(a.get("markPrice")),"index":safe_float(a.get("indexPrice")),"funding":safe_float(a.get("fundingRate"))}
    except Exception: return {"available":False}


def derivatives_score(d):
    if not d.get("available"): return 50, ["No derivatives data"]
    s=50; reasons=[]; f=d.get("funding",np.nan)
    if finite(f):
        if f>0.0005: s-=12; reasons.append("Funding is elevated; crowded longs can increase squeeze risk")
        elif f<-0.0003: s+=8; reasons.append("Negative funding creates potential short-squeeze fuel")
        else: reasons.append("Funding is not strongly stretched")
    if finite(d.get("basis")): reasons.append("Futures basis is available")
    return clamp(s,0,100), reasons

# ---------------------- DEX / DeFi ----------------------------

@st.cache_data(ttl=120)
def dex_search(symbol, name=""):
    try:
        q=symbol
        j=request_json(f"{DEX}/latest/dex/search",{"q":q})
        rows=[]
        for p in j.get("pairs",[])[:100]:
            li=safe_float((p.get("liquidity") or {}).get("usd")); vol=safe_float((p.get("volume") or {}).get("h24")); ch=safe_float((p.get("priceChange") or {}).get("h24"))
            rows.append({"chain":p.get("chainId"),"dex":p.get("dexId"),"pair":p.get("pairAddress"),"price_usd":safe_float(p.get("priceUsd")),"liquidity_usd":li,"volume_24h":vol,"change_24h_pct":ch,"url":p.get("url")})
        return pd.DataFrame(rows)
    except Exception: return pd.DataFrame()

@st.cache_data(ttl=120)
def defillama_search(name):
    try:
        data=request_json(f"{LLAMA}/protocols")
        q=name.lower(); rows=[]
        for x in data:
            n=str(x.get("name",""));
            if q in n.lower() or str(x.get("symbol","")).lower()==q:
                rows.append({"name":n,"symbol":x.get("symbol"),"category":x.get("category"),"tvl":safe_float(x.get("tvl")),"change_1d":safe_float(x.get("change_1d")),"change_7d":safe_float(x.get("change_7d"))})
        return pd.DataFrame(rows[:20])
    except Exception: return pd.DataFrame()

# ---------------------- News / catalysts ----------------------

RSS_FEEDS = {
    "CoinDesk":"https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Cointelegraph":"https://cointelegraph.com/rss",
    "Decrypt":"https://decrypt.co/feed",
}


def sentiment_from_text(text):
    t=str(text).lower(); pos=["approval","approved","partnership","launch","adoption","inflow","upgrade","listing","surge","record","bullish"]
    neg=["hack","exploit","lawsuit","ban","delist","outflow","liquidation","fraud","attack","bearish","unlock"]
    p=sum(w in t for w in pos); n=sum(w in t for w in neg)
    return clamp((p-n)/max(1,p+n),-1,1)


@st.cache_data(ttl=300)
def news_feed():
    rows=[]
    if feedparser is None: return pd.DataFrame()
    for source,url in RSS_FEEDS.items():
        try:
            f=feedparser.parse(url)
            for e in f.entries[:30]:
                title=str(e.get("title","")); summary=str(e.get("summary","")); link=e.get("link","")
                published=e.get("published",e.get("updated",""))
                rows.append({"source":source,"title":title,"summary":summary,"url":link,"published":published,"sentiment":sentiment_from_text(title+" "+summary)})
        except Exception: pass
    return pd.DataFrame(rows).drop_duplicates("title") if rows else pd.DataFrame()


@st.cache_data(ttl=120)
def fear_greed():
    try:
        j=request_json(FNG,{"limit":1}); a=(j.get("data") or [{}])[0]
        return {"value":safe_float(a.get("value")),"label":a.get("value_classification","N/A")}
    except Exception: return {"value":np.nan,"label":"Unavailable"}

# ---------------------- Optional keyed adapters ----------------

@st.cache_data(ttl=120)
def cryptopanic_news(symbol):
    token=key("CRYPTOPANIC_API_KEY")
    if not token: return pd.DataFrame()
    try:
        j=request_json("https://cryptopanic.com/api/developer/v2/posts/", {"auth_token":token,"currencies":symbol,"public":"true"})
        rows=[]
        for p in j.get("results",[]): rows.append({"source":"CryptoPanic","title":p.get("title"),"url":p.get("url"),"published":p.get("published_at"),"sentiment":np.nan})
        return pd.DataFrame(rows)
    except Exception: return pd.DataFrame()


@st.cache_data(ttl=300)
def goplus_security(chain_id, address):
    if not address: return {}
    try:
        j=request_json(f"{GOPLUS}/token_security/{chain_id}", {"contract_addresses":address})
        return (j.get("result") or {}).get(address.lower(), {})
    except Exception: return {}

# ---------------------- Tokenomics / security -----------------

def tokenomics_row(row):
    circ=safe_float(row.get("circulating_supply")); total=safe_float(row.get("total_supply")); maxs=safe_float(row.get("max_supply"))
    fdv=safe_float(row.get("fully_diluted_valuation")); mcap=safe_float(row.get("market_cap"))
    dilution=(fdv/mcap-1) if finite(fdv) and finite(mcap) and mcap>0 else np.nan
    return {"circulating_supply":circ,"total_supply":total,"max_supply":maxs,"market_cap":mcap,"fdv":fdv,"fdv_premium":dilution}


def security_flags(sec):
    if not sec: return ["No contract-security provider data for this asset"]
    fields=[("is_honeypot","Honeypot risk flagged"),("is_blacklisted","Blacklist mechanism flagged"),("is_proxy","Proxy contract"),("cannot_sell_all","Sell restriction flagged"),("is_mintable","Mint capability exists")]
    out=[]
    for k,msg in fields:
        v=str(sec.get(k,"0")).lower()
        if v in {"1","true","yes"}: out.append(msg)
    return out or ["No major supplied security flags"]

# ---------------------- Historical analog + ML ----------------

FEATURES=["rsi14","atr_pct","adx14","ret3","ret6","ret12","volume_ratio","volume_z","ema_spread","ema_long_spread","bb_z","range_pct"]


def analog_probability(tech, horizon_bars=24, k=60):
    if tech.empty: return {"p_up5":np.nan,"p_down5":np.nan,"mean_future":np.nan,"sample":0}
    x=tech.copy(); x["future_return"]=x.close.shift(-horizon_bars)/x.close-1
    x=x.dropna(subset=FEATURES+["future_return"])
    if len(x)<100: return {"p_up5":np.nan,"p_down5":np.nan,"mean_future":np.nan,"sample":len(x)}
    current=x.iloc[-1][FEATURES].astype(float); hist=x.iloc[:-horizon_bars].copy()
    med=hist[FEATURES].median(); mad=(hist[FEATURES]-med).abs().median().replace(0,np.nan)
    z=(hist[FEATURES]-med)/(1.4826*mad); cz=(current-med)/(1.4826*mad)
    dist=((z-cz).pow(2).mean(axis=1)).pow(.5)
    hist=hist.assign(distance=dist).replace([np.inf,-np.inf],np.nan).dropna(subset=["distance"])
    near=hist.nsmallest(min(k,len(hist)),"distance"); fr=near.future_return
    return {"p_up5":float((fr>=.05).mean()),"p_down5":float((fr<=-.05).mean()),"mean_future":float(fr.mean()),"sample":len(near)}


def make_ml_dataset(tech, horizon=24):
    x=tech.copy(); x["future_return"]=x.close.shift(-horizon)/x.close-1; x["y"]=(x.future_return>=.05).astype(int)
    return x.dropna(subset=FEATURES+["future_return"])


def walk_forward_ml(tech, horizon=24):
    if not SKLEARN_OK: return {"available":False,"reason":"scikit-learn unavailable"}
    d=make_ml_dataset(tech,horizon)
    if len(d)<180: return {"available":False,"reason":f"Only {len(d)} labeled observations"}
    split=int(len(d)*.7); train=d.iloc[:split]; test=d.iloc[split:]
    if len(test)<30: return {"available":False,"reason":"Too little out-of-sample data"}
    model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight="balanced"))
    model.fit(train[FEATURES],train.y); proba=model.predict_proba(test[FEATURES])[:,1]; pred=(proba>=.5).astype(int)
    out={"available":True,"model":"logistic","train":len(train),"test":len(test),"accuracy":accuracy_score(test.y,pred),"precision":precision_score(test.y,pred,zero_division=0),"recall":recall_score(test.y,pred,zero_division=0),"brier":brier_score_loss(test.y,proba)}
    try: out["auc"]=roc_auc_score(test.y,proba)
    except Exception: out["auc"]=np.nan
    out["current_probability"]=float(model.predict_proba(d[FEATURES].iloc[[-1]])[0,1])
    return out

# ---------------------- Regime / fusion -----------------------

def timeframe_alignment(latest):
    vals=[]
    for tf in TIMEFRAMES:
        f=latest.get(tf,{})
        if not f: continue
        s=0
        s += 1 if f.get("ema_spread",0)>0 else -1
        s += 1 if f.get("ema_long_spread",0)>0 else -1
        s += 1 if f.get("rsi14",50)>50 else -1
        s += 1 if f.get("ret3",0)>0 else -1
        vals.append(s/4)
    a=float(np.mean(vals)) if vals else 0
    return a, ("STRONG BULL" if a>.65 else "BULL" if a>.2 else "MIXED" if a>-.2 else "BEAR" if a>-.65 else "STRONG BEAR")


def market_regime(tech_map):
    votes=[]; reasons=[]
    for sym in ["BTC","ETH"]:
        x=tech_map.get(sym)
        if x is None or x.empty: continue
        f=last_features(x); v=1 if f.get("ema_spread",0)>0 and f.get("ema_long_spread",0)>0 else -1
        votes.append(v)
    v=np.mean(votes) if votes else 0
    if v>.5: regime="RISK-ON"
    elif v<-.5: regime="RISK-OFF"
    else: regime="TRANSITION"
    return regime


def evidence_scores(f, micro, deriv, news_sent, fg, ml, analog, alignment, regime):
    tech=50; reasons=[]
    if finite(f.get("ema_spread")): tech += 14 if f["ema_spread"]>0 else -14; reasons.append("EMA20/50 structure is bullish" if f["ema_spread"]>0 else "EMA20/50 structure is bearish")
    if finite(f.get("ema_long_spread")): tech += 10 if f["ema_long_spread"]>0 else -10
    if finite(f.get("rsi14")):
        if 52<=f["rsi14"]<=68: tech+=8; reasons.append("RSI supports positive momentum without extreme overbought conditions")
        elif f["rsi14"]>75: tech-=10; reasons.append("RSI is overheated")
        elif f["rsi14"]<30: tech+=4; reasons.append("RSI is deeply oversold; reversal potential exists but is unconfirmed")
    if finite(f.get("volume_ratio")) and f["volume_ratio"]>1.5: tech += 8 if f.get("ret3",0)>0 else -8; reasons.append("Abnormal volume is confirming recent direction")
    tech=clamp(tech,0,100)
    micro_score=50
    if finite(micro.get("book_imbalance")): micro_score += 15*np.sign(micro["book_imbalance"])
    if finite(micro.get("trade_flow_imbalance")): micro_score += 15*np.sign(micro["trade_flow_imbalance"])
    if finite(micro.get("spread_bps")) and micro["spread_bps"]>30: micro_score-=12
    micro_score=clamp(micro_score,0,100)
    deriv_score,_=derivatives_score(deriv)
    if finite(news_sent):
        tech += clamp(news_sent*10,-10,10)
    if finite(fg):
        # Extreme fear can be contrarian, but only a small influence.
        if fg<20: tech+=3
        elif fg>80: tech-=3
    if regime=="RISK-ON": tech+=4
    elif regime=="RISK-OFF": tech-=4
    tech=clamp(tech,0,100)
    base=.45*tech+.30*micro_score+.25*deriv_score
    p=base/100
    if finite(analog.get("p_up5")):
        shrink=min(1,analog.get("sample",0)/120); p=.75*p+.25*(.5+(analog["p_up5"]-.5)*shrink)
    if finite(ml.get("current_probability")):
        p=.70*p+.30*ml["current_probability"]
    p += .05*alignment
    return clamp(p,.03,.97), tech, micro_score, deriv_score, reasons


def confidence(available, agreement, sample, ml, source_count):
    c=25 + 7*available + 4*source_count
    if finite(agreement) and agreement<10: c+=8
    if sample>=60: c+=8
    if ml.get("available"): c+=6
    return int(clamp(c,0,90))


def risk_score(f,micro,dex,deriv,security_flags,regime):
    r=4.5
    if finite(f.get("atr_pct")): r += clamp(f["atr_pct"]*100-2,-1,2)
    if finite(micro.get("spread_bps")) and micro["spread_bps"]>30: r+=1
    if not dex.empty:
        li=safe_float(dex["liquidity_usd"].max())
        if finite(li) and li<100000: r+=1
    if deriv.get("available") and finite(deriv.get("funding")) and abs(deriv["funding"])>.0008: r+=.8
    if any("risk" in x.lower() or "flag" in x.lower() or "honeypot" in x.lower() for x in security_flags): r+=1.5
    if regime=="RISK-OFF": r+=.8
    return clamp(r,1,10)


def decision(p_up,p_down,conf,risk,alignment,regime):
    if conf<45 or risk>=8 or alignment=="MIXED": return "NO TRADE"
    if p_up>=.72 and p_up-p_down>=.30 and alignment in {"STRONG BULL","BULL"}: return "STRONG LONG"
    if p_up>=.60 and p_up-p_down>=.18 and alignment in {"STRONG BULL","BULL"}: return "LONG"
    if p_down>=.72 and p_down-p_up>=.30 and alignment in {"STRONG BEAR","BEAR"}: return "STRONG SHORT"
    if p_down>=.60 and p_down-p_up>=.18 and alignment in {"STRONG BEAR","BEAR"}: return "SHORT"
    return "WATCH"


def levels(price, atr_pct, dec):
    if not finite(price): return np.nan,np.nan
    vol=atr_pct if finite(atr_pct) else .02
    if dec in {"LONG","STRONG LONG"}: return price*(1-2*vol), price*(1+3*vol)
    if dec in {"SHORT","STRONG SHORT"}: return price*(1+2*vol), price*(1-3*vol)
    return price*(1-vol), price*(1+vol)

# ---------------------- Knowledge graph -----------------------

def graph_data(name,symbol,regime,decision,news_df,dex_df,deriv,security_flags):
    nodes=[(symbol,{"kind":"asset"}),("BTC",{"kind":"market"}),("ETH",{"kind":"market"}),(regime,{"kind":"regime"}),(decision,{"kind":"decision"})]
    edges=[(symbol,"BTC","market context"),(symbol,"ETH","market context"),(symbol,regime,"regime"),(symbol,decision,"decision")]
    if not dex_df.empty:
        for chain in dex_df.chain.dropna().astype(str).unique()[:4]:
            nodes.append((chain,{"kind":"dex"})); edges.append((symbol,chain,"DEX liquidity"))
    if deriv.get("available"):
        nodes.append(("PERPETUALS",{"kind":"derivatives"})); edges.append((symbol,"PERPETUALS","derivatives"))
    for sf in security_flags[:3]:
        n="SEC:"+sf[:28]; nodes.append((n,{"kind":"security"})); edges.append((symbol,n,"security"))
    if not news_df.empty:
        for _,r in news_df.head(5).iterrows():
            title=str(r.get("title",""))[:50]
            if title:
                nodes.append((title,{"kind":"news"})); edges.append((symbol,title,"news/catalyst"))
    return nodes,edges


def draw_graph(nodes,edges):
    if nx is None or go is None: return
    G=nx.Graph(); G.add_nodes_from(nodes); G.add_edges_from([(a,b,{'label':c}) for a,b,c in edges])
    pos=nx.spring_layout(G,seed=7,k=.9)
    xs=[];ys=[]
    for a,b,_ in edges:
        xs += [pos[a][0],pos[b][0],None]; ys += [pos[a][1],pos[b][1],None]
    edge=go.Scatter(x=xs,y=ys,mode="lines",hoverinfo="none")
    nxs=[];nys=[];texts=[]
    for n in G.nodes:
        nxs.append(pos[n][0]); nys.append(pos[n][1]); texts.append(n)
    node=go.Scatter(x=nxs,y=nys,mode="markers+text",text=texts,textposition="top center",marker={"size":12})
    st.plotly_chart(go.Figure([edge,node]).update_layout(height=550,showlegend=False,margin=dict(l=10,r=10,t=20,b=10)),use_container_width=True)

# ---------------------- Backtest ------------------------------

def simple_walkforward_backtest(tech, horizon=24, threshold=.02):
    if tech.empty: return {}
    x=tech.copy(); x["future"]=x.close.shift(-horizon)/x.close-1
    x=x.dropna(subset=FEATURES+["future"])
    if len(x)<150: return {"status":"INSUFFICIENT DATA","n":len(x)}
    split=int(len(x)*.7); train=x.iloc[:split]; test=x.iloc[split:]
    signal=(test["ema_spread"]>0)&(test["rsi14"].between(50,70))&(test["volume_ratio"]>1.1)
    ret=test.loc[signal,"future"]
    if len(ret)==0: return {"status":"NO SIGNALS","n":len(test)}
    wins=(ret>0).mean(); avg=ret.mean(); med=ret.median(); gross=(1+ret).prod()-1
    return {"status":"OK","train":len(train),"test":len(test),"signals":len(ret),"hit_rate":wins,"avg_return":avg,"median_return":med,"compound_return":gross}

# ---------------------- Asset analysis ------------------------

def load_tf_map(symbol):
    cb={}; tech={}; raw={}
    for tf in TIMEFRAMES:
        try:
            d=coinbase_candles(symbol+"-USD",tf,300)
            raw[tf]=d; tech[tf]=enrich(d)
        except Exception:
            raw[tf]=pd.DataFrame(); tech[tf]=pd.DataFrame()
    return raw,tech


def exchange_prices(symbol):
    out={}
    for name,fn in [("Coinbase",lambda:safe_float(coinbase_ticker(symbol))),]:
        pass
    try: out["Binance"]=safe_float(binance_ticker(symbol).get("price"))
    except Exception: out["Binance"]=np.nan
    try: out["Bybit"]=safe_float(bybit_ticker(symbol).get("price"))
    except Exception: out["Bybit"]=np.nan
    try: out["OKX"]=safe_float(okx_ticker(symbol).get("price"))
    except Exception: out["OKX"]=np.nan
    return out

# ---------------------- Main UI -------------------------------

st.title("Crypto Radar AI V100")
st.caption("Multi-source crypto intelligence • research + paper trading • no guaranteed predictions • no live orders")

with st.sidebar:
    st.header("V100 Controls")
    universe_n=st.selectbox("Universe size",[50,100,250],index=0)
    min_mcap=st.number_input("Minimum market cap",0,10_000_000_000,10_000_000,1_000_000)
    min_vol=st.number_input("Minimum 24h volume",0,10_000_000_000,1_000_000,100_000)
    if st.button("Refresh all public data"):
        st.cache_data.clear(); st.rerun()
    st.divider()
    st.write("**Optional keys**")
    st.caption("The build works without them. Keys should be stored in Streamlit Secrets, not GitHub code.")
    st.write("CryptoPanic: "+("configured" if key("CRYPTOPANIC_API_KEY") else "not configured"))
    st.write("GoPlus: public adapter")
    st.write("Etherscan: "+("configured" if key("ETHERSCAN_API_KEY") else "not configured"))

try:
    universe=get_universe(universe_n)
    if not universe.empty:
        universe=universe[(universe.market_cap.fillna(0)>=min_mcap)&(universe.volume_24h.fillna(0)>=min_vol)].copy()
except Exception as e:
    st.error(f"Universe error: {e}"); universe=pd.DataFrame()

if universe.empty:
    st.warning("No market universe returned. Try Refresh."); st.stop()

universe["turnover"]=universe.volume_24h/universe.market_cap.replace(0,np.nan)
universe["opportunity_rank"]=(50+np.clip(universe.change_24h.fillna(0)*1.2,-12,12)+np.clip(universe.change_7d.fillna(0)*.5,-8,8)+np.clip(universe.turnover.fillna(0)*40,0,10)).clip(0,100)
universe=universe.sort_values("opportunity_rank",ascending=False).reset_index(drop=True)

# Market overview
st.header("1. Global Market Radar")
fg=fear_greed(); c1,c2,c3,c4=st.columns(4)
c1.metric("Assets scanned",len(universe)); c2.metric("Fear & Greed",f"{fg['value']:.0f}" if finite(fg['value']) else "N/A",fg['label']); c3.metric("Top 24h",f"{universe.iloc[0]['name']} {universe.iloc[0]['change_24h']:.1f}%"); c4.metric("Timestamp",now_utc().strftime("%H:%M UTC"))
st.dataframe(universe[["market_cap_rank","name","symbol","price","market_cap","volume_24h","change_24h","change_7d","opportunity_rank"]].head(50),use_container_width=True,hide_index=True)

selected=st.selectbox("Choose asset for full V100 intelligence",universe.id.tolist(),format_func=lambda x:f"{universe.loc[universe.id==x,'name'].iloc[0]} ({universe.loc[universe.id==x,'symbol'].iloc[0].upper()})")
row=universe[universe.id==selected].iloc[0]; symbol=str(row.symbol).upper(); name=str(row.name)

st.header(f"2. Intelligence Center — {name} ({symbol})")
raw,tech_map=load_tf_map(symbol)
latest={tf:last_features(tech_map.get(tf,pd.DataFrame())) for tf in TIMEFRAMES}
align,align_label=timeframe_alignment(latest)

# Chart
if go is not None and not raw.get("1h",pd.DataFrame()).empty:
    x=raw["1h"].tail(150)
    fig=go.Figure(go.Candlestick(x=x.timestamp,open=x.open,high=x.high,low=x.low,close=x.close,name="1h"))
    fig.update_layout(height=430,title="1h market structure",xaxis_rangeslider_visible=False)
    st.plotly_chart(fig,use_container_width=True)

cols=st.columns(5)
f1=latest.get("1h",{})
cols[0].metric("Price",money(f1.get("close"))); cols[1].metric("RSI",f"{f1.get('rsi14',np.nan):.1f}" if finite(f1.get('rsi14')) else "N/A"); cols[2].metric("Volume",f"{f1.get('volume_ratio',np.nan):.2f}x" if finite(f1.get('volume_ratio')) else "N/A"); cols[3].metric("ADX",f"{f1.get('adx14',np.nan):.1f}" if finite(f1.get('adx14')) else "N/A"); cols[4].metric("TF alignment",align_label)

# Multi-timeframe table
mt=[]
for tf in TIMEFRAMES:
    f=latest.get(tf,{})
    mt.append({"timeframe":tf,"price":f.get("close"),"RSI":f.get("rsi14"),"ADX":f.get("adx14"),"volume_ratio":f.get("volume_ratio"),"EMA20/50":f.get("ema_spread"),"EMA50/100":f.get("ema_long_spread"),"return_3":f.get("ret3")})
st.dataframe(pd.DataFrame(mt),use_container_width=True,hide_index=True)

# Cross exchange
st.header("3. Multi-Exchange & Microstructure")
cb_price=safe_float(f1.get("close")); prices=exchange_prices(symbol); prices["Coinbase"]=cb_price
price_df=pd.DataFrame([{"exchange":k,"price":v,"spread_bps":(v-cb_price)/cb_price*10000 if finite(v) and finite(cb_price) else np.nan} for k,v in prices.items()])
st.dataframe(price_df,use_container_width=True,hide_index=True)
try: book=book_metrics(coinbase_book(symbol+"-USD")); flow=trade_flow_metrics(coinbase_trades(symbol+"-USD"))
except Exception: book={}; flow={}
mc1,mc2,mc3,mc4=st.columns(4)
mc1.metric("Book imbalance",f"{book.get('book_imbalance',np.nan):+.2f}" if finite(book.get('book_imbalance')) else "N/A")
mc2.metric("Trade imbalance",f"{flow.get('trade_flow_imbalance',np.nan):+.2f}" if finite(flow.get('trade_flow_imbalance')) else "N/A")
mc3.metric("Spread",f"{book.get('spread_bps',np.nan):.1f} bps" if finite(book.get('spread_bps')) else "N/A")
mc4.metric("Bid/ask depth",money(book.get('bid_depth'))+" / "+money(book.get('ask_depth')) if finite(book.get('bid_depth')) else "N/A")

# Derivatives
st.header("4. Derivatives Intelligence")
bd=binance_derivatives(symbol); yd=bybit_derivatives(symbol)
deriv=bd if bd.get("available") else yd
dscore,dreasons=derivatives_score(deriv)
dc1,dc2,dc3,dc4=st.columns(4)
dc1.metric("Source", "Binance" if bd.get("available") else "Bybit" if yd.get("available") else "None")
dc2.metric("Open interest",money(deriv.get("open_interest")))
dc3.metric("Funding",pct(deriv.get("funding")))
dc4.metric("Mark",money(deriv.get("mark")))
for r in dreasons: st.caption("• "+r)

# DEX / DeFi
st.header("5. DEX + DeFi Liquidity")
dex=dex_search(symbol,name); defi=defillama_search(name)
if not dex.empty: st.dataframe(dex.sort_values("liquidity_usd",ascending=False).head(15),use_container_width=True,hide_index=True)
else: st.info("No DEX pair data returned for this symbol.")
if not defi.empty: st.dataframe(defi,use_container_width=True,hide_index=True)

# News / social / catalyst layer
st.header("6. News, Social Proxy & Catalysts")
news=news_feed(); cp=cryptopanic_news(symbol)
if not cp.empty: news=pd.concat([news,cp],ignore_index=True)
if not news.empty:
    # crude asset relevance filter; broad macro articles are retained.
    rel=news[news.title.fillna("").str.contains(symbol,case=False,regex=False)|news.summary.fillna("").str.contains(symbol,case=False,regex=False)].copy()
    if rel.empty: rel=news.head(15)
    news_sent=safe_float(rel.sentiment.mean())
    st.metric("News sentiment proxy",f"{news_sent:+.2f}" if finite(news_sent) else "N/A")
    st.dataframe(rel[[c for c in ["source","title","published","sentiment","url"] if c in rel.columns]].head(20),use_container_width=True,hide_index=True)
else:
    rel=pd.DataFrame(); news_sent=np.nan; st.info("RSS news adapters returned no data.")

# Tokenomics
st.header("7. Tokenomics + Dilution")
tok=tokenomics_row(row.to_dict()); tc=st.columns(5)
tc[0].metric("Market cap",money(tok["market_cap"])); tc[1].metric("FDV",money(tok["fdv"])); tc[2].metric("FDV premium",pct(tok["fdv_premium"])); tc[3].metric("Circulating",f"{tok['circulating_supply']:.3g}" if finite(tok["circulating_supply"]) else "N/A"); tc[4].metric("Max supply",f"{tok['max_supply']:.3g}" if finite(tok["max_supply"]) else "N/A")
if finite(tok["fdv_premium"]) and tok["fdv_premium"]>.5: st.warning("FDV is substantially above current market cap; future supply/dilution deserves attention.")

# Security
st.header("8. Security / Manipulation")
sec={}
security_flags_list=security_flags(sec)
for r in security_flags_list: st.write("• "+r)
st.caption("For a contract-address security scan, a chain/address is required. GoPlus adapter is wired but deliberately not guessing contract addresses from ticker symbols.")

# Historical + ML
st.header("9. Historical Patterns + ML")
primary=tech_map.get("1h",pd.DataFrame()); analog=analog_probability(primary,24,60); ml=walk_forward_ml(primary,24); bt=simple_walkforward_backtest(primary,24)
a,b,c,d=st.columns(4); a.metric("Analog sample",analog.get("sample",0)); b.metric("Analog P(+5%)",pct(analog.get("p_up5"))); c.metric("ML status", "READY" if ml.get("available") else "LIMITED"); d.metric("WF backtest",bt.get("status","N/A"))
if ml.get("available"):
    st.dataframe(pd.DataFrame([ml]),use_container_width=True,hide_index=True)
else: st.info(ml.get("reason","ML unavailable"))
if bt.get("status")=="OK": st.dataframe(pd.DataFrame([bt]),use_container_width=True,hide_index=True)

# Final fusion
regime_tech={}
for s in ["BTC","ETH"]:
    try: regime_tech[s]=enrich(coinbase_candles(s+"-USD","1h",240))
    except Exception: regime_tech[s]=pd.DataFrame()
regime=market_regime(regime_tech)
fgv=fg.get("value",np.nan)
p_up,tech_score,micro_score,deriv_score,reasons=evidence_scores(f1,book,deriv,news_sent,fgv,ml,analog,align,regime)
p_down=clamp(1-p_up,.03,.97)
agreement_bps=np.nan
valid_prices=[v for v in prices.values() if finite(v)]
if len(valid_prices)>=2 and cb_price>0: agreement_bps=(max(valid_prices)-min(valid_prices))/cb_price*10000
available=sum([not primary.empty,bool(book),bool(flow),bool(deriv.get("available")),not dex.empty,not rel.empty,finite(analog.get("p_up5")),ml.get("available",False)])
conf=confidence(available,agreement_bps,analog.get("sample",0),ml,len(valid_prices))
secflags=security_flags_list
risk=risk_score(f1,book,dex,deriv,secflags,regime)
dec=decision(p_up,p_down,conf,risk,align_label,regime)
invalidation,target=levels(cb_price,f1.get("atr_pct"),dec)

st.header("10. V100 Decision Center")
cc=st.columns(7)
cc[0].metric("DECISION",dec); cc[1].metric("P(+5% / 24h)",pct(p_up)); cc[2].metric("P(-5% / 24h)",pct(p_down)); cc[3].metric("Confidence",str(conf)+"/100"); cc[4].metric("Risk",f"{risk:.1f}/10"); cc[5].metric("Regime",regime); cc[6].metric("Agreement",f"{agreement_bps:.1f} bps" if finite(agreement_bps) else "N/A")
st.dataframe(pd.DataFrame([{"reference_entry":cb_price,"invalidation":invalidation,"target":target,"decision":dec,"p_up_5pct":p_up,"p_down_5pct":p_down,"confidence":conf,"risk":risk,"regime":regime,"timeframe_alignment":align_label}]),use_container_width=True,hide_index=True)

st.subheader("Evidence behind the decision")
for r in reasons[:12]: st.write("• "+r)
if finite(analog.get("p_up5")): st.write(f"• Historical analogs: {analog['p_up5']*100:.1f}% of the nearest {analog['sample']} states reached +5% over 24 hourly bars; mean forward return {analog['mean_future']*100:.2f}%.")
if ml.get("available"): st.write(f"• Out-of-sample ML: AUC {ml.get('auc',np.nan):.2f}, Brier {ml.get('brier',np.nan):.3f}; current model probability {ml.get('current_probability',np.nan)*100:.1f}%.")
st.warning("These probabilities are experimental estimates, not guaranteed win rates. A calibrated probability requires larger out-of-sample testing across assets and market regimes.")

# Knowledge graph
st.header("11. Intelligence Graph")
nodes,edges=graph_data(name,symbol,regime,dec,rel,dex,deriv,secflags)
draw_graph(nodes,edges)

# Paper ledger
st.header("12. Paper Prediction Ledger")
thesis=f"{dec} {symbol}; P(+5%)={p_up:.3f}; P(-5%)={p_down:.3f}; confidence={conf}; risk={risk:.1f}/10; regime={regime}; alignment={align_label}; exchange_agreement={agreement_bps}." 
if st.button("Record paper thesis"):
    save_prediction(f"{name}:{symbol}",dec,cb_price,invalidation,target,p_up,p_down,conf,risk,regime,thesis)
    st.success("Saved to paper ledger. No order was placed.")
try:
    with sqlite3.connect(DB) as con:
        ledger=pd.read_sql_query("SELECT created_at,asset,direction,entry,invalidation,target,p_up,p_down,confidence,risk,regime,status FROM predictions ORDER BY id DESC LIMIT 50",con)
    st.dataframe(ledger,use_container_width=True,hide_index=True)
except Exception: pass

# Architecture / coverage
st.header("13. V100 Coverage Matrix")
coverage=pd.DataFrame([
 ["Market universe","LIVE","CoinGecko public"],["Coinbase candles","LIVE","15m/1h/4h/1D"],["Binance spot","LIVE","Price + candles"],["Bybit spot","LIVE","Price + candles"],["OKX spot","LIVE","Price + candles"],["Order book","LIVE","Coinbase"],["Trade flow","LIVE","Coinbase"],["Derivatives","LIVE","Binance + Bybit"],["DEX","LIVE","DEX Screener"],["DeFi","LIVE","DefiLlama"],["News","LIVE","RSS + optional CryptoPanic"],["Fear/Greed","LIVE","Alternative.me"],["Tokenomics","LIVE","CoinGecko fields"],["Security","ADAPTER","GoPlus when address supplied"],["On-chain wallets","ADAPTER","Requires chain/provider integration"],["Smart money","ADAPTER","Nansen/API-key class data"],["Unlock calendar","ADAPTER","Provider integration required"],["Knowledge graph","LIVE","Internal relationship graph"],["Historical analogs","LIVE","Local OHLCV"],["ML","LIVE/limited","Walk-forward logistic model"],["Calibration","RESEARCH","Requires much larger labeled history"],["Paper ledger","LIVE","SQLite"],["Live trading","OFF","Intentionally disabled"],
],columns=["Layer","Status","Source/Note"])
st.dataframe(coverage,use_container_width=True,hide_index=True)

# Diagnostics
with st.expander("System diagnostics"):
    st.write({"version":APP_VERSION,"python_ml":SKLEARN_OK,"db":DB,"universe":len(universe),"asset":symbol,"timestamp":now_utc().isoformat()})
    st.caption("For production use, replace SQLite with PostgreSQL/TimescaleDB and run scheduled collectors outside the Streamlit UI.")
