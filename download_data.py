import time
from pathlib import Path
import requests
import pandas as pd

BASE = "https://api.binance.com"
OUT = Path("data")
OUT.mkdir(exist_ok=True)

def download(symbol="BTCUSDT", interval="1h", start="2023-01-01", end=None):
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000) if end else None
    rows = []
    cursor = start_ms

    while True:
        params = {"symbol": symbol, "interval": interval, "limit": 1000, "startTime": cursor}
        if end_ms:
            params["endTime"] = end_ms
        r = requests.get(f"{BASE}/api/v3/klines", params=params, timeout=20)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + 1
        print(symbol, pd.to_datetime(cursor, unit="ms", utc=True), len(rows))
        if len(batch) < 1000:
            break
        if end_ms and cursor >= end_ms:
            break
        time.sleep(0.15)

    cols = ["open_time","open","high","low","close","volume","close_time",
            "quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"]
    df = pd.DataFrame(rows, columns=cols)
    for c in ["open","high","low","close","volume","quote_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.drop_duplicates("time").sort_values("time")
    path = OUT / f"{symbol}_{interval}.csv"
    df.to_csv(path, index=False)
    print("Saved:", path, "rows:", len(df))

if __name__ == "__main__":
    # Add/remove symbols as required.
    for symbol in ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","DOGEUSDT"]:
        download(symbol=symbol, interval="1h", start="2023-01-01")
