"""
Walk-forward crypto momentum backtester.

Important:
- Signals are calculated using information available at the close of bar t.
- Entry occurs on the next bar's open (avoids same-bar look-ahead).
- Fees and slippage are deducted.
- Final performance is evaluated on a chronological test period.
- This is research code, not financial advice.
"""

from pathlib import Path
import itertools
import numpy as np
import pandas as pd

DATA = Path("data")
FEE = 0.001       # 0.10% per side
SLIPPAGE = 0.0005 # 0.05% per side

def indicators(df, fast, slow, vol_window=24):
    x = df.copy()
    x["ema_fast"] = x["close"].ewm(span=fast, adjust=False).mean()
    x["ema_slow"] = x["close"].ewm(span=slow, adjust=False).mean()
    delta = x["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi"] = 100 - 100/(1+rs)
    x["vol_ratio"] = x["volume"] / x["volume"].rolling(vol_window).mean()
    x["breakout"] = x["close"] > x["high"].rolling(48).max().shift(1)
    x["signal"] = (
        (x["ema_fast"] > x["ema_slow"]) &
        (x["rsi"].between(52, 74)) &
        (x["vol_ratio"] > 1.15) &
        (x["breakout"])
    )
    return x

def run(df, fast=12, slow=26):
    x = indicators(df, fast, slow).dropna().reset_index(drop=True)
    cash = 1.0
    equity = []
    position = 0
    entry = None
    trades = []

    for i in range(len(x)-1):
        # Signal at close i; execute at open i+1.
        if position == 0 and bool(x.loc[i, "signal"]):
            entry = x.loc[i+1, "open"] * (1 + SLIPPAGE)
            position = 1
            cash *= (1 - FEE)
        elif position == 1 and not bool(x.loc[i, "signal"]):
            exit_px = x.loc[i+1, "open"] * (1 - SLIPPAGE)
            ret = exit_px / entry - 1
            cash *= (1 + ret) * (1 - FEE)
            trades.append(ret)
            position = 0
            entry = None

        mark = cash if position == 0 else cash * (x.loc[i, "close"] / entry)
        equity.append(mark)

    if position:
        exit_px = x.iloc[-1]["close"] * (1 - SLIPPAGE)
        ret = exit_px / entry - 1
        cash *= (1 + ret) * (1 - FEE)
        trades.append(ret)
        equity.append(cash)

    eq = pd.Series(equity).replace([np.inf, -np.inf], np.nan).dropna()
    peak = eq.cummax()
    dd = eq/peak - 1
    daily_like = eq.pct_change().dropna()

    win_rate = float(np.mean(np.array(trades) > 0)) if trades else 0.0
    gross_profit = sum(r for r in trades if r > 0)
    gross_loss = abs(sum(r for r in trades if r < 0))
    profit_factor = gross_profit / gross_loss if gross_loss else np.inf
    sharpe = (daily_like.mean()/daily_like.std()*np.sqrt(24*365)
              if daily_like.std() > 0 else 0)

    return {
        "final_equity": float(cash),
        "return_pct": (cash-1)*100,
        "max_drawdown_pct": float(dd.min()*100) if len(dd) else 0,
        "trades": len(trades),
        "win_rate_pct": win_rate*100,
        "profit_factor": float(profit_factor),
        "sharpe_like": float(sharpe)
    }

def load(symbol):
    p = DATA / f"{symbol}_1h.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run download_data.py first.")
    return pd.read_csv(p, parse_dates=["time"]).sort_values("time")

def walk_forward(df, fast, slow, train_months=12, test_months=3):
    start = df["time"].min()
    train_end = start + pd.DateOffset(months=train_months)
    test_end = train_end + pd.DateOffset(months=test_months)
    rows = []

    while test_end <= df["time"].max():
        train = df[(df.time >= start) & (df.time < train_end)]
        test = df[(df.time >= train_end) & (df.time < test_end)]
        if len(train) > 500 and len(test) > 100:
            # Parameters are supplied by the grid; only test segment is scored here.
            r = run(test, fast, slow)
            r.update({"train_end": train_end, "test_end": test_end,
                      "fast": fast, "slow": slow})
            rows.append(r)
        train_end = test_end
        test_end = train_end + pd.DateOffset(months=test_months)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    symbols = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","DOGEUSDT"]
    grid = list(itertools.product([8,12,16,20], [20,26,35,50]))
    all_results = []

    for symbol in symbols:
        try:
            df = load(symbol)
            # Simple chronological holdout: first 70% train, final 30% untouched test.
            cut = int(len(df)*0.70)
            train, test = df.iloc[:cut], df.iloc[cut:]

            # Select parameters ONLY on the training set.
            candidates = []
            for fast, slow in grid:
                if fast >= slow:
                    continue
                r = run(train, fast, slow)
                candidates.append((r["sharpe_like"], fast, slow, r))
            candidates.sort(reverse=True, key=lambda z: z[0])
            _, best_fast, best_slow, train_result = candidates[0]

            test_result = run(test, best_fast, best_slow)
            row = {
                "symbol": symbol,
                "best_fast": best_fast,
                "best_slow": best_slow,
                "train_return_pct": train_result["return_pct"],
                "test_return_pct": test_result["return_pct"],
                "test_max_drawdown_pct": test_result["max_drawdown_pct"],
                "test_trades": test_result["trades"],
                "test_win_rate_pct": test_result["win_rate_pct"],
                "test_profit_factor": test_result["profit_factor"],
                "test_sharpe_like": test_result["sharpe_like"],
            }
            all_results.append(row)
            print(row)
        except FileNotFoundError as e:
            print(e)

    if all_results:
        out = pd.DataFrame(all_results)
        out.to_csv("backtest_results.csv", index=False)
        print("\nSaved backtest_results.csv")
        print(out.to_string(index=False))
