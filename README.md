# ⚡ Crypto Radar AI — starter project

This is a research/paper-trading starter, not an automated money-making system.

## 1. Put the files in one folder

```text
crypto_radar/
  app.py
  backtest.py
  download_data.py
  news.py
  requirements.txt
  data/
```

## 2. Install Python

Use Python 3.11 or newer.

## 3. Open a terminal in the `crypto_radar` folder

Windows:
```bash
cd path\to\crypto_radar
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Mac/Linux:
```bash
cd path/to/crypto_radar
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. Download historical data

```bash
python download_data.py
```

This downloads hourly candles for BTC, ETH, SOL, BNB, XRP and DOGE from Binance and puts CSV files in `data/`.

## 5. Run the backtests

```bash
python backtest.py
```

It performs:
- a chronological 70/30 train/test split
- parameter search on the training period
- untouched out-of-sample testing
- fees and slippage
- maximum drawdown
- win rate
- profit factor
- Sharpe-like statistic

Do NOT optimize parameters using the test period.

## 6. Start the website

```bash
streamlit run app.py
```

Your browser will open a local address such as:
http://localhost:8501

## 7. Put it online

Create a GitHub repository and upload these files. Then use Streamlit Community Cloud:
https://share.streamlit.io/

Choose:
- repository
- branch: main
- main file: app.py

The resulting site will be on a `streamlit.app` subdomain.

## 8. Next development stage

Add:
- CoinGecko news
- historical timestamped news dataset
- AI sentiment/catalyst scoring
- whale/on-chain data
- DEX liquidity
- token-unlock calendar
- rug/scam checks
- paper trading ledger
- alert system
- walk-forward model selection
- Monte Carlo trade-sequence tests

## Critical backtesting rule

A prediction is only useful if it is reproducible without future information.

For example:

BAD:
- calculate today's final score
- look at today's closing price
- call that a successful prediction

GOOD:
- calculate score at 10:00
- enter at 10:01/next bar
- freeze the information set
- evaluate only what happened afterward
