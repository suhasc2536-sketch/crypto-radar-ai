Crypto Radar AI V100 — Complete Low-Budget Build

This build combines the V4 foundation into one research dashboard with a much broader public-data intelligence stack.


Included


CoinGecko market universe

Coinbase 15m/1h/4h/1D candles

Binance / Bybit / OKX spot cross-checks and candles

Coinbase order book + recent trade flow

Binance + Bybit derivatives

DEX Screener liquidity context

DefiLlama protocol context

RSS news + optional CryptoPanic adapter

Fear & Greed index

Tokenomics / FDV dilution context

GoPlus security adapter when a contract address is supplied

Historical analog engine

Walk-forward ML experiment

Simple walk-forward research backtest

Multi-timeframe alignment

BTC/ETH market regime

Cross-exchange price agreement

Knowledge graph visualization

Decision engine: STRONG LONG / LONG / WATCH / NO TRADE / SHORT / STRONG SHORT

Entry reference, invalidation and target levels

Paper prediction ledger

SQLite observation store

Coverage/diagnostic panel


Important limitations

This is not a guaranteed pump predictor. Probabilities are experimental until calibrated across a large, clean, out-of-sample dataset. Public APIs can rate-limit, change schema, or fail. Some professional on-chain, smart-money, unlock and social data requires paid/API-key providers.


Deploy


Upload app.py, requirements.txt, .gitignore, and secrets.toml.example to your GitHub repo.

Streamlit Community Cloud -> Create app -> select the repo, branch main, and app.py.

If using keys, put them into Streamlit Secrets, not GitHub.

Deploy.


No local Python is required for deployment.


Budget strategy

Keep the INR 5,000 untouched until the free stack is measured. Only pay for a data source after we identify a measurable bottleneck.

