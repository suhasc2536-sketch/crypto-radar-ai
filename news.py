"""
Optional news module.

Recommended production approach:
1. Store every article with UTC timestamp, source, title, URL, and related token.
2. Score each article using a sentiment/catalyst model.
3. When backtesting, ONLY attach news published before the trade decision.
4. Never use today's article to explain yesterday's return.

CoinGecko also provides a crypto news endpoint on supported API plans.
"""

import requests
import pandas as pd

def get_coingecko_news(api_key=None, page=1):
    url = "https://api.coingecko.com/api/v3/news"
    headers = {}
    if api_key:
        headers["x-cg-demo-api-key"] = api_key
    r = requests.get(url, headers=headers, params={"page": page}, timeout=15)
    r.raise_for_status()
    return r.json()

def simple_catalyst_score(title):
    t = title.lower()
    positive = ["listing", "partnership", "launch", "approval", "integration",
                "upgrade", "adoption", "etf", "mainnet", "funding"]
    negative = ["hack", "exploit", "lawsuit", "ban", "delist", "unlock",
                "liquidation", "attack", "scam"]
    p = sum(word in t for word in positive)
    n = sum(word in t for word in negative)
    return max(-1, min(1, (p-n)/3))
