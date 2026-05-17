"""CoinGecko REST client for price and market data."""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

_SYMBOL_TO_ID: dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "BNB": "binancecoin",
    "XRP": "ripple",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "AVAX": "avalanche-2",
    "DOT": "polkadot",
    "MATIC": "matic-network",
}


def symbol_to_coin_id(symbol: str) -> str:
    """Map trading symbol (e.g. BTCUSDT) to CoinGecko id."""
    base = symbol.upper().replace("USDT", "").replace("USD", "").replace("/", "")
    return _SYMBOL_TO_ID.get(base, base.lower())


async def fetch_market_data(symbol: str) -> dict[str, Any]:
    """
    Fetch current price, 24h volume, and market cap.

    Returns dict with keys: price_usd, volume_24h, market_cap, symbol, coin_id.
    """
    coin_id = symbol_to_coin_id(symbol)
    url = f"{COINGECKO_BASE}/coins/markets"
    params = {
        "vs_currency": "usd",
        "ids": coin_id,
        "order": "market_cap_desc",
        "sparkline": "false",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

    if not data:
        logger.warning("No CoinGecko data for %s (%s)", symbol, coin_id)
        return {
            "symbol": symbol,
            "coin_id": coin_id,
            "price_usd": None,
            "volume_24h": None,
            "market_cap": None,
        }

    row = data[0]
    result = {
        "symbol": symbol,
        "coin_id": coin_id,
        "price_usd": row.get("current_price"),
        "volume_24h": row.get("total_volume"),
        "market_cap": row.get("market_cap"),
    }
    logger.debug("CoinGecko %s: price=%s", symbol, result["price_usd"])
    return result


async def screen_universe(symbols: list[str], min_market_cap: float = 1e9) -> list[str]:
    """Return symbols with market cap above threshold."""
    passed = []
    for sym in symbols:
        try:
            data = await fetch_market_data(sym)
            cap = data.get("market_cap") or 0
            if cap >= min_market_cap:
                passed.append(sym)
        except Exception:
            logger.exception("Screen failed for %s", sym)
    return passed
