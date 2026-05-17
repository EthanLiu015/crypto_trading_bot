"""CCXT historical OHLCV fetcher."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import ccxt
import pandas as pd

from crypto_bot.exchange import create_data_exchange, create_exchange

logger = logging.getLogger(__name__)

TIMEFRAME_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


def _to_ccxt_symbol(symbol: str, exchange_id: str = "binance") -> str:
    symbol = symbol.upper().replace("/", "")
    if "/" in symbol:
        return symbol
    base = symbol.replace("USDT", "").replace("USD", "")
    if exchange_id in ("kraken", "coinbase"):
        return f"{base}/USD"
    if exchange_id == "binanceus":
        return f"{base}/USDT"
    return f"{base}/USDT"


def fetch_ohlcv(
    symbol: str,
    timeframe: str = "1h",
    lookback_days: int = 365,
    exchange: ccxt.Exchange | None = None,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV from Binance via CCXT (public API in paper mode).

    Returns DataFrame with columns: timestamp, open, high, low, close, volume.
    """
    ex = exchange or create_data_exchange()
    ccxt_symbol = _to_ccxt_symbol(symbol, ex.id)

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    since = int(
        (datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp() * 1000
    )
    ms_per_bar = TIMEFRAME_MS.get(timeframe, 3_600_000)
    limit = 1000
    all_rows: list[list] = []

    logger.info("Fetching %s %s lookback=%dd", ccxt_symbol, timeframe, lookback_days)
    while since < now_ms:
        batch = ex.fetch_ohlcv(ccxt_symbol, timeframe, since=since, limit=limit)
        if not batch:
            break
        prev_since = since
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        since = last_ts + ms_per_bar
        # Many exchanges cap batch size below `limit` (e.g. Kraken ~720). Keep paging until present.
        if last_ts >= now_ms - ms_per_bar:
            break
        if since <= prev_since:
            break

    if not all_rows:
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

    df = pd.DataFrame(
        all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if len(df) >= 2:
        span_days = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]) / 86_400_000
        logger.info(
            "Fetched %d bars (%.0f days) for %s via %s",
            len(df),
            span_days,
            ccxt_symbol,
            ex.id,
        )
        if span_days < lookback_days * 0.85:
            logger.warning(
                "Only %.0f days of %s available from %s; backtest chart will be shorter than requested",
                span_days,
                lookback_days,
                ex.id,
            )
    else:
        logger.info("Fetched %d bars for %s", len(df), ccxt_symbol)
    return df
