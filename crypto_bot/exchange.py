"""CCXT exchange factory with fallbacks for restricted regions."""

from __future__ import annotations

import logging

import ccxt

from crypto_bot import config

logger = logging.getLogger(__name__)

_PLACEHOLDER_KEY_MARKERS = ("your_", "placeholder", "changeme", "xxx")


def _exchange_credentials() -> dict:
    """Only attach API keys when they look real (invalid keys break public endpoints)."""
    key = (config.BINANCE_API_KEY or "").strip()
    secret = (config.BINANCE_SECRET or "").strip()
    if not key or not secret:
        return {}
    lowered = key.lower()
    if any(marker in lowered for marker in _PLACEHOLDER_KEY_MARKERS):
        return {}
    return {"apiKey": key, "secret": secret}


FALLBACK_EXCHANGES = ("binance", "binanceus", "kraken", "coinbase")
# Exchanges that paginate OHLCV well for long backtests (Kraken caps ~720 bars).
HISTORICAL_EXCHANGES = ("binanceus", "coinbase", "binance", "kraken")


def create_exchange(name: str | None = None) -> ccxt.Exchange:
    """Create exchange; uses DATA_EXCHANGE env or ordered fallbacks."""
    preferred = (name or config.DATA_EXCHANGE).lower()
    order = [preferred] + [e for e in FALLBACK_EXCHANGES if e != preferred]

    last_err: Exception | None = None
    for ex_id in order:
        if not hasattr(ccxt, ex_id):
            continue
        try:
            klass = getattr(ccxt, ex_id)
            exchange = klass({"enableRateLimit": True, **_exchange_credentials()})
            if ex_id == "binance" and config.BINANCE_TESTNET:
                exchange.set_sandbox_mode(True)
            exchange.load_markets()
            logger.info("Using exchange: %s", ex_id)
            return exchange
        except Exception as e:
            last_err = e
            logger.warning("Exchange %s unavailable: %s", ex_id, e)

    raise RuntimeError(f"No exchange available. Last error: {last_err}")


def create_binance() -> ccxt.Exchange:
    """Backward-compatible helper."""
    return create_exchange("binance")


def create_data_exchange() -> ccxt.Exchange:
    """Exchange for historical OHLCV; prefers sources with deep paginated history."""
    last_err: Exception | None = None
    for ex_id in HISTORICAL_EXCHANGES:
        if not hasattr(ccxt, ex_id):
            continue
        try:
            klass = getattr(ccxt, ex_id)
            exchange = klass({"enableRateLimit": True, **_exchange_credentials()})
            if ex_id == "binance" and config.BINANCE_TESTNET:
                exchange.set_sandbox_mode(True)
            exchange.load_markets()
            logger.info("Using data exchange: %s", ex_id)
            return exchange
        except Exception as e:
            last_err = e
            logger.warning("Data exchange %s unavailable: %s", ex_id, e)

    raise RuntimeError(f"No data exchange available. Last error: {last_err}")
