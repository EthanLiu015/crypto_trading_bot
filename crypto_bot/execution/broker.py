"""CCXT unified order API with retry logic."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Literal

from crypto_bot import config
from crypto_bot.exchange import create_exchange

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BACKOFF_BASE = 1.0


def _to_ccxt_symbol(symbol: str) -> str:
    symbol = symbol.upper().replace("/", "")
    if symbol.endswith("USDT"):
        return f"{symbol[:-4]}/USDT"
    return symbol


class Broker:
    """CCXT broker wrapper; paper mode simulates orders locally."""

    def __init__(self) -> None:
        self._exchange = None
        self._fees_paid: float = 0.0
        self._paper_positions: list[dict] = []

    @property
    def exchange(self):
        if self._exchange is None:
            self._exchange = create_exchange()
        return self._exchange

    async def _retry(self, fn, *args, **kwargs) -> Any:
        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                return await asyncio.to_thread(fn, *args, **kwargs)
            except Exception as e:
                last_err = e
                wait = BACKOFF_BASE * (2**attempt)
                logger.warning("Exchange call failed (attempt %d): %s", attempt + 1, e)
                await asyncio.sleep(wait)
        raise last_err  # type: ignore[misc]

    async def _paper_order(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        size_usd: float,
    ) -> str:
        ccxt_sym = _to_ccxt_symbol(symbol)
        ticker = await self._retry(self.exchange.fetch_ticker, ccxt_sym)
        price = float(ticker["last"] or ticker["close"])
        order_id = f"paper-{uuid.uuid4().hex[:12]}"
        fee = size_usd * config.FEE_PCT
        self._fees_paid += fee

        sym = symbol.upper().replace("/", "")
        if not sym.endswith("USDT"):
            sym = f"{sym}USDT"

        existing = next((p for p in self._paper_positions if p["symbol"] == sym), None)
        if side == "buy":
            if existing:
                existing["size_usd"] += size_usd
            else:
                self._paper_positions.append(
                    {
                        "symbol": sym,
                        "side": "long",
                        "size_usd": size_usd,
                        "entry_price": price,
                        "current_price": price,
                        "unrealized_pnl": 0.0,
                        "risk_usd": size_usd * 0.1,
                    }
                )
        elif existing:
            existing["size_usd"] = max(0.0, existing["size_usd"] - size_usd)
            if existing["size_usd"] < 1:
                self._paper_positions.remove(existing)

        logger.info(
            "Paper order %s %s $%.2f @ %.2f id=%s fee=%.4f",
            side,
            sym,
            size_usd,
            price,
            order_id,
            fee,
        )
        return order_id

    async def place_order(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        size_usd: float,
        order_type: Literal["market", "limit"] = "market",
        price: float | None = None,
    ) -> str:
        """Place market or limit order. Returns order ID."""
        if not config.LIVE_TRADING:
            return await self._paper_order(symbol, side, size_usd)

        ccxt_sym = _to_ccxt_symbol(symbol)
        ticker = await self._retry(self.exchange.fetch_ticker, ccxt_sym)
        last_price = ticker["last"] or ticker["close"]
        amount = size_usd / last_price

        logger.info(
            "Placing %s %s %s size_usd=%.2f amount=%.6f",
            order_type,
            side,
            ccxt_sym,
            size_usd,
            amount,
        )

        if order_type == "market":
            order = await self._retry(
                self.exchange.create_market_order, ccxt_sym, side, amount
            )
        else:
            if price is None:
                price = last_price
            order = await self._retry(
                self.exchange.create_limit_order, ccxt_sym, side, amount, price
            )

        fee = order.get("fee", {}) or {}
        fee_cost = fee.get("cost", 0) or 0
        self._fees_paid += float(fee_cost)
        order_id = str(order.get("id", order.get("orderId", "")))
        logger.info("Order filled id=%s fee=%.4f", order_id, fee_cost)
        return order_id

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        if not config.LIVE_TRADING:
            logger.info("Paper cancel %s on %s", order_id, symbol)
            return True
        ccxt_sym = _to_ccxt_symbol(symbol)
        logger.info("Cancelling order %s on %s", order_id, ccxt_sym)
        await self._retry(self.exchange.cancel_order, order_id, ccxt_sym)
        return True

    async def get_positions(self) -> list[dict]:
        """Return open positions with unrealized PnL."""
        if not config.LIVE_TRADING:
            for pos in self._paper_positions:
                try:
                    ticker = await self._retry(
                        self.exchange.fetch_ticker, _to_ccxt_symbol(pos["symbol"])
                    )
                    price = float(ticker["last"])
                    entry = float(pos["entry_price"])
                    pos["current_price"] = price
                    pos["unrealized_pnl"] = (price - entry) / entry * pos["size_usd"]
                except Exception:
                    logger.debug("Could not mark %s", pos["symbol"])
            return list(self._paper_positions)

        balance = await self._retry(self.exchange.fetch_balance)
        positions = []
        totals = balance.get("total") or {}
        for currency, amounts in totals.items():
            total = float(amounts or 0)
            if total <= 0 or currency in ("USDT", "USD", "BUSD"):
                continue
            sym = f"{currency}USDT"
            try:
                ticker = await self._retry(self.exchange.fetch_ticker, f"{currency}/USDT")
                price = float(ticker["last"])
                value_usd = total * price
                positions.append(
                    {
                        "symbol": sym,
                        "side": "long",
                        "size": total,
                        "size_usd": value_usd,
                        "entry_price": price,
                        "current_price": price,
                        "unrealized_pnl": 0.0,
                        "risk_usd": value_usd * 0.1,
                    }
                )
            except Exception:
                logger.debug("Skipping balance for %s", currency)
        return positions

    @property
    def total_fees(self) -> float:
        return self._fees_paid
