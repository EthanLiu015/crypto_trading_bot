"""Binance WebSocket OHLCV stream."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from typing import Callable

import websockets

from crypto_bot import config

logger = logging.getLogger(__name__)

BINANCE_WS_BASE = "wss://stream.binance.com:9443/ws"


def _stream_name(symbol: str, interval: str = "1m") -> str:
    return f"{symbol.lower()}@kline_{interval}"


def _build_combined_url(symbols: list[str], interval: str = "1m") -> str:
    streams = "/".join(_stream_name(s, interval) for s in symbols)
    return f"wss://stream.binance.com:9443/stream?streams={streams}"


class LiveFeed:
    """Streams real-time OHLCV candles; keeps last N candles per symbol."""

    def __init__(
        self,
        symbols: list[str] | None = None,
        interval: str = "1m",
        buffer_size: int = config.CANDLE_BUFFER_SIZE,
    ) -> None:
        self.symbols = symbols or config.SYMBOLS
        self.interval = interval
        self.buffer_size = buffer_size
        self.candles: dict[str, deque[dict]] = {
            s: deque(maxlen=buffer_size) for s in self.symbols
        }
        self._running = False
        self._task: asyncio.Task | None = None
        self._callbacks: list[Callable[[str, dict], None]] = []

    def on_candle(self, callback: Callable[[str, dict], None]) -> None:
        self._callbacks.append(callback)

    def get_candles(self, symbol: str) -> list[dict]:
        return list(self.candles.get(symbol, deque()))

    def load_history(self, symbol: str, rows: list[dict]) -> None:
        """Seed in-memory buffer from historical OHLCV rows."""
        if symbol not in self.candles:
            self.candles[symbol] = deque(maxlen=self.buffer_size)
        buf = self.candles[symbol]
        buf.clear()
        for row in rows[-self.buffer_size :]:
            buf.append(
                {
                    "timestamp": int(row["timestamp"]),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                    "closed": True,
                }
            )
        if buf:
            last = buf[-1]
            logger.info("Preloaded %d candles for %s (last close=%.2f)", len(buf), symbol, last["close"])

    def _emit(self, symbol: str, candle: dict) -> None:
        for cb in self._callbacks:
            try:
                cb(symbol, candle)
            except Exception:
                logger.exception("Callback error for %s", symbol)

    def _handle_kline(self, payload: dict) -> None:
        data = payload.get("data", payload)
        k = data.get("k", data)
        symbol = k.get("s", "").upper()
        if not symbol or symbol not in self.candles:
            return

        candle = {
            "timestamp": int(k["t"]),
            "open": float(k["o"]),
            "high": float(k["h"]),
            "low": float(k["l"]),
            "close": float(k["c"]),
            "volume": float(k["v"]),
            "closed": bool(k["x"]),
        }

        if candle["closed"]:
            buf = self.candles[symbol]
            if buf and buf[-1]["timestamp"] == candle["timestamp"]:
                buf[-1] = candle
            else:
                buf.append(candle)
            logger.debug("Closed candle %s @ %s close=%.2f", symbol, candle["timestamp"], candle["close"])
            self._emit(symbol, candle)
        elif self.candles[symbol]:
            self.candles[symbol][-1] = candle

    async def _run(self) -> None:
        url = _build_combined_url(self.symbols, self.interval)
        logger.info("Connecting to Binance WS: %s", url)
        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    async for raw in ws:
                        if not self._running:
                            break
                        msg = json.loads(raw)
                        self._handle_kline(msg)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("WebSocket error; reconnecting in 5s")
                await asyncio.sleep(5)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info("Live feed started for %s", self.symbols)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Live feed stopped")
