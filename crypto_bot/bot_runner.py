"""Live trading loop orchestration."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pandas as pd

from crypto_bot import config
from crypto_bot.data.live_feed import LiveFeed
from crypto_bot.data import historical
from crypto_bot.execution.broker import Broker
from crypto_bot.features import engineer
from crypto_bot.regime.detector import RegimeDetector
from crypto_bot.risk import manager as risk
from crypto_bot.strategy import ensemble

logger = logging.getLogger(__name__)

TIMEFRAME_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


class TradingBot:
    """Manages live feed, regime detection, signals, and execution."""

    def __init__(self) -> None:
        self.running = False
        self.feed = LiveFeed(symbols=config.SYMBOLS, interval=config.TIMEFRAME)
        self.broker = Broker()
        self.detector = RegimeDetector()
        self._detector_fitted = False
        self.regime_probs: dict[str, float] = {
            "trending": 1 / 3,
            "ranging": 1 / 3,
            "breakout": 1 / 3,
        }
        self.current_signal: float = 0.0
        self.open_positions: list[dict] = []
        self.daily_pnl: float = 0.0
        self.capital = config.CAPITAL_USD
        self._task: asyncio.Task | None = None
        self._last_bar_time: dict[str, int] = {}
        self._win_rate = 0.55
        self._avg_win = 1.5
        self._avg_loss = 1.0
        self.latest_prices: dict[str, float] = {}
        self._candle_handler_registered = False

    async def _preload_history(self) -> None:
        for symbol in config.SYMBOLS:
            try:
                df = await asyncio.to_thread(
                    historical.fetch_ohlcv,
                    symbol,
                    config.TIMEFRAME,
                    30,
                )
                if df.empty:
                    continue
                rows = df.to_dict("records")
                self.feed.load_history(symbol, rows)
                self.latest_prices[symbol] = float(df.iloc[-1]["close"])
            except Exception:
                logger.exception("Failed to preload history for %s", symbol)

    async def _ensure_detector(self) -> None:
        if self._detector_fitted:
            return
        for symbol in config.SYMBOLS:
            df = await asyncio.to_thread(
                historical.fetch_ohlcv, symbol, config.TIMEFRAME, 90
            )
            if len(df) > 100:
                featured = engineer.compute_features(df)
                self.detector.fit(featured)
                self._detector_fitted = True
                self.regime_probs = self.detector.latest_regime_dict(featured)
                self.current_signal = ensemble.generate_signal(featured, self.regime_probs)
                logger.info("Regime detector fitted on %s history", symbol)
                return
        logger.warning("Could not fit regime detector; using default probs")

    async def _on_closed_candle(self, symbol: str, candle: dict) -> None:
        ts = candle["timestamp"]
        if self._last_bar_time.get(symbol) == ts:
            return
        self._last_bar_time[symbol] = ts
        self.latest_prices[symbol] = candle["close"]

        candles = self.feed.get_candles(symbol)
        if len(candles) < 50:
            return

        df = pd.DataFrame(candles)
        featured = engineer.compute_features(df)

        if self._detector_fitted:
            self.regime_probs = self.detector.latest_regime_dict(featured)

        self.current_signal = ensemble.generate_signal(featured, self.regime_probs)
        logger.info(
            "%s regime=%s signal=%.3f close=%.2f",
            symbol,
            self.regime_probs,
            self.current_signal,
            candle["close"],
        )

        if not self.running:
            return
        if risk.check_kill_switch(self.daily_pnl, self.capital):
            self.running = False
            logger.warning("Kill switch active — halting trading")
            return

        try:
            self.open_positions = await self.broker.get_positions()
        except Exception:
            logger.exception("Failed to fetch positions")

        size_usd = risk.position_size(
            self.current_signal,
            self._win_rate,
            self._avg_win,
            self._avg_loss,
            self.capital,
        )
        if abs(self.current_signal) < 0.1 or size_usd < 10:
            return

        if not risk.can_open_trade(self.open_positions, self.capital, size_usd):
            return

        if config.LIVE_TRADING:
            side = "buy" if self.current_signal > 0 else "sell"
            await self.broker.place_order(symbol, side, size_usd, "market")
        else:
            action = "buy" if self.current_signal > 0 else "sell"
            logger.info("Paper mode: would %s %s $%.2f", action, symbol, size_usd)

    async def _loop(self) -> None:
        await self._preload_history()
        await self._ensure_detector()

        if not self._candle_handler_registered:
            def _schedule(s: str, c: dict) -> None:
                asyncio.create_task(self._on_closed_candle(s, c))

            self.feed.on_candle(_schedule)
            self._candle_handler_registered = True

        await self.feed.start()
        interval = TIMEFRAME_SECONDS.get(config.TIMEFRAME, 3600)
        while self.running:
            try:
                self.open_positions = await self.broker.get_positions()
            except Exception:
                logger.exception("Failed to refresh positions")
            await asyncio.sleep(min(interval, 60))

    async def start(self) -> None:
        if self.running and self._task and not self._task.done():
            return
        self.running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Trading bot started (live_trading=%s)", config.LIVE_TRADING)

    async def stop(self) -> None:
        self.running = False
        await self.feed.stop()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Trading bot stopped")

    def status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "live_trading": config.LIVE_TRADING,
            "regime_probs": self.regime_probs,
            "current_signal": self.current_signal,
            "open_positions": self.open_positions,
            "daily_pnl": self.daily_pnl,
            "capital": self.capital,
            "symbols": config.SYMBOLS,
            "latest_prices": self.latest_prices,
        }
