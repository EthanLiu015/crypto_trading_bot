"""FastAPI backend: REST + WebSocket."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from crypto_bot import config
from crypto_bot.backtest import engine as backtest_engine
from crypto_bot.logging_setup import setup_logging
from crypto_bot.state import bot

logger = logging.getLogger(__name__)
_DASHBOARD = Path(__file__).parent / "dashboard.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("API started (paper=%s)", not config.LIVE_TRADING)
    yield
    if bot.running:
        await bot.stop()


app = FastAPI(title="Crypto Trading Bot", version="1.0.0", lifespan=lifespan)


class TradeRequest(BaseModel):
    symbol: str
    side: str
    size_usd: float = Field(gt=0)
    order_type: str = "market"


_BACKTEST_SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(
        _DASHBOARD,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/status")
async def status() -> dict[str, Any]:
    return bot.status()


@app.get("/backtest")
async def backtest(symbol: str = "BTCUSDT", days: int = 365) -> dict[str, Any]:
    symbol = symbol.upper()
    allowed = _BACKTEST_SYMBOLS | set(config.SYMBOLS)
    if symbol not in allowed:
        raise HTTPException(status_code=400, detail=f"Symbol must be one of: {sorted(allowed)}")
    return await asyncio.to_thread(
        backtest_engine.run_backtest,
        symbol,
        config.TIMEFRAME,
        lookback_days=days,
    )


@app.post("/trade")
async def trade(req: TradeRequest) -> dict[str, Any]:
    side = req.side.lower()
    if side not in ("buy", "sell"):
        return {"error": "side must be buy or sell"}
    order_id = await bot.broker.place_order(
        req.symbol.upper(),
        side,  # type: ignore[arg-type]
        req.size_usd,
        req.order_type,  # type: ignore[arg-type]
    )
    return {
        "order_id": order_id,
        "symbol": req.symbol.upper(),
        "side": side,
        "size_usd": req.size_usd,
    }


@app.post("/bot/start")
async def start_bot() -> dict[str, str]:
    await bot.start()
    return {"status": "started"}


@app.post("/bot/stop")
async def stop_bot() -> dict[str, str]:
    await bot.stop()
    return {"status": "stopped"}


@app.websocket("/ws/feed")
async def ws_feed(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            payload = {
                "prices": bot.latest_prices,
                "signal": bot.current_signal,
                "regime": bot.regime_probs,
                "running": bot.running,
                "positions": bot.open_positions,
            }
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception:
        logger.exception("WebSocket error")
