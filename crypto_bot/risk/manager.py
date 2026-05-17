"""Position sizing, stops, kill switch, portfolio heat."""

from __future__ import annotations

import logging
from typing import Literal

from crypto_bot import config

logger = logging.getLogger(__name__)


def position_size(
    signal: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    capital: float,
) -> float:
    """
    Half-Kelly position size in USD.

    f = 0.5 * ((win_rate / avg_loss) - ((1 - win_rate) / avg_win))
    Position = f * capital * abs(signal)
    """
    if avg_win <= 0 or avg_loss <= 0:
        return 0.0

    kelly = (win_rate / avg_loss) - ((1 - win_rate) / avg_win)
    f = 0.5 * kelly
    f = max(0.0, min(f, 0.25))
    size = f * capital * abs(signal)
    logger.debug("position_size signal=%.2f f=%.4f size=%.2f", signal, f, size)
    return size


def trailing_stop(
    entry_price: float,
    atr: float,
    side: Literal["long", "short"],
    current_price: float | None = None,
    current_stop: float | None = None,
) -> float:
    """Stop distance = 2 * ATR. Updates as price moves favorably."""
    distance = 2.0 * atr
    if side == "long":
        stop = entry_price - distance
        if current_price is not None and current_stop is not None:
            new_stop = current_price - distance
            stop = max(stop, current_stop, new_stop)
    else:
        stop = entry_price + distance
        if current_price is not None and current_stop is not None:
            new_stop = current_price + distance
            stop = min(stop, current_stop, new_stop)
    return stop


def check_kill_switch(daily_pnl: float, capital: float) -> bool:
    """Halt trading if daily loss exceeds 5% of capital."""
    threshold = -config.KILL_SWITCH_DAILY_LOSS_PCT * capital
    triggered = daily_pnl < threshold
    if triggered:
        logger.warning("Kill switch triggered: daily_pnl=%.2f threshold=%.2f", daily_pnl, threshold)
    return triggered


def portfolio_heat(open_positions: list[dict], capital: float) -> float:
    """Sum of position risks as fraction of capital."""
    if capital <= 0:
        return 1.0
    total_risk = sum(abs(p.get("risk_usd", p.get("size_usd", 0))) for p in open_positions)
    return total_risk / capital


def can_open_trade(open_positions: list[dict], capital: float, new_risk_usd: float) -> bool:
    """Refuse new trades if portfolio heat would exceed 20%."""
    current_heat = portfolio_heat(open_positions, capital)
    new_heat = (portfolio_heat(open_positions, capital) * capital + new_risk_usd) / capital
    allowed = new_heat <= config.MAX_PORTFOLIO_HEAT
    if not allowed:
        logger.warning("Portfolio heat limit: %.2f%% > %.2f%%", new_heat * 100, config.MAX_PORTFOLIO_HEAT * 100)
    return allowed
