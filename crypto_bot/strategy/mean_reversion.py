"""Bollinger Band mean reversion strategy."""

from __future__ import annotations

import pandas as pd


def generate_signal(df: pd.DataFrame) -> float:
    """
    +1 if close < lower band and RSI < 35.
    -1 if close > upper band and RSI > 65.
    """
    if len(df) < 2:
        return 0.0

    row = df.iloc[-1]
    close = row.get("close")
    lower = row.get("bb_lower")
    upper = row.get("bb_upper")
    rsi = row.get("rsi")

    if any(pd.isna(x) for x in (close, lower, upper, rsi)):
        return 0.0

    if close < lower and rsi < 35:
        return 1.0
    if close > upper and rsi > 65:
        return -1.0
    return 0.0
