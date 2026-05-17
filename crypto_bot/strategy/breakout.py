"""Volatility expansion breakout strategy."""

from __future__ import annotations

import pandas as pd


def generate_signal(df: pd.DataFrame) -> float:
    """
    +1 if close breaks above 20-bar high with vol_ratio > 1.4.
    -1 if close breaks below 20-bar low with vol_ratio > 1.4.
    """
    if len(df) < 21:
        return 0.0

    row = df.iloc[-1]
    prev = df.iloc[-2]
    close = row.get("close")
    high_20 = prev.get("high_20")
    low_20 = prev.get("low_20")
    vol_ratio = row.get("vol_ratio")

    if any(pd.isna(x) for x in (close, high_20, low_20, vol_ratio)):
        return 0.0

    if vol_ratio <= 1.4:
        return 0.0

    if close > high_20:
        return 1.0
    if close < low_20:
        return -1.0
    return 0.0
