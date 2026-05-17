"""EMA crossover trend strategy."""

from __future__ import annotations

import pandas as pd


def generate_signal(df: pd.DataFrame) -> float:
    """
    +1 if EMA9 > EMA21 and ADX > 25, -1 if EMA9 < EMA21 and ADX > 25, else 0.
    Scaled by ADX/50.
    """
    if len(df) < 2:
        return 0.0

    row = df.iloc[-1]
    ema9 = row.get("ema_9")
    ema21 = row.get("ema_21")
    adx = row.get("adx")

    if pd.isna(ema9) or pd.isna(ema21) or pd.isna(adx):
        return 0.0

    if adx <= 25:
        return 0.0

    direction = 1.0 if ema9 > ema21 else -1.0
    scale = min(float(adx) / 50.0, 1.0)
    return direction * scale
