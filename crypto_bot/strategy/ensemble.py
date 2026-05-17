"""Regime-weighted strategy ensemble."""

from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_bot.strategy import breakout, mean_reversion, trend


def generate_signal(
    df: pd.DataFrame,
    regime_probs: dict[str, float] | None = None,
) -> float:
    """
    Blend strategy signals weighted by regime probabilities.
    regime_probs keys: trending, ranging, breakout.
    """
    if regime_probs is None:
        regime_probs = {"trending": 1 / 3, "ranging": 1 / 3, "breakout": 1 / 3}

    t_sig = trend.generate_signal(df)
    mr_sig = mean_reversion.generate_signal(df)
    bo_sig = breakout.generate_signal(df)

    final = (
        regime_probs.get("trending", 0) * t_sig
        + regime_probs.get("ranging", 0) * mr_sig
        + regime_probs.get("breakout", 0) * bo_sig
    )
    return float(np.clip(final, -1.0, 1.0))
